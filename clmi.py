import numpy as np
import queue
import cv2
import warnings

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
_C = 299792458  # speed of light, SI units
_dilation_erosion_steps = 1


def set_cosmology(new_cosmology):
    """Set astropy cosmology model for calculations in place of the default flat Lambda CDM with H0=70, OmegaM = 0.3."""
    global _cosmo
    _cosmo = new_cosmology


def distance_scale_factor(z_lens, z_source):
    """Ratio of angular diameter distances from lens to source and from observer to source.

    The ratio is used to scale deflection and magnification maps. 
    Uses module level defined cosmology, which can be redefined using set_cosmology.
    """
    Dds = _cosmo.angular_diameter_distance_z1z2(z_lens, z_source)
    Ds = _cosmo.angular_diameter_distance_z1z2(0, z_source)
    return Dds / Ds


def distance_parameter(z_lens, z_source):
    """Used in time delay equation, see https://arxiv.org/pdf/astro-ph/9606001.pdf Eq. 63.

    Uses module level defined cosmology, which can be redefined using set_cosmology."""
    Dds = _cosmo.angular_diameter_distance_z1z2(z_lens, z_source)
    Ds = _cosmo.angular_diameter_distance_z1z2(0, z_source)
    Dd = _cosmo.angular_diameter_distance_z1z2(0, z_lens)
    return (Dd * Ds / Dds).value


def magnification(kappa, gamma, scaling_factor):  # Works for both individual numbers and np arrays.
    """Basic lensing formula for magnification based on convergence (kappa) and shear (gamma) for given Dds/Ds."""
    return ((1 - kappa * scaling_factor) ** 2 - (gamma * scaling_factor) ** 2) ** -1


def time_delay(z_lens, dist_param, alpha, psi):             # TODO: fix this, what are the units???
    """Calculates the time delay.

    Results are in [unknown units], as long as angles are in [unknown units].
    dist_param is D_lens * D_source / D_(lens to source), NOT Dds_Ds
    alpha: Deflection angle, equal to theta - beta
    psi: Gravitational lensing potential
    Citation: https://arxiv.org/pdf/astro-ph/9606001.pdf equation 63
    """
    return (1 + z_lens) / _C * dist_param * (.5 * (alpha ** 2) - psi)


_direction_set_4 = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # Used in flood_fill to disable diagonals
_direction_set_8 = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),   # Used in flood_fill to enable diagonals
                             (-1, -1), (-1, 1), (1, -1), (1, 1)])


def _flood_fill(array, start_pixel, targeted_val, new_val, directions=4):
    """Flood fill tool on a numpy array using a queue."""

    def handle_pixel(pixel_coords):
        if (new_arr[pixel_coords]) == targeted_val:
            new_arr[pixel_coords] = new_val
            for i in range(directions):
                new_coords = pixel_coords + dir_set[i]
                if ((0, 0) <= new_coords).all() and (
                        new_coords < new_arr.shape).all():  # if new pixel coords are in array:
                    new_coords = tuple(new_coords)
                    if new_arr[new_coords] == targeted_val:
                        pixel_queue.put(new_coords)

    if array[start_pixel] != targeted_val:  # Maybe just remove this and make it take the starting val?
        raise ValueError("Start pixel has value "  # Or assume targeted_val = array[start] by default?
                         + str(array[start_pixel])
                         + ", but targeted value is "
                         + str(new_val)
                         + "!")

    if directions == 4:  # only side-adjacent pixels
        dir_set = _direction_set_4
    elif directions == 8:  # include diagonal pixels
        dir_set = _direction_set_8
    else:
        raise ValueError("'directions' needs to be 4 or 8.")
        # Can be replaced with 'dir_set = directions' to allow custom direction sets

    new_arr = np.copy(array)
    pixel_queue = queue.SimpleQueue()
    pixel_queue.put(start_pixel)
    while not pixel_queue.empty():
        handle_pixel(pixel_queue.get())

    return new_arr


def _check_and_load_file(initial_input):
    if initial_input is None:
        return None, None
    if isinstance(initial_input, str):
        initial_input = fits.open(initial_input)
    return initial_input, initial_input[0].data


class ClusterModel:
    """Handle individual cluster models and calculations based on them."""
    def __init__(self, cluster_z, working_z=9, kappa_file=None, gamma_file=None,
                 psi_file=None, x_deflect_file=None, y_deflect_file=None):
        self.kappa_file, self.kappa_map = _check_and_load_file(kappa_file)
        self.gamma_file, self.gamma_map = _check_and_load_file(gamma_file)
        self.psi_file, self.psi_map = _check_and_load_file(psi_file)
        self.x_deflect_file, self.x_deflect_map = _check_and_load_file(x_deflect_file)
        self.y_deflect_file, self.y_deflect_map = _check_and_load_file(y_deflect_file)

        self.cluster_z = cluster_z
        self.working_z = working_z
        self.distance_ratio = distance_scale_factor(self.cluster_z, self.working_z)
        self.distance_param = distance_parameter(self.cluster_z, self.working_z)
        self.magnification_map = None
        self.critical_area_map = None
        self.caustic_area_map = None        # This map has the same angular scale as lens plane maps.
        # TODO: add WCS loaded from one of the files. Can it always be loaded? Can WCS from one fits be used for others?
        # TODO: add checking deflection map units (arcsec, pixel etc) and conversion handling (can you use astropy?) 

    def set_working_z(self, new_z):
        self.working_z = new_z
        self.distance_ratio = distance_scale_factor(self.cluster_z, new_z)
        self.distance_param = distance_parameter(self.cluster_z, new_z)
        self.magnification_map = None       # Reset all z-specific data. It will be generated as needed.
        self.critical_area_map = None
        self.caustic_area_map = None

    def get_magnification_map(self):
        if self.magnification_map is None:
            self._generate_magnification_map()
        return self.magnification_map

    def _generate_magnification_map(self):
        self.magnification_map = magnification(self.kappa_map, self.gamma_map, self.distance_ratio)

    def get_critical_area_map(self):
        if self.critical_area_map is None:
            self._generate_critical_area_map()
        return self.critical_area_map

    def _generate_critical_area_map(self):
        magnif_map = self.get_magnification_map()
        magnif_sign = np.ones(magnif_map.shape, dtype=int)      # Get map of magnification signs, get rid of outer area.
        magnif_sign[np.where(magnif_map < 0.)] = -1
        magnif_sign = _flood_fill(magnif_sign, (0, 0), 1, 0)       # Note: we're assuming (0, 0) is outside
        self.critical_area_map = np.abs(magnif_sign)                # the critical curve. If it isn't, this will crash.

    def get_caustic_area_map(self):
        if self.caustic_area_map is None:
            self._generate_caustic_area_map()
        return self.caustic_area_map

    def _generate_caustic_area_map(self):       # Here we start getting prone to numerical issues.
        critical_map = self.get_critical_area_map()     # TODO: double check that this works on all data.
        hit_map = np.zeros(critical_map.shape)
        mapping_warning_issued = False
        for y in range(critical_map.shape[0]):
            for x in range(critical_map.shape[1]):
                if critical_map[y, x] > 0:
                    y_hit = int(y + self.y_deflect_map[y, x] * self.distance_ratio)     # This assumes deflect maps
                    x_hit = int(x + self.x_deflect_map[y, x] * self.distance_ratio)     # are given in pixels.
                    if 0 <= y_hit < hit_map.shape[0] and 0 <= x_hit < hit_map.shape[1]:
                        hit_map[y_hit, x_hit] = 1
                    elif not mapping_warning_issued:
                        warnings.warn("Some of the area inside the critical curve maps \
                                      outside the generated source plane map.")
                        mapping_warning_issued = True
        # Now we have a raw map from the critical to caustic curve. We need to make sure flood fill doesn't fill gaps
        # between individual hits.
        hit_map = cv2.dilate(hit_map, None, iterations=_dilation_erosion_steps)
        hit_map = cv2.erode(hit_map, None, iterations=_dilation_erosion_steps)
        # Now hopefully all of the gaps between hits were filled and we can flood fill:
        hit_map = _flood_fill(hit_map, (0, 0), 0, -1)       # -1 is the outer area outside the caustic curve.
        self.caustic_area_map = np.zeros(hit_map.shape)     # 0 and 1 are inside.
        self.caustic_area_map[np.where(hit_map >= 0)] = 1   # The final map has 1 inside, 0 outside the caustic.
