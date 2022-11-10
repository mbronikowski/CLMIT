import cv2
import numpy as np
# import queue
import warnings

from astropy import constants, units, wcs
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.ndimage.measurements import label

_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
_dilation_erosion_steps = 1                     # TODO: fine tune this


def set_cosmology(new_cosmology):
    """Set astropy cosmology model for calculations in place of the default flat Lambda CDM with H0=70, OmegaM = 0.3.

    Note that this needs to be performed before initiating a ClusterModel instance, otherwise a new working z
    needs to be set in the instance to reset cosmology-dependent calculations.
    """

    global _cosmology
    _cosmology = new_cosmology


def lensing_scaling_factor(z_lens, z_source):
    """Ratio of angular diameter distances from lens to source and from observer to source.

    The ratio is used to scale deflection and magnification maps. 
    Uses module level defined cosmology, which can be redefined using set_cosmology.
    """

    dist_deflector_source = _cosmology.angular_diameter_distance_z1z2(z_lens, z_source)   # commonly referred to as Dds.
    dist_source = _cosmology.angular_diameter_distance_z1z2(0, z_source)                  # Commonly referred to as Ds.
    return dist_deflector_source / dist_source


def distance_parameter(z_lens, z_source):       # TODO: rename?
    """Used in time delay equation, see https://arxiv.org/pdf/astro-ph/9606001.pdf Eq. 63.

    Uses module level defined cosmology, which can be redefined using set_cosmology.
    """

    dist_deflector_source = _cosmology.angular_diameter_distance_z1z2(z_lens, z_source)
    dist_source = _cosmology.angular_diameter_distance_z1z2(0, z_source)
    dist_deflector = _cosmology.angular_diameter_distance_z1z2(0, z_lens)
    return dist_deflector * dist_source / dist_deflector_source


def magnification(kappa, gamma, scaling_factor):  # Works for both individual numbers and np arrays.
    """Basic lensing formula for magnification based on convergence (kappa) and shear (gamma) for given Dds/Ds."""
    return ((1 - kappa * scaling_factor) ** 2 - (gamma * scaling_factor) ** 2) ** -1


def time_delay(z_lens, dist_param, alpha, psi):             # TODO: fix this, what are the units???
    """Calculates the time delay caused by the gravitational potential: Shapiro time delay plus geometric time delay.

    Results are in [unknown units], as long as angles are in [unknown units].
    dist_param is D_lens * D_source / D_(lens to source), NOT Dds_Ds
    alpha: Deflection angle, equal to theta - beta
    psi: Gravitational lensing potential
    Citation: https://arxiv.org/pdf/astro-ph/9606001.pdf equation 63
    """

    return (1 + z_lens) / constants.c * dist_param * (.5 * (alpha ** 2) - psi)


def _flood_fill(array, start_pixel, targeted_val, new_val):
    """Flood fill tool on a numpy array using a queue."""
    if array[start_pixel] != targeted_val:          # Sanity check
        raise ValueError("The flood fill tool has failed. Array had value of " +
                         str(array[start_pixel]) + ", expected " + str(targeted_val) + ".")
    new_arr = np.copy(array)
    label_map = label(new_arr == new_arr[start_pixel])[0]
    new_arr[label_map == label_map[start_pixel]] = new_val
    return new_arr


# noinspection SpellCheckingInspection
class ClusterModel:
    """Handles individual cluster models and calculations based on them.

    Keyword arguments:
    cluster_z -- redshift of the cluster (default 1.0)
    source_z -- redshift of the source plane for which calculations are made (default 9.0)
    kappa_file -- cluster lensing convergence map, either a FITS file or path to a FITS file (default None)
    gamma_file -- cluster lensing total shear map, either a FITS file or path to a FITS file (default None)
    psi_file -- cluster lensing potential map, either a FITS file or path to a FITS file (default None)
    x_pixel_deflect_file -- cluster lensing x axis deflection map measured in pixels,
        either a FITS file or path to a FITS file (default None)
    y_pixel_deflect_file -- cluster lensing y axis deflection map measured in pixels,
        either a FITS file or path to a FITS file (default None)
    x_as_deflect_file -- cluster lensing x axis deflection map measured in arcseconds,
        either a FITS file or path to a FITS file (default None)
    y_as_deflect_file -- cluster lensing y axis deflection map measured in arcseconds,
        either a FITS file or path to a FITS file (default None)
    map_wcs -- astropy World Coordinate System object to be used by the class.
        If not provided, the object will attempt to load one from provided maps (default None)
    """
    def __init__(self, cluster_z=1., source_z=9., kappa_file=None, gamma_file=None,
                 psi_file=None, x_pixel_deflect_file=None, y_pixel_deflect_file=None,
                 x_as_deflect_file=None, y_as_deflect_file=None, map_wcs=None):
        self.wcs = map_wcs
        self.kappa_map = self._check_and_load_data(kappa_file)
        self.gamma_map = self._check_and_load_data(gamma_file)
        self.psi_map = self._check_and_load_data(psi_file)
        self.x_pixel_deflect_map = self._check_and_load_data(x_pixel_deflect_file)
        self.y_pixel_deflect_map = self._check_and_load_data(y_pixel_deflect_file)
        self.x_as_deflect_map = self._check_and_load_data(x_as_deflect_file)
        self.y_as_deflect_map = self._check_and_load_data(y_as_deflect_file)

        self.cluster_z = cluster_z
        self.source_z = source_z
        self.cluster_angular_diameter_distance = _cosmology.angular_diameter_distance_z1z2(0, self.cluster_z)
        self.source_angular_diameter_distance = _cosmology.angular_diameter_distance_z1z2(0, self.source_z)
        self.distance_ratio = lensing_scaling_factor(self.cluster_z, self.source_z)
        self.distance_param = distance_parameter(self.cluster_z, self.source_z)

        self._magnification_map = None       # Private. Access through get_[type]_map.
        self._critical_area_map = None
        self._caustic_area_map = None        # This map has the same angular scale as lens plane maps.
        # TODO: add checking deflection map units (arcsec, pixel etc) and conversion handling (can you use astropy?)
        if self.wcs is None:
            warnings.warn("The cluster model was unable to load a working World Coordinate System from provided files.")

    def _check_and_load_data(self, initial_input):
        def add_wcs(fits_header):
            # Load WCS from FITS and set object WCS if the loaded one is valid:
            new_wcs = wcs.WCS(fits_header)
            if new_wcs.has_celestial:
                self.wcs = new_wcs

        if initial_input is None:       # If no file provided, keep None.
            return None
        elif isinstance(initial_input, str):          # If filename str is provided, open file, load data, close file.
            with fits.open(initial_input) as fits_file:
                data_table = fits_file[0].data.copy()
                if self.wcs is None:
                    add_wcs(fits_file[0].header)
            return data_table
        # Otherwise, assume a fits file was provided. Copy data from it in case it's closed while the object is in use.
        add_wcs(initial_input[0].header)
        return initial_input[0].data.copy()   # TODO: [0], ['SCI'] or something else? How to guarantee loading?
        # TODO: functionality for passing np tables with data?

    def set_source_z(self, new_z):
        """Set the redshift for which parameters will be calculated and reset previously computed z-dependent values."""
        self.source_z = new_z
        self.source_angular_diameter_distance = _cosmology.angular_diameter_distance_z1z2(0, self.source_z)
        self.distance_ratio = lensing_scaling_factor(self.cluster_z, new_z)
        self.distance_param = distance_parameter(self.cluster_z, new_z)
        self._magnification_map = None       # Reset all z-specific data. It will be generated as needed.
        self._critical_area_map = None
        self._caustic_area_map = None

    def _generate_arcsec_deflect_maps(self):
        pixel_scale = self.wcs.proj_plane_pixel_scales()[0].to(units.arcsec).value
        self.x_as_deflect_map = self.x_pixel_deflect_map.copy() * pixel_scale
        self.y_as_deflect_map = self.y_pixel_deflect_map.copy() * pixel_scale

    def _generate_pixel_deflect_maps(self):
        pixel_scale = self.wcs.proj_plane_pixel_scales()[0].to(units.arcsec).value
        self.x_pixel_deflect_map = self.x_as_deflect_map.copy() / pixel_scale
        self.y_pixel_deflect_map = self.y_as_deflect_map.copy() / pixel_scale

    def get_magnification_map(self):
        if self._magnification_map is None:
            self._generate_magnification_map()
        return self._magnification_map

    def _generate_magnification_map(self):
        self._magnification_map = magnification(self.kappa_map, self.gamma_map, self.distance_ratio)

    def get_critical_area_map(self):
        """Public method to access the map of the cluster's region inside the critical curve, with 1s inside it.."""
        if self._critical_area_map is None:
            self._generate_critical_area_map()
        return self._critical_area_map

    def _generate_critical_area_map(self):
        magnif_map = self.get_magnification_map()
        magnif_sign = np.ones(magnif_map.shape, dtype=int)      # Get map of magnification signs, get rid of outer area.
        magnif_sign[np.where(magnif_map < 0.)] = -1
        magnif_sign = _flood_fill(magnif_sign, (0, 0), 1, 0)         # Note: we're assuming (0, 0) is outside
        self._critical_area_map = np.abs(magnif_sign)                # the critical curve. If it isn't, this will crash.

    def get_caustic_area_map(self):
        """Public method to access the map of the region inside the caustic curve, denoted by 1s."""
        if self._caustic_area_map is None:
            self._caustic_area_map = self.map_to_source_plane(self.get_critical_area_map())
        return self._caustic_area_map

    def map_to_source_plane(self, lens_plane_map):
        """Map arbitrary area from the lens plane to the source plane.

        The input needs to be a numpy array of integers with 1s for pixels which are to be mapped, 0s otherwise.
        The function will fill in any holes in the mapped source plane area; the function does not handle areas with
        holes in them in the source plane. This aids with numerical issues inside the caustic curve.
        """
        if self.x_pixel_deflect_map is None or self.y_pixel_deflect_map is None:
            self._generate_pixel_deflect_maps()
        if lens_plane_map.shape != self.x_pixel_deflect_map.shape:
            raise ValueError("The array to be mapped back to the source plane needs to have the same shape \
                as deflection maps.")       # TODO: extend the map, make it wider
        hit_map = np.zeros(lens_plane_map.shape)
        mapping_warning_issued = False
        for y in range(lens_plane_map.shape[0]):
            for x in range(lens_plane_map.shape[1]):
                if lens_plane_map[y, x] > 0:
                    y_hit = int(y + self.y_pixel_deflect_map[y, x] * self.distance_ratio)   # This assumes deflect maps
                    x_hit = int(x + self.x_pixel_deflect_map[y, x] * self.distance_ratio)   # are given in pixels.
                    if 0 <= y_hit < hit_map.shape[0] and 0 <= x_hit < hit_map.shape[1]:
                        hit_map[y_hit, x_hit] = 1
                    elif not mapping_warning_issued:
                        warnings.warn("Some of the area passed to this function is mapped outside the generated " +
                                      "source plane map. This WILL affect area calculations.")
                        mapping_warning_issued = True
        # Now we have a raw map from the critical to caustic curve. We need to make sure flood fill doesn't fill gaps
        # between individual hits.
        hit_map = cv2.dilate(hit_map, None, iterations=_dilation_erosion_steps)
        hit_map = cv2.erode(hit_map, None, iterations=_dilation_erosion_steps)
        # Now hopefully all of the gaps between hits were filled and we can flood fill:
        # TODO: add check to avoid hitting a 1 at (0, 0). Possibly "run" along the edge of the map to find a 0 value?
        hit_map = _flood_fill(hit_map, (0, 0), 0, -1)       # -1 is the outer area outside the caustic curve.
        source_plane_map = np.zeros(hit_map.shape)     # 0 and 1 are inside.
        source_plane_map[np.where(hit_map >= 0)] = 1   # The final map has 1 inside, 0 outside the caustic.
        return source_plane_map.astype(int)

    def corresponding_plane_area(self, pixel_map, plane_redshift=None):
        """Calculate the area marked by integer 1s in pixel_map at given redshift, returns an astropy Quantity."""
        if plane_redshift is None:
            plane_redshift = self.source_z
        dist_source = _cosmology.angular_diameter_distance_z1z2(0, plane_redshift)
        pixel_area = dist_source ** 2 * self.wcs.proj_plane_pixel_area().to(units.rad ** 2) / units.rad ** 2
        values, counts = np.unique(pixel_map, return_counts=True)
        values_counts = dict(zip(values, counts))
        return pixel_area * values_counts[1]

    def caustic_area(self):
        return self.corresponding_plane_area(self.get_caustic_area_map(), self.source_z)
