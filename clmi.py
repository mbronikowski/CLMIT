import cv2
import numpy as np
# import queue
import warnings
import os
import re

from astropy import constants, units, wcs
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.ndimage.measurements import label

_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
_dilation_erosion_steps = 1                     # TODO: fine tune this?
_source_plane_map_rel_size = 1


def set_cosmology(new_cosmology):
    """Set astropy cosmology model for calculations in place of the default flat Lambda CDM with H0=70, OmegaM = 0.3.

    Note that this needs to be performed before initiating a ClusterModel instance, otherwise a new working z
    needs to be set in the instance to reset cosmology-dependent calculations.
    """
    global _cosmology
    _cosmology = new_cosmology


def set_source_plane_size(new_image_scale_factor):
    """Select how much bigger created maps in the source plane should be than in the lens plane; multiplicative."""
    global _source_plane_map_rel_size
    _source_plane_map_rel_size = new_image_scale_factor


def lensing_scaling_factor(z_lens, z_source):
    """Ratio of angular diameter distances from lens to source and from observer to source.

    The ratio is used to scale deflection and magnification maps. 
    Uses module level defined cosmology, which can be redefined using set_cosmology.
    """

    dist_deflector_source = _cosmology.angular_diameter_distance_z1z2(z_lens, z_source)   # commonly referred to as Dds.
    dist_source = _cosmology.angular_diameter_distance_z1z2(0, z_source)                  # Commonly referred to as Ds.
    return (dist_deflector_source / dist_source).value     # Otherwise an astropy Quantity unitless object is generated.


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
    """     # TODO: expand documentation

    def __init__(self, cluster_z=1., source_z=9., kappa_file=None, gamma_file=None,
                 psi_file=None, x_pixel_deflect_file=None, y_pixel_deflect_file=None,
                 x_as_deflect_file=None, y_as_deflect_file=None, map_wcs=None):

        def check_and_load_data(initial_input):
            """Load map from FITS file or path to FITS file and set complimentary attributes."""
            def add_wcs(fits_header):
                # Load WCS from FITS and set object WCS if the loaded one is valid:
                new_wcs = wcs.WCS(fits_header)
                if new_wcs.has_celestial:
                    self.wcs = new_wcs

            def add_lens_map_shape(shape):
                if self._lensing_map_shape is None:
                    self._lensing_map_shape = shape
                elif self._lensing_map_shape[0] != shape[0] or self._lensing_map_shape[1] != shape[1]:
                    raise ValueError("Some maps provided for the model have different shapes.")

            if initial_input is None:  # If no file provided, keep None.
                return None
            elif isinstance(initial_input, str):  # If filename str is provided, open file, load data, close file.
                with fits.open(initial_input) as fits_file:
                    data_table = fits_file[0].data.copy()
                    if self.wcs is None:
                        add_wcs(fits_file[0].header)
                    add_lens_map_shape(data_table.shape)
                return data_table
            # Otherwise, assume a fits file was provided.
            # Copy data from it in case it's closed while the object is in use.
            add_wcs(initial_input[0].header)
            data_table = initial_input[0].data.copy()  # TODO: [0], ['SCI'] or something else? How to guarantee loading?
            add_lens_map_shape(data_table.shape)
            return data_table       # TODO: functionality for passing np tables with data?

        if cluster_z > source_z:
            raise ValueError("Source cannot be placed in front of the cluster! source_z < cluster_z detected: " +
                             str(source_z))
        self.wcs = map_wcs
        self._lensing_map_shape = None
        self.kappa_map = check_and_load_data(kappa_file)
        self.gamma_map = check_and_load_data(gamma_file)
        self.psi_map = check_and_load_data(psi_file)
        self.x_pixel_deflect_map = check_and_load_data(x_pixel_deflect_file)
        self.y_pixel_deflect_map = check_and_load_data(y_pixel_deflect_file)
        self.x_as_deflect_map = check_and_load_data(x_as_deflect_file)
        self.y_as_deflect_map = check_and_load_data(y_as_deflect_file)

        self.cluster_z = cluster_z
        self.source_z = source_z
        self.cluster_angular_diameter_distance = _cosmology.angular_diameter_distance_z1z2(0, self.cluster_z)
        self.source_angular_diameter_distance = _cosmology.angular_diameter_distance_z1z2(0, self.source_z)
        self.distance_ratio = lensing_scaling_factor(self.cluster_z, self.source_z)
        self.distance_param = distance_parameter(self.cluster_z, self.source_z)

        if self._lensing_map_shape is not None:
            self._source_plane_rel_size = _source_plane_map_rel_size
            self._source_plane_map_shape = int(self._lensing_map_shape[0] * self._source_plane_rel_size), \
                int(self._lensing_map_shape[1] * self._source_plane_rel_size)
            self._source_plane_map_offset = int((self._source_plane_map_shape[0] - self._lensing_map_shape[0]) / 2), \
                int((self._source_plane_map_shape[1] - self._lensing_map_shape[1]) / 2)

        self._magnification_map = None              # Private. Access through get_[type]_map.
        self._critical_area_map = None
        self._caustic_area_map = None               # This map has the same angular scale as lens plane maps.
        self._is_multiply_imaged_map = None         # In the lens plane.
        # TODO: add checking deflection map units (arcsec, pixel etc) and conversion handling (can you use astropy?)
        if self.wcs is None:
            warnings.warn("The cluster model was unable to load a working World Coordinate System from provided files.")

    def set_source_z(self, new_z):
        """Set the redshift for which parameters will be calculated and reset previously computed z-dependent values."""
        if new_z < self.cluster_z:
            raise ValueError("Source cannot be placed in front of the cluster! source_z < cluster_z detected: " +
                             str(new_z))
        self.source_z = new_z
        self.source_angular_diameter_distance = _cosmology.angular_diameter_distance_z1z2(0, self.source_z)
        self.distance_ratio = lensing_scaling_factor(self.cluster_z, new_z)
        self.distance_param = distance_parameter(self.cluster_z, new_z)
        self._magnification_map = None        # Reset all z-specific data. It will be generated as needed.
        self._critical_area_map = None
        self._caustic_area_map = None
        self._is_multiply_imaged_map = None   # z-dependent, despite being in the lens plane.

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

    def _generate_critical_curve(self):
        magnif_map = self.get_magnification_map()
        magnif_sign = np.zeros(magnif_map.shape)      # Get map of magnification signs, get rid of outer area.
        magnif_sign[np.where(magnif_map > 0.)] = 1
        magnif_sign_eroded = cv2.erode(magnif_sign, None, iterations=1)
        magnif_sign_dilated = cv2.dilate(magnif_sign, None, iterations=1)
        return (magnif_sign_dilated - magnif_sign_eroded).astype(int)

    def get_caustic_area_map(self):
        """Public method to access the map of the region inside the caustic curve, denoted by 1s."""
        if self._caustic_area_map is None:
            self._caustic_area_map = self.map_to_source_plane(self.get_critical_area_map())
        return self._caustic_area_map

    def get_is_multiply_imaged_map(self):
        if self._is_multiply_imaged_map is None:
            self._generate_is_multiply_imaged_map()
        return self._is_multiply_imaged_map

    def _generate_is_multiply_imaged_map(self):
        result_map = np.zeros(self._lensing_map_shape, dtype=int)
        caustic_area_map = self.get_caustic_area_map()
        critical_area_map = self.get_critical_area_map()
        for y in range(self._lensing_map_shape[0]):
            for x in range(self._lensing_map_shape[1]):
                if critical_area_map[y, x] == 1:
                    result_map[y, x] = 1
                else:
                    y_mapped = int(y
                                   - self.y_pixel_deflect_map[y, x] * self.distance_ratio
                                   + self._source_plane_map_offset[0])
                    x_mapped = int(x
                                   - self.x_pixel_deflect_map[y, x] * self.distance_ratio
                                   + self._source_plane_map_offset[1])
                    if 0 <= y_mapped < caustic_area_map.shape[0] and 0 <= x_mapped < caustic_area_map.shape[1]:
                        if caustic_area_map[y_mapped, x_mapped] == 1:
                            result_map[y, x] = 1
        self._is_multiply_imaged_map = result_map

    def map_to_source_plane(self, lens_plane_map):
        """Map arbitrary area from the lens plane to the source plane.

        The input needs to be a numpy array of integers with 1s for pixels which are to be mapped, 0s otherwise.
        The function will fill in any holes in the mapped source plane area; the function does not handle areas with
        holes in them in the source plane. This aids with numerical issues inside the caustic curve.
        """
        if self.x_pixel_deflect_map is None or self.y_pixel_deflect_map is None:
            self._generate_pixel_deflect_maps()
        if lens_plane_map.shape != self._lensing_map_shape:
            raise ValueError("The array to be mapped back to the source plane needs to have the same shape \
                as deflection maps.")
        hit_map = np.zeros(self._source_plane_map_shape, dtype=int)
        offset = self._source_plane_map_offset
        mapping_warning_issued = False
        for y in range(lens_plane_map.shape[0]):
            for x in range(lens_plane_map.shape[1]):
                if lens_plane_map[y, x] > 0:        # "If the pixel in the lens plane is meant to be mapped"
                    y_hit = int(y - self.y_pixel_deflect_map[y, x] * self.distance_ratio + offset[0])
                    x_hit = int(x - self.x_pixel_deflect_map[y, x] * self.distance_ratio + offset[1])
                    if 0 <= y_hit < hit_map.shape[0] and 0 <= x_hit < hit_map.shape[1]:
                        hit_map[y_hit, x_hit] = 1
                    elif not mapping_warning_issued:
                        warnings.warn("Some of the area passed to this function is mapped outside the generated " +
                                      "source plane map. This WILL affect area calculations.")
                        mapping_warning_issued = True
        # Now we have a raw map, e.g. from the critical to caustic curve.
        # If for some reason the map isn't solid, i.e. there are empty spots between mapped pixels:
        # hit_map = cv2.dilate(hit_map, None, iterations=_dilation_erosion_steps)   # This requires hit_map to be float
        # hit_map = cv2.erode(hit_map, None, iterations=_dilation_erosion_steps)
        return hit_map

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


_gamma_regex = re.compile(r'(gamma|shear)[^12]*$')
_kappa_regex = re.compile(r'(kappa|convergence)[^12]*$')
_psi_regex = re.compile(r'(psi|poten)[^12]*fits$')


def load_to_model(path, cluster_z, source_z=9.):        # Disgusting boilerplate.
    """Attempt to load data into a ClusterModel object from a folder, assuming typical filenames were used."""
    def append_path(file_name):
        return path + '/' + file_name
    object_input = {
        "cluster_z": cluster_z,
        "source_z": source_z,
        "x_as_deflect_file": None,
        "y_as_deflect_file": None,
        "x_pixel_deflect_file": None,
        "y_pixel_deflect_file": None,
        "kappa_file": None,
        "gamma_file": None,
        "psi_file": None
    }
    file_list = os.listdir(path)
    for filename in file_list:
        gamma_match, kappa_match, psi_match = re.search(_gamma_regex, filename), re.search(_kappa_regex, filename), \
                                               re.search(_psi_regex, filename)
        if ".fits" not in filename:
            continue
        if gamma_match is not None:
            assert object_input["gamma_file"] is None, "Multiple files match attempted pattern for gamma/shear file."
            object_input["gamma_file"] = append_path(filename)
        if kappa_match is not None:
            assert object_input["kappa_file"] is None, \
                "Multiple files match attempted pattern for kappa/convergence file."
            object_input["kappa_file"] = append_path(filename)
        if psi_match is not None:
            assert object_input["psi_file"] is None, "Multiple files match attempted pattern for psi/potential file."
            object_input["psi_file"] = append_path(filename)
        if "x-arcsec-deflect.fits" in filename or "dx.fits" in filename or "deflect_arcsec_x.fits" in filename:
            assert object_input["x_as_deflect_file"] is None, \
                "Multiple files match attempted pattern for x [arcsec] deflect file."
            object_input["x_as_deflect_file"] = append_path(filename)
        if "y-arcsec-deflect.fits" in filename or "dy.fits" in filename or "deflect_arcsec_y.fits" in filename:
            assert object_input["y_as_deflect_file"] is None, \
                "Multiple files match attempted pattern for y [arcsec] deflect file."
            object_input["y_as_deflect_file"] = append_path(filename)
        if "x-pixels-deflect.fits" in filename:
            assert object_input["x_pixel_deflect_file"] is None, \
                "Multiple files match attempted pattern for x [px] deflect file."
            object_input["x_pixel_deflect_file"] = append_path(filename)
        if "y-pixels-deflect.fits" in filename:
            assert object_input["y_pixel_deflect_file"] is None, \
                "Multiple files match attempted pattern for y [px] deflect file."
            object_input["y_pixel_deflect_file"] = append_path(filename)
    return ClusterModel(**object_input)

