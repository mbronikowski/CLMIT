import cv2
import numpy as np
# import queue
import warnings
import os
import re
import math

from astropy import constants, units, wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.ndimage.measurements import label

_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)      # Default cosmology assumptions. Change with set_cosmology()
_dilation_erosion_steps = 1
_source_plane_map_rel_size = 1
_multip_map_kernel_size = 7


class _LUTManager:
    """LUTManager is the class for _LUTM, this project's lookup table manager.

        Any lookup tqable (LUT) access in the module is managed by this class.
        It holds the dictionary for cosmological comoving volume as a function of redshift.
        Setting a new cosmology resets the lookup table.
        """
    def __init__(self):
        self._comoving_volume_dict = {}

    def get_comoving_volume(self, z):
        """The preferred method of accessing comoving volume."""
        if z not in self._comoving_volume_dict:
            self._comoving_volume_dict[z] = _cosmology.comoving_volume(z)
        return self._comoving_volume_dict[z]

    def reset_comoving_volume_dict(self):   # Used when setting a new cosmology.
        self._comoving_volume_dict.clear()


_LUTM = _LUTManager()


def set_cosmology(new_cosmology):
    """Set astropy cosmology model for calculations in place of the default flat Lambda CDM with H0=70, OmegaM = 0.3."""
    global _cosmology
    _cosmology = new_cosmology
    _LUTM.reset_comoving_volume_dict()
    for instance in ClusterModel.active_instances:
        instance.source_z = instance.source_z       # Not very surgical, but will trigger a reset of z-dependent maps.


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


def magnification(kappa, gamma, scaling_factor):
    """Basic lensing formula for magnification based on convergence (kappa) and shear (gamma) for given Dds/Ds.

    :param kappa: Lensing convergence parameter or array thereof.
    :type kappa: float or np.ndarray
    :param gamma: Lensing shear parameter or an array thereof.
    :type gamma: float or np.ndarray
    :param float scaling_factor: Ratio of angular diameter distances from lens to source and from observer to source.
    """
    return ((1 - kappa * scaling_factor) ** 2 - (gamma * scaling_factor) ** 2) ** -1


def total_magnification(magnif_list):
    """Calculate total magnification from an array of point magnifications, assuming constant surface brightness."""
    mag_list = np.abs(np.array(magnif_list))
    return mag_list.size / np.sum(1. / mag_list)


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
    """Flood fill tool on a numpy array."""
    if array[start_pixel] != targeted_val:          # Sanity check
        raise ValueError("The flood fill tool has failed. Array had value of " +
                         str(array[start_pixel]) + ", expected " + str(targeted_val) + ".")
    new_arr = np.copy(array)
    label_map = label(new_arr == new_arr[start_pixel])[0]
    new_arr[label_map == label_map[start_pixel]] = new_val
    return new_arr


class ClusterModel:
    """Handles individual cluster models and calculations based on them.

    :params:
        :param float cluster_z: Redshift of the cluster. Default is 1.0.
        :param float source_z: Redshift of the source plane for which calculations are made. Default is 9.0.
        :param kappa_file: Cluster lensing convergence map, either a FITS file or path to a FITS file. Default is None.
        :type kappa_file: str or None or astropy HDUList
        :param gamma_file: Cluster lensing total shear map, either a FITS file or path to a FITS file. Default is None.
        :type gamma_file: str or None or astropy HDUList
        :param psi_file: Cluster lensing potential map, either a FITS file or path to a FITS file. Default is None.
        :type psi_file: str or None or astropy HDUList
        :param x_pixel_deflect_file: Cluster lensing x axis deflection map measured in pixels,
                                     either a FITS file or path to a FITS file. Default is None.
        :type x_pixel_deflect_file: str or None or astropy HDUList
        :param y_pixel_deflect_file: Cluster lensing y axis deflection map measured in pixels,
                                     either a FITS file or path to a FITS file. Default is None.
        :type y_pixel_deflect_file: str or None or astropy HDUList
        :param x_as_deflect_file: Cluster lensing x axis deflection map measured in arcseconds,
                                  either a FITS file or path to a FITS file. Default is None.
        :type x_as_deflect_file: str or None or astropy HDUList
        :param y_as_deflect_file: Cluster lensing y axis deflection map measured in arcseconds,
                                  either a FITS file or path to a FITS file. Default is None.
        :type y_as_deflect_file: str or None or astropy HDUList
        :param map_wcs: Astropy World Coordinate System object to be used by the class.
                        If not provided, the object will attempt to load one from provided maps. Default is None.
        :type map_wcs: astropy.wcs.WCS or None

    :methods:
        set_source_z: Set a new source redshift for the cluster.
        point_magnification: Calculate magnification for a set of individual points.
        map_to_source_plane: Map an area from the lens plane to the source plane.
        map_solid_angle: Calculate the solid angle marked by a map.

    :attributes:
        :cvar list active_instances: Stores references to all active instances.
        :ivar wcs: stores model maps.
        :type wcs: astropy.wcs.WCS or None
        :ivar kappa_map: Map of the lensing convergence / kappa parameter.
        :type kappa_map: np.ndarray or None
        :ivar gamma_map: Map of the lensing shear / gamma parameter.
        :type gamma_map: np.ndarray or None
        :ivar psi_map: Map of the lensing gravitational potential / psi parameter.
        :type psi_map: np.ndarray or None
        :ivar x_pixel_deflect_map: Map of light deflection in the x axis, in pixels.
        :type x_pixel_deflect_map: np.ndarray or None
        :ivar y_pixel_deflect_map : Map of light deflection in the y axis, in pixels.
        :type y_pixel_deflect_map: np.ndarray or None
        :ivar x_as_deflect_map : np.ndarray or None. Map of light deflection in the x axis, in arcseconds.
        :type x_as_deflect_map: np.ndarray or None
        :ivar y_as_deflect_map : np.ndarray or None. Map of light deflection in the y axis, in arcseconds.
        :type y_as_deflect_map: np.ndarray or None
        :ivar float cluster_z: Redshift of the cluster, used in most calculations.
        :ivar cluster_angular_diameter_distance: Cosmological angular diameter distance to the cluster.
        :type cluster_angular_diameter_distance: astropy.units.Quantity
        :ivar source_angular_diameter_distance: Cosmological angular diameter distance to the source.
        :type source_angular_diameter_distance: astropy.units.Quantity
        :ivar float distance_ratio: Ratio of angular diameter distances from lens to source and from observer to source.
        :ivar float distance_param: Used in time delay equation, see https://arxiv.org/pdf/astro-ph/9606001.pdf Eq. 63.

    :properties:
        source_z (float): Redshift of the source for which most model calculations are made.
        magnification_map (np.ndarray): Map of magnification of the source plane.
        critical_area_map (np.ndarray): Map of the area inside the critical curve in the lens plane.
        critical_curve (np.ndarray): Map of the critical curve in the source plane.
        caustic_area_map (np.ndarray): Map of the area inside the caustic curve in the source plane.
        is_multiply_imaged_map (np.ndarray): Map of the area in the source plane where objects are multiply imaged.
        is_singly_imaged_map (np.ndarray): Map of the area in the source plane where objects are singly imaged.
    """

    active_instances = []      # Used by the module's set_cosmology function to trigger a calculated data reset.

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
            raise ValueError(f"Source cannot be placed in front of the cluster! \
            source_z < cluster_z detected: {str(source_z)} < {str(cluster_z)}")
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
        self._source_z = source_z
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
        if self.wcs is None:
            warnings.warn("The cluster model was unable to load a working World Coordinate System from provided files.")

        self.active_instances.append(self)

    def __del__(self):
        self.active_instances.remove(self)

    @property
    def source_z(self):
        return self._source_z

    @source_z.setter
    def source_z(self, new_z):
        """Set the redshift for which parameters will be calculated and reset previously computed z-dependent values."""
        if new_z < self.cluster_z:
            raise ValueError("Source cannot be placed in front of the cluster! source_z < cluster_z detected: " +
                             str(new_z))
        self._source_z = new_z
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

    @property
    def magnification_map(self):
        """Obtain the magnification map for the model and source redshift.

        :returns: Magnification map for the cluster model and source redshift.
        :rtype: np.ndarray
        """
        if self._magnification_map is None:
            self._generate_magnification_map()
        return self._magnification_map

    def _generate_magnification_map(self):
        self._magnification_map = np.nan_to_num(magnification(self.kappa_map, self.gamma_map, self.distance_ratio),
                                                nan=1.0)

    def point_magnification(self, points, redshifts=None, coord_units=None, box_size=None):
        """
        Extracts magnification for a given point or set of points.

        :param points: Set of points for which magnifications are to be computed.
                       Can be a string, astropy.coordinates.SkyCoord object, or iterable thereof.
        :type points: str or astropy.coordinates.SkyCoord or iterable
        :param redshifts: Set of corresponding redshifts for which magnifications are to be computed.
                          If none are provided, all magnifications will be computed for the instance's source_z.
        :type redshifts: float or iterable of floats or None
        :param coord_units: If points is of str type, this will be passed to an astropy.coordinates.SkyCoord object
                            for coordinate calculations.
        :type coord_units: astropy.units.Unit or str or tuple of astropy.units.Unit or str or None
        :param box_size: Size of the box for which magnification should be estimated. If provided, instead of point
                          magnification, total magnification of a box will be provided.
                          If not Quantity, arcsec are assumed.
        :returns: Magnification values for every point in the provided data.
        :rtype: float or np.ndarray
        """
        def str_to_skycoord(string):
            if coord_units is not None:
                return SkyCoord(string, unit=coord_units)
            else:
                return SkyCoord(string)

        box_pixel_halfside = 0
        if box_size is not None:
            if not isinstance(box_size, units.Quantity):
                box_size = box_size * units.arcsec
            box_pixel_halfside = box_size.to(units.arcsec) / self.wcs.proj_plane_pixel_scales()[0].to(units.arcsec)
            box_pixel_halfside = int(box_pixel_halfside.value / 2)        # Half of distance from middle of box to edge

        # Handle one point:

        if isinstance(points, str):
            points = str_to_skycoord(points)

        if points.shape == ():
            if redshifts is not None:
                scaling_factor = lensing_scaling_factor(self.cluster_z, redshifts)
            else:
                scaling_factor = self.distance_ratio

            x_coord, y_coord = self.wcs.world_to_pixel(points)
            x_coord, y_coord = round(float(x_coord)), round(float(y_coord))       # Cast to int
            kappa, gamma = self.kappa_map[y_coord - box_pixel_halfside:y_coord + box_pixel_halfside + 1,
                                          x_coord - box_pixel_halfside:x_coord + box_pixel_halfside + 1], \
                           self.gamma_map[y_coord - box_pixel_halfside:y_coord + box_pixel_halfside + 1,
                                          x_coord - box_pixel_halfside:x_coord + box_pixel_halfside + 1]
            magnifs = magnification(kappa, gamma, scaling_factor)

            return total_magnification(magnifs)

        # Now handle iterable inputs:
        if isinstance(redshifts, float):
            redshifts = np.array([redshifts] * points.shape[0])     # If only one z provided, assume it for all points
        magnifications = np.empty(points.shape)
        for i, point in enumerate(points):
            if isinstance(point, str):
                if coord_units is not None:
                    point = SkyCoord(point, unit=coord_units)
                else:
                    point = SkyCoord(point)
            if redshifts is None:
                scaling_factor = self.distance_ratio
            else:
                scaling_factor = lensing_scaling_factor(self.cluster_z, redshifts[i])

            x_coord, y_coord = self.wcs.world_to_pixel(point)
            x_coord, y_coord = round(float(x_coord)), round(float(y_coord))       # Cast to int
            kappa, gamma = self.kappa_map[y_coord - box_pixel_halfside:y_coord + box_pixel_halfside,
                                          x_coord - box_pixel_halfside:x_coord + box_pixel_halfside + 1], \
                           self.gamma_map[y_coord - box_pixel_halfside:y_coord + box_pixel_halfside,
                                          x_coord - box_pixel_halfside:x_coord + box_pixel_halfside + 1]
            magnifs = magnification(kappa, gamma, scaling_factor)
            magnifications[i] = total_magnification(magnifs)

        return magnifications

    @property
    def critical_area_map(self):
        """Obtain a map of the area inside the critical curve in the lens plane.

        :returns: Map of the area inside the critical curve, with 1 for inside, 0 for outside.
        :rtype: np.ndarray
        """
        if self._critical_area_map is None:
            self._generate_critical_area_map()
        return self._critical_area_map

    def _generate_critical_area_map(self):
        magnif_map = self.magnification_map
        magnif_sign = np.ones(magnif_map.shape, dtype="uint8")  # Get map of magnification signs, get rid of outer area.
        magnif_sign[np.where(magnif_map < 0.)] = -1
        magnif_sign = _flood_fill(magnif_sign, (0, 0), 1, 0)         # Note: we're assuming (0, 0) is outside
        self._critical_area_map = np.abs(magnif_sign)                # the critical curve. If it isn't, this will crash.

    @property
    def source_plane_mapping_map(self):
        """Calculates an array of indices, which show how pixels are mapped from lens plane to source plane.

        The arrays are 2 x dimension of source plane maps. The 1st array contains indices, the 2nd contains
        a mask of which indices are valid. It's less than ideal clarity-wise.
        """
        if self.x_pixel_deflect_map is None or self.y_pixel_deflect_map is None:
            self._generate_pixel_deflect_maps()
        coords_mapped = np.indices(self.y_pixel_deflect_map.shape)
        coords_mapped[0] += self._source_plane_map_offset[0] \
                            - (self.distance_ratio * self.y_pixel_deflect_map).astype(int)
        coords_mapped[1] += self._source_plane_map_offset[1] \
                            - (self.distance_ratio * self.x_pixel_deflect_map).astype(int)
        np.nan_to_num(coords_mapped, copy=False, nan=-1)
        valid_px_mask = (0 <= coords_mapped[0]) \
                        & (coords_mapped[0] < self._source_plane_map_shape[0]) \
                        & (0 <= coords_mapped[1]) \
                        & (coords_mapped[1] < self._source_plane_map_shape[1])
        return coords_mapped, valid_px_mask

    @property
    def critical_curve(self):
        magnif_map = self.magnification_map
        magnif_sign = np.zeros(magnif_map.shape)      # Get map of magnification signs, get rid of outer area.
        magnif_sign[np.where(magnif_map > 0.)] = 1
        magnif_sign_eroded = cv2.erode(magnif_sign, None, iterations=1)
        magnif_sign_dilated = cv2.dilate(magnif_sign, None, iterations=1)
        return (magnif_sign_dilated - magnif_sign_eroded).astype(int)

    @property
    def caustic_area_map(self):
        """Obtain a map of the area inside the caustic curve in the source plane.

        :returns: Map of the area inside the caustic curve, with 1 for inside, 0 for outside.
        :rtype: np.ndarray
        """
        if self._caustic_area_map is None:
            self._caustic_area_map = self.map_to_source_plane(self.critical_area_map)
            self._caustic_area_map = cv2.erode(cv2.dilate(      # Gets rid of numerical issues
                self._caustic_area_map,
                None, iterations=_dilation_erosion_steps), None, iterations=_dilation_erosion_steps)
        return self._caustic_area_map

    def image_multiplicity_map(self, minimum_magnif=0):
        source_map = np.zeros(self._source_plane_map_shape)
        magnif_map = np.abs(self.magnification_map)
        coords_mapped, valid_px_mask = self.source_plane_mapping_map
        valid_px_mask = np.where(magnif_map > minimum_magnif, valid_px_mask, False)
        magnif_map = np.where(valid_px_mask, 1 / magnif_map, 0)     # Now this stores 1/mu for every valid pixel.
        coords_mapped[0] = np.where(valid_px_mask, coords_mapped[0], 0)     # Pointing out-of-bounds pixels to
        coords_mapped[1] = np.where(valid_px_mask, coords_mapped[1], 0)     # a harmless [0, 0]
        np.add.at(source_map, (coords_mapped[0], coords_mapped[1]), magnif_map)
        return cv2.medianBlur(np.array(np.rint(source_map), dtype="uint8"), _multip_map_kernel_size)
        # For some reason, np.rint(source_map, dtype='uint8') throws errors in Jupyter.

    # TODO: The following two functions seem completely redundant

    @property
    def is_multiply_imaged_map(self):
        """Obtain a map of the area in the source plane where objects are multiply imaged.

            :returns: Map of the multiply imaged area, with 1 for multiply-, 0 for singly-imaged.
            :rtype: np.ndarray
        """
        if self._is_multiply_imaged_map is None:
            self._generate_is_multiply_imaged_map()
        return self._is_multiply_imaged_map

    @property
    def is_singly_imaged_map(self):
        """Obtain a map of the area in the source plane where objects are singly imaged.

                :returns: Map of the multiply imaged area, with 0 for multiply-, 1 for singly-imaged.
                :rtype: np.ndarray
        """
        return np.ones(self.is_multiply_imaged_map.shape, dtype=int) - self.is_multiply_imaged_map

    def _generate_is_multiply_imaged_map(self):
        coords_mapped, valid_px_mask = self.source_plane_mapping_map
        self._is_multiply_imaged_map = np.where((self.caustic_area_map[coords_mapped[0], coords_mapped[1]] == 1)
                                                & valid_px_mask, 1, 0)

    def map_to_source_plane(self, lens_plane_map):
        """Map arbitrary area from the lens plane to the source plane.

            :param np.ndarray(int) lens_plane_map: Map to be mapped to source plane.
                    Area which is to be mapped should be denoted by 1s, the rest by 0s.
            :returns: Map in the source plane, where 1s correspond to the area with 1s in the lens plane.
            :rtype: np.ndarray

        The function will fill in any holes in the mapped source plane area; the function does not handle areas with
        holes in them in the source plane. This aids with numerical issues inside the caustic curve.
        """
        if lens_plane_map.shape != self._lensing_map_shape:
            raise ValueError("The array to be mapped back to the source plane needs to have the same shape \
                as deflection maps.")

        coords_mapped, valid_px_mask = self.source_plane_mapping_map
        valid_px_mask = np.where(lens_plane_map == 1, valid_px_mask, False)

        hit_map = np.zeros(self._source_plane_map_shape, dtype="uint8")
        coords_mapped = coords_mapped[0][valid_px_mask], coords_mapped[1][valid_px_mask]
        hit_map[coords_mapped] = 1

        return hit_map

    def map_solid_angle(self, pixel_map):   # TODO: is this useful?
        """Calculate angular area marked by integer 1s in pixel_map at given redshift, returns an astropy Quantity."""
        pixel_area = self.wcs.proj_plane_pixel_area().to(units.rad ** 2)
        values, counts = np.unique(pixel_map, return_counts=True)
        values_counts = dict(zip(values, counts))
        return pixel_area * values_counts[1]

    def caustic_area_solid_angle(self):     # TODO: is this useful?
        return self.map_solid_angle(self.caustic_area_map)

    def solid_angle_from_magnif(self, magnif_map_mask):         # TODO: is this useful?
        pixel_area = self.wcs.proj_plane_pixel_area().to(units.rad ** 2)
        magnif_map = self.magnification_map
        area_in_px = 0.
        for y in range(magnif_map.shape[0]):
            for x in range(magnif_map.shape[1]):
                if not magnif_map_mask[y, x] == 0:
                    area_in_px += 1 / magnif_map[y, x]
        return pixel_area * area_in_px


_gamma_regex = re.compile(r'(gamma|shear)[^12]*$')
_kappa_regex = re.compile(r'(kappa|convergence)[^12]*$')
_psi_regex = re.compile(r'(psi|poten)[^12]*fits$')


def load_to_model(path, cluster_z, source_z=9.):        # Disgusting boilerplate, but it works.
    """Attempt to load data into a ClusterModel object from a folder, assuming typical filenames were used."""
    # This is disgusting boilerplate, but I guess it works.
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


def redshift_volume_bins(cluster_model, lens_plane_map, redshift_bins, *args, **kwargs):        # TODO: is this useful?
    """Takes a cluster model and an array of z bins and generates a histogram of delensed comoving volumes in bins."""
    map_is_callable = callable(lens_plane_map)
    result_list = []
    for i in range(len(redshift_bins) - 1):
        mid_z = (redshift_bins[i] + redshift_bins[i+1]) / 2
        cluster_model.source_z = mid_z
        if map_is_callable:
            source_plane_map = cluster_model.map_to_source_plane(lens_plane_map(*args, **kwargs))
        else:
            source_plane_map = cluster_model.map_to_source_plane(lens_plane_map)
        vol_in_bin = (cluster_model.map_solid_angle(source_plane_map)
                      / (4 * math.pi * units.rad ** 2)
                      * (_LUTM.get_comoving_volume(redshift_bins[i+1]) - _LUTM.get_comoving_volume(redshift_bins[i])))
        result_list.append(vol_in_bin)
    return units.Quantity(result_list, units.Mpc ** 3)      # Converting list of Quantity objects to one Quantity.
