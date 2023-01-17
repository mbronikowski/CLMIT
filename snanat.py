import cv2
import numpy as np
# import queue
import warnings
import os
import re
import math

from astropy import constants, units, wcs
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.ndimage.measurements import label


_snana_documentation = "DOCUMENTATION:\n\
    PURPOSE: Host library for imaging-only sims\n\
    INTENT:  Nominal\n\
    REF:\n\
    - AUTHOR: R. Hounell et al, 2018 (WFIRST SN sims)\n\
      ADS:    https://ui.adsabs.harvard.edu/abs/2018ApJ...867...23H\n\
    USAGE_KEY:  HOSTLIB_FILE\n\
    USAGE_CODE: snlc_sim.exe\n\
    NOTES:\n\
    - based on CANDELS data; see Appendix B of H18\n\
    - B band NaN replaced with 99 to avoid abort (July 6 2021)\n\
    VERSIONS:\n\
    - DATE: 2017\n\
      AUTHORS: R. Hounsell, D.Scolnic, R. Kessler\n\
DOCUMENTATION_END:\n\
VARNAMES: LENSID NIMG ZLENS ZSRC XIMG_SRC YIMG_SRC MAGNIF DELAY\n"


def show_img(data):
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    """VERY crude plotting for quick and dirty previews."""
    plt.imshow(data, norm=simple_norm(data, 'sqrt', percent=99.), origin='lower', cmap='Greys')
    plt.plot()
    plt.show()


def snana_mock_magnif_catalogue(magnif_map, magnif_mask, z_lens, z_source, filename,
                                magnif_cutoff=None, n_lenses=100, id_modifier=0):
    """Create a mock catalogue of lensing systems as input for SNANA that follows the distribution of magnifications."""
    magnif_abs_map = np.abs(magnif_map)
    if magnif_cutoff is not None:       # truncate too large magnifications
        magnif_abs_map = np.where(magnif_abs_map > magnif_cutoff, magnif_cutoff, magnif_abs_map)
    magnif_weights = np.where(magnif_mask != 0, 1. / magnif_abs_map, 0).flatten()
    magnif_weights = np.where(magnif_weights > 1., 0., magnif_weights)  # Eliminate magnification < 1
    magnif_weights /= np.sum(magnif_weights)
    magnif_selections = np.random.choice(magnif_abs_map.flatten(), size=n_lenses, p=magnif_weights)
    with open(filename, 'w') as outfile:
        file_header = _snana_documentation
        outfile.write(file_header)
        middle_entry_part = ' 1 ' + str(z_lens) + ' ' + str(z_source) + ' 0.0 0.0 '
        for i in range(n_lenses):
            catalogue_entry = "LENS: " + str(i + id_modifier) + middle_entry_part + str(magnif_selections[i]) + ' 0.0\n'
            outfile.write(catalogue_entry)

