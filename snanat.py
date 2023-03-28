import numpy as np
import math
import os
from sncosmo import read_snana_fits
from astropy.io import ascii


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


ROMAN_FIELD_NAMES = ["WIDE", "DEEP"]
ROMAN_LIB_DICT = {1: "DEEP", 2: "DEEP", 3: "WIDE", 4: "WIDE", 5: "DEEP", 6: "WIDE", 7: "DEEP", 8: "WIDE", 9: "WIDE",
                  10: "DEEP", 11: "WIDE", 12: "WIDE", 13: "WIDE", 14: "WIDE", 15: "DEEP", 16: "WIDE", 17: "WIDE",
                  18: "WIDE", 19: "WIDE", 20: "WIDE", 21: "WIDE", 22: "DEEP", 23: "WIDE", 24: "DEEP", 25: "WIDE",
                  26: "WIDE", 27: "DEEP", 28: "WIDE", 29: "WIDE", 30: "WIDE", 31: "DEEP", 32: "WIDE", 33: "DEEP",
                  34: "WIDE", 35: "DEEP", 36: "DEEP", 37: "WIDE", 38: "WIDE", 39: "WIDE", 40: "WIDE", 41: "WIDE",
                  42: "WIDE", 43: "WIDE", 44: "WIDE", 45: "WIDE", 46: "DEEP", 47: "DEEP", 48: "WIDE", 49: "WIDE",
                  50: "DEEP", 51: "WIDE", 52: "WIDE", 53: "WIDE", 54: "WIDE", 55: "WIDE", 56: "WIDE", 57: "WIDE",
                  58: "WIDE", 59: "WIDE", 60: "WIDE", 61: "WIDE", 62: "WIDE", 63: "WIDE", 64: "WIDE", 65: "WIDE",
                  66: "WIDE", 67: "WIDE", 68: "DEEP", 69: "DEEP", 70: "WIDE", 71: "WIDE", 72: "WIDE", 73: "DEEP",
                  74: "WIDE", 75: "DEEP", 76: "WIDE", 77: "DEEP", 78: "WIDE", 79: "WIDE", 80: "WIDE", 81: "WIDE",
                  82: "DEEP", 83: "WIDE", 84: "WIDE", 85: "DEEP", 86: "WIDE", 87: "WIDE", 88: "WIDE", 89: "WIDE",
                  90: "WIDE", 91: "WIDE", 92: "WIDE", 93: "WIDE", 94: "WIDE", 95: "DEEP", 96: "WIDE", 97: "WIDE",
                  98: "WIDE", 99: "WIDE", 100: "DEEP", 101: "DEEP", 102: "DEEP", 103: "WIDE", 104: "WIDE", 105: "WIDE",
                  106: "WIDE", 107: "WIDE", 108: "WIDE", 109: "DEEP", 110: "WIDE", 111: "WIDE", 112: "WIDE",
                  113: "WIDE", 114: "DEEP", 115: "WIDE", 116: "WIDE", 117: "WIDE", 118: "WIDE", 119: "WIDE",
                  120: "WIDE", 121: "WIDE", 122: "WIDE", 123: "WIDE", 124: "WIDE", 125: "DEEP", 126: "DEEP",
                  127: "DEEP", 128: "WIDE", 129: "WIDE", 130: "WIDE", 131: "DEEP", 132: "DEEP", 133: "WIDE",
                  134: "DEEP", 135: "WIDE", 136: "DEEP", 137: "WIDE", 138: "WIDE", 139: "WIDE", 140: "WIDE",
                  141: "WIDE", 142: "DEEP", 143: "WIDE", 144: "DEEP", 145: "WIDE", 146: "DEEP", 147: "WIDE",
                  148: "WIDE", 149: "WIDE", 150: "WIDE", 151: "WIDE", 152: "WIDE", 153: "DEEP", 154: "DEEP",
                  155: "WIDE", 156: "WIDE", 157: "WIDE", 158: "WIDE", 159: "WIDE", 160: "WIDE", 161: "WIDE",
                  162: "WIDE", 163: "WIDE", 164: "WIDE", 165: "WIDE", 166: "WIDE", 167: "DEEP", 168: "WIDE",
                  169: "WIDE", 170: "WIDE", 171: "WIDE", 172: "WIDE", 173: "WIDE", 174: "WIDE", 175: "WIDE",
                  176: "WIDE", 177: "WIDE", 178: "DEEP", 179: "WIDE", 180: "WIDE", 181: "DEEP", 182: "DEEP",
                  183: "DEEP", 184: "WIDE", 185: "WIDE", 186: "WIDE", 187: "WIDE", 188: "DEEP", 189: "WIDE",
                  190: "WIDE", 191: "WIDE", 192: "WIDE", 193: "WIDE", 194: "DEEP", 195: "WIDE", 196: "WIDE",
                  197: "DEEP", 198: "WIDE", 199: "WIDE", 200: "WIDE", 201: "DEEP", 202: "WIDE", 203: "WIDE",
                  204: "WIDE", 205: "DEEP", 206: "DEEP", 207: "WIDE", 208: "WIDE", 209: "WIDE", 210: "WIDE",
                  211: "DEEP", 212: "DEEP", 213: "WIDE", 214: "WIDE", 215: "WIDE", 216: "WIDE", 217: "WIDE",
                  218: "DEEP", 219: "DEEP", 220: "WIDE", 221: "WIDE", 222: "WIDE", 223: "WIDE", 224: "WIDE",
                  225: "DEEP", 226: "WIDE", 227: "WIDE", 228: "WIDE", 229: "WIDE", 230: "WIDE", 231: "WIDE",
                  232: "DEEP", 233: "WIDE", 234: "WIDE", 235: "DEEP", 236: "WIDE", 237: "DEEP", 238: "WIDE",
                  239: "WIDE", 240: "WIDE", 241: "WIDE", 242: "WIDE", 243: "WIDE", 244: "WIDE", 245: "WIDE",
                  246: "WIDE", 247: "WIDE", 248: "WIDE", 249: "WIDE", 250: "WIDE", 251: "WIDE", 252: "WIDE",
                  253: "WIDE", 254: "WIDE", 255: "DEEP", 256: "DEEP", 257: "DEEP", 258: "DEEP", 259: "WIDE",
                  260: "WIDE", 261: "DEEP", 262: "WIDE", 263: "WIDE", 264: "DEEP", 265: "WIDE", 266: "WIDE",
                  267: "WIDE", 268: "WIDE", 269: "WIDE", 270: "WIDE", 271: "DEEP", 272: "WIDE", 273: "WIDE",
                  274: "WIDE", 275: "WIDE", 276: "WIDE", 277: "WIDE", 278: "DEEP", 279: "DEEP", 280: "DEEP",
                  281: "DEEP", 282: "DEEP", 283: "WIDE", 284: "DEEP", 285: "WIDE", 286: "WIDE", 287: "WIDE",
                  288: "WIDE", 289: "DEEP", 290: "DEEP", 291: "WIDE", 292: "WIDE", 293: "WIDE", 294: "WIDE",
                  295: "WIDE", 296: "WIDE", 297: "DEEP", 298: "WIDE", 299: "WIDE", 300: "WIDE"}


def show_img(data):
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    """VERY crude plotting for quick and dirty previews."""
    plt.imshow(data, norm=simple_norm(data, 'sqrt', percent=99.), origin='lower', cmap='Greys')
    plt.plot()
    plt.show()


def snana_mock_magnif_catalogue_setz(magnif_map, magnif_mask, z_lens, z_source, output_filename,
                                     magnif_cutoff=None, n_lenses=100, id_modifier=0):
    """Create a mock catalogue of lensing systems at set z for SNANA that follows the distribution of magnifications."""
    magnif_abs_map = np.abs(magnif_map)
    if magnif_cutoff is not None:       # truncate too large magnifications
        magnif_abs_map = np.where(magnif_abs_map > magnif_cutoff, magnif_cutoff, magnif_abs_map)
    magnif_weights = np.where(magnif_mask != 0, 1. / magnif_abs_map, 0).flatten()
    magnif_weights = np.where(magnif_weights > 1., 0., magnif_weights)  # Eliminate magnification < 1
    magnif_weights /= np.sum(magnif_weights)
    magnif_selections = np.random.choice(magnif_abs_map.flatten(), size=n_lenses, p=magnif_weights)
    with open(output_filename, 'w') as outfile:
        file_header = _snana_documentation
        outfile.write(file_header)
        middle_entry_part = ' 1 ' + str(z_lens) + ' ' + str(z_source) + ' 0.0 0.0 '
        for i in range(n_lenses):
            catalogue_entry = "LENS: " + str(i + id_modifier) + middle_entry_part + str(magnif_selections[i]) + ' 0.0\n'
            outfile.write(catalogue_entry)


def snana_mock_magnif_catalogue(output_filename, z_lens, set_z_method, get_magnif_method, get_magnif_mask_method,
                                z_range, n_lenses_per_z=100, id_modifier=0, magnif_cutoff=None, z_precision=None):
    """Create a mock catalogue of lensing systems as input for SNANA that follows the distribution of magnification."""

    with open(output_filename, 'w') as outfile:
        file_header = _snana_documentation
        outfile.write(file_header)
        for z in z_range:
            # Dealing with unique IDs per z
            z_id_order = math.floor(math.log10(n_lenses_per_z))
            if z_precision is not None:
                z = round(z, z_precision)
                z_modifier = int(z * 10 ** (z_precision + z_id_order))
            else:
                z_modifier = int(z * 10 ** (2 + z_id_order))                    # assume 0.01 z step
            # Now the main part.
            set_z_method(z)
            magnif_abs_map = np.abs(get_magnif_method())
            if magnif_cutoff is not None:  # truncate too large magnifications
                magnif_abs_map = np.where(magnif_abs_map > magnif_cutoff, magnif_cutoff, magnif_abs_map)
            magnif_mask = get_magnif_mask_method()
            magnif_weights = np.where(magnif_mask != 0, 1. / magnif_abs_map, 0).flatten()
            magnif_weights = np.where(magnif_weights > 1., 0., magnif_weights)  # Eliminate magnification < 1
            magnif_weights /= np.sum(magnif_weights)
            magnif_selections = np.random.choice(magnif_abs_map.flatten(), size=n_lenses_per_z, p=magnif_weights)
            middle_entry_part = ' 1 ' + str(z_lens) + ' ' + str(z) + ' 0.0 0.0 '
            for i in range(n_lenses_per_z):
                lens_id = id_modifier + i + z_modifier
                catalogue_entry = "LENS: " + str(lens_id) + middle_entry_part + str(magnif_selections[i]) + ' 0.0\n'
                outfile.write(catalogue_entry)


def snana_lens_catalogue(output_filename, lens_table):
    pass


def read_binned_sn_detection_rate(catalogue_path, field_names=None, field_dict=None):
    """Calculates the SN detection rate per 0.1-wide redshift bin."""
    file_list = os.listdir(catalogue_path)
    z_bins_detected = np.zeros(61, dtype=int)   # Hardcoded 0.1 wide bins from 0.0 to 6.0. TODO: customize this.
    if field_names is not None:
        field_counts_total = {}
        field_counts = {}
        for name in field_names:
            field_counts_total[name] = np.zeros(61, dtype=int)
            field_counts[name] = np.zeros(61, dtype=int)
    for filename in file_list:
        if "HEAD.FITS.gz" in filename:
            head_file_path = catalogue_path + filename
            phot_file_path = catalogue_path + filename.replace("HEAD", "PHOT")
            lightcurves = read_snana_fits(head_file_path, phot_file_path)
            for i in lightcurves:
                # lens_id = i.meta["SIM_STRONGLENS_IDLENS"]     # DUMB WAY TO DO IT DO NOT DO IT LIKE THIS
                # z_times_ten = int(str(lens_id)[-4:-2])
                z_times_ten = int(round(i.meta["REDSHIFT_FINAL"] * 10, 0))
                z_bins_detected[z_times_ten] += 1
                if field_names is not None:
                    field_counts[field_dict[i.meta["SIM_LIBID"]]][z_times_ten] += 1
        elif ".DUMP" in filename:
            dump_file_path = catalogue_path + filename
            dump_file = ascii.read(dump_file_path)
            z_bins_total = np.zeros(61, dtype=int)
            for i in dump_file:
                z_times_ten = int(round(i["ZCMB"] * 10, 0))
                z_bins_total[z_times_ten] += 1
                if field_names is not None:
                    field_counts_total[i["FIELD"]][z_times_ten] += 1
    if field_names is not None:
        return z_bins_detected, z_bins_total, field_counts, field_counts_total
    return z_bins_detected, z_bins_total


def read_specific_sn_detection_rate(catalogue_path, field_names=None, field_dict=None):
    """Calculates detection rate per individual lens."""
    pass

