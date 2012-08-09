"""
This module contains functions used for analyzing beam maps and cuts.
"""
from __future__ import division

# Improve using interpolation.
def FWHM(x, y_dB):
    """
    Return the full-width at half-maximum of the given peaked function
    y_dB using the independent coordinate x.

    This implementation may fail if y_dB passes below -3 dB more than
    once on each side of the peak, but such a function doesn't have an
    unequivocal FWHM. Otherwise, the upper bound on its error equals
    the spacing between x values, since it may be off by half this
    spacing on each side.
    """
    half_dB = -10 * np.log10(2) # = -3.0103
    i_peak = np.argmax(y_dB)
    right = y_dB[i_peak:]
    # Swap the order of left so that argmin finds the instance closest
    # to the peak. This is paranoid.
    left = y_dB[i_peak::-1]
    i_right = np.argmin(np.abs(right - half_dB))
    i_left = np.argmin(np.abs(left - half_dB))
    fwhm = x[i_peak + i_right] - x[i_peak - i_left]
    return fwhm
