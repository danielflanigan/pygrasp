# To do:
# -rewrite fitting routines for TimeOrderedDataMap, when this exists.
# -write functions to fit without noise.

from __future__ import division

import numpy as np
from scipy.signal import convolve
from scipy.optimize import leastsq

def smooth(fm, fwhm):
    spacing = fm.bounds()[4]
    n = np.ceil(fwhm/spacing)
    x = y = np.linspace(-n * spacing, n * spacing, 2 * n + 1)
    g = circular_gaussian(x, y, 0, 0, 1, fwhm)
    g /= g.sum()
    return convolve(fm.I, g, mode='same')

def circular_gaussian(x, y, x0, y0, A, fwhm):
    xx, yy = np.meshgrid(x, y)
    xx = xx.T
    yy = yy.T
    return A * np.exp(-np.log(2) * ((xx - x0)**2 + (yy - y0)**2) / (fwhm/2)**2)
                      
def circular_gaussian_plus_noise(x, y, x0, y0, A, fwhm, N):
    return circular_gaussian(x, y, x0, y0, A, fwhm) + N

# Figure out how to obtain convergence without a good guess for the FWHM.
def fit_circular_gaussian(fm, fwhm, smooth_first=False, x0=None, y0=None, A=None, N=0):
    def error(params):
        return (fm.I_weight *(fm.I -
                              circular_gaussian_plus_noise(fm.x, fm.y, *params))).flatten()
    #if fwhm is None:
    #    fwhm = np.sqrt(((fm.x[-1]-fm.x[0])/10)**2 +
    #                   ((fm.y[-1]-fm.y[0])/10)**2)
    if smooth_first:
        guess_map = smooth(fm, fwhm)
    else:
        guess_map = fm.I
    center = np.where(guess_map == np.max(guess_map))
    if x0 is None:
        x0 = fm.x[center[0][0]]
    if y0 is None:
        y0 = fm.y[center[1][0]]
    if A is None:
        A = np.max(guess_map)
    guess = (x0, y0, A, fwhm, N)
    params, cov = leastsq(error, guess)
    return params

def elliptical_gaussian(x, y, x0, y0, A, fwhm_u, fwhm_v, psi):
    """
    Return an (x.size, y.size) array representing a 2D gaussian with
    the given parameters evaluated on the given (x, y) points. The u-axis forms an angle psi with the x-axis.  
    """
    xx, yy = np.meshgrid(x, y)
    xx = xx.T.flatten()
    yy = yy.T.flatten()
    R_psi = np.array([[np.cos(psi), np.sin(psi)],
                      [-np.sin(psi), np.cos(psi)]])
    rot_xx, rot_yy = np.vsplit(np.dot(R_psi, np.vstack((xx - x0, yy - y0))), 1)[0]
    rot_xx = rot_xx.reshape((x.size, y.size))
    rot_yy = rot_yy.reshape((x.size, y.size))
    return A * np.exp(-np.log(2) * (rot_xx**2 / (fwhm_u/2)**2 +
                                    rot_yy**2 / (fwhm_v/2)**2))

def elliptical_gaussian_plus_noise(x, y, x0, y0, A, fwhm_u, fwhm_v, psi, N):
    return elliptical_gaussian(x, y, x0, y0, A, fwhm_u, fwhm_v, psi) + N

def fit_elliptical_gaussian(fm, fwhm, smooth_first=False, x0=None, y0=None, A=None, N=0):
    def error(params):
        return (fm.I_weight *(fm.I -
                              elliptical_gaussian_plus_noise(fm.x, fm.y, *params))).flatten()
    if smooth_first:
        guess_map = smooth(fm, fwhm)
    else:
        guess_map = fm.I
    center = np.where(guess_map == np.max(guess_map))
    if x0 is None:
        x0 = fm.x[center[0][0]]
    if y0 is None:
        y0 = fm.y[center[1][0]]
    if A is None:
        A = np.max(guess_map)
    guess = (x0, y0, A, fwhm, fwhm, 0, N)
    params, cov = leastsq(error, guess)
    return params

def clean_elliptical_params(params):
    x0, y0, A, fwhm_u, fwhm_v, psi, N = params
    if fwhm_u >= fwhm_v:
        return x0, y0, A, fwhm_u, fwhm_v, np.mod(psi, np.pi), N
    else:
        return x0, y0, A, fwhm_v, fwhm_u, np.mod(psi+np.pi/2, np.pi), N
