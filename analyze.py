"""
This module contains functions used for analyzing beam maps and cuts.
"""
from __future__ import division

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift
from scipy.signal import get_window
from scipy.optimize import leastsq

from pygrasp.flat_map import JonesMap, MuellerMap
from pygrasp.mapping import circular_gaussian, elliptical_gaussian

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

def B_ell(flat_map, component, bin_size, window=None):
    """
    Return the binned beam function of the given flat map component,
    using ell bins of bin_size. If a window name is given, multiply
    the map component by the window function before taking the 2-D
    DFT.

    The returned beam function is the absolute value of the DFT, so it
    has the same units as the map component. The returned bins array
    contains the left edges of the bins corresponding to the returned
    data: for all i in range(bins.size),
    bins[i] <= data[i] < bins[i+1]
    """
    ell_x, ell_y= np.meshgrid(fftshift(fftfreq(flat_map.x.size, flat_map.dx()/(2*np.pi))),
                              fftshift(fftfreq(flat_map.y.size, flat_map.dy()/(2*np.pi))))
    ell_x = ell_x.T
    ell_y = ell_y.T
    ell_r = np.sqrt(ell_x**2 + ell_y**2)
    ell_bins = np.arange(0, np.max(ell_r), bin_size)
    beam = flat_map[component]
    if window is not None:
        beam *= get_window(window, flat_map.y.size)
        beam *= get_window(window, flat_map.x.size)[:, np.newaxis]
    dft = fftshift(fft2(beam))
    # Shift the indices down by 1 so that they correspond to the data
    # indices. These indices should always be valid indices for the
    # DFT data because r_ell has a lower bound at 0 and max(r_ell)
    # will have index ell_bins.size.
    bin_indices = np.digitize(ell_r.flatten(), ell_bins) - 1
    binned = np.zeros(ell_bins.size) # dtype?
    for i in range(binned.size):
        binned[i] = np.sqrt(np.mean(abs(dft.flatten()[i==bin_indices])**2))
    return ell_bins, binned, dft

# Break this out into individual functions.  We are running sims using
# unit vector components, but we want to express results in
# arcminutes. The beam power is normalized in unit vector components,
# so we have to do the integration in these units.
def characterize_beam(u, v):
    """
    Return a JonesMap, a MuellerMap, and a dictionary with
    characterization data created from the two given feed maps. The
    feeds should be normalized to radiate 4 pi W. All angles are given
    in map units. The JonesMap and MuellerMap are both recentered
    using a Gaussian fit to the Mueller TT beam so that the peak of
    the fit is the (0, 0) point of the map.
    """
    # If the feed maps u and v are normalized to radiate 4 pi W total,
    # then the JonesMap will be normalized to 1 W total per feed.
    jones = JonesMap(u, v, normalize=True)
    mueller = MuellerMap(jones)
    meta = {}
    meta['u'] = {'fit': fit_elliptical_gaussian(abs(u[0])**2, u.x, u.y)}
    meta['v'] = {'fit': fit_elliptical_gaussian(abs(v[1])**2, v.x, v.y)}
    meta['differential pointing'] = np.sqrt((meta['u']['fit'][0] -
                                             meta['v']['fit'][0])**2 +
                                            (meta['u']['fit'][1] -
                                             meta['v']['fit'][1])**2)
    # If a is the major axis and b is the minor axis, then the
    # eccentricity is
    # sqrt(1 - (b/a)**2),
    # and the ellipticity is
    # (a - b) / a.
    # Check if this is right - found multiple definitions.
    meta['u']['eccentricity'] = np.sqrt(1 - (meta['u']['fit'][4] /
                                             meta['u']['fit'][3])**2)
    meta['v']['eccentricity'] = np.sqrt(1 - (meta['v']['fit'][4] /
                                             meta['v']['fit'][3])**2)
    # The directivity is
    # D = U / (P / 4 pi),
    # where U is the radiated power density and P is the total power.
    meta['u']['directivity'] = 10 * np.log10(np.max(abs(u[0])**2))
    meta['v']['directivity'] = 10 * np.log10(np.max(abs(v[1])**2))
    # The total power is 4 pi so the fractional power is the integral
    # of abs(E_cx)**2 / (4 pi), which is equivalent to the Jones map
    # normalization.
    #
    # NOTE: since earlier code did not copy the feed maps to produce
    # the JonesMap, all results before 2013-02-24 for the following 6
    # parameters are suspect due to a possible extra division by 4 pi.
    meta['u']['integrated co-pol'] = u.integrate(abs(u[0])**2) / (4 * np.pi)
    meta['u']['integrated cross-pol'] = u.integrate(abs(u[1])**2) / (4 * np.pi)
    meta['v']['integrated co-pol'] = v.integrate(abs(v[1])**2) / (4 * np.pi)
    meta['v']['integrated cross-pol'] = v.integrate(abs(v[0])**2) / (4 * np.pi)
    meta['u']['spillover'] = 1 - (meta['u']['integrated co-pol'] +
                                  meta['u']['integrated cross-pol'])
    meta['v']['spillover'] = 1 - (meta['v']['integrated co-pol'] +
                                  meta['v']['integrated cross-pol'])
    # Characterize the Mueller map.
    meta['TT'] = {'fit': fit_elliptical_gaussian(mueller['TT'], mueller.x, mueller.y)}
    meta['TT']['eccentricity'] = np.sqrt(1 - (meta['TT']['fit'][4] /
                                              meta['TT']['fit'][3])**2)
    meta['TT']['spillover'] = 1 - mueller.integrate('TT')
    jones.recenter(*meta['TT']['fit'][:2])
    mueller.recenter(*meta['TT']['fit'][:2])
    xx, yy = np.meshgrid(mueller.x, mueller.y)
    rr = np.sqrt(xx.T**2 + yy.T**2)
    # This is the minimum distance from the origin to the edge of the
    # map, which equals the radius of the largest circle centered at
    # the origin that can be fully contained in the map.
    r_circle = min([abs(v) for v in (mueller.x[0], mueller.x[-1],
                                     mueller.y[0], mueller.y[-1])])
    # With numpy 1.7 use np.sum(mueller.map, axis=(0, 1)).
    meta['total sums'] = np.empty(mueller.shape)
    meta['circle sums'] = np.empty(mueller.shape)
    meta['total integrals'] = np.empty(mueller.shape)
    meta['circle integrals'] = np.empty(mueller.shape)
    for i in range(mueller.shape[0]):
        for j in range(mueller.shape[1]):
            meta['total sums'][i, j] = np.sum(mueller[i, j] *
                                              mueller.dx() * mueller.dy())
            meta['circle sums'][i, j] = np.sum(mueller[i, j] * (rr <= r_circle)
                                               * mueller.dx() * mueller.dy())
            meta['total integrals'][i, j] = mueller.integrate((i, j))
            meta['circle integrals'][i, j] = mueller.integrate(mueller[i, j] *
                                                               (rr <= r_circle))
    return jones, mueller, meta

def fit_circular_gaussian(data, x, y):
    def error(params):
        return (data - circular_gaussian(x, y, *params)).flatten()
    center = np.where(data == np.max(data))
    x0 = x[center[0][0]]
    y0 = y[center[1][0]]
    A = np.max(data)
    fwhm = 1
    initial = (x0, y0, A, fwhm)
    params, cov = leastsq(error, initial, xtol=1e-10)
    return params

def fit_elliptical_gaussian(data, x, y):
    def error(params):
        return (data - elliptical_gaussian(x, y, *params)).flatten()
    center = np.where(data == np.max(data))
    initial = (x[center[0][0]], y[center[1][0]], np.max(data),
               (x[-1]-x[0])/3, (y[-1]-y[0])/3, 0)
    params, cov = leastsq(error, initial, xtol=1e-10)
    # For some reason, leastsq tends to converge on values of |psi| >> 2 pi.
    x0, y0, A, fwhm_u, fwhm_v, psi = params
    if fwhm_u >= fwhm_v:
        return x0, y0, A, fwhm_u, fwhm_v, np.mod(psi, np.pi)
    else:
        return x0, y0, A, fwhm_v, fwhm_u, np.mod(psi+np.pi/2, np.pi)
