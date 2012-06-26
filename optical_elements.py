from __future__ import division

import numpy as np

# The sign of a phase lead or lag depends on the convention used for
# harmonic time dependence. This module uses the GRASP convention with
# time dependence
# exp(1j * omega * t).
# It uses the circular polarization convention used in modern physics
# and engineering, also used by GRASP, that at a fixed point in space
# the electric field vector of positive helicity (right circular)
# light rotates in the right-handed sense with respect to the
# direction of photon propagation.

def linear_polarizer(angle_radians):
    """
    Return the Jones matrix of a linear polarizer oriented at the
    given angle in radians from the positive x-axis.
    """
    pass

def circular_polarizer(positive_helicity):
    if positive_helicity:
        return 1/2 * np.array([[1, 1j],
                               [-1j, 1]])
    else:
        return 1/2 * np.array([[1, -1j],
                               [1j, 1]])

def quarter_wave_plate(fast_axis_radians):
    pass

def half_wave_plate(fast_axis_radians):
    pass
