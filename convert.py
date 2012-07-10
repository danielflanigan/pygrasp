from __future__ import division

import numpy as np
from scipy import constants

class Convert(object):
    """
    This class converts between GRASP units and SI units,
    and between different angle representations.
    """

    # Permittivity of free space [F / m].
    epsilon_0 = constants.epsilon_0
    # Permeability of free space [H / m].
    mu_0 = constants.mu_0
    # Impedance of free space [ohm].
    Z_0 = np.sqrt(mu_0 / epsilon_0)
    # Speed of light [m / s].
    c = constants.c

    @classmethod
    def k(cls, frequency):
        """
        Return the free space wavenumber k corresponding to the given
        frequency in hertz.
        """
        return 2 * np.pi * frequency / cls.c

    # Convert electric and magnetic fields, and electric and magnetic
    # surface current densities.  The notation follows the GRASP
    # Technical Description, which uses J for surface current density,
    # not areal current density.  The SI units are, using the Weber
    # convention for magnetic current,
    # E: V / m
    # H: A / m
    # J_{e}: A / m
    # J_{m}: V / m
    # All of these quantities have units of W^{1/2} in GRASP.
    @classmethod
    def E_to_SI(cls, E_GRASP, f):
        return cls.k(f) * np.sqrt(2 * cls.Z_0) * E_GRASP

    @classmethod
    def E_to_GRASP(cls, E_SI, f):
        return 1 / (cls.k(f) * np.sqrt(2 * cls.Z_0)) * E_SI

    @classmethod
    def H_to_SI(cls, H_GRASP, f):
        return cls.k(f) * np.sqrt(2 / cls.Z_0) * H_GRASP

    @classmethod
    def H_to_GRASP(cls, H_SI, f):
        return 1 / cls.k(f) * np.sqrt(cls.Z_0 / 2) * H_SI

    @classmethod
    def Je_to_SI(cls, Je_GRASP, f):
        return cls.k(f) * np.sqrt(2 / cls.Z_0) * Je_GRASP

    @classmethod
    def Je_to_GRASP(cls, Je_SI, f):
        return 1 / cls.k(f) * np.sqrt(cls.Z_0 / 2) * Je_SI

    @classmethod
    def Jm_to_SI(cls, Jm_GRASP, f):
        return cls.k(f) * np.sqrt(2 * cls.Z_0) * Jm_GRASP

    @classmethod
    def Jm_to_GRASP(cls, Jm_SI, f):
        return 1 / (cls.k(f) * np.sqrt(2 * cls.Z_0)) * Jm_SI

    # Convert angles and unit vectors.
    @classmethod
    def u_v_to_theta_phi(cls, u, v):
        theta = np.arcsin(np.sqrt(u**2 + v**2))
        phi = np.arctan2(v, u)
        return theta, phi

    @classmethod
    def theta_phi_to_u_v(cls, theta, phi):
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        return u, v
