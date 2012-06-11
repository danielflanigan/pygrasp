import numpy as np
from scipy import constants

"""
This class converts between GRASP units and SI units,
and between different angle representations.
"""

class Convert(object):

    # Permittivity of free space [F / m].
    epsilon_0 = constants.epsilon_0
    # Permeability of free space [H / m].
    mu_0 = constants.mu_0
    # Impedance of free space [ohm].
    Z_0 = np.sqrt(mu_0 / epsilon_0)
    # Speed of light [m / s].
    c = constants.c

    # Convert electric and magnetic fields.
    @classmethod
    def E_to_SI(cls, E_GRASP, k):
        return k * np.sqrt(2 * cls.Z_0) * E_GRASP

    @classmethod
    def E_to_GRASP(cls, E_SI, k):
        return 1 / (k * np.sqrt(2 * cls.Z_0)) * E_SI

    @classmethod
    def H_to_SI(cls, H_GRASP, k):
        return k * np.sqrt(2 / cls.Z_0) * H_GRASP

    @classmethod
    def H_to_GRASP(cls, H_SI, k):
        return 1 / k * np.sqrt(cls.Z_0 / 2) * H_SI

    # Convert electric and magnetic currents.

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
