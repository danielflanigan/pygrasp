import numpy as np
from scipy import constants

"""
This class converts between GRASP units and SI units.
"""

class Units(object):

    # Permittivity of free space [?]
    epsilon_0 = constants.epsilon_0
    # Permeability of free space [?]
    mu_0 = constants.mu_0
    # Impedance of free space [ohm].
    Z_0 = np.sqrt(mu_0 / epsilon_0)
    # Speed of light [m s^-1].
    c = constants.c

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

    
