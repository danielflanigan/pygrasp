import numpy as np
import matplotlib.pyplot as plt

# Tools for handling polarization on the sphere.
# Encode Ludwig-3.
# Rotations using rotation matrices or quaternions.

x_hat = np.array([1., 0., 0.])
y_hat = np.array([0., 1., 0.])
z_hat = np.array([0., 0., 1.])

def unit_theta_phi_r(theta, phi):
    theta_hat = np.array([np.cos(theta) * np.cos(phi),
                          np.cos(theta) * np.sin(phi),
                          -np.sin(theta)])
    phi_hat = np.array([-np.sin(phi),
                        np.cos(phi),
                        0])
    r_hat = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
    return theta_hat, phi_hat, r_hat

def grasp_rotation(theta, phi, psi):
    theta_hat, phi_hat, r_hat = unit_theta_phi_r(theta, phi)
    x1_hat = theta_hat * np.cos(phi - psi) - phi_hat * np.sin(phi - psi)
    y1_hat = theta_hat * np.sin(phi - psi) + phi_hat * np.cos(phi - psi)
    z1_hat = r_hat
    return x1_hat, y1_hat, z1_hat

def ludwig_3_unit_co_cx_r(theta, phi):
    theta_hat, phi_hat, r_hat = unit_theta_phi_r(theta, phi)
    co_hat = theta_hat * np.cos(phi) - phi_hat * np.sin(phi)
    cx_hat = theta_hat * np.sin(phi) + phi_hat * np.cos(phi)
    return co_hat, cx_hat, r_hat
