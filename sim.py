import os
from glob import glob
import numpy as np

from pygrasp.grid import Grid
from pygrasp.jones import JonesMap

# This is a library for dealing specifically with Brad's GRASP simulations.

base = '/Users/daniel/Johnson-Miller Lab/optics'

# 150 GHz.
HF_150 = os.path.join(base, 'HF_focal_plane_150')
maps_150 = os.path.join(HF_150, 'maps')
pics_150 = os.path.join(HF_150, 'pics')

# 225 GHz.
HF_225 = os.path.join(base, 'HF_focal_plane_225')
maps_225 = os.path.join(HF_225, 'maps')
pics_225 = os.path.join(HF_225, 'pics')

def make_grid(folder, feed, key):
    """
    Return a Grid created from the simulation with key 1_A, 1_B, 2_A,
    or 2_B, corresponding to the given feed number in the given folder.
    """
    filename = glob(os.path.join(folder, 'beam_map_*F%s_pol%s.grd' %(feed, key)))
    return Grid(filename[0])

def make_diff_jones_map(folder, feed):
    co = make_grid(folder, feed, '1_A')
    cx = make_grid(folder, feed, '2_A')
    return JonesMap((co, cx))

def make_hwp_jones_map(folder, feed):
    co = make_grid(folder, feed, '1_A')
    cx = make_grid(folder, feed, '1_B')
    return JonesMap((co, cx))

def theta_phi_to_Brad_angles(theta, phi):
    """
    Convert spherical polar angles to Brad's Az/El-like angles in degrees.
    """
    x = np.degrees(theta * np.cos(phi))
    y = np.degrees(theta * np.sin(phi))
    return x, y

def Brad_angles_to_theta_phi(xd, yd):
    """
    Convert Brad's Az/El-like angles in degrees to spherical polar angles.
    """
    xr = np.radians(xd)
    yr = np.radians(yd)
    theta = np.sqrt(xr**2 + yr**2)
    phi = np.arctan2(y/theta, x/theta)
    return theta, phi
