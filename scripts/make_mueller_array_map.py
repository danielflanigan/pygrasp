import os
import pickle
import numpy as np

from pygrasp.convert import Convert
from pygrasp.mueller import MuellerMap
from pygrasp import sim 

# Pair grid files in the differencing or HWP configuration to make
# Jones maps and hence Mueller maps for each beam. Coadd these beams
# into a MuellerMap.

# The feeds in the HF focal plane are numbered 1 to 310.
mueller_maps = []
beam_centers = pickle.load(open(os.path.join(sim.base, 'sky_beam_centers.pkl')))
feeds = np.arange(1, 311)
#feeds = np.arange(1, 311, 10)
#feeds = [1, 2]
for feed in feeds:
    print("Creating a Mueller matrix map for feed %s." %(feed))
    #j = sim.make_diff_jones_map(sim.maps_150, feed)
    #j = sim.make_hwp_jones_map(sim.maps_150, feed)
    #j = sim.make_diff_jones_map(sim.maps_225, feed)
    j = sim.make_hwp_jones_map(sim.maps_225, feed)
    # Calculate the beam center location in pseudo-degrees.
    #u, v = beam_centers[150][feed]
    u, v = beam_centers[225][feed]
    theta, phi = Convert.u_v_to_theta_phi(u, v)
    # These are essentially az-el in degrees, near (0, 0).
    x, y = sim.theta_phi_to_Brad_angles(theta, phi)
    # Convert the units of the maps to degrees.
    j.X = np.degrees(j.X)
    j.Y = np.degrees(j.Y)
    # In the old pixelization, find the center of the pixel closest to the new center.
    x0 = j.dx() * round(x / j.dx())
    y0 = j.dy() * round(y / j.dy())
    # Recenter the map on the center of the pixel in the old
    # pixelization. This ensures that all maps share the same
    # pixelization.
    j.recenter(x0, y0)
    m = MuellerMap(j)
    mueller_maps.append(m)

print("Coadding %s feeds." %(len(feeds)))
array_map = MuellerMap.coadd(mueller_maps)

folder = os.path.join(sim.HF_225, 'array')
print("Saving array map to %s" %(folder))
array_map.save(folder, 'hwp225')
