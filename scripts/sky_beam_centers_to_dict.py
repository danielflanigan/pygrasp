#! /usr/bin/env python

"""
Convert beam centers in degrees from Brad's sky_beam_centers.pro
format to a pickled dictionary containing (u, v) coordinates for all
frequencies.
"""

import sys
import pickle
import numpy as np
from pygrasp.convert import Convert

filename = sys.argv[1]
d = {}
with open(filename) as f:
    try:
        while True:
            strings = [s.strip(":").strip("'").strip(",").strip("]")
                       for s in f.next().split()]
            frequency, feed = tuple(strings[0].split("F"))
            frequency = int(frequency)
            feed = int(feed)
            # Convert degrees to radians. These values actually
            # approximate (u, v), for small angles, and are not angles
            # themselves.
            x = np.radians(float(strings[4]))
            y = np.radians(float(strings[5]))
            # x = theta * cos(phi)
            # y = theta * sin(phi)
            theta = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y/theta, x/theta)
            u, v = Convert.theta_phi_to_u_v(theta, phi)
            if not frequency in d:
                d[frequency] = {}
            d[frequency][feed] = (u, v)
    except StopIteration:
        pass

pickle.dump(d, open("%s.pkl" %filename, 'w'))
