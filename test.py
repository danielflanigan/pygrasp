import os
import pickle
from copy import deepcopy

from pygrasp.jones import JonesMap
from pygrasp.mueller import MuellerMap
from pygrasp import sim

beam_centers = os.path.join(sim.base, 'sky_beam_centers.pkl')
beam_centers = pickle.load(open(beam_centers))

# This is a central feed at the minimum radius 0.0025 in beam_centers units (150 GHz).
j10hwp = sim.make_hwp_jones_map(sim.maps_150, 10)
m10hwp = MuellerMap(j10hwp)

j10hwpr = JonesMap()
j10hwpr.X = j10hwp.X
j10hwpr.Y = j10hwp.Y
j10hwpr.J = np.array([[j10hwp.J[0, 0], -j10hwp.J[0, 1]],
                      [-j10hwp.J[1, 0], j10hwp.J[1, 1]]])
m10hwpr=MuellerMap(j10hwpr)

# This is an outer feed at radius 0.049 in beam_centers units (150 GHz).
"""
jones310 = JonesMap(os.path.join(maps225, 'beam_map_225F310_pol1_A.grd'),
                    os.path.join(maps225, 'beam_map_225F310_pol1_B.grd'))
mueller310 = MuellerMap(jones310)
"""

"""
names = {0: 'T', 1: 'Q', 2: 'U', 3: 'V'}
for i in range(4):
    for j in range(4):
        name = '%s%s' %(names[j], names[i])
        mueller10.save_plot(os.path.join(pics225, 'feed_10_%s.png' %name),
                            mueller10.M[:, :, i, j],
                            title='225 GHz feed 10 %s at \n %s' %(name, beam_centers[225][10]))
        mueller310.save_plot(os.path.join(pics225, 'feed_310_%s.png' %name),
                            mueller310.M[:, :, i, j],
                            title='225 GHz feed 310 %s at \n %s' %(name, beam_centers[225][310]))
"""
