import sys
import numpy as np
from copy import deepcopy

from pygrasp.grid import Grid
from pygrasp.jones import JonesMap
from pygrasp.mueller import MuellerMap

co = Grid(sys.argv[1])
cx = Grid(sys.argv[2])

co.X = np.degrees(co.X)
co.Y = np.degrees(co.Y)
cx.X = np.degrees(cx.X)
cx.Y = np.degrees(cx.Y)

j = JonesMap((co, cx))
m = MuellerMap(j)

cor = deepcopy(co)
cxr = deepcopy(cx)
cor.E_co = -(cor.E_co)
cxr.E_co = -(cxr.E_co)
jr = JonesMap((cor, cxr))
mr = MuellerMap(jr)
