from __future__ import division

import numpy as np

from pygrasp.map import Map
from pygrasp.grid import Grid

class JonesMap(Map):

    # This is the shape of the matrix at each pixel.
    shape = (2, 2)
    # This is the map data type.
    data_type = np.complex

    # This is the mapping between array indices and Jones vector components.
    jones = {0: 'co', 'co': 0,
             1: 'cx', 'cx': 1}

    def __init__(self, co=None, cx=None):
        """
        Create a new empty JonesMap or create one from two Grid instances.
        """
        if co is None and cx is None:
            super(JonesMap, self).__init__(self.shape, self.data_type)
        else:
            assert(isinstance(co, Grid) and isinstance(cx, Grid))            
            assert(all(co.X == cx.X))
            self.X = co.X
            assert(all(co.Y == cx.Y))
            self.Y = co.Y
            # Create a Jones matrix map with
            # map.shape = (2, 2, X.size, Y.size)
            self.map = np.array([[co.map[0, 0], cx.map[0, 0]],
                                 [co.map[1, 0], cx.map[1, 0]]])

    # Not yet tested.
    # Move to Map?
    @classmethod
    def coadd(cls, jones_maps):
        assert all([isinstance(j, JonesMap) for j in jones_maps])
        # Ensure all maps have the same pixel spacing.
        j0 = jones_maps[0]
        # Check pixel spacing.
        assert all([j.dx() == j0.dx() and
                    j.dy() == j0.dy() for j in jones_maps])
        # Find the edges of the new map and its pixelization.
        x_min = min([j.X[0] for j in jones_maps])
        x_max = max([j.X[-1] for j in jones_maps])
        y_min = min([j.Y[0] for j in jones_maps])
        y_max = max([j.Y[-1] for j in jones_maps])
        # Check that this is ideal.
        nx = 1 + int(round((x_max - x_min) / j0.dx()))
        ny = 1 + int(round((y_max - y_min) / j0.dy()))
        coadded = JonesMap()
        coadded.X = np.linspace(x_min, x_max, nx)
        coadded.Y = np.linspace(y_min, y_max, ny)
        coadded.map = np.zeros((JonesMap.shape[0],
                                JonesMap.shape[1],
                                coadded.X.size,
                                coadded.Y.size),
                               dtype=np.complex)
        for j in jones_maps:
            i_x, i_y, within = coadded.indices(j.X, j.Y)
            coadded.map[:, :, i_x[0]:i_x[-1]+1, i_y[0]:i_y[-1]+1] += j.map
        return coadded

    # It's not clear what this means.
    def invert(self):
        map = np.empty((self.map.shape))
        for x in range(self.X.size):
            for y in range(self.Y.size):
                map[:, :, x, y] = np.mat(self.map[:, :, x, y]).getI()
        self.map = map
