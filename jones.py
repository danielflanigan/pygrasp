from __future__ import division

import numpy as np

from pygrasp.map import Map
from pygrasp.grid import Grid

class JonesMap(Map):

    def __init__(self, co_and_cx=None):
        """
        Create a new empty JonesMap or create one from two Grid instances.
        """
        if co_and_cx is None:
            super(JonesMap, self).__init__()
            self.J = np.array([0])
        else:
            co = co_and_cx[0]
            cx = co_and_cx[1]
            assert(isinstance(co, Grid) and isinstance(cx, Grid))            
            assert(all(co.X == cx.X))
            self.X = co.X
            assert(all(co.Y == cx.Y))
            self.Y = co.Y

            # Create a Jones matrix map with
            # J.shape = (2, 2, X.size, Y.size)
            self.J = np.array([[co.E_co, cx.E_co],
                               [co.E_cx, cx.E_cx]])

    def invert(self):
        J = np.empty((self.J.shape))
        for x in range(self.X.size):
            for y in range(self.Y.size):
                J[:, :, x, y] = np.mat(self.J[:, :, x, y]).getI()
        self.J = J

    # Not yet tested.
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
        coadded.J = np.zeros((2, 2, coadded.X.size, coadded.Y.size))
        for j in jones_maps:
            i_x, i_y, within = coadded.indices(j.X, j.Y)
            coadded.J[:, :, i_x[0]:i_x[-1]+1, i_y[0]:i_y[-1]+1] += j.J
        return coadded

    # Move these methods to Map.
    def save(self, folder, name):
        np.save(os.path.join(folder, '%s.X.npy' %(name)), self.X)
        np.save(os.path.join(folder, '%s.Y.npy' %(name)), self.Y)
        np.save(os.path.join(folder, '%s.J.npy' %(name)), self.J)

    def load(self, folder, name):
        self.X = np.load(os.path.join(folder, '%s.X.npy' %(name)))
        self.Y = np.load(os.path.join(folder, '%s.Y.npy' %(name)))
        self.M = np.load(os.path.join(folder, '%s.J.npy' %(name)))
