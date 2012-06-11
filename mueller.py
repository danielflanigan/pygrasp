from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt

from pygrasp.map import Map
from pygrasp.jones import JonesMap

class MuellerMap(Map):

    A = np.mat(np.array([[1,   0,   0,   1],
                         [1,   0,   0,  -1],
                         [0,   1,   1,   0],
                         [0,  1j, -1j,   0]]))
    AI = A.getI()

    # This is the mapping between array indices and Stokes parameters.
    stokes = {0: 'T', 'T': 0,
              1: 'Q', 'Q': 1,
              2: 'U', 'U': 2,
              3: 'V', 'V': 3}

    def __init__(self, jones_map=None):
        """
        Create a new empty MuellerMap or create one from a JonesMap instance.
        """
        if jones_map is None:
            super(MuellerMap, self).__init__()
            self.M = np.array([0])
        else:
            assert isinstance(jones_map, JonesMap)
            self.X = jones_map.X
            self.Y = jones_map.Y
            J = jones_map.J
            self.M = np.empty((4, 4, self.X.size, self.Y.size), dtype='float')
            for x in range(self.X.size):
                for y in range(self.Y.size):
                    Jxy = np.mat(J[:, :, x, y])
                    Mxy = self.A * np.kron(Jxy, Jxy.conj()) * self.AI
                    assert np.all(Mxy.imag == 0), "Nonzero complex value in M."
                    self.M[:, :, x, y] = Mxy.real

    # This isn't what we want.
    def inverse_from_jones(self, jones_map):
        assert isinstance(jones_map, JonesMap)
        self.X = jones_map.X
        self.Y = jones_map.Y
        J = jones_map.J
        self.M = np.empty((4, 4, self.X.size, self.Y.size), dtype='float')
        for x in range(self.X.size):
            for y in range(self.Y.size):
                Jxy = np.mat(J[:, :, x, y])
                Mxy = np.mat(self.A * np.kron(Jxy, Jxy.conj()) * self.AI).getI()
                assert np.all(Mxy.imag == 0), "Nonzero complex value in M."
                self.M[:, :, x, y] = Mxy.real
        
    @classmethod
    def coadd(cls, mueller_maps):
        assert all([isinstance(m, MuellerMap) for m in mueller_maps])
        # Check pixel spacing; np.spacing(1) is the difference between
        # 1 and the next float, or about 2.2e-16 on my machine.
        m0 = mueller_maps[0]
        assert all([abs(m.dx() - m0.dx()) < np.spacing(1) and
                    abs(m.dy() - m0.dy()) < np.spacing(1) for m in mueller_maps])
        # Find the edges of the new map and its pixelization.
        x_min = min([m.X[0] for m in mueller_maps])
        x_max = max([m.X[-1] for m in mueller_maps])
        y_min = min([m.Y[0] for m in mueller_maps])
        y_max = max([m.Y[-1] for m in mueller_maps])
        # Check that this is ideal.
        nx = 1 + int(round((x_max - x_min) / m0.dx()))
        ny = 1 + int(round((y_max - y_min) / m0.dy()))
        coadded = MuellerMap()
        coadded.X = np.linspace(x_min, x_max, nx)
        coadded.Y = np.linspace(y_min, y_max, ny)
        coadded.M = np.zeros((4, 4, coadded.X.size, coadded.Y.size))
        for m in mueller_maps:
            i_x, i_y, within = coadded.indices(m.X, m.Y)
            coadded.M[:, :, i_x[0]:i_x[-1]+1, i_y[0]:i_y[-1]+1] += m.M
        return coadded

    # Move these methods to Map.
    def save(self, folder, name):
        np.save(os.path.join(folder, '%s.X.npy' %(name)), self.X)
        np.save(os.path.join(folder, '%s.Y.npy' %(name)), self.Y)
        np.save(os.path.join(folder, '%s.M.npy' %(name)), self.M)

    def load(self, folder, name):
        self.X = np.load(os.path.join(folder, '%s.X.npy' %(name)))
        self.Y = np.load(os.path.join(folder, '%s.Y.npy' %(name)))
        self.M = np.load(os.path.join(folder, '%s.M.npy' %(name)))

    def show_tile(self, color=None):
        plt.ioff()
        fig = plt.figure()
        for i in range(4):
            for j in range(4):
                name = '%s%s' %(self.stokes[j], self.stokes[i])
                sub = plt.subplot(4, 4, 4 * i + j + 1)
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
                a = self.M[i, j]
                c=np.linspace(np.min(a.flatten()),
                              np.max(a.flatten()),
                              8)
                if all(c == 0):
                    plt.plot()
                else:
                    plt.contour(a.T,
                                contours=c,
                                cmap=color)
                sub.title.set_text(name)
        plt.ion()
        plt.show()
        return fig
