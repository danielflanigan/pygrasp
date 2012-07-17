from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt

from pygrasp.output import Grid

class FlatMap(object):
    """
    This is an abstract class that contains methods useful for
    maps.
    
    """

    # Subclasses should define these.
    shape = None
    data_type = None
    indices = {}

    def __init__(self):
        self.x = np.array([0.])
        self.y = np.array([0.])
        self.map = np.zeros((self.shape[0], self.shape[1], self.x.size, self.y.size),
                            dtype=self.data_type)

    def dx(self):
        """
        Return the pixel spacing in x.
        """
        return (self.x[-1] - self.x[0]) / (self.x.size - 1)

    def dy(self):
        """
        Return the pixel spacing in y.
        """
        return (self.y[-1] - self.y[0]) / (self.y.size - 1)

    def recenter(self, x, y):
        """
        Set the map center to the new coordinates, preserving all
        other aspects of the pixelization.
        """
        x0 = (self.x[-1] + self.x[0]) / 2
        y0 = (self.y[-1] + self.y[0]) / 2
        self.x += x - x0
        self.y += y - y0

    def swap(self):
        """
        Return a view of self.map with shape
        (x.size, y.size, self.shape[0], self.shape[1]).
        Use this for broadcasting into multiple map points.
        """
        return self.map.swapaxes(0, 2).swapaxes(1, 3)

    def indices(self, x, y, clip=False):
        """
        Return the grid pixel indices (i_x, i_y) corresponding to the
        given grid coordinates. Arrays x and y must have the same
        length. Also, return a boolean array of the same length that
        is True where the pixels are within the grid bounds and False
        elsewhere.
		
        If clip is False, a ValueError is raised if any of the pixel
        centers are outside the grid bounds, and array within will be
        all True. If clip is True, then the i_x and i_y values where
        within is False will be nonsense; the safe thing is to use
        only i_x[within] and i_y[within].
        """
        if x.size != y.size:
            raise ValueError("Arrays x and y must have the same length.")
        # This is a workaround for the behavior of int_: when given an
        # array of size 1 it returns an int instead of an array.
        if x.size == 1:
            i_x = np.array([np.int(np.round((x[0] - self.x[0]) / self.dx()))])
            i_y = np.array([np.int(np.round((y[0] - self.y[0]) / self.dy()))])
        else:
            i_x = np.int_(np.round_((x - self.x[0]) / self.dx()))
            i_y = np.int_(np.round_((y - self.y[0]) / self.dy()))
        within = ((0 <= i_x) & (i_x < self.x.size) & (0 <= i_y) & (i_y < self.y.size))
        if not clip and not all(within):
            raise ValueError("Not all points are inside the grid bounds, and clipping is not allowed.")
        return i_x, i_y, within

    def coordinates(self, x, y):
        """
        Return two arrays (c_x, c_y) containing the pixel center
        coordinates corresponding to the given (x, y) coordinates,
        which must all be within the map bounds.
        """
        i_x, i_y, within = self.indices(x, y, clip=False)
        return self.x[i_x], self.y[i_y]

    def save_npy(self, folder):
        os.mkdir(folder)
        np.save(os.path.join(folder, 'x.npy'), self.x)
        np.save(os.path.join(folder, 'y.npy'), self.y)
        np.save(os.path.join(folder, 'map.npy'), self.map)

    def load_npy(self, folder):
        self.x = np.load(os.path.join(folder, 'x.npy'))
        self.y = np.load(os.path.join(folder, 'y.npy'))
        self.map = np.load(os.path.join(folder, 'map.npy'))
        assert map.shape[:2] == self.shape

    # Switch the plotting methods to return the figure.
    # Work out transposition and extents.
    def make_plot(self, a, title="", xlabel="", ylabel="", color=plt.cm.hot):
        plt.ioff()
        plt.figure()
        plt.imshow(a.T,
                   cmap=color,
                   aspect='equal',			 
                   interpolation='nearest',
                   origin='lower',
                   extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
        plt.colorbar(shrink=0.8, aspect=20*0.8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def show_plot(self, a, title="", xlabel="", ylabel="", color=plt.cm.hot):
        """Create and show a plot of the data or weights map; see make_plot() for usage."""
        self.make_plot(a, title=title, xlabel=xlabel, ylabel=ylabel, color=color)
        plt.ion()
        plt.show()
        
    def save_plot(self, filename, a, title="", xlabel="", ylabel="", color=plt.cm.hot):
        """Create and save a plot of the data or weights map; see make_plot() for usage."""
        interactive = plt.isinteractive()
        self.make_plot(a, title=title, xlabel=xlabel, ylabel=ylabel, color=color)
        plt.savefig(filename)
        if interactive:
            plt.ion()
        else:
            plt.close()

    # Work out transposition and extents.
    def make_contour(self, a, contours=None, title="", xlabel="", ylabel="", color=plt.cm.jet):
        """
        Create a contour plot of the given array a with the map
        shape.
        """
        if contours is None:
            contours = np.linspace(np.min(a.flatten()), np.max(a.flatten()), 10)
        plt.ioff()
        plt.figure()
        plt.contour(a.T,
                    contours,
                    cmap=color,
                    extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
        plt.colorbar(shrink=0.8, aspect=20*0.8, format='%3.3g')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def show_contour(self, a, contours=None, title="", xlabel="", ylabel="", color=plt.cm.jet):
        self.make_contour(a, contours, title, xlabel, ylabel, color)
        plt.ion()
        plt.show()
        
    def save_contour(self, filename, a, contours=None, title="", xlabel="", ylabel="", color=plt.cm.jet):
        interactive = plt.isinteractive()
        self.make_contour(a, contours, title, xlabel, ylabel, color)
        plt.savefig(filename)
        if interactive:
            plt.ion()
        else:
            plt.close()


class GridMap(FlatMap, Grid):
    """
    A FlatMap created from a .grd file. The file handling logic is
    contained in class pygrasp.output.Grid.
    
    This class can be created from near field, far field, and coupling
    .grd files.  Implement subclasses if necessary.
    """
    data_type = np.complex
    # Subclasses should define the meaning of the map indices.

    # Since a generic grid doesn't have a shape, there's no way to
    # make a generic blank.
    def __init__(self, filename):
        self.meta, data = self.load_grd(filename)
        self.x = self.meta['XCEN'] + np.linspace(self.meta['XS'], self.meta['XE'], self.meta['NX'])
        self.y = self.meta['YCEN'] + np.linspace(self.meta['YS'], self.meta['YE'], self.meta['NY'])
        self.shape = (self.meta['NCOMP'], 1)
        self.map = np.array([[(data[:, 2*c] +
                               1j * data[:, 2*c+1]).reshape(self.meta['NX'], self.meta['NY'], order='F')]
                             for c in range(self.shape[0])])

    def save_grd(self, filename):
        points = self.meta['NX'] * self.meta['NY']
        components = self.meta['NCOMP']
        data = np.empty((points, 2 * components))
        for component in range(components):
            data[:, 2*component] = self.map[component, 0].reshape(points, order='F').real
            data[:, 2*component+1] = self.map[component, 0].reshape(points, order='F').imag
        super(GridMap, self).save_grd(filename, self.meta, data)


class JonesMap(FlatMap):

    shape = (2, 2)
    data_type = np.complex
    indices = {0: 'co', 'co': 0,
               1: 'cx', 'cx': 1}

    def __init__(self, co=None, cx=None):
        """
        Create a new empty JonesMap or create one from two Grid instances.
        """
        if co is None and cx is None:
            super(JonesMap, self).__init__()
        else:
            assert(isinstance(co, GridMap) and isinstance(cx, GridMap))
            assert(all(co.X == cx.X))
            self.X = co.X
            assert(all(co.Y == cx.Y))
            self.Y = co.Y
            # Create a Jones matrix map with
            # map.shape = (2, 2, X.size, Y.size)
            self.map = np.array([[co.map[0, 0], cx.map[0, 0]],
                                 [co.map[1, 0], cx.map[1, 0]]])

    # Not yet tested.
    # Move to FlatMap?
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


class MuellerMap(FlatMap):

    # This is the shape of the matrix at each pixel.
    shape = (4, 4)
    # This is the map data type.
    data_type = np.float

    # This is the mapping between array indices and Stokes parameters.
    indices = {0: 'T', 'T': 0,
               1: 'Q', 'Q': 1,
               2: 'U', 'U': 2,
               3: 'V', 'V': 3}

    A = np.mat(np.array([[1,   0,   0,   1],
                         [1,   0,   0,  -1],
                         [0,   1,   1,   0],
                         [0,  1j, -1j,   0]]))
    AI = A.getI()

    def __init__(self, jones_map=None):
        """
        Create a new empty MuellerMap or create one from a JonesMap instance.
        """
        if jones_map is None:
            super(MuellerMap, self).__init__()
        else:
            assert isinstance(jones_map, JonesMap)
            self.X = jones_map.X
            self.Y = jones_map.Y
            J = jones_map.map
            self.map = np.empty((self.shape[0],
                                 self.shape[1],
                                 self.X.size,
                                 self.Y.size),
                                dtype='float')
            for x in range(self.X.size):
                for y in range(self.Y.size):
                    J_xy = np.mat(J[:, :, x, y])
                    # The matrix cast is redundant since numpy takes *
                    # to mean matrix multiplication when either element
                    # is a matrix.
                    M_xy = self.A * np.mat(np.kron(J_xy, J_xy.conj())) * self.AI
                    assert np.all(M_xy.imag == 0), "Nonzero complex value in M."
                    self.map[:, :, x, y] = M_xy.real

    # Figure out what this means.
    def inverse_from_jones(self, jones_map):
        assert isinstance(jones_map, JonesMap)
        self.X = jones_map.X
        self.Y = jones_map.Y
        J = jones_map.map
        self.map = np.empty((4, 4, self.X.size, self.Y.size), dtype='float')
        for x in range(self.X.size):
            for y in range(self.Y.size):
                J_xy = np.mat(J[:, :, x, y])
                M_xy = np.mat(self.A * np.kron(J_xy, J_xy.conj()) * self.AI).getI()
                assert np.all(M_xy.imag == 0), "Nonzero complex value in map."
                self.map[:, :, x, y] = M_xy.real
        
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

    # Figure out how to create a title and axes labels.
    def contour_tile(self, color=None):
        plt.ioff()
        fig = plt.figure(figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                # Verify.
                name = '{}{}'.format(self.stokes[i], self.stokes[j])
                sub = plt.subplot(4, 4, 4 * i + j + 1)
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
                a = self.map[i, j]
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

    def plot_tile(self, color=None):
        plt.ioff()
        fig = plt.figure(figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                name = '{}{}'.format(self.stokes[j], self.stokes[i])
                sub = plt.subplot(4, 4, 4 * i + j + 1)
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
                a = self.map[i, j]
                plt.imshow(a.T,
                           cmap=color,
                           aspect='equal',			 
                           interpolation='nearest',
                           origin='lower')
                sub.title.set_text(name)
        plt.ion()
        plt.show()
        return fig
