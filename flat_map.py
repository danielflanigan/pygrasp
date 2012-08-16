from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt

from pygrasp.output import load_grd, save_grd

class FlatMap(object):
    """
    This is an abstract class that contains methods useful for
    maps.
    
    """

    # Subclasses should define these.
    shape = ()
    data_type = None
    key = {}

    def __init__(self):
        self.x = np.array([0.])
        self.y = np.array([0.])
        self.map = np.zeros(self.shape + (self.x.size, self.y.size),
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

    def indices(self, x, y, clip=False):
        """
        Return the grid pixel indices (i_x, i_y) corresponding to the
        given arrays of grid coordinates. Arrays x and y must have the
        same size. Also return a boolean array of the same length that
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

    def single_indices(self, x, y):
        """
        Return the grid pixel indices (i_x, i_y) corresponding to the
        given grid coordinates, where x and y are numbers.
        same size. Also return a boolean that
        is True if the point pixels are within the grid bounds and False
        elsewhere.
        """
        i_x = int(round((x - self.x[0]) / self.dx()))
        i_y = int(round((y - self.y[0]) / self.dy()))
        if not ((0 <= i_x) & (i_x < self.x.size) & (0 <= i_y) & (i_y < self.y.size)):
            raise ValueError("The point is not inside the grid bounds.")
        return i_x, i_y

    def coordinates(self, x, y):
        """
        Return two arrays (c_x, c_y) containing the pixel center
        coordinates corresponding to the given (x, y) coordinates,
        which must all be within the map bounds.
        """
        i_x, i_y, within = self.indices(x, y, clip=False)
        return self.x[i_x], self.y[i_y]

    def single_coordinates(self, x, y):
        """
        Return two numbers (c_x, c_y) representing the pixel center
        coordinates corresponding to the given (x, y)
        coordinates. Raise a ValueError if the given point is not
        within the map bounds.
        """
        i_x, i_y = self.single_indices(x, y)
        return self.x[i_x], self.y[i_y]

    def save_npy(self, folder):
        """
        Save x.npy, y.npy, and map.npy arrays to the given folder, which must not exist.
        """
        os.mkdir(folder)
        np.save(os.path.join(folder, 'map.npy'), self.map)
        np.save(os.path.join(folder, 'x.npy'), self.x)
        np.save(os.path.join(folder, 'y.npy'), self.y)

    def load_npy(self, folder):
        """
        Return this instance after loading x.npy, y.npy, and map.npy
        arrays from the given folder. This allows, for example,
        mueller = MuellerMap().load_npy('/saved/mueller/map/folder')
        """
        map = np.load(os.path.join(folder, 'map.npy'))
        if map.shape[:2] != self.shape:
            raise ValueError("Array shape {} does not match map shape {}.".format(map.shape, self.shape))
        if map.dtype != self.data_type:
            raise ValueError("Array data type {} does not match map data type {}.".format(map.dtype, self.data_type))
        self.map = map
        self.x = np.load(os.path.join(folder, 'x.npy'))
        self.y = np.load(os.path.join(folder, 'y.npy'))
        return self

    @classmethod
    def coadd(cls, maps):
        # Check pixel spacing; np.spacing(1) is the difference between
        # 1 and the next float, or about 2.2e-16 on my machine.
        tolerance = np.spacing(1)
        m0 = maps[0]
        if not all([abs(m.dx() - m0.dx()) < tolerance and
                    abs(m.dy() - m0.dy()) < tolerance for m in maps]):
            raise ValueError("Cannot coadd maps with different pixel spacings.")
        # Find the edges of the new map and its pixelization.
        x_min = min([m.x[0] for m in maps])
        x_max = max([m.x[-1] for m in maps])
        y_min = min([m.y[0] for m in maps])
        y_max = max([m.y[-1] for m in maps])
        # Check that this is ideal.
        nx = 1 + int(round((x_max - x_min) / m0.dx()))
        ny = 1 + int(round((y_max - y_min) / m0.dy()))
        coadded = cls()
        coadded.x = np.linspace(x_min, x_max, nx)
        coadded.y = np.linspace(y_min, y_max, ny)
        coadded.map = np.zeros((cls.shape[0], cls.shape[1], nx, ny),
                               dtype=cls.data_type)
        for m in maps:
            i_x, i_y, within = coadded.indices(m.x, m.y)
            coadded.map[:, :, i_x[0]:i_x[-1]+1, i_y[0]:i_y[-1]+1] += m.map
        return coadded

    def make_plot(self, a, title="", xlabel="", ylabel="", color=plt.cm.hot, vmin=None, vmax=None):
        """
        Return a plot of the given array with horizontal axis self.x
        and vertical axis self.y. The array is transposed so that the
        first axis is horizontal and the second axis is vertical. The
        [0, 0] element of the array is in the lower left corner.
        """
        if vmin is None:
            vmin = np.min(a)
        if vmax is None:
            vmax = np.max(a)
        plt.ioff()
        fig = plt.figure()
        plt.imshow(a.T,
                   cmap=color,
                   aspect='equal',			 
                   interpolation='nearest',
                   vmin=vmin,
                   vmax=vmax,
                   origin='lower',
                   extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig

    def show_plot(self, a, title="", xlabel="", ylabel="", color=plt.cm.hot, vmin=None, vmax=None):
        """
        Display and return a plot of the given array; see make_plot()
        for usage.
        """
        fig = self.make_plot(a, title, xlabel, ylabel, color, vmin, vmax)
        plt.ion()
        plt.show()
        return fig
        
    def save_plot(self, filename, a, title="", xlabel="", ylabel="", color=plt.cm.hot, vmin=None, vmax=None):
        """
        Save a plot of the given array; see make_plot() for usage.
        """
        interactive = plt.isinteractive()
        fig = self.make_plot(a, title, xlabel, ylabel, color, vmin, vmax)
        plt.savefig(filename)
        if interactive:
            plt.ion()
            return fig
        else:
            plt.close()

    def make_contour(self, a, contours=None, title="", xlabel="", ylabel="", color=plt.cm.jet):
        """
        Return a contour plot of the given array with horizontal axis
        self.x and vertical axis self.y. The array is transposed so
        that the first axis is horizontal and the second axis is
        vertical. The [0, 0] element of the array is in the lower left
        corner.
        """
        if contours is None:
            contours = np.linspace(np.min(a.flatten()), np.max(a.flatten()), 10)
        plt.ioff()
        fig = plt.figure()
        plt.contour(a.T,
                    contours,
                    cmap=color,
                    extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))
        plt.colorbar(format='%3.3g')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig

    def show_contour(self, a, contours=None, title="", xlabel="", ylabel="", color=plt.cm.jet):
        """
        Display and return a contour plot of the given array; see
        make_plot() for usage.
        """
        fig = self.make_contour(a, contours, title, xlabel, ylabel, color)
        plt.ion()
        plt.show()
        return fig
        
    def save_contour(self, filename, a, contours=None, title="", xlabel="", ylabel="", color=plt.cm.jet):
        """
        Save a contour plot of the given array; see make_plot() for usage.
        """
        interactive = plt.isinteractive()
        fig = self.make_contour(a, contours, title, xlabel, ylabel, color)
        plt.savefig(filename)
        if interactive:
            plt.ion()
            return fig
        else:
            plt.close()

    def cut(self, a, angle, center=(0, 0), single_sided=False):
        angle = np.mod(angle, 2 * np.pi)
        # This shifts the line slightly, but ensures that the center
        # pixel is always part of the cut.
        x0, y0 = self.single_coordinates(*center)
        if (np.pi / 4 < angle < 3 * np.pi / 4 or
            5 * np.pi / 4 < angle < 7 * np.pi / 4):
            parity = np.sign(np.sin(angle))
            y = self.y[::parity]
            nonnegative = parity * (y - y0) >= 0
            # There is no vectorized cotangent.
            x = x0 + (y - y0) * np.cos(angle) / np.sin(angle)
        else:
            parity = np.sign(np.cos(angle))
            x = self.x[::parity]
            nonnegative = parity * (x - x0) >= 0
            y = y0 + (x - x0) * np.tan(angle)
        i_x, i_y, within = self.indices(x, y, clip=True)
        i_x = i_x[within]
        i_y = i_y[within]
        nonnegative = nonnegative[within]
        r = np.sqrt((self.x[i_x]-x0)**2 + (self.y[i_y]-y0)**2) * np.where(nonnegative, 1, -1)
        cut = a[i_x, i_y]
        if single_sided:
            return r[nonnegative], cut[nonnegative]
        else:
            return r, cut


class GridMap(FlatMap):
    """
    A FlatMap created from a .grd file. The file handling logic is
    contained in pygrasp.output.
    
    This class can load near field, far field, and coupling .grd
    files.  It currently cannot load elliptically truncated
    grids. Implement subclasses if necessary.
    """
    data_type = np.complex
    # Subclasses should define the key to the map indices.

    def __init__(self, filename=None):
        if filename is None:
            self.shape = (1, 1)
            super(GridMap, self).__init__()
        else:
            self.load_grd(filename)

    def load_grd(self, filename):
        self.meta, self.map = load_grd(filename)
        self.shape = self.map.shape[:2]
        # Check that XS and XE are offsets.
        self.x = self.meta['XCEN'] + np.linspace(self.meta['XS'], self.meta['XE'], self.meta['NX'])
        self.y = self.meta['YCEN'] + np.linspace(self.meta['YS'], self.meta['YE'], self.meta['NY'])

    def save_grd(self, filename):
        save_grd(filename, self.meta, self.map)

    def load_npy(self, folder, shape):
        """
        Load x.npy, y.npy, and map.npy arrays from the given
        folder. Shape is a tuple of length 2 that must match
        map[:2]. This is necessary because a generic GridMap can take
        different shapes depending on the number of components in the
        grid.
        """
        self.shape = shape
        return super(GridMap, self).load_npy(folder)

    def dB(self, component):
        """
        Return the given map component in decibels.
        """
        return 20 * np.log10(abs(self.map[component, 0]))


class JonesMap(FlatMap):

    shape = (2, 2)
    data_type = np.complex
    key = {0: 'co', 'co': 0,
           1: 'cx', 'cx': 1}

    def __init__(self, co=None, cx=None):
        """
        Create a new empty JonesMap or create one from two GridMap instances.
        """
        if co is None and cx is None:
            super(JonesMap, self).__init__()
        else:
            if not all(co.x == cx.x):
                raise ValueError("Map x values differ.")
            self.x = co.x
            if not all(co.y == cx.y):
                raise ValueError("Map y values dziffer.")
            self.y = co.y
            # Create a Jones matrix map with
            # map.shape = (2, 2, x.size, y.size)
            self.map = np.array([[co.map[0, 0], cx.map[0, 0]],
                                 [co.map[1, 0], cx.map[1, 0]]])

    # It's not clear what this means.
    def invert(self):
        map = np.empty((self.map.shape))
        for x in range(self.x.size):
            for y in range(self.y.size):
                map[:, :, x, y] = np.mat(self.map[:, :, x, y]).getI()
        self.map = map


class MuellerMap(FlatMap):

    # This is the shape of the matrix at each pixel.
    shape = (4, 4)
    # This is the map data type.
    data_type = np.float

    # This is the mapping between array indices and Stokes parameters.
    key = {0: 'T', 'T': 0,
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
            self.x = jones_map.x
            self.y = jones_map.y
            J = jones_map.map
            self.map = np.empty(self.shape + (self.x.size, self.y.size),
                                dtype=self.data_type)
            for x in range(self.x.size):
                for y in range(self.y.size):
                    J_xy = np.mat(J[:, :, x, y])
                    # The matrix cast is redundant since numpy takes *
                    # to mean matrix multiplication when either element
                    # is a matrix.
                    M_xy = self.A * np.mat(np.kron(J_xy, J_xy.conj())) * self.AI
                    if np.any(M_xy.imag):
                        raise ValueError("Nonzero complex value in M.")
                    self.map[:, :, x, y] = M_xy.real

    # Figure out what this means.
    def inverse_from_jones(self, jones_map):
        self.x = jones_map.x
        self.y = jones_map.y
        J = jones_map.map
        self.map = np.empty((4, 4, self.x.size, self.y.size), dtype='float')
        for x in range(self.x.size):
            for y in range(self.y.size):
                J_xy = np.mat(J[:, :, x, y])
                M_xy = np.mat(self.A * np.kron(J_xy, J_xy.conj()) * self.AI).getI()
                if np.any(M_xy.imag):
                    raise ValueError("Nonzero complex value in M.")
                self.map[:, :, x, y] = M_xy.real
        
    # Figure out how to create a title and axes labels.
    def contour_tile(self, color=None):
        plt.ioff()
        fig = plt.figure(figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                # Verify.
                name = self.key[i] + self.key[j]
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
                name = self.key[i] + self.key[j]
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
