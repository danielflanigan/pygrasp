from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

"""
This is an abstract class that contains methods useful for maps.
"""

class Map(object):

    def __init__(self):
        self.X = np.array([0.])
        self.Y = np.array([0.])

    def dx(self):
        """
        Return the pixel spacing in x.
        """
        return (self.X[-1] - self.X[0]) / (self.X.size - 1)

    def dy(self):
        """
        Return the pixel spacing in y.
        """
        return (self.Y[-1] - self.Y[0]) / (self.Y.size - 1)

    def recenter(self, x, y):
        """
        Set the map center to the new coordinates, preserving all
        other aspects of the pixelization.
        """
        x0 = (self.X[-1] + self.X[0]) / 2
        y0 = (self.Y[-1] + self.Y[0]) / 2
        self.X += x - x0
        self.Y += y - y0

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
            i_x = np.array([np.int(np.round((x[0] - self.X[0]) / self.dx()))])
            i_y = np.array([np.int(np.round((y[0] - self.Y[0]) / self.dy()))])
        else:
            i_x = np.int_(np.round_((x - self.X[0]) / self.dx()))
            i_y = np.int_(np.round_((y - self.Y[0]) / self.dy()))
        within = ((0 <= i_x) & (i_x < self.X.size) & (0 <= i_y) & (i_y < self.Y.size))
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
        return self.X[i_x], self.Y[i_y]

    # Work out transposition and extents.
    def make_plot(self, a, title="", xlabel="", ylabel="", color=plt.cm.jet):
        plt.ioff()
        plt.figure()
        plt.imshow(a.T,
                   cmap=color,
                   aspect='equal',			 
                   interpolation='nearest',
                   origin='lower',
                   extent=(self.X[0], self.X[-1], self.Y[0], self.Y[-1]))
        plt.colorbar(shrink=0.8, aspect=20*0.8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def show_plot(self, a, title="", xlabel="", ylabel="", color=plt.cm.jet):
        """Create and show a plot of the data or weights map; see make_plot() for usage."""
        self.make_plot(a, title=title, xlabel=xlabel, ylabel=ylabel, color=color)
        plt.ion()
        plt.show()
        
    def save_plot(self, filename, a, title="", xlabel="", ylabel="", color=plt.cm.jet):
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
                    extent=(self.X[0], self.X[-1], self.Y[0], self.Y[-1]))
        plt.colorbar(shrink=0.8, aspect=20*0.8, format='%.3f')
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

    def show_tile(self):
        plt.ioff()
        fig = plt.figure()
        names = {0: 'T', 1: 'Q', 2: 'U', 3: 'V'}
        for i in range(4):
            for j in range(4):
                name = '%s%s' %(names[j], names[i])
                sub = plt.subplot(4, 4, 4 * i + j + 1)
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
                a = self.M[i, j]
                plt.contour(a.T,
                            contours=np.linspace(np.min(a.flatten()), np.max(a.flatten()), 10))
                sub.title.set_text(name)
        plt.ion()
        plt.show()
        return fig
    
