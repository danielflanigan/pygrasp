from __future__ import division

import os
import numpy as np

from pygrasp.map import Map

class Grid(Map):

    # This is the shape of the matrix at each pixel.
    shape = (2, 1)
    # This is the map data type.
    data_type = np.complex

    # This is the mapping between array indices and Jones vector components.
    jones = {0: 'co', 'co': 0,
             1: 'cx', 'cx': 1}

    def __init__(self, filename=None):
        """
        Create a new grid file. If a filename is supplied, load the
        data from that file, otherwise create a blank grid.
        """
        if filename is None:
            super(Grid, self).__init__(self.shape, self.data_type)
        else:
            self.load_grd(filename)

    def load_grd(self, filename):
        """
        Read and parse data from the GRASP .grd file. The variables in
        capital letters match those in the GRASP-10 manual.
        """
        with open(filename, 'r') as f:
            self.header = []
            self.header.append(f.readline().rstrip('\n'))
            while self.header[-1] != '++++':
                self.header.append(f.readline().rstrip('\n'))

            # These determine the type of grid and the field format.
            self.KTYPE = int(f.readline().split()[0])
            if self.KTYPE != 1:
                raise ValueError("Not implemented.")
            self.NSET, self.ICOMP, self.NCOMP, self.IGRID = [int(s) for s in f.readline().split()]

            # The grid center in units of the x and y grid spacing.
            self.IX, self.IY = [int(s) for s in f.readline().split()]

            # These are the x and y grid limits: S is lower, and E is upper.
            self.XS, self.YS, self.XE, self.YE = [float(s) for s in f.readline().split()]
            
            # These are the numbers of grid points in x and y.
            self.NX, self.NY, self.KLIMIT = [int(s) for s in f.readline().split()]
            if self.KLIMIT != 0:
                raise ValueError("Not implemented.")

            # Load the field data.
            data = np.loadtxt(f)
            
        # Determine the grid spacing and center values.
        self.DX = (self.XE - self.XS) / (self.NX - 1)
        self.DY = (self.YE - self.YS) / (self.NY - 1)
        self.XCEN = self.DX * self.IX
        self.YCEN = self.DY * self.IY

        # Determine the x and y grid values.
        self.X = self.XCEN + np.linspace(self.XS, self.XE, self.NX)
        self.Y = self.YCEN + np.linspace(self.YS, self.YE, self.NY)        

        # Organize the data. The manual says that the file is
        # organized so that X varies faster than Y, which corresponds
        # to column-major (Fortran) ordering in the reshape.
        E_co = (data[:, 0] + 1j * data[:, 1]).reshape(self.NX, self.NY, order='F')
        E_cx = (data[:, 2] + 1j * data[:, 3]).reshape(self.NX, self.NY, order='F')
        self.map = np.array([[E_co], [E_cx]])

    def save_grd(self, filename):
        """
        Write the data in this Grid to a new .grd file. Will not overwrite.
        """
        if os.path.exists(filename):
            raise ValueError("File already exists: {}".format(filename))
        with open(filename, 'w') as f:
            for line in self.header:
                f.write('{}\n'.format(line))
            f.write('{:2d}\n'.format(self.KTYPE))
            f.write('{:12d}{:12d}{:12d}{:12d}\n'.format(self.NSET, self.ICOMP, self.NCOMP, self.IGRID))
            f.write('{:12d}{:12d}\n'.format(self.IX, self.IY))
            f.write(' {: 0.10E} {: 0.10E} {: 0.10E} {: 0.10E}\n'.format(self.XS, self.YS, self.XE, self.YE))
            f.write('{:12d}{:12d}{:12d}\n'.format(self.NX, self.NY, self.KLIMIT))
            # This creates new views into the arrays, not copies.
            E_co = self.map[0, 0].reshape(self.NX * self.NY, order='F')
            E_cx = self.map[1, 0].reshape(self.NX * self.NY, order='F')
            for i in range(self.NX * self.NY):
                f.write(' {: 0.10E} {: 0.10E} {: 0.10E} {: 0.10E}\n'.format(E_co[i].real, E_co[i].imag, E_cx[i].real, E_cx[i].imag))
