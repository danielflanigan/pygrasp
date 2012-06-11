from __future__ import division

import numpy as np

from pygrasp.map import Map

class Grid(Map):

    def __init__(self, filename):
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
        self.E_co = (data[:, 0] + 1j * data[:, 1]).reshape(self.NX, self.NY, order='F')
        self.E_cx = (data[:, 2] + 1j * data[:, 3]).reshape(self.NX, self.NY, order='F')
