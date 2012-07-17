import numpy as np

from pygrasp.flat_map import Grid

# Add the option to change the extents.
# Add the option to make other kinds of blank grids.
def blank_spherical_grid(nx, ny): #, x_min, y_min, x_max, y_max):
    g = Grid()
    g.header = ['Created by pygrasp.fake.blank_spherical_grid()',
                '++++']
    g.KTYPE = 1
    g.NSET = 1
    g.ICOMP = 3
    g.NCOMP = 2
    g.IGRID = 1
    g.IX = 0
    g.IY = 0
    g.XS = -0.5
    g.YS = -0.5
    g.XE = 0.5
    g.YE = 0.5
    g.NX = int(nx)
    g.NY = int(ny)
    g.KLIMIT = 0
    # Create the correct variables.
    g.DX = (g.XE - g.XS) / (g.NX - 1)
    g.DY = (g.YE - g.YS) / (g.NY - 1)
    g.XCEN = g.DX * g.IX
    g.YCEN = g.DY * g.IY
    g.X = g.XCEN + np.linspace(g.XS, g.XE, g.NX)
    g.Y = g.YCEN + np.linspace(g.YS, g.YE, g.NY)
    g.map = np.zeros((Grid.shape[0], Grid.shape[1], nx, ny), dtype=np.complex)
    return g
