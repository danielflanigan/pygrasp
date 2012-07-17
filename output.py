"""
This module contains classes that parse the output of GRASP simulations.

To do:
Implement Cut to read .cut files.
Hack .cur files.
"""
from __future__ import division

import os
import numpy as np

class Grid(object):
    """
    This class represents a generic .grd file. Classes that represent
    maps of grids inherit from it.
    """

    @classmethod
    def load_grd(cls, filename):
        """
        Read and parse data from the GRASP .grd file. The variables in
        capital letters match those in the GRASP-10 manual.
        """
        # This handles numbers that have three-digit exponents, which GRASP
        # writes as, e.g., 0.123456789-100 for 1.23456789E-99
        def convert(string):
            try:
                return float(string)
            except ValueError:
                if '-' in string[1:]:
                    return float('E-'.join(string.rsplit('-', 1)))
                else:
                    return float('E+'.join(string.rsplit('+', 1)))

        with open(filename, 'r') as f:
            meta = {}
            meta['header'] = []
            meta['header'].append(f.readline().rstrip('\n'))
            while meta['header'][-1] != '++++':
                meta['header'].append(f.readline().rstrip('\n'))
            # These determine the type of grid and the field format.
            meta['KTYPE'] = int(f.readline().split()[0])
            if meta['KTYPE'] != 1:
                raise ValueError("Not implemented.")
            meta['NSET'], meta['ICOMP'], meta['NCOMP'], meta['IGRID'] = [int(s) for s in f.readline().split()]
            # The grid center in units of the x and y grid spacing.
            meta['IX'], meta['IY'] = [int(s) for s in f.readline().split()]
            # These are the x and y grid limits: S is lower, and E is upper.
            meta['XS'], meta['YS'], meta['XE'], meta['YE'] = [float(s) for s in f.readline().split()]
            # These are the numbers of grid points in x and y.
            meta['NX'], meta['NY'], meta['KLIMIT'] = [int(s) for s in f.readline().split()]
            if meta['KLIMIT'] != 0:
                raise ValueError("Not implemented.")
            # Determine the grid spacing and center values.
            meta['DX'] = (meta['XE'] - meta['XS']) / (meta['NX'] - 1)
            meta['DY'] = (meta['YE'] - meta['YS']) / (meta['NY'] - 1)
            meta['XCEN'] = meta['DX'] * meta['IX']
            meta['YCEN'] = meta['DY'] * meta['IY']
            # Load the field data. This returns an array with shape (NX * NY, 2 * NCOMP).
            conv = dict([(column, convert) for column in range(2 * meta['NCOMP'])])
            data = np.loadtxt(f, dtype=float, converters=conv)
        return meta, data

    @classmethod
    def save_grd(cls, filename, meta, data):
        """
        Write the data in this Grid to a new .grd file. Will not overwrite.
        """
        # This handles numbers that have three-digit exponents, which GRASP
        # writes as, e.g., 0.123456789-100 for 1.23456789E-99
        def convert(number):
            if number == 0 or abs(np.log10(abs(number))) < 100:
                return ' {: 0.10E}'.format(number)
            else:
                # Check this.
                return ' {: 0.10E}'.format(number).replace('E', '')

        if os.path.exists(filename):
            raise ValueError("File already exists: {}".format(filename))
        if data.shape != (meta['NX'] * meta['NY'], 2 * meta['NCOMP']):
            raise ValueError("The data array shape does not match the metadata dictionary.")
        with open(filename, 'w') as f:
            for line in meta['header']:
                f.write('{}\n'.format(line))
            f.write('{:2d}\n'.format(meta['KTYPE']))
            f.write('{:12d}{:12d}{:12d}{:12d}\n'.format(meta['NSET'], meta['ICOMP'], meta['NCOMP'], meta['IGRID']))
            f.write('{:12d}{:12d}\n'.format(meta['IX'], meta['IY']))
            f.write(' {: 0.10E} {: 0.10E} {: 0.10E} {: 0.10E}\n'.format(meta['XS'], meta['YS'], meta['XE'], meta['YE']))
            f.write('{:12d}{:12d}{:12d}\n'.format(meta['NX'], meta['NY'], meta['KLIMIT']))
            for p in range(data.shape[0]):
                f.write(''.join([convert(number) for number in data[p, :]]) + '\n')
