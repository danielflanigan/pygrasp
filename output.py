"""
This module contains classes that parse the output of GRASP simulations.

To do:
Implement Cut to read .cut files.
Hack .cur files.
"""
from __future__ import division

import os
import numpy as np

# This two functions handle numbers that have three-digit exponents,
# which GRASP writes as, e.g., 0.123456789-100 for 1.23456789E-99
# To do: match GRASP format.
def string_to_float(string):
    try:
        return float(string)
    except ValueError:
        if '-' in string[1:]:
            return float('E-'.join(string.rsplit('-', 1)))
        else:
            return float('E+'.join(string.rsplit('+', 1)))

def float_to_string(number):
    if number == 0 or abs(np.log10(abs(number))) < 100:
        return ' {: 0.10E}'.format(number)
    else:
        return ' {: 0.10E}'.format(number).replace('E', '')

def load_cut(filename):
    """
    Read and parse data from the GRASP .cut file. The variables in
    capital letters match those in the GRASP-10 manual.
    """
    with open(filename, 'r') as f:
        meta = {}
        meta['header'] = f.readline().rstrip('\n').strip()
        V_INI, V_INC, V_NUM, C, ICOMP, ICUT, NCOMP = f.readline().split()
        meta['V_INI'] = float(V_INI)
        meta['V_INC'] = float(V_INC)
        meta['V_NUM'] = int(V_NUM)
        meta['C'] = float(C)
        meta['ICOMP'] = int(ICOMP)
        meta['ICUT'] = int(ICUT)
        meta['NCOMP'] = int(NCOMP)

        conv = dict([(column, string_to_float) for column in range(2 * meta['NCOMP'])])
        data = np.loadtxt(f, dtype=float, converters=conv)
    cut = np.array([data[:, 2 * column] +
                    1j * data[:, 2 * column + 1]
                    for column in range(meta['NCOMP'])])
    return meta, cut

def save_cut(filename, meta, cut):
    pass


def load_grd(filename):
    """
    Read and parse data from the GRASP .grd file. The variables in
    capital letters match those in the GRASP-10 manual.
    """
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
        # Implement this to read elliptically truncated grids.
        if meta['KLIMIT'] != 0:
            raise ValueError("Not implemented.")
        # Load the field data. This returns an array with shape (NX * NY, 2 * NCOMP).
        conv = dict([(column, string_to_float) for column in range(2 * meta['NCOMP'])])
        data = np.loadtxt(f, dtype=float, converters=conv)
    # Determine the grid spacing and center values.
    meta['DX'] = (meta['XE'] - meta['XS']) / (meta['NX'] - 1)
    meta['DY'] = (meta['YE'] - meta['YS']) / (meta['NY'] - 1)
    meta['XCEN'] = meta['DX'] * meta['IX']
    meta['YCEN'] = meta['DY'] * meta['IY']
    # Reshape the data.
    map = np.array([[(data[:, 2 * column] +
                      1j * data[:, 2 * column + 1]).reshape(meta['NX'], meta['NY'], order='F')]
                    for column in range(meta['NCOMP'])])
    return meta, map

def save_grd(filename, meta, map):
    """
    Write the data in this Grid to a new .grd file. Will not overwrite.
    """
    if os.path.exists(filename):
        raise ValueError("File already exists: {}".format(filename))
    if map.shape != (meta['NCOMP'], 1, meta['NX'], meta['NY']):
        raise ValueError("The map shape does not match the metadata dictionary.")
    points = meta['NX'] * meta['NY']
    components = meta['NCOMP']
    data = np.empty((points, 2 * components))
    for component in range(components):
        data[:, 2 * component] = map[component, 0].reshape(points, order='F').real
        data[:, 2 * component + 1] = map[component, 0].reshape(points, order='F').imag
    with open(filename, 'w') as f:
        for line in meta['header']:
            f.write('{}\n'.format(line))
        f.write('{:2d}\n'.format(meta['KTYPE']))
        f.write('{:12d}{:12d}{:12d}{:12d}\n'.format(meta['NSET'], meta['ICOMP'], meta['NCOMP'], meta['IGRID']))
        f.write('{:12d}{:12d}\n'.format(meta['IX'], meta['IY']))
        f.write(' {: 0.10E} {: 0.10E} {: 0.10E} {: 0.10E}\n'.format(meta['XS'], meta['YS'], meta['XE'], meta['YE']))
        f.write('{:12d}{:12d}{:12d}\n'.format(meta['NX'], meta['NY'], meta['KLIMIT']))
        for p in range(data.shape[0]):
            f.write(''.join([float_to_string(number) for number in data[p, :]]) + '\n')
