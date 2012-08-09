from __future__ import division

import numpy as np

from pygrasp.output import load_cut, save_cut

class Cut(object):

    def __init__(self, filename):
        self.meta, self.cut = load_cut(filename)
        self.x = np.linspace(self.meta['V_INI'],
                             self.meta['V_INI'] + self.meta['V_INC'] * self.meta['V_NUM'],
                             self.meta['V_NUM'],
                             endpoint=False)


    
