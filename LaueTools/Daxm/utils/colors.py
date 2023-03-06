#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import numpy as np

import matplotlib.pyplot as mplp


def gen_colormap(qty, name="rainbow"):
    cmap = getattr(mplp.cm, name)
    return [cmap(i) for i in np.linspace(0, 1, qty)]


