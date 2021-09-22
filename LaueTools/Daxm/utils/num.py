#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import numpy as np

# should be used for low values only
def find_closest_product(val):

    n = np.floor(np.sqrt(val))
    m = np.ceil(val / n)

    fun = lambda n, m: 0.7*(n*m - val) + 0.3*(m - n)

    res = [n, m]
    obj = fun(n, m)

    while n > m/2 and obj > 0:
        n = n - 1
        m = np.ceil(val / n)

        if fun(n, m) < obj:
            res = [n, m]
            obj = fun(n, m)

    return res

def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def crop_curve(x, y, xmin, xmax):
    x, y = np.array(x), np.array(y)

    idx = np.logical_and(x >= xmin, x <= xmax)

    return x[idx], y[idx]


def is_within(val, a, b):
    return val >= a and val < b


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":

    pass