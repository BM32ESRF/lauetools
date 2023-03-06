#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import math


def is_list_of(obj, thetype):
    # This if statement makes sure input is a list that is not empty
    if obj and isinstance(obj, list):
        return all(isinstance(s, thetype) for s in obj)
    else:
        return False


def allequal_float(a, eps=1E-8):
    for i in range(len(a)):
        if math.fabs(a[i] - a[0]) > eps:
            return False
    return True


def allequal_str(a):
    for i in range(len(a)):
        if a[i] != a[0]:
            return False
    return True


def unique_order(a):
    used = set()
    return [x for x in a if x not in used and (used.add(x) or True)]


def array_indices(a, idx):
    return [a[i] for i in idx]


def closest_value_idx(mylist, value):
    cval = min(mylist, key=lambda x: abs(x - value))
    return list(mylist).index(cval)


def chunks(l, n):
    m = int(math.ceil( len(l) * 1./ n ))
    return [l[i:i+m] for i in range(0, len(l), m)]