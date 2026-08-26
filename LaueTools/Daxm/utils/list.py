#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import math

import numpy as np


from LaueTools.Daxm.classes.wire import CircularWire  # Import your CircularWire class

def is_list_of(obj, type_=None):
    """
    Check if `obj` is a list or NumPy array of objects of type `type_`.
    Special handling for CircularWire objects:
    - Returns True if `obj` is a list/array of CircularWire objects.
    - Returns False if `obj` is a single CircularWire object.

    Parameters
    ----------
    obj : object
        Object to check.
    type_ : type, optional
        Expected type of elements in the list/array.

    Returns
    -------
    bool
        True if `obj` is a list or NumPy array of objects of type `type_`.
    """
    # Explicitly check for list or np.ndarray (not subclasses)
    if type(obj) is list or type(obj) is np.ndarray:
        if type_ is None:
            return True
        # Special case: If type_ is CircularWire, check if all elements are CircularWire
        if type_ is CircularWire:
            return all(isinstance(x, CircularWire) for x in obj)
        return all(isinstance(x, type_) for x in obj)
    return False


def is_list_of(obj, type_=None):
    """
    Check if `obj` is a list or NumPy array of objects of type `type_`.
    Explicitly excludes custom objects (e.g., CircularWire) even if they inherit from list/array.

    Parameters
    ----------
    obj : object
        Object to check.
    type_ : type, optional
        Expected type of elements in the list/array.

    Returns
    -------
    bool
        True if `obj` is a list or NumPy array of objects of type `type_`.
    """
    # Explicitly check for list or np.ndarray, but not subclasses (e.g., custom objects)
    if type(obj) is list or type(obj) is np.ndarray:
        if type_ is None:
            return True
        return all(isinstance(x, type_) for x in obj)
    return False

def is_list_of_old(obj, thetype):
    # This if statement makes sure input is a list that is not empty
    if 1: #verbose>0:
        print(obj)
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