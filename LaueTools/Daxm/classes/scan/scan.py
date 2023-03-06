#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


from LaueTools.Daxm.classes.scan.point import PointScan, load_scan_dict
from LaueTools.Daxm.classes.scan.line import LineScan
from LaueTools.Daxm.classes.scan.mesh import MeshScan


def is_type_scan(obj):
    """return True is obj is of class PointScan, LineScan or MeshScan
    
    """
    return isinstance(obj, (PointScan, LineScan, MeshScan))


def new_scan(scan_inp, verbose=True):

    scan = None
    print('scan_inp', scan_inp)
    if is_type_scan(scan_inp):

        scan = scan_inp

        scan.set_verbosity(verbose)

    elif isinstance(scan_inp, dict):

        scan = new_scan_fromdict(scan_inp, verbose=verbose)

    elif isinstance(scan_inp, str):

        scan = new_scan_fromfile(scan_inp, verbose=verbose)

    else:
        pass

    return scan

def new_scan_fromdict(scan_dict=None, verbose=True):

    if scan_dict is None:

        scan_dict = {"type":"point"}

    if scan_dict["type"] == "line":

        return LineScan(scan_dict, verbose)

    elif scan_dict["type"] == "mesh":

        return MeshScan(scan_dict, verbose)

    else:

        return PointScan(scan_dict, verbose)

def new_scan_fromfile(scan_file, directory="", verbose=True):

    if 1:#verbose:
        print(scan_file, directory)
    scan_dict = load_scan_dict(scan_file, directory)

    return new_scan(scan_dict, verbose)


