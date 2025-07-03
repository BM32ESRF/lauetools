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


def new_scan(scan_inp, verbose=True)->'ScanObject':

    """
    Return a ScanObject based on input

    Parameters
    ----------
    scan_inp : str, dict or ScanObject from classes PointScan, LineScan, MeshScan
        either a filepath to a scan file, a dictionary with scan parameters or an existing ScanObject
    verbose : bool, optional
        set verbosity level for ScanObject
    addscan0001 : bool, optional
        add 'scan0001' subfolder to filepath if not already present

    Returns
    -------
    ScanObject
        the created ScanObject

    """
    return_scanObj = None
    print('type of scan_inp is', type(scan_inp))

    if is_type_scan(scan_inp):

        return_scanObj = scan_inp
        return_scanObj.set_verbosity(verbose)

    elif isinstance(scan_inp, dict):

        return_scanObj = new_scan_fromdict(scan_inp, verbose=verbose)

    elif isinstance(scan_inp, str): # filepath

        return_scanObj = new_scan_fromfile(scan_inp, verbose=verbose)

    else:
        pass

    return return_scanObj

def new_scan_fromdict(scan_dict=None, verbose=True):

    if scan_dict is None:

        scan_dict = {"type":"point"}

    if scan_dict["type"] == "line":

        return LineScan(scan_dict, verbose)

    elif scan_dict["type"] == "mesh":

        return MeshScan(scan_dict, verbose)

    else:

        return PointScan(scan_dict, verbose)

def new_scan_fromfile(scan_file, directory="", verbose=True, addscan0001=False):

    if 1:#verbose:
        print(scan_file, directory)
    scan_dict = load_scan_dict(scan_file, directory)

    return new_scan(scan_dict, verbose)


