#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import types

from LaueTools.Daxm.classes.scan.point import PointScan, ScanError


def new_simscan_dict(scan_dict=None):

    new_dict = {'specFile': None,
                'scanNumber': 0,
                'scanCmd': [0, 0.5, 500, 1.],
                'CCDType': 'sCMOS',
                'detCalib': [77., 978., 975., 0., 0., 0.07340],
                'wire': ['W'],
                'wireTrajAngle': 0.}

    if scan_dict is not None:
        for key in new_dict.keys():
            new_dict[key] = scan_dict[key]
    return new_dict


class SimScan(PointScan):

    def __init__(self, inp, verbose=True):

        self.disable_data = True
        self.disable_mon = True

        PointScan.__init__(self, inp, verbose)

    def __getattribute__(self, item):
        # /!\ cant use "self." in here to avoid infinite loop
        # instead call Scan.__getattribute(self, ...)
        attr = PointScan.__getattribute__(self, item)

        if isinstance(attr, types.MethodType):

            if PointScan.__getattribute__(self, "disable_data") \
                    and any(word in item for word in ['image', 'profile', 'data']):
                #pass
                raise ScanError(item + " was disabled.")

            if PointScan.__getattribute__(self, "disable_mon") \
                    and any(word in item for word in ['monitor']):
                #pass
                raise ScanError(item + " was disabled.")

        return attr

    def update(self, scan_dict, part):

        if part in ("setup", "all"):
            self.update_setup(scan_dict)

        if part in ("wire", "all"):
            self.update_wire(scan_dict)

        if part in ("spec", "all"):
            self.update_spec(scan_dict)

    def update_setup(self, scan_dict):

        PointScan.update_input(self, scan_dict, ['CCDType', 'detCalib'])

        PointScan.init_detector(self)

    def update_spec(self, scan_dict):

        PointScan.update_input(self, scan_dict, ['specFile', 'scanNumber', 'scanCmd'])

        PointScan.init_spec(self)
