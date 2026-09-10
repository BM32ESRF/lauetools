#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os
import json

import numpy as np

from LaueTools.Daxm.classes.wire import CircularWire

from LaueTools.Daxm.classes.calibration.diffraction import CalibDiff
from LaueTools.Daxm.classes.calibration.fluorescence import CalibFluo


def new_calib(scan, sample, kind="fluo", verbose=True):

    calib = None

    if "fluorescence".startswith(kind):

        calib = CalibFluo(scan, sample, verbose)

    elif "diffraction".startswith(kind):

        calib = CalibDiff(scan, sample, verbose)

    else:
        # TODO : choose kind from sample
        pass

    return calib


class CalibManager:
    # Constructors
    def __init__(self, filename, yref, directory=""):

        self.wires_dict = []

        self.yref = float(yref)

        self.load_calib(filename, directory)

    # Getters
    def get_wires_dict(self, y=None):

        if y is None:
            y = self.yref
        else:
            y = float(y)

        tmp_dict = [dict(wdict) for wdict in self.wires_dict]

        for i in range(len(tmp_dict)):

            for key in ['R', 'h', 'p0', 'u1', 'u2', 'f1', 'f2']:
                tmp_dict[i][key] = float(tmp_dict[i][key])

            tmp_dict[i]['h'] = float(tmp_dict[i]['h'] + (self.yref - y) * np.sin(np.deg2rad(40.)))

        return tmp_dict

    def get_wires(self, y=None):

        return [CircularWire(**wdict) for wdict in self.get_wires_dict(y)]

    # Setters
    def set_yref(self, yref):
        self.yref = float(yref)

    # File IO
    def load_calib(self, filename, directory=""):

        # load wires from json file
        fn = os.path.join(directory, filename)

        with open(fn, 'r') as pfile:
            tmp = json.load(pfile)

        traj_dict, wires_dict = tmp['traj'], tmp['wires']

        traj_dict['u1'] = np.deg2rad(traj_dict['u1'])
        traj_dict['u2'] = np.deg2rad(traj_dict['u2'])

        for i in range(len(wires_dict)):
            wires_dict[i]['f1'] = np.deg2rad(wires_dict[i]['f1'])
            wires_dict[i]['f2'] = np.deg2rad(wires_dict[i]['f2'])

        for i in range(len(wires_dict)):
            wires_dict[i].update(traj_dict)

        self.wires_dict = wires_dict


# Test
if __name__ == '__main__':

    testfile = "/home/renversa/Research/data/nov17_kirchlechner/data/sunday/calibration/CuSxx1_N8L2_calibtest.calib"

    CM = CalibManager(testfile, 0.)

    for d in CM.get_wires_dict():
        print(d)

    for d in CM.get_wires_dict(0.010):
        print(d)
