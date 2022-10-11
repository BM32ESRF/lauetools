#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


from LaueTools.Daxm.utils.num import is_within

from LaueTools.Daxm.classes.scan.point import PointScan, save_scan_dict, load_scan_dict


class LineScan(PointScan):

    def __init__(self, inp, verbose=True):
        PointScan.__init__(self, inp, verbose)

        self.ix = 0
        self.size = 1
        self.skip = 0

        self.init_line()

    def init_line(self, pos=None):

        self.size = self.input['size']
        self.skip = self.input['skipFrame']
        self.ix = min(self.ix, self.size-1)

        self.goto(pos)

    # Getters
    def get_type(self):
        return "line"

    def get_current(self):
        return self.ix

    def get_size(self):
        return self.size

    # Methods to safely and cleanly modify the scan
    def update(self, scan_dict, part):

        pos = self.get_current()

        self.goto(0)

        PointScan.update(self, scan_dict, part)

        if part in ("data", "line", "all"):
            self.update_line(scan_dict)

        self.goto(pos)

    def update_line(self, scan_dict):

        self.update_input(scan_dict, ['size', 'skipFrame'])

        self.init_line()

    # Methods to navigate through scan serie
    def goto(self, ix=None):

        if ix is None:
            ix = self.ix

        if is_within(ix, 0, self.size):

            idx0 = self.img_idx0 + (ix - self.ix) * (self.number_images + self.skip)

            scan_dict = {'imageFolder': self.img_folder,
                         'imagePrefix': self.img_pref,
                         'imageFirstIndex': idx0,
                         'imageDigits': self.img_nbdigits}

            PointScan.update(self, scan_dict, part="data")

            self.ix = ix
        else:
            self.print_msg("Reached line limit...", mode="W")

    def goto_centre(self):
        self.goto(int(self.size)/2)

    def next(self, count=1):
        self.goto(self.ix + count)

    def previous(self, count=1):
        self.goto(self.ix - count)

    # Methods to load from file and save to file
    def save(self, filename, directory=""):

        save_scan_dict(self.to_dict(), filename, directory)

    def load(self, filename, directory=""):

        scan_dict = load_scan_dict(filename, directory)

        self.update(scan_dict, "all")

    def to_dict(self, part="all"):

        pos = self.get_current()

        self.goto(0)

        scan_dict = PointScan.to_dict(self, part)

        self.goto(pos)

        scan_dict.update({'type': 'line',
                          'size':self.size,
                          'skipFrame':self.skip})

        return scan_dict

