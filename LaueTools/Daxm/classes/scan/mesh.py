#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os

import numpy as np

from LaueTools.Daxm.utils.num import is_within

from LaueTools.Daxm.classes.scan.point import PointScan, save_scan_dict, load_scan_dict
import LaueTools.Daxm.utils.read_image as rimg

class MeshScan(PointScan):

    def __init__(self, inp, verbose=True):
        PointScan.__init__(self, inp, verbose)

        self.ix, self.iy = 0, 0
        self.size = (1, 1)
        self.skip = 0
        self.line_subfolder = False
        self.line_subname = ""
        self.line_subindex = 0
        self.img_mainfolder = ""

        self.init_mesh()

    def init_mesh(self):

        self.size = self.input['size']
        self.skip = self.input['skipFrame']
        self.line_subfolder = self.input['lineSubFolder']

        self.line_subname = ""
        self.line_subindex = 0
        self.img_mainfolder = self.img_folder

        if self.line_subfolder is None:
            self.line_subfolder = False

        if self.line_subfolder:
            parts = rimg.split_linesubfolder(self.img_folder)

            if parts is not None:
                self.line_subname = parts[0]
                self.line_subindex = parts[1]
                self.img_mainfolder = parts[2]

            else:
                self.line_subfolder = False

        self.ix = min(self.ix, self.size[0]-1)
        self.iy = min(self.iy, self.size[1]-1)

        self.goto()

    # Methods to safely and cleanly modify the scan
    def update(self, scan_dict, part):

        pos = self.get_current()

        self.goto(0, 0) # always modify in zero position

        PointScan.update(self, scan_dict, part)

        if part in ("data", "mesh", "all"):
            self.update_mesh(scan_dict)

        self.goto(*pos)

    def update_mesh(self, scan_dict):

        self.update_input(scan_dict, ['size', 'skipFrame', 'lineSubFolder'])

        self.init_mesh()

    # Getters
    def get_type(self):
        return "mesh"

    def get_current(self):
        return self.ix, self.iy

    def get_size(self):
        return self.size

    def get_images_tophat(self, step=5, ix=None, iy=None):

        if ix is None:
            ix = range(0, self.size[0])
        elif np.isscalar(ix):
            ix = [int(ix),]
        else:
            ix = [int(i) for i in ix]

        if iy is None:
            iy = range(0, self.size[0])
        elif np.isscalar(iy):
            iy = [int(iy),]
        else:
            iy = [int(i) for i in iy]

        ix_current, iy_current = self.ix, self.iy

        tophat = np.zeros(self.get_img_params(['framedim']))

        for x in ix:
            for y in iy:
                self.goto(x, y)
                tophat = np.maximum(tophat, PointScan.get_images_tophat(self, step))

        return tophat

    # Methods to navigate through scan serie
    def goto(self, ix=None, iy=None):

        if ix is None:
            ix = self.ix
        if iy is None:
            iy = self.iy

        if is_within(ix, 0, self.size[0]) and is_within(iy, 0, self.size[1]):

            if self.line_subfolder:
                idx0 = self.img_idx0 + (ix - self.ix) * (self.number_images + self.skip)
                img_subfolder_old = "{}{:d}".format(self.line_subname, self.iy + self.line_subindex)
                img_subfolder_new = "{}{:d}".format(self.line_subname, iy + self.line_subindex)
                img_folder = os.path.join(self.img_mainfolder, img_subfolder_new)
                img_pref = self.img_pref
                if img_subfolder_old in img_pref:
                    img_pref = img_pref.replace(img_subfolder_old, img_subfolder_new)
            else:
                dix = (ix - self.ix)
                diy = (iy - self.iy)
                idx0 = self.img_idx0 + (diy*self.size[0] + dix)*(self.number_images + self.skip)
                img_folder = self.img_mainfolder
                img_pref = self.img_pref

            scan_dict = {'imageFolder': img_folder,
                         'imagePrefix': img_pref,
                         'imageFirstIndex': idx0,
                         'imageDigits': self.img_nbdigits}
            PointScan.update(self, scan_dict, part="data")

            self.ix = ix
            self.iy = iy
        else:
            self.print_msg("Reached mesh limits...", mode="W")

    def goto_centre(self):
        self.goto(int(self.size[0])/2, int(self.size[1])/2)

    def right(self, count=1):
        self.goto(ix=self.ix + count)

    def left(self, count=1):
        self.goto(ix=self.ix - count)

    def up(self, count=1):
        self.goto(iy=self.iy + count)

    def down(self, count=1):
        self.goto(iy=self.iy - count)

    # Methods to load from file and save to file
    def save(self, filename, directory=""):

        save_scan_dict(self.to_dict(), filename, directory)

    def load(self, filename, directory=""):

        scan_dict = load_scan_dict(filename, directory)

        self.update(scan_dict, "all")

    def to_dict(self, part="all"):

        pos = self.get_current()

        self.goto(0, 0)

        scan_dict = PointScan.to_dict(self, part)

        self.goto(*pos)

        scan_dict.update({'type': 'mesh',
                          'size':self.get_size(),
                          'skipFrame':self.skip,
                          'lineSubFolder':self.line_subfolder})

        return scan_dict
