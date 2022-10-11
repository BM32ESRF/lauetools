#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import time

import numpy as np

from LaueTools.Daxm.utils.write_image import calc_ndigits
from LaueTools.Daxm.classes.reconstruction.scan import ScanReconstructor


class RecManager:
    # Constructors
    def __init__(self, scan, calib, seg):

        self.scan = scan
        self.calib = calib
        self.seg = seg

        self.fitfile = None

        self.grid_ix = []
        self.grid_ix = []

        self.grid_x = []
        self.grid_y = []

        try:
            self.grid_ix = range(0, self.scan.size[0])
            self.grid_iy = range(0, self.scan.size[1])
        

            self.grid_x, self.grid_y = np.array(self.grid_ix, dtype=float), np.array(self.grid_ix, dtype=float)

            self.grid_x = self.grid_x - self.grid_x[int(self.scan.size[0])/2]
            self.grid_y = self.grid_y - self.grid_y[int(self.scan.size[1])/2]

        except AttributeError:
            self.grid_ix = range(1)
            self.grid_iy = range(1)
            self.grid_x, self.grid_y  = np.array([0.,]),np.array([0.,])

    def set_grid(self, x=None, y=None):

        if x is not None:
            self.grid_x = x

        if y is not None:
            self.grid_y = y

    def set_calib_yref(self, yref):

        self.calib.set_yref(yref)

    def set_fitfile(self, fitfile):

        self.fitfile = fitfile

    def reconstruct(self, depth_range, fileprefix, depth_step=0.001, nproc=1, directory="", rec_par={}, depth_range_print=None):

        grid_depth = np.arange(depth_range_print[0], depth_range_print[1], depth_step)

        imgqty_per_scan = len(grid_depth)

        try:
            imgqty = imgqty_per_scan * self.scan.size[0] * self.scan.size[1]
            img_idx = np.zeros(self.scan.size, dtype=int)
        except AttributeError:
            imgqty = imgqty_per_scan
            img_idx = np.array([0,], dtype=int)

        

        ndigits = calc_ndigits(imgqty)

        prev_index = 0
        for iy in self.grid_iy:
            for ix in self.grid_ix:
                if ix > 0 or iy > 0:
                    img_idx[ix][iy] = int(prev_index)
                prev_index = prev_index + imgqty_per_scan

        self.scan.set_verbosity(False)

        for iy, y in zip(self.grid_iy, self.grid_y):

            print("[rec] ---------- Reconstruction of LINE %d ----------"%(iy,))

            print("[rec]  > computing tophat image for the line...")

            if len(self.grid_x) > 1:
                self.seg.update({"I": self.scan.get_images_tophat(iy=iy)})
            else:
                self.seg.update({"I": self.scan.get_images_tophat()})

            for ix, x in zip(self.grid_ix, self.grid_x):

                start_time = time.time()

                print("[rec] > Reconstructing Line %d, X = %d..." %(iy, ix))

                if len(self.grid_x)>1:
                    self.scan.goto(ix, iy)
                
                rec = ScanReconstructor(self.scan, wires = self.calib.get_wires(y))

                rec.set_regions_fromsearch(**self.seg)

                if self.fitfile is None:

                    rec.init_abscoeff()

                else:
                    rec.set_abscoeff_fromfitfile(self.fitfile)

                rec.assign_wire_peaks()

                rec.reconstruct(yrange=depth_range, halfboxsize=None, ystep=0.001, nproc=nproc, rec_args=rec_par)

                if len(self.grid_x)>1:
                    rec.print_images(prefix=fileprefix, first_index=img_idx[ix][iy], directory=directory, yrange=depth_range_print, nbdigits=ndigits)
                else:
                    rec.print_images(prefix=fileprefix, first_index=0, directory=directory, yrange=depth_range_print, nbdigits=ndigits)

                rec.free()

                print("[rec] elapsed time = %s seconds" % (time.time() - start_time))
