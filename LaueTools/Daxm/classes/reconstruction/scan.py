#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os, sys, gc

import numpy as np
import scipy as sp

sys.path.insert(1, '/home/micha/LaueToolsPy3')


import multiprocessing as mpi
import scipy.ndimage as ndimage


from LaueTools import readmccd as rmccd
from LaueTools import IOLaueTools as rwa


from LaueTools.Daxm.utils.geom import closest_point
import LaueTools.Daxm.utils.write_image as wimg
import LaueTools.Daxm.modules.segmentation2 as seg
import LaueTools.Daxm.material.absorption as abso

from LaueTools.Daxm.classes.reconstruction.spot import SpotReconstructor, RecError


class ScanReconstructor:

    def __init__(self, scan, peaks_XY=None, peaks_energy=None, wires=None):

        self.scan = scan

        if wires is None:
            self.wires = scan.wire
        else:
            self.wires = wires

        self.init_peaks(peaks_XY)

        self.init_halfboxsize()

        self.init_abscoeff(peaks_energy)

    def free(self):

        del self.spots_rec[:]

        gc.collect()

    # getters
    def get_wireqty(self):

        return len(self.wires)

        # ----------------- regions -----------------
    def set_regions(self, peaks_XY, halfboxsize):

        self.set_peaks(peaks_XY)

        self.set_halfboxsize(halfboxsize)

    def set_regions_fromsearch(self, I=None, max_size=100, min_size=3, thr=20, erode=2, dilate=2, merge=True):

        if I is None:
            I = self.scan.get_images_tophat()

        mask, _ = seg.apply_threshold(I, max_size, min_size, thr, erode, dilate)

        peaks_XY, peaks_hbs = seg.draw_bbox(I, mask, merge)

        self.set_regions(peaks_XY, peaks_hbs)

        return mask

    # ----------------- peaks -----------------

    def init_peaks(self, peaks_XY=None):

        if peaks_XY is None:

            peaks_XY = []

        self.set_peaks(peaks_XY)

    def set_peaks(self, peaks_XY):

        if isinstance(peaks_XY, str):

            self.set_peaks_fromfitfile(peaks_XY)

        else:

            self.peaks_all = peaks_XY

            self.peakqty_all = len(self.peaks_all)

            self.peaks_abscoeff_all = np.zeros((self.peakqty_all, self.get_wireqty()))

            self.halfboxsize_all = np.ones((self.peakqty_all, 2), dtype=int)

            self.assign_wire_peaks()

    def set_peaks_fromfitfile(self, fit_file, usecols=[7, 8], skiprows=5):

        peaks_XY = np.loadtxt(fit_file, skiprows=skiprows, usecols=usecols)

        self.set_peaks(peaks_XY)

    def set_peaks_fromsearch(self, image_file=None, threshold=150, fit_peaks=1):

        if image_file is None:

            image_file = self.scan.get_image_filedir(0)

        res = rmccd.PeakSearch(image_file,
                             return_histo=0,
                             local_maxima_search_method=0,
                             IntensityThreshold=threshold,
                             fit_peaks_gaussian=fit_peaks,
                             Data_for_localMaxima='auto_background',
                             CCDLabel=self.scan.ccd_type)

        peaks_XY = res[0][:, :2] - 1

        self.set_peaks(peaks_XY)

    # ----------------- abscoeff -----------------

    def init_abscoeff(self, peaks_energy=None):

        if peaks_energy is None:

            peaks_energy = np.ones(self.peakqty_all) * 10.  # default value of 10 keV

        self.set_abscoeff_fromenergy(peaks_energy)

    def set_abscoeff(self, abscoeff = None):

        self.peaks_abscoeff_all = abscoeff

        self.peaks_abscoeff = abscoeff[self.peaks_assigned]

    def set_abscoeff_fromenergy(self, energy):

        self.peaks_energy_all = energy

        self.peaks_abscoeff_all = np.zeros((self.peakqty_all, self.get_wireqty()))

        for i, wire in enumerate(self.wires):

            self.peaks_abscoeff_all[:, i] = wire.calc_abscoeff(self.peaks_energy_all)

        self.peaks_abscoeff = [self.peaks_abscoeff_all[i, j] for i, j in enumerate(self.peaks_wireid)]

    def set_abscoeff_fromfitfile(self, fit_file, max_dist=10., default_energy=11.):

        list_x, list_y, list_energy = [], [], []

        if isinstance(fit_file, str):
            fit_file = [fit_file]

        for fn in fit_file:
            data = rwa.readfitfile_multigrains(fn)
            list_x.extend(data[4][:, 7])
            list_y.extend(data[4][:, 8])
            list_energy.extend(data[4][:, 6])

        list_xy = np.array(list(zip(list_x, list_y)))

        peaks_energy = np.ones(self.peakqty_all) * default_energy

        count = 0
        for k, pk in enumerate(self.peaks_all):

            mindist, idx = closest_point(pk, list_xy)

            if mindist < max_dist:
                peaks_energy[k] = list_energy[idx]
                count = count + 1

        print("energies were set for %d peaks." % count)

        self.set_abscoeff_fromenergy(peaks_energy)

    def set_abscoeff_fromWfilter(self, img0, time0, img1, time1, thickness):

        if isinstance(img0, str):
            img0, _, _ = rmccd.readCCDimage(img0, CCDLabel=self.scan.ccd_type)

        if isinstance(img1, str):
            img1, _, _ = rmccd.readCCDimage(img1, CCDLabel=self.scan.ccd_type)

        img0 = np.array(img0, dtype=float)
        img0 = np.transpose(img0) * time1

        img1 = np.array(img1, dtype=float)
        img1 = np.transpose(img1) * time0

        x0 = np.array(self.peaks_all[: ,0] - self.halfboxsize_all[: ,0], dtype=int)
        x1 = np.array(self.peaks_all[: ,0] + self.halfboxsize_all[: ,0] + 1, dtype=int)
        y0 = np.array(self.peaks_all[: ,1] - self.halfboxsize_all[: ,1], dtype=int)
        y1 = np.array(self.peaks_all[: ,1] + self.halfboxsize_all[: ,1] + 1, dtype=int)


        I0ref = [np.mean(img0[x0[i]:x1[i], y0[i]:y1[i]], axis=(0 ,1)) for i in range(len(x0))]
        I1ref = [np.mean(img1[x0[i]:x1[i], y0[i]:y1[i]], axis=(0 ,1)) for i in range(len(x0))]
        I0max = [ np.max(img0[x0[i]:x1[i], y0[i]:y1[i]], axis=(0 ,1)) for i in range(len(x0))]
        I1max = [ np.max(img1[x0[i]:x1[i], y0[i]:y1[i]], axis=(0 ,1)) for i in range(len(x0))]

        I0 = np.array(I0max) - np.array(I0ref)
        I1 = np.array(I1max) - np.array(I1ref) + 1E-6

        abscoeff = 1. / thickness * np.array([np.log(i0 / i1) for i0, i1 in zip(I0, I1)])

        self.set_abscoeff(abscoeff)

    def set_abscoeff_fromAlFilter(self, images, thickness, exposure=None, fsize=31):

        if exposure is None:
            exposure = [1.] * len(images)

        thickness = np.array(thickness)

        for i, expo in enumerate(exposure):
            images[i] = (np.array(images[i], dtype=np.float) - ndimage.filters.minimum_filter(images[i],
                                                                                              size=fsize)) / expo

        images = np.array(images, dtype=np.float)

        peaks_energy = np.ones(self.peakqty_all) * 10.

        muAl, energyAl, _ = abso.calc_absorption("Al", energy=np.arange(5, 25, 0.1), absolute=True)

        funAl = sp.interpolate.interp1d(muAl, energyAl, fill_value="extrapolate")

        for i, xy in enumerate(self.peaks_all):
            xmin, xmax, ymin, ymax = self.scan.clip_bbox(xy, self.halfboxsize_all[i])

            I0 = images[0, xmin:xmax, ymin:ymax]

            x, y = np.unravel_index(np.argmax(I0, axis=None), I0.shape)

            xmin, xmax, ymin, ymax = self.scan.clip_bbox([x + xmin, y + ymin], [2, 2])

            data_I = images[:, xmin:xmax, ymin:ymax]

            data_I = np.log(np.divide(np.maximum(data_I[0], 1.),
                                      np.maximum(data_I, 1.)))

            data_I = np.where(data_I > 0, data_I, 0)

            data_I = np.mean(data_I, axis=(1, 2))

            p = np.polyfit(thickness, data_I, deg=1)

            peaks_energy[i] = funAl(p[0])

        self.set_abscoeff_fromenergy(peaks_energy)

    # ----------------- halfboxsize -----------------

    def init_halfboxsize(self, hbs=5):

        self.set_halfboxsize(hbs)

    def set_halfboxsize(self, hbs):

        if isinstance(hbs, (float, int)):
            hbs = np.ones((self.peakqty_all, 2), dtype=int) * int(hbs)

        self.halfboxsize_all = np.array(hbs)

        self.halboxsize = self.halfboxsize_all[self.peaks_assigned]

    # ----------------- wire -----------------

    def set_wire_fromfile(self, filename):

        if not isinstance(filename, (list, tuple)):
            filename = [filename]

        for i, fn in enumerate(filename):
            self.wires[i].set_fromfile(fn)

        self.assign_wire_peaks()

    def assign_wire_peaks(self):

        self.peaks = []
        self.peakqty = 0
        self.peaks_assigned = []
        self.peaks_wireid = []
        self.peaks_abscoeff = []
        self.halfboxsize = []

        if self.peakqty_all:
            ylim = self.scan.calc_wires_range_scan(wire=self.wires)

            tmp = np.zeros((self.peakqty_all, self.get_wireqty() + 1), dtype=int)

            for j, lim in enumerate(ylim):
                tmp[:, j + 1] = (self.peaks_all[:, 1] - lim[0]) * (lim[1] - self.peaks_all[:, 1])

            wireid = np.argmax(tmp, axis=1) - 1

            self.peaks_assigned = (wireid >= 0)

            self.peaks = self.peaks_all[self.peaks_assigned]

            self.peakqty = len(self.peaks)

            self.peaks_wireid = wireid[self.peaks_assigned]

            if len(self.peaks_abscoeff_all):

                self.peaks_abscoeff = self.peaks_abscoeff_all[self.peaks_assigned]

                if self.peaks_abscoeff.ndim == 2:
                    self.peaks_abscoeff = [self.peaks_abscoeff[i, j] for i, j in enumerate(self.peaks_wireid)]

            if len(self.halfboxsize_all):
                self.halfboxsize = self.halfboxsize_all[self.peaks_assigned]

    def reconstruct(self, yrange, halfboxsize=None, ystep=0.001, nproc=1, save_spot=None, rec_args={}):

        print("Running reconstruction on %d cpus"%(nproc,))

        if nproc == 1:

            self.reconstruct_serial(yrange, halfboxsize, ystep, rec_args, save_spot)

        else:

            self.reconstruct_parallel(yrange, halfboxsize, ystep, nproc, rec_args, save_spot)

    def reconstruct_serial(self, yrange, halfboxsize=None, ystep=0.001, rec_args={}, save_spot=None):

        self.spots_rec = []
        # enumerate(self.peaks)
        for i, spot in enumerate(self.peaks):

            wireid = self.peaks_wireid[i]

            abscoeff = self.peaks_abscoeff[i]

            if halfboxsize is None:
                halfboxsize = self.halfboxsize[i]

            rec = SpotReconstructor(scan=self.scan,
                                    XYcam=spot,
                                    yrange=yrange,
                                    wire=self.wires[wireid],
                                    abscoeff=abscoeff,
                                    halfboxsize=halfboxsize)

            self.spots_rec.append(rec)

            self.reconstruct_spot(i, rec_args)

    def reconstruct_parallel(self, yrange, halfboxsize, ystep=0.001, nproc=2, rec_args={}, save_spot=None):

        self.spots_rec = []

        if halfboxsize is None:

            args = [(self.scan,
                     self.peaks[i],
                     yrange,
                     self.wires[self.peaks_wireid[i]],
                     self.peaks_abscoeff[i],
                     self.halfboxsize[i],
                     i, self.peakqty, rec_args, save_spot) for i in range(self.peakqty)]
        else:
            args = [(self.scan,
                     self.peaks[i],
                     yrange,
                     self.wires[self.peaks_wireid[i]],
                     self.peaks_abscoeff[i],
                     halfboxsize,
                     i, self.peakqty, rec_args, save_spot) for i in range(self.peakqty)]

        pool = mpi.Pool(nproc, maxtasksperchild=1)

        self.spots_rec = pool.map(fun_reconstruct_spot, args, chunksize=1)

        pool.close()

        pool.join()

    def reconstruct_spot(self, spot, reg_args={}):

        if not self.spots_rec[spot].grid_leftb and not self.spots_rec[spot].grid_rightb:
            recsize = self.spots_rec[spot].get_rec_size()

            print_msg("Reconstructing spot {}/{} with {}x{}x{} voxels...".format(spot + 1,
                                                                                 self.peakqty,
                                                                                 *recsize))

            self.spots_rec[spot].reconstruct(**reg_args)

    def generate_image_depth(self, y):

        img = np.zeros(self.scan.get_img_params(['framedim']))

        for spot in self.spots_rec:

            if spot.is_reconstructed:
                rec, _ = spot.interpolate(y)

                img[spot.grid_xlim[0]:spot.grid_xlim[1] + 1,
                spot.grid_ylim[0]:spot.grid_ylim[1] + 1] = rec

        return img

    def print_images(self, prefix, first_index=0, directory="", nbdigits=4, yrange=None, ystep=0.001):

        # try:
        #     print("Creating directory: {}".format(directory))
        #     os.mkdir(directory)
        # except OSError:
        #     pass

        if yrange is None:
            yrange = self.yrange

        ys = np.arange(yrange[0], yrange[1], ystep)

        fn_format = "{}_{{:0>{:d}d}}".format(prefix, nbdigits)

        for i, y in enumerate(ys):
            img = self.generate_image_depth(y)

            wimg.write_image(img.transpose(), fn_format.format(i + first_index),
                            CCDLabel=self.scan.ccd_type, dirname=directory, verbose=1)

        return len(ys)

    # Methods to print and plot
    def print_msg(self, msg, fmt=None, mode='I'):
        if self.verbose:

            if type == 'F':
                pref = ''
            elif type == 'E':
                pref = 'Error: '
            elif type == 'W':
                pref = 'Warning: '
            else:
                pref = ''

            if fmt is not None:
                msg = msg.format(*fmt)

            msg = "[rec] " + pref + msg

            if mode == 'F':
                raise RecError(msg)

            else:
                print(msg)


def print_msg(msg):
    print(msg)
    sys.stdout.flush()


def fun_reconstruct_spot(args):
    scan, XYcam, yrange, wire, abscoeff, halfboxsize, i, N, rec_args, save_spot = args

    rec = SpotReconstructor(scan=scan,
                            XYcam=XYcam,
                            yrange=yrange,
                            wire=wire,
                            abscoeff=abscoeff,
                            halfboxsize=halfboxsize)
    if not rec.grid_leftb and not rec.grid_rightb:
        recsize = rec.get_rec_size()

        print_msg("Reconstructing spot {}/{} with {}x{}x{} voxels...".format(i + 1, N,
                                                                             *recsize))

        rec.reconstruct(**rec_args)

    if save_spot is not None:

        fn = save_spot['filename']+"_%d_%d__%04d"%(XYcam[0], XYcam[1], i,)

        print_msg("Saving spot {}/{} to file %s".format(i + 1, N, fn))

        rec.to_npy(fn, save_spot['yres'], save_spot['ystep'])

    return rec
