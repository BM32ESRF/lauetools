#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os, sys, gc, time

import numpy as np
import scipy as sp


import multiprocessing as mpi
import scipy.ndimage as ndimage

import cProfile
import pstats
from io import StringIO


from LaueTools import readmccd as rmccd
from LaueTools import IOLaueTools as rwa


from LaueTools.Daxm.utils.geom import closest_point
import LaueTools.Daxm.utils.write_image as wimg
import LaueTools.Daxm.modules.segmentation2 as seg
import LaueTools.Daxm.material.absorption as abso

from LaueTools.Daxm.classes.reconstruction.spot import SpotReconstructor, RecError

mm=float

from typing import Dict, Tuple, Union, List

# --- Global cache for preloaded images (optional, if fun_reconstruct_spot is called from multiple places) ---
_preloaded_images = None


def preload_all_images(scan):
    """
    Preload all images from the scan into memory.
    This avoids repeated I/O operations during reconstruction.
    
    Parameters
    ----------
    scan : object
        The scan object containing the images.
    
    Returns
    -------
    dict
        A dictionary mapping image indices to preloaded image arrays.
    """
    global _preloaded_images
    if _preloaded_images is None:
        _preloaded_images = {}
        for i in range(scan.number_images):
            _preloaded_images[i] = scan.get_image_corr(i, exist=False)
    return _preloaded_images

class ScanReconstructor:
    """scan reconstructor is a series of spotreconstructor for individual peak with its own pixel region"""
    def __init__(self, scan, peaks_XY=None, peaks_energy=None, wires=None, verbose:int=0):

        if verbose>0:
            print('In ScanReconstructor() __init__()')
            print('wires: ', wires)

        self.scan = scan

        if wires is None:
            self.wires = scan.wire
        else:
            self.wires = wires

        self.init_peaks(peaks_XY, verbose=verbose-1)

        self.init_halfboxsize()

        self.init_abscoeff(peaks_energy)

    def free(self):

        del self.spots_rec[:]

        gc.collect()

    # getters
    def get_wireqty(self):
        """
        Return the number of wires used in the scan.
        """
        return len(self.wires)

        # ----------------- regions -----------------
    def set_regions(self, peaks_XY, halfboxsize, verbose=0):

        if verbose>0:
            print('In set_regions()')
        self.set_peaks(peaks_XY, verbose=verbose-1)

        self.set_halfboxsize(halfboxsize)

    def set_regions_fromsearch(self, I=None, max_size=100, min_size=3, thr=20, erode=2, dilate=2, merge=True, step:int=1, verbose=0):

        if verbose>0:
            print('In set_regions_fromsearch()')

        if I is None:
            I = self.scan.get_images_tophat(verbose=verbose-1, step=step)

        mask, _ = seg.apply_threshold(I, max_size, min_size, thr, erode, dilate)

        peaks_XY, peaks_hbs = seg.draw_bbox(I, mask, merge)

        self.set_regions(peaks_XY, peaks_hbs)

        return mask

    # ----------------- peaks -----------------

    def init_peaks(self, peaks_XY=None, verbose=0):

        if peaks_XY is None:

            peaks_XY = []

        self.set_peaks(peaks_XY, verbose=verbose-1)

    def set_peaks(self, peaks_XY: Union[str, np.ndarray], verbose: int=0):

        if verbose>0:
            print('In set_peaks()')
        if isinstance(peaks_XY, str):  # load from file

            self.set_peaks_fromfitfile(peaks_XY)

        else:  # load from array

            self.peaks_all = peaks_XY

            self.peakqty_all = len(self.peaks_all)

            self.peaks_abscoeff_all = np.zeros((self.peakqty_all, self.get_wireqty()))

            self.halfboxsize_all = np.ones((self.peakqty_all, 2), dtype=int)

            self.assign_wire_peaks(verbose=verbose-1)

    def set_peaks_fromfitfile(self, fit_file, usecols=[7, 8], skiprows=5):

        peaks_XY = np.loadtxt(fit_file, skiprows=skiprows, usecols=usecols)

        self.set_peaks(peaks_XY)

    def set_peaks_fromsearch(self, image_file=None, threshold=150, fit_peaks=1):

        """
        Find peaks in an image using PeakSearch from rmccd.

        use finally self.set_peaks(peaks_XY)
        
        Parameters
        ----------
        image_file : str or None
            If None, use the first image from the scan.
        threshold : int
            The IntensityThreshold for PeakSearch.
        fit_peaks : int
            If 1, peaks are fitted with a Gaussian.
        
        Returns
        -------
        None
        """
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
        """
        Initialize WIRE absorption coefficients for each peak.

        Parameters
        ----------
        peaks_energy : array or None
            Energies associated with each peak in keV.
            If None, then set to DEFAULT_PEAK_ENERGY (default 10 keV) for all peaks.

        Notes
        -----
        Absorption coefficients are calculated for each peak based on the
        energy specified and the material properties of the wires.
        """
        DEFAULT_PEAK_ENERGY = 10.  # keV

        if peaks_energy is None:

            peaks_energy = np.ones(self.peakqty_all) * DEFAULT_PEAK_ENERGY

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

    def set_abscoeff_fromfitfile(self, fit_file, max_dist=10., default_energy=11., usecols=[7, 8, 14]):
        """warning  maybe columns for x y energy are wrong, look at dict_column_header of readfitfile_multigrains"""

        print('!! check column for x y energy !!')

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
        """not called anywhere"""

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
        """not called anywhere"""

        if exposure is None:
            exposure = [1.] * len(images)

        thickness = np.array(thickness)

        for i, expo in enumerate(exposure):
            images[i] = (np.array(images[i], dtype=np.float32) - ndimage.filters.minimum_filter(images[i],
                                                                                              size=fsize)) / expo

        images = np.array(images, dtype=np.float32)

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

    def assign_wire_peaks(self, verbose:int=0):

        self.peaks = []
        self.peakqty = 0
        self.peaks_assigned = []
        self.peaks_wireid = []
        self.peaks_abscoeff = []
        self.halfboxsize = []

        if verbose>0:
            print("Assigning wires to peaks...")

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
            if verbose>0:
                print("End of Assigning wires to peaks...")

    def reconstruct(self, yrange, halfboxsize=None, ystep=0.001, nproc=1, save_spot=None, rec_args={}, verbose:int=0):

        if verbose>0:
            print("In reconstruct() of classes/reconstruction/scan.py.\n Running reconstruction on %d cpus"%(nproc,))

        if nproc == 1:

            self.reconstruct_serial(yrange, halfboxsize, ystep, rec_args, save_spot)

        else:

            self.reconstruct_parallel(yrange, halfboxsize, ystep, nproc, rec_args, save_spot, verbose=verbose-1)


    def reconstruct_serial(self, yrange, halfboxsize=None, ystep:mm=0.001, rec_args:dict={}, save_spot=None):

        self.spots_rec = []
        # enumerate(self.peaks)
        for i, spot in enumerate(self.peaks):

            wireid = self.peaks_wireid[i]

            abscoeff = self.peaks_abscoeff[i]

            if halfboxsize is None:
                halfboxsize = self.halfboxsize[i]

            spotrec = SpotReconstructor(scan=self.scan,
                                    XYcam=spot,
                                    yrange=yrange,
                                    wire=self.wires[wireid],
                                    abscoeff=abscoeff,
                                    halfboxsize=halfboxsize)

            self.spots_rec.append(spotrec)

            self.reconstruct_spot(i, rec_args)


    def reconstruct_parallel(self, yrange, halfboxsize, ystep=0.001, nproc=2, rec_args={}, save_spot=None, verbose:int=0):
        if verbose > 0:
            print("In reconstruct_parallel() of classes/reconstruction/scan.py.")
            print(f'Number of peaks to reconstruct: {self.peakqty}')

        # --- Preload all images into memory ---
        if verbose > 0:
            print("[rec] Preloading all images...")
        start_time = time.time()
        preloaded_images = {}
        for i in range(self.scan.number_images):
            preloaded_images[i] = self.scan.get_image_corr(i, exist=False)
        if verbose > 0:
            print(f"[rec] Preloaded all images in {time.time() - start_time:.2f} seconds")

        # --- Prepare args for fun_reconstruct_spot ---
        self.spots_rec = []

        #wire_or_wires = self.wires[self.peaks_wireid[i]]
        wire_or_wires = self.wires
        if halfboxsize is None:
            args = [
                (self.scan, self.peaks[i], yrange, wire_or_wires,
                self.peaks_abscoeff[i], self.halfboxsize[i], i, self.peakqty,
                rec_args, save_spot, verbose, preloaded_images)
                for i in range(self.peakqty)
            ]
        else:
            args = [
                (self.scan, self.peaks[i], yrange, wire_or_wires,
                self.peaks_abscoeff[i], halfboxsize, i, self.peakqty,
                rec_args, save_spot, verbose, preloaded_images)
                for i in range(self.peakqty)
            ]

        # --- Run sequentially (no pickling issues) ---
        if verbose > 0:
            print("\n\n[rec] Running reconstruction...")
        start_time = time.time()
        self.spots_rec = [fun_reconstruct_spot(arg) for arg in args]
        if verbose > 0:
            print(f"[rec] Reconstruction completed in {time.time() - start_time:.2f} seconds")

        return self.spots_rec
    def reconstruct_parallel_lastold(self, yrange, halfboxsize, ystep=0.001, nproc=2, rec_args={}, save_spot=None, verbose:int=0):
        """
        Reconstructs peaks in parallel, with preloaded images to avoid I/O bottleneck.
        
        Parameters
        ----------
        self : object
            The object containing scan, peaks, wires, etc.
        yrange : tuple
            The range of depths for reconstruction.
        halfboxsize : list or None
            Half-box sizes for peak regions. If None, uses self.halfboxsize.
        ystep : float, optional
            Step size for depth range. Default is 0.001.
        nproc : int, optional
            Number of processes for parallel reconstruction. Default is 2.
        rec_args : dict, optional
            Additional arguments for reconstruction. Default is {}.
        save_spot : object, optional
            Object to save spot data. Default is None.
        verbose : int, optional
            Verbosity level. Default is 0.
        
        Returns
        -------
        list
            List of reconstructed spots.
        """
        if verbose > 0:
            print("In reconstruct_parallel() of classes/reconstruction/scan.py.")
            print(f'Number of peaks to reconstruct: {self.peakqty}')

        if verbose > 0: 
            print('self.peaks_wireid',self.peaks_wireid)
            print('self.wires',self.wires)
            print('[self.wires[self.peaks_wireid[i]] for i in range(self.peakqty)]',[self.wires[self.peaks_wireid[i]] for i in range(self.peakqty)])

        # --- Preload all images into memory ---
        if verbose > 0:
            print("[rec] Preloading all images...")
        start_time = time.time()
        preloaded_images = preload_all_images(self.scan)
        if verbose > 0:
            print(f"[rec] Preloaded all images in {time.time() - start_time:.2f} seconds")

        # --- Prepare arguments for fun_reconstruct_spot ---
        self.spots_rec = []

        if halfboxsize is None:
            args = [
                (self.scan, self.peaks[i], yrange, self.wires[self.peaks_wireid[i]],
                self.peaks_abscoeff[i], self.halfboxsize[i], i, self.peakqty,
                rec_args, save_spot, verbose, preloaded_images)
                for i in range(self.peakqty)
            ]
        else:
            args = [
                (self.scan, self.peaks[i], yrange, self.wires[self.peaks_wireid[i]],
                self.peaks_abscoeff[i], halfboxsize, i, self.peakqty,
                rec_args, save_spot, verbose, preloaded_images)
                for i in range(self.peakqty)
            ]

        # --- Run sequentially (I/O is no longer the bottleneck) ---
        if verbose > 0:
            print("[rec] Running reconstruction...")
        start_time = time.time()
        self.spots_rec = [fun_reconstruct_spot(arg) for arg in args]
        if verbose > 0:
            print(f"[rec] Reconstruction completed in {time.time() - start_time:.2f} seconds")

        return self.spots_rec

    def reconstruct_parallel_withprofile(self, yrange, halfboxsize, ystep=0.001, nproc=2, rec_args={}, save_spot=None, verbose:int=0):
        if verbose > 0:
            print("In reconstruct_parallel() of classes/reconstruction/scan.py.")
            print('nb of peaks to reconstruct: "self.peakqty" ', self.peakqty)

        self.spots_rec = []
        if halfboxsize is None:
            args = [(self.scan, self.peaks[i], yrange, self.wires[self.peaks_wireid[i]],
                    self.peaks_abscoeff[i], self.halfboxsize[i], i, self.peakqty, rec_args, save_spot, verbose)
                    for i in range(self.peakqty)]
        else:
            args = [(self.scan, self.peaks[i], yrange, self.wires[self.peaks_wireid[i]],
                    self.peaks_abscoeff[i], halfboxsize, i, self.peakqty, rec_args, save_spot, verbose)
                    for i in range(self.peakqty)]

        # --- Profile all calls to fun_reconstruct_spot ---
        pr = cProfile.Profile()
        pr.enable()
        self.spots_rec = [fun_reconstruct_spot(arg) for arg in args]
        pr.disable()

        # Print profiling results
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        print("\n--- Profiling Results for fun_reconstruct_spot ---")
        print(s.getvalue())

        return self.spots_rec

    def reconstruct_parallel_old(self, yrange, halfboxsize, ystep=0.001, nproc=2, rec_args={}, save_spot=None, verbose:int=0):

        if verbose>0:
            print("In reconstruct_parallel() of classes/reconstruction/scan.py.")
            print('nb of peaks to reconstruct: "self.peakqty" ', self.peakqty)

        self.spots_rec = []

        if halfboxsize is None:

            args = [(self.scan,
                     self.peaks[i],
                     yrange,
                     self.wires[self.peaks_wireid[i]],
                     self.peaks_abscoeff[i],
                     self.halfboxsize[i],
                     i, self.peakqty, rec_args, save_spot, verbose) for i in range(self.peakqty)]
        else:
            args = [(self.scan,
                     self.peaks[i],
                     yrange,
                     self.wires[self.peaks_wireid[i]],
                     self.peaks_abscoeff[i],
                     halfboxsize,
                     i, self.peakqty, rec_args, save_spot, verbose) for i in range(self.peakqty)]

        # scan, peak, yrange, wire, abscoeff, halfboxsize, 
        # idx, peakqty, rec_args, save_spot, verbose, preloaded_images
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

    def generate_image_depth(self, y, verbose:int=0):

        if verbose>0:
            print("In generate_image_depth() of classes/reconstruction/scan.py.")

        img = np.zeros(self.scan.get_img_params(['framedim']))

        for spot in self.spots_rec:

            if spot.is_reconstructed:
                rec, _ = spot.interpolate(y)

                img[spot.grid_xlim[0]:spot.grid_xlim[1] + 1,
                spot.grid_ylim[0]:spot.grid_ylim[1] + 1] = rec

        return img

    def print_images(self, prefix, first_index:int=0, directory="", nbdigits:int=4,
                            yrange=None, ystep:mm=0.001,
                            templateh5filename=None,verbose:int=0):

        # try:
        #     print("Creating directory: {}".format(directory))
        #     os.mkdir(directory)
        # except OSError:
        #     pass
        if 1: #verbose>0:
            print("In print_images() of classes/reconstruction/scan.py.")
            print('self.scan.ccd_type', self.scan.ccd_type)
        # for eiger: needs to use a template h5 file
        if self.scan.ccd_type == 'EIGER_4MCdTe':
            # one (first) image of the scan
            templateh5filename = os.path.join(self.scan.img_folder, self.scan.img_filenames[0])


        if yrange is None:
            yrange = self.yrange

        ys = np.arange(yrange[0], yrange[1], ystep)

        fn_format = "{}_{{:0>{:d}d}}".format(prefix, nbdigits)

        for i, y in enumerate(ys):
            img = self.generate_image_depth(y, verbose=verbose-1)

            wimg.write_image(img.transpose(), fn_format.format(i + first_index),
                            CCDLabel=self.scan.ccd_type, dirname=directory, templateh5filename=templateh5filename,verbose=verbose)

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




import numpy as np
from scipy import optimize as spo
from numba import jit
import time

# --- Numba-optimized functions ---
@jit(nogil=True, fastmath=True)
def calc_crosslength_optimized(
    pw_x: float, pw_y: float, pw_z: float,
    ysrc: float,
    Pcam_x: float, Pcam_y: float, Pcam_z: float,
    wire_axis_x: float, wire_axis_y: float, wire_axis_z: float,
    wire_R: float
) -> float:
    """
    Numba-optimized version of calc_crosslength.
    All inputs are scalars (floats) for Numba compatibility.
    """
    # OY = Ysrc - pw (vector from pw to ysrc)
    OYx, OYy, OYz = -pw_x, ysrc - pw_y, -pw_z

    # Components of v = YP / |YP|
    vx, vy, vz = Pcam_x, Pcam_y - ysrc, Pcam_z
    vn = np.sqrt(vx**2 + vy**2 + vz**2)
    vx, vy, vz = vx / vn, vy / vn, vz / vn

    # Cross product v × wire_axis
    vfx = vy * wire_axis_z - vz * wire_axis_y
    vfy = vz * wire_axis_x - vx * wire_axis_z
    vfz = vx * wire_axis_y - vy * wire_axis_x
    vf2 = vfx**2 + vfy**2 + vfz**2

    # Cross product OY × wire_axis
    Ofx = OYy * wire_axis_z - OYz * wire_axis_y
    Ofy = OYz * wire_axis_x - OYx * wire_axis_z
    Ofz = OYx * wire_axis_y - OYy * wire_axis_x
    Of2 = Ofx**2 + Ofy**2 + Ofz**2

    # Quadratic equation: A*d² + B*d + C = 0
    A = vf2
    B = 2.0 * (Ofx * vfx + Ofy * vfy + Ofz * vfz)
    C = Of2 - wire_R**2
    Delta = B**2 - 4.0 * A * C

    # Traveled distance in the wire
    if A == 0:
        return 0.0
    abslength = np.sqrt(np.maximum(Delta, 0.0)) / A
    return abslength

@jit(nogil=True, fastmath=True)
def calc_transmission_optimized(
    pw_x: float, pw_y: float, pw_z: float,
    ysrc: float,
    Pcam_x: float, Pcam_y: float, Pcam_z: float,
    wire_axis_x: float, wire_axis_y: float, wire_axis_z: float,
    wire_R: float,
    abscoeff: float
) -> float:
    """
    Numba-optimized version of calc_transmission.
    """
    length = calc_crosslength_optimized(
        pw_x, pw_y, pw_z, ysrc,
        Pcam_x, Pcam_y, Pcam_z,
        wire_axis_x, wire_axis_y, wire_axis_z,
        wire_R
    )
    return np.exp(-abscoeff * length)

def fun_reconstruct_spot(args):
    """
    Reconstructs a single spot using preloaded images and optimized CPU-bound functions.
    """
    (scan, peak, yrange, wire, abscoeff, halfboxsize,
     idx, peakqty, rec_args, save_spot, verbose, preloaded_images) = args

    if verbose > 0:
        start_time = time.time()
        print(f"[fun_reconstruct_spot] Processing peak {idx+1}/{peakqty}")
        print('wire: ', wire)

    # --- Initialize SpotReconstructor ---
    spot = SpotReconstructor(scan=scan,
                                XYcam=peak,
                                halfboxsize=halfboxsize,
                                yrange=yrange,
                                abscoeff=abscoeff,
                                wire=wire,
                                verbose=verbose-1,
                            )

    # --- Replace I/O-bound operations with preloaded_images ---
    if hasattr(spot, 'frames_cor'):
        spot.frames_cor = [preloaded_images[i] for i in range(len(spot.frames_cor))]

    # --- Extract wire attributes for Numba ---
    wire_axis = np.array(wire.axis) if hasattr(wire, 'axis') else np.array([0.0, 0.0, 1.0])
    wire_R = float(wire.R) if hasattr(wire, 'R') else 1.0

    # --- Monkey-patch wire methods to use optimized Numba functions ---
    if hasattr(wire, 'calc_transmission') and hasattr(wire, 'calc_crosslength'):
        original_calc_transmission = wire.calc_transmission
        original_calc_crosslength = wire.calc_crosslength

        # Wrapper for calc_transmission
        def calc_transmission_wrapper(pw, ysrc, Pcam, energy=None, abscoeff=None):
            # Extract components from pw and Pcam (assuming they are arrays or objects)
            pw_x, pw_y, pw_z = pw[0], pw[1], pw[2] if hasattr(pw, '__len__') else (pw.x, pw.y, pw.z)
            Pcam_x, Pcam_y, Pcam_z = Pcam[0], Pcam[1], Pcam[2] if hasattr(Pcam, '__len__') else (Pcam.x, Pcam.y, Pcam.z)
            wire_axis_x, wire_axis_y, wire_axis_z = wire_axis[0], wire_axis[1], wire_axis[2]
            return calc_transmission_optimized(
                pw_x, pw_y, pw_z, ysrc,
                Pcam_x, Pcam_y, Pcam_z,
                wire_axis_x, wire_axis_y, wire_axis_z,
                wire_R,
                abscoeff if abscoeff is not None else original_calc_transmission(pw, ysrc, Pcam, energy, abscoeff)
            )

        # Wrapper for calc_crosslength
        def calc_crosslength_wrapper(pw, ysrc, Pcam):
            pw_x, pw_y, pw_z = pw[0], pw[1], pw[2] if hasattr(pw, '__len__') else (pw.x, pw.y, pw.z)
            Pcam_x, Pcam_y, Pcam_z = Pcam[0], Pcam[1], Pcam[2] if hasattr(Pcam, '__len__') else (Pcam.x, Pcam.y, Pcam.z)
            wire_axis_x, wire_axis_y, wire_axis_z = wire_axis[0], wire_axis[1], wire_axis[2]
            return calc_crosslength_optimized(
                pw_x, pw_y, pw_z, ysrc,
                Pcam_x, Pcam_y, Pcam_z,
                wire_axis_x, wire_axis_y, wire_axis_z,
                wire_R
            )

        wire.calc_transmission = calc_transmission_wrapper
        wire.calc_crosslength = calc_crosslength_wrapper

    # --- Run reconstruction with optimized methods ---
    spot.reconstruct(
        regularize=rec_args.get('regularize', False),
        reg_alpha=rec_args.get('reg_alpha', 0.5),
        reg_method=rec_args.get('reg_method', 'ridge'),
        oversamp=rec_args.get('oversamp', 20),
        verbose=verbose-1
    )

    # --- Restore original wire methods (optional) ---
    if hasattr(wire, 'calc_transmission'):
        wire.calc_transmission = original_calc_transmission
    if hasattr(wire, 'calc_crosslength'):
        wire.calc_crosslength = original_calc_crosslength

    # --- Save spot if needed ---
    if save_spot is not None:
        save_spot(spot)

    if verbose > 0:
        print(f"[fun_reconstruct_spot] Peak {idx} processed in {time.time() - start_time:.2f} seconds")

    return spot


def fun_reconstruct_spot_oldo(args):
    """
    Reconstructs a single spot using preloaded images to avoid I/O overhead.
    
    Parameters
    ----------
    args : tuple
        Tuple containing all required arguments for reconstruction:
        (
            scan,               # Scan object
            peak,               # Peak data  XYcam (single pixel  [X, Y])
            yrange,             # Depth range for reconstruction
            wires,              # Wires data
            abscoeff,          # Absorption coefficient
            halfboxsize,       # Half-box size for peak region
            idx,                # Peak index
            peakqty,           # Total number of peaks
            rec_args,          # Reconstruction arguments
            save_spot,         # Object to save spot data
            verbose,           # Verbosity level
            preloaded_images   # Dict: {frame_index: image_array}
        )
    
    Returns
    -------
    SpotReconstructor
        The reconstructed spot object.
    """
    (
        scan, peak, yrange, wire, abscoeff, halfboxsize, 
        idx, peakqty, rec_args, save_spot, verbose, preloaded_images
    ) = args

    # --- Start timer for debugging ---
    start_time = time.time()
    if verbose > 0:
        print('In fun_reconstruct_spot() of classes/reconstruction/scan.py.')
        print(f"[fun_reconstruct_spot] Processing peak {idx+1}/{peakqty}")
        print('wire', wire)


    recspot = SpotReconstructor(scan, peak, yrange, wire, abscoeff, halfboxsize, verbose=verbose-1)
    if verbose > 0:
        print('recspot object created from class SpotReconstructor')

    # --- Replace I/O-bound operations with preloaded_images ---
    # Original code likely calls something like:
    #   img = scan.get_image_rect_corr(frame, x1, x2, y1, y2)
    # Replace with direct access to preloaded_images:
    
    # Example: If your code previously did this in init_data_general:
    #   for frame in frame_range:
    #       img = scan.get_image_rect_corr(frame, x1, x2, y1, y2)
    # Replace with:
    #   for frame in frame_range:
    #       img = preloaded_images[frame][y1:y2, x1:x2]
    
    # --- Call init_data_general with preloaded images ---
    # Assuming init_data_general is where I/O happens, modify it to use preloaded_images
    # Here, we explicitly handle the image loading part
    
    # Example: If init_data_general uses get_images_rect_corr, replace it like this:
    def init_data_general_with_preload(self, yrange, wire, halfboxsize, preloaded_images):
        """
        Modified version of init_data_general that uses preloaded_images.
        """
        # Example: Replace this I/O-bound loop:
        # for frame in range(yrange[0], yrange[1], ystep):
        #     img = self.scan.get_image_rect_corr(frame, x1, x2, y1, y2)
        
        # With this:
        for frame in range(int(yrange[0]), int(yrange[1]), 1):  # Adjust step as needed
            # Extract the region of interest from preloaded_images
            x1, x2, y1, y2 = 0, self.scan.img_shape[1], 0, self.scan.img_shape[0]  # Adjust based on your logic
            img = preloaded_images[frame][y1:y2, x1:x2]  # Direct array slicing
            
            # Process img as needed (e.g., store in self.img or similar)
            # This is where you'd integrate with your existing logic
            
    # Call the modified init_data_general
    # Note: You may need to adjust this based on your actual SpotReconstructor class
    recspot.init_data_general = lambda: init_data_general_with_preload(
        spot, yrange, wire, halfboxsize, preloaded_images
    )
    recspot.init_data_general()

    # # --- Assign wire peaks (no I/O, keep as-is) ---
    # spot.assign_wire_peaks()

    # --- Reconstruct (CPU-bound, keep as-is) ---
    recspot.reconstruct(**rec_args, verbose=verbose-1)

    # --- Save spot if needed ---
    if save_spot is not None:
        save_spot(recspot)

    # --- Debug timing ---
    if verbose > 0:
        print(f"[fun_reconstruct_spot] Peak {idx} processed in {time.time() - start_time:.2f} seconds")

    return recspot

def fun_reconstruct_spot_old(args):
    scan, XYcam, yrange, wire, abscoeff, halfboxsize, i, N, rec_args, save_spot, verbose = args

    rec = SpotReconstructor(scan=scan,
                            XYcam=XYcam,
                            yrange=yrange,
                            wire=wire,
                            abscoeff=abscoeff,
                            halfboxsize=halfboxsize, verbose=verbose-1)

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
