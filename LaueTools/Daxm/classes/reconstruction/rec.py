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

        self.fitfile = None   # str or None, path to .fit file to assign energy to pixels close to peaks in .fit file

        self.grid_ix = []
        self.grid_iy = []

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

    def reconstruct(self, depth_range, fileprefix, depth_step=0.001, nproc:int=1, directory="", rec_par={}, depth_range_print=None, addscan0001=False, usefitfiles_peaks=False):

        """
        Reconstructs a series of 2D images around peaks.

        use self.fitfile to assign energy to peaks
        use

        Parameters
        ----------
        depth_range : tuple of two floats
            The range of depths to reconstruct the image for.
        fileprefix : str
            The prefix of the output image file names.
        depth_step : float, optional
            The step of depths to reconstruct the image for. Defaults to 0.001.
        nproc : int, optional
            The number of processes to use for the reconstruction. Defaults to 1.
        directory : str, optional
            The directory where the output image files will be saved. Defaults to "".
        rec_par : dict, optional
            The parameters for the reconstruction. Defaults to {}.
        depth_range_print : tuple of two floats, optional
            The range of depths to print the reconstructed image for. Defaults to None.
        usefitfiles_peaks : bool, optional
            If True, consider only pixels for reconstruction that are in a bounding box centered on peaks listed in the fitfile specified in `self.fitfile`. Default is False. 

        Returns
        -------
        None
        """
        DEFAULT_YSTEP = 0.001

        grid_depth = np.arange(depth_range_print[0], depth_range_print[1], depth_step)

        imgqty_per_scan = len(grid_depth)

        try:
            imgqty = imgqty_per_scan * self.scan.size[0] * self.scan.size[1]
            img_idx = np.zeros(self.scan.size, dtype=int)
        except AttributeError:
            imgqty = imgqty_per_scan
            img_idx = np.array([0,], dtype=int)

        

        ndigits = 4#
        #ndigits = calc_ndigits(imgqty)

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
                # consider pixels in a bounding box centered on peaks (no opencv segmentation)
                if usefitfiles_peaks==True:  #  seg.par not
                    default_hbs = [12,12]
                    
                    print("[rec] > Optional read of fitfile(s) for peaks")
                    from LaueTools import IOLaueTools as rwa
                    #Create list of peaks
                    peaks_XY = []
                    #Since the nature of 'self.fitfile' is not explicit we can define 2 cases
                    if isinstance (self.fitfile,list):
                        #fn = Single fit_file in 'self.fitfile'
                        for _k , fn in enumerate(self.fitfile):
                            #Alternate reusing code from 'scan.py set_abscoeff_fromfitfile'
                            data = rwa.readfitfile_multigrains(fn)[4] #20250613 - Pick element for blc15488/ech15_Z1 p25um files
                            
                           
                            if _k > 0:
                                _d = np.concatenate((_d,data[:,7:9]))
                            else:
                                _d = data[:,7:9]
                        peaks_XY = np.array(_d)
                        
                    else:
                        data = rwa.readfitfile_multigrains(self.fitfile) #data type <class 'tuple'>

                        #CAUTION: Data = Tuple --> Extract element [4]
                        peaks_XY = data[4][:,7:9]
                        #For Example GOI fit file SHORT: Expected - TBC [array([[ 111.07, 1790.09], [1924.01, 1316.77]])]


                    print('*******')
                    # print('Test peaks_XY type', type(peaks_XY))
                    print('peaks_XY', peaks_XY)
                    print('*******')

                    #CAUTION halfboxsize must be a List - Default halfboxsize proposed
                    #Create default spot bounding box for all peaks - Same format than expected for ScanReconstructor
                    halfboxsize = []
                    
                    for _ in range(peaks_XY.shape[0]):
                        halfboxsize.append(default_hbs)
                    print('*******')
                    # print('Test halfboxsize type', type(halfboxsize))
                    print('Test halfboxsize', halfboxsize)
                    print('*******')
                    #Set regions (peaks + bounding box) on which to run the reconstruction
                    rec.set_regions(peaks_XY,halfboxsize)

                #Standard method by Renversade et Molin: use opencv segmentation to find pixels for reconstruction. Pixels are gathered by peaks (with varying bounding box size)
                else:
                    rec.set_regions_fromsearch(**self.seg)

                if self.fitfile is None:
                    rec.init_abscoeff()
                else:
                    rec.set_abscoeff_fromfitfile(self.fitfile)

                rec.assign_wire_peaks()

                rec.reconstruct(yrange=depth_range, halfboxsize=None, ystep=DEFAULT_YSTEP, nproc=nproc, rec_args=rec_par)

                if len(self.grid_x)>1:
                    rec.print_images(prefix=fileprefix, first_index=img_idx[ix][iy], directory=directory, yrange=depth_range_print, nbdigits=ndigits)
                else:
                    rec.print_images(prefix=fileprefix, first_index=0, directory=directory, yrange=depth_range_print, nbdigits=ndigits)

                rec.free()

                print("[rec] elapsed time = %s seconds" % (time.time() - start_time))
