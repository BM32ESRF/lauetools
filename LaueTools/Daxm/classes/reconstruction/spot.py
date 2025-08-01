#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np

import scipy.optimize as spo


import LaueTools.Daxm.classes.wire as mywire

import LaueTools.Daxm.modules.geometry as geom

from LaueTools.Daxm.utils.list import is_list_of


# Class definitions
class RecError(Exception):
    pass


# Spot reconstruction class
class SpotReconstructor:
    # Constructors
    def __init__(self, scan, XYcam, halfboxsize, yrange, abscoeff, wire=None, verbose=True):

        # inputs
        self.verbose = verbose

        self.scan = scan  # corresponding DAXM (wires) scan

        self.XYcam = XYcam  # 2D center pixel coordinates of the ROI (spot)

        self.wire = wire  # corresponding wire(s)

        self.yrange = yrange # y // beam   range (2 values) of depth along the beam (in mm)

        self.abscoeff = abscoeff  # absorption coefficient of wire

        self.hbs = halfboxsize

        # init attributes
        self.Pcam = []

        self.grid_leftb = []
        self.grid_rightb = []
        self.grid_xlim = []
        self.grid_ylim = []
        self.grid_nx = 0
        self.grid_ny = 0
        self.grid_x = []
        self.grid_y = []
        self.grid_Pcam = []
        # wire positions (mm)
        self.pw_all = []  # all motor positions (yf, zf) in mm holding the wire(s)
        self.pw_idx = []
        self.pw = []

        self.yw = []
        self.ygrid = []

        self.frames_ini = []  # raw images frames of the given ROI
        self.frames_bkg = []  # background images frames of the given ROI. linear along X pixel and constant along Y pixel
        self.frames_cor = [] # Xlinear-Yconstant background-corrected images frames

        self.rec = []
        self.is_reconstructed = False

        # init
        self.init_hbs()

        self.init_wire()

        self.init_grid()

        self.init_depth(verbose=verbose)

        self.init_data()

    def init_hbs(self):
        """
        Initialize the ROI halfboxsize attribute of the class.

        If self.hbs is a tuple, convert it to a list. If self.hbs is an int, use it as the half box size for the x and y directions. If self.hbs is a list or tuple of two elements, use the first element as the x half box size and the second element as the y half box size.

        set self.hbs
        """
        if isinstance(self.hbs, tuple):
            self.hbs = list(self.hbs)

        if not isinstance(self.hbs, (list, tuple)) and isinstance(self.hbs, int):
            self.hbs = [self.hbs] * 2

    def init_wire(self):
        """
        Initialize the wire attribute of the class. 

        If self.wire is None, use the wire with the highest available information in scan data for a given Y pixel position (XYcam[1]). If self.wire is a list of CircularWire, use the first one. If self.wire is an int, use it as the index of the wire in self.scan.wire. If self.wire is a CircularWire, use it as is. Otherwise, raise an error.

        set self.wire
        """
        if self.wire is None or is_list_of(self.wire, mywire.CircularWire):
            self.init_wire_assign()

        elif isinstance(self.wire, int):
            self.wire = self.scan.wire[self.wire]

        elif isinstance(self.wire, mywire.CircularWire):
            pass

        else:
            self.print_msg("Invalid wire argument", mode='F')

    def init_wire_assign(self, margin=0):
        """determine and the wire withthe  highest available information in scan data for a given Y pixel position"""
        if is_list_of(self.wire, mywire.CircularWire):
            wire = self.wire
        else:
            wire = self.scan.wire

        # Y pixel boundaries for each wire 
        Yshadow_limits_list = self.scan.calc_wires_range_scan(wire=wire)

        tmp = []
        Ypixel = self.XYcam[1]
        for Yshadow_limits in Yshadow_limits_list:
            yshadow_min, Yshadow_max = Yshadow_limits
            tmp.append((Ypixel - yshadow_min) * (Yshadow_max - Ypixel))
        
        assigned_wireindex = np.argmax(tmp)
        self.wire = wire[assigned_wireindex]

        return assigned_wireindex

    def init_grid(self):
        """init grid of pixel points lying on detector plane
        
        set self.grid_Pcam"""

        par = self.scan.get_ccd_params()

        # center
        self.Pcam = geom.transf_pix_to_coo(par, *self.XYcam)

        # actual xy-limits
        xmin, xmax = int(self.XYcam[0] - self.hbs[0]), int(self.XYcam[0] + self.hbs[0])
        ymin, ymax = int(self.XYcam[1] - self.hbs[1]), int(self.XYcam[1] + self.hbs[1])

        # stuck to left or right sides
        self.grid_rightb = (xmin <= 0)
        self.grid_leftb = (xmax >= self.scan.get_img_params(['framedim'])[0] - 1)

        xmin, xmax = max(0, xmin), min(self.scan.get_img_params(['framedim'])[0] - 1, xmax)
        ymin, ymax = max(0, ymin), min(self.scan.get_img_params(['framedim'])[1] - 1, ymax)

        # grid
        self.grid_xlim = [xmin, xmax]
        self.grid_ylim = [ymin, ymax]

        self.grid_nx = xmax - xmin + 1
        self.grid_ny = ymax - ymin + 1

        self.grid_x = np.arange(xmin, xmax + 1, dtype=int)
        self.grid_y = np.arange(ymin, ymax + 1, dtype=int)

        self.grid_Pcam = np.zeros((self.grid_nx, self.grid_ny, 3), dtype=float)

        for ix, x in enumerate(self.grid_x):
            for iy, y in enumerate(self.grid_y):
                self.grid_Pcam[ix, iy] = geom.transf_pix_to_coo(par, x, y)

    def init_depth(self, verbose=False):
        """
        Initialize the wire position and corresponding depth position attributes of the class.

        Calculates the motor positions of the wire(s) that cast a shadow on the entire ROI and their corresponding depth positions.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints out the calculated motor positions and depth positions.

        Sets the following attributes:
        - pw_all : 1D array of motor positions of the wire(s)
        - pw_idx : 1D array of indices of pw_all, where the wire(s) cast a shadow on the entire ROI
        - pw : 1D array of motor positions of the wire(s) that cast a shadow on the entire ROI
        - yw : 1D array of depth positions of the wire(s) corresponding to pw
        - ygrid : 1D array of interpolated middle values of yw
        """
        # motor pos of wire

        self.pw_all = self.scan.wire_position

        pfall, pball = [], []
       
        par = self.scan.get_ccd_params()

        c1 = np.subtract(self.XYcam, [-self.hbs[0], -self.hbs[1]])
        c2 = np.subtract(self.XYcam, [-self.hbs[0], self.hbs[1]])
        c3 = np.subtract(self.XYcam, [self.hbs[0], -self.hbs[1]])
        c4 = np.subtract(self.XYcam, [self.hbs[0], self.hbs[1]])
        corners = [c1, c2, c3, c4]
	
        for c in corners:
            P = geom.transf_pix_to_coo(par, *c)
            pf, pb = self.wire.intersect_ray_fronts(self.yrange, P)
            pfall.append(pf[0])
            pball.append(pb[1])

        pf = np.min(pfall)
        pb = np.max(pball)

        # indices of elements in motor position list corresponding to a shadowing of all ROI pixels
        self.pw_idx = np.logical_and(self.pw_all > pf, self.pw_all < pb).nonzero()[0]

        # list of motor position wire corresponding to a shadowing of all ROI pixels 
        self.pw = self.pw_all[self.pw_idx]
        if verbose:
            print('In init_depth() ----------')
            print('self.pw', self.pw)

        # corresponding depth position (considering central pixel)
        self.yw, _ = self.wire.mask_fronts(self.pw, self.Pcam)
        if verbose: print('self.yw', self.yw)
        # crude interpolated middle values of self.yw
        self.ygrid = 0.5 * (self.yw[:-1] + self.yw[1:])
        if verbose:
            print('self.ygrid', self.ygrid)
            print('len(self.ygrid)', len(self.ygrid))
            print('End of init_depth() ----------')


    def init_data(self):

        # extend map for background correction
        if self.grid_rightb:

            pass
            # self.init_data_rightb()

        elif self.grid_leftb:

            pass
            # self.init_data_leftb()

        else:
            self.init_data_general()

    def init_data_general(self):
        # set self.frames_cor  

        """
        set self.frames_cor from self.frames_ini (corrected by monitor value)
        and bkg subtracted with a linear fit in the x direction
        for each frame and each row of pixels
        finaly transpose the array to have the shape:
        (nz, ny, nx)
        """
        xlim, ylim = self.grid_xlim, self.grid_ylim

        xmin, xmax = xlim[0] - 1, xlim[1] + 1

        # read  roi pixel intensity (corrected by monitor value)
        self.frames_ini = np.array(self.scan.get_images_rect_corr([xmin, xmax], ylim,
                                                                  xy=False), dtype=float)[self.pw_idx]

        self.frames_bkg = np.array(self.frames_ini, dtype=float)

        xbkg = [0, 1, self.grid_nx, self.grid_nx + 1]
        xall = np.arange(0, self.grid_nx + 2)

        # Iterate over each image in the initial frames array 
        for i, img in enumerate(self.frames_ini):

            # Iterate over each profile in the image (row of pixels, i.e. along X pixel direction)
            for j, prof in enumerate(img):
                # Perform a linear fit on the background pixels
                p = np.polyfit(xbkg, prof[xbkg], 1)

                # Evaluate the polynomial to get the background profile
                # self.frames_bkg is a 2D array where each row is varying linearly (along X pixel direction)  and ech column is constant (along Y pixel direction)
                self.frames_bkg[i][j] = np.polyval(p, xall)

        self.frames_cor = np.maximum(np.subtract(self.frames_ini, self.frames_bkg * 0.999),
                                     0)
        print('shape self.frames_cor', self.frames_cor.shape)

        self.frames_cor = np.transpose(self.frames_cor[:, :, 1:-1], axes=(2, 1, 0))

        print('after transpose shape self.frames_cor', self.frames_cor.shape)

    def init_data_leftb(self):
        """not used,   background correction from the right border of roi rectangle"""

        xlim, ylim = self.grid_xlim, self.grid_ylim
        xlim[1] = xlim[1] + 4

        self.frames_ini = np.array(self.scan.get_images_rect_corr(xlim, ylim,
                                                                  xy=False), dtype=float)[self.pw_idx]

        self.frames_bkg = np.array(self.frames_ini, dtype=float)

        xbkg = range(self.frames_ini.shape[2] - 5, self.frames_ini.shape[2])
        xall = np.arange(0, self.frames_ini.shape[2])

        for i, img in enumerate(self.frames_ini):

            for j, prof in enumerate(img):
                p = np.polyfit(xbkg, prof[xbkg], 1)

                self.frames_bkg[i][j] = np.polyval(p, xall)

        self.frames_cor = np.maximum(np.subtract(self.frames_ini, self.frames_bkg),
                                     0)

        self.frames_cor = np.transpose(self.frames_cor[:, :, :-4], axes=(1, 2, 0))

    def init_data_rightb(self):
        """not used,   background correction from the left border of roi rectangle"""
        xlim, ylim = self.grid_xlim, self.grid_ylim
        xlim[0] = xlim[0] - 4

        self.frames_ini = np.array(self.scan.get_images_rect_corr(xlim, ylim,
                                                                  xy=False), dtype=float)[self.pw_idx]

        self.frames_bkg = np.array(self.frames_ini, dtype=float)

        xbkg = range(0, 5)
        xall = np.arange(0, self.frames_ini.shape[1])

        for i, img in enumerate(self.frames_ini):

            for j, prof in enumerate(img):
                p = np.polyfit(xbkg, prof[xbkg], 1)

                self.frames_bkg[i][j] = np.polyval(p, xall)

        self.frames_cor = np.maximum(np.subtract(self.frames_ini, self.frames_bkg),
                                     0)

        self.frames_cor = np.transpose(self.frames_cor[:, :, 4:], axes=(1, 2, 0))

    # Reconstruction methods
    def reconstruct(self, regularize=False, reg_alpha=0.5, reg_method='ridge', oversamp=20):

        yi = [(1 - alpha) * self.yw[:-1] + alpha * self.yw[1:] for alpha in np.linspace(0, 1, oversamp)]

        mask = self.scan.img_exist[self.pw_idx]

        # prep result
        self.rec = np.zeros((self.grid_nx, self.grid_ny, len(self.ygrid)), dtype=float)

        if regularize:

            if reg_method == 'lasso':

                def lnorm(x):
                    return np.sum(np.abs(x))
            else:

                def lnorm(x):
                    return np.sum(np.square(x))

            def objfun(x, MatSys, VecSignal):

                # prepare
                Mx_S = np.dot(MatSys, x) - VecSignal

                fval = np.sum(np.square(Mx_S)) + reg_alpha * lnorm(x)

                jac = 2. * (np.dot(MatSys.transpose(), Mx_S) + reg_alpha * x)

                return fval, jac

            for ix in range(self.grid_nx):
                for iy in range(self.grid_ny):

                    S = self.frames_cor[ix][iy]

                    M = [self.wire.calc_transmission(self.pw, y,
                                                     self.grid_Pcam[ix][iy],
                                                     abscoeff=self.abscoeff) for y in yi]

                    M = np.mean(M, axis=0)

                    if reg_method == "lasso":
                        x0 = np.zeros((len(self.ygrid)), dtype=float)
                        ub = None
                    else:
                        x0, _ = spo.nnls(M[mask, :], S[mask])
                        ub = np.sum(x0)

                    res = spo.minimize(objfun, args=(M[mask, :], S[mask]), jac=True,
                                       x0=x0, bounds=[(0, ub)] * len(x0),
                                       options={'eps': 1e-06, 'ftol': 1e-06})

                    self.rec[ix, iy] = res.x

        else:

            for ix in range(self.grid_nx):
                for iy in range(self.grid_ny):
                    S = self.frames_cor[ix][iy]

                    M = [self.wire.calc_transmission(self.pw, y,
                                                     self.grid_Pcam[ix][iy],
                                                     abscoeff=self.abscoeff) for y in yi]

                    M = np.mean(M, axis=0)

                    self.rec[ix, iy], _ = spo.nnls(M[mask, :], S[mask])

        if self.ygrid[0] > self.ygrid[-1]:
            self.ygrid = self.ygrid[::-1]

            self.rec = self.rec[:, :, ::-1]

        self.is_reconstructed = True

    def interpolate(self, yres=None, ystep=0.001):

        if yres is None:

            yres = self.yrange

            yres = np.arange(yres[0], yres[1], ystep)

        elif isinstance(yres, (int, float)):

            yres = [float(yres)]

        else:

            yres = np.arange(yres[0], yres[-1], ystep)

        res = np.zeros((self.grid_nx, self.grid_ny, len(yres)), dtype=float)

        # reconstruct
        for ix in range(self.grid_nx):
            for iy in range(self.grid_ny):
                res[ix, iy] = np.interp(yres,
                                        self.ygrid, self.rec[ix, iy, :],
                                        left=0, right=0)

        if len(yres) == 1:
            res = res[:, :, 0]

        return res, yres

    def to_npy(self, filename, yres=None, ystep=0.001):

        if self.is_reconstructed:
            rec, _ = self.interpolate(yres, ystep)
            np.save(filename, rec)


    def get_rec_size(self):

        return self.grid_nx, self.grid_ny, len(self.ygrid)

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

            msg = "[spotrec] " + pref + msg

            if mode == 'F':
                raise RecError(msg)

            else:
                print(msg)
