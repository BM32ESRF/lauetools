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

        self.scan = scan

        self.XYcam = XYcam

        self.wire = wire

        self.yrange = yrange

        self.abscoeff = abscoeff

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

        self.pw_all = []
        self.pw_idx = []
        self.pw = []

        self.yw = []
        self.ygrid = []

        self.frames_ini = []
        self.frames_bkg = []
        self.frames_cor = []

        self.rec = []
        self.is_reconstructed = False

        # init
        self.init_hbs()

        self.init_wire()

        self.init_grid()

        self.init_depth()

        self.init_data()

    def init_hbs(self):

        if isinstance(self.hbs, tuple):
            self.hbs = list(self.hbs)

        if not isinstance(self.hbs, (list, tuple)) and isinstance(self.hbs, int):
            self.hbs = [self.hbs] * 2

    def init_wire(self):

        if self.wire is None or is_list_of(self.wire, mywire.CircularWire):
            self.init_wire_assign()

        elif isinstance(self.wire, int):
            self.wire = self.scan.wire[self.wire]

        elif isinstance(self.wire, mywire.CircularWire):
            pass

        else:
            self.print_msg("Invalid wire argument", mode='F')

    def init_wire_assign(self, margin=0):

        if is_list_of(self.wire, mywire.CircularWire):
            wire = self.wire
        else:
            wire = self.scan.wire

        ylim = self.scan.calc_wires_range_scan(wire=wire)

        tmp = []
        for lim in ylim:
            tmp.append((self.XYcam[1] - lim[0]) * (lim[1] - self.XYcam[1]))

        self.wire = wire[np.argmax(tmp)]

    def init_grid(self):

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

    def init_depth(self):

        # wire pos
        self.pw_all = self.scan.wire_position

        pf, pb = self.wire.intersect_ray_fronts(self.yrange, self.Pcam)

        self.pw_idx = np.logical_and(self.pw_all > pf[0], self.pw_all < pb[1]).nonzero()[0]

        self.pw = self.pw_all[self.pw_idx]

        # corresponding depth position (considering central pixel)
        self.yw, _ = self.wire.mask_fronts(self.pw, self.Pcam)
	self.yw = np.arange(self.yw[0]-self.wire.R, self.yw[1]+self.wire.R, 0.001) 

        self.ygrid = 0.5 * (self.yw[:-1] + self.yw[1:])

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

        xlim, ylim = self.grid_xlim, self.grid_ylim

        xmin, xmax = xlim[0] - 1, xlim[1] + 1

        self.frames_ini = np.array(self.scan.get_images_rect_corr([xmin, xmax], ylim,
                                                                  xy=False), dtype=float)[self.pw_idx]

        self.frames_bkg = np.array(self.frames_ini, dtype=float)

        xbkg = [0, 1, self.grid_nx, self.grid_nx + 1]
        xall = np.arange(0, self.grid_nx + 2)

        for i, img in enumerate(self.frames_ini):

            for j, prof in enumerate(img):
                p = np.polyfit(xbkg, prof[xbkg], 1)

                self.frames_bkg[i][j] = np.polyval(p, xall)

        self.frames_cor = np.maximum(np.subtract(self.frames_ini, self.frames_bkg * 0.999),
                                     0)

        self.frames_cor = np.transpose(self.frames_cor[:, :, 1:-1], axes=(2, 1, 0))

    def init_data_leftb(self):

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
