#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import numpy as np

import matplotlib.pylab as mplp

from LaueTools.Daxm.classes.calibration.generic import Calib, CalibError

from LaueTools.Daxm.utils.num import find_closest_product
from LaueTools.Daxm.utils.geom import closest_point


class CalibDiff(Calib):

    def __init__(self, scan, sample, verbose=True):
        Calib.__init__(self, scan,  sample, verbose)

        self.data_XYcam_energy = []

        self.print_msg(" > for diffraction-based calibration.")

        Calib.init(self)

    # Set calibration points from grid
    def set_points(self, XYcam, energy, halfboxsize=[5, 0]):

        Calib.set_points(self, XYcam, halfboxsize)

        self.data_XYcam_energy = energy

    def set_points_fromfile(self, filename, halfboxsize, energy_min = 5., margin=0, blacklist=[]):

        data = np.loadtxt(filename, usecols=(6, 7, 8))

        XYcam_all = np.array(data[:,1:], dtype='int')

        energy_all = data[:,0]

        # resovle blacklist and energy
        is_blacklisted = np.zeros(len(energy_all), dtype='int')

        for i, XY in enumerate(XYcam_all):
            if len(blacklist) and closest_point(XY, blacklist)[0] <= 2.:
                is_blacklisted[i] = 1
            if energy_all[i] < energy_min:
                is_blacklisted[i] = 1

        XYcam_all = XYcam_all[np.logical_not(is_blacklisted)] - 1. # LaueTools convention ?M
        energy_all = energy_all[np.logical_not(is_blacklisted)]

        # assign wires
        XYcam = []
        energy = []

        if len(XYcam_all):

            ylim = self.scan.calc_wires_range_scan(margin=margin, span="inner")

            tmp = np.zeros((len(XYcam_all), len(self.scan.wire) + 1), dtype=int)

            for j, lim in enumerate(ylim):
                tmp[:, j + 1] = (XYcam_all[:, 1] - lim[0]) * (lim[1] - XYcam_all[:, 1])

            wireid = np.argmax(tmp, axis=1) - 1

            for i in range(len(self.scan.wire)):
                subset=np.squeeze(np.nonzero(wireid==i))
                XYcam.append(XYcam_all[subset,:])
                energy.append(energy_all[subset])

        Calib.set_XYcam(self, XYcam=XYcam, energy=energy, halfboxsize=halfboxsize)

    # Solver
    def run_init(self):

        self.print_msg("- preparing sample...")
        self.run_init_sample()

        Calib.run_init(self)

    def run_init_sample(self):

        # "transmitted" src
        src_E = np.concatenate(self.data_XYcam_energy)

        src_I, self.src_y, = self.model_src.get_source_trans(src_E, interpolate=True)

        # put in good shape
        self.src_I, self.src_E = [], []

        for E, I in zip(src_E, src_I):
            self.src_I.append([I])
            self.src_E.append([E])

        self.src_I = np.array(self.src_I)
        self.src_E = np.array(self.src_E)

    # Plots
    def plot_exp_wire(self, wid):

        qty = len(self.data_pw[wid])

        #Nx, Ny = int(np.ceil(np.sqrt(4.*qty/3.))), int(np.ceil(np.sqrt(3.*qty/4.)))
        Nx, Ny = find_closest_product(qty)

        fig, axarr = mplp.subplots(Ny, Nx, sharey=False, sharex=True, squeeze=False)

        k = 0

        for i in range(Ny):

            for j in range(Nx):

                if k < qty:

                    axarr[i, j].plot(self.data_pw[wid][k], self.data_I[wid][k], 'r-o', linewidth=1.5)

                    axarr[i, j].annotate(self.data_XYcam[wid][k], xy=(1, 0), xycoords='axes fraction',
                                         xytext=(0.975, 0.025), textcoords='axes fraction',
                                         horizontalalignment='right', verticalalignment='bottom')

                if j == 0:
                    axarr[i, j].set_ylabel('Intensity', fontsize=14)

                if i == Ny - 1:
                    axarr[i, j].set_xlabel('Wire motor (mm)', fontsize=14)

                k = k + 1

        fig.suptitle('Experimental profiles of wire #{}'.format(wid + 1), fontsize=16)

    def plot_calib_wire(self, wid, sim_I):

        qty = len(self.data_pw[wid])

        #Nx, Ny = int(np.ceil(np.sqrt(4. * qty / 3.))), int(np.ceil(np.sqrt(3. * qty / 4.)))
        Nx, Ny = find_closest_product(qty)

        fig, axarr = mplp.subplots(Ny, Nx, sharey=False, sharex=True, squeeze=False)

        k = 0

        for i in range(Ny):

            for j in range(Nx):

                if k < qty:

                    axarr[i, j].plot(self.data_pw[wid][k], self.data_I[wid][k], 'r-o', linewidth=1.5)

                    axarr[i, j].plot(self.data_pw[wid][k], sim_I[k], 'k', linewidth=1.5)

                    axarr[i, j].annotate(self.data_XYcam[wid][k], xy=(1, 0), xycoords='axes fraction',
                                         xytext=(0.975, 0.025), textcoords='axes fraction',
                                         horizontalalignment='right', verticalalignment='bottom')

                if j == 0:
                    axarr[i, j].set_ylabel('Intensity', fontsize=14)

                if i == Ny - 1:
                    axarr[i, j].set_xlabel('Wire motor (mm)', fontsize=14)

                k = k + 1

        fig.suptitle('Calibration results: experimental vs simulated profiles of wire #{}'.format(wid + 1), fontsize=16)