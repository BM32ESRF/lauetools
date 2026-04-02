#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

# revised by JS Micha March 2026 namely for python 3.12 and EIGER_4MCdTe detector

import numpy as np
import scipy.signal as sps
import matplotlib.pylab as mplp

from LaueTools.Daxm.classes.calibration.generic import Calib, CalibError

class CalibFluo(Calib):
    """class to handle fluorescence-based calibration of wires trajectory and spacingand orientation"""

    def __init__(self, scan, sample, verbose=True):
        Calib.__init__(self, scan, sample, verbose)

        self.print_msg(" > for fluorescence-based calibration.")

        self.grid_dim = None

        Calib.init(self)

    # Set calibration points from grid
    def set_points_grid(self, dims=None, halfboxsize=None, adjust=True, verbose=False):
        """
        Set calibration points on detector frame located on a grid. One grid is generated for each wire.

        objective is to use self.set_points() with suited XYcam list of points on detector

        Parameters
        ----------
        dims : list or None (optional)
            Number of points along x and y axes. Default is [2, 2].
        halfboxsize : list or None (optional)
            Half box size in pixels. Default is [2, 0].
        adjust : bool (optional)
            If True, adjust locations on the grid to avoid diffraction peaks. Default is True.
        """
        if dims is None:
            dims = [2, 2]

        if halfboxsize is None:
            halfboxsize = [2, 0]

        self.grid_dim = dims

        XYcam = self.gen_wires_points(dims, method="grid", adjust=adjust)

        if verbose:
            print('XYcam results of set_points_grid',XYcam)
            print(halfboxsize)

        #self.data_XYcam = np.array(XYcam)
        # inherited method
        self.set_points(XYcam, halfboxsize)

    # Calibration points
    def gen_wires_points(self, dims=None, method='grid', adjust=True):
        """
        Generate calibration points on a grid for each wire.

        Parameters
        ----------
        dims : list or None (optional)
            Number of points along x and y axes. Default is [2, 2].
        method : str (optional)
            Method to generate points. Default is 'grid'.
        adjust : bool (optional)
            If True, adjust locations on the grid to avoid diffraction peaks. Default is True.

        Returns
        -------
        XYcam : list
            A list of a list of calibration points for each wire.
        """

        if dims is None:
            dims = [2, 2]

        self.print_msg("- Generating calibration points...")

        XYcam = []

        for wid in range(self.scan.wire_qty):
            XYcam.append(self.gen_wire_points(wid, dims, method, adjust))

        return XYcam

    def gen_wire_points(self, wire, dims=None, method='grid', adjust=True):

        if dims is None:
            dims = [2, 2]

        self.print_msg(" > Wire {:d}:", (wire,))

        if method == 'grid':

            XYcam_1wire = self.gen_points_grid(wire, dims, adjust)

        else:

            self.print_msg("Unknown method to generate calibration points: '{}'", (method,), "F")

        # depends method
        return XYcam_1wire

    def gen_points_grid(self, wire, dims, adjust=False):
        """generate a grid of 2D points
        dims = nx, ny
        nx points along x pixel axis, ny points along y pixel axis
        """

        # initial regular grid according to dims
        self.print_msg("   regular {}x{} grid", dims)

        XYcam = gen_grid([0, self.scan.get_img_params(['framedim'])[0]],
                         self.scan.calc_wire_range_scan_int(wire), dims)

        # for EIGER or detector with band gaps, we need to avoid them
        if self.scan.ccd_type == 'EIGER_4MCdTe':
            self.print_msg("Adjusting to avoid EIGER detector bands")
            XYcam = adjust_not_in_bands(XYcam)

        # TODO: optimize calibration grid according to image and delta
        if adjust:
            self.print_msg("   auto-adjusting to avoid peaks.")
            # dX is later used as margin ...
            dX = int(1. / 3 * self.scan.get_img_params(['framedim'])[0] / self.grid_dim[0])

            XYcam = adjust_grid(XYcam, self.scan.get_image(0), self.data_hbs, dX)

        return np.array(XYcam)

    # solver
    def run_init(self, verbose=0):

        self.print_msg("- preparing sample...")
        self.run_init_sample(verbose=verbose-1)

        Calib.run_init(self, verbose=verbose-1)

    def run_init_sample(self,verbose=0):

        # fluo src
        src_I, self.src_y, src_E, _ = self.model_src.get_source_fluo()

        # for consistency, duplicate for each point
        self.src_I, self.src_E = [], []
        for _ in range(self.data_qty):
            self.src_I.append(src_I)
            self.src_E.append(src_E)

        self.src_I = np.array(self.src_I)
        self.src_E = np.array(self.src_E)

    # plots
    def plot_exp_wire(self, wid):

        fig, axarr = mplp.subplots(self.grid_dim[1], self.grid_dim[0], sharex=True, sharey=True, squeeze=False)

        k = 0

        for i in range(self.grid_dim[1]):

            for j in range(self.grid_dim[0]):

                axarr[i, j].plot(self.data_pw[wid][k], self.data_I[wid][k], 'r-o', linewidth=1.5)

                if j == 0:
                    axarr[i, j].set_ylabel('Intensity', fontsize=14)

                if i == self.grid_dim[1] - 1:
                    axarr[i, j].set_xlabel('Wire motor (mm)', fontsize=14)

                axarr[i, j].annotate(self.data_XYcam[wid][k], xy=(1, 0), xycoords='axes fraction',
                                     xytext=(0.975, 0.025), textcoords='axes fraction',
                                     horizontalalignment='right', verticalalignment='bottom')

                axarr[i, j].grid(True)
                k = k + 1

        fig.suptitle('Experimental profiles of wire #{}'.format(wid + 1), fontsize=16)

    def plot_calib_wire(self, wid, sim_I):

        fig, axarr = mplp.subplots(self.grid_dim[1], self.grid_dim[0], sharex=True, sharey=True, squeeze=False)

        k = 0

        for i in range(self.grid_dim[1]):

            for j in range(self.grid_dim[0]):

                axarr[i, j].plot(self.data_pw[wid][k], self.data_I[wid][k], 'r-o', linewidth=1.5)

                axarr[i, j].plot(self.data_pw[wid][k], sim_I[k], 'k', linewidth=1.5)

                if j == 0:
                    axarr[i, j].set_ylabel('Intensity', fontsize=14)

                if i == self.grid_dim[1] - 1:
                    axarr[i, j].set_xlabel('Wire motor (mm)', fontsize=14)

                axarr[i, j].annotate(self.data_XYcam[wid][k], xy=(1, 0), xycoords='axes fraction',
                                     xytext=(0.975, 0.025), textcoords='axes fraction',
                                     horizontalalignment='right', verticalalignment='bottom')
                k = k + 1

        fig.suptitle('Calibration results: experimental vs simulated profiles of wire #{}'.format(wid + 1), fontsize=16)


# Calibration grids
def gen_grid(x_limits, y_limits, dims):
    """generate a flatten list of 2D points
    x_limits: x pixel limits
    y_limits: y pixel limits
    dims = nbpts in x, nbpoints in y

    return a flatten list of 2D points (dims[0] * dims[1])
    """
    x0, xf = [int(x) for x in x_limits]
    y0, yf = [int(y) for y in y_limits]

    # initial regular grid according to dims
    xp = np.linspace(x0, xf, num=dims[0] + 2)
    yp = np.linspace(y0, yf, num=dims[1] + 2)

    xy_grid = [[int(x), int(y)] for y in yp[1:-1] for x in xp[1:-1]]

    return xy_grid


def adjust_grid(xy_grid, img_ref, halfboxsize=None, margin=5):
    """move x coordinates if hot regions
    
    xy_grid :  list (array) of 2D points
    return: list of 2D points
    """
    if halfboxsize is None:
        halfboxsize = [5, 0]

    meanfilter = np.ones([2 * d + 1 for d in halfboxsize])

    I0m = sps.convolve2d(img_ref, meanfilter, mode='same', boundary="symm")

    res = []

    for x, y in xy_grid:
        kx = np.argmin(I0m[(x - margin):(x + margin + 1), y])

        res.append([x - margin + kx, y])

    return res

def adjust_not_in_bands(XYcam, verbose=False):
    """shift if needed pts if falling in band gaps
    XYcam is a list of 2D pts (shape = n1,2)
    return an array of 2D pts of the same shape
    """
    if verbose: print("Initial points:", XYcam)

    XYcam = np.array(XYcam)

    # Forbidden y bands
    y_bands = [(512, 549), (1062, 1099), (1612, 1649)]
    # Forbidden x bands
    x_bands = [(513, 514), (1028, 1039), (1553, 1554)]

    # Create masks for y and x bands
    y_mask = np.zeros(len(XYcam), dtype=bool)
    for band in y_bands:
        y_mask |= (XYcam[:, 1] >= band[0]) & (XYcam[:, 1] <= band[1])

    x_mask = np.zeros(len(XYcam), dtype=bool)
    for band in x_bands:
        x_mask |= (XYcam[:, 0] >= band[0]) & (XYcam[:, 0] <= band[1])

    # Count modifications
    y_modified = np.sum(y_mask)
    x_modified = np.sum(x_mask)

    # Modify y for points in forbidden y bands
    XYcam[y_mask, 1] -= 40
    # Modify x for points in forbidden x bands
    XYcam[x_mask, 0] -= 20
    
    if verbose:
        print('(y_modified, x_modified)')
        print((y_modified, x_modified))

        print("Modified points:", XYcam)

    # Return modified list
    return XYcam

    
    
    
    


