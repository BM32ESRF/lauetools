#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

# general imports
import os
import json

import math
import numpy as np
import scipy.interpolate as spi

import matplotlib.pylab as mplp

# # LaueTools imports
# from dict_LaueTools import dict_CCD
# import readwriteASCII as rwa
# import readmccd as rmccd

# LaueTools imports
from LaueTools.dict_LaueTools import dict_CCD
import LaueTools.IOLaueTools as rwa
import LaueTools.IOimagefile as rmccd

# daxm imports
import LaueTools.Daxm.utils.read_image as rimg
import LaueTools.Daxm.contrib.spec_reader as rspec

import LaueTools.Daxm.modules.geometry as geom
import LaueTools.Daxm.modules.calibration as calib

import LaueTools.Daxm.classes.wire as mywire


def new_scan_dict(scan_dict=None):
    """Return a default scan as dict."""

    new_dict = {'type': 'point',
                'size':0,
                'skipFrame':0,
                'lineSubFolder':None,
                'specFile': None,
                'scanNumber': 0,
                'scanCmd': [],
                'CCDType': 'sCMOS',
                'detCalib': '',
                'wire': ['W'],
                'wireTrajAngle': 0.,
                'imageFolder': '',
                'imagePrefix': '',
                'imageFirstIndex': 0,
                'imageDigits': 4,
                'imageOffset': 0,
                'monitor': 'spec',
                'monitorROI': (1024, 1024, -1, -1),
                'monitorOffset': 10000}

    if scan_dict is not None:
        new_dict.update(scan_dict)
    return new_dict


def save_scan_dict(scan_dict, filename, directory=""):
    """Write the dict scan to file (json)."""
    filedir = os.path.join(directory, filename)

    with open(filedir, 'w') as pfile:
        pfile.write(json.dumps(scan_dict, indent=1))


def load_scan_dict(filename, directory=""):
    """Load scan from (json) file and return it as dict."""
    filedir = os.path.join(directory, filename)
    import codecs
    print('#######     read .scan file  ############')
    with open(filedir, 'rb') as pfile:
        scan_dict = json.load(pfile)
        # scan_dict = json.load(codecs.open(pfile, 'r', 'utf-8'))

    return new_scan_dict(scan_dict)


# Class definitions
class ScanError(Exception):
    pass


class StaticPointScan(object):
    """General scan class"""
    # Constructors and initialization
    def __init__(self, inp, verbose=True):
        self.verbose = verbose
        self.print_msg("Creating class instance...")

        if isinstance(inp, str):
            self.print_msg(" from file: " + inp)
            inp = load_scan_dict(inp)
        else:
            self.print_msg(" from dict.")
            inp = new_scan_dict(inp)

        self.input = inp

        # attributes related to spec
        self.spec_file = None
        self.spec = None
        self.spec_scan_num = None
        self.spec_scan = None
        self.spec_expo = None
        self.scan_cmd = None
        self.spec_data = None
        self.spec_motor = None
        self.spec_monitor = None
        self.wire_position = None
        self.wire_step = None
        self.number_images = None

        self.init_spec()

        # attributes related to detector and geometry
        self.ccd_type = None
        self.det_calib = None
        self.img_params = None
        self.detector_params = None

        self.init_detector()

        # attributes related the wire(s)
        self.wire_ini = None
        self.wire_traj_angle = None
        self.wire_traj = None
        self.wire_qty = None
        self.wire_params = None
        self.wire = None

        self.init_wire()

        # attributes related to the data
        self.img_folder = None
        self.img_pref = None
        self.img_idx0 = None
        self.img_nbdigits = None
        self.img_offset = None
        self.img_idx = None
        self.img_filenames = None
        self.img_idx_use = None
        self.img_exist = None

        if not hasattr(self, "disable_data") or not getattr(self, "disable_data"):
            self.init_data()

        # attributes relate to the monitor
        self.monitor_ready = False
        self.monitor = None
        self.monitor_roi = None
        self.monitor_offset = None
        self.monitor_val = None
        self.monitor_corrcoeff = None

        if not hasattr(self, "disable_mon") or not getattr(self, "disable_mon"):
            self.init_mon()

        self.print_msg("Ready to work.")
        # End of __init__

    def init_spec(self):

        self.print_msg("- Reading spec...")

        self.spec_file = self.input['specFile']
        self.spec_scan_num = self.input['scanNumber']
        self.scan_cmd = self.input['scanCmd']
        self.spec_data = None
        self.spec_motor = []
        self.spec_monitor = []

        # In case we can use the spec file,
        if self.spec_file is not None:
            self.init_spec_file()
        # Otherwise,
        else:
            self.init_spec_custom()

        self.wire_step = np.diff(self.wire_position).mean()
        self.print_msg("   scanning step is {:.2f} um", fmt=(1000*self.wire_step,))

        self.number_images = self.scan_cmd[2] + 1

    def init_spec_file(self):
        self.spec = rspec.SpecFile(self.spec_file)
        self.spec_data = rspec.Scan(self.spec, self.spec_scan_num)
        self.scan_cmd = self.spec.cmd_list[self.spec_scan_num].split()[3:]
        self.spec_monitor = getattr(self.spec_data, "Monitor")
        self.spec_motor = self.spec.cmd_list[self.spec_scan_num].split()[2]

        for i, dtype in enumerate([float, float, int, float]):
            self.scan_cmd[i] = dtype(self.scan_cmd[i])

        self.spec_expo = self.scan_cmd[3]

        self.wire_position = getattr(self.spec_data, self.spec_motor)

        # Printing stuff
        self.print_msg("   from file: " + self.spec_file)
        self.print_msg('   retrieving scan #{}', fmt=(self.spec_scan_num,))
        self.print_msg("   with command: ascan " + self.spec_motor + " {:f} {:f} {:d} {:f}", fmt=self.scan_cmd)

    def init_spec_custom(self):
        self.spec_motor = "yf"
        self.spec_monitor = np.ones(self.scan_cmd[2] + 1)
        self.spec_expo = self.scan_cmd[3]

        self.wire_position = np.linspace(self.scan_cmd[0], self.scan_cmd[1], self.scan_cmd[2] + 1)

        # Printing stuff
        self.print_msg("   custom command: ascan yf {:f} {:f} {:d} {:f}", fmt=self.scan_cmd)

    def init_detector(self):

        self.ccd_type = self.input['CCDType']
        self.det_calib = self.input['detCalib']

        # CCD from LaueTools dictionary
        self.print_msg("- Retrieving image properties...")

        self.img_params = {'framedim': dict_CCD[self.ccd_type][0][::-1],  # transpose of LaueTools convention
                           'pixelsize': dict_CCD[self.ccd_type][1],
                           'saturation': dict_CCD[self.ccd_type][2],
                           'fliprot': dict_CCD[self.ccd_type][3],
                           'header': dict_CCD[self.ccd_type][4],
                           'formatdata': dict_CCD[self.ccd_type][5],
                           'description': dict_CCD[self.ccd_type][6],
                           'file_extension': dict_CCD[self.ccd_type][7],
                           'offset': self.input['imageOffset']}

        for key in self.img_params:
            self.print_msg("   {:<14s}: {}", fmt=(key, self.img_params[key]))

        # params from detfile
        self.print_msg("- Loading geometry parameters...")

        if isinstance(self.det_calib, str):
            params, _ = rwa.readfile_det(self.det_calib, nbCCDparameters=6, verbose=False)
        else:
            params = self.det_calib
        self.detector_params = {'distance': params[0],
                                'xcen': params[1],
                                'ycen': params[2],
                                'xbet': math.radians(params[3]),
                                'xgam': math.radians(params[4]),
                                'pixelsize': params[5]}

        for key in self.detector_params:
            if key in ('xbet', 'xgam'):
                val = math.degrees(self.detector_params[key])
            else:
                val = self.detector_params[key]
            self.print_msg("   {:<9s}: {}", fmt=(key, val))

    def init_wire(self):

        self.print_msg("- Preparing wires...")

        self.wire_traj_angle = math.radians(self.input['wireTrajAngle'])
        self.wire_ini = self.input['wire']

        self.wire_traj = mywire.new_dict_traj(u2=self.wire_traj_angle)
        self.wire_qty = len(self.wire_ini)
        self.wire_params = []
        self.wire = []

        for ini in self.wire_ini:

            if isinstance(ini, str):
                par = mywire.new_dict(material=ini)
            elif isinstance(ini, dict):
                par = mywire.new_dict(**ini)
            elif hasattr(ini, '__len__') and len(ini)==4:
                par = mywire.new_dict(material=ini[0], R=ini[1], h=ini[2], p0=ini[3])
            else:
                par = []
                self.print_msg("Invalid wire argument! {}", fmt=ini, mode="F")

            self.wire_params.append(par)
            par.update(self.wire_traj)
            self.wire.append(mywire.CircularWire(**par))

        # Printing stuff
        if self.wire_qty == 1:
            w = self.wire[0]
            self.print_msg("   a single wire of {}, diameter {} um", fmt=(w.get_material(), w.get_radius()*2000))
        else:
            for i, w in enumerate(self.wire):
                self.print_msg("   wire {}: {}, diameter {} um", fmt=(i, w.get_material(), w.get_radius()*2000))

        self.print_msg("   moving at {:.2f} deg wrt the beam", fmt=(math.degrees(self.wire_traj_angle),))

    def init_data(self):

        self.print_msg("- Examining dataset...")

        self.img_folder = self.input['imageFolder']
        self.img_pref = self.input['imagePrefix']
        self.img_idx0 = self.input['imageFirstIndex']
        self.img_nbdigits = self.input['imageDigits']
        self.img_offset = float(self.input['imageOffset'])
        self.img_params['offset'] = self.img_offset

        self.img_idx = np.array(range(self.number_images)) + self.img_idx0
        self.img_filenames = [rimg.sprintf_filename(self.img_pref, i, self.img_params['file_extension'],
                                                    self.img_nbdigits) for i in self.img_idx]

        self.check_images_missing(self.verbose)

        self.print_msg("   images in: " + self.img_folder)
        self.print_msg("   named as: " + self.img_pref + "X"*self.img_nbdigits + "."
                       + self.img_params['file_extension'])
        self.print_msg("   found {} images out of {}", fmt=(sum(self.img_exist), self.number_images))

    def init_mon(self):

        self.print_msg("- Loading monitor...")

        self.monitor = self.input['monitor']
        self.monitor_roi = self.input['monitorROI']
        self.monitor_offset = self.input['monitorOffset']
        self.img_offset = float(self.input['imageOffset'])
        self.img_params['offset'] = self.img_offset

        self.monitor_ready = False

        self.set_monitor_none()

        self.set_monitor_offset(self.monitor_offset)

    def set_verbosity(self, verbose=True):
        self.verbose = verbose

    def get_type(self):
        return "point"

    # Methods to get setup parameters
    def get_ccd_dict(self):

        return self.detector_params

    def get_ccd_params(self, keys=None):

        if keys is None:
            keys = ['distance', 'xcen', 'ycen', 'xbet', 'xgam', 'pixelsize']

        res = [self.detector_params[k] for k in keys]

        if len(res) == 1:
            res = res[0]

        return res

    def get_ccd_params_deg(self, keys=None):

        if keys is None:
            keys = ['distance', 'xcen', 'ycen', 'xbet', 'xgam', 'pixelsize']

        res = []

        for k in keys:

            res.append(self.detector_params[k])

            if k in ('xbet', 'xgam'):
                res[-1] = np.degrees(res[-1])

        if len(res) == 1:
            res = res[0]

        return res

    # Methods to get or set the parameters of the wires
    def get_wires_dict(self):

        return [self.get_wire_dict(i) for i in range(self.wire_qty)]

    def get_wire_dict(self, wire):

        dic = self.wire_params[wire]

        dic.update(self.wire_traj)

        return dic

    def get_wires_params(self, keys=None):

        if keys is None:
            keys = ['material', 'R', 'h', 'p0']

        return [self.get_wire_params(i, keys) for i in range(self.wire_qty)]

    def get_wire_params(self, wire=0, keys=None):

        if keys is None:
            keys = ['material', 'R', 'h', 'p0']

        par = self.wire_params[wire]

        par.update(self.wire_traj)

        res = [par[k] for k in keys]

        if len(res) == 1:
            res = res[0]

        return res

    def get_wires_params_deg(self, keys=None):

        if keys is None:
            keys = ['material', 'R', 'h', 'p0']

        return [self.get_wire_params_deg(i, keys) for i in range(self.wire_qty)]

    def get_wire_params_deg(self, wire=0, keys=None):

        if keys is None:
            keys = ['material', 'R', 'h', 'p0']

        par = self.wire_params[wire]

        par.update(self.wire_traj)

        res = []

        for k in keys:

            res.append(par[k])

            if k in ('f1', 'f2', 'u1', 'u2'):
                res[-1] = np.degrees(res[-1])

        if len(res) == 1:
            res = res[0]

        return res

    def get_wire_traj_angle(self):

        return self.wire_traj_angle

    def get_wire_traj_angle_deg(self):

        return np.degrees(self.get_wire_traj_angle())

    def set_wires(self, wires_dict):

        for i, wdict in enumerate(wires_dict):
            self.set_wire(i, wdict)

    def set_wires_deg(self, wires_dict):

        for i in range(len(wires_dict)):
            wires_dict[i] = dict(wires_dict[i])
            wires_dict[i]['f1'] = np.deg2rad(wires_dict[i]['f1'])
            wires_dict[i]['f2'] = np.deg2rad(wires_dict[i]['f2'])

        self.set_wires(wires_dict)

    def set_wires_fromfile(self, wire_files):

        if isinstance(wire_files, str):

            with open(wire_files, 'r') as pfile:
                tmp = json.load(pfile)
            traj_dict, wires_dict = tmp['traj'], tmp['wires']

            self.set_wires_deg(wires_dict)
            self.set_wire_traj_deg(traj_dict)

        elif isinstance(wire_files, list):

            for wid, fn in enumerate(wire_files):
                self.set_wire_fromfile(wid, fn)

        else:
            pass

    def set_wire(self, wid, wire_dict):

        self.wire_params[wid].update(wire_dict)

        new_dict = self.get_wire_dict(wid)

        new_dict.update(wire_dict)

        self.wire[wid].set_par(**new_dict)

        if 'material' in wire_dict:

            self.wire[wid].set_material(wire_dict['material'])

    def set_wire_fromfile(self, wid, wire_file):

        self.set_wire(wid, mywire.load_dict(wire_file))

    def set_wire_traj(self, traj_dict):

        self.wire_traj.update(traj_dict)

        for wire in self.wire:
            wire.set_traj(**traj_dict)

    def set_wire_traj_deg(self, traj_dict):

        traj_dict = dict(traj_dict)
        traj_dict['u1'] = np.deg2rad(traj_dict['u1'])
        traj_dict['u2'] = np.deg2rad(traj_dict['u2'])

        self.set_wire_traj(traj_dict)

    # Methods to get imaging parameters and raw/corrected images
    def get_img_dict(self):

        return self.img_params

    def get_img_params(self, keys=None):

        if keys is None:
            keys = ['framedim', 'pixelsize', 'saturation',
                    'fliprot', 'header', 'formatdata',
                    'description', 'file_extension', 'offset']

        res = [self.img_params[k] for k in keys]

        if len(res) == 1:
            res = res[0]

        return res

    def get_image(self, idx, exist=False):

        if exist:
            frame = self.img_idx_use[idx]
        else:
            frame = idx

        fn = self.get_image_filedir(frame)

        if self.img_exist[frame]:

            img, _, _ = rmccd.readCCDimage(fn, self.ccd_type, verbose=False)

        else:

            img = np.zeros(self.get_img_params(['framedim'])[::-1])

        return img.transpose()

    def get_image_corr(self, idx, exist=False):

        if exist:
            frame = self.img_idx_use[idx]
        else:
            frame = idx

        return self.get_monitor()[frame] * (self.get_image(frame) - self.img_offset) + self.img_offset

    def get_image_index(self, frame):

        return self.img_idx0 + frame

    def get_image_filedir(self, frame):

        return os.path.join(self.img_folder, self.img_filenames[frame])

    def gen_image_filename(self, idx, relative=True, fullpath=True):

        if relative:
            idx = self.img_idx0 + idx

        fn = rimg.sprintf_filename(self.img_pref, idx, self.img_params['file_extension'], self.img_nbdigits)

        if fullpath:
            fn = os.path.join(self.img_folder, fn)

        return fn

    def get_image_rect(self, i, xlim, ylim, xy=True):

        fdir = self.img_folder

        if self.img_exist[i]:

            res = rimg.read_image_rectangle(os.path.join(fdir, self.img_filenames[i]),
                                            xlim, ylim, CCDLabel=self.ccd_type)

        else:

            res = np.ones((ylim[1] - ylim[0] + 1, xlim[1] - xlim[0] + 1)) * self.img_offset

        if xy:
            res = res.transpose()

        return np.array(res, dtype=float)

    def get_images_rect(self, xlim, ylim, xy=True):

        return [self.get_image_rect(i, xlim, ylim, xy) for i in range(self.number_images)]

    def get_images_rect_corr(self, xlim, ylim, xy=True):

        return [corr * (self.get_image_rect(i, xlim, ylim, xy) - self.img_offset)
                + self.img_offset for i, corr in enumerate(self.get_monitor())]

    def get_image_roi(self, i, xcam, ycam, halfboxsize, xy=True):

        fdir = self.img_folder

        if self.img_exist[i]:

            res = rmccd.readrectangle_in_image(os.path.join(fdir, self.img_filenames[i]), xcam, ycam,
                                               halfboxsize[0], halfboxsize[1], CCDLabel=self.ccd_type, verbose=False)

        else:

            res = np.ones((2 * halfboxsize[1] + 1, 2 * halfboxsize[0] + 1)) * self.img_offset

        if xy:
            res = res.transpose()

        return np.array(res, dtype=float)

    def get_images_roi(self, xcam, ycam, halfboxsize=(1, 1), xy=True):

        return [self.get_image_roi(i, xcam, ycam, halfboxsize, xy) for i in range(self.number_images)]

    def get_images_roi_corr(self, xcam, ycam, halfboxsize=(1, 1), xy=True):

        return [corr * (self.get_image_roi(i, xcam, ycam, halfboxsize, xy) - self.img_offset)
                + self.img_offset for i, corr in enumerate(self.get_monitor())]

    def get_images_pixel(self, xcam, ycam, halfboxsize=(1, 1), fun='mean'):

        tmp = self.get_images_roi(xcam, ycam, halfboxsize)

        if fun == "median":
            return np.array(np.median(tmp, axis=(1, 2)), dtype=np.double)

        elif fun == "gmean":
            tmp = np.array(tmp, dtype=np.double)
            return np.exp(np.sum(np.log(tmp + 1), axis=(1, 2)) / np.size(tmp[0])) - 1

        else:  # mean by default in any case
            tmp = np.array(tmp, dtype=np.double)
            return np.mean(tmp, axis=(1, 2))

    def get_images_pixel_corr(self, xcam, ycam, halfboxsize=(1, 1), fun='mean'):

        return self.get_monitor() * (self.get_images_pixel(xcam=xcam,
                                                           ycam=ycam,
                                                           halfboxsize=halfboxsize,
                                                           fun=fun) - self.img_offset) + self.img_offset

    def get_images_tophat(self, step=1):

        img0 = self.get_image_corr(0)

        for i in range(0, self.number_images, step):
            img0 = np.maximum(img0, self.get_image_corr(i))

        return img0

    def check_images_missing(self, verbose=True):

        self.img_exist = []

        for i in range(self.number_images):

            self.img_exist.append(os.path.isfile(self.gen_image_filename(i)))

            if verbose and not self.img_exist[-1]:
                self.print_msg("Frame {}/{} is missing! ({})",
                               fmt=(i + 1, self.number_images, self.gen_image_filename(i)),
                               mode="W")

        self.img_idx_use = np.nonzero(self.img_exist)[0]

        self.img_exist = np.array(self.img_exist, dtype=np.bool_)

    def clip_bbox(self, xy, hbs):

        x, y = int(xy[0]), int(xy[1])
        hx, hy = hbs

        xmin = max(x - hx, 0)
        xmax = min(x + hx + 1, self.get_img_params(['framedim'])[0])
        ymin = max(y - hy, 0)
        ymax = min(y + hy + 1, self.get_img_params(['framedim'])[1])

        return xmin, xmax, ymin, ymax

    # Methods to manipulate the monitor
    def load_monitor(self):

        if self.monitor is None:
            self.print_msg("   no correction.")
            self.set_monitor_none()

        elif self.monitor == "spec":
            self.print_msg("   from spec counter.")
            self.set_monitor_spec()

        elif self.monitor == "detector":
            self.print_msg("   estimated from detector.")
            self.set_monitor_dtt()

        else:
            self.print_msg("Unknown monitor argument... using None instead.", mode="W")

        self.update_img_corrcoeff()

        self.monitor_ready = True

    def get_monitor(self):

        if not self.monitor_ready:
            self.load_monitor()

        return self.monitor_corrcoeff

    def set_monitor_none(self):

        self.monitor_val = np.ones(self.number_images)

    def set_monitor_spec(self):

        expo = self.spec_expo if self.spec_expo > 0 else 1.

        self.monitor_val = np.array(self.spec_monitor, dtype=np.double) / expo - self.monitor_offset

    def set_monitor_dtt(self):

        xcam, ycam, sx, sy = self.monitor_roi

        sx, sy = max(sx, 0), max(sy, 0)

        self.monitor_val = self.get_images_pixel(xcam, ycam, (sx, sy), "mean") - self.img_offset

    def set_monitor_offset(self, value):

        self.monitor_offset = value

        self.update_img_corrcoeff()

    def update_img_corrcoeff(self):

        self.monitor_corrcoeff = np.divide(np.mean(self.monitor_val[self.img_idx_use] + 1E-6),
                                           (self.monitor_val + 1E-6))

    def fit_monitor_offset(self, nb_iter=10, setvalue=False, plot=False):

        if not self.monitor_ready:
            self.load_monitor()

        # monitor profile
        mspec = self.spec_monitor

        # total intensity measured on detector
        xcam, ycam, sx, sy = self.monitor_roi

        if (sx < 0) or (sy < 0):

            img_exp = np.zeros(self.number_images)

            for i in range(self.number_images):
                img = self.get_image(i)

                img_exp[i] = np.mean(img[img > 0].flat)

        else:
            img_exp = self.get_images_pixel(xcam, ycam, (sx, sy), "mean") - self.img_offset

        img_dat = img_exp / img_exp.mean()

        # fit Mspec to Idat with offset as variable
        result, success = calib.fit_monitor_offset(mspec, img_dat, nb_iter)

        if setvalue:
            if success:
                self.set_monitor_offset(result)

        if plot:
            fig = mplp.figure()

            ax1 = fig.add_subplot(111)

            ax1.plot(img_exp, 'b')

            ax1.set_ylabel('Average intensity', color='b', fontsize=14)

            ax1.tick_params('y', colors='b')

            ax2 = ax1.twinx()

            ax2.plot(mspec, 'r')

            coeff = (mspec.mean() - result) / img_exp.mean()

            ax2.set_ylim(np.array(ax1.get_ylim()) * coeff + result)

            ax2.set_ylabel('Monitor', color='r', fontsize=14)

            ax2.tick_params('y', colors='r')

            ax1.set_xlabel('Image index', fontsize=14)

            fig.suptitle('Fit of the monitor offset', fontsize=16)

            mplp.show(True)

        return result, success

    # Methods to get scan profiles
    def get_profile_pixel_full(self, xycam, halfboxsize):

        img = self.get_images_pixel_corr(xycam[0], xycam[1], halfboxsize) - self.img_offset

        return img[self.img_idx_use], self.wire_position[self.img_idx_use]

    def get_profile_pixel_centred(self, wire, xycam, halfboxsize, span=3):

        img, pw = self.get_profile_pixel_full(xycam, halfboxsize)

        p0 = self.calc_wire_intersect_ray(wire, *xycam)

        pinf = np.min(p0) - span * self.get_wire_params(wire, ['R']),
        psup = np.max(p0) + span * self.get_wire_params(wire, ['R'])

        idx = np.logical_and(pw >= pinf, pw <= psup)

        return img[idx], pw[idx]

    def get_profile_manypixels_full(self, xycam, halfboxsize):

        pw = np.array([np.array(self.wire_position) for _ in xycam])

        I = np.zeros((len(xycam), self.number_images, 2*halfboxsize[0]+1, 2*halfboxsize[1]+1))

        # prepare region indices
        xlims, ylims = [], []

        for xy in xycam:

            xmin, xmax, ymin, ymax = self.clip_bbox(xy, halfboxsize)

            xlims.append([xmin, xmax])
            ylims.append([ymin, ymax])

        # extract profiles
        for i in range(self.number_images):

            img = self.get_image_corr(i) #self.get_monitor()[i] *( - self.img_offset) + self.img_offset

            for k in range(len(xycam)):
                I[k, i, :, :] = img[xlims[k][0]:xlims[k][1], ylims[k][0]:ylims[k][1]]

        I = np.mean(I, axis=(2, 3))

        return I[:, np.array(self.img_idx_use)], pw[:, np.array(self.img_idx_use)]

    def get_profile_manypixels_centred(self, wire, xycam, halfboxsize, span=3):

        I_full, pw_full = self.get_profile_manypixels_full(xycam, halfboxsize)

        I, pw = [], []

        for i, wid in enumerate(wire):

            print('wire ... ',wid, type(wid))

            p0 = self.calc_wire_intersect_ray(wid, *xycam[i])

            pinf = np.min(p0) - span * self.get_wire_params(wid, ['R']),
            psup = np.max(p0) + span * self.get_wire_params(wid, ['R'])

            idx = np.logical_and(pw_full[i] >= pinf, pw_full[i] <= psup)

            I.append(I_full[i, idx])
            pw.append(pw_full[i, idx])

        return I, pw

    # Methods to manipulate scan geometry
    def calc_wires_range_scan(self, ysrc=0, wire=None, span="outer"):

        if wire is None:
            wire = self.wire

        return [self.calc_wire_range_scan(w, ysrc, span) for i, w in enumerate(wire)]

    def calc_wire_range_scan(self, wire=0, ysrc=0, span='outer'):

        if isinstance(wire, int):
            wire = self.wire[wire]

        xcam = self.get_img_params(['framedim'])[0] / 2

        ycam = range(0, self.get_img_params(['framedim'])[1], 10)

        pw = [self.calc_wire_intersect_ray(wire, xcam, y, ysrc=ysrc) for y in ycam]

        pf, pb = [t[1] for t in pw], [t[2] for t in pw]

        p0 = self.wire_position[0]
        p1 = self.wire_position[-1]
        p0, p1 = min(p0, p1), max(p0, p1)

        args = {'fill_value': (ycam[0], ycam[-1]),
                'bounds_error': False}
        if span == 'inner':
            y0 = spi.interp1d(pf, ycam, **args)(p0)
            y1 = spi.interp1d(pb, ycam, **args)(p1)
        else:
            y0 = spi.interp1d(pb, ycam, **args)(p0)
            y1 = spi.interp1d(pf, ycam, **args)(p1)

        return y0, y1

    def calc_wire_range_scan_int(self, wire=0, ysrc=0):

        return self.calc_wire_range_scan(wire, ysrc, span='inner')

    def calc_wire_range_scan_ext(self, wire=0, ysrc=0):

        return self.calc_wire_range_scan(wire, ysrc, span='outer')

    def calc_wires_range_shadow(self, frame, ysrc=0, xcam=None):

        return [self.calc_wire_range_shadow(i, frame, ysrc, xcam) for i in range(self.wire_qty)]

    def calc_wire_range_shadow(self, wire, frame, ysrc=0, xcam=None):

        if xcam is None:
            xcam = self.get_img_params(['framedim'])[0] / 2

        ycam = range(0, self.get_img_params(['framedim'])[1], 10)

        pw = [self.calc_wire_intersect_ray(wire, xcam, y, ysrc=ysrc) for y in ycam]

        pa, pf, pb = [t[0] for t in pw], [t[1] for t in pw], [t[2] for t in pw]

        p0 = self.wire_position[frame]

        args = {'fill_value': (ycam[0], ycam[-1]),
                'bounds_error': False}

        ya = spi.interp1d(pa, ycam, **args)(p0)
        yb = spi.interp1d(pb, ycam, **args)(p0)
        yf = spi.interp1d(pf, ycam, **args)(p0)

        return ya, yb, yf

    def calc_wires_position(self, frame, offset=0):

        return [self.calc_wire_position(i, frame, offset) for i in range(self.wire_qty)]

    def calc_wire_position(self, wire, frame, offset=0):

        return self.wire[wire].calc_position(self.wire_position[frame] + offset)

    def calc_wire_intersect_ray(self, wire, xcam, ycam, ysrc=0):

        if isinstance(wire, (int, np.int, np.int64)):
            thewire = self.wire[wire]
        else:
            thewire = wire

        pcam = geom.transf_pix_to_coo(self.get_ccd_params(), xcam, ycam)

        pf, pb = thewire.intersect_ray_fronts(ysrc, pcam)
        pa = thewire.intersect_ray_axis(ysrc, pcam)

        return pa, pf, pb

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

            msg = "[scan] " + pref + msg

            if mode == 'F':
                raise ScanError(msg)

            else:
                print(msg)

    def plot_image(self, idx, img=None):
        print('in plot_image in point.py')
        if img is None:
            img = self.get_image(idx)

        img0 = self.get_image(0)

        c1 = int(np.mean(img0, axis=(0, 1))) + 4 * int(np.std(img0, axis=(0, 1)))
        c0, c1 = self.img_offset, int(c1)

        fig = mplp.figure()

        mplp.imshow(img.transpose(), origin='lower')

        mplp.xlim([0, self.get_img_params(['framedim'])[0]])
        mplp.ylim([0, self.get_img_params(['framedim'])[1]])

        mplp.xlabel("Xcam")
        mplp.ylabel("Ycam")

        mplp.clim(c0, c1)

        return fig

    def plot_monitor(self, fig=None, fontsize=14):

        if not self.monitor_ready:
            self.load_monitor()

        if fig is None:
            fig = mplp.figure()

        ax1 = fig.add_subplot(121)

        ax1.plot(self.img_idx_use, self.monitor_val[self.img_idx_use])

        ax1.set_xlabel("Image index", fontsize=fontsize)

        ax1.set_ylabel("Monitor (from %s)" % self.monitor, fontsize=fontsize)

        ax2 = fig.add_subplot(122)

        ax2.plot(self.img_idx_use, self.monitor_corrcoeff[self.img_idx_use] - 1.)

        ax2.plot(ax2.get_xlim(), [0, 0], 'k-.')

        ax2.set_xlabel("Image index", fontsize=fontsize)

        ax2.set_ylabel("Intensity correction factor - 1", fontsize=fontsize)

        fig.suptitle('Selected monitor and resulting correction', fontsize=fontsize + 2)

        fig.tight_layout()

        fig.show(True)

    def plot_monitor_corrected(self, fig=None, fontsize=14):

        if not self.monitor_ready:
            self.load_monitor()

        if fig is None:
            fig = mplp.figure()

        xs = [500, 1000, 1500]
        ys = [500, 1000, 1500]

        ylim = 1
        ax = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                k = 1 + i + len(xs) * j
                ax.append(fig.add_subplot(len(ys), len(xs), k))
                prof_raw = self.get_images_pixel(x, y, (5, 0))
                prof_cor = self.get_images_pixel_corr(x, y, (5, 0))
                ax[-1].plot(self.img_idx_use, prof_raw[self.img_idx_use], 'kx')
                ax[-1].plot(self.img_idx_use, prof_cor[self.img_idx_use], 'r')
                ax[-1].set_xlabel("Image index", fontsize=fontsize)
                ax[-1].set_ylabel("Intensity", fontsize=fontsize)
                ax[-1].set_title("X={}, Y={}".format(x, y), fontsize=fontsize)
                ylim = max(ylim, ax[-1].get_ylim()[1])

        print('ax : ', ax)
        for a in ax:
            a.set_ylim(self.img_offset, ylim)

        ax[len(ax) / 2].legend(["raw profile", "corrected profile"], loc=0)

        fig.suptitle('Raw and corrected intensity profiles', fontsize=fontsize + 2)

        fig.tight_layout()

        fig.show(True)


class PointScan(StaticPointScan):
    """Editable scan class"""
    def __init__(self, inp, verbose=True):

        StaticPointScan.__init__(self, inp, verbose)

    # Methods to safely and cleanly modify the scan
    def update(self, scan_dict, part):

        if part in ("setup", "all"):
            self.update_setup(scan_dict)

        if part in ("wire", "all"):
            self.update_wire(scan_dict)

        if part in ("spec", "all"):
            self.update_spec(scan_dict)

        if part in ("data", "all"):
            self.update_data(scan_dict)

        if part in ("mon", "all"):
            self.update_mon(scan_dict)

    def update_setup(self, scan_dict):

        self.update_input(scan_dict, ['CCDType', 'detCalib'])

        self.init_detector()
        self.init_data()

    def update_wire(self, scan_dict):

        self.update_input(scan_dict, ['wire', 'wireTrajAngle'])

        self.init_wire()

    def update_spec(self, scan_dict):

        self.update_input(scan_dict, ['specFile', 'scanNumber', 'scanCmd'])

        self.init_spec()
        self.init_data()
        self.init_mon()

    def update_data(self, scan_dict):

        self.update_input(scan_dict, ['imageFolder', 'imagePrefix', 'imageFirstIndex', 'imageDigits'])

        self.init_data()
        self.init_mon()

    def update_mon(self, scan_dict):

        self.update_input(scan_dict, ['monitor', 'monitorROI', 'monitorOffset', 'imageOffset'])

        self.init_mon()

    def update_input(self, scan_dict, keys=None):

        if keys is None:

            self.input.update(scan_dict)

        else:
            for key in keys:

                self.input[key] = scan_dict[key]

    # overloaded functions
    def goto_centre(self):
        pass

    # Methods to load from file and save to file
    def load(self, fname, directory=""):

        self.print_msg("Loading scan from: {}", fmt=fname)

        dat = load_scan_dict(fname, directory)

        self.update(dat, "all")

        return dat

    def save(self, fname, directory=""):

        self.print_msg("Saving scan to: {}", fmt=fname)

        dat = self.to_dict()

        save_scan_dict(dat, fname, directory)

    def to_dict(self, part="all"):

        dict_res = {'type':"point"}

        if part in ("setup", "all"):
            dict_setup = {'CCDType': self.ccd_type,
                          'detCalib': self.det_calib}
            dict_res.update(dict_setup)

        if part in ("wire", "all"):
            dict_wire = {'wire': self.wire_ini,
                         'wireTrajAngle': self.get_wire_traj_angle_deg()}
            dict_res.update(dict_wire)

        if part in ("spec", "all"):
            dict_spec = {'specFile': self.spec_file,
                         'scanNumber': self.spec_scan_num,
                         'scanCmd': self.scan_cmd}
            dict_res.update(dict_spec)

        if part in ("data", "all"):
            dict_dat = {'imageFolder': self.img_folder,
                        'imagePrefix': self.img_pref,
                        'imageFirstIndex': self.img_idx0,
                        'imageDigits': self.img_nbdigits,
                        'imageOffset': self.img_offset,
                        'size': 0,
                        'skipFrame': 0,
                        'lineSubFolder': False}
            dict_res.update(dict_dat)

        if part in ("mon", "all"):
            dict_mon = {'monitor': self.monitor,
                        'monitorROI': self.monitor_roi,
                        'monitorOffset': self.monitor_offset,
                        'imageOffset': self.img_offset}
            dict_res.update(dict_mon)

        return dict_res


# End of scan module.
if __name__ == "__main__":

    # save_scan_dict(new_scan_dict(), "example.wscan")

    #s = SimScan("example.wscan")

    #s.get_image(0)

    pass
