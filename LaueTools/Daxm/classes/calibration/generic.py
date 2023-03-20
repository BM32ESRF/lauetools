#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os
import json

try:
    from typing import Union
except ImportError:
    pass

import numpy as np
import scipy.optimize as spo

import LaueTools.Daxm.classes.scan as myscan
import LaueTools.Daxm.classes.source as mysrc
import LaueTools.Daxm.classes.wire as mywire

import LaueTools.Daxm.modules.geometry as geom
import LaueTools.Daxm.material.absorption as abso
import LaueTools.Daxm.modules.calibration as calib

from LaueTools.Daxm.utils.path import add_extension
from LaueTools.Daxm.utils.colors import gen_colormap


# Class definitions
class CalibError(Exception):
    pass


class Calib:
    """General calibration class"""

    # Constructors and initialization
    def __init__(self, scan, sample, verbose=True):

        self.verbose = verbose

        self.print_msg("Creating class instance...")

        # input
        self.arg_scan = scan
        self.arg_src = sample

        # calibration model
        self.scan = None  # type: Union[myscan.PointScan, myscan.LineScan, myscan.MeshScan]
        self.model_src = None  # type: mysrc.SecondarySource
        self.model_geo = []
        self.model_wire = []

        # experimental data
        self.data_XYcam = []
        self.data_qty = 0
        self.data_pw = []
        self.data_I = []
        self.data_span = 5
        self.data_hbs = [2, 0]

        # pre-compute
        self.exp_Pcam = []
        self.exp_wire = []
        self.exp_pw = []
        self.exp_I = []
        self.exp_size = 0
        self.exp_qty = 0
        self.sim_I0 = []
        self.sim_I = []
        self.src_y = []
        self.src_I = []
        self.src_E = []
        self.src_mu = []
        self.wire_par = []
        self.wire_mu = []

        # optimization stuff
        self.opt_wire = []
        self.opt_var = []
        self.opt_bnd = []
        self.opt_fun = []
        self.opt_algo = []
        self.opt_residuals = []
        self.opt_objfun = None

        self.var_ini = []
        self.var_cur = []
        self.var_lut = []
        self.var_x0 = []
        self.var_lb = []
        self.var_ub = []
        self.var_key = []
        self.var_kM = []
        self.var_km = []
        self.var_dm = 0.
        self.var_bnd = {'Re': [-0.010, 0.010],
                        'R': [-0.010, 0.010],
                        'p0': [-0.5, 0.5],
                        'h': [-0.5, 0.5],
                        'f1': [-5., 5.],
                        'f2': [-5., 5.],
                        'u1': [-5., 5.],
                        'u2': [-5., 5.], }

        self.inp_wire = ''
        self.inp_sam = ''
        self.inp_data = ''

        # history and results

        self.log_qty = 0
        self.log_select = -1

        self.log_opt = []
        self.log_var = []
        self.log_wdict = []
        self.log_wlist = []
        self.log_err = []
        self.log_dm = []
        self.log_err_dm = []
        self.log_sim_I = []
        self.log_sim_kM = []
        self.log_sim_km = []
        self.log_fval = []
        self.log_gof = []

    # --- ~~~~~~~ initializers ~~~~~~~
    def init(self):

        # init from args
        self.treat_args()

        self.init_model()

        self.init_log()

        self.print_msg("Ready to work.")
        # End of __init__

    def treat_args(self):

        self.print_msg("- Reading arguments...")
        # scan
        self.print_msg("  loading and centering scan...", mode='I')

        self.scan = myscan.new_scan(self.arg_scan, verbose=False)

        self.scan.goto_centre()

        if self.scan is None:

            self.print_msg("   failed to load scan!", mode='F')

        # source/sample
        self.print_msg("  loading sample...", mode='I')

        self.set_sample(self.arg_src)

    def init_model(self):

        self.print_msg("- Loading geometrical model...")

        self.model_geo = self.scan.get_ccd_params()

        self.model_wire = [mywire.CircularWire(**par) for par in self.scan.get_wires_dict()]

    def init_data(self):

        self.data_pw = [[] for _ in self.data_XYcam]
        self.data_I = [[] for _ in self.data_XYcam]

        if len(self.data_XYcam):

            self.print_msg("- Loading experimental profiles...")

            XYcam = np.concatenate(self.data_XYcam)

            wire = np.concatenate([[i] * len(xy) for i, xy in enumerate(self.data_XYcam)])

            data_I, data_pw = self.scan.get_profile_manypixels_centred(wire, XYcam, self.data_hbs, self.data_span)

            for i, wid in enumerate(wire):
                self.data_pw[wid].append(data_pw[i])

                self.data_I[wid].append(data_I[i])

            self.print_msg("   loaded {} profiles / {} data points.", (len(data_I),
                                                                       sum(len(I) for I in data_I)))

    def init_log(self):

        self.print_msg("- Initialize log...")

        self.log_append(self.scan.get_wires_dict())

        self.log_report()

    # Setters (angles in radians)
    def set_points(self, XYcam, halfboxsize=None):

        if halfboxsize is None:
            halfboxsize = [2, 0]

        self.data_XYcam = XYcam

        self.data_hbs = halfboxsize

        self.data_qty = np.sum([1 for XY in self.data_XYcam for xy in XY])

        self.init_data()

    def set_wires(self, wires_dict, traj_dict=None):

        self.scan.set_wires(wires_dict)

        if traj_dict is not None:
            self.scan.set_wire_traj(traj_dict)

    def set_wire(self, wid, wire_dict):

        self.scan.set_wire(wid, wire_dict)

    def set_wire_traj(self, traj_dict):

        self.scan.set_wire_traj({k: traj_dict[k] for k in ('u1', 'u2')})

    def set_span(self, span=5):

        self.data_span = span

    def set_sample(self, sample):
        print("self.arg_src as input of SecondarySource", self.arg_src)
        print("type of self.arg_src as input of SecondarySource", type(self.arg_src))
        if isinstance(self.arg_src, mysrc.SecondarySource):
            print('I go there  !   works  !!!')
            self.model_src = self.arg_src
        else:
            print('I go HERRRRRRE  !   Bad !')
            self.model_src = mysrc.SecondarySource(self.arg_src)

    # getters (angles in radians)
    def get_wires(self, keys=None):

        if keys is None:
            keys = ['R', 'p0', 'h', 'f1', 'f2', 'u1', 'u2']

        wires_par = self.scan.get_wires_params(keys)

        return [{key: val for key, val in zip(keys, wire)} for wire in wires_par]

    def get_wires_traj(self):

        keys = ['u1', 'u2']

        wire0_par = self.scan.get_wire_params(0, keys)

        return {key: val for key, val in zip(keys, wire0_par)}

    def get_wire(self, wid, keys=None):

        if keys is None:
            keys = ['R', 'p0', 'h', 'f1', 'f2', 'u1', 'u2']

        wire_par = self.scan.get_wire_params(wid, keys)

        return {key: val for key, val in zip(keys, wire_par)}

    # Saving/Loading wire parameters
    def save_wires(self, filename, directory=""):
        """ save a json file of wire(s) parameters. Output file has .calib extension"""
        if directory == "" or os.path.isdir(directory):

            fn = os.path.join(directory, add_extension(filename, "calib"))

            traj_dict = self.get_wires_traj()

            wires_dict = self.get_wires(keys=['material', 'R', 'p0', 'h', 'f1', 'f2'])

            traj_dict['u1'] = np.rad2deg(traj_dict['u1'])
            traj_dict['u2'] = np.rad2deg(traj_dict['u2'])

            for i in range(len(wires_dict)):
                wires_dict[i]['f1'] = np.rad2deg(wires_dict[i]['f1'])
                wires_dict[i]['f2'] = np.rad2deg(wires_dict[i]['f2'])

            self.print_msg("Saving wire parameters to: {}", fmt=(fn,))

            with open(fn, 'w') as pfile:
                pfile.write(json.dumps({'traj': traj_dict,
                                        'wires': wires_dict}, indent=1))

        else:
            self.print_msg("   directory is not valid or does not exist! ({})", (directory,), "E")

    def load_wires(self, filename, directory=""):

        fn = os.path.join(directory, filename)

        if os.path.isfile(fn):

            self.print_msg("Loading wire parameters from: {}", fmt=(fn,))

            with open(fn, 'r') as pfile:
                tmp = json.load(pfile)

            traj_dict, wires_dict = tmp['traj'], tmp['wires']

            traj_dict['u1'] = np.deg2rad(traj_dict['u1'])
            traj_dict['u2'] = np.deg2rad(traj_dict['u2'])

            for i in range(len(wires_dict)):
                wires_dict[i]['f1'] = np.deg2rad(wires_dict[i]['f1'])
                wires_dict[i]['f2'] = np.deg2rad(wires_dict[i]['f2'])

            self.set_wires(wires_dict, traj_dict)

            self.log_append(self.get_wires())

            self.log_report()

        else:
            self.print_msg(" file does not exit! ({})", (fn,), "E")

    # Calibration solver
    def run(self, wire=None, var=None, bounds=None):

        self.run_arg(wire, var, bounds)

        self.print_msg("Calibration of {} on wires {}...", (var, wire))

        # prepare calibration
        self.print_msg("- Initializing:")
        self.run_init()

        # run calibration
        self.print_msg("- Running...")
        res = self.run_optim()

        # get results
        self.print_msg("- Getting results...")
        self.run_result(res)

        # print report
        self.log_report()

    def run_arg(self, wire, var, bounds):

        # wires to calibrate
        if wire is None:
            wire = range(self.scan.wire_qty)

        self.opt_wire = wire

        # variables and boundaries
        if var is None:
            var = ['Re', 'h', 'p0', 'f1', 'f2', 'u2']
        elif var == 'all' or 'all' in var:
            var = ['Re', 'h', 'p0', 'f1', 'f2', 'u1', 'u2']

        if bounds is None:
            bounds = [None] * len(var)

        opt_var = []
        opt_bnd = []
        for key, bound in zip(var, bounds):
            if key in ['Re', 'h', 'p0', 'f1', 'f2', 'u1', 'u2']:
                opt_var.append(key)
                opt_bnd.append(bound)
            elif key == 'R' and 'Re' not in var:
                opt_var.append(key)
                opt_bnd.append(bound)
            elif key in ['axis', 'f']:
                opt_var.extend(['f1', 'f2'])
                opt_bnd.append(bound)
                opt_bnd.append(bound)
            elif key in ['traj', 'u']:
                opt_var.extend(['u1', 'u2'])
                opt_bnd.append(bound)
                opt_bnd.append(bound)
            elif key == 'dm':
                opt_var.append('dm')
                opt_bnd.append(None)
            else:
                pass

        self.opt_var = []
        self.opt_bnd = []
        for var, bnd in zip(opt_var, opt_bnd):
            if var not in self.opt_var:
                self.opt_var.append(var)
                self.opt_bnd.append(bnd)

    def run_init(self):

        self.print_msg("- preparing data...")
        self.run_init_data()

        self.print_msg("- guessing parameters...")
        self.run_init_var(self.opt_var)

        self.print_msg("- setting bounds...")
        self.run_init_bounds(self.opt_bnd)

        self.print_msg("- objective function...")
        self.run_init_objfun(self.opt_var)

    def run_init_data(self):

        # Pcam
        self.exp_Pcam = []
        self.exp_wire = []
        self.exp_pw = []
        self.exp_I = []
        detpar = self.scan.get_ccd_params()
        for w, XYcam_wire in enumerate(self.data_XYcam):
            if w in self.opt_wire:
                for k, XYcam in enumerate(XYcam_wire):
                    self.exp_wire.append(w)
                    self.exp_Pcam.append(geom.transf_pix_to_coo(detpar, *XYcam))
                    self.exp_pw.append(self.data_pw[w][k])
                    self.exp_I.append(self.data_I[w][k])

        self.exp_Pcam = np.array(self.exp_Pcam)
        self.exp_wire = np.array(self.exp_wire, dtype=np.int16)
        self.exp_qty = len(self.exp_Pcam)
        self.exp_size = np.sum([len(I) for I in self.exp_I])

        # wires
        self.wire_par = [self.scan.get_wire_params(i, ['R', 'p0', 'h', 'f1', 'f2', 'u1', 'u2']) for i in self.opt_wire]

        # absorption
        coeff, energy, _ = abso.calc_absorption(self.scan.get_wire_params(0, ['material']), absolute=True)
        self.wire_mu = np.interp(self.src_E, energy, coeff)

        abscoeff, energy, _ = self.model_src.get_absorption()
        self.src_mu = np.interp(self.src_E, energy, abscoeff[0])

        # simu
        self.sim_I0 = []
        for i, Pcam in enumerate(self.exp_Pcam):
            d1 = geom.calc_sample_abslength(self.model_src.angle, self.src_y[np.newaxis, :], Pcam, relative=False)
            I0 = [np.array([Isrc]) * np.exp(-d1 * self.src_mu[i][k]) for k, Isrc in enumerate(self.src_I[i])]
            I0_integ = np.sum(np.trapz(I0, self.src_y, axis=2), axis=(0, 1))
            self.sim_I0.append(I0 / I0_integ)

        self.sim_I = [np.zeros(len(Iexp)) for Iexp in self.exp_I]

        # reshape
        self.src_y = np.array(self.src_y)[np.newaxis, :]
        self.exp_pw = [np.array(arr)[:, np.newaxis] for arr in self.exp_pw]

    def run_init_sample(self):

        pass

    def run_init_var(self, var):

        self.var_ini = [{'R': par[0],
                         'p0': par[1],
                         'h': par[2],
                         'f1': par[3],
                         'f2': par[4],
                         'u1': par[5],
                         'u2': par[6]} for par in self.wire_par]

        self.var_lut = [{'R': None,
                         'p0': None,
                         'h': None,
                         'f1': None,
                         'f2': None,
                         'u1': None,
                         'u2': None} for _ in self.wire_par]

        self.var_cur = [dict(ini) for ini in self.var_ini]

        x0 = []
        keys = []
        dof = 0

        for i, par0 in enumerate(self.var_ini):
            for k in ['R', 'p0', 'h', 'f1', 'f2']:
                if k in var:
                    keys.append(k)
                    x0.append(par0[k])
                    self.var_lut[i][k] = dof
                    dof = dof + 1

        if 'Re' in var:
            keys.append('Re')
            x0.append(self.var_ini[0]['R'])
            for i in range(len(self.var_ini)):
                self.var_lut[i]['R'] = dof
            dof = dof + 1

        if 'u1' in var:
            keys.append('u1')
            x0.append(self.var_ini[0]['u1'])
            for i in range(len(self.var_ini)):
                self.var_lut[i]['u1'] = dof
            dof = dof + 1

        if 'u2' in var:
            keys.append('u2')
            x0.append(self.var_ini[0]['u2'])
            for i in range(len(self.var_ini)):
                self.var_lut[i]['u2'] = dof
            dof = dof + 1

        if 'dm' in var:
            keys.append('dm')
            if self.log_select:
                dm0 = self.log_dm[self.log_select]
            else:
                dm0 = np.max(self.model_src.source_ysrc)*0.5
            x0.append(dm0)

        for Iexp in self.exp_I:
            keys.extend(['kM', 'km'])
            x0.append(np.max(Iexp) - np.min(Iexp))
            x0.append(np.min(Iexp))

        self.var_x0 = x0
        self.var_key = keys

    def run_init_bounds(self, bounds):

        # build LUT
        for key, bnd in zip(self.opt_var, self.opt_bnd):
            if bnd is not None:
                self.var_bnd[key] = np.array(bnd)

        for key in ['f1', 'f2', 'u1', 'u2']:
            self.var_bnd[key] = np.deg2rad(self.var_bnd[key])

        # init boundaries
        self.var_lb = []
        self.var_ub = []

        for i, key in enumerate(self.var_key):

            if key in ['R', 'Re', 'h']:

                lb = np.maximum(self.var_x0[i] + self.var_bnd[key][0], 0.001)
                ub = self.var_x0[i] + self.var_bnd[key][1]

            elif key in ['p0', 'u1', 'u2', 'f1', 'f2']:

                lb = self.var_x0[i] + self.var_bnd[key][0]
                ub = self.var_x0[i] + self.var_bnd[key][1]

            elif key == 'kM':

                lb = 0.1 * self.var_x0[i]
                ub = 3. * self.var_x0[i]

            elif key == 'km':

                lb = 0.
                ub = self.var_x0[i] + self.var_x0[i - 1]

            elif key == 'dm':

                lb = 0.002
                ub = np.max(self.model_src.source_ysrc)-0.001

            else:
                # should not happen
                continue

            self.var_lb.append(lb)
            self.var_ub.append(ub)

    def run_init_objfun(self, opt_var):

        if 'dm' in self.opt_var:

            self.opt_objfun = calib.objfun_depth_residuals

        else:

            self.opt_objfun = calib.objfun_residuals

    def run_optim(self):

        self.opt_residuals = np.zeros(self.exp_size)

        res = spo.least_squares(self.run_optim_fun,
                                self.var_x0, bounds=(self.var_lb, self.var_ub),
                                method='trf', ftol=1e-08, xtol=1e-08, verbose=2)

        return res

    def run_optim_fun(self, x):

        self.run_optim_unpack(x)

        R = np.array([var['R'] for var in self.var_cur])
        h = np.array([var['h'] for var in self.var_cur])
        p0 = np.array([var['p0'] for var in self.var_cur])
        axis = np.array([geom.calc_axis(var['f1'], var['f2']) for var in self.var_cur])
        traj = np.array([geom.calc_traj(var['u1'], var['u2']) for var in self.var_cur])

        # self.exp_pw, self.sim_I0, self.sim_I, self.exp_I

        #print('self.exp_pw', self.exp_pw)
        #print('traj', traj)
        # self.opt_objfun(R, h, p0, axis, traj, self.var_dm, self.var_kM, self.var_km,
        #                         self.exp_Pcam, self.exp_pw, self.exp_wire, self.src_y, self.wire_mu,
        #                         self.sim_I0, self.sim_I, self.exp_I, self.opt_residuals)

        self.opt_objfun(R, h, p0, axis, traj, self.var_dm, self.var_kM, self.var_km,
                                self.exp_Pcam, self.exp_pw, self.exp_wire, self.src_y, self.wire_mu,
                                self.sim_I0, self.sim_I, self.exp_I, self.opt_residuals)

        return np.array(self.opt_residuals)

    def run_optim_sim(self, x):

        self.run_optim_unpack(x)

        axis = [geom.calc_axis(var['f1'], var['f2']) for var in self.var_cur]
        traj = [geom.calc_traj(var['u1'], var['u2']) for var in self.var_cur]

        Isim = []

        if 'dm' in self.opt_var:
            dmask = calib.objfun_depth_mask(self.src_y[0], self.var_dm)
        else:
            dmask = np.ones(len(self.src_y[0]))

        for i, wire in enumerate(self.exp_wire):
            dist = geom.calc_wire_abslength(self.var_cur[wire]['R'], self.var_cur[wire]['h'], self.var_cur[wire]['p0'],
                                            axis[wire], traj[wire],
                                            self.exp_pw[i], self.src_y, self.exp_Pcam[i])

            Iabs = [np.exp(-dist * mu) for mu in self.wire_mu[i]]

            Isim.append(
                self.var_kM[i] * np.sum(np.trapz(self.sim_I0[i] * Iabs * dmask, self.src_y[0], axis=2), axis=0)
                + self.var_km[i])

        return Isim

    def run_optim_unpack(self, x):

        for i, lut in enumerate(self.var_lut):
            for key in lut.keys():
                if lut[key] is not None:
                    self.var_cur[i][key] = x[lut[key]]

        if 'dm' in self.opt_var:
            self.var_dm = x[-2 * self.exp_qty - 1]

        kMkm = x[-2 * self.exp_qty:]

        self.var_kM = np.array(kMkm[::2])
        self.var_km = np.array(kMkm[1::2])

    def run_result(self, res):

        params, kM, km, dm = self.run_result_params(res.x)

        errors, error_dm, fval, gof = self.run_result_errors(res.fun, res.cost, res.jac)

        sim_I = self.run_result_simu(res.x)

        self.log_append(params, self.opt_wire, self.opt_var, errors,  dm, error_dm, kM, km, sim_I, fval, gof)

    def run_result_params(self, x):

        self.run_optim_unpack(x)

        for i, wid in enumerate(self.opt_wire):

            self.set_wire(wid, self.var_cur[i])

            self.set_wire_traj(self.var_cur[i])

        return [var.copy() for var in self.var_cur], np.array(self.var_kM), np.array(self.var_km), self.var_dm

    def run_result_errors(self, residuals, chisq, jac):

        nvar, ndat = len(self.var_x0), self.exp_size

        nu = ndat - nvar

        # min fval
        fval = chisq / nu

        # goodness of fit
        boxsize = np.prod(np.array(self.data_hbs) * 2 + 1)

        gof = np.divide(np.square(residuals),
                        np.clip(np.concatenate(self.exp_I), 1., None))

        gof = 1. / nu * np.sum(gof) * boxsize

        # std errors
        hessian = np.matmul(np.transpose(jac), jac)

        covar = np.linalg.inv(hessian)

        err = np.sqrt(fval * np.diag(covar))

        self.run_optim_unpack(err)

        errors = [var.copy() for var in self.var_cur]

        error_dm = self.var_dm

        return errors, error_dm, fval, gof

    def run_result_simu(self, x):

        result = [[] for _ in self.opt_wire]

        Isim = self.run_optim_sim(x)

        for i, wire in enumerate(self.exp_wire):
            result[wire].append(Isim[i])

        return result

    # Calibration log
    def log_append(self, wdict, wlist=None, var=None, errors=None, dm=None, error_dm=None, kM=None, km=None, sim_I=None, fval=None, gof=None):

        self.log_qty = self.log_qty + 1
        self.log_select = self.log_qty - 1

        self.log_wdict.append([dict(w) for w in wdict])
        self.log_wlist.append(wlist)
        self.log_var.append(var)
        self.log_err.append(errors)
        self.log_dm.append(dm)
        self.log_err_dm.append(error_dm)
        self.log_sim_I.append(sim_I)
        self.log_sim_kM.append(kM)
        self.log_sim_km.append(km)
        self.log_fval.append(fval)
        self.log_gof.append(gof)

    def log_restart(self):

        self.print_msg("- Clearing calibration log...", mode="W")

        self.log_qty = 0
        self.log_select = -1

        self.log_var = []
        self.log_wdict = []
        self.log_wlist = []
        self.log_err = []
        self.log_dm = []
        self.log_err_dm = []
        self.log_sim_I = []
        self.log_sim_kM = []
        self.log_sim_km = []
        self.log_fval = []
        self.log_gof = []

        self.log_append(self.scan.get_wires_dict())

        self.log_report()

    def log_report(self, i=None):

        if i is None:
            i = self.log_select

        if i == 0:  # initial values

            self.log_report_ini()

        else:
            if self.log_wlist[i] is not None:  # result of a fit

                self.log_report_calib(i)

            else:  # applied from file, etc.

                self.log_report_current()

    def log_report_ini(self):

        var = ['R', 'h', 'p0', 'f1', 'f2', 'u1', 'u2']
        header = [key for key in var]

        self.print_msg("Initial parameters:")

        self.print_msg("wire\\" + "{:^10}" * 7, header)

        for wid in range(self.scan.wire_qty):
            par = self.log_wdict[0][wid]
            par = [np.rad2deg(par[key]) if key in ('u1', 'u2', 'f1', 'f2') else par[key] for key in var]

            self.print_msg(" #{} |".format(wid + 1) + " {:>8f} " * 7, par)

    def log_report_current(self):

        var = ['R', 'h', 'p0', 'f1', 'f2', 'u1', 'u2']
        header = [key for key in var]

        self.print_msg("Current parameters:")

        self.print_msg("wire\\" + "{:^10}" * 7, header)

        for wid in range(self.scan.wire_qty):
            par = self.log_wdict[0][wid]
            par = [np.rad2deg(par[key]) if key in ('u1', 'u2', 'f1', 'f2') else par[key] for key in var]

            self.print_msg(" #{} |".format(wid + 1) + " {:>8f} " * 7, par)

    def log_report_calib(self, i=None):

        if i is None:
            i = self.log_select

        self.print_msg("Calibrated parameters:")
        self.print_msg("optimized ({}) on wires {}", (','.join(self.log_var[i]),
                                                      self.log_wlist[i]))
        self.print_msg("resulting in fval = {:.2f} and GoF = {:.2f}", (self.log_fval[i],
                                                                       self.log_gof[i]))
        self.print_msg("with parameter values and errors as:")

        var = ['R', 'h', 'p0', 'f1', 'f2', 'u1', 'u2']
        header = ['* ' + key + ' *' if (key in self.log_var[i]) or (key + 'e' in self.log_var[i]) else key for key
                  in var]
        self.print_msg("wire\\" + "{:^10}" * 7, header)

        for j, wid in enumerate(self.log_wlist[i]):
            par = self.log_wdict[i][wid]
            par = [np.rad2deg(par[key]) if key in ('u1', 'u2', 'f1', 'f2') else par[key] for key in var]

            err = self.log_err[i][j]
            err = [err[key] for key in var]  # angles are already in degrees

            self.print_msg(" #{} |".format(wid + 1) + " {:>8f} " * 7, par)
            self.print_msg("    |" + " {:>8f} " * 7, err)

        if 'dm' in self.log_var[i]:
            self.print_msg("and fitted depth = {:>8f} +- {:>8f}", (self.log_dm[i], self.log_err_dm[i]))

    def log_plot(self, i=-1):

        if i == -1:
            i = self.log_select

        self.plot_calib(self.log_wlist[i], self.log_sim_I[i])

    # Plotting functions
    def plot_points(self):

        fig = self.scan.plot_image(0)

        fig.suptitle("Calibration points.")

        ax = fig.gca()

        ax.set_prop_cycle('markerfacecolor', gen_colormap(self.scan.wire_qty))

        handles = []

        for XY in self.data_XYcam:
            h = ax.plot([x for x, _ in XY],
                        [y for _, y in XY],
                        's', mec='w', mew=1, ms=7)

            handles.extend(h)

        fig.legend(handles,
                   ["wire #{}".format(i + 1) for i in range(self.scan.wire_qty)])

    def plot_exp(self, wlist=None):

        if wlist is None:
            wlist = range(self.scan.wire_qty)

        for wid in wlist:
            self.plot_exp_wire(wid)

    def plot_exp_wire(self, wid):

        pass

    def plot_calib(self, wlist=None, sim_I=None):

        if wlist is None:
            wlist = range(self.scan.wire_qty)

        if sim_I is None:
            sim_I = self.log_sim_I[self.log_select]

        for wid in wlist:
            self.plot_calib_wire(wid, sim_I[wid])

    def plot_calib_wire(self, wid, sim_I):

        pass

    # Miscellaneous
    def calc_nvar(self, wlist, var, algo):

        nvar_prof = sum([len(self.data_XYcam[i]) for i in wlist])

        nvar_wire = 0

        for key in ['R', 'p0', 'h', 'f1', 'f2']:

            if key in var:
                nvar_wire = nvar_wire + len(wlist)

        for key in ['Re', 'u1', 'u2']:

            if key in var:
                nvar_wire = nvar_wire + 1

        return nvar_wire, nvar_prof

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

            msg = "[calib] " + pref + msg

            if mode == 'F':
                raise CalibError(msg)

            else:
                print(msg)
