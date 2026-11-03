#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fluorescence calibration of the DAXM wires with a layer above the fluorescent material (e.g. Cr coating on Zr)

CalibFluoLayer is the fluorescence calibration of the legacy classes (CalibFluo, as Daxm/scripts/calib.py) with one
more source in the model: a layer of thickness d just above the surface of the fluorescent material (sources at
ybeam in [-d, 0], ybeam = 0 at the surface of the fluorescent material), uniform along the beam, contributing a
fraction of the detected signal (fL, free for each profile: it depends on the absorption of the layer along the way
out, i.e. on the pixel). It is the signal of the layer that passes the energy threshold of the detector (fluorescence
of the layer partly above the threshold, beam scattered by the layer, secondary fluorescence...).

Without it, the shallower signal of the layer pulls the fitted surface of the fluorescent material towards the
surface (about 1.3 micron along the beam for Zr5 of a321217: 5 % of the signal from a ~24 microns Cr layer) and blurs
the edges of the dips.

Variables: those of CalibFluo, plus 'ld' (thickness of the layer along the beam, mm, common) and 'fL' (one per
profile: intensity of the layer / intensity of the fluorescent material).

The legacy class is not changed (reference of the former calibrations).
"""
from typing import List

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import expit

from LaueTools.Daxm.classes.calibration.fluorescence import CalibFluo
import LaueTools.Daxm.modules.geometry as geom
import LaueTools.Daxm.modules.calibration as calib

# sharpness of the limits of the layer (mm^-1), as the depth mask of CalibFluo
_EDGE = 3000.


def layer_profile(y: np.ndarray, thickness: float) -> np.ndarray:
    """source of the layer along the beam (y, mm): uniform in [-thickness, 0], integral 1"""
    prof = expit(_EDGE * (y + thickness)) * expit(-_EDGE * y)
    norm = trapezoid(prof, y)
    return prof / norm if norm > 0 else prof


class CalibFluoLayer(CalibFluo):
    """fluorescence calibration (CalibFluo) with a layer above the fluorescent material (see module docstring)

    :param layer_thickness: start value of the thickness of the layer along the beam (mm)
    :param layer_max: largest thickness of the layer along the beam (mm): the model of the sources is extended to
        ybeam = -layer_max
    :param fit_thickness: False: thickness fixed to layer_thickness (e.g. the one of a reference calibration)
    """

    def __init__(self, scan, sample, layer_thickness: float = 0.025, layer_max: float = 0.08,
                 fit_thickness: bool = True, verbose=True):
        self.fit_thickness = bool(fit_thickness)
        self.layer_thickness0 = float(layer_thickness)
        self.layer_max = float(max(layer_max, 1.5 * layer_thickness))
        self.var_ld = self.layer_thickness0
        self.var_fL = np.zeros(0)
        self.log_ld, self.log_fL = [], []
        CalibFluo.__init__(self, scan, sample, verbose)

    # sources: grid extended above the surface of the fluorescent material (no fluorescence there)
    def run_init_sample(self, verbose=0):
        CalibFluo.run_init_sample(self, verbose)
        y = np.ravel(self.src_y)
        step = float(np.median(np.diff(y)))
        above = -np.arange(int(np.ceil(self.layer_max / step)), 0, -1) * step
        self.src_y = np.concatenate([above, y])
        pad = [(0, 0)] * (self.src_I.ndim - 1) + [(len(above), 0)]
        self.src_I = np.pad(self.src_I, pad)

    # variables: CalibFluo ones, then ld, then (kM, km, fL) per profile
    def run_init_var(self, var):
        CalibFluo.run_init_var(self, var)
        nb = self.exp_qty
        head_x, head_k = list(self.var_x0[:-2 * nb]), list(self.var_key[:-2 * nb])
        pairs_x = self.var_x0[-2 * nb:]
        start = self.log_ld[-1] if self.log_ld else self.layer_thickness0   # previous stage, if any
        x0, keys = head_x + [start], head_k + ['ld']
        for i in range(nb):
            x0 += [pairs_x[2 * i], pairs_x[2 * i + 1], 0.05]
            keys += ['kM', 'km', 'fL']
        self.var_x0, self.var_key = x0, keys

    def run_init_bounds(self, bounds):
        nb = self.exp_qty
        x0, keys = self.var_x0, self.var_key
        nhead = len(keys) - 1 - 3 * nb
        # bounds of the CalibFluo variables, computed with the CalibFluo layout
        self.var_x0 = x0[:nhead] + [v for i in range(nb) for v in x0[nhead + 1 + 3 * i:nhead + 3 + 3 * i]]
        self.var_key = keys[:nhead] + ['kM', 'km'] * nb
        CalibFluo.run_init_bounds(self, bounds)
        lb, ub = list(self.var_lb), list(self.var_ub)
        self.var_x0, self.var_key = x0, keys
        if self.fit_thickness:
            self.var_lb = lb[:nhead] + [0.002]
            self.var_ub = ub[:nhead] + [self.layer_max - 0.001]
        else:
            self.var_lb = lb[:nhead] + [x0[nhead]]
            self.var_ub = ub[:nhead] + [x0[nhead]]
        for i in range(nb):
            self.var_lb += [lb[nhead + 2 * i], lb[nhead + 2 * i + 1], 0.]
            self.var_ub += [ub[nhead + 2 * i], ub[nhead + 2 * i + 1], 1.]

    def run_optim_unpack(self, x):
        for i, lut in enumerate(self.var_lut):
            for key in lut.keys():
                if lut[key] is not None:
                    self.var_cur[i][key] = float(x[lut[key]])
        nb = self.exp_qty
        tail = np.asarray(x[-3 * nb:], dtype=float)
        self.var_kM, self.var_km, self.var_fL = tail[0::3], tail[1::3], tail[2::3]
        self.var_ld = float(x[-3 * nb - 1])
        if 'dm' in self.opt_var:
            self.var_dm = float(x[-3 * nb - 2])

    def _sources(self) -> List[np.ndarray]:
        """sources of each profile (fluorescent material + layer), on the line of the first energy"""
        layer = layer_profile(np.ravel(self.src_y), self.var_ld)
        res = []
        for i, sim in enumerate(self.sim_I0):
            sources = np.array(sim, dtype=float, copy=True)
            sources[0, 0, :] += self.var_fL[i] * layer
            res.append(sources)
        return res

    def run_optim_fun(self, x):
        self.run_optim_unpack(x)
        R = np.array([var['R'] for var in self.var_cur])
        h = np.array([var['h'] for var in self.var_cur])
        p0 = np.array([var['p0'] for var in self.var_cur])
        axis = np.array([geom.calc_axis(var['f1'], var['f2']) for var in self.var_cur])
        traj = np.array([geom.calc_traj(var['u1'], var['u2']) for var in self.var_cur])
        self.opt_objfun(R, h, p0, axis, traj, self.var_dm, self.var_kM, self.var_km,
                        self.exp_Pcam, self.exp_pw, self.exp_wire, self.src_y, self.wire_mu,
                        self._sources(), self.sim_I, self.exp_I, self.opt_residuals)
        return np.array(self.opt_residuals)

    def run_optim_sim(self, x) -> List:
        self.run_optim_unpack(x)
        axis = [geom.calc_axis(var['f1'], var['f2']) for var in self.var_cur]
        traj = [geom.calc_traj(var['u1'], var['u2']) for var in self.var_cur]
        if 'dm' in self.opt_var:
            dmask = calib.objfun_depth_mask(self.src_y[0], self.var_dm)
        else:
            dmask = np.ones(len(self.src_y[0]))
        sources = self._sources()
        Isim = []
        for i, wire in enumerate(self.exp_wire):
            dist = geom.calc_wire_abslength(self.var_cur[wire]['R'], self.var_cur[wire]['h'], self.var_cur[wire]['p0'],
                                            axis[wire], traj[wire], self.exp_pw[i], self.src_y, self.exp_Pcam[i])
            Iabs = [np.exp(-dist * mu) for mu in self.wire_mu[i]]
            Isim.append(self.var_kM[i] * np.sum(trapezoid(sources[i] * Iabs * dmask, self.src_y[0], axis=2), axis=0)
                        + self.var_km[i])
        return Isim

    def run_result(self, res):
        CalibFluo.run_result(self, res)
        self.run_optim_unpack(res.x)
        self.log_ld.append(self.var_ld)
        self.log_fL.append(np.array(self.var_fL))
        fraction = self.var_fL / (1. + self.var_fL)
        self.print_msg("   layer: thickness {:.1f} microns along the beam, fraction of the signal {:.3f} "
                       "(median; {:.3f} - {:.3f})", (self.var_ld * 1000., float(np.median(fraction)),
                                                     float(np.min(fraction)), float(np.max(fraction))))

    def log_restart(self):
        CalibFluo.log_restart(self)
        self.log_ld, self.log_fL = [], []
