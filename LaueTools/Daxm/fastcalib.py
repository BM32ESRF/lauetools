#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast and parallel optimizer of the fluorescence calibration of the DAXM wires

Same model, variables, bounds, results and log as Calib.run() of the legacy classes (CalibFluo, and CalibFluoLayer of
LaueTools.Daxm.fluolayer), with two changes of the optimization only:

1. the linear parameters of each profile (kM: amplitude, km: offset, and fL: fraction of the layer for
   CalibFluoLayer) are solved exactly (non negative least squares) at each evaluation, instead of being variables of
   the non linear least squares: the variables are only the wires parameters (and dm, layer thickness), ~20 instead
   of ~100-450 for a 6x6 grid per wire
2. the jacobian (finite differences, as scipy '2-point') is computed in parallel, one process per variable (fork of
   the process with the calibration loaded)

At the end, the full jacobian (wires and linear parameters, legacy residuals) is computed in parallel too, so that
the errors of the parameters are those of Calib.run_result().

Usage: run_fast(calib, wire=None, var=['h', 'p0', 'axis'], nb_cpus=None) in place of calib.run(wire, var)
"""
import multiprocessing
import os
import time
import types
from typing import Optional, Sequence

import numpy as np
from scipy import optimize

LINEAR_KEYS = ('kM', 'km', 'fL')
X_SCALE = 'jac'   # scaling of the variables of least_squares (mm, radians, very different sensitivities)

_STATE = {}   # calibration problem of this process (inherited by the forked workers)


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class FastProblem:
    """non linear least squares of the fluorescence calibration with the linear parameters solved exactly"""

    def __init__(self, calib):
        self.calib = calib
        keys = list(calib.var_key)
        self.x_full0 = np.asarray(calib.var_x0, dtype=float)
        self.lb_full = np.asarray(calib.var_lb, dtype=float)
        self.ub_full = np.asarray(calib.var_ub, dtype=float)
        self.i_kM = [i for i, k in enumerate(keys) if k == 'kM']
        self.i_km = [i for i, k in enumerate(keys) if k == 'km']
        self.i_fL = [i for i, k in enumerate(keys) if k == 'fL']
        others = [i for i, k in enumerate(keys) if k not in LINEAR_KEYS]
        # variables with lb == ub stay fixed (e.g. thickness of the layer of a reference calibration)
        self.free = [i for i in others if self.ub_full[i] - self.lb_full[i] > 1e-12]
        self.x_full = self.x_full0.copy()
        self.nb_eval = 0

    @property
    def x0(self) -> np.ndarray:
        return self.x_full0[self.free]

    @property
    def bounds(self):
        return self.lb_full[self.free], self.ub_full[self.free]

    def _profiles(self, x):
        """model profiles of each measured profile: fluorescent material (kM = 1, km = 0, fL = 0), and the layer"""
        calib = self.calib
        x = x.copy()
        x[self.i_kM], x[self.i_km] = 1., 0.
        if self.i_fL:
            x[self.i_fL] = 0.
        material = [np.ravel(v) for v in calib.run_optim_sim(x)]
        layer = None
        if self.i_fL:
            x[self.i_fL] = 1.
            layer = [np.ravel(both) - one for both, one in zip(calib.run_optim_sim(x), material)]
        return material, layer

    def residuals(self, xfree, store: bool = False) -> np.ndarray:
        x = self.x_full.copy()
        x[self.free] = xfree
        material, layer = self._profiles(x)
        res = []
        for i, intensity in enumerate(self.calib.exp_I):
            intensity = np.ravel(intensity).astype(float)
            columns = [material[i]] + ([layer[i]] if layer is not None else []) + [np.ones_like(intensity)]
            design = np.vstack(columns).T
            coef, _ = optimize.nnls(design, intensity)
            res.append(design @ coef - intensity)
            if store:
                x[self.i_kM[i]], x[self.i_km[i]] = coef[0], coef[-1]
                if layer is not None:
                    x[self.i_fL[i]] = coef[1] / coef[0] if coef[0] > 0 else 0.
        self.nb_eval += 1
        if store:
            self.x_full = x
        return np.concatenate(res)


def _step(x, lb, ub):
    """finite difference steps as scipy '2-point', reversed at an upper bound"""
    h = np.sqrt(np.finfo(float).eps) * np.maximum(1., np.abs(x))
    return np.where(x + h > ub, -h, h)


def _column_free(args):
    j, x, h, f0 = args
    xp = np.array(x, dtype=float)
    xp[j] += h
    return (_STATE['problem'].residuals(xp) - f0) / h


def _column_full(args):
    j, x, h, f0 = args
    xp = np.array(x, dtype=float)
    xp[j] += h
    return (np.array(_STATE['problem'].calib.run_optim_fun(xp)) - f0) / h


def _parallel_jacobian(function, column, x, lb, ub, pool, f0=None):
    f0 = function(x) if f0 is None else f0
    h = _step(x, lb, ub)
    tasks = [(j, x, h[j], f0) for j in range(len(x))]
    columns = pool.map(column, tasks, chunksize=1) if pool is not None else [column(t) for t in tasks]
    return np.column_stack(columns)


def _result_errors(calib, residuals, chisq, jac):
    """errors as Calib.run_result_errors(), with a pseudo inverse (a parameter at a bound, e.g. fL = 0)"""
    nvar, ndat = len(calib.var_x0), calib.exp_size
    nu = ndat - nvar
    fval = chisq / nu
    boxsize = np.prod(np.array(calib.data_hbs) * 2 + 1)
    gof = np.divide(np.square(residuals), np.clip(np.concatenate(calib.exp_I), 1., None))
    gof = 1. / nu * np.sum(gof) * boxsize
    covar = np.linalg.pinv(np.matmul(np.transpose(jac), jac))
    err = np.sqrt(np.abs(fval * np.diag(covar)))
    calib.run_optim_unpack(err)
    errors = [var.copy() for var in calib.var_cur]
    return errors, calib.var_dm, fval, gof


def run_fast(calib, wire: Optional[Sequence[int]] = None, var=None, bounds=None, nb_cpus: Optional[int] = None,
             verbose: bool = True) -> dict:
    """calibration of var on wires (as calib.run(wire, var, bounds)) with the fast and parallel optimizer

    :param nb_cpus: processes of the jacobian (default: all the cpus of this machine / SLURM job)
    :return: dict: seconds, nb_variables (wires parameters optimized), nb_evaluations, nb_cpus, cost
    """
    t0 = time.time()
    calib.run_arg(wire, var, bounds, 0)
    calib.print_msg("Calibration of {} on wires {} (fast optimizer)...", (var, wire))
    calib.print_msg("- Initializing:")
    calib.run_init()
    calib.opt_residuals = np.zeros(calib.exp_size)
    problem = FastProblem(calib)
    problem.x_full = problem.x_full0.copy()
    lb, ub = problem.bounds
    x0 = np.clip(problem.x0, lb, ub)
    nb_cpus = int(nb_cpus or _available_cpus())
    nb_proc = max(1, min(nb_cpus, len(x0) + 1, len(problem.x_full0)))
    _STATE['problem'] = problem
    pool = None
    if nb_proc > 1 and 'fork' in multiprocessing.get_all_start_methods():
        pool = multiprocessing.get_context('fork').Pool(nb_proc)
    try:
        calib.print_msg("- Running ({} variables, linear parameters of {} profiles solved exactly, {} processes)...",
                        (len(x0), calib.exp_qty, nb_proc if pool is not None else 1))

        def jac(x):
            return _parallel_jacobian(problem.residuals, _column_free, x, lb, ub, pool)

        res = optimize.least_squares(problem.residuals, x0, jac=jac, bounds=(lb, ub), method='trf',
                                     ftol=1e-08, xtol=1e-08, x_scale=X_SCALE, verbose=2 if verbose else 0)
        problem.residuals(res.x, store=True)   # linear parameters at the optimum
        x_full = problem.x_full
        calib.print_msg("- Errors (jacobian of the {} parameters)...", (len(x_full),))
        fun = np.array(calib.run_optim_fun(x_full), dtype=float)
        jac_full = _parallel_jacobian(lambda x: np.array(calib.run_optim_fun(x)), _column_full, x_full,
                                      problem.lb_full - 1e300, problem.ub_full, pool, f0=fun)
    finally:
        if pool is not None:
            pool.terminate()
        _STATE.pop('problem', None)
    calib.print_msg("- Getting results...")
    calib.run_result_errors = types.MethodType(_result_errors, calib)
    try:
        calib.run_result(types.SimpleNamespace(x=x_full, fun=fun, cost=0.5 * float(np.sum(fun ** 2)), jac=jac_full))
    finally:
        del calib.run_result_errors
    calib.log_report()
    summary = {'seconds': round(time.time() - t0, 1), 'nb_variables': int(len(x0)),
               'nb_evaluations': int(res.nfev), 'nb_jacobians': int(res.njev or 0), 'nb_cpus': nb_proc,
               'cost': float(res.cost)}
    if verbose:
        print(f"fast calibration: {summary['nb_variables']} variables, {summary['nb_evaluations']} evaluations, "
              f"{summary['nb_jacobians']} jacobians, {summary['nb_cpus']} processes, {summary['seconds']:.0f} s")
    return summary
