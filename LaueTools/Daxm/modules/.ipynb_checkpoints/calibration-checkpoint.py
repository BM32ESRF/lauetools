#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np
import numba as nb

from numba.typed import List
#https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types

import LaueTools.Daxm.modules.geometry as geom


@nb.jit(nopython=True, cache=True, parallel=False) # /!\ parallel option must be FALSE
def objfun_residuals(R, h, p0, axis, traj, dm, kM, km,
                         Pcam, pw, wire, ysrc, mu,
                         sim_I0, sim_I, exp_I, residuals):

    for i in range(len(Pcam)):

        wid = wire[i]

        dist = geom.calc_wire_abslength(R[wid], h[wid], p0[wid], axis[wid], traj[wid], pw[i], ysrc, Pcam[i])

        Itmp = [np.exp(-dist * mu_i) for mu_i in mu[i]]

        for ip in range(len(sim_I[i])):

            sim_I[i][ip] = 0.

            for iy in range(len(ysrc[0]) - 1):

                tmp = 0.5 * (ysrc[0][iy + 1] - ysrc[0][iy])

                for k in range(len(mu[i])):

                    sim_I[i][ip] += tmp * (Itmp[k][ip][iy+1] * sim_I0[i][k][0][iy+1] + Itmp[k][ip][iy] * sim_I0[i][k][0][iy] )

    k = 0
    for i in range(len(sim_I)):
        for j in range(len(sim_I[i])):
            residuals[k] = sim_I[i][j]* kM[i] + km[i] - exp_I[i][j]
            k += 1


@nb.jit(nopython=True, cache=True, parallel=False) # /!\ parallel option must be FALSE
def objfun_depth_residuals(R, h, p0, axis, traj, dm, kM, km,
                         Pcam, pw, wire, ysrc, mu,
                         sim_I0, sim_I, exp_I, residuals):

    dmask = [float(0.)] * len(ysrc[0])

    for i in range(len(dmask)):
        dmask[i] = objfun_depth_mask(float(ysrc[0][i]),float(dm))

    for i in range(len(Pcam)):

        wid = wire[i]

        dist = geom.calc_wire_abslength(R[wid], h[wid], p0[wid], axis[wid], traj[wid], pw[i], ysrc, Pcam[i])

        Itmp = [np.exp(-dist * mu_i) for mu_i in mu[i]]

        #I0_integ = float(0.)

        for ip in range(len(sim_I[i])):

            sim_I[i][ip] = 0.

            for iy in range(len(ysrc[0]) - 1):

                tmp = 0.5 * (ysrc[0][iy + 1] - ysrc[0][iy])

                for k in range(len(mu[i])):

                   # I0_integ += tmp * (sim_I0[i][k][0][iy+1] * dmask[iy+1] + sim_I0[i][k][0][iy] * dmask[iy])

                    sim_I[i][ip] += tmp * (Itmp[k][ip][iy+1] * sim_I0[i][k][0][iy+1] * dmask[iy+1]
                                           + Itmp[k][ip][iy] * sim_I0[i][k][0][iy] * dmask[iy])

        #for ip in range(len(sim_I[i])):
            # sim_I[i][ip] = sim_I[i][ip] / I0_integ

    k = 0
    for i in range(len(sim_I)):
        for j in range(len(sim_I[i])):
            residuals[k] = sim_I[i][j]* kM[i] + km[i] - exp_I[i][j]
            k += 1


@nb.jit(nopython=True, cache=True, parallel=False)
def objfun_depth_mask(y, dm):

    return 1./ (1. + np.exp(3000.*(y - dm)))


