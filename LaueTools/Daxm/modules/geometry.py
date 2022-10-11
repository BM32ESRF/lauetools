#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import math

import numpy as np
import numba as nb


def transf_pix_to_coo (ccd_parameters, xcam, ycam):
    """Transform image pixel coordinates to laboratory coordinates"""
    detect, xcen, ycen, xbet, xgam, pixelsize = ccd_parameters

    cosbeta = np.cos(math.pi / 2. - xbet)
    sinbeta = np.sin(math.pi / 2. - xbet)

    cosgam = np.cos(- xgam )
    singam = np.sin(- xgam )

    xcam1 = (xcam - xcen) * pixelsize
    ycam1 = (ycam - ycen) * pixelsize

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    xO = 0.
    yO = detect * cosbeta
    zO = detect * sinbeta

    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = -yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xlab = xO + xOM
    ylab = yO + yOM
    zlab = zO + zOM

    return np.array([xlab, ylab, zlab])


def transf_vect_to_pix(ccd_parameters, vect, ysrc=0):
    """Calculate the pixel coordinates where a given X-ray would hit the detector"""
    detect, xcen, ycen, xbet, xgam, pixelsize = ccd_parameters

    cosbeta = np.cos(math.pi / 2. - xbet)
    sinbeta = np.sin(math.pi / 2. - xbet)

    cosgam = np.cos( -xgam )
    singam = np.sin( -xgam )

    n  = np.array([0., cosbeta, sinbeta])
    Dc = detect * n

    Y = np.array([0, ysrc, 0])

    ksi = np.divide( np.dot(Dc -Y, n) , np.dot(vect, n))

    xOM = Y[0] + ksi * vect[0] - Dc[0]
    yOM = Y[1] + ksi * vect[1] - Dc[1]
    zOM = Y[2] + ksi * vect[2] - Dc[2]

    xca0 = xOM
    yca0 = (yOM + zOM) / (sinbeta - cosbeta)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    return xcam1 / pixelsize + xcen, ycam1 / pixelsize + ycen


def calc_ray_length(ccd_parameters, ysrc, xcam, ycam):
    """Calculate the length of a given X-ray from sample to detector"""
    detect, xcen, ycen, _, _, pixelsize = ccd_parameters

    dx = (xcam-xcen)*pixelsize
    dy = ysrc-(ycam-ycen)*pixelsize
    dz = detect

    lray = np.sqrt(dy**2 + dz**2 + dx**2)

    return lray


def calc_solid_angle_coeff(ccd_parameters, xcam, ycam):
    """Evaluate the solid angle correction for a detector pixel"""
    detect = ccd_parameters[0]

    lray = calc_ray_length(ccd_parameters, 0, xcam, ycam)

    return (detect/lray)**3 # 2 for spherical wave + 1 for plane detector


# Wire geometry
def calc_axis(f1, f2):
    """Return the axis vector of a wire"""
    return  np.array([np.cos(f1) * np.cos(f2), np.sin(f1) * np.cos(f2), -np.sin(f2)])


def calc_traj(u1, u2):
    """Return the translation vector of a wire"""
    return np.array([ -np.sin(u1) * np.cos(u2), np.cos(u1) * np.cos(u2), np.sin(u2)])  


@nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64, nb.float64[:],
                         nb.float64[:], nb.float64[:, :], nb.float64[:, :], nb.float64[:]),
        nopython=True, cache=True)
def calc_wire_abslength(R, h, p0, axis, traj, p, ysrc, Pcam):
    # components of OwY vector
    dp = (p - p0)
    Ox = dp * traj[0]
    Oy = dp * traj[1]
    Oz = h + dp * traj[2]
    OYx, OYy, OYz = -Ox, ysrc - Oy, -Oz

    # components of v = YP / |YP|
    vx, vy, vz = Pcam[0], Pcam[1] - ysrc, Pcam[2]
    vn = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    vx, vy, vz = vx / vn, vy / vn, vz / vn

    # cross product v.f
    vfx = vy * axis[2] - vz * axis[1]
    vfy = vz * axis[0] - vx * axis[2]
    vfz = vx * axis[1] - vy * axis[0]
    vf2 = vfx ** 2 + vfy ** 2 + vfz ** 2

    # cross product OY.f
    Ofx = OYy * axis[2] - OYz * axis[1]
    Ofy = OYz * axis[0] - OYx * axis[2]
    Ofz = OYx * axis[1] - OYy * axis[0]
    Of2 = Ofx ** 2 + Ofy ** 2 + Ofz ** 2

    # calculate A, B, C and Delta  in  A*d^2 + B*d + C = 0
    A = vf2

    B = 2. * (Ofx * vfx + Ofy * vfy + Ofz * vfz)

    C = Of2 - R ** 2

    Delta = B ** 2 - 4. * A * C

    # traveled distance in the wire = difference between solutions = sqrt(delta)/a
    abslength = np.divide(np.sqrt(np.maximum(Delta, 0.)), A)

    return abslength


# Sample geometry
@nb.jit(nb.float64[:, :](nb.float64, nb.float64[:, :], nb.float64[:], nb.boolean),
        nopython=True, cache=True)
def calc_sample_abslength(surface_angle, ysrc, pcam, relative=False):
    """Calculate the distance traveled in the sample by a given X-ray"""
    tmp1 = np.sqrt(pcam[0] ** 2 + pcam[2] ** 2 + (pcam[1] - ysrc) ** 2)
    tmp2 = pcam[2] * math.cos(surface_angle) + math.sin(surface_angle) * (ysrc - pcam[1])

    if relative:
        return math.sin(surface_angle) * np.divide(tmp1, tmp2)

    else:
        return math.sin(surface_angle) * ysrc * np.divide(tmp1, tmp2)
