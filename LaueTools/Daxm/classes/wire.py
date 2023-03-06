#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os

import math
import numpy as np
import json

import matplotlib.pylab as mplp

import LaueTools.Daxm.material.absorption as abso
import LaueTools.Daxm.material.dict_datamat as dm
import LaueTools.Daxm.modules.geometry as geom


def new_dict(material="W", R=0.025, h=1., p0=0., f1=0., f2=0., u1=0., u2=0., include_material=True):
    if include_material:
        return {'material': material,
                'R': R,
                'h': h,
                'p0': p0,
                'f1': f1,
                'f2': f2,
                'u1': u1,
                'u2': u2}
    else:
        return {'R': R,
                'h': h,
                'p0': p0,
                'f1': f1,
                'f2': f2,
                'u1': u1,
                'u2': u2}


def new_dict_traj(u1=0, u2=0):
    return {'u1': u1,
            'u2': u2}


def load_dict(filename, directory="", material=None):
    filedir = os.path.join(directory, filename)

    try:
        with open(filedir, 'r') as pfile:
            wire_dict = new_dict(**json.load(pfile))

    except ValueError:
        par = np.loadtxt(filedir)
        wire_dict = new_dict(R=par[0], h=par[2], p0=par[1],
                             f1=math.radians(par[3]), f2=math.radians(par[4]),
                             u1=math.radians(par[5]), u2=math.radians(par[6]),
                             material=material)

    return wire_dict


def save_dict(wire_dict, filename, directory=""):
    """Write the dict scan to file (json)."""
    filedir = os.path.join(directory, filename)

    with open(filedir, 'w') as pfile:
        pfile.write(json.dumps(wire_dict, indent=1))


def list_available_material():
    mat = list(dm.dict_mat.keys())

    for item in abso.list_available_element():
        mat.append(item)

    return mat


def gen_wires_grid(material, qty, incl, spacing, traj, radius, height, offset):

    sinIncl = math.sin(math.radians(float(incl)))
    cosIncl = math.cos(math.radians(float(incl)))

    sinTraj = math.sin(math.radians(float(traj)))
    cosTraj = math.cos(math.radians(float(traj)))

    if int(incl) == int(traj):
        h = [0.] * qty
        p0 = [-spacing * i for i in range(qty)]
    elif int(incl) == 0 and int(traj) == 40:
        h = [-spacing * i * sinTraj / cosTraj for i in range(qty)]
        p0 = [-spacing * i / cosTraj for i in range(qty)]
    else:
        h = [spacing * i * sinIncl for i in range(qty)]
        p0 = [-spacing * i * cosIncl for i in range(qty)]

    p0 = p0 - np.mean(p0) + offset
    h = h - np.mean(h) + height

    wires = [[material, float(radius), float(hi), float(p0i)] for p0i, hi in zip(p0, h)]

    return wires


class CircularWire:
    """Circular wire class"""
    # Constructors
    def __init__(self, material='W', R=0.025, f1=0., f2=0., u1=0., u2=0., h=1., p0=0.):

        # Attributes related to the material
        self.material = None
        self.absfun_coeff = None
        self.absfun_energy = None

        self.set_material(material)

        # Attributes describing geometry
        self.R = None
        self.f1 = None
        self.f2 = None
        self.u1 = None
        self.u2 = None
        self.h = None
        self.p0 = None
        self.axis = None
        self.traj = None

        self.set_par(R, f1, f2, u1, u2, h, p0)

    @classmethod
    def load(cls, filename, directory=""):

        return cls(**load_dict(filename, directory=directory))

    # Methods to set and modify properties of the wire
    def set(self, R, f1, f2, u1, u2, h, p0, material):

        self.set_par(R, f1, f2, u1, u2, h, p0)

        if material is not None:
            self.set_material(material)

    def set_fromfile(self, filename, directory=""):

        self.set(**load_dict(filename, directory=directory))

    def set_par(self, R, f1, f2, u1, u2, h, p0, material=None):

        self.set_radius(R)

        self.set_axis(f1, f2)

        self.set_traj(u1, u2)

        self.set_pos(h, p0)

    def set_material(self, material):

        self.material = material

        abscoeff, energy, _ = abso.calc_absorption(material, absolute=True)

        self.absfun_coeff = abscoeff

        self.absfun_energy = energy

    def set_radius(self, R):

        self.R = R

    def set_axis(self, f1=0., f2=0.):

        self.f1, self.f2 = f1, f2

        self.axis = geom.calc_axis(f1, f2)

    def set_axis_deg(self, f1=0., f2=0.):

        self.set_axis(math.radians(f1), math.radians(f2))

    def set_traj(self, u1=0., u2=0.):

        self.u1, self.u2 = u1, u2

        self.traj = geom.calc_traj(u1, u2)

    def set_traj_deg(self, u1=0., u2=0.):

        self.set_traj(math.radians(u1), math.radians(u2))

    def set_height(self, h):

        self.h = h

    def set_offset(self, p0=0.):

        self.p0 = p0

    def set_pos(self, h, p0=0.):

        self.set_height(h)

        self.set_offset(p0)

    # Methods to get attribute values
    def get_radius(self):

        return self.R

    def get_material(self):

        return self.material

    # Methods for geometry calculations
    def calc_position(self, p):
        """Calculate the reference point coordinates at motor position p"""
        dp = (p - self.p0)

        Ox = dp * self.traj[0]
        Oy = dp * self.traj[1]
        Oz = self.h + dp * self.traj[2]

        return Ox, Oy, Oz

    def calc_crosslength(self, p, ysrc, Pcam):
        """Calculate the length of the segment travelled in the wire by a given X-ray"""
        # wire positions
        p = np.reshape(p, (-1, 1))
        # depth positions
        ysrc = np.reshape(ysrc, (1, -1))

        # components of OwY vector
        OYx, OYy, OYz = self.calc_position(p)
        OYx, OYy, OYz = -OYx, ysrc - OYy, -OYz

        # components of v = YP / |YP| 
        vx, vy, vz = Pcam[0], Pcam[1] - ysrc, Pcam[2]
        vn = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        vx, vy, vz = vx / vn, vy / vn, vz / vn

        # cross product v.f
        vfx = vy * self.axis[2] - vz * self.axis[1]
        vfy = vz * self.axis[0] - vx * self.axis[2]
        vfz = vx * self.axis[1] - vy * self.axis[0]
        vf2 = vfx ** 2 + vfy ** 2 + vfz ** 2

        # cross product OY.f
        Ofx = OYy * self.axis[2] - OYz * self.axis[1]
        Ofy = OYz * self.axis[0] - OYx * self.axis[2]
        Ofz = OYx * self.axis[1] - OYy * self.axis[0]
        Of2 = Ofx ** 2 + Ofy ** 2 + Ofz ** 2

        # calculate A, B, C and Delta  in  A*d^2 + B*d + C = 0
        A = vf2

        B = 2. * (Ofx * vfx + Ofy * vfy + Ofz * vfz)

        C = Of2 - self.R ** 2

        Delta = B ** 2 - 4. * A * C

        # traveled distance in the wire = difference between solutions = sqrt(delta)/a
        abslength = np.divide(np.sqrt(np.maximum(Delta, 0)), A)

        return abslength

    def calc_distance_to(self, p, ysrc, Pcam):
        """Calculate the shortest distance between the wire axis and a given X-ray"""
        # wire positions
        p = np.reshape(p, (-1, 1))
        # depth positions
        ysrc = np.reshape(ysrc, (1, -1))

        # components of OwY vector
        OYx, OYy, OYz = self.calc_position(p)
        OYx, OYy, OYz = -OYx, ysrc - OYy, -OYz

        # components of v = YP / |YP| 
        vx, vy, vz = Pcam[0], Pcam[1] - ysrc, Pcam[2]
        vn = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        vx, vy, vz = vx / vn, vy / vn, vz / vn

        # cross product v.f
        vfx = vy * self.axis[2] - vz * self.axis[1]
        vfy = vz * self.axis[0] - vx * self.axis[2]
        vfz = vx * self.axis[1] - vy * self.axis[0]
        vf2 = vfx ** 2 + vfy ** 2 + vfz ** 2

        # distance rays and wire axis
        return np.divide(np.fabs(OYx * vfx + OYy * vfy + OYz * vfz), np.sqrt(vf2)) - self.R

    # intersect_ray_axis
    def intersect_ray_axis(self, ysrc, Pcam):
        """Calculate the motor position at which a given X-ray will intersect the wire axis"""
        Y = np.array([0, ysrc, 0])
        Oh = np.array([0, 0, self.h])

        # solve Y + e1*v = Ow + e2*f
        # as A*x = B
        # with e1, e2, p as unknowns

        # write A
        v = Pcam - Y
        v = v / np.linalg.norm(v)

        A = np.array([v, -self.axis, -self.traj])

        # write B
        B = Oh - Y - self.p0 * self.traj

        # result [tmp1, tmp2, p]
        x = np.linalg.solve(A.transpose(), B)

        return x[2]

    # intersect_ray_frontback
    def intersect_ray_fronts(self, ysrc, Pcam):
        """Calculate the 2 motor positions at which a given X-ray will intersect the wire surface"""
        # depth positions
        ysrc = np.reshape(ysrc, (-1, 1))
        Y = ysrc * np.array([[0., 1., 0.]])

        # ray vectors
        v = np.array([Pcam]) - Y
        v = v / np.linalg.norm(v, axis=1, keepdims=True)

        # prepare calculations
        fv = np.cross(np.array([self.axis]), v)
        fvu = np.sum(fv * np.array([self.traj]), axis=1, keepdims=True)

        H = Y - np.array([self.calc_position(0)])
        fvH = np.sum(fv * H, axis=1, keepdims=True)

        # equation coefficients A*p^2 + B*p + C = 0
        A = fvu ** 2
        B = -2. * np.sum(fvH * fvu, axis=1, keepdims=True)
        C = fvH * fvH - np.sum(fv * fv, axis=1, keepdims=True) * self.R ** 2

        A, B, C = np.sign(A) * A, np.sign(A) * B, np.sign(A) * C

        # solutions
        Delta = B ** 2 - 4. * A * C

        pback = 0.5 * (-B + np.sqrt(Delta)) / A
        pfront = 0.5 * (-B - np.sqrt(Delta)) / A

        return pfront.squeeze(), pback.squeeze()

    def mask_fronts(self, p, Pcam):
        """Calculate the depth interval masked by the wire for a given pixel"""
        # wire positions
        p = np.reshape(p, (-1, 1))

        Ow = self.calc_position(p.squeeze())
        Ow = np.array(Ow).transpose()

        OwP = np.array([Pcam]) - Ow

        # cross products
        fey = np.cross(self.axis, np.array([0, 1., 0]))
        fP = np.cross(self.axis, np.array(Pcam))

        # mixed products
        feyOwP = np.sum(np.array([fey]) * OwP, axis=1, keepdims=True)
        fPOwP = np.sum(np.array([fP]) * OwP, axis=1, keepdims=True)

        # equation coefficients A*p^2 + B*p + C = 0
        R2 = self.R ** 2
        A = feyOwP ** 2 - np.sum(fey * fey) * R2
        B = -2. * fPOwP * feyOwP + 2. * np.dot(fey, fP) * R2
        C = fPOwP ** 2 - np.sum(fP * fP) * R2

        A, B, C = np.sign(A) * A, np.sign(A) * B, np.sign(A) * C

        # solutions
        Delta = B ** 2 - 4. * A * C

        yfront = 0.5 * (-B + np.sqrt(Delta)) / A
        yback = 0.5 * (-B - np.sqrt(Delta)) / A

        return yfront.squeeze(), yback.squeeze()

    def is_crossed(self, p, ysrc, Pcam):
        """Test if the wire is crossed by a given X-ray"""
        return self.calc_distance_to(p, ysrc, Pcam) > self.R

    # Methods related to material absorption and transmission
    def calc_abscoeff(self, energy):
        """Calculate and interpolate the energy-absorption function of the constituent material"""
        return np.interp(energy, self.absfun_energy, self.absfun_coeff)

    def calc_transmission(self, p, ysrc, Pcam, energy=None, abscoeff=None):
        """Calculate the transmission rate of a given X-ray"""
        if abscoeff is None:
            abscoeff = self.calc_abscoeff(energy)

        length = self.calc_crosslength(p, ysrc, Pcam)

        return np.exp(-abscoeff * length)

    # Methods used to plot stuff
    def plot_transmission(self, energy=None, fontsize=14, label=None, fig=None, show=True):

        if energy is None:
            energy = [5., 10., 15., 20., 25.]

        # data
        coeff = self.calc_abscoeff(energy)

        R = self.get_radius()

        x = np.arange(0, R + 0.0001, 0.0001)

        yy = [np.exp(-c * 2. * np.sqrt(R ** 2 - x ** 2)) for c in coeff]

        # plot
        if fig is None:
            fig = mplp.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        if label is None:
            label = 'the ' + self.get_material() + ' wire'

        title = 'Transmission profile of ' + label

        xlabel = 'Distance from wire axis (um)'

        ylabel = 'Transmitted intensity I/I0'

        lgd = ["%d keV" % e for e in energy]

        ax.plot(x * 1000, np.transpose(yy), linewidth=2)

        ax.set_xlim(0, 1000 * R + 5)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        fig.suptitle(title, fontsize=fontsize + 2)
        mplp.legend(lgd, loc='upper left', fontsize=fontsize)

        if show:
            fig.show()

# End of class wire
