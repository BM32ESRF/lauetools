#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os, sys

import numpy as np


CONVERSION_EPS = 1e-8


class OrientationError(Exception):
    pass


class Orientation(object):

    # Constructors
    def __init__(self, mat=None):

        if mat is None:
            mat = np.identity(3, dtype=np.float32)

        if mat.shape != (3, 3):
            raise OrientationError("3x3 matrix expected for initialization!")

        self.mat = np.matrix(mat, dtype=np.float32)

    @classmethod
    def mat(cls, mat):
        # from rotation matrix
        return cls(mat)

    @classmethod
    def euler_deg(cls, euler):
        # from euler angles in degrees according to bunge convention
        return cls.euler(np.radians(euler))

    @classmethod
    def euler(cls, euler_rad):
        # from euler angles in radians
        return cls(cls.euler_to_mat(euler_rad))

    @classmethod
    def rtheta(cls, rtheta):
        # from rotation axis/angle pair
        return cls(cls.rtheta_to_mat(rtheta))

    @classmethod
    def rod(cls, rod):
        # from rodrigues vector
        return cls(cls.rod_to_mat(rod))

    @classmethod
    def quat(cls, quat):
        # from quaternion
        return cls(cls.quat_to_mat(quat))

    # Descriptor conversion methods
    @staticmethod
    def euler_to_mat(euler):

        phi1, phi, phi2 = euler

        mat = np.identity(3, dtype=np.float32)

        cosphi1, sinphi1 = np.cos(phi1), np.sin(phi1)
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        cosphi2, sinphi2 = np.cos(phi2), np.sin(phi2)

        mat[0, 0] = cosphi1*cosphi2 - sinphi1*sinphi2*cosphi
        mat[0, 1] = sinphi1*cosphi2 + cosphi1*sinphi2*cosphi
        mat[0, 2] = sinphi2*sinphi
        mat[1, 0] = -cosphi1*sinphi2 - sinphi1*cosphi2*cosphi
        mat[1, 1] = -sinphi1*sinphi2 + cosphi1*cosphi2*cosphi
        mat[1, 2] = cosphi2*sinphi
        mat[2, 0] = sinphi1*sinphi
        mat[2, 1] = -cosphi1*sinphi
        mat[2, 2] = cosphi

        return np.matrix(mat)

    @staticmethod
    def mat_to_euler(mat):

        phi = np.arccos(mat[2, 2])

        if phi < CONVERSION_EPS or np.abs(phi - np.pi) < CONVERSION_EPS:

            phi1 = np.arctan2(mat[2, 0], -mat[2, 1])
            phi2 = np.arctan2(mat[0, 2],  mat[1, 2])

        else:  # by convention
            phi1 = np.arctan2(mat[0, 1], mat[0, 0])
            phi2 = 0

        return [phi1, phi, phi2]

    @staticmethod
    def rtheta_to_mat(rtheta):

        mat = np.identity(3, dtype=np.float32)

        r, theta = rtheta
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        mat[0, 0] = r[0]*r[0]*(1 - costheta) + costheta
        mat[0, 1] = r[0]*r[1]*(1 - costheta) + r[2]*sintheta
        mat[0, 2] = r[0]*r[2]*(1 - costheta) - r[1]*sintheta
        mat[1, 0] = r[1]*r[0]*(1 - costheta) - r[2]*sintheta
        mat[1, 1] = r[1]*r[1]*(1 - costheta) + costheta
        mat[1, 2] = r[1]*r[2]*(1 - costheta) + r[0]*sintheta
        mat[2, 0] = r[2]*r[0]*(1 - costheta) + r[1]*sintheta
        mat[2, 1] = r[2]*r[1]*(1 - costheta) - r[0]*sintheta
        mat[2, 2] = r[2]*r[2]*(1 - costheta) + costheta

        return np.matrix(mat)

    @staticmethod
    def mat_to_rtheta(mat):

        theta = np.arccos(0.5*(mat[0, 0] + mat[1, 1] + mat[2, 2] - 1))

        if theta < CONVERSION_EPS:
            theta = 0.
            r = [1, 0, 0]

        elif np.abs(np.pi - theta) < CONVERSION_EPS:
            theta = np.pi
            r = [np.sqrt(0.5*(mat[i, i] + 1)) for i in range(3)]

            m = np.argmax(r)

            sr = [np.sign(mat[i, m]) for i in range(3)]
            sr[m] = 1

            r = [r[i]*sr[i] for i in range(3)]

        else:
            r = [(mat[1, 2] - mat[2, 1])/(2.*np.sin(theta)),
                 (mat[2, 0] - mat[0, 2])/(2.*np.sin(theta)),
                 (mat[0, 1] - mat[1, 0])/(2.*np.sin(theta))]

        return [r, theta]

    @staticmethod
    def rtheta_to_quat(rtheta):

        r, theta = rtheta

        return [np.cos(0.5 * theta),
                np.sin(0.5 * theta) * r[0],
                np.sin(0.5 * theta) * r[1],
                np.sin(0.5 * theta) * r[2]]

    @staticmethod
    def quat_to_rtheta(quat):

        theta = 2.*np.arccos(quat[0])

        if theta > CONVERSION_EPS:
            r = [quat[i+1] / np.sin(0.5*theta) for i in range(3)]

        else:
            r = [1., 0., 0.]

        return [r, theta]

    @staticmethod
    def quat_to_mat(quat):

        mat = np.identity(3, dtype=np.float32)

        mat[0, 0] = quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2
        mat[0, 1] = 2.*(quat[1]*quat[2] + quat[0]*quat[3])
        mat[0, 2] = 2.*(quat[1]*quat[3] - quat[0]*quat[2])
        mat[1, 0] = 2.*(quat[1]*quat[2] - quat[0]*quat[3])
        mat[1, 1] = quat[0]**2 - quat[1]**2 + quat[2]**2 - quat[3]**2
        mat[1, 2] = 2.*(quat[2]*quat[3] + quat[0]*quat[1])
        mat[2, 0] = 2.*(quat[1]*quat[3] + quat[0]*quat[2])
        mat[2, 1] = 2.*(quat[2]*quat[3] + quat[0]*quat[1])
        mat[2, 2] = quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2

        return np.matrix(mat)

    @staticmethod
    def mat_to_quat(mat):

        q0 = 0.5*np.sqrt(mat[0, 0] + mat[1, 1] + mat[2, 2])

        if q0 > CONVERSION_EPS:

            q1 = (mat[1, 2] - mat[2, 1]) / (4*q0)
            q2 = (mat[2, 0] - mat[0, 2]) / (4 * q0)
            q3 = (mat[0, 1] - mat[1, 0]) / (4 * q0)

        else:
            q0 = 0.
            q1 = np.sqrt(0.5*(mat[0, 0] + 1))
            q2 = np.sqrt(0.5*(mat[1, 1] + 1))
            q3 = np.sqrt(0.5*(mat[2, 2] + 1))

            m = np.argmax([q1, q2, q3])

            sq = [np.sign(mat[i, m]) for i in range(3)]
            sq[m] = 1

            q1, q2, q3 = q1*sq[0], q2*sq[1], q3*sq[2]

        return [q0, q1, q2, q3]

    @staticmethod
    def quat_to_euler(quat):

        phi = 2.*np.arctan2(np.sqrt(quat[1]**2 + quat[2]**2), np.sqrt(quat[0]**2 + quat[3]**2))

        if phi < CONVERSION_EPS:
            phi = 0.
            phi1 = 2.*np.arctan2(quat[3], quat[0])
            phi2 = 0.

        elif np.abs(np.pi - phi) < CONVERSION_EPS:
            phi = np.pi
            phi1 = 2.*np.arctan2(quat[2], quat[0])
            phi2 = 0.

        else:
            phi1 = np.arctan2(quat[3], quat[0]) + np.arctan2(quat[2], quat[1])
            phi2 = np.arctan2(quat[3], quat[0]) - np.arctan2(quat[2], quat[1])

        return [phi1, phi, phi2]

    @staticmethod
    def euler_to_quat(euler):

        phi1, phi, phi2 = euler

        return [np.cos(0.5*phi)*np.cos(0.5*(phi1 + phi2)),
                np.sin(0.5*phi)*np.cos(0.5*(phi1 - phi2)),
                np.sin(0.5*phi)*np.sin(0.5*(phi1 - phi2)),
                np.cos(0.5*phi)*np.sin(0.5*(phi1 + phi2))]

    @staticmethod
    def rod_to_rtheta(rod):

        theta = 2. * np.arctan(np.linalg.norm(rod))

        if theta > CONVERSION_EPS:

            r = [rod[i]/np.tan(0.5*theta) for i in range(3)]

        else:

            r = [1, 0, 0]

        return [r, theta]

    @staticmethod
    def rtheta_to_rod(rtheta):

        r, theta = rtheta

        return [np.tan(0.5*theta)*r(i) for i in range(3)]

    @staticmethod
    def rod_to_mat(rod):
        return Orientation.rtheta_to_mat(Orientation.rod_to_rtheta(rod))

    @staticmethod
    def mat_to_rod(mat):
        return Orientation.rtheta_to_rod(Orientation.mat_to_rtheta(mat))

    def to_mat(self):
        return self.mat
