# -*- coding: utf-8 -*-
"""
test full strain computation from deviatoric strain with stress_zz = 0 (sample frame) assumption

A strained UB matrix is built from a known full strain satisfying stress_zz = 0,
the deviatoric strain is extracted as in indexingSpotsSet, then the full strain must be recovered
"""
import numpy as np

import LaueTools.CrystalParameters as CP
import LaueTools.generaltools as GT
from LaueTools.dict_LaueTools import dict_Materials


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def check_material(key_material, rng, sampletilt=40.0):
    lat = dict_Materials[key_material][1]
    B0 = CP.calc_B_RR(lat)
    Adirect = CP.calc_B_RR(lat, directspace=0)
    # direct crystal frame (x // a) to B0 frame (x // a*)
    Q = np.dot(np.linalg.inv(B0).T, np.linalg.inv(Adirect))
    C = CP.get_stiffness_matrix(key_material)

    R = random_rotation(rng)
    M = np.dot(GT.matRot([0, 1, 0], -sampletilt).T, np.dot(R, Q))
    n = M[2]
    w = np.array([n[0] ** 2, n[1] ** 2, n[2] ** 2, 2 * n[1] * n[2], 2 * n[0] * n[2], 2 * n[0] * n[1]])

    # random strain (direct crystal frame) whose trace is set such as stress_zz(sample) = 0
    eps = rng.normal(size=(3, 3)) * 1e-3
    eps = (eps + eps.T) / 2.0
    voigt = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
    trace_shift = -np.dot(w, np.dot(C, voigt)) / np.dot(w, np.dot(C, [1, 1, 1, 0, 0, 0]))
    eps_true = eps + trace_shift * np.eye(3)

    # strained UB: UB B0 = R (Id + eps)^-T B0 (eps expressed in B0 frame)
    eps_B0frame = np.dot(Q, np.dot(eps_true, Q.T))
    UB = np.dot(R, np.linalg.inv(np.eye(3) + eps_B0frame).T)

    devstrain, _, _ = CP.evaluate_strain_fromUBmat(UB, key_material, dictmaterials=dict_Materials)
    res = CP.fullstrain_from_deviatoricstrain(devstrain, UB, key_material, sampletilt=sampletilt)

    # first order theory: residual error is of second order (~1e-6)
    assert np.abs(res["fullstrain_crystal"] - eps_true).max() < 1e-5
    assert np.abs(res["fullstrain_sample"] - np.dot(M, np.dot(eps_true, M.T))).max() < 1e-5
    assert abs(res["stress_sample"][2, 2]) < 1e-9


def test_fullstrain_stresszz0():
    rng = np.random.default_rng(0)
    for key_material in ("Si", "Cu", "Ge", "Ti", "GaN"):
        for _ in range(5):
            check_material(key_material, rng)


def test_no_stiffness_data():
    assert CP.fullstrain_from_deviatoricstrain(np.zeros((3, 3)), np.eye(3), "CdTe") is None


if __name__ == "__main__":
    test_fullstrain_stresszz0()
    test_no_stiffness_data()
    print("full strain tests passed")
