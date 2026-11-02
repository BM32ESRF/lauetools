# -*- coding: utf-8 -*-
"""
refinementresults: strain, refined lattice parameters and orientation matrices from a refined UB matrix, and their
text report (shared by the GUIs DetectorCalibration and PlotRefineGUI)
"""
import numpy as np
import pytest

import LaueTools.CrystalParameters as CP
import LaueTools.dict_LaueTools as DictLT
import LaueTools.generaltools as GT
import LaueTools.refinementresults as RR

ROTATION = GT.fromelemangles_toMatrix([12.0, -35.0, 70.0])
# keys read by the results window (GUI/PlotRefineGUI.FitResultsBoard)
BOARD_KEYS = ["NbspotsResidues", "devstrain_crystal", "devstrain_sample", "fullstrain_sample", "Initial lattice",
              "Refined lattice", "UBmatrix", "B0matrix", "HKLxyz_names", "HKLxyz"]


def ub_from_lattice(key_material, latticeparams, rotation=ROTATION):
    """UB matrix of a crystal with these lattice parameters, indexed with the reference ones of key_material"""
    B0 = CP.calc_B_RR(DictLT.dict_Materials[key_material][1])
    return np.dot(rotation, np.dot(CP.calc_B_RR(latticeparams), np.linalg.inv(B0)))


def check_U_B(results):
    """UB = U Bmat_tri, U a rotation, Bmat_tri upper triangular"""
    U, B = results["Umat"], results["Bmat_tri"]
    np.testing.assert_allclose(np.dot(U, B), results["UBmat"], atol=1e-12)
    np.testing.assert_allclose(np.dot(U.T, U), np.eye(3), atol=1e-12)
    assert np.linalg.det(U) == pytest.approx(1.0)
    np.testing.assert_allclose(np.tril(B, -1), 0.0, atol=1e-12)


def test_unstrained_crystal():
    results = RR.strain_results(ROTATION, "Cu")
    np.testing.assert_allclose(results["devstrain"], 0.0, atol=1e-12)
    np.testing.assert_allclose(results["new_latticeparameters"], DictLT.dict_Materials["Cu"][1], atol=1e-9)
    assert RR.max_strain(results) == 0.0
    check_U_B(results)
    np.testing.assert_allclose(results["Bmat_tri"], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(results["Umat"], ROTATION, atol=1e-12)


def test_tetragonal_distortion_of_cubic_cell():
    """c elongated by 1e-3 (a kept): deviatoric strain diag(-1/3, -1/3, 2/3) 1e-3 in the crystal frame,
    whatever the orientation"""
    a = DictLT.dict_Materials["Cu"][1][0]
    results = RR.strain_results(ub_from_lattice("Cu", [a, a, a * 1.001, 90, 90, 90]), "Cu", constantlength="a")
    np.testing.assert_allclose(results["devstrain"] * 1000, np.diag([-1 / 3, -1 / 3, 2 / 3]), atol=1e-6)
    np.testing.assert_allclose(results["new_latticeparameters"], [a, a, a * 1.001, 90, 90, 90], atol=1e-9)
    assert RR.max_strain(results) == pytest.approx(0.667, abs=1e-3)
    assert results["equivalentstrain"] > 0
    check_U_B(results)


@pytest.mark.parametrize("key_material", ["Ti", "ZrO2", "Al2O3"])  # hexagonal, monoclinic, trigonal
def test_non_orthogonal_cells(key_material):
    params = np.array(DictLT.dict_Materials[key_material][1], dtype=float)
    strained = params * np.array([1.0, 1.0005, 0.999, 1.0, 1.0, 1.0])
    results = RR.strain_results(ub_from_lattice(key_material, strained), key_material, constantlength="a")
    np.testing.assert_allclose(results["new_latticeparameters"], strained, rtol=1e-9)
    assert np.trace(results["devstrain"]) == pytest.approx(0.0, abs=1e-12)  # deviatoric
    check_U_B(results)
    # UB_new = scale x rotation, referred to the refined lattice parameters
    assert results["rotdev"] < 1e-9


def test_results_texts():
    a = DictLT.dict_Materials["Cu"][1][0]
    results = RR.strain_results(ub_from_lattice("Cu", [a, a, a * 1.001, 90, 90, 90]), "Cu")
    residues = [0.1, 0.3]
    texts = RR.results_texts(results, residues, 2, filename="img_0001.cor", folder="/data/run1")
    assert all(key in texts for key in BOARD_KEYS)
    assert "Ts parameters" not in texts
    assert "img_0001.cor" in texts["NbspotsResidues"] and "Folder: /data/run1" in texts["NbspotsResidues"]
    assert "Mean Pixel Deviation: 0.200" in texts["NbspotsResidues"]
    assert "Number of refined Laue spots: 2" in texts["NbspotsResidues"]
    assert "depth" not in texts["NbspotsResidues"]
    rows = texts["devstrain_crystal"].splitlines()[1:4]
    np.testing.assert_allclose([[float(v) for v in row.split()] for row in rows],
                               np.diag([-0.333, -0.333, 0.667]), atol=1e-9)
    assert "  c\t\t%.6f" % (a * 1.001) in texts["Refined lattice"]
    assert "UB_new" in texts["UBmatrix"] and "B0_new" in texts["B0matrix"]

    Ts = np.array([[1.0, 0.1, 0.2], [0.0, 1.1, 0.3], [0.0, 0.0, 0.9]])
    texts = RR.results_texts(results, residues, 2, depth=12.5, Tsresults=(None, None, Ts))
    assert "Normal sample depth (microns): 12.5" in texts["NbspotsResidues"]
    assert "Ts00  Ts01  Ts02 Ts11 Ts12 Ts22 = [1.000000, 0.100000, 0.200000, 1.100000, 0.300000, 0.900000]" \
        in texts["Ts parameters"]
