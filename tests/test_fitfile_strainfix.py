# -*- coding: utf-8 -*-
"""
test the rewrite of the strain of .fit files written before Oct. 2026 (fitfile_strainfix, TEMPORARY-STRAINFIX)

A hexagonal (Zr) .fit file is written with the OLD sample frame strain (UB eps UB-1, no a // x -> a* // x
rotation); the rewritten file must contain the strain computed by the current code and the OLD one in the
[TEMPORARY] block, the spots and other blocks unchanged
"""
import numpy as np

import LaueTools.CrystalParameters as CP
import LaueTools.IOLaueTools as IOLT
import LaueTools.generaltools as GT
import LaueTools.fitfile_strainfix as FSF
from LaueTools.dict_LaueTools import dict_Materials
from LaueTools.fitfilereader import parsed_fitfile


def write_old_fitfile(path, key_material="Zr"):
    lat = dict_Materials[key_material][1]
    B0 = CP.calc_B_RR(lat)
    q, r = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    q *= np.sign(np.linalg.det(q))
    eps = np.array([[2, 0.5, -1], [0.5, -1.5, 0.8], [-1, 0.8, 0.3]]) * 1e-3
    UB = np.dot(q, np.eye(3) + eps)
    dev, dev_sample, newlat = CP.evaluate_strain_fromUBmat(UB, key_material)
    P = GT.matRot([0, 1, 0], -40.0)
    dev_sample_old = np.dot(P.T, np.dot(UB, np.dot(dev, np.dot(np.linalg.inv(UB), P))))
    data = np.array([[0, 1000, 1, 0, 0, 50, 10, 1000, 1000, 10, 0, 0.1]])
    cols = "#spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev"
    dict_matrices = {"Element": key_material, "grainIndex": 0, "UBmat": UB, "B0": B0, "UBB0": np.dot(UB, B0),
                     "euler_angles": np.array([1.0, 2.0, 3.0]), "devstrain_crystal": dev,
                     "devstrain_sample": dev_sample_old, "CCDLabel": "sCMOS",
                     "detectorparameters": [77.0, 1000.0, 1000.0, 0.4, 0.2], "pixelsize": 0.0734,
                     "framedim": (2018, 2016)}
    IOLT.writefitfile(str(path), data, 1, dict_matrices, 0.1, "x.cor", cols, modulecaller="indexingSpotsSet.py")
    return UB, dev, dev_sample, dev_sample_old


def test_rewrite_hexagonal(tmp_path):
    oldfit = tmp_path / "Zr_g0.fit"
    UB, dev, dev_sample, dev_sample_old = write_old_fitfile(oldfit)
    res = FSF.rewrite_fitfile_strain(str(oldfit))
    assert res["status"] == "rewritten"
    assert res["outputfilename"] == str(tmp_path / "strainfix" / "Zr_g0.fit")
    assert res["maxdiff"] > 1.0  # 10-3 unit: OLD strain wrong for hexagonal cells

    fit = parsed_fitfile(res["outputfilename"])
    np.testing.assert_allclose(fit.UB, UB, atol=1e-8)
    np.testing.assert_allclose(fit.dev_sample, dev_sample, atol=1e-5)
    np.testing.assert_allclose(fit.dev_sample_old, dev_sample_old, atol=1e-5)
    assert fit.reference_material == "Zr"
    assert abs(fit.stress_sample[2, 2]) < 0.1  # MPa, stress_zz = 0
    full = CP.fullstrain_from_deviatoricstrain(dev, UB, "Zr")
    np.testing.assert_allclose(fit.fullstrain_sample, full["fullstrain_sample"], atol=1e-5)

    # spots table and blocks other than strain unchanged
    old_lines = open(oldfit).read().splitlines()
    new_lines = open(res["outputfilename"]).read().splitlines()
    assert [line for line in old_lines if not line.startswith("#")] == \
           [line for line in new_lines if not line.startswith("#")]
    for label in ("#UB matrix", "#B0 matrix", "#DetectorParameters", "#Frame dimensions"):
        k_old = [k for k, line in enumerate(old_lines) if line.startswith(label)][0]
        k_new = [k for k, line in enumerate(new_lines) if line.startswith(label)][0]
        assert old_lines[k_old:k_old + 2] == new_lines[k_new:k_new + 2]

    # already corrected file not rewritten again
    assert FSF.rewrite_fitfile_strain(res["outputfilename"])["status"] == "already corrected"


def test_rewrite_with_reference(tmp_path):
    # file indexed with Zr, strain rewritten w.r.t. Zr_ref: as if indexed with Zr_ref
    oldfit = tmp_path / "Zr_g0.fit"
    UB, _, _, dev_sample_old = write_old_fitfile(oldfit)
    B0 = CP.calc_B_RR(dict_Materials["Zr"][1])
    UBref = np.dot(np.dot(UB, B0), np.linalg.inv(CP.calc_B_RR(dict_Materials["Zr_ref"][1])))
    dev, dev_sample, newlat = CP.evaluate_strain_fromUBmat(UBref, "Zr_ref")

    res = FSF.rewrite_fitfile_strain(str(oldfit), reference={"Zr": "Zr_ref"})
    fit = parsed_fitfile(res["outputfilename"])
    assert fit.reference_material == "Zr_ref"
    np.testing.assert_allclose(fit.UB, UB, atol=1e-8)  # UB and B0 blocks of the file kept
    np.testing.assert_allclose(fit.deviatoric, dev, atol=1e-5)
    np.testing.assert_allclose(fit.dev_sample, dev_sample, atol=1e-5)
    np.testing.assert_allclose(fit.dev_sample_old, dev_sample_old, atol=1e-5)
    full = CP.fullstrain_from_deviatoricstrain(dev, UBref, "Zr_ref", Cvoigt=CP.get_stiffness_matrix("Zr"))
    np.testing.assert_allclose(fit.fullstrain_sample, full["fullstrain_sample"], atol=1e-5)
    assert res["stiffness_material"] == "Zr"


def test_stiffness_key():
    assert FSF.stiffness_key("Zr") == "Zr"
    assert FSF.stiffness_key("Zr_ech15_labXRD") == "Zr"
    assert FSF.stiffness_key("Zr_ref", stiffness_material={"Zr_ref": "Ti"}) == "Ti"
    assert FSF.stiffness_key("Unobtainium") is None
