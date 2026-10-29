# -*- coding: utf-8 -*-
"""
test lattice parameters conversions of CrystalParameters with list, integer and float inputs
(dlat_to_rlat([5, 5, 5, 90, 90, 90]) crashed with integer input, vol_cell crashed with any list)
"""
import numpy as np
import pytest

import LaueTools.CrystalParameters as CP

CUBIC_INPUTS = [[5, 5, 5, 90, 90, 90],
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                np.array([5, 5, 5, 90, 90, 90]),
                np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])]


@pytest.mark.parametrize("dlat", CUBIC_INPUTS)
def test_dlat_to_rlat_cubic(dlat):
    rlat = CP.dlat_to_rlat(dlat)
    np.testing.assert_allclose(rlat, [0.2, 0.2, 0.2, 90, 90, 90])


@pytest.mark.parametrize("dlat", CUBIC_INPUTS)
def test_vol_cell_cubic(dlat):
    assert CP.vol_cell(dlat) == pytest.approx(125.0)


@pytest.mark.parametrize("dlat", CUBIC_INPUTS)
def test_calc_epsp_cubic_unstrained(dlat):
    np.testing.assert_allclose(CP.calc_epsp_cubic(dlat), np.zeros(6), atol=1e-15)


def test_input_not_modified():
    dlat = np.array([3.0, 4.0, 5.0, 80.0, 95.0, 100.0])
    ref = dlat.copy()
    CP.dlat_to_rlat(dlat)
    CP.vol_cell(dlat)
    CP.calc_epsp_cubic(dlat)
    np.testing.assert_array_equal(dlat, ref)


def test_dlat_rlat_round_trip_triclinic():
    dlat = [3.0, 4.0, 5.0, 80.0, 95.0, 100.0]
    np.testing.assert_allclose(CP.dlat_to_rlat(CP.dlat_to_rlat(dlat)), dlat)
    assert CP.vol_cell(dlat) * CP.vol_cell(CP.dlat_to_rlat(dlat)) == pytest.approx(1.0)
