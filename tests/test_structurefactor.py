# -*- coding: utf-8 -*-
"""
structurefactor: symmetry operations and general reflection conditions of space groups computed from them
"""
import itertools

import numpy as np
import pytest

import LaueTools.structurefactor as SF
import LaueTools.wyckpos_lauetools as WYCK

HKL = np.array([h for h in itertools.product(range(-4, 5), repeat=3) if any(h)])


def test_parse_symop():
    R, t = SF.parse_symop("-x+1/2, y, -z")
    np.testing.assert_array_equal(R, np.diag([-1, 1, -1]))
    np.testing.assert_allclose(t, [0.5, 0, 0])
    R, t = SF.parse_symop("(x-y, -y, -z+1/2)")
    np.testing.assert_array_equal(R, [[1, -1, 0], [0, -1, 0], [0, 0, -1]])
    np.testing.assert_allclose(t, [0, 0, 0.5])
    R, t = SF.parse_symop("1/2+x,1/2-y,2z")  # CIF style, implicit multiplication
    np.testing.assert_array_equal(R, np.diag([1, -1, 2]))
    np.testing.assert_allclose(t, [0.5, 0.5, 0])


@pytest.mark.parametrize("op", ["x, y", "__import__('os')", "9**9**9, y, z", "a, b, c"])
def test_parse_symop_rejects(op):
    with pytest.raises(ValueError):
        SF.parse_symop(op)


def table_settings():
    """settings of wyckpos_lauetools with tabulated general reflection conditions"""
    for key, positions in WYCK.wp.items():
        general = sorted(positions, key=lambda label: int(label[:-1]))[-1]
        if positions[general][2] not in (None, "n/a") or ":" not in key:
            yield key


def test_general_conditions_agree_with_tables():
    """for every tabulated setting: the computation never allows a reflection forbidden by the table, and the
    reflections it forbids in addition are symmetry equivalents of reflections forbidden by the table"""
    for key in table_settings():
        table = {tuple(h) for h in WYCK.testhklcond_generalrules_array(key, HKL)}
        computed = {tuple(h) for h in HKL[SF.general_allowed(HKL, key)]}
        assert computed <= table, key
        forbidden_table = {tuple(h) for h in HKL} - table
        for h in table - computed:
            assert any(tuple(np.rint(np.array(h) @ R).astype(int)) in forbidden_table
                       for R, _ in SF.spacegroup_symops(key)), (key, h)


@pytest.mark.parametrize("number", [146, 148, 155, 160, 161, 166, 167])
def test_rhombohedral_and_hexagonal_axes(number):
    """same crystal indexed in rhombohedral axes and in (obverse) hexagonal axes: same absences"""
    to_hexagonal = np.array([[1, -1, 0], [0, 1, -1], [1, 1, 1]])
    np.testing.assert_array_equal(SF.general_allowed(HKL, "%d:R" % number),
                                  SF.general_allowed(HKL @ to_hexagonal.T, "%d:H" % number))


@pytest.mark.parametrize("number", [7, 9, 13, 14, 15])
def test_monoclinic_unique_axes(number):
    """unique axis b (a, b, c) and unique axis c (c, a, b relabelled): same absences"""
    np.testing.assert_array_equal(SF.general_allowed(HKL, "%d:b" % number),
                                  SF.general_allowed(HKL[:, (2, 0, 1)], "%d:c" % number))


def test_origin_choices():
    for number in (48, 141, 227):
        np.testing.assert_array_equal(SF.general_allowed(HKL, "%d:1" % number),
                                      SF.general_allowed(HKL, "%d:2" % number))
