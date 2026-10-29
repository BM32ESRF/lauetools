# -*- coding: utf-8 -*-
"""
DAXM element tables (Daxm/material/dict_datamat.py): densities, fluorescence yields, K edges.

Typos in element keys ('Tb' for 'Th', 'Sr' for 'Sb') silently overwrote the entries of Tb and Sr
(a later duplicate key wins in a dict literal) and left Th and Sb out:
Sr got the K edge of Sb (30.49 keV instead of 16.10 keV), Tb the density of Th.
"""
import ast
from pathlib import Path

import pytest

import LaueTools.Daxm.material.dict_datamat as dm

TABLES = ('dict_mat', 'dict_density', 'dict_fluo_yield', 'dict_fluo_K')


@pytest.mark.parametrize('name', TABLES)
def test_no_duplicate_keys(name):
    tree = ast.parse(Path(dm.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            keys = [k.value for k in node.value.keys]
            assert sorted({k for k in keys if keys.count(k) > 1}) == []
            return
    raise AssertionError(f'{name} not found')


def test_values_of_previously_overwritten_elements():
    # density g/cm3
    assert dm.dict_density['Tb'] == 8.23
    assert dm.dict_density['Th'] == 11.72
    # fluorescence yield
    assert dm.dict_fluo_yield['Sr'] == 0.690
    assert dm.dict_fluo_yield['Sb'] == 0.870
    assert dm.dict_fluo_yield['Tb'] == 0.938
    assert 'Th' in dm.dict_fluo_yield
    # K edge and K alpha energies (keV)
    assert dm.dict_fluo_K['Sr'][1:] == (16.1046, 14.165)
    assert dm.dict_fluo_K['Sb'][1:] == (30.4912, 26.3591)
    assert 'Th' in dm.dict_fluo_K
