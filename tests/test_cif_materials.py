# -*- coding: utf-8 -*-
"""
materials from CIF files (cifreader) and space group numbers as extinction codes
(CrystalParameters.ApplyExtinctionrules, lauecore, materials.yaml)
"""
import itertools

import numpy as np
import pytest

import LaueTools.cifreader as CIF
import LaueTools.CrystalParameters as CP
import LaueTools.dict_LaueTools as DictLT
import LaueTools.lauecore as LAUE

HKL = np.array([h for h in itertools.product(range(-6, 7), repeat=3) if any(h)])

SI_CIF = """# silicon, CIF syntax: comments, quoted values, text field, loops, uncertainties
data_global
_journal_name_full 'Not a structure block'
data_Si
_chemical_name_common 'silicon'
_chemical_formula_sum Si
_publ_section_title
;
 A multi-line text field
 with 'quotes' and _tags_ inside
;
_cell_length_a 5.43088(5)
_cell_length_b 5.43088(5)
_cell_length_c 5.43088(5)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 227
_space_group_name_H-M_alt 'F d -3 m :1'
loop_
_space_group_symop_operation_xyz
x,y,z
'-x,-y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0.00000 0.00000 0.00000 1.0
"""

CORUNDUM_CIF = """data_corundum
_chemical_formula_sum 'Al2 O3'
_cell.length_a 4.7605
_cell.length_b 4.7605
_cell.length_c 12.9956
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 120
_symmetry_Int_Tables_number 167
_symmetry_space_group_name_H-M 'R -3 c :H'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al3+ 0 0 0.35216(2)
O1 O2- 0.30668(9) 0 0.25
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_read_cif_syntax(tmp_path):
    structure = CIF.read_cif(write(tmp_path, "Si.cif", SI_CIF))
    assert structure["lattice"] == [5.43088, 5.43088, 5.43088, 90.0, 90.0, 90.0]
    assert (structure["spacegroup"], structure["setting"]) == (227, ":1")
    assert structure["symops"] == ["x,y,z", "-x,-y,z"]
    assert structure["atoms"] == [{"label": "Si1", "element": "Si", "x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0}]
    assert CIF.material_from_cif(write(tmp_path, "Si.cif", SI_CIF)) == ("Si", [5.43088] * 3 + [90.0] * 3, "227:1")


def test_corundum_hexagonal_setting(tmp_path):
    structure = CIF.read_cif(write(tmp_path, "corundum.cif", CORUNDUM_CIF))
    assert [a["element"] for a in structure["atoms"]] == ["Al", "O"]  # charges removed
    assert structure["atoms"][0]["z"] == 0.35216
    assert CIF.material_from_cif(write(tmp_path, "corundum.cif", CORUNDUM_CIF)) == \
        ("Al2O3", [4.7605, 4.7605, 12.9956, 90.0, 90.0, 120.0], "167:H")


@pytest.mark.parametrize("cell, symbol, expected", [
    ("4.7605 4.7605 12.9956 90 90 120", None, "167:H"),  # hexagonal axes, no setting in the CIF
    ("5.128 5.128 5.128 55.28 55.28 55.28", None, "167:R"),  # rhombohedral axes
])
def test_rhombohedral_setting_from_cell(tmp_path, cell, symbol, expected):
    a, b, c, alpha, beta, gamma = cell.split()
    text = ("data_x\n_chemical_formula_sum X\n_cell_length_a %s\n_cell_length_b %s\n_cell_length_c %s\n"
            "_cell_angle_alpha %s\n_cell_angle_beta %s\n_cell_angle_gamma %s\n_space_group_IT_number 167\n"
            % (a, b, c, alpha, beta, gamma))
    assert CIF.material_from_cif(write(tmp_path, "x.cif", text))[2] == expected


@pytest.mark.parametrize("symbol, expected", [("P 1 21/c 1", ":b"), ("P 1 1 21/b", ":c"), ("P 21/c", ""),
                                              ("F d -3 m :2", ":2"), ("F m -3 m", "")])
def test_spacegroup_setting(symbol, expected):
    number = {"P": 14, "F d": 227, "F m": 225}[next(k for k in ("F d", "F m", "P") if symbol.startswith(k))]
    assert CIF.spacegroup_setting(number, symbol) == expected


def test_no_spacegroup(tmp_path):
    text = SI_CIF.replace("_space_group_IT_number 227\n", "").replace("_space_group_name_H-M_alt 'F d -3 m :1'\n", "")
    assert CIF.material_from_cif(write(tmp_path, "Si.cif", text))[2] == "no"


def test_missing_cell(tmp_path):
    with pytest.raises(ValueError):
        CIF.read_cif(write(tmp_path, "bad.cif", "data_x\n_chemical_formula_sum X\n"))


def test_extinction_codes():
    for code in ("fcc", "no", "225", 225, "227:1", "227:2", "167:H", "167:R", "14:c"):
        assert CP.is_extinction_code(code), code
    for code in ("231", "0", "227:H", "167:1", "abc", True, None):
        assert not CP.is_extinction_code(code), code


def test_spacegroup_rules():
    allowed = lambda code: {tuple(h) for h in CP.ApplyExtinctionrules(HKL, code)}
    assert allowed("225") == allowed(225) == allowed("fcc")
    assert allowed("229") == allowed("bcc")
    assert allowed("227:1") == allowed("227:2") == allowed("227")  # origin choice: same conditions
    # setting kept: hexagonal axes (-h+k+l=3n) and rhombohedral axes give different conditions
    assert allowed("167:H") != allowed("167:R")
    assert (1, 0, 0) not in allowed("167:H") and (1, 1, 0) in allowed("167:R")
    # tetragonal P4(2)2(1)2: 'h00: h=2n' applies to 0k0 too (equivalent by the 4-fold axis)
    assert (0, 3, 0) not in allowed("94") and (0, 4, 0) in allowed("94")


def test_unquoted_spacegroup_in_yaml(tmp_path):
    path = tmp_path / "materials.yaml"
    lattice = "  lattice: [5.43088, 5.43088, 5.43088, 90, 90, 90]\n"
    path.write_text("Si_sg:\n  label: Si_sg\n" + lattice + "  extinction: 227\n"
                    "Si_sg1:\n  label: Si_sg1\n" + lattice + "  extinction: 227:1\n")  # YAML 1.1: 13621
    materials = DictLT.Materials(yaml_file_path=str(path))
    assert materials.get_extinction("Si_sg") == "227"
    assert materials.get_extinction("Si_sg1") == "227:1"


def test_simulation_with_cif_material(tmp_path):
    """material from a CIF file, saved in a materials file and used for a Laue pattern simulation"""
    path = tmp_path / "materials.yaml"
    path.write_text("")
    materials = DictLT.Materials(yaml_file_path=str(path))
    label, lattice, extinction = CIF.material_from_cif(write(tmp_path, "Si.cif", SI_CIF))
    materials.add_or_update_material(label, lattice, extinction, yaml_file_path=str(path))
    materials.add_or_update_material("Si_fcc", lattice, "fcc", yaml_file_path=str(path))
    materials = DictLT.Materials(yaml_file_path=str(path))
    assert materials.get_extinction("Si") == "227:1"

    def spots(key):
        out = LAUE.SimulateLaue_full_np([None, None, np.eye(3), key], 5, 22, [77.0, 1024.0, 1024.0, 0.0, 0.0],
                                        dictmaterials=materials)
        return {tuple(h) for h in np.asarray(out[2]).astype(int)}
    si, si_fcc = spots("Si"), spots("Si_fcc")
    assert si and si < si_fcc  # space group 227 removes reflections allowed by the F lattice only
