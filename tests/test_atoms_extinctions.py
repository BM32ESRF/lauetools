# -*- coding: utf-8 -*-
"""
extinctions computed from the atom positions of a crystal structure (extinction 'atoms' of a material):
structurefactor.structure_allowed(), materials.yaml fields 'atoms', 'spacegroup', 'symops', CIF import,
simulation and indexing
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pytest

import LaueTools.cifreader as CIF
import LaueTools.CrystalParameters as CP
import LaueTools.dict_LaueTools as DictLT
import LaueTools.generaltools as GT
import LaueTools.lauecore as LAUE
import LaueTools.structurefactor as SF

HKL = np.array([h for h in itertools.product(range(-6, 7), repeat=3) if any(h)])
SI = [("Si", 0.0, 0.0, 0.0, 1.0)]
CORUNDUM = [("Al", 0.0, 0.0, 0.35216, 1.0), ("O", 0.30668, 0.0, 0.25, 1.0)]
CALIB = [77.0, 1020.0, 1030.0, 0.3, -0.2]


def allowed_by_structure(label, atoms, spacegroup=None, symops=None):
    SF.register_structure(label, atoms, spacegroup, symops)
    return {tuple(h) for h in HKL[SF.structure_allowed(HKL, label)]}


def allowed_by_code(code):
    return {tuple(h) for h in CP.ApplyExtinctionrules(HKL, code)}


@pytest.mark.parametrize("label, atoms, spacegroup, code", [
    ("t_Si", SI, "227:1", "dia"),  # diamond: 222-type reflections absent (special position 8a)
    ("t_Si2", [("Si", 0.125, 0.125, 0.125, 1.0)], "227:2", "dia"),  # origin choice 2
    ("t_Cu", [("Cu", 0, 0, 0, 1)], "225", "fcc"),
    ("t_NaCl", [("Na", 0, 0, 0, 1), ("Cl", 0.5, 0.5, 0.5, 1)], "225", "fcc"),  # Na and Cl do not cancel
    ("t_Fe", [("Fe", 0, 0, 0, 1)], "229", "bcc"),
    ("t_corundum", CORUNDUM, "167:H", "167:H"),  # no absences beyond the space group ones
])
def test_structure_extinctions(label, atoms, spacegroup, code):
    assert allowed_by_structure(label, atoms, spacegroup) == allowed_by_code(code)


def test_structure_within_general_conditions():
    for label, atoms, spacegroup in (("t_quartz", [("Si", 0.4697, 0, 0, 1), ("O", 0.4135, 0.2669, 0.1191, 1)],
                                      "152"),
                                     ("t_ZrO2", [("Zr", 0.2758, 0.0411, 0.2082, 1), ("O", 0.0703, 0.3359, 0.3406, 1),
                                                 ("O", 0.4423, 0.7549, 0.4789, 1)], "14:b")):
        assert allowed_by_structure(label, atoms, spacegroup) <= allowed_by_code(spacegroup)


def test_unit_cell_and_rounded_coordinates():
    """equivalent positions merged, coordinates written with few decimals (0.3333) snapped to 1/3"""
    cell = SF.expand_structure(CORUNDUM, "167:H")
    assert (len(cell["Al"][0]), len(cell["O"][0])) == (12, 18)
    cell = SF.expand_structure([("Si", 0.1250, 0.1250, 0.1250, 1.0)], "227:2")
    assert len(cell["Si"][0]) == 8
    cell = SF.expand_structure([("C", 0.3333, 0.6667, 0.0, 1.0)], "191")  # graphite-like 2c site (1/3, 2/3, 0)
    assert len(cell["C"][0]) == 2


def test_occupancy_of_mixed_site():
    """two elements sharing the fcc sites: still fcc absences, whatever the occupancies"""
    atoms = [("Cu", 0, 0, 0, 0.7), ("Au", 0, 0, 0, 0.3)]
    assert allowed_by_structure("t_CuAu", atoms, "225") == allowed_by_code("fcc")


def test_materials_file_with_atoms(tmp_path):
    path = str(tmp_path / "materials.yaml")
    Path(path).write_text("")
    library = DictLT.Materials(yaml_file_path=path)
    lattice = [5.43088, 5.43088, 5.43088, 90, 90, 90]
    library.add_or_update_material("Si_atoms", lattice, "atoms", yaml_file_path=path, atoms=SI,
                                   spacegroup="227:1")
    text = Path(path).read_text()
    assert "extinction: atoms" in text and "- Si 0 0 0 1" in text and "spacegroup: '227:1'" in text

    library = DictLT.Materials(yaml_file_path=path)
    assert library.get_extinction("Si_atoms") == "atoms:Si_atoms"
    assert library["Si_atoms"][2] == "atoms:Si_atoms"
    assert library.get_structure("Si_atoms") == {"atoms": SI, "spacegroup": "227:1", "symops": None}
    assert CP.is_extinction_code("atoms:Si_atoms") and not CP.is_extinction_code("atoms:unknown_material")

    # update of the lattice only (Materials Editor): atom positions kept
    library.add_or_update_material("Si_atoms", [5.44] * 3 + [90] * 3, "atoms", yaml_file_path=path)
    assert library.get_structure("Si_atoms")["atoms"] == SI
    # material created from an other one (e.g. strained material of DetectorCalibration): its atom positions
    library.add_or_update_material("Si_strained", [5.45, 5.43, 5.43, 90, 90, 90], "atoms:Si_atoms",
                                   yaml_file_path=path)
    assert library.get_extinction("Si_strained") == "atoms:Si_strained"
    assert library.get_structure("Si_strained")["atoms"] == SI
    with pytest.raises(ValueError):
        library.add_or_update_material("NoAtoms", lattice, "atoms", yaml_file_path=path)


def test_atom_site_text():
    assert DictLT.parse_atom("O 0.30668 0 0.25 0.5") == ("O", 0.30668, 0.0, 0.25, 0.5)
    assert DictLT.parse_atom("C 1/3 2/3 0") == ("C", 1 / 3, 2 / 3, 0.0, 1.0)
    assert DictLT.parse_atom(DictLT.format_atom("Al", 0, 0, 0.35216)) == ("Al", 0.0, 0.0, 0.35216, 1.0)
    with pytest.raises(ValueError):
        DictLT.parse_atom("Si 0 0")


CIF_SI = """data_Si
_chemical_formula_sum Si
_cell_length_a 5.43088
_cell_length_b 5.43088
_cell_length_c 5.43088
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 227
_space_group_name_H-M_alt 'F d -3 m :2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.12500 0.12500 0.12500
"""


def test_structure_from_cif(tmp_path):
    path = tmp_path / "Si.cif"
    path.write_text(CIF_SI)
    structure = CIF.structure_from_cif(str(path))
    assert structure == {"atoms": [("Si", 0.125, 0.125, 0.125, 1.0)], "spacegroup": "227:2", "symops": None}
    # no atom positions
    path.write_text(CIF_SI.split("loop_")[0])
    with pytest.raises(ValueError):
        CIF.structure_from_cif(str(path))


def spots(key, dictmaterials, U=np.eye(3)):
    out = LAUE.SimulateLaue_full_np([None, None, U, key], 5, 22, CALIB, dictmaterials=dictmaterials)
    return {tuple(h) for h in np.asarray(out[2]).astype(int)}


def test_simulation_with_atoms(tmp_path):
    path = tmp_path / "Si.cif"
    path.write_text(CIF_SI)
    yaml_path = str(tmp_path / "materials.yaml")
    Path(yaml_path).write_text("")
    library = DictLT.Materials(yaml_file_path=yaml_path)
    label, lattice, _ = CIF.material_from_cif(str(path))
    library.add_or_update_material(label, lattice, "atoms", yaml_file_path=yaml_path,
                                   **CIF.structure_from_cif(str(path)))
    library.add_or_update_material("Si_dia", lattice, "dia", yaml_file_path=yaml_path)
    library.add_or_update_material("Si_sg", lattice, "227", yaml_file_path=yaml_path)
    with_atoms, dia, spacegroup = spots(label, library), spots("Si_dia", library), spots("Si_sg", library)
    assert with_atoms == dia and with_atoms < spacegroup


def test_indexing_with_atoms_material(tmp_path):
    """material with atom positions (session material of dict_Materials) used by the indexing of a simulated
    silicon pattern"""
    import LaueTools.indexing_batch as IB
    import LaueTools.IOLaueTools as IOLT
    import LaueTools.LaueGeometry as LG

    label = "Si_atoms_test_indexing"
    DictLT.dict_Materials[label] = {"lattice": [5.43088] * 3 + [90] * 3, "extinction": "atoms",
                                    "atoms": [DictLT.format_atom(*SI[0])], "spacegroup": "227:1"}
    try:
        pixelsize = 165.0 / 2048
        U = GT.fromelemangles_toMatrix([20.0, 35.0, -10.0])
        _, _, hkl, X, Y, _ = LAUE.SimulateLaue_full_np([None, None, U, label], 5, 22, CALIB,
                                                       pixelsize=pixelsize)[:6]
        X, Y = np.asarray(X), np.asarray(Y)
        keep = (X > 0) & (X < 2048) & (Y > 0) & (Y < 2048)
        X, Y = X[keep], Y[keep]
        tth, chi = LG.calc_uflab(X, Y, CALIB, pixelsize=pixelsize)
        intensity = 1e5 / (np.asarray(hkl)[keep] ** 2).sum(axis=1)
        order = np.argsort(-intensity)
        corfolder = tmp_path / "cor"
        corfolder.mkdir()
        IOLT.writefile_cor("img_0000", tth[order], chi[order], X[order], Y[order], intensity[order],
                           param={"dd": CALIB[0], "xcen": CALIB[1], "ycen": CALIB[2], "xbet": CALIB[3],
                                  "xgam": CALIB[4], "pixelsize": pixelsize, "framedim": (2048, 2048),
                                  "CCDLabel": "MARCCD165", "kf_direction": "Z>0"},
                           dirname_output=str(corfolder))
        template = Path(IB.__file__).parent / "notebooks" / "indexation" / "config_template.yaml"
        cfgfile = tmp_path / "cfg.yaml"
        cfgfile.write_text(template.read_text())
        cfg = IB.load_config(cfgfile, overrides={
            "scan": {"cor_folder": str(corfolder), "prefix": "img_", "nbdigits": 4, "mapdims": [1, 1],
                     "CCDLabel": "MARCCD165"},
            "output": {"fit_folder": str(tmp_path / "fit")},
            "material": {"key_material": label},
            "run": {"nb_cpus": 1}})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = IB.index_refine(corfolder / "img_0000.cor", IB.params_from_config(cfg))
        grains = res[1]
        assert grains and grains[0][1] is not None, "not indexed: %s" % (res[2],)
    finally:
        del DictLT.dict_Materials[label]
