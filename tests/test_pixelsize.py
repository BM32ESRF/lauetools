# -*- coding: utf-8 -*-
"""
Pixel size used for indexing and refinement of a .cor file.

The 2theta, chi of the spots in a .cor file are computed with the pixel size written in that file,
so indexing and refinement must use the same value, even when the detector table
(dict_CCD[CCDLabel][1]) gives a different one.
Case tested: the LaueTools default pixel size 165/2048 = 0.08057 mm with CCDLabel MARCCD165
(table value 0.079142 mm). Using the table value gave about 1.3% fake strain on unstrained MgO.
"""
import io
import warnings
from pathlib import Path

import numpy as np
import pytest

import LaueTools.CrystalParameters as CP
import LaueTools.IOLaueTools as IOLT
import LaueTools.LaueGeometry as LG
import LaueTools.lauecore as LC
from LaueTools.dict_LaueTools import dict_CCD, dict_Materials

PIXELSIZE = 165.0 / 2048  # LaueTools default, not the MARCCD165 table value
CALIB = [77.0, 1020.0, 1030.0, 0.3, -0.2]


def test_file_pixelsize_wins_over_table():
    table = dict_CCD['MARCCD165'][1]
    assert abs(table - PIXELSIZE) > 1e-3  # the case is meaningful
    with pytest.warns(UserWarning, match='differs'):
        ps = IOLT.get_pixelsize({'pixelsize': PIXELSIZE, 'CCDLabel': 'MARCCD165'})
    assert ps == PIXELSIZE


def test_table_pixelsize_when_file_has_none():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert IOLT.get_pixelsize({'CCDLabel': 'sCMOS'}) == dict_CCD['sCMOS'][1]
        # same value in file and table: no warning
        assert IOLT.get_pixelsize({'pixelsize': dict_CCD['sCMOS'][1], 'CCDLabel': 'sCMOS'}) == dict_CCD['sCMOS'][1]
    assert IOLT.get_pixelsize({'CCDLabel': None}) is None


def test_unknown_pixelsize_does_not_crash_label_guess():
    header = io.StringIO("# dd : 77.0\n# pixelsize : 1.2345\n")
    calib = IOLT.readCalibParametersInFile(header)
    assert calib['CCDLabel'] is None
    assert calib['pixelsize'] == 1.2345


def _write_simulated_cor(folder: Path, rng) -> Path:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    U = q * np.sign(np.diag(r))
    if np.linalg.det(U) < 0:
        U[:, 0] *= -1
    _, _, hkl, X, Y, _ = LC.SimulateLaue_full_np([None, None, U, 'MgO'], 5, 22, CALIB,
                                                pixelsize=PIXELSIZE)[:6]
    X, Y = np.asarray(X), np.asarray(Y)
    keep = (X > 0) & (X < 2048) & (Y > 0) & (Y < 2048)
    X, Y = X[keep], Y[keep]
    h2 = (np.asarray(hkl)[keep] ** 2).sum(axis=1)
    tth, chi = LG.calc_uflab(X, Y, CALIB, pixelsize=PIXELSIZE)
    intensity = 1e5 / h2
    order = np.argsort(-intensity)
    IOLT.writefile_cor('img_0000', tth[order], chi[order], X[order], Y[order], intensity[order],
                       param={'dd': CALIB[0], 'xcen': CALIB[1], 'ycen': CALIB[2],
                              'xbet': CALIB[3], 'xgam': CALIB[4], 'pixelsize': PIXELSIZE,
                              'framedim': (2048, 2048), 'CCDLabel': 'MARCCD165',
                              'kf_direction': 'Z>0'},
                       dirname_output=str(folder))
    return folder / 'img_0000.cor'


def test_indexing_uses_file_pixelsize(tmp_path):
    import LaueTools.indexing_batch as IB

    corfolder = tmp_path / 'cor'
    corfolder.mkdir()
    corfile = _write_simulated_cor(corfolder, np.random.default_rng(3))

    template = Path(IB.__file__).parent / 'notebooks' / 'indexation' / 'config_template.yaml'
    cfgfile = tmp_path / 'cfg.yaml'
    cfgfile.write_text(template.read_text())
    cfg = IB.load_config(cfgfile, overrides={
        'scan': {'cor_folder': str(corfolder), 'prefix': 'img_', 'nbdigits': 4,
                 'mapdims': [1, 1], 'CCDLabel': 'MARCCD165'},
        'output': {'fit_folder': str(tmp_path / 'fit')},
        'material': {'key_material': 'MgO'},
        'run': {'nb_cpus': 1}})
    params = IB.params_from_config(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = IB.index_refine(corfile, params)
    grains = res[1]
    # a failed indexing gives grains = [(0, None)]
    assert grains and grains[0][1] is not None, f'not indexed: nb spots, matching rate {res[2]}'

    # unstrained MgO: refined lattice must match the reference one
    B0 = CP.calc_B_RR(dict_Materials['MgO'][1])
    dlat = CP.dlat_to_rlat(CP.matrix_to_rlat(np.asarray(grains[0][1]) @ B0))
    a_ref = dict_Materials['MgO'][1][0]
    # B0 fixes the scale only up to a; compare the shape (ratios and angles)
    np.testing.assert_allclose(dlat[1:3] / dlat[0], [1.0, 1.0], atol=2e-4)
    np.testing.assert_allclose(dlat[3:], [90.0, 90.0, 90.0], atol=0.02)
    assert abs(dlat[0] - a_ref) / a_ref < 2e-3
