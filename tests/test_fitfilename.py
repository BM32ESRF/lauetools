# -*- coding: utf-8 -*-
"""
Output file names built from input file names that contain dots.

Names were cut at the first dot (name.split('.')[0]): MgO.r1_0000.cor and MgO.r1_0001.cor
both wrote MgO_g0.fit, the second silently overwriting the first.
Only the last extension must be removed.
"""
import warnings
from pathlib import Path

import numpy as np

import LaueTools.IOimagefile as IOimage
import LaueTools.IOLaueTools as IOLT
import LaueTools.LaueGeometry as LG
import LaueTools.lauecore as LC
from LaueTools.dict_LaueTools import dict_CCD

CALIB = [77.0, 1020.0, 1030.0, 0.3, -0.2]
PIXELSIZE = dict_CCD['MARCCD165'][1]


def _write_simulated_cor(folder: Path, rng, name: str) -> Path:
    """write name.cor: simulated MgO pattern, random orientation"""
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
    IOLT.writefile_cor(name, tth[order], chi[order], X[order], Y[order], intensity[order],
                       param={'dd': CALIB[0], 'xcen': CALIB[1], 'ycen': CALIB[2],
                              'xbet': CALIB[3], 'xgam': CALIB[4], 'pixelsize': PIXELSIZE,
                              'framedim': (2048, 2048), 'CCDLabel': 'MARCCD165',
                              'kf_direction': 'Z>0'},
                       dirname_output=str(folder))
    return folder / (name + '.cor')


def test_two_dotted_images_give_two_fitfiles(tmp_path):
    import LaueTools.indexing_batch as IB

    corfolder = tmp_path / 'cor'
    corfolder.mkdir()
    rng = np.random.default_rng(5)
    corfiles = [_write_simulated_cor(corfolder, rng, name=f'MgO.r1_{i:04d}') for i in range(2)]

    template = Path(IB.__file__).parent / 'notebooks' / 'indexation' / 'config_template.yaml'
    cfgfile = tmp_path / 'cfg.yaml'
    cfgfile.write_text(template.read_text())
    fitfolder = tmp_path / 'fit'
    cfg = IB.load_config(cfgfile, overrides={
        'scan': {'cor_folder': str(corfolder), 'prefix': 'MgO.r1_', 'nbdigits': 4,
                 'mapdims': [2, 1], 'CCDLabel': 'MARCCD165'},
        'output': {'fit_folder': str(fitfolder)},
        'material': {'key_material': 'MgO'},
        'run': {'nb_cpus': 1}})
    params = IB.params_from_config(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for corfile in corfiles:
            res = IB.index_refine(corfile, params)
            assert res[1] and res[1][0][1] is not None, f'{corfile.name} not indexed: {res[2]}'

    names = sorted(p.name for p in fitfolder.rglob('*.fit'))
    assert names == ['MgO.r1_0000_g0.fit', 'MgO.r1_0001_g0.fit']


def test_image_index_from_dotted_filename():
    assert IOimage.getIndex_fromfilename('MgO.r1_0012.tif', nbdigits=None, CCDLabel='sCMOS') == 12
    assert IOimage.getIndex_fromfilename('MgO.r1_0012.tif', nbdigits=4, CCDLabel='sCMOS') == 12
