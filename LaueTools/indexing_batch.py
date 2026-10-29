# -*- coding: utf-8 -*-
"""
Module of LaueTools project to index and refine a series of Laue patterns (.cor files -> .fit files)
with multiprocessing, driven by a YAML configuration file.

Used by the notebooks of LaueTools/notebooks/indexation/:

- 1_index_refine.ipynb      : .cor files -> .fit files
- 2_postprocess_maps.ipynb  : .fit files -> maps and statistics (strain, lattice parameters, orientation)
- 3_segmentation_grains.ipynb : .fit files -> grains (clusters), KAM, GROD, boundaries

Typical use::

    import LaueTools.indexing_batch as IB
    cfg = IB.load_config('configs/a321220_MgO.yaml')
    IB.check_config(cfg)
    params = IB.params_from_config(cfg)
    res = IB.index_refine(IB.corfile_path(cfg, 1300), params)          # test on 1 image
    allresults = IB.run_multiprocessing(IB.select_corfiles(cfg), params, nb_cpus=cfg['run']['nb_cpus'])
    summary = IB.summarize_results(allresults, cfg['scan']['mapdims'])
    IB.save_allresults(cfg, params, allresults, summary)

Map conventions (as in all LaueTools notebooks):

- mapdims = (nb of points along fast motor axis, nb of points along slow motor axis)
  e.g. for BLISS command: dmesh motor1 v v nbsteps1 motor2 v v nbsteps2 exposure_time
  mapdims = (nbsteps1 + 1, nbsteps2 + 1)
- invmapdims = (slow, fast) = shape of 2D arrays (nb rows, nb cols) used in imshow
- image index = row * mapdims[0] + col
"""
import copy
import os
import pickle
import time
import datetime
import dataclasses
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from LaueTools import indexingSpotsSet as ISS
from LaueTools import indexingAnglesLUT as IAL
from LaueTools import IOLaueTools as IOLT
from LaueTools import generaltools as GT
from LaueTools import dict_LaueTools as DictLT
from LaueTools import indexing_neighbours as IN

os.environ["OMP_NUM_THREADS"] = "1"   # 1 or 2 or 4

import socket
print("hostname =", socket.gethostname(), flush=True)
# --------------------------------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    'name': '',
    'description': '',
    'scan': {
        'cor_folder': None,       # folder of .cor files (peaksearch results)
        'prefix': 'img_',         # e.g. img_0023.cor -> 'img_'
        'nbdigits': 4,            # zero padding of image index: img_0023.cor -> 4
        # None: all .cor files in cor_folder. n (e.g. 10000): .cor files in subfolders of n files
        # <cor_folder>/<subfolder_prefix><start>_<end> (as written by peaksearch_batch with output: nbfiles_per_folder)
        'nbfiles_per_folder': None,
        'subfolder_prefix': 'images_',
        'CCDLabel': 'sCMOS',
        'detfile': None,          # calibration .det file (for information)
        'image_folder': None,     # folder of images (for information)
        'hdf5_logfile': None,     # BLISS hdf5 file of the experiment (optional)
        'mapdims': None,          # [fast, slow] nb of points
        'fast_motor': 'xech',     # label of the fast motor axis (plots)
        'slow_motor': 'yech',     # label of the slow motor axis (plots)
        'stepsizes': [1., 1.],    # [fast, slow] step sizes (plots)
    },
    'output': {
        'fit_folder': None,       # folder where .fit files are written (and read in postprocessing)
        'results_folder': None,   # folder of allresults_*.pickle (default: fit_folder)
        'results_stamp': None,    # allresults_<stamp>.pickle (default: key_material)
        # False: all .fit files in fit_folder. True (with scan: nbfiles_per_folder): .fit files in subfolders of
        # fit_folder with the same names as the subfolders of .cor files
        'fit_subfolders': False,
    },
    'material': {
        'key_material': None,
        # own materials (not in LaueTools default dict): {key: [key, [a, b, c, alpha, beta, gamma], extinction]}
        'mymaterials': None,
    },
    'indexing': {
        'e_min': 5,               # keV
        'e_max': 22,              # keV, for strain refinement
        'e_max_MR': 19,           # keV, matching rate of candidate UB matrices and first refinement step (then e_max)
        'nbGrainstoFind': 1,
        'depth': 0,               # microns (normal to sample surface, >0 below it). Reflection geometry Z>0; xbet, xgam neglected
        'MatchingRate_List': None,  # default: [2] * len(list matching tol angles)
        'MIN_NUMBERSPOTS_FOR_INDEXING': 6,
        'MAX_NUMBERSPOTS_FOR_INDEXING': 10000,
        'MAXNBSPOTS': None,       # max nb of spots read in .cor file (None: all)
        'crudeMReval': True,
        'maxnbspots_MReval': 10000,
        'stop_Nb_Matches': 40,
        # null: refinement steps given by dict_indexrefine: 'list matching tol angles' and MatchingRate_List, all
        # with e_max. Or list of steps [e_max (keV), matching tolerance angle (deg), min nb of matches]
        # (see refinement_ramp()), e.g. [[19, 0.3, 10], [23, 0.2, 12], [27, 0.1, 15]]: e_max increases from the
        # energy of indexing (e_max_MR) to the full range, for low symmetry or large unit cells.
        # Then e_max, 'list matching tol angles' and MatchingRate_List are not used
        'refinement_steps': None,
        # null, or criteria of a correct solution, checked for each grain found (result info, seeds validation of
        # the indexing guided by neighbours): steps [e_max (keV), tolerance angle (deg), min nb of matches]; from
        # the UB matrix found, at each step the spots up to e_max closer than the tolerance angle are linked, UB
        # and strain are refined, and at least min nb of matches indexed spots are needed, e.g. (NiMnGa)
        # [[18, 0.2, 100], [21, 0.2, 100], [24, 0.2, 100], [27, 0.2, 100], [27, 0.1, 100]]
        'solution_criteria': None,
        # max mean pixel deviation (pixel) of the last step of solution_criteria (null: not checked)
        'solution_max_pixdev': 1.0,
        # true: grains failing solution_criteria are rejected (.fit file removed; if the UB matrix came from
        # usepreviousUB and skipindexing is false, the image is then indexed from scratch). false: only reported
        'solution_reject': False,
        # spots removed from each .cor file before indexing (e.g. spots of the substrate):
        # null, a .cor, .dat or .fit file, a list of [X, Y] pixel positions, or a list of files
        # (the .cor files without these spots are written in <fit folder>/purged_corfiles)
        'blacklist': None,
        'blacklist_tolerance': 2.,  # pixel, max distance of a spot to a blacklisted spot to be removed
        'dict_indexrefine': {
            # spots set A (int n means range(n)). CAUTION max(A) <= MAXINDEXPROBED
            # or 'zones', 'triplets', 'zones+triplets': set A proposed for each image (spots likely to have low
            # Miller indices, see ISS.propose_spots_setA()); then set B = set A if 'spots set B' is None
            'central spots indices': 10,
            'proposal nb spots': 10,       # nb of spots proposed by each method
            'proposal zones tol': 0.05,    # deg, max distance of a spot to a zone axis (0.03 - 0.08)
            'proposal triplets tol': 0.03,  # deg, tolerance on the angles of a triplet (accurate lattice parameters)
            'proposal nb probed': 200,     # proposal among the first (most intense) spots
            'AngleTolLUT': 0.2,   # tolerance angle [deg] to recognise angles in the angular LUT
            'nlutmax': 4,         # max miller index in the angular LUT
            'MAXINDEXPROBED': 25,  # spots set B = [0, ..., MAXINDEXPROBED] (included). Former name: NBMAXPROBED
            # None: set B defined by MAXINDEXPROBED. Or set B (same syntax as set A): only pairs (spot of A, spot of B)
            # are tested, MAXINDEXPROBED is not used
            'spots set B': None,
            'MATCHINGRATE_ANGLE_TOL': 0.2,  # deg, tolerance angle for the first matching rate computation
            'MinimumMatchingRate': 5,       # minimum MR (%) to accept a previous results matrix for refinement
            'MinimumNumberMatches': 10,     # minimum nb of matches to keep a potential UB
            'CheckOrientation': None,
            'MATCHINGRATE_THRESHOLD_IAL': 2,
            'GuessedUBMatrix': None,
            'list matching tol angles': [0.2, 0.15, 0.1],  # refinement steps tolerance (deg)
            'UseIntensityWeights': False,
            'nbSpotsToIndex': 10000,
        },
    },
    'run': {
        'nb_cpus': None,           # None: all cpus - 1
        'ignorefitfileresults': True,   # True: reanalyse all files, False: only files without _g0.fit
        # False: indexing from scratch
        # 'fromfitfile': refine UB matrix of existing _g0.fit file
        # list of UB matrices (3x3 lists) or names of `ub_matrices`: refine these matrices
        # True: all the matrices of `ub_matrices`
        # 'neighbours': check first the UB matrices found in the neighbouring map points (see `neighbours`)
        'usepreviousUB': False,
        'skipindexing': False,     # True: only refine usepreviousUB matrices (no indexing from scratch)
        # .fit files of a previous indexation read by ignorefitfileresults: false and usepreviousUB: fromfitfile.
        # null: fit_folder (MW.run_step(): the previous results folder, archived as <fit_folder>_previous_<date>)
        'previous_fit_folder': None,
        # with ignorefitfileresults: false, previous .fit files kept only if their best grain has at least min_nb_spots
        # indexed spots and a mean pixel deviation at most max_pixdev (null: not checked); the others indexed again
        'fit_acceptance': {'min_nb_spots': None, 'max_pixdev': None},
        # usepreviousUB 'neighbours' (see LaueTools/indexing_neighbours.py): seed pass (points every seed_step indexed
        # from scratch), then each image checks the UB matrices of indexed points within radius, then the library
        # of the orientations found from scratch; accepted with min_nb_matches indexed spots, else from scratch
        'neighbours': dict(IN.DEFAULT_NEIGHBOURS),
        'writefitfile': True,
        'starting_grainindex': 0,
        'verboselevel': 0,
        'test_image_index': 0,     # image used for single image tests
    },
    'selection': {
        # None: all .cor files of cor_folder. Otherwise [start, stop] (stop included)
        'image_range': None,
        # rectangle region of interest in the map: [center image index, [half size fast, half size slow]]
        'roi': None,
        # list of image indices (e.g. seeds of the indexing guided by neighbours); None: no selection by list
        'image_indices': None,
    },
    'postprocess': {
        'grainindex': 0,               # read files ####_g<grainindex>.fit
        'MinimumNbIndexedSpots': 20,   # mask map points with less indexed spots
        'LargestMeanPixelResidue': 0.6,  # mask map points with larger mean pixel deviation
        'reference_image_index': None,   # for misorientation maps
        'symmetry': 'auto',             # 'cubic', None (raw angles) or 'auto' (segmentation, misorientation)
        'laue_group': 'auto',           # IPF colors: 'auto' (from lattice parameters), 'm-3m', '6/mmm', '-3m1', '4/mmm', 'mmm'
    },
    'ub_matrices': {},   # named UB matrices {name: 3x3 list}, usable in run: usepreviousUB
    # null: single material indexing. Or sequence of indexing steps (substrate, several materials ...),
    # see LaueTools/indexing_sequence.py
    'sequence': None,
}

_PATH_KEYS = (('scan', 'cor_folder'), ('scan', 'detfile'), ('scan', 'image_folder'), ('scan', 'hdf5_logfile'),
              ('output', 'fit_folder'), ('output', 'results_folder'))


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and key not in ('mymaterials', 'ub_matrices'):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(path, basedir: Path):
    if path in (None, ''):
        return None
    ppath = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not ppath.is_absolute():
        ppath = basedir / ppath
    return ppath


_SPOTS_SET_KEYS = ('central spots indices', 'spots set B')
# former names of parameters of dict_indexrefine (still accepted in configuration files and in changes)
_KEY_ALIASES = {'NBMAXPROBED': 'MAXINDEXPROBED'}


def _rename_former_keys(cfg: Optional[dict]) -> Optional[dict]:
    """configuration (as read in a file, or overrides) with former parameter names replaced (see _KEY_ALIASES)"""
    dir_ = ((cfg or {}).get('indexing') or {}).get('dict_indexrefine')
    if isinstance(dir_, dict):
        for old, new in _KEY_ALIASES.items():
            if old in dir_:
                value = dir_.pop(old)
                dir_.setdefault(new, value)
    return cfg


def _parse_indices(value) -> Optional[List[int]]:
    """spots set read in a configuration file: int n -> [0, ..., n-1] (convention of the YAML files),
    otherwise as ISS.parse_spots_set(): 'to12' -> [0, ..., 12], '80to90' -> [80, ..., 90],
    'start:stop' -> [start, ..., stop-1], list, None"""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return list(range(int(value)))
    if ISS.is_spots_proposal(value):   # 'zones', 'triplets', 'zones+triplets': set A proposed for each image
        return value.replace(' ', '').lower()
    return ISS.parse_spots_set(value)


def _spots_set_change(name: str, value):
    """value of a spots set given in python (int n -> [n], 'to12', list ... see ISS.parse_spots_set()),
    as written in a configuration file (never a bare int, whose meaning differs in YAML files)"""
    if name.replace('_', ' ') not in _SPOTS_SET_KEYS or value is None or ISS.is_spots_proposal(value):
        return value
    return _compact_indices(ISS.parse_spots_set(value))


def load_config(configfile: Union[str, Path], overrides: Optional[dict] = None) -> dict:
    """read a YAML configuration file and complete it with default values

    - '~' and environment variables ($HOME, ...) are expanded in paths
    - relative paths are relative to the folder of the configuration file
    - `overrides`: nested dict of values replacing those of the file (e.g. {'run': {'nb_cpus': 8}})

    Returns
    -------
    cfg : nested dict. Additional keys: cfg['configfile'], cfg['scan']['invmapdims']
    """
    configfile = Path(configfile).expanduser().resolve()
    with open(configfile, 'r') as f:
        usercfg = yaml.safe_load(f) or {}
    cfg = _deep_update(copy.deepcopy(DEFAULT_CONFIG), _rename_former_keys(usercfg))
    if overrides:
        _deep_update(cfg, _rename_former_keys(copy.deepcopy(overrides)))
    cfg['configfile'] = configfile

    for section, key in _PATH_KEYS:
        cfg[section][key] = _resolve_path(cfg[section][key], configfile.parent)
    if cfg['output']['results_folder'] is None:
        cfg['output']['results_folder'] = cfg['output']['fit_folder']

    mapdims = cfg['scan']['mapdims']
    if mapdims is not None:
        cfg['scan']['mapdims'] = (int(mapdims[0]), int(mapdims[1]))
        cfg['scan']['invmapdims'] = (int(mapdims[1]), int(mapdims[0]))
    else:
        cfg['scan']['invmapdims'] = None

    blacklist = cfg['indexing']['blacklist']   # files: relative to the folder of the configuration file
    if isinstance(blacklist, str):
        cfg['indexing']['blacklist'] = _resolve_path(blacklist, configfile.parent)
    elif isinstance(blacklist, list) and any(isinstance(b, str) for b in blacklist):
        cfg['indexing']['blacklist'] = [_resolve_path(b, configfile.parent) if isinstance(b, str) else b
                                        for b in blacklist]

    dir_ = cfg['indexing']['dict_indexrefine']
    for key in _SPOTS_SET_KEYS:
        dir_[key] = _parse_indices(dir_.get(key))
    if isinstance(cfg.get('sequence'), dict):   # steps of a sequence: same conventions as dict_indexrefine
        for step in cfg['sequence'].get('steps') or []:
            _rename_former_keys({'indexing': {'dict_indexrefine': step.get('dict_indexrefine')}})
            for key in _SPOTS_SET_KEYS:
                if key in (step.get('dict_indexrefine') or {}):
                    step['dict_indexrefine'][key] = _parse_indices(step['dict_indexrefine'][key])
    if cfg['indexing']['MatchingRate_List'] is None:
        cfg['indexing']['MatchingRate_List'] = [2] * len(dir_['list matching tol angles'])
    if cfg['output']['results_stamp'] is None:
        cfg['output']['results_stamp'] = cfg['material']['key_material']
    return cfg


# keys of a configuration not in DEFAULT_CONFIG but valid: added by load_config(), or free contents (names of
# materials, of UB matrices, steps of a sequence)
_EXTRA_KEYS = (('configfile',), ('scan', 'invmapdims'))
_FREE_KEYS = (('ub_matrices',), ('material', 'mymaterials'), ('sequence',))


def unknown_keys(cfg: dict, tree: Optional[dict] = None) -> Dict[tuple, Optional[str]]:
    """keys of cfg (icfg, scfg, ucfg ... or a peak search configuration) that are not parameters of the
    configuration (DEFAULT_CONFIG tree; peak search: PB.DEFAULT_CONFIG): created by mistake, e.g.
    cfg['indexing']['emax'] = 27 (typo), or a parameter in another section, they are not used (nor written by
    save_config(), except in dict_indexrefine). Returns {path: suggestion (path of a parameter of close name) or None}"""
    import difflib
    if tree is None:
        if 'peaksearch' in cfg and 'indexing' not in cfg:
            from LaueTools import peaksearch_batch as PB
            tree = PB.DEFAULT_CONFIG
        else:
            tree = DEFAULT_CONFIG
    leaves = _leaf_keys(tree)   # {name: path} of all the parameters
    out = {}

    def walk(node, subtree, path):
        for key, value in node.items():
            p = path + (key,)
            if p in _EXTRA_KEYS or p in _FREE_KEYS:
                continue
            if not isinstance(subtree, dict) or key not in subtree:
                name = _KEY_ALIASES.get(key, key)
                if name in leaves and leaves[name] != p:   # parameter of another section
                    out[p] = leaves[name]
                    continue
                close = difflib.get_close_matches(str(key), list(subtree) if isinstance(subtree, dict) else [], n=1,
                                                  cutoff=0.6) or difflib.get_close_matches(str(key), list(leaves), n=1,
                                                                                           cutoff=0.75)
                out[p] = (path + (close[0],) if isinstance(subtree, dict) and close[0] in subtree
                          else leaves[close[0]]) if close else None
            elif isinstance(value, dict) and isinstance(subtree[key], dict) and subtree[key]:
                walk(value, subtree[key], p)
    walk(cfg, tree, ())
    return out


def _unknown_messages(cfg: dict) -> List[str]:
    """messages of unknown_keys(): one per key created by mistake"""
    msgs = []
    for path, sugg in unknown_keys(cfg).items():
        where = ''.join(f'[{k!r}]' for k in path)
        hint = f": did you mean {''.join(f'[{k!r}]' for k in sugg)}?" if sugg else ''
        # all the keys of dict_indexrefine are written by save_config() (see _memory_changes())
        saved = 'written in the file by save_config()' if path[:2] == ('indexing', 'dict_indexrefine') else 'not saved'
        msgs.append(f'unknown parameter cfg{where} (not used, {saved}){hint}')
    return msgs


def check_config(cfg: dict, verbose: bool = True) -> bool:
    """check the consistency of the configuration (folders, material, indexing parameters).

    Print errors (red) and warnings (yellow). Return True if no error.
    """
    errors, warnings = [], _unknown_messages(cfg)
    scan = cfg['scan']
    nbpf = scan['nbfiles_per_folder']
    if nbpf is not None and (not isinstance(nbpf, int) or nbpf < 1):
        errors.append(f'scan: nbfiles_per_folder must be null or a positive integer, not {nbpf}')
    elif cfg['output']['fit_subfolders'] and not nbpf:
        warnings.append('output: fit_subfolders needs scan: nbfiles_per_folder: all .fit files in fit_folder')
    if scan['cor_folder'] is None or not Path(scan['cor_folder']).is_dir():
        errors.append(f"scan: cor_folder does not exist: {scan['cor_folder']}")
    elif not errors:
        nbcor = len(_corfiles(cfg))
        if nbcor == 0:
            where = f"{scan['subfolder_prefix']}*/ subfolders of " if nbpf else ''
            errors.append(f"no {scan['prefix']}*.cor file in {where}{scan['cor_folder']}")
            if not nbpf and any(Path(scan['cor_folder']).glob(f"{scan['subfolder_prefix']}*/{scan['prefix']}*.cor")):
                errors.append(f"  .cor files are in subfolders {scan['subfolder_prefix']}*: set scan: nbfiles_per_folder")
        elif verbose:
            print(f"{nbcor} .cor files found in {scan['cor_folder']}")
            if scan['mapdims'] is not None and nbcor != scan['mapdims'][0] * scan['mapdims'][1]:
                warnings.append(f"nb of .cor files ({nbcor}) differs from map size "
                                f"{scan['mapdims'][0]}x{scan['mapdims'][1]} = {scan['mapdims'][0] * scan['mapdims'][1]}")
    if scan['mapdims'] is None:
        errors.append('scan: mapdims is missing ([nb points fast axis, nb points slow axis])')
    if scan['detfile'] is not None and not Path(scan['detfile']).exists():
        warnings.append(f"scan: detfile not found: {scan['detfile']}")
    if cfg['output']['fit_folder'] is None:
        errors.append('output: fit_folder is missing')

    key_material = cfg['material']['key_material']
    mymaterials = cfg['material']['mymaterials']
    if key_material is None:
        errors.append('material: key_material is missing')
    elif not (mymaterials and key_material in mymaterials) and key_material not in DictLT.dict_Materials:
        errors.append(f"material: '{key_material}' is neither in LaueTools dict of materials nor in mymaterials. "
                      "Add it in mymaterials: {key: [key, [a, b, c, alpha, beta, gamma], extinction]}")

    dir_ = cfg['indexing']['dict_indexrefine']
    if ISS.is_spots_proposal(dir_['central spots indices']):
        if verbose:
            print(f"spots set A proposed for each image by '{dir_['central spots indices']}' "
                  f"({dir_.get('proposal nb spots', 10)} spots per method among the "
                  f"{dir_.get('proposal nb probed', 200)} first spots)"
                  + ('; set B = set A' if dir_.get('spots set B') is None else ''))
    elif dir_.get('spots set B') is None and max(dir_['central spots indices']) > int(dir_['MAXINDEXPROBED']):
        warnings.append(f"indexing: MAXINDEXPROBED ({dir_['MAXINDEXPROBED']}) must be at least the largest central "
                        f"spots index ({max(dir_['central spots indices'])}): {max(dir_['central spots indices'])} "
                        "will be used (or set 'spots set B')")
    elif dir_.get('spots set B') is not None and len(set(dir_['spots set B']) | set(dir_['central spots indices'])) < 2:
        errors.append("indexing: 'spots set B' and 'central spots indices' must contain at least 2 different spots")
    if cfg['indexing'].get('refinement_steps') is not None:
        try:
            steps = ISS.parse_refinement_steps(cfg['indexing']['refinement_steps'])
            if verbose:
                print('refinement steps [e_max, tol angle, min nb matches]: '
                      + ', '.join(f'[{e:g}, {t:g}, {n}]' for e, t, n, _ in steps)
                      + " (e_max, 'list matching tol angles' and MatchingRate_List are not used)")
                if steps and cfg['indexing']['e_max_MR'] is not None and steps[0][0] != cfg['indexing']['e_max_MR']:
                    GT.printyellow(f"NB: first refinement step up to {steps[0][0]:g} keV, whereas candidate UB matrices "
                                   f"are evaluated up to e_max_MR = {cfg['indexing']['e_max_MR']:g} keV")
        except (ValueError, TypeError) as err:
            errors.append(f'indexing: refinement_steps: {err}')
    if cfg['indexing'].get('solution_criteria') is not None:
        try:
            ISS.parse_refinement_steps(cfg['indexing']['solution_criteria'])
        except (ValueError, TypeError) as err:
            errors.append(f'indexing: solution_criteria: {err}')
    acc = cfg['run'].get('fit_acceptance') or {}
    unknown = set(acc) - {'min_nb_spots', 'max_pixdev'}
    if unknown:
        errors.append(f'run: fit_acceptance: unknown key(s) {sorted(unknown)} (min_nb_spots, max_pixdev)')
    for key in ('min_nb_spots', 'max_pixdev'):
        if acc.get(key) is not None and (not isinstance(acc[key], (int, float)) or acc[key] <= 0):
            errors.append(f'run: fit_acceptance: {key} must be null or a positive number, not {acc[key]}')
    if cfg['run'].get('previous_fit_folder') and not Path(cfg['run']['previous_fit_folder']).is_dir():
        errors.append(f"run: previous_fit_folder {cfg['run']['previous_fit_folder']} not found")
    maxpd = cfg['indexing'].get('solution_max_pixdev')
    if maxpd is not None and (not isinstance(maxpd, (int, float)) or maxpd <= 0):
        errors.append(f'indexing: solution_max_pixdev must be null or a positive number, not {maxpd}')
    if len(cfg['indexing']['MatchingRate_List']) < len(dir_['list matching tol angles']):
        errors.append("indexing: MatchingRate_List must be at least as long as 'list matching tol angles'")
    try:
        get_previousUB(cfg)
    except (KeyError, ValueError) as err:
        errors.append(f'run: usepreviousUB: {err}')
    if cfg['run']['usepreviousUB'] == 'neighbours':
        errors.extend(IN.check_settings(cfg['run']['neighbours'], cfg['ub_matrices']))
        if cfg.get('sequence') is not None:
            errors.append("run: usepreviousUB 'neighbours' is not available with a sequence of indexing steps")
    if cfg.get('sequence') is not None:
        from LaueTools import indexing_sequence as IS
        errors.extend(IS.check_sequence(cfg))
    if cfg['indexing']['blacklist'] is not None:
        try:
            nbblack = len(read_blacklist(cfg['indexing']['blacklist']))
            if verbose:
                print(f"blacklist: {nbblack} spots removed (within {cfg['indexing']['blacklist_tolerance']} pixel) "
                      "from each .cor file before indexing")
        except (OSError, ValueError, IndexError) as err:
            errors.append(f'indexing: blacklist: {err}')

    if verbose:
        for msg in warnings:
            GT.printyellow('WARNING: ' + msg)
        for msg in errors:
            GT.printred('ERROR: ' + msg)
        if not errors:
            GT.printgreen('Configuration OK')
    return not errors


def print_config(cfg: dict):
    """print the main configuration parameters"""
    scan, out = cfg['scan'], cfg['output']
    print(f"---- {cfg['name']} ----  ({cfg['configfile']})")
    subfolder = f"{scan['subfolder_prefix']}<start>_<end>/" if scan['nbfiles_per_folder'] else ''
    print(f".cor files:  {scan['cor_folder']}/{subfolder}{scan['prefix']}{'#' * scan['nbdigits']}.cor"
          + (f"  ({scan['nbfiles_per_folder']} files per subfolder)" if subfolder else ''))
    print(f".fit files:  {out['fit_folder']}" + (f'/{subfolder}' if subfolder and out['fit_subfolders'] else ''))
    print(f"map:         mapdims (fast, slow) = {scan['mapdims']}, motors ({scan['fast_motor']}, {scan['slow_motor']})")
    print(f"detector:    {scan['CCDLabel']}, calibration {scan['detfile']}")
    ind = cfg['indexing']
    print(f"material:    {cfg['material']['key_material']}, energy {ind['e_min']}-{ind['e_max']} keV "
          f"(indexing up to {ind['e_max_MR']} keV), nbGrainstoFind = {ind['nbGrainstoFind']}")


def show_config(cfg: dict, sections=None, changes: bool = True, maxlist: int = 10, decimals: int = 5):
    """pretty print of a configuration (icfg, scfg, ucfg ... of the notebooks): nested dicts indented, one
    parameter per line, 3x3 matrices (ub_matrices) on 3 lines, lists longer than maxlist shortened

    sections: name or list of names of the sections printed (e.g. 'indexing', ['scan', 'ub_matrices']), None: all
    changes: parameters changed in memory (e.g. icfg['indexing']['e_max'] = 27) and not yet in the configuration
        file are marked '*' with the value of the file (written by save_config())
    keys that are not parameters of the configuration (typo, wrong section: not used, not saved, see
    unknown_keys()) are marked '!' and listed in red at the end"""
    memory = {}
    if changes:
        try:
            if 'peaksearch' in cfg and 'indexing' not in cfg:
                from LaueTools import peaksearch_batch as PB
                memory = _memory_changes(cfg, PB.DEFAULT_CONFIG, PB.load_config, with_old=True)
            else:
                memory = _memory_changes(cfg, DEFAULT_CONFIG, load_config, with_old=True)
        except Exception:   # configuration file missing or not readable: no comparison
            pass
    unknown = unknown_keys(cfg)
    if isinstance(sections, str):
        sections = [sections]
    width = 32

    def fmt(value):
        value = _to_plain(value)
        if isinstance(value, float):
            return f'{value:.{decimals}g}'
        if isinstance(value, list) and len(value) > maxlist:
            return f'{value[:maxlist // 2]} ... {value[-2:]} ({len(value)} values)'
        return repr(value) if isinstance(value, str) else str(value)

    def is_matrix(value):
        arr = np.asarray(_to_plain(value), dtype=object)
        return arr.shape == (3, 3) and all(isinstance(v, (int, float)) for v in arr.ravel())

    def show(node, path, indent):
        pad = '   ' * indent
        for key, value in node.items():
            mark = '!' if path + (key,) in unknown else '*' if path + (key,) in memory else ' '
            if isinstance(value, dict):
                print(f'{pad}{mark}{key}:' + ('  {}' if not value else ''))
                show(value, path + (key,), indent + 1)
            elif is_matrix(value) and ('ub_matrices' in path or 'UB' in str(key)):
                rows = np.asarray(_to_plain(value), dtype=float)
                print(f'{pad}{mark}{key}:')
                for row in rows:
                    print(f'{pad}      [' + ', '.join(f'{v:{decimals + 4}.{decimals}f}' for v in row) + ']')
            else:
                old = f'   (file: {fmt(memory[path + (key,)][1])})' if mark == '*' else ''
                print(f'{pad}{mark}{str(key):<{width - len(pad)}}  {fmt(value)}{old}')

    print(f"---- {cfg.get('name', '')} ----  ({cfg.get('configfile')})")
    for key, value in cfg.items():
        if sections is not None and key not in sections:
            continue
        if isinstance(value, dict):
            mark = '!' if (key,) in unknown else '*' if (key,) in memory else ' '
            print(f'\n{mark}[{key}]' + ('  {}' if not value else ''))
            show(value, (key,), 1)
        elif sections is None and key not in ('name', 'configfile'):
            print(f"{'!' if (key,) in unknown else ' '}{str(key):<{width}}  {fmt(value)}")
    for msg in _unknown_messages(cfg):
        GT.printred('! ' + msg)
    if memory:
        GT.printyellow(f'* {len(memory)} parameter(s) changed in memory, not in {cfg.get("configfile")} '
                       '(written by IB.save_config())')


# --------------------------------------------------------------------------------------
#  WRITE CONFIGURATION FILES (no manual editing of YAML indentation)
# --------------------------------------------------------------------------------------
TEMPLATE_CONFIG = Path(__file__).parent / 'notebooks' / 'indexation' / 'config_template.yaml'


def _leaf_keys(tree: dict = DEFAULT_CONFIG, path: tuple = ()) -> Dict[str, tuple]:
    """{parameter name: path in the configuration} e.g. 'nbGrainstoFind': ('indexing', 'nbGrainstoFind')"""
    leaves = {}
    for key, value in tree.items():
        if isinstance(value, dict) and key not in ('mymaterials', 'ub_matrices'):
            leaves.update(_leaf_keys(value, path + (key,)))
        else:
            leaves[key] = path + (key,)
    return leaves


def _key_path(name: str, tree: dict = DEFAULT_CONFIG) -> tuple:
    """path of a parameter given by its name ('central_spots_indices' is accepted for 'central spots indices')"""
    leaves = _leaf_keys(tree)
    name = _KEY_ALIASES.get(name, name)
    for candidate in (name, name.replace('_', ' ')):
        if candidate in leaves:
            return leaves[candidate]
    import difflib
    close = difflib.get_close_matches(name, list(leaves), n=3)
    raise KeyError(f"unknown configuration parameter '{name}'" + (f'. Did you mean {close}?' if close else ''))


def check_misplaced_keys(cfg: dict, tree: dict = DEFAULT_CONFIG, name: str = 'cfg'):
    """error if a parameter is set at the top level of a configuration (e.g. pcfg['IntensityThreshold'] = 300 instead
    of pcfg['peaksearch']['IntensityThreshold']): it would not be used (silently)"""
    leaves = _leaf_keys(tree)
    misplaced = {}
    for key in cfg:
        if key in tree or not isinstance(key, str):
            continue
        for candidate in (_KEY_ALIASES.get(key, key), key.replace('_', ' ')):
            if candidate in leaves and len(leaves[candidate]) > 1:
                misplaced[key] = leaves[candidate]
                break
    if misplaced:
        where = ', '.join(f"{name}[{k!r}] -> {name}" + ''.join(f'[{p!r}]' for p in path) for k, path in misplaced.items())
        raise KeyError(f'parameter(s) set at the wrong place of the configuration, they would not be used: {where}. '
                       f'Set them with MW.set_parameters({name}, ...) (dict PEAKSEARCH_TEST / INDEXING_TEST of the test '
                       f'cells of the quickstart notebook), and delete the wrong keys (or run the cell of STEP 2 again)')


def _to_plain(value):
    """numpy arrays/scalars, Path, tuple -> python types writable in YAML"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


def _same(a, b) -> bool:
    a, b = _to_plain(a), _to_plain(b)
    if a is None or b is None or isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return a == b
    try:
        return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    except (ValueError, TypeError):
        return a == b


def _compact_indices(indices):
    """[0, ..., n] -> 'to<n>', [s, ..., e] -> '<s>to<e>' (contiguous, last index included), else list"""
    indices = [int(i) for i in indices]
    if len(indices) > 3 and indices == list(range(indices[0], indices[-1] + 1)):
        return f'to{indices[-1]}' if indices[0] == 0 else f'{indices[0]}to{indices[-1]}'
    return indices


def _write_config(source: Union[str, Path], newfile: Union[str, Path], changes: Dict[tuple, Any],
                  overwrite: bool = False, header: str = '', path_keys: tuple = _PATH_KEYS) -> Path:
    """copy the YAML file `source` to `newfile` with values replaced by `changes` {path: value}.

    Comments and layout are kept if ruamel.yaml is installed.
    Relative paths of `source` (keys `path_keys`) are made absolute if `newfile` is in another folder.
    """
    source = Path(source).expanduser().resolve()
    newfile = Path(newfile).expanduser().resolve()
    if newfile.exists() and not overwrite and newfile != source:
        raise FileExistsError(f'{newfile} already exists. Use overwrite=True or another file name')
    if newfile == source and not overwrite:
        raise FileExistsError(f'{newfile} is the source file: use overwrite=True to update it in place')
    text = source.read_text()

    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap, CommentedSeq
        ryaml = YAML()
        ryaml.preserve_quotes = True
        ryaml.width = 200
        ryaml.representer.add_representer(type(None), lambda rep, _: rep.represent_scalar('tag:yaml.org,2002:null', 'null'))
        data = ryaml.load(text) or CommentedMap()

        def to_yaml(value):
            if isinstance(value, list):
                seq = CommentedSeq([to_yaml(v) for v in value])
                if not any(isinstance(v, dict) for v in value):   # lists of dicts (steps): block style
                    seq.fa.set_flow_style()
                return seq
            if isinstance(value, dict):
                cmap = CommentedMap()
                for k, v in value.items():
                    cmap[k] = to_yaml(v)
                return cmap
            return value
        newsection = CommentedMap
    except ImportError:
        ryaml = None
        data = yaml.safe_load(text) or {}
        to_yaml = lambda value: value
        newsection = dict
        GT.printyellow('ruamel.yaml not installed: comments of the configuration file are not kept '
                       '(pip install ruamel.yaml)')

    if newfile.parent != source.parent:
        for section, key in path_keys:
            value = (data.get(section) or {}).get(key)
            if value not in (None, '') and (section, key) not in changes:
                abspath = _resolve_path(value, source.parent)
                if str(abspath) != os.path.expandvars(os.path.expanduser(str(value))):
                    changes[(section, key)] = str(abspath)

    for path, value in changes.items():
        node = data
        for key in path[:-1]:
            if node.get(key) is None:
                node[key] = newsection()
            node = node[key]
        former = [old for old, new in _KEY_ALIASES.items() if new == path[-1] and old in node]
        if former and hasattr(node, 'insert'):   # ruamel: new name at the place of the former one
            position = list(node.keys()).index(former[0])
            del node[former[0]]
            node.insert(position, path[-1], to_yaml(_to_plain(value)))
            continue
        for old in former:
            del node[old]
        node[path[-1]] = to_yaml(_to_plain(value))

    newfile.parent.mkdir(parents=True, exist_ok=True)
    if ryaml is not None:
        import io
        buf = io.StringIO()
        ryaml.dump(data, buf)
        body = buf.getvalue()
    else:
        body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=200)
    newfile.write_text(header + body)
    return newfile


def new_config(newfile: Union[str, Path], template: Optional[Union[str, Path]] = None,
               overwrite: bool = False, check: bool = True, **changes) -> dict:
    """create a configuration file from a template, with some parameters changed (no manual YAML editing)

    template: config_template.yaml of LaueTools (default) or the configuration file of another experiment.
    changes: parameters given by their name, whatever their section, e.g.
        IB.new_config('configs/myexp.yaml', cor_folder='/data/.../corfiles', prefix='img_', mapdims=[51, 51],
                      fit_folder='/data/.../fitfiles', key_material='Al', nbGrainstoFind=2,
                      central_spots_indices='to9')
    ('central_spots_indices' stands for 'central spots indices', 'list_matching_tol_angles' for
    'list matching tol angles'). Spots sets ('central spots indices', 'spots set B'): as in dict_indexrefine,
    e.g. 'to9' (0 to 9), '80to90', [0, 1, 8], 5 (-> [5]) (see ISS.parse_spots_set()).

    check: check_config() of the new configuration (False e.g. if the .cor files do not exist yet)
    Returns the configuration loaded from the new file (as load_config()).
    """
    template = TEMPLATE_CONFIG if template is None else template
    pathchanges = {_key_path(name): _spots_set_change(name, value) for name, value in changes.items()}
    header = f'# created {datetime.datetime.now():%Y-%m-%d %H:%M} from {Path(template).name} (IB.new_config)\n'
    newfile = _write_config(template, newfile, pathchanges, overwrite=overwrite, header=header)
    GT.printgreen(f'configuration written: {newfile}')
    cfg = load_config(newfile)
    if check:
        check_config(cfg)
    return cfg


# IndexRefineParams fields saved by save_config() (run options of tests, e.g. verboselevel, are not saved)
_PARAMS_TO_CONFIG = {name: ('indexing', name) for name in
                     ('e_min', 'e_max', 'e_max_MR', 'nbGrainstoFind', 'depth', 'MIN_NUMBERSPOTS_FOR_INDEXING',
                      'MAX_NUMBERSPOTS_FOR_INDEXING', 'MAXNBSPOTS', 'crudeMReval', 'maxnbspots_MReval',
                      'stop_Nb_Matches', 'MatchingRate_List', 'blacklist', 'blacklist_tolerance',
                      'refinement_steps')}
_PARAMS_TO_CONFIG.update({'key_material': ('material', 'key_material'),
                          'mymaterials': ('material', 'mymaterials'),
                          'fit_folder': ('output', 'fit_folder'),
                          'skipindexing': ('run', 'skipindexing'),
                          'CCDLabel': ('scan', 'CCDLabel')})


def _memory_changes(cfg: dict, tree: dict, loader, with_old: bool = False) -> Dict[tuple, Any]:
    """{path: value} of the parameters of cfg changed in memory (e.g. icfg['indexing']['e_max'] = 27 in a notebook)
    with respect to its configuration file cfg['configfile'] (read again with loader, e.g. load_config), for
    save_config(): parameters of the default configuration tree, all the keys of dict_indexrefine and ub_matrices.
    with_old: {path: (value, value in the file)}"""
    if not cfg.get('configfile') or not Path(cfg['configfile']).exists():
        return {}
    ondisk = loader(cfg['configfile'])

    def get(node, path):
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        return node

    paths = list(_leaf_keys(tree).values())
    dir_ = get(cfg, ('indexing', 'dict_indexrefine'))
    if isinstance(dir_, dict):   # keys of dict_indexrefine not in the default tree
        paths += [('indexing', 'dict_indexrefine', k) for k in dir_ if ('indexing', 'dict_indexrefine', k) not in paths]
    if 'ub_matrices' in cfg:
        paths.append(('ub_matrices',))
    changes = {}
    for path in paths:
        if path[0] == 'ub_matrices' and len(path) > 1:
            continue
        value, old = get(cfg, path), get(ondisk, path)
        if isinstance(value, Path) or isinstance(old, Path):   # paths compared resolved
            same = (value is None) == (old is None) and (value is None or Path(value).resolve() == Path(old).resolve())
        else:
            same = _same(value, old)
        if not same:
            changes[path] = (value, old) if with_old else value
    return changes


def save_config(cfg: dict, newfile: Union[str, Path], params: Optional['IndexRefineParams'] = None,
                overwrite: bool = False, **changes) -> dict:
    """save a configuration file = file of `cfg` + parameters changed in `params` (e.g. after a test) + `changes`

    Only the parameters of `params` that differ from `cfg` are written (indexing parameters, dict_indexrefine,
    material, fit_folder, usepreviousUB, skipindexing), with the comments of the original file.
    Run options of tests (verboselevel, ignorefitfileresults, outputlistindices ...) are not saved.
    `changes`: other parameters given by their name, as in new_config() (e.g. test_image_index=1300).

    Example, after a successful test on 1 image:
        cfg = IB.save_config(cfg, 'configs/a321220_MgO_2grains.yaml', params=params_test)

    Parameters of cfg changed in memory (e.g. icfg['indexing']['e_max'] = 27 in a notebook) are written too.

    Returns the configuration loaded from the new file: use it for the batch (and its path as CONFIG
    in the postprocessing notebooks).
    """
    memory = _memory_changes(cfg, DEFAULT_CONFIG, load_config, with_old=True)
    pathchanges = {path: (_spots_set_change(path[-1], value) if path[-1] in _SPOTS_SET_KEYS else value)
                   for path, (value, _) in memory.items()}
    if pathchanges:
        print(f'{len(pathchanges)} parameter(s) changed in memory (cfg) written too')
    if params is not None:
        for field, path in _PARAMS_TO_CONFIG.items():
            value = getattr(params, field)
            old = cfg[path[0]][path[1]]
            if field == 'fit_folder':
                value = Path(os.path.expanduser(str(value))).resolve()
                old = None if old is None else Path(old).resolve()
            if not _same(value, old):
                pathchanges[path] = value

        cfg_dir = cfg['indexing']['dict_indexrefine']
        for key, value in params.dict_indexrefine.items():
            if key in _SPOTS_SET_KEYS and not ISS.is_spots_proposal(value):
                value = ISS.parse_spots_set(value)
            if not _same(value, cfg_dir.get(key)):
                pathchanges[('indexing', 'dict_indexrefine', key)] = (_spots_set_change(key, value)
                                                                      if key in _SPOTS_SET_KEYS else value)

        oldUB, newUB = get_previousUB(cfg), params.usepreviousUB
        if isinstance(newUB, np.ndarray):
            newUB = [newUB]
        sameUB = (isinstance(oldUB, list) and isinstance(newUB, list) and len(oldUB) == len(newUB)
                  and all(np.allclose(a, b) for a, b in zip(oldUB, newUB)))
        if not sameUB and not _same(oldUB, newUB):
            if isinstance(newUB, list):   # matrices -> names of ub_matrices (existing names or new ones)
                ub_matrices = dict(cfg['ub_matrices'] or {})
                names = []
                for mat in newUB:
                    name = next((k for k, v in ub_matrices.items() if np.allclose(v, mat)), None)
                    if name is None:
                        name = f'UB_{len(ub_matrices) + 1}'
                        while name in ub_matrices:
                            name += '_'
                        ub_matrices[name] = np.asarray(mat, dtype=float).tolist()
                        pathchanges[('ub_matrices', name)] = ub_matrices[name]
                    names.append(name)
                newUB = names
            pathchanges[('run', 'usepreviousUB')] = newUB

    pathchanges.update({_key_path(name): _spots_set_change(name, value) for name, value in changes.items()})

    if not pathchanges:
        print('no parameter differs from the configuration file')
    for path, value in pathchanges.items():
        old = cfg
        for key in path:
            old = old.get(key) if isinstance(old, dict) else None
        if path in memory:   # value in the file (cfg holds the new value)
            old = memory[path][1]
        print(f"  {': '.join(path)}: {_to_plain(old)} -> {_to_plain(value)}")

    header = (f"# saved {datetime.datetime.now():%Y-%m-%d %H:%M} from {Path(cfg['configfile']).name} "
              f"(IB.save_config)\n")
    newfile = _write_config(cfg['configfile'], newfile, pathchanges, overwrite=overwrite, header=header)
    GT.printgreen(f'configuration written: {newfile}')
    return load_config(newfile)


# --------------------------------------------------------------------------------------
#  .irp FILES (index & refine parameters of the GUIs and of FileSeries) -> YAML configuration
# --------------------------------------------------------------------------------------
def _irp_value(value, kind=str):
    """value of a .irp option (read as text) -> python value"""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text in ('None', 'none', ''):
        return None
    if kind is bool:
        return text in ('True', 'true', '1')
    if kind is list:
        import ast
        return list(ast.literal_eval(text))
    return kind(text)


def irp_to_changes(irpfile: Union[str, Path], material_index: int = 0, verbose: bool = True) -> dict:
    """parameters of a .irp file (material #material_index) as configuration changes {parameter name: value}
    (see new_config()), with the settings used by FileSeries (indexFilesSeries) for the parameters not in .irp files:
    matching rate up to 'emax MR' (emax if missing), crudeMReval=False, stop_Nb_Matches=1000, MatchingRate_List 1 % at each step.
    Parameters of .irp files not used by indexing_batch (ResolutionAngstrom, setCentralSpotsHKL) are reported."""
    params_list = ISS.readIndexRefineConfigFile(str(irpfile))
    if not params_list:
        raise ValueError(f'can not read the .irp file {irpfile}')
    if len(params_list) > 1 and verbose:
        GT.printyellow(f'{irpfile} contains {len(params_list)} materials: only material #{material_index} '
                       f"({params_list[material_index].get('key material')}) is converted")
    irp = params_list[material_index]
    maxindex, setB = ISS.maxindexprobed_from_irp(irp['MAXINDEXPROBED'])
    tolangles = [float(a) for a in _irp_value(irp['List Matching Tol Angles'], list)]
    emax = float(irp['emax'])
    emax_MR = _irp_value(irp.get('emax MR'), float)   # matching rate during indexing (None: emax)
    changes = {'key_material': irp['key material'],
               'nbGrainstoFind': int(irp['nbGrainstoFind']),
               'e_min': float(irp['emin']), 'e_max': emax, 'e_max_MR': emax if emax_MR is None else emax_MR,
               'MatchingRate_List': [1] * len(tolangles),
               'crudeMReval': False, 'stop_Nb_Matches': 1000,
               'central spots indices': ISS.parse_spots_set(irp['central spots indices']),
               'MAXINDEXPROBED': maxindex, 'spots set B': setB,
               'AngleTolLUT': float(irp['AngleTolLUT']),
               'MATCHINGRATE_ANGLE_TOL': float(irp['MATCHINGRATE ANGLE TOL']),
               'MATCHINGRATE_THRESHOLD_IAL': float(irp['MATCHINGRATE THRESHOLD IAL']),
               'MinimumNumberMatches': int(irp['MinimumNumberMatches']),
               'list matching tol angles': tolangles}
    if irp.get('nLUTmax') is not None:
        changes['nlutmax'] = int(irp['nLUTmax'])
    if irp.get('UseIntensityWeights') is not None:
        changes['UseIntensityWeights'] = _irp_value(irp['UseIntensityWeights'], bool)
    if irp.get('nbSpotsToIndex') is not None:
        changes['nbSpotsToIndex'] = int(irp['nbSpotsToIndex'])
    ignored = {key: irp[key] for key in ('ResolutionAngstrom', 'setCentralSpotsHKL')
               if _irp_value(irp.get(key)) not in (None, False, 'False', 0, '0')}
    if ignored and verbose:
        GT.printyellow(f'parameters of {Path(irpfile).name} not used by indexing_batch: {ignored}')
    return changes


def config_from_irp(irpfile: Union[str, Path], newfile: Union[str, Path], cor_folder: Union[str, Path],
                    fit_folder: Union[str, Path], prefix: str, nbdigits: int = 4, image_range=None, mapdims=None,
                    guessed_matrices: Optional[Union[str, Path]] = None, minimum_matching_rate: Optional[float] = None,
                    material_index: int = 0, template: Optional[Union[str, Path]] = None, overwrite: bool = False,
                    check: bool = True, **changes) -> dict:
    """YAML configuration file of indexing_batch from a .irp file (of the GUIs, FileSeries) and the settings of the
    FileSeries Index & Refine board

    irpfile: .irp file (index & refine parameters); only one material (material_index) is converted
    cor_folder, fit_folder, prefix, nbdigits: .cor files <cor_folder>/<prefix>####.cor, .fit files in fit_folder
    image_range: [first, last] image indices (None: all .cor files)
    mapdims: [nb points fast axis, nb points slow axis] (None: [largest image index + 1, 1])
    guessed_matrices: .mat or .mats file of UB matrices tested before indexing from scratch (as FileSeries
        'Guessed Matrix(ces)'), with minimum_matching_rate (%) to accept them. .ubs files are not supported
    changes: other parameters by name (see new_config()), e.g. detfile='...det', CCDLabel='sCMOS', nb_cpus=8

    Returns the configuration loaded from newfile (as load_config())
    """
    allchanges = irp_to_changes(irpfile, material_index)
    import re
    cor_folder = Path(cor_folder)
    corfiles = sorted(cor_folder.glob(f'{prefix}*.cor'))
    indices = [int(m.group(1)) for f in corfiles if (m := re.fullmatch(rf'{re.escape(prefix)}(\d+)\.cor', f.name))]
    if image_range is not None:
        image_range = [int(image_range[0]), int(image_range[1])]
    if mapdims is None:
        # 1D map containing all image indices (image index = position in the map)
        last = image_range[1] if image_range is not None else max(indices, default=0)
        mapdims = [last + 1, 1]
    allchanges.update({'cor_folder': str(cor_folder), 'fit_folder': str(fit_folder), 'prefix': prefix,
                       'nbdigits': int(nbdigits), 'image_range': image_range, 'mapdims': list(mapdims),
                       'name': Path(newfile).stem, 'description': f'converted from {irpfile}'})
    if corfiles and 'CCDLabel' not in changes:
        res = IOLT.readfile_cor(str(corfiles[0]), output_CCDparamsdict=True)
        if res is not None and res[-1].get('CCDLabel') not in (None, 'None'):
            allchanges['CCDLabel'] = res[-1]['CCDLabel']
    if guessed_matrices not in (None, 'None', 'none', ''):
        if str(guessed_matrices).endswith('.ubs'):
            raise ValueError(f'{guessed_matrices}: .ubs files (check orientation parameters) are not supported by '
                             'indexing_batch')
        _, matrices = IOLT.readListofMatrices(str(guessed_matrices))
        names = [f'guessed_{k + 1}' for k in range(len(matrices))]
        allchanges['ub_matrices'] = {n: np.asarray(m, dtype=float).tolist() for n, m in zip(names, matrices)}
        allchanges['usepreviousUB'] = names
        allchanges['skipindexing'] = False
    if minimum_matching_rate is not None:
        allchanges['MinimumMatchingRate'] = float(minimum_matching_rate)
    allchanges.update(changes)
    return new_config(newfile, template=template, overwrite=overwrite, check=check, **allchanges)


def merge_grain_fitfiles(fit_folder: Union[str, Path], stem: str) -> Optional[Path]:
    """write <fit_folder>/<stem>.fit with the spots of all grains (<stem>_g0.fit, <stem>_g1.fit ...), as written
    by FileSeries (read by FileSeries Build Summary). Returns the written file (None if no grain file)"""
    import re
    fit_folder = Path(fit_folder)
    grainfiles = [f for f in fit_folder.glob(f'{stem}_g*.fit') if re.fullmatch(rf'{re.escape(stem)}_g\d+\.fit', f.name)]
    grainfiles.sort(key=lambda f: int(re.search(r'_g(\d+)\.fit$', f.name).group(1)))
    if not grainfiles:
        return None
    merged = fit_folder / f'{stem}.fit'
    merged.write_text(''.join(f.read_text() + '\n' for f in grainfiles))
    return merged


def get_previousUB(cfg: dict):
    """return `usepreviousUB` of cfg['run'] with named matrices replaced by numpy arrays

    True: all the matrices of ub_matrices. Returns False, 'fromfitfile', 'neighbours' or a list of 3x3 arrays
    """
    usepreviousUB = cfg['run']['usepreviousUB']
    if usepreviousUB in (False, None):
        return False
    if usepreviousUB in ('fromfitfile', 'neighbours'):
        return usepreviousUB
    if usepreviousUB is True:   # all the matrices of ub_matrices
        usepreviousUB = list(cfg['ub_matrices'] or {})
        if not usepreviousUB:
            raise ValueError('usepreviousUB true: ub_matrices is empty')
    if isinstance(usepreviousUB, str):
        usepreviousUB = [usepreviousUB]
    matrices = []
    for item in usepreviousUB:
        if isinstance(item, str):
            if item not in cfg['ub_matrices']:
                raise KeyError(f"UB matrix '{item}' not found in ub_matrices")
            item = cfg['ub_matrices'][item]
        mat = np.array(item, dtype=float)
        if mat.shape != (3, 3):
            raise ValueError(f'UB matrix must be 3x3, got shape {mat.shape}')
        matrices.append(mat)
    return matrices


# --------------------------------------------------------------------------------------
#  INDEX & REFINE of 1 .cor file
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class IndexRefineParams:
    """all parameters needed by index_refine(). Built from a config by params_from_config()"""
    key_material: str
    e_min: float
    e_max: float
    e_max_MR: float
    nbGrainstoFind: int
    dict_indexrefine: dict
    MatchingRate_List: list
    fit_folder: str
    mymaterials: Optional[dict] = None
    depth: float = 0
    MIN_NUMBERSPOTS_FOR_INDEXING: int = 6
    MAX_NUMBERSPOTS_FOR_INDEXING: int = 10000
    MAXNBSPOTS: Optional[int] = None
    crudeMReval: bool = True
    maxnbspots_MReval: int = 10000
    stop_Nb_Matches: int = 40
    printindexingstats: bool = False
    ignorefitfileresults: bool = True
    usepreviousUB: Any = False
    skipindexing: bool = False
    writefitfile: bool = True
    starting_grainindex: int = 0
    verboselevel: int = 0
    verbosefilename: bool = False
    useinternalmultiprocessing: bool = False
    outputlistindices: bool = False
    # None, or steps [e_max, matching tolerance angle, min nb of matches] (see ISS.parse_refinement_steps())
    refinement_steps: Optional[list] = None
    outputrefinementsteps: bool = False   # True: history of the refinement steps appended to the result
    # None, or steps [e_max, tolerance angle, min nb of matches] that each grain found must pass (see
    # verify_solution()): result reported in the info of the result (' g0 criteria pass [...]')
    solution_criteria: Optional[list] = None
    solution_max_pixdev: Optional[float] = 1.0    # max mean pixel deviation of the last step of solution_criteria
    solution_reject: bool = False                  # True: grains failing solution_criteria rejected
    previous_fit_folder: Optional[str] = None      # .fit files of a previous indexation (None: fit_folder)
    fit_acceptance: Optional[dict] = None          # {'min_nb_spots', 'max_pixdev'} of previous .fit files kept
    blacklistfile: Optional[str] = None     # former name of blacklist (used if blacklist is None)
    # spots removed before indexing: .cor/.dat/.fit file, array [[X, Y], ...] of pixel positions, or a list of them
    blacklist: Any = None
    blacklist_tolerance: float = 2.          # pixel
    CCDLabel: str = 'sCMOS'
    fit_nbfiles_per_folder: Optional[int] = None   # .fit files in subfolders of fit_folder (see fitfile_folder())
    subfolder_prefix: str = 'images_'
    LUT: Any = None
    # None: single material. Or sequence of indexing steps (see indexing_sequence.normalize_sequence())
    sequence: Optional[dict] = None
    # usepreviousUB 'neighbours': parameters (see indexing_neighbours.DEFAULT_NEIGHBOURS), map dims [fast, slow] and
    # map step sizes [fast, slow] (micron, real distances between map points)
    neighbours: Optional[dict] = None
    mapdims: Optional[tuple] = None
    stepsizes: Optional[tuple] = None

    def fitfile_folder(self, image_index: int) -> Path:
        """folder of the .fit files of an image: fit_folder, or its subfolder <subfolder_prefix><start>_<end>"""
        if self.fit_nbfiles_per_folder:
            return Path(self.fit_folder) / GT.subfolder_name(image_index, self.fit_nbfiles_per_folder,
                                                             self.subfolder_prefix)
        return Path(self.fit_folder)

    def build_LUT(self):
        """compute the angular LUT of key_material once (shared by all images)"""
        if self.mymaterials is not None and self.key_material in self.mymaterials:
            latticeparameters = self.mymaterials[self.key_material][1]
        else:
            latticeparameters = DictLT.dict_Materials[self.key_material][1]
        self.LUT = IAL.build_AnglesLUT_fromlatticeparameters(latticeparameters,
                                                             self.dict_indexrefine['nlutmax'])
        return self


def _sequence_of(cfg: dict) -> Optional[dict]:
    if cfg.get('sequence') is None:
        return None
    from LaueTools import indexing_sequence as IS
    return IS.normalize_sequence(cfg)


def params_from_config(cfg: dict, build_LUT: bool = True, **kwargs) -> IndexRefineParams:
    """build IndexRefineParams from a configuration (see load_config()).

    kwargs replace any field of IndexRefineParams, e.g. verboselevel=2
    """
    check_misplaced_keys(cfg, DEFAULT_CONFIG, 'icfg')
    ind, run = cfg['indexing'], cfg['run']
    params = IndexRefineParams(key_material=cfg['material']['key_material'],
                               mymaterials=cfg['material']['mymaterials'],
                               e_min=ind['e_min'], e_max=ind['e_max'], e_max_MR=ind['e_max_MR'],
                               nbGrainstoFind=ind['nbGrainstoFind'],
                               dict_indexrefine=copy.deepcopy(ind['dict_indexrefine']),
                               MatchingRate_List=list(ind['MatchingRate_List']),
                               fit_folder=str(cfg['output']['fit_folder']),
                               depth=ind['depth'],
                               MIN_NUMBERSPOTS_FOR_INDEXING=ind['MIN_NUMBERSPOTS_FOR_INDEXING'],
                               MAX_NUMBERSPOTS_FOR_INDEXING=ind['MAX_NUMBERSPOTS_FOR_INDEXING'],
                               MAXNBSPOTS=ind['MAXNBSPOTS'],
                               crudeMReval=ind['crudeMReval'],
                               maxnbspots_MReval=ind['maxnbspots_MReval'],
                               stop_Nb_Matches=ind['stop_Nb_Matches'],
                               refinement_steps=ind.get('refinement_steps'),
                               solution_criteria=ind.get('solution_criteria'),
                               solution_max_pixdev=ind.get('solution_max_pixdev'),
                               solution_reject=bool(ind.get('solution_reject')),
                               previous_fit_folder=(str(run['previous_fit_folder']) if run.get('previous_fit_folder')
                                                    else None),
                               fit_acceptance=run.get('fit_acceptance'),
                               blacklist=ind['blacklist'],
                               blacklist_tolerance=ind['blacklist_tolerance'],
                               ignorefitfileresults=run['ignorefitfileresults'],
                               usepreviousUB=get_previousUB(cfg),
                               skipindexing=run['skipindexing'],
                               writefitfile=run['writefitfile'],
                               starting_grainindex=run['starting_grainindex'],
                               verboselevel=run['verboselevel'],
                               CCDLabel=cfg['scan']['CCDLabel'],
                               sequence=_sequence_of(cfg),
                               neighbours=({**IN.settings(run['neighbours']), 'reference_UBs':
                                            IN.reference_matrices(run['neighbours'], cfg['ub_matrices'])}
                                           if run['usepreviousUB'] == 'neighbours' else None),
                               mapdims=tuple(cfg['scan']['mapdims']) if cfg['scan'].get('mapdims') else None,
                               stepsizes=(tuple(cfg['scan']['stepsizes']) if cfg['scan'].get('stepsizes') is not None
                                          else None),
                               **fitfiles_layout(cfg))
    params = dataclasses.replace(params, **kwargs)
    if isinstance(params.usepreviousUB, list):
        params.nbGrainstoFind = max(len(params.usepreviousUB), params.nbGrainstoFind)
    if build_LUT and params.LUT is None:
        params.build_LUT()
    return params


def index_refine(filename: Union[str, Path], params: IndexRefineParams) -> list:
    """index and refine Laue spots of a .cor file. Write .fit file(s) in params.fit_folder (or in its subfolder,
    see IndexRefineParams.fitfile_folder())

    Returns
    -------
    [image_index, [(grainindex, UB matrix), ...], [(grainindex, [nb indexed spots, matching rate %]), ...], info]
    + [list of indexed spots indices of first grain] if params.outputlistindices
    """
    filename = Path(filename)
    filecor = filename.name
    image_index = GT.getfileindex(filecor)

    if not filename.exists():
        return [image_index, [], [], 'missing .cor file']

    blacklist_info = ''
    original_corfile = None   # set if spots are removed: indices of the purged .cor file -> rows of this file
    blacklist = params.blacklist if params.blacklist is not None else params.blacklistfile
    if blacklist is not None:
        # .cor file without blacklisted spots in the (writable) folder of .fit files
        purgedfolder = params.fitfile_folder(image_index) / 'purged_corfiles'
        purged = _purge_corfile(blacklist, filename, params.blacklist_tolerance, None, purgedfolder, 'purged_', None)
        if purged is None:
            return [image_index, [], [], '.cor file is empty!']
        corfile_written, nbefore, nafter = purged
        original_corfile = filename
        filename = purgedfolder / corfile_written
        blacklist_info = f'{nbefore - nafter} blacklisted spots removed ({nafter} left). '

    dataset = ISS.spotsset()
    dataset.useinternalmultiprocessing = params.useinternalmultiprocessing
    dataset.stop_Nb_Matches = params.stop_Nb_Matches

    info = dataset.importdatafromfile(str(filename), verbose=params.verboselevel, maxnbspots=params.MAXNBSPOTS)
    if 'empty' in str(info):
        return [image_index, [], [], '.cor file is empty!']

    dataset.key_material = params.key_material
    dataset.emin = params.e_min
    dataset.emax = params.e_max
    dataset.emax_MR = params.e_max_MR
    dataset.crudeMReval = params.crudeMReval
    dataset.maxnbspots_MReval = params.maxnbspots_MReval
    dataset.printindexingstats = params.printindexingstats

    if (dataset.nbspots < params.MIN_NUMBERSPOTS_FOR_INDEXING
            or dataset.nbspots > params.MAX_NUMBERSPOTS_FOR_INDEXING):
        return [image_index, [], [], f'too few or too many spots to index ({dataset.nbspots})']

    if params.sequence is not None:   # indexing by a sequence of steps (substrate, several materials ...)
        from LaueTools import indexing_sequence as IS
        records, seqinfo, trace = IS.run_sequence(dataset, params, filecor, image_index)
        if records is None:
            return [image_index, [], [], seqinfo]
        if original_corfile is not None:
            # spot indices of the purged .cor file (without blacklisted spots) -> rows of the original .cor file
            original_rows = purged_to_original_rows(filename, original_corfile)
            for r in records:
                r['spot_indices'] = [int(original_rows[i]) for i in r['spot_indices']]
                if r['fitfile']:
                    restore_spot_indices(r['fitfile'], original_corfile, purged_corfile=filename)
        if params.verbosefilename:
            print(f'For {filecor}: {seqinfo}')
        toreturn = [image_index,
                    [(n, r['UB']) for n, r in enumerate(records)],
                    [(n, [r['nb_indexed'], r['matching_rate']]) for n, r in enumerate(records)],
                    blacklist_info + seqinfo, records, trace]
        if params.outputlistindices:
            toreturn.append(records[0]['spot_indices'] if records else [])
        return toreturn

    previousResults = None
    fitfolder = params.fitfile_folder(image_index)
    prevfolder = previous_fitfile_folder(params, image_index)
    info = blacklist_info
    nbgrains = params.nbGrainstoFind
    if not params.ignorefitfileresults:   # previous .fit files kept if accepted, else indexed again
        previous = fitfile_grains(prevfolder, filecor[:-4])
        status = fit_accepted(previous, params.fit_acceptance)
        if status == 'accepted':
            if prevfolder.resolve() != fitfolder.resolve():
                fitfolder.mkdir(parents=True, exist_ok=True)
                for g in previous:
                    shutil.copy2(g['path'], fitfolder / Path(g['path']).name)
            return [image_index, [(g['grain'], g['UB']) for g in previous],
                    [(g['grain'], [g['nb'], np.nan]) for g in previous],
                    f"previous .fit file kept ({max(g['nb'] for g in previous)} spots, "
                    f"{min(g['pixdev'] for g in previous):.2f} pixel)."]
        info += f'previous .fit file {status}: indexed again. '

    usepreviousUB = params.usepreviousUB
    if usepreviousUB == 'fromfitfile':
        previous = fitfile_grains(prevfolder, filecor[:-4])
        if previous:
            # UB matrices of the grains of the previous .fit files: refinement (and indexing from scratch if needed)
            previousResults = (len(previous), [g['UB'] for g in previous], 0, 0)
            nbgrains = max(len(previous), nbgrains) if isinstance(nbgrains, int) else nbgrains
            info += f'Tried the UB matrices of {len(previous)} previous .fit file(s). '
        else:
            info += 'Could not try UB matrix because .fit file was missing. '
    elif usepreviousUB in (False, None):
        pass
    elif usepreviousUB == 'neighbours':   # 1 image (test): from scratch. Map: see run_multiprocessing()
        info += 'usepreviousUB neighbours: single image indexed from scratch. '
    elif isinstance(usepreviousUB, (list, tuple)):
        previousResults = (len(usepreviousUB), list(usepreviousUB), 0, 0)
    else:
        raise ValueError(f"usepreviousUB={usepreviousUB} should be False, 'fromfitfile' or a list of UB matrices")

    if params.skipindexing and usepreviousUB and usepreviousUB != 'neighbours':
        dataset.inhibitindexing = True
        info += ' Check orientation and refine only (no indexing from scratch).'

    fitfolder.mkdir(parents=True, exist_ok=True)

    dataset.IndexSpotsSet(None,
                          params.key_material,
                          params.e_min,
                          dataset.emax,
                          params.dict_indexrefine,
                          None,
                          starting_grainindex=params.starting_grainindex,
                          use_file=0,
                          IMM=False,
                          n_LUT=params.dict_indexrefine['nlutmax'],
                          LUT=params.LUT,
                          angletol_list=params.dict_indexrefine['list matching tol angles'],
                          nbGrainstoFind=nbgrains,
                          previousResults=previousResults,
                          dirnameout_fitfile=fitfolder,
                          corfilename=filecor,
                          verbose=params.verboselevel,
                          MatchingRate_List=params.MatchingRate_List,
                          depth=params.depth,
                          dictmaterials=params.mymaterials,
                          writefitfile=params.writefitfile,
                          emax_matchringrate=params.e_max_MR,   # matching rate up to e_max_MR (keV)
                          refinement_steps=params.refinement_steps)

    listspotindices = dataset.getSpotsFamily(params.starting_grainindex)
    if original_corfile is not None:
        # spot indices of the purged .cor file (without blacklisted spots) -> rows of the original .cor file
        original_rows = purged_to_original_rows(filename, original_corfile)
        listspotindices = [int(original_rows[i]) for i in listspotindices]
        if params.writefitfile:
            for fitfile in fitfolder.glob(f'{filecor[:-4]}_g*.fit'):
                if fitfile_source(fitfile) == str(filename):   # written now from the purged .cor file
                    restore_spot_indices(fitfile, original_corfile, purged_corfile=filename)
    if params.verbosefilename:
        print(f'For {filecor}: grain #{params.starting_grainindex}, nb indexed spots: {len(listspotindices)}')
    info += ' Treated by IndexSpotsSet().' + indexing_pairs_info(dataset)
    rejected = []
    if params.solution_criteria:   # check of each grain found with the criteria of a correct solution
        for grain, ub in sorted(dataset.dict_grain_matrix.items()):
            if ub is not None:
                verification = verify_solution(original_corfile or filename, ub, params)
                info += ' ' + criteria_info(grain, verification)
                if params.solution_reject and not verification['passed']:
                    rejected.append(grain)
    if rejected:   # grains failing the criteria: .fit files removed, results dropped
        for grain in rejected:
            (fitfolder / f'{filecor[:-4]}_g{grain}.fit').unlink(missing_ok=True)
            dataset.dict_grain_matrix[grain] = None
            dataset.dict_grain_matching_rate[grain] = [0, 0.]
        info += f' grain(s) {rejected} rejected (criteria).'
        kept = [g for g, ub in dataset.dict_grain_matrix.items() if ub is not None]
        if not kept and previousResults is not None and not params.skipindexing:
            # UB matrices tried first (usepreviousUB) rejected: indexing from scratch
            res = index_refine(original_corfile or filename, dataclasses.replace(params, usepreviousUB=False))
            res[3] = f'{info} Indexed again from scratch: {res[3]}'
            return res

    toreturn = [image_index,
                [(key, value) for key, value in dataset.dict_grain_matrix.items()],  # UB matrices
                [(key, value) for key, value in dataset.dict_grain_matching_rate.items()],  # nb indexed, MR
                info]
    if params.outputlistindices:
        toreturn.append(listspotindices)
    if params.outputrefinementsteps:
        toreturn.append({'refinement_steps': dict(dataset.dict_grain_refinementsteps),
                         'indexing': {g: v for g, v in dataset.dict_grain_indexingpair.items() if v is not None}})
    return toreturn


def previous_fitfile_folder(params: 'IndexRefineParams', image_index: int) -> Path:
    """folder of the .fit files of a previous indexation of an image: params.previous_fit_folder (same layout as
    fit_folder), else the .fit files folder of params"""
    if not params.previous_fit_folder:
        return params.fitfile_folder(image_index)
    return dataclasses.replace(params, fit_folder=params.previous_fit_folder).fitfile_folder(image_index)


def fitfile_grains(folder: Union[str, Path], stem: str) -> List[dict]:
    """grains of the .fit files <stem>_g#.fit of a folder: [{'grain', 'UB', 'nb', 'pixdev', 'path'}] (sorted)"""
    import re
    grains = []
    for path in sorted(Path(folder).glob(f'{stem}_g*.fit')):
        m = re.match(rf'{re.escape(stem)}_g(\d+)\.fit$', path.name)
        if not m:
            continue
        nb, pixdev = None, None
        lines = path.read_text(errors='replace').splitlines()
        for k, line in enumerate(lines):
            if line.startswith('#Number of indexed spots'):
                nb = int(line.split(':')[1])
            elif line.startswith('#Mean Deviation(pixel)'):
                value = line.split(':')[1].strip() if ':' in line and line.split(':')[1].strip() else lines[k + 1]
                pixdev = float(value.lstrip('#'))
        try:
            UB = np.array(IOLT.readfitfile_multigrains(str(path))[3], dtype=float).reshape(3, 3)
        except Exception:   # unreadable file: ignored
            continue
        grains.append({'grain': int(m.group(1)), 'UB': UB, 'nb': nb or 0,
                       'pixdev': np.inf if pixdev is None else pixdev, 'path': str(path)})
    return sorted(grains, key=lambda g: g['grain'])


def fit_accepted(grains: List[dict], acceptance: Optional[dict]) -> str:
    """'accepted', 'missing' or 'rejected (...)': best grain (most indexed spots) of previous .fit files with at
    least acceptance['min_nb_spots'] spots and a mean pixel deviation at most acceptance['max_pixdev'] (None: any)"""
    if not grains:
        return 'missing'
    acceptance = acceptance or {}
    best = max(grains, key=lambda g: g['nb'])
    minnb, maxpd = acceptance.get('min_nb_spots'), acceptance.get('max_pixdev')
    if minnb is not None and best['nb'] < minnb:
        return f"rejected ({best['nb']} < {minnb} spots)"
    if maxpd is not None and best['pixdev'] > maxpd:
        return f"rejected (pixel deviation {best['pixdev']:.2f} > {maxpd})"
    return 'accepted'


def verify_solution(filename: Union[str, Path], UB, params: 'IndexRefineParams', criteria=None,
                    max_pixdev: Optional[float] = None) -> dict:
    """check a UB matrix with the criteria of a correct solution (params.solution_criteria, or criteria): steps
    [e_max, tolerance angle, min nb of matches] done in sequence from UB: at each step the spots (up to e_max) closer
    than the tolerance angle are linked, UB and strain are refined with these links, and the nb of indexed spots
    (links after refinement) must be at least min nb of matches; and the mean pixel deviation of the last step at
    most max_pixdev (or params.solution_max_pixdev; None: not checked). No .fit file written.

    Returns {'passed': bool, 'steps': [(e_max, tol, min nb, nb of indexed spots or None), ...],
             'pixdev': [mean pixel deviation of the steps (None: not done)],
             'failed_step': index of the first step failed or None, 'UB': refined UB matrix or None (failed),
             'UB_refined': UB matrix of the last step done, 'nb_spots', 'matching_rate': of the last step done}"""
    criteria = ISS.parse_refinement_steps(criteria if criteria is not None else params.solution_criteria)
    max_pixdev = max_pixdev if max_pixdev is not None else params.solution_max_pixdev
    dir_ = copy.deepcopy(params.dict_indexrefine)
    dir_['MATCHINGRATE_ANGLE_TOL'] = criteria[0][1]   # raw UB: matching of the first step
    dir_['MinimumMatchingRate'] = 1e-3                 # (the decision is made by the nb of matches of the steps)
    check = dataclasses.replace(params, usepreviousUB=[np.asarray(UB, dtype=float)], skipindexing=True,
                                nbGrainstoFind=1, refinement_steps=[list(c) for c in criteria],
                                e_max_MR=criteria[0][0], dict_indexrefine=dir_, solution_criteria=None,
                                solution_max_pixdev=None,
                                writefitfile=False, outputrefinementsteps=True, outputlistindices=False,
                                useinternalmultiprocessing=False, verbosefilename=False, neighbours=None)
    res = index_refine(filename, check)
    history = res[-1]['refinement_steps'].get(check.starting_grainindex, []) if isinstance(res[-1], dict) else []
    steps, failed = [], None
    for k, (emax, tol, minnb, _) in enumerate(criteria):
        nb = history[k]['nb_matches'] if k < len(history) else None
        steps.append((emax, tol, minnb, nb))
        if failed is None and (nb is None or nb < minnb):
            failed = k
    ubs = dict(res[1])
    pixdev = [history[k].get('pixdev') if k < len(history) else None for k in range(len(criteria))]
    if failed is None and max_pixdev is not None and (pixdev[-1] is None or pixdev[-1] > max_pixdev):
        failed = len(criteria) - 1   # last step: too large mean pixel deviation
    nbmr = dict(res[2]).get(check.starting_grainindex) or [None, None]
    return {'passed': failed is None, 'steps': steps, 'pixdev': pixdev, 'failed_step': failed,
            'UB': ubs.get(check.starting_grainindex) if failed is None else None,
            'UB_refined': ubs.get(check.starting_grainindex), 'nb_spots': nbmr[0], 'matching_rate': nbmr[1]}


def criteria_info(grain: int, verification: dict) -> str:
    """text of a verification in the info of a result, e.g. ' g0 criteria pass [18keV/0.2deg:123, ...].'
    (read by parse_criteria_info())"""
    pds = verification.get('pixdev') or [None] * len(verification['steps'])
    steps = ', '.join(f"{e:g}keV/{t:g}deg:{'-' if nb is None else nb}" + ('' if pd is None else f'/{pd:.2f}px')
                      for (e, t, _, nb), pd in zip(verification['steps'], pds))
    status = 'pass' if verification['passed'] else f"fail@{verification['failed_step']}"
    return f'g{grain} criteria {status} [{steps}].'


def parse_criteria_info(info: str) -> dict:
    """{grain: {'passed': bool, 'nb': [nb of indexed spots of the steps (None: not done)],
    'pixdev': [mean pixel deviations of the steps (None: not done)]}} from the info of a result (criteria_info())"""
    import re
    out = {}
    for grain, status, steps in re.findall(r'g(\d+) criteria (pass|fail@\d+) \[([^\]]*)\]', str(info)):
        vals = [v.split(':')[-1].split('/') for v in steps.split(', ') if v]
        nbs = [None if v[0] == '-' else int(v[0]) for v in vals]
        pds = [float(v[1][:-2]) if len(v) > 1 else None for v in vals]
        out[int(grain)] = {'passed': status == 'pass', 'nb': nbs, 'pixdev': pds}
    return out


def indexing_pairs_info(dataset) -> str:
    """text for the info of a result: spots pair (set A, set B) of the best solution of the indexing from scratch
    of each indexed grain, e.g. ' g0 from spots pair A=0 B=7.' (read by central_spots_statistics())"""
    text = ''
    for grain in sorted(dataset.indexedgrains):
        entry = dataset.dict_grain_indexingpair.get(grain)
        if entry and entry['solutions'] and dataset.dict_grain_matrix.get(grain) is not None:
            best = entry['solutions'][0]
            text += f" g{grain} from spots pair A={best['spot_A']} B={best['spot_B']}."
    return text


def central_spots_statistics(allresults: list, verbose: bool = True) -> dict:
    """statistics over the results of a batch of the central spot (set A) and partner spot (set B) of the pair of
    spots that gave each indexed grain (indexing from scratch), to choose the size of the spots sets

    Spot indices are positions in the list of spots not yet indexed (as in 'central spots indices')
    Returns {'A': {spot index: nb of grains}, 'B': {spot index: nb of grains}, 'nb_grains': n}
    """
    import re
    import collections
    countA, countB, nb = collections.Counter(), collections.Counter(), 0
    for res in allresults:
        for _, a, b in re.findall(r'g(\d+) from spots pair A=(\d+) B=(\d+)', str(res[3]) if len(res) > 3 else ''):
            countA[int(a)] += 1
            countB[int(b)] += 1
            nb += 1
    if verbose and nb:
        print(f'{nb} grains indexed from scratch')
        for name, count in (('A (central spot)', countA), ('B (partner spot)', countB)):
            cumul, lines = 0, []
            for spot in sorted(count):
                cumul += count[spot]
                lines.append(f'{spot}: {count[spot]} ({100. * cumul / nb:.0f} %)')
            print(f'spot of set {name}: nb of grains (cumulated %)\n   ' + ', '.join(lines))
        print(f"largest index used: set A {max(countA)}, set B {max(countB)}")
    return {'A': dict(sorted(countA.items())), 'B': dict(sorted(countB.items())), 'nb_grains': nb}


def refinement_ramp(e_start: float, e_end: float, nb_steps: int = 4, tol_angles=(0.5, 0.1),
                    min_nb_matches=(8, 15)) -> list:
    """refinement steps [e_max, matching tolerance angle, min nb of matches] with e_max increasing linearly from
    e_start (e.g. e_max_MR, energy used for indexing) to e_end (full energy range), tolerance angle from
    tol_angles[0] to tol_angles[1] (a single value: constant) and min nb of matches from min_nb_matches[0] to [1]

    e.g. IB.refinement_ramp(18, 27, 4, (0.3, 0.1), (10, 20))
         -> [[18.0, 0.3, 10], [21.0, 0.233, 13], [24.0, 0.167, 17], [27.0, 0.1, 20]]
    Use: indexing: refinement_steps of the configuration, or params.refinement_steps
    """
    tols = np.atleast_1d(np.asarray(tol_angles, dtype=float))
    mins = np.atleast_1d(np.asarray(min_nb_matches, dtype=float))
    energies = np.linspace(e_start, e_end, nb_steps)
    tolerances = np.linspace(tols[0], tols[-1], nb_steps)
    minima = np.linspace(mins[0], mins[-1], nb_steps)
    return [[round(float(e), 2), round(float(t), 3), int(round(m))] for e, t, m in zip(energies, tolerances, minima)]


def print_refinement(history, grains=None):
    """print how each grain was indexed and refined: spots pair of the best solution of the indexing from scratch
    (and best solution of each central spot probed, proposed spots set A), then the refinement steps

    history: result of index_refine() with params.outputrefinementsteps=True (or its last element), or a result
    of index_refine() with a sequence (kept grains)
    """
    indexing = {}
    if isinstance(history, list):   # result of index_refine()
        last = history[-1] if history else None
        if isinstance(last, dict) and 'refinement_steps' in last:
            history, indexing = last['refinement_steps'], last.get('indexing', {})
        else:   # sequence: records of the kept grains
            records = history[4] if len(history) > 4 and isinstance(history[4], list) else []
            records = [r for r in records if isinstance(r, dict)]
            label = lambda r: f"{r['step']} {r['key_material']} g{r['grain']}"
            history = {label(r): r.get('refinement', []) for r in records}
            indexing = {label(r): r['indexing_pair'] for r in records if r.get('indexing_pair')}
    elif isinstance(history, dict) and 'refinement_steps' in history:
        history, indexing = history['refinement_steps'], history.get('indexing', {})
    for grain, steps in history.items():
        if grains is not None and grain not in grains:
            continue
        entry = indexing.get(grain)
        if entry:
            prop = entry.get('spots_proposal')
            if prop:
                print(f"grain {grain}: spots set A proposed by '{prop['method']}': {prop['spots']}"
                      + (' (no proposal: most intense spots used)' if prop['fallback'] else ''))
                for method, scores in prop['scores'].items():
                    print(f'   {method}: spot (score) ' + ', '.join(f'{sp} ({sc})' for sp, sc in scores))
            print(f"grain {grain}: indexing from scratch, best solution of each central spot probed "
                  f"(set A {_compact_indices(entry['set_A'])}, set B "
                  + (f"0 to {entry['maxindex_probed']}" if entry['set_B'] is None
                     else str(_compact_indices(entry['set_B']))) + ')')
            print('   spot A  spot B  (.cor rows)     hkl A        hkl B      nb matches  matching rate(%)')
            for k, sol in enumerate(entry['solutions']):
                line = (f"   {sol['spot_A']:6d}  {sol['spot_B']:6d}  ({sol['row_A']:4d},{sol['row_B']:4d})  "
                        f"{str(sol['hkl_A']):>11s}  {str(sol['hkl_B']):>11s}  {sol['nb_matches']:10d}  "
                        f"{sol['matching_rate']:16.1f}")
                (GT.printgreen if k == 0 else print)(line + ('   <- used' if k == 0 else ''))
        print(f'grain {grain}: refinement steps')
        print('   step  e_max(keV)  tol(deg)  nb matches (min)  matching rate(%)  pixdev')
        for st in steps:
            nb = '-' if st['nb_matches'] is None else str(st['nb_matches'])
            mr = '-' if st['matching_rate'] is None else f"{st['matching_rate']:.1f}"
            pix = '-' if st['pixdev'] is None else f"{st['pixdev']:.3f}"
            line = (f"   {st['step']:4d}  {st['emax']:10.2f}  {st['tol_angle']:8.3f}  {nb:>10s} ({st['min_nb_matches']:3d})"
                    f"  {mr:>16s}  {pix:>6s}")
            (GT.printgreen if st['accepted'] else GT.printred)(line + ('' if st['accepted'] else '   rejected'))


def print_result(res: list, minnbindexed: int = 20, minMR: float = 33):
    """print (in color) the results of index_refine() for 1 image"""
    print(f'image index {res[0]}')
    if not res[1]:
        GT.printred('... Nothing found ...')
    for (gindex, ub), (_, score) in zip(res[1], res[2]):
        if ub is None or score is None:
            GT.printred(f'grain {gindex}: nothing found')
            continue
        nir, mr = score
        msg = f'grain {gindex}: Nb indexed Reflections: {nir}, Matching Rate {mr:.2f}%'
        if nir > minnbindexed and mr > minMR:
            GT.printgreen(msg)
        else:
            GT.printred(msg + '. These (poor?) results may be not reliable')
        print('UB matrix found: ' + str(np.array(ub).tolist()) + '\n')
    print('Info: ', res[3])


# --------------------------------------------------------------------------------------
#  SET OF FILES & MULTIPROCESSING
# --------------------------------------------------------------------------------------
def corfile_path(cfg: dict, imageindex: int) -> Path:
    """full path of .cor file of a given image index (in its subfolder if scan: nbfiles_per_folder)"""
    scan = cfg['scan']
    folder = Path(scan['cor_folder'])
    if scan['nbfiles_per_folder']:
        folder = folder / GT.subfolder_name(imageindex, scan['nbfiles_per_folder'], scan['subfolder_prefix'])
    return folder / f"{scan['prefix']}{imageindex:0{scan['nbdigits']}d}.cor"


def fitfiles_layout(cfg: dict) -> dict:
    """{'fit_nbfiles_per_folder': n or None, 'subfolder_prefix': str}: layout of .fit files in fit_folder

    readers of .fit files accept the same layout, e.g.
        lay = IB.fitfiles_layout(cfg)
        parsed_fitfileseries(..., nbfiles_per_folder=lay['fit_nbfiles_per_folder'], subfolder_prefix=lay['subfolder_prefix'])
    """
    scan = cfg['scan']
    return {'fit_nbfiles_per_folder': scan['nbfiles_per_folder'] if cfg['output']['fit_subfolders'] else None,
            'subfolder_prefix': scan['subfolder_prefix']}


def fitfile_path(cfg: dict, imageindex: int, grainindex: int = 0, fit_folder: Optional[Union[str, Path]] = None) -> Path:
    """full path of the .fit file of a given image and grain index (fit_folder: default output: fit_folder)"""
    scan, lay = cfg['scan'], fitfiles_layout(cfg)
    folder = Path(fit_folder or cfg['output']['fit_folder'])
    if lay['fit_nbfiles_per_folder']:
        folder = folder / GT.subfolder_name(imageindex, lay['fit_nbfiles_per_folder'], lay['subfolder_prefix'])
    return folder / f"{scan['prefix']}{imageindex:0{scan['nbdigits']}d}_g{grainindex}.fit"


def _corfiles(cfg: dict) -> List[Path]:
    """prefix####.cor files of cor_folder (or of its subfolders), sorted by image index"""
    scan = cfg['scan']
    pattern = f"{scan['prefix']}*.cor"
    if scan['nbfiles_per_folder']:
        pattern = f"{scan['subfolder_prefix']}*/{pattern}"
    files = Path(scan['cor_folder']).glob(pattern)
    # keep only prefix####.cor (exclude e.g. purged or merged files)
    files = [ff for ff in files if ff.stem[len(scan['prefix']):].isdigit()]
    return sorted(files, key=lambda p: GT.getfileindex(p))


def select_corfiles(cfg: dict, image_range=None, roi=None) -> List[str]:
    """list of .cor files to analyse (sorted by image index)

    image_range: [start, stop] (stop included), default cfg['selection']['image_range']
    roi: [center image index, [half size fast, half size slow]] rectangle in the map,
         default cfg['selection']['roi']
    None for both: all .cor files of cor_folder (or of its subfolders) with the prefix of the configuration
    """
    scan = cfg['scan']
    if image_range is None:
        image_range = cfg['selection']['image_range']
    if roi is None:
        roi = cfg['selection']['roi']

    files = _corfiles(cfg)

    if image_range is not None:
        start, stop = image_range
        files = [ff for ff in files if start <= GT.getfileindex(ff) <= stop]
    if roi is not None:
        center, halfsizes = roi
        indices = GT.extract2Dslice(center, halfsizes,
                                    np.arange(scan['mapdims'][0] * scan['mapdims'][1]).reshape(scan['invmapdims']))
        indices = set(np.ravel(indices).tolist())
        files = [ff for ff in files if GT.getfileindex(ff) in indices]
    if cfg['selection'].get('image_indices') is not None:
        wanted = {int(i) for i in cfg['selection']['image_indices']}
        files = [ff for ff in files if GT.getfileindex(ff) in wanted]
    return [str(ff) for ff in files]


_WORKER_PARAMS = None


def _init_worker(params, internal_ncpus=None):
    """internal_ncpus: nb of processes of the pools of matching rates of this worker (internal multiprocessing)"""
    global _WORKER_PARAMS
    _WORKER_PARAMS = params
    IAL.set_internal_ncpus(internal_ncpus)


def _index_refine_worker(filename):
    return index_refine(filename, _WORKER_PARAMS)


def mp_context():
    """multiprocessing context of the pools of the batches (forkserver in a jupyter kernel, see GT.mp_context())"""
    return GT.mp_context(preload=('LaueTools.indexing_batch', 'LaueTools.peaksearch_batch'))


def available_cpus() -> int:
    """nb of cpus this process may use (cpus allocated by slurm or jupyter-slurm, cpu quota of a VISA instance or
    of a container, not all cpus of the machine; LAUETOOLS_NB_CPUS): see GT.available_cpus()"""
    return GT.available_cpus()


# min nb of cpus per image to compute the matching rates of each image with its own pool (internal multiprocessing):
# few images (e.g. seeds) for many cpus. Measured (NiMnGa seed indexed from scratch): 115 s on 1 cpu, 87 s with 2,
# 58 s with 4
MIN_CPUS_INTERNAL = 2


def run_multiprocessing(listfiles: List[str], params: IndexRefineParams, nb_cpus: Optional[int] = None,
                        progress_interval: float = 0.1) -> list:
    """index & refine all .cor files of listfiles with min(nb_cpus, nb of files) processes (1 image per process).
    With at least MIN_CPUS_INTERNAL cpus per image (few images, e.g. seeds), the matching rates of the candidate UB
    matrices of each image are computed by a pool of nb_cpus // nb of files processes (internal multiprocessing,
    at most IAL.INTERNAL_MAX_CPUS); otherwise one process per image only.

    nb_cpus: None: all cpus available for this process (see available_cpus())
    progress_interval: min time (s) between progress bar updates (large value for log files of batch jobs)

    params.usepreviousUB 'neighbours': indexing guided by the UB matrices of the neighbouring map points
    (listfiles: images of the map), see indexing_neighbours.run_neighbours()

    Returns the list of results of index_refine() (unordered)
    """
    if nb_cpus is None:
        nb_cpus = available_cpus()
    if nb_cpus > available_cpus():
        GT.printyellow(f'{nb_cpus} processes for {available_cpus()} available cpus: '
                       'processes share cpus (no speed gain)')
    if params.usepreviousUB == 'neighbours':
        return IN.run_neighbours(listfiles, params, nb_cpus=nb_cpus, progress_interval=progress_interval)
    nproc = max(1, min(nb_cpus, len(listfiles)))
    cpus_per_image = min(nb_cpus // nproc, IAL.INTERNAL_MAX_CPUS)
    internal = cpus_per_image >= MIN_CPUS_INTERNAL
    params = dataclasses.replace(params, useinternalmultiprocessing=internal, outputlistindices=False)
    if params.LUT is None:
        params.build_LUT()

    t0 = time.time()
    allresults = []
    # parameters (and LUT) are sent once to each worker process
    if internal:
        print(f'{len(listfiles)} images for {nb_cpus} cpus: {nproc} processes, matching rates of each image '
              f'computed with {cpus_per_image} processes')
        # workers of a ProcessPoolExecutor are not daemonic: they may start their own pools (a multiprocessing.Pool
        # worker may not). Also with 1 image: the pools of matching rates are not forked from a jupyter kernel
        with ProcessPoolExecutor(max_workers=nproc, mp_context=mp_context(), initializer=_init_worker,
                                 initargs=(params, cpus_per_image)) as executor:
            futures = [executor.submit(_index_refine_worker, f) for f in listfiles]
            for future in tqdm(as_completed(futures), total=len(futures), mininterval=progress_interval):
                allresults.append(future.result())
    else:
        with mp_context().Pool(processes=nproc, initializer=_init_worker, initargs=(params,)) as pool:
            for res in tqdm(pool.imap_unordered(_index_refine_worker, listfiles, chunksize=1),
                            total=len(listfiles), mininterval=progress_interval):
                allresults.append(res)
    dt = time.time() - t0
    print(f"It took {int(dt // 60)} min {int(dt % 60)} s to index {len(listfiles)} images with {nb_cpus} cpus.")
    return allresults


def _verify_worker(task):
    filename, name, UB, criteria, max_pixdev = task
    return name, GT.getfileindex(Path(filename).name), verify_solution(filename, UB, _WORKER_PARAMS, criteria,
                                                                        max_pixdev)


def test_ub_matrices(listfiles: List[str], cfg: dict, names=None, criteria=None, max_pixdev: Optional[float] = None,
                     nb_cpus: Optional[int] = None) -> dict:
    """check each UB matrix of cfg['ub_matrices'] (names: those names, default all) on each .cor file of listfiles
    (e.g. select_corfiles(seeds config), or of the whole map) with the criteria of a correct solution (see
    verify_solution(); default: indexing: solution_criteria of cfg, otherwise one step [e_max_MR,
    MATCHINGRATE_ANGLE_TOL, MinimumNumberMatches]). One (image, matrix) pair per task, no .fit file written.

    Prints the table images x matrices: nb of indexed spots / mean pixel deviation of the last step reached
    (* criteria passed)
    Returns {name: {image index: verification of verify_solution()}}
    """
    names = list(cfg['ub_matrices'] or {}) if names is None else ([names] if isinstance(names, str) else list(names))
    missing = [n for n in names if n not in (cfg['ub_matrices'] or {})]
    if missing or not names:
        raise KeyError(f'UB matrices {missing} not found in ub_matrices' if missing else 'ub_matrices is empty')
    params = params_from_config(cfg, usepreviousUB=False, neighbours=None)
    if criteria is None:
        criteria = params.solution_criteria or [[params.e_max_MR, params.dict_indexrefine['MATCHINGRATE_ANGLE_TOL'],
                                                 params.dict_indexrefine['MinimumNumberMatches']]]
    tasks = [(str(f), n, np.asarray(cfg['ub_matrices'][n], dtype=float), criteria, max_pixdev)
             for f in listfiles for n in names]
    nb_cpus = min(nb_cpus or available_cpus(), len(tasks))
    out = {n: {} for n in names}
    with mp_context().Pool(processes=nb_cpus, initializer=_init_worker, initargs=(params,)) as pool:
        for name, index, verif in tqdm(pool.imap_unordered(_verify_worker, tasks, chunksize=1), total=len(tasks)):
            out[name][index] = verif
    indices = sorted({i for v in out.values() for i in v})
    maxpd = max_pixdev if max_pixdev is not None else params.solution_max_pixdev
    print(f'criteria {criteria}, max mean pixel deviation {maxpd}')
    print('image   ' + ''.join(f'{n:>15}' for n in names))
    for i in indices:
        cells = []
        for n in names:
            v = out[n][i]
            done = [(st[3], pd) for st, pd in zip(v['steps'], v['pixdev']) if st[3] is not None]
            nb, pd = done[-1] if done else (0, None)
            cell = f"{nb}/{pd:.2f}px" if pd is not None else f'{nb}'
            cells.append(f"{cell:>14}{'*' if v['passed'] else ' '}")
        print(f'{i:<8}' + ''.join(cells))
    print('passed: ' + ', '.join(f"{n} {sum(v['passed'] for v in out[n].values())}/{len(indices)}" for n in names))
    return out


# --------------------------------------------------------------------------------------
#  REFINE AGAIN EACH GRAIN (cluster of orientations) WITH ITS OWN UB MATRICES
# --------------------------------------------------------------------------------------
def grain_refinement_tasks(result, analyzer, cluster_ids: Optional[List[int]] = None, min_size: int = 30,
                           region: str = 'points', dilate: int = 0, box_halfsize=10, start: str = 'local',
                           align: bool = True, envelope_radius: float = 2.0,
                           verbose: bool = True) -> Dict[int, Dict[int, list]]:
    """images and starting UB matrices to refine again each grain (cluster of orientations) of a segmentation

    result, analyzer: from orientationclustering.analyze_ub_matrices() (notebook 3_segmentation_grains)
    cluster_ids: clusters to refine (default: clusters with at least min_size matrices, largest first)
    region: 'points': map points where the cluster was found
            'envelope': + points inside the envelope of the cluster (see OC.cluster_envelope(), closing
                        with envelope_radius): holes where the grain was missed or not the primary grain
            'box': all map points of a box centred on the cluster barycenter, of half size box_halfsize
                   (int, or (half size fast axis, half size slow axis)) in map steps
    dilate: + map points within `dilate` map steps of the region (grain borders)
    start: 'local': UB matrix of the cluster at this point (or at the nearest point of the cluster),
                    then the cluster reference matrix if the first one does not match
           'reference': cluster reference matrix only (same starting UB for all points)
    align: local UB matrices brought to the symmetry variant of the cluster reference (UB.S, see
           OC.align_to_cluster_reference()): the same reflection has the same hkl at all points of a grain
           (needed to select a fixed set of reflections, see grain_spots_selection())

    Returns {cluster_id: {image_index: [candidate UB matrices (3x3 arrays), tested in this order]}}
    """
    from LaueTools import orientationclustering as OC
    from scipy import ndimage as ndi

    if start not in ('local', 'reference'):
        raise ValueError("start must be 'local' or 'reference'")
    if region not in ('points', 'envelope', 'box'):
        raise ValueError("region must be 'points', 'envelope' or 'box'")
    nrows, ncols = analyzer.check_mapdimension(result.mapdimension)
    stats = result.get_cluster_stats_dict(analyzer, (nrows, ncols))
    if cluster_ids is None:
        cluster_ids = [c.cluster_id for c in result.clusters if c.size >= min_size]
    matrices = (OC.align_to_cluster_reference(result, analyzer, (nrows, ncols)) if align
                else np.array([m.matrix for m in analyzer.matrices]).reshape(-1, 3, 3))

    tasks = {}
    for cid in cluster_ids:
        cluster = result.get_cluster(cid)
        if cluster is None:
            raise ValueError(f'cluster {cid} not found')
        reference = np.asarray(stats[cid].reference_matrix, dtype=float)

        # UB matrix of the cluster at each of its points (most indexed spots if several)
        local = {}
        for i in cluster.matrix_indices:
            img = int(analyzer.image_indices[i])
            if img not in local or analyzer.nb_indexed[i] > analyzer.nb_indexed[local[img]]:
                local[img] = i
        member_images = np.array(sorted(local))
        member_rows, member_cols = np.divmod(member_images, ncols)

        mask = OC.cluster_mask(result, analyzer, cid, (nrows, ncols))
        if region == 'envelope':
            mask |= OC.cluster_envelope(result, analyzer, cid, radius=envelope_radius,
                                        mapdimension=(nrows, ncols))['mask']
        elif region == 'box':
            half_fast, half_slow = (box_halfsize, box_halfsize) if np.ndim(box_halfsize) == 0 else box_halfsize
            row0, col0 = (int(round(v)) for v in stats[cid].mean_position)
            mask[max(row0 - half_slow, 0):row0 + half_slow + 1, max(col0 - half_fast, 0):col0 + half_fast + 1] = True
        if dilate:
            mask = ndi.binary_dilation(mask, structure=OC._disk(dilate))

        tasks[cid] = {}
        for row, col in zip(*np.nonzero(mask)):
            img = int(row * ncols + col)
            if start == 'reference':
                tasks[cid][img] = [reference]
                continue
            if img in local:
                i = local[img]
            else:   # nearest point of the cluster
                i = local[int(member_images[np.argmin((member_rows - row) ** 2 + (member_cols - col) ** 2)])]
            UB = np.asarray(matrices[i], dtype=float)
            tasks[cid][img] = [UB] if np.allclose(UB, reference) else [UB, reference]
        if verbose:
            print(f'cluster {cid:3d}: {cluster.nb_pixels:5d} points -> {len(tasks[cid]):5d} images to refine')
    return tasks


def grain_folder(output_folder: Union[str, Path], cluster_id: int, folder_prefix: str = 'grain_') -> Path:
    """folder of the .fit files of a grain refined by refine_grains(): <output_folder>/grain_003"""
    return Path(output_folder) / f'{folder_prefix}{cluster_id:03d}'


def _grain_task_list(cfg: dict, tasks: Dict[int, Dict[int, list]], output_folder: Union[str, Path],
                     folder_prefix: str = 'grain_', cor_folders: Optional[Dict[int, Union[str, Path]]] = None) -> list:
    """[(cluster_id, .cor file, candidate UBs, .fit folder), ...] (1 item = 1 image of 1 grain)"""
    cor_folders = cor_folders or {}
    alltasks = []
    for cid, images in tasks.items():
        fitdir = grain_folder(output_folder, cid, folder_prefix)
        for img, UBs in images.items():
            corfile = corfile_path(cfg, img)
            if cid in cor_folders:
                corfile = Path(cor_folders[cid]) / corfile.name
            alltasks.append((cid, str(corfile), [np.asarray(UB, dtype=float) for UB in UBs], str(fitdir)))
    return alltasks


def _grain_params(params: IndexRefineParams) -> IndexRefineParams:
    """parameters of refine_grains(): check orientation and refine only (1 grain per .fit file)"""
    params = dataclasses.replace(params, skipindexing=True, nbGrainstoFind=1, starting_grainindex=0,
                                 ignorefitfileresults=True, writefitfile=True, useinternalmultiprocessing=False,
                                 outputlistindices=False, verbosefilename=False)
    if params.LUT is None:
        params.build_LUT()
    return params


def _write_grain_info(cfg: dict, tasks: dict, params: IndexRefineParams, output_folder: Path, folder_prefix: str,
                      result=None, cor_folders=None, extra: Optional[dict] = None):
    """settings of a grain refinement in <output_folder>/grain_refinement.yaml"""
    info = {'date': f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}',
            'configfile': str(cfg['configfile']),
            'cor_folder': str(cfg['scan']['cor_folder']),
            'segmentation': None if result is None else {'threshold': result.threshold, 'mode': result.mode,
                                                         'symmetry': result.symmetry
                                                         if isinstance(result.symmetry, (str, type(None)))
                                                         else 'custom'},
            'parameters': {field: getattr(params, field) for field in
                           ('key_material', 'e_min', 'e_max', 'e_max_MR', 'depth', 'MAXNBSPOTS',
                            'MatchingRate_List', 'dict_indexrefine')},
            'grains': {cid: {'folder': grain_folder(output_folder, cid, folder_prefix).name,
                             'nb_images': len(images),
                             'cor_folder': None if not cor_folders or cid not in cor_folders else str(cor_folders[cid]),
                             'starting_matrices': sorted({len(UBs) for UBs in images.values()})}
                       for cid, images in tasks.items()}}
    if extra:
        info.update(extra)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / 'grain_refinement.yaml', 'w') as f:
        f.write(yaml.safe_dump(_to_plain(info), sort_keys=False, default_flow_style=None, width=200))


def _refine_task_worker(task):
    cluster_id, filename, UBs, fit_folder = task
    params = dataclasses.replace(_WORKER_PARAMS, usepreviousUB=list(UBs), fit_folder=str(fit_folder))
    return cluster_id, index_refine(filename, params)


def run_grain_tasks(alltasks: list, params: IndexRefineParams, nb_cpus: Optional[int] = None,
                    progress_interval: float = 0.1) -> Dict[int, list]:
    """run the tasks of _grain_task_list() with a pool of nb_cpus processes. Returns {cluster_id: [results]}"""
    if nb_cpus is None:
        nb_cpus = available_cpus()
    params = _grain_params(params)
    t0 = time.time()
    results = {}
    for cid, *_ in alltasks:
        results.setdefault(cid, [])
    with mp_context().Pool(processes=nb_cpus, initializer=_init_worker, initargs=(params,)) as pool:
        for cid, res in tqdm(pool.imap_unordered(_refine_task_worker, alltasks, chunksize=1),
                             total=len(alltasks), mininterval=progress_interval):
            results[cid].append(res)
    dt = time.time() - t0
    print(f'It took {int(dt // 60)} min {int(dt % 60)} s to refine {len(results)} grains '
          f'({len(alltasks)} images) with {nb_cpus} cpus.', flush=True)
    return results


def refine_grains(cfg: dict, tasks: Dict[int, Dict[int, list]], params: IndexRefineParams,
                  output_folder: Union[str, Path], nb_cpus: Optional[int] = None, folder_prefix: str = 'grain_',
                  cor_folders: Optional[Dict[int, Union[str, Path]]] = None, result=None,
                  progress_interval: float = 0.1) -> Dict[int, list]:
    """refine again each grain at its images with its own UB matrices (check orientation and refine only,
    no indexing from scratch), in this jupyter session. Same work on the cluster: prepare_slurm_grain_job()

    tasks: from grain_refinement_tasks() (or grain_spots_selection())
    params: refinement parameters, e.g. params_from_config(cfg) then changed: e_max (nb of spots of the model),
            dict_indexrefine['list matching tol angles'], dict_indexrefine['MinimumMatchingRate'] (min matching
            rate % to accept a starting matrix), MatchingRate_List, MAXNBSPOTS ...
    output_folder: .fit files of grain #cluster_id in <output_folder>/grain_<cluster_id>/<prefix>####_g0.fit
            (1 grain per .fit file: each grain is refined independently of the other grains of the image)
    cor_folders: {cluster_id: folder of .cor files} replacing the cor_folder of cfg for these grains
            (e.g. .cor files restricted to selected spots, see grain_spots_selection())
    result: ClusterResult of the segmentation (optional, information written in grain_refinement.yaml)

    Returns {cluster_id: list of results of index_refine()}. Settings are saved in <output_folder>/grain_refinement.yaml
    """
    output_folder = Path(output_folder).expanduser()
    params = _grain_params(params)
    _write_grain_info(cfg, tasks, params, output_folder, folder_prefix, result=result, cor_folders=cor_folders)
    alltasks = _grain_task_list(cfg, tasks, output_folder, folder_prefix, cor_folders)
    results = run_grain_tasks(alltasks, params, nb_cpus=nb_cpus, progress_interval=progress_interval)
    print_grain_refinement(results)
    return results


def prepare_slurm_grain_job(cfg: dict, tasks: Dict[int, Dict[int, list]], params: IndexRefineParams,
                            output_folder: Union[str, Path], folder_prefix: str = 'grain_',
                            cor_folders: Optional[Dict[int, Union[str, Path]]] = None, result=None,
                            machine: str = 'magnifix', nb_cpus: int = 192, time: str = '01:00:00',
                            mem_per_cpu: str = '2000M', nchunks: int = 1, job_name: Optional[str] = None,
                            jobs_folder: Optional[Union[str, Path]] = None, env_setup: Optional[List[str]] = None,
                            python: Optional[str] = None, mail_user: Optional[str] = None) -> dict:
    """write a slurm job folder doing refine_grains() on the cluster (same arguments + slurm resources as
    prepare_slurm_job()). Submit with submit_slurm_job(job), follow with slurm_job_status(job), and read the
    results with load_slurm_grain_results(job)

    The tasks (image, .cor file, starting UBs, .fit folder of each grain) are saved in job_params.pickle and
    shared among nchunks jobs (job array) of nb_cpus processes.
    """
    output_folder = Path(output_folder).expanduser().resolve()
    params_job = _grain_params(params)
    alltasks = _grain_task_list(cfg, tasks, output_folder, folder_prefix, cor_folders)
    nb_cpus, mem_per_cpu, nchunks = _check_slurm_resources(machine, nb_cpus, mem_per_cpu, nchunks, len(alltasks))
    job_name = job_name or f'grains_{params.key_material}'
    jobdir = _new_jobdir(Path(jobs_folder) if jobs_folder else output_folder.parent / 'slurm_jobs', job_name)

    _write_grain_info(cfg, tasks, params_job, output_folder, folder_prefix, result=result, cor_folders=cor_folders,
                      extra={'slurm_job': str(jobdir)})
    with open(jobdir / 'job_params.pickle', 'wb') as f:
        pickle.dump({'cfg': cfg, 'params': params_job, 'grain_tasks': alltasks}, f)
    with open(jobdir / 'job_params.txt', 'w') as f:
        f.write(f"# grain refinement job {jobdir.name} (config {cfg['configfile']})\n"
                f"# {len(tasks)} grains, {len(alltasks)} images, .fit files in {output_folder}/{folder_prefix}###\n"
                f"# settings: {output_folder / 'grain_refinement.yaml'}\n")

    script = _write_slurm_script(jobdir, 'LaueTools.indexing_batch', job_name, machine, nb_cpus, time, mem_per_cpu,
                                 nchunks, env_setup=env_setup, python=python, mail_user=mail_user)
    job = {'jobdir': str(jobdir), 'script': str(script), 'command': f'sbatch {script}', 'machine': machine,
           'nb_cpus': nb_cpus, 'nchunks': nchunks, 'nfiles': len(alltasks), 'job_id': None,
           'fit_folder': str(output_folder), 'kind': 'grains'}
    _write_job(job)
    print(f"job folder: {jobdir}\n{len(tasks)} grains, {len(alltasks)} images, {nchunks} job(s) of {nb_cpus} cpus "
          f"on {machine}, time limit {time}\n.fit files -> {output_folder}/{folder_prefix}###")
    print(f"to submit from a terminal (jupyter-slurm):\n    {job['command']}")
    return job


def load_slurm_grain_results(job: Union[dict, str, Path]) -> Optional[Dict[int, list]]:
    """{cluster_id: [results]} of a job written by prepare_slurm_grain_job() (all chunks). None if no result yet"""
    pairs = load_slurm_results(job)
    if pairs is None:
        return None
    results = {}
    for cid, res in pairs:
        results.setdefault(cid, []).append(res)
    print_grain_refinement(results)
    return results


def print_grain_refinement(results: Dict[int, list]):
    """nb of images where each grain was refined, mean nb of indexed spots and matching rate"""
    print(f"{'grain':>6} {'images':>7} {'refined':>8} {'mean nb spots':>14} {'mean MR %':>10}")
    for cid, allres in sorted(results.items()):
        nbs = [res[2][0][1] for res in allres
               if len(res[1]) and res[1][0][1] is not None and len(res[2]) and res[2][0][1] is not None]
        nbs = np.array(nbs, dtype=float).reshape(-1, 2)
        mean = nbs.mean(axis=0) if len(nbs) else [np.nan, np.nan]
        print(f'{cid:6d} {len(allres):7d} {len(nbs):8d} {mean[0]:14.1f} {mean[1]:10.1f}')


# --------------------------------------------------------------------------------------
#  FIXED SET OF RELIABLE SPOTS PER GRAIN (same reflections at all points -> comparable strain)
# --------------------------------------------------------------------------------------
def read_fitfile_spots(fitfile: Union[str, Path]):
    """indexed spots of a 1-grain .fit file as a pandas DataFrame (columns of the .fit file: spot_index,
    Intensity, h, k, l, pixDev, Xexp, Yexp, peak_fwaxmaj, peak_fwaxmin, Xdev, Ydev ...) and its UB matrix"""
    import pandas as pd
    with open(fitfile, 'r') as f:
        lines = f.read().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('##spot_index'))
    columns = lines[header].lstrip('#').split()
    rows = []
    for line in lines[header + 1:]:
        if line.startswith('#') or not line.strip():
            break
        rows.append([float(v) for v in line.split()])
    ub_line = next(i for i, line in enumerate(lines) if line.startswith(('#UB matrix', 'UB matrix')))
    UB = np.array([[float(v) for v in lines[ub_line + k].lstrip('#').replace('[', ' ').replace(']', ' ').split()]
                   for k in (1, 2, 3)])
    return pd.DataFrame(rows, columns=columns[:len(rows[0])] if rows else columns), UB


def read_grain_spots(cfg: dict, grain_dir: Union[str, Path], grainindex: int = 0, isolation: bool = True):
    """indexed spots of all .fit files of a grain folder (written by refine_grains())

    isolation: add column 'isolation' = distance (pixels) to the nearest other spot of the .cor file
               (spots of other grains, spurious spots...), read from the .cor file of each image

    Returns (spots DataFrame with columns image, h, k, l, Xexp, Yexp, Intensity, peak_fwaxmaj, peak_fwaxmin,
    elongation, Xdev, Ydev, dev, isolation ..., {image index: UB matrix})
    """
    import pandas as pd
    from scipy.spatial import cKDTree
    grain_dir = Path(grain_dir)
    fitfiles = sorted(grain_dir.rglob(f"{cfg['scan']['prefix']}*_g{grainindex}.fit"))
    if not fitfiles:
        raise FileNotFoundError(f"no {cfg['scan']['prefix']}*_g{grainindex}.fit file in {grain_dir}")
    tables, UBs = [], {}
    for fitfile in tqdm(fitfiles, mininterval=1):
        img = GT.getfileindex(fitfile.name.replace(f'_g{grainindex}', ''))
        spots, UB = read_fitfile_spots(fitfile)
        if spots.empty:
            continue
        UBs[img] = UB
        spots.insert(0, 'image', img)
        if isolation:
            corfile = corfile_path(cfg, img)
            XY = IOLT.readfile_cor(str(corfile))[0][:, 2:4]
            dist, _ = cKDTree(XY).query(spots[['Xexp', 'Yexp']].to_numpy(), k=2)
            spots['isolation'] = dist[:, 1]
        tables.append(spots)
    spots = pd.concat(tables, ignore_index=True)
    for col in ('spot_index', 'h', 'k', 'l'):
        spots[col] = spots[col].round().astype(int)
    if {'peak_fwaxmaj', 'peak_fwaxmin'} <= set(spots.columns):
        fw = spots[['peak_fwaxmaj', 'peak_fwaxmin']].abs()
        spots['elongation'] = fw.max(axis=1) / fw.min(axis=1).clip(lower=1e-3)
    if {'Xdev', 'Ydev'} <= set(spots.columns):
        spots['dev'] = np.hypot(spots['Xdev'], spots['Ydev'])
    return spots, UBs


def spot_quality(spots, max_dev: Optional[float] = 1.0, max_elongation: Optional[float] = 2.0,
                 max_fwhm: Optional[float] = None, min_isolation: Optional[float] = 10.,
                 max_pixdev: Optional[float] = None):
    """boolean Series: spots reliable for strain refinement

    max_dev: max distance (pixels) between fitted spot center and its initial position (Xdev, Ydev of
             peak search): large for asymmetric or multi-component spots
    max_elongation: max ratio of the fwhm along the 2 axes of the spot
    max_fwhm: max fwhm (pixels) of the large axis (None: no limit)
    min_isolation: min distance (pixels) to the nearest other spot of the image
    max_pixdev: max residual (pixels) of the spot in the previous refinement (None: no limit; caution, a
                small limit biases the strain towards the previous solution)
    """
    good = np.ones(len(spots), dtype=bool)
    for col, limit, op in (('dev', max_dev, np.less_equal), ('elongation', max_elongation, np.less_equal),
                           ('peak_fwaxmaj', max_fwhm, np.less_equal), ('isolation', min_isolation, np.greater_equal),
                           ('pixDev', max_pixdev, np.less_equal)):
        if limit is not None and col in spots:
            good &= op(spots[col].abs() if col == 'peak_fwaxmaj' else spots[col], limit).to_numpy()
    return good


def reference_reflections(spots, good=None, min_fraction: float = 0.8, nb_spots: Optional[int] = None):
    """reflections (hkl) of a grain reliable at most of its points: the same set is then used at all points

    spots: from read_grain_spots(); good: from spot_quality() (default: all spots)
    min_fraction: min fraction of the images of the grain where the reflection is indexed AND reliable
    nb_spots: keep only the nb_spots most frequent reflections (then most intense)

    Returns DataFrame (1 row per hkl): fraction, nb_images, median intensity / dev / elongation / isolation,
    'selected' (bool), sorted by decreasing fraction
    """
    good = np.ones(len(spots), dtype=bool) if good is None else np.asarray(good)
    nb_images = spots['image'].nunique()
    grouped = spots.assign(good=good).groupby(['h', 'k', 'l'])
    table = grouped.agg(nb_images=('good', 'sum'), nb_indexed=('image', 'nunique'),
                        intensity=('Intensity', 'median'),
                        **{f'median_{col}': (col, 'median') for col in ('dev', 'elongation', 'isolation', 'pixDev')
                           if col in spots})
    table['fraction'] = table['nb_images'] / nb_images
    table = table.sort_values(['fraction', 'intensity'], ascending=False)
    table['selected'] = table['fraction'] >= min_fraction
    if nb_spots is not None:
        table.loc[table.index[table['selected'].cumsum() > nb_spots], 'selected'] = False
    return table


def grain_spots_selection(cfg: dict, results_folder: Union[str, Path], cluster_ids: Optional[List[int]] = None,
                          output_cor_folder: Optional[Union[str, Path]] = None, min_fraction: float = 0.8,
                          nb_spots: Optional[int] = None, min_spots: int = 8, require_all: bool = False,
                          folder_prefix: str = 'grain_', verbose: bool = True, **quality):
    """.cor files restricted to a fixed set of reliable reflections per grain, for a 2nd refine_grains()

    1. spots of the .fit files of each grain (results of a 1st refine_grains() in results_folder, with align=True
       so that hkl are the same at all points of a grain)
    2. reliable spots (spot_quality(**quality): max_dev, max_elongation, max_fwhm, min_isolation, max_pixdev)
    3. reference reflections of the grain (reference_reflections(): min_fraction, nb_spots)
    4. for each image: .cor file with only the reference reflections that are reliable in this image
       (<output_cor_folder>/grain_###/<name>.cor, other columns of the .cor file kept). Images with less than
       min_spots such spots (or without all of them if require_all) are skipped

    Returns (tasks {cluster_id: {image: [UB of the 1st refinement]}}, cor_folders {cluster_id: folder},
    tables {cluster_id: reference_reflections() table}), for
        refine_grains(cfg, tasks, params, output_folder_2, cor_folders=cor_folders)
    """
    results_folder = Path(results_folder)
    output_cor_folder = Path(output_cor_folder) if output_cor_folder else results_folder / 'selected_corfiles'
    if cluster_ids is None:
        cluster_ids = sorted(int(p.name[len(folder_prefix):]) for p in results_folder.glob(f'{folder_prefix}[0-9]*')
                             if p.is_dir())
    tasks, cor_folders, tables = {}, {}, {}
    for cid in cluster_ids:
        spots, UBs = read_grain_spots(cfg, grain_folder(results_folder, cid, folder_prefix))
        good = spot_quality(spots, **quality)
        table = reference_reflections(spots, good, min_fraction=min_fraction, nb_spots=nb_spots)
        tables[cid] = table
        reference = set(table.index[table['selected']])
        keep = good & np.array([hkl in reference for hkl in zip(spots['h'], spots['k'], spots['l'])])
        outdir = grain_folder(output_cor_folder, cid, folder_prefix)
        outdir.mkdir(parents=True, exist_ok=True)
        tasks[cid], skipped = {}, 0
        for img, selected in spots[keep].groupby('image'):
            if len(selected) < min_spots or (require_all and len(selected) < len(reference)):
                skipped += 1
                continue
            _write_selected_corfile(corfile_path(cfg, img), selected['spot_index'].to_numpy(), outdir,
                                    CCDLabel=cfg['scan']['CCDLabel'])
            tasks[cid][img] = [UBs[img]]
        cor_folders[cid] = outdir
        if verbose:
            nbsel = spots[keep].groupby('image').size()
            print(f"grain {cid:3d}: {len(reference):3d} reference reflections (of {len(table)}), "
                  f"{100 * good.mean():.0f}% reliable spots, {len(tasks[cid])} images "
                  f"(median {nbsel.median() if len(nbsel) else 0:.0f} spots), {skipped} skipped (< {min_spots} spots)")
    return tasks, cor_folders, tables


def _write_selected_corfile(corfile: Path, spot_indices, outputfolder: Path, CCDLabel: str = 'sCMOS') -> str:
    """write <outputfolder>/<name of corfile> with only the spots spot_indices (rows of corfile), all columns kept"""
    out = IOLT.readfile_cor(str(corfile), output_CCDparamsdict=True, output_only5columns=False)
    rawdata, CCDcalibdict = out[0], out[7]
    props = out[8] if len(out) > 8 else None
    CCDcalibdict['CCDLabel'] = CCDLabel
    rows = np.sort(np.asarray(spot_indices, dtype=int))
    data = rawdata[rows]
    if props and 'data_spotsproperties' in props:
        props = dict(props, data_spotsproperties=np.asarray(props['data_spotsproperties'])[rows])
    else:
        props = None
    return IOLT.writefile_cor(corfile.stem, *data[:, :5].T, param=CCDcalibdict, initialfilename=str(corfile),
                              comments=f'{len(rows)} selected spots (grain_spots_selection)',
                              dirname_output=str(outputfolder), dict_data_spotsproperties=props)


# --------------------------------------------------------------------------------------
#  RESULTS
# --------------------------------------------------------------------------------------
def summarize_results(allresults: list, mapdims) -> dict:
    """2D maps (shape invmapdims = (slow, fast)) of nb of indexed spots and matching rate of first grain

    Returns dict with keys: treatedimages, nonindexedimages, indexedimages, grainlistindices,
    maskindexed2D, Nbindexedimages2D, MRindexedimages2D, outofmapimages
    """
    nfast, nslow = mapdims
    invmapdims = (nslow, nfast)
    Nbindexedimages2D = np.full(invmapdims, np.nan)
    MRindexedimages2D = np.full(invmapdims, np.nan)
    maskindexed2D = np.zeros(invmapdims, dtype=bool)
    treatedimages, indexedimages, nonindexedimages, outofmap = [], [], [], []

    for ix, dictmatrix, dictMR, _info, *_ in allresults:
        treatedimages.append(ix)
        if not 0 <= ix < nfast * nslow:
            outofmap.append(ix)
            continue
        row, col = ix // nfast, ix % nfast
        if len(dictmatrix) > 0 and dictmatrix[0][1] is not None:
            maskindexed2D[row, col] = True
            indexedimages.append(ix)
            score = dictMR[0][1]
            if score is not None:
                nbindexed, MR = score
                Nbindexedimages2D[row, col] = nbindexed
                MRindexedimages2D[row, col] = MR
        else:
            nonindexedimages.append(ix)
    if outofmap:
        GT.printyellow(f'{len(outofmap)} image indices are outside the map {mapdims}: {outofmap[:10]} ...')

    print('nb of treated datasets :', len(treatedimages))
    print('number of non indexed datasets', len(nonindexedimages))
    print('number of indexed (or updated) datasets', len(indexedimages))
    return {'treatedimages': treatedimages,
            'nonindexedimages': nonindexedimages,
            'indexedimages': indexedimages,
            'grainlistindices': list(indexedimages),
            'maskindexed2D': maskindexed2D,
            'Nbindexedimages2D': Nbindexedimages2D,
            'MRindexedimages2D': MRindexedimages2D,
            'outofmapimages': outofmap}


def save_allresults(cfg: dict, params: IndexRefineParams, allresults: list, summary: dict,
                    filepath: Optional[Union[str, Path]] = None, overwrite: bool = False,
                    nb_cpus: Optional[int] = None) -> Path:
    """save results in a pickle file (same structure as in the former Indexation_MultiProcessing notebooks)

    default filepath: <results_folder>/allresults_<results_stamp>.pickle
    """
    if filepath is None:
        filepath = Path(cfg['output']['results_folder']) / f"allresults_{cfg['output']['results_stamp']}.pickle"
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f'{filepath} already exists. Use overwrite=True or another filepath')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    scan = cfg['scan']
    dictresults = {
        'allresults': allresults,
        'dict_scan_data': {'imagefolder': scan['image_folder'],
                           'corfilefolder': str(scan['cor_folder']),
                           'fitfilefolder': str(cfg['output']['fit_folder']),
                           'prefixfilename': scan['prefix'],
                           'CCDLabel': scan['CCDLabel'],
                           'sizeofzeropadding': scan['nbdigits'],
                           'nbfiles_per_folder': scan['nbfiles_per_folder'],
                           'subfolder_prefix': scan['subfolder_prefix'],
                           'fit_nbfiles_per_folder': params.fit_nbfiles_per_folder,
                           'detfile': scan['detfile'],
                           'mapdims': scan['mapdims'],
                           'invmapdims': scan['invmapdims'],
                           'configfile': str(cfg['configfile'])},
        'dict_index_refine_for_mpi': {'e_min': params.e_min,
                                      'e_max': params.e_max,
                                      'e_max_MR': params.e_max_MR,
                                      'key_material': params.key_material,
                                      'nbGrainstoFind': params.nbGrainstoFind,
                                      'depth': params.depth,
                                      'dict_indexrefine': params.dict_indexrefine,
                                      'MatchingRate_List': params.MatchingRate_List,
                                      'MIN_NUMBERSPOTS_FOR_INDEXING': params.MIN_NUMBERSPOTS_FOR_INDEXING,
                                      'MAX_NUMBERSPOTS_FOR_INDEXING': params.MAX_NUMBERSPOTS_FOR_INDEXING},
        'dict_info_collect_mpi': {'nb_cpus': nb_cpus,
                                  'ignorefitfileresults': params.ignorefitfileresults,
                                  'usepreviousUB': params.usepreviousUB,
                                  'skipindexing': params.skipindexing,
                                  'writefitfile': params.writefitfile},
        'dict_results_summary': summary,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(dictresults, f)
    GT.printgreen(f'allresults saved at {datetime.datetime.now():%Y-%m-%d %H:%M:%S} in\n{filepath}')
    return filepath


def load_allresults(filepath: Union[str, Path]) -> dict:
    """load a allresults_*.pickle file written by save_allresults() (or by the former notebooks)"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# --------------------------------------------------------------------------------------
#  .cor files tools (remove spots of a substrate, merge files)
# --------------------------------------------------------------------------------------
def _write_corfile(rawdata, CCDcalibdict, fullpath, outputfilename, outputfolder, colnames=None):
    """write the spots of rawdata (rows of a .cor file). colnames: names of the columns after the 5 first ones
    (spots properties of peak search, written too)"""
    twicetheta, chi, data_x, data_y, dataintensity = rawdata[:, :5].T
    data_props = None
    if colnames and rawdata.shape[1] > 5:
        data_props = (rawdata[:, 5:5 + len(colnames)], colnames[:rawdata.shape[1] - 5])
    return IOLT.writefile_cor(outputfilename, twicetheta, chi, data_x, data_y, dataintensity,
                              param=CCDcalibdict, initialfilename=str(fullpath), dirname_output=str(outputfolder),
                              data_props=data_props)


def _read_corfile(fullpath, CCDLabel=None):
    """rawdata (all columns), CCDcalibdict and names of the columns after the 5 first ones of a .cor file

    CCDLabel: replaces the one of the .cor file (None: kept)
    """
    fullpath = Path(fullpath)
    if fullpath.suffix != '.cor':
        raise ValueError(f'{fullpath} extension is not .cor !')
    res = IOLT.readfile_cor(str(fullpath), output_CCDparamsdict=True)
    if res is None:
        raise ValueError(f'{fullpath} is empty')   # (no spot)
    rawdata, CCDcalibdict = np.atleast_2d(res[0]), res[-1]
    if CCDLabel is not None:
        CCDcalibdict['CCDLabel'] = CCDLabel
    with open(fullpath, 'r') as f:
        colnames = f.readline().split()[5:]
    return rawdata, CCDcalibdict, colnames


def removespots_fromlist(blacklistindices, fullpath, outputfilename=None, outputfolder=None,
                         prependprefix='p', CCDLabel=None):
    """write a .cor file without the spots of given indices

    outputfilename: stem only (no suffix). Default: prependprefix + stem of fullpath
    CCDLabel: detector of the written file (None: the one of the .cor file)
    Returns the name of the written .cor file
    """
    fullpath = Path(fullpath)
    rawdata, CCDcalibdict, colnames = _read_corfile(fullpath, CCDLabel)
    keep_indices = sorted(set(range(len(rawdata))) - set(np.ravel(blacklistindices).tolist()))
    print(f'nb of spots: initial {len(rawdata)}, final {len(keep_indices)}')
    return _write_corfile(np.take(rawdata, keep_indices, axis=0), CCDcalibdict, fullpath,
                          outputfilename or prependprefix + fullpath.stem, outputfolder or fullpath.parent,
                          colnames)


# --------------------------------------------------------------------------------------
#  BLACKLIST: spots (e.g. of a substrate) removed before indexing
# --------------------------------------------------------------------------------------
def _fitfile_xy(fitfile) -> np.ndarray:
    """pixel positions (Xexp, Yexp) of the indexed spots of all grains of a .fit file, shape (n, 2)"""
    xy, cols = [], None
    with open(fitfile, 'r') as f:
        for line in f:
            if line.startswith('##spot_index'):
                cols = line.lstrip('#').split()
                ix, iy = cols.index('Xexp'), cols.index('Yexp')
            elif cols is not None:
                if line.startswith('#') or not line.strip():
                    cols = None   # end of the block of spots of a grain
                else:
                    vals = line.split()
                    xy.append((float(vals[ix]), float(vals[iy])))
    return np.array(xy, dtype=float).reshape(-1, 2)


def read_blacklist(blacklist) -> np.ndarray:
    """pixel positions [X, Y] of blacklisted spots, shape (n, 2)

    blacklist:
    - None: no spot
    - a file: .cor or .dat (peak search), .fit (indexed spots of all grains, e.g. of the substrate)
    - an array of pixel positions [[X1, Y1], [X2, Y2], ...], shape (n, 2)
    - a list of files and/or arrays (all their spots)
    """
    if blacklist is None:
        return np.empty((0, 2))
    if isinstance(blacklist, (str, Path)):
        path = Path(os.path.expandvars(os.path.expanduser(str(blacklist))))
        if not path.exists():
            raise FileNotFoundError(f'blacklist file not found: {path}')
        if path.suffix == '.cor':
            res = IOLT.readfile_cor(str(path))
            return np.empty((0, 2)) if res is None else np.atleast_2d(res[0])[:, 2:4].astype(float)
        if path.suffix == '.dat':
            return np.atleast_2d(IOLT.read_Peaklist(str(path)))[:, :2].astype(float)
        if path.suffix == '.fit':
            return _fitfile_xy(path)
        raise ValueError(f'blacklist file {path} must be a .cor, .dat or .fit file')
    if isinstance(blacklist, (list, tuple)) and any(isinstance(b, (str, Path)) for b in blacklist):
        return np.vstack([read_blacklist(b) for b in blacklist])
    xy = np.asarray(blacklist, dtype=float)
    if xy.size == 0:
        return np.empty((0, 2))
    xy = np.atleast_2d(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f'blacklist array must be [[X1, Y1], [X2, Y2], ...] (shape (n, 2)), not shape {xy.shape}')
    return xy


def blacklisted_indices(XY, blacklist, dist_tolerance: float = 2.) -> np.ndarray:
    """indices of the spots of XY (pixel positions, shape (n, 2)) closer than dist_tolerance (pixel)
    to a spot of blacklist (see read_blacklist())"""
    from scipy.spatial import cKDTree
    black = read_blacklist(blacklist)
    XY = np.atleast_2d(np.asarray(XY, dtype=float))
    if len(black) == 0 or XY.size == 0:
        return np.array([], dtype=int)
    dist = cKDTree(black).query(XY[:, :2], distance_upper_bound=dist_tolerance)[0]
    return np.flatnonzero(dist <= dist_tolerance)


def _purge_corfile(blacklist, fullpath, dist_tolerance, outputfilename, outputfolder, prependprefix, CCDLabel):
    """write the .cor file without blacklisted spots. Returns (written file name, nb of spots before, after),
    None if the .cor file has no spot"""
    fullpath = Path(fullpath)
    if IOLT.readfile_cor(str(fullpath)) is None:
        return None
    rawdata, CCDcalibdict, colnames = _read_corfile(fullpath, CCDLabel)
    toremove = blacklisted_indices(rawdata[:, 2:4], blacklist, dist_tolerance)
    purged = np.delete(rawdata, toremove, axis=0)
    outputfolder = Path(outputfolder or fullpath.parent)
    outputfolder.mkdir(parents=True, exist_ok=True)
    written = _write_corfile(purged, CCDcalibdict, fullpath, outputfilename or prependprefix + fullpath.stem,
                             outputfolder, colnames)
    return written, len(rawdata), len(purged)


def removespots_blacklisted(blacklist, fullpath, dist_tolerance: float = 2., outputfilename=None, outputfolder=None,
                            prependprefix='p', CCDLabel=None, verbose=True):
    """write a .cor file without the spots closer than dist_tolerance (pixel) to the blacklisted spots

    blacklist: .cor, .dat or .fit file, array [[X1, Y1], ...] of pixel positions, or a list of them
    (see read_blacklist())
    outputfilename: stem only (no suffix). Default: prependprefix + stem of fullpath
    outputfolder: default: folder of fullpath
    CCDLabel: detector of the written file (None: the one of the .cor file)
    Returns the name of the written .cor file
    """
    purged = _purge_corfile(blacklist, fullpath, dist_tolerance, outputfilename, outputfolder, prependprefix, CCDLabel)
    if purged is None:
        raise ValueError(f'{fullpath} is empty')
    written, nbefore, nafter = purged
    if verbose:
        print(f'nb of spots: initial {nbefore}, final {nafter} ({nbefore - nafter} blacklisted spots removed)')
    return written


def removespots_fromfile(blacklistfile, fullpath, dist_tolerance=2, outputfilename=None, outputfolder=None,
                         CCDLabel=None, prependprefix='p'):
    """write a .cor file without the spots (within dist_tolerance pixel) of blacklistfile (.cor, .dat or .fit)

    Same as removespots_blacklisted(), which accepts also arrays of pixel positions.
    Returns the name of the written .cor file
    """
    return removespots_blacklisted(blacklistfile, fullpath, dist_tolerance=dist_tolerance,
                                   outputfilename=outputfilename, outputfolder=outputfolder,
                                   prependprefix=prependprefix, CCDLabel=CCDLabel)


def _match_rows(XY, corfile, tolerance: float = 0.01) -> np.ndarray:
    """rows of corfile of the spots at pixel positions XY (shape (n, 2)), within tolerance (pixel)"""
    from scipy.spatial import cKDTree
    corXY = np.atleast_2d(IOLT.readfile_cor(str(corfile))[0])[:, 2:4]
    dist, rows = cKDTree(corXY).query(np.atleast_2d(XY), k=min(2, len(corXY)))
    dist, rows = np.atleast_2d(dist.T).T, np.atleast_2d(rows.T).T
    if np.any(dist[:, 0] > tolerance):
        raise ValueError(f'{np.sum(dist[:, 0] > tolerance)} spot(s) not found in {corfile} (within {tolerance} pixel)')
    if dist.shape[1] > 1 and np.any(dist[:, 1] <= tolerance):
        raise ValueError(f'spots of {corfile} closer than {tolerance} pixel: rows can not be identified')
    return rows[:, 0]


def purged_to_original_rows(purged_corfile, original_corfile, tolerance: float = 0.01) -> np.ndarray:
    """rows of original_corfile of the spots of purged_corfile (a copy without some spots, e.g. blacklisted):
    result[i] = row in original_corfile of row i of purged_corfile (identified by pixel positions X, Y)"""
    res = IOLT.readfile_cor(str(purged_corfile))
    if res is None:
        return np.array([], dtype=int)
    return _match_rows(np.atleast_2d(res[0])[:, 2:4], original_corfile, tolerance)


def fitfile_source(fitfile) -> Optional[str]:
    """.cor file written in the first line of a .fit file ('... Refinement from experimental file: <path>')"""
    with open(fitfile, 'r') as f:
        first = f.readline()
    return first.split('experimental file:', 1)[1].strip() if 'experimental file:' in first else None


def restore_spot_indices(fitfile, original_corfile, purged_corfile=None, tolerance: float = 0.01) -> int:
    """rewrite the spot_index column of a .fit file obtained from a purged copy of original_corfile (e.g. without
    blacklisted spots): spot_index becomes the row of the spot in original_corfile (found from Xexp, Yexp).
    The first line then refers to original_corfile. Returns the nb of spots changed"""
    lines = Path(fitfile).read_text().splitlines(keepends=True)
    spotlines, cols = [], None   # (line number, columns of X and Y) of the spots of all grains
    for k, line in enumerate(lines):
        if line.startswith('##spot_index'):
            cols = line.lstrip('#').split()
            ixy = cols.index('Xexp'), cols.index('Yexp')
        elif cols is not None:
            if line.startswith('#') or not line.strip():
                cols = None   # end of the spots of a grain
            else:
                spotlines.append((k, ixy))
    if not spotlines:
        return 0
    XY = [[float(lines[k].split()[ix]), float(lines[k].split()[iy])] for k, (ix, iy) in spotlines]
    rows = _match_rows(XY, original_corfile, tolerance)
    nbchanged = 0
    for (k, _), row in zip(spotlines, rows):
        vals = lines[k].split()
        nbchanged += int(int(row) != int(float(vals[0])))
        vals[0] = f'{int(row):.6f}'
        lines[k] = ' '.join(vals) + '\n'
    if lines and 'experimental file:' in lines[0]:
        note = f' (spots indexed from {purged_corfile})' if purged_corfile is not None else ''
        lines[0] = lines[0].split('experimental file:', 1)[0] + f'experimental file: {original_corfile}{note}\n'
    Path(fitfile).write_text(''.join(lines))
    return nbchanged


def merge_corfiles(filecor1, filecor2, outputfilename, outputfolder):
    """raw concatenation of spots of two .cor files (only 5 first columns). Returns the written file name"""
    for ff in (filecor1, filecor2):
        if Path(ff).suffix != '.cor':
            raise ValueError(f'{ff} extension is not .cor !')
    rawdata1 = IOLT.readfile_cor(str(filecor1), output_CCDparamsdict=True)[0]
    rawdata2, _, _, _, _, _, _, CCDcalibdict = IOLT.readfile_cor(str(filecor2), output_CCDparamsdict=True)
    stackeddata = np.concatenate((rawdata1[:, :5], rawdata2[:, :5]))
    return _write_corfile(stackeddata, CCDcalibdict, f'concatenate of {filecor1} and {filecor2}',
                          outputfilename, outputfolder)


# --------------------------------------------------------------------------------------
#  MAP tools
# --------------------------------------------------------------------------------------
def show_figure(fig, live: bool = False, dpi: int = 100):
    """display a pyplot figure in a notebook

    live=False: static PNG image and figure closed (fast and light in JupyterLab with the ipympl
    'widget' backend, where each live canvas of a multi-panel figure is slow to display)
    live=True: usual display (interactive canvas with ipympl: zoom, hover values)
    """
    import io
    import matplotlib.pyplot as plt
    if live:
        plt.show()
        return
    from IPython.display import Image, display
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    plt.close(fig)   # closed figures are not displayed as live canvas at the end of the cell
    # rendered with an Agg canvas: with the ipympl 'widget' backend, saving through the live canvas of the figure
    # can give partly drawn (transparent) images
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    display(Image(buf.getvalue()))


def getimageindex_fromxyech(xech, yech, invmapdims, xech_stepsize=1., yech_stepsize=1., startingindex=0) -> int:
    """image index from relative positions along fast (xech) and slow (yech) axes

    invmapdims = dimensions (slow, fast)
    """
    ii = int(xech / xech_stepsize)
    jj = int(yech / yech_stepsize)
    return startingindex + invmapdims[1] * jj + ii


def quality_masks(ffs, invmapdims, MinimumNbIndexedSpots=20, LargestMeanPixelResidue=0.6) -> dict:
    """masks (True = masked) of a parsed_fitfileseries: points not indexed (no .fit file), too few indexed spots,
    too large pixel deviation, OR of them ('combined')"""
    nbspots = np.asarray(ffs.NumberOfIndexedSpots, dtype=float).reshape(invmapdims)
    pixdev = np.asarray(ffs.MeanDevPixel, dtype=float).reshape(invmapdims)
    cond_notindexed = ~np.isfinite(nbspots) | ~np.isfinite(pixdev)   # (NaN fails the comparisons below)
    cond_lowNbindexed = nbspots < MinimumNbIndexedSpots
    cond_largepixeldev = pixdev > LargestMeanPixelResidue
    return {'notindexed': cond_notindexed,
            'lowNbindexed': cond_lowNbindexed,
            'largepixeldev': cond_largepixeldev,
            'combined': cond_notindexed | cond_lowNbindexed | cond_largepixeldev}


# --------------------------------------------------------------------------------------
#  BATCH JOBS ON SLURM (ESRF cluster)
# --------------------------------------------------------------------------------------
# 1 job = 1 node, 1 task with nb_cpus cpus (multiprocessing pool, no MPI)
# cpus: nb of cpus per node. Check partitions and node features with: sinfo -N -o "%N %P %c %m %f %T"
# nodes > 1: the files are split into (at least) `nodes` jobs of a slurm job array, each job on 1 node
# whole_nodes: jobs always ask for all the cpus of the node (ESRF IT: any node of magnifix is given only to jobs of
# 192 cpus per node: 192 for magnifix, 2 x 192 for magnifix2, 3 x 192 for magnifix3). magnifix2, magnifix3: the
# partition magnifix with 2 or 3 nodes (no other partition, sinfo -s)
SLURM_MACHINES = {
    'magnifix': {'partition': 'magnifix', 'constraint': None, 'cpus': 192, 'whole_nodes': True},
    'magnifix2': {'partition': 'magnifix', 'constraint': None, 'cpus': 192, 'nodes': 2, 'whole_nodes': True},
    'magnifix3': {'partition': 'magnifix', 'constraint': None, 'cpus': 192, 'nodes': 3, 'whole_nodes': True},
    'hpc6': {'partition': 'nice', 'constraint': 'hpc6', 'cpus': 96},
    'hpc7': {'partition': 'nice', 'constraint': 'hpc7', 'cpus': 96},
    'hpc8': {'partition': 'nice', 'constraint': 'hpc8', 'cpus': 96},
    'nice': {'partition': 'nice', 'constraint': None, 'cpus': 96},
}
PRIORITY_MACHINE = 'magnifix'         # used when more than FALLBACK_MAX_CPUS cpus are asked
FALLBACK_MACHINES = ('hpc6', 'hpc7', 'hpc8')
FALLBACK_MAX_CPUS = 96

# each worker process (python + numpy + LaueTools + LUT + 1 image) needs several hundreds of MB.
# Below this, slurm OOM-kills workers: the pool loses their images and the job hangs until its time limit
MIN_MEM_PER_CPU_MB = 1000


def _mem_to_mb(mem: str) -> float:
    """slurm memory string ('300M', '2G', '2000') -> MB"""
    mem = str(mem).strip().upper().rstrip('B')
    factor = {'K': 1 / 1024, 'M': 1, 'G': 1024, 'T': 1024 ** 2}.get(mem[-1:], None)
    return float(mem[:-1]) * factor if factor else float(mem)


def _run(cmd: List[str], timeout: float = 30):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def slurm_idle_cpus(machine: str) -> Optional[int]:
    """largest nb of idle cpus on one node of a machine (see SLURM_MACHINES). None if sinfo is not available"""
    mach = SLURM_MACHINES[machine]
    try:
        out = _run(['sinfo', '-h', '-N', '-p', mach['partition'], '-o', '%N|%f|%C|%T']).stdout
    except (FileNotFoundError, OSError):
        return None
    best = 0
    for line in out.splitlines():
        try:
            node, features, cpus, state = line.split('|')
        except ValueError:
            continue
        if mach['constraint'] and mach['constraint'] not in features.split(','):
            continue
        if any(bad in state for bad in ('down', 'drain', 'fail', 'maint', 'reserved')):
            continue
        idle = int(cpus.split('/')[1])   # allocated/idle/other/total
        best = max(best, idle)
    return best


def choose_slurm_machine(nb_cpus: int, verbose: bool = True):
    """(machine, nb_cpus) for a job of nb_cpus cpus

    - more than FALLBACK_MAX_CPUS (96) cpus: PRIORITY_MACHINE (magnifix) if one of its nodes has enough idle cpus
    - otherwise: node of FALLBACK_MACHINES (hpc6, 7, 8) with most idle cpus, and at most 96 cpus
    Without sinfo (not on the ESRF cluster): magnifix if more than 96 cpus, else 'nice'
    """
    if nb_cpus > FALLBACK_MAX_CPUS:
        nb_cpus = min(nb_cpus, SLURM_MACHINES[PRIORITY_MACHINE]['cpus'])
        if SLURM_MACHINES[PRIORITY_MACHINE].get('whole_nodes'):   # a whole idle node is needed
            nb_cpus = SLURM_MACHINES[PRIORITY_MACHINE]['cpus']
        idle = slurm_idle_cpus(PRIORITY_MACHINE)
        if idle is None or idle >= nb_cpus:
            if verbose:
                print(f'{PRIORITY_MACHINE}: {"?" if idle is None else idle} idle cpus -> job of {nb_cpus} cpus on {PRIORITY_MACHINE}')
            return PRIORITY_MACHINE, nb_cpus
        if verbose:
            GT.printyellow(f'{PRIORITY_MACHINE}: only {idle} idle cpus on one node (< {nb_cpus}) '
                           f'-> {FALLBACK_MAX_CPUS} cpus on another machine')
    nb_cpus = min(nb_cpus, FALLBACK_MAX_CPUS)
    idles = {mach: slurm_idle_cpus(mach) for mach in FALLBACK_MACHINES}
    if any(v is None for v in idles.values()):
        return 'nice', nb_cpus
    machine = max(idles, key=idles.get)
    if verbose:
        print('idle cpus (1 node): ' + ', '.join(f'{m} {v}' for m, v in idles.items()) + f' -> {machine}')
        if idles[machine] < nb_cpus:
            GT.printyellow(f'the job will wait in the queue: no node with {nb_cpus} idle cpus')
    return machine, nb_cpus


def _check_slurm_resources(machine: str, nb_cpus: int, mem_per_cpu: str, nchunks: int, nfiles: int):
    """(nb_cpus, mem_per_cpu, nchunks) valid for a job on machine (see SLURM_MACHINES)"""
    if machine not in SLURM_MACHINES:
        raise ValueError(f'unknown machine {machine}: {list(SLURM_MACHINES)}')
    mach = SLURM_MACHINES[machine]
    if mach.get('whole_nodes') and nb_cpus != mach['cpus']:
        GT.printyellow(f"{machine}: jobs of whole nodes only ({mach['cpus']} cpus per node): nb_cpus {nb_cpus} -> "
                       f"{mach['cpus']}")
        nb_cpus = mach['cpus']
    elif nb_cpus > mach['cpus']:
        GT.printyellow(f"{machine} has {mach['cpus']} cpus per node: nb_cpus set to {mach['cpus']}")
        nb_cpus = mach['cpus']

    if _mem_to_mb(mem_per_cpu) < MIN_MEM_PER_CPU_MB:
        GT.printyellow(f'mem_per_cpu={mem_per_cpu} is too small (worker processes would be OOM killed and the job '
                       f'would hang): set to {MIN_MEM_PER_CPU_MB}M')
        mem_per_cpu = f'{MIN_MEM_PER_CPU_MB}M'

    nchunks = max(1, min(max(int(nchunks), mach.get('nodes', 1)), nfiles))
    return nb_cpus, mem_per_cpu, nchunks


def _new_jobdir(jobs_folder: Union[str, Path], job_name: str) -> Path:
    """create <jobs_folder>/<job_name>_<date>/logs and return the job folder"""
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    jobdir = (Path(jobs_folder) / f'{job_name}_{date}').resolve()
    (jobdir / 'logs').mkdir(parents=True, exist_ok=True)
    return jobdir


def _write_slurm_script(jobdir: Path, module: str, job_name: str, machine: str, nb_cpus: int, time: str,
                        mem_per_cpu: str, nchunks: int, env_setup: Optional[List[str]] = None,
                        python: Optional[str] = None, mail_user: Optional[str] = None) -> Path:
    """write <jobdir>/job.slurm running `python -m <module> --job <jobdir>` (1 node, 1 task of nb_cpus cpus)

    nchunks > 1: slurm job array, each job runs with --chunk $SLURM_ARRAY_TASK_ID --nchunks nchunks
    env_setup: shell lines run before python (default: activate the conda environment of this python)
    python: python executable (default: the one running this notebook)
    """
    import sys
    mach = SLURM_MACHINES[machine]
    python = python or sys.executable
    if env_setup is None:
        env_setup = ['module load mamba', f'conda activate {sys.prefix}']
    array = nchunks > 1
    log = jobdir / 'logs' / ('%x_%A_%a.out' if array else '%x_%j.out')
    lines = ['#!/bin/bash -l',
             f'#SBATCH --job-name={job_name}',
             f"#SBATCH --partition={mach['partition']}"]
    if mach['constraint']:
        lines.append(f"#SBATCH --constraint={mach['constraint']}")
    lines += ['#SBATCH --nodes=1',
              '#SBATCH --ntasks=1',
              f'#SBATCH --cpus-per-task={nb_cpus}',
              f'#SBATCH --mem-per-cpu={mem_per_cpu}',
              f'#SBATCH --time={time}',
              f'#SBATCH --output={log}']
    if array:
        lines.append(f'#SBATCH --array=0-{nchunks - 1}')
    if mail_user:
        lines += ['#SBATCH --mail-type=END,FAIL', f'#SBATCH --mail-user={mail_user}']
    chunk_args = f' --chunk $SLURM_ARRAY_TASK_ID --nchunks {nchunks}' if array else ''
    lines += ['',
              'echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname -s): $SLURM_CPUS_PER_TASK cpus, start $(date)"',
              *env_setup,
              'export OMP_NUM_THREADS=1',
              f'{python} -u -m {module} --job {jobdir} --ncpus $SLURM_CPUS_PER_TASK{chunk_args}',
              'echo "end $(date)"', '']
    script = jobdir / 'job.slurm'
    script.write_text('\n'.join(lines))
    return script


def prepare_slurm_job(cfg: dict, params: IndexRefineParams, listfiles: List[str], machine: str = 'magnifix',
                      nb_cpus: int = 192, time: str = '02:00:00', mem_per_cpu: str = '2000M', nchunks: int = 1,
                      job_name: Optional[str] = None, jobs_folder: Optional[Union[str, Path]] = None,
                      env_setup: Optional[List[str]] = None, python: Optional[str] = None,
                      mail_user: Optional[str] = None) -> dict:
    """write a job folder with parameters and slurm script to index & refine listfiles with params

    The job uses exactly `params` and `listfiles` (including changes made in the notebook), saved in job_params.pickle
    nchunks > 1: slurm job array of nchunks jobs, each one analysing 1/nchunks of listfiles
    machine with 'nodes' > 1 (e.g. 'magnifix2', 'magnifix3'): nchunks is at least the nb of nodes
    jobs_folder: default <parent folder of params.fit_folder>/slurm_jobs. Job folder: <jobs_folder>/<job_name>_<date>
    env_setup: shell lines run before python (default: activate the conda environment of this python)
    python: python executable (default: the one running this notebook)

    Returns job dict (also written in <job folder>/job.json): jobdir, script, command, ...
    """
    import json
    nb_cpus, mem_per_cpu, nchunks = _check_slurm_resources(machine, nb_cpus, mem_per_cpu, nchunks, len(listfiles))
    job_name = job_name or f'laue_{params.key_material}'
    # default: next to the .fit files folder of params (which may differ from the one of the configuration file)
    jobs_folder = Path(jobs_folder) if jobs_folder else Path(params.fit_folder).parent / 'slurm_jobs'
    jobdir = _new_jobdir(jobs_folder, job_name)

    params_job = dataclasses.replace(params, useinternalmultiprocessing=False, outputlistindices=False,
                                     verbosefilename=False)
    if params_job.LUT is None:
        params_job.build_LUT()
    if params_job.usepreviousUB == 'neighbours':   # UB store shared by the jobs: UB matrices of a previous run removed
        IN.UBStore(params_job.fit_folder).clear()
    with open(jobdir / 'job_params.pickle', 'wb') as f:
        pickle.dump({'cfg': cfg, 'params': params_job, 'listfiles': [str(ff) for ff in listfiles]}, f)
    readable = {k: v for k, v in dataclasses.asdict(params_job).items() if k != 'LUT'}
    with open(jobdir / 'job_params.txt', 'w') as f:
        f.write(f"# parameters of job {jobdir.name} (config {cfg['configfile']})\n")
        f.write(f"# {len(listfiles)} .cor files: {Path(listfiles[0]).name} ... {Path(listfiles[-1]).name}\n")
        f.write(yaml.safe_dump(json.loads(json.dumps(readable, default=lambda o: np.asarray(o).tolist())),
                               sort_keys=False))

    script = _write_slurm_script(jobdir, 'LaueTools.indexing_batch', job_name, machine, nb_cpus, time, mem_per_cpu,
                                 nchunks, env_setup=env_setup, python=python, mail_user=mail_user)

    job = {'jobdir': str(jobdir), 'script': str(script), 'command': f'sbatch {script}', 'machine': machine,
           'nb_cpus': nb_cpus, 'nchunks': nchunks, 'nfiles': len(listfiles), 'job_id': None,
           'fit_folder': str(params_job.fit_folder)}
    _write_job(job)
    print(f"job folder: {jobdir}\n{len(listfiles)} .cor files, {nchunks} job(s) of {nb_cpus} cpus on {machine}, "
          f"time limit {time}\n.fit files -> {params_job.fit_folder}")
    print(f"to submit from a terminal (jupyter-slurm):\n    {job['command']}")
    return job


def _write_job(job: dict):
    import json
    with open(Path(job['jobdir']) / 'job.json', 'w') as f:
        json.dump(job, f, indent=2)


def load_slurm_job(job: Union[dict, str, Path]) -> dict:
    """job dict from a job dict or a job folder path (e.g. after a restart of the notebook kernel)"""
    import json
    if isinstance(job, dict):
        return job
    with open(Path(job) / 'job.json') as f:
        return json.load(f)


def submit_slurm_job(job: Union[dict, str, Path]) -> Optional[str]:
    """sbatch the job script (possible from a jupyter-slurm session). Returns slurm job id"""
    import re
    job = load_slurm_job(job)
    try:
        res = _run(['sbatch', job['script']])
    except FileNotFoundError:
        GT.printred('sbatch not found: submit from a terminal of jupyter-slurm with:\n    ' + job['command'])
        return None
    if res.returncode != 0:
        GT.printred(f'sbatch failed: {res.stderr.strip()}')
        return None
    match = re.search(r'Submitted batch job (\d+)', res.stdout)
    job['job_id'] = match.group(1) if match else None
    job['submitted'] = datetime.datetime.now().isoformat(timespec='seconds')
    _write_job(job)
    GT.printgreen(res.stdout.strip())
    return job['job_id']


def slurm_job_status(job: Union[dict, str, Path], nblines: int = 3):
    """print state of the job (squeue, or sacct when finished), progress and last lines of the log file(s)"""
    import re
    job = load_slurm_job(job)
    if job.get('job_id') is None:
        # submitted from a terminal: job id from the log file names
        ids = {re.search(r'_(\d+)(_\d+)?\.out$', p.name).group(1)
               for p in (Path(job['jobdir']) / 'logs').glob('*.out') if re.search(r'_(\d+)(_\d+)?\.out$', p.name)}
        if ids:
            job['job_id'] = sorted(ids)[-1]
            _write_job(job)
    job_id = job.get('job_id')
    if job_id:
        try:
            out = _run(['squeue', '-h', '-j', job_id, '-o', '%i %P %T %M/%l %C cpus %R']).stdout.strip()
            if not out:
                out = _run(['sacct', '-X', '-n', '-j', job_id, '-o', 'JobID,Partition,State,Elapsed,NCPUS,NodeList']).stdout.strip()
            print(f'job {job_id}:\n{out}')
        except FileNotFoundError:
            print(f'job {job_id}: squeue/sacct not available here')
    else:
        print('job not submitted yet (or no log file yet)')
    parts = sorted(Path(job['jobdir']).glob('allresults_part*.pickle'))
    print(f"{len(parts)}/{job['nchunks']} result file(s) written")
    for logfile in sorted((Path(job['jobdir']) / 'logs').glob('*.out')):
        text = logfile.read_text(errors='replace')
        progress = re.findall(r'(\d+)/(\d+) \[', text)
        if progress:
            done, total = progress[-1]
            print(f'{logfile.name}: {done}/{total} images ({100 * int(done) / int(total):.0f}%)')
        last = [line for line in text.replace('\r', '\n').splitlines() if line.strip()][-nblines:]
        print('    ' + '\n    '.join(last))


def load_slurm_results(job: Union[dict, str, Path]) -> Optional[list]:
    """concatenated allresults of the job (all chunks). None if no result yet"""
    job = load_slurm_job(job)
    parts = sorted(Path(job['jobdir']).glob('allresults_part*.pickle'))
    if len(parts) < job['nchunks']:
        GT.printyellow(f"{len(parts)}/{job['nchunks']} result file(s) found: job(s) not finished or failed "
                       f"(see logs in {job['jobdir']}/logs)")
    if not parts:
        return None
    allresults = []
    for part in parts:
        allresults += load_allresults(part)['allresults']
    print(f"{len(allresults)} results / {job['nfiles']} {job.get('filetype', '.cor')} files")
    return allresults


def benchmark(listfiles: List[str], params: IndexRefineParams, nb_images: Optional[int] = None,
              nb_cpus: Optional[int] = None) -> float:
    """cpu time (s) per image of a short local batch (on nb_images files taken over listfiles).
    NB: .fit files of these images are written"""
    nb_cpus = nb_cpus or available_cpus()
    nb_images = min(nb_images or 2 * nb_cpus, len(listfiles))
    sample = [listfiles[i] for i in np.linspace(0, len(listfiles) - 1, nb_images).astype(int)]
    t0 = time.time()
    run_multiprocessing(sample, params, nb_cpus=nb_cpus)
    sec_cpu_per_image = (time.time() - t0) * nb_cpus / nb_images
    print(f'{sec_cpu_per_image:.1f} s x cpu per image')
    return sec_cpu_per_image


def walltime_estimate(nb_images: int, nb_cpus: int, sec_cpu_per_image: float, safety: float = 2.,
                      minimum_minutes: int = 10) -> str:
    """slurm time limit 'hh:mm:ss' for nb_images on nb_cpus (safety factor included)"""
    seconds = max(minimum_minutes * 60, safety * nb_images * sec_cpu_per_image / nb_cpus)
    hours, rest = divmod(int(seconds), 3600)
    return f'{hours:02d}:{rest // 60:02d}:{rest % 60:02d}'


# --------------------------------------------------------------------------------------
#  COMMAND LINE (used by slurm jobs)
# --------------------------------------------------------------------------------------
def main(argv=None):
    """python -m LaueTools.indexing_batch --job <job folder> [--chunk i --nchunks n] [--ncpus N]
    python -m LaueTools.indexing_batch --config <config.yaml> [--image-range start stop] [--ncpus N] [--overwrite]
    """
    import argparse
    import socket
    parser = argparse.ArgumentParser(description='LaueTools: index & refine .cor files (multiprocessing)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--job', help='job folder written by prepare_slurm_job()')
    source.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--ncpus', type=int, default=None, help='nb of processes (default: allocated cpus)')
    parser.add_argument('--chunk', type=int, default=0, help='index of the part of the files to analyse')
    parser.add_argument('--nchunks', type=int, default=1, help='nb of parts')
    parser.add_argument('--image-range', type=int, nargs=2, default=None, help='[--config] first and last image index')
    parser.add_argument('--overwrite', action='store_true', help='[--config] overwrite allresults pickle file')
    args = parser.parse_args(argv)

    if args.job:
        with open(Path(args.job) / 'job_params.pickle', 'rb') as f:
            job = pickle.load(f)
        if 'grain_tasks' in job:   # job of prepare_slurm_grain_job(): check orientation and refine each grain
            alltasks = job['grain_tasks']
            if args.nchunks > 1:
                alltasks = [alltasks[i] for i in np.array_split(np.arange(len(alltasks)), args.nchunks)[args.chunk]]
            nb_cpus = args.ncpus or available_cpus()
            print(f'{socket.gethostname()}: {len(alltasks)} grain refinement tasks (part {args.chunk + 1}/'
                  f'{args.nchunks}), {nb_cpus} processes, material {job["params"].key_material}', flush=True)
            results = run_grain_tasks(alltasks, job['params'], nb_cpus=nb_cpus, progress_interval=60)
            with open(Path(args.job) / f'allresults_part{args.chunk:03d}.pickle', 'wb') as f:
                pickle.dump({'allresults': [(cid, res) for cid, allres in results.items() for res in allres]}, f)
            print_grain_refinement(results)
            return
        cfg, params, listfiles = job['cfg'], job['params'], job['listfiles']
    else:
        cfg = load_config(args.config)
        if not check_config(cfg):
            raise SystemExit(1)
        params = params_from_config(cfg)
        listfiles = select_corfiles(cfg, image_range=args.image_range)
    nb_cpus = args.ncpus or available_cpus()
    if params.usepreviousUB == 'neighbours':   # band of the map of this job (UB matrices shared by the UB store)
        print(f'{socket.gethostname()}: part {args.chunk + 1}/{args.nchunks} of the map, {nb_cpus} processes, '
              f'material {params.key_material}\n.fit files -> {params.fit_folder}', flush=True)
        allresults = IN.run_neighbours(listfiles, params, nb_cpus=nb_cpus, progress_interval=60, chunk=args.chunk,
                                       nchunks=args.nchunks, clear_store=False if args.job else None)
    else:
        if args.nchunks > 1:
            listfiles = [str(ff) for ff in np.array_split(np.array(listfiles, dtype=object), args.nchunks)[args.chunk]]
        print(f'{socket.gethostname()}: {len(listfiles)} .cor files (part {args.chunk + 1}/{args.nchunks}), '
              f'{nb_cpus} processes, material {params.key_material}\n.fit files -> {params.fit_folder}', flush=True)
        allresults = run_multiprocessing(listfiles, params, nb_cpus=nb_cpus, progress_interval=60)
    summary = summarize_results(allresults, cfg['scan']['mapdims'])
    if args.job:
        filepath = Path(args.job) / f'allresults_part{args.chunk:03d}.pickle'
        save_allresults(cfg, params, allresults, summary, filepath=filepath, overwrite=True, nb_cpus=nb_cpus)
    else:
        save_allresults(cfg, params, allresults, summary, overwrite=args.overwrite, nb_cpus=nb_cpus)


if __name__ == '__main__':
    main()
