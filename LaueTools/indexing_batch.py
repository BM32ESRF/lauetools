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
import multiprocessing
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
        'e_max_MR': 19,           # keV, for matching rate during indexing step
        'nbGrainstoFind': 1,
        'depth': 0,               # microns (normal to surface sample)
        'MatchingRate_List': None,  # default: [2] * len(list matching tol angles)
        'MIN_NUMBERSPOTS_FOR_INDEXING': 6,
        'MAX_NUMBERSPOTS_FOR_INDEXING': 10000,
        'MAXNBSPOTS': None,       # max nb of spots read in .cor file (None: all)
        'crudeMReval': True,
        'maxnbspots_MReval': 10000,
        'stop_Nb_Matches': 40,
        'dict_indexrefine': {
            # spots set A (int n means range(n)). CAUTION max(A) < NBMAXPROBED
            'central spots indices': 10,
            'AngleTolLUT': 0.2,   # tolerance angle [deg] to recognise angles in the angular LUT
            'nlutmax': 4,         # max miller index in the angular LUT
            'NBMAXPROBED': 25,    # spots set B = [0, ..., NBMAXPROBED-1]
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
        'usepreviousUB': False,
        'skipindexing': False,     # True: only refine usepreviousUB matrices (no indexing from scratch)
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


def _parse_indices(value) -> List[int]:
    """spots indices: int n -> [0, ..., n-1], 'start:stop' -> [start, ..., stop-1], or list"""
    if isinstance(value, int):
        return list(range(value))
    if isinstance(value, str):
        start, stop = value.split(':')
        return list(range(int(start), int(stop)))
    return [int(v) for v in value]


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
    cfg = _deep_update(copy.deepcopy(DEFAULT_CONFIG), usercfg)
    if overrides:
        _deep_update(cfg, overrides)
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

    dir_ = cfg['indexing']['dict_indexrefine']
    dir_['central spots indices'] = _parse_indices(dir_['central spots indices'])
    if cfg['indexing']['MatchingRate_List'] is None:
        cfg['indexing']['MatchingRate_List'] = [2] * len(dir_['list matching tol angles'])
    if cfg['output']['results_stamp'] is None:
        cfg['output']['results_stamp'] = cfg['material']['key_material']
    return cfg


def check_config(cfg: dict, verbose: bool = True) -> bool:
    """check the consistency of the configuration (folders, material, indexing parameters).

    Print errors (red) and warnings (yellow). Return True if no error.
    """
    errors, warnings = [], []
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
    if max(dir_['central spots indices']) >= int(dir_['NBMAXPROBED']):
        errors.append(f"indexing: NBMAXPROBED ({dir_['NBMAXPROBED']}) must be larger than the largest "
                      f"central spots index ({max(dir_['central spots indices'])})")
    if len(cfg['indexing']['MatchingRate_List']) < len(dir_['list matching tol angles']):
        errors.append("indexing: MatchingRate_List must be at least as long as 'list matching tol angles'")
    try:
        get_previousUB(cfg)
    except (KeyError, ValueError) as err:
        errors.append(f'run: usepreviousUB: {err}')

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
    for candidate in (name, name.replace('_', ' ')):
        if candidate in leaves:
            return leaves[candidate]
    import difflib
    close = difflib.get_close_matches(name, list(leaves), n=3)
    raise KeyError(f"unknown configuration parameter '{name}'" + (f'. Did you mean {close}?' if close else ''))


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
    """[0, ..., n-1] -> n, [s, ..., e-1] -> 's:e' (contiguous), else list"""
    indices = [int(i) for i in indices]
    if len(indices) > 3 and indices == list(range(indices[0], indices[-1] + 1)):
        return len(indices) if indices[0] == 0 else f'{indices[0]}:{indices[-1] + 1}'
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
                      central_spots_indices=10)
    ('central_spots_indices' stands for 'central spots indices', 'list_matching_tol_angles' for
    'list matching tol angles').

    check: check_config() of the new configuration (False e.g. if the .cor files do not exist yet)
    Returns the configuration loaded from the new file (as load_config()).
    """
    template = TEMPLATE_CONFIG if template is None else template
    pathchanges = {_key_path(name): value for name, value in changes.items()}
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
                      'stop_Nb_Matches', 'MatchingRate_List')}
_PARAMS_TO_CONFIG.update({'key_material': ('material', 'key_material'),
                          'mymaterials': ('material', 'mymaterials'),
                          'fit_folder': ('output', 'fit_folder'),
                          'skipindexing': ('run', 'skipindexing'),
                          'CCDLabel': ('scan', 'CCDLabel')})


def save_config(cfg: dict, newfile: Union[str, Path], params: Optional['IndexRefineParams'] = None,
                overwrite: bool = False, **changes) -> dict:
    """save a configuration file = file of `cfg` + parameters changed in `params` (e.g. after a test) + `changes`

    Only the parameters of `params` that differ from `cfg` are written (indexing parameters, dict_indexrefine,
    material, fit_folder, usepreviousUB, skipindexing), with the comments of the original file.
    Run options of tests (verboselevel, ignorefitfileresults, outputlistindices ...) are not saved.
    `changes`: other parameters given by their name, as in new_config() (e.g. test_image_index=1300).

    Example, after a successful test on 1 image:
        cfg = IB.save_config(cfg, 'configs/a321220_MgO_2grains.yaml', params=params_test)

    Returns the configuration loaded from the new file: use it for the batch (and its path as CONFIG
    in the postprocessing notebooks).
    """
    pathchanges = {}
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
            if not _same(value, cfg_dir.get(key)):
                if key == 'central spots indices':
                    value = _compact_indices(value)
                pathchanges[('indexing', 'dict_indexrefine', key)] = value

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

    pathchanges.update({_key_path(name): value for name, value in changes.items()})

    if not pathchanges:
        print('no parameter differs from the configuration file')
    for path, value in pathchanges.items():
        old = cfg
        for key in path:
            old = old.get(key) if isinstance(old, dict) else None
        print(f"  {': '.join(path)}: {_to_plain(old)} -> {_to_plain(value)}")

    header = (f"# saved {datetime.datetime.now():%Y-%m-%d %H:%M} from {Path(cfg['configfile']).name} "
              f"(IB.save_config)\n")
    newfile = _write_config(cfg['configfile'], newfile, pathchanges, overwrite=overwrite, header=header)
    GT.printgreen(f'configuration written: {newfile}')
    return load_config(newfile)


def get_previousUB(cfg: dict):
    """return `usepreviousUB` of cfg['run'] with named matrices replaced by numpy arrays

    Returns False, 'fromfitfile' or a list of 3x3 arrays
    """
    usepreviousUB = cfg['run']['usepreviousUB']
    if usepreviousUB in (False, None):
        return False
    if usepreviousUB == 'fromfitfile':
        return 'fromfitfile'
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
    blacklistfile: Optional[str] = None
    CCDLabel: str = 'sCMOS'
    fit_nbfiles_per_folder: Optional[int] = None   # .fit files in subfolders of fit_folder (see fitfile_folder())
    subfolder_prefix: str = 'images_'
    LUT: Any = None

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


def params_from_config(cfg: dict, build_LUT: bool = True, **kwargs) -> IndexRefineParams:
    """build IndexRefineParams from a configuration (see load_config()).

    kwargs replace any field of IndexRefineParams, e.g. verboselevel=2
    """
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
                               ignorefitfileresults=run['ignorefitfileresults'],
                               usepreviousUB=get_previousUB(cfg),
                               skipindexing=run['skipindexing'],
                               writefitfile=run['writefitfile'],
                               starting_grainindex=run['starting_grainindex'],
                               verboselevel=run['verboselevel'],
                               CCDLabel=cfg['scan']['CCDLabel'],
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

    if params.blacklistfile:
        corfile_written = removespots_fromfile(params.blacklistfile, filename, prependprefix='purged',
                                               CCDLabel=params.CCDLabel)
        filename = filename.parent / corfile_written

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

    previousResults = None
    fitfolder = params.fitfile_folder(image_index)
    relatedfitfile = fitfolder / (filecor[:-4] + '_g0.fit')
    info = ''
    if not params.ignorefitfileresults:
        if relatedfitfile.exists():
            return [image_index, [], [], 'not reanalyzed']
        info = '.fit file was missing. '

    usepreviousUB = params.usepreviousUB
    if usepreviousUB == 'fromfitfile':
        if relatedfitfile.exists():
            # read matrix from .fit file to skip indexing and go directly to refinement step
            UB_g0 = IOLT.readfitfile_multigrains(str(relatedfitfile))[3].reshape((3, 3))
            previousResults = (1, [np.array(UB_g0)], 0, 0)
            info = 'Tried UB matrix from .fit file. '
        else:
            info = 'Could not try UB matrix because .fit file was missing. '
    elif usepreviousUB in (False, None):
        pass
    elif isinstance(usepreviousUB, (list, tuple)):
        previousResults = (len(usepreviousUB), list(usepreviousUB), 0, 0)
    else:
        raise ValueError(f"usepreviousUB={usepreviousUB} should be False, 'fromfitfile' or a list of UB matrices")

    if params.skipindexing and usepreviousUB:
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
                          nbGrainstoFind=params.nbGrainstoFind,
                          previousResults=previousResults,
                          dirnameout_fitfile=fitfolder,
                          corfilename=filecor,
                          verbose=params.verboselevel,
                          MatchingRate_List=params.MatchingRate_List,
                          depth=params.depth,
                          dictmaterials=params.mymaterials,
                          writefitfile=params.writefitfile)

    listspotindices = dataset.getSpotsFamily(params.starting_grainindex)
    if params.verbosefilename:
        print(f'For {filecor}: grain #{params.starting_grainindex}, nb indexed spots: {len(listspotindices)}')
    info += ' Treated by IndexSpotsSet().'

    toreturn = [image_index,
                [(key, value) for key, value in dataset.dict_grain_matrix.items()],  # UB matrices
                [(key, value) for key, value in dataset.dict_grain_matching_rate.items()],  # nb indexed, MR
                info]
    if params.outputlistindices:
        toreturn.append(listspotindices)
    return toreturn


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
    return [str(ff) for ff in files]


_WORKER_PARAMS = None


def _init_worker(params):
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _index_refine_worker(filename):
    return index_refine(filename, _WORKER_PARAMS)


def mp_context():
    """multiprocessing context of the pools of the batches (forkserver in a jupyter kernel, see GT.mp_context())"""
    return GT.mp_context(preload=('LaueTools.indexing_batch', 'LaueTools.peaksearch_batch'))


def available_cpus() -> int:
    """nb of cpus this process may use (cpus allocated by slurm or jupyter-slurm, not all cpus of the machine)"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:   # not available on macOS / Windows
        return multiprocessing.cpu_count()


def run_multiprocessing(listfiles: List[str], params: IndexRefineParams, nb_cpus: Optional[int] = None,
                        progress_interval: float = 0.1) -> list:
    """index & refine all .cor files of listfiles with a pool of nb_cpus processes (1 image per process)

    nb_cpus: None: all cpus available for this process (see available_cpus())
    progress_interval: min time (s) between progress bar updates (large value for log files of batch jobs)

    Returns the list of results of index_refine() (unordered)
    """
    if nb_cpus is None:
        nb_cpus = available_cpus()
    if nb_cpus > available_cpus():
        GT.printyellow(f'{nb_cpus} processes for {available_cpus()} available cpus: '
                       'processes share cpus (no speed gain)')
    params = dataclasses.replace(params, useinternalmultiprocessing=False, outputlistindices=False)
    if params.LUT is None:
        params.build_LUT()

    t0 = time.time()
    allresults = []
    # parameters (and LUT) are sent once to each worker process
    with mp_context().Pool(processes=nb_cpus, initializer=_init_worker, initargs=(params,)) as pool:
        for res in tqdm(pool.imap_unordered(_index_refine_worker, listfiles, chunksize=1), total=len(listfiles),
                        mininterval=progress_interval):
            allresults.append(res)
    dt = time.time() - t0
    print(f"It took {int(dt // 60)} min {int(dt % 60)} s to index {len(listfiles)} images with {nb_cpus} cpus.")
    return allresults


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
def _write_corfile(rawdata, CCDcalibdict, fullpath, outputfilename, outputfolder):
    twicetheta, chi, data_x, data_y, dataintensity = rawdata[:, :5].T
    return IOLT.writefile_cor(outputfilename, twicetheta, chi, data_x, data_y, dataintensity,
                              param=CCDcalibdict, initialfilename=str(fullpath), dirname_output=str(outputfolder))


def removespots_fromlist(blacklistindices, fullpath, outputfilename=None, outputfolder=None,
                         prependprefix='p', CCDLabel='sCMOS'):
    """write a .cor file without the spots of given indices

    outputfilename: stem only (no suffix). Default: prependprefix + stem of fullpath
    Returns the name of the written .cor file
    """
    fullpath = Path(fullpath)
    if fullpath.suffix != '.cor':
        raise ValueError(f'{fullpath} extension is not .cor !')
    rawdata, _, _, _, _, _, _, CCDcalibdict = IOLT.readfile_cor(str(fullpath), output_CCDparamsdict=True)
    CCDcalibdict['CCDLabel'] = CCDLabel
    keep_indices = sorted(set(range(len(rawdata))) - set(np.ravel(blacklistindices).tolist()))
    print(f'nb of spots: initial {len(rawdata)}, final {len(keep_indices)}')
    return _write_corfile(np.take(rawdata, keep_indices, axis=0), CCDcalibdict, fullpath,
                          outputfilename or prependprefix + fullpath.stem, outputfolder or fullpath.parent)


def removespots_fromfile(blacklistfile, fullpath, dist_tolerance=2, outputfilename=None, outputfolder=None,
                         CCDLabel='sCMOS', prependprefix='p'):
    """write a .cor file without the spots (within dist_tolerance pixel) contained in the .cor blacklistfile

    outputfilename: stem only (no suffix). Default: prependprefix + stem of fullpath
    Returns the name of the written .cor file
    """
    fullpath = Path(fullpath)
    if fullpath.suffix != '.cor':
        raise ValueError(f'{fullpath} extension is not .cor !')
    rawdata, _, _, _, _, _, _, CCDcalibdict = IOLT.readfile_cor(str(fullpath), output_CCDparamsdict=True)
    CCDcalibdict['CCDLabel'] = CCDLabel
    rawdata_black = IOLT.readfile_cor(str(blacklistfile), output_CCDparamsdict=True)[0]
    _, _, keep_indices = GT.removeClosePoints_two_sets(rawdata[:, 2:4].T, rawdata_black[:, 2:4].T,
                                                       dist_tolerance=dist_tolerance)
    print(f'nb of spots: initial {len(rawdata)}, final {len(keep_indices)}')
    return _write_corfile(np.take(rawdata, keep_indices, axis=0), CCDcalibdict, fullpath,
                          outputfilename or prependprefix + fullpath.stem, outputfolder or fullpath.parent)


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
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)   # closed figures are not displayed as live canvas at the end of the cell
    display(Image(buf.getvalue()))


def getimageindex_fromxyech(xech, yech, invmapdims, xech_stepsize=1., yech_stepsize=1., startingindex=0) -> int:
    """image index from relative positions along fast (xech) and slow (yech) axes

    invmapdims = dimensions (slow, fast)
    """
    ii = int(xech / xech_stepsize)
    jj = int(yech / yech_stepsize)
    return startingindex + invmapdims[1] * jj + ii


def quality_masks(ffs, invmapdims, MinimumNbIndexedSpots=20, LargestMeanPixelResidue=0.6) -> dict:
    """masks (True = masked) of a parsed_fitfileseries: too few indexed spots, too large pixel deviation, OR of both"""
    cond_lowNbindexed = ffs.NumberOfIndexedSpots.reshape(invmapdims) < MinimumNbIndexedSpots
    cond_largepixeldev = ffs.MeanDevPixel.reshape(invmapdims) > LargestMeanPixelResidue
    return {'lowNbindexed': cond_lowNbindexed,
            'largepixeldev': cond_largepixeldev,
            'combined': np.logical_or(cond_lowNbindexed, cond_largepixeldev)}


# --------------------------------------------------------------------------------------
#  BATCH JOBS ON SLURM (ESRF cluster)
# --------------------------------------------------------------------------------------
# 1 job = 1 node, 1 task with nb_cpus cpus (multiprocessing pool, no MPI)
# cpus: nb of cpus per node. Check partitions and node features with: sinfo -N -o "%N %P %c %m %f %T"
# nodes > 1: the files are split into (at least) `nodes` jobs of a slurm job array, each job on 1 node
SLURM_MACHINES = {
    'magnifix': {'partition': 'magnifix', 'constraint': None, 'cpus': 192},
    'magnifix2': {'partition': 'magnifix', 'constraint': None, 'cpus': 192, 'nodes': 2},
    'magnifix3': {'partition': 'magnifix', 'constraint': None, 'cpus': 192, 'nodes': 3},
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
    if nb_cpus > mach['cpus']:
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
    if args.nchunks > 1:
        listfiles = [str(ff) for ff in np.array_split(np.array(listfiles, dtype=object), args.nchunks)[args.chunk]]
    nb_cpus = args.ncpus or available_cpus()
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
