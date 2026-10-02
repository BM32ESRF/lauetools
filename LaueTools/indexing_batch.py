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
    if scan['cor_folder'] is None or not Path(scan['cor_folder']).is_dir():
        errors.append(f"scan: cor_folder does not exist: {scan['cor_folder']}")
    else:
        nbcor = len(list(Path(scan['cor_folder']).glob(f"{scan['prefix']}*.cor")))
        if nbcor == 0:
            errors.append(f"no {scan['prefix']}*.cor file in {scan['cor_folder']}")
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
    print(f".cor files:  {scan['cor_folder']}/{scan['prefix']}{'#' * scan['nbdigits']}.cor")
    print(f".fit files:  {out['fit_folder']}")
    print(f"map:         mapdims (fast, slow) = {scan['mapdims']}, motors ({scan['fast_motor']}, {scan['slow_motor']})")
    print(f"detector:    {scan['CCDLabel']}, calibration {scan['detfile']}")
    ind = cfg['indexing']
    print(f"material:    {cfg['material']['key_material']}, energy {ind['e_min']}-{ind['e_max']} keV "
          f"(indexing up to {ind['e_max_MR']} keV), nbGrainstoFind = {ind['nbGrainstoFind']}")


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
    LUT: Any = None

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
                               CCDLabel=cfg['scan']['CCDLabel'])
    params = dataclasses.replace(params, **kwargs)
    if isinstance(params.usepreviousUB, list):
        params.nbGrainstoFind = max(len(params.usepreviousUB), params.nbGrainstoFind)
    if build_LUT and params.LUT is None:
        params.build_LUT()
    return params


def index_refine(filename: Union[str, Path], params: IndexRefineParams) -> list:
    """index and refine Laue spots of a .cor file. Write .fit file(s) in params.fit_folder

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
    relatedfitfile = Path(params.fit_folder) / (filecor[:-4] + '_g0.fit')
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

    Path(params.fit_folder).mkdir(parents=True, exist_ok=True)

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
                          dirnameout_fitfile=Path(params.fit_folder),
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
    """full path of .cor file of a given image index"""
    scan = cfg['scan']
    return Path(scan['cor_folder']) / f"{scan['prefix']}{imageindex:0{scan['nbdigits']}d}.cor"


def select_corfiles(cfg: dict, image_range=None, roi=None) -> List[str]:
    """list of .cor files to analyse (sorted by image index)

    image_range: [start, stop] (stop included), default cfg['selection']['image_range']
    roi: [center image index, [half size fast, half size slow]] rectangle in the map,
         default cfg['selection']['roi']
    None for both: all .cor files of cor_folder with the prefix of the configuration
    """
    scan = cfg['scan']
    if image_range is None:
        image_range = cfg['selection']['image_range']
    if roi is None:
        roi = cfg['selection']['roi']

    files = sorted(Path(scan['cor_folder']).glob(f"{scan['prefix']}*.cor"), key=lambda p: GT.getfileindex(p))
    # keep only prefix####.cor (exclude e.g. purged or merged files)
    files = [ff for ff in files if ff.stem[len(scan['prefix']):].isdigit()]

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
    with multiprocessing.Pool(processes=nb_cpus, initializer=_init_worker, initargs=(params,)) as pool:
        for res in tqdm(pool.imap_unordered(_index_refine_worker, listfiles, chunksize=1), total=len(listfiles),
                        mininterval=progress_interval):
            allresults.append(res)
    dt = time.time() - t0
    print(f"It took {int(dt // 60)} min {int(dt % 60)} s to index {len(listfiles)} images with {nb_cpus} cpus.")
    return allresults


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
    import sys
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

    nchunks = max(1, min(max(int(nchunks), mach.get('nodes', 1)), len(listfiles)))
    python = python or sys.executable
    if env_setup is None:
        env_setup = ['module load mamba', f'conda activate {sys.prefix}']
    job_name = job_name or f'laue_{params.key_material}'
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # default: next to the .fit files folder of params (which may differ from the one of the configuration file)
    jobs_folder = Path(jobs_folder) if jobs_folder else Path(params.fit_folder).parent / 'slurm_jobs'
    jobdir = (jobs_folder / f'{job_name}_{date}').resolve()
    (jobdir / 'logs').mkdir(parents=True, exist_ok=True)

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
              f'{python} -u -m LaueTools.indexing_batch --job {jobdir} --ncpus $SLURM_CPUS_PER_TASK{chunk_args}',
              'echo "end $(date)"', '']
    script = jobdir / 'job.slurm'
    script.write_text('\n'.join(lines))

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
    print(f"{len(allresults)} results / {job['nfiles']} .cor files")
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
