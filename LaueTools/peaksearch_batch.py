# -*- coding: utf-8 -*-
"""
Module of LaueTools project to search the peaks (Laue spots) of a series of images (images -> .dat and .cor files)
with multiprocessing, driven by a YAML configuration file.

Used by the notebook LaueTools/notebooks/peaksearch/peaksearch_map.ipynb. Its results (.cor files) are the input
of LaueTools/notebooks/indexation/ (indexing_batch.py), see indexing_config().

Typical use::

    import LaueTools.peaksearch_batch as PB
    cfg = PB.load_config('configs/ma6034_Zn.yaml')
    PB.check_config(cfg)
    params = PB.params_from_config(cfg)
    imageindex, nbpeaks, peaklist = PB.peaksearch_image(PB.image_path(cfg, 600), params)    # test on 1 image
    allresults = PB.run_multiprocessing(PB.select_images(cfg), params, nb_cpus=cfg['run']['nb_cpus'])
    summary = PB.summarize_results(allresults, cfg['scan']['mapdims'])
    PB.save_results(cfg, params, allresults, summary)
    PB.indexing_config(cfg, '../indexation/configs/ma6034_Zn.yaml', key_material='Zn')   # next step: indexing

Map conventions are those of indexing_batch.py: mapdims = (fast, slow), invmapdims = (slow, fast),
image index = row * mapdims[0] + col

The batch jobs on the ESRF cluster (SLURM) use the machines (partitions) of indexing_batch.SLURM_MACHINES.
"""
import copy
import os
import pickle
import time
import datetime
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from LaueTools import readmccd as RMCCD
from LaueTools import IOLaueTools as IOLT
from LaueTools import LaueGeometry as LaueGeo
from LaueTools import generaltools as GT
from LaueTools import dict_LaueTools as DictLT
from LaueTools import indexing_batch as IB

# shared with indexing_batch: SLURM machines and jobs, available cpus, display of figures
from LaueTools.indexing_batch import (SLURM_MACHINES, available_cpus, choose_slurm_machine, submit_slurm_job,
                                      slurm_job_status, load_slurm_job, load_slurm_results, walltime_estimate,
                                      show_figure)

os.environ["OMP_NUM_THREADS"] = "1"

# --------------------------------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    'name': '',
    'description': '',
    'scan': {
        'image_folder': None,     # folder of images
        'prefix': 'img_',         # e.g. img_0023.tif -> 'img_'
        'suffix': None,           # e.g. img_0023.tif -> '.tif'. None: extension of the detector (dict_CCD)
        'nbdigits': 4,            # zero padding of image index: img_0023.tif -> 4
        'CCDLabel': 'sCMOS',
        'detfile': None,          # calibration .det file: .cor files are written if given
        'hdf5_logfile': None,     # BLISS hdf5 file of the experiment (optional)
        'mapdims': None,          # [fast, slow] nb of points
        'fast_motor': 'xech',     # label of the fast motor axis (plots)
        'slow_motor': 'yech',     # label of the slow motor axis (plots)
        'stepsizes': [1., 1.],    # [fast, slow] step sizes (plots)
    },
    'output': {
        'dat_folder': None,       # folder of .dat and .cor files (= cor_folder of the indexation)
        'added_string': '',       # inserted in output file names: img_0023.tif -> img_<added_string>0023.dat
        # None: all files in dat_folder. n (e.g. 10000): files in subfolders of n files
        # <dat_folder>/<subfolder_prefix><start>_<end>, e.g. images_0_9999, images_10000_19999 ...
        'nbfiles_per_folder': None,
        'subfolder_prefix': 'images_',
        'write_cor': True,        # write .cor files (needs scan: detfile)
        'results_folder': None,   # folder of peaksearch_<stamp>.pickle (default: dat_folder)
        'results_stamp': 'peaks',
    },
    'peaksearch': {
        # former .psp file (LaueTools GUI): its values replace the ones of this section at each loading.
        # Prefer import_pspfile(): values written once in this section, which is then the only reference
        'pspfile': None,
        # background removed before local maxima search: 'auto' (from the image itself, minimum filter),
        # null (raw image), or path to a background image
        'background': 'auto',
        'formulaexpression': 'A-1.1*B',   # A: image, B: background
        'fit_without_background': True,   # fit peaks on the image without background (Fit_with_Data_for_localMaxima)
        # arguments of readmccd.PeakSearch()
        'local_maxima_search_method': 0,  # 0: threshold, 1: shifted arrays, 2: convolution (thresholdConvolve, paramsHat)
        'IntensityThreshold': 200,        # minimum amplitude of a local maximum (above background)
        'thresholdConvolve': 200,
        'paramsHat': [4, 5, 2],
        'PixelNearRadius': 5,
        'boxsize': 3,                     # half size of the box of fitted pixels
        'fit_peaks_gaussian': 1,          # 0: no fit, 1: gaussian, 2: lorentzian
        'xtol': 0.001,
        'FitPixelDev': 6,                 # max distance (pixel) between local maximum and fitted position
        'PeakSizeRange': [0.65, 3],       # accepted fitted peak sizes (pixel)
        'MinIntensity': 0,
        'Saturation_value': None,         # peaks above are rejected. None: saturation of the detector (dict_CCD)
        'position_definition': 1,
        'maxPixelDistanceRejection': 10.,  # distance (pixel) to the spots of blacklist_file to remove a peak
        'NumberMaxofFits': 5000,
        'reject_negative_baseline': True,
        'npixels': 0,                     # remove peaks within npixels of the gaps of the detector (sCMOS, EIGER_4MCdTe)
        'blacklist_file': None,           # .dat or .fit file: its spots are removed from every peak list
    },
    # peaks inside these rectangles are removed: [[Xcenter, Ycenter, halfwidth, halfheight], ...] (pixel)
    'exclusion_boxes': [],
    'run': {
        'nb_cpus': None,           # None: all available cpus
        'overwrite': True,         # False: only images without .dat file (e.g. to resume an interrupted batch)
        'verboselevel': 0,
        'test_image_index': 0,     # image used for single image tests
    },
    'selection': {
        # None: all images of image_folder. Otherwise [start, stop] (stop included)
        'image_range': None,
        # rectangle region of interest in the map: [center image index, [half size fast, half size slow]]
        'roi': None,
    },
}

_PATH_KEYS = (('scan', 'image_folder'), ('scan', 'detfile'), ('scan', 'hdf5_logfile'),
              ('output', 'dat_folder'), ('output', 'results_folder'),
              ('peaksearch', 'pspfile'), ('peaksearch', 'blacklist_file'))

# keys of the 'peaksearch' section that are not arguments of readmccd.PeakSearch()
_NOT_PEAKSEARCH_ARGS = ('pspfile', 'background', 'fit_without_background', 'blacklist_file')

TEMPLATE_CONFIG = Path(__file__).parent / 'notebooks' / 'peaksearch' / 'config_template.yaml'


def read_pspfile(pspfile: Union[str, Path]) -> dict:
    """parameters of a .psp file (LaueTools GUI) with the names of the 'peaksearch' section of the configuration"""
    psp = RMCCD.readPeakSearchConfigFile(str(pspfile))
    if psp is None:
        raise ValueError(f'cannot read {pspfile}')
    if 'MinPeakSize' in psp or 'MaxPeakSize' in psp:
        psp['PeakSizeRange'] = [psp.pop('MinPeakSize', 0.65), psp.pop('MaxPeakSize', 3.)]
    if 'npixels_bandgap' in psp:
        psp['npixels'] = psp.pop('npixels_bandgap')
    ignored = [key for key in psp if key.startswith(('use_skimage', 'skimage_'))]
    for key in ignored:
        psp.pop(key)
    return {key: value for key, value in psp.items() if key in DEFAULT_CONFIG['peaksearch']}


def import_pspfile(cfg: Union[dict, str, Path], pspfile: Optional[Union[str, Path]] = None,
                   newfile: Optional[Union[str, Path]] = None) -> dict:
    """write the values of a .psp file (LaueTools GUI) in the 'peaksearch' section of a YAML configuration file,
    which then does not depend on the .psp file any more (pspfile: null)

    cfg: configuration (or its file). pspfile: default the pspfile of the configuration
    newfile: default: the configuration file itself is updated (comments kept)
    Returns the configuration loaded from the written file
    """
    if not isinstance(cfg, dict):
        cfg = load_config(cfg)
    pspfile = pspfile or cfg['peaksearch']['pspfile']
    if pspfile is None:
        raise ValueError('no .psp file: give pspfile')
    values = read_pspfile(pspfile)
    print(f'{len(values)} parameters of {pspfile}:')
    return save_config(cfg, newfile or cfg['configfile'], overwrite=True, pspfile=None, **values)


def load_config(configfile: Union[str, Path], overrides: Optional[dict] = None) -> dict:
    """read a YAML configuration file and complete it with default values

    - '~' and environment variables ($HOME, ...) are expanded in paths
    - relative paths are relative to the folder of the configuration file
    - peaksearch: pspfile: values of the .psp file replace those of the 'peaksearch' section
    - `overrides`: nested dict of values replacing those of the file (e.g. {'run': {'nb_cpus': 8}})

    Returns
    -------
    cfg : nested dict. Additional keys: cfg['configfile'], cfg['scan']['invmapdims']
    """
    configfile = Path(configfile).expanduser().resolve()
    with open(configfile, 'r') as f:
        usercfg = yaml.safe_load(f) or {}
    cfg = IB._deep_update(copy.deepcopy(DEFAULT_CONFIG), usercfg)
    if overrides:
        IB._deep_update(cfg, overrides)
    cfg['configfile'] = configfile

    for section, key in _PATH_KEYS:
        cfg[section][key] = IB._resolve_path(cfg[section][key], configfile.parent)
    if cfg['output']['results_folder'] is None:
        cfg['output']['results_folder'] = cfg['output']['dat_folder']
    if cfg['peaksearch']['pspfile'] is not None and Path(cfg['peaksearch']['pspfile']).exists():
        cfg['peaksearch'].update(read_pspfile(cfg['peaksearch']['pspfile']))

    mapdims = cfg['scan']['mapdims']
    if mapdims is not None:
        cfg['scan']['mapdims'] = (int(mapdims[0]), int(mapdims[1]))
        cfg['scan']['invmapdims'] = (int(mapdims[1]), int(mapdims[0]))
    else:
        cfg['scan']['invmapdims'] = None
    cfg['exclusion_boxes'] = cfg['exclusion_boxes'] or []
    if cfg['scan']['suffix'] is None and cfg['scan']['CCDLabel'] in DictLT.dict_CCD:
        cfg['scan']['suffix'] = '.' + DictLT.dict_CCD[cfg['scan']['CCDLabel']][7]
    return cfg


def check_config(cfg: dict, verbose: bool = True) -> bool:
    """check the consistency of the configuration (folders, detector, peak search parameters).

    Print errors (red) and warnings (yellow). Return True if no error.
    """
    errors, warnings = [], []
    scan, out, ps = cfg['scan'], cfg['output'], cfg['peaksearch']
    if scan['image_folder'] is None or not Path(scan['image_folder']).is_dir():
        errors.append(f"scan: image_folder does not exist: {scan['image_folder']}")
    else:
        nbimages = len(_image_files(cfg))
        if nbimages == 0:
            errors.append(f"no {scan['prefix']}{'#' * scan['nbdigits']}{scan['suffix']} image in {scan['image_folder']}")
        elif verbose:
            print(f"{nbimages} images found in {scan['image_folder']}")
            if scan['mapdims'] is not None and nbimages != scan['mapdims'][0] * scan['mapdims'][1]:
                warnings.append(f"nb of images ({nbimages}) differs from map size "
                                f"{scan['mapdims'][0]}x{scan['mapdims'][1]} = {scan['mapdims'][0] * scan['mapdims'][1]}")
    if scan['mapdims'] is None:
        warnings.append('scan: mapdims is missing ([nb points fast axis, nb points slow axis]): no map of results')
    if scan['CCDLabel'] not in DictLT.dict_CCD:
        errors.append(f"scan: unknown CCDLabel '{scan['CCDLabel']}' (keys of dict_LaueTools.dict_CCD)")
    if out['dat_folder'] is None:
        errors.append('output: dat_folder is missing')
    elif 'RAW_DATA' in Path(out['dat_folder']).parts:
        warnings.append(f"output: dat_folder is in RAW_DATA ({out['dat_folder']}): PROCESSED_DATA is recommended")
    nbpf = out['nbfiles_per_folder']
    if nbpf is not None and (not isinstance(nbpf, int) or nbpf < 1):
        errors.append(f'output: nbfiles_per_folder must be null or a positive integer, not {nbpf}')
    if out['write_cor']:
        if scan['detfile'] is None:
            warnings.append('scan: no detfile (calibration): .cor files are not written (needed for indexation)')
        elif not Path(scan['detfile']).exists():
            errors.append(f"scan: detfile not found: {scan['detfile']}")

    if ps['pspfile'] is not None and not Path(ps['pspfile']).exists():
        errors.append(f"peaksearch: pspfile not found: {ps['pspfile']}")
    if ps['blacklist_file'] is not None and not Path(ps['blacklist_file']).exists():
        errors.append(f"peaksearch: blacklist_file not found: {ps['blacklist_file']}")
    background = ps['background']
    if background not in (None, 'auto') and not Path(IB._resolve_path(background, cfg['configfile'].parent)).exists():
        errors.append(f"peaksearch: background image not found: {background}")
    if ps['IntensityThreshold'] <= 0:
        warnings.append(f"peaksearch: IntensityThreshold = {ps['IntensityThreshold']}: every pixel is a peak candidate "
                        "(slow, noise, images with more than NumberMaxofFits candidates get no peak)"
                        + (f" (value of pspfile {Path(ps['pspfile']).name})" if ps['pspfile'] else ''))
    if len(ps['PeakSizeRange']) != 2:
        errors.append('peaksearch: PeakSizeRange must be [min, max]')
    for box in cfg['exclusion_boxes']:
        if len(box) != 4:
            errors.append(f'exclusion_boxes: {box} is not [Xcenter, Ycenter, halfwidth, halfheight]')

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
    scan, out, ps = cfg['scan'], cfg['output'], cfg['peaksearch']
    print(f"---- {cfg['name']} ----  ({cfg['configfile']})")
    print(f"images:      {scan['image_folder']}/{scan['prefix']}{'#' * scan['nbdigits']}{scan['suffix']}")
    subfolder = (f"{out['subfolder_prefix']}<start>_<end>/" if out['nbfiles_per_folder'] else '')
    print(f"peak lists:  {out['dat_folder']}/{subfolder}{scan['prefix']}{out['added_string']}{'#' * scan['nbdigits']}.dat"
          f"{' and .cor' if out['write_cor'] and scan['detfile'] else ''}"
          + (f"  ({out['nbfiles_per_folder']} files per subfolder)" if out['nbfiles_per_folder'] else ''))
    print(f"map:         mapdims (fast, slow) = {scan['mapdims']}, motors ({scan['fast_motor']}, {scan['slow_motor']})")
    print(f"detector:    {scan['CCDLabel']}, calibration {scan['detfile']}")
    print(f"peaksearch:  method {ps['local_maxima_search_method']}, IntensityThreshold {ps['IntensityThreshold']}, "
          f"background {ps['background']}, boxsize {ps['boxsize']}, PeakSizeRange {ps['PeakSizeRange']}"
          + (f", .psp {ps['pspfile']}" if ps['pspfile'] else '')
          + (f", {len(cfg['exclusion_boxes'])} exclusion box(es)" if cfg['exclusion_boxes'] else ''))


# --------------------------------------------------------------------------------------
#  WRITE CONFIGURATION FILES (no manual editing of YAML indentation)
# --------------------------------------------------------------------------------------
def new_config(newfile: Union[str, Path], template: Optional[Union[str, Path]] = None,
               overwrite: bool = False, **changes) -> dict:
    """create a configuration file from a template, with some parameters changed (no manual YAML editing)

    template: config_template.yaml of LaueTools/notebooks/peaksearch (default) or the configuration file of another
    experiment. changes: parameters given by their name, whatever their section, e.g.
        PB.new_config('configs/myexp.yaml', image_folder='/data/.../scan0001', prefix='img_', mapdims=[51, 51],
                      dat_folder='/data/.../datfiles', detfile='/data/.../calib.det', IntensityThreshold=300)

    Returns the configuration loaded from the new file (as load_config()).
    """
    template = TEMPLATE_CONFIG if template is None else template
    pathchanges = {IB._key_path(name, DEFAULT_CONFIG): value for name, value in changes.items()}
    header = f'# created {datetime.datetime.now():%Y-%m-%d %H:%M} from {Path(template).name} (PB.new_config)\n'
    newfile = IB._write_config(template, newfile, pathchanges, overwrite=overwrite, header=header,
                               path_keys=_PATH_KEYS)
    GT.printgreen(f'configuration written: {newfile}')
    cfg = load_config(newfile)
    check_config(cfg)
    return cfg


def save_config(cfg: dict, newfile: Union[str, Path], params: Optional['PeakSearchParams'] = None,
                overwrite: bool = False, **changes) -> dict:
    """save a configuration file = file of `cfg` + parameters changed in `params` (e.g. after a test) + `changes`

    Only the parameters of `params` that differ from `cfg` are written (peaksearch section, exclusion_boxes,
    dat_folder, added_string), with the comments of the original file. If cfg uses a .psp file, all parameters
    of the peaksearch section are written and pspfile is set to null (the new file does not depend on the .psp file).
    `changes`: other parameters given by their name, as in new_config() (e.g. test_image_index=1300).

    Example, after a successful test on 1 image:
        cfg = PB.save_config(cfg, 'configs/myexp_v2.yaml', params=params_test)

    Returns the configuration loaded from the new file: use it for the batch.
    """
    pathchanges = {}
    if params is not None:
        frompsp = cfg['peaksearch']['pspfile'] is not None
        for key, value in params.peaksearch.items():
            if frompsp or not IB._same(value, cfg['peaksearch'].get(key)):
                pathchanges[('peaksearch', key)] = value
        if frompsp:
            pathchanges[('peaksearch', 'pspfile')] = None
        if not IB._same(params.exclusion_boxes, cfg['exclusion_boxes']):
            pathchanges[('exclusion_boxes',)] = params.exclusion_boxes
        newfolder = Path(os.path.expanduser(str(params.dat_folder))).resolve()
        if cfg['output']['dat_folder'] is None or newfolder != Path(cfg['output']['dat_folder']).resolve():
            pathchanges[('output', 'dat_folder')] = newfolder
        if params.added_string != cfg['output']['added_string']:
            pathchanges[('output', 'added_string')] = params.added_string
    pathchanges.update({IB._key_path(name, DEFAULT_CONFIG): value for name, value in changes.items()})

    if not pathchanges:
        print('no parameter differs from the configuration file')
    for path, value in pathchanges.items():
        old = cfg
        for key in path:
            old = old.get(key) if isinstance(old, dict) else None
        print(f"  {': '.join(path)}: {IB._to_plain(old)} -> {IB._to_plain(value)}")

    header = (f"# saved {datetime.datetime.now():%Y-%m-%d %H:%M} from {Path(cfg['configfile']).name} "
              f"(PB.save_config)\n")
    newfile = IB._write_config(cfg['configfile'], newfile, pathchanges, overwrite=overwrite, header=header,
                               path_keys=_PATH_KEYS)
    GT.printgreen(f'configuration written: {newfile}')
    return load_config(newfile)


def indexing_config(cfg: dict, newfile: Union[str, Path], template: Optional[Union[str, Path]] = None,
                    overwrite: bool = False, check: bool = True, **changes) -> dict:
    """write the configuration file of the indexation (notebooks/indexation, indexing_batch.py) of the .cor files
    of this peak search: cor_folder, prefix, nbdigits, CCDLabel, detfile, image_folder, hdf5_logfile, map and
    motors (and subfolders: nbfiles_per_folder, subfolder_prefix) are copied from cfg. `changes`: other parameters of the indexation, e.g. key_material='Zn',
    fit_folder='/data/.../fitfiles', nbGrainstoFind=2.

    template: config_template.yaml of notebooks/indexation (default) or the configuration of another indexation.
    check: check_config() of the new configuration (False before the peak search: no .cor file yet)
    Returns the indexation configuration (IB.load_config()).
    """
    scan, out = cfg['scan'], cfg['output']
    fromscan = {'cor_folder': out['dat_folder'], 'prefix': scan['prefix'] + out['added_string'],
                'nbfiles_per_folder': out['nbfiles_per_folder'], 'subfolder_prefix': out['subfolder_prefix'],
                'nbdigits': scan['nbdigits'], 'CCDLabel': scan['CCDLabel'], 'detfile': scan['detfile'],
                'image_folder': scan['image_folder'], 'hdf5_logfile': scan['hdf5_logfile'],
                'mapdims': scan['mapdims'], 'fast_motor': scan['fast_motor'], 'slow_motor': scan['slow_motor'],
                'stepsizes': scan['stepsizes'], 'test_image_index': cfg['run']['test_image_index']}
    if cfg['name']:
        fromscan['name'] = cfg['name']
    fromscan.update(changes)
    return IB.new_config(newfile, template=template, overwrite=overwrite, check=check, **fromscan)


# --------------------------------------------------------------------------------------
#  PEAK SEARCH of 1 image
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class PeakSearchParams:
    """all parameters needed by peaksearch_image(). Built from a config by params_from_config()"""
    CCDLabel: str
    prefix: str
    nbdigits: int
    dat_folder: Union[str, Path]
    peaksearch: dict                      # 'peaksearch' section of the configuration
    exclusion_boxes: list = dataclasses.field(default_factory=list)
    added_string: str = ''
    nbfiles_per_folder: Optional[int] = None   # files in subfolders of dat_folder (see datfile_path())
    subfolder_prefix: str = 'images_'
    calibration: Optional[dict] = None    # calibration parameters (.det file). None: no .cor file
    detfile: Optional[str] = None
    overwrite: bool = True
    writefiles: bool = True               # False: no .dat and .cor file (tests)
    verboselevel: int = 0

    def peaksearch_kwargs(self) -> dict:
        """arguments of readmccd.PeakSearch() (except filename)"""
        ps = self.peaksearch
        kwargs = {key: value for key, value in ps.items() if key not in _NOT_PEAKSEARCH_ARGS}
        kwargs['PeakSizeRange'] = tuple(kwargs['PeakSizeRange'])
        if kwargs['Saturation_value'] is None:
            kwargs['Saturation_value'] = DictLT.dict_CCD[self.CCDLabel][2]
        kwargs['paramsHat'] = tuple(kwargs['paramsHat'])
        background = ps['background']
        kwargs['Data_for_localMaxima'] = 'auto_background' if background == 'auto' else (
            None if background is None else str(background))
        kwargs['Fit_with_Data_for_localMaxima'] = bool(ps['fit_without_background']) and background is not None
        kwargs['Remove_BlackListedPeaks_fromfile'] = None if ps['blacklist_file'] is None else str(ps['blacklist_file'])
        kwargs.update(CCDLabel=self.CCDLabel, return_histo=0, write_execution_time=0,
                      verbose=max(self.verboselevel - 1, 0))
        return kwargs


def params_from_config(cfg: dict, **kwargs) -> PeakSearchParams:
    """PeakSearchParams from configuration. kwargs replace any field (e.g. verboselevel=1, writefiles=False)"""
    scan, out = cfg['scan'], cfg['output']
    calibration, detfile = None, None
    if out['write_cor'] and scan['detfile'] is not None:
        detfile = str(scan['detfile'])
        calibration = IOLT.readCalib_det_file(detfile)
    ps = copy.deepcopy(cfg['peaksearch'])
    if ps['background'] not in (None, 'auto'):
        ps['background'] = str(IB._resolve_path(ps['background'], cfg['configfile'].parent))
    params = PeakSearchParams(CCDLabel=scan['CCDLabel'], prefix=scan['prefix'], nbdigits=scan['nbdigits'],
                              dat_folder=out['dat_folder'], peaksearch=ps,
                              exclusion_boxes=copy.deepcopy(cfg['exclusion_boxes']),
                              added_string=out['added_string'], nbfiles_per_folder=out['nbfiles_per_folder'],
                              subfolder_prefix=out['subfolder_prefix'], calibration=calibration, detfile=detfile,
                              overwrite=cfg['run']['overwrite'], verboselevel=cfg['run']['verboselevel'])
    return dataclasses.replace(params, **kwargs)


def remove_peaks_in_boxes(peaklist: np.ndarray, boxes: list) -> np.ndarray:
    """peaklist without the peaks (rows) inside the rectangles [Xcenter, Ycenter, halfwidth, halfheight]"""
    if peaklist is None or len(peaklist) == 0 or not boxes:
        return peaklist
    X, Y = peaklist[:, 0], peaklist[:, 1]
    inside = np.zeros(len(peaklist), dtype=bool)
    for Xcenter, Ycenter, halfwidth, halfheight in boxes:
        inside |= (np.fabs(X - Xcenter) < halfwidth) & (np.fabs(Y - Ycenter) < halfheight)
    return peaklist[~inside]


def datfile_path(params: PeakSearchParams, imageindex: int) -> Path:
    """full path of the .dat file (peak list) of an image (.cor file: same path with .cor)

    with params.nbfiles_per_folder = n: in subfolder <dat_folder>/<subfolder_prefix><start>_<end> of n files
    """
    folder = Path(params.dat_folder)
    if params.nbfiles_per_folder:
        folder = folder / GT.subfolder_name(imageindex, params.nbfiles_per_folder, params.subfolder_prefix)
    return folder / f"{params.prefix}{params.added_string}{imageindex:0{params.nbdigits}d}.dat"


def peaksearch_image(imagefile: Union[str, Path], params: PeakSearchParams) -> tuple:
    """peak search on 1 image, then .dat file (and .cor file if calibration) written in params.dat_folder
    (or in its subfolder, see datfile_path())

    Returns (imageindex, nb of peaks, peaklist) with peaklist an array (nb peaks, 10 columns):
    X, Y, I (background subtracted), fwhm major axis, fwhm minor axis, inclination, Xdev, Ydev, background, Ipixmax.
    nb of peaks = -1 if the image could not be read or the peak search failed, peaklist = None if no peak.
    Without params.overwrite, images with an existing .dat file are skipped: (imageindex, -2, None)
    """
    imagefile = Path(imagefile)
    imageindex = GT.getfileindex(str(imagefile))
    datfile = datfile_path(params, imageindex)
    if not params.overwrite and datfile.exists():
        return imageindex, -2, None
    try:
        res = RMCCD.PeakSearch(str(imagefile), **params.peaksearch_kwargs())
    except Exception as err:   # unreadable image, ...: the batch goes on
        if params.verboselevel > 0:
            GT.printred(f'{imagefile.name}: {type(err).__name__} {err}')
        return imageindex, -1, None
    peaklist = None if res is None else res[0]
    peaklist = remove_peaks_in_boxes(peaklist, params.exclusion_boxes)
    if peaklist is None or len(peaklist) == 0:
        return imageindex, 0, None

    if params.writefiles:
        datfile.parent.mkdir(parents=True, exist_ok=True)
        fullpath_datfile = RMCCD.writepeaklist(peaklist, datfile.stem, outputfolder=str(datfile.parent),
                                               initialfilename=str(imagefile), verbose=max(params.verboselevel - 1, 0))
        if params.calibration is not None:
            LaueGeo.convert2corfile(Path(fullpath_datfile).name, [], dirname_in=str(datfile.parent),
                                    dirname_out=str(datfile.parent),
                                    CCDCalibdict=copy.deepcopy(params.calibration), add_props=True, verbose=0)
        if params.verboselevel > 0:
            print(f'{imagefile.name}: {len(peaklist)} peaks -> {fullpath_datfile}')
    return imageindex, len(peaklist), peaklist


def print_result(res: tuple):
    """print the result of peaksearch_image()"""
    imageindex, nbpeaks, peaklist = res
    if nbpeaks == -1:
        GT.printred(f'image {imageindex}: peak search failed (image not readable?)')
    elif nbpeaks == -2:
        print(f'image {imageindex}: skipped (.dat file exists and overwrite is False)')
    elif nbpeaks == 0:
        GT.printyellow(f'image {imageindex}: no peak found')
    else:
        print(f'image {imageindex}: {nbpeaks} peaks, intensities {peaklist[0, 2]:.0f} ... {peaklist[-1, 2]:.0f}, '
              f'mean fwhm {np.mean(peaklist[:, 3:5]):.2f} pixel')


# --------------------------------------------------------------------------------------
#  SET OF IMAGES & MULTIPROCESSING
# --------------------------------------------------------------------------------------
def image_path(cfg: dict, imageindex: int) -> Path:
    """full path of the image of a given index"""
    scan = cfg['scan']
    return Path(scan['image_folder']) / f"{scan['prefix']}{imageindex:0{scan['nbdigits']}d}{scan['suffix']}"


def _image_files(cfg: dict) -> List[Path]:
    """images prefix####suffix of image_folder, sorted by index"""
    scan = cfg['scan']
    files = Path(scan['image_folder']).glob(f"{scan['prefix']}*{scan['suffix']}")
    lp, ls = len(scan['prefix']), len(scan['suffix'])
    files = [ff for ff in files if ff.name[lp:len(ff.name) - ls].isdigit()]
    return sorted(files, key=lambda p: int(p.name[lp:len(p.name) - ls]))


def select_images(cfg: dict, image_range=None, roi=None) -> List[str]:
    """list of images to analyse (sorted by image index)

    image_range: [start, stop] (stop included), default cfg['selection']['image_range']
    roi: [center image index, [half size fast, half size slow]] rectangle in the map, default cfg['selection']['roi']
    None for both: all images of image_folder with the prefix and suffix of the configuration
    """
    scan = cfg['scan']
    if image_range is None:
        image_range = cfg['selection']['image_range']
    if roi is None:
        roi = cfg['selection']['roi']
    files = _image_files(cfg)
    index = lambda p: int(p.name[len(scan['prefix']):len(p.name) - len(scan['suffix'])])
    if image_range is not None:
        start, stop = image_range
        files = [ff for ff in files if start <= index(ff) <= stop]
    if roi is not None:
        center, halfsizes = roi
        indices = GT.extract2Dslice(center, halfsizes,
                                    np.arange(scan['mapdims'][0] * scan['mapdims'][1]).reshape(scan['invmapdims']))
        indices = set(np.ravel(indices).tolist())
        files = [ff for ff in files if index(ff) in indices]
    return [str(ff) for ff in files]


_WORKER_PARAMS = None


def _init_worker(params):
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _peaksearch_worker(imagefile):
    return peaksearch_image(imagefile, _WORKER_PARAMS)


def run_multiprocessing(listfiles: List[str], params: PeakSearchParams, nb_cpus: Optional[int] = None,
                        progress_interval: float = 0.1) -> list:
    """peak search on all images of listfiles with a pool of nb_cpus processes (1 image per process)

    nb_cpus: None: all cpus available for this process (see available_cpus())
    progress_interval: min time (s) between progress bar updates (large value for log files of batch jobs)

    Returns the list of results of peaksearch_image() (unordered)
    """
    if nb_cpus is None:
        nb_cpus = available_cpus()
    if nb_cpus > available_cpus():
        GT.printyellow(f'{nb_cpus} processes for {available_cpus()} available cpus: '
                       'processes share cpus (no speed gain)')
    if params.writefiles:
        Path(params.dat_folder).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    allresults = []
    with IB.mp_context().Pool(processes=nb_cpus, initializer=_init_worker, initargs=(params,)) as pool:
        for res in tqdm(pool.imap_unordered(_peaksearch_worker, listfiles, chunksize=1), total=len(listfiles),
                        mininterval=progress_interval):
            allresults.append(res)
    dt = time.time() - t0
    print(f"It took {int(dt // 60)} min {int(dt % 60)} s to search peaks in {len(listfiles)} images with {nb_cpus} "
          f"cpus ({len(listfiles) / max(dt, 1e-6):.1f} images/s).")
    return allresults


# --------------------------------------------------------------------------------------
#  RESULTS
# --------------------------------------------------------------------------------------
def summarize_results(allresults: list, mapdims=None) -> dict:
    """nb of peaks per image, and its 2D map (shape invmapdims = (slow, fast), nan for images not treated)

    Returns dict with keys: imageindices, nbpeaks (sorted by image index), failedimages, emptyimages,
    skippedimages, nbpeaks2D (None without mapdims), outofmapimages
    """
    ordered = sorted(allresults, key=lambda r: r[0])
    imageindices = np.array([r[0] for r in ordered], dtype=int)
    nbpeaks = np.array([r[1] for r in ordered], dtype=int)
    summary = {'imageindices': imageindices,
               'nbpeaks': nbpeaks,
               'failedimages': imageindices[nbpeaks == -1].tolist(),
               'emptyimages': imageindices[nbpeaks == 0].tolist(),
               'skippedimages': imageindices[nbpeaks == -2].tolist(),
               'nbpeaks2D': None,
               'outofmapimages': []}
    if mapdims is not None:
        nfast, nslow = mapdims
        nbpeaks2D = np.full((nslow, nfast), np.nan)
        inmap = (imageindices >= 0) & (imageindices < nfast * nslow) & (nbpeaks >= 0)
        nbpeaks2D.flat[imageindices[inmap]] = nbpeaks[inmap]
        summary['nbpeaks2D'] = nbpeaks2D
        summary['outofmapimages'] = imageindices[(imageindices < 0) | (imageindices >= nfast * nslow)].tolist()
        if summary['outofmapimages']:
            GT.printyellow(f"{len(summary['outofmapimages'])} image indices are outside the map {mapdims}: "
                           f"{summary['outofmapimages'][:10]} ...")
    treated = nbpeaks >= 0
    print('nb of treated images :', int(np.sum(treated)))
    if np.any(treated):
        print(f'nb of peaks per image: min {nbpeaks[treated].min()}, mean {nbpeaks[treated].mean():.1f}, '
              f'max {nbpeaks[treated].max()}')
    for key, label in (('emptyimages', 'without peak'), ('failedimages', 'failed (not readable?)'),
                       ('skippedimages', 'skipped (existing .dat file)')):
        if summary[key]:
            print(f'nb of images {label}: {len(summary[key])}  {summary[key][:10]}{" ..." if len(summary[key]) > 10 else ""}')
    return summary


def peaks_of_image(allresults: list, imageindex: int) -> Optional[np.ndarray]:
    """peaklist of an image from allresults (None if no peak or not treated)"""
    for index, _, peaklist in allresults:
        if index == imageindex:
            return peaklist
    return None


def save_results(cfg: dict, params: PeakSearchParams, allresults: list, summary: Optional[dict] = None,
                 filepath: Optional[Union[str, Path]] = None, overwrite: bool = False,
                 nb_cpus: Optional[int] = None) -> Path:
    """save all peak lists and the parameters in a pickle file

    default filepath: <results_folder>/peaksearch_<results_stamp>.pickle
    """
    if filepath is None:
        filepath = Path(cfg['output']['results_folder']) / f"peaksearch_{cfg['output']['results_stamp']}.pickle"
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f'{filepath} already exists. Use overwrite=True or another filepath')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    scan = cfg['scan']
    dictresults = {
        'allresults': allresults,   # list of (imageindex, nbpeaks, peaklist array (nbpeaks, 10) or None)
        'dict_scan_data': {'imagefolder': str(scan['image_folder']),
                           'datfilefolder': str(params.dat_folder),
                           'prefixfilename': scan['prefix'],
                           'suffix': scan['suffix'],
                           'added_string': params.added_string,
                           'nbfiles_per_folder': params.nbfiles_per_folder,
                           'subfolder_prefix': params.subfolder_prefix,
                           'CCDLabel': scan['CCDLabel'],
                           'sizeofzeropadding': scan['nbdigits'],
                           'detfile': params.detfile,
                           'mapdims': scan['mapdims'],
                           'invmapdims': scan['invmapdims'],
                           'configfile': str(cfg['configfile'])},
        'dict_peaksearch': {'peaksearch': params.peaksearch, 'exclusion_boxes': params.exclusion_boxes,
                            'nb_cpus': nb_cpus},
        'dict_results_summary': summary,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(dictresults, f)
    GT.printgreen(f'results saved at {datetime.datetime.now():%Y-%m-%d %H:%M:%S} in\n{filepath}')
    return filepath


def load_results(filepath: Union[str, Path]) -> dict:
    """load a peaksearch_*.pickle file written by save_results()"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# --------------------------------------------------------------------------------------
#  PLOTS
# --------------------------------------------------------------------------------------
def read_image(imagefile: Union[str, Path], CCDLabel: str = 'sCMOS') -> np.ndarray:
    """2D array of pixel intensities of an image (as read by the peak search)"""
    from LaueTools import IOimagefile as IOimage
    return IOimage.readCCDimage(str(imagefile), CCDLabel=CCDLabel)[0]


def plot_peaks(imagedata: np.ndarray, peaklist: Optional[np.ndarray], exclusion_boxes: Optional[list] = None,
               vmin: Optional[float] = None, vmax: Optional[float] = None, title: str = '', ax=None):
    """image with found peaks (green +) and exclusion boxes (red rectangles)

    vmin, vmax: color scale (default: 1st and 99.9th percentiles of the image)
    peaks positions follow position_definition=1 (first pixel at (1, 1)): plotted at (X - 1, Y - 1)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    if vmin is None or vmax is None:
        lo, hi = np.percentile(imagedata, [1, 99.9])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    ax.imshow(imagedata, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='gray_r')
    if peaklist is not None and len(peaklist):
        ax.scatter(peaklist[:, 0] - 1, peaklist[:, 1] - 1, marker='+', color='limegreen', s=40, lw=1)
    for Xcenter, Ycenter, halfwidth, halfheight in exclusion_boxes or []:
        ax.add_patch(Rectangle((Xcenter - 1 - halfwidth, Ycenter - 1 - halfheight), 2 * halfwidth, 2 * halfheight,
                               fill=False, ec='r', ls='--'))
    ax.set_xlim(0, imagedata.shape[1])
    ax.set_ylim(imagedata.shape[0], 0)
    ax.set_title(f"{title}\n{0 if peaklist is None else len(peaklist)} peaks", fontsize=9)
    return ax


def plot_vignettes(imagedata: np.ndarray, peaklist: np.ndarray, nbmax: int = 100, ncols: int = 10):
    """zoom on the nbmax most intense peaks with their fitted ellipse (fwhm)"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    npeaks = min(nbmax, 0 if peaklist is None else len(peaklist))
    if npeaks == 0:
        print('no peak')
        return None
    ncols = min(ncols, npeaks)
    nrows = int(np.ceil(npeaks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.1), squeeze=False)
    for iii, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if iii >= npeaks:
            ax.axis('off')
            continue
        X, Y, Isub, fwaxmaj, fwaxmin, inclination = peaklist[iii, :6]
        x, y = X - 1, Y - 1
        size = max(abs(fwaxmaj), abs(fwaxmin), 3)
        xmin, xmax = max(int(round(x - 1.5 * size)), 0), int(round(x + 1.5 * size)) + 1
        ymin, ymax = max(int(round(y - 1.5 * size)), 0), int(round(y + 1.5 * size)) + 1
        zoom = imagedata[ymin:ymax, xmin:xmax]
        ax.pcolormesh(np.arange(xmin - .5, xmin + zoom.shape[1]), np.arange(ymin - .5, ymin + zoom.shape[0]), zoom,
                      vmin=np.amin(zoom), vmax=np.amax(zoom), cmap='inferno')
        ax.scatter(x, y, marker='+', color='limegreen', s=30)
        ax.add_artist(Ellipse(xy=(x, y), width=fwaxmaj, height=fwaxmin, angle=inclination + 90, ec='limegreen', fc='none'))
        ax.set_xlim(xmin - .5, xmin + zoom.shape[1] - .5)
        ax.set_ylim(ymin + zoom.shape[0] - .5, ymin - .5)
        ax.set_title(f'{iii}', fontsize=6, pad=1)
    fig.suptitle(f'{npeaks} most intense peaks', fontsize=9)
    fig.tight_layout()
    return fig


def plot_nbpeaks_map(summary: dict, cfg: dict, gradient: bool = True):
    """map of nb of peaks per image (and of its gradient: changes of the pattern, e.g. grain boundaries)"""
    import matplotlib.pyplot as plt
    scan = cfg['scan']
    nbpeaks2D = summary['nbpeaks2D']
    if nbpeaks2D is None:
        print('no mapdims in the configuration: no map')
        return None
    maps = [('nb of peaks', nbpeaks2D)]
    if gradient:
        import scipy.ndimage as scind
        filled = np.nan_to_num(nbpeaks2D)
        grad = np.hypot(scind.sobel(filled, axis=0, mode='nearest'), scind.sobel(filled, axis=1, mode='nearest'))
        # images not treated (nan) and their neighbours: no gradient
        grad[scind.binary_dilation(np.isnan(nbpeaks2D), structure=np.ones((3, 3)))] = np.nan
        maps.append(('gradient of nb of peaks (Sobel)', grad))
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.5), squeeze=False)
    for ax, (title, data) in zip(axes.flat, maps):
        im = ax.imshow(data, origin='lower')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{cfg['name']}: {title}", fontsize=9)
        ax.set_xlabel(scan['fast_motor'])
        ax.set_ylabel(scan['slow_motor'])
        ax.format_coord = lambda x, y: GT.format_getimageindex_imshow(x, y, scan['invmapdims'])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
#  BATCH JOBS ON SLURM (ESRF cluster): machines, submission and monitoring of indexing_batch
# --------------------------------------------------------------------------------------
def prepare_slurm_job(cfg: dict, params: PeakSearchParams, listfiles: List[str], machine: str = 'hpc6',
                      nb_cpus: int = 96, time: str = '01:00:00', mem_per_cpu: str = '2000M', nchunks: int = 1,
                      job_name: Optional[str] = None, jobs_folder: Optional[Union[str, Path]] = None,
                      env_setup: Optional[List[str]] = None, python: Optional[str] = None,
                      mail_user: Optional[str] = None) -> dict:
    """write a job folder with parameters and slurm script to search the peaks of listfiles with params

    The job uses exactly `params` and `listfiles` (including changes made in the notebook), saved in job_params.pickle
    machine: key of SLURM_MACHINES (partition, constraint, cpus per node), see IB.choose_slurm_machine()
    nchunks > 1: slurm job array of nchunks jobs, each one analysing 1/nchunks of listfiles
    jobs_folder: default <parent folder of params.dat_folder>/slurm_jobs. Job folder: <jobs_folder>/<job_name>_<date>
    env_setup, python, mail_user: see indexing_batch.prepare_slurm_job()

    Then: IB.submit_slurm_job(job), IB.slurm_job_status(job), IB.load_slurm_results(job)
    (also available as PB.submit_slurm_job ...)
    Returns job dict (also written in <job folder>/job.json)
    """
    import json
    nb_cpus, mem_per_cpu, nchunks = IB._check_slurm_resources(machine, nb_cpus, mem_per_cpu, nchunks, len(listfiles))
    job_name = job_name or 'peaksearch'
    jobs_folder = Path(jobs_folder) if jobs_folder else Path(params.dat_folder).parent / 'slurm_jobs'
    jobdir = IB._new_jobdir(jobs_folder, job_name)

    params_job = dataclasses.replace(params, writefiles=True)
    with open(jobdir / 'job_params.pickle', 'wb') as f:
        pickle.dump({'cfg': cfg, 'params': params_job, 'listfiles': [str(ff) for ff in listfiles]}, f)
    readable = IB._to_plain(dataclasses.asdict(params_job))
    with open(jobdir / 'job_params.txt', 'w') as f:
        f.write(f"# parameters of job {jobdir.name} (config {cfg['configfile']})\n")
        f.write(f"# {len(listfiles)} images: {Path(listfiles[0]).name} ... {Path(listfiles[-1]).name}\n")
        f.write(yaml.safe_dump(json.loads(json.dumps(readable, default=str)), sort_keys=False))

    script = IB._write_slurm_script(jobdir, 'LaueTools.peaksearch_batch', job_name, machine, nb_cpus, time,
                                    mem_per_cpu, nchunks, env_setup=env_setup, python=python, mail_user=mail_user)
    job = {'jobdir': str(jobdir), 'script': str(script), 'command': f'sbatch {script}', 'machine': machine,
           'nb_cpus': nb_cpus, 'nchunks': nchunks, 'nfiles': len(listfiles), 'filetype': 'image', 'job_id': None,
           'dat_folder': str(params_job.dat_folder)}
    IB._write_job(job)
    print(f"job folder: {jobdir}\n{len(listfiles)} images, {nchunks} job(s) of {nb_cpus} cpus on {machine}, "
          f"time limit {time}\npeak lists -> {params_job.dat_folder}")
    print(f"to submit from a terminal (jupyter-slurm):\n    {job['command']}")
    return job


def benchmark(listfiles: List[str], params: PeakSearchParams, nb_images: Optional[int] = None,
              nb_cpus: Optional[int] = None) -> float:
    """cpu time (s) per image of a short local batch (on nb_images files taken over listfiles). No file is written"""
    nb_cpus = nb_cpus or available_cpus()
    nb_images = min(nb_images or 2 * nb_cpus, len(listfiles))
    sample = [listfiles[i] for i in np.linspace(0, len(listfiles) - 1, nb_images).astype(int)]
    t0 = time.time()
    run_multiprocessing(sample, dataclasses.replace(params, writefiles=False, overwrite=True), nb_cpus=nb_cpus)
    sec_cpu_per_image = (time.time() - t0) * nb_cpus / nb_images
    print(f'{sec_cpu_per_image:.2f} s x cpu per image')
    return sec_cpu_per_image


# --------------------------------------------------------------------------------------
#  COMMAND LINE (used by slurm jobs)
# --------------------------------------------------------------------------------------
def main(argv=None):
    """python -m LaueTools.peaksearch_batch --job <job folder> [--chunk i --nchunks n] [--ncpus N]
    python -m LaueTools.peaksearch_batch --config <config.yaml> [--image-range start stop] [--ncpus N] [--overwrite]
    """
    import argparse
    import socket
    parser = argparse.ArgumentParser(description='LaueTools: peak search of images (multiprocessing)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--job', help='job folder written by prepare_slurm_job()')
    source.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--ncpus', type=int, default=None, help='nb of processes (default: allocated cpus)')
    parser.add_argument('--chunk', type=int, default=0, help='index of the part of the files to analyse')
    parser.add_argument('--nchunks', type=int, default=1, help='nb of parts')
    parser.add_argument('--image-range', type=int, nargs=2, default=None, help='[--config] first and last image index')
    parser.add_argument('--overwrite', action='store_true', help='[--config] overwrite results pickle file')
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
        listfiles = select_images(cfg, image_range=args.image_range)
    if args.nchunks > 1:
        listfiles = [str(ff) for ff in np.array_split(np.array(listfiles, dtype=object), args.nchunks)[args.chunk]]
    nb_cpus = args.ncpus or available_cpus()
    print(f'{socket.gethostname()}: {len(listfiles)} images (part {args.chunk + 1}/{args.nchunks}), '
          f'{nb_cpus} processes\npeak lists -> {params.dat_folder}', flush=True)

    allresults = run_multiprocessing(listfiles, params, nb_cpus=nb_cpus, progress_interval=60)
    summary = summarize_results(allresults, cfg['scan']['mapdims'])
    if args.job:
        # name read by IB.slurm_job_status() and IB.load_slurm_results()
        filepath = Path(args.job) / f'allresults_part{args.chunk:03d}.pickle'
        save_results(cfg, params, allresults, summary, filepath=filepath, overwrite=True, nb_cpus=nb_cpus)
    else:
        save_results(cfg, params, allresults, summary, overwrite=args.overwrite, nb_cpus=nb_cpus)


if __name__ == '__main__':
    main()
