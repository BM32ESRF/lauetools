# -*- coding: utf-8 -*-
"""
Module of LaueTools project: complete analysis of 2D maps of Laue patterns, from the list of scans of a BLISS
experiment to orientation and grain maps, with few settings (notebooks/quickstart/laue_maps_quickstart.ipynb).

Steps (each one uses the dedicated modules, with YAML configuration files written in PROCESSED_DATA):

1. list the maps of the experiment (BLISS hdf5 file)         list_maps(), ScanSelector, scan_info()
2. configuration files of peak search and indexation           make_configs()
3. peak search (images -> .cor files, .dat files without calibration) run_step('peaksearch', ...)   (peaksearch_batch)
4. indexation & refinement (.cor -> .fit files)                run_step('indexing', ...)     (indexing_batch)
5. maps: quality, orientation (IPF), grains (segmentation)     load_fitfiles(), quality_maps(), segmentation()
   figures saved next to the results (and in GALLERY)          savefigresults()
   all maps: peak search, browser of the saved figures         peaksearch_maps(), FigureBrowser

Each step runs on this machine (multiprocessing) or as a job on the ESRF cluster (SLURM, e.g. magnifix partition).
Steps already done are skipped (results are reloaded), unless redo=True: running the notebook again is safe.
Results are written next to the images, with RAW_DATA replaced by PROCESSED_DATA:
    <PROCESSED_DATA>/<sample>/<dataset>/scan0001/configs/   peaksearch.yaml, indexing.yaml
                                                 /corfiles/  .cor files (.dat: output write_dat)
                                                 /fitfiles_<material>/  .fit files
                                                 /slurm_jobs/  (jobs on the cluster)
"""
import functools
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from LaueTools import generaltools as GT
from LaueTools import dict_LaueTools as DictLT
from LaueTools import peaksearch_batch as PB
from LaueTools import indexing_batch as IB

MAP_COMMANDS = ('amesh', 'dmesh', 'fscan2d')
# image file names of the detectors: CCDLabel: (prefix, suffix)
DETECTOR_FILES = {'EIGER_4MCdTe': ('eiger4m_', '.h5'), 'sCMOS': ('img_', '.tif')}
# peak search parameters of new configurations for each detector (values used at BM32 with the LaueTools GUI)
DETECTOR_PEAKSEARCH = {'EIGER_4MCdTe': {'boxsize': 15, 'PixelNearRadius': 10, 'npixels': 5}}
# cpu time per image (s) to estimate the time limit of the cluster jobs (generous: the time limit is not a cost)
SEC_CPU_PER_IMAGE = {'peaksearch': 3., 'indexing': 15.}
# seconds between two updates of the progress of a cluster job (run_step(..., poll=...)), e.g. MW.SLURM_POLL = 10
SLURM_POLL = 30.
_GALLERY_NOT_WRITABLE = set()   # GALLERY folders without write permission (warned once, see savefigresults())


# --------------------------------------------------------------------------------------
#  WHERE AM I ?
# --------------------------------------------------------------------------------------
def slurm_available() -> bool:
    """True if jobs can be submitted to the cluster from here (sbatch command, e.g. jupyter-slurm session)"""
    return shutil.which('sbatch') is not None


def environment(execution: str = 'auto', verbose: bool = True) -> str:
    """print machine, cpus and SLURM availability. Return the execution mode: 'local' or 'slurm'"""
    if execution not in ('auto', 'local', 'slurm'):
        raise ValueError("EXECUTION must be 'auto', 'local' or 'slurm'")
    slurm = slurm_available()
    if execution == 'auto':
        execution = 'slurm' if slurm else 'local'
    elif execution == 'slurm' and not slurm:
        GT.printred('EXECUTION = "slurm" but sbatch is not available here: use a jupyter-slurm session '
                    '(https://jupyter-slurm.esrf.fr) or EXECUTION = "local"')
    if verbose:
        print(f'machine: {socket.gethostname()}, {IB.available_cpus()} cpus available for this notebook')
        print(f"cluster (SLURM): {'available' if slurm else 'not available from this machine'}")
        GT.printgreen(f"-> computations will run {'on the cluster (SLURM)' if execution == 'slurm' else 'on this machine'}")
    return execution


def find_material(text: str):
    """print the materials of the LaueTools dictionary whose name contains text (case insensitive)"""
    found = {k: v for k, v in DictLT.dict_Materials.items() if text.lower() in k.lower()}
    for key, value in found.items():
        print(f'{key:20s} lattice {value[1]}, extinctions {value[2]}')
    if not found:
        GT.printyellow(f'no material containing "{text}" (see LaueTools/dict_LaueTools.py or materials.yaml)')


# --------------------------------------------------------------------------------------
#  SCANS OF THE EXPERIMENT
# --------------------------------------------------------------------------------------
def find_logfile(raw_data_folder: Union[str, Path]) -> Path:
    """BLISS hdf5 file of the experiment (list of all scans) in raw_data_folder, e.g. <proposal>_bm32.h5"""
    from LaueTools import blissdatafolderstructure as BF
    raw_data_folder = Path(raw_data_folder)
    if not raw_data_folder.is_dir():
        raise FileNotFoundError(f'folder not found: {raw_data_folder}\n'
                                'set RAW_DATA_FOLDER, e.g. /data/visitor/<proposal>/bm32/<date>/RAW_DATA')
    path, found = BF.findmasterh5file(str(raw_data_folder))
    if not found:
        raise FileNotFoundError(f'no .h5 file in {raw_data_folder}: is it the RAW_DATA folder of the experiment?')
    return Path(path)


def guess_detector(image_folder: Union[str, Path]):
    """(CCDLabel, prefix, suffix) from the image files of a scan folder, (None, None, None) if no known image"""
    try:
        with os.scandir(image_folder) as entries:
            for entry in entries:
                for ccdlabel, (prefix, suffix) in DETECTOR_FILES.items():
                    if entry.name.startswith(prefix) and entry.name.endswith(suffix):
                        return ccdlabel, prefix, suffix
    except OSError:
        pass
    return None, None, None


def processed_folder(image_folder: Union[str, Path], output_root: Optional[Union[str, Path]] = None) -> Path:
    """folder of the results of a scan: image folder with RAW_DATA replaced by PROCESSED_DATA

    output_root: other root folder of the results (the path of the image folder after RAW_DATA is kept)
    """
    parts = Path(image_folder).parts
    if 'RAW_DATA' not in parts:
        if output_root is None:
            raise ValueError(f'{image_folder} is not in a RAW_DATA folder: set OUTPUT_ROOT (folder of the results)')
        return Path(output_root) / Path(image_folder).name
    k = len(parts) - 1 - parts[::-1].index('RAW_DATA')
    if output_root is None:
        return Path(*parts[:k], 'PROCESSED_DATA', *parts[k + 1:])
    return Path(output_root, *parts[k + 1:])


def _count_files(folder: Path, prefix: str, suffix: str) -> int:
    try:
        with os.scandir(folder) as entries:
            return sum(1 for e in entries if e.name.startswith(prefix) and e.name.endswith(suffix))
    except OSError:
        return 0


def _count_results(folder: Path, suffix: str) -> int:
    """nb of files ending with suffix in folder and in its subfolders (layout with nbfiles_per_folder)"""
    n = 0
    try:
        with os.scandir(folder) as entries:
            for e in entries:
                if e.name.endswith(suffix):
                    n += 1
                elif '_previous_' not in e.name and e.is_dir(follow_symlinks=False):
                    n += _count_files(Path(e.path), '', suffix)
    except OSError:
        pass
    return n


def _status(workdir: Path, count_files: bool = False) -> str:
    """results already in workdir: 'peaks' (corfiles), 'fit:<material>' (fitfiles_<material>)
    count_files: with the nb of .cor files and of indexed images (_g0.fit files), e.g. 'peaks (22801 .cor)'"""
    status = []
    if (workdir / 'corfiles').is_dir():
        status.append(f"peaks ({_count_results(workdir / 'corfiles', '.cor')} .cor)" if count_files else 'peaks')
    if workdir.is_dir():
        for p in sorted(workdir.glob('fitfiles_*')):
            if p.is_dir() and '_previous_' not in p.name:
                fit = f"fit:{p.name[len('fitfiles_'):]}"
                status.append(f'{fit} ({_count_results(p, "_g0.fit")} _g0.fit)' if count_files else fit)
    return ', '.join(status)


def raw_data_folder(exp_id: Optional[str] = None, parent: Union[str, Path] = '/data/visitor',
                    date: Optional[str] = None, beamline: str = 'bm32',
                    raw_data_folder: Optional[Union[str, Path]] = None, verbose: bool = True) -> Path:
    """RAW_DATA folder of an experiment: <parent>/<exp_id>/<beamline>/<date>/RAW_DATA (ESRF), e.g. parent
    '/data/visitor' (user experiments) or '/data/bm32/inhouse'. date: None: the only date folder of the experiment,
    or the most recent one if there are several (printed: set the date to choose another one).
    raw_data_folder: given folder (other organisation of the data), returned as is"""
    if raw_data_folder:
        return Path(raw_data_folder)
    if not exp_id:
        raise ValueError('set EXP_ID (experiment, e.g. a321220) or RAW_DATA_FOLDER')
    base = Path(parent) / exp_id / beamline
    if not base.is_dir() and (Path(parent) / exp_id.lower() / beamline).is_dir():
        base = Path(parent) / exp_id.lower() / beamline
    if not base.is_dir():
        raise FileNotFoundError(f'no folder {base}: check EXP_ID and PARENT_PATH')
    dates = sorted(d.name for d in base.iterdir() if (d / 'RAW_DATA').is_dir())
    if not dates:
        raise FileNotFoundError(f'no <date>/RAW_DATA folder in {base}')
    if date is not None and str(date) not in dates:
        raise FileNotFoundError(f'no RAW_DATA folder for the date {date} in {base}: dates {dates}')
    chosen = str(date) if date is not None else dates[-1]
    if verbose:
        if date is None and len(dates) > 1:
            GT.printyellow(f'several dates in {base}: {dates}, the most recent one is used (EXP_DATE to choose)')
        print(f'RAW_DATA_FOLDER = {base / chosen / "RAW_DATA"}')
    return base / chosen / 'RAW_DATA'


# keys of a scan filter (SCAN_FILTER of the quickstart notebook, see scan_filter())
SCAN_FILTER_KEYS = {'contains': '', 'include': [], 'logic': 'AND', 'exclude': [], 'prefix': None,
                    'only_success': False, 'min_points': None, 'max_points': None, 'last': None}


def _words_match(text: str, pattern: Optional[str]) -> bool:
    """all the words of pattern are in text (case insensitive); a word starting with '-' must not be in it"""
    text = text.lower()
    for word in (pattern or '').lower().split():
        if word.startswith('-') and len(word) > 1:
            if word[1:] in text:
                return False
        elif word not in text:
            return False
    return True


def scan_filter(spec=None) -> dict:
    """filter of the scans of list_maps() from a text or a dict (keys of SCAN_FILTER_KEYS):
    - text: words of the dataset name (case insensitive), a word starting with '-': without it, e.g. 'map2D -align'
    - dict:
        'contains': text as above
        'include': text or list of texts of the scan name in the .h5 file (case sensitive), 'logic': 'AND' (all of
            them, default) or 'OR' (at least one)
        'exclude': text or list of texts: scans whose name contains one of them are not listed (case sensitive)
        'prefix': scan name starting with this text (e.g. the sample name)
        'only_success': True: only the scans ended with SUCCESS (not aborted, not failed)
        'min_points', 'max_points': nb of map points (e.g. min_points 100: no small alignment maps)
        'last': only the last scans of the experiment (nb of scans)
    All the conditions except 'min_points', 'max_points' are applied while reading the .h5 file (faster for large
    experiments)"""
    if spec is None or isinstance(spec, str):
        spec = {'contains': spec or ''}
    if not isinstance(spec, dict):
        raise TypeError(f'SCAN_FILTER must be a text or a dict, not {type(spec).__name__}')
    unknown = set(spec) - set(SCAN_FILTER_KEYS)
    if unknown:
        raise KeyError(f'unknown keys {sorted(unknown)} of SCAN_FILTER: {list(SCAN_FILTER_KEYS)}')
    flt = {**SCAN_FILTER_KEYS, **spec}
    for key in ('include', 'exclude'):
        flt[key] = [flt[key]] if isinstance(flt[key], str) else list(flt[key] or [])
    if str(flt['logic']).upper() not in ('AND', 'OR'):
        raise ValueError(f"SCAN_FILTER 'logic' must be 'AND' or 'OR', not {flt['logic']!r}")
    flt['logic'] = str(flt['logic']).upper()
    return flt


def list_maps(logfile: Union[str, Path], contains=None, only_success: bool = False,
              output_root: Optional[Union[str, Path]] = None, count_images: bool = True, verbose: int = 1):
    """table (pandas DataFrame) of the 2D maps (amesh, dmesh, fscan2d) of the experiment

    contains: filter of the maps (SCAN_FILTER): text (words of the dataset name, case insensitive, '-word': without
        it) or dict of conditions, see scan_filter()
    only_success: only the maps ended with SUCCESS (also 'only_success' of the filter)
    count_images: count the image files of each map (column 'images': found / expected)
    Column 'done': results already in PROCESSED_DATA (peaks: peak search, fit:<material>: indexation)
    index of the table: scan index in the hdf5 file (used by scan_info()); it depends on the filter: the dataset
    name (column 'dataset') can be used instead (see scan_index())
    """
    import pandas as pd
    from LaueTools import logfile_reader as iohdf5
    t0 = time.time()
    flt = scan_filter(contains)
    only_success = only_success or bool(flt['only_success'])
    lf = iohdf5.H5file(str(logfile))
    lf.getscans(verbose=0, n_last=int(flt['last']) if flt['last'] else 100000, node_prefix=flt['prefix'],
                node_substrings=flt['include'] or None, substrings_logic=flt['logic'],
                exclude_substrings=flt['exclude'] or None, only_endreason_SUCCESS=only_success,
                node_filter=(lambda name: _words_match(name, flt['contains'])) if flt['contains'] else None)
    df = lf.dfallscans
    df = df[df['scantype'].isin(MAP_COMMANDS)]
    if only_success:
        df = df[df['endreason'] == 'SUCCESS']
    if flt['contains']:
        df = df[[_words_match(name, flt['contains']) for name in df['sample_dataset_scanindex']]]

    rows = []
    for idx, row in df.iterrows():
        try:
            dc = iohdf5.read_fullcommand(row['fullcommand'])
        except ValueError:
            continue
        nfast, nslow = dc['npts_fast'], dc['npts_slow']
        if (flt['min_points'] and nfast * nslow < flt['min_points']) or (
                flt['max_points'] and nfast * nslow > flt['max_points']):
            continue
        image_folder = Path(row['imagefolder'])
        ccdlabel, prefix, suffix = guess_detector(image_folder)
        nbimages = _count_files(image_folder, prefix, suffix) if (count_images and prefix) else None
        try:
            done = _status(processed_folder(image_folder, output_root))
        except ValueError:
            done = ''
        rows.append({'index': int(idx),
                     'dataset': row['sample_dataset_scanindex'],
                     'start': row['start_time'].strftime('%Y-%m-%d %H:%M'),
                     'hours': row['duration_hours'],
                     'map': f'{nfast} x {nslow}',
                     'images': f'{nbimages}/{nfast * nslow}' if nbimages is not None else f'?/{nfast * nslow}',
                     'detector': ccdlabel or '?',
                     'end': row['endreason'],
                     'done': done,
                     'command': row['fullcommand'],
                     'image_folder': str(image_folder),
                     'hdf5_file': row['localhdf5file']})
    maps = pd.DataFrame(rows)
    if len(maps):
        maps = maps.set_index('index')
    maps.attrs['output_root'] = output_root   # (refresh_done())
    if verbose:
        print(f'{len(maps)} maps found in {Path(logfile).name} ({time.time() - t0:.0f} s)')
    return maps


def refresh_done(maps, output_root: Optional[Union[str, Path]] = None, count_files: bool = False,
                 nb_threads: int = 16, verbose: bool = True):
    """update the column 'done' of the table of list_maps() (results in PROCESSED_DATA: peaks, fit:<material>),
    e.g. after analyses done since the table was made. output_root: None: that of list_maps()
    Without count_files: only 1 or 2 folders read per map (fast, also for hundreds of maps).
    count_files: also the nb of .cor files and of indexed images (_g0.fit files): all the files of the results
    folders are listed (nb_threads folders at a time), e.g. ~0.1 s for 20000 files"""
    if not len(maps):
        return maps
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    output_root = maps.attrs.get('output_root') if output_root is None else output_root

    def status(index):
        try:
            return _status(processed_folder(maps.at[index, 'image_folder'], output_root), count_files)
        except ValueError:
            return maps.at[index, 'done']
    with ThreadPoolExecutor(nb_threads if count_files else 1) as pool:
        maps['done'] = list(pool.map(status, maps.index))
    if verbose and time.time() - t0 > 2:
        print(f"column 'done' of {len(maps)} maps updated ({time.time() - t0:.0f} s)")
    return maps


def scan_index(maps, index) -> Optional[int]:
    """index of a map in the table of list_maps(): index of the table (int), or dataset name (str, e.g.
    'MH73_HT_p5_fishing_1': does not depend on the filter of the table). None: None"""
    if index is None or isinstance(index, (int, np.integer)):
        return None if index is None else int(index)
    found = [int(i) for i in maps.index if maps.loc[i, 'dataset'] == str(index)]
    if not found:
        raise KeyError(f'no map {index!r} in the table (filter SCAN_FILTER?)')
    return found[-1]


def show_maps(maps, nmax: Optional[int] = None):
    """display the table of maps (main columns)"""
    from IPython.display import display
    import pandas as pd
    if not len(maps):
        GT.printred('no map: change SCAN_FILTER (or EXP_ID, EXP_DATE)')
        return
    with pd.option_context('display.max_rows', nmax or len(maps), 'display.max_colwidth', 70, 'display.width', 250):
        display(maps[['dataset', 'start', 'hours', 'map', 'images', 'detector', 'end', 'done', 'command']])


class ScanSelector:
    """menu to choose a map in the table of list_maps(), with a filter (words of the menu lines, '-word': without
    it) and a checkbox 'only SUCCESS' (maps not aborted). The selected scan index is ScanSelector.index
    index: index of the table or dataset name (see scan_index())
    refresh: column 'done' of the table updated first (refresh_done())
    count_files (also checkbox 'count files'): column 'done' with the nb of .cor files and of indexed images"""

    def __init__(self, maps, index=None, only_success: bool = False, refresh: bool = True,
                 count_files: bool = False):
        self.maps = refresh_done(maps, count_files=count_files) if refresh else maps   # column 'done' up to date
        index = scan_index(maps, index)
        self.index = index if index is not None else (int(maps.index[-1]) if len(maps) else None)
        if not len(maps):
            GT.printred('no map in the table: change SCAN_FILTER')
            return
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print('ipywidgets not installed: set SCAN_INDEX in SETTINGS (index of the table or dataset name)')
            return
        self._make_labels()
        self.filter = widgets.Text(description='filter:', placeholder="words of the menu lines, '-word': without it",
                                   continuous_update=False, layout=widgets.Layout(width='50%'))
        self.success = widgets.Checkbox(value=bool(only_success), description='only SUCCESS (not aborted)',
                                        indent=False)
        self.count_files = widgets.Checkbox(value=bool(count_files), description='count files (.cor, _g0.fit)',
                                            indent=False)
        self.count = widgets.HTML()
        self.menu = widgets.Select(options=[], rows=min(max(len(self.labels), 3), 12),
                                   layout=widgets.Layout(width='98%'))
        self.menu.add_class('lt-scan-menu')
        style = widgets.HTML('<style>.lt-scan-menu select {font-family: monospace; font-size: 12px}</style>')
        self.info = widgets.HTML()
        self.menu.observe(self._on_change, names='value')
        self.filter.observe(self._apply_filter, names='value')
        self.success.observe(self._apply_filter, names='value')
        self.count_files.observe(self._on_count_files, names='value')
        self._apply_filter(None)
        display(widgets.VBox([style, widgets.HBox([self.filter, self.success, self.count_files, self.count]),
                              self.menu, self.info]))

    def _make_labels(self):
        self.labels = [(f"{i:5d} | {r['dataset']} | {r['map']} | images {r['images']} | {r['end']}"
                        + (f" | done: {r['done']}" if r['done'] else ''), int(i)) for i, r in self.maps.iterrows()]

    def _on_count_files(self, change):
        self.count.value = '&nbsp;counting the files ...'
        t0 = time.time()
        refresh_done(self.maps, count_files=self.count_files.value, verbose=False)
        self._make_labels()
        self._apply_filter(None)
        if self.count_files.value:
            self.count.value += f' (files counted in {time.time() - t0:.1f} s)'

    def _apply_filter(self, change):
        options = [o for o in self.labels if _words_match(o[0], self.filter.value)
                   and (not self.success.value or self.maps.loc[o[1], 'end'] == 'SUCCESS')]
        self.count.value = f'&nbsp;{len(options)} / {len(self.labels)} maps'
        current = self.index   # (new options select their first one)
        self.menu.options = options
        if current in [o[1] for o in options]:
            self.menu.value = current
        elif options:
            self.menu.value = options[-1][1]

    def _on_change(self, change):
        if self.menu.value is None:
            return
        self.index = int(self.menu.value)
        r = self.maps.loc[self.index]
        self.info.value = (f"<b>{r['dataset']}</b> (SCAN_INDEX = {self.index} or '{r['dataset']}'): "
                           f"<code>{r['command']}</code><br>{r['image_folder']}<br>then run the next cells")


def find_configs(exp_id: Optional[str] = None, parent: Union[str, Path] = '/data/visitor', beamline: str = 'bm32',
                 dates=None, pattern: str = 'indexing*.yaml', root: Optional[Union[str, Path]] = None,
                 max_depth: int = 3, verbose: bool = True):
    """configuration files of the maps of an experiment: <configs folder>/<pattern> in the folders of
    <parent>/<exp_id>/<beamline>/<date>/PROCESSED_DATA/ (e.g. <sample>/<dataset>/scan0001/configs/indexing.yaml)

    dates: None: all the dates (session folders) of the experiment, or a date / list of dates ('20261007')
    root: search in this folder instead (e.g. another PROCESSED_DATA folder); exp_id, parent, beamline not used
    max_depth: depth of the folders (below PROCESSED_DATA) containing a 'configs' folder; the deeper folders (.cor,
        .fit files) are never listed
    Returns a table (pandas DataFrame, most recently modified first) with the columns 'date', 'dataset' (folder
    relative to PROCESSED_DATA), 'file', 'material', 'map', 'fit' (nb of .fit files in fit_folder), 'modified',
    'path', 'fit_folder', for ConfigSelector()"""
    import datetime
    import pandas as pd
    import yaml
    t0 = time.time()
    if root is not None:
        roots = [(None, Path(root))]
    else:
        if not exp_id:
            raise ValueError('exp_id (experiment, e.g. ma7342) or root is needed')
        base = Path(parent) / exp_id / beamline
        if not base.is_dir() and (Path(parent) / exp_id.lower() / beamline).is_dir():
            base = Path(parent) / exp_id.lower() / beamline
        if not base.is_dir():
            raise FileNotFoundError(f'no folder {base}')
        dates = [dates] if isinstance(dates, (str, int)) else dates
        sessions = sorted(d for d in base.iterdir() if d.is_dir() and (dates is None or d.name in map(str, dates)))
        roots = [(d.name, d / 'PROCESSED_DATA') for d in sessions if (d / 'PROCESSED_DATA').is_dir()]
        if dates is not None and len(roots) < len(dates):
            GT.printyellow(f"date(s) without PROCESSED_DATA in {base}: "
                           f"{sorted(set(map(str, dates)) - {r[0] for r in roots})}")

    def configs_folders(folder: Path, depth: int):
        """'configs' folders below folder (depth: levels of folders listed)"""
        try:
            with os.scandir(folder) as entries:
                subdirs = [Path(e.path) for e in entries if e.is_dir()]
        except OSError:
            return
        for sub in sorted(subdirs):
            if sub.name == 'configs':
                yield sub
            elif depth > 0 and not sub.name.startswith('.'):
                yield from configs_folders(sub, depth - 1)

    rows = []
    for date, processed in roots:
        for cfgdir in configs_folders(processed, max_depth):
            for path in sorted(cfgdir.glob(pattern)):
                try:
                    with open(path, 'r') as f:
                        usercfg = yaml.safe_load(f) or {}
                except (OSError, yaml.YAMLError):
                    continue
                if not isinstance(usercfg, dict) or 'indexing' not in usercfg and 'output' not in usercfg:
                    continue
                fit_folder = (usercfg.get('output') or {}).get('fit_folder')
                if fit_folder:
                    fit_folder = Path(os.path.expandvars(os.path.expanduser(str(fit_folder))))
                    fit_folder = fit_folder if fit_folder.is_absolute() else (path.parent / fit_folder).resolve()
                nbfit = None
                if fit_folder and fit_folder.is_dir():
                    with os.scandir(fit_folder) as entries:
                        nbfit = sum(1 for e in entries if e.name.endswith('.fit'))
                mapdims = (usercfg.get('scan') or {}).get('mapdims')
                rows.append({'date': date or '',
                             'dataset': str(cfgdir.parent.relative_to(processed)),
                             'file': path.name,
                             'material': (usercfg.get('material') or {}).get('key_material') or '',
                             'map': f'{mapdims[0]} x {mapdims[1]}' if mapdims else '?',
                             'fit': nbfit if nbfit is not None else '-',
                             'modified': datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                             'path': str(path), 'fit_folder': str(fit_folder) if fit_folder else ''})
    configs = pd.DataFrame(rows, columns=['date', 'dataset', 'file', 'material', 'map', 'fit', 'modified', 'path',
                                          'fit_folder'])
    configs = configs.sort_values('modified', ascending=False).reset_index(drop=True)
    if verbose:
        where = root if root is not None else f"{base}/{{{', '.join(r[0] for r in roots)}}}/PROCESSED_DATA"
        print(f'{len(configs)} configuration files {pattern} in {where} ({time.time() - t0:.1f} s)')
    return configs


class ConfigSelector:
    """menu to choose a configuration file in the table of find_configs() (most recently modified first, with a
    filter on the text of the menu). The selected file is ConfigSelector.path (use it as CONFIG)"""

    def __init__(self, configs, path: Optional[str] = None):
        self.configs = configs
        self.path = path if path is not None else (configs['path'].iloc[0] if len(configs) else None)
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print('ipywidgets not installed: copy the path of the table into CONFIG')
            return
        w_date = max([len(str(d)) for d in configs['date']] + [4])
        w_data = max([len(d) for d in configs['dataset']] + [7])
        w_file = max([len(f) for f in configs['file']] + [4])
        self.labels = [(f"{r['date']:<{w_date}} | {r['dataset']:<{w_data}} | {r['file']:<{w_file}} | {r['material']}"
                        f" | {r['map']} | .fit {r['fit']} | {r['modified']}", r['path']) for _, r in configs.iterrows()]
        self.filter = widgets.Text(description='filter:', placeholder='text of the menu (date, dataset, file ...)',
                                   layout=widgets.Layout(width='60%'))
        self.menu = widgets.Select(options=self.labels, rows=min(max(len(self.labels), 3), 15),
                                   layout=widgets.Layout(width='98%', font_family='monospace'))
        self.menu.add_class('lt-config-menu')
        if self.path in configs['path'].values:
            self.menu.value = self.path
        self.info = widgets.HTML()
        self.menu.observe(self._on_change, names='value')
        self.filter.observe(self._on_filter, names='value')
        self._on_change(None)
        style = widgets.HTML('<style>.lt-config-menu select {font-family: monospace; font-size: 12px}</style>')
        display(widgets.VBox([style, self.filter, self.menu, self.info]))

    def _on_filter(self, change):
        text = change['new'].lower()
        options = [o for o in self.labels if text in o[0].lower()] if text else self.labels
        current = self.menu.value
        self.menu.options = options
        if current in [o[1] for o in options]:
            self.menu.value = current

    def _on_change(self, change):
        if self.menu.value is None:
            return
        self.path = self.menu.value
        r = self.configs[self.configs['path'] == self.path].iloc[0]
        self.info.value = (f"<b>CONFIG = '{self.path}'</b><br>.fit files ({r['fit']}): {r['fit_folder']}"
                           f"<br>then run the next cells")


def scan_info(maps, index: int, output_root: Optional[Union[str, Path]] = None) -> dict:
    """all information of a map needed by the analysis (from the table of list_maps()). index: index of the table
    or dataset name (see scan_index())"""
    from LaueTools import logfile_reader as iohdf5
    index = scan_index(maps, index)
    if index not in maps.index:
        raise KeyError(f'scan index {index} is not in the table of maps')
    row = maps.loc[index]
    dc = iohdf5.read_fullcommand(row['command'])
    ccdlabel, prefix, suffix = guess_detector(row['image_folder'])
    if ccdlabel is None:
        raise FileNotFoundError(f"no image (eiger4m_*.h5, img_*.tif) in {row['image_folder']}")
    steps = [abs(dc['fmotmax'] - dc['fmotmin']) / max(dc['fmotnbsteps'], 1) * 1000,
             abs(dc['smotmax'] - dc['smotmin']) / max(dc['smotnbsteps'], 1) * 1000]   # mm -> micron
    nbdigits = 4
    try:   # zero padding from an image file name
        with os.scandir(row['image_folder']) as entries:
            name = next(e.name for e in entries if e.name.startswith(prefix) and e.name.endswith(suffix))
        nbdigits = len(name[len(prefix):-len(suffix)])
    except (StopIteration, OSError):
        pass
    return {'index': int(index), 'name': row['dataset'], 'command': row['command'],
            'image_folder': row['image_folder'], 'hdf5_file': row['hdf5_file'],
            'CCDLabel': ccdlabel, 'prefix': prefix, 'suffix': suffix, 'nbdigits': nbdigits,
            'mapdims': [dc['npts_fast'], dc['npts_slow']], 'fast_motor': dc['fastmotor'],
            'slow_motor': dc['slowmotor'], 'stepsizes': [round(s, 4) for s in steps],
            'workdir': str(processed_folder(row['image_folder'], output_root))}


_EXPERIMENT_FILES = {}   # (raw_data_folder, max_depth): (time, files), see _experiment_files()


def _experiment_files(raw_data_folder: str, max_depth: int = 4, max_age: float = 60.) -> tuple:
    """.det files of the experiment (RAW_DATA and PROCESSED_DATA, up to max_depth subfolders)
    (kept max_age seconds: the folders of the images are long to list, and new .det files are found after max_age)"""
    key = (raw_data_folder, max_depth)
    if key in _EXPERIMENT_FILES and time.time() - _EXPERIMENT_FILES[key][0] < max_age:
        return _EXPERIMENT_FILES[key][1]
    found = []
    roots = [Path(raw_data_folder)]
    if Path(raw_data_folder).name == 'RAW_DATA':
        roots.append(Path(raw_data_folder).parent / 'PROCESSED_DATA')
    for root in roots:
        root_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).parts) - root_depth >= max_depth:
                dirnames[:] = []
            found += [Path(dirpath) / f for f in filenames if f.endswith('.det')]
    _EXPERIMENT_FILES[key] = (time.time(), tuple(found))
    return tuple(found)


def _most_recent_first(files) -> List[Path]:
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


@functools.lru_cache(maxsize=256)
def _dataset_start(dataset_folder: str):
    """start of the acquisition of a BLISS dataset (datetime): earliest start_time of the scans of the .h5 files of
    the dataset folder RAW_DATA/<sample>/<dataset>/. None if unknown"""
    import datetime
    import h5py
    times = []
    for h5file in Path(dataset_folder).glob('*.h5'):
        try:
            with h5py.File(h5file, 'r', locking=False) as f:
                for entry in f.values():
                    if isinstance(entry, h5py.Group) and 'start_time' in entry:
                        value = entry['start_time'][()]
                        times.append(datetime.datetime.fromisoformat(
                            value.decode() if isinstance(value, bytes) else str(value)))
        except (OSError, ValueError):
            continue
    return min(times) if times else None


def acquisition_time(path: Union[str, Path]):
    """date of the acquisition of the data a file (e.g. a .det calibration file) or folder belongs to: start of its
    BLISS dataset (RAW_DATA/<sample>/<dataset>/<dataset>.h5, start_time), for a path in RAW_DATA or in the same
    folders of PROCESSED_DATA (or GALLERY). Independent of the dates of the files (restored data, file written
    later). None if the path is not in a dataset folder"""
    parts = Path(path).parts
    roots = [i for i, part in enumerate(parts) if part in ('RAW_DATA', 'PROCESSED_DATA', 'GALLERY')]
    if not roots or len(parts) < roots[-1] + 3:   # <root>/<sample>/<dataset>/...
        return None
    k = roots[-1]
    return _dataset_start(str(Path(*parts[:k], 'RAW_DATA', *parts[k + 1:k + 3])))


def find_detfiles(raw_data_folder: Union[str, Path], max_depth: int = 4, order: str = 'acquisition',
                  before=None, verbose: bool = False, max_age: float = 60.) -> List[Path]:
    """calibration .det files of the experiment (RAW_DATA and PROCESSED_DATA), most recent first

    order 'acquisition': by date of the acquisition of the calibration dataset they belong to (acquisition_time():
        .det file in the folder of the calibration scan, in RAW_DATA or PROCESSED_DATA), not by date of the file
        (restored data, or calibration refined later). Files outside dataset folders come after, by file date.
    order 'file': by date of the files (modification)
    before: date (datetime or 'YYYY-mm-dd HH:MM') of a map: calibrations acquired before it first (most recent
        first), then the ones acquired after it (closest first)
    verbose: print the files with their dates
    max_age: the list of files of the experiment is kept max_age seconds (new files found after)
    """
    import datetime
    files = [f for f in _experiment_files(str(raw_data_folder), max_depth, max_age) if f.suffix == '.det']
    if order == 'file':
        ordered = _most_recent_first(files)
        dates = {f: datetime.datetime.fromtimestamp(f.stat().st_mtime) for f in ordered}
    elif order == 'acquisition':
        dates = {f: acquisition_time(f) for f in files}
        dated = [f for f in files if dates[f] is not None]
        undated = _most_recent_first(f for f in files if dates[f] is None)
        if before is not None:
            if isinstance(before, str):
                before = datetime.datetime.fromisoformat(before)

            def naive(d):   # comparison with a date without time zone
                return d.replace(tzinfo=None) if before.tzinfo is None else d
            previous = sorted((f for f in dated if naive(dates[f]) <= before), key=dates.get, reverse=True)
            later = sorted((f for f in dated if naive(dates[f]) > before), key=dates.get)
            dated = previous + later
        else:
            dated = sorted(dated, key=dates.get, reverse=True)
        ordered = dated + undated
    else:
        raise ValueError(f"order {order!r}: 'acquisition' or 'file'")
    if verbose:
        for f in ordered:
            d = dates.get(f)
            print(f"{d.strftime('%Y-%m-%d %H:%M') if d else 'no dataset      '}  {f}")
    return ordered


def detfile_for_map(maps, index, max_age: float = 60., verbose: bool = True) -> Optional[Path]:
    """calibration .det file of a map of the table of list_maps(): the last one acquired before the map (start of
    the map, see find_detfiles(before=...)), else the closest one acquired after it, else the most recent file
    without dataset. None if the experiment has no .det file"""
    folder = Path(maps.at[index, 'image_folder'])
    parts = folder.parts
    raw = Path(*parts[:len(parts) - parts[::-1].index('RAW_DATA')]) if 'RAW_DATA' in parts else folder.parent
    start = str(maps.at[index, 'start']).strip() or None
    files = find_detfiles(raw, before=start, max_age=max_age)
    if verbose:
        if files:
            when = acquisition_time(files[0])
            print(f"calibration: {files[0]} (acquired {when.strftime('%Y-%m-%d %H:%M') if when else '?'}"
                  f"{', map started ' + start if start else ''})")
        else:
            GT.printyellow(f'no .det calibration file in {raw}')
    return files[0] if files else None


def find_pspfiles(image_folder: Union[str, Path]) -> List[Path]:
    """peak search parameters files (.psp, saved by the LaueTools GUI) of a scan folder, most recent first

    only the folder of the scan: .psp files of other scans may have been saved for other conditions (or tests)
    """
    return _most_recent_first(Path(image_folder).glob('*.psp'))


# --------------------------------------------------------------------------------------
#  CONFIGURATION FILES
# --------------------------------------------------------------------------------------
def config_paths(info: dict) -> Dict[str, Path]:
    configs = Path(info['workdir']) / 'configs'
    return {'peaksearch': configs / 'peaksearch.yaml', 'indexing': configs / 'indexing.yaml'}


def _differences(cfg: dict, changes: dict, tree: dict) -> dict:
    """changes {parameter name: value} whose value differs from cfg"""
    diff = {}
    for name, value in changes.items():
        node = cfg
        for key in IB._key_path(name, tree):
            node = node.get(key) if isinstance(node, dict) else None
        if (name.replace('_', ' ') in IB._SPOTS_SET_KEYS and value is not None
                and not IB.ISS.is_spots_proposal(value)):
            value = IB.ISS.parse_spots_set(value)   # python syntax of spots sets (5 -> [5])
        if not IB._same(value, node):
            diff[name] = value
    return diff


def make_configs(info: dict, key_material: str, detfile: Union[str, Path], pspfile: Optional[Union[str, Path]] = None,
                 peaksearch: Optional[dict] = None, indexing: Optional[dict] = None,
                 peaksearch_template: Optional[Union[str, Path]] = None,
                 indexing_template: Optional[Union[str, Path]] = None, verbose: bool = True,
                 update: bool = True):
    """write (or update) <workdir>/configs/peaksearch.yaml and indexing.yaml of a map. Returns (pcfg, icfg)

    update=False: only read the existing files (unchanged, whatever the other arguments), e.g. to show the results
    of a map analysed before

    - first call: files written from the templates (default config_template.yaml of LaueTools, or the configuration
      files of another scan, e.g. to process a series of maps with the same parameters)
    - next calls: existing files are kept (with your manual edits); only the parameters given here are updated
    peaksearch, indexing: parameters by name, e.g. {'IntensityThreshold': 300}, {'nbGrainstoFind': 2, 'e_max': 22}
    pspfile: .psp file of the LaueTools GUI whose values are copied in peaksearch.yaml (the YAML file stays the only
        reference of the parameters)
    """
    paths = config_paths(info)
    if detfile is not None:   # (spaces around a path copied in the notebook)
        detfile = Path(str(detfile).strip())
    if not update:
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f'this map has not been analysed (no settings file {missing}): '
                                    "use MODE = 'compute'")
        print(f"settings files of the analysis (unchanged): {paths['peaksearch'].parent}")
        return PB.load_config(paths['peaksearch']), IB.load_config(paths['indexing'])
    workdir = Path(info['workdir'])
    peaksearch = dict(peaksearch or {})
    indexing = dict(indexing or {})
    if pspfile is not None:   # values of the .psp file written in peaksearch.yaml (no link to the .psp file)
        peaksearch = {**PB.read_pspfile(pspfile), **peaksearch}
    fromscan = dict(name=info['name'], description=info['command'], image_folder=info['image_folder'],
                    prefix=info['prefix'], suffix=info['suffix'], nbdigits=info['nbdigits'],
                    CCDLabel=info['CCDLabel'], detfile=str(detfile), hdf5_logfile=info['hdf5_file'],
                    mapdims=info['mapdims'], fast_motor=info['fast_motor'], slow_motor=info['slow_motor'],
                    stepsizes=info['stepsizes'], dat_folder=str(workdir / 'corfiles'),
                    test_image_index=info['mapdims'][0] * info['mapdims'][1] // 2)

    if not paths['peaksearch'].exists():
        changes = dict(fromscan)
        if peaksearch_template is None:   # default parameters of the detector
            changes.update(Saturation_value=None, **DETECTOR_PEAKSEARCH.get(info['CCDLabel'], {}))
        changes.update(peaksearch)
        pcfg = PB.new_config(paths['peaksearch'], template=peaksearch_template, **changes)
    else:
        pcfg = PB.load_config(paths['peaksearch'])
        diff = _differences(pcfg, {'detfile': str(detfile), **peaksearch}, PB.DEFAULT_CONFIG)
        if diff:
            pcfg = PB.save_config(pcfg, paths['peaksearch'], overwrite=True, **diff)
        elif verbose:
            print(f"peak search configuration: {paths['peaksearch']}")

    fit_folder = workdir / f'fitfiles_{key_material}'
    if not paths['indexing'].exists():
        changes = {'key_material': key_material, 'fit_folder': str(fit_folder), **indexing}
        icfg = PB.indexing_config(pcfg, paths['indexing'], template=indexing_template, check=False, **changes)
    else:
        icfg = IB.load_config(paths['indexing'])
        changes = {'key_material': key_material, 'detfile': str(detfile), **indexing}
        if key_material != icfg['material']['key_material']:
            changes['fit_folder'] = str(fit_folder)   # new material: new folder of .fit files
        diff = _differences(icfg, changes, IB.DEFAULT_CONFIG)
        if diff:
            icfg = IB.save_config(icfg, paths['indexing'], overwrite=True, **diff)
        elif verbose:
            print(f"indexation configuration: {paths['indexing']}")
    return pcfg, icfg


# --------------------------------------------------------------------------------------
#  RUN A STEP (local or SLURM), skipped if already done
# --------------------------------------------------------------------------------------
def _results_file(step: str, cfg: dict) -> Path:
    if step == 'peaksearch':
        return Path(cfg['output']['results_folder']) / f"peaksearch_{cfg['output']['results_stamp']}.pickle"
    return Path(cfg['output']['results_folder']) / f"allresults_{cfg['output']['results_stamp']}.pickle"


def _indexing_fingerprint_keys(indexing: dict) -> dict:
    """indexing parameters for the fingerprint, written as before the new parameters, so that maps indexed before
    are not indexed again: parameters added later are left out when they have their default (no effect) value,
    and renamed parameters keep their former name (MAXINDEXPROBED -> NBMAXPROBED)"""
    keys = {k: v for k, v in indexing.items()
            if not (k in ('blacklist', 'blacklist_tolerance') and indexing.get('blacklist') is None)
            and not (k in ('refinement_steps', 'solution_criteria') and v is None)
            and not (k in ('solution_max_pixdev', 'solution_reject') and indexing.get('solution_criteria') is None)}
    dir_ = dict(keys['dict_indexrefine'])
    if dir_.get('spots set B') is None:
        dir_.pop('spots set B', None)
    if not IB.ISS.is_spots_proposal(dir_.get('central spots indices')):   # parameters of the proposal of set A
        for key in [k for k in dir_ if k.startswith('proposal ')]:
            dir_.pop(key)
    for old, new in IB._KEY_ALIASES.items():
        if new in dir_:
            dir_[old] = dir_.pop(new)
    keys['dict_indexrefine'] = dir_
    return keys


def _run_keys(cfg: dict) -> dict:
    """run options of the fingerprint (options added later left out when they have their default value:
    fingerprints of former maps)"""
    run = cfg['run']
    keys = {k: run[k] for k in ('usepreviousUB', 'skipindexing', 'starting_grainindex')}
    if not run.get('ignorefitfileresults', True):
        keys['ignorefitfileresults'] = False
        keys['fit_acceptance'] = run.get('fit_acceptance')
    if run.get('previous_fit_folder'):
        keys['previous_fit_folder'] = str(run['previous_fit_folder'])
    return keys


def _selection_keys(cfg: dict) -> dict:
    """selection of the fingerprint (image_indices left out when None: fingerprints of former maps)"""
    return {k: v for k, v in cfg['selection'].items() if not (k == 'image_indices' and v is None)}


def _fingerprint(step: str, cfg: dict) -> str:
    """hash of the parameters of a configuration that change the results of a step (not names, plots, cpus ...)"""
    import hashlib
    scan, out = cfg['scan'], cfg['output']
    if step == 'peaksearch':
        keys = {'scan': {k: scan[k] for k in ('image_folder', 'prefix', 'suffix', 'nbdigits', 'CCDLabel', 'detfile')},
                'output': {k: out[k] for k in ('dat_folder', 'added_string', 'write_cor', 'nbfiles_per_folder',
                                               'subfolder_prefix')},
                'peaksearch': cfg['peaksearch'], 'exclusion_boxes': cfg['exclusion_boxes'],
                'selection': _selection_keys(cfg)}
    else:
        peaks = sorted(Path(scan['cor_folder']).glob('peaksearch_*.pickle')) if scan['cor_folder'] else []
        keys = {'scan': {k: scan[k] for k in ('cor_folder', 'prefix', 'nbdigits', 'CCDLabel', 'nbfiles_per_folder',
                                              'subfolder_prefix')},
                'output': {k: out[k] for k in ('fit_folder', 'fit_subfolders')},
                'material': cfg['material'], 'indexing': _indexing_fingerprint_keys(cfg['indexing']),
                'selection': _selection_keys(cfg),
                'run': _run_keys(cfg),
                'ub_matrices': cfg['ub_matrices'],
                'peaksearch_results': [p.stat().st_mtime for p in peaks]}   # new peak search -> new indexation
        if cfg.get('sequence') is not None:   # (key absent for single material: fingerprints of former maps)
            keys['sequence'] = cfg['sequence']
        if cfg['run']['usepreviousUB'] == 'neighbours':   # (idem)
            keys['neighbours'] = {k: v for k, v in IB.IN.settings(cfg['run']['neighbours']).items()
                                  if k != 'library_refresh'}
            seedsfile = keys['neighbours'].get('seeds_file')
            if seedsfile and Path(seedsfile).exists():   # validated seeds changed -> index again
                keys['neighbours']['seeds_file_content'] = hashlib.sha1(Path(seedsfile).read_bytes()).hexdigest()
    text = json.dumps(IB._to_plain(keys), sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()


def _fingerprint_file(resfile: Path) -> Path:
    return resfile.with_name(resfile.name + '.fingerprint')


def _archive_previous(folder: Path) -> Optional[Path]:
    """rename a folder of results of a previous run to <folder>_previous_<date> (nothing is deleted), so that files
    of the previous run (images without peak now, grains not found now ...) are not mixed with the new ones"""
    if not folder.is_dir() or not any(folder.iterdir()):
        return None
    archived = folder.with_name(f"{folder.name}_previous_{time.strftime('%Y%m%d_%H%M%S')}")
    folder.rename(archived)
    GT.printyellow(f'results of the previous run moved to {archived}\n(delete this folder when you do not need it)')
    return archived


def _job_pointer(step: str, cfg: dict) -> Path:
    """file of the SLURM job submitted for a configuration file (e.g. configs/indexing_slurm_job.json for
    indexing.yaml, configs/indexing_seeds_slurm_job.json for indexing_seeds.yaml)"""
    stem = Path(cfg['configfile']).stem
    return Path(cfg['configfile']).parent / f'{stem if stem.startswith(step) else step}_slurm_job.json'


def _forget_ended_job(pointer: Path) -> bool:
    """remove the pointer to a job submitted before that ended without all its results (cancelled, time limit,
    failure), so that the step is computed again (previous results folder archived). True if removed.
    The job state is known only where SLURM is available: elsewhere the pointer is kept (error)"""
    job = IB.load_slurm_job(json.loads(pointer.read_text())['jobdir'])
    parts = list(Path(job['jobdir']).glob('allresults_part*.pickle'))
    if len(parts) >= job.get('nchunks', 1):
        return False   # finished: its results are loaded
    if not slurm_available():
        raise RuntimeError(f"a SLURM job was submitted for this step ({job.get('job_id')}, {job['jobdir']}) and its "
                           f"state cannot be known from this machine: run the cell where SLURM is available, or "
                           f"delete {pointer} if this job is not running any more")
    if job.get('job_id') and _slurm_states(job['job_id']):
        return False   # still running or pending: waited for
    GT.printyellow(f"job {job.get('job_id')} submitted before ended without all its results ({len(parts)}/"
                   f"{job.get('nchunks', 1)} parts: cancelled, time limit or failure, see {job['jobdir']}/logs): "
                   f"computing again (peak search: only the images not done, if the parameters are the same)")
    pointer.unlink()
    return True


def _slurm_states(job_id: str) -> List[str]:
    """states of the jobs (job array) in the queue, [] if finished (or squeue not available)"""
    try:
        out = IB._run(['squeue', '-h', '-j', str(job_id), '-o', '%T']).stdout
    except (FileNotFoundError, OSError):
        return []
    return [s.strip() for s in out.splitlines() if s.strip()]


def wait_slurm_job(job: Union[dict, str, Path], poll: Optional[float] = None, verbose: bool = True) -> bool:
    """wait for the end of a SLURM job (status updated every poll seconds, None: SLURM_POLL, at least 2 s). True if
    all its results are written

    Interrupting the wait (kernel interrupt button) does not stop the job: run the cell again to wait again.
    """
    from IPython.display import display, HTML
    job = IB.load_slurm_job(job)
    poll = max(2., SLURM_POLL if poll is None else float(poll))
    handle = display(HTML(f"<pre>job {job.get('job_id')}: submitted</pre>"), display_id=True) if verbose else None
    t0 = time.time()
    try:
        while True:
            nparts = len(list(Path(job['jobdir']).glob('allresults_part*.pickle')))
            states = _slurm_states(job['job_id']) if job.get('job_id') else []
            # progress: files written in the results folder (the progress bars of the logs are written rarely)
            if job.get('dat_folder'):
                progress = f"peak lists (.cor) in {Path(job['dat_folder']).name}: " \
                           f"{_count_results(Path(job['dat_folder']), '.cor')}"
            elif job.get('fit_folder') and job.get('kind') != 'grains':
                progress = f"indexed images (_g0.fit) in {Path(job['fit_folder']).name}: " \
                           f"{_count_results(Path(job['fit_folder']), '_g0.fit')}"
            else:
                progress = 'images done:'
                for logfile in sorted((Path(job['jobdir']) / 'logs').glob('*.out')):
                    found = re.findall(r'(\d+)/(\d+) \[', logfile.read_text(errors='replace'))
                    if found:
                        progress += f' {found[-1][0]}/{found[-1][1]}'
            text = (f"job {job.get('job_id')} on {job['machine']}: {', '.join(sorted(set(states))) or 'finished'}"
                    f" | results {nparts}/{job['nchunks']} | {progress} ({job.get('nfiles', '?')} to do)"
                    f" | waiting for {int(time.time() - t0)} s")
            if handle is not None:
                handle.update(HTML(f'<pre>{text}</pre>'))
            if nparts >= job['nchunks']:
                return True
            if not states:   # not in the queue any more, but results missing: failed (or squeue not available)
                time.sleep(5)   # results may be written just after the end of the job
                if len(list(Path(job['jobdir']).glob('allresults_part*.pickle'))) >= job['nchunks']:
                    return True
                GT.printred(f"job finished without all its results: see the logs in {job['jobdir']}/logs")
                IB.slurm_job_status(job, nblines=10)
                return False
            time.sleep(poll)
    except KeyboardInterrupt:
        GT.printyellow('waiting interrupted: the job goes on. Run the cell again to wait for its end')
        return False


def _inprogress_file(cfg: dict) -> Path:
    """marker of a peak search started and not finished (its fingerprint), in dat_folder"""
    return Path(cfg['output']['dat_folder']) / 'peaksearch_inprogress.fingerprint'


def _image_index(cfg: dict, imagefile: Union[str, Path]) -> Optional[int]:
    scan = cfg['scan']
    m = re.match(rf"^{re.escape(scan['prefix'])}(\d+){re.escape(scan['suffix'])}$", Path(imagefile).name)
    return int(m.group(1)) if m else None


def _resume_peaksearch(cfg: dict, listfiles: List[str], nb_recent: int) -> List[str]:
    """images of listfiles still to search after an unfinished peak search (same parameters): those without a
    readable .cor file, and the nb_recent last written ones (possibly cut when the job or process was killed)"""
    files = _peaklist_files(cfg)
    recent = set(sorted(files, key=lambda i: os.path.getmtime(files[i]))[-nb_recent:]) if nb_recent else set()
    tasks = [(i, f) for i, f in files.items() if i not in recent and f.endswith('.cor')]
    readable = {i for i, n, _ in _read_peaklists(tasks) if n >= 0}
    todo = [f for f in listfiles if _image_index(cfg, f) not in readable]
    GT.printgreen(f'peak search resumed (unfinished before, same parameters): {len(listfiles) - len(todo)} images '
                  f'already done (.cor files kept), {len(todo)} to search')
    return todo


def _slurm_seconds(text: str) -> Optional[float]:
    """seconds of a SLURM time: 'days-hh:mm:ss', 'hh:mm:ss', 'mm:ss', 'minutes'; None: 'infinite', 'UNLIMITED' ..."""
    text = text.strip()
    days, _, rest = text.rpartition('-') if '-' in text else ('0', '', text)
    try:
        parts = [float(x) for x in rest.split(':')]
        days = float(days)
    except ValueError:
        return None
    if len(parts) == 1:
        parts = [0., parts[0], 0.]   # minutes
    elif len(parts) == 2:
        parts = [0.] + parts         # mm:ss
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _hms(seconds: float) -> str:
    hours, rest = divmod(int(np.ceil(seconds)), 3600)
    return f'{hours:02d}:{rest // 60:02d}:{rest % 60:02d}'


@functools.lru_cache(maxsize=16)
def partition_max_time(partition: str) -> Optional[float]:
    """largest time limit (s) of the jobs of a SLURM partition (sinfo), None if unknown (no sinfo) or infinite"""
    try:
        out = IB._run(['sinfo', '-h', '-p', partition, '-o', '%l']).stdout
    except (FileNotFoundError, OSError):
        return None
    limits = [_slurm_seconds(line) for line in out.split() if line.strip()]
    return None if not limits or None in limits else max(limits)


def measure_peaksearch_time(cfg: dict, nb_images: int = 2) -> Optional[float]:
    """cpu seconds of the peak search of one image (1 process, the parameters of cfg): nb_images images around the
    test image (run: test_image_index), the slowest one. None if failed"""
    params = PB.params_from_config(cfg, writefiles=False)
    first = int(cfg['run']['test_image_index'])
    times = []
    for index in range(first, first + nb_images):
        try:
            imagefile = PB.image_path(cfg, index)
            t0 = time.time()
            PB.peaksearch_image(imagefile, params)
            times.append(time.time() - t0)
        except Exception:   # image missing (end of the map ...)
            continue
    return max(times) if times else None


def _fit_nodes(machine: str, nbfiles: int) -> str:
    """machine of several nodes (e.g. magnifix3): the same machine with fewer nodes so that each node has at least as
    many files as cpus (nbfiles // cpus per node nodes, at least 1): a job of several nodes waits in the queue until
    all its nodes are free, which is not worth it for nodes not fully used"""
    mach = IB.SLURM_MACHINES[machine]
    nodes = mach.get('nodes', 1)
    needed = max(1, nbfiles // mach['cpus'])
    if needed >= nodes:
        return machine
    same = [(m.get('nodes', 1), name) for name, m in IB.SLURM_MACHINES.items()
            if m['partition'] == mach['partition'] and m['constraint'] == mach['constraint'] and m['cpus'] == mach['cpus']
            and m.get('nodes', 1) >= needed]
    newnodes, newmachine = min(same)
    if newmachine != machine:
        GT.printyellow(f"{nbfiles} files for {nodes} nodes of {mach['cpus']} cpus ({machine}): {newnodes} node(s) "
                       f"used ({newmachine})")
    return newmachine


def _timing_file(cfg: dict) -> Path:
    """time of the peak search of a configuration (measured on the test image, or of the last job), see
    _job_time_limit(): configs/<configuration>_timing.json"""
    return Path(cfg['configfile']).with_name(Path(cfg['configfile']).stem + '_timing.json')


def _save_job_timing(cfg: dict, job: dict):
    """cpu seconds per image of a finished peak search job (all its processes busy: real speed of the node), from
    its log ('It took ... to search peaks in ... images with ... cpus'), for the next time limits"""
    total_s, total_images = 0., 0
    for logfile in (Path(job['jobdir']) / 'logs').glob('*.out'):
        for mins, secs, nimages, ncpus in re.findall(r'It took (\d+) min (\d+) s to search peaks in (\d+) images '
                                                    r'with (\d+)', logfile.read_text(errors='replace')):
            total_s += (int(mins) * 60 + int(secs)) * int(ncpus)
            total_images += int(nimages)
    if total_images:
        _timing_file(cfg).write_text(json.dumps({'fingerprint': _fingerprint('peaksearch', cfg),
                                                 'sec_per_image': total_s / total_images, 'source': 'last job'}))


def _job_time_limit(step: str, cfg: dict, nbfiles: int, nb_cpus: int, machine: str, nchunks: int,
                    time_limit: Union[str, float, None], sec_per_image: Optional[float], verbose: bool = True):
    """(time limit 'hh:mm:ss', nchunks) of the SLURM job of a step

    time_limit None (estimate) or a number (factor applied to the estimate): estimate = safety x nb of images per
    job x cpu seconds per image / nb_cpus (min 20 min). Cpu seconds per image: sec_per_image if given, else for the
    peak search: the real speed of the last job with the same parameters (safety 1.5), or measured on the test image
    (safety 3: slower with all cpus busy; saved, not measured again), else SEC_CPU_PER_IMAGE (safety 2).
    time_limit 'hh:mm:ss' (or 'days-hh:mm:ss'): used as is.
    The time is capped at the largest time of the partition (sinfo): if more is needed, the map is split in more
    jobs (job array, nchunks increased)"""
    nbjobs = max(nchunks, IB.SLURM_MACHINES[machine].get('nodes', 1))
    if isinstance(time_limit, str):
        seconds = _slurm_seconds(time_limit)
        if seconds is None:
            raise ValueError(f"time_limit {time_limit!r}: 'hh:mm:ss', 'days-hh:mm:ss', a factor or None")
        origin = 'given'
    else:
        safety, origin = 2., f'{SEC_CPU_PER_IMAGE[step]} s/image (default)'
        if sec_per_image is None and step == 'peaksearch':
            timing, fingerprint = {}, _fingerprint(step, cfg)
            try:   # time measured before for these parameters (test image, or speed of the last job)
                timing = json.loads(_timing_file(cfg).read_text())
            except (OSError, ValueError):
                pass
            if timing.get('fingerprint') == fingerprint and timing.get('sec_per_image'):
                sec_per_image = timing['sec_per_image']
                safety = 1.5 if timing.get('source') == 'last job' else 3.
                origin = f"{sec_per_image:.2f} s/image ({timing.get('source')}, saved)"
            else:
                sec_per_image = measure_peaksearch_time(cfg)
                if sec_per_image is not None:
                    safety, origin = 3., f'{sec_per_image:.2f} s/image measured on the test image'
                    _timing_file(cfg).write_text(json.dumps({'fingerprint': fingerprint, 'sec_per_image': sec_per_image,
                                                             'source': 'test image'}))
        elif sec_per_image is not None:
            safety, origin = 3., f'{sec_per_image:.2f} s/image'
        if sec_per_image is None:
            sec_per_image = SEC_CPU_PER_IMAGE[step]
        grains = cfg['indexing']['nbGrainstoFind'] if step == 'indexing' else 1
        seconds = max(20 * 60, safety * nbfiles / nbjobs * sec_per_image * grains / nb_cpus)
        if time_limit is not None:   # factor
            seconds *= float(time_limit)
            origin += f' x {time_limit}'
    maxtime = partition_max_time(IB.SLURM_MACHINES[machine]['partition'])
    if maxtime is not None and seconds > maxtime:
        needed = int(np.ceil(seconds * nbjobs / maxtime))
        GT.printyellow(f"time needed {_hms(seconds)} > largest time of the partition "
                       f"{IB.SLURM_MACHINES[machine]['partition']} ({_hms(maxtime)}): {needed} jobs instead of {nbjobs}")
        nbjobs, seconds = needed, maxtime
    if verbose:
        print(f'time limit {_hms(seconds)} per job ({nbjobs} job(s), {origin})')
    return _hms(seconds), nbjobs


def run_step(step: str, cfg: dict, execution: str = 'local', nb_cpus: Optional[int] = None,
             machine: str = 'auto', time_limit: Union[str, float, None] = None, mem_per_cpu: str = '2000M',
             nchunks: int = 1, redo: bool = False, show_only: bool = False, wait: bool = True,
             sec_per_image: Optional[float] = None, resume: bool = True, poll: Optional[float] = None):
    """run 'peaksearch' (images -> .cor) or 'indexing' (.cor -> .fit) on all images of a map

    Already done with the same parameters (results file and its fingerprint): results are reloaded, nothing is
    computed, unless redo=True. Done with other parameters (configuration file changed): computed again.
    Before computing again, the folder of the previous results is renamed <folder>_previous_<date> (not deleted).
    resume (peak search): a peak search not finished (job killed: time limit, cancelled; interrupted) with the same
    parameters goes on: only the images without .cor file are searched (and the last written ones, possibly cut),
    the others are read from their .cor files. resume=False: all images searched again (previous folder archived).
    show_only=True: never compute, only reload the results (error if there are none)
    execution 'local': on this machine with nb_cpus processes (None: all available cpus)
    execution 'slurm': job on the cluster, machine 'auto' or key of IB.SLURM_MACHINES ('magnifix', 'hpc6' ...),
        nb_cpus per job (None: 96 on machines other than magnifix; magnifix: always whole nodes of 192 cpus).
        time_limit of each job: None: estimated from the nb of images and the time of the peak search of the test
        image (indexation: sec_per_image or SEC_CPU_PER_IMAGE); a number: factor applied to the estimate (e.g. 2:
        twice the estimate, for heavy images); 'hh:mm:ss' or 'days-hh:mm:ss': fixed. Capped at the largest time of
        the partition: the map is then split in more jobs. sec_per_image: cpu seconds per image (not measured).
        The notebook waits for the end of the job; if the wait is interrupted, running the cell again waits for the
        same job (it is not submitted again). poll: seconds between two updates of the progress of the job
        (None: SLURM_POLL, 30 s; e.g. MW.SLURM_POLL = 10 for all the jobs of the notebook). wait=False: the job is submitted (or found running) and None returned
        at once: call run_step() again to wait for it and get its results (e.g. jobs of several maps submitted
        together, see peaksearch_maps())
    Returns allresults (list of results of the images)
    """
    if step not in ('peaksearch', 'indexing'):
        raise ValueError("step must be 'peaksearch' or 'indexing'")
    module = PB if step == 'peaksearch' else IB
    resfile = _results_file(step, cfg)
    pointer = _job_pointer(step, cfg)

    if show_only:
        if not resfile.exists():
            raise FileNotFoundError(f"no {step} results for this map ({resfile}): use MODE = 'compute'")
        GT.printgreen(f'{step} results reloaded from {resfile}')
        return module.load_results(resfile)['allresults'] if step == 'peaksearch' else IB.load_allresults(resfile)['allresults']

    tree, loader = (PB.DEFAULT_CONFIG, PB.load_config) if step == 'peaksearch' else (IB.DEFAULT_CONFIG, IB.load_config)
    if IB._memory_changes(cfg, tree, loader):   # parameters changed in memory (e.g. pcfg['peaksearch'][...] = ...):
        # written in the configuration file first (the file stays the reference of the results)
        cfg = module.save_config(cfg, cfg['configfile'], overwrite=True)
    fingerprint = _fingerprint(step, cfg)
    if resfile.exists() and not redo:
        fpfile = _fingerprint_file(resfile)
        if not fpfile.exists() or fpfile.read_text().strip() == fingerprint:
            if not fpfile.exists():   # results of a former version: computed with the current configuration file
                fpfile.write_text(fingerprint)
            GT.printgreen(f'{step} already done: results reloaded from {resfile}\n(redo=True to compute again)')
            return module.load_results(resfile)['allresults'] if step == 'peaksearch' else IB.load_allresults(resfile)['allresults']
        GT.printyellow(f'{step} done before with other parameters (configuration or peak search changed): '
                       'computing again')
        redo = True

    if not module.check_config(cfg, verbose=True):
        raise RuntimeError(f"fix the errors of the configuration file {cfg['configfile']} and run the cell again")
    params = module.params_from_config(cfg)
    if step == 'peaksearch':
        params.overwrite = True
        listfiles = PB.select_images(cfg)
    else:
        listfiles = IB.select_corfiles(cfg)
    if not listfiles:
        raise RuntimeError(f'no file to analyse for {step}')
    print(f'{step}: {len(listfiles)} files')
    outfolder = Path(cfg['output']['dat_folder'] if step == 'peaksearch' else cfg['output']['fit_folder'])
    if pointer.exists():
        _forget_ended_job(pointer)
    resuming = (step == 'peaksearch' and resume and not redo and not resfile.exists() and not pointer.exists()
                and _inprogress_file(cfg).exists() and _inprogress_file(cfg).read_text().strip() == fingerprint)
    if resuming:
        listfiles = _resume_peaksearch(cfg, listfiles, nb_recent=2 * (nb_cpus or IB.available_cpus()))
    elif not pointer.exists() and (redo or resfile.exists() or (outfolder.is_dir() and any(outfolder.iterdir()))):
        archived = _archive_previous(outfolder)   # (not while waiting for a job submitted before: it writes there)
        if (step == 'indexing' and archived is not None and not cfg['run'].get('previous_fit_folder')
                and (cfg['run']['usepreviousUB'] == 'fromfitfile' or not cfg['run']['ignorefitfileresults'])):
            params.previous_fit_folder = str(archived)   # previous .fit files read from the archived folder
            print(f'previous .fit files read from {archived}')

    if step == 'peaksearch':   # marker of an unfinished peak search (resume), removed when the results are saved
        outfolder.mkdir(parents=True, exist_ok=True)
        _inprogress_file(cfg).write_text(fingerprint)

    job = None
    if execution == 'slurm' and pointer.exists():   # job submitted before (e.g. wait interrupted): wait again
        job = IB.load_slurm_job(json.loads(pointer.read_text())['jobdir'])
        if job.get('job_id') and (_slurm_states(job['job_id'])
                                  or len(list(Path(job['jobdir']).glob('allresults_part*.pickle'))) >= job['nchunks']):
            print(f"job {job['job_id']} submitted before: {'waiting for it ' if wait else ''}(not submitted again)")
        else:
            job = None
    if execution == 'slurm' and job is None and not listfiles:
        pass   # resumed: all images already done
    elif execution == 'slurm' and job is None:
        if machine == 'auto':
            machine, nb_cpus = IB.choose_slurm_machine(nb_cpus or 192)
        else:
            machine = _fit_nodes(machine, len(listfiles))
            nb_cpus = nb_cpus or IB.SLURM_MACHINES[machine]['cpus']
        time_limit, nchunks = _job_time_limit(step, cfg, len(listfiles), nb_cpus, machine, nchunks, time_limit,
                                              sec_per_image)
        job = module.prepare_slurm_job(cfg, params, listfiles, machine=machine, nb_cpus=nb_cpus, time=time_limit,
                                       mem_per_cpu=mem_per_cpu, nchunks=nchunks,
                                       job_name=f"{step}_{Path(cfg['configfile']).parent.parent.parent.name}"[:60])
        if IB.submit_slurm_job(job) is None:
            raise RuntimeError('the job could not be submitted (see above)')
        pointer.write_text(json.dumps({'jobdir': job['jobdir']}))

    if execution == 'slurm' and not wait and job is not None:
        print(f"job {job.get('job_id')} on {job['machine']}: not waited for (run_step() again to get its results)")
        return None
    if not listfiles and job is None:   # resumed: all images already done
        allresults = []
    elif execution == 'slurm':
        if not wait_slurm_job(job, poll=poll):
            return None
        allresults = IB.load_slurm_results(job)
        nb_cpus = job['nb_cpus']
        if step == 'peaksearch':
            _save_job_timing(cfg, job)
    else:
        allresults = module.run_multiprocessing(listfiles, params, nb_cpus=nb_cpus)

    mapdims = cfg['scan']['mapdims']
    if step == 'peaksearch':
        # images not searched here (resumed peak search): results read from their .cor files
        computed = {r[0] for r in allresults}
        selected = {_image_index(cfg, f) for f in PB.select_images(cfg)}
        kept = [(i, f) for i, f in _peaklist_files(cfg).items() if i in selected and i not in computed]
        if kept:
            allresults = sorted(list(allresults) + _read_peaklists(kept, nb_cpus), key=lambda r: r[0])
            print(f'{len(kept)} peak lists of the images done before read from their .cor files')
        PB.save_results(cfg, params, allresults, PB.summarize_results(allresults, mapdims), filepath=resfile,
                        overwrite=True, nb_cpus=nb_cpus)
        _inprogress_file(cfg).unlink(missing_ok=True)
    else:
        IB.save_allresults(cfg, params, allresults, IB.summarize_results(allresults, mapdims), filepath=resfile,
                           overwrite=True, nb_cpus=nb_cpus)
    _fingerprint_file(resfile).write_text(fingerprint)
    if pointer.exists():
        pointer.unlink()
    GT.printgreen(f'{step} done')
    return allresults


# --------------------------------------------------------------------------------------
#  INDEXING WORKFLOWS (with or without the .fit files of a previous indexation)
# --------------------------------------------------------------------------------------
INDEXING_WORKFLOWS = {
    # name: (run options, description) (MODE of the notebook Indexation_MultiProcessing, Index_Refine GUI)
    'scratch': ({'ignorefitfileresults': True, 'usepreviousUB': False, 'skipindexing': False},
                'every point indexed from scratch, previous .fit files ignored (MODE 0)'),
    'refine': ({'ignorefitfileresults': True, 'usepreviousUB': 'fromfitfile', 'skipindexing': True},
               'UB matrices of the previous .fit files (all grains) refined only, with the current refinement '
               'parameters (MODE 1)'),
    'refine_or_scratch': ({'ignorefitfileresults': True, 'usepreviousUB': 'fromfitfile', 'skipindexing': False},
                          'UB matrices of the previous .fit files refined; indexing from scratch if they are rejected '
                          'or missing (MODE 3, GUI: update preexisting results)'),
    'matrices': ({'ignorefitfileresults': True, 'skipindexing': True},
                 'list of UB matrices (UB_MATRICES) checked and refined at every point, no indexing from scratch '
                 '(MODE 2)'),
    'matrices_or_scratch': ({'ignorefitfileresults': True, 'skipindexing': False},
                            'list of UB matrices checked and refined first; indexing from scratch if they are '
                            'rejected (MODE 4, GUI: guessed matrices)'),
    'redo_poor': ({'ignorefitfileresults': False, 'usepreviousUB': False, 'skipindexing': False},
                  'previous .fit files kept if accepted (ACCEPTANCE: min nb of spots, max pixel deviation); the '
                  'others (and the missing ones) indexed from scratch'),
}


def indexing_workflow(icfg: dict, workflow: str = 'scratch', ub_matrices: Optional[dict] = None,
                      acceptance: Optional[dict] = None, criteria=None, max_pixdev: Optional[float] = 1.0,
                      reject: Optional[bool] = None, previous_fit_folder=None, verbose: bool = True) -> dict:
    """write in the indexing configuration file of icfg the options of an indexing workflow (see
    INDEXING_WORKFLOWS) and return the configuration to run with run_step('indexing', ...)

    ub_matrices: {name: 3x3 matrix or .fit file (its UB matrix)} of workflows 'matrices' and 'matrices_or_scratch'
    acceptance: {'min_nb_spots': n, 'max_pixdev': p} of the previous .fit files kept by 'redo_poor'
    criteria: None, or criteria of a correct solution (steps [e_max, tolerance angle, min nb of matches], see
        IB.verify_solution()) checked for every grain found, with max_pixdev (pixel) at the last step;
        reject (default: True with criteria): grains failing them rejected (then indexing from scratch if the
        workflow allows it)
    previous_fit_folder: .fit files of the previous indexation (None: the results folder of this configuration,
        archived as <fit_folder>_previous_<date> when the indexation is run again)
    """
    if workflow not in INDEXING_WORKFLOWS:
        raise ValueError(f'workflow must be one of {list(INDEXING_WORKFLOWS)}, not {workflow!r}')
    changes = dict(INDEXING_WORKFLOWS[workflow][0])
    if workflow in ('matrices', 'matrices_or_scratch'):
        if not ub_matrices:
            raise ValueError(f"workflow {workflow!r}: give the UB matrices to check (ub_matrices / UB_MATRICES)")
        mats = {}
        for name, value in ub_matrices.items():
            if isinstance(value, (str, Path)) and str(value).endswith('.fit'):
                value = IB.fitfile_grains(Path(value).parent, Path(value).name.rsplit('_g', 1)[0])
                value = next(g['UB'] for g in value if Path(g['path']).name == Path(str(ub_matrices[name])).name)
            mats[name] = np.asarray(value, dtype=float).tolist()
        changes['ub_matrices'] = {**(icfg.get('ub_matrices') or {}), **mats}
        changes['usepreviousUB'] = list(mats)
    acc = {'min_nb_spots': None, 'max_pixdev': None, **((acceptance or {}) if workflow == 'redo_poor' else {})}
    changes['min_nb_spots'], changes['max_pixdev'] = acc['min_nb_spots'], acc['max_pixdev']   # (fit_acceptance)
    changes['solution_criteria'] = criteria
    changes['solution_max_pixdev'] = max_pixdev
    changes['solution_reject'] = bool(criteria) if reject is None else bool(reject)
    changes['previous_fit_folder'] = None if previous_fit_folder is None else str(previous_fit_folder)
    if workflow != 'scratch' and workflow not in ('matrices', 'matrices_or_scratch'):
        fitfolder = Path(previous_fit_folder or icfg['output']['fit_folder'])
        if not (fitfolder.is_dir() and any(fitfolder.glob('*.fit')) or any(fitfolder.glob('*/*.fit'))):
            GT.printyellow(f'workflow {workflow!r}: no .fit file in {fitfolder} (previous indexation)')
    icfg = IB.save_config(icfg, icfg['configfile'], overwrite=True, **changes)
    if verbose:
        GT.printgreen(f"indexing workflow {workflow!r}: {INDEXING_WORKFLOWS[workflow][1]}")
        if criteria:
            print(f"criteria of a correct solution: {criteria}, max pixel deviation {max_pixdev} "
                  f"({'failing grains rejected' if changes['solution_reject'] else 'reported only'})")
    return icfg


# --------------------------------------------------------------------------------------
#  RESULTS: maps
# --------------------------------------------------------------------------------------
# peak list columns of PB.peaksearch_image(): names in the .cor files (.dat: same names, with peak_X peak_Y)
_PEAKLIST_COLUMNS = ['X', 'Y', 'peak_Isub', 'peak_fwaxmaj', 'peak_fwaxmin', 'peak_inclination', 'Xdev', 'Ydev',
                     'peak_bkg', 'Ipixmax']


def _read_peaklist_file(args) -> tuple:
    """(imageindex, nb of peaks, peaklist) of a .cor (or .dat) file, as PB.peaksearch_image() (-1 if not readable)"""
    imageindex, path = args
    try:
        with open(path) as f:
            names = f.readline().split()
        data = np.atleast_2d(np.loadtxt(path, skiprows=1, comments='#'))
    except Exception:
        return imageindex, -1, None
    if data.size == 0:
        return imageindex, 0, None
    names = [{'peak_X': 'X', 'peak_Y': 'Y'}.get(n, n) for n in names]
    peaklist = np.full((len(data), len(_PEAKLIST_COLUMNS)), np.nan)
    for k, name in enumerate(_PEAKLIST_COLUMNS):
        if name in names:
            peaklist[:, k] = data[:, names.index(name)]
        elif name == 'peak_Isub' and 'I' in names:   # .cor file without the peak properties
            peaklist[:, k] = data[:, names.index('I')]
    return imageindex, len(peaklist), peaklist


def _peaklist_files(pcfg: dict) -> Dict[int, str]:
    """{image index: peak list file} of the .cor files (or .dat files if there is no .cor file) in dat_folder"""
    scan, out = pcfg['scan'], pcfg['output']
    stem = f"{scan['prefix']}{out['added_string']}"
    folder = Path(out['dat_folder'])
    files = {}
    for ext in ('.dat', '.cor'):   # .cor files preferred
        pattern = re.compile(rf"^{re.escape(stem)}(\d{{{scan['nbdigits']},}})\{ext}$")
        found = folder.glob(f"{out['subfolder_prefix']}*/{stem}*{ext}") if out['nbfiles_per_folder'] \
            else folder.glob(f'{stem}*{ext}')
        files.update({int(m.group(1)): str(f) for f in found if (m := pattern.match(f.name))})
    return files


def _read_peaklists(tasks, nb_cpus: Optional[int] = None, min_files_pool: int = 2000, max_cpus_pool: int = 32) -> list:
    """_read_peaklist_file() of the (imageindex, path) tasks, with a pool of processes if there are many"""
    nb_cpus = nb_cpus or min(IB.available_cpus(), max_cpus_pool)
    if len(tasks) >= min_files_pool and nb_cpus > 1:
        with IB.mp_context().Pool(processes=nb_cpus) as pool:
            return pool.map(_read_peaklist_file, tasks, chunksize=16)
    return [_read_peaklist_file(t) for t in tasks]


def load_corfiles(pcfg: dict, nb_cpus: Optional[int] = None, min_files_pool: int = 2000,
                  max_cpus_pool: int = 32) -> list:
    """peak search results read from the .cor files (or .dat files if there is no .cor file) already written in
    dat_folder, without computing anything (e.g. peak search still running, interrupted, or done on SLURM without
    the results pickle)

    Returns allresults as run_step('peaksearch', ...): list of (imageindex, nb of peaks, peaklist array (nb peaks,
    10 columns) X, Y, I, fwhm major, fwhm minor, inclination, Xdev, Ydev, background, Ipixmax), for
    PB.summarize_results(), PB.peaks_of_image() ... Images without peak list file (not treated yet, or without peak)
    are not in the list (nan in the map of PB.summarize_results()).
    Read with a pool of nb_cpus processes (None: cpus available, at most max_cpus_pool) if there are at least
    min_files_pool files, else in this process
    """
    scan = pcfg['scan']
    folder = Path(pcfg['output']['dat_folder'])
    tasks = sorted(_peaklist_files(pcfg).items())
    t0 = time.time()
    allresults = _read_peaklists(tasks, nb_cpus, min_files_pool, max_cpus_pool)
    nfast, nslow = scan['mapdims']
    nbcor = sum(1 for _, f in tasks if f.endswith('.cor'))
    print(f"{len(allresults)} peak lists read ({nbcor} .cor, {len(tasks) - nbcor} .dat files) / {nfast * nslow} map "
          f"points in {folder} ({time.time() - t0:.0f} s)")
    return allresults


def load_fitfiles(icfg: dict, grainindex: int = 0, nb_cpus: Optional[int] = None, min_files_pool: int = 2000,
                  max_cpus_pool: int = 32):
    """parsed_fitfileseries of the .fit files of grain grainindex (UB, strain, nb of spots ... of all map points)

    Read silently (the reader prints a message for each line of the .fit files that it does not use), with a pool of
    nb_cpus processes (None: cpus available, see IB.available_cpus(), at most max_cpus_pool: more processes are slower
    on a network file system) if the map has at least min_files_pool points (reading a .fit file on a network file
    system takes ~10 ms), else in this process
    """
    import contextlib
    import io
    from LaueTools.fitfilereader import parsed_fitfileseries
    scan = icfg['scan']
    layout = IB.fitfiles_layout(icfg)
    t0 = time.time()
    nbpoints = scan['mapdims'][0] * scan['mapdims'][1]
    nb_cpus = nb_cpus or min(IB.available_cpus(), max_cpus_pool)
    pool = nbpoints >= min_files_pool and nb_cpus > 1
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ffs = parsed_fitfileseries(folderpath=str(icfg['output']['fit_folder']), nb_cols=scan['mapdims'][0],
                                   nb_rows=scan['mapdims'][1], prefix=scan['prefix'], suffix=f'_g{grainindex}.fit',
                                   use_multiprocessing=pool, nbdigits=scan['nbdigits'],
                                   nbfiles_per_folder=layout['fit_nbfiles_per_folder'],
                                   subfolder_prefix=layout['subfolder_prefix'],
                                   nb_cpus=nb_cpus if pool else None)
    nbfound = int(np.sum(np.isfinite(ffs.UB[:, 0, 0])))
    print(f"{nbfound} .fit files of grain {grainindex} read / {scan['mapdims'][0] * scan['mapdims'][1]} map points "
          f"({time.time() - t0:.0f} s{', ' + str(ffs.nb_cpus) + ' processes' if pool else ''})")
    return ffs


def map_transposed(cfg: dict) -> bool:
    """True if maps are drawn transposed, so that the y motor (yech, yps, sy) is along the vertical axis
    (generaltools.map_transposed(), same convention as the LaueTools GUI)"""
    return GT.map_transposed(cfg['scan']['fast_motor'], cfg['scan']['slow_motor'])


def _map_format_coord(cfg: dict, data: Optional[np.ndarray] = None, transposed: bool = False):
    """hover text: motor indices, image index (and value) of the map point under the cursor"""
    scan = cfg['scan']
    nfast, nslow = scan['mapdims']

    def fmt(x, y):
        i, j = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        row, col = (i, j) if transposed else (j, i)   # row: slow motor, col: fast motor
        if 0 <= col < nfast and 0 <= row < nslow:
            txt = f"{scan['fast_motor']} #{col}, {scan['slow_motor']} #{row}, image {row * nfast + col}"
            if data is not None and data.ndim == 2:
                txt += f', value {data[row, col]:.4g}'
            return txt
        return ''
    return fmt


def plot_map(data: np.ndarray, cfg: dict, title: str = '', ax=None, cmap='viridis', vmin=None, vmax=None,
             colorbar: bool = True, label: str = ''):
    """2D map (shape invmapdims = (slow, fast), or (slow, fast, 3) rgb colors) with the y motor (yech, yps, sy)
    along the vertical axis and motors as axes labels; hover shows the motor indices and the image index"""
    import matplotlib.pyplot as plt
    scan = cfg['scan']
    transposed = map_transposed(cfg)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    shown = (data.transpose(1, 0, 2) if data.ndim == 3 else data.T) if transposed else data
    im = ax.imshow(shown, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if colorbar and data.ndim == 2:
        ax.figure.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_title(title, fontsize=9)
    xlabel, ylabel = GT.map_axes_labels(scan['fast_motor'], scan['slow_motor'], transposed)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.format_coord = _map_format_coord(cfg, data, transposed)
    return ax


def gallery_folder(folder: Union[str, Path]) -> Optional[Path]:
    """folder mirrored in GALLERY (ESRF data policy, images shown in the data portal): PROCESSED_DATA replaced by
    GALLERY. None if folder is not in a PROCESSED_DATA folder (e.g. OUTPUT_ROOT elsewhere)"""
    parts = Path(folder).parts
    if 'PROCESSED_DATA' not in parts:
        return None
    k = len(parts) - 1 - parts[::-1].index('PROCESSED_DATA')
    return Path(*parts[:k], 'GALLERY', *parts[k + 1:])


def _path_two_lines(folder: Union[str, Path]) -> str:
    """folder on 2 lines: up to PROCESSED_DATA/ (or RAW_DATA/, GALLERY/), then the rest (else cut at the middle)"""
    parts = Path(folder).parts
    k = next((i for i in range(len(parts) - 1, -1, -1) if parts[i] in ('PROCESSED_DATA', 'RAW_DATA', 'GALLERY')),
             len(parts) // 2)
    first, rest = Path(*parts[:k + 1]), Path(*parts[k + 1:]) if k + 1 < len(parts) else ''
    return f'{first}/\n    {rest}/' if rest else f'{first}/'


def add_data_path(fig, folder: Union[str, Path], fontsize: float = 6.):
    """write folder (on 2 lines, see _path_two_lines()) above the figure. The figure is made taller by the height
    of the text and its contents moved down: titles and axes keep their size and position relative to each other.
    Called again: the text is only updated"""
    text = _path_two_lines(folder)
    existing = getattr(fig, '_lauetools_datapath', None)
    if existing is not None:
        existing.set_text(text)
        return existing
    width, height = fig.get_size_inches()
    extra = 3 * fontsize / 72   # inches: 2 lines and margins
    ratio = height / (height + extra)
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position(original=True).bounds
        ax.set_position([x0, y0 * ratio, w, h * ratio])
    for txt in fig.texts:   # e.g. suptitle
        txt.set_y(txt.get_position()[1] * ratio)
    fig.set_size_inches(width, height + extra)
    fig._lauetools_datapath = fig.text(0.01, 1 - 0.4 * fontsize / 72 / (height + extra), text, ha='left', va='top',
                                       fontsize=fontsize, family='monospace', color='0.35')
    return fig._lauetools_datapath


def _wait_interactive_figure(fig, timeout: float = 2.):
    """interactive figure (ipympl): wait until JupyterLab has initialised it (first image sent). Its messages are
    handled in another thread of the kernel (subshell) while the cell runs: a figure drawn (or saved) meanwhile may
    be black or cut. Without front-end (e.g. notebook executed by nbconvert): returns after timeout (s)"""
    canvas = fig.canvas
    if not hasattr(canvas, 'syncing_data_url'):   # not ipympl
        return
    t0 = time.time()
    while (canvas.syncing_data_url or canvas._png_is_old) and time.time() - t0 < timeout:
        time.sleep(0.02)


def _detached_copy(fig):
    """copy of figure fig, not attached to its canvas nor to pyplot (pickled and unpickled): saving the copy does not
    touch the interactive figure. Local functions (e.g. format_coord of plot_map()) are None in the copy"""
    import io
    import pickle
    import types
    from matplotlib.figure import Figure

    class DetachedFigure(Figure):
        def __setstate__(self, state):
            state.pop('_restore_to_pylab', None)   # no new pyplot figure (new widget in the notebook)
            super().__setstate__(state)

    class Pickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, types.FunctionType) and '<' in obj.__qualname__:   # <locals>, <lambda>
                return type(None), ()
            return NotImplemented

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ('matplotlib.figure', 'Figure'):
                return DetachedFigure
            return super().find_class(module, name)

    buf = io.BytesIO()
    Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(fig)
    buf.seek(0)
    return Unpickler(buf).load()


def savefigresults(fig, cfg: dict, name: str, where: Optional[str] = 'results', fmt: str = 'png',
                   dpi: int = 150, datapath: bool = True, verbose: bool = True) -> List[Path]:
    """save a figure of the results of a step as <name>.<fmt> next to its results file (results_folder of cfg:
    peaksearch_<stamp>.pickle of the peak search, allresults_<stamp>.pickle of the indexation)

    fig: matplotlib Figure or Axes (e.g. returned by plot_map())
    where: 'results' (results folder only), 'gallery' (also in the same folder of GALLERY instead of PROCESSED_DATA),
        None or False: not saved
    datapath: results folder written above the figure (add_data_path(), also on the figure shown in the notebook
        with %matplotlib widget)
    Returns the paths of the written files

    Call it after plt.show(): with %matplotlib widget (ipympl), JupyterLab initialises the figure with messages
    handled in another thread of the kernel while the cell runs; a figure saved before being shown may stay black
    or be cut. For the same reason a copy of the figure is saved (savefig() of the interactive figure removes its
    figure manager while saving, and the messages received meanwhile are lost).
    """
    if not where:
        return []
    if where not in ('results', 'gallery'):
        raise ValueError(f"where={where!r}: 'results', 'gallery' or None")
    fig = getattr(fig, 'figure', fig)   # Axes -> Figure
    folders = [Path(cfg['output']['results_folder'])]
    if where == 'gallery':
        gallery = gallery_folder(folders[0])
        if gallery is None:
            GT.printyellow(f'{folders[0]} is not in PROCESSED_DATA: figure {name} not saved in GALLERY')
        else:
            folders.append(gallery)
    _wait_interactive_figure(fig)
    if datapath:
        add_data_path(fig, folders[0])
        fig.canvas.draw_idle()
    tosave = None
    for _ in range(3):   # copy may fail if the figure is drawn meanwhile (other thread)
        try:
            tosave = _detached_copy(fig)
            break
        except Exception as err:
            error = err
            time.sleep(0.2)
    if tosave is None:   # e.g. figure that cannot be pickled: saved directly
        GT.printyellow(f'figure {name} saved directly (no copy: {error!r}): an interactive figure may stay black')
        tosave = fig
    written = []
    for folder in folders:
        filepath = folder / f'{name}.{fmt}'
        try:
            folder.mkdir(parents=True, exist_ok=True)
            tosave.savefig(filepath, dpi=dpi, bbox_inches='tight')
        except PermissionError:
            if folder == folders[0]:   # results folder: error
                raise
            root = gallery_folder(folders[0])   # GALLERY not writable (e.g. restored data): warned once
            root = Path(*root.parts[:root.parts.index('GALLERY') + 1])
            if root not in _GALLERY_NOT_WRITABLE:
                _GALLERY_NOT_WRITABLE.add(root)
                GT.printyellow(f"no permission to write in {root}: figures saved only next to the results "
                               f"(SAVE_FIGURES = 'results' to avoid this message)")
            continue
        written.append(filepath)
    if verbose:
        print('figure saved:', ', '.join(str(f) for f in written))
    return written


def print_parameters(cfg: dict, step: str):
    """parameters of a step (peak search or indexation) with their names, usable in the dicts PEAKSEARCH and
    INDEXING of the quickstart notebook (e.g. PEAKSEARCH = {'IntensityThreshold': 300})"""
    if step == 'peaksearch':
        params = {k: v for k, v in cfg['peaksearch'].items() if k != 'pspfile'}
        params['exclusion_boxes'] = cfg['exclusion_boxes']
        name = 'PEAKSEARCH'
    else:
        ind = cfg['indexing']
        params = {k: v for k, v in ind.items() if k != 'dict_indexrefine'}
        params.update(ind['dict_indexrefine'])
        params.update({k: cfg['postprocess'][k] for k in ('MinimumNbIndexedSpots', 'LargestMeanPixelResidue')})
        name = 'INDEXING'
    print(f"{step} parameters ({cfg['configfile']}), names usable in {name} = {{name: value}}:")
    for key, value in params.items():
        value = IB._to_plain(value)
        if isinstance(value, list) and len(value) > 12:
            value = f'{value[:6]} ... ({len(value)} values)'
        print(f'   {key!r:34s}: {value}')


def set_parameters(cfg: dict, step: str, changes: Optional[dict], verbose: bool = True) -> dict:
    """set parameters of a step ('peaksearch' or 'indexing') given by their names (as printed by print_parameters(),
    e.g. {'IntensityThreshold': 300}) and write them in the configuration file (peaksearch.yaml, indexing.yaml):
    the test on one image and the analysis of the whole map use them. Unknown names: error with the close names.
    Returns the configuration (read again from its file): use it, e.g. pcfg = MW.set_parameters(pcfg, ...)"""
    if step not in ('peaksearch', 'indexing'):
        raise ValueError("step must be 'peaksearch' or 'indexing'")
    module = PB if step == 'peaksearch' else IB
    if not changes:
        return cfg
    for name in changes:
        IB._key_path(name, module.DEFAULT_CONFIG)   # KeyError with close names if unknown
    diff = _differences(cfg, dict(changes), module.DEFAULT_CONFIG)
    if not diff:
        if verbose:
            print(f'{step} parameters {list(changes)}: same values as in {Path(cfg["configfile"]).name}')
        return cfg
    return module.save_config(cfg, cfg['configfile'], overwrite=True, **diff)   # (prints the changes)


# --------------------------------------------------------------------------------------
#  all grains of the .fit files
# --------------------------------------------------------------------------------------
def load_grains(icfg: dict, nbgrains: Optional[int] = None, nb_cpus: Optional[int] = None) -> dict:
    """.fit files of all grains (_g0.fit, _g1.fit, ...) of the map, and at each point the 'best' grain: the one with
    the most indexed spots (g0 is the first grain found, not always the best one)

    Returns dict: 'ffs' (list of parsed_fitfileseries), 'nbspots' (nbgrains, nbpoints), 'pixdev' (nbgrains, nbpoints),
    'best' (grain index of the best grain, -1 if none), 'nbgrains_found', 'best_nbspots', 'best_pixdev',
    'best_UB' (nbpoints, 3, 3) (NaN where no grain)
    """
    nbgrains = nbgrains or icfg['indexing']['nbGrainstoFind']
    ffs = [load_fitfiles(icfg, g, nb_cpus=nb_cpus) for g in range(nbgrains)]   # nb_cpus: see load_fitfiles()
    nbspots = np.array([np.asarray(f.NumberOfIndexedSpots, dtype=float).ravel() for f in ffs])
    pixdev = np.array([np.asarray(f.MeanDevPixel, dtype=float).ravel() for f in ffs])
    npts = nbspots.shape[1]
    found = np.isfinite(nbspots) & (nbspots > 0)
    best = np.where(found.any(axis=0), np.argmax(np.where(found, nbspots, -1), axis=0), -1)
    pts = np.arange(npts)
    g = np.maximum(best, 0)
    best_UB = np.array([np.asarray(f.UB)[:npts] for f in ffs])[g, pts]
    best_UB[best < 0] = np.nan
    return {'ffs': ffs, 'nbspots': nbspots, 'pixdev': pixdev, 'best': best,
            'nbgrains_found': found.sum(axis=0),
            'best_nbspots': np.where(best >= 0, nbspots[g, pts], np.nan),
            'best_pixdev': np.where(best >= 0, pixdev[g, pts], np.nan),
            'best_UB': best_UB}


def quality_maps(grains: dict, icfg: dict, min_spots: Optional[int] = None, max_pixdev: Optional[float] = None):
    """quality of the indexation at each point, for its best grain (most indexed spots, see load_grains())

    A point is reliable if its best grain has at least min_spots indexed spots and a mean pixel deviation of at
    most max_pixdev (default: postprocess: MinimumNbIndexedSpots, LargestMeanPixelResidue of the configuration).
    Masked points are black in the next maps: the map 'why masked' and the histograms show the reason.
    Returns (fig, mask) with mask (shape invmapdims) True for unreliable points
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    post, invmapdims = icfg['postprocess'], icfg['scan']['invmapdims']
    min_spots = post['MinimumNbIndexedSpots'] if min_spots is None else min_spots
    max_pixdev = post['LargestMeanPixelResidue'] if max_pixdev is None else max_pixdev
    nb, dev = grains['best_nbspots'], grains['best_pixdev']
    notindexed = grains['best'] < 0
    fewspots = ~notindexed & (nb < min_spots)
    largedev = ~notindexed & (dev > max_pixdev)
    reason = np.zeros(nb.shape, dtype=int)            # 0: reliable
    reason[largedev] = 1
    reason[fewspots] = 2
    reason[fewspots & largedev] = 3
    reason[notindexed] = 4
    mask = (reason > 0).reshape(invmapdims)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    plot_map(grains['nbgrains_found'].reshape(invmapdims).astype(float), icfg, 'nb of grains found', ax=axes[0, 0],
             cmap='viridis', vmin=0)
    plot_map(nb.reshape(invmapdims), icfg, 'best grain: nb of indexed spots', ax=axes[0, 1])
    plot_map(dev.reshape(invmapdims), icfg, 'best grain: mean pixel deviation', ax=axes[0, 2],
             vmax=np.nanpercentile(dev, 98) if np.any(np.isfinite(dev)) else None)
    names = ['reliable', f'deviation > {max_pixdev}', f'< {min_spots} spots', 'both', 'not indexed']
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', 'black']
    ax = plot_map(reason.reshape(invmapdims).astype(float), icfg, 'why masked', ax=axes[1, 0],
                  cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5, colorbar=False)
    for k, (n, c) in enumerate(zip(names, colors)):
        ax.plot([], [], 's', color=c, label=f'{n} ({np.sum(reason == k)})')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    for ax, values, threshold, xlabel in ((axes[1, 1], nb, min_spots, 'nb of indexed spots (best grain)'),
                                          (axes[1, 2], dev, max_pixdev, 'mean pixel deviation (best grain)')):
        ax.hist(values[np.isfinite(values)], bins=50, color='gray')
        ax.axvline(threshold, color='r', ls='--', label=f'threshold {threshold}')
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
    fig.tight_layout()

    n = nb.size
    print(f'{np.sum(reason == 0)} reliable points / {n} ({100 * np.sum(reason == 0) / n:.0f}%), best grain of each point '
          f'(g0 is the best grain at {np.sum(grains["best"] == 0)} points)')
    print(f'masked: {np.sum(largedev)} with mean pixel deviation > {max_pixdev}, {np.sum(fewspots)} with < {min_spots} '
          f'indexed spots, {np.sum(notindexed)} not indexed')
    if np.any(np.isfinite(dev)) and np.sum(largedev) > 0.3 * n:
        GT.printyellow(f'many points have a deviation > {max_pixdev} pixel (median {np.nanmedian(dev):.2f}): the threshold '
                       'may be too strict for these data (e.g. calibration at another temperature, strained sample): '
                       'try e.g. MAX_PIXEL_DEVIATION = 1.0')
    return fig, mask


def orientation_maps(grains: dict, icfg: dict, mask: Optional[np.ndarray] = None, sample_tilt: float = 40.):
    """inverse pole figure (IPF) color maps along sample x, y, z of the best grain of each point, and color key"""
    import matplotlib.pyplot as plt
    from LaueTools import ipfmap as IPF
    scan, post = icfg['scan'], icfg['postprocess']
    group = laue_group(icfg)
    UB = grains['best_UB']
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    for ax, axis in zip(axes, ('x', 'y', 'z')):
        rgb = IPF.ipf_rgb_map(UB, scan['invmapdims'], group, axis, sample_tilt, mask)
        rgba = np.ones(rgb.shape[:2] + (4,))
        rgba[..., :3] = np.nan_to_num(rgb)
        rgba[..., 3] = np.all(np.isfinite(rgb), axis=2)   # masked and missing points: transparent (black background)
        ax.set_facecolor('black')
        plot_map(rgba, icfg, f'IPF-{axis.upper()} (sample frame), best grain', ax=ax)
    directions = IPF.sample_directions_in_crystal(UB, 'z', sample_tilt)
    if mask is not None:
        directions = directions[~np.asarray(mask, dtype=bool).ravel()]
    IPF.plot_ipf_key(group, ax=axes[-1], directions=directions, title=f'{group}  (points: IPF-Z)')
    fig.tight_layout()
    return fig


def laue_group(icfg: dict) -> str:
    """Laue group of the IPF colors and pole figures: postprocess: laue_group of the configuration ('auto': from the
    lattice parameters of the material, e.g. '2/m' for a monoclinic cell with unique axis b)"""
    from LaueTools import ipfmap as IPF
    group = icfg['postprocess']['laue_group']
    if group == 'auto':
        group = IPF.laue_group_of_material(icfg['material']['key_material'], icfg['material']['mymaterials'])
    return group


# default plane families of pole figures for each Laue group
POLE_FIGURES_HKLS = {'m-3m': ((0, 0, 1), (0, 1, 1), (1, 1, 1)), '4/mmm': ((0, 0, 1), (1, 0, 0), (1, 1, 0)),
                     '6/mmm': ((0, 0, 1), (1, 0, 0), (1, 1, 0)), '-3m1': ((0, 0, 1), (1, 0, 0), (1, 1, 0))}


def pole_figures(grains: dict, icfg: dict, hkls=None, mask: Optional[np.ndarray] = None,
                 sample_tilt: float = 40., equivalents: bool = True, markersize: float = 3):
    """pole figures (stereographic projection, upper hemisphere, sample frame: X right, Y up, Z sample normal at the
    center) of the plane families hkls of the best grain of each point (see load_grains()), any crystal symmetry.
    hkls: Miller indices of the planes, one pole figure each (None: {001} {011} {111} for cubic, {001} {100} {110}
    for tetragonal, hexagonal, trigonal, else {001} {010} {100})
    Points are colored as in the IPF-Z map, to relate a pole to its points of the map.
    equivalents: also the planes equivalent by the Laue group (postprocess: laue_group)
    mask: (True = masked) points left out (e.g. from quality_maps())"""
    import matplotlib.pyplot as plt
    from LaueTools import ipfmap as IPF
    from LaueTools import CrystalParameters as CP
    from LaueTools import dict_LaueTools as DictLT
    mat, mym = icfg['material']['key_material'], icfg['material']['mymaterials']
    latticeparams = (mym[mat] if mym and mat in mym else DictLT.dict_Materials[mat])[1]
    B0 = CP.calc_B_RR(latticeparams)
    group = laue_group(icfg)
    if hkls is None:
        hkls = POLE_FIGURES_HKLS.get(group, ((0, 0, 1), (0, 1, 0), (1, 0, 0)))
    UB = np.array(grains['best_UB'], dtype=float)
    if mask is not None:
        UB[np.asarray(mask, dtype=bool).ravel()[:len(UB)]] = np.nan
    colors = IPF.ipf_colors(IPF.sample_directions_in_crystal(UB, 'z', sample_tilt), group)
    fig, axes = plt.subplots(1, len(hkls) + 1, figsize=(4 * len(hkls) + 3.2, 4.2), squeeze=False,
                             gridspec_kw={'width_ratios': [1] * len(hkls) + [0.8]})
    IPF.plot_pole_figures(UB, hkls, B0, group if equivalents else None, anglesample=sample_tilt, colors=colors,
                          axes=axes[0][:-1], markersize=markersize, title_prefix=f'{mat} ')
    IPF.plot_ipf_key(group, ax=axes[0][-1], title=f'{group}: colors (IPF-Z)')
    fig.tight_layout()
    return fig


def plot_clusters(result, analyzer, icfg: dict, min_size: int = 1, ax=None):
    """map of the grains (clusters) with the y motor along the vertical axis"""
    from LaueTools import orientationclustering as OC
    scan = icfg['scan']
    transposed = map_transposed(icfg)
    xlabel, ylabel = GT.map_axes_labels(scan['fast_motor'], scan['slow_motor'], transposed)
    return OC.plot_cluster_map(result, analyzer, min_size=min_size, ax=ax, transpose=transposed,
                               xlabel=xlabel, ylabel=ylabel)


def segmentation(icfg: dict, threshold: float = 1., min_cluster_size: int = 10):
    """grains: clusters of neighbouring map points with misorientation < threshold (deg), all grains of the
    .fit files. Returns (result, analyzer) of orientationclustering.analyze_ub_matrices()"""
    from LaueTools import orientationclustering as OC
    scan, post = icfg['scan'], icfg['postprocess']
    layout = IB.fitfiles_layout(icfg)
    return OC.analyze_ub_matrices(input_dir=icfg['output']['fit_folder'], threshold=threshold,
                                  mapdimension=scan['invmapdims'], prefix=scan['prefix'], symmetry=post['symmetry'],
                                  spatial_connectivity=True, connectivity_type='8', min_cluster_size=min_cluster_size,
                                  check_symmetry=False, nbfiles_per_folder=layout['fit_nbfiles_per_folder'],
                                  subfolder_prefix=layout['subfolder_prefix'])


# --------------------------------------------------------------------------------------
#  SERIES OF MAPS with the parameters of a reference map
# --------------------------------------------------------------------------------------
def process_maps(maps, indices: List[int], reference: dict, key_material: str, detfile: Union[str, Path],
                 output_root: Optional[Union[str, Path]] = None, **run_kwargs) -> dict:
    """peak search and indexation of several maps with the configuration files of a reference map (info dict of
    scan_info()). run_kwargs: arguments of run_step() (execution, nb_cpus, machine ...).
    detfile 'auto': for each map, the last calibration acquired before it (detfile_for_map())
    Returns {scan index: indexing configuration} of the processed maps. Maps already done are skipped."""
    refpaths = config_paths(reference)
    done = {}
    for k, index in enumerate(indices):
        print(f'\n========== map {k + 1}/{len(indices)}: scan {index} ==========')
        try:
            info = scan_info(maps, index, output_root)
            print(info['name'], '|', info['command'])
            det = detfile_for_map(maps, index, max_age=3600) if str(detfile) == 'auto' else detfile
            if det is None:
                raise FileNotFoundError('no .det calibration file')
            pcfg, icfg = make_configs(info, key_material, det, peaksearch_template=refpaths['peaksearch'],
                                      indexing_template=refpaths['indexing'])
            if run_step('peaksearch', pcfg, **run_kwargs) is None:
                continue
            if run_step('indexing', icfg, **run_kwargs) is None:
                continue
            done[index] = icfg
        except Exception as err:   # next map
            GT.printred(f'scan {index}: {type(err).__name__}: {err}')
    GT.printgreen(f'\n{len(done)}/{len(indices)} maps processed')
    return done


def peaksearch_maps(maps, indices: List[int], reference: dict, key_material: str, detfile: Union[str, Path],
                    output_root: Optional[Union[str, Path]] = None, save_figures: Optional[str] = 'gallery',
                    redo: bool = False, submit_all: bool = False, **run_kwargs):
    """peak search of several maps (e.g. all the maps ended with SUCCESS) with the peak search parameters of a
    reference map (info dict of scan_info(): its peaksearch.yaml is the template of the maps without settings files;
    maps with settings files keep theirs). Maps of another detector than the reference are skipped.
    For each map: .cor files and peaksearch_peaks.pickle in PROCESSED_DATA (as run_step()), and the map of the nb of
    peaks saved by savefigresults() (where=save_figures; not shown: see FigureBrowser).
    Maps already done with the same parameters (and their figure) are skipped, unless redo=True.
    detfile: calibration of all the maps, or 'auto': for each map, the last calibration acquired before it
        (detfile_for_map(); a map with another calibration than before is searched again)
    run_kwargs: arguments of run_step() (execution, nb_cpus, machine ...)
    submit_all (execution 'slurm'): the jobs of all the maps are submitted first (they run at the same time if the
        cluster has free nodes), then waited for one after the other. Interrupted wait: run again, the jobs are
        not submitted again. Without submit_all: one job at a time.
    Returns a table (pandas DataFrame) of the maps: dataset, images, mean nb of peaks, status"""
    import pandas as pd
    from matplotlib.figure import Figure
    refpaths = config_paths(reference)
    submit_all = submit_all and run_kwargs.get('execution') == 'slurm'
    rows = {}
    t0 = time.time()

    def prepare(k, index):
        """peak search configuration of a map, None if skipped (detector, already done)"""
        row = rows[index] = {'dataset': maps.loc[index, 'dataset'] if index in maps.index else '?',
                             'images': None, 'mean nb peaks': None, 'status': '', 'detfile': ''}
        print(f"\n========== map {k + 1}/{len(indices)}: scan {index} {row['dataset']} ==========")
        info = scan_info(maps, index, output_root)
        if info['CCDLabel'] != reference['CCDLabel']:
            row['status'] = f"skipped: detector {info['CCDLabel']}"
            GT.printyellow(f"{row['status']} (reference map: {reference['CCDLabel']})")
            return None
        det = detfile_for_map(maps, index, max_age=3600) if str(detfile) == 'auto' else detfile
        if det is None:
            row['status'] = 'skipped: no .det calibration file'
            return None
        row['detfile'] = Path(det).name
        pcfg, _ = make_configs(info, key_material, det, peaksearch_template=refpaths['peaksearch'],
                               indexing_template=refpaths['indexing'], verbose=False)
        resfile = _results_file('peaksearch', pcfg)
        figfile = Path(pcfg['output']['results_folder']) / 'peaksearch_nbpeaks.png'
        fpfile = _fingerprint_file(resfile)
        if (not redo and resfile.exists() and fpfile.exists()
                and fpfile.read_text().strip() == _fingerprint('peaksearch', pcfg)
                and (not save_figures or figfile.exists())):
            row['status'] = 'done before'
            GT.printgreen(f'already done (results and figure): {resfile.parent}')
            return None
        return pcfg

    def finish(index, pcfg, redo_map):
        """peak search of a map (or results of its job), summary and figure"""
        row = rows[index]
        peaks = run_step('peaksearch', pcfg, redo=redo_map, **run_kwargs)
        if peaks is None:
            row['status'] = 'not done (job)'
            return
        summary = PB.summarize_results(peaks, pcfg['scan']['mapdims'])
        del peaks
        ok = summary['nbpeaks'] >= 0
        row['images'] = len(summary['nbpeaks'])
        row['mean nb peaks'] = round(float(np.mean(summary['nbpeaks'][ok])), 1) if np.any(ok) else 0
        row['status'] = 'done'
        if summary['toomanyimages']:
            row['status'] = (f"done, {len(summary['toomanyimages'])} images NOT FITTED: NumberMaxofFits >= "
                             f"{summary['nb_local_maxima_max'] + 1} needed")
        if save_figures:
            fig = Figure(figsize=(5.5, 4.5))   # not a pyplot figure: nothing displayed in the notebook
            plot_map(summary['nbpeaks2D'], pcfg, f"{row['dataset']}: nb of peaks", ax=fig.subplots())
            savefigresults(fig, pcfg, 'peaksearch_nbpeaks', where=save_figures)

    def error(index, err):
        rows[index]['status'] = f'error: {type(err).__name__}: {err}'
        GT.printred(f"scan {index}: {rows[index]['status']}")

    if not submit_all:
        for k, index in enumerate(indices):
            try:
                pcfg = prepare(k, index)
                if pcfg is not None:
                    finish(index, pcfg, redo)
            except Exception as err:   # next map
                error(index, err)
    else:
        submitted = {}
        for k, index in enumerate(indices):   # 1. all jobs submitted
            try:
                pcfg = prepare(k, index)
                if pcfg is not None:
                    run_step('peaksearch', pcfg, redo=redo, wait=False, **run_kwargs)
                    submitted[index] = pcfg
                    rows[index]['status'] = 'submitted'
            except Exception as err:
                error(index, err)
        GT.printgreen(f'\n{len(submitted)} jobs submitted: waiting for them one after the other')
        for k, (index, pcfg) in enumerate(submitted.items()):   # 2. results of each job
            print(f"\n========== results {k + 1}/{len(submitted)}: scan {index} {rows[index]['dataset']} ==========")
            try:
                finish(index, pcfg, False)   # (redo=False: the job submitted above is waited for, not a new one)
            except Exception as err:
                error(index, err)
    refresh_done(maps)
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.index.name = 'index'
    ndone = int(np.sum(table['status'].str.startswith('done'))) if len(table) else 0
    GT.printgreen(f'\n{ndone}/{len(indices)} maps with peak search results ({(time.time() - t0) / 60:.1f} min)')
    return table


class FigureBrowser:
    """browser of the figures (.png) saved by savefigresults() for the maps of the table of list_maps(): menu of the
    figures (filter by words, '-word': without it; figure name), up/down arrow keys to go from one to the next.
    where: 'gallery' (GALLERY folder of each map, if it exists, else PROCESSED_DATA) or 'results' (PROCESSED_DATA)
    Only the folders of the maps are read (not all PROCESSED_DATA)"""

    def __init__(self, maps, output_root: Optional[Union[str, Path]] = None, where: str = 'gallery'):
        import ipywidgets as widgets
        from IPython.display import display
        output_root = maps.attrs.get('output_root') if output_root is None else output_root
        self.files = []   # (label, path)
        for index in maps.index:
            try:
                workdir = processed_folder(maps.at[index, 'image_folder'], output_root)
            except ValueError:
                continue
            gallery = gallery_folder(workdir) if where == 'gallery' else None
            folder = gallery if gallery is not None and gallery.is_dir() else workdir
            if not folder.is_dir():
                continue
            for sub in sorted(p for p in folder.iterdir() if p.is_dir() and '_previous_' not in p.name):
                for png in sorted(sub.glob('*.png')):
                    self.files.append((f"{index:5d} | {maps.at[index, 'dataset']} | {sub.name}/{png.name}", png))
        names = sorted({png.stem for _, png in self.files})
        self.filter = widgets.Text(description='filter:', placeholder="words of the menu lines, '-word': without it",
                                   continuous_update=False, layout=widgets.Layout(width='45%'))
        self.name = widgets.Dropdown(options=['all figures'] + names, description='figure:')
        self.count = widgets.HTML()
        self.menu = widgets.Select(options=[], rows=8, layout=widgets.Layout(width='98%'))
        self.menu.add_class('lt-fig-menu')
        style = widgets.HTML('<style>.lt-fig-menu select {font-family: monospace; font-size: 12px}</style>')
        self.path = widgets.HTML()
        self.image = widgets.Image(format='png', layout=widgets.Layout(max_width='100%'))
        self.menu.observe(self._show, names='value')
        self.filter.observe(self._apply_filter, names='value')
        self.name.observe(self._apply_filter, names='value')
        self._apply_filter(None)
        if not self.files:
            self.path.value = (f'no figure (.png) in the folders of the {len(maps)} maps of the table '
                               '(saved by MW.savefigresults(), SAVE_FIGURES)')
        display(widgets.VBox([style, widgets.HBox([self.filter, self.name, self.count]), self.menu, self.path,
                              self.image]))

    def _apply_filter(self, change):
        options = [(label, str(png)) for label, png in self.files if _words_match(label, self.filter.value)
                   and self.name.value in ('all figures', png.stem)]
        self.count.value = f'&nbsp;{len(options)} / {len(self.files)} figures'
        self.menu.options = options
        if options:
            self.menu.value = options[0][1]

    def _show(self, change):
        if self.menu.value is None:
            return
        png = Path(self.menu.value)
        self.path.value = f'<code>{png}</code> ({time.strftime("%Y-%m-%d %H:%M", time.localtime(png.stat().st_mtime))})'
        self.image.value = png.read_bytes()
