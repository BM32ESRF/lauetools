#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
daxm_workflow: workflows of DAXM wire scans driven by a YAML configuration file
(template: LaueTools/notebooks/daxm/config_template.yaml), from a notebook, the command line, in
this process or as a job on this machine or on a SLURM partition (scripts/workflowlauncher).

Workflows:
    quickdaxm: depth of the Laue spots of a wire scan from the dips of their intensity profile,
        with the wires trajectory parameters of a .calib file (see LaueTools/Daxm/quickdaxm.py)

From a notebook::

    import LaueTools.daxm_workflow as DW
    cfg = DW.new_config('configs/myscan.yaml', scanfile='..._calib.scan', calibfile='....calib')
    # or, without .scan file:  hdf5_scan=('/data/.../<dataset>/<dataset>.h5', 1), detfile='....det'
    results = DW.run_quickdaxm(cfg, resource='magnifix')      # or 'kernel', 'local'
    files = DW.write_quickdaxm_files(cfg, results)            # .cor with depth column, .cor per slice

Command line::

    python -m LaueTools.daxm_workflow quickdaxm configs/myscan.yaml --resource magnifix   # whole node (192 cpus)
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from LaueTools import generaltools as GT
from LaueTools import IOLaueTools as IOLT
from LaueTools.Daxm import quickdaxm as QD

TEMPLATE_CONFIG = Path(__file__).parent / 'notebooks' / 'daxm' / 'config_template.yaml'

DEFAULT_CONFIG = {
    'name': 'daxm scan',
    'description': '',
    'scan': {'image_folder': None, 'prefix': 'img_', 'suffix': '.tif', 'nbdigits': 4, 'first_image_index': 0,
             'nb_images': None, 'CCDLabel': 'sCMOS', 'nbframes_per_file': 1, 'hdf5_file': None,
             'scan_number': None, 'hdf5_node': None, 'wire_motor': 'zf', 'wire_start': None, 'wire_stop': None},
    'detector': {'detfile': None},
    'wires': {'calibfile': None, 'yref': 0.},
    'sample': {'tilt': 40., 'detector_offset': 'auto'},
    'output': {'folder': None},
    'quickdaxm': {**QD.DEFAULT_PARAMS, 'slice_step': 5., 'slice_thickness': None, 'slice_min_spots': 6},
    'wirescalib': {'nb_wires': None, 'var': ['h', 'p0', 'f1', 'R'], 'ybeam': 0., 'fluo_element': 'Ge',
                   'fluo_thickness': 0.3, 'monitor_offset': None, 'fluo_layer': None, 'fluo_layer_max': 0.08,
                   'fluo_layer_fixed': False},
}

_PATH_KEYS = (('scan', 'image_folder'), ('scan', 'hdf5_file'), ('detector', 'detfile'),
              ('wires', 'calibfile'), ('output', 'folder'), ('quickdaxm', 'peaklist'))


# --------------------------------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------------------------------
def _key_path(name: str, tree: dict = DEFAULT_CONFIG) -> tuple:
    """path of keys of parameter name in the configuration (e.g. 'detfile' -> ('detector', 'detfile'))"""
    if name in tree and not isinstance(tree[name], dict):
        return (name,)
    found = [(section, name) for section, values in tree.items() if isinstance(values, dict) and name in values]
    if len(found) != 1:
        raise KeyError(f"unknown parameter '{name}'" if not found else f"ambiguous parameter '{name}'")
    return found[0]


def _expand(path, folder: Path) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return str(path if path.is_absolute() else (folder / path).resolve())


def mirror_folder(image_folder: Union[str, Path]) -> str:
    """default results folder: <PROCESSED_DATA mirror of image_folder>/daxm"""
    parts = list(Path(image_folder).parts)
    if 'RAW_DATA' in parts:
        parts[len(parts) - 1 - parts[::-1].index('RAW_DATA')] = 'PROCESSED_DATA'
    return str(Path(*parts) / 'daxm')


def load_config(configfile: Union[str, Path]) -> dict:
    """read a daxm YAML configuration file: default values added, paths made absolute"""
    configfile = Path(configfile).resolve()
    with open(configfile, 'r') as f:
        usercfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in usercfg.items():
        if key == 'quickdaxm' and value:   # former names of parameters (e.g. coresize -> boxsize_probe)
            value = {QD.PARAM_ALIASES.get(name, name): v for name, v in value.items()}
        if isinstance(cfg.get(key), dict):
            unknown = set(value or {}) - set(cfg[key])
            if unknown:
                raise KeyError(f"unknown parameter(s) {sorted(unknown)} in section '{key}' of {configfile}")
            cfg[key].update(value or {})
        else:
            cfg[key] = value
    for section, key in _PATH_KEYS:
        value = cfg[section][key]
        if value is not None and not (key == 'peaklist' and str(value).lower() == 'auto'):
            cfg[section][key] = _expand(value, configfile.parent)
    if cfg['output']['folder'] is None and cfg['scan']['image_folder']:
        cfg['output']['folder'] = mirror_folder(cfg['scan']['image_folder'])
    cfg['configfile'] = str(configfile)
    return cfg


def _yaml_value(value) -> str:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, (Path, np.str_)):
        value = str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        value = [v.item() if isinstance(v, np.generic) else v for v in value]
    return yaml.safe_dump(value, default_flow_style=True, width=10000).strip().removesuffix('...').strip()


def _update_text(text: str, changes: Dict[str, Any]) -> str:
    """set the values of parameters in the text of a YAML file, keeping its comments"""
    lines = text.split('\n')
    for name, value in changes.items():
        path = _key_path(name)
        section = None if len(path) == 1 else path[0]
        current = None
        for k, line in enumerate(lines):
            top = re.match(r'^(\w+):', line)
            if top:
                current = top.group(1)
            indent = '' if section is None else r'\s+'
            match = re.match(rf'^({indent}{re.escape(path[-1])}:)(\s*)(.*?)(\s+#.*)?$', line)
            if match and (section is None and current == path[-1] or section is not None and current == section):
                lines[k] = f"{match.group(1)} {_yaml_value(value)}{match.group(4) or ''}"
                break
        else:   # parameter not in the file: added at the end of its section
            if section is None:
                lines.append(f'{name}: {_yaml_value(value)}')
            else:
                start = next((k for k, line in enumerate(lines) if line.startswith(f'{section}:')), None)
                if start is None:
                    lines += [f'{section}:', f'  {path[-1]}: {_yaml_value(value)}']
                else:
                    stop = next((k for k in range(start + 1, len(lines)) if re.match(r'^\S', lines[k])
                                 and not lines[k].startswith('#')), len(lines))
                    while stop > start + 1 and (not lines[stop - 1].strip() or lines[stop - 1].startswith('#')):
                        stop -= 1
                    lines.insert(stop, f'  {path[-1]}: {_yaml_value(value)}')
    return '\n'.join(lines)


def scan_parameters_from_scanfile(scanfile: Union[str, Path]) -> Dict[str, Any]:
    """scan and detector parameters of a .scan file (json) of the DAXM GUI or of the wires calibration notebook"""
    with open(scanfile, 'r') as f:
        scan = json.load(f)
    from LaueTools import dict_LaueTools as DictLT
    suffix = '.' + str(DictLT.dict_CCD[scan['CCDType']][7]).lstrip('.')
    start, stop, nbsteps = scan['scanCmd'][:3]
    return {'image_folder': scan['imageFolder'], 'prefix': scan['imagePrefix'], 'suffix': suffix,
            'nbdigits': scan['imageDigits'], 'first_image_index': scan['imageFirstIndex'],
            'nb_images': int(nbsteps) + 1, 'CCDLabel': scan['CCDType'],
            'hdf5_file': scan['specFile'] if scan.get('filetype') == 'hdf5' else None,
            'scan_number': scan.get('scanNumber'),
            'hdf5_node': f"{scan['hdf5scanId']}.1" if scan.get('hdf5scanId') else None,
            'wire_start': start, 'wire_stop': stop,
            'detfile': scan['detCalib'] if isinstance(scan['detCalib'], str) else None,
            'name': scan.get('hdf5scanId') or Path(scanfile).stem}


def scan_parameters_from_hdf5(hdf5_file: Union[str, Path], scan_number: int = 1,
                              CCDLabel: Optional[str] = None) -> Dict[str, Any]:
    """scan parameters of wire scan scan_number (ascan or fscan of the wire motor) read in the BLISS hdf5
    file of its dataset (<dataset>/<dataset>.h5, images in <dataset>/scanXXXX)

    :param CCDLabel: default: EIGER_4MCdTe for .h5 images (EIGER_4MCdTestack if all images are in one file),
        sCMOS for .tif images
    """
    import glob
    import h5py
    hdf5_file = Path(hdf5_file)
    with h5py.File(hdf5_file, 'r', locking=False) as f:
        node = f[f'{int(scan_number)}.1']
        command = node['title'][()]
        command = (command.decode() if isinstance(command, bytes) else str(command)).split()
        motor = command[1]
        if command[0] not in ('ascan', 'fscan') or motor not in ('zf', 'yf'):
            raise ValueError(f"scan {scan_number} of {hdf5_file} is not a wire scan: {' '.join(command)}")
        positions = node[f'measurement/{motor}'][()]
    image_folder = hdf5_file.parent / f'scan{int(scan_number):04d}'
    # image files <prefix><digits>.<h5, tif...>: most frequent prefix (other files may be in the folder)
    matches = [re.match(r'^(.*?)(\d+)(\.(?:h5|tif|tiff|edf|mccd))$', os.path.basename(path))
               for path in sorted(glob.glob(str(image_folder / '*')))]
    matches = [m for m in matches if m]
    if not matches:
        raise FileNotFoundError(f'no image in {image_folder}')
    prefixes = [m.group(1) + m.group(3) for m in matches]
    main = max(set(prefixes), key=prefixes.count)
    matches = [m for m, key in zip(matches, prefixes) if key == main]
    files = [m.group(0) for m in matches]
    prefix, digits, suffix = matches[0].groups()
    stacked = suffix == '.h5' and len(files) < len(positions)
    if CCDLabel is None:
        CCDLabel = ('EIGER_4MCdTestack' if stacked else 'EIGER_4MCdTe') if suffix == '.h5' else 'sCMOS'
    nbframes = int(np.ceil(len(positions) / len(files))) if stacked else 1
    return {'image_folder': str(image_folder), 'prefix': prefix, 'suffix': suffix, 'nbdigits': len(digits),
            'first_image_index': 0 if stacked else int(digits), 'nb_images': len(positions),
            'CCDLabel': CCDLabel, 'nbframes_per_file': nbframes, 'hdf5_file': str(hdf5_file),
            'scan_number': int(scan_number), 'wire_motor': motor, 'wire_start': float(positions[0]),
            'wire_stop': float(positions[-1]), 'name': f'{hdf5_file.stem}_{int(scan_number)}'}


# wire scan command: ascan / fscan / dscan of a wire motor
_WIRE_SCAN = re.compile(r'^\s*(ascan|fscan|dscan)\s+(zf|yf)\s')


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


def _wire_scans_of_dataset(hdf5_file: str) -> list:
    """wire scans of the hdf5 file of a dataset (file opened once)"""
    import h5py
    rows = []
    try:
        with h5py.File(hdf5_file, 'r', locking=False) as f:
            for node in f.keys():
                try:
                    scan = f[node]
                    title = scan['title'][()]
                    title = title.decode() if isinstance(title, bytes) else str(title)
                    match = _WIRE_SCAN.match(title)
                    if not match:
                        continue
                    motor = match.group(2)
                    start = scan['start_time'][()] if 'start_time' in scan else b''
                    end = scan['end_reason'][()] if 'end_reason' in scan else b''
                    npts = scan[f'measurement/{motor}'].shape[0] if f'measurement/{motor}' in scan else None
                except (KeyError, OSError, ValueError):
                    continue
                number = int(float(node))
                folder = os.path.join(os.path.dirname(hdf5_file), f'scan{number:04d}')
                try:
                    nbfiles = sum(1 for e in os.scandir(folder) if e.name.endswith(('.h5', '.tif', '.tiff', '.edf')))
                except OSError:
                    nbfiles = 0
                rows.append({'dataset': f'{Path(hdf5_file).stem}_{number}', 'scan': number, 'command': title,
                             'motor': motor, 'start': start.decode() if isinstance(start, bytes) else str(start),
                             'points': npts, 'image_files': nbfiles,
                             'end': end.decode() if isinstance(end, bytes) else str(end),
                             'image_folder': folder, 'hdf5_file': hdf5_file})
    except OSError:
        pass
    return rows


def list_wire_scans(raw_data_folder: Union[str, Path], contains: Optional[str] = None,
                    nb_cpus: Optional[int] = None, output_root: Optional[Union[str, Path]] = None,
                    verbose: bool = True):
    """table (pandas DataFrame) of the wire scans (ascan, fscan, dscan of zf or yf) of an experiment, read directly
    in the hdf5 files of the datasets (RAW_DATA/<sample>/<dataset>/<dataset>.h5), each file opened once, in
    parallel: a few seconds even for thousands of scans (the master hdf5 file of the experiment is not read)

    :param contains: filter of the datasets: words of the dataset file name (case insensitive), a word starting
        with '-' must not be in it, e.g. 'daxm -grid'
    :param output_root: root folder of the results (None: PROCESSED_DATA of the experiment), see workdir()
    :return: DataFrame sorted by start time, columns dataset (<dataset>_<scan>, as the name of the scan in the
        master file), scan, command, motor, start, points (recorded wire positions), image_files (in the scan
        folder), end, image_folder, hdf5_file, map and images (labels of the menu of WireScanSelector),
        workdir (folder of the results), done (results already there, see daxm_status())
    """
    import glob
    import multiprocessing
    import time
    import pandas as pd
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(str(raw_data_folder), '*', '*', '*.h5')))
    files = [path for path in files if os.path.basename(path)[:-3] == os.path.basename(os.path.dirname(path))
             and _words_match(os.path.basename(path)[:-3], contains)]
    nb_cpus = max(1, min(nb_cpus or 32, len(files) // 20 + 1, len(os.sched_getaffinity(0))))
    if nb_cpus == 1:
        rows = [row for path in files for row in _wire_scans_of_dataset(path)]
    else:
        with multiprocessing.Pool(nb_cpus) as pool:
            rows = [row for res in pool.map(_wire_scans_of_dataset, files) for row in res]
    table = pd.DataFrame(rows, columns=['dataset', 'scan', 'command', 'motor', 'start', 'points', 'image_files',
                                        'end', 'image_folder', 'hdf5_file'])
    table = table.sort_values('start').reset_index(drop=True)
    table['map'] = [f"{m} {'?' if p != p or p is None else int(p)} pts" for m, p in zip(table['motor'], table['points'])]
    table['images'] = [f'{n} file' + ('s' if n != 1 else '') for n in table['image_files']]
    table['workdir'] = [str(workdir(folder, output_root)) for folder in table['image_folder']]
    table['done'] = [daxm_status(folder) for folder in table['workdir']]
    table.attrs['output_root'] = output_root
    if verbose:
        print(f'{len(table)} wire scans in {len(files)} datasets of {raw_data_folder} ({time.time() - t0:.1f} s)')
    return table


def select_wire_scan(table, scan: Union[int, str]) -> Dict[str, Any]:
    """row of the table of list_wire_scans(): scan = index (left column) or dataset name (column 'dataset')"""
    if isinstance(scan, str):
        rows = table[table['dataset'] == scan]
        if not len(rows):
            raise KeyError(f"no wire scan named {scan!r} (column 'dataset')")
        return rows.iloc[0].to_dict()
    return table.loc[int(scan)].to_dict()


def workdir(image_folder: Union[str, Path], output_root: Optional[Union[str, Path]] = None) -> Path:
    """folder of the DAXM results of a scan: <image folder with RAW_DATA replaced by PROCESSED_DATA (or output_root)>
    /daxm, e.g. PROCESSED_DATA/<sample>/<dataset>/scan0001/daxm"""
    parts = Path(image_folder).parts
    if 'RAW_DATA' not in parts:
        if output_root is None:
            raise ValueError(f'{image_folder} is not in a RAW_DATA folder: set OUTPUT_ROOT (folder of the results)')
        return Path(output_root) / Path(image_folder).name / 'daxm'
    k = len(parts) - 1 - parts[::-1].index('RAW_DATA')
    base = Path(*parts[:k], 'PROCESSED_DATA') if output_root is None else Path(output_root)
    return base.joinpath(*parts[k + 1:], 'daxm')


def daxm_status(folder: Union[str, Path]) -> str:
    """DAXM results already in the folder of the results of a scan (workdir()): 'config', 'quickdaxm',
    'slices:<n>' (.cor files of slices), 'wires:dips' (approximate wires from the dips of a reference crystal),
    'wires:fluo' (wires refined with the fluorescence), 'wires:envelope' (approximate wires from the envelope of the
    dips of the scan itself), 'calib:<n>' (.calib files of the former notebooks)"""
    folder = Path(folder)
    status = []
    if (folder / 'configs' / 'daxm.yaml').exists():
        status.append('config')
    if (folder / 'quickdaxm.h5').exists():
        status.append('quickdaxm')
    slices = folder / 'depthslices'
    if slices.is_dir():
        status.append(f"slices:{sum(1 for p in slices.glob('*.cor'))}")
    wires = folder / 'wirescalib'
    if (wires / 'wires_from_dips.calib').exists():
        status.append('wires:dips')
    if wires.is_dir() and any(wires.glob('*_stage*.calib')):
        status.append('wires:fluo')
    if (wires / 'wires_from_envelope.calib').exists():
        status.append('wires:envelope')
    legacy = list(folder.parent.glob('calibration/*.calib'))   # 1_Pre_Calib_wire notebook, DAXM GUI
    if legacy:
        status.append(f'calib:{len(legacy)}')
    return ', '.join(status)


def refresh_status(table):
    """update the column 'done' of the table of list_wire_scans() (e.g. after analyses)"""
    table['done'] = [daxm_status(folder) for folder in table['workdir']]
    return table


def show_wire_scans(table, nmax: Optional[int] = None):
    """display the table of list_wire_scans() (main columns)"""
    from IPython.display import display
    import pandas as pd
    if not len(table):
        GT.printred('no wire scan: change SCAN_FILTER (or RAW_DATA_FOLDER)')
        return
    with pd.option_context('display.max_rows', nmax or len(table), 'display.max_colwidth', 70, 'display.width', 250):
        display(table[['dataset', 'start', 'command', 'points', 'images', 'end', 'done']])


def scan_index(table, scan) -> Optional[int]:
    """index of a scan in the table of list_wire_scans(): index (int) or dataset name (str). None: None"""
    if scan is None or isinstance(scan, (int, np.integer)):
        return None if scan is None else int(scan)
    found = [int(i) for i in table.index if table.loc[i, 'dataset'] == str(scan)]
    if not found:
        raise KeyError(f'no wire scan {scan!r} in the table (filter SCAN_FILTER?)')
    return found[-1]


def WireScanSelector(table, index=None, only_success: bool = False):
    """menu to choose a wire scan in the table of list_wire_scans() (filter, 'only SUCCESS'): the scan browser of
    the maps (map_workflow.ScanSelector) with the DAXM results in the menu lines ('done'). Selected: .index"""
    import LaueTools.map_workflow as MW

    class _Selector(MW.ScanSelector):
        def _on_count_files(self, change):   # nb of .cor files of the depth slices are in 'done' anyway
            refresh_status(self.maps)
            self._make_labels()
            self._apply_filter(None)

    return _Selector(table, scan_index(table, index), only_success=only_success, refresh=False)


def is_reference_scan(dataset: str) -> bool:
    """wire scan of a reference crystal for the wires calibration: Ge in the dataset name ('_Ge', '_ge', 'Ge_'...)"""
    return re.search(r'(^|_)ge', str(dataset), flags=re.IGNORECASE) is not None


def detfile_for_scan(table, index, verbose: bool = True) -> Optional[Path]:
    """detector calibration .det file of a wire scan of the table of list_wire_scans(): the last one acquired before
    the scan, else the closest one after it (see map_workflow.find_detfiles())"""
    import LaueTools.map_workflow as MW
    return MW.detfile_for_map(table, index, verbose=verbose)


def calib_for_scan(table, index, verbose: bool = True) -> Optional[Path]:
    """wires calibration (.calib file) for a wire scan of the table of list_wire_scans(): the one of a reference
    (Ge) wire scan of the same sample folder (RAW_DATA/<sample>/...) if any, else of the whole table; among them the
    one acquired last before the scan (else the closest after it), preferably refined with the fluorescence
    (wirescalib/*_stage<last>.calib), else a .calib file of the former notebooks (<scan folder>/calibration), else
    the approximate wires from the dips (wirescalib/wires_from_dips.calib). None if there is none"""
    import datetime
    start = datetime.datetime.fromisoformat(str(table.at[index, 'start']))

    def sample_folder(image_folder):
        parts = Path(image_folder).parts
        return parts[len(parts) - parts[::-1].index('RAW_DATA')] if 'RAW_DATA' in parts else None
    sample = sample_folder(table.at[index, 'image_folder'])
    candidates = []
    for i, row in table.iterrows():
        chosen = existing_calibration(row['workdir'])
        if chosen is not None:
            when = datetime.datetime.fromisoformat(str(row['start']))
            candidates.append((when, chosen, row['dataset'], sample_folder(row['image_folder']) == sample))
    if any(c[3] for c in candidates):
        candidates = [c for c in candidates if c[3]]
    elif candidates and verbose:
        GT.printyellow(f'no wires calibration of a reference (Ge) wire scan in the sample folder {sample}: '
                       'calibration of another sample folder (wires and sample positions may differ)')
    if not candidates:
        if verbose:
            GT.printyellow('no wires calibration (.calib) in the results of the wire scans of the table: '
                           'calibrate the wires with a reference (Ge) wire scan (STEP 4) or set CALIBFILE')
        return None
    before = sorted((c for c in candidates if c[0] <= start), key=lambda c: c[0], reverse=True)
    after = sorted((c for c in candidates if c[0] > start), key=lambda c: c[0])
    when, chosen, dataset, _ = (before + after)[0]
    if verbose:
        print(f"wires calibration: {chosen}\n  (of {dataset}, acquired {when.strftime('%Y-%m-%d %H:%M')}"
              f"{' before' if before else ' after'} the scan)")
    return chosen


def _sample_folder(image_folder) -> Optional[str]:
    """sample folder of an image folder RAW_DATA/<sample>/<dataset>/scanXXXX"""
    parts = Path(image_folder).parts
    return parts[len(parts) - parts[::-1].index('RAW_DATA')] if 'RAW_DATA' in parts else None


def _raw_data_folder(image_folder) -> Optional[Path]:
    parts = Path(image_folder).parts
    return Path(*parts[:len(parts) - parts[::-1].index('RAW_DATA')]) if 'RAW_DATA' in parts else None


def existing_calibration(workdir: Union[str, Path]) -> Optional[Path]:
    """wires calibration already made with a wire scan (results folder workdir()), the best one: refined with the
    fluorescence (wirescalib/*_stage<n>.calib, the most recent), else a .calib file of the former notebooks
    (<scan folder>/calibration), else the approximate wires from the dips of a reference crystal
    (wires_from_dips.calib), else from the envelope of the dips of the sample (wires_from_envelope.calib). None"""
    folder = Path(workdir)
    stages = sorted((folder / 'wirescalib').glob('*_stage*.calib'), key=lambda p: p.stat().st_mtime)
    legacy = sorted(folder.parent.glob('calibration/*.calib'), key=lambda p: p.stat().st_mtime)
    for candidates in (stages, legacy, [folder / 'wirescalib' / 'wires_from_dips.calib'],
                       [folder / 'wirescalib' / 'wires_from_envelope.calib']):
        existing = [p for p in candidates if p.exists()]
        if existing:
            return existing[-1]
    return None


def reference_wires(table, index, detfile: Optional[Union[str, Path]] = 'auto', nb_cpus: Optional[int] = None,
                    verbose: bool = True) -> Optional[Path]:
    """wires calibration of a reference (Ge) wire scan of the table of list_wire_scans(), computed if needed: its
    wires refined with the fluorescence if done (wirescalib/*_stage<last>.calib), else a .calib file of the former
    notebooks (<scan folder>/calibration), else its approximate wires from the dips of its Laue spots
    (wirescalib/wires_from_dips.calib), computed now if needed (quick daxm of the reference scan in this process, then
    approximate_wires(), with the parameters of the configuration file of the reference scan)"""
    row = table.loc[index]
    chosen = existing_calibration(row['workdir'])
    if chosen is None:
        if verbose:
            print(f"wires calibration from the reference wire scan {row['dataset']}: quick daxm of its dips, then "
                  'approximate wires')
        cfg = config_for_scan(table, index, detfile=detfile, calibfile=None, verbose=verbose)
        results = run_quickdaxm(cfg, resource='kernel', nb_cpus=nb_cpus)
        chosen = Path(approximate_wires(cfg, results, verbose=verbose)['calibfile'])
    if verbose:
        print(f"wires calibration: {chosen}\n  (of the reference wire scan {row['dataset']}, {row['start'][:16]})")
    return chosen


def write_wires(wires: Sequence[dict], path: Union[str, Path], u1: float = 0., u2: float = 0.,
                excluded_wires: Optional[Sequence[int]] = None) -> Path:
    """write wires parameters given by hand in a .calib file: list of dict per wire with h, p0 (mm), optional
    R (mm, default 0.025), f1, f2 (degrees, default 0), material (default W); u1, u2: trajectory angles (degrees)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {'traj': {'u1': float(u1), 'u2': float(u2)},
               'wires': [{'material': w.get('material', 'W'), 'R': float(w.get('R', 0.025)), 'p0': float(w['p0']),
                          'h': float(w['h']), 'f1': float(w.get('f1', 0.)), 'f2': float(w.get('f2', 0.))}
                         for w in wires]}
    if excluded_wires:
        content['excluded_wires'] = sorted(int(w) for w in excluded_wires)
    path.write_text(json.dumps(content, indent=1))
    return path


def resolve_calibfile(table, index, calib='auto', detfile: Optional[Union[str, Path]] = 'auto',
                      nb_cpus: Optional[int] = None, verbose: bool = True) -> Optional[Path]:
    """wires calibration (.calib file) to use for the wire scan index of the table of list_wire_scans(), from calib:

    - 'auto': the calibration made with a wire scan of the same sample folder (RAW_DATA/<sample>) acquired last
      before the scan (else the closest after it), see existing_calibration() (e.g. with notebook 1_wires_calibration);
      else the reference (Ge) wire scan of the same sample folder, see reference_wires(): refined wires if done, else
      its approximate wires from its dips, computed if needed. Without reference scan in the sample folder: the most recent calibration of the
      wire scans of the table (calib_for_scan(), with a warning). None for a reference scan itself (its wires are
      calibrated from its own dips, STEP 4 of the notebook)
    - path of a .calib file
    - a reference wire scan: dataset name (e.g. 'Zr4_Gedaxm_1') or index of the table: see reference_wires()
    - wires parameters given by hand: list of dict per wire (h, p0 in mm, optional R, f1, f2 in degrees), written in
      <results folder>/wirescalib/wires_manual.calib
    - None: no wires calibration (dips only, no depth)
    """
    row = table.loc[index]
    if calib is None:
        return None
    if isinstance(calib, (list, tuple)):
        return write_wires(calib, Path(row['workdir']) / 'wirescalib' / 'wires_manual.calib')
    if isinstance(calib, (int, np.integer)):
        return reference_wires(table, int(calib), detfile, nb_cpus, verbose)
    calib = str(calib)
    if calib == 'auto':
        if is_reference_scan(row['dataset']):
            return None
        sample = _sample_folder(row['image_folder'])
        # calibrations already made with wire scans of the same sample folder (reference crystal, or the sample itself
        # with notebook 1_wires_calibration)
        same = table[[_sample_folder(f) == sample for f in table['image_folder']]]
        calibrated = [int(i) for i in same.index if existing_calibration(same.at[i, 'workdir']) is not None]
        if calibrated:
            chosen = _closest_before(same.loc[calibrated], row['start'])
            path = existing_calibration(table.at[chosen, 'workdir'])
            if verbose:
                print(f"wires calibration: {path}\n  (made with the wire scan {table.at[chosen, 'dataset']} of the same "
                      f"sample folder, {table.at[chosen, 'start'][:16]})")
            return path
        references = table[[_sample_folder(f) == sample and is_reference_scan(d)
                            for f, d in zip(table['image_folder'], table['dataset'])]]
        if not len(references):   # (reference scans maybe not in the table: filter of the table)
            raw = _raw_data_folder(row['image_folder'])
            others = list_wire_scans(raw, contains=sample, output_root=table.attrs.get('output_root'), verbose=False)
            others = others[[_sample_folder(f) == sample and is_reference_scan(d)
                             for f, d in zip(others['image_folder'], others['dataset'])]]
            if len(others):
                return reference_wires(others, _closest_before(others, row['start']), detfile, nb_cpus, verbose)
            return calib_for_scan(table, index, verbose)
        return reference_wires(table, _closest_before(references, row['start']), detfile, nb_cpus, verbose)
    if os.path.exists(calib):
        return Path(calib)
    return reference_wires(table, scan_index(table, calib), detfile, nb_cpus, verbose)


def _closest_before(table, start: str) -> int:
    """index of the scan of the table acquired last before start (else the closest after it)"""
    import datetime
    when = datetime.datetime.fromisoformat(str(start))
    times = {int(i): datetime.datetime.fromisoformat(str(t)) for i, t in zip(table.index, table['start'])}
    before = [i for i in times if times[i] <= when]
    return max(before, key=times.get) if before else min(times, key=lambda i: times[i] - when)


def config_for_scan(table, index, detfile: Optional[Union[str, Path]] = 'auto',
                    calibfile='auto', nb_cpus: Optional[int] = None, verbose: bool = True, **changes) -> dict:
    """write (or update) the configuration file <workdir>/configs/daxm.yaml of a wire scan of the table of
    list_wire_scans() and return the configuration (see new_config(): an existing file is kept, only the given
    parameters are updated)

    :param detfile: 'auto' (detfile_for_scan()), path, or None (keep the value of the existing file)
    :param calibfile: wires calibration, see resolve_calibfile(): 'auto', path of a .calib file, reference (Ge) wire
        scan (dataset name or index of the table), list of wires parameters, or None (no wires calibration: dips only)
    :param changes: other parameters by name, e.g. ybeam_range=[-50, 150]
    """
    row = table.loc[index]
    configfile = Path(row['workdir']) / 'configs' / 'daxm.yaml'
    if str(detfile) == 'auto':
        detfile = detfile_for_scan(table, index, verbose)
    calibfile = resolve_calibfile(table, index, calibfile, detfile, nb_cpus, verbose)
    if detfile is not None:
        changes['detfile'] = str(detfile)
    changes['calibfile'] = None if calibfile is None else str(calibfile)
    changes['folder'] = str(row['workdir'])
    return new_config(configfile, hdf5_scan=(row['hdf5_file'], int(row['scan'])), **changes)


def scan_parameters_from_logfile(scandict: Dict[str, Any]) -> Dict[str, Any]:
    """scan parameters of a wire scan selected in the BLISS hdf5 logfile (dict returned by
    logfile_reader.H5file.build_dict_scan())"""
    if scandict.get('scantype') != 'daxm':
        raise ValueError(f"not a wire scan: {scandict.get('fullcommand')}")
    name = scandict['sample_dataset_scanindex']
    scan_number = int(name.rsplit('_', 1)[1])
    if scandict.get('localhdf5file') and os.path.exists(scandict['localhdf5file']):
        return {**scan_parameters_from_hdf5(scandict['localhdf5file'], scan_number, scandict.get('CCDLabel')),
                'name': name}
    command = str(scandict['fullcommand']).split()   # ascan zf start stop nbsteps time
    return {'image_folder': scandict['imagefolder'], 'prefix': scandict['prefix'],
            'suffix': '.' + scandict['suffix'].lstrip('.'), 'nb_images': int(scandict['mapdimensions'][0]),
            'CCDLabel': scandict['CCDLabel'], 'hdf5_file': scandict['localhdf5file'],
            'scan_number': scan_number, 'wire_motor': command[1],
            'wire_start': float(command[2]), 'wire_stop': float(command[3]), 'name': name}


def new_config(newfile: Union[str, Path], template: Optional[Union[str, Path]] = None,
               scanfile: Optional[Union[str, Path]] = None, scandict: Optional[dict] = None,
               hdf5_scan: Optional[tuple] = None, overwrite: bool = False, **changes) -> dict:
    """write (or update) a daxm configuration file and return the configuration

    - file does not exist: written from the template (default config_template.yaml of LaueTools, or the
      configuration file of another scan) with the parameters of the scan and the changes
    - file exists: kept (with your manual edits), only the given parameters are updated

    :param scanfile: .scan file (DAXM GUI or wires calibration notebook) giving the scan and detector parameters
    :param scandict: or scan selected in the BLISS logfile (logfile_reader.H5file.build_dict_scan())
    :param hdf5_scan: or (BLISS hdf5 file of the dataset, scan number), see scan_parameters_from_hdf5()
    :param changes: parameters by name, e.g. calibfile='....calib', detfile='....det', ybeam_range=[-50, 100]
    """
    newfile = Path(newfile)
    fromscan = {}
    if scanfile is not None:
        fromscan = scan_parameters_from_scanfile(scanfile)
    elif scandict is not None:
        fromscan = scan_parameters_from_logfile(scandict)
    elif hdf5_scan is not None:
        fromscan = scan_parameters_from_hdf5(*hdf5_scan)
    if newfile.exists() and not overwrite:
        text = newfile.read_text()
        fromscan = {key: value for key, value in fromscan.items() if key != 'name'}
    else:
        text = Path(template or TEMPLATE_CONFIG).read_text()
    allchanges = {key: value for key, value in {**fromscan, **changes}.items()
                  if not (value is None and key in fromscan and key not in changes)}
    newfile.parent.mkdir(parents=True, exist_ok=True)
    newfile.write_text(_update_text(text, allchanges))
    cfg = load_config(newfile)
    print(f'daxm configuration: {newfile}')
    return cfg


def check_config(cfg: dict, workflow: str = 'quickdaxm', verbose: bool = True) -> bool:
    """check files and parameters needed by a workflow"""
    errors, warnings = [], []
    scan = cfg['scan']
    if not scan['image_folder'] or not os.path.isdir(scan['image_folder']):
        errors.append(f"image_folder not found: {scan['image_folder']}")
    elif scan['nb_images']:
        first = QD.image_file(scan, 0)[0]
        if not os.path.exists(first):
            errors.append(f'first image not found: {first}')
    if not scan['nb_images']:
        errors.append('nb_images not set')
    detfile = cfg['detector']['detfile']
    needs_detfile = workflow != 'quickdaxm' or cfg['wires']['calibfile']   # dips only: no geometry
    if needs_detfile and (not detfile or not os.path.exists(detfile)):
        errors.append(f'detfile not found: {detfile}')
    hdf5 = scan['hdf5_file'] and os.path.exists(scan['hdf5_file']) and scan['scan_number'] is not None
    if not hdf5 and (scan['wire_start'] is None or scan['wire_stop'] is None):
        errors.append('wire positions: set hdf5_file and scan_number, or wire_start and wire_stop')
    elif not hdf5:
        warnings.append('hdf5_file not found: wire positions evenly spaced from wire_start to wire_stop')
    if workflow == 'quickdaxm':
        calibfile = cfg['wires']['calibfile']
        if not calibfile:
            warnings.append('no calibfile (wires calibration): dips only, no position along the beam')
        elif not os.path.exists(calibfile):
            errors.append(f'calibfile not found: {calibfile}')
        peaklist = cfg['quickdaxm']['peaklist']
        if str(peaklist).lower() != 'auto' and not os.path.exists(peaklist):
            errors.append(f'peaklist not found: {peaklist}')
    if verbose:
        for message in warnings:
            GT.printyellow('warning: ' + message)
        for message in errors:
            GT.printred('error: ' + message)
    return not errors


# --------------------------------------------------------------------------------------
#  QUICK DAXM
# --------------------------------------------------------------------------------------
def output_files(cfg: dict) -> Dict[str, str]:
    folder = cfg['output']['folder']
    return {'quickdaxm': os.path.join(folder, 'quickdaxm.h5'),
            'quickdaxm_cor': os.path.join(folder, 'quickdaxm_spots.cor'),
            'quickdaxm_slices': os.path.join(folder, 'depthslices')}


def _quickdaxm_params(cfg: dict) -> Dict[str, Any]:
    """parameters of quickdaxm.run() in the section 'quickdaxm' of cfg (former names accepted, e.g. coresize)"""
    params = {QD.PARAM_ALIASES.get(key, key): value for key, value in (cfg.get('quickdaxm') or {}).items()}
    return {key: value for key, value in params.items() if key in QD.DEFAULT_PARAMS}


def compute_quickdaxm(cfg: dict, nb_cpus: Optional[int] = None, progress_callback=None) -> Dict[str, Any]:
    """quick daxm analysis of the scan of the configuration, in this process (see quickdaxm.run())"""
    params = _quickdaxm_params(cfg)
    results = QD.run(cfg['scan'], cfg['detector']['detfile'], cfg['wires']['calibfile'], params,
                     yref=cfg['wires']['yref'], sample_tilt=cfg['sample']['tilt'], nb_cpus=nb_cpus,
                     progress_callback=progress_callback)
    return results


def run_quickdaxm(cfg: dict, resource: str = 'kernel', nb_cpus: Optional[int] = None, redo: bool = False,
                  wait: bool = True, time_limit: str = '00:30:00', mem_per_cpu: str = '2G'):
    """quick daxm analysis of the scan of the configuration. Results saved in <output folder>/quickdaxm.h5

    Already done: results are reloaded, unless redo=True.
    :param resource: 'kernel' (in this process), 'local' (background process on this machine) or a SLURM
        resource of workflowlauncher.COMPUTING_RESOURCES ('magnifix', 'nice-hpc7' ...)
    :param wait: [not 'kernel'] wait for the end of the job and return the results; False: return the job
        (job.wait(), then DW.load_quickdaxm(cfg))
    :return: results (see quickdaxm.run()), or the job if wait=False
    """
    output = output_files(cfg)['quickdaxm']
    if os.path.exists(output) and not redo:
        results = QD.load_results(output)
        params = _quickdaxm_params(cfg)
        changed = [name for name, old_value, new_value in
                   (('calibfile', results.get('calibfile'), cfg['wires']['calibfile']),
                    ('detfile', results.get('detfile'), cfg['detector']['detfile']),
                    ('yref', results.get('yref'), cfg['wires']['yref']),
                    ('sample tilt', results.get('sample_tilt'), cfg['sample']['tilt']))
                   if old_value != new_value]
        # parameters of the dips analysis (not of the spots and profiles, which need redo=True)
        changed += [key for key in params if key not in ('peaklist', 'spots_threshold', 'spots_max_number',
                                                         'boxsize', 'boxsize_probe') and
                    QD.full_params(results['params']).get(key) != params[key]]
        if changed:
            GT.printyellow(f'quick daxm results reloaded from {output}: {changed} changed, dips and depths analysed '
                           'again (collected profiles kept)')
            results.update(detfile=cfg['detector']['detfile'], calibfile=cfg['wires']['calibfile'],
                           yref=cfg['wires']['yref'], sample_tilt=cfg['sample']['tilt'],
                           params={**results['params'], **params})
            results['dips'], results['spots'] = QD.analyse(results['signal'], results['xy'],
                                                           QD.geometry_of_results(results), results['params'])
            QD.save_results(results, output)
        else:
            GT.printgreen(f'quick daxm already done: results reloaded from {output}\n(redo=True to compute again)')
        return results
    if not check_config(cfg, 'quickdaxm'):
        raise RuntimeError(f"fix the errors of the configuration file {cfg.get('configfile')}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if resource == 'kernel':
        results = compute_quickdaxm(cfg, nb_cpus)
        QD.save_results(results, output)
        GT.printgreen(f'quick daxm results saved in {output}')
        return results
    import LaueTools.scripts.workflowlauncher as WL
    definition = WL.COMPUTING_RESOURCES.get(resource, {})
    if definition.get('nodes', 1) > 1:
        # one wire scan: one node (the images of a scan are not shared between nodes; several nodes are for the
        # wire scans of a grid, notebook 3_quick_daxm_grid)
        single = next((name for name, res in WL.COMPUTING_RESOURCES.items()
                       if res.get('partition') == definition.get('partition') and res.get('nodes', 1) == 1), None)
        GT.printyellow(f'{resource}: {definition["nodes"]} nodes for one wire scan: {single} (1 node) used instead')
        resource = single or 'local'
    job = WL.launch(WL.make_config(cfg, output, workflow='quickdaxm'), resource=resource, nbcpus=nb_cpus,
                    time_limit=time_limit, mem_per_cpu=mem_per_cpu)
    if not wait:
        return job
    status = job.wait()
    if status['state'] != 'COMPLETED':
        raise RuntimeError(f"quick daxm job {status['state']}: {status.get('message', '')}\n{job.log_tail()}")
    return QD.load_results(output)


def load_quickdaxm(cfg: dict) -> Dict[str, Any]:
    """results of the quick daxm analysis of the scan of the configuration"""
    return QD.load_results(output_files(cfg)['quickdaxm'])


def write_quickdaxm_files(cfg: dict, results: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """write the peak lists of the quick daxm analysis in the output folder:

    - quickdaxm_spots.cor: all spots with a position along the beam, with columns ybeam, depth,
      extent (microns), wire, nb_dips (2theta and chi computed for the source of each spot)
    - depthslices/<name>_0000.cor ...: one .cor file per slice along the beam (slice_step,
      slice_thickness, slice_min_spots), and depthslices/slices.yaml (ybeam, depth, nb of spots of the slices)

    :return: dict 'cor' (path), 'slices' (list of dict, see quickdaxm.write_depth_slices())
    """
    files = output_files(cfg)
    params = cfg['quickdaxm']
    results = dict(results, detector_offset=cfg['sample'].get('detector_offset', 'auto'))
    corfile = QD.write_cor(results, Path(files['quickdaxm_cor']).stem, os.path.dirname(files['quickdaxm_cor']))
    prefix = re.sub(r'[^\w.-]+', '_', str(cfg['name'])).strip('_') or 'daxm'
    slices = QD.write_depth_slices(results, prefix, files['quickdaxm_slices'], step=params['slice_step'],
                                   thickness=params['slice_thickness'], min_spots=params['slice_min_spots'])
    with open(os.path.join(files['quickdaxm_slices'], 'slices.yaml'), 'w') as f:
        yaml.safe_dump({'prefix': prefix, 'slice_step': params['slice_step'],
                        'detector_offset_um': QD.detector_offset_um(results), 'slices': slices}, f, sort_keys=False)
    if verbose:
        written = [s for s in slices if s['file']]
        print(f'{corfile}: {len(QD.select_spots(results))} spots with columns ybeam, depth')
        print(f"{files['quickdaxm_slices']}: {len(written)} .cor files of slices"
              + (f" (ybeam {written[0]['ybeam']:g} to {written[-1]['ybeam']:g} µm)" if written else ''))
    return {'cor': corfile, 'slices': slices}


# --------------------------------------------------------------------------------------
#  WIRES CALIBRATION: approximate parameters from the dips of a reference crystal, refined with the fluorescence
# --------------------------------------------------------------------------------------
# stages of the fluorescence refinement (as Daxm/scripts/calib.py): (grid of points per wire, data_span, variables)
FLUO_STAGES = (([2, 2], None, ['h', 'p0', 'axis', 'dm']),
               ([3, 3], None, ['h', 'p0', 'axis', 'R', 'u2', 'dm']),
               ([6, 6], 3.5, ['R', 'h', 'p0', 'axis', 'u2', 'dm']))
# monitor offset (counts per second) of the legacy scan class, by detector family
MONITOR_OFFSETS = {'EIGER': 28.3, 'sCMOS': 25.}


def calib_files(cfg: dict) -> Dict[str, str]:
    folder = os.path.join(cfg['output']['folder'], 'wirescalib')
    return {'folder': folder, 'dips': os.path.join(folder, 'wires_from_dips.calib'),
            'scan': os.path.join(folder, 'wirescalib.scan')}


def approximate_wires(cfg: dict, results: Optional[Dict[str, Any]] = None, verbose: bool = True) -> Dict[str, Any]:
    """first approximation of the wires trajectory parameters from the dips of the Laue spots of a reference
    crystal (Ge) found by the quick daxm analysis of the scan (run without calibfile), written in
    <output folder>/wirescalib/wires_from_dips.calib (see quickdaxm.wires_from_dips())"""
    if results is None:
        results = load_quickdaxm(cfg)
    params = cfg['wirescalib']
    fit = QD.wires_from_dips(results, nb_wires=params['nb_wires'], var=params['var'], ybeam_um=params['ybeam'],
                             verbose=verbose)
    files = calib_files(cfg)
    os.makedirs(files['folder'], exist_ok=True)
    fit['calibfile'] = QD.save_calib(fit, files['dips'])
    if verbose:
        print(f"approximate wires parameters: {fit['calibfile']}")
    return fit


def crude_wires(cfg: dict, results: Dict[str, Any], method: str = 'auto', nb_wires: Optional[int] = None,
                radius_each_wire: bool = False, outliers: float = 0.02,
                excluded_wires: Optional[Sequence[int]] = None, verbose: bool = True) -> Dict[str, Any]:
    """crude wires calibration (seconds) from the dips of the Laue spots of a wire scan (quick daxm without wires
    calibration), written in <results folder>/wirescalib:

    - method 'dips' (reference crystal at the surface, Ge): least squares fit of the wires to the dips edges, sources
      at ybeam = 0 (quickdaxm.wires_from_dips()) -> wires_from_dips.calib
    - method 'envelope' (any sample): the shallowest grains seen by each wire define ybeam = 0
      (quickdaxm.wires_from_envelope()) -> wires_from_envelope.calib
    - 'auto': 'dips' for a reference scan (Ge in the dataset name), else 'envelope'

    :return: dict of the fit (see the quickdaxm functions) with 'calibfile' and 'method'
    """
    if method == 'auto':
        method = 'dips' if is_reference_scan(cfg['name']) else 'envelope'
    files = calib_files(cfg)
    os.makedirs(files['folder'], exist_ok=True)
    if method == 'dips':
        cfg['wirescalib']['var'] = ['h', 'p0', 'f1', 'Rw' if radius_each_wire else 'R']
        cfg['wirescalib']['nb_wires'] = nb_wires
        fit = approximate_wires(cfg, results, verbose=verbose)
    elif method == 'envelope':
        fit = QD.wires_from_envelope(results, nb_wires=nb_wires, outliers=outliers, verbose=verbose)
        fit['calibfile'] = QD.save_calib(fit, 'wires_from_envelope.calib', directory=files['folder'])
    else:
        raise ValueError(f"method {method!r}: 'dips', 'envelope' or 'auto'")
    QD.set_excluded_wires(fit['calibfile'], excluded_wires)
    fit['method'] = method
    if verbose:
        print(f"crude wires calibration ({method}): {fit['calibfile']}")
    return fit


def compare_calibrations(results: Dict[str, Any], calibfiles: Dict[str, str], plot: bool = True):
    """wires parameters of several calibrations (dict name: .calib file) and the positions along the beam of the Laue
    spots of the scan they give (same dips: quick daxm results): table (pandas DataFrame) one line per calibration and
    wire, and a histogram of ybeam per calibration. A shift of the median ybeam between two calibrations is the
    difference of their depth zeros (e.g. surface of the shallowest grains vs top of the fluorescent layer)"""
    import pandas as pd
    rows, histos = [], {}
    for name, calibfile in calibfiles.items():
        if calibfile is None or not os.path.exists(str(calibfile)):
            continue
        with open(calibfile, 'r') as f:
            content = json.load(f)
        geometry = QD.WireGeometry.from_files(str(calibfile), results['detfile'], results['wire_positions'],
                                              results.get('yref', 0.), sample_tilt=results.get('sample_tilt', 40.))
        dips, spots = QD.analyse(results['signal'], results['xy'], geometry, results['params'])
        histos[name] = spots['ybeam'][np.isfinite(spots['ybeam'])]
        two = spots['nb_dips'] > 1
        for iw, w in enumerate(content['wires']):
            sel = (dips['wire'] == iw) & np.isfinite(dips['ybeam'])
            rows.append({'calibration': name, 'wire': iw, 'excluded': iw in geometry.excluded_wires,
                         'h': w['h'], 'p0': w['p0'], 'R (µm)': w['R'] * 1000., 'f1': w['f1'], 'f2': w['f2'],
                         'u2': content['traj']['u2'], 'dips with depth': int(sel.sum()),
                         'median ybeam (µm)': float(np.median(dips['ybeam'][sel])) if sel.any() else np.nan,
                         'median mismatch (µm)': float(np.nanmedian(dips['mismatch'][sel])) if sel.any() else np.nan,
                         'spots with depth': int(np.isfinite(spots['ybeam']).sum()),
                         'median 2-wire spread (µm)': float(np.median(spots['spread'][two])) if two.any() else np.nan})
    table = pd.DataFrame(rows)
    if plot and histos:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.5))
        values = np.concatenate(list(histos.values()))
        bins = np.arange(np.floor(np.percentile(values, 1)) - 10, np.ceil(np.percentile(values, 99)) + 10, 2.)
        for name, values in histos.items():
            ax.hist(values, bins=bins, histtype='step', label=f'{name} (median {np.median(values):.1f} µm)')
        ax.set_xlabel('ybeam of the Laue spots (µm)')
        ax.set_ylabel('nb of spots')
        ax.legend(fontsize=8)
    return table


def legacy_scan_dict(cfg: dict, calibfile: str) -> Dict[str, Any]:
    """scan of the configuration as a dict of the legacy DAXM scan class (PointScan, .scan file), with the
    wires of calibfile"""
    import h5py
    from LaueTools.Daxm.classes.scan.point import new_scan_dict
    scan = cfg['scan']
    if not scan['hdf5_file'] or scan['scan_number'] is None:
        raise ValueError('the legacy scan class needs hdf5_file and scan_number')
    with h5py.File(scan['hdf5_file'], 'r', locking=False) as f:
        node = scan['hdf5_node'] if scan['hdf5_node'] and scan['hdf5_node'] in f else f"{int(scan['scan_number'])}.1"
        command = f[node]['title'][()]
    command = command.decode() if isinstance(command, bytes) else str(command)
    with open(calibfile, 'r') as f:
        calib = json.load(f)
    family = 'EIGER' if 'EIGER' in scan['CCDLabel'] else 'sCMOS'
    offset = cfg['wirescalib']['monitor_offset']
    return new_scan_dict({'CCDType': scan['CCDLabel'], 'specFile': scan['hdf5_file'],
                          'localhdf5file': scan['hdf5_file'], 'scanNumber': int(scan['scan_number']),
                          'hdf5scanId': str(cfg['name']), 'imageFolder': scan['image_folder'],
                          'imagePrefix': scan['prefix'], 'imageFirstIndex': int(scan['first_image_index']),
                          'imageDigits': int(scan['nbdigits']), 'detCalib': cfg['detector']['detfile'],
                          'wire': [[w['material'], w['R'], w['h'], w['p0']] for w in calib['wires']],
                          'wireTrajAngle': calib['traj']['u2'], 'scantype': command.split()[0],
                          'fullcommand': command, 'imageOffset': 0. if family == 'EIGER' else 1000.,
                          'monitorOffset': MONITOR_OFFSETS[family] if offset is None else offset})


def read_fluo_layer(calibfile: Optional[str]) -> Optional[dict]:
    """layer above the fluorescent material fitted by refine_wires_fluo() with cfg['wirescalib']['fluo_layer'] (key
    'fluo_layer' of the .calib file: thickness (mm along the beam), fraction (median fraction of the signal)), or None"""
    if not calibfile or not os.path.isfile(calibfile):
        return None
    with open(calibfile, 'r') as f:
        return json.load(f).get('fluo_layer')


def _set_fluo_layer(calibfile: str, thickness: float, fractions) -> None:
    with open(calibfile, 'r') as f:
        content = json.load(f)
    fractions = np.asarray(fractions, dtype=float)
    content['fluo_layer'] = {'thickness': float(thickness), 'fraction': float(np.median(fractions)),
                             'fraction_min': float(np.min(fractions)), 'fraction_max': float(np.max(fractions))}
    with open(calibfile, 'w') as f:
        f.write(json.dumps(content, indent=1))


def _legacy_calib(cfg: dict, calibfile: str, verbose: bool = False):
    """scan (PointScan) and fluorescence calibration (CalibFluo) of the legacy DAXM classes, with the wires of
    calibfile. With cfg['wirescalib']['fluo_layer'] (start thickness along the beam, mm), CalibFluoLayer: the model
    has a layer above the fluorescent material (e.g. Cr coating on Zr), see LaueTools.Daxm.fluolayer"""
    from LaueTools.Daxm.classes.scan.point import PointScan
    from LaueTools.Daxm.classes.source import new_source
    from LaueTools.Daxm.classes.calibration import new_calib
    scan = PointScan(legacy_scan_dict(cfg, calibfile), verbose=False)
    sample = new_source(cfg['wirescalib']['fluo_element'], cfg['wirescalib']['fluo_thickness'], ystep=0.001)
    layer = cfg['wirescalib'].get('fluo_layer')
    if layer:   # layer above the fluorescent material (e.g. Cr coating on Zr): CalibFluoLayer
        from LaueTools.Daxm.fluolayer import CalibFluoLayer
        previous = read_fluo_layer(calibfile)   # thickness fitted by a former stage, if any
        thickness = previous['thickness'] if previous else float(layer)
        calib = CalibFluoLayer(scan, sample, layer_thickness=thickness,
                               layer_max=cfg['wirescalib'].get('fluo_layer_max') or 0.08,
                               fit_thickness=not cfg['wirescalib'].get('fluo_layer_fixed'), verbose=verbose)
    else:
        calib = new_calib(scan, sample, kind='fluo', verbose=verbose)
    # first images not used (wire position not reliable, quickdaxm parameter skip_images)
    calib.skip_images = int((cfg.get('quickdaxm') or {}).get('skip_images') or 0)
    calib.load_wires(os.path.basename(calibfile), directory=os.path.dirname(os.path.abspath(calibfile)))
    calib.log_restart()
    return scan, calib


# criteria of a profile usable for the fluorescence calibration (check_fluo_profiles())
FLUO_PROFILE_CRITERIA = {'min_contrast': 0.05,   # (plateau - floor) / plateau: depth of the fluorescence dip
                         'min_snr': 10.,         # (plateau - floor) / noise
                         'max_rms': 0.05,        # residual rms of the fluorescence dip template / dip contrast
                         'max_spikes': 0,        # points far (6 noise) from the template (hot pixel, spot)
                         'max_offset': 25.,      # microns: |sharp edge of the dip - edge predicted by the wires|
                         'width_tolerance': 0.35,  # |shadow width of the template / predicted width - 1|
                         'spot_distance': 6,     # pixels: no Laue spot closer than this to the box of the profile
                         'probe_gain': 0.8}      # probe_all: a usable point is moved only if the template rms is
                                                 # multiplied by less than this (keeps the points spread)


def _exp_source_cdf(u, length, sigma):
    """fraction of a source decreasing exponentially (decay length) from 0 lying at positions < u, blurred by a
    gaussian (sigma); u, length, sigma in the same units"""
    from scipy.special import ndtr, log_ndtr
    z = u / sigma
    arg = -u / length + 0.5 * (sigma / length) ** 2 + log_ndtr(z - sigma / length)
    return ndtr(z) - np.exp(np.minimum(arg, 50.))


def fluo_dip_template(pw, intensity, entry: float, exit_: float) -> Dict[str, Any]:
    """fit of an intensity profile by the template of a fluorescence dip: an opaque wire of shadow width w passing
    in front of a source starting at the sample surface and decreasing exponentially along the beam (length L),
    blurred by a gaussian (sigma):  I(p) = b - a * [C(p - pe) - C(p - pe - w)]   (C: cumulative of the source)
    The edge of the dip on the side of the surface is sharp, the other one has a tail (deeper sources).
    Both orientations of the tail are tried.

    :param pw: wire positions (mm)
    :param entry, exit_: wire positions (mm) where the wire starts and stops to mask the ray from the surface
        (ybeam = 0) to the pixel, predicted by the wires calibration
    :return: dict: curve (template at pw), rms (residual rms / contrast), contrast (a / b), edge (mm, position of
        the sharp edge), offset (microns, edge - predicted one), width (microns), width_pred, length, sigma
        (microns), tail (+1: towards larger wire positions), residual
    """
    from scipy import optimize
    pw = np.ravel(pw).astype(float)
    intensity = np.ravel(intensity).astype(float)
    origin = float(entry)
    p = (pw - origin) * 1000.
    entry_um, exit_um = 0., (float(exit_) - origin) * 1000.
    width0 = abs(exit_um - entry_um)

    def shape(params, tail):
        pe, w, length, sigma = params
        if tail > 0:
            return _exp_source_cdf(p - pe, length, sigma) - _exp_source_cdf(p - pe - w, length, sigma)
        return _exp_source_cdf(-p + pe + w, length, sigma) - _exp_source_cdf(-p + pe, length, sigma)

    def solve(params, tail):
        design = np.vstack([shape(params, tail), np.ones_like(p)]).T
        coef = np.linalg.lstsq(design, intensity, rcond=None)[0]
        return design @ coef, coef

    best = None
    for tail in (1, -1):
        for shift in np.arange(-60., 60.1, 3.):
            pe = entry_um + shift if tail > 0 else exit_um + shift - width0
            curve, _ = solve((pe, width0, 10., 1.), tail)
            cost = np.sum((curve - intensity) ** 2)
            if best is None or cost < best[0]:
                best = (cost, tail, pe)
    _, tail, pe0 = best
    lower = [p.min() - width0, 0.5 * width0, 0.2, 0.05]
    upper = [p.max(), 2. * width0, 500., 10.]
    start = np.clip([pe0, width0, 10., 1.], lower, upper)
    res = optimize.least_squares(lambda x: solve(x, tail)[0] - intensity, start, bounds=(lower, upper))
    curve, (amp, base) = solve(res.x, tail)
    pe, w, length, sigma = (float(v) for v in res.x)
    contrast = -amp
    edge_um = pe if tail > 0 else pe + w
    edge_pred = entry_um if tail > 0 else exit_um
    residual = intensity - curve
    return {'curve': curve, 'residual': residual,
            'rms': float(np.sqrt(np.mean(residual ** 2)) / contrast) if contrast > 0 else np.inf,
            'contrast': float(contrast / base) if base > 0 else np.nan,
            'edge': origin + edge_um * 1e-3, 'offset': edge_um - edge_pred, 'width': w, 'width_pred': width0,
            'length': length, 'sigma': sigma, 'tail': int(tail)}


def _predicted_shadow(scan, wid: int, x: float, y: float) -> Tuple[float, float]:
    """wire positions (mm) where wire wid starts and stops to mask the ray from the surface to pixel x, y"""
    positions = np.ravel(scan.calc_wire_intersect_ray(wid, x, y, ysrc=0.)).astype(float)
    return float(min(positions[1], positions[2])), float(max(positions[1], positions[2]))


def _spot_in_box(spots: Optional[np.ndarray], x: float, y: float, halfboxsize, distance: float) -> bool:
    """True if a Laue spot (X, Y of spots) is closer than distance pixels to the box of a profile"""
    if spots is None or not len(spots):
        return False
    hx, hy = (list(halfboxsize) + [0, 0])[:2]
    return bool(np.any((np.abs(spots[:, 0] - x) <= hx + distance) & (np.abs(spots[:, 1] - y) <= hy + distance)))


def fluo_spots(cfg: dict, threshold: float = 100.) -> np.ndarray:
    """X, Y of the Laue spots of the wire scan of cfg, to keep them out of the boxes of the fluorescence profiles:
    spots of the quick daxm results if computed, else found in the max projection of the scan (lower threshold
    than quick daxm: weak spots also perturb the fluorescence)"""
    path = output_files(cfg)['quickdaxm']
    if os.path.isfile(path):
        import h5py
        with h5py.File(path, 'r') as f:
            if 'xy' in f:
                return np.asarray(f['xy'][()], dtype=float)[:, :2]
    return QD.find_spots(QD.max_projection(cfg['scan']), threshold=threshold, max_number=5000)[:, :2]


def _trim_profile(scan, skip: int, pw, intensity):
    """profile without the points of the first skip images of the scan"""
    pw, intensity = np.asarray(pw), np.asarray(intensity)
    if not skip:
        return pw, intensity
    positions = np.asarray(scan.wire_position, dtype=float)
    limit = positions[min(int(skip), len(positions) - 1)]
    keep = np.ravel(pw >= limit - 1e-9) if positions[-1] > positions[0] else np.ravel(pw <= limit + 1e-9)
    return pw[keep], intensity[keep]


def trim_calib_profiles(scan, calib):
    """remove from the profiles of calib the points of its first calib.skip_images images"""
    skip = getattr(calib, 'skip_images', 0)
    if not skip:
        return
    for wid in range(len(calib.data_XYcam)):
        for k in range(len(calib.data_XYcam[wid])):
            calib.data_pw[wid][k], calib.data_I[wid][k] = _trim_profile(scan, skip, calib.data_pw[wid][k],
                                                                         calib.data_I[wid][k])


def _fluo_profile_quality(scan, wid: int, x: float, y: float, pw, intensity, crit: dict,
                          spots: Optional[np.ndarray] = None, halfboxsize=(2, 0)) -> Tuple[dict, Any, Any, float]:
    """quality of one intensity profile of the fluorescence calibration (see check_fluo_profiles()): statistics of
    the profile and fit by the template of a fluorescence dip (fluo_dip_template())

    :return: row (dict: wire, X, Y, nb, contrast, snr, complete, spikes, rms, offset, width, width_pred, length,
        tail, spot_in_box, ok), smoothed profile and template (None if too few images), predicted dip center (mm)
    """
    from scipy import ndimage
    pw = np.ravel(pw)
    intensity = np.ravel(intensity).astype(float)
    entry, exit_ = _predicted_shadow(scan, wid, x, y)
    row = {'wire': int(wid), 'X': int(x), 'Y': int(y), 'nb': len(intensity), 'contrast': np.nan, 'snr': np.nan,
           'complete': False, 'spikes': -1, 'rms': np.nan, 'offset': np.nan, 'width': np.nan,
           'width_pred': abs(exit_ - entry) * 1000., 'length': np.nan, 'tail': 0,
           'spot_in_box': _spot_in_box(spots, x, y, halfboxsize, crit['spot_distance']), 'ok': False}
    predicted = 0.5 * (entry + exit_)
    if len(intensity) < 10:
        return row, None, None, predicted
    smooth = ndimage.median_filter(intensity, size=5, mode='nearest')
    plateau, floor = np.percentile(smooth, 90), smooth.min()
    diff = np.diff(intensity)
    noise = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.) or 1e-9
    edge = max(2, len(intensity) // 10)
    level = floor + 0.7 * (plateau - floor)
    template = fluo_dip_template(pw, intensity, entry, exit_)
    row.update(contrast=(plateau - floor) / plateau if plateau > 0 else np.nan,
               snr=(plateau - floor) / noise,
               complete=bool(np.median(smooth[:edge]) > level and np.median(smooth[-edge:]) > level),
               spikes=int(np.sum(np.abs(template['residual']) > 6. * noise)),
               rms=template['rms'], offset=template['offset'], width=template['width'],
               length=template['length'], tail=template['tail'])
    row['ok'] = not _fluo_profile_reasons(row, crit)
    return row, smooth, template['curve'], predicted


def _fluo_profile_reasons(row: dict, crit: dict) -> List[str]:
    """why a profile is not usable (empty list: usable)"""
    if row['nb'] < 10:
        return ['too few images']
    width_ratio = row['width'] / row['width_pred'] if row['width_pred'] > 0 else np.nan
    return [name for name, failed in
            (('Laue spot in the box', row['spot_in_box']),
             ('weak dip', not row['contrast'] >= crit['min_contrast']),
             ('noisy', not row['snr'] >= crit['min_snr']),
             ('incomplete dip', not row['complete']),
             ('not a fluorescence dip (template rms)', not row['rms'] <= crit['max_rms']),
             ('spikes', row['spikes'] > crit['max_spikes']),
             ('far from prediction', not abs(row['offset']) <= crit['max_offset']),
             ('shadow width', not abs(width_ratio - 1.) <= crit['width_tolerance']))
            if failed]


def _fluo_candidates(scan, wid: int, x: float, y: float, cell_x: float, nb_x: int = 9,
                     y_fractions: Sequence[float] = (0., 0.15, 0.3, 0.45)) -> List[List[int]]:
    """candidate points of the profile (x, y) of wire wid, the point itself first: probed along X within its cell of
    the grid (width cell_x: the points stay spread over the detector), at the same Y (shadow lines are about
    parallel to X) and moved towards the middle of the Y range where the wire shadows the whole scan (dip cut by
    the scan limits), out of the EIGER band gaps"""
    from LaueTools.Daxm.classes.calibration.fluorescence import adjust_not_in_bands
    framedim = scan.get_img_params(['framedim'])
    y0, y1 = sorted(float(v) for v in scan.calc_wire_range_scan_int(wid))
    ymid = 0.5 * (y0 + y1)
    xs = sorted(np.linspace(x - 0.5 * cell_x, x + 0.5 * cell_x, nb_x), key=lambda v: abs(v - x))
    points = [[int(round(x)), int(round(y))]]
    for fy in y_fractions:
        for xc in xs:
            points.append([int(round(xc)), int(round(y + fy * (ymid - y)))])
    if str(scan.ccd_type).upper().startswith('EIGER'):
        points = [list(map(int, p)) for p in adjust_not_in_bands(points)]
    res, seen = [], set()
    for p in points:
        if tuple(p) not in seen and 5 <= p[0] < framedim[0] - 5 and 5 <= p[1] < framedim[1] - 5:
            seen.add(tuple(p))
            res.append(p)
    return res


def replace_fluo_points(scan, calib, criteria: Optional[dict] = None, spots: Optional[np.ndarray] = None,
                        probe_all: bool = False, verbose: bool = True) -> List[dict]:
    """choose in calib (CalibFluo with its points and profiles loaded) the points of the fluorescence calibration:
    each unusable point (criteria of check_fluo_profiles(), e.g. a Laue spot in the box, a dip cut by the scan
    limits, a profile not looking like a fluorescence dip) is replaced by the best usable point probed nearby,
    first along X at the same Y (as the background pixels of the former PixelMonitoring notebooks), then towards
    the middle of the shadow region of the wire. With probe_all, every point is probed this way and the best
    profile (smallest template rms) is taken. A point without usable candidate is kept.

    :param spots: X, Y of the Laue spots (fluo_spots()): boxes near a spot are not used
    :return: list of dict per point changed or kept unusable: wire, point, X, Y, new_X, new_Y (None if kept),
        reasons (of the old point), rms (old, new)
    """
    crit = {**FLUO_PROFILE_CRITERIA, **(criteria or {})}
    hbs = calib.data_hbs
    framedim = scan.get_img_params(['framedim'])
    cell_x = framedim[0] / float((calib.grid_dim or [3, 3])[0])
    todo = []
    for wid, points in enumerate(calib.data_XYcam):
        for k, (x, y) in enumerate(points):
            row, _, _, _ = _fluo_profile_quality(scan, wid, x, y, calib.data_pw[wid][k], calib.data_I[wid][k], crit,
                                                 spots, hbs)
            if probe_all or not row['ok']:
                todo.append((wid, k, int(x), int(y), row))
    if not todo:
        return []
    taken = {tuple(int(v) for v in xy) for points in calib.data_XYcam for xy in points}
    candidates, owners = [], []
    for itodo, (wid, k, x, y, _) in enumerate(todo):
        for p in _fluo_candidates(scan, wid, x, y, cell_x)[1:]:
            if tuple(p) not in taken and not _spot_in_box(spots, p[0], p[1], hbs, crit['spot_distance']):
                candidates.append(p)
                owners.append(itodo)
    report = []
    best = {}
    if candidates:
        wires = [todo[i][0] for i in owners]
        data_I, data_pw = scan.get_profile_manypixels_centred(wires, candidates, hbs, calib.data_span)
        data_I, data_pw = list(data_I), list(data_pw)
        for j in range(len(candidates)):
            data_pw[j], data_I[j] = _trim_profile(scan, getattr(calib, 'skip_images', 0), data_pw[j], data_I[j])
        for j, itodo in enumerate(owners):
            row, _, _, _ = _fluo_profile_quality(scan, wires[j], *candidates[j], data_pw[j], data_I[j], crit,
                                                 spots, hbs)
            if row['ok'] and (itodo not in best or row['rms'] < best[itodo][0]):
                best[itodo] = (row['rms'], j)
    for itodo, (wid, k, x, y, old) in enumerate(todo):
        entry = {'wire': wid, 'point': k, 'X': x, 'Y': y, 'new_X': None, 'new_Y': None,
                 'reasons': _fluo_profile_reasons(old, crit), 'rms': (old['rms'], np.nan)}
        if itodo in best and (not old['ok'] or best[itodo][0] < crit['probe_gain'] * old['rms']):
            j = best[itodo][1]
            if tuple(candidates[j]) in taken:   # chosen meanwhile for another point
                continue
            taken.add(tuple(candidates[j]))
            xy = np.array(calib.data_XYcam[wid])
            xy[k] = candidates[j]
            calib.data_XYcam[wid] = xy
            calib.data_pw[wid][k] = data_pw[j]
            calib.data_I[wid][k] = data_I[j]
            entry.update(new_X=int(candidates[j][0]), new_Y=int(candidates[j][1]), rms=(old['rms'], best[itodo][0]))
        if entry['new_X'] is not None or not old['ok']:
            report.append(entry)
    if verbose:
        for r in report:
            why = ', '.join(r['reasons']) if r['reasons'] else f"template rms {r['rms'][0]:.3f}"
            text = f"wire {r['wire']} point {r['point']} (X {r['X']}, Y {r['Y']}: {why})"
            if r['new_X'] is None:
                GT.printyellow(text + ': no usable point nearby, kept')
            else:
                GT.printgreen(text + f" -> X {r['new_X']}, Y {r['new_Y']} (template rms {r['rms'][1]:.3f})")
    return report


def check_fluo_profiles(cfg: dict, calibfile: Optional[str] = None, dims=(3, 3), data_span: Optional[float] = None,
                        criteria: Optional[dict] = None, replace_bad_points: bool = False, probe_all: bool = False,
                        avoid_spots: bool = True, plot: bool = True, verbose: bool = True):
    """check the intensity profiles used by the fluorescence calibration of the wires (refine_wires_fluo()):
    same points on the detector (grid dims per wire, regions where each wire masks the fluorescence predicted by the
    wires of calibfile) and same window of wire positions (data_span), before a refinement

    A profile is usable ('ok') if: no Laue spot near its box (avoid_spots), the fluorescence dip is deep enough
    (contrast) and above the noise (snr: dip depth / noise of the differences of consecutive images), complete
    (plateau on both sides of the dip in the window), and it looks like a fluorescence dip: fitted by the template
    fluo_dip_template() (sharp edge on the side of the surface, exponential tail of the deeper sources) with a small
    residual (rms), no point far from it (spikes), its sharp edge close to the one predicted by the wires (offset)
    and its shadow width close to the predicted one (else the starting wires are wrong: the refinement may fail)

    :param calibfile: wires (default: wires_from_dips.calib of approximate_wires())
    :param criteria: changes of FLUO_PROFILE_CRITERIA
    :param replace_bad_points: replace the unusable points by usable points probed nearby (replace_fluo_points(), as
        refine_wires_fluo(replace_bad_points=True)); the table shows the points finally used
    :param probe_all: probe all the points (best template rms nearby), not only the unusable ones
    :param avoid_spots: boxes near the Laue spots of the scan (fluo_spots()) are not used
    :return: table (pandas DataFrame) one line per profile: wire, point, X, Y, contrast, snr, complete, spikes, rms,
        offset (sharp edge - predicted, microns of wire position), width, width_pred (microns), length (microns,
        decay of the source along the wire travel), tail, spot_in_box, ok, replaced
    """
    import pandas as pd
    calibfile = calibfile or calib_files(cfg)['dips']
    crit = {**FLUO_PROFILE_CRITERIA, **(criteria or {})}
    spots = fluo_spots(cfg) if avoid_spots else None
    scan, calib = _legacy_calib(cfg, calibfile)
    if data_span is not None:
        calib.data_span = data_span
    calib.set_points_grid(dims=list(dims), adjust=False)
    trim_calib_profiles(scan, calib)
    replaced = set()
    if replace_bad_points or probe_all:
        replaced = {(r['wire'], r['point']) for r in replace_fluo_points(scan, calib, crit, spots, probe_all,
                                                                        verbose=verbose)
                    if r['new_X'] is not None}
    rows, curves = [], []
    for wid, points in enumerate(calib.data_XYcam):
        for k, (x, y) in enumerate(points):
            row, smooth, template, predicted = _fluo_profile_quality(scan, wid, x, y, calib.data_pw[wid][k],
                                                                     calib.data_I[wid][k], crit, spots, calib.data_hbs)
            row = {'wire': wid, 'point': k, **row, 'replaced': (wid, k) in replaced}
            if smooth is not None:
                curves.append((wid, k, np.ravel(calib.data_pw[wid][k]), np.ravel(calib.data_I[wid][k]), template,
                               predicted, row['ok']))
            rows.append(row)
    table = pd.DataFrame(rows)
    if verbose:
        for wid, group in table.groupby('wire'):
            bad = group[~group['ok']]
            text = f"wire {wid}: {int(group['ok'].sum())}/{len(group)} profiles usable"
            if len(bad):
                text += ' -- ' + '; '.join(f"point {int(r['point'])} (X {r['X']}, Y {r['Y']}): "
                                           f"{', '.join(_fluo_profile_reasons(r, crit))}" for _, r in bad.iterrows())
            (GT.printgreen if not len(bad) else GT.printyellow)(text)
    if plot and curves:
        import matplotlib.pyplot as plt
        nbwires = len(calib.data_XYcam)
        nbpoints = max(len(p) for p in calib.data_XYcam)
        fig, axes = plt.subplots(nbwires, nbpoints, figsize=(2.6 * nbpoints, 2.2 * nbwires), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for wid, k, pw, intensity, template, predicted, ok in curves:
            ax = axes[wid][k]
            ax.set_visible(True)
            ax.plot(pw, intensity, '.', ms=2, color='0.4')
            ax.plot(pw, template, '-', lw=1, color='tab:green' if ok else 'tab:red')
            ax.axvline(predicted, color='k', lw=0.8, ls='--')
            title = f'wire {wid} point {k}' + (' (replaced)' if (wid, k) in replaced else '')
            ax.set_title(title, fontsize=8, color='k' if ok else 'tab:red')
            ax.tick_params(labelsize=6)
        fig.suptitle(f'profiles of the fluorescence calibration and fluorescence dip template (green: usable, red: '
                     f'not; dashed: dip center predicted by {os.path.basename(calibfile)})', fontsize=9)
        fig.tight_layout()
    return table


def fluo_resolution(cfg: dict, calibfile: str, dims=(3, 3), data_span: Optional[float] = None,
                    replace_bad_points: bool = True, probe_all: bool = False, avoid_spots: bool = True,
                    max_sigma: float = 15., plot: bool = True, verbose: bool = True):
    """depth resolution measured with the fluorescence: each intensity profile of the fluorescence calibration
    (points as in check_fluo_profiles()) is fitted, with the wires of calibfile fixed, by the profile of the source
    model (CalibFluo: incident beam absorption, fluorescence, self absorption on the way out, wire transmission)
    blurred by a gaussian along the wire travel:
        I(p) = a * (S * G_sigma)(p - shift) + b
    sigma is the instrument blur (beam size, wire edge, motor, pixel), shift the local error of the calibration.
    Both are given in microns of wire travel and of ybeam (sources along the beam).

    The blur is mostly measured on the sharp side of the dip edges (surface of the sample): an error of the source
    model (incident spectrum, depth extension of the source) is partly absorbed in sigma, an upper bound.

    :return: table (pandas DataFrame) one line per profile: wire, point, X, Y, two_theta, chi, ok (usable profile),
        sigma, sigma_err, shift (microns of wire travel), sigma_ybeam, shift_ybeam (microns along the beam), rms
        (residual / dip contrast), fit_ok (rms not larger than twice the median rms)
    """
    import pandas as pd
    from scipy import ndimage, optimize
    crit = dict(FLUO_PROFILE_CRITERIA)
    spots = fluo_spots(cfg) if avoid_spots else None
    scan, calib = _legacy_calib(cfg, calibfile)
    if data_span is not None:
        calib.data_span = data_span
    calib.set_points_grid(dims=list(dims), adjust=False)
    trim_calib_profiles(scan, calib)
    if replace_bad_points or probe_all:
        replace_fluo_points(scan, calib, crit, spots, probe_all, verbose=verbose)
    calib.run_arg(None, ['p0'], None)
    calib.run_init()
    nb = calib.exp_qty
    # profile of the model on a fine grid of wire positions (0.1 micron), with kM = 1, km = 0
    x = np.array(calib.var_x0, dtype=float)
    with_layer = hasattr(calib, 'var_fL')
    measured = [np.ravel(p) for p in calib.exp_pw]
    dp = 1e-4
    margin = 3. * max_sigma * 1e-3 + 0.03
    calib.exp_pw = [np.arange(p.min() - margin, p.max() + margin, dp)[:, np.newaxis] for p in measured]
    if with_layer:   # profiles of the fluorescent material and of the layer (thickness of the calibration)
        layer = read_fluo_layer(calibfile)
        if layer:
            x[-3 * nb - 1] = layer['thickness']
        x[-3 * nb::3], x[-3 * nb + 1::3], x[-3 * nb + 2::3] = 1., 0., 0.
        models = calib.run_optim_sim(x)
        x[-3 * nb + 2::3] = 1.
        layer_models = [np.ravel(both) - np.ravel(one) for both, one in zip(calib.run_optim_sim(x), models)]
    else:
        x[-2 * nb::2], x[-2 * nb + 1::2] = 1., 0.
        models = calib.run_optim_sim(x)
        layer_models = None
    fine = [np.ravel(p) for p in calib.exp_pw]

    geometry = QD.WireGeometry.from_files(calibfile, cfg['detector']['detfile'], QD.wire_positions_of_scan(cfg['scan']),
                                          sample_tilt=cfg['sample']['tilt'])
    rows, curves = [], []
    for i in range(nb):
        wid = int(calib.exp_wire[i])
        k = sum(1 for j in range(i) if calib.exp_wire[j] == wid)
        xcam, ycam = calib.data_XYcam[wid][k]
        pw, intensity = measured[i], np.ravel(calib.exp_I[i]).astype(float)
        quality, _, _, _ = _fluo_profile_quality(scan, wid, xcam, ycam, pw, intensity, crit, spots, calib.data_hbs)
        shapes = [np.ravel(models[i])] + ([layer_models[i]] if layer_models is not None else [])

        def curve(params, return_ab=False):
            shift, sigma = params
            columns = [np.interp(pw - shift * 1e-3, fine[i],
                                 ndimage.gaussian_filter1d(shape, max(sigma, 1e-3) * 1e-3 / dp, mode='nearest'))
                       for shape in shapes]
            design = np.vstack(columns + [np.ones_like(pw)]).T
            if len(shapes) > 1:   # material and layer amplitudes >= 0
                coef = optimize.nnls(design, intensity)[0]
            else:
                coef = np.linalg.lstsq(design, intensity, rcond=None)[0]
            fitted_values = design @ coef
            ab = (coef[0], coef[-1]) if not return_ab else coef
            return (fitted_values, ab) if return_ab else fitted_values - intensity

        shifts = np.arange(-25., 25.1, 0.5)
        start = shifts[np.argmin([np.sum(curve((sh, 1.)) ** 2) for sh in shifts])]
        res = optimize.least_squares(curve, [start, 1.], bounds=([-40., 0.], [40., max_sigma]), x_scale=[1., 1.])
        fitted, coef = curve(res.x, return_ab=True)
        a, b = coef[0], coef[-1]
        layer_fraction = float(coef[1] / (coef[0] + coef[1])) if len(coef) > 2 and coef[0] + coef[1] > 0 else np.nan
        dof = max(1, len(pw) - 4)
        s2 = np.sum(res.fun ** 2) / dof
        try:
            cov = np.linalg.inv(res.jac.T @ res.jac) * s2
            sigma_err = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            sigma_err = np.nan
        # microns of ybeam per micron of wire travel at this pixel
        _, axis0, _ = geometry.tangent_positions(xcam, ycam, [0., 10.])
        dy_dp = 10. / abs((axis0[wid][1] - axis0[wid][0]) * 1e3)
        two_theta, chi = geometry.scattering_angles(xcam, ycam, 0.)
        contrast = abs(a) * (shapes[0].max() - shapes[0].min())
        rows.append({'wire': wid, 'point': k, 'X': int(xcam), 'Y': int(ycam),
                     'two_theta': float(np.ravel(two_theta)[0]), 'chi': float(np.ravel(chi)[0]),
                     'ok': quality['ok'], 'sigma': float(res.x[1]), 'sigma_err': sigma_err,
                     'shift': float(res.x[0]), 'sigma_ybeam': float(res.x[1] * dy_dp),
                     'shift_ybeam': float(res.x[0] * dy_dp),
                     'rms': float(np.sqrt(s2) / contrast) if contrast > 0 else np.nan,
                     'layer_fraction': layer_fraction})
        curves.append((wid, k, pw, intensity, fitted))
    table = pd.DataFrame(rows)
    # fits clearly worse than the others (e.g. a dip cut by the window): not used in the summary
    table['fit_ok'] = table['rms'] <= 2. * table['rms'].median()
    if verbose:
        good = table[table['ok'] & table['fit_ok']]
        print(f"blur of the fluorescence dips ({os.path.basename(calibfile)}), {len(good)}/{len(table)} usable profiles:")
        for wid, group in good.groupby('wire'):
            print(f"  wire {wid}: sigma {group['sigma_ybeam'].median():.2f} um along the beam (median of "
                  f"{len(group)}; min {group['sigma_ybeam'].min():.2f}, max {group['sigma_ybeam'].max():.2f}), "
                  f"shift {group['shift_ybeam'].median():+.2f} um")
        if len(good):
            GT.printgreen(f"  all: sigma {good['sigma_ybeam'].median():.2f} um along the beam "
                          f"(FWHM {2.355 * good['sigma_ybeam'].median():.2f} um)")
    if plot and curves:
        import matplotlib.pyplot as plt
        nbwires = len(calib.data_XYcam)
        nbpoints = max(len(p) for p in calib.data_XYcam)
        fig, axes = plt.subplots(nbwires, nbpoints, figsize=(2.6 * nbpoints, 2.2 * nbwires), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for (wid, k, pw, intensity, fitted), (_, row) in zip(curves, table.iterrows()):
            ax = axes[wid][k]
            ax.set_visible(True)
            ax.plot(pw, intensity, '.', ms=2, color='0.4')
            ax.plot(pw, fitted, '-', lw=1, color='tab:blue' if row['ok'] and row['fit_ok'] else 'tab:red')
            ax.set_title(f"w{wid} p{k}: sigma {row['sigma_ybeam']:.2f} um", fontsize=8,
                         color='k' if row['ok'] else 'tab:red')
            ax.tick_params(labelsize=6)
        fig.suptitle(f'fluorescence dips fitted with a gaussian blur ({os.path.basename(calibfile)}; red: profile '
                     f'not usable or bad fit)', fontsize=9)
        fig.tight_layout()
    return table


def refine_wires_fluo(cfg: dict, init_calib: Optional[str] = None, stages=FLUO_STAGES, prefix: str = 'wires_fluo',
                      wires: Optional[Sequence[int]] = None, excluded_wires: Optional[Sequence[int]] = None,
                      replace_bad_points: bool = False, probe_all: bool = False, avoid_spots: bool = True,
                      fast: bool = True, nb_cpus: Optional[int] = None, verbose: bool = True) -> list:
    """refine the wires trajectory parameters with the usual fluorescence calibration (CalibFluo, stages of
    Daxm/scripts/calib.py), starting from init_calib (default: wires_from_dips.calib of approximate_wires()).
    Each stage writes <output folder>/wirescalib/<prefix>_stage<n>.calib; the last one is the result.
    In this process: see refine_wires_fluo_job() for a background or SLURM job (magnifix).

    :param wires: wires refined (e.g. [0, 1, 2]); the others keep their starting parameters (they stay in the
        .calib files: their shadows are known). None: all the wires
    :param excluded_wires: wires excluded from the depth analysis and the reconstruction, written in the .calib files
        (key 'excluded_wires'). None: the wires not refined
    :param replace_bad_points: at each stage, replace the points of the grid whose profile is not usable (criteria
        of check_fluo_profiles(), e.g. a dip cut by the scan limits, a Laue spot in the box) by usable points nearby
    :param probe_all: at each stage, probe all the points (best fluorescence dip template nearby)
    :param avoid_spots: with replace_bad_points or probe_all, boxes near Laue spots (fluo_spots()) are not used
    :param fast: optimizer of LaueTools.Daxm.fastcalib (same model and results; amplitudes of the profiles solved
        exactly, jacobian computed in parallel on nb_cpus); False: Calib.run() of the legacy classes (one cpu)
    :param nb_cpus: [fast] processes for the jacobian (default: all the cpus of the machine or SLURM job)

    :return: list of dict per stage: stage, dims, var, fval, gof, dm, seconds, calibfile (, layer, optimizer)
    """
    import time
    from LaueTools.Daxm.classes.scan.point import save_scan_dict
    files = calib_files(cfg)
    init_calib = init_calib or files['dips']
    os.makedirs(files['folder'], exist_ok=True)
    save_scan_dict(legacy_scan_dict(cfg, init_calib), files['scan'])
    scan, calib = _legacy_calib(cfg, init_calib, verbose=verbose)
    spots = fluo_spots(cfg) if avoid_spots and (replace_bad_points or probe_all) else None
    if excluded_wires is None:
        excluded = [] if wires is None else sorted(set(range(scan.wire_qty)) - {int(w) for w in wires})
    else:
        excluded = sorted(int(w) for w in excluded_wires)
    if verbose and wires is not None:
        print(f'wires refined: {list(wires)}; excluded from the depth analysis and the reconstruction: {excluded}')
    summary = []
    for k, (dims, span, var) in enumerate(stages):
        t0 = time.time()
        if span:
            calib.data_span = span
        calib.set_points_grid(dims=dims, adjust=False)
        trim_calib_profiles(scan, calib)
        if replace_bad_points or probe_all:
            replace_fluo_points(scan, calib, spots=spots, probe_all=probe_all, verbose=verbose)
        optimizer = None
        if fast:
            from LaueTools.Daxm.fastcalib import run_fast
            optimizer = run_fast(calib, wire=None if wires is None else list(wires), var=var, nb_cpus=nb_cpus,
                                 verbose=verbose)
        else:
            calib.run(wire=None if wires is None else list(wires), var=var)
        name = f'{prefix}_stage{k + 1}'
        calib.save_wires(name, directory=files['folder'])
        if excluded:
            QD.set_excluded_wires(os.path.join(files['folder'], name + '.calib'), excluded)
        layer = None
        if hasattr(calib, 'var_fL'):   # CalibFluoLayer: layer written in the .calib (key 'fluo_layer')
            fractions = calib.var_fL / (1. + calib.var_fL)
            _set_fluo_layer(os.path.join(files['folder'], name + '.calib'), calib.var_ld, fractions)
            layer = {'thickness_um': round(calib.var_ld * 1000., 2), 'fraction': round(float(np.median(fractions)), 4)}
        summary.append({'stage': k + 1, 'dims': dims, 'var': var, 'fval': float(calib.log_fval[-1]),
                        'gof': float(calib.log_gof[-1]),
                        'dm': None if calib.log_dm[-1] is None else float(calib.log_dm[-1]),
                        'seconds': round(time.time() - t0, 1),
                        'calibfile': os.path.join(files['folder'], name + '.calib')})
        if layer is not None:
            summary[-1]['layer'] = layer
        if optimizer is not None:
            summary[-1]['optimizer'] = optimizer
        if verbose:
            GT.printgreen(f"stage {k + 1}: GoF {summary[-1]['gof']:.2f} ({summary[-1]['seconds']:.0f} s) "
                          f"-> {summary[-1]['calibfile']}"
                          + (f"; layer {layer['thickness_um']} um along the beam, {100 * layer['fraction']:.1f} % "
                             f"of the signal" if layer else ''))
    with open(os.path.join(files['folder'], f'{prefix}_summary.yaml'), 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return summary


def refine_wires_fluo_job(cfg: dict, resource: str = 'magnifix', wait: bool = True, nb_cpus: Optional[int] = None,
                          time_limit: str = '02:00:00', mem_per_cpu: str = '2G', **kwargs):
    """refine_wires_fluo() as a job: 'local' (background process on this machine) or a SLURM resource of
    workflowlauncher.COMPUTING_RESOURCES (e.g. 'magnifix': the jacobian of the fast optimizer on all the cpus of a
    node). Same arguments as refine_wires_fluo() (kwargs: init_calib, stages, prefix, wires, excluded_wires,
    replace_bad_points, probe_all, avoid_spots); 'kernel': refine_wires_fluo() in this process.

    :return: list of dict per stage (see refine_wires_fluo()), or the job if wait=False (job.wait(), then
        job.results())
    """
    if resource == 'kernel':
        return refine_wires_fluo(cfg, nb_cpus=nb_cpus, **kwargs)
    import LaueTools.scripts.workflowlauncher as WL
    definition = WL.COMPUTING_RESOURCES.get(resource, {})
    if definition.get('nodes', 1) > 1:   # one calibration: one node
        single = next((name for name, res in WL.COMPUTING_RESOURCES.items()
                       if res.get('partition') == definition.get('partition') and res.get('nodes', 1) == 1), None)
        GT.printyellow(f'{resource}: {definition["nodes"]} nodes for one calibration: {single} (1 node) used instead')
        resource = single or 'local'
    if 'stages' in kwargs:
        kwargs['stages'] = [list(stage) for stage in kwargs['stages']]
    output = os.path.join(calib_files(cfg)['folder'], f"{kwargs.get('prefix', 'wires_fluo')}_job.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    job = WL.launch(WL.make_config({'cfg': cfg, 'kwargs': kwargs}, output, workflow='wirescalib'), resource=resource,
                    nbcpus=nb_cpus, time_limit=time_limit, mem_per_cpu=mem_per_cpu)
    if not wait:
        return job
    status = job.wait()
    if status['state'] != 'COMPLETED':
        raise RuntimeError(f"wires calibration job {status['state']}: {status.get('message', '')}\n{job.log_tail()}")
    return job.results()


# stage of refine_wires_local(): wires of a reference calibration, only p0 (or h, p0) refined
FLUO_LOCAL_STAGES = (([3, 3], None, ['p0']),)


def refine_wires_local(cfg: dict, reference_calib: str, var: Sequence[str] = ('p0',), dims=(3, 3),
                       p0_offset_um: Optional[float] = None, prefix: str = 'wires_local', **kwargs) -> list:
    """wires calibration at another place of the sample from a reference calibration (refined with all the
    parameters, e.g. on magnifix): the wires parameters describe the wires relative to the beam and the detector,
    only the depth zero (surface entry point of the beam) changes. Starts from reference_calib, p0 of all the wires
    shifted by p0_offset_um (e.g. envelope offset of quick daxm grid, see shift_calib()), and refines only var
    (default p0; ('h', 'p0') if the wires may have moved vertically) with the fluorescence of the scan of cfg: a few
    variables, seconds to a minute. The layer of a coated sample (key 'fluo_layer' of the reference) keeps its
    thickness, its fraction of the signal is fitted per profile. The excluded wires of the reference are kept.

    :param kwargs: of refine_wires_fluo() (replace_bad_points, fast, nb_cpus...)
    :return: list of dict of the stage (see refine_wires_fluo()), calibfile: <output folder>/wirescalib/<prefix>_stage1.calib
    """
    folder = calib_files(cfg)['folder']
    start = reference_calib
    if p0_offset_um:
        start = shift_calib(reference_calib, p0_offset_um, os.path.join(folder, f'{prefix}_start.calib'))
    cfg = copy.deepcopy(cfg)
    layer = read_fluo_layer(reference_calib)
    if layer:
        cfg['wirescalib']['fluo_layer'] = layer['thickness']
        cfg['wirescalib']['fluo_layer_fixed'] = True
    excluded = QD.read_excluded_wires(reference_calib)
    wires = None
    if excluded:
        with open(reference_calib, 'r') as f:
            nbwires = len(json.load(f)['wires'])
        wires = [w for w in range(nbwires) if w not in excluded]
    stages = (([int(d) for d in dims], None, list(var)),)
    summary = refine_wires_fluo(cfg, init_calib=start, stages=stages, prefix=prefix, wires=wires,
                                excluded_wires=excluded or None, **kwargs)
    # depth zero of this place relative to the one of the reference (mean p0 change of the refined wires), for
    # the source position of 2theta, chi (quickdaxm.detector_offset_um())
    with open(reference_calib, 'r') as f:
        ref = json.load(f)
    for stage in summary:
        with open(stage['calibfile'], 'r') as f:
            content = json.load(f)
        used = [w for w in range(len(content['wires'])) if w not in excluded]
        shift = 1000. * np.mean([content['wires'][w]['p0'] - ref['wires'][w]['p0'] for w in used])
        content['p0_offset_um'] = float(ref.get('p0_offset_um', 0.)) + float(shift)
        content['reference_calib'] = str(reference_calib)
        with open(stage['calibfile'], 'w') as f:
            f.write(json.dumps(content, indent=1))
    return summary


# --------------------------------------------------------------------------------------
#  QUICK DAXM GRID: all the wire scans of a raster / grid, depth layers, indexation
# --------------------------------------------------------------------------------------
# settings of a quick daxm grid (notebook 3_quick_daxm_grid)
GRID_SETTINGS = {
    'raw_data_folder': None, 'contains': None, 'output_root': None,
    'grid_folder': None,          # folder of the grid results (layers .cor files, indexation, summary)
    'datasets': [],               # wire scans of the grid (dataset names of list_wire_scans())
    'detfile': 'auto', 'calibfile': None,
    'reference_offset': 0.,       # envelope offset (microns) of the calibration scan with its calibration
    'p0_offset': 'envelope',      # p0 of the wires shifted for each scan: 'envelope' or None (calibration as it is)
    'offset_outliers': 0.02, 'offset_min_dips': 10,
    'quickdaxm': {}, 'redo': False,
    'layer_bandwidth': 2., 'layer_min_spots': 8, 'layer_prominence': 0.1,
    'write_slices': False,        # also the slices of fixed thickness of each scan (as 2_quick_daxm)
    'positions': None,            # {dataset: position in the grid} (set for the jobs of run_quickdaxm_grid())
}


def grid_position(dataset: str) -> Tuple[Optional[int], Optional[int]]:
    """(row, column) of a wire scan of a grid from its dataset name (..._row_col_<row>_<col>_<scan>), else (None, None)"""
    match = re.search(r'row_col_(\d+)_(\d+)', str(dataset))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def envelope_offset(results: Dict[str, Any], calibfile: str, outliers: float = 0.02,
                    min_dips: int = 10) -> Tuple[float, Dict[int, float]]:
    """position along the beam (microns) of the shallowest sources of a wire scan with a calibration: for each wire,
    the quantile 'outliers' of ybeam of its dips; the offset is the median over the wires with min_dips dips
    (nan if none). The difference of offsets of two scans is the shift of the sample surface along the beam. Offsets
    are found within ybeam_range of the quick daxm parameters (shifts larger than this range are not found)"""
    geometry = QD.WireGeometry.from_files(calibfile, results['detfile'], results['wire_positions'],
                                          results.get('yref', 0.), sample_tilt=results.get('sample_tilt', 40.))
    # ybeam_range of the analysis: a wider range lets dips be assigned to the neighbouring wire (one wire spacing away)
    dips, _ = QD.analyse(results['signal'], results['xy'], geometry, results['params'])
    per_wire = {}
    for iw in range(geometry.nb_wires):
        values = dips['ybeam'][(dips['wire'] == iw) & np.isfinite(dips['ybeam'])]
        if len(values) >= min_dips:
            per_wire[iw] = float(np.quantile(values, outliers))
    return (float(np.median(list(per_wire.values()))) if per_wire else np.nan), per_wire


def shift_calib(calibfile: str, offset_um: float, path: Union[str, Path]) -> str:
    """copy of a .calib file with p0 of all the wires shifted by offset_um (microns): the depth zero moved by
    offset_um along the beam (wires moving along the beam, u1 = u2 ~ 0). Other parameters unchanged"""
    with open(calibfile, 'r') as f:
        content = json.load(f)
    for wire in content['wires']:
        wire['p0'] = float(wire['p0']) + float(offset_um) * 1e-3
    content['p0_offset_um'] = float(offset_um)
    content['reference_calib'] = str(calibfile)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=1))
    return str(path)


def grid_folder(table, index, name: Optional[str] = None) -> Path:
    """default folder of the results of a grid: <PROCESSED_DATA (or output root)>/<sample>/<name> (name: default
    <sample>_quickdaxm_grid), from the results folder of a wire scan of the grid"""
    workdir = Path(table.at[index, 'workdir'])        # <root>/<sample>/<dataset>/scanXXXX/daxm
    sample = workdir.parents[2]
    return sample / (name or f'{sample.name}_quickdaxm_grid')


def reference_offset(table, calibfile: str, first_index: int, nb_cpus: Optional[int] = None,
                     settings: Optional[dict] = None, verbose: bool = True) -> float:
    """envelope offset (envelope_offset()) of the wire scan the calibration was made with (the scan of the table
    whose results folder contains calibfile; its quick daxm results are computed if needed): the p0 of each scan of the
    grid is shifted by (its offset - this offset). Calibration of another origin (Ge scan of another sample folder,
    .calib file given): the first scan of the grid (index first_index) is the reference"""
    calibfile = Path(calibfile).resolve()
    owner = next((int(i) for i in table.index if Path(table.at[i, 'workdir']).resolve() in calibfile.parents), None)
    index = owner if owner is not None else int(first_index)
    if verbose:
        print(f"reference of the p0 shifts: {table.at[index, 'dataset']}"
              + ('' if owner is not None else ' (first scan of the grid: calibration not made with a scan of the table)'))
    output = Path(table.at[index, 'workdir']) / 'quickdaxm.h5'
    if output.exists():
        results = QD.load_results(str(output))
    else:
        cfg = config_for_scan(table, index, detfile=(settings or {}).get('detfile', 'auto'), calibfile=None,
                              verbose=False, **(settings or {}).get('quickdaxm', {}))
        results = run_quickdaxm(cfg, resource='kernel', nb_cpus=nb_cpus)
    offset, per_wire = envelope_offset(results, str(calibfile))
    if verbose:
        print(f'reference offset: {offset:.2f} µm (per wire: {{{", ".join(f"{k}: {v:.1f}" for k, v in per_wire.items())}}})')
    return offset


def _layer_files_folder(settings: dict) -> Path:
    return Path(settings['grid_folder']) / 'layers'


def process_grid_scan(table, index, settings: dict, position: int, nb_cpus: Optional[int] = None,
                      verbose: bool = False) -> dict:
    """quick daxm of one wire scan of a grid: dips, depth with the calibration (p0 shifted by the envelope offset of
    the scan), checks, depth layers and their .cor files (<grid folder>/layers/layer_<position><layer>.cor, 2theta
    and chi of each spot from its own depth). Summary written in <results folder of the scan>/grid_summary.json"""
    row = table.loc[index]
    cfg = config_for_scan(table, index, detfile=settings['detfile'], calibfile=settings['calibfile'],
                          verbose=verbose, **settings['quickdaxm'])
    results = run_quickdaxm(cfg, resource='kernel', nb_cpus=nb_cpus, redo=settings['redo'])
    summary = {'dataset': row['dataset'], 'position': int(position), 'workdir': str(row['workdir']),
               'start': row['start'], 'calibfile': settings['calibfile'], 'offset': np.nan, 'p0_shift': 0.}
    summary['row'], summary['col'] = grid_position(row['dataset'])
    if settings['calibfile'] and settings['p0_offset'] == 'envelope':
        offset, per_wire = envelope_offset(results, settings['calibfile'], settings['offset_outliers'],
                                           settings['offset_min_dips'])
        summary.update(offset=offset, offsets_per_wire=per_wire)
        if np.isfinite(offset):
            shift = offset - float(settings['reference_offset'])
            calib = shift_calib(settings['calibfile'], shift, Path(row['workdir']) / 'wirescalib' / 'wires_grid.calib')
            new_config(cfg['configfile'], calibfile=calib)
            results.update(calibfile=calib)
            results['dips'], results['spots'] = QD.analyse(results['signal'], results['xy'],
                                                           QD.geometry_of_results(results), results['params'])
            QD.save_results(results, output_files(cfg)['quickdaxm'])
            summary.update(p0_shift=shift, calibfile=calib)
    spots, dips = results['spots'], results['dips']
    two = spots['nb_dips'] > 1
    found = np.isfinite(spots['ybeam'])
    summary.update(nb_spots=int(len(spots['X'])), nb_dips=int(len(dips['spot'])), spots_with_depth=int(found.sum()),
                   spread=float(np.median(spots['spread'][two])) if two.any() else np.nan,
                   nb_two_wires=int(two.sum()),
                   mismatch=float(np.nanmedian(dips['mismatch'])) if np.isfinite(dips['mismatch']).any() else np.nan,
                   ybeam_median=float(np.median(spots['ybeam'][found])) if found.any() else np.nan)
    layers = []
    if results.get('calibfile'):
        folder = _layer_files_folder(settings)
        results = dict(results, detector_offset=cfg['sample'].get('detector_offset', 'auto'))
        for layer in QD.depth_layers(results, settings['layer_bandwidth'], settings['layer_min_spots'],
                                     settings['layer_prominence']):
            name = f"layer_{int(position):04d}{layer['index']:02d}"
            corfile = QD.write_cor(results, name, str(folder), layer['spots'], ybeam_for_angles='spot')
            layers.append({key: value for key, value in layer.items() if key != 'spots'} | {'file': corfile})
        if settings['write_slices']:
            write_quickdaxm_files(cfg, results, verbose=False)
    summary['layers'] = layers
    summary['nb_layers'] = len(layers)
    Path(summary['workdir'], 'grid_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _grid_table(settings: dict):
    table = list_wire_scans(settings['raw_data_folder'], contains=settings['contains'],
                            output_root=settings['output_root'], verbose=False)
    # position of each scan in the whole grid (a job of run_quickdaxm_grid() gets a part of the scans)
    positions = settings.get('positions') or {name: k for k, name in enumerate(settings['datasets'])}
    return table, positions


def compute_quickdaxm_grid(settings: dict, nb_cpus: Optional[int] = None, progress_callback=None) -> List[dict]:
    """process_grid_scan() of all the wire scans settings['datasets'], one after the other (each scan with nb_cpus
    processes), in this process. Returns the summaries"""
    import time
    settings = {**GRID_SETTINGS, **settings}
    table, positions = _grid_table(settings)
    indices = {table.at[i, 'dataset']: int(i) for i in table.index}
    summaries = []
    t0 = time.time()
    for k, name in enumerate(settings['datasets']):
        try:
            summaries.append(process_grid_scan(table, indices[name], settings, positions[name], nb_cpus))
        except Exception as exc:   # one bad scan does not stop the grid
            summaries.append({'dataset': name, 'position': positions[name], 'error': f'{type(exc).__name__}: {exc}'})
            print(f'{name}: {type(exc).__name__}: {exc}', flush=True)
        if progress_callback is not None:
            progress_callback(k + 1, len(settings['datasets']))
        print(f"{k + 1}/{len(settings['datasets'])} wire scans done ({time.time() - t0:.0f} s)", flush=True)
    return summaries


def run_quickdaxm_grid(settings: dict, resource: str = 'kernel', nb_cpus: Optional[int] = None,
                       nb_jobs: Optional[int] = None, time_limit: str = '02:00:00', mem_per_cpu: str = '2G',
                       wait: bool = True):
    """quick daxm of the wire scans of a grid (settings: see GRID_SETTINGS):
    resource 'kernel': in this process, one scan after the other; 'local' or a SLURM resource: the scans are shared
    between nb_jobs jobs (default: nb of nodes of the resource, e.g. 3 for magnifix3: 3 jobs of one node each)

    :return: table of the summaries (load_grid_summary()) if wait, else the list of jobs
    """
    settings = {**GRID_SETTINGS, **settings}
    grid_folder = Path(settings['grid_folder'])
    grid_folder.mkdir(parents=True, exist_ok=True)
    if resource == 'kernel':
        compute_quickdaxm_grid(settings, nb_cpus)
        return load_grid_summary(settings)
    import LaueTools.scripts.workflowlauncher as WL
    definition = WL.COMPUTING_RESOURCES.get(resource, {})
    nodes = int(definition.get('nodes', 1))
    nb_jobs = nb_jobs or nodes
    if nodes > 1:   # one node per job
        resource = next((name for name, res in WL.COMPUTING_RESOURCES.items()
                         if res.get('partition') == definition.get('partition') and res.get('nodes', 1) == 1), resource)
    jobs = []
    for k, chunk in enumerate(np.array_split(np.array(settings['datasets'], dtype=object), nb_jobs)):
        if not len(chunk):
            continue
        part = dict(settings, datasets=list(chunk))
        part['positions'] = {name: settings['datasets'].index(name) for name in chunk}
        output = str(grid_folder / 'jobs' / f'grid_part{k}of{nb_jobs}.json')
        jobs.append(WL.launch(WL.make_config(part, output, workflow='quickdaxm_grid'), resource=resource,
                              nbcpus=nb_cpus, time_limit=time_limit, mem_per_cpu=mem_per_cpu))
    if not wait:
        return jobs
    for job in jobs:
        status = job.wait()
        if status['state'] != 'COMPLETED':
            GT.printred(f"job {job.describe(status)}\n{job.log_tail()}")
    return load_grid_summary(settings)


def load_grid_summary(settings: dict):
    """table (pandas DataFrame) of the summaries of the wire scans of a grid (grid_summary.json of each scan; scans
    not processed yet: missing), one line per scan, with row, col"""
    import pandas as pd
    settings = {**GRID_SETTINGS, **settings}
    table, positions = _grid_table(settings)
    rows = []
    for i in table.index:
        name = table.at[i, 'dataset']
        if name not in positions:
            continue
        path = Path(table.at[i, 'workdir']) / 'grid_summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            summary.pop('layers', None)
            summary.pop('offsets_per_wire', None)
            rows.append(summary)
    return pd.DataFrame(rows).sort_values('position').reset_index(drop=True) if rows else pd.DataFrame()


def load_grid_layers(settings: dict):
    """table (pandas DataFrame) of the depth layers of all the wire scans of a grid, one line per layer"""
    import pandas as pd
    settings = {**GRID_SETTINGS, **settings}
    table, positions = _grid_table(settings)
    rows = []
    for i in table.index:
        name = table.at[i, 'dataset']
        path = Path(table.at[i, 'workdir']) / 'grid_summary.json'
        if name in positions and path.exists():
            summary = json.loads(path.read_text())
            for layer in summary.get('layers', []):
                rows.append({'dataset': name, 'position': summary['position'], 'row': summary['row'],
                             'col': summary['col'], 'workdir': summary.get('workdir'), 'layer': layer['index'],
                             **{k: v for k, v in layer.items() if k != 'index'}})
    return pd.DataFrame(rows)


def grid_map(summary, column: str, layers=None) -> np.ndarray:
    """2D array (rows, cols) of a column of the table of load_grid_summary() (or of load_grid_layers() with layers:
    e.g. the shallowest layer of each scan, see the notebook), nan where missing"""
    data = summary if layers is None else layers
    rows, cols = data['row'].astype(int), data['col'].astype(int)
    out = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    out[rows, cols] = data[column].astype(float)
    return out


def index_layers(settings: dict, key_material: str, nb_grains: int = 1, nb_cpus: Optional[int] = None,
                 detfile: Optional[str] = None, CCDLabel: Optional[str] = None, **indexing) -> List[list]:
    """indexation and strain refinement of the .cor files of the depth layers of a grid (LaueTools indexation,
    indexing_batch), with the configuration file <grid folder>/configs/indexing_<material>.yaml (written at the
    first call from the template of notebooks/indexation, then kept: edit it for other parameters; indexing:
    parameters changed by name, e.g. e_max=24). .fit files in <grid folder>/fitfiles_<material>

    :return: results of indexing_batch.run_multiprocessing()
    """
    from LaueTools import indexing_batch as IB
    settings = {**GRID_SETTINGS, **settings}
    folder = Path(settings['grid_folder'])
    listfiles = sorted(str(p) for p in _layer_files_folder(settings).glob('layer_*.cor'))
    if not listfiles:
        raise FileNotFoundError(f'no .cor file of layers in {_layer_files_folder(settings)}')
    configfile = folder / 'configs' / f'indexing_{key_material}.yaml'
    changes = dict(indexing, key_material=key_material, nbGrainstoFind=nb_grains)
    if not configfile.exists():
        changes.update(cor_folder=str(_layer_files_folder(settings)), prefix='layer_', nbdigits=6,
                       fit_folder=str(folder / f'fitfiles_{key_material}'), mapdims=[len(listfiles), 1])
        if detfile:
            changes['detfile'] = str(detfile)
        if CCDLabel:
            changes['CCDLabel'] = CCDLabel
        cfg = IB.new_config(configfile, check=False, **changes)
    else:
        cfg = IB.load_config(configfile)
        cfg = IB.save_config(cfg, configfile, overwrite=True, **changes) if changes else cfg
    params = IB.params_from_config(cfg)
    Path(params.fit_folder).mkdir(parents=True, exist_ok=True)
    return IB.run_multiprocessing(listfiles, params, nb_cpus=nb_cpus)


def layers_results(settings: dict, key_material: str, grain: int = 0, depths: bool = True, sigma: float = 0.65):
    """table of the depth layers of a grid (load_grid_layers()) with the results of their indexation with
    key_material (grain index): nb of indexed spots, pixel deviation, deviatoric strain (10-3, crystal and sample
    frames, Voigt notation xx yy zz yz xz xy), new lattice parameters if written in the .fit file. The strain and
    lattice parameters are averages over the volume of the grain along the beam.

    :param depths: also the position and thickness along the beam of the grain from the dips of its spots
        (quickdaxm.grain_depth_summary(), sigma: blur of the instrument, microns): columns grain_nb_dips,
        grain_ybeam, grain_ybeam_min, grain_ybeam_max, grain_thickness_beam, grain_depth, grain_depth_min,
        grain_depth_max, grain_thickness_depth (below the depth zero of the wires calibration of the scan),
        grain_ybeam_detector (relative to the source point of the detector calibration)"""
    layers = load_grid_layers(settings)
    if not len(layers):
        return layers
    folder = Path({**GRID_SETTINGS, **settings}['grid_folder']) / f'fitfiles_{key_material}'
    values, cache = [], {}
    for corfile in layers['file']:
        fitfile = folder / f'{Path(corfile).stem}_g{grain}.fit'
        res = {'indexed': np.nan, 'pixdev': np.nan}
        if fitfile.exists():
            try:
                data = IOLT.readfitfile_multigrains(str(fitfile), readmore=True)
                res.update(indexed=float(data[1][0]), pixdev=float(np.ravel(data[6])[0]))
                strain, strain_sample = np.ravel(data[7])[:6], np.ravel(data[8])[:6]
                for k, name in enumerate(('xx', 'yy', 'zz', 'yz', 'xz', 'xy')):
                    res[f'eps_{name}'] = float(strain[k])
                    res[f'eps_sample_{name}'] = float(strain_sample[k])
                lines = fitfile.read_text(errors='replace').splitlines()
                for k, line in enumerate(lines):
                    if line.strip('# ').startswith('new lattice parameters') and k + 1 < len(lines):
                        numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', lines[k + 1])
                        for name, value in zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'), numbers[:6]):
                            res[name] = float(value)
                        break
            except Exception:   # unreadable file
                pass
        if depths and fitfile.exists():
            res.update(_grain_depths(layers.loc[len(values), 'workdir'] if 'workdir' in layers else None,
                                     fitfile, sigma, cache))
        values.append(res)
    import pandas as pd
    return pd.concat([layers.reset_index(drop=True), pd.DataFrame(values)], axis=1)


def _grain_depths(workdir, fitfile, sigma: float, cache: dict) -> dict:
    """columns grain_* of the position and thickness along the beam of the grain of fitfile, from the quick daxm
    results of workdir (results of a scan kept in cache)"""
    if not workdir:
        return {}
    path = Path(workdir) / 'quickdaxm.h5'
    if not path.exists():
        return {}
    if str(path) not in cache:
        results = QD.load_results(str(path))
        cache[str(path)] = (results, QD.geometry_of_results(results))
    results, geometry = cache[str(path)]
    if geometry is None:
        return {}
    row = QD.grains_depth_table(results, [str(fitfile)], geometry, sigma).iloc[0].to_dict()
    return {f'grain_{k}': v for k, v in row.items() if k != 'grain'}


# --------------------------------------------------------------------------------------
#  INDEXATION OF THE DEPTH SLICES OF ONE WIRE SCAN (notebook 2_quick_daxm)
# --------------------------------------------------------------------------------------
def slice_files(cfg: dict) -> Tuple[List[str], dict]:
    """.cor files of the depth slices of the scan of cfg (write_quickdaxm_files()) and the content of slices.yaml"""
    folder = Path(output_files(cfg)['quickdaxm_slices'])
    info = yaml.safe_load((folder / 'slices.yaml').read_text()) if (folder / 'slices.yaml').exists() else {}
    files = sorted(str(p) for p in folder.glob('*.cor'))
    return files, info


def index_slices(cfg: dict, materials: Sequence[str] = ('Cr', 'Zr'), nb_grains: int = 1, strategy: str = 'best_of',
                 nb_cpus: Optional[int] = None, files: Optional[Sequence[str]] = None, **indexing):
    """indexation from scratch of the .cor files of the depth slices of the scan of cfg, each slice trying all the
    materials (sequence of indexing_sequence, strategy 'best_of': at each grain, the material indexing the most
    spots is kept; 'ordered': materials in this order). Configuration <results folder>/configs/indexing_slices.yaml
    (written at each call from the template of notebooks/indexation; indexing: its parameters changed by name, e.g.
    e_max=24); .fit files in <results folder>/indexing_slices/fitfiles_<material>. See slices_ub_table().

    :param nb_grains: max nb of grains per slice (all materials)
    :param files: .cor files to index (default: all the slices, slice_files())
    :return: results of indexing_batch.run_multiprocessing()
    """
    from LaueTools import indexing_batch as IB
    listfiles = list(files) if files is not None else slice_files(cfg)[0]
    if not listfiles:
        raise FileNotFoundError(f"no .cor file of slices in {output_files(cfg)['quickdaxm_slices']}: "
                                'write_quickdaxm_files() first')
    stem = Path(listfiles[0]).stem
    match = re.match(r'(.*?)(\d+)$', stem)
    prefix, nbdigits = (match.group(1), len(match.group(2))) if match else (stem, 4)
    root = Path(cfg['output']['folder']) / 'indexing_slices'
    configfile = Path(cfg['output']['folder']) / 'configs' / 'indexing_slices.yaml'
    configfile.parent.mkdir(parents=True, exist_ok=True)
    changes = dict(indexing, key_material=materials[0], nbGrainstoFind=nb_grains,
                   cor_folder=str(Path(listfiles[0]).parent), prefix=prefix, nbdigits=nbdigits,
                   fit_folder=str(root), mapdims=[len(listfiles), 1], detfile=str(cfg['detector']['detfile']),
                   CCDLabel=cfg['scan']['CCDLabel'])
    icfg = IB.new_config(configfile, check=False, overwrite=True, **changes)
    sequence = {'strategy': strategy, 'max_grains': nb_grains, 'max_grains_per_material': nb_grains,
                'folders_root': str(root),
                'steps': [{'name': m, 'key_material': m, 'folder': m, 'nbGrainstoFind': nb_grains} for m in materials]}
    icfg = IB.save_config(icfg, configfile, overwrite=True, sequence=sequence, fit_folder=str(root),
                          results_folder=str(root), results_stamp='slices')
    params = IB.params_from_config(icfg)
    root.mkdir(parents=True, exist_ok=True)
    return IB.run_multiprocessing(listfiles, params, nb_cpus=nb_cpus)


def _fit_material(fitfile: Path) -> Optional[str]:
    lines = fitfile.read_text(errors='replace').splitlines()
    for k, line in enumerate(lines[:-1]):
        if line.strip() == '#Element':
            return lines[k + 1].strip('# ').strip()
    return None


def slices_ub_table(cfg: dict, results: Optional[Dict[str, Any]] = None, sigma: float = 0.65,
                    write_yaml: bool = True):
    """grains found by index_slices() in each depth slice: material, raw UB matrix (as written in the .fit file by
    the indexation, LaueTools frame), nb of indexed spots, pixel deviation, and the position and thickness along the
    beam of the grain from the dips of its spots (quickdaxm.grain_depth_summary(), with the quick daxm results of the
    scan, sigma: blur of the instrument). The UB matrices are also written in
    <results folder>/indexing_slices/slices_ub.yaml (section ub_matrices, names <material>_s<slice>_g<grain>: to be
    refined with the proper material and the mean depth of the grain, e.g. usepreviousUB of the indexation)

    :return: pandas DataFrame, one line per grain of each slice
    """
    import pandas as pd
    files, info = slice_files(cfg)
    slices = {Path(s['file']).stem: s for s in info.get('slices', []) if s.get('file')}
    root = Path(cfg['output']['folder']) / 'indexing_slices'
    if results is None and os.path.exists(output_files(cfg)['quickdaxm']):
        results = load_quickdaxm(cfg)
    if results is not None:
        results = dict(results, detector_offset=cfg['sample'].get('detector_offset', 'auto'))
    geometry = QD.geometry_of_results(results) if results is not None else None
    rows, ubs = [], {}
    for corfile in files:
        stem = Path(corfile).stem
        for fitfile in sorted(root.glob(f'fitfiles_*/{stem}_g*.fit')):
            try:
                data = IOLT.readfitfile_multigrains(str(fitfile), readmore=True)
            except Exception:
                continue
            material = _fit_material(fitfile) or fitfile.parent.name.replace('fitfiles_', '')
            ub = np.reshape(np.ravel(data[3])[:9], (3, 3))
            grain = int(re.search(r'_g(\d+)$', fitfile.stem).group(1))
            index = slices.get(stem, {}).get('index')
            row = {'slice': index, 'file': stem, 'slice_ybeam': slices.get(stem, {}).get('ybeam'),
                   'material': material, 'grain': grain, 'indexed': int(np.ravel(data[1])[0]),
                   'pixdev': float(np.ravel(data[6])[0]), 'UB': ub.round(8).tolist(), 'fitfile': str(fitfile)}
            if geometry is not None:
                depth = QD.grains_depth_table(results, [str(fitfile)], geometry, sigma).iloc[0].to_dict()
                row.update({k: v for k, v in depth.items() if k != 'grain'})
            rows.append(row)
            ubs[f'{material}_s{index if index is not None else stem}_g{grain}'] = ub.round(8).tolist()
    table = pd.DataFrame(rows)
    if write_yaml and ubs:
        meta = {name: {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()
                       if k in ('slice', 'material', 'grain', 'indexed', 'pixdev', 'slice_ybeam', 'ybeam', 'ybeam_min',
                                'ybeam_max', 'thickness_beam', 'depth', 'ybeam_detector')}
                for name, row in zip(ubs, rows)}
        meta = {name: {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
                for name, m in meta.items()}
        (root / 'slices_ub.yaml').write_text(yaml.safe_dump({'ub_matrices': ubs, 'grains': meta}, sort_keys=False))
    return table


def _proper_rotations(key_material: str) -> np.ndarray:
    """proper rotations of the point group of key_material in the frame of B0 (cubic: 24, hexagonal: 12 with c // z,
    else identity), for the misorientation between UB matrices (UB and UB.S equivalent)"""
    from LaueTools import orientationclustering as OC
    from LaueTools import dict_LaueTools as DictLT
    try:
        a, b, c, alpha, beta, gamma = DictLT.dict_Materials[key_material][1]
    except Exception:
        return OC.get_symmetry_operators(None)
    if np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90.):
        return OC.get_symmetry_operators('cubic')
    if np.isclose(a, b) and np.allclose([alpha, beta], 90.) and np.isclose(gamma, 120.):
        rz = [np.array([[np.cos(t), -np.sin(t), 0.], [np.sin(t), np.cos(t), 0.], [0., 0., 1.]])
              for t in np.radians(np.arange(0, 360, 60))]
        rx = np.diag([1., -1., -1.])
        return np.array(rz + [r @ rx for r in rz])
    return OC.get_symmetry_operators(None)


def group_slice_grains(table, threshold: float = 1.0):
    """same grain found in several slices: grains of slices_ub_table() of the same material with UB matrices
    misoriented by less than threshold (deg, with the symmetry of the material). Adds the columns grain_id (e.g.
    'Zr 1', by increasing ybeam) and misorientation (deg, to the first slice of its grain)"""
    from LaueTools import orientationclustering as OC
    table = table.copy()
    table['grain_id'], table['misorientation'] = '', np.nan
    if not len(table):
        return table
    order = table['ybeam'].fillna(table['slice_ybeam']).sort_values().index
    for material, rows in table.loc[order].groupby('material', sort=False):
        ops = _proper_rotations(material)
        references = []
        for i in rows.index:
            rot = OC.orthonormalize(np.asarray(table.at[i, 'UB'], dtype=float))
            found = None
            for k, ref in enumerate(references):
                angle = float(OC.misorientation(ref, rot, ops)[0][0])
                if angle < threshold:
                    found = (k, angle)
                    break
            if found is None:
                references.append(rot)
                found = (len(references) - 1, 0.)
            table.at[i, 'grain_id'] = f'{material} {found[0] + 1}'
            table.at[i, 'misorientation'] = found[1]
    return table


def plot_slices_grains(table, threshold: float = 1.0, detector_offset: Optional[float] = None,
                       sample_tilt: float = 40., axes=None):
    """results of the indexation of the depth slices (slices_ub_table()): left, the grains along the beam (one line
    per grain, group_slice_grains(): bar from the onset to the end of the dips of its spots, blur removed in the
    label thickness; points: the slices where it was indexed, size ~ nb of indexed spots); right, nb of indexed
    spots and pixel deviation of each slice per material. Depth zero of the wires calibration at 0; dashed line:
    source point of the detector calibration (-detector_offset)

    :return: the table with the columns grain_id, misorientation
    """
    import matplotlib.pyplot as plt
    table = group_slice_grains(table, threshold)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(15, max(4., 0.35 * table['grain_id'].nunique() + 2.)),
                               gridspec_kw={'width_ratios': [3, 2]})
    materials = sorted(table['material'].unique()) if len(table) else []
    colors = {m: f'C{k}' for k, m in enumerate(materials)}
    ax = axes[0]
    grains = (table.groupby('grain_id').agg(material=('material', 'first'), ybeam=('ybeam', 'median'),
                                            ybeam_min=('ybeam_min', 'median'), ybeam_max=('ybeam_max', 'median'),
                                            thickness=('thickness_beam', 'median'), nb=('slice', 'size'))
              .sort_values('ybeam') if len(table) else None)
    if grains is not None:
        for row, (gid, g) in enumerate(grains.iterrows()):
            c = colors[g['material']]
            ax.plot([g['ybeam_min'], g['ybeam_max']], [row, row], '-', color=c, lw=6, alpha=0.45, solid_capstyle='butt')
            sel = table[table['grain_id'] == gid]
            ax.scatter(sel['slice_ybeam'], [row] * len(sel), s=10 + 2 * sel['indexed'], color=c, edgecolor='k',
                       zorder=3)
            ax.text(np.nanmax([g['ybeam_max'], sel['slice_ybeam'].max()]) + 1.5, row,
                    f"{gid}: {g['thickness']:.1f} µm", va='center', fontsize=8)
        ax.set_yticks(range(len(grains)))
        ax.set_yticklabels(grains.index, fontsize=8)
        ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.8)
    if detector_offset:
        ax.axvline(-detector_offset, color='k', lw=0.8, ls='--')
    ax.set_xlabel('ybeam (µm along the beam; 0: depth zero of the wires calibration)')
    sin = np.sin(np.radians(sample_tilt))
    top = ax.secondary_xaxis('top', functions=(lambda y: y * sin, lambda d: d / sin))
    top.set_xlabel('depth (µm)')
    ax.set_title(f'grains along the beam (same grain: misorientation < {threshold:g} deg)', fontsize=9)
    ax = axes[1]
    width = 0.8 * (np.median(np.diff(np.sort(table['slice_ybeam'].unique()))) if table['slice_ybeam'].nunique() > 1
                   else 5.)
    bottom = None
    if len(table):
        per = table.groupby(['slice_ybeam', 'material'])['indexed'].sum().unstack(fill_value=0)
        bottom = np.zeros(len(per))
        for m in per.columns:
            ax.bar(per.index, per[m], width=width, bottom=bottom, color=colors[m], label=m)
            bottom += per[m].values
        twin = ax.twinx()
        twin.plot(table['slice_ybeam'], table['pixdev'], 'k.', ms=5, label='pixel deviation')
        twin.set_ylabel('pixel deviation')
        ax.legend(fontsize=8, loc='upper right')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('ybeam of the slice (µm)')
    ax.set_ylabel('nb of indexed spots')
    ax.set_title('indexation of each slice', fontsize=9)
    axes[0].figure.tight_layout()
    return table


def fit_summary(fitfile) -> Dict[str, Any]:
    """first grain of a .fit file: nb of indexed spots, pixel deviation, UB matrix, deviatoric strain (10-3, crystal
    and sample frames, Voigt notation xx yy zz yz xz xy), new lattice parameters if written"""
    res = {'indexed': np.nan, 'pixdev': np.nan}
    data = IOLT.readfitfile_multigrains(str(fitfile), readmore=True)
    res.update(indexed=int(np.ravel(data[1])[0]), pixdev=float(np.ravel(data[6])[0]),
               UB=np.reshape(np.ravel(data[3])[:9], (3, 3)).round(8).tolist())
    strain, strain_sample = np.ravel(data[7])[:6], np.ravel(data[8])[:6]
    for k, name in enumerate(('xx', 'yy', 'zz', 'yz', 'xz', 'xy')):
        res[f'eps_{name}'] = float(strain[k])
        res[f'eps_sample_{name}'] = float(strain_sample[k])
    lines = Path(fitfile).read_text(errors='replace').splitlines()
    for k, line in enumerate(lines):
        if line.strip('# ').startswith('new lattice parameters') and k + 1 < len(lines):
            numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', lines[k + 1])
            for name, value in zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'), numbers[:6]):
                res[name] = float(value)
            break
    return res


def append_daxm_block(fitfile, info: Dict[str, Any]) -> None:
    """append to a .fit file the DAXM information of its grain (comment lines '#DAXM key: value' after a line
    '#DAXM depth of the grain', ignored by the readers of .fit files)"""
    lines = ['#DAXM depth of the grain (microns; ybeam along the beam from the depth zero of the wires calibration, '
             'depth normal to the surface)']
    for key, value in info.items():
        if isinstance(value, (float, np.floating)):
            value = f'{float(value):.4f}'
        lines.append(f'#DAXM {key}: {value}')
    with open(fitfile, 'a') as f:
        f.write('\n' + '\n'.join(lines) + '\n')


def read_daxm_block(fitfile) -> Dict[str, str]:
    """DAXM information appended by append_daxm_block() (values as strings)"""
    res = {}
    for line in Path(fitfile).read_text(errors='replace').splitlines():
        if line.startswith('#DAXM ') and ': ' in line:
            key, value = line[6:].split(': ', 1)
            res[key.strip()] = value.strip()
    return res


def refine_slice_grains(cfg: dict, table=None, results: Optional[Dict[str, Any]] = None, threshold: float = 1.0,
                        margin: float = 2., sigma: float = 0.65, nb_cpus: Optional[int] = None, **indexing):
    """refinement of the grains found in the depth slices (slices_ub_table(), grouped by group_slice_grains()) with
    their material, their UB matrix (of the slice where it indexes the most spots) and the parallax of their own
    position along the beam: for each grain, a .cor file of the spots of the scan with ybeam in [ybeam_min - margin,
    ybeam_max + margin] of the grain, 2theta and chi computed for a source at the median ybeam of the grain (+ the
    detector offset, detector_offset_um()); indexation with this UB only (usepreviousUB, skipindexing), i.e.
    refinement of orientation and deviatoric strain averaged over the thickness of the grain along the beam.
    Files in <results folder>/grains: grain_<n>.cor, fitfiles/grain_<n>_g0.fit, with the DAXM block (grain id,
    slices, source position, ybeam, ybeam_min, ybeam_max, thickness_beam, depth, depth_min, depth_max ...,
    read_daxm_block()) appended

    :param indexing: changes of the indexation parameters (e.g. e_max=24)
    :return: pandas DataFrame, one line per grain: grain_id, material, slices, results of the refinement
        (fit_summary(): indexed, pixdev, UB, strains, lattice parameters) and depth of the grain from the dips of
        its indexed spots (quickdaxm.grain_depth_summary())
    """
    import pandas as pd
    from LaueTools import indexing_batch as IB
    if table is None:
        table = slices_ub_table(cfg, results, sigma=sigma, write_yaml=False)
    if results is None:
        results = load_quickdaxm(cfg)
    results = dict(results, detector_offset=cfg['sample'].get('detector_offset', 'auto'))
    offset = QD.detector_offset_um(results)
    geometry = QD.geometry_of_results(results)
    table = group_slice_grains(table, threshold)
    folder = Path(cfg['output']['folder']) / 'grains'
    fitfolder = folder / 'fitfiles'
    fitfolder.mkdir(parents=True, exist_ok=True)
    configfile = Path(cfg['output']['folder']) / 'configs' / 'indexing_grains.yaml'
    first = table['material'].iloc[0] if len(table) else 'Si'
    icfg = IB.new_config(configfile, check=False, overwrite=True,
                         **dict(indexing, key_material=first, cor_folder=str(folder), prefix='grain_', nbdigits=4,
                                fit_folder=str(fitfolder), mapdims=[max(1, table['grain_id'].nunique()), 1],
                                detfile=str(cfg['detector']['detfile']), CCDLabel=cfg['scan']['CCDLabel'],
                                nbGrainstoFind=1))
    params_of = {}
    rows = []
    grains = table.groupby('grain_id', sort=False)
    order = sorted(grains.groups, key=lambda gid: np.nanmedian(table.loc[grains.groups[gid], 'ybeam']))
    for number, gid in enumerate(order):
        sel = table.loc[grains.groups[gid]]
        best = sel.loc[sel['indexed'].idxmax()]
        material = best['material']
        ymin, ymax = np.nanmedian(sel['ybeam_min']), np.nanmedian(sel['ybeam_max'])
        ycenter = np.nanmedian(sel['ybeam'])
        indices = QD.select_spots(results, (ymin - margin, ymax + margin))
        name = f'grain_{number:04d}'
        row = {'grain_id': gid, 'material': material, 'slices': [int(v) for v in sel['slice'] if pd.notna(v)],
               'nb_spots_cor': int(len(indices)), 'source_ybeam': float(ycenter), 'detector_offset': float(offset)}
        if len(indices) < 6:
            rows.append(row)
            continue
        corfile = QD.write_cor(results, name, str(folder), indices, ybeam_for_angles=ycenter)
        if material not in params_of:
            params_of[material] = IB.params_from_config(icfg, key_material=material, skipindexing=True,
                                                        ignorefitfileresults=True, writefitfile=True, nbGrainstoFind=1)
        import dataclasses
        params = dataclasses.replace(params_of[material], usepreviousUB=[np.array(best['UB'], dtype=float)])
        IB.index_refine(corfile, params)
        fitfiles = sorted(fitfolder.glob(f'{name}_g*.fit'))
        if not fitfiles:
            row['message'] = 'UB of the slice does not index the spots of the grain'
            rows.append(row)
            continue
        fitfile = fitfiles[0]
        row.update(fit_summary(fitfile))
        depth = QD.grains_depth_table(results, [str(fitfile)], geometry, sigma).iloc[0].to_dict()
        row.update({k: v for k, v in depth.items() if k != 'grain'})
        row['fitfile'] = str(fitfile)
        append_daxm_block(fitfile, {
            'grain_id': gid, 'material': material, 'slices': row['slices'],
            'slice_UB_from': f"slice {best['slice']} ({Path(best['fitfile']).name})",
            'source_ybeam_for_2theta_chi': ycenter, 'detector_offset': offset,
            'spots_ybeam_range': f'[{ymin - margin:.2f}, {ymax + margin:.2f}]',
            **{k: depth[k] for k in ('nb_dips', 'ybeam', 'ybeam_min', 'ybeam_max', 'thickness_beam',
                                     'thickness_beam_spread', 'depth', 'depth_min', 'depth_max', 'thickness_depth',
                                     'ybeam_detector')},
            'instrument_blur_sigma': sigma, 'sample_tilt': results.get('sample_tilt', 40.),
            'calibfile': results.get('calibfile'), 'quickdaxm_results': output_files(cfg)['quickdaxm']})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_json(folder / 'grains.json', orient='records', indent=1, default_handler=str)
    return out


# --------------------------------------------------------------------------------------
#  COMMAND LINE
# --------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m LaueTools.daxm_workflow',
                                     description='DAXM workflows described in a YAML configuration file')
    parser.add_argument('workflow', choices=['quickdaxm'])
    parser.add_argument('config', help='daxm YAML configuration file')
    parser.add_argument('--resource', default='kernel',
                        help="'kernel' (this process, default), 'local' (background) or SLURM resource (magnifix ...)")
    parser.add_argument('--ncpus', type=int, default=None)
    parser.add_argument('--redo', action='store_true', help='compute again existing results')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.workflow == 'quickdaxm':
        results = run_quickdaxm(cfg, args.resource, args.ncpus, redo=args.redo)
        write_quickdaxm_files(cfg, results)


if __name__ == '__main__':
    main()
