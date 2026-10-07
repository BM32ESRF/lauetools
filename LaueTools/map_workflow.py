# -*- coding: utf-8 -*-
"""
Module of LaueTools project: complete analysis of 2D maps of Laue patterns, from the list of scans of a BLISS
experiment to orientation and grain maps, with few settings (notebooks/quickstart/laue_maps_quickstart.ipynb).

Steps (each one uses the dedicated modules, with YAML configuration files written in PROCESSED_DATA):

1. list the maps of the experiment (BLISS hdf5 file)         list_maps(), ScanSelector, scan_info()
2. configuration files of peak search and indexation           make_configs()
3. peak search (images -> .dat, .cor files)                    run_step('peaksearch', ...)   (peaksearch_batch)
4. indexation & refinement (.cor -> .fit files)                run_step('indexing', ...)     (indexing_batch)
5. maps: quality, orientation (IPF), grains (segmentation)     load_fitfiles(), quality_maps(), segmentation()

Each step runs on this machine (multiprocessing) or as a job on the ESRF cluster (SLURM, e.g. magnifix partition).
Steps already done are skipped (results are reloaded), unless redo=True: running the notebook again is safe.
Results are written next to the images, with RAW_DATA replaced by PROCESSED_DATA:
    <PROCESSED_DATA>/<sample>/<dataset>/scan0001/configs/   peaksearch.yaml, indexing.yaml
                                                 /corfiles/  .dat, .cor files
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


def _status(workdir: Path) -> str:
    """results already in workdir: 'peaks' (corfiles), 'fit:<material>' (fitfiles_<material>)"""
    status = []
    if (workdir / 'corfiles').is_dir():
        status.append('peaks')
    if workdir.is_dir():
        status += [f"fit:{p.name[len('fitfiles_'):]}" for p in sorted(workdir.glob('fitfiles_*'))
                   if p.is_dir() and '_previous_' not in p.name]
    return ', '.join(status)


def list_maps(logfile: Union[str, Path], contains: Optional[str] = None, only_success: bool = False,
              output_root: Optional[Union[str, Path]] = None, count_images: bool = True, verbose: int = 1):
    """table (pandas DataFrame) of the 2D maps (amesh, dmesh, fscan2d) of the experiment

    contains: keep the maps whose dataset name contains this text (case insensitive)
    count_images: count the image files of each map (column 'images': found / expected)
    Column 'done': results already in PROCESSED_DATA (peaks: peak search, fit:<material>: indexation)
    index of the table: scan index in the hdf5 file (used by scan_info())
    """
    import pandas as pd
    from LaueTools import logfile_reader as iohdf5
    t0 = time.time()
    lf = iohdf5.H5file(str(logfile))
    lf.getscans(verbose=0)
    df = lf.dfallscans
    df = df[df['scantype'].isin(MAP_COMMANDS)]
    if only_success:
        df = df[df['endreason'] == 'SUCCESS']
    if contains:
        df = df[df['sample_dataset_scanindex'].str.contains(contains, case=False, regex=False)]

    rows = []
    for idx, row in df.iterrows():
        try:
            dc = iohdf5.read_fullcommand(row['fullcommand'])
        except ValueError:
            continue
        nfast, nslow = dc['npts_fast'], dc['npts_slow']
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
    if verbose:
        print(f'{len(maps)} maps found in {Path(logfile).name} ({time.time() - t0:.0f} s)')
    return maps


def show_maps(maps, nmax: Optional[int] = None):
    """display the table of maps (main columns)"""
    from IPython.display import display
    import pandas as pd
    with pd.option_context('display.max_rows', nmax or len(maps), 'display.max_colwidth', 70, 'display.width', 250):
        display(maps[['dataset', 'start', 'hours', 'map', 'images', 'detector', 'end', 'done', 'command']])


class ScanSelector:
    """menu to choose a map in the table of list_maps(). The selected scan index is ScanSelector.index"""

    def __init__(self, maps, index: Optional[int] = None):
        self.maps = maps
        self.index = index if index is not None else (int(maps.index[-1]) if len(maps) else None)
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print('ipywidgets not installed: set SCAN_INDEX in SETTINGS (index of the table)')
            return
        options = [(f"{i:5d} | {r['dataset']} | {r['map']} | images {r['images']} | {r['end']}"
                    + (f" | done: {r['done']}" if r['done'] else ''), int(i)) for i, r in maps.iterrows()]
        self.menu = widgets.Dropdown(options=options, value=self.index, description='map:',
                                     layout=widgets.Layout(width='95%'))
        self.info = widgets.HTML()
        self.menu.observe(self._on_change, names='value')
        self._on_change(None)
        display(widgets.VBox([self.menu, self.info]))

    def _on_change(self, change):
        self.index = int(self.menu.value)
        r = self.maps.loc[self.index]
        self.info.value = (f"<b>{r['dataset']}</b>: <code>{r['command']}</code><br>{r['image_folder']}"
                           f"<br>then run the next cells")


def scan_info(maps, index: int, output_root: Optional[Union[str, Path]] = None) -> dict:
    """all information of a map needed by the analysis (from the table of list_maps())"""
    from LaueTools import logfile_reader as iohdf5
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


@functools.lru_cache(maxsize=8)
def _experiment_files(raw_data_folder: str, max_depth: int = 4) -> tuple:
    """.det files of the experiment (RAW_DATA and PROCESSED_DATA, up to max_depth subfolders)"""
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
    return tuple(found)


def _most_recent_first(files) -> List[Path]:
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def find_detfiles(raw_data_folder: Union[str, Path], max_depth: int = 4) -> List[Path]:
    """calibration .det files of the experiment (RAW_DATA and PROCESSED_DATA), most recent first"""
    return _most_recent_first(f for f in _experiment_files(str(raw_data_folder), max_depth) if f.suffix == '.det')


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


def _fingerprint(step: str, cfg: dict) -> str:
    """hash of the parameters of a configuration that change the results of a step (not names, plots, cpus ...)"""
    import hashlib
    scan, out = cfg['scan'], cfg['output']
    if step == 'peaksearch':
        keys = {'scan': {k: scan[k] for k in ('image_folder', 'prefix', 'suffix', 'nbdigits', 'CCDLabel', 'detfile')},
                'output': {k: out[k] for k in ('dat_folder', 'added_string', 'write_cor', 'nbfiles_per_folder',
                                               'subfolder_prefix')},
                'peaksearch': cfg['peaksearch'], 'exclusion_boxes': cfg['exclusion_boxes'],
                'selection': cfg['selection']}
    else:
        peaks = sorted(Path(scan['cor_folder']).glob('peaksearch_*.pickle')) if scan['cor_folder'] else []
        keys = {'scan': {k: scan[k] for k in ('cor_folder', 'prefix', 'nbdigits', 'CCDLabel', 'nbfiles_per_folder',
                                              'subfolder_prefix')},
                'output': {k: out[k] for k in ('fit_folder', 'fit_subfolders')},
                'material': cfg['material'], 'indexing': cfg['indexing'], 'selection': cfg['selection'],
                'run': {k: cfg['run'][k] for k in ('usepreviousUB', 'skipindexing', 'starting_grainindex')},
                'ub_matrices': cfg['ub_matrices'],
                'peaksearch_results': [p.stat().st_mtime for p in peaks]}   # new peak search -> new indexation
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
    return Path(cfg['configfile']).parent / f'{step}_slurm_job.json'


def _slurm_states(job_id: str) -> List[str]:
    """states of the jobs (job array) in the queue, [] if finished (or squeue not available)"""
    try:
        out = IB._run(['squeue', '-h', '-j', str(job_id), '-o', '%T']).stdout
    except (FileNotFoundError, OSError):
        return []
    return [s.strip() for s in out.splitlines() if s.strip()]


def wait_slurm_job(job: Union[dict, str, Path], poll: float = 30, verbose: bool = True) -> bool:
    """wait for the end of a SLURM job (status updated every poll seconds). True if all its results are written

    Interrupting the wait (kernel interrupt button) does not stop the job: run the cell again to wait again.
    """
    from IPython.display import display, HTML
    job = IB.load_slurm_job(job)
    handle = display(HTML(f"<pre>job {job.get('job_id')}: submitted</pre>"), display_id=True) if verbose else None
    t0 = time.time()
    try:
        while True:
            nparts = len(list(Path(job['jobdir']).glob('allresults_part*.pickle')))
            states = _slurm_states(job['job_id']) if job.get('job_id') else []
            progress = ''
            for logfile in sorted((Path(job['jobdir']) / 'logs').glob('*.out')):
                found = re.findall(r'(\d+)/(\d+) \[', logfile.read_text(errors='replace'))
                if found:
                    progress += f' {found[-1][0]}/{found[-1][1]}'
            text = (f"job {job.get('job_id')} on {job['machine']}: {', '.join(sorted(set(states))) or 'finished'}"
                    f" | results {nparts}/{job['nchunks']} | images done:{progress or ' -'}"
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


def run_step(step: str, cfg: dict, execution: str = 'local', nb_cpus: Optional[int] = None,
             machine: str = 'auto', time_limit: Optional[str] = None, mem_per_cpu: str = '2000M',
             nchunks: int = 1, redo: bool = False, show_only: bool = False):
    """run 'peaksearch' (images -> .cor) or 'indexing' (.cor -> .fit) on all images of a map

    Already done with the same parameters (results file and its fingerprint): results are reloaded, nothing is
    computed, unless redo=True. Done with other parameters (configuration file changed): computed again.
    Before computing again, the folder of the previous results is renamed <folder>_previous_<date> (not deleted).
    show_only=True: never compute, only reload the results (error if there are none)
    execution 'local': on this machine with nb_cpus processes (None: all available cpus)
    execution 'slurm': job on the cluster, machine 'auto' or key of IB.SLURM_MACHINES ('magnifix', 'hpc6' ...),
        nb_cpus per job (None: 192 on magnifix, 96 on other machines), time_limit 'hh:mm:ss' (None: estimated).
        The notebook waits for the end of the job; if the wait is interrupted, running the cell again waits for the
        same job (it is not submitted again).
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
    if not pointer.exists() and (redo or resfile.exists() or (outfolder.is_dir() and any(outfolder.iterdir()))):
        _archive_previous(outfolder)   # (not while waiting for a job submitted before: it writes there)

    job = None
    if execution == 'slurm' and pointer.exists():   # job submitted before (e.g. wait interrupted): wait again
        job = IB.load_slurm_job(json.loads(pointer.read_text())['jobdir'])
        if job.get('job_id') and (_slurm_states(job['job_id'])
                                  or len(list(Path(job['jobdir']).glob('allresults_part*.pickle'))) >= job['nchunks']):
            print(f"job {job['job_id']} submitted before: waiting for it (not submitted again)")
        else:
            job = None
    if execution == 'slurm' and job is None:
        if machine == 'auto':
            machine, nb_cpus = IB.choose_slurm_machine(nb_cpus or 192)
        else:
            nb_cpus = nb_cpus or IB.SLURM_MACHINES[machine]['cpus']
        nbjobs = max(nchunks, IB.SLURM_MACHINES[machine].get('nodes', 1))
        if time_limit is None:
            grains = cfg['indexing']['nbGrainstoFind'] if step == 'indexing' else 1
            time_limit = IB.walltime_estimate(len(listfiles) / nbjobs, nb_cpus, SEC_CPU_PER_IMAGE[step] * grains,
                                              safety=2., minimum_minutes=20)
        job = module.prepare_slurm_job(cfg, params, listfiles, machine=machine, nb_cpus=nb_cpus, time=time_limit,
                                       mem_per_cpu=mem_per_cpu, nchunks=nchunks,
                                       job_name=f"{step}_{Path(cfg['configfile']).parent.parent.parent.name}"[:60])
        if IB.submit_slurm_job(job) is None:
            raise RuntimeError('the job could not be submitted (see above)')
        pointer.write_text(json.dumps({'jobdir': job['jobdir']}))

    if execution == 'slurm':
        if not wait_slurm_job(job):
            return None
        allresults = IB.load_slurm_results(job)
        nb_cpus = job['nb_cpus']
    else:
        allresults = module.run_multiprocessing(listfiles, params, nb_cpus=nb_cpus)

    mapdims = cfg['scan']['mapdims']
    if step == 'peaksearch':
        PB.save_results(cfg, params, allresults, PB.summarize_results(allresults, mapdims), filepath=resfile,
                        overwrite=True, nb_cpus=nb_cpus)
    else:
        IB.save_allresults(cfg, params, allresults, IB.summarize_results(allresults, mapdims), filepath=resfile,
                           overwrite=True, nb_cpus=nb_cpus)
    _fingerprint_file(resfile).write_text(fingerprint)
    if pointer.exists():
        pointer.unlink()
    GT.printgreen(f'{step} done')
    return allresults


# --------------------------------------------------------------------------------------
#  RESULTS: maps
# --------------------------------------------------------------------------------------
def load_fitfiles(icfg: dict, grainindex: int = 0):
    """parsed_fitfileseries of the .fit files of grain grainindex (UB, strain, nb of spots ... of all map points)

    read in this process (~1 ms per file, faster than a pool for usual maps) and silently: the reader prints a
    message for each line of the .fit files that it does not use
    """
    import contextlib
    import io
    from LaueTools.fitfilereader import parsed_fitfileseries
    scan = icfg['scan']
    layout = IB.fitfiles_layout(icfg)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ffs = parsed_fitfileseries(folderpath=str(icfg['output']['fit_folder']), nb_cols=scan['mapdims'][0],
                                   nb_rows=scan['mapdims'][1], prefix=scan['prefix'], suffix=f'_g{grainindex}.fit',
                                   use_multiprocessing=False, nbdigits=scan['nbdigits'],
                                   nbfiles_per_folder=layout['fit_nbfiles_per_folder'],
                                   subfolder_prefix=layout['subfolder_prefix'])
    nbfound = int(np.sum(np.isfinite(ffs.UB[:, 0, 0])))
    print(f"{nbfound} .fit files of grain {grainindex} read / {scan['mapdims'][0] * scan['mapdims'][1]} map points "
          f"({time.time() - t0:.0f} s)")
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


# --------------------------------------------------------------------------------------
#  all grains of the .fit files
# --------------------------------------------------------------------------------------
def load_grains(icfg: dict, nbgrains: Optional[int] = None) -> dict:
    """.fit files of all grains (_g0.fit, _g1.fit, ...) of the map, and at each point the 'best' grain: the one with
    the most indexed spots (g0 is the first grain found, not always the best one)

    Returns dict: 'ffs' (list of parsed_fitfileseries), 'nbspots' (nbgrains, nbpoints), 'pixdev' (nbgrains, nbpoints),
    'best' (grain index of the best grain, -1 if none), 'nbgrains_found', 'best_nbspots', 'best_pixdev',
    'best_UB' (nbpoints, 3, 3) (NaN where no grain)
    """
    nbgrains = nbgrains or icfg['indexing']['nbGrainstoFind']
    ffs = [load_fitfiles(icfg, g) for g in range(nbgrains)]
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
    laue_group = post['laue_group']
    if laue_group == 'auto':
        laue_group = IPF.laue_group_of_material(icfg['material']['key_material'], icfg['material']['mymaterials'])
    UB = grains['best_UB']
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    for ax, axis in zip(axes, ('x', 'y', 'z')):
        rgb = IPF.ipf_rgb_map(UB, scan['invmapdims'], laue_group, axis, sample_tilt, mask)
        rgba = np.ones(rgb.shape[:2] + (4,))
        rgba[..., :3] = np.nan_to_num(rgb)
        rgba[..., 3] = np.all(np.isfinite(rgb), axis=2)   # masked and missing points: transparent (black background)
        ax.set_facecolor('black')
        plot_map(rgba, icfg, f'IPF-{axis.upper()} (sample frame), best grain', ax=ax)
    directions = IPF.sample_directions_in_crystal(UB, 'z', sample_tilt)
    if mask is not None:
        directions = directions[~np.asarray(mask, dtype=bool).ravel()]
    IPF.plot_ipf_key(laue_group, ax=axes[-1], directions=directions, title=f'{laue_group}  (points: IPF-Z)')
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
    Returns {scan index: indexing configuration} of the processed maps. Maps already done are skipped."""
    refpaths = config_paths(reference)
    done = {}
    for k, index in enumerate(indices):
        print(f'\n========== map {k + 1}/{len(indices)}: scan {index} ==========')
        try:
            info = scan_info(maps, index, output_root)
            print(info['name'], '|', info['command'])
            pcfg, icfg = make_configs(info, key_material, detfile, peaksearch_template=refpaths['peaksearch'],
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
