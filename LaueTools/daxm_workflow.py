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

    python -m LaueTools.daxm_workflow quickdaxm configs/myscan.yaml --resource magnifix --ncpus 96
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from LaueTools import generaltools as GT
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
    'sample': {'tilt': 40.},
    'output': {'folder': None},
    'quickdaxm': {**QD.DEFAULT_PARAMS, 'slice_step': 5., 'slice_thickness': None, 'slice_min_spots': 6},
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
    files = sorted(glob.glob(str(image_folder / '*_[0-9]*.*')))
    if not files:
        raise FileNotFoundError(f'no image in {image_folder}')
    match = re.match(r'^(.*?)(\d+)(\.\w+)$', os.path.basename(files[0]))
    prefix, digits, suffix = match.groups()
    stacked = suffix == '.h5' and len(files) < len(positions)
    if CCDLabel is None:
        CCDLabel = ('EIGER_4MCdTestack' if stacked else 'EIGER_4MCdTe') if suffix == '.h5' else 'sCMOS'
    nbframes = int(np.ceil(len(positions) / len(files))) if stacked else 1
    return {'image_folder': str(image_folder), 'prefix': prefix, 'suffix': suffix, 'nbdigits': len(digits),
            'first_image_index': 0 if stacked else int(digits), 'nb_images': len(positions),
            'CCDLabel': CCDLabel, 'nbframes_per_file': nbframes, 'hdf5_file': str(hdf5_file),
            'scan_number': int(scan_number), 'wire_motor': motor, 'wire_start': float(positions[0]),
            'wire_stop': float(positions[-1]), 'name': f'{hdf5_file.stem}_{int(scan_number)}'}


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


def compute_quickdaxm(cfg: dict, nb_cpus: Optional[int] = None, progress_callback=None) -> Dict[str, Any]:
    """quick daxm analysis of the scan of the configuration, in this process (see quickdaxm.run())"""
    params = {key: value for key, value in cfg['quickdaxm'].items() if key in QD.DEFAULT_PARAMS}
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
        GT.printgreen(f'quick daxm already done: results reloaded from {output}\n(redo=True to compute again)')
        return QD.load_results(output)
    if not check_config(cfg, 'quickdaxm'):
        raise RuntimeError(f"fix the errors of the configuration file {cfg.get('configfile')}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if resource == 'kernel':
        results = compute_quickdaxm(cfg, nb_cpus)
        QD.save_results(results, output)
        GT.printgreen(f'quick daxm results saved in {output}')
        return results
    import LaueTools.scripts.workflowlauncher as WL
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
    corfile = QD.write_cor(results, Path(files['quickdaxm_cor']).stem, os.path.dirname(files['quickdaxm_cor']))
    prefix = re.sub(r'[^\w.-]+', '_', str(cfg['name'])).strip('_') or 'daxm'
    slices = QD.write_depth_slices(results, prefix, files['quickdaxm_slices'], step=params['slice_step'],
                                   thickness=params['slice_thickness'], min_spots=params['slice_min_spots'])
    with open(os.path.join(files['quickdaxm_slices'], 'slices.yaml'), 'w') as f:
        yaml.safe_dump({'prefix': prefix, 'slice_step': params['slice_step'], 'slices': slices}, f, sort_keys=False)
    if verbose:
        written = [s for s in slices if s['file']]
        print(f'{corfile}: {len(QD.select_spots(results))} spots with columns ybeam, depth')
        print(f"{files['quickdaxm_slices']}: {len(written)} .cor files of slices"
              + (f" (ybeam {written[0]['ybeam']:g} to {written[-1]['ybeam']:g} µm)" if written else ''))
    return {'cor': corfile, 'slices': slices}


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
