# -*- coding: utf-8 -*-
"""
Module of LaueTools project: indexing of a Laue pattern by a SEQUENCE of steps (several materials, substrate ...)

A sequence (section `sequence:` of the YAML configuration of indexing_batch) is a list of steps, each one
indexing grains of one material on the spots not yet indexed, with its own parameters and decisions:

    sequence:
      strategy: ordered            # ordered | round_robin | best_of
      max_grains: 10               # max total nb of grains (all materials). null: no limit
      max_grains_per_material: 5   # max nb of grains of each material (all steps of a material). null: no limit
      min_gain_indexed_spots: 0    # stop when the indexed spots added are fewer (see strategies)
      best_of_criterion: nb_indexed   # best_of: nb_indexed or matching_rate
      folders_root: null           # .fit files in <folders_root>/fitfiles_<folder> (null: parent of output: fit_folder)
      steps:
        - name: substrate
          key_material: Al2O3
          folder: substrate        # .fit files in fitfiles_substrate (default: key_material)
          nbGrainstoFind: 1        # max nb of grains of this step
          e_max_MR: 19             # keV, matching rate during indexing (default: indexing: e_max_MR)
          e_max: 27                # keV, refinement (default: indexing: e_max)
          dict_indexrefine: {central spots indices: to5, MAXINDEXPROBED: 20}   # changes of indexing: dict_indexrefine
          usepreviousUB: [Al2O3_ref]   # names of ub_matrices (or 3x3 lists) tested first (False: indexing from scratch)
          skipindexing: false      # true: only refine usepreviousUB matrices
          accept: {min_nb_indexed: 15, min_matching_rate: 30, max_pixdev: 0.6}   # a grain is kept if all are fulfilled
          remove_spots: true       # spots of a kept grain are removed for the next grains (false: kept, max 1 grain)
          write_fitfile: true
          next: always             # ordered strategy: always | if_found | if_not_found (probe the next step or stop)
        - name: sample
          key_material: Al
          nbGrainstoFind: 3

Strategies (a grain is 'kept' if found, accepted by `accept` and within the limits of grains):
- ordered: steps one after the other; each step indexes grains (up to its nbGrainstoFind) until a grain is not kept
  or adds fewer than min_gain_indexed_spots spots; then `next` of the step decides to probe the next step or to stop
- round_robin: rounds of one grain of each step (in the order of the steps); a step without kept grain is not probed
  any more; stop when a round (all steps tried) adds fewer than min_gain_indexed_spots spots or no grain
- best_of: at each round, one grain is searched with each step (on the same spots) and only the best one is kept
  (most indexed spots or highest matching rate); stop when the best one adds fewer than min_gain_indexed_spots spots
Indexing stops in any case when max_grains is reached.

Each kept grain is written in <folders_root>/fitfiles_<folder>/<prefix><index>_g<n>.fit with n = 0, 1 ... in each folder.
With remove_spots: true (default), a spot of the .cor file belongs to one grain only (one .fit file).
"""
import copy
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from LaueTools import indexingAnglesLUT as IAL
from LaueTools import dict_LaueTools as DictLT
from LaueTools import generaltools as GT

STRATEGIES = ('ordered', 'round_robin', 'best_of')
NEXT_RULES = ('always', 'if_found', 'if_not_found')
BEST_OF_CRITERIA = ('nb_indexed', 'matching_rate')
ACCEPT_KEYS = ('min_nb_indexed', 'min_matching_rate', 'max_pixdev')
STEP_KEYS = ('name', 'key_material', 'folder', 'nbGrainstoFind', 'e_min', 'e_max', 'e_max_MR', 'dict_indexrefine',
             'MatchingRate_List', 'usepreviousUB', 'skipindexing', 'accept', 'remove_spots', 'write_fitfile', 'next')
SEQUENCE_KEYS = ('strategy', 'max_grains', 'max_grains_per_material', 'min_gain_indexed_spots', 'best_of_criterion',
                 'folders_root', 'steps')


# --------------------------------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------------------------------
def _ub_list(cfg: dict, value):
    """usepreviousUB of a step -> False or list of 3x3 arrays (names of cfg['ub_matrices'] or matrices)"""
    from LaueTools import indexing_batch as IB
    if value in (False, None):
        return False
    if value == 'fromfitfile':
        raise ValueError("usepreviousUB: 'fromfitfile' is not available in a sequence")
    return IB.get_previousUB({'run': {'usepreviousUB': value}, 'ub_matrices': cfg.get('ub_matrices') or {}})


def check_sequence(cfg: dict) -> List[str]:
    """errors of the section `sequence` of a configuration (empty list if correct)"""
    seq = cfg.get('sequence')
    errors = []
    if not isinstance(seq, dict):
        return ['sequence: must be a dict with keys ' + ', '.join(SEQUENCE_KEYS)]
    for key in seq:
        if key not in SEQUENCE_KEYS:
            errors.append(f'sequence: unknown key {key!r} (keys: {", ".join(SEQUENCE_KEYS)})')
    if seq.get('strategy', 'ordered') not in STRATEGIES:
        errors.append(f"sequence: strategy must be one of {STRATEGIES}, not {seq.get('strategy')!r}")
    if seq.get('best_of_criterion', 'nb_indexed') not in BEST_OF_CRITERIA:
        errors.append(f"sequence: best_of_criterion must be one of {BEST_OF_CRITERIA}")
    for key in ('max_grains', 'max_grains_per_material'):
        if seq.get(key) is not None and (not isinstance(seq[key], int) or seq[key] < 1):
            errors.append(f'sequence: {key} must be null or a positive integer, not {seq[key]!r}')
    steps = seq.get('steps') or []
    if not steps:
        errors.append('sequence: steps is empty')
    mymaterials = cfg['material'].get('mymaterials') or {}
    for k, st in enumerate(steps):
        label = f"sequence: step #{k} ({st.get('name', '')})"
        for key in st:
            if key not in STEP_KEYS:
                errors.append(f'{label}: unknown key {key!r} (keys: {", ".join(STEP_KEYS)})')
        mat = st.get('key_material', cfg['material']['key_material'])
        if mat is None or (mat not in mymaterials and mat not in DictLT.dict_Materials):
            errors.append(f'{label}: unknown material {mat!r}')
        if st.get('next', 'always') not in NEXT_RULES:
            errors.append(f"{label}: next must be one of {NEXT_RULES}")
        for key in (st.get('accept') or {}):
            if key not in ACCEPT_KEYS:
                errors.append(f'{label}: unknown accept criterion {key!r} ({", ".join(ACCEPT_KEYS)})')
        nb = st.get('nbGrainstoFind', 1)
        if not isinstance(nb, int) or nb < 1:
            errors.append(f'{label}: nbGrainstoFind must be a positive integer')
        try:
            _ub_list(cfg, st.get('usepreviousUB', False))
        except (KeyError, ValueError) as err:
            errors.append(f'{label}: usepreviousUB: {err}')
        if st.get('skipindexing') and not st.get('usepreviousUB'):
            errors.append(f'{label}: skipindexing needs usepreviousUB matrices')
    return errors


def folders_root(cfg: dict) -> Path:
    """folder containing the fitfiles_<folder> folders of the steps"""
    root = (cfg.get('sequence') or {}).get('folders_root')
    return Path(root) if root else Path(cfg['output']['fit_folder']).parent


def normalize_sequence(cfg: dict) -> dict:
    """sequence of a configuration with all the parameters of each step (defaults from indexing: and material:)"""
    errors = check_sequence(cfg)
    if errors:
        raise ValueError('; '.join(errors))
    seq = cfg['sequence']
    ind = cfg['indexing']
    root = folders_root(cfg)
    steps = []
    for k, st in enumerate(seq['steps']):
        mat = st.get('key_material', cfg['material']['key_material'])
        dir_ = copy.deepcopy(ind['dict_indexrefine'])
        dir_.update(copy.deepcopy(st.get('dict_indexrefine') or {}))
        tolangles = dir_['list matching tol angles']
        mrlist = st.get('MatchingRate_List') or ind['MatchingRate_List']
        if len(mrlist) < len(tolangles):
            mrlist = list(mrlist) + [mrlist[-1]] * (len(tolangles) - len(mrlist))
        folder = str(st.get('folder') or mat)
        accept = {'min_nb_indexed': 0, 'min_matching_rate': 0., 'max_pixdev': None}
        accept.update(st.get('accept') or {})
        steps.append({'name': st.get('name') or f'step{k}_{mat}', 'key_material': mat, 'folder': folder,
                      'folder_path': str(root / f'fitfiles_{folder}'),
                      'nbGrainstoFind': int(st.get('nbGrainstoFind', 1)),
                      'e_min': float(st.get('e_min', ind['e_min'])), 'e_max': float(st.get('e_max', ind['e_max'])),
                      'e_max_MR': float(st.get('e_max_MR', ind['e_max_MR'])),
                      'dict_indexrefine': dir_, 'MatchingRate_List': list(mrlist),
                      'usepreviousUB': _ub_list(cfg, st.get('usepreviousUB', False)),
                      'skipindexing': bool(st.get('skipindexing', False)),
                      'accept': accept, 'remove_spots': bool(st.get('remove_spots', True)),
                      'write_fitfile': bool(st.get('write_fitfile', True)), 'next': st.get('next', 'always')})
    return {'strategy': seq.get('strategy', 'ordered'), 'max_grains': seq.get('max_grains'),
            'max_grains_per_material': seq.get('max_grains_per_material'),
            'min_gain_indexed_spots': int(seq.get('min_gain_indexed_spots') or 0),
            'best_of_criterion': seq.get('best_of_criterion', 'nb_indexed'),
            'folders_root': str(root), 'steps': steps}


def step_folders(sequence: dict) -> Dict[str, Path]:
    """{folder name: path of fitfiles_<folder>} of the steps of a (normalized) sequence"""
    return {st['folder']: Path(st['folder_path']) for st in sequence['steps']}


def folder_config(cfg: dict, folder: str, newfile=None, overwrite: bool = True) -> dict:
    """configuration of the grains of one folder of a sequence (fit_folder = fitfiles_<folder>, its material and
    no sequence), e.g. for the postprocessing notebooks. Written in newfile (default:
    <folder of the configuration>/<configuration name>_<folder>.yaml)"""
    from LaueTools import indexing_batch as IB
    seq = normalize_sequence(cfg)
    steps = [st for st in seq['steps'] if st['folder'] == folder]
    if not steps:
        raise KeyError(f"no step with folder {folder!r}: {sorted(step_folders(seq))}")
    if newfile is None:
        cfgfile = Path(cfg['configfile'])
        newfile = cfgfile.with_name(f'{cfgfile.stem}_{folder}.yaml')
    return IB.save_config(cfg, newfile, overwrite=overwrite, sequence=None, key_material=steps[0]['key_material'],
                          fit_folder=steps[0]['folder_path'], results_stamp=folder)


# --------------------------------------------------------------------------------------
#  INDEXING OF ONE IMAGE
# --------------------------------------------------------------------------------------
_LUTS = {}   # angles LUT per (material, nlutmax), built once per process


def _lut(key_material: str, nlut: int, mymaterials: Optional[dict]):
    key = (key_material, int(nlut))
    if key not in _LUTS:
        if mymaterials is not None and key_material in mymaterials:
            latticeparameters = mymaterials[key_material][1]
        else:
            latticeparameters = DictLT.dict_Materials[key_material][1]
        _LUTS[key] = IAL.build_AnglesLUT_fromlatticeparameters(latticeparameters, int(nlut))
    return _LUTS[key]


def _image_folder(path: str, image_index: int, params) -> Path:
    folder = Path(path)
    if params.fit_nbfiles_per_folder:
        folder = folder / GT.subfolder_name(image_index, params.fit_nbfiles_per_folder, params.subfolder_prefix)
    return folder


def _trial(dataset, step: dict, grain_index: int, params, filecor: str):
    """search one grain with step on a copy of dataset. Returns (copy, candidate dict) or (None, None)"""
    trial = copy.deepcopy(dataset)
    UBs = step['usepreviousUB']
    trial.inhibitindexing = bool(step['skipindexing'] and UBs)
    previousResults = (len(UBs), list(UBs), 0, 0) if UBs else None
    dir_ = step['dict_indexrefine']
    tmpdir = tempfile.mkdtemp(prefix='lt_sequence_')
    try:
        trial.IndexSpotsSet(None, step['key_material'], step['e_min'], step['e_max'], dir_, None,
                            starting_grainindex=grain_index, use_file=0, IMM=False,
                            n_LUT=dir_['nlutmax'], LUT=_lut(step['key_material'], dir_['nlutmax'], params.mymaterials),
                            angletol_list=dir_['list matching tol angles'], nbGrainstoFind=1,
                            previousResults=previousResults, dirnameout_fitfile=tmpdir, corfilename=filecor,
                            verbose=params.verboselevel, MatchingRate_List=step['MatchingRate_List'],
                            depth=params.depth, dictmaterials=params.mymaterials, writefitfile=False,
                            emax_matchringrate=step['e_max_MR'])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if grain_index not in trial.indexedgrains or trial.dict_grain_matrix.get(grain_index) is None:
        return None, None
    spots = [int(i) for i in trial.getSpotsFamily(grain_index)]
    score = trial.dict_grain_matching_rate.get(grain_index) or [len(spots), 0.]
    pixdev = float(np.mean(trial.pixelresidues)) if trial.pixelresidues is not None else float('nan')
    return trial, {'grain_index': grain_index, 'nb_indexed': len(spots), 'matching_rate': float(score[1]),
                   'pixdev': pixdev, 'spot_indices': spots,
                   'UB': np.asarray(trial.dict_grain_matrix[grain_index], dtype=float)}


def _rejection(step: dict, cand: dict) -> str:
    """'' if the grain fulfils the accept criteria of the step, else the reason"""
    acc = step['accept']
    reasons = []
    if cand['nb_indexed'] < (acc.get('min_nb_indexed') or 0):
        reasons.append(f"nb indexed {cand['nb_indexed']} < {acc['min_nb_indexed']}")
    if cand['matching_rate'] < (acc.get('min_matching_rate') or 0):
        reasons.append(f"MR {cand['matching_rate']:.1f} < {acc['min_matching_rate']}")
    if acc.get('max_pixdev') is not None and not cand['pixdev'] <= acc['max_pixdev']:
        reasons.append(f"pixdev {cand['pixdev']:.2f} > {acc['max_pixdev']}")
    return ', '.join(reasons)


def _write_fitfile(trial, cand: dict, folder: Path, local_index: int, filecor: str) -> Path:
    """write the .fit file of the grain of trial as <stem>_g<local_index>.fit in folder"""
    folder.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix='.lt_sequence_', dir=folder))
    try:
        trial.writeFitFile(cand['grain_index'], corfilename=filecor, dirname=str(tmpdir), addpixdev=True,
                           add_strain_sampleframe=True)
        written = tmpdir / f"{filecor.split('.')[0]}_g{cand['grain_index']}.fit"
        lines = written.read_text().splitlines(keepends=True)
        for k in range(len(lines) - 1):   # grain label of the folder
            if lines[k].startswith('#grainIndex') and re.fullmatch(r'#?G_\d+\s*', lines[k + 1]):
                prefix = '#' if lines[k + 1].startswith('#') else ''
                lines[k + 1] = f'{prefix}G_{local_index}\n'
        target = folder / f"{filecor.split('.')[0]}_g{local_index}.fit"
        target.write_text(''.join(lines))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return target


def run_sequence(dataset, params, filecor: str, image_index: int):
    """index the spots of dataset (spotsset with data of filecor) with the sequence params.sequence

    Returns (records, info, trace):
    records = list of kept grains, in the order they were found, each one a dict: step, key_material, folder,
    grain (index in the folder), nb_indexed, matching_rate, pixdev, UB, spot_indices, fitfile (or None)
    trace = list of all the attempts: dict round, step, key_material, decision, nb_indexed, matching_rate, pixdev
    (decision: 'kept', 'not found', 'rejected: <reason>', 'gain too small', 'not the best')
    (None, 'not reanalyzed', []) if the image has results and params.ignorefitfileresults is False
    """
    seq = params.sequence
    steps = seq['steps']
    stem = filecor.split('.')[0]
    folders = {st['folder']: _image_folder(st['folder_path'], image_index, params) for st in steps}
    pattern = re.compile(rf'{re.escape(stem)}_g\d+\.fit')

    if not params.ignorefitfileresults:
        if any(f.exists() and any(pattern.fullmatch(p.name) for p in f.glob(f'{stem}_g*.fit'))
               for f in folders.values()):
            return None, 'not reanalyzed', []
    elif params.writefitfile:   # results of a previous analysis of this image are replaced
        for f in folders.values():
            for old in (f.glob(f'{stem}_g*.fit') if f.exists() else []):
                if pattern.fullmatch(old.name):
                    old.unlink()

    LUT_main, dataset.LUT = dataset.LUT, None   # LUTs are given to IndexSpotsSet (not copied with the trials)
    state = {'dataset': dataset, 'next_grain': params.starting_grainindex, 'records': [], 'trace': [], 'round': 0}
    count_step = {k: 0 for k in range(len(steps))}
    count_material, count_folder = {}, {}
    max_grains, max_per_mat = seq['max_grains'], seq['max_grains_per_material']
    min_gain = seq['min_gain_indexed_spots']

    def total():
        return len(state['records'])

    def quota(k):
        st = steps[k]
        return (count_step[k] < st['nbGrainstoFind']
                and (max_per_mat is None or count_material.get(st['key_material'], 0) < max_per_mat)
                and (max_grains is None or total() < max_grains))

    def note(k, decision, cand=None):
        state['trace'].append({'round': state['round'], 'step': steps[k]['name'],
                               'key_material': steps[k]['key_material'], 'decision': decision,
                               'nb_indexed': None if cand is None else cand['nb_indexed'],
                               'matching_rate': None if cand is None else cand['matching_rate'],
                               'pixdev': None if cand is None else cand['pixdev']})

    def search(k):
        """grain found with step k and accepted by its criteria, or (None, None) (attempt noted in the trace)"""
        trial, cand = _trial(state['dataset'], steps[k], state['next_grain'], params, filecor)
        if cand is None:
            note(k, 'not found')
            return None, None
        reason = _rejection(steps[k], cand)
        if reason:
            note(k, 'rejected: ' + reason, cand)
            return None, None
        return trial, cand

    def keep(k, trial, cand):
        st = steps[k]
        note(k, 'kept', cand)
        local = count_folder.get(st['folder'], 0)
        fitfile = None
        if params.writefitfile and st['write_fitfile']:
            fitfile = _write_fitfile(trial, cand, folders[st['folder']], local, filecor)
        count_folder[st['folder']] = local + 1
        count_step[k] += 1
        count_material[st['key_material']] = count_material.get(st['key_material'], 0) + 1
        state['records'].append({'step': st['name'], 'key_material': st['key_material'], 'folder': st['folder'],
                                 'grain': local, 'nb_indexed': cand['nb_indexed'],
                                 'matching_rate': cand['matching_rate'], 'pixdev': cand['pixdev'],
                                 'UB': cand['UB'], 'spot_indices': cand['spot_indices'],
                                 'fitfile': None if fitfile is None else str(fitfile)})
        state['next_grain'] += 1
        if st['remove_spots']:
            state['dataset'] = trial

    strategy = seq['strategy']
    if strategy == 'ordered':
        for k, st in enumerate(steps):
            found = False
            while quota(k):
                state['round'] += 1
                trial, cand = search(k)
                if cand is None:
                    break
                if cand['nb_indexed'] < min_gain:
                    note(k, 'gain too small', cand)
                    break
                keep(k, trial, cand)
                found = True
                if not st['remove_spots']:
                    break   # spots not removed: the same grain would be found again
            if max_grains is not None and total() >= max_grains:
                break
            if (st['next'] == 'if_found' and not found) or (st['next'] == 'if_not_found' and found):
                break
    elif strategy == 'round_robin':
        active = list(range(len(steps)))
        while active and (max_grains is None or total() < max_grains):
            gain, kept = 0, 0
            state['round'] += 1
            for k in list(active):
                if not quota(k):
                    active.remove(k)
                    continue
                trial, cand = search(k)
                if cand is None:
                    active.remove(k)   # no grain on the spots left: not probed any more
                    continue
                keep(k, trial, cand)
                gain += cand['nb_indexed']
                kept += 1
                if not steps[k]['remove_spots']:
                    active.remove(k)
            if kept == 0 or gain < min_gain:
                break
    elif strategy == 'best_of':
        active = list(range(len(steps)))
        criterion = seq['best_of_criterion']
        while active and (max_grains is None or total() < max_grains):
            candidates = []
            state['round'] += 1
            for k in list(active):
                if not quota(k):
                    active.remove(k)
                    continue
                trial, cand = search(k)
                if cand is None:
                    active.remove(k)
                    continue
                candidates.append((k, trial, cand))
            if not candidates:
                break
            k, trial, cand = max(candidates, key=lambda c: (c[2][criterion], c[2]['nb_indexed']))
            for kk, _, cc in candidates:
                if kk != k:
                    note(kk, 'not the best', cc)
            if cand['nb_indexed'] < min_gain:
                note(k, 'gain too small', cand)
                break
            keep(k, trial, cand)
            if not steps[k]['remove_spots']:
                active.remove(k)
    else:
        raise ValueError(f'unknown strategy {strategy!r}')

    dataset.LUT = LUT_main
    counts = {}
    for r in state['records']:
        counts[f"{r['step']}({r['key_material']})"] = counts.get(f"{r['step']}({r['key_material']})", 0) + 1
    info = f"sequence {strategy}: " + (', '.join(f'{n} grain(s) {s}' for s, n in counts.items()) or 'no grain')
    return state['records'], info, state['trace']


# --------------------------------------------------------------------------------------
#  RESULTS
# --------------------------------------------------------------------------------------
def trace_of(result) -> list:
    """attempts (trace) of a result of index_refine() with a sequence ([] otherwise)"""
    return result[5] if len(result) > 5 and isinstance(result[5], list) and \
        (not result[5] or isinstance(result[5][0], dict)) else []


def records_of(result) -> list:
    """kept grains (records) of a result of index_refine() with a sequence ([] otherwise)"""
    return result[4] if len(result) > 4 and isinstance(result[4], list) and \
        (not result[4] or isinstance(result[4][0], dict)) else []


def summarize_sequence(allresults: list, mapdims, folders: Optional[List[str]] = None) -> dict:
    """2D maps (shape (slow, fast)) per folder of a sequence: nb of grains, total nb of indexed spots, and
    nb of indexed spots and matching rate of grain 0 of the folder; plus 'nb_grains_total' (all folders)

    Returns {folder: {'nb_grains': 2D, 'nb_indexed_total': 2D, 'nb_indexed_g0': 2D, 'MR_g0': 2D}, 'nb_grains_total': 2D}
    """
    nfast, nslow = mapdims
    shape = (nslow, nfast)
    allrecords = [(res[0], records_of(res)) for res in allresults]
    if folders is None:
        folders = sorted({r['folder'] for _, recs in allrecords for r in recs})
    out = {f: {'nb_grains': np.zeros(shape), 'nb_indexed_total': np.zeros(shape),
               'nb_indexed_g0': np.full(shape, np.nan), 'MR_g0': np.full(shape, np.nan)} for f in folders}
    out['nb_grains_total'] = np.zeros(shape)
    for ix, recs in allrecords:
        if not 0 <= ix < nfast * nslow:
            continue
        row, col = ix // nfast, ix % nfast
        out['nb_grains_total'][row, col] = len(recs)
        for r in recs:
            if r['folder'] not in out:
                continue
            m = out[r['folder']]
            m['nb_grains'][row, col] += 1
            m['nb_indexed_total'][row, col] += r['nb_indexed']
            if r['grain'] == 0:
                m['nb_indexed_g0'][row, col] = r['nb_indexed']
                m['MR_g0'][row, col] = r['matching_rate']
    return out


def print_records(result, trace: bool = True):
    """print the kept grains of a result of index_refine() with a sequence (and all the attempts if trace)"""
    print(f'image {result[0]}: {result[3]}')
    if trace and trace_of(result):
        print('   attempts:')
        for t in trace_of(result):
            score = '' if t['nb_indexed'] is None else \
                f"  ({t['nb_indexed']} spots, MR {t['matching_rate']:.1f} %, pixdev {t['pixdev']:.2f})"
            print(f"     round {t['round']:2d}  {t['step']:<12s} {t['key_material']:<8s} {t['decision']}{score}")
        print('   kept grains:')
    recs = records_of(result)
    if not recs:
        GT.printred('   no grain kept')
    for n, r in enumerate(recs):
        print(f"   #{n} step {r['step']:<12s} {r['key_material']:<8s} -> fitfiles_{r['folder']}/..._g{r['grain']}.fit: "
              f"{r['nb_indexed']:4d} spots, MR {r['matching_rate']:5.1f} %, pixdev {r['pixdev']:.2f}")


def phase_map(allresults: list, mapdims, folders: Optional[List[str]] = None):
    """2D map (shape (slow, fast)) of the folder (material) of the grain with most indexed spots in each image
    (among `folders`, default all), as an index in `labels` (-1: no grain). Returns (map, labels)"""
    nfast, nslow = mapdims
    allrecords = [(res[0], records_of(res)) for res in allresults]
    labels = folders or sorted({r['folder'] for _, recs in allrecords for r in recs})
    phases = np.full((nslow, nfast), -1, dtype=int)
    for ix, recs in allrecords:
        recs = [r for r in recs if r['folder'] in labels]
        if recs and 0 <= ix < nfast * nslow:
            best = max(recs, key=lambda r: r['nb_indexed'])
            phases[ix // nfast, ix % nfast] = labels.index(best['folder'])
    return phases, labels


def misorientation_map(allresults: list, mapdims, folder: str, UB_ref) -> np.ndarray:
    """2D map of the misorientation angle (deg, no symmetry) between grain 0 of `folder` and UB_ref
    (e.g. drift of the orientation of a substrate)"""
    nfast, nslow = mapdims
    out = np.full((nslow, nfast), np.nan)
    for res in allresults:
        ix = res[0]
        for r in records_of(res):
            if r['folder'] == folder and r['grain'] == 0 and 0 <= ix < nfast * nslow:
                R = np.asarray(r['UB'], dtype=float) @ np.linalg.inv(np.asarray(UB_ref, dtype=float))
                cosine = np.clip(0.5 * (np.trace(R) - 1), -1., 1.)   # (UB with strain: not exactly a rotation)
                out[ix // nfast, ix % nfast] = np.degrees(np.arccos(cosine))
    return out


def plot_records(result, corfile, ax=None, title: Optional[str] = None):
    """experimental spots of corfile (grey) and spots of the kept grains of a sequence result (one color per grain)"""
    import matplotlib.pyplot as plt
    from LaueTools import IOLaueTools as IOLT
    data = np.atleast_2d(IOLT.readfile_cor(str(corfile))[0])
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data[:, 2], data[:, 3], s=8, c='0.6', label=f'{len(data)} spots')
    for n, r in enumerate(records_of(result)):
        xy = data[r['spot_indices'], 2:4]
        ax.scatter(xy[:, 0], xy[:, 1], s=40, facecolors='none', edgecolors=f'C{n % 10}',
                   label=f"#{n} {r['step']} ({r['key_material']}) g{r['grain']}: {r['nb_indexed']} spots")
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(title or Path(str(corfile)).name, fontsize=9)
    return ax
