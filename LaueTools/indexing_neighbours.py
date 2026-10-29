"""
Indexing of a map guided by the UB matrices already found (usepreviousUB: neighbours in the indexing YAML file)

Indexing from scratch (angular LUT) takes minutes per image for large cells, whereas checking and refining a UB
matrix that already indexes a neighbouring point of the map takes seconds. Each image is then indexed by:
    1. checking candidate UB matrices: UB of the previous image of the same process, UBs of the map points already
       indexed within `radius` (nearest first), the library of the orientations found from scratch in the map, then
       the reference UB matrices `reference_UBs` (if any).
       Check of a candidate:
       a. raw candidate (not refined): spots matched within `check_angle_tol` (deg) up to e_max_MR; a matching rate
          above `check_min_matching_rate` (%) is needed to go on (else next candidate, in ~0.1 s)
       b. refinement, with the same steps as the indexing from scratch ('list matching tol angles': first step with
          the spots linked up to e_max_MR, next steps up to e_max; or refinement_steps): all the .fit files of the
          map are refined the same way
       c. accepted if the last refinement step (smallest tolerance angle) gives at least `min_nb_matches` indexed
          spots and a matching rate of at least `min_matching_rate` %
    2. indexing from scratch if no candidate is accepted (new orientation: added to the library)

Order of the map points:
    - seed pass: points of a regular grid of the map (`seed_grid` rectangular or hexagonal, spacing `seed_step` map
      points or `seed_spacing` micron in real space), indexed from scratch in parallel (after checking the reference
      UB matrices). Rectangular grid with seed_step = 2 * radius: every map point has a seed within `radius`
    - main pass: the other points, by tiles of the map (several tiles per process for load balancing); in a tile,
      points are taken by increasing distance to the nearest seed, so that a neighbour is already indexed.

UB store (shared by the processes of this machine and by the jobs of a SLURM array on several nodes, e.g.
magnifix3): files in <fit_folder>/ub_store/ (no shared memory between nodes, no database server)
    <image index>.npy           grains of the image: rows [UB (9 values), nb of indexed spots, matching rate,
                                how the image was indexed (index in SOURCES)] (no row: image treated, nothing found)
    library/<image index>.npy   the same, for images indexed from scratch (orientations of the library)
Files are written under a temporary name then renamed (a reader never sees a partly written file).

Configuration (indexing YAML file):
    run:
      usepreviousUB: neighbours
      neighbours:
        radius: 10            # map points (square neighbourhood) where indexed UB matrices are taken
        min_nb_matches: 100   # nb of indexed spots to accept a candidate UB matrix
        min_matching_rate: 0  # and matching rate (%) to accept it (0: not used)
        seed_step: 20         # seed pass: every seed_step points (0: no seed pass)
        seed_grid: rectangular  # or hexagonal (most compact: ~23 % less seeds for the same max distance to a seed),
                                # or none: only seed_images
        seed_images: []       # image indices of seeds added to those of the grid (e.g. chosen grains, regions)
        seed_spacing: null    # micron: grid spacing in real space (map step sizes), replaces seed_step
        seeds_file: null      # validated seeds (indexed beforehand, checked by the user): no seed pass
        library_size: 20      # max nb of orientations of the library tested
        max_candidates: 8     # max nb of distinct UB matrices of neighbours tested
        dispatch: front       # front: points next to accepted points first, deferred indexing from scratch; tiles
        front_attempts: 2     # front: nb of UB guess attempts of a point before indexing from scratch (at the end)
        tile_size: null       # tiles: [fast, slow] nb of points of the tiles (null: automatic)
        reference_UBs: []     # names of ub_matrices (or 3x3 lists) checked by all images (seeds included)
        check_angle_tol: null # deg, matching of a raw candidate UB (null: MATCHINGRATE_ANGLE_TOL of dict_indexrefine)
        check_min_matching_rate: null  # %, raw candidate UB (null: MinimumMatchingRate of dict_indexrefine)
"""
import copy
import dataclasses
import os
import shutil
import time
from pathlib import Path
from contextlib import nullcontext as _nullcontext
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from LaueTools import generaltools as GT

DEFAULT_NEIGHBOURS = {
    'radius': 2,
    'min_nb_matches': 30,
    'min_matching_rate': 0.,   # %, also needed to accept a candidate (0: not used)
    'seed_step': 10,
    'seed_grid': 'rectangular',   # 'rectangular' or 'hexagonal' (most compact) grid of the seeds, 'none': no grid
    'seed_images': [],        # image indices of seeds added to those of the grid (seed_grid 'none': the only seeds)
    'seed_spacing': None,     # micron: spacing of the seeds grid in real space (map step sizes), replaces seed_step
    # file of the validated seeds (seeds indexed from scratch beforehand, see seeds_config(), seed_table(),
    # save_validated_seeds()): their results are used and the seed pass is skipped. None: seed pass
    'seeds_file': None,
    'library_size': 20,
    'max_candidates': 8,
    'tile_size': None,
    # 'front': one task per point, points next to the most accepted points first (front from the seeds), points without
    # accepted UB guess deferred (retried when a neighbour is accepted, front_attempts times), indexed from scratch at
    # the end only. 'tiles': tiles of the map (several per cpu), points of a tile by increasing distance to the seeds
    'dispatch': 'front',
    'front_attempts': 2,
    'library_refresh': 60.,   # s, period of reading the library written by the other processes
    'reference_UBs': [],      # names of ub_matrices (or 3x3 lists) checked by all images (seeds included)
    # first check of a raw (not refined) candidate UB matrix: tolerance angle (deg) of the spots matching (up to
    # e_max_MR) and minimum matching rate (%) to go on with the refinement. None: MATCHINGRATE_ANGLE_TOL and
    # MinimumMatchingRate of dict_indexrefine (as for the indexing from scratch)
    'check_angle_tol': None,
    'check_min_matching_rate': None,
}
SAME_ORIENTATION_DEG = 0.5   # candidates closer than this (raw angle, no symmetry) are tested once
DEFERRED = 'deferred: no UB guess accepted'   # info of a result of index_image(defer=True) to try again later


def settings(neighbours: Optional[dict]) -> dict:
    """neighbours parameters completed with the default values"""
    return {**DEFAULT_NEIGHBOURS, **(neighbours or {})}


def check_params(params):
    """IndexRefineParams of the check of candidate UB matrices: check_angle_tol and check_min_matching_rate of
    params.neighbours in dict_indexrefine (MATCHINGRATE_ANGLE_TOL and MinimumMatchingRate)"""
    nb = settings(params.neighbours)
    dir_ = copy.deepcopy(params.dict_indexrefine)
    if nb['check_angle_tol'] is not None:
        dir_['MATCHINGRATE_ANGLE_TOL'] = nb['check_angle_tol']
    if nb['check_min_matching_rate'] is not None:
        dir_['MinimumMatchingRate'] = nb['check_min_matching_rate']
    return dataclasses.replace(params, dict_indexrefine=dir_)


def check_description(params) -> str:
    """text of the tolerances and thresholds of the check of candidate UB matrices"""
    nb, dir_ = settings(params.neighbours), check_params(params).dict_indexrefine
    if params.refinement_steps:
        from LaueTools import indexingSpotsSet as ISS
        steps = ISS.parse_refinement_steps(params.refinement_steps)
        refine = 'refinement steps ' + ', '.join(f'{t:g} deg ({e:g} keV)' for e, t, _, _ in steps)
        last = f'{steps[-1][1]:g} deg, {steps[-1][0]:g} keV'
    else:
        tols = dir_['list matching tol angles']
        energies = [params.e_max_MR if (k == 0 and len(tols) > 1 and params.e_max_MR) else params.e_max
                    for k in range(len(tols))]
        refine = 'refinement steps ' + ', '.join(f'{t:g} deg ({e:g} keV)' for t, e in zip(tols, energies))
        last = f'{tols[-1]:g} deg, {params.e_max:g} keV'
    return (f"candidate UB: raw matching within {dir_['MATCHINGRATE_ANGLE_TOL']:g} deg up to {params.e_max_MR:g} keV, "
            f"matching rate > {dir_['MinimumMatchingRate']:g}% to refine; {refine}; accepted with >= "
            f"{nb['min_nb_matches']} indexed spots and matching rate >= {nb['min_matching_rate']:g}% ({last})")


def reference_matrices(neighbours: Optional[dict], ub_matrices: Optional[dict]) -> list:
    """[(name, UB 3x3 array)] of reference_UBs (names of ub_matrices, or 3x3 lists)"""
    out = []
    for k, item in enumerate(settings(neighbours)['reference_UBs'] or []):
        if isinstance(item, str):
            if item not in (ub_matrices or {}):
                raise KeyError(f"reference UB matrix '{item}' not found in ub_matrices")
            name, item = item, ub_matrices[item]
        else:
            name = f'reference_UBs[{k}]'
        mat = np.array(item, dtype=float)
        if mat.shape != (3, 3):
            raise ValueError(f'reference UB matrix {name} must be 3x3, got shape {mat.shape}')
        out.append((name, mat))
    return out


def check_settings(neighbours: Optional[dict], ub_matrices: Optional[dict] = None) -> List[str]:
    """errors of the neighbours parameters"""
    errors = []
    nb = settings(neighbours)
    try:
        reference_matrices(neighbours, ub_matrices)
    except (KeyError, ValueError, TypeError) as err:
        errors.append(f'run: neighbours: reference_UBs: {err}')
    unknown = set(neighbours or {}) - set(DEFAULT_NEIGHBOURS)
    if unknown:
        errors.append(f'run: neighbours: unknown parameter(s) {sorted(unknown)} (known: {list(DEFAULT_NEIGHBOURS)})')
    for key in ('radius', 'min_nb_matches', 'max_candidates'):
        if not isinstance(nb[key], int) or nb[key] < 1:
            errors.append(f'run: neighbours: {key} must be a positive integer, not {nb[key]}')
    if not isinstance(nb['min_matching_rate'], (int, float)) or not 0 <= nb['min_matching_rate'] < 100:
        errors.append(f"run: neighbours: min_matching_rate must be a number in [0, 100[, not {nb['min_matching_rate']}")
    if nb['dispatch'] not in ('front', 'tiles'):
        errors.append(f"run: neighbours: dispatch must be 'front' or 'tiles', not {nb['dispatch']}")
    if not isinstance(nb['front_attempts'], int) or nb['front_attempts'] < 0:
        errors.append(f"run: neighbours: front_attempts must be an integer >= 0, not {nb['front_attempts']}")
    if nb['seed_grid'] not in ('rectangular', 'hexagonal', 'none', None):
        errors.append(f"run: neighbours: seed_grid must be 'rectangular', 'hexagonal' or 'none', not {nb['seed_grid']}")
    if not (isinstance(nb['seed_images'], (list, tuple))
            and all(isinstance(i, (int, np.integer)) and i >= 0 for i in nb['seed_images'])):
        errors.append(f"run: neighbours: seed_images must be a list of image indices, not {nb['seed_images']}")
    if nb['seed_spacing'] is not None and (not isinstance(nb['seed_spacing'], (int, float)) or nb['seed_spacing'] <= 0):
        errors.append(f"run: neighbours: seed_spacing must be null or a positive number (micron), not {nb['seed_spacing']}")
    if nb['seeds_file'] is not None and not Path(str(nb['seeds_file'])).is_file():
        errors.append(f"run: neighbours: seeds_file {nb['seeds_file']} not found")
    for key in ('check_angle_tol', 'check_min_matching_rate'):
        if nb[key] is not None and (not isinstance(nb[key], (int, float)) or nb[key] <= 0):
            errors.append(f'run: neighbours: {key} must be null or a positive number, not {nb[key]}')
    for key in ('seed_step', 'library_size'):
        if not isinstance(nb[key], int) or nb[key] < 0:
            errors.append(f'run: neighbours: {key} must be an integer >= 0, not {nb[key]}')
    if nb['tile_size'] is not None and not (isinstance(nb['tile_size'], (list, tuple)) and len(nb['tile_size']) == 2
                                            and all(isinstance(v, int) and v > 0 for v in nb['tile_size'])):
        errors.append(f"run: neighbours: tile_size must be null or [fast, slow] positive integers, not {nb['tile_size']}")
    return errors


# --------------------------------------------------------------------------------------
#  UB STORE (files)
# --------------------------------------------------------------------------------------
class UBStore:
    """UB matrices found in the images of a map, in files of <fit_folder>/ub_store (see module docstring)"""

    def __init__(self, fit_folder):
        self.folder = Path(fit_folder) / 'ub_store'
        self.libfolder = self.folder / 'library'

    def create(self):
        self.libfolder.mkdir(parents=True, exist_ok=True)
        return self

    def clear(self):
        """remove the UB matrices of a previous run (they are also in the .fit files)"""
        if self.folder.is_dir():
            shutil.rmtree(self.folder)
        return self.create()

    @staticmethod
    def _save(path: Path, rows: np.ndarray):
        tmp = path.with_name(f'.{path.stem}_{os.getpid()}.tmp.npy')
        np.save(tmp, rows)
        os.replace(tmp, path)

    def write(self, image_index: int, rows: np.ndarray, library: bool = False):
        self._save(self.folder / f'{image_index}.npy', rows)
        if library and len(rows):
            self._save(self.libfolder / f'{image_index}.npy', rows)

    def read(self, image_index: int) -> Optional[np.ndarray]:
        """rows of an image, None if not treated yet"""
        try:
            return np.load(self.folder / f'{image_index}.npy')
        except (FileNotFoundError, ValueError, OSError):
            return None

    def library(self) -> List[tuple]:
        """[(image index, rows)] of the images indexed from scratch"""
        out = []
        for f in self.libfolder.glob('*.npy'):
            try:
                out.append((int(f.stem), np.load(f)))
            except (ValueError, OSError):
                pass
        return out

    def treated(self) -> set:
        return {int(f.stem) for f in self.folder.glob('*.npy') if f.stem.isdigit()}

    def nb_treated(self) -> int:
        return len(self.treated())


def result_rows(result: list) -> np.ndarray:
    """rows [UB (9), nb indexed spots, matching rate] of the grains of a result of index_refine()"""
    ubs = dict(result[1]) if len(result) > 1 else {}
    rows = []
    for grain, (nb, mr) in (result[2] if len(result) > 2 else []):
        ub = ubs.get(grain)
        if ub is not None and nb is not None:
            rows.append(np.r_[np.asarray(ub, dtype=float).ravel(), nb, mr])
    return np.array(rows).reshape(-1, 11)


# --------------------------------------------------------------------------------------
#  MAP GEOMETRY: seeds and tiles
# --------------------------------------------------------------------------------------
def map_coords(image_index, mapdims):
    """(fast, slow) position in the map of image index(es)"""
    return np.asarray(image_index) % mapdims[0], np.asarray(image_index) // mapdims[0]


def _metric(params_or_steps):
    """(units along fast, slow) of the map distances: step sizes (micron) if known, else (1, 1) (map points)"""
    steps = getattr(params_or_steps, 'stepsizes', params_or_steps)
    if steps is not None and len(steps) == 2 and all(np.isfinite(steps)) and all(abs(v) > 0 for v in steps):
        return float(abs(steps[0])), float(abs(steps[1]))
    return 1., 1.


def seed_indices(indices, mapdims, seed_step: int, grid: str = 'rectangular', spacing: Optional[float] = None,
                 stepsizes=None, seed_images=None) -> np.ndarray:
    """image indices of the seed pass, on a regular grid of the map, each grid node taken at the nearest map point
    (nodes whose map point has no image are skipped), and the images seed_images (those among indices)

    seed_step: grid spacing in map points (in both directions); spacing: grid spacing in micron (with stepsizes
        [fast, slow] (micron) of the map: same spacing in real space in both directions), replaces seed_step
    grid 'rectangular': nodes every spacing in both directions, centred on the map (seed_step without spacing:
        every seed_step points, offset seed_step // 2). Inside the grid, every point is within spacing / sqrt(2) of
        a seed
    grid 'hexagonal': rows every spacing * sqrt(3) / 2 (slow direction), nodes every spacing along the rows
        (fast direction), shifted by spacing / 2 every other row, centred on the map. Inside the grid, every point
        is within spacing / sqrt(3) of a seed (most compact: ~23 % less seeds than the square grid for the same max
        distance to a seed, in a large map)
    grid 'none' (or None): no grid, the seeds are seed_images only
    """
    indices = np.asarray(indices, dtype=int)
    grid_seeds = _grid_seeds(indices, mapdims, seed_step, grid, spacing, stepsizes)
    if not seed_images:
        return grid_seeds
    return indices[np.isin(indices, grid_seeds) | np.isin(indices, np.asarray(seed_images, dtype=int))]


def _grid_seeds(indices, mapdims, seed_step, grid, spacing, stepsizes) -> np.ndarray:
    """image indices of the nodes of the seed grid (see seed_indices())"""
    if grid in ('none', None) or (spacing is None and not seed_step):
        return indices[:0]
    nfast, nslow = mapdims
    uf, us = _metric(stepsizes) if spacing is not None else (1., 1.)
    d = float(spacing if spacing is not None else seed_step)
    def centred(length, step):   # nodes every step, centred on [0, length] (same margins on both sides)
        n = max(1, int(np.ceil(length / step)))
        return (length - (n - 1) * step) / 2 + step * np.arange(n)

    # map points at 0 .. (n - 1) * unit: extent of the map
    lf, ls = (nfast - 1) * uf, (nslow - 1) * us
    if grid == 'hexagonal':
        ys = centred(ls, d * np.sqrt(3) / 2)
        xs = centred(lf, d)
        nodes = [(x + (d / 2 if j % 2 else 0), y) for j, y in enumerate(ys) for x in np.r_[xs, xs[0] - d]]
    elif grid == 'rectangular':
        if spacing is None:   # every seed_step points, offset seed_step // 2 (exact integers)
            fast, slow = map_coords(indices, mapdims)
            off = seed_step // 2
            return indices[((fast - off) % seed_step == 0) & ((slow - off) % seed_step == 0)]
        nodes = [(x, y) for y in centred(ls, d) for x in centred(lf, d)]
    else:
        raise ValueError(f"seed_grid must be 'rectangular' or 'hexagonal', not {grid}")
    seeds = set()
    for x, y in nodes:
        f, sl = int(round(x / uf)), int(round(y / us))
        if 0 <= f < nfast and 0 <= sl < nslow:
            seeds.add(sl * nfast + f)
    return indices[np.isin(indices, list(seeds))]


def seeds_of(indices, params) -> np.ndarray:
    """seeds of the images `indices` with the neighbours parameters of params (IndexRefineParams)"""
    nb = settings(params.neighbours)
    return seed_indices(indices, params.mapdims, nb['seed_step'], nb['seed_grid'], nb['seed_spacing'],
                        getattr(params, 'stepsizes', None), nb['seed_images'])


def tiles(indices, mapdims, seeds, tile_size=None, nb_tiles: int = 1, stepsizes=None) -> List[List[int]]:
    """tiles (lists of image indices) of the main pass, each tile ordered by increasing distance to the nearest seed
    (real distance with stepsizes [fast, slow] (micron) of the map, else in map points)
    tile_size: [fast, slow] nb of points, None: square tiles, about nb_tiles tiles"""
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    fast, slow = map_coords(indices, mapdims)
    if tile_size is None:
        side = max(1, int(np.sqrt(len(indices) / max(nb_tiles, 1))))
        tile_size = (side, side)
    tf, ts = tile_size
    if len(seeds):
        sf, ss = map_coords(np.asarray(seeds), mapdims)
        uf, us = _metric(stepsizes)
        dist = np.min(np.hypot((fast[:, None] - sf[None]) * uf, (slow[:, None] - ss[None]) * us), axis=1)
    else:   # no seed: serpentine order of the map
        dist = slow * mapdims[0] + np.where(slow % 2, mapdims[0] - 1 - fast, fast)
    keys = (slow // ts) * (mapdims[0] // tf + 1) + fast // tf
    out = []
    for key in np.unique(keys):
        sel = np.flatnonzero(keys == key)
        out.append(indices[sel[np.argsort(dist[sel], kind='stable')]].tolist())
    return out


def neighbour_offsets(radius: int, mapdims, stepsizes=None) -> list:
    """index offsets (dfast, dslow) of the square neighbourhood (radius in map points), nearest first in real space
    with stepsizes [fast, slow] (micron) of the map, else in map points (the point itself excluded)"""
    uf, us = _metric(stepsizes)
    offs = [(df, ds) for ds in range(-radius, radius + 1) for df in range(-radius, radius + 1) if (df, ds) != (0, 0)]
    return sorted(offs, key=lambda o: ((o[0] * uf) ** 2 + (o[1] * us) ** 2, abs(o[1])))


# --------------------------------------------------------------------------------------
#  WORKER
# --------------------------------------------------------------------------------------
def accepted(rows: np.ndarray, nb: dict) -> np.ndarray:
    """mask of the grains (rows of result_rows()) accepted: min_nb_matches spots and min_matching_rate"""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    return (rows[:, 9] >= nb['min_nb_matches']) & (rows[:, 10] >= nb['min_matching_rate'])


class _Guide:
    """state of a worker process: UB store, UB matrices read, library, UB of the previous image"""

    def __init__(self, params, tag: str = ''):
        self.params = params
        self.nb = settings(params.neighbours)
        self.mapdims = params.mapdims
        self.store = UBStore(params.fit_folder)
        self.cache = {}            # image index -> rows (read from the store or found by this process)
        self.lib, self.lib_time = [], -np.inf
        self.last = None           # rows of the previous image of this process
        self.offsets = neighbour_offsets(self.nb['radius'], self.mapdims, getattr(params, 'stepsizes', None))
        # .fit files of the UB checks: own folder per process (tag: node and job of a SLURM array)
        self.trials = Path(params.fit_folder) / 'ub_store' / 'trials' / f'{tag}{os.getpid()}'
        self.references = [(np.r_[np.asarray(ub, dtype=float).ravel(), 0, 0], f'reference UB {name}')
                           for name, ub in self.nb['reference_UBs']]   # (resolved by params_from_config())

    def rows(self, image_index: int):
        if image_index not in self.cache:
            rows = self.store.read(image_index)
            if rows is None:
                return None
            self.cache[image_index] = rows
        return self.cache[image_index]

    def library(self):
        if time.time() - self.lib_time > self.nb['library_refresh']:
            self.lib_time = time.time()
            entries = sorted(self.store.library(), key=lambda e: -e[1][:, 9].max())
            self.lib = [(i, r) for i, rows in entries for r in rows]
        return self.lib

    def candidates(self, image_index: int) -> List[tuple]:
        """[(UB 3x3, source text)] distinct UB matrices to check for an image"""
        from LaueTools import orientationclustering as OC
        nfast, nslow = self.mapdims
        fast, slow = image_index % nfast, image_index // nfast
        cands, rots = [], []

        def add(row, source):
            ub = row[:9].reshape(3, 3)
            rot = OC.orthonormalize(ub)
            if rots:
                ang, _ = OC.misorientation(np.broadcast_to(rot, (len(rots), 3, 3)), np.array(rots), np.eye(3)[None])
                if np.min(ang) < SAME_ORIENTATION_DEG:
                    return
            cands.append((ub, source))
            rots.append(rot)

        if self.last is not None:
            for row in self.last[1]:
                add(row, f'previous image {self.last[0]}')
        for df, ds in self.offsets:
            if len(cands) >= self.nb['max_candidates']:
                break
            f, s = fast + df, slow + ds
            if 0 <= f < nfast and 0 <= s < nslow:
                rows = self.rows(s * nfast + f)
                if rows is not None:
                    for row in rows:
                        if accepted(row, self.nb)[0]:
                            add(row, f'neighbour {s * nfast + f}')
        nmax = len(cands) + self.nb['library_size']
        for index, row in self.library():
            if len(cands) >= nmax:
                break
            add(row, f'library {index}')
        for row, source in self.references:
            add(row, source)
        return cands

    def _move_trial_fitfiles(self, stem: str, image_index: int):
        dest = self.params.fitfile_folder(image_index)
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob(f'{stem}_g*.fit'):
            old.unlink()
        for f in self.trials.glob(f'{stem}_g*.fit'):
            os.replace(f, dest / f.name)

    def index_image(self, filename: str, from_scratch: bool = False, defer: bool = False) -> list:
        """index_refine() of an image, checking the candidate UB matrices first (see module docstring)
        defer: no candidate accepted -> no indexing from scratch: result with info starting with DEFERRED (nothing
        written: the image is tried again later, see run_neighbours() dispatch 'front')"""
        from LaueTools import indexing_batch as IB
        params = self.params
        image_index = GT.getfileindex(Path(filename).name)
        stem = Path(filename).stem
        base = dataclasses.replace(params, usepreviousUB=False, neighbours=None)
        tried = 0
        cands = [(r[:9].reshape(3, 3), src) for r, src in self.references] if from_scratch \
            else self.candidates(image_index)
        if cands:
            ngrains = params.nbGrainstoFind if isinstance(params.nbGrainstoFind, int) else None
            trial = dataclasses.replace(check_params(params), usepreviousUB=False, neighbours=None, skipindexing=True,
                                        fit_folder=str(self.trials), fit_nbfiles_per_folder=None)
            if ngrains == 1:   # one candidate at a time: the first one giving min_nb_matches spots is accepted
                for ub, source in cands:
                    tried += 1
                    res = IB.index_refine(filename, dataclasses.replace(trial, usepreviousUB=[ub]))
                    rows = result_rows(res)
                    if len(rows) and accepted(rows[:1], self.nb)[0]:
                        self._move_trial_fitfiles(stem, image_index)
                        return self._done(image_index, res, rows, ('seed: ' if from_scratch else '')
                                          + f'UB guess accepted: {source} ({tried} tried).')
                    if res[3] in ('missing .cor file', '.cor file is empty!') or 'too few or too many' in str(res[3]):
                        return self._done(image_index, res, rows, '')
            else:   # several grains: all candidates at once, then from scratch for the grains not found
                tried = len(cands)
                res = IB.index_refine(filename, dataclasses.replace(trial, usepreviousUB=[c[0] for c in cands]))
                rows = result_rows(res)
                good = rows[accepted(rows, self.nb)]
                if ngrains is not None and len(good) >= ngrains:
                    self._move_trial_fitfiles(stem, image_index)
                    return self._done(image_index, res, rows, f'UB guesses accepted ({tried} tried).')
                if len(good):
                    base = dataclasses.replace(base, usepreviousUB=[r[:9].reshape(3, 3) for r in good])
            for f in self.trials.glob(f'{stem}_g*.fit'):
                f.unlink()
        if defer and not from_scratch and base.usepreviousUB is False:   # tried again later (front)
            return [image_index, [], [], f'{DEFERRED} ({tried} tried).']
        if params.skipindexing and not from_scratch:   # only UB checks (no indexing from scratch, except seeds)
            return self._done(image_index, [image_index, [], [], ''], np.zeros((0, 11)),
                              f'no UB guess accepted ({tried} tried), not indexed from scratch (skipindexing).')
        res = IB.index_refine(filename, base)
        rows = result_rows(res)
        what = 'seed, indexed from scratch.' if from_scratch else f'no UB guess accepted ({tried} tried), indexed from scratch.'
        return self._done(image_index, res, rows, what, library=bool(np.any(accepted(rows, self.nb))))

    def index_validated_seed(self, filename: str, UB) -> list:
        """validated seed (indexed beforehand, see seeds_config()): its UB matrix refined with the parameters of the
        map (as the other points: all the .fit files of the map are refined the same way)"""
        from LaueTools import indexing_batch as IB
        image_index = GT.getfileindex(Path(filename).name)
        trial = dataclasses.replace(check_params(self.params), usepreviousUB=[np.asarray(UB, dtype=float)],
                                    neighbours=None, skipindexing=True, fit_folder=str(self.trials),
                                    fit_nbfiles_per_folder=None)
        res = IB.index_refine(filename, trial)
        rows = result_rows(res)
        if len(rows) and accepted(rows[:1], self.nb)[0]:
            self._move_trial_fitfiles(Path(filename).stem, image_index)
            return self._done(image_index, res, rows, 'seed, indexed from scratch (validated seed).')
        for f in self.trials.glob(f'{Path(filename).stem}_g*.fit'):
            f.unlink()
        return self.index_image(filename)   # as the other points (its UB is in the library)

    def _done(self, image_index, res, rows, text, library=False):
        code = source_code(text, rows)
        self.store.write(image_index, np.c_[rows, np.full(len(rows), code)], library=bool(library))
        self.cache[image_index] = rows
        if np.any(accepted(rows, self.nb)):
            self.last = (image_index, rows[accepted(rows, self.nb)])
        if library:
            self.lib_time = -np.inf   # read the library again (with this new orientation)
        res = list(res)
        res[3] = f'{text} {res[3]}'.strip()
        return res


_GUIDE = None


def _trials_tag(chunk: int) -> str:
    import socket
    return f'{socket.gethostname()}_{chunk}_'


def _init_worker(params, tag=''):
    global _GUIDE
    _GUIDE = _Guide(params, tag)
    _GUIDE.trials.mkdir(parents=True, exist_ok=True)


def _validated_seed_worker(task):
    return _GUIDE.index_validated_seed(*task)


def _front_worker(task):
    """task (image index, filename, defer) -> (image index, result)"""
    index, filename, defer = task
    return index, _GUIDE.index_image(filename, defer=defer)


def _seed_worker(filename):
    return _GUIDE.index_image(filename, from_scratch=True)


def _tile_worker(filenames):
    return [_GUIDE.index_image(f) for f in filenames]


# --------------------------------------------------------------------------------------
#  BATCH
# --------------------------------------------------------------------------------------
def guide_statistics(allresults: list, verbose: bool = True) -> dict:
    """nb of images indexed by checking a UB guess (previous image, neighbour, library) or from scratch"""
    import collections
    count = collections.Counter()
    for res in allresults:
        info = str(res[3]) if len(res) > 3 else ''
        for key, text in (('previous image', 'accepted: previous image'), ('neighbour', 'accepted: neighbour'),
                          ('library', 'accepted: library'), ('reference UB', 'accepted: reference UB'),
                          ('several grains', 'UB guesses accepted'),
                          ('seed', 'seed, indexed from scratch'), ('from scratch', 'no UB guess accepted ('),
                          ('not indexed (skipindexing)', 'not indexed from scratch (skipindexing)')):
            if text in info:
                count[key] += 1
    if verbose:
        print('images: ' + ', '.join(f'{v} {k}' for k, v in count.items()))
    return dict(count)


def run_neighbours(listfiles: List[str], params, nb_cpus: Optional[int] = None, progress_interval: float = 0.1,
                   chunk: int = 0, nchunks: int = 1, clear_store: Optional[bool] = None) -> list:
    """index & refine the .cor files of listfiles (all images of the map), guided by the UB matrices already found
    (see module docstring). Returns the list of results of index_refine() (unordered)

    chunk, nchunks: jobs of a SLURM array (e.g. 3 nodes of magnifix3): job `chunk` indexes the points of band
        `chunk` of the map (consecutive rows); UB matrices are shared between jobs by the UB store.
    clear_store: remove the UB store of a previous run first (default: True if nchunks == 1. Jobs of a SLURM array
        do not clear it: done when the job is prepared)
    """
    from LaueTools import indexing_batch as IB
    if params.mapdims is None:
        raise ValueError('usepreviousUB neighbours: params.mapdims is needed (map dimensions [fast, slow])')
    nb = settings(params.neighbours)
    nb_cpus = nb_cpus or IB.available_cpus()
    params = dataclasses.replace(params, useinternalmultiprocessing=False, outputlistindices=False)
    if params.LUT is None:
        params.build_LUT()
    store = UBStore(params.fit_folder)
    if clear_store is None:
        clear_store = nchunks == 1
    store.clear() if clear_store else store.create()

    byindex = {GT.getfileindex(Path(f).name): str(f) for f in listfiles}
    indices = np.array(sorted(byindex), dtype=int)
    if nchunks > 1:   # band of consecutive rows of the map (rows with images)
        slow = indices // params.mapdims[0]
        bands = np.array_split(np.unique(slow), nchunks)[chunk]
        indices = indices[np.isin(slow, bands)]
    validated = load_validated_seeds(nb['seeds_file']) if nb['seeds_file'] else None
    if validated is not None:   # seeds indexed beforehand and validated: no seed pass
        if not validated['seeds']:
            raise ValueError(f"no validated seed in {nb['seeds_file']}: validate seeds (or set seeds_file to null)")
        seeds = np.array([i for i in sorted(validated['seeds']) if i in set(indices.tolist())], dtype=int)
    else:
        seeds = seeds_of(indices, params)
    others = np.setdiff1d(indices, seeds)
    front = nb['dispatch'] == 'front'
    tilelist = [] if front else tiles(others, params.mapdims, seeds, nb['tile_size'], nb_tiles=4 * nb_cpus,
                                      stepsizes=getattr(params, 'stepsizes', None))
    seedgrid = ('no grid' if nb['seed_grid'] in ('none', None) else
                f"{nb['seed_grid']} grid, spacing " + (f"{nb['seed_spacing']:g} micron" if nb['seed_spacing'] is not None
                                                       else f"{nb['seed_step']} points"))
    if nb['seed_images']:
        seedgrid += f" + {len(nb['seed_images'])} seed_images"
    if validated is not None:
        seedgrid = f"validated seeds of {nb['seeds_file']}, not indexed again"
    print(f"{len(indices)} images{f' (band {chunk + 1}/{nchunks} of the map)' if nchunks > 1 else ''}: "
          f"{len(seeds)} seeds ({seedgrid}), then {len(others)} images "
          + ('by a front from the accepted points (points next to the most accepted points first, deferred indexing '
             f"from scratch after {nb['front_attempts']} attempts)" if front else f'in {len(tilelist)} tiles')
          + f" checking first the UB matrices of neighbours within {nb['radius']} points.\n"
          f"{check_description(params)}\nUB store: {store.folder}", flush=True)

    t0 = time.time()
    allresults = []
    tag = _trials_tag(chunk)
    if validated is not None:   # orientations of the validated seeds in the library of the UB store
        use_validated_seeds(validated, seeds, store)
    with IB.mp_context().Pool(processes=nb_cpus, initializer=_init_worker, initargs=(params, tag)) as pool:
        if len(seeds) and validated is not None:   # validated seeds refined with the parameters of the map
            tasks = [(byindex[i], validated['seeds'][int(i)]['UB']) for i in seeds]
            for res in tqdm(pool.imap_unordered(_validated_seed_worker, tasks, chunksize=1), total=len(tasks),
                            mininterval=progress_interval, desc='validated seeds'):
                allresults.append(res)
        if len(seeds) and validated is None:
            for res in tqdm(pool.imap_unordered(_seed_worker, [byindex[i] for i in seeds], chunksize=1),
                            total=len(seeds), mininterval=progress_interval, desc='seeds'):
                allresults.append(res)
            t1 = time.time()
            nfound = sum(1 for r in allresults if len(result_rows(r)))
            print(f'seed pass: {nfound}/{len(seeds)} seeds indexed in {int((t1 - t0) // 60)} min {int((t1 - t0) % 60)} s',
                  flush=True)
        if front:
            accepted_seeds = [r[0] for r in allresults if np.any(accepted(result_rows(r), nb))]
            allresults += _front_dispatch(pool, others, accepted_seeds, seeds, byindex, params, nb, nb_cpus,
                                          progress_interval)
        else:
            with tqdm(total=len(others), mininterval=progress_interval, desc='map') as bar:
                for results in pool.imap_unordered(_tile_worker, [[byindex[i] for i in t] for t in tilelist],
                                                   chunksize=1):
                    allresults += results
                    bar.update(len(results))
    for trials in (store.folder / 'trials').glob(f'{tag}*'):   # (not those of the other jobs, still running)
        shutil.rmtree(trials, ignore_errors=True)
    dt = time.time() - t0
    print(f"It took {int(dt // 60)} min {int(dt % 60)} s to index {len(indices)} images with {nb_cpus} cpus.")
    guide_statistics(allresults)
    return allresults


def _front_dispatch(pool, others, accepted_seeds, seeds, byindex, params, nb, nb_cpus, progress_interval) -> list:
    """index the points `others` one task per point (as many tasks running as cpus), in the order of a front:
        1. points with the most accepted neighbours (8 neighbours in the map), from a priority queue updated with the
           results; a point without accepted UB guess is deferred (tried again when a new neighbour is accepted, at
           most front_attempts guess attempts)
        2. front empty: points not tried yet (nearest to a seed first), UB guesses only (library, references ...)
        3. nothing else to try: deferred points indexed from scratch (their new orientations restart the front)
    Returns the results of the points (one per point)"""
    import heapq
    import queue
    nfast, nslow = params.mapdims
    attempts_max = nb['front_attempts']
    state = {int(i): 'todo' for i in others}        # todo, inflight, deferred, done
    attempts = {i: 0 for i in state}
    count = {i: 0 for i in state}                   # nb of accepted neighbours
    heap, tiebreak = [], [0]
    # points not tried yet when the front is empty: nearest to a seed first
    order = tiles(np.asarray(sorted(state), dtype=int), params.mapdims, seeds, tile_size=(nfast, nslow),
                  stepsizes=getattr(params, 'stepsizes', None))
    order = iter(order[0] if order else [])
    deferred_ever, final_phase = set(), [False]
    results_queue = queue.Queue()
    results, inflight = [], [0]

    def neighbours8(i):
        f, s = i % nfast, i // nfast
        for ds in (-1, 0, 1):
            for df in (-1, 0, 1):
                if (df or ds) and 0 <= f + df < nfast and 0 <= s + ds < nslow:
                    yield (s + ds) * nfast + f + df

    def on_accepted(i):
        for j in neighbours8(i):
            if state.get(j) in ('todo', 'deferred'):
                count[j] += 1
                tiebreak[0] += 1
                heapq.heappush(heap, (-count[j], tiebreak[0], j))

    def pick():
        """(index, defer) of the next point, None if nothing to start now"""
        while heap:   # 1. front
            negcount, _, j = heapq.heappop(heap)
            if state.get(j) in ('todo', 'deferred') and -negcount == count[j]:
                return j, attempts[j] < attempts_max
        for j in order:   # 2. points not tried yet: UB guesses only
            if state[j] == 'todo':
                return j, attempts_max > 0
        if inflight[0] == 0:
            final_phase[0] = True
        if final_phase[0]:   # 3. deferred points: indexing from scratch
            for j in sorted((j for j, st in state.items() if st == 'deferred'), key=lambda j: -count[j]):
                return j, False
        return None

    for i in accepted_seeds:
        on_accepted(int(i))
    with tqdm(total=len(state), mininterval=progress_interval, desc='map') as bar:
        while True:
            while inflight[0] < nb_cpus:
                task = pick()
                if task is None:
                    break
                j, defer = task
                state[j] = 'inflight'
                if defer:
                    attempts[j] += 1
                inflight[0] += 1
                pool.apply_async(_front_worker, ((j, byindex[j], defer),), callback=results_queue.put,
                                 error_callback=lambda err, j=j: results_queue.put((j, err)))
            if inflight[0] == 0:
                if all(st == 'done' for st in state.values()):
                    break
                continue   # (final phase starts at the next pick)
            j, res = results_queue.get()
            inflight[0] -= 1
            if isinstance(res, BaseException):
                raise res
            if str(res[3]).startswith(DEFERRED):
                state[j] = 'deferred'
                deferred_ever.add(j)   # (tried again when a neighbour is accepted, from scratch at the end)
                continue
            state[j] = 'done'
            if j in deferred_ever:
                res = list(res)
                res[3] = f'{res[3]} (deferred {attempts[j]} time(s) before)'
            results.append(res)
            bar.update(1)
            if np.any(accepted(result_rows(res), nb)):
                on_accepted(j)
    later = sum(1 for r in results if r[0] in deferred_ever and 'UB guess accepted:' in str(r[3]))
    scratch = sum(1 for r in results if r[0] in deferred_ever and 'indexed from scratch' in str(r[3]))
    print(f'front: {len(deferred_ever)} point(s) deferred: {later} accepted later from an indexed neighbour, '
          f'{scratch} indexed from scratch at the end', flush=True)
    return results


# --------------------------------------------------------------------------------------
#  SEEDS INDEXED BEFOREHAND AND VALIDATED BY THE USER
# --------------------------------------------------------------------------------------
def seeds_config(cfg: dict, neighbours: Optional[dict] = None, criteria=None, max_pixdev: Optional[float] = 1.0,
                 newfile=None, fit_folder=None, overwrite: bool = True) -> dict:
    """configuration file of the indexing from scratch of the seeds only (step 1 of the indexing guided by
    neighbours): copy of cfg (indexing.yaml) with
        - selection: image_indices = seeds (grid of `neighbours`: seed_grid, seed_step or seed_spacing, and the
          images seed_images; default: run: neighbours of cfg), within the images selected by cfg
        - usepreviousUB: false (indexing from scratch, no reference UB matrix)
        - fit_folder: <folder of cfg fit_folder>/seeds_<material> (or fit_folder)
        - solution_criteria: criteria (steps [e_max, tolerance angle, min nb of matches]) that each seed found must
          pass to be validated (see IB.verify_solution()), checked in the seeds job; None: no check
        - solution_max_pixdev: max_pixdev, max mean pixel deviation of the last step of the criteria (None: not
          checked)
    newfile: default <configs>/indexing_seeds.yaml. Returns the new configuration (run it with MW.run_step())"""
    from LaueTools import indexing_batch as IB
    nb = settings({**(cfg['run'].get('neighbours') or {}), **(neighbours or {})})
    indices = [GT.getfileindex(Path(f).name) for f in IB.select_corfiles(cfg)]
    seeds = seed_indices(indices, cfg['scan']['mapdims'], nb['seed_step'], nb['seed_grid'], nb['seed_spacing'],
                         cfg['scan'].get('stepsizes'), nb['seed_images'])
    missing = sorted(set(int(i) for i in nb['seed_images']) - set(int(i) for i in indices))
    if missing:
        GT.printyellow(f'seed_images {missing} skipped: not among the images selected by the configuration')
    newfile = Path(newfile) if newfile else Path(cfg['configfile']).parent / 'indexing_seeds.yaml'
    fit_folder = Path(fit_folder) if fit_folder else (Path(cfg['output']['fit_folder']).parent
                                                     / f"seeds_{cfg['material']['key_material']}")
    print(seed_grid_description(nb, cfg) + f': {len(seeds)} seeds')
    # (seed grid settings written too: the seeds of image_indices are those of this grid)
    return IB.save_config(cfg, newfile, overwrite=overwrite, usepreviousUB=False, skipindexing=False,
                          fit_folder=str(fit_folder), image_indices=[int(i) for i in seeds],
                          solution_criteria=criteria, solution_max_pixdev=max_pixdev,
                          seed_grid=nb['seed_grid'], seed_spacing=nb['seed_spacing'], seed_step=nb['seed_step'],
                          seed_images=[int(i) for i in nb['seed_images']])


def update_seed_images(scfg: dict, seed_images) -> dict:
    """seeds configuration scfg (seeds_config()) with seed_images replaced by the list seed_images (e.g.
    SEEDS['seed_images'] after SeedPicker.accept()): image_indices = seeds of the grid of scfg + seed_images (those
    with a .cor file); written in the file of scfg. Returns the new configuration (run it with MW.run_step())"""
    from LaueTools import indexing_batch as IB
    nb = settings(scfg['run'].get('neighbours'))
    nfast, nslow = scfg['scan']['mapdims']
    old = np.asarray(scfg['selection']['image_indices'] or [], dtype=int)
    # seeds of the grid: grid nodes among the former seeds (grid nodes without image are not seeds)
    grid = np.intersect1d(_grid_seeds(np.arange(nfast * nslow), (nfast, nslow), nb['seed_step'], nb['seed_grid'],
                                      nb['seed_spacing'], scfg['scan'].get('stepsizes')), old)
    seed_images = [int(i) for i in dict.fromkeys(seed_images)]
    missing = [i for i in seed_images if not IB.corfile_path(scfg, i).is_file()]
    if missing:
        GT.printyellow(f'seed_images {missing} skipped: no .cor file')
    added = [i for i in seed_images if i not in missing]
    seeds = sorted(set(grid.tolist()) | set(added))
    print(f"{len(grid)} seeds of the grid + {len(added)} seed_images ({len(set(added) - set(grid.tolist()))} not "
          f"on the grid): {len(seeds)} seeds")
    return IB.save_config(scfg, scfg['configfile'], overwrite=True, image_indices=seeds, seed_images=seed_images)


def seed_grid_description(neighbours: Optional[dict], cfg: dict) -> str:
    """text of the seed grid: setting used (seed_spacing in micron, else seed_step in map points) and spacing in map
    points along the fast and slow motors"""
    nb = settings(neighbours)
    extra = f" + {len(nb['seed_images'])} seed_images" if nb['seed_images'] else ''
    return _grid_description(nb, cfg) + extra


def _grid_description(nb: dict, cfg: dict) -> str:
    scan = cfg['scan']
    fast, slow = scan.get('fast_motor', 'fast'), scan.get('slow_motor', 'slow')
    if nb['seed_grid'] in ('none', None):
        return 'no seed grid'
    if nb['seed_spacing'] is not None:
        uf, us = _metric(scan.get('stepsizes'))
        if (uf, us) == (1., 1.):
            return (f"{nb['seed_grid']} grid, seed_spacing {nb['seed_spacing']:g} (map step sizes unknown: map points)"
                    f" (seed_step not used)")
        return (f"{nb['seed_grid']} grid, seed_spacing {nb['seed_spacing']:g} micron (seed_step not used) = "
                f"{nb['seed_spacing'] / uf:.1f} points along {fast} (fast, step {uf:g} micron), "
                f"{nb['seed_spacing'] / us:.1f} points along {slow} (slow, step {us:g} micron)"
                + (' (hexagonal: rows every spacing * 0.866)' if nb['seed_grid'] == 'hexagonal' else ''))
    return (f"{nb['seed_grid']} grid, seed_step {nb['seed_step']} map points along {fast} and {slow} "
            f"(seed_spacing None)")


def plot_seed_grid(cfg: dict, seeds=None, ax=None, annotate: bool = True, cmap='viridis_r'):
    """plot of the seeds of the map in real space (micron, equal aspect: a hexagonal grid looks hexagonal), over the
    map of the distance from each map point to its nearest seed, before indexing them

    cfg: configuration of the seeds (seeds_config()), or of the map (then give seeds: image indices)
    annotate: image index of each seed. Seeds of seed_images (run: neighbours of cfg): magenta diamonds
    Returns ax"""
    import matplotlib.pyplot as plt
    from LaueTools import map_workflow as MW
    scan = cfg['scan']
    nfast, nslow = scan['mapdims']
    uf, us = _metric(scan.get('stepsizes'))
    unit = 'micron' if (uf, us) != (1., 1.) else 'map points'
    seeds = np.asarray(cfg['selection']['image_indices'] if seeds is None else seeds, dtype=int)
    sf, ss = seeds % nfast, seeds // nfast
    # distance of every map point to the nearest seed (real space)
    F, S = np.meshgrid(np.arange(nfast), np.arange(nslow))
    dist = np.full(F.shape, np.inf)
    for f0, s0 in zip(sf, ss):
        np.minimum(dist, np.hypot((F - f0) * uf, (S - s0) * us), out=dist)
    transposed = MW.map_transposed(cfg)   # (same orientation as the maps: y motor vertical)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    if transposed:   # horizontal: slow motor, vertical: fast motor
        img, extent = dist.T, [-us / 2, (nslow - 0.5) * us, -uf / 2, (nfast - 0.5) * uf]
        xs, ys = ss * us, sf * uf
        xlabel, ylabel = scan.get('slow_motor', 'slow'), scan.get('fast_motor', 'fast')
    else:
        img, extent = dist, [-uf / 2, (nfast - 0.5) * uf, -us / 2, (nslow - 0.5) * us]
        xs, ys = sf * uf, ss * us
        xlabel, ylabel = scan.get('fast_motor', 'fast'), scan.get('slow_motor', 'slow')
    im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax, shrink=0.7, label=f'distance to the nearest seed ({unit})')
    # seeds of seed_images (chosen by the user) in another color
    chosen = np.isin(seeds, [int(i) for i in settings(cfg['run'].get('neighbours'))['seed_images']])
    ax.scatter(xs[~chosen], ys[~chosen], s=25, c='red', edgecolors='white', linewidths=0.5, zorder=3)
    if chosen.any():
        ax.scatter(xs[chosen], ys[chosen], s=40, c='magenta', marker='D', edgecolors='white', linewidths=0.5,
                   zorder=3, label='seed_images')
        ax.legend(loc='upper right', fontsize=8)
    if annotate:
        for x, y, i in zip(xs, ys, seeds):
            ax.annotate(str(i), (x, y), textcoords='offset points', xytext=(3, 3), fontsize=6, color='white')
    ax.set_aspect('equal')
    ax.set_xlabel(f'{xlabel} ({unit})')
    ax.set_ylabel(f'{ylabel} ({unit})')
    ax.set_title(f'{len(seeds)} seeds: max distance to a seed {np.max(dist):.0f}, mean {np.mean(dist):.0f} {unit}',
                 fontsize=9)
    seedset = set(int(i) for i in seeds)

    def fmt(x, y):   # status bar text (interactive backend, e.g. %matplotlib widget)
        point = seed_plot_point(cfg, x, y)
        if point is None:
            return ''
        index, f, sl = point
        return (f"image {index} ({scan.get('fast_motor', 'fast')} #{f}, {scan.get('slow_motor', 'slow')} #{sl}), "
                f"distance to the nearest seed {dist[sl, f]:.0f} {unit}" + (' (seed)' if index in seedset else ''))
    ax.format_coord = fmt
    return ax


def seed_plot_point(cfg: dict, x: float, y: float):
    """(image index, fast index, slow index) of the map point at (x, y) of plot_seed_grid() (real space), or None"""
    nfast, nslow = cfg['scan']['mapdims']
    uf, us = _metric(cfg['scan'].get('stepsizes'))
    from LaueTools import map_workflow as MW
    fx, sx = (y / uf, x / us) if MW.map_transposed(cfg) else (x / uf, y / us)
    f, sl = int(np.floor(fx + 0.5)), int(np.floor(sx + 0.5))
    if 0 <= f < nfast and 0 <= sl < nslow:
        return sl * nfast + f, f, sl
    return None


class SeedPicker:
    """plot_seed_grid() with the mouse (interactive backend: %matplotlib widget, package ipympl):
        - hover: image index of the map point under the mouse (and whether it is a seed)
        - click: picks the map point (white cross); zoom / pan mode of the toolbar: no picking
        - key 'a' (plot focused) or accept(): the picked point is added to seeds_settings['seed_images']
          (magenta diamond); remove(index) removes an image of seed_images
        - update_config(): seeds configuration written with these seed_images (returns the new scfg)
    seeds_settings: dict of the seed settings (SEEDS of the notebook), changed in place (copy the printed
    seed_images into SEEDS to keep them when the settings cell is run again)

    Keep a reference to the picker (e.g. picker = IN.SeedPicker(...)): the mouse events need it"""

    def __init__(self, cfg: dict, seeds_settings: dict, ax=None, figsize=(8, 8), **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        self.cfg, self.settings, self.ax = cfg, seeds_settings, ax
        self.settings.setdefault('seed_images', [])
        plot_seed_grid(cfg, ax=ax, **kwargs)
        self.seeds = set(int(i) for i in cfg['selection']['image_indices'] or [])
        self.picked = None
        self._cross = None
        self._added = {}   # image index -> marker of the seeds added here
        self._hover = ax.annotate('', (0, 0), textcoords='offset points', xytext=(8, 8), fontsize=8,
                                  bbox={'boxstyle': 'round', 'fc': 'white', 'alpha': 0.85}, zorder=5, visible=False)
        canvas = ax.figure.canvas
        ax.figure._seed_picker = self   # (callbacks of mpl_connect are weak references)
        self._cids = [canvas.mpl_connect('motion_notify_event', self._on_move),
                      canvas.mpl_connect('button_press_event', self._on_click),
                      canvas.mpl_connect('key_press_event', self._on_key)]

    def _xy(self, index):
        nfast = self.cfg['scan']['mapdims'][0]
        uf, us = _metric(self.cfg['scan'].get('stepsizes'))
        from LaueTools import map_workflow as MW
        f, sl = index % nfast, index // nfast
        return (sl * us, f * uf) if MW.map_transposed(self.cfg) else (f * uf, sl * us)

    def _on_move(self, event):
        point = seed_plot_point(self.cfg, event.xdata, event.ydata) if event.inaxes is self.ax else None
        if point is None:
            if self._hover.get_visible():
                self._hover.set_visible(False)
                self.ax.figure.canvas.draw_idle()
            return
        index = point[0]
        self._hover.xy = (event.xdata, event.ydata)
        self._hover.set_text(f'{index}' + (' (seed)' if index in self.seeds or index in self._added else ''))
        self._hover.set_visible(True)
        self.ax.figure.canvas.draw_idle()

    def _on_click(self, event):
        toolbar = getattr(self.ax.figure.canvas, 'toolbar', None)
        if event.inaxes is not self.ax or (toolbar is not None and getattr(toolbar, 'mode', '')):
            return
        point = seed_plot_point(self.cfg, event.xdata, event.ydata)
        if point is None:
            return
        self.picked = point[0]
        x, y = self._xy(self.picked)
        if self._cross is None:
            self._cross = self.ax.scatter([x], [y], s=120, marker='x', c='white', linewidths=2, zorder=4)
        else:
            self._cross.set_offsets([[x, y]])
        self.ax.set_xlabel(self.ax.get_xlabel().split('  |')[0] + f"  |  picked: image {self.picked} (key 'a': add)")
        self.ax.figure.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'a':
            self.accept()

    def accept(self) -> Optional[int]:
        """add the picked point to seed_images (returns its image index, None if no point is picked)"""
        if self.picked is None:
            print('no picked point: click on the map first')
            return None
        index = int(self.picked)
        if index not in self.settings['seed_images']:
            self.settings['seed_images'].append(index)
        if index not in self._added:
            x, y = self._xy(index)
            self._added[index] = self.ax.scatter([x], [y], s=60, c='magenta', marker='D', edgecolors='white',
                                                 linewidths=0.8, zorder=4)
            self.ax.figure.canvas.draw_idle()
        print(f"image {index} added. seed_images: {self.settings['seed_images']} "
              '(scfg = picker.update_config() writes them; copy them into SEEDS to keep them)')
        return index

    def update_config(self) -> dict:
        """write the seeds configuration with seed_images = the current list (see update_seed_images()).
        Returns the new seeds configuration (scfg)"""
        self.cfg = update_seed_images(self.cfg, self.settings['seed_images'])
        self.seeds = set(int(i) for i in self.cfg['selection']['image_indices'] or [])
        return self.cfg

    def remove(self, index: int):
        """remove an image of seed_images"""
        index = int(index)
        if index in self.settings['seed_images']:
            self.settings['seed_images'].remove(index)
        marker = self._added.pop(index, None)
        if marker is not None:
            marker.remove()
            self.ax.figure.canvas.draw_idle()
        print(f"seed_images: {self.settings['seed_images']}")


def same_lattice_orientation(UB_A, UB_B, tol: float = 0.03) -> bool:
    """True if UB_B = UB_A . M with M a unimodular integer matrix (within tol): the same crystal orientation
    described with other Miller indices (symmetry equivalent UB matrices, any crystal system)"""
    M = np.linalg.solve(np.asarray(UB_A, dtype=float), np.asarray(UB_B, dtype=float))
    R = np.round(M)
    return bool(np.all(np.abs(M - R) < tol) and abs(abs(np.linalg.det(R)) - 1) < 1e-6)


def seed_table(allresults: list, cfg: dict, grain: int = 0, max_angle: float = 2.) -> List[dict]:
    """table of the seeds indexed from scratch (results of MW.run_step() with the configuration of seeds_config())
    [{'image_index', 'fast', 'slow', 'nb_spots', 'matching_rate', 'UB', 'orientation'}] sorted by image index.
    'orientation': number of the group of seeds with the same orientation (symmetry equivalent UB matrices and
    misorientation < max_angle deg), 0 = most frequent; -1: not indexed
    'criteria': True / False: criteria of a correct solution passed or not (solution_criteria of the seeds
    configuration), None: not checked; 'criteria_nb': nb of indexed spots at each step of the criteria"""
    from LaueTools import indexing_batch as IB
    nfast = cfg['scan']['mapdims'][0]
    table = []
    for res in sorted(allresults, key=lambda r: r[0]):
        rows = result_rows(res)
        row = rows[grain] if len(rows) > grain else None
        table.append({'image_index': int(res[0]), 'fast': int(res[0]) % nfast, 'slow': int(res[0]) // nfast,
                      'nb_spots': int(row[9]) if row is not None else 0,
                      'matching_rate': float(row[10]) if row is not None else 0.,
                      'UB': row[:9].reshape(3, 3) if row is not None else None, 'orientation': -1})
        crit = IB.parse_criteria_info(res[3] if len(res) > 3 else '').get(grain) if row is not None else None
        table[-1]['criteria'] = None if crit is None else crit['passed']
        table[-1]['criteria_nb'] = None if crit is None else crit['nb']
        table[-1]['criteria_pixdev'] = None if crit is None else crit['pixdev'][-1]
    return _orientation_groups(table, max_angle)


def _orientation_groups(table: List[dict], max_angle: float = 2.) -> List[dict]:
    """'orientation' of the seeds of table: number of the group of seeds with the same orientation (symmetry
    equivalent UB matrices and misorientation < max_angle deg), 0 = most frequent; -1: no UB matrix"""
    from LaueTools import orientationclustering as OC
    groups = []   # [representative UB, members]
    for t in sorted([t for t in table if t['UB'] is not None], key=lambda t: -t['nb_spots']):
        for g in groups:
            ref = g[0]
            ang, _ = OC.misorientation(OC.orthonormalize(ref)[None], OC.orthonormalize(t['UB'])[None], np.eye(3)[None])
            if ang[0] < max_angle or same_lattice_orientation(ref, t['UB'], tol=np.radians(max_angle)):
                g[1].append(t)
                break
        else:
            groups.append([t['UB'], [t]])
    for k, g in enumerate(sorted(groups, key=lambda g: -len(g[1]))):
        for t in g[1]:
            t['orientation'] = k
    return table


def check_seeds(scfg: dict, ub_matrices: dict, names=None, criteria=None, max_pixdev: Optional[float] = None,
                nb_cpus: Optional[int] = None, max_angle: float = 2.) -> List[dict]:
    """seeds of scfg (seeds_config(): grid and/or seed_images) checked with the UB matrices ub_matrices (names:
    those names, default all) and the criteria of a correct solution (default: those of scfg), instead of indexing
    them from scratch: each (seed, matrix) pair is linked and refined with the steps of the criteria
    (IB.test_ub_matrices(), no .fit file). Each seed keeps the matrix passing the criteria with the most indexed
    spots (if none passes: the one with the most indexed spots, criteria False).

    Returns the table of seed_table() (refined UB matrices, orientation groups ...) + 'matrix': name of the matrix,
    for validate_seeds(), print_seed_table(), plot_seeds() and save_validated_seeds()"""
    from LaueTools import indexing_batch as IB
    cfg = {**scfg, 'ub_matrices': dict(ub_matrices or {})}
    criteria = criteria if criteria is not None else scfg['indexing'].get('solution_criteria')
    tests = IB.test_ub_matrices(IB.select_corfiles(scfg), cfg, names=names, criteria=criteria,
                                max_pixdev=max_pixdev, nb_cpus=nb_cpus)
    nfast = scfg['scan']['mapdims'][0]
    table = []
    for index in sorted({i for v in tests.values() for i in v}):
        def score(name):
            v = tests[name][index]
            return (bool(v['passed']), v['nb_spots'] or 0)
        name = max((n for n in tests if index in tests[n]), key=score)
        v = tests[name][index]
        ub = v['UB_refined']
        table.append({'image_index': int(index), 'fast': int(index) % nfast, 'slow': int(index) // nfast,
                      'nb_spots': int(v['nb_spots'] or 0) if ub is not None else 0,
                      'matching_rate': float(v['matching_rate'] or 0.) if ub is not None else 0.,
                      'UB': None if ub is None else np.asarray(ub, dtype=float), 'orientation': -1,
                      'criteria': bool(v['passed']), 'criteria_nb': [st[3] for st in v['steps']],
                      'criteria_pixdev': v['pixdev'][-1], 'matrix': name})
    return _orientation_groups(table, max_angle)


def seeds_from_fitfiles(fitfiles, cfg: Optional[dict] = None, prefix: str = 'fit') -> dict:
    """UB matrices and seeds from .fit files (e.g. indexed and refined with the LaueTools GUI), to check them as
    seeds with the parameters of the map (check_seeds()) and reproduce these results before indexing the map.

    fitfiles: list of .fit files, a folder (all its .fit files) or a glob pattern
    cfg: configuration of the map: material and detector parameters of each .fit file compared with those of the
        map (key_material, calibration in the header of the .cor file of the image): a difference explains
        different results (nb of indexed spots, pixel deviation)
    Image index: of the .cor file written in the first line of the .fit file, else of the .fit file name.
    Name of the UB matrix of grain g of image i: '<prefix>_<i>_g<g>'

    Returns {'ub_matrices': {name: 3x3 list}, 'seed_images': [image indices],
             'table': [{'name', 'image_index', 'grain', 'nb_spots', 'pixdev', 'material', 'fitfile'}]}"""
    import glob
    import re
    from LaueTools import indexing_batch as IB
    from LaueTools import IOLaueTools as IOLT
    if isinstance(fitfiles, (str, Path)):
        path = Path(fitfiles)
        fitfiles = sorted(path.glob('*.fit')) if path.is_dir() else sorted(glob.glob(str(fitfiles)))
    out = {'ub_matrices': {}, 'seed_images': [], 'table': []}
    for fitfile in [str(f) for f in fitfiles]:
        source = IB.fitfile_source(fitfile)
        try:
            image_index = GT.getfileindex(Path(source or fitfile).name)
            grains, nbs, _, ubs = IOLT.readfitfile_multigrains(fitfile)[:4]
        except Exception as err:
            GT.printyellow(f'{fitfile} skipped: {err}')
            continue
        lines = Path(fitfile).read_text(errors='replace').splitlines()
        material = next((lines[k + 1].lstrip('#').strip() for k, l in enumerate(lines[:-1])
                         if l.strip() == '#Element'), None)
        pixdevs = [float(l.split(':')[1]) for l in lines if l.startswith('#Mean Deviation(pixel)') and ':' in l
                   and l.split(':')[1].strip()]
        detline = next((lines[k + 1] for k, l in enumerate(lines[:-1]) if l.strip() == '#DetectorParameters'), '')
        det = [float(v) for v in re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', re.sub(r'np\.float64', '', detline))]
        for k, (grain, nb, ub) in enumerate(zip(np.atleast_1d(grains), np.atleast_1d(nbs), ubs)):
            name = f'{prefix}_{image_index}_g{int(grain)}'
            out['ub_matrices'][name] = np.asarray(ub, dtype=float).reshape(3, 3).tolist()
            out['table'].append({'name': name, 'image_index': image_index, 'grain': int(grain), 'nb_spots': int(nb),
                                 'pixdev': pixdevs[k] if k < len(pixdevs) else None, 'material': material,
                                 'detector': det[:5] or None, 'fitfile': fitfile})
        if image_index not in out['seed_images']:
            out['seed_images'].append(image_index)
    print('name                 image  spots  pixdev  material')
    for t in out['table']:
        pd = '-' if t['pixdev'] is None else f"{t['pixdev']:.2f}"
        print(f"{t['name']:<20} {t['image_index']:6d} {t['nb_spots']:6d} {pd:>7}  {t['material']}")
    if cfg is not None:
        key = cfg['material']['key_material']
        for t in out['table']:
            if t['material'] and t['material'] != key:
                GT.printyellow(f"{t['name']}: material {t['material']} (.fit) != {key} (map): the results differ")
        for index in out['seed_images']:
            t = next(t for t in out['table'] if t['image_index'] == index)
            corfile = IB.corfile_path(cfg, index)
            if not corfile.is_file():
                GT.printyellow(f'image {index}: no .cor file {corfile} in the map')
                continue
            calib = IOLT.readfile_cor(str(corfile), output_CCDparamsdict=True)[-1]
            cor_det = [calib[k] for k in ('dd', 'xcen', 'ycen', 'xbet', 'xgam')] if isinstance(calib, dict) else None
            if t['detector'] and cor_det and not np.allclose(t['detector'], cor_det, atol=1e-3):
                GT.printyellow(f'image {index}: detector parameters {t["detector"]} (.fit) != {cor_det} (.cor of '
                               'the map): the results differ (calibration)')
    return out


def merge_ub_matrices(ub_matrices: dict, max_angle: float = 0.5, symmetry: bool = True,
                      verbose: bool = True) -> dict:
    """UB matrices of ub_matrices closer than max_angle (deg) merged (greedy, in the order of ub_matrices): each
    group keeps its first matrix, with its name (and so its reference image index, e.g. 'UB_<image>',
    'fit_<image>_g<grain>'). symmetry: symmetry equivalent UB matrices (same lattice, other Miller indices) are
    also merged, else raw angle only (no symmetry)

    Returns {'ub_matrices': {name: 3x3 list} kept, 'merged': {kept name: [names merged into it]}}"""
    from LaueTools import orientationclustering as OC
    kept, merged = {}, {}
    for name, ub in (ub_matrices or {}).items():
        ub = np.asarray(ub, dtype=float).reshape(3, 3)
        for ref_name, ref in kept.items():
            ang, _ = OC.misorientation(OC.orthonormalize(ref)[None], OC.orthonormalize(ub)[None], np.eye(3)[None])
            if ang[0] < max_angle or (symmetry and same_lattice_orientation(ref, ub, tol=np.radians(max_angle))):
                merged[ref_name].append((name, float(ang[0])))
                break
        else:
            kept[name] = ub
            merged[name] = []
    if verbose:
        print(f'{len(ub_matrices or {})} UB matrices -> {len(kept)} (merged within {max_angle} deg'
              f"{', symmetry equivalent matrices included' if symmetry else ''})")
        for name, others in merged.items():
            if others:
                print(f"   {name} <- " + ', '.join(f'{n} ({a:.2f} deg)' if a < max_angle else f'{n} (symmetry equivalent)'
                                                   for n, a in others))
    return {'ub_matrices': {n: ub.tolist() for n, ub in kept.items()},
            'merged': {n: [o for o, _ in others] for n, others in merged.items()}}


def ub_map_config(cfg: dict, name: str, newfile=None, fit_folder=None, overwrite: bool = True) -> dict:
    """configuration of the check and refinement of the UB matrix `name` (of ub_matrices) on all the images of the
    map selected by cfg: usepreviousUB [name], skipindexing (no indexing from scratch): each image whose matching
    rate with this matrix is at least MinimumMatchingRate (dict_indexrefine) is refined (1 grain, .fit file),
    the others are not indexed: first maps of the grain of this orientation.
    newfile: default <configs>/indexing_ub_<name>.yaml; fit_folder: default <folder of fit_folder>/ub_<name>.
    Returns the new configuration (run it with MW.run_step())"""
    from LaueTools import indexing_batch as IB
    if name not in (cfg['ub_matrices'] or {}):
        raise KeyError(f"UB matrix '{name}' not found in ub_matrices")
    newfile = Path(newfile) if newfile else Path(cfg['configfile']).parent / f'indexing_ub_{name}.yaml'
    fit_folder = Path(fit_folder) if fit_folder else Path(cfg['output']['fit_folder']).parent / f'ub_{name}'
    return IB.save_config(cfg, newfile, overwrite=overwrite, usepreviousUB=[name], skipindexing=True,
                          fit_folder=str(fit_folder), nbGrainstoFind=1, solution_criteria=None)


def plot_ub_maps(fits_by_name: dict, cfg: dict, min_nb_spots: int = 0, live: bool = False, ncols_max: int = 4):
    """maps of the UB matrices checked over the whole map (results of MW.run_step() with ub_map_config()):
    nb of indexed spots for each matrix, and the matrix giving the most indexed spots at each map point (points
    with at least min_nb_spots). Returns the figure

    live=True (ipympl 'widget' backend and ipywidgets): figure displayed as a live canvas, hover shows the image
    index and the nb of indexed spots of each matrix at that point, a range slider (vmin, vmax) for each map and a
    slider of min_nb_spots for the map of the best matrix; maps on rows of at most ncols_max panels.
    live=False: the figure is returned, to display with plt.show() or IB.show_figure(fig) (static image)"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from LaueTools import map_workflow as MW
    nfast, nslow = cfg['scan']['mapdims']
    names = list(fits_by_name)
    nbs = np.full((len(names), nslow, nfast), np.nan)
    for k, name in enumerate(names):
        for res in fits_by_name[name] or []:
            if 0 <= res[0] < nfast * nslow:
                rows = result_rows(res)
                nbs[k].flat[res[0]] = rows[0, 9] if len(rows) else 0
    filled = np.where(np.isnan(nbs), -1, nbs)

    def best_map(minspots):
        best = np.argmax(filled, axis=0).astype(float) if len(names) else np.full((nslow, nfast), np.nan)
        best[(np.max(filled, axis=0) < max(minspots, 1))] = np.nan
        return best

    best = best_map(min_nb_spots)
    if live:
        try:
            import ipywidgets as widgets
            from IPython.display import display
            live = 'ipympl' in plt.get_backend() or 'widget' in plt.get_backend().lower()
        except ImportError:
            live = False
        if not live:
            GT.printyellow("live=True needs ipywidgets and the ipympl backend (%matplotlib widget): static figure")
    npanels = len(names) + 1
    ncols = min(npanels, ncols_max) if live else npanels
    nrows = -(-npanels // ncols)
    with plt.ioff() if live else _nullcontext():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[npanels:]:
        ax.set_visible(False)
    for k, name in enumerate(names):
        MW.plot_map(nbs[k], cfg, f'{name}: nb of indexed spots', ax=axes[k])
    cmap = ListedColormap(plt.get_cmap('tab10').colors[:max(len(names), 1)])
    axbest = axes[len(names)]
    MW.plot_map(best, cfg, f'matrix with the most indexed spots (>= {max(min_nb_spots, 1)})', ax=axbest,
                cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, colorbar=False)
    handles = [plt.Line2D([], [], marker='s', ls='', color=cmap(k), label=n) for k, n in enumerate(names)]
    axbest.legend(handles=handles, fontsize=7, loc='upper right')

    # hover: image index (and motors) + nb of indexed spots of each matrix at this point, best matrix
    transposed = MW.map_transposed(cfg)
    base = MW._map_format_coord(cfg, None, transposed)

    def format_coord(x, y):
        txt = base(x, y)
        if not txt:
            return ''
        i, j = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        row, col = (i, j) if transposed else (j, i)
        vals = ', '.join(f"{n} {'-' if np.isnan(nbs[k, row, col]) else int(nbs[k, row, col])}"
                         for k, n in enumerate(names))
        kbest = best_now[0][row, col]
        return f"{txt} | {vals} | best: {'-' if np.isnan(kbest) else names[int(kbest)]}"

    best_now = [best]
    for ax in axes[:npanels]:
        ax.format_coord = format_coord
    for k, name in enumerate(names):
        n = int(np.sum(best == k))
        print(f'{name}: best matrix at {n} points, '
              f'{int(np.sum(nbs[k] >= max(min_nb_spots, 1)))} points with >= {max(min_nb_spots, 1)} indexed spots')
    fig.tight_layout()
    if not live:
        return fig

    # sliders: (vmin, vmax) of each map, min nb of spots of the map of the best matrix
    vmax_all = int(np.nanmax(nbs)) if np.any(np.isfinite(nbs)) else 1
    layout = widgets.Layout(width='520px')
    style = {'description_width': '160px'}
    sliders = []
    for k, name in enumerate(names):
        im = axes[k].images[0]
        lo, hi = im.get_clim()
        slider = widgets.IntRangeSlider(value=[int(np.floor(lo)), int(np.ceil(hi))], min=0, max=max(vmax_all, 1),
                                        description=f'{name} vmin, vmax', layout=layout, style=style,
                                        continuous_update=False)

        def on_range(change, im=im):
            im.set_clim(*change['new'])
            fig.canvas.draw_idle()
        slider.observe(on_range, names='value')
        sliders.append(slider)
    minslider = widgets.IntSlider(value=int(min_nb_spots), min=0, max=max(vmax_all, 1),
                                  description='best: min nb of spots', layout=layout, style=style,
                                  continuous_update=False)

    def on_min(change):
        best_now[0] = best_map(change['new'])
        shown = best_now[0].T if transposed else best_now[0]
        axbest.images[0].set_data(shown)
        axbest.set_title(f"matrix with the most indexed spots (>= {max(change['new'], 1)})", fontsize=9)
        fig.canvas.draw_idle()
    minslider.observe(on_min, names='value')
    fig.canvas.header_visible = False
    display(widgets.VBox([fig.canvas, widgets.VBox(sliders + [minslider])]))
    return fig


def print_seed_table(table: List[dict], validated: Optional[set] = None):
    """seeds: image index, position, nb of spots, matching rate, orientation group (* validated)"""
    withcrit = any(t.get('criteria') is not None for t in table)
    withmat = any(t.get('matrix') for t in table)
    width = max([len(str(t.get('matrix') or '')) for t in table] + [6])
    print('image   fast  slow  spots  MR(%)  orientation' + (f"  {'matrix':<{width}}" if withmat else '')
          + ('  criteria (nb of spots of the steps, mean pixel deviation of the last step)' if withcrit else ''))
    for t in table:
        mark = '' if validated is None else ('  *' if t['image_index'] in validated else '')
        crit = ''
        if withcrit and t.get('criteria') is not None:
            crit = ('  pass ' if t['criteria'] else '  FAIL ') + ' '.join('-' if n is None else str(n)
                                                                         for n in t['criteria_nb'])
            if t.get('criteria_pixdev') is not None:
                crit += f"  {t['criteria_pixdev']:.2f} px"
        mat = f"  {str(t.get('matrix') or ''):<{width}}" if withmat else ''
        print(f"{t['image_index']:6d} {t['fast']:5d} {t['slow']:5d} {t['nb_spots']:6d} {t['matching_rate']:6.1f}  "
              f"{t['orientation'] if t['orientation'] >= 0 else '-':>6}{mat}{crit}{mark}")
    import collections
    count = collections.Counter(t['orientation'] for t in table if t['orientation'] >= 0)
    if withcrit:
        print(f"criteria passed by {sum(1 for t in table if t.get('criteria'))} seeds")
    print(f"{len(table)} seeds, {sum(count.values())} indexed, {len(count)} orientation(s): "
          + ', '.join(f'#{k}: {n} seeds' for k, n in sorted(count.items())))


def validate_seeds(table: List[dict], min_nb_spots: int = 0, min_matching_rate: float = 0.,
                   orientations=None, exclude=(), include=(), require_criteria: bool = True) -> set:
    """image indices of the seeds validated: criteria of a correct solution passed (if checked and
    require_criteria), at least min_nb_spots indexed spots and min_matching_rate %, orientation group in
    `orientations` (None: all), not in `exclude`; `include`: seeds validated whatever their values"""
    ok = {t['image_index'] for t in table if t['UB'] is not None and t['nb_spots'] >= min_nb_spots
          and t['matching_rate'] >= min_matching_rate and (orientations is None or t['orientation'] in orientations)
          and not (require_criteria and t.get('criteria') is False)}
    return (ok - set(exclude)) | ({int(i) for i in include} & {t['image_index'] for t in table if t['UB'] is not None})


def plot_seeds(table: List[dict], cfg: dict, validated: Optional[set] = None, ax=None):
    """map of the seeds: color = orientation group, size = nb of indexed spots, black cross = not validated"""
    import matplotlib.pyplot as plt
    from LaueTools import map_workflow as MW
    nfast, nslow = cfg['scan']['mapdims']
    data = np.full((nslow, nfast), np.nan)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    MW.plot_map(data, cfg, 'seeds: color = orientation, size = nb of spots, x = not validated', ax=ax,
                colorbar=False)
    transposed = MW.map_transposed(cfg)
    cmap = plt.get_cmap('tab10')
    nmax = max([t['nb_spots'] for t in table] + [1])
    for t in table:
        x, y = (t['slow'], t['fast']) if transposed else (t['fast'], t['slow'])
        color = 'lightgrey' if t['orientation'] < 0 else cmap(t['orientation'] % 10)
        ax.scatter([x], [y], s=10 + 120 * t['nb_spots'] / nmax, color=color, edgecolors='k', linewidths=0.3)
        if validated is not None and t['image_index'] not in validated:
            ax.scatter([x], [y], marker='x', s=40, color='k')
    return ax


def save_validated_seeds(table: List[dict], validated: set, cfg_seeds: dict, path=None) -> Path:
    """write the validated seeds (UB matrices, nb of spots, matching rates, folder of their .fit files) in a JSON
    file (default: <configs>/validated_seeds.json), to give as seeds_file in run: neighbours of indexing.yaml"""
    import json
    path = Path(path) if path else Path(cfg_seeds['configfile']).parent / 'validated_seeds.json'
    seeds = {str(t['image_index']): {'UB': np.asarray(t['UB']).tolist(), 'nb_spots': t['nb_spots'],
                                     'matching_rate': t['matching_rate'], 'orientation': t['orientation']}
             for t in table if t['image_index'] in validated and t['UB'] is not None}
    from LaueTools import indexing_batch as IB
    content = {'seeds_fit_folder': str(cfg_seeds['output']['fit_folder']),
               'fit_layout': IB.fitfiles_layout(cfg_seeds), 'prefix': cfg_seeds['scan']['prefix'],
               'nbdigits': cfg_seeds['scan']['nbdigits'], 'seeds': seeds}
    path.write_text(json.dumps(content, indent=1))
    GT.printgreen(f'{len(seeds)} validated seeds written in {path}')
    return path


def load_validated_seeds(path) -> dict:
    import json
    content = json.loads(Path(path).read_text())
    content['seeds'] = {int(k): v for k, v in content['seeds'].items()}
    return content


def use_validated_seeds(validated: dict, seeds, store: UBStore):
    """UB matrices of the validated seeds written in the library of the UB store (orientations checked by all the
    points, even if a seed is not accepted with the map parameters)"""
    for i in seeds:
        v = validated['seeds'][int(i)]
        row = np.r_[np.asarray(v['UB'], dtype=float).ravel(), v['nb_spots'], v['matching_rate'], 1.][None]
        store._save(store.libfolder / f'{int(i)}.npy', row)


# --------------------------------------------------------------------------------------
#  TESTS AND RESULTS
# --------------------------------------------------------------------------------------
def check_references(listfiles: List[str], params, nb_images: int = 8, image_indices=None, criteria=None,
                     max_pixdev: Optional[float] = None) -> dict:
    """nb of indexed spots given by each reference UB matrix (reference_UBs) on a few images spread over the map
    (in this process, ~2 s per image and matrix), to choose min_nb_matches. No .fit file is kept.
    criteria (or params.solution_criteria): criteria of a correct solution checked instead (IB.verify_solution())
    Returns {reference name: {image index: [nb of indexed spots, matching rate]}} (criteria: verifications)"""
    from LaueTools import indexing_batch as IB
    byindex = {GT.getfileindex(Path(f).name): str(f) for f in listfiles}
    if image_indices is None:
        keys = sorted(byindex)
        image_indices = [keys[i] for i in np.linspace(0, len(keys) - 1, min(nb_images, len(keys))).astype(int)]
    trials = Path(params.fit_folder) / 'ub_store' / 'trials' / f'check_{os.getpid()}'
    trial = dataclasses.replace(check_params(params), usepreviousUB=False, neighbours=None, skipindexing=True,
                                fit_folder=str(trials), fit_nbfiles_per_folder=None,
                                useinternalmultiprocessing=False, verboselevel=0)
    print(check_description(params))
    nb = settings(params.neighbours)
    out = {}
    criteria = criteria if criteria is not None else params.solution_criteria
    if criteria:   # criteria of a correct solution
        for name, ub in settings(params.neighbours)['reference_UBs']:
            out[name] = {}
            for index in image_indices:
                v = IB.verify_solution(byindex[index], ub, params, criteria, max_pixdev)
                out[name][index] = v
                print(f'{name}, image {index}: ' + IB.criteria_info(0, v)[3:])
        return out
    for name, ub in settings(params.neighbours)['reference_UBs']:
        out[name] = {}
        for index in image_indices:
            rows = result_rows(IB.index_refine(byindex[index], dataclasses.replace(trial, usepreviousUB=[ub])))
            out[name][index] = rows[0, 9:].tolist() if len(rows) else [0, 0.]
        print(f'{name}: nb of indexed spots (matching rate %) ' + ', '.join(
            f'image {i}: {n:.0f} ({mr:.1f})' + ('' if accepted(np.r_[np.zeros(9), n, mr], nb)[0] else ' rejected')
            for i, (n, mr) in out[name].items()))
    shutil.rmtree(trials, ignore_errors=True)
    return out


SOURCES = ['not indexed', 'seed (from scratch)', 'previous image', 'neighbour', 'library', 'reference UB',
           'from scratch (no guess accepted)', 'several grains', 'unknown (store of a former version)']


def source_code(info: str, rows) -> int:
    """index in SOURCES of how an image was indexed, from the info of its result"""
    if not len(rows):
        return 0
    if 'seed, indexed from scratch' in info:
        return 1
    if 'no UB guess accepted' in info:
        return 6
    if 'UB guesses accepted' in info:
        return 7
    return next((k for k, key in ((2, 'previous image'), (3, 'neighbour'), (4, 'library'), (5, 'reference UB'))
                 if f'accepted: {key}' in info), 8)


_SOURCE_INFO = {1: 'seed, indexed from scratch.', 2: 'UB guess accepted: previous image.',
                3: 'UB guess accepted: neighbour.', 4: 'UB guess accepted: library.',
                5: 'UB guess accepted: reference UB.', 6: 'no UB guess accepted, indexed from scratch.',
                7: 'UB guesses accepted.'}


def source_map(allresults: list, mapdims) -> np.ndarray:
    """2D map (slow, fast) of the index in SOURCES of how each image was indexed (nan: not treated)"""
    out = np.full((mapdims[1], mapdims[0]), np.nan)
    for res in allresults:
        if 0 <= res[0] < mapdims[0] * mapdims[1]:
            out.flat[res[0]] = source_code(str(res[3]) if len(res) > 3 else '', result_rows(res))
    return out


def load_store_results(cfg: dict) -> list:
    """results of the images already treated, read from the UB store of the fit_folder of cfg: e.g. after an
    interrupted run (SLURM job cancelled or time limit reached), when the results file is not written.
    Same format as the results of run_step('indexing', ...) (for IB.summarize_results(), plot_sources() ...);
    the .fit files of these images are in fit_folder (MW.load_grains() ...)"""
    store = UBStore(cfg['output']['fit_folder'])
    allresults = []
    files = [f for f in store.folder.glob('*.npy') if f.stem.isdigit()]
    for f in tqdm(files, desc='UB store', mininterval=0.5):
        try:
            rows = np.atleast_2d(np.load(f))
        except (ValueError, OSError):
            continue
        rows = rows if rows.size else np.zeros((0, 11))
        code = int(rows[0, 11]) if len(rows) and rows.shape[1] > 11 else None
        info = _SOURCE_INFO.get(code, 'read from the UB store.') if len(rows) else 'nothing found.'
        allresults.append([int(f.stem), [(g, r[:9].reshape(3, 3)) for g, r in enumerate(rows)],
                           [(g, [int(r[9]), float(r[10])]) for g, r in enumerate(rows)], info])
    allresults.sort(key=lambda r: r[0])
    nfast, nslow = cfg['scan']['mapdims']
    print(f"{len(allresults)} images / {nfast * nslow} treated (UB store {store.folder}), "
          f"{sum(1 for r in allresults if r[1])} indexed")
    return allresults


def plot_sources(allresults: list, cfg: dict, ax=None):
    """map of how each image was indexed (seed, UB of the previous image, of a neighbour, library, from scratch)"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from LaueTools import map_workflow as MW
    data = source_map(allresults, cfg['scan']['mapdims'])
    colors = ['lightgrey', 'black', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:red', 'tab:brown',
              'tab:olive']
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    MW.plot_map(data, cfg, 'how each image was indexed', ax=ax, cmap=ListedColormap(colors), vmin=-0.5,
                vmax=len(colors) - 0.5, colorbar=False)
    present = [k for k in range(len(SOURCES)) if np.any(data == k)]
    ax.legend(handles=[Patch(color=colors[k], label=f'{SOURCES[k]} ({int(np.sum(data == k))})') for k in present],
              loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    return ax


def distinct_orientations(cfg: dict, min_nb_spots: int = 100, max_angle: float = 2., nmax: int = 6,
                          grainindex: int = 0, verbose: bool = True) -> List[dict]:
    """distinct orientations of the .fit files of the fit_folder of cfg (e.g. a previous indexation of the map), to
    use as reference_UBs: images with at least min_nb_spots indexed spots grouped (greedy, best images first) when
    their UB matrices are closer than max_angle (deg, raw angle: symmetry equivalent matrices are distinct)
    Returns the nmax largest groups [{'name': 'UB_<image index>', 'image_index', 'UB', 'nb_spots', 'nb_images'}]"""
    from LaueTools import map_workflow as MW
    from LaueTools import orientationclustering as OC
    ffs = MW.load_fitfiles(cfg, grainindex=grainindex)
    UB = np.asarray(ffs.UB).reshape(-1, 3, 3)
    nb = np.asarray(ffs.NumberOfIndexedSpots, dtype=float).ravel()
    idx = np.flatnonzero(np.isfinite(UB[:, 0, 0]) & (np.nan_to_num(nb) >= min_nb_spots))
    rot = OC.orthonormalize(UB[idx]) if len(idx) else np.zeros((0, 3, 3))
    groups = []   # [position of the best image, nb of images]
    reps = np.zeros((0, 3, 3))
    for k in np.argsort(-nb[idx], kind='stable'):
        if len(reps):
            ang, _ = OC.misorientation(np.broadcast_to(rot[k], reps.shape), reps, np.eye(3)[None])
            j = int(np.argmin(ang))
            if ang[j] < max_angle:
                groups[j][1] += 1
                continue
        groups.append([k, 1])
        reps = np.concatenate([reps, rot[k][None]])
    groups.sort(key=lambda g: -g[1])
    out = [{'name': f'UB_{int(idx[k])}', 'image_index': int(idx[k]), 'UB': UB[idx[k]].tolist(),
            'nb_spots': int(nb[idx[k]]), 'nb_images': n} for k, n in groups[:nmax]]
    if verbose:
        print(f"{len(idx)} images with >= {min_nb_spots} indexed spots in {cfg['output']['fit_folder']}: "
              f"{len(groups)} distinct orientations (within {max_angle} deg)")
        for o in out:
            print(f"   {o['name']}: {o['nb_spots']} spots, {o['nb_images']} images")
    return out
