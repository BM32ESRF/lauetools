#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
quickdaxm: depth of the Laue spots of a DAXM wire scan, without reconstruction of the images.

Principle
---------
During a wire scan, the intensity of a Laue spot drops to zero while a wire crosses the ray going
from the scattering grain to the pixel. For each spot:

1. intensity profile (function of image index) of the spot core minus the local background taken
   on the border of a small box around the spot. Fluorescence varies slowly on the detector: it is
   the same in the core and on the border, so the smooth fluorescence dips cancel and only the
   sharp dips of the scattering signal remain.
2. dips = image ranges where the profile is extinguished (below a fraction of its plateau), each
   limited by an entry and an exit edge located at half height (sub image accuracy).
3. the wire position at an edge is the position where the wire is tangent to the ray. With the wires
   trajectory parameters of a .calib file (same geometrical model as the wires calibration and the
   reconstruction: LaueTools.Daxm.classes.wire.CircularWire), each edge gives the position ybeam of
   the source along the beam. Entry and exit edges give two independent values that must agree.

The empirical lines 'image index of dip = f(Y pixel)' of the former quick daxm notebooks
(parameters WP) are given by the same model: WireGeometry.shadow_lines() and wp_parameters().

Units: pixels (X, Y LaueTools convention), image index, mm for wire positions, microns for ybeam
(position along the beam, >0 downstream i.e. deeper in the sample, 0 at the reference point of the
wires calibration) and depth = ybeam * sin(sample tilt) (below the sample surface).

Main functions: collect_profiles(), find_dips(), analyse(), run(), write_cor(),
write_depth_slices(), plot_profile(), plot_abacus().
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import json
import multiprocessing
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy import ndimage

import LaueTools.Daxm.modules.geometry as geom
from LaueTools.Daxm.classes.calibration import CalibManager
from LaueTools.Daxm.classes.wire import CircularWire
from LaueTools import IOLaueTools as IOLT
from LaueTools import IOimagefile as IOimage
from LaueTools import dict_LaueTools as DictLT

DEG = np.pi / 180.

# default parameters of the analysis (section 'quickdaxm' of the daxm YAML configuration file)
DEFAULT_PARAMS = {
    'peaklist': 'auto',        # 'auto' (spots found in the scan) or path of a .dat, .cor or .fit file
    'spots_threshold': 300.,   # [auto] min intensity of a spot above local background
    'spots_max_number': 2000,  # [auto] strongest spots kept
    'boxsize': 5,              # half size (pixels) of the box around a spot; local background on its border
    'boxsize_probe': 1,        # half size (pixels) of the probe box on the spot whose mean is the signal: 1 (3 x 3
                               # pixels), or [half size along X, half size along Y]: [2, 0] is a 5 x 1 pixels box
    'low_fraction': 0.3,       # dip: signal below this fraction of the plateau
    'min_edge_fraction': 0.5,  # edge height / plateau must be larger (rejects intensity fluctuations)
    'min_snr': 8.,             # plateau / noise must be larger
    'min_signal': 50.,         # plateau (counts above local background) must be larger
    'min_width': 4,            # min nb of images of a dip
    'edge_window': 20,         # nb of images outside a dip used to estimate the plateau
    'ybeam_range': [-100., 200.],  # microns: accepted source positions along the beam
    'max_mismatch': 10.,       # microns: max difference of ybeam given by entry and exit edges
    'skip_images': 0,          # first images of the scan not used for the edges of the dips (wire position not reliable
                               # at the start of the scans of a grid after the return of the wire, see
                               # mismatch_vs_image()): a dip starting there has only its exit edge
    'onset_fraction': 0.1,     # onset of an edge (start of the step, shallowest sources): crossings of the levels
                               # onset_fraction and 1 - onset_fraction of the edge height
}

# former names of parameters
PARAM_ALIASES = {'coresize': 'boxsize_probe'}


def full_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """DEFAULT_PARAMS updated by params (former names of PARAM_ALIASES accepted)"""
    params = {PARAM_ALIASES.get(key, key): value for key, value in (params or {}).items()}
    return {**DEFAULT_PARAMS, **params}


# 20% - 80% width of a step blurred by a gaussian of standard deviation 1
_WIDTH2080_PER_SIGMA = 1.683


# --------------------------------------------------------------------------------------
#  SCAN: detector, wire positions
# --------------------------------------------------------------------------------------
def read_detector(detfile: str, pixelsize: Optional[float] = None) -> List[float]:
    """detector parameters [distance, xcen, ycen, xbet, xgam, pixelsize] (angles in radians, as in
    PointScan.get_ccd_params()) from a .det file"""
    params, _ = IOLT.readfile_det(detfile, nbCCDparameters=6, verbose=0)
    return [float(params[0]), float(params[1]), float(params[2]), float(params[3]) * DEG,
            float(params[4]) * DEG, float(pixelsize or params[5])]


def read_wire_positions(hdf5_file: str, scan_number: int, motor: str = 'zf',
                        node: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """wire positions (mm) of the images of a BLISS scan, and monitor values (or None)

    :param node: name of the scan in the file (default '<scan_number>.1', as in the hdf5 file of a
        dataset; '<sample>_<dataset>_<scan_number>.1' in the hdf5 file of a proposal)
    """
    with h5py.File(hdf5_file, 'r', locking=False) as f:
        names = [name for name in (node, f'{int(scan_number)}.1') if name]
        found = [name for name in names if name in f]
        if not found:
            raise KeyError(f'scan {names} not found in {hdf5_file}')
        scannode = f[found[0]]
        if f'measurement/{motor}' in scannode:
            positions = scannode[f'measurement/{motor}'][()]
        else:
            positions = scannode[f'instrument/positioners/{motor}'][()]
        monitor = scannode['measurement/mon'][()] if 'measurement/mon' in scannode else None
    return np.asarray(positions, dtype=float), monitor


def wire_positions_of_scan(scan: Dict[str, Any]) -> np.ndarray:
    """wire positions (mm) of the images of a scan (section 'scan' of the configuration): read in
    the BLISS hdf5 file, else evenly spaced from wire_start to wire_stop"""
    if scan.get('hdf5_file') and scan.get('scan_number') is not None and os.path.exists(scan['hdf5_file']):
        try:
            return read_wire_positions(scan['hdf5_file'], scan['scan_number'], scan.get('wire_motor', 'zf'),
                                       scan.get('hdf5_node'))[0]
        except (KeyError, OSError) as exc:
            if scan.get('wire_start') is None or scan.get('wire_stop') is None:
                raise
            print(f'wire positions not read ({exc}): evenly spaced from wire_start to wire_stop')
    return np.linspace(float(scan['wire_start']), float(scan['wire_stop']), int(scan['nb_images']))


# --------------------------------------------------------------------------------------
#  GEOMETRY: wires of a .calib file <-> image index of the shadow on a pixel
# --------------------------------------------------------------------------------------
class WireGeometry:
    """wires (CircularWire), detector and wire positions of a scan: where and when a ray going from
    the beam to a pixel is masked by a wire"""

    def __init__(self, wires: Sequence, ccd_params: Sequence[float], wire_positions: Sequence[float],
                 sample_tilt: float = 40., excluded_wires: Sequence[int] = ()):
        self.wires = list(wires)
        # wires whose dips give no depth (e.g. a wire badly calibrated, seen during a part of the scan only): their
        # dips are still assigned to them (not to another wire), without ybeam
        self.excluded_wires = sorted(int(w) for w in excluded_wires)
        self.ccd_params = list(ccd_params)
        self.positions = np.asarray(wire_positions, dtype=float)
        self.sample_tilt = float(sample_tilt)
        self._index = np.arange(len(self.positions), dtype=float)
        # linear motor motion (extrapolation outside the scan)
        self._slope, self._intercept = np.polyfit(self._index, self.positions, 1)
        self.increasing = self._slope > 0

    @classmethod
    def from_files(cls, calibfile: str, detfile: str, wire_positions: Sequence[float], yref: float = 0.,
                   pixelsize: Optional[float] = None, sample_tilt: float = 40.):
        """wires of a .calib file (wires calibration, with its optional list 'excluded_wires') and detector of a
        .det file"""
        wires = CalibManager(calibfile, yref).get_wires()
        return cls(wires, read_detector(detfile, pixelsize), wire_positions, sample_tilt,
                   excluded_wires=read_excluded_wires(calibfile))

    @property
    def nb_wires(self) -> int:
        return len(self.wires)

    @property
    def step(self) -> float:
        """wire step (mm / image, signed)"""
        return float(self._slope)

    def index_to_position(self, index):
        """wire position (mm) at (non integer) image index"""
        index = np.asarray(index, dtype=float)
        inside = np.interp(index, self._index, self.positions)
        outside = (index < 0) | (index > self._index[-1])
        return np.where(outside, self._slope * index + self._intercept, inside)

    def position_to_index(self, position):
        """(non integer) image index at wire position (mm), extrapolated outside the scan"""
        return (np.asarray(position, dtype=float) - self._intercept) / self._slope

    def pixel_point(self, x: float, y: float) -> np.ndarray:
        """3D position (mm, y // beam) of pixel X, Y"""
        return np.asarray(geom.transf_pix_to_coo(self.ccd_params, x, y), dtype=float)

    def tangent_positions(self, x: float, y: float, ybeam_um) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """wire positions (mm) where each wire starts to mask (entry), is centered on (axis) and
        stops to mask (exit), in the order of the scan, the ray from source(s) at ybeam_um to pixel X, Y

        :return: 3 arrays of shape (nb wires, nb ybeam)
        """
        ysrc = np.atleast_1d(np.asarray(ybeam_um, dtype=float)) * 1e-3
        pcam = self.pixel_point(x, y)
        p_entry, p_axis, p_exit = [], [], []
        for wire in self.wires:
            pfront, pback = wire.intersect_ray_fronts(ysrc, pcam)
            pfront, pback = np.atleast_1d(pfront), np.atleast_1d(pback)
            plow, phigh = np.minimum(pfront, pback), np.maximum(pfront, pback)
            p_entry.append(plow if self.increasing else phigh)
            p_exit.append(phigh if self.increasing else plow)
            p_axis.append(0.5 * (plow + phigh))
        return np.array(p_entry), np.array(p_axis), np.array(p_exit)

    def shadow_indices(self, x: float, y: float, ybeam_um: float = 0.) -> np.ndarray:
        """image indices (entry, center, exit) of the shadow of each wire on pixel X, Y for a source
        at ybeam_um. Shape (nb wires, 3). Indices may be outside the scan."""
        return np.array([self.position_to_index(p[:, 0]) for p in self.tangent_positions(x, y, ybeam_um)]).T

    def shadow_lines(self, ys: Optional[Sequence[float]] = None, x: Optional[float] = None,
                     ybeam_um: float = 0.) -> Dict[str, np.ndarray]:
        """image index of the dip (entry, center, exit) as a function of pixel Y at pixel X (default:
        detector center), for each wire: the 'abacus' of the quick daxm notebooks

        :return: dict 'Y' (nb Y), 'entry', 'center', 'exit' (nb wires, nb Y)
        """
        if ys is None:
            ys = np.arange(0, DEFAULT_FRAME_Y, 20)
        x = self.ccd_params[1] if x is None else x
        lines = np.array([self.shadow_indices(x, y, ybeam_um) for y in ys])  # (nb Y, nb wires, 3)
        return {'Y': np.asarray(ys, dtype=float), 'entry': lines[:, :, 0].T, 'center': lines[:, :, 1].T,
                'exit': lines[:, :, 2].T}

    def wp_parameters(self, x: Optional[float] = None, ybeam_um: float = 0., yrange=(0, 2000)) -> List[List[float]]:
        """straight line approximation 'image index of dip center = a / 450 * Y + b' of each wire, i.e.
        the parameters [a, b] of the list WP of the former quick daxm notebooks, computed from the
        wires calibration (one [a, b] per wire, in the order of the .calib file)"""
        lines = self.shadow_lines(np.linspace(yrange[0], yrange[1], 21), x, ybeam_um)
        res = []
        for center in lines['center']:
            slope, intercept = np.polyfit(lines['Y'], center, 1)
            res.append([float(slope * 450.), float(intercept)])
        return res

    def ybeam_of_edge(self, x: float, y: float, index: float, kind: str, ybeam_grid_um: np.ndarray,
                      tangents: Optional[tuple] = None) -> np.ndarray:
        """ybeam (microns) of the source of a ray to pixel X, Y masked from (kind='entry') or until
        (kind='exit') image index, for each wire (nan if outside ybeam_grid_um)"""
        if tangents is None:
            tangents = self.tangent_positions(x, y, ybeam_grid_um)
        curves = tangents[0] if kind == 'entry' else tangents[2]
        pedge = float(self.index_to_position(index))
        res = np.full(self.nb_wires, np.nan)
        for iw, curve in enumerate(curves):
            order = np.argsort(curve)
            if curve[order[0]] <= pedge <= curve[order[-1]]:
                res[iw] = np.interp(pedge, curve[order], ybeam_grid_um[order])
        return res

    def scattering_angles(self, x, y, ybeam_um=0.) -> Tuple[np.ndarray, np.ndarray]:
        """2theta and chi (degrees, LaueTools convention) of rays from sources at ybeam_um (scalar or
        one value per spot) to pixels X, Y"""
        pcam = self.pixel_point(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ysrc = np.asarray(ybeam_um, dtype=float) * 1e-3
        ux, uy, uz = pcam[0], pcam[1] - ysrc, pcam[2]
        norm = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
        return np.arccos(uy / norm) / DEG, np.arctan(-ux / uz) / DEG

    def depth(self, ybeam_um):
        """depth below the sample surface (microns) of a source at ybeam_um along the beam"""
        return np.asarray(ybeam_um, dtype=float) * np.sin(self.sample_tilt * DEG)


DEFAULT_FRAME_Y = 2048


# --------------------------------------------------------------------------------------
#  IMAGES: spots, intensity profiles
# --------------------------------------------------------------------------------------
def image_file(scan: Dict[str, Any], imageindex: int) -> Tuple[str, int]:
    """path of the file of image imageindex (0 for the first image of the scan) and index of the
    frame in this file (-1 for a file with a single image)"""
    index = int(imageindex) + int(scan.get('first_image_index', 0))
    nbdigits = int(scan.get('nbdigits', 4))
    representative = f"{scan['prefix']}{0:0{nbdigits}d}.{scan['suffix'].lstrip('.')}"
    if scan['CCDLabel'] in IOimage.STACK_CCDLABELS:
        filename, frame = IOimage.stack_file_and_frame(representative, index,
                                                       int(scan.get('nbframes_per_file', 1)))
    else:
        filename = IOimage.setfilename(representative, index, CCDLabel=scan['CCDLabel'], nbdigits=nbdigits)
        frame = -1
    return os.path.join(scan['image_folder'], filename), frame


def read_image(scan: Dict[str, Any], imageindex: int) -> np.ndarray:
    """image imageindex of the scan (2D array [Y, X])"""
    path, frame = image_file(scan, imageindex)
    return IOimage.readCCDimage(path, CCDLabel=scan['CCDLabel'], stackimageindex=frame)[0]


_WORKER = {}


def _init_worker(scan, boxes):
    _WORKER['scan'] = scan
    _WORKER['boxes'] = boxes


def _max_of_image(imageindex):
    try:
        return read_image(_WORKER['scan'], imageindex).astype(np.float32)
    except Exception:   # missing or unreadable image
        return None


def _profiles_of_image(imageindex):
    """(core mean, border median) of all boxes in one image, or None for a missing image"""
    try:
        data = read_image(_WORKER['scan'], imageindex)
    except Exception:
        return None
    iy, ix, core, border = _WORKER['boxes']
    values = data[iy, ix].astype(np.float64)   # (nb spots, box, box)
    return values[:, core].mean(axis=1), np.median(values[:, border], axis=1)


def _pool_map(function, indices, scan, boxes, nb_cpus, progress_callback=None):
    from LaueTools import generaltools as GT
    nb_cpus = max(1, min(nb_cpus or GT.available_cpus(), len(indices)))
    results = []
    if nb_cpus == 1:
        _init_worker(scan, boxes)
        iterator, pool = map(function, indices), None
    else:
        pool = multiprocessing.Pool(nb_cpus, initializer=_init_worker, initargs=(scan, boxes))
        iterator = pool.imap(function, indices, chunksize=max(1, min(8, len(indices) // (4 * nb_cpus))))
    try:
        for k, res in enumerate(iterator):
            results.append(res)
            if progress_callback is not None:
                progress_callback(k + 1, len(indices))
    finally:
        if pool is not None:
            pool.terminate()
    return results


def max_projection(scan: Dict[str, Any], nb_images_used: int = 40, nb_cpus: Optional[int] = None) -> np.ndarray:
    """max over (about nb_images_used evenly spaced) images of the scan: all spots are visible,
    whatever the position of the wires"""
    step = max(1, int(scan['nb_images']) // nb_images_used)
    images = _pool_map(_max_of_image, list(range(0, int(scan['nb_images']), step)), scan, None, nb_cpus)
    images = [im for im in images if im is not None]
    if not images:
        raise RuntimeError(f"no image read in {scan['image_folder']}")
    return np.max(images, axis=0)


def find_spots(image: np.ndarray, threshold: float = 300., max_number: int = 2000, boxsize: int = 5,
               background_size: int = 15, subpixel: bool = True) -> np.ndarray:
    """spots of an image (e.g. max_projection()): local maxima higher than threshold above the local
    background (median filter), at more than boxsize pixels from the borders

    :param subpixel: X, Y = centroid of the signal above background in 5 x 5 pixels around the maximum (the
        position along the beam depends on pixel Y: ~0.6 micron per pixel), else the pixel of the maximum
    :return: array (nb spots, 3): X, Y (pixels), intensity above background, strongest first
    """
    image = np.asarray(image, dtype=np.float32)
    signal = image - ndimage.median_filter(image, size=background_size)
    labels, nb = ndimage.label(signal > threshold)
    if nb == 0:
        return np.zeros((0, 3))
    index = np.arange(1, nb + 1)
    pos = np.array(ndimage.maximum_position(signal, labels, index))
    amp = np.array(ndimage.maximum(signal, labels, index))
    inside = ((pos[:, 0] > boxsize) & (pos[:, 0] < image.shape[0] - boxsize - 1)
              & (pos[:, 1] > boxsize) & (pos[:, 1] < image.shape[1] - boxsize - 1))
    pos, amp = pos[inside], amp[inside]
    order = np.argsort(amp)[::-1][:max_number]
    pos, amp = pos[order].astype(float), amp[order]
    if subpixel:
        oy, ox = np.mgrid[-2:3, -2:3]
        for k, (iy, ix) in enumerate(pos.astype(int)):
            if 2 <= iy < image.shape[0] - 2 and 2 <= ix < image.shape[1] - 2:
                weights = np.clip(signal[iy - 2:iy + 3, ix - 2:ix + 3], 0., None)
                if weights.sum() > 0:
                    pos[k] = [iy + (weights * oy).sum() / weights.sum(), ix + (weights * ox).sum() / weights.sum()]
    return np.column_stack((pos[:, 1], pos[:, 0], amp)).astype(float)


def read_peaklist(path: str) -> np.ndarray:
    """X, Y, intensity (nb spots, 3) of the spots of a .dat, .cor or .fit file"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.cor':
        res = IOLT.readfile_cor(path)
        return np.column_stack((res[3], res[4], res[5]))
    if ext == '.fit':
        data = np.atleast_2d(IOLT.readfitfile_multigrains(path)[4])
        return np.column_stack((data[:, 7], data[:, 8], data[:, 1]))
    data = np.atleast_2d(IOLT.read_Peaklist(path))
    return np.column_stack((data[:, 0], data[:, 1], data[:, 3]))


def _boxes(xy: np.ndarray, boxsize: int, boxsize_probe, framedim: Tuple[int, int]):
    """pixel indices of the boxes centered on xy and masks of probe (signal) and border (background) pixels of a box

    :param boxsize: half size of the box (its border: local background)
    :param boxsize_probe: half size of the probe box (int), or [half size along X, half size along Y]: e.g. [2, 0]
        is a 5 x 1 pixels box, keeping the edges of the dips sharp (the shadow moves along Y by ~1.7 pixel per micron
        of wire travel)
    """
    if np.ndim(boxsize_probe) == 0:
        core_x = core_y = int(boxsize_probe)
    else:
        core_x, core_y = int(boxsize_probe[0]), int(boxsize_probe[1])
    xy = np.rint(np.asarray(xy, dtype=float)[:, :2]).astype(int)
    xc = np.clip(xy[:, 0], boxsize, framedim[1] - boxsize - 1)
    yc = np.clip(xy[:, 1], boxsize, framedim[0] - boxsize - 1)
    offsets = np.arange(-boxsize, boxsize + 1)
    iy = (yc[:, None, None] + offsets[None, :, None]).repeat(len(offsets), axis=2).reshape(len(xy), -1)
    ix = (xc[:, None, None] + offsets[None, None, :]).repeat(len(offsets), axis=1).reshape(len(xy), -1)
    oy, ox = np.meshgrid(offsets, offsets, indexing='ij')
    core = ((np.abs(oy) <= core_y) & (np.abs(ox) <= core_x)).ravel()
    border = ((np.abs(oy) == boxsize) | (np.abs(ox) == boxsize)).ravel()
    return iy, ix, core, border


def collect_profiles(scan: Dict[str, Any], xy: np.ndarray, boxsize: int = 5, boxsize_probe=1,
                     nb_cpus: Optional[int] = None, progress_callback=None) -> Dict[str, np.ndarray]:
    """intensity profiles of spots over the images of a wire scan, images being read in parallel

    :param scan: image_folder, prefix, suffix, nbdigits, CCDLabel, nb_images, first_image_index
    :param xy: (nb spots, 2) pixel positions X, Y
    :return: dict 'signal' (nb spots, nb images): mean of the core of the box minus 'background',
        'background' (nb spots, nb images): median of the border of the box (fluorescence...),
        'missing' (nb images): True for missing or unreadable images (nan in the profiles)
    """
    framedim = DictLT.dict_CCD[scan['CCDLabel']][0]
    boxes = _boxes(xy, boxsize, boxsize_probe, framedim)
    nb_images = int(scan['nb_images'])
    t0 = time.time()
    results = _pool_map(_profiles_of_image, list(range(nb_images)), scan, boxes, nb_cpus, progress_callback)
    signal = np.full((len(xy), nb_images), np.nan)
    background = np.full((len(xy), nb_images), np.nan)
    missing = np.ones(nb_images, dtype=bool)
    for k, res in enumerate(results):
        if res is not None:
            missing[k] = False
            background[:, k] = res[1]
            signal[:, k] = res[0] - res[1]
    print(f'profiles of {len(xy)} spots in {nb_images} images collected in {time.time() - t0:.1f} s'
          + (f' ({missing.sum()} missing images)' if missing.any() else ''), flush=True)
    return {'signal': signal, 'background': background, 'missing': missing}


# --------------------------------------------------------------------------------------
#  DIPS of an intensity profile
# --------------------------------------------------------------------------------------
def _edge(profile: np.ndarray, inside: int, start: int, direction: int, floor: float, window: int,
          onset_fraction: float = 0.1):
    """edge of a dip, going outwards (direction -1 or +1) from image inside (middle of the dip);
    start: last image of the dip in this direction

    :return: (position at half height, 20-80 % width, plateau level, (positions at the levels onset_fraction and
        1 - onset_fraction of the edge height)) or None if the profile does not come back to the plateau within
        window images after start
    """
    stop = start + direction * (window + 1)
    stop = max(-1, stop) if direction < 0 else min(len(profile), stop)
    indices = np.arange(inside, stop, direction)
    nboutside = abs(stop - start) - 1
    if nboutside < 2:
        return None
    values = profile[indices]
    plateau = np.percentile(values[-nboutside:], 75)
    height = plateau - floor
    if height <= 0:
        return None
    crossings = []
    for level in (0.2, 0.5, 0.8, onset_fraction, 1. - onset_fraction):
        above = np.flatnonzero(values >= floor + level * height)
        if len(above) == 0 or above[0] == 0:
            return None
        k = above[0]
        fraction = (floor + level * height - values[k - 1]) / (values[k] - values[k - 1])
        crossings.append(indices[k - 1] + direction * fraction)
    return crossings[1], abs(crossings[2] - crossings[0]), plateau, (crossings[3], crossings[4])


def find_dips(signal: np.ndarray, low_fraction: float = 0.3, min_edge_fraction: float = 0.5,
              min_snr: float = 8., min_signal: float = 50., min_width: int = 4,
              edge_window: int = 20, onset_fraction: float = 0.1, skip_images: int = 0) -> List[Dict[str, float]]:
    """dips (extinctions by a wire) of the intensity profile of a spot (local background subtracted)

    A dip is a range of images where the signal is lower than low_fraction * plateau (plateau: 90th
    percentile of the profile), limited by edges higher than min_edge_fraction * plateau. Smooth
    and partial dips (fluorescence, intensity fluctuations) do not fulfil these conditions.

    :return: list of dict with keys:
        'entry', 'exit': image index (float) of the edges at half height (nan for a dip cut by the
            start or the end of the scan)
        'entry_width', 'exit_width': standard deviation (nb images) of the edges (gaussian blur)
        'entry_lo', 'entry_hi', 'exit_lo', 'exit_hi': image index of the edges at the levels onset_fraction (lo) and
            1 - onset_fraction (hi) of their height (one of them is the onset of the edge, see analyse())
        'floor', 'plateau': signal in the dip and outside
    """
    if skip_images and skip_images > 0:   # profile from image skip_images: indices shifted back at the end
        dips = find_dips(np.asarray(signal, dtype=float)[int(skip_images):], low_fraction, min_edge_fraction,
                         min_snr, min_signal, min_width, edge_window, onset_fraction, 0)
        for dip in dips:
            for key in ('entry', 'exit', 'entry_lo', 'entry_hi', 'exit_lo', 'exit_hi'):
                dip[key] = dip[key] + int(skip_images)
        return dips
    s = np.asarray(signal, dtype=float)
    finite = np.isfinite(s)
    if finite.sum() < max(10, 2 * min_width):
        return []
    if not finite.all():   # missing images
        s = np.interp(np.arange(len(s)), np.flatnonzero(finite), s[finite])
    smooth = ndimage.median_filter(s, size=3, mode='nearest')   # keeps the steps
    plateau = np.percentile(smooth, 90)
    diff = np.diff(s)
    noise = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.)
    if plateau < max(min_signal, min_snr * noise):
        return []

    low = smooth < low_fraction * plateau
    bounds = np.flatnonzero(np.diff(np.concatenate(([0], low.astype(int), [0]))))
    runs = [[a, b - 1] for a, b in zip(bounds[::2], bounds[1::2])]
    merged = []
    for run in runs:   # runs separated by 1 or 2 images: same dip
        if merged and run[0] - merged[-1][1] <= 3:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    dips = []
    last = len(s) - 1
    for a, b in merged:
        if b - a + 1 < min_width:
            continue
        floor = float(np.median(smooth[a:b + 1]))
        middle = (a + b) // 2
        entry = _edge(smooth, middle, a, -1, floor, edge_window, onset_fraction) if a > 0 else None
        exit_ = _edge(smooth, middle, b, +1, floor, edge_window, onset_fraction) if b < last else None
        dip = {'floor': floor, 'entry': np.nan, 'exit': np.nan, 'entry_width': np.nan,
               'exit_width': np.nan, 'entry_lo': np.nan, 'entry_hi': np.nan, 'exit_lo': np.nan,
               'exit_hi': np.nan, 'plateau': np.nan}
        levels = []
        for name, edge in (('entry', entry), ('exit', exit_)):
            if edge is not None and edge[2] - floor >= min_edge_fraction * plateau:
                dip[name] = float(edge[0])
                dip[name + '_width'] = float(edge[1] / _WIDTH2080_PER_SIGMA)
                dip[name + '_lo'], dip[name + '_hi'] = float(edge[3][0]), float(edge[3][1])
                levels.append(edge[2])
        if not levels:
            continue
        # a missing edge is accepted only for a dip cut by the limits of the scan
        if (np.isnan(dip['entry']) and a > 0) or (np.isnan(dip['exit']) and b < last):
            continue
        dip['plateau'] = float(np.mean(levels))
        dips.append(dip)
    return dips


# --------------------------------------------------------------------------------------
#  ANALYSIS: dips -> position of the sources along the beam
# --------------------------------------------------------------------------------------
DIPS_COLUMNS = ('spot', 'X', 'Y', 'entry', 'exit', 'entry_width', 'exit_width', 'entry_lo', 'entry_hi', 'exit_lo',
                'exit_hi', 'floor', 'plateau', 'wire', 'ybeam', 'mismatch', 'extent', 'width', 'width_calc',
                'nb_candidates', 'excluded', 'ybeam_entry', 'ybeam_exit', 'ybeam_onset', 'onset_mismatch',
                'ybeam_end')
SPOTS_COLUMNS = ('X', 'Y', 'intensity', 'nb_dips', 'wire', 'ybeam', 'depth', 'spread', 'extent')


def _assign_wire(geometry: WireGeometry, x: float, y: float, dip: Dict[str, float], grid: np.ndarray,
                 tangents: tuple, yrange: Sequence[float], max_mismatch: float) -> Dict[str, float]:
    """wire and ybeam of a dip: wire for which entry and exit edges give the same ybeam, in yrange"""
    res = {'wire': -1, 'ybeam': np.nan, 'mismatch': np.nan, 'extent': np.nan, 'width_calc': np.nan,
           'nb_candidates': 0, 'excluded': 0, 'ybeam_entry': np.nan, 'ybeam_exit': np.nan, 'ybeam_onset': np.nan,
           'onset_mismatch': np.nan, 'ybeam_end': np.nan}
    y_entry = (geometry.ybeam_of_edge(x, y, dip['entry'], 'entry', grid, tangents)
               if np.isfinite(dip['entry']) else None)
    y_exit = (geometry.ybeam_of_edge(x, y, dip['exit'], 'exit', grid, tangents)
              if np.isfinite(dip['exit']) else None)
    candidates = []
    for iw in range(geometry.nb_wires):
        values = [v[iw] for v in (y_entry, y_exit) if v is not None]
        if not values or not np.all(np.isfinite(values)):
            continue
        ybeam = float(np.mean(values))
        mismatch = float(values[0] - values[1]) if len(values) == 2 else np.nan
        if not yrange[0] <= ybeam <= yrange[1] or (len(values) == 2 and abs(mismatch) > max_mismatch):
            continue
        candidates.append((abs(mismatch) if len(values) == 2 else max_mismatch, iw, ybeam, mismatch))
    if not candidates:
        return res
    _, iw, ybeam, mismatch = min(candidates)
    if iw in geometry.excluded_wires:   # dip of an excluded wire: no depth from it
        res.update(wire=iw, excluded=1, nb_candidates=len(candidates))
        return res
    # microns of ybeam per image, and width of the shadow, at this ybeam
    k = int(np.clip(np.searchsorted(grid, ybeam), 1, len(grid) - 1))
    dp_dy = (tangents[0][iw][k] - tangents[0][iw][k - 1]) / (grid[k] - grid[k - 1])   # mm / micron
    um_per_image = abs(geometry.step / dp_dy)
    widths = [w for w in (dip['entry_width'], dip['exit_width']) if np.isfinite(w)]
    width_calc = abs(np.interp(ybeam, grid, tangents[2][iw] - tangents[0][iw]) / geometry.step)
    res.update(wire=iw, ybeam=ybeam, mismatch=mismatch, extent=float(np.mean(widths) * um_per_image),
               width_calc=float(width_calc), nb_candidates=len(candidates),
               ybeam_entry=float(y_entry[iw]) if y_entry is not None else np.nan,
               ybeam_exit=float(y_exit[iw]) if y_exit is not None else np.nan)
    # onset of each edge: the start of the step, i.e. the shallowest sources (smallest ybeam of its lo / hi levels),
    # independent of the extension of the source along the beam (half height: median of the source)
    # and its end: the deepest sources (largest ybeam), e.g. the bottom of a grain
    onsets, ends = [], []
    for kind in ('entry', 'exit'):
        values = [geometry.ybeam_of_edge(x, y, dip[kind + level], kind, grid, tangents)[iw]
                  for level in ('_lo', '_hi') if np.isfinite(dip.get(kind + level, np.nan))]
        values = [v for v in values if np.isfinite(v)]
        if len(values) == 2:
            onsets.append(min(values))
            ends.append(max(values))
    if onsets:
        res['ybeam_onset'] = float(np.mean(onsets))
        res['ybeam_end'] = float(np.mean(ends))
        if len(onsets) == 2:
            res['onset_mismatch'] = float(onsets[0] - onsets[1])
    return res


def analyse(signal: np.ndarray, xy: np.ndarray, geometry: Optional[WireGeometry] = None,
            params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """dips of the profiles of the spots and position along the beam of their sources

    :param signal: (nb spots, nb images) profiles of collect_profiles()
    :param xy: (nb spots, 2 or 3) X, Y (, intensity) of the spots
    :param geometry: WireGeometry (None: dips only, without wire and ybeam)
    :param params: parameters (see DEFAULT_PARAMS)
    :return: dips, spots: dicts of columns (1D arrays), see DIPS_COLUMNS and SPOTS_COLUMNS.
        dips (one line per dip): 'spot' index, 'entry', 'exit' (image indices), 'entry_width',
            'exit_width' (std of edges, images), 'wire' (index in .calib file, -1: none),
            'ybeam' (microns), 'mismatch' (ybeam of entry edge - ybeam of exit edge, microns),
            'extent' (std of the source along the beam from the blur of the edges, microns),
            'width', 'width_calc' (measured and calculated nb of images of the dip),
            'nb_candidates' (nb of possible wires),
            'ybeam_entry', 'ybeam_exit' (microns, from each edge at half height),
            'ybeam_onset' (microns, shallowest sources: onsets of the edges, mean of entry and exit),
            'onset_mismatch' (ybeam_onset of entry - exit edges),
            'ybeam_end' (microns, deepest sources: ends of the edges, mean of entry and exit; ybeam_end - ybeam_onset:
            extension of the source along the beam blurred by the instrument, see thickness_from_width())
        spots (one line per spot): 'nb_dips' with a ybeam, 'ybeam', 'depth' (mean over its dips,
            nan if no dip), 'spread' (max - min of ybeam of its dips), 'wire' (of the best dip)
    """
    params = full_params(params)
    xy = np.asarray(xy, dtype=float)
    yrange = [float(v) for v in params['ybeam_range']]
    margin = 2. * float(params['max_mismatch'])
    grid = np.arange(yrange[0] - margin, yrange[1] + margin + 1., 2.)
    dipkeys = ('low_fraction', 'min_edge_fraction', 'min_snr', 'min_signal', 'min_width', 'edge_window',
               'onset_fraction', 'skip_images')
    dipargs = {key: params[key] for key in dipkeys}

    rows = []
    spots = {key: np.full(len(xy), np.nan) for key in SPOTS_COLUMNS}
    spots['X'], spots['Y'] = xy[:, 0].copy(), xy[:, 1].copy()
    spots['nb_dips'][:] = 0
    spots['wire'][:] = -1
    for ispot, (x, y) in enumerate(xy[:, :2]):
        finite = signal[ispot][np.isfinite(signal[ispot])]
        spots['intensity'][ispot] = np.percentile(finite, 90) if len(finite) else np.nan
        dips = find_dips(signal[ispot], **dipargs)
        if not dips:
            continue
        tangents = geometry.tangent_positions(x, y, grid) if geometry is not None else None
        found = []
        for dip in dips:
            row = dict(dip, spot=ispot, X=x, Y=y, width=dip['exit'] - dip['entry'], wire=-1, ybeam=np.nan,
                       mismatch=np.nan, extent=np.nan, width_calc=np.nan, nb_candidates=0, excluded=0,
                       ybeam_entry=np.nan, ybeam_exit=np.nan, ybeam_onset=np.nan, onset_mismatch=np.nan,
                       ybeam_end=np.nan)
            if geometry is not None:
                row.update(_assign_wire(geometry, x, y, dip, grid, tangents, yrange, params['max_mismatch']))
            rows.append(row)
            if row['wire'] >= 0 and np.isfinite(row['ybeam']):   # (not a dip of an excluded wire)
                found.append(row)
        if found:
            values = np.array([row['ybeam'] for row in found])
            # best dip: complete (2 edges) with the smallest mismatch
            best = min(found, key=lambda row: abs(row['mismatch']) if np.isfinite(row['mismatch']) else np.inf)
            spots['nb_dips'][ispot] = len(found)
            spots['wire'][ispot] = best['wire']
            spots['ybeam'][ispot] = values.mean()
            spots['spread'][ispot] = np.ptp(values)
            spots['extent'][ispot] = np.nanmean([row['extent'] for row in found])
    if geometry is not None:
        spots['depth'] = geometry.depth(spots['ybeam'])
    dipstable = {key: np.array([row[key] for row in rows], dtype=float) for key in DIPS_COLUMNS}
    return dipstable, spots


def run(scan: Dict[str, Any], detfile: str, calibfile: Optional[str], params: Optional[Dict[str, Any]] = None,
        yref: float = 0., sample_tilt: float = 40., nb_cpus: Optional[int] = None,
        progress_callback=None) -> Dict[str, Any]:
    """quick daxm analysis of a wire scan: spots, intensity profiles, dips and ybeam of the spots

    :param scan: section 'scan' of the daxm configuration (see collect_profiles())
    :param calibfile: wires calibration file (None: dips only)
    :return: results dict: 'xy' (X, Y, I of spots), 'signal', 'background', 'missing',
        'wire_positions', 'dips', 'spots' (see analyse()), 'params', 'scan', 'detfile', 'calibfile'
    """
    params = full_params(params)
    if str(params['peaklist']).lower() == 'auto':
        xy = find_spots(max_projection(scan, nb_cpus=nb_cpus), params['spots_threshold'],
                        params['spots_max_number'], params['boxsize'])
        print(f'{len(xy)} spots found in the scan', flush=True)
    else:
        xy = read_peaklist(params['peaklist'])
        print(f"{len(xy)} spots read in {params['peaklist']}", flush=True)
    if len(xy) == 0:
        raise RuntimeError('no spot: check peaklist or spots_threshold')
    profiles = collect_profiles(scan, xy, params['boxsize'], params['boxsize_probe'], nb_cpus, progress_callback)
    positions = wire_positions_of_scan(scan)
    if len(positions) != profiles['signal'].shape[1]:
        raise ValueError(f"{len(positions)} wire positions for {profiles['signal'].shape[1]} images: "
                         'check nb_images and scan_number')
    geometry = None
    if calibfile:
        geometry = WireGeometry.from_files(calibfile, detfile, positions, yref, sample_tilt=sample_tilt)
    dips, spots = analyse(profiles['signal'], xy, geometry, params)
    nbfound = int(np.sum(spots['nb_dips'] > 0))
    print(f"{len(dips['spot'])} dips; {nbfound}/{len(xy)} spots with a position along the beam", flush=True)
    return dict(profiles, xy=xy, wire_positions=positions, dips=dips, spots=spots, params=params,
                scan=dict(scan), detfile=detfile, calibfile=calibfile, yref=yref, sample_tilt=sample_tilt)


def geometry_of_results(results: Dict[str, Any]) -> Optional[WireGeometry]:
    """WireGeometry of results of run() (None if no wires calibration)"""
    if not results.get('calibfile'):
        return None
    return WireGeometry.from_files(results['calibfile'], results['detfile'], results['wire_positions'],
                                   results.get('yref', 0.), sample_tilt=results.get('sample_tilt', 40.))


# --------------------------------------------------------------------------------------
#  WIRES: approximate trajectory parameters from the dips of a reference crystal (Ge)
# --------------------------------------------------------------------------------------
WIRE_KEYS = ('h', 'p0', 'f1', 'f2', 'Rw')   # parameters of each wire (Rw: radius of each wire)
TRAJ_KEYS = ('R', 'u1', 'u2')               # parameters common to all wires (R: same radius for all wires)


def wire_bands(dips: Dict[str, np.ndarray], min_dips: int = 3, bin_size: float = 4.,
               max_gap: float = 25.) -> Tuple[np.ndarray, float]:
    """assign the dips to the wires: in the plot 'image index of dip center' vs 'pixel Y' (plot_abacus()),
    the dips of each wire lie on a band of the same slope (the lines WP of the former quick daxm notebooks)

    :return: labels (band index of each dip, -1: not assigned; bands sorted by increasing image index),
        slope (image index / pixel Y)
    """
    center = 0.5 * (dips['entry'] + dips['exit'])
    ok = np.isfinite(center)
    labels = np.full(len(center), -1)
    if ok.sum() < min_dips:
        return labels, np.nan
    y, c = dips['Y'][ok], center[ok]
    best = (-1., 0.)
    for slope in np.arange(0.01, 3., 0.005):   # slope making the bands the narrowest
        u = c - slope * y
        counts, _ = np.histogram(u, bins=np.arange(u.min(), u.max() + bin_size, bin_size))
        score = np.sum(counts.astype(float) ** 2)
        if score > best[0]:
            best = (score, slope)
    slope = best[1]
    u = c - slope * y
    order = np.argsort(u)
    band, current = np.zeros(len(u), dtype=int), 0
    for k in range(1, len(order)):
        if u[order[k]] - u[order[k - 1]] > max_gap:
            current += 1
        band[order[k]] = current
    sizes = np.bincount(band)
    keep = {b: i for i, b in enumerate([b for b in range(len(sizes)) if sizes[b] >= min_dips])}
    labels[np.flatnonzero(ok)] = [keep.get(b, -1) for b in band]
    return labels, float(slope)


def _linear_wire(x, y, position, ccd_params) -> Tuple[float, float]:
    """h, p0 of a wire perpendicular to the beam (f1 = f2 = 0) moving along the beam (u1 = u2 = 0) whose axis
    is crossed at wire position(s) 'position' by rays from the origin to pixels x, y: p = p0 + h * Py / Pz"""
    pcam = np.asarray(geom.transf_pix_to_coo(ccd_params, np.asarray(x, float), np.asarray(y, float)))
    A = np.column_stack((np.ones(pcam.shape[1]), pcam[1] / pcam[2]))
    (p0, h), *_ = np.linalg.lstsq(A, np.asarray(position, float), rcond=None)
    return float(h), float(p0)


def wires_from_wp(wp: Sequence[Sequence[float]], detfile: str, wire_positions: Sequence[float],
                  x: Optional[float] = None, yrange=(0, 2000), R: float = 0.025, material: str = 'W') -> Dict[str, Any]:
    """approximate wires parameters from the lines WP = [[a, b], ...] of the former quick daxm notebooks
    (image index of dip center = a / 450 * Y + b, at pixel X, default detector center), for wires perpendicular to
    the beam and moving along it (f1 = f2 = u1 = u2 = 0). Inverse of WireGeometry.wp_parameters()

    :return: dict 'wires' (list of dict R, h, p0, f1, f2 ...), 'traj' (u1, u2), as wires_from_dips()
    """
    geometry = WireGeometry([], read_detector(detfile), wire_positions)
    x = geometry.ccd_params[1] if x is None else x
    ys = np.linspace(yrange[0], yrange[1], 21)
    wires = []
    for a, b in wp:
        position = geometry.index_to_position(a / 450. * ys + b)
        h, p0 = _linear_wire(np.full(len(ys), x), ys, position, geometry.ccd_params)
        wires.append({'material': material, 'R': R, 'h': h, 'p0': p0, 'f1': 0., 'f2': 0.})
    return {'wires': wires, 'traj': {'u1': 0., 'u2': 0.}}


def _shadow_y(wire: Dict[str, float], geometry: WireGeometry) -> float:
    """pixel Y (at detector center X) of the shadow center of a wire (dict R, h, p0, f1, f2) at the middle image of
    the scan, for a source at ybeam = 0 (extrapolated beyond the detector)"""
    obj = CircularWire(wire.get('material', 'W'), wire['R'], wire['f1'], wire['f2'], 0., 0., wire['h'], wire['p0'])
    one = WireGeometry([obj], geometry.ccd_params, geometry.positions)
    ys = np.linspace(-3000., 5000., 81)
    centers = one.shadow_lines(ys)['center'][0]
    order = np.argsort(centers)
    return float(np.interp(0.5 * (len(geometry.positions) - 1), centers[order], ys[order]))


def wires_from_dips(results: Dict[str, Any], nb_wires: Optional[int] = None,
                    var: Sequence[str] = ('h', 'p0', 'f1', 'R'), ybeam_um: float = 0., R: float = 0.025,
                    material: str = 'W', min_dips: int = 3, verbose: bool = True) -> Dict[str, Any]:
    """approximate wires trajectory parameters from the dips of the Laue spots of a reference crystal (Ge)
    found by run() without wires calibration: all scattering sources are assumed at ybeam_um along the beam,
    and the edges of the dips (half height) are the wire positions where a wire is tangent to the rays.

    First approximation, to be refined by the usual wires calibration from the fluorescence (calib.py,
    CalibFluo): save it with save_calib() and load it with calib.load_wires().

    1. dips assigned to the wires (wire_bands())
    2. h, p0 of each wire from a linear model (wire perpendicular to the beam, moving along it)
    3. least squares fit of var (WIRE_KEYS of each wire and TRAJ_KEYS common to the wires) to the
       positions of the entry and exit edges of all the dips. 'R': same radius for all the wires, 'Rw': radius of
       each wire (wires of different diameters, e.g. 50 and 80 microns, to tell their shadows apart)

    :param nb_wires: keep the nb_wires bands with most dips (default: all bands with min_dips dips)
    :return: dict 'wires' (list of dict material, R, h, p0, f1, f2 in mm and radians; wire 0 has the largest shadow
        pixel Y, see _shadow_y()),
        'traj' (u1, u2 radians), 'labels' (wire of each dip, -1: not used), 'rms' (images, per wire),
        'residuals' (images, entry and exit edges of the dips used), 'slope' (of the bands)
    """
    from scipy.optimize import least_squares
    dips = results['dips']
    ccd = read_detector(results['detfile'])
    geometry = WireGeometry([], ccd, results['wire_positions'])
    labels, slope = wire_bands(dips, min_dips)
    nbbands = labels.max() + 1
    if nbbands < 1:
        raise RuntimeError('no band of dips: not enough dips of the reference crystal')
    if nb_wires is not None and nbbands > nb_wires:
        sizes = np.bincount(labels[labels >= 0])
        kept = sorted(np.argsort(sizes)[::-1][:nb_wires])
        labels = np.array([kept.index(b) if b in kept else -1 for b in labels])
        nbbands = nb_wires

    # 2. linear model of each band (dip center)
    center = geometry.index_to_position(0.5 * (dips['entry'] + dips['exit']))
    wires = []
    for band in range(nbbands):
        sel = labels == band
        h, p0 = _linear_wire(dips['X'][sel], dips['Y'][sel], center[sel], ccd)
        wires.append({'material': material, 'R': R, 'Rw': R, 'h': h, 'p0': p0, 'f1': 0., 'f2': 0.})
    # wires numbered by decreasing pixel Y of their shadow (convention of the wires calibration notebook: wire 0 has
    # the largest shadow Y, it is the closest to the sample surface)
    order = np.argsort([-_shadow_y(w, geometry) for w in wires])
    wires = [wires[i] for i in order]
    labels = np.array([int(np.flatnonzero(order == b)[0]) if b >= 0 else -1 for b in labels])
    traj = {'R': R, 'u1': 0., 'u2': 0.}

    # 3. least squares fit to the edges
    names = [(iw, key) for iw in range(nbbands) for key in WIRE_KEYS if key in var] + \
            [(None, key) for key in TRAJ_KEYS if key in var]
    used = np.flatnonzero(labels >= 0)
    measured = {name: geometry.index_to_position(dips[name][used]) for name in ('entry', 'exit')}
    xs, ys, ws = dips['X'][used], dips['Y'][used], labels[used]
    ysrc = float(ybeam_um) * 1e-3

    def build(values):
        wd = [dict(w) for w in wires]
        tr = dict(traj)
        for (iw, key), value in zip(names, values):
            if iw is None:
                tr[key] = value
            else:
                wd[iw][key] = value
        return wd, tr

    def model(values):
        wd, tr = build(values)
        objs = [CircularWire(material, w['Rw'] if 'Rw' in var else tr['R'], w['f1'], w['f2'], tr['u1'], tr['u2'],
                             w['h'], w['p0']) for w in wd]
        entry, exit_ = np.empty(len(used)), np.empty(len(used))
        for k, (x, y, iw) in enumerate(zip(xs, ys, ws)):
            pcam = np.asarray(geom.transf_pix_to_coo(ccd, x, y))
            pf, pb = objs[iw].intersect_ray_fronts(ysrc, pcam)
            low, high = min(float(pf), float(pb)), max(float(pf), float(pb))
            entry[k], exit_[k] = (low, high) if geometry.increasing else (high, low)
        return entry, exit_

    def residuals(values):
        entry, exit_ = model(values)
        res = np.concatenate((entry - measured['entry'], exit_ - measured['exit'])) / abs(geometry.step)
        return np.nan_to_num(res)   # nan: edge cut by the limits of the scan

    start = [traj[key] if iw is None else wires[iw][key] for iw, key in names]
    fit = least_squares(residuals, start, loss='soft_l1', f_scale=2.)
    wires, traj = build(fit.x)
    for w in wires:
        w['R'] = w.pop('Rw') if 'Rw' in var else traj['R']
        w.pop('Rw', None)
    res = residuals(fit.x)
    entry_res, exit_res = res[:len(used)], res[len(used):]
    rms = []
    for iw in range(nbbands):
        values = np.concatenate((entry_res[ws == iw], exit_res[ws == iw]))
        values = values[values != 0]
        rms.append(float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan)
    if verbose:
        print(f'{len(used)} dips assigned to {nbbands} wires (bands slope {slope:.3f} image / pixel Y)')
        for iw, w in enumerate(wires):
            print(f"  wire {iw}: h {w['h']:.4f} mm  p0 {w['p0']:.4f} mm  f1 {np.degrees(w['f1']):+.3f} deg  "
                  f"f2 {np.degrees(w['f2']):+.3f} deg  R {w['R'] * 1000:.2f} microns  "
                  f"({np.sum(ws == iw)} dips, rms {rms[iw]:.2f} images)")
        print(f"  u1 {np.degrees(traj['u1']):+.3f} deg  u2 {np.degrees(traj['u2']):+.3f} deg"
              + ('  (radius of each wire fitted)' if 'Rw' in var else '  (same radius for all wires)'))
    return {'wires': wires, 'traj': {'u1': traj['u1'], 'u2': traj['u2']}, 'labels': labels, 'rms': rms,
            'residuals': res, 'slope': slope, 'ybeam_um': ybeam_um, 'var': list(var)}


def wires_from_envelope(results: Dict[str, Any], nb_wires: Optional[int] = None, outliers: float = 0.02,
                        R: Optional[float] = None, material: str = 'W', min_dips: int = 3,
                        method: str = 'geometric', min_dips_per_wire_h: int = 30, tilts: bool = True,
                        verbose: bool = True) -> Dict[str, Any]:
    """approximate wires parameters without reference crystal nor wires calibration, from the dips of the sample
    itself: all the scattering sources are below the surface, so in the plot 'image index of dip center' vs 'pixel Y'
    (plot_abacus()) the dips of each wire lie on one side of the line of the sources at ybeam = 0. For each band of
    dips (one per wire, wire_bands()), this line is the envelope of the band (the shallowest sources): a line of
    the slope common to the bands, with only a fraction 'outliers' of the dips of the band beyond it.

    The depth zero of each wire is then the shallowest source seen by this wire (the sample surface if some grains
    of the surface scatter in its region of the detector). First approximation: prefer a reference crystal (Ge).

    :param outliers: fraction of the dips of a band allowed beyond its line (outliers, noise)
    :param R: radius of the wires (mm); None: from the median width of the dips
    :param method: 'geometric': envelope of each wire = shadow line of a wire (h, p0, f1, f2; u1 = u2 = 0) for
        sources at ybeam = 0, curved in pixel Y and depending on pixel X: wire position of the dip center
        p = p0 + h * (t - tan(f1) * s - tan(f2) * s * t), t = Py / Pz, s = Px / Pz (P: pixel), with its own h
        (and tilts) for bands of at least min_dips_per_wire_h dips, else the common h (no tilt);
        'lines': straight lines of a common slope at the detector center X (former method)
    :param tilts: method 'geometric': also the axis tilts f1, f2 of each wire (dependence on pixel X)
    :return: dict as wires_from_dips() ('wires', 'traj', 'labels'), with 'lines' ([slope, intercept] of each wire:
        image index = slope * Y + intercept, at the detector center X), 'distances' (vertical distance of each dip to
        the line of its wire: image index of the dip - image index of the line at the same pixel Y, > 0 deeper,
        ~ microns along the beam for a wire step of 1 micron; nan: dip not assigned), 'outliers'
    """
    dips = results['dips']
    ccd = read_detector(results['detfile'])
    geometry = WireGeometry([], ccd, results['wire_positions'])
    labels, slope0 = wire_bands(dips, min_dips)
    nbbands = labels.max() + 1
    if nbbands < 1:
        raise RuntimeError('no band of dips')
    if nb_wires is not None and nbbands > nb_wires:
        sizes = np.bincount(labels[labels >= 0])
        kept = sorted(np.argsort(sizes)[::-1][:nb_wires])
        labels = np.array([kept.index(b) if b in kept else -1 for b in labels])
        nbbands = nb_wires
    center = 0.5 * (dips['entry'] + dips['exit'])
    ys = dips['Y']
    # deeper sources are masked later in the scan if the wire positions increase (larger image index)
    sign = 1. if geometry.increasing else -1.

    def envelope(slope):
        """intercepts of the envelope lines and their sharpness (mean distance of the 25 % closest dips)"""
        intercepts, spread = [], 0.
        for band in range(nbbands):
            u = sign * (center[labels == band] - slope * ys[labels == band])
            low = np.quantile(u, outliers)
            intercepts.append(sign * low)
            near = np.sort(u - low)
            spread += np.mean(near[near >= 0][:max(1, len(near) // 4)])
        return intercepts, spread / nbbands

    # common slope: the one making the envelopes the sharpest, around the slope of the bands
    slopes = slope0 + np.linspace(-0.08, 0.08, 81)
    slope = float(slopes[np.argmin([envelope(sl)[1] for sl in slopes])])
    intercepts, _ = envelope(slope)
    # lines at the detector center X: correction of the X dependence of the dips (detector tilts) neglected
    x0 = geometry.ccd_params[1]
    wp = [[slope * 450., b] for b in intercepts]
    fit = wires_from_wp(wp, results['detfile'], results['wire_positions'], x=x0, R=0.025, material=material)
    wires = fit['wires']
    if method == 'geometric':
        # wire position of the dip center vs t = Py / Pz of the pixel: p = p0 + h * t for sources at ybeam = 0
        pcam = np.asarray(geom.transf_pix_to_coo(ccd, np.asarray(dips['X'], float), np.asarray(ys, float)))
        t, sx = pcam[1] / pcam[2], pcam[0] / pcam[2]
        pcenter = np.where(np.isfinite(center), geometry.index_to_position(np.nan_to_num(center)), np.nan)
        psign = 1. if geometry.increasing else -1.   # deeper sources: masked later in the scan

        def envelope_h(band, h, f1=0., f2=0.):
            sel = labels == band
            u = psign * (pcenter[sel] - h * (t[sel] - np.tan(f1) * sx[sel] - np.tan(f2) * sx[sel] * t[sel]))
            low = np.quantile(u, outliers)
            near = np.sort(u - low)
            return psign * low, float(np.mean(near[near >= 0][:max(1, len(near) // 4)]))

        def best_h(bands, h0):
            hs = h0 * (1. + np.linspace(-0.06, 0.06, 121))
            return float(hs[np.argmin([sum(envelope_h(b, h)[1] for b in bands) for h in hs])])

        h_common = best_h(range(nbbands), float(np.median([w['h'] for w in wires])))
        for band, w in enumerate(wires):
            h, f1, f2 = h_common, 0., 0.
            if np.sum(labels == band) >= min_dips_per_wire_h:
                h = best_h([band], h_common)
                if tilts:   # coordinate descent on h, f1, f2 (envelope the sharpest)
                    for _ in range(3):
                        f1s = f1 + np.radians(np.linspace(-1.5, 1.5, 61))
                        f1 = float(f1s[np.argmin([envelope_h(band, h, v, f2)[1] for v in f1s])])
                        f2s = f2 + np.radians(np.linspace(-2., 2., 81))
                        f2 = float(f2s[np.argmin([envelope_h(band, h, f1, v)[1] for v in f2s])])
                        hs = h * (1. + np.linspace(-0.03, 0.03, 61))
                        h = float(hs[np.argmin([envelope_h(band, v, f1, f2)[1] for v in hs])])
            w['h'], w['f1'], w['f2'] = h, f1, f2
            w['p0'] = envelope_h(band, h, f1, f2)[0]
    if R is None:   # radius from the widths of the dips: width of the shadow proportional to the radius
        widths = []
        for band, w in enumerate(wires):
            sel = (labels == band) & np.isfinite(dips['width'])
            if sel.sum():
                one = WireGeometry([CircularWire(material, w['R'], w.get('f1', 0.), w.get('f2', 0.), 0., 0., w['h'],
                                                 w['p0'])], ccd,
                                   results['wire_positions'])
                calc = [np.diff(one.shadow_indices(x, y, 0.)[0, [0, 2]])[0] for x, y in
                        zip(dips['X'][sel][:20], dips['Y'][sel][:20])]
                widths.append(np.median(dips['width'][sel]) / np.median(np.abs(calc)))
        R = 0.025 * float(np.median(widths)) if widths else 0.025
    for w in wires:
        w['R'] = R
    order = np.argsort([-_shadow_y(w, geometry) for w in wires])
    wires = [wires[i] for i in order]
    lines = [[slope, intercepts[i]] for i in order]
    labels = np.array([int(np.flatnonzero(order == b)[0]) if b >= 0 else -1 for b in labels])
    distances = np.full(len(center), np.nan)
    for band, (sl, b) in enumerate(lines):
        sel = labels == band
        if method == 'geometric':   # image index of the dip - image index of the shadow line of the wire
            wb = wires[band]
            line = geometry.position_to_index(wb['p0'] + wb['h'] * (t[sel] - np.tan(wb.get('f1', 0.)) * sx[sel]
                                                                    - np.tan(wb.get('f2', 0.)) * sx[sel] * t[sel]))
            distances[sel] = sign * (center[sel] - line)
        else:
            distances[sel] = sign * (center[sel] - sl * ys[sel] - b)   # vertical distance (image index)
    if verbose:
        kind = ('shadow lines of the wires (h per wire)' if method == 'geometric'
                else f'straight lines of slope {slope:.4f} image / pixel Y')
        print(f'{int(np.sum(labels >= 0))} dips assigned to {nbbands} wires; envelope: {kind} '
              f'({100 * outliers:g} % of the dips of each wire allowed beyond)')
        for iw, w in enumerate(wires):
            print(f"  wire {iw}: h {w['h']:.4f} mm  p0 {w['p0']:.4f} mm  f1 {np.degrees(w.get('f1', 0.)):+.2f} deg  "
                  f"f2 {np.degrees(w.get('f2', 0.)):+.2f} deg  ({np.sum(labels == iw)} dips)")
        print(f"  R {R * 1000:.2f} microns (from the widths of the dips)  u1 = u2 = 0")
    return {'wires': wires, 'traj': {'u1': 0., 'u2': 0.}, 'labels': labels, 'lines': lines,
            'distances': distances, 'outliers': outliers, 'slope': slope, 'method': method}


def plot_envelope(results: Dict[str, Any], fit: Dict[str, Any], axes=None):
    """dips of the sample with the envelope lines of wires_from_envelope() (left), and the distribution of the
    vertical distances of the dips to the line of their wire (right: image index of the dip - image index of the line
    at the same pixel Y; sharp start at 0 expected; ~ positions along the beam in microns for a wire step of 1 micron)"""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(15, 5))
    dips = results['dips']
    center = 0.5 * (dips['entry'] + dips['exit'])
    labels = fit['labels']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ys = np.linspace(0, 2 * read_detector(results['detfile'])[2], 50)
    ax = axes[0]
    ax.plot(dips['Y'][labels < 0], center[labels < 0], '.', color='0.7', ms=3, label='not assigned')
    for iw, (slope, intercept) in enumerate(fit['lines']):
        sel = labels == iw
        ax.plot(dips['Y'][sel], center[sel], '.', ms=4, color=colors[iw % len(colors)])
        if fit.get('method') == 'geometric':   # shadow line of the wire at the detector center X
            ccd = read_detector(results['detfile'])
            pcam = np.asarray(geom.transf_pix_to_coo(ccd, np.full(len(ys), ccd[1]), ys))
            geometry = WireGeometry([], ccd, results['wire_positions'])
            wf, t, sx = fit['wires'][iw], pcam[1] / pcam[2], pcam[0] / pcam[2]
            line = geometry.position_to_index(wf['p0'] + wf['h'] * (t - np.tan(wf.get('f1', 0.)) * sx
                                                                    - np.tan(wf.get('f2', 0.)) * sx * t))
        else:
            line = slope * ys + intercept
        ax.plot(ys, line, '-', color=colors[iw % len(colors)], lw=1, label=f'wire {iw}: ybeam 0')
    ax.set_ylim(0, len(results['wire_positions']) - 1)
    ax.set_xlabel('pixel Y of spot')
    ax.set_ylabel('image index of dip center')
    ax.legend(fontsize=7)
    ax = axes[1]
    for iw in range(len(fit['lines'])):
        values = fit['distances'][labels == iw]
        ax.hist(values[np.isfinite(values)], bins=np.arange(-30, 400, 4), histtype='step',
                color=colors[iw % len(colors)], label=f'wire {iw}')
    ax.axvline(0, color='k', lw=0.8)
    step = abs(float(np.median(np.diff(results['wire_positions'])))) * 1000.
    ax.set_xlabel(f'vertical distance of the dips to the line of their wire (images x {step:.2f} µm of wire travel '
                  '~ µm along the beam, > 0 deeper)')
    ax.set_ylabel('nb of dips')
    ax.legend(fontsize=7)
    return axes


def depth_layers(results: Dict[str, Any], bandwidth: float = 2., min_spots: int = 8, prominence: float = 0.1,
                 ybeam_range: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    """layers along the beam of the Laue spots of a wire scan: peaks of the distribution of ybeam of the spots
    (histogram 1 micron, smoothed by a gaussian of std bandwidth), each layer limited by the minima between peaks.
    A layer gathers the spots of the grain(s) at this depth: their indexation gives the orientation and the strain
    averaged over the thickness of the grain along the beam.

    :param prominence: min prominence of a peak, fraction of the highest peak of the smoothed distribution
    :param min_spots: layers with fewer spots are left out
    :return: list of dict (index, ybeam_min, ybeam_max, ybeam (median), extent (std), nb_spots, spots (indices
        of the spots, strongest first)), sorted by ybeam
    """
    from scipy.signal import find_peaks
    spots = results['spots']
    ybeam = spots['ybeam']
    ok = np.flatnonzero(np.isfinite(ybeam))
    if ybeam_range is not None:
        ok = ok[(ybeam[ok] >= ybeam_range[0]) & (ybeam[ok] <= ybeam_range[1])]
    if len(ok) < min_spots:
        return []
    edges = np.arange(np.floor(ybeam[ok].min()) - 3 * bandwidth, np.ceil(ybeam[ok].max()) + 3 * bandwidth + 1, 1.)
    counts, _ = np.histogram(ybeam[ok], bins=edges)
    smooth = ndimage.gaussian_filter1d(counts.astype(float), bandwidth)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks, _ = find_peaks(smooth, prominence=prominence * smooth.max())
    if not len(peaks):
        peaks = np.array([int(np.argmax(smooth))])
    bounds = [edges[0]] + [centers[a + int(np.argmin(smooth[a:b + 1]))] for a, b in zip(peaks[:-1], peaks[1:])] + [edges[-1]]
    layers = []
    for k in range(len(peaks)):
        sel = ok[(ybeam[ok] >= bounds[k]) & (ybeam[ok] < bounds[k + 1])]
        if len(sel) < min_spots:
            continue
        sel = sel[np.argsort(spots['intensity'][sel])[::-1]]
        layers.append({'index': len(layers), 'ybeam_min': float(bounds[k]), 'ybeam_max': float(bounds[k + 1]),
                       'ybeam': float(np.median(ybeam[sel])), 'extent': float(np.std(ybeam[sel])),
                       'nb_spots': int(len(sel)), 'spots': sel})
    return layers


def read_excluded_wires(calibfile: str) -> List[int]:
    """wires excluded from the depth analysis and the reconstruction (key 'excluded_wires' of a .calib file, [] if
    none). The former readers of .calib files (CalibManager, calib.load_wires()) ignore this key"""
    with open(calibfile, 'r') as f:
        return [int(w) for w in json.load(f).get('excluded_wires', [])]


def set_excluded_wires(calibfile: str, excluded_wires: Optional[Sequence[int]]) -> str:
    """write the list of excluded wires (e.g. [3]) in a .calib file (key 'excluded_wires'; None or []: removed).
    Excluded wires stay in the file with their parameters, so that their shadows are known"""
    with open(calibfile, 'r') as f:
        content = json.load(f)
    if excluded_wires:
        content['excluded_wires'] = sorted(int(w) for w in excluded_wires)
    else:
        content.pop('excluded_wires', None)
    with open(calibfile, 'w') as f:
        f.write(json.dumps(content, indent=1))
    return calibfile


def save_calib(wires_fit: Dict[str, Any], filename: str, directory: str = '',
               excluded_wires: Optional[Sequence[int]] = None) -> str:
    """write wires parameters (wires_from_dips(), wires_from_wp()) in a .calib file (json, angles in degrees),
    the format of the wires calibration (Calib.save_wires(), loaded by calib.load_wires() and CalibManager)

    :param excluded_wires: wires excluded from the depth analysis and the reconstruction (see set_excluded_wires())"""
    path = os.path.join(directory, filename if filename.endswith('.calib') else filename + '.calib')
    traj = {key: float(np.degrees(wires_fit['traj'][key])) for key in ('u1', 'u2')}
    wires = [{'material': w.get('material', 'W'), 'R': float(w['R']), 'p0': float(w['p0']), 'h': float(w['h']),
              'f1': float(np.degrees(w['f1'])), 'f2': float(np.degrees(w['f2']))} for w in wires_fit['wires']]
    content = {'traj': traj, 'wires': wires}
    if excluded_wires:
        content['excluded_wires'] = sorted(int(w) for w in excluded_wires)
    with open(path, 'w') as f:
        f.write(json.dumps(content, indent=1))
    return path


# --------------------------------------------------------------------------------------
#  FILES
# --------------------------------------------------------------------------------------
_ARRAYS = ('xy', 'signal', 'background', 'missing', 'wire_positions')
_ATTRIBUTES = ('params', 'scan', 'detfile', 'calibfile', 'yref', 'sample_tilt')


def save_results(results: Dict[str, Any], path: str) -> str:
    """save results of run() in a hdf5 file"""
    with h5py.File(path, 'w') as f:
        for key in _ARRAYS:
            data = results[key]
            f.create_dataset(key, data=data.astype(np.float32) if key in ('signal', 'background') else data,
                             compression='gzip' if key in ('signal', 'background') else None)
        for table in ('dips', 'spots'):
            group = f.create_group(table)
            for key, values in results[table].items():
                group.create_dataset(key, data=values)
        for key in _ATTRIBUTES:
            f.attrs[key] = json.dumps(results.get(key), default=str)
    return path


def load_results(path: str) -> Dict[str, Any]:
    """results of run() saved by save_results()"""
    with h5py.File(path, 'r') as f:
        results = {key: f[key][()] for key in _ARRAYS}
        for table in ('dips', 'spots'):
            results[table] = {key: f[table][key][()] for key in f[table]}
        for key in _ATTRIBUTES:
            results[key] = json.loads(f.attrs[key])
    return results


def select_spots(results: Dict[str, Any], ybeam_range: Optional[Sequence[float]] = None,
                 max_spread: Optional[float] = None, min_intensity: Optional[float] = None) -> np.ndarray:
    """indices of the spots with a position along the beam (in ybeam_range, microns), strongest first"""
    spots = results['spots']
    ok = np.isfinite(spots['ybeam'])
    if ybeam_range is not None:
        ok &= (spots['ybeam'] >= ybeam_range[0]) & (spots['ybeam'] < ybeam_range[1])
    if max_spread is not None:
        ok &= spots['spread'] <= max_spread
    if min_intensity is not None:
        ok &= spots['intensity'] >= min_intensity
    indices = np.flatnonzero(ok)
    return indices[np.argsort(spots['intensity'][indices])[::-1]]


def _cor_columns(spots: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, Any]:
    names = ['ybeam', 'depth', 'extent', 'wire', 'nb_dips']
    return {'columnsname': names, 'data_spotsproperties': np.column_stack([spots[name][indices] for name in names])}


def detector_offset_um(results: Dict[str, Any], value=None) -> float:
    """position along the beam (microns, > 0 downstream) of the depth zero of the wires calibration (ybeam = 0)
    relative to the source point of the detector calibration (.det, e.g. made at the surface of a Ge crystal placed
    where the surface of the sample is): 2theta and chi of a spot are computed for a source at ybeam + this offset.

    value (default: results['detector_offset']): a number (microns), or 'auto' / None: from the .calib file of
    results: thickness of its layer above the fluorescent material (key 'fluo_layer', e.g. Cr coating on Zr: the
    depth zero of a Zr fluorescence calibration is the Zr surface, below the top surface of the sample aligned at the
    calibration point) + its p0 shift (key 'p0_offset_um': other place of the sample, shift_calib()); 0 without
    these keys (Ge or envelope calibration: zero at the top surface)"""
    value = results.get('detector_offset', 'auto') if value is None else value
    if value is not None and not isinstance(value, str):
        return float(value)
    calibfile = results.get('calibfile')
    if not calibfile or not os.path.isfile(calibfile):
        return 0.
    with open(calibfile, 'r') as f:
        content = json.load(f)
    layer = content.get('fluo_layer') or {}
    return 1000. * float(layer.get('thickness', 0.)) + float(content.get('p0_offset_um', 0.))


def write_cor(results: Dict[str, Any], prefix: str, folder: str, indices: Optional[np.ndarray] = None,
              ybeam_for_angles='spot', detector_offset=None) -> str:
    """write a .cor file of spots (default: all spots with a position along the beam) with
    additional columns ybeam, depth, extent (microns), wire and nb_dips

    :param ybeam_for_angles: source position used for 2theta and chi: 'spot' (ybeam of each spot),
        or a value in microns (same source for all spots, e.g. center of a depth slice; 0: angles of
        a standard .cor file)
    :param detector_offset: position of the depth zero relative to the source point of the detector calibration
        (microns, see detector_offset_um(); default: results['detector_offset'] or 'auto'), added to the source
        position of 2theta and chi
    :return: path of the file
    """
    geometry = geometry_of_results(results)
    if geometry is None:
        raise ValueError('no wires calibration: no position along the beam')
    spots = results['spots']
    if indices is None:
        indices = select_spots(results)
    ysource = spots['ybeam'][indices] if isinstance(ybeam_for_angles, str) else float(ybeam_for_angles)
    offset = detector_offset_um(results, detector_offset)
    twicetheta, chi = geometry.scattering_angles(spots['X'][indices], spots['Y'][indices], ysource + offset)
    detparams, _ = IOLT.readfile_det(results['detfile'], nbCCDparameters=6, verbose=0)
    source = 'ybeam of each spot' if isinstance(ybeam_for_angles, str) else f'ybeam = {ysource:.1f} microns'
    source += f' (+ {offset:.2f} microns: depth zero relative to the source point of the detector calibration)'
    os.makedirs(folder, exist_ok=True)
    ccdlabel = results['scan']['CCDLabel']
    pixelsize = geometry.ccd_params[5]
    # calibration with the detector label (read by the indexation: frame dimensions, pixel size)
    param = {key: float(value) for key, value in zip(('dd', 'xcen', 'ycen', 'xbet', 'xgam'), detparams[:5])}
    param.update(pixelsize=pixelsize, xpixelsize=pixelsize, ypixelsize=pixelsize, CCDLabel=ccdlabel,
                 framedim=DictLT.dict_CCD[ccdlabel][0], kf_direction='Z>0')
    IOLT.writefile_cor(prefix, twicetheta, chi, spots['X'][indices], spots['Y'][indices],
                       spots['intensity'][indices], param=param,
                       comments=[f'quick daxm: 2theta and chi computed for a source at {source}'],
                       dirname_output=folder, dict_data_spotsproperties=_cor_columns(spots, indices))
    return os.path.join(folder, prefix + '.cor')


def write_depth_slices(results: Dict[str, Any], prefix: str, folder: str, step: float = 5.,
                       ybeam_range: Optional[Sequence[float]] = None, thickness: Optional[float] = None,
                       min_spots: int = 1, detector_offset=None) -> List[Dict[str, Any]]:
    """write one .cor file per slice along the beam: <prefix>_0000.cor, <prefix>_0001.cor ...
    (file index = slice index, so the series can be indexed as a line scan along the depth)

    :param step: distance between slices centers (microns of ybeam)
    :param thickness: thickness of a slice (default: step; larger: overlapping slices)
    :param ybeam_range: centers of the first and last slices (default: ybeam_range of the analysis)
    :param min_spots: slices with fewer spots are written empty of spots i.e. not written
    :return: list of dict (index, ybeam, depth, nb_spots, file) of the slices
    """
    geometry = geometry_of_results(results)
    if ybeam_range is None:
        ybeam_range = results['params']['ybeam_range']
    thickness = step if thickness is None else thickness
    slices = []
    for k, center in enumerate(np.arange(ybeam_range[0], ybeam_range[1] + 0.5 * step, step)):
        indices = select_spots(results, (center - 0.5 * thickness, center + 0.5 * thickness))
        info = {'index': k, 'ybeam': float(center), 'depth': float(geometry.depth(center)),
                'nb_spots': int(len(indices)), 'file': None}
        if len(indices) >= max(1, min_spots):
            info['file'] = write_cor(results, f'{prefix}_{k:04d}', folder, indices, ybeam_for_angles=center,
                                     detector_offset=detector_offset)
        slices.append(info)
    return slices


# --------------------------------------------------------------------------------------
#  PLOTS
# --------------------------------------------------------------------------------------
def read_fit_spots(fitfile: str) -> Dict[str, np.ndarray]:
    """spots of a .fit file (one grain): columns of its '##' header line (e.g. 'Xexp', 'Yexp', 'energy(keV)', 'h',
    'k', 'l'), read until the next comment line"""
    names, rows = None, []
    with open(fitfile, 'r') as f:
        for line in f:
            if line.startswith('##'):
                names = line[2:].split()
                continue
            if names is None:
                continue
            if line.startswith('#') or not line.strip():
                if rows:
                    break
                continue
            values = line.split()
            if len(values) != len(names):
                break
            rows.append([float(v) for v in values])
    if names is None:
        return {}
    data = np.array(rows, dtype=float).reshape(-1, len(names))
    return {name: data[:, k] for k, name in enumerate(names)}


def dips_table(results: Dict[str, Any], geometry: Optional[WireGeometry] = None, fitfiles: Sequence[str] = (),
               max_distance: float = 1.5):
    """one line per dip with a position along the beam, with what may bias it: wire, 2theta, chi, energy and grain
    (spots of indexed .fit files), to study the discrepancies of the dips of a same grain (plot_dips_table())

    :param geometry: wires (default: calibfile of results); the dips are analysed again with it
    :param fitfiles: .fit files (one grain each, e.g. the indexation of the depth layers or of the pattern of a Ge
        crystal): energy, hkl and grain (file name) of the spots within max_distance pixels
    :return: pandas DataFrame: spot, X, Y, wire, two_theta, chi, energy, grain, h, k, l, ybeam, ybeam_entry,
        ybeam_exit, ybeam_onset, ybeam_end, mismatch, onset_mismatch, entry_width, exit_width (images), plateau,
        dev, dev_onset (ybeam - median of the grain, microns; nan without grain)
    """
    import pandas as pd
    if geometry is None:
        geometry = geometry_of_results(results)
    if geometry is None:
        raise ValueError('a wires calibration is needed (calibfile of the results or geometry)')
    dips, _ = analyse(results['signal'], results['xy'], geometry, results['params'])
    keep = (dips['wire'] >= 0) & np.isfinite(dips['ybeam'])
    keys = ('spot', 'X', 'Y', 'wire', 'ybeam', 'ybeam_entry', 'ybeam_exit', 'ybeam_onset', 'ybeam_end', 'mismatch',
            'onset_mismatch', 'entry_width', 'exit_width', 'plateau')
    table = pd.DataFrame({key: dips[key][keep] for key in keys})
    table[['spot', 'wire']] = table[['spot', 'wire']].astype(int)
    two_theta, chi = geometry.scattering_angles(table['X'].values, table['Y'].values,
                                                table['ybeam'].values + detector_offset_um(results))
    table.insert(4, 'two_theta', np.ravel(two_theta))
    table.insert(5, 'chi', np.ravel(chi))
    nbspots = len(results['xy'])
    energy, grain, hkl = np.full(nbspots, np.nan), np.full(nbspots, '', dtype=object), np.full((nbspots, 3), np.nan)
    xy = np.asarray(results['xy'], dtype=float)[:, :2]
    for fitfile in fitfiles:
        spots = read_fit_spots(fitfile)
        if not spots or 'Xexp' not in spots:
            continue
        name = os.path.splitext(os.path.basename(fitfile))[0]
        for k in range(len(spots['Xexp'])):
            dist = np.hypot(xy[:, 0] - spots['Xexp'][k], xy[:, 1] - spots['Yexp'][k])
            j = int(np.argmin(dist))
            if dist[j] <= max_distance:
                energy[j] = spots.get('energy(keV)', np.full(len(spots['Xexp']), np.nan))[k]
                grain[j] = name
                hkl[j] = [spots.get(c, np.full(len(spots['Xexp']), np.nan))[k] for c in ('h', 'k', 'l')]
    spot = table['spot'].values
    table.insert(6, 'energy', energy[spot])
    table.insert(7, 'grain', grain[spot])
    for c, col in enumerate(('h', 'k', 'l')):
        table.insert(8 + c, col, hkl[spot, c])
    for column, source in (('dev', 'ybeam'), ('dev_onset', 'ybeam_onset')):
        median = table[table['grain'] != ''].groupby('grain')[source].transform('median')
        table[column] = np.where(table['grain'] != '', table[source] - median.reindex(table.index), np.nan)
    return table


def mismatch_vs_image(results: Dict[str, Any], geometry: Optional[WireGeometry] = None,
                      edges: Sequence[int] = (0, 10, 20, 30, 40, 50, 60, 80, 120, 200, 300, 360, 380, 401),
                      verbose: bool = True):
    """median entry - exit mismatch of the dips as a function of the image index of their entry edge (and of their
    exit edge): a wrong wire position in a part of the scan (e.g. the first ~40 images of the scans of a grid after
    the return of the wire, -8 microns for Zr5 of a321217) shows as a step; then set skip_images

    :return: pandas DataFrame: images (range), mismatch_entry, nb_entry, mismatch_exit, nb_exit
    """
    import pandas as pd
    geometry = geometry or geometry_of_results(results)
    params = dict(results['params'], skip_images=0)
    dips, _ = analyse(results['signal'], results['xy'], geometry, params)
    ok = (dips['wire'] >= 0) & np.isfinite(dips['mismatch'])
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        e = ok & (dips['entry'] >= a) & (dips['entry'] < b)
        x = ok & (dips['exit'] >= a) & (dips['exit'] < b)
        rows.append({'images': f'{a}-{b}', 'mismatch_entry': np.median(dips['mismatch'][e]) if e.any() else np.nan,
                     'nb_entry': int(e.sum()), 'mismatch_exit': np.median(dips['mismatch'][x]) if x.any() else np.nan,
                     'nb_exit': int(x.sum())})
    table = pd.DataFrame(rows)
    if verbose:
        reference = np.median(dips['mismatch'][ok])
        bad = table[(np.abs(table['mismatch_entry'] - reference) > 3) | (np.abs(table['mismatch_exit'] - reference) > 3)]
        if len(bad):
            print(f"mismatch far from its median ({reference:+.1f} um) for edges in images {list(bad['images'])}: "
                  'wire position not reliable there? (option skip_images)')
    return table


def width_of_thickness(thickness, sigma: float, onset_fraction: float = 0.1) -> np.ndarray:
    """distance (microns along the beam) between the levels onset_fraction and 1 - onset_fraction of the edge of a
    dip made by a source uniform along the beam over thickness (microns), blurred by a gaussian of std sigma (the
    instrument: beam, wire edge, step): ybeam_end - ybeam_onset of a dip (analyse())"""
    from scipy.special import ndtr
    thickness = np.atleast_1d(np.asarray(thickness, dtype=float))
    sigma = max(float(sigma), 1e-6)
    res = []
    for t in thickness:
        y = np.linspace(-6 * sigma - 1., t + 6 * sigma + 1., 4001)
        if t <= 1e-9:
            cdf = ndtr(y / sigma)
        else:
            g = lambda v: v * ndtr(v / sigma) + sigma * np.exp(-0.5 * (v / sigma) ** 2) / np.sqrt(2 * np.pi)
            cdf = (g(y) - g(y - t)) / t
        res.append(np.interp(1. - onset_fraction, cdf, y) - np.interp(onset_fraction, cdf, y))
    return np.array(res)


def thickness_from_width(width, sigma: float = 0.65, onset_fraction: float = 0.1) -> np.ndarray:
    """thickness along the beam (microns) of a uniform source whose dip edges have this width between the levels
    onset_fraction and 1 - onset_fraction (ybeam_end - ybeam_onset), the blur of the instrument (std sigma, microns:
    ~0.65 measured with the fluorescence, fluo_resolution()) removed. 0 if the width is not larger than the blur"""
    grid = np.concatenate([[0.], np.geomspace(0.05, 500., 300)])
    widths = width_of_thickness(grid, sigma, onset_fraction)
    return np.interp(np.asarray(width, dtype=float), widths, grid, left=0.)


def grain_depth_summary(dips_of_grain, sigma: float = 0.65, sample_tilt: float = 40., onset_fraction: float = 0.1,
                        detector_offset: float = 0.) -> Dict[str, float]:
    """position and thickness along the beam of a grain from the dips of its Laue spots (rows of dips_table() of
    the grain): all its spots come from the same volume, the beginning of the step down of a dip (onset) gives the
    shallowest sources, its end the deepest ones (medians over the dips, robust to a few wrong dips)

    :param sigma: blur of the instrument (microns along the beam, std), removed from the width of the edges
    :param detector_offset: microns, position of the depth zero relative to the source point of the detector
        calibration (detector_offset_um()), to give also the positions relative to it
    :return: dict: nb_dips, ybeam (median of the half height positions: middle of the grain), ybeam_min, ybeam_max
        (medians of the onsets and ends of the edges: blurred limits), thickness_beam (blur removed, uniform grain),
        depth, depth_min, depth_max, thickness_depth (normal to the surface, x sin(tilt); below the depth zero of
        the wires calibration), ybeam_detector (ybeam + detector_offset: source used for 2theta, chi)
    """
    sin = np.sin(np.radians(sample_tilt))
    ok = np.isfinite(dips_of_grain['ybeam'])
    ybeam = np.asarray(dips_of_grain['ybeam'], dtype=float)[ok]
    start = np.asarray(dips_of_grain['ybeam_onset'], dtype=float)[ok]
    end = np.asarray(dips_of_grain['ybeam_end'], dtype=float)[ok]
    both = np.isfinite(start) & np.isfinite(end)
    res = {'nb_dips': int(ok.sum()), 'ybeam': np.nan, 'ybeam_min': np.nan, 'ybeam_max': np.nan,
           'thickness_beam': np.nan, 'thickness_beam_spread': np.nan}
    if ok.any():
        res['ybeam'] = float(np.median(ybeam))
    if both.any():
        width = np.median(end[both] - start[both])
        thickness = thickness_from_width(end[both] - start[both], sigma, onset_fraction)
        res.update(ybeam_min=float(np.median(start[both])), ybeam_max=float(np.median(end[both])),
                   thickness_beam=float(thickness_from_width(width, sigma, onset_fraction)),
                   thickness_beam_spread=float(1.4826 * np.median(np.abs(thickness - np.median(thickness)))))
    for key, name in (('ybeam', 'depth'), ('ybeam_min', 'depth_min'), ('ybeam_max', 'depth_max'),
                      ('thickness_beam', 'thickness_depth')):
        res[name] = res[key] * sin
    res['ybeam_detector'] = res['ybeam'] + float(detector_offset)
    return res


def grains_depth_table(results: Dict[str, Any], fitfiles: Sequence[str], geometry: Optional[WireGeometry] = None,
                       sigma: float = 0.65, max_distance: float = 1.5):
    """grain_depth_summary() of each .fit file (one grain each, spots matched to the dips of results within
    max_distance pixels): pandas DataFrame, one line per .fit file (column 'grain': file name)"""
    import pandas as pd
    table = dips_table(results, geometry, fitfiles, max_distance)
    offset = detector_offset_um(results)
    tilt = results.get('sample_tilt', 40.)
    rows = []
    for fitfile in fitfiles:
        name = os.path.splitext(os.path.basename(fitfile))[0]
        rows.append({'grain': name, **grain_depth_summary(table[table['grain'] == name], sigma, tilt,
                                                          results['params'].get('onset_fraction', 0.1), offset)})
    return pd.DataFrame(rows)


def plot_dips_table(table, column: str = 'dev_onset', variables: Sequence[str] = ('two_theta', 'chi', 'energy', 'wire'),
                    max_abs: Optional[float] = None, axes=None):
    """discrepancies of the dips (column of dips_table(), e.g. 'dev_onset', 'dev', 'mismatch', 'onset_mismatch',
    'ybeam_onset') as a function of 2theta, chi, energy and wire: points coloured by wire and medians per bin"""
    import matplotlib.pyplot as plt
    data = table[np.isfinite(table[column])]
    if max_abs is not None:
        data = data[np.abs(data[column]) <= max_abs]
    variables = [v for v in variables if v in data and np.isfinite(data[v]).any()]
    if axes is None:
        _, axes = plt.subplots(1, len(variables), figsize=(4. * len(variables), 3.4), squeeze=False, sharey=True)
        axes = axes[0]
    for ax, var in zip(axes, variables):
        values = data[list(dict.fromkeys([var, column, 'wire']))].dropna()
        if var == 'wire':
            groups = [values[column][values['wire'] == w].values for w in sorted(values['wire'].unique())]
            ax.boxplot(groups, positions=sorted(values['wire'].unique()), widths=0.6, showfliers=False)
            for w, g in zip(sorted(values['wire'].unique()), groups):
                ax.plot(np.full(len(g), w) + np.random.uniform(-0.2, 0.2, len(g)), g, '.', ms=2, alpha=0.4,
                        color=f'C{int(w)}')
        else:
            for w in sorted(values['wire'].unique()):
                sub = values[values['wire'] == w]
                ax.plot(sub[var], sub[column], '.', ms=3, alpha=0.5, color=f'C{int(w)}', label=f'wire {int(w)}')
            edges = np.unique(np.quantile(values[var], np.linspace(0, 1, 9)))
            if len(edges) > 2:
                index = np.clip(np.digitize(values[var], edges[1:-1]), 0, len(edges) - 2)
                centers = [values[var][index == b].median() for b in range(len(edges) - 1)]
                medians = [values[column][index == b].median() for b in range(len(edges) - 1)]
                ax.plot(centers, medians, 'k-o', ms=4, lw=1.5, label='median')
        ax.axhline(0., color='0.5', lw=0.8)
        ax.set_xlabel({'two_theta': '2theta (deg)', 'chi': 'chi (deg)', 'energy': 'energy (keV)'}.get(var, var))
    axes[0].set_ylabel(f'{column} (microns)')
    axes[0].legend(fontsize=7)
    return axes


def plot_profile(results: Dict[str, Any], spot: int, ax=None, show_background: bool = True):
    """profile of a spot with its dips (entry edges in red, exit edges in green) and the images
    where each wire would mask a source at ybeam = 0 (grey bands): a dip shifted from its band comes
    from a source deeper or shallower"""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    signal = results['signal'][spot]
    ax.plot(signal, lw=1, label='spot - local background')
    if show_background:
        background = results['background'][spot]
        ax.plot(background - np.nanmin(background), lw=1, color='tab:orange', label='local background (shifted)')
    dips, spots = results['dips'], results['spots']
    title = f"spot {spot}  X={spots['X'][spot]:.0f} Y={spots['Y'][spot]:.0f}"
    for k in np.flatnonzero(dips['spot'] == spot):
        ax.axvline(dips['entry'][k], color='r', lw=0.8)
        ax.axvline(dips['exit'][k], color='g', lw=0.8)
        if dips['wire'][k] >= 0:
            title += f"\nwire {int(dips['wire'][k])}: ybeam {dips['ybeam'][k]:.1f} µm (mismatch {dips['mismatch'][k]:.1f})"
    geometry = geometry_of_results(results)
    if geometry is not None:
        shadows = geometry.shadow_indices(spots['X'][spot], spots['Y'][spot], 0.)
        for iw, (first, _, last) in enumerate(shadows):
            if last > 0 and first < len(signal) - 1:
                ax.axvspan(max(first, 0), min(last, len(signal) - 1), color='0.85', zorder=0)
                ax.text(np.clip(0.5 * (first + last), 0, len(signal) - 1), ax.get_ylim()[1], f'w{iw}',
                        ha='center', va='top', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('image index')
    ax.set_ylabel('intensity')
    ax.legend(fontsize=7, loc='lower right')
    return ax


def plot_abacus(results: Dict[str, Any], ax=None, ybeams: Sequence[float] = (0.,)):
    """image index of the dips (center) of all spots as a function of their pixel Y, with the lines
    calculated from the wires calibration for sources at ybeams (microns): spots of a grain are on
    a line parallel to the lines of ybeam = 0"""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    dips = results['dips']
    center = 0.5 * (dips['entry'] + dips['exit'])
    found = (dips['wire'] >= 0) & np.isfinite(dips['ybeam'])
    ax.plot(dips['Y'][~found], center[~found], '.', color='0.6', ms=3, label='dips without depth')
    sc = ax.scatter(dips['Y'][found], center[found], c=dips['ybeam'][found], s=8, cmap='turbo')
    ax.figure.colorbar(sc, ax=ax, label='ybeam (µm)')
    geometry = geometry_of_results(results)
    if geometry is not None:
        framedim = DictLT.dict_CCD[results['scan']['CCDLabel']][0]
        ys = np.arange(0, framedim[0], 25)
        for ybeam, style in zip(ybeams, ('-', '--', ':', '-.')):
            lines = geometry.shadow_lines(ys, ybeam_um=ybeam)
            for iw, line in enumerate(lines['center']):
                ax.plot(ys, line, style, color='k', lw=0.8,
                        label=f'wires, ybeam {ybeam:g} µm' if iw == 0 else None)
    ax.set_ylim(0, len(results['wire_positions']) - 1)
    ax.set_xlabel('pixel Y of spot')
    ax.set_ylabel('image index of dip center')
    ax.legend(fontsize=7)
    return ax


def plot_spots(results: Dict[str, Any], ax=None, ybeam_range: Optional[Sequence[float]] = None):
    """spots on the detector, coloured by their position along the beam (grey: no position)"""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    spots = results['spots']
    found = np.isfinite(spots['ybeam'])
    ax.plot(spots['X'][~found], spots['Y'][~found], '.', color='0.7', ms=3)
    vmin, vmax = ybeam_range if ybeam_range is not None else (None, None)
    sc = ax.scatter(spots['X'][found], spots['Y'][found], c=spots['ybeam'][found], s=12, cmap='turbo',
                    vmin=vmin, vmax=vmax)
    ax.figure.colorbar(sc, ax=ax, label='ybeam (µm)')
    framedim = DictLT.dict_CCD[results['scan']['CCDLabel']][0]
    ax.set_xlim(0, framedim[1])
    ax.set_ylim(framedim[0], 0)
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    return ax


def shadow_polygons(geometry: WireGeometry, imageindex: float, ybeam_um: float = 0., xs: Optional[Sequence[float]] = None,
                    ys: Optional[Sequence[float]] = None) -> List[Optional[np.ndarray]]:
    """detector regions masked by each wire in image imageindex for a source at ybeam_um: polygon (n, 2) of pixel
    X, Y between the entry and exit edges of the shadow (None if the shadow is not on the grid of pixels)"""
    if xs is None:
        xs = np.linspace(0, 2 * geometry.ccd_params[1], 9)
    if ys is None:
        ys = np.arange(-1000., 3200., 20.)
    polygons = []
    for iw in range(geometry.nb_wires):
        one = WireGeometry([geometry.wires[iw]], geometry.ccd_params, geometry.positions)
        front, back = [], []
        for x in xs:
            lines = one.shadow_lines(ys, x=x, ybeam_um=ybeam_um)
            yy = []
            for key in ('entry', 'exit'):
                index = lines[key][0]
                order = np.argsort(index)
                if index[order[0]] <= imageindex <= index[order[-1]]:
                    yy.append(float(np.interp(imageindex, index[order], ys[order])))
            if len(yy) == 2:
                front.append((x, yy[0]))
                back.append((x, yy[1]))
        polygons.append(np.array(front + back[::-1]) if len(front) >= 2 else None)
    return polygons


def plot_image_shadows(scan: Dict[str, Any], geometry: WireGeometry, imageindex: int, ybeam_um: float = 0.,
                       ax=None, vmax: Optional[float] = None, spots: Optional[np.ndarray] = None):
    """Laue image of the wire scan with the shadows of the wires (colored bands, wire index at the right) predicted
    for a source at ybeam_um: check of a wires calibration (Laue spots of the reference crystal at ybeam_um are
    dark in the bands)

    :param spots: optional (n, 2) X, Y of spots drawn as circles
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    image = read_image(scan, imageindex)
    vmax = np.percentile(image, 99.5) if vmax is None else vmax
    ax.imshow(image, cmap='gray', vmin=0, vmax=vmax, origin='upper')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    height, width = image.shape
    for iw, poly in enumerate(shadow_polygons(geometry, imageindex, ybeam_um, xs=np.linspace(0, width, 9))):
        if poly is None:
            continue
        ax.add_patch(Polygon(poly, closed=True, alpha=0.35, facecolor=colors[iw % len(colors)],
                             edgecolor=colors[iw % len(colors)]))
        right = poly[np.argmax(poly[:, 0])]
        if 0 <= right[1] <= height:
            ax.text(width - 10, right[1], f'w{iw}', color=colors[iw % len(colors)], ha='right', va='bottom')
    if spots is not None and len(spots):
        ax.scatter(np.asarray(spots)[:, 0], np.asarray(spots)[:, 1], s=60, facecolors='none', edgecolors='y', lw=0.8)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f'image {imageindex}: wire shadows for a source at ybeam {ybeam_um:g} µm', fontsize=9)
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    return ax


def read_cor_spots(path: str) -> Dict[str, Any]:
    """spots of a .cor file (e.g. written by write_cor(), write_depth_slices()): dict 'tth', 'chi', 'X', 'Y', 'I'
    and the additional columns (ybeam, depth ...) if any, 'gnomon' (n, 2) gnomonic coordinates"""
    from LaueTools import indexingImageMatching as IIM
    res = IOLT.readfile_cor(path, output_only5columns=False)
    spots = {'tth': np.atleast_1d(res[1]), 'chi': np.atleast_1d(res[2]), 'X': np.atleast_1d(res[3]),
             'Y': np.atleast_1d(res[4]), 'I': np.atleast_1d(res[5])}
    extra = res[7] if len(res) > 7 and isinstance(res[7], dict) else None
    if extra and extra.get('columnsname') is not None:
        data = np.atleast_2d(extra['data_spotsproperties'])
        for k, name in enumerate(extra['columnsname']):
            if k < data.shape[1]:
                spots[name] = data[:, k]
    gx, gy = IIM.ComputeGnomon_2((spots['tth'], spots['chi'])) if len(spots['tth']) else ([], [])
    spots['gnomon'] = np.column_stack((gx, gy)) if len(spots['tth']) else np.zeros((0, 2))
    return spots


def plot_peaklist(path: str, axes=None, nbmax: Optional[int] = None, background: Optional[Dict[str, Any]] = None,
                  vrange: Optional[Sequence[float]] = None):
    """spots of a .cor file on the detector (left) and in gnomonic projection (right): spots of a grain on lines
    (zone axes). Colored by ybeam if the file has this column. nbmax: only the nbmax most intense spots.
    background: spots (read_cor_spots()) drawn in light grey, e.g. all the spots of the scan
    vrange: (min, max) of the color scale of ybeam (microns), the same for all the files of a browser"""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 6))
    spots = read_cor_spots(path)
    order = np.argsort(spots['I'])[::-1][:nbmax]
    for ax in axes:
        ax.clear()
    if background is not None:
        axes[0].scatter(background['X'], background['Y'], s=4, c='0.85')
        axes[1].scatter(*background['gnomon'].T, s=4, c='0.85')
    color = spots['ybeam'][order] if 'ybeam' in spots else 'tab:blue'
    size = 8 + 40 * np.sqrt(spots['I'][order] / max(spots['I'].max(), 1e-9)) if len(order) else 8
    vmin, vmax = vrange if vrange is not None else (None, None)
    axes[0].scatter(spots['X'][order], spots['Y'][order], s=size, c=color, cmap='turbo', vmin=vmin, vmax=vmax)
    sc = axes[1].scatter(*spots['gnomon'][order].T, s=size, c=color, cmap='turbo', vmin=vmin, vmax=vmax)
    # two square panels of the same size: the detector (equal scales) and the gnomonic projection stretched to fill
    # its square (scales of its two axes different), limits fixed by all the spots of the scan (stable from file to file)
    axes[0].set_xlim(-50, 2150)
    axes[0].set_ylim(2200, 0)
    axes[0].set_aspect('equal', adjustable='box')
    points = background['gnomon'] if background is not None and len(background['gnomon']) else spots['gnomon']
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points):
        low, high = points.min(axis=0), points.max(axis=0)
        pad = 0.03 * np.maximum(high - low, 1e-6)
        axes[1].set_xlim(low[0] - pad[0], high[0] + pad[0])
        axes[1].set_ylim(low[1] - pad[1], high[1] + pad[1])
    axes[1].set_aspect('auto')
    for ax in axes:
        ax.set_box_aspect(1)
    axes[0].set_xlabel('X (pixel)')
    axes[0].set_ylabel('Y (pixel)')
    axes[1].set_xlabel('gnomonic X')
    axes[1].set_ylabel('gnomonic Y')
    title = f'{os.path.basename(path)}: {len(order)} spots'
    if 'ybeam' in spots and len(order):
        title += f" (ybeam {np.min(spots['ybeam'][order]):.0f} to {np.max(spots['ybeam'][order]):.0f} µm)"
    axes[0].figure.suptitle(title, fontsize=10)
    return axes, sc


class PeakListBrowser:
    """browser of peak lists (.cor files, e.g. the slices along the beam of write_depth_slices()): spots on the
    detector and in gnomonic projection (spots of a grain lie on lines: zone axes), file chosen with a slider.
    Colors: ybeam (same scale for all the files), size: intensity.

    Options: nb of most intense spots shown, background (all the spots of the first file in light grey, e.g.
    quickdaxm_spots.cor). Needs ipywidgets; interactive figure with %matplotlib widget (ipympl), else redrawn as a
    static image"""

    def __init__(self, files: Sequence[str], nbmax: Optional[int] = 200, background_file: Optional[str] = None,
                 figsize=(13, 6)):
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display
        self.files = [str(f) for f in files]
        if not self.files:
            raise ValueError('no peak list file')
        self.background = read_cor_spots(background_file) if background_file else None
        # one color scale of ybeam for all the files (from the background spots, else the first file)
        reference = self.background if self.background is not None else read_cor_spots(self.files[0])
        self.vrange = ((np.percentile(reference['ybeam'], 2), np.percentile(reference['ybeam'], 98))
                       if 'ybeam' in reference and len(reference['ybeam']) else None)
        self.slider = widgets.IntSlider(value=0, min=0, max=len(self.files) - 1, description='file',
                                        continuous_update=False, layout=widgets.Layout(width='60%'))
        self.nbmax = widgets.IntText(value=nbmax or 0, description='nb spots', layout=widgets.Layout(width='18%'),
                                     tooltip='most intense spots shown (0: all)')
        self.showbg = widgets.Checkbox(value=self.background is not None, description='all spots in grey',
                                       disabled=self.background is None, indent=False)
        self.label = widgets.HTML()
        self.widget_canvas = 'ipympl' in plt.get_backend() or 'widget' in plt.get_backend()
        if self.widget_canvas:
            with plt.ioff():
                self.fig, self.axes = plt.subplots(1, 2, figsize=figsize)
            self.out = self.fig.canvas
        else:
            self.out = widgets.Output()
        for w in (self.slider, self.nbmax, self.showbg):
            w.observe(lambda change: self.update(), names='value')
        display(widgets.VBox([widgets.HBox([self.slider, self.nbmax, self.showbg]), self.label, self.out]))
        self.update()

    def update(self):
        import matplotlib.pyplot as plt
        path = self.files[self.slider.value]
        self.label.value = f'<code>{path}</code>'
        nbmax = self.nbmax.value or None
        background = self.background if self.showbg.value else None
        if self.widget_canvas:
            plot_peaklist(path, self.axes, nbmax, background, self.vrange)
            self.fig.canvas.draw_idle()
        else:
            self.out.clear_output(wait=True)
            with self.out:
                fig, axes = plt.subplots(1, 2, figsize=(13, 6))
                plot_peaklist(path, axes, nbmax, background, self.vrange)
                plt.show()


def plot_histogram(results: Dict[str, Any], ax=None, step: float = 2.):
    """nb of spots as a function of their position along the beam"""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    yrange = results['params']['ybeam_range']
    values = results['spots']['ybeam']
    ax.hist(values[np.isfinite(values)], bins=np.arange(yrange[0], yrange[1] + step, step))
    ax.set_xlabel('ybeam (µm)')
    ax.set_ylabel('nb of spots')
    return ax
