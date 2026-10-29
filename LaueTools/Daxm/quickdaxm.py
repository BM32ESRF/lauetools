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
    'coresize': 1,             # half size (pixels) of the spot core (1: 3x3 pixels)
    'low_fraction': 0.3,       # dip: signal below this fraction of the plateau
    'min_edge_fraction': 0.5,  # edge height / plateau must be larger (rejects intensity fluctuations)
    'min_snr': 8.,             # plateau / noise must be larger
    'min_signal': 50.,         # plateau (counts above local background) must be larger
    'min_width': 4,            # min nb of images of a dip
    'edge_window': 20,         # nb of images outside a dip used to estimate the plateau
    'ybeam_range': [-100., 200.],  # microns: accepted source positions along the beam
    'max_mismatch': 10.,       # microns: max difference of ybeam given by entry and exit edges
}

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
                 sample_tilt: float = 40.):
        self.wires = list(wires)
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
        """wires of a .calib file (wires calibration) and detector of a .det file"""
        wires = CalibManager(calibfile, yref).get_wires()
        return cls(wires, read_detector(detfile, pixelsize), wire_positions, sample_tilt)

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
               background_size: int = 15) -> np.ndarray:
    """spots of an image (e.g. max_projection()): local maxima higher than threshold above the local
    background (median filter), at more than boxsize pixels from the borders

    :return: array (nb spots, 3): X, Y (integer pixels), intensity above background, strongest first
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
    return np.column_stack((pos[order, 1], pos[order, 0], amp[order])).astype(float)


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


def _boxes(xy: np.ndarray, boxsize: int, coresize: int, framedim: Tuple[int, int]):
    """pixel indices of the boxes centered on xy and masks of core and border pixels of a box"""
    xy = np.rint(np.asarray(xy, dtype=float)[:, :2]).astype(int)
    xc = np.clip(xy[:, 0], boxsize, framedim[1] - boxsize - 1)
    yc = np.clip(xy[:, 1], boxsize, framedim[0] - boxsize - 1)
    offsets = np.arange(-boxsize, boxsize + 1)
    iy = (yc[:, None, None] + offsets[None, :, None]).repeat(len(offsets), axis=2).reshape(len(xy), -1)
    ix = (xc[:, None, None] + offsets[None, None, :]).repeat(len(offsets), axis=1).reshape(len(xy), -1)
    oy, ox = np.meshgrid(offsets, offsets, indexing='ij')
    core = ((np.abs(oy) <= coresize) & (np.abs(ox) <= coresize)).ravel()
    border = ((np.abs(oy) == boxsize) | (np.abs(ox) == boxsize)).ravel()
    return iy, ix, core, border


def collect_profiles(scan: Dict[str, Any], xy: np.ndarray, boxsize: int = 5, coresize: int = 1,
                     nb_cpus: Optional[int] = None, progress_callback=None) -> Dict[str, np.ndarray]:
    """intensity profiles of spots over the images of a wire scan, images being read in parallel

    :param scan: image_folder, prefix, suffix, nbdigits, CCDLabel, nb_images, first_image_index
    :param xy: (nb spots, 2) pixel positions X, Y
    :return: dict 'signal' (nb spots, nb images): mean of the core of the box minus 'background',
        'background' (nb spots, nb images): median of the border of the box (fluorescence...),
        'missing' (nb images): True for missing or unreadable images (nan in the profiles)
    """
    framedim = DictLT.dict_CCD[scan['CCDLabel']][0]
    boxes = _boxes(xy, boxsize, coresize, framedim)
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
def _edge(profile: np.ndarray, inside: int, start: int, direction: int, floor: float, window: int):
    """edge of a dip, going outwards (direction -1 or +1) from image inside (middle of the dip);
    start: last image of the dip in this direction

    :return: (position at half height, 20-80 % width, plateau level) or None if the profile does
        not come back to the plateau within window images after start
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
    for level in (0.2, 0.5, 0.8):
        above = np.flatnonzero(values >= floor + level * height)
        if len(above) == 0 or above[0] == 0:
            return None
        k = above[0]
        fraction = (floor + level * height - values[k - 1]) / (values[k] - values[k - 1])
        crossings.append(indices[k - 1] + direction * fraction)
    return crossings[1], abs(crossings[2] - crossings[0]), plateau


def find_dips(signal: np.ndarray, low_fraction: float = 0.3, min_edge_fraction: float = 0.5,
              min_snr: float = 8., min_signal: float = 50., min_width: int = 4,
              edge_window: int = 20) -> List[Dict[str, float]]:
    """dips (extinctions by a wire) of the intensity profile of a spot (local background subtracted)

    A dip is a range of images where the signal is lower than low_fraction * plateau (plateau: 90th
    percentile of the profile), limited by edges higher than min_edge_fraction * plateau. Smooth
    and partial dips (fluorescence, intensity fluctuations) do not fulfil these conditions.

    :return: list of dict with keys:
        'entry', 'exit': image index (float) of the edges at half height (nan for a dip cut by the
            start or the end of the scan)
        'entry_width', 'exit_width': standard deviation (nb images) of the edges (gaussian blur)
        'floor', 'plateau': signal in the dip and outside
    """
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
        entry = _edge(smooth, middle, a, -1, floor, edge_window) if a > 0 else None
        exit_ = _edge(smooth, middle, b, +1, floor, edge_window) if b < last else None
        dip = {'floor': floor, 'entry': np.nan, 'exit': np.nan, 'entry_width': np.nan,
               'exit_width': np.nan, 'plateau': np.nan}
        levels = []
        for name, edge in (('entry', entry), ('exit', exit_)):
            if edge is not None and edge[2] - floor >= min_edge_fraction * plateau:
                dip[name] = float(edge[0])
                dip[name + '_width'] = float(edge[1] / _WIDTH2080_PER_SIGMA)
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
DIPS_COLUMNS = ('spot', 'X', 'Y', 'entry', 'exit', 'entry_width', 'exit_width', 'floor', 'plateau',
                'wire', 'ybeam', 'mismatch', 'extent', 'width', 'width_calc', 'nb_candidates')
SPOTS_COLUMNS = ('X', 'Y', 'intensity', 'nb_dips', 'wire', 'ybeam', 'depth', 'spread', 'extent')


def _assign_wire(geometry: WireGeometry, x: float, y: float, dip: Dict[str, float], grid: np.ndarray,
                 tangents: tuple, yrange: Sequence[float], max_mismatch: float) -> Dict[str, float]:
    """wire and ybeam of a dip: wire for which entry and exit edges give the same ybeam, in yrange"""
    res = {'wire': -1, 'ybeam': np.nan, 'mismatch': np.nan, 'extent': np.nan, 'width_calc': np.nan,
           'nb_candidates': 0}
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
    # microns of ybeam per image, and width of the shadow, at this ybeam
    k = int(np.clip(np.searchsorted(grid, ybeam), 1, len(grid) - 1))
    dp_dy = (tangents[0][iw][k] - tangents[0][iw][k - 1]) / (grid[k] - grid[k - 1])   # mm / micron
    um_per_image = abs(geometry.step / dp_dy)
    widths = [w for w in (dip['entry_width'], dip['exit_width']) if np.isfinite(w)]
    width_calc = abs(np.interp(ybeam, grid, tangents[2][iw] - tangents[0][iw]) / geometry.step)
    res.update(wire=iw, ybeam=ybeam, mismatch=mismatch, extent=float(np.mean(widths) * um_per_image),
               width_calc=float(width_calc), nb_candidates=len(candidates))
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
            'nb_candidates' (nb of possible wires)
        spots (one line per spot): 'nb_dips' with a ybeam, 'ybeam', 'depth' (mean over its dips,
            nan if no dip), 'spread' (max - min of ybeam of its dips), 'wire' (of the best dip)
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    xy = np.asarray(xy, dtype=float)
    yrange = [float(v) for v in params['ybeam_range']]
    margin = 2. * float(params['max_mismatch'])
    grid = np.arange(yrange[0] - margin, yrange[1] + margin + 1., 2.)
    dipkeys = ('low_fraction', 'min_edge_fraction', 'min_snr', 'min_signal', 'min_width', 'edge_window')
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
                       mismatch=np.nan, extent=np.nan, width_calc=np.nan, nb_candidates=0)
            if geometry is not None:
                row.update(_assign_wire(geometry, x, y, dip, grid, tangents, yrange, params['max_mismatch']))
            rows.append(row)
            if row['wire'] >= 0:
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
    params = {**DEFAULT_PARAMS, **(params or {})}
    if str(params['peaklist']).lower() == 'auto':
        xy = find_spots(max_projection(scan, nb_cpus=nb_cpus), params['spots_threshold'],
                        params['spots_max_number'], params['boxsize'])
        print(f'{len(xy)} spots found in the scan', flush=True)
    else:
        xy = read_peaklist(params['peaklist'])
        print(f"{len(xy)} spots read in {params['peaklist']}", flush=True)
    if len(xy) == 0:
        raise RuntimeError('no spot: check peaklist or spots_threshold')
    profiles = collect_profiles(scan, xy, params['boxsize'], params['coresize'], nb_cpus, progress_callback)
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


def write_cor(results: Dict[str, Any], prefix: str, folder: str, indices: Optional[np.ndarray] = None,
              ybeam_for_angles='spot') -> str:
    """write a .cor file of spots (default: all spots with a position along the beam) with
    additional columns ybeam, depth, extent (microns), wire and nb_dips

    :param ybeam_for_angles: source position used for 2theta and chi: 'spot' (ybeam of each spot),
        or a value in microns (same source for all spots, e.g. center of a depth slice; 0: angles of
        a standard .cor file)
    :return: path of the file
    """
    geometry = geometry_of_results(results)
    if geometry is None:
        raise ValueError('no wires calibration: no position along the beam')
    spots = results['spots']
    if indices is None:
        indices = select_spots(results)
    ysource = spots['ybeam'][indices] if isinstance(ybeam_for_angles, str) else float(ybeam_for_angles)
    twicetheta, chi = geometry.scattering_angles(spots['X'][indices], spots['Y'][indices], ysource)
    detparams, _ = IOLT.readfile_det(results['detfile'], nbCCDparameters=6, verbose=0)
    source = 'ybeam of each spot' if isinstance(ybeam_for_angles, str) else f'ybeam = {ysource:.1f} microns'
    os.makedirs(folder, exist_ok=True)
    IOLT.writefile_cor(prefix, twicetheta, chi, spots['X'][indices], spots['Y'][indices],
                       spots['intensity'][indices], param=list(detparams[:5]) + [geometry.ccd_params[5]],
                       comments=f'quick daxm: 2theta and chi computed for a source at {source}',
                       dirname_output=folder, dict_data_spotsproperties=_cor_columns(spots, indices))
    return os.path.join(folder, prefix + '.cor')


def write_depth_slices(results: Dict[str, Any], prefix: str, folder: str, step: float = 5.,
                       ybeam_range: Optional[Sequence[float]] = None, thickness: Optional[float] = None,
                       min_spots: int = 1) -> List[Dict[str, Any]]:
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
            info['file'] = write_cor(results, f'{prefix}_{k:04d}', folder, indices, ybeam_for_angles=center)
        slices.append(info)
    return slices


# --------------------------------------------------------------------------------------
#  PLOTS
# --------------------------------------------------------------------------------------
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
    found = dips['wire'] >= 0
    ax.plot(dips['Y'][~found], center[~found], '.', color='0.6', ms=3, label='dips without wire')
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
