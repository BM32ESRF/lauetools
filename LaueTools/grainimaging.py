"""
Mosaic and grain imaging of 2D maps of Laue patterns (no indexation): pixel intensities in ROIs of the detector
monitored over all the images of a map

- browser of the datasets of an experiment (folders of RAW_DATA, filter on the dataset name: fast for experiments
  with thousands of datasets) and of the scans of a dataset (its own small BLISS .h5 file)
- one Laue pattern of the map: peak search, positions picked with the mouse, regular grid of ROI centers
- computation of the ROI counters (LaueTools.scripts.workflows.ROICountersWorkflow) in this kernel or as a job
  (this machine, SLURM partition), results reloaded when computed before with the same parameters
- mosaic: pixels of a ROI in every image of the map, arranged as the map (one big image)
- grain imaging: map of the intensity of each Laue spot (a grain lights its spots where it is), overview of the
  ROIs lit at each map point

Used by notebooks/mosaicgrainimaging/mosaic_grainimaging.ipynb and notebooks/quickstart/laue_maps_quickstart.ipynb

ROI centers (X, Y) are pixel indices of the image array, image[Y, X] (first pixel at (0, 0)), as in
LaueTools.scripts.workflows.read_rois(). Peak positions of the peak search start at (1, 1): X - 1, Y - 1
"""
import contextlib
import io
import os
import time
from pathlib import Path
from contextlib import nullcontext as _nullcontext
from typing import Any, Dict, List, Optional, Union

import numpy as np

from LaueTools import generaltools as GT
import LaueTools.scripts.workflows as wf

# default detector (H5file of a dataset) when it is not found from the image files
DEFAULT_CCDLABEL = 'EIGER_4MCdTe'
# max nb of datasets in the menu of DatasetSelector (filter to see the others)
MAX_MENU_DATASETS = 1000


# --------------------------------------------------------------------------------------
#  BROWSER OF THE DATASETS OF AN EXPERIMENT (RAW_DATA folders)
# --------------------------------------------------------------------------------------
def raw_data_folders(exp_id: Optional[str] = None, parent: Union[str, Path] = '/data/visitor',
                     beamline: str = 'bm32', dates=None, raw_data_folder=None) -> List[tuple]:
    """[(date, RAW_DATA folder)] of an experiment: <parent>/<exp_id>/<beamline>/<date>/RAW_DATA for the dates
    (session folders) of dates (None: all), or [('', raw_data_folder)] if raw_data_folder is given"""
    if raw_data_folder is not None:
        return [('', Path(raw_data_folder))]
    if not exp_id:
        raise ValueError('exp_id (experiment, e.g. a321217) or raw_data_folder is needed')
    base = Path(parent) / exp_id / beamline
    if not base.is_dir() and (Path(parent) / exp_id.lower() / beamline).is_dir():
        base = Path(parent) / exp_id.lower() / beamline
    if not base.is_dir():
        raise FileNotFoundError(f'no folder {base}')
    dates = [dates] if isinstance(dates, (str, int)) else dates
    sessions = sorted(d for d in base.iterdir() if d.is_dir() and (dates is None or d.name in map(str, dates)))
    folders = [(d.name, d / 'RAW_DATA') for d in sessions if (d / 'RAW_DATA').is_dir()]
    if dates is not None and len(folders) < len(dates):
        GT.printyellow(f"date(s) without RAW_DATA in {base}: {sorted(set(map(str, dates)) - {f[0] for f in folders})}")
    if not folders:
        raise FileNotFoundError(f'no RAW_DATA folder in {base}/<date>')
    return folders


def _matches(text: str, pattern: Optional[str]) -> bool:
    """all the words of pattern are in text (case insensitive); a word starting with '-' must not be in it"""
    text = text.lower()
    for word in (pattern or '').lower().split():
        if word.startswith('-') and len(word) > 1:
            if word[1:] in text:
                return False
        elif word not in text:
            return False
    return True


def find_datasets(exp_id: Optional[str] = None, parent: Union[str, Path] = '/data/visitor', beamline: str = 'bm32',
                  dates=None, raw_data_folder=None, contains: Optional[str] = None, verbose: bool = True):
    """datasets of an experiment: folders RAW_DATA/<sample>/<dataset> (BLISS), listed without reading any .h5 file
    (fast, also for thousands of datasets)

    exp_id, parent, beamline, dates: experiment folders (see raw_data_folders()), or raw_data_folder
    contains: keep the datasets whose name contains all the words of this text (case insensitive; '-word':
        without this word), e.g. 'map2D -align'
    Returns a table (pandas DataFrame) with the columns 'date', 'sample', 'dataset', 'path' (dataset folder),
    'h5file' (.h5 file of the dataset, its scans), 'modified', for DatasetSelector()"""
    import datetime
    import pandas as pd
    t0 = time.time()
    rows = []
    for date, raw in raw_data_folders(exp_id, parent, beamline, dates, raw_data_folder):
        with os.scandir(raw) as samples:
            samples = sorted(e.path for e in samples if e.is_dir() and not e.name.startswith(('_', '.')))
        for sample in samples:
            try:
                with os.scandir(sample) as entries:
                    datasets = sorted((e.name, e.path, e.stat().st_mtime) for e in entries if e.is_dir())
            except OSError:
                continue
            for name, path, mtime in datasets:
                if not _matches(name, contains):
                    continue
                rows.append({'date': date, 'sample': Path(sample).name, 'dataset': name, 'path': path,
                             'h5file': os.path.join(path, f'{name}.h5'),
                             'modified': datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')})
    datasets = pd.DataFrame(rows, columns=['date', 'sample', 'dataset', 'path', 'h5file', 'modified'])
    if verbose:
        print(f"{len(datasets)} datasets{f' containing {contains!r}' if contains else ''} "
              f"({time.time() - t0:.1f} s)")
    return datasets


def _nbdigits(folder, prefix: str, suffix: str, default: int = 4) -> int:
    try:
        with os.scandir(folder) as entries:
            name = next(e.name for e in entries if e.name.startswith(prefix) and e.name.endswith(suffix))
        return len(name[len(prefix):-len(suffix)])
    except (StopIteration, OSError):
        return default


def list_scans(h5file, CCDLabel: Optional[str] = None, count_images: bool = True):
    """scans of the .h5 file of a dataset (maps, line scans ...): table (pandas DataFrame) with the columns
    'scantype', 'command', 'end', 'images' (found / expected), 'imagefolder'; index: scan index of
    scan_params()"""
    import pandas as pd
    from LaueTools import logfile_reader as iohdf5
    from LaueTools import map_workflow as MW
    lf = iohdf5.H5file(str(h5file), CCDLabel or DEFAULT_CCDLABEL)
    lf.getscans(verbose=0)
    rows = []
    for idx, row in lf.dfallscans.iterrows():
        try:
            dc = iohdf5.read_fullcommand(row['fullcommand'])
            npts = (dc['fmotnbsteps'] + 1) * (dc['smotnbsteps'] + 1 if dc.get('smotnbsteps') is not None else 1)
        except Exception:
            npts = None
        images = f"?/{npts or '?'}"
        if count_images:
            _, prefix, suffix = MW.guess_detector(row['imagefolder'])
            if prefix:
                try:
                    with os.scandir(row['imagefolder']) as entries:
                        names = sorted(e.name for e in entries if e.name.startswith(prefix) and e.name.endswith(suffix))
                except OSError:
                    names = []
                images = f"{len(names)}/{npts or '?'}"
                if names and suffix == '.h5':   # stacked EIGER files (e.g. fscan2d: one file per line)
                    from LaueTools import IOimagefile as IOimage
                    nbframes = IOimage.get_eiger_h5_nbframes(os.path.join(row['imagefolder'], names[0]))
                    if nbframes and nbframes > 1:
                        images = f"{len(names)} files x {nbframes} frames/{npts or '?'}"
        rows.append({'index': int(idx), 'scantype': row['scantype'], 'command': row['fullcommand'],
                     'end': row['endreason'], 'images': images,
                     'imagefolder': str(row['imagefolder'])})
    return pd.DataFrame(rows, columns=['index', 'scantype', 'command', 'end', 'images', 'imagefolder']
                        ).set_index('index')


def _detect_stack(params: Dict[str, Any]) -> Dict[str, Any]:
    """EIGER images saved as stacked files (several frames per .h5 file, e.g. fscan2d: one file per line of the
    map): CCDLabel 'EIGER_4MCdTestack', 'nbframes_per_file' and 'filename_representative' set in params"""
    from LaueTools import IOimagefile as IOimage
    if params['CCDLabel'] not in IOimage.EIGER_SINGLE_STACK_LABELS:
        return params
    first = os.path.join(params['folder'], f"{params['prefix']}{0:0{params['nbdigits']}d}.{params['suffix']}")
    if not os.path.exists(first):
        return params
    label, nbframes = IOimage.autodetect_eiger_stack_label(params['CCDLabel'], first)
    if label in IOimage.STACK_CCDLABELS:
        params.update(CCDLabel=label, nbframes_per_file=int(IOimage.get_stack_nbframes_per_file(first) or nbframes),
                      filename_representative=os.path.basename(first))
    return params


def scan_params(h5file, scan_index: int, CCDLabel: Optional[str] = None) -> Dict[str, Any]:
    """parameters of the images of a scan (scan_index of list_scans()) for the ROI counters workflow:
    'folder', 'CCDLabel', 'prefix', 'suffix', 'nbdigits', 'listindices', 'mapdimensions' (fast, slow),
    'fastaxis', 'slowaxis', 'fullcommand', 'dataset', 'h5file', 'scan_index'.
    CCDLabel: None: found from the image files of the scan"""
    from LaueTools import logfile_reader as iohdf5
    from LaueTools import map_workflow as MW
    lf = iohdf5.H5file(str(h5file), CCDLabel or DEFAULT_CCDLABEL)
    lf.getscans(verbose=0)
    d = lf.build_dict_scan(scan_index, verbose=False)
    if d.get('scantype') is None:
        raise ValueError(f"scan {scan_index} of {h5file} is not a map or a line scan: {d.get('fullcommand')}")
    folder = str(d['imagefolder'])
    found, prefix, suffix = MW.guess_detector(folder)
    if CCDLabel is not None and CCDLabel != found:
        prefix, suffix = MW.DETECTOR_FILES.get(CCDLabel, (d['prefix'], '.' + d['suffix'].lstrip('.')))
    ccdlabel = CCDLabel or found
    if ccdlabel is None:
        raise FileNotFoundError(f'no image (eiger4m_*.h5, img_*.tif) in {folder}: set CCDLabel')
    mapdims = [int(v) for v in d['mapdimensions']]
    return _detect_stack({'folder': folder, 'CCDLabel': ccdlabel, 'prefix': prefix, 'suffix': suffix.lstrip('.'),
            'nbdigits': _nbdigits(folder, prefix, suffix), 'listindices': list(range(mapdims[0] * mapdims[1])),
            'mapdimensions': mapdims, 'fastaxis': d.get('fastaxis'), 'slowaxis': d.get('slowaxis'),
            'fullcommand': d['fullcommand'], 'dataset': d['sample_dataset_scanindex'], 'h5file': str(h5file),
            'scan_index': int(scan_index)})


def scan_params_from_info(info: dict) -> Dict[str, Any]:
    """parameters of the images of a map for the ROI counters workflow (see scan_params()) from the map chosen in
    the quickstart notebook (map_workflow.scan_info())"""
    nfast, nslow = info['mapdims']
    return _detect_stack({'folder': str(info['image_folder']), 'CCDLabel': info['CCDLabel'], 'prefix': info['prefix'],
            'suffix': info['suffix'].lstrip('.'), 'nbdigits': info['nbdigits'],
            'listindices': list(range(nfast * nslow)), 'mapdimensions': [nfast, nslow],
            'fastaxis': info['fast_motor'], 'slowaxis': info['slow_motor'], 'fullcommand': info['command'],
            'dataset': info['name'], 'h5file': str(info.get('hdf5_file')), 'scan_index': info['index']})


def print_scan(params: Dict[str, Any]):
    """summary of the scan of scan_params()"""
    nfast, nslow = params['mapdimensions']
    print(f"{params['dataset']}: {params['fullcommand']}")
    print(f"  map      : {nfast} x {nslow} points (fast motor {params['fastaxis']}, slow motor {params['slowaxis']})")
    stack = (f", {params['nbframes_per_file']} frames per file" if params.get('nbframes_per_file') else '')
    print(f"  images   : {params['folder']}/{params['prefix']}{'#' * params['nbdigits']}.{params['suffix']} "
          f"({params['CCDLabel']}{stack})")


class DatasetSelector:
    """menus to choose a dataset of find_datasets() (filter on the dataset name: all the words, '-word': without
    it) and one of its scans (read from the .h5 file of the dataset). Selection: DatasetSelector.h5file,
    DatasetSelector.scan_index; DatasetSelector.scan_params(): parameters of the images of the scan"""

    def __init__(self, datasets, CCDLabel: Optional[str] = None, contains: str = ''):
        self.datasets = datasets
        self.CCDLabel = CCDLabel
        self.h5file, self.scan_index, self.scans = None, None, None
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print('ipywidgets not installed: set H5FILE and SCAN_INDEX in SETTINGS (table: datasets)')
            return
        w_sample = max([len(s) for s in datasets['sample']] + [6])
        self.labels = [(f"{r['date']} | {r['sample']:<{w_sample}} | {r['dataset']}", i) for i, r in datasets.iterrows()]
        self.filter = widgets.Text(value=contains, description='filter:', continuous_update=False,
                                   placeholder="words of the dataset name, '-word': without it",
                                   layout=widgets.Layout(width='60%'))
        self.count = widgets.HTML()
        self.menu = widgets.Select(options=[], rows=12, layout=widgets.Layout(width='98%'))
        self.scanmenu = widgets.Select(options=[], rows=6, layout=widgets.Layout(width='98%'))
        self.info = widgets.HTML()
        for w in (self.menu, self.scanmenu):
            w.add_class('lt-dataset-menu')
        style = widgets.HTML('<style>.lt-dataset-menu select {font-family: monospace; font-size: 12px}</style>')
        self.filter.observe(lambda change: self._apply_filter(change['new']), names='value')
        self.menu.observe(self._on_dataset, names='value')
        self.scanmenu.observe(self._on_scan, names='value')
        self._apply_filter(contains)
        display(widgets.VBox([style, widgets.HBox([self.filter, self.count]), self.menu,
                              widgets.HTML('<b>scans of the dataset</b> (maps: amesh, fscan2d, ...)'),
                              self.scanmenu, self.info]))

    def _apply_filter(self, text):
        options = [o for o in self.labels if _matches(self.datasets.loc[o[1], 'dataset'], text)]
        self.count.value = (f'&nbsp;{len(options)} datasets'
                            + (f' (first {MAX_MENU_DATASETS} shown)' if len(options) > MAX_MENU_DATASETS else ''))
        current = self.menu.value
        self.menu.options = options[:MAX_MENU_DATASETS]
        if current in [o[1] for o in self.menu.options]:
            self.menu.value = current

    def _on_dataset(self, change):
        if self.menu.value is None:
            return
        row = self.datasets.loc[self.menu.value]
        self.h5file, self.scan_index = row['h5file'], None
        if not Path(row['h5file']).exists():
            self.scanmenu.options = []
            self.info.value = f"<span style='color:red'>no .h5 file {row['h5file']}</span>"
            return
        try:
            self.scans = list_scans(row['h5file'], self.CCDLabel)
        except Exception as err:
            self.scanmenu.options = []
            self.info.value = f"<span style='color:red'>{row['h5file']}: {type(err).__name__} {err}</span>"
            return
        self.scanmenu.options = [(f"{i:3d} | {r['scantype']} | {r['command']} | {r['end']} | images {r['images']}", int(i))
                                 for i, r in self.scans.iterrows()]
        if not len(self.scans):
            self.info.value = f"no map (or line scan) in {row['h5file']}"
        else:
            self.scanmenu.value = int(self.scans.index[0])

    def _on_scan(self, change):
        if self.scanmenu.value is None:
            return
        self.scan_index = int(self.scanmenu.value)
        r = self.scans.loc[self.scan_index]
        self.info.value = (f"<b>H5FILE = '{self.h5file}', SCAN_INDEX = {self.scan_index}</b><br>"
                           f"<code>{r['command']}</code> ({r['end']}, images {r['images']})<br>{r['imagefolder']}"
                           f"<br>then run the next cell")

    def scan_params(self) -> Dict[str, Any]:
        if self.h5file is None or self.scan_index is None:
            raise ValueError('choose a dataset and a scan in the menus first')
        return scan_params(self.h5file, self.scan_index, self.CCDLabel)


# --------------------------------------------------------------------------------------
#  ONE LAUE PATTERN: image, peak search, ROI centers
# --------------------------------------------------------------------------------------
def image_path(params: Dict[str, Any], imageindex: Optional[int] = None):
    """(path of the image file, frame index, image index) of an image of the scan (None: middle of the map, or the
    first image if this one is missing, e.g. scan aborted)"""
    if imageindex is None:
        for imageindex in (params['listindices'][len(params['listindices']) // 2], params['listindices'][0]):
            path, frame = wf.image_filepath(params, int(imageindex))
            if os.path.exists(path):
                break
    path, frame = wf.image_filepath(params, int(imageindex))
    if not os.path.exists(path):
        raise FileNotFoundError(f'image {imageindex} not recorded: no file {path}')
    return path, frame, int(imageindex)


def read_image(params: Dict[str, Any], imageindex: Optional[int] = None) -> np.ndarray:
    """2D array of the pixel intensities of an image of the scan (None: middle of the map)"""
    from LaueTools import IOimagefile as IOimage
    path, frame, _ = image_path(params, imageindex)
    return IOimage.readCCDimage(path, CCDLabel=params['CCDLabel'], stackimageindex=frame)[0]


class TooManyPeaksError(ValueError):
    """peak search given up: more local maxima than NumberMaxofFits (readmccd.PeakSearch() returns no peak)"""


def peaksearch(params: Dict[str, Any], imageindex: Optional[int] = None, verbose: bool = True,
               **changes) -> Optional[np.ndarray]:
    """peak search of an image of the scan (no file written) with the default parameters of the detector and
    changes (names of the peaksearch section of peaksearch.yaml, e.g. IntensityThreshold=300).
    Returns the peak list (nb peaks, 10 columns: X, Y (first pixel at (1, 1)), I, fwhm ...), most intense first.
    Raises TooManyPeaksError if the image has more local maxima than NumberMaxofFits (no peak fitted)"""
    import tempfile
    from LaueTools import peaksearch_batch as PB
    from LaueTools import map_workflow as MW
    from LaueTools import readmccd as RMCCD
    path, frame, imageindex = image_path(params, imageindex)
    if verbose:
        print(f'peak search of image {imageindex} running ...', flush=True)
    detector = params['CCDLabel'].replace('stack', '')   # peak search parameters of the detector
    ps = {**PB.DEFAULT_CONFIG['peaksearch'], **MW.DETECTOR_PEAKSEARCH.get(detector, {}), **changes}
    unknown = set(changes) - set(PB.DEFAULT_CONFIG['peaksearch'])
    if unknown:
        raise ValueError(f'unknown peak search parameters {sorted(unknown)}: see PB.DEFAULT_CONFIG["peaksearch"]')
    psparams = PB.PeakSearchParams(CCDLabel=params['CCDLabel'], prefix=params['prefix'],
                                   nbdigits=params['nbdigits'], dat_folder=tempfile.gettempdir(), peaksearch=ps,
                                   writefiles=False, write_dat=False)
    kwargs = psparams.peaksearch_kwargs()
    kwargs['verbose'] = 1   # message of PeakSearch() when it gives up (too many local maxima), captured
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = RMCCD.PeakSearch(path, stackimageindex=frame, **kwargs)
    if res is None and 'TOO MUCH peaks' in out.getvalue():
        raise TooManyPeaksError(f"image {imageindex}: more local maxima than NumberMaxofFits "
                                f"({ps['NumberMaxofFits']}), no peak fitted: increase IntensityThreshold "
                                f"({ps['IntensityThreshold']}) or NumberMaxofFits")
    peaks = None if res is None else res[0]
    if verbose:
        print(f"peak search done: image {imageindex}, {0 if peaks is None else len(peaks)} peaks "
              f"(IntensityThreshold {ps['IntensityThreshold']})")
    if peaks is not None and len(peaks):
        peaks = peaks[np.argsort(-peaks[:, 2], kind='stable')]
    return peaks


def roicenters_from_peaks(peaks: Optional[np.ndarray], nbmax: Optional[int] = None) -> List[List[int]]:
    """ROI centers (X, Y) of the nbmax first (most intense) peaks of a peak list (peak positions start at (1, 1))"""
    if peaks is None or not len(peaks):
        return []
    XY = np.rint(np.asarray(peaks)[:nbmax, :2] - 1).astype(int)
    return [[int(x), int(y)] for x, y in XY]


def grid_roicenters(CCDLabel: str, step, boxsize=(8, 8), xlim=None, ylim=None, around=None,
                    margin: int = 100) -> List[List[int]]:
    """regular grid of ROI centers (X, Y) on the detector, step (pixel) or (stepX, stepY)

    xlim, ylim: (min, max) pixel ranges (default: whole detector, ROIs inside)
    around: list of (X, Y) (e.g. picked positions): grid in the rectangle around them (+- margin pixel)"""
    from LaueTools import dict_LaueTools as DictLT
    bx, by = boxsize
    if around is not None and len(around):
        framedim = DictLT.dict_CCD[CCDLabel][0]
        XY = np.asarray(around, dtype=float).reshape(-1, 2)
        xlim = (max(bx, int(XY[:, 0].min() - margin)), min(framedim[1] - 1 - bx, int(XY[:, 0].max() + margin)))
        ylim = (max(by, int(XY[:, 1].min() - margin)), min(framedim[0] - 1 - by, int(XY[:, 1].max() + margin)))
    return wf.roicenters_grid(CCDLabel, step, bx, by, xlim, ylim)


class RoiPicker:
    """Laue pattern with peaks, grid of ROI centers and the positions picked with the mouse (interactive backend
    %matplotlib widget):
    - click: picks the position (snap 'peak': nearest peak within snap_radius pixel, 'max': pixel of max intensity
      in the ROI around the click, None: clicked pixel). Not while zoom or pan is active in the toolbar
    - right click (or key 'd'): removes the nearest picked position
    RoiPicker.picked: list of (X, Y) picked. picker.add(X, Y), picker.remove(k): same by program"""

    def __init__(self, image: np.ndarray, peaks: Optional[np.ndarray] = None, grid=None, boxsize=(8, 8),
                 picked=None, snap: Optional[str] = 'peak', snap_radius: float = 15, percentiles=(1, 99.9),
                 title: str = '', figsize=(8, 8)):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        self._Rectangle = Rectangle
        self.image = np.asarray(image)
        self.peaks, self.grid = [], []
        self.boxsize = tuple(boxsize)
        self.snap, self.snap_radius = snap, snap_radius
        self.picked = []
        self._patches = []
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=figsize)
        vmin, vmax = np.percentile(self.image, percentiles)
        self._im = self.ax.imshow(self.image, cmap='gray_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        self._peaks_artist = self.ax.scatter([], [], marker='+', color='limegreen', s=40, lw=1, label='peaks')
        self._grid_artist = self.ax.scatter([], [], marker='.', color='tab:blue', s=6, label='grid')
        self.set_peaks(peaks, draw=False)
        self.set_grid(grid, draw=False)
        self.ax.set_xlim(-0.5, self.image.shape[1] - 0.5)
        self.ax.set_ylim(self.image.shape[0] - 0.5, -0.5)
        self.ax.format_coord = self._format_coord
        for xy in picked or []:
            self.add(*xy)
        self._update_title()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._mouse = None

    def _legend(self):
        handles = [a for a, n in ((self._peaks_artist, len(self.peaks)), (self._grid_artist, len(self.grid))) if n]
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        if handles:
            self.ax.legend(handles=handles, loc='upper right', fontsize=8)

    def set_image(self, image: np.ndarray, title: Optional[str] = None, draw: bool = True):
        """change the image (zoom and contrast kept)"""
        self.image = np.asarray(image)
        self._im.set_data(self.image)
        if title is not None:
            self.title = title
        if draw:
            self._update_title()

    def set_peaks(self, peaks: Optional[np.ndarray], draw: bool = True):
        """change the peaks (peak list of the peak search, positions starting at (1, 1)), None: no peak"""
        self.peaks = roicenters_from_peaks(peaks) if peaks is not None else []
        self._peaks_artist.set_offsets(np.array(self.peaks, dtype=float).reshape(-1, 2))
        self._peaks_artist.set_label(f'{len(self.peaks)} peaks')
        self._legend()
        if draw:
            self.fig.canvas.draw_idle()

    def set_grid(self, grid, draw: bool = True):
        """change the grid of ROI centers (list of (X, Y)), None: no grid"""
        self.grid = [list(map(int, g)) for g in (grid or [])]
        self._grid_artist.set_offsets(np.array(self.grid, dtype=float).reshape(-1, 2))
        self._grid_artist.set_label(f'grid {len(self.grid)} ROIs')
        self._legend()
        if draw:
            self.fig.canvas.draw_idle()

    def set_clim(self, vmin: float, vmax: float):
        """contrast of the image"""
        self._im.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _format_coord(self, x, y):
        X, Y = int(round(x)), int(round(y))
        if 0 <= X < self.image.shape[1] and 0 <= Y < self.image.shape[0]:
            return f'X {X}, Y {Y}, I {self.image[Y, X]:.0f}'
        return f'X {x:.0f}, Y {y:.0f}'

    def _snapped(self, x, y):
        X, Y = int(round(x)), int(round(y))
        if self.snap == 'peak' and self.peaks:
            P = np.array(self.peaks)
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            k = int(np.argmin(d))
            if d[k] <= self.snap_radius:
                return int(P[k, 0]), int(P[k, 1])
        if self.snap in ('max', 'peak'):
            bx, by = self.boxsize
            y0, x0 = max(Y - by, 0), max(X - bx, 0)
            sub = self.image[y0:Y + by + 1, x0:X + bx + 1]
            if sub.size:
                j, i = np.unravel_index(int(np.argmax(sub)), sub.shape)
                return x0 + int(i), y0 + int(j)
        return X, Y

    def add(self, X, Y, snap: bool = False):
        """pick (X, Y) (snapped as a click if snap)"""
        X, Y = self._snapped(X, Y) if snap else (int(X), int(Y))
        bx, by = self.boxsize
        k = len(self.picked)
        self.picked.append([X, Y])
        rect = self.ax.add_patch(self._Rectangle((X - bx - 0.5, Y - by - 0.5), 2 * bx + 1, 2 * by + 1, fill=False,
                                                 ec='r', lw=1.5))
        text = self.ax.annotate(str(k), (X + bx + 2, Y - by - 2), color='r', fontsize=9)
        self._patches.append((rect, text))
        self._update_title()

    def remove(self, k: int):
        """remove the picked position k"""
        self.picked.pop(k)
        for artist in self._patches.pop(k):
            artist.remove()
        for j, (_, text) in enumerate(self._patches):
            text.set_text(str(j))
        self._update_title()

    def _update_title(self):
        self.ax.set_title(f"{self.title}\n{len(self.picked)} picked position(s): {self.picked[:6]}"
                          + (' ...' if len(self.picked) > 6 else ''), fontsize=9)
        self.fig.canvas.draw_idle()

    def _nearest_picked(self, x, y):
        if not self.picked:
            return None
        P = np.array(self.picked)
        return int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))

    def _on_click(self, event):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if event.inaxes is not self.ax or (toolbar is not None and getattr(toolbar, 'mode', '')):
            return
        if event.button == 3:
            k = self._nearest_picked(event.xdata, event.ydata)
            if k is not None:
                self.remove(k)
        elif event.button == 1:
            self.add(event.xdata, event.ydata, snap=True)

    def _on_key(self, event):
        if event.key == 'd' and event.inaxes is self.ax:
            k = self._nearest_picked(event.xdata, event.ydata)
            if k is not None:
                self.remove(k)


class LaueImageBrowser:
    """browser of the Laue patterns of a scan (interactive backend %matplotlib widget and ipywidgets):
    - image index: slider or box; the zoom is kept when the image changes
    - contrast: one range slider (vmin, vmax), kept when the image changes
    - 'peak search' button: peak search of the current image (see peaksearch(), changes of its parameters:
      peaksearch dict, e.g. {'IntensityThreshold': 300}), peaks shown as green +; checkbox: peak search of each
      new image. No peak search before the button is pressed (run_peaksearch=True: peak search of the first image)
    - positions picked with the mouse as in RoiPicker: click (snapped on the nearest peak, or on the brightest
      pixel of the ROI around the click), right click or key 'd' removes the nearest one
    Attributes: picked (picked positions (X, Y)), peaks (peak list of the last peak search) of the image
    peaks_imageindex, imageindex (current image), picker (RoiPicker: figure)"""

    def __init__(self, params: Dict[str, Any], imageindex: Optional[int] = None, peaksearch: Optional[dict] = None,
                 boxsize=(8, 8), grid=None, picked=None, snap: Optional[str] = 'peak', snap_radius: float = 15,
                 run_peaksearch: bool = False, figsize=(8, 8)):
        import matplotlib.pyplot as plt
        self.params = params
        self.peaksearch_params = dict(peaksearch or {})
        self.peaks, self.peaks_imageindex = None, None
        _, _, self.imageindex = image_path(params, imageindex)
        image = read_image(params, self.imageindex)
        lo, hi = (float(v) for v in np.percentile(image, (1, 99.9)))
        try:
            import ipywidgets as widgets
            from IPython.display import display
            interactive = 'inline' not in plt.get_backend().lower()
        except ImportError:
            interactive = False
        with plt.ioff() if interactive else _nullcontext():
            self.picker = RoiPicker(image, None, grid, boxsize, picked, snap, snap_radius, title=self._title(),
                                    figsize=figsize)
        self.picker.set_clim(lo, hi)
        if not interactive:
            GT.printyellow('no interactive figure (%matplotlib widget, ipywidgets): no slider nor picking with the '
                           'mouse; browser.show_image(index), browser.run_peaksearch(), browser.picker.add(X, Y)')
            if run_peaksearch:
                self.run_peaksearch()
            return
        indices = params['listindices']
        top = float(max(hi * 10, np.percentile(image, 99.99), lo + 1))
        self.w_index = widgets.IntSlider(value=self.imageindex, min=int(min(indices)), max=int(max(indices)),
                                         description='image', continuous_update=False,
                                         layout=widgets.Layout(width='60%'))
        self.w_indexbox = widgets.BoundedIntText(value=self.imageindex, min=int(min(indices)), max=int(max(indices)),
                                                 layout=widgets.Layout(width='110px'))
        widgets.jslink((self.w_index, 'value'), (self.w_indexbox, 'value'))
        self.w_contrast = widgets.FloatRangeSlider(value=[lo, hi], min=float(min(0., lo)), max=top,
                                                   step=max(top / 2000, 0.01), description='vmin, vmax',
                                                   readout_format='.0f', continuous_update=False,
                                                   layout=widgets.Layout(width='60%'))
        self.w_peaksearch = widgets.Button(description='peak search', tooltip='peak search of the current image')
        self.w_auto = widgets.Checkbox(value=False, description='peak search of each new image', indent=False)
        self.w_info = widgets.HTML()
        self.w_index.observe(lambda change: self.show_image(change['new']), names='value')
        self.w_contrast.observe(lambda change: self.picker.set_clim(*change['new']), names='value')
        self.w_peaksearch.on_click(lambda _: self.run_peaksearch())
        self.picker.fig.canvas.header_visible = False
        display(widgets.VBox([self.picker.fig.canvas, widgets.HBox([self.w_index, self.w_indexbox]),
                              self.w_contrast, widgets.HBox([self.w_peaksearch, self.w_auto, self.w_info])]))
        if run_peaksearch:
            self.run_peaksearch()
        else:
            self._message("press 'peak search' for the peaks of the current image")

    @property
    def picked(self) -> List[List[int]]:
        return self.picker.picked

    def roicenters(self, source='picked', nb_peaks: Optional[int] = None, grid=None) -> List[List[int]]:
        """ROI centers (X, Y) from source: 'picked' (positions picked with the mouse), 'peaks' (nb_peaks most
        intense peaks of the last peak search, None: all), 'grid' (grid of the browser, or grid given here), or
        several of them: 'picked+peaks' or ['picked', 'grid']. Duplicates removed"""
        sources = source.replace(' ', '').split('+') if isinstance(source, str) else list(source)
        unknown = set(sources) - {'picked', 'peaks', 'grid'}
        if unknown:
            raise ValueError(f"unknown ROI source(s) {sorted(unknown)}: 'picked', 'peaks', 'grid' (e.g. 'picked+peaks')")
        centers = []
        for src in sources:
            if src == 'picked':
                if not self.picked:
                    GT.printyellow('no picked position: click on the Laue pattern of the browser')
                centers += self.picked
            elif src == 'peaks':
                if self.peaks is None:
                    GT.printyellow("no peak: press the 'peak search' button of the browser")
                centers += roicenters_from_peaks(self.peaks, nb_peaks)
            else:
                g = self.picker.grid if grid is None else grid
                if not g:
                    GT.printyellow('no grid of ROIs (grid step not set)')
                centers += [list(map(int, c)) for c in g]
        unique = list(dict.fromkeys(tuple(c) for c in centers))
        print(f"{len(unique)} ROIs from {'+'.join(sources)}"
              + (f' (peaks of image {self.peaks_imageindex})' if 'peaks' in sources and self.peaks is not None else ''))
        return [list(c) for c in unique]

    def _title(self) -> str:
        peaks = ('' if self.peaks_imageindex is None else
                 f", {0 if self.peaks is None else len(self.peaks)} peaks of image {self.peaks_imageindex}")
        return f"{self.params.get('dataset', '')} image {self.imageindex}{peaks}"

    def _message(self, text: str, color: str = 'black'):
        if hasattr(self, 'w_info'):
            self.w_info.value = f"<span style='color:{color}'>{text}</span>"
        elif color == 'red':
            GT.printred(text)

    def show_image(self, imageindex: int):
        """show the image imageindex (zoom and contrast kept)"""
        try:
            image = read_image(self.params, imageindex)
        except (OSError, ValueError, IndexError) as err:
            self._message(f'image {imageindex}: {err}', 'red')
            return
        self.imageindex = int(imageindex)
        if hasattr(self, 'w_contrast') and np.percentile(image, 99.9) > self.w_contrast.max:
            self.w_contrast.max = float(np.percentile(image, 99.99))
        if hasattr(self, 'w_auto') and self.w_auto.value:
            self.picker.set_image(image, draw=False)
            self.run_peaksearch()
        else:
            self.picker.set_image(image, self._title())
            self._message('')

    def run_peaksearch(self):
        """peak search of the current image (parameters: peaksearch of the browser). While it runs: button
        'running...' (disabled) and message"""
        button = getattr(self, 'w_peaksearch', None)
        if button is not None:
            button.disabled, button.description, button.icon = True, 'running...', 'spinner'
        try:
            self._peaksearch()
        finally:
            if button is not None:
                button.disabled, button.description, button.icon = False, 'peak search', ''

    def _peaksearch(self):
        t0 = time.time()
        self._message(f'peak search of image {self.imageindex} running ...', 'darkorange')
        if not hasattr(self, 'w_info'):
            print(f'peak search of image {self.imageindex} running ...', flush=True)
        try:
            self.peaks = peaksearch(self.params, self.imageindex, verbose=False, **self.peaksearch_params)
        except TooManyPeaksError as err:   # no peak: markers of a previous peak search removed
            self.peaks, self.peaks_imageindex = None, self.imageindex
            self.picker.set_peaks(None, draw=False)
            self.picker.title = self._title()
            self.picker._update_title()
            self._message(str(err), 'red')
            return
        except Exception as err:
            self._message(f'peak search: {type(err).__name__} {err}', 'red')
            return
        self.peaks_imageindex = self.imageindex
        self.picker.set_peaks(self.peaks, draw=False)
        self.picker.title = self._title()
        self.picker._update_title()
        threshold = self.peaksearch_params.get('IntensityThreshold')
        text = (f"peak search done: image {self.imageindex}, {0 if self.peaks is None else len(self.peaks)} peaks"
                + (f' (IntensityThreshold {threshold})' if threshold is not None else '')
                + f', {time.time() - t0:.1f} s')
        self._message(text, 'green')
        if not hasattr(self, 'w_info'):
            GT.printgreen(text)


# --------------------------------------------------------------------------------------
#  COMPUTATION of the ROI counters
# --------------------------------------------------------------------------------------
def results_folder(params: Dict[str, Any], output: str = 'mirror') -> str:
    """folder of the results: 'image' (folder of the images, if writable), 'mirror' (PROCESSED_DATA folder of the
    images) or path of a folder (created)"""
    import LaueTools.scripts.workflowlauncher as WL
    if output in ('image', 'mirror'):
        return WL.output_folder(params['folder'], output)
    os.makedirs(output, exist_ok=True)
    return str(output)


def run(params: Dict[str, Any], roicenters, counters=('max',), boxsize=(8, 8), output: str = 'mirror',
        resource: str = 'kernel', nb_cpus: Optional[int] = None, recompute: bool = False,
        time_limit: str = '01:00:00', prefix: str = 'roicounters', **extra) -> Dict[str, Any]:
    """ROI counters (e.g. 'mosaic', 'max', 'XYcentroid', see workflows.ROI_COUNTERS) of roicenters in all the images
    of the scan. Results file <results folder>/<prefix>_<hash of the parameters>.h5: reloaded if computed before
    with the same parameters (recompute=True: computed again)

    output: results folder (see results_folder())
    resource: 'kernel' (in this notebook, blocking), or a computing resource of
        workflowlauncher.COMPUTING_RESOURCES: 'local' (background process on this machine), 'magnifix', 'magnifix3'
        (3 nodes), 'nice-hpc7' ... (SLURM job, the notebook waits for its end)
    extra: other parameters of the workflow, e.g. fitparameters={'peaksizeStart': 2.} for the counter 'fit'"""
    import LaueTools.scripts.workflowlauncher as WL
    p = {**params, **extra, 'roicenters': [list(map(int, c)) for c in roicenters], 'counters': list(counters),
         'boxsize_X': int(boxsize[0]), 'boxsize_Y': int(boxsize[1])}
    if not p['roicenters']:
        raise ValueError('no ROI center')
    folder = results_folder(params, output)
    if resource == 'kernel':
        return wf.run_roi_counters_cached(p, folder, nb_cpus=nb_cpus, progressbar=True, recompute=recompute,
                                          prefix=prefix)
    workflow = wf.ROICountersWorkflow(p)
    path = os.path.join(folder, f'{prefix}_{wf.params_hash(workflow.params)}.h5')   # as run_roi_counters_cached()
    if os.path.exists(path) and not recompute:
        try:
            results = wf.load_results(path)
            if not np.any(results['missing']):
                print(f'results loaded (same parameters) from {path}')
                results['resultsfile'] = path
                return results
        except Exception as err:
            print(f'cannot read results file {path}: {err}')
    job = WL.launch(WL.make_config(workflow.params, path), resource=resource, nbcpus=nb_cpus, time_limit=time_limit)
    status = job.wait()
    if status['state'] != 'COMPLETED':
        raise RuntimeError(f"job {status['state']}: see the log {job.files['log']}")
    results = job.results()
    results['resultsfile'] = path
    return results


# --------------------------------------------------------------------------------------
#  PLOTS: mosaics and grain imaging
# --------------------------------------------------------------------------------------
def plot_mosaics(results: Dict[str, Any], roiindices=None, nbcols: int = 3, percentiles=(1, 99.5),
                 width: float = 5.):
    """mosaic of each ROI (counter 'mosaic'): its pixels in every image arranged as the map (fast axis horizontal,
    slow axis vertical, origin at the bottom). Hover: image index, map position, intensity. Returns fig, axes"""
    import matplotlib.pyplot as plt
    if 'mosaic' not in results:
        raise ValueError("no counter 'mosaic' in the results")
    params = results['params']
    nfast, nslow = params['mapdimensions']
    roicenters = params['roicenters']
    roiindices = list(range(len(roicenters)) if roiindices is None else roiindices)
    nbcols = max(1, min(nbcols, len(roiindices)))
    nbrows = -(-len(roiindices) // nbcols)
    boxY, boxX = results['mosaic'].shape[2:]
    ratio = (nslow * boxY) / (nfast * boxX)
    fig, axes = plt.subplots(nbrows, nbcols, squeeze=False,
                             figsize=(width * nbcols, width * nbrows * min(2., max(0.3, ratio)) + 0.5))
    imageindices = np.asarray(results['imageindices'])
    for ax, iroi in zip(axes.flat, roiindices):
        mosaic = wf.arrange_mosaic2D(results['mosaic'][:, iroi], (nfast, nslow))['singleimage']
        finite = mosaic[np.isfinite(mosaic)]
        vmin, vmax = np.percentile(finite, percentiles) if finite.size else (None, None)
        aspect = 'auto' if ratio > 3 or ratio < 1 / 3 else 'equal'
        ax.imshow(mosaic, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest', aspect=aspect)
        ax.set_title(f'mosaic ROI {iroi} {list(roicenters[iroi])}', fontsize=9)
        ax.set_xlabel(f"fast axis {params.get('fastaxis')} ({boxX} pixels per point)", fontsize=8)
        ax.set_ylabel(f"slow axis {params.get('slowaxis')} ({boxY} pixels per point)", fontsize=8)

        def format_coord(x, y, mosaic=mosaic):
            col, row = int(round(x)), int(round(y))
            if not (0 <= col < mosaic.shape[1] and 0 <= row < mosaic.shape[0]):
                return ''
            ifast, islow = col // boxX, row // boxY
            k = islow * nfast + ifast
            image = f'image {int(imageindices[k])}' if k < len(imageindices) else 'no image'
            return f'{image} (fast {ifast}, slow {islow}) I {mosaic[row, col]:.0f}'
        ax.format_coord = format_coord
    for ax in axes.flat[len(roiindices):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes


def normalized_maps(results: Dict[str, Any], counter: str = 'max', percentiles=(5, 99.5)) -> np.ndarray:
    """values of a counter of each ROI (nb images, nb ROIs) scaled to [0, 1] between its percentiles (background
    of the ROI -> 0, brightest points -> 1)"""
    from LaueTools.scripts import roicountersplot as RCP
    values = RCP.counter_values(results, counter, 0)
    lo = np.nanpercentile(values, percentiles[0], axis=0)
    hi = np.nanpercentile(values, percentiles[1], axis=0)
    return np.clip((values - lo) / np.where(hi > lo, hi - lo, 1.), 0, 1)


def plot_grain_overview(results: Dict[str, Any], counter: str = 'max', threshold: float = 0.5,
                        percentiles=(5, 99.5), figsize=(15, 4.5)):
    """grain imaging overview: intensities of each ROI scaled to [0, 1] (normalized_maps()), then at each map point
    the nb of lit ROIs (scaled intensity > threshold), the sum of the scaled intensities, and the brightest ROI
    (points lit by the same spots, i.e. the same grain, have the same color). Hover: image index, value (and the
    brightest ROI). Returns fig, axes"""
    import matplotlib.pyplot as plt
    from LaueTools.scripts import roicountersplot as RCP
    norm = normalized_maps(results, counter, percentiles)
    display = RCP.MapDisplay(results)
    nblit = np.sum(norm > threshold, axis=1).astype(float)
    total = np.nansum(norm, axis=1)
    brightest = np.where(np.nanmax(norm, axis=1) > threshold, np.nanargmax(np.nan_to_num(norm, nan=-1), axis=1), np.nan)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for ax, values, title, cmap in ((axes[0], nblit, f'nb of ROIs lit (> {threshold})', 'viridis'),
                                    (axes[1], total, f'sum of the scaled {counter}', 'magma'),
                                    (axes[2], brightest, 'brightest ROI (index)', 'tab20')):
        im = display.imshow(ax, values, cmap=cmap, interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(display.xlabel, fontsize=8)
    axes[0].set_ylabel(display.ylabel, fontsize=8)
    roicenters = results['params']['roicenters']

    def format_coord(x, y):
        pos = display.position(x, y)
        if pos is None or pos[0] is None:
            return ''
        k, ifast, islow = pos
        kb = brightest[k]
        best = '-' if np.isnan(kb) else f'ROI {int(kb)} {list(roicenters[int(kb)])}'
        return (f'image {int(display.imageindices[k])} (fast {ifast}, slow {islow}) | {int(nblit[k])} lit, '
                f'sum {total[k]:.2f}, brightest {best}')
    for ax in axes:
        ax.format_coord = format_coord
    fig.tight_layout()
    return fig, axes
