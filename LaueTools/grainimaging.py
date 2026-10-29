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
- grains without indexation: groups of ROIs lit at the same map points (find_grains(), plot_grains(),
  plot_grain_segmentation(), grains_table(), brightest_rois(), GrainBrowser: map of a grain, its ROIs on the Laue pattern and in gnomonic
  coordinates), spots of a grain fitted in one image and written as a .dat/.cor file to index it (grain_peaks())

Used by notebooks/mosaicgrainimaging/mosaic_grainimaging.ipynb and notebooks/quickstart/laue_maps_quickstart.ipynb

ROI centers (X, Y) are pixel indices of the image array, image[Y, X] (first pixel at (0, 0)), as in
LaueTools.scripts.workflows.read_rois(). Peak positions of the peak search start at (1, 1): X - 1, Y - 1
"""
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
    try:
        res = RMCCD.PeakSearch(path, stackimageindex=frame, raise_too_many=True, **kwargs)
    except RMCCD.TooManyLocalMaximaError as err:
        raise TooManyPeaksError(f'image {imageindex}: {err}') from None
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
        workflowlauncher.COMPUTING_RESOURCES: 'local' (background process on this machine), 'magnifix', 'magnifix2', 'magnifix3'
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


# --------------------------------------------------------------------------------------
#  GRAINS: groups of ROIs lit at the same map points (no indexation)
# --------------------------------------------------------------------------------------
def select_rois(results: Dict[str, Any], xlim=None, ylim=None, every: int = 1) -> np.ndarray:
    """indices of a subset of the ROIs of grain imaging results (e.g. for find_grains(rois=...)): ROI centers with
    xlim[0] <= X <= xlim[1] and ylim[0] <= Y <= ylim[1] (pixel), and for a grid of ROIs one grid line out of every
    along X and Y (every=2: 4 times fewer ROIs, e.g. step 30 instead of 15 pixels: ROIs no longer overlapping)"""
    centers = np.asarray(results['params']['roicenters'])
    keep = np.ones(len(centers), bool)
    if xlim is not None:
        keep &= (centers[:, 0] >= xlim[0]) & (centers[:, 0] <= xlim[1])
    if ylim is not None:
        keep &= (centers[:, 1] >= ylim[0]) & (centers[:, 1] <= ylim[1])
    if every > 1:
        for axis in (0, 1):
            levels = np.unique(centers[:, axis])
            keep &= np.isin(centers[:, axis], levels[::every])
    return np.flatnonzero(keep)


def find_grains(results: Dict[str, Any], counter: str = 'max', method: str = 'regions', min_jaccard: float = 0.5,
                min_rois: int = 5, min_points: int = 2, nsigma: float = 10., min_level: float = 0.15,
                merge_correlation: float = 0.8, threshold: float = 0.5, max_fraction: float = 0.8,
                percentiles=(5, 99.5), max_rois: Optional[int] = None, min_area: int = 1,
                max_area: Optional[int] = None, rois=None, max_dense: int = 4000,
                progress: bool = True) -> Dict[str, Any]:
    """grains of the map from grain imaging results: groups of ROIs (Laue spots) lit at the same map points

    method 'regions' (default): a ROI map may show several grains, at different places and intensity levels
    1. points of a ROI lit: intensity above its background by nsigma x its noise (median, robust spread over the
       map) and by min_level x its range (5th to 99.5th percentile): weak grains are kept. ROIs without signal are
       left out
    2. regions: connected groups (8 neighbours) of at least min_points lit points of each ROI map: a ROI lighting
       two places of the map gives two regions, which may belong to two grains
    3. similarity of two regions: Jaccard index of their points (common points / points of either); hierarchical
       clustering (average linkage) cut at min_jaccard (computed only between overlapping regions: sparse, fast);
       grains: groups of regions of at least min_rois different ROIs
    4. merge of two grains with nested regions (90 % of the points of one in the other) and intensity maps
       correlated (Pearson >= merge_correlation on their points): one grain split by the size of its weak and strong
       spots (uncorrelated nested grains: different grains, e.g. in depth)
    5. map of a grain: mean scaled intensity (normalized_maps()) of its ROIs, each ROI only in its regions of the
       grain; grain lit at a point: at least half of its ROIs have a region there; segmentation: at each lit point
       the grain of highest value
    6. grains lit at fewer than min_area points or more than max_area points of the map (area: nb of points, each
       point = 1 image) left out. With max_area, regions of more than max_area / min_jaccard points are left out
       before step 3 (they cannot be similar enough to the regions of such a grain): faster with many ROIs
    Size of the problem (memory, time; large sets of ROIs, e.g. a grid of 20000 ROIs): rois: indices of the ROIs used
    (None: all); max_rois: only the max_rois ROIs of strongest signal (peak above background / noise); min_points
    or nsigma larger: fewer regions (small regions of noise). A group of more than max_dense overlapping regions is
    split by increasing similarity before its clustering (approximation; memory 4 bytes x max_dense^2).
    progress: steps and progress bars
    method 'rois': one level per ROI (scaled intensity > threshold), Jaccard of the lit points of the ROIs, a ROI
       belongs to one grain at most (ROIs lit at more than max_fraction of the map left out; max_rois: largest nb
       of ROIs clustered (default 15000), memory ~ 12 bytes x nb ROIs^2); grain lit where its map > threshold

    Returns dict: 'grains' (list of dicts sorted by decreasing nb of ROIs: 'rois' indices of the ROIs,
    'roicenters', 'nbrois', 'regions' {ROI: indices of its points (map order) in the grain}, 'area' nb of points
    where the grain is lit, 'centroid' (fast, slow) index, 'best_image' image index where it is brightest),
    'score' (nb images, nb grains), 'lit' (nb images, nb grains) bool, 'label' (grain index at each point, -1:
    none), 'nblit' (nb of grains lit at each point), 'imageindices', 'settings'
    """
    if method not in ('regions', 'rois'):
        raise ValueError("method must be 'regions' or 'rois'")
    norm = normalized_maps(results, counter, percentiles)   # (nb images, nb ROIs) in [0, 1]
    nbimages = norm.shape[0]
    roicenters = np.asarray(results['params']['roicenters'])
    settings = dict(counter=counter, method=method, min_jaccard=min_jaccard, min_rois=min_rois,
                    min_points=min_points, percentiles=percentiles, min_area=min_area, max_area=max_area,
                    max_rois=max_rois,
                    rois=None if rois is None else len(rois))
    if method == 'regions':
        settings.update(nsigma=nsigma, min_level=min_level, merge_correlation=merge_correlation)
        members, nbregions, nblitrois, nsplit = _grains_from_regions(
            results, counter, min_jaccard, min_rois, min_points, nsigma, min_level, percentiles, rois=rois,
            max_rois=max_rois, max_dense=max_dense, progress=progress,
            max_region=None if max_area is None else int(max_area / min_jaccard))
        t0 = time.time()
        score, lit = _grain_maps(norm, members)
        members, score, lit, nmerged = _merge_grains(members, score, lit, norm, merge_correlation)
        if progress:
            print(f'maps of {len(members)} grains, {nmerged} merges ({time.time() - t0:.1f} s)')
        origin = (f"{nbregions} regions of {nblitrois} ROIs with signal ({norm.shape[1]} in all), "
                  f"{nmerged} merges" + (f", {nsplit} large groups split" if nsplit else ''))
    else:
        settings.update(threshold=threshold, max_fraction=max_fraction)
        if rois is not None:   # only these ROIs: the others never lit
            mask = np.zeros(norm.shape[1], bool)
            mask[np.asarray(rois, dtype=int)] = True
            norm = np.where(mask, norm, 0.)
        members, ncandidates = _grains_from_rois(norm, min_jaccard, min_rois, min_points, threshold, max_fraction,
                                                 15000 if max_rois is None else max_rois)
        score = np.stack([norm[:, list(m)].mean(axis=1) for m in members], axis=1) if members \
            else np.zeros((nbimages, 0))
        lit = score > threshold
        origin = f"{ncandidates} ROIs ({norm.shape[1]} in all)"
    if (min_area > 1 or max_area is not None) and members:   # small (and large) grains left out
        area = lit.sum(axis=0)
        keep = (area >= min_area) & (area <= (max_area if max_area is not None else np.inf))
        origin += (f", {int(np.sum(~keep))} grains of fewer than {min_area}"
                   + (f" or more than {max_area}" if max_area is not None else '') + " points left out")
        members =[m for m, k in zip(members, keep) if k]
        score, lit = score[:, keep], lit[:, keep]
    nblit = lit.sum(axis=1)
    label = np.where(nblit > 0, np.argmax(np.where(lit, score, -1.), axis=1), -1) if members \
        else -np.ones(nbimages, int)
    imageindices = np.asarray(results['imageindices'])
    dimfast = results['params']['mapdimensions'][0]
    grains = []
    for k, m in enumerate(members):
        rois = np.array(sorted(m), dtype=int)
        points = np.flatnonzero(lit[:, k])
        grains.append({'rois': rois, 'roicenters': roicenters[rois].tolist(), 'nbrois': len(rois),
                       'regions': {int(roi): np.asarray(m[roi], dtype=int) for roi in rois},
                       'area': len(points),
                       'centroid': ((float(np.mean(points % dimfast)), float(np.mean(points // dimfast)))
                                    if len(points) else (np.nan, np.nan)),
                       'best_image': int(imageindices[np.argmax(score[:, k])])})
    print(f"{len(grains)} grains (method '{method}': >= {min_rois} ROIs lit together, Jaccard >= {min_jaccard}) from "
          f"{origin}; {np.sum(label >= 0)} / {nbimages} points with a grain, {np.sum(nblit > 1)} with several")
    return {'grains': grains, 'score': score, 'lit': lit, 'label': label, 'nblit': nblit,
            'imageindices': imageindices, 'settings': settings}


def _grains_from_rois(norm: np.ndarray, min_jaccard, min_rois, min_points, threshold, max_fraction, max_rois):
    """method 'rois' of find_grains(): ([{ROI: indices of all the points}, ...] sorted by decreasing nb of ROIs,
    nb of ROIs clustered)"""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    nbimages = len(norm)
    lit = norm > threshold
    nlit = lit.sum(axis=0)
    candidates = np.flatnonzero((nlit >= min_points) & (nlit <= max_fraction * nbimages))
    if len(candidates) > max_rois:   # memory: the ROIs lit at the most points
        candidates = np.sort(candidates[np.argsort(-nlit[candidates], kind='stable')[:max_rois]])
        GT.printyellow(f'{max_rois} ROIs clustered (max_rois), the most often lit')
    members = []
    if len(candidates) >= max(2, min_rois):
        L = lit[:, candidates].astype(np.float32)
        inter = L.T @ L                                            # common lit points of each pair of ROIs
        n = nlit[candidates].astype(np.float32)
        dist = 1. - inter / np.maximum(n[:, None] + n[None, :] - inter, 1.)   # 1 - Jaccard
        del inter, L
        np.fill_diagonal(dist, 0.)
        groups = fcluster(linkage(squareform(dist, checks=False), 'average'), 1. - min_jaccard, 'distance')
        del dist
        ids, counts = np.unique(groups, return_counts=True)
        allpoints = np.arange(nbimages)
        for gid in ids[np.argsort(-counts, kind='stable')]:
            rois = candidates[groups == gid]
            if len(rois) < min_rois:
                break
            members.append({int(roi): allpoints for roi in rois})
    return members, len(candidates)


def _split_large_group(graph, nodes, threshold: float, max_dense: int, step: float = 0.05):
    """pieces of at most max_dense nodes of a group of connected regions: groups connected by pairs of similarity >=
    threshold, threshold raised by step until the pieces are small enough (approximation of the clustering of the
    whole group). Returns [(nodes, too_large)] (too_large: still larger than max_dense with similarities ~1)"""
    from scipy.sparse.csgraph import connected_components
    pieces, stack = [], [(np.asarray(nodes), threshold)]
    while stack:
        part, thr = stack.pop()
        if len(part) <= max_dense:
            pieces.append((part, False))
            continue
        if thr >= 0.99:   # nearly identical regions: one cluster
            pieces.append((part, True))
            continue
        sub = graph[part][:, part]
        sub.data[sub.data < thr + step] = 0.
        sub.eliminate_zeros()
        ncomp, comp = connected_components(sub, directed=False)
        order = np.argsort(comp, kind='stable')
        bounds = np.flatnonzero(np.diff(comp[order])) + 1
        stack += [(part[g], thr + step) for g in np.split(order, bounds)]
    return pieces


def _grains_from_regions(results, counter, min_jaccard, min_rois, min_points, nsigma, min_level, percentiles,
                         rois=None, max_rois=None, max_dense=4000, progress=True, max_region=None):
    """steps 1-3 of method 'regions' of find_grains(): ([{ROI: indices of the points of its regions in the grain},
    ...] sorted by decreasing nb of ROIs, nb of regions, nb of ROIs with signal, nb of groups split)"""
    from scipy import ndimage, sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from LaueTools.scripts import roicountersplot as RCP
    tqdm = _progressbar(progress)
    t0 = time.time()
    values = RCP.counter_values(results, counter, 0)   # (nb images, nb ROIs)
    nbimages, nbrois = values.shape
    selected = np.arange(nbrois) if rois is None else np.unique(np.asarray(rois, dtype=int))
    values = values[:, selected]
    dimfast, dimslow = results['params']['mapdimensions']
    med = np.nanmedian(values, axis=0)
    noise = np.maximum(1.4826 * np.nanmedian(np.abs(values - med), axis=0), 1e-9)   # robust std (MAD)
    top = np.nanpercentile(values, percentiles[1], axis=0)
    level = np.maximum(med + nsigma * noise, med + min_level * (top - med))
    lit = np.nan_to_num(values, nan=-np.inf) > level
    withsignal = np.flatnonzero(lit.sum(axis=0) >= min_points)
    if max_rois is not None and len(withsignal) > max_rois:   # strongest signal: peak above background / noise
        snr = (top - med)[withsignal] / noise[withsignal]
        withsignal = np.sort(withsignal[np.argsort(-snr, kind='stable')[:max_rois]])
        if progress:
            print(f'{max_rois} ROIs of strongest signal kept (max_rois)')
    if progress:
        print(f'1/4 noise levels of {len(selected)} ROIs: {len(withsignal)} with signal ({time.time() - t0:.1f} s)')
    region_roi, region_points = [], []
    padded = dimfast * dimslow
    for j in tqdm(withsignal, desc='2/4 regions of the ROI maps', unit='ROI'):
        mask = np.zeros(padded, bool)
        mask[:nbimages] = lit[:, j]
        labels, nbl = ndimage.label(mask.reshape(dimslow, dimfast), structure=np.ones((3, 3)))
        labels = labels.ravel()[:nbimages]
        sizes = np.bincount(labels, minlength=nbl + 1)
        order = np.argsort(labels, kind='stable')
        bounds = np.cumsum(sizes)
        large = np.inf if max_region is None else max_region
        for k in np.flatnonzero((sizes[1:] >= min_points) & (sizes[1:] <= large)) + 1:
            region_roi.append(int(selected[j]))
            region_points.append(order[bounds[k - 1]:bounds[k]])
    del lit, values
    nregions = len(region_roi)
    if nregions == 0:
        return [], 0, 0, 0
    t1 = time.time()
    region_roi = np.asarray(region_roi)
    sizes = np.array([len(p) for p in region_points], dtype=np.float32)
    U = sparse.csr_matrix((np.ones(int(sizes.sum()), np.float32),
                           (np.repeat(np.arange(nregions), sizes.astype(int)), np.concatenate(region_points))),
                          shape=(nregions, nbimages))
    inter = (U @ U.T).tocoo()   # common points of the overlapping regions only
    jac = inter.data / (sizes[inter.row] + sizes[inter.col] - inter.data)
    # pairs with Jaccard < min_jaccard / 2 cannot be in the same cluster (average linkage cut at min_jaccard):
    # clustering of each group of connected regions separately
    keep = (jac >= min_jaccard / 2) & (inter.row != inter.col)
    graph = sparse.csr_matrix((jac[keep], (inter.row[keep], inter.col[keep])), shape=(nregions, nregions))
    del inter, jac, keep, U
    _, component = connected_components(graph, directed=False)
    order = np.argsort(component, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(component[order])) + 1)
    groups = [g for g in groups if len(np.unique(region_roi[g])) >= min_rois]
    if progress:
        print(f'3/4 similarity of {nregions} regions: {len(groups)} groups of overlapping regions, largest '
              f'{max((len(g) for g in groups), default=0)} regions ({time.time() - t1:.1f} s)')
    members, nsplit = [], 0
    for idx in tqdm(sorted(groups, key=len, reverse=True), desc='4/4 clustering of the groups', unit='group'):
        pieces = [(idx, False)]
        if len(idx) > max_dense:   # dense similarity matrix too large: pieces of more similar regions
            pieces = _split_large_group(graph, idx, min_jaccard / 2, max_dense)
            nsplit += 1
        for part, too_large in pieces:
            if len(np.unique(region_roi[part])) < min_rois:
                continue
            if too_large or len(part) == 1:
                clusters = np.ones(len(part), int)
            else:
                sim = graph[part][:, part].toarray()
                np.fill_diagonal(sim, 1.)
                clusters = fcluster(linkage(squareform(1. - sim, checks=False), 'average'), 1. - min_jaccard,
                                    'distance')
            for cl in np.unique(clusters):
                regs = part[clusters == cl]
                if len(np.unique(region_roi[regs])) < min_rois:
                    continue
                m = {}
                for reg in regs:
                    roi = int(region_roi[reg])
                    m[roi] = np.union1d(m.get(roi, []), region_points[reg]).astype(int)
                members.append(m)
    if nsplit and progress:
        GT.printyellow(f'{nsplit} group(s) of more than {max_dense} overlapping regions split by increasing '
                       f'similarity before the clustering (approximation; fewer regions: increase min_points or '
                       f'nsigma, decrease max_rois)')
    members.sort(key=lambda m: -len(m))
    return members, nregions, len(np.unique(region_roi)), nsplit


def _progressbar(progress: bool):
    """tqdm (notebook or text progress bar) if progress, else an iterable without bar"""
    if progress:
        try:
            from tqdm.auto import tqdm
            return tqdm
        except ImportError:
            pass
    return lambda iterable, **kwargs: iterable


def _grain_maps(norm: np.ndarray, members):
    """(score, lit) of grains {ROI: points}: mean scaled intensity of the ROIs, each only at its points; lit where
    at least half of the ROIs have a point"""
    nbimages = len(norm)
    score = np.zeros((nbimages, len(members)))
    lit = np.zeros((nbimages, len(members)), bool)
    for k, m in enumerate(members):
        count = np.zeros(nbimages)
        for roi, points in m.items():
            score[points, k] += norm[points, roi]
            count[points] += 1
        score[:, k] /= len(m)
        lit[:, k] = count >= len(m) / 2.
    return score, lit


def _merge_grains(members, score, lit, norm, merge_correlation):
    """merge grains with nested regions (90 %) and correlated intensity maps (one grain split by the size of the
    regions of its weak and strong spots). Returns (members, score, lit, nb of merges)"""
    if len(members) < 2 or merge_correlation is None:
        return members, score, lit, 0
    support = np.stack([np.isin(np.arange(len(norm)), np.concatenate(list(m.values()))) for m in members], axis=1)
    s = support.astype(np.float32)
    inter = s.T @ s
    n = s.sum(axis=0)
    nested = inter / np.maximum(np.minimum(n[:, None], n[None, :]), 1.) >= 0.9
    parent = list(range(len(members)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    nmerged = 0
    for i, j in zip(*np.nonzero(np.triu(nested, 1))):
        common = support[:, i] | support[:, j]
        if common.sum() < 4 or np.std(score[common, i]) == 0 or np.std(score[common, j]) == 0:
            continue
        if np.corrcoef(score[common, i], score[common, j])[0, 1] >= merge_correlation and root(i) != root(j):
            parent[root(j)] = root(i)
            nmerged += 1
    if not nmerged:
        return members, score, lit, 0
    groups = {}
    for k in range(len(members)):
        groups.setdefault(root(k), []).append(k)
    merged = []
    for ks in groups.values():
        m = {}
        for k in ks:
            for roi, points in members[k].items():
                m[roi] = np.union1d(m.get(roi, []), points).astype(int)
        merged.append(m)
    merged.sort(key=lambda m: -len(m))
    score, lit = _grain_maps(norm, merged)
    return merged, score, lit, nmerged


def grain_colors(n: int, seed: int = 1):
    """n distinct colors (RGB in [0, 1]): hues spread over the circle with 3 levels of lightness, shuffled so that
    grains of close indices have different colors"""
    import colorsys
    rng = np.random.default_rng(seed)
    golden = 0.618033988749895
    colors = [colorsys.hls_to_rgb((k * golden) % 1., (0.42, 0.62, 0.30)[k % 3], (0.85, 0.75, 0.9)[k % 3])
              for k in range(n)]
    return np.array(colors)[rng.permutation(n)] if n else np.zeros((0, 3))


def plot_grain_segmentation(grains: Dict[str, Any], results: Dict[str, Any], boundaries: bool = True,
                            figsize=(8, 8)):
    """map of the grain of highest value at each point (find_grains()), one distinct color per grain, boundaries
    between grains (white lines), black: no grain. Hover: grain, its value, other grains lit at this point, image
    index. Returns fig, ax"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from LaueTools.scripts import roicountersplot as RCP
    display = RCP.MapDisplay(results)
    label, score, lit = grains['label'], grains['score'], grains['lit']
    ngrains = max(len(grains['grains']), 1)
    cmap = ListedColormap(grain_colors(ngrains))
    cmap.set_bad('black')
    fig, ax = plt.subplots(figsize=figsize)
    shown = display.data(np.where(label >= 0, label, np.nan).astype(float))
    ax.imshow(shown, origin='lower', cmap=cmap, vmin=-0.5, vmax=ngrains - 0.5, interpolation='nearest')
    if boundaries:   # segments between neighbouring points of different grains
        lab = np.nan_to_num(shown, nan=-1)
        ny, nx = lab.shape
        segments = [[(j + 0.5, i - 0.5), (j + 0.5, i + 0.5)] for i in range(ny) for j in range(nx - 1)
                    if lab[i, j] != lab[i, j + 1]]
        segments += [[(j - 0.5, i + 0.5), (j + 0.5, i + 0.5)] for i in range(ny - 1) for j in range(nx)
                     if lab[i, j] != lab[i + 1, j]]
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segments, colors='white', linewidths=0.6))
    ax.set_xlabel(display.xlabel)
    ax.set_ylabel(display.ylabel)
    nwith = int(np.sum(label >= 0))
    ax.set_title(f"grain of highest value: {len(grains['grains'])} grains, {nwith} / {len(label)} points with a "
                 f"grain (black: none)", fontsize=10)

    def format_coord(x, y):
        pos = display.position(x, y)
        if pos is None or pos[0] is None:
            return ''
        k, ifast, islow = pos
        g = label[k]
        others = [str(o) for o in np.flatnonzero(lit[k]) if o != g]
        head = f'grain {g} (value {score[k, g]:.2f})' if g >= 0 else 'no grain'
        return (f"image {int(display.imageindices[k])} (fast {ifast}, slow {islow}) | {head} | also lit: "
                f"{', '.join(others) or '-'}")
    ax.format_coord = format_coord
    fig.tight_layout()
    return fig, ax


def grains_table(grains: Dict[str, Any]):
    """table (pandas DataFrame) of the grains of find_grains(): nb of ROIs, area (nb of points), centroid (fast,
    slow indices), image where the grain is brightest"""
    import pandas as pd
    rows = [{'grain': k, 'nb ROIs': g['nbrois'], 'area (points)': g['area'],
             'centroid fast': round(g['centroid'][0], 1), 'centroid slow': round(g['centroid'][1], 1),
             'brightest image': g['best_image']} for k, g in enumerate(grains['grains'])]
    return pd.DataFrame(rows).set_index('grain') if rows else pd.DataFrame(rows)


def plot_grains(grains: Dict[str, Any], results: Dict[str, Any], nbgrains: int = 6, figsize=(15, 9)):
    """grains of find_grains(): top: segmentation (grain of highest value at each point), nb of grains lit, ROIs
    (Laue spots) of the grains on the detector; bottom: maps of the first nbgrains grains. Hover: image index,
    grain(s). Returns fig"""
    import matplotlib.pyplot as plt
    from LaueTools.scripts import roicountersplot as RCP
    display = RCP.MapDisplay(results)
    score, label, nblit = grains['score'], grains['label'], grains['nblit']
    nbgrains = min(nbgrains, len(grains['grains']))
    ncols = max(nbgrains, 4)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols)
    seg, lit = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    det = fig.add_subplot(gs[0, 2:])
    maps = [seg, lit] + [fig.add_subplot(gs[1, k]) for k in range(nbgrains)]
    cmap = plt.get_cmap('tab20')
    display.imshow(seg, np.where(label >= 0, label % 20, np.nan).astype(float), cmap=cmap, vmin=-0.5, vmax=19.5,
                   interpolation='nearest')
    seg.set_title('grain of highest value', fontsize=9)
    im = display.imshow(lit, nblit.astype(float), cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=lit, fraction=0.08, pad=0.02)
    lit.set_title('nb of grains lit', fontsize=9)
    for k in range(nbgrains):
        display.imshow(maps[2 + k], score[:, k], cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        maps[2 + k].set_title(f"grain {k} ({grains['grains'][k]['nbrois']} ROIs)", fontsize=9, color=cmap(k % 20))
    for ax in maps:
        ax.set_xlabel(display.xlabel, fontsize=7)
        ax.set_ylabel(display.ylabel, fontsize=7)
        ax.tick_params(labelsize=7)

    def format_coord(x, y):
        pos = display.position(x, y)
        if pos is None or pos[0] is None:
            return ''
        k, ifast, islow = pos
        lit_here = [f'{g} ({score[k, g]:.2f})' for g in np.flatnonzero(grains['lit'][k])]
        return (f"image {int(display.imageindices[k])} (fast {ifast}, slow {islow}) | grain "
                f"{label[k] if label[k] >= 0 else '-'} | lit: {', '.join(lit_here) or '-'}")
    for ax in maps:
        ax.format_coord = format_coord
    det.scatter(*np.asarray(results['params']['roicenters']).T, s=1, c='0.88')
    for k in range(min(len(grains['grains']), 20)):
        det.scatter(*np.asarray(grains['grains'][k]['roicenters']).T, s=14, color=cmap(k % 20), label=f'grain {k}')
    det.set_aspect('equal')
    det.invert_yaxis()
    det.set_xlabel('X (pixel)', fontsize=8)
    det.set_ylabel('Y (pixel)', fontsize=8)
    det.set_title('ROIs (Laue spots) of the grains on the detector', fontsize=9)
    det.legend(fontsize=7, markerscale=1.5, ncol=2, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def grain_peaks(results: Dict[str, Any], grains: Dict[str, Any], grain: int, imageindex: Optional[int] = None,
                detfile: Optional[Union[str, Path]] = None, outputfolder: Optional[Union[str, Path]] = None,
                fitpixeldev: Optional[float] = None, verbose: bool = True):
    """Laue spots of a grain of find_grains() in one image (default: the image where it is brightest): 2D gaussian
    fit of the brightest pixel of each of its ROIs (rejection as in the peak search), written as a peak list
    <outputfolder>/<prefix><imageindex>_grain<grain>.dat (with detfile: .cor file, 2theta and chi, to be indexed,
    e.g. with IB.index_refine()). outputfolder: default the folder of the grain imaging results.
    Returns (peak list array (X, Y, I, ... as readmccd.PeakSearch(), most intense first), path of the file)"""
    from LaueTools import readmccd as RMCCD
    params = results['params']
    g = grains['grains'][grain]
    imageindex = g['best_image'] if imageindex is None else int(imageindex)
    path, frame, imageindex = image_path(params, imageindex)
    box = int(max(params['boxsize_X'], params['boxsize_Y']))
    res = RMCCD.fitoneimage_manypeaks(path, np.asarray(g['roicenters'], dtype=float), box, stackimageindex=frame,
                                      CCDLabel=params['CCDLabel'], position_start='max',
                                      FitPixelDev=fitpixeldev if fitpixeldev is not None else 1.5 * box,
                                      PeakSizeRange=(0.4, box), NumberMaxofFits=len(g['roicenters']) + 1,
                                      purgeDuplicates=True)
    peaks = None if res is None else res[0]
    if peaks is None or len(peaks) == 0:
        GT.printyellow(f'grain {grain}: no spot fitted in image {imageindex}')
        return None, None
    folder = Path(outputfolder) if outputfolder is not None else Path(results.get('resultsfile', '.')).parent
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{params['prefix']}{imageindex:0{params.get('nbdigits', 4)}d}_grain{grain}"
    comments = f"grain {grain} of the grain imaging {results.get('resultsfile', '')}: {g['nbrois']} ROIs"
    if detfile is not None:
        from LaueTools import LaueGeometry as LaueGeo
        from LaueTools import IOLaueTools as IOLT
        written, _ = LaueGeo.peaklist_to_corfile(peaks, stem, str(folder), IOLT.readCalib_det_file(str(detfile)),
                                                 initialfilename=path, comments=comments)
    else:
        written = RMCCD.writepeaklist(peaks, stem, outputfolder=str(folder), initialfilename=path, comments=comments)
    if verbose:
        print(f'grain {grain}: {len(peaks)} spots fitted in image {imageindex} (of {g["nbrois"]} ROIs) -> {written}')
    return peaks, written


def brightest_rois(results: Dict[str, Any], grains: Optional[Dict[str, Any]] = None, nb: int = 20,
                   counter: str = 'max'):
    """table (pandas DataFrame) of the nb ROIs of highest intensity (counter) over the map, brightest first: ROI
    index, center X, Y, highest value, image and map point (fast, slow) where it is, grains of find_grains() it
    belongs to ('-': none; method 'regions': several if the ROI lights several places) and grains lit at this
    point"""
    import pandas as pd
    from LaueTools.scripts import roicountersplot as RCP
    values = RCP.counter_values(results, counter, 0)   # (nb images, nb ROIs)
    highest = np.nanmax(np.where(np.isfinite(values), values, -np.inf), axis=0)
    roicenters = np.asarray(results['params']['roicenters'])
    imageindices = np.asarray(results['imageindices'])
    dimfast = results['params']['mapdimensions'][0]
    grains_of = {}
    if grains is not None:
        for g, grain in enumerate(grains['grains']):
            for i in grain['rois']:
                grains_of.setdefault(int(i), []).append(g)
    rows = []
    for roi in np.argsort(-highest, kind='stable')[:nb]:
        k = int(np.nanargmax(np.where(np.isfinite(values[:, roi]), values[:, roi], -np.inf)))
        row = {'ROI': int(roi), 'X': int(roicenters[roi][0]), 'Y': int(roicenters[roi][1]),
               f'highest {counter}': float(highest[roi]), 'image': int(imageindices[k]), 'fast': k % dimfast,
               'slow': k // dimfast}
        if grains is not None:
            row['grains of the ROI'] = ', '.join(map(str, grains_of.get(int(roi), []))) or '-'
            lit = np.flatnonzero(grains['lit'][k])
            row['grains lit at this point'] = ', '.join(map(str, lit)) or '-'
        rows.append(row)
    return pd.DataFrame(rows).set_index('ROI')


def gnomonic_coordinates(roicenters, detfile: Union[str, Path]) -> np.ndarray:
    """gnomonic coordinates (nb ROIs, 2) of ROI centers (X, Y pixel indices of the image array) with the detector
    calibration of a .det file (spots of a grain lie on lines: zone axes)"""
    from LaueTools import IOLaueTools as IOLT
    from LaueTools import LaueGeometry as LaueGeo
    from LaueTools import indexingImageMatching as IIM
    calib = IOLT.readCalib_det_file(str(detfile))
    xy = np.asarray(roicenters, dtype=float) + 1.   # peak positions start at (1, 1)
    tth, chi = LaueGeo.calc_uflab(xy[:, 0], xy[:, 1], calib['CCDCalibParameters'], pixelsize=calib['pixelsize'],
                                  kf_direction=calib.get('kf_direction', 'Z>0'))
    return np.array(IIM.ComputeGnomon_2([np.asarray(tth), np.asarray(chi)])).T


class GrainBrowser:
    """browser of the grains of find_grains() (interactive backend %matplotlib widget and ipywidgets):
    - left: map of the current grain (mean scaled intensity of its ROIs); hover: image index, grains lit;
      click: Laue pattern of this map point on the right (white +: point shown)
    - right: Laue pattern (default: image where the grain is brightest) with the ROIs of the grain (boxes of the
      ROI size) and the centers of the other ROIs (grey dots)
    - grain: slider and previous / next buttons; contrast of the Laue pattern; checkbox: Laue pattern (else the
      ROIs only, faster)
    - with detfile (calibration): ROIs in gnomonic coordinates (spots of a grain on lines: zone axes)
    - spots of the peak search of the image shown (its .cor file in corfolder, default: corfiles folder of the map
      in PROCESSED_DATA) inside the ROI boxes of the grain: green circles, and their indices in the .cor file (from
      0, as in the LaueTools GUI) listed to be copied (text box); spot_margin: pixels added around the ROI boxes for
      this search. .cor file of an image: in corfolder if given, else in the folder corfiles of the map in
      PROCESSED_DATA, else in a subfolder of the map folder in PROCESSED_DATA or RAW_DATA (e.g. img_CORfiles) with
      this file (the most recent)
    - info: nb of ROIs, area, centroid, brightest image, ROIs of the grain (index and center X, Y)
    Methods without the widgets: show_grain(k), show_image(imageindex), grain_spots()"""

    def __init__(self, grains: Dict[str, Any], results: Dict[str, Any], grain: int = 0, show_image: bool = True,
                 detfile: Optional[Union[str, Path]] = None, corfolder: Optional[Union[str, Path]] = None,
                 spot_margin: float = 0., figsize=None):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        from LaueTools.scripts import roicountersplot as RCP
        self._Rectangle, self._PatchCollection = Rectangle, PatchCollection
        if not grains['grains']:
            raise ValueError('no grain: change the settings of find_grains()')
        self.grains, self.results, self.params = grains, results, results['params']
        self.display = RCP.MapDisplay(results)
        self.roicenters = np.asarray(self.params['roicenters'])
        self.box = (int(self.params['boxsize_X']), int(self.params['boxsize_Y']))
        self.grain, self.imageindex, self.boxes = None, None, None
        self.with_image = show_image
        self.corfolder = Path(corfolder) if corfolder is not None else None   # None: searched (_corfile_path())
        self._peaklists = {}   # image index: (path of the .cor file, peak list) read
        self.spot_margin = float(spot_margin)
        try:
            import ipywidgets as widgets
            from IPython.display import display
            interactive = 'inline' not in plt.get_backend().lower()
        except ImportError:
            interactive = False
        self.gnomonic = gnomonic_coordinates(self.roicenters, detfile) if detfile is not None else None
        ncols = 2 if self.gnomonic is None else 3
        with plt.ioff() if interactive else _nullcontext():
            self.fig, axes = plt.subplots(1, ncols, figsize=figsize or ((10.5, 5.2) if ncols == 2 else (13.5, 5.2)),
                                          gridspec_kw={'width_ratios': [1, 1.6, 1.3][:ncols]})
        if interactive:   # widgets only with an interactive figure (ipympl canvas is a widget)
            interactive = isinstance(self.fig.canvas, widgets.DOMWidget)
        self.axmap, self.axdet = axes[0], axes[1]
        self.axgno, self.gnodots = (axes[2], None) if self.gnomonic is not None else (None, None)
        if self.axgno is not None:
            self.axgno.scatter(*self.gnomonic.T, s=1, c='tab:blue', alpha=0.25)
            self.axgno.set_aspect('equal')
            self.axgno.set_xlabel('gnomonic X', fontsize=8)
            self.axgno.set_ylabel('gnomonic Y', fontsize=8)
            self.axgno.set_title(f'gnomonic projection ({Path(detfile).name})', fontsize=10)
        score = grains['score']
        self.immap = self.axmap.imshow(self.display.data(score[:, 0]), origin='lower', cmap='magma', vmin=0, vmax=1,
                                       interpolation='nearest')
        self.fig.colorbar(self.immap, ax=self.axmap, fraction=0.05, pad=0.02, label='mean scaled intensity')
        self.axmap.set_xlabel(self.display.xlabel, fontsize=8)
        self.axmap.set_ylabel(self.display.ylabel, fontsize=8)
        self.mark, = self.axmap.plot([], [], '+', color='w', ms=12, mew=2)
        self.axmap.format_coord = self._format_coord
        # detector: Laue pattern (if shown) under the ROIs
        nx, ny = self._detector_size()
        self.imdet = self.axdet.imshow(np.zeros((ny, nx)), cmap='gray_r', interpolation='nearest',
                                       extent=(-0.5, nx - 0.5, ny - 0.5, -0.5))
        self.axdet.scatter(*self.roicenters.T, s=0.5, c='tab:blue', alpha=0.12)   # all the ROI centers
        self.corspots = self.axdet.scatter([], [], s=70, facecolor='none', edgecolor='lime', lw=1.3)
        self.axdet.set_xlim(-0.5, nx - 0.5)
        self.axdet.set_ylim(ny - 0.5, -0.5)
        self.axdet.set_xlabel('X (pixel)', fontsize=8)
        self.axdet.set_ylabel('Y (pixel)', fontsize=8)
        self.fig.tight_layout()
        self.interactive = interactive
        if interactive:
            ngrains = len(grains['grains'])
            self.w_grain = widgets.IntSlider(value=grain, min=0, max=ngrains - 1, description='grain',
                                             continuous_update=False, layout=widgets.Layout(width='45%'))
            self.w_prev = widgets.Button(description='◀', layout=widgets.Layout(width='40px'))
            self.w_next = widgets.Button(description='▶', layout=widgets.Layout(width='40px'))
            self.w_image = widgets.Checkbox(value=show_image, description='Laue pattern', indent=False)
            self.w_contrast = widgets.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.001,
                                                       description='contrast', readout_format='.0f',
                                                       continuous_update=False, layout=widgets.Layout(width='45%'))
            self.w_info = widgets.HTML()
            self.w_spots = widgets.Textarea(value='', description='.cor spots', placeholder='spot indices',
                                            layout=widgets.Layout(width='98%', height='60px'))
            self.w_grain.observe(lambda change: self.show_grain(change['new']), names='value')
            self.w_prev.on_click(lambda _: setattr(self.w_grain, 'value', max(self.w_grain.value - 1, 0)))
            self.w_next.on_click(lambda _: setattr(self.w_grain, 'value', min(self.w_grain.value + 1, ngrains - 1)))
            self.w_image.observe(self._on_image_checkbox, names='value')
            self.w_contrast.observe(lambda change: self.imdet.set_clim(*change['new']) or self.fig.canvas.draw_idle(),
                                    names='value')
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.fig.canvas.header_visible = False
            display(widgets.VBox([widgets.HBox([self.w_prev, self.w_grain, self.w_next, self.w_image]),
                                  self.fig.canvas, self.w_contrast, self.w_spots, self.w_info]))
        self.show_grain(grain)

    def _detector_size(self):
        """(nb pixels X, Y) of the detector (from an image, else from the ROI centers)"""
        try:
            from LaueTools import dict_LaueTools as DictLT
            ny, nx = DictLT.dict_CCD[self.params['CCDLabel']][0][:2]
            return int(nx), int(ny)
        except (KeyError, IndexError, TypeError, ValueError):
            return int(self.roicenters[:, 0].max()) + 1, int(self.roicenters[:, 1].max()) + 1

    def _format_coord(self, x, y):
        pos = self.display.position(x, y)
        if pos is None or pos[0] is None:
            return ''
        k, ifast, islow = pos
        score = self.grains['score']
        lit = [f'{g}' for g in np.flatnonzero(self.grains['lit'][k])]
        return (f'image {int(self.display.imageindices[k])} (fast {ifast}, slow {islow}) | grain {self.grain}: '
                f'{score[k, self.grain]:.2f} | grains lit: {", ".join(lit) or "-"} (click: its Laue pattern)')

    def _on_click(self, event):
        if event.inaxes is self.axmap and event.button == 1 and self.fig.canvas.toolbar.mode == '':
            imageindex = self.display.imageindex(event.xdata, event.ydata)
            if imageindex is not None:
                self.show_image(imageindex)

    def _on_image_checkbox(self, change):
        self.with_image = change['new']
        self.show_image(self.imageindex)

    def show_grain(self, k: int):
        """map and ROIs of grain k, Laue pattern of the image where it is brightest"""
        self.grain = int(k)
        g = self.grains['grains'][self.grain]
        self.immap.set_data(self.display.data(self.grains['score'][:, self.grain]))
        self.axmap.set_title(f"grain {self.grain}: {g['nbrois']} ROIs, {g['area']} points", fontsize=10)
        if self.boxes is not None:
            self.boxes.remove()
        bx, by = self.box
        self.boxes = self._PatchCollection([self._Rectangle((x - bx - 0.5, y - by - 0.5), 2 * bx + 1, 2 * by + 1)
                                            for x, y in g['roicenters']], facecolor='none', edgecolor='red', lw=1.2)
        self.axdet.add_collection(self.boxes)
        if self.axgno is not None:
            if self.gnodots is not None:
                self.gnodots.remove()
            self.gnodots = self.axgno.scatter(*self.gnomonic[g['rois']].T, s=30, facecolor='none', edgecolor='red',
                                              lw=1.5)
        self.show_image(g['best_image'])

    def show_image(self, imageindex: int):
        """Laue pattern of an image of the map (if shown) with the ROIs of the current grain, info"""
        self.imageindex = int(imageindex)
        g = self.grains['grains'][self.grain]
        if self.with_image:
            image = np.asarray(read_image(self.params, self.imageindex), dtype=float)
            self.imdet.set_data(image)
            lo, hi = (float(v) for v in np.percentile(image, (1, 99.8)))
            self.imdet.set_clim(lo, hi)
            if self.interactive:
                top = float(max(np.percentile(image, 99.99), hi + 1))
                with self.w_contrast.hold_trait_notifications():
                    self.w_contrast.max = top
                    self.w_contrast.step = max(top / 2000, 0.01)
                    self.w_contrast.value = [lo, hi]
            title = f'image {self.imageindex}'
        else:
            self.imdet.set_data(np.zeros(self.imdet.get_array().shape))
            title = 'ROIs'
        k = int(np.flatnonzero(self.display.imageindices == self.imageindex)[0]) \
            if np.any(self.display.imageindices == self.imageindex) else None
        if k is not None:
            dimfast = self.display.mapdimensions[0]
            ifast, islow = k % dimfast, k // dimfast
            self.mark.set_data([islow], [ifast]) if self.display.transposed else self.mark.set_data([ifast], [islow])
            value = f", grain value {self.grains['score'][k, self.grain]:.2f}"
        else:
            value = ''
        spots, per_roi, corfile = self.grain_spots()
        xy = np.array([self._peaklists[self.imageindex][1][i, :2] - 1 for i in spots]) if spots else np.zeros((0, 2))
        self.corspots.set_offsets(xy)   # (.cor positions start at (1, 1))
        best = ' (brightest)' if self.imageindex == g['best_image'] else ''
        self.axdet.set_title(f"{title}{best}{value}: {g['nbrois']} ROIs of grain {self.grain} (red boxes)", fontsize=10)
        self.fig.canvas.draw_idle()
        if self.interactive:
            rois = ', '.join(f'{int(i)} ({x}, {y})' for i, (x, y) in zip(g['rois'], g['roicenters']))
            self.w_spots.value = str(spots) if corfile is not None else ''
            if corfile is None:
                spotinfo = (f'no .cor file of image {self.imageindex} found (folder corfiles of the map, or a '
                            f'subfolder of the map in PROCESSED_DATA or RAW_DATA; else give corfolder=...)')
            else:
                withspot = sum(1 for v in per_roi.values() if v)
                details = ', '.join(f'{roi}: {v}' for roi, v in per_roi.items() if v)
                spotinfo = (f"spots of <b>{corfile}</b> in the {g['nbrois']} ROIs of grain {self.grain} "
                            f"({withspot} ROIs with a spot, green circles, list in the box above), indices in the "
                            f".cor file:<br>"
                            f"<code style='user-select: all'>{spots}</code><br>spots of each ROI: {details or '-'}")
            self.w_info.value = (f"<b>grain {self.grain}</b> / {len(self.grains['grains'])}: <b>{g['nbrois']} ROIs</b>, "
                                 f"area {g['area']} points, centroid (fast, slow) ({g['centroid'][0]:.1f}, "
                                 f"{g['centroid'][1]:.1f}), brightest image {g['best_image']}, image shown "
                                 f"{self.imageindex}<br>{spotinfo}<br>ROIs (index (X, Y)): {rois}")

    def _peaklist(self, imageindex: int):
        """(path, peak list (X, Y ... in the order of the file)) of the .cor file of an image, (None, None) if none"""
        if imageindex not in self._peaklists:
            found = self._corfile_path(imageindex)
            peaklist = None
            if found is not None:
                from LaueTools import map_workflow as MW
                _, nbpeaks, peaklist = MW._read_peaklist_file((imageindex, str(found)))
                peaklist = peaklist if nbpeaks > 0 else np.zeros((0, 2))
            self._peaklists[imageindex] = (found, peaklist)
        return self._peaklists[imageindex]

    def _corfile_path(self, imageindex: int) -> Optional[Path]:
        """.cor file of an image: corfolder (or its subfolders) if given, else folder corfiles of the map in
        PROCESSED_DATA (or its subfolders), else the most recent one in the subfolders of the map folders in
        PROCESSED_DATA and RAW_DATA (e.g. img_CORfiles, datfiles_notebook)"""
        name = f"{self.params['prefix']}{imageindex:0{self.params.get('nbdigits', 4)}d}.cor"
        if self.corfolder is not None:
            roots = [self.corfolder]
        else:
            roots = []
            try:
                from LaueTools import map_workflow as MW
                processed = MW.processed_folder(self.params['folder'])
                roots = [processed / 'corfiles']
            except (ValueError, KeyError):
                processed = None
        for root in roots:
            for candidate in [root / name] + sorted(root.glob(f'*/{name}')):
                if candidate.exists():
                    return candidate
        if self.corfolder is not None:
            return None
        if not hasattr(self, '_corfolders'):   # subfolders of the map with .cor files (searched once)
            folders = []
            for base in [processed, Path(self.params['folder'])]:
                if base is not None and base.is_dir():
                    folders += [d for d in base.iterdir() if d.is_dir() and '_previous_' not in d.name]
            self._corfolders = folders
        found = [d / name for d in self._corfolders if (d / name).exists()]
        return max(found, key=lambda path: path.stat().st_mtime) if found else None

    def grain_spots(self, imageindex: Optional[int] = None, grain: Optional[int] = None):
        """spots of the .cor file of an image (default: image shown) inside the ROI boxes of a grain (default: grain
        shown). Returns (sorted list of their indices in the .cor file (from 0, as in the LaueTools GUI), {ROI index:
        indices of its spots, nearest to the ROI center first}, path of the .cor file (None: no file))"""
        imageindex = self.imageindex if imageindex is None else int(imageindex)
        g = self.grains['grains'][self.grain if grain is None else int(grain)]
        corfile, peaklist = self._peaklist(imageindex)
        if corfile is None:
            return [], {}, None
        xy = peaklist[:, :2] - 1. if len(peaklist) else np.zeros((0, 2))   # .cor positions start at (1, 1)
        bx, by = self.box[0] + 0.5 + self.spot_margin, self.box[1] + 0.5 + self.spot_margin
        per_roi = {}
        for roi, (xc, yc) in zip(g['rois'], g['roicenters']):
            inside = np.flatnonzero((np.abs(xy[:, 0] - xc) <= bx) & (np.abs(xy[:, 1] - yc) <= by))
            dist = np.hypot(xy[inside, 0] - xc, xy[inside, 1] - yc)
            per_roi[int(roi)] = [int(i) for i in inside[np.argsort(dist)]]
        spots = sorted({i for v in per_roi.values() for i in v})
        return spots, per_roi, str(corfile)
