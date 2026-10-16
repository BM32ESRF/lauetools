r"""
roicountersplot: plots of ROI counters maps (results of workflows.ROICountersWorkflow over a 2D map)

- plot_counter_maps(): grid of maps of one counter for several ROIs (shared zoom)
- CounterMapBrowser: browser of maps (ROI index, counter, contrast), keeping zoom when browsing,
  ROIs centers on detector (click on a peak: its map),
  image index under mouse, ROI pixel intensities of clicked image ('mosaic' counter)

Maps are displayed with the sample y motor (yech, yps ...) vertical, whatever the fast and slow axes
(data are transposed for display only). Mouse hovering gives image index for both cases.

Zoom keeping and mouse hovering need an interactive matplotlib backend in notebook
(%matplotlib widget, with ipympl installed)::

    import LaueTools.scripts.roicountersplot as RCP
    RCP.plot_counter_maps(results, 'max', roiindices=range(12), fastaxis='yech')
    browser = RCP.CounterMapBrowser(results, fastaxis='yech', slowaxis='xech')
    browser.show()
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

from typing import Any, Dict, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import LaueTools.scripts.workflows as wf


def is_vertical_axis(motorname: Optional[str]) -> bool:
    """True for motor displayed along vertical axis of maps (y motors: yech, yps, ...)"""
    return motorname is not None and str(motorname).lower().startswith('y')


def counter_columns(counter: str) -> Sequence[str]:
    """names of values of a counter (for 'column' argument)"""
    if counter == 'fit':
        return wf.FIT_COLUMNS
    if counter in ('XYmax', 'XYcentroid'):
        return ('X', 'Y')
    return (counter,)


def counter_values(results: Dict[str, Any], counter: str, column=0) -> np.ndarray:
    """values of counter (column: index or name, see counter_columns()) of shape (nb images, nb rois)"""
    if counter not in results:
        raise KeyError(f"counter '{counter}' not computed. Available: "
                       f"{[c for c in wf.ROI_COUNTERS if c in results and c != 'mosaic']}")
    if wf.ROI_COUNTERS[counter] is None:
        raise ValueError(f"counter '{counter}' is an array of pixel intensities (not a scalar)")
    if isinstance(column, str):
        column = list(counter_columns(counter)).index(column)
    return np.asarray(results[counter][:, :, column], dtype=float)


class MapDisplay:
    """display of 2D map of values (in map order, line by line along fast axis): y motor vertical,
    image index of display pixel"""

    def __init__(self, results: Dict[str, Any], fastaxis: Optional[str] = None,
                 slowaxis: Optional[str] = None):
        params = results.get('params', {})
        self.mapdimensions = tuple(params['mapdimensions'])  # (nb along fast axis, nb along slow axis)
        self.imageindices = np.asarray(results['imageindices'])
        self.fastaxis = fastaxis if fastaxis is not None else params.get('fastaxis', 'fast axis')
        self.slowaxis = slowaxis if slowaxis is not None else params.get('slowaxis', 'slow axis')
        # map2D shape (slow, fast): fast axis horizontal. Transposed if fast axis is a y motor
        self.transposed = is_vertical_axis(self.fastaxis) and not is_vertical_axis(self.slowaxis)
        if self.transposed:
            self.xlabel, self.ylabel = f'slow axis {self.slowaxis}', f'fast axis {self.fastaxis}'
        else:
            self.xlabel, self.ylabel = f'fast axis {self.fastaxis}', f'slow axis {self.slowaxis}'

    def data(self, values: np.ndarray) -> np.ndarray:
        """2D array to display (origin='lower') from values of images (map order)"""
        map2D = wf.counter_map2D(values, self.mapdimensions)  # (slow, fast)
        return map2D.T if self.transposed else map2D

    def position(self, x: float, y: float):
        """(map position k in images list, fast index, slow index) of display point x, y, or None"""
        col, row = int(round(x)), int(round(y))
        ifast, islow = (row, col) if self.transposed else (col, row)
        dimfast, dimslow = self.mapdimensions
        if not (0 <= ifast < dimfast and 0 <= islow < dimslow):
            return None
        k = islow * dimfast + ifast
        return (k if k < len(self.imageindices) else None), ifast, islow

    def imageindex(self, x: float, y: float) -> Optional[int]:
        pos = self.position(x, y)
        return None if pos is None or pos[0] is None else int(self.imageindices[pos[0]])

    def format_coord_function(self, getvalues):
        """format_coord for axes: image index and value (getvalues(): current values of images)"""
        def format_coord(x, y):
            pos = self.position(x, y)
            if pos is None:
                return f'x={x:.1f}, y={y:.1f}'
            k, ifast, islow = pos
            if k is None:
                return f'fast {ifast}, slow {islow}: no image'
            return (f'image {int(self.imageindices[k])} (fast {ifast}, slow {islow}) '
                    f'value {getvalues()[k]:.6g}')
        return format_coord

    def imshow(self, ax, values: np.ndarray, **kwargs):
        im = ax.imshow(self.data(values), origin='lower', **kwargs)
        ax.format_coord = self.format_coord_function(lambda: values)
        return im


def contrast_limits(values: np.ndarray, percentiles=(1, 99)):
    """(vmin, vmax) from percentiles of finite values (None, None: matplotlib default)"""
    finite = values[np.isfinite(values)]
    if percentiles is None or finite.size == 0:
        return None, None
    return tuple(np.percentile(finite, percentiles))


def plot_counter_maps(results: Dict[str, Any], counter: str = 'max', roiindices=None, column=0,
                      nbcols: int = 5, percentiles=(1, 99), colorbar: bool = True,
                      fastaxis: Optional[str] = None, slowaxis: Optional[str] = None,
                      mapwidth: float = 3.2, cmap=None):
    """grid of maps of a counter for several ROIs (each with its own contrast), zoom shared by all maps

    :param roiindices: indices of ROIs (in results['params']['roicenters']), default: 10 first ROIs
    :param column: value of counter (index or name, see counter_columns(), e.g. 'X' for 'fit')
    :param percentiles: (low, high) percentiles of contrast of each map, None: min, max
    :return: fig, axes
    """
    values = counter_values(results, counter, column)
    roicenters = results['params']['roicenters']
    if roiindices is None:
        roiindices = range(min(10, len(roicenters)))
    roiindices = list(roiindices)
    display = MapDisplay(results, fastaxis, slowaxis)
    nbcols = max(1, min(nbcols, len(roiindices)))
    nbrows = -(-len(roiindices) // nbcols)
    ratio = display.data(values[:, 0]).shape
    fig, axes = plt.subplots(nbrows, nbcols, sharex=True, sharey=True, squeeze=False,
                             figsize=(mapwidth * nbcols,
                                      mapwidth * nbrows * min(2., max(0.3, ratio[0] / ratio[1])) + 0.6))
    columnname = counter_columns(counter)[column] if isinstance(column, int) else column
    fig.suptitle(counter if columnname == counter else f'{counter} {columnname}')
    for ax, iroi in zip(axes.flat, roiindices):
        vmin, vmax = contrast_limits(values[:, iroi], percentiles)
        im = display.imshow(ax, values[:, iroi], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f'ROI {iroi} {list(roicenters[iroi])}', fontsize=8)
        ax.tick_params(labelsize=7)
        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    for ax in axes.flat[len(roiindices):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(display.xlabel, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(display.ylabel, fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig, axes


def interactive_backend() -> bool:
    """True if figures are interactive (e.g. %matplotlib widget): zoom, mouse hovering and clicks.
    False for inline figures (static images)"""
    return 'inline' not in matplotlib.get_backend().lower()


NO_INTERACTIVE_BACKEND_MSG = ("<b>Inline figures</b>: no zoom, mouse hovering nor click. For an interactive "
                              "browser, run <code>%matplotlib widget</code> before the plots "
                              "(needs <code>pip install ipympl</code> in the kernel environment) and restart.")


class CounterMapBrowser:
    """browser (ipywidgets) of maps of ROI counters

    - left: map of a counter of current ROI. Zoom is kept when browsing. Mouse hovering: image index,
      map position and value. Click: image index and file (and ROI pixels of this image)
    - right: ROIs centers on detector (peak list). Click close to a ROI center: map of this ROI
    - (if 'mosaic' counter computed) ROI pixel intensities of the clicked image
    - widgets: ROI index (slider, play), counter and its value, contrast (percentiles), log10
    """

    def __init__(self, results: Dict[str, Any], counter: Optional[str] = None, iroi: int = 0,
                 fastaxis: Optional[str] = None, slowaxis: Optional[str] = None,
                 figsize=None, cmap=None, framedim=None, showroipixels: bool = True):
        import ipywidgets as widgets

        self.results = results
        self.params = results['params']
        self.display = MapDisplay(results, fastaxis, slowaxis)
        self.counters = [c for c in wf.ROI_COUNTERS if c in results and wf.ROI_COUNTERS[c] is not None]
        if not self.counters:
            raise ValueError('no scalar counter in results')
        counter = counter if counter in self.counters else self.counters[0]
        self.roicenters = np.array(self.params['roicenters'], dtype=float).reshape((-1, 2))
        nbrois = len(self.roicenters)
        self.hasmosaic = 'mosaic' in results and showroipixels
        self.clickedk = None
        self.interactive = interactive_backend()
        if framedim is None:
            try:
                framedim = wf.DictLT.dict_CCD[self.params['CCDLabel']][0]  # (nb lines, nb columns)
            except (KeyError, TypeError):
                framedim = None

        self.w_roi = widgets.IntSlider(value=iroi, min=0, max=nbrois - 1, description='ROI index',
                                       continuous_update=False, layout=widgets.Layout(width='450px'))
        self.w_play = widgets.Play(value=iroi, min=0, max=nbrois - 1, interval=700)
        widgets.jslink((self.w_play, 'value'), (self.w_roi, 'value'))
        self.w_counter = widgets.Dropdown(options=self.counters, value=counter, description='counter')
        self.w_column = widgets.Dropdown(options=list(counter_columns(counter)), description='value')
        self.w_contrast = widgets.FloatRangeSlider(value=(1, 99), min=0, max=100, step=0.5,
                                                   description='percentiles', continuous_update=False)
        self.w_log = widgets.Checkbox(value=False, description='log10')
        self.w_hover = widgets.HTML(value='' if self.interactive else NO_INTERACTIVE_BACKEND_MSG)
        self.w_info = widgets.HTML()

        # figure: map | ROIs centers on detector | (ROI pixels)
        ncols = 3 if self.hasmosaic else 2
        ratios = [1.3, 1.1, 0.9][:ncols]
        if figsize is None:
            figsize = (4.2 * sum(ratios), 4.5)
        with plt.ioff():  # displayed in self.widget only
            self.fig, axes = plt.subplots(1, ncols, figsize=figsize, gridspec_kw={'width_ratios': ratios})
        if not self.interactive:  # else displayed again by plt.show() of inline backend
            plt.close(self.fig)
        self.ax, self.axpeaks = axes[0], axes[1]
        self.axroi = axes[2] if self.hasmosaic else None

        self.im = self.ax.imshow(self.display.data(self._values()), origin='lower', cmap=cmap)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.03)
        self.ax.set_xlabel(self.display.xlabel)
        self.ax.set_ylabel(self.display.ylabel)
        self.ax.format_coord = self.display.format_coord_function(self._values)
        self.marker, = self.ax.plot([], [], '+', color='r', ms=12, mew=2)

        X, Y = self.roicenters.T
        self.axpeaks.scatter(X, Y, s=10, c='tab:blue')
        self.currentpeak, = self.axpeaks.plot([], [], 'o', mfc='none', mec='r', ms=12, mew=2)
        self.currentlabel = self.axpeaks.text(0, 0, '', color='r', fontsize=8)
        if framedim is not None:
            self.axpeaks.set_xlim(0, framedim[1])
            self.axpeaks.set_ylim(framedim[0], 0)  # detector Y downwards
        else:
            self.axpeaks.invert_yaxis()
        self.axpeaks.set_aspect('equal')
        self.axpeaks.set_xlabel('X (pixel)')
        self.axpeaks.set_ylabel('Y (pixel)')
        self.axpeaks.set_title(f'{nbrois} ROIs centers (click: select ROI)', fontsize=9)
        self.axpeaks.format_coord = lambda x, y: self._peak_text(x, y)
        if self.hasmosaic:
            self.imroi = None
            self.axroi.set_title('click on map: ROI pixels', fontsize=9)
        self.fig.tight_layout()

        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._onmove)

        self.w_counter.observe(self._oncounter, names='value')
        for w in (self.w_roi, self.w_column, self.w_contrast, self.w_log):
            w.observe(lambda change: self._update(), names='value')

        if self.interactive and isinstance(self.fig.canvas, widgets.DOMWidget):  # ipympl canvas widget
            self.fig.canvas.header_visible = False
            figurewidget = self.fig.canvas
        else:
            figurewidget = self.out = widgets.Output()
        self.widget = widgets.VBox([widgets.HBox([self.w_roi, self.w_play]),
                                    widgets.HBox([self.w_counter, self.w_column]),
                                    widgets.HBox([self.w_contrast, self.w_log]),
                                    self.w_hover, figurewidget, self.w_info])
        self._update()

    def show(self):
        from IPython.display import display
        display(self.widget)

    def _values(self) -> np.ndarray:
        values = counter_values(self.results, self.w_counter.value, self.w_column.value)[:, self.w_roi.value]
        if self.w_log.value:
            with np.errstate(divide='ignore', invalid='ignore'):
                values = np.where(values > 0, np.log10(values), np.nan)
        return values

    def _oncounter(self, change):
        self.w_column.options = list(counter_columns(change['new']))
        self._update()

    def _update(self):
        values = self._values()
        # set_data(): axes limits (zoom) are unchanged
        self.im.set_data(self.display.data(values))
        vmin, vmax = contrast_limits(values, self.w_contrast.value)
        if vmin is not None:
            self.im.set_clim(vmin, vmax if vmax > vmin else vmin + 1e-9)
        iroi = self.w_roi.value
        X, Y = self.roicenters[iroi]
        column = self.w_column.value
        name = self.w_counter.value if column == self.w_counter.value else f'{self.w_counter.value} {column}'
        self.ax.set_title(f"{'log10 ' if self.w_log.value else ''}{name}, ROI {iroi} ({X:.0f}, {Y:.0f})",
                          fontsize=9)
        self.currentpeak.set_data([X], [Y])
        self.currentlabel.set_position((X, Y))
        self.currentlabel.set_text(f' {iroi}')
        if self.hasmosaic and self.clickedk is not None:
            self._plotroi()
        self._redraw()

    def _redraw(self):
        if self.interactive:
            self.fig.canvas.draw_idle()
        else:  # inline figure: displayed again
            from IPython.display import display
            self.out.clear_output(wait=True)
            with self.out:
                display(self.fig)

    def _nearest_roi(self, x: float, y: float):
        """index of ROI center closest to x, y and its distance (pixels)"""
        dist = np.hypot(self.roicenters[:, 0] - x, self.roicenters[:, 1] - y)
        k = int(np.argmin(dist))
        return k, dist[k]

    def _peak_text(self, x: float, y: float) -> str:
        k, dist = self._nearest_roi(x, y)
        X, Y = self.roicenters[k]
        return f'X={x:.1f}, Y={y:.1f}, nearest ROI {k} ({X:.0f}, {Y:.0f})'

    def _onmove(self, event):
        if event.xdata is None:
            return
        if event.inaxes is self.ax:
            self.w_hover.value = self.ax.format_coord(event.xdata, event.ydata)
        elif event.inaxes is self.axpeaks:
            self.w_hover.value = self._peak_text(event.xdata, event.ydata)
        elif self.axroi is not None and event.inaxes is self.axroi and self.imroi is not None:
            X, Y = int(round(event.xdata)), int(round(event.ydata))
            x0, x1, y1, y0 = self.imroi.get_extent()
            roi = self.imroi.get_array()
            i, j = int(X - x0 - 0.5), int(Y - y0 - 0.5)
            value = roi[j, i] if 0 <= j < roi.shape[0] and 0 <= i < roi.shape[1] else np.nan
            self.w_hover.value = f'pixel X={X}, Y={Y}, intensity {value:.6g}'

    def _onclick(self, event):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if event.xdata is None or toolbar is not None and getattr(toolbar, 'mode', '') != '':
            return  # outside axes or click for zoom or pan
        if event.inaxes is self.axpeaks:
            k, dist = self._nearest_roi(event.xdata, event.ydata)
            # tolerance: 3% of displayed detector width
            xmin, xmax = self.axpeaks.get_xlim()
            if dist <= max(0.03 * abs(xmax - xmin), 2.):
                self.w_roi.value = k  # triggers _update()
            return
        if event.inaxes is not self.ax:
            return
        pos = self.display.position(event.xdata, event.ydata)
        if pos is None or pos[0] is None:
            return
        self.clickedk, ifast, islow = pos
        imageindex = int(self.display.imageindices[self.clickedk])
        self.marker.set_data([round(event.xdata)], [round(event.ydata)])
        try:
            imagepath = wf.image_filepath(self.params, imageindex)[0]
        except Exception:
            imagepath = ''
        self.w_info.value = (f'clicked: image <b>{imageindex}</b> (fast {ifast}, slow {islow}), '
                             f'value {self._values()[self.clickedk]:.6g}<br>{imagepath}')
        if self.hasmosaic:
            self._plotroi()
        self._redraw()

    def _plotroi(self):
        iroi = self.w_roi.value
        roi = self.results['mosaic'][self.clickedk, iroi]
        bX, bY = self.params['boxsize_X'], self.params['boxsize_Y']
        X, Y = self.roicenters[iroi]
        extent = (X - bX - 0.5, X + bX + 0.5, Y + bY + 0.5, Y - bY - 0.5)  # detector pixels, Y downwards
        if self.imroi is None:
            self.imroi = self.axroi.imshow(roi, extent=extent)
            self.axroi.set_xlabel('X (pixel)')
            self.axroi.set_ylabel('Y (pixel)')
        else:
            self.imroi.set_data(roi)
            self.imroi.set_extent(extent)
            if np.isfinite(roi).any():
                self.imroi.set_clim(np.nanmin(roi), np.nanmax(roi))
        self.axroi.set_title(f'ROI {iroi}, image {int(self.display.imageindices[self.clickedk])}',
                             fontsize=9)
