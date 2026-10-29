"""
Bands of the detectors without data (gaps between the modules or quadrants), and their outlines in the spaces
where Laue spots are plotted (pixels, 2theta chi, gnomonic projection), e.g. to show them in the GUIs.

The bands are those removed by the peak search (see readmccd.ptsindices_in_bands(), ptsindices_in_bands_scmos())
"""
from typing import List, Optional

import numpy as np

# pixel bands [first, last] (included) along X (vertical bands) and Y (horizontal bands) of the image
DETECTOR_BANDS = {
    'EIGER_4MCdTe': {'x': [[513, 514], [1028, 1039], [1553, 1554]],
                     'y': [[512, 549], [1062, 1099], [1612, 1649]]},
    'sCMOS': {'x': [[1007, 1009]], 'y': [[1007, 1009]]},
}


def has_bands(CCDLabel: str) -> bool:
    return CCDLabel in DETECTOR_BANDS


def band_rectangles(CCDLabel: str, framedim) -> List[np.ndarray]:
    """rectangles (4 corners [X, Y], pixels) of the bands of the detector CCDLabel (framedim: (nb of rows, nb of
    columns) of the image)"""
    bands = DETECTOR_BANDS.get(CCDLabel)
    if bands is None:
        return []
    ny, nx = int(framedim[0]), int(framedim[1])
    rects = []
    for x0, x1 in bands['x']:
        rects.append(np.array([[x0 - 0.5, -0.5], [x1 + 0.5, -0.5], [x1 + 0.5, ny - 0.5], [x0 - 0.5, ny - 0.5]]))
    for y0, y1 in bands['y']:
        rects.append(np.array([[-0.5, y0 - 0.5], [nx - 0.5, y0 - 0.5], [nx - 0.5, y1 + 0.5], [-0.5, y1 + 0.5]]))
    return rects


def _outline(rect: np.ndarray, npoints: int) -> np.ndarray:
    """points along the 4 edges of a rectangle (closed outline)"""
    pts = []
    for k in range(4):
        a, b = rect[k], rect[(k + 1) % 4]
        t = np.linspace(0., 1., npoints, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    return np.vstack(pts)


def band_polygons(CCDLabel: str, framedim, space: str = 'pixels', detectorparameters=None,
                  pixelsize: Optional[float] = None, kf_direction: str = 'Z>0', chi_offset: float = 0.,
                  npoints: int = 200) -> List[np.ndarray]:
    """outlines (arrays of [x, y] points) of the bands of the detector in the plot space:
    'pixels' (X, Y), '2thetachi' (2theta, chi in deg, chi + chi_offset) or 'gnomon' (gnomonic X, Y, as
    indexingImageMatching.ComputeGnomon_2()). detectorparameters (dd, xcen, ycen, xbet, xgam), pixelsize and
    kf_direction are needed for '2thetachi' and 'gnomon'. Empty list: detector without bands"""
    rects = band_rectangles(CCDLabel, framedim)
    if space == 'pixels' or not rects:
        return rects
    from LaueTools import LaueGeometry as F2TC
    polygons = []
    for rect in rects:
        pts = _outline(rect, npoints)
        tth, chi = F2TC.calc_uflab(pts[:, 0], pts[:, 1], list(detectorparameters)[:5], pixelsize=pixelsize,
                                   kf_direction=kf_direction, returnAngles=1)
        tth, chi = np.asarray(tth, dtype=float), np.asarray(chi, dtype=float) + chi_offset
        if space == '2thetachi':
            polygons.append(np.c_[tth, chi])
        elif space == 'gnomon':
            from LaueTools import indexingImageMatching as IIM
            gx, gy = IIM.ComputeGnomon_2((tth, chi))[:2]
            polygons.append(np.c_[gx, gy])
        else:
            raise ValueError(f"space must be 'pixels', '2thetachi' or 'gnomon', not {space}")
    return [p[np.all(np.isfinite(p), axis=1)] for p in polygons]


def draw_bands(ax, polygons, offset=(0., 0.), color='0.5', alpha=0.25):
    """draw the band outlines (band_polygons()) as transparent filled polygons on matplotlib axes ax
    (offset subtracted from the points, e.g. the pixel offset of the plotted spots). Returns the patches"""
    from matplotlib.patches import Polygon
    patches = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        patch = Polygon(np.asarray(poly) - np.asarray(offset), closed=True, facecolor=color, edgecolor='none',
                        alpha=alpha, zorder=0.5)
        ax.add_patch(patch)
        patches.append(patch)
    return patches
