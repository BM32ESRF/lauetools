# -*- coding: utf-8 -*-
"""
mosaicdata: data side of the Mosaic & Monitor tools (no GUI): peak fits on series of ROIs; and the names that
GUI/mosaic.py still provides for existing scripts and notebooks
"""
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import LaueTools.mosaicdata as MD

ROOT = Path(__file__).resolve().parents[1]
MOVED = ["DEFAULT_DICTfittingparameters", "results_writers", "COUNTERS_QUANTITY", "counter_quantity",
         "MAPFILE_DICT_PARAM_KEYS", "LEGACY_MAPFILE_PATTERNS", "WORKFLOW_TO_GUI_COUNTERS", "map_file_header",
         "read_map_file", "legacy_image_indices", "read_counters_results", "counters_of_results_file",
         "list_map_files", "FitPeakOnMap", "filter_fitted_peaks", "CollectData", "CollectData_oneImage"]


def gaussian_rois(centers, amplitude=1000.0, background=50.0, sigma=1.5, boxsize=21):
    """array (nlines, ncols, boxsize, boxsize) of 2D gaussian peaks; centers[i][j] = (x, y) in the box"""
    y, x = np.mgrid[0:boxsize, 0:boxsize]
    rois = np.zeros(np.shape(centers)[:2] + (boxsize, boxsize))
    for i, line in enumerate(centers):
        for j, (cx, cy) in enumerate(line):
            rois[i, j] = background + amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return rois


def test_fit_peaks():
    """all peaks fitted (amplitude, width) and accepted, at positions near the ROI center whose mutual distances
    are those of the peaks in the boxes (independent of the pixel axes convention)"""
    centers = [[(10.0, 10.0), (10.3, 9.5), (10.6, 9.0)],
               [(9.6, 10.2), (9.9, 9.7), (10.2, 9.2)]]
    with contextlib.redirect_stdout(io.StringIO()):
        XY, fitted = MD.FitPeakOnMap(gaussian_rois(centers), (500, 600), peaksizeStart=1.5, verbose=0)
    assert not np.ma.is_masked(XY)
    XY = np.ma.getdata(XY)
    np.testing.assert_allclose(np.asarray(fitted)[:, 2], 1000.0, rtol=1e-4)  # amplitude
    np.testing.assert_allclose(np.asarray(fitted)[:, 3], 1.5, rtol=1e-4)  # width
    assert np.all(np.abs(XY - (500, 600)) < 2)
    true = np.array([c for line in centers for c in line])
    distances = lambda pts: np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    np.testing.assert_allclose(distances(XY), distances(true), atol=1e-3)


def test_weak_peak_rejected():
    rois = gaussian_rois([[(10.0, 10.0), (10.0, 10.0)]])
    rois[0, 1] = gaussian_rois([[(10.0, 10.0)]], amplitude=10.0)[0, 0]  # below MinimumPeakAmplitude=25
    with contextlib.redirect_stdout(io.StringIO()):
        XY, _ = MD.FitPeakOnMap(rois, (500, 600), reject_weakPeaks=True, MinimumPeakAmplitude=25,
                                peaksizeStart=1.5, verbose=0)
    mask = np.ma.getmaskarray(XY)
    assert not mask[0].any() and mask[1].all()


def test_imports_without_wx():
    code = "import sys; sys.modules['wx'] = None; import LaueTools.mosaicdata"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr.strip().splitlines()[-1:]


def test_names_still_in_gui_mosaic():
    """scripts and notebooks using LaueTools.GUI.mosaic.<name> keep working"""
    MOS = pytest.importorskip("LaueTools.GUI.mosaic")
    for name in MOVED:
        assert getattr(MOS, name) is getattr(MD, name), name
