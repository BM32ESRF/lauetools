# -*- coding: utf-8 -*-
"""
results files of GUI/mosaic.py read again by show2DmapGUI: text map files (with header, or without
header as before Oct. 2026) and roicounters .h5 files of workflows.ROICountersWorkflow (no display needed)
"""
import numpy as np
import pytest

MOS = pytest.importorskip("LaueTools.GUI.mosaic")

DICT_PARAM = {"pixelX_center": 995.0, "pixelY_center": 847.0, "imagesfolder": "/data/RAW_DATA/scan0001",
              "filename_representative": "eiger4m_0003.h5", "CCDLabel": "EIGER_4MCdTe", "NormalizeWithMonitor": False}


def test_map_with_header(tmp_path):
    data = np.arange(12.).reshape(3, 4)
    data[2, 3] = np.nan  # unfinished scan
    indices = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8])
    header = MOS.map_file_header("PositionX", "fitted peak X position (pixel)", indices, 4, 3, 0, 0,
                                 "eiger4m_0003.h5 [000000-000010]\npixel ROI at [995,847]", DICT_PARAM)
    path = tmp_path / "Position XY_2D_X_date_20261029_120000"
    np.savetxt(path, data, header=header, comments="")
    assert np.allclose(np.loadtxt(path), data, equal_nan=True)  # header skipped by np.loadtxt

    args = MOS.read_map_file(str(path))
    assert np.allclose(args["dataarray"], data, equal_nan=True)
    assert args["datatype"] == "PositionX" and args["mosaic"] == 0
    assert (args["nb_col"], args["nb_lines"]) == (4, 3)
    assert list(args["Imageindices"]) == list(indices)
    assert args["dict_param"] == DICT_PARAM
    assert args["quantity"] == "fitted peak X position (pixel)"
    assert args["imagename"] == "eiger4m_0003.h5 [000000-000010] pixel ROI at [995,847]"


def test_mosaic_with_header(tmp_path):
    data = np.ones((6, 9))
    header = MOS.map_file_header("Intensity", "ROI pixel intensity", np.arange(6), 3, 2, 1, 1, "title",
                                 DICT_PARAM, mosaic=1, absolutecornerindices=(994, 846))
    path = tmp_path / "MOSAIC_image_Plot_date_20261029_120000"
    np.savetxt(path, data, header=header, comments="")
    args = MOS.read_map_file(str(path))
    assert args["mosaic"] == 1 and args["datatype"] == "Intensity"
    assert (args["nb_col"], args["nb_lines"], args["boxsize_row"], args["boxsize_line"]) == (3, 2, 1, 1)
    assert args["absolutecornerindices"] == (994, 846)


def test_legacy_map_without_header(tmp_path):
    """image indices read in the 1D file of the same date (transposed map: 0, 3, 6, ...)"""
    data = np.arange(9.).reshape(3, 3)
    indices = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
    np.savetxt(tmp_path / "max_2D_date_20261005_124846", data)
    np.savetxt(tmp_path / "max_1D_date_20261005_124846",
               np.array([indices, np.zeros(9), np.zeros(9), np.ravel(data)]).T, header="image fastindex slowindex intensity")
    np.savetxt(tmp_path / "Position XY_2D_Y__date_20260618_153444", data + 800)  # without 1D file

    args = MOS.read_map_file(str(tmp_path / "max_2D_date_20261005_124846"))
    assert args["datatype"] == "Intensity" and args["quantity"] == "max intensity in ROI"
    assert list(args["Imageindices"]) == list(indices)

    args = MOS.read_map_file(str(tmp_path / "Position XY_2D_Y__date_20260618_153444"))
    assert args["datatype"] == "PositionY"
    assert list(args["Imageindices"]) == list(range(9))
    assert args["dict_param"]["pixelY_center"] == pytest.approx(804.)  # background level: mean position

    assert len(MOS.list_map_files(str(tmp_path))) == 2  # the 1D file is not a map


def test_not_a_map_file(tmp_path):
    path = tmp_path / "Shape_2Dshape_date_20261005_124846"
    np.savetxt(path, np.ones((9, 3)))
    with pytest.raises(ValueError):
        MOS.read_map_file(str(path))


def test_counters_results_file(tmp_path):
    WF = pytest.importorskip("LaueTools.scripts.workflows")
    params = {"folder": "/data/RAW_DATA/scan0001", "filename_representative": "eiger4m_0003.h5",
              "CCDLabel": "EIGER_4MCdTe", "nbdigits": 4, "listindices": list(range(4)), "mapdimensions": (2, 2),
              "roicenters": [(995, 847)], "boxsize_X": 10, "boxsize_Y": 10, "counters": ["max", "fit"]}
    path = WF.save_results({"params": params, "max": np.ones((4, 1, 1)), "fit": np.ones((4, 1, 9))},
                           str(tmp_path / "roicounters_X995_Y847_date_20261029_120000.h5"))
    expected = ["max", "Position XY", "Displacement", "Amplitude", "Shape"]
    assert MOS.counters_of_results_file(path) == expected
    results, dict_param = MOS.read_counters_results(path)
    assert dict_param["selectedcounters"] == expected
    assert (dict_param["pixelX_center"], dict_param["pixelY_center"]) == (995.0, 847.0)
    assert dict_param["save_results"] is False  # replot only
    assert results["fit"].shape == (4, 1, 9)
