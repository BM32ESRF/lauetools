# -*- coding: utf-8 -*-
"""
BLISS hdf5 logfiles (logfile_reader): scan commands, monitor counter, lighter logfiles proposed for heavy ones,
listing of a logfile without mesh. Small synthetic files (RAW_DATA/<proposal>_bm32.h5 -> sample -> dataset).
"""
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
LR = pytest.importorskip("LaueTools.logfile_reader")


def test_read_fullcommand_fscan2d():
    """fscan2d slow_motor start step npoints fast_motor start step npoints: last positions from the steps"""
    dc = LR.read_fullcommand("fscan2d xps -2.13666 0.0005 33 yps -4.15143 0.0005 33 0.1 0.100004")
    assert (dc["slowmotor"], dc["fastmotor"]) == ("xps", "yps")
    assert (dc["npts_slow"], dc["npts_fast"]) == (33, 33)
    assert (dc["smotstep"], dc["fmotstep"]) == (0.0005, 0.0005)
    assert dc["smotmax"] == pytest.approx(-2.12066)
    assert dc["fmotmax"] == pytest.approx(-4.13543)


def test_read_fullcommand_fscan_and_amesh():
    dc = LR.read_fullcommand("fscan xech 1.0 0.002 51 0.1 0.1")
    assert dc["npts_fast"] == 51 and dc["fmotmax"] == pytest.approx(1.1)
    dc = LR.read_fullcommand("amesh yech -6.43556 -6.28556 50 xech 0.94557 1.09557 50 0.5")
    assert (dc["fastmotor"], dc["fmotmin"], dc["fmotmax"], dc["npts_fast"]) == ("yech", -6.43556, -6.28556, 51)
    assert (dc["slowmotor"], dc["npts_slow"], dc["expotime"]) == ("xech", 51, 0.5)


@pytest.mark.parametrize("counters, expected", [(["epoch", "mon", "xech"], "mon"), (["epoch", "I0"], "I0"),
                                                (["ct2_mon", "xech"], "ct2_mon"), (["epoch", "xech"], None)])
def test_find_monitor_counter(counters, expected):
    assert LR.find_monitor_counter(counters) == expected


def write_dataset_logfile(path, commands):
    """dataset logfile of BLISS: one entry per scan with title, dates and 1D counters"""
    with h5py.File(path, "w") as f:
        for k, command in enumerate(commands, start=1):
            g = f.create_group("%d.1" % k)
            g["title"] = command.encode()
            g["start_time"] = ("2026-10-29T10:%02d:00" % k).encode()
            g["end_time"] = ("2026-10-29T10:%02d:30" % k).encode()
            g["end_reason"] = b"SUCCESS"
            g.create_group("measurement")["mon"] = np.ones(4)


def make_raw_data(root):
    """RAW_DATA with an experiment logfile -> 2 sample logfiles -> dataset logfiles (external links),
    and a detector file in a scan folder"""
    raw = root / "RAW_DATA"
    links = {}
    for sample, datasets in (("Ge", {"Ge_align": ["ascan xech 0 1 10 0.1"]}),
                             ("Zr", {"Zr_map": ["amesh xech 0 1 2 yech 0 1 1 0.1"], "Zr_ct": ["ct 1"]})):
        (raw / sample).mkdir(parents=True)
        sample_links = {}
        for dataset, commands in datasets.items():
            folder = raw / sample / dataset
            (folder / "scan0001").mkdir(parents=True)
            write_dataset_logfile(folder / ("%s.h5" % dataset), commands)
            with h5py.File(folder / "scan0001" / "eiger4m_0000.h5", "w") as f:  # detector file
                f["data"] = np.zeros((2, 4, 4))
            for k in range(1, len(commands) + 1):
                sample_links["%s_%d.1" % (dataset, k)] = ("%s/%s.h5" % (dataset, dataset), "/%d.1" % k)
        with h5py.File(raw / sample / ("prop_%s.h5" % sample), "w") as f:
            for name, (filename, h5path) in sample_links.items():
                f[name] = h5py.ExternalLink(filename, h5path)
        links.update({name: ("%s/%s" % (sample, filename), h5path) for name, (filename, h5path) in sample_links.items()})
    with h5py.File(raw / "prop_bm32.h5", "w") as f:
        for name, (filename, h5path) in links.items():
            f[name] = h5py.ExternalLink(filename, h5path)
    return raw


def test_lighter_logfiles(tmp_path):
    raw = make_raw_data(tmp_path)
    assert LR.nb_scans_in_logfile(str(raw / "prop_bm32.h5")) == 3
    proposed = LR.lighter_logfiles(str(raw / "prop_bm32.h5"))  # sample logfiles
    assert sorted((os.path.relpath(p, raw), n) for p, n in proposed) == [("Ge/prop_Ge.h5", 1), ("Zr/prop_Zr.h5", 2)]
    proposed = LR.lighter_logfiles(str(raw / "Zr" / "prop_Zr.h5"))  # dataset logfiles, no detector file
    assert sorted(os.path.basename(p) for p, _ in proposed) == ["Zr_ct.h5", "Zr_map.h5"]
    assert LR.lighter_logfiles(str(raw / "Zr" / "Zr_map" / "Zr_map.h5")) == []  # scan0001 not explored


def test_listing_without_mesh(tmp_path):
    """a logfile without mesh gives empty lists (IndexError before Oct. 2026)"""
    raw = make_raw_data(tmp_path)
    assert len(LR.getmeshscan_from_hdf5file(str(raw / "Ge" / "prop_Ge.h5"))) == 0
    meshes = LR.getmeshscan_from_hdf5file(str(raw / "prop_bm32.h5"))
    assert [m[5].split(" ", 1)[1] for m in meshes] == ["amesh xech 0 1 2 yech 0 1 1 0.1"]


@pytest.mark.parametrize("command, motors", [("ascan xech 0 1 10 0.1", ["xech"]), ("dscan yech -1 1 20 0.1", ["yech"]),
                                             ("a2scan xech 0 1 yech 0 1 10 0.1", ["xech", "yech"]),
                                             ("a3scan xech 0 1 yech 0 1 zech 0 1 10 0.1", ["xech", "yech", "zech"]),
                                             ("fscan ho3 0.71 0.0014 700 0.01 0.0178", ["ho3"]),
                                             ("amesh xech 0 1 2 yech 0 1 2 0.1", []), ("fscan2d xps 0 1 2 yps 0 1 2 0.1", []),
                                             ("loopscan 10 0.1", []), ("ct 1", [])])
def test_scan1D_motors(command, motors):
    assert LR.is_scan1D_command(command) == bool(motors)
    assert LR.scan1D_motors(command) == motors


def test_find_image_counter():
    assert LR.find_image_counter(["mon", "img_cam", "xech"]) == "img_cam"
    assert LR.find_image_counter(["mon", "eiger4m_fulldet"]) is None  # EIGER: image index = point index


def test_listing_scans1D(tmp_path):
    """1D scans of the ASCAN list of plotmeshGUI: ascan, not amesh nor ct"""
    raw = make_raw_data(tmp_path)
    scans = LR.getscans_from_hdf5file(str(raw / "prop_bm32.h5"), collectallscans=False, onlyscan1D=True)[0]
    assert [m[5].split(" ", 1)[1] for m in scans] == ["ascan xech 0 1 10 0.1"]


def test_scan_folder(tmp_path):
    """scan folder next to the dataset logfile (subtitle of plotmeshGUI plots), path from RAW_DATA"""
    raw = make_raw_data(tmp_path)
    datasetfile = raw / "Zr" / "Zr_map" / "Zr_map.h5"
    assert LR.scan_folder(datasetfile, "1") == str(raw / "Zr" / "Zr_map" / "scan0001")
    assert LR.scan_folder(datasetfile, 2) is None  # no folder scan0002
    assert LR.path_from_raw_data(LR.scan_folder(datasetfile, 1)) == "RAW_DATA/Zr/Zr_map/scan0001"
    assert LR.path_from_raw_data("/tmp/data/scan0001") == "/tmp/data/scan0001"
