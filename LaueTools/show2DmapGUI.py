# -*- coding: utf-8 -*-
"""
show2DmapGUI: plot again the results files of the Mosaic & Monitor tools of PeakSearchGUI (GUI/mosaic.py),
in the same windows as the GUI:

- roicounters_*.h5: results of the parallel computation of the counters (all counters: mosaic, mean, max,
  ptp, positions, fit results), see mosaic.plot_counters_results()
- text map files: MOSAIC_image_Plot_*, <counter>_2D_* and files of File/Save Data of the map windows,
  see mosaic.read_map_file()

usage:  show2DmapGUI [file or folder ...]
    without argument: dialog to choose the files. For a folder: dialog to choose among its results files.
"""
import os
import sys


def choose_files(paths, MOS, wx):
    """results files to plot: files of paths, and files chosen in a dialog for folders of paths
    (or among all files if paths is empty)"""
    if not paths:
        dlg = wx.FileDialog(None, "Open results of mosaic (roicounters .h5 or text map files)",
                            wildcard=MOS.MAPFILES_WILDCARD, style=wx.FD_OPEN | wx.FD_MULTIPLE)
        files = list(dlg.GetPaths()) if dlg.ShowModal() == wx.ID_OK else []
        dlg.Destroy()
        return files
    files = []
    for path in paths:
        if not os.path.isdir(path):
            files.append(path)
            continue
        candidates = MOS.list_map_files(path)
        if not candidates:
            print("no results file of mosaic in", path)
            continue
        dlg = wx.MultiChoiceDialog(None, "Results files of %s\n(most recent first)" % path, "show2DmapGUI",
                                   [os.path.basename(name) for name in candidates])
        if dlg.ShowModal() == wx.ID_OK:
            files += [candidates[index] for index in dlg.GetSelections()]
        dlg.Destroy()
    return files


def choose_counters(path, MOS, wx):
    """counters of roicounters_*.h5 file to plot (dialog, all counters selected), None if cancelled"""
    counters = MOS.counters_of_results_file(path)
    if len(counters) <= 1:
        return counters
    dlg = wx.MultiChoiceDialog(None, "Counters to plot", os.path.basename(path), counters)
    dlg.SetSelections(list(range(len(counters))))
    selected = [counters[index] for index in dlg.GetSelections()] if dlg.ShowModal() == wx.ID_OK else None
    dlg.Destroy()
    return selected


def start(argv=None):
    """launch show2DmapGUI on the files or folders of argv (default: command line arguments)"""
    import wx
    from LaueTools.GUI import mosaic as MOS

    paths = sys.argv[1:] if argv is None else argv
    app = wx.App()
    for path in choose_files(paths, MOS, wx):
        try:
            counters = None
            if path.endswith((".h5", ".hdf5")):
                counters = choose_counters(path, MOS, wx)
                if not counters:
                    continue
            MOS.show_map_file(path, counters=counters)
        except (ValueError, KeyError, OSError) as exc:
            print("cannot plot %s: %s" % (path, exc))
    if wx.GetTopLevelWindows():
        app.MainLoop()


if __name__ == "__main__":
    start()
