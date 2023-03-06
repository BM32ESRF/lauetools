# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:20:21 2013

@author: micha

from initially T. Cerba

lasr revision June 2021

"""
import os
import sys
import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        """ translate name of tooltip
        """
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip
from wx.lib.agw.shapedbutton import SButton

sys.path.append("..")

import LaueTools.FileSeries.multigrainFS as MGFS

list_txtparamPM = ["Map Summary File", "File xyz", "maptype"]
list_txtparamPM_type_dict = {"Map Summary File": str, "File xyz": str, "maptype": str}


class MainFrame_plotmaps(wx.Frame):
    def __init__(self, parent, _id, title, dict_params):
        wx.Frame.__init__(self, parent, _id, title, size=(1000, 2000))

        self.dict_params = dict_params
        self.parent = parent
        self.list_of_windows = []
        self.list_txtctrl_new = []
        print((self.dict_params))
        fullpath_summaryfile = self.dict_params["Map Summary File"]
        folderpath, summaryfile = os.path.split(fullpath_summaryfile)
        fullpath_filexyz = self.dict_params["File xyz"]
        folderpathxyz, summaryfilexyz = os.path.split(fullpath_filexyz)

        font = wx.Font(18, wx.MODERN, wx.ITALIC, wx.NORMAL)

        # GUI
        self.panel = wx.Panel(self)

        txt_fileparameters = wx.StaticText(self.panel, -1, "File parameters ")
        txt_fileparameters.SetFont(font)

        txt_summary = wx.StaticText(self.panel, -1, "Map SUMMARY .dat File")
        btnbrowse1 = wx.Button(self.panel, -1, "Browse")
        btnbrowse1.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filesum)

        self.summarypath_folder = wx.TextCtrl(
            self.panel, -1, folderpath, size=(400, -1))
        self.summarypath_file = wx.TextCtrl(self.panel, -1, summaryfile, size=(400, -1))

        hbox1summaryfile = wx.BoxSizer(wx.HORIZONTAL)
        hbox1summaryfile.Add(btnbrowse1, 0, wx.ALL)
        hbox1summaryfile.AddSpacer(10)
        hbox1summaryfile.Add(txt_summary, 0, wx.ALL)

        # file xyz
        txt_filexyz = wx.StaticText(self.panel, -1, "Map File XYZ .dat ")
        btnbrowse2 = wx.Button(self.panel, -1, "Browse")
        btnbrowse2.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filexyz)

        self.filexyzpath_folder = wx.TextCtrl(
            self.panel, -1, folderpathxyz, size=(400, -1))
        self.filexyzpath_file = wx.TextCtrl(
            self.panel, -1, summaryfilexyz, size=(400, -1))

        hbox3summaryfile = wx.BoxSizer(wx.HORIZONTAL)
        hbox3summaryfile.Add(btnbrowse2, 0, wx.ALL)
        hbox3summaryfile.AddSpacer(10)
        hbox3summaryfile.Add(txt_filexyz, 0, wx.ALL)

        txt_mapparameters = wx.StaticText(self.panel, -1, "Map Type ")
        txt_mapparameters.SetFont(font)

        self.choice_maptype = wx.Choice(self.panel, -1,
                        choices=["fit", "strain6_crystal", "rgb_x_sample",
                        "strain6_sample", "euler3", "stress6_crystal", "stress6_sample",
                        "res_shear_stress", "max_rss", "von_mises", "w_mrad", ], )
        self.choice_maptype.SetSelection(0)

        self.choice_maptype.Bind(wx.EVT_CHOICE, self.Onchoice_maptype)

        grainindextxt = wx.StaticText(self.panel, -1, "grain index")
        self.grainindexctrl = wx.SpinCtrl(self.panel, -1, '0',size=(60, -1), min=0, max=5)

        self.btnclearwindows = wx.Button(
            self.panel, -1, "Clear Windows", size=(200, -1))
        self.btnclearwindows.Bind(wx.EVT_BUTTON, self.OnClearChildWindows)

        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        hbox0.Add(grainindextxt, 0, wx.ALL)
        hbox0.Add(self.grainindexctrl, 1, wx.EXPAND)
        vbox5 = wx.BoxSizer(wx.VERTICAL)
        vbox5.Add(self.choice_maptype, 0, wx.EXPAND)
        vbox5.Add(hbox0, 0, wx.EXPAND)
        vbox5.Add(self.btnclearwindows, 0, wx.ALL)

        btnplot = SButton(self.panel, -1, "PLOT", size=(-1, 60))
        btnplot.Bind(wx.EVT_BUTTON, self.OnPlot)

        hboxplot = wx.BoxSizer(wx.HORIZONTAL)
        hboxplot.Add(vbox5, 0, wx.ALL)
        hboxplot.Add(btnplot, 1, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt_fileparameters, 0, wx.CENTER, 0)
        vbox.Add(hbox1summaryfile, 0, wx.ALL, 0)
        vbox.Add(self.summarypath_folder, 0, wx.ALL)
        vbox.Add(self.summarypath_file, 0, wx.ALL)
        vbox.AddSpacer(20)
        vbox.Add(hbox3summaryfile, 0, wx.ALL, 0)
        vbox.Add(self.filexyzpath_folder, 0, wx.ALL, 0)
        vbox.Add(self.filexyzpath_file, 0, wx.ALL)
        vbox.AddSpacer(20)
        vbox.Add(txt_mapparameters, 0, wx.CENTER, 0)
        vbox.Add(hboxplot, 0, wx.EXPAND, 0)

        self.panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

        # tooltip
        txt_summary.SetToolTipString(
            "File elaborated by Build_Summary.py module as a summary of all images analysis")
        btnbrowse1.SetToolTipString(
            "Browse to find : File elaborated by Build_Summary.py module as a summary of all images analysis"
        )
        self.summarypath_folder.SetToolTipString("folder of summary file")
        self.summarypath_file.SetToolTipString("Filename of summary file")
        txt_filexyz.SetToolTipString("A file containing 3 columns imageindex, x, y")
        self.filexyzpath_folder.SetToolTipString("folder of file xy")
        self.filexyzpath_file.SetToolTipString("Filename of file xy")

        tiptype = "Select the quantity to plot"
        txt_mapparameters.SetToolTipString(tiptype)
        self.choice_maptype.SetToolTipString(tiptype)

        self.btnclearwindows.SetToolTipString("Clear Previous plotting windows")

    def OnClearChildWindows(self, _):
        print("killing child windows!")
        #         print "list_of_windows", self.list_of_windows
        for child in self.list_of_windows:
            try:
                child.Close()
            except:
                continue
        self.list_of_windows = []

    def OnReadParameters(self):
        self.list_txtctrl_new = []

        filesummary_path_folder = str(self.summarypath_folder.GetValue())
        filesummary_path_file = str(self.summarypath_file.GetValue())
        filesummary_path = os.path.join(filesummary_path_folder, filesummary_path_file)

        filexyz_path_folder = str(self.filexyzpath_folder.GetValue())
        filexyz_path_file = str(self.filexyzpath_file.GetValue())

        filexyz_path = os.path.join(filexyz_path_folder, filexyz_path_file)

        if filexyz_path_file in ("None",):
            filexyz_path = None

        self.dict_params["Map Summary File"] = filesummary_path
        self.dict_params["File xyz"] = filexyz_path

        self.dict_params["maptype"] = self.choice_maptype.GetString(
            self.choice_maptype.GetSelection())

        return self.dict_params

    def Onchoice_maptype(self, _):
        self.dict_params["maptype"] = self.choice_maptype.GetString(
            self.choice_maptype.GetSelection())

    def OnbtnBrowse_filesum(self, _):
        folder = wx.FileDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            fold, filen = os.path.split(folder.GetPath())
            self.summarypath_folder.SetValue(fold)
            self.summarypath_file.SetValue(filen)

    def OnbtnBrowse_filexyz(self, _):
        folder = wx.FileDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            fold, filen = os.path.split(folder.GetPath())
            self.filexyzpath_folder.SetValue(fold)
            self.filexyzpath_file.SetValue(filen)

    def OnPlot(self, _):
        dictpars = self.OnReadParameters()
        print(("self.dict_params", self.dict_params))

        maptype = dictpars["maptype"]
        grain_index = int(self.grainindexctrl.GetValue())
        MGFS.plot_map_new2(self.dict_params, maptype, grain_index, App_parent=self)


class Stock_parameters_PlotMaps:
    def __init__(self, _list_txtparamPM, _list_valueparamPM):
        self.list_txtparamPM = _list_txtparamPM
        self.dict_params = _list_valueparamPM

initialparameters = {}

LaueToolsProjectFolder = os.path.dirname(os.path.abspath(os.curdir))

print(("LaueToolProjectFolder", LaueToolsProjectFolder))

MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "GeGaN")

print(("MainFolder", MainFolder))

initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
initialparameters["Map Summary File"] = os.path.join(
    MainFolder, "fitfiles", "nanox2_400__SUMMARY_0_to_5_add_columns.dat")
initialparameters["File xyz"] = os.path.join(
    MainFolder, "fitfiles", "nanox2_400__xy_0_to_5.dat")
initialparameters["maptype"] = "fit"
initialparameters["filetype"] = "LT"

initialparameters["Map shape"] = (31, 41)  # (nb lines, nb images per line)

initialparameters["(stepX, stepY) microns"] = (1.0, 1.0)

MainFolder = "/home/micha/LaueProjects/LeBaudy"
initialparameters["Map Summary File"] = os.path.join(
    MainFolder, "mappos1__SUMMARY_0_to_1425_add_columns.dat")
initialparameters["File xyz"] = os.path.join(MainFolder, "mappos1__xy_0_to_1425.dat")

initialparameters["stiffness file"] = os.path.join(MainFolder, "si.stf")

initialparameters["nbdigits"] = 4
initialparameters["startingindex"] = 0
initialparameters["finalindex"] = 5
initialparameters["stepindex"] = 1
initialparameters["fast axis: x or y"] = "x"

print(os.path.abspath(__file__))
absmodulefolder = os.path.split(os.path.abspath(__file__))[0]
LaueToolsProjectFolder = os.path.split(absmodulefolder)[0]

MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "UO2")
initialparameters["Map Summary File"] = os.path.join(MainFolder, "UO2_SUMMARY_0_to_609.dat")
initialparameters["File xyz"] = os.path.join(MainFolder, "xy_0_609.dat")

# MainFolder = '/home/micha/LaueProjects/LeBaudy'
# initialparameters['Map Summary File'] = os.path.join(MainFolder,
#                                            'mappos1__SUMMARY_0_to_1425_add_columns.dat')
# initialparameters['File xyz'] = os.path.join(MainFolder, 'mappos1__xy_0_to_1425.dat')
# initialparameters['stiffness file'] = os.path.join(MainFolder, 'si.stf')

list_valueparamPM = [initialparameters["Map Summary File"], initialparameters["File xyz"], "fit"]

def prepare_params_for_plot(list_val, list_key=None):
    if list_key is None:
        list_key = list_txtparamPM

    if len(list_val) != len(list_key):
        sentence = "Lengthes of list_val and list_key differ in prepare_params_for_plot() in Plot_maps\n"
        sentence += "Nb_of_parameters= %d, Nb of values %d !!!\n" % (len(list_key), len(list_val))
        sentence += "Please check the number of parameters for Plot_Maps inputs"
        raise ValueError(sentence)

    zippeddict = list(zip(list_key, list_val))

    resdict = {}
    for key, val in zippeddict:
        resdict[key] = val

    return resdict

def start():
    dict_parameters = prepare_params_for_plot(list_valueparamPM)
    PlotMapsApp = wx.App()
    PMFrame = MainFrame_plotmaps(None, -1, "Plot_Maps2.py GUI Board", dict_parameters)
    PMFrame.Show(True)
    PlotMapsApp.MainLoop()


if __name__ == "__main__":
    start()