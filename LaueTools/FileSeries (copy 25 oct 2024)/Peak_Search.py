# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:23:18 2013

@author: js micha

from initially T. Cerba
"""
import sys
import os

sys.path.append("..")

import wx

# this is for running through ipython
import matplotlib
matplotlib.use("WXAgg")
#------------------------

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

# LaueTools modules
if sys.version_info.major == 3:
    from .. import readmccd as RMCCD
    from .. import dict_LaueTools as DictLT
    from .. import IOLaueTools as IOLT

else:
    import readmccd as RMCCD
    import dict_LaueTools as DictLT
    import IOLaueTools as IOLT

dict_CCD = DictLT.dict_CCD
LIST_OF_CCDS = list(dict_CCD.keys())
LIST_OF_CCDS.sort()


# --- ---- Local maxima and fit parameters
LIST_TXTPARAMS = RMCCD.LIST_OPTIONS_PEAKSEARCH
# ['local_maxima_search_method',
#                 'IntensityThreshold',
#                 'thresholdConvolve',
#                 'boxsize',
#                 'PixelNearRadius',
#                 'fit_peaks_gaussian',
#                 'xtol',
#                 'FitPixelDev',
#                 'position_definition',
#                 'maxPixelDistanceRejection']

LIST_VALUESPARAMS = RMCCD.LIST_OPTIONS_VALUESPARAMS
# = [1,
#                 1000,
#                 5000,
#                 15,
#                 10,
#                 1,
#                 0.001,
#                 2.0,
#                 1,
#                 15.]

LIST_UNITSPARAMS = RMCCD.LIST_OPTIONS_TYPE_PEAKSEARCH
# ['integer flag',
#                 'integer count',
#                 'float count',
#                 'pixel',
#                 'pixel',
#                 'integer flag',
#                 '',
#                 'pixel',
#                 'integer flag',
#                 'float']


LIST_TXTPARAM_FILE_PS = [
    "ImageFolder",
    "OutputFolder",
    "ImageFilename Prefix",
    "ImageFilename Suffix",
    "ImageFilename Nbdigits",
    "Starting Image index",
    "Final Image index",
    "Image index step",
    "Background Removal",
    "BlackListed Peaks File",
    "Selected ROIs File",
    "PeakSearch Parameters File (.psp)",
]

TIP_PS = ["Folder containing image files",
    "Folder containing results Peaks List .dat files",
    "Prefix for peaks list .dat filename prefix####suffix where #### are digits of file index",
    "Image file suffix.",
    "maximum nb of digits for zero padding of filename index.(e.g. nb of # in prefix####.dat)\n0 "
    "for no zero padding.",
    "starting file index (integer)",
    "final file index (integer)",
    "incremental step for file index (integer)",
    "Background removal type: None, auto (self background) or file path to image B and optionally "
    "formula expression separated by ;\ne.g. /home/lauetools/myimageB.mccd;A-3.0*B",
    "full path to .dat file (as made by peak search procedure) containing peaks to be removed from "
    "the list of peaks found in the current image.",
    "full path to .rois file containing list of pixel centers and boxsizex, boxsizey with the "
    "simple format in each line for a roi: x y halfboxx halboxy (separated or not by , ; etc...",
    "full path to .psp file containing Peak Search parameters",
]

DICT_TOOLTIP = {}
for key, tip in zip(LIST_TXTPARAM_FILE_PS, TIP_PS):
    DICT_TOOLTIP[key] = "%s : %s" % (key, tip)


class Stock_parameters_PeakSearch:
    def __init__(self, _list_txtparamPS, _list_valueparamPS):
        self.list_txtparamIR = _list_txtparamPS
        self.list_valueparamIR = _list_valueparamPS


# --- --  Class of Local Maxima and peak fitting board
class PeakSearchParameters(wx.Frame):
    def __init__(self, parent, _id, title, _listParameters):

        _list_txtparamPS, _list_valueparamPS, _list_unitsparams = _listParameters

        self.parent = parent

        self.list_txtparamIR = _list_txtparamPS
        self.list_valueparamIR = _list_valueparamPS
        self.list_unitsparams = _list_unitsparams

        nbrows = len(_list_txtparamPS)

        wx.Frame.__init__(self, parent, _id, title, wx.DefaultPosition, wx.Size(500, nbrows * 40))

        if WXPYTHON4:
            grid = wx.FlexGridSizer(3, 10, 10)
        else:
            grid = wx.FlexGridSizer(nbrows, 3, 10, 10)

        grid.SetFlexibleDirection(wx.HORIZONTAL)
        self.panel = wx.Panel(self)

        self.tooltips()

        self.list_txtctrl = []

        # set value according to list_txtparamIR, list_valueparamIR, list_unitsparams
        for kk, elem in enumerate(self.list_txtparamIR):
            text_field = "  " + elem
            if kk == 2:
                text_field += " (method 2)"
            wxtxt = wx.StaticText(self.panel, -1, text_field)
            grid.Add(wxtxt)

            self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(-1, 25))
            self.list_txtctrl.append(self.txtctrl)
            grid.Add(self.txtctrl)

            grid.Add(wx.StaticText(self.panel, -1, self.list_unitsparams[kk]))

            wxtxt.SetToolTipString(self.tips_dict[kk])
            self.txtctrl.SetToolTipString(self.tips_dict[kk])

        self.dict_param = {}
        self.setParams()

        btnSave = wx.Button(self.panel, -1, "ACCEPT && SAVE", size=(-1, 50))
        btnSave.Bind(wx.EVT_BUTTON, self.OnSave)

        btnLoad = wx.Button(self.panel, -1, "Load File", size=(-1, 50))
        btnLoad.Bind(wx.EVT_BUTTON, self.OnLoad)

        btnLoadDefault = wx.Button(self.panel, -1, "Reset", size=(-1, 50))
        btnLoadDefault.Bind(wx.EVT_BUTTON, self.OnLoadDefault)

        btnQuit = wx.Button(self.panel, -1, "Cancel && Quit", size=(-1, 50))
        btnQuit.Bind(wx.EVT_BUTTON, self.OnQuit)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(btnSave, 0, wx.EXPAND)
        hbox.Add(btnLoad, 0, wx.EXPAND)
        hbox.Add(btnLoadDefault, 0, wx.EXPAND)
        hbox.Add(btnQuit, 0, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        vbox.AddSpacer(10)
        vbox.Add(hbox, 0, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

        # tooltip
        btnSave.SetToolTipString("Save current parameters to a .psp file")
        btnLoadDefault.SetToolTipString("Reset parameters to default values")
        btnLoad.SetToolTipString("Select a .psp file to load peak search parameters")
        btnQuit.SetToolTipString("Quit without any changes")

    def tooltips(self):
        self.tips_dict = {}
        for kk, elem in enumerate(self.list_txtparamIR):
            self.tips_dict[kk] = "%s : " % elem

        self.tips_dict[0] += "Local maxima search method to provide initial guesses parameters for spot refinement.\n"
        self.tips_dict[0] += "0 : basic threshold; 1: Array shift method 2: kernel convolution method"
        self.tips_dict[1] += "Minimum threshold level on image pixel intensity. Peaks weaker than this threshold will be omitted\n"
        self.tips_dict[1] += "For method 2: Minimum threshold level on image pixel intensity with respect to local background intensity level"
        self.tips_dict[1] += "(lowest intensity in a box centered on local maximum position)"
        self.tips_dict[2] += "Minimum threshold level on convoluted image pixel (method 2 only)"
        self.tips_dict[3] += "half boxsize of the data array (centered on local maximum pixel) to be refined"

        self.tips_dict[4] += "Minimum distance separating two local maxima"
        self.tips_dict[5] += ("Model for spot shape refinement: 0 no fit; 1 2D gaussian; 2: 2D lorentzian")
        self.tips_dict[6] += "fit parameter (useless). To be removed."

        self.tips_dict[7] += "Maximum pixel distance separating initial spot pixel position (guessed by local maxima search method) and spot shape and position refinement.\n"
        self.tips_dict[7] += "Spot whose refined position is too far from guessed one will be rejected as bad results."
        self.tips_dict[8] += "Offset convention on Spot position coordinates with respect to array indices.\n"
        self.tips_dict[8] += "0: no shift; 1: first element in array has (1,1) pixel coordinates."
        self.tips_dict[9] += "Distance between found and blacklisted peaks below which found peak will be rejected (Exclusion radius of region around each unwanted peaks)"

    def hascorrectvalue(self, kk, val):
        flag = True
        # int  = 0 1 2
        if kk in (0, 5):
            try:
                v = int(val)
                if v not in (0, 1, 2):
                    wx.MessageBox(
                        "%s must be equal to 0,1 or 2" % self.list_txtparamIR[kk],
                        "Error")
                    flag = False
            except ValueError:
                wx.MessageBox("wrong type for %s! (kk=%d) Must be integer" % (self.list_txtparamIR[kk], kk), "Error")
                flag = False

        # int  = 0 1
        if kk in (8,):
            try:
                v = int(val)
                if v not in (0, 1):
                    wx.MessageBox("%s must be equal to 0 or 1" % self.list_txtparamIR[kk], "Error")
                    flag = False
            except ValueError:
                wx.MessageBox("wrong type for %s! (kk=%d) Must be integer"
                    % (self.list_txtparamIR[kk], kk), "Error")
                flag = False

        #         elif kk == 1:
        #             try:
        #                 v = int(float(val))
        #             except ValueError:
        #                 wx.MessageBox('wrong type for %s! (kk=%d) Must be integer' % (self.list_txtparamIR[kk], kk), 'Error')
        #                 flag = False
        elif kk == 2 and val not in (None, "None"):
            try:
                v = int(float(val))
                if v <= 1:
                    wx.MessageBox("%s must be greater than 1" % self.list_txtparamIR[kk], "Error")
                    flag = False
            except ValueError:
                wx.MessageBox("wrong type for %s! (kk=%d) Must be integer"
                                % (self.list_txtparamIR[kk], kk), "Error")
                flag = False

        # positive integer > 1
        elif kk in (1, 3, 4):
            try:
                v = int(val)
                if v < 1:
                    wx.MessageBox("%s must be >= 1" % self.list_txtparamIR[kk], "Error")
                    flag = False
            except ValueError:
                wx.MessageBox("wrong type for %s! (kk=%d) Must be integer"
                                % (self.list_txtparamIR[kk], kk), "Error")
                flag = False

        # positive float
        elif kk in (6, 7, 9):
            try:
                v = float(val)
                if v <= 0:
                    wx.MessageBox("%s must be positive" % self.list_txtparamIR[kk], "Error")
                    flag = False
            except ValueError:
                wx.MessageBox("wrong type for %s! (kk=%d) Must be float"
                                % (self.list_txtparamIR[kk], kk), "Error")
                flag = False

        return flag

    def setParams(self):
        for kk, _ in enumerate(self.list_txtparamIR):
            self.list_txtctrl[kk].SetValue(str(self.list_valueparamIR[kk]))

    def getParams(self):
        self.dict_param = {}
        flag = True
        for kk, _ in enumerate(self.list_txtparamIR):
            val = self.list_txtctrl[kk].GetValue()

            if not self.hascorrectvalue(kk, val):
                flag = False
                break

            self.list_valueparamIR[kk] = val

            self.dict_param[self.list_txtparamIR[kk]] = val

        print("Get dict of Params:")
        print(self.dict_param)
        return flag

    def OnSave(self, _):

        if not self.getParams():
            return

        wcd = "PeakSearch Param.(*.psp)|*.psp|All files(*)|*"

        defaultdir = self.parent.list_txtctrl[0].GetValue()
        if not os.path.isdir(defaultdir):
            defaultdir = os.getcwd()

        file = wx.FileDialog(
                            self,
                            "Save psp File",
                            defaultDir=defaultdir,
                            wildcard=wcd,
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if file.ShowModal() == wx.ID_OK:

            outputfile = file.GetPath()

            outputfilename = RMCCD.savePeakSearchConfigFile(self.dict_param, outputfilename=outputfile)

            # outputfilename has .psp extension
            self.parent.list_txtctrl[11].SetValue(outputfilename)

        self.Close()

    def OnLoad(self, _):
        pspfile = wx.FileDialog(
                                self,
                                "Open Peak Search Parameters",
                                wildcard="PeakSearch Param.(*.psp)|*.psp|All files(*)|*")
        if pspfile.ShowModal() == wx.ID_OK:

            filepsp = pspfile.GetPath()

            dict_param = RMCCD.readPeakSearchConfigFile(filepsp)

            print("dict_param", dict_param)

            for kk, key in enumerate(LIST_TXTPARAMS):
                if key in list(dict_param.keys()):
                    self.list_valueparamIR[kk] = dict_param[key]

            print("self.list_valueparamIR", self.list_valueparamIR)

            self.setParams()

    def OnLoadDefault(self, _):
        print("reset")
        self.list_valueparamIR = LIST_VALUESPARAMS
        self.setParams()

    def OnQuit(self, _):
        self.Close()


# ---  -- Class of Folder and Filename parameters for PeakSearch Series
class MainFrame_peaksearch(wx.Frame):
    def __init__(self, parent, _id, title, _initialparameters, objet_PS):
        wx.Frame.__init__(self, parent, _id, title, wx.DefaultPosition, wx.Size(1000, 500))

        self.initialparameters = _initialparameters

        self.parent = parent
        self.objet_PS = objet_PS

        self.allMaterialsnames = LIST_OF_CCDS
        self.CCDlabel = None

        if WXPYTHON4:
            grid = wx.FlexGridSizer(3, 7, 7)
        else:
            grid = wx.FlexGridSizer(11, 3, 7, 7)

        grid.SetFlexibleDirection(wx.HORIZONTAL)
        self.panel = wx.Panel(self)

        dict_tooltip = DICT_TOOLTIP
        keys_list_dicttooltip = list(DICT_TOOLTIP.keys())

        self.list_txtctrl = []

        for kk, txt_elem in enumerate(objet_PS.list_txtparamIR):
            txt = wx.StaticText(self.panel, -1, "     %s" % txt_elem)
            grid.Add(txt)

            self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))
            self.txtctrl.SetValue(str(objet_PS.list_valueparamIR[kk]))
            self.list_txtctrl.append(self.txtctrl)
            grid.Add(self.txtctrl)

            if txt_elem in keys_list_dicttooltip:

                txt.SetToolTipString(dict_tooltip[txt_elem])
                self.txtctrl.SetToolTipString(dict_tooltip[txt_elem])

            if kk in (0, 1, 2, 8, 9, 10, 11):
                btnbrowse = wx.Button(self.panel, kk + 10, "Browse File/Folder")
                grid.Add(btnbrowse)
                if kk == 0:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepath)
                    btnbrowse.SetToolTipString("Select Folder containing image files (.tif, .mccd)")
                elif kk == 1:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout)
                    btnbrowse.SetToolTipString(
                        "Select Folder to write results (peaks list .dat files) if different to that proposed automatically")
                elif kk == 2:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_CCDfileimage)
                    btnbrowse.SetToolTipString(
                        "Select one image file to get (and guess) the generic prefix of all image filenames")
                elif kk == 8:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_bkgfile)
                    btnbrowse.SetToolTipString("Select image file to be used as background")
                elif kk == 9:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_Blacklistfile)
                    btnbrowse.SetToolTipString(
                        "Select peak list .dat file to remove peaks from the current peaks list")
                elif kk == 10:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_ROIslistfile)
                    btnbrowse.SetToolTipString(
                        "Select ROIs list .rois file to restrict peaksearch in these regions")
                elif kk == 11:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_pspfile)
                    btnbrowse.SetToolTipString("Select .psp file containing Peak Search Parameters to be applied on all images")

            elif kk == 3:
                self.comboCCD = wx.ComboBox(self.panel, -1, _initialparameters["CCDLabel"], size=(150, -1),
                                                                choices=self.allMaterialsnames,
                                                                style=wx.CB_READONLY)
                self.comboCCD.Bind(wx.EVT_COMBOBOX, self.EnterComboCCD)
                grid.Add(self.comboCCD)
            else:
                nothing = wx.StaticText(self.panel, -1, "")
                grid.Add(nothing)
            #grid.Add(wx.Button(self.panel, kk + 12, "?", size=(25, 25)))

        Createcfgbtn = wx.Button(self.panel, -1, "Create .psp file", size=(150, -1))
        Createcfgbtn.Bind(wx.EVT_BUTTON, self.OnCreatePSP)

        # multiprocessing handling
        txt_cpus = wx.StaticText(self.panel, -1, "nb CPU(s)")
        self.txtctrl_cpus = wx.TextCtrl(self.panel, -1, "1")

        # button START
        btnStart = wx.Button(self.panel, -1,
                            "START PEAK SEARCH (Peaklist files .dat in OutputFolder)",
                            size=(300, 50))

        btnStart.Bind(wx.EVT_BUTTON, self.OnStart)

        # widgets layout--------------
        hfinal = wx.BoxSizer(wx.HORIZONTAL)
        hfinal.Add(Createcfgbtn, 0)
        hfinal.AddSpacer(10)
        hfinal.Add(txt_cpus, 0)
        hfinal.Add(self.txtctrl_cpus, 0)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        vbox.Add(hfinal, 0, wx.EXPAND)
        vbox.AddSpacer(5)
        vbox.Add(btnStart, 1, wx.EXPAND)

        self.panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

        # tooltips
        btnStart.SetToolTipString("Start Peak Search on all images and create a list of peaks "
        "(.dat file) for each image")
        Createcfgbtn.SetToolTipString("Create .psp file containing parameters to peak search & "
        "peak position refinement")
        tipcpus = "nb of cores to use to find peaks in all images"
        txt_cpus.SetToolTipString(tipcpus)
        self.txtctrl_cpus.SetToolTipString(tipcpus)

    def OnCreatePSP(self, _):
        filepsp = self.list_txtctrl[11].GetValue()
        if not os.path.exists(filepsp):
            return

        dict_param = RMCCD.readPeakSearchConfigFile(filepsp)

        print("dict_param", dict_param)
        listvals = [] * len(LIST_VALUESPARAMS)
        for _, key in enumerate(LIST_TXTPARAMS):
            if key in list(dict_param.keys()):
                listvals.append(dict_param[key])
            else:
                print("key = %s is not in dict_param.keys()" % key)
                listvals.append(None)

        PSPboard = PeakSearchParameters(self, -1, "PeakSearch Parameters",
                                                    (LIST_TXTPARAMS, listvals, LIST_UNITSPARAMS))
        PSPboard.Show(True)

    def OnbtnBrowse_filepath(self, _):
        folder = wx.DirDialog(self, "Select a folder containing images")
        if folder.ShowModal() == wx.ID_OK:
            abspath = folder.GetPath()
            self.list_txtctrl[0].SetValue(abspath)
            self.list_txtctrl[1].SetValue(os.path.join(abspath, "datfiles"))

    def OnbtnBrowse_filepathout(self, _):
        folder = wx.DirDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[1].SetValue(folder.GetPath())

    def OnbtnBrowse_CCDfileimage(self, _):
        folder = wx.FileDialog(self, "Select CCD Image file")
        if folder.ShowModal() == wx.ID_OK:
            abspath = folder.GetPath()
            filename = os.path.split(abspath)[-1]
            intension, extension = filename.split(".")
            try:
                nbdigits = int(self.list_txtctrl[4].GetValue())
            except ValueError:
                nbdigits = 4

            prefixfilename = intension[:-nbdigits]
            whole_extension = "." + extension

            # special case for  blahblah_##_mar.tif case with varying number of #...
            if filename.endswith("mar.tif"):
                prefixfilename = intension.split("_")[0] + "_"
                whole_extension = "_mar.tif"
                self.list_txtctrl[4].SetValue("varying")

            self.list_txtctrl[2].SetValue(prefixfilename)
            self.list_txtctrl[3].SetValue(whole_extension)

    def EnterComboCCD(self, event):
        item = event.GetSelection()
        self.CCDlabel = self.allMaterialsnames[item]
        print("item", item)
        print("CCDlabel", self.CCDlabel)

        if self.CCDlabel == "VHR_Mar13":
            extension = str(dict_CCD[self.CCDlabel][7])
            self.list_txtctrl[4].SetValue("varying")
        else:
            extension = "." + str(dict_CCD[self.CCDlabel][7])

        self.list_txtctrl[3].SetValue(extension)

        event.Skip()

    def OnbtnBrowse_bkgfile(self, _):
        bkgfile = wx.FileDialog(self, "Select background image",
                                    wildcard="MARCCD or ROPER file (*.mccd)|*.mccd|All files(*)|*")
        if bkgfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[8].SetValue(bkgfile.GetPath())

    def OnbtnBrowse_Blacklistfile(self, _):
        blacklist_datfile = wx.FileDialog(self, "Select BlackList .dat file image",
                                            wildcard="peaklist file (*.dat)|*.dat|All files(*)|*")
        if blacklist_datfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[9].SetValue(blacklist_datfile.GetPath())

    def OnbtnBrowse_ROIslistfile(self, _):
        ROIslist_file = wx.FileDialog(self, "Select ROIs .rois file image",
                                            wildcard="ROIs file (*.rois)|*.rois|All files(*)|*")
        if ROIslist_file.ShowModal() == wx.ID_OK:

            self.list_txtctrl[10].SetValue(ROIslist_file.GetPath())

    def OnbtnBrowse_pspfile(self, _):
        pspfile = wx.FileDialog(self,
            "Select Peak Search Parameters File",
            wildcard="PeakSearch Param.(*.psp)|*.psp|All files(*)|*")
        if pspfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[11].SetValue(pspfile.GetPath())

    def Onbtnhelp_filepath(self, _):
        help_tip = "Folder containing images file"
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_Nbpicture1(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_Nblastpicture(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_increment(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_filepathout(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_fileprefix(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_filesuffix(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def Onbtnhelp_Nbdigits(self, _):
        help_tip = "to be filled..."
        self.help.SetValue(str(help_tip))

    def datFolderExists(self):
        datfolder = self.list_txtctrl[1].GetValue()
        if not os.path.isdir(datfolder):
            try:
                print("I have created a folder for resulting .dat files")
                print("datfolder", datfolder)
                os.mkdir(datfolder)
                return True
            except IOError:
                wx.MessageBox("Can not create %s to contain peak list .dat files !" % datfolder, "Error")
                return False

        return True

    def imagesFolderExists(self):
        imagesfolder = self.list_txtctrl[0].GetValue()

        if not os.path.isdir(imagesfolder):
            wx.MessageBox("Can not find %s containing images!" % imagesfolder, "Error")
            return False

        return True

    def OnStart(self, _):
        """   read all ctrls and start processing  """
        # read .psp file
        filepsp = self.list_txtctrl[11].GetValue()
        if not os.path.exists(filepsp):
            wx.MessageBox("Missing file ! %s"%str(filepsp), "ERROR")
            return
        print("read peak search parameters in:", filepsp)

        dict_param = RMCCD.readPeakSearchConfigFile(filepsp)

        print("Peak Search parameters", dict_param)

        self.CCDlabel = self.comboCCD.GetValue()
        print("self.CCDlabel", self.CCDlabel)

        try:
            imageindexmin = int(self.list_txtctrl[5].GetValue())
            imageindexmax = int(self.list_txtctrl[6].GetValue())
            stepimage = int(self.list_txtctrl[7].GetValue())

        except ValueError:
            wx.MessageBox("indices must be integer", "ERROR")

        if (imageindexmin >= imageindexmax or imageindexmin < 0 or imageindexmax < 0):
            wx.MessageBox("Problems with image indices. starting index should be lower "
                                                                    "than final index", "ERROR")

        fileindexrange = (imageindexmin, imageindexmax, stepimage)

        filenameprefix = self.list_txtctrl[2].GetValue()
        suffix = self.list_txtctrl[3].GetValue()

        nbdigits = self.list_txtctrl[4].GetValue()
        try:
            nbdigits_resultfiles = int(nbdigits)
        except ValueError:
            DEFAULT_DIGITSENCODING = 4
            nbdigits_resultfiles = DEFAULT_DIGITSENCODING
        if not self.datFolderExists() or not self.imagesFolderExists():
            return

        dirname_in = self.list_txtctrl[0].GetValue()
        dirname_out = self.list_txtctrl[1].GetValue()

        # setting folders for the next analysis steps

        if self.parent is not None:
            parent = self.parent  # IR

            # print("parent.initialparameters", parent.initialparameters)

            parent.initialparameters["ImageFolder"] = dirname_in
            parent.initialparameters["Output Folder (Peaklist)"] = dirname_out
            parent.initialparameters["ImageFilename Prefix"] = filenameprefix
            parent.initialparameters["ImageFilename Suffix"] = suffix

            parent.initialparameters["PeakList Folder"] = dirname_out
            parent.initialparameters["IndexRefine PeakList Folder"] = os.path.join(os.path.dirname(dirname_out), "fitfiles")
            parent.initialparameters["PeakListCor Folder"] = os.path.join(os.path.dirname(dirname_out), "corfiles")
            parent.initialparameters["PeakList Filename Prefix"] = filenameprefix

            parent.initialparameters["startingindex"] = imageindexmin
            parent.initialparameters["finalindex"] = imageindexmax
            parent.initialparameters["nbdigits"] = nbdigits_resultfiles
            parent.initialparameters["stepindex"] = stepimage

            # print("AFTER start pressed button: parent.initialparameters", parent.initialparameters)

        #        progressMax = 100
        #        dialog = wx.ProgressDialog("A progress box", "Time remaining", progressMax,
        #        style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
        #        keepGoing = True
        #        count = 0
        #        while keepGoing and count < progressMax:
        #            count = count + 1
        #            wx.Sleep(1)
        #            keepGoing = dialog.Update(count)
        #
        #        dialog.Destroy()

        # set parameter for background removal
        background_flag = str(self.list_txtctrl[8].GetValue())
        Data_for_localMaxima, formulaexpression = RMCCD.read_background_flag(background_flag)
        dict_param["Data_for_localMaxima"] = Data_for_localMaxima
        dict_param["formulaexpression"] = formulaexpression

        # set parameter  for blacklisted peaks
        blacklistpeaklist = self.list_txtctrl[9].GetValue()
        dict_param["Remove_BlackListedPeaks_fromfile"] = RMCCD.set_blacklist_filepath(blacklistpeaklist)

        # set parameter  for blacklisted peaks
        ROIslistfile = self.list_txtctrl[10].GetValue()

        pathroisfile = RMCCD.set_rois_file(ROIslistfile)
        print('pathroisfile', pathroisfile)
        if pathroisfile is not None:
            dict_param["listrois"] = IOLT.read_roisfile(pathroisfile)
        else:
            dict_param["listrois"] = None
        if dict_param["listrois"] is not None:
            print('\n  Finding only peaks in given ROIs according to parameters of .psp file')

        try:
            nb_cpus = int(self.txtctrl_cpus.GetValue())
        except ValueError:
            wx.MessageBox("nb of cpu(s) must be positive integer!", "Error")
            return
        if nb_cpus <= 0:
            wx.MessageBox("nb of cpu(s) must be positive integer!", "Error")
            return

        # check the first imagefile to read:
        #         print "dict_param in file series", dict_param
        # dict_param['listrois']=[(723,1530,35,35),(723,1530,5,5),(723,1538,7,7),(673,1769,15,41),]
        # dict_param['IntensityThreshold']=1000
        if nb_cpus == 1:
            RMCCD.peaksearch_fileseries(fileindexrange, filenameprefix=filenameprefix,
                                                        suffix=suffix,
                                                        nbdigits=nbdigits,
                                                        dirname_in=dirname_in,
                                                        outputname=None,
                                                        dirname_out=dirname_out,
                                                        CCDLABEL=self.CCDlabel,
                                                        KF_DIRECTION="Z>0",
                                                        dictPeakSearch=dict_param,
                                                        verbose=0,
                                                        writeResultDicts=0,
                                                        computetime=1)
        else:
            RMCCD.peaksearch_multiprocessing(fileindexrange, filenameprefix=filenameprefix,
                                                            suffix=suffix,
                                                            nbdigits=nbdigits,
                                                            dirname_in=dirname_in,
                                                            outputname=None,
                                                            dirname_out=dirname_out,
                                                            CCDLABEL=self.CCDlabel,
                                                            KF_DIRECTION="Z>0",
                                                            dictPeakSearch=dict_param,
                                                            nb_of_cpu=nb_cpus,
                                                            verbose=0,
                                                            writeResultDicts=0)


LaueToolsProjectFolder = DictLT.LAUETOOLSFOLDER
print("LaueToolProjectFolder in main", LaueToolsProjectFolder)
MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "GeGaN")
print("MainFolder in main", MainFolder)

initialparameters = {}
initialparameters["ImageFolder"] = MainFolder
initialparameters["Output Folder (Peaklist)"] = os.path.join(MainFolder, "datfiles")
initialparameters["ImageFilename Prefix"] = "nanox2_400_"
initialparameters["ImageFilename Suffix"] = ".mccd"
initialparameters["PeakSearch Parameters File"] = os.path.join(
    MainFolder, "PeakSearch_nanox2_400_0000_LT_4.psp")
initialparameters["CCDLabel"] = "MARCCD165"
initialparameters["BackgroundRemoval"] = "auto"
initialparameters["BlackListed Peaks File"] = None
initialparameters["Selected ROIs File"] = None

list_valueparamPS = [initialparameters["ImageFolder"],
                    initialparameters["Output Folder (Peaklist)"],
                    initialparameters["ImageFilename Prefix"],
                    initialparameters["ImageFilename Suffix"],
                    4, 0, 5, 1,
                    initialparameters["BackgroundRemoval"],
                    initialparameters["BlackListed Peaks File"],
                    initialparameters["Selected ROIs File"],
                    initialparameters["PeakSearch Parameters File"]]

def start():
    """ launcher """
    Stock_PS = Stock_parameters_PeakSearch(LIST_TXTPARAM_FILE_PS, list_valueparamPS)

    PeakSearchSeriesApp = wx.App()
    PeakSearchSeries = MainFrame_peaksearch(None, -1, "Peak Search Parameters Board",
                                                                        initialparameters, Stock_PS)
    PeakSearchSeries.Show()
    PeakSearchSeriesApp.MainLoop()

if __name__ == "__main__":

    start()
