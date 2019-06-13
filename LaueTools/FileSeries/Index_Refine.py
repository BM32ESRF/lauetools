# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:23:40 2013

@author: micha

from initially T. Cerba

Revised May 2019
"""
import sys
import os

sys.path.append("..")

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

try:
    from multigrain import (
        filter_peaks,
        index_refine_calib_one_image,
        serial_index_refine_multigrain,
        serial_index_refine_multigrain_v2,
    )
except ImportError:
    print("Missing modules or functions of multigrain.py. But It does not matter!")

import IOLaueTools as IOLT
import indexingSpotsSet as ISS

from dict_LaueTools import LAUETOOLSFOLDER
LaueToolsProjectFolder = os.path.abspath(LAUETOOLSFOLDER)
print("LaueToolProjectFolder", LaueToolsProjectFolder)

# --- ---- core index and refine parameters
LIST_TXTPARAMS = ISS.LIST_OPTIONS_INDEXREFINE[1:]

LIST_VALUESPARAMS = [ "Ge", 1, 5, 22, 100.0, 0.5, 0.5, 10, 6, [0],
                    False, 3, None, True, 1000, [1, 1], None, ]

# WARNING when adding parameters above:
# check if field position is correct in def hascorrectvalue(self, kk, val):
# and name does not contain "_"


LIST_UNITSPARAMS = ISS.LIST_OPTIONS_TYPE_INDEXREFINE[1:]


LIST_TXTPARAM_FILE_INDEXREFINE = [
    "Peak List .dat Folder",
    "Peak List (Output) .cor Folder",
    "Peak List (Output) .fit Folder",
    "PeakList Filename (for prefix)",
    "PeakList Filename Suffix",
    "Nbdigits in index filename",
    "Starting Image index",
    "Final Image index",
    "Image index step",
    "Detector Calibration File (.det)",
    "Guessed Matrix(ces) (.mat,.mats,.ubs)",
    "Minimum Matching Rate",
    "IndexRefine Parameters File (.irp)",
]

TIP_IR = [
    "Folder containing indexed Peaks List .dat files",
    "Folder containing (results) Peaks List .cor files",
    "Folder containing (results) indexed Peaks List .fit files",
    "Prefix for .fit files filename prefix####suffix where #### are digits of file index",
    'peak list filename suffix. ".dat" or ".cor"',
    "maximum nb of digits for zero padding of filename index.(e.g. nb of # in prefix####.dat)\n0 for no zero padding.",
    "starting file index (integer)",
    "final file index (integer)",
    "incremental step file index (integer)",
    "full path to detector calibration .det file containing detector plane position and angles parameters\nNot used if PeakList Filename Suffix is .cor",
    "full path to a file (.mat or .mats) containing one or several guessed orientation matrix(ces) or check orientation parameters file (.ubs) to be tested prior to indexation from scratch",
    "Minimum matching rate (nb of matches/ nb of theoritical spots) corresponding to guessed orientation matrices tested.\n if higher than 100, then test of guessed solution orientation matrix(ces) will be omitted",
    "full path to .irp file containing index & refine parameters",
]

DICT_TOOLTIP = {}
for key, tip in zip(LIST_TXTPARAM_FILE_INDEXREFINE, TIP_IR):
    DICT_TOOLTIP[key] = "%s : %s" % (key, tip)

DEFAULT_KF_DIRECTION = "Z>0"


class IndexRefineParameters(wx.Frame):
    """
    class for GUI to create a .irp file
    """
    def __init__(self, parent, _id, title, listParameters, nb_of_materials=1):

        wx.Frame.__init__(self, parent, _id, title, size=(700, 600))

        self.panel = wx.Panel(self)

        self.listParameters = listParameters

        _list_txtparamIR, _list_valueparamIR, _list_unitsparams = listParameters

        self.parent = parent

        self.list_txtparamIR = _list_txtparamIR
        self.list_valueparamIR = _list_valueparamIR
        self.list_unitsparams = _list_unitsparams

        self.nb_of_materials = nb_of_materials
        self.dict_param_list

        # GUI widgets

        nbmaterialtxt = wx.StaticText(self.panel, -1, "Nb Material")
        self.nbmaterialctrl = wx.SpinCtrl(self.panel, -1, "1", min=1, max=15)

        self.Bind(wx.EVT_SPINCTRL, self.OnChangeNbMaterial, self.nbmaterialctrl)

        self.nb = wx.Notebook(self.panel, -1, style=0)

        self.InitTabs()

        # TODO bind with self.Show_Image
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnTabChange_PeakSearchMethod)

        btnSave = wx.Button(self.panel, -1, "ACCEPT & SAVE", size=(-1, 50))
        btnSave.Bind(wx.EVT_BUTTON, self.OnSaveConfigFile)

        btnLoad = wx.Button(self.panel, -1, "Load File", size=(-1, 50))
        btnLoad.Bind(wx.EVT_BUTTON, self.OnLoad)

        btnLoadDefault = wx.Button(self.panel, -1, "Reset", size=(-1, 50))
        btnLoadDefault.Bind(wx.EVT_BUTTON, self.OnLoadDefault)

        btnQuit = wx.Button(self.panel, -1, "Cancel", size=(-1, 50))
        btnQuit.Bind(wx.EVT_BUTTON, self.OnQuit)

        #widgets layout------
        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        hbox0.Add(nbmaterialtxt, 0, wx.EXPAND)
        hbox0.Add(self.nbmaterialctrl, 0, wx.EXPAND)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(btnSave, 0, wx.EXPAND)
        hbox.Add(btnLoad, 0, wx.EXPAND)
        hbox.Add(btnLoadDefault, 0, wx.EXPAND)
        hbox.Add(btnQuit, 0, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(hbox0, 0, wx.EXPAND)
        vbox.AddSpacer(5)
        vbox.Add(self.nb, 1, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND)

        self.panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

    def OnChangeNbMaterial(self, evt):
        self.nb_of_materials_new = int(self.nbmaterialctrl.GetValue())

        print("use now %d materials" % self.nb_of_materials_new)
        self.AddDeleteTabs()

    def InitTabs(self):
        self.materialpages_list = []
        for material_index in range(self.nb_of_materials):
            pagematerial = PageMaterialPanel(self.nb)
            self.materialpages_list.append(pagematerial)
            self.nb.AddPage(pagematerial, "Material %d" % material_index)

    def AddDeleteTabs(self):
        diff = self.nb_of_materials_new - self.nb_of_materials
        if diff >= 1:
            for material_index in range(diff):
                abs_material_index = material_index + self.nb_of_materials
                pagematerial = PageMaterialPanel(self.nb)
                self.materialpages_list.append(pagematerial)
                self.nb.AddPage(pagematerial, "Material %d" % abs_material_index)

            self.nb_of_materials = self.nb_of_materials_new

        elif diff <= -1:
            #             print "i kill!"
            #             tabs_to_kill = range(-diff)
            nb_tabs = len(self.materialpages_list)
            for tabindex in range(-diff):
                self.nb.DeletePage(nb_tabs - (tabindex + 1))
                self.materialpages_list.pop(nb_tabs - (tabindex + 1))

            self.nb_of_materials = len(self.materialpages_list)
    #             print dir(self.nb)

    def OnTabChange_PeakSearchMethod(self, evt):
        pass

    def OnSaveConfigFile(self, evt):

        if not self.getParams():
            return

        wcd = "IndexRefine Param.(*.irp)|*.irp|All files(*)|*"

        defaultdir = self.parent.list_txtctrl[0].GetValue()
        if not os.path.isdir(defaultdir):
            defaultdir = os.getcwd()

        file = wx.FileDialog(
            self,
            "Save irp File",
            defaultDir=defaultdir,
            wildcard=wcd,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if file.ShowModal() == wx.ID_OK:

            outputfile = file.GetPath()

            outputfilename = ISS.saveIndexRefineConfigFile(
                self.dict_param_list, outputfilename=outputfile
            )

            # outputfilename has .irp extension
            self.parent.list_txtctrl[12].SetValue(outputfilename)

        self.Close()

    def OnLoad(self, evt):
        irpfile = wx.FileDialog(
            self,
            "Open Index Refine Parameters",
            wildcard="IndexRefine Param.(*.irp)|*.irp|All files(*)|*",
        )
        if irpfile.ShowModal() == wx.ID_OK:

            fileirp = irpfile.GetPath()

            self.dict_param_list = ISS.readIndexRefineConfigFile(fileirp)

            print("dict_param_list in OnLoad()", self.dict_param_list)

            self.nb.DeleteAllPages()

            self.nb_of_materials = len(self.dict_param_list)

            print("nb_materials loaded", self.nb_of_materials)

            self.InitTabs()

            for k_page, pagematerial in enumerate(self.materialpages_list):
                for kk, _key in enumerate(LIST_TXTPARAMS):
                    if _key in list(self.dict_param_list[k_page].keys()):
                        pagematerial.list_valueparamIR[kk] = self.dict_param_list[
                            k_page
                        ][_key]

                pagematerial.setParams()

    def setParams(self):
        for materialpage in self.materialpages_list:
            materialpage.setParams()

    def getParams(self):
        self.dict_param_list = []
        flag = True
        for materialindex, materialpage in enumerate(self.materialpages_list):
            flag = flag and materialpage.getParams()

            self.dict_param_list.append(materialpage.dict_param_list)

        print("self.dict_param_list", self.dict_param_list)

        return flag

    def OnLoadDefault(self, evt):
        print("reset")
        self.list_valueparamIR = LIST_VALUESPARAMS
        self.setParams()
            #             print "i add"

    def OnQuit(self, evt):
        self.Close()


class PageMaterialPanel(wx.Panel):
    """
    class for GUI to have widgets corresponding to indexation parameters of 1 material or element
    """

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.granparent = parent.GetParent().GetParent()

        self.dict_param_list = {}

        (
            self.list_txtparamIR,
            self.list_valueparamIR,
            self.list_unitsparams,
        ) = self.granparent.listParameters

        print("self.granparent.listParameters")
        print(self.granparent.listParameters)

        nbrows = len(self.list_txtparamIR)

        #        wx.Frame.__init__(self, None, -1, title,
        #                          wx.DefaultPosition, wx.Size(500, nbrows * 40))

        self.tooltips()
        if WXPYTHON4:
            grid = wx.FlexGridSizer(3, 10, 10)
        else:
            grid = wx.FlexGridSizer(nbrows, 3, 1, 1)

        grid.SetFlexibleDirection(wx.HORIZONTAL)

        self.list_txtctrl = []
        # set value according to list_txtparamIR, list_valueparamIR, list_unitsparams
        for kk, elem in enumerate(self.list_txtparamIR):
            wxtxt = wx.StaticText(self, -1, elem)
            grid.Add(wxtxt)
            self.txtctrl = wx.TextCtrl(self, -1, "", size=(100, 25))
            self.list_txtctrl.append(self.txtctrl)

            grid.Add(self.txtctrl)
            grid.Add(wx.StaticText(self, -1, self.list_unitsparams[kk]))

            wxtxt.SetToolTipString(self.tips_dict[kk])
            self.txtctrl.SetToolTipString(self.tips_dict[kk])

        self.setParams()

        self.SetSizer(grid, wx.EXPAND)

        # tooltips

    def tooltips(self):
        self.tips_dict = {}
        for kk, elem in enumerate(self.list_txtparamIR):
            self.tips_dict[kk] = "%s : " % elem

        self.tips_dict[0] += "Material, element or structure Label"
        self.tips_dict[1] += "Number of grains for the given material to try to index"
        self.tips_dict[2] += "Minimum energy bandpass (keV)"
        self.tips_dict[3] += "Maximum energy bandpass (keV)"

        self.tips_dict[
            4
        ] += "Minimum matching rate to stop the loop over mutual spots angular distance recognition."
        self.tips_dict[
            4
        ] += "Then the corresponding unit cell orientation matrix and strain will be refined.\n"
        self.tips_dict[
            4
        ] += "100.0 implies that all mutual spots distances will be checked"

        self.tips_dict[
            5
        ] += "Angular tolerance (deg) for looking up a distance in the reference distances database (LUT)"
        self.tips_dict[
            6
        ] += "Maximum angle separating two spots forming a pair (1 exp. and 1 theo.) to compute the number of spots matches (or matching rate)."
        self.tips_dict[
            7
        ] += "Number of the most intense spots from which angular distances with central spots will be tested for recognition."
        self.tips_dict[
            8
        ] += "Minimum number of spots matches for a indexation solution to be stored"

        self.tips_dict[
            9
        ] += "Central spot or list of spot indices: e.g.\n[1,2,3,5]\n[0]\n5\n8:20\n:10"
        self.tips_dict[
            10
        ] += "if not False, minimum lattice spacing (for cubic structure) of spot to be simulated for matching with experimental spots data."
        self.tips_dict[
            11
        ] += "highest miller indices order to calculate the reference mutual angular distances table (LUT)."
        self.tips_dict[
            12
        ] += "if not None, [h,k,l] miller indices of all central spots used for recognition."

        self.tips_dict[
            13
        ] += "maximum number of spots to be used for refined (first ones in the list)"
        self.tips_dict[
            14
        ] += "True/False to use or not the experimental spot intensities as a weight in the refinement (minimization of distances between matched spots)"
        self.tips_dict[
            15
        ] += "list of angular tolerance used at each step after the refinement procedure to link exp. and modeled spots"

    def hascorrectvalue(self, kk, val):

        flag = True

        if kk in (1, 11):
            try:
                v = int(val)
            except ValueError:
                wx.MessageBox(
                    "Error in Index_Refine.py hascorrectvalue().\nWrong type %s! Must be integer"
                    % self.list_txtparamIR[kk],
                    "Error",
                )
                flag = False

        if kk == 12:
            if val == "None":
                return True
            try:
                vals = val.split(",")
                print("vals", vals)
                h, k, l = vals
            except:
                wx.MessageBox(
                    "Error in Index_Refine.py hascorrectvalue().\nWrong type %s! Must be list of 3 integers"
                    % self.list_txtparamIR[kk],
                    "Error",
                )
                flag = False

        return flag

    def setParams(self):
        for kk, elem in enumerate(self.list_txtparamIR):
            self.list_txtctrl[kk].SetValue(str(self.list_valueparamIR[kk]))

    def getParams(self):
        self.dict_param_list = {}
        flag = True
        for kk, elem in enumerate(self.list_txtparamIR):
            val = self.list_txtctrl[kk].GetValue()

            print("kk,val", kk, val)

            if not self.hascorrectvalue(kk, val):
                flag = False
                break

            self.list_valueparamIR[kk] = val

            self.dict_param_list[self.list_txtparamIR[kk]] = val

        print("self.dict_param_list", self.dict_param_list)

        return flag


class Stock_parameters_IndexRefine:
    """ class to stock parameters
    """

    def __init__(self, list_txtparamIR, _list_valueparamIR):
        self.list_txtparamIR = list_txtparamIR
        self.list_valueparamIR = _list_valueparamIR


class MainFrame_indexrefine(wx.Frame):
    """
    main class providing a board from which to launch indexation and refinement of data
    """

    def __init__(self, parent, _id, title, _initialparameters, objet_IR):
        wx.Frame.__init__(
            self, parent, _id, title, wx.DefaultPosition, wx.Size(900, 650)
        )

        self.initialparameters = _initialparameters

        self.parent = parent
        if WXPYTHON4:
            grid = wx.FlexGridSizer(3, 10, 10)
        else:
            grid = wx.FlexGridSizer(14, 3, 10, 10)

        grid.SetFlexibleDirection(wx.HORIZONTAL)
        self.panel = wx.Panel(self)

        dict_tooltip = DICT_TOOLTIP
        keys_list_dicttooltip = list(DICT_TOOLTIP.keys())

        self.list_txtctrl = []

        for kk, txt_elem in enumerate(objet_IR.list_txtparamIR):
            txt = wx.StaticText(self.panel, -1, "     %s" % txt_elem)
            grid.Add(txt)

            print("kk,txt_elem", kk, txt_elem)
            print("objet_IR.list_valueparamIR[kk]", objet_IR.list_valueparamIR[kk])

            self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))
            self.txtctrl.SetValue(str(objet_IR.list_valueparamIR[kk]))
            self.list_txtctrl.append(self.txtctrl)
            grid.Add(self.txtctrl)

            if txt_elem in keys_list_dicttooltip:

                txt.SetToolTipString(dict_tooltip[txt_elem])
                self.txtctrl.SetToolTipString(dict_tooltip[txt_elem])

            if kk in (0, 1, 2, 3, 9, 10, 12):
                btnbrowse = wx.Button(self.panel, -1, "Browse")
                grid.Add(btnbrowse)
                if kk == 0:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathdat)
                    btnbrowse.SetToolTipString("Select Folder containing .dat files")
                elif kk == 1:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout_cor)
                    btnbrowse.SetToolTipString(
                        "Select Folder containing (results or input) .cor files"
                    )
                elif kk == 2:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout_fit)
                    btnbrowse.SetToolTipString(
                        "Select Folder containing indexed peaks list results .fit files"
                    )
                elif kk == 3:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filedat)
                    btnbrowse.SetToolTipString(
                        "Select one .dat or .cor file to get (and guess) the generic prefix of all peaks list filenames"
                    )
                elif kk == 9:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filedet)
                    btnbrowse.SetToolTipString(
                        "Select detector calibration parameters .det file"
                    )
                elif kk == 10:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_matsfile)
                    btnbrowse.SetToolTipString(
                        "Select list of guessed UB matrices or check orientation parameters (.mat,.mats or .ubs) file"
                    )
                elif kk == 12:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_irpfile)
                    btnbrowse.SetToolTipString("Select index and refine .irp file")
            else:
                nothing = wx.StaticText(self.panel, -1, "")
                grid.Add(nothing)

        Createcfgbtn = wx.Button(self.panel, -1, "Create .irp file")
        Createcfgbtn.Bind(wx.EVT_BUTTON, self.OnCreateIRP)

        self.previousreschk = wx.CheckBox(self.panel, -1, "Index n using n-1 results ")
        self.previousreschk.SetValue(True)

        grid.Add(Createcfgbtn)
        grid.Add(self.previousreschk)

        # multiprocessing handling
        txt_cpus = wx.StaticText(self.panel, -1, "nb CPU(s)")
        self.txtctrl_cpus = wx.TextCtrl(self.panel, -1, "1")

        self.chck_renanalyse = wx.CheckBox(
            self.panel, -1, "(Re)Analyse (overwrite results)"
        )
        self.chck_renanalyse.SetValue(True)

        self.updatefitfiles = wx.CheckBox(self.panel, -1, "Update preexisting results")
        self.updatefitfiles.SetValue(False)

        #          bouton STARTdfd
        btnStart = wx.Button(
            self.panel, -1, "START INDEX and REFINE (files .fit in OutPutFolder)", size=(-1, 60))
        btnStart.Bind(wx.EVT_BUTTON, self.OnStart)

        #widgets layout-----
        hfinal = wx.BoxSizer(wx.HORIZONTAL)
        hfinal.Add(txt_cpus, 0)
        hfinal.Add(self.txtctrl_cpus, 0)
        hfinal.AddSpacer(30)
        hfinal.Add(self.chck_renanalyse, 0)
        hfinal.Add(self.updatefitfiles, 0, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        vbox.Add(hfinal, 0, wx.EXPAND)
        vbox.Add(btnStart, 0, wx.EXPAND)

        self.panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

        # tooltips
        sentencereanaylse = "If not checked, indexation will be performed for images for which corresponding .res file is missing.\n"
        sentencereanaylse += "If checked, indexation will be (re)performed and overwrite all .fit and .res files (if any)."
        self.chck_renanalyse.SetToolTipString(sentencereanaylse)

        sentenceupdate = (
            "If checked, indexation and refinement will be performed again\n"
        )
        sentenceupdate += "by checking first the orientation already existing in corresponding fit file\n"
        sentenceupdate += '(matching rate of checked matrix must be higher than the above "Minimum Matching Rate").'
        self.updatefitfiles.SetToolTipString(sentenceupdate)

        Createcfgbtn.SetToolTipString(
            "Create .irp file containing parameters to index & refine peaks list"
        )
        self.previousreschk.SetToolTipString(
            "If checked, prior to indexation from scratch (according to .irp file)first if  orientation matrix of image n-1 is a good guess for indexing the current image n"
        )
        tipcpus = "nb of cores to use to index&refine all peaks list files"
        txt_cpus.SetToolTipString(tipcpus)
        self.txtctrl_cpus.SetToolTipString(tipcpus)

        btnStart.SetToolTipString("Start indexing & refining all the peaks list files")

    def OnbtnBrowse_filepathdat(self, event):
        folder = wx.DirDialog(self, "Select folder for peaklist files")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            self.list_txtctrl[0].SetValue(abspath)

            projectpath = abspath

            if "datfiles" in abspath:
                projectpath, lastpath = os.path.split(abspath)

            self.list_txtctrl[2].SetValue(os.path.join(projectpath, "fitfiles"))
            self.list_txtctrl[1].SetValue(os.path.join(projectpath, "corfiles"))

    def OnbtnBrowse_filepathout_fit(self, event):
        folder = wx.DirDialog(
            self, "Select folder for indexed and refined peaklist .fit files"
        )
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[2].SetValue(folder.GetPath())

    def OnbtnBrowse_filepathout_cor(self, event):
        folder = wx.DirDialog(self, "Select folder for peaklist .cor files")
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[1].SetValue(folder.GetPath())

    def OnbtnBrowse_filedat(self, event):
        folder = wx.FileDialog(
            self,
            "Select Peaklist File .dat (or .cor)",
            wildcard="PeakList (*.dat)|*.dat|PeakList (*.cor)|*.cor|All files(*)|*",
        )
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            #             print "folder.GetPath()", abspath

            filename = os.path.split(abspath)[-1]
            #             print "filename", filename
            intension, extension = filename.split(".")

            self.list_txtctrl[4].SetValue("." + extension)

            nbdigits = int(self.list_txtctrl[5].GetValue())
            self.list_txtctrl[3].SetValue(intension[:-nbdigits])

    def OnbtnBrowse_filedet(self, event):
        folder = wx.FileDialog(
            self,
            "Select CCD Calibration Parameters file .det",
            wildcard="Detector Parameters File (*.det)|*.det|All files(*)|*",
        )
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[9].SetValue(folder.GetPath())

    #     def OnbtnBrowse_fileReferenceCalibrationdat(self, event):
    #         folder = wx.FileDialog(self, "Select Peaklist File .dat of Calibration Reference",
    #                                wildcard='PeakList (*.dat)|*.dat|All files(*)|*')
    #         if folder.ShowModal() == wx.ID_OK:
    #
    #             self.list_txtctrl[10].SetValue(folder.GetPath())

    def OnbtnBrowse_matsfile(self, evt):
        print("OnbtnBrowse_matsfile")

        matsfile = wx.FileDialog(
            self,
            "Select Guessed Matrices File or check orientation parameters file (.ubs)",
            wildcard="Guessed Matrices (*.mat;*.mats;*.ubs)|*.mat;*.mats;*.ubs|All files(*)|*",
        )
        if matsfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[10].SetValue(matsfile.GetPath())

    def OnbtnBrowse_irpfile(self, evt):
        print("OnbtnBrowse_irpfile")

        irpfile = wx.FileDialog(
            self,
            "Select Index Refine Parameters File",
            wildcard="Index Refine Param.(*.irp)|*.irp|All files(*)|*",
        )
        if irpfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[12].SetValue(irpfile.GetPath())

    def OnCreateIRP(self, event):
        print("OnCreateIRP")

        fileirp = str(self.list_txtctrl[12].GetValue())
        print("fileirp", fileirp)
        if not os.path.exists(fileirp):
            dict_param = {}
        else:
            #             raise IOError, "parameters file irp not implemented yet"
            dict_param_list = ISS.readIndexRefineConfigFile(fileirp)
            print("dict_param_list", dict_param_list)
            if dict_param_list is not None:
                # TODO: read multimaterial irp file  see ISS
                dict_param = dict_param_list[0]
            else:
                dict_param = {}

        listvals = [] * len(LIST_VALUESPARAMS)
        for kk, _key in enumerate(LIST_TXTPARAMS):
            if _key in list(dict_param.keys()):
                listvals.append(dict_param[_key])
            else:
                listvals.append(None)

        #        print "listvals", listvals

        IRPboard = IndexRefineParameters(
            self,
            -1,
            "Index and Refine Parameters",
            (LIST_TXTPARAMS, LIST_VALUESPARAMS, LIST_UNITSPARAMS),
        )
        IRPboard.Show(True)

    def calcCalibrationfitFile(self):
        """
        produce a .fit file of the reference crystal used for CCD calibration parameters
        """
        # needs to remove bad shaped spots for calibration refinement
        if self.initialparameters["filter_peaks_index_refine_calib"]:
            filedet = self.list_txtctrl[
                9
            ].GetValue()  # to guess the initial CCD parameters
            #             referencefiledat_init = self.list_txtctrl[10].GetValue()
            referencefiledat_init = None

            if referencefiledat_init is not None:
                MAXPIXDEV_CALIBRATIONREFINEMENT = self.initialparameters[
                    "maxpixdev_filter_peaks_index_refine_calib"
                ]
                self.referencefiledat_purged = filter_peaks(
                    referencefiledat_init, maxpixdev=MAXPIXDEV_CALIBRATIONREFINEMENT
                )
                (
                    calib_fitfilename,
                    npeaks_LT,
                    pixdev_LT,
                ) = index_refine_calib_one_image(
                    self.referencefiledat_purged, filedet=filedet
                )
            else:
                raise ValueError(
                    "filter_peaks_index_refine_calib=1 without .dat file of peaks used for calibration is no more used in Index_refine()"
                )

        else:
            (calib_fitfilename, npeaks_LT, pixdev_LT) = index_refine_calib_one_image(
                self.referencefiledat_purged, filedet=filedet
            )

        self.initialparameters["CCDcalibrationReference .fit file"] = calib_fitfilename
        print("CCDcalibrationReference .fit file : %s" % calib_fitfilename)

    def fitFolderExists(self):
        fitfolder = str(self.list_txtctrl[2].GetValue())

        print("fitfolder in fitFolderExists", fitfolder)
        if not os.path.isdir(fitfolder):
            try:
                os.mkdir(fitfolder)
                return True
            except IOError:
                wx.MessageBox(
                    "Can not create %s to contain peaks list .fit files !" % fitfolder,
                    "Error",
                )
                return False

        return True

    def corFolderExists(self):
        corfolder = str(self.list_txtctrl[1].GetValue())
        if not os.path.isdir(corfolder):
            try:
                os.mkdir(corfolder)
                return True
            except IOError:
                wx.MessageBox(
                    "Can not create %s to contain peaks list .cor files !" % corfolder,
                    "Error",
                )
                return False

        return True

    def datFolderExists(self):
        datfolder = str(self.list_txtctrl[0].GetValue())
        if not os.path.isdir(datfolder):
            wx.MessageBox(
                "Can not see %s containing peak list .dat files !" % datfolder, "Error"
            )
            return False

        return True

    def PeaklistCorFile_FolderExists(self):
        corfolder = self.list_txtctrl[1].GetValue()

        if not os.path.isdir(corfolder):
            wx.MessageBox(
                "Can not find %s containing peaklist .cor file!" % corfolder, "Error"
            )
            return False

        return True

    def OnStart(self, event):
        print("OnStart in index_Refine.py MainFrame class")

        # read .irp file ---------------------------
        fileirp = self.list_txtctrl[12].GetValue()
        print("read index refine parameters in:")

        if not os.path.exists(fileirp):
            wx.MessageBox(
                "Index_refine config file %s does not exist!\n" % fileirp, "Error"
            )
            return

        try:
            self.dict_param_list = ISS.readIndexRefineConfigFile(fileirp)
        except IndexError:
            wx.MessageBox(
                "Can't read properly index_refine config file %s\n" % fileirp, "Error"
            )
            return

        print("dict_param_list in OnStart", self.dict_param_list)

        self.nb_of_materials = len(self.dict_param_list)

        print("nb_materials loaded", self.nb_of_materials)

        if (
            not self.fitFolderExists()
            or not self.corFolderExists()
            or not self.datFolderExists()
        ):
            print("some folder missing ")
            return

        fileprefix = self.list_txtctrl[3].GetValue()
        filesuffix = self.list_txtctrl[4].GetValue()

        nbdigits_filename = int(self.list_txtctrl[5].GetValue())

        if 0:  # odile's way
            # refine calibration
            self.calcCalibrationfitFile()
            filefitcalib = self.initialparameters["CCDcalibrationReference .fit file"]
            # TODO correct multigrain to use os.path.join
            filepathout = self.list_txtctrl[2].GetValue() + "/"

            # TODO correct multigrain to use os.path.join
            filepathdat = self.list_txtctrl[0].GetValue() + "/"

            indimg = list(
                range(
                    int(self.list_txtctrl[6].GetValue()),
                    int(self.list_txtctrl[7].GetValue()) + 1,
                    int(self.list_txtctrl[8].GetValue()),
                )
            )

            serial_index_refine_multigrain(
                filepathdat, fileprefix, indimg, filesuffix, filefitcalib, filepathout
            )

            serial_index_refine_multigrain_v2(
                filepathdat, fileprefix, indimg, filesuffix, filefitcalib, filepathout
            )

        if 1:  # Lauetools ISS way

            filepathdat = self.list_txtctrl[0].GetValue()
            filepathcor = self.list_txtctrl[1].GetValue()
            filepathout = self.list_txtctrl[2].GetValue()

            print("filepathcor", filepathcor)
            print("filepathout", filepathout)

            filedet = self.list_txtctrl[9].GetValue()

            # checking if at least one peak list filename with prefix exist
            listfiles = os.listdir(filepathdat)
            #             print "listfiles", listfiles
            nbfiles = len(listfiles)
            print("nb of files", nbfiles)
            if nbfiles == 0:
                wx.MessageBox(
                    "Apparently the folder %s is empty!" % filepathdat, "ERROR"
                )
                return

            indexfile = 0
            FileNotFound = True
            while FileNotFound:
                if listfiles[indexfile].endswith(filesuffix):
                    #                     print listfiles[indexfile]
                    if listfiles[indexfile].startswith(fileprefix):
                        break
                if indexfile == nbfiles - 1:
                    wx.MessageBox(
                        "No peaklist filename %s starting with\n%s\nin folder\n%s"
                        % (filesuffix, fileprefix, filepathdat),
                        "ERROR",
                    )
                    FileNotFound = False
                indexfile += 1

            # at least one file has been found
            if not FileNotFound:
                return

            #             CCDparams, calibmatrix = IOLT.readfile_det(filedet, nbCCDparameters=8)
            
            CCDCalibdict = None
            if filesuffix in ('.dat'):
                CCDCalibdict = IOLT.readCalib_det_file(filedet)

            Index_Refine_Parameters_dict = {}

            #             Index_Refine_Parameters_dict['CCDCalibParameters'] = CCDparams[:5]
            #             Index_Refine_Parameters_dict['pixelsize'] = CCDparams[5]
            #             Index_Refine_Parameters_dict['framedim'] = CCDparams[6:8]
            #             Index_Refine_Parameters_dict['detectordiameter'] = max(CCDparams[6:8]) * CCDparams[5]
            #             Index_Refine_Parameters_dict['kf_direction'] = DEFAULT_KF_DIRECTION

            Index_Refine_Parameters_dict["CCDCalibdict"] = CCDCalibdict
            Index_Refine_Parameters_dict["PeakList Folder"] = filepathdat
            Index_Refine_Parameters_dict["PeakListCor Folder"] = filepathcor
            Index_Refine_Parameters_dict["nbdigits"] = nbdigits_filename
            Index_Refine_Parameters_dict["prefixfilename"] = fileprefix
            Index_Refine_Parameters_dict["suffixfilename"] = filesuffix
            Index_Refine_Parameters_dict["prefixdictResname"] = fileprefix + "_dict_"

            Index_Refine_Parameters_dict["PeakListFit Folder"] = filepathout
            Index_Refine_Parameters_dict["Results Folder"] = filepathout

            Index_Refine_Parameters_dict["dict params list"] = self.dict_param_list

            try:
                startindex = int(self.list_txtctrl[6].GetValue())
                finalindex = int(self.list_txtctrl[7].GetValue())
                stepindex = int(self.list_txtctrl[8].GetValue())
            except:
                wx.MessageBox(
                    "You should enter integer values for images index fields", "ERROR"
                )
                return

            fileindexrange = (startindex, finalindex, stepindex)

            use_previous_results = self.previousreschk.GetValue()
            reanalyse = self.chck_renanalyse.GetValue()
            updatefitfiles = self.updatefitfiles.GetValue()

            # read file containing guessed UB matrix or params to check orientation in .ubs file to check potential matching --------------
            # before doing (maybe long) indexation from scratch
            guessedMatricesFile = str(self.list_txtctrl[10].GetValue())
            print("guessedMatricesFile", guessedMatricesFile)
            # -----------------------------------------------------------------------

            # corresponding minimum matching rate -----------------------------------------------
            MinimumMatchingRate = float(self.list_txtctrl[11].GetValue())
            print(
                "MinimumMatchingRate to avoid starting general indexation is ",
                MinimumMatchingRate,
            )
            if guessedMatricesFile not in ("None", "none"):
                print("Reading general file for guessed UB solutions")

                # read list or single matrix (ces) in GUI field
                if not guessedMatricesFile.endswith(".ubs"):
                    nbguesses, guessedSolutions = IOLT.readListofMatrices(
                        guessedMatricesFile
                    )

                    print("guessedmatrix", guessedSolutions)
                    Index_Refine_Parameters_dict["GuessedUBMatrix"] = guessedSolutions
                # read .ubs file
                else:
                    Index_Refine_Parameters_dict[
                        "CheckOrientation"
                    ] = guessedMatricesFile

                Index_Refine_Parameters_dict[
                    "MinimumMatchingRate"
                ] = MinimumMatchingRate
            elif updatefitfiles:
                Index_Refine_Parameters_dict[
                    "MinimumMatchingRate"
                ] = MinimumMatchingRate
            else:
                # we are sure to be less than that!
                Index_Refine_Parameters_dict["MinimumMatchingRate"] = 101.0
            # ----------------------------------------------------------------------

            if self.parent is not None:
                object_to_set = self.parent  # IR

                print(
                    "object_to_set.initialparameters", object_to_set.initialparameters
                )

                object_to_set.initialparameters[
                    "IndexRefine PeakList Folder"
                ] = filepathout
                object_to_set.initialparameters["file xyz"] = "None"
                object_to_set.initialparameters[
                    "IndexRefine PeakList Prefix"
                ] = fileprefix
                object_to_set.initialparameters["IndexRefine PeakList Suffix"] = ".fit"
                object_to_set.initialparameters["stiffness file"] = None
                object_to_set.initialparameters["Map shape"] = (0, 0)
                object_to_set.initialparameters["fast axis: x or y"] = "x"
                object_to_set.initialparameters["(stepX, stepY) microns"] = (1.0, 1.0)

                object_to_set.initialparameters["startingindex"] = startindex
                object_to_set.initialparameters["finalindex"] = finalindex
                object_to_set.initialparameters["nbdigits"] = nbdigits_filename
                object_to_set.initialparameters["stepindex"] = stepindex

            print("start indexing multifiles")
            NB_MATERIALS = 2

            NB_MATERIALS = len(self.dict_param_list)

            try:
                nb_cpus = int(self.txtctrl_cpus.GetValue())
            except ValueError:
                wx.MessageBox("nb of cpu(s) must be positive integer!", "Error")
                return
            if nb_cpus <= 0:
                wx.MessageBox("nb of cpu(s) must be positive integer!", "Error")
                return

            flagcompleted = True
            if nb_cpus == 1:
                output_index_fileseries_3 = ISS.index_fileseries_3(
                    fileindexrange,
                    Index_Refine_Parameters_dict=Index_Refine_Parameters_dict,
                    saveObject=0,
                    verbose=0,
                    nb_materials=NB_MATERIALS,
                    build_hdf5=True,
                    prefixfortitle=fileprefix,
                    reanalyse=reanalyse,
                    use_previous_results=use_previous_results,
                    updatefitfiles=updatefitfiles,
                    CCDCalibdict=CCDCalibdict,
                )

                if output_index_fileseries_3 is not None:
                    dictRes, outputdict_filename = output_index_fileseries_3
                else:
                    wx.MessageBox(
                        "Indexation and Refinement not completed.\n An error occured during the procedure\n"
                        + "See stdout or terminal window for details.",
                        "INFO",
                    )

            elif nb_cpus > 1:
                print("Using %d processors" % nb_cpus)
                flagcompleted = ISS.indexing_multiprocessing(
                    fileindexrange,
                    dirname_dictRes=filepathout,
                    Index_Refine_Parameters_dict=Index_Refine_Parameters_dict,
                    saveObject=0,
                    verbose=0,
                    nb_materials=NB_MATERIALS,
                    nb_of_cpu=nb_cpus,
                    build_hdf5=True,
                    prefixfortitle=fileprefix,
                    reanalyse=reanalyse,
                    use_previous_results=use_previous_results,
                    updatefitfiles=updatefitfiles,
                    CCDCalibdict=CCDCalibdict,
                )

                print("flagcompleted", flagcompleted)
                if not flagcompleted:
                    print(
                        "\n\n ****** \nIndexation and Refinement not completed\n***********\n\n"
                    )
                wx.MessageBox(
                    "Indexation and Refinement not completed.\n Check the prefixfilename of .dat file! Launch the task with only one CPU",
                    "INFO",
                )

        return


def fill_list_valueparamIR(initialparameters):
    """
    return a list of default value for index_refine board from a dict initialparameters
    """
    list_valueparamIR = [
        initialparameters["PeakList Folder"],
        initialparameters["PeakListCor Folder"],
        initialparameters["IndexRefine PeakList Folder"],
        initialparameters["PeakList Filename Prefix"],
        initialparameters["PeakList Filename Suffix"],
        initialparameters["nbdigits"],
        initialparameters["startingindex"],
        initialparameters["finalindex"],
        initialparameters["stepindex"],
        initialparameters["Detector Calibration File .det"],
        initialparameters["GuessedUBMatrix"],
        initialparameters["MinimumMatchingRate"],
        initialparameters["IndexRefine Parameters File"],
    ]

    return list_valueparamIR


# default values for the fields appearing in the Index_Refine.py GUI
initialparameters = {}



MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "GeGaN")

initialparameters["PeakList Folder"] = os.path.join(MainFolder, "datfiles")
initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
initialparameters["PeakList Filename Prefix"] = "orig_nanox2_400_"
initialparameters["IndexRefine Parameters File"] = os.path.join(MainFolder, "GeGaN.irp")
initialparameters["Detector Calibration File .det"] = os.path.join(
    MainFolder, "calibGe_nanowMARCCD165.det"
)
initialparameters["Detector Calibration File (.dat)"] = os.path.join(
    MainFolder, "nanox2_400_0000_LT_1.dat"
)
initialparameters["PeakList Filename Suffix"] = ".dat"

initialparameters["nbdigits"] = 0
initialparameters["startingindex"] = 0
initialparameters["finalindex"] = 5
initialparameters["stepindex"] = 1

# --- To have a nice calibration file -Odile's way---------
# remove bad shaped peaks to calibrate CCD from reference sample
initialparameters["filter_peaks_index_refine_calib"] = 1
# highest accepted pixdev of fit
initialparameters["maxpixdev_filter_peaks_index_refine_calib"] = 0.7

initialparameters["GuessedUBMatrix"] = "None"
initialparameters["MinimumMatchingRate"] = 4.0


# for local test:
MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "CuSi")
print("MainFolder", MainFolder)
initialparameters["PeakList Folder"] = os.path.join(MainFolder, "corfiles")
initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
initialparameters["PeakList Filename Prefix"] = "SiCustrain"
initialparameters["IndexRefine Parameters File"] = os.path.join(MainFolder, "cusi.irp")
initialparameters["PeakList Filename Suffix"] = ".cor"


# prepare sorted list of values
list_valueparamIR = fill_list_valueparamIR(initialparameters)

if __name__ == "__main__":

    #     if 0:
    #         MainFolder = '/media/data3D/data/2013/July13/MA1724/'
    #
    #         initialparameters['PeakList Folder'] = MainFolder + 'Snsurfscan/datfiles'
    #         initialparameters['IndexRefine PeakList Folder'] = MainFolder + 'Snsurfscan/fitfiles'
    #         initialparameters['PeakListCor Folder'] = MainFolder + 'Snsurfscan/corfiles'
    #         initialparameters['PeakList Filename Prefix'] = 'SnsurfscanBig_'
    #         initialparameters['IndexRefine Parameters File'] = MainFolder + 'Snsurfscan/indexSn.irp'
    #         initialparameters['Detector Calibration File .det'] = MainFolder + 'Gemono/GeMAR_HallJul13.det'
    #         initialparameters['Detector Calibration File (.dat)'] = MainFolder + 'Gemono/Ge_0005_LT_1.dat'

    # -----------------------------------------------------------

    Stock_INDEXREFINE = Stock_parameters_IndexRefine(
        LIST_TXTPARAM_FILE_INDEXREFINE, list_valueparamIR
    )

    print("Stock_INDEXREFINE", Stock_INDEXREFINE.list_txtparamIR)
    print("Stock_INDEXREFINE", Stock_INDEXREFINE.list_valueparamIR)
    IndexRefineSeriesApp = wx.App()
    IndexRefineSeries = MainFrame_indexrefine(
        None, -1, "Index Refine Parameters Board", initialparameters, Stock_INDEXREFINE
    )
    IndexRefineSeries.Show(True)
    IndexRefineSeriesApp.MainLoop()
