# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:23:40 2013

@author: micha

from initially T. Cerba

Revised February 2020
"""
import sys
import os
import math

sys.path.append("..")

# this is for running through ipython
import matplotlib
matplotlib.use("WXAgg")
#------------------------

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        """ modification tooltip fct"""
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

try:
    from multigrain import filter_peaks, index_refine_calib_one_image
except (ImportError, SyntaxError):
    print("Missing modules or functions of multigrain.py. But It does not matter!")

if sys.version_info.major == 3:
    import LaueTools.IOLaueTools as IOLT
    import LaueTools.indexingSpotsSet as ISS
    import LaueTools.dict_LaueTools as dictLT
else:
    import IOLaueTools as IOLT
    import indexingSpotsSet as ISS
    import dict_LaueTools as dictLT


LAUETOOLSFOLDER = dictLT.LAUETOOLSFOLDER
LaueToolsProjectFolder = os.path.abspath(dictLT.LAUETOOLSFOLDER)
print("LaueToolProjectFolder", LaueToolsProjectFolder)

# --- ---- core index and refine parameters   (see .irp file)
LIST_TXTPARAMS = ISS.LIST_OPTIONS_INDEXREFINE[1:]

LIST_VALUESPARAMS = ["Ge", 1, 5, 22, 100.0, 0.5, 0.5, 10, 6, [0],
                    False, 3, None, True, 1000, [1, 1]]

# WARNING when adding parameters above:
# check if field position is correct in def hascorrectvalue(self, kk, val):
# and name does not contain "_"

LIST_UNITSPARAMS = ISS.LIST_OPTIONS_TYPE_INDEXREFINE[1:]

LIST_TXTPARAM_FILE_INDEXREFINE = ["Peak List .dat Folder",
                                "Peak List (Output) .cor Folder",
                                "Peak List (Output) .fit Folder",
                                "PeakList Filename (for prefix)",
                                "PeakList Filename Suffix",
                                "Nbdigits in index filename",
                                "Start Image index or List indices File",
                                "Final Image index",
                                "Image index step",
                                "Detector Calibration File (.det)",
                                "Guessed Matrix(ces) (.mat,.mats,.ubs)",
                                "Minimum Matching Rate",
                                "IndexRefine Parameters File (.irp)",
                                "Selected Peaks from File"]

TIP_IR = ["Folder containing indexed Peaks List .dat files",
    "Folder containing (results) Peaks List .cor files",
    "Folder containing (results) indexed Peaks List .fit files",
    "Prefix for .fit files filename prefix####suffix where #### are digits of file index. Check the value of 'nbdigits' for a correct automatic recognition of prefix from full file name",
    'peak list filename suffix. ".dat" or ".cor"',
    "maximum nb of digits for zero padding of filename index.(e.g. nb of # in prefix####.dat)\n0 for no zero padding.",
    "starting file index (integer) or full path (str) to a file with list of file indices",
    "final file index (integer). Not considered if starting file index is a list of indices or a path to a file",
    "incremental step file index (integer). Not considered if starting file index is a list of indices or a path to a file",
    "full path to detector calibration .det file containing detector plane position and angles parameters\nNot used if PeakList Filename Suffix is .cor",
    "full path to a file (.mat or .mats) containing one or several guessed orientation matrix(ces) or check orientation parameters file (.ubs) to be tested prior to indexation from scratch",
    "Minimum matching rate (nb of matches/ nb of theoritical spots) to consider that test with Guessed Matrix(ces) is positive and then start directly refinement of this solution. (so indexation from scratch is skipped).\n If higher than 100, then test of guessed solution orientation matrix(ces) will be skipped",
    "Full path to .irp file containing index & refine parameters",
    "Full path to a peakslist (.fit or .cor files) to restrict refinement to some particular peaks only",
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
        self.nb_of_materials_new = None
        self.dict_param_list = []

        # GUI widgets----------------
        nbmaterialtxt = wx.StaticText(self.panel, -1, "Nb Material")
        self.nbmaterialctrl = wx.SpinCtrl(self.panel, -1, "1", min=1, max=15)

        self.Bind(wx.EVT_SPINCTRL, self.OnChangeNbMaterial, self.nbmaterialctrl)

        self.nb = wx.Notebook(self.panel, -1, style=0)

        self.InitTabs()

        # TODO bind with self.Show_Image
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnTabChange)

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

    def OnChangeNbMaterial(self, _):
        """ call back fct   read nb of materials
        set  self.nb_of_materials_new"""
        self.nb_of_materials_new = int(self.nbmaterialctrl.GetValue())

        print("use now %d materials" % self.nb_of_materials_new)
        self.AddDeleteTabs()

    def InitTabs(self):
        """ set material pages
        set self.materialpages_list  """
        self.materialpages_list = []
        for material_index in range(self.nb_of_materials):
            pagematerial = PageMaterialPanel(self.nb)
            self.materialpages_list.append(pagematerial)
            self.nb.AddPage(pagematerial, "Material %d" % material_index)

    def AddDeleteTabs(self):
        """ add or delete materialpage"""
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

    def OnTabChange(self, evt):
        """ nothing particular implemented  """
        pass

    def OnSaveConfigFile(self, _):
        """ save .irp config file """
        if not self.getParams():
            return

        wcd = "IndexRefine Param.(*.irp)|*.irp|All files(*)|*"

        defaultdir = self.parent.list_txtctrl[0].GetValue()
        if not os.path.isdir(defaultdir):
            defaultdir = os.getcwd()

        file = wx.FileDialog(self, "Save irp File", defaultDir=defaultdir, wildcard=wcd,
                                                        style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if file.ShowModal() == wx.ID_OK:

            outputfile = file.GetPath()

            outputfilename = ISS.saveIndexRefineConfigFile(
                self.dict_param_list, outputfilename=outputfile)

            # outputfilename has .irp extension
            self.parent.list_txtctrl[12].SetValue(outputfilename)

        self.Close()

    def OnLoad(self, _):
        """read irp.file containing indexation and refinement parameters
        """
        irpfile = wx.FileDialog(self, "Open Index Refine Parameters",
                                        wildcard="IndexRefine Param.(*.irp)|*.irp|All files(*)|*")
        if irpfile.ShowModal() == wx.ID_OK:

            fileirp = irpfile.GetPath()

            self.dict_param_list = ISS.readIndexRefineConfigFile(fileirp)

            print("dict_param_list in OnLoad()", self.dict_param_list)

            self.nb.DeleteAllPages()

            self.nb_of_materials = len(self.dict_param_list)

            print("nb_materials loaded", self.nb_of_materials)

            self.InitTabs()

            for k_page, pagematerialpanel in enumerate(self.materialpages_list):
                for kk, _key in enumerate(LIST_TXTPARAMS):
                    if _key in list(self.dict_param_list[k_page].keys()):
                        pagematerialpanel.list_valueparamIR[kk] = self.dict_param_list[k_page][_key]

                pagematerialpanel.setParams()

    def setParams(self):
        """call materialpage.setParams() for all materialpage
        """
        for materialpage in self.materialpages_list:
            materialpage.setParams()

    def getParams(self):
        """set all .irp parameters in self.dict_param_list """
        self.dict_param_list = []
        flag = True
        for _, materialpage in enumerate(self.materialpages_list):
            flag = flag and materialpage.getParams()

            self.dict_param_list.append(materialpage.dict_param_list)

        print("self.dict_param_list", self.dict_param_list)

        return flag

    def OnLoadDefault(self, _):
        """ reset .irp parameters to default LIST_VALUESPARAMS """
        print("reset")
        self.list_valueparamIR = LIST_VALUESPARAMS
        self.setParams()

    def OnQuit(self, _):
        """ quit """
        self.Close()


class PageMaterialPanel(wx.Panel):
    """
    class for GUI to have widgets corresponding to indexation parameters of 1 material or element
    """

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.granparent = parent.GetParent().GetParent()

        self.dict_param_list = {}

        (self.list_txtparamIR,
            self.list_valueparamIR,
            self.list_unitsparams) = self.granparent.listParameters

        print("self.granparent.listParameters")
        print(self.granparent.listParameters)

        nbrows = len(self.list_txtparamIR)

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

    def tooltips(self):
        """ tooltips
        """
        self.tips_dict = {}
        for kk, elem in enumerate(self.list_txtparamIR):
            self.tips_dict[kk] = "%s : " % elem

        self.tips_dict[0] += "Material, element or structure Label"
        self.tips_dict[1] += "Number of grains for the given material to try to index"
        self.tips_dict[2] += "Minimum energy bandpass (keV)"
        self.tips_dict[3] += "Maximum energy bandpass (keV)"

        self.tips_dict[4] += "Minimum matching rate to stop the loop over mutual spots angular "
        "distance recognition."
        self.tips_dict[4] += "Then the corresponding unit cell orientation matrix and "
        "strain will be refined.\n"
        self.tips_dict[4] += "100.0 implies that all mutual spots distances will be checked"

        self.tips_dict[5] += "Angular tolerance (deg) for looking up a distance "
        "in the reference distances database (LUT)"
        self.tips_dict[6] += "Maximum angle separating two spots forming a pair "
        "(1 exp. and 1 theo.) to compute the number of spots matches (or matching rate)."
        self.tips_dict[7] += "Number of the most intense spots from which angular distances "
        "with central spots will be tested for recognition."
        self.tips_dict[8] += "Minimum number of spots matches for a indexation solution to be stored"

        self.tips_dict[9] += "Central spot or list of spot indices: "
        "e.g.\n[1,2,3,5]\n[0]\n5\n8:20\n:10"
        self.tips_dict[10] += "if not False, minimum lattice spacing (for cubic structure) "
        "of spot to be simulated for matching with experimental spots data."
        self.tips_dict[11] += "highest miller indices order to calculate "
        "the reference mutual angular distances table (LUT)."
        self.tips_dict[12] += "if not None, [h,k,l] miller indices of "
        "all central spots used for recognition."
        self.tips_dict[13] += "maximum number of spots to be used for refined (first ones in the list)"
        self.tips_dict[14] += "True/False to use or not the experimental spot intensities "
        "as a weight in the refinement (minimization of distances between matched spots)"
        self.tips_dict[15] += "list of angular tolerance used at each step "
        "after the refinement procedure to link exp. and modeled spots"

    def hascorrectvalue(self, kk, val):
        """ return boolean depending on type of velue val entered in self.list_txtparamIR[kk]"""
        flag = True

        if kk in (1, 11):
            try:
                _ = int(val)
            except ValueError:
                wx.MessageBox("Error in Index_Refine.py hascorrectvalue().\nWrong type %s! Must be integer"
                    % self.list_txtparamIR[kk], "Error")
                flag = False

        if kk == 12:
            if val == "None":
                return True
            try:
                vals = val.split(",")
                print("vals", vals)
                # h, k, l = vals
            except:
                wx.MessageBox("Error in Index_Refine.py hascorrectvalue().\nWrong type %s! Must "
                    "be list of 3 integers" % self.list_txtparamIR[kk], "Error")
                flag = False

        return flag

    def setParams(self):
        """ set all self.list_txtctrl with values of self.list_valueparamIR """
        for kk, _ in enumerate(self.list_txtparamIR):
            self.list_txtctrl[kk].SetValue(str(self.list_valueparamIR[kk]))

    def getParams(self):
        """
        read parameters from text controllers self.list_txtctrl
        and set self.dict_param_list

        :return: boolean
        """
        self.dict_param_list = {}
        flag = True
        for kk, _ in enumerate(self.list_txtparamIR):
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
        print('entering MainFrame_indexrefine')
        wx.Frame.__init__(self, parent, _id, title, size=(900, 650))

        self.initialparameters = _initialparameters
        self.parent = parent
        self.dict_param_list = {}
        self.nb_of_materials = 1

        #---  widgets -----------------------------
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

            # print("kk,txt_elem", kk, txt_elem)
            # print("objet_IR.list_valueparamIR[kk]", objet_IR.list_valueparamIR[kk])

            self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))
            self.txtctrl.SetValue(str(objet_IR.list_valueparamIR[kk]))
            self.list_txtctrl.append(self.txtctrl)
            grid.Add(self.txtctrl)

            if txt_elem in keys_list_dicttooltip:

                txt.SetToolTipString(dict_tooltip[txt_elem])
                self.txtctrl.SetToolTipString(dict_tooltip[txt_elem])

            if kk in (0, 1, 2, 3, 6, 9, 10, 12, 13):
                btnbrowse = wx.Button(self.panel, -1, "browse")
                grid.Add(btnbrowse)
                if kk == 0:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathdat)
                    btnbrowse.SetToolTipString("Select Folder containing .dat files")
                elif kk == 1:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout_cor)
                    btnbrowse.SetToolTipString("Select Folder containing (results or input) .cor files")
                elif kk == 2:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout_fit)
                    btnbrowse.SetToolTipString("Select Folder containing indexed peaks list results .fit files")
                elif kk == 3:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filedat)
                    btnbrowse.SetToolTipString("Select one .dat or .cor file to get (and guess) "
                                                "the generic prefix of all peaks list filenames")
                elif kk == 6:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filelistindices)
                    btnbrowse.SetToolTipString("Select a file with list of image indices to be analysed")
                elif kk == 9:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filedet)
                    btnbrowse.SetToolTipString(
                        "Select detector calibration parameters .det file")
                elif kk == 10:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_matsfile)
                    btnbrowse.SetToolTipString("Select list of guessed UB matrices or check "
                    "orientation parameters (.mat,.mats or .ubs) file")
                elif kk == 12:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_irpfile)
                    btnbrowse.SetToolTipString("Select index and refine .irp file")
                elif kk == 13:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_reffitfile)
                    btnbrowse.SetToolTipString("Select .fit file to restrict refinement to those peaks")
            else:
                nothing = wx.StaticText(self.panel, -1, "")
                grid.Add(nothing)

        Createcfgbtn = wx.Button(self.panel, -1, "Create .irp file")
        Createcfgbtn.Bind(wx.EVT_BUTTON, self.OnCreateIRP)

        self.previousreschk = wx.CheckBox(self.panel, -1, "Index n using n-1 results ")
        self.previousreschk.SetValue(True)

        txt_mapshape = wx.StaticText(self.panel, -1, "Map Shape")
        self.txtctrl_mapshape = wx.TextCtrl(self.panel, -1, "(1000,1)")

        self.trackingmode = wx.CheckBox(self.panel, -1, "Tracking mode")
        self.trackingmode.SetValue(False)


        grid.Add(Createcfgbtn)
        grid.Add(self.previousreschk)

        # multiprocessing handling
        txt_cpus = wx.StaticText(self.panel, -1, "nb CPU(s)")
        self.txtctrl_cpus = wx.TextCtrl(self.panel, -1, "1")

        self.chck_renanalyse = wx.CheckBox(self.panel, -1, "(Re)Analyse (overwrite results)")
        self.chck_renanalyse.SetValue(True)

        self.updatefitfiles = wx.CheckBox(self.panel, -1, "Update preexisting results")
        self.updatefitfiles.SetValue(False)

        self.verbosemode = wx.CheckBox(self.panel, -1, "Verbose mode")
        self.verbosemode.SetValue(True)

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
        hfinal.Add(self.verbosemode, 0, wx.EXPAND)

        hmap = wx.BoxSizer(wx.HORIZONTAL)
        hmap.Add(txt_mapshape, 0)
        hmap.Add(self.txtctrl_mapshape, 0)
        hmap.Add(self.trackingmode, 0)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        vbox.Add(hmap, 0)
        vbox.Add(hfinal, 0, wx.EXPAND)
        vbox.Add(btnStart, 0, wx.EXPAND)

        self.panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

        # tooltips
        sentencereanaylse = "If not checked, indexation will be performed for images "
        "for which corresponding .res file is missing.\n"
        sentencereanaylse += "If checked, indexation will be (re)performed and overwrite "
        "all .fit and .res files (if any)."
        self.chck_renanalyse.SetToolTipString(sentencereanaylse)

        sentenceupdate = ("If checked, indexation and refinement will be performed again\n")
        sentenceupdate += "by checking first the orientation already existing in corresponding fit file\n"
        sentenceupdate += '(matching rate of checked matrix must be higher than the above "Minimum Matching Rate").'
        self.updatefitfiles.SetToolTipString(sentenceupdate)

        Createcfgbtn.SetToolTipString(
            "Create .irp file containing parameters to index & refine peaks list")
        self.previousreschk.SetToolTipString("If checked, for indexing the current image n, "
        "first check if orientation matrix of image n-1 is a good guess before starting "
        "an indexation from scratch (according to .irp file). 'Guessed Matrix(ces) must be None'")
        tipcpus = "nb of cores to use to index&refine all peaks list files"
        txt_cpus.SetToolTipString(tipcpus)
        self.txtctrl_cpus.SetToolTipString(tipcpus)

        tipshape = "List of dimensions of raster scan  in two (resp. three) directions for 2D (resp. 3D) map. Example: [41,21] or [15,8,100]. Only used when tracking spots positions with 'Selected Peaks from File' is not None "
        txt_mapshape.SetToolTipString(tipshape)
        self.txtctrl_mapshape.SetToolTipString(tipshape)

        btnStart.SetToolTipString("Start indexing & refining all the peaks list files")

    def OnbtnBrowse_filepathdat(self, _):
        """ select file

        set self.list_txtctrl[0], self.list_txtctrl[1], self.list_txtctrl[2] """
        folder = wx.DirDialog(self, "Select folder for peaklist files")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            self.list_txtctrl[0].SetValue(abspath)

            projectpath = abspath

            if "datfiles" in abspath:
                projectpath, _ = os.path.split(abspath)

            self.list_txtctrl[2].SetValue(os.path.join(projectpath, "fitfiles"))
            self.list_txtctrl[1].SetValue(os.path.join(projectpath, "corfiles"))

    def OnbtnBrowse_filepathout_fit(self, _):
        """ select file
        set self.list_txtctrl[2] """
        folder = wx.DirDialog(
            self, "Select folder for indexed and refined peaklist .fit files")
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[2].SetValue(folder.GetPath())

    def OnbtnBrowse_filepathout_cor(self, _):
        """ select file
        set self.list_txtctrl[1] """
        folder = wx.DirDialog(self, "Select folder for peaklist .cor files")
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[1].SetValue(folder.GetPath())

    def OnbtnBrowse_filelistindices(self, _):
        """ select file where list indices is written
        set self.list_txtctrl[6] file
        set self.list_txtctrl[7] None
        set self.list_txtctrl[8] None"""
        print("OnbtnBrowse_filelistindices")

        listinndfile = wx.FileDialog(self, "Select file with list of image index (integers)",
                wildcard="All files(*)|*")
        if listinndfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[6].SetValue(listinndfile.GetPath())
            self.list_txtctrl[7].SetValue('None')
            self.list_txtctrl[8].SetValue('None')

    def OnbtnBrowse_filedat(self, _):
        """ select file  and deduce prefix filename
        read nb of digits
        set self.list_txtctrl[3], self.list_txtctrl[4] """
        folder = wx.FileDialog(self, "Select Peaklist File .dat (or .cor)",
                        wildcard="PeakList (*.dat)|*.dat|PeakList (*.cor)|*.cor|All files(*)|*")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            filename = os.path.split(abspath)[-1]
            intension, extension = filename.split(".")

            self.list_txtctrl[4].SetValue("." + extension)

            nbdigits = self.getnbdigits()
            self.list_txtctrl[3].SetValue(intension[: -nbdigits])

    def getnbdigits(self):
        """get integer from self.list_txtctrl[5]

        :return: nb of digits
        :rtype: int
        """
        try:
            val = int(self.list_txtctrl[5].GetValue())
            _ = math.sqrt(val)
        except ValueError:
            wx.MessageBox("nb of digits in filename must be a positive integer! Please check the "
                                                                    "corresponding field!", "Info")
        return val

    def OnbtnBrowse_filedet(self, _):
        """ select file of detector geometry calibration
        set self.list_txtctrl[9]"""
        folder = wx.FileDialog(self, "Select CCD Calibration Parameters file .det",
                                wildcard="Detector Parameters File (*.det)|*.det|All files(*)|*")
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[9].SetValue(folder.GetPath())

    def OnbtnBrowse_matsfile(self, _):
        """ get .mats file fullpath and set corresponding txtctrl"""
        print("OnbtnBrowse_matsfile")

        matsfile = wx.FileDialog(self, "Select Guessed Matrices File or check orientation "
                "parameters file (.ubs)",
                wildcard="Guessed Matrices (*.mat;*.mats;*.ubs)|*.mat;*.mats;*.ubs|All files(*)|*")
        if matsfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[10].SetValue(matsfile.GetPath())

    def OnbtnBrowse_irpfile(self, _):
        """ get .irp file fullpath and set corresponding txtctrl"""
        print("OnbtnBrowse_irpfile")

        irpfile = wx.FileDialog(self, "Select Index Refine Parameters File",
                                wildcard="Index Refine Param.(*.irp)|*.irp|All files(*)|*")
        if irpfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[12].SetValue(irpfile.GetPath())

    def OnbtnBrowse_reffitfile(self, _):
        """ get .fit file fullpath and set corresponding txtctrl"""
        print("OnbtnBrowse_reffitfile")

        fitfile = wx.FileDialog(self, "Select peaks list .fit File (i.e. containing h,k,l)",
                                wildcard="indexed and refined peaks list (*.fit)|*.fit|All files(*)|*")
        if fitfile.ShowModal() == wx.ID_OK:

            self.list_txtctrl[13].SetValue(fitfile.GetPath())

    def OnCreateIRP(self, _):
        """ open GUI to set indexing and refinement parameters (irp)"""
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
        for _, _key in enumerate(LIST_TXTPARAMS):
            if _key in list(dict_param.keys()):
                listvals.append(dict_param[_key])
            else:
                listvals.append(None)

        IRPboard = IndexRefineParameters(self, -1, "Index and Refine Parameters",
                                        (LIST_TXTPARAMS, LIST_VALUESPARAMS, LIST_UNITSPARAMS))
        IRPboard.Show(True)

    def calcCalibrationfitFile(self):
        """
        produce a .fit file of the reference crystal used for CCD calibration parameters.

        Need proper version of module multigrain

        .. note:: this function is not called anywhere. OnbtnBrowse_fileReferenceCalibrationdat() is commented in this module
        """
        # needs to remove bad shaped spots for calibration refinement
        if self.initialparameters["filter_peaks_index_refine_calib"]:
            filedet = self.list_txtctrl[9].GetValue()  # to guess the initial CCD parameters
            #             referencefiledat_init = self.list_txtctrl[10].GetValue()
            referencefiledat_init = None

            if referencefiledat_init is not None:
                MAXPIXDEV_CALIBRATIONREFINEMENT = self.initialparameters[
                    "maxpixdev_filter_peaks_index_refine_calib"]
                self.referencefiledat_purged = filter_peaks(referencefiledat_init,
                                                        maxpixdev=MAXPIXDEV_CALIBRATIONREFINEMENT)
                #(calib_fitfilename, npeaks_LT, pixdev_LT,
                calib_fitfilename = index_refine_calib_one_image(self.referencefiledat_purged, filedet=filedet)[0]
            else:
                raise ValueError("filter_peaks_index_refine_calib=1 without .dat file of peaks "
                "used for calibration is no more used in Index_refine()")

        else:
            # (calib_fitfilename, npeaks_LT, pixdev_LT) = index_refine_calib_one_image
            calib_fitfilename = index_refine_calib_one_image(self.referencefiledat_purged, filedet=filedet)[0]

        self.initialparameters["CCDcalibrationReference .fit file"] = calib_fitfilename
        print("CCDcalibrationReference .fit file : %s" % calib_fitfilename)

    def fitFolderExists(self):
        """ check if fitfolder exists"""
        fitfolder = str(self.list_txtctrl[2].GetValue())

        print("fitfolder in fitFolderExists", fitfolder)
        if not os.path.isdir(fitfolder):
            try:
                os.mkdir(fitfolder)
                return True
            except IOError:
                wx.MessageBox("Can not create %s to contain peaks list .fit files !" % fitfolder,
                                                                                        "Error")
                return False

        return True

    def corFolderExists(self):
        """ check if corfolder exists"""
        corfolder = str(self.list_txtctrl[1].GetValue())
        if not os.path.isdir(corfolder):
            try:
                os.mkdir(corfolder)
                return True
            except IOError:
                wx.MessageBox(
                    "Can not create %s to contain peaks list .cor files !" % corfolder, "Error")
                return False

        return True

    def datFolderExists(self):
        """ check if datfolder exists"""
        datfolder = str(self.list_txtctrl[0].GetValue())
        if not os.path.isdir(datfolder):
            wx.MessageBox("Can not see %s containing peak list .dat files !" % datfolder, "Error")
            return False

        return True

    def readmapshape(self):
        """get mapshape from self.txtctrl_mapshape

        :return: tuple of integers, None if troubles
        """
        mapshapestr = self.txtctrl_mapshape.GetValue()
        errormapshape = False
        if mapshapestr in ('None', "None",):
            errormapshape = True
        else:
            mapshape = IOLT.readStringOfIterable(mapshapestr)
            if isinstance(mapshape, str):
                errormapshape = True
            if not isinstance(mapshape[0], int) or not isinstance(mapshape[1], int):
                errormapshape = True
        if errormapshape:
            return None
        else:
            return mapshape

    def OnStart(self, _):
        """
        Start indexation and refinement of a series of files.

        read all self.list_txtctrl[3] and GUI widgets
        """
        print("OnStart in index_Refine.py MainFrame class")

        #-------- field 12  :  read .irp file ---------------------------
        fileirp = self.list_txtctrl[12].GetValue()
        print("read index refine parameters in:")

        if not os.path.exists(fileirp):
            wx.MessageBox("Index_refine config file %s does not exist!\n" % fileirp, "Error")
            return

        try:
            self.dict_param_list = ISS.readIndexRefineConfigFile(fileirp)
        except IndexError:
            wx.MessageBox("Can't read properly index_refine config file %s\n" % fileirp, "Error")
            return

        print("dict_param_list in OnStart", self.dict_param_list)

        self.nb_of_materials = len(self.dict_param_list)

        print("nb_materials loaded", self.nb_of_materials)

        if (not self.fitFolderExists() or not self.corFolderExists() or not self.datFolderExists()):
            print("some folder missing ")
            return
        #-------  field 3 & 4  : PeakList Filename (for prefix) & PeakList Filename suffix ----------
        fileprefix = self.list_txtctrl[3].GetValue()
        filesuffix = self.list_txtctrl[4].GetValue()

        #-------  field 5 : Nbdigits in index filename --------------
        nbdigits_filename = self.getnbdigits()

        #-------  field 1, 2 & 3: PeakList .dat Folder, .cor folder & .fit folder  ----------
        filepathdat = self.list_txtctrl[0].GetValue()
        filepathcor = self.list_txtctrl[1].GetValue()
        filepathout = self.list_txtctrl[2].GetValue()

        print("filepathcor", filepathcor)
        print("filepathout", filepathout)

        #-------  field 9 : Detector Calibration File (.det) --------------
        filedet = self.list_txtctrl[9].GetValue()

        # checking if at least one peak list filename with prefix exist
        listfiles = os.listdir(filepathdat)
        #             print "listfiles", listfiles
        nbfiles = len(listfiles)
        print("nb of files in directory %s   : " % filepathdat, nbfiles)
        if nbfiles == 0:
            wx.MessageBox("Apparently the folder %s is empty!" % filepathdat, "ERROR")
            return

        indexfile = 0
        FileNotFound = True
        while FileNotFound:
            if listfiles[indexfile].endswith(filesuffix):
                #                     print listfiles[indexfile]
                if listfiles[indexfile].startswith(fileprefix):
                    break
            if indexfile == nbfiles - 1:
                wx.MessageBox("No peaklist filename %s starting with\n%s\nin folder\n%s"
                    % (filesuffix, fileprefix, filepathdat), "ERROR")
                FileNotFound = False
            indexfile += 1

        # at least one file has been found
        if not FileNotFound:
            return

        CCDCalibdict = None
        if filesuffix in ('.dat',):
            CCDCalibdict = IOLT.readCalib_det_file(filedet)

        Index_Refine_Parameters_dict = {}

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
        #-------  field 6 : Start Image index or List indices File --------------
        startindexstr = self.list_txtctrl[6].GetValue()
        enable_finalandstepindex = True
        if startindexstr.startswith(('[', '(', '{')):
            print('startindex is a list of indices.')
            startindex = startindexstr
            enable_finalandstepindex = False
        elif startindexstr[0] in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
            print('startindex is an integer')
            try:
                startindex = int(startindexstr)
            except ValueError:
                wx.MessageBox("You should enter a single integer value for startindex", "ERROR")
                return
        else:
            print("startindex is a path (str) to a file with all indices")
            startindex = startindexstr
            enable_finalandstepindex = False

        #-------  fields 7 & 8 : Final Image index and  Image index step --------------
        if enable_finalandstepindex:
            try:
                finalindex = int(self.list_txtctrl[7].GetValue())
                stepindex = int(self.list_txtctrl[8].GetValue())
            except ValueError:
                wx.MessageBox("You should enter integer values for finalindex and stepindex fields", "ERROR")
                return
        else:
            finalindex = None
            stepindex = None

        fileindexrange = (startindex, finalindex, stepindex)

        # ------ check boxes ------------------------------
        #   ----  Index n using n-1 results  ---
        use_previous_results = self.previousreschk.GetValue()
        #   ----  (Re)Analyse (overwrite results)  ---
        reanalyse = self.chck_renanalyse.GetValue()
        #   ----  Update preexisting results  ---
        updatefitfiles = self.updatefitfiles.GetValue()


        verbosemode = self.verbosemode.GetValue()

        # ------  field 11: Minimum matching rate
        MinimumMatchingRate = float(self.list_txtctrl[11].GetValue())
        print("MinimumMatchingRate to avoid starting general indexation is ", MinimumMatchingRate)

        #---------------  field 10 : Guesses Matrix(ces) (.mat, .mats, .ubs)
        # read file containing guessed UB matrix or params to check orientation in .ubs file to check potential matching --------------
        # before doing (maybe long) indexation from scratch
        guessedMatricesFile = str(self.list_txtctrl[10].GetValue())
        print("guessedMatricesFile", guessedMatricesFile)

        if guessedMatricesFile not in ("None", "none", 'NONE'):
            print("Reading general file for guessed UB solutions")

            if use_previous_results:
                wx.MessageBox('"index n using n-1 results" can be checked if "Guessed Matrix(ces)" is None', 'INFO')
                return

            # read list or single matrix (ces) in GUI field
            if not guessedMatricesFile.endswith(".ubs"):
                _, guessedSolutions = IOLT.readListofMatrices(guessedMatricesFile)

                print("guessedmatrix", guessedSolutions)
                Index_Refine_Parameters_dict["GuessedUBMatrix"] = guessedSolutions
            # read .ubs file
            else:
                Index_Refine_Parameters_dict["CheckOrientation"] = guessedMatricesFile

            Index_Refine_Parameters_dict["MinimumMatchingRate"] = MinimumMatchingRate
        elif updatefitfiles:
            Index_Refine_Parameters_dict["MinimumMatchingRate"] = MinimumMatchingRate
        else:
            # we are sure to be less than that!
            Index_Refine_Parameters_dict["MinimumMatchingRate"] = max(0.123456, MinimumMatchingRate)

        # ----------------------------------------------------------------------

        # ----- selecting part of peaks that belong to "refposfile"
        #------------  field 13: ---- 'Selected Peaks from File -----------------
        # if spots positions evolve  for successive images
        trackingmode = self.trackingmode.GetValue()
        Index_Refine_Parameters_dict['trackingmode'] = trackingmode

        rsl = self.list_txtctrl[13].GetValue()
        if rsl in ('None', "None", 'none', "none"):
            Index_Refine_Parameters_dict['Reference Spots List'] = None
        else:
            Index_Refine_Parameters_dict['Reference Spots List'] = rsl

        # print("\n\n\n ------Index_Refine_Parameters_dict['Reference Spots List']", Index_Refine_Parameters_dict['Reference Spots List'])
        if Index_Refine_Parameters_dict['Reference Spots List'] is not None:

            mapshape = self.readmapshape()
            if mapshape is None:
                wx.MessageBox('You need to fill Map Shape field:\n"[nb steps // x, nb steps //y]".\n '
                    'At least two numbers! "(nb steps,1)" could work as well for 1D scan\n '
                    'Please put brackets or parentheses', 'INFO')
                return
            else:
                Index_Refine_Parameters_dict['mapshape'] = mapshape

        if self.parent is not None:
            object_to_set = self.parent  # IR

            print("object_to_set.initialparameters", object_to_set.initialparameters)

            object_to_set.initialparameters["IndexRefine PeakList Folder"] = filepathout
            object_to_set.initialparameters["file xyz"] = "None"
            object_to_set.initialparameters["IndexRefine PeakList Prefix"] = fileprefix
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

        # NB_MATERIALS = len(self.dict_param_list)

        # print('self.dict_param_list', self.dict_param_list)

        try:
            nb_cpus = int(self.txtctrl_cpus.GetValue())
        except ValueError:
            wx.MessageBox("nb of cpu(s) must be positive integer!", "Error")
            return
        if nb_cpus <= 0:
            wx.MessageBox("nb of cpu(s) must be positive integer!", "Error")
            return

        ISS.indexFilesSeries(filepathdat, filepathcor, filepathout,
                    fileprefix, filesuffix, nbdigits_filename,
                    startindex, finalindex, stepindex,
                    filedet,
                     guessedMatricesFile, MinimumMatchingRate,
                     fileirp,
                     Index_Refine_Parameters_dict['Reference Spots List'],
                     nb_cpus,
                     reanalyse,
                     use_previous_results,
                     updatefitfiles,
                     trackingmode=trackingmode,
                     build_hdf5=True,
                     verbose=verbosemode)
        return


def fill_list_valueparamIR(initialparameters):
    """
    return a list of default value for index_refine board from a dict initialparameters
    """
    list_valueparamIR = [initialparameters["PeakList Folder"],
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
                        initialparameters["Selected Peaks from File"]]

    return list_valueparamIR


# default values for the fields appearing in the Index_Refine.py GUI
initialparameters = {}

MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "GeGaN")

initialparameters["PeakList Folder"] = os.path.join(MainFolder, "datfiles")
initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
initialparameters["PeakList Filename Prefix"] = "nanox2_400_"
initialparameters["IndexRefine Parameters File"] = os.path.join(MainFolder, "GeGaN.irp")
initialparameters["Detector Calibration File .det"] = os.path.join(
    MainFolder, "calibGe_Feb2020.det")
initialparameters["Detector Calibration File (.dat)"] = os.path.join(
    MainFolder, "nanox2_400_0000_LT_1.dat")
initialparameters["PeakList Filename Suffix"] = ".dat"

initialparameters["nbdigits"] = 4
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
initialparameters["Selected Peaks from File"] = None

# for local test:
if 0:
    MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "CuSi")
    print("MainFolder", MainFolder)
    initialparameters["PeakList Folder"] = os.path.join(MainFolder, "corfiles")
    initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
    initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
    initialparameters["PeakList Filename Prefix"] = "SiCustrain"
    initialparameters["IndexRefine Parameters File"] = os.path.join(MainFolder, "sicu.irp")
    initialparameters["PeakList Filename Suffix"] = ".cor"
    initialparameters["Detector Calibration File .det"] = None
    initialparameters["nbdigits"] = 0
    # initialparameters["Selected Peaks from File"] = os.path.join(MainFolder,
    #                                                       "corfiles", "SiCustrain5_Cu20spots.fit")
    initialparameters["Selected Peaks from File"] = 'None'
    initialparameters["startingindex"] = 0
    initialparameters["finalindex"] = 5
    initialparameters["stepindex"] = 1
# for local test    guessUBmatrices on 3 grains
if 1:
    MainFolder = '/home/micha/LaueProjects/LauraConvert_TiLaser_Nov2020/LaserTi/T40'
    print("MainFolder", MainFolder)
    initialparameters["PeakList Folder"] = os.path.join(MainFolder, "datfiles")
    initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
    initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
    initialparameters["PeakList Filename Prefix"] = "T40N20_"
    initialparameters["IndexRefine Parameters File"] = "/home/micha/LaueProjects/LauraConvert_TiLaser_Nov2020/LaserTi/T40/datfiles/T40N20.irp"
    initialparameters["PeakList Filename Suffix"] = ".dat"
    initialparameters["Detector Calibration File .det"] = '/home/micha/LaueProjects/LauraConvert_TiLaser_Nov2020/LaserTi/calibLaura20Nov20.det'
    initialparameters["nbdigits"] = 4
    initialparameters["Selected Peaks from File"] = 'None'

    initialparameters["startingindex"] = 8
    initialparameters["finalindex"] = 10
    initialparameters["stepindex"] = 1

# for local test:
if 0:
    MainFolder = '/home/micha/LaueProjects/SiSibulle_Lukas'
    print("MainFolder", MainFolder)
    initialparameters["PeakList Folder"] = MainFolder
    initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
    initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
    initialparameters["PeakList Filename Prefix"] = "LukasHR_"
    initialparameters["IndexRefine Parameters File"] = os.path.join(MainFolder, "SiSi.irp")
    initialparameters["PeakList Filename Suffix"] = ".dat"
    initialparameters["Detector Calibration File .det"] = '/home/micha/LaueProjects/SiSibulle_Lukas/calibSilukas.det'
    initialparameters["nbdigits"] = 4
    initialparameters["Selected Peaks from File"] = os.path.join(MainFolder,
                                                          "corfiles", "SiCustrain5_Cu20spots.fit")
    initialparameters["Selected Peaks from File"] = '/home/micha/LaueProjects/SiSibulle_Lukas/SiLukas_acceptedspots.cor'
    initialparameters["startingindex"] = 0
    initialparameters["finalindex"] = 5
    initialparameters["stepindex"] = 1


# prepare sorted list of values
list_valueparamIR = fill_list_valueparamIR(initialparameters)

def start():
    """ start of GUI for module launch"""
    Stock_INDEXREFINE = Stock_parameters_IndexRefine(LIST_TXTPARAM_FILE_INDEXREFINE, list_valueparamIR)

    print("Stock_INDEXREFINE", Stock_INDEXREFINE.list_txtparamIR)
    print("Stock_INDEXREFINE", Stock_INDEXREFINE.list_valueparamIR)
    IndexRefineSeriesApp = wx.App()
    IndexRefineSeries = MainFrame_indexrefine(None, -1, "Index Refine Parameters Board",
                                                    initialparameters, Stock_INDEXREFINE)
    IndexRefineSeries.Show()
    IndexRefineSeriesApp.MainLoop()

if __name__ == "__main__":
    start()
