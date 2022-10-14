# -*- coding: utf-8 -*-
r"""
plotrefineGUI is a GUI class to plot laue pattern, index it and refine the corresponding unit cell strain

This module belongs to the open source LaueTools project with a free code repository at
https://gitlab.esrf.fr/micha/lauetools
mailto: micha -at* esrf *dot- fr

March 2020
"""
from __future__ import division
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import copy
import os
import sys
import time

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        """ alias fct """
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

import numpy as np

np.set_printoptions(precision=15)

# Plot & Tools Frame Class
from pylab import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                    NavigationToolbar2WxAgg as NavigationToolbar)
from matplotlib import __version__ as matplotlibversion
import matplotlib

if sys.version_info.major == 3:
    from .. import CrystalParameters as CP
    from .. import IOLaueTools as IOLT
    from .. import generaltools as GT
    from .. import dict_LaueTools as DictLT
    from .. import lauecore as LAUE
    from .. import FitOrient as FitO
    from . import LaueSpotsEditor as LSEditor
    from .. import indexingSpotsSet as ISS
    from .. import matchingrate
    from .. import imageprocessing as ImProc
    from .. import orientations as ORI
    from .. import IOimagefile as IOimage
    from . import OpenSpotsListFileGUI as OSLFGUI

else:
    import CrystalParameters as CP
    import IOLaueTools as IOLT
    import generaltools as GT
    import dict_LaueTools as DictLT
    import lauecore as LAUE
    import FitOrient as FitO
    import GUI.LaueSpotsEditor as LSEditor
    import indexingSpotsSet as ISS
    import matchingrate
    import imageprocessing as ImProc
    import orientations as ORI
    import IOimagefile as IOimage
    import OpenSpotsListFileGUI as OSLFGUI


# class MessageDataBox(wx.Dialog):
#     def __init__(self, parent, _id, title, matrix, defaultname):
class MessageDataBox(wx.Frame):
    """ class GUI allow to store matrix """
    def __init__(self, parent, _id, title, matrix, defaultname):
        wx.Frame.__init__(self, parent, _id, title, size=(400, 200))

        self.parent = parent
        self.matrix = matrix

        txt = wx.StaticText(self, -1, "Choose Matrix name")

        self.comments = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(300, 100))
        self.comments.SetValue(self.CreateStringfromMatrix(matrix))
        self.text2 = wx.TextCtrl(self, -1, "", style=wx.TE_MULTILINE, size=(300, 30))
        self.text2.SetValue(defaultname)

        btna = wx.Button(self, wx.ID_OK, "Accept", size=(150, 40))
        btna.Bind(wx.EVT_BUTTON, self.OnAccept)
        btna.SetDefault()

        btnc = wx.Button(self, -1, "Cancel", size=(100, 40))
        btnc.Bind(wx.EVT_BUTTON, self.OnQuit)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(btna, 1)
        hbox.Add(btnc, 1)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt, 0, wx.EXPAND)
        vbox.Add(self.comments, 0, wx.EXPAND)
        vbox.Add(self.text2, 0, wx.EXPAND)
        vbox.Add(hbox, 0)
        vbox.Add(wx.StaticText(self, -1, ""), 0)

        self.SetSizer(vbox)

    def OnQuit(self, _):
        """ quit """
        self.Close()

    def OnAccept(self, _):
        """ accept and quit
        set self.parent.selectedName and self.parent.dict_Rot[matrixname]
        """
        matrixname = str(self.text2.GetValue())
        self.parent.dict_Rot[matrixname] = self.matrix
        self.parent.selectedName = matrixname

        self.Close()

    def CreateStringfromMatrix(self, matrix):
        """ create string from 3*3 matrix"""
        matstr = "UBmat=\n["
        for elem in matrix:
            matstr += str(elem) + ",\n"

        res = matstr[:-2] + "]"

        print("res")

        return res


def call_counter(func):
    """ decorator function to count the nb of calls of 'func'
    """
    def helper(*args, **kwargs):
        """ call func with args"""
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__ = func.__name__

    return helper


# --- ---------Plot & Tools Frame Class
class Plot_RefineFrame(wx.Frame):
    """
    Class to implement a window enabling indexation and strain refinement
    """
    def __init__(self, parent, _id, title, data_added=None, datatype="2thetachi", ImageArray=None,
                                                kf_direction="Z>0",
                                                key_material="Ge",
                                                Params_to_simulPattern=None,  # Grain, Emin, Emax
                                                ResolutionAngstrom=False,
                                                MATR=0.5,
                                                CCDdetectorparameters=None,
                                                IndexationParameters=None,
                                                StorageDict=None,
                                                DataSetObject=None,
                                                **kwds):

        wx.Frame.__init__(self, parent, _id, title, size=(1000, 1200), **kwds)
        # wxmpl.PlotFrame(self, -1,'fdgf', size =(600, 400),dpi = 96)

        self.panel = wx.Panel(self)
        self.parent = parent

        self.IndexationParameters = IndexationParameters

        self.mainframe = IndexationParameters["mainAppframe"]
        if "indexationframe" in IndexationParameters:
            self.indexationframe = IndexationParameters["indexationframe"]
        else:
            self.indexationframe = "unknown"

        # -----image array pixels data ------------------------
        self.ImageArrayInit = ImageArray
        self.ImageArray = ImageArray  # this array is effectively displayed and can be modified
        self.ImageArrayMinusBckg = None
        self.data_dict = {}
        self.data_dict["Imin"] = 1
        self.data_dict["Imax"] = 1000
        self.data_dict["vmin"] = 1
        self.data_dict["vmax"] = 1000
        self.data_dict["lut"] = "jet"
        self.data_dict["logscale"] = True
        self.data_dict["markercolor"] = "b"
        self.data_dict["removebckg"] = False

        self.datatype_unchanged = None
        self.centerx, self.centery = None, None
        self.tth, self.chi, self.pixelX, self.pixelY = None, None, None, None
        self.linkExpMiller_link = None
        self.linkResidues_link = None
        self.linkIntensity_link = None
        self.Energy_Exp_spot = None
        self.fields = None
        self.linkResidues = None

        self.linkExpMiller_fit = None
        self.linkResidues_fit = None
        self.linkIntensity_fit = None
        self.fit_completed = False
        self.residues_non_weighted = None
        self.varyingstrain = None
        self.newUmat = None
        self.newUBmat = None
        self.previous_Umat = None
        self.previous_Bmat = None
        self.UBB0mat = None
        self.deviatoricstrain_sampleframe = None
        self.HKLxyz_names = None
        self.HKLxyz = None

        self.allparameters = None
        self.fitting_parameters_keys = None
        self.fitting_parameters_values = None

        self.init_plot = None
        self.myplot = None
        self.data_theo_displayed = None

        self.selectedName = None
        self._dataANNOTE_exp = None
        self.sigmanoise = 0

        self.datatype = datatype

        if IndexationParameters is not None:
            # all data to be indexed in this board
            #             AllData = self.IndexationParameters['AllDataToIndex']
            self.dirname = self.IndexationParameters["dirname"]
            DataToIndex = self.IndexationParameters["DataToIndex"]
            print("\n\nPlot_RefineFrame\n\n****\n\nNumber of spots in DataToIndex",
                                                                    len(DataToIndex["data_theta"]))
            if self.datatype is "2thetachi":
                self.Data_X = 2.0 * DataToIndex["data_theta"]
                self.Data_Y = DataToIndex["data_chi"]
            elif self.datatype is "pixels":
                self.Data_X = DataToIndex["data_X"]
                self.Data_Y = DataToIndex["data_Y"]

            self.Data_I = DataToIndex["data_I"]
            self.File_NAME = self.IndexationParameters["Filename"]
            self.data_XY = DataToIndex["data_X"], DataToIndex["data_Y"]
            self.data_2thetachi = 2 * DataToIndex["data_theta"], DataToIndex["data_chi"]

            self.data = self.Data_X, self.Data_Y, self.Data_I, self.File_NAME
            #             print DataToIndex.keys()
            self.selectedAbsoluteSpotIndices_init = DataToIndex["current_exp_spot_index_list"]
            self.selectedAbsoluteSpotIndices = copy.copy(self.selectedAbsoluteSpotIndices_init)

        self.setcoordinates()

        self.Millerindices = None
        self.Data_index_expspot = np.arange(len(self.Data_X))

        self.MATR = MATR

        # simulation parameters
        # self.SimulParam =(grain, emin, self.emax.GetValue())
        self.SimulParam = Params_to_simulPattern
        print("self.SimulParam in Plot_RefineFrame", self.SimulParam)

        self.kf_direction = kf_direction

        if IndexationParameters is not None:
            self.DataPlot_filename = IndexationParameters["DataPlot_filename"]
            self.current_processedgrain = IndexationParameters["current_processedgrain"]
        else:
            self.DataPlot_filename = self.mainframe.DataPlot_filename
            self.current_processedgrain = self.mainframe.current_processedgrain

        # initial parameters of calibration ----------------------*
        if CCDdetectorparameters is not None:
            print('CCDdetectorparameters is known in __init__ of Plot_RefineFrame')
            self.CCDcalib = CCDdetectorparameters["CCDcalib"]
            self.framedim = CCDdetectorparameters["framedim"]
            self.pixelsize = CCDdetectorparameters["pixelsize"]
            self.CCDLabel = CCDdetectorparameters["CCDLabel"]
            self.detectordiameter = CCDdetectorparameters["detectordiameter"]
        else:
            self.CCDcalib = self.mainframe.defaultParam
            self.framedim = self.mainframe.framedim
            self.pixelsize = self.mainframe.pixelsize
            self.CCDLabel = self.mainframe.CCDLabel

        if StorageDict is not None:
            self.mat_store_ind = StorageDict["mat_store_ind"]
            self.Matrix_Store = StorageDict["Matrix_Store"]
            self.dict_Rot = StorageDict["dict_Rot"]
            self.dict_Materials = StorageDict["dict_Materials"]
        else:
            self.mat_store_ind = self.mainframe.mat_store_ind
            self.Matrix_Store = self.mainframe.Matrix_Store
            self.dict_Rot = self.mainframe.dict_Rot
            self.dict_Materials = self.mainframe.dict_Materials

        print("detector parameters in Plot_RefineFrame")
        print(self.CCDcalib, self.framedim, self.pixelsize, self.CCDLabel)

        # simulated 2theta,chi
        self.data_theo = data_added
        # overwrite self.data_theo
        self.ResolutionAngstrom = ResolutionAngstrom
        self.Simulate_Pattern()
        # self.data_theo = data_added # a way in the past to put simulated data

        self.xlim, self.ylim = self.getDataLimits()

        self.points = []  # to store points
        self.selectionPoints = []
        self.twopoints = []
        self.nbclick = 1
        self.nbclick_dist = 1

        self.recognition_possible = True
        self.listbuttonstate = None
        self.toshow = []

        self.savedfileindex = 0  # index in fit results filename

        self.onlyclosest = 1

        # defaut value for Miller attribution
        self.B0matrix = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]  # means: columns are a*,b*,c* in xyz frame
        self.key_material = key_material
        self.detectordistance = None

        # for annotation
        self.drawnAnnotations_exp = {}
        self.links_exp = []

        self.drawnAnnotations_theo = {}
        self.links_theo = []
        # ------------------

        # No links between exp and theo spots have been done so far
        self.linkedspots = []
        self.linkExpMiller = []
        self.linkIntensity = []
        # No fit results
        self.linkedspots_fit = None
        self.linkedspots_link = None
        self.fitresults = False
        self.fitcounterindex = 0

        # highlight spots
        self.plotlinks = None   # exp spot linked to theo spot
        self.highlighttheospot = None  # closest theo. spot from mouse position
        self.highlightexpspot = None # closest exp. spot from mouse position

        self.Bmat = None
        self.Umat = None
        self.Umat2 = None
        self.Bmat_tri = None

        self.UBmat = copy.copy(self.SimulParam[0][2])
        print('self.UBmat: ', self.UBmat)
        self.current_matrix = self.UBmat
        # saving previous unit cell strain and orientation
        self.previous_UBmat = self.UBmat

        self.deviatoricstrain = None

        self.Tsresults = None
        self.new_latticeparameters = None
        self.constantlength = "a"

        self.DataSet = DataSetObject
        print("self.DataSet.detectordiameter in init plot refine frame", self.DataSet.detectordiameter)

        self.initGUI()

    @call_counter
    def initGUI(self):
        """ init GUI of Plot_RefineFrame """
        colourb_bkg = [242, 241, 240, 255]
        colourb_bkg = np.array(colourb_bkg) / 255.0

        self.dpi = 100
        self.figsizex, self.figsizey = 4, 3
        self.fig = Figure(
            (self.figsizex, self.figsizey), dpi=self.dpi, facecolor=tuple(colourb_bkg))
        self.fig.set_size_inches(self.figsizex, self.figsizey, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        self.init_plot = True

        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas)

        self.sb = wx.StatusBar(self, -1)
        self.sb.SetFieldsCount(2)
        self.SetStatusBar(self.sb)

        self.fig.canvas.mpl_connect("button_press_event", self.onClick)

        self.tooltip = wx.ToolTip(tip="Welcome on LaueTools UB refinement board")
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)

        self.pointButton5 = wx.ToggleButton(self.panel, -1, "Accept Matching")
        self.pointButton6 = wx.ToggleButton(self.panel, -1, "Draw Exp. Spot index")
        self.pointButton7 = wx.ToggleButton(self.panel, -1, "Draw Theo. Spot index")
        self.listbuttons = [self.pointButton5, self.pointButton6, self.pointButton7]
        self.defaultColor = self.GetBackgroundColour()
        self.p5S, self.p6S, self.p7S = 0, 0, 0
        self.listbuttonstate = [self.p5S, self.p6S, self.p7S]
        self.listbuttonstate = [0, 0, 0]

        self.pointButton5.Bind(wx.EVT_TOGGLEBUTTON, self.OnAcceptMatching)
        self.pointButton6.Bind(wx.EVT_TOGGLEBUTTON, self.T6) # Draw Exp. Spot index
        self.pointButton7.Bind(wx.EVT_TOGGLEBUTTON, self.T7) # Draw Theo. Spot index

        self.switchCoordinatesbtn = wx.Button(self.panel, -1, "Switch Coord.")
        self.switchCoordinatesbtn.Bind(wx.EVT_BUTTON, self.OnSwitchCoords)
        self.datatype_unchanged = True

        self.imagescalebtn = wx.Button(self.panel, -1, "Load Image")
        self.imagescalebtn.Bind(wx.EVT_BUTTON, self.onSetImageScale)

        self.noisebtn = wx.Button(self.panel, -1, "Add Noise")
        self.noisebtn.Bind(wx.EVT_BUTTON, self.onSetNoise)

        self.enterUBbtn = wx.Button(self.panel, -1, "Enter Matrix")
        self.enterUBbtn.Bind(wx.EVT_BUTTON, self.onEnterMatrix)

        self.txt2 = wx.StaticText(self.panel, -1, "Spot Size")
        self.spotsize_ctrl = wx.TextCtrl(self.panel, -1, '1')

        self.txt1 = wx.StaticText(self.panel, -1, "Match. Ang. Tol.")
        self.matr_ctrl = wx.TextCtrl(self.panel, -1, str(self.MATR))

        self.eminmaxtxt = wx.StaticText(self.panel, -1, "Energy min. and max.(keV): ")
        self.SCEmin = wx.SpinCtrl(self.panel, -1, "5", min=5, max=150)

        if self.SimulParam is not None:
            self.SCEmax = wx.SpinCtrl(self.panel, -1, str(self.SimulParam[2]), min=6, max=150)
        else:
            self.SCEmax = wx.SpinCtrl(self.panel, -1, "25", min=6, max=150)

        self.resolutiontxt = wx.StaticText(self.panel, -1, "Latt. Spacing Res.")
        self.resolutionctrl = wx.TextCtrl(self.panel, -1, str(self.ResolutionAngstrom))

        self.list_Materials = sorted(self.dict_Materials.keys())
        self.txtelem = wx.StaticText(self.panel, -1, "Material Element Structure:       ")
        self.comboElem = wx.ComboBox(self.panel, -1, self.key_material,
                                                choices=self.list_Materials, style=wx.CB_READONLY)
        self.comboElem.Bind(wx.EVT_COMBOBOX, self.EnterComboElem)

        self.txtmatrix = wx.StaticText(self.panel, -1, "Orientation Matrix (UB)")

        indexcall = self.initGUI.calls
        matrixkeyname = "Input UBmatrix_%d"%indexcall
        DictLT.dict_Rot[matrixkeyname] = copy.copy(self.UBmat)

        self.comboUBmatrix = wx.ComboBox(self.panel, -1, matrixkeyname,
                                        choices=list(DictLT.dict_Rot.keys()), style=wx.CB_READONLY)
        self.comboUBmatrix.Bind(wx.EVT_COMBOBOX, self.onSelectUBmatrix)

        self.btnreplot = wx.Button(self.panel, -1, "Replot")
        self.btnreplot.Bind(wx.EVT_BUTTON, self.OnReplot)

        self.UpdateFromRefinement = wx.CheckBox(self.panel, -1, "Use fitting results")
        self.UpdateFromRefinement.SetValue(False)

        self.undo_btn = wx.Button(self.panel, -1, "Undo Replot")
        self.undo_btn.Bind(wx.EVT_BUTTON, self.OnUndoReplot)

        self.btnfilterdata = wx.Button(self.panel, -1, "Filter Exp. Data")
        self.btnfilterdata.Bind(wx.EVT_BUTTON, self.BuildDataDict)

        self.btnautolink = wx.Button(self.panel, -1, "Auto. Links")
        self.btnautolink.Bind(wx.EVT_BUTTON, self.OnAutoLink)

        self.btnfilterlink = wx.Button(self.panel, -1, "Filter Links")
        self.btnfilterlink.Bind(wx.EVT_BUTTON, self.BuildDataDictAfterLinks)

        self.btnrefine = wx.Button(self.panel, -1, "Refine")
        self.btnrefine.Bind(wx.EVT_BUTTON, self.OnRefine_UB_and_Strain)

        self.btnShowResults = wx.Button(self.panel, -1, "Show Results")
        self.btnShowResults.Bind(wx.EVT_BUTTON, self.build_FitResults_Dict)

        self.btnstoreUBmat = wx.Button(self.panel, -1, "Store UBMat")
        self.btnstoreUBmat.Bind(wx.EVT_BUTTON, self.OnStoreMatrix)

        self.use_forfit1 = wx.RadioButton(self.panel, -1, "", style=wx.RB_GROUP)
        self.use_forfit2 = wx.RadioButton(self.panel, -1, "")
        self.use_forfit3 = wx.RadioButton(self.panel, -1, "")
        self.use_forfit1.SetValue(True)
        self.txtuserefine = wx.StaticText(self.panel, -1,
                                                "<--- Use either built links for Refinement --->")

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.title1 = wx.StaticText(self.panel, -1, "Fitting parameters")
        self.title1.SetFont(font3)

        self.use_weights = wx.CheckBox(self.panel, -1, "Use Intensity Weights")
        self.use_weights.SetValue(False)

        self.fitTrecip = wx.CheckBox(self.panel, -1, "Fit Global operator")
        self.fitTrecip.SetValue(True)

        self.fitycen = wx.CheckBox(self.panel, -1, "Fit depth")
        self.fitycen.SetValue(False)

        self.sampledepthtxt = wx.StaticText(self.panel, -1, "Sample depth")
        # self.matr_ctrl = wx.TextCtrl(self.parampanel, -1,'0.5',(350, 40))
        self.sampledepthctrl = wx.TextCtrl(self.panel, -1, '0', size=(30, -1))  # in microns!

        self.fitorient1 = wx.CheckBox(self.panel, -1, "Orient. 1")
        self.fitorient1.SetValue(False)
        self.fitorient2 = wx.CheckBox(self.panel, -1, "Orient. 2")
        self.fitorient2.SetValue(False)
        self.fitorient3 = wx.CheckBox(self.panel, -1, "Orient. 3")
        self.fitorient3.SetValue(False)

        self.txtfitlatticeparameters = wx.StaticText(self.panel, -1, "+ ---> ")

        self.fita = wx.CheckBox(self.panel, -1, "a")
        self.fita.SetValue(False)
        self.fitb = wx.CheckBox(self.panel, -1, "b")
        self.fitb.SetValue(False)
        self.fitc = wx.CheckBox(self.panel, -1, "c")
        self.fitc.SetValue(False)
        self.fitalpha = wx.CheckBox(self.panel, -1, "alpha")
        self.fitalpha.SetValue(False)
        self.fitbeta = wx.CheckBox(self.panel, -1, "beta")
        self.fitbeta.SetValue(False)
        self.fitgamma = wx.CheckBox(self.panel, -1, "gamma")
        self.fitgamma.SetValue(False)

        self.txtfitTsparameters = wx.StaticText(self.panel, -1, "OR + ---> ")
        self.Ts00chck = wx.CheckBox(self.panel, -1, "Ts00")
        self.Ts00chck.SetValue(False)
        self.Ts01chck = wx.CheckBox(self.panel, -1, "Ts01")
        self.Ts01chck.SetValue(False)
        self.Ts02chck = wx.CheckBox(self.panel, -1, "Ts02")
        self.Ts02chck.SetValue(False)
        self.Ts11chck = wx.CheckBox(self.panel, -1, "Ts11")
        self.Ts11chck.SetValue(False)
        self.Ts12chck = wx.CheckBox(self.panel, -1, "Ts12")
        self.Ts12chck.SetValue(False)
        self.Ts22chck = wx.CheckBox(self.panel, -1, "Ts22")
        self.Ts22chck.SetValue(False)

        self.fita.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_latticeparams)
        self.fitb.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_latticeparams)
        self.fitc.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_latticeparams)
        self.fitalpha.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_latticeparams)
        self.fitbeta.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_latticeparams)
        self.fitgamma.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_latticeparams)
        self.Ts00chck.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_Tsparams)
        self.Ts01chck.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_Tsparams)
        self.Ts02chck.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_Tsparams)
        self.Ts11chck.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_Tsparams)
        self.Ts12chck.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_Tsparams)
        self.Ts22chck.Bind(wx.EVT_CHECKBOX, self.Onselectfitparameters_Tsparams)

        self.incrementfilename = wx.CheckBox(self.panel, -1, "Auto Increment .fit file")
        self.incrementfilename.SetValue(True)

        self.svbutton = wx.Button(self.panel, -1, "Save Results")
        self.svbutton.Bind(wx.EVT_BUTTON, self.onWriteFitFile)
        self.svbutton.SetFont(font3)

        self._layout()
        self._replot()

        # tooltips
        self.use_weights.SetToolTipString("Weight each pair distance (separating exp. and modeled "
        "spot) by intensity of exp. spot.")
        self.incrementfilename.SetToolTipString("If checked, increment the index appearing in "
        "the name of the results file (avoiding overwriting).")

        linktip = "Maximum angle separating one exp. and one simulated spot to form a spot pairs "
        "for automatic spots linksprocedue."
        self.txt1.SetToolTipString(linktip)
        self.matr_ctrl.SetToolTipString(linktip)

        self.use_forfit1.SetToolTipString("Refine model build from automatic spots pairing")
        self.use_forfit2.SetToolTipString('Refine model build from spots pairs selected by means '
        'of "Filter links" board.')
        self.use_forfit3.SetToolTipString('Refine model build from spots pairs selected '
        'by means of "Show results" board.')

        self.btnstoreUBmat.SetToolTipString("Store in LaueToolsGUI software the current refined "
        "orientation matrix.")

        self.pointButton5.SetToolTipString('Accept the current (refined) model to index the spots. '
        'Within "angular tolerance" each exp. spot will get the hkl of the nearest simulated '
        'spot. Indexed spots will not belong to the peaks list for further indexation.')

        self.pointButton6.SetToolTipString('Draw on plot experimental props (intensity, spot index) of clicked spot')

        self.pointButton7.SetToolTipString('Draw on plot theoretical  props (Miller indices, energy (keV)) of clicked spot')

        self.UpdateFromRefinement.SetToolTipString("If checked, update the plot of simulated spots "
        "according to the last refined model.")

        self.btnfilterdata.SetToolTipString("Select experimental spots to be plot and used to "
        "build a spots pairs model for refinement.")

        self.btnautolink.SetToolTipString('Build automatically a list of pairs composed of exp. '
        'and simulated spots. Each pair is found from close spots withing "angular tolerance".')

        self.btnfilterlink.SetToolTipString("Open a board to select pairs from those coming "
        "from the automatic spots pairing.")

        self.btnShowResults.SetToolTipString('Open a board to view results of model refinement. '
        'As with "filter links" button, some pairs can be selected to form a model to be refined.')

        self.btnrefine.SetToolTipString("Start the refinement of the model of a unit cell  "
        "(orientation and strain) by minimizing the pair distances between exp. and simulated "
        "spots. List of pairs to be used in model refinement can be selected by clicking on one "
        "circular radio button.")

        self.btnreplot.SetToolTipString("Replot simulated spots (red hollow circles) according "
        "to unit cell parameters.")

        self.svbutton.SetToolTipString("write .fit file with spots belonging "
        "to the current indexation")

        self.switchCoordinatesbtn.SetToolTipString("Switch coordinates: Scattering Angles (2theta, chi)/pixels (X,Y)")

        fittip = "Parameters to be refined:\n"
        fittip += "--- Global: Crystal orientation and Strain of reciprocal vectors\n"
        fittip += "qref = UBref Trecipref B0 G*\nwhere UBref=Rotx,y,z UBinit\n Trecipref is triangular up matrix with first element set to 1\n"
        fittip += "If Checked: next boxes will not be considered\n\n"
        fittip += "--- Crystal orientation and Unit cell lattice parameters a,b,c,alpha, beta, gamma except that one distance must be fixed\n"
        fittip += "(default a is set to a of the reference structure)\n"
        fittip += "qref = UBref B0 G*\n where UBref=Rotx,y,z UBinit Mref,\nMref is triangular up matrix with first element set to 1\n"
        fittip += "\n     OR      \n\n"
        fittip += ("--- Crystal orientation and Strain in sample frame (tilted by 40deg)\n")
        fittip += "6 elements of Ts which is a triangular up matrix, except that one element must be fixed"
        fittip += "qref = Tref Uref B0 G*\n"
        fittip += "with Tref= P-1 Tsref P\n"
        fittip += "P conversion matrix from sample frame to Lauetools one\n"
        fittip += "     (Ts00  Ts01   Ts02)\n"
        fittip += "Ts=  (0     Ts11   Ts12)\n"
        fittip += "     (0     0      Ts22)\n"
        fittip += "default value= Ts00 Ts11 Ts22 = 1 and Ts01 Ts02 Ts12 =0"

        self.fitTrecip.SetToolTipString(fittip)
        self.fitorient1.SetToolTipString(fittip)
        self.fitorient2.SetToolTipString(fittip)
        self.fitorient3.SetToolTipString(fittip)
        self.fita.SetToolTipString(fittip)
        self.fitb.SetToolTipString(fittip)
        self.fitc.SetToolTipString(fittip)
        self.fitalpha.SetToolTipString(fittip)
        self.fitbeta.SetToolTipString(fittip)
        self.fitgamma.SetToolTipString(fittip)
        self.Ts00chck.SetToolTipString(fittip)
        self.Ts01chck.SetToolTipString(fittip)
        self.Ts02chck.SetToolTipString(fittip)
        self.Ts11chck.SetToolTipString(fittip)
        self.Ts12chck.SetToolTipString(fittip)
        self.Ts22chck.SetToolTipString(fittip)

        self.fitycen.SetToolTipString('Refine also ycen (detector geometry parameter) to take into accound sample depth wrt depth of Ge reference sample during calibration. NOT IMPLEMENTED YET !')

        self.enterUBbtn.SetToolTipString("Enter Orientation Matrix UB")

        self.sampledepthctrl.SetToolTipString('depth in micrometre = 10-3 mm)')

    def _layout(self):
        """ layout
        """
        btnSizer0 = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer0.Add(self.pointButton6, 0, wx.ALL)
        btnSizer0.Add(self.pointButton7, 0, wx.ALL)
        btnSizer0.Add(self.switchCoordinatesbtn, 0, wx.ALL)

        energy0Sizer = wx.BoxSizer(wx.HORIZONTAL)
        energy0Sizer.Add(self.eminmaxtxt, 0, wx.ALL)
        energy0Sizer.Add(wx.StaticText(self.panel, -1, "        "), 0, wx.ALL)
        energy0Sizer.Add(self.resolutiontxt, 0, wx.ALL)

        energy1Sizer = wx.BoxSizer(wx.HORIZONTAL)
        energy1Sizer.Add(self.SCEmin, 0, wx.ALL)
        energy1Sizer.Add(self.SCEmax, 0, wx.ALL)
        energy1Sizer.Add(wx.StaticText(self.panel, -1, "    "), 0, wx.ALL)
        energy1Sizer.Add(self.resolutionctrl, 0, wx.ALL)

        fitparamSizer = wx.BoxSizer(wx.HORIZONTAL)
        fitparamSizer.Add(self.fitorient1, 0, wx.ALL)
        fitparamSizer.Add(self.fitorient2, 0, wx.ALL)
        fitparamSizer.Add(self.fitorient3, 0, wx.ALL)

        fitparam2Sizer = wx.BoxSizer(wx.HORIZONTAL)
        fitparam2Sizer.Add(self.txtfitlatticeparameters, 0, wx.ALL)
        fitparam2Sizer.Add(self.fita, 0, wx.ALL)
        fitparam2Sizer.Add(self.fitb, 0, wx.ALL)
        fitparam2Sizer.Add(self.fitc, 0, wx.ALL)
        fitparam2Sizer.Add(self.fitalpha, 0, wx.ALL)
        fitparam2Sizer.Add(self.fitbeta, 0, wx.ALL)
        fitparam2Sizer.Add(self.fitgamma, 0, wx.ALL)

        fitparam3Sizer = wx.BoxSizer(wx.HORIZONTAL)
        fitparam3Sizer.Add(self.txtfitTsparameters, 0, wx.ALL)
        fitparam3Sizer.Add(self.Ts00chck, 0, wx.ALL)
        fitparam3Sizer.Add(self.Ts01chck, 0, wx.ALL)
        fitparam3Sizer.Add(self.Ts02chck, 0, wx.ALL)
        fitparam3Sizer.Add(self.Ts11chck, 0, wx.ALL)
        fitparam3Sizer.Add(self.Ts12chck, 0, wx.ALL)
        fitparam3Sizer.Add(self.Ts22chck, 0, wx.ALL)

        if WXPYTHON4:
            vbox3 = wx.GridSizer(3, 10, 10)
        else:
            vbox3 = wx.GridSizer(2, 3)

        vbox3.AddMany(
            [(self.btnautolink, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                (self.btnfilterlink, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                (self.btnShowResults, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                (self.use_forfit1, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                (self.use_forfit2, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                (self.use_forfit3, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)])

        plotparSizer = wx.BoxSizer(wx.HORIZONTAL)
        plotparSizer.Add(self.btnreplot, 0, wx.ALL)
        plotparSizer.Add(self.UpdateFromRefinement, 0, wx.ALL)
        plotparSizer.Add(self.undo_btn, 0, wx.ALL)

        finalbtnSizer = wx.BoxSizer(wx.HORIZONTAL)
        finalbtnSizer.Add(self.btnstoreUBmat, 0, wx.ALL)
        finalbtnSizer.Add(self.svbutton, 0, wx.ALL)
        finalbtnSizer.Add(self.pointButton5, 0, wx.ALL)  # validate matching

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.txt1, 0, wx.ALL)
        hbox.Add(self.matr_ctrl, 0, wx.ALL)
        hbox.Add(self.txt2, 0, wx.ALL)
        hbox.Add(self.spotsize_ctrl, 0, wx.ALL)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.btnfilterdata, 0, wx.ALL)
        hbox2.Add(self.imagescalebtn, 0, wx.ALL)
        hbox2.Add(self.noisebtn, 0, wx.ALL)
        hbox2.Add(self.enterUBbtn, 0, wx.ALL)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.txtelem, 0, wx.ALL)
        hbox3.Add(self.txtmatrix, 0, wx.ALL)

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        hbox4.Add(self.comboElem, 0, wx.ALL)
        hbox4.Add(self.comboUBmatrix, 0, wx.ALL)

        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        hbox5.Add(self.use_weights, 0, wx.ALL)
        hbox5.Add(self.incrementfilename, 0, wx.ALL)

        hbox6 = wx.BoxSizer(wx.HORIZONTAL)
        hbox6.Add(self.fitTrecip, 0, wx.ALL)
        hbox6.Add(self.fitycen, 5, wx.ALL)
        hbox6.Add(self.sampledepthtxt, 5, wx.ALL)
        hbox6.Add(self.sampledepthctrl, 5, wx.ALL)

        sizerparam = wx.BoxSizer(wx.VERTICAL)
        sizerparam.Add(hbox2, 0, wx.ALL)
        sizerparam.Add(hbox, 0, wx.ALL)
        sizerparam.Add(energy0Sizer, 0, wx.ALL)
        sizerparam.Add(energy1Sizer, 0, wx.ALL)
        sizerparam.Add(hbox3, 0, wx.ALL)
        sizerparam.Add(hbox4, 0, wx.ALL)
        sizerparam.AddSpacer(5)
        sizerparam.Add(plotparSizer, 0, wx.ALL)
        sizerparam.AddSpacer(5)

        sizerparam.Add(vbox3, 0, wx.EXPAND)

        sizerparam.Add(self.txtuserefine, 1, wx.ALIGN_CENTER_HORIZONTAL)
        sizerparam.Add(self.title1, 0, wx.ALL)
        sizerparam.Add(hbox6, 0, wx.ALL)
        sizerparam.Add(fitparamSizer, 0, wx.ALL)
        sizerparam.Add(fitparam2Sizer, 0, wx.ALL)
        sizerparam.Add(fitparam3Sizer, 0, wx.ALL)
        sizerparam.Add(self.btnrefine, 0, wx.ALL | wx.EXPAND)
        sizerparam.AddSpacer(5)
        sizerparam.Add(hbox5, 0, wx.ALL)
        sizerparam.Add(finalbtnSizer, 0, wx.ALL)

        btnsSizer = wx.BoxSizer(wx.VERTICAL)
        btnsSizer.Add(btnSizer0, 0, wx.ALL)
        btnsSizer.Add(sizerparam, 0, wx.ALL)

        sizerplot = wx.BoxSizer(wx.VERTICAL)
        sizerplot.Add(self.canvas, 1, wx.TOP | wx.GROW)
        sizerplot.Add(self.toolbar, 0, wx.EXPAND)

        sizerH = wx.BoxSizer(wx.HORIZONTAL)
        sizerH.Add(sizerplot, 1, wx.ALL | wx.GROW, 5)
        sizerH.Add(btnsSizer, 0, wx.ALL, 5)

        self.panel.SetSizer(sizerH)
        sizerH.Fit(self)
        self.Layout()

    def Onselectfitparameters_latticeparams(self, _):
        """ uncheck self.fitTrecip and all Ts chcks"""
        self.fitTrecip.SetValue(False)
        if (self.fita.GetValue()
            or self.fitb.GetValue()
            or self.fitc.GetValue()
            or self.fitalpha.GetValue()
            or self.fitbeta.GetValue()
            or self.fitgamma.GetValue()):
            for chck in (self.Ts00chck,
                        self.Ts01chck,
                        self.Ts02chck,
                        self.Ts11chck,
                        self.Ts12chck,
                        self.Ts22chck):
                chck.SetValue(False)

    def Onselectfitparameters_Tsparams(self, _):
        """ uncheck self.fitTrecip and all fit# chcks"""
        self.fitTrecip.SetValue(False)
        if (self.Ts00chck.GetValue()
            or self.Ts01chck.GetValue()
            or self.Ts02chck.GetValue()
            or self.Ts11chck.GetValue()
            or self.Ts12chck.GetValue()
            or self.Ts22chck.GetValue()):
            for chck in (self.fita,
                        self.fitb,
                        self.fitc,
                        self.fitalpha,
                        self.fitbeta,
                        self.fitgamma):
                chck.SetValue(False)

    def onSetImageScale(self, _):
        """
        open a board to change image scale
        """
        IScaleBoard = IntensityScaleBoard(self, -1, "Image & scale setting Board", self.data_dict)

        IScaleBoard.Show(True)

    def onSetNoise(self, _):
        """
        open a board to add noise in experimental data
        """
        NLevelBoard = NoiseLevelBoard(self, -1, "Noise level setting Board", self.data_dict)

        NLevelBoard.Show(True)

    def onEnterMatrix(self, evt):
        """ enter a matrix """
        helptstr = "Enter Matrix elements : \n [[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]]"
        helptstr += " Or list of Matrices"
        dlg = wx.TextEntryDialog(
            self, helptstr, "Calibration- Orientation Matrix elements Entry")

        _param = "[[1,0,0],[0, 1,0],[0, 0,1]]"
        dlg.SetValue(_param)
        if dlg.ShowModal() == wx.ID_OK:
            paramraw = str(dlg.GetValue())
            import re

            listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", paramraw)
            #             print "listval", listval
            listelem = []
            for elem in listval:
                try:
                    val = float(elem)
                    listelem.append(val)
                except ValueError:
                    continue

            nbval = len(listelem)
            #             print "nbval", nbval

            if (nbval % 9) != 0:
                txt = "Something wrong, I can't read matrix or matrices"
                print(txt)

                wx.MessageBox(txt, "INFO")
                return

            nbmatrices = nbval / 9
            ListMatrices = np.zeros((nbmatrices, 3, 3))
            ind_elem = 0
            for ind_matrix in range(nbmatrices):
                for i in range(3):
                    for j in range(3):
                        floatval = listelem[ind_elem]
                        ListMatrices[ind_matrix][i][j] = floatval
                        ind_elem += 1
                        
                _allm = np.array(ListMatrices[ind_matrix])
                if np.linalg.det(_allm)<0:
                    txt = "Matrix is not direct (det(UB)<0)"
                    print(txt)

                    wx.MessageBox(txt, "ERROR")
                    return

            # save in list of orientation matrix
            # default name
            inputmatrixname = "InputMat_"

            initlength = len(DictLT.dict_Rot)
            for k, mat in enumerate(ListMatrices):
                mname = inputmatrixname + "%d" % k
                DictLT.dict_Rot[mname] = mat
                self.comboUBmatrix.Append(mname)
            print("len dict", len(DictLT.dict_Rot))

            self.comboUBmatrix.SetSelection(initlength)

            dlg.Destroy()

            self.OnReplot(evt)

    def setcoordinates(self):
        """ set coordinates of spots depending on chosen representing space 'self.datatype' """
        if self.datatype == "2thetachi":
            #             self.tth, self.chi = self.Data_X, self.Data_Y
            self.tth, self.chi = self.data_2thetachi
            self.pixelX, self.pixelY = self.data_XY
        elif self.datatype == "gnomon":
            pass
        elif self.datatype == "pixels":
            self.tth, self.chi = self.data_2thetachi
            self.pixelX, self.pixelY = self.data_XY
            print("pixels plot")

    def getDataLimits(self):
        """
        returns plot limits from experimental data
        """
        if self.datatype == "2thetachi":
            return get2Dlimits(self.tth, self.chi)
        elif self.datatype == "pixels":
            # we like upper origin for y axis
            if self.pixelX is None:
                return (0, 2300), (2300, 0)

            # xlim, ylim = get2Dlimits(self.pixelX, self.pixelY)

            if self.CCDLabel in ("MARCCD165", "PRINCETON"):
                ylim = (2048, 0)
                xlim = (0, 2048)
            elif self.CCDLabel in ("VHR_PSI",):
                ylim = (3000, 0)
                xlim = (0, 4000)
            elif self.CCDLabel.startswith("sCMOS"):
                ylim = (2050, 0)
                xlim = (0, 2050)
            else:
                ylim = (2500, 0)
                xlim = (0, 2500)
            return xlim, ylim


    def OnSwitchCoords(self, _):
        """ from btn switch representation space for spots position"""
        if self.datatype == "2thetachi":
            print("was 2theta")
            self.datatype = "pixels"

        elif self.datatype == "pixels":
            print("was pixels")
            self.datatype = "2thetachi"

        self.setcoordinates()

        print("space coordinates is now:", self.datatype)
        self.datatype_unchanged = False

        self._replot()

    # def onMouseOver(self, evt):
    #     name = evt.GetEventObject().GetClassName()
    #     self.sb.SetStatusText(name + " widget" + "     " + str(evt.GetEventObject()))
    #     # print("I am on it!")
    #     print(evt.GetEventObject())
    #     print(name)
    #     evt.Skip()

    def OnUndoReplot(self, _):
        """   replot with UB matrix prior to refinement  """
        self.UBmat = copy.copy(self.previous_UBmat)

        print("back to self.UBmat,self.Umat,self.Bmat")
        print(self.UBmat)

        self.OnReplot(1)

        self.UpdateFromRefinement.SetValue(False)

    def OnReplot(self, _):
        """ replot spots when pressing btn 'replot'"""

        if self.SimulParam is not None:
            # check element if user needs to change it
            keymaterial = str(self.comboElem.GetValue())
            #print("Element in pickyframe OnReplot() ", keymaterial)
            # check Emin,Emax
            Emin, Emax = float(self.SCEmin.GetValue()), float(self.SCEmax.GetValue())

            Grain = self.SimulParam[0]
            Grain[3] = keymaterial

            Grain[2] = DictLT.dict_Rot[str(self.comboUBmatrix.GetValue())]

            if self.UpdateFromRefinement.GetValue():
                print("Using refined UB matrix")
                #                 if self.Umat != None and self.Bmat != None:
                if self.fit_completed:
                    # this comes from the end of onrefinePicky()
                    # self.UBmat = UBmat
                    # self.Umat = np.dot(deltamat,starting_orientmatrix)
                    # self.Bmat = Bmat

                    # doesn't work
                    # Grain[0] = self.Bmat
                    # Grain[2] = self.Umat

                    # doesn't work
                    # Grain[0] = self.UBmat
                    # Grain[2] = np.eye(3)

                    # this works but some absolute lattice parameter may have been lost(spectral band maximum is perhaps a little bit too small)
                    Grain[2] = self.UBmat
                    Grain[0] = np.eye(3)

                else:
                    wx.MessageBox("You have not refined data yet !\n I will keep using non "
                        "distorted matrices...", "INFO")

            self.SimulParam = (Grain, Emin, Emax)
            resActrl = self.resolutionctrl.GetValue()
            if resActrl in ("False", "None", None, 0, "0", "0.0"):
                self.ResolutionAngstrom = False
            else:
                self.ResolutionAngstrom = float(resActrl)
            self.Simulate_Pattern()
            self._replot()
        else:
            wx.MessageBox("There is not simulated data to replot !!", "INFO")

    def onClick(self, event):
        """ onclick
        """
        if event.inaxes:
            print(("inaxes x,y", event.x, event.y))
            print(("inaxes  xdata, ydata", event.xdata, event.ydata))
            self.centerx, self.centery = event.xdata, event.ydata

            if self.pointButton6.GetValue():
                self.Annotate_exp(event)

            if self.pointButton7.GetValue():
                self.Annotate_theo(event)

    def onMotion_ToolTip(self, event):
        """tool tip to show data (exp. and theo. spots) when mouse hovers on plot
        """

        if len(self.data[0]) == 0:
            return

        collisionFound_exp = False
        collisionFound_theo = False

        if self.datatype == "2thetachi":
            xtol = 5
            ytol = 5
        elif self.datatype == "pixels":
            xtol = 200
            ytol = 200

        if self.datatype == "2thetachi":
            xdata, ydata, _annotes_exp = (self.Data_X,
                                        self.Data_Y,
                                        list(zip(self.Data_index_expspot, self.Data_I)))
        elif self.datatype == "pixels":
            xdata, ydata, _annotes_exp = (self.data_XY[0],
                                        self.data_XY[1],
                                        list(zip(self.Data_index_expspot, self.Data_I)))

        xdata_theo, ydata_theo, _annotes_theo = (self.data_theo[0],
                                                self.data_theo[1],
                                                list(zip(*self.data_theo[2:])))

        if event.xdata != None and event.ydata != None:

            evx, evy = event.xdata, event.ydata

            if self.datatype == "pixels":
                tip = "(X,Y)=(%.2f,%.2f)"%(evx, evy)
            if self.datatype == "2thetachi":
                tip = "(2theta,chi)=(%.2f,%.2f)"%(evx, evy)

            annotes_exp = []
            for x, y, aexp in zip(xdata, ydata, _annotes_exp):
                if (evx - xtol < x < evx + xtol) and (evy - ytol < y < evy + ytol):
                    #                     print "got exp. spot!! at x,y", x, y
                    annotes_exp.append((GT.cartesiandistance(x, evx, y, evy), x, y, aexp))

            annotes_theo = []
            for x, y, atheo in zip(xdata_theo, ydata_theo, _annotes_theo):
                if (evx - xtol < x < evx + xtol) and (evy - ytol < y < evy + ytol):
                    #                     print "got theo. spot!!"
                    #                     print "with info: ", atheo
                    annotes_theo.append((GT.cartesiandistance(x, evx, y, evy), x, y, atheo))

            if annotes_exp != []:
                collisionFound_exp = True
            if annotes_theo != []:
                collisionFound_theo = True

            if not collisionFound_exp and not collisionFound_theo:
                self.tooltip.SetTip(tip)
                return

            tip_exp = ""
            tip_theo = ""
            if self.datatype == "2thetachi":
                closedistance = 2.0
            elif self.datatype == "pixels":
                closedistance = 100.0

            if collisionFound_exp:
                annotes_exp.sort()
                _distanceexp, x, y, annote_exp = annotes_exp[0]

                # if exp. spot is close enough
                if _distanceexp < closedistance:
                    tip_exp = "spot index=%d. Intensity=%.1f" % (annote_exp[0], annote_exp[1])
                    print('found ->  at (%.2f,%.2f)'% (x, y), tip_exp)
                    self.updateStatusBar(x, y, annote_exp, spottype="exp")

                    self.highlightexpspot = annote_exp[0]
                else:
                    self.sb.SetStatusText("", 1)
                    tip_exp = ""
                    collisionFound_exp = False
                    self.highlightexpspot = None

            if collisionFound_theo:
                annotes_theo.sort()
                _distancetheo, x, y, annote_theo = annotes_theo[0]

                # if theo spot is close enough
                if _distancetheo < closedistance:
                    # print("\nthe nearest theo point is at(%.2f,%.2f)" % (x, y))
                    # print("with info (hkl, other coordinates, energy)", annote_theo)

                    tip_theo = "[h k l]=%s Energy=%.2f keV" % (str(annote_theo[0]), annote_theo[3])
                    if self.datatype == "pixels":
                        tip_theo += "\n(X,Y)=(%.2f,%.2f) (2theta,Chi)=(%.2f,%.2f)" % (
                            x, y, annote_theo[1], annote_theo[2])
                    if self.datatype == "2thetachi":
                        tip_theo += "\n(X,Y)=(%.2f,%.2f) (2theta,Chi)=(%.2f,%.2f)" % (
                            annote_theo[1], annote_theo[2], x, y)
                    self.updateStatusBar(x, y, annote_theo, spottype="theo")

                    # find theo spot index
                    hkl0 = annote_theo[0]
                    #print('hkl0',hkl0)
                    hkls = self.data_theo[2]
                    theoindex = np.where(np.sum(np.hypot(hkls - hkl0, 0), axis=1) < 0.01)[0]
                    #print('theoindex',theoindex)
                    self.highlighttheospot = theoindex
                    hklstr = '[h,k,l]=[%d,%d,%d]'%(annote_theo[0][0], annote_theo[0][1], annote_theo[0][2])
                    print('theo spot index : %d, '%theoindex + hklstr + ' X,Y=(%.2f,%.2f) Energy=%.3f keV'%(annote_theo[1], annote_theo[2], annote_theo[3]))
                else:
                    self.sb.SetStatusText("", 0)
                    tip_theo = ""
                    collisionFound_theo = False
                    self.highlighttheospot = None
                    self._replot()

            if collisionFound_exp or collisionFound_theo:
                if tip_exp is not "":
                    fulltip = tip_exp + "\n" + tip_theo
                else:
                    fulltip = tip_theo

                self.tooltip.SetTip(tip + "\n" + fulltip)
                self.tooltip.Enable(True)

                self._replot()
                return

        if not collisionFound_exp and not collisionFound_theo:
            self.tooltip.SetTip("")
            # set to False to avoid blocked tooltips from btns and checkboxes on windows platforms

    #             self.tooltip.Enable(False)

    # --- ---------Spot Links Editor
    def BuildDataDict(self, _):  # filter Exp Data spots
        """
        in Plot_RefineFrame class

        Filter Exp. Data Button

        Filter exp. spots data to be displayed
        """

        C0 = self.selectedAbsoluteSpotIndices_init
        AllDataToIndex = self.IndexationParameters["AllDataToIndex"]

        C1 = 2.0 * AllDataToIndex["data_theta"][C0]
        C2 = AllDataToIndex["data_chi"][C0]
        C3 = AllDataToIndex["data_I"][C0]
        C4 = AllDataToIndex["data_pixX"][C0]
        C5 = AllDataToIndex["data_pixY"][C0]

        if self.datatype is "2thetachi":
            fields = ["Spot index", "2theta", "Chi", "Intensity"]

            to_put_in_dict = C0, C1, C2, C3

        if self.datatype is "pixels":
            fields = ["Spot index", "pixelX", "pixelY", "Intensity", "2Theta", "Chi"]

            to_put_in_dict = C0, C4, C5, C3, C1, C2

        mySpotData = {}
        for k, ff in enumerate(fields):
            mySpotData[ff] = to_put_in_dict[k]
        dia = LSEditor.SpotsEditor(None,
                                    -1,
                                    "Filter Experimental Spots Data",
                                    mySpotData,
                                    func_to_call=self.readdata_fromEditor,
                                    field_name_and_order=fields)
        dia.Show(True)

    def readdata_fromEditor(self, data):
        """
        update exp. spots data according to the user selected filter
        """
        selectedSpotsPropsarray = data.T

        col0 = selectedSpotsPropsarray[0]
        col1, col2, col3 = selectedSpotsPropsarray[1:4]

        self.selectedAbsoluteSpotIndices = np.array(col0, dtype=np.int)

        if self.datatype is "2thetachi":

            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            # pixelX pixelY
            self.data_XY = (
                self.IndexationParameters["AllDataToIndex"]["data_pixX"][self.selectedAbsoluteSpotIndices],
                self.IndexationParameters["AllDataToIndex"]["data_pixY"][self.selectedAbsoluteSpotIndices])
            self.pixelX, self.pixelY = self.data_XY

            print("\n****SELECTED and DISPLAYED PART OF EXPERIMENTAL SPOTS\n")
        if self.datatype is "pixels":
            col4, col5 = selectedSpotsPropsarray[4: 6]
            self.data_2thetachi = col4, col5
            self.tth, self.chi = col4, col5
            self.Data_I = col3
            self.data_XY = (self.IndexationParameters["AllDataToIndex"]["data_pixX"][self.selectedAbsoluteSpotIndices],
                self.IndexationParameters["AllDataToIndex"]["data_pixY"][self.selectedAbsoluteSpotIndices])
            self.pixelX, self.pixelY = self.data_XY

        # update experimental spots display
        self._replot()

    def OnAutoLink(self, _):
        r""" create automatically links between currently DISPLAYED close experimental and
        theoretical spots in 2theta, chi representation

        .. todo::
            TODO: use a similar autolink function with indexingspotsset
        """
        if self.SimulParam is None:
            wx.MessageBox("There is not simulated data to plot and match with experimental data !!",
                "INFO")
            return
        print("\n ******  Matching Theo. and Exp. spots ***********\n")

        veryclose_angletol = float(self.matr_ctrl.GetValue())  # in degrees
        # ---------  theoretical data
        # (twicetheta, chi, Miller_ind, posx, posy, energy) = self.Simulate_Pattern()
        (twicetheta, chi, Miller_ind, _, _, energy) = self.Simulate_Pattern()

        # selected exp. spots ------------------
        # starting from(may be filtered) experimental data
        if self.datatype == "2thetachi":
            #             twicetheta_exp, chi_exp, dataintensity_exp = self.data[:3]
            twicetheta_exp, chi_exp, dataintensity_exp = self.tth, self.chi, self.Data_I

            #             print "twicetheta_exp",twicetheta_exp
            print("nb of spots in OnAutoLink", len(twicetheta_exp))
        elif self.datatype == "pixels":
            twicetheta_exp, chi_exp = self.tth, self.chi
            dataintensity_exp = self.Data_I

        print("Nb of exp. spots", len(twicetheta_exp))
        # print("twicetheta_exp", twicetheta_exp)
        # print("chi_exp", chi_exp)
        # print("theo 2theta", twicetheta)
        # toc = []
        # for k,val in enumerate(Miller_ind):
        # toc.append([k,val])

        # print "toc", toc
        # print "exp", twicetheta_exp, chi_exp
        Resi, ProxTable = matchingrate.getProximity(
                            np.array([twicetheta, chi]),  # warning array(2theta, chi)
                            twicetheta_exp / 2.0,
                            chi_exp,  # warning theta, chi for exp
                            proxtable=1,
                            angtol=5.0,
                            verbose=0,
                            signchi=1)[:2]  # sign of chi is +1 when apparently SIGN_OF_GAMMA=1

        # len(Resi) = nb of theo spots
        # len(ProxTable) = nb of theo spots
        # ProxTable[index_theo]  = index_exp   closest link
        #print("Resi", Resi)
        # print("ProxTable",ProxTable)
        # print("Nb of theo spots", len(ProxTable))

        # array theo spot index
        very_close_ind = np.where(Resi < veryclose_angletol)[0]
        # print "In OnLinkSpotsAutomatic() very close indices",very_close_ind
        nb_very_close = len(very_close_ind)

        List_Exp_spot_close = []
        Miller_Exp_spot = []
        Energy_Exp_spot = []
        # todisplay = ''
        if nb_very_close > 0:
            for theospot_ind in very_close_ind:  # loop over theo spots index

                List_Exp_spot_close.append(ProxTable[theospot_ind])
                Miller_Exp_spot.append(Miller_ind[theospot_ind])
                Energy_Exp_spot.append(energy[theospot_ind])

        # print("List_Exp_spot_close", List_Exp_spot_close)
        # print("Miller_Exp_spot", Miller_Exp_spot)

        if List_Exp_spot_close == []:
            wx.MessageBox("No links have been found for tolerance angle : %.2f deg" % veryclose_angletol,
                                    "INFO")
            return

        # removing exp spot which appears many times(close to several simulated spots of one grain)--------------
        arrayLESC = np.array(List_Exp_spot_close, dtype=float)

        sorted_LESC = np.sort(arrayLESC)

        diff_index = sorted_LESC - np.array(list(sorted_LESC[1:]) + [sorted_LESC[0]])
        toremoveindex = np.where(diff_index == 0)[0]

        # print "List_Exp_spot_close", List_Exp_spot_close
        # print "sorted_LESC", sorted_LESC
        # print "toremoveindex", toremoveindex

        # print "number labelled exp spots", len(List_Exp_spot_close)
        # print "List_Exp_spot_close", List_Exp_spot_close
        # print "Miller_Exp_spot", Miller_Exp_spot

        if len(toremoveindex) > 0:
            # index of exp spot in arrayLESC that are duplicated
            ambiguous_exp_ind = GT.find_closest(
                np.array(sorted_LESC[toremoveindex], dtype=float), arrayLESC, 0.1)[1]
            # print "ambiguous_exp_ind", ambiguous_exp_ind

            # marking exp spots(belonging ambiguously to several simulated grains)
            for ind in ambiguous_exp_ind:
                Miller_Exp_spot[ind] = None
                Energy_Exp_spot[ind] = 0.0

        # -----------------------------------------------------------------------------------------------------
        ProxTablecopy = copy.copy(ProxTable)
        # tag duplicates in ProxTable with negative sign ----------------------
        # ProxTable[index_theo]  = index_exp   closest link

        #ProxTable = list of   theo_ind, exp_ind
        for _, exp_ind in enumerate(ProxTable):
            where_th_ind = np.where(ProxTablecopy == exp_ind)[0]
            # print "theo_ind, exp_ind ******** ",theo_ind, exp_ind
            if len(where_th_ind) > 1:
                # exp spot(exp_ind) is close to several theo spots
                # then tag the index with negative sign
                for indy in where_th_ind:
                    ProxTablecopy[indy] = -ProxTable[indy]
                # except that which corresponds to the closest
                closest = np.argmin(Resi[where_th_ind])
                # print "residues = Resi[where_th_ind]",Resi[where_th_ind]
                # print "closest",closest
                # print "where_exp_ind[closest]",where_th_ind[closest]
                # print "Resi[where_th_ind[closest]]", Resi[where_th_ind[closest]]
                ProxTablecopy[where_th_ind[closest]] = -ProxTablecopy[where_th_ind[closest]]
        # ------------------------------------------------------------------
        # print "ProxTable after duplicate removal tagging"
        # print ProxTablecopy

        #         print "len List_Exp_spot_close", len(List_Exp_spot_close)
        # print "Results",[Miller_Exp_spot, List_Exp_spot_close]

        singleindices = []
        refine_indexed_spots = {}

        # loop over close exp. spots
        for k in range(len(List_Exp_spot_close)):

            exp_index = List_Exp_spot_close[k]
            if not singleindices.count(exp_index):
                # there is not exp_index in singleindices
                singleindices.append(exp_index)

                theo_index = np.where(ProxTablecopy == exp_index)[0]

                # print "exp_index", exp_index
                # print "theo_index", theo_index

                if len(theo_index) == 1:  # only one theo spot close to the current exp. spot
                    # print "add in dict refine_indexed_spots\n"
                    refine_indexed_spots[exp_index] = [exp_index,
                                                        theo_index,
                                                        Miller_Exp_spot[k]]
                # else:  # test whether all theo spots are harmonics
                # ar_miller = np.take(Miller_ind, theo_index, axis =0)
                # print "ar_miller",ar_miller
                # filtered_miller = FindO.FilterHarmonics(ar_miller)
                # if len(filtered_miller) == 1:
                # refine_indexed_spots[exp_index] = [exp_index,theo_index,Miller_Exp_spot[k]]
                else:  # recent PATCH:
                    # print "Resi[theo_index]", Resi[theo_index]
                    closest_theo_ind = np.argmin(Resi[theo_index])
                    # print theo_index[closest_theo_ind]
                    if Resi[theo_index][closest_theo_ind] < veryclose_angletol:
                        refine_indexed_spots[exp_index] = [exp_index,
                                                        theo_index[closest_theo_ind],
                                                        Miller_Exp_spot[k]]
            else:
                print("Experimental spot #%d may belong to several theo. spots!" % exp_index)

        # find theo spot linked to exp spot ---------------------------------

        # refine_indexed_spots is a dictionary:
        # key is experimental spot index and value is [exp. spotindex, h, k, l]
        # print "refine_indexed_spots",refine_indexed_spots

        listofpairs = []
        linkExpMiller = []
        linkIntensity = []
        linkResidues = []
        # Dataxy = []

        print('self.selectedAbsoluteSpotIndices', self.selectedAbsoluteSpotIndices)
        print('refine_indexed_spots', refine_indexed_spots)

        for val in list(refine_indexed_spots.values()):
            if val[2] is not None:
                localspotindex = val[0]
                if not isinstance(val[1], (list, np.ndarray)):
                    closetheoindex = val[1]
                else:
                    closetheoindex = val[1][0]
                # print('localspotindex',localspotindex)
                absolute_spot_index = self.selectedAbsoluteSpotIndices[localspotindex]

                listofpairs.append([absolute_spot_index, closetheoindex])  # Exp, Theo,  where -1 for specifying that it came from automatic linking
                linkExpMiller.append([float(absolute_spot_index)] + [float(elem) for elem in val[2]])  # float(val) for further handling as floats array
                linkIntensity.append(dataintensity_exp[localspotindex])
                linkResidues.append([absolute_spot_index, closetheoindex, Resi[closetheoindex]])
                # Dataxy.append([ LaueToolsframe.data_pixX[val[0]], LaueToolsframe.data_pixY[val[0]]])


        self.linkedspots_link = np.array(listofpairs)
        self.linkExpMiller_link = linkExpMiller
        self.linkIntensity_link = linkIntensity
        self.linkResidues_link = linkResidues
        self.Energy_Exp_spot = Energy_Exp_spot
        self.fields = ["#Spot Exp", "#Spot Theo", "h", "k", "l", "Intensity", "residues(deg)"]

        print("Nb of links between exp. and theo. spots  : ", len(self.linkedspots_link))

        self.plotlinks = self.linkedspots_link
        self._replot()

        return refine_indexed_spots

    def BuildDataDictAfterLinks(self, _):  # filter links between spots(after OnAutoLink() )
        """
        open editor to look at spots links and filter them
        button Filter Links
        """
        if self.linkedspots_link is not None:

            indExp = np.array(self.linkedspots_link[:, 0], dtype=np.int)
            indTheo = np.array(self.linkedspots_link[:, 1], dtype=np.int)
            _h, _k, _l = np.transpose(np.array(self.linkExpMiller_link, dtype=np.int))[1:4]
            intens = self.linkIntensity_link
            if self.linkResidues_link is not None:
                residues = np.array(self.linkResidues_link)[:, 2]
            else:
                residues = -1 * np.ones(len(indExp))

            to_put_in_dict = indExp, indTheo, _h, _k, _l, intens, residues

            mySpotData = {}
            for k, ff in enumerate(self.fields):
                mySpotData[ff] = to_put_in_dict[k]
            dia = LSEditor.SpotsEditor(None, -1, "Filter spots links Editor", mySpotData,
                                    func_to_call=self.readdata_fromEditor_after,
                                    field_name_and_order=self.fields)
            dia.Show(True)

        else:
            wx.MessageBox('There are not existing links between simulated and experimental '
            'data!! Click on "Auto Links" button ', "INFO")

    def readdata_fromEditor_after(self, data):
        """ set self.linkedspots, self.linkExpMiller, self.linkIntensity, self.linkResidues
        from data array"""
        ArrayReturn = np.array(data)

        self.linkedspots = ArrayReturn[:, :2]
        self.linkExpMiller = np.take(ArrayReturn, [0, 2, 3, 4], axis=1)
        self.linkIntensity = ArrayReturn[:, 5]
        self.linkResidues = np.take(ArrayReturn, [0, 1, 6], axis=1)

        self.plotlinks = self.linkedspots
        self._replot()

    # --- ------------ Fitting functions ----
    def OnRefine_UB_and_Strain(self, _):
        """
        in plot_RefineFrame

        Note: only strain and orientation simultaneously
        # TODO: fit only a set of parameters, useful
        NOTE: refine strained and oriented crystal by minimizing peaks positions (pixels).
        NOTE: experimental peaks pixel positions are reevaluated from 2theta and chi angles
        """

        if self.use_forfit1.GetValue():
            if self.linkedspots_link is None:
                wx.MessageBox('There are not existing links between simulated and experimental '
                'data for the refinement!! Click on "Auto Links" button ', "INFO")
                return

            if len(self.linkedspots_link) < 8:
                wx.MessageBox(
                    "You have only %d over the 8 links needed between exp. and theo. spots to "
                    "refine orientation and strain" % len(self.linkedspots_link), "INFO")
                return
            self.linkedspots_fit = self.linkedspots_link
            self.linkExpMiller_fit = self.linkExpMiller_link
            self.linkIntensity_fit = self.linkIntensity_link
            self.linkResidues_fit = None
        elif self.use_forfit2.GetValue():
            if self.linkedspots == []:
                txt = "There are not existing links between simulated and experimental data for the refinement!!\n"
                txt += 'Click first on "Auto Links" button and then filter and select data for the refinement'
                wx.MessageBox(txt, "INFO")
                return
            self.linkedspots_fit = self.linkedspots
            self.linkExpMiller_fit = self.linkExpMiller
            self.linkIntensity_fit = self.linkIntensity
            self.linkResidues_fit = None
        elif self.use_forfit3.GetValue():
            print("I will use for the refinement the(filtered) results of the previous fit")
            if self.linkedspots_fit is None:
                wx.MessageBox(
                    "You need to refine once for starting the refinement from previous fit(filtered) results",
                    "INFO")
                return

        print("\nStarting fit of strain and orientation from spots links ...\n")
        # print "Pairs of spots used",self.linkedspots
        arraycouples = np.array(self.linkedspots_fit)

        #         print "\n\n***arraycouples", arraycouples

        exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
        sim_indices = np.array(arraycouples[:, 1], dtype=np.int)

        nb_pairs = len(exp_indices)
        print("Nb of pairs: ", nb_pairs)
        # print "exp_indices, sim_indices",exp_indices, sim_indices

        # self.data_theo contains the current simulated spots: twicetheta, chi, Miller_ind, posx, posy
        # Data_Q = self.data_theo[2]  # all miller indices must be entered with sim_indices = arraycouples[:,1]

        # print "self.linkExpMiller",self.linkExpMiller
        Data_Q = np.array(self.linkExpMiller_fit)[:, 1:]
        sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...
        #         print "DataQ from self.linkExpMiller", Data_Q

        # experimental spots selection -------------------------------------
        AllData = self.IndexationParameters["AllDataToIndex"]
        _twth, _chi = (np.take(2.0 * AllData["data_theta"], exp_indices),
                        np.take(AllData["data_chi"], exp_indices))  # 2theta chi coordinates

        pixX = np.take(AllData["data_pixX"], exp_indices)
        pixY = np.take(AllData["data_pixY"], exp_indices)

        if self.sigmanoise not in (0,):
            nbpeaks = len(pixX)

            sigma = self.sigmanoise
            mu = 0
            pixX = pixX + sigma * np.random.randn(nbpeaks) + mu
            pixY = pixY + sigma * np.random.randn(nbpeaks) + mu

        #         starting_orientmatrix = np.array(self.SimulParam[0][2])
        starting_orientmatrix = self.UBmat

        # print "nb_pairs",nb_pairs
        # print "indices of simulated spots(selection in whole Data_Q list)",sim_indices
        # print "Experimental pixX, pixY",pixX, pixY
        print("starting_orientmatrix in OnRefine_UB_and_Strain()", starting_orientmatrix)

        if self.use_weights.GetValue():
            weights = self.linkIntensity_fit
        else:
            weights = None

        results = None
        self.fitresults = False
        self.Tsresults = None
        self.new_latticeparameters = None

        # ----------------------------------
        #  refinement model
        # ----------------------------------
        if self.fitTrecip.GetValue():
            # --------------------------------------------------------
            # fitting procedure refining right distortion of UB
            # q = (Rot x,y,z_refined) UBinit (Trecip_refined) B0 G*
            # (Trecip refined) is operator triangular up: ((1,refined_#,refined_#),(0,refined_22,refined_#),(0,0,refined_33))
            # (Rot x,y,z_refined)  =  3 elementary rotations small angles
            # -------------------------------------------------------

            # starting B0matrix corresponding to the unit cell   -----
            latticeparams = self.dict_Materials[self.key_material][1]
            self.B0matrix = CP.calc_B_RR(latticeparams)
            # -------------------------------------------------------

            # initial distorsion is  1 1 0 0 0  = refined_22,refined_33, 0,0,0
            allparameters = np.array(self.CCDcalib + [1, 1, 0, 0, 0] + [0, 0, 0])

            # change ycen if grain is below the surface (NOT ALONG beam direction (ybeam)):
            # depth is counted positively below surface in microns
            depth = float(self.sampledepthctrl.GetValue())
            depth_along_beam = depth / np.sin(40 * np.pi / 180.)
            delta_ycen = depth_along_beam/1000./self.pixelsize
            allparameters[2] += delta_ycen

            # nspots = np.arange(nb_pairs)
            # miller = Data_Q

            # strain & orient
            initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
            arr_indexvaryingparameters = np.arange(5, 13)

            if self.fitycen.GetValue():
                initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, allparameters[2]])
                arr_indexvaryingparameters = np.append(np.arange(5, 13), 2)

            print("\nInitial error--------------------------------------\n")

            print("initial_values, allparameters, arr_indexvaryingparameters")
            print(initial_values, allparameters, arr_indexvaryingparameters)

            residues, deltamat, _ = FitO.error_function_on_demand_strain(
                                                                    initial_values,
                                                                    Data_Q,
                                                                    allparameters,
                                                                    arr_indexvaryingparameters,
                                                                    sim_indices,
                                                                    pixX,
                                                                    pixY,
                                                                    initrot=starting_orientmatrix,
                                                                    Bmat=self.B0matrix,
                                                                    pureRotation=0,
                                                                    verbose=1,
                                                                    pixelsize=self.pixelsize,
                                                                    dim=self.framedim,
                                                                    weights=weights,
                                                                    kf_direction=self.kf_direction)

            print("mean Initial residues", np.mean(residues))
            print("---------------------------------------------------\n")

            results = FitO.fit_on_demand_strain(initial_values,
                                                    Data_Q,
                                                    allparameters,
                                                    FitO.error_function_on_demand_strain,
                                                    arr_indexvaryingparameters,
                                                    sim_indices,
                                                    pixX,
                                                    pixY,
                                                    initrot=starting_orientmatrix,
                                                    Bmat=self.B0matrix,
                                                    pixelsize=self.pixelsize,
                                                    dim=self.framedim,
                                                    verbose=1,
                                                    weights=weights,
                                                    kf_direction=self.kf_direction)



            print("\n********************\n       Results of Fit        \n********************")
            print("results", results)

            self.fit_completed = False

            if results is None:
                return

            self.fitresults = True

            print("\nFinal error--------------------------------------\n")
            residues, deltamat, refinedUB = FitO.error_function_on_demand_strain(
                                                                    results,
                                                                    Data_Q,
                                                                    allparameters,
                                                                    arr_indexvaryingparameters,
                                                                    sim_indices,
                                                                    pixX,
                                                                    pixY,
                                                                    initrot=starting_orientmatrix,
                                                                    Bmat=self.B0matrix,
                                                                    pureRotation=0,
                                                                    verbose=1,
                                                                    pixelsize=self.pixelsize,
                                                                    dim=self.framedim,
                                                                    weights=weights,
                                                                    kf_direction=self.kf_direction)

            self.residues_non_weighted = FitO.error_function_on_demand_strain(
                                                                results,
                                                                Data_Q,
                                                                allparameters,
                                                                arr_indexvaryingparameters,
                                                                sim_indices,
                                                                pixX,
                                                                pixY,
                                                                initrot=starting_orientmatrix,
                                                                Bmat=self.B0matrix,
                                                                pureRotation=0,
                                                                verbose=1,
                                                                pixelsize=self.pixelsize,
                                                                dim=self.framedim,
                                                                weights=None,
                                                                kf_direction=self.kf_direction)[0]

            print("Final residues", residues)
            print("---------------------------------------------------\n")
            print("mean", np.mean(residues))

            # building B mat
            param_strain_sol = results
            self.varyingstrain = np.array([[1.0, param_strain_sol[2], param_strain_sol[3]],
                                            [0, param_strain_sol[0], param_strain_sol[4]],
                                            [0, 0, param_strain_sol[1]]])
            print("varyingstrain results")
            print(self.varyingstrain)

            if self.fitycen.GetValue():
                print("fitted ycen", param_strain_sol[8])
                print('calib ref. ycen: ', allparameters[2])
                print('delta ycen: ', param_strain_sol[8] - allparameters[2])

            self.newUmat = np.dot(deltamat, starting_orientmatrix)

            # building UBmat(= newmatrix)
            self.newUBmat = np.dot(self.newUmat, self.varyingstrain)
            print("newUBmat", self.newUBmat)
            print("refinedUB", refinedUB)

            self.constantlength = "a"

        elif (self.fita.GetValue() or self.fitb.GetValue() or self.fitc.GetValue()
            or self.fitalpha.GetValue() or self.fitbeta.GetValue() or self.fitgamma.GetValue()):
            # --------------------------------------------------
            # ---  lattice parameters refinement
            # --------------------------------------------------
            print("starting_orientmatrix in plotrefineGUI before fit_lattice_parameters",
                starting_orientmatrix)
            self.fit_lattice_parameters(pixX, pixY, Data_Q, starting_orientmatrix, self.key_material, weights)

        else:
            # --------------------------------------------------
            # ---  transform matrix in sample frame elements refinement
            # --------------------------------------------------
            self.fit_transform_parameters(pixX, pixY, Data_Q,
                                                starting_orientmatrix, self.key_material, weights)

        # ---------------------------------------------------------------
        # postprocessing of unit cell orientation and strain refinement
        # ---------------------------------------------------------------
        print("self.newUBmat after fitting", self.newUBmat)
        self.evaluate_strain_display_results(self.newUBmat,
                                            self.key_material,
                                            self.residues_non_weighted,
                                            nb_pairs,
                                            constantlength=self.constantlength,
                                            Tsresults=self.Tsresults)


        # ---------------------------------------------
        # treatment of results of GUI
        # ---------------------------------------------

        # update linked spots with residues
        self.linkResidues_fit = np.array([exp_indices, sim_indices, self.residues_non_weighted]).T

        # saving previous unit cell strain and orientation
        self.previous_UBmat = copy.copy(self.UBmat)
        self.previous_Umat = copy.copy(self.Umat)
        self.previous_Bmat = copy.copy(self.Bmat)

        # for further use
        self.UBmat = self.newUBmat
        self.Umat = self.newUmat
        self.Bmat = self.varyingstrain  # strain in reciprocal space

        self.UBB0mat = np.dot(self.newUBmat, self.B0matrix)

        print("self.UBmat", self.UBmat)
        print("self.Umat = newUmat", self.Umat)
        print("Umat2 = ", self.Umat2)
        print("self.Bmat", self.Bmat)
        print("self.Bmat_tri", self.Bmat_tri)
        print("self.HKLxyz_names", self.HKLxyz_names)
        print("self.HKLxyz", self.HKLxyz)
        print("B0matrix", self.B0matrix)
        print("self.UBB0mat = np.dot(newUBmat, B0matrix)", self.UBB0mat)
        print("np.dot(newUBmat, B0matrix)", np.dot(self.newUBmat, self.B0matrix))
        print("deviatoric strain", self.deviatoricstrain)
        print("deviatoric strain sample frame", self.deviatoricstrain_sampleframe)

        # update dictionnary of UB matrices
        DictLT.dict_Rot["Last Refined UB matrix %d" % self.fitcounterindex] = copy.copy(
            self.newUBmat)
        self.comboUBmatrix.Append("Last Refined UB matrix %d" % self.fitcounterindex)

        self.fitcounterindex += 1

        self.fit_completed = True

    def evaluate_strain_display_results(self,
                                        newUBmat,
                                        key_material,
                                        residues_non_weighted,
                                        nb_pairs,
                                        constantlength="a",
                                        Tsresults=None):
        """
        evaluate strain and display fitting results
        """
        # compute new lattice parameters  -----
        latticeparams = self.dict_Materials[key_material][1]
        B0matrix = CP.calc_B_RR(latticeparams)

        UBmat = copy.copy(newUBmat)

        (devstrain, lattice_parameter_direct_strain) = CP.compute_deviatoricstrain(
                                                                UBmat, B0matrix, latticeparams)
        # overwrite and rescale possibly lattice lengthes
        lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(
                                                            UBmat, key_material, constantlength,
                                                            dictmaterials=self.dict_Materials)

        print("final lattice_parameter_direct_strain", lattice_parameter_direct_strain)

        deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(
                                                                        devstrain, UBmat)

        devstrain_sampleframe_round = np.round(deviatoricstrain_sampleframe * 1000, decimals=3)
        devstrain_round = np.round(devstrain * 1000, decimals=3)

        self.new_latticeparameters = lattice_parameter_direct_strain
        self.deviatoricstrain = devstrain
        self.deviatoricstrain_sampleframe = deviatoricstrain_sampleframe

        # TODO: to complete ---------------------
        # devstrain_crystal_voigt = np.take(np.ravel(np.array(devstrain)), (0, 4, 8, 5, 2, 1))

        self.UBB0mat = np.dot(self.newUBmat, B0matrix)

        Umat = None

        Umat = CP.matstarlab_to_matstarlabOND(matstarlab=None, matLT3x3=np.array(self.UBmat))
        # TODO to be translated !----------------------
        # conversion en np array necessaire apres automatic indexation, pas necessaire apres check orientation

        print("**********test U ****************************")
        print("U matrix = ")
        print(Umat.round(decimals=9))
        print("norms :")
        for i in range(3):
            print(i, GT.norme_vec(Umat[:, i]).round(decimals=5))
        print("scalar products")
        for i in range(3):
            j = np.mod(i + 1, 3)
            print(i, j, np.inner(Umat[:, i], Umat[:, j]).round(decimals=5))
        print("determinant")
        print(np.linalg.det(Umat).round(decimals=5))

        Bmat_triang_up = np.dot(np.transpose(Umat), self.UBmat)

        print(" Bmat_triang_up= ")
        print(Bmat_triang_up.round(decimals=9))

        self.Umat2 = Umat
        self.Bmat_tri = Bmat_triang_up

        (list_HKL_names,
        HKL_xyz) = CP.matrix_to_HKLs_along_xyz_sample_and_along_xyz_lab(
                                                        matstarlab=None,  # OR
                                                        UBmat=self.UBB0mat,  # LT , UBB0 ici
                                                        omega=None,  # was MG.PAR.omega_sample_frame,
                                                        mat_from_lab_to_sample_frame=None,
                                                        results_in_OR_frames=0,
                                                        results_in_LT_frames=1,
                                                        sampletilt=40.0)
        self.HKLxyz_names = list_HKL_names
        self.HKLxyz = HKL_xyz

        texts_dict = {}

        txt0 = "Filename: %s\t\t\tDate: %s\t\tPlotRefineGUI.py\n" % (self.DataPlot_filename,
                                                                    time.asctime())
        txt0 += "Mean Pixel Deviation: %.3f\n" % np.mean(residues_non_weighted)
        txt0 += "Number of refined Laue spots: %d\n" % nb_pairs
        texts_dict["NbspotsResidues"] = txt0

        txt1 = "Deviatoric Strain (10-3 units) in crystal frame (direct space) \n"
        for k in range(3):
            txt1 += "%.3f   %.3f   %.3f\n" % tuple(devstrain_round[k])
        texts_dict["devstrain_crystal"] = txt1

        txt2 = "Deviatoric Strain (10-3 units) in sample frame (tilt=40deg)\n"
        for k in range(3):
            txt2 += "%.3f   %.3f   %.3f\n" % tuple(devstrain_sampleframe_round[k])
        texts_dict["devstrain_sample"] = txt2

        #         txt3 = 'Full Strain (10-3 units) sample frame (tilt=40deg)\n'
        #         txt3 += 'Assumption: cubic material + stress33=0\n'
        #         for k in range(3):
        #             txt3 += '%.3f   %.3f   %.3f\n' % tuple(fullstrain_round[k])
        txt3 = ""
        texts_dict["fullstrain_sample"] = txt3

        txtinitlattice = "Initial Lattice Parameters\n"
        paramcellname = ["  a", "  b", "  c", "alpha", "beta", "gamma"]
        for name, val in zip(paramcellname, latticeparams):
            txtinitlattice += "%s\t\t%.6f\n" % (name, val)

        texts_dict["Initial lattice"] = txtinitlattice

        txtfinallattice = "Refined Lattice Parameters\n"
        paramcellname = ["  a", "  b", "  c", "alpha", "beta", "gamma"]
        for name, val in zip(paramcellname, lattice_parameter_direct_strain):
            txtfinallattice += "%s\t\t%.6f\n" % (name, val)

        texts_dict["Refined lattice"] = txtfinallattice

        if Tsresults is not None:

            Ts = Tsresults[2]

            Tsvalues = [Ts[0, 0], Ts[0, 1], Ts[0, 2], Ts[1, 1], Ts[1, 2], Ts[2, 2]]

            Tsmat = [[Ts[0, 0], Ts[0, 1], Ts[0, 2]],
                    [0.0, Ts[1, 1], Ts[1, 2]],
                    [0.0, 0.0, Ts[2, 2]]]

            txtTs = "Final Transform Parameters\n"
            transformelements = ["  Ts00", "  Ts01", "  Ts02", " Ts11", " Ts12", " Ts22"]
            for name in transformelements:
                txtTs += name
            txtTs += " = ["
            for val in Tsvalues:
                txtTs += "%.6f, " % val
            txtTs = txtTs[:-2] + "]\n"
            txtTs += "Ts matrix in q = P Ts P-1 U B0 G*\n"
            txtTs += "["
            for k in range(3):
                txtTs += "[%.8f, %.8f, %.8f],\n" % tuple(Tsmat[k])
            #             for name, val in zip(transformelements, Tsvalues):
            #                 txtTs += '%s\t\t%.6f\n' % (name, val)

            texts_dict["Ts parameters"] = txtTs[:-2] + "]"

        txtUB = "UB matrix in q = UB B0 G*\n"
        txtUB += "["
        for k in range(3):
            txtUB += "[%.8f, %.8f, %.8f],\n" % tuple(UBmat[k])
        texts_dict["UBmatrix"] = txtUB[:-2] + "]"

        txtB0 = "B0 matrix in q = UB B0 G*\n"
        txtB0 += "["
        for k in range(3):
            txtB0 += "[%.8f, %.8f, %.8f],\n" % tuple(B0matrix[k])
        texts_dict["B0matrix"] = txtB0[:-2] + "]"

        txtHKLxyz_names = "                                 HKL frame coordinates\n"
        listvectors = ["x=[100]_LT :",
                        "y=[010]_LT :",
                        "z=[001]_LT :",
                        "xs=[100]_LTsample :",
                        "ys=[010]_LTsample :",
                        "zs=[001]_LTsample :"]
        for k in range(6):
            txtHKLxyz_names += listvectors[k] + "\t [%.3f, %.3f, %.3f]\n" % tuple(
                self.HKLxyz[k])
            texts_dict["HKLxyz_names"] = txtHKLxyz_names

        print(txtHKLxyz_names)

        # if 0:
        #     txtHKLxyz = "HKL = \n"
        #     txtHKLxyz += "["
        #     for k in range(6):
        #         txtHKLxyz += "[%.3f, %.3f, %.3f],\n" % tuple(self.HKLxyz[k])
        #         texts_dict["HKLxyz"] = txtHKLxyz[:-2] + "]"
        #     print(txtHKLxyz)
        texts_dict["HKLxyz"] = ""

        frb = FitResultsBoard(self, -1, "REFINEMENT RESULTS", texts_dict)
        frb.ShowModal()

        frb.Destroy()

    def fit_transform_parameters(self, pixX, pixY, hkls, starting_orientmatrix,
                                                                            key_material, weights):
        """ refine and find the best transform to match pixX and pixY
        """

        self.selectFittingTransformParameters(key_material)

        pureUmatrix, _ = GT.UBdecomposition_RRPP(starting_orientmatrix)

        print("self.fitting_parameters_values, self.fitting_parameters_keys")
        print(self.fitting_parameters_values, self.fitting_parameters_keys)
        print("len(allparameters)", len(self.allparameters))
        print("starting_orientmatrix", starting_orientmatrix)

        absolutespotsindices = np.arange(len(pixX))

        print("initial errors")
        print(FitO.error_function_strain(self.fitting_parameters_values,
                                self.fitting_parameters_keys,
                                hkls,
                                self.allparameters,
                                absolutespotsindices,
                                pixX,
                                pixY,
                                initrot=pureUmatrix,
                                B0matrix=self.B0matrix,
                                pureRotation=0,
                                verbose=1,
                                pixelsize=self.pixelsize,
                                dim=self.framedim,
                                weights=None,
                                kf_direction=self.kf_direction,
                                returnalldata=True))

        print("\n\n*** fit_transform_parameters (PlotRefineGUI.py) *****\n")
        print("fitting_parameters_values", self.fitting_parameters_values)
        print("fitting_parameters_keys", self.fitting_parameters_keys)
        print("hkls", hkls)
        print("allparameters", self.allparameters)
        print("absolutespotsindices", absolutespotsindices)
        print("pixX", pixX)
        print("pixY", pixY)
        print("UBmatrix_start", pureUmatrix)
        print("B0matrix", self.B0matrix)
        print("self.pixelsize", self.pixelsize)
        print("self.dim", self.framedim)
        print("weights", weights)

        print("fitting strain parameters")
        results = FitO.fit_function_strain(self.fitting_parameters_values,
                                            self.fitting_parameters_keys,
                                            hkls,
                                            self.allparameters,
                                            absolutespotsindices,
                                            pixX,
                                            pixY,
                                            UBmatrix_start=pureUmatrix,
                                            B0matrix=self.B0matrix,
                                            nb_grains=1,
                                            pureRotation=0,
                                            verbose=0,
                                            pixelsize=self.pixelsize,
                                            dim=self.framedim,
                                            weights=weights,
                                            kf_direction=self.kf_direction)

        self.fit_completed = False
        print("results", results)


        if results is None:
            return

        self.fitresults = True
        print("\nFinal errors--------------------------------------\n")
        # alldistances_array, Uxyz, newmatrix, Ts, T
        (residues, Uxyz, newUmat, refinedTs, refinedT) = FitO.error_function_strain(
                                                                    results,
                                                                    self.fitting_parameters_keys,
                                                                    hkls,
                                                                    self.allparameters,
                                                                    absolutespotsindices,
                                                                    pixX,
                                                                    pixY,
                                                                    initrot=pureUmatrix,
                                                                    B0matrix=self.B0matrix,
                                                                    pureRotation=0,
                                                                    verbose=0,
                                                                    pixelsize=self.pixelsize,
                                                                    dim=self.framedim,
                                                                    weights=weights,
                                                                    kf_direction=self.kf_direction,
                                                                    returnalldata=True)

        self.residues_non_weighted = FitO.error_function_strain(results,
                                                                self.fitting_parameters_keys,
                                                                hkls,
                                                                self.allparameters,
                                                                absolutespotsindices,
                                                                pixX,
                                                                pixY,
                                                                initrot=pureUmatrix,
                                                                B0matrix=self.B0matrix,
                                                                pureRotation=0,
                                                                verbose=0,
                                                                pixelsize=self.pixelsize,
                                                                dim=self.framedim,
                                                                weights=None,
                                                                kf_direction=self.kf_direction,
                                                                returnalldata=False)

        print("Final pixel residues", residues)
        print("---------------------------------------------------\n")
        print("Final mean pixel residues", np.mean(residues))

        # q = T Uxyz U B0init G*
        # q = Tnew newU B0init G*
        # q = P Tsnew P-1 newU B0init G*
        # q = newUB B0init G*
        self.newUBmat = newUmat

        Umat_init = pureUmatrix

        print("in q = T Uxyz U B0init G*")
        print("q = T newU B0init G*")
        print("q = P Tsnew P-1 newU B0init G*")
        print("q = newUB B0init G*")
        print("in q = newUmat varyingstrain B0init G*")
        print("Uxyz, Umat_init, newUmat, refinedT,refinedTs")
        print(Uxyz, Umat_init, newUmat, refinedT, refinedTs)
        print("in q = newUBmat B0init G*")
        print("newUBmat, B0matrix_init")
        print(self.newUBmat, self.B0matrix)

        self.newUmat = newUmat
        self.varyingstrain = np.dot(np.linalg.inv(self.newUmat), self.newUBmat)

        self.Tsresults = ("triangular", "sampleframe", refinedTs)

    def fit_lattice_parameters(
        self, pixX, pixY, hkls, starting_orientmatrix, key_material, weights):
        """refine lattice parameters
        """

        self.selectFittingLatticeParameters(key_material)

        print("self.kf_direction in fit_lattice_parameters", self.kf_direction)

        #pureUmatrix, residualdistortion = GT.UBdecomposition_RRPP(starting_orientmatrix)
        pureUmatrix, _ = GT.UBdecomposition_RRPP(starting_orientmatrix)


        print("self.fitting_parameters_values, self.fitting_parameters_keys")
        print(self.fitting_parameters_values, self.fitting_parameters_keys)

        print("len(allparameters)", len(self.allparameters))
        print("starting_orientmatrix", starting_orientmatrix)
        print("pureUmatrix", pureUmatrix)

        absolutespotsindices = np.arange(len(pixX))

        print("\n\n -**--  Using fit_lattice_parameters() ----**--")
        print("\n----------- Initial errors ------------------")
        print(FitO.error_function_latticeparameters(self.fitting_parameters_values,
                                                    self.fitting_parameters_keys,
                                                    hkls,
                                                    self.allparameters,
                                                    absolutespotsindices,
                                                    pixX,
                                                    pixY,
                                                    initrot=pureUmatrix,
                                                    pureRotation=0,
                                                    verbose=1,
                                                    pixelsize=self.pixelsize,
                                                    dim=self.framedim,
                                                    weights=None,
                                                    kf_direction=self.kf_direction,
                                                    returnalldata=True))

        print("\n--- Now fitting lattice parameters ---------")
        results = FitO.fit_function_latticeparameters(self.fitting_parameters_values,
                                                        self.fitting_parameters_keys,
                                                        hkls,
                                                        self.allparameters,
                                                        absolutespotsindices,
                                                        pixX,
                                                        pixY,
                                                        UBmatrix_start=pureUmatrix,
                                                        nb_grains=1,
                                                        pureRotation=0,
                                                        verbose=0,
                                                        pixelsize=self.pixelsize,
                                                        dim=self.framedim,
                                                        weights=weights,
                                                        kf_direction=self.kf_direction)

        self.fit_completed = False
        print("res", results)

        #             print 'self.CCDcalib', self.CCDcalib
        #             print 'pixelsize', self.pixelsize
        #             print 'pixX', pixX.tolist()
        #             print 'pixY', pixY.tolist()
        #             print 'Data_Q', Data_Q.tolist()
        #             print 'starting_orientmatrix', starting_orientmatrix
        #
        #             print 'self.B0matrix', self.B0matrix.tolist()
        if results is None:
            return

        self.fitresults = True
        print("\n---------Final errors----------------------------\n")

        (residues,
            Uxyz,
            newUmat,
            newB0matrix,
            _,
        ) = FitO.error_function_latticeparameters(results,
                                                self.fitting_parameters_keys,
                                                hkls,
                                                self.allparameters,
                                                absolutespotsindices,
                                                pixX,
                                                pixY,
                                                initrot=pureUmatrix,
                                                pureRotation=0,
                                                verbose=0,
                                                pixelsize=self.pixelsize,
                                                dim=self.framedim,
                                                weights=weights,
                                                kf_direction=self.kf_direction,
                                                returnalldata=True)

        self.residues_non_weighted = FitO.error_function_latticeparameters(results,
                                                                            self.fitting_parameters_keys,
                                                                            hkls,
                                                                            self.allparameters,
                                                                            absolutespotsindices,
                                                                            pixX,
                                                                            pixY,
                                                                            initrot=pureUmatrix,
                                                                            pureRotation=0,
                                                                            verbose=0,
                                                                            pixelsize=self.pixelsize,
                                                                            dim=self.framedim,
                                                                            weights=None,
                                                                            kf_direction=self.kf_direction,
                                                                            returnalldata=False)

        print("Final pixel all residues", residues)
        print("---------------------------------------------------\n")
        print("Final mean pixel residues", np.mean(residues))

        # q = Uxyz U newB0 G*
        # q = UB B0init G*
        self.newUBmat = np.dot(np.dot(newUmat, newB0matrix), np.linalg.inv(self.B0matrix))

        Umat_init = pureUmatrix

        print("in q = Uxyz Umat_init newB0matrix G*")
        print("q= newUmat newB0matrix G*")
        print("Uxyz, Umat_init, newUmat, newB0matrix")
        print(Uxyz, Umat_init, newUmat, newB0matrix)
        print("in q = newUBmat B0matrix_init G*")
        print("newUBmat, B0matrix_init")
        print(self.newUBmat, self.B0matrix)
        print("in q = newUmat varyingstrain B0matrix_init G*")
        print("newUBmat, B0matrix_init")

        self.newUmat = newUmat
        self.varyingstrain = np.dot(newB0matrix, np.linalg.inv(self.B0matrix))

    def selectFittingTransformParameters(self, key_material):
        """ set structural parameters with transform model to be refined
        set self.fitting_parameters_keys, self.fitting_parameters_values
        according to checkboxes of fitting parameters"""
        # fixed parameters (must be an array)

        # triangular up transform in sample frame
        TransformParameters = [1, 0, 0, 1, 0, 1.0]
        # corresponding to Identity matrix

        latticeparameters = self.dict_Materials[key_material][1]
        self.B0matrix = CP.calc_B_RR(latticeparameters)

        self.allparameters = np.array(self.CCDcalib + [0, 0, 0] + TransformParameters)

        if (self.Ts00chck.GetValue()
            and self.Ts01chck.GetValue()
            and self.Ts02chck.GetValue()
            and self.Ts11chck.GetValue()
            and self.Ts12chck.GetValue()
            and self.Ts22chck.GetValue()):
            self.Ts00chck.SetValue(False)

        List_of_defaultvalues = np.array(
            self.CCDcalib + [0, 0, 0] + TransformParameters)
        List_of_checkwidgets = [self.fitorient1,
                                self.fitorient2,
                                self.fitorient3,
                                self.Ts00chck,
                                self.Ts01chck,
                                self.Ts02chck,
                                self.Ts11chck,
                                self.Ts12chck,
                                self.Ts22chck]

        List_of_keys = ["anglex", "angley", "anglez",
                        "Ts00", "Ts01", "Ts02", "Ts11", "Ts12", "Ts22",]

        print("List_of_defaultvalues", List_of_defaultvalues)

        self.fitting_parameters_keys = []
        self.fitting_parameters_values = []
        for k, checkwidget in enumerate(List_of_checkwidgets):

            if checkwidget.GetValue():
                print("k,checkwidget", k, checkwidget)
                self.fitting_parameters_keys.append(List_of_keys[k])
                self.fitting_parameters_values.append(List_of_defaultvalues[k + 5])

    def selectFittingLatticeParameters(self, key_material):
        """ set structural parameters with Lattice parameters model to be refined
        set self.fitting_parameters_keys, self.fitting_parameters_values
        according to checkboxes of fitting parameters"""
        # fixed parameters (must be an array)
        latticeparameters = self.dict_Materials[key_material][1]
        self.B0matrix = CP.calc_B_RR(latticeparameters)
        self.allparameters = np.array(self.CCDcalib + [0, 0, 0] + latticeparameters)

        self.constantlength = "a"
        if self.fita.GetValue() and self.fitb.GetValue() and self.fitc.GetValue():
            self.fita.SetValue(False)

        elif not self.fitb.GetValue():
            self.constantlength = "b"
        elif not self.fitc.GetValue():
            self.constantlength = "c"

        List_of_defaultvalues = np.array(self.CCDcalib + [0, 0, 0] + latticeparameters)
        List_of_checkwidgets = [self.fitorient1,
                                self.fitorient2,
                                self.fitorient3,
                                self.fita,
                                self.fitb,
                                self.fitc,
                                self.fitalpha,
                                self.fitbeta,
                                self.fitgamma]

        List_of_keys = ["anglex", "angley", "anglez",
                            "a", "b", "c",
                            "alpha", "beta", "gamma"]

        print("List_of_defaultvalues", List_of_defaultvalues)

        self.fitting_parameters_keys = []
        self.fitting_parameters_values = []
        for k, checkwidget in enumerate(List_of_checkwidgets):

            if checkwidget.GetValue():
                print("k,checkwidget", k, checkwidget)
                self.fitting_parameters_keys.append(List_of_keys[k])
                self.fitting_parameters_values.append(List_of_defaultvalues[k + 5])

    def selectFittingParameters(self):
        """ set structural parameters with Tc distorstion model to be refined
        set self.fitting_parameters_keys, self.fitting_parameters_values
        according to checkboxes of fitting parameters"""
        # fixed parameters (must be an array)
        self.allparameters = np.array(self.CCDcalib + [0.0, 0, 0,  # 3 misorientation / initial UB matrix
                                        1, 0, 0, 0, 1, 0, 0, 0, 1,  # Tc
                                        1, 0, 0, 0, 1, 0, 0, 0, 1,  # T
                                        1, 0, 0, 0, 1, 0, 0, 0, 1, ])  # Ts

        #         all_keys = ['anglex', 'angley', 'anglez',
        #         'Tc00', 'Tc01', 'Tc02', 'Tc10', 'Tc11', 'Tc12', 'Tc20', 'Tc21', 'Tc22',
        #         'T00', 'T01', 'T02', 'T10', 'T11', 'T12', 'T20', 'T21', 'T22',
        #         'Ts00', 'Ts01', 'Ts02', 'Ts10', 'Ts11', 'Ts12', 'Ts20', 'Ts21', 'Ts22']

        List_of_keys = ["anglex", "angley", "anglez",
                        "Tc11", "Tc22", "Tc01", "Tc02", "Tc12"]
        List_of_defaultvalues = [0, 0, 0, 1, 1, 0, 0, 0]
        List_of_checkwidgets = [self.fitorient1,
                                self.fitorient2,
                                self.fitorient3,
                                self.fitbovera,
                                self.fitcovera,
                                self.fitshear1,
                                self.fitshear2,
                                self.fitshear3]

        self.fitting_parameters_keys = []
        self.fitting_parameters_values = []
        for k, checkwidget in enumerate(List_of_checkwidgets):
            if checkwidget.GetValue():
                self.fitting_parameters_keys.append(List_of_keys[k])
                self.fitting_parameters_values.append(List_of_defaultvalues[k])

    def build_FitResults_Dict(self, _):
        """
        button OnShowResults of fit

        build dict of results of pairs distance minimization launched by show Results button
        """
        if self.fitresults:

            fields = ["#Spot Exp", "#Spot Theo", "h", "k", "l", "Intensity", "residues(pix.)"]
            # filter results Data spots

            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data

            indExp = np.array(self.linkedspots_fit[:, 0], dtype=np.int)
            indTheo = np.array(self.linkedspots_fit[:, 1], dtype=np.int)
            _h, _k, _l = np.transpose(np.array(self.linkExpMiller_fit, dtype=np.int))[1:4]
            intens = self.linkIntensity_fit

            if self.linkResidues_fit is not None:
                residues = np.array(self.linkResidues_fit)[:, 2]
            else:
                residues = -1 * np.ones(len(indExp))

            to_put_in_dict = indExp, indTheo, _h, _k, _l, intens, residues

            mySpotData = {}
            for k, ff in enumerate(fields):
                mySpotData[ff] = to_put_in_dict[k]
            dia = LSEditor.SpotsEditor(None, -1, "Show and Filter fit results Spots Editor ",
                                        mySpotData,
                                        func_to_call=self.readdata_fromEditor_Res,
                                        field_name_and_order=fields)
            dia.Show(True)

        else:
            wx.MessageBox("You must have run once a data refinement!", "INFO")

    def readdata_fromEditor_Res(self, data):
        """ set self attribute from data:
        self.linkedspots_fit
        self.linkExpMiller_fit
        self.linkIntensity_fit
        self.linkResidues_fit"""
        ArrayReturn = np.array(data)

        self.linkedspots_fit = ArrayReturn[:, :2]
        self.linkExpMiller_fit = np.take(ArrayReturn, [0, 2, 3, 4], axis=1)
        self.linkIntensity_fit = ArrayReturn[:, 5]
        self.linkResidues_fit = np.take(ArrayReturn, [0, 1, 6], axis=1)

        self.plotlinks = self.linkedspots_fit
        self._replot()

    def onWriteFitFile(self, _):
        """
        write a .fit file of indexation results of 1 grain
        """
        if not self.fitresults:
            wx.MessageBox("You must have run once a data refinement!", "INFO")
            return

        suffix = ""
        if self.incrementfilename.GetValue():
            self.savedfileindex += 1
            suffix = "_fitnb_%d" % self.savedfileindex

        outputfilename = self.DataPlot_filename.split(".")[0] + suffix + ".fit"

        # absolute exp spot index as found in .cor or .dat file
        indExp = np.array(self.linkedspots_fit[:, 0], dtype=np.int)
        _h, _k, _l = np.transpose(np.array(self.linkExpMiller_fit, dtype=np.int))[1:4]
        intens = self.linkIntensity_fit
        residues = np.array(self.linkResidues_fit)[:, 2]

        fullresults = 1
        # addCCDparams = 1

        if fullresults:
            # theoretical spots properties
            dictCCD = {}
            dictCCD["dim"] = self.framedim
            dictCCD["CCDparam"] = self.CCDcalib
            dictCCD["pixelsize"] = self.pixelsize
            dictCCD["kf_direction"] = self.kf_direction
            SpotsProperties = LAUE.calcSpots_fromHKLlist(self.UBmat, self.B0matrix,
                                                            np.array([_h, _k, _l]).T,
                                                            dictCCD)
            # (H, K, L, Qx, Qy, Qz, Xtheo, Ytheo, twthetheo, chitheo, Energytheo) = SpotsProperties
            (_, _, _, Qx, Qy, Qz, Xtheo, Ytheo, twthetheo, chitheo, Energytheo) = SpotsProperties

            # self.Data_2theta, self.Data_chi, self.Data_I, self.File_NAME = self.data

            AllDataToIndex = self.IndexationParameters["AllDataToIndex"]

            twtheexp = 2.0 * AllDataToIndex["data_theta"][indExp]
            chiexp = AllDataToIndex["data_chi"][indExp]
            Xexp = AllDataToIndex["data_pixX"][indExp]
            Yexp = AllDataToIndex["data_pixY"][indExp]


            Columns = [indExp, intens, _h, _k, _l, residues, Energytheo, Xexp, Yexp, twtheexp,
                                        chiexp, Xtheo, Ytheo, twthetheo, chitheo, Qx, Qy, Qz]

            columnsname = "#spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz\n"

        else:  # old only 5 columns in .fit file
            Columns = [indExp, intens, _h, _k, _l, residues]
            columnsname = "#spot_index Intensity h k l pixDev\n"

        datatooutput = np.transpose(np.array(Columns)).round(decimals=6)

        dict_matrices = {}
        dict_matrices["UBmat"] = self.UBmat

        # OR
        dict_matrices["Umat2"] = self.Umat2
        dict_matrices["Bmat_tri"] = self.Bmat_tri
        dict_matrices["HKLxyz_names"] = self.HKLxyz_names
        dict_matrices["HKLxyz"] = self.HKLxyz

        dict_matrices["B0"] = self.B0matrix
        dict_matrices["UBB0"] = self.UBB0mat
        dict_matrices["devstrain_crystal"] = self.deviatoricstrain
        dict_matrices["devstrain_sample"] = self.deviatoricstrain_sampleframe
        dict_matrices["CCDLabel"] = self.CCDLabel
        dict_matrices["detectorparameters"] = self.CCDcalib
        dict_matrices["pixelsize"] = self.pixelsize
        dict_matrices["framedim"] = self.framedim
        dict_matrices["kf_direction"] = self.kf_direction

        euler_angles = ORI.calc_Euler_angles(self.UBB0mat).round(decimals=3)
        dict_matrices["euler_angles"] = euler_angles

        if self.new_latticeparameters is not None:
            dict_matrices["LatticeParameters"] = self.new_latticeparameters

        if self.Tsresults is not None:
            dict_matrices["Ts"] = self.Tsresults

        if self.IndexationParameters["writefolder"] is None:
            self.IndexationParameters["writefolder"] = OSLFGUI.askUserForDirname(self)


        dlg = wx.TextEntryDialog(self, "Enter File name with .fit extension: \n",
                                                            "Saving refined peaks list")
        dlg.SetValue("%s" % outputfilename)
        filenamefit = None
        if dlg.ShowModal() == wx.ID_OK:
            filenamefit = str(dlg.GetValue())
            fullpath = os.path.abspath(os.path.join(self.IndexationParameters["writefolder"], filenamefit))

        dlg.Destroy()

        IOLT.writefitfile(fullpath, datatooutput, len(indExp), meanresidues=np.mean(residues),
                            PeakListFilename=self.DataPlot_filename,
                            columnsname=columnsname,
                            dict_matrices=dict_matrices,
                            modulecaller="LaueToolsGUI.py")

        wx.MessageBox("Fit results saved in %s" % fullpath, "INFO")

    def Simulate_Pattern(self):
        """
        in plot_RefineFrame

        compute

        self.data_theo = [Twicetheta, Chi, Miller_ind, posx, posy, energy]

        self.data_theo_pixXY = [posx, posy, Miller_ind, Twicetheta, Chi, energy]
        """
        # print(" self.detectordiameter in plot_RefineFrame.Simulate_Pattern() ",
        #     self.detectordiameter)

        # for squared detector need to increase a bit
        if self.CCDLabel.startswith("sCMOS"):
            diameter_for_simulation = self.detectordiameter * 1.4 * 1.25
        else:
            diameter_for_simulation = self.detectordiameter
        if self.SimulParam != None:
            # print "self.SimulParam in Plot_RefineFrame.Annotate_theo()",self.SimulParam
            Grain, Emin, Emax = self.SimulParam
            (Twicetheta, Chi,
            Miller_ind,
            posx, posy,
            energy) = LAUE.SimulateLaue(Grain,
                                        Emin,
                                        Emax,
                                        self.CCDcalib,
                                        kf_direction=self.kf_direction,
                                        removeharmonics=1,
                                        pixelsize=self.pixelsize,
                                        dim=self.framedim,
                                        ResolutionAngstrom=self.ResolutionAngstrom,
                                        detectordiameter=diameter_for_simulation * 1.25,
                                        dictmaterials=self.dict_Materials)

            self.data_theo = [Twicetheta, Chi, Miller_ind, posx, posy, energy]

            self.data_theo_pixXY = [posx, posy, Miller_ind, Twicetheta, Chi, energy]

            # if self.datatype == 'gnomon':  # really needed?
            # # compute Gnomonic projection
            # _nbofspots = len(Twicetheta)
            # sim_dataselected = IOLT.createselecteddata((Twicetheta,Chi,np.ones(_nbofspots)),
            # np.arange(_nbofspots),
            # _nbofspots)[0]
            # sim_gnomonx, sim_gnomony = IIM.ComputeGnomon_2(sim_dataselected)
            # self.data_theo=[sim_gnomonx, sim_gnomony, Miller_ind]

            # print "I have(Re)-simulated data in Plot_RefineFrame *************"
            return Twicetheta, Chi, Miller_ind, posx, posy, energy

    def EnterComboElem(self, event):
        """
        in plot_RefineFrame
        """
        print("EnterComboElem in RefineFrame")
        item = event.GetSelection()
        self.key_material = self.list_Materials[item]

        print("self.key_material", self.key_material)

        self.sb.SetStatusText("Selected material: %s" % str(self.dict_Materials[self.key_material]))

        self.OnReplot(1)

        event.Skip()

    def onSelectUBmatrix(self, event):
        """ select UB matrix and replot"""
        self.sb.SetStatusText("Selected Ubmatrix: %s" % str(self.comboUBmatrix.GetValue()))
        self.UpdateFromRefinement.SetValue(False)
        self.OnReplot(1)
        event.Skip()

    def all_reds(self):
        """ set all buttons to red"""
        for butt in self.listbuttons:
            butt.SetBackgroundColour(self.defaultColor)

    def what_was_pressed(self, flag):
        """ print function to show all buttons state"""
        if flag:
            print("-------------------")
            print([butt.GetValue() for butt in self.listbuttons])
            print("self.listbuttonstate", self.listbuttonstate)

    def OnAcceptMatching(self, evt):  # accept AcceptMatching indexation
        """ accept current spots theo and exp. matching as completed indexation
        call Spot_MillerAttribution()"""
        self.what_was_pressed(0)
        if self.listbuttonstate[0] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pointButton5.SetValue(True)
            self.pointButton5.SetBackgroundColour("Green")
            self.listbuttonstate = [1, 0, 0]
            print("\n\n********************\nAccepting Matching...\n********************\n\n")
            
            self.Spot_MillerAttribution(evt)
        else:
            self.pointButton5.SetBackgroundColour(self.defaultColor)
            self.pointButton5.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

            evt.Skip()

    def T6(self, evt):
        """ Draw Exp. Spot index mode """
        self.what_was_pressed(0)
        if self.listbuttonstate[1] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pointButton6.SetValue(True)
            self.pointButton6.SetBackgroundColour("Green")
            self.listbuttonstate = [0, 1, 0]
        else:
            self.pointButton6.SetBackgroundColour(self.defaultColor)
            self.pointButton6.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

            evt.Skip()

    def T7(self, _):
        """ Draw Theo. Spot index mode"""
        self.what_was_pressed(0)
        print(self.GetId())
        if self.listbuttonstate[2] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pointButton7.SetValue(True)
            self.pointButton7.SetBackgroundColour("Green")
            self.listbuttonstate = [0, 0, 1]
        else:
            self.pointButton7.SetBackgroundColour(self.defaultColor)
            self.pointButton7.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

    def close(self, _):
        """ close """
        self.Close(True)

    # --- ---------Plot  functions
    def setplotlimits_fromcurrentplot(self):
        """ set self.xlim and self.ylim to current plot borders """
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()

    def _replot(self):
        """
        _replot() in Plot_RefineFrame
        """
        # print("_replot")

        # offsets to match imshow and scatter plot coordinates frames
        if self.datatype == "pixels":
            X_offset = 1
            Y_offset = 1
        else:
            X_offset = 0
            Y_offset = 0

        if self.datatype_unchanged:
            if not self.init_plot:
                self.setplotlimits_fromcurrentplot()
        else:
            self.init_plot = True
            self.xlim, self.ylim = self.getDataLimits()

        # Data_X, Data_Y, Data_I, File_NAME = self.data

        #        fig = self.plotPanel.get_figure()
        #        self.axes = fig.gca()
        self.axes.clear()

        # clear the axes and replot everything
        #        self.axes.cla()

        def fromindex_to_pixelpos_x(index, _):
            """ return pixel position X from plot """
            if self.datatype == "pixels":
                return int(index)
            else:
                return index

        def fromindex_to_pixelpos_y(index, _):
            """ return pixel position Y from plot"""
            if self.datatype == "pixels":
                return int(index)
            else:
                return index

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))

        # image array
        if self.ImageArray is not None and self.datatype == "pixels":

            # array to display: raw
            print('self.data_dict["removebckg"]', self.data_dict["removebckg"])
            if not self.data_dict["removebckg"]:
                self.ImageArray = self.ImageArrayInit
            # array to display: raw - bkg
            else:
                if self.ImageArrayMinusBckg is None:
                    # compute
                    backgroundimage = ImProc.compute_autobackground_image(self.ImageArrayInit,
                                                                            boxsizefilter=10)
                    # basic substraction
                    self.ImageArrayMinusBckg = ImProc.computefilteredimage(self.ImageArrayInit,
                                                            backgroundimage, self.CCDLabel,
                                                            usemask=True, formulaexpression='A-B')

                self.ImageArray = self.ImageArrayMinusBckg

            self.myplot = self.axes.imshow(self.ImageArray, interpolation="nearest", origin="upper")

            if not self.data_dict["logscale"]:
                norm = matplotlib.colors.Normalize(vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"])
            else:
                norm = matplotlib.colors.LogNorm(vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"])

            self.myplot.set_norm(norm)
            self.myplot.set_cmap(self.data_dict["lut"])

        #             if self.init_plot:
        #                 self.axes.set_ylim((2048, 0))
        #                 self.axes.set_xlim((0, 2048))

        # --------  plot theoretical(simulated) data  spots -----------
        if self.data_theo is not None:
            #             print "there is theo. data in plot_RefineFrame"

            markerstyle = "*"  # 'o' 'h'
            markersizetheo = 100

            if self.datatype in ("2thetachi", ):
                # define self.data_theo
                self.Simulate_Pattern()
                self.data_theo_displayed = self.data_theo

            elif self.datatype == "pixels":
                self.data_theo = self.data_theo_pixXY
                self.data_theo_displayed = (np.array(self.data_theo_pixXY[:2]).T - np.array([X_offset, Y_offset])).T
            #                 print "self.data_theo_displayed", self.data_theo_displayed

            try:  # in linux there is not keyword 'OS'...!
                # TODO: very strange !
                os.environ["OS"] == "Windows_NT"
            except KeyError:
                # print "No acces to this key is os.environ ??!!"
                # self.axes.scatter(self.data_theo[0], self.data_theo[1],s = 50, marker = 'o',facecolor = 'None',edgecolor = 'r',alpha = 1.)  # ok for window, matplotlib 0.99.1.1
                self.axes.scatter(self.data_theo_displayed[0], self.data_theo_displayed[1],
                                        s=markersizetheo, marker=markerstyle, edgecolor="r", facecolors="None")

                if matplotlibversion == "0.99.1":  # ok linux with matplotlib 0.99.1
                    # print "matplotlibversion  ==  '0.99.1'"
                    self.axes.scatter(self.data_theo_displayed[0], self.data_theo_displayed[1],
                                        s=markersizetheo, marker=markerstyle, edgecolor="r", facecolor="None")

            else:
                print("else of KeyError")
                # ok for windows matplotlib 0.99.1
                self.axes.scatter(self.data_theo_displayed[0],
                                    self.data_theo_displayed[1],
                                    s=markersizetheo,
                                    marker=markerstyle,
                                    facecolor="None",
                                    edgecolor="r")

        # ---------------------------------------------------------------
        # Experimental spots  (including linked ones to theo. spot)
        # ---------------------------------------------------------------
        if self.datatype == "2thetachi":

            coefsize = float(self.spotsize_ctrl.GetValue())

            self.axes.scatter(self.data_2thetachi[0],
                                self.data_2thetachi[1],
                                s=coefsize*self.Data_I / np.amax(self.Data_I) * 100.0,
                                alpha=0.5)

        elif self.datatype == "pixels":

            # background image
            if self.ImageArray is not None:
                kwords = {"marker": "o", "facecolor": "None",
                                                        "edgecolor": self.data_dict["markercolor"]}
            else:
                kwords = {"edgecolor": "None", "facecolor": self.data_dict["markercolor"]}

            self.axes.scatter(self.pixelX - X_offset,
                            self.pixelY - Y_offset,
                            s=self.Data_I / np.amax(self.Data_I) * 100.0,
                            alpha=0.5,
                            **kwords)

        # ---------------------------------------------------------------
        # plot experimental spots linked to 1 theo. spot)
        # ---------------------------------------------------------------
        if self.plotlinks is not None:
            exp_indices = np.array(np.array(self.plotlinks)[:, 0], dtype=np.int)
            #print('exp_indices in yellow links plot ',exp_indices)

            # experimental spots selection -------------------------------------
            AllData = self.IndexationParameters["AllDataToIndex"]
            if self.datatype == "2thetachi":
                Xlink = np.take(2.0 * AllData["data_theta"], exp_indices)
                Ylink = np.take(AllData["data_chi"], exp_indices)
            elif self.datatype == "pixels":
                pixX = np.take(AllData["data_pixX"], exp_indices)
                pixY = np.take(AllData["data_pixY"], exp_indices)

                Xlink = pixX - X_offset
                Ylink = pixY - Y_offset

            self.axes.scatter(Xlink, Ylink, s=100., alpha=0.5, c='yellow')

        if self.highlighttheospot is not None:
            iHL = self.highlighttheospot
            XtheoHL, YtheoHL = self.data_theo_displayed[0][iHL], self.data_theo_displayed[1][iHL]
            self.axes.scatter(XtheoHL, YtheoHL, s=100., alpha=0.5, c='k', marker='X')

        if self.highlightexpspot is not None:
            indexHL = self.highlightexpspot
            if self.datatype == "2thetachi":
                XexpHL = self.data_2thetachi[0][indexHL]
                YexpHL = self.data_2thetachi[1][indexHL]

            elif self.datatype == "pixels":
                XexpHL = self.pixelX[indexHL] - X_offset
                YexpHL = self.pixelY[indexHL] - Y_offset

            self.axes.scatter(XexpHL, YexpHL, s=100., alpha=0.5, c='k', marker='X')

        # axes labels
        if self.datatype == "2thetachi":
            self.axes.set_xlabel("2theta(deg.)")
            self.axes.set_ylabel("chi(deg)")
        elif self.datatype == "pixels":
            self.axes.set_xlabel("X pixel")
            self.axes.set_ylabel("Y pixel")

        nbspotstoindex = len(self.IndexationParameters["DataToIndex"]["current_exp_spot_index_list"])

        texttitle = "%s %d/%d spots" % (self.File_NAME, len(self.Data_I), nbspotstoindex)
        if self.ImageArray is not None:
            if hasattr(self, "fullpathimagefile"):
                texttitle += "\nImage: %s" % self.fullpathimagefile

        self.axes.set_title(texttitle)
        self.axes.grid(True)

        # restore the zoom limits(unless they're for an empty plot)
        if self.xlim != (0.0, 1.0) or self.ylim != (0.0, 1.0):
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)

        self.init_plot = False
        self.datatype_unchanged = True

        # redraw the display
        #        self.plotPanel.draw()
        self.canvas.draw()

    def allbuttons_off(self):
        """ set all buttons state to False """
        for butt in self.listbuttons:
            butt.SetValue(False)

    def readlogicalbuttons(self):
        """ return list of all buttons state """
        return [butt.GetValue() for butt in self.listbuttons]

    def _on_point_choice(self, evt):
        """
        Plot_RefineFrame
        """
        all_states = self.readlogicalbuttons()
        # print "all_states in plotToolsFrame",all_states

        if all_states == [True, False, False]:  # accept indexation
            self.pointButton5.SetValue(True)
            self.Spot_MillerAttribution(evt)

        if all_states == [False, True, False]:
            self.pointButton6.SetValue(True)
            self.Annotate_exp(evt)

        if all_states == [False, False, True]:
            self.pointButton7.SetValue(True)
            #            print "self.Annotate_theo(evt) in Plot_RefineFrame"
            self.Annotate_theo(evt)
        if all_states == [False, False, False]:
            # evt.Skip()
            pass

    #    def Store_matrix(self, evt):
    #        """
    #        Plot_RefineFrame
    #        """
    #        if self.current_matrix != []:
    #            LaueToolsframe.mat_store_ind += 1
    #            if self.Bmat != None and self.Umat != None:
    #                tostore = self.UBmat
    #            else:
    #                tostore = self.current_matrix
    #
    #            LaueToolsframe.Matrix_Store.append(tostore)
    #            LaueToolsframe.dict_Rot["MatManualIndex_%d" % LaueToolsframe.mat_store_ind] = tostore
    #
    #            print "Stored matrix ..."
    #            print "MatManualIndex_%d" % LaueToolsframe.mat_store_ind
    #
    #            wx.MessageBox('Matrix is stored as %s\nin rotation matrix list for further simulation with B matrix = Id' % ("MatManualIndex_%d" % LaueToolsframe.mat_store_ind), 'INFO')
    #        else:
    #            print "No simulation matrix available!..."

    def OnStoreMatrix(self, _):
        """
        store matrix

        in Plot_RefineFrame
        """
        if self.current_matrix != []:
            self.mat_store_ind = len(self.Matrix_Store)
            if self.Bmat is not None and self.Umat is not None:
                tostore = self.UBmat
            else:
                tostore = self.current_matrix

            self.Matrix_Store.append(tostore)

            prefix = "MatrixManualIndex"

            proposed_name = "%s_%d" % (prefix, self.mat_store_ind)

            self.selectedName = None
            dlg = MessageDataBox(self, -1, "Storing Matrix Name Entry", tostore, proposed_name)

            dlg.Show(True)

        else:
            print("No simulation matrix available!...")

    def Spot_MillerAttribution(self, evt):
        """
        in Plot_RefineFrame

        button Accept Matching

        Assign Miller indices of exp. spots for simulated spots close to them
        Assumption: undistorted cubic crystal in CCD on top geometry

        matching is done in 2theta, chi in order to take the geometry into account
        before this Miller assignement

        TODO: use other reference crystal with other geometry
        """
        if self.mainframe is None:
            wx.MessageBox("This button is useful only for sequential indexation of several "
                            "grains from LaueToolsGUI", "Info")
            self.pointButton5.SetBackgroundColour(self.defaultColor)
            self.pointButton5.SetValue(False)
            self.listbuttonstate = [0, 0, 0]
            return

        try:
            if self.linkResidues_fit is not None:
                pass
        except AttributeError:
            wx.MessageBox('You must perform a fitting by for instance making '
                            '"Automatic Links" then "Refine"!', 'Info')
            return

        if self.linkResidues_fit is None:
            wx.MessageBox('You must perform a fitting by for instance making '
                            '"Automatic Links" then "Refine"!', 'Info')
            return


        Grain, emin, emax = self.SimulParam

        # Umat = Grain[2]
        # Bmat = Grain[0]

        updategrain = False
        # like for OnReplot : to match exp and simul data according to results of refinement
        # Grain that must be input is : [Id, dilat = [1,1,1], orientMatrix = self.UBmat, 'element']
        # instead of what one would have expected: [self. Bmat  , dilat =[1,1,1], self.Umat, 'elem']

        # TODO: align way of simulating data(2theta,chi with LAUE6.generalfabrique...) and refined data(in FitOrient....)
        # or this may due to the fact that U matrix given from indexation relies on RS vectors in an other order than in simulation?

        # if self.Bmat != None:
        # Grain[0] = self.Bmat
        # updategrain = True

        # if self.Umat != None:
        # Grain[2] = self.Umat
        # updategrain = True

        if self.Bmat is not None and self.Umat is not None:
            Grain[0] = np.eye(3)
            Grain[2] = self.UBmat
            updategrain = True

        epsil = np.zeros((3, 3))
        if self.deviatoricstrain is not None:
            epsil = self.deviatoricstrain

        if updategrain:
            self.SimulParam = (Grain, emin, emax)

        # Generating link automatically
        self.OnAutoLink(evt)  # inside there is simulatePattern() which uses self.SimulParam

        # self.linkedspots_link
        # self.linkExpMiller_link
        # self.linkIntensity_link
        # self.linkResidues_link
        # self.Energy_Exp_spot
        #print('self.linkExpMiller_link', self.linkExpMiller_link)

        Miller_Exp_spot = np.array(self.linkExpMiller_link, dtype=np.int)[:, 1:4]
        List_Exp_spot_close = np.array(self.linkedspots_link[:, 0], dtype=np.int)
        Energy_Exp_spot = self.Energy_Exp_spot

        #print("List_Exp_spot_close in Spot_MillerAttribution", List_Exp_spot_close)

        # Updating the DB ------------------
        if self.Bmat is None and self.Umat is None:
            # non  pure U matrix but rather UB matrix due to refinement procedure to find a matrix from two spots
            self.mainframe.last_orientmatrix_fromindexation[self.current_processedgrain] = Grain[2]
            # Identity use for recognition
            self.mainframe.last_Bmatrix_fromindexation[self.current_processedgrain] = Grain[0]
        else:  # use results of refinement
            # a UB matrix
            self.mainframe.last_orientmatrix_fromindexation[self.current_processedgrain] = self.UBmat
            # the B matrix
            self.mainframe.last_Bmatrix_fromindexation[self.current_processedgrain] = self.Bmat
            # we can if needed from UB and B extract B  ...

        # update DataSetObject: -----------------------------
        #         after refineUB dicts created
        grain_index = self.current_processedgrain
        self.DataSet.key_material = self.key_material
        self.DataSet.emin = emin
        self.DataSet.emax = emax
        print("UB best refined", self.UBmat)
        print("DataSet.B0matrix", self.Bmat)
        print("DataSet.deviatoricstrain", self.deviatoricstrain)
        print("DataSet.pixelresidues", self.linkResidues_link)
        print("self.DataSet.detectordiameter", self.DataSet.detectordiameter)
        #         print "DataSet.spotindexabs", DataSet.spotindexabs

        matching_angle_tolerance = float(self.matr_ctrl.GetValue())

        # --- Indexed  spots will be only those used for the refinement
        #         self.DataSet.resetSpotsFamily(self.current_processedgrain)
        #         if self.linkResidues_fit is not None:
        #             self.DataSet.getSpotsFromSpotIndices(np.array(self.linkResidues_fit[:,0],dtype=np.int))
        #         else:
        #             wx.MessageBox('You must perform a fitting!','Info')
        #             return

        # matching

        print("\n\n\n self.linkResidues_fit[:,0]", self.linkResidues_fit[:, 0])
        print("self.current_processedgrain", self.current_processedgrain)
        print("\n\n\n")
        self.DataSet.AssignHKL(ISS.OrientMatrix(self.UBmat),
                                grain_index,
                                matching_angle_tolerance,
                                selectbyspotsindices=np.array(self.linkResidues_fit[:, 0], dtype=np.int))
        self.DataSet.dict_grain_devstrain[grain_index] = self.deviatoricstrain
        self.DataSet.dict_indexedgrains_material[grain_index] = self.key_material

        print(self.DataSet.dict_grain_devstrain[grain_index])
        print(self.DataSet.dict_grain_matrix[grain_index])  # = matrix
        print(self.DataSet.dict_grain_matching_rate[grain_index])  # = [nb_updates, matching_rate]
        print(self.DataSet.dict_Missing_Reflections[grain_index])  # = missingRefs

        self.DataSet.indexedgrains.append(self.current_processedgrain)

        print("self.DataSet.dict_grain_matrix", self.DataSet.dict_grain_matrix)
        print("self.DataSet.dict_grain_matching_rate", self.DataSet.dict_grain_matching_rate)
        # -----------------------------------

        # update spots properties with respect to indexation results (and self.current_processedgrain +=1)
        self.mainframe.last_epsil_fromindexation[self.current_processedgrain] = epsil
        self.mainframe.Update_DataToIndex_Dict([Miller_Exp_spot,
                                                Energy_Exp_spot,
                                                List_Exp_spot_close])
        self.mainframe.Update_DB_fromIndexation([Miller_Exp_spot,
                                                Energy_Exp_spot,
                                                List_Exp_spot_close])

        # closing windows
        self.Close(True)
        self.parent.Close(True)

        if self.indexationframe is not "unknown":
            self.indexationframe.Close(True)
        else:
            # preventing to close the mainframe App !!
            if self.GetGrandParent().GetId() != self.mainframe.GetId():
                self.GetGrandParent().Close(True)

        evt.Skip()

    # --- ---------  Plot annotations
    def drawAnnote_exp(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index
        #Plot_RefineFrame
        """
        # TODO texts offset as a function of plot size
        if self.datatype == "2thetachi":
            textshifts = (1, 1, 2, -2)
        elif self.datatype == "pixels":
            textshifts = (15, 15, 40, -40)
        if (x, y) in self.drawnAnnotations_exp:
            markers = self.drawnAnnotations_exp[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            t1 = axis.text(
                x + textshifts[0], y + textshifts[1], "#spot %d" % (annote[0]), size=8
            )
            t2 = axis.text(
                x + textshifts[2],
                y + textshifts[3],
                "Intens. %.1f" % (annote[1]),
                size=8,
            )
            if matplotlibversion <= "0.99.1":
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False
                )
            else:
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None"
                )  # matplotlib 0.99.1.1
            self.drawnAnnotations_exp[(x, y)] = (t1, t2, m)
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()

    def drawSpecificAnnote_exp(self, annote):
        """
        in Plot_RefineFrame
        """
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_exp if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_exp(self.axes, x, y, a)

    def Annotate_exp(self, event):
        """
        in Plot_RefineFrame
        """
        if self.datatype == "2thetachi":
            xtol = 5
            ytol = 5
        elif self.datatype == "pixels":
            xtol = 200
            ytol = 200
        # """
        # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
        # self.Data_index_expspot = np.arange(len(self.Data_X))
        # """
        if self.datatype == "2thetachi":
            xdata, ydata, annotes = (self.Data_X, self.Data_Y,
                list(zip(self.Data_index_expspot, self.Data_I)))
        elif self.datatype == "pixels":
            xdata, ydata, annotes = (self.data_XY[0], self.data_XY[1],
                list(zip(self.Data_index_expspot, self.Data_I)))
        # print self.Idat
        # print self.Mdat
        # print annotes
        self._dataANNOTE_exp = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        # print clickX, clickY

        annotes = []
        for x, y, a in self._dataANNOTE_exp:
            if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]
            print("the nearest experimental point is at(%.2f,%.2f)" % (x, y))
            print("with index %d and intensity %.1f" % (annote[0], annote[1]))
            self.drawAnnote_exp(self.axes, x, y, annote)
            for l in self.links_exp:
                l.drawSpecificAnnote_exp(annote)

            self.updateStatusBar(x, y, annote, spottype="exp")

    def drawAnnote_theo(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index

        in Plot_RefineFrame
        """
        if self.datatype == "pixels":
            textoffset = 10
        if self.datatype == "2thetachi":
            textoffset = 2

        if (x, y) in self.drawnAnnotations_theo:
            markers = self.drawnAnnotations_theo[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            t1 = axis.text(x + textoffset,
                            y,
                            "%s\n%.2f" % (str(annote[0]), annote[3]),
                            size=8,
                            color="r")

            if matplotlibversion <= "0.99.1":
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False)
            else:
                m = axis.scatter([x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None")  # matplotlib 0.99.1.1

            self.drawnAnnotations_theo[(x, y)] = (t1, m)
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()

    def drawSpecificAnnote_theo(self, annote):
        """ draw annote according to self._dataANNOTE_theo"""
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_theo if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_theo(self.axes, x, y, a)

    def Annotate_theo(self, event):
        """ Display Miller indices of user clicked spot

        Add annotation in plot for the theoretical spot that is closest to user clicked point

        #in Plot_RefineFrame
        """
        print("Plot_RefineFrame.Annotate_theo()")

        if self.datatype == "2thetachi":
            xtol = 5
            ytol = 5
        elif self.datatype == "pixels":
            xtol = 200
            ytol = 200

        if self.datatype == "2thetachi":
            # 2theta, chi, (miller posx posy energy)
            annotes = self.data_theo[2:]

        if self.datatype == "pixels":
            # x, y, (miller 2theta chi energy)
            _, _, annotes = (self.data_theo_pixXY[0],
                                    self.data_theo_pixXY[1],
                                    self.data_theo_pixXY[2:])

        #         elif self.datatype == 'gnomon':
        #             xdata, ydata, annotes = self.data_theo  # sim_gnomonx, sim_gnomony, Miller_ind

        #         self._dataANNOTE_theo = zip(xdata, ydata, annotes)
        xdata_theo, ydata_theo, _annotes_theo = (self.data_theo[0],
                                                self.data_theo[1],
                                                list(zip(*self.data_theo[2:])))

        clickX = event.xdata
        clickY = event.ydata

        print("clickX, clickY", clickX, clickY)

        annotes = []
        for x, y, atheo in zip(xdata_theo, ydata_theo, _annotes_theo):
            if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):

                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, atheo))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]  # this the best of annotes
            # print("the nearest theo point is at(%.2f,%.2f)" % (x, y))
            # print("with index %s " % (str(annote)))
            self.drawAnnote_theo(self.axes, x, y, annote)
            for l in self.links_theo:
                l.drawSpecificAnnote_theo(annote)

            self.updateStatusBar(x, y, annote)

    def updateStatusBar(self, x, y, annote, spottype="theo"):
        """ display spots properties in statusbar"""
        if self.datatype == "2thetachi":
            Xplot = "2theta"
            Yplot = "chi"
        else:
            Xplot = "x"
            Yplot = "y"

        if spottype == "theo":
            self.sb.SetStatusText(("%s= %.2f " % (Xplot, x)
                                        + " %s= %.2f " % (Yplot, y)
                                        + "  HKL=%s " % str(annote[0])
                                        + "E=%.2f keV" % annote[3]), 0)

        elif spottype == "exp":

            self.sb.SetStatusText(("%s= %.2f " % (Xplot, x)
                                        + " %s= %.2f " % (Yplot, y)
                                        + "   Spotindex=%d " % annote[0]
                                        + "   Intensity=%.2f" % annote[1]), 1)


# --- ---------Results of fitting Dialog class ------------------
class FitResultsBoard(wx.Dialog):
    """
    Class to set image file parameters
    """

    def __init__(self, parent, _id, title, data_dict):
        """
        initialize FitResultsBoard window
        """
        textdsc = data_dict["devstrain_crystal"]
        textdss = data_dict["devstrain_sample"]
        textfulls = data_dict["fullstrain_sample"]
        textub = data_dict["UBmatrix"]
        textb0 = data_dict["B0matrix"]
        textil = data_dict["Initial lattice"]
        textrl = data_dict["Refined lattice"]
        texttnbresidues = data_dict["NbspotsResidues"]

        textts = data_dict.get("Ts parameters", "")
        textHKLxyz_names = data_dict["HKLxyz_names"]
        textHKLxyz = data_dict["HKLxyz"]


        wx.Dialog.__init__(self, parent, -1, title=title, pos=(200, 200), size=(810, 660))

        # Start of sizers and widgets contained within.
        self.background = self  # wx.Panel(self)

        self.OKBtn = wx.Button(self.background, wx.ID_OK)
        self.OKBtn.SetDefault()

        WIDTH = 400

        self.txtnbresidues = wx.StaticText(self.background, -1, texttnbresidues)
        self.initlat = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 140))
        self.finallat = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 140))
        self.ub = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 100))
        self.b0 = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 100))
        self.devstraincryst = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 100))
        self.devstrainsample = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 100))
        self.Ts = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 120))
        self.fullstrainsample = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 120))

        self.HKLxyz_names = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 140))
        self.HKLxyz = wx.TextCtrl(
            self.background, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER, size=(WIDTH, 140))

        self.initlat.SetValue(textil)
        self.finallat.SetValue(textrl)
        self.ub.SetValue(textub)
        self.b0.SetValue(textb0)
        self.devstraincryst.SetValue(textdsc)
        self.devstrainsample.SetValue(textdss)
        self.Ts.SetValue(textts)
        self.fullstrainsample.SetValue(textfulls)
        self.HKLxyz_names.SetValue(textHKLxyz_names)
        self.HKLxyz.SetValue(textHKLxyz)

        if 0:
            self.devstrainsample.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.initlat.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.finallat.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.ub.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.b0.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.Ts.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.devstraincryst.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.HKLxyz_names.Bind(wx.EVT_TEXT_ENTER, self.onNothing)
            self.HKLxyz.Bind(wx.EVT_TEXT_ENTER, self.onNothing)

        # layout----------------------
        horizontalBox1 = wx.BoxSizer()
        horizontalBox1.Add(self.txtnbresidues, proportion=1, border=0)

        horizontalBox2 = wx.BoxSizer()
        horizontalBox2.Add(self.initlat, proportion=1, border=0)
        horizontalBox2.Add(self.finallat, proportion=1, border=0)

        horizontalBox3 = wx.BoxSizer()
        horizontalBox3.Add(self.ub, proportion=1, border=0)
        horizontalBox3.Add(self.b0, proportion=1, border=0)

        horizontalBox = wx.BoxSizer()
        horizontalBox.Add(self.devstraincryst, proportion=1, border=0)
        horizontalBox.Add(self.devstrainsample, proportion=1, border=0)

        horizontalBox4 = wx.BoxSizer()
        horizontalBox4.Add(self.Ts, proportion=1, border=0)
        horizontalBox4.Add(self.fullstrainsample, proportion=1, border=0)

        horizontalBox5 = wx.BoxSizer()
        horizontalBox5.Add(self.HKLxyz_names, proportion=1, border=0)
        horizontalBox5.Add(self.HKLxyz, proportion=1, border=0)

        verticalBox = wx.BoxSizer(wx.VERTICAL)
        verticalBox.Add(horizontalBox1, proportion=0, flag=wx.EXPAND, border=5)
        verticalBox.Add(horizontalBox2, proportion=0, flag=wx.EXPAND, border=5)
        verticalBox.Add(horizontalBox3, proportion=0, flag=wx.EXPAND, border=5)
        verticalBox.Add(horizontalBox, proportion=0, flag=wx.EXPAND, border=5)
        verticalBox.Add(horizontalBox5, proportion=0, flag=wx.EXPAND, border=5)
        verticalBox.Add(self.OKBtn, proportion=0, flag=wx.EXPAND, border=0)
        verticalBox.Add(horizontalBox4, proportion=0, flag=wx.EXPAND, border=5)

        self.background.SetSizer(verticalBox)

        self.CentreOnParent(wx.BOTH)
        self.SetFocus()

        txt_define_reference_frames = "Lab frame Rlab (LT) :\n xlab along incident beam ki (downstream),\n zlab perpendicular to CCD screen (upwards),\n ylab = z^x approx along -xech (door-wards)\n"
        txt_define_reference_frames += "Sample frame Rsample (LT) :\n obtained from lab frame by rotation of sampletilt = 40 degrees around -ylab. \n"
        txt_define_reference_frames += " xsample approx along yech (downstream + upwards),\n ysample = approx along -xech (door-wards),\n zsample approx along microscope axis (upstream upwards) \n"
        txt_define_reference_frames += "Warning : xech yech here give the position of the microbeam spot with respect to the sample, xech right-wards and yech upwards with respect to the sample image on the microscope camera. \n"
        txt_define_reference_frames += "The UB B0 matrix gives as columns the components of astar, btar and cstar in Rlab frame. \n"
        txt_define_reference_frames += 'The B0 matrix gives as columns the "before-fit" components of astar bstar cstar on the cartesian frame built from Rstar,\n'
        txt_define_reference_frames += " i.e. the initial shape of the reciprocal unit cell \n"
        txt_define_reference_frames += "Deviatoric strain in crystal frame is in the cartesian frame built from the vectors of the direct-space crystal unit cell. \n"
        self.OKBtn.SetToolTipString(txt_define_reference_frames)

    def onNothing(self, _):
        """ to be implemented """
        return


# --- ---------------  Plot limits board  parameters
class IntensityScaleBoard(wx.Dialog):
    """
    Class to set image file parameters
    """
    def __init__(self, parent, _id, title, data_dict):
        """
        initialize board window

        """
        wx.Dialog.__init__(self, parent, _id, title, size=(400, 300))

        self.parent = parent
        print("self.parent in IntensityScaleBoard ", self.parent)

        self.ImageArray = None
        if self.parent is not None:
            if hasattr(self.parent, "ImageArray"):
                self.ImageArray = self.parent.ImageArray

        self.fullpathimagefile = None

        self.data_dict = data_dict
        self.dict_colors = {0: "b", 1: "g", 2: "r", 3: "yellow", 4: "m", 5: "grey",
                                6: "black", 7: "white", 8: "pink"}
        self.colorindex = 0

        Imin = self.data_dict["Imin"]
        Imax = self.data_dict["Imax"]
        self.IminDisplayed = vmin = self.data_dict["vmin"]
        self.ImaxDisplayed = vmax = self.data_dict["vmax"]
        lut = self.data_dict["lut"]

        self.init_Imin = copy.copy(Imin)
        self.init_Imax = copy.copy(Imax)
        self.init_vmin = copy.copy(vmin)
        self.init_vman = copy.copy(vmax)
        self.init_lut = copy.copy(lut)

        self.mapsLUT = ["jet", "GnBu", "cool", "BuGn", "PuBu", "autumn",
                        "copper", "gist_heat", "hot", "spring"]

        self.xlim = None
        self.ylim = None

        # WIDGETS
        wx.StaticText(self, -1, "LUT", (5, 7))
        self.comboLUT = wx.ComboBox(self, -1, self.init_lut, (70, 5), choices=self.mapsLUT)  # ,
        # style=wx.CB_READONLY)

        self.changecolorbtn = wx.Button(self, -1, "Change Circle Color", pos=(180, 7))

        posv = 40

        self.slider_label = wx.StaticText(self, -1, "Imin: ", (5, posv + 5))

        self.vminctrl = wx.SpinCtrl(self, -1, "1", pos=(50, posv), size=(80, -1), min=-200, max=100000)

        # second horizontal band
        self.slider_label2 = wx.StaticText(self, -1, "Imax: ", (5, posv + 35))

        self.vmaxctrl = wx.SpinCtrl(self, -1, "1000", pos=(50, posv + 30),
                                                                size=(80, -1), min=2, max=1000000)

        self.slider_vmin = wx.Slider(self, -1, pos=(150, posv + 5), size=(220, -1),
                                    value=0, minValue=0, maxValue=1000,
                                    style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_vmin.SetTickFreq(500)
        else:
            self.slider_vmin.SetTickFreq(500, 1)

        self.slider_vmax = wx.Slider(self, -1, pos=(150, posv + 35), size=(220, -1), value=1000,
                                    minValue=1, maxValue=1000,
                                    style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_vmax.SetTickFreq(500)
        else:
            self.slider_vmax.SetTickFreq(500, 1)

        self.Iminvaltxt = wx.StaticText(self, -1, "0", pos=(400, posv + 5))
        self.Imaxvaltxt = wx.StaticText(self, -1, "1000", pos=(400, posv + 35))

        self.chck_scaletypeplot = wx.CheckBox(self, -1, "Logscale", pos=(5, posv + 70))
        self.chck_scaletypeplot.SetValue(True)

        self.chck_removebckg = wx.CheckBox(self, -1, "Remove Background", pos=(145, posv + 70))
        self.chck_removebckg.SetValue(False)

        wx.StaticText(self, -1, "Image:", pos=(5, posv + 107))
        wx.StaticText(self, -1, "folder:", pos=(70, posv + 107))
        wx.StaticText(self, -1, "filename:", pos=(70, posv + 137))

        self.folderexpimagetxtctrl = wx.TextCtrl(
            self, -1, "", size=(240, -1), pos=(140, posv + 105))
        self.expimagetxtctrl = wx.TextCtrl(
            self, -1, "", size=(240, -1), pos=(140, posv + 135))
        self.expimagebrowsebtn = wx.Button(
            self, -1, "Open", size=(50, -1), pos=(5, posv + 165))
        self.btncloseimage = wx.Button(
            self, -1, "Close", size=(50, -1), pos=(100, posv + 165))
        self.btncloseimage.Disable()

        # BINDS
        self.comboLUT.Bind(wx.EVT_COMBOBOX, self.OnChangeLUT)
        self.Bind(wx.EVT_SPINCTRL, self.OnSpinCtrl_IminDisplayed, self.vminctrl)
        self.Bind(wx.EVT_SPINCTRL, self.OnSpinCtrl_ImaxDisplayed, self.vmaxctrl)
        self.slider_vmin.Bind(
            wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_slider_IminDisplayed)
        self.slider_vmax.Bind(
            wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_slider_ImaxDisplayed)
        self.chck_scaletypeplot.Bind(wx.EVT_CHECKBOX, self.onChangeScaleTypeplot)
        self.chck_removebckg.Bind(wx.EVT_CHECKBOX, self.onChangebckgremoval)
        self.changecolorbtn.Bind(wx.EVT_BUTTON, self.onChangeColor)
        self.expimagebrowsebtn.Bind(wx.EVT_BUTTON, self.OpenImage)
        self.btncloseimage.Bind(wx.EVT_BUTTON, self.onCloseImage)

        #-------------- tooltips
        t1 = "Change Look-up-Table of Colors"
        tmin = "Set min. Intensity displayed"
        tmax = "Set max. Intensity displayed"
        tscale = "Set Linear or Log pixel intensity scale"

        self.comboLUT.SetToolTipString(t1)
        self.vminctrl.SetToolTipString(tmin)
        self.slider_vmin.SetToolTipString(tmin)
        self.vmaxctrl.SetToolTipString(tmax)
        self.slider_vmax.SetToolTipString(tmax)
        self.chck_scaletypeplot.SetToolTipString(tscale)
        self.changecolorbtn.SetToolTipString(
            "Change Color of Simulated Spots hollow circles ")
        self.expimagebrowsebtn.SetToolTipString(
            "Open new or update displayed Image according to image filename field")

    def OpenImage(self, _):
        """ launch file selector and launch update_ImageArray"""
        #         print "self.ImageArray in OpenImage", self.ImageArray
        if hasattr(self, "fullpathimagefile"):
            if self.fullpathimagefile in ("", None, " "):
                self.onSelectImageFile(1)
        else:
            self.onSelectImageFile(1)

        self.readnewimage()
        self.update_ImageArray()

    def onSelectImageFile(self, evt):
        """ launch file dialog and set GUI attributes
        self.folderexpimagetxtctrl and self.expimagetxtctrl"""
        self.GetfullpathFile(evt)

        folder, imagefile = os.path.split(self.fullpathimagefile)
        self.folderexpimagetxtctrl.SetValue(folder)
        self.expimagetxtctrl.SetValue(imagefile)

    def GetfullpathFile(self, _):
        """ open file dialog and set self.fullpathimagefile"""
        myFileDialog = wx.FileDialog(self, "Choose an image file", style=wx.OPEN,
                                    # defaultDir=self.dirname,
                                    wildcard=DictLT.getwildcardstring(self.parent.CCDLabel))
        dlg = myFileDialog
        dlg.SetMessage("Choose an image file")
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()

            #             self.dirnameBlackList = dlg.GetDirectory()
            self.fullpathimagefile = str(filename)
        else:
            pass

    def readnewimage(self):
        """ open image from GUI path txtctrls
        set self.ImageArray
        set self.fullpathimagefile
        if parent exists:
        set self.parent.fullpathimagefile"""
        folder = str(self.folderexpimagetxtctrl.GetValue())
        imagename = str(self.expimagetxtctrl.GetValue())
        self.fullpathimagefile = os.path.join(folder, imagename)

        if self.fullpathimagefile in ("", None, " "):
            self.onSelectImageFile(1)

        print("read %s" % self.fullpathimagefile)

        if not os.path.isfile(self.fullpathimagefile):
            dlg = wx.MessageDialog(self,
                                    "Path to image file : %s\n\ndoes not exist!!" % self.fullpathimagename,
                                    "error", wx.OK | wx.ICON_ERROR)

            dlg.ShowModal()
            dlg.Destroy()
            return

        CCDLabel = self.parent.CCDLabel

        print("using CCDLabel", CCDLabel)

        ImageArray, _, _ = IOimage.readCCDimage(self.fullpathimagefile,
                                                                            CCDLabel, dirname=None)

        self.ImageArray = ImageArray
        if self.parent is not None:
            self.parent.fullpathimagefile = self.fullpathimagefile

    def update_ImageArray(self):
        """ set self.parent.ImageArray and call updateplot()"""
        self.parent.ImageArray = self.ImageArray
        self.parent.ImageArrayInit = self.ImageArray
        self.parent.ImageArrayMinusBckg = None

        self.btncloseimage.Enable()
        self.updateplot()

    def onCloseImage(self, _):
        """ clear image from plot and call updateplot()"""
        self.ImageArray = None
        self.parent.ImageArray = self.ImageArray
        self.btncloseimage.Disable()
        self.updateplot()

    def OnChangeLUT(self, _):
        """ change LUT"""
        self.data_dict["lut"] = self.comboLUT.GetValue()

        print("now selected lut:%s" % self.data_dict["lut"])
        self.updateplot()

    def onChangeColor(self, _):
        """ change marker coler """
        self.colorindex += 1
        self.colorindex = self.colorindex % 8
        self.data_dict["markercolor"] = self.dict_colors[self.colorindex]

        print("now selected markercolor:%s" % self.data_dict["markercolor"])

        self.updateplot()

    def OnSpinCtrl_IminDisplayed(self, _):
        """ set image plot vmin vmax.
        set self.IminDisplayed, self.ImaxDisplayed and call self.normalizeplot() """
        self.IminDisplayed = self.vminctrl.GetValue()
        self.ImaxDisplayed = self.vmaxctrl.GetValue()

        if self.IminDisplayed >= self.ImaxDisplayed:
            self.IminDisplayed = self.ImaxDisplayed - 1

        self.slider_vmin.SetMin(int(self.IminDisplayed))
        self.slider_vmax.SetMin(int(self.IminDisplayed))
        self.normalizeplot()

    def OnSpinCtrl_ImaxDisplayed(self, _):
        """ set image plot vmin vmax.
        set self.IminDisplayed, self.ImaxDisplayed and call self.normalizeplot() """
        self.IminDisplayed = self.vminctrl.GetValue()
        self.ImaxDisplayed = self.vmaxctrl.GetValue()

        if self.IminDisplayed >= self.ImaxDisplayed:
            self.ImaxDisplayed = self.IminDisplayed + 1

        self.slider_vmax.SetMax(int(self.ImaxDisplayed))
        self.slider_vmin.SetMax(int(self.ImaxDisplayed))

        self.normalizeplot()

    def normalizeplot(self):
        """ set self.data_dict["vmin"] and self.data_dict["vmax"]
        call self.updateplot()"""
        self.data_dict["vmin"] = self.IminDisplayed
        self.data_dict["vmax"] = self.ImaxDisplayed

        #         print "self.data_dict in normalizeplot", self.data_dict

        self.updateplot()

    def onChangeScaleTypeplot(self, _):
        """ switch change between linear or logscale"""
        self.data_dict["logscale"] = not self.data_dict["logscale"]

        print("Now logscale is %s" % str(self.data_dict["logscale"]))

        if self.data_dict["logscale"]:
            positivemin = max(1, int(self.vminctrl.GetValue()))
            self.vminctrl.SetValue(positivemin)
            self.data_dict["vmin"] = positivemin
            self.data_dict["Imin"] = positivemin
            self.slider_vmin.SetMin(positivemin)

            self.on_slider_IminDisplayed(1)

        self.updateplot()

    def onChangebckgremoval(self, _):
        """ switch remove auto background or not"""
        self.data_dict["removebckg"] = not self.data_dict["removebckg"]

        print("Now removebckg state is %s" % str(self.data_dict["removebckg"]))

        if self.data_dict["removebckg"]:
            # data to display are now
            pass

        self.updateplot()


    def on_slider_IminDisplayed(self, _):
        """ set self.IminDisplayed and call self.normalizeplot() """
        self.IminDisplayed = self.slider_vmin.GetValue()

        #         self.viewingLUTpanel.vminctrl.SetValue(int(self.IminDisplayed))

        if self.ImaxDisplayed <= self.IminDisplayed:
            self.IminDisplayed = self.ImaxDisplayed - 1
            self.slider_vmin.SetValue(self.IminDisplayed)

        self.displayIMinMax()
        self.normalizeplot()

    def on_slider_ImaxDisplayed(self, _):
        """ set  self.ImaxDisplayed and call self.normalizeplot() """
        self.ImaxDisplayed = self.slider_vmax.GetValue()
        #         self.viewingLUTpanel.vmaxctrl.SetValue(int(self.ImaxDisplayed))

        #        print "self.ImaxDisplayed", self.ImaxDisplayed
        #         self.viewingLUTpanel.slider_vmax.SetValue(self.ImaxDisplayed)

        if self.ImaxDisplayed <= self.IminDisplayed:
            self.ImaxDisplayed = self.IminDisplayed + 1
            self.slider_vmax.SetValue(self.ImaxDisplayed)

        self.displayIMinMax()
        self.normalizeplot()

    def displayIMinMax(self):
        """ set GUI txt"""
        self.Iminvaltxt.SetLabel(str(self.IminDisplayed))
        self.Imaxvaltxt.SetLabel(str(self.ImaxDisplayed))

    def updateplot(self):
        """ call parent methods: setplotlimits_fromcurrentplot() and _replot()"""
        #         self.parent.xlim = self.xlim
        #         self.parent.ylim = self.ylim
        #         self.parent.fullpathimagefile = self.fullpathimagefile
        self.parent.setplotlimits_fromcurrentplot()
        self.parent._replot()

    def readvalues(self):
        """ set self.xlim, self.ylim from txtctrls"""
        xmin = float(self.txtctrl_xmin.GetValue())
        xmax = float(self.txtctrl_xmax.GetValue())
        ymin = float(self.txtctrl_ymin.GetValue())
        ymax = float(self.txtctrl_ymax.GetValue())

        self.xlim = (xmin, xmax)
        self.ylim = (ymin, ymax)

    def onAccept(self, _):
        """ update plot limits values and close
        set parent xlim and ylim attributes"""
        self.readvalues()
        self.parent.xlim = self.xlim
        self.parent.ylim = self.ylim
        self.Close()

    def onCancel(self, _):
        """ close
        keep initial parent xlim and ylim attributes"""
        self.parent.xlim = self.init_xlim
        self.parent.ylim = self.init_ylim
        self.Close()


# --- ---------------  Plot limits board  parameters
class NoiseLevelBoard(wx.Dialog):
    """
    Class to set noise level ONLY when refining crystal strain and orientation
    """
    def __init__(self, parent, _id, title, data_dict):
        """
        initialize board window
        """
        wx.Dialog.__init__(self, parent, _id, title, size=(400, 150))

        self.parent = parent
        print("self.parent in NoiseLevelBoard ", self.parent)

        self.data_dict = data_dict

        posv = 5

        wx.StaticText(self, -1, "RMS Radial pixel noise", pos=(5, posv + 57))

        self.noisetxtctrl = wx.TextCtrl(self, -1, "0", size=(150, -1), pos=(120, posv + 55))
        self.applynoisebtn = wx.Button(self, -1, "Apply", size=(70, -1), pos=(300, posv + 55))

        self.applynoisebtn.Bind(wx.EVT_BUTTON, self.onApplyNoise)

    def onApplyNoise(self, _):
        """ set parent.sigmanoise and call parent._replot()"""
        self.parent.sigmanoise = float(self.noisetxtctrl.GetValue())
        self.parent._replot()


def get2Dlimits(X, Y):
    """ return xlim, ylim from min max of X and Y"""
    Xmin = min(X)
    Xmax = max(X)
    Ymin = min(Y)
    Ymax = max(Y)
    xlim = Xmin, Xmax
    ylim = Ymin, Ymax
    return xlim, ylim


if __name__ == "__main__":

    class App(wx.App):
        """ App to launch IntensityScaleBoard"""
        data_dict = {}

        Imin = 1
        Imax = 1000
        vmin = 5
        vmax = 500
        lut = "jet"

        data_dict["Imin"] = Imin
        data_dict["Imax"] = Imax
        data_dict["vmin"] = vmin
        data_dict["vmax"] = vmax
        data_dict["lut"] = lut

        def OnInit(self):
            """Create the main window and insert the custom frame"""
            dlg = IntensityScaleBoard(None, -1, "Laue Simulation Frame", data_dict=self.data_dict)
            dlg.Show(True)
            return True

    app = App(0)
    app.MainLoop()
