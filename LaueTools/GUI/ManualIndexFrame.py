# -*- coding: utf-8 -*-
r"""
GUI class to index manually laue pattern

This module belongs to the open source LaueTools project with a free code repository at
https://gitlab.esrf.fr/micha/lauetools
mailto: micha -at* esrf *dot- fr

March 2020
"""
from __future__ import division
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
import sys
import copy

import matplotlib
from matplotlib import __version__ as matplotlibversion

matplotlib.use("WXAgg")

from matplotlib.figure import Figure

from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                NavigationToolbar2WxAgg as NavigationToolbar)

from pylab import FuncFormatter, Rectangle

import numpy as np
import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

if sys.version_info.major == 3:
    from .. import indexingAnglesLUT as INDEX
    from .. import indexingImageMatching as IIM
    from .. import indexingSpotsSet as ISS
    from .. import lauecore as LAUE
    from .. import LaueGeometry as F2TC
    from . import LaueSpotsEditor as LSEditor
    from .. import CrystalParameters as CP
    from .. import generaltools as GT
    from .. import dict_LaueTools as DictLT
    from .. import dragpoints as DGP
    from . import threadGUI2 as TG
    from . ResultsIndexationGUI import RecognitionResultCheckBox
    from . import PlotLimitsBoard
    from .. import imageprocessing as ImProc
else:
    import indexingAnglesLUT as INDEX
    import indexingImageMatching as IIM
    import indexingSpotsSet as ISS
    import lauecore as LAUE
    import LaueGeometry as F2TC
    import CrystalParameters as CP
    import generaltools as GT
    import dict_LaueTools as DictLT
    import dragpoints as DGP
    import GUI.LaueSpotsEditor as LSEditor
    from GUI.ResultsIndexationGUI import RecognitionResultCheckBox
    import GUI.threadGUI2 as TG
    import GUI.PlotLimitsBoard
    import imageprocessing as ImProc

# --- ---------------  Manual indexation Frame
class ManualIndexFrame(wx.Frame):
    """
    Class to implement a window enabling manual indexation

    parent must have:
    CCDLabel
    dict_Materials
    ClassicalIndexation_Tabledist
    DataPlot_filename

    kf_direction
    defaultParam
    detectordiameter
    pixelsize
    framedim

    dict_Rot
    """
    def __init__(self, parent, _id, title, data=(1, 1, 1, 1), data_added=None,
                                            datatype="2thetachi",
                                            kf_direction="Z>0",
                                            element="Ge",
                                            Params_to_simulPattern=None,  # Grain, Emin, Emax
                                            DRTA=0.5,
                                            MATR=0.5,
                                            indexation_parameters=None,
                                            StorageDict=None,
                                            DataSetObject=None):

        wx.Frame.__init__(self, parent, _id, title, size=(600, 1000))

        self.panel = wx.Panel(self)

        self.parent = parent

        # dict related to image data
        self.data_dict = {}
        self.data_dict["Imin"] = 1
        self.data_dict["Imax"] = 1000
        self.data_dict["vmin"] = 1
        self.data_dict["vmax"] = 1000
        self.data_dict["lut"] = "jet"
        self.data_dict["fullpathimagefile"] = ""
        self.data_dict["CCDLabel"] = self.parent.CCDLabel
        self.data_dict["ImageArray"] = None
        self.data_dict["logscale"] = True
        self.data_dict["markercolor"] = "b"
        self.data_dict["removebckg"] = False

        if self.data_dict["CCDLabel"] in ("MARCCD165", "ROPER"):
            # flip Y axis for marccd type image display
            self.flipyaxis = True
            # TODO to include in data_dict
        else:
            self.flipyaxis = False

        self.CCDLabel = self.parent.CCDLabel

        self.data_dict["flipyaxis"] = self.flipyaxis

        self.ImageArray = None
        self.init_plot = True

        self.getlimitsfromplot = False

        self.datatype = datatype

        self.factorsize = None
        self.powerscale = None
        self.toreturn = None
        self.tth, self.chi, self.pixelX, self.pixelY = None, None, None, None
        self.gnomonX, self.gnomonY = None, None
        self.xlim = None
        self.ylim = None
        self.locatespotmarkersize = 1

        self.centerx, self.centery = None, None
        self.listbuttonstate = []
        self.twospots = None
        self._dataANNOTE_exp = None

        self.select_chi, self.select_theta = None, None
        self.select_dataX, self.select_dataY = None, None
        self.select_I = None
        self.set_central_spots_hkl = None
        self.spot_index_central = None
        self.Nb_criterium, self.NBRP, self.B, self.energy_max = None, None, None, None

        self.ResolutionAngstrom, self.nLUT = None, None
        self.rough_tolangle, self.resindexation = None, None

        self.fine_tolangle, self.TGframe, self.worker = None, None, None

        self.UBs_MRs, self.bestmat = None, None
        self.TwicethetaChi_solution = None


        # depending of datatype self.Data_X, self.Data_Y can be 2theta, chi or gnomonX,gnomonY, or pixelX,pixelY
        # print "data",data
        # Data_X, Data_Y, Data_I, File_NAME = data

        #         self.data = data
        #         #self.alldata = copy.copy(data)
        #         self.data_2thetachi = data_2thetachi
        #         self.data_XY = data_XY

        self.indexation_parameters = indexation_parameters

        print("self.indexation_parameters['detectordiameter']",
            self.indexation_parameters["detectordiameter"])

        self.pixelsize = self.indexation_parameters["pixelsize"]

        if indexation_parameters is not None:
            DataToIndex = self.indexation_parameters["DataToIndex"]
            if self.datatype is "2thetachi":
                self.Data_X = 2.0 * DataToIndex["data_theta"]
                self.Data_Y = DataToIndex["data_chi"]
            elif self.datatype is "pixels":
                self.Data_X = DataToIndex["data_X"]
                self.Data_Y = DataToIndex["data_Y"]

            elif self.datatype is "gnomon":
                self.Data_X = DataToIndex["data_gnomonX"]
                self.Data_Y = DataToIndex["data_gnomonY"]
                self.data_gnomonXY = (
                    DataToIndex["data_gnomonX"],
                    DataToIndex["data_gnomonY"])

            self.Data_I = DataToIndex["data_I"]
            self.File_NAME = self.indexation_parameters["DataPlot_filename"]
            self.data_XY = DataToIndex["data_X"], DataToIndex["data_Y"]

            self.data_2thetachi = 2 * DataToIndex["data_theta"], DataToIndex["data_chi"]

            self.data = self.Data_X, self.Data_Y, self.Data_I, self.File_NAME
            self.alldata = copy.copy(data)
            self.selectedAbsoluteSpotIndices_init = DataToIndex[
                "current_exp_spot_index_list"]

            self.selectedAbsoluteSpotIndices = copy.copy(
                self.selectedAbsoluteSpotIndices_init)

        # create attributes X Y tth chi
        self.init_data()
        self.DataSet = DataSetObject

        self.kf_direction = kf_direction
        # print("self.kf_direction in ManualIndexFrame", self.kf_direction)

        self.current_matrix = []
        self.Millerindices = None

        # simulation parameters
        # self.SimulParam =(grain, emin, self.emax.GetValue())
        self.SimulParam = Params_to_simulPattern

        # simulated 2theta,chi
        self.data_theo = data_added
        # overwrite self.data_theo
        #        self.Simulate_Pattern()
        # self.data_theo = data_added # a way in the past to put simulated data

        self.points = []  # to store points
        self.selectionPoints = []
        self.twopoints = []
        self.nbclick = 1
        self.nbclick_dist = 0
        self.clicked_indexSpot = []

        self.recognition_possible = True
        self.toshow = []

        # draggable line for gnomon data
        self.dragLines = []
        self.addlines = False

        self.onlyclosest = 1

        # absolute exp. spot index list after using filter data button
        self.abs_spotindex = None

        # defaut value for Miller attribution
        self.B0matrix = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]  # means: columns are a*,b*,c* in xyz frame
        self.key_material = element
        self.dict_Materials = self.indexation_parameters['dict_Materials']
        self.detectordistance = None

        self.DRTA = DRTA
        self.MATR = MATR

        self.Params_to_simulPattern = Params_to_simulPattern

        self.StorageDict = StorageDict

        self.indexation_parameters["mainAppframe"] = self.parent
        self.indexation_parameters["indexationframe"] = self

        self.initGUI()

    def initGUI(self):
        """ init GUI of ManualIndexFrame """
        self.sb = self.CreateStatusBar()

        colourb_bkg = [242, 241, 240, 255]
        colourb_bkg = np.array(colourb_bkg) / 255.0

        self.dpi = 100
        self.figsizex, self.figsizey = 4, 3
        self.fig = Figure(
            (self.figsizex, self.figsizey), dpi=self.dpi, facecolor=tuple(colourb_bkg))
        self.fig.set_size_inches(self.figsizex, self.figsizey, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas)

        self.fig.canvas.mpl_connect("button_press_event", self.onClick)

        self.tooltip = wx.ToolTip(tip="Welcome on LaueTools UB Refinement board")
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)

        self.pickdistbtn = wx.ToggleButton(self.panel, 2, "Pick distance")  # T2
        self.recongnisebtn = wx.ToggleButton(
            self.panel, 3, "Recognise distance", size=(150, 40)) # T3
        self.pointButton6 = wx.ToggleButton(self.panel, 6, "Show Exp. Spot Props")  # T6

        self.listbuttons = [self.pickdistbtn, self.recongnisebtn, self.pointButton6]

        self.defaultColor = self.GetBackgroundColour()
        self.p2S, self.p3S, self.p6S = 0, 0, 0
        self.listbuttonstate = [self.p2S, self.p3S, self.p6S]
        self.listbuttonstate = [0, 0, 0]

        self.Bind(wx.EVT_TOGGLEBUTTON, self.T2, id=2)  # pick distance
        self.Bind(wx.EVT_TOGGLEBUTTON, self.T3, id=3)  # recognise distance
        self.Bind(wx.EVT_TOGGLEBUTTON, self.T6, id=6)  # show Exp. spot props

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        self.txttolerances = wx.StaticText(self.panel, -1, "Tolerance angles (deg)")
        self.txttolerances.SetFont(font3)

        self.txt1 = wx.StaticText(self.panel, -1, "Dist. Recognition      ")
        # self.DRTA = wx.TextCtrl(self.parampanel, -1,'0.5',(350, 10))
        self.DRTA = wx.TextCtrl(self.panel, -1, str(self.DRTA))
        self.txt2 = wx.StaticText(self.panel, -1, "Spots Matching")
        # self.matr_ctrl = wx.TextCtrl(self.parampanel, -1,'0.5',(350, 40))
        self.matr_ctrl = wx.TextCtrl(self.panel, -1, str(self.MATR))

        self.txtnlut = wx.StaticText(self.panel, -1, "n LUT:    ")
        self.nlut = wx.SpinCtrl(self.panel, -1, "4", min=1, max=7, size=(50, -1))

        self.nlut.Bind(wx.EVT_SPINCTRL, self.onNlut)

        self.slider_exp = wx.Slider(self.panel, -1, 50, 0, 100, size=(120, -1))
        self.slidertxt_exp = wx.StaticText(self.panel, -1, "Exp. spot size", (5, 5))
        self.slider_exp.Bind(wx.EVT_SLIDER, self.sliderUpdate_exp)

        self.slider_ps = wx.Slider(self.panel, -1, 50, 0, 100, size=(120, -1))
        self.slidertxt_ps = wx.StaticText(self.panel, -1, "power scale", (5, 5))
        self.slider_ps.Bind(wx.EVT_SLIDER, self.sliderUpdate_ps)

        self.findspotchck = wx.CheckBox(self.panel, -1, "Show Spot Index")
        self.findspotchck.SetValue(False)
        self.spotindexspinctrl = wx.SpinCtrl(self.panel, -1, "0", min=0,
                                    max=len(self.data[0]) - 1, size=(70, -1))
        self.findspotchck.Bind(wx.EVT_CHECKBOX, self.locateSpot)
        self.spotindexspinctrl.Bind(wx.EVT_SPINCTRL, self.locateSpot)

        self.UCEP = wx.CheckBox(self.panel, -1, "Select closest Exp. Spots")
        self.UCEP.SetValue(True)
        self.UCEP.Disable()

        self.COCD = wx.CheckBox(self.panel, -1, "Consider only closest distance")
        self.COCD.SetValue(True)
        self.COCD.Disable()
        self.MWFD = wx.CheckBox(self.panel, -1, "Matching Rate computed with filtered Data")
        self.MWFD.SetValue(False)

        self.txtEbandpass = wx.StaticText(self.panel, -1, "Energy Bandpass (keV): ")
        self.txtEbandpass.SetFont(font3)
        self.emintxt = wx.StaticText(self.panel, -1, "min.                       ")
        self.SCEmin = wx.SpinCtrl(self.panel, -1, "5", min=5, max=49)
        self.emaxtxt = wx.StaticText(self.panel, -1, "max. ")

        if self.datatype == "gnomon":
            self.drawlinebtn = wx.Button(self.panel, -1, "Draw line")
            self.drawlinebtn.Bind(wx.EVT_BUTTON, self.OnDrawLine)

            self.clearlinesbtn = wx.Button(self.panel, -1, "Clear lines")
            self.clearlinesbtn.Bind(wx.EVT_BUTTON, self.OnClearLines)

        if self.Params_to_simulPattern is not None:
            self.SCEmax = wx.SpinCtrl(self.panel, -1, str(self.Params_to_simulPattern[2]), min=6, max=150)
        else:
            self.SCEmax = wx.SpinCtrl(self.panel, -1, "22", min=6, max=150)

        self.txtelem = wx.StaticText(self.panel, -1, "Elem: ")
        self.list_materials = sorted(self.parent.dict_Materials.keys())
        self.combokeymaterial = wx.ComboBox(self.panel, -1, self.key_material,
                                    choices=self.list_materials, style=wx.CB_READONLY)
        self.combokeymaterial.Bind(wx.EVT_COMBOBOX, self.EnterCombokeymaterial)

        self.sethklchck = wx.CheckBox(self.panel, -1, "set spot1 hkl")
        self.sethklcentral = wx.TextCtrl(self.panel, -1, "[1,0,0]", size=(100, -1))

        self.filterDatabtn = wx.Button(self.panel, -1, "Filter Exp. Data")
        self.filterDatabtn.Bind(wx.EVT_BUTTON, self.BuildDataDict)

        self.imagescalebtn = wx.Button(self.panel, -1, "Set Image Scale")
        self.imagescalebtn.Bind(wx.EVT_BUTTON, self.onSetImageScale)
        self.imagescalebtn.Disable()
        if self.datatype == "pixels":
            self.imagescalebtn.Enable()

        self.plotlimitsbtn = wx.Button(self.panel, -1, "Plot limits")
        self.plotlimitsbtn.Bind(wx.EVT_BUTTON, self.SetPlotLimits)

        self.spotsizebtn = wx.Button(self.panel, -1, "Spot Size LUT")
        self.spotsizebtn.Bind(wx.EVT_BUTTON, self.SetSpotSize)
        self.spotsizebtn.Disable()

        self.txtctrldistance = wx.StaticText(self.panel, -1, "==> %s deg" % "", size=(80, -1))

        self.verbose = wx.CheckBox(self.panel, -1, "display details")
        self.verbose.SetValue(False)

        # for annotation
        self.drawnAnnotations_exp = {}
        self.links_exp = []

        self._layout()
        self.initplotlimits()
        self._replot()

        # tooltips
        entip = ("set energy range to simulate Laue pattern from potential orientation matrix "
                "found by recognised angular distance")
        self.emintxt.SetToolTipString(entip)
        self.SCEmin.SetToolTipString(entip)
        self.emaxtxt.SetToolTipString(entip)
        self.SCEmax.SetToolTipString(entip)

        nlutip = "Largest integer of miller indices used to build the angular distances reference database table (LUT)"
        self.nlut.SetToolTipString(nlutip)
        self.txtnlut.SetToolTipString(nlutip)

        self.pickdistbtn.SetToolTipString(
            "Compute distance between the two next clicked spots or points in plot.\nThe "
            "angle is the separation angle between the two corresponding lattice planes normals")

        rectip = "Press this button and then click on two spots that are likely to have small hkl indices such as to be recognised"
        rectip += "Index spots from separation angles between lattice planes normals from the two next clicked spots or points in plot.\n"
        rectip += "Each angular distance found in reference structure LUT will lead to potential orientation matrices from which a Laue Pattern can be simulated.\n"
        rectip += "Matching rate is given by the number of close pairs of exp. and simulated spots.\n"
        rectip += "All separation distances between first clicked spot and a set from the most intense (index 0) to the 2nd clicked spot index will tested"
        self.recongnisebtn.SetToolTipString(rectip)

        self.pointButton6.SetToolTipString(
            "Display info of experimental spots by clicking on them")

        self.UCEP.SetToolTipString(
            "Select nearest spot position or exact clicked position")

        sethkltip = "Set the [h,k,l] Miller indices of the first clicked spot. "
        "For cubic structure (simple, body & face centered, diamond etc...) "
        "l index must be positive"
        self.sethklchck.SetToolTipString(sethkltip)
        self.sethklcentral.SetToolTipString(sethkltip)

        self.filterDatabtn.SetToolTipString("Filter (select, remove) experimental spots to display")

        self.MWFD.SetToolTipString(
            "Compute matching rate of simulated Laue pattern with filtered set of experiment spots")

        tipshow = "Show on plot experimental spot with given absolute index"
        self.findspotchck.SetToolTipString(tipshow)
        self.spotindexspinctrl.SetToolTipString(tipshow)

    def _layout(self):
        """ set the widgets layout """
        lutbox = wx.BoxSizer(wx.HORIZONTAL)
        lutbox.Add(self.txtnlut, 0, wx.ALL)
        lutbox.Add(self.nlut, 0, wx.ALL)
        lutbox.Add(self.spotsizebtn, 0, wx.ALL)

        ang1box = wx.BoxSizer(wx.VERTICAL)
        ang1box.Add(self.txt1, 0, wx.ALL)
        ang1box.Add(self.DRTA, 0, wx.ALL)

        ang2box = wx.BoxSizer(wx.VERTICAL)
        ang2box.Add(self.txt2, 0, wx.ALL)
        ang2box.Add(self.matr_ctrl, 0, wx.ALL)

        anglesbox = wx.BoxSizer(wx.HORIZONTAL)
        anglesbox.Add(ang1box, 0, wx.ALL)
        anglesbox.Add(ang2box, 0, wx.ALL)

        eminbox = wx.BoxSizer(wx.VERTICAL)
        eminbox.Add(self.emintxt, 0, wx.ALL)
        eminbox.Add(self.SCEmin, 0, wx.ALL)

        emaxbox = wx.BoxSizer(wx.VERTICAL)
        emaxbox.Add(self.emaxtxt, 0, wx.ALL)
        emaxbox.Add(self.SCEmax, 0, wx.ALL)

        henergiesbox = wx.BoxSizer(wx.HORIZONTAL)
        henergiesbox.Add(eminbox, 0, wx.ALL)
        henergiesbox.Add(emaxbox, 0, wx.ALL)

        sizerparam = wx.BoxSizer(wx.VERTICAL)
        sizerparam.Add(lutbox, 0, wx.ALL)
        sizerparam.AddSpacer(5)

        sizerparam.Add(self.txttolerances, 0, wx.ALL)
        sizerparam.Add(anglesbox, 0, wx.ALL)
        sizerparam.Add(self.txtEbandpass, 0, wx.ALL)
        sizerparam.AddSpacer(5)
        sizerparam.Add(henergiesbox, 0, wx.ALL)
        sizerparam.Add(self.txtelem, 0, wx.ALL)
        sizerparam.Add(self.combokeymaterial, 0, wx.ALL)

        btnSizer0 = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer0.Add(self.pointButton6, 0, wx.ALL)
        btnSizer0.Add(self.filterDatabtn, 0, wx.ALL)
        btnSizer0.Add(self.plotlimitsbtn, 0, wx.ALL)
        btnSizer0.Add(self.imagescalebtn, 0, wx.ALL)

        btnSizerfindspot = wx.BoxSizer(wx.HORIZONTAL)
        btnSizerfindspot.Add(self.findspotchck, 0, wx.ALL)
        btnSizerfindspot.Add(self.spotindexspinctrl, 0, wx.ALL)

        boxsliders = wx.BoxSizer(wx.HORIZONTAL)
        boxsliders.Add(self.slider_exp, 0, wx.ALL)
        boxsliders.Add(self.slidertxt_exp, 0, wx.ALL)
        boxsliders.Add(self.slider_ps, 0, wx.ALL)
        boxsliders.Add(self.slidertxt_ps, 0, wx.ALL)

        if self.datatype == "gnomon":
            Sizerline = wx.BoxSizer(wx.HORIZONTAL)
            Sizerline.Add(self.drawlinebtn, 0, wx.ALL)
            Sizerline.Add(self.clearlinesbtn, 0, wx.ALL)

        btnSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer1.Add(self.pickdistbtn, 0, wx.ALL, 0)
        btnSizer1.Add(self.txtctrldistance, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.recongnisebtn, 0, wx.ALL, 0)
        hsizer.Add(self.sethklchck, 0, wx.ALL, 0)
        hsizer.Add(self.sethklcentral, 0, wx.ALL, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(self.UCEP, 0, wx.ALL, 0)
        hsizer2.Add(self.verbose, 0, wx.ALL, 0)

        btnsSizer = wx.BoxSizer(wx.VERTICAL)
        btnsSizer.Add(btnSizer0, 0, wx.ALL)
        btnsSizer.Add(btnSizerfindspot, 0, wx.ALL, 5)
        btnsSizer.Add(boxsliders, 0, wx.ALL, 5)
        if self.datatype == "gnomon":
            btnsSizer.Add(Sizerline, 0, wx.ALL)
        btnsSizer.Add(sizerparam, 0, wx.ALL)
        btnsSizer.AddSpacer(2)
        btnsSizer.Add(btnSizer1, 0, wx.ALL)
        btnsSizer.AddSpacer(5)
        btnsSizer.Add(hsizer, 0, wx.ALL)
        btnsSizer.Add(hsizer2, 0, wx.ALL)
        btnsSizer.Add(self.COCD, 0, wx.ALL)
        btnsSizer.Add(self.MWFD, 0, wx.ALL)

        sizerplot = wx.BoxSizer(wx.VERTICAL)
        sizerplot.Add(self.canvas, 1, wx.TOP | wx.GROW)
        sizerplot.Add(self.toolbar, 0, wx.EXPAND)

        sizerH = wx.BoxSizer(wx.HORIZONTAL)
        sizerH.Add(sizerplot, 1, wx.ALL | wx.GROW, 5)
        sizerH.Add(btnsSizer, 0, wx.ALL, 5)

        self.panel.SetSizer(sizerH)
        sizerH.Fit(self)
        self.Layout()

    def sliderUpdate_exp(self, _):
        """handle exp. spot size slider """
        self.factorsize = int(self.slider_exp.GetValue())
        self.getlimitsfromplot = True
        self._replot()
        print("factor spot size = %f " % self.factorsize)

    def sliderUpdate_ps(self, _):
        """ handle intensity scale factor"""
        # ps from -5 5
        ps = (int(self.slider_ps.GetValue()) - 50) / 10.0

        print("ps", ps)

        if ps < 1:  # from 1/5 to 1
            jump_ps = 4 / 5.0
            deltaps = 6.0
            self.powerscale = jump_ps / deltaps * (ps - 1) + 1
        if ps >= 1:
            self.powerscale = ps * 2

        self.getlimitsfromplot = True
        self._replot()
        print("powerscale factor spot size = %f " % self.powerscale)

    def onSetImageScale(self, _):
        """
        open a board to change image scale
        """
        from . PlotRefineGUI import IntensityScaleBoard

        IScaleBoard = IntensityScaleBoard(self, -1, "Image scale setting Board", self.data_dict)

        IScaleBoard.Show(True)

    def OnDrawLine(self, _):
        """ set mode on to draw line (for gnomonic data)"""
        self.addlines = True

        self.getlimitsfromplot = True
        self._replot()

    def OnClearLines(self, _):
        """ set mode off to draw line (for gnomonic data)"""
        self.addlines = False

        self.getlimitsfromplot = True
        self._replot()

    def init_data(self):
        """ set coordinates of spots depending on chosen representing space (self.datatype)"""
        if self.datatype == "2thetachi":
            #             self.tth, self.chi = self.Data_X, self.Data_Y
            self.tth, self.chi = self.data_2thetachi
            self.pixelX, self.pixelY = self.data_XY
        elif self.datatype == "gnomon":
            self.tth, self.chi = self.data_2thetachi
            self.gnomonX, self.gnomonY = self.data_gnomonXY
        elif self.datatype == "pixels":
            self.tth, self.chi = self.data_2thetachi
            self.pixelX, self.pixelY = self.data_XY
            print("pixels plot")

        # TODO to remove
        self.Data_index_expspot = np.arange(len(self.Data_X))

    def reinit_data(self):
        """ reset self.data to initial data"""
        self.data = copy.copy(self.alldata)

    def BuildDataDict_old(self, _):  # filter Exp Data spots
        """
        in ManualIndexFrame class
        launch by 'filter Exp Data spots' button

        Open spots filter editor and build a dict of data to be used for indexation
        """
        self.toreturn = None

        self.reinit_data()
        self.init_data()

        if self.datatype == "2thetachi":
            fields = ["Spot index", "2Theta", "Chi", "Intensity"]
            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
            to_put_in_dict = (np.arange(len(self.data[0])),
                                self.data[0],
                                self.data[1],
                                self.data[2])

        elif self.datatype == "gnomon":
            fields = ["Spot index",
                        "X_gmonon",
                        "Y_gmonon",
                        "Intensity",
                        "2Theta",
                        "Chi"]
            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
            to_put_in_dict = (
                                np.arange(len(self.data[0])),
                                self.Data_X,
                                self.Data_Y,
                                self.data[2],
                                self.data[0],
                                self.data[1])

        elif self.datatype == "pixels":
            fields = ["Spot index", "X_CCD", "Y_CCD", "Intensity", "2Theta", "Chi"]
            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
            to_put_in_dict = (np.arange(len(self.data[0])),
                            self.data[0],
                            self.data[1],
                            self.data[2],
                            self.tth,
                            self.chi)

        mySpotData = {}
        for k, ff in enumerate(fields):
            mySpotData[ff] = to_put_in_dict[k]
        dia = LSEditor.SpotsEditor(self, -1, "Filter Experimental Spots Data", mySpotData,
                                    func_to_call=self.readdata_fromEditor,
                                    field_name_and_order=fields)
        dia.Show(True)

    def BuildDataDict(self, _):  # filter Exp Data spots
        """
        in ManualIndexFrame class

        Filter Exp. Data Button

        Filter exp. spots data to be displayed
        """
        self.toreturn = None

        C0 = self.selectedAbsoluteSpotIndices_init
        AllDataToIndex = self.indexation_parameters["AllDataToIndex"]

        C1 = 2.0 * AllDataToIndex["data_theta"][C0]
        C2 = AllDataToIndex["data_chi"][C0]
        C3 = AllDataToIndex["data_I"][C0]
        C4 = AllDataToIndex["data_pixX"][C0]
        C5 = AllDataToIndex["data_pixY"][C0]
        C6 = AllDataToIndex["data_gnomonX"][C0]
        C7 = AllDataToIndex["data_gnomonY"][C0]

        if self.datatype is "2thetachi":
            fields = ["Spot index", "2theta", "Chi", "Intensity"]

            to_put_in_dict = C0, C1, C2, C3

        if self.datatype is "pixels":
            fields = ["Spot index", "pixelX", "pixelY", "Intensity", "2Theta", "Chi"]

            to_put_in_dict = C0, C4, C5, C3, C1, C2

        if self.datatype is "gnomon":
            fields = ["Spot index", "gnomonX", "gnomonY", "Intensity", "2Theta", "Chi"]
            to_put_in_dict = C0, C6, C7, C3, C1, C2

        mySpotData = {}
        for k, ff in enumerate(fields):
            mySpotData[ff] = to_put_in_dict[k]
        dia = LSEditor.SpotsEditor(None, -1, "Filter Experimental Spots Data", mySpotData,
                                    func_to_call=self.readdata_fromEditor,
                                    field_name_and_order=fields)
        dia.Show(True)
    def readdata_fromEditor(self, data):
        """
        update exp. spots data according to the user selected filter
        """
        #                 # update data according to the user selected filter
        #
        #         # take only the first 4 columns
        #         abs_spotindex, self.Data_X, self.Data_Y, self.Data_I = np.transpose(toreturn)[:4]
        #         self.abs_spotindex = np.array(abs_spotindex,dtype=np.int)

        selectedSpotsPropsarray = data.T

        col0 = selectedSpotsPropsarray[0]
        col1, col2, col3 = selectedSpotsPropsarray[1:4]

        print("\n****SELECTED and DISPLAYED PART OF EXPERIMENTAL SPOTS\n")
        self.selectedAbsoluteSpotIndices = np.array(col0, dtype=np.int)

        if self.datatype is "2thetachi":

            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            self.data_XY = (self.indexation_parameters["AllDataToIndex"]["data_pixX"][
                    self.selectedAbsoluteSpotIndices], self.indexation_parameters["AllDataToIndex"]["data_pixY"][self.selectedAbsoluteSpotIndices])

        elif self.datatype is "pixels":
            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            self.pixelX, self.pixelY = (self.indexation_parameters["AllDataToIndex"]["data_pixX"][
                    self.selectedAbsoluteSpotIndices], self.indexation_parameters["AllDataToIndex"]["data_pixY"][self.selectedAbsoluteSpotIndices])

        elif self.datatype is "gnomon":
            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            self.gnomonX = self.indexation_parameters["AllDataToIndex"]["data_gnomonX"][self.selectedAbsoluteSpotIndices]
            self.gnomonY = self.indexation_parameters["AllDataToIndex"]["data_gnomonY"][self.selectedAbsoluteSpotIndices]
        self._replot()

    def onMotion_ToolTip(self, evt):
        """tool tip to show data when mouse hovers on plot
        """
        if len(self.data[0]) == 0:
            return

        collisionFound = False

        xtol = 20
        ytol = 20

        xdata, ydata, _annotes = (self.Data_X, self.Data_Y, list(zip(self.Data_index_expspot,
                                                                    self.Data_I)))

        #         print "DATA:    \n\n\n", xdata[:5], ydata[:5], annotes

        if evt.xdata is not None and evt.ydata is not None:

            clickX = evt.xdata
            clickY = evt.ydata

            #             print 'clickX,clickY in onMotion_ToolTip', clickX, clickY

            annotes = []
            for x, y, a in zip(xdata, ydata, _annotes):
                if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                    annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

            if annotes == []:
                collisionFound = False
                return

            annotes.sort()
            _distance, x, y, annote = annotes[0]

            self.updateStatusBar(x, y, annote)

            self.tooltip.SetTip(
                "Spot abs. index=%d. Intensity=%.1f" % (annote[0], annote[1]))
            self.tooltip.Enable(True)
            collisionFound = True

            return

        if not collisionFound:
            pass

    def SetSpotSize(self, _):
        """ not implemented """
        wx.MessageBox("To be implemented", "info")
        pass

    def SetPlotLimits(self, _):
        """
        open a board to change plot limits
        """
        data_dict = {}
        data_dict["datatype"] = self.datatype
        data_dict["dataXmin"] = min(self.Data_X)
        data_dict["dataXmax"] = max(self.Data_X)
        data_dict["dataYmin"] = min(self.Data_Y)
        data_dict["dataYmax"] = max(self.Data_Y)
        data_dict["xlim"] = self.xlim
        data_dict["ylim"] = self.ylim
        data_dict["kf_direction"] = self.kf_direction

        print("data_dict", data_dict)

        PlotLismitsBoard = PlotLimitsBoard.PlotLimitsBoard(self, -1, "Data Plot limits Board", data_dict)

        PlotLismitsBoard.Show(True)

    def initplotlimits(self):
        """ set self.xlim and ylim from self.datatype and self.kf_direction and self.CCDLabel """
        if self.datatype == "2thetachi":
            self.locatespotmarkersize = 3
            if self.kf_direction == "Z>0":
                self.xlim = (34, 146)
                self.ylim = (-50, 50)
            elif self.kf_direction in ("X>0", "X<0"):
                self.xlim = (-1, 60)
                self.ylim = (-180, 180)

        elif self.datatype == "gnomon":
            self.locatespotmarkersize = 0.05
            self.ylim = (-0.6, 0.6)
            self.xlim = (-0.6, 0.6)

        elif self.datatype == "pixels":
            self.locatespotmarkersize = 50

            if self.CCDLabel in ("MARCCD165", "PRINCETON"):
                self.ylim = (2048, 0)
                self.xlim = (0, 2048)
            elif self.CCDLabel in ("VHR_PSI",):
                self.ylim = (3000, 0)
                self.xlim = (0, 4000)
            elif self.CCDLabel.startswith("sCMOS"):
                self.ylim = (2050, 0)
                self.xlim = (0, 2050)
            elif self.CCDLabel.startswith("psl"):
                self.ylim = (2000, 0)
                self.xlim = (0, 1500)

        self.factorsize = 50
        self.powerscale = 1.0

    def setplotlimits_fromcurrentplot(self):
        """ set self.xlim and ylim for current plot """
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
        #print("new limits x", self.xlim)
        #print("new limits y", self.ylim)

    def _replot(self):
        """
        replot Laue pattern theo. and exp. spots in ManualIndexFrame
        """
        # offsets to match imshow and scatter plot coordinates frames
        if self.datatype == "pixels":
            X_offset = 1
            Y_offset = 1
        else:
            X_offset = 0
            Y_offset = 0

        if not self.init_plot and self.getlimitsfromplot:
            self.setplotlimits_fromcurrentplot()

            self.getlimitsfromplot = False

        self.axes.clear()

        # clear the axes and replot everything
        #        self.axes.cla()

        def fromindex_to_pixelpos_x(index, _):
            """ return pixel pos X"""
            if self.datatype == "pixels":
                return int(index)
            else:
                return index

        def fromindex_to_pixelpos_y(index, _):
            """ return pixel pos Y"""
            if self.datatype == "pixels":
                return int(index)
            else:
                return index

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))

        # background image
        if self.ImageArray is not None and self.datatype == "pixels":

            # array to display: raw
            print('self.data_dict["removebckg"]',self.data_dict["removebckg"])
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

            print("self.ImageArray", self.ImageArray.shape)
            self.myplot = self.axes.imshow(self.ImageArray, interpolation="nearest")

            if not self.data_dict["logscale"]:
                norm = matplotlib.colors.Normalize(
                    vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"])
            else:
                norm = matplotlib.colors.LogNorm(
                    vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"])

            self.myplot.set_norm(norm)
            self.myplot.set_cmap(self.data_dict["lut"])

        # exp scale lin=0
        spotsizescale = 0
        s0, s_asympt, tau, smin, smax = 0, 1.0, 65000 / 3.0, 1.0, 5.0
        params_spotsize = s0, s_asympt, tau, smin, smax

        # linear scale lin=1
        spotsizescale = 1
        s0, slope, smin, smax = 0, 1.0, 1, 5.0
        params_spotsize = s0, slope, smin, smax

        # power scale lin=2
        spotsizescale = 2
        s0, s_at_Imax, powerscale, smin, smax = 0, 70000, self.powerscale, 1, 5.0
        params_spotsize = s0, s_at_Imax, powerscale, smin, smax

        if self.datatype == "pixels":
            # background image
            if self.ImageArray is not None:
                kwords = {"marker": "o",
                            "facecolor": "None",
                            "edgecolor": self.data_dict["markercolor"]}
            else:
                kwords = {"edgecolor": "None", "facecolor": self.data_dict["markercolor"]}

            self.axes.scatter(self.pixelX - X_offset, self.pixelY - Y_offset, s=self.factorsize
                * self.func_size_peakintensity(np.array(self.Data_I), params_spotsize,
                                                            lin=spotsizescale),
                                                            alpha=1.0,
                                                            **kwords)

        elif self.datatype == "2thetachi":
            self.axes.scatter(
                self.data_2thetachi[0],
                self.data_2thetachi[1],
                #                           s=self.func_size_intensity(np.array(self.Data_I), self.factorsize, 0, lin=1))
                s=self.factorsize
                * self.func_size_peakintensity(np.array(self.Data_I),
                                                        params_spotsize,
                                                        lin=spotsizescale))
            # c=self.Data_I / 50.)#, cmap = GT.SPECTRAL)
        elif self.datatype == "gnomon":
            self.axes.scatter(
                self.gnomonX - X_offset,
                self.gnomonY - Y_offset,
                #                           s=self.func_size_intensity(np.array(self.Data_I), self.factorsize, 0, lin=1))
                s=self.factorsize
                * self.func_size_peakintensity(np.array(self.Data_I),
                                                params_spotsize,
                                                lin=spotsizescale))

        # axes labels
        if self.datatype == "2thetachi":
            self.axes.set_xlabel("2theta(deg.)")
            self.axes.set_ylabel("chi(deg)")
        elif self.datatype == "gnomon":
            self.axes.set_xlabel("X gnomon")
            self.axes.set_ylabel("Y gnomon")
        elif self.datatype == "pixels":
            self.axes.set_xlabel("X pixel")
            self.axes.set_ylabel("Y pixel")

        # plot title
        if self.init_plot:
            self.axes.set_title("%s %d spots" % (self.File_NAME, len(self.Data_X)))
            self.axes.grid(True)

        # restore the zoom limits(unless they're for an empty plot)
        if self.xlim != (0.0, 1.0) or self.ylim != (0.0, 1.0):
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)

        # for gnomonic coordinates case
        if self.addlines:

            pt1 = [0.0, 0.0]
            pt2 = [0.5, 0.5]
            ptcenter = DGP.center_pts(pt1, pt2)

            circles = [DGP.patches.Circle(pt1, 0.03, fc="r", alpha=0.5),
                        DGP.patches.Circle(ptcenter, 0.03, fc="r", alpha=0.5),
                        DGP.patches.Circle(pt2, 0.03, fc="r", alpha=0.5)]

            line, = self.axes.plot([pt1[0], ptcenter[0], pt2[0]], [pt1[1], ptcenter[1], pt2[1]],
                                                                                picker=0.03,
                                                                                c="r")

            for circ in circles:
                self.axes.add_patch(circ)

            self.dragLines.append(DGP.DraggableLine(circles, line, tolerance=0.03,
                                                                    parent=self,
                                                                    datatype=self.datatype))
            self.addlines = False

        self.init_plot = False

        # redraw the display
        #        self.plotPanel.draw()
        self.canvas.draw()

    def func_size_energy(self, val, factor):
        """ spot size function from energy (val)"""
        return 400.0 * factor / (val + 1.0)

    def func_size_intensity(self, val, factor, offset, lin=1):
        """ spot size (or color?) function from intensity (val)"""
        if lin:
            return 0.1 * (factor * val + offset)
        else:  # log scale
            return factor * np.log(np.clip(val, 0.000000001, 1000000000000)) + offset

    def func_size_peakintensity(self, intensity, params, lin=1):
        """ spot size (or color?) function from intensity (val)"""
        if lin == 1:
            s0, slope, smin, smax = params
            s = np.clip(slope * intensity + s0, smin, smax)
        elif lin == 0:  # log scale
            s0, _, tau, smin, smax = params
            s = np.clip((smax - smin) * (1 - np.exp(-intensity / tau)) + s0, smin, smax)

        elif lin == 2:
            s0, s_at_Imax, powerscale, smin, smax = params
            s = np.clip((smax - s0) * (intensity / s_at_Imax) ** powerscale + s0, smin, smax)

        return s

    def onNlut(self, _):
        """ check if n<=7"""
        nlut = int(self.nlut.GetValue())
        if nlut > 7:
            dlg = wx.MessageDialog(self, "nlut must be reasonnably less or equal to 7",
                                                                    "warning",
                                                                    wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()

    def onClick(self, evt):
        """ onclick with mouse. Behavior depends on toggle button state
        """

        if evt.inaxes:
            #            print("inaxes", evt)
            print(("inaxes x,y", evt.x, evt.y))
            print(("inaxes  xdata, ydata", evt.xdata, evt.ydata))
            self.centerx, self.centery = evt.xdata, evt.ydata

            if self.pointButton6.GetValue():
                self.Annotate_exp(evt)

            if self.pickdistbtn.GetValue():
                self.nbclick_dist += 1
                print("self.nbclick_dist", self.nbclick_dist)

                if self.nbclick_dist == 1:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(evt)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = evt.xdata, evt.ydata
                    else:
                        x, y = evt.xdata, evt.ydata

                    self.twopoints = [(x, y)]

                if self.nbclick_dist == 2:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(evt)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = evt.xdata, evt.ydata
                    else:
                        x, y = evt.xdata, evt.ydata

                    self.twopoints.append((x, y))

                    spot1 = self.twopoints[0]
                    spot2 = self.twopoints[1]

                    if self.datatype == "2thetachi":
                        _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))

                    if self.datatype == "gnomon":
                        tw, ch = IIM.Fromgnomon_to_2thetachi(
                            [np.array([spot1[0], spot2[0]]),
                                np.array([spot1[1], spot2[1]]),
                            ], 0)[:2]
                        _dist = GT.distfrom2thetachi(
                            np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]]))

                    if self.datatype == "pixels":

                        detectorparameters = self.indexation_parameters["detectorparameters"]

                        print("LaueToolsframe.defaultParam", detectorparameters)
                        tw, ch = F2TC.calc_uflab(np.array([spot1[0], spot2[0]]),
                                                    np.array([spot1[1], spot2[1]]),
                                                    detectorparameters,
                                                    pixelsize=self.pixelsize,
                                                    kf_direction=self.kf_direction)
                        print('tw, ch', tw, ch)

                        _dist = GT.distfrom2thetachi(np.array([tw[0], ch[0]]),
                                                    np.array([tw[1], ch[1]]))
                    # TODO: add if self.datatype == 'pixels':

                    print("angular distance (q1,q2):  %.3f deg " % _dist)

                    self.nbclick_dist = 0
                    # self.twopoints = []
                    self.pickdistbtn.SetValue(False)
                    self.pickdistbtn.SetBackgroundColour(self.defaultColor)

                    print("RES =", _dist)
                    sentence = "Corresponding lattice planes angular distance"
                    sentence += "\n between two scattered directions : %.2f " % _dist
                    self.txtctrldistance.SetLabel(
                        "==> %s deg" % str(np.round(_dist, decimals=3)))

            if self.recongnisebtn.GetValue():
                self.nbclick_dist += 1
                print("self.nbclick_dist", self.nbclick_dist)

                if self.nbclick_dist == 1:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(evt)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = evt.xdata, evt.ydata
                    else:
                        x, y = evt.xdata, evt.ydata

                    self.twospots = [(x, y)]

                if self.nbclick_dist == 2:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(evt)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = evt.xdata, evt.ydata
                    else:
                        x, y = evt.xdata, evt.ydata

                    self.twospots.append((x, y))

                    self.Reckon_2pts_new(evt)
        else:
            print("outside!! axes object")

    def EnterCombokeymaterial(self, evt):
        """
        in ManualIndexFrame
        """
        item = evt.GetSelection()
        self.key_material = self.list_materials[item]

        self.sb.SetStatusText("Selected material: %s" % str(self.parent.dict_Materials[self.key_material]))
        evt.Skip()

    def store_pts(self, evt):
        """ add mouse clicked point to self.points list"""
        self.points.append((evt.xdata, evt.ydata))
        print("# selected points", self.nbclick)
        print("Coordinates(%.3f,%.3f)" % (evt.xdata, evt.ydata))
        self.nbclick += 1

    def drawAnnote_exp(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index
        #ManualIndexFrame
        """
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
            t1 = axis.text(x + 1, y + 1, "%d" % (annote[0]), size=8)
            t2 = axis.text(x + 1, y - 1, "%.1f" % (annote[1]), size=8)
            if matplotlibversion <= "0.99.1":
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False)
            else:
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None")  # matplotlib 0.99.1.1
            self.drawnAnnotations_exp[(x, y)] = (t1, t2, m)
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()

    def drawSpecificAnnote_exp(self, annote):
        """
        ManualIndexFrame
        """
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_exp if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_exp(self.axes, x, y, a)

    def Annotate_exp(self, evt):
        """
        ManualIndexFrame
        """
        xtol = 20
        ytol = 20

        if self.datatype == "pixels":
            xtol = 200
            ytol = 200
        # """
        # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
        # self.Data_index_expspot = np.arange(len(self.Data_X))
        # """
        #         xdata, ydata, annotes = (self.Data_X,
        #                                  self.Data_Y,
        #                                  zip(self.Data_index_expspot, self.Data_I))

        xdata, ydata, annotes = (self.Data_X, self.Data_Y,
                                        list(zip(self.selectedAbsoluteSpotIndices_init, self.Data_I)))
        # print self.Idat
        # print self.Mdat
        # print annotes
        self._dataANNOTE_exp = list(zip(xdata, ydata, annotes))

        clickX = evt.xdata
        clickY = evt.ydata

        print("in Annotate_exp", clickX, clickY)

        annotes = []
        for x, y, a in self._dataANNOTE_exp:
            if (clickX - xtol < x < clickX + xtol) and (
                clickY - ytol < y < clickY + ytol):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes == []:
            return None

        annotes.sort()
        _distance, x, y, annote = annotes[0]
        print("the nearest experimental point is at(%.2f,%.2f)" % (x, y))
        print("with index %d and intensity %.1f" % (annote[0], annote[1]))
        self.clicked_indexSpot.append(annote[0])
        self.drawAnnote_exp(self.axes, x, y, annote)
        for l in self.links_exp:
            l.drawSpecificAnnote_exp(annote)

        self.updateStatusBar(x, y, annote)

        return x, y, annote[0], annote[1]

    def updateStatusBar(self, x, y, annote, spottype="exp"):
        """ display in status bar spot coordinates and other info"""
        if self.datatype == "2thetachi":
            Xplot = "2theta"
            Yplot = "chi"
        else:
            Xplot = "x"
            Yplot = "y"

        if spottype == "exp":
            self.sb.SetStatusText(("%s= %.2f " % (Xplot, x) + " %s= %.2f " % (Yplot, y) +
                                    "   Spotindex=%d " % annote[0] +
                                    "   Intensity=%.2f" % annote[1]), 0)

    def locateSpot(self, _):
        """ add rectangle at spot given by its index (read from self.findspotchck)"""
        if self.findspotchck.GetValue():
            spotindex = int(self.spotindexspinctrl.GetValue())
            print("Showing spot #: %d" % spotindex)
            X = self.Data_X[spotindex]
            Y = self.Data_Y[spotindex]
            Intens = self.Data_I[spotindex]

            if len(self.axes.patches) > 0:
                self.RemoveLastRectangle()
            self.addPatchRectangle(X, Y, size=self.locatespotmarkersize)

            self.updateStatusBar(X, Y, (spotindex, Intens), spottype="exp")

            self.canvas.draw()

    def RemoveLastRectangle(self):
        """ remove last added rectangle """
        if isinstance(self.axes.patches[-1], Rectangle):
            del self.axes.patches[-1]

    def addPatchRectangle(self, X, Y, size=50):
        """ add rectangle at X,Y in self.axes """
        hsize = size / 2.0
        self.axes.add_patch(Rectangle((X - hsize, Y - hsize), size, size, fill=False, color="r"))

    def close(self, _):
        """ close """
        self.Close(True)

    def all_reds(self):
        """ set all btns to red """
        for butt in self.listbuttons:
            butt.SetBackgroundColour(self.defaultColor)

    def what_was_pressed(self, flag):
        """ print btn pressed """
        if flag:
            print("-------------------")
            print([butt.GetValue() for butt in self.listbuttons])
            print("self.listbuttonstate", self.listbuttonstate)

    def T2(self, _):  # pick distance  see onClick
        """ read toggle button for pick distance"""
        self.what_was_pressed(0)
        if self.listbuttonstate[0] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pickdistbtn.SetValue(True)
            self.pickdistbtn.SetBackgroundColour("Green")
            self.listbuttonstate = [1, 0, 0]
            self.nbclick_dist = 0
        else:
            self.pickdistbtn.SetBackgroundColour(self.defaultColor)
            self.pickdistbtn.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

    def T3(self, evt):  # Recognise distance  see onClick
        """ read toggle button for Recognise distance"""
        self.what_was_pressed(0)
        if self.listbuttonstate[1] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.recongnisebtn.SetValue(True)
            self.recongnisebtn.SetBackgroundColour("Green")
            self.listbuttonstate = [0, 1, 0]
            self.nbclick_dist = 0
        else:
            self.recongnisebtn.SetBackgroundColour(self.defaultColor)
            self.recongnisebtn.SetValue(False)
            self.listbuttonstate = [0, 0, 0]
            evt.Skip()

    def T6(self, evt): # show exp spot props see onClick
        """ read toggle button for show exp spot props"""
        self.what_was_pressed(0)
        if self.listbuttonstate[2] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pointButton6.SetValue(True)
            self.pointButton6.SetBackgroundColour("Green")
            self.listbuttonstate = [0, 0, 1]
        else:
            self.pointButton6.SetBackgroundColour(self.defaultColor)
            self.pointButton6.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

            evt.Skip()

    def allbuttons_off(self):
        """ set all btns to False state """
        for butt in self.listbuttons:
            butt.SetValue(False)

    def readlogicalbuttons(self):
        """ return list of btns states """
        return [butt.GetValue() for butt in self.listbuttons]

    def Reckon_2pts_new(self, _):
        """ Start indexation from picked spots

        Index Laue Pattern by Recognising distance from two user clicked spots
        First press button 'recognise distance' then click on two spots

        in ManualIndexFrame

        Activate the recognition of the angle between two atomic planes in a LUT of angular distances:
        Need the selection of 2 spots, compute the interplanar angle, find all possibilities in the LUT
        Open the recognitionBox with possible solutions with indexing quality based on matching rate
        """
        # for LUT
        # MaxRadiusHKL = True
        MaxRadiusHKL = False

        if self.twospots is not None:  # two spots are selected
            self.recongnisebtn.SetValue(False)
            self.recongnisebtn.SetBackgroundColour(self.defaultColor)
            print("twospots", self.twospots)
            spot1 = self.twospots[0]
            spot2 = self.twospots[1]
            print("---Selected points")

            if self.datatype == "2thetachi":
                _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))
                last_index = self.clicked_indexSpot[-1]
                print("last clicked", last_index)
                if len(self.clicked_indexSpot) > 1:
                    last_last_index = self.clicked_indexSpot[-2]
                    print("former clicked", last_last_index)
                print("(2theta, chi) ")

            elif self.datatype == "gnomon":
                tw, ch = IIM.Fromgnomon_to_2thetachi([np.array([spot1[0], spot2[0]]),
                                                        np.array([spot1[1], spot2[1]])],
                                                        0)[:2]
                print("gnomon")
                last_index = self.clicked_indexSpot[-1]
                print("last clicked", last_index)
                if len(self.clicked_indexSpot) > 1:
                    last_last_index = self.clicked_indexSpot[-2]
                    print("former clicked", last_last_index)

                spot1 = [tw[0], ch[0]]
                spot2 = [tw[1], ch[1]]
                _dist = GT.distfrom2thetachi(np.array([tw[0], ch[0]]),
                                                    np.array([tw[1], ch[1]]))

            elif self.datatype == "pixels":
                print("pixels")
                last_index = self.clicked_indexSpot[-1]
                print("last clicked", last_index)
                if len(self.clicked_indexSpot) > 1:
                    last_last_index = self.clicked_indexSpot[-2]
                    print("former clicked", last_last_index)

                detectorparameters = self.indexation_parameters["detectorparameters"]
                print("LaueToolsframe.defaultParam", detectorparameters)
                tw, ch = F2TC.calc_uflab(np.array([spot1[0], spot2[0]]),
                                                    np.array([spot1[1], spot2[1]]),
                                                    detectorparameters,
                                                    pixelsize=self.pixelsize,
                                                    kf_direction=self.kf_direction)
                spot1 = [tw[0], ch[0]]
                spot2 = [tw[1], ch[1]]
                _dist = GT.distfrom2thetachi(np.array([tw[0], ch[0]]),
                                            np.array([tw[1], ch[1]]))

            print("spot1 [%.3f,%.3f]" % (tuple(spot1)))
            print("spot2 [%.3f,%.3f]" % (tuple(spot2)))
            print(
                "Angular distance between corresponding reflections for recognition :  %.3f deg "
                % _dist)

            spot1_ind = last_last_index
            spot2_ind = last_index

            print("spot1 index, spot2 index", spot1_ind, spot2_ind)

        if (spot1_ind not in self.selectedAbsoluteSpotIndices
            or spot2_ind not in self.selectedAbsoluteSpotIndices):
            wx.MessageBox("You must select two spots displayed in the current plot", "info")
            return

        if self.COCD.GetValue():
            self.onlyclosest = 1
        else:
            self.onlyclosest = 0

        spot_index_central = spot1_ind
        # nbmax_probed = spot2_ind + 1

        # if first clicked spot index is larger than second spot index
        # then try to recognise all distances between spot1 and [0, spot1+1]
        if spot1_ind > spot2_ind:
            # nbmax_probed = spot1_ind + 1
            pass

        energy_max = float(self.SCEmax.GetValue())

        ResolutionAngstrom = None
        nb_exp_spots_data = len(self.data[0])

        print("ResolutionAngstrom in Reckon_2pts_new() indexation", ResolutionAngstrom)
        print("nb of spots in Laue Pattern ", nb_exp_spots_data)

        #         # there is no precomputed angular distances between spots
        #         if not self.parent.ClassicalIndexation_Tabledist:
        #             print "main frame ClassicalIndexation_Tabledist is None. Calculate it!"
        #             # select 1rstly spots that have not been indexed and 2ndly reduced list by user
        # #             index_to_select = np.take(self.parent.current_exp_spot_index_list,
        # #                                       np.arange(nb_exp_spots_data))
        #             print "\n\n\n*****\n"
        #             print "self.parent.current_exp_spot_index_list",self.parent.current_exp_spot_index_list
        #             print "np.arange(nb_exp_spots_data)",np.arange(nb_exp_spots_data)
        #             print "nb_exp_spots_data",nb_exp_spots_data
        #             print "self.abs_spotindex",self.abs_spotindex
        # #                                       np.arange(nb_exp_spots_data))
        # #                                       np.arange(nb_exp_spots_data))"
        #             index_to_select = np.take(self.parent.current_exp_spot_index_list,
        #                                       self.abs_spotindex)
        #
        #             self.select_theta = self.parent.data_theta[index_to_select]
        #             self.select_chi = self.parent.data_chi[index_to_select]
        #             self.select_I = self.parent.data_I[index_to_select]
        #             print "len(self.select_theta)",len(self.select_theta)
        #             # print select_chi
        #             listcouple = np.array([self.select_theta, self.select_chi]).T
        #             # compute angles between spots
        #             Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        # there is no precomputed angular distances between spots
        if not self.parent.ClassicalIndexation_Tabledist:

            # Selection of spots among the whole data
            #             # MSSS number
            #             MatchingSpotSetSize = int(self.nbspotmaxformatching.GetValue())
            #
            #             # select 1rstly spots that have not been indexed and 2ndly reduced list by user
            #             index_to_select = np.take(self.current_exp_spot_index_list,
            #                                       np.arange(MatchingSpotSetSize))

            index_to_select = self.selectedAbsoluteSpotIndices

            self.select_theta = self.indexation_parameters["AllDataToIndex"]["data_theta"
            ][index_to_select]
            self.select_chi = self.indexation_parameters["AllDataToIndex"]["data_chi"][index_to_select]
            self.select_I = self.indexation_parameters["AllDataToIndex"]["data_I"][index_to_select]

            # print("index_to_select", index_to_select)

            self.select_dataX = self.indexation_parameters["AllDataToIndex"]["data_pixX"][index_to_select]
            self.select_dataY = self.indexation_parameters["AllDataToIndex"]["data_pixY"][index_to_select]
            # print select_theta
            # print select_chi

            # listcouple = np.array([self.select_theta, self.select_chi]).T
            # compute angles between spots
            # Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        else:
            print("Reuse computed ClassicalIndexation_Tabledist with size: %d"
                                    % len(self.parent.ClassicalIndexation_Tabledist))

        self.data = (2 * self.select_theta,
                    self.select_chi,
                    self.select_I,
                    self.parent.DataPlot_filename)

        self.key_material = str(self.combokeymaterial.GetValue())
        latticeparams = self.parent.dict_Materials[self.key_material][1]
        B = CP.calc_B_RR(latticeparams)
        # print type(key_material)
        # print type(nbmax_probed)
        # print type(energy_max)

        # read maximum index of hkl for building angles Look Up Table(LUT)
        nLUT = int(self.nlut.GetValue())

        rough_tolangle = float(self.DRTA.GetValue())
        fine_tolangle = float(self.matr_ctrl.GetValue())
        Nb_criterium = 3

        NBRP = 3

        # detector geometry
        detectorparameters = {}
        detectorparameters["kf_direction"] = self.parent.kf_direction
        detectorparameters["detectorparameters"] = self.parent.defaultParam
        detectorparameters["detectordiameter"] = self.parent.detectordiameter
        detectorparameters["pixelsize"] = self.parent.pixelsize
        detectorparameters["dim"] = self.parent.framedim

        print("detectorparameters", detectorparameters)

        restrictLUT_cubicSymmetry = True
        set_central_spots_hkl = None

        if self.sethklchck.GetValue():
            # could be advised for cubic symmetry to have positive H K L sorted by decreasing order
            strhkl = str(self.sethklcentral.GetValue())[1:-1].split(",")
            H, K, L = strhkl
            H, K, L = int(H), int(K), int(L)
            # LUT with cubic symmetry does not have negative L
            if L < 0:
                restrictLUT_cubicSymmetry = False

            set_central_spots_hkl = [[int(H), int(K), int(L)]]

        # restrict LUT if allowed and if crystal is cubic
        if restrictLUT_cubicSymmetry:
            restrictLUT_cubicSymmetry = CP.hasCubicSymmetry(self.key_material,
                                            dictmaterials=self.dict_Materials)

        print("set_central_spots_hkl", set_central_spots_hkl)
        print("restrictLUT_cubicSymmetry", restrictLUT_cubicSymmetry)

        verbosedetails = self.verbose.GetValue()

        self.set_central_spots_hkl = set_central_spots_hkl
        self.spot_index_central = spot_index_central
        self.Nb_criterium = Nb_criterium
        self.NBRP = NBRP
        self.B = B
        self.energy_max = energy_max
        self.ResolutionAngstrom = ResolutionAngstrom
        self.nLUT = nLUT
        self.rough_tolangle = rough_tolangle
        self.fine_tolangle = fine_tolangle

        USETHREAD = 1
        if USETHREAD:
            # with a thread 2----------------------------------------

            self.resindexation = None
            fctparams = [INDEX.getUBs_and_MatchingRate,
                    (spot1_ind, spot2_ind, rough_tolangle, _dist,
                    spot1, spot2, nLUT, B, 2 * self.select_theta, self.select_chi),
                {"set_hkl_1": set_central_spots_hkl,
                    "key_material": self.key_material,
                    "emax": energy_max,
                    "ResolutionAngstrom": ResolutionAngstrom,
                    "ang_tol_MR": fine_tolangle,
                    "detectorparameters": detectorparameters,
                    "LUT": None,
                    "MaxRadiusHKL": MaxRadiusHKL,
                    "verbose": 0,
                    "verbosedetails": verbosedetails,
                    "Minimum_Nb_Matches": Nb_criterium,
                    "worker": None,
                    "dictmaterials":self.dict_Materials}]

            # update DataSetObject
            self.DataSet.dim = detectorparameters["dim"]
            self.DataSet.pixelsize = detectorparameters["pixelsize"]
            self.DataSet.detectordiameter = detectorparameters["detectordiameter"]
            self.DataSet.kf_direction = detectorparameters["kf_direction"]
            self.DataSet.key_material = self.key_material
            self.DataSet.emin = 5
            self.DataSet.emax = energy_max

            print("self.DataSet.detectordiameter", self.DataSet.detectordiameter)
            print("self.DataSet.kf_direction", self.DataSet.kf_direction)
            print("self.DataSet.pixelsize", self.DataSet.pixelsize)

            self.TGframe = TG.ThreadHandlingFrame(self, -1, threadFunctionParams=fctparams,
                                                        parentAttributeName_Result="resindexation",
                                                        parentNextFunction=self.simulateAllResults)
            self.TGframe.OnStart(1)
            self.TGframe.Show(True)

            # will set self.UBs_MRs to the output of INDEX.getUBs_and_MatchingRate

        else:
            # case USETHREAD == 0
            # ---- indexation in Reckon_2pts_new() in Manualindexframe
            self.worker = None
            self.resindexation = INDEX.getUBs_and_MatchingRate(spot1_ind, spot2_ind, rough_tolangle,
                                                    _dist, spot1, spot2, nLUT, B,
                                                    2 * self.select_theta, self.select_chi,
                                                    set_hkl_1=set_central_spots_hkl,
                                                    key_material=self.key_material,
                                                    emax=energy_max,
                                                    ResolutionAngstrom=ResolutionAngstrom,
                                                    ang_tol_MR=fine_tolangle,
                                                    detectorparameters=detectorparameters,
                                                    LUT=None,
                                                    MaxRadiusHKL=MaxRadiusHKL,
                                                    verbose=0,
                                                    verbosedetails=verbosedetails,
                                                    Minimum_Nb_Matches=Nb_criterium,
                                                    worker=self.worker,
                                                    dictmaterials=self.dict_Materials)
            self.UBs_MRs, _ = self.resindexation
            # self.bestmat, stats_res = self.UBs_MRs
            self.bestmat, _ = self.UBs_MRs
            # update DataSetObject
            self.DataSet.dim = detectorparameters["dim"]
            self.DataSet.pixelsize = detectorparameters["pixelsize"]
            self.DataSet.detectordiameter = detectorparameters["detectordiameter"]
            self.DataSet.kf_direction = detectorparameters["kf_direction"]
            self.DataSet.key_material = self.key_material
            self.DataSet.emin = 5
            self.DataSet.emax = energy_max

            print("self.DataSet.detectordiameter", self.DataSet.detectordiameter)

            self.simulateAllResults()

    def simulateAllResults(self):
        """ simulate all potential UB matrix solution from indexing.

        Read self.resindexation
        """
        #print('self.resindexation',self.resindexation)
        print('len self.resindexation', len(self.resindexation))

        self.UBs_MRs, _ = self.resindexation

        if not self.UBs_MRs[0] or self.UBs_MRs is None:
            wx.MessageBox('Sorry Nothing found !!', 'INFO')
            return

        print("Entering simulateAllResults\n\n")
        print("self.UBs_MRs", self.UBs_MRs)

        self.bestmat, stats_res = self.UBs_MRs

        #         print "stats_res", stats_res
        nb_sol = len(self.bestmat)

        keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[self.key_material][1])

        if self.set_central_spots_hkl not in (None, [None]):
            keep_only_equivalent = False

        if nb_sol > 1:
            print("Merging matrices")
            print("keep_only_equivalent = %s" % keep_only_equivalent)
            self.bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(
                                                            self.bestmat,
                                                            stats_res,
                                                            self.Nb_criterium,
                                                            tol=0.0001,
                                                            keep_only_equivalent=keep_only_equivalent)

        print("stats_res", stats_res)
        nb_sol = len(self.bestmat)
        print("Max. Number of Solutions", self.NBRP)
        print("spot_index_central", self.spot_index_central)

        if nb_sol:
            print("%d matrice(s) found" % nb_sol)
            print(self.bestmat)
            print("Each Matrix is stored in 'MatIndex_#' for further simulation")
            for k in range(nb_sol):
                self.parent.dict_Rot["MatIndex_%d" % (k + 1)] = self.bestmat[k]

            stats_properformat = []
            for elem in stats_res:
                elem[0] = int(elem[0])
                elem[1] = int(elem[1])
                stats_properformat.append(tuple(elem))

            # one central point were used for distance recognition
            self.TwicethetaChi_solution = [0 for p in range(nb_sol)]
            paramsimul = []

            for p in range(nb_sol):

                orientmatrix = self.bestmat[p]

                # only orientmatrix, self.key_material are used ----------------------
                vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
                # old definition of grain
                grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]
                # ------------------------------------------------------------------

                #                 print "self.indexation_parameters.keys()", self.indexation_parameters.keys()
                #                 print "self.indexation_parameters.keys()", self.indexation_parameters.keys()
                #                 print "self.indexation_parameters['detectordiameter']", self.indexation_parameters['detectordiameter']
                TwicethetaChi = LAUE.SimulateResult(grain, 5, self.energy_max, self.indexation_parameters,
                                                        ResolutionAngstrom=self.ResolutionAngstrom,
                                                        fastcompute=1,
                                                        dictmaterials=self.dict_Materials)
                self.TwicethetaChi_solution[p] = TwicethetaChi
                paramsimul.append((grain, 5, self.energy_max))

            self.indexation_parameters["paramsimul"] = paramsimul
            self.indexation_parameters["bestmatrices"] = self.bestmat
            self.indexation_parameters["TwicethetaChi_solutions"] = self.TwicethetaChi_solution
            self.indexation_parameters["plot_xlim"] = self.xlim
            self.indexation_parameters["plot_ylim"] = self.ylim
            self.indexation_parameters["flipyaxis"] = self.data_dict["flipyaxis"]

            datatype = self.datatype
            # display "statistical" results
            if self.datatype == "gnomon":
                print("plot results in 2thetachi coordinates")
                datatype = "2thetachi"

            RRCBClassical = RecognitionResultCheckBox(self, -1, "Classical Indexation Solutions",
                                                        stats_properformat,
                                                        self.data,
                                                        self.rough_tolangle,
                                                        self.fine_tolangle,
                                                        key_material=self.key_material,
                                                        emax=self.energy_max,
                                                        ResolutionAngstrom=self.ResolutionAngstrom,
                                                        kf_direction=self.kf_direction,
                                                        datatype=datatype,
                                                        data_2thetachi=self.data_2thetachi,
                                                        data_XY=self.data_XY,
                                                        ImageArray=self.ImageArray,
                                                        CCDdetectorparameters=self.indexation_parameters,
                                                        IndexationParameters=self.indexation_parameters,
                                                        StorageDict=self.StorageDict,
                                                        DataSetObject=self.DataSet)

            RRCBClassical.Show(True)

        else:  # any matrix was found
            print("!!  Nothing found   !!!")
            print("with LaueToolsGUI.Reckon_2pts_new()")
            # MessageBox will freeze the computer
        # wx.MessageBox('! NOTHING FOUND !\nTry to reduce the Minimum Number Matched Spots to catch something!', 'INFO')

        self.nbclick_dist = 0
        self.recongnisebtn.SetValue(False)
