# -*- coding: utf-8 -*-
"""
Class to replot backuped 2D map

IN DEV...

this class belongs to the open-source LaueTools project
JS micha March 2025
"""

import os
import sys
import time
import copy
import pickle


import numpy as np
import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

from pylab import cm as pcm
from pylab import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle as PatchRectangle

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                NavigationToolbar2WxAgg as NavigationToolbar)

from LaueTools.GUI.mosaic import MyCustomToolbar 
import LaueTools.generaltools as GT
import LaueTools.IOimagefile as IOimage
from .. import MessageCommand as MC

import wx.lib.agw.customtreectrl as CT

from libtiff import TIFF #, libtiff_ctypes


class TreePanel(wx.Panel):
    """ class of tree organisation of map

    granparent class must provide  ReadScan_SpecFile()

    sets granparent scan_index_mesh  or scan_index_ascan to selected item index
    """
    def __init__(self, parent, scantype=None, _id=wx.ID_ANY, **kwd):
        wx.Panel.__init__(self, parent=parent, id=_id, **kwd)

        self.parent = parent
        self.scantype = scantype
        self.frameparent = self.parent.GetParent()
        # self.tree = wx.TreeCtrl(self, -1, wx.DefaultPosition, (-1, -1),
        #                                                     wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS)
        # agwStyle=wx.TR_DEFAULT_STYLE
        self.tree = CT.CustomTreeCtrl(self, -1, agwStyle=wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS | wx.TR_MULTIPLE, size=(200,-1))
        self.tree.DoGetBestSize()

        self.maketree()

        # multiple selection ------
        self.keypressed = None
        self.multiitems = False
        # --------------------

        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelChanged)
        #self.tree.Bind(wx.EVT_TREE_SEL_CHANGING, self.OnSelChanged)
        self.tree.Bind(wx.EVT_TREE_KEY_DOWN, self.OnkeyPressed)

        # wx.EVT_TREE_ITEM_RIGHT_CLICK
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.tree, 1, wx.EXPAND)
        vbox.AddSpacer(1)

        self.SetSizer(vbox)

    def maketree(self):
        self.root = self.tree.AddRoot("Map Files")
        self.tree.AppendItem(self.root, str(self.scantype))

    def OnkeyPressed(self, event):
        #print(dir(event))
        key = event.GetKeyCode()
        # print('key pressed is ', key)
        
        if key == wx.WXK_DOWN:
            # arrow down
            nextitem = self.tree.GetNext(self.last_item)
            try:
                self.tree.DoSelectItem(nextitem)
                self.OnSelChanged(event, scan_index=nextitem)
            except AttributeError: # reaching last element!
                pass

        elif key == wx.WXK_UP:
            # arrow up

            previtem = self.tree.GetPrev(self.last_item)
            self.tree.DoSelectItem(previtem)
            #self.tree.GetPrev(lastclickeditem)
            # lastclickeditem = self.tree.GetFocusedItem()
            # previtem = self.tree.GetPrevVisible(self.lastclickeditem)
            # self.tree.SetFocusedItem(previtem)

            self.OnSelChanged(1, scan_index=previtem)

    def OnSelChanged(self, event, scan_index=None):
        
        #self.frameparent.fig.clear() # to do for wxpython 4.1.1
        self.SelChangedFile(event, item=None)

    def SelChangedFile(self, event, item=None):
        """ read item for map file tree    """
        #print('\n\n SelChangeHdf5')
        if item is None:
            if event != 1:
                item = event.GetItem()
        
        if item is None:
            return
        selected_item = self.tree.GetItemText(item)
        #scan_index = int(selected_item)
        print("item selected: ", selected_item)
        #print("selected_item ", dir(item))
        
        # single selection
        
        print('Single selection MODE', self.frameparent.currentfolder)

        fullpath=os.path.join(self.frameparent.currentfolder, selected_item)
        
        self.frameparent.readData(fullpath)
        
        self.frameparent.scan_index_mesh = selected_item
        
        #self.last_sel_scan_index = selected_item
        self.last_item = item

        # # tooltip----------------
        # speccommand = self.frameparent.scancommand
        # date = self.frameparent.scan_date
        # # print('speccommand', speccommand)
        # # print('date',date)
        # tooltip = "command: %s\n date: %s" % (speccommand, date)
        # #print('tooltip',tooltip)
        # event.GetEventObject().SetToolTipString(tooltip)
        # event.Skip()
        # #------------------
    
class ShowMapFrame(wx.Frame):
    """
    Class to load and visualise 2D array scalar data generated by mosaic

    .. note:: not finished
    """
    def __init__(self, parent, _id, title, datarois, absolutecornerindices=None,
                                                    Imageindices=np.arange(2),
                                                    nb_row=10,
                                                    nb_lines=10,
                                                    boxsize_row=0,  # TODO row actually is column
                                                    boxsize_line=0,  # TODO line is row
                                                    stepindex=1,
                                                    imagename="",
                                                    mosaic=1,
                                                    dict_param=None,
                                                    datatype="Intensity",
                                                    dictrois=None,
                                                    maxpositions=None):
        """
        datatype = Intensity, PositionX, PositionY, RadialPosition
        """
        print("\n\n*****\nCREATING PLOT of 2D Map of Scalar data: %s\n****\n"%title)
        # dat=dat.reshape(((self.nb_row)*2*self.boxsize_row,(self.nb_lines)*2*self.boxsize_line))
        self.appFrame = wx.Frame.__init__(self, parent, _id, title, size=(900, 700))
        self.panel = wx.Panel(self, -1)
        
        dict_param = {}
        try:
            self.dict_ROI = parent.dict_ROI
        except AttributeError:
            self.dict_ROI = None
        self.parent = parent
        self.mosaic = mosaic

        self.createMenu()
        self.sb = self.CreateStatusBar()

        self.treemesh = TreePanel(self.panel, scantype="MESH", _id=0, size=(250,-1))

        self.currentfolder = None

        self.dpi = 100
        self.figsize = 5
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.fig.set_size_inches(self.figsize, self.figsize, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        self.axes = self.fig.add_subplot(111)
        #         self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar = MyCustomToolbar(self.canvas, self)
        self.canvas.mpl_connect("motion_notify_event", self.mouse_move)

        self.Bind(wx.EVT_PAINT, self.OnPaint)

        self.toolbar.Realize()

        self.tooltip = wx.ToolTip(tip="tip with a long %s line and a newline\n" % (" " * 100))
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)
        self.fig.canvas.mpl_connect("button_press_event", self.onClick)
        self.fig.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.dataraw = copy.copy(datarois)
        # data to be displayed
        self.datarois = datarois
        self.dataroiindex = 0
        self.data = self.datarois[self.dataroiindex]
        self.datatype = datatype
        self.maxpositions = maxpositions


        # test ---------  need datarois,  maxpoistions, index and threshold
        modifiedpositions=[]
        mythreshold = 500
        for roiidx in range(dictrois['roimaxindex']):
            radialdist = GT.to2Darray(np.sqrt(np.sum(maxpositions[:,roiidx]**2, axis=1)), nb_row)
            threshpos = radialdist*np.where(datarois[roiidx]>mythreshold,1,0)
            modifiedpositions.append(threshpos)
        self.datarois = np.array(modifiedpositions)
        self.dataroiindex = 0
        self.data = self.datarois[self.dataroiindex]
        #-----------------------------

        self.absolutecornerindices = absolutecornerindices
        self.title = title
        self.Imageindices = Imageindices  # 1D list of image indices
        self.nb_columns = nb_row
        self.nb_lines = nb_lines
        self.boxsize_row = boxsize_row
        self.boxsize_line = boxsize_line
        print('nb_row  nb_lines', nb_row, nb_lines)
        print('boxsize_row  boxsize_line',boxsize_row, boxsize_line)
        self.stepindex = stepindex
        self.imagename = imagename
        self.currentpointedImageIndex = None
        self.dictrois=dictrois

        self.dict_param = copy.copy(dict_param)

        self.dirname = None
        self.filename = None
        self.originYaxis = "lower"

        self.memorizedxlimits = None
        self.memorizedylimits = None

        print('----- In ImshowFrame -------')
        print("self.data.shape", self.data.shape)
        print("self.nb_columns, self.nb_lines", self.nb_columns, self.nb_lines)
        print("self.boxsize_row, self.boxsize_line", self.boxsize_row, self.boxsize_line)

        self.removebackground = False
        if self.datatype == "Intensity":
            self.backgroundlevel = self.getMeanLevel()
        elif self.datatype == "PositionX":
            self.backgroundlevel = self.dict_param["pixelX_center"]
        elif self.datatype == "PositionY":
            self.backgroundlevel = self.dict_param["pixelY_center"]
        else:
            self.backgroundlevel = 0

        # --- -----------widgets to change plot display

        if "datasigntype" in dict_param:
            self.datasigntype = dict_param["datasigntype"]
        else:
            self.datasigntype = "positive"

        if "palette" in dict_param:
            self.palette = dict_param["palette"]
        else:
            if self.datasigntype == "positive":
                self.palette = copy.copy(GT.ORRD)
                self.LastLUT = "OrRd"
            else:
                self.palette = copy.copy(GT.SEISMIC)
                self.LastLUT = "seismic"

        self.palette.set_bad(color="black")
        self.colorbar = None
        self.LastLUT = self.palette
        self.plotgrid = False

        self.IminDisplayed = 0
        self.ImaxDisplayed = 100
        #         if self.datatype == 'scalar':
        self.slidertxt_min = wx.StaticText(self.panel, -1, "Min :")
        self.slider_min = wx.Slider(self.panel, -1, size=(200, 50),
                                    value=self.IminDisplayed,
                                    minValue=0,
                                    maxValue=99,
                                    style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_min.SetTickFreq(50)
        else:
            self.slider_min.SetTickFreq(50, 1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMin, self.slider_min)

        self.slidertxt_max = wx.StaticText(self.panel, -1, "Max :")
        self.slider_max = wx.Slider(self.panel, -1, size=(200, 50),
                                    value=self.ImaxDisplayed,
                                    minValue=1,
                                    maxValue=100,
                                    style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_max.SetTickFreq(50)
        else:
            self.slider_max.SetTickFreq(50, 1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMax, self.slider_max)

        self.vmintxtctrl = wx.TextCtrl(self.panel, -1, str(np.amin(self.data)),
                                                                        style=wx.TE_PROCESS_ENTER)
        self.vmaxtxtctrl = wx.TextCtrl(self.panel, -1, str(np.amax(self.data)),
                                                                        style=wx.TE_PROCESS_ENTER)
        self.vmintxtctrl.Bind(wx.EVT_TEXT_ENTER, self.OnChangeVmin)
        self.vmaxtxtctrl.Bind(wx.EVT_TEXT_ENTER, self.OnChangeVmax)

        # loading LUTS
        self.mapsLUT = [m for m in pcm.datad if not m.endswith("_r")]
        self.mapsLUT.sort()

        self.luttxt = wx.StaticText(self.panel, -1, "LUT")
        self.comboLUT = wx.ComboBox(self.panel,
                                    -1,
                                    str(self.LastLUT),
                                    choices=self.mapsLUT,
                                    style=wx.TE_PROCESS_ENTER)

        self.comboLUT.Bind(wx.EVT_COMBOBOX, self.OnChangeLUT)
        self.comboLUT.Bind(wx.EVT_TEXT_ENTER, self.OnTypeLUT)

        if self.datatype == "Intensity":
            self.scaletype = "Linear" #self.scaletype = "Log"

        else:
            self.scaletype = "Linear"
        self.scaletxt = wx.StaticText(self.panel, -1, "Scale")
        self.comboscale = wx.ComboBox(self.panel, -1, self.scaletype, choices=["Linear", "Log"])

        self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)

        self.aspect = "auto"
        self.aspecttxt = wx.StaticText(self.panel, -1, "Aspect Ratio")
        self.comboaspect = wx.ComboBox(self.panel, -1, self.aspect,
                                        choices=["equal", "auto"],
                                        style=wx.TE_PROCESS_ENTER)

        self.comboaspect.Bind(wx.EVT_COMBOBOX, self.OnChangeAspect)
        self.comboaspect.Bind(wx.EVT_TEXT_ENTER, self.OnChangeAspect)

        self.chckgrid = wx.CheckBox(self.panel, -1, "Grid")
        self.chckgrid.SetValue(self.plotgrid)
        self.chckgrid.Bind(wx.EVT_CHECKBOX, self.Oncheckgrid)

        self.txtslidertoiindex = wx.StaticText(self.panel, -1, "Roi index :")
        self.slidertoiindex = wx.Slider(self.panel, -1, size=(200, 50),
                                    value=0,
                                    minValue=0,
                                    maxValue=self.dictrois['roimaxindex'],
                                    style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        #self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.onchangeroiindex, self.slidertoiindex)
        self.Bind(wx.EVT_SLIDER, self.onchangeroiindex, self.slidertoiindex)
        #self.Bind(wx.EVT_COMMAND_SCROLL_LINEDOWN, self.onchangeroiindex, self.slidertoiindex)

        if self.datatype in ("Intensity", "PositionX", "PositionY"):
            self.bkgchckbox = wx.CheckBox(self.panel, -1, "Substract Background")
            self.bkgchckbox.SetValue(self.removebackground)
            self.bkgchckbox.Bind(wx.EVT_CHECKBOX, self.OnSubstractBackground)

            self.bkgsettxt = wx.StaticText(self.panel, -1, "Level ")
            self.bkgsetctrl = wx.TextCtrl(self.panel, -1, "%.2f" % self.backgroundlevel)

        if self.datatype in ("RadialPosition", "Vector"):
            self.centerbtn = wx.Button(self.panel, -1, "Change Center (X,Y) ")
            self.centerbtn.Bind(wx.EVT_BUTTON, self.OnChangeCenter)

            self.Xcenterctrl = wx.TextCtrl(self.panel, -1, "%.1f" % self.dict_param["pixelX_center"])
            self.Ycenterctrl = wx.TextCtrl(self.panel, -1, "%.1f" % self.dict_param["pixelY_center"])

        if self.datatype in ("Vector",):
            self.arrowWidthDisplayed = 50
            self.arrowwidth = self.convertArrowWidth(self.arrowWidthDisplayed)
            self.slidertxt_arrowwidth = wx.StaticText(self.panel, -1, "Arrow width :")
            self.slider_arrowwidth = wx.Slider(self.panel,
                                                -1,
                                                size=(200, 50),
                                                value=self.arrowWidthDisplayed,
                                                minValue=1,
                                                maxValue=100,
                                                style=wx.SL_AUTOTICKS | wx.SL_LABELS)
            if WXPYTHON4:
                self.slider_arrowwidth.SetTickFreq(50)
            else:
                self.slider_arrowwidth.SetTickFreq(50, 1)
            self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderArrowSize, self.slider_arrowwidth)

            self.arrowScaleDisplayed = 50
            self.arrowscale = self.convertArrowScale(self.arrowScaleDisplayed)
            self.slidertxt_arrowscale = wx.StaticText(self.panel, -1, "Arrow scale :")
            self.slider_arrowscale = wx.Slider(self.panel,
                                                -1,
                                                size=(200, 50),
                                                value=self.arrowWidthDisplayed,
                                                minValue=1,
                                                maxValue=100,
                                                style=wx.SL_AUTOTICKS | wx.SL_LABELS)
            if WXPYTHON4:
                self.slider_arrowscale.SetTickFreq(50)
            else:
                self.slider_arrowscale.SetTickFreq(50, 1)
            self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderArrowScale, self.slider_arrowscale)

            #             self.arrowScaleDisplayedCutOff=50
            #             self.arrowscaleCutOff = self.convertArrowScale(self.arrowScaleDisplayedCutOff)
            #             self.slidertxt_arrowscalecutoff = wx.StaticText(self.panel, -1,
            #                 "Arrow scale high cutoff:")
            #             self.slider_arrowscaleCutOff = wx.Slider(self.panel, -1, size=(200, 50),
            #                 value=self.arrowWidthDisplayed,
            #                 minValue=1,
            #                 maxValue=100,
            #                 style=wx.SL_AUTOTICKS | wx.SL_LABELS)
            #             self.slider_arrowscaleCutOff.SetTickFreq(50, 1)
            #             self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderArrowScaleCutOff, self.slider_arrowscaleCutOff)

            lengththreshold = 10
            self.masklongvectortxt = wx.StaticText(self.panel, -1, "Vector Pix. length <")
            self.masklongvectorctrl = wx.TextCtrl(self.panel, -1, "%.1f" % lengththreshold)

        if "FilteredfittedPeaksData" in self.dict_param:
            # self.dict_param['FilteredfittedPeaksData']
            maskthreshold = 25

            self.ApplyMaskOnPeaksProperties(maskthreshold)

            self.maskbtn = wx.Button(self.panel, -1, "Mask weak Peaks")
            self.maskbtn.Bind(wx.EVT_BUTTON, self.OnMask)
            self.maskthresholdtxt = wx.StaticText(self.panel, -1, "Peak Amplitude <")
            self.maskthresholdctrl = wx.TextCtrl(self.panel, -1, "%d" % maskthreshold)

        self.layout()
        self.setArrayImageIndices()

        print('self.datatype', self.datatype)
        print('detailed data')
        print('self.data.shape',self.data.shape)
        print('self.Imageindices',self.Imageindices)
        print('self.tabindices',self.tabindices)
        self._replot()

    def layout(self):

        h0box = wx.BoxSizer(wx.HORIZONTAL)
        h0box.Add(self.toolbar, 0)
        h0box.Add(self.luttxt, 0)
        h0box.Add(self.comboLUT, 0)
        h0box.Add(self.aspecttxt, 0)
        h0box.Add(self.comboaspect, 0)
        h0box.Add(self.scaletxt, 0)
        h0box.Add(self.comboscale, 0)
        h0box.Add(self.chckgrid, 0)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.slidertxt_min, 0)
        hbox.Add(self.vmintxtctrl, 0)
        hbox.Add(self.slider_min, 0)
        hbox.Add(self.slidertxt_max, 0)
        hbox.Add(self.slider_max, 0)
        hbox.Add(self.vmaxtxtctrl, 0)

        hroi = wx.BoxSizer(wx.HORIZONTAL)
        hroi.Add(self.txtslidertoiindex, 0)
        hroi.Add(self.slidertoiindex, 0)
 
        if self.datatype in ("Intensity", "PositionX", "PositionY"):
            hbox.Add(self.bkgchckbox, 0)
            hbox.Add(self.bkgsettxt, 0)
            hbox.Add(self.bkgsetctrl, 0)
        if self.datatype in ("RadialPosition", "Vector"):
            hbox.Add(self.centerbtn, 0)
            hbox.Add(self.Xcenterctrl, 0)
            hbox.Add(self.Ycenterctrl, 0)

        if "FilteredfittedPeaksData" in self.dict_param:
            hboxmask = wx.BoxSizer(wx.HORIZONTAL)
            hboxmask.Add(self.maskbtn, 0)
            hboxmask.Add(self.maskthresholdtxt, 0, wx.ALL,5)

            hboxmask.Add(self.maskthresholdctrl, 0, wx.ALL,5)

        if self.datatype in ("Vector",):
            hboxarrow = wx.BoxSizer(wx.HORIZONTAL)
            hboxarrow.Add(self.slidertxt_arrowwidth, 0)
            hboxarrow.Add(self.slider_arrowwidth, 0)

            hboxarrow.Add(self.slidertxt_arrowscale, 0, wx.ALL,5)
            hboxarrow.Add(self.slider_arrowscale, 0, wx.ALL,5)

            #             hboxarrow.Add(self.slidertxt_arrowscalecutoff, 0)
            #             hboxarrow.Add(self.slider_arrowscaleCutOff, 0)

            if not hboxmask:
                hboxmask = wx.BoxSizer(wx.HORIZONTAL)
            hboxmask.Add(self.masklongvectortxt, 0)
            hboxmask.Add(self.masklongvectorctrl, 0)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        #         self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        #         self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        #         self.vbox.Add(self.btnflipud, 0, wx.EXPAND)
        self.vbox.Add(self.canvas, 1, wx.ALL | wx.TOP | wx.EXPAND)
        self.vbox.Add(h0box, 0, wx.EXPAND)
        self.vbox.Add(hbox, 0, wx.EXPAND)
        self.vbox.Add(hroi, 0, wx.EXPAND)
        if self.datatype in ("Vector",):
            self.vbox.Add(hboxarrow, 0, wx.EXPAND)
        if "FilteredfittedPeaksData" in self.dict_param:
            self.vbox.Add(hboxmask, 0, wx.EXPAND)
        
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.treemesh, 1, wx.LEFT | wx.TOP | wx.GROW)#wx.EXPAND)
        self.hbox.Add(self.vbox, 0, wx.LEFT | wx.TOP | wx.GROW)
        self.panel.SetSizer(self.hbox)
        self.hbox.Fit(self)

        self.Layout()

    def createMenu(self):
        """
        Set up the MenuBar
        """
        MenuBar = wx.MenuBar()

        MenuBar.Attach(self.appFrame)

        FileMenu = wx.Menu()

        saveimagemenu = FileMenu.Append(wx.ID_ANY, "&Save", "Save Image")
        self.Bind(wx.EVT_MENU, self.OnSave, saveimagemenu)

        SaveDatamenu = FileMenu.Append(wx.ID_ANY, "&Save Data", "Save Data")
        self.Bind(wx.EVT_MENU, self.SaveData, SaveDatamenu)

        openDatamenu = FileMenu.Append(wx.ID_ANY, "&Open Data", "Open Data")
        self.Bind(wx.EVT_MENU, self.OpenData, openDatamenu)

        menuFolder = FileMenu.Append(wx.ID_ANY, "Open folder", "Open a folder containing some map files (lauetools mosaic's made)")
        self.Bind(wx.EVT_MENU, self.OpenFolder, menuFolder)

        CloseMenu = FileMenu.Append(wx.ID_ANY, "&Close", "Close Application")
        self.Bind(wx.EVT_MENU, self.OnQuit, CloseMenu)

        MenuBar.Append(FileMenu, "&File")

        roi_menu = wx.Menu()
        SaveRoi = roi_menu.Append(1111, "&Capture && Save ROI",
            "Capture current rectangular image indices ROI and save")
        self.Bind(wx.EVT_MENU, self.SaveRectROI, SaveRoi)

        SaveAsRoi = roi_menu.Append(1112, "Capture && &Save ROI As",
            "Capture current rectangular image indices ROI and save with user defined name")
        self.Bind(wx.EVT_MENU, self.SaveAsRectROI, SaveAsRoi)

        LoadROI = roi_menu.Append(1113, "&Load ROI", "Load ROI from file")
        self.Bind(wx.EVT_MENU, self.LoadROI, LoadROI)

        EditRoi = roi_menu.Append(1114, "&Edit ROI", "Edit ROI parameters")
        self.Bind(wx.EVT_MENU, self.EditROI, EditRoi)
        MenuBar.Append(roi_menu, "&ROI")

        help_menu = wx.Menu()
        AboutMenu = help_menu.Append(wx.ID_ANY, "&About", "More information About this program")
        self.Bind(wx.EVT_MENU, self.OnAbout, AboutMenu)
        MenuBar.Append(help_menu, "&Help")

        # TODO: sample map ROI selection is ONLY implemented from mosaic plot, not from 2D scalar plot
        if not self.mosaic:
            roi_menu.Enable(id=1111, enable=False)
            roi_menu.Enable(id=1112, enable=False)
            roi_menu.Enable(id=1113, enable=False)
            roi_menu.Enable(id=1114, enable=False)
        #             MenuBar.EnableTop(1, False)

        self.SetMenuBar(MenuBar)

    def OnAbout(self, event):
        description = """mosaic aims mainly at looking at 2D data images and navigating among a set of them (sample map); Mapcanvas is a part of LaueTools toolkit for white beam x-ray microdiffraction Laue Pattern analysis. It allows Simulation & Indexation procedures written in python by Jean-Sebastien MICHA \n  micha@esrf.fr\n\n French CRG-IF beamline \n at BM32 (European Synchrotron Radiation Facility).

        Support and help in developing this package:
        https://gitlab.esrf.fr/micha/lauetools/
        """

        licence = """mosaic is free software; you can redistribute it and/or modify it 
            under the terms of the GNU General Public License as published by the Free Software Foundation; 
            either version 2 of the License, or (at your option) any later version.

            mosaic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
            without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
            See the GNU General Public License for more details. You should have received a copy of 
            the GNU General Public License along with File Hunter; if not, write to 
            the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA"""

        info = wx.AboutDialogInfo()

        info.SetIcon(wx.Icon(os.path.join("icons", "transmissionLaue_fcc_111.png"),
                                                                                wx.BITMAP_TYPE_PNG))
        info.SetName("LaueTools")
        info.SetVersion("6.0")
        info.SetDescription(description)
        info.SetCopyright("(C) 2021 Jean-Sebastien Micha")
        info.SetWebSite("http://www.esrf.eu/UsersAndScience/Experiments/CRG/BM32/")
        info.SetLicence(licence)
        info.AddDeveloper("Jean-Sebastien Micha")
        info.AddDocWriter("Jean-Sebastien Micha")
        # info.AddArtist('The Tango crew')
        info.AddTranslator("Jean-Sebastien Micha")

        wx.AboutBox(info)
        event.Skip()

    def OnSave(self, _):
        """  save image as png or tiff (for mosaic)"""

        dlg = wx.TextEntryDialog(self, "Enter filename for image with extension (.png, .tiff)", "Saving image")

        if dlg.ShowModal() == wx.ID_OK:
            filename = str(dlg.GetValue())
            if self.dirname is None:
                self.dirname = os.path.curdir

            fullpath = os.path.join(str(self.dirname), str(filename))

            if fullpath.endswith('.png'):
                self.axes.get_figure().savefig(fullpath)
                print("Image saved in ", fullpath)
            elif fullpath.endswith('.tiff'):  # TODO convert better float to uint16
                datint = np.array(self.data, dtype=np.uint16)
                TIFF.imsave(fullpath, datint)
                print("Image saved in tiff format ", fullpath)
        dlg.Destroy()

    def SaveData(self, _):
        wx.MessageBox("To be implemented, but data are automatically saved in the same folder "
                                                                "than the images one", "INFO")

    def OpenFolder(self, _):
        """find 2dmaps file in a folder"""
        
        folder = wx.FileDialog(self, "Select folder containing 2D map files",
                                wildcard="Bliss ESRF hdf5 (*.h5)|*.h5|All files(*)|*",
                                defaultDir=str(self.currentfolder))

        self.last_folder = self.currentfolder
        if folder.ShowModal() == wx.ID_OK:

            self.currentfolder = os.path.split(folder.GetPath())[0]
            print('\nselected folder is ===> ', self.currentfolder)

        # simply update list fo scans
        # if (self.currentfolder == self.last_folder and self.currentfolder is not None):
        #     self.onUpdatehdf5File(1)

        self.get_list_files()

        if (self.currentfolder != self.last_folder and self.currentfolder is not None):
            #print("\n\ndeleting last old items\n\n")
            self.treemesh.tree.DeleteAllItems()
            wx.CallAfter(self.treemesh.maketree)

        self.listfilestoAdd = self.listfiles

        wx.CallAfter(self.fill_tree)

    def fill_tree(self):
        for fileelem in self.listfilestoAdd:
            self.treemesh.tree.AppendItem(self.treemesh.root, str(fileelem))
    
    def select2Dmapfile(self, listfiles):
        finallist=[]
        for elem in listfiles:
            print('elem', elem)
            if 'mean_2D' in elem or 'max_2D' in elem:
                finallist.append(elem)
        return finallist

    def get_list_files(self):
        """ retrieve list of map files in the folder"""
        samefolder = False
    
        print('in get_list_files')
        currentfolder = self.currentfolder
        if self.currentfolder != self.last_folder:
            samefolder = True
            self.listfiles = None

        lastscan_listfiles = 0
        list_lastfiles_indices = []
        if self.listfiles is not None:
            print("self.listfiles already exists")
            print("self.listfiles", self.listfiles)
            list_lastfiles_indices = []
            for ms in self.listfiles:
                list_lastfiles_indices.append(ms[0])
            lastscan_listfiles = max(list_lastfiles_indices)

        print("lastscan_listfiles", lastscan_listfiles)

        listfilesall  = self.select2Dmapfile(os.listdir(currentfolder))
        #listfilesall = ['toto','titi','tata']

        print('listfilesall',listfilesall)


        if listfilesall == []:
            print('No map files apparently in %s'%self.currentfolder)
            self.listfiles = []
            return

        # list_meshscan_indices = []
        # for ms in listfilesall:
        #     if ms[0] not in list_lastfiles_indices:
        #         list_meshscan_indices.append(ms[0])

        self.listfiles = listfilesall
        #self.list_meshscan_indices = list_meshscan_indices

        print('self.listfiles', self.listfiles)
        #print('hdf5 self.list_meshscan_indices', self.list_meshscan_indices)


    def OpenData(self, _):

        if self.askUserForFilename():
            fpath = os.path.join(self.dirname, self.filename)
            print("Read file ", fpath)

            self.readData(fpath)

    def readData(self,fpath):

            with open(fpath, 'r') as f:
                d=read2Dmapfile(fpath)

            self.dirname,self.filename=os.path.split(fpath)

            print('type',type(d))
            dshape = d.get('shape')
            dataarray = d['data']
            n1, n2 = d['shape']
            self.title=d.get('title','%s'%self.filename)

            self.dataraw = copy.copy(dataarray)
            # data to be displayed
            self.data = dataarray
            #self.datatype = datatype

            if not self.filename.startswith('MOSAIC'):
                defaut_nb_col = n2
                defaut_nb_lines = n1
                self.mosaic = 0
            else:
                wx.MessageBox('Reload mosaic not yet implemented','INFO')
            #self.absolutecornerindices = absolutecornerindices
            self.nb_columns = d.get('nb_col',defaut_nb_col)
            self.nb_lines = d.get('nb_lines',defaut_nb_lines)

            self.Imageindices=d.get('Imageindices',np.arange(n1*n2))
            self.boxsize_row = d.get('boxsize_col',10) # nb columns !! nb of images per line
            self.boxsize_line = d.get('boxsize_line',10)
            #print('nb_row  nb_lines', nb_row, nb_lines)
            #print('boxsize_row  boxsize_line',boxsize_row, boxsize_line)
            self.stepindex = 1
            self.imagename = self.filename

            print('end of readData')
            print('data', self.data.shape, self.data[3])

            self.setArrayImageIndices()
            self._replot()

    def askUserForFilename(self, **dialogOptions):
        dialog = wx.FileDialog(self, **dialogOptions)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True

            pf=dialog.GetPath()
            self.dirname,self.filename=os.path.split(pf)
            
            print('selected filename', self.filename)
            print('selectred folder', self.dirname)

        else:
            userProvidedFilename = False
        dialog.Destroy()
        return userProvidedFilename

    def defaultFileDialogOptions(self):
        """ Return a dictionary with file dialog options that can be
            used in both the save file dialog as well as in the open
            file dialog. """

        return dict(message="Choose a file", defaultDir=self.dirname, wildcard="*.*")

    def OnQuit(self, _):
        self.Close(True)

    def _replotWithCurrentLimits(self):
        self.memorizedxlimits = self.axes.get_xlim()
        self.memorizedylimits = self.axes.get_ylim()

        self.OnMaskWeakPeaks()
        self._replot()

    def OnChangeVmin(self, _):
        self.IminDisplayed = float(self.vmintxtctrl.GetValue())

        if self.datatype in ("Vector",):
            self._replotWithCurrentLimits()
        else:
            self.normalizeplot(shrinkrange=True)
            self.canvas.draw()

    def OnChangeVmax(self, _):
        self.ImaxDisplayed = float(self.vmaxtxtctrl.GetValue())
        # print("self.ImaxDisplayed ------->", self.ImaxDisplayed)

        if self.datatype in ("Vector",):
            self._replotWithCurrentLimits()
        else:
            self.normalizeplot(shrinkrange=True)
            self.canvas.draw()

    def OnSliderMin(self, _):

        self.ImaxDisplayed = int(self.slider_min.GetValue())
        if self.IminDisplayed > self.ImaxDisplayed:
            self.slider_min.SetValue(self.ImaxDisplayed - 1)
            self.IminDisplayed = self.ImaxDisplayed - 1

        if self.datatype in ("Vector",):
            self._replotWithCurrentLimits()
        else:
            self.normalizeplot()
            self.canvas.draw()

    def OnSliderMax(self, _):

        self.ImaxDisplayed = int(self.slider_max.GetValue())
        # print("self.ImaxDisplayed ------->", self.ImaxDisplayed)

        if self.ImaxDisplayed < self.IminDisplayed:
            self.slider_max.SetValue(self.IminDisplayed + 1)
            self.ImaxDisplayed = self.IminDisplayed + 1
        if self.datatype in ("Vector",):
            self._replotWithCurrentLimits()
        else:
            self.normalizeplot()
            self.canvas.draw()

    def OnSliderArrowSize(self, _):
        self.arrowWidthDisplayed = int(self.slider_arrowwidth.GetValue())
        print("self.arrowWidthDisplayed ", self.arrowWidthDisplayed)

        self.arrowwidth = self.convertArrowWidth(self.arrowWidthDisplayed)
        print("new self.arrowwidth", self.arrowwidth)

        self._replotWithCurrentLimits()

    def convertArrowWidth(self, percent, mini=0.0001, maxi=0.02):

        return (maxi - mini) / 100.0 * percent

    def OnSliderArrowScale(self, _):
        self.arrowScaleDisplayed = int(self.slider_arrowscale.GetValue())
        print("self.arrowScaleDisplayed ", self.arrowScaleDisplayed)

        self.arrowscale = self.convertArrowScale(self.arrowScaleDisplayed)
        print("new self.arrowscale", self.arrowscale)

        self._replotWithCurrentLimits()

    def convertArrowScale(self, percent, mini=0.1, maxi=1000):

        return 10 ** (0 + (4) / 100.0 * percent)

    #     def OnSliderArrowScaleCutOff(self,event):
    #         self.arrowScaleDisplayedCutOff=int(self.slider_arrowscaleCutOff.GetValue())
    #
    #         self.arrowscaleCutOff = self.convertArrowScaleCutOff(self.arrowScaleDisplayedCutOff)
    #
    #         print 'max length for vector shift : ',self.arrowscaleCutOff
    #
    #         self._replotWithCurrentLimits()
    #
    #     def convertArrowScaleCutOff(self,percent, mini=0.1, maxi=10):
    #         """mini and maxi in pixel unit
    #
    #         """
    #         return (maxi-mini)/100.*percent

    def OnChangeLUT(self, _):

        #         print "OnChangeLUT"
        self.cmap = self.comboLUT.GetValue()
        self.myplot.set_cmap(self.cmap)
        self.canvas.draw()

    def OnTypeLUT(self, _):

        print("OnTypeLUT")
        self.cmap = self.comboLUT.GetValue()
        self.myplot.set_cmap(self.cmap)
        self.canvas.draw()

    def OnChangeScale(self, _):
        self.scaletype = str(self.comboscale.GetValue())
        self.normalizeplot()
        self.canvas.draw()

    def onchangeroiindex(self, _):
        self.dataroiindex = int(self.slidertoiindex.GetValue())
        self.data = self.datarois[self.dataroiindex]
        self.myplot.set_data(self.data)
        self.normalizeplot()
        self.canvas.draw()

    def OnChangeAspect(self, _):
        self.aspect = str(self.comboaspect.GetValue())
        self.axes.set_aspect(self.aspect)
        self.canvas.draw()

    def OnChangeCenter(self, _, newCenter=None):

        self.memorizedxlimits = self.axes.get_xlim()
        self.memorizedylimits = self.axes.get_ylim()

        if self.datatype == "Vector":

            (VectorX, VectorY, OriginVector) = self.dataVectorinit

            if newCenter is None:
                pixelCenter = (float(self.Xcenterctrl.GetValue()), float(self.Ycenterctrl.GetValue()))
            else:
                pixelCenter = newCenter

            xshift = pixelCenter[0] - OriginVector[0]
            yshift = pixelCenter[1] - OriginVector[1]

            newVectorX = VectorX - xshift
            newVectorY = VectorY - yshift

            self.dict_param["dataVector"] = [newVectorX, newVectorY, pixelCenter]
            self.dataVectorinit_shifted = [newVectorX, newVectorY, pixelCenter]

        else:

            (dataX_2D, dataY_2D, pixelCenter) = self.dict_param["dataRadialPosition"]

            if newCenter is None:
                pixelCenter = (float(self.Xcenterctrl.GetValue()), float(self.Ycenterctrl.GetValue()))
            else:
                pixelCenter = newCenter

            radialdistance_2D = np.sqrt(
                (dataX_2D - pixelCenter[0]) ** 2 + (dataY_2D - pixelCenter[1]) ** 2)
            self.data = radialdistance_2D

        self.OnMaskWeakPeaks()
        self._replot()

    def getMeanLevel(self):
        return np.mean(self.data)

    def Oncheckgrid(self, _):
        self.plotgrid = not self.plotgrid
        self._replot()

    def OnSubstractBackground(self, _):
        self.removebackground = not self.removebackground

        print("self.removebackground is now", self.removebackground)

        if self.removebackground:
            bkglevel = float(self.bkgsetctrl.GetValue())
            #             self.data = np.fabs(self.dataraw-bkglevel)
            self.data = self.dataraw - bkglevel

            print("removing ", bkglevel)
            print(self.dataraw[0, 0], self.data[0, 0])
        else:
            self.data = self.dataraw

        #         self.myplot.set_data(self.data)

        self._replot()

    #         self.normalizeplot()
    #         self.canvas.draw()

    def OnMask(self, _):
        self.OnMaskWeakPeaks()

        self._replot()

    def OnMaskWeakPeaks(self):
        """ mask self.data according self.maskthresholdctrl value (lowest accepted peak amplitude)
        + mask longest peak position shift vectors

        """
        print("OnMaskWeakPeaks")
        try:
            maskthreshold = float(self.maskthresholdctrl.GetValue())
        except:
            wx.MessageBox("Peak amplitude threshold must be float or integer", "INFO")
        (peak_X,
            peak_Y,
            peak_I,
            peak_fwaxmaj,
            peak_fwaxmin,
            peak_inclination,
            Xdev,
            Ydev,
            peak_bkg,
            maskedrows,
        ) = self.ApplyMaskOnPeaksProperties(maskthreshold).T

        xDATA, yDATA = peak_X, peak_Y

        #         print 'xDATA',xDATA

        (n0, n1) = self.dataraw.shape
        # dataX_2D = xDATA.reshape((n0, n1))
        # dataY_2D = yDATA.reshape((n0, n1))
        masked_2D = maskedrows.reshape((n0, n1))

        mask = masked_2D.mask
        #         print 'masked_2D.mask',mask

        if self.datatype in ("Vector",):

            vx, vy, center = self.dataVectorinit_shifted
            #             vx,vy = self.dict_param['dataVector'][:2]

            Vx = np.ma.array(vx, mask=mask)
            Vy = np.ma.array(vy, mask=mask)

            #             maxVectorLength=self.arrowscaleCutOff
            maxVectorLength = float(self.masklongvectorctrl.GetValue())

            cond_TooLongVector = Vx ** 2 + Vy ** 2 > maxVectorLength ** 2

            VX = np.ma.masked_where(cond_TooLongVector, Vx)
            VY = np.ma.masked_where(cond_TooLongVector, Vy)

            print("vx", vx)
            print("Vx", Vx)
            print("VX", VX)

            self.dict_param["dataVector"][0] = VX
            self.dict_param["dataVector"][1] = VY

        else:

            self.data = np.ma.array(self.dataraw, mask=mask)

        return

    def ApplyMaskOnPeaksProperties(self, maskthreshold):
        """ filter  fittedPeaksData according to maskthreshold
        
        return FilteredfittedPeaksData
        """
        print("ApplyMask keeping peaks results with amplitude larger than %d" % maskthreshold)

        fittedPeaksData = self.dict_param["FilteredfittedPeaksData"]

        (peak_X, peak_Y, peak_I,
            peak_fwaxmaj, peak_fwaxmin, peak_inclination,
            Xdev, Ydev, peak_bkg, maskedrows, ) = fittedPeaksData.T

        cond = (peak_I - peak_bkg) < maskthreshold
        to_reject0 = np.where(cond)[0]
        #         print 'to_reject0 due to peak amplitude < %d'%maskthreshold, to_reject0

        print("nb peaks", len(peak_X))

        ToR = set(to_reject0)  # to reject

        print("After amplitude thres, %d/%d peaks have been rejected" % (len(ToR), len(peak_X)))

        ToTake = set(np.arange(len(peak_X))) - ToR

        #         print 'Remaining non masked indices',ToTake

        BoolToTake = False * np.ones(len(peak_X))
        BoolToTake[list(ToTake)] = True

        BoolToMask = True * np.ones(len(peak_X))
        BoolToMask[list(ToTake)] = False

        #         print 'BoolToMask',BoolToMask

        #         print "where Masked", np.where(BoolToMask==True)
        maskedrows2 = np.where(BoolToMask == True)[0]

        FilterX0 = np.ma.array(peak_X)
        FilterX = np.ma.masked_where(BoolToMask, FilterX0)

        # all peaks list building
        fittedPeaksData = np.ma.array([peak_X, peak_Y, peak_I,
                                        peak_fwaxmaj, peak_fwaxmin, peak_inclination,
                                        Xdev, Ydev, peak_bkg, FilterX]).T

        FilteredfittedPeaksData = np.ma.mask_rowcols(fittedPeaksData, axis=0)

        return FilteredfittedPeaksData

    def normalizeplot(self, shrinkrange=False):

        if self.datatype in ("Vector",):
            DxArray, DyArray, OriginV = self.dict_param["dataVector"]
            Dnorm = np.hypot(DxArray, DyArray)

            dataforNormalization = Dnorm
        else:
            dataforNormalization = self.data

        if not shrinkrange:
            self.maxvals = np.amax(dataforNormalization)
            self.minvals = np.amin(dataforNormalization)
        else:
            self.maxvals = self.ImaxDisplayed
            self.minvals = self.IminDisplayed

        # print("in ImshowFrame()")
        # print("self.minvals", self.minvals)
        # print("self.maxvals", self.maxvals)
        # print("self.IminDisplayed", self.IminDisplayed)
        # print("self.ImaxDisplayed", self.ImaxDisplayed)

        if not shrinkrange:
            self.deltavals = (self.maxvals - self.minvals) / 100.0

            # print("self.deltavals", self.deltavals)

            vmin = self.minvals + self.IminDisplayed * self.deltavals
            vmax = self.minvals + self.ImaxDisplayed * self.deltavals
        else:
            vmin = self.minvals
            vmax = self.maxvals

        # print("\nvmin", vmin)
        # print("vmax", vmax, " \n")

        if self.scaletype == "Linear":

            self.cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.scaletype == "Log":
            if self.minvals <= 0.0:
                self.minvals = 0.000000000001
                vmin = 0.000000000001
            self.cNorm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        else:
            self.cNorm = None

        self.myplot.set_norm(self.cNorm)

    def _replot(self):
        """
        in ImshowFrame
        """

        print('self.datatype in _replot', self.datatype)
        def fromindex_to_pixelpos_x(index, _):
            return index  # self.center[0]-self.boxsize[0]+index

        def fromindex_to_pixelpos_y(index, _):
            return index  # self.center[1]-self.boxsize[1]+index

        def fromindex_to_pixelpos_x_mosaic(index, _):
            return index  # self.center[0]-self.boxsize[0]+index

        def fromindex_to_pixelpos_y_mosaic(index, _):
            return index  # self.center[1]-self.boxsize[1]+index

        #         fig = self.plotPanel.get_figure()
        #         self.axes = fig.gca()
        #
        #         # clear the axes and replot everything
        #         self.axes.cla()

        self.axes.clear()
        #        self.axes.set_autoscale_on(False) # Otherwise, infinite loop
        self.axes.set_autoscale_on(True)

        # quiver plot
        if self.datatype == "Vector":
            U, V, center = self.dict_param["dataVector"]

            n0, n1 = U.shape
            M = np.hypot(U, V)
            #             print "n0,n1",n0,n1
            X, Y = np.meshgrid(np.arange(n1), np.arange(n0))

            #             print "X.shape",X.shape
            #             print "X",X
            self.myplot = self.axes.quiver(X, Y, V, U, M,
                width=self.arrowwidth,
                scale=self.arrowscale,
                pivot="mid",
                edgecolors=("k"))
            self.axes.plot(X, Y, "k.", markersize=2)
            if self.memorizedxlimits is None:
                self.axes.axis([-1, n1, -1, n0])
        else:  # 2D imshow scalar plot

            self.myplot = self.axes.imshow(self.data,
                                            cmap=self.LastLUT,
                                            interpolation="nearest",
                                            origin=self.originYaxis)
            if self.colorbar is None:                                
                self.colorbar = self.fig.colorbar(self.myplot)


            if self.plotgrid:
                # adding grid to separate imagelet
                from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
                # Change major ticks to show every n pixels along x and along y.
                self.axes.xaxis.set_major_locator(MultipleLocator(2*self.boxsize_row+1))
                self.axes.yaxis.set_major_locator(MultipleLocator(2*self.boxsize_line+1))

                # Change minor ticks to show every 5. (20/4 = 5)
                #self.axes.xaxis.set_minor_locator(AutoMinorLocator(4))
                #self.axes.yaxis.set_minor_locator(AutoMinorLocator(4))

                # Turn grid on for both major and minor ticks and style minor slightly
                # differently.
                self.axes.grid(which='major', color='#0e2f44', linestyle='-')
                #self.axes.grid(which='minor', color='#CCCCCC', linestyle=':')

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))

        font0 = FontProperties()
        font0.set_size("x-small")

        self.axes.set_title("%s\n" % self.imagename)

        self.axes.set_aspect("auto")

        self.normalizeplot()

        if self.memorizedxlimits is not None:
            self.axes.set_xlim(self.memorizedxlimits)
            self.axes.set_ylim(self.memorizedylimits)

            self.memorizedxlimits = None
            self.memorizedylimits = None

        if "xlabel" in self.dict_param and "ylabel" in self.dict_param:
            self.axes.set_xlabel(self.dict_param["xlabel"])
            self.axes.set_ylabel(self.dict_param["ylabel"])

        self.canvas.draw()

    # --- start of cursor
    def OnPaint(self, event):
        self.erase_cursor()
        try:
            del self.lastInfo
        except AttributeError:
            pass
        self.canvas.draw()
        event.Skip()

    def mouse_move(self, event):
        self.draw_cursor(event)

    def onClick(self, event):
        """ on mouse click
        """
        self.centerx, self.centery = event.xdata, event.ydata

        if event.inaxes:
            if event.button in (2, 3):
                self.OnRightButtonMousePressed(1)

        else:
            pass


    def OnRightButtonMousePressed(self, _):
        self.DisplayXYZMotorsPositions()

    def GetXYZmotorsfromCurrentImage(self):
        """
        Read metadata of motors and exposure time
        #TODO   add exposure time
        """
        print("current image index pointed", self.currentpointedImageIndex)
        imagesfolder = self.dict_param["imagesfolder"]
        imagefilenameexample = self.dict_param["filename_representative"]
        CCDLabel = self.dict_param["CCDLabel"]

        imagefilename = IOimage.setfilename(imagefilenameexample, self.currentpointedImageIndex,
                                                                                    4, CCDLabel)

        fullpathtoimagefile = os.path.join(imagesfolder, imagefilename)
        xyzech, expotime = IOimage.read_motorsposition_fromheader(fullpathtoimagefile)
        return xyzech

    def DisplayXYZMotorsPositions(self):
        xyzech = self.GetXYZmotorsfromCurrentImage()

        if xyzech is None:
            return

        tip = "Pointed sample position:\nxech=%.6f yech=%.6f zech=%.6f" % tuple(xyzech)
        print(tip)

        self.tooltip.SetTip(tip)
        self.tooltip.Enable(True)

    def onKeyPressed(self, event):

        key = event.key
        print("key ==> ", key)

        if key == "escape":

            ret = wx.MessageBox("Are you sure to quit?", "Question", wx.YES_NO | wx.NO_DEFAULT, self)

            if ret == wx.YES:
                self.Close()

        elif key == "p":  # 'p'

            xyzech = self.GetXYZmotorsfromCurrentImage()

            if xyzech is None:
                return

            posmotor1, posmotor2 = xyzech[:2]
            motor1name, motor2name = "xech", "yech"

            print("SPEC COMMAND:\nmv %s %.6f %s %.6f" % (motor1name, posmotor1, motor2name, posmotor2))

            sentence = ("%s=%.6f\n%s=%.6f\n\nSPEC COMMAND to move to this point:\n\nmv %s %.5f %s %.5f"
                % (motor1name, posmotor1, motor2name, posmotor2, motor1name,
                                            posmotor1, motor2name, posmotor2))

            command = "mv %s %.5f %s %.5f" % (motor1name, posmotor1, motor2name, posmotor2)

            #                     wx.MessageBox(sentence, 'INFO')

            msgdialog = MC.MessageCommand(self,
                                            -1,
                                            "motors command",
                                            sentence=sentence,
                                            speccommand=command,
                                            specconnection=None)
            msgdialog.ShowModal()

            return

        elif key == "c":  # capture the center of the given image

            rx = int(np.round(self.centerx))
            ry = int(np.round(self.centery))

            col = int(rx + 0.5)
            row = int(ry + 0.5)
            numrows, numcols = self.data.shape

            #         print "self.tabindices", self.tabindices
            if col >= 0 and col < numcols and row >= 0 and row < numrows:

                absoluteImageIndex = self.getImageIndexfromxy(self.centerx, self.centery)

                if self.datatype in ("RadialPosition", "Vector"):
                    if self.datatype in ("Vector",):
                        print("Changing center of radial distance to ...", self.centerx, self.centery)
                        VectorX, VectorY, OriginVector = self.dict_param["dataVector"]
                        centerx = VectorX[row, col] + OriginVector[0]
                        centery = VectorY[row, col] + OriginVector[1]

                    elif self.datatype in ("RadialPosition",):
                        print("Changing center of radial distance to ...", self.centerx, self.centery)
                        (dataX_2D, dataY_2D, pixelCenter) = self.dict_param["dataRadialPosition"]
                        print("row,col", row, col)
                        centerx = dataX_2D[row, col]
                        centery = dataY_2D[row, col]

                    if not isinstance(centerx, (int, float)):
                        return

                    ret = wx.MessageBox("Set new center to fitted position of image %d\n(%.2f,%.2f)"
                        % (absoluteImageIndex, centerx, centery),
                        "Radial Distance Map Centering", wx.YES_NO | wx.NO_DEFAULT, self)

                    if ret == wx.YES:
                        print("OK Centering")
                        self.OnChangeCenter(1, (centerx, centery))
                else:
                    print("Changing center of Y data  according to value of clicked image")

                    print("row,col", row, col)
                    neworigin = self.dataraw[row, col]

                    if not isinstance(neworigin, (int, float)):
                        return

                    ret = wx.MessageBox(
                        "Set new center of Y data with origin = Y value of image %d\n%.2f"
                        % (absoluteImageIndex, neworigin),
                        "Change of Y value Origin", wx.YES_NO | wx.NO_DEFAULT, self)

                    if ret == wx.YES:
                        print("OK Centering")
                        self.data = self.dataraw - neworigin
                        self._replot()

            return

    def draw_cursor(self, event):
        """event is a MplEvent.  Draw a cursor over the axes"""
        if event.inaxes is None:
            self.erase_cursor()
            try:
                del self.lastInfo
            except AttributeError:
                pass
            return
        canvas = self.canvas
        #         print 'canvas', dir(canvas.figure.bbox)
        #         figheight = canvas.figure.bbox.height()
        figheight = canvas.figure.bbox.height
        ax = event.inaxes
        #         left, bottom, width, height = ax.bbox.get_bounds()
        left, bottom, width, height = ax.bbox.bounds
        bottom = figheight - bottom
        top = bottom - height
        right = left + width
        x, y = event.x, event.y
        y = figheight - y

        dc = wx.ClientDC(canvas)
        dc.SetLogicalFunction(wx.XOR)
        wbrush = wx.Brush(wx.Colour(255, 255, 255), wx.TRANSPARENT)
        wpen = wx.Pen(wx.Colour(200, 200, 200), 1, wx.SOLID)
        dc.SetBrush(wbrush)
        dc.SetPen(wpen)

        dc.ResetBoundingBox()

        if 0:  # issue of cursor refreshing when using tunnel ssh
            if sys.platform not in ("darwin",):
                if not WXPYTHON4:
                    dc.BeginDrawing()

                x, y, left, right, bottom, top = [int(val) for val in (x, y, left, right, bottom, top)]

                self.erase_cursor()
                line1 = (x, bottom, x, top)
                line2 = (left, y, right, y)
                self.lastInfo = line1, line2, ax, dc
                dc.DrawLine(*line1)  # draw new
                dc.DrawLine(*line2)  # draw new
                if not WXPYTHON4:
                    dc.EndDrawing()

        xabs = int(np.round(event.xdata))
        yabs = int(np.round(event.ydata))

        textsb = "Mosaic: x=%d y=%d" % (xabs, yabs)

        if 1:  # self.viewingLUTpanel.show2thetachi.GetValue():
            if 1:  # self.CCDcalib is not None:
                #                 tth, chi = F2TC.calc_uflab([xabs, xabs], [yabs, yabs], self.CCDcalib['CCDCalibParameters'],
                #                                                returnAngles=1, pixelsize=165. / 2048,
                #                                                kf_direction='Z>0')
                #                 textsb += '   (2theta, chi)= %.2f, %.2f' % (tth[0], chi[0])
                textsb += "   (2theta, chi)= %.2f, %.2f" % (0.0, 0.0)

        self.sb.SetStatusText(textsb, 0)

    def erase_cursor(self):
        try:
            lastline1, lastline2, lastax, lastdc = self.lastInfo
        except AttributeError:
            pass
        else:
            lastdc.DrawLine(*lastline1)  # erase old
            lastdc.DrawLine(*lastline2)  # erase old

    # --- end of cursor

    def setArrayImageIndices(self):
        imageindices = np.arange(0, self.nb_lines * self.nb_columns) * self.stepindex

        self.tabindices = imageindices.reshape((self.nb_lines, self.nb_columns))

    def getImageIndexfromxy(self, x, y):
        """  return absolute image index from x,y coordinates of a mosaic plot

        use self.boxsize_line, self.boxsize_row
        self.tabindices
        self.Imageindices
        """
        indi = int(y / (2 * self.boxsize_line + 1))
        indj = int(x / (2 * self.boxsize_row + 1))

        nlines, ncols = self.tabindices.shape

        if indi < 0:
            indi = 0
        if indi >= nlines:
            indi = nlines - 1

        if indj < 0:
            indj = 0
        if indj >= ncols:
            indj = ncols - 1

        relativeImageindex = self.tabindices[indi, indj]

        #         print 'relativeindex',relativeImageindex
        absoluteImageIndex = self.Imageindices[relativeImageindex]

        return absoluteImageIndex

    def getImageIndexfromxy_scalarplot(self, x, y):
        """  return absolute image index from x,y coordinates of a scalar plot
        use 
        self.tabindices
        self.Imageindices
        """
        # print('self.tabindices',self.tabindices)
        # print('self.Imageindices',self.Imageindices)

        indi = int(y)
        indj = int(x)

        nlines, ncols = self.tabindices.shape

        if indi < 0:
            indi = 0
        if indi >= nlines:
            indi = nlines - 1

        if indj < 0:
            indj = 0
        if indj >= ncols:
            indj = ncols - 1

        relativeImageindex = self.tabindices[indi, indj]

        absoluteImageIndex = self.Imageindices[relativeImageindex]

        return absoluteImageIndex

    def format_coord(self, x, y):
        """
        ImshowFrame : show when hovering mouse
        """
        col = int(x + 0.5)
        row = int(y + 0.5)
        numrows, numcols = self.data.shape
        #         print "self.Imageindices", self.Imageindices
        #         print "len()", len(self.Imageindices)
        #         print "to be reshaped", (self.nb_lines, self.nb_columns)

        if len(self.Imageindices) != (self.nb_lines * self.nb_columns):
            print(' len self.Imageindices',len(self.Imageindices))
            print('nb kines and columne',self.nb_lines, self.nb_columns)
            print("WARNING:  display may not work , check strictly that ")
            print("the number of images is a multiple of the number of lines !!!")

        #         print "self.tabindices", self.tabindices
        if col >= 0 and col < numcols and row >= 0 and row < numrows:

            # print int(y/(2*self.boxsize_row)),int(x/(2*self.boxsize_line))

            if self.mosaic:
                z = self.data[row, col]
                #                 print "hello in mosaic"
                logz = np.log(z)
                self.currentpointedImageIndex = self.getImageIndexfromxy(x, y)
                #                 print "self.absolutecornerindices",self.absolutecornerindices
                if self.absolutecornerindices is not None:
                    jmin, imin = self.absolutecornerindices
                    xpixel, ypixel = (jmin + x % (2 * self.boxsize_row + 1),
                        imin + 2 * self.boxsize_line - y % (2 * self.boxsize_line + 1))
                    #                     xpixel, ypixel = (jmin + x % (2 * self.boxsize_row + 1),
                    #                                       imin + y % (2 * self.boxsize_line + 1))
                    
                    return "MOSAIC x=%1.4f, y=%1.4f, log(I)=%1.4f, I=%5.1f, ImageIndex: %d" % (
                                        xpixel, ypixel, logz, z, self.currentpointedImageIndex)
                else:
                    
                    return "Mosaic x=%1.4f, y=%1.4f, log(I)=%1.4f, I=%5.1f, ImageIndex: %d" % ( x, y,
                                                        logz, z, self.currentpointedImageIndex)
            else:
                #                 self.currentpointedImageIndex=self.getImageIndexfromxy(x,y)
                self.currentpointedImageIndex = self.getImageIndexfromxy_scalarplot(x, y)

                if self.datatype == "Vector":
                    DxArray, DyArray, _ = self.dict_param["dataVector"]
                    Dx = DxArray[row, col]
                    Dy = DyArray[row, col]
                    Dnorm = np.hypot(Dx, Dy)

                    return ("x=%1.4f, y=%1.4f, Dx=%.2f Dy=%.2f,\nDnorm %.2f, ImageIndex: %d"
                        % (x, y, Dx, Dy, Dnorm, self.currentpointedImageIndex))
                elif self.datatype == "ReciprocalMap":
                    z = self.data[row, col]
                    qx, qz, qn, E = self.dict_param["QxyznE"][row, col]
                    sentence = "x=%1.4f, y=%1.4f, I=%5.1f, ImageIndex: %d" % (x, y, z,
                                                                    self.currentpointedImageIndex)
                    sentence += "\n qx=%1.4f, qz=%1.4f, qn=%.5f, Energy: %.4f" % (qx, qz, qn, E)
                    return sentence

                else:  # basic scalar plot
                    z = self.data[row, col]
                    return "SCALAR x=%1.4f, y=%1.4f, I=%5.5f, ImageIndex: %d" % (x, y, z,
                                                                    self.currentpointedImageIndex)

        else:
            return "x=%1.4f, y=%1.4f" % (x, y)

    def onMotion_ToolTip(self, event):
        """tool tip to show data when mouse hovers on plot
        """

        if self.data is None:
            return

        collisionFound = False

        if event.xdata != None and event.ydata != None:  # mouse is inside the axes

            rx = int(np.round(event.xdata))
            ry = int(np.round(event.ydata))

            # create str expression to show on plot
            tip = self.format_coord(rx, ry)

            self.tooltip.SetTip(tip)
            self.tooltip.Enable(True)
            collisionFound = True
            #            break
            return
        if not collisionFound:
            pass

    def SaveAsRectROI(self, _):
        print("SaveAsRectROI")
        ROI_extent_values = self.getROIproperties()

        val_ROI = ROI_extent_values
        imageindexcenter = ROI_extent_values[0]

        helptstr = ("Enter ROI name for rectangle centered on image index = %d" % imageindexcenter)

        dlg = wx.TextEntryDialog(self, helptstr, "User sample map ROI name Entry")

        defautROIname = "G0_%d" % imageindexcenter
        dlg.SetValue(defautROIname)
        if dlg.ShowModal() == wx.ID_OK:
            ROIname = str(dlg.GetValue())

            dlg.Destroy()

        key_ROI = str(ROIname)

        self.updateROIdict(key_ROI, val_ROI)

    def SaveRectROI(self, _):
        print("SaveRectROI")
        ROI_extent_values = self.getROIproperties()

        key_ROI = str(ROI_extent_values[0])
        val_ROI = ROI_extent_values

        self.updateROIdict(key_ROI, val_ROI)

    def updateROIdict(self, key_ROI, val_ROI):
        self.dict_ROI[key_ROI] = val_ROI

        self.parent.comboROI.Append(key_ROI)

        print("ROI saved with key=%s" % str(key_ROI))
        print("with value:", val_ROI)

    def getROIproperties(self):

        xlimits = self.axes.get_xlim()
        ylimits = self.axes.get_ylim()

        print("self.axes.get_xlim()", xlimits)

        print("self.axes.get_ylim()", ylimits)

        xcenterROI, ycenterROI = (0.5 * (xlimits[0] + xlimits[1]), 0.5 * (ylimits[0] + ylimits[1]))
        halfxwidth = 0.5 * (xlimits[1] - xlimits[0])
        halfywidth = 0.5 * (ylimits[1] - ylimits[0])

        centerimageindex = self.getImageIndexfromxy(xcenterROI, ycenterROI)
        rightimage = self.getImageIndexfromxy(xcenterROI + halfxwidth, ycenterROI)
        leftimage = self.getImageIndexfromxy(xcenterROI - halfxwidth, ycenterROI)
        topimage = self.getImageIndexfromxy(xcenterROI, ycenterROI + halfywidth)
        bottomimage = self.getImageIndexfromxy(xcenterROI, ycenterROI - halfywidth)

        print("centerimageindex", centerimageindex)
        print("leftimage,rightimage", leftimage, rightimage)
        print("bottomimage,topimage", bottomimage, topimage)

        print("self.tabindices.shape", self.tabindices.shape)

        nbimagesxwidth = rightimage - leftimage
        nbimagesywidth = (topimage - bottomimage) / self.tabindices.shape[1]

        print("nbimagesxwidth", nbimagesxwidth)
        print("nbimagesywidth", nbimagesywidth)

        ROI_extent_values = (centerimageindex, nbimagesxwidth, nbimagesywidth)

        return ROI_extent_values

        self.dict_ROI[str(centerimageindex)] = ROI_extent_values

        self.parent.comboROI.Append(str(centerimageindex))

        print("ROI saved with key=%d" % centerimageindex)
        print("with value:", ROI_extent_values)
        return

    def LoadROI(self, _):
        wx.MessageBox("Not implemented yet!", "INFO")
        return

    def EditROI(self, _):
        wx.MessageBox("Not implemented yet!", "INFO")
        return

def read2Dmapfile(fullpath):
    with open(fullpath) as f:
        folder, mapfile = os.path.split(fullpath)
        if not mapfile.startswith(('MOSAIC','max_2D', 'mean_2D',
                                   'ptp_2D','Amplitude_2D',
                                   'Displacement_2D','Position XY_2D')):
            return None
        # test if there is header
        firstline=f.readline()
        #print(firstline)
        extractdata = False
        d={}
        headerexist=False
        
        if firstline.startswith('#'):  # there is a header with some metadata
            headerexist = True
            f.seek(0)
            inheader = True
            nb_rows_to_skip=0
            while inheader:
                l=f.readline()
                if l.startswith('# datatype'):
                    d['datatype']=l.strip().split(':')[-1]
                    nb_rows_to_skip+=1
                elif l.startswith('# dims'): # full data dimensions (= sample map for scalar map ?)
                    print((l.split(':')[-1]).strip().split(' '))
                    n1,n2=(l.split(':')[-1]).strip().split(' ')
                    d['dims']=(int(n1.strip()),int(n2.strip()))
                    nb_rows_to_skip+=1
                elif l.startswith('# Imageindices'):
                    tt=(l.split(':')[-1].strip())
                    d['Imageindices'] =[int(el) for el in tt.split()]
                    nb_rows_to_skip+=1
                elif l.startswith('# nb_col'):
                    d['nb_col']=int((l.split(':')[-1].strip()))
                    nb_rows_to_skip+=1
                elif l.startswith('# nb_lines'):
                    d['nb_lines']=int((l.split(':')[-1].strip()))
                    nb_rows_to_skip+=1
                elif l.startswith('# boxsize_col'):
                    d['boxsize_col']=int((l.split(':')[-1].strip()))
                    nb_rows_to_skip+=1
                elif l.startswith('# boxsize_line'):
                    d['boxsize_line']=int((l.split(':')[-1].strip()))
                    nb_rows_to_skip+=1
                elif l.startswith('# title'):
                    d['title']=(l.split(':')[-1].strip())
                    nb_rows_to_skip+=1
                    inheader = False
                    extractdata=True
                    print('nb_rows_to_skip',nb_rows_to_skip)


        if not firstline.startswith('#') or extractdata:
            print('d',d)
            print('now extracting data...')
            f.seek(0)
            if not headerexist: #"no header"
                nb_rows_to_skip=0
            data=np.loadtxt(f, skiprows=nb_rows_to_skip)
            print('data.shape',data.shape)
            d["data"]=data
            d["shape"]=data.shape
            return d    

def start():
    #dataarray = np.random.random((20,20))
    
    folder = os.path.split((os.path.abspath(__file__)))[0]
    print('absolute path of folder', folder)
    # 41 x n
    startindex = 0
    finalindex= 1383
    stepindex = 1
    nbimagesperline = 41

    dictrois = {}
    dictrois['nbrois']=80
    dictrois['roisinddices']=np.arange(0,80,1)
    dictrois['roimaxindex']=79
    dictroiprops = {}
    dictroiprops[0]=[200,400,200,400,300,300,200,200] # xmin, xmax, ymin,ymax, xc,yc, boxx,boxy
    dictrois['roisprops']=dictroiprops

    with open(os.path.join(folder,'alldetsSn1383.pickle'),'rb') as f:
        alldets=pickle.load(f)

    with open(os.path.join(folder,'allmaxsSn1383.pickle'),'rb') as f:
        allmaxs=pickle.load(f)  # relative positions of maximum pixel intensity in the all rois

    nbimages = finalindex-startindex+1
    nb_lines = nbimages//nbimagesperline+1
    if nbimages%nbimagesperline==0:
        nb_lines = nb_lines-1
    imageindices = np.arange(startindex,finalindex+1,stepindex)

    roi0= alldets[0]
    print('roi0.shape', roi0.shape)

    PSGUIApp = wx.App()
    PSGUIframe = ShowMapFrame(None, -1, 'show2Dmap',alldets,Imageindices=imageindices,
                                            nb_row=nbimagesperline, nb_lines=nb_lines,dictrois=dictrois,
                                            maxpositions=allmaxs,
                                            mosaic=0)
    PSGUIframe.Show()
    PSGUIApp.MainLoop()

if __name__ == "__main__":
    start()