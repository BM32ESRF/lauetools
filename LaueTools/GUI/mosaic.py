# -*- coding: utf-8 -*-
"""
Module for 2D rearrangement to be displayed as map of given quantities

mosaic.py belongs to the LaueTools Software

May 2019
"""
import os
import time
import sys
from copy import copy

try:
    import Image
except ImportError:
    print("-- warning. module Image or PIL is not installed but only for command line mosaic builder (mosaic.py)")

try:
    import wx
except ImportError:
    print("-- wx is not installed! Could be some trouble from this lack if you use GUIs...")

if wx.__version__ < "4.":
    WXPYTHON4 = False
    print("-- OK! You are using wxpython3 ....")

else:
    WXPYTHON4 = True
    print("-- OK! You are using wxpython4 ....")
    wx.OPEN = wx.FD_OPEN

    def sttip(argself, strtip):
        """ alias """
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

from scipy import ndimage as scind
import numpy as np

from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as pltticker
from matplotlib.font_manager import FontProperties

from matplotlib.figure import Figure

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavigationToolbar

import matplotlib as mpl
import matplotlib.cm as mplcm
from pylab import cm as pcm

# LaueTools modules
if sys.version_info.major == 3:
    from .. import dict_LaueTools as DictLT
    from .. import generaltools as GT
    from .. import readmccd as RMCCD
    from . import Plot1DFrame as PLOT1D
    from .. import MessageCommand as MC
    from .. import IOimagefile as IOimage
    from .. import imageprocessing as ImProc
else:
    import dict_LaueTools as DictLT
    import generaltools as GT
    import readmccd as RMCCD
    import Plot1DFrame as PLOT1D
    import MessageCommand as MC
    import IOimagefile as IOimage
    import imageprocessing as ImProc


# class ImshowFrameNew(wx.Frame):
#     r"""
#     Class to show 2D array intensity data

#     only loarded (but not used) in FileSeries/multigrainFS.py
#     """
#     def __init__(self, parent, _id, title, dataarray, posarray_twomotors=None,
#                                                     datatype="scalar",
#                                                     absolutecornerindices=None,
#                                                     Imageindices=None,
#                                                     nb_row=10,
#                                                     nb_lines=10,
#                                                     boxsize_row=10,
#                                                     boxsize_line=10,
#                                                     stepindex=1,
#                                                     imagename="",
#                                                     mosaic=0,
#                                                     extent=None,
#                                                     xylabels=None):
#         r"""
#         plot 2D plot of dataarray

#         posarray_twomotors  =  additional info to show in status bar when hovering on plot

#         """
#         # dat=dat.reshape(((self.nb_row)*2*self.boxsize_row,(self.nb_lines)*2*self.boxsize_line))
#         wx.Frame.__init__(self, parent, _id, title, size=(700, 700))

#         self.data = np.flipud(dataarray)
#         try:
#             self.dataarray_info = np.flipud(posarray_twomotors)
#         except ValueError:
#             print("data_info", posarray_twomotors)
#             self.dataarray_info = None
#         #         print "dataarray", dataarray
#         self.datatype = datatype

#         self.absolutecornerindices = absolutecornerindices
#         self.title = title
#         self.Imageindices = np.flipud(Imageindices)
#         self.nb_columns = nb_row
#         self.nb_lines = nb_lines
#         self.boxsize_row = boxsize_row
#         self.boxsize_line = boxsize_line
#         self.stepindex = stepindex
#         self.extent = extent
#         self.xylabels = xylabels
#         self.imagename = imagename
#         self.mosaic = mosaic
#         self.dirname = None
#         self.filename = None

#         print("self.data.shape in ImshowFrame", self.data.shape)
#         print("self.nb_columns, self.nb_lines", self.nb_columns, self.nb_lines)
#         print("self.boxsize_row, self.boxsize_line", self.boxsize_row, self.boxsize_line)

#         self.LastLUT = "OrRd"

#         self.create_main_panel()

#         self.clear_axes_create_imshow()

#     def create_main_panel(self):
#         r""" create main GUI panel of ImshowFrameNew class
#         """
#         # # Set up the MenuBar
#         MenuBar = wx.MenuBar()

#         FileMenu = wx.Menu()

#         OpenMenu = FileMenu.Append(wx.ID_ANY, "&Save", "Save Image")
#         self.Bind(wx.EVT_MENU, self.OnSave, OpenMenu)

#         # SaveMenu = FileMenu.Append(wx.ID_ANY, "&Save","Save BNA")
#         # self.Bind(wx.EVT_MENU, self.SaveBNA, SaveMenu)

#         CloseMenu = FileMenu.Append(wx.ID_ANY, "&Close", "Close Application")
#         self.Bind(wx.EVT_MENU, self.OnQuit, CloseMenu)

#         MenuBar.Append(FileMenu, "&File")

#         # view_menu = wx.Menu()
#         # ZoomMenu = view_menu.Append(wx.ID_ANY, "Zoom to &Fit","Zoom to fit the window")
#         # self.Bind(wx.EVT_MENU, self.ZoomToFit, ZoomMenu)
#         # MenuBar.Append(view_menu, "&View")

#         help_menu = wx.Menu()
#         AboutMenu = help_menu.Append(wx.ID_ANY, "&About", "More information About this program")
#         self.Bind(wx.EVT_MENU, self.OnAbout, AboutMenu)
#         MenuBar.Append(help_menu, "&Help")

#         self.SetMenuBar(MenuBar)

#         # #
#         self.CreateStatusBar()

#         self.panel = wx.Panel(self)

#         self.dpi = 100
#         self.figsize = 5
#         self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
#         self.canvas = FigCanvas(self.panel, -1, self.fig)

#         self.axes = self.fig.add_subplot(111)

#         self.toolbar = NavigationToolbar(self.canvas)

#         self.calc_norm_minmax_values()

#         self.slidertxt_min = wx.StaticText(self.panel, -1, "Min :")
#         self.slider_min = wx.Slider(self.panel, -1, size=(200, 50), value=0,
#                             minValue=0, maxValue=99, style=wx.SL_AUTOTICKS | wx.SL_LABELS, )
#         if WXPYTHON4:
#             self.slider_min.SetTickFreq(50)
#         else:
#             self.slider_min.SetTickFreq(50, 1)
#         self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMin, self.slider_min)

#         self.slidertxt_max = wx.StaticText(self.panel, -1, "Max :")
#         self.slider_max = wx.Slider(self.panel, -1, size=(200, 50), value=100,
#                             minValue=1, maxValue=100, style=wx.SL_AUTOTICKS | wx.SL_LABELS, )
#         if WXPYTHON4:
#             self.slider_max.SetTickFreq(50)
#         else:
#             self.slider_max.SetTickFreq(50, 1)
#         self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMax, self.slider_max)

#         # loading LUTS
#         self.mapsLUT = [m for m in pcm.datad if not m.endswith("_r")]
#         self.mapsLUT.sort()

#         luttxt = wx.StaticText(self.panel, -1, "LUT")
#         self.comboLUT = wx.ComboBox(self.panel, -1, self.LastLUT, choices=self.mapsLUT)  # ,
#         # style=wx.CB_READONLY)

#         self.comboLUT.Bind(wx.EVT_COMBOBOX, self.OnChangeLUT)

#         self.scaletype = "Linear"
#         # scaletxt = wx.StaticText(self, -1, "Scale")
#         self.comboscale = wx.ComboBox(self, -1, self.scaletype,
#                                                         choices=["Linear", "Log"], size=(-1, 40))

#         self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)

#         # --- ---layout
#         self.slidersbox = wx.BoxSizer(wx.HORIZONTAL)
#         self.slidersbox.Add(self.slidertxt_min, 0)
#         self.slidersbox.Add(self.slider_min, 0)
#         self.slidersbox.Add(self.slidertxt_max, 0)
#         self.slidersbox.Add(self.slider_max, 0)
#         self.slidersbox.Add(luttxt, 0)
#         self.slidersbox.Add(self.comboLUT, 0)

#         self.vbox = wx.BoxSizer(wx.VERTICAL)
#         self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
#         self.vbox.Add(self.slidersbox, 0, wx.EXPAND)
#         self.vbox.Add(self.toolbar, 0, wx.EXPAND)

#         self.panel.SetSizer(self.vbox)
#         self.vbox.Fit(self)
#         self.Layout()

#     def OnAbout(self, _):
#         pass

#     def OnChangeLUT(self, _):
#         print("OnChangeLUT")
#         self.myplot.set_cmap(self.comboLUT.GetValue())

#         self.canvas.draw()

#     def OnSliderMin(self, _):

#         self.IminDisplayed = int(self.slider_min.GetValue())
#         if self.IminDisplayed > self.ImaxDisplayed:
#             self.slider_min.SetValue(self.ImaxDisplayed - 1)

#         self.normalizeplot()
#         self.canvas.draw()

#     def OnSliderMax(self, _):
#         self.ImaxDisplayed = int(self.slider_max.GetValue())
#         if self.ImaxDisplayed < self.IminDisplayed:
#             self.slider_max.SetValue(self.IminDisplayed + 1)
#         self.normalizeplot()
#         self.canvas.draw()

#     def normalizeplot(self):
#         deltavals = (self.maxvals - self.minvals) / 100.0
#         vmin = self.minvals + self.IminDisplayed * deltavals
#         vmax = self.maxvals + self.ImaxDisplayed * deltavals

#         self.cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#         self.myplot.set_norm(self.cNorm)

#     def OnSave(self, _):
#         # if self.askUserForFilename(defaultFile='truc', style=wx.SAVE,**self.defaultFileDialogOptions()):
#         #    self.OnSave(event)
#         if self.askUserForFilename():
#             fig = self.plotPanel.get_figure()
#             fig.savefig(os.path.join(str(self.dirname), str(self.filename)))
#             print("Image saved in ", os.path.join(self.dirname, self.filename) + ".png")

#     def askUserForFilename(self, **dialogOptions):
#         dialog = wx.FileDialog(self, **dialogOptions)
#         if dialog.ShowModal() == wx.ID_OK:
#             userProvidedFilename = True
#             self.filename = dialog.GetFilename()
#             self.dirname = dialog.GetDirectory()
#             print(self.filename)
#             print(self.dirname)
#         else:
#             userProvidedFilename = False
#         dialog.Destroy()
#         return userProvidedFilename

#     def defaultFileDialogOptions(self):
#         """ Return a dictionary with file dialog options that can be
#             used in both the save file dialog as well as in the open
#             file dialog. """

#         return dict(message="Choose a file", defaultDir=self.dirname, wildcard="*.*")

#     def OnQuit(self, _):
#         self.Close(True)

#     def calc_norm_minmax_values(self):
#         if self.dataarray_info is not None:
#             self.maxvals = np.amax(self.dataarray_info)
#             self.minvals = np.amin(self.dataarray_info)

#             print("self.dataarray_info", self.dataarray_info)
#             print("self.dataarray_info max ", self.maxvals)
#             print("self.dataarray_info min ", self.minvals)

#         self.data_to_Display = self.data
#         self.cNorm = None

#         if self.datatype == "scalar":
#             print("plot of datatype = %s" % self.datatype)
#             try:
#                 self.data_to_Display = self.data[:, :, 0]
#             except IndexError:
#                 self.data_to_Display = self.data

#             self.maxvals = np.amax(self.data_to_Display)
#             self.minvals = np.amin(self.data_to_Display)
#             #             from matplotlib.colors import colorConverter
#             import matplotlib.colors as colors

#             #             import matplotlib.pyplot as plt
#             #             import matplotlib.cm as cmx
#             #             OrRd = cm = plt.get_cmap('OrRd')
#             self.cNorm = colors.Normalize(vmin=self.minvals, vmax=self.maxvals)

#     #             scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=OrRd)
#     #             print scalarMap.get_clim()
#     #         colorVal = scalarMap.to_rgba(values[idx])

#     def clear_axes_create_imshow(self):
#         """ init the figure
#         """
#         def fromindex_to_pixelpos_x_mosaic(index, _):
#             return index  # self.center[0]-self.boxsize[0]+index

#         def fromindex_to_pixelpos_y_mosaic(index, _):
#             return index  # self.center[1]-self.boxsize[1]+index

#         # clear the axes and replot everything
#         self.axes.cla()
#         self.axes.set_title(self.title)
#         self.myplot = self.axes.imshow(self.data_to_Display,
#                                         cmap=GT.ORRD,
#                                         interpolation="nearest",
#                                         norm=self.cNorm,
#                                         #                          extent=self.extent,
#                                         origin="lower")

#         self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x_mosaic))
#         self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y_mosaic))

#         self.axes.set_xlabel(self.xylabels[0])
#         self.axes.set_ylabel(self.xylabels[1])

#         self.axes.locator_params("x", tight=True, nbins=5)
#         self.axes.locator_params("y", tight=True, nbins=5)

#         self.IminDisplayed = 0
#         self.ImaxDisplayed = 200

#         self.axes.grid(True)

#         #         numrows, numcols, color = self.data.shape
#         numrows, numcols = self.data.shape[:2]

#         #         print "self.Imageindices", self.Imageindices
#         #         print "self.Imageindices.shape", self.Imageindices.shape
#         #         print "self.data.shape", self.data.shape

#         #         imageindices = self.Imageindices[0] + arange(0, numrows, numcols) * self.stepindex

#         tabindices = self.Imageindices
#         #         print "tabindices", tabindices

#         def format_coord(x, y):
#             col = int(x + 0.5)
#             row = int(y + 0.5)
#             if col >= 0 and col < numcols and row >= 0 and row < numrows:
#                 z = self.data[row, col]

#                 print("z", z)
#                 print("x,y,row,col", x, y, row, col)
#                 Imageindex = tabindices[row, col]
#                 if self.dataarray_info is None:
#                     sentence = "x=%1.4f, y=%1.4f, color=%s, ImageIndex: %d" % (x, y,
#                                                                                 str(z), Imageindex)
#                 else:
#                     sentence = "x=%1.4f, y=%1.4f, val=%s, ImageIndex: %d" % (x, y,
#                                                         self.dataarray_info[row, col], Imageindex)
#                 self.SetStatusText(sentence)
#                 return sentence
#             else:
#                 sentence = "x=%1.4f, y=%1.4f" % (x, y)
#                 self.SetStatusText(sentence)
#                 return sentence

#         self.axes.format_coord = format_coord

#         self.canvas.draw()


class ImshowFrame_Scalar(wx.Frame):
    """
    Class to show 2D array intensity data

    used in multigrains.py to plot maps
    """
    def __init__(self, parent, _id, title, dataarray, posarray_twomotors=None,
                                                        posmotorname=(None, None),
                                                        datatype="scalar",
                                                        maptype=None,
                                                        absolutecornerindices=None,
                                                        Imageindices=None,
                                                        absolute_motorposition_unit="micron",
                                                        colorbar_label="Fluo counts",
                                                        nb_row=10,
                                                        nb_lines=10,
                                                        boxsize_row=10,
                                                        boxsize_line=10,
                                                        stepindex=1,
                                                        imagename="",
                                                        mosaic=0,
                                                        extent=None,
                                                        xylabels=None):
        """
        plot 2D plot of dataarray

        posarray_twomotors  =  additional info to show in status bar when hovering on plot
        """

        wx.Frame.__init__(self, parent, _id, title, size=(700, 700))

        self.data = dataarray

        print("self.data.shape", self.data.shape)

        self.dataarray_info = posarray_twomotors
        if posmotorname is not None:
            self.motor1name, self.motor2name = posmotorname

        self.absolute_motorposition_unit = absolute_motorposition_unit
        #         print "dataarray", dataarray
        self.datatype = datatype
        self.maptype = maptype

        self.absolutecornerindices = absolutecornerindices
        self.title = title
        self.Imageindices = Imageindices

        self.colorbar_label = colorbar_label
        self.nb_columns = nb_row
        self.nb_lines = nb_lines
        self.boxsize_row = boxsize_row
        self.boxsize_line = boxsize_line
        self.stepindex = stepindex
        self.extent = extent
        self.xylabels = xylabels
        self.imagename = imagename
        self.mosaic = mosaic
        self.dirname = None
        self.filename = None

        #         print "self.data.shape in ImshowFrame", self.data.shape
        #         print "self.nb_columns, self.nb_lines", self.nb_columns, self.nb_lines
        #         print "self.boxsize_row, self.boxsize_line", self.boxsize_row, self.boxsize_line

        if self.datatype == "scalar":
            self.LastLUT = "gist_earth"
        elif self.datatype == "symetricscalar":
            self.LastLUT = "bwr"

        self.create_main_panel()

        self.clear_axes_create_imshow()

    def create_main_panel(self):
        """
        """
        # ----------------- Set up the MenuBar
        MenuBar = wx.MenuBar()

        FileMenu = wx.Menu()

        OpenMenu = FileMenu.Append(wx.ID_ANY, "&Save", "Save Image")
        self.Bind(wx.EVT_MENU, self.OnSave, OpenMenu)

        # SaveMenu = FileMenu.Append(wx.ID_ANY, "&Save","Save BNA")
        # self.Bind(wx.EVT_MENU, self.SaveBNA, SaveMenu)

        CloseMenu = FileMenu.Append(wx.ID_ANY, "&Close", "Close Application")
        self.Bind(wx.EVT_MENU, self.OnQuit, CloseMenu)

        MenuBar.Append(FileMenu, "&File")

        # view_menu = wx.Menu()
        # ZoomMenu = view_menu.Append(wx.ID_ANY, "Zoom to &Fit","Zoom to fit the window")
        # self.Bind(wx.EVT_MENU, self.ZoomToFit, ZoomMenu)
        # MenuBar.Append(view_menu, "&View")

        help_menu = wx.Menu()
        AboutMenu = help_menu.Append(wx.ID_ANY, "&About", "More information About this program")
        self.Bind(wx.EVT_MENU, self.OnAbout, AboutMenu)
        MenuBar.Append(help_menu, "&Help")

        self.SetMenuBar(MenuBar)
        # -----------------------------------

        # statusbar
        self.stbar = self.CreateStatusBar(3)

        self.stbar.SetStatusWidths([100, -1, -1])

        self.panel = wx.Panel(self)

        self.stbar0 = wx.StatusBar(self.panel)

        self.dpi = 100
        self.figsize = 5
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        #         self.tooltip = wx.ToolTip(tip='tip with a long %s line and a newline\n' % (' ' * 100))
        #         self.canvas.SetToolTip(self.tooltip)
        #         self.tooltip.Enable(False)
        #         self.tooltip.SetDelay(0)
        #         self.fig.canvas.mpl_connect('motion_notify_event', self.onMotion_ToolTip)

        self.toolbar = NavigationToolbar(self.canvas)

        # read data and adapt lut
        self.calc_norm_minmax_values()

        if self.datatype == "symetricscalar":
            self.IminDisplayed = 50
            self.ImaxDisplayed = 100
            labelslide1, labelslide2 = "factor1 :", "factor 2 :"
        else:
            self.IminDisplayed = 0
            self.ImaxDisplayed = 100
            labelslide1, labelslide2 = "Min :", "Max :"

        self.slidertxt_min = wx.StaticText(self.panel, -1, labelslide1)
        self.slider_min = wx.Slider(self.panel,
                                    -1,
                                    size=(200, 50),
                                    value=self.IminDisplayed,
                                    minValue=0,
                                    maxValue=100,
                                    style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_min.SetTickFreq(50)
        else:
            self.slider_min.SetTickFreq(50, 1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMin, self.slider_min)

        self.slidertxt_max = wx.StaticText(self.panel, -1, labelslide2)
        self.slider_max = wx.Slider(self.panel,
                                        -1,
                                        size=(200, 50),
                                        value=self.ImaxDisplayed,
                                        minValue=1,
                                        maxValue=100,
                                        style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_max.SetTickFreq(50)
        else:
            self.slider_max.SetTickFreq(50, 1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMax, self.slider_max)

        if self.datatype in ("symetricscalar", "scalar"):
            # loading LUTS
            self.mapsLUT = [m for m in pcm.datad if not m.endswith("_r")]
            self.mapsLUT.sort()

            luttxt = wx.StaticText(self.panel, -1, "LUT")
            self.comboLUT = wx.ComboBox(self.panel, -1, self.LastLUT, size=(-1, 40),
                                                                        choices=self.mapsLUT)

            self.comboLUT.Bind(wx.EVT_COMBOBOX, self.OnChangeLUT)

            self.scaletype = "Linear"
            scaletxt = wx.StaticText(self.panel, -1, "Scale")
            self.comboscale = wx.ComboBox(self.panel, -1, self.scaletype,
                                                        choices=["Linear", "Log"], size=(-1, 40))

            self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)

            intmintxt = wx.StaticText(self.panel, -1, "Int. Min.")
            self.intmintxtctrl = wx.TextCtrl(self.panel, -1, str(self.minvals))
            intmaxtxt = wx.StaticText(self.panel, -1, "Int. MAX.")
            self.intmaxtxtctrl = wx.TextCtrl(self.panel, -1, str(self.maxvals))

            self.slidersbox = wx.BoxSizer(wx.HORIZONTAL)
            self.slidersbox.Add(self.slidertxt_min, 0)
            self.slidersbox.Add(self.slider_min, 0)
            self.slidersbox.Add(self.slidertxt_max, 0)
            self.slidersbox.Add(self.slider_max, 0)
            self.slidersbox.AddSpacer(5)
            self.slidersbox.Add(luttxt, 0)
            self.slidersbox.Add(self.comboLUT, 0)

            htoolbar = wx.BoxSizer(wx.HORIZONTAL)
            htoolbar.Add(self.toolbar, 0)
            htoolbar.Add(scaletxt, 0)
            htoolbar.Add(self.comboscale, 0)
            htoolbar.Add(intmintxt, 0)
            htoolbar.Add(self.intmintxtctrl, 0)
            htoolbar.Add(intmaxtxt, 0)
            htoolbar.Add(self.intmaxtxtctrl, 0)

            self.vbox = wx.BoxSizer(wx.VERTICAL)
            self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
            self.vbox.Add(self.slidersbox, 0, wx.EXPAND)
            self.vbox.Add(htoolbar, 0, wx.EXPAND)
            self.vbox.Add(self.stbar0, 0, wx.EXPAND)

        elif self.datatype == "RGBvector":
            self.scaletype = None
            self.vbox = wx.BoxSizer(wx.VERTICAL)
            self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
            self.vbox.Add(self.toolbar, 0, wx.EXPAND)
            self.vbox.Add(self.stbar0, 0, wx.EXPAND)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.Layout()

    def OnAbout(self, event):
        pass

    def OnChangeScale(self, _):
        self.scaletype = str(self.comboscale.GetValue())
        if self.scaletype == "Log" and self.minvals < 0:
            self.scaletype = "Linear"
        self.normalizeplot()
        self.canvas.draw()

    def OnChangeLUT(self, _):
        print("OnChangeLUT")
        strcmap = str(self.comboLUT.GetValue())
        self.cmap = mplcm.get_cmap(strcmap)
        self.cmap.set_over("k", self.maxvals)
        self.cmap.set_under("k", self.minvals)
        self.myplot.set_cmap(self.cmap)

        self.canvas.draw()

    def OnSliderMin(self, _):

        self.IminDisplayed = int(self.slider_min.GetValue())
        if self.datatype == "symetricscalar":
            pass
        elif self.IminDisplayed > self.ImaxDisplayed:
            self.slider_min.SetValue(self.ImaxDisplayed - 1)
            self.IminDisplayed = self.ImaxDisplayed - 1

        self.normalizeplot()
        self.canvas.draw()

    def OnSliderMax(self, _):
        self.ImaxDisplayed = int(self.slider_max.GetValue())
        if self.datatype == "symetricscalar":
            pass
        elif self.ImaxDisplayed < self.IminDisplayed:
            self.slider_max.SetValue(self.IminDisplayed + 1)
            self.ImaxDisplayed = self.IminDisplayed + 1
        self.normalizeplot()
        self.canvas.draw()

    def normalizeplot(self):

        if self.datatype == "symetricscalar":
            #             vmin = self.minvals* (1-self.IminDisplayed /100.)
            #             vmax = self.maxvals*(self.ImaxDisplayed/100.)

            vmax = (self.maxvals * (self.ImaxDisplayed / 100.0)
                * (self.IminDisplayed / 100.0))
            vmin = -vmax

        else:

            self.minvals = float(self.intmintxtctrl.GetValue())
            self.maxvals = float(self.intmaxtxtctrl.GetValue())

            self.deltavals = (self.maxvals - self.minvals) / 100.0

            vmin = self.minvals + self.IminDisplayed * self.deltavals
            vmax = self.minvals + self.ImaxDisplayed * self.deltavals

        if self.scaletype == "Linear":

            self.cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.scaletype == "Log":
            self.cNorm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            self.cNorm = None

        #
        #         cond1 = (self.data<self.minvals)
        #         cond2 = (self.data>self.maxvals)
        #         cond = np.logical_and(cond1,cond2)
        #
        #         self.data_to_Display = np.ma.masked_where(cond, self.data)
        #
        #         self.myplot.set_data(self.data_to_Display)

        self.myplot.set_norm(self.cNorm)

    def OnSave(self, _):
        # if self.askUserForFilename(defaultFile='truc', style=wx.SAVE,**self.defaultFileDialogOptions()):
        #    self.OnSave(event)
        if self.askUserForFilename():
            fig = self.plotPanel.get_figure()
            fig.savefig(os.path.join(str(self.dirname), str(self.filename)))
            print("Image saved in ", os.path.join(self.dirname, self.filename) + ".png")

    def askUserForFilename(self, **dialogOptions):
        dialog = wx.FileDialog(self, **dialogOptions)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True
            self.filename = dialog.GetFilename()
            self.dirname = dialog.GetDirectory()
            print(self.filename)
            print(self.dirname)

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

    #     def onMotion_ToolTip(self, event):  # tool tip to show data when mouse hovers on plot
    #
    #         if self.data == None:
    #             return
    #
    #         collisionFound = False
    #
    #         dims, dimf = self.data.shape[:2]
    # #        print "self.data_2D.shape onmotion", self.data_2D.shape
    #         radius = .5
    #         if event.xdata != None and event.ydata != None:  # mouse is inside the axes
    # #            for i in xrange(len(self.dataX)):
    # #                radius = 1
    # #                if abs(event.xdata - self.dataX[i]) < radius and abs(event.ydata - self.dataY[i]) < radius:
    # #                    top = tip = 'x=%f\ny=%f' % (event.xdata, event.ydata)
    # #            for i in xrange(dims * dimf):
    # #                X, Y = self.Xin[0, i % dimf], self.Yin[i % dimf, 0]
    #             rx = int(np.round(event.xdata))
    #             ry = int(np.round(event.ydata))
    #
    #             if abs(rx - (dimf - 1) / 2) <= (dimf - 1) / 2 and abs(ry - (dims - 1) / 2) <= (dims - 1) / 2:
    # #                print X, Y
    # #                print event.xdata, event.ydata
    #
    #                 zvalue = self.data[ry, rx]
    #
    #
    #                 tip = 'X=%d\nY=%d\n(x,y):(%d %d)\nI=%.5f' % (self.center[0] - self.boxsize[0] + rx,
    #                                                           self.center[1] - self.boxsize[1] + ry,
    #                                                           rx, ry, zvalue)
    #
    #
    #                 self.tooltip.SetTip(tip)
    #                 self.tooltip.Enable(True)
    #                 collisionFound = True
    #     #            break
    #                 return
    #         if not collisionFound:
    #             self.tooltip.Enable(False)

    def calc_norm_minmax_values(self):
        #         if self.dataarray_info is not None:
        #             self.maxvals = np.amax(self.dataarray_info)
        #             self.minvals = np.amin(self.dataarray_info)
        #
        #
        # #             print 'self.dataarray_info', self.dataarray_info
        #             print 'self.dataarray_info max ', self.maxvals
        #             print 'self.dataarray_info min ', self.minvals

        self.data_to_Display = self.data

        if self.datatype in ("scalar", "symetricscalar"):
            print("plot of datatype = %s" % self.datatype)

            self.cNorm = None

            self.maxvals = np.amax(self.data)
            self.minvals = np.amin(self.data)

            mini = self.minvals
            maxi = self.maxvals

            # there are negative and positive values close to zero
            if self.datatype == "symetricscalar":
                maxlimits = max(-mini, maxi)
                self.maxvals = maxlimits
                self.minvals = -maxlimits
                self.deltavals = (self.maxvals - self.minvals) / 200.0

            else:
                self.maxvals = maxi
                self.minvals = mini
                self.deltavals = (self.maxvals - self.minvals) / 100.0

            #             from matplotlib.colors import colorConverter
            import matplotlib.colors as colors

            #             import matplotlib.pyplot as plt
            #             import matplotlib.cm as cmx
            #             OrRd = cm = plt.get_cmap('OrRd')
            self.cNorm = colors.Normalize(vmin=self.minvals, vmax=self.maxvals)
        #             scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=OrRd)
        #             print scalarMap.get_clim()
        #         colorVal = scalarMap.to_rgba(values[idx])

        #             cond1 = (self.data<self.minvals)
        #             cond2 = (self.data>self.maxvals)
        #             cond = np.logical_and(cond1,cond2)
        #
        #             self.data_to_Display = np.ma.masked_where(cond, self.data)

        if self.datatype == "RGBvector":
            print("plot of datatype = %s" % self.datatype)

            self.data_to_Display = self.data
            self.cNorm = None

            self.maxvals = np.amax(self.data_to_Display[:, :, 0])
            self.minvals = np.amin(self.data_to_Display[:, :, 0])

    def forceAspect(self, aspect=1.0):
        im = self.axes.get_images()
        extent = im[0].get_extent()
        self.axes.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    def clear_axes_create_imshow(self):
        """
        init the figure
        """

        if self.datatype == "symetricscalar":
            self.cmap = GT.BWR
        else:
            self.cmap = GT.ORRD
        # clear the axes and replot everything
        self.axes.cla()
        self.axes.set_title(self.title)

        self.cmap.set_over("k", self.maxvals)
        self.cmap.set_under("k", self.minvals)

        if self.datatype in ("scalar", "symetricscalar"):
            self.myplot = self.axes.imshow(self.data_to_Display,
                                            cmap=self.cmap,
                                            interpolation="nearest",
                                            norm=self.cNorm,
                                            aspect="equal",
                                            #                              extent=self.extent,
                                            origin="lower")
            #             self.myplot = self.axes.pcolor(self.data_to_Display,
            #                                            cmap=self.cmap,
            #                                         interpolation='nearest', norm=self.cNorm,
            #                                         aspect='equal',
            # #                              extent=self.extent,
            #                             origin='lower')
            self.colorbar = self.fig.colorbar(self.myplot)
            self.colorbar.set_label(self.colorbar_label)

        if self.datatype == "RGBvector":

            #             print "self.data_to_Display",self.data_to_Display
            self.myplot = self.axes.imshow(self.data_to_Display,
                                            interpolation="nearest",
                                            #                          extent=self.extent,
                                            origin="lower")

        posmotors = self.dataarray_info

        if posmotors is not None:

            initmotor1 = posmotors[0, 0, 0]
            initmotor2 = posmotors[0, 0, 1]

            posmotor1 = posmotors[0, :, 0]
            posmotor2 = posmotors[:, 0, 1]

            print("starting motor1 %f %s" % (initmotor1, self.absolute_motorposition_unit))
            print("starting motor2 %f %s" % (initmotor2, self.absolute_motorposition_unit))

            #         print 'posmotor1', posmotor1
            #         print 'posmotor2', posmotor2

            nby, nbx = posmotors.shape[:2]

            print("center motor1", posmotor1[nbx / 2])
            print("center motor2", posmotor2[nby / 2])

            # x= fast motor  (first in spec scan)
            # y slow motor (second in spec scan)
            step_x = (posmotor1[-1] - posmotor1[0]) / (nbx - 1)
            step_y = (posmotor2[-1] - posmotor2[0]) / (nby - 1)

            print("step_x %f %s " % (step_x, self.absolute_motorposition_unit))
            print("step_y %f %s " % (step_y, self.absolute_motorposition_unit))
            #         def fromindex_to_pixelpos_x_mosaic(index, pos):
            #                 return index  # self.center[0]-self.boxsize[0]+index
            #         def fromindex_to_pixelpos_y_mosaic(index, pos):
            #                 return index  # self.center[1]-self.boxsize[1]+index

            step_factor = 1.0
            if self.absolute_motorposition_unit == "mm":
                step_factor = 1000.0
                step_x = step_x * step_factor
                step_y = step_y * step_factor

            def fromindex_to_pixelpos_x_mosaic(index, _):
                return np.fix((step_x * index) * 1000.0) / 1000.0 # self.center[0]-self.boxsize[0]+index

            def fromindex_to_pixelpos_y_mosaic(index, _):
                return np.fix((step_y * index) * 1000.0) / 1000.0  # self.center[1]-self.boxsize[1]+index

            #         def fromindex_to_motor1pos(index, pos):
            #             print posmotor1[int(index)]
            #             return posmotor1[int(index)]
            # #             return fix((index * (posmotor1[-1] - initmotor1) / (nb1 - 1) + \
            # #                                 initmotor1) * 100000) / 100000
            #         def fromindex_to_motor2pos(index, pos):
            #             return posmotor2[int(index)]
            # #             return fix((index * (posmotor2[-1] - initmotor2) / (nb2 - 1) + \
            # #                                 initmotor2) * 100000) / 100000

            self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x_mosaic))
            self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y_mosaic))
            #
            #         self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_motor1pos))
            #         self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_motor2pos))

            self.axes.set_xlabel(self.xylabels[0])
            self.axes.set_ylabel(self.xylabels[1])

            nb_of_microns_x = round((posmotor1[-1] - posmotor1[0])) * step_factor
            nb_of_microns_y = round((posmotor2[-1] - posmotor2[0])) * step_factor

            print("nb_points_y,nb_points_x", nby, nbx)
            print("nb_of_microns_x", nb_of_microns_x)
            print("nb_of_microns_y", nb_of_microns_y)

            #         self.axes.locator_params('x', tight=True, nbins=6)  # round(nb_of_microns_x) + 1)
            #         self.axes.locator_params('y', tight=True, nbins=round(nb_of_microns_y) + 1)

            # fix the ticks distance: either 1 micron or a given length such as 5 ticks are plotted

            tickx_micron_sampling = 1.0 / step_x  # 1 micron
            tickx_sampling_length = max(np.fabs(nb_of_microns_x / 5.0 / step_x), np.fabs(tickx_micron_sampling))
            ticky_micron_sampling = 1.0 / step_y
            ticky_sampling_length = max(np.fabs(nb_of_microns_y / 5.0 / step_y), np.fabs(ticky_micron_sampling))

            print("ticks every : tickx_sampling_length (micron)", tickx_sampling_length)
            print("ticks every : ticky_sampling_length (micron)", ticky_sampling_length)

            locx = pltticker.MultipleLocator(base=np.fabs(tickx_sampling_length))  # this locator puts ticks at regular intervals
            self.axes.xaxis.set_major_locator(locx)
            locy = pltticker.MultipleLocator(base=np.fabs(ticky_sampling_length))  # this locator puts ticks at regular intervals
            self.axes.yaxis.set_major_locator(locy)

        self.axes.grid(True)

        #         numrows, numcols, color = self.data.shape
        numrows, numcols = self.data.shape[:2]

        #         print "self.Imageindices", self.Imageindices
        print("self.Imageindices.shape", self.Imageindices.shape)
        print("self.data.shape", self.data.shape)

        #         imageindices = self.Imageindices[0] + arange(0, numrows, numcols) * self.stepindex

        tabindices = self.Imageindices

        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = self.data[row, col]
                #                 print "z", z
                #                 print "x,y,row,col", x, y, row, col
                Imageindex = tabindices[row, col]
                if posmotors is None:
                    #                     print "self.dataarray_info is None"
                    if self.datatype != "RGBvector":
                        zvalue = z
                        if self.maptype == "orientation":
                            angle = np.arccos(zvalue) * 180.0 / np.pi
                            addedvalue = "angle(deg)=%.3f" % angle
                        else:
                            addedvalue = ""
                        sentence0 = "x=%1.4f, y=%1.4f, value=%s, %s ImageIndex: %d" % ( x, y, zvalue,
                                                                            addedvalue, Imageindex)
                    else:
                        zvalue = str(z)
                        sentence0 = "x=%1.4f, y=%1.4f, value=%s, ImageIndex: %d" % ( x, y, zvalue,
                                                                                        Imageindex)
                    sentence_corner = ""
                    sentence_center = ""
                    sentence = "No motors positions"
                else:
                    #                     print "col=", col
                    #                     print posmotor1[col]
                    sentence_corner = "CORNER =[[%s=%.2f,%s=%.2f]]" % (
                        self.motor1name, (posmotor1[col] - posmotor1[0]) * step_factor,
                        self.motor2name, (posmotor2[row] - posmotor2[0]) * step_factor)
                    sentence_center = "CENTER =[[%s=%.2f,%s=%.2f]]" % (
                        self.motor1name, (posmotor1[col] - posmotor1[nbx / 2]) * step_factor,
                        self.motor2name, (posmotor2[row] - posmotor2[nby / 2]) * step_factor)
                    sentence0 = ("j=%d, i=%d, val = %s, ABSOLUTE=[%s=%.5f,%s=%.5f], ImageIndex: %d"
                        % (col, row, str(z), self.motor1name, posmotor1[col],
                                self.motor2name, posmotor2[row], Imageindex))
                    sentence = "POSITION (micron) from: "

                self.stbar0.SetStatusText(sentence0)
                self.stbar.SetStatusText(sentence)
                self.stbar.SetStatusText(sentence_corner, 1)
                self.stbar.SetStatusText(sentence_center, 2)

                return sentence0
            else:
                print("out of plot")
                return "out of plot"

        self.axes.format_coord = format_coord

        #         if step_y != 0:
        #             self.forceAspect(1.*step_x / step_y)

        self.intmintxtctrl.SetValue(str(self.minvals))
        self.intmaxtxtctrl.SetValue(str(self.maxvals))
        self.canvas.draw()


class ImshowFrame(wx.Frame):
    """
    Class to show 2D array scalar data
    """
    def __init__(self, parent, _id, title, dataarray, absolutecornerindices=None,
                                                    Imageindices=np.arange(2),
                                                    nb_row=10,
                                                    nb_lines=10,
                                                    boxsize_row=10,  # TODO row actually is column
                                                    boxsize_line=10,  # TODO line is row
                                                    stepindex=1,
                                                    imagename="",
                                                    mosaic=1,
                                                    dict_param=None,
                                                    datatype="Intensity"):
        """
        datatype = Intensity, PositionX, PositionY, RadialPosition

        """
        print("\n\n*****\nCREATING PLOT of 2D Map of Scalar data: %s\n****\n"%title)
        # dat=dat.reshape(((self.nb_row)*2*self.boxsize_row,(self.nb_lines)*2*self.boxsize_line))
        self.appFrame = wx.Frame.__init__(self, parent, _id, title, size=(900, 700))

        try:
            self.dict_ROI = parent.dict_ROI
        except AttributeError:
            self.dict_ROI = None
        self.parent = parent
        self.mosaic = mosaic

        self.panel = wx.Panel(self)

        self.createMenu()
        self.sb = self.CreateStatusBar()
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

        self.dataraw = copy(dataarray)
        # data to be displayed
        self.data = dataarray
        self.datatype = datatype

        self.absolutecornerindices = absolutecornerindices
        self.title = title
        self.Imageindices = Imageindices
        self.nb_columns = nb_row
        self.nb_lines = nb_lines
        self.boxsize_row = boxsize_row
        self.boxsize_line = boxsize_line
        print('nb_row  nb_lines', nb_row, nb_lines)
        print('boxsize_row  boxsize_line',boxsize_row, boxsize_line)
        self.stepindex = stepindex
        self.imagename = imagename
        self.currentpointedImageIndex = None

        self.dict_param = copy(dict_param)

        if "dataVector" in dict_param:
            print("BINGO !!")
            self.dataVectorinit = copy(dict_param["dataVector"])
            self.dataVectorinit_shifted = copy(dict_param["dataVector"])

        self.dirname = None
        self.filename = None
        self.originYaxis = "lower"

        self.memorizedxlimits = None
        self.memorizedylimits = None

        print("self.data.shape in ImshowFrame", self.data.shape)
        print("self.nb_columns, self.nb_lines", self.nb_columns, self.nb_lines)
        print("self.boxsize_row, self.boxsize_line", self.boxsize_row, self.boxsize_line)
        #         print 'self.Imageindices', self.Imageindices

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
                self.palette = copy(GT.ORRD)
                self.LastLUT = "OrRd"
            else:
                self.palette = copy(GT.SEISMIC)
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
                                    size=(-1, 40),
                                    choices=self.mapsLUT,
                                    style=wx.TE_PROCESS_ENTER)

        self.comboLUT.Bind(wx.EVT_COMBOBOX, self.OnChangeLUT)
        self.comboLUT.Bind(wx.EVT_TEXT_ENTER, self.OnTypeLUT)

        if self.datatype == "Intensity":
            self.scaletype = "Linear" #self.scaletype = "Log"

        else:
            self.scaletype = "Linear"
        self.scaletxt = wx.StaticText(self.panel, -1, "Scale")
        self.comboscale = wx.ComboBox(self.panel, -1, self.scaletype, choices=["Linear", "Log"],
                                                                                    size=(-1, 40))

        self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)

        self.aspect = "auto"
        self.aspecttxt = wx.StaticText(self.panel, -1, "Aspect Ratio")
        self.comboaspect = wx.ComboBox(self.panel,
                                        -1,
                                        self.aspect,
                                        choices=["equal", "auto"],
                                        size=(-1, 40),
                                        style=wx.TE_PROCESS_ENTER)

        self.comboaspect.Bind(wx.EVT_COMBOBOX, self.OnChangeAspect)
        self.comboaspect.Bind(wx.EVT_TEXT_ENTER, self.OnChangeAspect)

        self.chckgrid = wx.CheckBox(self.panel, -1, "Grid")
        self.chckgrid.SetValue(self.plotgrid)
        self.chckgrid.Bind(wx.EVT_CHECKBOX, self.Oncheckgrid)

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
            hboxmask.Add(self.maskthresholdtxt, 0)

            hboxmask.Add(self.maskthresholdctrl, 0)

        if self.datatype in ("Vector",):
            hboxarrow = wx.BoxSizer(wx.HORIZONTAL)
            hboxarrow.Add(self.slidertxt_arrowwidth, 0)
            hboxarrow.Add(self.slider_arrowwidth, 0)

            hboxarrow.Add(self.slidertxt_arrowscale, 0)
            hboxarrow.Add(self.slider_arrowscale, 0)

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
        if self.datatype in ("Vector",):
            self.vbox.Add(hboxarrow, 0, wx.EXPAND)
        if "FilteredfittedPeaksData" in self.dict_param:
            self.vbox.Add(hboxmask, 0, wx.EXPAND)
        self.panel.SetSizer(self.vbox)

        self.vbox.Fit(self)
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
        https://sourceforge.net/projects/lauetools
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
        info.SetCopyright("(C) 2016 Jean-Sebastien Micha")
        info.SetWebSite("http://www.esrf.eu/UsersAndScience/Experiments/CRG/BM32/")
        info.SetLicence(licence)
        info.AddDeveloper("Jean-Sebastien Micha")
        info.AddDocWriter("Jean-Sebastien Micha")
        # info.AddArtist('The Tango crew')
        info.AddTranslator("Jean-Sebastien Micha")

        wx.AboutBox(info)
        event.Skip()

    def OnSave(self, _):

        dlg = wx.TextEntryDialog(self, "Enter filename for image", "Saving in png format")

        if dlg.ShowModal() == wx.ID_OK:
            filename = str(dlg.GetValue())

            fig = self.plotPanel.get_figure()
            if self.dirname is None:
                self.dirname = os.path.curdir

            if filename.endswith(".png"):
                filename = filename[:-4]

            fullpath = os.path.join(str(self.dirname), str(filename))

            fig.savefig(fullpath)
            print("Image saved in ", fullpath)

        dlg.Destroy()

    def SaveData(self, _):
        wx.MessageBox("To be implemented, but data are automatically saved in the same folder "
                                                                "than the images one", "INFO")

    def askUserForFilename(self, **dialogOptions):
        dialog = wx.FileDialog(self, **dialogOptions)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True
            self.filename = dialog.GetFilename()
            self.dirname = dialog.GetDirectory()
            print(self.filename)
            print(self.dirname)

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
        dataX_2D = xDATA.reshape((n0, n1))
        dataY_2D = yDATA.reshape((n0, n1))
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
        ) = fittedPeaksData.T

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
        fittedPeaksData = np.ma.array([peak_X,
                                        peak_Y,
                                        peak_I,
                                        peak_fwaxmaj,
                                        peak_fwaxmin,
                                        peak_inclination,
                                        Xdev,
                                        Ydev,
                                        peak_bkg,
                                        FilterX]).T

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
        #        print 'clicked on mouse'

        self.centerx, self.centery = event.xdata, event.ydata

        if event.inaxes:
            if event.button in (2, 3):
                self.OnRightButtonMousePressed(1)

        #            print("inaxes", event)
        #             print("inaxes", event.x, event.y)
        #             print("inaxes", event.xdata, event.ydata)
        #             self.centerx, self.centery = event.xdata, event.ydata

        else:
            pass

    #            print("out axes", event)
    #            print("out axes", event.x, event.y)
    #            print("out axes", event.xdata, event.ydata)

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
        #         print "xyzech",xyzech
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
        """
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

        #         print 'relativeindex',relativeImageindex
        absoluteImageIndex = self.Imageindices[relativeImageindex]

        #         print "absoluteImageIndex",absoluteImageIndex
        return absoluteImageIndex

    def format_coord(self, x, y):
        """
        ImshowFrame
        """
        col = int(x + 0.5)
        row = int(y + 0.5)
        numrows, numcols = self.data.shape
        #         print "self.Imageindices", self.Imageindices
        #         print "len()", len(self.Imageindices)
        #         print "to be reshaped", (self.nb_lines, self.nb_columns)

        if len(self.Imageindices) != (self.nb_lines * self.nb_columns):
            print("WARNING:  display may not work , check strictly that ")
            print("the number of images is a multiple of the number of lines !!!")

        #         print "self.tabindices", self.tabindices
        if col >= 0 and col < numcols and row >= 0 and row < numrows:

            #             print "x,y in format_coord",x,y
            # print int(y/(2*self.boxsize_row)),int(x/(2*self.boxsize_line))

            if self.mosaic:
                z = self.data[row, col]
                #                 print "hello in mosaic"
                logz = np.log(z)

                self.currentpointedImageIndex = self.getImageIndexfromxy(x, y)

                #                 print "self.absolutecornerindices",self.absolutecornerindices
                if self.absolutecornerindices is not None:
                    jmin, imin = self.absolutecornerindices
                    #                     xpixel, ypixel = (jmin + x % (2 * self.boxsize_row + 1),
                    #                                       imin + y % (2 * self.boxsize_line + 1))
                    xpixel, ypixel = (jmin + x % (2 * self.boxsize_row + 1),
                        imin + 2 * self.boxsize_line - y % (2 * self.boxsize_line + 1))
                    return "x=%1.4f, y=%1.4f, log(I)=%1.4f, I=%5.1f, ImageIndex: %d" % (
                                        xpixel, ypixel, logz, z, self.currentpointedImageIndex)
                else:
                    return "x=%1.4f, y=%1.4f, log(I)=%1.4f, I=%5.1f, ImageIndex: %d" % ( x, y,
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

                else:
                    z = self.data[row, col]
                    return "x=%1.4f, y=%1.4f, I=%5.5f, ImageIndex: %d" % (x, y, z,
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
        #         im = self.axes.get_images()
        #         extent = im[0].get_extent()
        #         print "extent",extent
        #         xmin,xmax,ymin,ymax=extent
        #
        #         indlb = self.getImageIndexfromxy(xmin, ymin)
        #         indrb = self.getImageIndexfromxy(xmax, ymin)
        #         indlt = self.getImageIndexfromxy(xmin, ymax)
        #         indrt = self.getImageIndexfromxy(xmax, ymax)
        #         print "indlb,indrb,indlt,indrt",indlb,indrb,indlt,indrt

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


class MyCustomToolbar(NavigationToolbar):
    #     ON_YAXIS_UPPER = wx.NewId()
    #     ON_YAXIS_LOWER = wx.NewId()

    ON_YAXIS_UPPER = 7777
    ON_YAXIS_LOWER = 7778

    def __init__(self, canvas, parentframe):
        """
        parentframe must have the attribute : originYaxis  (lower or upper)

        """
        # create the default toolbar
        NavigationToolbar.__init__(self, canvas)

        self.canvas = canvas
        self.parentframe = parentframe
        # add new toolbar buttons
        # remove the unwanted button
        POSITION_OF_CONFIGURE_SUBPLOTS_BTN = 6
        self.DeleteToolByPos(POSITION_OF_CONFIGURE_SUBPLOTS_BTN)

        #         self.AddSimpleTool(self.ON_CUSTOM_LEFT, _load_bitmap('stock_left.xpm'),
        #                            'Pan to the left', 'Pan graph to the left')
        #         wx.EVT_TOOL(self, self.ON_CUSTOM_LEFT, self._on_custom_pan_left)
        #         self.AddSimpleTool(self.ON_CUSTOM_RIGHT, _load_bitmap('stock_right.xpm'),
        #                            'Pan to the right', 'Pan graph to the right')
        #         wx.EVT_TOOL(self, self.ON_CUSTOM_RIGHT, self._on_custom_pan_right)

        image = wx.Bitmap(
            os.path.join(DictLT.LAUETOOLSFOLDER, "icons", "transmissionLauesmall.png"))
        image.SetSize((30, 30))
        #         self.SetToolBitmapSize((30,30))
        if WXPYTHON4:
            t1 = self.AddTool(7777, "Y axis origin upper", image)
            t2 = self.AddTool(7778, "Y axis origin lower", image)

            self.Bind(wx.EVT_TOOL, self.onYaxisUpper, id=7777)
            self.Bind(wx.EVT_TOOL, self.onYaxisLower, id=7778)
        else:
            self.AddSimpleTool(self.ON_YAXIS_UPPER, image, "Y axis origin upper", "Y axis origin upper")

            self.AddSimpleTool(self.ON_YAXIS_LOWER, image, "Y axis origin lower", "Y axis origin lower")
            self.Bind(wx.EVT_TOOL, self.onYaxisUpper, id=self.ON_YAXIS_UPPER)
            self.Bind(wx.EVT_TOOL, self.onYaxisLower, id=self.ON_YAXIS_LOWER)

    def onYaxisUpper(self, evt):

        self.parentframe.originYaxis = "upper"
        self.parentframe._replot()

    def onYaxisLower(self, evt):

        self.parentframe.originYaxis = "lower"
        self.parentframe._replot()


#     # pan the graph to the left
#     def _on_custom_pan_left(self, evt):
#         ONE_SCREEN = 1
#         axes = self.canvas.figure.axes[0]
#         x1, x2 = axes.get_xlim()
#         ONE_SCREEN = x2 - x1
#         axes.set_xlim(x1 - ONE_SCREEN, x2 - ONE_SCREEN)
#         self.canvas.draw()
#
#     # pan the graph to the right
#     def _on_custom_pan_right(self, evt):
#         ONE_SCREEN = 1
#         axes = self.canvas.figure.axes[0]
#         x1, x2 = axes.get_xlim()
#         ONE_SCREEN = x2 - x1
#         axes.set_xlim(x1 + ONE_SCREEN, x2 + ONE_SCREEN)
#         self.canvas.draw()


def rebin(a, *args):
    """
    see http://www.scipy.org/Cookbook/Rebinning
    here mean value of rebinned array

    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:

    Examples
    -----------

    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)

    rebin(a,5,2) means division of slow dimension number of elements by 5 and fast dim number of elements by 2
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(args)
    evList = (["a.reshape("] + ["args[%d],factor[%d]," % (i, i) for i in range(lenShape)]
        + [")"] + [".mean(%d)" % (i + 1) for i in range(lenShape)])
    print("".join(evList))
    return eval("".join(evList))


def myformattime():
    """
    convert secs since the Epoch to my string format
    """
    tt = time.localtime()

    return "date_%04d%02d%02d_%02d%02d%02d" % (tt.tm_year,
                                                tt.tm_mon,
                                                tt.tm_mday,
                                                tt.tm_hour,
                                                tt.tm_min,
                                                tt.tm_sec)


def buildMosaic3(dict_param, outputfolder, ccdlabel="MARCCD165", plot=1, parent=None, verbose=0):
    """
    build mosaic image from arrangement of image ROI data taken from selected images
    """
    CountersData = {}

    #     (dirname, filename, startind, endind, stepind, nb_lines, nb_images_per_line,
    #             xpic, ypic, boxsize_col, boxsize_line, selectedcounters) = dict_param

    dirname = dict_param["imagesfolder"]
    filename_representative = dict_param["filename_representative"]
    ccdlabel = dict_param["CCDLabel"]
    nbdigits = dict_param["nbdigits"]
    #     startind = int(dict_param['starting_imageindex'])
    #     endind = int(dict_param['final_imageindex'])
    selected2Darray_imageindex = dict_param["selected2Darray_imageindex"]
    #     nb_lines, nb_images_per_line = dict_param['nb_lines'], dict_param['nb_images_per_line']

    xpic, ypic = dict_param["pixelX_center"], dict_param["pixelY_center"]
    boxsize_col, boxsize_line = (dict_param["pixelboxsize_X"], dict_param["pixelboxsize_Y"])

    selectedcounters = dict_param["selectedcounters"]
    monitoroffset = dict_param["monitoroffset"]

    print("selectedcounters", selectedcounters)

    halfboxsizes = boxsize_col, boxsize_line

    print("selected2Darray_imageindex in buildMosaic3", selected2Darray_imageindex)

    #    print "boxsize_col, boxsize_line", halfboxsizes
    selected1Darray_absoluteimageindex = np.ravel(selected2Darray_imageindex)
    tabindices = selected1Darray_absoluteimageindex
    startind, endind = (selected1Darray_absoluteimageindex[0], selected1Darray_absoluteimageindex[-1])

    #        nb_image = endind - startind + 1
    datashape = selected2Darray_imageindex.shape
    if len(datashape) == 1:
        nb_col = datashape[0]
        nb_lines = 1
    else:
        nb_col = datashape[1]
        nb_lines = datashape[0]

    print("nb_lines,nb_col", nb_lines, nb_col)

    dict_map_imageindex = {}
    print("selected1Darray_absoluteimageindex", selected1Darray_absoluteimageindex)
    if nb_col > 0:
        for map_imageindex, absolute_imageindex in enumerate(selected1Darray_absoluteimageindex):

            dict_map_imageindex[map_imageindex] = [absolute_imageindex,
                                                    map_imageindex,
                                                    map_imageindex // nb_col,
                                                    map_imageindex % nb_col]
    else:
        print("!***!****!****!")
        print("something wrong")
        print("Cannot build map between sample position and image index")
        print("!***!****!****!")

    boxpixelsize = (2 * halfboxsizes[0] + 1, 2 * halfboxsizes[1] + 1)

    #        print "boxpixelsize", boxpixelsize

    try:
        mosaic = np.zeros((nb_lines, nb_col, boxpixelsize[0], boxpixelsize[1]))
    except MemoryError:
        print("!***!****!****!")
        print("something wrong with pixel center or boxsize")
        print("Cannot Prepare Image")
        print("!***!****!****!")

    monitor = np.zeros((nb_lines, nb_col))

    for map_imageindex, absolute_imageindex in enumerate(selected1Darray_absoluteimageindex):
        imageindex = absolute_imageindex
        filename = IOimage.setfilename(filename_representative,
                                    imageindex,
                                    CCDLabel=ccdlabel,
                                    nbdigits=nbdigits)

        filename = os.path.join(dirname, filename)
        print("filename", filename)

        try:
            framedimraw = DictLT.dict_CCD[ccdlabel][0]
            fliprot = DictLT.dict_CCD[ccdlabel][3]
            if verbose > 0:
                print("framedim of ccdlabel", framedimraw, ccdlabel)
                print("fliprot", fliprot)

            center_pixel = (xpic, ypic)
            if fliprot in ("sCMOS_fliplr",):
                center_pixel = (framedimraw[1] - xpic, ypic)

            if not filename.endswith("tif.gz"):

                indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                            (halfboxsizes[0], halfboxsizes[1]),
                                                            framedimraw,
                                                            flipxycenter=0)
                imin, imax, jmin, jmax = indicesborders

                if verbose > 0: print("indicesborders", indicesborders)

                # avoid to wrong indices when slicing the data
                imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                            framedim=framedimraw)

                if verbose > 0:print("imin, imax, jmin, jmax", imin, imax, jmin, jmax)
                # new fast way to read specific area in file directly
                datacrop = IOimage.readrectangle_in_image(filename,
                                                        xpic,
                                                        ypic,
                                                        halfboxsizes[0],
                                                        halfboxsizes[1],
                                                        dirname=None,
                                                        CCDLabel=ccdlabel)
            else:

                framedim = framedimraw
                indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                            (halfboxsizes[0], halfboxsizes[1]),
                                                            framedimraw,
                                                            flipxycenter=0)
                imin, imax, jmin, jmax = indicesborders

                if verbose > 0: print("indicesborders", indicesborders)

                # avoid to wrong indices when slicing the data
                imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                            framedim=framedimraw)

                if verbose > 0: print("imin, imax, jmin, jmax", imin, imax, jmin, jmax)

                datawhole, fdim, flrot = IOimage.readCCDimage(filename, ccdlabel)
                datacrop = datawhole[imin:imax, jmin:jmax]

            if datacrop is None:
                # go to the next image
                continue

            if dict_param["NormalizeWithMonitor"]:

                if ccdlabel in ("sCMOS", "sCMOS_fliplr"):
                    pedestal = 1000.0
                    dictMonitor = IOimage.read_header_scmos(filename)
                    print("dictMonitor.keys()", list(dictMonitor.keys()))
                    monitor_val = dictMonitor["mon"] - monitoroffset * dictMonitor["exposure"] / 1000.0
                if ccdlabel in ("MARCCD165",):
                    pedestal = 10.0
                    monitor_val = 1.0
                #                     IOimage.read_header_marccd2(filename)
                datcrop = (datacrop - pedestal) / monitor_val

            else:
                datcrop = datacrop
                monitor_val = 1

        # TODO:  test if file exists
        except IOError:
            print("!***!****!****!")
            print("something wrong with image. cannot find %s" % filename)
            print("!***!****!****!")
            continue

        if imageindex % 10 == 1:
            print(filename, "up to", IOimage.setfilename(filename, endind, CCDLabel=ccdlabel))

        kx, ky = dict_map_imageindex[map_imageindex][2:]

        # print("dict_map_imageindex", dict_map_imageindex)
        # print("kx, ky, map_imageindex, imageindex", kx, ky, map_imageindex, imageindex)

        #             mosaic[kx, ky] = datcrop.T
        mosaic[kx, ky] = np.flipud(datcrop).T

        monitor[kx, ky] = monitor_val

    # ----------   end of images scan

    title = "%s [%06d-%06d] " % (filename_representative, startind, endind)
    title += "pixel ROI at [%d,%d]" % (xpic, ypic)

    print("selectedcounters", selectedcounters)

    print("mosaic.shape", mosaic.shape)

    for counter in selectedcounters:
        print("counter", counter)

        if counter in ("mean", "max", "ptp"):
            if counter == "mean":
                dat = np.mean(np.mean(mosaic, axis=2), axis=2)
            elif counter == "max":
                dat = np.amax(np.amax(mosaic, axis=2), axis=2)
            elif counter == "ptp":
                dat = np.amax(np.amax(mosaic, axis=2), axis=2) - np.amin(np.amin(mosaic, axis=2), axis=2)

            #                 print "rawdat",rawdat
            #                 print "rawdat.shape", rawdat.shape

            if dict_param["NormalizeWithMonitor"]:
                if verbose > 0: print("monitor", monitor)
                monitor = np.where(monitor <= 0.0, 1.0, monitor)

            CountersData[counter + "2D"] = dat

            if plot:

                plapla = ImshowFrame(parent, -1, "image Plot %s" % counter, dat,
                                    Imageindices=tabindices,
                                    nb_row=nb_col,
                                    nb_lines=nb_lines,
                                    stepindex=1,
                                    boxsize_row=1,
                                    boxsize_line=1,
                                    imagename=title,
                                    mosaic=0,
                                    dict_param=dict_param)

                #            plapla.dirname = self.dirname

                plapla.Show()

                np.savetxt("%s_2D" % counter + "_%s" % myformattime(), dat)

                parent.list_of_windows.append(plapla)

            nbimages = len(np.arange(startind, endind + 1))

            # Instens monitors selection
            Intens = np.ravel(dat)
            Intens = Intens[:nbimages]

            tabindices1D = tabindices[:nbimages]

            XYdat = [tabindices1D, Intens]

            CountersData[counter + "1D"] = XYdat

            outfilename = os.path.join(outputfolder, "%s" % (counter + "_1D"))

            np.savetxt(outfilename + "_%s" % myformattime(), np.array(XYdat).T)

            if plot:

                plotI = PLOT1D.Plot1DFrame(parent,
                                            -1,
                                            counter + " Intensity",
                                            title + " Intensity",
                                            XYdat,
                                            logscale=0)
                plotI.Show(True)

                parent.list_of_windows.append(plotI)

        elif counter == "mosaic":  # true mosaic

            print("true mosaic of image ROIs")
            #                 title = '%s indexrange [%06d-%06d]' % (counter, startind, endind)

            dat = mosaic.transpose((0, 3, 1, 2))

            #            print "shape dat", shape(dat)

            dat = dat.reshape((nb_lines * (2 * boxsize_line + 1), nb_col * (2 * boxsize_col + 1)))

            CountersData[counter + "2D"] = dat

            if plot:
                ploplo = ImshowFrame(parent, -1, "MOSAIC image Plot", dat,
                                    absolutecornerindices=(jmin, imin),
                                    Imageindices=tabindices,
                                    nb_row=nb_col,
                                    nb_lines=nb_lines,
                                    boxsize_row=boxsize_col,
                                    stepindex=1,
                                    #                                boxsize_row=boxsize_col,
                                    boxsize_line=boxsize_line,
                                    imagename=title,
                                    mosaic=1,
                                    dict_param=dict_param)

                #            ploplo.dirname = self.dirname

                ploplo.Show()

                outfilename = os.path.join(outputfolder, "%s" % ("MOSAIC_image_Plot"))

                np.savetxt(outfilename + "_%s" % myformattime(), dat)

                parent.list_of_windows.append(ploplo)

        elif counter in ("Position XY", "Position MAX", "Position Centroid", "Displacement",
                                                                        "Amplitude", "Shape"):
            print("\n\n\n ***************************  Position XY****************\n\n\n")
            n0, n1, n2, n3 = mosaic.shape

            print("dat.shape", mosaic.shape)

            jj = np.arange(n0 * n1).reshape((n0, n1))
            label = np.repeat(jj, n2 * n3, axis=1).reshape((n0, n1, n2, n3))

            #                print 'label', label
            #
            # max value of background
            datminimum = scind.measurements.maximum(mosaic, label, np.arange(n0 * n1))
            datminimums = np.repeat(datminimum.reshape((n0, n1)), n2 * n3, axis=1).reshape((n0, 
                                                                                    n1, n2, n3))
            print("datminimums", datminimums.shape)
            # center of mass without background removal
            #                 datcenterofmass = array(scind.measurements.center_of_mass(mosaic,
            #                                                                              label,
            #                                                                              arange(n0 * n1)))
            # remove baseline level set to minimum pixel intensity
            datcenterofmass2 = np.array(scind.measurements.center_of_mass(
                    mosaic - datminimums, label, np.arange(n0 * n1)))
            # from relative position to absolute initial CCD position
            centerofmass = datcenterofmass2[:, 2:] + np.array([jmin, imin])

            datmaximumpos = scind.measurements.maximum_position(mosaic, label,
                                                                            np.arange(n0 * n1))
            datmaximumpos = np.array(datmaximumpos, dtype="uint32")
            imageindex = datmaximumpos[:, 0] * n1 + datmaximumpos[:, 1] + startind
            posmax = datmaximumpos[:, 2:] + np.array([jmin, imin])

            #                    datmaximum = scind.measurements.maximum(mosaic, label, arange(n0 * n1))
            #                    print "res maximum", datmaximum
            #                    dat = datmaximum.reshape((n0, n1))

            #                    print "res maximum", datminimum
            #                print "res datcenterofmass", datcenterofmass
            #                print "res datcenterofmass2", datcenterofmass2
            #                print "res datmaximumpos", datmaximumpos
            # #
            #                print "imageindex", imageindex
            #                print 'posmax', posmax

            meanposmax_local = np.mean(datmaximumpos[:, 2:], axis=0)
            meanposmax_global = np.mean(posmax, axis=0)

            #                print "meanposmax_local", meanposmax_local
            #                print "meanposmax_global", meanposmax_global

            relative_posmax = posmax - meanposmax_global

            #                    print "relative_posmax", relative_posmax

            nbimages = len(np.arange(startind, endind + 1))
            # ---------------------------------------------------
            # position monitors selection
            # ---------------------------------------------------
            if counter == "Position MAX":
                XY = posmax
            elif counter == "Position Centroid":
                XY = centerofmass
            # fit 2D pixel intensities array with a 2D shape gaussian function
            else:  # default
                #                 elif counter == 'Position XY':
                XY, FilteredfittedPeaksData = FitPeakOnMap(mosaic,
                                                        (xpic, ypic),
                                                        reject_negative_baseline=True,
                                                        reject_large_PixelDeviation=True,
                                                        reject_weakPeaks=False,
                                                        FitPixelDev=None,
                                                        MinimumPeakAmplitude=25,
                                                        modelFunction="gaussian",
                                                        framedimensions=framedimraw,
                                                        positionStart="max",
                                                        verbose=1)

                dict_param["FilteredfittedPeaksData"] = FilteredfittedPeaksData

            #                     FilteredfittedPeaksData =np.array([peak_X, peak_Y,
            #                                             peak_I, peak_fwaxmaj, peak_fwaxmin,
            #                                             peak_inclination, Xdev, Ydev, peak_bkg, FilterX]
            # 3rd value (gaussian fit) -------------------------------------------

            #                print "XY", XY
            #                print "lenXY", len(XY)

            xDATA, yDATA = XY[:nbimages].T

            tabindices1D = tabindices[:nbimages]

            XYdat_x = [tabindices1D, xDATA]
            XYdat_y = [tabindices1D, yDATA]

            CountersData["posX"] = XYdat_x
            CountersData["posY"] = XYdat_y

            dataX_2D = xDATA.reshape((n0, n1))
            dataY_2D = yDATA.reshape((n0, n1))

            print("type array", type(dataX_2D))

            # plot 1D graph  ( X or Y as fct 1D index in data )
            if plot and counter == "Position XY":

                plotX = PLOT1D.Plot1DFrame(parent,
                                            -1,
                                            counter + "Xpos",
                                            title + "Xpos",
                                            XYdat_x,
                                            logscale=0)
                plotX.Show(True)

                outfilename = os.path.join(outputfolder, "%s" % (counter + "Xpos"))

                np.savetxt(outfilename + "_%s" % myformattime(), np.array(XYdat_x).T)

                parent.list_of_windows.append(plotX)

                plotY = PLOT1D.Plot1DFrame(parent,
                                            -1,
                                            counter + "Ypos",
                                            title + "Ypos",
                                            XYdat_y,
                                            logscale=0)
                plotY.Show(True)

                outfilename = os.path.join(outputfolder, "%s" % (counter + "Ypos"))

                np.savetxt(outfilename + "_%s" % myformattime(), np.array(XYdat_y).T)

                parent.list_of_windows.append(plotY)

            # plot 2D X position
            if plot and counter == "Position XY":
                print("tabindices in plot 2D X", tabindices.shape)
                print("nb_col,nb_lines", nb_col, nb_lines)
                plot2DX = ImshowFrame(parent, -1, "image 2D Plot X %s" % counter,
                                        dataX_2D,
                                        Imageindices=tabindices,
                                        nb_row=nb_col,
                                        nb_lines=nb_lines,
                                        stepindex=1,
                                        boxsize_row=0,
                                        boxsize_line=0,
                                        imagename=title,
                                        mosaic=0,
                                        dict_param=dict_param,
                                        datatype="PositionX")

                #            plapla.dirname = self.dirname

                plot2DX.Show()

                np.savetxt("%s_2D_X_" % counter + "_%s" % myformattime(), dataX_2D)

                parent.list_of_windows.append(plot2DX)

            # plot 2D Y position
            if plot and counter == "Position XY":
                plot2DY = ImshowFrame(parent, -1, "image 2D Plot Y %s" % counter,
                                        dataY_2D,
                                        Imageindices=tabindices,
                                        nb_row=nb_col,
                                        nb_lines=nb_lines,
                                        stepindex=1,
                                        boxsize_row=0,
                                        boxsize_line=0,
                                        imagename=title,
                                        mosaic=0,
                                        dict_param=dict_param,
                                        datatype="PositionY")

                #            plapla.dirname = self.dirname

                plot2DY.Show()

                np.savetxt("%s_2D_Y_" % counter + "_%s" % myformattime(), dataY_2D)

                parent.list_of_windows.append(plot2DY)

            # plot 2D radial distance from mean X and Y  position
            if plot and counter == "Displacement":
                meanX = np.mean(dataX_2D)
                meanY = np.mean(dataY_2D)

                # data for radial distzance plot
                pixelCenter = (meanX, meanY)
                radialdistance_2D = np.sqrt((dataX_2D - pixelCenter[0]) ** 2
                    + (dataY_2D - pixelCenter[1]) ** 2)
                dict_param["dataRadialPosition"] = (dataX_2D, dataY_2D, pixelCenter)

                plot2Dradial = ImshowFrame(parent, -1, "image 2D Plot radial %s" % counter,
                                            radialdistance_2D,
                                            Imageindices=tabindices,
                                            nb_row=nb_col,
                                            nb_lines=nb_lines,
                                            stepindex=1,
                                            boxsize_row=0,
                                            boxsize_line=0,
                                            imagename=title,
                                            mosaic=0,
                                            dict_param=dict_param,
                                            datatype="RadialPosition")

                #            plapla.dirname = self.dirname

                plot2Dradial.Show()

                np.savetxt("%s_2D_radial_" % counter + "_%s" % myformattime(), radialdistance_2D)

                parent.list_of_windows.append(plot2Dradial)

            if plot and counter == "Shape":

                maxpeaksize = np.amax(FilteredfittedPeaksData[:, 3:5], axis=1)

                maxpeaksize2D = maxpeaksize.reshape((n0, n1))

                plot2Dpeaksize = ImshowFrame(parent, -1, "image 2D Plot peak size %s" % counter,
                                            maxpeaksize2D,
                                            Imageindices=tabindices,
                                            nb_row=nb_col,
                                            nb_lines=nb_lines,
                                            stepindex=1,
                                            boxsize_row=0,
                                            boxsize_line=0,
                                            imagename=title,
                                            mosaic=0,
                                            dict_param=dict_param,
                                            datatype="PositionX")

                #            plapla.dirname = self.dirname

                plot2Dpeaksize.Show()

                np.savetxt("%s_2D_size_" % counter + "_%s" % myformattime(), maxpeaksize2D)

                np.savetxt("%s_2Dshape_" % counter + "_%s" % myformattime(), FilteredfittedPeaksData[:, 3:6])

                parent.list_of_windows.append(plot2Dpeaksize)

            # peak amplitude
            if plot and counter == "Amplitude":
                PeakAmplitude = (FilteredfittedPeaksData[:, 2] - FilteredfittedPeaksData[:, 8])
                PeakAmplitude2D = PeakAmplitude.reshape((n0, n1))

                plot2Dpeaksize = ImshowFrame(parent, -1, "image 2D Plot peak amplitude %s" % counter,
                                                PeakAmplitude2D,
                                                Imageindices=tabindices,
                                                nb_row=nb_col,
                                                nb_lines=nb_lines,
                                                stepindex=1,
                                                boxsize_row=0,
                                                boxsize_line=0,
                                                imagename=title,
                                                mosaic=0,
                                                dict_param=dict_param,
                                                datatype="PositionX")

                #            plapla.dirname = self.dirname

                plot2Dpeaksize.Show()

                np.savetxt("%s_2D_Amplitude_" % counter + "_%s" % myformattime(), PeakAmplitude2D)

                parent.list_of_windows.append(plot2Dpeaksize)

                dataamplitude = [tabindices1D, PeakAmplitude]
                plotAmplitude = PLOT1D.Plot1DFrame(parent,
                                                        -1,
                                                        counter + "Amplitude",
                                                        title + "Amplitude",
                                                        dataamplitude,
                                                        logscale=0)
                plotAmplitude.Show(True)

                outfilename = os.path.join(outputfolder, "%s" % (counter + "Amplitude"))

                np.savetxt(outfilename + "_%s" % myformattime(), np.array(dataamplitude).T)

                parent.list_of_windows.append(plotAmplitude)

            # vector quiver plot 2D
            if plot and counter == "Displacement":
                meanX = np.mean(dataX_2D)
                meanY = np.mean(dataY_2D)

                # data for radial distzance plot
                pixelCenter = (meanX, meanY)
                VectorX = dataX_2D - pixelCenter[0]
                VectorY = dataY_2D - pixelCenter[1]
                NormVector = np.hypot(VectorX, VectorY)

                dict_param["dataVector"] = [VectorX, VectorY, pixelCenter]
                datatypevector = "Vector"

                plot2Dpeaksize = ImshowFrame(parent, -1, "image 2D vector %s" % counter,
                                                NormVector,
                                                Imageindices=tabindices,
                                                nb_row=nb_col,
                                                nb_lines=nb_lines,
                                                stepindex=1,
                                                boxsize_row=0,
                                                boxsize_line=0,
                                                imagename=title,
                                                mosaic=0,
                                                dict_param=dict_param,
                                                datatype=datatypevector)

                #            plapla.dirname = self.dirname

                plot2Dpeaksize.Show()

                np.savetxt("%s_2D_quiver_" % counter + "_%s" % myformattime(), maxpeaksize2D)

                parent.list_of_windows.append(plot2Dpeaksize)

    return CountersData




def FitPeakOnMap(mosaic,
                ROIcenter,
                reject_negative_baseline=True,
                reject_large_PixelDeviation=True,
                reject_weakPeaks=True,
                FitPixelDev=None,
                MinimumPeakAmplitude=25,
                modelFunction="gaussian",
                framedimensions=(2048, 2048),
                positionStart="max",
                verbose=1):
    """
    Fit peak on series of 2D pixel intensities array
    """
    n0, n1, n2, n3 = mosaic.shape
    xpic, ypic = ROIcenter

    reject_negative_baseline = 1
    if FitPixelDev is None:
        # limited by boxsize
        FitPixelDev = max(n2, n3)

    FittingParametersDict = {}
    FittingParametersDict["boxsize"] = (n2, n3)
    FittingParametersDict["framedim"] = framedimensions
    FittingParametersDict["saturation_value"] = 65000
    FittingParametersDict["baseline"] = "auto"

    FittingParametersDict["startangles"] = 0
    FittingParametersDict["position_start"] = positionStart
    FittingParametersDict["start_sigma1"] = 2.0
    FittingParametersDict["start_sigma2"] = 2.0

    FittingParametersDict["fitfunction"] = modelFunction
    FittingParametersDict["xtol"] = 0.0001
    FittingParametersDict["offsetposition"] = 1

    centers = xpic * np.ones(n0 * n1), ypic * np.ones(n0 * n1)
    CentralPeaks = np.array(centers).T
    #     print "CentralPeaks",CentralPeaks

    Resfit = RMCCD.fitPeakMultiROIs(mosaic.reshape((n0 * n1, n2, n3)),
                                    CentralPeaks,
                                    FittingParametersDict,
                                    showfitresults=False)
    # filter fit results
    params, cov, info, message, baseline = Resfit

    par = np.array(params)

    peak_bkg = par[:, 0]
    peak_I = par[:, 1]
    peak_X = par[:, 2]
    peak_Y = par[:, 3]
    peak_fwaxmaj = par[:, 4]
    peak_fwaxmin = par[:, 5]
    peak_inclination = par[:, 6] % 360

    # pixel deviations from guessed initial position before fitting
    Xdev = peak_X - CentralPeaks[:, 0]
    Ydev = peak_Y - CentralPeaks[:, 1]
    #     print 'Xdev', Xdev
    #     print "Ydev", Ydev
    #     print "peak_X",peak_X
    #     print "peak_Y",peak_Y

    # --- --- PEAKS REJECTION -------------------------------

    to_reject = []
    k = 0
    for inf in info:
        if inf["nfev"] > 1550:
            if verbose:
                print("k= %d   too much iteration" % k)
            to_reject.append(k)
        k += 1

    #                 if CCDLabel == 'FRELONID15_corrected':
    #                     reject_negative_baseline = False
    if reject_weakPeaks:
        cond = peak_I - peak_bkg < MinimumPeakAmplitude
        to_reject0 = np.where(cond)[0]
        print("to_reject0 peak amplitude < %d" % MinimumPeakAmplitude, to_reject0)
    else:
        to_reject0 = []

    # negative peak width
    if 1:
        cond = np.logical_or(peak_fwaxmaj <= 0, peak_fwaxmin <= 0)
        to_reject1 = np.where(cond)[0]
        print("to_reject1 negative peak width", to_reject1)
    else:
        to_reject1 = []

    # negative intensity rejection
    if reject_negative_baseline:
        to_reject2 = np.where((peak_bkg - baseline) < 0)[0]
        print("to_reject2 negative baseline", to_reject2)
    else:
        to_reject2 = []

    # too far found peak rejection
    if reject_large_PixelDeviation:
        to_reject3 = np.where(np.sqrt(Xdev ** 2 + Ydev ** 2) > FitPixelDev)[0]
        if len(to_reject3) > 0:
            print("to_reject3  too far from initial guess minimum pixel distance = %f"
                % FitPixelDev,
                to_reject3)
    else:
        to_reject3 = []

    ToR = (set(to_reject)
        | set(to_reject0)
        | set(to_reject1)
        | set(to_reject2)
        | set(to_reject3))  # to reject

    print("After fitting, %d/%d peaks have been rejected" % (len(ToR), len(CentralPeaks)))
    print("to reject indices: ", ToR)

    ToTake = set(np.arange(len(CentralPeaks))) - ToR

    print("ToTake", ToTake)

    BoolToTake = False * np.ones(len(CentralPeaks))
    BoolToTake[list(ToTake)] = True

    BoolToMask = True * np.ones(len(CentralPeaks))
    BoolToMask[list(ToTake)] = False

    print("BoolToMask", BoolToMask)

    #     print "where Masked", np.where(BoolToMask==True)
    maskedrows = np.where(BoolToMask == True)[0]

    # all peaks list building
    fittedPeaksData = np.array([peak_X,
                                peak_Y,
                                peak_I,
                                peak_fwaxmaj,
                                peak_fwaxmin,
                                peak_inclination,
                                Xdev,
                                Ydev,
                                peak_bkg,
                                maskedrows]).T
    #                 DEFAULT_WRONG_X = np.nan
    #                 DEFAULT_WRONG_Y = np.nan
    #                 FilterX = np.where(BoolToTake, peak_X, DEFAULT_WRONG_X)
    #                 FilterY = np.where(BoolToTake, peak_Y, DEFAULT_WRONG_Y)

    FilterX0 = np.ma.array(peak_X)
    FilterY0 = np.ma.array(peak_Y)
    FilterX = np.ma.masked_where(BoolToMask, FilterX0)
    FilterY = np.ma.masked_where(BoolToMask, FilterY0)

    # all peaks list building
    fittedPeaksData = np.ma.array([peak_X,
                                    peak_Y,
                                    peak_I,
                                    peak_fwaxmaj,
                                    peak_fwaxmin,
                                    peak_inclination,
                                    Xdev,
                                    Ydev,
                                    peak_bkg,
                                    FilterX]).T
    #                 DEFAULT_WRONG_X = np.nan
    #                 DEFAULT_WRONG_Y = np.nan
    #                 FilterX = np.where(BoolToTake, peak_X, DEFAULT_WRONG_X)
    #                 FilterY = np.where(BoolToTake, peak_Y, DEFAULT_WRONG_Y)

    #     m =False*np.zeros_like(fittedPeaksData)
    #     m[maskedrows,:]=True*ones(10)
    #
    #     print "m",m
    #
    FilteredfittedPeaksData = np.ma.mask_rowcols(fittedPeaksData, axis=0)

    XY = np.ma.array([FilterX, FilterY]).T
    return XY, FilteredfittedPeaksData


def CollectData(param, outputfolder, ccdlabel="MARCCD165"):
    """

    """
    (dirname, filename, startind, endind, nb_lines, nb_images_per_line,
        peaklist, boxsize_row, boxsize_line, selectedcounters) = param

    print("selectedcounters", selectedcounters)

    nbpeaks = len(peaklist)

    print("nbpeaks", nbpeaks)
    print("peaklist", peaklist)

    startind, endind = int(startind), int(endind)
    nb_lines, boxsize_row, boxsize_line = list(map(int, [nb_lines, boxsize_row, boxsize_line]))

    halfboxsize = boxsize_row, boxsize_line

    #    print "boxsize_row, boxsize_line", halfboxsize

    #        nb_image = endind - startind + 1
    nb_col = nb_images_per_line

    dic_carto = {}

    if nb_col > 0:
        for ind in range(startind, endind + 1, 1):
            _ind = ind - startind

            dic_carto[ind] = [_ind, _ind / nb_col, _ind % nb_col]
    else:
        print("!***!****!****!")
        print("something wrong with startind,endind,nb_lines")
        print("Cannot build map between sample position and image index")
        print("!***!****!****!")

    boxpixelsize = (2 * halfboxsize[0] + 1, 2 * halfboxsize[1] + 1)

    #        print "boxpixelsize", boxpixelsize

    try:
        mosaic = np.zeros((nb_lines, nb_col, boxpixelsize[0], boxpixelsize[1]))
    except MemoryError:
        print("!***!****!****!")
        print("something wrong with pixel center or boxsize")
        print("Cannot Prepare Image")
        print("!***!****!****!")

    tabindices = np.arange(startind, endind + 1, 1)
    nbimages = len(tabindices)

    CountersData = {"mean1D": np.zeros((nbpeaks, nbimages)),
                    "max1D": np.zeros((nbpeaks, nbimages)),
                    "ptp1D": np.zeros((nbpeaks, nbimages)),
                    "posX": np.zeros((nbpeaks, nbimages)),
                    "posY": np.zeros((nbpeaks, nbimages))}

    for k, imageindex in enumerate(tabindices):
        filename = IOimage.setfilename(filename, imageindex)

        filename = os.path.join(dirname, filename)
        #            print "filename", filename

        for grain_index, peak in enumerate(peaklist):
            try:
                framedim = DictLT.dict_CCD[ccdlabel][0]
                dataimage, framedim, fliprot = IOimage.readCCDimage(filename, CCDLabel=ccdlabel,
                                                                                dirname=None)

                center_pixel = (round(peak[0]), round(peak[1]))

                indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                            (halfboxsize[0], halfboxsize[1]),
                                                            framedim,
                                                            flipxycenter=0)
                imin, imax, jmin, jmax = indicesborders

                # avoid to wrong indices wHen slicing the data
                imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                            framedim=framedim)

                datcrop = dataimage[imin:imax, jmin:jmax]

            except IOError:
                print("!***!****!****!")
                print("something wrong with image. cannot find %s" % filename)
                print("!***!****!****!")
                break

            if k % 10 == 1:
                print(filename, "up to", IOimage.setfilename(filename, endind))

            piece_dat = datcrop

            title = "indexrange [%06d-%06d]" % (startind, endind)

            print("selectedcounters", selectedcounters)

            for counter in selectedcounters:
                print("counter", counter)

                if counter in ("mean", "max", "ptp"):

                    if counter == "mean":
                        CountersData["mean1D"][grain_index, k] = np.mean(piece_dat)
                    elif counter == "max":
                        CountersData["max1D"][grain_index, k] = np.amax(piece_dat)
                    elif counter == "ptp":
                        CountersData["ptp1D"][grain_index, k] = np.ptp(piece_dat)

                elif counter in ("Position XY",):

                    datminimum = scind.measurements.maximum(piece_dat)
                    # center of mass without background removal
                    datcenterofmass = np.array(scind.measurements.center_of_mass(piece_dat))
                    # remove baseline level set to minimum pixel intensity
                    datcenterofmass2 = np.array(scind.measurements.center_of_mass(piece_dat - datminimum))

                    print("datcenterofmass2", datcenterofmass2)

                    centerofmass = datcenterofmass2 + np.array([jmin, imin])

                    datmaximumpos = scind.measurements.maximum_position(piece_dat)
                    datmaximumpos = np.array(datmaximumpos, dtype="uint32")

                    posmax = datmaximumpos + np.array([jmin, imin])

                    nbimages = endind - startind + 1

                    # position monitors selection
                    XY = posmax
                    XY = centerofmass

                    xDATA, yDATA = XY

                    CountersData["posX"][grain_index, k] = xDATA
                    CountersData["posY"][grain_index, k] = yDATA

    return CountersData




def CollectData_oneImage(param, outputfolder, ccdlabel="MARCCD165",
                selectedcounters=("Imean", "Imax", "Iptp", "posX", "posY"), ndivisions=(1,15)):
    """
    return dictionary of counters

    if selectedcounters have  Imean_multiple, Imax_multiple, Iptp_multiple, then array will subdivided
    according to nbdivsions = (n1,n2) ie n1*n2 subarrays.
    n1 divisions along slow axis (Y, vert), n2 along fast axis (X, horiz)

    param  = (dirname, filename, imageindex, peaklist, boxsize_row, boxsize_line)

    peaklist : list of pixel X pixel Y (horiz, vert)   . For multiple detector : box around [[X, Y]] will be split
    boxsize_row = half boxsize (in pixel) // pixel X axis horiz
    boxsize_line = half boxsize (in pixel) // pixel Y axis vert
    """
    (dirname, filename, imageindex, peaklist, boxsize_row, boxsize_line) = param

    #    print "selectedcounters", selectedcounters

    nbpeaks = len(peaklist)

    print("nbpeaks", nbpeaks)
    #     print 'peaklist', peaklist

    boxsize_row, boxsize_line = list(map(int, [boxsize_row, boxsize_line]))

    halfboxsize = boxsize_row, boxsize_line

    dic_carto = {}

    boxpixelsize = (2 * halfboxsize[0] + 1, 2 * halfboxsize[1] + 1)

    CountersData = {}
    for counter in selectedcounters:
        if 'multiple' not in counter:
            CountersData[counter] = np.zeros(nbpeaks)
        else:
            ny, nx = ndivisions
            CountersData[counter] = np.zeros((nbpeaks, nx*ny))

            CountersData["posmax_multiple"] = np.zeros((nbpeaks, nx*ny, 2))
    CountersData["Monitor"] = 1.0
    CountersData["ExposureTime"] = 1000.0  # milliseconds

    filename = IOimage.setfilename(filename, imageindex)

    filename = os.path.join(dirname, filename)

    for peak_index, peak in enumerate(peaklist):
        try:
            framedim = DictLT.dict_CCD[ccdlabel][0]
            dataimage, framedim, fliprot = IOimage.readCCDimage(filename, CCDLabel=ccdlabel,
                                                                                    dirname=None)
        except IOError:
            print("!***!****!****!")
            print("Missing image file : %s" % filename)
            print("!***!****!****!")
            raise IOError

        if ccdlabel == "MARCCD165":
            comments, expo_time = IOimage.read_header_marccd2(filename)
            I0 = float(comments.split()[3])
            CountersData["Monitor"] = I0
            CountersData["ExposureTime"] = expo_time
        elif ccdlabel == "sCMOS":
            dictpar = IOimage.read_header_scmos(filename)
            CountersData["Monitor"] = dictpar["mon"]
            CountersData["ExposureTime"] = dictpar["exposure"] * 1000  # in milliseconds

        center_pixel = (round(peak[0]), round(peak[1]))

        indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                (halfboxsize[0], halfboxsize[1]),
                                                framedim,
                                                flipxycenter=0)
        imin, imax, jmin, jmax = indicesborders

        # avoid to wrong indices when slicing the data
        imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                                framedim=framedim)

        piece_dat = dataimage[imin:imax, jmin:jmax]

        if 'multiple' in counter:  #split array into several subarrays

            piece_dat, _, (box1, box2) = GT.splitarray(piece_dat, ndivisions)

        for counter in selectedcounters:

            if counter.startswith("I"):
                if 'multiple' not in counter:
                    if counter == "Imean":
                        CountersData["Imean"][peak_index] = np.mean(piece_dat)
                    elif counter == "Imax":
                        CountersData["Imax"][peak_index] = np.amax(piece_dat)
                    elif counter == "Iptp":
                        CountersData["Iptp"][peak_index] = np.ptp(piece_dat)
                else:
                    if counter == "Imean_multiple":
                        CountersData["Imean_multiple"][peak_index] = np.mean(piece_dat, axis=(1, 2))
                    elif counter == "Imax_multiple":
                        CountersData["Imax_multiple"][peak_index] = np.amax(piece_dat, axis=(1, 2))
                    elif counter == "Iptp_multiple":
                        CountersData["Iptp_multiple"][peak_index] = np.ptp(piece_dat, axis=(1, 2))

            elif counter.startswith("pos"):

                if 'multiple' in counter:
                    if counter == "posmax_multiple":
                        nbrois = ndivisions[0] * ndivisions[1]
                        labels = np.repeat(np.arange(nbrois), box1 * box2).reshape((nbrois, box1, box2))
                        #print('labels.shape',labels.shape)
                        index = np.arange(nbrois)
                        datmaximumpos = np.array(scind.measurements.maximum_position(piece_dat, labels=labels, index=index))
                        CountersData["posmax_multiple"][peak_index] = datmaximumpos[:, 1:]

                else:
                    datminimum = scind.measurements.maximum(piece_dat)
                    # center of mass without background removal
                    datcenterofmass = np.array(scind.measurements.center_of_mass(piece_dat))
                    # remove baseline level set to minimum pixel intensity
                    datcenterofmass2 = np.array(scind.measurements.center_of_mass(piece_dat - datminimum))

                    centerofmass = datcenterofmass2 + np.array([jmin, imin])

                    datmaximumpos = scind.measurements.maximum_position(piece_dat)
                    datmaximumpos = np.array(datmaximumpos, dtype="uint32")

                    posmax = datmaximumpos + np.array([jmin, imin])

                    # position monitors selection
                    XY = posmax
                    XY = centerofmass

                    xDATA, yDATA = XY

                    CountersData["posX"][peak_index] = xDATA
                    CountersData["posY"][peak_index] = yDATA

    return CountersData


import math


def floatRgb(mag, cmin, cmax):
    """
    Return a tuple of floats between 0 and 1 for the red, green and
    blue amplitudes.
    """

    try:
        # normalize to [0,1]
        x = float(mag - cmin) / float(cmax - cmin)
    except:
        # cmax = cmin
        x = 0.5
    blue = min((max((4 * (0.75 - x), 0.0)), 1.0))
    red = min((max((4 * (x - 0.25), 0.0)), 1.0))
    green = min((max((4 * np.fabs(x - 0.5) - 1.0, 0.0)), 1.0))
    return (red, green, blue)


def strRgb(mag, cmin, cmax):
    """
    Return a tuple of strings to be used in Tk plots.
    """

    red, green, blue = floatRgb(mag, cmin, cmax)
    return "#%02x%02x%02x" % (red * 255, green * 255, blue * 255)


def rgb(mag, cmin, cmax):
    """
    Return a tuple of integers to be used in AWT/Java plots.
    """

    red, green, blue = floatRgb(mag, cmin, cmax)
    return (int(red * 255), int(green * 255), int(blue * 255))


def htmlRgb(mag, cmin, cmax):
    """
    Return a tuple of strings to be used in HTML documents.
    """
    return "#%02x%02x%02x" % rgb(mag, cmin, cmax)

   
