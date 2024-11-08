# --- ------------  IMAGE VIEWER and PEAK SEARCH Tools GUI
import os
import sys
import time
import copy

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
from matplotlib.artist import Artist


import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                NavigationToolbar2WxAgg as NavigationToolbar)

WXPYTHON3 = False
if wx.__version__ >= "3.0.0":
    WXPYTHON3 = True
    from matplotlib.widgets import RectangleSelector

try:
    from ObjectListView2 import ObjectListView, ColumnDefn #, GroupListView

    ObjectListView_Present = True
except ImportError:
    ObjectListView_Present = False

if not ObjectListView_Present:
    try:
        from ObjectListView import ObjectListView, ColumnDefn #, GroupListView
        ObjectListView_Present = True
    except ImportError:
        print("ObjectListView is missing! You may want to have it: pip install ObjectListView")
        ObjectListView_Present = False

# LaueTools modules
if sys.version_info.major == 3:
    from .. import dragpoints as DGP
    from . import mosaic as MOS
    from .. import SpotModel
    from .. import dict_LaueTools as DictLT
    from .. import readmccd as RMCCD
    from .. import fit2Dintensity as fit2d
    from .. import LaueGeometry as F2TC
    # from . import peaklistfit2d as plf2d
    from . import Plot1DFrame as PLOT1D
    from . import HistogramPlot as HISTOPLOT
    from . import ImshowFrame as IMSHOW
    from .. import generaltools as GT
    from . import CCDFileParametersGUI as CCDParamGUI
    from .. import IOLaueTools as IOLT
    from . import PeaksListBoard
    from .. import IOimagefile as IOimage
    from .. import imageprocessing as ImProc
else:
    import dragpoints as DGP
    import GUI.mosaic as MOS
    import SpotModel
    import dict_LaueTools as DictLT
    import readmccd as RMCCD
    import fit2Dintensity as fit2d
    import LaueGeometry as F2TC
    # import peaklistfit2d as plf2d
    import GUI.Plot1DFrame as PLOT1D
    import GUI.HistogramPlot as HISTOPLOT
    import GUI.ImshowFrame as IMSHOW
    import generaltools as GT
    import GUI.CCDFileParametersGUI as CCDParamGUI
    import IOLaueTools as IOLT
    import GUI.PeaksListBoard
    import IOimagefile as IOimage
    import imageprocessing as ImProc

 
class ViewColorPanel(wx.Panel):
    """class to play with color LUT and intensity scale
    """
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()  # layout2()

        self.xc, self.yc = None, None
        self.indexcirclespatchlist = None
        self.drg = None
        self.DetFilename = None

        self.plotlineprofileframe = None
        self.plotlineXprofileframe = None
        self.plotlineYprofileframe = None
        self.x0, self.y0, self.x2, self.y2 = 200, 200, 1800, 1800

        # widgets
        luttxt = wx.StaticText(self, -1, "LUT")
        self.comboLUT = wx.ComboBox(self, -1, self.mainframe.LastLUT,
                                                    choices=self.mainframe.mapsLUT)

        self.comboLUT.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeLUT)

        showhisto_btn = wx.Button(self, -1, "Intensity Distribution")
        showhisto_btn.Bind(wx.EVT_BUTTON, self.mainframe.ShowHisto)

        self.slider_label = wx.StaticText(self, -1, "Imin: ")
        self.vminminctrl = wx.SpinCtrl(self, -1, str(self.mainframe.vminmin), #size=(110, -1),
                                                                        min=-1000,
                                                                        max=1000)
        self.vminminctrl.Bind(wx.EVT_SPINCTRL, self.mainframe.OnSpinCtrl_IminDisplayed)

        self.vmiddlectrl = wx.SpinCtrl(self, -1, str(self.mainframe.vmiddle), #size=(110, -1),
                                                                        min=-1000000,
                                                                        max=10000)
        self.vmiddlectrl.Bind(wx.EVT_SPINCTRL, self.mainframe.OnSpinCtrl_IminDisplayed)

        # second horizontal band
        self.slider_label2 = wx.StaticText(self, -1, "Imax: ")

        self.vmaxmaxctrl = wx.SpinCtrl(self, -1, str(self.mainframe.vmaxmax), min=-1000, max=10000000)
        self.vmaxmaxctrl.Bind(wx.EVT_SPINCTRL, self.mainframe.OnSpinCtrl_IminDisplayed)

        self.slider_vmin = wx.Slider(self, -1, size=(220, -1),
                            value=int(self.mainframe.vmin),
                            minValue=int(self.mainframe.vminmin),
                            maxValue=int(self.mainframe.vmiddle),
                            style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_vmin.SetTickFreq(500)
        else:
            self.slider_vmin.SetTickFreq(500, 1)
        self.slider_vmin.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK,
                                                        self.mainframe.on_slider_IminDisplayed)

        # second horizontal band
        self.slider_vmax = wx.Slider(self, -1, size=(220, -1),
                                                        value=int(self.mainframe.vmax),
                                                        minValue=int(self.mainframe.vmiddle),
                                                        maxValue=int(self.mainframe.vmaxmax),
                                                        style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_vmax.SetTickFreq(500)
        else:
            self.slider_vmax.SetTickFreq(500, 1)
        self.slider_vmax.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK,
                                                            self.mainframe.on_slider_ImaxDisplayed)

        self.Iminvaltxt = wx.StaticText(self, -1, str(self.mainframe.vmin))
        self.Imaxvaltxt = wx.StaticText(self, -1, str(self.mainframe.vmax))

        self.lineXYprofilechck = wx.CheckBox(self, -1, "Enable X Y profiler")
        InitStateXYProfile = False
        self.lineXYprofilechck.SetValue(InitStateXYProfile)
        self.lineXYprofilechck.Bind(wx.EVT_CHECKBOX, self.OnTogglelineXYprofiles)
        self.plotlineXprofile = InitStateXYProfile
        self.plotlineYprofile = InitStateXYProfile

        self.lineprof_btn = wx.Button(self, -1, "Open LineProfiler", (130, -1))
        self.lineprof_btn.Bind(wx.EVT_BUTTON, self.OnShowLineProfiler)

        savefig_btn = wx.Button(self, -1, "SaveFig", (80, 100))
        savefig_btn.Bind(wx.EVT_BUTTON, self.mainframe.OnSaveFigure)

        replot_btn = wx.Button(self, -1, "Replot", (80, 100))
        replot_btn.SetFont(wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD))
        replot_btn.Bind(wx.EVT_BUTTON, self.OnReplot)

        self.show2thetachi = wx.CheckBox(self, -1, "Show 2theta Chi")
        self.detfiletxtctrl = wx.TextCtrl(self, -1, "", size=(100, -1))
        self.opendetfilebtn = wx.Button(self, -1, "...", size=(100, -1))
        self.show2thetachi.SetValue(False)
        self.opendetfilebtn.Bind(wx.EVT_BUTTON, self.onOpenDetFile)

        # layout
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(luttxt, 0, wx.EXPAND, 5)
        hbox1.Add(self.comboLUT, 0, wx.EXPAND, 5)
        hbox1.Add(showhisto_btn)

        hboxmin = wx.BoxSizer(wx.HORIZONTAL)
        hboxmin.Add(self.slider_label, 0, wx.ALL, 5)
        hboxmin.Add(self.Iminvaltxt, 0, wx.ALL, 5)
        hboxmin.Add(self.slider_vmin, 0, wx.ALL, 5)

        hboxmax = wx.BoxSizer(wx.HORIZONTAL)
        hboxmax.Add(self.slider_label2, 0, wx.ALL, 5)
        hboxmax.Add(self.Imaxvaltxt, 0, wx.ALL, 5)
        hboxmax.Add(self.slider_vmax, 0, wx.ALL, 5)

        hboxprofs = wx.BoxSizer(wx.HORIZONTAL)
        hboxprofs.Add(self.lineXYprofilechck, 0, wx.EXPAND, 10)
        hboxprofs.Add(self.lineprof_btn)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.show2thetachi, 0, wx.EXPAND, 10)
        hbox2.Add(self.detfiletxtctrl, 0, wx.EXPAND, 10)
        hbox2.Add(self.opendetfilebtn)

        hboxbtn = wx.BoxSizer(wx.HORIZONTAL)
        hboxbtn.Add(savefig_btn, 0, wx.EXPAND, 10)
        hboxbtn.Add(replot_btn, 0, wx.EXPAND|wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(hbox1, 0, wx.EXPAND, 10)
        vbox.Add(self.vminminctrl, 0)
        vbox.Add(hboxmin, 0, wx.EXPAND, 5)
        vbox.Add(self.vmiddlectrl, 0)
        vbox.Add(hboxmax, 0, wx.EXPAND, 10)
        vbox.Add(self.vmaxmaxctrl, 0)
        vbox.Add(hboxprofs, 0, wx.EXPAND, 10)
        vbox.Add(hbox2, 0, wx.EXPAND, 10)
        vbox.Add(hboxbtn, 0, wx.EXPAND, 10)

        self.SetSizer(vbox)

        # tooltip
        tp1 = "selection of various intensity mapping color Looking_up table (LUT)"

        luttxt.SetToolTipString("selection of various intensity mappingcolor Looking_up table (LUT)")
        self.comboLUT.SetToolTipString(tp1)

        showhisto_btn.SetToolTipString("Show histogram of pixel intensity distribution of raw image")

        tp2a = "color scale minimum value"
        tp2 = "color scale: minimum value of minimum slider value"
        tp2b = "color scale maximum value"
        tp2c = "color scale: maximum value of maximum slider value"

        self.slider_label.SetToolTipString(tp2a)
        self.vminminctrl.SetToolTipString(tp2)
        self.slider_label2.SetToolTipString(tp2b)
        self.vmaxmaxctrl.SetToolTipString(tp2c)
        self.vmiddlectrl.SetToolTipString("color scale: maximum value of minimum slider value and also minimum value of maximum slider value")

        tp3 = "minimum slider: fine color scale intensity minimum"
        self.slider_vmin.SetToolTipString(tp3)
        tp4 = "maximum slider: fine color scale intensity maximum"
        self.slider_vmax.SetToolTipString(tp4)

        tpmin = "current color scale intensity minimum"
        tpmax = "current color scale intensity maximum"

        self.Iminvaltxt.SetToolTipString(tpmin)
        self.Imaxvaltxt.SetToolTipString(tpmax)

        savefig_btn.SetToolTipString("Save current plot figure in png format file")

        replot_btn.SetToolTipString("Reset and replot displayed image in plot")

        tiplineprof = "Open/Close Draggable Line Pixel intensity Profiler.\n"
        tiplineprof += "Press left mouse button to move line.\n"
        tiplineprof += ("Press right mouse button to change line length (or, press + or -).\n")

        self.lineprof_btn.SetToolTipString(tiplineprof)

        self.lineXYprofilechck.SetToolTipString("Open X and Y pixel intensity line cross sections")
        tipst2 = "Show in status bar the 2 scattering angles. It needs a calibration file .det"
        self.show2thetachi.SetToolTipString(tipst2)

    def OnReplot(self, _):
        """trigger main self.mainframe.OnReplot(1)
        """

        self.mainframe.OnReplot(1)

    def showprofiles(self, event):

        self.updateLineProfile()
        #         print 'event update', event
        if self.plotlineXprofile and self.plotlineYprofile:
            if isinstance(event, mpl.backend_bases.MouseEvent):
                self.OnShowLineXYProfiler(event)

    def OnTogglelineXYprofiles(self, evt):
        self.plotlineXprofile = not self.plotlineXprofile
        self.plotlineYprofile = not self.plotlineYprofile

        self.showprofiles(evt)

    def getlineXYprofiledata(self, event):
        ax = self.mainframe.axes

        xmin, xmax = ax.get_xlim()
        # warning of ymax and ymin swapping!
        ymax, ymin = ax.get_ylim()

        xmin, xmax, ymin, ymax = [int(val) for val in (xmin, xmax, ymin, ymax)]

        xmin, xmax, ymin, ymax = self.restrictxylimits_to_imagearray(xmin, xmax, ymin, ymax)

        #         print "xmin, xmax, ymin, ymax", xmin, xmax, ymin, ymax

        if self.mainframe.dataimage_ROI_display is not None:
            # Extract the values along the line
            z = self.mainframe.dataimage_ROI_display
        else:
            z = self.mainframe.dataimage_ROI

        xyc = self.getclickposition(event)
        if xyc:
            self.xc, self.yc = xyc

        x = np.arange(xmin, xmax + 1)
        y = np.arange(ymin, ymax + 1)
        zx = z[int(np.round(self.yc)), xmin : xmax + 1]
        zy = z[ymin : ymax + 1, int(np.round(self.xc))]

        return x, y, zx, zy

    def getclickposition(self, event):
        """return closest pixel integer coordinates to clicked pt
        """
        if not isinstance(event, mpl.backend_bases.MouseEvent):
            return None

        return int(np.round(event.xdata)), int(np.round(event.ydata))

    def OnShowLineXYProfiler(self, event):
        """ show line X and Y profilers
        """
        LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT = 900, 300
        print("self.plotlineXprofile and self.plotlineYprofile == ",
            self.plotlineXprofile and self.plotlineYprofile)

        if self.plotlineXprofile and self.plotlineYprofile:

            if (self.plotlineXprofileframe is None and self.plotlineYprofileframe is None):

                x, y, zx, zy = self.getlineXYprofiledata(event)

                xp, yp = self.bestposition(LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT)

                # -- Plot lineprofile...
                self.plotlineXprofileframe = PLOT1D.Plot1DFrame(self, -1, "Intensity profile X",
                                                    "",
                                                    [x, zx],
                                                    logscale=0,
                                                    figsize=(8, 3),
                                                    dpi=100,
                                                    size=(LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT))
                self.plotlineXprofileframe.SetPosition((xp, yp))
                self.plotlineXprofileframe.Show(True)

                self.plotlineYprofileframe = PLOT1D.Plot1DFrame(self, -1, "Intensity profile Y",
                                                    "",
                                                    [y, zy],
                                                    logscale=0,
                                                    figsize=(8, 3),
                                                    dpi=100,
                                                    size=(LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT))
                self.plotlineYprofileframe.SetPosition((xp - 200, yp))
                self.plotlineYprofileframe.Show(True)

            if (self.plotlineXprofileframe is not None or self.plotlineYprofileframe is not None):
                self.updateLineXYProfile(event)

    def bestposition(self, LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT):
        """ return xp, yp (best position) """
        # screen size
        dws, dhs = wx.DisplaySize()
        # peaksearchframe size
        wf, hf = self.mainframe.GetSize()
        print("wf, hf", wf, hf)
        print("dws, dhs", dws, dhs)

        if dws > wf:
            Xwindow = dws - LINEPROFILE_WIDTH
        if dhs > hf:
            Ywindow = dhs - LINEPROFILE_HEIGHT

        xp, yp = max(Xwindow, 0), max(Ywindow, 0)
        print("xp,yp", xp, yp)

        return xp, yp

    def restrictxylimits_to_imagearray(self, xmin, ymin, xmax, ymax):
        """return compatible extremal value of x,y
        xmin, ymin, xmax, ymax
        """
        dim = self.mainframe.framedim

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, dim[1] - 1)
        ymax = min(ymax, dim[0] - 1)

        return xmin, ymin, xmax, ymax

    def getlineprofiledata(self, x0, y0, x2, y2):
        """
        get pixel intensities array between two points (x0,y0) and (x2,y2)

        return dataX, dataY
        """
        x0, y0, x2, y2 = self.restrictxylimits_to_imagearray(x0, y0, x2, y2)

        length = int(np.hypot(x2 - x0, y2 - y0))
        x1 = (x2 + x0) / 2.0
        y1 = (y2 + y0) / 2.0

        x, y = np.linspace(x0, x2, length), np.linspace(y0, y2, length)

        if self.mainframe.dataimage_ROI_display is not None:
            # Extract the values along the line
            z = self.mainframe.dataimage_ROI_display
        else:
            z = self.mainframe.dataimage_ROI

        # transpose data
        i_ind = y.astype(np.int16)
        j_ind = x.astype(np.int16)

        zi = z[i_ind, j_ind]

        if x0 != x2:
            signabscissa = np.sign(x - x1)
        else:
            signabscissa = np.sign(y - y1)

        dataX = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) * signabscissa
        dataY = zi

        return dataX, dataY

    def OnShowLineProfiler(self, _):
        """
        show a movable and draggable line profiler and draw line and circle
        """
        if self.plotlineprofileframe is None:
            self.plotlineXprofile = False
            self.plotlineYprofile = False
            self.lineXYprofilechck.SetValue(False)

            self.lineprof_btn.SetLabel("Close LineProfiler")

            # -- Plot lineprofile...
            x, zi = self.getlineprofiledata(self.x0, self.y0, self.x2, self.y2)

            self.plotlineprofileframe = PLOT1D.Plot1DFrame(self, -1, "Intensity profile", "",
                                                            [x, zi], logscale=0, figsize=(8, 3))
            self.plotlineprofileframe.Show(True)

            # plot line on canvas (CCD image)

            pt1 = [self.x0, self.y0]
            pt2 = [self.x2, self.y2]
            ptcenter = DGP.center_pts(pt1, pt2)

            circles = [Circle(pt1, 50, fill=True, color="r", alpha=0.5),
                        Circle(ptcenter, 50, fc="r", alpha=0.5),
                        Circle(pt2, 50, fill=True, color="r", alpha=0.5)]

            line, = self.mainframe.axes.plot([pt1[0], ptcenter[0], pt2[0]],
                                            [pt1[1], ptcenter[1], pt2[1]],
                                            picker=1,
                                            c="r")
            self.indexcirclespatchlist = []
            init_patches_nb = len(self.mainframe.axes.patches)
            for k, circ in enumerate(circles):
                self.mainframe.axes.add_patch(circ)
                self.indexcirclespatchlist.append(init_patches_nb + k)
            #             print "building draggable line"
            #             print "peak search canvas", self.mainframe.canvas
            self.drg = DGP.DraggableLine(circles, line, tolerance=200, parent=self.mainframe,
                                            framedim=self.mainframe.framedim, datatype="pixels")

            self.mainframe.axes.set_xlim(0, 2047)
            self.mainframe.axes.set_ylim(2047, 0)
            self.mainframe.canvas.draw()

        else:
            self.lineprof_btn.SetLabel("Open LineProfiler")

            self.drg.connectingline.set_data([0], [0])  # empty line

            # warning patches list can contain circle marker from peak search
            #             print "self.mainframe.axes.patches", self.mainframe.axes.patches
            for k in range(len(self.indexcirclespatchlist)):
                del self.mainframe.axes.patches[self.indexcirclespatchlist[0]]

            dim = self.mainframe.framedim

            self.mainframe.axes.set_xlim(0, dim[1] - 1)
            self.mainframe.axes.set_ylim(dim[0] - 1, 0)
            self.mainframe.canvas.draw()
            self.drg = None
            self.plotlineprofileframe.Destroy()
            self.plotlineprofileframe = None

        return

    def updateLineXYProfile(self, event):
        """recompute line section intensity profile
        horizontal (x, zx)
        vertical (y, zy)
        """
        if self.plotlineXprofileframe is not None:
            x, y, zx, zy = self.getlineXYprofiledata(event)

            if len(x) != len(zx) or len(y) != len(zy):
                print("STRANGE")
                return

            xyc = self.getclickposition(event)
            if xyc:
                self.xc, self.yc = xyc

            lineXprofileframe = self.plotlineXprofileframe

            lineXprofileframe.line.set_data(x, zx)
            lineXprofileframe.axes.set_title("%s\n@y=%s" % (self.mainframe.imagefilename, self.yc))
            lineXprofileframe.axes.relim()
            lineXprofileframe.axes.autoscale_view(True, True, True)
            lineXprofileframe.canvas.draw()

        if self.plotlineYprofileframe is not None:
            x, y, zx, zy = self.getlineXYprofiledata(event)

            if len(x) != len(zx) or len(y) != len(zy):
                print("STRANGE")
                return

            xyc = self.getclickposition(event)
            if xyc:
                self.xc, self.yc = xyc

            lineYprofileframe = self.plotlineYprofileframe

            lineYprofileframe.line.set_data(y, zy)
            lineYprofileframe.axes.set_title("%s\n@x=%s" % (self.mainframe.imagefilename, self.xc))
            lineYprofileframe.axes.relim()
            lineYprofileframe.axes.autoscale_view(True, True, True)
            lineYprofileframe.canvas.draw()

    def updateLineProfile(self):
        # print("updateLineProfile")
        if self.plotlineprofileframe is not None:

            x, zi = self.getlineprofiledata(self.x0, self.y0, self.x2, self.y2)
            # print("x", x)
            # print("xmin=", min(x))
            # print("xmax=", max(x))
            # print("Imin=", min(zi))
            # print("Imax=", max(zi))
            lineprofileframe = self.plotlineprofileframe

            lineprofileframe.line.set_data(x, zi)
            lineprofileframe.axes.set_title("%s\nline [%d,%d]-[%d,%d]"
                % (self.mainframe.imagefilename, self.x0, self.y0, self.x2, self.y2))
            lineprofileframe.axes.relim()
            lineprofileframe.axes.autoscale_view(True, True, True)
            lineprofileframe.canvas.draw()

    def onOpenDetFile(self, _):
        """open and read .det file with geometry detection calibration parameters

        set self.DetFilename
        set self.mainframe.DetFilename
        """
        print("onOpenDetFile")
        self.mainframe.ReadDetFile(1)
        self.DetFilename = self.mainframe.DetFilename
        self.detfiletxtctrl.SetValue(self.DetFilename)

    def showImage(self):
        """
        branching from button of ViewColorPanel class: show blur/raw image
        """
        print("entering showImage() of ViewColorPanel class with Imagetype %s"
                                                % self.mainframe.ImageFilterpanel.ImageType)

        # display raw after having displayed blur image
        if self.mainframe.ImageFilterpanel.ImageType == "Blur":

            self.mainframe.ImageFilterpanel.ShowblurImagebtn.SetLabel("Show Raw Image")
            # if auto_backgroiund image is already there
            if self.mainframe.ImageFilterpanel.blurimage is not None:
                self.mainframe.dataimage_ROI_display = self.mainframe.ImageFilterpanel.blurimage
                self.mainframe.Show_Image(1, datatype="Blur Image")
            # if auto_backgroiund image is missing, compute it!
            else:
                self.mainframe.ImageFilterpanel.onComputeBlurImage(1)

        # display blur after having displayed raw image
        elif self.mainframe.ImageFilterpanel.ImageType == "Raw":
            self.mainframe.ImageFilterpanel.ShowblurImagebtn.SetLabel("Show Blur Image")
            # if substract blur as bck is checked
            if self.mainframe.ImageFilterpanel.FilterImage.GetValue():
                if self.mainframe.ImageFilterpanel.filteredimage is None:
                    self.mainframe.ImageFilterpanel.Computefilteredimage()

                self.mainframe.dataimage_ROI_display = self.mainframe.ImageFilterpanel.filteredimage
                self.mainframe.Show_Image(1, datatype="Raw Image - Background")
            # raw image
            else:
                self.mainframe.dataimage_ROI_display = self.mainframe.dataimage_ROI
                self.mainframe.Show_Image(1, datatype="Raw Image")


class FilterBackGroundPanel(wx.Panel):
    """class to handle image background tools
    """
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.mainframe = parent.GetParent().GetParent()  # layout2()

        self.BImageFilename = self.mainframe.BImageFilename
        self.BlackListedPeaks = None
        self.BlackListFilename = None
        self.blurimage = None
        self.filteredimage = None
        self.KERNELSIZE = 5
        self.ImageType_index = 0
        self.ImageTypes = ["Raw", "Blur"]
        self.ImageType = self.ImageTypes[self.ImageType_index]

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        sb = wx.StaticBox(self, label="Image Background")
        #sb.SetFont(font3)

        # print("mainframe of FilterBackGroundPanel", self.mainframe)
        # widgets -------------------------
        self.ComputeBlurredImage = wx.Button(sb, -1, "Blur Image")
        self.ShowblurImagebtn = wx.ToggleButton(sb, -1, "Show Blur Image")
        self.SaveBlurredImage = wx.Button(sb, -1, "Save Blur Image",)
        self.FilterImage = wx.CheckBox(sb, -1, "Substract blur as background")

        self.FilterImage.Bind(wx.EVT_CHECKBOX, self.OnSwitchFilterRawImage)
        self.FilterImage.SetValue(False)

        self.ComputeBlurredImage.Bind(wx.EVT_BUTTON, self.onComputeBlurImage)
        self.ShowblurImagebtn.Bind(wx.EVT_TOGGLEBUTTON, self.OnSwitchBlurRawImage)
        self.SaveBlurredImage.Bind(wx.EVT_BUTTON, self.onSaveBlurImage)

        sb2 = wx.StaticBox(self, label="Image Arithmetics")
        #sb2.SetFont(font3)

        self.UseImage = wx.CheckBox(sb2, -1, "Update A = f(A,B) ")
        self.Bequal = wx.StaticText(sb2, -1, "B = ")
        self.imageBctrl = wx.TextCtrl(sb2, -1, self.mainframe.BImageFilename, (400, -1))
        self.openImagebtn = wx.Button(sb2, -1, "...")
        self.UseImage.SetValue(False)
        self.openImagebtn.Bind(wx.EVT_BUTTON, self.onGetBImagefilename)
        self.UseImage.Bind(wx.EVT_CHECKBOX, self.OnChangeUseFormula)

        self.usealsoforfit = wx.CheckBox(sb2, -1, "use also for fit")
        self.usealsoforfit.SetValue(True)

        txtform = wx.StaticText(sb2, -1, "in formula (A: current image)")
        self.formulatxtctrl = wx.TextCtrl(sb2, -1, "A-1.1*B", (150, -1))

        btnsaveformularesult = wx.Button(sb2, -1, "Save result")

        btnsaveformularesult.Bind(wx.EVT_BUTTON, self.onSaveFormulaResultImage)

        sb3 = wx.StaticBox(self, label='Filter Peaks')
        self.RemoveBlackpeaks = wx.CheckBox(sb3, -1, "Remove BlackListed Peaks")
        self.BlackListtoltxt = wx.StaticText(sb3, -1, "Max. Distance")
        self.BlackListRejection_pixeldistanceMax = wx.SpinCtrl(sb3, -1, "15", #size=(150, -1),
                                                               min=1, max=9999)
        self.BlackListedPeaks = wx.TextCtrl(sb3, -1, "", (220, -1))
        self.openBlackListFile = wx.Button(sb3, -1, "...")

        self.RemoveBlackpeaks.SetValue(False)
        self.openBlackListFile.Bind(wx.EVT_BUTTON, self.onGetBlackListfilename)
        # layout

        v1box = wx.StaticBoxSizer(sb, wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.ComputeBlurredImage, 0, wx.EXPAND, 10)
        hbox1.Add(self.ShowblurImagebtn, 0, wx.EXPAND|wx.ALL, 5)
        hbox1.Add(self.SaveBlurredImage, 0, wx.EXPAND|wx.ALL, 5)

        v1box.Add(hbox1, 0, wx.EXPAND, 5)
        v1box.Add(self.FilterImage, 0, wx.EXPAND, 5)

        v2box = wx.StaticBoxSizer(sb2, wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.UseImage, 0, wx.EXPAND, 10)
        hbox2.Add(self.usealsoforfit, 0, wx.EXPAND|wx.ALL, 5)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.Bequal, 0, wx.EXPAND, 10)
        hbox3.Add(self.imageBctrl, 0, wx.EXPAND|wx.ALL, 5)
        hbox3.Add(self.openImagebtn, 0, wx.EXPAND|wx.ALL, 5)

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        hbox4.Add(txtform, 0, wx.EXPAND, 10)
        hbox4.Add(self.formulatxtctrl, 0, wx.EXPAND|wx.ALL, 5)
        hbox4.Add(btnsaveformularesult, 0, wx.EXPAND|wx.ALL, 5)

        v2box.Add(hbox2, 0, wx.EXPAND, 5)
        v2box.Add(hbox3, 0, wx.EXPAND, 5)
        v2box.Add(hbox4, 0, wx.EXPAND, 5)

        v3box = wx.StaticBoxSizer(sb3, wx.VERTICAL)
        hbox5= wx.BoxSizer(wx.HORIZONTAL)
        hbox5.Add(self.RemoveBlackpeaks, 0,wx.EXPAND, 10)
        hbox5.Add(self.BlackListtoltxt, 0, wx.EXPAND|wx.ALL, 5)
        hbox5.Add(self.BlackListRejection_pixeldistanceMax, 0, wx.EXPAND|wx.ALL, 5)

        hbox6= wx.BoxSizer(wx.HORIZONTAL)
        hbox6.Add(self.BlackListedPeaks, 0, wx.EXPAND, 10)
        hbox6.Add(self.openBlackListFile, 0, wx.EXPAND|wx.ALL, 5)

        v3box.Add(hbox5, 0, wx.EXPAND, 5)
        v3box.Add(hbox6, 0, wx.EXPAND, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(v1box, 0, wx.EXPAND, 10)
        vbox.AddSpacer(10)
        vbox.Add(v2box, 0, wx.EXPAND, 10)
        vbox.AddSpacer(10)
        vbox.Add(v3box, 0, wx.EXPAND, 10)

        self.SetSizer(vbox)

        # tooltip
        self.RemoveBlackpeaks.SetToolTipString("Peaks from current PeakSearch belonging to the "
                                                            "blackList will be removed")
        self.ShowblurImagebtn.SetToolTipString("Toggle button to show either blurred or raw image")
        self.ComputeBlurredImage.SetToolTipString("Apply a blur filter to the raw image")
        self.UseImage.SetToolTipString("Compute a new image according to the formula with A and B "
                                                            "resp. the raw and other input image ")
        self.openBlackListFile.SetToolTipString("Browse a File containing a list of blacklisted peaks")
        self.openImagebtn.SetToolTipString("Browse a image File as B (with the same format of "
                                                                                "the raw image A)")
        self.imageBctrl.SetToolTipString("Current B image path")
        self.BlackListedPeaks.SetToolTipString("Current blacklisted peaks File path")

        self.FilterImage.SetToolTipString("set raw image = raw image - blur image)")

        self.usealsoforfit.SetToolTipString("Use resulting image to refine peak position and shape")

        self.formulatxtctrl.SetToolTipString("Mathematical expression to compute a new A as f(A,B) "
                        "for local Maxima search\n(to have initial peak position guesses for fit)")
        btnsaveformularesult.SetToolTipString("Save current image (resulting from mathematical "
                                                                "operation according to formula)")
        self.SaveBlurredImage.SetToolTipString("Save blur image.")

        tpbl = "Maximum pixel distance between peaks in blacklisted peaks list and current peaks "
        "(found by peak search) to be rejected"
        self.BlackListtoltxt.SetToolTipString(tpbl)
        self.BlackListRejection_pixeldistanceMax.SetToolTipString(tpbl)

    def OnSwitchFilterRawImage(self, _):
        if self.ImageType != "Raw":
            return

        if self.blurimage is not None:
            self.mainframe.viewingLUTpanel.showImage()
        else:
            wx.MessageBox("You need to compute first a blur image!", "INFO")

    def Computefilteredimage(self):
        """
        remove background if self.blurimage exists
        """
        if self.blurimage is not None:
            print("ok, I have got self.blurimage")
            print("Computing self.filteredimage")
            CCDlabel = self.mainframe.CCDlabel

            self.filteredimage = ImProc.computefilteredimage(self.mainframe.dataimage_ROI,
                                                            self.blurimage, CCDlabel, kernelsize=5)
        else:
            print("self.blurimage is None !!")

    def onComputeBlurImage(self, _):
        """ Compute background, blurred, filtered or low frequency spatial image
        from current image

        set self.blurimage
        """
        CCDlabel = self.mainframe.CCDlabel

        print('min of self.mainframe.dataimage_ROI in onComputeBlurImage',np.amin(self.mainframe.dataimage_ROI))

        self.blurimage = ImProc.compute_autobackground_image(self.mainframe.dataimage_ROI,
                                                            boxsizefilter=10)

        print("self.blurimage.shape", self.blurimage.shape)

        if self.blurimage is None:
            wx.MessageBox("Binning and Image dimensions are not compatible", "ERROR")
            return

        self.mainframe.viewingLUTpanel.showImage()

    def OnSwitchBlurRawImage(self, _):
        """set viewing of raw image (- background) or background (=filtered image)
        """
        self.ImageType_index += 1
        self.ImageType_index = self.ImageType_index % 2
        self.ImageType = self.ImageTypes[self.ImageType_index]
        self.mainframe.viewingLUTpanel.showImage()

    def onSaveBlurImage(self, _):
        """save on hard disk blurred or background image obtained from current image
        """
        filename = self.mainframe.imagefilename
        dirname = self.mainframe.dirname
        OUTPUTFILENAME_BLURIMAGE = "blur_" + filename

        _header = IOimage.readheader(os.path.join(dirname, filename))

        fullpathname = os.path.join(dirname, OUTPUTFILENAME_BLURIMAGE)

        RMCCD.writeimage(fullpathname, _header, np.ravel(self.blurimage))

        wx.MessageBox("Blurred Image written in %s" % fullpathname, "INFO")

    def onSaveFormulaResultImage(self, _):
        """save image on hard disk of data obtained by arithmetical formula
        """
        filename = self.mainframe.imagefilename
        dirname = self.mainframe.dirname
        OUTPUTFILENAME_RESULTIMAGE = "result_" + filename

        CCDlabel = self.mainframe.CCDlabel

        _header = IOimage.readheader(os.path.join(dirname, filename), CCDLabel=CCDlabel)

        fullpathname = os.path.join(dirname, OUTPUTFILENAME_RESULTIMAGE)

        RMCCD.writeimage(fullpathname, _header, np.ravel(self.mainframe.dataimage_ROI_display))

        wx.MessageBox("Blurred Image written in %s" % fullpathname, "INFO")

    def onGetBImagefilename(self, _):
        """open image as B image
        set self.BImageFilename
        """
        self.mainframe.onOpenBImage(1)
        self.BImageFilename = self.mainframe.BImageFilename
        self.imageBctrl.SetValue(self.mainframe.BImageFilename)

    def onGetBlackListfilename(self, _):
        """ open black peakslist file
        set self.BlackListFilename
        set self.mainframe.BlackListFilename
        """
        self.mainframe.onOpenBlackListFile(1)
        self.BlackListFilename = self.mainframe.BlackListFilename
        self.BlackListedPeaks.SetValue(self.BlackListFilename)

    def OnChangeUseFormula(self, evt):
        """change arithmetical formula
        """
        print("OnChangeUseFormula")
        # use image and formula
        if self.UseImage.GetValue():
            if self.BImageFilename != "":
                self.mainframe.dataimage_ROI_display = self.mainframe.OnUseFormula(1)
                print("new value for image (dataimage_ROI_display)")
                self.mainframe.ConvolvedData = None
            else:
                wx.MessageBox("missing image B to used in formula! Select one, please!", "Error")
                self.UseImage.SetValue(False)
        # not use image and formula
        else:
            self.mainframe.dataimage_ROI_display = self.mainframe.dataimage_ROI

        if self.mainframe.page3.TogglebtnState == 1:  # show convolved
            self.mainframe.Show_ConvolvedImage(evt)
        elif self.mainframe.page3.TogglebtnState == 0:  # show raw image
            print("show image")
            self.mainframe.Show_Image(1)


class BrowseCropPanel(wx.Panel):
    """class to handle crop operation on images"""
    def __init__(self, parent):
        """init"""
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()  # layout2()

        self.stepindex = 10
        startimageindex = self.mainframe.imageindex
        self.imageindexmax = 1000

        print('startimageindex', startimageindex)
        inivalX = int(startimageindex % self.stepindex)
        inivalY = int(startimageindex // self.stepindex)
        # print("inivalX", inivalX)
        # print("inivalY", inivalY)

        # widgets -----------------------
        self.toggleBtnCrop = wx.Button(self, -1, "CropData")
        self.toggleBtnCrop.Bind(wx.EVT_BUTTON, self.mainframe.onToggleCrop)
        self.boxsizetxt = wx.StaticText(self, -1, "boxsize:")
        self.boxxtxt = wx.StaticText(self, -1, "X")
        self.boxxctrl = wx.SpinCtrl(self, -1, "10", #size=(100,-1),
                                    min=0, max=9999)
        #        self.Bind(wx.EVT_SPINCTRL, self.OnBoxSizes, self.boxxctrl)
        self.boxytxt = wx.StaticText(self, -1, "Y")
        self.boxyctrl = wx.SpinCtrl(self, -1, "10", #size=(100,-1),
                                    min=0, max=9999)
        #        self.Bind(wx.EVT_SPINCTRL, self.OnBoxSizes, self.boxyctrl)

        plusbtn = wx.Button(self, -1, "index +1")
        plusbtn.Bind(wx.EVT_BUTTON, self.mainframe.OnPlus)
        minusbtn = wx.Button(self, -1, "index -1")
        minusbtn.Bind(wx.EVT_BUTTON, self.mainframe.OnMinus)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.mainframe.update, self.timer)
        self.toggleBtn = wx.Button(self, wx.ID_ANY, "Auto index+1")
        self.toggleBtn.Bind(wx.EVT_BUTTON, self.mainframe.onToggle)

        self.stepctrl = wx.SpinCtrl(self, -1, "%d" % self.stepindex, #size=(100,-1),
                                    min=2, max=9999)
        self.stepctrl.Bind(wx.EVT_SPINCTRL, self.mainframe.OnStepChange)

        imagemintxt = wx.StaticText(self, -1, "Min: ")
        imagemaxtxt = wx.StaticText(self, -1, "Max: ")
        steptxt = wx.StaticText(self, -1, "Nb images/line: ")
        self.imagemintxtctrl = wx.TextCtrl(self, -1, "0", size=(100, -1), style=wx.TE_PROCESS_ENTER)
        self.imagemaxtxtctrl = wx.TextCtrl(self, -1,
        str(self.imageindexmax), size=(100, -1), style=wx.TE_PROCESS_ENTER)
        self.imagemintxtctrl.Bind(wx.EVT_TEXT_ENTER, self.mainframe.OnChangeImageMin)
        self.imagemaxtxtctrl.Bind(wx.EVT_TEXT_ENTER, self.mainframe.OnChangeImageMax)

        self.slider_image = wx.Slider(self, -1, size=(250, -1), value=inivalX, minValue=0,
                                        maxValue=self.stepindex - 1,
                                        style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)

        self.slider_imagevert = wx.Slider(self, -1, size=(-1, 180), value=inivalY, minValue=0,
                        maxValue=self.imageindexmax // self.stepindex,
                        style=wx.SL_AUTOTICKS | wx.SL_VERTICAL | wx.SL_INVERSE)  # | wx.SL_LABELS)

        self.slider_image.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK,
                                                        self.mainframe.onChangeIndex_slider_image)

        self.slider_imagevert.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK,
                                                    self.mainframe.onChangeIndex_slider_imagevert)

        self.txtnbdigits = wx.StaticText(self, -1, "    Nb of digits\n     in ImageFilename")
        self.nbdigitsctrl = wx.TextCtrl(self, -1, "4")

        self.largeplusbtn = wx.Button(self, -1, "index +%d" % self.stepindex)
        self.largeplusbtn.Bind(wx.EVT_BUTTON, self.mainframe.OnLargePlus)
        self.largeminusbtn = wx.Button(self, -1, "index -%d" % self.stepindex)
        self.largeminusbtn.Bind(wx.EVT_BUTTON, self.mainframe.OnLargeMinus)

        gotobutton = wx.Button(self, -1, "Go to index")
        gotobutton.Bind(wx.EVT_BUTTON, self.mainframe.OnGoto)
        self.fileindexctrl = wx.TextCtrl(self, -1, str(startimageindex), style=wx.TE_PROCESS_ENTER)
        self.fileindexctrl.Bind(wx.EVT_TEXT_ENTER, self.mainframe.OnGoto)

        imagepropstxt = wx.StaticText(self, -1, "Image indices properties (2D map arrangement)")
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        imagepropstxt.SetFont(font3)

        # layout ------------------------
        self.NavigBoxsizer0 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer0.Add(self.toggleBtnCrop, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxsizetxt, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxxtxt, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxxctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxytxt, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxyctrl, 0, wx.ALL, 5)

        self.NavigBoxsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer.Add(minusbtn, 0, wx.ALL, 5)
        self.NavigBoxsizer.Add(plusbtn, 0, wx.ALL, 5)
        self.NavigBoxsizer.Add(wx.StaticText(self, -1, "     "), 0, wx.ALL, 5)
        self.NavigBoxsizer.Add(self.txtnbdigits, 0, wx.ALL, 5)

        self.NavigBoxsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer2.Add(self.largeminusbtn, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.largeplusbtn, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(wx.StaticText(self, -1, "         "), 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.nbdigitsctrl, 0, wx.ALL, 5)

        self.NavigBoxsizer3 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer3.Add(gotobutton, 0, wx.ALL, 5)
        self.NavigBoxsizer3.Add(self.fileindexctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer3.Add(wx.StaticText(self, -1, "     "), 0, wx.ALL, 5)
        self.NavigBoxsizer3.Add(self.toggleBtn, 0, wx.ALL, 5)

        self.NavigBoxsizer4 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer4.Add(imagemintxt, 0)
        self.NavigBoxsizer4.Add(self.imagemintxtctrl, 0)
        self.NavigBoxsizer4.Add(steptxt, 0)
        self.NavigBoxsizer4.Add(self.stepctrl, 0)
        self.NavigBoxsizer4.Add(imagemaxtxt, 0)
        self.NavigBoxsizer4.Add(self.imagemaxtxtctrl, 0)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.NavigBoxsizer0, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer2, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer3, 0, wx.EXPAND)
        vbox.Add(imagepropstxt, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer4, 0, wx.EXPAND)
        vbox.Add(self.slider_image, 0, wx.EXPAND)

        vboxslider = wx.BoxSizer(wx.VERTICAL)
        vboxslider.Add(self.slider_imagevert, 0, wx.EXPAND)

        hhbox = wx.BoxSizer(wx.HORIZONTAL)
        hhbox.Add(vbox, 0)
        hhbox.Add(vboxslider, 0)

        self.SetSizer(hhbox)

        # tooltips -----------------------------------------
        self.toggleBtnCrop.SetToolTipString("Enable/disable crop of image according to a "
        "Region of interest centered on pixel clicked by user with size defined by boxsize")
        self.boxsizetxt.SetToolTipString("X and Y pixel HALF size of the box for cropping image")

        tpx = "Half Size of the crop box in X direction"
        tpy = "Half Size of the crop box in Y direction"
        self.boxxtxt.SetToolTipString(tpx)
        self.boxxctrl.SetToolTipString(tpx)

        self.boxytxt.SetToolTipString(tpy)
        self.boxyctrl.SetToolTipString(tpy)

        plusbtn.SetToolTipString("Increase image filename by 1. In a raster or line scan it "
        "enables to display to the next image")
        minusbtn.SetToolTipString("Decrease image filename by 1. In a raster or line scan it "
        "enables to display to the previous image")

        self.largeplusbtn.SetToolTipString('Increase image filename by "step index". '
        'If "step index" is the number of images per line in a raster scan, it enables to display '
        'the image directly belonging to next line')
        self.largeplusbtn.SetToolTipString(
            'Decrease image filename by "step index". If "step index" is the number of images per '
            'line in a raster scan, it enables to display the image directly '
            'belonging to previous line')

        gotobutton.SetToolTipString("Display image with corresponding index")
        self.fileindexctrl.SetToolTipString("Filename index of the image to display")

        tpstep = "Number of images per line for a raster scan (step index) enabling to display "
        "images collected just above or below the sample position of the current "
        "image.(If the number is enrered, press enter to update the buttons label)."
        steptxt.SetToolTipString(tpstep)
        self.stepctrl.SetToolTipString(tpstep)

        self.toggleBtn.SetToolTipString("Display automatically the next image with filename "
        "index +1 and keep waiting for the image")

        tpnbdigits = "Minimal Number of digits to encode the integer image index. For instance, if index is encoded with at least 4 digits, with zero padding, set to 4. (default value) "
        "Set to None if image index exceeds 9999. Image filename should contain a "
        "single _ character, e.g. myimage_1234.tif"
        self.txtnbdigits.SetToolTipString(tpnbdigits)
        self.nbdigitsctrl.SetToolTipString(tpnbdigits)


class MosaicAndMonitor(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.dict_ROI = {"None": (5, 2, 2)}

        self.cselected = ["mosaic"]

        self.list_of_windows = []

        self.mainframe = parent.GetParent().GetParent()  # layout2()

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        txt1 = wx.StaticText(self, -1, "Image pixel Single ROI selection")
        txt1.SetFont(font3)

        self.boxsizetxt = wx.StaticText(self, -1, "boxsize:")
        self.boxxtxt = wx.StaticText(self, -1, "X")
        self.boxxctrl = wx.SpinCtrl(self, -1, "10", min=0, max=9999)

        self.boxytxt = wx.StaticText(self, -1, "Y")
        self.boxyctrl = wx.SpinCtrl(self, -1, "10", min=0, max=9999)

        self.btnclearwindows = wx.Button(self, wx.ID_ANY, "Clear Windows")
        self.btnclearwindows.Bind(wx.EVT_BUTTON, self.OnClearChildWindows)

        txt2 = wx.StaticText(self, -1, "Image file indices selection")
        txt2.SetFont(font3)

        self.generalindexradiobtn = wx.RadioButton(self, -1, "-->")
        self.generalindexradiobtn.SetValue(True)

        self.startindex = wx.StaticText(self, -1, "Start ")
        self.startindexctrl = wx.SpinCtrl(self, -1, "0", min=0, max=9999)
        #        self.Bind(wx.EVT_SPINCTRL, self.OnBoxSizes, self.boxxctrl)
        self.lastindex = wx.StaticText(self, -1, "Last")
        self.lastindexctrl = wx.SpinCtrl(self, -1, "1", min=0, max=9999)

        self.stepimageindex = wx.StaticText(self, -1, "Step")
        self.stepimageindexctrl = wx.SpinCtrl(self, -1, "1", min=0, max=9999)

        self.rectangleindexradiobtn = wx.RadioButton(self, -1, "-->")

        self.txtimagecenter = wx.StaticText(self, -1, "Center")
        self.centerindexctrl = wx.SpinCtrl(self, -1, "0", min=0, max=9999)

        self.txtimagefastindexbox = wx.StaticText(self, -1, "Boxsize (X)")
        self.txtimagefastindexboxctrl = wx.SpinCtrl(self, -1, "1", min=1, max=9999)

        self.txtimageslowindexbox = wx.StaticText(self, -1, "(Y)")
        self.txtimageslowindexboxctrl = wx.SpinCtrl(self, -1, "1", min=1, max=9999)

        self.predefinedROIradiobtn = wx.RadioButton(self, -1, "-->")

        self.twtROI = wx.StaticText(self, -1, "User-defined ROI")

        self.comboROI = wx.ComboBox(self, -1, "None", size=(-1, 40),
                                        choices=list(self.dict_ROI.keys()))

        self.comboROI.Bind(wx.EVT_COMBOBOX, self.OnChangeROI)

        self.txtnbdigits = wx.StaticText(self, -1, "#digits")
        self.nbdigitsctrl = wx.TextCtrl(self, -1, "4")

        txt3 = wx.StaticText(self, -1, "Counters & Monitors selection")
        txt3.SetFont(font3)

        self.mosaiccounter = wx.CheckBox(self, -1, "Mosaic")
        self.meancounter = wx.CheckBox(self, -1, "Mean Value")
        self.maxcounter = wx.CheckBox(self, -1, "Max Value")
        self.peaktopeakcounter = wx.CheckBox(self, -1, "Peak to Peak")
        self.xyposcounter = wx.CheckBox(self, -1, "Peak Position")  # from fit with 2D gaussuan
        self.amplitudecounter = wx.CheckBox(self, -1, "Peak Amplitude") # from fit with 2D gaussuan
        self.relativexyposcounter = wx.CheckBox(self, -1, "Peak Displacement") # from fit with 2D gaussuan
        self.peaksizecounter = wx.CheckBox(self, -1, "Peak Size") # from fit with 2D gaussuan

        self.normalizechck = wx.CheckBox(self, -1, "Norm. to Monitor")
        self.monitoroffsetctrl = wx.TextCtrl(self, -1, "0")

        self.mosaiccounter.SetValue(True)
        self.normalizechck.SetValue(True)

        txt4 = wx.StaticText(self, -1, "Map Properties")
        txt4.SetFont(font3)
        self.txtnbimagesperline = wx.StaticText(self, -1, "Nb images per line")
        self.stepindex = 10
        self.stepctrl = wx.TextCtrl(self, -1, "%d" % self.stepindex)

        self.txtmapstartingindex = wx.StaticText(self, -1, "Starting index")
        self.mapstartingimageindexctrl = wx.TextCtrl(self, -1, "0")

        self.btnMosaic = wx.Button(self, wx.ID_ANY, "Start")
        self.btnMosaic.Bind(wx.EVT_BUTTON, self.OnMosaic)

        #layout
        self.NavigBoxsizer0 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer0.Add(self.boxsizetxt, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxxtxt, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxxctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxytxt, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.boxyctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer0.Add(self.btnclearwindows, 0, wx.ALL, 5)

        self.NavigBoxsizer = wx.FlexGridSizer(3, 3, 0, 0)
        self.NavigBoxsizer.SetFlexibleDirection(wx.HORIZONTAL)
        self.NavigBoxsizer.Add(self.mosaiccounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.meancounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.maxcounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.peaktopeakcounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.xyposcounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.amplitudecounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.relativexyposcounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(self.peaksizecounter, 0, wx.ALL, 2)
        self.NavigBoxsizer.Add(wx.StaticText(self, -1, ""), 0, wx.ALL, 2)

        NavigBoxsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        NavigBoxsizer1.Add(self.normalizechck, 0, wx.ALL, 2)
        NavigBoxsizer1.Add(self.monitoroffsetctrl, 0, wx.ALL, 2)

        self.NavigBoxsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer2.Add(self.generalindexradiobtn, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.startindex, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.startindexctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.lastindex, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.lastindexctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.stepimageindex, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.stepimageindexctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.txtnbdigits, 0, wx.ALL, 5)
        self.NavigBoxsizer2.Add(self.nbdigitsctrl, 0, wx.ALL, 5)

        NavigBoxsizer2b = wx.BoxSizer(wx.HORIZONTAL)
        NavigBoxsizer2b.Add(self.rectangleindexradiobtn, 0, wx.ALL, 5)
        NavigBoxsizer2b.Add(self.txtimagecenter, 0, wx.ALL, 5)
        NavigBoxsizer2b.Add(self.centerindexctrl, 0, wx.ALL, 5)
        NavigBoxsizer2b.Add(self.txtimagefastindexbox, 0, wx.ALL, 5)
        NavigBoxsizer2b.Add(self.txtimagefastindexboxctrl, 0, wx.ALL, 5)
        NavigBoxsizer2b.Add(self.txtimageslowindexbox, 0, wx.ALL, 5)
        NavigBoxsizer2b.Add(self.txtimageslowindexboxctrl, 0, wx.ALL, 5)

        ROIBoxsizer = wx.BoxSizer(wx.HORIZONTAL)
        ROIBoxsizer.Add(self.predefinedROIradiobtn, 0, wx.ALL, 5)
        ROIBoxsizer.Add(self.twtROI, 0, wx.ALL, 5)
        ROIBoxsizer.Add(self.comboROI, 0, wx.ALL, 5)

        self.NavigBoxsizer3 = wx.BoxSizer(wx.HORIZONTAL)
        self.NavigBoxsizer3.Add(self.txtnbimagesperline, 0, wx.ALL, 5)
        self.NavigBoxsizer3.Add(self.stepctrl, 0, wx.ALL, 5)
        self.NavigBoxsizer3.Add(self.txtmapstartingindex, 0, wx.ALL, 5)
        self.NavigBoxsizer3.Add(self.mapstartingimageindexctrl, 0, wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt1, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer0, 0, wx.EXPAND)
        vbox.Add(txt2, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer2, 0, wx.EXPAND)
        vbox.Add(NavigBoxsizer2b, 0, wx.EXPAND)
        vbox.Add(ROIBoxsizer, 0, wx.EXPAND)

        vbox.Add(txt3, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer, 0, wx.EXPAND)
        vbox.Add(NavigBoxsizer1, 0, wx.EXPAND)

        vbox.Add(txt4, 0, wx.EXPAND)
        vbox.Add(self.NavigBoxsizer3, 0, wx.EXPAND)
        vbox.Add(self.btnMosaic, 0, wx.EXPAND)

        self.SetSizer(vbox)

        # tooltips
        self.boxsizetxt.SetToolTipString(
            "HALF sizes in Laue Pattern image of the Region of Interest(ROI) box")

        tpx = "Pixel HALF box size in Laue Pattern image of the crop box in X direction to be cut"
        tpy = "Pixel HALF box size in Laue Pattern image of the crop box in Y direction to be cut"
        self.boxxtxt.SetToolTipString(tpx)
        self.boxxctrl.SetToolTipString(tpx)

        self.boxytxt.SetToolTipString(tpy)
        self.boxyctrl.SetToolTipString(tpy)

        self.btnclearwindows.SetToolTipString("Clear children plot windows")

        txt2.SetToolTipString("Set parameters to build list of image indices (2 methods)")
        self.generalindexradiobtn.SetToolTipString("Set parameters for continuous variation of image index")

        tps = "Starting image filename index"
        self.startindex.SetToolTipString(tps)
        self.startindexctrl.SetToolTipString(tps)

        tpe = "Last image filename index"
        self.lastindex.SetToolTipString(tpe)
        self.lastindexctrl.SetToolTipString(tpe)

        tpstep = "image filename step"
        self.stepimageindex.SetToolTipString(tpstep)
        self.stepimageindexctrl.SetToolTipString(tpstep)

        self.rectangleindexradiobtn.SetToolTipString("Set parameters for image indices contained "
                                            "in rectangle defined by its center and half lengthes")
        tipcenter = "Image index center of rectangle area"
        self.txtimagecenter.SetToolTipString(tipcenter)
        self.centerindexctrl.SetToolTipString(tipcenter)
        tiprectboxX = "Nb of image indices along X (fast motor scan direction) as half rectangle width"
        self.txtimagefastindexbox.SetToolTipString(tiprectboxX)
        self.txtimagefastindexboxctrl.SetToolTipString(tiprectboxX)
        tiprectboxY = "Nb of image indices along Y (slow motor scan direction) as half rectangle height"
        self.txtimageslowindexbox.SetToolTipString(tiprectboxY)
        self.txtimageslowindexboxctrl.SetToolTipString(tiprectboxY)

        txt4.SetToolTipString("Sample 2D Map/Mesh/Raster scan properties in terms of image indices")
        tpnb = "Number of images per line to arrange results in 2D plot"
        self.txtnbimagesperline.SetToolTipString(tpnb)
        self.stepctrl.SetToolTipString(tpnb)
        tipstartmap = "Starting Image index of the raster 2D (mesh) scan"
        self.txtmapstartingindex.SetToolTipString(tipstartmap)
        self.mapstartingimageindexctrl.SetToolTipString(tipstartmap)

        self.btnMosaic.SetToolTipString("Compute and plot Mosaic and Monitors from pixel intensity "
        "in selected ROI. Check boxes to select type of monitors")

        self.mosaiccounter.SetToolTipString("Recompose a 2D raster scan from the selected ROI as "
                                                                "a function of image index")
        self.meancounter.SetToolTipString("Recompose a 2D raster scan with MEAN pixel intensity "
                                                "in selected ROI as a function of image index")
        self.maxcounter.SetToolTipString("Recompose a 2D raster scan with MAXIMUM pixel intensity found in selected ROI as a "
                                                                        "function of image index")
        self.peaktopeakcounter.SetToolTipString("Recompose a 2D raster scan with largest "
        "PEAK-TO-PEAK (or Peak-to-Valley) pixel intensity found selected ROI as a function of image index")
        self.xyposcounter.SetToolTipString("Sample Map of 2D gaussian refined X Y peak position"
        "distribution of the ROI")

        self.amplitudecounter.SetToolTipString("Sample Map of Peak Amplitude obtained from 2D gaussian peak profile fittings")
        self.relativexyposcounter.SetToolTipString("Sample Map of Peak relative position from 2D gaussian peak profile fittings")
        self.peaksizecounter.SetToolTipString("Sample Map of Peak Size from 2D gaussian peak profile fittings")

        tpdigits = 'nb of digits for zero padding in image filename'
        self.txtnbdigits.SetToolTipString(tpdigits)
        self.nbdigitsctrl.SetToolTipString(tpdigits)

    #     def onSortROIname(self, evt):
    #         listROI = self.dict_ROI.keys()
    #         listROIs = sorted(listROI, key=str.lower)
    #         self.comboROI.Clear()
    #         self.comboROI.AppendItems(listROIs)

    def OnChangeROI(self, _):
        ROIselected = self.comboROI.GetValue()
        print("selected ", ROIselected)
        print(self.dict_ROI[ROIselected])

    def OnMosaic(self, _):
        """  launch main procedure of computing mosaic and displaying it
        """
        self.getcounters()

        print("current dict of ROIs", self.dict_ROI)

        self.mainframe.buildMosaic(parent=self)

    def getcounters(self):
        dictCounters = {0: "mosaic",
                        1: "mean",
                        2: "max",
                        3: "ptp",
                        4: "Position XY",
                        5: "Amplitude",
                        6: "Displacement",
                        7: "Shape"}

        list_chckcounters = [self.mosaiccounter,
                            self.meancounter,
                            self.maxcounter,
                            self.peaktopeakcounter,
                            self.xyposcounter,
                            self.amplitudecounter,
                            self.relativexyposcounter,
                            self.peaksizecounter]
        cselected = []
        for k, chckbtn in enumerate(list_chckcounters):
            if chckbtn.GetValue():
                selected = dictCounters[k]
                print("counter %s selected" % selected)
                cselected.append(selected)

        self.cselected = cselected

    def OnClearChildWindows(self, _):
        print("killing children!")
        print("list_of_windows", self.list_of_windows)
        for child in self.list_of_windows:
            if isinstance(child, wx.Frame):
                child.Close()

        self.list_of_windows = []


class ROISelection(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.mainframe = parent.GetParent().GetParent()  # layout2()

        self.list_of_windows = []
        self.dict_ROI = {"None": (5, 2, 2)}
        self.ROIsarray = None
        self.cselected = None

        # widgets --------------------------------
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        txt1 = wx.StaticText(self, -1, "Image pixel ROI size")
        txt1.SetFont(font3)

        self.boxxtxt = wx.StaticText(self, -1, "X")
        self.boxxctrl = wx.SpinCtrl(self, -1, "30", min=0, max=9999)#, size=(100, -1))
        self.boxytxt = wx.StaticText(self, -1, "Y")
        self.boxyctrl = wx.SpinCtrl(self, -1, "30", min=0, max=9999)#, size=(100, -1))

        txt2 = wx.StaticText(self, -1, "ROI selection mode")
        txt2.SetFont(font3)

        self.ROIfromPeaklistbtn = wx.Button(self, wx.ID_ANY, "Auto. on peaks")
        self.ROIfromPeaklistbtn.Bind(wx.EVT_BUTTON, self.onAddROIsonPeaks)

        self.ROIfromManualtbtn = wx.Button(self, wx.ID_ANY, "Manual Rectangle")
        self.ROIfromManualtbtn.Bind(wx.EVT_BUTTON, self.onManualROIsSelection)

        self.centerROIbtn = wx.Button(self, wx.ID_ANY, "Center on pixel")
        self.centerROIbtn.Bind(wx.EVT_BUTTON, self.onCenterROI)

        self.deleteROIsbtn = wx.Button(self, wx.ID_ANY, "Delete ROIs")
        self.deleteROIsbtn.Bind(wx.EVT_BUTTON, self.onDeleteROIs)

        txt3 = wx.StaticText(self, -1, "Export ROIs")
        txt3.SetFont(font3)

        txt4 = wx.StaticText(self, -1, "Collect Data from ROIs")
        txt4.SetFont(font3)
        self.collectbtn = wx.Button(self, wx.ID_ANY, "start collect")
        self.collectbtn.Bind(wx.EVT_BUTTON, self.onstartcollectrois)

        self.saveROIbtn = wx.Button(self, wx.ID_ANY, "Save ROIs in file")
        self.saveROIbtn.Bind(wx.EVT_BUTTON, self.onSaveROIs)
        self.sendROItoSPECbtn = wx.Button(self, wx.ID_ANY, "Send ROIs->SPEC")
        self.sendROItoSPECbtn.Bind(wx.EVT_BUTTON, self.onSendToSpec)

        NavigBoxsizer0 = wx.BoxSizer(wx.HORIZONTAL)
        NavigBoxsizer0.Add(self.boxxtxt, 0, wx.ALL, 5)
        NavigBoxsizer0.Add(self.boxxctrl, 0, wx.ALL, 5)
        NavigBoxsizer0.Add(self.boxytxt, 0, wx.ALL, 5)
        NavigBoxsizer0.Add(self.boxyctrl, 0, wx.ALL, 5)

        NavigBoxsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        NavigBoxsizer2.Add(self.ROIfromPeaklistbtn, 0, wx.ALL, 5)
        NavigBoxsizer2.Add(self.ROIfromManualtbtn, 0, wx.ALL, 5)

        NavigBoxsizer3 = wx.BoxSizer(wx.HORIZONTAL)
        NavigBoxsizer3.Add(self.saveROIbtn, 0, wx.ALL, 5)
        NavigBoxsizer3.Add(self.sendROItoSPECbtn, 0, wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt1, 0, wx.EXPAND)
        vbox.Add(NavigBoxsizer0, 0, wx.EXPAND)
        vbox.Add(txt2, 0, wx.EXPAND)
        vbox.Add(NavigBoxsizer2, 0, wx.EXPAND)
        vbox.Add(self.deleteROIsbtn, 0, wx.ALL, 5)
        vbox.Add(txt3, 0, wx.EXPAND)
        vbox.Add(NavigBoxsizer3, 0, wx.EXPAND)
        vbox.Add(txt4, 0, wx.EXPAND)
        vbox.Add(self.collectbtn, 0, wx.EXPAND)

        self.SetSizer(vbox)

        # tooltips -----------------------------------------
        tpx = "Pixel HALF box size in Laue Pattern image of the ROI box in X direction horizontal"
        tpy = "Pixel HALF box size in Laue Pattern image of the ROI box in Y direction vertical "
        self.boxxtxt.SetToolTipString(tpx)
        self.boxxctrl.SetToolTipString(tpx)

        self.boxytxt.SetToolTipString(tpy)
        self.boxyctrl.SetToolTipString(tpy)

        self.ROIfromPeaklistbtn.SetToolTipString("Automatic ROIs centered on current peaks (from "
                                                                    "the peak search procedure)")
        self.centerROIbtn.SetToolTipString("Center ROI on clicked pixel")

        # txt2.SetToolTipString('Set parameters to build list of image indices (2 methods)')

        self.ROIfromManualtbtn.SetToolTipString(
            'Select with mouse a rectangle: Accept with "q", delete with "d"')
    
    def onstartcollectrois(self,_):
        folder, filename = '/home/micha/LaueProjects/MapSn', 'SnsurfscanBig_1382.mccd'
        listimages= np.arange(0,1383)   # 41 images/line and aborted
        nbimagesperline = 41
        import pickle
        imagefilename = filename

        nbimages = len(listimages)

        Xc, Yc = 1000,1400
        hboxX,hboxY = 300,300
        ndivisions = (4,20)  # y, x 
        nrois = ndivisions[0]*ndivisions[1]

        multidtector = []
        maxposs = np.zeros((nbimages,nrois, 2))
        k=0
        for idx in listimages:
            
            if idx % 10 == 0: print(" image %d / %d  "%(idx, listimages[-1]))
            filename = imagefilename[:-10]+'_%04d'%idx + '.mccd'
            print('filename', filename)
            
            param = (folder, filename, idx,[[Xc, Yc]], hboxX,hboxY)
            cdata = MOS.CollectData_oneImage(param, folder,selectedcounters=['Imax_multiple'], ndivisions=ndivisions)
            maxIs = cdata['Imax_multiple'][0]
            multidtector.append(maxIs)
            
            if 0:  #max pixel position
                cd = MOS.CollectData_oneImage(param, folder, selectedcounters=['posmax_multiple'],
                                                                                            ndivisions=ndivisions)
                maxposs[k]= cd['posmax_multiple'][0]
            if idx % 200 == 0:  # partial pickeling to be optimised
                dets = np.array(multidtector)
                kk = 0
                alldets=[]
                for i in range(ndivisions[0]):
                    for j in range(ndivisions[1]):
                        singledet = GT.to2Darray(dets[:,kk],nbimagesperline)
                        alldets.append(singledet)
                        kk+=1
                with open(os.path.join(folder,'alldets_%d.pickle'%k),'wb') as f:
                    pickle.dump(alldets, f)
            
            k+=1
            
        dets = np.array(multidtector)
        maxs = np.array(maxposs)

        kk = 0
        alldets=[]
        for i in range(ndivisions[0]):
            for j in range(ndivisions[1]):
                singledet = GT.to2Darray(dets[:,kk],nbimagesperline)
                alldets.append(singledet)
                kk+=1
        with open(os.path.join(folder,'alldets.pickle'),'wb') as f:
            pickle.dump(alldets, f)

    def buildROIsarray(self):
        ROIslist = []
        for _, roi in self.mainframe.ROIs.items():
            x, y, width, minusheight, _, Lauetoolsindex, visibleflag = roi
            if visibleflag == "visible":
                xmin = int(x)
                xmax = int(x + width)
                ymin = int(y)
                ymax = int(y - minusheight)
                ROIslist.append([xmin, xmax, ymin, ymax, Lauetoolsindex])

        return np.array(ROIslist)

    def onSaveROIs(self, _):
        print("self.ROIs", self.mainframe.ROIs)

        self.ROIsarray = self.buildROIsarray()

        outputfile = open("ROIs.dat", "w")
        outputfile.write("xmin xmax ymin ymax roiindex\n")
        np.savetxt(outputfile, self.ROIsarray, fmt="%.6f")
        outputfile.close()

    def onSendToSpec(self, _):

        return

        # try:
        #     from SpecClient_gevent import SpecVariable, SpecVariable #, SpecCommand
        #     import ConnectPSL as psl
        # except ImportError:
        #     wx.MessageBox('Module to connect to current beamline control software '
        #                                                                 'is not installed', 'INFO')
        #     return
        # spec = "laue"

        # pslroi = SpecVariable.SpecVariable("PSL_ROI", spec)
        # pslnroi = SpecVariable.SpecVariable("PSL_ROICNT", spec)

        # self.ROIsarray = self.buildROIsarray()

        # n = 1
        # roi = {}
        # psl.SendAndRecv("ClearStatRoi")
        # print("")
        # for roielem in self.ROIsarray:
        #     xmin, xmax, ymin, ymax, _ = roielem
        #     psl.SendAndRecv("AddStatRoi;%d;%d;%d;%d" % (xmin, ymin, xmax, ymax))
        #     psl.SendAndRecv("DrawStatRoi;%d" % (n))
        #     roi[n - 1] = {"xmin": round(xmin),
        #                     "xmax": round(xmax),
        #                     "ymin": round(ymin),
        #                     "ymax": round(ymax),
        #                     "Meas": "Sum"}
        #     n += 1
        # if spec != "":
        #     pslroi.setValue(roi)
        #     pslnroi.setValue(n - 1)


    def onCenterROI(self, evt):
        """simply click and later press q
        """
        pass

    def onManualROIsSelection(self, _):
        if len(self.mainframe.ROIs) > 0:
            self.mainframe.roiindex = max(self.mainframe.ROIs.keys())
        else:
            self.mainframe.roiindex = 0

    def onAddROIsonPeaks(self, evt):

        peakslist = self.mainframe.peaklistPixels

        if peakslist is None:
            wx.MessageBox(
                'Peaks list is empty! Please, search for peaks by pressing on "Search All Peaks" button for instance',
                "Info")
            return

        # correction only to fit peak position to the display
        if self.mainframe.position_definition == 1:
            offset_convention = np.array([1, 1])
            if peakslist.shape == (10,):
                XYlist = (peakslist[:2] - offset_convention,)
            else:
                XYlist = peakslist[:, :2] - offset_convention

            halfboxx = int(self.boxxctrl.GetValue())
            halfboxy = int(self.boxyctrl.GetValue())

            height, width = 2 * halfboxx + 1, 2 * halfboxy + 1
            for k, po in enumerate(XYlist):

                x, y = po[0] - halfboxx, po[1] + halfboxy

                rectproperties = [x, y, height, width, None, k, None]

                self.mainframe.ROIs[k] = rectproperties
                self.mainframe.addPatchRectangleROI(rectproperties)

            print("updated ROIs", self.mainframe.ROIs)

            self.mainframe.update_draw(evt)

    def onDeleteROIs(self, evt):

        for _, rect in self.mainframe.ROIs.items():

            rect[4].set_visible(False)
            rect[4].set_picker(None)

        self.mainframe.ROIs = {}
        self.mainframe.roiindex = 0

        self.mainframe.update_draw(evt)

    def OnMosaic(self, _):
        """in ROIselection class"""

        self.getcounters()

        print("current dict of ROIs", self.dict_ROI)

        self.mainframe.buildMosaic(parent=self)

    def getcounters(self):
        dictCounters = {0: "mosaic",
                        1: "mean",
                        2: "max",
                        3: "ptp",
                        4: "Position XY",
                        5: "Amplitude",
                        6: "Displacement",
                        7: "Shape"}

        list_chckcounters = [self.mosaiccounter,
                            self.meancounter,
                            self.maxcounter,
                            self.peaktopeakcounter,
                            self.xyposcounter,
                            self.amplitudecounter,
                            self.relativexyposcounter,
                            self.peaksizecounter]
        cselected = []
        for k, chckbtn in enumerate(list_chckcounters):
            if chckbtn.GetValue():
                selected = dictCounters[k]
                print("counter %s selected" % selected)
                cselected.append(selected)

        self.cselected = cselected

    def OnClearChildWindows(self, _):
        print("killing children!")
        print("list_of_windows", self.list_of_windows)
        for child in self.list_of_windows:
            if isinstance(child, wx.Frame):
                child.Close()

        self.list_of_windows = []


class PlotPeakListPanel(wx.Panel):
    """panel class to handle peaks list within GUI

    IN DEV, NOT USED
    """
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()  # layout2()
        self.xc, self.yc = None, None
        self.indexcirclespatchlist = None
        self.drg = None
        self.DetFilename = None
        #         print "mainframe of ViewColorPanel", self.mainframe

        # widgets -----------------------
        luttxt = wx.StaticText(self, -1, "LUT", (5, 7))
        self.comboLUT = wx.ComboBox(self, -1, self.mainframe.LastLUT, (70, 5),
                                                                choices=self.mainframe.mapsLUT)

        self.comboLUT.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeLUT)

        posv = 40
        self.slider_label = wx.StaticText(self, -1, "Imin: ", (5, posv + 5))
        self.vminctrl = wx.SpinCtrl(self, -1, "0", pos=(50, posv), #size=(100, -1),
                                                                min=-200, max=9999)
        self.Bind(wx.EVT_SPINCTRL, self.mainframe.OnSpinCtrl_IminDisplayed, self.vminctrl)

        # second horizontal band
        self.slider_label2 = wx.StaticText(self, -1, "Imax: ", (5, posv + 35))

        self.vmaxctrl = wx.SpinCtrl(self, -1, "1000", pos=(50, posv + 30), #size=(100, -1),
                                                                min=2, max=9999)
        self.Bind(wx.EVT_SPINCTRL, self.mainframe.OnSpinCtrl_ImaxDisplayed, self.vmaxctrl)
        self.slider_vmin = wx.Slider(self, -1, pos=(150, posv + 5), size=(220, -1), value=0,
                                                        minValue=0,
                                                        maxValue=1000,
                                                        style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_vmin.SetTickFreq(500)
        else:
            self.slider_vmin.SetTickFreq(500, 1)
        self.slider_vmin.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.mainframe.on_slider_IminDisplayed)

        # second horizontal band
        self.slider_vmax = wx.Slider(self, -1, pos=(150, posv + 35), size=(220, -1), value=1000,
                                                        minValue=1,
                                                        maxValue=1000,
                                                        style=wx.SL_AUTOTICKS)  # | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_vmax.SetTickFreq(500)
        else:
            self.slider_vmax.SetTickFreq(500, 1)
        self.slider_vmax.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK,
                            self.mainframe.on_slider_ImaxDisplayed)

        self.Iminvaltxt = wx.StaticText(self, -1, "0", pos=(400, posv + 5))
        self.Imaxvaltxt = wx.StaticText(self, -1, "1000", pos=(400, posv + 35))

        self.lineXYprofilechck = wx.CheckBox(
            self, -1, "Enable X Y profiler", (5, posv + 60))
        InitStateXYProfile = False
        self.lineXYprofilechck.SetValue(InitStateXYProfile)
        self.lineXYprofilechck.Bind(wx.EVT_CHECKBOX, self.OnTogglelineXYprofiles)
        self.plotlineXprofile = InitStateXYProfile
        self.plotlineYprofile = InitStateXYProfile

        self.lineprof_btn = wx.Button(self, -1, "Open LineProfiler", (200, posv + 60), (130, -1))
        self.lineprof_btn.Bind(wx.EVT_BUTTON, self.OnShowLineProfiler)

        savefig_btn = wx.Button(self, -1, "SaveFig", (5, posv + 150), (80, -1))
        savefig_btn.Bind(wx.EVT_BUTTON, self.mainframe.OnSaveFigure)

        replot_btn = wx.Button(self, -1, "Replot", (120, posv + 150), (80, -1))
        replot_btn.Bind(wx.EVT_BUTTON, self.mainframe.OnReplot)

        self.show2thetachi = wx.CheckBox(self, -1, "Show 2theta Chi", (5, posv + 100))
        self.detfiletxtctrl = wx.TextCtrl(self, -1, "", (180, posv + 100), (100, -1))
        self.opendetfilebtn = wx.Button(self, -1, "...", (300, posv + 100))
        self.show2thetachi.SetValue(False)
        self.opendetfilebtn.Bind(wx.EVT_BUTTON, self.onOpenDetFile)

        showhisto_btn = wx.Button(self, -1, "Intensity Distribution", (260, 5))
        showhisto_btn.Bind(wx.EVT_BUTTON, self.mainframe.ShowHisto)

        self.plotlineprofileframe = None
        self.plotlineXprofileframe = None
        self.plotlineYprofileframe = None
        self.x0, self.y0, self.x2, self.y2 = 200, 200, 1800, 1800

        # tooltip
        tp1 = "selection of various intensity mapping color Looking_up table (LUT) "

        luttxt.SetToolTipString(tp1)
        self.comboLUT.SetToolTipString(tp1)

        showhisto_btn.SetToolTipString("Show histogram of pixel intensity distribution of raw image")

        tp2 = "Coarse color scale intensity: min and max"

        self.slider_label.SetToolTipString(tp2)
        self.vminctrl.SetToolTipString(tp2)
        self.slider_label2.SetToolTipString(tp2)
        self.vmaxctrl.SetToolTipString(tp2)

        tp3 = "fine color scale intensity minimum"
        self.slider_vmin.SetToolTipString(tp3)
        tp4 = "fine color scale intensity maximum"
        self.slider_vmax.SetToolTipString(tp4)

        tpmin = "current color scale intensity minimum"
        tpmax = "current color scale intensity maximum"

        self.Iminvaltxt.SetToolTipString(tpmin)
        self.Imaxvaltxt.SetToolTipString(tpmax)

        savefig_btn.SetToolTipString("Save current plot figure in png format file")

        replot_btn.SetToolTipString("Reset and replot displayed image in plot")

        tiplineprof = "Open/Close Draggable Line Pixel intensity Profiler.\n"
        tiplineprof += "Press left mouse button to move line.\n"
        tiplineprof += ("Press right mouse button to change line length (or, press + or -).\n")

        self.lineprof_btn.SetToolTipString(tiplineprof)

    def showprofiles(self, event):

        self.updateLineProfile()
        #         print 'event update', event
        if self.plotlineXprofile and self.plotlineYprofile:
            if isinstance(event, mpl.backend_bases.MouseEvent):
                self.OnShowLineXYProfiler(event)

    #         self.OnShowLineProfiler(evt)

    def OnTogglelineXYprofiles(self, evt):
        self.plotlineXprofile = not self.plotlineXprofile
        self.plotlineYprofile = not self.plotlineYprofile

        self.showprofiles(evt)

    def getlineXYprofiledata(self, event):
        ax = self.mainframe.axes

        xmin, xmax = ax.get_xlim()
        # warning of ymax and ymin swapping!
        ymax, ymin = ax.get_ylim()

        xmin, xmax, ymin, ymax = [int(val) for val in (xmin, xmax, ymin, ymax)]

        xmin, xmax, ymin, ymax = self.restrictxylimits_to_imagearray(xmin, xmax, ymin, ymax)

        #         print "xmin, xmax, ymin, ymax", xmin, xmax, ymin, ymax

        if self.mainframe.dataimage_ROI_display is not None:
            # Extract the values along the line
            z = self.mainframe.dataimage_ROI_display
        else:
            z = self.mainframe.dataimage_ROI

        xyc = self.getclickposition(event)
        if xyc:
            self.xc, self.yc = xyc

        #         print "xc, yc", xc, yc

        x = np.arange(xmin, xmax + 1)
        y = np.arange(ymin, ymax + 1)
        zx = z[int(np.round(self.yc)), xmin : xmax + 1]
        zy = z[ymin : ymax + 1, int(np.round(self.xc))]

        #         print "len(y)", len(y)
        #         print "len(zy)", len(zy)
        return x, y, zx, zy

    def getclickposition(self, event):
        if not isinstance(event, mpl.backend_bases.MouseEvent):
            return None

        return int(np.round(event.xdata)), int(np.round(event.ydata))

    def OnShowLineXYProfiler(self, event):
        """ show line X and Y profilers
        """
        LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT = 900, 300
        print("self.plotlineXprofile and self.plotlineYprofile == ",
                                                self.plotlineXprofile and self.plotlineYprofile)
        if self.plotlineXprofile and self.plotlineYprofile:

            if (self.plotlineXprofileframe is None and self.plotlineYprofileframe is None):

                x, y, zx, zy = self.getlineXYprofiledata(event)

                xp, yp = self.bestposition(LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT)

                # -- Plot lineprofile...
                self.plotlineXprofileframe = PLOT1D.Plot1DFrame(self, -1, "Intensity profile X",
                                            "", [x, zx], logscale=0, figsize=(8, 3),
                                            dpi=100, size=(LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT))
                self.plotlineXprofileframe.SetPosition((xp, yp))
                self.plotlineXprofileframe.Show(True)

                self.plotlineYprofileframe = PLOT1D.Plot1DFrame(self, -1, "Intensity profile Y",
                                            "", [y, zy], logscale=0, figsize=(8, 3),
                                            dpi=100, size=(LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT))
                self.plotlineYprofileframe.SetPosition((xp - 200, yp))
                self.plotlineYprofileframe.Show(True)

            if (self.plotlineXprofileframe is not None or self.plotlineYprofileframe is not None):
                self.updateLineXYProfile(event)

    def bestposition(self, LINEPROFILE_WIDTH, LINEPROFILE_HEIGHT):
        # screen size
        dws, dhs = wx.DisplaySize()
        # peaksearchframe size
        wf, hf = self.mainframe.GetSize()
        print("wf, hf", wf, hf)
        print("dws, dhs", dws, dhs)

        if dws > wf:
            Xwindow = dws - LINEPROFILE_WIDTH
        if dhs > hf:
            Ywindow = dhs - LINEPROFILE_HEIGHT

        xp, yp = max(Xwindow, 0), max(Ywindow, 0)
        print("xp,yp", xp, yp)

        return xp, yp

    def restrictxylimits_to_imagearray(self, xmin, ymin, xmax, ymax):
        dim = self.mainframe.framedim

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, dim[1] - 1)
        ymax = min(ymax, dim[0] - 1)

        return xmin, ymin, xmax, ymax

    def getlineprofiledata(self, x0, y0, x2, y2):
        """
        get pixel intensities array between two points (x0,y0) and (x2,y2)
        """

        x0, y0, x2, y2 = self.restrictxylimits_to_imagearray(x0, y0, x2, y2)
        #
        #         print 'x0, y0, x2, y2'
        #         print x0, y0, x2, y2

        length = int(np.hypot(x2 - x0, y2 - y0))
        x1 = (x2 + x0) / 2.0
        y1 = (y2 + y0) / 2.0

        x, y = np.linspace(x0, x2, length), np.linspace(y0, y2, length)

        if self.mainframe.dataimage_ROI_display is not None:
            # Extract the values along the line
            z = self.mainframe.dataimage_ROI_display
        else:
            z = self.mainframe.dataimage_ROI

        # transpose data
        i_ind = y.astype(np.int16)
        j_ind = x.astype(np.int16)

        zi = z[i_ind, j_ind]

        if x0 != x2:
            signabscissa = np.sign(x - x1)
        else:
            signabscissa = np.sign(y - y1)

        dataX = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) * signabscissa
        dataY = zi

        return dataX, dataY

    def OnShowLineProfiler(self, _):
        """
        show a movable and draggable line profiler and draw line and circle
        """
        if self.plotlineprofileframe is None:
            self.plotlineXprofile = False
            self.plotlineYprofile = False
            self.lineXYprofilechck.SetValue(False)

            self.lineprof_btn.SetLabel("Close LineProfiler")
            #             self.plotlineXprofile = False
            #             self.plotlineYprofile = False
            #             self.lineXYprofilechck.SetValue(False)

            # -- Plot lineprofile...
            x, zi = self.getlineprofiledata(self.x0, self.y0, self.x2, self.y2)

            self.plotlineprofileframe = PLOT1D.Plot1DFrame(
                self, -1, "Intensity profile", "", [x, zi], logscale=0, figsize=(8, 3))
            self.plotlineprofileframe.Show(True)

            # plot line on canvas (CCD image)

            pt1 = [self.x0, self.y0]
            pt2 = [self.x2, self.y2]
            ptcenter = DGP.center_pts(pt1, pt2)

            circles = [Circle(pt1, 50, fill=True, color="r", alpha=0.5),
                        Circle(ptcenter, 50, fc="r", alpha=0.5),
                        Circle(pt2, 50, fill=True, color="r", alpha=0.5)]

            line, = self.mainframe.axes.plot([pt1[0], ptcenter[0], pt2[0]],
                                            [pt1[1], ptcenter[1], pt2[1]],
                                            picker=1,
                                            c="r")
            self.indexcirclespatchlist = []
            init_patches_nb = len(self.mainframe.axes.patches)
            for k, circ in enumerate(circles):
                self.mainframe.axes.add_patch(circ)
                self.indexcirclespatchlist.append(init_patches_nb + k)
            #             print "building draggable line"
            #             print "peak search canvas", self.mainframe.canvas
            self.drg = DGP.DraggableLine(circles,
                                        line,
                                        tolerance=200,
                                        parent=self.mainframe,
                                        framedim=self.mainframe.framedim,
                                        datatype="pixels")

            self.mainframe.axes.set_xlim(0, 2047)
            self.mainframe.axes.set_ylim(2047, 0)
            self.mainframe.canvas.draw()

        else:
            self.lineprof_btn.SetLabel("Open LineProfiler")

            self.drg.connectingline.set_data([0], [0])  # empty line

            # warning patches list can contain circle marker from peak search
            #             print "self.mainframe.axes.patches", self.mainframe.axes.patches
            for k in range(len(self.indexcirclespatchlist)):
                del self.mainframe.axes.patches[self.indexcirclespatchlist[0]]

            dim = self.mainframe.framedim

            self.mainframe.axes.set_xlim(0, dim[1] - 1)
            self.mainframe.axes.set_ylim(dim[0] - 1, 0)
            self.mainframe.canvas.draw()
            self.drg = None
            self.plotlineprofileframe.Destroy()
            self.plotlineprofileframe = None

        return

    def updateLineXYProfile(self, event):

        if self.plotlineXprofileframe is not None:
            x, y, zx, zy = self.getlineXYprofiledata(event)

            if len(x) != len(zx) or len(y) != len(zy):
                print("STRANGE")
                return

            xyc = self.getclickposition(event)
            if xyc:
                self.xc, self.yc = xyc
    
            lineXprofileframe = self.plotlineXprofileframe

            lineXprofileframe.line.set_data(x, zx)
            lineXprofileframe.axes.set_title("%s\n@y=%s" % (self.mainframe.imagefilename, self.yc))
            lineXprofileframe.axes.relim()
            lineXprofileframe.axes.autoscale_view(True, True, True)
            lineXprofileframe.canvas.draw()

        if self.plotlineYprofileframe is not None:
            x, y, zx, zy = self.getlineXYprofiledata(event)

            if len(x) != len(zx) or len(y) != len(zy):
                print("STRANGE")
                return

            xyc = self.getclickposition(event)
            if xyc:
                self.xc, self.yc = xyc
 
            lineYprofileframe = self.plotlineYprofileframe

            lineYprofileframe.line.set_data(y, zy)
            lineYprofileframe.axes.set_title("%s\n@x=%s" % (self.mainframe.imagefilename, self.xc))
            lineYprofileframe.axes.relim()
            lineYprofileframe.axes.autoscale_view(True, True, True)
            lineYprofileframe.canvas.draw()

    def updateLineProfile(self):
        if self.plotlineprofileframe is not None:

            x, zi = self.getlineprofiledata(self.x0, self.y0, self.x2, self.y2)
  
            lineprofileframe = self.plotlineprofileframe

            lineprofileframe.line.set_data(x, zi)
            lineprofileframe.axes.set_title("%s\nline [%d,%d]-[%d,%d]"
                % (self.mainframe.imagefilename, self.x0, self.y0, self.x2, self.y2))
            lineprofileframe.axes.relim()
            lineprofileframe.axes.autoscale_view(True, True, True)
            lineprofileframe.canvas.draw()

    def onOpenDetFile(self, _):
        self.mainframe.ReadDetFile(1)
        self.DetFilename = self.mainframe.DetFilename
        self.detfiletxtctrl.SetValue(self.mainframe.DetFilename)


class findLocalMaxima_Meth_1(wx.Panel):
    """
    class of method 1 for local maxima search (intensity threshold)
    """

    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.methodnumber = 1
        mainframe = parent.GetParent().GetParent()
        CCDlabel = mainframe.CCDlabel
        if CCDlabel.startswith("sCMOS"):
            pedestal = 1000
        else:
            pedestal = 0

        #defaultthreshold = pedestal + 500
        _, defaultthreshold = mainframe.gethisto()

        mintxt = wx.StaticText(self, -1, "MinimumDistance")
        self.PNR = wx.SpinCtrl(self, -1, "10", (100, -1), min=2, max=9999)

        ittxt = wx.StaticText(self, -1, "IntensityThreshold")
        self.IT = wx.SpinCtrl(self, -1, str(defaultthreshold), #(100, -1),
                                            min=-6000, max=10000000000)

        # layout
        h1 = wx.BoxSizer(wx.HORIZONTAL)
        h1.Add(mintxt, 0 , wx.ALL, 5)
        h1.Add(self.PNR, 0 , wx.ALL, 5)

        h2 = wx.BoxSizer(wx.HORIZONTAL)
        h2.Add(ittxt,0 , wx.ALL, 5)
        h2.Add(self.IT,0 , wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(h1,0, wx.EXPAND,10)
        vbox.Add(h2,0, wx.EXPAND,5)
        
        self.SetSizer(vbox)

        # tooltips
        mintp = "Minimum pixel distances between local maxima"
        mintxt.SetToolTipString(mintp)
        self.PNR.SetToolTipString(mintp)

        ittp = "Threshold level above which local maxima must be found"
        ittxt.SetToolTipString(ittp)
        self.IT.SetToolTipString(ittp)


class findLocalMaxima_Meth_2(wx.Panel):
    """
    class of method parameters for 2nd method of local maxima(shifted arrays)
    """

    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.methodnumber = 2

        #        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        #        self.title2 = wx.StaticText(self, -1, 'Local Maxima Search Parameters(Shifted Arrays)', (5, 6))
        #        self.title2.SetFont(font3)

        pnrtxt = wx.StaticText(self, -1, "PixelNearRadius", (5, 5))
        self.PNR = wx.SpinCtrl(self, -1, "10", (140, 5), #(100, -1),
                                                    min=5, max=9999)

        ittxt = wx.StaticText(self, -1, "IntensityThreshold", (5, 35))
        self.IT = wx.SpinCtrl(self, -1, "500", (140, 35), #(100, -1),
                                                    min=0, max=9999)

        # tooltips
        pnrtp = "Minimum pixel distances between local maxima"
        pnrtxt.SetToolTipString(pnrtp)
        self.PNR.SetToolTipString(pnrtp)

        ittp = "Threshold level - RELATIVE to local background- above which local maxima must be found.\n"
        ittp += "Local background is a flat level given by the minimum intensity in a box centered on local maximum and with size 2*PixelNearRadius+1"
        ittxt.SetToolTipString(ittp)
        self.IT.SetToolTipString(ittp)

class findLocalMaxima_Meth_3(wx.Panel):
    """
    class of method 3 for local maxima search (convolution by a mexican hat kernel)
    """

    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """
        instantiate the panel and widgets for Convolution Method

        mainframe= parent.GetParent().GetParent())

        this mainframe must have following attributes and methods:
        ShowHisto_ConvolvedData()
        ComputeConvolvedData()
        Show_ConvolvedImage()
        Show_Image()
        OnSpinCtrl_IminDisplayed()
        dataimage_ROI
        dataimage_ROI_display
        """
        self.methodnumber = 3
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        # widgets ---------------------------------
        posv = 30

        pnrtxt = wx.StaticText(self, -1, "PixelNearRadius", (5, 32 - posv))
        self.PNR = wx.SpinCtrl(self, -1, "10", (125, 30 - posv), #(100, -1),
                                                        min=5, max=9999)

        showhisto_btn = wx.Button(self, -1, "ShowHisto", (350, 30 - posv))
        showhisto_btn.Bind(wx.EVT_BUTTON, self.mainframe.ShowHisto_ConvolvedData)

        Recompute_btn = wx.Button(self, -1, "Compute Conv.", (220, 30 - posv))
        Recompute_btn.Bind(wx.EVT_BUTTON, self.mainframe.ComputeConvolvedData)

        tctxt = wx.StaticText(self, -1, "ThresholdConvolve", (5, 72 - posv))
        self.ThresholdConvolveCtrl = wx.SpinCtrl(self, -1, "1000", (135, 70 - posv), #(100, -1),
                                                                                min=0, max=9999)
        self.Bind(wx.EVT_SPINCTRL, self.mainframe.Show_ConvolvedImage, self.ThresholdConvolveCtrl)

        self.showconvolvedImage_btn = wx.ToggleButton(self, -1, "Show Conv. Image", (260, 70 - posv))
        self.showconvolvedImage_btn.Bind(wx.EVT_TOGGLEBUTTON, self.OnSwitchImageDisplay)

        self.TogglebtnState = 0  # show raw image

        self.Applythreshold = wx.CheckBox(self, -1, "Show thresholding", (5, 110 - posv))
        self.Applythreshold.SetValue(True)
        self.Applythreshold.Bind(wx.EVT_CHECKBOX, self.mainframe.Show_ConvolvedImage)

        vmaxtxt = wx.StaticText(self, -1, "Max. Intensity", (190, 110 - posv))
        self.vmaxctrl = wx.SpinCtrl(self, -1, "65000", #(290, 112 - posv),
                                                    min=2, max=9999)
        self.vmaxctrl.Bind(wx.EVT_SPINCTRL, self.mainframe.OnSpinCtrl_IminDisplayed)

        ittxt = wx.StaticText(self, -1, "Intensity Threshold (raw data)", (5, 152 - posv))
        ittxt2 = wx.StaticText(self, -1, "with respect to local background", (5, 172 - posv))
        self.IT = wx.SpinCtrl(self, -1, "500", (210, 150 - posv), #(100, -1),
                                                                min=0, max=9999)

        # tooltips ---------------------------
        pnrtp = "Minimum pixel distances between local maxima"
        pnrtxt.SetToolTipString(pnrtp)
        self.PNR.SetToolTipString(pnrtp)

        ittp = "Threshold level - RELATIVE to local background- above which local maxima must be found.\n"
        ittp += "Local background is a flat level given by the minimum intensity in a box centered on local maximum and with size 2*PixelNearRadius+1"
        ittxt.SetToolTipString(ittp)
        ittxt2.SetToolTipString(ittp)
        self.IT.SetToolTipString(ittp)

        tctp = "Threshold level -for image convolved by a gaussian kernel- above which local maxima must be found.\n"
        tctp += "Take care of larger scale of intensity"
        tctxt.SetToolTipString(tctp)
        self.ThresholdConvolveCtrl.SetToolTipString(tctp)

        showhisto_btn.SetToolTipString("Show histogram of the pixel intensity distribution of "
                                                        "image convoluted by a gaussian kernel")
        Recompute_btn.SetToolTipString("Compute or Recompute convolution of the raw image by a "
        "gaussian kernel.\n.Raw image is the current displayed image (raw or without background).")
        self.showconvolvedImage_btn.SetToolTipString("Display convolved image by a gaussian kernel")

        self.Applythreshold.SetToolTipString('Apply on the displayed convolved image the '
                                                        'threshold given by "thresholdConvolve".')

        vmaxtp = "Maximum displayed pixel intensity in convolved image."
        vmaxtxt.SetToolTipString(vmaxtp)
        self.vmaxctrl.SetToolTipString(vmaxtp)

    def OnSwitchImageDisplay(self, evt):
        """
        switch between raw and convolved image display (performed by MainPeakSearchFrame class)
        """
        self.TogglebtnState += 1

        self.TogglebtnState = self.TogglebtnState % 2

        if self.TogglebtnState == 1:  # now show conv image
            self.showconvolvedImage_btn.SetLabel("Show Image")
            self.mainframe.Show_ConvolvedImage(evt)

        elif self.TogglebtnState == 0:  # now show raw image
            self.showconvolvedImage_btn.SetLabel("Show Conv. Image")
            self.mainframe.dataimage_ROI_display = self.mainframe.dataimage_ROI
            self.mainframe.Show_Image(evt)

class findLocalMaxima_Meth_4(wx.Panel):
    """
    class of method 4 for local maxima search with skimage
    """

    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """
        """
        self.mainframe = parent.GetParent().GetParent()

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.methodnumber = 4

        mintxt = wx.StaticText(self, -1, "MinimumDistance")
        self.PNR = wx.SpinCtrl(self, -1, "10", #(100, -1),
                                            min=2, max=9999)
        ittxt = wx.StaticText(self, -1, "IntensityThreshold")
        self.IT = wx.SpinCtrl(self, -1, "15", #(100, -1),
                                            min=1, max=9999)
        bstxt = wx.StaticText(self, -1, "BoxSize")
        self.BS = wx.SpinCtrl(self, -1, "6", #(100, -1),
                                            min=1, max=9999)
        fittxt1 = wx.StaticText(self, -1, "FitOption")
        self.fitfunc_peak = wx.ComboBox(self, -1, "Gaussian_Strictbounds",
                                choices=["Gaussian_Strictbounds", "Gaussian_Relaxedbounds", "Gaussian_nobounds", "NoFit"], style=wx.CB_READONLY)
        modetxt2 = wx.StaticText(self, -1, "Mode")
        self.processMode = wx.ComboBox(self, -1, "single_CPU",
                                choices=["single_CPU", "multiprocessing"], style=wx.CB_READONLY)

        # layout
        h1 = wx.BoxSizer(wx.HORIZONTAL)
        h1.Add(mintxt, 0 , wx.ALL, 5)
        h1.Add(self.PNR, 0 , wx.ALL, 5)

        h2 = wx.BoxSizer(wx.HORIZONTAL)
        h2.Add(ittxt,0 , wx.ALL, 5)
        h2.Add(self.IT, 0 , wx.ALL, 5)
        
        h3 = wx.BoxSizer(wx.HORIZONTAL)
        h3.Add(bstxt,0 , wx.ALL, 5)
        h3.Add(self.BS, 0 , wx.ALL, 5)
        
        h4 = wx.BoxSizer(wx.HORIZONTAL)
        h4.Add(fittxt1, 0 , wx.ALL, 5)
        h4.Add(self.fitfunc_peak, 0 , wx.ALL, 5)
        
        h5 = wx.BoxSizer(wx.HORIZONTAL)
        h5.Add(modetxt2, 0 , wx.ALL, 5)
        h5.Add(self.processMode, 0 , wx.ALL, 5)
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(h1,0, wx.EXPAND,10)
        vbox.Add(h2,0, wx.EXPAND,5)
        vbox.Add(h3,0, wx.EXPAND,5)
        vbox.Add(h4,0, wx.EXPAND,5)
        vbox.Add(h5,0, wx.EXPAND,5)
        self.SetSizer(vbox)

        # tooltips
        mintp = "Minimum pixel distances between local maxima"
        mintxt.SetToolTipString(mintp)
        self.PNR.SetToolTipString(mintp)

        ittp = "Threshold level above which local maxima must be found (usually between 1-5)"
        ittxt.SetToolTipString(ittp)
        self.IT.SetToolTipString(ittp)
        
        bstp = "Box size for fitting Gaussians to local maximas"
        bstxt.SetToolTipString(bstp)
        self.BS.SetToolTipString(bstp)
        
class FitParametersPanel(wx.Panel):
    def __init__(self, parent):
        """
        method 1 parameters for method #0 for local maxima search + fit(intensity threshold)

        parent must have granparent (call of parent.GetParent())

        this granparent must have following attributes and methods:
        OnPeakSearch()
        GetParent().CCDlabel
        """

        self.granparent = parent.GetParent()
        self.parent = parent

        self.methodnumber = 5 

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        txt0 = wx.StaticText(self, -1, "General parameters")
        txt0.SetFont(font3)

        maxnbtxt = wx.StaticText(self, -1, "Max. Nb of Fits")
        self.NbMaxFits = wx.TextCtrl(self, -1, "2000")
        #        self.peaksearchComments = wx.TextCtrl(self, -1, '', (100, 105))

        self.onWholeImage = wx.CheckBox(self, -1, "Apply on whole Image")
        self.onWholeImage.SetValue(True)
        self.onWholeImage.Disable()

        usetxt = wx.StaticText(self, -1, "Use peak position")
        self.keepHottestPixel = wx.RadioButton(self, -1, "Hottest Pixel")
        self.keepCentroid = wx.RadioButton(self, -1, "Centroid")

        txt1 = wx.StaticText(self, -1, "Fitting parameters")
        txt1.SetFont(font3)

        fittxt = wx.StaticText(self, -1, "FitFunction")
        self.fitfunc = wx.ComboBox(self, -1, "Gaussian",
                                choices=["Gaussian", "Lorentzian", "NoFit"], style=wx.CB_READONLY)
        #        self.fitfunc.Bind(wx.EVT_COMBOBOX, self.OnfitfuncChange)

        xtoltxt = wx.StaticText(self, -1, "xtol")
        self.xtol = wx.TextCtrl(self, -1, "0.001")

        boxtxt = wx.StaticText(self, -1, "Boxsize")
        self.boxsize = wx.SpinCtrl(self, -1, "15", #size=(100,-1),
                                            min=1, max=9999)

        peaksizetxt = wx.StaticText(self, -1, "Guessed Peak size")
        self.peaksizectrl = wx.TextCtrl(self, -1, "0.9")

        # rejection
        txt2 = wx.StaticText(self, -1, "Rejection parameters")
        txt2.SetFont(font3)

        saturation_value = DictLT.dict_CCD[self.granparent.GetParent().CCDlabel][2]

        pixdevtxt = wx.StaticText(self, -1, "Max. Deviation (pixel)")
        self.FitPixelDev = wx.TextCtrl(self, -1, "6.0")

        maxItxt = wx.StaticText(self, -1, "Peak Intensity    Max: ")
        self.maxIctrl = wx.TextCtrl(self, -1, str(int(saturation_value) - 1))

        minItxt = wx.StaticText(self, -1, "Min:")
        self.minIctrl = wx.TextCtrl(self, -1, "0")

        peaksizemaxtxt = wx.StaticText(self, -1, "Peak size   Max :")
        self.peaksizemaxctrl = wx.TextCtrl(self, -1, "3.0")

        peaksizemintxt = wx.StaticText(self, -1, "Min :")
        self.peaksizeminctrl = wx.TextCtrl(self, -1, "0.65")

        self.initbuttons()

        # layout
        HBoxsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer1.Add(usetxt, 0, wx.ALL, 5)
        HBoxsizer1.Add(self.keepHottestPixel, 0, wx.ALL, 5)
        HBoxsizer1.Add(self.keepCentroid, 0, wx.ALL, 5)

        HBoxsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer2.Add(fittxt, 0, wx.ALL, 5)
        HBoxsizer2.Add(self.fitfunc, 0, wx.ALL, 5)
        HBoxsizer2.Add(xtoltxt, 0, wx.ALL, 5)
        HBoxsizer2.Add(self.xtol, 0, wx.ALL, 5)

        HBoxsizer3 = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer3.Add(boxtxt, 0, wx.ALL, 5)
        HBoxsizer3.Add(self.boxsize, 0, wx.ALL, 5)
        HBoxsizer3.Add(peaksizetxt, 0, wx.ALL, 5)
        HBoxsizer3.Add(self.peaksizectrl, 0, wx.ALL, 5)

        HBoxsizer4 = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer4.Add(pixdevtxt, 0, wx.ALL, 5)
        HBoxsizer4.Add(self.FitPixelDev, 0, wx.ALL, 5)
        HBoxsizer4b = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer4b.Add(maxItxt, 0, wx.ALL, 5)
        HBoxsizer4b.Add(self.maxIctrl, 0, wx.ALL, 5)
        HBoxsizer4b.Add(minItxt, 0, wx.ALL, 5)
        HBoxsizer4b.Add(self.minIctrl, 0, wx.ALL, 5)
        HBoxsizer4c = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer4c.Add(peaksizemaxtxt, 0, wx.ALL, 5)
        HBoxsizer4c.Add(self.peaksizemaxctrl, 0, wx.ALL, 5)
        HBoxsizer4c.Add(peaksizemintxt, 0, wx.ALL, 5)
        HBoxsizer4c.Add(self.peaksizeminctrl, 0, wx.ALL, 5)

        HBoxsizer5 = wx.BoxSizer(wx.HORIZONTAL)
        HBoxsizer5.Add(maxnbtxt, 0, wx.ALL, 5)
        HBoxsizer5.Add(self.NbMaxFits, 0, wx.ALL, 5)
        HBoxsizer5.Add(self.onWholeImage, 0, wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt0, 0, wx.EXPAND)
        vbox.Add(HBoxsizer5, 0, wx.EXPAND)

        vbox.Add(HBoxsizer1, 0, wx.EXPAND)
        vbox.Add(txt1, 0, wx.EXPAND)
        vbox.Add(HBoxsizer2, 0, wx.EXPAND)
        vbox.Add(HBoxsizer3, 0, wx.EXPAND)
        vbox.Add(txt2, 0, wx.EXPAND)
        vbox.Add(HBoxsizer4, 0, wx.EXPAND)
        vbox.Add(HBoxsizer4b, 0, wx.EXPAND)
        vbox.Add(HBoxsizer4c, 0, wx.EXPAND)

        self.SetSizer(vbox)

        # tooltips
        usetp = "Select the intial guessed peak pixel position from results of local maxima search.\n"
        usetp += "Hottest Pixel: (integers) pixels position of the highest pixel close the local maxima.\n"
        usetp += "Centroid: (floats) pixels position of the center of mass of blob (local cluster of hot pixels)."

        usetxt.SetToolTipString(usetp)
        self.keepHottestPixel.SetToolTipString(usetp)
        self.keepCentroid.SetToolTipString(usetp)

        fittp = "Select the 2D peak shape model"
        fittxt.SetToolTipString(fittp)
        self.fitfunc.SetToolTipString(fittp)

        boxtp = "HALF size (in pixel) of the squared box length to be fitted by a peak model"
        boxtxt.SetToolTipString(boxtp)
        self.boxsize.SetToolTipString(boxtp)

        pixdevtp = 'Pixel deviation "FitPixDev" threshold above which a single fitted peak is '
        'rejected (will not belong to the final peaks list).\n'
        pixdevtp += "Pixel deviation is the pixel distance between peak position result and "
        "starting initial guessed position given by one of the three local maxima "
        "search methods and radio button choice (Hottest pixel or Centroid) "
        "if using Convolution Method."
        pixdevtxt.SetToolTipString(pixdevtp)
        self.FitPixelDev.SetToolTipString(pixdevtp)

        maxnbtp = "Maximum number of local maxima to be fitted"
        maxnbtxt.SetToolTipString(maxnbtp)
        self.NbMaxFits.SetToolTipString(maxnbtp)

        tippeaksize = "Guess peak size in pixel"
        peaksizetxt.SetToolTipString(tippeaksize)
        self.peaksizectrl.SetToolTipString(tippeaksize)

    def OnStart(self, evt):

        self.granparent.OnPeakSearch(evt)

    def initbuttons(self,):
        # state resulting from the type of result of page3 (first local maxima method)
        self.keepHottestPixel.Disable()
        self.keepCentroid.Disable()
        self.keepHottestPixel.SetValue(True)


DICT_FIELDS_SIZE = {"intensity": 100,
                    "pixX": 100,
                    "pixY": 100,
                    "spotindex": 60,
                    "grainindex": 60,
                    "H": 60,
                    "K": 60,
                    "L": 60,
                    "energy": 80,
                    "twotheta": 80,
                    "chi": 80,
                    "MatchingRate": 30,
                    "NbindexedSpots": 20,
                    "peak_amplitude": 100,
                    "peak_background": 80,
                    "peak_width1": 50,
                    "peak_width2": 50,
                    "peak_inclin": 45,
                    "PixDev_x": 50,
                    "PixDev_y": 50,
                    "PixMax": 100,
                    "XfitErr": 75,
                    "YfitErr": 75,
                    "distfitErr": 75}

DICT_FIELDS_ALIGN = {"intensity": "left",
                    "pixX": "left",
                    "pixY": "left",
                    "spotindex": "center",
                    "grainindex": "center",
                    "H": "left",
                    "K": "left",
                    "L": "left",
                    "energy": "left",
                    "twotheta": "left",
                    "chi": "left",
                    "MatchingRate": "left",
                    "NbindexedSpots": "left",
                    "peak_amplitude": "left",
                    "peak_background": "left",
                    "peak_width1": "left",
                    "peak_width2": "left",
                    "peak_inclin": "left",
                    "PixDev_x": "left",
                    "PixDev_y": "left",
                    "PixMax": "left",
                    "XfitErr": "left",
                    "YfitErr": "left",
                    "distfitErr": "left"}

LIST_OF_FIELDS_DATAPEAK = ["pixX",
                            "pixY",
                            "peak_amplitude",
                            "peak_width1",
                            "peak_width2",
                            "peak_inclin",
                            "PixDev_x",
                            "PixDev_y",
                            "peak_background",
                            "PixMax"]


class PeakListOLV(wx.Panel):
    """
    panel embedding an ObjectListViewer from ObjectListView module

    need of:
    self.grangranparent.peaklistPixels
    self.grangranparent.onRemovePeaktoPeaklist
    self.grangranparent.OnReplot
    self.grangranparent.framedim
    and lot of other things with mainframe...
    """
    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """init"""
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY, size=(-1, 100))
        #        scrolled.ScrolledPanel.__init__(self, parent, -1)

        #        print self.GetId()

        self.parent = parent  # notebook
        self.granparent = parent.GetParent()  # MainPeakSearchFrame.panel
        self.grangranparent = self.granparent.GetParent()

        self.mainframe = self.grangranparent

        self.methodnumber = 6

        self.listofdata = None
        self.listoffields = LIST_OF_FIELDS_DATAPEAK

        self.focuseditem = None

        self.allspots = None
        self.list_selectedspots = None

        # widgets ----------------------------------
        self.updatepeaklist = wx.CheckBox(self, -1, "update peaks list", style=wx.ALIGN_LEFT)
        self.updatepeaklist.SetValue(True)
        self.updatepeaklist.Bind(wx.EVT_CHECKBOX, self.updateView)

        self.useselection = wx.Button(self, -1, "Remove Peak")
        self.useselection.Bind(wx.EVT_BUTTON, self.OnRemove)

        self.btnremoveall = wx.Button(self, -1, "Purge List")
        self.btnremoveall.Bind(wx.EVT_BUTTON, self.OnRemoveAll)

        self.showROIpeak = wx.CheckBox(self, -1, "show peak", style=wx.ALIGN_LEFT)
        self.showROIpeak.Bind(wx.EVT_CHECKBOX, self.OnShowPeak)

        #        self.showspecificgrains = wx.CheckBox(self, -1, "show spots of grains", style=wx.ALIGN_LEFT)
        #        self.showspecificgrains.Disable()
        #        self.showspecificgrains.Bind(wx.EVT_CHECKBOX, self.OnShowGrainsSpots)

        self.boxsizetxt = wx.StaticText(self, -1, "boxsize(x,y)")
        self.boxsizex = wx.TextCtrl(self, -1, "20", style=wx.ALIGN_LEFT)
        self.boxsizey = wx.TextCtrl(self, -1, "50", style=wx.ALIGN_LEFT)

        # layout -------------------------------------
        sizerh1 = wx.BoxSizer(wx.HORIZONTAL)
        sizerh2 = wx.BoxSizer(wx.HORIZONTAL)

        sizerh1.Add(self.updatepeaklist, 0, 0)
        sizerh1.Add(self.useselection, 0, 0)
        sizerh1.Add(self.btnremoveall, 0, 0)
        sizerh2.Add(self.showROIpeak, 0, 0)
        sizerh2.Add(self.boxsizetxt, 0, 0)
        sizerh2.Add(self.boxsizex, 0, 0)
        sizerh2.Add(self.boxsizey, 0, 0)

        #        self.buildlistofspots()

        self.myOlv = ObjectListView(self, -1, style=wx.LC_REPORT | wx.SUNKEN_BORDER)
        self.myOlv.Bind(wx.EVT_LIST_ITEM_FOCUSED, self.OnClickRow)
         
        self.myOlv.EnableSorting()

        if self.mainframe.peaklistPixels is not None:
            self.updateView(1)

        #        self.InitObjectListView()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizerh1, 0, wx.ALL, 5)
        sizer.Add(sizerh2, 0, wx.ALL, 5)
        sizer.AddSpacer(5)
        sizer.Add(self.myOlv, 1, wx.EXPAND)

        self.SetSizer(sizer)

    def buildlistofspots(self):
        flag = False
        if self.grangranparent.peaklistPixels is None:
            wx.MessageBox("Peak List is empty", "INFO")
            return flag

        if self.updatepeaklist.GetValue():
            # numpy array of datapeak
            self.listofdata = self.grangranparent.peaklistPixels
            # list of spot objects
            if self.listofdata.shape[1]==12:
                self.listoffields.append('XfitErr')
                self.listoffields.append('YfitErr')
                self.listoffields.append('distfitErr')
                coldistfierr = np.sqrt(self.listofdata[:,10]**2+self.listofdata[:,11]**2)
                self.listofdata = np.c_[self.listofdata,coldistfierr]
            self.allspots = SpotModel.GetAllspots(self.listofdata, self.listoffields)

            flag = True

        return flag

    def AddOneSpot(self, datapeak):
        spotobject = SpotModel.Spot(datapeak, self.listoffields)
        self.myOlv.AddObject(spotobject)

    def RemoveOneSpot(self, peakXY):
        RADIUS = 1.0
        X, Y = peakXY
        listobjects = self.myOlv.GetObjects()
        ind = -1
        for k, obj in enumerate(listobjects):
            if (obj.pixX - X) ** 2 + (obj.pixY - Y) ** 2 < RADIUS ** 2:
                ind = k
                break

        if ind != -1:

            #        selectedobject_filter = self.myOlv.GetObjects()[indexObject]
            #        print "listobjects", listobjects
            #        print self.myOlv.GetItemText(indexObject)
            self.myOlv.RemoveObject(obj)

    #        self.myOlv.RemoveObject(selectedobject_filter)
    #        print "self.myOlv.GetObjectAt(indexObject)", self.myOlv.GetObjectAt(indexObject)
    #        self.myOlv.RemoveObject(self.myOlv.GetObjectAt(indexObject))

    def InitObjectListView(self):

        if self.allspots is not None:

            # columns definition
            coldef = [ColumnDefn(field, DICT_FIELDS_ALIGN[field], DICT_FIELDS_SIZE[field], field)
                for field in self.listoffields]
            
            
            self.myOlv.SetColumns(coldef)
            self.myOlv.SetObjects(self.allspots)

            # to color row

    #            self.myOlv.rowFormatter = self.rowFormatter

    #    def rowFormatter(self, listItem, spot):
    #        if spot.grainindex <= -1:
    #            listItem.SetBackgroundColour((225, 0, 0, 0.5))
    #        elif spot.grainindex == 0:
    #            listItem.SetBackgroundColour(wx.GREEN)
    #        elif spot.grainindex >= 1:
    #            listItem.SetBackgroundColour((0, int(255 * (1 - spot.grainindex / 9.)), 0))

    def updateView(self, _):
        flag = self.buildlistofspots()
        if flag:
            self.InitObjectListView()
            #            if not self.showROIpeak.GetValue() and not self.showspecificgrains.GetValue():
            if not self.showROIpeak.GetValue():
                self.myOlv.SetFilter(None)

    def OnRemove(self, _):
        """ remove one element corresponding to the current highlighted row
        """
        if self.grangranparent.peaklistPixels is None:
            wx.MessageBox("Peak List is empty", "INFO")
            return None
        #        print dir(self.myOlv)
        selectedRowindices = self.getDataSelected()

        print("selectedRowindices", selectedRowindices)

        peakXY = selectedRowindices[0][:2]

        self.grangranparent.onRemovePeaktoPeaklist(1, centerXY=peakXY)

    def OnRemoveAll(self, _):

        print("self.peaklistPixels before", self.grangranparent.peaklistPixels)
        self.grangranparent.peaklistPixels = None

        self.myOlv.RemoveObjects(self.myOlv.GetObjects())
        #         self.buildlistofspots()

        # remove marker on image
        #         self.plotPeaks = True
        self.grangranparent.OnReplot(1)

    #         self.plotPeaks = False

    #         listpeaks = list(self.grangranparent.peaklistPixels)
    #         while len(listpeaks)>=1:
    #             peak=listpeaks.pop()
    #             self.grangranparent.onRemovePeaktoPeaklist(1, centerXY=peak)
    #         print "self.peaklistPixels after",self.grangranparent.peaklistPixels

    def OnClickRow(self, _):

        newitemfocused = False
        selectitem = self.myOlv.GetFocusedItem()
        if selectitem != self.focuseditem:
            newitemfocused = True
            self.focuseditem = selectitem
        if selectitem == -1:
            return

        # name = self.myOlv.GetItemText(selectitem)
        #        print "hobby", self.myOlv.GetObjectAt(self.myOlv.GetFocusedRow())
        #        print "name", name
        #        print "selectitem", selectitem

        if self.showROIpeak.GetValue():
            # crop image on peak
            self.OnShowPeak(1)
        else:
            if newitemfocused:
                # remove the previous drawn rectangle
                self.mainframe.RemoveLastRectangle()
            # add marker around peak
            self.OnCrossToSeePeak(1)

    def getDataSelected(self):
        self.list_selectedspots = self.myOlv.GetSelectedObjects()

        print("self.list_selectedspots", self.list_selectedspots)
        print("focus on ", self.myOlv.GetFocusedRow())
        allselected = []
        for sspot in self.list_selectedspots:
            listval = []
            for field in self.listoffields:
                listval.append(sspot.__getattribute__(field))

            allselected.append(listval)

        return np.array(allselected)

    def getObjectAtRow(self, _):
        return self.myOlv.GetObjectAt(self.myOlv.GetFocusedRow())

    def getXYatRow(self):
        selected_spot = self.getObjectAtRow(1)
        listval = []
        for field in self.listoffields:
            try:
                listval.append(selected_spot.__getattribute__(field))
            except AttributeError:
                continue

        return listval[:2]

    def getXYfromClickOnRow(self):
        dataselected = self.getDataSelected()
        if len(dataselected.shape) == 1:
            # single selection
            return dataselected[:2]
        else:
            return dataselected[-1, :2]

    def OnCrossToSeePeak(self, _):

        X, Y = self.getXYatRow()

        #        self.mainframe.reinit_aftercrop_draw()
        self.mainframe.addPatchRectangle(X, Y)

        #        self.mainframe.updatePlotTitle()
        self.mainframe.canvas.draw()

    def OnShowPeak(self, _):

        if self.showROIpeak.GetValue():

            X, Y = self.getXYatRow()

            #            self.mainframe.toggleBtnCrop.SetLabel("UnCrop Data")
            self.mainframe.CropIsOn = True
            self.mainframe.centerx, self.mainframe.centery = int(Y), int(X)
            self.mainframe.boxx, self.mainframe.boxy = (int(self.boxsizey.GetValue()),
                                                        int(self.boxsizex.GetValue()))

            centeri, centerj = self.mainframe.centerx, self.mainframe.centery
            boxi, boxj = self.mainframe.boxx, self.mainframe.boxy

            imin, imax, jmin, jmax = (centeri - boxi, centeri + boxi, centerj - boxj, centerj + boxj)

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax, jmin, jmax,
                                                            framedim=self.grangranparent.framedim)

            self.mainframe.dataimage_ROI_display = self.mainframe.dataimage_ROI[imin:imax, jmin:jmax]
            self.mainframe.reinit_aftercrop_draw()

            self.mainframe.updatePlotTitle()
            self.mainframe.canvas.draw()

        else:
            self.mainframe.CropIsOn = False
            self.mainframe.dataimage_ROI_display = self.mainframe.dataimage_ROI
            self.mainframe.reinit_aftercrop_draw()
            self.mainframe.plotPeaks = True
            self.mainframe.addPeaksMarker()
            self.mainframe.plotPeaks = False

            self.mainframe.updatePlotTitle()
            self.mainframe.canvas.draw()


class WriteFileBoxINFOBox(wx.Frame):
    """ class GUI allow to show info and selectable path to files 
    the layout is the following vertically :
    headertext, msg1, filepath1, [msg2,filepath2]
    """
    def __init__(self, parent, _id, title, headertext, msg1, filepath1, msg2='',filepath2=''):
        wx.Frame.__init__(self, parent, _id, title, size=(800,200))
        self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )

        self.userfilename = None
        
        header = wx.StaticText(self, -1, headertext)
        header.SetFont(wx.Font(12, wx.MODERN, wx.NORMAL, wx.BOLD))
        self.path1ctrl = wx.TextCtrl(self, size=(800, 40))#, style=wx.TE_MULTILINE)
        self.path1ctrl.SetValue(filepath1)

        btna = wx.Button(self, wx.ID_OK, "OK", size=(150, 40))
        btna.Bind(wx.EVT_BUTTON, self.OnAccept)
        btna.SetDefault()

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(header, 0)
        vbox.Add(wx.StaticText(self, -1, msg1), 0)
        vbox.Add(self.path1ctrl, 0, wx.EXPAND)
        if msg2 != '':
            path2ctrl = wx.TextCtrl(self, size=(800, 40))
            path2ctrl.SetValue(filepath2)
            vbox.Add(wx.StaticText(self, -1, msg2), 0)
            vbox.Add(path2ctrl, 0, wx.EXPAND)
        vbox.Add(btna, 0, wx.EXPAND)
        self.SetSizer(vbox)

    def OnAccept(self, _):
        self.userfilename = self.path1ctrl.GetValue()
        self.Close()

class MainPeakSearchFrame(wx.Frame):
    """
    Class to show CCD frame pixel intensities
    and provide tools for searching peaks
    """
    def __init__(self, parent, _id, _initialParameter, title, size=4):
        wx.Frame.__init__(self, parent, _id, title, size=(600, 1000))

        self.initialParameter = _initialParameter

        self.title = self.initialParameter["title"]
        self.imagefilename = self.initialParameter["imagefilename"]
        self.dirname = self.initialParameter["dirname"]
        self.LastLUT = self.initialParameter["mapsLUT"]
        self.writefolder = self.initialParameter["dirname"]
        self.CCDlabel = self.initialParameter["CCDLabel"]
        # for stacked images in hdf5 file
        self.stackedimages = self.initialParameter["stackedimages"]
        self.stackimageindex = self.initialParameter["stackimageindex"]
        self.Nbstackedimages = self.initialParameter["Nbstackedimages"]
        
        self.nbdigits = 4   # sCMOS with bliss

        (self.framedim, self.pixelsize, self.saturationvalue, self.fliprot, self.headeroffset,
                self.dataformat, self.comments,
                self.file_extension, ) = DictLT.dict_CCD[self.CCDlabel]

        self.figsize = size

        self.peaklistPixels = None  # peaklist results of peaksearch
        self.last_peakfit_result = None
        self.file_index_increment = 0

        self.plotPeaks = False
        self.ROIs = {}
        self.roiindex = 0

        self.dict_param = None

        # loading LUTS
        self.mapsLUT = [m for m in pcm.datad if not m.endswith("_r")]
        self.mapsLUT.sort()

        self.BImageFilename = ""

        self.centerx, self.centery = None, None
        self.currentROIpatch = None
        self.ROIRectangleselected = None
        self.steppresent = None
        self.stepmissing = None

        self.stepindex = None
        self.imageindex = None
        self.justcheckedShowValues = None
        self.boxx = None
        self.boxy = None
        self.myplot = None
        self.dataimage_ROI = None
        self.FileDialog = None
        self.Bdirname = None
        self.BlackListFilename = None
        self.DetFilename = None
        self.IminDisplayed = None
        self.ImaxDisplayed = None
        self.lastinfo = None
        self.boxsize_fit = None
        self.guessed_amplitude = None
        self.guessed_bkg = None
        self.position_definition = None
        self.dict_param_LocalMaxima = None

        # read data
        self.currentime = time.time()
        self.getIndex_fromfilename()
        self.CropIsOn = False
        self.imin_crop, self.jmin_crop = 0, 0
        self.imax_crop, self.jmax_crop = None, None
        self.read_data()
        # initial displayed data(no background substraction)
        self.dataimage_ROI_display = self.dataimage_ROI
        self.current_data_display = "Raw Image"
        self.ConvolvedData = None
        # panels
        self.nb0 = None
        self.viewingLUTpanel = None
        self.ImageFilterpanel = None
        self.ImagesBrowser = None
        self.Monitor = None
        if WXPYTHON3:
            self.RoiSelector = None

        self.font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        # default method and buttons layout
        self.firstdisplay = 1
        self.method = 3  # index of method

        self.paramsHat = (4, 5, 2)
        self.largehollowcircles = []
        self.smallredcircles = []

        self.CCDcalib = None
        self.gettime()

        self.OnFlyMode = False

        # widgets and layout
        self.createMenuBar()
        self.sb = self.CreateStatusBar()
        self.create_main_panel()
        self.init_figure_draw()

    def createMenuBar(self):
        menubar = wx.MenuBar()

        filemenu = wx.Menu()

        menuCCDfileparams = filemenu.Append(5896, "CCD File parameters",
                                        "Open a board to choose paramters to read CCD image file")
        filemenu.AppendSeparator()
        openimagemenu = filemenu.Append(wx.ID_ANY, "Open Image", "Open binary CCD image file")
        filemenu.AppendSeparator()
        menuFileSeries = filemenu.Append(wx.ID_ANY, "File Series",
                                                        "Launch the File Series Peak Search Board")

        self.Bind(wx.EVT_MENU, self.OpenImage, openimagemenu)
        self.Bind(wx.EVT_MENU, self.OnSetFileCCDParam, menuCCDfileparams)
        self.Bind(wx.EVT_MENU, self.OnFileSeries, menuFileSeries)

        filemenu.Enable(id=5896, enable=False)

        preferences = wx.Menu()
        menuSetPreference = preferences.Append(wx.ID_ANY, "PeakList Folder",
                                                        "Set folder to write peaklist .dat files")
        menudisplayprops = preferences.Append(wx.ID_ANY, "Set Plot Size",
                                        "Set Minimal plot size to fit with small computer screen")
        self.Bind(wx.EVT_MENU, self.OnFolderPreferences, menuSetPreference)
        self.Bind(wx.EVT_MENU, self.OnSetPlotSize, menudisplayprops)

        helpmenu = wx.Menu()

        menuAbout = helpmenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        menuExit = helpmenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Set events.
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

        menubar.Append(filemenu, "&File")
        menubar.Append(preferences, "&Preferences")
        menubar.Append(helpmenu, "&Help")

        self.SetMenuBar(menubar)

    def create_main_panel(self):
        """ Creates the main panel with all the controls on it:
             * mpl canvas
             * mpl navigation toolbar
             * Control panel for interaction
        """
        self.panel = wx.Panel(self)

        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.fig.set_size_inches(self.figsize, self.figsize, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)
        self.bbox = (0, 200, 0, 200)

        self.canvas.mpl_connect("motion_notify_event", self.mouse_move)
        
        if 0:
            if WXPYTHON4:
                self.Bind(wx.EVT_PAINT, self.OnPaint)
            else:
                wx.EVT_PAINT(self, self.OnPaint)

        #         self.tooltip = wx.ToolTip(tip='tip with a long %s line and a newline\n' % (' ' * 100))
        self.tooltip = wx.ToolTip(tip="Welcome on peaksearch board of LaueTools")
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)
        self.fig.canvas.mpl_connect("button_press_event", self.onClick)
        self.fig.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.fig.canvas.mpl_connect("pick_event", self.onPick)

        if WXPYTHON3:
            # drawtype is 'box' or 'line' or 'none'
            self.RS = RectangleSelector(self.axes,
                                        self.line_select_callback,
                                        #drawtype="box",
                                        useblit=True,
                                        button=[1, 2, 3],  # don't use middle button
                                        minspanx=5,
                                        minspany=5,
                                        spancoords="pixels",
                                        interactive=True)
            self.RS.set_active(False)

        self.toolbar = NavigationToolbar(self.canvas)

        self.numvalues_chck = wx.CheckBox(self.panel, -1, "Show Values", size=(-1, 40))
        self.numvalues_chck.SetValue(False)
        # flag when asked to draw values
        self.justcheckedShowValues = False
        self.numvalues_chck.Bind(wx.EVT_CHECKBOX, self.OnCheckPlotValues)

        # notebook on top
        self.toplayout2()

        self.nb = wx.Notebook(self.panel, -1, style=0)

        self.localfitbtn = wx.Button(self.panel, -1, "Fit Peak", size=(-1, 40))
        self.AddPeak = wx.Button(self.panel, -1, "Add Peak", size=(-1, 40))
        self.RemovePeak = wx.Button(self.panel, -1, "Remove Peak", size=(-1, 40))
        self.RemoveAllPeaksbtn = wx.Button(self.panel, -1, "Remove All Peaks", size=(-1, 40))

        self.localfitbtn.Bind(wx.EVT_BUTTON, self.onFitOnePeak)
        self.AddPeak.Bind(wx.EVT_BUTTON, self.onAddPeaktoPeaklist)
        self.RemovePeak.Bind(wx.EVT_BUTTON, self.onRemovePeaktoPeaklist)
        self.RemoveAllPeaksbtn.Bind(wx.EVT_BUTTON, self.onRemoveAllPeakstoPeaklist)

        self.plot_singlefitresults_chck = wx.CheckBox(self.panel, -1, "plot single fit results",
                                                                                    size=(-1, 40))
        self.plot_singlefitresults_chck.SetValue(True)

        self.allways_addpeak_chck = wx.CheckBox(self.panel, -1, "auto. add peak", size=(-1, 40))
        self.allways_addpeak_chck.SetValue(True)

        self.btnOpenPeakList = wx.Button(self.panel, -1, "Open PeakLists", size=(-1, 40))
        self.btnOpenPeakList.Bind(wx.EVT_BUTTON, self.onOpenPeakListBoard)

        startbutton = wx.Button(self.panel, 2, "Search All Peaks", size=(-1, 40))
        startbutton.SetFont(wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD))
        startbutton.Bind(wx.EVT_BUTTON, self.OnPeakSearch)
        savepeaklistbutton = wx.Button(self.panel, 2, "Save PeakListc", size=(-1, 40))
        savepeaklistbutton.Bind(wx.EVT_BUTTON, self.SavePeakList_PSPfile)

        savelistroibtn = wx.Button(self.panel, -1, "Save ROIs from Peaks", size=(-1,50))
        savelistroibtn.Bind(wx.EVT_BUTTON, self.onSaveROIsList)

        quitbutton = wx.Button(self.panel, 3, "Quit", size=(-1, 40))
        quitbutton.Bind(wx.EVT_BUTTON, self.OnQuit)

        #        self.page1 = PeakListPanel(nb)
        #        self.page1 = PeakListOLV(nb)
        self.page1 = findLocalMaxima_Meth_1(self.nb)
        self.page2 = findLocalMaxima_Meth_2(self.nb)
        self.page3 = findLocalMaxima_Meth_3(self.nb)
        self.page4 = findLocalMaxima_Meth_4(self.nb)
        self.fitparampanel = FitParametersPanel(self.nb)
        if ObjectListView_Present:
            self.pagepeaknav = PeakListOLV(self.nb)  # PeakListOLV

        self.nb.AddPage(self.page1, "1_Threshold")
        self.nb.AddPage(self.page2, "1_ArrayShift")
        self.nb.AddPage(self.page3, "1_Convolution")
        self.nb.AddPage(self.page4, "1_skimage")
        self.nb.AddPage(self.fitparampanel, "2_FitParams")
        if ObjectListView_Present:
            self.nb.AddPage(self.pagepeaknav, "PeakNavigator")

        # TODO bind with self.Show_Image ?
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnTabChange_PeakSearchMethod)
        #        self.nb.GetPosition()

        # LAYOUT -------------------------------------------
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        vbox.Add(self.toolbar, 0, wx.EXPAND)

        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer.Add(self.localfitbtn, 0, wx.ALL, 5)
        btnSizer.AddSpacer(60)
        btnSizer.Add(self.AddPeak, 0, wx.ALL, 5)
        btnSizer.Add(self.RemovePeak, 0, wx.ALL, 5)
        btnSizer.Add(self.RemoveAllPeaksbtn, 0, wx.ALL, 5)
        btnSizer.AddSpacer(60)
        btnSizer.Add(startbutton, 0, wx.ALL, 5)
        btnSizer.Add(savepeaklistbutton, 0, wx.ALL, 5)
        btnSizer.Add(savelistroibtn, 0, wx.ALL, 5)
        btnSizer.AddSpacer(15)
        btnSizer.Add(quitbutton, 0, wx.ALL, 5)

        btnSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer2.Add(self.plot_singlefitresults_chck, 0, wx.ALL)
        btnSizer2.Add(self.allways_addpeak_chck, 0, wx.ALL)
        btnSizer2.Add(self.numvalues_chck, 0, wx.ALL)
        btnSizer2.Add(self.btnOpenPeakList, 0, wx.ALL)

        vbox2 = wx.BoxSizer(wx.VERTICAL)
        vbox2.Add(self.nb0, 1, wx.EXPAND, 0)
        vbox2.Add(self.nb, 1, wx.EXPAND, 0)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(vbox, 1, wx.EXPAND)
        hbox.Add(vbox2, 1, wx.EXPAND)

        vboxgeneral = wx.BoxSizer(wx.VERTICAL)
        vboxgeneral.Add(hbox, 1, wx.EXPAND)
        vboxgeneral.Add(btnSizer2, 0, wx.EXPAND, 0)
        vboxgeneral.Add(btnSizer, 0, wx.EXPAND, 0)

        self.panel.SetSizer(vboxgeneral)
        vboxgeneral.Fit(self)
        self.Layout()

        # tooltips
        self.localfitbtn.SetToolTipString('Fit peak position close the last clicked point in '
                        'image (or press "f").\n ROI size is set by the boxsize in fitparams tab')
        self.AddPeak.SetToolTipString("Add single fitted peak in current list of peaks")
        self.RemovePeak.SetToolTipString('Remove nearest peak from clicked position on plot '
                                            '(or press "r").')

        self.plot_singlefitresults_chck.SetToolTipString("Display data pixel intensities and "
                                                        "fitted gaussian peak.")
        self.allways_addpeak_chck.SetToolTipString("Always add the fitted peak to the current "
                                                    "list of peaks.")
        self.numvalues_chck.SetToolTipString("Draw numerical data values if field of view is "
                                            "smaller than 25 pixels")

        allpeaks_tp = "Search all peaks in CURRENT DISPLAYED image.\n"
        allpeaks_tp += "1- Find local maxima by using one of the three methods\n"
        allpeaks_tp += "(Threshold, ArrayShift, Convolution) whose parameters are defined in "
        "respective tabs\n"
        allpeaks_tp += "2- Fit all local maxima (or not) by 2D shaped intensity peak model. "
        "Parameters are defined in FitParams tab\n"

        startbutton.SetToolTipString(allpeaks_tp)

        self.btnOpenPeakList.SetToolTip(wx.ToolTip("Select and plot peaks contained in .dat or .fit file"))

        savepeaklistbutton.SetToolTipString("Save current peaks list in a file (with incremented name)")
        savelistroibtn.SetToolTipString('Save current peaks list as a list of ROI with current boxsize used in fitparams')

        self.page1.SetToolTipString("Guess initial peaks positions for peak refinement by "
        "a basic image thresholding")
        self.page2.SetToolTipString("Guess initial peaks positions for peak refinement by "
        "array shifting")
        self.page3.SetToolTipString("Guess initial peaks positions for peak refinement by "
        "peak-kernel like convolution")

    def toplayout2(self):
        """
        init top notebook tabs for image visualisation and processing
        """
        self.nb0 = wx.Notebook(self.panel, -1, style=0)

        self.viewingLUTpanel = ViewColorPanel(self.nb0)
        self.ImageFilterpanel = FilterBackGroundPanel(self.nb0)
        self.ImagesBrowser = BrowseCropPanel(self.nb0)
        self.Monitor = MosaicAndMonitor(self.nb0)
        if WXPYTHON3:
            self.RoiSelector = ROISelection(self.nb0)

        self.nb0.AddPage(self.viewingLUTpanel, "View && Color")
        self.nb0.AddPage(self.ImageFilterpanel, "ImageFilter")
        self.nb0.AddPage(self.ImagesBrowser, "Browse && Crop")
        self.nb0.AddPage(self.Monitor, "Mosaic && Monitor")
        if WXPYTHON3:
            self.nb0.AddPage(self.RoiSelector, "ROIs Selector")

        self.nb0.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnTabChange_nb0)

        # tooltips
        self.viewingLUTpanel.SetToolTipString("View and Colors parameters")
        self.ImageFilterpanel.SetToolTipString("Digital image processing to remove Background or undesired peaks")
        self.ImagesBrowser.SetToolTipString("Browse a set of images arranged in a array over lines  or columns")
        self.Monitor.SetToolTipString("Monitor and Tracks a selected ROI properties over the data set")
        if WXPYTHON3:
            self.RoiSelector.SetToolTipString("Pixel ROIs Selector from current peaks list of manual selection")

    def line_select_callback(self, eclick, erelease):
        """eclick and erelease are the press and release events"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2)))
        print((" The button you used were: %s %s" % (eclick.button, erelease.button)))

    def OnTabChange_nb0(self, event):
        """
        handling changing tab of top notebook
        """

        #selected_tab = self.nb0.GetSelection()
        # print("selected tab:", selected_tab)
        # print(self.nb0.GetPage(self.nb0.GetSelection()))
        # print(self.nb0.GetPage(self.nb0.GetSelection()).GetName())

        event.Skip()  # patch for windows to update the tab display

    def OnTabChange_PeakSearchMethod(self, event):
        #        print 'tab changed'
        selected_tab = self.nb.GetSelection()
        #         print "selected tab:", selected_tab
        #         print self.nb.GetPage(self.nb.GetSelection())
        #         print self.nb.GetPage(self.nb.GetSelection()).GetName()

        # display raw data or convolved data
        if selected_tab in (0, 1,3):
            pass

        elif selected_tab in (2,):  #convolution method
            #            print "self.page3.TogglebtnState", self.page3.TogglebtnState
            if self.page3.TogglebtnState == 1:  # current plot of convolved data
                if self.ConvolvedData is None:
                    self.getConvolvedData()
                self.dataimage_ROI_display = self.ConvolvedData
                self.Show_ConvolvedImage(1)
            else:  # current plot of raw data
                self.dataimage_ROI_display = self.dataimage_ROI
                self.CropIsOn = False
                self.reinit_aftercrop_draw()
                self.plotPeaks = True
                self.addPeaksMarker()
                self.plotPeaks = False

                self.updatePlotTitle(datatype="Raw Image")
                self.canvas.draw()

        # enable or disable position peak results (centroid or hot pixels)
        if selected_tab in (0, 1, 3):
            self.fitparampanel.keepCentroid.Disable()
            self.fitparampanel.keepHottestPixel.Disable()
            self.fitparampanel.keepHottestPixel.SetValue(True)

        elif selected_tab == 2:
            self.fitparampanel.keepCentroid.Enable()
            self.fitparampanel.keepHottestPixel.Enable()

        elif selected_tab == 4:
            print("selected fitting param panel")

        elif selected_tab == 5:
            if ObjectListView_Present:
                self.pagepeaknav.updateView(event)

        event.Skip()  # patch for windows OS to update the tab display

    def OnAbout(self, _):
        wx.MessageBox(
            'Peak Search GUI from Lauetools Package\n Sept 2024.\n Please contact staff of beamline CRG-IF BM32 at ESRF or micha"_at_"esrf"_dot_"fr', "INFO")

    def OnExit(self, _):
        self.Close()

    def OnQuit(self, _):
        self.Close()

    def askUserForDirname(self):
        """
        provide a dialog to browse the folders and files
        """
        dialog = wx.DirDialog(self, message="Choose folder for results .dat file", defaultPath=self.dirname)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True
            # self.filename = dialog.GetFilename()
            # #self.dirname = dialog.GetDirectory()

            allpath = dialog.GetPath()
            print(allpath)
            self.writefolder = allpath
        else:
            userProvidedFilename = False
        dialog.Destroy()
        return userProvidedFilename

    def OpenImage(self, _):
        # wcd0 = "All files(*)|*|MAR CCD image(*.mccd)|*.mccd|mar tiff(*.tiff)|*.tiff|mar tif(*.tif)|*.tif|Princeton(*.spe)|*.spe|Frelon(*.edf)|*.edf"

        filepath_dlg = wx.FileDialog(self, "Select binary image file",
                                                wildcard=DictLT.getwildcardstring(self.CCDlabel))
        if filepath_dlg.ShowModal() == wx.ID_OK:

            abspath = filepath_dlg.GetPath()

            filename = os.path.split(abspath)[-1]
            dirname = os.path.dirname(abspath)

            self.initialParameter["imagefilename"] = filename
            self.initialParameter["dirname"] = dirname

            self.imagefilename = filename
            self.dirname = dirname

            self.getIndex_fromfilename()
            self.resetfilename_and_plot()

            self.ImageFilterpanel.UseImage.SetValue(False)
            self.ImageFilterpanel.imageBctrl.SetValue("")

    def OnSetFileCCDParam(self, _):
        """Enter manually CCD file params
        Launch Entry dialog
        """
        DPBoard = CCDParamGUI.CCDFileParameters(self, -1, "CCD File Parameters Board", self.CCDlabel)
        DPBoard.ShowModal()
        DPBoard.Destroy()

    def OnFileSeries(self, _):
        wx.MessageBox("not implemented yet. Use better FileSeries/peak_search.py", "INFO")

    def OnFolderPreferences(self, _):
        if self.askUserForDirname():
            print("Peak list .dat file will be written in:")
            print(self.writefolder)

    def OnSetPlotSize(self, _):
        """set marker size
        """
        wx.MessageBox("not implemented yet", "INFO")
        return

    def onClick(self, event):
        """ onclick
        """
        if event.inaxes:

            self.centerx, self.centery = event.xdata, event.ydata

            if not self.CropIsOn:
                print("current clicked positions", self.centerx, self.centery)
            else:
                #                 print "current local clicked positions", self.centerx, self.centery
                #                 print self.imin_crop, self.imax_crop, self.jmin_crop, self.jmax_crop
                self.centerx += self.jmin_crop
                self.centery += self.imin_crop

                print("new clicked positions (crop mode)", self.centerx, self.centery)

            self.viewingLUTpanel.showprofiles(event)
        else:
            pass


    def onPick(self, event):
        """ on pick """
        self.currentROIpatch = None

        if isinstance(event.artist, Rectangle):
            self.currentROIpatch = event.artist

            print(("onPick Rectangle label:", self.currentROIpatch.get_label()))
            print(("onPick Rectangle gid:", self.currentROIpatch.get_gid()))
            print(("onPick Rectangle picker:", self.currentROIpatch.get_picker()))

            self.ROIRectangleselected = True

    def toggle_selector(self, event):
        print(" Key pressed.a or A")
        if event.key in ["A", "a"] and self.RS.active:
            print(" RectangleSelector deactivated.")
            self.RS.set_active(False)
        elif event.key in ["A", "a"] and not self.RS.active:
            print(" RectangleSelector activated.")
            self.RS.set_active(True)

    def onKeyPressed(self, event):
        """Handle key pressed
        """
        key = event.key
        print("key ==> ", key)

        if key == "escape":

            ret = wx.MessageBox("Are you sure to quit?", "Question", wx.YES_NO | wx.NO_DEFAULT, self)

            if ret == wx.YES:
                self.Close()

        elif key == "s":  # 's'
            #            self.timer.Stop()
            print("stop")

        # fit 2D array of pixel intensity centered on click region
        elif key == "f":
            self.onFitOnePeak(1)
        # remove peaks close to the region
        elif key == "r":
            self.onRemovePeaktoPeaklist(1)
        elif key == "c":  # 'c'  continue
            #            self.imageindex = STARTINDEX
            #            self.timer.Start(100)
            self.CropIsOn = not self.CropIsOn
            print("now CropIsOn is ", self.CropIsOn)

            if self.CropIsOn:
                self.activateCrop(event)

            self.readdata_updateplot_aftercrop_uncrop()

        elif key == "z":  # 'c'
            print("zoom")

        elif key == "+":  # 'l'
            print("pressed +")

        elif key == "shift":  # 'l'
            print("shift is pressed")

        # active or not the mode of rectangle selection for pixel ROI
        elif key in ["a", "A"] and WXPYTHON3:
            self.toggle_selector(event)
        # memorize current pixel ROI
        elif key in ["q", "Q"] and WXPYTHON3:
            self.roiindex += 1

            print("self.roiindex", self.roiindex)

            print("pressed on q or Q")
            if self.RS.active:
                print("save and draw selected ROI")
                # print self.RS.extents
                #                 print dir(self.RS)
                #                 print dir(self.RS.artists)

                x, y, width, height = (self.RS.extents[0],
                                        self.RS.extents[3],
                                        self.RS.extents[1] - self.RS.extents[0],
                                        self.RS.extents[3] - self.RS.extents[2])

                rectboxproperties = [x, y, width, height, None, self.roiindex, None]

                self.ROIs[self.roiindex] = rectboxproperties

                print("new ROI", rectboxproperties)
            else:
                return

            # no rectangle drawn, just a clicked pixel. So building a rectangle
            if x == 0.0 or y == 0.0 or width == 0 or height == 0:
                xc, yc = self.centerx, self.centery

                print("single click ROI")

                halfboxx = int(self.RoiSelector.boxxctrl.GetValue())
                halfboxy = int(self.RoiSelector.boxyctrl.GetValue())

                height, width = 2 * halfboxy + 1, 2 * halfboxx + 1

                x, y = xc - halfboxx, yc - halfboxy

                rectboxproperties = [x, y, width, -height, None, self.roiindex, None]

            self.ROIs[self.roiindex] = rectboxproperties

            print("new ROI", rectboxproperties)
            self.addPatchRectangleROI(rectboxproperties)

            print("updated ROIs", self.ROIs)

            self.update_draw(event)

        # delete selected ROI
        elif key in ["d", "D"] and WXPYTHON3:
            if self.ROIRectangleselected:
                print("self.ROIS before", self.ROIs)
                # print dir(self.axes)
                print("I will delete this ROI")
                #                 print dir(self.currentROIpatch)
                #                 print "extents",self.currentROIpatch.get_extents()
                print("x", self.currentROIpatch.get_x())
                print("y", self.currentROIpatch.get_y())
                print("height", self.currentROIpatch.get_height())
                print("width", self.currentROIpatch.get_width())
                print("label", self.currentROIpatch.get_label())

                label_roiindex = int(self.currentROIpatch.get_label())

                if self.ROIs[label_roiindex][6] == "visible":

                    print("Removing ", label_roiindex)

                    self.currentROIpatch.set_visible(False)
                    self.currentROIpatch.set_picker(None)

                    self.ROIs[label_roiindex][6] = "invisible"

                    print("self.ROIS after", self.ROIs)
                else:
                    print("caught invisible rectangle")

            self.update_draw(event)
        #             self.OnReplot(1)

        elif key in ["x", "X"] and WXPYTHON3:
            print("self.ROIs", self.ROIs)
            visibleROIs = []
            for _, val in self.ROIs.items():
                if val[-1] == "visible":
                    print("ROI property")
                    visibleROIs.append(val)

            print("visibleROIs", visibleROIs)

    def gettime(self):
        """set self.currentime to current time
        """
        self.currentime = time.time()

    def getdeltatime(self):
        print("deltatime: %.3f second(s)" % (time.time() - self.currentime))

    def onToggle(self, event):
        """
        handling on auto index button
        """
        self.steppresent = 1500
        self.stepmissing = 1000
        if self.ImagesBrowser.timer.IsRunning():
            self.ImagesBrowser.timer.Stop()
            self.ImagesBrowser.toggleBtn.SetLabel("On Fly")
            print("timer stopped!")
        else:
            print("start to on-fly images viewing mode  ----------------")

            self.ImagesBrowser.toggleBtn.SetLabel("Wait!...")
            self.OnFlyMode = True
            # loop for already present data
            while self.update(event):
                time.sleep(self.steppresent / 1000.0)

            # timer loop for missing data
            print("*******  WAITING DATA   !!!! *********")
            self.ImagesBrowser.timer.Start(self.stepmissing)
            self.ImagesBrowser.toggleBtn.SetLabel("Stop")

    def onToggleCrop(self, event):
        """activate/deactivate crop image mode
        """
        # crop already enabled
        if self.CropIsOn:
            self.ImagesBrowser.toggleBtnCrop.SetLabel("Crop Data")
            self.CropIsOn = False
        else:
            self.ImagesBrowser.toggleBtnCrop.SetLabel("UnCrop Data")
            self.CropIsOn = True
            self.activateCrop(event)

        self.readdata_updateplot_aftercrop_uncrop()

    def update(self, _):
        """
        update at each time step time
        """
        print("\nupdated: ")
        print(time.ctime())

        if self.CurrentFileIsReady():
            self.read_data()
            self.dataimage_ROI_display = self.dataimage_ROI
            self.Show_Image(1)
            self.lastindex=self.imageindex
            self.imageindex += 1
            self.setfilename()
            return True

        else:
            print("waiting for image   :%s" % self.imagefilename)
            # stop the first timer
            return False

    # --- -------  Open File and Navigate on Data Set
    def CurrentFileIsReady(self):
        """
        return True if self.imagefilename is in folder and entire (correct size)
        """
        condition = False

        filename = self.imagefilename.split("/")[-1]

        condexist = filename in os.listdir(self.dirname)

        print("self.currentfilename in CurrentFileIsReady", filename)
        #        print "self.dirname", self.dirname
        if condexist:

            condsize = os.stat(os.path.join(self.dirname, filename))[6] >= 2101248
            if condsize:
                condition = True
                print("file present and correct size!")

        return condition

    def getIndex_fromfilename(self):
        """
        get index of image from the image filename
        """
        self.image_with_index = True
        try:
            self.imageindex = IOimage.getIndex_fromfilename(self.imagefilename,
                                                            CCDLabel=self.CCDlabel,
                                                            stackimageindex=self.stackimageindex,
                                                            nbdigits=self.nbdigits)
            print("************\n\n\nself.imageindex %d \n\n****************" % self.imageindex)
        except (ValueError, TypeError):
            self.imageindex = 0
            self.image_with_index = False

    def setfilename(self):
        """set filename from self.imagefilename, self.imageindex,
                                                    CCDLabel=self.CCDlabel
        """
        print('***** \n\nself.image_with_index',self.image_with_index)
        
        if self.image_with_index and not self.stackedimages:
            self.misstext=''
            self.lastimagefilename = self.imagefilename
            self.imagefilename = IOimage.setfilename(self.imagefilename, self.imageindex,
                                                    CCDLabel=self.CCDlabel, nbdigits=self.nbdigits)
            print('self.imagefilename',self.imagefilename)
            print('self.dirname',self.dirname)
            if not self.imagefilename in os.listdir(self.dirname):
                print("\n\n %s IS MISSING!"%self.imagefilename)
                self.misstext="MISSING FILE with index %d\nfilename: %s"%(self.imageindex,self.imagefilename)

                self.axes.set_title(self.misstext, color='red')

                self.imageindex = self.lastindex
                self.imagefilename = self.lastimagefilename

                self.canvas.draw() 

                return False

            self.axes.set_title('', color='black')
            
            return True

        if self.stackedimages:
            print('setfilename() in PeakSearchGUI()')
            print('self.imagefilename  unchanged ... only stack index',self.imagefilename)
            return True

    def OnStepChange(self, _):
        self.ImagesBrowser.stepindex = int(self.ImagesBrowser.stepctrl.GetValue())
        stepindex = self.ImagesBrowser.stepindex
        self.ImagesBrowser.largeplusbtn.SetLabel("index +%d" % stepindex)
        self.ImagesBrowser.largeminusbtn.SetLabel("index -%d" % stepindex)

        self.ImagesBrowser.slider_image.SetValue(self.imageindex % stepindex)
        self.ImagesBrowser.slider_image.SetMax(stepindex - 1)

        self.ImagesBrowser.slider_imagevert.SetValue(self.imageindex // stepindex)
        self.ImagesBrowser.slider_imagevert.SetMax(self.ImagesBrowser.imageindexmax // stepindex)

    def OnChangeImageMin(self, evt):
        pass

    def OnChangeImageMax(self, _):
        imagemax = int(self.ImagesBrowser.imagemaxtxtctrl.GetValue())

        self.ImagesBrowser.stepindex = int(self.ImagesBrowser.stepctrl.GetValue())
        stepindex = self.ImagesBrowser.stepindex

        self.ImagesBrowser.slider_imagevert.SetMax(imagemax // stepindex)

    def OnLargePlus(self, _):
        """increase self.imageindex by self.stepindex (vertical descending in sample raster scan)
        and read new image and plot
        """
        #        print self.canvas.GetRect()
        #        print self.canvas.GetScreenRect()
        self.stepindex = int(self.ImagesBrowser.stepctrl.GetValue())
        if self.stackedimages:
            #         if self.CCDlabel in ('EIGER_4Mstack',):
            self.stackimageindex += self.stepindex
            self.stackimageindex = max(0,self.stackimageindex)
        else:
            self.lastindex = self.imageindex
            self.imageindex += self.stepindex
        
        self.resetfilename_and_plot()

    def OnLargeMinus(self, _):
        """decrease self.imageindex by self.stepindex (vertical ascendindg in sample raster scan)
        and read new image and plot
        """
        self.stepindex = int(self.ImagesBrowser.stepctrl.GetValue())
        if self.stackedimages:
            #         if self.CCDlabel in ('EIGER_4Mstack',):
            self.stackimageindex -= self.stepindex
            self.stackimageindex = max(0,self.stackimageindex)
        else:
            self.lastindex = self.imageindex
            self.imageindex -= self.stepindex
        self.resetfilename_and_plot()

    def OnPlus(self, _):
        """increase  self.imageindex by 1 (horizontal ascending to the right in sample raster scan)
        and read new image and plot

        Note: if self.stackedimages is True: imageindex is used for stackimageindex
        """
        print(self.canvas.GetRect())
        print(self.canvas.GetScreenRect())
        if self.stackedimages:
            print('\n!!stacked images!!\n')
            print('self.Nbstackedimages', self.Nbstackedimages)
            self.stackimageindex += 1
            self.stackimageindex = max(0,self.stackimageindex)
        else:
            self.lastindex = self.imageindex
            self.imageindex += 1

        self.resetfilename_and_plot()

    def OnMinus(self, _):
        """decrease  self.imageindex by 1 (horizontal descending to the left in sample raster scan)
        and read new image and plot
        """
        print('onMinus')
        if self.stackedimages:
            #         if self.CCDlabel in ('EIGER_4Mstack',):
            self.stackimageindex -= 1
            self.stackimageindex = max(0,self.stackimageindex)
        else:
            self.lastindex = self.imageindex
            self.imageindex -= 1

        self.resetfilename_and_plot()

    def OnGoto(self, _):
        """
        read image with selected self.imageindex and plot
        """
        if self.stackedimages:
            self.stackimageindex = int(self.ImagesBrowser.fileindexctrl.GetValue())
            self.stackimageindex = max(0,self.stackimageindex)
        else:
            self.imageindex = int(self.ImagesBrowser.fileindexctrl.GetValue())

        self.resetfilename_and_plot()

    def onChangeIndex_slider_image(self, _):
        self.stepindex = int(self.ImagesBrowser.stepctrl.GetValue())

        print("self.ImagesBrowser.slider_image.GetValue()",
            self.ImagesBrowser.slider_image.GetValue())

        print("self.imageindex before", self.imageindex)
        if self.stackedimages:
            pass
        #             self.stackimageindex=int(self.ImagesBrowser.fileindexctrl.GetValue())
        #             self.stackimageindex=(self.stackimageindex%self.Nbstackedimages)
        else:
            self.imageindex = int(self.ImagesBrowser.slider_image.GetValue()) + \
                                self.stepindex * int(self.ImagesBrowser.slider_imagevert.GetValue())

        print("self.imageindex after", self.imageindex)

        self.resetfilename_and_plot()

    def onChangeIndex_slider_imagevert(self, _):
        """plot new image obtained by new index changed by vertical (slow axis) slider
        """
        self.stepindex = int(self.ImagesBrowser.stepctrl.GetValue())

        print("self.ImagesBrowser.slider_imagevert.GetValue()",
                                                    self.ImagesBrowser.slider_imagevert.GetValue())
        if self.stackedimages:
            pass
        #             self.stackimageindex=int(self.ImagesBrowser.fileindexctrl.GetValue())
        #             self.stackimageindex=(self.stackimageindex%self.Nbstackedimages)
        else:
            self.imageindex = int(self.ImagesBrowser.slider_image.GetValue()
            ) + self.stepindex * int(self.ImagesBrowser.slider_imagevert.GetValue())

        self.resetfilename_and_plot()

    def resetfilename_and_plot(self):
        print("***   resetfilename_and_plot   *****")

        print("self.CCDLabel", self.CCDlabel)

        nbd = self.ImagesBrowser.nbdigitsctrl.GetValue()
        try:
            self.nbdigits = int(nbd)
        except ValueError:
            self.nbdigits = 0
        
        self.Monitor.nbdigitsctrl.SetValue(str(nbd))

        fileexists = self.setfilename()
        print('fileexists',fileexists)
        if not fileexists:
            return

        self.read_data()

        if (self.ImageFilterpanel.FilterImage and self.ImageFilterpanel.ImageType == "Raw"):
            print("self.ImageFilterpanel.ImageType == 'Raw'")
            self.ImageFilterpanel.blurimage = ImProc.compute_autobackground_image(
                                                            self.dataimage_ROI, boxsizefilter=10)
            self.ImageFilterpanel.Computefilteredimage()
            self.viewingLUTpanel.showImage()
        elif self.ImageFilterpanel.UseImage and self.ImageFilterpanel.ImageType == "Raw":
            self.dataimage_ROI_display = self.OnUseFormula(1)
            self.Show_Image(1, datatype="Raw Image")
        else:

            self.ImageFilterpanel.OnChangeUseFormula(1)

        self.gettime()
        #         self.dataimage_ROI_display = self.dataimage_ROI

        #         self.Show_Image(1)
        print("new image display execution time :")
        self.getdeltatime()
        self.viewingLUTpanel.updateLineXYProfile(1)
        #         self.viewingLUTpanel.OnShowLineXYProfiler(1)
        self.viewingLUTpanel.updateLineProfile()

    def read_data(self, secondaryImage=False, secondaryImagefilename=None):
        """
        read binary image file

        if secondaryImage update
            self.dataimage_ROI_B
        else update
            self.dataimage_ROI

        """
        if not secondaryImage:
            print("Reading image data:")
            print("Directory :", self.dirname)
            print("Filename :", self.imagefilename)

            imagefilename = self.imagefilename
        else:
            print("Reading B image data:")
            print("Directory :", self.dirname)
            print("Filename :", secondaryImagefilename)

            imagefilename = secondaryImagefilename

        _, extension = str(imagefilename).rsplit(".", 1)

        (self.framedim,
            _,
            self.saturationvalue,
            self.fliprot,
            self.offset,
            self.format,
            _,
            self.extension,
        ) = DictLT.dict_CCD[self.initialParameter["CCDLabel"]]

        if extension != self.extension:
            txt = "warning : file extension does not match CCD type set in Set CCD File Parameters ??"
            print(txt)
            wx.MessageBox(f'{txt}', 'Info')

        if self.CCDlabel == "LaueImaging":

            self.paramsHat = (6, 8, 4)

        print("CCD label in PeakSearchGUI: ", self.CCDlabel)
        print("self.stackimageindex", self.stackimageindex)

        dataimage, framedim, _ = IOimage.readCCDimage(imagefilename,
                                                        CCDLabel=self.CCDlabel,
                                                        dirname=self.dirname,
                                                        stackimageindex=self.stackimageindex)

        if secondaryImage:
            self.dataimage_ROI_B = dataimage
        else:  # TODO better use self.format ??
            # type np.int to test with cython module arr.pyx
            #             self.dataimage_ROI = dataimage.astype(np.int16)
            if self.CCDlabel in ("EIGER_4M","EIGER_4MCdTe", "EIGER_4MCdTestack", 'EIGER_1M'):
                img_dataformat = np.uint32
            elif self.CCDlabel in ("MaxiPIXCdTe",):
                img_dataformat = np.int32
                if self.stackedimages:
                    self.Nbstackedimages = framedim[0]
            else:
                img_dataformat = np.uint16 
            self.dataimage_ROI = dataimage.astype(img_dataformat)

        if self.CropIsOn:
            if self.stackedimages != 1:
                self.Nbstackedimages = framedim[0]

            xpic, ypic = np.round(self.centerx), np.round(self.centery)

            self.dataimage_ROI = IOimage.readrectangle_in_image(imagefilename,
                                                                xpic,
                                                                ypic,
                                                                int(self.boxx),
                                                                int(self.boxy),
                                                                dirname=self.dirname,
                                                                CCDLabel=self.CCDlabel)

        if self.CCDlabel in ("sCMOS", "sCMOS_fliplr"):
            self.vminmin = 0
            self.vmiddle = 1010
            self.vmaxmax = 10000
            self.vmin = 1000
            self.vmax = 2000
        else:
            self.vminmin = -100
            self.vmiddle = 100
            self.vmaxmax = 10000
            self.vmin = 1
            self.vmax = 2000

    # ---   --- DISPLAY IMAGE
    def OnCheckPlotValues(self, _):
        """enable or disable drawing of numerical pixel intensity value on plot
        """
        if self.numvalues_chck.GetValue():
            self.justcheckedShowValues = True
        else:
            self.justcheckedShowValues = False

        print("self.justcheckedShowValues in OnCheckPlotValues ", self.justcheckedShowValues)

        if not self.numvalues_chck.GetValue():
            if len(self.axes.texts) > 0:
                for txt in self.axes.texts:
                    txt.set_visible(False)
            self.axes.texts = []
            self.canvas.draw()
        else:
            self.PlotValues()

    def PlotValues(self):
        """Draw numerical pixel intensity value on plot
        if self.numvalues_chck is True
        """
        if not self.numvalues_chck.GetValue():
            return

        if len(self.axes.texts) > 0:
            for txt in self.axes.texts:
                txt.set_visible(False)
            self.axes.texts = []

        xmin, xmax, ymin, ymax = self.getDisplayedImageSize()

        if np.abs(xmax - xmin) > 25 or np.abs(ymax - ymin) > 25:
            #             wx.MessageBox('Field of view of pixel intensities is too large! Please zoom in!','info')
            print("Field of view of pixel intensities is too large!")
            return

        # Add new drawn values on plot
        for i in np.arange(int(ymax) + 1, int(ymin) + 2, 1):
            for j in np.arange(int(xmin), int(xmax) + 2, 1):
                label = self.dataimage_ROI_display[i, j]
                #                 print "label",label
                self.axes.text(j, i, label, color="black", ha="center", va="center", size=7)

        #         print "fig.texts",self.fig.texts # is a list of Text objects
        #         print "len(axes.texts)",len(self.axes.texts) # is a list of Text objects
        self.justcheckedShowValues = False

        self.canvas.draw()

    def init_figure_draw(self):
        """ init the figure
        """
        # clear the axes and redraw the plot anew
        self.axes.clear()
        #        self.axes.set_autoscale_on(False) # Otherwise, infinite loop
        self.axes.set_autoscale_on(True)

        self.IminDisplayed = 1
        # highest pixel intensity
        #        self.ImaxDisplayed = DictLT.dict_CCD[self.CCDlabel][2]
        # value defined
        self.ImaxDisplayed = self.viewingLUTpanel.slider_vmax.GetValue()
        self.IminDisplayed = self.viewingLUTpanel.slider_vmin.GetValue()

        self.myplot = self.axes.imshow(self.dataimage_ROI_display,  # aspect = 'equal',
                                    interpolation="nearest",
                                    norm=LogNorm(vmin=self.IminDisplayed, vmax=self.ImaxDisplayed))

        title = self.imagefilename
        suptitle = self.dirname
        if len(suptitle)>30:
            splitwords =['RAW_DATA','inhouse']
            for sw in splitwords:
                if sw in suptitle:
                    s1,s2 = suptitle.split(sw)
                    s1sw = os.path.join(s1,sw)
                    suptitle= '%s\n%s'%(s1sw,s2)
        if self.stackedimages:
            title += "\nsstack index %d" % self.stackimageindex
        self.axes.set_title(title)
        self.fig.suptitle(suptitle)
        # self.myplot.set_clim=(1,200)  # work?
        self.myplot.set_cmap(self.viewingLUTpanel.comboLUT.GetValue())

        self.fig.colorbar(self.myplot)
        self.normalizeplot()

        self.canvas.draw()

    def Show_Image(self, event, datatype="Raw Image"):
        """
        show self.dataimage_ROI_display
        """
        self.current_data_display = "Raw Image"

        self.updatePlotTitle(datatype=datatype)

        self.myplot.set_data(self.dataimage_ROI_display)

        self.OnSpinCtrl_IminDisplayed(event)

        # update line profiler
        self.viewingLUTpanel.showprofiles(event)

        self.PlotValues()

    def Show_ConvolvedImage(self, event, datatype="Convolved Image"):
        """set displayed data to be convolved data
        """

        if self.ConvolvedData is None:
            print("Calculate Convolved Data")
            self.getConvolvedData()
        else:
            print("Use already computed Convolved data")

        if self.page3.Applythreshold.GetValue():
            self.ConvolvedData = np.clip(self.ConvolvedData,
                                            float(self.page3.ThresholdConvolveCtrl.GetValue()),
                                            np.amax(self.ConvolvedData))
        else:
            self.getConvolvedData()

        self.dataimage_ROI_display = self.ConvolvedData

        self.current_data_display = "Convolved Image"

        self.updatePlotTitle(datatype=datatype)
        self.myplot.set_data(self.dataimage_ROI_display)

        self.OnSpinCtrl_IminDisplayed(event)

    def updatePlotTitle(self, datatype=None):
        """update plot title
        """
        if datatype == None:
            datatype = ""
        titlestring = "%s\n%s" % (self.imagefilename, datatype)

        if self.stackedimages:
            titlestring += "\nsstack index %d" % self.stackimageindex
        if 1:  # not self.OnFlyMode:
            if self.peaklistPixels is not None:
                nbpeaks = len(self.peaklistPixels)
                if nbpeaks > 0:
                    titlestring += "\n%d peaks" % nbpeaks

            if self.current_data_display == "Raw Image":

                MaxI = np.amax(self.dataimage_ROI_display)
                titlestring += "  max I= %.1f" % MaxI

        self.axes.set_title(titlestring)

    def normalizeplot(self):
        """normalize current displayed array according to vmin vmax sliders
        """

        norm = mpl.colors.Normalize(vmin=self.IminDisplayed, vmax=self.ImaxDisplayed)
        self.myplot.set_norm(norm)

    #        self.myplot.set_clim(self.IminDisplayed, self.ImaxDisplayed)

    def update_draw(self, _):
        """update 2D plot taken into account change of LUT table and vmin vamax values
        """
        #        if self.data_2D == None:
        #            return
        #
        #        self.ReadData()
        #        self.myplot.set_data(self.data_2D)
        if not self.OnFlyMode:

            self.normalizeplot()
        #            print 'normalized data for plot ---',
        #            self.getdeltatime()

        self.PlotValues()
        self.canvas.draw()

    def getDisplayedImageSize(self):
        """get xmin, xmax, ymin, ymax from current displayed image
        """
        # bbox = self.axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        ymin, ymax = self.axes.get_ylim()
        xmin, xmax = self.axes.get_xlim()
        return xmin, xmax, ymin, ymax

    def set_circleradius(self, artists):
        """
        self.set_circleradius(self.viewingLUTpanel.drg.artists)
        """
        xmin, xmax, ymin, ymax = self.getDisplayedImageSize()

        print("xmin, xmax, ymin, ymax", xmin, xmax, ymin, ymax)

        r = int(min(5, max(abs(xmin - xmax), abs(ymin - ymax)) / 2048.0 * 50))

        for artist in artists:
            artist.radius = r

    # ---  Crop Data
    def activateCrop(self, _):
        """ set boxx and boxy from ctrls
        """
        self.boxx = int(self.ImagesBrowser.boxxctrl.GetValue())
        self.boxy = int(self.ImagesBrowser.boxyctrl.GetValue())

        print("self.boxx,self.boxy", self.boxx, self.boxy)

    def readdata_updateplot_aftercrop_uncrop(self):
        """read data and update data to be displayed and redraw
        """
        self.read_data()
        self.dataimage_ROI_display = self.dataimage_ROI
        self.reinit_aftercrop_draw()

        self.updatePlotTitle()
        self.canvas.draw()

    def reinit_aftercrop_draw(self):
        """ reinit the figure
        """
        # clear the axes and redraw the plot anew
        #
        self.axes.clear()

        self.myplot = self.axes.imshow(self.dataimage_ROI_display,  # aspect = 'equal',
                                    interpolation="nearest",
                                    norm=LogNorm(vmin=self.IminDisplayed, vmax=self.ImaxDisplayed))

        self.normalizeplot()

        if self.CropIsOn:

            offset_x = self.jmin_crop
            offset_y = self.imin_crop

            def tick_indexx(indexx, __dict__):
                """return integer from index
                """
                return int(indexx + offset_x)

            def tick_indexy(indexy, _):
                """return integer from index
                """
                return int(indexy + offset_y)

            self.axes.xaxis.set_major_formatter(FuncFormatter(tick_indexx))
            self.axes.yaxis.set_major_formatter(FuncFormatter(tick_indexy))

        self.canvas.draw()

    def cropdata_array_center(self, centeri, centerj, boxi, boxj):

        print("centeri, centerj", centeri, centerj)

        (imin, imax, jmin, jmax) = (centeri - boxi, centeri + boxi + 1,
                                    centerj - boxj, centerj + boxj + 1)

        # avoid to wrong indices when slicing the data
        imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax, jmin, jmax,
                                                                            framedim=self.framedim)

        #         print "imin, imax, jmin, jmax", imin, imax, jmin, jmax

        self.imin_crop, self.imax_crop, self.jmin_crop, self.jmax_crop = (imin, imax, jmin, jmax)

        self.dataimage_ROI = self.dataimage_ROI[imin:imax, jmin:jmax]

    def cropdata_center(self, centerx, centery, boxx, boxy):
        self.cropdata_array_center(centery, centerx, boxy, boxx)

    def OnReplot(self, event):  # due to a background correction
        # Create a gaussian bkg and substract the data according to it -------------------
        #        if self.viewingLUTpanel.SubBKG.GetValue():
        #            # update self.dataimage_ROI_display
        #            self.remove_bkg_on_datatodisplay()
        # ---------------------------------------------------------------------------------

        self.updatePlotTitle()

        self.addPeaksMarker()

        #        self._replot()
        self.update_draw(event)

    def buildMosaic(self, parent=None):
        """ launch MOS.buildMosaic3() with GUI inputs as arguments
        """
        # self.Monitor

        dirname = self.initialParameter["dirname"]
        # filename = self.initialParameter["imagefilename"]

        # filepathname = os.path.join(dirname, filename)

        # use images indices from start final and step fields
        if self.Monitor.generalindexradiobtn.GetValue():

            startind = int(self.Monitor.startindexctrl.GetValue())
            endind = int(self.Monitor.lastindexctrl.GetValue())
            stepind = int(self.Monitor.stepimageindexctrl.GetValue())
            #        endind = 9

            nbimages_per_line = int(self.Monitor.stepctrl.GetValue())

            nbimages_asked = int(len(np.arange(startind, endind + 1, stepind)))

            print("in buildMosaic() of PEAKSEARCHGUI.py------------------------------------\n")
            print("nbimages_asked", nbimages_asked)
            print("nbimages_per_line", nbimages_per_line)
            # print("nbimages_asked", type(nbimages_asked))
            # print("nbimages_per_line", type(nbimages_per_line))

            # reminder of integer division of nbimages_asked by nbimages_per_line
            rr = nbimages_asked % nbimages_per_line
            if rr != 0:
                print("reminder != 0")
                nb_lines = int(nbimages_asked // nbimages_per_line + 1)
            else:
                print("reminder == 0")
                nb_lines = int(nbimages_asked / nbimages_per_line)

            print("nb_lines", nb_lines)

            selected2Darray_imageindex = np.arange(startind, startind + nb_lines * nbimages_per_line, 1)
            print("len(selected2Darray_imageindex)", len(selected2Darray_imageindex))

            selected2Darray_imageindex.shape = (nb_lines, nbimages_per_line)
            print("selected2Darray_imageindex", selected2Darray_imageindex)

        else:
            # rectangular 2D slice image index selection
            if self.Monitor.rectangleindexradiobtn.GetValue():

                centerimageindex = int(self.Monitor.centerindexctrl.GetValue())
                imageindexboxX = int(self.Monitor.txtimagefastindexboxctrl.GetValue())
                imageindexboxY = int(self.Monitor.txtimageslowindexboxctrl.GetValue())

            elif self.Monitor.predefinedROIradiobtn.GetValue():
                print("using predefined ROI")
                key_ROI = str(self.Monitor.comboROI.GetValue())
                ROI_extent = self.Monitor.dict_ROI[key_ROI]

                print("ROI_extent", ROI_extent)

                centerimageindex = int(ROI_extent[0])
                imageindexboxX = int(ROI_extent[1] // 2 + 0.75)
                imageindexboxY = int(ROI_extent[2] // 2 + 0.75)

            # in all images map data set (from mesh scan in SPEC for instance nb of points + 1 with fast motor)
            nbimages_per_line = int(self.Monitor.stepctrl.GetValue())

            mapstarting_index = int(self.Monitor.mapstartingimageindexctrl.GetValue())

            print("in PEAKSEARCHGUI.py------------------------------------\n")
            print("centerimageindex", centerimageindex)
            print("imageindexboxX", imageindexboxX)
            print("imageindexboxY", imageindexboxY)
            print("mapstarting_index", mapstarting_index)

            nb_lines_max = (int((centerimageindex - mapstarting_index) // nbimages_per_line)
                + imageindexboxY + 1)
            nbmaximages_asked = nb_lines_max * nbimages_per_line
            print("nbmaximages_asked", nbmaximages_asked)
            print("nb_lines_max", nb_lines_max)

            absoluteimageindexarray2D = (mapstarting_index + np.arange(nbmaximages_asked)).reshape((nb_lines_max, nbimages_per_line))

            print("absoluteimageindexarray2D", absoluteimageindexarray2D)

            #             jcenter = (centerimageindex - mapstarting_index) % nbimages_per_line
            #             icenter = int((centerimageindex - mapstarting_index) / nbimages_per_line)
            #
            #             print 'icenter,jcenter', icenter, jcenter
            #             selected2Darray_imageindex = GT.extract_array((icenter, jcenter),
            #                                                         (imageindexboxY, imageindexboxX),
            #                                                         absoluteimageindexarray2D)

            selected2Darray_imageindex = GT.extract2Dslice(centerimageindex,
                                                        (imageindexboxY, imageindexboxX),
                                                        absoluteimageindexarray2D)

            nb_lines, nbimages_per_line = selected2Darray_imageindex.shape
            print("selected2Darray_imageindex", selected2Darray_imageindex)
            print("nb_lines", nb_lines)

        try:  # check if self.centerx exists
            if self.centerx is None:
                print('self.centerx',self.centerx)
                return
        except AttributeError:
            wx.MessageBox("Click before on a point in image to select the center of the ROI", "INFO")
            return

        xpic, ypic = np.round(self.centerx), np.round(self.centery)
        boxsize_col = int(self.Monitor.boxxctrl.GetValue())
        boxsize_line = int(self.Monitor.boxyctrl.GetValue())

        print("selected pixel position: xpic, ypic", xpic, ypic)

        selectedcounters = self.Monitor.cselected

        #         param = (self.dirname, self.imagefilename, startind, endind, stepind,
        #                  nb_lines, nbimages_per_line,
        #                 xpic, ypic,
        #                 boxsize_col, boxsize_line,
        #                 selectedcounters)
        #         MOS.buildMosaic2(param, dirname,
        #                          ccdlabel=self.CCDlabel,
        #                          parent=parent)

        # continuous indices extract

        print("len(selected2Darray_imageindex)", len(selected2Darray_imageindex))
        print("(nb_lines, nbimages_per_line)", (nb_lines, nbimages_per_line))
        # 2D slice (rectangular)indices extract
        #         selected2Darray_imageindex = None

        nbdigits = int(self.Monitor.nbdigitsctrl.GetValue())
        self.ImagesBrowser.nbdigitsctrl.SetValue(str(nbdigits))

        dict_param = {}
        dirname = dict_param["imagesfolder"] = self.dirname
        dict_param["filename_representative"] = self.imagefilename
        dict_param["CCDLabel"] = self.CCDlabel
        dict_param["nbdigits"] = nbdigits

        dict_param["selected2Darray_imageindex"] = selected2Darray_imageindex
        dict_param["pixelX_center"], dict_param["pixelY_center"] = xpic, ypic
        dict_param["pixelboxsize_X"], dict_param["pixelboxsize_Y"] = (boxsize_col, boxsize_line)

        dict_param["selectedcounters"] = selectedcounters

        # normalization of data according to monitor value read in image header
        dict_param["NormalizeWithMonitor"] = False
        if self.Monitor.normalizechck.GetValue():
            dict_param["NormalizeWithMonitor"] = True
        dict_param["monitoroffset"] = float(self.Monitor.monitoroffsetctrl.GetValue())

        # for fitting peak over several images
        guessed_peaksize = float(self.fitparampanel.peaksizectrl.GetValue())
        dictfittingparameters = {'peaksizeStart':guessed_peaksize}

        outputfolder = dirname

        MOS.buildMosaic3(dict_param, outputfolder, parent=parent,
                         ccdlabel=self.CCDlabel, dictfittingparameters=dictfittingparameters)

    def onOpenBImage(self, _):
        self.FileDialog = wx.FileDialog(self, "Choose an image", style=wx.OPEN,
                                                                        defaultDir=self.dirname)
        dlg = self.FileDialog
        dlg.SetMessage("Choose an image as 'B' array")
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()

            self.Bdirname = dlg.GetDirectory()
            self.BImageFilename = str(filename)

        else:
            pass

    def onOpenBlackListFile(self, _):
        myFileDialog = wx.FileDialog(self, "Choose a List of peaks not to be considered",
                                    style=wx.OPEN,
                                    defaultDir=self.dirname)
        dlg = myFileDialog
        dlg.SetMessage("Choose a List of peaks not to be considered")
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()

            #             self.dirnameBlackList = dlg.GetDirectory()
            self.BlackListFilename = str(filename)

        else:
            pass

    def ReadDetFile(self, _):
        myFileDialog = wx.FileDialog(self,
                                    "Choose a detector calibration File .det",
                                    style=wx.OPEN,
                                    defaultDir=self.dirname,
                                    wildcard="det. calib. files(*.det)|*.det|All files(*)|*")
        dlg = myFileDialog
        dlg.SetMessage("Choose a detector calibration File .det")
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()

            #             self.dirnameBlackList = dlg.GetDirectory()
            self.DetFilename = str(filename)
            self.CCDcalib = IOLT.readCalib_det_file(filename)

        else:
            pass

    # ---   Background correction
    def OnUseFormula(self, _):

        print("read B image")
        self.read_data(secondaryImage=True, secondaryImagefilename=self.BImageFilename)
        # this where the formula is used

        formulaexpression = str(self.ImageFilterpanel.formulatxtctrl.GetValue())

        print("use formula to calculate new image")
        print(formulaexpression)

        SaturationLevel = DictLT.dict_CCD[self.CCDlabel][2]

        newarray = ImProc.applyformula_on_images(self.dataimage_ROI,
                                                self.dataimage_ROI_B,
                                                formulaexpression=formulaexpression,
                                                SaturationLevel=SaturationLevel,
                                                clipintensities=True)

        print("OnUseFormula resulting histogram")
        print(np.histogram(newarray))

        return newarray

    def remove_bkg_on_datatodisplay(self):
        # TODO: extend for other image file(only for mccd now) !!
        # for VHR
        yc, xc = str(self.ImageFilterpanel.bkgcenter.GetValue())[1:-1].split(",")
        cst = float(self.ImageFilterpanel.bkgconstant.GetValue())
        amp_gauss = int(self.ImageFilterpanel.bkgamplitude.GetValue())

        dim1, dim2 = self.framedim

        print("self.framedim", self.framedim)

        Xin, Yin = np.mgrid[0:dim1, 0:dim2]
        cond_circle = (Xin - int(dim1 / 2.0)) ** 2 + (Yin - int(dim2 / 2.0)) ** 2 <= 1023 ** 2

        if amp_gauss != 0:

            data_gauss = fit2d.gaussian(amp_gauss, int(xc), int(yc), 500, 500)(Xin, Yin) + cst
            self.dataimage_ROI_display = np.where(cond_circle, self.dataimage_ROI - data_gauss,
                                                                                self.dataimage_ROI)

            print("data modified ...")

        else:  # only constant substraction

            self.dataimage_ROI_display = np.where(cond_circle, self.dataimage_ROI - cst,
                                                                                self.dataimage_ROI)

    def onMotion_ToolTip(self, event):
        """tool tip to show data when mouse hovers on plot
        Some pixels at the image border could not be detected
        """

        if self.dataimage_ROI is None:
            return

        collisionFound = False

        dims, dimf = self.dataimage_ROI_display.shape[:2]
        #        print "self.data_2D.shape onmotion", self.data_2D.shape
        # radius = 0.5
        if event.xdata != None and event.ydata != None:  # mouse is inside the axes
            #            for i in xrange(len(self.dataX)):
            #                radius = 1
            #                if abs(event.xdata - self.dataX[i]) < radius and abs(event.ydata - self.dataY[i]) < radius:
            #                    top = tip = 'x=%f\ny=%f' % (event.xdata, event.ydata)
            #            for i in xrange(dims * dimf):
            #                X, Y = self.Xin[0, i % dimf], self.Yin[i % dimf, 0]
            evx = event.xdata
            evy = event.ydata
            rx = int(np.round(evx))
            ry = int(np.round(evy))

            if (abs(rx - (dimf - 1) // 2) <= (dimf - 1) // 2
                and abs(ry - (dims - 1) // 2) <= (dims - 1) // 2):
                #                print X, Y
                #                print event.xdata, event.ydata

                zvalue = self.dataimage_ROI_display[ry, rx]

                #                tip = 'x=%f\ny=%f\nI=%.5f' % (event.xdata, event.ydata, zvalue)
                if self.CropIsOn:
                    #                    tip = 'x=%d\ny=%d\nI=%.5f' % (rx + self.centerx - self.boxx,
                    #                                                  ry + self.centery - self.boxy,
                    #                                                  zvalue)

                    tip = "x=%d\ny=%d\nI=%.5f" % (rx + self.jmin_crop, ry + self.imin_crop, zvalue)
                    xabs, yabs = rx + self.jmin_crop, ry + self.imin_crop
                else:
                    #tip = "x=%d\ny=%d\nI=%.5f" % (rx, ry, zvalue)
                    if isinstance(zvalue, (float, np.float64)):
                        tip = "x=%.2f\ny=%.2f\nI=%.5f" % (evx, evy, zvalue)
                    else:
                        tip = "x=%.2f\ny=%.2f\nI=%s" % (evx, evy, str(zvalue))

                    xabs, yabs = rx, ry

                if self.viewingLUTpanel.show2thetachi.GetValue():
                    if self.CCDcalib is not None:

                        #                         print "self.CCDcalib['CCDCalibParameters']", self.CCDcalib['CCDCalibParameters']

                        tth, chi = F2TC.calc_uflab([xabs, xabs],
                                                    [yabs, yabs],
                                                    self.CCDcalib["CCDCalibParameters"],
                                                    returnAngles=1,
                                                    pixelsize=165.0 / 2048,
                                                    kf_direction="Z>0")
                        tip += "\n(2theta, chi)= %.2f,%.2f" % (tth[0], chi[0])

                self.tooltip.SetTip(tip)
                self.tooltip.Enable(True)
                collisionFound = True
                #            break
                return
        if not collisionFound:
            pass
            # if false, it will block others tooltip from buttons, statictext etc...

    #             self.tooltip.Enable(False)

    # ---  ----Image display WIDGETS
    def OnChangeLUT(self, event):
        #         print "OnChangeLUT"
        self.myplot.set_cmap(self.viewingLUTpanel.comboLUT.GetValue())

        self.update_draw(event)

    def displayIMinMax(self):
        self.viewingLUTpanel.Iminvaltxt.SetLabel(str(self.IminDisplayed))
        self.viewingLUTpanel.Imaxvaltxt.SetLabel(str(self.ImaxDisplayed))

    def on_slider_IminDisplayed(self, event):
        self.IminDisplayed = self.viewingLUTpanel.slider_vmin.GetValue()

        #         self.viewingLUTpanel.vminctrl.SetValue(int(self.IminDisplayed))

        if self.ImaxDisplayed <= self.IminDisplayed:
            self.IminDisplayed = self.ImaxDisplayed - 1
            self.viewingLUTpanel.slider_vmin.SetValue(self.IminDisplayed)

        self.displayIMinMax()
        self.update_draw(event)

    def OnSpinCtrl_IminDisplayed(self, event):
        vminmin = self.viewingLUTpanel.vminminctrl.GetValue()
        vmaxmax = self.viewingLUTpanel.vmaxmaxctrl.GetValue()
        vmiddle = self.viewingLUTpanel.vmiddlectrl.GetValue()

        self.viewingLUTpanel.slider_vmin.SetMin(int(vminmin))
        self.viewingLUTpanel.slider_vmin.SetMax(int(vmiddle))
        self.viewingLUTpanel.slider_vmax.SetMin(int(vmiddle))
        self.viewingLUTpanel.slider_vmax.SetMax(int(vmaxmax))
        self.update_draw(event)

    def on_slider_ImaxDisplayed(self, event):
        self.ImaxDisplayed = self.viewingLUTpanel.slider_vmax.GetValue()
        #         self.viewingLUTpanel.vmaxctrl.SetValue(int(self.ImaxDisplayed))

        #        print "self.ImaxDisplayed", self.ImaxDisplayed
        #         self.viewingLUTpanel.slider_vmax.SetValue(self.ImaxDisplayed)

        if self.ImaxDisplayed <= self.IminDisplayed:
            self.ImaxDisplayed = self.IminDisplayed + 1
            self.viewingLUTpanel.slider_vmax.SetValue(self.ImaxDisplayed)

        self.displayIMinMax()
        self.update_draw(event)

    # def OnSpinCtrl_ImaxDisplayed(self, event):
    #     """on change Imax by spin control
    #     """
    #     #        print "OnSpinCtrl_ImaxDisplayed !!!"

    #     if self.current_data_display == "Raw Image" and not self.OnFlyMode:
    #         IminDisplayed = self.viewingLUTpanel.slider_vmin.GetValue()
    #         ImaxDisplayed = self.viewingLUTpanel.slider_vmax.GetValue()

    #         if IminDisplayed >= ImaxDisplayed:
    #             ImaxDisplayed = IminDisplayed + 1

    #         self.viewingLUTpanel.slider_vmax.SetMax(int(ImaxDisplayed))
    #     #             self.viewingLUTpanel.slider_vmin.SetMax(int(ImaxDisplayed))

    #     #            print "raw image ImaxDisplayed"
    #     elif self.current_data_display == "Convolved Image":

    #         self.ImaxDisplayed = self.page3.vmaxctrl.GetValue()
    #     #            print "Convolved image ImaxDisplayed"

    #     #        print "self.ImaxDisplayed in OnSpinCtrl_ImaxDisplayed", self.ImaxDisplayed

    #     self.update_draw(event)

    def Get_XYI_from_fit2dpeaksfile(self, filename):
        """
        useless ?
        """
        return F2TC.Compute_data2thetachi(filename, sorting_intensity="yes",
                                            param=self.parent.defaultParam,
                                            pixelsize=self.parent.pixelsize,
                                            dim=self.parent.dim,  # only for peaks coming from fit2d doing an y direction inversion
                                        )

    def addPeaksMarker(self):
        #print('in addPeaksMarker')
        if self.plotPeaks is False or self.peaklistPixels is None:
            # delete previous patches:
            if self.largehollowcircles != []:
                for ar in self.axes.patches:
                    Artist.remove(ar)
            #self.axes.patches = []
            return

        # delete previous patches:
        if self.largehollowcircles != []:
            for ar in self.axes.patches:
                Artist.remove(ar)
            #self.axes.patches = []

        # rebuild circular markers
        self.largehollowcircles = []
        self.smallredcircles = []
        # correction only to fit peak position to the display
        if self.position_definition == 1:
            offset_convention = np.array([1, 1])
            if len(self.peaklistPixels.shape) == 1:
                XYlist = (self.peaklistPixels[:2] - offset_convention,)
            else:
                XYlist = self.peaklistPixels[:, :2] - offset_convention

            for po in XYlist:

                large_circle = Circle(po, 7, fill=False, color="b")
                center_circle = Circle(po, 0.5, fill=True, color="r")
                self.axes.add_patch(large_circle)
                self.axes.add_patch(center_circle)

                self.largehollowcircles.append(large_circle)
                self.smallredcircles.append(center_circle)

        if self.position_definition == 2:

            PointToPlot = np.zeros(self.peaklistPixels.shape)
            PointToPlot = self.peaklistPixels - np.array([1.5, 0.5])

            for po in PointToPlot:

                large_circle = Circle(po, 7, fill=False, color="b")
                center_circle = Circle(po, 0.5, fill=True, color="r")
                self.axes.add_patch(large_circle)
                self.axes.add_patch(center_circle)

    def addPatchRectangleROI(self, rectboxproperties):

        print("rectboxproperties", rectboxproperties)

        labelroiindex = rectboxproperties[5]

        rect = PatchRectangle(rectboxproperties[0:2],
                                rectboxproperties[2],
                                -rectboxproperties[3],
                                facecolor="none",
                                edgecolor="k",
                                picker=20,
                                alpha=0.5,
                                label=labelroiindex)
        print("adding labelroiindex", labelroiindex)

        self.axes.add_patch(rect)

        self.ROIs[labelroiindex][4] = rect
        self.ROIs[labelroiindex][6] = "visible"

    def RemoveLastRectangle(self):

        if type(self.axes.patches[-1]) == type(Rectangle((1, 1), 1, 1)):
            del self.axes.patches[-1]

    def addPatchRectangle(self, X, Y, size=50):
        hsize = size / 2.0
        self.axes.add_patch(Rectangle((X - hsize, Y - hsize), size, size, fill=False))

    def onOpenPeakListBoard(self, _):

        PListsBoard = PeaksListBoard.PeaksListBoard(self, -1)

        PListsBoard.Show(True)

    def gethisto(self, nbhotpixels=1000):
        """compute intensity histogram and minimum intensity of the 'nbhotpixels' most intense pixels """
        mini = np.amin(self.dataimage_ROI)
        maxi = np.amax(self.dataimage_ROI)
        ravI = np.ravel(self.dataimage_ROI)
        histo = np.histogram(ravI, 100, range=(mini, maxi))  # N,bins

        csum = np.cumsum(histo[0])
        nbtot = np.size(ravI)
        try:  # for EIGER int32 case...
            bb = np.where(csum > nbtot - nbhotpixels)[0][0]
            th = histo[1][bb]
        except IndexError:
            th=2000

        return histo, int(th)

    def ShowHisto(self, _):
        histo, _ = self.gethisto()

        plothisto = HISTOPLOT.HistogramPlot(self, -1, self.imagefilename, "Intensity: ", histo, logscale=1)

        plothisto.Show(True)

    # --- cursor
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

        if 0:  # refresh display cursor  issue when using tunnel ssh 
            if sys.platform not in ("darwin",):
                if not WXPYTHON4:
                    dc.BeginDrawing()

                x, y, left, right, bottom, top = [int(val) for val in (x, y, left, right, bottom, top)]

                self.erase_cursor()
                line1 = (x, bottom, x, top)
                line2 = (left, y, right, y)
                # warning there is a test if lastInfo is a self attribute.
                # So don't set self.lastInfo in __init__()
                self.lastInfo = line1, line2, ax, dc
                dc.DrawLine(*line1)  # draw new
                dc.DrawLine(*line2)  # draw new
                if not WXPYTHON4:
                    dc.EndDrawing()

        xabs = int(np.round(event.xdata))
        yabs = int(np.round(event.ydata))

        textsb = "pixX=%d y=%d" % (xabs, yabs)

        if self.viewingLUTpanel.show2thetachi.GetValue():
            if self.CCDcalib is not None:
                tth, chi = F2TC.calc_uflab([xabs, xabs],
                                            [yabs, yabs],
                                            self.CCDcalib["CCDCalibParameters"],
                                            returnAngles=1,
                                            pixelsize=165.0 / 2048,
                                            kf_direction="Z>0")

                textsb += "   (2theta, chi)= %.2f, %.2f" % (tth[0], chi[0])

        self.sb.SetStatusText(textsb, 0)

    def erase_cursor(self):
        try:
            lastline1, lastline2, _, lastdc = self.lastInfo
        except AttributeError:
            pass
        else:
            lastdc.DrawLine(*lastline1)  # erase old
            lastdc.DrawLine(*lastline2)  # erase old

    # --- ---   Convolved Data Functions
    def ComputeConvolvedData(self, _):
        self.getConvolvedData()

        self.page3.showconvolvedImage_btn.SetLabel("Show Image")

        self.page3.TogglebtnState = 1

        self.Show_ConvolvedImage(1)

    def getConvolvedData(self):
        """ convolve data according to check value of
        """
        # TODO: add convolution parameters
        #        if self.viewingLUTpanel.UseImage.GetValue():
        if self.ImageFilterpanel.FilterImage.GetValue():
            toconvolve = self.dataimage_ROI_display
        else:
            toconvolve = self.dataimage_ROI

        self.ConvolvedData = ImProc.ConvolvebyKernel(toconvolve, 4, 5, 2)

    def ShowHisto_ConvolvedData(self, _):
        if self.ConvolvedData is None:
            print("Calculate Convolved Data")
            self.getConvolvedData()
        else:
            print("Use already computed Convolved data")

        mini = np.amin(self.ConvolvedData)
        maxi = np.amax(self.ConvolvedData)
        histo = np.histogram(np.ravel(self.ConvolvedData), 100, range=(mini, maxi))  # N,bins
        plothisto = HISTOPLOT.HistogramPlot(self, -1, self.imagefilename, "Convolved Intensity", histo, logscale=1)

        plothisto.Show(True)

        print("histo")
        print(len(histo[0]))
        print(len(histo[1]))
        accum = np.cumsum(histo[0][::-1])[::-1]
        plotaccum_hotpixelfrequencies = PLOT1D.Plot1DFrame(self,
                                                    -1,
                                                    self.imagefilename,
                                                    "Accumulated frequency from hottestIntensity",
                                                    np.array([histo[1][1:], accum]),
                                                    logscale=1)

        plotaccum_hotpixelfrequencies.Show(True)

    def OnSaveFigure(self, _):

        dlg = wx.FileDialog(self, "Saving in png format. Choose a file", self.dirname,
                                                    "", "*.*", wx.SAVE | wx.OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            # Open the file for write, write, close
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()

            if len(str(filename).split(".")) > 1:  # filename has a extension
                Pre, Ext = str(filename).rsplit(".", 1)
                if Ext != "png":
                    filename = Pre + ".png"
            else:
                filename = filename + ".png"

            self.fig.savefig(os.path.join(dirname, filename), dpi=300)

        dlg.Destroy()

    # ---  ----Peak Search and Fit
    def SavePeakList_PSPfile(self, _):
        """
        save peak list .dat and save .psp file
        """
        print("Saving list of peaks in SavePeakList_PSPfile()")
        if self.peaklistPixels is None:
            wx.MessageBox("Peak list is empty !", "INFO")
        # write file with peak search parameters in comments line
        prefix, _ = self.imagefilename.rsplit(".", 1)

        comments_in_file = None

        finalfilename = prefix + "_LT_%d" % self.file_index_increment

        print("dirname", self.dirname)
        print("writefolder", self.writefolder)
        if self.dirname is not None and self.writefolder is None:
            outputfolder = self.dirname
            if not os.access(outputfolder, os.W_OK):
                self.OnFolderPreferences(1)
                outputfolder = self.writefolder
        else:
            outputfolder = self.writefolder

        print("self.peaklistPixels.shape", self.peaklistPixels.shape)

        if len(self.peaklistPixels.shape) == 1:
            nb_of_peaks = 1
        else:
            nb_of_peaks = self.peaklistPixels.shape[0]

        
        # writing ascii peaksearch parameters .psp file ---------
        pspfile_fullpath = os.path.join(outputfolder, "PeakSearch_%s.psp" % finalfilename)

        if self.dict_param is not None:

            dictparam = copy.copy(self.dict_param)
            dictparam.update(self.dict_param_LocalMaxima)

            RMCCD.savePeakSearchConfigFile(dictparam, outputfilename=pspfile_fullpath)

            params_comments = "Peak Search and Fit parameters\n"
            # usercomments = str(self.fitparampanel.peaksearchComments.GetValue())
            usercomments = ""
            comments_in_file = params_comments + "# user comments: " + usercomments


        mesg = "%d Peak(s) found"%nb_of_peaks +"\nPeakSearch Parameters written in %s\n"%pspfile_fullpath+"List of peaks will be written  in (.dat) file"
        userfilename = os.path.join(os.path.abspath(outputfolder), finalfilename+".dat")
        with wx.TextEntryDialog(self, mesg, caption='Save Peak list .dat file', value=userfilename) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                userfilename = dlg.GetValue()

        outputfolder, finalfilename = os.path.split(userfilename)

        print('finalfilename',finalfilename)
        if finalfilename.endswith('.dat'):
            outputfilename = finalfilename[:-4]
        else:
            outputfilename = finalfilename
        # writing ascii peak list  .dat file
        initialfilename=os.path.join(self.dirname, self.imagefilename)
        RMCCD.writepeaklist(self.peaklistPixels,
                                outputfilename,
                                outputfolder,
                                comments_in_file,
                                initialfilename)
        # MyMessageBox(None, "INFO", "%d Peak(s) found.\n List written in \n%s\n\nPeakSearch Parameters (.psp) file written in %s"
        #     % (nb_of_peaks, os.path.join(os.path.abspath(outputfolder), finalfilename + ".dat"),
        #         pspfile_fullpath))

        self.file_index_increment += 1

    def onSaveROIsList(self, _):
        """
        save rois list from current peaks list with boxsize of fitting procedure
        """

        if self.peaklistPixels is None:
            wx.MessageBox("Peak list is empty !", "INFO")

        print("Saving list of rois from peaks")

        prefix, _ = self.imagefilename.rsplit(".", 1)
        finalfilename = prefix + "_LT_%d" % self.file_index_increment

        if self.dirname is not None:
            outputfolder = self.dirname
        else:
            outputfolder = self.writefolder

        # halfboxsize for all rois
        boxsize = int(self.fitparampanel.boxsize.GetValue())

        if len(self.peaklistPixels.shape) == 1:
            rois = np.array([self.peaklistPixels[0], self.peaklistPixels[1],boxsize])
        else:
            nb_of_peaks = self.peaklistPixels.shape[0]
            boxsizearray = boxsize * np.ones(nb_of_peaks)
            rois = np.array([self.peaklistPixels[:,0], self.peaklistPixels[:,1], boxsizearray, boxsizearray])

        fullpathlistrois = os.path.join(outputfolder, finalfilename + '.rois')
        f = open(fullpathlistrois, 'w')
        np.savetxt(f, rois.T)
        f.close()

        wx.MessageBox('List of ROIs from peaks saved in %s' % fullpathlistrois, 'INFO')

    def onFitOnePeak(self, _):
        """
        fit one peak centered on where user has clicked

        in displayed image coordinates: self.centerx, self.centery
        """
        self.boxsize_fit = int(self.fitparampanel.boxsize.GetValue())

        # boxx, boxy = self.boxsize_fit, self.boxsize_fit

        print("self.framedim in onFitOnePeak", self.framedim)

        # patch switch: framedim
        framedim = self.framedim[1], self.framedim[0]

        (min_value, max_value, min_position, max_position) = ImProc.getExtrema(self.dataimage_ROI,
                                                                        [np.round(self.centerx),
                                                                        np.round(self.centery)],
                                                                        self.boxsize_fit,
                                                                        framedim,
                                                                        ROIcoords=0,
                                                                        flipxycenter=0)

        print("min,max,posmin,posmax", (min_value, max_value, min_position, max_position))

        print("Highest intensity %.f at (X,Y): (%d,%d) "
            % (max_value, max_position[0], max_position[1]))
        print("Peak Amplitude estimate :", max_value - min_value)

        print("Integrated Intensity",
            ImProc.getIntegratedIntensity(self.dataimage_ROI,
                                            [np.round(self.centerx), np.round(self.centery)],
                                            self.boxsize_fit,
                                            framedim,
                                            thresholdlevel=0.2,
                                            flipxycenter=0))

        self.guessed_amplitude = max_value - min_value
        self.guessed_bkg = min_value

        if str(self.fitparampanel.fitfunc.GetValue()) in ("NoFit",):
            self.last_peakfit_result = np.array([np.round(self.centerx),
                                                    np.round(self.centery),
                                                    max_value,
                                                    -1.0, -1.0, -1.0, 0.0, 0.0,
                                                    min_value,
                                                    max_value])
        else:
            self.OnFit()

        print("got peak with properties:", self.last_peakfit_result)

        if self.last_peakfit_result is not None:

            if self.allways_addpeak_chck.GetValue():
                self.onAddPeaktoPeaklist(1)

    def OnFit(self):
        """
        fit image array in a ROI with a 2D gaussian shape
        """
        POSITION_DEFINITION = 1

        self.position_definition = POSITION_DEFINITION

        center_pixel = np.round(self.centerx), np.round(self.centery)

        # trick:  ask to fit the same peak twice to use mutlipeaks fitting procedure...
        peaklist = np.array([center_pixel, center_pixel])
        boxsize = xboxsize, yboxsize = self.boxsize_fit, self.boxsize_fit

        filename = self.imagefilename
        dirname = os.path.abspath(self.dirname)
        CCDLabel = self.CCDlabel

        # use image resulting form formula e.g. : A =  A-B
        if self.ImageFilterpanel.usealsoforfit.GetValue():
            use_data_corrected = (self.dataimage_ROI_display, self.framedim, self.fliprot)
            reject_negative_baseline = False
        else:
            use_data_corrected = None
            reject_negative_baseline = True

        guessed_peaksize = float(self.fitparampanel.peaksizectrl.GetValue())

        FitPixelDev = float(self.fitparampanel.FitPixelDev.GetValue())

        tabIsorted, params_res, _ = RMCCD.fitoneimage_manypeaks(filename,
                                            peaklist,
                                            boxsize,
                                            CCDLabel=CCDLabel,
                                            dirname=dirname,
                                            position_start="max",
                                            type_of_function="gaussian",
                                            guessed_peaksize=(guessed_peaksize, guessed_peaksize),
                                            xtol=0.001,  # accept all
                                            FitPixelDev=FitPixelDev,  # accept all pixel deviation
                                            Ipixmax=None,
                                            verbose=0,
                                            position_definition=self.position_definition,
                                            use_data_corrected=use_data_corrected,
                                            reject_negative_baseline=reject_negative_baseline,
                                            computerrorbars=True)

        print("tabIsorted", tabIsorted)

        if tabIsorted is None:
            wx.MessageBox("Sorry. There is not peak around the region you cliked on...", "info")
            return

        # peak_X, peak_Y,peak_I, peak_fwaxmaj, peak_fwaxmin,peak_inclination, Xdev, Ydev, peak_bkg, Ipixmax
        datapeak = tabIsorted[0]
        params_res = params_res[0]

        (peak_X, peak_Y, peak_I, peak_fwaxmaj, peak_fwaxmin,
            peak_inclination, _, _, peak_bkg, _) = datapeak[:10]
        

        fitresults = [peak_bkg, peak_I, peak_X, peak_Y, peak_fwaxmaj, peak_fwaxmin, peak_inclination]

        print("fitresults", fitresults)
        if len(datapeak)==12: # computerrorbars == True
            print('Error Bars on X and Y:',datapeak[10:12])

        # parameter for function:
        params_loc = copy.copy(params_res)
        # [start_baseline, start_amplitude, start_j, start_i,  start_sigma1, start_sigma2,   start_anglerot])
        params_loc[2] = params_res[3] + yboxsize - center_pixel[1]
        params_loc[3] = params_res[2] + xboxsize - center_pixel[0]

        #        if position_definition == 1:
        #            params_loc[3] -= 1.
        #            params_loc[2] -= 1.

        #        print "center_pixel", center_pixel
        #        print "params_res", params_res
        #        print "params_loc", params_loc

        if self.plot_singlefitresults_chck.GetValue():  # showplot:
            framedim = self.framedim
            # patch ------------------------------------
            if self.CCDlabel in ("VHR_PSI","EIGER_4M"):#, "EIGER_4MCdTe"):
                framedim = self.framedim[1], self.framedim[0]
            # ----------------------------
            # crop data for local fit
            indicesborders = RMCCD.getindices2cropArray(center_pixel, [xboxsize, yboxsize],
                                                                        framedim, flipxycenter=0)
            imin, imax, jmin, jmax = indicesborders

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax, jmin, jmax,
                                                                            framedim=self.framedim)

            dat_ROIpeak = self.dataimage_ROI[imin:imax, jmin:jmax]

            #            print "max in dat_ROIpeak", np.amax(dat_ROIpeak)

            fitfunc = fit2d.twodgaussian(params_loc, 0, 1, 1)

            if np.all(dat_ROIpeak > 0):
                print("logscale")

                ploplo = IMSHOW.ImshowFrame(self, -1, self.imagefilename, dat_ROIpeak,
                                            center=center_pixel,
                                            boxsize=(xboxsize, yboxsize),
                                            fitfunc=fitfunc,
                                            fitresults=fitresults,
                                            cmap=GT.GIST_EARTH_R,
                                            interpolation="nearest",
                                            origin="upper",
                                            logscale=1)

            else:
                ploplo = IMSHOW.ImshowFrame(self, -1, self.imagefilename, dat_ROIpeak,
                                            center=center_pixel,
                                            boxsize=(xboxsize, yboxsize),
                                            fitfunc=fitfunc,
                                            fitresults=fitresults,
                                            cmap=GT.GIST_EARTH_R,
                                            interpolation="nearest",
                                            origin="upper",
                                            logscale=0)

            ploplo.Show(True)
            # imsave("testimage",log(dat),format='png')

        self.last_peakfit_result = datapeak

        return datapeak

    def onAddPeaktoPeaklist(self, _):
        dist_tolerance = 1

        if self.peaklistPixels is None:
            self.peaklistPixels = self.last_peakfit_result
            newpeak = self.last_peakfit_result
        else:
            # all peaks
            data_current_peaks = self.peaklistPixels
            #            print "data_current_peaks.shape", data_current_peaks.shape
            if data_current_peaks.shape == (10,):
                XYpeaklist = [data_current_peaks[:2]]
            else:
                XYpeaklist = data_current_peaks[:, :2]

            # new fitted peak
            XY = self.last_peakfit_result[:2]
            #            print "len(self.last_peakfit_result)", len(self.last_peakfit_result)

            acceptPeak = False
            posclose, dist = GT.FindClosestPoint(XYpeaklist, XY, returndist=1)

            if dist[posclose] <= dist_tolerance:
                print("peak at (%d,%d) has been updated" % (XY[0], XY[1]))

                # peak already exists, reset to new input values
                # data_current_peaks[posclose] = np.zeros(11)  # to test
                data_current_peaks[posclose] = self.last_peakfit_result
            else:
                # definitively a new peak to be added
                newpeak = self.last_peakfit_result

                acceptPeak = True

            if acceptPeak:
                # merge:
                if data_current_peaks.shape == (10,):
                    data_current_peaks = np.concatenate(([data_current_peaks], [newpeak]), axis=0)
                else:
                    data_current_peaks = np.concatenate((data_current_peaks, [newpeak]), axis=0)

            # sort by third column = peak amplitude
            self.peaklistPixels = data_current_peaks[
                np.argsort(data_current_peaks[:, 2])[::-1]]


        # update marker display
        self.plotPeaks = True
        self.OnReplot(1)
        self.plotPeaks = False

        # add object in OLV peak list
        if ObjectListView_Present:
            print("newpeak to be added", newpeak)
            print("with %d elements", len(newpeak))
            self.page4.AddOneSpot(newpeak)

    def onRemovePeaktoPeaklist(self, _, centerXY=None):
        """remove picked peak from the current peaks list
        """
        if self.peaklistPixels is None:
            wx.MessageBox("Peak list is empty!", "INFO")
            return

        closestPeak = self.getClosestPeak(centerXY=centerXY)
        if closestPeak is not None:
            peakProperties, index_close = closestPeak
            peakXY = peakProperties[:2]

            self.deleteOnePeak(index_close, peakXY)

    def onRemoveAllPeakstoPeaklist(self, _):
        """
        remove all spots of the peaks list and update the plot (remove circular markers)
        """
        self.page4.OnRemoveAll(1)

    def getClosestPeak(self, centerXY=None):
        """
        return peak in self.peaklistPixels
        that is close to the clicked pixel position or the given value
        """
        TOLERANCE_DIST = 20

        if centerXY is None:
            # where user have clicked
            center_pixel = [int(self.centerx), int(self.centery)]
        else:
            center_pixel = [int(centerXY[0]), int(centerXY[1])]


        index_close, dist = GT.FindClosestPoint(self.peaklistPixels[:, :2], center_pixel,
                                                                                    returndist=1)

        if np.amin(dist) > TOLERANCE_DIST:
            wx.MessageBox("No Peak in PeakList found close to this pixel position within %d pixels"
                % TOLERANCE_DIST, "INFO")
            return None

        peakProperties = self.peaklistPixels[index_close]

        return peakProperties, index_close

    def deleteOnePeak(self, index_close, peakXY):
        """
        delete one peak and update display and lists
        """

        print("Deleting peak #:%d  at (%.1f,%.1f)" % (index_close, peakXY[0], peakXY[1]))
        self.peaklistPixels = np.delete(self.peaklistPixels, index_close, 0)

        # update display

        # remove marker on image
        self.plotPeaks = True
        self.OnReplot(1)
        self.plotPeaks = False

        # delete object in OLV peak list
        if ObjectListView_Present:
            self.page4.RemoveOneSpot(peakXY)

    def deleteAllPeaks(self):
        """
        delete all peaks and update display and lists
        """

        #         print "Deleting peak #:%d  at (%.1f,%.1f)" % (index_close, peakXY[0], peakXY[1])
        self.peaklistPixels = None

        # update display

        # remove marker on image
        self.plotPeaks = True
        self.OnReplot(1)
        self.plotPeaks = False

        # delete object in OLV peak list
        if ObjectListView_Present:
            # self.page4.RemoveOneSpot(peakXY) # ??
            self.page4.OnRemoveAll(1)  # ??

    def OnPeakSearch(self, _):  # python & Lauetools
        """
        launch peak search by calling methods in readmccd.py
        """
        NB_MAX_FITS = int(self.fitparampanel.NbMaxFits.GetValue())

        currentLocalMaximaMethod = self.nb.GetCurrentPage()
        if currentLocalMaximaMethod.methodnumber in (5,6,):
            wx.MessageBox("Select one of the four tabs for the local Maxima Search Method", "INFO")
            return

        self.method = currentLocalMaximaMethod.methodnumber

        print("self.method for finding local maxima ", self.method)

        # read fitting function selected by user
        fitfunc = str(self.fitparampanel.fitfunc.GetValue())
        
        fitfunc1 = str(self.fitparampanel.fitfunc.GetValue())
        
        print("fitfunc for fitting ", fitfunc)
        if fitfunc == "NoFit":
            fit_peaks_gaussian = 0
        elif fitfunc == "Gaussian":
            fit_peaks_gaussian = 1
        elif fitfunc == "Lorentzian":
            fit_peaks_gaussian = 2

        print("fit_peaks_gaussian", fit_peaks_gaussian)

        # default offset to be compatible with XMAS convention of array reading
        self.position_definition = 1
        
        if self.method == 4:
            boxsizeSKIMAGE = currentLocalMaximaMethod.BS.GetValue()
            fitfunc1 = str(currentLocalMaximaMethod.fitfunc_peak.GetValue())
            processMode = str(currentLocalMaximaMethod.processMode.GetValue())
            if processMode =="multiprocessing":
                multip_peak = True
            else:
                multip_peak = False
            if fitfunc1 == "NoFit":
                fit_peaks_ = 0
            elif fitfunc1 == "Gaussian_nobounds":
                fit_peaks_ = 1
            elif fitfunc1 == "Gaussian_Relaxedbounds":
                fit_peaks_ = 2 
            elif fitfunc1 == "Gaussian_Strictbounds":
                fit_peaks_ = 3
        else:
            boxsizeSKIMAGE = self.fitparampanel.boxsize.GetValue()
        # build dict of common parameters --------------------------------------
        list_param_key = ["IntensityThreshold",
                        "PixelNearRadius",
                        "boxsize",
                        "xtol",
                        "FitPixelDev",
                        "MaxIntensity",
                        "MinIntensity",
                        "MaxPeakSize",
                        "MinPeakSize",
                        "boxsizeSKIMAGE"]

        list_param_val = [currentLocalMaximaMethod.IT.GetValue(),
                            currentLocalMaximaMethod.PNR.GetValue(),
                            self.fitparampanel.boxsize.GetValue(),
                            float(self.fitparampanel.xtol.GetValue()),
                            float(self.fitparampanel.FitPixelDev.GetValue()),
                            float(self.fitparampanel.maxIctrl.GetValue()),
                            float(self.fitparampanel.minIctrl.GetValue()),
                            float(self.fitparampanel.peaksizemaxctrl.GetValue()),
                            float(self.fitparampanel.peaksizeminctrl.GetValue()),
                            boxsizeSKIMAGE]

        self.dict_param = {}
        for key, val in zip(list_param_key, list_param_val):
            self.dict_param[key] = val
        # ----------------------------------------------------------------
        # Local Maxima search + fit

        imagefilename = os.path.join(self.dirname, self.imagefilename)

        reject_negative_baseline = True

        Data_for_localMaxima = None
        Fit_with_Data_for_localMaxima = False
        formulaexpression = "A-B"

        if self.ImageFilterpanel.UseImage.GetValue():
            # update raw data and use formula (convolution may be done later)
            Data_for_localMaxima = self.OnUseFormula(1)
            reject_negative_baseline = False
            formulaexpression = str(self.ImageFilterpanel.formulatxtctrl.GetValue())

            if self.ImageFilterpanel.usealsoforfit.GetValue():
                print(" Fit_with_Data_for_localMaxima = True")
                Data_for_localMaxima = self.BImageFilename
                Fit_with_Data_for_localMaxima = True

        elif self.ImageFilterpanel.FilterImage.GetValue():
            # use only filtered image for finding blobs (local maxima)
            Data_for_localMaxima = self.ImageFilterpanel.filteredimage
            #             Data_for_localMaxima = 'auto_background'
            reject_negative_baseline = True

        Remove_BlackListedPeaks_fromfile = None
        maxPixelDistanceRejection = 0
        if self.ImageFilterpanel.RemoveBlackpeaks.GetValue():
            #             Remove_BlackListedPeaks_fromfile = os.path.join(self.dirnameBlackList,
            #                                                             self.BlackListFilename)
            Remove_BlackListedPeaks_fromfile = self.BlackListFilename
            maxPixelDistanceRejection = (self.ImageFilterpanel.BlackListRejection_pixeldistanceMax.GetValue())
            self.dict_param["maxPixelDistanceRejection"] = maxPixelDistanceRejection

        self.dict_param_LocalMaxima = {}
        computerrorbars = True
        if self.method == 1:  # basic local maxima search (threshold on raw intensity)
            print('method threshold')
            print(self.stackimageindex, NB_MAX_FITS, self.dict_param["PixelNearRadius"],
                  self.dict_param["IntensityThreshold"], self.dict_param["boxsize"], fit_peaks_gaussian,
                  self.dict_param["xtol"], self.dict_param["FitPixelDev"], self.dict_param["MinIntensity"], Data_for_localMaxima, Fit_with_Data_for_localMaxima, maxPixelDistanceRejection, formulaexpression)
            ResPeakSearch = RMCCD.PeakSearch(imagefilename,
                            stackimageindex=self.stackimageindex,
                            CCDLabel=self.CCDlabel,
                            NumberMaxofFits=NB_MAX_FITS,
                            PixelNearRadius=self.dict_param["PixelNearRadius"],
                            removeedge=2,
                            IntensityThreshold=self.dict_param["IntensityThreshold"],
                            local_maxima_search_method=0,
                            # thresholdConvolve = 1000,
                            boxsize=self.dict_param["boxsize"],
                            position_definition=self.position_definition,
                            verbose=1,
                            fit_peaks_gaussian=fit_peaks_gaussian,
                            xtol=self.dict_param["xtol"],
                            FitPixelDev=self.dict_param["FitPixelDev"],
                            return_histo=0,
                            Saturation_value=self.dict_param["MaxIntensity"],
                            Saturation_value_flatpeak=self.dict_param["MaxIntensity"],
                            MinIntensity=self.dict_param["MinIntensity"],
                            PeakSizeRange=(self.dict_param["MinPeakSize"], self.dict_param["MaxPeakSize"]),
                            write_execution_time=1,
                            Data_for_localMaxima=Data_for_localMaxima,
                            Fit_with_Data_for_localMaxima=Fit_with_Data_for_localMaxima,
                            reject_negative_baseline=reject_negative_baseline,
                            Remove_BlackListedPeaks_fromfile=Remove_BlackListedPeaks_fromfile,
                            maxPixelDistanceRejection=maxPixelDistanceRejection,
                            formulaexpression=formulaexpression,
                            computerrorbars=computerrorbars)

            self.dict_param_LocalMaxima["fit_peaks_gaussian"] = fit_peaks_gaussian
            self.dict_param_LocalMaxima["local_maxima_search_method"] = 0
            self.dict_param_LocalMaxima["position_definition"] = self.position_definition

        if self.method == 2:  # shifted array maxima search
            ResPeakSearch = RMCCD.PeakSearch(imagefilename,
                                    stackimageindex=self.stackimageindex,
                                    CCDLabel=self.CCDlabel,
                                    NumberMaxofFits=NB_MAX_FITS,
                                    PixelNearRadius=self.dict_param["PixelNearRadius"],
                                    removeedge=2,
                                    IntensityThreshold=self.dict_param["IntensityThreshold"],
                                    local_maxima_search_method=1,
                                    thresholdConvolve=1000,
                                    boxsize=self.dict_param["boxsize"],
                                    position_definition=self.position_definition,
                                    verbose=0,
                                    fit_peaks_gaussian=fit_peaks_gaussian,
                                    xtol=self.dict_param["xtol"],
                                    FitPixelDev=self.dict_param["FitPixelDev"],
                                    return_histo=0,
                                    Saturation_value=self.dict_param["MaxIntensity"],
                                    Saturation_value_flatpeak=self.dict_param["MaxIntensity"],
                                    MinIntensity=self.dict_param["MinIntensity"],
                                    PeakSizeRange=(self.dict_param["MinPeakSize"],
                                        self.dict_param["MaxPeakSize"]),
                                    write_execution_time=1,
                                    Data_for_localMaxima=Data_for_localMaxima,
                                    Fit_with_Data_for_localMaxima=Fit_with_Data_for_localMaxima,
                                    reject_negative_baseline=reject_negative_baseline,
                                    Remove_BlackListedPeaks_fromfile=Remove_BlackListedPeaks_fromfile,
                                    maxPixelDistanceRejection=maxPixelDistanceRejection,
                                    formulaexpression=formulaexpression,
                                    computerrorbars=computerrorbars)

            self.dict_param_LocalMaxima["fit_peaks_gaussian"] = fit_peaks_gaussian
            self.dict_param_LocalMaxima["local_maxima_search_method"] = 1
            self.dict_param_LocalMaxima["position_definition"] = self.position_definition

        if self.method == 3:  # convolution for local maxima search

            Thresconvolve = float(self.page3.ThresholdConvolveCtrl.GetValue())

            if self.fitparampanel.keepCentroid.GetValue():
                peakposition_definition = "center"
            else:
                peakposition_definition = "max"

            ResPeakSearch = RMCCD.PeakSearch(imagefilename,
                                        stackimageindex=self.stackimageindex,
                                        CCDLabel=self.CCDlabel,
                                        NumberMaxofFits=NB_MAX_FITS,
                                        PixelNearRadius=self.dict_param["PixelNearRadius"],
                                        removeedge=2,
                                        IntensityThreshold=self.dict_param["IntensityThreshold"],
                                        local_maxima_search_method=2,
                                        peakposition_definition=peakposition_definition,
                                        thresholdConvolve=Thresconvolve,
                                        boxsize=self.dict_param["boxsize"],
                                        paramsHat=self.paramsHat,
                                        position_definition=self.position_definition,
                                        verbose=0,
                                        fit_peaks_gaussian=fit_peaks_gaussian,
                                        xtol=self.dict_param["xtol"],
                                        FitPixelDev=self.dict_param["FitPixelDev"],
                                        return_histo=0,
                                        Saturation_value=self.dict_param["MaxIntensity"],
                                        Saturation_value_flatpeak=self.dict_param["MaxIntensity"],
                                        MinIntensity=self.dict_param["MinIntensity"],
                                        PeakSizeRange=(self.dict_param["MinPeakSize"],
                                                        self.dict_param["MaxPeakSize"]),
                                        write_execution_time=1,
                                        Data_for_localMaxima=Data_for_localMaxima,
                                        Fit_with_Data_for_localMaxima=Fit_with_Data_for_localMaxima,
                                        reject_negative_baseline=reject_negative_baseline,
                                        Remove_BlackListedPeaks_fromfile=Remove_BlackListedPeaks_fromfile,
                                        maxPixelDistanceRejection=maxPixelDistanceRejection,
                                        formulaexpression=formulaexpression,
                                        computerrorbars=computerrorbars)

            self.dict_param_LocalMaxima["fit_peaks_gaussian"] = fit_peaks_gaussian
            self.dict_param_LocalMaxima["local_maxima_search_method"] = 2
            self.dict_param_LocalMaxima["position_definition"] = self.position_definition
            self.dict_param_LocalMaxima["thresholdConvolve"] = Thresconvolve
        
        if self.method == 4:  # SKIMAGE PEAK SEARCH METHOD
            ResPeakSearch = RMCCD.peaksearch_skimage(imagefilename, 
                                                     self.dict_param["PixelNearRadius"], 
                                                     self.dict_param["IntensityThreshold"], 
                                                     self.dict_param["boxsizeSKIMAGE"], 
                                                     fit_peaks_, 
                                                     self.CCDlabel,
                                                     use_multiprocessing=multip_peak)
            
            self.dict_param_LocalMaxima["fit_peaks_gaussian"] = fit_peaks_gaussian
            self.dict_param_LocalMaxima["local_maxima_search_method"] = 0
            self.dict_param_LocalMaxima["position_definition"] = self.position_definition
            
        # print("ResPeakSearch", ResPeakSearch)
        if ResPeakSearch is not None:
            Isorted = ResPeakSearch[0]
            if Isorted is not None:
                self.peaklistPixels = Isorted
            else:
                wx.MessageBox("No Peaks found !! \n(some local maxima may have been rejected "
                                                                        "after fitting)", "INFO")
                return
        else:
            wx.MessageBox("Too many or two less Peaks to fit !! \nTry to change some thresholds\n "
                                                                "or Max. Nb of Fits", "INFO")
            return

        # save peaklist
        self.SavePeakList_PSPfile(1)

        # plot data and markers at peaks position
        self.plotPeaks = True
        self.OnReplot(1)
        self.plotPeaks = False


def start_func():
    startfolder = os.path.split(__file__)[0]

    initialParameter = {}
    initialParameter["writefolder"] = os.path.join(DictLT.LAUETOOLSFOLDER, "LaueImages")

    initialParameter["stackedimages"] = False
    initialParameter["stackimageindex"] = -1
    initialParameter["Nbstackedimages"] = 0

    initialParameter["title"] = "test_peaksearchframe"
    initialParameter["mapsLUT"] = "OrRd"

    initialParameter["dirname"] = os.path.join(DictLT.LAUETOOLSFOLDER, "LaueImages")
    initialParameter["imagefilename"] = "AH12_CMT_r14_0200.tif"
    initialParameter["CCDLabel"] = "sCMOS"

    _PSGUIApp = wx.App()
    _PSGUIframe = MainPeakSearchFrame(None, -1, initialParameter, "MainPeakSearchFrame")
    _PSGUIframe.Show()
    _PSGUIApp.MainLoop()


if __name__ == "__main__":

    start_func()
