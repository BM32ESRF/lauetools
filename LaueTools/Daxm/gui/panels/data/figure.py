#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np

import wx
from wx.lib.pubsub import pub

import matplotlib as mpl
import matplotlib.pyplot as mplp

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from LaueTools.Daxm.gui.widgets.combobox import LabelComboBox
from LaueTools.Daxm.gui.widgets.spin import SpinSliderCtrl
from LaueTools.Daxm.gui.widgets.text import LabelTxtCtrl, LabelTxtCtrlNum

from LaueTools.Daxm.utils.path import nbasename


class PanelDataFigure(wx.Panel):
    # constructors
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self._parent = parent
        self._mainbox = wx.BoxSizer(wx.VERTICAL)
        self._observers = []

        # figure canvas
        self._figure = None
        self._fig_axes = None
        self._fig_canvas = None
        self._fig_toolbar = None
        self._fig_plot = None
        self._fig_lastclick = None

        # img axes
        self._img = []
        self._img_orient = "ij"
        self._img_xlims = []
        self._img_ylims = []

        # img colormap and colorscale
        self._img_colormap = "CMRmap"
        self._img_scale = "linear"
        self._img_Imin = 0
        self._img_Iopt = 65535
        self._img_Imax = 65535
        self._img_scale_min = self._img_Imin
        self._img_scale_max = self._img_Imax

        # scan navitor
        self._scan_file = ""
        self._scan_frame = 1
        self._scan_qty = 100
        self._scan_pos = None

        self._disp_shadow = False
        self._disp_shadow_line = []
        self._disp_shadow_patch = []
        self._disp_shadow_label = []
        self._disp_centre = False
        self._disp_centre_pt = []
        self._disp_limits = False
        self._disp_limits_lines = []
        self._disp_monitor = False
        self._disp_monitor_roi = None

        self._Create()
        self.Init()

    def _Create(self):

        # create
        self._CreatePlot()

        self._CreateCtrl()

        # assemble
        self._mainbox.Add(self._fig_canvas,  1, wx.EXPAND)
        self._mainbox.Add(self._fig_toolbar, 0)

        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(self.img_box, 1, wx.EXPAND|wx.RIGHT, 5)
        box.Add(self.scan_box, 1, wx.EXPAND)
        self._mainbox.Add(box, 0, wx.EXPAND)

        # fit
        self.SetSizer(self._mainbox)
        self.Layout()
        self._mainbox.Layout()
        self._mainbox.Fit(self)
        self.Fit()

    def _CreatePlot(self):

        self._fig = Figure(tight_layout=True)
        self._fig_axes = self._fig.add_subplot(111)
        self._fig_canvas = FigureCanvas(self, wx.ID_ANY, self._fig)
        self._fig_toolbar = NavigationToolbar(self._fig_canvas)
        self._fig_plot = None

        #binding
        self._fig_canvas.mpl_connect('motion_notify_event', self.OnMouseMotion)
        self._fig_canvas.mpl_connect('key_press_event', self.OnKeyPress)

    def _CreateCtrl(self):

        # view
        img_sbox = wx.StaticBox(self, -1, "View")
        self.img_box = wx.StaticBoxSizer(img_sbox, wx.VERTICAL)
        img_sbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        img_sbox.SetForegroundColour('DIM GREY')

        cm_list = [m for m in mpl.pylab.cm.datad if not m.endswith("_r")]
        cm_list.sort()
        self.cm_cbx = LabelComboBox(self, label="Colormap:  ", style=wx.CB_SORT, value=self._img_colormap, choices=cm_list)
        self.ori_cbx = LabelComboBox(self, label="Axes:  ", style=wx.CB_SORT,
                                     value=self._img_orient, choices=["ij", "xy"])
        self.min_sld = SpinSliderCtrl(self, "Imin: ", size=(60, -1),
                                      initial=self._img_Imin, min=self._img_scale_min, max=self._img_scale_max)
        self.max_sld = SpinSliderCtrl(self, "Imax:", size=(60, -1), style=wx.SL_MIN_MAX_LABELS,
                                      initial=self._img_Imax, min=self._img_scale_min, max=self._img_scale_max)

        box = wx.GridBagSizer(vgap=5, hgap=5)
        box.Add(self.ori_cbx, pos=(0, 0), span=(1, 1), flag=wx.EXPAND)
        box.Add(self.cm_cbx, pos=(0, 1), span=(1, 1), flag=wx.EXPAND)
        box.Add(self.min_sld, pos=(1, 0), span=(1, 2), flag=wx.EXPAND)
        box.Add(self.max_sld, pos=(2, 0), span=(1, 2), flag=wx.EXPAND)

        for i in range(2):
            box.AddGrowableCol(i, 1)

        self.img_box.Add(box, 1, wx.EXPAND|wx.ALL, 5)

        # scan
        scan_sbox = wx.StaticBox(self, -1, "Scan display")
        self.scan_box = wx.StaticBoxSizer(scan_sbox, wx.VERTICAL)
        scan_sbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        scan_sbox.SetForegroundColour('DIM GREY')

        self.file_txt = LabelTxtCtrl(self, "Filename: ", "", style=wx.TE_READONLY)

        self.frame_sld = SpinSliderCtrl(self, "Frame: ", size=(60, -1),
                                        style=wx.SL_MIN_MAX_LABELS | wx.SL_TOP,
                                        initial=1, min=1, max=self._scan_qty)

        self.wirep_txt = LabelTxtCtrlNum(self, "Motor position: ", 0, style=wx.TE_READONLY)

        self.center_ckb = wx.CheckBox(self, wx.ID_ANY, label="Show detector center")
        self.shadow_ckb = wx.CheckBox(self, wx.ID_ANY, label="Show wire shadow")
        self.lims_ckb = wx.CheckBox(self, wx.ID_ANY, label="Show scan limits")
        self.mon_ckb = wx.CheckBox(self, wx.ID_ANY, label="Show monitor ROI")

        box = wx.GridBagSizer(vgap=5, hgap=5)
        box.Add(self.file_txt, pos=(0, 0), span=(1, 2), flag=wx.EXPAND)
        box.Add(self.frame_sld, pos=(1, 0), span=(1, 2), flag=wx.EXPAND)
        box.Add(self.wirep_txt, pos=(2, 0), span=(1, 1), flag=wx.EXPAND)
        box.Add(self.shadow_ckb, pos=(3, 0), span=(1, 1))
        box.Add(self.lims_ckb, pos=(3, 1), span=(1, 1))
        box.Add(self.center_ckb, pos=(4, 0), span=(1, 1))
        box.Add(self.mon_ckb, pos=(4, 1), span=(1, 1))

        for i in range(2):
            box.AddGrowableCol(i, 1)

        self.scan_box.Add(box, 1, wx.EXPAND|wx.ALL, 5)


        # bindings
        self.cm_cbx.Bind_to(self.OnSelectColormap)
        self.ori_cbx.Bind_to(self.OnSelectAxes)
        self.min_sld.Bind_to(self.OnChangeImin)
        self.max_sld.Bind_to(self.OnChangeImax)

        self.frame_sld.Bind_to(self.OnChangeFrame)
        self.shadow_ckb.Bind(wx.EVT_CHECKBOX, self.OnEnableShadow)
        self.center_ckb.Bind(wx.EVT_CHECKBOX, self.OnEnableCenter)
        self.lims_ckb.Bind(wx.EVT_CHECKBOX, self.OnEnableLimits)
        self.mon_ckb.Bind(wx.EVT_CHECKBOX, self.OnEnableMonitor)


    def Init(self):

        self.UpdateValues()

        self.UpdateFig(self.GetScan().get_image(self._scan_frame - 1))

        self.UpdateWidgets()

        self._fig.colorbar(self._fig_plot, fraction=0.1)

        self._fig_canvas.draw()

    # Getters
    def ToggledTool(self):

        return self._fig_toolbar._active == "PAN" \
               or self._fig_toolbar._active == "ZOOM"

    def GetScan(self):
        return self._parent.scan

    # Updaters
    def Update(self, part="all"):

        self.UpdateValues()

        self.UpdateWidgets()

        if part in ("setup", "all"):
            self.UpdateFigLims()

            self.UpdateFigColormap()

            self.ShowDetectorCentre()

        if part in ("all", "spec", "wire"):
            self.ShowScanLimits()

            self.ShowWireShadow()

        if part in ("all", "mon"):
            self.ShowMonitorRegion()

        if part in ("all", "setup", "data", "spec", "move"):
            self.UpdateFigImage()

        self._fig_canvas.draw()

    def UpdateFig(self, img=None):

        self.UpdateFigImage(img)

        self.UpdateFigLims()

        self.UpdateFigScale()

        self.UpdateFigColormap()

    def UpdateFigImage(self, img=None):

        if img is not None:
            self._img = img
            self._fig_axes.clear()
            self._fig_axes.set_autoscale_on(False)
            self._fig_plot = self._fig_axes.imshow(self._img.transpose(), interpolation='nearest')
        else:
            self._img = self.GetScan().get_image(self._scan_frame - 1)
            self._fig_plot.set_data(self._img.transpose())

    def UpdateFigLims(self, xlims=None, ylims=None):

        if xlims is not None:
            self._img_xlims = xlims

        if ylims is not None:
            self._img_ylims = ylims

        self._fig_axes.set_xlim(self._img_xlims)

        if self._img_orient == "xy":

            self._fig_axes.set_ylim(bottom=min(self._img_ylims),
                                    top=max(self._img_ylims))

        else:

            self._fig_axes.set_ylim(bottom=max(self._img_ylims),
                                    top=min(self._img_ylims))

    def UpdateFigScale(self, vmax=None, vmin=None):

        if vmax is None:
            img = self._img #self.GetScan().get_image(self.scan_img_frame - 1)

            vmax = int(np.mean(img, axis=(0, 1))) + 4 * int(np.std(img, axis=(0, 1)))

        if vmin is None:
            img = self._img #self.GetScan().get_image(self.scan_img_frame - 1)

            vmin = int(np.mean(np.min(img, axis=(1,))))

        self._img_scale_min = np.maximum(vmin, 0)

        self._img_scale_max = np.maximum(vmax, vmin+1)

    def UpdateFigScaleRange(self):

        if self._img_scale_max < self._img_Iopt*0.39:

            while self._img_scale_max < self._img_Iopt*0.39:

                self._img_Iopt = int(0.5*self._img_Iopt)

        elif self._img_Iopt != self._img_Imax and self._img_scale_max > self._img_Iopt*0.81:

            self._img_Iopt = max(2.*self._img_Iopt, self._img_Imax)

        else:

            pass

    def UpdateFigColormap(self):

        self._fig_plot.set_cmap(self._img_colormap)

        norm = mpl.colors.Normalize(vmin=self._img_scale_min,
                                    vmax=self._img_scale_max)

        self._fig_plot.set_norm(norm)

    def UpdateValues(self):

        self.UpdateValuesView()

        self.UpdateValuesScan()

    def UpdateValuesView(self):

        scan = self.GetScan()

        # view
        self._img_Imax = scan.get_img_params(('saturation',))

        if self._img_scale_min >= self._img_Imax:
            self._img_scale_min = self._img_Imin

        self._img_scale_max = min(self._img_scale_max, self._img_Imax)

        self._img_xlims = [0, scan.get_img_params(['framedim'])[0] - 1]

        self._img_ylims = [0, scan.get_img_params(['framedim'])[1] - 1]

    def UpdateValuesScan(self, frame=None):

        scan = self.GetScan()

        if frame is not None:
            self._scan_frame = frame

        if self._scan_frame > self._scan_qty:
            self._scan_frame = 1

        self._scan_pos = scan.wire_position[self._scan_frame - 1]

        #self._scan_file = scan.img_filenames[self._scan_frame - 1]
        file = scan.get_image_filedir(self._scan_frame - 1)

        if scan.get_type() == "mesh" and scan.line_subfolder:
            self._scan_file = nbasename(file, 2)
        else:
            self._scan_file = nbasename(file, 1)

        if not scan.img_exist[self._scan_frame - 1]:
            self._scan_file = self._scan_file + " - FILE MISSING!"

        self._scan_qty = scan.number_images

    def UpdateWidgets(self):

        self.UpdateWidgetsView()

        self.UpdateWidgetsScan()

    def UpdateWidgetsView(self):

        self.UpdateFigScaleRange()

        self.min_sld.SetValue(self._img_scale_min)

        self.max_sld.SetValue(self._img_scale_max)

        self.min_sld.SetMaxValue(self._img_Iopt)

        self.max_sld.SetMaxValue(self._img_Iopt)

    def UpdateWidgetsScan(self):

        self.frame_sld.SetMaxValue(self._scan_qty)

        self.frame_sld.SetValue(self._scan_frame)

        self.wirep_txt.SetValue(self._scan_pos)

        self.file_txt.SetValue(self._scan_file)

    # Event handlers
    def OnSelectColormap(self, event):

        self._img_colormap = event.GetValue()

        self.UpdateFigColormap()

        self._fig_canvas.draw()

    def OnSelectAxes(self, event):

        self._img_orient = event.GetValue()

        self.UpdateFigLims()

        self._fig_canvas.draw()

    def OnChangeImin(self, event):

        Imin = event.GetValue()

        if Imin > self._img_scale_max:
            Imin = self._img_scale_max

            event.SetValue(Imin)

        self._img_scale_min = Imin

        self.UpdateFigColormap()

        self._fig_canvas.draw()

    def OnChangeImax(self, event):

        Imax = event.GetValue()

        if Imax < self._img_scale_min:
            Imax = self._img_scale_min

            event.SetValue(Imax)

        self._img_scale_max = Imax

        self.UpdateFigScaleRange()

        self.UpdateWidgetsView()

        self.UpdateFigColormap()

        self._fig_canvas.draw()

    def OnChangeFrame(self, event):

        self.UpdateValuesScan(event.GetValue())

        self.UpdateWidgetsScan()

        self.UpdateFigImage()

        self.ShowWireShadow()

        self._fig_canvas.draw()

    def OnEnableShadow(self, event):

        self._disp_shadow = event.GetEventObject().IsChecked()

        self.ShowWireShadow()

        self._fig_canvas.draw()

    def OnEnableCenter(self, event):

        self._disp_centre = event.GetEventObject().IsChecked()

        self.ShowDetectorCentre()

        self._fig_canvas.draw()

    def OnEnableLimits(self, event):

        self._disp_limits = event.GetEventObject().IsChecked()

        self.ShowScanLimits()

        self._fig_canvas.draw()

    def OnEnableMonitor(self, event):

        self._disp_monitor = event.GetEventObject().IsChecked()

        self.ShowMonitorRegion()

        self._fig_canvas.draw()

    def OnMouseMotion(self, event):

        if event.inaxes == self._fig_axes:

            X, Y = event.xdata, event.ydata

            I = self._img[int(X), int(Y)]

            msg = ("Pixel info: X={}, Y={}, I={}").format(int(X), int(Y), I)

            pub.sendMessage('set_status_text', msg=msg)
        else:
            pass

    def OnKeyPress(self, event):

        if (event.inaxes == self._fig_axes):

            if self._disp_monitor and event.key == 'm':

                print("Moving monitor position to (%d, %d)..."%(event.xdata, event.ydata))

                scan_dict = self.GetScan().to_dict("mon")

                _, _, sx, sy = scan_dict['monitorROI']

                xcen, ycen = int(event.xdata), int(event.ydata)

                scan_dict['monitorROI'] = (xcen, ycen, sx, sy)

                myevent = FigureEvent(part="mon", args=scan_dict)

                self.Notify(myevent)

                self.ShowMonitorRegion()

                self._fig_canvas.draw()

            elif event.key == 'p':

                print("Plotting profile at (%d, %d)..."%(event.xdata, event.ydata))

                y, x = self.GetScan().get_profile_pixel_full([event.xdata, event.ydata], [1, 0])
                y0 = self.GetScan().get_img_params(('offset',))

                fig = mplp.figure()

                ax = fig.add_subplot(111)

                ax.plot(x, y + y0, 'bx')

                ax.set_ylim(bottom=min(y0, ax.get_ylim()[0]))

                xlabel = 'depth (mm)'
                ylabel = 'Intensity'
                title = 'Scan profile at (%d, %d)'%(event.xdata, event.ydata)

                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                fig.suptitle(title, fontsize=16)

                fig.show()

            elif event.key == 'v':

                xver = int(event.xdata)

                print("Plotting vertical profile along X = %d" % (xver,))

                xmin=max(xver - 1, 0)
                xmax=min(xver + 2, self._img_xlims[1])

                I = np.mean(self._img[xmin:xmax,:], axis=(0,))
                I0 = self.GetScan().get_img_params(('offset',))

                fig = mplp.figure()

                ax = fig.add_subplot(111)

                ax.plot(I, 'kx', I, 'r')

                ax.set_ylim(bottom=min(I0, ax.get_ylim()[0]))

                xlabel = 'Y (pixels)'
                ylabel = 'Intensity'
                title = 'Intensity profile along X = %d' % (xver,)

                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                fig.suptitle(title, fontsize=16)

                # TODO: shadows

                fig.show()

            elif event.key == 'h':

                yhor = int(event.ydata)

                print("Plotting horizontal profile along Y = %d" % (yhor,))

                ymin=max(yhor-1, 0)
                ymax=min(yhor+2, self._img_ylims[1])

                I = np.mean(self._img[:, ymin:ymax], axis=(1,))
                I0 = self.GetScan().get_img_params(('offset',))

                fig = mplp.figure()

                ax = fig.add_subplot(111)

                ax.plot(I, 'kx', I, 'r')

                ax.set_ylim(bottom=min(I0, ax.get_ylim()[0]))

                xlabel = 'Y (pixels)'
                ylabel = 'Intensity'
                title = 'Intensity profile along X = %d' % (yhor,)

                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                fig.suptitle(title, fontsize=16)

                # TODO: shadows

                fig.show()

            else:
                pass

    # ---------- ~~~ Figure update  ~~~

    def ShowDetectorCentre(self):

        if self._disp_centre:

            xcen, ycen = self.GetScan().get_ccd_params(['xcen', 'ycen'])

            if not self._disp_centre_pt:

                pt1 = self._fig_axes.plot([xcen], [ycen],
                                     marker='x', markeredgecolor='r',
                                     markeredgewidth=1.1, markersize=10)
                pt2 = self._fig_axes.plot([xcen], [ycen], fillstyle='none',
                                     marker='s', markeredgecolor='r',
                                     markeredgewidth=1.1, markersize=10)
                self._disp_centre_pt.extend(pt1)
                self._disp_centre_pt.extend(pt2)

            else:

                for item in self._disp_centre_pt:
                    item.set_xdata(xcen)
                    item.set_ydata(ycen)

        else:
            pass

        if self._disp_centre_pt:
            for item in self._disp_centre_pt:
                item.set_visible(self._disp_centre)

    def ShowWireShadow(self):

        if self._disp_shadow:

            scan = self.GetScan()

            xmin, xmax = 0, scan.get_img_params(['framedim'])[0]
            ys_right = scan.calc_wires_range_shadow(self._scan_frame - 1, xcam=xmax)
            ys_left = scan.calc_wires_range_shadow(self._scan_frame - 1, xcam=xmin)

            align = 'center'

            if len(self._disp_shadow_line) != len(ys_right):
                for lines, patches in zip(self._disp_shadow_line, self._disp_shadow_patch):
                    #for line in lines:
                    self._fig_axes.lines.remove(lines)
                    #for patch in patches:
                    self._fig_axes.patches.remove(patches)

                for txt in self._disp_shadow_label:
                    self._fig_axes.texts.remove(txt)

                self._disp_shadow_line = []
                self._disp_shadow_patch = []
                self._disp_shadow_label = []

            if len(self._disp_shadow_line) == 0:

                colors = mplp.rcParams['axes.prop_cycle'].by_key()['color']

                for i, ymin in enumerate(ys_left):
                    ymax = ys_right[i]
                    line = Line2D([xmin, xmax], [ymin[0], ymax[0]],
                                  antialiased=True, color=colors[i], linestyle='-')
                    poly = Polygon(list(zip([xmin, xmax, xmax, xmin],
                                       [ymin[1], ymax[1], ymax[2], ymin[2]])),
                                   closed=True, antialiased=True, alpha=0.3333, facecolor=colors[i])

                    txt =  self._fig_axes.text(xmax+10, ymin[0], str(i+1), fontsize='small',
                                               horizontalalignment='left', verticalalignment=align)

                    self._disp_shadow_line.append(line)
                    self._disp_shadow_patch.append(poly)
                    self._disp_shadow_label.append(txt)

                    self._fig_axes.add_line(line)
                    self._fig_axes.add_patch(poly)
            else:
                for k, line in enumerate(self._disp_shadow_line):
                    line.set_xdata([xmin, xmax])
                    line.set_ydata([ys_left[k][0], ys_right[k][0]])
                for k, poly in enumerate(self._disp_shadow_patch):
                    poly.set_xy(list(zip([xmin, xmax, xmax, xmin],
                                     [ys_left[k][1], ys_right[k][1], ys_right[k][2], ys_left[k][2]])))
                for k, txt in enumerate(self._disp_shadow_label):
                    txt.set_position((xmax-10, ys_left[k][1]))
                    txt.set_verticalalignment(align)

        if self._disp_shadow_patch:
            for item in self._disp_shadow_label:
                item.set_visible(self._disp_shadow)
            for item in self._disp_shadow_line:
                item.set_visible(self._disp_shadow)
            for item in self._disp_shadow_patch:
                item.set_visible(self._disp_shadow)

    def ShowScanLimits(self):

        for scan in self._disp_limits_lines:
            self._fig_axes.lines.remove(scan[0])
            self._fig_axes.lines.remove(scan[1])

        self._disp_limits_lines = []

        if self._disp_limits:

            scan = self.GetScan()

            colors = mplp.rcParams['axes.prop_cycle'].by_key()['color']

            ys = scan.calc_wires_range_scan()

            xmin, xmax = 0, scan.get_img_params(['framedim'])[0]

            for i, y in enumerate(ys):
                scan = []
                scan.extend(self._fig_axes.plot([xmin, xmax], [y[0]] * 2, linestyle='--', color=colors[i]))
                scan.extend(self._fig_axes.plot([xmin, xmax], [y[1]] * 2, linestyle='--', color=colors[i]))
                self._disp_limits_lines.append(scan)

    def ShowMonitorRegion(self):

        if self._disp_monitor and len(self.GetScan().monitor_roi):

            xcen, ycen, sx, sy = self.GetScan().monitor_roi

            xcorners = [xcen - sx, xcen + sx, xcen + sx, xcen - sx]
            ycorners = [ycen - sy, ycen - sy, ycen + sy, ycen + sy]

            if self._disp_monitor_roi is None:

                pt = self._fig_axes.plot([xcen], [ycen],
                                          marker='x', markeredgecolor='k',
                                          markeredgewidth=1.1, markersize=5)
                poly = Polygon(list(zip(xcorners, ycorners)), closed=True,
                               antialiased=True, alpha=0.3333, facecolor='w')

                self._fig_axes.add_patch(poly)

                self._disp_monitor_roi = [pt[0], poly]

            else:
                self._disp_monitor_roi[0].set_xdata(xcen)
                self._disp_monitor_roi[0].set_ydata(ycen)
                self._disp_monitor_roi[1].set_xy(list(zip(xcorners, ycorners)))

        else:
            pass

        if self._disp_monitor_roi is not None:
            self._disp_monitor_roi[0].set_visible(self._disp_monitor)
            self._disp_monitor_roi[1].set_visible(self._disp_monitor)

    # Binders
    def Bind_to(self, callback):
        self._observers.append(callback)

    def Notify(self, event):
        # print self.GetValue()
        for callback in self._observers:
            callback(event)


# Event class
class FigureEvent(wx.CommandEvent):

    def __init__(self, part=None, args=None):
        wx.CommandEvent.__init__(self)

        if part is None:
            part = "all"

        if args is None:
            args = {}

        self.part = part
        self.args = args

    def GetPart(self):
        return self.part

    def GetArgs(self):
        return self.args