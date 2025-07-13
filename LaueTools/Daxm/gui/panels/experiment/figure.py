#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

import matplotlib.pyplot as mplp

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from LaueTools.Daxm.gui.widgets.combobox import LabelComboBox
from LaueTools.Daxm.gui.widgets.spin import SpinSliderCtrl

from LaueTools.Daxm.utils.num import clamp


class PanelExperimentFigure(wx.Panel):

    # Constructors
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self._parent = parent
        self._mainbox = wx.BoxSizer(wx.VERTICAL)

        self._figure = None
        self._axes = None
        self._canvas = None
        self._toolbar = None
        self._plot = None
        self._img = []
        self._img_orient = "ij"
        self._img_xlims = []
        self._img_ylims = []

        self._img_frame = 1
        self._img_qty = 100
        self._img_pos = None

        self._disp_shadow_line = []
        self._disp_shadow_patch = []
        self._disp_shadow_label = []
        self._disp_centre = False
        self._disp_centre_pt = []
        self._disp_limits = False
        self._disp_limits_lines = []

        # create
        self._CreatePlot()

        self._CreateCtrl()

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

        self._mainbox.Add(self._fig_canvas,  1)
        self._mainbox.Add(self._fig_toolbar, 0)

    def _CreateCtrl(self):

        # vbox = wx.StaticBox(self, -1)
        # vbox_sizer = wx.StaticBoxSizer(vbox, wx.VERTICAL)

        vbox = wx.BoxSizer(wx.VERTICAL)

        # # orientation
        # self._ctrl_ori = LabelComboBox(vbox, label="Axes:  ", style=wx.CB_SORT,
        #                                value=self._img_orient, choices=["ij", "xy"])

        # # scan slider
        # self._ctrl_scan = SpinSliderCtrl(vbox, "Frame: ", size=(60, -1),
        #                                  style=wx.SL_MIN_MAX_LABELS | wx.SL_TOP,
        #                                  min=1, max=self._img_qty, initial=1)

        # # checkboxes
        # self._ctrl_center = wx.CheckBox(vbox, wx.ID_ANY, label="Show detector center")
        # self._ctrl_lims = wx.CheckBox(vbox, wx.ID_ANY, label="Show scan limits")

        # orientation
        self._ctrl_ori = LabelComboBox(self, label="Axes:  ", style=wx.CB_SORT,
                                       value=self._img_orient, choices=["ij", "xy"])

        # scan slider
        self._ctrl_scan = SpinSliderCtrl(self, "Frame: ", size=(60, -1),
                                         style=wx.SL_MIN_MAX_LABELS | wx.SL_TOP,
                                         min=1, max=self._img_qty, initial=1)

        # checkboxes
        self._ctrl_center = wx.CheckBox(self, wx.ID_ANY, label="Show detector center")
        self._ctrl_lims = wx.CheckBox(self, wx.ID_ANY, label="Show scan limits")


        #assemble
        hbox = wx.GridSizer(rows=1, cols=3, vgap=0, hgap=5)
        hbox.Add(self._ctrl_ori, 0, wx.ALIGN_CENTER_VERTICAL)
        hbox.Add(self._ctrl_center, 0, wx.ALIGN_CENTER_VERTICAL)
        hbox.Add(self._ctrl_lims, 0, wx.ALIGN_CENTER_VERTICAL)

        # vbox_sizer.Add(hbox, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL)#, 10)
        # vbox_sizer.Add(self._ctrl_scan, 0, wx.EXPAND| wx.ALIGN_CENTER_VERTICAL | wx.ALL)#, 10)

        # vbox.Add(hbox, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        # vbox.Add(self._ctrl_scan, 0, wx.EXPAND| wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)

        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 10)
        vbox.Add(self._ctrl_scan, 0, wx.EXPAND | wx.ALL, 10)

        # self._mainbox.Add(vbox_sizer, 0, wx.EXPAND|wx.ALL, 5)
        self._mainbox.Add(vbox, 0, wx.EXPAND|wx.ALL, 5)


        #bindings
        self._ctrl_ori.Bind_to(self._OnSelectAxes)
        self._ctrl_scan.Bind_to(self._OnChangeFrame)
        self._ctrl_center.Bind(wx.EVT_CHECKBOX, self._OnToggleCenter)
        self._ctrl_lims.Bind(wx.EVT_CHECKBOX, self._OnToggleLimits)

    # Getters
    def GetScan(self):
        return self._parent.GetScan()
        #return SimScan(inp=new_simscan_dict(), verbose=False)

    # Event handlers
    def _OnSelectAxes(self, event):

        self._img_orient = event.GetValue()

        self.SetFigLimits()

        self._fig_canvas.draw()

    def _OnChangeFrame(self, event):

        self._img_frame = event.GetValue()

        self.ShowWireShadows()

    def _OnToggleCenter(self, event):

        self._disp_centre = event.GetEventObject().IsChecked()

        self.ShowBeamCentre()

    def _OnToggleLimits(self, event):

        self._disp_limits = event.GetEventObject().IsChecked()

        self.ShowScanLimits()

    # Methods to selectively update panel
    def Update(self, part="all"):

        scan = self.GetScan()

        if part in ("setup", "all"):

            self._img_xlims = [0, scan.get_img_params(['framedim'])[0] -1]
            self._img_ylims = [0, scan.get_img_params(['framedim'])[1] -1]

            self.SetFigLimits()
            self.ShowBeamCentre()


        if part in ("spec", "wire", "all"):

            self._img_qty = scan.number_images
            self._img_frame = clamp(self._img_frame, 1, self._img_qty)

            self._UpdateWidgets()

        self.ShowWireShadows()
        self.ShowScanLimits()

    def _UpdateWidgets(self):

        # update frame slide
        self._ctrl_scan.SetMaxValue(self._img_qty)
        self._ctrl_scan.SetValue(self._img_frame)

    # Display
    def SetFigLimits(self, xlims=None, ylims=None):

        if xlims is not None:
            self._img_xlims = xlims

        if ylims is not None:
            self._img_ylims = ylims

        self._fig_axes.set_xlim(self._img_xlims)

        if self._img_orient == "xy":
            self._fig_axes.set_ylim(min(self._img_ylims),
                                    max(self._img_ylims))

        else:
            self._fig_axes.set_ylim(max(self._img_ylims),
                                    min(self._img_ylims))

        if len(self._disp_shadow_label):

            for k, txt in enumerate(self._disp_shadow_label):
                txt.set_verticalalignment('top' if self._img_orient == "xy" else 'bottom')

        self._fig_axes.set_aspect("equal")

    def ShowWireShadows(self):

        scan = self.GetScan()

        xmin, xmax = 0, scan.get_img_params(['framedim'])[0]
        ys_right = scan.calc_wires_range_shadow(self._img_frame - 1, xcam=xmax)
        ys_left = scan.calc_wires_range_shadow(self._img_frame - 1, xcam=xmin)

        if self._img_orient == "xy":
            align = 'top'
        else:
            align = 'bottom'

        print('self._fig_axes.lines', self._fig_axes.lines)
        if len(self._disp_shadow_line) != len(ys_right):
            for lines, patches in zip(self._disp_shadow_line, self._disp_shadow_patch):
                print('lines',lines)
                #for line in lines:
                self._fig_axes.lines.remove(lines)
                #self._fig_axes.lines.remove(lines)
                
                
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
                               closed=True, antialiased=True, alpha=0.5, facecolor=colors[i])

                txt =  self._fig_axes.text(xmax-10, ymin[1], str(i+1), fontsize='small',
                                           horizontalalignment='right', verticalalignment=align)

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


        self._fig_canvas.draw()

    def ShowBeamCentre(self):

        if self._disp_centre:

            xcen, ycen = self.GetScan().get_ccd_params(['xcen', 'ycen'])

            if not self._disp_centre_pt:

                pt1 = self._fig_axes.plot([xcen], [ycen],
                                     marker='x', markeredgecolor='k',
                                     markeredgewidth=1.1, markersize=10)
                pt2 = self._fig_axes.plot([xcen], [ycen], fillstyle='none',
                                     marker='s', markeredgecolor='k',
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

        self._fig_canvas.draw()

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

        self._fig_canvas.draw()

if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")
    box = wx.BoxSizer(wx.HORIZONTAL)
    tmp = PanelExperimentFigure(frame)
    tmp.Update()
    box.Add(tmp, 1, wx.EXPAND)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
