#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

from LaueTools.Daxm.gui.widgets.spin import LabelSpinCtrl

import LaueTools.Daxm.gui.icons.icon_manager as mycons

MON_NONE = -1
MON_DTT = 0
MON_SPEC = 1

EVT_MON_MODIFY = "event_modify"
EVT_MON_PLOT = "event_button_plot"
EVT_MON_CHECK = "event_button_check"


class InputMonitor(wx.StaticBoxSizer):

    # Constructors
    def __init__(self, parent):
        mainbox = wx.StaticBox(parent, wx.ID_ANY, "Monitor")

        wx.StaticBoxSizer.__init__(self, mainbox, wx.VERTICAL)

        mainbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        mainbox.SetForegroundColour('DIM GREY')

        self._box = mainbox
        self._parent = parent
        self._observers = {EVT_MON_MODIFY: [],
                           EVT_MON_PLOT: [],
                           EVT_MON_CHECK: []}

        self.framedimx = 2048
        self.framedimy = 2048

        self.notebook = None
        self.apply_ckb = None
        self.dttoff_spn = None
        self.mon4off_spn = None
        self.xcen_spn = None
        self.ycen_spn = None
        self.xwidth_spn = None
        self.ywidth_spn = None
        self.plot_btn = None
        self.check_btn = None

        self.Create()
        self.Init()

    def Create(self):

        self.apply_ckb = wx.CheckBox(self._parent, wx.ID_ANY, label=" apply correction")

        self.dttoff_spn = LabelSpinCtrl(self._parent, label="detector offset: ",
                                        initial=1000, min=0, max=60000, size=(50, -1))

        # notebook
        self.notebook = wx.Notebook(self._parent)

        # monitor from dtt
        panel1 = wx.Panel(self.notebook)
        txt1 = wx.StaticText(panel1, label="Centre of monitored region: ")
        txt2 = wx.StaticText(panel1, label="Half widths: ")

        self.xcen_spn = LabelSpinCtrl(panel1, "X: ", initial=self.framedimx / 2,
                                      min=0, max=self.framedimx - 1, size=(55, -1))

        self.ycen_spn = LabelSpinCtrl(panel1, "Y: ", initial=self.framedimy / 2,
                                      min=0, max=self.framedimy - 1, size=(55, -1))

        self.xwidth_spn = LabelSpinCtrl(panel1, "Sx: ", initial=0,
                                        min=-1, max=1000, size=(50, -1))

        self.ywidth_spn = LabelSpinCtrl(panel1, "Sy: ", initial=0,
                                        min=-1, max=1000, size=(50, -1))

        grid1 = wx.GridSizer(rows=4, cols=2, hgap=5, vgap=5)

        grid1.Add(txt1, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.LEFT, 10)
        grid1.Add(wx.BoxSizer())
        grid1.Add(self.xcen_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER)
        grid1.Add(self.ycen_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER)
        grid1.Add(txt2, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        grid1.Add(wx.BoxSizer())
        grid1.Add(self.xwidth_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER | wx.BOTTOM, 10)
        grid1.Add(self.ywidth_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER | wx.BOTTOM, 10)

        panel1.SetSizer(grid1)
        grid1.Layout()
        panel1.Fit()

        self.notebook.AddPage(panel1, "From detector", True)

        # monitor from spec
        panel2 = wx.Panel(self.notebook)

        self.mon4off_spn = LabelSpinCtrl(panel2, "Mon4 offset /sec: ", initial=10000,
                                         min=0, max=1000000, size=(80, -1))

        grid2 = wx.GridSizer(rows=1, cols=2, hgap=5, vgap=5)

        grid2.Add(self.mon4off_spn, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 10)

        panel2.SetSizer(grid2)
        grid2.Layout()
        panel2.Fit()

        self.notebook.AddPage(panel2, "From spec", False)

        # buttons
        hbox = wx.GridSizer(rows=1, cols=2, hgap=10, vgap=0)

        self.plot_btn = wx.Button(self._parent, id=wx.ID_ANY, label=" Plot")
        self.plot_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        self.check_btn = wx.Button(self._parent, id=wx.ID_ANY, label=" Check correction ")
        self.check_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        hbox.Add(self.plot_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        hbox.Add(self.check_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)

        # assemble
        # self.Add(self.dttoff_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.LEFT, 10)
        # self.Add(self.apply_ckb, 0,  wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        # self.Add(self.notebook, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        # self.Add(hbox, 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 5)

        self.Add(self.dttoff_spn, 0,  wx.TOP | wx.LEFT, 10)
        self.Add(self.apply_ckb, 0,  wx.ALL, 10)
        self.Add(self.notebook, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        self.Add(hbox, 0, wx.EXPAND | wx.ALL, 5)

        # bindings
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnModifyAnything)
        self.apply_ckb.Bind(wx.EVT_CHECKBOX, self.OnApplyCorrection)
        self.dttoff_spn.Bind_to(self.OnModifyAnything)
        self.mon4off_spn.Bind_to(self.OnModifyAnything)
        self.xcen_spn.Bind_to(self.OnModifyRegion)
        self.ycen_spn.Bind_to(self.OnModifyRegion)
        self.xwidth_spn.Bind_to(self.OnModifyAnything)
        self.ywidth_spn.Bind_to(self.OnModifyAnything)
        self.plot_btn.Bind(wx.EVT_BUTTON, self.OnButtonPlotCorrection)
        self.check_btn.Bind(wx.EVT_BUTTON, self.OnButtonCheckCorrection)

    def Init(self):
        self.SetMode(MON_DTT)

    # Getters
    def GetValue(self):

        mode = self.GetModeString()
        roi = []
        monoff = self.mon4off_spn.GetValue()
        ccdoff = self.dttoff_spn.GetValue()

        if self.GetMode() == MON_DTT:
            roi = self.GetRegion()

        return mode, roi, monoff, ccdoff

    def GetMode(self):

        mode = MON_NONE
        if self.apply_ckb.GetValue():
            mode = self.notebook.GetSelection()

        return mode

    def GetModeString(self):

        mode_str = ["detector", "spec", None]

        return mode_str[self.GetMode()]

    def GetRegion(self):

        xcen = self.xcen_spn.GetValue()
        ycen = self.ycen_spn.GetValue()
        sx = self.xwidth_spn.GetValue()
        sy = self.ywidth_spn.GetValue()

        return xcen, ycen, sx, sy

    # Setters
    def SetValue(self, mode, roi, monoff, ccdoff):
        self.SetModeFromString(mode)
        self.SetRegion(*roi)
        self.SetOffsetMonitor4(monoff)
        self.SetOffsetDetector(ccdoff)

    def SetMode(self, mode=None):

        if mode is None:
            mode = self.GetMode()

        if mode == MON_NONE:

            self.apply_ckb.SetValue(0)
            self.notebook.Enable(0)

        else:
            self.apply_ckb.SetValue(1)
            self.notebook.Enable(1)
            self.notebook.ChangeSelection(mode)

    def SetModeFromString(self, mode_str):

        mode = {"detector": MON_DTT,
                "spec": MON_SPEC}

        if str in mode:
            self.SetMode(mode[mode_str])
        else:
            self.SetMode(MON_NONE)

    def SetRegion(self, x=None, y=None, sx=None, sy=None):

        x0, y0, sx0, sy0 = self.GetRegion()

        if x or y:
            x = x or x0
            y = y or y0
            self.SetRegionCentre(x=x, y=y)

        if sx or sy:
            sx = sx or sx0
            sy = sy or sy0
            self.SetRegionSize(sx=sx, sy=sy)

    def SetRegionCentre(self, x, y):

        self.xcen_spn.SetValue(x)
        self.ycen_spn.SetValue(y)
        self.UpdateRegionLimits()

    def UpdateRegionLimits(self):
        xcen, ycen, _, _ = self.GetRegion()
        sx = min(self.framedimx - xcen, xcen)
        sy = min(self.framedimy - ycen, ycen)
        self.xwidth_spn.SetMax(sx - 1)
        self.ywidth_spn.SetMax(sy - 1)

    def SetRegionSize(self, sx, sy):
        self.xwidth_spn.SetValue(sx)
        self.ywidth_spn.SetValue(sy)

    def SetOffsetMonitor4(self, value):

        self.mon4off_spn.SetValue(value)

    def SetOffsetDetector(self, value):

        self.dttoff_spn.SetValue(value)

    def SetFrameDimension(self, Dx=None, Dy=None):

        if Dx is not None:
            self.framedimx = Dx

            self.xcen_spn.SetMax(Dx - 1)

        if Dy is not None:
            self.framedimy = Dy

            self.ycen_spn.SetMax(Dy - 1)

    # Event handlers
    def OnModifyAnything(self, event):
        self.Notify(EVT_MON_MODIFY)

    def OnModifyRegion(self, event):
        self.UpdateRegionLimits()
        self.OnModifyAnything(event)

    def OnApplyCorrection(self, event):
        self.SetMode()
        self.OnModifyAnything(event)

    def OnButtonPlotCorrection(self, event):
        self.Notify(EVT_MON_PLOT)

    def OnButtonCheckCorrection(self, event):
        self.Notify(EVT_MON_CHECK)

    # Binders
    def Bind_to(self, event, callback):

        self._observers[event].append(callback)

    def Notify(self, event):

        for callback in self._observers[event]:
            callback(self)


# Test
if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = InputMonitor(frame)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()