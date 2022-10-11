#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

import LaueTools.dict_LaueTools as dlt
import LaueTools.IOLaueTools as rwa

from LaueTools.Daxm.gui.widgets.text import LabelTxtCtrlNum
from LaueTools.Daxm.gui.widgets.file import FileSelectOpen
from LaueTools.Daxm.gui.widgets.combobox import LabelComboBox


class InputDetector(wx.StaticBoxSizer):

    # Constructors
    def __init__(self, parent, label="Detector"):
        mainbox = wx.StaticBox(parent, wx.ID_ANY, label)

        wx.StaticBoxSizer.__init__(self, mainbox, wx.VERTICAL)

        mainbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        mainbox.SetForegroundColour('DIM GREY')

        self._box = mainbox
        self._parent = parent
        self._observers = []

        self.ccd_cbx = None
        self.geo_fs = None
        self.manual_chk = None
        self.xcen_txt = None
        self.ycen_txt = None
        self.dist_txt = None
        self.xgam_txt = None
        self.xbet_txt = None

        self.Create()
        self.Init()

    def Create(self):

        # widgets
        self.ccd_cbx = LabelComboBox(self._parent, label="Name:  ", value="sCMOS", choices=dlt.dict_CCD.keys())

        self.geo_fs = FileSelectOpen(self._parent, "Parameter file:", "",
                                    "LaueTools detector file (*.det)|*.det",
                                    tip='Calibration file *.det')

        self.manual_chk = wx.CheckBox(self._parent, wx.ID_ANY, "user-defined:")

        self.xcen_txt = LabelTxtCtrlNum(self._parent, "Xcen: ", 0)      # , style=wx.TE_READONLY)
        self.ycen_txt = LabelTxtCtrlNum(self._parent, "Ycen: ", 0)      # , style=wx.TE_READONLY)
        self.dist_txt = LabelTxtCtrlNum(self._parent, "D: ", 0)         # , style=wx.TE_READONLY)
        self.xbet_txt = LabelTxtCtrlNum(self._parent, u"x\u03B2: ", 0)  # , style=wx.TE_READONLY)
        self.xgam_txt = LabelTxtCtrlNum(self._parent, u"x\u03B3: ", 0)  # , style=wx.TE_READONLY)

        grid1 = wx.GridSizer(rows=1, cols=2, hgap=15, vgap=0)
        grid1.Add(self.xcen_txt, 0, wx.EXPAND)
        grid1.Add(self.ycen_txt, 0, wx.EXPAND)

        grid2 = wx.GridSizer(rows=1, cols=3, hgap=15, vgap=0)
        grid2.Add(self.dist_txt, 0, wx.EXPAND)
        grid2.Add(self.xbet_txt, 0, wx.EXPAND)
        grid2.Add(self.xgam_txt, 0, wx.EXPAND)

        # assemble
        self.AddSpacer(10)
        self.Add(self.ccd_cbx, 0, wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(10)
        self.Add(self.geo_fs, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(20)
        self.Add(self.manual_chk, 0, wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(10)
        self.Add(grid1, 0, wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(10)
        self.Add(grid2, 0, wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(15)

        # bindings
        self.manual_chk.Bind(wx.EVT_CHECKBOX, self.OnToggleManual)
        self.geo_fs.Bind_to(self.OnSelectFile)
        self.xcen_txt.Bind_to(self.OnModifyParam)
        self.ycen_txt.Bind_to(self.OnModifyParam)
        self.dist_txt.Bind_to(self.OnModifyParam)
        self.xbet_txt.Bind_to(self.OnModifyParam)
        self.xgam_txt.Bind_to(self.OnModifyParam)

    def Init(self):

        self.SetManual(True)

    # Getters
    def GetValue(self):

        label = self.ccd_cbx.GetValue()

        if self.IsManual() or self.geo_fs.IsBlank():

            params = [self.dist_txt.GetValue(),
                      self.xcen_txt.GetValue(),
                      self.ycen_txt.GetValue(),
                      self.xbet_txt.GetValue(),
                      self.xgam_txt.GetValue(),
                      dlt.dict_CCD[self.ccd_cbx.GetValue()][1]]

        else:
            params = self.geo_fs.GetValue()

        return label, params

    def IsManual(self):
        return self.manual_chk.IsChecked()

    # Setters
    def SetValue(self, label, params):

        self.ccd_cbx.SetValue(label)

        if isinstance(params, str):

            self.SetManual(False)

            self.geo_fs.SetValue(params)

            self.SetParams()

        else:

            self.SetManual(True)

            self.SetParams(params)

    def SetManual(self, value):

        value = bool(value)

        self.manual_chk.SetValue(value)

        self.geo_fs.Enable(not value)

        self.xcen_txt.SetEditable(value)
        self.ycen_txt.SetEditable(value)
        self.dist_txt.SetEditable(value)
        self.xgam_txt.SetEditable(value)
        self.xbet_txt.SetEditable(value)

    def SetParams(self, params=None):

        if params is None and not self.geo_fs.IsBlank():
            params, _ = rwa.readfile_det(self.geo_fs.GetValue(), nbCCDparameters=6, verbose=False)

        if params is not None:
            distance, xcen, ycen, xbet, xgam = params[:5]

            self.xcen_txt.SetValue(xcen)
            self.ycen_txt.SetValue(ycen)
            self.dist_txt.SetValue(distance)
            self.xgam_txt.SetValue(xbet)
            self.xbet_txt.SetValue(xgam)

    def SetDefaultDir(self, default_dir):

        self.geo_fs.SetDefaultDir(default_dir)

    # Event handlers
    def OnToggleManual(self, event):

        self.SetManual(event.IsChecked())

        self.SetParams()

        self.Notify()

    def OnSelectFile(self, event):

        self.SetParams()

        self.Notify()

    def OnModifyParam(self, event):
        self.Notify()

    # Binders
    def Bind_to(self, callback):

        self._observers.append(callback)

    def Notify(self):

        for callback in self._observers:
            callback(self)

    # Appearance
    def Enable(self, enable=True):

        self.ccd_cbx.Enable(enable)
        self.manual_chk.Enable(enable)

        if enable:
            self.SetManual(self.IsManual())
        else:
            self.geo_fs.Enable(False)
            self.xcen_txt.SetEditable(False)
            self.ycen_txt.SetEditable(False)
            self.dist_txt.SetEditable(False)
            self.xgam_txt.SetEditable(False)
            self.xbet_txt.SetEditable(False)

# End of class


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = InputDetector(frame)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
