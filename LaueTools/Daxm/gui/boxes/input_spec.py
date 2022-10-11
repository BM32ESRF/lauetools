#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

import LaueTools.Daxm.contrib.spec_reader as spr

from LaueTools.Daxm.gui.widgets.text import LabelTxtCtrlNum
from LaueTools.Daxm.gui.widgets.file import FileSelectOpen
from LaueTools.Daxm.gui.widgets.combobox import LabelComboBox
from LaueTools.Daxm.gui.widgets.spin import LabelSpinCtrl

class InputSpec(wx.StaticBoxSizer):
    """ Gui class to open spec file or bliss hdf  (used in Load DLaue Scan)"""
    # Constructors
    def __init__(self, parent, label="Spec"):
        mainbox = wx.StaticBox(parent, wx.ID_ANY, label)

        wx.StaticBoxSizer.__init__(self, mainbox, wx.VERTICAL)

        mainbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        mainbox.SetForegroundColour('DIM GREY')

        self._box = mainbox
        self._parent = parent
        self._observers = []

        self.spec = None

        self.spec_fs = None
        self.cmd_cbx = None
        self.manual_chk = None
        self.ystart_txt = None
        self.ystop_txt = None
        self.step_txt = None
        self.expo_txt = None

        self.Create()
        self.Init()

    def Create(self):

        self.spec_fs = FileSelectOpen(self._parent, "Spec file:", "", tip='Load spec or bliss hdf5 file')

        self.cmd_cbx = LabelComboBox(self._parent, "Scan:  ", tip='list of wire scans from spec or hdf file',
                                    choices=["xxxx ascan yf xxxx xxxx xxxx xxxx"])

        self.manual_chk = wx.CheckBox(self._parent, wx.ID_ANY, "user-defined:")

        self.ystart_txt = LabelTxtCtrlNum(self._parent, "", 0, size=(40, -1))
        self.ystop_txt = LabelTxtCtrlNum(self._parent, "", 0, size=(40, -1))
        self.step_txt = LabelSpinCtrl(self._parent, "", min=1, max=10000, initial=1, size=(60, -1))
        self.expo_txt = LabelTxtCtrlNum(self._parent, "", 0, size=(40, -1))

        grid = wx.GridSizer(rows=1, cols=5, hgap=5, vgap=0)
        grid.Add(wx.StaticText(parent=self._parent, id=wx.ID_ANY, label="ascan yf"),
                 0, wx.ALIGN_CENTER)
        grid.Add(self.ystart_txt, 0, wx.EXPAND | wx.ALIGN_CENTER)
        grid.Add(self.ystop_txt, 0, wx.EXPAND | wx.ALIGN_CENTER)
        grid.Add(self.step_txt, 0, wx.EXPAND | wx.ALIGN_CENTER)
        grid.Add(self.expo_txt, 0, wx.EXPAND | wx.ALIGN_CENTER)

        self.AddSpacer(10)
        self.Add(self.spec_fs,  0, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(10)
        self.Add(self.cmd_cbx, 0, wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(20)
        self.Add(self.manual_chk, 0, wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(10)
        self.Add(grid, 0,  wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(15)

        # bindings
        self.manual_chk.Bind(wx.EVT_CHECKBOX, self.OnToggleManual)
        self.spec_fs.Bind_to(self.OnSelectFile)
        self.cmd_cbx.Bind_to(self.OnSelectScan)
        self.ystart_txt.Bind_to(self.OnModifyParam)
        self.ystop_txt.Bind_to(self.OnModifyParam)
        self.step_txt.Bind_to(self.OnModifyParam)
        self.expo_txt.Bind_to(self.OnModifyParam)

    def Init(self):

        self.SetManual()

    # Getters
    def GetValue(self):

        if self.manual_chk.IsChecked() or self.spec_fs.IsBlank():

            fname = None
            scan = 0
            comm = [self.ystart_txt.GetValue(), self.ystop_txt.GetValue(),
                    self.step_txt.GetValue(), self.expo_txt.GetValue()]

        else:
            fname = self.spec_fs.GetValue()
            scan = int(self.cmd_cbx.GetValue().split()[0])
            comm = []

        return fname, scan, comm

    def GetSpecList(self):

        if self.spec is not None:
            scan_list = [item for item in self.spec.cmd_list.values() if (('yf' in item) | ('zf' in item))]

        else:
            scan_list = []

        return scan_list

    def GetSpecScan(self, scan_num):

        if scan_num in self.spec.cmd_list:
            cmd = self.spec.cmd_list[scan_num]
        else:
            cmd = None

        return cmd

    def GetSpecScanParams(self, scan_num=None):

        if scan_num is None:
            cmd = self.cmd_cbx.GetValue()
        else:
            cmd = self.GetSpecScan(scan_num)

        cmd_parts = cmd.split()

        if cmd_parts[1] in ("ascan", "dscan"):
            params = cmd_parts[3:]
        else: # should be mesh or dmesh...
            params = cmd_parts[7:]

        params = [float(param) for param in params]
        params[2] = int(params[2])

        return params

    def IsManual(self):
        return self.manual_chk.IsChecked()

    # Setters
    def SetValue(self, fname, scan, comm):

        if fname is not None:

            self.SetManual(False)

            self.SetFile(fname)

            self.SetScan(scan)

            self.SetParams()

        else:
            self.SetManual(True)

            self.SetParams(comm)

    def SetFile(self, fname):

        self.spec_fs.SetValue(fname)

        self.spec = spr.SpecFile(fname)

        self.SetScanList()

    def SetManual(self, value=True):
        value = bool(value)

        self.manual_chk.SetValue(value)

        self.spec_fs.Enable(not value)
        self.cmd_cbx.Enable(not value)

        self.ystart_txt.SetEditable(value)
        self.ystop_txt.SetEditable(value)
        self.step_txt.Enable(value)
        self.expo_txt.SetEditable(value)

    def SetParams(self, params=None):

        if params is None and not self.spec_fs.IsBlank():
            params = self.GetSpecScanParams()

        if params is not None:
            ystart, ystop, step, expo = params

            self.ystart_txt.SetValue(ystart)
            self.ystop_txt.SetValue(ystop)
            self.step_txt.SetValue(int(step))
            self.expo_txt.SetValue(expo)

    def SetScan(self, scan_num):

        cmd = self.GetSpecScan(scan_num)

        self.cmd_cbx.SetValue(cmd)

    def SetScanList(self, choices=None):

        if choices is None:
            choices = self.GetSpecList()

        self.cmd_cbx.SetChoices(choices)

    # Event handlers
    def OnSelectFile(self, event):
        """ start the spec or hdf5 file reader """
        self.spec = spr.SpecFile(event.GetValue())

        self.SetScanList()

        self.SetParams()

        self.Notify()

    def OnSelectScan(self, event):

        self.SetParams()

        self.Notify()

    def OnModifyParam(self, event):

        self.Notify()

    def OnToggleManual(self, event):

        self.SetManual(event.IsChecked())

        self.SetParams()

        self.Notify()

    # Binders
    def Bind_to(self, callback):

        self._observers.append(callback)

    def Notify(self):

        for callback in self._observers:
            callback(self)

    # Appearance
    def Enable(self, enable=True):

        if enable:
            self.manual_chk.Enable(True)
            self.SetManual(self.IsManual())
        else:
            self.manual_chk.Enable(False)
            self.spec_fs.Enable(False)
            self.cmd_cbx.Enable(False)
            self.ystart_txt.SetEditable(False)
            self.ystop_txt.SetEditable(False)
            self.step_txt.Enable(False)
            self.expo_txt.SetEditable(False)


class InputSpec2(InputSpec):

    # Constructor
    def __init__(self, parent, label="Spec"):
        InputSpec.__init__(self, parent, label=label)

        # Create
        self.img_fs = FileSelectOpen(self._parent, "First image:", "", tip="select first image of wire daxm scan")

        self.AddSpacer(10)
        self.Add(self.img_fs,  0, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        self.AddSpacer(10)

        self.img_fs.Bind_to(self.OnModifyParam)

        # Init
        self.SetManual(False)

    # Setters
    def SetValue(self, fname, scan, comm, img):

        InputSpec.SetValue(self, fname, scan, comm)

        self.img_fs.SetValue(img)

    def SetWildCard(self, wildcard):
        self.img_fs.SetWildCard(wildcard)

    def SetDefaultDir(self, default_dir):

        self.spec_fs.SetDefaultDir(default_dir)
        self.img_fs.SetDefaultDir(default_dir)

    # Getters
    def GetValue(self):
        return InputSpec.GetValue(self) + (self.img_fs.GetValue(),)

    # Appearance
    def Enable(self, enable=True):

        self.img_fs.Enable(enable)

        InputSpec.Enable(self, enable)


# End of class


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = InputSpec2(frame, "Scan information:")

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
