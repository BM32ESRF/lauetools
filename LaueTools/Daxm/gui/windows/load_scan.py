#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os
__path__ = os.path.dirname(__file__)

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

import LaueTools.readmccd as rmccd

from LaueTools.Daxm.gui.icons import icon_manager as mycons
from LaueTools.Daxm.gui.widgets.file import FileSelectOpen
from LaueTools.Daxm.gui.boxes.input_detector import InputDetector
from LaueTools.Daxm.gui.boxes.input_spec import InputSpec2

from LaueTools.Daxm.utils.read_image import split_filename

from LaueTools.Daxm.classes.scan.point import new_scan_dict, load_scan_dict


SET_FROMFILE = False
SET_MANUAL = True


class LoadScan(wx.Dialog):

    def __init__(self, parent, workdir=None):
        wx.Dialog.__init__(self, parent, wx.ID_ANY, 
                           "Load 3DLaue Scan",
                           style=wx.RESIZE_BORDER)

        self._parent = parent
        self._mainbox = None
        self._workdir = ""
        self._mode = SET_FROMFILE

        self.wgt_file = None
        self.ckb_manual = None
        self.inp_dtt = None
        self.inp_spc = None
        self.btn_ok = None
        self.bnt_cl = None

        self.Create()
        self.Init(workdir)

    def Create(self):

        self.SetIcon(mycons.get_icon("logo_IF.png"))

        DEFAULTFILE='/home/micha/LaueProjects/DAXM/BN16_JBMolin_IH_MA_53/calib0_calib_ok.scan'
        # widgets
        self.wgt_file = FileSelectOpen(self, "From file: ", "",
                                       wildcard="(*.scan)|*.scan",
                                       tip='Load already built DAXM summary file .scan\n%s'%DEFAULTFILE)

        self.ckb_manual = wx.CheckBox(self, wx.ID_ANY, label="Manually:")

        self.inp_dtt = InputDetector(self, label="Detector information")

        self.inp_spc = InputSpec2(self, label="Scan information")

        self.btn_ok = wx.Button(self, wx.ID_OK)
        self.btn_cl = wx.Button(self, wx.ID_CANCEL)

        # assemble
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.btn_ok, 0, wx.ALL, 5)
        hbox.Add(self.btn_cl, 0, wx.ALL, 5)
        vbox.Add(hbox, 0, wx.EXPAND)

        self._mainbox = wx.BoxSizer(wx.VERTICAL)
        self._mainbox.Add(wx.StaticLine(self, wx.ID_ANY, style=wx.LI_HORIZONTAL), 0, wx.EXPAND | wx.ALL, 10)
        self._mainbox.Add(self.wgt_file, 0, wx.EXPAND | wx.ALL, 5)
        self._mainbox.Add(self.ckb_manual, 0, wx.EXPAND | wx.ALL, 5)
        self._mainbox.Add(self.inp_dtt, 0, wx.EXPAND | wx.ALL, 5)
        self._mainbox.Add(self.inp_spc, 0, wx.EXPAND | wx.ALL, 5)
        self._mainbox.Add(vbox, 1, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)


        # bindings
        self.ckb_manual.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)
        self.wgt_file.Bind_to(self.OnOpenInputFile)

    def Init(self, workdir=None):

        # values
        self._workdir = os.getcwd() if workdir is None else workdir

        self.inp_spc.SetManual(False)
        self.inp_dtt.SetManual(False)

        self.SetCCDType()
        self.SetMode() # at last !

        # Layout
        self.SetSizer(self._mainbox)
        self.Layout()
        self._mainbox.Layout()
        self._mainbox.Fit(self)
        self.Fit()

    # Setters
    def SetMode(self, mode=None):
        if mode is None:
            mode = self.GetMode()

        self.wgt_file.Enable(not mode)
        self.inp_spc.Enable(mode)
        self.inp_dtt.Enable(mode)

    def SetWorkDir(self, workdir=None):

        if workdir is not None:
            self.inp_spc.SetDefaultDir(workdir)
            self.inp_dtt.SetDefaultDir(workdir)

    def SetCCDType(self, ccd_type=None):

        if ccd_type is None:
            ccd_type = self.inp_dtt.GetValue()[0]

        wildcard = rmccd.getwildcardstring(ccd_type)

        self.inp_spc.SetWildCard(wildcard)

    # Getters
    def GetScan(self):

        if self.GetMode() == SET_FROMFILE:

            scan = load_scan_dict(self.wgt_file.GetValue())

        else:

            ccd_label, det_calib = self.inp_dtt.GetValue()

            spec_file, scan, comm, img_file = self.inp_spc.GetValue()

            filedir, filename = os.path.split(img_file)
            pref, idx, _ = split_filename(filename)

            # filedir, fname = os.path.split(img_file)
            # pref, remaining = fname.split('_')
            # pref += '_'
            # idx = remaining.split('.')[0]

            new_dict = {'specFile': spec_file,
                        'scanNumber': scan,
                        'scanCmd': comm,
                        'CCDType': ccd_label,
                        'detCalib': det_calib,
                        'imageFolder': filedir,
                        'imagePrefix': pref,
                        'imageFirstIndex': int(idx),
                        'imageDigits': len(idx)
                        }

            scan = new_scan_dict(new_dict)

        return scan

    def GetMode(self):
        return SET_MANUAL if self.ckb_manual.IsChecked() else SET_FROMFILE

    def GetFilename(self):

        if self.GetMode() == SET_FROMFILE:

            fn =  self.wgt_file.GetValue()

        else:

            scan = self.GetScan()

            fn = os.path.join(scan['imageFolder'], scan['imagePrefix']+"LT3D.scan")

        return fn

    # Event Handlers
    def OnOpenInputFile(self, event):
        self.SetWorkDir(self.wgt_file.GetDefaultDir())

    def OnCheckBox(self, event):
        self.SetMode()


# test$
if __name__ == "__main__":

    app = wx.App(False)
    frame = LoadScan(parent=None)

    frame.Show(True)

    res = frame.ShowModal() == wx.ID_OK

    if res:
        print(frame.GetScan())

    frame.Destroy()
