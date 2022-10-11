#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

import wx.lib.scrolledpanel as wxls

from LaueTools.Daxm.gui.boxes.input_detector import InputDetector
from LaueTools.Daxm.gui.boxes.input_spec import InputSpec
from LaueTools.Daxm.gui.boxes.input_wire import InputWire
from LaueTools.Daxm.gui.boxes.input_sample import InputSample
from LaueTools.Daxm.gui.widgets.toolbar_io import OpenSaveBar, EVT_SAVE, EVT_OPEN

class PanelExperimentControls(wxls.ScrolledPanel):
    # constructors
    def __init__(self, parent):
        wxls.ScrolledPanel.__init__(self, parent, style=wx.BORDER_RAISED)

        self._parent = parent
        print(('self._parent  of controls.PanelExperimentControls', self._parent))
        self._mainbox = None
        self._observers = []

        self.tb_file = None
        self.inp_dtt = None
        self.inp_spc = None
        self.inp_wire = None
        self.inp_sam = None

        self._Create()
        self._Init()

    def _Create(self):
        # boxes
        self.tb_file = OpenSaveBar(self)

        self.inp_dtt = InputDetector(self)

        self.inp_spc = InputSpec(self)

        self.inp_wire= InputWire(self)

        self.inp_sam = InputSample(self)

        # sizer
        self._mainbox = wx.BoxSizer(wx.VERTICAL)

        box = wx.FlexGridSizer(rows=2, cols=2, hgap=7, vgap=7)
        box.SetFlexibleDirection(wx.VERTICAL)

        box.AddMany([(self.inp_dtt, 0, wx.EXPAND),
                               (self.inp_spc, 0, wx.EXPAND),
                               (self.inp_wire, 0, wx.EXPAND),
                               (self.inp_sam, 0, wx.EXPAND)])

        box.AddGrowableCol(0)
        box.AddGrowableCol(1)

        self._mainbox.Add(self.tb_file, 0, wx.EXPAND)
        self._mainbox.Add(box, 0, wx.EXPAND|wx.TOP, 7)

        # vbox1 = wx.BoxSizer(wx.VERTICAL)
        # vbox1.Add(, 0, wx.EXPAND)
        # vbox1.Add(self.inp_wire, 0, wx.EXPAND|wx.TOP, 7)
        #
        # vbox2 = wx.BoxSizer(wx.VERTICAL)
        # vbox2.Add(self.inp_spc, 1, wx.EXPAND)
        # vbox2.Add(self.inp_sam, 0, wx.EXPAND|wx.TOP, 7)

        # self._mainbox.Add(vbox1, 0, wx.EXPAND)
        # self._mainbox.Add(vbox2, 0, wx.EXPAND)

        # bindings
        self.inp_dtt.Bind_to(self.OnModifyDetector)
        self.inp_spc.Bind_to(self.OnModifySpec)
        self.inp_wire.Bind_to(self.OnModifyWire)
        self.tb_file.Bind_to(EVT_OPEN, self._parent.OnOpen)
        self.tb_file.Bind_to(EVT_SAVE, self._parent.OnSave)

        # start scrolling
        self.SetupScrolling()

        # Layout
        self.SetSizer(self._mainbox)
        self.Layout()
        self._mainbox.Layout()
        self._mainbox.Fit(self)
        self.Fit()

    def _Init(self):
        self.Update()
        self.tb_file.SetValue("")

    # Getters
    def GetScan(self):
        return self._parent.GetScan()

    # Setters
    def Update(self, part="all"):

        scan = self.GetScan()

        if part in ("setup", "all"):

            self.SetDetector(scan)

        if part in ("spec", "all"):

            self.SetSpec(scan)

        if part in ("wire", "all"):

            self.SetWire(scan)

    def SetDetector(self, scan):
        scan_dict = scan.to_dict("setup")
        self.inp_dtt.SetValue(label=scan_dict['CCDType'], params=scan_dict['detCalib'])

    def SetSpec(self, scan):
        scan_dict = scan.to_dict("spec")
        self.inp_spc.SetValue(fname=scan_dict['specFile'], scan=scan_dict['scanNumber'], comm=scan_dict['scanCmd'])

    def SetWire(self, scan):
        scan_dict = scan.to_dict("wire")
        self.inp_wire.SetValue(wires=scan_dict['wire'], traj=scan_dict['wireTrajAngle'])

    # Event handlers
    def OnModifyDetector(self, event):
        ccdtype, params = event.GetValue()
        myevent = ControlEvent("setup", {'CCDType':ccdtype, 'detCalib':params})
        self.Notify(myevent)

    def OnModifyWire(self, event):
        wire, traj = event.GetValue()
        myevent = ControlEvent("wire", {'wire':wire, 'wireTrajAngle':traj})
        self.Notify(myevent)

    def OnModifySpec(self, event):
        fname, scan, comm = event.GetValue()
        myevent = ControlEvent("spec", {'specFile':fname, 'scanNumber':scan, 'scanCmd':comm})
        self.Notify(myevent)

    # Binders
    def Bind_to(self, callback):
        self._observers.append(callback)

    def Notify(self, event):
        # print self.GetValue()
        for callback in self._observers:
            callback(event)


# Event class
class ControlEvent(wx.CommandEvent):

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

# End


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")
    box = wx.BoxSizer(wx.HORIZONTAL)
    tmp = PanelExperimentControls(frame)
    box.Add(tmp, 1, wx.EXPAND)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()