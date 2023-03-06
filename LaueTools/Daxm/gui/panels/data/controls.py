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
from LaueTools.Daxm.gui.boxes.input_monitor import InputMonitor, EVT_MON_MODIFY, EVT_MON_PLOT, EVT_MON_CHECK
from LaueTools.Daxm.gui.boxes.input_data import InputData, EVT_DATA_MODIFY, EVT_DATA_MOVE
from LaueTools.Daxm.gui.widgets.toolbar_io import OpenSaveBar, EVT_SAVE, EVT_OPEN


class PanelDataControls(wxls.ScrolledPanel):
    # constructors
    def __init__(self, parent):
        wxls.ScrolledPanel.__init__(self, parent, style=wx.BORDER_RAISED)

        self._parent = parent
        self._mainbox = None
        self._observers = []

        self.inp_dtt = None
        self.inp_spc = None
        self.inp_wire = None
        self.inp_mon = None
        self.inp_dat = None

        self._Create()
        self._Init()

    def _Create(self):
        # boxes
        self.tb_file  = OpenSaveBar(self)

        self.inp_dtt = InputDetector(self)

        self.inp_spc = InputSpec(self)

        self.inp_wire= InputWire(self)

        self.inp_mon = InputMonitor(self)

        self.inp_dat = InputData(self)

        # sizer
        self._mainbox = wx.BoxSizer(wx.VERTICAL)

        vbox1 = wx.BoxSizer(wx.VERTICAL)
        vbox1.Add(self.inp_dtt, 0, wx.EXPAND)
        vbox1.Add(self.inp_wire, 0, wx.EXPAND|wx.TOP, 7)
        vbox1.Add(self.inp_spc, 0, wx.EXPAND|wx.TOP, 7)

        vbox2 = wx.BoxSizer(wx.VERTICAL)
        vbox2.Add(self.inp_dat, 0, wx.EXPAND)
        vbox2.Add(self.inp_mon, 1, wx.EXPAND|wx.TOP, 7)

        gbox = wx.GridSizer(rows=1, cols=2, hgap=7, vgap=0)
        gbox.Add(vbox1, 1, wx.EXPAND)
        gbox.Add(vbox2, 1, wx.EXPAND)

        self._mainbox.Add(self.tb_file, 0, wx.EXPAND|wx.BOTTOM, 7)
        self._mainbox.Add(gbox, 0, wx.EXPAND)

        # bindings
        self.inp_dtt.Bind_to(self.OnModifyDetector)
        self.inp_spc.Bind_to(self.OnModifySpec)
        self.inp_wire.Bind_to(self.OnModifyWire)
        self.inp_mon.Bind_to(EVT_MON_MODIFY, self.OnModifyMonitor)
        self.inp_mon.Bind_to(EVT_MON_PLOT, self.OnPlotMonCorrection)
        self.inp_mon.Bind_to(EVT_MON_CHECK, self.OnCheckMonCorrection)
        self.inp_dat.Bind_to(EVT_DATA_MODIFY, self.OnModifyData)
        self.inp_dat.Bind_to(EVT_DATA_MOVE, self.OnMoveData)
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

    # Getters
    def GetScan(self):
        """return PanelData.scan obj"""
        return self._parent.GetScan()  # main.py PanelData.GetScan  =PanelDate.scan obj

    # Setters
    def Update(self, part="all"):
        print('---  in PanelDataControls.Update() ----\n')
        scan = self.GetScan()

        print('scan in PanelDataControls.Update()',scan)
        
        print(scan.hdf5scanId)
        print(scan.spec_file)

        if part in ("detector", "all"):

            self.SetDetector(scan)

        if part in ("spec", "all"):

            self.SetSpec(scan)

        if part in ("wire", "all"):

            self.SetWire(scan)

        if part in ("data", "all"):

            self.SetData(scan)

        if part in ("mon", "all"):

            self.SetMonitor(scan)

    def SetFilename(self, filepath):
        self.tb_file.SetValue(filepath)

    def SetDetector(self, scan):
        scan_dict = scan.to_dict("setup")
        self.inp_dtt.SetValue(label=scan_dict['CCDType'], params=scan_dict['detCalib'])

    def SetSpec(self, scan):
        scan_dict = scan.to_dict("spec")
        if not scan_dict['specFile'].endswith('.h5'): # spec file
            self.inp_spc.filetype='spec'
            self.inp_spc.SetValue(fname=scan_dict['specFile'], scan=scan_dict['scanNumber'], comm=scan_dict['scanCmd'])
        else: #hdf5 file
            self.inp_spc.filetype='hdf5'
            print("scan_dict['hdf5scanId']", scan_dict['hdf5scanId'])
            self.inp_spc.SetValue(fname=scan_dict['specFile'],
                                    scan=scan_dict['scanNumber'],
                                    comm=scan_dict['scanCmd'],
                                    hdf5scanId=scan_dict['hdf5scanId'])

    def SetWire(self, scan):
        scan_dict = scan.to_dict("wire")
        self.inp_wire.SetValue(wires=scan_dict['wire'], traj=scan_dict['wireTrajAngle'])

    def SetData(self, scan):
        scan_dict = scan.to_dict("data")
        self.inp_dat.SetValue(scantype=scan_dict["type"], folder=scan_dict["imageFolder"],
                              prefix=scan_dict["imagePrefix"], index0=scan_dict["imageFirstIndex"],
                              ndigits=scan_dict["imageDigits"], skip=scan_dict["skipFrame"],
                              setsize=scan_dict["size"], subFolder=scan_dict["lineSubFolder"])

    def SetMonitor(self, scan):
        scan_dict = scan.to_dict("mon")
        self.inp_mon.SetValue(mode=scan_dict["monitor"], roi=scan_dict["monitorROI"],
                              monoff=int(scan_dict["monitorOffset"]), ccdoff=int(scan_dict["imageOffset"]))

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
        myevent = ControlEvent(part="spec", args={'specFile':fname, 'scanNumber':scan, 'scanCmd':comm})
        self.Notify(myevent)

    def OnModifyMonitor(self, event):
        mode, roi, monoff, ccdoff = event.GetValue()
        myevent = ControlEvent(part="mon", args={'monitor':mode, 'monitorROI':roi, 'monitorOffset':monoff, 'imageOffset':ccdoff})
        self.Notify(myevent)

    def OnModifyData(self, event):
        scantype, folder, prefix, index0, ndigits, skip, setsize, subFolder = event.GetValue()
        myevent = ControlEvent(part="data", args={'type': scantype, 'imageFolder': folder, 'imagePrefix': prefix,
                                        'imageFirstIndex': index0, 'imageDigits': ndigits,
                                        'skipFrame': skip, 'size': setsize, 'lineSubFolder': subFolder})
        self.Notify(myevent)

    def OnMoveData(self, event):

        myevent = ControlEvent(part="move", args={'position':event.GetPosition()})

        self.Notify(myevent)

    def OnPlotMonCorrection(self, event):

        self.GetScan().plot_monitor()

    def OnCheckMonCorrection(self, event):

        self.GetScan().plot_monitor_corrected()

    # Binders
    def Bind_to(self, callback):
        self._observers.append(callback)

    def Notify(self, event):
        # print self.GetValue()
        for callback in self._observers:
            callback(event)


# Event classes
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
    tmp = PanelDataControls(frame)
    box.Add(tmp, 1, wx.EXPAND)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()