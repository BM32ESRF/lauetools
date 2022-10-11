#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

from LaueTools.Daxm.classes.scan.scan import new_scan
from LaueTools.Daxm.classes.scan.point import load_scan_dict, save_scan_dict


from LaueTools.Daxm.gui.panels.data.figure import PanelDataFigure
from LaueTools.Daxm.gui.panels.data.controls import PanelDataControls


class PanelData(wx.Panel):

    # Constructors
    def __init__(self, parent, scan_dict):
        wx.Panel.__init__(self, parent)

        self.parent = parent
        self.mainbox = None
        self.fig_panel = None
        self.ctrl_panel = None

        self.scan = new_scan(scan_dict, verbose=False)

        self.Create()
        self.Init()

    def Create(self):

        # create panels
        self.fig_panel = PanelDataFigure(self)

        self.ctrl_panel = PanelDataControls(self)

        # create sizer
        self.mainbox = wx.BoxSizer(wx.HORIZONTAL)

        self.mainbox.Add(self.fig_panel, 3, wx.EXPAND|wx.ALL, 5)
        self.mainbox.Add(self.ctrl_panel, 4, wx.EXPAND|wx.ALL, 5)

        self.SetSizer(self.mainbox)
        self.Layout()
        self.mainbox.Layout()
        self.mainbox.Fit(self)
        self.Fit()

        #binding
        self.ctrl_panel.Bind_to(self.OnControlEvent)
        self.fig_panel.Bind_to(self.OnFigureEvent)

    def Init(self):
        # run it
        self.ctrl_panel.Update()
        self.fig_panel.Update()

    # Setters
    def SetFilename(self, filepath):
        self.ctrl_panel.SetFilename(filepath)

    # Getters
    def GetScan(self):
        return self.scan

    # Event handlers
    def OnControlEvent(self, event):
        if event.GetPart() == "move":
            self.scan.goto(*event.GetArgs()['position'])
        else:
            if event.GetPart() == "data" and (self.scan.get_type() != event.GetArgs()['type']):
                scan_dict = self.scan.to_dict()
                scan_dict.update(event.GetArgs())
                self.scan = new_scan(scan_dict, verbose=False)
            else:
                pass

            self.scan.update(scan_dict=event.GetArgs(), part=event.GetPart())

        self.fig_panel.Update(part=event.GetPart())

    def OnFigureEvent(self, event):
        self.scan.update(scan_dict=event.GetArgs(), part=event.GetPart())
        self.ctrl_panel.Update(part=event.GetPart())

    def OnOpen(self, event):

        print("In PanelData: Loading scan from file: %s"%event.GetPath())

        self.scan.load(event.GetPath())

        self.ctrl_panel.Update()

        self.fig_panel.Update()

    def OnSave(self, event):

        print("Saving scan to file: %s"%event.GetPath())

        self.scan.save(event.GetPath())
