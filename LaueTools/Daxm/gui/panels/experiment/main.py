#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

#from LaueTools.Daxm.classes.scan.scan import load_scan_dict, save_scan_dict, new_simscan_dict#, SimScan
from LaueTools.Daxm.classes.scan.point import load_scan_dict, save_scan_dict
from LaueTools.Daxm.classes.scan.simu import  new_simscan_dict, SimScan

from LaueTools.Daxm.gui.panels.experiment.figure import PanelExperimentFigure
from LaueTools.Daxm.gui.panels.experiment.controls import PanelExperimentControls


class PanelExperiment(wx.Panel):

    # Constructors
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.parent = parent
        self.mainbox = None
        self.fig_panel = None
        self.ctrl_panel = None
        self.scan = None

        self.Create()
        self.Init()

    def Create(self):

        # init simscan
        self.scan = SimScan(inp=new_simscan_dict(), verbose=False)

        # create panels
        self.fig_panel = PanelExperimentFigure(self)

        self.ctrl_panel = PanelExperimentControls(self)

        # create sizer
        self.mainbox = wx.BoxSizer(wx.HORIZONTAL)

        self.mainbox.Add(self.fig_panel, 0, wx.EXPAND|wx.ALL, 5)
        self.mainbox.Add(self.ctrl_panel, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(self.mainbox)
        self.Layout()
        self.mainbox.Layout()
        self.mainbox.Fit(self)
        self.Fit()

        #binding
        self.ctrl_panel.Bind_to(self.OnControlEvent)

    def Init(self):
        # run it
        self.fig_panel.Update()

    # Getters
    def GetScan(self):
        return self.scan

    # Event handlers
    def OnControlEvent(self, event):
        self.scan.update(scan_dict=event.GetArgs(), part=event.GetPart())
        self.fig_panel.Update()

    def OnOpen(self, event):

        print(("In PanelExperiment: Loading scan from file: %s"%event.GetPath()))

        self.scan.load(event.GetPath())

        self.ctrl_panel.Update()

        self.fig_panel.Update()

    def OnSave(self, event):

        print(("Saving scan to file: %s"%event.GetPath()))

        self.scan.save(event.GetPath())