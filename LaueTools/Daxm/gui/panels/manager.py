#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os
import wx

import wx.lib.agw.aui as aui

from LaueTools.Daxm.gui.widgets.toolbar import RadioButtonBar
from LaueTools.Daxm.gui.widgets.directory import DirSelectHistory
from LaueTools.Daxm.gui.widgets.text import LabelTxtCtrl

from LaueTools.Daxm.gui.panels.experiment import PanelExperiment
from LaueTools.Daxm.gui.panels.data import PanelData

from LaueTools.Daxm.gui.windows.load_scan import LoadScan

import LaueTools.Daxm.gui.icons.icon_manager as mycons

TB_COLOR_BG = "DIM GREY"
TB_COLOR_FG = "WHITE"


class PanelManager2(aui.AuiNotebook):

    def __init__(self, parent):
        aui.AuiNotebook.__init__(self, parent, wx.ID_ANY,
                                 agwStyle=aui.AUI_NB_CLOSE_ON_ALL_TABS
                                          |aui.AUI_NB_TAB_MOVE
                                          |aui.AUI_NB_SCROLL_BUTTONS
                                          |aui.AUI_NB_WINDOWLIST_BUTTON)

        self.parent = parent
        self.pageqty = 0
        self.newmenu = None

        self.Create()

    def Create(self):

        self.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

        self.NewPage("simulation")

        # menu
        # menu = wx.Menu()
        # addsimu = wx.MenuItem(menu, wx.ID_ANY, text="Simulation")
        # addsimu.SetBitmap(mycons.get_icon_bmp("pencil.png"))
        # menu.Append(addsimu)

        # addexpe = wx.MenuItem(menu, wx.ID_ANY, text="Experiment")
        # addexpe.SetBitmap(mycons.get_icon_bmp("harddrive.png"))
        # menu.Append(addexpe)

        menu = wx.Menu()

        addsimu= menu.Append(wx.ID_ANY, "Simulation","")
        addsimu.SetBitmap(mycons.get_icon_bmp("pencil.png"))

        addexpe = menu.Append(wx.ID_ANY, "Experiment","")
        addexpe.SetBitmap(mycons.get_icon_bmp("harddrive.png"))

        menu.Bind(wx.EVT_MENU, self.OnNewPageSimulation, addsimu)
        menu.Bind(wx.EVT_MENU, self.OnNewPageExperiment, addexpe)

        self.newmenu = menu

        # events
        self.Bind(aui.EVT_AUINOTEBOOK_TAB_RIGHT_UP, self.OnRightClick)
        self.Bind(aui.EVT_AUINOTEBOOK_BG_RIGHT_UP, self.OnRightClick)

    # new pages
    def NewPage(self, type):

        page, name, bitmap = None, None, None

        if type == "simulation":

            page, name, bitmap = self.NewPageSimulation()

        elif type == "experiment":

            page, name, bitmap = self.NewPageExperiment()

        else:
            pass

        if page is not None:
            self.InsertPage(self.pageqty, page, name, select=True, bitmap=bitmap)
            self.SetRenamable(self.pageqty, renamable=True)
            self.pageqty += 1

        if self.pageqty == 1:
            self.SetCloseButton(0, False)

    def NewPageSimulation(self, event=None):

        return PanelExperiment(self.parent), "Simulation", mycons.get_icon_bmp("pencil.png")

    def NewPageExperiment(self, event=None):

        dlg = LoadScan(self.parent)

        if dlg.ShowModal() == wx.ID_OK:

            page = PanelData(self.parent, dlg.GetScan())

            page.SetFilename(dlg.GetFilename())

        else:
            page = None

        dlg.Destroy()

        return page, "Experiment",  mycons.get_icon_bmp("harddrive.png")

    # event handlers
    def OnNewPageSimulation(self, event):

        self.NewPage("simulation")

    def OnNewPageExperiment(self, event):

        self.NewPage("experiment")

    def OnRightClick(self, event):
        self.parent.PopupMenu(self.newmenu)
