#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os
import sys

path_to_daxm = os.path.abspath(os.path.dirname(__file__))
if path_to_daxm not in sys.path:
    print("Adding DAXM to the PATH...", path_to_daxm)
    print(path_to_daxm)
    sys.path.insert(0, path_to_daxm)

path_to_lauetools = os.path.dirname(path_to_daxm)
if path_to_lauetools not in sys.path:
    print("Adding lauetools to the PATH...", path_to_lauetools)
    sys.path.insert(0, path_to_lauetools)

import wx
import pprint
#from wx.lib.pubsub import pub
from pubsub import pub


try:
    from wx.adv import SplashScreen as SplashScreen
    splash_options = wx.adv.SPLASH_CENTRE_ON_SCREEN | wx.adv.SPLASH_TIMEOUT
except:
    from wx import SplashScreen as SplashScreen
    splash_options = wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT

import matplotlib as mpl
mpl.use('WXAgg')

from gui.panels.manager import PanelManager2 as PanelManager

#import LaueTools.Daxm.gui.icons.icon_manager as mycons
import gui.icons.icon_manager as mycons

class MainWindow(wx.Frame):
    """
    class of the main window of DAXM Analyzer GUI
    """
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(600, -1), style=wx.DEFAULT_FRAME_STYLE|wx.MAXIMIZE)

        self._mainbox = wx.BoxSizer()

        self.SetIcon(mycons.get_icon("logo_IF.png"))

        font = self.GetFont()
        font.SetPointSize(9)
        self.SetFont(font)

        self.CreatePanelManager()

        self.CreateStatusBar()

    def Run(self):

        self.Show(True)

    def SetStatusText(self, msg):

        self.statusbar.SetStatusText(msg)  

    def CreateStatusBar(self):

        self.statusbar = wx.Frame.CreateStatusBar(self, 1) 

        pub.subscribe(self.SetStatusText, 'set_status_text')

    def CreatePanelManager(self):

        self.manager = PanelManager(self)

        self._mainbox.Add(self.manager, 1, wx.EXPAND)

        self.SetSizer(self._mainbox)
        self.Layout()
        self._mainbox.Layout()
        self._mainbox.Fit(self)
        self.Fit()

    def OnAbout(self, e):

        dlg = wx.MessageDialog(self, "... work in progress...", "About 3D Laue Micro-Diffraction GUI", wx.OK)
        dlg.ShowModal() # Show it
        dlg.Destroy() # finally destroy it when finished.

    def OnExit(self,e):
        self.Close(True)  # Close the frame.  


def showSplashScreen(duration=1500):

    bmp = mycons.get_image_bmp("splash_screen_small.png")

    SplashScreen(bmp, splash_options,
                 duration, None, -1, wx.DefaultPosition, wx.DefaultSize,
                 wx.BORDER_SIMPLE | wx.STAY_ON_TOP)

    wx.Yield()
    wx.Sleep(1)

def start():
    app = wx.App(False)
    frame = MainWindow(None, "3D Laue Micro-Diffraction Analyzer")
    showSplashScreen()
    frame.Run()
    app.MainLoop()
    
if __name__ == '__main__':
    start()
