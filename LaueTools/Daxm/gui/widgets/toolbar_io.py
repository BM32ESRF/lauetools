#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os

import wx

from LaueTools.Daxm.gui.widgets.text import LabelTxtCtrl

EVT_OPEN = "event_open"
EVT_SAVE = "event_save"


class OpenSaveBar(wx.BoxSizer):

    def __init__(self, parent):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent
        self._observers = {EVT_OPEN: [],
                           EVT_SAVE: []}

        self.path_txt = None
        self.open_btn = None
        self.save_btn = None
        self.saveas_btn = None

        self.Create()

    def Create(self):

        self.path_txt = LabelTxtCtrl(self._parent, "File: ", "", wx.TE_READONLY)

        open_bmp = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.open_btn = wx.BitmapButton(self._parent, id=wx.ID_ANY, bitmap= open_bmp)
        self.open_btn.SetToolTip(wx.ToolTip("Open"))

        saveas_bmp = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE_AS, wx.ART_TOOLBAR, (16,16))
        self.saveas_btn = wx.BitmapButton(self._parent, id=wx.ID_ANY, bitmap=saveas_bmp)
        self.saveas_btn.SetToolTip(wx.ToolTip("Save as"))

        save_bmp = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, (16,16))
        self.save_btn = wx.BitmapButton(self._parent, id=wx.ID_ANY, bitmap=save_bmp)
        self.save_btn.SetToolTip(wx.ToolTip("Save"))

        self.Add(self.path_txt, 1, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self.open_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self.save_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self.saveas_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        # bindings
        self.open_btn.Bind(wx.EVT_BUTTON, self.OnOpen)
        self.save_btn.Bind(wx.EVT_BUTTON, self.OnSave)
        self.saveas_btn.Bind(wx.EVT_BUTTON, self.OnSaveAs)

        # setters
    def SetValue(self, filepath):

        self.path_txt.SetValue(filepath)

        self.save_btn.Enable(filepath != "" and os.path.exists(filepath))

    def GetValue(self):

        return self.path_txt.GetValue()

    # event handlers
    def OnOpen(self, event):

        dirname, basename = os.path.split(self.path_txt.GetValue())

        dlg = wx.FileDialog(self._parent,
                            defaultDir=dirname,
                            defaultFile=basename,
                            wildcard="",
                            style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            self.SetValue(dlg.GetPath())
            self._Notify(EVT_OPEN)

        dlg.Destroy()

    def OnSave(self, event):
        
        self._Notify(EVT_SAVE)

    def OnSaveAs(self, event):

        dirname, basename = os.path.split(self.path_txt.GetValue())

        dlg = wx.FileDialog(self._parent,
                            defaultDir=dirname,
                            defaultFile=basename,
                            wildcard="",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if dlg.ShowModal() == wx.ID_OK:
            self.SetValue(dlg.GetPath())
            self._Notify(EVT_SAVE)

        dlg.Destroy()

    # bindings
    def Bind_to(self, event, callback):

        self._observers[event].append(callback)

    def _Notify(self, event):

        newevent = wx.FileDirPickerEvent()
        newevent.SetPath(self.GetValue())

        for callback in self._observers[event]:
            callback(newevent)


# Test
if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = OpenSaveBar(frame)
    box.SetPath("")

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
