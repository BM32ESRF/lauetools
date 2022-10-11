#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os
import wx

from LaueTools.Daxm.utils.list import unique_order


class DirPickerButton(wx.Button):

    # Constructor
    def __init__(self, parent, default):
        wx.Button.__init__(self, parent, label="...", style=wx.BU_EXACTFIT)

        self.parent = parent
        self.default = default
        self._observers = []

        self.Bind(wx.EVT_BUTTON, self.OnPushButton)

    # Event handlers
    def OnPushButton(self, event):

        dlg = wx.DirDialog(self.parent, message="Select directory",
                           defaultPath=self.default,
                           style=wx.DD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.default = path
            self.Notify(path)

        dlg.Destroy()

    # Outside binding and notifying
    def Bind_to(self, callback):
        self._observers.append(callback)

    def Notify(self, path):

        event = wx.CommandEvent()
        event.SetString(path)

        for callback in self._observers:
            callback(event)

class DirSelect(wx.BoxSizer):

    def __init__(self, parent, label, value):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self.parent = parent
        self.value = value
        self._observers = []

        self.label = wx.StaticText(self.parent, label=label)
        self.text = wx.TextCtrl(self.parent, style=wx.TE_PROCESS_ENTER)
        self.button = DirPickerButton(self.parent, value)

        self.Add(self.label, 0, wx.CENTER)
        self.Add(self.text, 1, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
        self.Add(self.button, 0)

        self.text.SetValue(value)

        # bindings
        self.button.Bind_to(self.OnPushButton)
        self.text.Bind(wx.EVT_TEXT_ENTER, self.OnTextEnter)

    # Set and get current directory
    def SetValue(self, fn):

        if os.path.exists(fn):
            self.value = fn

        else:
            pass

        self.text.ChangeValue(self.value)

    def _SetValue(self, fn):

        if os.path.exists(fn):
            self.value = fn
            self.Notify()
        else:
            pass

        self.text.ChangeValue(self.value)

    def GetValue(self):

        return self.value

    # Event handlers
    def OnPushButton(self, event):

        self._SetValue(event.GetString())

    def OnTextEnter(self, event):

        fn = self.text.GetValue()

        self._SetValue(fn)

    # Outside binding and notifying
    def Bind_to(self, callback):

        self._observers.append(callback)

    def Notify(self):

        for callback in self._observers:
            callback(self)

    # Display methods
    def Enable(self, status=True):

        self.text.Enable(status)

        self.button.Enable(status)

class DirSelectHistory(wx.BoxSizer):

    # Constructor
    def __init__(self, parent, label, value="", size=(-1, -1)):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self.parent = parent
        self._observers = []

        self.label = wx.StaticText(self.parent, label=label)
        self.cbx = wx.ComboBox(self.parent, wx.ID_ANY, style=wx.CB_READONLY | wx.CB_DROPDOWN,
                               value="", choices=[], size=size)
        self.btn = DirPickerButton(self.parent, value)

        self.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self.cbx, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT | wx.LEFT | wx.EXPAND, 5)
        self.Add(self.btn, 0, wx.ALIGN_CENTER_VERTICAL)

        if len(value):
           self.SetValue(value)

        self.cbx.Bind(wx.EVT_COMBOBOX, self.OnSelect)
        self.btn.Bind_to(self.OnPickDir)

    # Method to manipulate the list of directories
    def SetValue(self, item):

        self.cbx.Insert(item, 0)

        self.UniqueChoices()

    def SetChoices(self, choices):

        self.ClearChoices()

        for item in choices:
            self.cbx.Append(item)

        self.SetSelection(0)

    def GetValue(self):

        return self.cbx.GetStringSelection()

    def GetChoices(self):

        return [self.cbx.GetString(i) for i in range(self.cbx.GetCount())]

    def ClearChoices(self):

        self.cbx.Clear()

        self.cbx.Append("")

        self.cbx.SetSelection(0)

        self.cbx.Clear()

    def UniqueChoices(self):

        choices = self.GetChoices()

        choices = unique_order(choices)

        self.SetChoices(choices)

    def SetSelection(self, idx):

        self.cbx.SetSelection(idx)

    # Event handlers
    def OnSelect(self, event):

        choices = self.GetChoices()

        choices.insert(0, choices.pop(event.GetSelection()))

        self.SetChoices(choices)

        self.cbx.SetSelection(0)

        self.Notify()

    def OnPickDir(self, event):

        self.SetValue(event.GetString())

        self.Notify()

    # Display method
    def SetFocus(self):

        self.cbx.SetFocus()

    def Enable(self, status=True):

        self.cbx.Enable(status)

        self.btn.Enable(status)

    # Outside binding and notifying
    def Bind_to(self, callback):

        self._observers.append(callback)

    def Notify(self):

        for callback in self._observers:
            callback(self)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")
    bb = DirSelectHistory(frame, "Folder: ", "/home/renversa")

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()