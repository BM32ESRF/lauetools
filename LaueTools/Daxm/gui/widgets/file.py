#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os
import wx


class FilePickerButton(wx.Button):

    # Constructor
    def __init__(self, parent, default_dir, wildcard, style=wx.FD_DEFAULT_STYLE, tip=''):
        wx.Button.__init__(self, parent, label="...", style=wx.BU_EXACTFIT)

        self._parent = parent
        self._observers = []

        self.default = default_dir
        self.style = style
        self.wildcard = wildcard

        self.tip=tip

        self.Bind(wx.EVT_BUTTON, self._OnPushButton)

    # Setters
    def SetWildCard(self, wildcard=None):

        if wildcard is None:
            wildcard = wx.FileSelectorDefaultWildcardStr

        self.wildcard = wildcard

    def SetDefaultDir(self, defaultdir):

        if os.path.isdir(defaultdir):
            self.default = defaultdir

    # Event handlers
    def _OnPushButton(self, event):

        dlg = wx.FileDialog(self._parent, message="Select a file: "+self.tip,
                            defaultDir=self.default,
                            defaultFile="",
                            wildcard=self.wildcard,
                            style=self.style)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.default = os.path.dirname(path)
            self._Notify(path)

        dlg.Destroy()

    # Outside binding and notifying
    def Bind_to(self, callback):
        self._observers.append(callback)

    def _Notify(self, path):

        event = wx.CommandEvent()
        event.SetString(path)

        for callback in self._observers:
            callback(event)


class FileSelectOpen(wx.BoxSizer):
    """ gui class to select a file (spec or hdf) to be opened"""
    def __init__(self, parent, label, value, wildcard=wx.FileSelectorDefaultWildcardStr, tip=''):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent
        self._observers = []

        self.value = value
        self.tip=tip

        self._label = wx.StaticText(self._parent, label=label)
        self._text = wx.TextCtrl(self._parent, style=wx.TE_PROCESS_ENTER)
        self._button = FilePickerButton(self._parent, default_dir=os.path.dirname(value), wildcard=wildcard,
                                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        self.Add(self._label, 0, wx.CENTER)
        self.Add(self._text, 1, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
        self.Add(self._button, 0)

        self._text.SetValue(value)

        # bindings
        self._button.Bind_to(self._OnPushButton)
        self._text.Bind(wx.EVT_TEXT_ENTER, self._OnTextEnter)

        #tool tip
        self._text.SetToolTipString(self.tip)

    # Setters
    def SetValue(self, fn):

        if os.path.exists(fn):
            self.value = fn

        else:
            pass

        self._text.ChangeValue(self.value)

    def SetWildCard(self, wildcard=None):

        self._button.SetWildCard(wildcard)

    def SetDefaultDir(self, default_dir):

        self._button.SetDefaultDir(default_dir)

    def _SetValue(self, fn):

        if os.path.exists(fn):
            self.value = fn
            self._Notify()
        else:
            pass

        self._text.ChangeValue(self.value)

    # Getters
    def GetValue(self):

        return self.value

    def IsBlank(self):

        return len(self.value) == 0

    def GetDefaultDir(self):

        return self._button.default

    # Event handlers
    def _OnPushButton(self, event):

        self._SetValue(event.GetString())

    def _OnTextEnter(self, event):

        fn = self._text.GetValue()

        self._SetValue(fn)

    # Outside binding and notifying
    def Bind_to(self, callback):

        self._observers.append(callback)

    def _Notify(self):

        for callback in self._observers:
            callback(self)

    # Display methods
    def Enable(self, status=True):

        self._text.Enable(status)

        self._button.Enable(status)


class FileSelectSave(wx.BoxSizer):

    def __init__(self, parent, label, value, wildcard=wx.FileSelectorDefaultWildcardStr):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent
        self.value = value
        self._observers = []

        self._label = wx.StaticText(self._parent, label=label)
        self._text = wx.TextCtrl(self._parent, style=wx.TE_PROCESS_ENTER)
        self._button = FilePickerButton(self._parent, default_dir=os.path.dirname(value), wildcard=wildcard,
                                        style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        self.Add(self._label, 0, wx.CENTER)
        self.Add(self._text, 1, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
        self.Add(self._button, 0)

        self._text.SetValue(value)

        # bindings
        self._button.Bind_to(self._OnPushButton)
        self._text.Bind(wx.EVT_TEXT_ENTER, self._OnTextEnter)

    # Set and get current directory
    def SetValue(self, fn):

        self.value = fn
        self._text.ChangeValue(self.value)

    def _SetValue(self, fn):

        self.value = fn
        self._Notify()
        self._text.ChangeValue(self.value)

    def GetValue(self):

        return self.value

    # Event handlers
    def _OnPushButton(self, event):

        self._SetValue(event.GetString())

    def _OnTextEnter(self, event):

        fn = self._text.GetValue()

        self._SetValue(fn)

    # Outside binding and notifying
    def Bind_to(self, callback):

        self._observers.append(callback)

    def _Notify(self):

        for callback in self._observers:
            callback(self)

    # Display methods
    def Enable(self, status=True):

        self._text.Enable(status)

        self._button.Enable(status)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")
    bb = FileSelectSave(frame, "File: ", "")

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()