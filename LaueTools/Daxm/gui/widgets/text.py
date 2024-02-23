#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

from LaueTools.Daxm.utils.mystring import is_float


COLOR_READONLY = wx.Colour(225, 225, 225)


class LabelTxtCtrlEnter(wx.BoxSizer):

    def __init__(self, parent, label, value, style=0):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent
        self._observers = []

        self.value = value

        self._label = wx.StaticText(self._parent, label=label)
        self._text = wx.TextCtrl(self._parent, style=style | wx.TE_PROCESS_ENTER)

        self.Add(self._label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self._text, 1, wx.ALIGN_CENTER_VERTICAL)

        self._text.SetValue(value)
        self._text.Bind(wx.EVT_TEXT_ENTER, self._OnTextEnter)

        if not self._text.IsEditable():
            self._text.SetBackgroundColour(COLOR_READONLY)

    def GetValue(self):

        return self.value

    def SetValue(self, val):

        self.value = val

        if not self._text.IsEditable():

            self._text.SetEditable(True)

            self._text.ChangeValue(str(val))

            self._text.SetEditable(False)

        else:
            self._text.ChangeValue(str(val))

    def _SetValue(self, val):

        self.SetValue(val)

        self._Notify()

    def Bind_to(self, callback):
        self._observers.append(callback)

    def SetEditable(self, status=True):

        self._text.SetEditable(status)

    def Enable(self, status=True):

        self._text.Enable(status)

    def _OnTextEnter(self, event):

        val = event.GetString()

        self._SetValue(val)

        self._Notify()

    def _Notify(self):

        for callback in self._observers:
            callback(self)


class LabelTxtCtrl(wx.BoxSizer):
    
    def __init__(self, parent, label, value, style=wx.TE_PROCESS_ENTER):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)
        
        self._parent = parent
        self._observers = []

        self.value = value

        self._label = wx.StaticText(self._parent, label=label)
        self._text = wx.TextCtrl(self._parent, style=style)# | wx.TE_PROCESS_ENTER)
        
        self.Add(self._label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self._text,  1,  wx.ALIGN_CENTER_VERTICAL)
        
        self._text.SetValue(value)
        self._text.Bind(wx.EVT_TEXT, self._OnTextEnter)
        
        if not self._text.IsEditable():
        
            self._text.SetBackgroundColour(COLOR_READONLY)

    def GetValue(self):

        return self.value

    def SetValue(self, val):

        self.value = val

        if not self._text.IsEditable():

            self._text.SetEditable(True)

            self._text.ChangeValue(str(val))

            self._text.SetEditable(False)

        else:
            self._text.ChangeValue(str(val))

    def _SetValue(self, val):

        self.SetValue(val)

        self._Notify()

    def Bind_to(self, callback):
        self._observers.append(callback)

    def SetEditable(self, status=True):

        self._text.SetEditable(status)

    def Enable(self, status=True):
                
        self._text.Enable(status)

    def _OnTextEnter(self, event):

        val = event.GetString()

        self._SetValue(val)

        self._Notify()

    def _Notify(self):

        for callback in self._observers:
            callback(self)


class LabelTxtCtrlNum(wx.BoxSizer):
    
    def __init__(self, parent, label, value, size=wx.DefaultSize, style=wx.TE_PROCESS_ENTER):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        if style == wx.TE_READONLY:
            style = wx.TE_READONLY | wx.TE_PROCESS_ENTER
        
        self._parent = parent
        self._observers = []

        self.value = value

        self._label = wx.StaticText(parent=self._parent, label=label)
        self._text = wx.TextCtrl(parent=self._parent, size=size, style=style)# | wx.TE_PROCESS_ENTER)
        
        self.Add(self._label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self._text,  1,  wx.ALIGN_CENTER_VERTICAL)
        
        self._text.SetValue(str(value))
        self._text.Bind(wx.EVT_TEXT_ENTER, self._OnTextEnter)
        # self._text.Bind(wx.TE_PROCESS_ENTER, self._OnTextEnter)
        
        if not self._text.IsEditable():
            self._text.SetBackgroundColour(COLOR_READONLY)
            
    def GetValue(self):
        
        return self.value
    
    def SetValue(self, val):        
        
        self.value = val
        
        if not self._text.IsEditable():
            
            self._text.SetEditable(True)
            
            self._text.ChangeValue(str(val))
            
            self._text.SetEditable(False)
            
        else:
            self._text.ChangeValue(str(val))

    def Bind_to(self, callback):
        self._observers.append(callback)

    def SetEditable(self, status=True):
        self._text.SetEditable(status)

        if status:
            self._text.SetBackgroundColour(wx.WHITE)
        else:
            self._text.SetBackgroundColour(wx.LIGHT_GREY)

    def Enable(self, status=True):
                
        self._text.Enable(status)

    def _SetValue(self, val):

        self.SetValue(val)

        self._Notify()

    def _OnTextEnter(self, event):

        val = event.GetString()

        self._SetValue(float(val))

        #if is_float(val):
        #
        #else:
        #    self._text.SetValue(str(self.value))
        #    self._text.SetValue(str(self.value))

    def _Notify(self):

        for callback in self._observers:
            callback(self)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")
    bb = LabelTxtCtrlNum(frame, "Folder: ", 1.333432324)

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
