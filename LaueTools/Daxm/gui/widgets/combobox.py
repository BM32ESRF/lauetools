#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx
if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

class LabelComboBox(wx.BoxSizer):
    
    def __init__(self, parent, label, value="", choices=(), style=wx.DEFAULT,
                                        flag=wx.ALL, size=(-1,-1), tip=''):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)
        
        self.parent = parent
        self._observers=[]
        
        self.label  = wx.StaticText(self.parent, label=label)
        self.cbx   = wx.ComboBox(self.parent, wx.ID_ANY, value,  style=wx.CB_DROPDOWN|wx.CB_READONLY|style,
                                choices=list(choices), size=size)        
        
        self.Add(self.label, 0, wx.CENTER)
        self.Add(self.cbx,   1, wx.FIXED_MINSIZE)
        
        self.cbx.Bind(wx.EVT_COMBOBOX, self.OnSelect)

        #tooltip
        self.cbx.SetToolTipString(tip)

    def GetValue(self):
        
        return self.cbx.GetStringSelection()
            
    def SetValue(self, txt):        
        
        self.cbx.SetValue(txt)
    
    def _SetValue(self, txt):        
        
        self.cbx.SetValue(txt)
        
        for callback in self._observers:
            
            callback(self)    
                 
    def SetChoices(self, choices):
        
        val = self.GetValue()
        
        self.ClearChoices()
        
        for item in choices:
            
            self.cbx.Append(item)
            
        if not choices:
            
            self.SetValue("")
            
        elif val in choices:  
            
            self.SetValue(val)
        else:
        
            self.SetValue(choices[0])

    def ClearChoices(self):
        
        self.cbx.Clear()
        
        self.cbx.Append("")
        
        self.cbx.SetSelection(0)
        
        self.cbx.Clear()

    def OnSelect(self, event):
        
        self._Notify()
            
    def Bind_to(self, callback):
        
        self._observers.append(callback)
        
    def Enable(self, status=True):
                
        self.cbx.Enable(status) 
        
    def Clear(self):
        
        self.cbx.Clear()
        
    def Append(self, item):
        
        self.cbx.Append(item)    
        
    def SetSelection(self, idx):
        
        self.cbx.SetSelection(idx)
        
    def SetFocus(self):
        
        self.cbx.SetFocus()

    def _Notify(self):

        for callback in self._observers:
            callback(self)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")
    bb = LabelComboBox(frame, "Folder: ", "1", ["1", "2", "3", "11"])

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()