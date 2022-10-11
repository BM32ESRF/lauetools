#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx
from wx import SpinCtrlDouble as WxSpinCtrlDouble

from LaueTools.Daxm.utils.mystring import is_int, is_float
from LaueTools.Daxm.gui.widgets.spin import USE_WXSPINCTRL

class MySpinCtrlDouble(wx.BoxSizer):

    # Constructors
    def __init__(self, parent, id=wx.ID_ANY, value="", pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.TE_PROCESS_ENTER, min=0, max=100, initial=0, inc=1, name="wxSpinCtrl"):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = None
        self._status = None
        self._observers = []

        self._text = None
        self._format = None
        self._btn_down = None
        self._btn_up = None
        self._timer_down = None
        self._timer_up = None

        self.Digits = 2
        self.Increment = 1
        self.Max  = 0
        self.Min  = 100
        self.Range= [self.Min, self.Max]
        self.Value= 0

        self.Create(parent, id, value, pos, size, style, min, max, initial, inc, name)

    def Create(self, parent, id=wx.ID_ANY, value="", pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.TE_PROCESS_ENTER, min=0, max=100, initial=0, inc=1, name="wxSpinCtrlDouble"):

        self._parent = parent
        self._status = True
        self._timer_down = wx.Timer(parent)
        self._timer_up   = wx.Timer(parent)

        self._text = wx.TextCtrl(self._parent, id, value=value, size=size, style=style|wx.TE_RIGHT, name=name)

        self._btn_down = wx.Button(self._parent, wx.ID_ANY, chr(9207), size=(16, 13))

        self._btn_up = wx.Button(self._parent, wx.ID_ANY, chr(9206), size=(16, 13))

        vbox = wx.GridSizer(2, 1, 0, 0)
        vbox.Add(self._btn_up, 0)
        vbox.Add(self._btn_down, 0)

        self.Add(self._text, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 1)
        self.Add(vbox, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 1)

        # bindings
        # self.text.Bind(wx.EVT_TEXT, self.on_text_enter)
        self._text.Bind(wx.EVT_TEXT_ENTER, self._OnTextEnter)
        self._btn_down.Bind(wx.EVT_LEFT_DOWN, self._OnButtonDownPressed)
        self._btn_up.Bind(wx.EVT_LEFT_DOWN, self._OnButtonUpPressed)
        self._btn_down.Bind(wx.EVT_LEFT_UP, self._OnButtonDownReleased)
        self._btn_up.Bind(wx.EVT_LEFT_UP, self._OnButtonUpReleased)
        self._parent.Bind(wx.EVT_TIMER, self._OnTimerDown, self._timer_down)
        self._parent.Bind(wx.EVT_TIMER, self._OnTimerUp, self._timer_up)

        # init
        self.SetDigits()
        self.SetIncrement(inc)
        self.SetMin(min)
        self.SetMax(max)
        self.SetValue(initial)

    # Getters
    def GetDigits(self):
        return self.Digits

    def GetIncrement(self):
        return self.Increment

    def GetMax(self):
        return self.Max

    def GetMin(self):
        return self.Min

    def GetRange(self):
        return self.Min, self.Max

    def GetValue(self):
        return self.Value

    # Setters
    def SetDigits(self, digits=None):

        if digits is None:
            digits = self.Digits

        if digits <=0 or not is_int(digits):
            raise ValueError

        self.Digits = digits
        self._format = "{:."+str(self.Digits)+"f}"
        self.SetValue()

    def SetIncrement(self, inc):
        self.Increment = float(inc)

    def SetMax(self, maxVal):

        self.Max = float(maxVal)

        if self.Value > maxVal:

            self.SetValue(maxVal)

    def SetMin(self, minVal):

        self.Min = float(minVal)

        if self.Value < minVal:

            self.SetValue(minVal)

    def SetRange(self, minVal, maxVal):

        if minVal is None:
            minVal = self.Min

        if maxVal is None:
            maxVal = self.Max

        self.SetMin(minVal)

        self.SetMax(maxVal)

        self._EnableSpin()

    def SetValue(self, value=None):

        if value is None:

            value = self.Value

        if is_float(value):

            value = float(value)

            if value > self.Max:

                self.Value = self.Max

            elif value < self.Min:

                self.Value = self.Min

            else:
                self.Value = value

            self._text.ChangeValue(self._format.format(self.Value))

            self._EnableSpin()

    # Event handlers
    def _OnTextEnter(self, event):

        val = event.GetString()

        if is_float(val):

            self.SetValue(float(val))

            self._Notify()
        else:

            self.SetValue(None)

    def _OnButtonDownPressed(self, event):

        if self._timer_down.IsRunning():
            self._timer_down.Stop()
            # print("timer stopped!")
        else:
            # print("starting timer...")
            self._DecreaseValue(self.Increment)
            self._timer_down.Start(200)

        event.Skip()

    def _OnButtonDownReleased(self, event):

        if self._timer_down.IsRunning():
            self._timer_down.Stop()
            # print("timer stopped!")

        event.Skip()

    def _DecreaseValue(self, step=1):

        if self.Value > self.Min:

            self.SetValue(self.Value - step)

            self._Notify()

    def _OnTimerDown(self, event):

        self._DecreaseValue(self.Increment)

        if self.Value == self.Min:
            self._timer_down.Stop()

    def _OnButtonUpPressed(self, event):

        if self._timer_up.IsRunning():
            self._timer_up.Stop()
            # print("timer stopped!")
        else:
            # print("starting timer...")
            self._IncreaseValue(self.Increment)
            self._timer_up.Start(200)

        event.Skip()

    def _OnButtonUpReleased(self, event):

        if self._timer_up.IsRunning():
            self._timer_up.Stop()
            # print("timer stopped!")

        event.Skip()

    def _IncreaseValue(self, step=1):

        if self.Value < self.Max:

            self.SetValue(self.Value + step)

            self._Notify()

    def _OnTimerUp(self, event):

        self._IncreaseValue(self.Increment)

        if self.Value == self.Max:
            self._timer_up.Stop()

    # Methods to enable/disable widget
    def _EnableSpin(self, status=None):

        if status is None:

            status = self._status

        if self.Value > self.Min:
            self._btn_down.Enable(self._status)
        else:
            self._btn_down.Enable(False)
            if self._timer_down.IsRunning():
                self._timer_down.Stop()

        if self.Value < self.Max:
            self._btn_up.Enable(self._status)
        else:
            self._btn_up.Enable(False)
            if self._timer_up.IsRunning():
                self._timer_up.Stop()

    def Enable(self, status=True):

        self._status=status

        self._text.Enable(status)

        self._EnableSpin(status)

    # Outside bindings
    def Bind(self, event, callback):

        if event == wx.EVT_SPINCTRLDOUBLE:
            self._observers.append(callback)

    def _Notify(self):

        for callback in self._observers:
            callback(self)


if USE_WXSPINCTRL:
    SpinCtrlDouble = WxSpinCtrlDouble
else:
    SpinCtrlDouble = MySpinCtrlDouble


class LabelSpinCtrlDouble(wx.BoxSizer):

    def __init__(self, parent, label="", min=0, max=100, initial=0, inc=1, size=wx.DefaultSize):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent

        self._label = wx.StaticText(self._parent, label=label)
        self._spin = SpinCtrlDouble(self._parent, wx.ID_ANY, size=size, min=float(min), max=float(max), initial=initial, inc=inc)

        self.Add(self._label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self._spin, 0, wx.ALIGN_CENTER_VERTICAL)

        self.SetDigits()

    def GetValue(self):
        return self._spin.GetValue()

    def SetValue(self, value):
        self._spin.SetValue(float(value))

    def SetMin(self, minVal):
        self._spin.SetMin(float(minVal))

    def SetMax(self, maxVal):
        self._spin.SetMax(float(maxVal))

    def SetIncrement(self, inc):
        self._spin.SetIncrement(inc)

        self.SetDigits()

    def SetDigits(self, digits = None):

        if digits is None:
            # automatically set number of digits
            for i in range(8):
                if self._spin.GetIncrement() <= 10**-i:
                    digits = i
            self._spin.SetDigits(digits)

        else:
            self._spin.SetDigits(digits)

    def Bind_to(self, callback):
        self._spin.Bind(wx.EVT_SPINCTRLDOUBLE, callback)

    def Enable(self, status=True):
        self._spin.Enable(status)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Spin test")
    bb = LabelSpinCtrlDouble(frame, label="test: ", min=0, max=10, initial=5, inc=0.0001)

    if 0:
        def test(event):
            print("prout", event.GetValue())

        bb.Bind(wx.EVT_SPINCTRL, test)

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
