#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx

from LaueTools.Daxm.utils.mystring import is_int

from wx import SpinCtrl as WxSpinCtrl

USE_WXSPINCTRL = False

class MySpinCtrl(wx.BoxSizer):

    # Constructors
    def __init__(self, parent, id=wx.ID_ANY, value="", pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.TE_PROCESS_ENTER, min=0, max=100, initial=0, name="wxSpinCtrl"):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = None
        self._status = None
        self._observers = []

        self._text = None
        self._btn_down = None
        self._btn_up = None
        self._timer_down = None
        self._timer_up = None
        self._counter_up = 0
        self._step_up = 1

        self.Base = 10
        self.Max  = 0
        self.Min  = 100
        self.Range= [self.Min, self.Max]
        self.Value= 0

        self.Create(parent, id, value, pos, size, style, min, max, initial, name)

    def Create(self, parent, id=wx.ID_ANY, value="", pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.TE_PROCESS_ENTER, min=0, max=100, initial=0, name="wxSpinCtrl"):

        self._parent = parent
        self._status = True
        self._timer_down = wx.Timer(parent)
        self._timer_up   = wx.Timer(parent)
        self._counter_up = 0
        self._ste_up = 1

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
        self.SetMin(min)
        self.SetMax(max)
        self.SetValue(initial)

    # Getters
    def GetBase(self):
        return 10

    def GetMax(self):
        return self.Max

    def GetMin(self):
        return self.Min

    def GetRange(self):
        return self.Min, self.Max

    def GetValue(self):
        return self.Value

    # Setters
    def SetBase(self, base):
        return False

    def SetMax(self, maxVal):

        self.Max = maxVal

        if self.Value > maxVal:

            self.SetValue(maxVal)

        if not self._btn_up.IsEnabled():

            self._EnableSpin()

    def SetMin(self, minVal):

        self.Min = minVal

        if self.Value < minVal:

            self.SetValue(minVal)

        if not self._btn_up.IsEnabled():

            self._EnableSpin()

    def SetRange(self, minVal, maxVal):

        if minVal is None:
            minVal = self.Min

        if maxVal is None:
            maxVal = self.Max

        self.SetMin(minVal)

        self.SetMax(maxVal)

        self._EnableSpin()

    def SetSelection(self, from_, to_):

        self._text.SetSelection(from_, to_)

    def SetValue(self, value=None):

        if value is None:

            value = self.Value

        if isinstance(value, int):

            if value > self.Max:

                self.Value = self.Max

            elif value < self.Min:

                self.Value = self.Min

            else:
                self.Value = value

            self._text.ChangeValue(str(self.Value))

            self._EnableSpin()

    # Event handlers
    def _OnTextEnter(self, event):

        val = event.GetString()

        if is_int(val):

            self.SetValue(int(val))

            self._Notify()
        else:

            self.SetValue(None)

    def _OnButtonDownPressed(self, event):

        if self._timer_down.IsRunning():
            self._timer_down.Stop()
            # print("timer stopped!")
        else:
            # print("starting timer...")
            self._DecreaseValue()
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

        self._DecreaseValue()

        if self.Value == self.Min:
            self._timer_down.Stop()

    def _OnButtonUpPressed(self, event):

        if self._timer_up.IsRunning():
            self._timer_up.Stop()
            # print("timer stopped!")
        else:
            # print("starting timer...")
            self._IncreaseValue()
            self._timer_up.Start(200)
            
        event.Skip()

    def _OnButtonUpReleased(self, event):

        if self._timer_up.IsRunning():
            self._timer_up.Stop()
            # print("timer stopped!")
            self._counter_up = 0
            self._step_up = 1
        event.Skip()

    def _IncreaseValue(self, step=1):

        if self.Value < self.Max:

            self.SetValue(self.Value + step)

            self._Notify()

    def _OnTimerUp(self, event):

        self._IncreaseValue()

        if self.Value == self.Max:
            self._timer_up.Stop()

    # A simple test to check feasibility
    def _OnTimerUpAccelerate(self, event):

        if self._counter_up == 10:
            self._step_up = self._step_up*2
            self._counter_up = 0

        self._IncreaseValue(self._step_up)

        self._counter_up = self._counter_up + 1

        if self.Value >= self.Max:
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

        if event == wx.EVT_SPINCTRL:
            self._observers.append(callback)

    def _Notify(self):

        for callback in self._observers:
            callback(self)


if USE_WXSPINCTRL:
    SpinCtrl = WxSpinCtrl
else:
    SpinCtrl = MySpinCtrl


class LabelSpinCtrl(wx.BoxSizer):

    def __init__(self, parent, label="", min=0, max=100, initial=0, size=wx.DefaultSize):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent

        self._label = wx.StaticText(self._parent, label=label)
        self._spin  = SpinCtrl(self._parent, wx.ID_ANY, size=size, min=min, max=max, initial=initial)

        self.Add(self._label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self._spin, 0, wx.ALIGN_CENTER_VERTICAL)

    def GetValue(self):
        return self._spin.GetValue()

    def SetValue(self, value):
        self._spin.SetValue(value)

    def SetMin(self, minVal):
        self._spin.SetMin(minVal)

    def SetMax(self, maxVal):
        self._spin.SetMax(maxVal)

    def Bind_to(self, callback):
        self._spin.Bind(wx.EVT_SPINCTRL, callback)

    def Enable(self, status=True):
        self._spin.Enable(status)

    def ForceNotify(self):
        # cleaner way ?
        self._spin._Notify()


class LabelTwoSpinsCtrl(wx.BoxSizer):

    def __init__(self, parent, label, value, bounds, size=(-1, -1)):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self.parent = parent
        self.value = value
        self.range = bounds
        self._observers = []

        self.label = wx.StaticText(self.parent, label=label)
        self.spin1 = SpinCtrl(self.parent, wx.ID_ANY, size=size,
                              initial=value[0], min=bounds[0], max=value[1])
        self.spin2 = SpinCtrl(self.parent, wx.ID_ANY, size=size,
                              initial=value[1], min=value[0], max=bounds[1])

        self.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self.spin1, 1, wx.ALIGN_CENTER_VERTICAL)
        self.Add(self.spin2, 1, wx.ALIGN_CENTER_VERTICAL)

        self.spin1.Bind(wx.EVT_SPINCTRL, self.OnChange1)
        self.spin2.Bind(wx.EVT_SPINCTRL, self.OnChange2)

    def OnChange1(self, event):

        val = [self.spin1.GetValue(), self.GetValue()[1]]

        self._SetValue(val)

    def OnChange2(self, event):

        val = [self.GetValue()[0], self.spin2.GetValue()]

        self._SetValue(val)

    def GetValue(self):

        return self.value

    def GetBounds(self):

        return self.range

    def SetValue(self, val):

        if val[0] <= val[1]:

            self.value = val

            self.spin1.SetRange(*self.range)
            self.spin2.SetRange(*self.range)

            self.spin1.SetValue(self.value[0])
            self.spin2.SetValue(self.value[1])

            self.spin1.SetRange(self.range[0], self.value[1])
            self.spin2.SetRange(self.value[0], self.range[1])

        else:
            self.SetValue(val[::-1])

    def _SetValue(self, val):

        self.value = val

        self.spin1.SetRange(self.range[0], self.value[1])

        self.spin2.SetRange(self.value[0], self.range[1])

        for callback in self._observers:
            callback(self)

    def SetBounds(self, minval=None, maxval=None):

        if minval is None:
            minval = self.range[0]

        if maxval is None:
            maxval = self.range[1]

        self.range = [minval, maxval]

        if self.value[0] < self.range[0]:
            self.spin1.SetValue(self.range[0])

            self._SetValue([self.range[0], self.value[1]])

        if self.value[1] > self.range[1]:
            self.spin2.SetValue(self.range[1])

            self._SetValue([self.value[0], self.range[1]])

        self.spin1.SetRange(self.range[0], self.value[1])

        self.spin2.SetRange(self.value[0], self.range[1])

    def Bind_to(self, callback):

        self._observers.append(callback)

    def Enable(self, status=True):

        self.spin1.Enable(status)

        self.spin2.Enable(status)


class SpinSliderCtrl(wx.BoxSizer):

    def __init__(self, parent, label, min, max, initial, size=(-1, -1), style=wx.SL_HORIZONTAL):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)

        self._parent = parent
        self._observers = []

        self._label = wx.StaticText(self._parent, wx.ID_ANY, label)
        self._spin = SpinCtrl(self._parent, wx.ID_ANY, size=size, min=min, max=max, initial=initial)
        self._slider = wx.Slider(self._parent, wx.ID_ANY, style=wx.SL_AUTOTICKS | style,
                                minValue=min, maxValue=max, value=initial)

        self.Add(self._label, 0, wx.CENTER)
        self.Add(self._spin, 0, wx.CENTER)
        self.Add(self._slider, 1, wx.CENTER | wx.LEFT, 15)

        self._spin.Bind(wx.EVT_SPINCTRL, self._OnChangeValueSpin)
        self._slider.Bind(wx.EVT_SLIDER, self._OnChangeValueSlider)

    def GetValue(self):
        return self._spin.GetValue()

    def SetValue(self, val):
        self._spin.SetValue(val)
        self._slider.SetValue(val)

    def SetMinValue(self, minval):
        value = self.GetValue()

        self._spin.SetRange(minval, None)
        self._slider.SetMin(minval)

        self.Enable(self._spin.GetMin() != self._spin.GetMax())

        if self.GetValue() != value:
            self._Notify()

    def SetMaxValue(self, maxval):
        value = self.GetValue()

        self._spin.SetRange(None, maxval)
        self._slider.SetMax(maxval)

        self.Enable(self._spin.GetMin() != self._spin.GetMax())

        if self.GetValue() != value:
            self._Notify()

    def Bind_to(self, callback):
        self._observers.append(callback)

    def _OnChangeValueSpin(self, event):
        value = self._spin.GetValue()
        self._slider.SetValue(value)
        self._Notify()

    def _OnChangeValueSlider(self, event):
        value = self._slider.GetValue()
        self._spin.SetValue(value)
        self._Notify()

    def _Notify(self):
        for callback in self._observers:
            callback(self)

    def Enable(self, status):
        self._spin.Enable(status)
        self._slider.Enable(status)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Spin test")
    bb = LabelSpinCtrl(frame, "Test: ", 0, 1000, 5)

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
