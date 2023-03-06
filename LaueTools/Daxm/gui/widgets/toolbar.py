#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx
import wx.lib.buttons as wxlb

import LaueTools.Daxm.gui.icons.icon_manager as mycons


IMG_W, IMG_H = 25, 25


class RadioButtonBar(wx.GridSizer):

    # Constructors
    def __init__(self, parent, buttons):
        wx.GridSizer.__init__(self, rows=1, cols=len(buttons), vgap=1, hgap=0)

        self._parent = None
        self._buttons = None
        self._buttons_id = None
        self._observers = None

        self.toggle = None

        self.Create(parent, buttons)

    def Create(self, parent, buttons):

        self._parent = parent
        self._buttons = []
        self._buttons_id = []
        self._observers = []

        self.toggle = 0

        for lbl, bitmap in buttons:

            if isinstance(bitmap, str):
                bitmap = mycons.get_icon_bmp(bitmap, (IMG_W, IMG_H))

            button = wxlb.ThemedGenBitmapTextToggleButton(self._parent, label=lbl, bitmap=bitmap)

            button.Bind(wx.EVT_BUTTON, self._OnToggleButton)

            self._buttons.append(button)

        self._buttons_id = [btn.GetId() for btn in self._buttons]

        self.AddMany([(btn, 0, wx.EXPAND, wx.ALL, 1) for btn in self._buttons])

        # self.SetToggle(0)

    # Methods to set and get toggled button
    def GetToggle(self):
        return self.toggle

    def SetToggle(self, button_id):

        self.toggle = button_id

        for i, btn in enumerate(self._buttons):

            if i == self.toggle:
                btn.SetValue(True)
            else:
                btn.SetValue(False)

        self._Notify()

    # Event Handlers
    def _OnToggleButton(self, event):

        btn_id = event.GetId()
        btn_id = self._buttons_id.index(btn_id)

        self.SetToggle(btn_id)

    # Outside binding and notifying
    def Bind_to(self, callback):

        self._observers.append(callback)

    def _Notify(self):

        event = wx.CommandEvent()
        event.SetInt(self.toggle)

        for callback in self._observers:
            callback(event)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    imageFiles = ("pencil.png",
                  "harddrive.png",
                  "controls.png",
                  "chainsaw.png",
                  "wall.png",
                  "visual.png")

    label = ["   Experiment   ",
             "    Dataset     ",
             "   Calibration  ",
             "  Segmentation  ",
             " Reconstruction ",
             "  Visualization "]

    bb = RadioButtonBar(frame, zip(label, imageFiles))

    frame.SetSizer(bb)
    frame.Layout()
    bb.Layout()
    bb.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
