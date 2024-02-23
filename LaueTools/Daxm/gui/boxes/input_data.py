#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import wx
import os

from LaueTools.Daxm.gui.widgets.text import LabelTxtCtrlEnter
from LaueTools.Daxm.gui.widgets.file import FilePickerButton
from LaueTools.Daxm.gui.widgets.spin import LabelSpinCtrl, SpinSliderCtrl

import LaueTools.readmccd as rmccd
import LaueTools.Daxm.utils.read_image as rimg

EVT_DATA_MODIFY = "event_modify"
EVT_DATA_MOVE = "event_move"


class InputData(wx.StaticBoxSizer):

    def __init__(self, parent, label="Dataset"):
        mainbox = wx.StaticBox(parent, wx.ID_ANY, label)

        wx.StaticBoxSizer.__init__(self, mainbox, wx.VERTICAL)

        mainbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        mainbox.SetForegroundColour('DIM GREY')

        self._box = mainbox
        self._parent = parent
        self._observers = {EVT_DATA_MODIFY: [],
                           EVT_DATA_MOVE: []}

        self.img_folder = ""
        self.img_prefix = ""
        self.img_digits = 4
        self.img_index0 = 0
        self.img_ext = "tif"

        self.folder_txt = None
        self.prefix_txt = None
        self.digits_spn = None
        self.index0_spn = None
        self.ext_txt = None
        self.firstimage_fp = None
        self.series_ckb = None
        self.notebook = None
        self.line_size_spn = None
        self.line_skip_spn = None
        self.line_pos_sld = None
        self.mesh_sizex_spn = None
        self.mesh_sizey_spn = None
        self.mesh_skip_spn = None
        self.mesh_posx_sld = None
        self.mesh_posy_sld = None
        self.mesh_line_ckb = None

        self.Create()
        self.Init()

    def Create(self):

        self.folder_txt = LabelTxtCtrlEnter(self._parent, "Folder: ", "")
        self.prefix_txt = LabelTxtCtrlEnter(self._parent, "Prefix: ", "")
        self.digits_spn = LabelSpinCtrl(self._parent, "ndigits: ", 0, 10, 4, size=(50, -1))
        self.index0_spn = LabelSpinCtrl(self._parent, "first index: ", 0, wx.INT64_MAX, 0, size=(60, -1))
        self.firstimage_fp = FilePickerButton(self._parent, "", "")

        self.series_ckb = wx.CheckBox(self._parent, wx.ID_ANY, label=" scan series")

        # notebook
        self.notebook = wx.Notebook(self._parent)

        panel1 = wx.Panel(self.notebook)

        self.line_size_spn = LabelSpinCtrl(panel1, "Size: ", 1, wx.INT64_MAX, 1)

        self.line_skip_spn = LabelSpinCtrl(panel1, "Skip: ", 0, wx.INT64_MAX, 0)

        self.line_pos_sld = SpinSliderCtrl(panel1, "Pos: ", size=(40, -1),
                                           min=0, max=1, initial=0)

        panel2 = wx.Panel(self.notebook)

        self.mesh_sizex_spn = LabelSpinCtrl(panel2, "SizeX: ", 1, wx.INT64_MAX, 1)

        self.mesh_sizey_spn = LabelSpinCtrl(panel2, "SizeY: ", 1, wx.INT64_MAX, 1)

        self.mesh_skip_spn = LabelSpinCtrl(panel2, "Skip: ", 0, wx.INT64_MAX, 0)

        self.mesh_line_ckb = wx.CheckBox(panel2, wx.ID_ANY, label=" line subfolders")

        self.mesh_posx_sld = SpinSliderCtrl(panel2, "PosX: ", size=(50, -1),
                                            min=0, max=1, initial=0)

        self.mesh_posy_sld = SpinSliderCtrl(panel2, "PosY: ", size=(50, -1),
                                            min=0, max=1, initial=0)

        # sizers
        sizer0 = wx.GridBagSizer(vgap=5, hgap=5)
        sizer0.Add(self.folder_txt, pos=(0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL)
        sizer0.Add(self.prefix_txt, pos=(1, 0), span=(1, 1), flag=wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL)
        sizer0.Add(self.digits_spn, pos=(2, 0), span=(1, 1), flag=wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL)
        sizer0.Add(self.index0_spn, pos=(2, 1), span=(1, 1), flag=wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL)
        sizer0.Add(self.firstimage_fp, pos=(3, 1), span=(1, 1), flag=wx.ALIGN_RIGHT)

        sizer1 = wx.GridBagSizer(vgap=5, hgap=5)
        sizer1.Add(self.line_size_spn, pos=(0, 0), span=(1, 1), flag=wx.LEFT | wx.TOP, border=5)
        sizer1.Add(self.line_skip_spn, pos=(0, 1), span=(1, 1), flag=wx.RIGHT | wx.TOP, border=5)
        sizer1.Add(self.line_pos_sld, pos=(1, 0), span=(1, 2),
                   flag=wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, border=5)

        sizer2 = wx.GridBagSizer(vgap=5, hgap=5)
        sizer2.Add(self.mesh_sizex_spn, pos=(0, 0), span=(1, 1), flag=wx.LEFT | wx.TOP, border=5)
        sizer2.Add(self.mesh_sizey_spn, pos=(0, 1), span=(1, 1), flag=wx.RIGHT | wx.TOP, border=5)
        sizer2.Add(self.mesh_skip_spn, pos=(1, 0), span=(1, 1), flag=wx.LEFT, border=5)
        sizer2.Add(self.mesh_line_ckb, pos=(1, 1), span=(1, 1), flag=wx.RIGHT, border=5)
        sizer2.Add(self.mesh_posx_sld, pos=(2, 0), span=(1, 2), flag=wx.RIGHT | wx.LEFT | wx.EXPAND, border=5)
        sizer2.Add(self.mesh_posy_sld, pos=(3, 0), span=(1, 2),
                   flag=wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, border=5)

        for i in range(2):
            sizer0.AddGrowableCol(i)
            sizer1.AddGrowableCol(i)
            sizer2.AddGrowableCol(i)

        panel1.SetSizer(sizer1)
        sizer1.Layout()
        panel1.Fit()

        panel2.SetSizer(sizer2)
        sizer2.Layout()
        panel2.Fit()

        # assemble
        self.notebook.AddPage(panel1, "Line", False)
        self.notebook.AddPage(panel2, "Mesh", False)

        # self.Add(sizer0,  0, wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        # self.Add(self.series_ckb, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL | wx.ALIGN_LEFT, 10)
        # self.Add(self.notebook, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.Add(sizer0,  0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.Add(self.series_ckb, 0,  wx.ALL | wx.ALIGN_LEFT, 10)
        self.Add(self.notebook, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # bindings
        self.series_ckb.Bind(wx.EVT_CHECKBOX, self.OnModifySeries)
        self.folder_txt.Bind_to(self.OnEditFolder)
        self.prefix_txt.Bind_to(self.OnEditPrefix)
        self.index0_spn.Bind_to(self.OnEditIndex)
        self.digits_spn.Bind_to(self.OnEditDigits)
        self.firstimage_fp.Bind_to(self.OnPickFile)

        self.line_size_spn.Bind_to(self.OnSetLineSize)
        self.line_skip_spn.Bind_to(self.OnModifyAnything)
        self.line_pos_sld.Bind_to(self.OnMoveSlider)

        self.mesh_sizex_spn.Bind_to(self.OnSetMeshSizeX)
        self.mesh_sizey_spn.Bind_to(self.OnSetMeshSizeY)
        self.mesh_skip_spn.Bind_to(self.OnModifyAnything)
        self.mesh_line_ckb.Bind(wx.EVT_CHECKBOX, self.OnToggleSubfolder)
        self.mesh_posx_sld.Bind_to(self.OnMoveSlider)
        self.mesh_posy_sld.Bind_to(self.OnMoveSlider)

        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnModifySeries)

    def Init(self):
        self.SetType()
        self.SetDataExtensionFromDetector("sCMOS")

    def ResetPosition(self):
        self.mesh_posx_sld.SetValue(0)
        self.mesh_posy_sld.SetValue(0)
        self.line_pos_sld.SetValue(0)

    # Getters
    def GetValue(self):

        if self.GetType() == "mesh":
            skip = self.mesh_skip_spn.GetValue()
            scansize = self.mesh_sizex_spn.GetValue(), self.mesh_sizey_spn.GetValue()
        elif self.GetType() == "line":
            skip = self.mesh_skip_spn.GetValue()
            scansize = self.line_size_spn.GetValue()
        else:
            skip = 0
            scansize = 0

        return self.GetType(), self.img_folder, self.img_prefix, self.img_index0, \
               self.img_digits, skip, scansize, self.mesh_line_ckb.IsChecked()

    def GetType(self):
        if self.series_ckb.IsChecked():
            scantype = ["line", "mesh"][self.notebook.GetSelection()]
        else:
            scantype = "point"

        return scantype

    def GetPosition(self):
        if self.GetType() == "mesh":
            pos = self.mesh_posx_sld.GetValue(), self.mesh_posy_sld.GetValue()
        elif self.GetType() == "line":
            pos = (self.line_pos_sld.GetValue(),)
        else:
            pos = 0
        return pos

    # Setters
    def SetValue(self, scantype, folder, prefix, index0, ndigits, skip, setsize, subFolder):

        self.SetType(scantype)

        self.SetDataTree(folder, prefix, index0, ndigits)

        self.SetDataSize(skip, setsize, subFolder)

    def SetValueFromFile(self, filepath):

        folder, filename = os.path.split(filepath)
        pref, idx, _ = rimg.split_filename(filename)
        # folder, fname = os.path.split(filepath)
        # pref, remaining = fname.rsplit('_', 1)
        # pref += '_'
        # idx = remaining.split('.')[0]

        self.SetDataTree(folder, pref, int(idx), len(idx))

        self.SetTypeAndSizeFromFile(filepath)

    def SetType(self, scantype=None):
        if scantype is None:
            scantype = self.GetType()

        self.ResetPosition()

        if scantype == "mesh":
            self.series_ckb.SetValue(1)
            self.notebook.Enable(1)
            self.notebook.ChangeSelection(1)
        elif scantype == "line":
            self.series_ckb.SetValue(1)
            self.notebook.Enable(1)
            self.notebook.ChangeSelection(0)
        else:
            self.series_ckb.SetValue(0)
            self.notebook.Enable(0)

    def SetTypeAndSizeFromFile(self, filepath):

        if rimg.split_linesubfolder(os.path.dirname(filepath)) is not None:
            self.SetType("mesh")
            self.SetDataSize(0, (1, 1), True)

        else:
            pass

    def SetDataTree(self, folder, prefix, index0, ndigits):

        self.folder_txt.SetValue(folder)
        self.prefix_txt.SetValue(prefix)
        self.index0_spn.SetValue(index0)
        self.digits_spn.SetValue(ndigits)

        self.firstimage_fp.SetDefaultDir(folder)

        self.img_folder = folder
        self.img_prefix = prefix
        self.img_index0 = index0
        self.img_digits = ndigits

    def SetDataSize(self, skip=0, setsize=None, linesub=True):

        if setsize is None:
            setsize = (1, 1) if self.GetType() == "mesh" else 1

        if self.GetType() == "mesh":
            self.mesh_skip_spn.SetValue(skip)
            self.mesh_line_ckb.SetValue(linesub)
            self.mesh_sizex_spn.SetValue(setsize[0])
            self.mesh_sizey_spn.SetValue(setsize[1])
            self.mesh_sizex_spn.ForceNotify()
            self.mesh_sizey_spn.ForceNotify()

        elif self.GetType() == "line":
            self.line_skip_spn.SetValue(skip)
            self.line_size_spn.SetValue(setsize)
            self.line_size_spn.ForceNotify()

        else:
            pass

    def SetDataExtensionFromDetector(self, dtt_label):

        self.firstimage_fp.SetWildCard(rmccd.getwildcardstring(dtt_label))

        self.img_ext = rimg.ccd_to_extension(dtt_label)

    # Event handlers
    def OnEditFolder(self, event):

        folder = event.GetValue()
        pref = self.img_prefix
        idx = self.img_index0
        ndigits = self.img_digits

        if rimg.test_filename(pref, idx, extension=self.img_ext, ndigits=ndigits, folder=folder):
            self.img_folder = folder
            self.Notify(EVT_DATA_MODIFY)

        else:
            self.folder_txt.SetValue(self.img_folder)

        self.firstimage_fp.SetDefaultDir(self.img_folder)

    def OnEditPrefix(self, event):

        folder = self.img_folder
        pref = event.GetValue()
        idx = self.img_index0
        ndigits = self.img_digits

        if rimg.test_filename(pref, idx, extension=self.img_ext, ndigits=ndigits, folder=folder):
            self.img_prefix = pref
            self.Notify(EVT_DATA_MODIFY)

        else:
            self.prefix_txt.SetValue(self.img_prefix)

    def OnEditIndex(self, event):

        folder = self.img_folder
        pref = self.img_prefix
        idx = event.GetValue()
        ndigits = self.img_digits

        if rimg.test_filename(pref, idx, extension=self.img_ext, ndigits=ndigits, folder=folder):
            self.img_index0 = idx
            self.Notify(EVT_DATA_MODIFY)

        else:
            self.index0_spn.SetValue(self.img_index0)

    def OnEditDigits(self, event):

        folder = self.img_folder
        pref = self.img_prefix
        idx = self.img_index0
        ndigits = event.GetValue()

        if rimg.test_filename(pref, idx, extension=self.img_ext, ndigits=ndigits, folder=folder):
            self.img_digits = ndigits
            self.Notify(EVT_DATA_MODIFY)

        else:
            self.digits_spn.SetValue(self.img_digits)

    def OnPickFile(self, event):

        self.SetValueFromFile(event.GetString())

        self.Notify(EVT_DATA_MODIFY)

    def OnModifyAnything(self, event):
        self.Notify(EVT_DATA_MODIFY)

    def OnModifySeries(self, event):
        self.SetType()
        self.Notify(EVT_DATA_MODIFY)

    def OnSetLineSize(self, event):
        self.line_pos_sld.SetMaxValue(event.GetValue()-1)
        self.Notify(EVT_DATA_MODIFY)

    def OnSetMeshSizeX(self, event):
        self.mesh_posx_sld.SetMaxValue(event.GetValue()-1)
        self.Notify(EVT_DATA_MODIFY)

    def OnSetMeshSizeY(self, event):
        self.mesh_posy_sld.SetMaxValue(event.GetValue()-1)
        self.Notify(EVT_DATA_MODIFY)

    def OnMoveSlider(self, event):
        self.Notify(EVT_DATA_MOVE)

    def OnToggleSubfolder(self, event):
        if rimg.split_linesubfolder(self.img_folder) is not None:
            self.Notify(EVT_DATA_MODIFY)
        else:
            self.mesh_line_ckb.SetValue(0)

    # Binders
    def Bind_to(self, event, callback):

        self._observers[event].append(callback)

    def Notify(self, event):

        for callback in self._observers[event]:
            callback(self)


# Test
if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = InputData(frame)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
