#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import math
import numpy as np

import wx

import matplotlib as mpl
mpl.use('WXAgg')

import LaueTools.Daxm.contrib.ObjectListView as olv

from LaueTools.Daxm.utils.list import allequal_float, allequal_str

from LaueTools.Daxm.gui.widgets.combobox import LabelComboBox
from LaueTools.Daxm.gui.widgets.spin import LabelSpinCtrl
from LaueTools.Daxm.gui.widgets.spindouble import LabelSpinCtrlDouble

import LaueTools.Daxm.gui.icons.icon_manager as mycons

import LaueTools.Daxm.classes.wire as mywire

INP_MODE_NORMAL = 0
INP_MODE_CUSTOM = 1


class InputWire(wx.StaticBoxSizer):

    def __init__(self, parent):
        mainbox = wx.StaticBox(parent, wx.ID_ANY, "Wires")

        wx.StaticBoxSizer.__init__(self, mainbox, wx.VERTICAL)

        mainbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        mainbox.SetForegroundColour('DIM GREY')

        self._box = mainbox
        self._parent = parent
        self._observers = []

        self.notebook = None
        self.qty_spn = None
        self.radius_spn = None
        self.height_spn = None
        self.offset_spn = None
        self.space_spn = None
        self.incl_cbx = None
        self.traj_cbx = None
        self.trf_btn = None
        self.edit_btn = None
        self.qty2_spn = None
        self.traj2_cbx = None
        self.wire_dv = None
        self.trf2_btn = None
        self.mat_cbx = None

        self.Create()
        self.Init()
        
    def Create(self):
        # Notebook
        self.notebook = wx.Notebook(self._parent)

        # regular
        panel1 = wx.Panel(self.notebook)

        self.qty_spn = LabelSpinCtrl(panel1, "Qty: ", min=1, max=100, initial=1, size=(60, -1))

        self.mat_cbx = LabelComboBox(panel1, label="Mat: ", style=wx.CB_SORT,
                                     value="W", choices=mywire.list_available_material())

        self.radius_spn = LabelSpinCtrlDouble(panel1, "R (mm): ",
                                              initial=0.025, min=0, max=1, inc=0.001, size=(60, -1))

        self.height_spn = LabelSpinCtrlDouble(panel1, "h (mm): ",
                                              initial=1.0, min=0., max=60., inc=0.01, size=(60, -1))

        self.offset_spn = LabelSpinCtrlDouble(panel1, "p0 (mm): ",
                                              initial=0.0, min=-100, max=100, inc=0.01, size=(60, -1))

        self.traj_cbx = LabelComboBox(panel1, label=u"Traj (°): ",
                                      value="0", choices=["0", "40"], size=(60, -1))

        self.incl_cbx = LabelComboBox(panel1, label=u"Incl (°): ",
                                      value="0", choices=["0", "40"], size=(60, -1))

        self.space_spn = LabelSpinCtrlDouble(panel1, u"\u0394 (mm): ",
                                             initial=0.5, min=0, max=10, inc=0.01, size=(60, -1))

        self.trf_btn = wx.Button(panel1, id=wx.ID_ANY, label=" Transmission ")
        self.trf_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        self.edit_btn = wx.Button(panel1, id=wx.ID_EDIT, label="    Tune up    ")
        self.edit_btn.SetBitmap(mycons.get_icon_bmp("icon_tune.png"), wx.LEFT)

        grid1 = wx.GridSizer(rows=5, cols=2, hgap=5, vgap=5)

        grid1.Add(self.qty_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 10)
        grid1.Add(self.space_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 10)
        grid1.Add(self.incl_cbx, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.traj_cbx, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.mat_cbx, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.radius_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.height_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.offset_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.edit_btn, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM | wx.LEFT, 5)
        grid1.Add(self.trf_btn,  0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM | wx.RIGHT, 5)

        panel1.SetSizer(grid1)
        grid1.Layout()
        panel1.Fit()

        self.notebook.AddPage(panel1, "Basic", True)

        # user-def
        panel2 = wx.Panel(self.notebook)

        self.qty2_spn = LabelSpinCtrl(panel2, "Qty: ", min=1, max=100, initial=1, size=(60, -1))

        self.traj2_cbx = LabelComboBox(panel2, label=u"Traj (°): ",
                                       value="0", choices=["0", "40"], size=(60, -1))

        self.wire_dv = WireList(panel2)

        self.trf2_btn = wx.Button(panel2, id=wx.ID_ANY, label=" Transmission ")
        self.trf2_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        grid2 = wx.GridSizer(rows=1, cols=2, hgap=5, vgap=5)
        grid2.Add(self.qty2_spn, 0,  wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid2.Add(self.traj2_cbx, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)

        grid3 = wx.GridSizer(rows=1, cols=2, hgap=5, vgap=5)
        grid3.Add(wx.BoxSizer(), 0)
        grid3.Add(self.trf2_btn, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM | wx.RIGHT, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(10)
        vbox.Add(grid2, 0, wx.EXPAND)
        vbox.AddSpacer(10)
        vbox.Add(self.wire_dv, 0, wx.EXPAND)
        vbox.AddSpacer(10)
        vbox.Add(grid3, 0, wx.EXPAND)
        vbox.AddSpacer(5)

        panel2.SetSizer(vbox)
        vbox.Layout()
        panel2.Fit()

        self.notebook.AddPage(panel2, "Customized", False)

        # assemble
        # self.Add(self.notebook, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND | wx.ALL, 5)
        self.Add(self.notebook, 1,  wx.EXPAND | wx.ALL, 5)


        # bindings
        self.qty_spn.Bind_to(self.OnModifyQty)
        self.qty2_spn.Bind_to(self.OnModifyQty)

        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnModifyAnything)
        self.edit_btn.Bind(wx.EVT_BUTTON, self.OnPressTune)
        self.trf_btn.Bind(wx.EVT_BUTTON, self.OnPressTransmission)
        self.trf2_btn.Bind(wx.EVT_BUTTON, self.OnPressTransmission)

        self.mat_cbx.Bind_to(self.OnModifyAnything)
        self.radius_spn.Bind_to(self.OnModifyAnything)
        self.height_spn.Bind_to(self.OnModifyAnything)
        self.offset_spn.Bind_to(self.OnModifyAnything)
        self.incl_cbx.Bind_to(self.OnModifyAnything)
        self.space_spn.Bind_to(self.OnModifyAnything)
        self.traj_cbx.Bind_to(self.OnModifyAnything)
        self.traj2_cbx.Bind_to(self.OnModifyAnything)

        self.wire_dv.Bind(olv.EVT_CELL_EDIT_FINISHED, self.OnModifyAnything)

    def Init(self):
        self.SetQty(1)
        self.wire_dv.Append(["W", 0.025, 1.0, 0.0])

    # Getters
    def GetValue(self):

        if self.GetMode() == INP_MODE_CUSTOM:
            wires = [obj.ToArray() for obj in self.wire_dv.GetObjects()]
            traj = float(self.traj2_cbx.GetValue())
        else:
            wires = []
            h0 = self.height_spn.GetValue()
            p0 = self.offset_spn.GetValue()
            sinIncl = math.sin(math.radians(float(self.incl_cbx.GetValue())))
            cosIncl = math.cos(math.radians(float(self.incl_cbx.GetValue())))
            space = self.space_spn.GetValue()
            #TODO
            wires = mywire.gen_wires_grid(material=self.mat_cbx.GetValue(),qty=self.qty_spn.GetValue(),
                                          incl=self.incl_cbx.GetValue(),spacing=self.space_spn.GetValue(),
                                          traj=self.traj_cbx.GetValue(), radius=self.radius_spn.GetValue(),
                                          height=self.height_spn.GetValue(), offset=self.offset_spn.GetValue())
            #for i in range(self.qty_spn.GetValue()):
            #   wires.append([self.mat_cbx.GetValue(), self.radius_spn.GetValue(),
            #                         h0 + i * sinIncl * space, p0 - i * cosIncl * space])
            traj = float(self.traj_cbx.GetValue())

        return wires, traj

    def GetMode(self):
        return self.notebook.GetSelection()

    # Setters
    def SetValue(self, wires, traj):

        tmp = []
        for wire in wires:

            if isinstance(wire, str):
                par = mywire.new_dict(material=wire)
            elif isinstance(wire, dict):
                par = mywire.new_dict(**wire)
            elif hasattr(wire, '__len__') and len(wire)==4:
                par = mywire.new_dict(material=wire[0], R=wire[1], h=wire[2], p0=wire[3])
            else:
                par = mywire.new_dict()

            tmp.append([par['material'], par['R'], par['h'], par['p0']])

        wires = tmp

        mat = [wire[0] for wire in wires]
        rad = [wire[1] for wire in wires]
        hei = [wire[2] for wire in wires]
        off = [wire[3] for wire in wires]

        if len(wires) == 1:
            mode = INP_MODE_NORMAL
        elif allequal_str(mat) \
                and allequal_float(rad, 1E-6) \
                and allequal_float(hei, 1E-6) \
                and allequal_float(np.diff(off), 1E-6):
            mode = INP_MODE_NORMAL
        else:
            mode = INP_MODE_CUSTOM

        self.SetMode(mode)

        if mode == INP_MODE_CUSTOM:
            self.wire_dv.Clear()
            for wire in wires:
                self.wire_dv.Append(wire)
            self.SetQty(self.wire_dv.GetItemCount())
            self.traj2_cbx.SetValue(str(int(traj)))
        else:
            self.SetQty(len(mat))
            if len(mat) > 1:
                self.space_spn.SetValue(-np.diff(off)[0])
            else:
                self.space_spn.SetValue(0)
            self.incl_cbx.SetValue("0")
            self.traj_cbx.SetValue(str(int(traj)))
            self.mat_cbx.SetValue(mat[0])
            self.radius_spn.SetValue(rad[0])
            self.height_spn.SetValue(hei[0])
            self.offset_spn.SetValue(off[0])

    def SetQty(self, qty):
        if self.GetMode() == INP_MODE_CUSTOM:
            self.qty2_spn.SetValue(qty)
        else:
            self.qty_spn.SetValue(qty)
            if qty == 1:
                self.incl_cbx.Enable(False)
                self.space_spn.Enable(False)
            else:
                self.incl_cbx.Enable(True)
                self.space_spn.Enable(True)

    def SetMode(self, mode):
        self.notebook.ChangeSelection(mode)

    # Event handlers
    def OnModifyAnything(self, event):
        self.Notify()

    def OnModifyQty(self, event):
        if self.GetMode() == INP_MODE_CUSTOM:

            self.wire_dv.SetQty(event.GetValue())

        else:
            if event.GetValue() == 1:
                self.incl_cbx.Enable(False)
                self.space_spn.Enable(False)
            else:
                self.incl_cbx.Enable(True)
                self.space_spn.Enable(True)

        self.Notify()

    def OnPressTransmission(self, event):

        if self.GetMode() == INP_MODE_CUSTOM:
            obj = self.wire_dv.GetSelectedObject()
            tmp = mywire.CircularWire(**obj.ToDict())
            idx = self.wire_dv.GetIndexOf(obj)

            tmp.plot_transmission(label="wire #{} ({})".format(idx+1, tmp.get_material()))
        else:
            tmp = mywire.CircularWire(material=self.mat_cbx.GetValue(),
                                      R=self.radius_spn.GetValue())

            tmp.plot_transmission(label="{} wire".format(tmp.get_material()))

    def OnPressTune(self, event):
        #h0 = self.height_spn.GetValue()
        #p0 = self.offset_spn.GetValue()
        #sinIncl = math.sin(math.radians(float(self.incl_cbx.GetValue())))
        #cosIncl = math.cos(math.radians(float(self.incl_cbx.GetValue())))
        #space = self.space_spn.GetValue()
        #for i in range(self.qty_spn.GetValue()):
        #    self.wire_dv.Append([self.mat_cbx.GetValue(), self.radius_spn.GetValue(),
        #                         h0 + i*sinIncl*space, p0 - i*cosIncl*space])

        wires, traj = self.GetValue()
        qty = len(wires)

        self.SetMode(INP_MODE_CUSTOM)
        self.wire_dv.Clear()

        for wire in wires:
            self.wire_dv.Append(wire)
        self.qty2_spn.SetValue(qty)
        self.traj2_cbx.SetValue(str(int(traj)))

        self.Notify()

    # Binders
    def Bind_to(self, callback):
        
        self._observers.append(callback)
        
    def Notify(self):
        # print self.GetValue()
        for callback in self._observers:
            
            callback(self)

# End of main class


# Table for customized wire set
class WireList(olv.ObjectListView):

    def __init__(self, parent, size=(-1, -1)):

        olv.ObjectListView.__init__(self, parent, wx.ID_ANY, size=size, sortable=False,
                                    style= wx.LC_REPORT  | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE,
                                    cellEditMode=olv.ObjectListView.CELLEDIT_DOUBLECLICK)

        # /!\ Must initialize stuff because sortable=False
        if self.smallImageList is None:
            self.SetImageLists()
        if (not self.smallImageList.HasName(self.NAME_DOWN_IMAGE) and
                self.smallImageList.GetSize(0) == (16, 16)):
            self.RegisterSortIndicators()

        self.Create()

    def Create(self):

        self.SetColumns([
            olv.ColumnDefn("mat", "centre", -1, "material", isSpaceFilling=True, cellEditorCreator=MakeMaterialEditor),
            olv.ColumnDefn("R (mm)", "centre", -1, "radius", isSpaceFilling=True, stringConverter="%f",
                           valueSetter=Wire.SetRadius),
            olv.ColumnDefn("h (mm)", "centre", -1, "height", isSpaceFilling=True, stringConverter="%f",
                           valueSetter=Wire.SetHeight),
            olv.ColumnDefn("p0 (mm)", "centre", -1, "offset", isSpaceFilling=True, stringConverter="%f")
        ])

    def Append(self, item):
        self.AddObject(Wire(*item))
        self.SetSelectionRow(self.GetItemCount()-1)

    def Pop(self):
        self.RemoveObject(self.GetObjectAt(self.GetItemCount()-1))

    def Clear(self):

        self.SetObjects([])

    def SetQty(self, qty):

        if qty > self.GetItemCount():
            if self.GetItemCount():
                self.Append(self.GetObjectAt(self.GetItemCount()-1).ToArray())
            else:
                self.Append(["W", 0.025, 1.0, 0.0])
            self.SetQty(qty)
        elif qty < self.GetItemCount():
            self.Pop()
            self.SetQty(qty)
        else:
            pass

    def GetSelectionRow(self):

        item = self.GetSelectedObject()

        return self.GetIndexOf(item)

    def SetSelectionRow(self, sel):

        item = self.GetObjectAt(sel)

        self.SelectObject(item, ensureVisible=True)

    def UnselectAll(self):

        self.DeselectAll()

    def Enable(self, status=True):

        olv.ObjectListView.Enable(self, status)

        if status:
            self.SetEmptyListMsg("This list is empty")
            self.stEmptyListMsg.Hide()
        else:
            self.SetEmptyListMsg("This list is disabled")
            self.stEmptyListMsg.Show()

def MakeMaterialEditor(olv, rowIndex, subItemIndex):
    odcb = MaterialEditor(olv)
    return odcb

class MaterialEditor(wx.Choice):
# The user may not put another value than those in the choices list
    def __init__(self, parent):
        wx.Choice.__init__(self, parent, -1, style=wx.CB_SORT,
                           choices=mywire.list_available_material(),)
        self.parent = parent
        self.Bind(wx.EVT_CHOICE, self._ForceFinishCellEdit)

    def SetValue(self, value):
        # Sets the value only if in the choices list
        wx.Choice.SetSelection(self, wx.Choice.FindString(self, value))

    def GetValue(self):
        # Get the value from the editor
        strValue = wx.Choice.GetString(self, wx.Choice.GetSelection(self))
        return strValue

    def _ForceFinishCellEdit(self, evt):
        # required because OLV does not properly end cell edition
        # evt.Skip()
        self.parent.FinishCellEdit()

class Wire(object):

    def __init__(self, material, radius, height, offset):
        self.material = material
        self.radius = radius
        self.height = height
        self.offset = offset

    def SetRadius(self, value):

        if value > 0:
            self.radius = value
        else:
            pass

    def SetHeight(self, value):

        if value > 0:
            self.height = value
        else:
            pass

    def ToArray(self):
        return [self.material, self.radius, self.height, self.offset]

    def ToDict(self):
        return {"material":self.material,
                "   R": self.radius,
                "h": self.height,
                "p0": self.offset}
# End of table


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = InputWire(frame)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
