#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


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

import LaueTools.Daxm.classes.source as mysrc

import LaueTools.Daxm.material.dict_datamat as dm

INP_MODE_NORMAL = 0
INP_MODE_CUSTOM = 1


class InputSample(wx.StaticBoxSizer):

    # Constructors
    def __init__(self, parent):
        mainbox = wx.StaticBox(parent, wx.ID_ANY, "Sample")

        wx.StaticBoxSizer.__init__(self, mainbox, wx.VERTICAL)

        mainbox.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        mainbox.SetForegroundColour('DIM GREY')

        self._box = mainbox
        self._parent = parent
        self._observers = []

        self.notebook = None
        self.mat_cbx = None
        self.incl_spn = None
        self.thick_spn = None
        self.edit_btn = None
        self.qty_spn = None
        self.incl2_spn = None
        self.compo_dv = None
        self.abs_btn = None
        self.src_btn = None
        self.fluo_btn = None

        self.Create()
        self.Init()

    def Create(self):
        # Notebook
        self.notebook = wx.Notebook(self._parent)

        # regular
        panel1 = wx.Panel(self.notebook)

        self.mat_cbx = LabelComboBox(panel1, label="Mat: ", style=wx.CB_SORT,
                                     value="Ge", choices=mysrc.list_available_all())

        self.thick_spn = LabelSpinCtrlDouble(panel1, label="Thick (mm): ",
                                             initial=0.3, min=0, max=10, inc=0.001, size=(50, -1))

        self.incl_spn = LabelSpinCtrl(panel1, label="Incl (°): ",
                                      initial=40, min=0, max=90, size=(50, -1))

        self.edit_btn = wx.Button(panel1, id=wx.ID_ANY, label=" Edit ")
        self.edit_btn.SetBitmap(mycons.get_icon_bmp("icon_tune.png"), wx.LEFT)

        grid1 = wx.GridSizer(rows=2, cols=2, hgap=5, vgap=5)

        grid1.Add(self.mat_cbx, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 10)
        grid1.Add(self.thick_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 10)
        grid1.Add(self.incl_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid1.Add(self.edit_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM | wx.RIGHT, 5)

        panel1.SetSizer(grid1)
        grid1.Layout()
        panel1.Fit()

        self.notebook.AddPage(panel1, "Basic", True)

        # user-def
        panel2 = wx.Panel(self.notebook)

        self.qty_spn = LabelSpinCtrl(panel2, "Qty: ", min=1, max=100, initial=1, size=(60, -1))

        self.incl2_spn = LabelSpinCtrl(panel2, label="Incl (°): ",
                                       initial=40, min=0, max=90, size=(60, -1))

        self.compo_dv = WireList(panel2)

        grid2 = wx.GridSizer(rows=1, cols=2, hgap=5, vgap=5)
        grid2.Add(self.qty_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
        grid2.Add(self.incl2_spn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(10)
        vbox.Add(grid2, 0, wx.EXPAND)
        vbox.AddSpacer(10)
        vbox.Add(self.compo_dv, 1, wx.EXPAND)
        vbox.AddSpacer(5)

        panel2.SetSizer(vbox)
        vbox.Layout()
        panel2.Fit()

        self.notebook.AddPage(panel2, "Customized", False)

        # buttons
        hbox = wx.GridSizer(rows=1, cols=3, hgap=10, vgap=0)

        self.abs_btn = wx.Button(self._parent, id=wx.ID_ANY, label=" Absorption")
        self.abs_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        self.src_btn = wx.Button(self._parent, id=wx.ID_ANY, label="   Source  ")
        self.src_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        self.fluo_btn = wx.Button(self._parent, id=wx.ID_ANY, label="    Fluo.  ")
        self.fluo_btn.SetBitmap(mycons.get_icon_bmp("icon_graph.png"), wx.LEFT)

        hbox.Add(self.abs_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        hbox.Add(self.src_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
        hbox.Add(self.fluo_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)

        # assemble
        # self.Add(self.notebook, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND | wx.ALL, 5)
        # self.Add(hbox, 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 5)
        
        self.Add(self.notebook, 0, wx.EXPAND | wx.ALL, 5)
        self.Add(hbox, 0,wx.EXPAND | wx.ALL, 5)

        # bindings
        self.mat_cbx.Bind_to(self.OnModifyAnything)
        self.thick_spn.Bind_to(self.OnModifyAnything)
        self.incl_spn.Bind_to(self.OnModifyAnything)
        self.incl2_spn.Bind_to(self.OnModifyAnything)

        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnModifyAnything)
        self.edit_btn.Bind(wx.EVT_BUTTON, self.OnPressEdit)

        self.qty_spn.Bind_to(self.OnModifyQty)
        self.compo_dv.Bind(olv.EVT_CELL_EDIT_FINISHED, self.OnModifyAnything)

        self.abs_btn.Bind(wx.EVT_BUTTON, self.OnAbsButton)
        self.src_btn.Bind(wx.EVT_BUTTON, self.OnSourceButton)
        self.fluo_btn.Bind(wx.EVT_BUTTON, self.OnFluoButton)

    def Init(self):
        self.compo_dv.SetQty(1)

    # Getters
    def GetValue(self):
        if self.GetMode() == INP_MODE_CUSTOM:
            compo = [obj.ToArray() for obj in self.compo_dv.GetObjects()]
            incl = float(self.incl2_spn.GetValue())
        else:
            compo = mysrc.get_compo_from_mat(self.mat_cbx.GetValue(),
                                             self.thick_spn.GetValue(),
                                             detail=False)
            incl = float(self.incl_spn.GetValue())

        return compo, incl

    def GetCompoInclined(self):

        compo, incl = self.GetValue()
        for elt in compo:
            elt[1] = [yi / np.sin(np.radians(incl)) for yi in elt[1]]

        return compo

    def GetMode(self):
        return self.notebook.GetSelection()

    # Setters
    def SetValue(self, compo, incl):

        tmp = []
        for elt in compo:
            if isinstance(elt, str):
                newelt = mysrc.get_compo_from_mat(elt)
            elif hasattr(elt, '__len__') and len(elt) == 3:
                newelt = elt
            else:
                newelt = mysrc.get_compo_from_mat("Ge")

            tmp.append(newelt)

        compo = tmp

        mat = [elt[0] for elt in compo]
        y0 = [elt[1][0] for elt in compo]
        y1 = [elt[1][1] for elt in compo]
        d = [elt[2] for elt in compo]

        if len(compo) == 1 and compo[0][1][0] == 0:
            mode = INP_MODE_NORMAL
        elif allequal_str(mat) \
                and allequal_float(y0, 1E-6) \
                and allequal_float(y1, 1E-6) \
                and allequal_float(d, 1E-6):
            mode = INP_MODE_NORMAL
        else:
            mode = INP_MODE_CUSTOM

        self.SetMode(mode)

        if mode == INP_MODE_CUSTOM:

            self.compo_dv.Clear()
            for elt in compo:
                self.compo_dv.Append(elt)

            self.qty_spn.SetValue(len(compo))
            self.incl2_spn.SetValue(incl)

        else:
            self.mat_cbx.SetValue(compo[0][0])
            self.thick_spn.SetValue(compo[0][1][1])
            self.incl_spn.SetValue(incl)

    def SetMode(self, mode):
        self.notebook.ChangeSelection(mode)

    # Event handlers
    def OnModifyAnything(self, event):
        self.Notify()

    def OnModifyQty(self, event):
        self.compo_dv.SetQty(event.GetValue())
        self.Notify()

    def OnPressEdit(self, event):
        self.SetMode(INP_MODE_CUSTOM)

        self.compo_dv.Clear()

        compo = mysrc.get_compo_from_mat(self.mat_cbx.GetValue(),
                                         self.thick_spn.GetValue())

        for elt in compo:
            self.compo_dv.Append(elt)

        self.incl2_spn.SetValue(self.incl_spn.GetValue())

        self.Notify()

    def OnAbsButton(self, event):

        src = mysrc.SecondarySource(arg=self.GetValue()[0])

        src.plot_absorption()

    def OnSourceButton(self, event):

        src = mysrc.SecondarySource(arg=self.GetCompoInclined(), ystep=0.0005)

        src.plot_source()

    def OnFluoButton(self, event):

        src = mysrc.SecondarySource(arg=self.GetCompoInclined(), ystep=0.0005)

        src.plot_source_fluo()

    # Binders
    def Bind_to(self, callback):
        self._observers.append(callback)

    def Notify(self):
        # print self.GetValue()
        for callback in self._observers:
            callback(self)


# Table for customized sample composition
class WireList(olv.ObjectListView):

    def __init__(self, parent, size=(-1, -1)):

        olv.ObjectListView.__init__(self, parent, wx.ID_ANY, size=size, sortable=False,
                                    style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE,
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
            olv.ColumnDefn("mat", "centre", -1, "material", isSpaceFilling=True,
                           cellEditorCreator=MakeMaterialEditor),
            olv.ColumnDefn("y0 (mm)", "centre", -1, "y0", isSpaceFilling=True, stringConverter="%f",
                           valueSetter=Component.SetY0),
            olv.ColumnDefn("y1 (mm)", "centre", -1, "y1", isSpaceFilling=True, stringConverter="%f",
                           valueSetter=Component.SetY1),
            olv.ColumnDefn("density", "centre", -1, "density", isSpaceFilling=True, stringConverter="%f",
                           valueSetter=Component.SetDensity)
        ])

    def Append(self, item):
        self.AddObject(Component(*item))
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
                self.Append(["Ge", [0, 0.3], 0])
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


def MakeMaterialEditor(parent, rowIndex, subItemIndex):
    odcb = MaterialEditor(parent)
    return odcb


class MaterialEditor(wx.Choice):
    # The user may not put another value than those in the choices list
    def __init__(self, parent):
        wx.Choice.__init__(self, parent, -1, style=wx.CB_SORT,
                           choices=mysrc.list_available_all())
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
        # evt.skip()
        self.parent.FinishCellEdit()


class Component(object):

    def __init__(self, material, bounds=None, density=0):

        if bounds is None:
            bounds = [0, 0.3]

        self.material = material
        self.bounds = None
        self.y0 = None
        self.y1 = None
        self.density = None

        self.SetBounds(bounds)
        self.SetDensity(density)

    def SetBounds(self, bounds):

        bounds = list(bounds)

        if bounds[0] < bounds[1]:
            self.bounds = bounds
            self.y0 = bounds[0]
            self.y1 = bounds[1]
        else:
            pass

    def SetY0(self, value):

        if value <= self.y1:
            self.y0 = value
            self.bounds[0] = value
        else:
            # TODO
            pass

    def SetY1(self, value):

        if value > self.y0:
            self.y1 = value
            self.bounds[1] = value
        else:
            # TODO
            pass

    def SetDensity(self, value):

        if value > 0:
            self.density = value
        elif self.material in dm.dict_density:
            self.density = dm.dict_density[self.material]
        elif self.material in dm.dict_mat:
            self.density = dm.dict_mat[self.material][2]
        else:
            # should not be possible with readonly material cells
            pass

    def ToArray(self):
        return [self.material, [self.y0, self.y1], self.density]


# Test
if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(parent=None, title="Test")

    box = InputSample(frame)
    box.SetValue(mysrc.get_compo_from_mat('316L', 0.1, detail=False), 40)

    frame.SetSizer(box)
    frame.Layout()
    box.Layout()
    box.Fit(frame)
    frame.Fit()

    frame.Show(True)
    app.MainLoop()
