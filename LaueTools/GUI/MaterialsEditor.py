# -*- coding: utf-8 -*-
r"""
GUI widgets for the materials library of LaueTools (dict_LaueTools.dict_Materials and materials.yaml):

- MaterialComboBox: drop down menu of materials whose list shrinks to the materials
  starting with the typed letters
- MaterialsEditorFrame: table of the materials (sortable by any column) and form to add a material

This module belongs to the open source LaueTools project
JS Micha   October 2026
"""
import sys

import wx

if sys.version_info.major == 3:
    from .. import dict_LaueTools as DictLT
else:
    import dict_LaueTools as DictLT

COLUMNS = ["name", "a", "b", "c", "alpha", "beta", "gamma", "extinction"]
LATTICE_LABELS = COLUMNS[1:7]


def material_as_list(dict_materials, key_material):
    """return [label, [a, b, c, alpha, beta, gamma], extinction] of a material
    of dict_materials (dict or DictLT.Materials object)"""
    material = dict_materials[key_material]
    if isinstance(material, dict):
        return [key_material, material['lattice'], material['extinction']]
    return [key_material, material[1], material[2]]


def sorted_materials(labels):
    """sort materials labels by alphabetical order whatever the case"""
    return sorted(labels, key=lambda label: (str(label).lower(), str(label)))


class MaterialComboBox(wx.ComboBox):
    """ComboBox of materials labels in which the user can type the first letters of a label:
    the list shrinks to the materials starting with these letters (whatever the case).

    Enter key selects the typed material (or the first one of the shrunk list).
    GetValue() always returns a material of the list: the last one selected as long as
    the typed text is not a material label.

    Handlers of wx.EVT_COMBOBOX must read the material with GetValue() (or event.GetString())
    and not with event.GetSelection(), which is the index in the shrunk list.
    """
    def __init__(self, parent, id=-1, value="", pos=wx.DefaultPosition, size=wx.DefaultSize,
                                                                    choices=(), style=0, **kwargs):
        style = (style & ~wx.CB_READONLY) | wx.TE_PROCESS_ENTER
        self._allitems = list(choices)
        self._lastvalid = value if value in self._allitems else ""
        self._busy = False
        self._popupshown = False
        self._previoustext = value
        wx.ComboBox.__init__(self, parent, id, value, pos, size, self._allitems, style, **kwargs)
        self._setautocomplete()

        self.Bind(wx.EVT_TEXT, self._ontext)
        self.Bind(wx.EVT_TEXT_ENTER, self._onenter)
        self.Bind(wx.EVT_COMBOBOX, self._onselect)
        self.Bind(wx.EVT_COMBOBOX_DROPDOWN, self._ondropdown)
        self.Bind(wx.EVT_COMBOBOX_CLOSEUP, self._oncloseup)
        self.Bind(wx.EVT_KILL_FOCUS, self._onkillfocus)

    def Bind(self, event, handler, *args, **kwargs):
        # selected material is recorded even if the handler of the selection does not call event.Skip()
        if event is wx.EVT_COMBOBOX and handler is not None and handler != self._onselect:
            userhandler = handler
            def handler(evt):
                self._recordselection(evt)
                userhandler(evt)
        wx.ComboBox.Bind(self, event, handler, *args, **kwargs)

    # --- same methods as wx.ComboBox, applied to the full list of materials
    def GetValue(self):
        text = wx.ComboBox.GetValue(self)
        if text in self._allitems or not self._lastvalid:
            return text
        return self._lastvalid

    def SetValue(self, value):
        self._busy = True
        try:
            if value in self._allitems:
                self._lastvalid = value
            self._showitems(self._allitems, value)
        finally:
            self._busy = False

    def GetItems(self):
        return list(self._allitems)

    def FindString(self, string, caseSensitive=False):
        for index, item in enumerate(self._allitems):
            if item == string or (not caseSensitive and item.lower() == string.lower()):
                return index
        return wx.NOT_FOUND

    def Clear(self):
        self._allitems = []
        self._lastvalid = ""
        self._busy = True
        try:
            wx.ComboBox.Clear(self)
        finally:
            self._busy = False

    def Append(self, items, *args):
        if isinstance(items, str):
            items = [items]
        self.SetItems(self._allitems + [item for item in items if item not in self._allitems])

    AppendItems = Append

    def SetItems(self, items):
        """set the full list of materials (the current material is kept if still in the list)"""
        text = wx.ComboBox.GetValue(self)
        self._allitems = list(items)
        if self._lastvalid not in self._allitems:
            self._lastvalid = ""
        self._busy = True
        try:
            self._showitems(self._allitems, text)
        finally:
            self._busy = False
        self._setautocomplete()

    def Set(self, items):
        self.SetItems(items)

    # --- shrinking of the list
    def _setautocomplete(self):
        """list of the materials starting with the typed letters, shown while typing"""
        try:
            self.AutoComplete(self._allitems)
        except Exception:  # not available on all platforms
            pass

    def _showitems(self, items, text):
        """set the items of the drop down list, with text in the text field"""
        if list(wx.ComboBox.GetItems(self)) != list(items):
            wx.ComboBox.Set(self, items)  # (wx.ComboBox.SetItems calls self.Set)
        if text in items:
            wx.ComboBox.SetSelection(self, items.index(text))
        if wx.ComboBox.GetValue(self) != text:
            self.ChangeValue(text)
        self._previoustext = text

    def _matching(self, text):
        return [item for item in self._allitems if item.lower().startswith(text.lower())]

    def _ontext(self, event):
        event.Skip()
        if not self._busy:
            wx.CallAfter(self._shrink)

    def _shrink(self):
        if not self or self._busy or self._popupshown:
            return
        text = wx.ComboBox.GetValue(self)
        previoustext, self._previoustext = self._previoustext, text
        # material picked in the list shown while typing (several letters added at once): it is selected
        if text in self._allitems and text != self._lastvalid and abs(len(text) - len(previoustext)) > 1:
            self._validate(pickfirst=False)
            return
        # full list when nothing is typed: empty field or material already selected
        if not text or text == self._lastvalid:
            items = self._allitems
        else:
            items = self._matching(text)
        if list(wx.ComboBox.GetItems(self)) == items:
            return
        self._busy = True
        try:
            position = self.GetInsertionPoint()
            self._showitems(items, text)
            self.SetInsertionPoint(position)
        finally:
            self._busy = False

    def _select(self, key_material, notify):
        """set key_material as the selected one, with the full list in the drop down list"""
        changed = key_material != self._lastvalid
        self.SetValue(key_material)
        self.SetInsertionPointEnd()
        if changed and notify:
            event = wx.CommandEvent(wx.wxEVT_COMBOBOX, self.GetId())
            event.SetEventObject(self)
            event.SetInt(self._allitems.index(key_material))
            event.SetString(key_material)
            self.GetEventHandler().ProcessEvent(event)

    def _validate(self, pickfirst):
        """select the typed material, or else go back to the last selected one

        :param pickfirst: True to select the first material starting with the typed letters
        """
        text = wx.ComboBox.GetValue(self).strip()
        key_material = self._lastvalid
        if text in self._allitems:
            key_material = text
        elif text:
            samelabels = [item for item in self._allitems if item.lower() == text.lower()]
            if samelabels:
                key_material = samelabels[0]
            elif pickfirst and self._matching(text):
                key_material = self._matching(text)[0]
        if key_material:
            self._select(key_material, notify=True)

    def _onselect(self, event):
        """material selected in the drop down list"""
        event.Skip()
        self._recordselection(event)

    def _recordselection(self, event):
        key_material = event.GetString() or wx.ComboBox.GetValue(self)
        if key_material in self._allitems:
            self._lastvalid = key_material
            wx.CallAfter(self._restore)

    def _restore(self):
        if self and not self._popupshown:
            self._select(self._lastvalid, notify=False)

    def _onenter(self, event):
        self._validate(pickfirst=True)
        event.Skip()

    def _ondropdown(self, event):
        self._popupshown = True
        event.Skip()

    def _oncloseup(self, event):
        self._popupshown = False
        event.Skip()

    def _onkillfocus(self, event):
        event.Skip()
        wx.CallAfter(self._validateiffocuslost)

    def _validateiffocuslost(self):
        # focus goes to the drop down list when it opens: the shrunk list must be kept
        if not self or self._popupshown or self.HasFocus():
            return
        self._validate(pickfirst=False)


class MaterialsRefreshButton(wx.Button):
    """Button updating the list of a MaterialComboBox with the very last materials: the materials file
    is read again if it has been modified by an other program (e.g. lauetools-materials editor).
    The selected material is kept.

    :param combo: MaterialComboBox to update
    :param getmaterials: function without argument returning the materials (dict or DictLT.Materials
        object) of the board
    :param sortkey: key to sort the labels of the materials in the list
    :param onrefresh: function called with the new list of materials labels
    """
    def __init__(self, parent, combo, getmaterials, sortkey=None, onrefresh=None, label="Refresh",
                                                                                        **kwargs):
        wx.Button.__init__(self, parent, -1, label, **kwargs)
        self.combo = combo
        self.getmaterials = getmaterials
        self.sortkey = sortkey
        self.onrefresh = onrefresh
        self.SetToolTip("Refresh the list of materials: read again the materials file "
                        "if materials have been added or modified (e.g. with the Materials Editor)")
        self.Bind(wx.EVT_BUTTON, self.OnRefresh)

    def OnRefresh(self, _=None):
        dict_materials = self.getmaterials()
        DictLT.reload_materials(dict_materials)
        labels = sorted(dict_materials.keys(), key=self.sortkey)
        self.combo.SetItems(labels)
        if labels and self.combo.GetValue() not in labels:
            self.combo.SetValue(labels[0])
        if self.onrefresh is not None:
            self.onrefresh(labels)


class MaterialsEditorFrame(wx.Frame):
    """Table of the materials (click on a column title to sort the materials by increasing order,
    click again for decreasing order) and form to add, update or delete a material.

    :param dict_materials: materials to display and modify: dict {label: [label, lattice, extinction]}
        or DictLT.Materials object. Default: DictLT.dict_Materials, used by all LaueTools modules.
    :param onchange: function called without argument when a material is added, updated or deleted

    Materials are saved in the materials.yaml file (in the user's folder ~/.lauetools if the file
    installed with LaueTools cannot be modified).
    """
    def __init__(self, parent=None, _id=-1, title="LaueTools Materials Editor", dict_materials=None,
                                                                                    onchange=None):
        wx.Frame.__init__(self, parent, _id, title, size=(900, 650))

        self.dict_materials = DictLT.dict_Materials if dict_materials is None else dict_materials
        self.onchange = onchange
        self.library = None  # DictLT.Materials object where materials are saved (see getlibrary())
        self.sortcolumn = 0
        self.ascending = True
        self.rows = []

        panel = wx.Panel(self)

        self.infotxt = wx.StaticText(panel, -1, "")

        filtertxt = wx.StaticText(panel, -1, "Name starting with: ")
        self.filterctrl = wx.TextCtrl(panel, -1, "", size=(150, -1))
        self.filterctrl.Bind(wx.EVT_TEXT, self.OnFilter)
        self.counttxt = wx.StaticText(panel, -1, "")

        self.listctrl = wx.ListCtrl(panel, -1, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_HRULES)
        for col, name in enumerate(COLUMNS):
            self.listctrl.InsertColumn(col, name, wx.LIST_FORMAT_LEFT if col in (0, 7) else wx.LIST_FORMAT_RIGHT,
                                       width={0: 220, 7: 150}.get(col, 68))
        self.listctrl.Bind(wx.EVT_LIST_COL_CLICK, self.OnSort)
        self.listctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelectMaterial)

        # --- form
        box = wx.StaticBox(panel, -1, "Add or update a material")
        self.namectrl = wx.TextCtrl(panel, -1, "", size=(190, -1))
        self.latticectrls = [wx.TextCtrl(panel, -1, "", size=(80, -1)) for _ in LATTICE_LABELS]
        extinctions = sorted(set(DictLT.dict_Extinc.values()), key=str.lower)
        self.extinctionctrl = wx.ComboBox(panel, -1, "no", choices=extinctions, size=(180, -1))

        addbtn = wx.Button(panel, -1, "Add / Update")
        addbtn.Bind(wx.EVT_BUTTON, self.OnAddMaterial)
        deletebtn = wx.Button(panel, -1, "Delete")
        deletebtn.Bind(wx.EVT_BUTTON, self.OnDeleteMaterial)
        clearbtn = wx.Button(panel, -1, "Clear")
        clearbtn.Bind(wx.EVT_BUTTON, self.OnClear)
        closebtn = wx.Button(panel, -1, "Close")
        closebtn.Bind(wx.EVT_BUTTON, lambda _: self.Close())

        # --- tooltips
        self.listctrl.SetToolTip("Click on a column title to sort the materials by increasing values. "
                                 "Click again for decreasing values. Select a material to copy it in the form below")
        self.namectrl.SetToolTip("Label of the material (key_material). An existing label updates this material")
        for ctrl in self.latticectrls:
            ctrl.SetToolTip("Lengths in Angstrom, angles in degree. "
                            "Arithmetical expressions are accepted, e.g.: 5.982*5.05/5")
        self.extinctionctrl.SetToolTip("Label of the extinction rules (see CrystalParameters.ApplyExtinctionrules()). "
                                       "'no' for no extinction")
        addbtn.SetToolTip("Add the material (or update the material with the same name) and save the materials file")
        deletebtn.SetToolTip("Delete the material with the name of the form and save the materials file")

        # --- layout
        hfilter = wx.BoxSizer(wx.HORIZONTAL)
        hfilter.Add(filtertxt, 0, wx.ALIGN_CENTER_VERTICAL)
        hfilter.Add(self.filterctrl, 0, wx.RIGHT, 15)
        hfilter.Add(self.counttxt, 0, wx.ALIGN_CENTER_VERTICAL)

        grid = wx.FlexGridSizer(2, 8, 2, 5)
        for name in ["name"] + LATTICE_LABELS + ["extinction"]:
            grid.Add(wx.StaticText(panel, -1, name))
        grid.Add(self.namectrl)
        for ctrl in self.latticectrls:
            grid.Add(ctrl)
        grid.Add(self.extinctionctrl)

        hbtns = wx.BoxSizer(wx.HORIZONTAL)
        for btn in (addbtn, deletebtn, clearbtn):
            hbtns.Add(btn, 0, wx.RIGHT, 10)
        hbtns.AddStretchSpacer()
        hbtns.Add(closebtn)

        sbox = wx.StaticBoxSizer(box, wx.VERTICAL)
        sbox.Add(grid, 0, wx.ALL, 5)
        sbox.Add(hbtns, 0, wx.EXPAND | wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.infotxt, 0, wx.EXPAND | wx.ALL, 8)
        vbox.Add(hfilter, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        vbox.Add(self.listctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        vbox.Add(sbox, 0, wx.EXPAND | wx.ALL, 8)
        panel.SetSizer(vbox)

        self.sb = self.CreateStatusBar()
        self.vbox = vbox
        self.SetMinSize((900, 400))
        self.ShowInfo()
        self.RefreshList()

    # --- materials library
    def uselibrary(self):
        """True if the displayed materials are those of a materials.yaml file"""
        return isinstance(self.dict_materials, DictLT.Materials)

    def getlibrary(self):
        """return the DictLT.Materials object where the materials are saved"""
        if self.uselibrary():
            return self.dict_materials
        if self.library is None:
            self.library = DictLT.Materials()
        return self.library

    def ShowInfo(self):
        if self.uselibrary():
            txt = "Materials of the file: %s" % self.dict_materials.yaml_file_path
        else:
            txt = ("Materials hard coded in dict_LaueTools.py (USE_MATERIALS_LIBRARY = False). "
                   "Added materials can be used until LaueTools is closed.\n"
                   "They are also saved in %s (read when LaueTools starts only if USE_MATERIALS_LIBRARY = True)."
                   % DictLT.get_materials_file())
        self.infotxt.SetLabel(txt)
        self.infotxt.Wrap(max(self.GetClientSize().width, 880) - 20)
        self.vbox.SetItemMinSize(self.infotxt, self.infotxt.GetBestSize())
        self.vbox.Layout()

    # --- table
    def RefreshList(self, select=None):
        """fill the table with the materials sorted according to the last clicked column

        :param select: label of the material to select and show
        """
        prefix = self.filterctrl.GetValue().strip().lower()
        self.rows = []
        for key_material in self.dict_materials.keys():
            if prefix and not str(key_material).lower().startswith(prefix):
                continue
            label, lattice, extinction = material_as_list(self.dict_materials, key_material)
            self.rows.append([str(label)] + list(lattice) + [str(extinction)])

        col = self.sortcolumn
        if col in (0, 7):
            # the name is the second sorting key of the extinction column
            self.rows.sort(key=lambda row: (row[col].lower(), row[0].lower(), row[0]))
        else:
            # lattice parameters that are not numbers (wrong expression in the file) come first
            self.rows.sort(key=lambda row: (isinstance(row[col], (int, float)),
                                            row[col] if isinstance(row[col], (int, float)) else 0,
                                            row[0].lower()))
        if not self.ascending:
            self.rows.reverse()

        self.listctrl.Freeze()
        self.listctrl.DeleteAllItems()
        for index, row in enumerate(self.rows):
            self.listctrl.InsertItem(index, row[0])
            for col_index in range(1, 7):
                value = row[col_index]
                self.listctrl.SetItem(index, col_index,
                                      "%.6g" % value if isinstance(value, (int, float)) else str(value))
            self.listctrl.SetItem(index, 7, row[7])
        self.listctrl.Thaw()

        for col_index, name in enumerate(COLUMNS):
            item = self.listctrl.GetColumn(col_index)
            arrow = ""
            if col_index == self.sortcolumn:
                arrow = u" ▲" if self.ascending else u" ▼"
            item.SetText(name + arrow)
            self.listctrl.SetColumn(col_index, item)

        self.counttxt.SetLabel("%d materials shown / %d" % (len(self.rows), len(self.dict_materials.keys())))

        if select is not None:
            for index, row in enumerate(self.rows):
                if row[0] == select:
                    self.listctrl.Select(index)
                    self.listctrl.EnsureVisible(index)
                    break

    def OnSort(self, event):
        col = event.GetColumn()
        if col < 0:
            return
        if col == self.sortcolumn:
            self.ascending = not self.ascending
        else:
            self.sortcolumn = col
            self.ascending = True
        self.RefreshList()

    def OnFilter(self, _):
        self.RefreshList()

    def OnSelectMaterial(self, event):
        """copy the selected material in the form"""
        row = self.rows[event.GetIndex()]
        self.namectrl.SetValue(row[0])
        for ctrl, value in zip(self.latticectrls, row[1:7]):
            ctrl.SetValue("%.10g" % value if isinstance(value, (int, float)) else str(value))
        self.extinctionctrl.SetValue(row[7])

    # --- form
    def OnClear(self, _):
        self.namectrl.SetValue("")
        for ctrl in self.latticectrls:
            ctrl.SetValue("")
        self.extinctionctrl.SetValue("no")

    def ReadForm(self):
        """return label, lattice, extinction of the form

        :raises ValueError: with the message to show to the user
        """
        label = self.namectrl.GetValue().strip()
        if not label:
            raise ValueError("Name of the material is missing.")
        if any(char.isspace() for char in label):
            raise ValueError("Name of the material must not contain spaces.")

        lattice = []
        for name, ctrl in zip(LATTICE_LABELS, self.latticectrls):
            text = ctrl.GetValue().strip().replace(",", ".")
            if not text:
                raise ValueError("Lattice parameter %s is missing." % name)
            try:
                value = DictLT.evaluate_arithmetic(text)
            except ValueError:
                raise ValueError("Lattice parameter %s: '%s' is not a number." % (name, text))
            if isinstance(value, float) and value.is_integer() and "." not in text:
                value = int(value)
            lattice.append(value)
        if min(lattice[:3]) <= 0:
            raise ValueError("Lengths a, b, c must be positive.")
        if min(lattice[3:]) <= 0 or max(lattice[3:]) >= 180:
            raise ValueError("Angles alpha, beta, gamma must be between 0 and 180 degrees.")

        extinction = self.extinctionctrl.GetValue().strip()
        if not extinction:
            raise ValueError("Extinction is missing ('no' for no extinction).")
        return label, lattice, extinction

    def OnAddMaterial(self, _):
        try:
            label, lattice, extinction = self.ReadForm()
        except ValueError as error:
            wx.MessageBox(str(error), "Materials Editor", wx.OK | wx.ICON_ERROR)
            return

        if extinction not in DictLT.dict_Extinc_inv:
            answer = wx.MessageBox("Extinction '%s' is not a known label of extinction rules (dict_Extinc in "
                                   "dict_LaueTools.py):\nsome LaueTools modules will fail with this material."
                                   "\n\nAdd the material anyway?" % extinction,
                                   "Materials Editor", wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
            if answer != wx.YES:
                return

        if label in self.dict_materials.keys():
            _, oldlattice, oldextinction = material_as_list(self.dict_materials, label)
            answer = wx.MessageBox("Material '%s' already exists:\n%s  %s\n\nReplace it by:\n%s  %s ?"
                                   % (label, list(oldlattice), oldextinction, lattice, extinction),
                                   "Materials Editor", wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
            if answer != wx.YES:
                return

        try:
            yaml_file = self.getlibrary().add_or_update_material(label, lattice, extinction)
        except (OSError, ValueError) as error:
            wx.MessageBox("Material '%s' is not saved:\n%s" % (label, error),
                          "Materials Editor", wx.OK | wx.ICON_ERROR)
            return
        if not self.uselibrary():
            self.dict_materials[label] = [label, lattice, extinction]

        self.AfterChange("Material '%s' saved in %s" % (label, yaml_file), select=label)

    def OnDeleteMaterial(self, _):
        label = self.namectrl.GetValue().strip()
        if label not in self.dict_materials.keys():
            wx.MessageBox("'%s' is not the name of a material." % label,
                          "Materials Editor", wx.OK | wx.ICON_ERROR)
            return
        answer = wx.MessageBox("Delete material '%s'?" % label, "Materials Editor",
                               wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
        if answer != wx.YES:
            return

        try:
            yaml_file = self.getlibrary().remove_material(label)
        except (OSError, ValueError) as error:
            wx.MessageBox("Material '%s' is not deleted:\n%s" % (label, error),
                          "Materials Editor", wx.OK | wx.ICON_ERROR)
            return
        if not self.uselibrary():
            del self.dict_materials[label]

        if yaml_file is None:
            self.AfterChange("Material '%s' deleted (it was not in the materials file)" % label)
        else:
            self.AfterChange("Material '%s' deleted in %s" % (label, yaml_file))

    def AfterChange(self, message, select=None):
        self.ShowInfo()
        self.RefreshList(select=select)
        self.sb.SetStatusText(message)
        if self.onchange is not None:
            self.onchange()


def start():
    """command lauetools-materials: launch the materials editor"""
    app = wx.App(False)
    frame = MaterialsEditorFrame(None)
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    start()
