# -*- coding: utf-8 -*-

import sys
import numpy as np

import wx

try:
    from wx.lib.embeddedimage import PyEmbeddedImage

    PyEmbeddedImageOk = True
except:
    PyEmbeddedImageOk = False
import wx.lib.mixins.listctrl

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

#    def ssitem(*arg):
#        return wx.lib.mixins.listctrl.SetItem(*arg))
#    wx.lib.mixins.listctrl.SetStringItem = ssitem

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

fields = ["spot index", "h", "k", "l", "Energy", "2theta", "chi", "X", "Y", "Intensity", "deviation"]


class SpotsEditor(wx.Frame):
    """ GUI class for Spot properties edition and selection
  
    """
    def __init__(self, parent, _id, title, dict_spots_data,
                    func_to_call=None, field_name_and_order=fields):
        """
        dictionnary of data is as following:
        dict_spots_data = { 'spot index':[0,1,3,6,9,10],
                            'Energy' : [8.2,12.2,6.3,13.6,5.,15.,19.],
                                'h': [0,0,5,1,2,2],
                                'k': [1,1,1,2,0,2],
                                'l': [-1,0,6,10,1,0],
                                '2theta': [23.125,90.1258,68.36512,49.98598,65.0000236,101.101101]}
        """
        # wx.Dialog.__init__(self, parent, _id, title, size=(600,500), style=wx.DEFAULT_DIALOG_STYLE)
        #        wx.Dialog.__init__(self, parent, _id, title, size=(1100, 500), style=wx.OK)
        wx.Frame.__init__(self, parent, _id, title, size=(600, 500))

        # print(("parent", parent))
        self.parent = parent

        self.func_to_call = func_to_call

        proceed = True

        self.field_name = []
        for field in field_name_and_order:
            if field in list(dict_spots_data.keys()):
                self.field_name.append(field)
        for field in list(dict_spots_data.keys()):
            if field not in field_name_and_order:
                wx.MessageBox('The field "%s" is unknown!' % field, "INFO")
                proceed = False

        # print "self.field_name in SpotsEditor",self.field_name
        print(("Current Fields", self.field_name))

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)  # panel 1 and panel 2 dans dialog
        if WXPYTHON4:
            vbox3 = wx.FlexGridSizer(2, 10, 10)
        else:
            vbox3 = wx.FlexGridSizer(7, 2, 0, 5)

        vbox3.SetFlexibleDirection(wx.HORIZONTAL)
        vbox4 = wx.BoxSizer(wx.VERTICAL)  # dans panel 2
        pnl1 = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)
        pnl2 = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)

        self.Bind(wx.EVT_KEY_DOWN, self.onKeypressed)

        self.listcontrol = MyListCtrl(self, self.field_name, dict_spots_data)

        wx.lib.mixins.listctrl.ColumnSorterMixin.__init__(self.listcontrol, len(self.field_name) + 1)

        self.listcontrol.Bind(wx.EVT_KEY_DOWN, self.onKeypressed)

        # layout

        hbox1.Add(pnl1, 1, wx.EXPAND | wx.ALL, 3)
        hbox1.Add(pnl2, 1, wx.EXPAND | wx.ALL, 3)

        self.tc1 = wx.TextCtrl(pnl1, -1, size=(150, -1))
        self.tc2 = wx.TextCtrl(pnl1, -1, size=(150, -1))
        self.tc3 = wx.TextCtrl(pnl1, -1, size=(150, -1))

        list_tcs = [self.tc1, self.tc2, self.tc3]

        self.f1 = wx.ComboBox(pnl1, 700, self.field_name[0], choices=self.field_name,
                                style=wx.CB_READONLY, size=(120, -1))
        self.f1.Bind(wx.EVT_COMBOBOX, self.EnterCombocolumn1, id=700)

        self.f2 = wx.ComboBox(pnl1, 701, self.field_name[1], choices=self.field_name,
                                style=wx.CB_READONLY, size=(120, -1))
        self.f2.Bind(wx.EVT_COMBOBOX, self.EnterCombocolumn2, id=701)

        self.f3 = wx.ComboBox(pnl1, 702, self.field_name[2], choices=self.field_name,
                                style=wx.CB_READONLY, size=(120, -1))
        self.f3.Bind(wx.EVT_COMBOBOX, self.EnterCombocolumn3, id=702)

        list_fs = [self.f1, self.f2, self.f3]

        # default filter fields if not selecting columns
        self.filterfield1, self.filterfield2, self.filterfield3 = self.field_name[:3]

        stf1 = wx.StaticText(pnl1, -1, "expression 1")
        stf2 = wx.StaticText(pnl1, -1, "expression 2")
        stf3 = wx.StaticText(pnl1, -1, "expression 3")

        vbox3.Add(wx.StaticText(pnl1, -1, "FILTERS (logical AND)"))
        vbox3.Add(wx.StaticText(pnl1, -1, ""))
        for k, elem in enumerate([stf1, stf2, stf3]):
            vbox3.Add(elem, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
            vbox3.Add(
                wx.StaticText(pnl1, -1, "selected property"),
                0,
                wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL,
            )
            vbox3.Add(list_tcs[k], 0)

            vbox3.Add(list_fs[k], 0)

        pnl1.SetSizer(vbox3)

        # -------------------------------
        st = wx.StaticText(pnl2, -1, "Nb of spots links")
        #         self.tcnb = wx.TextCtrl(pnl2, -1, style=wx.TE_READONLY)
        #         self.tcnb.Disable()
        self.tcnb = wx.StaticText(pnl2, -1)

        self.initialnblinks = wx.StaticText(pnl2, -1)

        hbobox = wx.BoxSizer(wx.HORIZONTAL)
        hbobox.Add(st, 0, 10)
        hbobox.Add(self.tcnb, 0, 10)
        hbobox.Add(self.initialnblinks, 0, 10)

        # ---------------------------------------------
        apfilterbtn = wx.Button(pnl2, 10, "Apply Filter")
        rlbtn = wx.Button(pnl2, 100, "Reload")
        rmv1spotbtn = wx.Button(pnl2, 11, "Remove one spot")
        plotfreqbtn = wx.Button(pnl2, 106, "Plot Freq.")

        # vbox4.Add(wx.Button(pnl2, 10, 'Filter'),   0, wx.ALIGN_CENTER | wx.TOP, 15)
        hbobox0 = wx.BoxSizer(wx.HORIZONTAL)
        hbobox0.Add(apfilterbtn, 1, wx.EXPAND | wx.ALL, 5)
        hbobox0.Add(plotfreqbtn, 1, wx.EXPAND | wx.ALL, 5)
        vbox4.Add(hbobox0, 0, wx.ALIGN_CENTER, 15)

        vbox4.Add(rlbtn, 0, wx.ALIGN_CENTER | wx.TOP, 5)
        vbox4.Add(rmv1spotbtn, 0, wx.ALIGN_CENTER | wx.TOP, 5)
        vbox4.Add(hbobox, 0, wx.ALIGN_CENTER | wx.TOP, 15)
        vbox4.Add(wx.Button(pnl2, wx.ID_OK, "Accept and Quit",
                            size=(-1, 60)), 0, wx.EXPAND | wx.ALL, 5)

        self.Bind(wx.EVT_BUTTON, self.OnApplyFilter, id=10)
        self.Bind(wx.EVT_BUTTON, self.OnPlot, id=106)
        self.Bind(wx.EVT_BUTTON, self.OnReload, id=100)
        self.Bind(wx.EVT_BUTTON, self.OnRemove, id=11)
        self.Bind(wx.EVT_BUTTON, self.OnAcceptQuit, id=wx.ID_OK)

        pnl2.SetSizer(vbox4)

        # final layout
        vbox.Add(self.listcontrol, 1, wx.EXPAND | wx.ALL)
        vbox.Add(hbox1, 0, wx.EXPAND | wx.ALL)
        self.SetSizer(vbox)

        # pre defined links
        self.dict_spots_data = dict_spots_data
        if self.dict_spots_data != None:

            # check if all fields have the same length
            length = []
            allfield = []
            for field, data in list(self.dict_spots_data.items()):
                length.append(len(data))
                allfield.append(field)

            for ldata in length[1:]:

                if ldata != length[0]:

                    proceed = False
                    txt = "The input Data contain list of data of different lengths!\n"
                    txt += "List of data for each field must have the same length!\n"
                    txt += "Field %s\nLength %s" % (allfield, length)
                    wx.MessageBox(txt, "INFO")
                    break

            if proceed:
                self.nbspots = length[0]
                self.listcontrol.add_rows(self.dict_spots_data, self.nbspots)
                #                 self.tcnb.SetValue(str(self.nbspots))
                self.tcnb.SetLabel(str(self.nbspots))
                self.initialnblinks.SetLabel("/%s" % str(self.nbspots))

        self.toreturn = None

        # tooltips
        tp1 = "To keep spots associations according to the spots properties values. Three expressions can be used to select only spots whose property values satisfy all expressions.\n"
        tp1 += 'For instances: Expression 1 : <50  with selected property "#Spot Exp" and <0.25 with selected property "residues"'
        tp1 += " select only spots associations (links) with experimental spot index smaller than 50 and angular distance (residues) between exp. and theo. spots smaller than 0.25 degree"
        pnl1.SetToolTipString(tp1)

        apfilterbtn.SetToolTipString(
            "Apply filter(s) designed by logical expression on columns properties"
        )
        rlbtn.SetToolTipString("Reload and Display the complete set of associations")
        rmv1spotbtn.SetToolTipString(
            "Remove the spot association selected in the links list"
        )
        plotfreqbtn.SetToolTipString(
            "Plot residues distribution frequency of the spot links list"
        )

    def onKeypressed(self, event):
        #         print "key ==", dir(event)
        #         print event.KeyCode
        #         print event.RawKeyCode
        # delete items
        if event.KeyCode == 68 and event.RawKeyCode == 100:  # it means 'd'
            self.OnRemove(event)

    def EnterCombocolumn1(self, event):

        item = event.GetSelection()
        self.filterfield1 = self.field_name[item]
        # print "selected filter field", self.filterfield1

    def EnterCombocolumn2(self, event):

        item = event.GetSelection()
        self.filterfield2 = self.field_name[item]
        # print "selected filter field", self.filterfield2

    def EnterCombocolumn3(self, event):

        item = event.GetSelection()
        self.filterfield3 = self.field_name[item]
        # print "selected filter field", self.filterfield3

    def OnReload(self, event):
        self.OnClear(event)
        self.listcontrol.add_rows(self.dict_spots_data, self.nbspots)
        #         self.tcnb.SetValue(str(self.nbspots))
        self.tcnb.SetLabel(str(self.nbspots))

    def OnApplyFilter(self, event):
        # read filter fields
        field = [ff for ff in (self.filterfield1, self.filterfield2, self.filterfield3)]

        # read conditionnal expressions
        cond = [str(tc.GetValue()) for tc in (self.tc1, self.tc2, self.tc3)]

        AllConds = True * np.ones(
            self.nbspots
        )  # by default we start the filter from the whole initial data

        for k, condition in enumerate(cond):
            if condition != "":
                # print "condition is ",condition

                array_data = np.array(self.dict_spots_data[field[k]])
                # print "before filtering", array_data
                toeval = "array_data" + condition

                try:
                    _cond = eval(toeval)
                except NameError:
                    wx.MessageBox(
                        "Wrong condition:\nType simply in field: >5, <=58.5 ==6.",
                        "INFO",
                    )
                    return

                # print "after filtering", array_data[_cond]
                # print "_cond",_cond
                # print "len(_cond)", len(_cond)
                AllConds = np.logical_and(AllConds, _cond)
                # print "AllConds",AllConds

        self.OnClear(event)
        filterdata = {}
        filternbspots = len(np.where(AllConds == True)[0])
        print(("filternbspots", filternbspots))
        for key, dat in list(self.dict_spots_data.items()):
            filterdata[key] = np.array(dat)[AllConds]
        # print "filterdata",filterdata
        self.listcontrol.add_rows(filterdata, filternbspots)
        #         self.tcnb.SetValue(str(filternbspots))
        self.tcnb.SetLabel(str(filternbspots))

    def OnPlot(self, evt):
        data = self.ReadSortedData()
        columdata = data[:, -1]

        bibins = np.linspace(
            np.amin(columdata), np.max(columdata), max(10, len(columdata) / 10)
        )
        databinned = np.histogram(columdata, bins=bibins)  # heights, bins
        print(("databinned", databinned))
        barplt = BarPlotFrame(
            self, -1, "Plot Frequency", databinned, title_in_plot="last column"
        )
        barplt.Show(True)

    def OnRemove(self, evt):
        index = self.listcontrol.GetFocusedItem()
        self.listcontrol.DeleteItem(index)
        #         oldnb = int(self.tcnb.GetValue())
        oldnb = int(self.tcnb.GetLabel())
        if oldnb > 0:
            #             self.tcnb.SetValue(str(oldnb - 1))
            self.tcnb.SetLabel(str(oldnb - 1))

    def ReadSortedData(self):
        self.toreturn = []
        #         print "self.field_name", self.field_name
        for idx in range(self.listcontrol.GetItemCount()):  # loop over spots
            dataline = []
            for k, field in enumerate(self.field_name):  # loop over fields
                dataline.append(str(self.listcontrol.GetItem(idx, k).GetText()))
            self.toreturn.append(dataline)

        self.toreturn = np.array(self.toreturn, dtype=np.float)
        return self.toreturn

    def OnAcceptQuit(self, evt):

        self.toreturn = self.ReadSortedData()

        #         print 'self.toreturn', self.toreturn

        #        if self.parent is not None:
        #            print "self.parent", self.parent
        #            self.parent.readdata_fromEditor(self.toreturn)

        if self.func_to_call is not None:
            #             print "self.func_to_call", self.func_to_call

            #             print "self.toreturn", self.toreturn
            print(("nb of spots selected", len(self.toreturn)))
            self.func_to_call(self.toreturn)

        self.Close()

    def OnClear(self, evt):
        self.listcontrol.DeleteAllItems()


class MyListCtrl(wx.ListCtrl, wx.lib.mixins.listctrl.ColumnSorterMixin):
    def __init__(self, parent, field_name, dict_data):
        wx.ListCtrl.__init__(self, parent, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)

        self.parent = parent
        self.field_name = field_name
        self.dict_data = dict_data
        # print "self.field_name MyListCtrl",self.field_name
        # print "self.dict_data MyListCtrl",self.dict_data

        if PyEmbeddedImageOk:
            SmallUpArrow = PyEmbeddedImage(
                "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAADxJ"
                "REFUOI1jZGRiZqAEMFGke2gY8P/f3/9kGwDTjM8QnAaga8JlCG3CAJdt2MQxDCAUaOjyjKMp"
                "cRAYAABS2CPsss3BWQAAAABJRU5ErkJggg=="
            )
            SmallDnArrow = PyEmbeddedImage(
                "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAEhJ"
                "REFUOI1jZGRiZqAEMFGke9QABgYGBgYWdIH///7+J6SJkYmZEacLkCUJacZqAD5DsInTLhDR"
                "bcPlKrwugGnCFy6Mo3mBAQChDgRlP4RC7wAAAABJRU5ErkJggg=="
            )

            self.il = wx.ImageList(16, 16)
            self.sort_up = self.il.Add(SmallUpArrow.GetBitmap())
            self.sort_dn = self.il.Add(SmallDnArrow.GetBitmap())

        for k, field in enumerate(self.field_name):
            self.InsertColumn(k, str(field))
            widthcolumn = 100
            if field in ("h", "k", "l"):
                widthcolumn = 30
            self.SetColumnWidth(k, widthcolumn)

        self.itemDataMap = {}

    def add_rows(self, dict_data, nbspots):
        # print "Building list of Data spots"

        for row in range(nbspots):  # loop over spots
            # print "row",row
            self.add_row(row, dict_data)
            # print "str(dict_data[self.field_name[0]][spot_index])",str(dict_data[self.field_name[0]][spot_index])
            # index = self.InsertStringItem( num_items, str(dict_data[self.field_name[0]][spot_index]) ) # creating item and filling first field
            # for k,field in enumerate(self.field_name[1:]): # loop over other fields to fill
            # self.SetStringItem( num_items, k+1, str(dict_data[field][spot_index]) )

    def add_row(self, row, dict_data):

        # if 0: wxpython 4.0.1
        #     index = self.InsertItem(
        #         sys.maxsize, str(dict_data[self.field_name[0]][row])
        #     )  # ,    self.img_list[content.nature])
        #     _content = []

        #     for k, field in enumerate(
        #         self.field_name
        #     ):  # loop over other fields to fill
        #         # print "k,field",k,field
        #         self.SetItem(index, k, str(dict_data[field][row]))
        #         _content.append(float(dict_data[field][row]))

        #     # print "index,row",index, row
        #     self.SetItemData(index, row)
        #     self.itemDataMap[row] = _content
            
        if WXPYTHON4:
            item = wx.ListItem()
            item.SetId(row)
            self.InsertItem(item)

#            index = self.InsertItem(
#                sys.maxsize, str(dict_data[self.field_name[0]][row]))
#            # ,    self.img_list[content.nature])
            _content = []

            for k, field in enumerate(
                self.field_name
            ):  # loop over other fields to fill
                # print "k,field",k,field
                self.SetStringItem(row, k, str(dict_data[field][row]))
                _content.append(float(dict_data[field][row]))

            # print "index,row",index, row
            self.SetItemData(row, row)
            self.itemDataMap[row] = _content
        else:

            index = self.InsertStringItem(
                sys.maxsize, str(dict_data[self.field_name[0]][row])
            )  # ,    self.img_list[content.nature])
            _content = []

            for k, field in enumerate(
                self.field_name
            ):  # loop over other fields to fill
                # print "k,field",k,field
                self.SetStringItem(index, k, str(dict_data[field][row]))
                _content.append(float(dict_data[field][row]))

            # print "index,row",index, row
            self.SetItemData(index, row)
            self.itemDataMap[row] = _content

    def GetListCtrl(self):
        return self

    # def GetSortImages(self):
    # return (self.sort_dn, self.sort_up)

    def clean(self):
        self.DeleteAllItems()
        self.itemDataMap = {}


class BarPlotFrame(wx.Frame):
    """
    Class to plot Bar from data 
    """

    def __init__(self, parent, _id, title, dataarray, title_in_plot=None):

        wx.Frame.__init__(self, parent, _id, title, size=(500, 500))

        self.data = dataarray

        self.title = title
        self.title_in_plot = title_in_plot

        self.panel = wx.Panel(self)
        self.dpi = 100
        self.figsize = 4
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.fig.set_size_inches(self.figsize, self.figsize, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.Layout()

        self._replot()

    def _replot(self):

        self.axes.clear()
        #        self.axes.set_autoscale_on(False) # Otherwise, infinite loop
        self.axes.set_autoscale_on(True)

        heights = self.data[0]
        leftposition = self.data[1]

        print(("len(heights", len(heights)))
        print(("len(leftposition", len(leftposition)))

        self.axes.bar(
            leftposition[:-1], heights, width=leftposition[1] - leftposition[0]
        )

        if self.title_in_plot != None:
            self.axes.set_title(self.title_in_plot)

        self.axes.set_xlabel("intensity")
        self.axes.set_ylabel("nb of pairs")

        # **********************

        # def fromindex_to_pixelpos_x(index, pos): # must contain 2 args!
        # return index
        # def fromindex_to_pixelpos_y(index, pos):
        # return index

        # self.axes.xaxis.set_major_formatter(pylab.FuncFormatter(fromindex_to_pixelpos_x))
        # self.axes.yaxis.set_major_formatter(pylab.FuncFormatter(fromindex_to_pixelpos_y))

        self.axes.grid(True)
        # **********************

        #         # redraw the display
        #         self.plotPanel.draw()

        self.canvas.draw()


if __name__ == "__main__":

    # test data:
    mySpotData = {
        "spot index": [0, 1, 3, 6, 9, 10, 2, 5, 4, 11, 20],
        "Energy": [8.2, 12.2, 6.3, 13.6, 5.0, 15.0, 19.0, 6.325, 4.58, 9.8796, 10.2154],
        "h": [0, 0, 5, 1, 2, 2, 1, 1, 1, -1, -1],
        "k": [1, 1, 1, 2, 0, 2, 1, 0, -1, 2, 5],
        "l": [-1, 0, 6, 10, 1, 0, -6, -3, -2, -1, 0],
        "2theta": [
            23.125,
            90.1258,
            68.36512,
            49.98598,
            65.0000236,
            101.101101,
            89.8996,
            56.2356,
            45.2356,
            92.2365,
            48.2658,
        ],
        "Intensity": [
            10000.0,
            558.235,
            24.3265,
            -1.123665,
            65535,
            100,
            1,
            0,
            445,
            2048,
            6525.356,
        ],
    }

    class MyApp(wx.App):
        def OnInit(self):
            dia = SpotsEditor(None, -1, "Spots Editor.py", mySpotData, field_name_and_order=fields)
            #            dia.Destroy()
            dia.Show(True)
            print("Data Selected")
            #            print dia.toreturn
            print((dia.field_name))

            return True

    app = MyApp(0)
    app.MainLoop()
