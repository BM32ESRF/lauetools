r"""
plotmeshspecGUI.py is a module for the visualisation of 2D Laue microdiffraction map recorded on spec file at ESRF

This module belongs to the open source LaueTools project
with a free code repository at at gitlab.esrf.fr

https://gitlab.esrf.fr/micha/lauetools

J. S. Micha Jan 2022
mailto: micha --+at-+- esrf --+dot-+- fr
"""



import datetime
import math
import os
import re
import sys
import time

import matplotlib as mpl

from matplotlib import __version__ as matplotlibversion
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                    NavigationToolbar2WxAgg as NavigationToolbar)

import matplotlib.colors as colors
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MaxNLocator

from pylab import cm as pcm

import pyperclip

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        """ alias fct """
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

import numpy as np
    
try:
    import h5py
except ModuleNotFoundError:
    print('warning: h5py is missing. It is useful for playing with hdf5 for some LaueTools modules. Install it with pip> pip install h5py')
    
try:
    from SpecClient_gevent import SpecCommand
except ImportError:
    print('-- warning. spec control software and SpecClient_gevent missing ? (normal, if you are not at the beamline')

import LaueTools.generaltools as GT
# from LaueTools.Daxm.contrib.spec_reader import ReadSpec, ReadHdf5, readdata_from_hdf5key
from LaueTools.logfile_reader import ReadSpec, ReadHdf5, readdata_from_hdf5key
import LaueTools.MessageCommand as MC
import LaueTools.IOLaueTools as IOLT
# import LaueTools.Daxm.contrib.spec_reader as spec_reader
import LaueTools.logfile_reader as logfile_reader
import LaueTools.map_workflow as MW
from LaueTools.GUI.rangeslider import RangeSlider, EVT_RANGE_SLIDER


import wx.lib.agw.customtreectrl as CT

class TreePanel(wx.Panel):
    """ class of tree organisation of scans

    granparent class must provide  ReadScan_SpecFile()

    sets granparent scan_index_mesh  or scan_index_ascan to selected item index
    """
    def __init__(self, parent, scantype=None, _id=wx.ID_ANY, frameparent=None, **kwd):
        wx.Panel.__init__(self, parent=parent, id=_id, **kwd)

        self.parent = parent
        self.scantype = scantype
        # main frame (given when the parent is not the main panel, e.g. a splitter window)
        self.frameparent = frameparent if frameparent is not None else self.parent.GetParent()
        # self.tree = wx.TreeCtrl(self, -1, wx.DefaultPosition, (-1, -1),
        #                                                     wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS)
        # agwStyle=wx.TR_DEFAULT_STYLE
        self.tree = CT.CustomTreeCtrl(self, -1, agwStyle=wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS | wx.TR_MULTIPLE,
                                      size=(self.MINWIDTH, -1))
        self.tree.DoGetBestSize()
        self.tree.SetMinSize((self.MINWIDTH, -1))
        self.SetMinSize((self.MINWIDTH, -1))

        self.maketree()

        # multiple selection ------
        self.keypressed = None
        self.multiitems = False
        self.set_selected_indices = set()
        self.multipleselectionOn = False
        # --------------------

        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelChanged)
        #self.tree.Bind(wx.EVT_TREE_SEL_CHANGING, self.OnSelChanged)
        self.tree.Bind(wx.EVT_TREE_KEY_DOWN, self.OnkeyPressed)
        # tooltip of the scan under the mouse
        self.hovereditem = None
        self.tree.Bind(wx.EVT_MOTION, self.OnMotion)

        # wx.EVT_TREE_ITEM_RIGHT_CLICK
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.tree, 1, wx.EXPAND)
        vbox.AddSpacer(1)

        self.SetSizer(vbox)

    def OnMotion(self, event):
        """tooltip of the scan under the mouse (command, date, duration, end reason)"""
        item, _ = self.tree.HitTest(event.GetPosition())
        if item != self.hovereditem:
            self.hovereditem = item
            tooltip = None
            if item is not None:
                tooltip = self.frameparent.scan_tooltip(self.scantype, self.tree.GetItemText(item))
            self.tree.SetToolTip(tooltip or "")
        event.Skip()

    MINWIDTH = 200  # pixels
    MAXWIDTH = 600

    def fit_width(self):
        """minimum width of the list from its longest scan name (between MINWIDTH and MAXWIDTH)"""
        names = []
        item = self.tree.GetFirstChild(self.root)[0]
        while item is not None and item.IsOk():
            names.append(self.tree.GetItemText(item))
            item = self.tree.GetNextSibling(item)
        textwidth = max((self.tree.GetTextExtent(name)[0] for name in names), default=0)
        # + indentation of items, margins and vertical scrollbar
        width = int(min(max(textwidth + 70, self.MINWIDTH), self.MAXWIDTH))
        self.tree.SetMinSize((width, -1))
        self.SetMinSize((width, -1))

    def maketree(self):
        self.root = self.tree.AddRoot("SpecFiles")
        self.tree.AppendItem(self.root, str(self.scantype))

    def OnkeyPressed(self, event):
        #print(dir(event))
        key = event.GetKeyCode()
        # print('key pressed is ', key)
        if key == wx.WXK_SHIFT:
            self.multiitems = True
            self.keypressed = 'shift'
            print('shift')
            self.set_selected_indices = set()
        elif key == wx.WXK_RAW_CONTROL:
            print('ctrl')
            self.multiitems = True
            self.keypressed = 'ctrl'

        elif key == wx.WXK_DOWN:
            # arrow down
            if self.scantype == 'ASCAN' and self.frameparent.filetype == 'spec':
                nbascans = len(self.frameparent.list_ascanscan_indices)
                #print('self.frameparent.list_ascanscan_indices', self.frameparent.list_ascanscan_indices)
                poslastscan = self.frameparent.list_ascanscan_indices.index(self.last_sel_scan_index)
                #print('poslastscan', poslastscan)
                newascanindex = self.frameparent.list_ascanscan_indices[min(poslastscan + 1, nbascans - 1)]
                #print('newascanindex', newascanindex)

                self.OnSelChanged(event, scan_index=newascanindex)

            elif self.scantype == 'MESH' or self.frameparent.filetype == 'hdf5':
                if self.frameparent.filetype == 'spec':
                    #print('self.frameparent.list_meshscan_indices',self.frameparent.list_meshscan_indices)
                    #nbmeshes = len(self.frameparent.list_meshscan_indices)
                    poslastscan = self.frameparent.list_meshscan_indices.index(self.last_sel_scan_index)
                    #print('poslastscan',poslastscan)
                    newmeshindex = self.frameparent.list_meshscan_indices[max(poslastscan + 1, 0)]
                    #print('newmeshindex',newmeshindex)

                    self.OnSelChanged(event, scan_index=newmeshindex)

                else:
                    nextitem = self.tree.GetNext(self.last_item)
                    try:
                        self.tree.DoSelectItem(nextitem)
                        self.OnSelChanged(event, scan_index=nextitem)
                    except AttributeError: # reaching last element!
                        pass

        elif key == wx.WXK_UP:
            if self.scantype == 'ASCAN' and self.frameparent.filetype == 'spec':
                nbascans = len(self.frameparent.list_ascanscan_indices)
                poslastscan = self.frameparent.list_ascanscan_indices.index(self.last_sel_scan_index)
                newascanindex = self.frameparent.list_ascanscan_indices[max(poslastscan - 1, 0)]

                self.OnSelChanged(event, scan_index=newascanindex)

            elif self.scantype == 'MESH' or self.frameparent.filetype == 'hdf5':
                if self.frameparent.filetype == 'spec':
                    #nbmeshes = len(self.frameparent.list_meshscan_indices)
                    poslastscan = self.frameparent.list_meshscan_indices.index(self.last_sel_scan_index)
                    newmeshindex = self.frameparent.list_meshscan_indices[max(poslastscan - 1, 0)]

                    self.OnSelChanged(event, scan_index=newmeshindex)
                else:
                    #print('self.last_item',self.last_item)
                    # print('self.frameparent.list_meshscan_indices',self.frameparent.list_meshscan_indices)
                    # nbmeshes = len(self.frameparent.list_meshscan_indices)
                    # poslastscan = self.frameparent.list_meshscan_indices.index(self.last_sel_scan_index)
                    # print('poslastscan',poslastscan)
                    # newmeshindex = self.frameparent.list_meshscan_indices[max(poslastscan - 1, 0)]
                    # print('newmeshindex',newmeshindex)

                    #print(dir(self.tree))

                    previtem = self.tree.GetPrev(self.last_item)
                    self.tree.DoSelectItem(previtem)
                    #self.tree.GetPrev(lastclickeditem)
                    # lastclickeditem = self.tree.GetFocusedItem()
                    # previtem = self.tree.GetPrevVisible(self.lastclickeditem)
                    # self.tree.SetFocusedItem(previtem)

                    self.OnSelChanged(1, scan_index=previtem)

        elif key in (83, '83', 's'):
            print('\n\ns   !!!!\n\n\n')
            self.keypressed = 's'
            self.set_selected_indices = set()
            self.multipleselectionOn = not self.multipleselectionOn
            if self.multipleselectionOn and self.scantype == 'ASCAN':
                self.frameparent.txtselectionmode.SetLabel('Selection Mode: Multi')
            else:
                self.frameparent.txtselectionmode.SetLabel('Selection Mode: Single')

            #self.frameparent.ReadMultipleScans(self.set_selected_indices, resetlistcounters=True)

    def OnSelChanged(self, event, scan_index=None):
        print('***\n OnSelChanged  filetype = %s\n***'%self.frameparent.filetype)
        
        if self.frameparent.filetype == 'spec':
            self.SelChangedSpec(event, scan_index=scan_index)
        else:
            #print('self.frameparent.listmesh',self.frameparent.listmesh)
            self.SelChangedHdf5(event, item=None)

    def SelChangedHdf5(self, event, item=None):
        """ read item from HDF5 tree

        call self.frameparent.ReadScan_SpecFile()
        and set self.frameparent.scan_index_mesh 
        
        For now only single selection is implemented...
        """
        #print('\n\n SelChangeHdf5')
        if item is None:
            if event != 1:
                item = event.GetItem()
        
        if item is None:
            return
        selected_item = self.tree.GetItemText(item)
        #scan_index = int(selected_item)
        print("item selected: ", selected_item)
        if selected_item == str(self.scantype):  # 'ASCAN' item: list the 1D scans of the logfile
            if self.scantype == 'ASCAN':
                wx.CallAfter(self.frameparent.list_hdf5_scans1D)  # after this event: the tree is rebuilt
            return
        #print("selected_item ", dir(item))
        
        # multipe selection with ascan's  NOT YET WORKING !!! ....
        if self.multipleselectionOn and self.scantype == 'ASCAN': #self.keypressed == 's':

            if self.set_selected_indices is None:
                self.set_selected_indices = set([scan_index])

                print('first self.set_selected_indices in  selection mode', self.set_selected_indices)

            if scan_index not in self.set_selected_indices:

                self.set_selected_indices.add(scan_index)
                self.last_sel_scan_index = scan_index
            else:
                self.set_selected_indices.remove(scan_index)
            #print("self.set_selected_indices", self.set_selected_indices)
            #self.keypressed = None

            print('self.set_selected_indices in  selection mode', self.set_selected_indices)

            self.frameparent.ReadMultipleScans(self.set_selected_indices, resetlistcounters=True)

            print('ssssss')

        # single selection
        elif not self.keypressed in ('shift', 'ctrl'):
            print('Single selection MODE')
            
            self.set_selected_indices = set([selected_item])

            self.frameparent.ReadScan_SpecFile(selected_item, resetlistcounters=True)
            if self.scantype == 'MESH':
                self.frameparent.scan_index_mesh = selected_item
            elif self.scantype == 'ASCAN':
                self.frameparent.scan_index_ascan = selected_item

            self.last_sel_scan_index = selected_item
            self.last_item = item

        # tooltip of the scan under the mouse: see OnMotion()
        event.Skip()
    
    def SelChangedSpec(self, event, scan_index=None):     
        print('\n\n OnSelChangedSpec')
        if scan_index is None:
            print('scan_index',scan_index)
            item = event.GetItem()
            selected_item = self.tree.GetItemText(item)
            if selected_item in (str(self.scantype), ):
                return
            scan_index = int(selected_item)
            print("click on ", scan_index)
            #print("selected_item ", dir(item))

        if self.multipleselectionOn and self.scantype == 'ASCAN': #self.keypressed == 's':

            if self.set_selected_indices is None:
                self.set_selected_indices = set([scan_index])

                print('first self.set_selected_indices in  selection mode', self.set_selected_indices)

            if scan_index not in self.set_selected_indices:

                self.set_selected_indices.add(scan_index)
                self.last_sel_scan_index = scan_index
            else:
                self.set_selected_indices.remove(scan_index)
            #print("self.set_selected_indices", self.set_selected_indices)
            #self.keypressed = None

            print('self.set_selected_indices in  selection mode', self.set_selected_indices)

            self.frameparent.ReadMultipleScans(self.set_selected_indices, resetlistcounters=True)

            print('ssssss')

        elif not self.keypressed in ('shift', 'ctrl'):  # single selection
            print('Single selection MODE')
            self.set_selected_indices = set([scan_index])

            self.frameparent.ReadScan_SpecFile(scan_index, resetlistcounters=True)
            if self.scantype == 'MESH':
                self.frameparent.scan_index_mesh = scan_index
            elif self.scantype == 'ASCAN':
                self.frameparent.scan_index_ascan = scan_index

            self.last_sel_scan_index = scan_index

        # tooltip----------------
        speccommand = self.frameparent.scancommand
        date = self.frameparent.scan_date
        # print('speccommand', speccommand)
        # print('date',date)
        tooltip = "%s\ndate: %s" % (speccommand, date)
        #print('tooltip',tooltip)
        event.GetEventObject().SetToolTipString(tooltip)
        event.Skip()
        #------------------

        print("self.set_selected_indices", self.set_selected_indices)

class MainFrame(wx.Frame):
    """
    Class main GUI
    """
    def __init__(self, parent, _id, title):
        wx.Frame.__init__(self, parent, _id, title, size=(700, 500))

        self.folderpath_specfile, self.specfilename = None, None
        self.folderpath_hdf5file, self.hdf5filename = None, None

        self.filetype='spec'
        self.setdefaultcounters()
        self.normalizeintensity = False

        self.createMenuBar()

        self.create_main_panel()

        self.listmesh = None
        # filter of the scans of hdf5 logfiles (words of the dataset name, as in the quickstart notebook,
        # see map_workflow.scan_filter()), set when a heavy logfile is opened (see choose_lighter_logfile())
        self.hdf5_scanfilter = ""
        # 1D scans of hdf5 logfiles (listed when clicking on ASCAN), motor of the 1D plot (scans of several motors)
        self.listascan = []
        self.list_ascanscan_indices = []
        self.motor_ascan = None
        # data of the plotted 2D scan for the counters maps (see CountersMapsPanel)
        self.meshscan = None

    def setdefaultcounters(self):
        if self.filetype == 'spec':
            self.detectorname_mesh = "Monitor"
            self.detectorname_ascan = "Monitor"
            self.columns_name = ["Monitor", "fluo"]
        elif self.filetype == 'hdf5':
            self.detectorname_mesh = "mon"
            self.detectorname_ascan = "mon"
            self.columns_name = ["mon", "fluoCu"]
    def createMenuBar(self):
        menubar = wx.MenuBar()

        filemenu = wx.Menu()
        menuSpecFile = filemenu.Append(-1, "Open spec file", "Open a spec file")
        menuHdf5File = filemenu.Append(-1, "Open hdf5 file", "Open a hdf5 file (ESRF Bliss made)")
        menuSetPreference = filemenu.Append(-1, "Folder Preferences", "Set folder Preferences")
        savemeshdata = filemenu.Append(-1, "Save Data", "Save current 2D data")
        self.Bind(wx.EVT_MENU, self.OnOpenSpecFile, menuSpecFile)
        self.Bind(wx.EVT_MENU, self.OnOpenHdf5File, menuHdf5File)
        self.Bind(wx.EVT_MENU, self.OnSaveData, savemeshdata)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuSetPreference)

        #         displayprops = wx.Menu()
        #         menudisplayprops = displayprops.Append(-1, "Set Plot Size",
        #                                          "Set Minimal plot size to fit with small computer screen")
        #         self.Bind(wx.EVT_MENU, self.OnAbout, menudisplayprops)

        helpmenu = wx.Menu()

        menuAbout = helpmenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        menuExit = helpmenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Set events.
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

        menubar.Append(filemenu, "&File")
        #         menubar.Append(displayprops, '&Display Props')
        menubar.Append(helpmenu, "&Help")

        self.SetMenuBar(menubar)

    def OnAbout(self, _):
        pyversion='%s.%s'%(sys.version_info.major,sys.version_info.minor)
        wx.MessageBox('LaueTools module:\nPlot mesh scan recorded on logfile either by SPEC software (<2022) or BLISS hdf5 format (>2021)\n---------------\npython version : %s\nh5py version: %s'%(pyversion,h5py.__version__),'INFO')
        
    def OnExit(self, evt):
        pass

    def create_main_panel(self):
        """
        """
        self.panel = wx.Panel(self)

        z_values = np.arange(10 * 5).reshape((10, 5))  # + 10 * np.random.randn((5, 7))
        Imageindices = 708 + z_values
        posmotor = np.arange(10 * 5 * 2).reshape((10, 5, 2))

        #         z_values = None

        self.stbar = self.CreateStatusBar(4)

        self.stbar.SetStatusWidths([180, -1, -1, -1])
        #         print dir(self.stbar)

        self.stbar0 = wx.StatusBar(self.panel)

        self.plotmeshpanel = ImshowPanel(self.panel, -1, "test_plot", z_values,
                                Imageindices=Imageindices,
                                posmotorname=("xmotor", "ymotor"),
                                posarray_twomotors=posmotor,
                                absolute_motorposition_unit="mm",
                                filetype=self.filetype)

        self.plotascanpanel = PlotPanel(self.panel, -1, "test_plot", np.arange(150),
                                Imageindices=Imageindices,
                                posmotorname="xmotor",
                                posarray_motors=np.arange(150),
                                absolute_motorposition_unit="mm")

        # scans lists, one above the other with a movable bar between them (each one has the width of the
        # column, widened to the longest scan name, see TreePanel.fit_width())
        self.listssplitter = wx.SplitterWindow(self.panel, -1, style=wx.SP_LIVE_UPDATE | wx.SP_3D)
        self.treemesh = TreePanel(self.listssplitter, scantype="MESH", _id=0, frameparent=self,
                                  size=(TreePanel.MINWIDTH, -1))
        self.treeascan = TreePanel(self.listssplitter, scantype="ASCAN", _id=1, frameparent=self,
                                   size=(TreePanel.MINWIDTH, -1))
        self.listssplitter.SplitHorizontally(self.treemesh, self.treeascan)
        self.listssplitter.SetMinimumPaneSize(60)  # pixels: each list stays visible
        self.listssplitter.SetSashGravity(0.5)  # extra height shared by both lists when the window is resized
        self.listssplitter.SetMinSize((TreePanel.MINWIDTH, 300))
        self.listssplitter.SetToolTip("Drag the bar between the 2D and 1D scans lists to share their height")

        self.updatelistbtn = wx.Button(self.panel, -1, "Update scans list")
        self.updatelistbtn.Bind(wx.EVT_BUTTON, self.onUpdateSpecFile)

        self.txtselectionmode = wx.StaticText(self.panel,-1,'Selection mode: Single')

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.toggleBtn = wx.Button(self.panel, wx.ID_ANY, "Real Time Plot")
        self.toggleBtn.Bind(wx.EVT_BUTTON, self.onToggle)

        # maps of the counters (fluo..., eiger4m_...) of the 2D scan at the place of the 1D plot
        self.countersmapsbtn = wx.ToggleButton(self.panel, wx.ID_ANY, "Counters maps")
        self.countersmapsbtn.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleCountersMaps)
        self.countersmapsbtn.SetToolTipString("Show at the place of the 1D plot the maps of the counters of the "
                                              "selected 2D scan (names starting with fluo, eiger4m_, ...)")
        self.countersmaps = CountersMapsPanel(self.panel, self)

        # --- ----------tooltip
        self.updatelistbtn.SetToolTipString("Refresh list of scan from spec file")
        self.toggleBtn.SetToolTipString("On/Off Real time plot")
        self.txtselectionmode.SetToolTipString("Press s to toggle single/multi ascan selection")
        # --- ----------layout
        # hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        # hbox0.Add(self.treemesh, 1, wx.LEFT | wx.TOP | wx.GROW)
        # hbox0.Add(self.treeascan, 0, wx.EXPAND)
        

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.listssplitter, 1, wx.LEFT | wx.TOP | wx.GROW)
        vbox.Add(self.updatelistbtn, 0, wx.BOTTOM)
        vbox.Add(self.txtselectionmode, 0, wx.BOTTOM)
        vbox.Add(self.toggleBtn, 0, wx.BOTTOM)
        vbox.Add(self.countersmapsbtn, 0, wx.BOTTOM)
        #         vbox.Add(self.stopbtn, 0, wx.BOTTOM)

        # scans lists: width of their longest scan name; extra width of the window for the plots
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(vbox, 0, wx.LEFT | wx.TOP | wx.GROW)#wx.EXPAND)
        self.hbox.Add(self.plotmeshpanel, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.hbox.Add(self.plotascanpanel, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.hbox.Add(self.countersmaps, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.hbox.Hide(self.countersmaps)

        bigvbox = wx.BoxSizer(wx.VERTICAL)
        bigvbox.Add(self.hbox, 1, wx.LEFT | wx.TOP | wx.GROW)
        bigvbox.Add(self.stbar0, 0, wx.EXPAND)

        self.panel.SetSizer(bigvbox)
        bigvbox.Fit(self)
        # wide enough for the rows of controls of the plots on one line, if the screen allows it
        self.SetSize((min(self.screen_width(), self.COMFORTABLE_WIDTH), -1))
        self.Layout()
        # bar between the scans lists at the middle once the window is laid out (the splitter was split small)
        wx.CallAfter(lambda: self.listssplitter.SetSashPosition(self.listssplitter.GetSize()[1] // 2))

    COMFORTABLE_WIDTH = 1750  # pixels

    def screen_width(self):
        """width of the screen (client area) where the window is"""
        index = wx.Display.GetFromWindow(self)
        return wx.Display(index if index != wx.NOT_FOUND else 0).GetClientArea().width

    def OnSaveData(self, _):

        defaultdir = ""
        if not os.path.isdir(defaultdir):
            defaultdir = os.getcwd()

        file = wx.FileDialog(self,
                            "Save 2D Array Data in File",
                            defaultDir=defaultdir,
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if file.ShowModal() == wx.ID_OK:

            outputfile = file.GetPath()

            #             currentfig = self.plotmeshpanel.fig
            #             currentfig.savefig(outputfile)
            #             print "Image saved in ", outputfile + '.png'

            #         self.flat_data_z_values = data[detectorname]
            #         self.flat_motor1 = posmotor1
            #         self.flat_motor2 = posmotor2

            self.writefile_3columns(outputfile, [self.flat_data_z_values.tolist(),
                                                self.flat_motor1.tolist(),
                                                self.flat_motor2.tolist()])

    def writefile_3columns(self, output_filename, data):
        """
        Write  file containing data = [list_1,list_2,list_3]
        """
        longueur = len(data[0])

        outputfile = open(output_filename, "w")

        outputfile.write("data_z posmotor1 posmotor2\n")

        outputfile.write("\n".join(
                ["%.06f   %.06f   %06f" % tuple(list(zip(data[0], data[1], data[2]))[i])
                    for i in range(longueur)]))

        # outputfile.write(
        #     "\n".join(
        #         ["%.02f   %.02f   %.02f   %.02f   %.02f   %.02f    %.03f   %.02f   %.02f   %.02f   %d"
        #             % tuple(list(zip(peak_X.round(decimals=2),
        #                         peak_Y.round(decimals=2),

        outputfile.write("\n# File created at %s with PlotmeshspecGUI.py" % (time.asctime()))

        outputfile.close()
        print("Data written in %s" % output_filename)

    def askUserForFilename(self, **dialogOptions):
        dialog = wx.FileDialog(self, **dialogOptions)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True
            self.filename = dialog.GetFilename()
            self.dirname = dialog.GetDirectory()
            print(self.filename)
            print(self.dirname)

        else:
            userProvidedFilename = False
        dialog.Destroy()
        return userProvidedFilename

    def onUpdateSpecFile(self, _):
        """ update scan list  """

        self.folderpath_specfile, self.specfilename = os.path.split(self.fullpath_specfile)

        print("self.listmesh before", self.listmesh)
        list_lastmeshscan_indices = []
        for ms in self.listmesh:
            list_lastmeshscan_indices.append(ms[0])
        lastscan_listmesh = max(list_lastmeshscan_indices)

        print("lastscan_listmesh", lastscan_listmesh)

        listmeshall = getmeshscan_from_specfile(self.fullpath_specfile)

        self.listmeshtoAdd = []
        for ms in listmeshall:
            if ms[0] not in list_lastmeshscan_indices:
                self.listmesh.append(ms)
                self.listmeshtoAdd.append(ms)

        print("self.listmesh after", self.listmesh)

        wx.CallAfter(self.fill_tree)

        self.treemesh.set_selected_indices = set()
        self.treeascan.set_selected_indices = set()


    def OnOpenHdf5File(self, _):
        """ in menu :  File/open hdf5 file  """
        self.filetype = 'hdf5'
        self.setdefaultcounters()

        folder = wx.FileDialog(self, "Select hdf5 file",
                                wildcard="Bliss ESRF hdf5 (*.h5)|*.h5|All files(*)|*",
                                defaultDir=str(self.folderpath_hdf5file),
                                defaultFile=str(self.hdf5filename))

        self.last_hdf5filename = self.hdf5filename
        if folder.ShowModal() == wx.ID_OK:

            fullpath = self.choose_lighter_logfile(folder.GetPath())
            if fullpath is None:  # heavy logfile dialog cancelled
                return
            self.fullpath_hdf5file = fullpath
            self.folderpath_hdf5file, self.hdf5filename = os.path.split(self.fullpath_hdf5file)

        # simply update list fo scans
        if (self.hdf5filename == self.last_hdf5filename and self.hdf5filename is not None):
            self.onUpdatehdf5File(1)

        self.get_listmesh()

        # self.get_listascan(self.fullpath_hdf5file)
        self.listascan = []

        self.list_ascanscan_indices = []

        if (self.hdf5filename != self.last_hdf5filename and self.hdf5filename is not None):
            #print("\n\ndeleting last old items\n\n")
            self.treemesh.tree.DeleteAllItems()
            wx.CallAfter(self.treemesh.maketree)

            self.treeascan.tree.DeleteAllItems()
            wx.CallAfter(self.treeascan.maketree)

        self.listmeshtoAdd = self.listmesh
        self.listascantoAdd = self.listascan
        wx.CallAfter(self.fill_tree)

    def choose_lighter_logfile(self, fullpath):
        """fullpath, or the logfile of a sample or dataset chosen in a dialog when fullpath has more than
        logfile_reader.HEAVY_LOGFILE_NBSCANS scans (slow listing). None if the dialog is cancelled."""
        try:
            nbscans = logfile_reader.nb_scans_in_logfile(fullpath)
        except OSError:  # not an hdf5 file: reported when listing the scans
            return fullpath
        if nbscans <= logfile_reader.HEAVY_LOGFILE_NBSCANS:
            return fullpath
        wx.BeginBusyCursor()
        try:
            candidates = logfile_reader.lighter_logfiles(fullpath)
        finally:
            wx.EndBusyCursor()
        if not candidates:
            return fullpath
        seconds = nbscans * logfile_reader.LISTING_SECONDS_PER_SCAN
        folder, filename = os.path.split(fullpath)
        choices = ["open %s anyway (%d scans, up to about %d s)" % (filename, nbscans, seconds),
                   "open %s, only the datasets whose name contains some words (filter)" % filename]
        choices += [os.path.relpath(path, folder) + ("   (%d scans)" % n if n is not None else "")
                    for path, n in candidates]
        dlg = wx.SingleChoiceDialog(self, "%s contains %d scans: listing them takes up to about %d s.\n"
                                    "Filter them, or open instead a logfile of a sample or dataset (most recent first)?"
                                    % (filename, nbscans, seconds), "Heavy logfile", choices)
        dlg.SetSelection(2)
        chosen = None
        self.hdf5_scanfilter = ""
        if dlg.ShowModal() == wx.ID_OK:
            index = dlg.GetSelection()
            if index == 0:
                chosen = fullpath
            elif index == 1:
                filterdlg = wx.TextEntryDialog(self, "Words of the dataset name (case insensitive), '-word': "
                                               "without it.\ne.g.  map2D -align", "Filter of the scans")
                if filterdlg.ShowModal() == wx.ID_OK:
                    self.hdf5_scanfilter = filterdlg.GetValue().strip()
                    chosen = fullpath
                filterdlg.Destroy()
            else:
                chosen = candidates[index - 2][0]
        dlg.Destroy()
        return chosen

    def OnOpenSpecFile(self, _):
        """ in menu :  File/open spec file  """
        self.filetype = 'spec'

        self.setdefaultcounters()
        folder = wx.FileDialog(self,
                                "Select spec file",
                                wildcard="BM32 Specfile (laue.*)|laue.*|All files(*)|*",
                                defaultDir=str(self.folderpath_specfile),
                                defaultFile=str(self.specfilename))

        self.last_specfilename = self.specfilename
        if folder.ShowModal() == wx.ID_OK:

            self.fullpath_specfile = folder.GetPath()
            self.folderpath_specfile, self.specfilename = os.path.split(self.fullpath_specfile)

        # simply update list fo scans
        if (self.specfilename == self.last_specfilename and self.specfilename is not None):
            self.onUpdateSpecFile(1)

        self.get_listmesh()

        self.get_listascan(self.fullpath_specfile)

        if (self.specfilename != self.last_specfilename and self.specfilename is not None):
            print("\n\ndeleting last old items\n\n")
            self.treemesh.tree.DeleteAllItems()
            wx.CallAfter(self.treemesh.maketree)

            self.treeascan.tree.DeleteAllItems()
            wx.CallAfter(self.treeascan.maketree)

        self.listmeshtoAdd = self.listmesh
        self.listascantoAdd = self.listascan
        wx.CallAfter(self.fill_tree)

    def fill_tree(self):
        self.append_scan_items(self.treemesh, self.listmeshtoAdd)
        self.append_scan_items(self.treeascan, self.listascantoAdd)

    def append_scan_items(self, treepanel, listprops):
        """add the scans of listprops to a scans list, colored by end reason (hdf5 logfiles, see endreason_colour())"""
        for props in listprops:
            item = treepanel.tree.AppendItem(treepanel.root, str(props[0]))
            colour = endreason_colour(props[7]) if self.filetype == 'hdf5' and len(props) > 7 else None
            if colour is not None:
                treepanel.tree.SetItemTextColour(item, colour)
        treepanel.fit_width()
        # the splitter window does not take the minimum widths of its lists into account
        self.listssplitter.SetMinSize((max(self.treemesh.GetMinSize()[0], self.treeascan.GetMinSize()[0]), 300))
        self.fit_scans_lists()

    def OnToggleCountersMaps(self, _):
        """show the maps of the counters of the 2D scan instead of the 1D plot, or the 1D plot again"""
        show = self.countersmapsbtn.GetValue()
        self.hbox.Show(self.plotascanpanel, not show)
        self.hbox.Show(self.countersmaps, show)
        if show:
            self.countersmaps.set_scan(self.meshscan)
        self.panel.Layout()

    def monitor_normalization(self, data, tit):
        """divisor of the counters of a scan for the normalization by monitor (monitor counts per second minus
        offset), None if the scan has no monitor"""
        offset = getattr(self, "MonitorOffset", 0.)
        if self.filetype == 'spec':
            if "Monitor" not in data or "Seconds" not in data:
                return None
            return np.asarray(data["Monitor"], dtype=float) / np.asarray(data["Seconds"], dtype=float) - offset
        monitorname = logfile_reader.find_monitor_counter(data)
        if monitorname is None:
            return None
        monitor = np.asarray(data[monitorname], dtype=float)
        exposure = data.get("integration_time")
        if exposure is None or len(exposure) != len(monitor):
            exposure = scan_exposure_time(tit)
        return monitor / np.asarray(exposure, dtype=float) - offset

    def fit_scans_lists(self):
        """layout after a change of the scans lists widths: window widened if it is too narrow to show them"""
        self.panel.Layout()
        best = self.panel.GetSizer().GetMinSize()
        width, height = self.GetClientSize()
        newwidth = min(best.width, self.screen_width() - (self.GetSize()[0] - width))  # not wider than the screen
        if width < newwidth:
            self.SetClientSize((newwidth, max(height, best.height)))
            self.panel.Layout()

    def scan_tooltip(self, scantype, key):
        """tooltip of a scan of the MESH or ASCAN list: command, light date and duration, end reason"""
        listprops = self.listmesh if scantype == 'MESH' else self.listascan
        for props in (listprops if listprops is not None else []):
            if str(props[0]) != key:
                continue
            if self.filetype != 'hdf5' or len(props) < 8:  # SPEC file: '#S n command' line
                return str(props[-1]).strip()
            command = str(props[5]).split(' ', 1)[-1]
            return "%s\n%s\n%s" % (command, scan_dates_text(props[3], props[4]), props[7] or "end reason unknown")
        return None


    def get_listmesh(self):
        """ retrieve list of mesh in spec or hdf5 file  from self.fullpath_specfile or self.fullpath_hdf5file"""
        samefile = False
        if self.filetype=='spec':
            fullpathfilename = self.fullpath_specfile
            if self.specfilename != self.last_specfilename:
                samefile = True
                self.listmesh = None
        elif self.filetype == 'hdf5':
            fullpathfilename = self.fullpath_hdf5file
            if self.hdf5filename != self.last_hdf5filename:
                samefile = True
                self.listmesh = None

        lastscan_listmesh = 0
        list_lastmeshscan_indices = []
        if self.listmesh is not None:
            print("self.listmesh already exists")
            print("self.listmesh", self.listmesh)
            list_lastmeshscan_indices = []
            for ms in self.listmesh:
                list_lastmeshscan_indices.append(ms[0])
            lastscan_listmesh = max(list_lastmeshscan_indices)

        print("lastscan_listmesh", lastscan_listmesh)

        if self.filetype == 'spec':
            listmeshall = getmeshscan_from_specfile(fullpathfilename)

            if len(listmeshall) == 0:
                print('No mesh scan in %s'%fullpathfilename)
                self.listmesh = []
                return

            list_meshscan_indices = []
            for ms in listmeshall:
                if ms[0] not in list_lastmeshscan_indices:
                    list_meshscan_indices.append(ms[0])

            # print("list_meshscan_indices", list_meshscan_indices)

            if list_meshscan_indices[-1] != lastscan_listmesh and not samefile:
                print("adding only new meshes from file %s" % self.fullpath_specfile)
                indstart_newmeshes = np.searchsorted(list_meshscan_indices, lastscan_listmesh - 1)
            else:
                indstart_newmeshes = 0

            # print("listmeshall", listmeshall)
            # print("indstart_newmeshes", indstart_newmeshes)

            self.listmesh = listmeshall[indstart_newmeshes:]
            self.list_meshscan_indices = list_meshscan_indices

            print('self.listmesh', self.listmesh)
            print('self.list_meshscan_indices', self.list_meshscan_indices)
        
        if self.filetype == 'hdf5':
            listmeshall = self.list_hdf5_scans(fullpathfilename, onlymesh=True)

            if len(listmeshall) == 0:
                print('No mesh scan in %s'%fullpathfilename)
                self.listmesh = []
                return

            list_meshscan_indices = []
            for ms in listmeshall:
                if ms[0] not in list_lastmeshscan_indices:
                    list_meshscan_indices.append(ms[0])

            # print("list_meshscan_indices", list_meshscan_indices)

            # if list_meshscan_indices[-1] != lastscan_listmesh and not samefile:
            #     print("adding only new meshes from file %s" % self.fullpath_hdf5file)
            #     indstart_newmeshes = np.searchsorted(list_meshscan_indices, lastscan_listmesh - 1)
            # else:
            #     indstart_newmeshes = 0

            # print("listmeshall", listmeshall)
            # print("indstart_newmeshes", indstart_newmeshes)

            self.listmesh = listmeshall
            self.list_meshscan_indices = list_meshscan_indices

            print('hdf5 self.listmesh', self.listmesh)
            print('hdf5 self.list_meshscan_indices', self.list_meshscan_indices)

    def list_hdf5_scans(self, fullpathfilename, onlymesh=False, onlyscan1D=False):
        """meshes or 1D scans of a hdf5 logfile whose names pass self.hdf5_scanfilter (names filtered before
        reading the scans, as map_workflow.list_maps(): faster for large logfiles)"""
        flt = MW.scan_filter(self.hdf5_scanfilter)
        return logfile_reader.getscans_from_hdf5file(
            fullpathfilename, collectallscans=False, onlymesh=onlymesh, onlyscan1D=onlyscan1D,
            n_last=int(flt['last']) if flt['last'] else None, node_prefix=flt['prefix'] or None,
            node_substrings=flt['include'] or None, substrings_logic=flt['logic'],
            exclude_substrings=flt['exclude'] or None,
            node_filter=(lambda name: MW._words_match(name, flt['contains'])) if flt['contains'] else None)[0]

    def list_hdf5_scans1D(self):
        """list in the ASCAN tree the 1D scans (ascan, a2scan, ..., dscan, fscan) of the hdf5 logfile"""
        if self.filetype != 'hdf5' or not getattr(self, 'fullpath_hdf5file', None):
            return
        wx.BeginBusyCursor()
        try:
            self.listascan = self.list_hdf5_scans(self.fullpath_hdf5file, onlyscan1D=True)
        finally:
            wx.EndBusyCursor()
        self.list_ascanscan_indices = [props[0] for props in self.listascan]
        self.treeascan.tree.DeleteAllItems()
        self.treeascan.maketree()
        self.append_scan_items(self.treeascan, self.listascan)
        self.stbar0.SetStatusText("%d 1D scans in %s%s" % (len(self.listascan), self.hdf5filename,
                                  " (filter: %s)" % self.hdf5_scanfilter if self.hdf5_scanfilter else ""))

    def get_listascan(self, fullpathspecfilename):
        samefile = False
        if self.specfilename != self.last_specfilename:
            samefile = True
            self.listascan = None

        lastscan_listascan = 0
        list_lastascan_indices = []
        if self.listascan is not None:
            print("self.listascan already exists")
            print("self.listascan", self.listascan)
            list_lastascan_indices = []
            for ms in self.listascan:
                list_lastascan_indices.append(ms[0])
            lastscan_listascan = max(list_lastascan_indices)

        print("lastscan_listascan", lastscan_listascan)

        listascanall = getascan_from_specfile(fullpathspecfilename)

        if len(listascanall) == 0:
            print('No mesh scan in %s'%fullpathspecfilename)
            self.listascan = []
            return

        list_ascanscan_indices = []
        for ms in listascanall:
            if ms[0] not in list_lastascan_indices:
                list_ascanscan_indices.append(ms[0])

        print("list_ascanscan_indices", list_ascanscan_indices)

        if list_ascanscan_indices[-1] != lastscan_listascan and not samefile:
            print("adding only new ascanes from file %s" % self.fullpath_specfile)
            indstart_newascanes = np.searchsorted(list_ascanscan_indices, lastscan_listascan - 1)
        else:
            indstart_newascanes = 0

        print("listascanall", listascanall)
        print("indstart_newascanes", indstart_newascanes)
        self.listascan = listascanall[indstart_newascanes:]

        self.list_ascanscan_indices = list_ascanscan_indices

    def ReadMultipleScans(self, scan_indices, resetlistcounters=True):
        """
        read a multiple scans data in spec file and fill data for a updated figure plot
        """
        #superimposition of ASCAN only
        detectorname_ascan = self.detectorname_ascan
        detectorname_mesh = self.detectorname_mesh

        motorselected = None

        # trial ----------------
        # scattermode = True
        scattermode = False
        # -------------------

        zvalues = []
        motorsvalues = []
        scanindexvalues = []

        #self.fullpath_specfile = '/home/micha/LaueProjects/Laue_SpecLogFiles/laue.28Nov18'
        #scan_indices = np.arange(12,20+1)

        for k, scan_idx in enumerate(scan_indices):

            scanheader, data, self.scan_date = ReadSpec(self.fullpath_specfile, scan_idx, outputdate=True)
            tit = str(scanheader)

            self.scantype = tit.split()[2]

            if self.scantype not in ('ascan',):
                continue

            titlesplit = tit.split()
            movingmotor = tit.split()[3]
            # motor names
            if k == 0:
                motorselected = movingmotor
            motor2 = None

            if movingmotor != motorselected:
                print("moving motor of scan %d (%s )is not %s!"%(scan_idx, movingmotor,motorselected))
                continue

            scanindexvalues.append(scan_idx)

            # motor positions
            posmotor1 = np.fix(data[motorselected] * 100000) / 100000

            # nb of steps in both directions
            nb1 = int(tit.split()[6]) + 1

            # current nb of collected points in the mesh
            nbacc = len(data[list(data.keys())[0]])
            print("nb of points accumulated  :", nbacc)

            counterintensity1D = data[detectorname_ascan]

            if self.normalizeintensity:
                data_I0 = data["Monitor"]
                exposureTime = data["Seconds"]
                datay = counterintensity1D

                # self.MonitorOffset  in counts / sec

                counterintensity1D = normalized_by_monitor(datay, data_I0, exposureTime, self.MonitorOffset)

            print("building arrays [multiple scans]")
            if nb1 == nbacc:
                print("scan is finished")
                data_z_values = counterintensity1D
                try:
                    data_img = data["img"]
                except KeyError:
                    print("'img' column doesn't exist! Add fake dummy 0 value")
                    data_img = np.zeros(nb1)
                posmotorsinfo = np.array(posmotor1)
                scan_in_progress = False

            else:
                print("scan has been aborted")
                print("filling data with zeros...")
                # intensity array
                zz = np.zeros(nb1)
                zz.put(range(min(nbacc, len(counterintensity1D))), counterintensity1D[:nbacc])  # (counters of unfinished scans may have one value less)
                data_z_values = zz
                # image index array
                data_img = np.zeros(nb1)
                try:
                    data_img.put(range(nbacc), data["img"])
                except KeyError:
                    print("'img' column doesn't exist! Add fake dummy 0 value")
                    data_img.put(range(nbacc), 0)
                # motors positions
                ar_posmotor1 = np.zeros(nb1)
                ar_posmotor1.put(range(nbacc), posmotor1)
                #                     ar_posmotor1 = reshape(ar_posmotor1, (nb2, nb1))

                posmotorsinfo = np.array(ar_posmotor1)

                scan_in_progress = True

            print('\n================ compilation of scan data ============')
            zvalues.append(data_z_values)
            motorsvalues.append(posmotorsinfo)


        #AddedArrayInfo = data_img

        #datatype = "scalar"

        Apptitle = "%s\n Multiple ascan #%s" % (self.specfilename, str(scanindexvalues))

        #print("title", Apptitle)

        #--------------------
        self.flat_data_z_values = None #counterintensity1D
        self.flat_motor1 = None #motorselected

        self.scancommand = tit
        self.minmotor1 = float(titlesplit[4])
        self.maxmotor1 = float(titlesplit[5])
        scancommandextremmotorspositions = [self.minmotor1,
                                            self.maxmotor1]

        if not scattermode:
            self.update_fig_1D(zvalues,
                            motorsvalues,
                            motorselected,
                            Apptitle,
                            data_img,
                            detectorname_ascan,
                            scancommandextremmotorspositions,
                            multipleplots=True,
                            listscanindex=scanindexvalues)
        else:  #scattermode


            print('zvalues', zvalues)
            print('motorsvalues', motorsvalues)

            self.update_fig_2D(zvalues,
                                motorsvalues,
                                motorselected,
                                'scan index',
                                Apptitle,
                                data_img,
                                detectorname_ascan,
                                scancommandextremmotorspositions,
                                imshowmode=False,
                                listscanindex=scanindexvalues)

        if resetlistcounters:
            # counter and key name of data
            columns_name = list(data.keys())
            columns_name = sorted(columns_name)
            set_counters_menu(self.plotascanpanel.combocounters, columns_name, detectorname_ascan)

        return scan_in_progress

    def ReadScan_SpecFile(self, scan_index, resetlistcounters=True):
        """
        read a SINGLE scan data in spec file or hdf5 and fill data for a updated figure plot

        it can be SPEC or HDF5 file

        scan_index : int for spec file  or str for key in hdf5 file
        

        requires:
        self.fullpath_specfile
        self.detectorname_ascan
        self.detectorname_mesh
        self.normalizeintensity
        self.MonitorOffset
        self.specfilename
        self.plotascanpanel

        sets:
        self.scantype
        self.flat_data_z_values
        self.flat_motor1

        self.scancommand 
        self.minmotor1
        self.maxmotor1
        and for meshscan
        self.minmotor2 = float(titlesplit[8])
            self.maxmotor2 = float(titlesplit[9])
        """
        detectorname_ascan = self.detectorname_ascan
        detectorname_mesh = self.detectorname_mesh

        self.scan_subtitle = ""  # scan folder (from RAW_DATA) above the plots titles
        self.scan_folder_path = None  # scan folder (or dataset folder) of hdf5 scans, see figures_folder()
        if self.filetype=='spec':
            scanheader, data, self.scan_date = ReadSpec(self.fullpath_specfile, scan_index, outputdate=True)
            tit = str(scanheader)
            self.scantype = tit.split()[2]

        elif self.filetype == 'hdf5':  # only mesh now
            print('in ReadScan_SpecFile for hdf5 branch')

            print('scan_index',scan_index)
            scan1D = scan_index in self.list_ascanscan_indices  # from the ASCAN list, else from the MESH list
            listprops = self.listascan if scan1D else self.listmesh
            tit, data, posmotors, fullpath, self.scan_date = readdata_from_hdf5key(listprops, scan_index,
                                                                                    outputdate=True)
            scanindex = next((props[1] for props in listprops if str(props[0]) == str(scan_index)), None)
            folder = logfile_reader.scan_folder(fullpath, scanindex) if scanindex is not None else None
            if folder is not None:
                self.scan_subtitle = logfile_reader.path_from_raw_data(folder)
            self.scan_folder_path = folder or os.path.dirname(str(fullpath))
            print('tit',tit)
            # scanheader, data, self.scan_date = ReadHdf5(self.fullpath_hdf5file, scan_index, outputdate=True)

            self.scantype = 'scan1D' if scan1D else 'amesh'

        # print("splec/bliss" command    :", tit)
        # print("scan type  :", self.scantype)

        Apptitle = ''

        if self.scantype == 'scan1D':
            return self.plot_hdf5_scan1D(scan_index, tit, data, resetlistcounters)

        if self.scantype in ('ascan',):

            titlesplit = tit.split()

            # motor names
            motor1 = tit.split()[3]
            motor2 = None

            # motor positions
            posmotor1 = np.fix(data[motor1] * 100000) / 100000

            # nb of steps in both directions
            nb1 = int(tit.split()[6]) + 1

            # current nb of collected points in the mesh
            nbacc = len(data[list(data.keys())[0]])
            print("nb of points accumulated  :", nbacc)


            if detectorname_ascan not in data:
                Apptitle += '%s NOT AVAILABLE\n'%detectorname_ascan
                detectorname_ascan = 'Monitor'

            counterintensity1D = data[detectorname_ascan]

            if self.normalizeintensity:
                data_I0 = data["Monitor"]
                exposureTime = data["Seconds"]
                datay = counterintensity1D

                # self.MonitorOffset  in counts / sec

                counterintensity1D = normalized_by_monitor(datay, data_I0, exposureTime, self.MonitorOffset)

            print("building arrays")
            print('nb1',nb1,'nbacc',nbacc)
            if nb1 == nbacc:
                print("scan is finished")
                data_z_values = counterintensity1D
                if 'img' in data:
                    data_img = data["img"]
                elif 'img_scmos' in data:
                    data_img = data["img_scmos"]
                else:
                    print("'img' column doesn't exist! Add fake dummy 0 value")
                    data_img = np.zeros(nb1)
                posmotorsinfo = np.array(posmotor1)
                scan_in_progress = False

            else:
                print("scan has been aborted")
                print("filling data with zeros...")
                # intensity array
                zz = np.zeros(nb1)
                zz.put(range(min(nbacc, len(counterintensity1D))), counterintensity1D[:nbacc])  # (counters of unfinished scans may have one value less)
                data_z_values = zz
                # image index array
                data_img = np.zeros(nb1)
                if 'img' in data:
                    data_img.put(range(nbacc), data["img"])
                elif 'img_scmos' in data:
                    data_img.put(range(nbacc), data["img_scmos"])
                else:
                    print("'img' column doesn't exist! Add fake dummy 0 value")
                    data_img.put(range(nbacc), 0)
                # motors positions
                ar_posmotor1 = np.zeros(nb1)
                ar_posmotor1.put(range(nbacc), posmotor1)
                #                     ar_posmotor1 = reshape(ar_posmotor1, (nb2, nb1))

                posmotorsinfo = np.array(ar_posmotor1)

                scan_in_progress = True

            AddedArrayInfo = data_img

            datatype = "scalar"

            Apptitle += "%s #%d\n%s" % (self.specfilename, scan_index, " ".join(tit.split()[2:]))

            # print("Apptitle ", Apptitle)

            self.flat_data_z_values = counterintensity1D
            self.flat_motor1 = posmotor1

            self.scancommand = tit
            self.minmotor1 = float(titlesplit[4])
            self.maxmotor1 = float(titlesplit[5])

            scancommandextremmotorspositions = [self.minmotor1,
                                                self.maxmotor1]

            if resetlistcounters:
                # counter and key name of data
                columns_name = list(data.keys())
                columns_name = sorted(columns_name)
                set_counters_menu(self.plotascanpanel.combocounters, columns_name, detectorname_ascan)

            self.update_fig_1D(data_z_values,
                                posmotorsinfo,
                                motor1,
                                Apptitle,
                                data_img,
                                detectorname_ascan,
                                scancommandextremmotorspositions)

            return scan_in_progress

        elif self.scantype in ('mesh','amesh'):

            titlesplit = tit.split()
            # minmotor1 = float(titlesplit[4])
            # maxmotor1 = float(titlesplit[5])
            # minmotor2 = float(titlesplit[8])
            # maxmotor2 = float(titlesplit[9])

            if self.filetype=='spec':
                # motor names
                motor1 = tit.split()[3]
                motor2 = tit.split()[7]


                # nb of steps in both directions
                nb1 = int(tit.split()[6]) + 1
                nb2 = int(tit.split()[10]) + 1
            else:
                # amesh, dmesh, fscan2d (fscan2d: start step npoints, first motor is the slow one)
                dictcommand = logfile_reader.read_fullcommand(tit)
                motor1, motor2 = dictcommand['fastmotor'], dictcommand['slowmotor']
                nb1, nb2 = dictcommand['npts_fast'], dictcommand['npts_slow']

            # motor positions
            posmotor1 = np.fix(data[motor1] * 100000) / 100000
            posmotor2 = np.fix(data[motor2] * 100000) / 100000
            # current nb of collected points in the mesh (counters such as Nbunch_raw of fscan2d have other lengths)
            nbacc = len(data[motor1]) if self.filetype == 'hdf5' else len(data[list(data.keys())[0]])
            print("nb of points accumulated  :", nbacc)

            if detectorname_mesh not in data:
                Apptitle += '%s NOT AVAILABLE\n'%detectorname_mesh
                if self.filetype == 'spec':
                    detectorname_mesh = 'Monitor'
                
                else:
                    detectorname_mesh = logfile_reader.find_monitor_counter(data) or 'epoch'

            counterintensity1D = data[detectorname_mesh]

            if self.normalizeintensity:
                if self.filetype == 'spec':
                    data_I0 = data["Monitor"]
                    exposureTime = data["Seconds"]
                else:  # monitor found by name; fscan2d has no integration_time counter: exposure of the command
                    monitorname = logfile_reader.find_monitor_counter(data)
                    if monitorname is None:
                        wx.MessageBox("No monitor counter (mon, i0, io, ...) in this scan: data not normalized",
                                      "Normalization")
                        data_I0, exposureTime = 1., self.MonitorOffset + 1.  # datay / 1
                    else:
                        data_I0 = data[monitorname]
                        exposureTime = data.get("integration_time", dictcommand.get('expotime', 1.))
                datay = counterintensity1D

                # self.MonitorOffset  in counts / sec

                counterintensity1D = normalized_by_monitor(datay, data_I0, exposureTime, self.MonitorOffset)

            print("building arrays")
            if nb2 * nb1 == nbacc:
                print("scan is finished")
                data_z_values = np.reshape(counterintensity1D, (nb2, nb1))
                imagecounter = logfile_reader.find_image_counter(data)
                if imagecounter is not None:
                    data_img = np.reshape(data[imagecounter], (nb2, nb1))
                else:  # e.g. EIGER: image index = point index
                    data_img = np.arange(nb2 * nb1).reshape((nb2, nb1))
                posmotorsinfo = np.reshape(np.array([posmotor1, posmotor2]).T, (nb2, nb1, 2))
                scan_in_progress = False

            else:
                print("scan has been aborted")
                print("filling data with zeros...")
                # intensity array
                zz = np.zeros(nb2 * nb1)
                zz.put(range(min(nbacc, len(counterintensity1D))), counterintensity1D[:nbacc])  # (counters of unfinished scans may have one value less)
                data_z_values = np.reshape(zz, (nb2, nb1))
                # image index array
                data_img = np.arange(nb2 * nb1, dtype=float)  # image index = point index (e.g. EIGER)
                imagecounter = logfile_reader.find_image_counter(data)
                if imagecounter is not None:
                    data_img.put(range(nbacc), data[imagecounter])
                data_img = np.reshape(data_img, (nb2, nb1))
                # motors positions
                ar_posmotor1 = np.zeros(nb2 * nb1)
                ar_posmotor1.put(range(nbacc), posmotor1)
                #                     ar_posmotor1 = reshape(ar_posmotor1, (nb2, nb1))

                ar_posmotor2 = np.zeros(nb2 * nb1)
                ar_posmotor2.put(range(nbacc), posmotor2)
                #                     ar_posmotor2 = reshape(ar_posmotor2, (nb2, nb1))

                posmotorsinfo = np.array([ar_posmotor1, ar_posmotor2]).T

                posmotorsinfo = np.reshape(posmotorsinfo, (nb2, nb1, 2))
                scan_in_progress = True

            AddedArrayInfo = data_img

            datatype = "scalar"

            #         print "bothmotors", posmotorsinfo
            #         print 'nb2,nb1', nb2, nb1
            #         print posmotorsinfo.shape

            #         print "posmotorsinfo", posmotorsinfo
            if self.filetype=='spec':
                Apptitle += "%s #%d\n%s" % (self.specfilename, scan_index, " ".join(tit.split()[2:]))
            else:
                Apptitle += "%s\n%s" % (scan_index, tit)  # scan name, BLISS command
                
            #print("title", Apptitle)

            self.flat_data_z_values = counterintensity1D
            self.flat_motor1 = posmotor1
            self.flat_motor2 = posmotor2

            self.scancommand = tit

            if self.filetype=='spec':
                self.minmotor1 = float(titlesplit[4])
                self.maxmotor1 = float(titlesplit[5])
                self.minmotor2 = float(titlesplit[8])
                self.maxmotor2 = float(titlesplit[9])
            else:
                self.minmotor1, self.maxmotor1 = dictcommand['fmotmin'], dictcommand['fmotmax']
                self.minmotor2, self.maxmotor2 = dictcommand['smotmin'], dictcommand['smotmax']

            scancommandextremmotorspositions = [self.minmotor1,
                                                self.maxmotor1,
                                                self.minmotor2,
                                                self.maxmotor2]

            if resetlistcounters:
                # counter and key name of data
                columns_name = list(data.keys())
                columns_name = sorted(columns_name)
                set_counters_menu(self.plotmeshpanel.combocounters, columns_name, detectorname_mesh)

            # all the counters of the scan for the counters maps (fast motor along the lines of the arrays)
            self.meshscan = dict(data=data, nb1=nb1, nb2=nb2, fastmotor=motor1, slowmotor=motor2,
                                 imageindices=data_img, scan_index=scan_index, command=tit,
                                 subtitle=self.scan_subtitle, scanfolder=self.scan_folder_path,
                                 divisor=self.monitor_normalization(data, tit) if self.normalizeintensity else None)
            if self.countersmapsbtn.GetValue():
                self.countersmaps.set_scan(self.meshscan)

            self.update_fig_2D(data_z_values,
                                posmotorsinfo,
                                motor1,
                                motor2,
                                Apptitle,
                                data_img,
                                detectorname_mesh,
                                scancommandextremmotorspositions,
                                imshowmode=True)

            return scan_in_progress

    def plot_hdf5_scan1D(self, scan_index, tit, data, resetlistcounters=True):
        """plot a counter of a 1D scan of hdf5 logfile (ascan, a2scan, ..., dscan, fscan) as a function of one of
        its motors (self.motor_ascan if it is a motor of the scan, else the first one). The counter
        self.detectorname_ascan is kept from one scan to another (monitor if the scan has no such counter)."""
        Apptitle = ''
        counters1D = {name: values for name, values in data.items() if np.ndim(values) == 1}
        motors = [motor for motor in logfile_reader.scan1D_motors(tit) if motor in counters1D]
        if not motors:  # no motor dataset: time
            motors = [name for name in ('elapsed_time', 'epoch') if name in counters1D][:1]
        if not motors:
            wx.MessageBox("No motor positions in scan %s:\n%s" % (scan_index, tit), "1D scan")
            return False
        motor = self.motor_ascan if self.motor_ascan in motors else motors[0]
        xvalues = np.asarray(counters1D[motor], dtype=float)
        nbpoints = len(xvalues)
        # counters with one value per point (in fscan, Nbunch_raw for example has more values)
        counters = sorted(name for name, values in counters1D.items() if len(values) == nbpoints)

        detectorname = self.detectorname_ascan
        if detectorname not in counters:
            Apptitle += '%s NOT AVAILABLE\n' % detectorname
            detectorname = logfile_reader.find_monitor_counter(counters) or counters[0]
        yvalues = np.asarray(counters1D[detectorname], dtype=float)

        if self.normalizeintensity:
            monitorname = logfile_reader.find_monitor_counter(counters)
            if monitorname is None:
                wx.MessageBox("No monitor counter (mon, i0, io, ...) in this scan: data not normalized",
                              "Normalization")
            else:
                exposure = counters1D.get("integration_time")
                if exposure is None or len(exposure) != nbpoints:
                    exposure = scan_exposure_time(tit)
                yvalues = yvalues / (np.asarray(counters1D[monitorname], dtype=float) / exposure
                                     - self.MonitorOffset)

        imagecounter = logfile_reader.find_image_counter(counters)
        imageindices = np.asarray(counters1D[imagecounter]) if imagecounter else np.arange(nbpoints)

        Apptitle += "%s\n%s" % (scan_index, tit)  # scan name, BLISS command
        self.scancommand = tit
        self.flat_data_z_values, self.flat_motor1 = yvalues, xvalues
        set_counters_menu(self.plotascanpanel.combocounters, counters if resetlistcounters else None, detectorname)
        self.plotascanpanel.set_motors(motors, motor)

        self.update_fig_1D(yvalues, xvalues, motor, Apptitle, imageindices, detectorname,
                           (np.nanmin(xvalues), np.nanmax(xvalues)))
        return False

    def update_fig_1D(self,
                    data_z_values,
                    posmotorsinfo,
                    motor1,
                    Apptitle,
                    data_img,
                    detectorname,
                    scancommandextremmotorspositions,
                    multipleplots=False,
                    listscanindex=None):
        """update for ascan fig and plot"""
        #         self.plot.fig.clear()

        self.plotascanpanel.data = data_z_values
        self.plotascanpanel.posarray_motors = posmotorsinfo
        self.plotascanpanel.motor1name = motor1
        self.plotascanpanel.posmotorname = motor1
        self.plotascanpanel.absolute_motorposition_unit = "mm"
        self.plotascanpanel.title = Apptitle
        self.plotascanpanel.subtitle = getattr(self, "scan_subtitle", "")
        self.plotascanpanel.detectorname = detectorname
        self.plotascanpanel.Imageindices = data_img
        self.plotascanpanel.multipleplots = multipleplots
        self.plotascanpanel.listscanindex = listscanindex


        (self.plotascanpanel.minmotor1, self.plotascanpanel.maxmotor1) = scancommandextremmotorspositions

        self.plotascanpanel.xylabels = ("column index", "row index")
        self.plotascanpanel.datatype = "scalar"

        if self.plotmeshpanel.colorbar is not None:
            self.plotmeshpanel.colorbar_label = detectorname
            (self.plotascanpanel.myplot, _, self.plotascanpanel.data) = makefig_update(
                self.plotascanpanel.fig, self.plotascanpanel.myplot, None, data_z_values, datadim=1)
        else:  #update for 1D ascan or multiple
            #print("self.plotascanpanel.colorbar is None")
            self.plotascanpanel.create_axes()
            self.plotascanpanel.data_to_Display = self.plotascanpanel.data

            self.plotascanpanel.clear_axes_create_plot1D()

        # reset ticks and motors positions  ---------------

        self.plotascanpanel.draw_fig()
        return

    def update_fig_2D(self,
                    data_z_values,
                    posmotorsinfo,
                    motor1,
                    motor2,
                    Apptitle,
                    data_img,
                    detectorname,
                    scancommandextremmotorspositions,
                    imshowmode=True,
                    listscanindex=None):
        """update from mesh scan fig and plot
        data_z_values: 2D array of values
        posmotorsinfo: 2D array of motors positions  x, y
        motor1, motor2: str names of motors   (x or fast motor, y or slow motor)

        if imshowmode is False (scattermode): listscanindex must  be provided with list of scan
        index along y axis and also obviously data_z_values = list of z values and   posmotorsinfo list of x values"""
        #         self.plot.fig.clear()

        # y motor (yech, yps, sy) along the vertical axis whatever the fast motor, as in mosaic.py
        # (see generaltools.map_transposed()): arrays (slow, fast) transposed, motors 1 and 2 swapped
        self.plotmeshpanel.fastmotorname = motor1
        if imshowmode and GT.map_transposed(motor1, motor2):
            data_z_values = np.transpose(data_z_values)
            posmotorsinfo = np.transpose(posmotorsinfo, (1, 0, 2))[:, :, ::-1]  # (horizontal, vertical) positions
            data_img = np.transpose(data_img)
            motor1, motor2 = motor2, motor1
            scancommandextremmotorspositions = (tuple(scancommandextremmotorspositions[2:])
                                                + tuple(scancommandextremmotorspositions[:2]))

        self.plotmeshpanel.data = data_z_values
        self.plotmeshpanel.posarray_twomotors = posmotorsinfo
        self.plotmeshpanel.motor1name, self.plotmeshpanel.motor2name = motor1, motor2
        self.plotmeshpanel.absolute_motorposition_unit = "mm"
        self.plotmeshpanel.title = Apptitle
        self.plotmeshpanel.subtitle = getattr(self, "scan_subtitle", "")
        self.plotmeshpanel.Imageindices = data_img
        self.plotmeshpanel.detectorname = detectorname
        self.plotmeshpanel.imshowmode = imshowmode

        if not imshowmode:
            self.plotmeshpanel.listscanindex = listscanindex
        else:
            (self.plotmeshpanel.minmotor1,
                self.plotmeshpanel.maxmotor1,
                self.plotmeshpanel.minmotor2,
                self.plotmeshpanel.maxmotor2,
            ) = scancommandextremmotorspositions

        self.plotmeshpanel.xylabels = ("column index", "row index")
        self.plotmeshpanel.datatype = "scalar"

        if self.plotmeshpanel.colorbar is not None:
            self.plotmeshpanel.colorbar_label = detectorname
            (self.plotmeshpanel.myplot, self.plotmeshpanel.colorbar, self.plotmeshpanel.data) = makefig_update(
                self.plotmeshpanel.fig, self.plotmeshpanel.myplot, self.plotmeshpanel.colorbar, data_z_values)
        else:
            #print("self.plotmeshpanel.colorbar is None")
            self.plotmeshpanel.create_axes()
            self.plotmeshpanel.calc_norm_minmax_values(self.plotmeshpanel.data)
            self.plotmeshpanel.clear_axes_create_plot2D(imshowmode=imshowmode)

        # reset ticks and motors positions  ---------------

        self.plotmeshpanel.draw_fig()
        return

    def onToggle(self, event):
        self.steppresent = 1500
        self.stepmissing = 1000
        if self.timer.IsRunning():
            self.timer.Stop()
            self.toggleBtn.SetLabel("Real Time Plot")
            print("timer stopped!")
        else:
            print("start to on-fly images viewing mode  ----------------")

            self.toggleBtn.SetLabel("Wait!...")
            #             self.stopbtn.Enable()
            self.OnFlyMode = True
            self.scan_in_progress = True
            # loop for already present data
            #             while self.update(event):
            #                 time.sleep(self.steppresent / 1000.)

            self.update(event)

            # timer loop for missing data
            print("*******  WAITING DATA   !!!! *********")
            self.timer.Start(self.stepmissing)
            self.toggleBtn.SetLabel("STOP Real Time")

    def update(self, _, worker=None):
        """
        update at each time step time
        """
        print("\nupdated: ")
        print(time.ctime())
        if self.scantype in ('mesh',):
            scan_index = self.scan_index_mesh
        elif self.scantype in ('ascan',):
            scan_index = self.scan_index_ascan

        if self.scan_in_progress:
            self.scan_in_progress = self.ReadScan_SpecFile(scan_index, resetlistcounters=False)
            return True
        else:
            print("waiting for data  for scan :%d" % scan_index)
            # stop the first timer
            return False

def tick_label(value, step):
    """text of a tick of motor positions, rounded at the precision of the step between map points
    (2 significant digits of the step) without trailing zeros: 4.99995 with step 0.99999 -> '5'"""
    step = abs(float(step))
    decimals = max(0, 1 - int(np.floor(np.log10(step)))) if step > 0 and np.isfinite(step) else 4
    text = "%.*f" % (decimals, value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def normalized_by_monitor(values, monitor, exposure, offset=0.):
    """values / (monitor / exposure - offset), on the points recorded by all the counters (counters of an
    unfinished scan may have one value less)"""
    values = np.asarray(values, dtype=float)
    monitor = np.asarray(monitor, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    nb = min(len(values), len(monitor), len(exposure) if exposure.ndim else len(values))
    if exposure.ndim:
        exposure = exposure[:nb]
    return values[:nb] / (monitor[:nb] / exposure - offset)


def figures_folder(scanfolder):
    """folder where to save figures of a scan: mirror in PROCESSED_DATA of its RAW_DATA folder
    (RAW_DATA/<sample>/<dataset>/scanNNNN -> PROCESSED_DATA/<sample>/<dataset>/scanNNNN, created if needed),
    else the scan folder itself if it can be written, else None.

    :return: folder, list of the folders created (deepest first, see remove_empty_folders())
    """
    if not scanfolder:
        return None, []
    scanfolder = os.path.normpath(str(scanfolder))
    parts = scanfolder.split(os.sep)
    if "RAW_DATA" in parts:
        parts[parts.index("RAW_DATA")] = "PROCESSED_DATA"
        mirror = os.sep.join(parts)
        missing = []
        folder = mirror
        while folder and not os.path.isdir(folder):
            missing.append(folder)
            folder = os.path.dirname(folder)
        try:
            os.makedirs(mirror, exist_ok=True)
            return mirror, missing
        except OSError:  # PROCESSED_DATA not writable
            remove_empty_folders(missing)
    if os.path.isdir(scanfolder) and os.access(scanfolder, os.W_OK):
        return scanfolder, []
    return None, []


def remove_empty_folders(folders):
    """remove the folders (deepest first) that are empty, e.g. created by figures_folder() for nothing"""
    for folder in folders:
        try:
            os.rmdir(folder)
        except OSError:  # not empty, or already removed
            break


def set_counters_menu(combo, counters=None, value=""):
    """counters menu: items (if counters is not None) and displayed value, i.e. the plotted counter
    (Clear() of the menu erases its value: the menu was left blank)"""
    if counters is not None:
        names = [str(name) for name in counters]
        combo.Clear()
        combo.AppendItems(names)
        # width of the longest counter name (+ drop-down button)
        width = max((combo.GetTextExtent(name)[0] for name in names + [str(value)]), default=0) + 50
        combo.SetMinSize((int(min(max(width, 100), 450)), -1))
        combo.GetParent().Layout()
    combo.SetValue(str(value))


def plot_title(title, counter, normalized=False):
    """title of the plots: title lines (scan name, command) and the plotted counter"""
    if not counter:
        return title
    return "%s\n%s%s" % (title, counter, " (normalized by monitor)" if normalized else "")


# colors of the items of the scans lists from the end reason of the scans (BLISS hdf5 logfiles)
ENDREASON_COLOURS = {"SUCCESS": (0, 140, 0), "USER_ABORT": (230, 120, 0)}
ENDREASON_ERROR_COLOUR = (200, 0, 0)  # FAILURE, ...


def endreason_colour(endreason):
    """wx.Colour of a scan item: green SUCCESS, orange USER_ABORT, red other end reason (error),
    None if unknown (scan running, SPEC file)"""
    if endreason is None or endreason == "":
        return None
    if isinstance(endreason, bytes):
        endreason = endreason.decode()
    return wx.Colour(*ENDREASON_COLOURS.get(str(endreason).upper(), ENDREASON_ERROR_COLOUR))


def duration_text(seconds):
    """'25 s', '3 min 25 s', '2 h 05 min'"""
    seconds = int(round(seconds))
    if seconds < 60:
        return "%d s" % seconds
    if seconds < 3600:
        return "%d min %02d s" % divmod(seconds, 60)
    return "%d h %02d min" % (seconds // 3600, (seconds % 3600) // 60)


def scan_dates_text(start, end=None):
    """light text of the dates of a scan: '11 Jul 2026 21:06, 3 min 25 s' (from ISO dates)"""
    try:
        t0 = datetime.datetime.fromisoformat(str(start))
    except ValueError:
        return str(start)
    text = t0.strftime("%d %b %Y %H:%M")
    try:
        text += ", " + duration_text((datetime.datetime.fromisoformat(str(end)) - t0).total_seconds())
    except (ValueError, TypeError):
        pass
    return text


def scan_exposure_time(command, default=1.):
    """exposure time of a BLISS scan command (last argument for ascan, a2scan, ...), default if not found"""
    try:
        return float(logfile_reader.read_fullcommand(command)['expotime'])
    except (ValueError, KeyError, IndexError):
        pass
    try:
        return float(command.split()[-1])
    except (ValueError, IndexError):
        return default


def makefig_update(fig, myplot, cbar, data, datadim=2):
    if myplot:
        print("\n\n\nmyplot exists\n\n\n")
        # data *= 2  # change data, so there is change in output (look at colorbar)
        myplot.set_data(data)  # use this if you use new array
        myplot.autoscale()
        # cbar.update_normal(myplot) #cbar is updated automatically
    else:
        ax = fig.add_subplot(111)
        if datadim == 2:
            myplot = ax.imshow(data)
            cbar = fig.colorbar(myplot)
        else:
            datax, datay = data
            myplot = ax.plot(datax, datay)
    return myplot, cbar, data

def getascan_from_specfile(filename):
    print("getascan_from_specfile")
    f = open(filename, "r")
    listascan = []

    linepos = 0
    while 1:
        line = f.readline()
        if not line:
            break
        if line.startswith("#S"):
            linesplit = line.split()
            if linesplit[2] == "ascan":
                scan_index = int(linesplit[1])
                listascan.append([scan_index, linepos, f.tell(), line])

        linepos += 1
    return listascan

def getmeshscan_from_specfile(filename):
    print("getmeshscan_from_specfile")
    f = open(filename, "r")
    listmesh = []

    linepos = 0
    while 1:
        line = f.readline()
        if not line:
            break
        if line.startswith("#S"):
            linesplit = line.split()
            if linesplit[2] == "mesh":
                scan_index = int(linesplit[1])
                listmesh.append([scan_index, linepos, f.tell(), line])

        linepos += 1

    f.close()
    print("%d lines have been read" % (linepos))
    print("%s contains %d mesh scans" % (filename, len(listmesh)))


    return listmesh

class PlotPanel(wx.Panel):
    """
    Class to show 1D array intensity data
    """

    def __init__(self, parent, _id, title, dataarray, posarray_motors=None,
                                                        posmotorname=None,
                                                        datatype="scalar",
                                                        Imageindices=None,
                                                        absolute_motorposition_unit="micron",
                                                        colorbar_label="Fluo counts",
                                                        stepindex=1,
                                                        xylabels=None,
                                                        listscanindex=None):
        """
        plot 1D plot of dataarray
        """

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.parent = parent
        #print("parent", parent)

        self.frameparent = parent.GetParent()

        self.data = dataarray
        self.data_to_Display = self.data

        self.title = title
        self.datatype = datatype

        self.posarray_motors = posarray_motors
        self.posmotorname = posmotorname
        self.detectorname = 'Monitor'

        self.multipleplots = False
        self.listscanindex = listscanindex

        self.init_figurecanvas()
        self.create_main_panel()

        self.create_axes()
        self.calc_norm_minmax_values(self.data)

        self.clear_axes_create_plot1D()

        self.draw_fig()

    def draw_fig(self):
        # print("in draw_fig()    ascan")
        self.fig.set_canvas(self.canvas)

        # reset ticks and motors positions  ---------------

        self.canvas.draw()

    def init_figurecanvas(self):
        self.dpi = 100
        self.figsize = 4
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi, layout="constrained")
        self.canvas = FigCanvas(self, -1, self.fig)
        self.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.canvas.mpl_connect("button_press_event", self.onClick)

    def create_axes(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

    def create_main_panel(self):
        """
        set main panel of PlotPanel
        """
        self.toolbar = NavigationToolbar(self.canvas)
        self.normalizechckbox = wx.CheckBox(self, -1, "Normalize")
        self.normalizechckbox.SetValue(False)
        self.normalizechckbox.Bind(wx.EVT_CHECKBOX, self.OnNormalizeData)

        self.I0offsettxt = wx.StaticText(self, -1, "Mon. offset (cts/sec) ")
        self.I0offsetctrl = wx.TextCtrl(self, -1, "0.0")

        self.scaletype = "Linear"
        scaletxt = wx.StaticText(self, -1, "Scale")
        self.comboscale = wx.ComboBox(self, -1, self.scaletype, choices=["Linear", "Log10"],
                                                                            size=(-1, 40))

        self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)
        countertxt = wx.StaticText(self, -1, "counter")

        #print("self.frameparent.columns_name", self.frameparent.columns_name)
        sortedcounterslist = sorted(self.frameparent.columns_name)
        self.frameparent.columns_name.sort()
        self.combocounters = wx.ComboBox(self, -1, self.detectorname, #choices=sortedcounterslist,
                                                        choices=self.frameparent.columns_name,
                                                        size=(-1, 40), #style=wx.CB_READONLY)
                                                        style=wx.TE_PROCESS_ENTER)

        self.combocounters.Bind(wx.EVT_COMBOBOX, self.OnChangeCounter)
        self.combocounters.Bind(wx.EVT_TEXT_ENTER, self.OnChangeCounter)

        # motor of the horizontal axis, for scans of several motors (a2scan, ...)
        motortxt = wx.StaticText(self, -1, "motor")
        self.combomotors = wx.ComboBox(self, -1, "", choices=[], size=(-1, 40), style=wx.CB_READONLY)
        self.combomotors.Bind(wx.EVT_COMBOBOX, self.OnChangeMotor)
        self.combomotors.Enable(False)
        tipmotor = "Motor along the horizontal axis (scans of several motors: a2scan, a3scan, ...)"
        motortxt.SetToolTipString(tipmotor)
        self.combomotors.SetToolTipString(tipmotor)

        # --- --------layout
        htoolbar2 = wx.WrapSizer(wx.HORIZONTAL)
        htoolbar2.Add(countertxt, 0)
        htoolbar2.Add(self.combocounters, 0)
        htoolbar2.Add(motortxt, 0)
        htoolbar2.Add(self.combomotors, 0)
        htoolbar2.Add(self.normalizechckbox, 0)
        htoolbar2.Add(self.I0offsettxt, 0)
        htoolbar2.Add(self.I0offsetctrl, 0)

        htoolbar = wx.WrapSizer(wx.HORIZONTAL)
        htoolbar.Add(self.toolbar, 0)
        htoolbar.Add(scaletxt, 0)
        htoolbar.Add(self.comboscale, 0)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(htoolbar2, 0, wx.EXPAND)
        self.vbox.Add(htoolbar, 0, wx.EXPAND)

        self.SetSizer(self.vbox)

    def OnAbout(self, event):
        pass

    def OnNormalizeData(self, _):
        pass

        self.frameparent.normalizeintensity = not self.frameparent.normalizeintensity

        # self.frameparent.MonitorOffset = float(self.I0offsetctrl.GetValue())

        self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index_ascan, resetlistcounters=False)

        # TODO: divide self.data_to_display by monitor data or other counter

    def onClick(self, event):
        """ onclick
        """
        pass

    def onKeyPressed(self, event):
        pass

    def OnChangeScale(self, _):
        self.scaletype = str(self.comboscale.GetValue())
        self.normalizeplot()
        self.canvas.draw()

    def OnChangeCounter(self, _):
        self.detectorname = self.combocounters.GetValue()

        self.frameparent.detectorname_ascan = self.detectorname
        if not self.multipleplots:
            self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index_ascan, resetlistcounters=False)
        else:
            self.frameparent.ReadMultipleScans(self.frameparent.treeascan.set_selected_indices, resetlistcounters=False)

    def set_motors(self, motors, current):
        """motors of the plotted 1D scan in the motor menu (enabled for scans of several motors)"""
        if list(self.combomotors.GetItems()) != list(motors):
            self.combomotors.SetItems(list(motors))
        self.combomotors.SetValue(current or "")
        self.combomotors.Enable(len(motors) > 1)

    def OnChangeMotor(self, _):
        """motor of the horizontal axis chosen in the menu (kept for the next scans that have it)"""
        self.frameparent.motor_ascan = self.combomotors.GetValue()
        self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index_ascan, resetlistcounters=False)

    def normalizeplot(self):
        #TODO:
        pass

    def OnSave(self, _):
        if self.askUserForFilename():
            fig = self.plotmeshpanel.get_figure()
            fig.savefig(os.path.join(str(self.dirname), str(self.filename)))
            print("Image saved in ", os.path.join(self.dirname, self.filename) + ".png")

    def calc_norm_minmax_values(self, data):

        self.data_to_Display = data

    def clear_axes_create_plot1D(self):
        """
        init axes.plot()
        """
        if self.data_to_Display is None:
            return

        # clear the axes and replot everything
        self.axes.cla()
        self.axes.set_title(plot_title(self.title, self.detectorname, self.frameparent.normalizeintensity),
                            fontsize="small")
        # scan folder (from RAW_DATA) in smaller letters above the title
        self.fig.suptitle(getattr(self, "subtitle", ""), fontsize="x-small", color="dimgray")
        #         self.axes.set_autoscale_on(True)
        if self.datatype == "scalar":

            if not self.multipleplots:
                # print("self.data_to_Display.shape", self.data_to_Display.shape)
                self.myplot = self.axes.plot(self.posarray_motors, self.data_to_Display)
            else:
                nbscans = len(self.listscanindex)
                print('nb of scans to plot : ', nbscans)
                for k in range(nbscans):
                    self.axes.plot(self.posarray_motors[k], self.data_to_Display[k])

                self.axes.legend(self.listscanindex)
            self.axes.grid(True)
            self.axes.set_xlabel(self.posmotorname)
            self.axes.set_ylabel(self.detectorname)
            self.axes.format_coord = self.format_coord_single

            if 'NOT AVAILABLE' in self.title:
                misstext = self.title[:self.title.find('AVAILABLE')+9]
                self.axes.text(0.5, 0.5, misstext,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red',
                        transform=self.axes.transAxes)


    def format_coord_single(self, x, y):
        """hover text: nearest point of the plotted scan (index, motor position, counter value, image index)"""
        text = '(x,y) = (%f, %f)' % (x, y)
        xs = None if self.multipleplots else np.asarray(self.posarray_motors, dtype=float)
        if xs is not None and xs.ndim == 1 and np.isfinite(xs).any():
            k = int(np.nanargmin(np.abs(xs - x)))
            text = "point %d: %s=%.6g, %s=%.6g" % (k, self.posmotorname, xs[k], self.detectorname,
                                                  np.asarray(self.data_to_Display, dtype=float)[k])
            imageindices = getattr(self, "Imageindices", None)
            if imageindices is not None and np.ndim(imageindices) == 1 and k < len(imageindices):
                text += ", image %d" % imageindices[k]
        self.frameparent.stbar.SetStatusText(text, 3)
        return text

def natural_key(name):
    """sort key of names with numbers: eiger4m_roi2 before eiger4m_roi10"""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(name))]


class CountersMapsPanel(wx.Panel):
    """maps of all the counters of a 2D scan whose names start with given prefixes (fluo, eiger4m_, ...), by
    pages of 4, 9 or 16 maps, at the place of the 1D plot (toggle button 'Counters maps' of MainFrame).
    Same orientation (y motor vertical), colormap and normalization as the main 2D plot.
    Mouse over a map: point, image index and value in the status bar. Click on a map: its counter is plotted
    in the main 2D plot."""
    DEFAULT_PREFIXES = "fluo eiger4m_ xia_sum_fluo"
    PAGE_SIZES = ("4", "9", "16")

    def __init__(self, parent, frameparent):
        wx.Panel.__init__(self, parent, -1)
        self.frameparent = frameparent
        self.meshscan = None
        self.counters = []
        self.page = 0
        self.axes_counters = {}

        self.fig = Figure((5, 5), dpi=100, layout="constrained")
        self.fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.canvas.SetMinSize((400, 400))
        self.canvas.mpl_connect("button_press_event", self.OnClick)
        self.canvas.mpl_connect("motion_notify_event", self.OnMotion)

        prefixtxt = wx.StaticText(self, -1, "counters starting with")
        self.prefixesctrl = wx.TextCtrl(self, -1, self.DEFAULT_PREFIXES, size=(200, -1), style=wx.TE_PROCESS_ENTER)
        self.prefixesctrl.Bind(wx.EVT_TEXT_ENTER, self.OnChangePrefixes)
        tip = "Beginnings of the names of the counters to map, separated by spaces (press Enter)"
        prefixtxt.SetToolTipString(tip)
        self.prefixesctrl.SetToolTipString(tip)

        perpagetxt = wx.StaticText(self, -1, "maps per page")
        screenheight = wx.Display(0).GetClientArea().height
        self.comboperpage = wx.ComboBox(self, -1, "16" if screenheight >= 1000 else "9",
                                        choices=list(self.PAGE_SIZES), style=wx.CB_READONLY)
        self.comboperpage.Bind(wx.EVT_COMBOBOX, self.OnChangePerPage)

        self.prevbtn = wx.Button(self, -1, "<", size=(40, -1))
        self.nextbtn = wx.Button(self, -1, ">", size=(40, -1))
        self.prevbtn.Bind(wx.EVT_BUTTON, lambda _: self.change_page(-1))
        self.nextbtn.Bind(wx.EVT_BUTTON, lambda _: self.change_page(1))
        self.prevbtn.SetToolTipString("Previous page of maps")
        self.nextbtn.SetToolTipString("Next page of maps")
        self.pagetxt = wx.StaticText(self, -1, "")
        self.savebtn = wx.Button(self, -1, "Save figure")
        self.savebtn.Bind(wx.EVT_BUTTON, self.OnSave)
        self.savebtn.SetToolTipString("Save the maps of this page (png, pdf or svg)")

        controls = wx.WrapSizer(wx.HORIZONTAL)
        for widget in (prefixtxt, self.prefixesctrl, perpagetxt, self.comboperpage):
            controls.Add(widget, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        pages = wx.BoxSizer(wx.HORIZONTAL)
        pages.Add(self.prevbtn, 0)
        pages.Add(self.pagetxt, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 8)
        pages.Add(self.nextbtn, 0)
        pages.Add(self.savebtn, 0, wx.LEFT, 16)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.GROW)
        vbox.Add(pages, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 2)
        vbox.Add(controls, 0, wx.EXPAND | wx.TOP, 2)
        self.SetSizer(vbox)

    def matching_counters(self):
        """counters of the scan (1D) whose names start with one of the prefixes, natural order"""
        if self.meshscan is None:
            return []
        prefixes = tuple(self.prefixesctrl.GetValue().split())
        return sorted((name for name, values in self.meshscan["data"].items()
                       if prefixes and name.startswith(prefixes) and np.ndim(values) == 1 and len(values)),
                      key=natural_key)

    def set_scan(self, meshscan):
        """maps of the counters of a 2D scan (see MainFrame.meshscan); same page if the counters are the same"""
        self.meshscan = meshscan
        counters = self.matching_counters()
        if counters != self.counters:
            self.page = 0
        self.counters = counters
        self.draw()

    def OnChangePrefixes(self, _):
        self.counters = self.matching_counters()
        self.page = 0
        self.draw()

    def OnChangePerPage(self, _):
        self.page = 0
        self.draw()

    def change_page(self, step):
        self.page += step
        self.draw()

    def map_of_counter(self, name):
        """2D array (slow, fast) of the values of a counter (NaN for points not recorded), normalized by monitor
        as the main 2D plot"""
        ms = self.meshscan
        nb1, nb2 = ms["nb1"], ms["nb2"]
        values = np.asarray(ms["data"][name], dtype=float)[:nb1 * nb2]
        divisor = ms.get("divisor")
        if divisor is not None and len(divisor) >= len(values):
            values = values / np.asarray(divisor, dtype=float)[:len(values)]
        full = np.full(nb1 * nb2, np.nan)
        full[:len(values)] = values
        return full.reshape((nb2, nb1))

    def draw(self):
        """draw the maps of the current page"""
        self.fig.clear()
        self.axes_counters = {}
        perpage = int(self.comboperpage.GetValue())
        nbpages = max(1, math.ceil(len(self.counters) / perpage))
        self.page = min(max(self.page, 0), nbpages - 1)
        first = self.page * perpage
        pagecounters = self.counters[first:first + perpage]
        self.pagetxt.SetLabel("page %d/%d, counters %d-%d of %d" % (self.page + 1, nbpages, first + 1 if pagecounters else 0,
                                                                     first + len(pagecounters), len(self.counters)))
        self.prevbtn.Enable(self.page > 0)
        self.nextbtn.Enable(self.page < nbpages - 1)
        self.Layout()
        if self.meshscan is None or not pagecounters:
            message = ("Select a scan of the 2D scans list" if self.meshscan is None else
                       "No counter starting with: %s" % self.prefixesctrl.GetValue())
            self.fig.text(0.5, 0.5, message, ha="center", va="center")
            self.canvas.draw()
            return

        ms = self.meshscan
        self.transposed = GT.map_transposed(ms["fastmotor"], ms["slowmotor"])
        mainplot = self.frameparent.plotmeshpanel
        ncols = math.ceil(math.sqrt(perpage))
        nrows = math.ceil(perpage / ncols)
        # axes: positions (micron) from the corner of the map; tick labels on the left column and the bottom row
        steps = self.map_steps()
        hmotor, vmotor = ((ms["slowmotor"], ms["fastmotor"]) if self.transposed
                          else (ms["fastmotor"], ms["slowmotor"]))
        for k, name in enumerate(pagecounters):
            map2D = self.map_of_counter(name)
            display = map2D.T if self.transposed else map2D  # y motor along the vertical axis
            ax = self.fig.add_subplot(nrows, ncols, k + 1)
            ax.imshow(display, cmap=mainplot.cmap, interpolation="nearest", origin=mainplot.origin, aspect="equal")
            ax.set_title(name, fontsize=7)
            self.axes_counters[ax] = (name, display)
            if steps is None:
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            leftcolumn = k % ncols == 0
            bottomrow = k + ncols >= len(pagecounters)  # no map below
            for axis, step in ((ax.xaxis, steps[0]), (ax.yaxis, steps[1])):
                axis.set_major_locator(MaxNLocator(3, integer=True))
                axis.set_major_formatter(FuncFormatter(lambda index, _, step=step: tick_label(index * step, step)))
            ax.tick_params(labelsize=6, length=2, pad=1, labelbottom=bottomrow, labelleft=leftcolumn)
            if leftcolumn and bottomrow:
                fast = ms["fastmotor"]
                ax.set_xlabel("%s (micron, %s)" % (hmotor, "fast" if hmotor == fast else "slow"), fontsize=6, labelpad=1)
                ax.set_ylabel("%s (micron, %s)" % (vmotor, "fast" if vmotor == fast else "slow"), fontsize=6, labelpad=1)
        command = ms["command"]
        title = "%s\n%s" % (ms["scan_index"], command.split(None, 2)[2] if command.startswith("#S") else command)
        if ms["subtitle"]:
            title += "\n" + ms["subtitle"]
        self.fig.suptitle(title, fontsize=7)
        self.canvas.draw()

    def map_steps(self):
        """(horizontal, vertical) steps between the map points (micron) from the scan command, None if unknown"""
        ms = self.meshscan
        try:
            dc = logfile_reader.read_fullcommand(ms["command"])
            fast = abs(dc["fmotmax"] - dc["fmotmin"]) / max(dc["fmotnbsteps"], 1) * 1000.
            slow = abs(dc["smotmax"] - dc["smotmin"]) / max(dc["smotnbsteps"], 1) * 1000.
        except (ValueError, KeyError, IndexError):  # e.g. SPEC command
            return None
        return (slow, fast) if self.transposed else (fast, slow)

    def OnSave(self, _):
        """save the maps of the current page (default: <scan name>_counters_maps_page<n>.png in the mirror in
        PROCESSED_DATA of the RAW_DATA folder of the scan, created if needed, see figures_folder())"""
        if self.meshscan is None or not self.axes_counters:
            wx.MessageBox("No maps to save: select a scan of the 2D scans list", "Save figure")
            return
        folder, created = figures_folder(self.meshscan.get("scanfolder"))
        if folder is None:
            folder = os.getcwd()
        name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(self.meshscan["scan_index"]))
        dlg = wx.FileDialog(self, "Save the counters maps", defaultDir=str(folder),
                            defaultFile="%s_counters_maps_page%d.png" % (name, self.page + 1),
                            wildcard="PNG image (*.png)|*.png|PDF (*.pdf)|*.pdf|SVG (*.svg)|*.svg",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if os.path.splitext(path)[1].lower() not in (".png", ".pdf", ".svg"):
                path += (".png", ".pdf", ".svg")[dlg.GetFilterIndex()]
            try:
                self.fig.savefig(path, dpi=150)
                self.frameparent.stbar0.SetStatusText("counters maps saved in %s" % path)
            except OSError as exc:  # e.g. folder without write permission
                wx.MessageBox("Cannot save %s:\n%s" % (path, exc), "Save figure")
        remove_empty_folders(created)  # mirror folder created for nothing (cancelled, saved elsewhere)
        dlg.Destroy()

    def point_of(self, ax, x, y):
        """(counter name, value, fast index, slow index) of the map point under (x, y) of ax, None outside"""
        if ax not in self.axes_counters or x is None or y is None:
            return None
        name, display = self.axes_counters[ax]
        col, row = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        if not (0 <= row < display.shape[0] and 0 <= col < display.shape[1]):
            return None
        fast, slow = (row, col) if self.transposed else (col, row)
        return name, display[row, col], fast, slow

    def OnMotion(self, event):
        """status bar: counter, motors indices, image index and value of the point under the mouse"""
        point = self.point_of(event.inaxes, event.xdata, event.ydata)
        if point is None:
            return
        name, value, fast, slow = point
        ms = self.meshscan
        image = np.asarray(ms["imageindices"])[slow, fast]
        self.frameparent.stbar0.SetStatusText("%s: %s #%d, %s #%d, image %d, value %.6g"
                                              % (name, ms["fastmotor"], fast, ms["slowmotor"], slow, image, value))

    def OnClick(self, event):
        """click on a map: its counter is plotted in the main 2D plot"""
        if event.inaxes not in self.axes_counters:
            return
        name = self.axes_counters[event.inaxes][0]
        mainplot = self.frameparent.plotmeshpanel
        mainplot.combocounters.SetValue(name)
        mainplot.OnChangeCounter(None)


class ImshowPanel(wx.Panel):
    """
    Class to show 2D array intensity data
    """

    def __init__(self, parent, _id, title, dataarray, posarray_twomotors=None,
                                                        posmotorname=(None, None),
                                                        datatype="scalar",
                                                        absolutecornerindices=None,
                                                        Imageindices=None,
                                                        absolute_motorposition_unit="micron",
                                                        colorbar_label="Fluo counts",
                                                        stepindex=1,
                                                        xylabels=None,
                                                        imshowmode=True,
                                                        filetype='spec'):
        """
        plot 2D plot of dataarray
        """
        USE_COLOR_BAR = False

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.parent = parent
        #print("parent", parent)

        self.frameparent = parent.GetParent()

        self.data = dataarray
        self.data_to_Display = self.data

        self.posarray_twomotors = posarray_twomotors

        self.imshowmode = imshowmode
        if self.imshowmode:
            # print(self.posarray_twomotors[0, 0], self.posarray_twomotors[0, -1])
            # print(self.posarray_twomotors[-1, 0], self.posarray_twomotors[-1, -1])
            self.minmotor1, self.minmotor2 = posarray_twomotors[0, 0]
            self.maxmotor1, self.maxmotor2 = posarray_twomotors[-1, -1]
            self.absolute_motorposition_unit = absolute_motorposition_unit
            self.absolutecornerindices = absolutecornerindices

        if posmotorname is not None:
            self.motor1name, self.motor2name = posmotorname

        self.datatype = datatype

        self.title = title
        self.Imageindices = Imageindices

        self.filetype = filetype
        if self.filetype == 'hdf5':
            self.detectorname = 'mon'
        if self.filetype == 'spec':
            self.detectorname = 'Monitor'

        self.cNorm = None
        self.myplot = None
        self.colorbar = None

        self.stepindex = stepindex

        self.xylabels = xylabels
        self.dirname = None
        self.filename = None

        self.LastLUT = "gist_earth_r"
        self.YORIGINLIST = ["lower", "upper"]
        self.XORIGINLIST = ["left", "right"]
        self.origin = self.YORIGINLIST[0]
        self.flagyorigin = 0
        self.flagxorigin = 0

        self.init_figurecanvas()
        self.create_main_panel()

        self.create_axes()

        self.cmap = GT.GIST_EARTH_R

        self.calc_norm_minmax_values(self.data)
        self.clear_axes_create_plot2D()

        if USE_COLOR_BAR:
            self.colorbar_label = colorbar_label
            self.colorbar = self.fig.colorbar(self.myplot)

        self.draw_fig()

    def draw_fig(self):

        if not self.imshowmode:
            print("in draw_fig()   scatter mode")
            self.fig.set_canvas(self.canvas)
            self.canvas.draw()
            return

        #print("in draw_fig()   mesh scan")
        self.set_motorspositions_parameters()
        #         print "self.fromindex_to_pixelpos_x", self.fromindex_to_pixelpos_x
        #
        #         for k in range(10):
        #             print self.fromindex_to_pixelpos_x(k, 0)

        #         TICKS_FORMATTER_TYPE = 'ABSOLUTE'
        TICKS_FORMATTER_TYPE = "RELATIVE_CORNER"

        if TICKS_FORMATTER_TYPE == "ABSOLUTE":
            formatterfunc_x = self.fromindex_to_pixelpos_x_absolute
            formatterfunc_y = self.fromindex_to_pixelpos_y_absolute
            self.axes.set_xlabel("mm")
            self.axes.set_ylabel("mm")
        elif TICKS_FORMATTER_TYPE == "RELATIVE_CORNER":
            formatterfunc_x = self.fromindex_to_pixelpos_x_relative_corner
            formatterfunc_y = self.fromindex_to_pixelpos_y_relative_corner
            fast = getattr(self, "fastmotorname", None)
            self.axes.set_xlabel("%s (micron, %s motor)" % (self.motor1name, "fast" if self.motor1name == fast else "slow"))
            self.axes.set_ylabel("%s (micron, %s motor)" % (self.motor2name, "fast" if self.motor2name == fast else "slow"))

        self.axes.get_xaxis().set_major_formatter(FuncFormatter(formatterfunc_x))
        self.axes.get_yaxis().set_major_formatter(FuncFormatter(formatterfunc_y))
        self.axes.format_coord = self.format_coord
        self.fig.set_canvas(self.canvas)

        # reset ticks and motors positions  ---------------

        self.canvas.draw()

    def init_figurecanvas(self):
        self.dpi = 100
        self.figsize = 4
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi, layout="constrained")
        self.canvas = FigCanvas(self, -1, self.fig)
        self.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.canvas.mpl_connect("button_press_event", self.onClick)

    def create_axes(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

    def create_main_panel(self):
        """
        set main panel of ImshowPanel
        """
        self.toolbar = NavigationToolbar(self.canvas)

        # color scale limits: one slider with two thumbs (min, max), as in mosaic.py. Limits are kept as
        # fractions of the data range from one plot to another
        self.colorscalefractions = (0., 1.)
        self.rangetxt = wx.StaticText(self, -1, "Color scale")
        self.rangeslider = RangeSlider(self, 0., 1.)
        self.rangeslider.Bind(EVT_RANGE_SLIDER, self.OnColorRange)

        # loading LUTS
        self.mapsLUT = [m for m in pcm.datad if not m.endswith("_r")]
        self.mapsLUT.sort()

        luttxt = wx.StaticText(self, -1, "LUT")
        self.comboLUT = wx.ComboBox(self, -1, self.LastLUT, size=(-1, 40),
                                                    choices=self.mapsLUT,
                                                    style=wx.TE_PROCESS_ENTER)  # ,
                                                # style=wx.CB_READONLY)

        self.comboLUT.Bind(wx.EVT_COMBOBOX, self.OnChangeLUT)
        self.comboLUT.Bind(wx.EVT_TEXT_ENTER, self.OnChangeLUT)

        self.normalizechckbox = wx.CheckBox(self, -1, "Normalize")
        self.normalizechckbox.SetValue(False)
        self.normalizechckbox.Bind(wx.EVT_CHECKBOX, self.OnNormalizeData)

        self.I0offsettxt = wx.StaticText(self, -1, "Mon. offset (cts/sec) ")
        self.I0offsetctrl = wx.TextCtrl(self, -1, "0.0")

        self.scaletype = "Linear"
        scaletxt = wx.StaticText(self, -1, "Scale")
        self.comboscale = wx.ComboBox(self, -1, self.scaletype, choices=["Linear", "Log10"],
                                                                            size=(-1, 40))

        self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)

        btnflipud = wx.Button(self, -1, "Flip Vert.")
        btnflipud.Bind(wx.EVT_BUTTON, self.OnChangeYorigin)

        btnfliplr = wx.Button(self, -1, "Flip Hori.")
        btnfliplr.Bind(wx.EVT_BUTTON, self.OnChangeXorigin)

        countertxt = wx.StaticText(self, -1, "counter")

        #print("self.frameparent.columns_name", self.frameparent.columns_name)
        sortedcounterslist = sorted(self.frameparent.columns_name)
        self.combocounters = wx.ComboBox(self, -1, self.frameparent.detectorname_mesh,
                                                        choices=sortedcounterslist,
                                                        size=(-1, 40),
                                                        style=wx.TE_PROCESS_ENTER)

        self.combocounters.Bind(wx.EVT_COMBOBOX, self.OnChangeCounter)
        self.combocounters.Bind(wx.EVT_TEXT_ENTER, self.OnChangeCounter)

        # --- --------tooltip ---------------------

        btnflipud.SetToolTipString("Flip Plot Up/Down")
        btnfliplr.SetToolTipString("Flip Plot Left/Right")

        tipcnt = "Counters to be plot (from spec file list)"
        countertxt.SetToolTipString(tipcnt)
        self.combocounters.SetToolTipString(tipcnt)
        tiplut = "Look-Up-Table for intensity mapping"
        luttxt.SetToolTipString(tiplut)
        self.comboLUT.SetToolTipString(tiplut)

        self.rangetxt.SetToolTipString("Minimum and maximum of intensity mapping")
        # --- --------layout
        # rows of controls wrap when the window is narrow (small screens): the scans lists keep their width
        self.slidersbox = wx.WrapSizer(wx.HORIZONTAL)
        self.slidersbox.Add(self.rangetxt, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.slidersbox.Add(self.rangeslider, 0, wx.ALIGN_CENTER_VERTICAL)
        self.slidersbox.AddSpacer(5)
        self.slidersbox.Add(luttxt, 0)
        self.slidersbox.Add(self.comboLUT, 0)

        htoolbar2 = wx.WrapSizer(wx.HORIZONTAL)
        htoolbar2.Add(countertxt, 0)
        htoolbar2.Add(self.combocounters, 0)
        htoolbar2.Add(self.normalizechckbox, 0)
        htoolbar2.Add(self.I0offsettxt, 0)
        htoolbar2.Add(self.I0offsetctrl, 0)

        htoolbar = wx.WrapSizer(wx.HORIZONTAL)
        htoolbar.Add(self.toolbar, 0)
        htoolbar.Add(scaletxt, 0)
        htoolbar.Add(self.comboscale, 0)
        htoolbar.Add(btnflipud, 0)
        htoolbar.Add(btnfliplr, 0)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.slidersbox, 0, wx.EXPAND)
        self.vbox.Add(htoolbar2, 0, wx.EXPAND)
        self.vbox.Add(htoolbar, 0, wx.EXPAND)

        self.SetSizer(self.vbox)

    def OnAbout(self, event):
        pass

    def OnNormalizeData(self, _):
        self.frameparent.normalizeintensity = not self.frameparent.normalizeintensity

        self.frameparent.MonitorOffset = float(self.I0offsetctrl.GetValue())

        self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index_mesh, resetlistcounters=False)

    def onClick(self, event):
        """ onclick
        """
        #print(event.button)
        if event.inaxes:
            self.centerx, self.centery = event.xdata, event.ydata
            #print("current clicked positions", self.centerx, self.centery)
        # if event.button == 3:
        #     self.movingxy(False)

    def movingxy(self, msgbox):
        x, y = self.centerx, self.centery
        col = int(x + 0.5)
        row = int(y + 0.5)

        numrows, numcols = self.data.shape[:2]
        posmotors = self.posarray_twomotors

        if posmotors is not None:

            posmotor1 = posmotors[0, :, 0]
            posmotor2 = posmotors[:, 0, 1]

        if col >= 0 and col < numcols and row >= 0 and row < numrows:

            if posmotors is not None:
                current_posmotor1 = posmotor1[col]
                current_posmotor2 = posmotor2[row]

                # print("SPEC COMMAND:\nmv %s %.5f %s %.5f"
                #     % (self.motor1name, current_posmotor1,
                #         self.motor2name, current_posmotor2))

                sentence = (
                    "%s=%.6f\n%s=%.6f\n\nSPEC COMMAND to move to this point:\n\nmv %s %.5f %s %.5f"
                    % (self.motor1name, current_posmotor1, self.motor2name,
                        current_posmotor2, self.motor1name, current_posmotor1,
                        self.motor2name, current_posmotor2))

                command = "mv %s %.5f %s %.5f" % (self.motor1name, current_posmotor1,
                                                self.motor2name, current_posmotor2)
                commandBliss = "mv(%s,%.5f,%s,%.5f)" % (self.motor1name, current_posmotor1,
                                                self.motor2name, current_posmotor2)

                if msgbox:
                    wx.MessageBox(sentence + "\nBLISS command\n" + commandBliss, "INFO")
                    print('command to BLISS',command)

                # WARNING could do some instabilities to station ??
                # msgdialog = MC.MessageCommand(self, -1, "motors command",
                #     sentence=sentence, speccommand=command, specconnection=None)
                # msgdialog.ShowModal()

    def onKeyPressed(self, event):

        key = event.key
        #print("key ==> ", key)

        if key == "escape":

            ret = wx.MessageBox("Are you sure to quit?", "Question",
                                    wx.YES_NO | wx.NO_DEFAULT, self)

            if ret == wx.YES:
                self.Close()

        elif key == "p":  # 'p'
            print("in branch key ==> ", key)
            self.movingxy(True)

            return

    def OnChangeScale(self, _):
        self.scaletype = str(self.comboscale.GetValue())
        self.normalizeplot()
        self.canvas.draw()

    def OnChangeLUT(self, _):
        #         print "OnChangeLUT"
        self.cmap = self.comboLUT.GetValue()
        self.myplot.set_cmap(self.cmap)
        self.canvas.draw()

    def OnChangeYorigin(self, _):
        """
        reverse y origin
        """
        self.axes.set_ylim(self.axes.get_ylim()[::-1])
        self.flagyorigin += 1
        self.origin = self.YORIGINLIST[self.flagyorigin % 2]
        self.canvas.draw()

    def OnChangeXorigin(self, _):
        """
        reverse  x limits of plot, and update self.flagxorigin
        """
        self.axes.set_xlim(self.axes.get_xlim()[::-1])
        self.flagxorigin += 1
        self.canvas.draw()

    def OnChangeCounter(self, _):
        """ read selected counter column in currentr scan data and spec file"""

        self.detectorname = self.combocounters.GetValue()

        self.frameparent.detectorname_mesh = self.detectorname

        self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index_mesh, resetlistcounters=False)

    def OnColorRange(self, _):
        """color scale limits changed with the range slider (kept as fractions of the data range)"""
        low, high = self.rangeslider.GetValues()
        span = (self.maxvals - self.minvals) or 1.
        self.colorscalefractions = ((low - self.minvals) / span, (high - self.minvals) / span)
        self.normalizeplot()
        self.canvas.draw_idle()

    def colorscale_norm(self, vmin, vmax):
        """matplotlib normalization of the color scale between vmin and vmax for the scale type"""
        if self.scaletype == "Log10":
            values = np.asarray(self.data_to_Display, dtype=float)
            positive = values[np.isfinite(values) & (values > 0)]
            lowest = float(positive.min()) if positive.size else 1.
            vmin = max(vmin, lowest)
            return colors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
        return colors.Normalize(vmin=vmin, vmax=vmax)

    def normalizeplot(self):
        """normalize intensity scale according to GUI widgets parameters"""

        # print('self.minvals', self.minvals)
        # print('self.maxvals', self.maxvals)

        self.cNorm = self.colorscale_norm(*self.rangeslider.GetValues())
        self.myplot.set_norm(self.cNorm)

    def OnSave(self, _):
        """save image """
        if self.askUserForFilename():
            fig = self.plotmeshpanel.get_figure()
            fig.savefig(os.path.join(str(self.dirname), str(self.filename)))
            print("Image saved in ", os.path.join(self.dirname, self.filename) + ".png")

    def calc_norm_minmax_values(self, data):
        """ set self.self.cNorm  and self.maxvals,  self.minvals from data"""

        self.data_to_Display = data
        self.cNorm = None

        if data is None:
            return

        #print("plot of datatype = %s" % self.datatype)

        if not self.imshowmode:

            list_z_values = self.data
            list_x_values = self.posarray_twomotors # in reality  list of x's of different lengths
            l_idx = self.listscanindex
            list_y_values = []

            minx = list_x_values[0][0]
            maxx = list_x_values[0][-1]
            miny = 0
            maxy = 1
            maxnbelems = 0
            x = []
            y = []
            z = []
            for k, li in enumerate(list_z_values):

                list_y_values.append(k * np.ones(len(li)))
                # list_y_values.append(l_idx[k] * np.ones(len(li)))

                llx = list_x_values[k]
                lly = list_y_values[k]
                llz = list_z_values[k]
                minx = min(minx, min(llx))
                miny = min(miny, min(lly))
                maxx = max(maxx, max(llx))
                maxy = max(maxy, max(lly))
                maxnbelems = max(maxnbelems, len(llx))

                x = x + llx.tolist()
                y = y + lly.tolist()
                z = z + llz.tolist()

                self.x, self.y = x, z
                #self.axes.scatter(list_x_values[k], list_y_values[k], c=list_z_values[k], marker='h', s=50)

            # print('minx, maxx', minx, maxx)
            # print('miny, maxy', miny, maxy)
            from scipy.interpolate import griddata

            # define grid.
            nx = 60 # int(maxnbelems * 10)
            ny = 30   # len(l_idx) * 20
            xi = np.linspace(minx, maxx, nx)
            yi = np.linspace(0, len(l_idx) - 1, ny)

            # print(('xi', xi))
            # print(('yi', yi))
            # grid the data.
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

            # print('zi[0]', zi[0])
            # print('zi[:,0]', zi[:, 0])

            self.data_to_Display = zi

        finite = np.asarray(self.data_to_Display, dtype=float)
        finite = finite[np.isfinite(finite)]
        self.minvals, self.maxvals = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        # color scale limits: same fractions of the data range as for the previous plot
        f0, f1 = self.colorscalefractions
        span = self.maxvals - self.minvals
        low, high = self.minvals + f0 * span, self.minvals + f1 * span
        self.rangeslider.SetValues(low, high)  # (SetValues() extends the range to include the values)
        self.rangeslider.SetRange(self.minvals, self.maxvals)
        self.cNorm = self.colorscale_norm(low, high)


    # def forceAspect(self, aspect=1.0):
    #     """ force image plot aspect ratio"""
    #     im = self.axes.get_images()
    #     extent = im[0].get_extent()
    #     self.axes.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    def re_init_colorbar(self):
        self.colorbar.set_label(self.colorbar_label)
        self.colorbar.set_clim(vmin=self.minvals, vmax=self.maxvals)
        self.colorbar.draw_all()

    def clear_axes_create_plot2D(self, imshowmode=True):
        """
        init axes
        """
        if self.data_to_Display is None:
            return

        # clear the axes and replot everything
        self.axes.cla()
        self.axes.set_title(plot_title(self.title, self.detectorname, self.frameparent.normalizeintensity),
                            fontsize="small")
        # scan folder (from RAW_DATA) in smaller letters above the title
        self.fig.suptitle(getattr(self, "subtitle", ""), fontsize="x-small", color="dimgray")
        #         self.axes.set_autoscale_on(True)
        if self.datatype == "scalar":
            if imshowmode:
                #print("self.data_to_Display.shape", self.data_to_Display.shape)
                self.myplot = self.axes.imshow(self.data_to_Display, cmap=self.cmap,
                                                interpolation="nearest",
                                                norm=self.cNorm,
                                                aspect="equal",
                                                #                              extent=self.extent,
                                                origin=self.origin)

                if self.XORIGINLIST[self.flagxorigin % 2] == "right":
                    self.axes.set_xlim(self.axes.get_xlim()[::-1])
            else:  # scattermode
                self.cmap.set_bad(color='red')

                self.myplot = self.axes.imshow(self.data_to_Display, cmap=self.cmap,
                                                interpolation="nearest",
                                                #norm=self.cNorm,
                                                aspect="equal",
                                                #                              extent=self.extent,
                                                #origin=self.origin)
                                                )

            if 'NOT AVAILABLE' in self.title:
                misstext = self.title[:self.title.find('AVAILABLE')+9]
                self.axes.text(0.5, 0.5, misstext,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red',
                        transform=self.axes.transAxes)

    def fromindex_to_pixelpos_x_absolute(self, index, _):
        # absolute positions ticks (mm)
        step_factor = self.step_factor
        return tick_label(index * self.step_x / step_factor + self.posmotor1[0], self.step_x / step_factor)

    def fromindex_to_pixelpos_x_relative_corner(self, index, _):

        step_factor = self.step_factor
        # relative positions from bottom left corner ticks

        # print("self.step_x", self.step_x)

        return tick_label(index * self.step_x, self.step_x)

    def fromindex_to_pixelpos_y_absolute(self, index, _):
        # absolute positions ticks
        step_factor = self.step_factor
        # print("step_factor", step_factor)
        # print("self.step_x", self.step_x)
        return tick_label(index * self.step_y / step_factor + self.posmotor2[0], self.step_y / step_factor)

    def fromindex_to_pixelpos_y_relative_corner(self, index, _):

        step_factor = self.step_factor
        # relative positions from bottom left corner ticks

        # print("self.step_y", self.step_y)

        return tick_label(index * self.step_y, self.step_y)

    def set_motorspositions_parameters(self, verbose=0):
        self.posmotors = self.posarray_twomotors

        if self.posmotors is None:
            return "posmotors is None"
        if verbose:
            print("in set_motorspositions_parameters")

            print("absolute_motorposition_unit", self.absolute_motorposition_unit)

            print("pos extremes")
            print("first motor", self.posarray_twomotors[0, 0], self.posarray_twomotors[0, -1])
            print("second motor",
                self.posarray_twomotors[-1, 0],
                self.posarray_twomotors[-1, -1])

        rangeX = (np.fix((self.posarray_twomotors[0, -1] - self.posarray_twomotors[0, 0])[0]
                * 100000) / 100000)
        rangeY = (np.fix((self.posarray_twomotors[-1, -1] - self.posarray_twomotors[0, -1])[1]
                * 100000) / 100000)
        if verbose:
            print("first motor total range", rangeX)
            print("second motor total range", rangeY)

        #print('self.data.shape ...>', self.data.shape)

        self.numrows, self.numcols = self.data.shape[:2]

        self.tabindices = self.Imageindices

        # initmotor1 = self.posmotors[0, 0, 0]
        # initmotor2 = self.posmotors[0, 0, 1]

        self.posmotor1 = self.posmotors[0, :, 0]
        self.posmotor2 = self.posmotors[:, 0, 1]


        nby, nbx = self.posmotors.shape[:2]

        self.poscenter_motor1 = self.posmotor1[nbx // 2]
        self.poscenter_motor2 = self.posmotor2[nby // 2]

        # x= fast motor  (first in spec scan)
        # y slow motor (second in spec scan)
        #         self.step_x = (self.posmotor1[-1] - self.posmotor1[0]) / (nbx - 1)
        #         self.step_y = (self.posmotor2[-1] - self.posmotor2[0]) / (nby - 1)

        self.step_x = (self.maxmotor1 - self.minmotor1) / (nbx - 1)
        self.step_y = (self.maxmotor2 - self.minmotor2) / (nby - 1)

        if verbose:
            print("set self.step_x to mm", self.step_x)
            print("set self.step_y to mm", self.step_y)
        #         print "step_x %f %s " % (self.step_x, self.absolute_motorposition_unit)
        #         print "step_y %f %s " % (self.step_y, self.absolute_motorposition_unit)

        self.step_factor = 1.0
        if self.absolute_motorposition_unit == "mm":
            self.step_factor = 1000.0
            self.step_x = self.step_x * self.step_factor
            self.step_y = self.step_y * self.step_factor

    #         nb_of_microns_x = round((self.posmotor1[-1] - self.posmotor1[0]) * self.step_factor)
    #         nb_of_microns_y = round((self.posmotor2[-1] - self.posmotor2[0]) * self.step_factor

    def format_coord(self, x, y):

        col = int(x + 0.5)
        row = int(y + 0.5)

        numcols, numrows = self.numcols, self.numrows
        posmotors = self.posmotors
        posmotor1, posmotor2 = self.posmotor1, self.posmotor2
        tabindices = self.tabindices
        step_factor = self.step_factor
        poscenter_motor1, poscenter_motor2 = (self.poscenter_motor1, self.poscenter_motor2)

        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = self.data[row, col]

            Imageindex = tabindices[row, col]
            if posmotors is None:
                sentence0 = "x=%1.4f, y=%1.4f, val=%s, ImageIndex: %d" % (x, y, str(z), Imageindex)
                sentence_corner = ""
                sentence_center = ""
                sentence = "No motors positions"
            else:
                sentence0 = ("j=%d, i=%d, ABSOLUTE=[%s=%.5f,%s=%.5f], %s = %.6g, ImageIndex: %d"
                                % (col, row, self.motor1name, posmotor1[col],
                                    self.motor2name, posmotor2[row], self.detectorname, z, Imageindex))

                sentence = "POSITION (micron) from: "
                sentence_corner = "CORNER =[[%s=%.2f,%s=%.2f]]" % (self.motor1name,
                                                    (posmotor1[col] - posmotor1[0]) * step_factor,
                                                    self.motor2name,
                                                    (posmotor2[row] - posmotor2[0]) * step_factor)
                sentence_center = "CENTER =[[%s=%.2f,%s=%.2f]]" % (self.motor1name,
                                                (posmotor1[col] - poscenter_motor1) * step_factor,
                                                self.motor2name,
                                                (posmotor2[row] - poscenter_motor2) * step_factor)

            self.frameparent.stbar0.SetStatusText(sentence0)
            self.frameparent.stbar.SetStatusText(sentence)
            self.frameparent.stbar.SetStatusText(sentence_corner, 1)
            self.frameparent.stbar.SetStatusText(sentence_center, 2)

            return sentence0
        else:
            print("out of plot")
            return "out of plot"


class MyRectilinearAxes(Axes):
    name = "MyRectilinearAxes"

    def format_coord(self, x, y):
        # Massage your data here -- good place for scalar multiplication
        if x is None:
            xs = "???"
        else:
            xs = self.format_xdata(x * 0.5)
        if y is None:
            ys = "???"
        else:
            ys = self.format_ydata(y * 0.5)
        # Format your label here -- I transposed x and y labels
        return "x=%s y=%s" % (ys, xs)


class MyApp(wx.App):
    def OnInit(self):
        frame = MainFrame(None, -1, "plotmeshGUI.py")
        frame.Show(True)
        self.SetTopWindow(frame)
        return True


def start():
    """ launcher of plotmeshspecGUI as module"""
    GUIApp = wx.App()
    frame = MainFrame(None, -1, "plotmeshGUI.py")
    frame.Show()
    GUIApp.MainLoop()


if __name__ == "__main__":
    start()