r"""
plotmeshspecGUI.py is a module for the visualisation of 2D Laue microdiffraction map recorded on spec file at ESRF

This module belongs to the open source LaueTools project
with a free code repository at at gitlab.esrf.fr

https://gitlab.esrf.fr/micha/lauetools

J. S. Micha Jan 2022
mailto: micha --+at-+- esrf --+dot-+- fr
"""



import os
import sys
import time

import matplotlib as mpl

from matplotlib import __version__ as matplotlibversion
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                    NavigationToolbar2WxAgg as NavigationToolbar)

import matplotlib.colors as colors
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

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

if sys.version_info.major == 3:
    import LaueTools.generaltools as GT
    # from LaueTools.Daxm.contrib.spec_reader import ReadSpec, ReadHdf5, readdata_from_hdf5key
    from LaueTools.logfile_reader import ReadSpec, ReadHdf5, readdata_from_hdf5key
    import LaueTools.MessageCommand as MC
    import LaueTools.IOLaueTools as IOLT
    # import LaueTools.Daxm.contrib.spec_reader as spec_reader
    import LaueTools.logfile_reader as logfile_reader

else:
    import generaltools as GT
    # from LaueTools.Daxm.contrib.spec_reader import ReadSpec,ReadHdf5, readdata_from_hdf5key
    from LaueTools.logfile_reader import ReadSpec, ReadHdf5, readdata_from_hdf5key
    import MessageCommand as MC

import wx.lib.agw.customtreectrl as CT

class TreePanel(wx.Panel):
    """ class of tree organisation of scans

    granparent class must provide  ReadScan_SpecFile()

    sets granparent scan_index_mesh  or scan_index_ascan to selected item index
    """
    def __init__(self, parent, scantype=None, _id=wx.ID_ANY, **kwd):
        wx.Panel.__init__(self, parent=parent, id=_id, **kwd)

        self.parent = parent
        self.scantype = scantype
        self.frameparent = self.parent.GetParent()
        # self.tree = wx.TreeCtrl(self, -1, wx.DefaultPosition, (-1, -1),
        #                                                     wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS)
        # agwStyle=wx.TR_DEFAULT_STYLE
        self.tree = CT.CustomTreeCtrl(self, -1, agwStyle=wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS | wx.TR_MULTIPLE, size=(200,-1))
        self.tree.DoGetBestSize()

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

        # wx.EVT_TREE_ITEM_RIGHT_CLICK
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.tree, 1, wx.EXPAND)
        vbox.AddSpacer(1)

        self.SetSizer(vbox)

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
            if self.scantype == 'ASCAN':
                nbascans = len(self.frameparent.list_ascanscan_indices)
                #print('self.frameparent.list_ascanscan_indices', self.frameparent.list_ascanscan_indices)
                poslastscan = self.frameparent.list_ascanscan_indices.index(self.last_sel_scan_index)
                #print('poslastscan', poslastscan)
                newascanindex = self.frameparent.list_ascanscan_indices[min(poslastscan + 1, nbascans - 1)]
                #print('newascanindex', newascanindex)

                self.OnSelChanged(event, scan_index=newascanindex)

            elif self.scantype == 'MESH':
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
            if self.scantype == 'ASCAN':
                nbascans = len(self.frameparent.list_ascanscan_indices)
                poslastscan = self.frameparent.list_ascanscan_indices.index(self.last_sel_scan_index)
                newascanindex = self.frameparent.list_ascanscan_indices[max(poslastscan - 1, 0)]

                self.OnSelChanged(event, scan_index=newascanindex)

            elif self.scantype == 'MESH':
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
        self.frameparent.plotmeshpanel.fig.clear() # to do for wxpython 4.1.1
        
        if self.frameparent.filetype == 'spec':
            self.SelChangedSpec(event, scan_index=scan_index)
        else:
            #print('self.frameparent.listmesh',self.frameparent.listmesh)
            self.SelChangedHdf5(event, item=None)

    def SelChangedHdf5(self, event, item=None):
        """ read item for HDF5 tree    """
        #print('\n\n SelChangeHdf5')
        if item is None:
            if event != 1:
                item = event.GetItem()
        
        if item is None:
            return
        selected_item = self.tree.GetItemText(item)
        #scan_index = int(selected_item)
        print("item selected: ", selected_item)
        #print("selected_item ", dir(item))
        
        # multipe selection with ascan's  NOT YET WORKING....
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

        # tooltip----------------
        speccommand = self.frameparent.scancommand
        date = self.frameparent.scan_date
        # print('speccommand', speccommand)
        # print('date',date)
        tooltip = "command: %s\n date: %s" % (speccommand, date)
        #print('tooltip',tooltip)
        event.GetEventObject().SetToolTipString(tooltip)
        event.Skip()
        #------------------
    
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

        self.treemesh = TreePanel(self.panel, scantype="MESH", _id=0, size=(250,-1))
        self.treeascan = TreePanel(self.panel, scantype="ASCAN", _id=1, size=(50,-1))

        self.updatelistbtn = wx.Button(self.panel, -1, "Update scans list")
        self.updatelistbtn.Bind(wx.EVT_BUTTON, self.onUpdateSpecFile)

        self.txtselectionmode = wx.StaticText(self.panel,-1,'Selection mode: Single')

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.toggleBtn = wx.Button(self.panel, wx.ID_ANY, "Real Time Plot")
        self.toggleBtn.Bind(wx.EVT_BUTTON, self.onToggle)

        # --- ----------tooltip
        self.updatelistbtn.SetToolTipString("Refresh list of scan from spec file")
        self.toggleBtn.SetToolTipString("On/Off Real time plot")
        self.txtselectionmode.SetToolTipString("Press s to toggle single/multi ascan selection")
        # --- ----------layout
        # hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        # hbox0.Add(self.treemesh, 1, wx.LEFT | wx.TOP | wx.GROW)
        # hbox0.Add(self.treeascan, 0, wx.EXPAND)
        
        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        hbox0.Add(self.treemesh, 0, wx.EXPAND)
        hbox0.Add(self.treeascan, 0, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(hbox0, 1, wx.LEFT | wx.TOP | wx.GROW)
        vbox.Add(self.updatelistbtn, 0, wx.BOTTOM)
        vbox.Add(self.txtselectionmode, 0, wx.BOTTOM)
        vbox.Add(self.toggleBtn, 0, wx.BOTTOM)
        #         vbox.Add(self.stopbtn, 0, wx.BOTTOM)

        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(vbox, 1, wx.LEFT | wx.TOP | wx.GROW)#wx.EXPAND)
        self.hbox.Add(self.plotmeshpanel, 0, wx.LEFT | wx.TOP | wx.GROW)
        self.hbox.Add(self.plotascanpanel, 0, wx.LEFT | wx.TOP | wx.GROW)

        bigvbox = wx.BoxSizer(wx.VERTICAL)
        bigvbox.Add(self.hbox, 1, wx.LEFT | wx.TOP | wx.GROW)
        bigvbox.Add(self.stbar0, 0, wx.EXPAND)

        self.panel.SetSizer(bigvbox)
        bigvbox.Fit(self)
        self.Layout()

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

            self.fullpath_hdf5file = folder.GetPath()
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
        for meshelems in self.listmeshtoAdd:
            self.treemesh.tree.AppendItem(self.treemesh.root, str(meshelems[0]))
        for ascanelems in self.listascantoAdd:
            self.treeascan.tree.AppendItem(self.treeascan.root, str(ascanelems[0]))


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

            if listmeshall == []:
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
            listmeshall  = logfile_reader.getmeshscan_from_hdf5file(fullpathfilename)

            if listmeshall == []:
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

        if listascanall == []:
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

                counterintensity1D = datay / (data_I0 / (exposureTime / 1.0) - self.MonitorOffset)

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
                zz.put(range(nbacc), counterintensity1D)
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
            self.plotascanpanel.combocounters.Clear()
            self.plotascanpanel.combocounters.AppendItems(columns_name)

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

        if self.filetype=='spec':
            scanheader, data, self.scan_date = ReadSpec(self.fullpath_specfile, scan_index, outputdate=True)
            tit = str(scanheader)
            self.scantype = tit.split()[2]

        elif self.filetype == 'hdf5':  # only mesh now
            print('in ReadScan_SpecFile for hdf5 branch')

            print('scan_index',scan_index)
            tit, data, posmotors, fullpath, self.scan_date= readdata_from_hdf5key(self.listmesh, scan_index, outputdate=True)
            print('tit',tit)
            # scanheader, data, self.scan_date = ReadHdf5(self.fullpath_hdf5file, scan_index, outputdate=True)
            
            self.scantype = 'amesh'

        # print("splec/bliss" command    :", tit)
        # print("scan type  :", self.scantype)

        Apptitle = ''

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

                counterintensity1D = datay / (data_I0 / (exposureTime / 1.0) - self.MonitorOffset)

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
                zz.put(range(nbacc), counterintensity1D)
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

            Apptitle += "%s\nascan #%d" % (self.specfilename, scan_index)

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
                self.plotascanpanel.combocounters.Clear()
                self.plotascanpanel.combocounters.AppendItems(columns_name)

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
                # motor names
                motor1 = tit.split()[1]
                motor2 = tit.split()[5]


                # nb of steps in both directions
                nb1 = int(tit.split()[4]) + 1
                nb2 = int(tit.split()[8]) + 1

            # motor positions
            posmotor1 = np.fix(data[motor1] * 100000) / 100000
            posmotor2 = np.fix(data[motor2] * 100000) / 100000
            # current nb of collected points in the mesh
            nbacc = len(data[list(data.keys())[0]])
            print("nb of points accumulated  :", nbacc)

            if detectorname_mesh not in data:
                Apptitle += '%s NOT AVAILABLE\n'%detectorname_mesh
                if self.filetype == 'spec':
                    detectorname_mesh = 'Monitor'
                
                else:
                    detectorname_mesh = 'epoch'

            counterintensity1D = data[detectorname_mesh]

            if self.normalizeintensity:
                if self.filetype == 'spec':
                    data_I0 = data["Monitor"]
                    exposureTime = data["Seconds"]
                else:
                    data_I0 = data["mon"]
                    exposureTime = data["integration_time"]
                datay = counterintensity1D

                # self.MonitorOffset  in counts / sec

                counterintensity1D = datay / (data_I0 / (exposureTime / 1.0) - self.MonitorOffset)

            print("building arrays")
            if nb2 * nb1 == nbacc:
                print("scan is finished")
                data_z_values = np.reshape(counterintensity1D, (nb2, nb1))
                if 'img' in data:
                    data_img = np.reshape(data["img"], (nb2, nb1))
                elif 'img_scmos' in data:
                    data_img = np.reshape(data["img_scmos"], (nb2, nb1))
                else:
                    print("'img' column doesn't exist! Add fake dummy 0 value")
                    data_img = np.zeros((nb2, nb1))
                posmotorsinfo = np.reshape(np.array([posmotor1, posmotor2]).T, (nb2, nb1, 2))
                scan_in_progress = False

            else:
                print("scan has been aborted")
                print("filling data with zeros...")
                # intensity array
                zz = np.zeros(nb2 * nb1)
                zz.put(range(nbacc), counterintensity1D)
                data_z_values = np.reshape(zz, (nb2, nb1))
                # image index array
                data_img = np.zeros(nb2 * nb1)
                if 'img' in data:
                    data_img.put(range(nbacc), data["img"])
                elif 'img_scmos' in data:
                    data_img.put(range(nbacc), data["img_scmos"])
                else:
                    print("'img' column doesn't exist! Add fake dummy 0 value")
                    data_img.put(range(nbacc), 0)
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
                Apptitle += "%s\nmesh scan #%d" % (self.specfilename, scan_index)
            else:
                Apptitle += "%s\nmesh scan #%s" % (self.hdf5filename, scan_index)
                
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
                self.minmotor1 = float(titlesplit[2])
                self.maxmotor1 = float(titlesplit[3])
                self.minmotor2 = float(titlesplit[6])
                self.maxmotor2 = float(titlesplit[7])

            scancommandextremmotorspositions = [self.minmotor1,
                                                self.maxmotor1,
                                                self.minmotor2,
                                                self.maxmotor2]

            if resetlistcounters:
                # counter and key name of data
                columns_name = list(data.keys())
                columns_name = sorted(columns_name)
                self.plotmeshpanel.combocounters.Clear()
                self.plotmeshpanel.combocounters.AppendItems(columns_name)

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

        if imshowmode is Falsze (scattermode): listscanindex must  be provided with list of scan
        index along y axis and also obviously data_z_values = list of z values and   posmotorsinfo list of x values"""
        #         self.plot.fig.clear()

        self.plotmeshpanel.data = data_z_values
        self.plotmeshpanel.posarray_twomotors = posmotorsinfo
        self.plotmeshpanel.motor1name, self.plotmeshpanel.motor2name = motor1, motor2
        self.plotmeshpanel.absolute_motorposition_unit = "mm"
        self.plotmeshpanel.title = Apptitle
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
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.canvas.mpl_connect("button_press_event", self.onClick)

    def create_axes(self):
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

        # --- --------layout
        htoolbar2 = wx.BoxSizer(wx.HORIZONTAL)
        htoolbar2.Add(countertxt, 0)
        htoolbar2.Add(self.combocounters, 0)
        htoolbar2.Add(self.normalizechckbox, 0)
        htoolbar2.Add(self.I0offsettxt, 0)
        htoolbar2.Add(self.I0offsetctrl, 0)

        htoolbar = wx.BoxSizer(wx.HORIZONTAL)
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
        self.axes.set_title(self.title)
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

        self.frameparent.stbar.SetStatusText('(x,y) = (%f, %f)' % (x, y), 3)
        return

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
            self.axes.set_xlabel("%s (micron)" % self.motor1name)
            self.axes.set_ylabel("%s (micron)" % self.motor2name)

        self.axes.get_xaxis().set_major_formatter(FuncFormatter(formatterfunc_x))
        self.axes.get_yaxis().set_major_formatter(FuncFormatter(formatterfunc_y))
        self.axes.format_coord = self.format_coord
        self.fig.set_canvas(self.canvas)

        # reset ticks and motors positions  ---------------

        self.canvas.draw()

    def init_figurecanvas(self):
        self.dpi = 100
        self.figsize = 4
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.canvas.mpl_connect("button_press_event", self.onClick)

    def create_axes(self):
        self.axes = self.fig.add_subplot(111)

    def create_main_panel(self):
        """
        set main panel of ImshowPanel
        """
        self.toolbar = NavigationToolbar(self.canvas)

        self.IminDisplayed = 0
        self.ImaxDisplayed = 100
        #         if self.datatype == 'scalar':
        self.slidertxt_min = wx.StaticText(self, -1, "Min :")
        self.slider_min = wx.Slider(self, -1, size=(200, 50), value=self.IminDisplayed,
                                                                minValue=0,
                                                                maxValue=99,
                                                                style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_min.SetTickFreq(50)
        else:
            self.slider_min.SetTickFreq(50, 1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMin, self.slider_min)

        self.slidertxt_max = wx.StaticText(self, -1, "Max :")
        self.slider_max = wx.Slider(self, -1, size=(200, 50), value=self.ImaxDisplayed,
                                                            minValue=1,
                                                            maxValue=100,
                                                            style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        if WXPYTHON4:
            self.slider_max.SetTickFreq(50)
        else:
            self.slider_max.SetTickFreq(50, 1)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMax, self.slider_max)

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

        tipmin = "Minimum of intensity mapping"
        self.slidertxt_min.SetToolTipString(tipmin)
        self.slider_min.SetToolTipString(tipmin)
        tipmax = "Maximum of intensity mapping"
        self.slidertxt_max.SetToolTipString(tipmax)
        self.slider_max.SetToolTipString(tipmax)
        # --- --------layout
        self.slidersbox = wx.BoxSizer(wx.HORIZONTAL)
        self.slidersbox.Add(self.slidertxt_min, 0)
        self.slidersbox.Add(self.slider_min, 0)
        self.slidersbox.Add(self.slidertxt_max, 0)
        self.slidersbox.Add(self.slider_max, 0)
        self.slidersbox.AddSpacer(5)
        self.slidersbox.Add(luttxt, 0)
        self.slidersbox.Add(self.comboLUT, 0)

        htoolbar2 = wx.BoxSizer(wx.HORIZONTAL)
        htoolbar2.Add(countertxt, 0)
        htoolbar2.Add(self.combocounters, 0)
        htoolbar2.Add(self.normalizechckbox, 0)
        htoolbar2.Add(self.I0offsettxt, 0)
        htoolbar2.Add(self.I0offsetctrl, 0)

        htoolbar = wx.BoxSizer(wx.HORIZONTAL)
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

    def OnSliderMin(self, _):
        """ normalize plot according to vmin"""
        self.IminDisplayed = int(self.slider_min.GetValue())
        if self.IminDisplayed > self.ImaxDisplayed:
            self.slider_min.SetValue(self.ImaxDisplayed - 1)
            self.IminDisplayed = self.ImaxDisplayed - 1

        self.normalizeplot()
        self.canvas.draw()

    def OnSliderMax(self, _):
        """ normalize plot according to vmax"""
        self.ImaxDisplayed = int(self.slider_max.GetValue())
        if self.ImaxDisplayed < self.IminDisplayed:
            self.slider_max.SetValue(self.IminDisplayed + 1)
            self.ImaxDisplayed = self.IminDisplayed + 1
        self.normalizeplot()
        self.canvas.draw()

    def normalizeplot(self):
        """normalize intensity scale according to GUI widgets parameters"""

        # print('self.minvals', self.minvals)
        # print('self.maxvals', self.maxvals)
        # print('self.IminDisplayed', self.IminDisplayed)
        # print('self.ImaxDisplayed', self.ImaxDisplayed)

        vmin = self.minvals + self.IminDisplayed * self.deltavals
        vmax = self.minvals + self.ImaxDisplayed * self.deltavals

        if self.scaletype == "Linear":

            self.cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.scaletype == "Log10":
            self.cNorm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        else:
            self.cNorm = None

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

        data_wo_nan = self.data_to_Display[np.logical_not(np.isnan(self.data_to_Display))]
        self.maxvals = np.amax(data_wo_nan)
        self.minvals = np.amin(data_wo_nan)
        self.deltavals = (self.maxvals - self.minvals) / 100.0

        self.cNorm = colors.Normalize(vmin=self.minvals, vmax=self.maxvals)


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
        self.axes.set_title(self.title)
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
        # absolute positions ticks
        step_factor = self.step_factor
        return (np.fix((index * self.step_x / step_factor + self.posmotor1[0]) * 100000.0)
            / 100000.0)

    def fromindex_to_pixelpos_x_relative_corner(self, index, _):

        step_factor = self.step_factor
        # relative positions from bottom left corner ticks

        # print("self.step_x", self.step_x)

        return np.fix((index * self.step_x) * 10000.0) / 10000.0

    #         return np.fix((index * self.step_x) * 100.) / 100.

    def fromindex_to_pixelpos_y_absolute(self, index, _):
        # absolute positions ticks
        step_factor = self.step_factor
        # print("step_factor", step_factor)
        # print("self.step_x", self.step_x)
        return (np.fix((index * self.step_y / step_factor + self.posmotor2[0]) * 100000.0)
            / 100000.0)

    def fromindex_to_pixelpos_y_relative_corner(self, index, _):

        step_factor = self.step_factor
        # relative positions from bottom left corner ticks

        # print("self.step_y", self.step_y)

        return np.fix((index * self.step_y) * 10000.0) / 10000.0

    #         return np.fix((index * self.step_y) * 100.) / 100.

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
                sentence0 = ("j=%d, i=%d, ABSOLUTE=[%s=%.5f,%s=%.5f], z_intensity = %s, ImageIndex: %d"
                                % (col, row, self.motor1name, posmotor1[col],
                                    self.motor2name, posmotor2[row], str(z), Imageindex))

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
        frame = MainFrame(None, -1, "plotmeshspecGUI.py")
        frame.Show(True)
        self.SetTopWindow(frame)
        return True


def start():
    """ launcher of plotmeshspecGUI as module"""
    GUIApp = wx.App()
    frame = MainFrame(None, -1, "plotmeshspecGUI.py")
    frame.Show()
    GUIApp.MainLoop()


if __name__ == "__main__":
    start()