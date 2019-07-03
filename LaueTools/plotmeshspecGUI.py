#!/usr/bin/python

# plotmeshspecGUI.py

import os, sys
import time


import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

import numpy as np

import matplotlib

matplotlib.use("WXAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

import matplotlib.colors as colors

from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from pylab import cm as pcm

if sys.version_info.major == 3:
    from . import generaltools as GT
    from . IOLaueTools import ReadSpec
else:
    import generaltools as GT
    from IOLaueTools import ReadSpec


class TreePanel(wx.Panel):
    def __init__(self, parent, scantype=None):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        #     def __init__(self, parent, id, title):
        #         wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(450, 350))

        self.parent = parent
        self.scantype = scantype
        self.frameparent = self.parent.GetParent()
        self.tree = wx.TreeCtrl(
            self, -1, wx.DefaultPosition, (-1, -1), wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS
        )

        self.maketree()

        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelChanged)

        # wx.EVT_TREE_ITEM_RIGHT_CLICK
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.tree, 1, wx.EXPAND)
        vbox.AddSpacer(1)

        self.SetSizer(vbox)

    #         self.Centre()

    def maketree(self):
        self.root = self.tree.AddRoot("SpecFiles")
        self.tree.AppendItem(self.root, str(self.scantype))

    def OnSelChanged(self, event):
        item = event.GetItem()
        selected_item = self.tree.GetItemText(item)
        if selected_item in (str(self.scantype),):
            return
        scan_index = int(selected_item)
        print("click on ", scan_index)

        self.frameparent.scan_index = scan_index
        self.frameparent.ReadScan_SpecFile(scan_index)


# --- ---------------  Plot limits board  parameters
class MessageCommand(wx.Dialog):
    """
    Class to command with spec
    """

    def __init__(
        self, parent, _id, title, sentence=None, speccommand=None, specconnection=None
    ):
        """
        initialize board window
        """
        wx.Dialog.__init__(self, parent, _id, title, size=(400, 250))

        self.parent = parent
        print("self.parent", self.parent)

        self.speccommand = speccommand

        txt1 = wx.StaticText(self, -1, "%s\n\n%s" % (sentence, self.speccommand))

        acceptbtn = wx.Button(self, -1, "OK")
        tospecbtn = wx.Button(self, -1, "Send to Spec")
        cancelbtn = wx.Button(self, -1, "Cancel")

        acceptbtn.Bind(wx.EVT_BUTTON, self.onAccept)
        cancelbtn.Bind(wx.EVT_BUTTON, self.onCancel)
        tospecbtn.Bind(wx.EVT_BUTTON, self.onCommandtoSpec)

        btnssizer = wx.BoxSizer(wx.HORIZONTAL)
        btnssizer.Add(acceptbtn, 0, wx.ALL)
        btnssizer.Add(cancelbtn, 0, wx.ALL)
        btnssizer.Add(tospecbtn, 0, wx.ALL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt1)
        vbox.Add(btnssizer)
        self.SetSizer(vbox)

    def onAccept(self, evt):

        self.Close()

    def onCancel(self, evt):

        # todo save old positions and make inverse mvt
        self.Close()

    def onCommandtoSpec(self, evt):
        try: 
            from SpecClient_gevent import SpecCommand
        except:
            return

        myspec = SpecCommand.SpecCommand("", "crg1:laue")

        print("Sending command : " + self.speccommand)

        myspec.executeCommand(self.speccommand)

        self.Close()


class MainFrame(wx.Frame):
    """
    Class to show CCD frame pixel intensities
    and provide tools for searching peaks
    """

    def __init__(self, parent, _id, title, size=4):
        wx.Frame.__init__(self, parent, _id, title, size=(600, 1000))

        self.folderpath_specfile, self.specfilename = None, None

        self.detectorname = "Monitor"
        self.columns_name = ["Monitor", "fluoHg"]
        self.normalizeintensity = False

        self.createMenuBar()
        self.create_main_panel()

        self.listmesh = None

    def createMenuBar(self):
        menubar = wx.MenuBar()

        filemenu = wx.Menu()
        menuSpecFile = filemenu.Append(-1, "Open spec file", "Open a spec file")
        menuSetPreference = filemenu.Append(
            -1, "Folder Preferences", "Set folder Preferences"
        )
        savemeshdata = filemenu.Append(-1, "Save Data", "Save current 2D data")
        self.Bind(wx.EVT_MENU, self.OnOpenSpecFile, menuSpecFile)
        self.Bind(wx.EVT_MENU, self.OnSaveData, savemeshdata)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuSetPreference)

        #         displayprops = wx.Menu()
        #         menudisplayprops = displayprops.Append(-1, "Set Plot Size",
        #                                          "Set Minimal plot size to fit with small computer screen")
        #         self.Bind(wx.EVT_MENU, self.OnAbout, menudisplayprops)

        helpmenu = wx.Menu()

        menuAbout = helpmenu.Append(
            wx.ID_ABOUT, "&About", " Information about this program"
        )
        menuExit = helpmenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Set events.
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

        menubar.Append(filemenu, "&File")
        #         menubar.Append(displayprops, '&Display Props')
        menubar.Append(helpmenu, "&Help")

        self.SetMenuBar(menubar)

    def OnAbout(self, evt):
        print("open spec file")
        pass

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

        self.stbar = self.CreateStatusBar(3)

        self.stbar.SetStatusWidths([180, -1, -1])
        #         print dir(self.stbar)

        self.stbar0 = wx.StatusBar(self.panel)

        self.plot = ImshowPanel(
            self.panel,
            -1,
            "test_plot",
            z_values,
            Imageindices=Imageindices,
            posmotorname=("xmotor", "ymotor"),
            posarray_twomotors=posmotor,
            absolute_motorposition_unit="mm",
        )

        self.treespecfiles = TreePanel(self.panel, scantype="MESH")
        self.treeacanspecfiles = TreePanel(self.panel, scantype="ASCAN")

        self.updatelistbtn = wx.Button(self.panel, -1, "Update scans list")
        self.updatelistbtn.Bind(wx.EVT_BUTTON, self.onUpdateSpecFile)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.toggleBtn = wx.Button(self.panel, wx.ID_ANY, "Real Time Plot")
        self.toggleBtn.Bind(wx.EVT_BUTTON, self.onToggle)
        #         self.toggleBtn.Bind(wx.EVT_BUTTON, self.onOnlinePlot)

        #         self.stopbtn = wx.Button(self.panel, wx.ID_ANY, "Stop")
        #         self.stopbtn.Bind(wx.EVT_BUTTON, self.onStopTimer)
        #         self.stopbtn.Disable()

        # --- ----------tooltip
        self.updatelistbtn.SetToolTipString("Refresh list of scan from spec file")
        self.toggleBtn.SetToolTipString("On/Off Real time plot")
        # --- ----------layout
        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        hbox0.Add(self.treespecfiles, 1, wx.LEFT | wx.TOP | wx.GROW)
        hbox0.Add(self.treeacanspecfiles, 0, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(hbox0, 1, wx.LEFT | wx.TOP | wx.GROW)
        vbox.Add(self.updatelistbtn, 0, wx.BOTTOM)
        vbox.Add(self.toggleBtn, 0, wx.BOTTOM)
        #         vbox.Add(self.stopbtn, 0, wx.BOTTOM)

        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(vbox, 0, wx.EXPAND)
        self.hbox.Add(self.plot, 1, wx.LEFT | wx.TOP | wx.GROW)

        bigvbox = wx.BoxSizer(wx.VERTICAL)
        bigvbox.Add(self.hbox, 1, wx.LEFT | wx.TOP | wx.GROW)
        bigvbox.Add(self.stbar0, 0, wx.EXPAND)

        self.panel.SetSizer(bigvbox)
        bigvbox.Fit(self)
        self.Layout()

    def OnSaveData(self, evt):

        defaultdir = ""
        if not os.path.isdir(defaultdir):
            defaultdir = os.getcwd()

        file = wx.FileDialog(
            self,
            "Save 2D Array Data in File",
            defaultDir=defaultdir,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if file.ShowModal() == wx.ID_OK:

            outputfile = file.GetPath()

            #             currentfig = self.plot.fig
            #             currentfig.savefig(outputfile)
            #             print "Image saved in ", outputfile + '.png'

            #         self.flat_data_z_values = data[detectorname]
            #         self.flat_motor1 = posmotor1
            #         self.flat_motor2 = posmotor2

            self.writefile_3columns(
                outputfile,
                [
                    self.flat_data_z_values.tolist(),
                    self.flat_motor1.tolist(),
                    self.flat_motor2.tolist(),
                ],
            )

    def writefile_3columns(self, output_filename, data):
        """
        Write  file containing data = [list_1,list_2,list_3]
        """
        longueur = len(data[0])

        outputfile = open(output_filename, "w")

        outputfile.write("data_z posmotor1 posmotor2\n")

        outputfile.write(
            "\n".join(
                [
                    "%.06f   %.06f   %06f" % tuple(list(zip(data[0], data[1], data[2])[i]))
                    for i in range(longueur)
                ]
            )
        )

        outputfile.write(
            "\n# File created at %s with PlotmeshspecGUI.py" % (time.asctime())
        )

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

    def onUpdateSpecFile(self, evt):

        self.folderpath_specfile, self.specfilename = os.path.split(
            self.fullpath_specfile
        )

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

    def OnOpenSpecFile(self, evt):

        folder = wx.FileDialog(
            self,
            "Select spec file",
            wildcard="BM32 Specfile (laue.*)|laue.*|All files(*)|*",
            defaultDir=str(self.folderpath_specfile),
            defaultFile=str(self.specfilename),
        )

        self.last_specfilename = self.specfilename
        if folder.ShowModal() == wx.ID_OK:

            self.fullpath_specfile = folder.GetPath()

            #             print "folder.GetPath()", abs_fullpath

            self.folderpath_specfile, self.specfilename = os.path.split(
                self.fullpath_specfile
            )

        if (
            self.specfilename == self.last_specfilename
            and self.specfilename is not None
        ):
            self.onUpdateSpecFile(1)

        self.readspecfile(self.fullpath_specfile)

        #         print dir(self.treespecfiles.tree)

        if (
            self.specfilename != self.last_specfilename
            and self.specfilename is not None
        ):
            print("\n\ndeleting last old items\n\n")
            self.treespecfiles.tree.DeleteAllItems()
            wx.CallAfter(self.treespecfiles.maketree)

        #         print "dtt", dir(self.treespecfiles.tree)
        self.listmeshtoAdd = self.listmesh
        wx.CallAfter(self.fill_tree)

    def fill_tree(self):
        for meshelems in self.listmeshtoAdd:
            self.treespecfiles.tree.AppendItem(
                self.treespecfiles.root, str(meshelems[0])
            )

    def readspecfile(self, fullpathspecfilename):
        samespecfile = False
        if self.specfilename != self.last_specfilename:
            samespecfile = True
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

        listmeshall = getmeshscan_from_specfile(fullpathspecfilename)

        list_meshscan_indices = []
        for ms in listmeshall:
            if ms[0] not in list_lastmeshscan_indices:
                list_meshscan_indices.append(ms[0])

        print("list_meshscan_indices", list_meshscan_indices)

        if list_meshscan_indices[-1] != lastscan_listmesh and not samespecfile:
            print("adding only new meshes from file %s" % self.fullpath_specfile)
            indstart_newmeshes = np.searchsorted(
                list_meshscan_indices, lastscan_listmesh - 1
            )
        else:
            indstart_newmeshes = 0

        print("listmeshall", listmeshall)
        print("indstart_newmeshes", indstart_newmeshes)
        self.listmesh = listmeshall[indstart_newmeshes:]

    def ReadScan_SpecFile(self, scan_index):
        """
        read scan data in spec file and fill data for a updated figure plot
        """
        detectorname = self.detectorname

        scanheader, data = ReadSpec(self.fullpath_specfile, scan_index)
        tit = str(scanheader)

        print("spec command", tit)

        print("tit.split()")

        titlesplit = tit.split()
        minmotor1 = float(titlesplit[4])
        maxmotor1 = float(titlesplit[5])
        minmotor2 = float(titlesplit[8])
        maxmotor2 = float(titlesplit[9])

        # counter and key name of data
        columns_name = list(data.keys())
        self.columns_name = sorted(columns_name)
        # motor names
        motor1 = tit.split()[3]
        motor2 = tit.split()[7]
        # motor positions
        posmotor1 = np.fix(data[motor1] * 100000) / 100000
        posmotor2 = np.fix(data[motor2] * 100000) / 100000

        # nb of steps in both directions
        nb1 = int(tit.split()[6]) + 1
        nb2 = int(tit.split()[10]) + 1
        # current nb of collected points in the mesh
        nbacc = len(data[list(data.keys())[0]])
        print("nb of points accumulated  :", nbacc)

        counterintensity1D = data[detectorname]

        if self.normalizeintensity:
            data_I0 = data["Monitor"]
            exposureTime = data["Seconds"]
            datay = counterintensity1D

            # self.MonitorOffset  in counts / sec

            counterintensity1D = datay / (
                data_I0 / (exposureTime / 1.0) - self.MonitorOffset
            )

        print("building arrays")
        if nb2 * nb1 == nbacc:
            print("scan is finished")
            data_z_values = np.reshape(counterintensity1D, (nb2, nb1))
            try:
                data_img = np.reshape(data["img"], (nb2, nb1))
            except KeyError:
                print("'img' column doesn't exist! Add fake dummy 0 value")
                data_img = np.zeros((nb2, nb1))
            posmotorsinfo = np.reshape(
                np.array([posmotor1, posmotor2]).T, (nb2, nb1, 2)
            )
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
            try:
                data_img.put(range(nbacc), data["img"])
            except KeyError:
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

        Apptitle = "%s\nmesh scan #%d" % (self.specfilename, scan_index)

        print("title", Apptitle)

        self.flat_data_z_values = counterintensity1D
        self.flat_motor1 = posmotor1
        self.flat_motor2 = posmotor2

        self.scancommand = tit
        self.minmotor1 = float(titlesplit[4])
        self.maxmotor1 = float(titlesplit[5])
        self.minmotor2 = float(titlesplit[8])
        self.maxmotor2 = float(titlesplit[9])

        scancommandextremmotorspositions = [
            self.minmotor1,
            self.maxmotor1,
            self.minmotor2,
            self.maxmotor2,
        ]

        self.plot.combocounters.Clear()
        self.plot.combocounters.AppendItems(self.columns_name)

        self.update_fig(
            data_z_values,
            posmotorsinfo,
            motor1,
            motor2,
            Apptitle,
            data_img,
            detectorname,
            scancommandextremmotorspositions,
        )

        return scan_in_progress

    def update_fig(
        self,
        data_z_values,
        posmotorsinfo,
        motor1,
        motor2,
        Apptitle,
        data_img,
        detectorname,
        scancommandextremmotorspositions,
    ):
        """update fig and plot"""
        #         self.plot.fig.clear()

        self.plot.data = data_z_values
        self.plot.posarray_twomotors = posmotorsinfo
        self.plot.motor1name, self.plot.motor2name = motor1, motor2
        self.plot.absolute_motorposition_unit = "mm"
        self.plot.title = Apptitle
        self.plot.Imageindices = data_img

        (
            self.plot.minmotor1,
            self.plot.maxmotor1,
            self.plot.minmotor2,
            self.plot.maxmotor2,
        ) = scancommandextremmotorspositions

        self.plot.xylabels = ("column index", "row index")
        self.plot.datatype = "scalar"

        if self.plot.colorbar is not None:
            self.plot.colorbar_label = detectorname
            (self.plot.myplot, self.plot.colorbar, self.plot.data) = makefig_update(
                self.plot.fig, self.plot.myplot, self.plot.colorbar, data_z_values
            )
        else:
            print("self.plot.colorbar is None")
            self.plot.create_axes()

            self.plot.calc_norm_minmax_values(self.plot.data)
            self.plot.clear_axes_create_imshow()

        # reset ticks and motors positions  ---------------

        self.plot.draw_fig()
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

    def update(self, event, worker=None):
        """
        update at each time step time
        """
        print("\nupdated: ")
        print(time.ctime())
        if self.scan_in_progress:
            self.scan_in_progress = self.ReadScan_SpecFile(self.scan_index)
            return True
        else:
            print("waiting for data  for scan :%d" % self.scan_index)
            # stop the first timer
            return False

    #     def onStopTimer(self, evt):
    #         self.timer.Stop()
    #         print "EVT_TIMER timer stoped\n"
    # #         del self.timer
    #         self.stopbtn.Disable()
    #         self.scan_in_progress = False
    #         self.toggleBtn.SetLabel("OnLinePlot")
    #         self.OnFlyMode = False

    def onOnlinePlot(self, evt):
        """
        not used
        """
        USETHREAD = 1
        if USETHREAD:
            # with a thread 2----------------------------------------
            import threadGUI2 as TG

            self.worker = None
            self.results = None
            self.scan_in_progress = True
            fctparams = [self.update2, (evt,), {}]

            self.TGframe = TG.ThreadHandlingFrame(
                self,
                -1,
                threadFunctionParams=fctparams,
                parentAttributeName_Result="results",
                parentNextFunction=self.plot.canvas.draw,
            )
            self.TGframe.OnStart(1)
            self.TGframe.Show(True)

            # will set self.UBs_MRs to the output of INDEX.getUBs_and_MatchingRate
            return

    def update2(self, event, worker=None):
        """
        not used 
        update at each time step time
        """
        print("\nupdated: ")
        print(time.ctime())
        WORKEREXISTS = False
        if worker is not None:
            WORKEREXISTS = True
        while self.scan_in_progress:
            self.scan_in_progress = self.ReadScan_SpecFile(self.scan_index)
            if WORKEREXISTS and self.scan_in_progress:
                if worker._want_abort:
                    self.scan_in_progress = False
                    worker.callbackfct(None)
                    return

            print("waiting for data  for scan :%d" % self.scan_index)
        if WORKEREXISTS:
            worker.fctOutputResults = "OK"

            print("finished!")
            print("setting worker.fctOutputResults to", worker.fctOutputResults)
            worker.callbackfct("COMPLETED")
        return "OK"


def makefig_update(fig, myplot, cbar, data):
    if myplot:
        print("\n\n\nmyplot exists\n\n\n")
        # data *= 2  # change data, so there is change in output (look at colorbar)
        myplot.set_data(data)  # use this if you use new array
        myplot.autoscale()
        # cbar.update_normal(myplot) #cbar is updated automatically
    else:
        ax = fig.add_subplot(111)
        myplot = ax.imshow(data)
        cbar = fig.colorbar(myplot)
    return myplot, cbar, data


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
            #             print "line", line
            linesplit = line.split()
            if linesplit[2] == "mesh":
                #                 print "line", line
                scan_index = int(linesplit[1])
                listmesh.append([scan_index, linepos, f.tell(), line])

        linepos += 1

    f.close()
    print("%d lines have been read" % (linepos))
    print("%s contains %d mesh scans" % (filename, len(listmesh)))

    #     print "listmesh", listmesh

    return listmesh


class ImshowPanel(wx.Panel):
    """
    Class to show 2D array intensity data
    """

    def __init__(
        self,
        parent,
        _id,
        title,
        dataarray,
        posarray_twomotors=None,
        posmotorname=(None, None),
        datatype="scalar",
        absolutecornerindices=None,
        Imageindices=None,
        absolute_motorposition_unit="micron",
        colorbar_label="Fluo counts",
        stepindex=1,
        xylabels=None,
    ):
        """
        plot 2D plot of dataarray
        """
        USE_COLOR_BAR = False

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.parent = parent
        print("parent", parent)

        self.frameparent = parent.GetParent()

        self.data = dataarray
        self.data_to_Display = self.data

        self.posarray_twomotors = posarray_twomotors

        print(self.posarray_twomotors[0, 0], self.posarray_twomotors[0, -1])
        print(self.posarray_twomotors[-1, 0], self.posarray_twomotors[-1, -1])
        self.minmotor1, self.minmotor2 = posarray_twomotors[0, 0]
        self.maxmotor1, self.maxmotor2 = posarray_twomotors[-1, -1]

        if posmotorname is not None:
            self.motor1name, self.motor2name = posmotorname

        self.absolute_motorposition_unit = absolute_motorposition_unit
        #         print "dataarray", dataarray
        self.datatype = datatype

        self.absolutecornerindices = absolutecornerindices
        self.title = title
        self.Imageindices = Imageindices

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
        self.clear_axes_create_imshow()

        if USE_COLOR_BAR:
            self.colorbar_label = colorbar_label
            self.colorbar = self.fig.colorbar(self.myplot)

        self.draw_fig()

    def draw_fig(self):
        print("in draw_fig()")

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

    #         print "self.canvas", dir(self.canvas)

    def create_axes(self):
        self.axes = self.fig.add_subplot(111)

    def create_main_panel(self):
        """
        set main panel of ImshowPanel 
        """
        #         self.tooltip = wx.ToolTip(tip='tip with a long %s line and a newline\n' % (' ' * 100))
        #         self.canvas.SetToolTip(self.tooltip)
        #         self.tooltip.Enable(False)
        #         self.tooltip.SetDelay(0)
        #         self.fig.canvas.mpl_connect('motion_notify_event', self.onMotion_ToolTip)

        self.toolbar = NavigationToolbar(self.canvas)

        #         self.calc_norm_minmax_values()

        self.IminDisplayed = 0
        self.ImaxDisplayed = 100
        #         if self.datatype == 'scalar':
        self.slidertxt_min = wx.StaticText(self, -1, "Min :")
        self.slider_min = wx.Slider(
            self,
            -1,
            size=(200, 50),
            value=self.IminDisplayed,
            minValue=0,
            maxValue=99,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        if WXPYTHON4:
            self.slider_min.SetTickFreq(50, 1)
        else:
            self.slider_min.SetTickFreq(50)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMin, self.slider_min)

        self.slidertxt_max = wx.StaticText(self, -1, "Max :")
        self.slider_max = wx.Slider(
            self,
            -1,
            size=(200, 50),
            value=self.ImaxDisplayed,
            minValue=1,
            maxValue=100,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        if WXPYTHON4:
            self.slider_max.SetTickFreq(50, 1)
        else:
            self.slider_max.SetTickFreq(50)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.OnSliderMax, self.slider_max)

        # loading LUTS
        self.mapsLUT = [m for m in pcm.datad if not m.endswith("_r")]
        self.mapsLUT.sort()

        luttxt = wx.StaticText(self, -1, "LUT")
        self.comboLUT = wx.ComboBox(
            self,
            -1,
            self.LastLUT,
            size=(-1, 40),
            choices=self.mapsLUT,
            style=wx.TE_PROCESS_ENTER,
        )  # ,
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
        self.comboscale = wx.ComboBox(
            self, -1, self.scaletype, choices=["Linear", "Log10"], size=(-1, 40)
        )

        self.comboscale.Bind(wx.EVT_COMBOBOX, self.OnChangeScale)

        btnflipud = wx.Button(self, -1, "Flip Vert.")
        btnflipud.Bind(wx.EVT_BUTTON, self.OnChangeYorigin)

        btnfliplr = wx.Button(self, -1, "Flip Hori.")
        btnfliplr.Bind(wx.EVT_BUTTON, self.OnChangeXorigin)

        countertxt = wx.StaticText(self, -1, "counter")

        print("self.frameparent.columns_name", self.frameparent.columns_name)
        sortedcounterslist = sorted(self.frameparent.columns_name)
        self.combocounters = wx.ComboBox(
            self,
            -1,
            self.frameparent.detectorname,
            choices=sortedcounterslist,
            size=(-1, 40),
            style=wx.TE_PROCESS_ENTER,
        )

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

    def OnNormalizeData(self, evt):
        self.frameparent.normalizeintensity = not self.frameparent.normalizeintensity

        self.frameparent.MonitorOffset = float(self.I0offsetctrl.GetValue())

        self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index)

    def onClick(self, event):
        """ onclick
        """
        print(event.button)
        if event.inaxes:
            self.centerx, self.centery = event.xdata, event.ydata
            print("current clicked positions", self.centerx, self.centery)
        if event.button == 3:
            self.movingxy(False)

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

                print(
                    "SPEC COMMAND:\nmv %s %.5f %s %.5f"
                    % (
                        self.motor1name,
                        current_posmotor1,
                        self.motor2name,
                        current_posmotor2,
                    )
                )

                sentence = (
                    "%s=%.6f\n%s=%.6f\n\nSPEC COMMAND to move to this point:\n\nmv %s %.5f %s %.5f"
                    % (
                        self.motor1name,
                        current_posmotor1,
                        self.motor2name,
                        current_posmotor2,
                        self.motor1name,
                        current_posmotor1,
                        self.motor2name,
                        current_posmotor2,
                    )
                )

                command = "mv %s %.5f %s %.5f" % (
                    self.motor1name,
                    current_posmotor1,
                    self.motor2name,
                    current_posmotor2,
                )

                if msgbox == True:
                    wx.MessageBox(sentence + "\n" + command, "INFO")

                # WARNING could do some instabilities to station ??
                msgdialog = MessageCommand(
                    self,
                    -1,
                    "motors command",
                    sentence=sentence,
                    speccommand=command,
                    specconnection=None,
                )
                msgdialog.ShowModal()

    def onKeyPressed(self, event):

        key = event.key
        print("key ==> ", key)

        if key == "escape":

            ret = wx.MessageBox(
                "Are you sure to quit?", "Question", wx.YES_NO | wx.NO_DEFAULT, self
            )

            if ret == wx.YES:
                self.Close()

        elif key == "p":  # 'p'

            self.movingxy(True)

            return

    def OnChangeScale(self, evt):
        self.scaletype = str(self.comboscale.GetValue())
        self.normalizeplot()
        self.canvas.draw()

    def OnChangeLUT(self, event):
        #         print "OnChangeLUT"
        self.cmap = self.comboLUT.GetValue()
        self.myplot.set_cmap(self.cmap)
        self.canvas.draw()

    def OnChangeYorigin(self, event):
        """
        reverse y origin
        """
        self.axes.set_ylim(self.axes.get_ylim()[::-1])
        self.flagyorigin += 1
        self.origin = self.YORIGINLIST[self.flagyorigin % 2]
        self.canvas.draw()

    def OnChangeXorigin(self, event):
        """
        reverse  x limits of plot, and update self.flagxorigin
        """
        self.axes.set_xlim(self.axes.get_xlim()[::-1])
        self.flagxorigin += 1
        self.canvas.draw()

    def OnChangeCounter(self, evt):
        #         print "OnChangeCounter"

        self.detectorname = self.combocounters.GetValue()

        self.frameparent.detectorname = self.detectorname

        self.frameparent.ReadScan_SpecFile(self.frameparent.scan_index)

    def OnSliderMin(self, evt):

        self.IminDisplayed = int(self.slider_min.GetValue())
        if self.IminDisplayed > self.ImaxDisplayed:
            self.slider_min.SetValue(self.ImaxDisplayed - 1)
            self.IminDisplayed = self.ImaxDisplayed - 1

        self.normalizeplot()
        self.canvas.draw()

    def OnSliderMax(self, evt):
        self.ImaxDisplayed = int(self.slider_max.GetValue())
        if self.ImaxDisplayed < self.IminDisplayed:
            self.slider_max.SetValue(self.IminDisplayed + 1)
            self.ImaxDisplayed = self.IminDisplayed + 1
        self.normalizeplot()
        self.canvas.draw()

    def normalizeplot(self):

        vmin = self.minvals + self.IminDisplayed * self.deltavals
        vmax = self.minvals + self.ImaxDisplayed * self.deltavals

        if self.scaletype == "Linear":

            self.cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.scaletype == "Log10":
            self.cNorm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        else:
            self.cNorm = None

        self.myplot.set_norm(self.cNorm)

    def OnSave(self, event):
        # if self.askUserForFilename(defaultFile='truc', style=wx.SAVE,**self.defaultFileDialogOptions()):
        #    self.OnSave(event)
        if self.askUserForFilename():
            fig = self.plotPanel.get_figure()
            fig.savefig(os.path.join(str(self.dirname), str(self.filename)))
            print("Image saved in ", os.path.join(self.dirname, self.filename) + ".png")

    def calc_norm_minmax_values(self, data):
        #         if self.posarray_twomotors is not None:
        #             self.maxvals = np.amax(self.posarray_twomotors)
        #             self.minvals = np.amin(self.posarray_twomotors)
        #
        #
        # #             print 'self.posarray_twomotors', self.posarray_twomotors
        #             print 'self.posarray_twomotors max ', self.maxvals
        #             print 'self.posarray_twomotors min ', self.minvals

        self.data_to_Display = data
        self.cNorm = None

        if data is None:
            return

        print("plot of datatype = %s" % self.datatype)

        self.maxvals = np.amax(self.data_to_Display)
        self.minvals = np.amin(self.data_to_Display)

        self.deltavals = (self.maxvals - self.minvals) / 100.0

        #             from matplotlib.colors import colorConverter

        #             import matplotlib.pyplot as plt
        #             import matplotlib.cm as cmx
        #             jet = cm = plt.get_cmap('jet')
        self.cNorm = colors.Normalize(vmin=self.minvals, vmax=self.maxvals)

    #             scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #             print scalarMap.get_clim()
    #         colorVal = scalarMap.to_rgba(values[idx])

    def forceAspect(self, aspect=1.0):
        im = self.axes.get_images()
        extent = im[0].get_extent()
        self.axes.set_aspect(
            abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
        )

    def re_init_colorbar(self):

        #             print dir(self.colorbar)
        self.colorbar.set_label(self.colorbar_label)
        self.colorbar.set_clim(vmin=self.minvals, vmax=self.maxvals)
        self.colorbar.draw_all()

    def clear_axes_create_imshow(self):
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

            print("ploting")

            print("self.data_to_Display.shape", self.data_to_Display.shape)
            self.myplot = self.axes.imshow(
                self.data_to_Display,
                cmap=self.cmap,
                interpolation="nearest",
                norm=self.cNorm,
                aspect="equal",
                #                              extent=self.extent,
                origin=self.origin,
            )

            if self.XORIGINLIST[self.flagxorigin % 2] == "right":
                self.axes.set_xlim(self.axes.get_xlim()[::-1])

    def fromindex_to_pixelpos_x_absolute(self, index, pos):
        # absolute positions ticks
        step_factor = self.step_factor
        #         print "step_factor", step_factor
        #         print "self.step_x", self.step_x
        return (
            np.fix((index * self.step_x / step_factor + self.posmotor1[0]) * 100000.0)
            / 100000.0
        )

    def fromindex_to_pixelpos_x_relative_corner(self, index, pos):

        step_factor = self.step_factor
        # relative positions from bottom left corner ticks

        print("self.step_x", self.step_x)

        return np.fix((index * self.step_x) * 10000.0) / 10000.0

    #         return np.fix((index * self.step_x) * 100.) / 100.

    def fromindex_to_pixelpos_y_absolute(self, index, pos):
        # absolute positions ticks
        step_factor = self.step_factor
        print("step_factor", step_factor)
        print("self.step_x", self.step_x)
        return (
            np.fix((index * self.step_y / step_factor + self.posmotor2[0]) * 100000.0)
            / 100000.0
        )

    def fromindex_to_pixelpos_y_relative_corner(self, index, pos):

        step_factor = self.step_factor
        # relative positions from bottom left corner ticks

        print("self.step_y", self.step_y)

        return np.fix((index * self.step_y) * 10000.0) / 10000.0

    #         return np.fix((index * self.step_y) * 100.) / 100.

    #     def fromindex_to_pixelpos_y(self, index, pos):
    #         numrows = self.numrows
    #         posmotor2 = self.posmotor2
    #         step_factor = self.step_factor
    # #         poscenter_motor1, poscenter_motor2 = self.poscenter_motor1, self.poscenter_motor2
    #
    #         row = int(index + .5)
    #         if row >= 0 and row < numrows:
    #             return np.round(step_factor * (posmotor2[int(index + .5)] - posmotor2[0]))

    #         factor_ticks = 1.
    #         step_y = 10.
    #         return np.fix((step_y * index) * factor_ticks) / factor_ticks

    #     def calc_spatialpositions(self):
    #
    #         print "in calc_spatialpositions"
    # #         print "self.posarray_twomotors in calc_spatialpositions", self.posarray_twomotors
    #         print "absolute_motorposition_unit", self.absolute_motorposition_unit
    #
    #         print "pos extremes"
    #         print self.posarray_twomotors[0, 0], self.posarray_twomotors[0, -1]
    #         print self.posarray_twomotors[-1, 0], self.posarray_twomotors[-1, -1]
    #
    #         posmotors = self.posarray_twomotors
    #         print "self.posarray_twomotors.shape",self.posarray_twomotors.shape
    #
    #         if posmotors is not None:
    #
    #
    #             initmotor1 = posmotors[0, 0, 0]
    #             initmotor2 = posmotors[0, 0, 1]
    #
    #             posmotor1 = posmotors[0, :, 0]
    #             posmotor2 = posmotors[:, 0, 1]
    #
    # #             print "starting motor1 %f %s" % (initmotor1, self.absolute_motorposition_unit)
    # #             print "starting motor2 %f %s" % (initmotor2, self.absolute_motorposition_unit)
    #
    # #             print 'posmotor1', posmotor1
    # #             print 'posmotor2', posmotor2
    #
    #             nby, nbx = posmotors.shape[:2]
    #
    #             poscenter_motor1 = posmotor1[nbx / 2]
    #             poscenter_motor2 = posmotor2[nby / 2]
    #
    # #             print "center motor1", poscenter_motor1
    # #             print "center motor2", poscenter_motor2
    #
    #             # x= fast motor  (first in spec scan)
    #             # y slow motor (second in spec scan)
    #             step_x = (posmotor1[-1] - posmotor1[0]) / (nbx - 1)
    #             step_y = (posmotor2[-1] - posmotor2[0]) / (nby - 1)
    #
    # #             print "step_x %f %s " % (step_x, self.absolute_motorposition_unit)
    # #             print "step_y %f %s " % (step_y, self.absolute_motorposition_unit)
    #     #         def fromindex_to_pixelpos_x_mosaic(index, pos):
    #     #                 return index  # self.center[0]-self.boxsize[0]+index
    #     #         def fromindex_to_pixelpos_y_mosaic(index, pos):
    #     #                 return index  # self.center[1]-self.boxsize[1]+index
    #
    #             step_factor = 1.
    #             if self.absolute_motorposition_unit == 'mm':
    #                 step_factor = 1000.
    #                 step_x = step_x * step_factor
    #                 step_y = step_y * step_factor
    #
    # #             print "step_x %f micron " % (step_x)
    # #             print "step_y %f micron " % (step_y)
    #
    #             # TRYING fix clever ticks values and locations ------------------
    #             numrows, numcols = self.data.shape[:2]
    # #             print 'numrows, numcols', numrows, numcols
    #
    #
    #
    #             factor_ticks = 1.
    #             def fromindex_to_pixelpos_x_mosaic(index, pos):
    #                     return np.fix((step_x * index) * factor_ticks) / factor_ticks
    #             def fromindex_to_pixelpos_y_mosaic(index, pos):
    #                     return np.fix((step_y * index) * factor_ticks) / factor_ticks
    #
    # #             def fromindex_to_pixelpos_x_mosaic(index, pos):
    # #                     return index
    # #             def fromindex_to_pixelpos_y_mosaic(index, pos):
    # #                     return index
    #
    # #             print 'posmotor1', posmotor1
    # #             print 'posmotor2', posmotor2
    #
    #
    #             def fromindex_to_pixelpos_x_mosaic(index, pos):
    #                 col = int(index + .5)
    #                 if col >= 0 and col < numcols:
    #                     return step_factor * (posmotor1[int(index + .5)] - posmotor1[0])
    #             def fromindex_to_pixelpos_y_mosaic(index, pos):
    #                 row = int(index + 0.5)
    #                 if row >= 0 and row < numrows:
    #                     return step_factor * (posmotor2[int(index + .5)] - posmotor2[0])
    #
    # #
    # #             def fromindex_to_motor1pos(index, pos):
    # #                 return posmotor1[int(index)]
    # #     #             return fix((index * (posmotor1[-1] - initmotor1) / (nb1 - 1) + \
    # #     #                                 initmotor1) * 100000) / 100000
    # #             def fromindex_to_motor2pos(index, pos):
    # #                 return posmotor2[int(index)]
    # #     #             return fix((index * (posmotor2[-1] - initmotor2) / (nb2 - 1) + \
    # #     #                                 initmotor2) * 100000) / 100000
    #             currentaxes = self.fig.gca()
    #
    # #             print "self.fig", self.fig
    # #             print "currentaxes", currentaxes
    #
    # #             print dir(currentaxes)
    # #             print currentaxes
    #     #
    # #             currentaxes.get_xaxis().set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x_mosaic))
    # #             currentaxes.get_yaxis().set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y_mosaic))
    #     #
    #     #         self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_motor1pos))
    #     #         self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_motor2pos))
    #
    #             currentaxes.get_xaxis().set_major_formatter(FuncFormatter(self.fromindex_to_pixelpos_x))
    #             currentaxes.get_yaxis().set_major_formatter(FuncFormatter(self.fromindex_to_pixelpos_y))
    #
    #             # END of TRYING fix clever ticks values and locations ------------------
    #             import time
    #             currentaxes.set_xlabel(self.xylabels[0] + str(time.time()))
    #             currentaxes.set_ylabel(self.xylabels[1])
    #
    #             nb_of_microns_x = round((posmotor1[-1] - posmotor1[0]) * step_factor)
    #             nb_of_microns_y = round((posmotor2[-1] - posmotor2[0]) * step_factor)
    #
    # #             print "nb_points_y,nb_points_x", nby, nbx
    # #             print "nb_of_microns_x", nb_of_microns_x
    # #             print "nb_of_microns_y", nb_of_microns_y
    #
    #     #         self.axes.locator_params('x', tight=True, nbins=6)  # round(nb_of_microns_x) + 1)
    #     #         self.axes.locator_params('y', tight=True, nbins=round(nb_of_microns_y) + 1)
    #
    #             # fix the ticks distance: either 1 micron or a given length such as 5 ticks are plotted
    # #
    # #             tickx_micron_sampling = 1. / step_x  # 1 micron
    # #             tickx_sampling_length = max(math.fabs(nb_of_microns_x / 5. / step_x),
    # #                                         math.fabs(tickx_micron_sampling))
    # #             ticky_micron_sampling = 1. / step_y
    # #             ticky_sampling_length = max(math.fabs(nb_of_microns_y / 5. / step_y),
    # #                                         math.fabs(ticky_micron_sampling))
    # #
    # #             print "ticks every : tickx_sampling_length (micron)", tickx_sampling_length
    # #             print "ticks every : ticky_sampling_length (micron)", ticky_sampling_length
    # #
    # #             locx = pltticker.MultipleLocator(base=math.fabs(tickx_sampling_length))  # this locator puts ticks at regular intervals
    # #             self.axes.xaxis.set_major_locator(locx)
    # #             locy = pltticker.MultipleLocator(base=math.fabs(ticky_sampling_length))  # this locator puts ticks at regular intervals
    # #             self.axes.yaxis.set_major_locator(locy)
    #
    #         currentaxes.grid(True)
    #
    # #         numrows, numcols, color = self.data.shape
    #         numrows, numcols = self.data.shape[:2]
    #
    # #         print "self.Imageindices", self.Imageindices
    # #         print "self.Imageindices.shape", self.Imageindices.shape
    # #         print "self.data.shape", self.data.shape
    #
    # #         imageindices = self.Imageindices[0] + arange(0, numrows, numcols) * self.stepindex
    #
    #         tabindices = self.Imageindices
    # #         print "tabindices", tabindices
    #
    # #         print "motor positions"
    # #         print posmotor1
    # #         print len(posmotor1)
    # #         print posmotor2
    # #         print len(posmotor2)
    #
    #         def format_coord(x, y):
    #             col = int(x + 0.5)
    #             row = int(y + 0.5)
    #
    # #             print "x,y before in col and row", x, y
    #             if col >= 0 and col < numcols and row >= 0 and row < numrows:
    #                 z = self.data[row, col]
    # #                 print "z", z
    # #                 print "x,y,row,col", x, y, row, col
    #                 Imageindex = tabindices[row, col]
    #                 if posmotors is None:
    # #                     print "self.posarray_twomotors is None"
    #                     sentence0 = 'x=%1.4f, y=%1.4f, z_intensity=%s, ImageIndex: %d' % \
    #                                         (x, y, str(z), Imageindex)
    #                     sentence_corner = ''
    #                     sentence_center = ''
    #                     sentence = 'No motors positions'
    #                 else:
    # #                     print "col,row= ", col, row
    # #                     print "posmotor1[col],posmotor2[row]", posmotor1[col], posmotor2[row]
    #
    #                     sentence0 = "j=%d, i=%d, z_intensity = %s, ABSOLUTE=[%s=%.5f,%s=%.5f], ImageIndex: %d" % \
    #                             (col, row, str(z),
    #                              self.motor1name, posmotor1[col], self.motor2name, posmotor2[row], Imageindex)
    #
    #                     sentence = 'POSITION (micron) from: '
    #                     sentence_corner = "CORNER =[[%s=%.2f,%s=%.2f]]" % \
    #                             (self.motor1name, (posmotor1[col] - posmotor1[0]) * step_factor,
    #                              self.motor2name, (posmotor2[row] - posmotor2[0]) * step_factor)
    #                     sentence_center = "CENTER =[[%s=%.2f,%s=%.2f]]" % \
    #                             (self.motor1name, (posmotor1[col] - poscenter_motor1) * step_factor,
    #                              self.motor2name, (posmotor2[row] - poscenter_motor2) * step_factor)
    #
    #                 self.frameparent.stbar0.SetStatusText(sentence0)
    #                 self.frameparent.stbar.SetStatusText(sentence)
    #                 self.frameparent.stbar.SetStatusText(sentence_corner, 1)
    #                 self.frameparent.stbar.SetStatusText(sentence_center, 2)
    #
    #                 return sentence0
    #             else:
    #                 print 'out of plot'
    #                 return 'out of plot'
    #
    #         currentaxes.format_coord = self.format_coord
    #
    # #         if step_y != 0 and step_x != 0:
    # #             ratio = 1.*step_x / step_y
    # #             self.forceAspect(ratio)

    def set_motorspositions_parameters(self):
        self.posmotors = self.posarray_twomotors

        if self.posmotors is None:
            return "posmotors is None"

        print("in set_motorspositions_parameters")

        print("absolute_motorposition_unit", self.absolute_motorposition_unit)

        print("pos extremes")
        print(
            "first motor", self.posarray_twomotors[0, 0], self.posarray_twomotors[0, -1]
        )
        print(
            "second motor",
            self.posarray_twomotors[-1, 0],
            self.posarray_twomotors[-1, -1],
        )

        rangeX = (
            np.fix(
                (self.posarray_twomotors[0, -1] - self.posarray_twomotors[0, 0])[0]
                * 100000
            )
            / 100000
        )
        rangeY = (
            np.fix(
                (self.posarray_twomotors[-1, -1] - self.posarray_twomotors[0, -1])[1]
                * 100000
            )
            / 100000
        )

        print("first motor total range", rangeX)
        print("second motor total range", rangeY)

        self.numrows, self.numcols = self.data.shape[:2]

        self.tabindices = self.Imageindices

        initmotor1 = self.posmotors[0, 0, 0]
        initmotor2 = self.posmotors[0, 0, 1]

        self.posmotor1 = self.posmotors[0, :, 0]
        self.posmotor2 = self.posmotors[:, 0, 1]

        #         print "starting motor1 %f %s" % (initmotor1, self.absolute_motorposition_unit)
        #         print "starting motor2 %f %s" % (initmotor2, self.absolute_motorposition_unit)

        #             print 'posmotor1', posmotor1
        #             print 'posmotor2', posmotor2

        nby, nbx = self.posmotors.shape[:2]

        self.poscenter_motor1 = self.posmotor1[nbx / 2]
        self.poscenter_motor2 = self.posmotor2[nby / 2]

        #         print "center motor1", self.poscenter_motor1
        #         print "center motor2", self.poscenter_motor2

        # x= fast motor  (first in spec scan)
        # y slow motor (second in spec scan)
        #         self.step_x = (self.posmotor1[-1] - self.posmotor1[0]) / (nbx - 1)
        #         self.step_y = (self.posmotor2[-1] - self.posmotor2[0]) / (nby - 1)

        self.step_x = (self.maxmotor1 - self.minmotor1) / (nbx - 1)
        self.step_y = (self.maxmotor2 - self.minmotor2) / (nby - 1)

        print("set self.step_x to mm", self.step_x)
        print("set self.step_y to mm", self.step_y)
        #         print "step_x %f %s " % (self.step_x, self.absolute_motorposition_unit)
        #         print "step_y %f %s " % (self.step_y, self.absolute_motorposition_unit)

        self.step_factor = 1.0
        if self.absolute_motorposition_unit == "mm":
            self.step_factor = 1000.0
            self.step_x = self.step_x * self.step_factor
            self.step_y = self.step_y * self.step_factor

    #         print "step_x %f micron " % (self.step_x)
    #         print "step_y %f micron " % (self.step_y)

    #         nb_of_microns_x = round((self.posmotor1[-1] - self.posmotor1[0]) * self.step_factor)
    #         nb_of_microns_y = round((self.posmotor2[-1] - self.posmotor2[0]) * self.step_factor)

    #         print "nb_points_y,nb_points_x", nby, nbx
    #         print "nb_of_microns_x", nb_of_microns_x
    #         print "nb_of_microns_y", nb_of_microns_y

    def format_coord(self, x, y):

        col = int(x + 0.5)
        row = int(y + 0.5)

        numcols, numrows = self.numcols, self.numrows
        posmotors = self.posmotors
        posmotor1, posmotor2 = self.posmotor1, self.posmotor2
        tabindices = self.tabindices
        step_factor = self.step_factor
        poscenter_motor1, poscenter_motor2 = (
            self.poscenter_motor1,
            self.poscenter_motor2,
        )

        #         print "x,y before in col and row", x, y
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = self.data[row, col]
            #                 print "z", z
            #             print "\nx,y,row,col", x, y, row, col
            Imageindex = tabindices[row, col]
            if posmotors is None:
                #                     print "self.posarray_twomotors is None"
                sentence0 = "x=%1.4f, y=%1.4f, val=%s, ImageIndex: %d" % (
                    x,
                    y,
                    str(z),
                    Imageindex,
                )
                sentence_corner = ""
                sentence_center = ""
                sentence = "No motors positions"
            else:
                #                 print "col,row= ", col, row
                #                 print "posmotor1[col],posmotor2[row]", posmotor1[col], posmotor2[row]

                sentence0 = (
                    "j=%d, i=%d, ABSOLUTE=[%s=%.5f,%s=%.5f], z_intensity = %s, ImageIndex: %d"
                    % (
                        col,
                        row,
                        self.motor1name,
                        posmotor1[col],
                        self.motor2name,
                        posmotor2[row],
                        str(z),
                        Imageindex,
                    )
                )

                sentence = "POSITION (micron) from: "
                sentence_corner = "CORNER =[[%s=%.2f,%s=%.2f]]" % (
                    self.motor1name,
                    (posmotor1[col] - posmotor1[0]) * step_factor,
                    self.motor2name,
                    (posmotor2[row] - posmotor2[0]) * step_factor,
                )
                sentence_center = "CENTER =[[%s=%.2f,%s=%.2f]]" % (
                    self.motor1name,
                    (posmotor1[col] - poscenter_motor1) * step_factor,
                    self.motor2name,
                    (posmotor2[row] - poscenter_motor2) * step_factor,
                )

            self.frameparent.stbar0.SetStatusText(sentence0)
            self.frameparent.stbar.SetStatusText(sentence)
            self.frameparent.stbar.SetStatusText(sentence_corner, 1)
            self.frameparent.stbar.SetStatusText(sentence_center, 2)

            return sentence0
        else:
            print("out of plot")
            return "out of plot"


from matplotlib.axes import Axes


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


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
