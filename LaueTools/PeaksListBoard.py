import os
import sys

import numpy as np

import wx
import wx.grid

if sys.version_info.major == 3:
    from . import IOLaueTools as IOLT
else:
    import IOLaueTools as IOLT

# --- ---------------  Plot limits board  parameters
# class PeaksListBoard(wx.Dialog):
class PeaksListBoard(wx.Frame):
    """
    Class to select scatter plot properties of peak list
    """
    def __init__(self, parent, _id):
        """
        initialize board window
        """
        # - Initialize the window:
        wx.Frame.__init__(self, None, wx.ID_ANY, "Peak list plot properties", size=(600, 400))

        # Add a panel so it looks correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)

        self.parent = parent
        #print("self.parent", self.parent)
        self.selectedMarker = None

        self.list_markerstyle = ["+", "x", "o", "h", "*", "p"]

        self.scatterplot_list = []

        self.nb_peakslists = 0
        self.myDataList = []

        self.selectedcolor = None
        self.selectedpeaklist = None
        self.fullpathfilename = None
        self.myscatter = None

        # widgets ------------------------------
        txtimage = wx.StaticText(self.panel, -1, "Image file path:")

        self.expimagetxtctrl = wx.TextCtrl(self.panel, -1, "", size=(400, -1))
        self.expimagebrowsebtn = wx.Button(self.panel, -1, "...", size=(50, -1))

        self.expimagebrowsebtn.Bind(wx.EVT_BUTTON, self.onSelectPeaksListFile)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(txtimage, 0, wx.EXPAND)
        hsizer.Add(self.expimagetxtctrl, 0, wx.EXPAND)
        hsizer.Add(self.expimagebrowsebtn, 0, wx.EXPAND)

        num_of_rows = 1
        num_of_columns = 5
        self.grid = wx.FlexGridSizer(num_of_rows, num_of_columns, 0, 0)

        self.grid.Add(wx.StaticText(self.panel, -1, "Select"))
        self.grid.Add(wx.StaticText(self.panel, -1, "Name"))
        self.grid.Add(wx.StaticText(self.panel, -1, "Color"))
        self.grid.Add(wx.StaticText(self.panel, -1, "marker"))
        self.grid.Add(wx.StaticText(self.panel, -1, "size"))

        #         toolbar = self.CreateToolBar()
        #         qtool = toolbar.AddLabelTool(wx.ID_ANY, 'Refresh',
        #                                      wx.Bitmap(os.path.join(os.path.dirname(__file__),
        #                                                             'refresh.png')))
        #         toolbar.AddSeparator()
        #         self.Bind(wx.EVT_TOOL, self.refreshButton, qtool)
        #         toolbar.Realize()

        self.vsizer = wx.BoxSizer(wx.VERTICAL)
        self.vsizer.Add(hsizer, 0, wx.ALL)
        self.vsizer.Add(self.grid, 0, wx.EXPAND)

        self.panel.SetSizer(self.vsizer)
        #         self.Fit()
        self.vsizer.Layout()

    def onChangeColor(self, event, btnlabel):
        """
        This is mostly from the wxPython Demo!
        """
        print("event", event)
        index = int(btnlabel.split("_")[-1])
        print("clicked btn index", index)
        dlg = wx.ColourDialog(self)

        # Ensure the full colour dialog is displayed,
        # not the abbreviated version.
        dlg.GetColourData().SetChooseFull(True)

        if dlg.ShowModal() == wx.ID_OK:
            data = dlg.GetColourData()

            self.selectedcolor = data.GetColour().Get()
            print("You selected: (%d, %d, %d)\n" % self.selectedcolor)
            print("matplolib color: (%.2f, %.2f, %.2f)\n" % self.convert_to_rgbmatplotlib(self.selectedcolor))

            self.myDataList[index]["Color"] = self.selectedcolor

            # update color btn
            clickedbtn = getattr(self, "colorbtn_%d" % index)
            clickedbtn.SetBackgroundColour(wx.Colour(*self.selectedcolor))

        dlg.Destroy()

        if self.scatterplot_list is not []:
            self.scatterplot_list[index].set_edgecolor(self.convert_to_rgbmatplotlib(self.selectedcolor))
            self.parent.update_draw(1)

    def convert_to_rgbmatplotlib(self, rgbcolor):
        """convert rgb to rgbmatplotlib
        """
        return (rgbcolor[0] / 255.0, rgbcolor[1] / 255.0, rgbcolor[2] / 255.0)

    def refreshButton(self, _):
        """Refresh
        """
        self.refreshMyData()
        self.refreshGrid()

    def refreshwholeGrid(self):
        print("\n\n\n------------refresh grid")
        # - Clear the grid:
        print("Size of myDataList: ", len(self.myDataList))

        num_of_rows = len(self.myDataList) + 1

        self.grid.SetRows(num_of_rows)

        self.grid.Add(wx.StaticText(self.panel, -1, "Select"))
        self.grid.Add(wx.StaticText(self.panel, -1, "Name"))
        self.grid.Add(wx.StaticText(self.panel, -1, "Color"))
        self.grid.Add(wx.StaticText(self.panel, -1, "marker"))
        self.grid.Add(wx.StaticText(self.panel, -1, "size"))

        # - Populate the grid with new data:
        for i in range(len(self.myDataList)):
            colorbtn = wx.Button(self.panel, -1, "color_%d" % i)
            colorbtn.SetBackgroundColour(wx.Colour(*self.myDataList[i]["Color"]))
            setattr(self, "colorbtn_%d" % i, colorbtn)
            colorbtn.Bind(wx.EVT_BUTTON,
                lambda evt, name=colorbtn.GetLabel(): self.onChangeColor(evt, name))

            markercombo = wx.ComboBox(self.panel,
                                        -1,
                                        str(self.myDataList[i]["marker"]),
                                        choices=self.list_markerstyle,
                                        size=(20, -1))
            #             setattr(self, 'markercombo_%d' % i, markercombo)
            markercombo.Bind(wx.EVT_COMBOBOX,
                                lambda evt, comboindex=i: self.onChangeMarker(evt, comboindex))

            self.grid.Add(wx.StaticText(self.panel, -1, str(self.myDataList[i]["Select"])))
            self.grid.Add(wx.StaticText(self.panel, -1, str(self.myDataList[i]["Name"])))
            self.grid.Add(colorbtn)
            self.grid.Add(markercombo)
            self.grid.Add(wx.StaticText(self.panel, -1, str(self.myDataList[i]["size"])))

        self.panel.SetSizer(self.vsizer)
        self.vsizer.Layout()

    def addRow_Grid(self, datalist_index):
        print("\n\n\n------------refresh grid")
        # - Clear the grid:
        print("Size of myDataList: ", len(self.myDataList))

        num_of_rows = self.nb_peakslists + 1

        self.grid.SetRows(num_of_rows)

        # - Populate the grid with new data:
        chckbox = wx.CheckBox(self.panel, -1)
        chckbox.SetValue(True)
        chckbox.Bind(wx.EVT_CHECKBOX,
            lambda evt, chckindex=datalist_index: self.onCheckBox(evt, chckindex))
        setattr(self, "chckbox_%d" % datalist_index, chckbox)

        txtctrlname = wx.TextCtrl(self.panel, -1, str(self.myDataList[datalist_index]["Name"]))
        txtctrlname.SetMinSize((300, -1))
        #         setattr(self, 'namectrl_%d' % datalist_index, txtctrlname)

        colorbtn = wx.Button(self.panel, -1, "color_%d" % datalist_index)
        colorbtn.SetBackgroundColour(wx.Colour(*self.myDataList[datalist_index]["Color"]))
        setattr(self, "colorbtn_%d" % datalist_index, colorbtn)
        colorbtn.Bind(wx.EVT_BUTTON,
            lambda evt, name=colorbtn.GetLabel(): self.onChangeColor(evt, name))

        markercombo = wx.ComboBox(self.panel,
                                    -1,
                                    str(self.myDataList[datalist_index]["marker"]),
                                    choices=self.list_markerstyle,
                                    size=(80, -1))
        #             setattr(self, 'markercombo_%d' % datalist_index, markercombo)
        markercombo.Bind(wx.EVT_COMBOBOX,
            lambda evt, comboindex=datalist_index: self.onChangeMarker(evt, comboindex))

        sizectrl = wx.SpinCtrl(self.panel, -1, str(self.myDataList[datalist_index]["size"]))
        setattr(self, "sizectrl_%d" % datalist_index, sizectrl)
        sizectrl.Bind(wx.EVT_SPINCTRL,
            lambda evt, index=datalist_index: self.onChangeSize(evt, index))

        self.grid.Add(chckbox, 1)
        self.grid.Add(txtctrlname, 1)
        self.grid.Add(colorbtn, 1)
        self.grid.Add(markercombo, 1)
        self.grid.Add(sizectrl, 1)

        self.grid.AddGrowableCol(2, 1)

        self.panel.SetSizer(self.vsizer)
        self.vsizer.Layout()

    def onCheckBox(self, _, chckindex):
        print("chckindex", chckindex)

        chckbox = getattr(self, "chckbox_%d" % chckindex)

        newstate = chckbox.GetValue()
        print("newstate", newstate)

        if not newstate:
            self.scatterplot_list[chckindex].remove()
            self.parent.update_draw(1)
        else:
            self.plotmarkers(addscatteratindex=chckindex,
                            markerstyle=self.myDataList[chckindex]["marker"],
                            edgecolor=self.convert_to_rgbmatplotlib(
                                self.myDataList[chckindex]["Color"]))

    def onChangeSize(self, _, index):
        """ on change size
        """
        sizectrl = getattr(self, "sizectrl_%d" % index)

        newsize = int(sizectrl.GetValue())
        corresponding_scatterplot = self.scatterplot_list[index]

        print(dir(corresponding_scatterplot))

        if self.scatterplot_list is not []:
            self.scatterplot_list[index].remove()
            self.plotmarkers(addscatteratindex=index,
                            markerstyle=self.myDataList[index]["marker"],
                            edgecolor=self.convert_to_rgbmatplotlib(
                                self.myDataList[index]["Color"]),
                            markersize=newsize)

            self.myDataList[index]["size"] = newsize

    def onChangeMarker(self, event, comboindex):
        """ on change marker type
        """
        print("combo index", comboindex)
        item = event.GetSelection()
        self.selectedMarker = self.list_markerstyle[item]

        self.myDataList[comboindex]["marker"] = self.selectedMarker

        print("select marker: %s\n" % self.selectedMarker)

        if self.scatterplot_list is not []:
            self.scatterplot_list[comboindex].remove()
            self.plotmarkers(addscatteratindex=comboindex,
                            markerstyle=self.selectedMarker,
                            edgecolor=self.convert_to_rgbmatplotlib(
                                self.myDataList[comboindex]["Color"]))

        event.Skip()

    def readnewpeaklistfile(self):

        fullpathfilename = str(self.expimagetxtctrl.GetValue())
        if not os.path.isfile(fullpathfilename):
            dlg = wx.MessageDialog(self,
                                "peak list file : %s\n\ndoes not exist!!" % fullpathfilename,
                                "error",
                                wx.OK | wx.ICON_ERROR)
            #                 dlg = wx.MessageDialog(self, 'Detector parameters must be float with dot separator',
            #                                    'Bad Input Parameters',)
            dlg.ShowModal()
            dlg.Destroy()
            return

        if fullpathfilename.endswith(".dat"):
            self.selectedpeaklist = IOLT.read_Peaklist(fullpathfilename)[:, :2]
        elif fullpathfilename.endswith(".cor"):
            pl = IOLT.readfile_cor(fullpathfilename)[3:5]
            self.selectedpeaklist = np.array(pl).T
        elif fullpathfilename.endswith(".fit"):
            res = IOLT.readfitfile_multigrains(fullpathfilename, return_columnheaders=True)
            try:
                data, colname_dict = res
            except:
                print("problem when reading .fit")
                print(res)

            col_X = None
            if "Xtheo" in colname_dict:
                col_X = colname_dict["Xtheo"]
                print("column for theo. peaks X position found at index %d" % col_X)
            elif "Xexp" in colname_dict:
                col_X = colname_dict["Xexp"]
                print("column for exp. peaks X position found at index %d" % col_X)
            else:
                print("\n\n!!column for theo. or exp. peaks X position not found...!!\n")

            col_Y = None
            if "Ytheo" in colname_dict:
                col_Y = colname_dict["Ytheo"]
                print("column for theo. peaks Y position found at index %d" % col_Y)
            elif "Yexp" in colname_dict:
                col_Y = colname_dict["Yexp"]
                print("column for exp. peaks Y position found at index %d" % col_Y)
            else:
                print("\n\n!!column for theo. or exp. peaks Y position not found...!!\n")

            if len(data) == 2:
                allspotsdata = data[0][4]
            else:
                allspotsdata = data[4]

            if col_X is None or col_Y is None:
                print("\n\n!!Can't read the file ...!!\n")
                return

            self.selectedpeaklist = np.take(allspotsdata, (col_X, col_Y), axis=1)

        self.fullpathfilename = fullpathfilename
        self.nb_peakslists += 1

        print("self.selectedpeaklist", self.selectedpeaklist)

    def update_ImageArray(self):
        self.parent.ImageArray = self.ImageArray

        self.updateplot()

    def onSelectPeaksListFile(self, evt):
        self.GetfullpathFile(evt)
        self.expimagetxtctrl.SetValue(self.fullpathfilename)

        self.readnewpeaklistfile()

        self.myDataList.append({"Select": True,
                                "Name": os.path.split(self.fullpathfilename)[-1],
                                "Color": (0, 255, 0),
                                "marker": "*",
                                "size": 20,
                                "peakslist": self.selectedpeaklist})
        #         self.update_ImageArray()
        self.addRow_Grid(self.nb_peakslists - 1)
        self.plotmarkers()

    def GetfullpathFile(self, _):
        wcd = "Peaks list (*.dat)|*.dat|Peaks list (*.cor)|*.cor|indexed Peaks list (*.fit)|*.fit|All files(*)|*"
        myFileDialog = wx.FileDialog(self, "Choose an image file", style=wx.OPEN, wildcard=wcd)
        dlg = myFileDialog
        dlg.SetMessage("Choose an image file")
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()

            self.fullpathfilename = str(filename)
        else:
            pass

    def plotmarkers(self, addscatteratindex=None, markerstyle="o", edgecolor="g", markersize=20):
        """add plot markers at peaks pixel position from self.selectedpeaklist or self.myDataList
        """
        # offset convention
        X_OFFSET = 1
        Y_OFFSET = 1

        kwords = {"marker": markerstyle, "facecolor": "None",
                    "edgecolor": edgecolor, "s": markersize}

        if addscatteratindex is not None:
            X, Y = self.myDataList[addscatteratindex]["peakslist"].T
        else:
            X, Y = self.selectedpeaklist.T

        self.myscatter = self.parent.axes.scatter(X - X_OFFSET, Y - Y_OFFSET, alpha=1.0, **kwords)

        if addscatteratindex is not None:
            self.scatterplot_list[addscatteratindex] = self.myscatter
        else:
            self.scatterplot_list.append(self.myscatter)

        #         print dir(self.myscatter)
        self.parent.update_draw(1)


if __name__ == "__main__":

    class App(wx.App):
        def OnInit(self):
            """Create the main window and insert the custom frame"""
            dlg = PeaksListBoard(None, -1)

            dlg.Show(True)
            return True

    app = App(0)
    app.MainLoop()
