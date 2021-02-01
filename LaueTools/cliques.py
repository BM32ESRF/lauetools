import sys
import os.path

import matplotlib

matplotlib.use("WXAgg")

import numpy as np
import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
if sys.version_info.major == 3:
    from . import graingraph as GraGra
    from . import findorient as FO
    from . import dict_LaueTools as DictLT
else:
    import graingraph as GraGra
    import findorient as FO
    import dict_LaueTools as DictLT

# --- ---------------------  CLIQUES board
class CliquesFindingBoard(wx.Frame):
    """
    Class to display GUI for cliques finding

    parent frame must have following attributes:
    DataPlot_filename
    dirname
    list_of_cliques
    """

    def __init__(self, parent, _id, title, _LaueToolsProjectFolder, indexation_parameters):

        wx.Frame.__init__(self, parent, _id, title, size=(400, 330))

        self.panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(590, 390), pos=(5, 5))

        self.indexation_parameters = indexation_parameters
        self.parent = parent
        self.LaueToolsProjectFolder = _LaueToolsProjectFolder
        self.LUTfilename = None

        print('Inside CliquesFindingBoard', self.indexation_parameters.keys())
        print('Inside CliquesFindingBoard', self.indexation_parameters["AllDataToIndex"].keys())
        DataToIndex = self.indexation_parameters["AllDataToIndex"]

        MaxNbSpots = len(DataToIndex["data_theta"])
        print('MaxNbSpots: ', MaxNbSpots)

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        wx.StaticText(self.panel, -1, "Current File: %s" % self.parent.DataPlot_filename, (130, 15))

        title1 = wx.StaticText(self.panel, -1, "Parameters", (15, 15))
        title1.SetFont(font3)

        wx.StaticText(self.panel, -1, "Angular tolerance(deg): ", (15, 45))
        self.AT = wx.TextCtrl(self.panel, -1, "0.2", (200, 40), (60, -1))
        wx.StaticText(self.panel, -1, "(for distance recognition): ", (15, 60))

        wx.StaticText(self.panel, -1, "Size of spots set: ", (15, 90))
        self.nbspotmax = wx.SpinCtrl(self.panel, -1, str(MaxNbSpots),
                                                    (200, 85), (60, -1), min=3, max=MaxNbSpots)

        wx.StaticText(self.panel, -1, "List of spots index", (15, 125))
        self.spotlist = wx.TextCtrl(self.panel, -1, "to5", (150, 125), (200, -1))
        wx.StaticText(self.panel, -1, "(from 0 to set size-1)", (15, 145))

        compsavebtn = wx.Button(self.panel, -1, "Compute && Save", (260, 170), (60, 30))
        compsavebtn.Bind(wx.EVT_BUTTON, self.OnComputeSaveAnglesFile)

        wx.StaticText(self.panel, -1, "Angle list (default 'CUBIC'):", (15, 175))
        loadanglesbtn = wx.Button(self.panel, -1, "Load" (180, 170), (60, 30))
        loadanglesbtn.Bind(wx.EVT_BUTTON, self.OnLoadAnglesFile)

        self.indexchkbox = wx.CheckBox(self.panel, -1, "Use cliques for indexation", (15, 215))
        self.indexchkbox.SetValue(False)

        wx.Button(self.panel, 1, "Search", (40, 245), (110, 40))
        self.Bind(wx.EVT_BUTTON, self.OnSearch, id=1)
        wx.Button(self.panel, 2, "Quit", (230, 245), (110, 40))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=2)

        self.Show(True)
        self.Centre()

    def OnComputeSaveAnglesFile(self, _):
        # open dialog  for structur
        key_material = 'Ti'
        nLUT = 5

        latticeparameters = DictLT.dict_Materials[key_material][1]

        sortedangles = FO.computesortedangles(latticeparameters, nLUT)

        import pickle

        self.LUTfilename = '%s_nlut%d.angles'%(key_material,nLUT)
        with open(self.LUTfilename, "wb") as f:
            pickle.dump(sortedangles, f)

        print('Sorted angles written in %s'%self.LUTfilename)


    def OnLoadAnglesFile(self, _):
        # load specific lut angles
        # lutfolder = '/home/micha/LaueToolsPy3/LaueTools/'
        # lutfilename = 'sortedanglesCubic_nLut_5_angles_18_60.angles'
        # LUTfilename = os.path.join(lutfolder,lutfilename)

        defaultFolderAnglesFile = self.LaueToolsProjectFolder

        wcd = "AnglesLUT file (*.angles)|*.angles|All files(*)|*"
        open_dlg = wx.FileDialog(self, message="Choose a file", defaultDir=defaultFolderAnglesFile,
                                                                defaultFile="",
                                                                wildcard=wcd,
                                                                style=wx.OPEN | wx.CHANGE_DIR)
        if open_dlg.ShowModal() == wx.ID_OK:
            self.LUTfilename = open_dlg.GetPath()
        open_dlg.Destroy()


    def OnSearch(self, _):

        spot_list = self.spotlist.GetValue()

        print("spot_list in OnSearch", spot_list)
        if spot_list[0] != "-":
            if spot_list[0] == "[":
                spot_index_central = str(spot_list)[1:-1].split(",")
                print(spot_index_central)
                arr_index = np.array(spot_index_central)

                print(np.array(arr_index, dtype=int))
                spot_index_central = list(np.array(arr_index, dtype=int))
            elif spot_list[:2] == "to":
                spot_index_central = list(range(int(spot_list[2]) + 1))
            else:
                spot_index_central = int(spot_list)
        else:
            spot_index_central = 0

        nbmax_probed = self.nbspotmax.GetValue()
        ang_tol = float(self.AT.GetValue())
        print("ang_tol", ang_tol)
        Nodes = spot_index_central
        print("Clique finding for Nodes :%s" % Nodes)

        fullpath = os.path.join(self.parent.dirname, self.parent.DataPlot_filename,)

        res_cliques = GraGra.give_bestclique(fullpath, nbmax_probed, ang_tol,
                                                    nodes=Nodes, col_Int=-1,
                                                    LUTfilename=self.LUTfilename,
                                                    verbose=1)

        if isinstance(Nodes, int):
            DisplayCliques = res_cliques.tolist()
            self.parent.list_of_cliques = [res_cliques,]
        else:
            self.parent.list_of_cliques = res_cliques
            DisplayCliques = ''
            for cliq in res_cliques:
                DisplayCliques += '%s\n'%str(cliq.tolist())

        print("BEST CLIQUES **************")
        print(DisplayCliques)
        print("***************************")

        wx.MessageBox('Following spot indices are likely to belong to the same grain:\n %s'%DisplayCliques,
                            'CLIQUES RESULTS')

        if not self.indexchkbox.GetValue():
            self.parent.list_of_cliques = None

    def OnQuit(self, _):
        self.Close()
