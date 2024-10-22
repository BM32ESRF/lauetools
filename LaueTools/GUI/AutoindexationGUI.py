r"""
GUI module to refine orientation and strain from Laue spots lists

Main author is J. S. Micha:   micha [at] esrf [dot] fr

version Aug 2019
from LaueTools package hosted in

https://gitlab.esrf.fr/micha/lauetools
"""

import sys
import time
import os
import wx
import numpy as np

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        """alias for wxpython4
        """
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip


if sys.version_info.major == 3:
    from .. import lauecore as LT
    from .. import CrystalParameters as CP
    from .. import indexingSpotsSet as ISS
    from .. import indexingAnglesLUT as INDEX
    from .. import IOLaueTools as IOLT
    from .. import generaltools as GT
    from .. import dict_LaueTools as DictLT
    from . PlotRefineGUI import Plot_RefineFrame
    from . ResultsIndexationGUI import RecognitionResultCheckBox
    from . import OpenSpotsListFileGUI as OSLFGUI

else:
    import lauecore as LT
    import CrystalParameters as CP
    import indexingSpotsSet as ISS
    import indexingAnglesLUT as INDEX
    import IOLaueTools as IOLT
    import generaltools as GT
    import dict_LaueTools as DictLT
    from GUI.PlotRefineGUI import Plot_RefineFrame
    from GUI.ResultsIndexationGUI import RecognitionResultCheckBox
    import OpenSpotsListFileGUI as OSLFGUI


# --- ----------------   Classical Indexation Board
class DistanceScreeningIndexationBoard(wx.Frame):
    """
    Class of GUI for the automatic indexation board of
    a single peak list with a single material or structure

    called also by autoindexation
    """
    def __init__(self, parent, _id, indexation_parameters, title,
                                        StorageDict=None, DataSetObject=None):

        wx.Frame.__init__(self, parent, _id, title, size=(900, 800))

        if parent is not None:
            self.parent = parent
            self.mainframe = self.parent
        else:
            self.mainframe = self

        self.kf_direction = indexation_parameters["kf_direction"]
        self.DataPlot_filename = indexation_parameters["DataPlot_filename"]
        self.key_material = None
        self.dict_Materials = indexation_parameters["dict_Materials"]
        self.dict_Rot = indexation_parameters["dict_Rot"]
        self.list_of_cliques = indexation_parameters["Cliques"]

        self.current_exp_spot_index_list = indexation_parameters["DataToIndex"]["current_exp_spot_index_list"]
        #         print "self.current_exp_spot_index_list", self.current_exp_spot_index_list
        self.data_theta = indexation_parameters["DataToIndex"]["data_theta"]
        self.data_chi = indexation_parameters["DataToIndex"]["data_chi"]
        self.data_I = indexation_parameters["DataToIndex"]["data_I"]
        self.dataXY_exp = indexation_parameters["DataToIndex"]["dataXY"]

        self.ClassicalIndexation_Tabledist = indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"]

        self.defaultParam = indexation_parameters["detectorparameters"]
        self.detectordiameter = indexation_parameters["detectordiameter"]
        self.pixelsize = indexation_parameters["pixelsize"]
        self.framedim = indexation_parameters["dim"]

        self.CCDLabel = indexation_parameters["CCDLabel"]

        self.datatype = "2thetachi"

        self.CCDdetectorparameters = {}
        self.CCDdetectorparameters["CCDcalib"] = indexation_parameters["detectorparameters"]
        self.CCDdetectorparameters["framedim"] = indexation_parameters["dim"]
        self.CCDdetectorparameters["pixelsize"] = indexation_parameters["pixelsize"]
        self.CCDdetectorparameters["CCDLabel"] = indexation_parameters["CCDLabel"]
        self.CCDdetectorparameters["detectorparameters"] = indexation_parameters["detectorparameters"]
        self.CCDdetectorparameters["detectordiameter"] = indexation_parameters["detectordiameter"]

        self.IndexationParameters = indexation_parameters
        
        self.IndexationParameters["Filename"] = indexation_parameters["DataPlot_filename"]
        self.dirname = indexation_parameters["dirname"]
        self.IndexationParameters["DataPlot_filename"] = indexation_parameters["DataPlot_filename"]
        self.IndexationParameters["current_processedgrain"] = indexation_parameters["current_processedgrain"]
        self.IndexationParameters["mainAppframe"] = indexation_parameters["mainAppframe"]
        self.IndexationParameters["indexationframe"] = self

        # print("keys of self.IndexationParameters in DistanceScreeningIndexationBoard",
        #                                     list(self.IndexationParameters.keys()))

        if self.CCDLabel in ("MARCCD165", "ROPER159", "VHR_PSI"):
            self.IndexationParameters["flipyaxis"] = True
        else:
            self.IndexationParameters["flipyaxis"] = False

        if StorageDict is None:
            self.StorageDict = {}
            self.StorageDict["mat_store_ind"] = self.mainframe.mat_store_ind
            self.StorageDict["Matrix_Store"] = self.mainframe.Matrix_Store
            self.StorageDict["dict_Rot"] = self.mainframe.dict_Rot
        else:
            self.StorageDict = StorageDict

        self.DataSet = DataSetObject

        self.list_materials = []
        self.combokeymaterial = None
        self.spotsorder = None
        self.dict_param, self.dict_param_list = None, None
        self.select_chi, self.select_theta = None, None
        self.select_dataX, self.select_dataY, self.select_I = None, None, None
        self.data = None
        self.select_dataXY = None
        self.bestmatrices = []
        self.TwicethetaChi_solution = None
        self.config_irp_filename = None

        self.initGUI()

    def initGUI(self):
        """
        GUI widgets
        """
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        title1 = wx.StaticText(self, -1, "Spots Selection")
        title1.SetFont(font3)
        txtcf = wx.StaticText(self, -1, "Current File:        %s   " % self.DataPlot_filename)
        txtcfolder = wx.StaticText(self, -1, "Folder:        %s   " % self.dirname)
        nbspots_in_data = len(self.current_exp_spot_index_list)
        mssstxt = wx.StaticText(self, -1, "Spots set Size         ")
        self.nbspotmaxformatching = wx.SpinCtrl(self, -1, str(nbspots_in_data),
                                                        (60, -1), min=3, max=nbspots_in_data)

        self.setAchck = wx.CheckBox(self, -1, "")
        self.setAchck.SetValue(True)
        cstxt = wx.StaticText(self, -1, "Spots set A")
        self.spotlistA = wx.TextCtrl(self, -1, "0", size=(200, -1))
        self.sethklchck = wx.CheckBox(self, -1, "set hkl of set A spots")
        self.sethklcentral = wx.TextCtrl(self, -1, "[1,0,0]")

        self.setBchck = wx.CheckBox(self, -1, "")
        self.setBchck.SetValue(True)
        rsstxt = wx.StaticText(self, -1, "Spots Set B: ")
        self.spotlistB = wx.TextCtrl(self, -1, "to10", (200, -1))

        lutrectxt = wx.StaticText(self, -1, "Angles LUT Recognition")
        lutrectxt.SetFont(font3)

        drtatxt = wx.StaticText(self, -1, "Recognition Tol. Angle(deg)")
        self.DRTA = wx.TextCtrl(self, -1, "0.5")
        luttxt = wx.StaticText(self, -1, "LUT Nmax")
        self.nLUT = wx.SpinCtrl(self, -1, "4", (50, -1), min=3, max=7)

        elemtxt = wx.StaticText(self, -1, "Materials")
        self.SetMaterialsCombo(0)
        self.refresh = wx.Button(self, -1, "Refresh")
        self.refresh.Bind(wx.EVT_BUTTON, self.SetMaterialsCombo)
        #self.applyrulesLUT = wx.CheckBox(self, -1, "Apply Extinc. Rules")
        self.applyrulesLUT = wx.CheckBox(self, -1, "Apply Extinc. Rules")
        self.applyrulesLUT.SetValue(True)

        matchtxt = wx.StaticText(self, -1, "Matching")
        matchtxt.SetFont(font3)

        mtatxt = wx.StaticText(self, -1, "Matching Tol. Angle(deg)")
        self.MTA = wx.TextCtrl(self, -1, "0.2")
        resangtxt = wx.StaticText(self, -1, "Min. d-spacing")
        self.ResolutionAngstromctrl = wx.TextCtrl(self, -1, "False", (70, -1))

        emaxtxt = wx.StaticText(self, -1, "Energy max.: ")
        self.emax = wx.SpinCtrl(self, -1, "22", (60, -1), min=10, max=150)
        mnmstxt = wx.StaticText(self, -1, "Matching Threshold")
        self.MNMS = wx.SpinCtrl(self, -1, "15", (60, -1), min=1, max=500)

        pptxt = wx.StaticText(self, -1, "Filtering && Post Processing")
        pptxt.SetFont(font3)

        self.filterMatrix = wx.CheckBox(self, -1,
                                        "Remove equivalent Matrices (cubic symmetry)")
        self.filterMatrix.SetValue(True)
        self.verbose = wx.CheckBox(self, -1, "Print details")
        self.verbose.SetValue(False)

        self.showplotBox = wx.CheckBox(self, -1, "Plot Best result")
        self.showplotBox.SetValue(False)
        self.indexation_index = 0
        self.config_irp_filename = (self.DataPlot_filename[: -4] + "_%d.irp" % self.indexation_index)
        spcftxt= wx.StaticText(self, -1, "Saving parameters in config file")
        self.output_irp = wx.TextCtrl(self, -1, "%s" % self.config_irp_filename,
                                     size=(250, -1))

        self.StartButton = wx.Button(self, -1, "Start", size=(-1, 80))
        self.StartButton.SetFont(font3)
        quitbtn = wx.Button(self, 2, "Quit", size=(-1, 80))
        self.textprocess = wx.StaticText(self, -1, "                     ")
        self.gauge = wx.Gauge(self, -1, 1000, size=(250, -1))

        self.StartButton.Bind(wx.EVT_BUTTON, self.OnStart)
        quitbtn.Bind(wx.EVT_BUTTON, self.OnQuit)

        self.sb = self.CreateStatusBar()

        #-----  use input of cliques
        if self.list_of_cliques is not None:
            cliqueindex = 0
            self.spotlistA.SetValue(str(self.list_of_cliques[cliqueindex].tolist()))
            self.setBchck.SetValue(False)

        # layout
        h1box = wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add(mssstxt, 0, wx.EXPAND, 10)
        h1box.Add(self.nbspotmaxformatching, 0, wx.EXPAND, 10)

        h2box = wx.BoxSizer(wx.HORIZONTAL)
        h2box.Add(self.setAchck, 0, wx.EXPAND|wx.ALL, 10)
        h2box.Add(cstxt, 0, wx.EXPAND|wx.ALL, 10)
        h2box.Add(self.spotlistA, 0, wx.EXPAND|wx.ALL, 10)
        h2box.Add(self.sethklchck, 0, wx.EXPAND|wx.ALL, 10)
        h2box.Add(self.sethklcentral, 0, wx.EXPAND|wx.ALL, 10)

        h3box = wx.BoxSizer(wx.HORIZONTAL)
        h3box.Add(self.setBchck, 0, wx.EXPAND|wx.ALL, 10)
        h3box.Add(rsstxt, 0, wx.EXPAND|wx.ALL, 10)
        h3box.Add(self.spotlistB, 0, wx.EXPAND|wx.ALL, 10)

        h4box = wx.BoxSizer(wx.HORIZONTAL)
        h4box.Add(drtatxt, 0, wx.EXPAND|wx.ALL, 10)
        h4box.Add(self.DRTA, 0, wx.EXPAND|wx.ALL, 10)
        h4box.Add(luttxt, 0, wx.EXPAND|wx.ALL, 10)
        h4box.Add(self.nLUT, 0, wx.EXPAND|wx.ALL, 10)

        h5box = wx.BoxSizer(wx.HORIZONTAL)
        h5box.Add(elemtxt, 0, wx.EXPAND|wx.ALL, 10)
        h5box.Add(self.combokeymaterial, 0, wx.EXPAND|wx.ALL, 10)
        h5box.Add(self.refresh, 0, wx.EXPAND|wx.ALL, 10)
        h5box.Add(self.applyrulesLUT, 0, wx.EXPAND|wx.ALL, 10)

        h6box = wx.BoxSizer(wx.HORIZONTAL)
        h6box.Add(mtatxt, 0, wx.EXPAND|wx.ALL, 10)
        h6box.Add(self.MTA, 0, wx.EXPAND|wx.ALL, 10)
        h6box.Add(emaxtxt, 0, wx.EXPAND|wx.ALL, 10)
        h6box.Add(self.emax, 0, wx.EXPAND|wx.ALL, 10)

        h7box = wx.BoxSizer(wx.HORIZONTAL)
        h7box.Add(resangtxt, 0, wx.EXPAND|wx.ALL, 10)
        h7box.Add(self.ResolutionAngstromctrl, 0, wx.EXPAND|wx.ALL, 10)
        h7box.Add(mnmstxt, 0, wx.EXPAND|wx.ALL, 10)
        h7box.Add(self.MNMS, 0, wx.EXPAND|wx.ALL, 10)

        h8box = wx.BoxSizer(wx.HORIZONTAL)
        h8box.Add(self.filterMatrix, 0, wx.EXPAND, 10)
        h8box.Add(self.verbose, 0, wx.EXPAND, 10)

        h9box = wx.BoxSizer(wx.HORIZONTAL)
        h9box.Add(spcftxt, 0, wx.EXPAND, 10)
        h9box.Add(self.output_irp, 0, wx.EXPAND, 10)

        h10box = wx.BoxSizer(wx.HORIZONTAL)
        h10box.Add(self.StartButton, 1, wx.EXPAND, 10)
        h10box.Add(quitbtn, 0, wx.EXPAND, 10)
        h10box.Add(self.textprocess, 0, wx.EXPAND, 10)
        h10box.Add(self.gauge, 0, wx.EXPAND, 10)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txtcf, 0, wx.EXPAND, 5)
        vbox.Add(txtcfolder, 0, wx.EXPAND, 5)
        vbox.Add(h1box, 0, wx.EXPAND, 5)
        vbox.AddSpacer(5)
        vbox.Add(title1, 0, wx.EXPAND, 5)
        vbox.Add(h2box, 0, wx.EXPAND, 5)
        vbox.Add(h3box, 0, wx.EXPAND, 5)
        vbox.AddSpacer(5)
        vbox.Add(lutrectxt, 0, wx.EXPAND, 5)
        vbox.Add(h4box, 0, wx.EXPAND, 5)
        vbox.Add(h5box, 0, wx.EXPAND, 5)
        vbox.AddSpacer(5)
        vbox.Add(matchtxt, 0, wx.EXPAND, 5)
        vbox.Add(h6box, 0, wx.EXPAND, 5)
        vbox.Add(h7box, 0, wx.EXPAND, 5)
        vbox.AddSpacer(5)
        vbox.Add(pptxt, 0, wx.EXPAND, 5)
        vbox.Add(h8box, 0, wx.EXPAND, 5)
        vbox.Add(self.showplotBox, 0, wx.EXPAND, 5)
        vbox.Add(h9box, 0, wx.EXPAND, 5)
        vbox.Add(h10box, 0, wx.EXPAND, 5)

        self.SetSizer(vbox)

        self.Show(True)
        self.Centre()

        # tooltips
        emaxtp = "Maximum energy of the white beam bandpass in keV. The higher this number the larger the number of simulated spots"
        emaxtxt.SetToolTipString(emaxtp)
        self.emax.SetToolTipString(emaxtp)

        resangtp = "Raw crystal lattice spacing resolution in Angstrom. HKL spots with lattice spacing smaller than this value are not simulated.\n"
        resangtp += "When crystal disorder occurs spots with low lattice spacings (or large hkl's) are weaks"
        resangtxt.SetToolTipString(resangtp)
        self.ResolutionAngstromctrl.SetToolTipString(resangtp)

        elemtip = "Select reference structure from the Material or Element list."
        self.combokeymaterial.SetToolTipString(elemtip)
        elemtxt.SetToolTipString(elemtip)

        self.refresh.SetToolTipString(
            "Refresh the Material list to admit new materials built in other LaueToolsGUI menus.")

        luttip = "Choose the largest hkl index to build the reference angular distance Looking up Table (LUT). Given n, the LUT contains all mutual angles between normals of lattice planes from (0,0,1) to (n,n,n-1) types"
        luttxt.SetToolTipString(luttip)
        self.nLUT.SetToolTipString(luttip)

        tsstip = "If checked, set of spot index to compute all mutual angles between spots of setA and setB."
        rsstxt.SetToolTipString(tsstip)
        self.spotlistB.SetToolTipString(tsstip)

        mssstip = "Number of experimental spots used find the best matching with the Laue Pattern simulated according to a recognised angle in LUT."
        mssstxt.SetToolTipString(mssstip)
        self.nbspotmaxformatching.SetToolTipString(mssstip)

        cstip = 'Experimental spot index (integer), list of spot indices to be considered as central spots, OR for example "to12" meaning spot indices ranging from 0 to 12 (included). All mutual angles between spots of setA will be considered for recognition. If setB is checked, all mutual angles between spots of setA and spots of setB will be calculated and compared to angles in reference LUT for recognition.\n'
        # cstip += 'List of spots must written in with bracket (e.g. [0,1,2,5,8]). Central spots index must be strictly lower to nb of spots of the "recognition set".'
        cstxt.SetToolTipString(cstip)
        self.spotlistA.SetToolTipString(cstip)

        drtatip = "Tolerance angle (in degree) within which an experimental angle (between a central and a recognition set spot) must be close to a reference angle in LUT to be used for simulation the Laue pattern (to be matched to the experimenal one)."
        drtatxt.SetToolTipString(drtatip)
        self.DRTA.SetToolTipString(drtatip)

        mtatip = "Tolerance angle (in degree) within which an experimental and simulated spots are close enough to be considered as matched."
        mtatxt.SetToolTipString(mtatip)
        self.MTA.SetToolTipString(mtatip)

        mnmstip = "Minimum number of matched spots (experimental with simulated one) to display the matching results of a possible indexation solution."
        mnmstxt.SetToolTipString(mnmstip)
        self.MNMS.SetToolTipString(mnmstip)

        self.showplotBox.SetToolTipString('Plot all exp. and theo. Laue Patterns for which the '
                        'number of matched spots is larger than "Minimum Number Matched spots".')

        self.filterMatrix.SetToolTipString("Keep only one orientation matrix for matrices which "
                                "are equivalent (cubic symmetry unit cell vectors permutations).")

        sethkltip = "Set the [h,k,l] Miller indices of central spot. This will reduce the running time of recognition."
        self.sethklchck.SetToolTipString(sethkltip)
        self.sethklcentral.SetToolTipString(sethkltip)

        self.verbose.SetToolTipString("Display details for long indexation procedure")

        self.applyrulesLUT.SetToolTipString("Apply systematic lattice extinction Rules "
            "when calculating angles LUT from reciprocal directions. "
            "To index single grain high hkl spots, better uncheck this (e.g. back reflection or "
            "transmission geometry.")

    def SetMaterialsCombo(self, _):
        """ set material combo  from   self.dict_Materials
        .. todo:: better to use gridsizer and refresh/update of combo
        """
        self.list_materials = sorted(self.dict_Materials.keys())

        self.combokeymaterial = wx.ComboBox(self, -1, "Ge", (140, 170), size=(150, -1),
                                        choices=self.list_materials, style=wx.CB_READONLY)

        self.combokeymaterial.Bind(wx.EVT_COMBOBOX, self.EnterCombokeymaterial)

    def EnterCombokeymaterial(self, event):
        """
        in classicalindexation
        """
        item = event.GetSelection()
        self.key_material = self.list_materials[item]
        self.filterMatrix.SetValue(CP.hasCubicSymmetry(self.key_material, dictmaterials=self.dict_Materials))

        self.sb.SetStatusText("Selected material: %s" % str(self.dict_Materials[self.key_material]))
        event.Skip()

    def getparams_for_irpfile(self):
        """get indexation and refine parameters to be written in a .irp file

        :return: boolean for success
        """

        # first element is omitted
        List_options = ISS.LIST_OPTIONS_INDEXREFINE[1:]
        
        sethklcentral = "None"
        self.spotsorder = "None"
        if self.sethklchck.GetValue():
            sethklcentral = str(self.sethklcentral.GetValue())

        MatchingAngleTol = float(self.MTA.GetValue())

        nbSpotsToIndex = 1000

        List_Ctrls = [self.combokeymaterial, 1, 5.0, self.emax, 100.0,
                        self.DRTA, MatchingAngleTol, self.spotlistB, 6,
                        self.spotlistA, self.ResolutionAngstromctrl, self.nLUT,
                        sethklcentral, True, nbSpotsToIndex,
                        [MatchingAngleTol, MatchingAngleTol / 2.0],
                        self.spotsorder]

        self.dict_param = {}
        flag = True
        # print("len(List_options)", len(List_options))
        # print("len(List_Ctrls)", len(List_Ctrls))

        for kk, option_key in enumerate(List_options):
            if not isinstance(List_Ctrls[kk], (int, str, list, float, bool)):
                val = str(List_Ctrls[kk].GetValue())
            else:
                val = List_Ctrls[kk]

            self.dict_param[option_key] = val

        self.dict_param_list = [self.dict_param]

        # print("self.dict_param_list", self.dict_param_list)

        return flag

    def Save_irp_configfile(self, outputfile="mytest_irp.irp"):
        """
        save indexation parameters in .irp file
        """
        ISS.saveIndexRefineConfigFile(self.dict_param_list, outputfilename=outputfile)

    def readspotssetctrl(self, txtctrl):
        """read, parse a spotset txtctrl
        """
        spot_list = txtctrl.GetValue()
        israngefromzero = False
        if spot_list[0] != "-":
            # print "coucou"
            # this a list of spots
            if spot_list.startswith("["):
                # print "coucou2"
                spot_index_central = str(spot_list)[1:-1].split(",")
                # print spot_index_central
                arr_index = np.array(spot_index_central)

                # print np.array(arr_index, dtype = int)
                spot_index_central = list(np.array(arr_index, dtype=int))
                nb_central_spots = len(spot_index_central)

            # this is range from 0 to a spot index
            elif spot_list.startswith("to"):
                spot_index_central = list(range(int(spot_list[2:]) + 1))
                nb_central_spots = len(spot_index_central)
                israngefromzero = True

            #this is a single spot index (integer)
            else:
                spot_index_central = int(spot_list)
                nb_central_spots = 1

        else:  # minus in front of integer
            spot_index_central = 0
            nb_central_spots = 1

        return spot_index_central, nb_central_spots, israngefromzero

    def parse_spotssetctrls(self):
        """
        parse txtctrls of spotsset A and B
        """
        #----------   Spots set Selection for mutual angle computation
        spotsB = None

        spotsA, nbA, _ = self.readspotssetctrl(self.spotlistA)

        if nbA == 1:
            maxindA = spotsA
        else:
            maxindA = max(spotsA)

        #spotB is not checked
        if not self.setBchck.GetValue():
            if nbA == 1:
                wx.MessageBox("if only spots set A is checked, you must provide a set of spots "
                "by filling 'to5' or '[5,1,4,3]'", "Error")

            nbmax_probed = maxindA+1
            spot_index_central = spotsA
            nb_central_spots = nbA
            # this is a range set
            if (nbA-1) == maxindA:
                spotssettype = 'rangeset'
            # this is a list of spots
            else:
                spotssettype = 'listsetA'

            spotsB = np.arange(0, nbmax_probed)

        #spotB is checked
        else:
            spotsB, nbB, israngeB = self.readspotssetctrl(self.spotlistB)
            nbmax_probed = nbB
            spot_index_central = spotsA
            nb_central_spots = nbA
            # case of [5,3,6,17] with B = to18
            if maxindA < nbmax_probed and israngeB:
                spotssettype = 'rangeset'
            # case of [5,3,6,17] with B = to5
            else:
                spotssettype = 'listsetAB'

        return spotssettype, spot_index_central, nb_central_spots, nbmax_probed, spotsB


    def OnStart(self, _):
        """
        starts automatic (classical) indexation:

        Recognition is based on the angular distance between two spots from a set of distances
        """
        t0 = time.time()

        energy_max = int(self.emax.GetValue())

        ResolutionAngstrom = self.ResolutionAngstromctrl.GetValue()
        if ResolutionAngstrom == "False":
            ResolutionAngstrom = False
        else:
            ResolutionAngstrom = float(ResolutionAngstrom)
        print("ResolutionAngstrom in OnStart Classical indexation", ResolutionAngstrom)

        self.key_material = str(self.combokeymaterial.GetValue())
        latticeparams = self.dict_Materials[self.key_material][1]
        B = CP.calc_B_RR(latticeparams)

        # read maximum index of hkl for building angles Look Up Table(LUT)
        nLUT = self.nLUT.GetValue()
        try:
            n = int(nLUT)
            if n > 7:
                wx.MessageBox("! LUT Nmax is too high!\n This value is set to 7 ", "INFO")
            elif n < 1:
                wx.MessageBox("! LUT Nmax is not positive!\n This value is set to 1 ", "INFO")
            n = min(7, n)
            n = max(1, n)
        except ValueError:
            print("!!  maximum index for building LUT is not an integer   !!!")
            wx.MessageBox("! LUT Nmax is not an integer!\n This value is set to 3 ", "INFO")
            n = 3

        rough_tolangle = float(self.DRTA.GetValue())
        fine_tolangle = float(self.MTA.GetValue())
        Minimum_MatchesNb = int(self.MNMS.GetValue())
        #         print "Recognition tolerance angle ", rough_tolangle
        #         print "Matching tolerance angle ", fine_tolangle

        #----------   Spots set Selection for mutual angle computation
        (spotssettype, spot_index_central, nb_central_spots,
                                nbmax_probed, spotsB) = self.parse_spotssetctrls()
        print("--spotssettype --#\n\n    ", self.parse_spotssetctrls(), "      \n\n*")

        # TODO spot_index_central and spotsB to be combined to find UBS
        #------------------------------------------------

        # whole exp.data spots dict
        #         self.IndexationParameters['AllDataToIndex']
        self.data_theta = self.IndexationParameters["AllDataToIndex"]["data_theta"]
        self.data_chi = self.IndexationParameters["AllDataToIndex"]["data_chi"]
        self.data_I = self.IndexationParameters["AllDataToIndex"]["data_I"]
        self.dataXY_exp = (self.IndexationParameters["AllDataToIndex"]["data_pixX"],
                            self.IndexationParameters["AllDataToIndex"]["data_pixY"])

        # there is no precomputed angular distances between spots
        if not self.ClassicalIndexation_Tabledist:
            # Selection of spots among the whole data
            # MSSS number
            MatchingSpotSetSize = int(self.nbspotmaxformatching.GetValue())

            # select 1rstly spots that have not been indexed and 2ndly reduced list by user
            index_to_select = np.take(self.current_exp_spot_index_list,
                                        np.arange(MatchingSpotSetSize))

            self.select_theta = self.data_theta[index_to_select]
            self.select_chi = self.data_chi[index_to_select]
            self.select_I = self.data_I[index_to_select]

            # print("index_to_select", index_to_select)
            # print("len self.dataXY_exp[0]", len(self.dataXY_exp[0]))

            self.select_dataX = self.dataXY_exp[0][index_to_select]
            self.select_dataY = self.dataXY_exp[1][index_to_select]
            # print select_theta
            # print select_chi
            if spotssettype in ("rangeset", ):
                listcouple = np.array([self.select_theta, self.select_chi]).T
                # compute angles between spots
                Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

            #             # with CYTHON
            #             import angulardist
            #             listcouple = np.array([2.*self.select_theta, self.select_chi]).T
            #             lc = listcouple.copy(order='c')
            #             Tabledistance = angulardist.calculdist_from_2thetachi(lc, lc)
        else:
            print("Preset Tabledistance is Not implemented !")
            return

        self.data = (2 * self.select_theta, self.select_chi, self.select_I, self.DataPlot_filename)

        self.select_dataXY = (self.select_dataX, self.select_dataY)

        # detector geometry
        detectorparameters = {}
        detectorparameters["kf_direction"] = self.kf_direction
        detectorparameters["detectorparameters"] = self.defaultParam
        detectorparameters["detectordiameter"] = self.detectordiameter
        detectorparameters["pixelsize"] = self.pixelsize
        detectorparameters["dim"] = self.framedim

        restrictLUT_cubicSymmetry = True
        set_central_spots_hkl = None

        if self.kf_direction in ('Z>0',):
            LUTfraction = 1/2.
        elif self.kf_direction in ('X>0','X<0'):
            LUTfraction = 1

        if self.sethklchck.GetValue():
            strhkl = str(self.sethklcentral.GetValue())[1:-1].split(",")

            if not self.spotlistB.GetValue():
                wx.MessageBox('Please check Spots Set B', 'INFO')

            H, K, L = strhkl
            H, K, L = int(H), int(K), int(L)
            # LUT with cubic symmetry does not have negative L
            if L < 0:
                restrictLUT_cubicSymmetry = False

            set_central_spots_hkl = [[int(H), int(K), int(L)]]

        # restrict LUT if allowed and if crystal is cubic
        restrictLUT_cubicSymmetry = restrictLUT_cubicSymmetry and CP.hasCubicSymmetry(
            self.key_material, dictmaterials=self.dict_Materials)

        print("set_central_spots_hkl", set_central_spots_hkl)
        print("restrictLUT_cubicSymmetry", restrictLUT_cubicSymmetry)

        LUT_with_rules = self.applyrulesLUT.GetValue()

        self.getparams_for_irpfile()
        if self.IndexationParameters["writefolder"] is None:
            self.IndexationParameters["writefolder"] = OSLFGUI.askUserForDirname(self)
        fullpathirp = os.path.join(self.IndexationParameters["writefolder"],self.output_irp.GetValue())
        self.Save_irp_configfile(outputfile=fullpathirp)

        self.textprocess.SetLabel("Processing Indexation")
        self.gauge.SetRange(nbmax_probed * nb_central_spots)

        #         import longtask as LTask
        #
        #         taskboard = LTask.MainFrame(None, -1, INDEX.getOrientMatrices,
        #                                        [spot_index_central,
        #                                     energy_max,
        #                                     Tabledistance[:nbmax_probed, :nbmax_probed],
        #                                     self.select_theta, self.select_chi],
        #                                     (n,
        #                                     ResolutionAngstrom,
        #                                     B,
        #                                     restrictLUT_cubicSymmetry,
        #                                     None,
        #                                     rough_tolangle,
        #                                     fine_tolangle,
        #                                     Minimum_MatchesNb,
        #                                     self.key_material,
        #                                     0,
        #                                     3,
        #                                     0,
        #                                     None,  # addmatrix
        #                                     0,  # verbose
        #                                     detectorparameters,
        #                                     set_central_spots_hkl,
        #                                     verbosedetails,
        #                                     None))  # actually will be overwritten by self of MainFrame
        #
        #         taskboard.Show(True)
        #
        #         return

        # autoindexation core procedure
        # print("self.IndexationParameters['dict_Materials']",self.IndexationParameters['dict_Materials']
        excludespotspairs = [[0, 0]]
        print('spotssettype', spotssettype)
        if spotssettype in ("rangeset", ):
            res = INDEX.getOrientMatrices(spot_index_central,
                                    energy_max,
                                    Tabledistance[:nbmax_probed, :nbmax_probed],
                                    self.select_theta,
                                    self.select_chi,
                                    n=n,
                                    ResolutionAngstrom=ResolutionAngstrom,
                                    B=B,
                                    cubicSymmetry=restrictLUT_cubicSymmetry,
                                    hexagonalSymmetry=CP.isHexagonal(latticeparams),
                                    LUT=None,
                                    LUT_tol_angle=rough_tolangle,
                                    MR_tol_angle=fine_tolangle,
                                    Minimum_Nb_Matches=Minimum_MatchesNb,
                                    key_material=self.key_material,
                                    plot=0,
                                    verbose=0,
                                    detectorparameters=detectorparameters,
                                    addMatrix=None,  # To add a priori good candidates...
                                    set_central_spots_hkl=set_central_spots_hkl,
                                    verbosedetails=1,  # verbosedetails,
                                    gauge=self.gauge,
                                    dictmaterials=self.IndexationParameters['dict_Materials'],
                                    MaxRadiusHKL=False,#True could be OK for this workflow
                                    LUT_with_rules=LUT_with_rules,
                                    excludespotspairs=excludespotspairs,
                                    LUTfraction=LUTfraction)

        elif spotssettype in ('listsetA', 'listsetAB', ):
            # and spotsB is checked
            if spotssettype in ('listsetA', ):
                spotsB = spot_index_central
            print('arguments of  getOrientMatrices_fromTwoSets()')
            print("--->",spot_index_central, spotsB,
                                                energy_max, self.select_theta, self.select_chi,
                                                n, self.key_material, rough_tolangle,
                                                detectorparameters,
                                                set_central_spots_hkl,
                                                Minimum_MatchesNb,
                                                LUT_with_rules,
                                                excludespotspairs,
                                                LUTfraction)
            res = INDEX.getOrientMatrices_fromTwoSets(spot_index_central, spotsB,
                                                energy_max, self.select_theta, self.select_chi,
                                                n, self.key_material, rough_tolangle,
                                                detectorparameters,
                                                set_hkl_1=set_central_spots_hkl,
                                                minimumNbMatches=Minimum_MatchesNb,
                                                LUT_with_rules=LUT_with_rules,
                                                excludespotspairs=excludespotspairs,
                                                LUTfraction=LUTfraction)

        if len(res[0]) > 0:
            self.bestmatrices, stats_res = res
            print('getOrientMatrices_SubSpotsSets found %d solutions', len(res[0]))
        else:
            wx.MessageBox('Sorry! Nothing found !!\nTry to increase nLUT or the nb of spots '
                                                            'probed in spots sets A and B')
            return

        # Update DataSet Object
        if self.DataSet is None:
            print("self.bestmatrices, stats_res")
            print(self.bestmatrices, stats_res)
            return

        self.DataSet.key_material = self.key_material
        self.DataSet.emin = 5
        self.DataSet.emax = energy_max

        self.textprocess.SetLabel("Indexation Completed")

        print("General stats_res before filtering and removing duplicates", stats_res)
        nb_solutions = len(self.bestmatrices)

        keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[self.key_material][1])

        if set_central_spots_hkl not in (None, [None]):
            keep_only_equivalent = False

        # print("self.bestmatrices before")
        for ra, ub in enumerate(self.bestmatrices):
            print("\nrank : %d" % ra)
            print(ub)
        if nb_solutions > 1:
            print("Merging matrices")
            # print("keep_only_equivalent = %s" % keep_only_equivalent)
            self.bestmatrices, stats_res = ISS.MergeSortand_RemoveDuplicates(
                                                        self.bestmatrices,
                                                        stats_res,
                                                        Minimum_MatchesNb,
                                                        tol=0.005,
                                                        keep_only_equivalent=keep_only_equivalent,
                                                    )

        print("stats_res", stats_res)
        nb_solutions = len(self.bestmatrices)
        # print("self.bestmatrices after")
        # for ra, ub in enumerate(self.bestmatrices):
        #     print("\nrank : %d" % ra)
        #     print(ub)
        computingtime = time.time()-t0
        print('Computing time ===> %.2f' % computingtime)

        print("spot_index_central", spot_index_central)

        if nb_solutions:
            print("%d matrice(s) found" % nb_solutions)
            print("self.bestmatrices")
            print(self.bestmatrices)
            print("\nEach Matrix is stored in 'MatIndex_#' for further simulation")
            for k in range(nb_solutions):
                self.dict_Rot["MatIndex_%d" % (k + 1)] = self.bestmatrices[k]

            stats_properformat = []
            for elem in stats_res:
                elem[0] = int(elem[0])
                elem[1] = int(elem[1])
                stats_properformat.append(tuple(elem))

            simulparameters = {}
            simulparameters["detectordiameter"] = self.detectordiameter
            simulparameters["kf_direction"] = self.kf_direction
            simulparameters["detectordistance"] = self.defaultParam[0]
            simulparameters["pixelsize"] = self.pixelsize

            # a single central point were used for distance recognition
            if nb_central_spots == 1:

                self.TwicethetaChi_solution = [0 for k_solution in range(nb_solutions)]
                paramsimul = []

                emax = int(self.emax.GetValue())
                emin = 5

                for k_solution in range(nb_solutions):

                    orientmatrix = self.bestmatrices[k_solution]

                    # only orientmatrix, self.key_material are used ----------------------
                    vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
                    # old definition of grain
                    grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]
                    # ------------------------------------------------------------------

                    # normally in this method fastcompute = 1, gives 2theta, chi
                    TwicethetaChi = LT.SimulateResult(grain, emin, emax,
                                                        simulparameters,
                                                        ResolutionAngstrom=ResolutionAngstrom,
                                                        fastcompute=1,
                                                        dictmaterials=self.IndexationParameters['dict_Materials'])
                    self.TwicethetaChi_solution[k_solution] = TwicethetaChi

                    paramsimul.append((grain, emin, emax))

                    # to plot best results
                    if self.showplotBox.GetValue():

                        print("Plotting result for emin, emax = %.2f,%.2f"
                            % (emin, int(self.emax.GetValue())))
                        print("#central spot: %d" % spot_index_central)

                        title = ("Classical Indexation Result Plot :  #central spot: %d  solution # %d"
                            % (spot_index_central, k_solution))

                        plotresult = Plot_RefineFrame(self,
                                                        -1,
                                                        title,
                                                        datatype="2thetachi",
                                                        key_material=self.key_material,
                                                        kf_direction=self.kf_direction,
                                                        Params_to_simulPattern=(grain, emin, energy_max),
                                                        ResolutionAngstrom=ResolutionAngstrom,
                                                        MATR=fine_tolangle,
                                                        CCDdetectorparameters=self.CCDdetectorparameters,
                                                        IndexationParameters=self.IndexationParameters,
                                                        StorageDict=self.StorageDict,
                                                        DataSetObject=self.DataSet)

                        plotresult.Show(True)

            # many central points were used for distance recognition
            elif nb_central_spots > 1:

                nb_to_plot = nb_solutions

                emin = 5

                print("Plotting result for emin, emax = %.2f,%.2f"
                    % (emin, int(self.emax.GetValue())))
                self.TwicethetaChi_solution = [0 for m in range(nb_to_plot)]
                paramsimul = []

                emax = float(self.emax.GetValue())

                for m in range(nb_to_plot):

                    orientmatrix = self.bestmatrices[m]

                    # only orientmatrix, self.key_material are used ----------------------
                    vecteurref = np.eye(3)
                    # old definition of grain
                    grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]

                    # update of grain definition is done in SimulateResult()
                    # ------------------------------------------------------------------

                    TwicethetaChi = LT.SimulateResult(grain,
                                                        5,
                                                        emax,
                                                        simulparameters,
                                                        ResolutionAngstrom=ResolutionAngstrom,
                                                        fastcompute=1)

                    self.TwicethetaChi_solution[m] = TwicethetaChi
                    emax = int(self.emax.GetValue())
                    paramsimul.append((grain, emin, emax))

                    if self.showplotBox.GetValue():  # to plot best results selected a priori by user
                        title = "Classical Indexation Result Plot"

                        plotresult = Plot_RefineFrame(self,
                                                -1,
                                                title,
                                                data_added=[TwicethetaChi[0], -TwicethetaChi[1]],
                                                kf_direction=self.kf_direction,
                                                datatype="2thetachi",
                                                key_material=self.key_material,
                                                Params_to_simulPattern=(grain, emin, self.emax.GetValue()),
                                                ResolutionAngstrom=ResolutionAngstrom,
                                                CCDdetectorparameters=self.CCDdetectorparameters,
                                                IndexationParameters=self.IndexationParameters,
                                                StorageDict=self.StorageDict,
                                                DataSetObject=self.DataSet)

                        plotresult.Show(True)

            self.IndexationParameters["paramsimul"] = paramsimul
            self.IndexationParameters["bestmatrices"] = self.bestmatrices
            self.IndexationParameters["TwicethetaChi_solutions"] = self.TwicethetaChi_solution
            # display "statistical" results

            #print('self.IndexationParameters before RecognitionResultCheckBox ',self.IndexationParameters)
            RRCBClassical = RecognitionResultCheckBox(self, -1,
                                                    "Screening Distances Indexation Solutions",
                                                    stats_properformat,
                                                    self.data,
                                                    rough_tolangle,
                                                    fine_tolangle,
                                                    key_material=self.key_material,
                                                    emax=emax,
                                                    ResolutionAngstrom=ResolutionAngstrom,
                                                    kf_direction=self.kf_direction,
                                                    datatype="2thetachi",
                                                    data_2thetachi=(2.0 * self.select_theta, self.select_chi),
                                                    data_XY=self.select_dataXY,
                                                    CCDdetectorparameters=self.CCDdetectorparameters,
                                                    IndexationParameters=self.IndexationParameters,
                                                    StorageDict=self.StorageDict,
                                                    mainframe="billframerc",  # self.mainframe
                                                    DataSetObject=self.DataSet)

            RRCBClassical.Show(True)

            self.indexation_index += 1
            pos_ = self.config_irp_filename[::-1].find("_")
            pos_ = len(self.config_irp_filename) - (pos_ + 1)
            self.config_irp_filename = (
                self.config_irp_filename[:pos_] + "_%d.irp" % self.indexation_index)
            self.output_irp.SetValue(self.config_irp_filename)

        else:  # any matrix was found
            print("!!  Nothing found   !!!")
            wx.MessageBox(
                "! NOTHING FOUND !\nTry to reduce the Minimum Number Matched Spots to catch something!",
                "INFO")

    def OnQuit(self, _):
        """ quit
        """
        self.Close()

if __name__ == "__main__":
    filename = "/home/micha/LaueTools/Examples/Ge/dat_Ge0001.cor"

    (Current_peak_data,
        data_theta,
        data_chi,
        data_pixX,
        data_pixY,
        data_I,
        calib,
        CCDCalibDict,
    ) = IOLT.readfile_cor(filename, output_CCDparamsdict=True)

    indexation_parameters = {}
    indexation_parameters["kf_direction"] = "Z>0"
    indexation_parameters["DataPlot_filename"] = filename
    indexation_parameters["dict_Materials"] = DictLT.dict_Materials
    indexation_parameters["DataToIndex"] = {}
    indexation_parameters["DataToIndex"]["data_theta"] = data_theta
    indexation_parameters["DataToIndex"]["data_chi"] = data_chi
    indexation_parameters["DataToIndex"]["dataXY"] = data_pixX, data_pixY
    indexation_parameters["DataToIndex"]["data_I"] = data_I
    indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = np.arange(len(data_theta))
    indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"] = None
    indexation_parameters["current_processedgrain"] = 0
    indexation_parameters["dict_Rot"] = DictLT.dict_Rot
    indexation_parameters["index_foundgrain"] = 0
    indexation_parameters["detectordiameter"] = 165.0
    indexation_parameters["pixelsize"] = 165.0 / 2048
    indexation_parameters["dim"] = (2048, 2048)
    indexation_parameters["detectorparameters"] = calib
    indexation_parameters["CCDLabel"] = "MARCCD165"

    indexation_parameters["mainAppframe"] = None

    StorageDict = {}
    StorageDict["mat_store_ind"] = 0
    StorageDict["Matrix_Store"] = []
    StorageDict["dict_Rot"] = DictLT.dict_Rot
    StorageDict["dict_Materials"] = DictLT.dict_Materials

    AIGUIApp = wx.App()
    AIGUIframe = DistanceScreeningIndexationBoard(None,
                                                    -1,
                                                    indexation_parameters,
                                                    "test automatic indexation",
                                                    StorageDict=StorageDict,
                                                    DataSetObject=None)
    AIGUIframe.Show()
    AIGUIApp.MainLoop()
