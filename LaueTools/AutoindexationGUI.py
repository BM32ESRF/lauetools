import wx
import numpy as np
import copy
import sys

if sys.version_info.major == 3:
    from . import lauecore as LT
    from . import CrystalParameters as CP
    from . import indexingSpotsSet as ISS
    from . import indexingAnglesLUT as INDEX
    from . import matchingrate
    from . import IOLaueTools as IOLT
    from . import generaltools as GT
    from . import dict_LaueTools as DictLT
    from . PlotRefineGUI import Plot_RefineFrame

else:
    import lauecore as LT
    import CrystalParameters as CP
    import indexingSpotsSet as ISS
    import indexingAnglesLUT as INDEX
    import matchingrate
    import IOLaueTools as IOLT
    import generaltools as GT
    import dict_LaueTools as DictLT
    from PlotRefineGUI import Plot_RefineFrame


SIZE_PLOTTOOLS = (8, 6)


# --- ----------------   Classical Indexation Board
class DistanceScreeningIndexationBoard(wx.Frame):
    """
    Class of GUI for the automatic indexation board of
    a single peak list with a single material or structure
    """

    def __init__(
        self,
        parent,
        _id,
        indexation_parameters,
        title,
        StorageDict=None,
        DataSetObject=None,
    ):

        wx.Frame.__init__(self, parent, _id, title, size=(700, 500))

        self.panel = wx.Panel(
            self, -1, style=wx.SIMPLE_BORDER, size=(690, 385), pos=(5, 5)
        )

        if parent is not None:
            self.parent = parent
            self.mainframe = self.parent
        else:
            self.mainframe = self

        self.kf_direction = indexation_parameters["kf_direction"]
        self.DataPlot_filename = indexation_parameters["DataPlot_filename"]
        self.dict_Materials = indexation_parameters["dict_Materials"]
        self.dict_Rot = indexation_parameters["dict_Rot"]

        self.current_exp_spot_index_list = indexation_parameters["DataToIndex"][
            "current_exp_spot_index_list"
        ]
        #         print "self.current_exp_spot_index_list", self.current_exp_spot_index_list
        self.data_theta = indexation_parameters["DataToIndex"]["data_theta"]
        self.data_chi = indexation_parameters["DataToIndex"]["data_chi"]
        self.data_I = indexation_parameters["DataToIndex"]["data_I"]
        self.dataXY_exp = indexation_parameters["DataToIndex"]["dataXY"]

        self.ClassicalIndexation_Tabledist = indexation_parameters["DataToIndex"][
            "ClassicalIndexation_Tabledist"
        ]

        self.defaultParam = indexation_parameters["detectorparameters"]
        self.detectordiameter = indexation_parameters["detectordiameter"]
        self.pixelsize = indexation_parameters["pixelsize"]
        self.framedim = indexation_parameters["dim"]

        self.CCDLabel = indexation_parameters["CCDLabel"]

        self.datatype = "2thetachi"

        self.CCDdetectorparameters = {}
        self.CCDdetectorparameters["CCDcalib"] = indexation_parameters[
            "detectorparameters"
        ]
        self.CCDdetectorparameters["framedim"] = indexation_parameters["dim"]
        self.CCDdetectorparameters["pixelsize"] = indexation_parameters["pixelsize"]
        self.CCDdetectorparameters["CCDLabel"] = indexation_parameters["CCDLabel"]
        self.CCDdetectorparameters["detectorparameters"] = indexation_parameters[
            "detectorparameters"
        ]
        self.CCDdetectorparameters["detectordiameter"] = indexation_parameters[
            "detectordiameter"
        ]

        self.IndexationParameters = indexation_parameters
        self.IndexationParameters["Filename"] = indexation_parameters[
            "DataPlot_filename"
        ]
        self.IndexationParameters["DataPlot_filename"] = indexation_parameters[
            "DataPlot_filename"
        ]
        self.IndexationParameters["current_processedgrain"] = indexation_parameters[
            "current_processedgrain"
        ]
        self.IndexationParameters["mainAppframe"] = indexation_parameters[
            "mainAppframe"
        ]
        self.IndexationParameters["indexationframe"] = self

        print(
            "keys of self.IndexationParameters in DistanceScreeningIndexationBoard",
            list(self.IndexationParameters.keys()),
        )

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

        self.initGUI()

    def initGUI(self):
        # GUI

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        title1 = wx.StaticText(self.panel, -1, "Parameters", (15, 15))
        title1.SetFont(font3)
        wx.StaticText(
            self.panel,
            -1,
            "Current File:        %s" % self.DataPlot_filename,
            (180, 15),
        )

        emaxtxt = wx.StaticText(self.panel, -1, "Energy max.: ", (50, 45))
        self.emax = wx.SpinCtrl(
            self.panel, -1, "22", (220, 40), (60, -1), min=10, max=150
        )

        resangtxt = wx.StaticText(
            self.panel, -1, "Min. Resolved Lattice Spacing", (320, 45)
        )
        self.ResolutionAngstromctrl = wx.TextCtrl(
            self.panel, -1, "False", (550, 45), (100, -1)
        )

        elemtxt = wx.StaticText(self.panel, -1, "Materials or Structure: ", (50, 75))
        self.SetMaterialsCombo(0)

        self.refresh = wx.Button(self.panel, -1, "Refresh", (420, 70))
        self.refresh.Bind(wx.EVT_BUTTON, self.SetMaterialsCombo)

        luttxt = wx.StaticText(self.panel, -1, "LUT Nmax", (540, 75))
        self.nLUT = wx.TextCtrl(self.panel, -1, "3", (620, 70), (30, -1))

        rsstxt = wx.StaticText(
            self.panel, -1, "Recognition spots set Size(RSSS): ", (15, 115)
        )
        self.nbspotmax = wx.SpinCtrl(
            self.panel, -1, "10", (270, 110), (60, -1), min=1, max=1000
        )

        nbspots_in_data = len(self.current_exp_spot_index_list)
        mssstxt = wx.StaticText(
            self.panel, -1, "Matching spots set Size(MSSS): ", (370, 115)
        )
        self.nbspotmaxformatching = wx.SpinCtrl(
            self.panel,
            -1,
            str(nbspots_in_data),
            (600, 110),
            (60, -1),
            min=3,
            max=nbspots_in_data,
        )

        cstxt = wx.StaticText(self.panel, -1, "Central spot(s)", (15, 143))
        cstxt2 = wx.StaticText(self.panel, -1, "ex: 0 or [0,1,2,8] ", (15, 160))
        self.spotlist = wx.TextCtrl(self.panel, -1, "0", (250, 160), (160, -1))
        cstxt3 = wx.StaticText(self.panel, -1, "(must be <= RSSS-1)", (15, 177))

        self.sethklchck = wx.CheckBox(self.panel, -1, "set hkl", (460, 143))
        self.sethklcentral = wx.TextCtrl(
            self.panel, -1, "[1,0,0]", (460, 177), (160, -1)
        )

        drtatxt = wx.StaticText(
            self.panel, -1, "Dist. Recogn. Tol. Angle(deg)", (15, 210)
        )
        self.DRTA = wx.TextCtrl(self.panel, -1, "0.5", (250, 205))

        mtatxt = wx.StaticText(
            self.panel, -1, "Matching Tolerance Angle(deg)", (15, 240)
        )
        self.MTA = wx.TextCtrl(self.panel, -1, "0.2", (250, 235))

        mnmstxt = wx.StaticText(
            self.panel, -1, "Minimum Number Matched Spots: ", (15, 270)
        )
        self.MNMS = wx.SpinCtrl(
            self.panel, -1, "15", (250, 265), (60, -1), min=1, max=500
        )

        self.showplotBox = wx.CheckBox(self.panel, -1, "Plot Best result", (15, 305))
        self.showplotBox.SetValue(False)

        wx.StaticText(self.panel, -1, "Max. Nb solutions", (170, 305))
        self.Max_Nb_Solutions = wx.SpinCtrl(
            self.panel, -1, "3", (300, 300), (60, -1), min=1, max=20
        )

        self.indexation_index = 0
        self.config_irp_filename = (
            self.DataPlot_filename[:-4] + "_%d.irp" % self.indexation_index
        )
        wx.StaticText(self.panel, -1, "Saving parameters in config file", (410, 300))
        self.output_irp = wx.TextCtrl(
            self.panel, -1, "%s" % self.config_irp_filename, (410, 320), size=(250, -1)
        )

        self.filterMatrix = wx.CheckBox(
            self.panel, -1, "Remove equivalent Matrices (cubic symmetry)", (15, 340)
        )
        self.filterMatrix.SetValue(True)

        self.StartButton = wx.Button(self.panel, 1, "Start", (25, 380), (200, 60))
        self.Bind(wx.EVT_BUTTON, self.OnStart, id=1)

        wx.Button(self.panel, 2, "Quit", (250, 380), (110, 60))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=2)

        self.verbose = wx.CheckBox(self.panel, -1, "Print details", (460, 220))
        self.verbose.SetValue(False)

        self.textprocess = wx.StaticText(
            self.panel, -1, "                     ", (400, 375)
        )
        self.gauge = wx.Gauge(self.panel, -1, 1000, (400, 400), size=(250, 25))

        self.sb = self.CreateStatusBar()

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
            "Refresh the Material list to admit new materials built in other LaueToolsGUI menus."
        )

        luttip = "Choose the largest hkl index to build the reference angular distance Looking up Table (LUT). Given n, the LUT contains all mutual angles between normals of lattice planes from (0,0,1) to (n,n,n-1) types"
        luttxt.SetToolTipString(luttip)
        self.nLUT.SetToolTipString(luttip)

        tsstip = "Largest experimental spot index for trying to recognise a distance in reference structure distance LUT from all spots in central spots list."
        rsstxt.SetToolTipString(tsstip)
        self.nbspotmax.SetToolTipString(tsstip)

        mssstip = "Number of experimental spots used find the best matching with the Laue Pattern simulated according to a recognised angle in LUT."
        mssstxt.SetToolTipString(mssstip)
        self.nbspotmaxformatching.SetToolTipString(mssstip)

        cstip = 'Experimental spot index (integer), list of spot indices to be considered as central spots, OR for example "to12" meaning spot indices ranging from 0 to 12 (included). Angles between each central spot and every spots of "recognition spots set" will calculated and compared to angles in reference LUT.\n'
        cstip += 'List of spots must written in with bracket (e.g. [0,1,2,5,8]). Central spots index must be strictly lower to nb of spots of the "recognition set".'
        cstxt.SetToolTipString(cstip)
        cstxt2.SetToolTipString(cstip)
        self.spotlist.SetToolTipString(cstip)
        cstxt3.SetToolTipString(cstip)

        drtatip = "Tolerance angle (in degree) within which an experimental angle (between a central and a recognition set spot) must be close to a reference angle in LUT to be used for simulation the Laue pattern (to be matched to the experimenal one)."
        drtatxt.SetToolTipString(drtatip)
        self.DRTA.SetToolTipString(drtatip)

        mtatip = "Tolerance angle (in degree) within which an experimental and simulated spots are close enough to be considered as matched."
        mtatxt.SetToolTipString(mtatip)
        self.MTA.SetToolTipString(mtatip)

        mnmstip = "Minimum number of matched spots (experimental with simulated one) to display the matching results of a possible indexation solution."
        mnmstxt.SetToolTipString(mnmstip)
        self.MNMS.SetToolTipString(mnmstip)

        self.showplotBox.SetToolTipString(
            'Plot all exp. and theo. Laue Patterns for which the number of matched spots is larger than "Minimum Number Matched spots".'
        )

        self.filterMatrix.SetToolTipString(
            "Keep only one orientation matrix for matrices which are equivalent (cubic symmetry unit cell vectors permutations)."
        )

        sethkltip = "Set the [h,k,l] Miller indices of central spot. This will reduce the running time of recognition."
        self.sethklchck.SetToolTipString(sethkltip)
        self.sethklcentral.SetToolTipString(sethkltip)

        maxtip = "Maximum number of solutions per central spots to be kept since the same orientation matrix or equivalent can be found from different experimental pairs.\n"
        self.Max_Nb_Solutions.SetToolTipString(maxtip)

        self.verbose.SetToolTipString("Display details for long indexation procedure")

    def SetMaterialsCombo(self, evt):
        self.list_materials = sorted(self.dict_Materials.keys())

        self.combokeymaterial = wx.ComboBox(
            self.panel,
            4,
            "Ge",
            (220, 70),
            size=(150, -1),
            choices=self.list_materials,
            style=wx.CB_READONLY,
        )

        self.combokeymaterial.Bind(wx.EVT_COMBOBOX, self.EnterCombokeymaterial)

    def EnterCombokeymaterial(self, event):
        """
        in classicalindexation
        """
        item = event.GetSelection()
        self.key_material = self.list_materials[item]
        self.filterMatrix.SetValue(CP.hasCubicSymmetry(self.key_material))

        self.sb.SetStatusText(
            "Selected material: %s" % str(self.dict_Materials[self.key_material])
        )
        event.Skip()

    def getparams_for_irpfile(self):

        # first element is omitted
        List_options = ISS.LIST_OPTIONS_INDEXREFINE[1:]
        # save config file .irp
        #         LIST_OPTIONS_INDEXREFINE =['nb materials',
        #                             'key material',
        #                              'nbGrainstoFind',
        #                              'emin',
        #                             'emax',
        #                             'MATCHINGRATE THRESHOLD IAL',
        #                             'AngleTolLUT',
        #                             'MATCHINGRATE ANGLE TOL',
        #                             'NBMAXPROBED',
        #                             'MinimumNumberMatches',
        #                             'central spots indices',
        #                             'ResolutionAngstrom',
        #                             'nLUTmax',
        #                             'setCentralSpotsHKL',
        #                             'UseIntensityWeights',
        #                             'nbSpotsToIndex',
        #                             'List Matching Tol Angles',
        #                             'Spots Order Reference File']

        sethklcentral = "None"
        self.spotsorder = "None"
        if self.sethklchck.GetValue():
            sethklcentral = str(self.sethklcentral.GetValue())

        MatchingAngleTol = float(self.MTA.GetValue())

        nbSpotsToIndex = 1000

        List_Ctrls = [
            self.combokeymaterial,
            1,
            5.0,
            self.emax,
            100.0,
            self.DRTA,
            MatchingAngleTol,
            self.nbspotmax,
            6,
            self.spotlist,
            self.ResolutionAngstromctrl,
            self.nLUT,
            sethklcentral,
            True,
            nbSpotsToIndex,
            [MatchingAngleTol, MatchingAngleTol / 2.0],
            self.spotsorder,
        ]

        #         for opt, val in zip(List_options, List_Ctrls):
        #             print "%s =" % opt, val

        self.dict_param = {}
        flag = True
        print("len(List_options)", len(List_options))
        print("len(List_Ctrls)", len(List_Ctrls))

        for kk, option_key in enumerate(List_options):
            if not isinstance(List_Ctrls[kk], (int, str, list, float, bool)):
                val = str(List_Ctrls[kk].GetValue())

            #                 print "option_key,kk,val", option_key, kk, val
            else:
                val = List_Ctrls[kk]

            #             if not self.hascorrectvalue(kk, val):
            #                 flag = False
            #                 break

            self.dict_param[option_key] = val

        self.dict_param_list = [self.dict_param]

        print("self.dict_param_list", self.dict_param_list)

        return flag

    def Save_irp_configfile(self, outputfile="mytest_irp.irp"):
        """
        save indexation parameters in .irp file
        """
        ISS.saveIndexRefineConfigFile(self.dict_param_list, outputfilename=outputfile)

    def OnStart(self, event):
        """
        starts classical indexation:
        recognition by the angular distance between two spots from a set of distances
        """
        energy_max = int(self.emax.GetValue())

        ResolutionAngstrom = self.ResolutionAngstromctrl.GetValue()
        if ResolutionAngstrom == "False":
            ResolutionAngstrom = False
        else:
            ResolutionAngstrom = float(ResolutionAngstrom)
        print("ResolutionAngstrom in OnStart Classical indexation", ResolutionAngstrom)

        # whole exp.data spots dict
        #         self.IndexationParameters['AllDataToIndex']
        self.data_theta = self.IndexationParameters["AllDataToIndex"]["data_theta"]
        self.data_chi = self.IndexationParameters["AllDataToIndex"]["data_chi"]
        self.data_I = self.IndexationParameters["AllDataToIndex"]["data_I"]
        self.dataXY_exp = (
            self.IndexationParameters["AllDataToIndex"]["data_pixX"],
            self.IndexationParameters["AllDataToIndex"]["data_pixY"],
        )

        # there is no precomputed angular distances between spots
        if not self.ClassicalIndexation_Tabledist:
            # Selection of spots among the whole data
            # MSSS number
            MatchingSpotSetSize = int(self.nbspotmaxformatching.GetValue())

            # select 1rstly spots that have not been indexed and 2ndly reduced list by user
            index_to_select = np.take(
                self.current_exp_spot_index_list, np.arange(MatchingSpotSetSize)
            )

            self.select_theta = self.data_theta[index_to_select]
            self.select_chi = self.data_chi[index_to_select]
            self.select_I = self.data_I[index_to_select]

            print("index_to_select", index_to_select)
            print("len self.dataXY_exp[0]", len(self.dataXY_exp[0]))

            self.select_dataX = self.dataXY_exp[0][index_to_select]
            self.select_dataY = self.dataXY_exp[1][index_to_select]
            # print select_theta
            # print select_chi

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

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.DataPlot_filename,
        )

        self.select_dataXY = (self.select_dataX, self.select_dataY)

        wrongsets = False
        # recognition spots set size (RSSS)
        nbmax_probed = int(self.nbspotmax.GetValue())

        spot_list = self.spotlist.GetValue()

        # classical indexation
        # print "spot_list in OnLaunch",spot_list
        if spot_list[0] != "-":
            # print "coucou"
            if spot_list.startswith("["):
                # print "coucou2"
                spot_index_central = str(spot_list)[1:-1].split(",")
                # print spot_index_central
                arr_index = np.array(spot_index_central)

                # print np.array(arr_index, dtype = int)
                spot_index_central = list(np.array(arr_index, dtype=int))
                nb_central_spots = len(spot_index_central)

                if max(spot_index_central) >= nbmax_probed:
                    wrongsets = True
            elif spot_list.startswith("to"):
                spot_index_central = list(range(int(spot_list[2:]) + 1))
                nb_central_spots = len(spot_index_central)
                if max(spot_index_central) >= nbmax_probed:
                    wrongsets = True
            else:
                spot_index_central = int(spot_list)
                nb_central_spots = 1
                if spot_index_central >= nbmax_probed:
                    wrongsets = True

        else:  # minus in front of integer
            spot_index_central = 0
            nb_central_spots = 1

        if wrongsets is True:
            wx.MessageBox(
                "Central spots indices must be strictly lower than the size of Recognition spots set",
                "Error",
            )

        self.key_material = str(self.combokeymaterial.GetValue())
        latticeparams = self.dict_Materials[self.key_material][1]
        B = CP.calc_B_RR(latticeparams)
        # print type(key_material)
        # print type(nbmax_probed)
        # print type(energy_max)

        # read maximum index of hkl for building angles Look Up Table(LUT)
        nLUT = self.nLUT.GetValue()
        try:
            n = int(nLUT)
            if n > 7:
                wx.MessageBox(
                    "! LUT Nmax is too high!\n This value is set to 7 ", "INFO"
                )
            elif n < 1:
                wx.MessageBox(
                    "! LUT Nmax is not positive!\n This value is set to 1 ", "INFO"
                )
            n = min(7, n)
            n = max(1, n)
        except ValueError:
            print("!!  maximum index for building LUT is not an integer   !!!")
            wx.MessageBox(
                "! LUT Nmax is not an integer!\n This value is set to 3 ", "INFO"
            )
            n = 3

        rough_tolangle = float(self.DRTA.GetValue())
        fine_tolangle = float(self.MTA.GetValue())
        Minimum_MatchesNb = int(self.MNMS.GetValue())
        #         print "Recognition tolerance angle ", rough_tolangle
        #         print "Matching tolerance angle ", fine_tolangle
        nb_of_solutions_per_central_spot = int(self.Max_Nb_Solutions.GetValue())

        # detector geometry
        detectorparameters = {}
        detectorparameters["kf_direction"] = self.kf_direction
        detectorparameters["detectorparameters"] = self.defaultParam
        detectorparameters["detectordiameter"] = self.detectordiameter
        detectorparameters["pixelsize"] = self.pixelsize
        detectorparameters["dim"] = self.framedim

        restrictLUT_cubicSymmetry = True
        set_central_spots_hkl = None

        if self.sethklchck.GetValue():
            strhkl = str(self.sethklcentral.GetValue())[1:-1].split(",")
            H, K, L = strhkl
            H, K, L = int(H), int(K), int(L)
            # LUT with cubic symmetry does not have negative L
            if L < 0:
                restrictLUT_cubicSymmetry = False

            set_central_spots_hkl = [[int(H), int(K), int(L)]]

        # restrict LUT if allowed and if crystal is cubic
        restrictLUT_cubicSymmetry = restrictLUT_cubicSymmetry and CP.hasCubicSymmetry(
            self.key_material
        )

        print("set_central_spots_hkl", set_central_spots_hkl)
        print("restrictLUT_cubicSymmetry", restrictLUT_cubicSymmetry)

        self.getparams_for_irpfile()
        self.Save_irp_configfile(outputfile=self.output_irp.GetValue())

        verbosedetails = self.verbose.GetValue()

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
        #                                     nb_of_solutions_per_central_spot,
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

        # indexation procedure
        self.bestmatrices, stats_res = INDEX.getOrientMatrices(
            spot_index_central,
            energy_max,
            Tabledistance[:nbmax_probed, :nbmax_probed],
            self.select_theta,
            self.select_chi,
            n=n,
            ResolutionAngstrom=ResolutionAngstrom,
            B=B,
            cubicSymmetry=restrictLUT_cubicSymmetry,
            LUT=None,
            LUT_tol_angle=rough_tolangle,
            MR_tol_angle=fine_tolangle,
            Minimum_Nb_Matches=Minimum_MatchesNb,
            key_material=self.key_material,
            plot=0,
            nbbestplot=nb_of_solutions_per_central_spot,
            verbose=0,
            detectorparameters=detectorparameters,
            addMatrix=None,  # To add a priori good candidates...
            set_central_spots_hkl=set_central_spots_hkl,
            verbosedetails=1,  # verbosedetails,
            gauge=self.gauge,
        )
        # when nbbestplot is very high  self.bestmatrices contain all matrices
        # with matching rate above Minimum_MatchesNb

        # Update DataSet Object
        if self.DataSet is None:
            print("self.bestmatrices, stats_res")
            print(self.bestmatrices, stats_res)
            return

        self.DataSet.key_material = self.key_material
        self.DataSet.emin = 5
        self.DataSet.emax = energy_max

        self.textprocess.SetLabel("Indexation Completed")

        print("stats_res", stats_res)
        nb_solutions = len(self.bestmatrices)

        keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[self.key_material][1])

        if set_central_spots_hkl not in (None, [None]):
            keep_only_equivalent = False

        print("self.bestmatrices before")
        for ra, ub in enumerate(self.bestmatrices):
            print("\nrank : %d" % ra)
            print(ub)
        if nb_solutions > 1:
            print("Merging matrices")
            print("keep_only_equivalent = %s" % keep_only_equivalent)
            self.bestmatrices, stats_res = ISS.MergeSortand_RemoveDuplicates(
                self.bestmatrices,
                stats_res,
                Minimum_MatchesNb,
                tol=0.0001,
                keep_only_equivalent=keep_only_equivalent,
            )

        print("stats_res", stats_res)
        nb_solutions = len(self.bestmatrices)
        print("self.bestmatrices after")
        for ra, ub in enumerate(self.bestmatrices):
            print("\nrank : %d" % ra)
            print(ub)

        print("Max. Number of Solutions", nb_of_solutions_per_central_spot)
        print("spot_index_central", spot_index_central)

        if nb_solutions:
            print("%d matrice(s) found" % nb_solutions)
            print(self.bestmatrices)
            print("Each Matrix is stored in 'MatIndex_#' for further simulation")
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

                for k_solution in range(nb_solutions):

                    orientmatrix = self.bestmatrices[k_solution]

                    # only orientmatrix, self.key_material are used ----------------------
                    vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
                    # old definition of grain
                    grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]
                    # ------------------------------------------------------------------

                    # normally in this method fastcompute = 1, gives 2theta, chi
                    TwicethetaChi = LT.SimulateResult(
                        grain,
                        5,
                        float(self.emax.GetValue()),
                        simulparameters,
                        ResolutionAngstrom=ResolutionAngstrom,
                        fastcompute=1,
                    )
                    self.TwicethetaChi_solution[k_solution] = TwicethetaChi

                    paramsimul.append((grain, 5, emax))

                    # to plot best results
                    if self.showplotBox.GetValue():

                        print(
                            "Plotting result for emin, emax = %.2f,%.2f"
                            % (5, int(self.emax.GetValue()))
                        )
                        print("#central spot: %d" % spot_index_central)

                        title = (
                            "Classical Indexation Result Plot :  #central spot: %d  solution # %d"
                            % (spot_index_central, k_solution)
                        )

                        plotresult = Plot_RefineFrame(
                            self,
                            -1,
                            title,
                            data=self.data,
                            datatype="2thetachi",
                            data_2thetachi=(2.0 * self.select_theta, self.select_chi),
                            data_XY=self.select_dataXY,
                            key_material=self.key_material,
                            kf_direction=self.kf_direction,
                            Params_to_simulPattern=(grain, 5, energy_max),
                            ResolutionAngstrom=ResolutionAngstrom,
                            DRTA=rough_tolangle,
                            MATR=fine_tolangle,
                            CCDdetectorparameters=self.CCDdetectorparameters,
                            IndexationParameters=self.IndexationParameters,
                            StorageDict=self.StorageDict,
                            mainframe="billframe2",  # self.parent
                            DataSetObject=self.DataSet,
                        )

                        plotresult.Show(True)

            # many central points were used for distance recognition
            elif nb_central_spots > 1:

                nb_to_plot = nb_solutions

                print(
                    "Plotting result for emin, emax = %.2f,%.2f"
                    % (5, int(self.emax.GetValue()))
                )
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

                    TwicethetaChi = LT.SimulateResult(
                        grain,
                        5,
                        emax,
                        simulparameters,
                        ResolutionAngstrom=ResolutionAngstrom,
                        fastcompute=1,
                    )

                    self.TwicethetaChi_solution[m] = TwicethetaChi
                    emax = int(self.emax.GetValue())
                    paramsimul.append((grain, 5, emax))

                    if (
                        self.showplotBox.GetValue()
                    ):  # to plot best results selected a priori by user
                        title = "Classical Indexation Result Plot"

                        plotresult = Plot_RefineFrame(
                            self,
                            -1,
                            title,
                            data=self.data,
                            data_added=[TwicethetaChi[0], -TwicethetaChi[1]],
                            kf_direction=self.kf_direction,
                            datatype="2thetachi",
                            data_2thetachi=(2.0 * self.select_theta, self.select_chi),
                            data_XY=self.select_dataXY,
                            key_material=self.key_material,
                            Params_to_simulPattern=(grain, 5, self.emax.GetValue()),
                            ResolutionAngstrom=ResolutionAngstrom,
                            CCDdetectorparameters=self.CCDdetectorparameters,
                            IndexationParameters=self.IndexationParameters,
                            StorageDict=self.StorageDict,
                            mainframe="billframe2b",  # self.parent
                            DataSetObject=self.DataSet,
                        )

                        plotresult.Show(True)

            self.IndexationParameters["paramsimul"] = paramsimul
            self.IndexationParameters["bestmatrices"] = self.bestmatrices
            self.IndexationParameters[
                "TwicethetaChi_solutions"
            ] = self.TwicethetaChi_solution
            # display "statistical" results
            RRCBClassical = RecognitionResultCheckBox(
                self,
                -1,
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
                DataSetObject=self.DataSet,
            )

            RRCBClassical.Show(True)

            self.indexation_index += 1
            pos_ = self.config_irp_filename[::-1].find("_")
            pos_ = len(self.config_irp_filename) - (pos_ + 1)
            self.config_irp_filename = (
                self.config_irp_filename[:pos_] + "_%d.irp" % self.indexation_index
            )
            self.output_irp.SetValue(self.config_irp_filename)

        else:  # any matrix was found
            print("!!  Nothing found   !!!")
            wx.MessageBox(
                "! NOTHING FOUND !\nTry to reduce the Minimum Number Matched Spots to catch something!",
                "INFO",
            )

    def OnQuit(self, event):
        """ quit
        """
        self.Close()


# --- -------------------  Recognition Results
class RecognitionResultCheckBox(wx.Frame):
    """
    Class GUI frame displaying the list of matching results from indexation

    Checkboxes allow user selection to plot the patterns (simulation and exp. data)
    """

    def __init__(
        self,
        parent,
        _id,
        title,
        stats_residues,
        data,
        DRTA,
        MATR,
        key_material="Ge",
        emax=25,
        ResolutionAngstrom=False,
        kf_direction="Z>0",
        datatype="2thetachi",
        data_2thetachi=(None, None),
        data_XY=(None, None),
        ImageArray=None,
        CCDdetectorparameters=None,
        IndexationParameters=None,
        StorageDict=None,
        mainframe=None,
        DataSetObject=None,
    ):

        self.parent = parent
        self._id = _id
        self.titlew = title
        self.datatype = datatype
        self.stats_residues = stats_residues
        self.emax = emax

        print("self.datatype in RecognitionResultCheckBox ", self.datatype)

        self.CCDdetectorparameters = CCDdetectorparameters
        self.IndexationParameters = IndexationParameters
        self.StorageDict = StorageDict

        self.ImageArray = ImageArray

        if mainframe is not None:
            self.mainframe = mainframe
        else:
            self.mainframe = parent.parent

        print("RecognitionResultCheckBox my parent is ", self.parent)
        #         print "**** \n\nIndexationParameters",IndexationParameters

        self.nbPotentialSolutions = len(stats_residues)

        if self.datatype == "2thetachi":
            # data should be 2theta chi I filename
            pass
        elif self.datatype == "pixels":
            # data should be pixX pixY I filename
            # data_2thetachi containes 2theta chi I
            pass

        self.data = data

        self.data_XY_exp = data_XY

        #         print "data_XY_exp", self.data_XY_exp

        self.matr_ctrl, self.DRTA = MATR, DRTA
        self.key_material = key_material

        # parameter enabling further simulations
        self.paramsimul = self.IndexationParameters["paramsimul"]
        self.mat_solution = self.IndexationParameters["bestmatrices"]

        self.kf_direction = kf_direction

        self.ResolutionAngstrom = ResolutionAngstrom
        self.data_2thetachi = data_2thetachi

        self.DataSet = DataSetObject
        print(
            "self.DataSet.detectordiameter in init RecognitionResultCheckBox",
            self.DataSet.detectordiameter,
        )

        self.init_GUI()

    def init_GUI(self):
        wx.Frame.__init__(
            self,
            self.parent,
            self._id,
            self.titlew,
            size=(600, 50 + 20 * self.nbPotentialSolutions + 120),
        )

        panel = wx.Panel(
            self, -1, size=(400, 50 + 20 * self.nbPotentialSolutions + 120)
        )

        wx.StaticText(
            panel,
            -1,
            "   #Matrix     nb. <MTAR = %.2f       nb. <DRTA = %.2f         std. dev. (deg)"
            % (self.matr_ctrl, self.DRTA),
            (20, 10),
        )

        print("stats_residues in RecognitionResultCheckBox", self.stats_residues)
        self.cb = []
        for k in range(self.nbPotentialSolutions):
            self.cb.append(
                wx.CheckBox(
                    panel,
                    -1,
                    "     "
                    + str(k)
                    + "                      %d                                 %d                                   %.3f"
                    % tuple(self.stats_residues[k][:3]),
                    (10, 35 + 20 * k),
                )
            )
            self.cb[k].SetValue(False)

            # wx.EVT_CHECKBOX(self, self.cb.GetId(), self.Select(k))

        wx.StaticText(
            panel, -1, "Energy min: ", (15, 35 + 20 * self.nbPotentialSolutions + 30)
        )
        self.SCmin = wx.SpinCtrl(
            panel,
            -1,
            "5",
            (95, 35 + 20 * self.nbPotentialSolutions + 30),
            (60, -1),
            min=5,
            max=150,
        )

        wx.StaticText(
            panel, -1, "Energy max: ", (180, 35 + 20 * self.nbPotentialSolutions + 30)
        )
        self.SCmax = wx.SpinCtrl(
            panel,
            -1,
            str(int(self.emax)),
            (260, 35 + 20 * self.nbPotentialSolutions + 30),
            (60, -1),
            min=6,
            max=150,
        )

        wx.Button(panel, 1, "Plot", (40, 35 + 20 * self.nbPotentialSolutions + 60))
        self.Bind(wx.EVT_BUTTON, self.OnPlot, id=1)

        wx.Button(panel, 2, "Simul S3", (130, 35 + 20 * self.nbPotentialSolutions + 60))
        self.Bind(wx.EVT_BUTTON, self.OnSimulate_S3, id=2)

        wx.Button(panel, 3, "Quit", (220, 35 + 20 * self.nbPotentialSolutions + 60))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=3)

        self.Show(True)
        self.Centre()

    def Select(self, event, index):
        print(index, "!!!")

    def OnQuit(self, event):
        # LaueToolsframe.picky.recognition_possible = True
        self.parent.recognition_possible = True
        self.Close()

    def OnPlot(self, event):  # in RecognitionResultCheckBox
        """
        in RecognitionResultCheckBox
        """
        self.toshow = []
        for k in range(self.nbPotentialSolutions):
            if self.cb[k].GetValue() == True:
                self.toshow.append(k)
        # print "self.toshow",self.toshow
        # LaueToolsframe.picky.toshow = self.toshow

        # if LaueToolsframe.picky.TwicethetaChi_solution: print "LaueToolsframe.picky.TwicethetaChi_solution",LaueToolsframe.picky.TwicethetaChi_solution
        # print "self.TwicethetaChi_solution in RecognitionResultCheckBox"#,self.TwicethetaChi_solution

        if len(self.toshow) > 0:  # at least one plot is asked by user
            Emin = int(self.SCmin.GetValue())
            Emax = int(self.SCmax.GetValue())

            # build all selected plots
            for ind in self.toshow:

                # print "self.data in Onplot() of RecognitionResultCheckBox", self.data
                grain = copy.copy(
                    self.paramsimul[ind][0]
                )  # 1 grain simulation parameter

                print(
                    "\n***** selected grain in OnPlot in RecognitionResultCheckBox",
                    grain,
                )
                Params_to_simulPattern = (grain, Emin, Emax)
                print("\n****** Params_to_simulPattern", Params_to_simulPattern)

                newplot = Plot_RefineFrame(
                    self,
                    -1,
                    "matrix #%d" % ind,
                    data=self.data,
                    kf_direction=self.kf_direction,
                    data_XY=self.data_XY_exp,
                    ImageArray=self.ImageArray,
                    data_2thetachi=self.data_2thetachi,
                    datatype=self.datatype,
                    key_material=self.key_material,
                    Params_to_simulPattern=Params_to_simulPattern,
                    ResolutionAngstrom=self.ResolutionAngstrom,
                    MATR=self.matr_ctrl,
                    CCDdetectorparameters=self.CCDdetectorparameters,
                    IndexationParameters=self.IndexationParameters,
                    StorageDict=self.StorageDict,
                    mainframe="billframe",  # self.mainframe
                    DataSetObject=self.DataSet,
                )

                newplot.Show(True)

    def OnSimulate_S3(self, event):
        """ Simulate sigma3 children Laue Pattern from parent Laue Pattern
        in RecognitionResultCheckBox

        LaueToolsframe.dict_Vect = {'Default':[[1, 0,0],[0, 1,0],[0, 0,1]],
                        'sigma3_1':[[-1./3, 2./3, 2./3],[2./3,-1./3, 2./3],[2./3, 2./3,-1./3]],
                        'sigma3_2':[[-1./3,-2./3, 2./3],[-2./3,-1./3,-2./3],[2./3,-2./3,-1./3]],
                        'sigma3_3':[[-1./3, 2./3,-2./3],[2./3,-1./3,-2./3],[-2./3,-2./3,-1./3]],
                        'sigma3_4':[[-1./3,-2./3,-2./3],[-2./3,-1./3, 2./3],[-2./3, 2./3,-1./3]]
                        }
                        
        Quite old function 

        """
        emax = int(self.SCmax.GetValue())
        emin = int(self.SCmin.GetValue())

        self.toshow = []
        for k in range(self.nbPotentialSolutions):
            if self.cb[k].GetValue() == True:
                self.toshow.append(k)

        # taking the only and lowest index of selected grains matrix after recognition
        try:
            parent_matrix_index = self.toshow[0]
        except IndexError:
            wx.MessageBox("Please check a solution!", "info")
            return

        # plotting frame for parent grain
        print("self.data in OnSimulate_S3", self.data)
        if self.data is None:
            wx.MessageBox("self.data is empty!", "info")
            return

        grain = self.paramsimul[parent_matrix_index][0]

        Params_to_simulPattern = (grain, emin, emax)
        print("Params_to_simulPattern in OnSimulate_S3", Params_to_simulPattern)

        parentGrainPlot = Plot_RefineFrame(
            self,
            -1,
            "parent grain matrix #%d" % parent_matrix_index,
            data=self.data,
            datatype=self.datatype,
            data_XY=self.data_XY_exp,
            ImageArray=self.ImageArray,
            data_2thetachi=self.data_2thetachi,
            kf_direction=self.kf_direction,
            key_material=self.key_material,
            Params_to_simulPattern=Params_to_simulPattern,
            ResolutionAngstrom=self.ResolutionAngstrom,
            MATR=self.matr_ctrl,
            CCDdetectorparameters=self.CCDdetectorparameters,
            IndexationParameters=self.IndexationParameters,
            StorageDict=self.StorageDict,
            mainframe="billframe",  # self.mainframe
            DataSetObject=self.DataSet,
        )

        parentGrainPlot.Show(True)

        # plotting frame for the four sigma3 daughters

        listmatsigma = [DictLT.dict_Vect["sigma3_" + str(ind)] for ind in (1, 2, 3, 4)]

        res_sigma = []

        self.TwicethetaChi_solution = []
        paramsimul = []
        list_eq_matrix = []

        # four sigma 3 operator in listmatsigma
        for k_matsigma, vecteurref in enumerate(listmatsigma):

            parent_grain_matrix = self.mat_solution[parent_matrix_index]
            element = self.key_material
            grain = [vecteurref, [1.0, 1.0, 1.0], parent_grain_matrix, element]

            Equivalent_matrix = np.dot(parent_grain_matrix, vecteurref)
            list_eq_matrix.append(Equivalent_matrix)

            # PATCH: redefinition of grain to simulate any unit cell (not only cubic) ---
            key_material = grain[3]
            grain = CP.Prepare_Grain(key_material, Equivalent_matrix)

            # array(vec) and array(indices)(here with fastcompute = 0 array(indices) = 0) of spots exiting the crystal in 2pi steradian(Z>0)
            spots2pi = LT.getLaueSpots(
                DictLT.CST_ENERGYKEV / float(emax),
                DictLT.CST_ENERGYKEV / float(emin),
                [grain],
                [[""]],
                fastcompute=1,
                ResolutionAngstrom=self.ResolutionAngstrom,
                fileOK=0,
                verbose=0,
            )

            # # array(vec) and array(indices)(here with fastcompute = 0 array(indices) = 0) of spots exiting the crystal in 2pi steradian(Z>0)
            # spots2pi = LAUE.generalfabriquespot_fromMat_veryQuick(CST_ENERGYKEV/emax, CST_ENERGYKEV/emin,
            # [grain],
            # 1, fastcompute = 1,
            # fileOK = 0,
            # verbose = 0)

            # 2theta, chi of spot which are on camera(with harmonics)
            TwicethetaChi = LT.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)
            self.TwicethetaChi_solution.append(TwicethetaChi)

            print("*-**********************")
            # print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
            toutsigma3 = matchingrate.getProximity(
                TwicethetaChi, np.array(self.data[0]) / 2.0, np.array(self.data[1])
            )
            print("calcul residues", toutsigma3[2:])
            print(vecteurref)
            res_sigma.append(toutsigma3[2:])

            #             plotsigma = Plot_RefineFrame(self, -1, "sigma #%d" % k_matsigma,
            #                                 data=self.data,
            #                                 kf_direction=self.kf_direction,
            #                                 datatype=self.datatype,
            #                                 data_2thetachi=self.data_2thetachi,
            #                                 key_material=self.key_material,
            #                                 Params_to_simulPattern=(grain, emin, emax),
            #                                 ResolutionAngstrom=self.ResolutionAngstrom,
            #                                 CCDdetectorparameters=self.CCDdetectorparameters,
            #                                 IndexationParameters=self.IndexationParameters,
            #                                 StorageDict=self.StorageDict,
            #                                 mainframe='billframe',  # self.mainframe
            #                                 DataSetObject=self.DataSet
            #                                 )
            plotsigma = Plot_RefineFrame(
                self,
                -1,
                "sigma #%d" % k_matsigma,
                data=self.data,
                datatype=self.datatype,
                data_XY=self.data_XY_exp,
                ImageArray=self.ImageArray,
                data_2thetachi=self.data_2thetachi,
                kf_direction=self.kf_direction,
                key_material=self.key_material,
                Params_to_simulPattern=(grain, emin, emax),
                ResolutionAngstrom=self.ResolutionAngstrom,
                MATR=self.matr_ctrl,
                CCDdetectorparameters=self.CCDdetectorparameters,
                IndexationParameters=self.IndexationParameters,
                StorageDict=self.StorageDict,
                mainframe="billframe",  # self.mainframe
                DataSetObject=self.DataSet,
            )

            plotsigma.current_matrix = Equivalent_matrix
            plotsigma.current_elem_label = self.key_material

            # this order is very important!!
            plotsigma.SimulParam = (grain, emin, emax)
            plotsigma.ResolutionAngstrom = self.ResolutionAngstrom
            plotsigma.Simulate_Pattern()
            paramsimul.append((grain, emin, emax))

            plotsigma.recognition_possible = True
            #            plotsigma.plotPanel = wxmpl.PlotPanel(plotsigma, -1, size=(5, 3), autoscaleUnzoom=False)
            #            wxmpl.EVT_POINT(plotsigma, plotsigma.plotPanel.GetId(), plotsigma._on_point_choice)
            plotsigma.listbuttonstate = [0] * 3
            plotsigma._replot()
            plotsigma.Show(True)

        RRCB = RecognitionResultCheckBox(
            self,
            -1,
            "Potential Solutions from Sigma3 Simulations",
            res_sigma,
            self.data,
            self.DRTA,
            self.matr_ctrl,
            key_material=self.key_material,
            emax=emax,
            ResolutionAngstrom=self.ResolutionAngstrom,
            kf_direction=self.kf_direction,
            datatype=self.datatype,
            data_2thetachi=self.data_2thetachi,
            data_XY=self.data_XY_exp,
            #                                          data_XY=self.select_dataXY,
            CCDdetectorparameters=self.CCDdetectorparameters,
            IndexationParameters=self.IndexationParameters,
            StorageDict=self.StorageDict,
            mainframe="billframerc",  # self.mainframe
            DataSetObject=self.DataSet,
        )

        RRCB.key_material = self.key_material
        RRCB.TwicethetaChi_solution = self.TwicethetaChi_solution
        RRCB.mat_solution = list_eq_matrix
        RRCB.paramsimul = paramsimul
        RRCB.ResolutionAngstrom = self.ResolutionAngstrom

        return True


if __name__ == "__main__":
    filename = "/home/micha/LaueTools/Examples/Ge/dat_Ge0001.cor"

    (
        Current_peak_data,
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
    indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = np.arange(
        len(data_theta)
    )
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
    AIGUIframe = DistanceScreeningIndexationBoard(
        None,
        -1,
        indexation_parameters,
        "test automatic indexation",
        StorageDict=StorageDict,
        DataSetObject=None,
    )
    AIGUIframe.Show()
    AIGUIApp.MainLoop()
