r"""
GUI module to display results of indextion with check box for further plot and refinement

Main author is J. S. Micha:   micha [at] esrf [dot] fr

version Aug 2019
from LaueTools package hosted in

https://gitlab.esrf.fr/micha/lauetools
"""
import sys
import copy

import wx
import numpy as np


if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

if sys.version_info.major == 3:
    from . import lauecore as LT
    from . import CrystalParameters as CP
    from . import dict_LaueTools as DictLT
    from . PlotRefineGUI import Plot_RefineFrame
    from . matchingrate import getProximity

else:
    import lauecore as LT
    import CrystalParameters as CP
    import dict_LaueTools as DictLT
    from PlotRefineGUI import Plot_RefineFrame
    from matchingrate import getProximity

# --- -------------------  Recognition Results
class RecognitionResultCheckBox(wx.Frame):
    """
    Class GUI frame displaying the list of matching results from indexation

    Checkboxes allow user selection to plot the patterns (simulation and exp. data)
    """
    def __init__(self, parent, _id, title, stats_residues, data, DRTA, MATR, key_material="Ge",
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
                                                                        DataSetObject=None):

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

        # print("RecognitionResultCheckBox my parent is ", self.parent)

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

        self.matr_ctrl, self.DRTA = MATR, DRTA
        self.key_material = key_material

        # parameter enabling further simulations
        self.paramsimul = self.IndexationParameters["paramsimul"]
        self.mat_solution = self.IndexationParameters["bestmatrices"]

        self.kf_direction = kf_direction

        self.ResolutionAngstrom = ResolutionAngstrom
        self.data_2thetachi = data_2thetachi

        self.DataSet = DataSetObject
        print("self.DataSet.detectordiameter in init RecognitionResultCheckBox",
                                                                    self.DataSet.detectordiameter)

        self.init_GUI2()
        self.Show(True)

    def init_GUI2(self):

        import wx.lib.stattext as ST

        wx.Frame.__init__(self, self.parent, self._id, self.titlew,
                        size=(600, 50 + 20 * self.nbPotentialSolutions + 120))
        panel = wx.Panel(self, -1)

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        txt = wx.StaticText(panel, -1, "Select Potential Solutions to Check & Plot & Refine")
        txt.SetFont(font3)

        if WXPYTHON4:
            vbox3 = wx.GridSizer(6, 5, 10)
        else:
            vbox3 = wx.GridSizer(6, 3)

        txtmatched = wx.StaticText(panel, -1, "Matched")
        txttheomax = wx.StaticText(panel, -1, "Expected")
        txtmr = wx.StaticText(panel, -1, "Matching Rate(%)")
        txtstd = wx.StaticText(panel, -1, "std. dev.(deg)")

        vbox3.AddMany(
            [(wx.StaticText(panel, -1, "   "), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
            (wx.StaticText(panel, -1, "#Matrix"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                            (txtmatched, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
        (txttheomax, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
        (txtmr, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
        (txtstd, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)])

        print("stats_residues in RecognitionResultCheckBox", self.stats_residues)
        self.solutionline = []
        self.cb = []
        for k in range(self.nbPotentialSolutions):

            nmatched, nmax, std = self.stats_residues[k][:3]
            mattchingrate = nmatched / nmax * 100

            if mattchingrate >= 50.:
                color = (158, 241, 193)
            else:
                color = (255, 255, 255)

            # txtind = wx.StaticText(panel, -1, "%d" % k)
            txtind = ST.GenStaticText(panel, -1, "   %d   " % k)
            txtind.SetBackgroundColour(color)

            self.cb.append(wx.CheckBox(panel, -1))
            self.cb[k].SetValue(False)

            txtstats = []
            txtstats.append(ST.GenStaticText(panel, -1, "%d"%int(nmatched)))
            txtstats.append(ST.GenStaticText(panel, -1, "%d"%int(nmax)))
            txtstats.append(ST.GenStaticText(panel, -1, "%.2f"%float(mattchingrate)))
            txtstats.append(ST.GenStaticText(panel, -1, "%.2f"%float(std)))
            for kt in range(4):
                txtstats[kt].SetBackgroundColour(color)
            vbox3.AddMany([(self.cb[k], 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                        (txtind, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                        (txtstats[0], 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                        (txtstats[1], 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                        (txtstats[2], 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL),
                        (txtstats[3], 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)])

        emintxt = wx.StaticText(panel, -1, "Energy min: ")
        self.SCmin = wx.SpinCtrl(panel, -1, "5", min=5, max=150, size=(70, -1))

        emaxtxt = wx.StaticText(panel, -1, "Energy max: ")
        self.SCmax = wx.SpinCtrl(panel, -1, str(int(self.emax)), min=6, max=150, size=(70, -1))

        plotbtn = wx.Button(panel, -1, "Plot", size=(-1, 50))
        plotbtn.Bind(wx.EVT_BUTTON, self.OnPlot)

        simulbtn = wx.Button(panel, -1, "Simul S3", size=(-1, 50))
        simulbtn.Bind(wx.EVT_BUTTON, self.OnSimulate_S3)

        quitbtn = wx.Button(panel, -1, "Quit", size=(-1, 50))
        quitbtn.Bind(wx.EVT_BUTTON, self.OnQuit)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(plotbtn, 1, wx.ALL)
        hbox.Add(simulbtn, 1, wx.ALL)
        hbox.Add(quitbtn, 1, wx.ALL)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(emintxt, 0, wx.ALIGN_LEFT)
        hbox2.Add(self.SCmin, 1)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(emaxtxt, 0, wx.ALIGN_LEFT)
        hbox3.Add(self.SCmax, 1)

        sizerparam = wx.BoxSizer(wx.VERTICAL)
        sizerparam.Add(txt, 0, wx.ALIGN_CENTER_HORIZONTAL)
        sizerparam.AddSpacer(5)
        sizerparam.Add(vbox3, 0, wx.ALL)
        sizerparam.AddSpacer(10)
        sizerparam.Add(hbox2, 0)
        sizerparam.Add(hbox3, 0)
        sizerparam.Add(hbox, 0, wx.EXPAND)

        panel.SetSizer(sizerparam)
        sizerparam.Fit(self)

        #---------  tooltip
        simulbtn.SetToolTipString('Simulate and Plot Laue Pattern of the 4 children of a selected '
        'solution according to sigam3 operator')
        txtmatched.SetToolTipString('Nb of matched reciprocal space directions between exp. and '
        'simulated Laue Patterns')
        tipmaxnb = 'Nb of expected reciprocal space directions in simulated Laue pattern. '
        'A direction is counted as many as harmonics reciprocal nodes it contains'
        tipmaxnb += 'This number varies as a function of orientation matrix, the material and emax'
        txttheomax.SetToolTipString(tipmaxnb)
        txtmr.SetToolTipString('Matching rate ratio in percent of nb of matched directions and '
        'nb of simulated directions')
        txtstd.SetToolTipString('Standard deviation of angular residues distribution of the matched directions')

    def Select(self, _, index):
        print(index, "!!!")

    def OnQuit(self, _):
        # LaueToolsframe.picky.recognition_possible = True
        self.parent.recognition_possible = True
        self.Close()

    def OnPlot(self, _):  # in RecognitionResultCheckBox
        """
        in RecognitionResultCheckBox
        """
        self.toshow = []
        for k in range(self.nbPotentialSolutions):
            if self.cb[k].GetValue():
                self.toshow.append(k)
        # print "self.toshow",self.toshow
        # LaueToolsframe.picky.toshow = self.toshow

        # if LaueToolsframe.picky.TwicethetaChi_solution: print "LaueToolsframe.picky.TwicethetaChi_solution",LaueToolsframe.picky.TwicethetaChi_solution
        # print "self.TwicethetaChi_solution in RecognitionResultCheckBox"#,self.TwicethetaChi_solution

        if len(self.toshow) > 0:  # at least one plot is asked by user
            Emin = int(self.SCmin.GetValue())
            Emax = int(self.SCmax.GetValue())

            print('self.paramsimul in in RecognitionResultCheckBox', self.paramsimul)
            # build all selected plots
            for ind in self.toshow:

                # print "self.data in Onplot() of RecognitionResultCheckBox", self.data
                grain = copy.copy(self.paramsimul[ind][0])  # 1 grain simulation parameter

                print("\n***** selected grain in OnPlot in RecognitionResultCheckBox", grain)
                Params_to_simulPattern = (grain, Emin, Emax)
                print("\n****** Params_to_simulPattern", Params_to_simulPattern)

                newplot = Plot_RefineFrame(self, -1, "matrix #%d" % ind,
                                            kf_direction=self.kf_direction,
                                            ImageArray=self.ImageArray,
                                            datatype=self.datatype,
                                            key_material=self.key_material,
                                            Params_to_simulPattern=Params_to_simulPattern,
                                            ResolutionAngstrom=self.ResolutionAngstrom,
                                            MATR=self.matr_ctrl,
                                            CCDdetectorparameters=self.CCDdetectorparameters,
                                            IndexationParameters=self.IndexationParameters,
                                            StorageDict=self.StorageDict,
                                            DataSetObject=self.DataSet)

                newplot.Show(True)

    def OnSimulate_S3(self, _):
        """ Simulate sigma3 children Laue Pattern from parent Laue Pattern
        in RecognitionResultCheckBox

        LaueToolsframe.dict_Vect = {'Default':[[1, 0,0],[0, 1,0],[0, 0,1]],
                        'sigma3_1':[[-1./3, 2./3, 2./3],[2./3,-1./3, 2./3],[2./3, 2./3,-1./3]],
                        'sigma3_2':[[-1./3,-2./3, 2./3],[-2./3,-1./3,-2./3],[2./3,-2./3,-1./3]],
                        'sigma3_3':[[-1./3, 2./3,-2./3],[2./3,-1./3,-2./3],[-2./3,-2./3,-1./3]],
                        'sigma3_4':[[-1./3,-2./3,-2./3],[-2./3,-1./3, 2./3],[-2./3, 2./3,-1./3]]
                        }

        """
        emax = int(self.SCmax.GetValue())
        emin = int(self.SCmin.GetValue())

        self.toshow = []
        for k in range(self.nbPotentialSolutions):
            if self.cb[k].GetValue():
                self.toshow.append(k)

        # taking the only and lowest index of selected grains matrix after recognition
        try:
            parent_matrix_index = self.toshow[0]
        except IndexError:
            wx.MessageBox("Please check a solution!", "info")
            return

        # plotting frame for parent grain
        # print("self.data in OnSimulate_S3", self.data)
        if self.data is None:
            wx.MessageBox("self.data is empty!", "info")
            return

        print(' Choosing matrix solution #%d'%parent_matrix_index)
        grain = self.paramsimul[parent_matrix_index][0]

        Params_to_simulPattern = (grain, emin, emax)
        print("Params_to_simulPattern in OnSimulate_S3", Params_to_simulPattern)

        parentGrainPlot = Plot_RefineFrame(self,
                                            -1,
                                            "parent grain matrix #%d" % parent_matrix_index,
                                            datatype=self.datatype,
                                            ImageArray=self.ImageArray,
                                            kf_direction=self.kf_direction,
                                            key_material=self.key_material,
                                            Params_to_simulPattern=Params_to_simulPattern,
                                            ResolutionAngstrom=self.ResolutionAngstrom,
                                            MATR=self.matr_ctrl,
                                            CCDdetectorparameters=self.CCDdetectorparameters,
                                            IndexationParameters=self.IndexationParameters,
                                            StorageDict=self.StorageDict,
                                            DataSetObject=self.DataSet)

        parentGrainPlot.Show(True)

        # plotting frame for the four sigma3 daughters

        listmatsigma = [DictLT.dict_Vect["sigma3_" + str(ind)] for ind in (1, 2, 3, 4)]

        res_sigma = []

        self.TwicethetaChi_solution = []
        paramsimul = []
        list_childmatrices = []

        print('***********  ----- Calculating LP of child grains -----  *********\n')
        # four sigma 3 operator in listmatsigma
        for k_matsigma, vecteurref in enumerate(listmatsigma):

            parent_grain_matrix = self.mat_solution[parent_matrix_index]
            #print('self.mat_solution',parent_grain_matrix)
            element = self.key_material
            grain = [vecteurref, [1.0, 1.0, 1.0], parent_grain_matrix, element]

            ChildMatrix = np.dot(parent_grain_matrix, vecteurref)
            list_childmatrices.append(ChildMatrix)

            # PATCH: redefinition of grain to simulate any unit cell (not only cubic) ---
            key_material = grain[3]
            grain = CP.Prepare_Grain(key_material, ChildMatrix,
                                dictmaterials=self.IndexationParameters['dict_Materials'])

            # print('ChildMatrix  #%d'%k_matsigma, ChildMatrix)
            # print('child grain  ', grain)

            # array(vec) and array(indices)(here with fastcompute = 0 array(indices) = 0) of spots exiting the crystal in 2pi steradian(Z>0)
            spots2pi = LT.getLaueSpots(DictLT.CST_ENERGYKEV / float(emax),
                                        DictLT.CST_ENERGYKEV / float(emin),
                                        [grain],
                                        [[""]],
                                        fastcompute=1,
                                        ResolutionAngstrom=self.ResolutionAngstrom,
                                        fileOK=0,
                                        verbose=0,
                                        dictmaterials=self.StorageDict['dict_Materials'])
            # print('spots2pi',spots2pi)
            # print('len(spots2pi',len(spots2pi[0][0]))

            # print('PARAMS')
            # print(self.CCDdetectorparameters)
            # 2theta, chi of spot which are on camera(with harmonics)
            TwicethetaChi = LT.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1,
                                            kf_direction=self.kf_direction,
                                            detectordistance=self.CCDdetectorparameters['detectorparameters'][0],
                                            detectordiameter=self.CCDdetectorparameters['detectordiameter'],
                                            pixelsize=self.CCDdetectorparameters['pixelsize'],
                                            dim=self.CCDdetectorparameters['framedim'])

            #print('TwicethetaChi',TwicethetaChi)
            self.TwicethetaChi_solution.append(TwicethetaChi)

            # print("*-**********************")
            # print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
            toutsigma3 = getProximity(TwicethetaChi,
                                        np.array(self.data[0]) / 2.0,
                                        np.array(self.data[1]))
            # print("calcul residues", toutsigma3[2:])
            # print(vecteurref)
            res_sigma.append(toutsigma3[2:])

            paramsimul.append((grain, emin, emax))

            if 0:
                plotsigma = Plot_RefineFrame(self,
                                        -1,
                                        "sigma #%d" % k_matsigma,
                                        datatype=self.datatype,
                                        ImageArray=self.ImageArray,
                                        kf_direction=self.kf_direction,
                                        key_material=self.key_material,
                                        Params_to_simulPattern=(grain, emin, emax),
                                        ResolutionAngstrom=self.ResolutionAngstrom,
                                        MATR=self.matr_ctrl,
                                        CCDdetectorparameters=self.CCDdetectorparameters,
                                        IndexationParameters=self.IndexationParameters,
                                        StorageDict=self.StorageDict,
                                        DataSetObject=self.DataSet)

                plotsigma.current_matrix = ChildMatrix
                plotsigma.current_elem_label = self.key_material

                # this order is very important!!
                plotsigma.SimulParam = (grain, emin, emax)
                plotsigma.ResolutionAngstrom = self.ResolutionAngstrom
                plotsigma.Simulate_Pattern()


                plotsigma.recognition_possible = True
                #            plotsigma.plotPanel = wxmpl.PlotPanel(plotsigma, -1, size=(5, 3), autoscaleUnzoom=False)
                #            wxmpl.EVT_POINT(plotsigma, plotsigma.plotPanel.GetId(), plotsigma._on_point_choice)
                plotsigma.listbuttonstate = [0] * 3
                plotsigma._replot()
                plotsigma.Show(True)

        print('self.IndexationParameters["paramsimul"]', self.IndexationParameters["paramsimul"])
        dict_indexationparameters = copy.copy(self.IndexationParameters)
        dict_indexationparameters["paramsimul"] = paramsimul
        dict_indexationparameters["bestmatrices"] = list_childmatrices

        RRCB = RecognitionResultCheckBox(self,
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
                                            IndexationParameters=dict_indexationparameters,
                                            StorageDict=self.StorageDict,
                                            mainframe="billframerc",  # self.mainframe
                                            DataSetObject=self.DataSet)

        RRCB.TwicethetaChi_solution = self.TwicethetaChi_solution

        return True
