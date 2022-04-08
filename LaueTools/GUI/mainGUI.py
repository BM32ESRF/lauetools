r"""
mainGUI.py is a (big) central GUI for microdiffraction Laue Pattern analysis
and simulation

Formerly its name was LaueToolsGUI.py (located at the project root)

This module belongs to the open source LaueTools project
with a free code repository at at gitlab.esrf.fr

(former version with python 2.7 at https://sourceforge.net/projects/lauetools/)

or for python3 and 2 in

https://gitlab.esrf.fr/micha/lauetools

J. S. Micha July 2021
mailto: micha --+at-+- esrf --+dot-+- fr
"""
from __future__ import absolute_import, division

__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
import pkg_resources
__version__ = pkg_resources.get_distribution('LaueTools').version



import time
import sys
import copy
import os.path
import re

# last modified date of this module is displayed in title and documentation and about
import datetime
import webbrowser

import matplotlib

matplotlib.use("WXAgg")

#from matplotlib import __version__ as matplotlibversion

import numpy as np
import wx

print('----------------------------------------------')
print('-----              Welcome            --------')
print('-----                 to              --------')
print('-----             LaueTools           --------')
print('-----  main Graphical User Interface  --------')
print('----------------------------------------------')

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

    import wx.adv as wxadv

from LaueTools.indexingImageMatching import ComputeGnomon_2
from LaueTools.indexingSpotsSet import spotsset
from LaueTools.lauecore import SimulateResult
from LaueTools.graingraph import give_bestclique
from LaueTools.LaueGeometry import Compute_data2thetachi
from LaueTools.GUI.LaueSimulatorGUI import parametric_Grain_Dialog3

from LaueTools.CrystalParameters import calc_B_RR
from LaueTools.findorient import computesortedangles
from LaueTools.IOLaueTools import writefile_cor, createselecteddata
from LaueTools.dict_LaueTools import (dict_CCD, dict_calib, dict_Materials, dict_Extinc,
                                dict_Transforms, dict_Vect, dict_Rot,
                                dict_Eul, list_CCD_file_extensions,
                                readDict, getwildcardstring, LAUETOOLSFOLDER)
from LaueTools.GUI.PeakSearchGUI import MainPeakSearchFrame
from LaueTools.GUI.DetectorParameters import autoDetectDetectorType
from LaueTools.GUI.DetectorCalibration import MainCalibrationFrame
from LaueTools.GUI.CCDFileParametersGUI import CCDFileParameters
import LaueTools.matchingrate as matchingrate
from LaueTools.GUI.AutoindexationGUI import DistanceScreeningIndexationBoard
from LaueTools.GUI.B0matrixLatticeEditor import B0MatrixEditor
from LaueTools.GUI.ResultsIndexationGUI import RecognitionResultCheckBox
from LaueTools.GUI.OpenSpotsListFileGUI import (askUserForFilename, OpenPeakList, Launch_DetectorParamBoard,
                                                OpenCorfile, SetGeneralLaueGeometry)
from LaueTools.GUI.ManualIndexFrame import ManualIndexFrame
from LaueTools.GUI.MatrixEditor import MatrixEditor_Dialog
import LaueTools.GUI.OpenSpotsListFileGUI as OSLFGUI

LaueToolsProjectFolder = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

pyversion='%s.%s'%(sys.version_info.major,sys.version_info.minor)

print("-- LaueTools Project main folder is", LaueToolsProjectFolder)
print('\n----------------------------------------------')
print('-----              Enjoy              --------')
print('-----        Laue microdiffraction    --------')
print('-----    CRG-IF BM32 beamline at ESRF --------')
print('----------------------------------------------')
print('------LaueTools version --- %s'%__version__)
print('------python version --- %s'%pyversion)
print('------Wxpython version ----%s'%wx.__version__)

SIZE_PLOTTOOLS = (8, 6)
# --- ------------   CONSTANTS
PI = np.pi
DEG = PI / 180.0

# DEFAULT_DETECTORPARAMETERS = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201] #  'MARCCD165'
# DEFAULT_DETECTORPARAMETERS = [77., 1975., 2110., 0.43, -0.22] # sCMOS_16M, sans flip LR' # after debug readmccd 17Jul18
# DEFAULT_DETECTORPARAMETERS = [77.4, 983., 977., 0.32, -0.28] # sCMOS, sans flip LR #

DEFAULT_CCDCAMERA = "sCMOS"
DEFAULT_DETECTORPARAMETERS = [77.088, 1012.45, 1049.92, 0.423, 0.172]  # with intern bin 2x2 and flip LR

# --- --------  SOME GUI parameters
ID_FILE_OPEN = 101
ID_FILE_EXIT = 103

try:
    modifiedTime = os.path.getmtime(os.path.join(LaueToolsProjectFolder, "LaueToolsGUI.py"))
    DAY, MONTH, YEAR = (datetime.datetime.fromtimestamp(modifiedTime).strftime("%d %B %Y").split())
except (FileExistsError, FileNotFoundError):
    DAY, MONTH, YEAR = "FromDistribution", "", "2021"


# --- ------------  MAIN GUI WINDOW
class LaueToolsGUImainframe(wx.Frame):
    """
    class of the main window of LaueTools GUI
    """
    def __init__(self, parent, _id, title, filename="", consolefile="defaultLTlogfile.log",
            projectfolder=None):

        wx.Frame.__init__(self, parent, _id, title, size=(700, 500))
        panel = wx.Panel(self, -1)

        # self.SetIcon(
        #     wx.Icon(os.path.join(projectfolder, "icons", "transmissionLaue_fcc_111.png"),
        #         wx.BITMAP_TYPE_PNG))

        self.filename = filename
        self.dirname = projectfolder
        self.filenamepklist = filename
        self.dirnamepklist = projectfolder
        self.imgfilename = None
        self.imgdirname = projectfolder
        #print("self.dirname", self.dirname)
        self.writefolder = None
        self.resetwf = False
        self.consolefile = consolefile

        # --- begin of general layout -------------------
        self.CreateExteriorWindowComponents()

        # peak list editor
        self.control = wx.TextCtrl(panel, -1, "", size=(700, 400),
                        style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL)
        self.console = wx.TextCtrl(panel, -1, "", size=(700, 100),
                        style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.VSCROLL)

        sizer = wx.FlexGridSizer(cols=1, hgap=6, vgap=6)

        sizer.AddMany([self.control, self.console])
        panel.SetSizer(sizer)
        # ---  end of general layout ------------------

        # default color LUT for image viewer
        self.mapsLUT = "OrRd"

        # Default CCD file image props
        self.CCDLabel = DEFAULT_CCDCAMERA
        self.pixelsize = dict_CCD[self.CCDLabel][1]
        self.framedim = dict_CCD[self.CCDLabel][0]
        # self.framedim =(2671,4008)
        # for VHR camera as seen by XMAS LaueTools and other array readers(reverse order for fit2S)
        self.fliprot = dict_CCD[self.CCDLabel][3]
        self.headeroffset, self.dataformat = dict_CCD[self.CCDLabel][4:6]
        self.file_extension = dict_CCD[self.CCDLabel][7]
        self.saturationvalue = dict_CCD[self.CCDLabel][2]

        self.defaultParam = DEFAULT_DETECTORPARAMETERS

        self.detectordiameter = 165.0
        self.kf_direction = "Z>0"
        self.kf_direction_from_file = None

        self.SimulPrefixName = "mySimul_"
        self.indexsimulation = 0

        self.Current_Data = None
        self.DataPlot_filename = None
        self.PeakListDatFileName = None
        self.Current_peak_data = None
        self.data_theta = None
        self.data_gnomonx = None
        self.data_gnomony = None
        self.data_pixX, self.data_pixY = None, None

        self.select_theta, self.select_chi, self.select_I = None, None, None
        self.select_pixX, self.select_pixY = None, None
        self.select_dataXY = None
        self.select_gnomonx, self.select_gnomony = None, None
        self.data_pixXY = None
        self.data = None
        self.statsresidues = None
        self.TwicethetaChi_solution = None

        self.Matrix_Store = []
        self.mat_store_ind = 0
        self.list_of_cliques = None

        self.indexation_parameters = {}
        self.DataSet = None
        self.ClassicalIndexation_Tabledist = None
        self.key_material = None
        self.emax = None

        self.picky = None
        self.last_orientmatrix_fromindexation = {}  # dict of orientation matrix found
        self.last_Bmatrix_fromindexation = {}  # dict of Bmatrix matrix found
        self.last_epsil_fromindexation = {}  # dict of deviatoric strain found
        self.indexed_spots = {}  # dictionary of exp spots which are indexed
        self.current_processedgrain = 0  # index corresponding to the grain found in data
        self.current_exp_spot_index_list = []

        self.UBmatrix_toCheck = None

        # loading dictionaries
        self.dict_calib = dict_calib  # calibration parameter

        self.dict_Materials = dict_Materials  # Materials or compounds
        self.dict_Extinc = dict_Extinc
        self.dict_Transforms = dict_Transforms  # deformation dict
        self.dict_Vect = dict_Vect  # initial orientation and strain matrix(UB matrix)
        self.dict_Rot = dict_Rot  # additional matrix of rotation applied in left of UB
        self.dict_Eul = dict_Eul  # additional matrix of rotation enter as 3 angles / elemntary rotations applied in left of UB

        # make a list of safe functions
        self.safe_list = ["math", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "degrees", "e", "exp", "fabs", "floor", "fmod", "frexp", "hypot", "ldexp", "log", "log10", "modf", "pi", "pow", "radians", "sin", "sinh", "sqrt", "tan", "tanh"]
        # use the list to filter the local namespace
        self.safe_dict = dict([(k, locals().get(k, None)) for k in self.safe_list])
        # add any needed builtins back in.
        self.safe_dict["abs"] = abs

    def CreateExteriorWindowComponents(self):
        """
        Create "exterior" window components, such as menu and status
            bar.
        """
        self.CreateMenu()
        self.CreateStatusBar()
        self.SetTitle()

    def CreateMenu(self):
        """
        create menu and function calls
        """
        ReadImageMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (wx.ID_ANY, "&Set CCD File Parameters", "CCD ImageFile reading parameters dialog",
                                                                            self.OnSetFileCCDParam),
            (wx.ID_ANY, "&Set Laue Geometry", "Set general CCD position for Laue experiment",
                                                                    self.OnSetLaueDetectorGeometry),
            (None, None, None, None),
            (wx.ID_OPEN, "&Open Image && PeakSearch", "View Image & Peak Search", self.OnOpenImage),
            (None, None, None, None),
            (wx.ID_ANY, "&Detector Calibration", "Detector Calibration from a known reference crystal", self.OnDetectorCalibration),
            (5151, "&Set or Reset Detector Parameters",
                "Open detector parameters board. Set or Reset calibration parameters dialog, compute Laue spots scattering angles", self.recomputeScatteringAngles),
            (None, None, None, None),
            (wx.ID_ANY, "Folder Preferences", "Define where to write files", self.OnPreferences)]:

            if _id is None:
                ReadImageMenu.AppendSeparator()
            else:
                item = ReadImageMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        #         ReadImageMenu.Enable(id=5151, enable=False)

        ManualIndexation_SubMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (wx.ID_ANY, "&2thetaChi", "Selection and recognition Tools in(2theta, chi) coordinates",
                                                                            self.OnPlot_2ThetaChi),
            (wx.ID_ANY, "&Gnomon.", "Selection and recognition Tools in gnomonic plane coordinates",
                                                                                self.OnPlot_Gnomon),
            (wx.ID_ANY, "&CCD Pixel", "Selection and recognition Tools in CCD pixels coordinates",
                                                                            self.OnPlot_Pixels)]:
            if _id is None:
                ManualIndexation_SubMenu.AppendSeparator()
            else:
                item = ManualIndexation_SubMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        IndexationMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (wx.ID_ANY, "&Open Peak List", "Open a file with a peak list: .dat or .cor", self.OnOpenPeakList),
            (wx.ID_ANY, "&Reload Materials", "Update or Load Materials (dict_Materials.dat)",
                                                                            self.OnLoadMaterials),
            (None, None, None, None),
            (wx.ID_ANY, "&Check Orientation",
                "Enter Orientation matrix, Material and check Matching with the current experimental Laue Pattern spots list",
                self.OnCheckOrientationMatrix),
            (None, None, None, None),
            (wx.ID_ANY, "&Find Spots family",
                "Find cliques of spots for which all mutual angular distances are found in reference crsytal LUT",
                self.OnCliquesFinding),
            (None, None, None, None),
            (wx.ID_ANY, "&Automatic Indexation",
                "Indexation by recognition of mutual lattice planes normals angles calculated from pairs of Laue spots",
                self.OnClassicalIndexation),
            (5050, "&Image Matching", "Spots position matching with Databank(in Hough Space)",
                self.OnHoughMatching),
            (None, None, None, None),
            (wx.ID_ANY, "&Manual Indexation", "Plot Data and Manual indexation (Angles Recognition)",
                ManualIndexation_SubMenu),
            (None, None, None, None),
            (wx.ID_SAVE, "&Save Indexation Results",
                                    "Save indexation results viewed in control in *.idx file", self.OnSaveIndexationResultsfile),
            (wx.ID_SAVEAS, "Save &As Indexation Results",
                                    "Save indexation results viewed in control in *.idx file",
                                    self.OnFileResultsSaveAs),
            (wx.ID_ANY, "Save Non &Indexed .cor file",
                                "Save non indexed spots in a .cor file", self.SaveNonIndexedSpots)]:
            if _id is None:
                IndexationMenu.AppendSeparator()
            elif isinstance(handler, (wx.Menu, wx.Object)):
                if WXPYTHON4:
                    item = IndexationMenu.AppendSubMenu(handler, label, helpText)
                else:
                    item = IndexationMenu.AppendMenu(
                        wx.ID_ANY, label, handler, helpText)
            else:
                item = IndexationMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        IndexationMenu.Enable(id=5050, enable=False)

        SimulationMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (wx.ID_ANY, "&PolyGrains Simulation", "Polycrystal selection & simulation",
                                                                self.onParametricLaueSimulator),
            (None, None, None, None),
            (wx.ID_ANY, "&Edit Matrix", "Edit or Load Orientation Matrix", self.OnEditMatrix),
            (wx.ID_ANY, "&Edit UB, B, Crystal",
                                "Edit B or UB Matrix and unit cell structure and extinctions",
                                self.OnEditB0Matrix)]:
            if _id is None:
                SimulationMenu.AppendSeparator()
            else:
                item = SimulationMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        HelpMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (5051, "&Tutorial", "Tutorial", self.OnTutorial),
            (wx.ID_ANY, "&HTML Documentation", "Documentation", self.OnDocumentationhtml),
            (wx.ID_ANY, "&PDF Documentation", "Documentation", self.OnDocumentationpdf),
            (None, None, None, None),
            (wx.ID_ANY, "&About", "Information about this program", self.OnAbout)]:
            if _id is None:
                HelpMenu.AppendSeparator()
            else:
                item = HelpMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        HelpMenu.Enable(id=5051, enable=False)

        QuitMenu = wx.Menu()
        quitsubmenu = QuitMenu.Append(wx.ID_ANY, "Quit", "Exit Program")
        self.Bind(wx.EVT_MENU, self.OnExit, quitsubmenu)

        menuBar = wx.MenuBar()
        menuBar.Append(ReadImageMenu, "&ReadImage")  # Add the ReadImageMenu to the MenuBar
        menuBar.Append(IndexationMenu, "&Indexation")
        # menuBar.Append(RefinementMenu, '&Refinement')
        # menuBar.Append(FileSequenceMenu, '&Sequence')
        menuBar.Append(SimulationMenu, "&Simulation")
        # menuBar.Append(DefaultParametersMenu, '&Parameters')
        menuBar.Append(HelpMenu, "&Help")
        menuBar.Append(QuitMenu, "&Quit")
        self.SetMenuBar(menuBar)  # Add the menuBar to the Frame

    def SetTitle(self):
        """
        title of main lauetools GUI window
        """
        # LaueToolsGUImainframe.SetTitle overrides wx.Frame.SetTitle, so we have to
        # call it using super:
        super(LaueToolsGUImainframe, self).SetTitle(
            "LaueToolsGUI    Simulation & Indexation Program         %s %s" % (self.filename,__version__))

    # --- -------- Main FUNCTIONS called from MENU and submenus
    def OnOpenImage(self, _):
        """
        ask user to select folder and image file
        and launch the peak search board(class PeakSearchFrame)
        """
        if askUserForFilename(self, filetype = 'image', style=wx.OPEN, **self.defaultFileDialogOptionsImage()):

            nbparts = len(self.imgfilename.split("."))
            if nbparts == 2:
                _, file_extension = self.imgfilename.rsplit(".", 1)
            elif nbparts == 3:
                _, ext1, ext2 = self.imgfilename.rsplit(".", 2)
                file_extension = ext1 + "." + ext2

            if file_extension in list_CCD_file_extensions:

                detectedCCDlabel = autoDetectDetectorType(file_extension)
                if detectedCCDlabel is not None:
                    self.CCDLabel = autoDetectDetectorType(file_extension)
                DPBoard = CCDFileParameters(self, -1, "CCD File Parameters Board", self.CCDLabel)
                DPBoard.ShowModal()
                DPBoard.Destroy()

                initialParameter = {}
                initialParameter["title"] = "peaksearch Board"
                initialParameter["imagefilename"] = self.imgfilename
                initialParameter["dirname"] = self.imgdirname
                initialParameter["mapsLUT"] = "OrRd"
                initialParameter["CCDLabel"] = self.CCDLabel
                initialParameter["writefolder"] = self.writefolder
                initialParameter["stackimageindex"] = -1
                initialParameter["stackedimages"] = False
                initialParameter["Nbstackedimages"] = 0

                if self.CCDLabel in ("EIGER_4Mstack",):
                    initialParameter["stackedimages"] = True
                    initialParameter["stackimageindex"] = 0
                    initialParameter["Nbstackedimages"] = 20

                peaksearchframe = MainPeakSearchFrame(self, -1, initialParameter,
                                                                        "peaksearch Board")
                peaksearchframe.Show(True)


    def OnOpenPeakList(self, _):
        """ Open peak list either .dat or .cor file and initinalize peak list
        for further use (indexation)
        """
        # read peak list and detector calibration parameters
        OpenPeakList(self)

        # ---------------------------------------------
        self.filenamepklist = self.DataPlot_filename

        self.set_gnomonic_data()
        # ---------------------------------------------
        self.init_DataSet()

        # create DB spots(dictionary)
        self.CreateSpotDB()
        self.setAllDataToIndex_Dict()
        self.display_corfile_contents()

        self.current_processedgrain = 0
        self.last_orientmatrix_fromindexation = {}
        self.last_Bmatrix_fromindexation = {}
        self.last_epsil_fromindexation = {}

    def OnLoadMaterials(self, _):
        """Load an ASCII file with Materials properties
        """
        wcd = "All files(*)|*|dict_Materials files(*.dat)|*.mat"
        _dir = os.getcwd()
        open_dlg = wx.FileDialog(self,
                                    message="Choose a file",
                                    defaultDir=_dir,
                                    defaultFile="",
                                    wildcard=wcd,
                                    style=wx.OPEN)

        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()

            try:
                self.dict_Materials = readDict(path)

                # dict_Materials = self.dict_Materials

            except IOError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except UnicodeDecodeError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except ValueError as error:
                dlg = wx.MessageDialog(self, "Error opening file: Something went wrong "
                                                "when parsing materials line\n" + str(error))
                dlg.ShowModal()

        open_dlg.Destroy()

    def recomputeScatteringAngles(self, _):
        """ update scattering angles 2theta chi from user entered value from Launch_DetectorParamBoard"""
        if self.DataPlot_filename is None:
            wx.MessageBox("You must first open a peaks list like in .dat or .cor file")
            return

        prefix, file_extension = self.DataPlot_filename.rsplit(".", 1)

        #os.chdir(self.dirname)

        fullpathfile = os.path.join(self.dirname, self.DataPlot_filename)

        print("self.defaultParam before ", self.defaultParam)

        Launch_DetectorParamBoard(self)

        print("self.defaultParam after", self.defaultParam)

        (twicetheta, chi, dataintensity,
        data_x, data_y) = Compute_data2thetachi(fullpathfile, sorting_intensity="yes",
                                                    detectorparams=self.defaultParam,
                                                    pixelsize=self.pixelsize,
                                                    kf_direction=self.kf_direction)

        writefile_cor("update_" + prefix, twicetheta, chi, data_x, data_y,
                                                        dataintensity,
                                                        sortedexit=0,
                                                        param=self.defaultParam + [self.pixelsize],
                                                        initialfilename=self.PeakListDatFileName,
                                                        dirname_output=self.dirname)  # check sortedexit = 0 or 1 to have decreasing intensity sorted data
        print("%s has been created" % ("update_" + prefix + ".cor"))
        print("%s" % str(self.defaultParam))
        self.kf_direction_from_file = self.kf_direction
        # a .cor file is now created
        file_extension = "cor"
        self.DataPlot_filename = "update_" + prefix + "." + file_extension
        # WARNING: it will be read in the next "if" clause

        # for .cor file ------------------------------

        # read peak list and detector calibration parameters
        OpenCorfile(self.DataPlot_filename, self)

        # ---------------------------------------------
        self.filename = self.DataPlot_filename

        self.set_gnomonic_data()
        # ---------------------------------------------
        self.init_DataSet()

        # create DB spots(dictionary)
        self.CreateSpotDB()
        self.setAllDataToIndex_Dict()
        self.display_corfile_contents()

        self.current_processedgrain = 0
        self.last_orientmatrix_fromindexation = {}
        self.last_Bmatrix_fromindexation = {}
        self.last_epsil_fromindexation = {}

    def OnSetFileCCDParam(self, _):
        """Enter manually CCD file params
        Launch Entry dialog
        """
        DPBoard = CCDFileParameters(self, -1, "CCD File Parameters Board", self.CCDLabel)
        DPBoard.ShowModal()
        DPBoard.Destroy()

    def OnFileResultsSaveAs(self, evt):
        """ save indexed spots data as .res file"""
        dlg = wx.TextEntryDialog(self, "Enter Indexation filename(*.res):",
                                        "Indexation Results Filename Entry")

        if dlg.ShowModal() == wx.ID_OK:
            filename = str(dlg.GetValue())

            self.OnSaveIndexationResultsfile(evt, filename=str(filename))

        dlg.Destroy()

    def OnSetLaueDetectorGeometry(self, _):
        """ launch board to set Laue Detection Geometry"""
        LaueGeomBoard = SetGeneralLaueGeometry(self, -1, "Select Laue Geometry")
        LaueGeomBoard.ShowModal()
        LaueGeomBoard.Destroy()

    def OnPreferences(self, _):
        """
        call the board to define destination folder to  write files
        """
        PB = PreferencesBoard(self, -1, "Writing Results Folder Preferences Board")
        PB.ShowModal()
        PB.Destroy()

        print("New write folder", self.writefolder)

    # --- ------------ Indexation Functions
    def OnCheckOrientationMatrix(self, _):
        r"""
        Check if user input matrix, material parameters can produce a Laue pattern that matches
        the current experimental list of spots
        """
        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print("len(self.current_exp_spot_index_list)", len(self.current_exp_spot_index_list))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (self.data_pixX[self.current_exp_spot_index_list],
                            self.data_pixY[self.current_exp_spot_index_list])
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.data = (2 * self.select_theta, self.select_chi, self.select_I, self.DataPlot_filename)

        if not self.current_exp_spot_index_list:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #print("AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters)
        self.indexation_parameters["writefolder"] = self.writefolder
        self.indexation_parameters["dirname"] = self.dirname
        self.indexation_parameters["kf_direction"] = self.kf_direction
        self.indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        self.indexation_parameters["dict_Materials"] = self.dict_Materials
        self.indexation_parameters["DataToIndex"] = {}
        self.indexation_parameters["DataToIndex"]["data_theta"] = self.select_theta
        self.indexation_parameters["DataToIndex"]["data_chi"] = self.select_chi
        self.indexation_parameters["DataToIndex"]["data_I"] = self.select_I
        self.indexation_parameters["DataToIndex"]["dataXY"] = self.select_dataXY
        self.indexation_parameters["DataToIndex"]["data_X"] = self.select_pixX
        self.indexation_parameters["DataToIndex"]["data_Y"] = self.select_pixY
        self.indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"] = None

        # print("self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
        #     self.indexation_parameters["DataToIndex"]["data_theta"])

        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters["current_processedgrain"] = self.current_processedgrain
        self.indexation_parameters["detectordiameter"] = self.detectordiameter
        self.indexation_parameters["pixelsize"] = self.pixelsize
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectorparameters"] = self.defaultParam
        self.indexation_parameters["CCDLabel"] = self.CCDLabel
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectordistance"] = self.defaultParam[0]
        self.indexation_parameters["flipyaxis"] = True  # CCD MAr 165
        self.indexation_parameters["Filename"] = self.DataPlot_filename
        self.indexation_parameters["CCDcalib"] = self.defaultParam
        self.indexation_parameters["framedim"] = self.framedim

        self.indexation_parameters["mainAppframe"] = self

        # update DataSetObject
        self.DataSet.dim = self.indexation_parameters["dim"]
        self.DataSet.pixelsize = self.indexation_parameters["pixelsize"]
        self.DataSet.detectordiameter = self.indexation_parameters["detectordiameter"]
        self.DataSet.kf_direction = self.indexation_parameters["kf_direction"]

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = dict_Rot
        StorageDict["dict_Materials"] = self.dict_Materials
        # --------------end of common part before indexing------------------------
        self.EnterMatrix(1)
        if not self.Enterkey_material():
            return
        self.EnterEnergyMax()

        self.DataSet.pixelsize = self.pixelsize
        self.DataSet.detectordiameter = self.detectordiameter
        self.DataSet.dim = self.framedim
        self.DataSet.detectordistance = self.defaultParam[0]
        #         self.DataSet.kf_direction = self.kf_direction

        self.indexation_parameters["paramsimul"] = []
        self.indexation_parameters["bestmatrices"] = []
        self.indexation_parameters["TwicethetaChi_solutions"] = []

        self.statsresidues = []
        for orientmatrix in self.UBmatrix_toCheck:
            # only orientmatrix, self.key_material are used ----------------------
            vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
            # old definition of grain
            grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]
            # ------------------------------------------------------------------

            #                 print "self.indexation_parameters", self.indexation_parameters
            TwicethetaChi = SimulateResult(grain, 5, self.emax, self.indexation_parameters,
                                                                ResolutionAngstrom=False,
                                                                fastcompute=1,
                                                                dictmaterials=self.dict_Materials)
            self.TwicethetaChi_solution = TwicethetaChi
            paramsimul = (grain, 5, self.emax)

            self.indexation_parameters["paramsimul"].append(paramsimul)
            self.indexation_parameters["bestmatrices"].append(orientmatrix)
            self.indexation_parameters["TwicethetaChi_solutions"].append(
                self.TwicethetaChi_solution)

            self.statsresidues.append(self.computeAngularMatching(orientmatrix))

        self.PlotandRefineSolution()

    def OnClassicalIndexation(self, _):
        """
        Call the ClassicalIndexationBoard Class with current non indexed spots list

        see Autoindexation.py module
        """

        if self.data_theta is None:
            self.OpenDefaultData()

        #print('self.pixelsize in OnClassicalIndexation for %s'%self.DataPlot_filename, self.pixelsize)
        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        #print("len(self.current_exp_spot_index_list)", len(self.current_exp_spot_index_list))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (self.data_pixX[self.current_exp_spot_index_list],
                                self.data_pixY[self.current_exp_spot_index_list])
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.data = (2 * self.select_theta, self.select_chi, self.select_I, self.DataPlot_filename,)

        if not self.current_exp_spot_index_list:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        #print("AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters)
        self.indexation_parameters["writefolder"] = self.writefolder
        self.indexation_parameters["dirname"] = self.dirname
        self.indexation_parameters["kf_direction"] = self.kf_direction
        self.indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        self.indexation_parameters["dict_Materials"] = self.dict_Materials
        self.indexation_parameters["DataToIndex"] = {}
        self.indexation_parameters["DataToIndex"]["data_theta"] = self.select_theta
        self.indexation_parameters["DataToIndex"]["data_chi"] = self.select_chi
        self.indexation_parameters["DataToIndex"]["data_I"] = self.select_I
        self.indexation_parameters["DataToIndex"]["dataXY"] = self.select_dataXY
        self.indexation_parameters["DataToIndex"]["data_X"] = self.select_pixX
        self.indexation_parameters["DataToIndex"]["data_Y"] = self.select_pixY
        self.indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"] = None

        # print( "self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
        #                             self.indexation_parameters["DataToIndex"]["data_theta"])
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters["current_processedgrain"] = self.current_processedgrain
        self.indexation_parameters["detectordiameter"] = self.detectordiameter
        self.indexation_parameters["pixelsize"] = self.pixelsize
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectorparameters"] = self.defaultParam
        self.indexation_parameters["CCDLabel"] = self.CCDLabel
        self.indexation_parameters["mainAppframe"] = self
        self.indexation_parameters["Cliques"] = self.list_of_cliques

        # update DataSetObject
        self.DataSet.dim = self.indexation_parameters["dim"]
        self.DataSet.pixelsize = self.indexation_parameters["pixelsize"]
        self.DataSet.detectordiameter = self.indexation_parameters["detectordiameter"]
        self.DataSet.kf_direction = self.indexation_parameters["kf_direction"]

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = dict_Rot
        StorageDict["dict_Materials"] = dict_Materials

        titleboard = ("Spots interdistance Screening Indexation Board  file: %s" % self.DataPlot_filename)


        #print('self.indexation_parameters', self.indexation_parameters)
        # Launch Automatic brute force indexation procedure (angular distance recognition)
        DistanceScreeningIndexationBoard(self, -1, self.indexation_parameters, titleboard,
                                        StorageDict=StorageDict, DataSetObject=self.DataSet)

    def OnHoughMatching(self, _):
        """ not implemented """
        dialog = wx.MessageDialog(self, "Not yet implemented \n" "in wxPython",
                                                    "Classical Indexation Method", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()

    def OnRadialRecognition(self, _):
        """ not implemented """
        dialog = wx.MessageDialog(self, "Not yet implemented \n" "in wxPython",
                                                            "Radial Recognition Method", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()

    def OnPlot_2ThetaChi(self, _):
        """
        Plot 2theta-chi representation of data peaks list for manual indexation
        """

        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        #print("len(self.current_exp_spot_index_list)", len(self.current_exp_spot_index_list))

        # nb_exp_spots_data = len(self.data_theta)
        # index_to_select = np.take(self.current_exp_spot_index_list, np.arange(nb_exp_spots_data))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (self.data_pixX[self.current_exp_spot_index_list],
                                    self.data_pixY[self.current_exp_spot_index_list])
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.data = (2 * self.select_theta, self.select_chi, self.select_I, self.DataPlot_filename)

        if not self.current_exp_spot_index_list:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #print("AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters)
        self.indexation_parameters["writefolder"] = self.writefolder
        self.indexation_parameters["dirname"] = self.dirname
        self.indexation_parameters["kf_direction"] = self.kf_direction
        self.indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        self.indexation_parameters["dict_Materials"] = self.dict_Materials
        self.indexation_parameters["DataToIndex"] = {}
        self.indexation_parameters["DataToIndex"]["data_theta"] = self.select_theta
        self.indexation_parameters["DataToIndex"]["data_chi"] = self.select_chi
        self.indexation_parameters["DataToIndex"]["data_I"] = self.select_I
        self.indexation_parameters["DataToIndex"]["dataXY"] = self.select_dataXY
        self.indexation_parameters["DataToIndex"]["data_X"] = self.select_pixX
        self.indexation_parameters["DataToIndex"]["data_Y"] = self.select_pixY
        self.indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"] = None

        # print("self.indexation_parameters['DataToIndex']['data_theta']",
        #     self.indexation_parameters["DataToIndex"]["data_theta"])
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters["current_processedgrain"] = self.current_processedgrain
        self.indexation_parameters["detectordiameter"] = self.detectordiameter
        self.indexation_parameters["pixelsize"] = self.pixelsize
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectorparameters"] = self.defaultParam
        self.indexation_parameters["CCDLabel"] = self.CCDLabel
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectordistance"] = self.defaultParam[0]
        self.indexation_parameters["flipyaxis"] = True  # CCD MAr 165
        self.indexation_parameters["Filename"] = self.DataPlot_filename
        self.indexation_parameters["CCDcalib"] = self.defaultParam
        self.indexation_parameters["framedim"] = self.framedim

        self.indexation_parameters["mainAppframe"] = self

        # update DataSetObject
        self.DataSet.dim = self.indexation_parameters["dim"]
        self.DataSet.pixelsize = self.indexation_parameters["pixelsize"]
        self.DataSet.detectordiameter = self.indexation_parameters["detectordiameter"]
        self.DataSet.kf_direction = self.indexation_parameters["kf_direction"]

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = dict_Rot
        StorageDict["dict_Materials"] = dict_Materials

        # Open manual indextion Board
        self.picky = ManualIndexFrame(self, -1, self.DataPlot_filename, data=self.data,
                                            kf_direction=self.kf_direction,
                                            Params_to_simulPattern=None,
                                            indexation_parameters=self.indexation_parameters,
                                            StorageDict=StorageDict,
                                            DataSetObject=self.DataSet)

        self.picky.Show(True)

    def OnPlot_Gnomon(self, _):
        """
        Plot gnomonic representation of data peaks list for manual indexation
        """

        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print("len(self.current_exp_spot_index_list)", len(self.current_exp_spot_index_list))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (self.data_pixX[self.current_exp_spot_index_list],
                            self.data_pixY[self.current_exp_spot_index_list])
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.select_gnomonx = self.data_gnomonx[self.current_exp_spot_index_list]
        self.select_gnomony = self.data_gnomony[self.current_exp_spot_index_list]

        self.data = (2 * self.select_theta, self.select_chi, self.select_I, self.DataPlot_filename)

        if not self.current_exp_spot_index_list:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        print("AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters)
        self.indexation_parameters["writefolder"] = self.writefolder
        self.indexation_parameters["dirname"] = self.dirname
        self.indexation_parameters["kf_direction"] = self.kf_direction
        self.indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        self.indexation_parameters["dict_Materials"] = self.dict_Materials
        self.indexation_parameters["DataToIndex"] = {}
        self.indexation_parameters["DataToIndex"]["data_theta"] = self.select_theta
        self.indexation_parameters["DataToIndex"]["data_chi"] = self.select_chi
        self.indexation_parameters["DataToIndex"]["data_I"] = self.select_I
        self.indexation_parameters["DataToIndex"]["dataXY"] = self.select_dataXY
        self.indexation_parameters["DataToIndex"]["data_X"] = self.select_pixX
        self.indexation_parameters["DataToIndex"]["data_Y"] = self.select_pixY
        self.indexation_parameters["DataToIndex"]["data_gnomonX"] = self.select_gnomonx
        self.indexation_parameters["DataToIndex"]["data_gnomonY"] = self.select_gnomony
        self.indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"] = None

        # print("self.indexation_parameters['DataToIndex']['data_theta']",
        #                                     self.indexation_parameters["DataToIndex"]["data_theta"])
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters["current_processedgrain"] = self.current_processedgrain
        self.indexation_parameters["detectordiameter"] = self.detectordiameter
        self.indexation_parameters["pixelsize"] = self.pixelsize
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectorparameters"] = self.defaultParam
        self.indexation_parameters["CCDLabel"] = self.CCDLabel
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectordistance"] = self.defaultParam[0]
        self.indexation_parameters["flipyaxis"] = True  # CCD MAr 165
        self.indexation_parameters["Filename"] = self.DataPlot_filename
        self.indexation_parameters["CCDcalib"] = self.defaultParam
        self.indexation_parameters["framedim"] = self.framedim

        self.indexation_parameters["mainAppframe"] = self

        # update DataSetObject
        self.DataSet.dim = self.indexation_parameters["dim"]
        self.DataSet.pixelsize = self.indexation_parameters["pixelsize"]
        self.DataSet.detectordiameter = self.indexation_parameters["detectordiameter"]
        self.DataSet.kf_direction = self.indexation_parameters["kf_direction"]

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = dict_Rot
        StorageDict["dict_Materials"] = dict_Materials

        # Open manual indextion Board
        self.picky = ManualIndexFrame(self, -1, self.DataPlot_filename, data=self.data,
                                        kf_direction=self.kf_direction,
                                        datatype="gnomon",
                                        Params_to_simulPattern=None,
                                        indexation_parameters=self.indexation_parameters,
                                        StorageDict=StorageDict,
                                        DataSetObject=self.DataSet)

        self.picky.Show(True)

    def OnPlot_Pixels(self, _):
        """
        Plot detector X,Y pixel representation of data peaks list for manual indexation
        """
        # select spots accroding to previous indexation
        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        #print("len(self.current_exp_spot_index_list)", len(self.current_exp_spot_index_list))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (self.data_pixX[self.current_exp_spot_index_list],
                                self.data_pixY[self.current_exp_spot_index_list])
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.select_gnomonx = self.data_gnomonx[self.current_exp_spot_index_list]
        self.select_gnomony = self.data_gnomony[self.current_exp_spot_index_list]

        self.data = (2 * self.select_theta, self.select_chi, self.select_I, self.DataPlot_filename)

        if not self.current_exp_spot_index_list:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        print("AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters)
        self.indexation_parameters["writefolder"] = self.writefolder
        self.indexation_parameters["dirname"] = self.dirname
        self.indexation_parameters["kf_direction"] = self.kf_direction
        self.indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        self.indexation_parameters["dict_Materials"] = self.dict_Materials
        self.indexation_parameters["DataToIndex"] = {}
        self.indexation_parameters["DataToIndex"]["data_theta"] = self.select_theta
        self.indexation_parameters["DataToIndex"]["data_chi"] = self.select_chi
        self.indexation_parameters["DataToIndex"]["data_I"] = self.select_I
        self.indexation_parameters["DataToIndex"]["dataXY"] = self.select_dataXY
        self.indexation_parameters["DataToIndex"]["data_X"] = self.select_pixX
        self.indexation_parameters["DataToIndex"]["data_Y"] = self.select_pixY
        #         self.indexation_parameters['DataToIndex']['data_gnomonX'] = self.select_gnomonx
        #         self.indexation_parameters['DataToIndex']['data_gnomonY'] = self.select_gnomony
        self.indexation_parameters["DataToIndex"]["current_exp_spot_index_list"] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"]["ClassicalIndexation_Tabledist"] = None

        # print("self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
        #     self.indexation_parameters["DataToIndex"]["data_theta"])
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters["current_processedgrain"] = self.current_processedgrain
        self.indexation_parameters["detectordiameter"] = self.detectordiameter
        self.indexation_parameters["pixelsize"] = self.pixelsize
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectorparameters"] = self.defaultParam
        self.indexation_parameters["CCDLabel"] = self.CCDLabel
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectordistance"] = self.defaultParam[0]
        self.indexation_parameters["flipyaxis"] = True  # CCD MAr 165
        self.indexation_parameters["Filename"] = self.DataPlot_filename
        self.indexation_parameters["CCDcalib"] = self.defaultParam
        self.indexation_parameters["framedim"] = self.framedim

        self.indexation_parameters["mainAppframe"] = self

        # update DataSetObject
        self.DataSet.dim = self.indexation_parameters["dim"]
        self.DataSet.pixelsize = self.indexation_parameters["pixelsize"]
        self.DataSet.detectordiameter = self.indexation_parameters["detectordiameter"]
        self.DataSet.kf_direction = self.indexation_parameters["kf_direction"]

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = dict_Rot
        StorageDict["dict_Materials"] = dict_Materials
        # Open manual indextion Board
        self.picky = ManualIndexFrame(self, -1, self.DataPlot_filename, data=self.data,
                                        kf_direction=self.kf_direction,
                                        datatype="pixels",
                                        Params_to_simulPattern=None,
                                        indexation_parameters=self.indexation_parameters,
                                        StorageDict=StorageDict,
                                        DataSetObject=self.DataSet)

        self.picky.Show(True)

    def OnDetectorCalibration(self, _):
        """
        Method launching Calibration Board
        """
        starting_param = [77, 1000, 1000, 0, 0]

        print("Starting param", starting_param)

        initialParameter = {}
        initialParameter["CCDParam"] = starting_param

        initialParameter["CCDLabel"] = self.CCDLabel
        pixelsize = dict_CCD[self.CCDLabel][1]
        framedim = dict_CCD[self.CCDLabel][0]
        geomoperator = dict_CCD[self.CCDLabel][3]
        initialParameter["detectordiameter"] = max(framedim[0], framedim[1]) * pixelsize * 1.1
        initialParameter["filename"] = 'dat_Ge0001.cor'
        initialParameter["dirname"] = os.path.join(LaueToolsProjectFolder, "Examples", "Ge")
        initialParameter["dict_Materials"] = self.dict_Materials

        print("initialParameter when launching calibration", initialParameter)
        print('OnDetectorCalibration ==> pixelsize', pixelsize)

        file_peaks = os.path.join(initialParameter["dirname"], initialParameter["filename"])

        calibframe = MainCalibrationFrame(self, -1, "Detector Calibration Board",
                                    initialParameter, file_peaks=file_peaks,
                                    pixelsize=pixelsize, dim=framedim,
                                    kf_direction=self.kf_direction, fliprot=geomoperator)
        calibframe.Show(True)

    def OnCliquesFinding(self, _):
        """
        Method launching Cliques Finding  Board
        """
        if not self.ClassicalIndexation_Tabledist:
            print("OnCliquesFinding")

            if self.data_theta is None:
                self.OpenDefaultData()
                self.CreateSpotDB()

            self.current_exp_spot_index_list = (self.getAbsoluteIndices_Non_Indexed_Spots_())

        print('self.indexation_parameters', self.indexation_parameters)
        CliquesFindingBoard(self, -1, "Cliques Finding Board :%s" % self.DataPlot_filename,
                                                    indexation_parameters=self.indexation_parameters)

    def OnEditMatrix(self, _):
        """ launch board to edit/create/load orientation matrix"""
        MatrixEditor = MatrixEditor_Dialog(self, -1, "Create/Read/Save/Load/Convert Orientation Matrix")
        MatrixEditor.Show(True)

    def OnEditB0Matrix(self, _):
        """ launch editor for B0 matrix  (from lattice parameters)"""
        UBMatrixEditor = B0MatrixEditor(self, -1, "UB Matrix Editor and Board")
        UBMatrixEditor.Show(True)

    def EnterMatrix(self, _):
        """
        open matrix entry board in check orientation process
        """
        helptstr = "Enter Matrix elements : \n [[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]]"
        helptstr += " Or list of Matrices"
        dlg = wx.TextEntryDialog(self, helptstr, "Orientation Matrix elements Entry for Matching Check")

        _param = "[[1.,0,0],[0,1,0],[0,0,1]]"
        dlg.SetValue(_param)

        # OR
        dlg.SetToolTipString("please enter UB not UB.B0 (or list of UB's)")

        if dlg.ShowModal() == wx.ID_OK:
            paramraw = str(dlg.GetValue())

            listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", paramraw)
            listelem = []
            for elem in listval:
                try:
                    val = float(elem)
                    listelem.append(val)
                except ValueError:
                    continue

            nbval = len(listelem)

            if (nbval % 9) != 0:
                txt = "Something wrong, I can't read matrix or matrices"
                print(txt)

                wx.MessageBox(txt, "INFO")
                return

            nbmatrices = nbval // 9
            ListMatrices = np.zeros((nbmatrices, 3, 3))
            ind_elem = 0
            for ind_matrix in range(nbmatrices):
                for i in range(3):
                    for j in range(3):
                        floatval = listelem[ind_elem]
                        ListMatrices[ind_matrix][i][j] = floatval
                        ind_elem += 1

            # save in list of orientation matrix
            # default name
            inputmatrixname = "InputMat_"

            for k, mat in enumerate(ListMatrices):
                mname = inputmatrixname + "%d" % k
                dict_Rot[mname] = mat

            print("len dict", len(dict_Rot))

            # update with the first input matrix
            self.UBmatrix_toCheck = ListMatrices

            dlg.Destroy()

            nbmatrices = len(ListMatrices)

            return nbmatrices

    def Enterkey_material(self):
        """ launch key material dialog
        .. note:: used in checkOrientation workflow
        """
        helptstr = "Enter Material, structure or Element Label (example: Cu, UO2,)"

        dlg = wx.TextEntryDialog(self, helptstr, "Material and Crystallographic Structure Entry")

        _param = "Ge"
        flag = True
        dlg.SetValue(_param)
        if dlg.ShowModal() == wx.ID_OK:
            key_material = str(dlg.GetValue())

            # check
            if key_material in dict_Materials:
                self.key_material = key_material
            else:
                txt = "This material label is unknown. Please check typo or Reload Materials dict"
                print(txt)

                wx.MessageBox(txt, "INFO")
                flag = False

            dlg.Destroy()

        return flag

    def EnterEnergyMax(self):
        """ launch energy max dialog
        .. note:: used in checkOrientation workflow
        """
        helptstr = "Enter maximum energy of polychromatic beam"

        dlg = wx.TextEntryDialog(self, helptstr, "Energy Maximum Entry")

        _param = "23"
        dlg.SetValue(_param)
        if dlg.ShowModal() == wx.ID_OK:
            emax = float(dlg.GetValue())
            self.emax = emax

    def computeAngularMatching(self, UBmatrix_toCheck):
        """
        compute matching rate of selected data with current predefined structure and input orientation matrix

        set self.stats_properformat
        """

        ANGLETOLERANCE = 0.5
        RESOLUTIONANGSTROM = False

        AngRes = matchingrate.Angular_residues_np(UBmatrix_toCheck, 2.0 * self.select_theta,
                                                    self.select_chi,
                                                    key_material=self.key_material,
                                                    emax=self.emax,
                                                    ResolutionAngstrom=RESOLUTIONANGSTROM,
                                                    ang_tol=ANGLETOLERANCE,
                                                    detectorparameters=self.indexation_parameters,
                                                    dictmaterials=self.dict_Materials)

        print("AngRes", AngRes)

        if AngRes is None:
            return
        #(allres, resclose, nbclose, nballres, mean_residue, max_residue) = AngRes
        (allres, _, nbclose, nballres, _, _) = AngRes

        stats_properformat = [nbclose, nballres, np.std(allres)]

        return stats_properformat

    def PlotandRefineSolution(self):
        """ from self.statsresidues (results of checkOrientation)
        launch RecognitionResultCheckBox with potential solutions to be selected

        .. note:: used in checkOrientation workflow
        """
        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = dict_Rot
        StorageDict["dict_Materials"] = self.dict_Materials

        # display "statistical" results
        RRCBClassical = RecognitionResultCheckBox(self, -1, "Screening Distances Indexation Solutions",
                                            self.statsresidues,
                                            self.data,
                                            0.5,
                                            0.2,
                                            key_material=self.key_material,
                                            emax=self.emax,
                                            ResolutionAngstrom=False,
                                            kf_direction=self.kf_direction,
                                            datatype="2thetachi",
                                            data_2thetachi=(2.0 * self.select_theta, self.select_chi),
                                            data_XY=self.select_dataXY,
                                            CCDdetectorparameters=self.indexation_parameters,
                                            IndexationParameters=self.indexation_parameters,
                                            StorageDict=StorageDict,
                                            mainframe="billframerc",  # self.mainframe
                                            DataSetObject=self.DataSet)

        RRCBClassical.Show(True)

    def writeResFileSummary(self, corfilename=None, dirname=None):
        """
        write .res file
        containing all the exp. spots with their properties
        (indexed (grain index) or not (-1)), hkl energy etc.
        """

        if corfilename is not None:
            outputfilename = corfilename.rsplit(".", 1)[0] + ".res"
        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)

        print("Saving Summary file: %s" % outputfilename)

        Data = self.getSummaryallData()

        (spotindex, grain_index, tth, chi, posX, posY, intensity, H, K, L, Energy,) = Data.T

        Columns = [spotindex, grain_index, tth, chi, posX, posY, intensity, H, K, L, Energy,]

        nbspots = len(spotindex)

        datatooutput = np.transpose(np.array(Columns))
        datatooutput = np.round(datatooutput, decimals=7)

        header = "# Spots Summary of: %s\n" % (self.filename)
        header += "# File created at %s with indexingSpotsSet.py\n" % (time.asctime())
        header += "# Number of spots: %d\n" % nbspots

        header += ("#spot_index grain_index 2theta Chi Xexp Yexp intensity h k l Energy\n")
        outputfile = open(outputfilename, "w")

        outputfile.write(header)

        np.savetxt(outputfile, datatooutput, fmt="%.7f")

        print("self.indexedgrains", self.indexedgrains)
        for grain_index in self.indexedgrains:
            outputfile.write("#grainIndex\n")
            outputfile.write("G_%d\n" % grain_index)
            outputfile.write("#Element\n")
            key_material = self.dict_indexedgrains_material[grain_index]
            outputfile.write("%s\n" % key_material)

            nbspotsindexed, MatchRate = self.dict_grain_matching_rate[grain_index][:2]
            outputfile.write("#MatchingRate\n")
            outputfile.write("%.1f\n" % MatchRate)
            outputfile.write("#Nb indexed Spots\n")
            outputfile.write("%.d\n" % nbspotsindexed)
            outputfile.write("#UB matrix in q= (UB) B0 G*\n")
            #            outputfile.write(str(self.UBB0mat) + '\n')
            outputfile.write(str(self.dict_grain_matrix[grain_index]) + "\n")
            outputfile.write("#B0 matrix (starting unit cell) in q= UB (B0) G*\n")
            latticeparams = dict_Materials[key_material][1]
            B0matrix = calc_B_RR(latticeparams)
            outputfile.write(str(B0matrix) + "\n")
            outputfile.write("#deviatoric strain (10-3 unit)\n")
            outputfile.write(
                str(self.dict_grain_devstrain[grain_index] * 1000.0) + "\n")

        addCCDparams = 0
        if addCCDparams:
            outputfile.write("#CCDLabel\n")
            outputfile.write(self.CCDLabel + "\n")
            outputfile.write("#DetectorParameters\n")
            outputfile.write(str(self.CCDcalib) + "\n")
            outputfile.write("#pixelsize\n")
            outputfile.write(str(self.pixelsize) + "\n")
            outputfile.write("#Frame dimensions\n")
            outputfile.write(str(self.framedim) + "\n")

        outputfile.close()

    def OnSaveIndexationResultsfile(self, _, filename=None):
        """
        Save results of indexation and refinement procedures
        .res  :  summary file
        .cor  :  non indexed spots list

        .. note:: export this ASCII write function in readwriteASCII module
        """
        if filename is None:
            filename = self.DataPlot_filename

        wx.MessageBox("Fit results have been written in folder %s" % self.dirname, "INFO")

        self.DataSet.writeFileSummary(filename, self.dirname)

        prefixfilename = filename.rsplit(".", 1)[0]

        filename = prefixfilename + ".cor"

        print("self.DataSet.pixelsize", self.DataSet.pixelsize)
        self.DataSet.writecorFile_unindexedSpots(corfilename=filename, dirname=self.dirname)

    # --- ---------------- Simulation Functions
    def onParametricLaueSimulator(self, _):
        """
        Method launching polycrystal simulation Board
        """
        initialParameters = {}
        initialParameters["CalibrationParameters"] = self.defaultParam
        initialParameters["prefixfilenamesimul"] = "dummy_"
        initialParameters["indexsimulation"] = self.indexsimulation
        initialParameters["pixelsize"] = self.pixelsize
        initialParameters["framedim"] = self.framedim
        initialParameters["dict_Materials"] = self.dict_Materials
        initialParameters["ExperimentalData"] = None
        if self.data_theta is not None:
            initialParameters["ExperimentalData"] = (2 * self.data_theta, self.data_chi,
                                                        self.data_pixX, self.data_pixY,
                                                        self.data_I)

        # print("initialParameters", initialParameters)
        CurrentdialogGrainCreation_cont = parametric_Grain_Dialog3(
                        self, -1, "Polygrains parametric definition for Laue Simulation",
                        initialParameters)

        # self.SelectGrains_parametric = CurrentdialogGrainCreation_cont.SelectGrains
        CurrentdialogGrainCreation_cont.Show(True)

        return True

    # --- ------------- spots database
    def SaveNonIndexedSpots(self, _):
        """ launch user input filedialog to save current non indexed spots  in .cor file with suffix 'unindexed' """
        dlg = wx.TextEntryDialog(self,
                            "Enter new filename (*.cor):", "Peaks List .cor Filename Entry")

        if dlg.ShowModal() == wx.ID_OK:
            filename = str(dlg.GetValue())

            finalfilename = filename
            if filename.endswith(".cor"):
                finalfilename = filename[: -4]

            fullpath = os.path.join(self.dirname, finalfilename)

            self.SaveFileCorNonIndexedSpots(outputfilename=fullpath)

        dlg.Destroy()

    #         self.SaveFileCorNonIndexedSpots(os.path.join(self.dirname, 'nonindexed'))

    def SaveFileCorNonIndexedSpots(self, outputfilename=None):
        """ save current non indexed spots  in .cor file with suffix 'unindexed' """
        if outputfilename is None:
            pre = self.DataPlot_filename.strip(".")[0]
            outputfilename = pre + "nonindexed"

        # get data
        current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()
        print("current_exp_spot_index_list", current_exp_spot_index_list)

        #         nb_exp_spots_data = len(self.data_theta)
        #
        #         print "self.data_theta", self.data_theta
        #         print "len(self.data_theta)", len(self.data_theta)
        #
        #         index_to_select = np.take(current_exp_spot_index_list,
        #                                       np.arange(nb_exp_spots_data))

        Twicetheta = 2.0 * self.data_theta[current_exp_spot_index_list]
        Chi = self.data_chi[current_exp_spot_index_list]
        dataintensity = self.data_I[current_exp_spot_index_list]
        posx = self.data_pixX[current_exp_spot_index_list]
        posy = self.data_pixY[current_exp_spot_index_list]

        # comment
        strgrains = ["Remaining Non indexed spots of %s" % self.DataPlot_filename]

        writefile_cor(outputfilename, Twicetheta, Chi, posx, posy,
                                                        dataintensity,
                                                        param=self.defaultParam + [self.pixelsize],
                                                        initialfilename=self.DataPlot_filename,
                                                        comments=strgrains)

    def setAllDataToIndex_Dict(self):
        """  set dictionnary self.indexation_parameters
        """
        self.indexation_parameters = {}
        self.indexation_parameters["AllDataToIndex"] = {}
        self.indexation_parameters["AllDataToIndex"]["data_theta"] = self.data_theta
        self.indexation_parameters["AllDataToIndex"]["data_chi"] = self.data_chi
        self.indexation_parameters["AllDataToIndex"]["data_pixX"] = self.data_pixX
        self.indexation_parameters["AllDataToIndex"]["data_pixY"] = self.data_pixY
        self.indexation_parameters["AllDataToIndex"]["data_I"] = self.data_I
        self.indexation_parameters["AllDataToIndex"]["data_gnomonX"] = self.data_gnomonx
        self.indexation_parameters["AllDataToIndex"]["data_gnomonY"] = self.data_gnomony
        self.indexation_parameters["AllDataToIndex"]["data_XY"] = (self.data_pixX,
                                                                    self.data_pixY)
        self.indexation_parameters["AllDataToIndex"]["data_2thetachi"] = (2 * self.data_theta,
                                                                            self.data_chi)

        nbspotstoindex = len(self.data_theta)
        self.indexation_parameters["AllDataToIndex"]["NbSpotsToindex"] = nbspotstoindex
        self.indexation_parameters["AllDataToIndex"]["IndexedFlag"] = np.zeros(
                                                                        nbspotstoindex)
        self.indexation_parameters["AllDataToIndex"]["absolutespotindex"] = np.array(
                                            np.arange(nbspotstoindex), dtype=np.int)

    def OpenDefaultData(self):
        """
        Open default data for quick test if user has not yet loaded some data
        """
        DEFAULTFILE = "defaultGe0001.cor"
        defaultdatafile = os.path.join(LaueToolsProjectFolder, "Examples", "Ge", DEFAULTFILE)

        #print("self.detectordiameter in OpenDefaultData()", self.detectordiameter)

        self.dirnamepklist = os.path.split(os.path.abspath(defaultdatafile))[0]
        print('self.dirnamepklist', self.dirnamepklist)
        self.filenamepklist = DEFAULTFILE
        print('self.filenamepklist', self.filenamepklist)
        if os.access(defaultdatafile, os.W_OK):
            self.writefolder = self.dirnamepklist
        else:
            self.writefolder = OSLFGUI.askUserForDirname(self)

        self.DataPlot_filename = DEFAULTFILE

        OpenCorfile(defaultdatafile, self)

        self.set_gnomonic_data()

        # ---------------------------------------------
        self.filenamepklist = self.DataPlot_filename
        self.init_DataSet()

        # create DB spots(dictionary)
        self.CreateSpotDB()
        self.setAllDataToIndex_Dict()
        self.display_corfile_contents()

        self.current_processedgrain = 0
        self.last_orientmatrix_fromindexation = {}
        self.last_Bmatrix_fromindexation = {}
        self.last_epsil_fromindexation = {}

    def init_DataSet(self):
        """ init instance of spots list and properties for indexation """
        # DataSetObject init
        self.DataSet = spotsset()
        # get spots scattering angles,X,Y positions from .cor file
        fullpath = os.path.join(self.dirnamepklist, self.filenamepklist)
        warningflag = self.DataSet.importdatafromfile(fullpath)
        if warningflag:
            wx.MessageBox(warningflag + 'Please set an other Laue geometry if needed from the menu!', 'Info')
        self.DataSet.pixelsize = self.pixelsize

        self.SetTitle()  # Update the window title with the new filename
        # ----------------------------------

    def display_corfile_contents(self):
        """display spots list prpos in main LaueToolsGUI window
        """
        # -----------------------------------------------
        textfile = open(os.path.join(self.dirnamepklist, self.filenamepklist), "r")
        String_in_File_Data = textfile.read()
        self.control.SetValue(String_in_File_Data)
        textfile.close()

    def set_gnomonic_data(self):
        """ compute Gnomonic projection coordinates from self.data_theta * 2, self.data_chi
        set self.data_gnomonx, self.data_gnomony
        """
        dataselected = createselecteddata((self.data_theta * 2, self.data_chi, self.data_I),
                                                np.arange(len(self.data_theta)),
                                                len(self.data_theta))[0]
        self.data_gnomonx, self.data_gnomony = ComputeGnomon_2(dataselected)

    def select_exp_spots(self):
        """
        select spots to be indexed

        set self.current_exp_spot_index_list as array of absolute index in peaks experimental list of non indexed spots
        """
        # select default data for test
        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = (self.getAbsoluteIndices_Non_Indexed_Spots_())

        non_indexed_spots = np.array(self.current_exp_spot_index_list)

        if not non_indexed_spots:
            wx.MessageBox("There are no spots to be indexed now !", "INFO")

    def set_params_manualindexation(self):
        """ init dict of parameters for manual indexation """

        indexation_parameters = {}
        indexation_parameters["kf_direction"] = self.kf_direction
        # duplicate
        indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        indexation_parameters["Filename"] = self.DataPlot_filename

        indexation_parameters["dict_Materials"] = self.dict_Materials

        #         indexation_parameters['data_theta'] = self.data_theta
        #         indexation_parameters['data_chi'] = self.data_chi
        #         indexation_parameters['data_I'] = self.data_I
        indexation_parameters["current_exp_spot_index_list"] = self.current_exp_spot_index_list
        indexation_parameters["ClassicalIndexation_Tabledist"] = None
        indexation_parameters["dict_Rot"] = self.dict_Rot
        indexation_parameters["current_processedgrain"] = self.current_processedgrain
        indexation_parameters["detectordiameter"] = self.detectordiameter
        indexation_parameters["pixelsize"] = self.pixelsize
        # duplicate
        indexation_parameters["dim"] = self.framedim
        indexation_parameters["framedim"] = self.framedim
        # duplicate
        indexation_parameters["detectorparameters"] = self.defaultParam
        indexation_parameters["CCDcalib"] = self.defaultParam
        indexation_parameters["detectordistance"] = self.defaultParam[0]

        indexation_parameters["CCDLabel"] = self.CCDLabel

        indexation_parameters["mainAppframe"] = self

        # StorageDict = {}
        # StorageDict["mat_store_ind"] = self.mat_store_ind
        # StorageDict["Matrix_Store"] = self.Matrix_Store
        # StorageDict["dict_Rot"] = self.dict_Rot
        # StorageDict["dict_Materials"] = self.dict_Materials

        self.indexation_parameters = indexation_parameters
        # self.StorageDict = StorageDict

    def CreateSpotDB(self):
        """
        create a spots Database

        set dictionnary self.indexed_spots
        """
        # For now, only DB created for a single file...

        # dictionary of exp spots
        for k in range(len(self.data_theta)):
            self.indexed_spots[k] = [k,  # index of experimental spot in .cor file
                                    self.data_theta[k] * 2.0,
                                    self.data_chi[k],  # 2theta, chi coordinates
                                    self.data_pixX[k],
                                    self.data_pixY[k],  # pixel coordinates
                                    self.data_I[k],  # intensity
                                    0]  # 0 means non indexed yet

    def getAbsoluteIndices_Non_Indexed_Spots_(self):
        """
        return list of exp. indices of exp. spots not yet indexed

        TODO seems to be the same than method select_exp_spots()
        """
        # dictionary of exp spots
        list_nonindexed = []
        for k in range(len(self.data_theta)):
            if self.indexed_spots[k][-1] == 0:
                list_nonindexed.append(k)
        return list_nonindexed

    def Update_DataToIndex_Dict(self, data_list):
        """
        udpate dictionary of exp. spots taking into account data_list values(indexation results)
        """
        # data_Miller, data_Energy, list_indexspot = data_list
        list_indexspot = data_list[2]

        print("list_indexspot in Update_DataToIndex_Dict", list_indexspot)

        # AllDataToIndex = self.indexation_parameters["AllDataToIndex"]
        # DataToIndex = self.indexation_parameters["DataToIndex"]

        return

    def Update_DB_fromIndexation(self, data_list):
        """
        udpate dictionary of exp. spots taking into account data_list values(indexation results)
        """
        # data_Miller, data_Energy, list_indexspot = data_list
        data_Miller, _, list_indexspot = data_list

        # print "data_Miller",data_Miller
        # print "data_Energy",data_Energy
        # print "list_indexspot",list_indexspot
        # print "lengthes of data_Miller, data_Energy, list_indexspot",len(data_Miller),len(data_Energy),len(list_indexspot)

        # updating exp spot dict.
        singleindices = []

        for k in range(len(list_indexspot)):
            #             absoluteindex = self.current_exp_spot_index_list[list_indexspot[k]]
            absoluteindex = list_indexspot[k]
            if not singleindices.count(list_indexspot[k]):
                singleindices.append(list_indexspot[k])
                #                 spotenergy = data_Energy[k] # wrong
                if self.DataSet.indexed_spots_dict[absoluteindex][-1] == 1:
                    # spot belong to the grain grain_index
                    spotenergy = self.DataSet.indexed_spots_dict[absoluteindex][9]

                self.indexed_spots[absoluteindex] = self.indexed_spots[absoluteindex][:-1] + [
                                                data_Miller[k], spotenergy, self.current_processedgrain, 1]

            else:
                print("this experimental spot #%d is too ambiguous" % absoluteindex)

        #         print "List of exp. spots indices that have been indexed"
        #         print np.sort(np.array(self.current_exp_spot_index_list)[np.array(singleindices)])

        self.current_processedgrain += 1  # preparing the indexation of next grain

        # self.control.SetValue(str(self.indexed_spots).replace('],',']\n'))  # writing the control with
        texttoshow = "Experimental spots database current state:\n"
        texttoshow += "#spot    2theta     chi    pixX    pixY    Intens.   (hkl)    Energy    #Grain  IndexedFlag\n"
        for k in range(len(self.indexed_spots)):

            if len(self.indexed_spots[k]) == 7:
                val = self.indexed_spots[k]
                texttoshow += ("%d,  %.06f ,%.06f,  %.06f ,%.06f,  %.06f,  %d \n" % tuple(val))
            elif len(self.indexed_spots[k]) == 10:
                val = (self.indexed_spots[k][:6]
                    + self.indexed_spots[k][6].tolist()
                    + self.indexed_spots[k][7:])
                texttoshow += ("%d,  %.06f ,%.06f,  %.06f ,%.06f,  %.06f,  %d, %d, "
                                        "%d,  %.06f, %d, %d \n" % tuple(val))

        texttoshow += "#Orientation matrix(ces)\n"
        for key, val in self.last_orientmatrix_fromindexation.items():
            texttoshow += "#Grain %s:\n" % key
            for i in range(3):
                texttoshow += "#   %.06f     %.06f    %.06f\n" % tuple(val[i])

        texttoshow += "#BMatrix\n"
        for key, val in self.last_Bmatrix_fromindexation.items():
            texttoshow += "#Grain %s:\n" % key
            for i in range(3):
                texttoshow += "#   %.06f     %.06f    %.06f\n" % tuple(val[i])

        texttoshow += "#EMatrix(in 10-3 unit)\n"
        for key, val in self.last_epsil_fromindexation.items():
            texttoshow += "#Grain %s:\n" % key
            for i in range(3):
                texttoshow += "#   %.06f     %.06f    %.06f\n" % tuple(val[i])

        texttoshow += "#Calibration Parameters\n"
        for par, value in zip(["dd", "xcen", "ycen", "xbet", "xgam"], self.defaultParam):
            texttoshow += "# %s     :   %s\n" % (par, value)

        self.control.SetValue(texttoshow)  # writing the control with

    def Edit_String_Indexation(self, data=([0], [0], [0], [0], [""], 0)):
        """
        NOT USED YET
        input in data:
        [0] list of experimental spot index
        """
        lines = "Indexation file from LAUE Pattern Program v1.0 2008 \n"

        nb_grains = data[-1]
        TWT, CHI, ENE, MIL, NAME = data[:5]
        for index_grain in range(nb_grains):
            nb_of_simulspots = len(TWT[index_grain])
            startgrain = "#G %d\t%s\t%d\n" % (index_grain, NAME[index_grain], nb_of_simulspots)

            lines += startgrain
            print(nb_of_simulspots)
            for data_index in range(nb_of_simulspots):
                linedata = "%d\t%d\t%d\t%d\t%.3f\t%.4f\t%.4f\n" % (data_index,
                                                                MIL[index_grain][data_index][0],
                                                                MIL[index_grain][data_index][1],
                                                                MIL[index_grain][data_index][2],
                                                                ENE[index_grain][data_index],
                                                                TWT[index_grain][data_index],
                                                                CHI[index_grain][data_index])
                lines += linedata
        # print "in edit",lines
        self.control.SetValue(lines)
        return True

    # Helper methods:
    def defaultFileDialogOptions(self):
        """ Return a dictionary with file dialog options that can be
            used in both the save file dialog as well as in the open
            file dialog. """
        wcd = "All files(*)|*|cor file(*.cor)|*.cor|fit2d peaks(*.peaks)|*.peaks|XMAS peaks list(*.dat)|*.dat|MAR CCD image(*.mccd)|*.mccd"
        return dict(
                    message="Choose a data file(peaks list or image)",
                    defaultDir=self.dirname,
                    wildcard=wcd)

    def defaultFileDialogOptionsImage(self):
        """ Return a dictionary with file dialog options that can be
            used in both the save file dialog as well as in the open
            file dialog. """
        wcd0 = ("MAR CCD image(*.mccd)|*.mccd|mar tiff(*.tiff)|*.tiff|mar tif(*.tif)|*.tif|")
        wcd0 += ("Princeton(*.spe)|*.spe|Frelon(*.edf)|*.edf|hdf5(*.h5)|*.h5|All files(*)|*")

        try:
            wcd = getwildcardstring("*")
        except:
            wcd = wcd0

        return dict(message="Choose an Image File", defaultDir=self.dirname, wildcard=wcd)

    def OnDocumentationpdf(self, _):
        """ open pdf file of Lauetools Documentation """
        pdffile_adress = "file://%s" % os.path.join(LaueToolsProjectFolder, "Documentation", "latex",
                                                                                "LaueTools.pdf")

        webbrowser.open(pdffile_adress)

    def OnDocumentationhtml(self, _):
        """ open html file of Lauetools Documentation """
        html_file_address = "file://%s" % os.path.join(LaueToolsProjectFolder, "Documentation",
                                                            "build", "html", "index.html")

        webbrowser.open(html_file_address)

    def OnTutorial(self, _):
        """ not implemented """
        dialog = wx.MessageDialog(self, "Not yet implemented \n"
            "in wxPython\n\n See \nhttps://sourceforge.net/userapps/mediawiki/jsmicha/index.php?title=Main_Page",
            "Tutorial", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()

    def OnSerieIndexation(self, _):
        """ not implemented  """
        dialog = wx.MessageDialog(self, "Not yet implemented \n" "in wxPython",
                                                            "OnSerieIndexation", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()

    def OnMapAnalysis(self, _):
        """ not implemented  """
        dialog = wx.MessageDialog(self, "Not yet implemented \n" "in wxPython",
                                                            "OnMapAnalysis", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()

    def OnAbout(self, _):
        """
        about lauetools and license
        """
        description = """LaueTools is toolkit for white beam x-ray microdiffraction
        Laue Pattern analysis. It allows Simulation & Indexation procedures written 
        in python by Jean-Sebastien MICHA \n  micha@esrf.fr\n\n and Odile Robach at the French CRG-IF beamline
         \n at BM32(European Synchrotron Radiation Facility).

        %s %s

        Support and help in developing this package:
        https://sourceforge.net/projects/lauetools/
        https://gitlab.esrf.fr/micha/lauetools
        """ % (MONTH, YEAR)
        f = open("License", "r")
        lines = f.readlines()
        mylicense = ""
        for line in lines:
            mylicense += line
        f.close()

        info = wx.AboutDialogInfo()

        info.SetIcon(wx.Icon(os.path.join("icons", "transmissionLaue.png"), wx.BITMAP_TYPE_PNG))
        info.SetName("LaueTools")
        info.SetVersion("%s %s %s" % (DAY, MONTH, YEAR))
        info.SetDescription(description)
        info.SetCopyright("(C) 2020-%s Jean-Sebastien Micha" % YEAR)
        info.SetWebSite("http://www.esrf.eu/UsersAndScience/Experiments/CRG/BM32/")
        info.SetLicence(mylicense)
        info.AddDeveloper("Jean-Sebastien Micha")
        info.AddDocWriter("Jean-Sebastien Micha")
        # info.AddArtist('The Tango crew')
        info.AddTranslator("Jean-Sebastien Micha")
        wx.AboutBox(info)

    def OnExit(self, _):
        """
        exit
        """
        self.Close()


# --- -------------------- Folder preferences
class PreferencesBoard(wx.Dialog):
    """
    Dialog Class to display folder preferences board
    """
    def __init__(self, parent, _id, title):

        wx.Dialog.__init__(self, parent, _id, title, size=(700, 350))

        panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(590, 190), pos=(5, 5))

        self.parent = parent
        self.dirname = os.path.abspath(parent.dirname)
        self.writefolder = None
        # widgets ----------
        wf = parent.writefolder
        if parent.writefolder is None:
            wf = 'None'
        wx.StaticText(panel,-1, "Current writing folder: %s"%wf, (25, 15))
        wx.StaticText(panel,-1, "Set results folder to:", (25, 55))
        self.samefolder = wx.RadioButton(panel, -1, "same folder as images or peaks lists when loading them (Be careful of write access!)", (25, 95))
        self.askfolder = wx.RadioButton(panel, -1, "user selected folder when loading a new image or peak list", (25, 135))
        self.userdefinedrelfolder = wx.RadioButton(panel, -1, "path relative to lauetools", (25, 175))
        self.userdefinedabsfolder = wx.RadioButton(panel, -1, "absolute path", (25, 215))
        self.samefolder.SetValue(True)

        self.relativepathname = wx.TextCtrl(panel, -1, "./analysis", (250, 170), (340, -1))
        self.abspathname = wx.TextCtrl(panel, -1, self.dirname, (250, 210), (340, -1))
        self.browsebtn = wx.Button(panel, -1,'...', (600, 210), (60, -1))

        self.browsebtn.Bind(wx.EVT_BUTTON, self.onBrowseFolder)

        wx.Button(panel, 1, "Accept", (15, 250), (90, 40))
        self.Bind(wx.EVT_BUTTON, self.OnAccept, id=1)

        wx.Button(panel, 2, "Quit", (150, 250), (90, 40))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=2)

    def onBrowseFolder(self, _):
        folder = OSLFGUI.askUserForDirname(self)
        self.abspathname.SetValue(str(folder))

    def OnAccept(self, _):
        """ accept and set parent folders correspondingly and quit """
        resetwf = False
        if self.samefolder.GetValue():
            self.writefolder = "."
        elif self.askfolder.GetValue():
            self.writefolder = None
            resetwf = True
        elif self.userdefinedrelfolder.GetValue():
            self.writefolder = os.path.join(LAUETOOLSFOLDER,
                                            str(self.relativepathname.GetValue())[2:])
        elif self.userdefinedabsfolder.GetValue():
            self.writefolder = str(self.abspathname.GetValue())

        if self.writefolder is not None:
            abspath = os.path.abspath(self.writefolder)
            if not os.access(abspath, os.W_OK):
                wx.MessageBox('Not writable folder: %s'%abspath,'Error')
            else:
                self.parent.writefolder = self.writefolder
                self.resetwf = resetwf
                self.Close()
        else:
            self.parent.writefolder = self.writefolder
            self.resetwf = resetwf
            self.Close()

    def OnQuit(self, _):
        """ quit """
        self.Close()


# --- ---------------------  CLIQUES board
class CliquesFindingBoard(wx.Frame):
    """
    Class to display GUI for cliques finding
    """
    def __init__(self, parent, _id, title, indexation_parameters):

        wx.Frame.__init__(self, parent, _id, title, size=(400, 330))

        self.panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(590, 390), pos=(5, 5))

        self.indexation_parameters = indexation_parameters
        self.parent = parent
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

        wx.StaticText(self.panel, -1, "Load AnglesLUT file ", (15, 175))
        loadanglesbtn = wx.Button(self.panel, -1, "...", (180, 170), (60, 30))
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
        # open dialog  for structure
        key_material = 'Ti'
        nLUT = 4

        latticeparameters = dict_Materials[key_material][1]
        print('computing angles in material: %s.\n-- Wait a bit --'%key_material)
        sortedangles = computesortedangles(latticeparameters, nLUT)

        import pickle

        self.LUTfilename = '%s_nlut%d.angles'%(key_material,nLUT)
        with open(self.LUTfilename, "wb") as f:
            pickle.dump(sortedangles, f)

        print('Sorted angles written in %s'%self.LUTfilename)

    def OnLoadAnglesFile(self, _):
        """ load specific lut angles """
        # lutfolder = '/home/micha/LaueToolsPy3/LaueTools/'
        # lutfilename = 'sortedanglesCubic_nLut_5_angles_18_60.angles'
        # LUTfilename = os.path.join(lutfolder,lutfilename)

        defaultFolderAnglesFile = LaueToolsProjectFolder

        wcd = "AnglesLUT file (*.angles)|*.angles|All files(*)|*"
        open_dlg = wx.FileDialog(self, message="Choose a file", defaultDir=defaultFolderAnglesFile,
                                                                defaultFile="",
                                                                wildcard=wcd,
                                                                style=wx.OPEN | wx.CHANGE_DIR)
        if open_dlg.ShowModal() == wx.ID_OK:
            self.LUTfilename = open_dlg.GetPath()
        open_dlg.Destroy()


    def OnSearch(self, _):
        """ start cliques search """
        spot_list = self.spotlist.GetValue()

        print("spot_list in OnSearch", spot_list)
        if spot_list[0] != "-":
            if spot_list[0] == "[":
                spot_index_central = str(spot_list)[1: -1].split(",")
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
        #print("ang_tol", ang_tol)
        Nodes = spot_index_central
        print("Clique finding for Nodes :%s" % Nodes)

        fullpath = os.path.join(self.parent.dirname, self.parent.DataPlot_filename,)

        res_cliques = give_bestclique(fullpath, nbmax_probed, ang_tol,
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
        """ quit """
        self.Close()


# --- ----------   DISPLAY SPlASH SCREEN
if WXPYTHON4:

    class MySplash(wxadv.SplashScreen):
        """
        display during a given period a image and give back the window focus
        """
        def __init__(self, parent, duration=2000):
            # pick a splash image file you have in the working folder
            image_file = os.path.join(LaueToolsProjectFolder, "icons", "transmissionLaue_fcc_111.png")
            # print("image_file", image_file)
            bmp = wx.Bitmap(image_file)
            # covers the parent frame

            wxadv.SplashScreen(bmp, wxadv.SPLASH_CENTRE_ON_PARENT | wxadv.SPLASH_TIMEOUT,
                                                                        duration,
                                                                        parent,
                                                                        wx.ID_ANY)

else:

    class MySplash(wx.SplashScreen):
        """
        display during a given period a image and give back the window focus
        """

        def __init__(self, parent, duration=2000):
            # pick a splash image file you have in the working folder

            image_file = os.path.join(LaueToolsProjectFolder,"icons", "transmissionLaue_fcc_111.png")
            # print("image_file", image_file)
            bmp = wx.Bitmap(image_file)
            # covers the parent frame

            wx.SplashScreen(bmp, wx.SPLASH_CENTRE_ON_PARENT | wx.SPLASH_TIMEOUT, duration,
                                                                                parent,
                                                                                wx.ID_ANY)


# --- ---------   REDIRECT STDOUT in WX.TEXTCTRL
class RedirectText:
    """
    Class to redirect stdout in wx.TextCtrl
    thanks to: kyosohma@gmail.com
    """
    def __init__(self, A_WxTextCtrl):
        self.out = A_WxTextCtrl

    def write(self, Somestring):
        """
        write string in a textCtrl
        """
        self.out.WriteText(Somestring)


def start():
    """ launcher of LaueToolsGUI (as module) """
    LaueToolsGUIApp = wx.App()
    LaueToolsframe = LaueToolsGUImainframe(None, -1, "Image Viewer and PeakSearch Board",
                                            projectfolder=LaueToolsProjectFolder)
    LaueToolsframe.Show()

    MySplash(LaueToolsframe, duration=500)

    LaueToolsGUIApp.MainLoop()


if __name__ == "__main__":
    start()
