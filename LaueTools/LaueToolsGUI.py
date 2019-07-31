from __future__ import absolute_import
r"""
LaueToolsGUI.py is a (big) central GUI for microdiffraction Laue Pattern analysis
and simulation

This module belongs to the open source LaueTools project
with a free code repository at at gitlab.esrf.fr

(former version with python 2.7 at https://sourceforge.net/projects/lauetools/)

or for python3 and 2 in

https://gitlab.esrf.fr/micha/lauetools

J. S. Micha July 2019
mailto: micha --+at-+- esrf --+dot-+- fr
"""
__version__ = "$Revision: 2349 $"
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"


import time, sys
import copy
import os.path

import matplotlib

matplotlib.use("WXAgg")

from matplotlib.figure import Figure

from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

from pylab import FuncFormatter
from matplotlib import __version__ as matplotlibversion
from pylab import Rectangle

import numpy as np
import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
# import wx.lib.scrolledpanel as scrolled
# import ProportionalSplitter as PropSplit

import webbrowser

# TODO restrict imports to only used functions
if sys.version_info.major == 3:
    from . import indexingAnglesLUT as INDEX
    from . import indexingImageMatching as IIM
    from . import indexingSpotsSet as ISS
    from . import lauecore as LAUE
    from . import graingraph as GraGra
    from . import LaueGeometry as F2TC
    from . import LaueSpotsEditor as LSEditor
    from . import LaueSimulatorGUI as LSGUI
    from . import CrystalParameters as CP
    from . import IOLaueTools as IOLT
    from . import generaltools as GT
    from . import dict_LaueTools as DictLT
    from . import PeakSearchGUI
    from . import DetectorParameters as DP
    from . import DetectorCalibration as DC
    from . import CCDFileParametersGUI as CCDParamGUI
    from . import dragpoints as DGP
    from . import matchingrate
    from . AutoindexationGUI import (RecognitionResultCheckBox, DistanceScreeningIndexationBoard, )
    from . import threadGUI2 as TG
else:
    import indexingAnglesLUT as INDEX
    import indexingImageMatching as IIM
    import indexingSpotsSet as ISS
    import lauecore as LAUE
    import graingraph as GraGra
    import LaueGeometry as F2TC
    import LaueSpotsEditor as LSEditor
    import LaueSimulatorGUI as LSGUI
    import CrystalParameters as CP
    import IOLaueTools as IOLT
    import generaltools as GT
    import dict_LaueTools as DictLT
    import PeakSearchGUI
    import DetectorParameters as DP
    import DetectorCalibration as DC
    import CCDFileParametersGUI as CCDParamGUI
    import dragpoints as DGP
    import matchingrate
    from AutoindexationGUI import (RecognitionResultCheckBox, DistanceScreeningIndexationBoard, )
    import threadGUI2 as TG

SIZE_PLOTTOOLS = (8, 6)
# --- ------------   CONSTANTS
# sign of CCD camera angle =1 to mimic XMAS convention
SIGN_OF_GAMMA = 1

PI = np.pi
DEG = PI / 180.0
CST_ENERGYKEV = DictLT.CST_ENERGYKEV

# DEFAULT_CCDCAMERA = 'MARCCD165' #'sCMOS_16M' #   # VHR_Feb13'
# DEFAULT_DETECTORPARAMETERS = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201] #  'MARCCD165'
# DEFAULT_DETECTORPARAMETERS = [77., 1975., 2110., 0.43, -0.22] # sCMOS_16M, sans flip LR' # after debug readmccd 17Jul18
# DEFAULT_DETECTORPARAMETERS = [77.4, 983., 977., 0.32, -0.28] # sCMOS, sans flip LR #
DEFAULT_CCDCAMERA = "sCMOS"
DEFAULT_DETECTORPARAMETERS = [
    77.088,
    1012.45,
    1049.92,
    0.423,
    0.172,
]  # bin 2x2 avec flip LR

DICT_LAUE_GEOMETRIES = DictLT.DICT_LAUE_GEOMETRIES

DICT_LAUE_GEOMETRIES_INFO = DictLT.DICT_LAUE_GEOMETRIES_INFO

# --- --------  SOME GUI parameters
ID_FILE_OPEN = 101
ID_FILE_EXIT = 103

# LaueToolsProjectFolder = os.path.abspath(os.curdir)
#
# print "LaueToolsProjectFolder", LaueToolsProjectFolder
#
# print "__file__", __file__

LaueToolsProjectFolder = os.path.split(__file__)[0]

print("LaueToolsProjectFolder", LaueToolsProjectFolder)

# last modified date of this module is displayed in title and documentation and about
import datetime

try:
    # timesinceepoch = os.path.getmtime('LaueToolsGUI.py')
    modifiedTime = os.path.getmtime(
        os.path.join(LaueToolsProjectFolder, "LaueToolsGUI.py")
    )
    # print datetime.datetime.fromtimestamp(modifiedTime).strftime("%d%b%Y %H:%M:%S")
    DAY, MONTH, YEAR = (
        datetime.datetime.fromtimestamp(modifiedTime).strftime("%d %B %Y").split()
    )
except:
    DAY, MONTH, YEAR = "FromDistribution", "", "2014"

REV = __version__.split()[-2]


# --- ------------  MAIN GUI WINDOW
class MainWindow(wx.Frame):
    """
    class of the main window of LaueTools GUI
    """

    def __init__(
        self,
        parent,
        _id,
        title,
        filename="",
        consolefile="defaultLTlogfile.log",
        projectfolder=None,
    ):

        screenSize = wx.DisplaySize()
        screenWidth = screenSize[0]
        screenHeight = screenSize[1]
        # super(MainWindow, self).__init__(None, size =(1000, 800))
        wx.Frame.__init__(self, parent, _id, title, size=(700, 500))
        panel = wx.Panel(self, -1)

        self.SetIcon(
            wx.Icon(
                os.path.join(projectfolder, "icons", "transmissionLaue_fcc_111.png"),
                wx.BITMAP_TYPE_PNG,
            )
        )

        self.filename = filename
        self.dirname = projectfolder
        print("self.dirname", self.dirname)
        self.writefolder = self.dirname
        #        self.lauetoolsrootdirectory = os.curdir
        self.consolefile = consolefile

        # --- begin of general layout -------------------
        self.CreateExteriorWindowComponents()

        # peak list editor
        self.control = wx.TextCtrl(
            panel,
            -1,
            "",
            size=(700, 400),
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL,
        )
        self.console = wx.TextCtrl(
            panel,
            -1,
            "",
            size=(700, 100),
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.VSCROLL,
        )

        sizer = wx.FlexGridSizer(cols=1, hgap=6, vgap=6)

        sizer.AddMany([self.control, self.console])
        panel.SetSizer(sizer)

        # ---  end of general layout ------------------

        # default color LUT for image viewer
        self.mapsLUT = "OrRd"

        # Default CCD file image props
        self.CCDLabel = DEFAULT_CCDCAMERA
        # default pixel size and frame dimension:  MARCCD
        self.pixelsize = DictLT.dict_CCD[self.CCDLabel][1]  # 165./2048
        # self.pixelsize = 0.031  # for VHR camera
        self.framedim = DictLT.dict_CCD[self.CCDLabel][0]  # (2048, 2048)
        # self.framedim =(2671,4008)
        # for VHR camera as seen by XMAS LaueTools and other array readers(reverse order for fit2S)
        self.fliprot = DictLT.dict_CCD[self.CCDLabel][3]
        self.headeroffset, self.dataformat = DictLT.dict_CCD[self.CCDLabel][4:6]
        self.file_extension = DictLT.dict_CCD[self.CCDLabel][7]
        self.saturationvalue = DictLT.dict_CCD[self.CCDLabel][2]

        self.defaultParam = DEFAULT_DETECTORPARAMETERS

        self.detectordiameter = 165.0
        self.kf_direction = "Z>0"
        self.kf_direction_from_file = None

        self.SimulPrefixName = "mySimul_"
        self.indexsimulation = 0

        self.Current_Data = None
        self.DataPlot_filename = None
        self.PeakListFileName = None
        self.Current_peak_data = None
        self.data_theta = None
        self.data_gnomonx = None
        self.data_pixX, self.data_pixY = None, None

        self.Matrix_Store = []
        self.mat_store_ind = 0

        self.ClassicalIndexation_Tabledist = None

        self.picky = None
        self.last_orientmatrix_fromindexation = {}  # dict of orientation matrix found
        self.last_Bmatrix_fromindexation = {}  # dict of Bmatrix matrix found
        self.last_epsil_fromindexation = {}  # dict of deviatoric strain found
        self.indexed_spots = {}  # dictionary of exp spots which are indexed
        self.current_processedgrain = (
            0
        )  # index corresponding to the grain found in data
        self.current_exp_spot_index_list = []

        # loading dictionaries
        self.dict_calib = DictLT.dict_calib  # calibration parameter

        self.dict_Materials = DictLT.dict_Materials  # Materials or compounds
        self.dict_Extinc = DictLT.dict_Extinc
        self.dict_Transforms = DictLT.dict_Transforms  # deformation dict
        self.dict_Vect = (
            DictLT.dict_Vect
        )  # initial orientation and strain matrix(UB matrix)
        self.dict_Rot = (
            DictLT.dict_Rot
        )  # additional matrix of rotation applied in left of UB
        self.dict_Eul = (
            DictLT.dict_Eul
        )  # additional matrix of rotation enter as 3 angles / elemntary rotations applied in left of UB

        # make a list of safe functions
        self.safe_list = [ "math", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "degrees", "e", "exp", "fabs", "floor",
            "fmod", "frexp", "hypot", "ldexp", "log", "log10", "modf", "pi", "pow", "radians", "sin", "sinh", "sqrt", "tan", "tanh", ]
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
            (
                wx.ID_ANY,
                "&Set CCD File Parameters",
                "CCD ImageFile reading parameters dialog",
                self.OnSetFileCCDParam,
            ),
            (
                wx.ID_ANY,
                "&Set Laue Geometry",
                "Set general CCD position for Laue experiment",
                self.OnSetLaueDetectorGeometry,
            ),
            (None, None, None, None),
            (
                wx.ID_OPEN,
                "&Open Image && PeakSearch",
                "View Image & Peak Search",
                self.OnOpenImage,
            ),
            (None, None, None, None),
            (
                wx.ID_ANY,
                "&Detector Calibration",
                "Detector Calibration from a known reference crystal",
                self.OnDetectorCalibration,
            ),
            (
                5151,
                "&Set or Reset Detector Parameters",
                "Open detector parameters board. Set or Reset calibration parameters dialog, compute Laue spots scattering angles",
                self.recomputeScatteringAngles,
            ),
            (None, None, None, None),
            (
                wx.ID_ANY,
                "Folder Preferences",
                "Define where to write files",
                self.OnPreferences,
            ),
        ]:

            #             print '_id', _id
            if _id is None:
                ReadImageMenu.AppendSeparator()
            else:
                item = ReadImageMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        #         ReadImageMenu.Enable(id=5151, enable=False)

        ManualIndexation_SubMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (
                wx.ID_ANY,
                "&2thetaChi",
                "Selection and recognition Tools in(2theta, chi) coordinates",
                self.OnPlot_2ThetaChi,
            ),
            (
                wx.ID_ANY,
                "&Gnomon.",
                "Selection and recognition Tools in gnomonic plane coordinates",
                self.OnPlot_Gnomon,
            ),
            (
                wx.ID_ANY,
                "&CCD Pixel",
                "Selection and recognition Tools in CCD pixels coordinates",
                self.OnPlot_Pixels,
            ),
        ]:
            if _id is None:
                ManualIndexation_SubMenu.AppendSeparator()
            else:
                item = ManualIndexation_SubMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        IndexationMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (
                wx.ID_ANY,
                "&Open Peak List",
                "Open a data file(peak list)",
                self.OnOpenPeakList,
            ),
            (None, None, None, None),
            (
                wx.ID_ANY,
                "&Check Orientation",
                "Enter Orientation matrix, Material and check Matching with the current experimental Laue Pattern spots list",
                self.OnCheckOrientationMatrix,
            ),
            (None, None, None, None),
            (
                wx.ID_ANY,
                "&Automatic Indexation",
                "Indexation by recognition of mutual lattice planes normals angles calculated from pairs of Laue spots",
                self.OnClassicalIndexation,
            ),
            (
                5050,
                "&Image Matching",
                "Spots position matching with Databank(in Hough Space)",
                self.OnHoughMatching,
            ),
            (None, None, None, None),
            (
                wx.ID_ANY,
                "&Manual Indexation",
                "Plot Data and Manual indexation (Angles Recognition)",
                ManualIndexation_SubMenu,
            ),
            (None, None, None, None),
            (
                wx.ID_SAVE,
                "&Save Indexation Results",
                "Save indexation results viewed in control in *.idx file",
                self.OnSaveIndexationResultsfile,
            ),
            (
                wx.ID_SAVEAS,
                "Save &As Indexation Results",
                "Save indexation results viewed in control in *.idx file",
                self.OnFileResultsSaveAs,
            ),
            (
                wx.ID_ANY,
                "Save Non &Indexed .cor file",
                "Save non indexed spots in a .cor file",
                self.SaveNonIndexedSpots,
            ),
        ]:
            #             print("handler",type(handler))
            if _id is None:
                #                 print "None Menu"
                IndexationMenu.AppendSeparator()
            elif isinstance(handler, (wx.Menu, wx.Object)):
                #                 print 'submenu'
                if WXPYTHON4:
                    item = IndexationMenu.AppendSubMenu(handler, label, helpText)
                else:
                    item = IndexationMenu.AppendMenu(
                        wx.ID_ANY, label, handler, helpText
                    )
            else:
                #                 print 'default menu'
                item = IndexationMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        IndexationMenu.Enable(id=5050, enable=False)

        # FileSequenceMenu = wx.Menu()
        # for _id, label, helpText, handler in \
        # [(wx.ID_ANY, '&Indexation', 'Files serie indexation',
        # self.OnSerieIndexation),
        # (wx.ID_ANY, '&Postprocessing', 'File serie indexation results postprocessing', self.OnMapAnalysis)
        # ]:
        # if _id  ==  None:
        # FileSequenceMenu.AppendSeparator()
        # else:
        # item = FileSequenceMenu.Append(_id, label, helpText)
        # self.Bind(wx.EVT_MENU, handler, item)

        SimulationMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (
                wx.ID_ANY,
                "&PolyGrains Simulation",
                "Polycrystal selection & simulation",
                self.Creating_Grains_parametric,
            ),
            (None, None, None, None),
            (
                wx.ID_ANY,
                "&Edit Matrix",
                "Edit or Load Orientation Matrix",
                self.OnEditMatrix,
            ),
            (
                wx.ID_ANY,
                "&Edit UB, B, Crystal",
                "Edit B or UB Matrix and unit cell structure and extinctions",
                self.OnEditUBMatrix,
            ),
        ]:
            if _id is None:
                SimulationMenu.AppendSeparator()
            else:
                item = SimulationMenu.Append(_id, label, helpText)
                self.Bind(wx.EVT_MENU, handler, item)

        #        DefaultParametersMenu = wx.Menu()
        #        for _id, label, helpText, handler in \
        #            [(wx.ID_ANY, '&Analysis', 'Analysis parameters(Crystal, threshold, display,...)',
        #                self.OnAnalysisParameter),
        #           (wx.ID_ANY, '&Simulation', 'Simulation parameters(Crystals, display,...)',
        #                self.OnRecognitionParam)
        #            ]:
        #            if _id  ==  None:
        #                DefaultParametersMenu.AppendSeparator()
        #            else:
        #                item = DefaultParametersMenu.Append(_id, label, helpText)
        #                self.Bind(wx.EVT_MENU, handler, item)

        HelpMenu = wx.Menu()
        for _id, label, helpText, handler in [
            (5051, "&Tutorial", "Tutorial", self.OnTutorial),
            (
                wx.ID_ANY,
                "&HTML Documentation",
                "Documentation",
                self.OnDocumentationhtml,
            ),
            (wx.ID_ANY, "&PDF Documentation", "Documentation", self.OnDocumentationpdf),
            (None, None, None, None),
            (wx.ID_ANY, "&About", "Information about this program", self.OnAbout),
        ]:
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
        menuBar.Append(
            ReadImageMenu, "&ReadImage"
        )  # Add the ReadImageMenu to the MenuBar
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
        # MainWindow.SetTitle overrides wx.Frame.SetTitle, so we have to
        # call it using super:
        super(MainWindow, self).SetTitle(
            "LaueToolsGUI    Simulation & Indexation Program         %s" % self.filename
        )

    # --- -------- Main FUNCTIONS called from MENU and submenus
    def OnOpenImage(self, event):
        """
        ask user to select folder and file
        and launch the peak search board(class PeakSearchFrame)
        """
        if self.askUserForFilename(
            style=wx.OPEN, **self.defaultFileDialogOptionsImage()
        ):

            # print "Current directory",self.dirname
            #            self.lauetoolsrootdirectory = os.curdir
            os.chdir(self.dirname)
            print("self.dirname", self.dirname)
            print(os.curdir)

            # print String_in_File_Data # in stdout/stderr
            self.DataPlot_filename = str(self.filename)
            print("Current file   :", self.DataPlot_filename)

            #             prefix, file_extension = self.DataPlot_filename.split('.')
            nbparts = len(self.DataPlot_filename.split("."))
            if nbparts == 2:
                prefix, file_extension = self.DataPlot_filename.rsplit(".", 1)
            elif nbparts == 3:
                prefix, ext1, ext2 = self.DataPlot_filename.rsplit(".", 2)
                file_extension = ext1 + "." + ext2

            print("prefix", prefix)

            print("extension", file_extension)

            if file_extension in DictLT.list_CCD_file_extensions:

                detectedCCDlabel = DP.autoDetectDetectorType(file_extension)
                if detectedCCDlabel is not None:
                    self.CCDLabel = DP.autoDetectDetectorType(file_extension)
                DPBoard = CCDParamGUI.CCDFileParameters(
                    self, -1, "CCD File Parameters Board", self.CCDLabel
                )
                DPBoard.ShowModal()
                DPBoard.Destroy()

                initialParameter = {}
                initialParameter["title"] = "peaksearch Board"
                initialParameter["imagefilename"] = self.filename
                initialParameter["dirname"] = self.dirname
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

                peakserchframe = PeakSearchGUI.MainPeakSearchFrame(
                    self, -1, initialParameter, "peaksearch Board"
                )
                peakserchframe.Show(True)

    #                    self.DataPlot_filename = ploimage.peaks_filename

    def OnOpenPeakList(self, event):
        """
        Load Peak list data (dat or cor)
        """
        if self.askUserForFilename(style=wx.OPEN, **self.defaultFileDialogOptions()):
            # print "Current directory in OnOpenPeakList()",self.dirname
            os.chdir(self.dirname)

            # print String_in_File_Data # in stdout/stderr
            self.DataPlot_filename = str(self.filename)
            print("Current file   :", self.DataPlot_filename)

            prefix, file_extension = self.DataPlot_filename.rsplit(".", 1)

            if file_extension in ("dat", "DAT"):
                # need 4 columns !!
                self.Launch_DetectorParamBoard(event)
                LaueGeomBoard = SetGeneralLaueGeometry(self, -1, "Select Laue Geometry")
                LaueGeomBoard.ShowModal()
                LaueGeomBoard.Destroy()

                self.PeakListFileName = self.DataPlot_filename

                print("kf_direction in OnOpenPeakList", self.kf_direction)

                (
                    twicetheta,
                    chi,
                    dataintensity,
                    data_x,
                    data_y,
                ) = F2TC.Compute_data2thetachi(
                    self.PeakListFileName,
                    (0, 1, 3),
                    1,  # 2 for centroid intensity, 3 for integrated  intensity
                    sorting_intensity="yes",
                    param=self.defaultParam,
                    signgam=SIGN_OF_GAMMA,
                    pixelsize=self.pixelsize,
                    kf_direction=self.kf_direction,
                )

                IOLT.writefile_cor(
                    "dat_" + prefix,
                    twicetheta,
                    chi,
                    data_x,
                    data_y,
                    dataintensity,
                    sortedexit=0,
                    param=self.defaultParam + [self.pixelsize],
                    initialfilename=self.PeakListFileName,
                )  # check sortedexit = 0 or 1 to have decreasing intensity sorted data
                print("%s has been created with defaultparameter" % (prefix + ".cor"))
                print("%s" % str(self.defaultParam))
                self.kf_direction_from_file = self.kf_direction
                # a .cor file is now created
                file_extension = "cor"
                self.DataPlot_filename = "dat_" + prefix + "." + file_extension
                # WARNING: it will be read in the next "if" clause

            # for .cor file ------------------------------
            if file_extension == "cor":
                # read peak list and detector calibration parameters
                self.OpenCorfile(self.DataPlot_filename)

                # compute Gnomonic projection
                dataselected = IOLT.createselecteddata(
                    (self.data_theta * 2, self.data_chi, self.data_I),
                    np.arange(len(self.data_theta)),
                    len(self.data_theta),
                )[0]
                self.data_gnomonx, self.data_gnomony = IIM.ComputeGnomon_2(dataselected)

                # ---------------------------------------------
                self.filename = self.DataPlot_filename
                self.SetTitle()  # Update the window title with the new filename
                # create DB spots(dictionary)
                self.CreateSpotDB()
                self.setAllDataToIndex_Dict()

            # -----------------------------------------------
            textfile = open(os.path.join(self.dirname, self.DataPlot_filename), "r")
            String_in_File_Data = textfile.read()
            self.control.SetValue(String_in_File_Data)
            textfile.close()

            self.current_processedgrain = 0
            self.last_orientmatrix_fromindexation = {}
            self.last_Bmatrix_fromindexation = {}
            self.last_epsil_fromindexation = {}

    def Launch_DetectorParamBoard(self, event):
        """Board to enter manually detector params
        Launch Entry dialog
        """
        self.Parameters_dict = {}
        self.Parameters_dict["CCDLabel"] = self.CCDLabel
        self.Parameters_dict["CCDParam"] = self.defaultParam
        self.Parameters_dict["pixelsize"] = self.pixelsize
        self.Parameters_dict["framedim"] = self.framedim
        self.Parameters_dict["detectordiameter"] = self.detectordiameter
        self.Parameters_dict["kf_direction"] = self.kf_direction
        # print "old param",self.defaultParam+[self.pixelsize]
        DPBoard = DP.DetectorParameters(
            self, -1, "Detector parameters Board", self.Parameters_dict
        )
        DPBoard.ShowModal()
        DPBoard.Destroy()

        # update 2theta and chi experimental data

    #         self.create_read_corfile()

    def recomputeScatteringAngles(self, evt):

        if self.DataPlot_filename is None:
            wx.MessageBox("You must first open a peaks list like in .dat or .cor file")
            return

        prefix, file_extension = self.DataPlot_filename.rsplit(".", 1)

        os.chdir(self.dirname)

        fullpathfile = os.path.join(self.dirname, self.DataPlot_filename)

        print("self.defaultParam before ", self.defaultParam)

        self.Launch_DetectorParamBoard(evt)

        print("self.defaultParam after", self.defaultParam)

        (twicetheta, chi, dataintensity, data_x, data_y) = F2TC.Compute_data2thetachi(
            fullpathfile,
            (0, 1, 3),
            1,  # 2 for centroid intensity, 3 for integrated  intensity
            sorting_intensity="yes",
            param=self.defaultParam,
            signgam=SIGN_OF_GAMMA,
            pixelsize=self.pixelsize,
            kf_direction=self.kf_direction,
        )

        IOLT.writefile_cor(
            "update_" + prefix,
            twicetheta,
            chi,
            data_x,
            data_y,
            dataintensity,
            sortedexit=0,
            param=self.defaultParam + [self.pixelsize],
            initialfilename=self.PeakListFileName,
            dirname_output=self.dirname,
        )  # check sortedexit = 0 or 1 to have decreasing intensity sorted data
        print("%s has been created" % ("update_" + prefix + ".cor"))
        print("%s" % str(self.defaultParam))
        self.kf_direction_from_file = self.kf_direction
        # a .cor file is now created
        file_extension = "cor"
        self.DataPlot_filename = "update_" + prefix + "." + file_extension
        # WARNING: it will be read in the next "if" clause

        # for .cor file ------------------------------

        # read peak list and detector calibration parameters
        self.OpenCorfile(self.DataPlot_filename)

        # compute Gnomonic projection
        dataselected = IOLT.createselecteddata(
            (self.data_theta * 2, self.data_chi, self.data_I),
            np.arange(len(self.data_theta)),
            len(self.data_theta),
        )[0]
        self.data_gnomonx, self.data_gnomony = IIM.ComputeGnomon_2(dataselected)

        # ---------------------------------------------

        self.filename = self.DataPlot_filename
        self.SetTitle()  # Update the window title with the new filename
        # create DB spots(dictionary)
        self.CreateSpotDB()

        # -----------------------------------------------
        textfile = open(os.path.join(self.dirname, self.DataPlot_filename), "r")
        String_in_File_Data = textfile.read()
        self.control.SetValue(String_in_File_Data)
        textfile.close()

        self.current_processedgrain = 0
        self.last_orientmatrix_fromindexation = {}
        self.last_Bmatrix_fromindexation = {}
        self.last_epsil_fromindexation = {}

    def OnSetFileCCDParam(self, event):
        """Enter manually CCD file params
        Launch Entry dialog
        """

        DPBoard = CCDParamGUI.CCDFileParameters(
            self, -1, "CCD File Parameters Board", self.CCDLabel
        )
        DPBoard.ShowModal()
        DPBoard.Destroy()

    def OnFileResultsSaveAs(self, event):
        dlg = wx.TextEntryDialog(
            self,
            "Enter Indexation filename(*.res):",
            "Indexation Results Filename Entry",
        )

        if dlg.ShowModal() == wx.ID_OK:
            filename = str(dlg.GetValue())

            self.OnSaveIndexationResultsfile(event, filename=str(filename))

        dlg.Destroy()

    def CheckCCDCalibParameters(self):
        """
        check if all CCD parameters are read from file .cor
        """
        missing_parameters = []

        ccp = DictLT.CCD_CALIBRATION_PARAMETERS

        sorted_list_parameters = [ccp[7], ccp[10]]
        for key in sorted_list_parameters:
            if key not in self.CCDCalibDict:
                missing_param = key

                if (
                    missing_param == "kf_direction"
                    and self.kf_direction_from_file is None
                ):
                    LaueGeomBoard = SetGeneralLaueGeometry(
                        self, -1, "Select Laue Geometry"
                    )
                    LaueGeomBoard.ShowModal()
                    LaueGeomBoard.Destroy()

                if missing_param == "CCDLabel" or self.CCDLabel is None:
                    print(
                        "self.detectordiameter in CheckCCDCalibParameters",
                        self.detectordiameter,
                    )
                    DPBoard = CCDParamGUI.CCDFileParameters(
                        self, -1, "CCD File Parameters Board", self.CCDLabel
                    )
                    DPBoard.ShowModal()
                    DPBoard.Destroy()

    def OnSetLaueDetectorGeometry(self, event):
        LaueGeomBoard = SetGeneralLaueGeometry(self, -1, "Select Laue Geometry")
        #         LaueGeomBoard.Show(True)
        LaueGeomBoard.ShowModal()
        LaueGeomBoard.Destroy()

    def OnPreferences(self, event):
        """
        call the board to define destination folder to  write files        
        """
        PB = PreferencesBoard(self, -1, "Folder Preferences Board")
        PB.ShowModal()
        PB.Destroy()

        print("New write folder", self.writefolder)

    # --- ------------ Indexation Functions
    def OnCheckOrientationMatrix(self, event):
        r"""
        Check if user input matrix, material parameters can produce a Laue pattern that matches
        the current experimental list of spots
        """
        if 0:
            if self.data_theta is None:
                self.OpenDefaultData()

            #             dialog = wx.MessageDialog(self, 'Please load first a data file with cor extension \n'
            #             'in File/Open Menu', 'No Data Loaded', wx.OK)
            #             dialog.ShowModal()
            #             dialog.Destroy()
            #             event.Skip()
            #             return

            self.current_exp_spot_index_list = (
                self.getAbsoluteIndices_Non_Indexed_Spots_()
            )

            #         nb_exp_spots_data = len(self.data_theta)
            #
            #         index_to_select = np.take(self.current_exp_spot_index_list,
            #                                       np.arange(nb_exp_spots_data))

            self.select_theta = self.data_theta[self.current_exp_spot_index_list]
            self.select_chi = self.data_chi[self.current_exp_spot_index_list]
            self.select_I = self.data_I[self.current_exp_spot_index_list]
            self.select_dataXY = (
                self.data_pixX[self.current_exp_spot_index_list],
                self.data_pixY[self.current_exp_spot_index_list],
            )
            #         self.select_dataXY = self.data_XY[index_to_select]
            #         CCDdetectorparameters
            #         self.StorageDict=None
            self.data_pixXY = self.data_pixX, self.data_pixY

            self.data = (
                2 * self.select_theta,
                self.select_chi,
                self.select_I,
                self.DataPlot_filename,
            )

            if len(self.current_exp_spot_index_list) == 0:
                wx.MessageBox(
                    "There are no more spots left to be indexed now !", "INFO"
                )
                return

            #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

            self.indexation_parameters = {}
            self.indexation_parameters["kf_direction"] = self.kf_direction
            self.indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
            self.indexation_parameters["dict_Materials"] = self.dict_Materials

            self.indexation_parameters["data_theta"] = self.data_theta
            self.indexation_parameters["data_chi"] = self.data_chi
            self.indexation_parameters["data_I"] = self.data_I
            self.indexation_parameters["dataXY"] = self.data_pixXY
            self.indexation_parameters[
                "current_exp_spot_index_list"
            ] = self.current_exp_spot_index_list
            self.indexation_parameters["ClassicalIndexation_Tabledist"] = None
            self.indexation_parameters["dict_Rot"] = self.dict_Rot
            self.indexation_parameters[
                "current_processedgrain"
            ] = self.current_processedgrain
            self.indexation_parameters["detectordiameter"] = self.detectordiameter
            self.indexation_parameters["pixelsize"] = self.pixelsize
            self.indexation_parameters["dim"] = self.framedim
            self.indexation_parameters["detectorparameters"] = self.defaultParam
            self.indexation_parameters["detectordistance"] = self.defaultParam[0]
            self.indexation_parameters["CCDLabel"] = self.CCDLabel
            self.indexation_parameters["flipyaxis"] = True  # CCD MAr 165
            self.indexation_parameters["Filename"] = self.DataPlot_filename
            self.indexation_parameters["CCDcalib"] = self.defaultParam
            self.indexation_parameters["framedim"] = self.framedim

            self.indexation_parameters["mainAppframe"] = self

        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print(
            "len(self.current_exp_spot_index_list)",
            len(self.current_exp_spot_index_list),
        )

        #         nb_exp_spots_data = len(self.data_theta)
        #
        #         index_to_select = np.take(self.current_exp_spot_index_list,
        #                                       np.arange(nb_exp_spots_data))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (
            self.data_pixX[self.current_exp_spot_index_list],
            self.data_pixY[self.current_exp_spot_index_list],
        )
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        #         self.StorageDict=None
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.DataPlot_filename,
        )

        if len(self.current_exp_spot_index_list) == 0:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        print(
            "AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters
        )

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
        self.indexation_parameters["DataToIndex"][
            "current_exp_spot_index_list"
        ] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"][
            "ClassicalIndexation_Tabledist"
        ] = None

        print(
            "self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
            self.indexation_parameters["DataToIndex"]["data_theta"],
        )
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters[
            "current_processedgrain"
        ] = self.current_processedgrain
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
        StorageDict["dict_Rot"] = DictLT.dict_Rot
        StorageDict["dict_Materials"] = DictLT.dict_Materials
        # --------------end of common part before indexing------------------------

        nbmatrices = self.EnterMatrix(1)
        self.Enterkey_material()
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

            # self.kf_direction = 'Z>0'

            # only orientmatrix, self.key_material are used ----------------------
            vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
            # old definition of grain
            grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]
            # ------------------------------------------------------------------

            #                 print "self.indexation_parameters", self.indexation_parameters
            TwicethetaChi = LAUE.SimulateResult(
                grain,
                5,
                self.emax,
                self.indexation_parameters,
                ResolutionAngstrom=False,
                fastcompute=1,
            )
            self.TwicethetaChi_solution = TwicethetaChi
            paramsimul = (grain, 5, self.emax)

            self.indexation_parameters["paramsimul"].append(paramsimul)
            self.indexation_parameters["bestmatrices"].append(orientmatrix)
            self.indexation_parameters["TwicethetaChi_solutions"].append(
                self.TwicethetaChi_solution
            )

            self.statsresidues.append(self.computeAngularMatching(orientmatrix))

        self.PlotandRefineSolution()

    def OnClassicalIndexation(self, event):
        """
        Call the ClassicalIndexationBoard Class with current non indexed spots list
        
        see Autoindexation.py module
        """

        if self.data_theta is None:
            self.OpenDefaultData()

        #             dialog = wx.MessageDialog(self, 'Please load first a data file with cor extension \n'
        #             'in File/Open Menu', 'No Data Loaded', wx.OK)
        #             dialog.ShowModal()
        #             dialog.Destroy()
        #             event.Skip()
        #             return

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print(
            "len(self.current_exp_spot_index_list)",
            len(self.current_exp_spot_index_list),
        )

        #         nb_exp_spots_data = len(self.data_theta)
        #
        #         index_to_select = np.take(self.current_exp_spot_index_list,
        #                                       np.arange(nb_exp_spots_data))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (
            self.data_pixX[self.current_exp_spot_index_list],
            self.data_pixY[self.current_exp_spot_index_list],
        )
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        #         self.StorageDict=None
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.DataPlot_filename,
        )

        if len(self.current_exp_spot_index_list) == 0:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        print(
            "AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters
        )

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
        self.indexation_parameters["DataToIndex"][
            "current_exp_spot_index_list"
        ] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"][
            "ClassicalIndexation_Tabledist"
        ] = None

        print(
            "self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
            self.indexation_parameters["DataToIndex"]["data_theta"],
        )
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters[
            "current_processedgrain"
        ] = self.current_processedgrain
        self.indexation_parameters["detectordiameter"] = self.detectordiameter
        self.indexation_parameters["pixelsize"] = self.pixelsize
        self.indexation_parameters["dim"] = self.framedim
        self.indexation_parameters["detectorparameters"] = self.defaultParam
        self.indexation_parameters["CCDLabel"] = self.CCDLabel

        self.indexation_parameters["mainAppframe"] = self

        # update DataSetObject
        self.DataSet.dim = self.indexation_parameters["dim"]
        self.DataSet.pixelsize = self.indexation_parameters["pixelsize"]
        self.DataSet.detectordiameter = self.indexation_parameters["detectordiameter"]
        self.DataSet.kf_direction = self.indexation_parameters["kf_direction"]

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = DictLT.dict_Rot
        StorageDict["dict_Materials"] = DictLT.dict_Materials

        titleboard = (
            "Spots interdistance Screening Indexation Board  file: %s"
            % self.DataPlot_filename
        )

        # Launch Automatic brute force indexation procedure (angular distance recognition)
        DistanceScreeningIndexationBoard(
            self,
            -1,
            self.indexation_parameters,
            titleboard,
            StorageDict=StorageDict,
            DataSetObject=self.DataSet,
        )

    def OnHoughMatching(self, event):
        dialog = wx.MessageDialog(
            self,
            "Not yet implemented \n" "in wxPython",
            "Classical Indexation Method",
            wx.OK,
        )
        dialog.ShowModal()
        dialog.Destroy()

    def OnRadialRecognition(self, event):
        dialog = wx.MessageDialog(
            self,
            "Not yet implemented \n" "in wxPython",
            "Radial Recognition Method",
            wx.OK,
        )
        dialog.ShowModal()
        dialog.Destroy()

    def OnPlot_2ThetaChi(self, event):
        """
        Plot 2theta-chi representation of data peaks list for manual indexation
        """

        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print(
            "len(self.current_exp_spot_index_list)",
            len(self.current_exp_spot_index_list),
        )

        #         nb_exp_spots_data = len(self.data_theta)
        #
        #         index_to_select = np.take(self.current_exp_spot_index_list,
        #                                       np.arange(nb_exp_spots_data))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (
            self.data_pixX[self.current_exp_spot_index_list],
            self.data_pixY[self.current_exp_spot_index_list],
        )
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        #         self.StorageDict=None
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.DataPlot_filename,
        )

        if len(self.current_exp_spot_index_list) == 0:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        print(
            "AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters
        )

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
        self.indexation_parameters["DataToIndex"][
            "current_exp_spot_index_list"
        ] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"][
            "ClassicalIndexation_Tabledist"
        ] = None

        print(
            "self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
            self.indexation_parameters["DataToIndex"]["data_theta"],
        )
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters[
            "current_processedgrain"
        ] = self.current_processedgrain
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
        StorageDict["dict_Rot"] = DictLT.dict_Rot
        StorageDict["dict_Materials"] = DictLT.dict_Materials

        # Open manual indextion Board
        self.picky = ManualIndexFrame(
            self,
            -1,
            self.DataPlot_filename,
            data=self.data,
            data_XY=self.data_XY,
            data_2thetachi=self.data[:2],
            kf_direction=self.kf_direction,
            Size=SIZE_PLOTTOOLS,
            Params_to_simulPattern=None,
            indexation_parameters=self.indexation_parameters,
            StorageDict=StorageDict,
            DataSetObject=self.DataSet,
        )

        self.picky.Show(True)

    def OnPlot_Gnomon(self, event):
        """
        Plot gnomonic representation of data peaks list for manual indexation
        """

        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print(
            "len(self.current_exp_spot_index_list)",
            len(self.current_exp_spot_index_list),
        )

        #         nb_exp_spots_data = len(self.data_theta)
        #
        #         index_to_select = np.take(self.current_exp_spot_index_list,
        #                                       np.arange(nb_exp_spots_data))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (
            self.data_pixX[self.current_exp_spot_index_list],
            self.data_pixY[self.current_exp_spot_index_list],
        )
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        #         self.StorageDict=None
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.select_gnomonx = self.data_gnomonx[self.current_exp_spot_index_list]
        self.select_gnomony = self.data_gnomony[self.current_exp_spot_index_list]

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.DataPlot_filename,
        )

        if len(self.current_exp_spot_index_list) == 0:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        print(
            "AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters
        )

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
        self.indexation_parameters["DataToIndex"][
            "current_exp_spot_index_list"
        ] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"][
            "ClassicalIndexation_Tabledist"
        ] = None

        print(
            "self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
            self.indexation_parameters["DataToIndex"]["data_theta"],
        )
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters[
            "current_processedgrain"
        ] = self.current_processedgrain
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
        StorageDict["dict_Rot"] = DictLT.dict_Rot
        StorageDict["dict_Materials"] = DictLT.dict_Materials

        #         print "self.data_XY", self.data_XY
        # Open manual indextion Board
        self.picky = ManualIndexFrame(
            self,
            -1,
            self.DataPlot_filename,
            data=self.data,
            data_XY=self.data_XY,
            data_2thetachi=self.data[:2],
            kf_direction=self.kf_direction,
            datatype="gnomon",
            Size=SIZE_PLOTTOOLS,
            Params_to_simulPattern=None,
            indexation_parameters=self.indexation_parameters,
            StorageDict=StorageDict,
            DataSetObject=self.DataSet,
        )

        self.picky.Show(True)

    def OnPlot_Pixels(self, event):
        """
        Plot detector X,Y pixel representation of data peaks list for manual indexation
        """
        # select spots accroding to previous indexation
        if self.data_theta is None:
            self.OpenDefaultData()

        self.current_exp_spot_index_list = self.getAbsoluteIndices_Non_Indexed_Spots_()

        print(
            "len(self.current_exp_spot_index_list)",
            len(self.current_exp_spot_index_list),
        )

        #         nb_exp_spots_data = len(self.data_theta)
        #
        #         index_to_select = np.take(self.current_exp_spot_index_list,
        #                                       np.arange(nb_exp_spots_data))

        self.select_theta = self.data_theta[self.current_exp_spot_index_list]
        self.select_chi = self.data_chi[self.current_exp_spot_index_list]
        self.select_I = self.data_I[self.current_exp_spot_index_list]
        self.select_pixX = self.data_pixX[self.current_exp_spot_index_list]
        self.select_pixY = self.data_pixY[self.current_exp_spot_index_list]
        self.select_dataXY = (
            self.data_pixX[self.current_exp_spot_index_list],
            self.data_pixY[self.current_exp_spot_index_list],
        )
        #         self.select_dataXY = self.data_XY[index_to_select]
        #         CCDdetectorparameters
        #         self.StorageDict=None
        self.data_pixXY = self.data_pixX, self.data_pixY

        self.select_gnomonx = self.data_gnomonx[self.current_exp_spot_index_list]
        self.select_gnomony = self.data_gnomony[self.current_exp_spot_index_list]

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.DataPlot_filename,
        )

        if len(self.current_exp_spot_index_list) == 0:
            wx.MessageBox("There are no more spots left to be indexed now !", "INFO")
            return

        #         ClassicalIndexationBoard(self, -1, 'Classical Indexation Board :%s' % self.DataPlot_filename)

        # AllDataToIndex
        #         self.indexation_parameters['AllDataToIndex'] is already set
        #         self.indexation_parameters ={}
        print(
            "AllDataToIndex in dict: ", "AllDataToIndex" in self.indexation_parameters
        )

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
        self.indexation_parameters["DataToIndex"][
            "current_exp_spot_index_list"
        ] = copy.copy(self.current_exp_spot_index_list)
        self.indexation_parameters["DataToIndex"][
            "ClassicalIndexation_Tabledist"
        ] = None

        print(
            "self.indexation_parameters['DataToIndex']['data_theta'] = self.select_theta",
            self.indexation_parameters["DataToIndex"]["data_theta"],
        )
        self.indexation_parameters["dict_Rot"] = self.dict_Rot
        self.indexation_parameters[
            "current_processedgrain"
        ] = self.current_processedgrain
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
        StorageDict["dict_Rot"] = DictLT.dict_Rot
        StorageDict["dict_Materials"] = DictLT.dict_Materials
        # Open manual indextion Board
        self.picky = ManualIndexFrame(
            self,
            -1,
            self.DataPlot_filename,
            data=self.data,
            data_XY=self.data_XY,
            data_2thetachi=self.data[:2],
            kf_direction=self.kf_direction,
            datatype="pixels",
            Size=SIZE_PLOTTOOLS,
            Params_to_simulPattern=None,
            indexation_parameters=self.indexation_parameters,
            StorageDict=StorageDict,
            DataSetObject=self.DataSet,
        )

        self.picky.Show(True)

    def OnDetectorCalibration(self, event):
        """
        Method launching Calibration Board
        """
        self.OnOpenPeakList(event)

        print("self.DataPlot_filename", self.DataPlot_filename)
        # CalibrationFile = 'Cu_near_28May08_0259.peaks'
        # self.CalibrationFile = 'Ge_blanc.peaks' # peaks file
        # self.CalibrationFile = self.DataPlot_filename.split('.')[0]+'.peaks' # peaks file from fit2D

        # prefix_filename, extension_filename = self.DataPlot_filename.split('.')
        prefix_filename = self.DataPlot_filename.rsplit(".", 1)[0]

        if prefix_filename[:6] == "peaks_":
            self.CalibrationFile = prefix_filename[6:] + ".peaks"
        elif prefix_filename[:4] == "dat_":
            self.CalibrationFile = prefix_filename[4:] + ".dat"
        else:
            self.CalibrationFile = self.DataPlot_filename

        print("Calibrating with file: %s" % self.CalibrationFile)

        starting_param = self.defaultParam
        print("Starting param", starting_param)

        initialParameter = {}
        initialParameter["CCDParam"] = starting_param
        initialParameter["detectordiameter"] = self.detectordiameter
        initialParameter["CCDLabel"] = self.CCDLabel
        initialParameter["filename"] = self.CalibrationFile
        initialParameter["dirname"] = "/home/micha/LaueTools/Examples/Ge"

        print("initialParameter when launching calibration", initialParameter)

        self.calibframe = DC.MainCalibrationFrame(
            self,
            -1,
            "Detector Calibration Board",
            initialParameter,
            file_peaks=initialParameter["filename"],
            pixelsize=self.pixelsize,
            dim=self.framedim,
            kf_direction=self.kf_direction,
            fliprot=self.fliprot,
            starting_param=initialParameter["CCDParam"],
        )

        self.calibframe.Show(True)

    def OnCliquesFinding(self, event):
        """
        Method launching Cliques Finding  Board

        """
        if not self.ClassicalIndexation_Tabledist:
            print("hello in OnCliquesFinding")

            if self.data_theta is None:
                self.OpenDefaultData()
                self.CreateSpotDB()

            if 1:
                self.current_exp_spot_index_list = (
                    self.getAbsoluteIndices_Non_Indexed_Spots_()
                )
            else:
                self.current_exp_spot_index_list = np.arange(len(self.data_theta))

        #            select_theta = LaueToolsframe.data_theta[LaueToolsframe.current_exp_spot_index_list]
        #            select_chi = LaueToolsframe.data_chi[LaueToolsframe.current_exp_spot_index_list]
        # print select_theta
        # print select_chi
        #            listcouple = np.array([select_theta, select_chi]).T
        #            Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)
        # nbspots = len(Tabledistance)

        CliquesFindingBoard(
            None, -1, "Cliques Finding Board :%s" % self.DataPlot_filename
        )

    def OnRecognitionParam(self, event):
        return True

    def OnEditMatrix(self, event):
        self.MatrixEditor = MatrixEditor_Dialog(
            self, -1, "Create/Read/Save/Load/Convert Orientation Matrix"
        )
        self.MatrixEditor.Show(True)

    def OnEditUBMatrix(self, event):
        self.UBMatrixEditor = UBMatrixEditor_Dialog(
            self, -1, "UB Matrix Editor and Board"
        )
        self.UBMatrixEditor.Show(True)

    def EnterMatrix(self, evt):
        """
        open matrix entry board in check orientation process
        """
        helptstr = "Enter Matrix elements : \n [[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]]"
        helptstr += " Or list of Matrices"
        dlg = wx.TextEntryDialog(
            self, helptstr, "Orientation Matrix elements Entry for Matching Check"
        )

        _param = "[[1.,0,0],[0, 1,0],[0, 0,1]]"
        dlg.SetValue(_param)

        # OR
        dlg.SetToolTipString("please enter UB not UB.B0 (or list of UB's)")

        if dlg.ShowModal() == wx.ID_OK:
            paramraw = str(dlg.GetValue())
            import re

            listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", paramraw)
            #             print "listval", listval
            listelem = []
            for elem in listval:
                try:
                    val = float(elem)
                    listelem.append(val)
                except ValueError:
                    continue

            nbval = len(listelem)
            #             print "nbval", nbval

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

            initlength = len(DictLT.dict_Rot)
            for k, mat in enumerate(ListMatrices):
                mname = inputmatrixname + "%d" % k
                DictLT.dict_Rot[mname] = mat
            #                 self.crystalparampanel.comboMatrix.Append(mname)
            print("len dict", len(DictLT.dict_Rot))

            #             self.crystalparampanel.comboMatrix.SetSelection(initlength)

            # update with the first input matrix
            self.UBmatrix_toCheck = ListMatrices

            dlg.Destroy()

            nbmatrices = len(ListMatrices)

            return nbmatrices

    def Enterkey_material(self):
        helptstr = "Enter Material, structure or Element Label (example: Cu, UO2,)"

        dlg = wx.TextEntryDialog(
            self, helptstr, "Material and Crystallographic Structure Entry"
        )

        _param = "Ge"
        dlg.SetValue(_param)
        if dlg.ShowModal() == wx.ID_OK:
            key_material = str(dlg.GetValue())

            # check
            if key_material in DictLT.dict_Materials:
                self.key_material = key_material

            dlg.Destroy()

    def EnterEnergyMax(self):
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
        
        # TODO: compute std
        """

        ANGLETOLERANCE = 0.5
        RESOLUTIONANGSTROM = False

        AngRes = matchingrate.Angular_residues_np(
            UBmatrix_toCheck,
            2.0 * self.select_theta,
            self.select_chi,
            key_material=self.key_material,
            emax=self.emax,
            ResolutionAngstrom=RESOLUTIONANGSTROM,
            ang_tol=ANGLETOLERANCE,
            detectorparameters=self.indexation_parameters,
        )

        print("AngRes", AngRes)

        if AngRes is None:
            return

        (allres, resclose, nbclose, nballres, mean_residue, max_residue) = AngRes

        stats_properformat = [nbclose, nballres, np.std(allres)]

        return stats_properformat

    def PlotandRefineSolution(self):
        #         self.UBmatrix_toCheck

        StorageDict = {}
        StorageDict["mat_store_ind"] = 0
        StorageDict["Matrix_Store"] = []
        StorageDict["dict_Rot"] = DictLT.dict_Rot
        StorageDict["dict_Materials"] = DictLT.dict_Materials

        #         RecognitionResultCheckBox(self, -1,
        #                                               'Classical Indexation Solutions',
        #                                                 stats_properformat,
        #                                                 self.data,
        #                                                 self.rough_tolangle,
        #                                                 self.fine_tolangle,
        #                                                 key_material=self.key_material,
        #                                                 emax=self.energy_max,
        #                                                 ResolutionAngstrom=self.ResolutionAngstrom,
        #                                                 kf_direction=self.kf_direction,
        #                                                 datatype=datatype,
        #                                                 data_2thetachi=self.data_2thetachi,
        #                                                 data_XY=self.data_XY,
        #                                                 ImageArray=self.ImageArray,
        #                                                 CCDdetectorparameters=self.indexation_parameters,
        #                                                 IndexationParameters=self.indexation_parameters,
        #                                                 StorageDict=self.StorageDict,
        #                                                 DataSetObject=self.DataSet
        #                                                 )

        # display "statistical" results
        RRCBClassical = RecognitionResultCheckBox(
            self,
            -1,
            "Screening Distances Indexation Solutions",
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
            DataSetObject=self.DataSet,
        )

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

        (
            spotindex,
            grain_index,
            tth,
            chi,
            posX,
            posY,
            intensity,
            H,
            K,
            L,
            Energy,
        ) = Data.T

        Columns = [
            spotindex,
            grain_index,
            tth,
            chi,
            posX,
            posY,
            intensity,
            H,
            K,
            L,
            Energy,
        ]

        nbspots = len(spotindex)

        datatooutput = np.transpose(np.array(Columns))
        datatooutput = np.round(datatooutput, decimals=7)

        header = "# Spots Summary of: %s\n" % (self.filename)
        header += "# File created at %s with indexingSpotsSet.py\n" % (time.asctime())
        #         header += '# Number of indexed spots: %d\n' % nbindexedspots
        #         header += '# Number of unindexed spots: %d\n' % nbunindexedspots
        header += "# Number of spots: %d\n" % nbspots

        header += (
            "#spot_index grain_index 2theta Chi Xexp Yexp intensity h k l Energy\n"
        )
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
            latticeparams = DictLT.dict_Materials[key_material][1]
            B0matrix = CP.calc_B_RR(latticeparams)
            outputfile.write(str(B0matrix) + "\n")
            outputfile.write("#deviatoric strain (10-3 unit)\n")
            outputfile.write(
                str(self.dict_grain_devstrain[grain_index] * 1000.0) + "\n"
            )

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

    def OnSaveIndexationResultsfile(self, event, filename=None):
        """
        Save results of indexation and refinement procedures
        .res  :  summary file
        .cor  :  non indexed spots list
        
        TODO: export this ASCII write function in readwriteASCII module
        """
        if filename is None:
            filename = self.DataPlot_filename
        #
        #         header += '#  File created at %s with LaueToolsGUI.py\n' % (time.asctime())
        #
        #         comments_to_add = ''
        #         comments = 0
        #         nb_of_spots = 0
        #         data = ''
        #         for line in str(self.control.GetValue()).splitlines()[2:]:  # two lines header
        #
        #             if line[0] != '#' and comments == 0:  # line is made of data
        #                 listdata = line.split(',')
        #                 expindex = int(listdata[0])
        #                 twtheta = float(listdata[1])
        #                 chi = float(listdata[2])
        #                 pixX = float(listdata[3])
        #                 pixY = float(listdata[4])
        #                 intens = float(listdata[5])
        #                 # indexed spots case
        #                 if len(listdata) == 12:
        #                     nb_of_spots += 1
        #                     energy = float(listdata[9])
        #                     h = int(listdata[6])
        #                     k = int(listdata[7])
        #                     l = int(listdata[8])
        #                     Grain_index = int(listdata[10])
        #
        #                     data += '%d   %.06f   %.06f   %.06f   %.06f   %.06f   %s   %s   %s   %.06f  %d\n' % (expindex,
        #                                                                                                 twtheta, chi,
        #                                                                                                 pixX, pixY, intens,
        #                                                                                                 h, k, l, energy, Grain_index)
        #             # line is made of comments
        #             else:
        #                 comments = 1
        #                 comments_to_add = comments_to_add + line + '\n'
        #
        #         header += '#  Total Spots Number: %d\n' % nb_of_spots
        #         header += '#  Spot#    2theta    chi    pixX    pixY   intensity h k l energy grainindex\n'
        #
        #         textfile.write(header)
        #         textfile.write(data)
        #         textfile.write(comments_to_add)
        #
        #         textfile.close()

        wx.MessageBox(
            "Fit results have been written in folder %s" % self.dirname, "INFO"
        )

        self.DataSet.writeFileSummary(filename, self.dirname)

        #         if len(filename.split('.')) == 2:
        #             pre, ext = filename.split('.')
        #             filename = pre + '.cor'

        prefixfilename = filename.rsplit(".", 1)[0]

        filename = prefixfilename + ".cor"

        print("self.DataSet.pixelsize", self.DataSet.pixelsize)
        self.DataSet.writecorFile_unindexedSpots(
            corfilename=filename, dirname=self.dirname
        )

    # --- ---------------- Simulation Functions
    def Creating_Grains_parametric(self, event):
        """
        Method launching polycrystal simulation Board

        """

        # opening simulation parameters board

        # old WAY ------------------------------------
        #        self.CurrentdialogGrainCreation_cont = parametric_Grain_Dialog(self, -1,
        #                                                                       'Polygrains parametric definition for Laue Simulation')
        #
        #        self.SelectGrains_parametric = self.CurrentdialogGrainCreation_cont.SelectGrains
        #        self.CurrentdialogGrainCreation_cont.Show(True)
        # old WAY ------------------------------------

        # NEW WAY --------------
        initialParameters = {}
        initialParameters["CalibrationParameters"] = self.defaultParam
        initialParameters["prefixfilenamesimul"] = "dummy_"
        initialParameters["indexsimulation"] = self.indexsimulation
        initialParameters["ExperimentalData"] = None
        if self.data_theta is not None:
            initialParameters["ExperimentalData"] = (
                2 * self.data_theta,
                self.data_chi,
                self.data_pixX,
                self.data_pixY,
                self.data_I,
            )
        initialParameters["pixelsize"] = self.pixelsize
        initialParameters["framedim"] = self.framedim

        print("initialParameters", initialParameters)
        self.CurrentdialogGrainCreation_cont = LSGUI.parametric_Grain_Dialog3(
            self,
            -1,
            "Polygrains parametric definition for Laue Simulation",
            initialParameters,
        )

        self.SelectGrains_parametric = self.CurrentdialogGrainCreation_cont.SelectGrains
        self.CurrentdialogGrainCreation_cont.Show(True)

        return True

    def Edit_String_SimulData(
        self, data=([0], [0], [0], [0], [0], [0], [""], 0, [0.0, 0.0, 0.0, 0.0, 0.0], 0)
    ):
        """
        Writes in LaueToolsframe.control
        data =(list_twicetheta,
                list_chi,
                list_energy,
                list_Miller,
                list_posX,
                list_posY,
                ListName,
                nb of(parent) grains,
                calibration parameters list,
                total nb of grains)
        TODO: put the calibration parameters
        """
        nb_total_grains = data[9]
        lines = "Simulation Data from LAUE Pattern Program v1.0 2009 \n"
        lines += "Total number of grains : %s\n" % int(nb_total_grains)
        lines += "spot# h k l E 2theta chi X Y\n"
        nb = data[7]
        if type(nb) == type(5):  # multigrains simulations without transformations
            nb_grains = data[7]
            TWT, CHI, ENE, MIL, XX, YY = data[:6]
            NAME = data[6]
            calib = data[8]

            for index_grain in range(nb_grains):
                nb_of_simulspots = len(TWT[index_grain])
                startgrain = "#G %d\t%s\t%d\n" % (
                    index_grain,
                    NAME[index_grain],
                    nb_of_simulspots,
                )

                lines += startgrain
                # print nb_of_simulspots
                for data_index in range(nb_of_simulspots):
                    linedata = "%d\t%d\t%d\t%d\t%.5f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
                        data_index,
                        MIL[index_grain][data_index][0],
                        MIL[index_grain][data_index][1],
                        MIL[index_grain][data_index][2],
                        ENE[index_grain][data_index],
                        TWT[index_grain][data_index],
                        CHI[index_grain][data_index],
                        XX[index_grain][data_index],
                        YY[index_grain][data_index],
                    )
                    lines += linedata
            lines += "#calibration parameters\n"
            for param in calib:
                lines += "# %s\n" % param
            # print "in edit",lines
            self.control.SetValue(lines)
            return True

        if type(nb) == type(
            [[0, 5], [1, 10]]
        ):  # nb= list of [grain index, nb of transforms]
            print("nb in Edit_String_SimulData", nb)
            gen_i = 0
            TWT, CHI, ENE, MIL, XX, YY = data[:6]
            NAME = data[6]
            calib = data[8]
            for grain_ind in range(len(nb)):  # loop over parent grains
                for tt in range(nb[grain_ind][1]):
                    nb_of_simulspots = len(TWT[gen_i])
                    startgrain = "#G %d\t%s\t%d\t%d\n" % (
                        grain_ind,
                        NAME[grain_ind],
                        tt,
                        nb_of_simulspots,
                    )

                    lines += startgrain
                    for data_index in range(nb_of_simulspots):
                        linedata = "%d\t%d\t%d\t%d\t%.5f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
                            data_index,
                            MIL[gen_i][data_index][0],
                            MIL[gen_i][data_index][1],
                            MIL[gen_i][data_index][2],
                            ENE[gen_i][data_index],
                            TWT[gen_i][data_index],
                            CHI[gen_i][data_index],
                            XX[gen_i][data_index],
                            YY[gen_i][data_index],
                        )
                        lines += linedata
                    gen_i += 1

            lines += "#calibration parameters\n"
            for param in calib:
                lines += "# %s\n" % param
            # print "in edit",lines
            self.control.SetValue(lines)
            return True

    # --- ------------- spots database
    def SaveNonIndexedSpots(self, evt):
        dlg = wx.TextEntryDialog(
            self, "Enter new filename (*.cor):", "Peaks List .cor Filename Entry"
        )

        if dlg.ShowModal() == wx.ID_OK:
            filename = str(dlg.GetValue())

            finalfilename = filename
            if filename.endswith(".cor"):
                finalfilename = filename[:-4]

            fullpath = os.path.join(self.dirname, finalfilename)

            self.SaveFileCorNonIndexedSpots(outputfilename=fullpath)

        dlg.Destroy()

    #         self.SaveFileCorNonIndexedSpots(os.path.join(self.dirname, 'nonindexed'))

    def SaveFileCorNonIndexedSpots(self, outputfilename=None):
        if outputfilename is None:
            outputfilename
            pre, ext = self.DataPlot_filename.strip(".")
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

        detectorparameters = []

        # comment
        strgrains = ["Remaining Non indexed spots of %s" % self.DataPlot_filename]

        IOLT.writefile_cor(
            outputfilename,
            Twicetheta,
            Chi,
            posx,
            posy,
            dataintensity,
            param=self.defaultParam + [self.pixelsize],
            initialfilename=self.DataPlot_filename,
            comments=strgrains,
        )

    def OpenCorfile(self, filename):
        """
        .cor file is either:
        2theta chi I
        or(better)
        2theta chi pixX pixY I

        creates or updates self.Current_peak_data(all columns of .cor file)
        creates or updates self.data_theta, self.data_chi, self.data_I

        reads detector parameters and set defaultParam according to them
        returns self.data_theta, self.data_chi, self.data_pixX, self.data_pixY, self.data_I
        """
        (
            self.Current_peak_data,
            self.data_theta,
            self.data_chi,
            self.data_pixX,
            self.data_pixY,
            self.data_I,
            calib,
            self.CCDCalibDict,
        ) = IOLT.readfile_cor(filename, output_CCDparamsdict=True)

        print("CCDCalibDict in OpenCorfile", self.CCDCalibDict)

        self.CheckCCDCalibParameters()

        print("self.detectordiameter", self.detectordiameter)

        pixelsize_fromfile = IOLT.getpixelsize_from_corfile(filename)
        if pixelsize_fromfile:
            self.pixelsize = pixelsize_fromfile

        if calib is not None:
            self.defaultParam = calib

        self.data_XY = (self.data_pixX, self.data_pixY)

        # Spots List to index object ----------------------
        print("self.CCDCalibDict", self.CCDCalibDict)
        print("self.pixelsize", self.pixelsize)

        # DataSetObject init
        self.DataSet = ISS.spotsset()
        # get spots scattering angles,X,Y positions from .cor file
        self.DataSet.importdatafromfile(filename)
        self.DataSet.pixelsize = self.pixelsize
        # ----------------------------------

        print("self.DataSet", self.DataSet)

        return (
            self.data_theta,
            self.data_chi,
            self.data_pixX,
            self.data_pixY,
            self.data_I,
        )

    def setAllDataToIndex_Dict(self):
        self.indexation_parameters = {}
        self.indexation_parameters["AllDataToIndex"] = {}
        self.indexation_parameters["AllDataToIndex"]["data_theta"] = self.data_theta
        self.indexation_parameters["AllDataToIndex"]["data_chi"] = self.data_chi
        self.indexation_parameters["AllDataToIndex"]["data_pixX"] = self.data_pixX
        self.indexation_parameters["AllDataToIndex"]["data_pixY"] = self.data_pixY
        self.indexation_parameters["AllDataToIndex"]["data_I"] = self.data_I
        self.indexation_parameters["AllDataToIndex"]["data_gnomonX"] = self.data_gnomonx
        self.indexation_parameters["AllDataToIndex"]["data_gnomonY"] = self.data_gnomony
        self.indexation_parameters["AllDataToIndex"]["data_XY"] = (
            self.data_pixX,
            self.data_pixY,
        )
        self.indexation_parameters["AllDataToIndex"]["data_2thetachi"] = (
            2 * self.data_theta,
            self.data_chi,
        )

        nbspotstoindex = len(self.data_theta)
        self.indexation_parameters["AllDataToIndex"]["NbSpotsToindex"] = nbspotstoindex
        self.indexation_parameters["AllDataToIndex"]["IndexedFlag"] = np.zeros(
            nbspotstoindex
        )
        self.indexation_parameters["AllDataToIndex"]["absolutespotindex"] = np.array(
            np.arange(nbspotstoindex), dtype=np.int
        )

    def create_read_corfile(self):
        """
        create cor file from peaks list .dat file and calibration file .det
        AND read it for further indexation needs
        
        """

        print("self.DataPlot_filename", self.DataPlot_filename)

        if not self.DataPlot_filename.startswith("dat_"):
            wx.MessageBox(
                'Not implemented yet when filename does not start with "dat_"... Sorry!',
                "Info",
            )
            return

        (twicetheta, chi, dataintensity, data_x, data_y) = F2TC.Compute_data2thetachi(
            self.DataPlot_filename[4:-4] + ".dat",
            (0, 1, 3),
            1,  # 2 for centroid intensity, 3 for integrated  intensity
            sorting_intensity="yes",
            param=self.defaultParam,
            signgam=SIGN_OF_GAMMA,
            pixelsize=self.pixelsize,
            kf_direction=self.kf_direction,
        )

        prefix, extension = self.DataPlot_filename.rsplit(".", 1)

        prefixfilename_corfile = prefix

        IOLT.writefile_cor(
            prefixfilename_corfile,
            twicetheta,
            chi,
            data_x,
            data_y,
            dataintensity,
            sortedexit=0,
            param=self.defaultParam + [self.pixelsize],
            initialfilename=self.DataPlot_filename,
        )  # check sortedexit = 0 or 1 to have decreasing intensity sorted data
        print("%s has been updated" % (prefixfilename_corfile + ".cor"))
        print("new CCD calib parameter : %s" % str(self.defaultParam))
        # a .cor file is now created

        self.DataPlot_filename = prefixfilename_corfile + ".cor"
        # WARNING: it will be read in the next "if" clause

        self.OpenCorfile(self.DataPlot_filename)

        # compute Gnomonic projection
        dataselected = IOLT.createselecteddata(
            (self.data_theta * 2, self.data_chi, self.data_I),
            np.arange(len(self.data_theta)),
            len(self.data_theta),
        )[0]
        self.data_gnomonx, self.data_gnomony = IIM.ComputeGnomon_2(dataselected)

        # ---------------------------------------------
        self.filename = self.DataPlot_filename
        self.SetTitle()  # Update the window title with the new filename
        # create DB spots(dictionary)
        self.CreateSpotDB()

        self.setAllDataToIndex_Dict()

        # -----------------------------------------------
        textfile = open(os.path.join(self.dirname, self.DataPlot_filename), "r")
        String_in_File_Data = textfile.read()
        self.control.SetValue(String_in_File_Data)
        textfile.close()

        self.current_processedgrain = 0
        self.last_orientmatrix_fromindexation = {}
        self.last_Bmatrix_fromindexation = {}
        self.last_epsil_fromindexation = {}

    def OpenDefaultData(self):
        """
        Open default data for quick test if user has'nt yet loaded data
        """
        DEFAULTFILE = "dat_Ge0001.cor"
        defaultdatafile = os.path.join(self.dirname, "Examples", "Ge", DEFAULTFILE)

        print("self.detectordiameter in OpenDefaultData()", self.detectordiameter)

        self.dirname = os.path.split(os.path.abspath(defaultdatafile))[0]

        self.OpenCorfile(defaultdatafile)
        self.DataPlot_filename = DEFAULTFILE
        dataselected = IOLT.createselecteddata(
            (self.data_theta * 2, self.data_chi, self.data_I),
            np.arange(len(self.data_theta)),
            len(self.data_theta),
        )[0]
        self.data_gnomonx, self.data_gnomony = IIM.ComputeGnomon_2(dataselected)

        # create DB spots(dictionary)
        self.CreateSpotDB()
        self.setAllDataToIndex_Dict()

        textfile = open(os.path.join(self.dirname, self.DataPlot_filename), "r")
        String_in_File_Data = textfile.read()
        self.control.SetValue(String_in_File_Data)
        textfile.close()

        self.current_processedgrain = 0

    def select_exp_spots(self):
        """
        select spots to be indexed
        
        set self.non_indexed_spots as array of absolute index in peaks experimental list of non indexed spots
        """
        # select default data for test
        if self.data_theta is None:
            self.OpenDefaultData()

        if 1:
            self.current_exp_spot_index_list = (
                self.getAbsoluteIndices_Non_Indexed_Spots_()
            )  # plot non indexed spots
        else:
            self.current_exp_spot_index_list = np.arange(
                len(self.data_theta)
            )  # plot whole data

        self.non_indexed_spots = np.array(self.current_exp_spot_index_list)

        if len(self.non_indexed_spots) == 0:
            wx.MessageBox("There are no spots to be indexed now !", "INFO")

    def set_params_manualindexation(self):

        indexation_parameters = {}
        indexation_parameters["kf_direction"] = self.kf_direction
        # duplicate
        indexation_parameters["DataPlot_filename"] = self.DataPlot_filename
        indexation_parameters["Filename"] = self.DataPlot_filename

        indexation_parameters["dict_Materials"] = self.dict_Materials

        #         indexation_parameters['data_theta'] = self.data_theta
        #         indexation_parameters['data_chi'] = self.data_chi
        #         indexation_parameters['data_I'] = self.data_I
        indexation_parameters[
            "current_exp_spot_index_list"
        ] = self.current_exp_spot_index_list
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

        StorageDict = {}
        StorageDict["mat_store_ind"] = self.mat_store_ind
        StorageDict["Matrix_Store"] = self.Matrix_Store
        StorageDict["dict_Rot"] = self.dict_Rot
        StorageDict["dict_Materials"] = self.dict_Materials

        self.indexation_parameters = indexation_parameters
        self.StorageDict = StorageDict

    def CreateSpotDB(self):
        """
        create a spots Database
        """
        # For now, only DB created for a single file...

        # dictionary of exp spots
        for k in range(len(self.data_theta)):
            self.indexed_spots[k] = [
                k,  # index of experimental spot in .cor file
                self.data_theta[k] * 2.0,
                self.data_chi[k],  # 2theta, chi coordinates
                self.data_pixX[k],
                self.data_pixY[k],  # pixel coordinates
                self.data_I[k],  # intensity
                0,
            ]  # 0 means non indexed yet

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

    def Update_DataToIndex_Dict(self, data_list, grain_index):
        """ 
        udpate dictionary of exp. spots taking into account data_list values(indexation results)
        """
        #         print "************ \n\n self.indexation_parameters",self.indexation_parameters

        data_Miller, data_Energy, list_indexspot = data_list

        print("list_indexspot in Update_DataToIndex_Dict", list_indexspot)

        # AllDataToIndex = self.indexation_parameters["AllDataToIndex"]
        # DataToIndex = self.indexation_parameters["DataToIndex"]

        return

        # #         self.indexation_parameters['AllDataToIndex']['IndexedFlag']

        # # updating exp spot dict.
        # singleindices = []

        # for k in range(len(list_indexspot)):

        #     absoluteindex = self.current_exp_spot_index_list[list_indexspot[k]]
        #     if not singleindices.count(list_indexspot[k]):
        #         singleindices.append(list_indexspot[k])
        #         #                 spotenergy = data_Energy[k] # wrong
        #         if self.DataSet.indexed_spots_dict[absoluteindex][-1] == 1:
        #             # spot belong to the grain grain_index
        #             spotenergy = self.DataSet.indexed_spots_dict[absoluteindex][7]

        #         self.indexed_spots[absoluteindex] = self.indexed_spots[absoluteindex][
        #             :-1
        #         ] + [data_Miller[k], spotenergy, grain_index, 1]

    def Update_DB_fromIndexation(self, data_list):
        """ 
        udpate dictionary of exp. spots taking into account data_list values(indexation results)
        """
        #         print "************ \n\n self.indexation_parameters",self.indexation_parameters

        data_Miller, data_Energy, list_indexspot = data_list

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
                    spotenergy = self.DataSet.indexed_spots_dict[absoluteindex][7]

                self.indexed_spots[absoluteindex] = self.indexed_spots[absoluteindex][
                    :-1
                ] + [data_Miller[k], spotenergy, self.current_processedgrain, 1]

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
                texttoshow += (
                    "%d,  %.06f ,%.06f,  %.06f ,%.06f,  %.06f,  %d \n" % tuple(val)
                )
            elif len(self.indexed_spots[k]) == 10:
                val = (
                    self.indexed_spots[k][:6]
                    + self.indexed_spots[k][6].tolist()
                    + self.indexed_spots[k][7:]
                )
                texttoshow += (
                    "%d,  %.06f ,%.06f,  %.06f ,%.06f,  %.06f,  %d, %d, %d,  %.06f, %d, %d \n"
                    % tuple(val)
                )

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
        for par, value in zip(
            ["dd", "xcen", "ycen", "xbet", "xgam"], self.defaultParam
        ):
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
            startgrain = "#G %d\t%s\t%d\n" % (
                index_grain,
                NAME[index_grain],
                nb_of_simulspots,
            )

            lines += startgrain
            print(nb_of_simulspots)
            for data_index in range(nb_of_simulspots):
                linedata = "%d\t%d\t%d\t%d\t%.3f\t%.4f\t%.4f\n" % (
                    data_index,
                    MIL[index_grain][data_index][0],
                    MIL[index_grain][data_index][1],
                    MIL[index_grain][data_index][2],
                    ENE[index_grain][data_index],
                    TWT[index_grain][data_index],
                    CHI[index_grain][data_index],
                )
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
            wildcard=wcd,
        )

    def defaultFileDialogOptionsImage(self):
        """ Return a dictionary with file dialog options that can be
            used in both the save file dialog as well as in the open
            file dialog. """
        wcd0 = (
            "MAR CCD image(*.mccd)|*.mccd|mar tiff(*.tiff)|*.tiff|mar tif(*.tif)|*.tif|"
        )
        wcd0 += (
            "Princeton(*.spe)|*.spe|Frelon(*.edf)|*.edf|hdf5(*.h5)|*.h5|All files(*)|*"
        )

        try:
            wcd = DictLT.getwildcardstring(self.CCDLabel)
        except:
            wcd = wcd0

        return dict(
            message="Choose an Image File", defaultDir=self.dirname, wildcard=wcd
        )

    def askUserForFilename(self, **dialogOptions):
        """
        provide a dialog to browse the folders and files
        """
        dialog = wx.FileDialog(self, **dialogOptions)
        if dialog.ShowModal() == wx.ID_OK:
            userProvidedFilename = True
            # self.filename = dialog.GetFilename()
            # #self.dirname = dialog.GetDirectory()

            allpath = dialog.GetPath()
            print(allpath)
            self.dirname, self.filename = os.path.split(allpath)

        else:
            userProvidedFilename = False
        dialog.Destroy()
        return userProvidedFilename

    def OnDocumentationpdf(self, event):

        LaueToolsProjectFolder = os.path.abspath(os.curdir)

        #         print "LaueToolsProjectFolder", LaueToolsProjectFolder

        pdffile_adress = "file://%s" % os.path.join(
            LaueToolsProjectFolder, "Documentation", "latex", "LaueTools.pdf"
        )

        webbrowser.open(pdffile_adress)

    def OnDocumentationhtml(self, event):

        LaueToolsProjectFolder = os.path.abspath(os.curdir)

        #         print "LaueToolsProjectFolder", LaueToolsProjectFolder

        html_file_address = "file://%s" % os.path.join(
            LaueToolsProjectFolder, "Documentation", "html", "index.html"
        )

        webbrowser.open(html_file_address)

    def OnTutorial(self, event):
        dialog = wx.MessageDialog(
            self,
            "Not yet implemented \n"
            "in wxPython\n\n See \nhttps://sourceforge.net/userapps/mediawiki/jsmicha/index.php?title=Main_Page",
            "Tutorial",
            wx.OK,
        )
        dialog.ShowModal()
        dialog.Destroy()

    def OnSerieIndexation(self, event):
        dialog = wx.MessageDialog(
            self, "Not yet implemented \n" "in wxPython", "OnSerieIndexation", wx.OK
        )
        dialog.ShowModal()
        dialog.Destroy()

    def OnMapAnalysis(self, event):
        dialog = wx.MessageDialog(
            self, "Not yet implemented \n" "in wxPython", "OnMapAnalysis", wx.OK
        )
        dialog.ShowModal()
        dialog.Destroy()

    def OnAbout(self, event):
        """
        about lauetools and license
        """
        description = """LaueTools is toolkit for white beam x-ray microdiffraction 
        Laue Pattern analysis. It allows Simulation & Indexation procedures written 
        in python by Jean-Sebastien MICHA \n  micha@esrf.fr\n\n and Odile Robach at the French CRG-IF beamline
         \n at BM32(European Synchrotron Radiation Facility).

        %s %s
        Revision number: %s

        Support and help in developing this package:
        https://sourceforge.net/projects/lauetools/
        """ % (
            MONTH,
            YEAR,
            REV,
        )
        f = open("License", "r")
        lines = f.readlines()
        mylicense = ""
        for line in lines:
            mylicense += line
        f.close()

        info = wx.AboutDialogInfo()

        info.SetIcon(
            wx.Icon(os.path.join("icons", "transmissionLaue.png"), wx.BITMAP_TYPE_PNG)
        )
        info.SetName("LaueTools")
        info.SetVersion(
            "%s %s %s\nRevision Number (subversion): %s" % (DAY, MONTH, YEAR, REV)
        )
        info.SetDescription(description)
        info.SetCopyright("(C) 2009-%s Jean-Sebastien Micha" % YEAR)
        info.SetWebSite("http://www.esrf.eu/UsersAndScience/Experiments/CRG/BM32/")
        info.SetLicence(mylicense)
        info.AddDeveloper("Jean-Sebastien Micha")
        info.AddDocWriter("Jean-Sebastien Micha")
        # info.AddArtist('The Tango crew')
        info.AddTranslator("Jean-Sebastien Micha")
        wx.AboutBox(info)

    def OnExit(self, event):
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

        wx.Dialog.__init__(self, parent, _id, title, size=(600, 200))

        panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(590, 190), pos=(5, 5))

        self.parent = parent

        self.samefolder = wx.RadioButton(panel, -1, "Images folder", (25, 15))
        self.userdefinedrelfolder = wx.RadioButton(
            panel, -1, "path relative to lauetools", (25, 55)
        )
        self.userdefinedabsfolder = wx.RadioButton(panel, -1, "absolute path", (25, 95))
        self.samefolder.SetValue(True)

        self.relativepathname = wx.TextCtrl(
            panel, -1, "./analysis", (250, 50), (340, -1)
        )
        self.abspathname = wx.TextCtrl(
            panel, -1, "/myhome/myfolder", (250, 90), (340, -1)
        )

        wx.Button(panel, 1, "Accept", (15, 150), (90, 40))
        self.Bind(wx.EVT_BUTTON, self.OnAccept, id=1)

        wx.Button(panel, 2, "Quit", (150, 150), (90, 40))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=2)

    def OnAccept(self, evt):
        if self.samefolder.GetValue():
            self.writefolder = "."
        elif self.userdefinedrelfolder.GetValue():
            self.writefolder = os.path.join(
                DictLT.LAUETOOLSFOLDER, str(self.relativepathname.GetValue())[2:]
            )
        elif self.userdefinedabsfolder.GetValue():
            self.writefolder = str(self.abspathname.GetValue())

        self.parent.writefolder = self.writefolder

        self.Close()

    def OnQuit(self, evt):
        self.Close()


# --- -------------------- general Laue Geometry settings
class SetGeneralLaueGeometry(wx.Dialog):
    """
    Dialog Class to set  general Laue Geometry
    """

    def __init__(self, parent, _id, title):

        wx.Dialog.__init__(self, parent, _id, title, size=(400, 200))

        self.parent = parent

        txt = wx.StaticText(self, -1, "Choose Laue Geometry")
        font = wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        txt.SetFont(font)
        txt.SetForegroundColour((255, 0, 0))

        #         self.sb = self.CreateStatusBar()

        initialGeo = DICT_LAUE_GEOMETRIES[parent.kf_direction]

        self.combogeo = wx.ComboBox(
            self,
            -1,
            str(initialGeo),
            size=(-1, 40),
            choices=["Top Reflection (2theta=90)", "Transmission"],
            style=wx.CB_READONLY,
        )

        self.combogeo.Bind(wx.EVT_COMBOBOX, self.OnChangeGeom)

        txtinfo = wx.StaticText(self, -1, "Infos :  ")

        self.comments = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(300, 50))
        self.comments.SetValue(str(DictLT.DICT_LAUE_GEOMETRIES_INFO[initialGeo]))

        btna = wx.Button(self, 1, "Accept", size=(150, 40))
        btna.Bind(wx.EVT_BUTTON, self.OnAccept)
        btna.SetDefault()

        btnc = wx.Button(self, 2, "Cancel", size=(100, 40))
        btnc.Bind(wx.EVT_BUTTON, self.OnQuit)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(btna, 1)
        hbox.Add(btnc, 1)

        h2box = wx.BoxSizer(wx.HORIZONTAL)
        h2box.Add(txtinfo, 0)
        h2box.Add(self.comments, 0)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt, 0, wx.EXPAND)
        vbox.Add(self.combogeo, 0, wx.EXPAND)
        vbox.Add(hbox, 0)
        vbox.Add(wx.StaticText(self, -1, ""), 0)
        vbox.Add(h2box, 0, wx.EXPAND)

        self.SetSizer(vbox)

    def OnChangeGeom(self, evt):
        focus_geom = self.combogeo.GetValue()

        #         print "Laue Geometry info :", DICT_LAUE_GEOMETRIES_INFO[focus_geom]
        self.comments.SetValue(str(DictLT.DICT_LAUE_GEOMETRIES_INFO[focus_geom]))

    #         self.sb.SetStatusText(str(DICT_LAUE_GEOMETRIES_INFO[focus_geom]))

    def OnAccept(self, evt):
        LaueGeometry = self.combogeo.GetValue()

        if LaueGeometry == "Transmission":
            kf_direction = "X>0"
        elif LaueGeometry == "Top Reflection (2theta=90)":
            kf_direction = "Z>0"

        self.parent.kf_direction = kf_direction

        print("Laue geometry set to: %s" % LaueGeometry)
        print("kf_direction set to: %s" % kf_direction)

        #         wx.MessageBox('Laue geometry set to: %s\nkf_direction set to: %s' % \
        #                             (LaueGeometry, kf_direction),
        #                             'Info')

        self.Close()

    def OnQuit(self, evt):
        self.Close()


# --- ---------------------  CLIQUES board
class CliquesFindingBoard(wx.Frame):
    """
    Class to display GUI for cliques finding
    """

    def __init__(self, parent, _id, title):

        wx.Frame.__init__(self, parent, _id, title, size=(400, 300))

        self.panel = wx.Panel(
            self, -1, style=wx.SIMPLE_BORDER, size=(590, 390), pos=(5, 5)
        )

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        title1 = wx.StaticText(self.panel, -1, "Parameters", (15, 15))
        title1.SetFont(font3)

        wx.StaticText(self.panel, -1, "Angular tolerance(deg): ", (15, 45))
        self.AT = wx.TextCtrl(self.panel, -1, "0.5", (200, 40), (60, -1))
        wx.StaticText(self.panel, -1, "(for distance recognition): ", (15, 60))

        wx.StaticText(self.panel, -1, "Intense spots set Size(ISSS): ", (15, 90))
        self.nbspotmax = wx.SpinCtrl(
            self.panel, -1, "30", (200, 85), (60, -1), min=3, max=200
        )

        wx.StaticText(self.panel, -1, "List of spots index", (15, 125))
        self.spotlist = wx.TextCtrl(self.panel, -1, "-1", (120, 125), (250, -1))
        wx.StaticText(self.panel, -1, "(from 0 to ISSS-1)", (15, 145))

        wx.StaticText(self.panel, -1, "Show # found Cliques ", (15, 160))
        self.nbshowcliques = wx.SpinCtrl(
            self.panel, -1, "10", (200, 160), (60, -1), min=3, max=15
        )

        self.showplotBox = wx.CheckBox(self.panel, 55, "Plot Best result", (15, 190))
        self.showplotBox.SetValue(True)
        wx.StaticText(
            self.panel,
            -1,
            "Current File: %s" % self.parent.DataPlot_filename,
            (130, 15),
        )

        wx.Button(self.panel, 1, "Search", (40, 225), (110, 40))
        self.Bind(wx.EVT_BUTTON, self.OnSearch, id=1)

        wx.Button(self.panel, 2, "Quit", (230, 225), (110, 40))
        self.Bind(wx.EVT_BUTTON, self.OnQuit, id=2)

        self.Show(True)
        self.Centre()

    def OnSearch(self, event):

        spot_list = self.spotlist.GetValue()

        print("spot_list in OnSearch", spot_list)
        if spot_list[0] != "-":
            print("coucou")
            if spot_list[0] == "[":
                print("coucou2")
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
        nbshow = self.nbshowcliques.GetValue()
        print("Clique finding for Nodes :%s" % Nodes)
        res_cliques = GraGra.give_bestclique(
            self.parent.DataPlot_filename,
            nbmax_probed,
            ang_tol,
            nodes=Nodes,
            col_Int=-1,
            verbose=nbshow,
        )

        print("BEST CLIQUES **************")
        print(res_cliques)
        print("***************************")

    def OnQuit(self, event):
        self.Close()


# --- ---------------  Matrix Editor
class MatrixEditor_Dialog(wx.Frame):
    """
    class to handle edition of matrices
    """

    def __init__(self, parent, _id, title):

        wx.Frame.__init__(self, parent, _id, title, size=(600, 700))

        self.dirname = os.getcwd()

        self.parent = parent
        # variables
        self.modify = False
        self.last_name_saved = ""
        self.last_name_stored = ""
        self.replace = False

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(590, 690), pos=(5, 5))

        self.rbeditor = wx.RadioButton(
            panel, -1, "Text Editor Input", (25, 10), style=wx.RB_GROUP
        )
        wx.StaticText(panel, -1, "[[#,#,#],[#,#,#],[#,#,#]]", (40, 25))

        self.text = wx.TextCtrl(
            panel,
            1000,
            "",
            pos=(50, 45),
            size=(250, 85),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )
        self.text.SetFocus()
        self.text.Bind(wx.EVT_TEXT, self.OnTextChanged, id=1000)
        self.text.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        self.rbelem = wx.RadioButton(panel, -1, "Matrix Elements Input", (25, 150))
        self.rbeditor.SetValue(True)
        self.mat_a11 = wx.TextCtrl(panel, -1, "1", (20, 180), size=(80, -1))
        self.mat_a12 = wx.TextCtrl(panel, -1, "0", (120, 180), size=(80, -1))
        self.mat_a13 = wx.TextCtrl(panel, -1, "0", (220, 180), size=(80, -1))
        self.mat_a21 = wx.TextCtrl(panel, -1, "0", (20, 210), size=(80, -1))
        self.mat_a22 = wx.TextCtrl(panel, -1, "1", (120, 210), size=(80, -1))
        self.mat_a23 = wx.TextCtrl(panel, -1, "0", (220, 210), size=(80, -1))
        self.mat_a31 = wx.TextCtrl(panel, -1, "0", (20, 240), size=(80, -1))
        self.mat_a32 = wx.TextCtrl(panel, -1, "0", (120, 240), size=(80, -1))
        self.mat_a33 = wx.TextCtrl(panel, -1, "1", (220, 240), size=(80, -1))

        self.CurrentMat = "Identity"
        List_Rot_name = list(self.parent.dict_Rot.keys())
        List_Rot_name.remove("Identity")
        List_Rot_name.sort()
        self.list_of_Rot = ["Identity"] + List_Rot_name

        rr = wx.StaticText(panel, -1, "Rotation", (400, 15))
        rr.SetFont(font3)
        wx.StaticText(panel, -1, "Axis angles(long., lat) or vector", (320, 45))
        self.longitude = wx.TextCtrl(panel, -1, "0.0", (330, 70), size=(60, -1))
        self.latitude = wx.TextCtrl(panel, -1, "0.0", (410, 70), size=(60, -1))
        wx.StaticText(panel, -1, "in deg.", (485, 75))
        self.axisrot = wx.TextCtrl(panel, -1, "[1, 0,0]", (390, 100), size=(60, -1))
        wx.StaticText(panel, -1, "Rotation angle", (370, 130))
        self.anglerot = wx.TextCtrl(panel, -1, "0.0", (390, 150), size=(60, -1))
        wx.StaticText(panel, -1, "in deg.", (485, 155))

        buttoncomputemat_1 = wx.Button(
            panel, 555, "Compute", pos=(330, 180), size=(80, 25)
        )
        wx.StaticText(panel, -1, "from axis angles", (420, 185))
        buttoncomputemat_1.Bind(wx.EVT_BUTTON, self.OnComputeMatrix_axisangles, id=555)
        buttoncomputemat_2 = wx.Button(
            panel, 556, "Compute", pos=(330, 210), size=(80, 25)
        )
        wx.StaticText(panel, -1, "from axis vector", (420, 215))
        buttoncomputemat_2.Bind(wx.EVT_BUTTON, self.OnComputeMatrix_axisvector, id=556)

        buttonread = wx.Button(panel, 101, "Look", pos=(20, 300), size=(60, 25))
        # buttonread.SetFont(font3)
        buttonread.Bind(wx.EVT_BUTTON, self.OnLookMatrix, id=101)
        self.comboRot = wx.ComboBox(
            panel, 6, "Identity", (100, 300), choices=self.list_of_Rot
        )
        self.comboRot.Bind(wx.EVT_COMBOBOX, self.EnterComboRot, id=6)

        buttonsave = wx.Button(panel, 102, "Save", pos=(20, 340), size=(60, 25))
        # buttonsave.SetFont(font3)
        buttonsave.Bind(wx.EVT_BUTTON, self.OnSaveFile, id=102)
        wx.StaticText(panel, -1, "in", (110, 345))
        self.filenamesave = wx.TextCtrl(panel, -1, "", (160, 340), size=(100, 25))
        wx.StaticText(panel, -1, "(on hard disk)", (260, 345))

        buttonstore = wx.Button(panel, 103, "Store", pos=(20, 380), size=(60, 25))
        # buttonstore.SetFont(font3)
        buttonstore.Bind(wx.EVT_BUTTON, self.OnStoreMatrix, id=103)
        wx.StaticText(panel, -1, "in", (110, 385))
        self.filenamestore = wx.TextCtrl(panel, -1, "", (160, 380), size=(100, 25))
        wx.StaticText(panel, -1, "(will appear in simulation matrix menu)", (260, 385))

        buttonload = wx.Button(panel, 104, "Load", pos=(20, 420), size=(60, 25))
        # buttonload.SetFont(font3)
        buttonload.Bind(wx.EVT_BUTTON, self.OnOpenMatrixFile, id=104)
        wx.StaticText(
            panel,
            -1,
            "Matrix from saved file in simple ASCII text editor format",
            (100, 425),
        )

        buttonXMASload = wx.Button(
            panel, 105, "Read XMAS", pos=(20, 460), size=(100, 25)
        )
        # buttonload.SetFont(font3)
        buttonXMASload.Bind(wx.EVT_BUTTON, self.OnLoadXMAS_INDfile, id=105)
        wx.StaticText(panel, -1, ".IND file", (130, 465))

        buttonconvert = wx.Button(panel, 106, "Convert", pos=(200, 460), size=(70, 25))
        # buttonload.SetFont(font3)
        buttonconvert.Bind(wx.EVT_BUTTON, self.OnConvert, id=106)
        wx.StaticText(panel, -1, "XMAS to LaueTools lab. frame", (290, 465))

        buttonconvert2 = wx.Button(panel, 107, "Convert", pos=(200, 510), size=(70, 25))
        buttonconvert2.Bind(wx.EVT_BUTTON, self.OnConvertlabtosample, id=107)
        wx.StaticText(panel, -1, "LaueTools: from lab. to sample frame", (290, 515))

        self.invconvert = wx.CheckBox(panel, 108, "inv.", (210, 535))
        self.invconvert.SetValue(False)

        buttonquit = wx.Button(panel, 6, "Quit", (150, 620), size=(120, 50))
        buttonquit.Bind(wx.EVT_BUTTON, self.OnQuit, id=6)

        self.StatusBar()

        # tooltips
        buttonquit.SetToolTipString("Quit editor")

    def StatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-5, -2, -1])

    def ToggleStatusBar(self, event):
        if self.statusbar.IsShown():
            self.statusbar.Hide()
        else:
            self.statusbar.Show()

    def OnTextChanged(self, event):
        self.modify = True
        event.Skip()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_INSERT:
            if not self.replace:
                self.statusbar.SetStatusText("INS", 2)
                self.replace = True
            else:
                self.statusbar.SetStatusText("", 2)
                self.replace = False
        event.Skip()

    def OnComputeMatrix_axisangles(self, evt):
        longitude = float(self.longitude.GetValue())
        latitude = float(self.latitude.GetValue())
        angle = float(self.anglerot.GetValue())
        deg = DEG
        x = np.cos(longitude * deg) * np.cos(latitude * deg)
        y = np.sin(longitude * deg) * np.cos(latitude * deg)
        z = np.sin(latitude * deg)
        matrix = GT.matRot(np.array([x, y, z]), angle)

        self.mat_a11.SetValue(str(matrix[0][0]))
        self.mat_a12.SetValue(str(matrix[0][1]))
        self.mat_a13.SetValue(str(matrix[0][2]))
        self.mat_a21.SetValue(str(matrix[1][0]))
        self.mat_a22.SetValue(str(matrix[1][1]))
        self.mat_a23.SetValue(str(matrix[1][2]))
        self.mat_a31.SetValue(str(matrix[2][0]))
        self.mat_a32.SetValue(str(matrix[2][1]))
        self.mat_a33.SetValue(str(matrix[2][2]))
        self.text.SetValue(str(matrix.tolist()))

    def OnComputeMatrix_axisvector(self, evt):
        Axis = str(self.axisrot.GetValue())
        angle = float(self.anglerot.GetValue())

        aa = Axis.split(",")
        x = float(aa[0][1:])
        y = float(aa[1])
        z = float(aa[2][:-1])
        matrix = GT.matRot(np.array([x, y, z]), angle)

        self.mat_a11.SetValue(str(matrix[0][0]))
        self.mat_a12.SetValue(str(matrix[0][1]))
        self.mat_a13.SetValue(str(matrix[0][2]))
        self.mat_a21.SetValue(str(matrix[1][0]))
        self.mat_a22.SetValue(str(matrix[1][1]))
        self.mat_a23.SetValue(str(matrix[1][2]))
        self.mat_a31.SetValue(str(matrix[2][0]))
        self.mat_a32.SetValue(str(matrix[2][1]))
        self.mat_a33.SetValue(str(matrix[2][2]))
        self.text.SetValue(str(matrix.tolist()))

    def OnSaveFile(self, event):
        """
        Saves the matrix in ASCII editor or 9 entried elements on Hard Disk
        """
        self.last_name_saved = self.filenamesave.GetValue()

        if self.last_name_saved:

            try:
                if self.rbeditor.GetValue():
                    f = open(self.last_name_saved, "w")
                    text = self.text.GetValue()
                    f.write(text)
                    f.close()
                else:

                    m11 = float(self.mat_a11.GetValue())
                    m12 = float(self.mat_a12.GetValue())
                    m13 = float(self.mat_a13.GetValue())

                    m21 = float(self.mat_a21.GetValue())
                    m22 = float(self.mat_a22.GetValue())
                    m23 = float(self.mat_a23.GetValue())

                    m31 = float(self.mat_a31.GetValue())
                    m32 = float(self.mat_a32.GetValue())
                    m33 = float(self.mat_a33.GetValue())

                    _file = open(self.last_name_saved, "w")
                    text = (
                        "[[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f]]"
                        % (m11, m12, m13, m21, m22, m23, m31, m32, m33)
                    )
                    _file.write(text)
                    _file.close()

                self.statusbar.SetStatusText(
                    os.path.basename(self.last_name_saved) + " saved", 0
                )
                self.modify = False
                self.statusbar.SetStatusText("", 1)

                fullname = os.path.join(os.getcwd(), self.last_name_saved)
                wx.MessageBox("Matrix saved in %s" % fullname, "INFO")

            except IOError as error:

                dlg = wx.MessageDialog(self, "Error saving file\n" + str(error))
                dlg.ShowModal()
        else:
            print("No name input")

    def OnStoreMatrix(self, event):
        """
        Stores the matrix from the ASCII editor or the 9 entried elements in main list of orientation matrix for further simulation
        """
        self.last_name_stored = self.filenamestore.GetValue()

        if self.last_name_stored:

            if not self.rbeditor.GetValue():  # read 9 elements
                m11 = float(self.mat_a11.GetValue())
                m12 = float(self.mat_a12.GetValue())
                m13 = float(self.mat_a13.GetValue())

                m21 = float(self.mat_a21.GetValue())
                m22 = float(self.mat_a22.GetValue())
                m23 = float(self.mat_a23.GetValue())

                m31 = float(self.mat_a31.GetValue())
                m32 = float(self.mat_a32.GetValue())
                m33 = float(self.mat_a33.GetValue())

                self.parent.dict_Rot[self.last_name_stored] = [
                    [m11, m12, m13],
                    [m21, m22, m23],
                    [m31, m32, m33],
                ]

            else:  # read ASCII editor
                text = str(self.text.GetValue())
                tu = text.replace("[", "").replace("]", "")
                ta = tu.split(",")
                to = [float(elem) for elem in ta]
                self.parent.dict_Rot[self.last_name_stored] = [to[:3], to[3:6], to[6:]]

            self.statusbar.SetStatusText(
                os.path.basename(self.last_name_stored) + " stored", 0
            )

        else:
            print("No name input")

    def OnOpenMatrixFile(self, event):
        wcd = "All files(*)|*|Matrix files(*.mat)|*.mat"
        _dir = os.getcwd()
        open_dlg = wx.FileDialog(
            self,
            message="Choose a file",
            defaultDir=_dir,
            defaultFile="",
            wildcard=wcd,
            style=wx.OPEN | wx.CHANGE_DIR,
        )
        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()

            try:
                _file = open(path, "r")
                text = _file.readlines()
                _file.close()

                if self.text.GetLastPosition():
                    self.text.Clear()
                strmat = ""
                for line in text:
                    strmat += line[:-1]
                self.text.WriteText(strmat + "]")
                self.last_name_saved = path
                self.statusbar.SetStatusText("", 1)
                self.modify = False

            except IOError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except UnicodeDecodeError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

        open_dlg.Destroy()

    def EnterComboRot(self, event):
        """
        in matrix editor
        """
        item = event.GetSelection()
        self.CurrentMat = self.list_of_Rot[item]
        event.Skip()

    def OnLookMatrix(self, event):
        matrix = self.parent.dict_Rot[self.CurrentMat]
        print("%s is :" % self.CurrentMat)
        print(matrix)

        self.mat_a11.SetValue(str(matrix[0][0]))
        self.mat_a12.SetValue(str(matrix[0][1]))
        self.mat_a13.SetValue(str(matrix[0][2]))
        self.mat_a21.SetValue(str(matrix[1][0]))
        self.mat_a22.SetValue(str(matrix[1][1]))
        self.mat_a23.SetValue(str(matrix[1][2]))
        self.mat_a31.SetValue(str(matrix[2][0]))
        self.mat_a32.SetValue(str(matrix[2][1]))
        self.mat_a33.SetValue(str(matrix[2][2]))
        self.text.SetValue(str(matrix))

    def OnLoadXMAS_INDfile(self, event):
        """
        old and may be obsolete indexation file reading procedure ?!
        """
        wcd = "All files(*)|*|Indexation files(*.ind)|*.ind|StrainRefined files(*.str)|*.str"
        _dir = os.getcwd()
        open_dlg = wx.FileDialog(
            self,
            message="Choose a file",
            defaultDir=_dir,
            defaultFile="",
            wildcard=wcd,
            style=wx.OPEN | wx.CHANGE_DIR,
        )
        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()

            try:
                _file = open(path, "r")
                alllines = _file.read()
                _file.close()
                listlines = alllines.splitlines()

                if path.split(".")[-1] in ("IND", "ind"):

                    posmatrix = -1
                    lineindex = 0
                    for line in listlines:
                        if line[:8] == "matrix h":
                            print("Get it!: ", line)
                            posmatrix = lineindex
                            break
                        lineindex += 1

                    if posmatrix != -1:
                        print("firstrow", listlines[posmatrix + 1])
                        firstrow = np.array(
                            listlines[posmatrix + 1].split(), dtype=float
                        )
                        secondrow = np.array(
                            listlines[posmatrix + 2].split(), dtype=float
                        )
                        thirdrow = np.array(
                            listlines[posmatrix + 3].split(), dtype=float
                        )
                        mat = np.array([firstrow, secondrow, thirdrow])
                        # indexation file contains matrix
                        # such as U G = Q  G must be normalized ??
                    else:
                        print("Did not find matrix in this ind file!...")

                    if self.text.GetLastPosition():
                        self.text.Clear()

                    self.text.WriteText(str(mat.tolist()))
                    self.last_name_saved = path
                    self.statusbar.SetStatusText("", 1)
                    self.modify = False

                elif path.split(".")[-1] in ("STR", "str"):
                    posmatrix = -1
                    lineindex = 0
                    for line in listlines:
                        if line[:17] == "coordinates of a*":
                            print("Get it!: ", line)
                            posmatrix = lineindex
                            break
                        lineindex += 1

                    if posmatrix != -1:
                        print("firstrow", listlines[posmatrix + 1])
                        firstrow = np.array(
                            listlines[posmatrix + 1].split(), dtype=float
                        )
                        secondrow = np.array(
                            listlines[posmatrix + 2].split(), dtype=float
                        )
                        thirdrow = np.array(
                            listlines[posmatrix + 3].split(), dtype=float
                        )
                        mat = np.array([firstrow, secondrow, thirdrow])
                        # contrary to .ind file, matrix is transposed!!
                        #   UB G  = Q  with G normalized
                        mat = np.transpose(mat)
                    else:
                        print("Did not find matrix in this str file!...")

                    if self.text.GetLastPosition():
                        self.text.Clear()

                    self.text.WriteText(str(mat.tolist()))
                    self.last_name_saved = path
                    self.statusbar.SetStatusText("", 1)
                    self.modify = False

            except IOError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except UnicodeDecodeError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

        open_dlg.Destroy()

    def OnConvert(self, event):
        """
        #TODO: to check, obsolete ?
        """
        text = str(self.text.GetValue())
        tu = text.replace("[", "").replace("]", "")
        ta = tu.split(",")
        to = [float(elem) for elem in ta]
        matrix = np.array([to[:3], to[3:6], to[6:]])

        dlg = wx.MessageDialog(
            self,
            "New matrix will be displated in Text Editor Input\nConversion will use current calibration, namely xbet: %.4f \n  \nDo you want to continue ?"
            % self.parent.defaultParam[3],
        )
        if dlg.ShowModal() == wx.ID_OK:

            event.Skip()

            # newmatrix = F2TC.matxmas_to_OrientMatrix(matrix, LaueToolsframe.defaultParam)
            newmatrix = FXL.convert_fromXMAS_toLaueTools(
                matrix, 5.6575, anglesample=40.0, xbet=self.parent.defaultParam[3]
            )
            print("Matrix read in editor")
            print(matrix.tolist())
            print("Matrix as converted to Lauetools by F2TC")
            print(newmatrix.tolist())
            # matrix UB normalized
            # must be multiplied by lattic param*

            self.rbeditor.SetValue(True)
            self.text.SetValue(str(newmatrix.tolist()))
        else:
            # TODO: advise how to set new calibration parameter(in calibration menu ?)
            event.Skip()

    def OnConvertlabtosample(self, evt):
        """
        qs= R ql    q expressed in sample frame(xs,ys,zs) = R * ql with ql being q expressed in lab frame(x,y,z)
        G=ha*+kb*+lc*
        ql = UB * G with UB orientation and strain matrix  ie UB columns are a*, b*, c* expressed in x,y,z frame
        qs = R * UB * G
        R*UB in the orientation matrix in sample frame ie columns are a*, b*,c* expressed in xs,ys,zs frame

        Gs = R Gl    G expressed in sample frame(xs,ys,zs)  =  R * Gl  with Gl being G expressed in lab frame

        qs = R*UB*invR  Gs means that R*UB*invR is the orientation matrix . From a*,b*,c* in xs,ys,zs to q in xs,ys,zs
        """
        text = str(self.text.GetValue())
        tu = text.replace("[", "").replace("]", "")
        ta = tu.split(",")
        to = [float(elem) for elem in ta]
        UB = np.array([to[:3], to[3:6], to[6:]])

        if not self.invconvert.GetValue():  # direct conversion
            anglesample = 40.0 * DEG
        else:  # inverse conversion
            anglesample = -40.0 * DEG

        Rot = np.array(
            [
                [np.cos(anglesample), 0, np.sin(anglesample)],
                [0, 1, 0],
                [-np.sin(anglesample), 0, np.cos(anglesample)],
            ]
        )
        invRot = np.linalg.inv(Rot)
        UBs = np.dot(Rot, UB)
        print("UB as read in editor")
        print(UB.tolist())
        print(
            "UB converted in lauetools sample frame(From G=ha*+kb*+lc* with a*,b* and c* expressed in lab frame to q expressed in sample frame)"
        )
        print(UBs.tolist())
        print("UB converted in XMAS-like sample frame")
        print(np.dot(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), UBs).tolist())
        print(
            "UB converted in lauetools sample frame(From G=ha*+kb*+lc* with a*,b* and c* expressed in sample frame to q expressed in sample frame)"
        )
        print(np.dot(UBs, invRot).tolist())

        # matrix UB normalized
        # must be multiplied by lattic param*

        self.rbeditor.SetValue(True)
        self.text.SetValue(str(UBs.tolist()))

    def OnQuit(self, event):
        dlg = wx.MessageDialog(
            self,
            'To use stored Matrices in simulation boards that do not appear, click on "refresh choices" button before.',
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.Close()
            event.Skip()
        else:
            event.Skip()


# --- ---------------------   UB Matrix Editor
class UBMatrixEditor_Dialog(wx.Frame):
    def __init__(self, parent, _id, title):

        wx.Frame.__init__(self, parent, _id, title, size=(720, 1000))

        self.dirname = os.getcwd()
        self.parent = parent
        # variables
        self.modify = False
        self.last_name_saved = ""
        self.last_name_stored = ""
        self.replace = False

        self.CurrentMat = "Default"
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        self.panel = wx.Panel(
            self, -1, style=wx.SIMPLE_BORDER, size=(715, 995), pos=(5, 5)
        )

        dr = wx.StaticText(self.panel, -1, "Direct Space", (60, 10))
        dr.SetFont(font3)

        self.rbeditor = wx.RadioButton(
            self.panel, -1, "Text Editor Input", (25, 40), style=wx.RB_GROUP
        )
        wx.StaticText(self.panel, -1, "[[#,#,#],[#,#,#],[#,#,#]]", (25, 60))
        self.text = wx.TextCtrl(
            self.panel,
            1000,
            "",
            pos=(25, 85),
            size=(250, 90),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )
        self.text.SetFocus()
        self.text.Bind(wx.EVT_TEXT, self.OnTextChanged, id=1000)
        self.text.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        self.rbelem = wx.RadioButton(self.panel, -1, "Matrix Elements Input", (25, 190))

        self.mat_a11 = wx.TextCtrl(self.panel, -1, "1", (20, 220), size=(80, -1))
        self.mat_a12 = wx.TextCtrl(self.panel, -1, "0", (120, 220), size=(80, -1))
        self.mat_a13 = wx.TextCtrl(self.panel, -1, "0", (220, 220), size=(80, -1))
        self.mat_a21 = wx.TextCtrl(self.panel, -1, "0", (20, 250), size=(80, -1))
        self.mat_a22 = wx.TextCtrl(self.panel, -1, "1", (120, 250), size=(80, -1))
        self.mat_a23 = wx.TextCtrl(self.panel, -1, "0", (220, 250), size=(80, -1))
        self.mat_a31 = wx.TextCtrl(self.panel, -1, "0", (20, 280), size=(80, -1))
        self.mat_a32 = wx.TextCtrl(self.panel, -1, "0", (120, 280), size=(80, -1))
        self.mat_a33 = wx.TextCtrl(self.panel, -1, "1", (220, 280), size=(80, -1))

        self.rblatticeparam_direct = wx.RadioButton(
            self.panel, -1, "Lattice parameter Input", (25, 320)
        )
        wx.StaticText(self.panel, -1, "a", (40, 350))
        self.a = wx.TextCtrl(self.panel, -1, "1", (20, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "b", (120, 350))
        self.b = wx.TextCtrl(self.panel, -1, "1", (100, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "c", (200, 350))
        self.c = wx.TextCtrl(self.panel, -1, "1", (180, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "Angst.", (260, 385))
        wx.StaticText(self.panel, -1, "alpha", (20, 410))
        self.alpha = wx.TextCtrl(self.panel, -1, "90", (20, 440), size=(60, -1))
        wx.StaticText(self.panel, -1, "beta", (100, 410))
        self.beta = wx.TextCtrl(self.panel, -1, "90", (100, 440), size=(60, -1))
        wx.StaticText(self.panel, -1, "gamma", (180, 410))
        self.gamma = wx.TextCtrl(self.panel, -1, "90", (180, 440), size=(60, -1))
        wx.StaticText(self.panel, -1, "deg.", (260, 445))

        # reciprocal part

        hshift = 350
        drs = wx.StaticText(self.panel, -1, "Reciprocal Space", (hshift + 60, 10))
        drs.SetFont(font3)

        self.rbeditors = wx.RadioButton(
            self.panel, -1, "Text Editor Input Bmatrix", (hshift + 25, 40)
        )
        wx.StaticText(self.panel, -1, "[[#,#,#],[#,#,#],[#,#,#]]", (hshift + 25, 60))
        self.texts = wx.TextCtrl(
            self.panel,
            1003,
            "",
            pos=(hshift + 25, 85),
            size=(250, 90),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )
        self.texts.SetFocus()
        self.texts.Bind(wx.EVT_TEXT, self.OnTextChanged, id=1003)
        self.texts.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        self.rbelems = wx.RadioButton(
            self.panel, -1, "BMatrix Elements Input", (hshift + 25, 190)
        )

        self.mat_a11s = wx.TextCtrl(
            self.panel, -1, "1", (hshift + 20, 220), size=(80, -1)
        )
        self.mat_a12s = wx.TextCtrl(
            self.panel, -1, "0", (hshift + 120, 220), size=(80, -1)
        )
        self.mat_a13s = wx.TextCtrl(
            self.panel, -1, "0", (hshift + 220, 220), size=(80, -1)
        )
        self.mat_a21s = wx.TextCtrl(
            self.panel, -1, "0", (hshift + 20, 250), size=(80, -1)
        )
        self.mat_a22s = wx.TextCtrl(
            self.panel, -1, "1", (hshift + 120, 250), size=(80, -1)
        )
        self.mat_a23s = wx.TextCtrl(
            self.panel, -1, "0", (hshift + 220, 250), size=(80, -1)
        )
        self.mat_a31s = wx.TextCtrl(
            self.panel, -1, "0", (hshift + 20, 280), size=(80, -1)
        )
        self.mat_a32s = wx.TextCtrl(
            self.panel, -1, "0", (hshift + 120, 280), size=(80, -1)
        )
        self.mat_a33s = wx.TextCtrl(
            self.panel, -1, "1", (hshift + 220, 280), size=(80, -1)
        )

        self.rblatticeparam_reciprocal = wx.RadioButton(
            self.panel, -1, "Lattice parameter Input", (hshift + 25, 320)
        )
        wx.StaticText(self.panel, -1, "a*", (hshift + 40, 350))
        self.astar = wx.TextCtrl(self.panel, -1, "1", (hshift + 20, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "b*", (hshift + 120, 350))
        self.bstar = wx.TextCtrl(
            self.panel, -1, "1", (hshift + 100, 380), size=(60, -1)
        )
        wx.StaticText(self.panel, -1, "c*", (hshift + 200, 350))
        self.cstar = wx.TextCtrl(
            self.panel, -1, "1", (hshift + 180, 380), size=(60, -1)
        )
        wx.StaticText(self.panel, -1, "1/Angst.", (hshift + 260, 385))
        wx.StaticText(self.panel, -1, "alpha*", (hshift + 20, 410))
        self.alphastar = wx.TextCtrl(
            self.panel, -1, "90", (hshift + 20, 440), size=(60, -1)
        )
        wx.StaticText(self.panel, -1, "beta*", (hshift + 100, 410))
        self.betastar = wx.TextCtrl(
            self.panel, -1, "90", (hshift + 100, 440), size=(60, -1)
        )
        wx.StaticText(self.panel, -1, "gamma*", (hshift + 180, 410))
        self.gammastar = wx.TextCtrl(
            self.panel, -1, "90", (hshift + 180, 440), size=(60, -1)
        )
        wx.StaticText(self.panel, -1, "deg.", (hshift + 260, 445))

        self.rbeditors.SetValue(True)

        buttonConvert = wx.Button(
            self.panel, 105, "Convert", pos=(150, 480), size=(400, 50)
        )
        # buttonload.SetFont(font3)
        buttonConvert.Bind(wx.EVT_BUTTON, self.OnConvert, id=105)

        wx.StaticText(self.panel, -1, "_" * 90, (25, 530))

        self.Build_list_of_B()
        vshift = 80
        buttonread = wx.Button(
            self.panel, 101, "Look", pos=(20, vshift + 500), size=(60, 25)
        )
        # buttonread.SetFont(font3)
        buttonread.Bind(wx.EVT_BUTTON, self.OnLookMatrix, id=101)
        self.comboRot = wx.ComboBox(
            self.panel, 6, "Identity", (100, vshift + 500), choices=self.list_of_Vect
        )
        self.comboRot.Bind(wx.EVT_COMBOBOX, self.EnterComboRot, id=6)
        wx.StaticText(self.panel, -1, "in Bmatrix", (300, vshift + 500 + 5))

        buttonsave = wx.Button(
            self.panel, 102, "Save", pos=(20, vshift + 540), size=(60, 25)
        )
        # buttonsave.SetFont(font3)
        buttonsave.Bind(wx.EVT_BUTTON, self.OnSaveFile, id=102)
        wx.StaticText(self.panel, -1, "Bmatrix in", (110, vshift + 545))
        self.filenamesave = wx.TextCtrl(
            self.panel, -1, "*.mat", (200, vshift + 540), size=(100, 25)
        )
        wx.StaticText(self.panel, -1, "(on hard disk)", (300, vshift + 545))

        buttonstore = wx.Button(
            self.panel, 103, "Store", pos=(20, vshift + 580), size=(60, 25)
        )
        # buttonstore.SetFont(font3)
        buttonstore.Bind(wx.EVT_BUTTON, self.OnStoreFile, id=103)
        wx.StaticText(self.panel, -1, "Bmatrix in", (110, vshift + 585))
        self.filenamestore = wx.TextCtrl(
            self.panel, -1, "", (200, vshift + 580), size=(100, 25)
        )
        wx.StaticText(
            self.panel,
            -1,
            "(will appear in a*,b*,c* simulation menu)",
            (300, vshift + 585),
        )

        buttonload = wx.Button(
            self.panel, 104, "Load", pos=(20, vshift + 620), size=(60, 25)
        )
        # buttonload.SetFont(font3)
        buttonload.Bind(wx.EVT_BUTTON, self.DoOpenFile, id=104)
        wx.StaticText(
            self.panel,
            -1,
            "Matrix from saved file in simple ASCII text editor format",
            (100, vshift + 625),
        )

        self.vshift2 = vshift + 535 + 60

        wx.StaticText(self.panel, -1, "_" * 90, (25, self.vshift2 + 50))

        st000 = wx.StaticText(
            self.panel, -1, "Crystal Unit Cell", (250, self.vshift2 + 70)
        )
        st000.SetFont(font3)

        """
        q =   Da U Dc B  G*
        Da deformation in lab frame(or compute from D expressed in sample frame):  Da = I + strain_a
        strain_a is general strain neither symetric nor antisymetric, contain pure strain + pure rigid body rotation

        Dc deformation in crystal frame  a*,b*,c* Dc = I + strain_c
        strain_c is general strain neither symetric nor antisymetric, contain pure strain + pure rigid body rotation

        U is orient matrix in lab frame
        B is Bmatrix whose each colum is a*,b*,c* expressed in lab frame. Bmatrix can contain rotation part
        ( can different from triangular up matrix)
        """

        self.stepx = 120
        postext = 90
        pos0x = 70

        wx.StaticText(self.panel, -1, "q   = ", (10, self.vshift2 + postext))
        wx.StaticText(self.panel, -1, "Da       .", (pos0x, self.vshift2 + postext))
        wx.StaticText(
            self.panel,
            -1,
            "U        .",
            (pos0x + 1 * self.stepx, self.vshift2 + postext),
        )
        wx.StaticText(
            self.panel,
            -1,
            "B        .",
            (pos0x + 2 * self.stepx, self.vshift2 + postext),
        )
        wx.StaticText(
            self.panel,
            -1,
            "Dc       .",
            (pos0x + 3 * self.stepx, self.vshift2 + postext),
        )
        wx.StaticText(
            self.panel, -1, "G*", (pos0x + 3 * self.stepx + 100, self.vshift2 + postext)
        )
        wx.StaticText(
            self.panel,
            -1,
            "Extinc",
            (pos0x + 3 * self.stepx + 150, self.vshift2 + postext),
        )

        self.poscombos = 120

        # combos of unit cell reference parameters
        self.DisplayCombosUnitCell()

        posv = (self.poscombos + self.vshift2) + 40

        butstoreCell = wx.Button(
            self.panel, -1, "Store Cell", pos=(20, posv), size=(80, 25)
        )
        butstoreCell.Bind(wx.EVT_BUTTON, self.OnStoreRefCell)
        wx.StaticText(self.panel, -1, "in", (110, posv + 5))
        self.namestore = wx.TextCtrl(self.panel, -1, "", (150, posv), size=(100, 25))
        wx.StaticText(
            self.panel,
            -1,
            "(Will appear in Elem list for classical indexation)",
            (280, posv + 5),
        )

        buttonquit = wx.Button(
            self.panel, -1, "Quit", (575, self.vshift2 + 180), size=(120, 100)
        )
        buttonquit.Bind(wx.EVT_BUTTON, self.OnQuit)

        self.StatusBar()

    def Build_list_of_B(self):
        # building list of choices of B matrix

        List_Vect_name = list(self.parent.dict_Vect.keys())
        List_Vect_name.remove("Default")
        List_Vect_name.sort()

        self.list_of_Vect = ["Default"] + List_Vect_name

    def DisplayCombosUnitCell(self):

        self.Build_list_of_B()

        List_U_name = list(self.parent.dict_Rot.keys())
        List_U_name.remove("Identity")
        List_U_name.sort()

        self.list_of_Rot = ["Identity"] + List_U_name

        List_Transform_name = list(self.parent.dict_Transforms.keys())
        List_Transform_name.remove("Identity")
        List_Transform_name.sort()

        self.list_of_Strain = ["Identity"] + List_Transform_name

        self.comboDa = wx.ComboBox(
            self.panel,
            -1,
            "Identity",
            (50, self.vshift2 + self.poscombos),
            choices=self.list_of_Strain,
            size=(100, -1),
        )

        self.comboU = wx.ComboBox(
            self.panel,
            -1,
            "Identity",
            (50 + self.stepx, self.vshift2 + self.poscombos),
            choices=self.list_of_Rot,
            size=(100, -1),
        )

        self.comboB = wx.ComboBox(
            self.panel,
            -1,
            "Identity",
            (50 + 2 * self.stepx, self.vshift2 + self.poscombos),
            choices=self.list_of_Vect,
            size=(100, -1),
        )

        self.comboDc = wx.ComboBox(
            self.panel,
            -1,
            "Identity",
            (50 + 3 * self.stepx, self.vshift2 + self.poscombos),
            choices=self.list_of_Strain,
            size=(100, -1),
        )

        self.comboExtinc = wx.ComboBox(
            self.panel,
            -1,
            "NoExtinction",
            (50 + 4 * self.stepx + 40, self.vshift2 + self.poscombos),
            choices=list(self.parent.dict_Extinc.keys()),
            size=(100, -1),
        )

    def OnStoreRefCell(self, evt):
        key_material = str(self.namestore.GetValue())

        # factor structure peak extinction
        struct_extinc = self.parent.dict_Extinc[self.comboExtinc.GetValue()]

        # UnitCellParameters is either [a,b,c,alpha,beta,gamma] or list of 4 matrices [Da,U,Dc,B]
        Da = self.parent.dict_Transforms[self.comboDa.GetValue()]
        U = self.parent.dict_Rot[self.comboU.GetValue()]
        B = self.parent.dict_Vect[self.comboB.GetValue()]
        Dc = self.parent.dict_Transforms[self.comboDc.GetValue()]

        UnitCellParameters = [Da, U, B, Dc]
        print("UnitCellParameters", UnitCellParameters)
        print("Stored key_material", [key_material, UnitCellParameters, struct_extinc])
        self.parent.dict_Materials[key_material] = [
            key_material,
            UnitCellParameters,
            struct_extinc,
        ]

    def StatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-5, -2, -1])

    def ToggleStatusBar(self, event):
        if self.statusbar.IsShown():
            self.statusbar.Hide()
        else:
            self.statusbar.Show()

    def OnTextChanged(self, event):
        self.modify = True
        event.Skip()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_INSERT:
            if not self.replace:
                self.statusbar.SetStatusText("INS", 2)
                self.replace = True
            else:
                self.statusbar.SetStatusText("", 2)
                self.replace = False
        event.Skip()

    def OnSaveFile(self, event):
        """
        Saves the matrix in ASCII editor or 9 entried elements on Hard Disk
        """
        self.last_name_saved = self.filenamesave.GetValue()

        if self.last_name_saved:

            try:
                if self.rbeditor.GetValue():
                    _file = open(self.last_name_saved, "w")
                    text = self.texts.GetValue()
                    _file.write(text)
                    _file.close()
                else:

                    m11 = float(self.mat_a11s.GetValue())
                    m12 = float(self.mat_a12s.GetValue())
                    m13 = float(self.mat_a13s.GetValue())

                    m21 = float(self.mat_a21s.GetValue())
                    m22 = float(self.mat_a22s.GetValue())
                    m23 = float(self.mat_a23s.GetValue())

                    m31 = float(self.mat_a31s.GetValue())
                    m32 = float(self.mat_a32s.GetValue())
                    m33 = float(self.mat_a33s.GetValue())

                    allm = (m11, m12, m13, m21, m22, m23, m31, m32, m33)

                    f = open(self.last_name_saved, "w")
                    text = (
                        "[[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f]]"
                        % allm
                    )
                    f.write(text)
                    f.close()

                self.statusbar.SetStatusText(
                    os.path.basename(self.last_name_saved) + " saved", 0
                )
                self.modify = False
                self.statusbar.SetStatusText("", 1)

                fullname = os.path.join(os.getcwd(), self.last_name_saved)
                wx.MessageBox("Matrix saved in %s" % fullname, "INFO")

            except IOError as error:

                dlg = wx.MessageDialog(self, "Error saving file\n" + str(error))
                dlg.ShowModal()
        else:
            print("No name input!!")

    def OnStoreFile(self, event):
        """
        Stores the matrix from the ASCII editor or the 9 entried elements in main list of orientation matrix for further simulation
        """
        self.last_name_stored = self.filenamestore.GetValue()

        if self.last_name_stored:

            if not self.rbeditor.GetValue():  # read 9 elements
                m11 = float(self.mat_a11s.GetValue())
                m12 = float(self.mat_a12s.GetValue())
                m13 = float(self.mat_a13s.GetValue())

                m21 = float(self.mat_a21s.GetValue())
                m22 = float(self.mat_a22s.GetValue())
                m23 = float(self.mat_a23s.GetValue())

                m31 = float(self.mat_a31s.GetValue())
                m32 = float(self.mat_a32s.GetValue())
                m33 = float(self.mat_a33s.GetValue())

                self.parent.dict_Vect[self.last_name_stored] = [
                    [m11, m12, m13],
                    [m21, m22, m23],
                    [m31, m32, m33],
                ]

            else:  # read ASCII editor
                text = str(self.texts.GetValue())
                tu = text.replace("[", "").replace("]", "")
                ta = tu.split(",")
                to = [float(elem) for elem in ta]
                self.parent.dict_Rot[self.last_name_stored] = [to[:3], to[3:6], to[6:]]

            self.statusbar.SetStatusText(
                os.path.basename(self.last_name_stored) + " stored", 0
            )
            self.DisplayCombosUnitCell()

        else:
            print("No name input !!!")

    def DoOpenFile(self, event):
        wcd = "All files(*)|*|Matrix files(*.mat)|*.mat"
        _dir = os.getcwd()
        open_dlg = wx.FileDialog(
            self,
            message="Choose a file",
            defaultDir=_dir,
            defaultFile="",
            wildcard=wcd,
            style=wx.OPEN | wx.CHANGE_DIR,
        )
        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()

            try:
                _file = open(path, "r")
                text = _file.readlines()
                _file.close()

                if self.texts.GetLastPosition():
                    self.texts.Clear()
                strmat = ""
                for line in text:
                    strmat += line[:-1]
                self.text.WriteText(strmat + "]")
                self.last_name_saved = path
                self.statusbar.SetStatusText("", 1)
                self.modify = False

            except IOError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except UnicodeDecodeError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

        open_dlg.Destroy()

    def EnterComboRot(self, event):
        """
        in UB matrix editor
        """
        item = event.GetSelection()
        self.CurrentMat = self.list_of_Rot[item]
        event.Skip()

    # def EnterComboDa(self, event):
    # """
    # in UB matrix editor

    # """
    # item = event.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # event.Skip()

    # def EnterComboU(self, event):
    # """
    # in UB matrix editor

    # """
    # item = event.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # event.Skip()

    # def EnterComboDc(self, event):
    # """
    # in UB matrix editor

    # """
    # item = event.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # event.Skip()

    # def EnterComboB(self, event):
    # """
    # in UB matrix editor

    # """
    # item = event.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # event.Skip()

    # def EntercomboExtinc(self, event):
    # """
    # in UB matrix editor

    # """
    # item = event.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # event.Skip()

    def OnLookMatrix(self, event):
        matrix = self.parent.dict_Vect[self.CurrentMat]
        print("%s is :" % self.CurrentMat)
        print(matrix)

        self.mat_a11s.SetValue(str(matrix[0][0]))
        self.mat_a12s.SetValue(str(matrix[0][1]))
        self.mat_a13s.SetValue(str(matrix[0][2]))
        self.mat_a21s.SetValue(str(matrix[1][0]))
        self.mat_a22s.SetValue(str(matrix[1][1]))
        self.mat_a23s.SetValue(str(matrix[1][2]))
        self.mat_a31s.SetValue(str(matrix[2][0]))
        self.mat_a32s.SetValue(str(matrix[2][1]))
        self.mat_a33s.SetValue(str(matrix[2][2]))

        self.texts.SetValue(str(matrix))

    def Matrix_from_texteditor(self, texteditor):
        """
        read matrix element from text editor in [[#,#,#],[#,#,#],[#,#,#]] format
        return matrix(array type)
        """

        text = str(texteditor.GetValue())
        tu = text.replace("[", "").replace("]", "")
        ta = tu.split(",")
        try:
            to = [float(elem) for elem in ta]
        except ValueError:
            wx.MessageBox(
                "Text Editor input Bmatrix seems empty. Fill it or Fill others fields and select the associated button, and then click on Convert",
                "INFO",
            )
            return None
        matrix = np.array([to[:3], to[3:6], to[6:]])
        return matrix

    def Set_lattice_parameter(self, sixvalues, sixtextctrl):
        """
        from six lattice parameters fill the six txtctrls of sixtextctrl
        """
        for k, val in enumerate(sixvalues):
            sixtextctrl[k].SetValue(str(val))

    def Set_matrix_parameter(self, ninevalues, ninetextctrl):
        """
        from nine matrix elements fill the nine txtctrls of ninetextctrl
        """
        for k, val in enumerate(ninevalues):
            ninetextctrl[k].SetValue(str(val))

    def Read_matrix_parameter(self, ninetextctrl):
        """
        read nine matrix elements from ninetextctrl
        """
        mat = []
        for txtcontrol in ninetextctrl:
            mat.append(float(txtcontrol.GetValue()))
        return np.reshape(np.array(mat), (3, 3))

    def Read_lattice_parameter(self, sixtextctrl):
        """
        read six lattice parameters from sixtextctrl
        """
        mat = []
        for txtcontrol in sixtextctrl:
            mat.append(float(txtcontrol.GetValue()))
        return np.array(mat, dtype=float)

    def OnConvert(self, event):
        """
        compute matrices and lattice parameters both in direct and reciprocical
        spaces from data input
        """
        rbuttons = [
            self.rbeditor,
            self.rbelem,
            self.rblatticeparam_direct,
            self.rbeditors,
            self.rbelems,
            self.rblatticeparam_reciprocal,
        ]

        txtctrl_lattice_reciprocal = [
            self.astar,
            self.bstar,
            self.cstar,
            self.alphastar,
            self.betastar,
            self.gammastar,
        ]
        txtctrl_lattice_direct = [
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        ]

        txtctrl_matdirect = [
            self.mat_a11,
            self.mat_a12,
            self.mat_a13,
            self.mat_a21,
            self.mat_a22,
            self.mat_a23,
            self.mat_a31,
            self.mat_a32,
            self.mat_a33,
        ]
        txtctrl_matrecip = [
            self.mat_a11s,
            self.mat_a12s,
            self.mat_a13s,
            self.mat_a21s,
            self.mat_a22s,
            self.mat_a23s,
            self.mat_a31s,
            self.mat_a32s,
            self.mat_a33s,
        ]

        # rbuttons_state = [elem.GetValue() for elem in rbuttons]

        if rbuttons[0].GetValue():  # self.rbeditor

            BMatrix_direct = self.Matrix_from_texteditor(self.text)
            # strange need of this line otherwise :
            # UnboundLocalError: local variable 'Bmatrix_direct' referenced before assignment
            machin = BMatrix_direct
            truc = np.ravel(machin)
            self.Set_matrix_parameter(truc, txtctrl_matdirect)
            lattice_parameter_direct = CP.matrix_to_rlat(BMatrix_direct)
            self.Set_lattice_parameter(lattice_parameter_direct, txtctrl_lattice_direct)
            # reciprocal space
            Bmatrix = CP.calc_B_RR(lattice_parameter_direct, directspace=1)
            self.texts.SetValue(str(Bmatrix.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix), txtctrl_matrecip)
            lattice_parameter_reciprocal = CP.matrix_to_rlat(Bmatrix)
            self.Set_lattice_parameter(
                lattice_parameter_reciprocal, txtctrl_lattice_reciprocal
            )

        elif rbuttons[1].GetValue():  # self.rbelem

            BMatrix_direct = self.Read_matrix_parameter(txtctrl_matdirect)
            # strange need of this line otherwise :
            # UnboundLocalError: local variable 'Bmatrix_direct' referenced before assignment
            truc = str(BMatrix_direct.tolist())
            self.text.SetValue(truc)
            lattice_parameter_direct = CP.matrix_to_rlat(BMatrix_direct)
            self.Set_lattice_parameter(lattice_parameter_direct, txtctrl_lattice_direct)
            # reciprocal space
            Bmatrix = CP.calc_B_RR(lattice_parameter_direct, directspace=1)
            self.texts.SetValue(str(Bmatrix.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix), txtctrl_matrecip)
            lattice_parameter_reciprocal = CP.matrix_to_rlat(Bmatrix)
            self.Set_lattice_parameter(
                lattice_parameter_reciprocal, txtctrl_lattice_reciprocal
            )

        elif rbuttons[2].GetValue():  # self.rblatticeparam_direct
            # TODO limitations in angle to have a direct triedral ? 15 90 160 deg => Nan!!
            lattice_parameter_direct = self.Read_lattice_parameter(
                txtctrl_lattice_direct
            )
            Bmatrix_direct = CP.calc_B_RR(lattice_parameter_direct, directspace=0)
            self.text.SetValue(str(Bmatrix_direct.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix_direct), txtctrl_matdirect)
            # reciprocal space
            Bmatrix = CP.calc_B_RR(lattice_parameter_direct, directspace=1)
            self.texts.SetValue(str(Bmatrix.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix), txtctrl_matrecip)
            lattice_parameter_reciprocal = CP.matrix_to_rlat(Bmatrix)
            self.Set_lattice_parameter(
                lattice_parameter_reciprocal, txtctrl_lattice_reciprocal
            )

        elif rbuttons[3].GetValue():  # self.rbeditors
            # start from list(3*3 elements) of B matrix in RS
            Bmatrix = self.Matrix_from_texteditor(self.texts)
            self.Set_matrix_parameter(np.ravel(Bmatrix), txtctrl_matrecip)
            lattice_parameter_reciprocal = CP.matrix_to_rlat(Bmatrix)
            self.Set_lattice_parameter(
                lattice_parameter_reciprocal, txtctrl_lattice_reciprocal
            )
            # direct space
            lattice_parameter_direct = CP.dlat_to_rlat(lattice_parameter_reciprocal)
            self.Set_lattice_parameter(lattice_parameter_direct, txtctrl_lattice_direct)
            Bmatrix_direct = CP.calc_B_RR(lattice_parameter_direct, directspace=0)
            self.text.SetValue(str(Bmatrix_direct.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix_direct), txtctrl_matdirect)

        elif rbuttons[4].GetValue():  # self.rbelems
            # start from 9 entered elements of B matrix in RS
            BMatrix = self.Read_matrix_parameter(txtctrl_matrecip)
            self.texts.SetValue(str(BMatrix.tolist()))
            lattice_parameter_reciprocal = CP.matrix_to_rlat(BMatrix)
            self.Set_lattice_parameter(
                lattice_parameter_reciprocal, txtctrl_lattice_reciprocal
            )
            # direct space
            lattice_parameter_direct = CP.dlat_to_rlat(lattice_parameter_reciprocal)
            self.Set_lattice_parameter(lattice_parameter_direct, txtctrl_lattice_direct)
            Bmatrix_direct = CP.calc_B_RR(lattice_parameter_direct, directspace=0)
            self.text.SetValue(str(Bmatrix_direct.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix_direct), txtctrl_matdirect)

        elif rbuttons[5].GetValue():  # self.rblatticeparam_reciprocal
            lattice_parameter_reciprocal = self.Read_lattice_parameter(
                txtctrl_lattice_reciprocal
            )
            Bmatrix = CP.calc_B_RR(lattice_parameter_reciprocal, directspace=0)
            self.texts.SetValue(str(Bmatrix.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix), txtctrl_matrecip)
            # direct space
            lattice_parameter_direct = CP.dlat_to_rlat(lattice_parameter_reciprocal)
            self.Set_lattice_parameter(lattice_parameter_direct, txtctrl_lattice_direct)
            Bmatrix_direct = CP.calc_B_RR(lattice_parameter_direct, directspace=0)
            self.text.SetValue(str(Bmatrix_direct.tolist()))
            self.Set_matrix_parameter(np.ravel(Bmatrix_direct), txtctrl_matdirect)

    def OnQuit(self, event):
        dlg = wx.MessageDialog(
            self, 'To use stored UB Matrices, click on "refresh choices" button before.'
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.Close()
            event.Skip()
        else:
            event.Skip()


# --- ---------------  Manual indexation Frame
class ManualIndexFrame(wx.Frame):
    """
    Class to implement a window enabling manual indexation
    """

    def __init__(
        self,
        parent,
        _id,
        title,
        data=(1, 1, 1, 1),
        data_added=None,
        Size=SIZE_PLOTTOOLS,
        datatype="2thetachi",
        kf_direction="Z>0",
        element="Ge",
        data_2thetachi=(None, None),
        data_XY=(None, None),
        Params_to_simulPattern=None,  # Grain, Emin, Emax
        DRTA=0.5,
        MATR=0.5,
        indexation_parameters=None,
        StorageDict=None,
        DataSetObject=None,
        **kwds
    ):

        #        wx.Frame.__init__(self, parent, _id, title, size=(1000, 1200), **kwds)
        wx.Frame.__init__(self, parent, _id, title, size=(600, 1000))

        self.panel = wx.Panel(self)

        self.parent = parent
        # print "my parent is ",self.parent
        # print "my GrandParent is ",self.GetGrandParent()

        # dict related to image data
        self.data_dict = {}
        self.data_dict["Imin"] = 1
        self.data_dict["Imax"] = 1000
        self.data_dict["vmin"] = 1
        self.data_dict["vmax"] = 1000
        self.data_dict["lut"] = "jet"
        self.data_dict["fullpathimagefile"] = ""
        self.data_dict["CCDLabel"] = self.parent.CCDLabel
        self.data_dict["ImageArray"] = None
        self.data_dict["logscale"] = True
        self.data_dict["markercolor"] = "b"

        if self.data_dict["CCDLabel"] in ("MARCCD165", "ROPER"):
            # flip Y axis for marccd type image display
            self.flipyaxis = True
            # TODO to include in data_dict
        else:
            self.flipyaxis = False

        self.CCDLabel = self.parent.CCDLabel

        self.data_dict["flipyaxis"] = self.flipyaxis

        self.ImageArray = None
        self.init_plot = True

        self.getlimitsfromplot = False

        self.datatype = datatype
        # depending of datatype self.Data_X, self.Data_Y can be 2theta, chi or gnomonX,gnomonY, or pixelX,pixelY
        # print "data",data
        # Data_X, Data_Y, Data_I, File_NAME = data

        #         self.data = data
        #         #self.alldata = copy.copy(data)
        #         self.data_2thetachi = data_2thetachi
        #         self.data_XY = data_XY

        self.indexation_parameters = indexation_parameters

        print(
            "self.indexation_parameters['detectordiameter']",
            self.indexation_parameters["detectordiameter"],
        )

        if indexation_parameters is not None:
            DataToIndex = self.indexation_parameters["DataToIndex"]
            if self.datatype is "2thetachi":
                self.Data_X = 2.0 * DataToIndex["data_theta"]
                self.Data_Y = DataToIndex["data_chi"]
            elif self.datatype is "pixels":
                self.Data_X = DataToIndex["data_X"]
                self.Data_Y = DataToIndex["data_Y"]

            elif self.datatype is "gnomon":
                self.Data_X = DataToIndex["data_gnomonX"]
                self.Data_Y = DataToIndex["data_gnomonY"]
                self.data_gnomonXY = (
                    DataToIndex["data_gnomonX"],
                    DataToIndex["data_gnomonY"],
                )

            self.Data_I = DataToIndex["data_I"]
            self.File_NAME = self.indexation_parameters["DataPlot_filename"]
            self.data_XY = DataToIndex["data_X"], DataToIndex["data_Y"]

            self.data_2thetachi = 2 * DataToIndex["data_theta"], DataToIndex["data_chi"]
            print(
                "self.data_2thetachi[0][:5],self.data_2thetachi[1][:5]",
                self.data_2thetachi[0][:5],
                self.data_2thetachi[1][:5],
            )
            self.data = self.Data_X, self.Data_Y, self.Data_I, self.File_NAME
            self.alldata = copy.copy(data)
            self.selectedAbsoluteSpotIndices_init = DataToIndex[
                "current_exp_spot_index_list"
            ]
            self.selectedAbsoluteSpotIndices = copy.copy(
                self.selectedAbsoluteSpotIndices_init
            )

        # create attributes X Y tth chi
        self.init_data()
        self.DataSet = DataSetObject

        self.kf_direction = kf_direction
        print("self.kf_direction in ManualIndexFrame", self.kf_direction)

        self.current_matrix = []
        self.Millerindices = None

        # simulation parameters
        # self.SimulParam =(grain, emin, self.emax.GetValue())
        self.SimulParam = Params_to_simulPattern

        # simulated 2theta,chi
        self.data_theo = data_added
        # overwrite self.data_theo
        #        self.Simulate_Pattern()
        # self.data_theo = data_added # a way in the past to put simulated data

        self.points = []  # to store points
        self.selectionPoints = []
        self.twopoints = []
        self.nbclick = 1
        self.nbclick_dist = 0
        self.clicked_indexSpot = []

        self.recognition_possible = True
        self.toshow = []

        # draggable line for gnomon data
        self.dragLines = []
        self.addlines = False

        self.onlyclosest = 1

        # absolute exp. spot index list after using filter data button
        self.abs_spotindex = None

        # defaut value for Miller attribution
        self.B0matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]  # means: columns are a*,b*,c* in xyz frame
        self.key_material = element
        self.detectordistance = None

        self.DRTA = DRTA
        self.MATR = MATR

        self.Params_to_simulPattern = Params_to_simulPattern

        self.StorageDict = StorageDict

        self.indexation_parameters["mainAppframe"] = self.parent
        self.indexation_parameters["indexationframe"] = self

        self.initGUI()

    def initGUI(self):

        self.sb = self.CreateStatusBar()

        colourb_bkg = [242, 241, 240, 255]
        colourb_bkg = np.array(colourb_bkg) / 255.0

        self.dpi = 100
        self.figsizex, self.figsizey = 4, 3
        self.fig = Figure(
            (self.figsizex, self.figsizey), dpi=self.dpi, facecolor=tuple(colourb_bkg)
        )
        self.fig.set_size_inches(self.figsizex, self.figsizey, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas)

        self.fig.canvas.mpl_connect("button_press_event", self.onClick)

        self.tooltip = wx.ToolTip(tip="Welcome on LaueTools UB Refinement board")
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)

        self.pickdistbtn = wx.ToggleButton(self.panel, 2, "Pick distance")
        self.recongnisebtn = wx.ToggleButton(
            self.panel, 3, "Recognise distance", size=(150, 40)
        )
        self.pointButton6 = wx.ToggleButton(self.panel, 6, "Show Exp. Spot Props")

        self.listbuttons = [self.pickdistbtn, self.recongnisebtn, self.pointButton6]

        self.defaultColor = self.GetBackgroundColour()
        self.p2S, self.p3S, self.p6S = 0, 0, 0
        self.listbuttonstate = [self.p2S, self.p3S, self.p6S]
        self.listbuttonstate = [0, 0, 0]

        self.Bind(wx.EVT_TOGGLEBUTTON, self.T2, id=2)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.T3, id=3)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.T6, id=6)

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        self.txttolerances = wx.StaticText(self.panel, -1, "Tolerance angles (deg)")
        self.txttolerances.SetFont(font3)

        self.txt1 = wx.StaticText(self.panel, -1, "Dist. Recognition      ")
        # self.DRTA = wx.TextCtrl(self.parampanel, -1,'0.5',(350, 10))
        self.DRTA = wx.TextCtrl(self.panel, -1, str(self.DRTA))
        self.txt2 = wx.StaticText(self.panel, -1, "Spots Matching")
        # self.matr_ctrl = wx.TextCtrl(self.parampanel, -1,'0.5',(350, 40))
        self.matr_ctrl = wx.TextCtrl(self.panel, -1, str(self.MATR))

        self.txtnlut = wx.StaticText(self.panel, -1, "n LUT:    ")
        self.nlut = wx.SpinCtrl(self.panel, -1, "3", min=1, max=7, size=(50, -1))

        self.nlut.Bind(wx.EVT_SPINCTRL, self.onNlut)

        self.slider_exp = wx.Slider(self.panel, -1, 50, 0, 100, size=(120, -1))
        self.slidertxt_exp = wx.StaticText(self.panel, -1, "Exp. spot size", (5, 5))
        self.slider_exp.Bind(wx.EVT_SLIDER, self.sliderUpdate_exp)

        self.slider_ps = wx.Slider(self.panel, -1, 50, 0, 100, size=(120, -1))
        self.slidertxt_ps = wx.StaticText(self.panel, -1, "power scale", (5, 5))
        self.slider_ps.Bind(wx.EVT_SLIDER, self.sliderUpdate_ps)

        self.findspotchck = wx.CheckBox(self.panel, -1, "Show Spot Index")
        self.findspotchck.SetValue(False)
        self.spotindexspinctrl = wx.SpinCtrl(
            self.panel, -1, "0", min=0, max=len(self.data[0]) - 1, size=(70, -1)
        )
        self.spotindexspinctrl.Bind(wx.EVT_SPINCTRL, self.locateSpot)

        self.UCEP = wx.CheckBox(self.panel, -1, "Select closest Exp. Spots")
        self.UCEP.SetValue(True)
        self.UCEP.Disable()

        self.COCD = wx.CheckBox(self.panel, -1, "Consider only closest distance")
        self.COCD.SetValue(True)
        self.COCD.Disable()
        self.MWFD = wx.CheckBox(
            self.panel, -1, "Matching Rate computed with filtered Data"
        )
        self.MWFD.SetValue(False)

        self.txtEbandpass = wx.StaticText(self.panel, -1, "Energy Bandpass (keV): ")
        self.txtEbandpass.SetFont(font3)
        self.emintxt = wx.StaticText(self.panel, -1, "min.                       ")
        self.SCEmin = wx.SpinCtrl(self.panel, -1, "5", min=5, max=49)
        self.emaxtxt = wx.StaticText(self.panel, -1, "max. ")

        if self.datatype == "gnomon":
            self.drawlinebtn = wx.Button(self.panel, -1, "Draw line")
            self.drawlinebtn.Bind(wx.EVT_BUTTON, self.OnDrawLine)

            self.clearlinesbtn = wx.Button(self.panel, -1, "Clear lines")
            self.clearlinesbtn.Bind(wx.EVT_BUTTON, self.OnClearLines)

        if self.Params_to_simulPattern is not None:
            self.SCEmax = wx.SpinCtrl(
                self.panel, -1, str(self.Params_to_simulPattern[2]), min=6, max=150
            )
        else:
            self.SCEmax = wx.SpinCtrl(self.panel, -1, "22", min=6, max=150)

        self.txtelem = wx.StaticText(self.panel, -1, "Elem: ")
        self.list_materials = sorted(self.parent.dict_Materials.keys())
        self.combokeymaterial = wx.ComboBox(
            self.panel,
            -1,
            self.key_material,
            choices=self.list_materials,
            style=wx.CB_READONLY,
        )
        self.combokeymaterial.Bind(wx.EVT_COMBOBOX, self.EnterCombokeymaterial)

        self.sethklchck = wx.CheckBox(self.panel, -1, "set spot1 hkl")
        self.sethklcentral = wx.TextCtrl(self.panel, -1, "[1,0,0]", size=(100, -1))

        self.filterDatabtn = wx.Button(self.panel, -1, "Filter Exp. Data")
        self.filterDatabtn.Bind(wx.EVT_BUTTON, self.BuildDataDict)

        self.imagescalebtn = wx.Button(self.panel, -1, "Set Image Scale")
        self.imagescalebtn.Bind(wx.EVT_BUTTON, self.onSetImageScale)
        self.imagescalebtn.Disable()
        if self.datatype == "pixels":
            self.imagescalebtn.Enable()

        self.plotlimitsbtn = wx.Button(self.panel, -1, "Plot limits")
        self.plotlimitsbtn.Bind(wx.EVT_BUTTON, self.SetPlotLimits)

        self.spotsizebtn = wx.Button(self.panel, -1, "Spot Size LUT")
        self.spotsizebtn.Bind(wx.EVT_BUTTON, self.SetSpotSize)
        self.spotsizebtn.Disable()

        self.txtctrldistance = wx.StaticText(
            self.panel, -1, "==> %s deg" % "", size=(80, -1)
        )

        self.verbose = wx.CheckBox(self.panel, -1, "display details")
        self.verbose.SetValue(False)

        # for annotation
        self.drawnAnnotations_exp = {}
        self.links_exp = []

        self._layout()
        self.initplotlimits()
        self._replot()

        # tooltips
        entip = "set energy range to simulate Laue pattern from potential orientation matrix found by recognised angular distance"
        self.emintxt.SetToolTipString(entip)
        self.SCEmin.SetToolTipString(entip)
        self.emaxtxt.SetToolTipString(entip)
        self.SCEmax.SetToolTipString(entip)

        nlutip = "Largest integer of miller indices used to build the angular distances reference database table (LUT)"
        self.nlut.SetToolTipString(nlutip)
        self.txtnlut.SetToolTipString(nlutip)

        self.pickdistbtn.SetToolTipString(
            "Compute distance between the two next clicked spots or points in plot.\nThe angle is the separation angle between the two corresponding lattice planes normals"
        )

        rectip = "Press this button and then click on two spots that are likely to have small hkl indices such as to be recognised"
        rectip += "Index spots from separation angles between lattice planes normals from the two next clicked spots or points in plot.\n"
        rectip += "Each angular distance found in reference structure LUT will lead to potential orientation matrices from which a Laue Pattern can be simulated.\n"
        rectip += "Matching rate is given by the number of close pairs of exp. and simulated spots.\n"
        rectip += "All separation distances between first clicked spot and a set from the most intense (index 0) to the 2nd clicked spot index will tested"
        self.recongnisebtn.SetToolTipString(rectip)

        self.pointButton6.SetToolTipString(
            "Display info of experimental spots by clicking on them"
        )

        self.UCEP.SetToolTipString(
            "Select nearest spot position or exact clicked position"
        )

        sethkltip = "Set the [h,k,l] Miller indices of the first clicked spot. For cubic structure (simple, body & face centered, diamond etc...) l index must be positive"
        self.sethklchck.SetToolTipString(sethkltip)
        self.sethklcentral.SetToolTipString(sethkltip)

        self.filterDatabtn.SetToolTipString(
            "Filter (select, remove) experimental spots to display"
        )

        self.MWFD.SetToolTipString(
            "Compute matching rate of simulated Laue pattern with filtered set of experiment spots"
        )

        tipshow = "Show on plot experimental spot with given absolute index"
        self.findspotchck.SetToolTipString(tipshow)
        self.spotindexspinctrl.SetToolTipString(tipshow)

    def _layout(self):

        lutbox = wx.BoxSizer(wx.HORIZONTAL)
        lutbox.Add(self.txtnlut, 0, wx.ALL)
        lutbox.Add(self.nlut, 0, wx.ALL)
        lutbox.Add(self.spotsizebtn, 0, wx.ALL)

        ang1box = wx.BoxSizer(wx.VERTICAL)
        ang1box.Add(self.txt1, 0, wx.ALL)
        ang1box.Add(self.DRTA, 0, wx.ALL)

        ang2box = wx.BoxSizer(wx.VERTICAL)
        ang2box.Add(self.txt2, 0, wx.ALL)
        ang2box.Add(self.matr_ctrl, 0, wx.ALL)

        anglesbox = wx.BoxSizer(wx.HORIZONTAL)
        anglesbox.Add(ang1box, 0, wx.ALL)
        anglesbox.Add(ang2box, 0, wx.ALL)

        eminbox = wx.BoxSizer(wx.VERTICAL)
        eminbox.Add(self.emintxt, 0, wx.ALL)
        eminbox.Add(self.SCEmin, 0, wx.ALL)

        emaxbox = wx.BoxSizer(wx.VERTICAL)
        emaxbox.Add(self.emaxtxt, 0, wx.ALL)
        emaxbox.Add(self.SCEmax, 0, wx.ALL)

        henergiesbox = wx.BoxSizer(wx.HORIZONTAL)
        henergiesbox.Add(eminbox, 0, wx.ALL)
        henergiesbox.Add(emaxbox, 0, wx.ALL)

        sizerparam = wx.BoxSizer(wx.VERTICAL)
        sizerparam.Add(lutbox, 0, wx.ALL)
        sizerparam.AddSpacer(5)

        sizerparam.Add(self.txttolerances, 0, wx.ALL)
        sizerparam.Add(anglesbox, 0, wx.ALL)
        sizerparam.Add(self.txtEbandpass, 0, wx.ALL)
        sizerparam.AddSpacer(5)
        sizerparam.Add(henergiesbox, 0, wx.ALL)
        sizerparam.Add(self.txtelem, 0, wx.ALL)
        sizerparam.Add(self.combokeymaterial, 0, wx.ALL)

        btnSizer0 = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer0.Add(self.pointButton6, 0, wx.ALL)
        btnSizer0.Add(self.filterDatabtn, 0, wx.ALL)
        btnSizer0.Add(self.plotlimitsbtn, 0, wx.ALL)
        btnSizer0.Add(self.imagescalebtn, 0, wx.ALL)

        btnSizerfindspot = wx.BoxSizer(wx.HORIZONTAL)
        btnSizerfindspot.Add(self.findspotchck, 0, wx.ALL)
        btnSizerfindspot.Add(self.spotindexspinctrl, 0, wx.ALL)

        boxsliders = wx.BoxSizer(wx.HORIZONTAL)
        boxsliders.Add(self.slider_exp, 0, wx.ALL)
        boxsliders.Add(self.slidertxt_exp, 0, wx.ALL)
        boxsliders.Add(self.slider_ps, 0, wx.ALL)
        boxsliders.Add(self.slidertxt_ps, 0, wx.ALL)

        if self.datatype == "gnomon":
            Sizerline = wx.BoxSizer(wx.HORIZONTAL)
            Sizerline.Add(self.drawlinebtn, 0, wx.ALL)
            Sizerline.Add(self.clearlinesbtn, 0, wx.ALL)

        btnSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer1.Add(self.pickdistbtn, 0, wx.ALL, 0)
        btnSizer1.Add(self.txtctrldistance, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.recongnisebtn, 0, wx.ALL, 0)
        hsizer.Add(self.sethklchck, 0, wx.ALL, 0)
        hsizer.Add(self.sethklcentral, 0, wx.ALL, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(self.UCEP, 0, wx.ALL, 0)
        hsizer2.Add(self.verbose, 0, wx.ALL, 0)

        btnsSizer = wx.BoxSizer(wx.VERTICAL)
        btnsSizer.Add(btnSizer0, 0, wx.ALL)
        btnsSizer.Add(btnSizerfindspot, 0, wx.ALL, 5)
        btnsSizer.Add(boxsliders, 0, wx.ALL, 5)
        if self.datatype == "gnomon":
            btnsSizer.Add(Sizerline, 0, wx.ALL)
        btnsSizer.Add(sizerparam, 0, wx.ALL)
        btnsSizer.AddSpacer(2)
        btnsSizer.Add(btnSizer1, 0, wx.ALL)
        btnsSizer.AddSpacer(5)
        btnsSizer.Add(hsizer, 0, wx.ALL)
        btnsSizer.Add(hsizer2, 0, wx.ALL)
        btnsSizer.Add(self.COCD, 0, wx.ALL)
        btnsSizer.Add(self.MWFD, 0, wx.ALL)

        sizerplot = wx.BoxSizer(wx.VERTICAL)
        sizerplot.Add(self.canvas, 1, wx.TOP | wx.GROW)
        sizerplot.Add(self.toolbar, 0, wx.EXPAND)

        sizerH = wx.BoxSizer(wx.HORIZONTAL)
        sizerH.Add(sizerplot, 1, wx.ALL | wx.GROW, 5)
        sizerH.Add(btnsSizer, 0, wx.ALL, 5)

        self.panel.SetSizer(sizerH)
        sizerH.Fit(self)
        self.Layout()

    def sliderUpdate_exp(self, event):
        print("sliderUpdate_exp")
        self.factorsize = int(self.slider_exp.GetValue())
        self.getlimitsfromplot = True
        self._replot()
        print("factor spot size = %f " % self.factorsize)

    def sliderUpdate_ps(self, event):
        print("sliderUpdate_ps", self.slider_ps.GetValue())
        # ps from -5 5
        ps = (int(self.slider_ps.GetValue()) - 50) / 10.0

        print("ps", ps)

        if ps < 1:  # from 1/5 to 1
            jump_ps = 4 / 5.0
            deltaps = 6.0
            self.powerscale = jump_ps / deltaps * (ps - 1) + 1
        if ps >= 1:
            self.powerscale = ps * 2

        self.getlimitsfromplot = True
        self._replot()
        print("powerscale factor spot size = %f " % self.powerscale)

    def onSetImageScale(self, evt):
        """
        open a board to change image scale
        """
        import PlotRefineGUI as PRGUI

        IScaleBoard = PRGUI.IntensityScaleBoard(
            self, -1, "Image scale setting Board", self.data_dict
        )

        IScaleBoard.Show(True)

    #         PlotLismitsBoard.ShowModal()
    #         PlotLismitsBoard.Destroy()
    #         self._replot()

    def OnDrawLine(self, evt):
        self.addlines = True

        self.getlimitsfromplot = True
        self._replot()

    def OnClearLines(self, evt):
        self.addlines = False

        self.getlimitsfromplot = True
        self._replot()

    def init_data(self):
        if self.datatype == "2thetachi":
            #             self.tth, self.chi = self.Data_X, self.Data_Y
            self.tth, self.chi = self.data_2thetachi
            self.pixelX, self.pixelY = self.data_XY
        elif self.datatype == "gnomon":
            self.tth, self.chi = self.data_2thetachi
            self.gnomonX, self.gnomonY = self.data_gnomonXY
        elif self.datatype == "pixels":
            self.tth, self.chi = self.data_2thetachi
            self.pixelX, self.pixelY = self.data_XY
            print("pixels plot")

        # TODO to remove
        self.Data_index_expspot = np.arange(len(self.Data_X))

    def init_data_old(self):
        """
        calculate 2theta chi gnomonX gnomonY pixX pixY properties depending on datatype
        
        """
        # x,y data coordinates to consider
        self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data

        if self.datatype == "2thetachi":
            self.tth, self.chi = self.Data_X, self.Data_Y
        elif self.datatype == "gnomon":
            # TODO: only when not indexed spot is the whole list
            # Need to do like above from input data
            self.Data_X = (copy.copy(self.parent.data_gnomonx),)
            self.Data_Y = copy.copy(self.parent.data_gnomony)
            self.tth, self.chi = self.data_2thetachi
        #             print "init_data"
        #             print self.Data_X, self.Data_Y
        #             print min(self.Data_X), max(self.Data_X), min(self.Data_Y), max(self.Data_Y)

        elif self.datatype == "pixels":
            self.tth, self.chi = self.data_2thetachi
            print("pixels plot")
            pass

        self.Data_index_expspot = np.arange(len(self.Data_X))

    def reinit_data(self):
        self.data = copy.copy(self.alldata)

    def BuildDataDict_old(self, evt):  # filter Exp Data spots
        """
        in ManualIndexFrame class
        launch by 'filter Exp Data spots' button
        
        Open spots filter editor and build a dict of data to be used for indexation
        """
        self.toreturn = None

        self.reinit_data()
        self.init_data()

        if self.datatype == "2thetachi":
            fields = ["Spot index", "2Theta", "Chi", "Intensity"]
            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
            to_put_in_dict = (
                np.arange(len(self.data[0])),
                self.data[0],
                self.data[1],
                self.data[2],
            )

        elif self.datatype == "gnomon":
            fields = [
                "Spot index",
                "X_gmonon",
                "Y_gmonon",
                "Intensity",
                "2Theta",
                "Chi",
            ]
            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
            to_put_in_dict = (
                np.arange(len(self.data[0])),
                self.Data_X,
                self.Data_Y,
                self.data[2],
                self.data[0],
                self.data[1],
            )

        elif self.datatype == "pixels":
            fields = ["Spot index", "X_CCD", "Y_CCD", "Intensity", "2Theta", "Chi"]
            # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
            to_put_in_dict = (
                np.arange(len(self.data[0])),
                self.data[0],
                self.data[1],
                self.data[2],
                self.tth,
                self.chi,
            )

        mySpotData = {}
        for k, ff in enumerate(fields):
            mySpotData[ff] = to_put_in_dict[k]
        dia = LSEditor.SpotsEditor(
            self,
            -1,
            "Filter Experimental Spots Data",
            mySpotData,
            func_to_call=self.readdata_fromEditor,
            field_name_and_order=fields,
        )
        dia.Show(True)

    def BuildDataDict(self, evt):  # filter Exp Data spots
        """
        in ManualIndexFrame class
        
        Filter Exp. Data Button
        
        Filter exp. spots data to be displayed 
        """

        self.toreturn = None

        #         self.reinit_data()
        #         self.init_data()

        C0 = self.selectedAbsoluteSpotIndices_init
        AllDataToIndex = self.indexation_parameters["AllDataToIndex"]

        C1 = 2.0 * AllDataToIndex["data_theta"][C0]
        C2 = AllDataToIndex["data_chi"][C0]
        C3 = AllDataToIndex["data_I"][C0]
        C4 = AllDataToIndex["data_pixX"][C0]
        C5 = AllDataToIndex["data_pixY"][C0]
        C6 = AllDataToIndex["data_gnomonX"][C0]
        C7 = AllDataToIndex["data_gnomonY"][C0]

        if self.datatype is "2thetachi":
            fields = ["Spot index", "2theta", "Chi", "Intensity"]

            to_put_in_dict = C0, C1, C2, C3

        if self.datatype is "pixels":
            fields = ["Spot index", "pixelX", "pixelY", "Intensity", "2Theta", "Chi"]

            to_put_in_dict = C0, C4, C5, C3, C1, C2

        if self.datatype is "gnomon":
            fields = ["Spot index", "gnomonX", "gnomonY", "Intensity", "2Theta", "Chi"]
            to_put_in_dict = C0, C6, C7, C3, C1, C2

        mySpotData = {}
        for k, ff in enumerate(fields):
            mySpotData[ff] = to_put_in_dict[k]
        dia = LSEditor.SpotsEditor(
            None,
            -1,
            "Filter Experimental Spots Data",
            mySpotData,
            func_to_call=self.readdata_fromEditor,
            field_name_and_order=fields,
        )
        dia.Show(True)

    def readdata_fromEditor(self, data):
        """
        update exp. spots data according to the user selected filter
        """
        #                 # update data according to the user selected filter
        #
        #         # take only the first 4 columns
        #         abs_spotindex, self.Data_X, self.Data_Y, self.Data_I = np.transpose(toreturn)[:4]
        #         self.abs_spotindex = np.array(abs_spotindex,dtype=np.int)

        selectedSpotsPropsarray = data.T

        col0 = selectedSpotsPropsarray[0]
        col1, col2, col3 = selectedSpotsPropsarray[1:4]

        print("\n****SELECTED and DISPLAYED PART OF EXPERIMENTAL SPOTS\n")
        self.selectedAbsoluteSpotIndices = np.array(col0, dtype=np.int)

        if self.datatype is "2thetachi":

            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            self.data_XY = (
                self.indexation_parameters["AllDataToIndex"]["data_pixX"][
                    self.selectedAbsoluteSpotIndices
                ],
                self.indexation_parameters["AllDataToIndex"]["data_pixY"][
                    self.selectedAbsoluteSpotIndices
                ],
            )

        elif self.datatype is "pixels":
            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            self.pixelX, self.pixelY = (
                self.indexation_parameters["AllDataToIndex"]["data_pixX"][
                    self.selectedAbsoluteSpotIndices
                ],
                self.indexation_parameters["AllDataToIndex"]["data_pixY"][
                    self.selectedAbsoluteSpotIndices
                ],
            )

        elif self.datatype is "gnomon":
            self.data_2thetachi = col1, col2
            self.tth, self.chi = col1, col2
            self.Data_I = col3
            self.gnomonX, self.gnomonY = (
                self.indexation_parameters["AllDataToIndex"]["data_gnomonX"][
                    self.selectedAbsoluteSpotIndices
                ],
                self.indexation_parameters["AllDataToIndex"]["data_gnomonY"][
                    self.selectedAbsoluteSpotIndices
                ],
            )

        self._replot()

    def onMotion_ToolTip(self, event):
        """tool tip to show data when mouse hovers on plot
        """
        if len(self.data[0]) == 0:
            return

        collisionFound = False

        xtol = 20
        ytol = 20

        # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
        # self.Data_index_expspot = np.arange(len(self.Data_X))
        # """
        xdata, ydata, _annotes = (
            self.Data_X,
            self.Data_Y,
            list(zip(self.Data_index_expspot, self.Data_I)),
        )

        #         print "DATA:    \n\n\n", xdata[:5], ydata[:5], annotes

        if event.xdata is not None and event.ydata is not None:

            clickX = event.xdata
            clickY = event.ydata

            #             print 'clickX,clickY in onMotion_ToolTip', clickX, clickY

            annotes = []
            for x, y, a in zip(xdata, ydata, _annotes):
                if (clickX - xtol < x < clickX + xtol) and (
                    clickY - ytol < y < clickY + ytol
                ):
                    annotes.append(
                        (GT.cartesiandistance(x, clickX, y, clickY), x, y, a)
                    )

            #             print 'annotes', annotes

            if annotes == []:
                collisionFound = False
                return

            annotes.sort()
            _distance, x, y, annote = annotes[0]
            #             print "the nearest experimental point is at(%.2f,%.2f)" % (x, y)
            #             print "with index %d and intensity %.1f" % (annote[0], annote[1])

            self.updateStatusBar(x, y, annote)

            self.tooltip.SetTip(
                "Spot abs. index=%d. Intensity=%.1f" % (annote[0], annote[1])
            )
            self.tooltip.Enable(True)
            collisionFound = True

            return

        if not collisionFound:
            pass

    def SetSpotSize(self, evt):
        wx.MessageBox("To be implemented", "info")
        pass

    #         data_dict = {}
    #         PlotLismitsBoard = PlotLimitsBoard(self, -1, 'Spot size Board',
    #                                                 data_dict)
    #
    #         PlotLismitsBoard.Show(True)

    def SetPlotLimits(self, evt):
        """
        open a board to change plot limits
        """
        data_dict = {}
        data_dict["datatype"] = self.datatype
        data_dict["dataXmin"] = min(self.Data_X)
        data_dict["dataXmax"] = max(self.Data_X)
        data_dict["dataYmin"] = min(self.Data_Y)
        data_dict["dataYmax"] = max(self.Data_Y)
        data_dict["xlim"] = self.xlim
        data_dict["ylim"] = self.ylim
        data_dict["kf_direction"] = self.kf_direction

        print("data_dict", data_dict)

        PlotLismitsBoard = PlotLimitsBoard(
            self, -1, "Data Plot limits Board", data_dict
        )

        PlotLismitsBoard.Show(True)

    #         PlotLismitsBoard.ShowModal()
    #         PlotLismitsBoard.Destroy()

    #         self._replot()

    def initplotlimits(self):
        if self.datatype == "2thetachi":
            self.locatespotmarkersize = 3
            if self.kf_direction == "Z>0":
                self.xlim = (34, 146)
                self.ylim = (-50, 50)
            elif self.kf_direction == "X>0":
                self.xlim = (-1, 60)
                self.ylim = (-180, 180)

        elif self.datatype == "gnomon":
            self.locatespotmarkersize = 0.05
            self.ylim = (-0.6, 0.6)
            self.xlim = (-0.6, 0.6)

        elif self.datatype == "pixels":
            self.locatespotmarkersize = 50

            if self.CCDLabel in ("MARCCD165", "PRINCETON"):
                self.ylim = (2048, 0)
                self.xlim = (0, 2048)
            elif self.CCDLabel in ("VHR_PSI",):
                self.ylim = (3000, 0)
                self.xlim = (0, 4000)
            elif self.CCDLabel.startswith("sCMOS"):
                self.ylim = (2050, 0)
                self.xlim = (0, 2050)

        self.factorsize = 50
        self.powerscale = 1.0

    def setplotlimits_fromcurrentplot(self):
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
        print("new limits x", self.xlim)
        print("new limits y", self.ylim)

    def _replot(self):
        """
        ManualIndexFrame
        """

        # offsets to match imshow and scatter plot coordinates frames
        if self.datatype == "pixels":
            X_offset = 1
            Y_offset = 1
        else:
            X_offset = 0
            Y_offset = 0

        # Data_X, Data_Y, Data_I, File_NAME = self.data
        #         print "ManualIndexFrame _replot"
        #         print "self.init_plot", self.init_plot
        #         print "self.getlimitsfromplot", self.getlimitsfromplot
        #         print "self.flipyaxis", self.flipyaxis

        if not self.init_plot and self.getlimitsfromplot:
            self.setplotlimits_fromcurrentplot()

            self.getlimitsfromplot = False

        self.axes.clear()

        # clear the axes and replot everything
        #        self.axes.cla()

        def fromindex_to_pixelpos_x(index, pos):
            if self.datatype == "pixels":
                return int(index)
            else:
                return index

        def fromindex_to_pixelpos_y(index, pos):
            if self.datatype == "pixels":
                return int(index)
            else:
                return index

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))

        # background image
        if self.ImageArray is not None and self.datatype == "pixels":

            print("self.ImageArray", self.ImageArray.shape)
            self.myplot = self.axes.imshow(self.ImageArray, interpolation="nearest")

            if not self.data_dict["logscale"]:
                norm = matplotlib.colors.Normalize(
                    vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"]
                )
            else:
                norm = matplotlib.colors.LogNorm(
                    vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"]
                )

            self.myplot.set_norm(norm)
            self.myplot.set_cmap(self.data_dict["lut"])

        # exp scale lin=0
        spotsizescale = 0
        s0, s_asympt, tau, smin, smax = 0, 1.0, 65000 / 3.0, 1.0, 5.0
        params_spotsize = s0, s_asympt, tau, smin, smax

        # linear scale lin=1
        spotsizescale = 1
        s0, slope, smin, smax = 0, 1.0, 1, 5.0
        params_spotsize = s0, slope, smin, smax

        # power scale lin=2
        spotsizescale = 2
        s0, s_at_Imax, powerscale, smin, smax = 0, 70000, self.powerscale, 1, 5.0
        params_spotsize = s0, s_at_Imax, powerscale, smin, smax

        if self.datatype == "pixels":
            # background image
            if self.ImageArray is not None:
                kwords = {
                    "marker": "o",
                    "facecolor": "None",
                    "edgecolor": self.data_dict["markercolor"],
                }
            else:
                #                 self.axes.set_xbound(self.currentbounds[0])
                #                 self.axes.set_ybound(self.currentbounds[1])

                kwords = {
                    "edgecolor": "None",
                    "facecolor": self.data_dict["markercolor"],
                }

            self.axes.scatter(
                self.pixelX - X_offset,
                self.pixelY - Y_offset,
                #                           s=self.func_size_intensity(np.array(self.Data_I), self.factorsize, 0, lin=1))
                s=self.factorsize
                * self.func_size_peakintensity(
                    np.array(self.Data_I), params_spotsize, lin=spotsizescale
                ),
                alpha=1.0,
                **kwords
            )

        elif self.datatype == "2thetachi":
            self.axes.scatter(
                self.data_2thetachi[0],
                self.data_2thetachi[1],
                #                           s=self.func_size_intensity(np.array(self.Data_I), self.factorsize, 0, lin=1))
                s=self.factorsize
                * self.func_size_peakintensity(
                    np.array(self.Data_I), params_spotsize, lin=spotsizescale
                ),
            )
            # c=self.Data_I / 50.)#, cmap = GT.SPECTRAL)
        elif self.datatype == "gnomon":
            self.axes.scatter(
                self.gnomonX - X_offset,
                self.gnomonY - Y_offset,
                #                           s=self.func_size_intensity(np.array(self.Data_I), self.factorsize, 0, lin=1))
                s=self.factorsize
                * self.func_size_peakintensity(
                    np.array(self.Data_I), params_spotsize, lin=spotsizescale
                ),
            )

        # axes labels
        if self.datatype == "2thetachi":
            self.axes.set_xlabel("2theta(deg.)")
            self.axes.set_ylabel("chi(deg)")
        elif self.datatype == "gnomon":
            self.axes.set_xlabel("X gnomon")
            self.axes.set_ylabel("Y gnomon")
        elif self.datatype == "pixels":
            self.axes.set_xlabel("X pixel")
            self.axes.set_ylabel("Y pixel")

        # plot title
        if self.init_plot:
            self.axes.set_title("%s %d spots" % (self.File_NAME, len(self.Data_X)))
            self.axes.grid(True)

        # restore the zoom limits(unless they're for an empty plot)
        if self.xlim != (0.0, 1.0) or self.ylim != (0.0, 1.0):
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)

        # for gnomonic coordinates case
        if self.addlines:

            pt1 = [0.0, 0.0]
            pt2 = [0.5, 0.5]
            ptcenter = DGP.center_pts(pt1, pt2)

            circles = [
                DGP.patches.Circle(pt1, 0.03, fc="r", alpha=0.5),
                DGP.patches.Circle(ptcenter, 0.03, fc="r", alpha=0.5),
                DGP.patches.Circle(pt2, 0.03, fc="r", alpha=0.5),
            ]

            line, = self.axes.plot(
                [pt1[0], ptcenter[0], pt2[0]],
                [pt1[1], ptcenter[1], pt2[1]],
                picker=0.03,
                c="r",
            )

            for circ in circles:
                self.axes.add_patch(circ)

            self.dragLines.append(
                DGP.DraggableLine(
                    circles, line, tolerance=0.03, parent=self, datatype=self.datatype
                )
            )
            self.addlines = False

        self.init_plot = False

        # redraw the display
        #        self.plotPanel.draw()
        self.canvas.draw()

    def func_size_energy(self, val, factor):
        return 400.0 * factor / (val + 1.0)

    def func_size_intensity(self, val, factor, offset, lin=1):
        if lin:
            return 0.1 * (factor * val + offset)
        else:  # log scale
            return factor * np.log(np.clip(val, 0.000000001, 1000000000000)) + offset

    def func_size_peakintensity(self, intensity, params, lin=1):
        if lin == 1:
            s0, slope, smin, smax = params
            s = np.clip(slope * intensity + s0, smin, smax)
        elif lin == 0:  # log scale
            s0, s_asympt, tau, smin, smax = params
            s = np.clip((smax - smin) * (1 - np.exp(-intensity / tau)) + s0, smin, smax)

        elif lin == 2:
            s0, s_at_Imax, powerscale, smin, smax = params
            s = np.clip(
                (smax - s0) * (intensity / s_at_Imax) ** powerscale + s0, smin, smax
            )

        return s

    def onNlut(self, evt):

        nlut = int(self.nlut.GetValue())
        if nlut > 7:
            dlg = wx.MessageDialog(
                self,
                "nlut must be reasonnably less or equal to 7",
                "warning",
                wx.OK | wx.ICON_WARNING,
            )
            dlg.ShowModal()
            dlg.Destroy()

    def onClick(self, event):
        """ onclick
        """
        #         if self.dr is not None:
        #             print "self.dr exists"

        res = None
        #        print 'clicked on mouse'
        if event.inaxes:
            #            print("inaxes", event)
            print(("inaxes x,y", event.x, event.y))
            print(("inaxes  xdata, ydata", event.xdata, event.ydata))
            self.centerx, self.centery = event.xdata, event.ydata

            if self.pointButton6.GetValue():
                self.Annotate_exp(event)

            if self.pickdistbtn.GetValue():
                self.nbclick_dist += 1
                print("self.nbclick_dist", self.nbclick_dist)

                if self.nbclick_dist == 1:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(event)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = event.xdata, event.ydata
                    else:
                        x, y = event.xdata, event.ydata

                    self.twopoints = [(x, y)]

                if self.nbclick_dist == 2:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(event)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = event.xdata, event.ydata
                    else:
                        x, y = event.xdata, event.ydata

                    self.twopoints.append((x, y))

                    spot1 = self.twopoints[0]
                    spot2 = self.twopoints[1]

                    if self.datatype == "2thetachi":
                        _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))

                    if self.datatype == "gnomon":
                        tw, ch = IIM.Fromgnomon_to_2thetachi(
                            [
                                np.array([spot1[0], spot2[0]]),
                                np.array([spot1[1], spot2[1]]),
                            ],
                            0,
                        )[:2]
                        _dist = GT.distfrom2thetachi(
                            np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                        )

                    if self.datatype == "pixels":

                        detectorparameters = self.indexation_parameters[
                            "detectorparameters"
                        ]
                        print("LaueToolsframe.defaultParam", detectorparameters)
                        tw, ch = F2TC.calc_uflab(
                            np.array([spot1[0], spot2[0]]),
                            np.array([spot1[1], spot2[1]]),
                            detectorparameters,
                            kf_direction=self.kf_direction,
                        )
                        _dist = GT.distfrom2thetachi(
                            np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                        )
                    # TODO: add if self.datatype == 'pixels':

                    print("angular distance (q1,q2):  %.3f deg " % _dist)

                    self.nbclick_dist = 0
                    # self.twopoints = []
                    self.pickdistbtn.SetValue(False)
                    self.pickdistbtn.SetBackgroundColour(self.defaultColor)

                    print("RES =", _dist)
                    sentence = "Corresponding lattice planes angular distance"
                    sentence += "\n between two scattered directions : %.2f " % _dist
                    self.txtctrldistance.SetLabel(
                        "==> %s deg" % str(np.round(_dist, decimals=3))
                    )
                    # DOES NOT WORK !!
            #                    wx.MessageBox(sentence, 'INFO')

            if self.recongnisebtn.GetValue():
                self.nbclick_dist += 1
                print("self.nbclick_dist", self.nbclick_dist)

                if self.nbclick_dist == 1:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(event)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = event.xdata, event.ydata
                    else:
                        x, y = event.xdata, event.ydata

                    self.twospots = [(x, y)]

                if self.nbclick_dist == 2:
                    if self.UCEP.GetValue():
                        closestExpSpot = self.Annotate_exp(event)
                        if closestExpSpot is not None:
                            x, y = closestExpSpot[:2]
                        else:
                            x, y = event.xdata, event.ydata
                    else:
                        x, y = event.xdata, event.ydata

                    self.twospots.append((x, y))

                    #                     self.Reckon_2pts(event)
                    self.Reckon_2pts_new(event)
        else:
            print("outside!! axes object")

    def EnterCombokeymaterial(self, event):
        """
        in ManualIndexFrame
        """
        item = event.GetSelection()
        self.key_material = self.list_materials[item]

        self.sb.SetStatusText(
            "Selected material: %s" % str(self.parent.dict_Materials[self.key_material])
        )
        event.Skip()

    def store_pts(self, evt):
        self.points.append((evt.xdata, evt.ydata))
        print("# selected points", self.nbclick)
        print("Coordinates(%.3f,%.3f)" % (evt.xdata, evt.ydata))
        self.nbclick += 1

    def drawAnnote_exp(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index
        #ManualIndexFrame
        """
        if (x, y) in self.drawnAnnotations_exp:
            markers = self.drawnAnnotations_exp[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            t1 = axis.text(x + 1, y + 1, "%d" % (annote[0]), size=8)
            t2 = axis.text(x + 1, y - 1, "%.1f" % (annote[1]), size=8)
            if matplotlibversion <= "0.99.1":
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False
                )
            else:
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None"
                )  # matplotlib 0.99.1.1
            self.drawnAnnotations_exp[(x, y)] = (t1, t2, m)
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()

    def drawSpecificAnnote_exp(self, annote):
        """
        ManualIndexFrame
        """
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_exp if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_exp(self.axes, x, y, a)

    def Annotate_exp(self, event):
        """
        ManualIndexFrame
        """
        xtol = 20
        ytol = 20

        if self.datatype == "pixels":
            xtol = 200
            ytol = 200
        # """
        # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
        # self.Data_index_expspot = np.arange(len(self.Data_X))
        # """
        #         xdata, ydata, annotes = (self.Data_X,
        #                                  self.Data_Y,
        #                                  zip(self.Data_index_expspot, self.Data_I))

        xdata, ydata, annotes = (
            self.Data_X,
            self.Data_Y,
            list(zip(self.selectedAbsoluteSpotIndices_init, self.Data_I)),
        )
        # print self.Idat
        # print self.Mdat
        # print annotes
        self._dataANNOTE_exp = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        print("in Annotate_exp", clickX, clickY)

        annotes = []
        for x, y, a in self._dataANNOTE_exp:
            if (clickX - xtol < x < clickX + xtol) and (
                clickY - ytol < y < clickY + ytol
            ):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes == []:
            return None

        annotes.sort()
        _distance, x, y, annote = annotes[0]
        print("the nearest experimental point is at(%.2f,%.2f)" % (x, y))
        print("with index %d and intensity %.1f" % (annote[0], annote[1]))
        self.clicked_indexSpot.append(annote[0])
        self.drawAnnote_exp(self.axes, x, y, annote)
        for l in self.links_exp:
            l.drawSpecificAnnote_exp(annote)

        self.updateStatusBar(x, y, annote)

        return x, y, annote[0], annote[1]

    def updateStatusBar(self, x, y, annote, spottype="exp"):

        if self.datatype == "2thetachi":
            Xplot = "2theta"
            Yplot = "chi"
        else:
            Xplot = "x"
            Yplot = "y"

        #         if spottype == 'theo':
        #             self.sb.SetStatusText(("%s= %.2f " % (Xplot, x) +
        #                                     " %s= %.2f " % (Yplot, y) +
        #                                     "  HKL=%s " % str(annote)),
        #                                     0)

        if spottype == "exp":

            self.sb.SetStatusText(
                (
                    "%s= %.2f " % (Xplot, x)
                    + " %s= %.2f " % (Yplot, y)
                    + "   Spotindex=%d " % annote[0]
                    + "   Intensity=%.2f" % annote[1]
                ),
                0,
            )

    def locateSpot(self, evt):
        if self.findspotchck.GetValue():
            spotindex = int(self.spotindexspinctrl.GetValue())
            print("Showing spot #: %d" % spotindex)
            X = self.Data_X[spotindex]
            Y = self.Data_Y[spotindex]
            Intens = self.Data_I[spotindex]

            if len(self.axes.patches) > 0:
                self.RemoveLastRectangle()
            self.addPatchRectangle(X, Y, size=self.locatespotmarkersize)

            self.updateStatusBar(X, Y, (spotindex, Intens), spottype="exp")

            self.canvas.draw()

    def RemoveLastRectangle(self):

        if type(self.axes.patches[-1]) == type(Rectangle((1, 1), 1, 1)):
            del self.axes.patches[-1]

    #            print "deleted rectangle"

    def addPatchRectangle(self, X, Y, size=50):
        hsize = size / 2.0
        self.axes.add_patch(
            Rectangle((X - hsize, Y - hsize), size, size, fill=False, color="r")
        )

    def close(self, event):
        self.Close(True)

    def all_reds(self):
        for butt in self.listbuttons:
            butt.SetBackgroundColour(self.defaultColor)

    def what_was_pressed(self, flag):
        if flag:
            print("-------------------")
            print([butt.GetValue() for butt in self.listbuttons])
            print("self.listbuttonstate", self.listbuttonstate)

    def T2(self, evt):
        self.what_was_pressed(0)
        if self.listbuttonstate[0] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pickdistbtn.SetValue(True)
            self.pickdistbtn.SetBackgroundColour("Green")
            self.listbuttonstate = [1, 0, 0]
            self.nbclick_dist = 0
        else:
            self.pickdistbtn.SetBackgroundColour(self.defaultColor)
            self.pickdistbtn.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

    def T3(self, evt):  # Recognise distance
        self.what_was_pressed(0)
        if self.listbuttonstate[1] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.recongnisebtn.SetValue(True)
            self.recongnisebtn.SetBackgroundColour("Green")
            self.listbuttonstate = [0, 1, 0]
            self.nbclick_dist = 0

        else:
            self.recongnisebtn.SetBackgroundColour(self.defaultColor)
            self.recongnisebtn.SetValue(False)
            self.listbuttonstate = [0, 0, 0]
            evt.Skip()

    def T6(self, evt):
        self.what_was_pressed(0)
        if self.listbuttonstate[2] == 0:
            self.all_reds()
            self.allbuttons_off()
            self.pointButton6.SetValue(True)
            self.pointButton6.SetBackgroundColour("Green")
            self.listbuttonstate = [0, 0, 1]
        else:
            self.pointButton6.SetBackgroundColour(self.defaultColor)
            self.pointButton6.SetValue(False)
            self.listbuttonstate = [0, 0, 0]

            evt.Skip()

    def allbuttons_off(self):
        for butt in self.listbuttons:
            butt.SetValue(False)

    def readlogicalbuttons(self):
        return [butt.GetValue() for butt in self.listbuttons]

    def select_2pts(self, evt, displayMesssage=False):
        """#pick distance
        in ManualIndexFrame
        """
        toreturn = None
        if self.nbclick_dist <= 2:
            if self.nbclick_dist == 1:
                self.twopoints = []

            self.twopoints.append([evt.xdata, evt.ydata])
            print("# selected points", self.nbclick_dist)
            print("Coordinates(%.3f,%.3f)" % (evt.xdata, evt.ydata))
            print("click nb", self.nbclick_dist)

            if len(self.twopoints) == 2:
                # compute angular distance:
                spot1 = self.twopoints[0]  # (X, Y) (e.g. 2theta, chi)
                spot2 = self.twopoints[1]
                if self.datatype == "2thetachi":
                    _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))

                elif self.datatype == "gnomon":
                    tw, ch = IIM.Fromgnomon_to_2thetachi(
                        [
                            np.array([spot1[0], spot2[0]]),
                            np.array([spot1[1], spot2[1]]),
                        ],
                        0,
                    )[:2]
                    _dist = GT.distfrom2thetachi(
                        np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                    )

                elif self.datatype == "pixels":
                    detectorparameters = self.indexation_parameters["AllDataToIndex"][
                        "detectorparameters"
                    ]
                    print("LaueToolsframe.defaultParam", detectorparameters)
                    tw, ch = F2TC.calc_uflab(
                        np.array([spot1[0], spot2[0]]),
                        np.array([spot1[1], spot2[1]]),
                        detectorparameters,
                        kf_direction=self.kf_direction,
                    )
                    _dist = GT.distfrom2thetachi(
                        np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                    )

                print("Angular distance :  %.3f deg " % _dist)

                toreturn = self.twopoints, _dist
                self.nbclick_dist = 0
                # self.twopoints = []
                self.pickdistbtn.SetValue(False)
                self.pickdistbtn.SetBackgroundColour(self.defaultColor)

        self.nbclick_dist += 1

        if displayMesssage:
            if toreturn is not None:

                twopoints, distangle = toreturn

                print("RES =", toreturn)
                #                sentence = 'Corresponding lattice planes angular distance'
                #                sentence += "\n between two scattered direction LatticePlane  : %.2f " % distangle
                #                dial = wx.MessageBox(sentence, 'INFO')
                self.nbclick_dist = 1
                return None

        return toreturn

    def Reckon_2pts_new(self, evt):
        """ Start indexation
        Index Laue Pattern by Recognising distance from two user clicked spots
        First press button 'recognise distance' then click on two spots

        in ManualIndexFrame

        Activate the recognition of the angle between two atomic planes in a LUT of angular distances:
        Need the selection of 2 spots, compute the interplanar angle, find all possibilities in the LUT
        Open the recognitionBox with possible solutions with indexing quality based on matching rate
        """
        if self.twospots is not None:  # two spots are selected
            self.recongnisebtn.SetValue(False)
            self.recongnisebtn.SetBackgroundColour(self.defaultColor)
            print("twospots", self.twospots)
            spot1 = self.twospots[0]
            spot2 = self.twospots[1]
            print("---Selected points")

            if self.datatype == "2thetachi":
                _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))
                last_index = self.clicked_indexSpot[-1]
                print("last clicked", last_index)
                if len(self.clicked_indexSpot) > 1:
                    last_last_index = self.clicked_indexSpot[-2]
                    print("former clicked", last_last_index)
                print("(2theta, chi) ")

            elif self.datatype == "gnomon":
                tw, ch = IIM.Fromgnomon_to_2thetachi(
                    [np.array([spot1[0], spot2[0]]), np.array([spot1[1], spot2[1]])], 0
                )[:2]
                print("gnomon")
                last_index = self.clicked_indexSpot[-1]
                print("last clicked", last_index)
                if len(self.clicked_indexSpot) > 1:
                    last_last_index = self.clicked_indexSpot[-2]
                    print("former clicked", last_last_index)

                spot1 = [tw[0], ch[0]]
                spot2 = [tw[1], ch[1]]
                _dist = GT.distfrom2thetachi(
                    np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                )

            elif self.datatype == "pixels":
                print("pixels")
                last_index = self.clicked_indexSpot[-1]
                print("last clicked", last_index)
                if len(self.clicked_indexSpot) > 1:
                    last_last_index = self.clicked_indexSpot[-2]
                    print("former clicked", last_last_index)

                detectorparameters = self.indexation_parameters["detectorparameters"]
                print("LaueToolsframe.defaultParam", detectorparameters)
                tw, ch = F2TC.calc_uflab(
                    np.array([spot1[0], spot2[0]]),
                    np.array([spot1[1], spot2[1]]),
                    detectorparameters,
                    kf_direction=self.kf_direction,
                )
                spot1 = [tw[0], ch[0]]
                spot2 = [tw[1], ch[1]]
                _dist = GT.distfrom2thetachi(
                    np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                )

            print("spot1 [%.3f,%.3f]" % (tuple(spot1)))
            print("spot2 [%.3f,%.3f]" % (tuple(spot2)))
            print(
                "Angular distance between corresponding reflections for recognition :  %.3f deg "
                % _dist
            )

            spot1_ind = last_last_index
            spot2_ind = last_index

            print("spot1 index, spot2 index", spot1_ind, spot2_ind)

        if (
            spot1_ind not in self.selectedAbsoluteSpotIndices
            or spot2_ind not in self.selectedAbsoluteSpotIndices
        ):
            wx.MessageBox(
                "You must select two spots displayed in the current plot", "info"
            )
            return

        if self.COCD.GetValue():
            self.onlyclosest = 1
        else:
            self.onlyclosest = 0

        spot_index_central = spot1_ind
        nbmax_probed = spot2_ind + 1

        # if first clicked spot index is larger than second spot index
        # then try to recognise all distances between spot1 and [0, spot1+1]
        if spot1_ind > spot2_ind:
            nbmax_probed = spot1_ind + 1

        energy_max = float(self.SCEmax.GetValue())

        ResolutionAngstrom = None
        nb_exp_spots_data = len(self.data[0])

        print("ResolutionAngstrom in Reckon_2pts_new() indexation", ResolutionAngstrom)
        print("nb of spots in Laue Pattern ", nb_exp_spots_data)

        #         # there is no precomputed angular distances between spots
        #         if not self.parent.ClassicalIndexation_Tabledist:
        #             print "main frame ClassicalIndexation_Tabledist is None. Calculate it!"
        #             # select 1rstly spots that have not been indexed and 2ndly reduced list by user
        # #             index_to_select = np.take(self.parent.current_exp_spot_index_list,
        # #                                       np.arange(nb_exp_spots_data))
        #             print "\n\n\n*****\n"
        #             print "self.parent.current_exp_spot_index_list",self.parent.current_exp_spot_index_list
        #             print "np.arange(nb_exp_spots_data)",np.arange(nb_exp_spots_data)
        #             print "nb_exp_spots_data",nb_exp_spots_data
        #             print "self.abs_spotindex",self.abs_spotindex
        # #                                       np.arange(nb_exp_spots_data))
        # #                                       np.arange(nb_exp_spots_data))"
        #             index_to_select = np.take(self.parent.current_exp_spot_index_list,
        #                                       self.abs_spotindex)
        #
        #             self.select_theta = self.parent.data_theta[index_to_select]
        #             self.select_chi = self.parent.data_chi[index_to_select]
        #             self.select_I = self.parent.data_I[index_to_select]
        #             print "len(self.select_theta)",len(self.select_theta)
        #             # print select_chi
        #             listcouple = np.array([self.select_theta, self.select_chi]).T
        #             # compute angles between spots
        #             Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        # there is no precomputed angular distances between spots
        if not self.parent.ClassicalIndexation_Tabledist:

            # Selection of spots among the whole data
            #             # MSSS number
            #             MatchingSpotSetSize = int(self.nbspotmaxformatching.GetValue())
            #
            #             # select 1rstly spots that have not been indexed and 2ndly reduced list by user
            #             index_to_select = np.take(self.current_exp_spot_index_list,
            #                                       np.arange(MatchingSpotSetSize))

            index_to_select = self.selectedAbsoluteSpotIndices

            self.select_theta = self.indexation_parameters["AllDataToIndex"][
                "data_theta"
            ][index_to_select]
            self.select_chi = self.indexation_parameters["AllDataToIndex"]["data_chi"][
                index_to_select
            ]
            self.select_I = self.indexation_parameters["AllDataToIndex"]["data_I"][
                index_to_select
            ]

            print("index_to_select", index_to_select)

            self.select_dataX = self.indexation_parameters["AllDataToIndex"][
                "data_pixX"
            ][index_to_select]
            self.select_dataY = self.indexation_parameters["AllDataToIndex"][
                "data_pixY"
            ][index_to_select]
            # print select_theta
            # print select_chi

            listcouple = np.array([self.select_theta, self.select_chi]).T
            # compute angles between spots
            Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        else:
            print(
                "Reuse computed ClassicalIndexation_Tabledist with size: %d"
                % len(self.parent.ClassicalIndexation_Tabledist)
            )

        self.data = (
            2 * self.select_theta,
            self.select_chi,
            self.select_I,
            self.parent.DataPlot_filename,
        )

        self.key_material = str(self.combokeymaterial.GetValue())
        latticeparams = self.parent.dict_Materials[self.key_material][1]
        B = CP.calc_B_RR(latticeparams)
        # print type(key_material)
        # print type(nbmax_probed)
        # print type(energy_max)

        # read maximum index of hkl for building angles Look Up Table(LUT)
        nLUT = int(self.nlut.GetValue())

        rough_tolangle = float(self.DRTA.GetValue())
        fine_tolangle = float(self.matr_ctrl.GetValue())
        Nb_criterium = 3

        NBRP = 3

        # detector geometry
        detectorparameters = {}
        detectorparameters["kf_direction"] = self.parent.kf_direction
        detectorparameters["detectorparameters"] = self.parent.defaultParam
        detectorparameters["detectordiameter"] = self.parent.detectordiameter
        detectorparameters["pixelsize"] = self.parent.pixelsize
        detectorparameters["dim"] = self.parent.framedim

        print("detectorparameters", detectorparameters)

        restrictLUT_cubicSymmetry = True
        set_central_spots_hkl = None

        if self.sethklchck.GetValue():
            strhkl = str(self.sethklcentral.GetValue())[1:-1].split(",")
            H, K, L = strhkl
            H, K, L = int(H), int(K), int(L)
            # LUT with cubic symmetry does not have negative L
            if L < 0:
                restrictLUT_cubicSymmetry = False

            set_central_spots_hkl = [int(H), int(K), int(L)]

        # restrict LUT if allowed and if crystal is cubic
        if restrictLUT_cubicSymmetry:
            restrictLUT_cubicSymmetry = CP.hasCubicSymmetry(self.key_material)

        print("set_central_spots_hkl", set_central_spots_hkl)
        print("restrictLUT_cubicSymmetry", restrictLUT_cubicSymmetry)

        verbosedetails = self.verbose.GetValue()

        self.set_central_spots_hkl = set_central_spots_hkl
        self.spot_index_central = spot_index_central
        self.Nb_criterium = Nb_criterium
        self.NBRP = NBRP
        self.B = B
        self.energy_max = energy_max
        self.ResolutionAngstrom = ResolutionAngstrom
        self.nLUT = nLUT
        self.rough_tolangle = rough_tolangle
        self.fine_tolangle = fine_tolangle

        USETHREAD = 1
        if USETHREAD:
            # with a thread 2----------------------------------------
            

            self.UBs_MRs = None
            fctparams = [
                INDEX.getUBs_and_MatchingRate,
                (
                    spot1_ind,
                    spot2_ind,
                    rough_tolangle,
                    _dist,
                    spot1,
                    spot2,
                    nLUT,
                    B,
                    2 * self.select_theta,
                    self.select_chi,
                ),
                {
                    "set_hkl_1": set_central_spots_hkl,
                    "key_material": self.key_material,
                    "emax": energy_max,
                    "ResolutionAngstrom": ResolutionAngstrom,
                    "ang_tol_MR": fine_tolangle,
                    "detectorparameters": detectorparameters,
                    "LUT": None,
                    "MaxRadiusHKL": False,
                    "verbose": 0,
                    "verbosedetails": verbosedetails,
                    "Minimum_Nb_Matches": Nb_criterium,
                    "worker": None,
                },
            ]

            # update DataSetObject
            self.DataSet.dim = detectorparameters["dim"]
            self.DataSet.pixelsize = detectorparameters["pixelsize"]
            self.DataSet.detectordiameter = detectorparameters["detectordiameter"]
            self.DataSet.kf_direction = detectorparameters["kf_direction"]
            self.DataSet.key_material = self.key_material
            self.DataSet.emin = 5
            self.DataSet.emax = energy_max

            print("self.DataSet.detectordiameter", self.DataSet.detectordiameter)

            self.TGframe = TG.ThreadHandlingFrame(
                self,
                -1,
                threadFunctionParams=fctparams,
                parentAttributeName_Result="UBs_MRs",
                parentNextFunction=self.simulateAllResults,
            )
            self.TGframe.OnStart(1)
            self.TGframe.Show(True)

            # will set self.UBs_MRs to the output of INDEX.getUBs_and_MatchingRate
            return

        else:
            # case USETHREAD == 0
            # ---- indexation in Reckon_2pts_new() in Manualindexframe
            self.worker = None
            self.UBs_MRs = INDEX.getUBs_and_MatchingRate(
                spot1_ind,
                spot2_ind,
                rough_tolangle,
                _dist,
                spot1,
                spot2,
                nLUT,
                B,
                2 * self.select_theta,
                self.select_chi,
                set_hkl_1=set_central_spots_hkl,
                key_material=self.key_material,
                emax=energy_max,
                ResolutionAngstrom=ResolutionAngstrom,
                ang_tol_MR=fine_tolangle,
                detectorparameters=detectorparameters,
                LUT=None,
                MaxRadiusHKL=False,
                verbose=0,
                verbosedetails=verbosedetails,
                Minimum_Nb_Matches=Nb_criterium,
                worker=self.worker,
            )

            self.bestmat, stats_res = self.UBs_MRs
            # update DataSetObject
            self.DataSet.dim = detectorparameters["dim"]
            self.DataSet.pixelsize = detectorparameters["pixelsize"]
            self.DataSet.detectordiameter = detectorparameters["detectordiameter"]
            self.DataSet.kf_direction = detectorparameters["kf_direction"]
            self.DataSet.key_material = self.key_material
            self.DataSet.emin = 5
            self.DataSet.emax = energy_max

            print("self.DataSet.detectordiameter", self.DataSet.detectordiameter)

            self.simulateAllResults()

    def simulateAllResults(self):

        print("Entering simulateAllResults\n\n")
        print("self.UBs_MRs", self.UBs_MRs)

        if self.UBs_MRs is None:
            return

        self.bestmat, stats_res = self.UBs_MRs

        #         print "stats_res", stats_res
        nb_sol = len(self.bestmat)

        keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[self.key_material][1])

        if self.set_central_spots_hkl not in (None, [None]):
            keep_only_equivalent = False

        if nb_sol > 1:
            print("Merging matrices")
            print("keep_only_equivalent = %s" % keep_only_equivalent)
            self.bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(
                self.bestmat,
                stats_res,
                self.Nb_criterium,
                tol=0.0001,
                keep_only_equivalent=keep_only_equivalent,
            )

        print("stats_res", stats_res)
        nb_sol = len(self.bestmat)
        print("Max. Number of Solutions", self.NBRP)
        print("spot_index_central", self.spot_index_central)

        if nb_sol:
            print("%d matrice(s) found" % nb_sol)
            print(self.bestmat)
            print("Each Matrix is stored in 'MatIndex_#' for further simulation")
            for k in range(nb_sol):
                self.parent.dict_Rot["MatIndex_%d" % (k + 1)] = self.bestmat[k]

            stats_properformat = []
            for elem in stats_res:
                elem[0] = int(elem[0])
                elem[1] = int(elem[1])
                stats_properformat.append(tuple(elem))

            # one central point were used for distance recognition
            self.TwicethetaChi_solution = [0 for p in range(nb_sol)]
            paramsimul = []

            for p in range(nb_sol):

                orientmatrix = self.bestmat[p]

                # only orientmatrix, self.key_material are used ----------------------
                vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
                # old definition of grain
                grain = [vecteurref, [1, 1, 1], orientmatrix, self.key_material]
                # ------------------------------------------------------------------

                #                 print "self.indexation_parameters.keys()", self.indexation_parameters.keys()
                #                 print "self.indexation_parameters['detectordiameter']", self.indexation_parameters['detectordiameter']
                TwicethetaChi = LAUE.SimulateResult(
                    grain,
                    5,
                    self.energy_max,
                    self.indexation_parameters,
                    ResolutionAngstrom=self.ResolutionAngstrom,
                    fastcompute=1,
                )
                self.TwicethetaChi_solution[p] = TwicethetaChi
                paramsimul.append((grain, 5, self.energy_max))

            self.indexation_parameters["paramsimul"] = paramsimul
            self.indexation_parameters["bestmatrices"] = self.bestmat
            self.indexation_parameters[
                "TwicethetaChi_solutions"
            ] = self.TwicethetaChi_solution
            self.indexation_parameters["plot_xlim"] = self.xlim
            self.indexation_parameters["plot_ylim"] = self.ylim
            self.indexation_parameters["flipyaxis"] = self.data_dict["flipyaxis"]

            datatype = self.datatype
            # display "statistical" results
            if self.datatype == "gnomon":
                print("plot results in 2thetachi coordinates")
                datatype = "2thetachi"

            RRCBClassical = RecognitionResultCheckBox(
                self,
                -1,
                "Classical Indexation Solutions",
                stats_properformat,
                self.data,
                self.rough_tolangle,
                self.fine_tolangle,
                key_material=self.key_material,
                emax=self.energy_max,
                ResolutionAngstrom=self.ResolutionAngstrom,
                kf_direction=self.kf_direction,
                datatype=datatype,
                data_2thetachi=self.data_2thetachi,
                data_XY=self.data_XY,
                ImageArray=self.ImageArray,
                CCDdetectorparameters=self.indexation_parameters,
                IndexationParameters=self.indexation_parameters,
                StorageDict=self.StorageDict,
                DataSetObject=self.DataSet,
            )

            RRCBClassical.Show(True)

        else:  # any matrix was found
            print("!!  Nothing found   !!!")
            print("with LaueToolsGUI.Reckon_2pts_new()")
            # MessageBox will freeze the computer
        #             wx.MessageBox('! NOTHING FOUND !\nTry to reduce the Minimum Number Matched Spots to catch something!', 'INFO')

        self.nbclick_dist = 0
        self.recongnisebtn.SetValue(False)


# --- ---------------  Plot limits board  parameters
class PlotLimitsBoard(wx.Dialog):
    """
    Class to set limits parameters of plot
    """

    def __init__(self, parent, _id, title, data_dict):
        """
        initialize board window

        """
        wx.Dialog.__init__(self, parent, _id, title, size=(400, 250))

        self.parent = parent
        print("self.parent", self.parent)

        self.data_dict = data_dict

        xlim = self.data_dict["xlim"]
        ylim = self.data_dict["ylim"]

        self.init_xlim = copy.copy(xlim)
        self.init_ylim = copy.copy(ylim)

        #         data_dict['datatype'] = self.datatype
        #         data_dict['dataXmin'] = min(self.Data_X)
        #         data_dict['dataXmax'] = max(self.Data_X)
        #         data_dict['dataYmin'] = min(self.Data_Y)
        #         data_dict['dataYmax'] = max(self.Data_Y)
        #         data_dict['xlim'] = self.xlim
        #         data_dict['ylim'] = self.ylim
        #         data_dict['kf_direction'] = self.kf_direction

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        txt1 = wx.StaticText(self, -1, "X and Y limits controls")
        txt2 = wx.StaticText(self, -1, "Data type: %s" % self.data_dict["datatype"])

        txt1.SetFont(font3)

        self.txtctrl_xmin = wx.TextCtrl(
            self, -1, str(xlim[0]), style=wx.TE_PROCESS_ENTER
        )
        self.txtctrl_xmax = wx.TextCtrl(
            self, -1, str(xlim[1]), style=wx.TE_PROCESS_ENTER
        )
        self.txtctrl_ymin = wx.TextCtrl(
            self, -1, str(ylim[0]), style=wx.TE_PROCESS_ENTER
        )
        self.txtctrl_ymax = wx.TextCtrl(
            self, -1, str(ylim[1]), style=wx.TE_PROCESS_ENTER
        )

        self.txtctrl_xmin.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)
        self.txtctrl_xmax.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)
        self.txtctrl_ymin.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)
        self.txtctrl_ymax.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)

        fittodatabtn = wx.Button(self, -1, "Fit to Data")
        fittodatabtn.Bind(wx.EVT_BUTTON, self.onFittoData)

        acceptbtn = wx.Button(self, -1, "Accept")
        cancelbtn = wx.Button(self, -1, "Cancel")

        acceptbtn.Bind(wx.EVT_BUTTON, self.onAccept)
        cancelbtn.Bind(wx.EVT_BUTTON, self.onCancel)

        if WXPYTHON4:
            grid = wx.GridSizer(5, 10, 10)
        else:
            grid = wx.GridSizer(6, 5)

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "Y"), wx.ALIGN_CENTER_HORIZONTAL)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "MAX"), wx.ALIGN_CENTER_HORIZONTAL)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(self.txtctrl_ymax)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, "X    min"), wx.ALIGN_RIGHT)
        grid.Add(self.txtctrl_xmin)
        grid.Add(fittodatabtn)
        grid.Add(self.txtctrl_xmax)
        grid.Add(wx.StaticText(self, -1, "MAX"))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(self.txtctrl_ymin)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "min"), wx.ALIGN_CENTER_HORIZONTAL)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        btnssizer = wx.BoxSizer(wx.HORIZONTAL)
        btnssizer.Add(acceptbtn, 0, wx.ALL)
        btnssizer.Add(cancelbtn, 0, wx.ALL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt1)
        vbox.Add(txt2)
        vbox.Add(grid)
        vbox.Add(btnssizer)

        self.SetSizer(vbox)

    def onEnterValue(self, evt):
        self.readvalues()
        self.updateplot()

    def onFittoData(self, evt):

        xmin = self.data_dict["dataXmin"]
        xmax = self.data_dict["dataXmax"]
        ymin = self.data_dict["dataYmin"]
        ymax = self.data_dict["dataYmax"]

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.txtctrl_xmin.SetValue(str(self.xmin))
        self.txtctrl_xmax.SetValue(str(self.xmax))
        self.txtctrl_ymin.SetValue(str(self.ymin))
        self.txtctrl_ymax.SetValue(str(self.ymax))

        self.setxylim()

        self.updateplot()

    def updateplot(self):
        self.parent.xlim = self.xlim
        self.parent.ylim = self.ylim
        self.parent.getlimitsfromplot = False
        self.parent._replot()

    def readvalues(self):
        self.xmin = float(self.txtctrl_xmin.GetValue())
        self.xmax = float(self.txtctrl_xmax.GetValue())
        self.ymin = float(self.txtctrl_ymin.GetValue())
        self.ymax = float(self.txtctrl_ymax.GetValue())

        self.setxylim()

    def setxylim(self):
        self.xlim = (self.xmin, self.xmax)

        if self.parent.flipyaxis is not None:
            if not self.parent.flipyaxis:
                self.ylim = (self.ymin, self.ymax)
            else:
                # flip up for marccd roper...
                self.ylim = (self.ymax, self.ymin)
        else:
            self.ylim = (self.ymin, self.ymax)

    def onAccept(self, evt):

        self.readvalues()

        self.parent.xlim = self.xlim
        self.parent.ylim = self.ylim
        self.parent.getlimitsfromplot = False
        self.parent._replot()

        self.Close()

    def onCancel(self, evt):

        self.parent.xlim = self.init_xlim
        self.parent.ylim = self.init_ylim
        self.Close()


# --- ----------   DISPLAY SPlASH SCREEN
if WXPYTHON4:
    import wx.adv as wxadv

    class MySplash(wxadv.SplashScreen):
        """
        display during a given period a image and give back the window focus
        """

        def __init__(self, parent, duration=2000):
            # pick a splash image file you have in the working folder

            image_file = os.path.join(
                LaueToolsProjectFolder, "icons", "transmissionLaue_fcc_111.png"
            )
            print("image_file", image_file)
            bmp = wx.Bitmap(image_file)
            # covers the parent frame
            wxadv.SplashScreen(
                bmp,
                wxadv.SPLASH_CENTRE_ON_PARENT | wxadv.SPLASH_TIMEOUT,
                duration,
                parent,
                wx.ID_ANY,
            )

else:

    class MySplash(wx.SplashScreen):
        """
        display during a given period a image and give back the window focus
        """

        def __init__(self, parent, duration=2000):
            # pick a splash image file you have in the working folder

            image_file = os.path.join(
                LaueToolsProjectFolder,
                os.path.join("icons", "transmissionLaue_fcc_111.png"),
            )
            print("image_file", image_file)
            bmp = wx.Bitmap(image_file)
            # covers the parent frame
            wx.SplashScreen(
                bmp,
                wx.SPLASH_CENTRE_ON_PARENT | wx.SPLASH_TIMEOUT,
                duration,
                parent,
                wx.ID_ANY,
            )


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
    LaueToolsGUIApp = wx.App()
    LaueToolsframe = MainWindow(
        None,
        -1,
        "Image Viewer and PeakSearch Board",
        projectfolder=LaueToolsProjectFolder,
    )
    LaueToolsframe.Show()

    MySplash(LaueToolsframe, duration=500)

    LaueToolsGUIApp.MainLoop()


if __name__ == "__main__":
    start()
