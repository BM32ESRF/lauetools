from __future__ import absolute_import
r"""
DetectorCalibration.py is a GUI for microdiffraction Laue Pattern simulation to help visually
to match and exp. and theo. Laue patterns for mainly Detector geometry Calibration purpose.

This module belongs to the open source LaueTools project
with a free code repository at at gitlab.esrf.fr

(former version with python 2.7 at https://sourceforge.net/projects/lauetools/)

or for python3 and 2 in

https://gitlab.esrf.fr/micha/lauetools

J. S. Micha August 2019
mailto: micha --+at-+- esrf --+dot-+- fr
"""
import os
import time
import copy
import sys
import math
import numpy as np

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

from matplotlib.ticker import FuncFormatter

from matplotlib import __version__ as matplotlibversion
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                                NavigationToolbar2WxAgg as NavigationToolbar)

from matplotlib.figure import Figure

if sys.version_info.major == 3:
    from .. import dict_LaueTools as DictLT
    from .. import LaueGeometry as F2TC
    from .. import indexingAnglesLUT as INDEX
    from .. import indexingImageMatching as IIM
    from .. import matchingrate
    from .. import lauecore as LAUE
    from .. import findorient as FindO
    from .. import FitOrient as FitO
    from . import spotslinkeditor as SLE
    from . import LaueSpotsEditor as LSEditor
    from .. import generaltools as GT
    from .. import IOLaueTools as IOLT
    from .. import CrystalParameters as CP
    from . import DetectorParameters as DP
    from . ResultsIndexationGUI import RecognitionResultCheckBox
    from . import OpenSpotsListFileGUI as OSLFGUI
    from .. import orientations as ORI

else:

    import dict_LaueTools as DictLT
    import LaueGeometry as F2TC
    import indexingAnglesLUT as INDEX
    import indexingImageMatching as IIM
    import matchingrate
    import lauecore as LAUE
    import findorient as FindO
    import FitOrient as FitO
    import GUI.spotslinkeditor as SLE
    import GUI.LaueSpotsEditor as LSEditor
    import generaltools as GT
    import IOLaueTools as IOLT
    import CrystalParameters as CP
    import GUI.DetectorParameters as DP
    from GUI.ResultsIndexationGUI import RecognitionResultCheckBox
    import OpenSpotsListFileGUI as OSLFGUI
    import orientations as ORI



DEG = DictLT.DEG
PI = DictLT.PI
CST_ENERGYKEV = DictLT.CST_ENERGYKEV

# --- sub class panels
class PlotRangePanel(wx.Panel):
    """
    class for panel dealing with plot range and kf_direction
    """
    def __init__(self, parent):
        """ init """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()
        self.mainframeparent = parent.GetParent().GetParent().GetParent()

        # print("mainframe in PlotRangePanel", self.mainframe)
        # print('linked github version DetectorCalibration')

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        openbtn = wx.Button(self, -1, "Import Spots List", (5, 5))#, (200, 60))
        openbtn.SetFont(font3)

        t1 = wx.StaticText(self, -1, "2theta Range:")
        t2 = wx.StaticText(self, -1, "Chi Range:")

        self.mean2theta = wx.TextCtrl(self, -1, "90")#, (40, -1))
        self.meanchi = wx.TextCtrl(self, -1, "0")#, (40, -1))
        pm1 = wx.StaticText(self, -1, "+/-")
        pm2 = wx.StaticText(self, -1, "+/-")
        self.range2theta = wx.TextCtrl(self, -1, "45")
        self.rangechi = wx.TextCtrl(self, -1, "40")

        openbtn.Bind(wx.EVT_BUTTON, self.opendata)
        self.mean2theta.Bind(wx.EVT_TEXT, self.set_init_plot_True)
        self.meanchi.Bind(wx.EVT_TEXT, self.set_init_plot_True)
        self.range2theta.Bind(wx.EVT_TEXT, self.set_init_plot_True)
        self.rangechi.Bind(wx.EVT_TEXT, self.set_init_plot_True)

        self.shiftChiOrigin = wx.CheckBox(
            self, -1, "Shift Chi origin of Exp. Data")
        self.shiftChiOrigin.SetValue(False)

        t5 = wx.StaticText(self, -1, "SpotSize")
        self.spotsizefactor = wx.TextCtrl(self, -1, "1.")

        # Warning button id is 52 and used
        b3 = wx.Button(self, 52, "Update Plot")
        b3.SetFont(font3)

        # layout
        h1box = wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add(t1, 0, wx.EXPAND, 10)
        h1box.Add(self.mean2theta, 0, wx.EXPAND, 10)
        h1box.Add(pm1, 0, wx.EXPAND, 10)
        h1box.Add(self.range2theta, 0, wx.EXPAND, 10)
        h1box.Add(t2, 0, wx.EXPAND, 10)
        h1box.AddSpacer(5)
        h1box.Add(self.meanchi, 0, wx.EXPAND, 10)
        h1box.Add(pm2, 0, wx.EXPAND, 10)
        h1box.Add(self.rangechi, 0, wx.EXPAND, 10)
        h1box.Add(self.shiftChiOrigin, 0, wx.EXPAND, 10)

        h3box = wx.BoxSizer(wx.HORIZONTAL)
        h3box.Add(t5, 0, wx.EXPAND, 10)
        h3box.Add(self.spotsizefactor, 0, wx.EXPAND, 10)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(1)
        vbox.Add(openbtn, 0, wx.EXPAND, 5)
        vbox.Add(h1box, 0, wx.EXPAND|wx.ALL, 1)
        vbox.Add(h3box, 0, wx.EXPAND|wx.ALL, 1)
        vbox.Add(b3, 0, wx.EXPAND, 1)

        self.SetSizer(vbox)

        # tooltips
        tp1 = "raw central value (deg) of 2theta on CCD camera (kf scattered vector is defined by 2theta and chi)"
        tp2 = "raw central value (deg) of chi on CCD camera (kf scattered vector is defined by 2theta and chi)"
        tp3 = "range amplitude (deg) around the central 2theta value"
        tp4 = "range amplitude (deg) around the central chi value"

        tp5 = "Set experimental spots radius"

        tpb3 = "Replot simulated Laue spots"

        t1.SetToolTipString(tp1)
        self.mean2theta.SetToolTipString(tp1)

        t2.SetToolTipString(tp2)
        self.meanchi.SetToolTipString(tp2)

        t5.SetToolTipString(tp5)
        self.meanchi.SetToolTipString(tp3)

        self.range2theta.SetToolTipString(tp3)
        self.rangechi.SetToolTipString(tp4)

        self.shiftChiOrigin.SetToolTipString(
            "Check to shift the chi angles of experimental spots by the central chi value")

        b3.SetToolTipString(tpb3)

        openbtn.SetToolTipString('Open peaks list .dat (pixel positions and peak props) or .cor file '
                                            'pixel positions+ scattering angles + peak props + knowledge of detector calibtation')

    def set_init_plot_True(self, _):
        print("reset init_plot to True")
        self.mainframe.init_plot = True

    def opendata(self, evt):
        """open a GUI to select a file containing peaks list (.dat, or .cor)
        """
        OSLFGUI.OpenPeakList(self.mainframe)

        selectedFile = self.mainframe.DataPlot_filename

        print("\n\nIn opendata(): Selected file ", selectedFile)
        #print("dict_spotsproperties", self.mainframe.dict_spotsproperties)

        self.mainframe.initialParameter["dirname"] = self.mainframe.dirnamepklist
        self.mainframe.initialParameter["filename"] = selectedFile

        # prefix_filename, extension_filename = self.DataPlot_filename.split('.')
        prefix_filename = selectedFile.rsplit(".", 1)[0]

        # get PeakListDatFileName
        # cor file have been created from .dat  if name is dat_#######.cor
        if prefix_filename.startswith("dat_"):
            CalibrationFile = prefix_filename[4:] + ".dat"

            if not CalibrationFile in os.listdir(self.mainframe.dirnamepklist):
                wx.MessageBox('%s corresponding to the .dat file (all peaks properties) of '
                '%s is missing. \nPlease, change the name of %s (remove "dat_" for instance) '
                    'to work with %s but without peaks properties (shape, size, Imax, etc...)' %(CalibrationFile, selectedFile, selectedFile, selectedFile), 'Info')
                raise ValueError('%s corresponding to .dat file of %s is missing. '
                'Change the name of %s (remove "dat_" '
                'for instance)' % (CalibrationFile, selectedFile, selectedFile))

        else:
            CalibrationFile = selectedFile

        print("Calibrating with file: %s" % CalibrationFile)

        self.mainframe.filename = CalibrationFile

        self.mainframe.CCDParam = self.mainframe.defaultParam
        self.mainframe.ccdparampanel.pixelsize_txtctrl.SetValue(str(self.mainframe.pixelsize))
        self.mainframe.ccdparampanel.detectordiameter_txtctrl.SetValue(str(self.mainframe.detectordiameter))

        self.mainframe.update_data(evt)


class CrystalParamPanel(wx.Panel):
    """
    class for crystal simulation parameters
    """

    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        self.UBmatrix = DictLT.dict_Rot[self.mainframe.DEFAULT_UBMATRIX_CALIB]

        # print("self.mainframe in CrystalParamPanel", self.mainframe)
        # widgets layout
        t1 = wx.StaticText(self, -1, "Energy min/max(keV): ")
        self.eminC = wx.SpinCtrl( self, -1, "5", min=5, max=149)
        self.emaxC = wx.SpinCtrl( self, -1, "27", min=6, max=150)

        self.listsorted_materials = sorted(DictLT.dict_Materials.keys())
        t2 = wx.StaticText(self, -1, "Element")
        self.comboElem = wx.ComboBox(self, -1, "Ge",
                                        choices=self.listsorted_materials,
                                        style=wx.CB_READONLY)

        t3 = wx.StaticText(self, -1, "Tmatrix")  # in sample Frame columns are a*,b*,c* expressed in is,js,ks vector frame
        self.comboBmatrix = wx.ComboBox(self, 2424, "Identity",
                                            choices=sorted(DictLT.dict_Transforms.keys()),
                                            style=wx.CB_READONLY)

        t4 = wx.StaticText(self, -1, "Orient Matrix (UB)")

        self.comboMatrix = wx.ComboBox(self, 2525, self.mainframe.DEFAULT_UBMATRIX_CALIB,
                                        choices=list(DictLT.dict_Rot.keys()))

        #GT.propose_orientation_from_hkl(HKL, target2theta=90., randomrotation=False)
        self.btncenteronhkl = wx.Button(self, -1, "center on hkl = ")
        self.tchc = wx.TextCtrl(self, -1, "1", size=(30, -1))
        self.tckc = wx.TextCtrl(self, -1, "1", size=(30, -1))
        self.tclc = wx.TextCtrl(self, -1, "1", size=(30, -1))

        self.btn_mergeUB = wx.Button(self, -1, "set UB with B")

        t5 = wx.StaticText(self, -1, "Extinctions")
        self.comboExtinctions = wx.ComboBox(self, -1, "Diamond",
                                            choices=list(DictLT.dict_Extinc.keys()))

        self.comboExtinctions.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeExtinc)
        #         self.comboTransforms.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeTransforms)

        b1 = wx.Button(self, 1010, "Enter UB")
        b2 = wx.Button(self, 1011, "Store UB")
        btn_sortUBsname = wx.Button(self, -1, "sort UBs name")

        btnReloadMaterials = wx.Button(self, -1, "Reload Materials")

        # warning button id =52 is common with an other button
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        b3 = wx.Button(self, 52, "Replot Simul.")
        b3.SetFont(font3)

        # event handling
        self.emaxC.Bind(wx.EVT_SPINCTRL, self.mainframe.OnCheckEmaxValue)
        self.eminC.Bind(wx.EVT_SPINCTRL, self.mainframe.OnCheckEminValue)
        self.comboElem.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeElement)
        self.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeBMatrix, id=2424)
        self.btn_mergeUB.Bind(wx.EVT_BUTTON, self.mainframe.onSetOrientMatrix_with_BMatrix)
        self.btncenteronhkl.Bind(wx.EVT_BUTTON, self.mainframe.onCenterOnhkl)
        self.Bind(wx.EVT_COMBOBOX, self.mainframe.OnChangeMatrix, id=2525)
        self.Bind(wx.EVT_BUTTON, self.mainframe.EnterMatrix, id=1010)
        btn_sortUBsname.Bind(wx.EVT_BUTTON, self.onSortUBsname)

        btnReloadMaterials.Bind(wx.EVT_BUTTON, self.OnLoadMaterials)

        # layout
        h1box = wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add(t1, 0, wx.EXPAND|wx.ALL, 5)
        h1box.Add(self.eminC, 0, wx.EXPAND|wx.ALL, 5)
        h1box.Add(self.emaxC, 0, wx.EXPAND|wx.ALL, 5)
        h1box.Add(t2, 0, wx.EXPAND|wx.ALL, 5)
        h1box.Add(self.comboElem, 0, wx.EXPAND|wx.ALL, 5)
        h1box.Add(t5, 0, wx.EXPAND|wx.ALL, 5)
        h1box.Add(self.comboExtinctions, 0, wx.EXPAND, 5)

        h4box = wx.BoxSizer(wx.HORIZONTAL)
        h4box.Add(t4, 0, wx.EXPAND|wx.ALL, 5)
        h4box.Add(self.btncenteronhkl, 0, wx.EXPAND|wx.ALL, 5)
        h4box.Add(self.tchc, 0, wx.EXPAND|wx.ALL, 1)
        h4box.Add(self.tckc, 0, wx.EXPAND|wx.ALL, 1)
        h4box.Add(self.tclc, 0, wx.EXPAND|wx.ALL, 1)
        h4box.Add(self.comboMatrix, 0, wx.EXPAND|wx.ALL, 5)
        h4box.Add(self.btn_mergeUB, 0, wx.EXPAND|wx.ALL, 5)
        h4box.Add(t3, 0, wx.EXPAND|wx.ALL, 5)
        h4box.Add(self.comboBmatrix, 0, wx.EXPAND|wx.ALL, 5)

        h6box = wx.BoxSizer(wx.HORIZONTAL)
        h6box.Add(b1, 1, wx.EXPAND|wx.ALL, 5)
        h6box.Add(b2, 1, wx.EXPAND|wx.ALL, 5)
        h6box.Add(btn_sortUBsname, 0, wx.EXPAND|wx.ALL, 5)
        h6box.Add(btnReloadMaterials, 0, wx.EXPAND|wx.ALL, 5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(5)
        vbox.Add(h1box, 0, wx.EXPAND|wx.ALL, 0)
        vbox.Add(h4box, 0, wx.EXPAND|wx.ALL, 0)
        vbox.Add(h6box, 0, wx.EXPAND|wx.ALL, 0)
        vbox.Add(b3, 0, wx.EXPAND, 0)

        self.SetSizer(vbox)
        # tootips
        tp1 = "Energy minimum and maximum for simulated Laue Pattern spots"

        tp2 = "Element or material label (key of in Material dictionary)"

        #        tp3 = 'Matrix B in formula relating q vector = kf-ki and reciprocal vector nodes G*.\n'
        #        tp3 += 'q = U B B0 G* where U is the orientation matrix where B0 is the initial unit cell basis vectors orientation given by dictionary of elements.\n'
        #        tp3 += 'Each column of B0 is a reciprocal unit cell basis vector (among a*, b* and c*) expressed in Lauetools frame.\n'
        #        tp3 += 'B can then correspond to an initial state of rotation (to describe twin operation) or strain of the initial unit cell.'
        #
        tp3 = "The columns of the U*B*B0 matrix are the components of astar, bstar and cstar vectors (forming the Rstar frame) in the Rlab frame. \n"
        tp3 += "U is a pure rotation matrix \n"
        tp3 += "B is a triangular up matrix (strain + rotation), usually close to Identity (within 1e-3) \n"
        tp3 += 'B0 gives the initial shape of the reciprocal unit cell, as calculated from the lattice parameters defined by the "Material or Structure" parameter. \n'
        tp3 += "The columns of B0 are the components of astar bstar cstar on Rstar0.\n"
        tp3 += "Rstar0 is the cartesian frame built by orthonormalizing Rstar with Schmidt process. \n"
        tp3 += "Matrix T allows to apply a transform to a U*B0 matrix (preferably without B) via the formula U*T*B0 : \n"
        tp3 += "For example :\n"
        tp3 += "T = U2 (pure rotation) allows to switch to a twin orientation. \n"
        tp3 += "T = Eps (pure strain, symmetric) or T=B (triangular up) allows to change the shape of the unit cell. \n"

        #        tp3 = 'Matrix B in formula relating q vector = kf-ki and reciprocal vector nodes G*.\n'
        #        tp3 += 'q = U B B0 G* where U is the orientation matrix where B0 is the initial unit cell basis vectors orientation given by dictionary of elements.\n'
        #        tp3 += 'Each column of B0 is a reciprocal unit cell basis vector (among a*, b* and c*) expressed in Lauetools frame.\n'
        #        tp3 += 'B can then correspond to an initial state of rotation (to describe twin operation) or strain of the initial unit cell.'

        tp4 = "Orientation (and strained) Matrix UB. see explanations for B0 matrix"
        tp5 = "Set Extinctions rules due to special atoms positions in the unit cell"

        tpb1 = "Enter the 9 numerical elements for orientation matrix UB (not UB*B0 !)"
        tpb2 = "Store current orientation matrix UB in LaueToolsGUI"
        tpb3 = "Replot simulated Laue spots"

        # OR
        tpsetub = "inject current T transformation matrix into current UB matrix\n"
        tpsetub += "and reset T matrix to Identity.\n"
        tpsetub += "UB_new = UB_old * T \n"
        tpsetub += "typical use : UB_old = pure rotation U, T1 = twin transform, T2 = shear strain\n"

        # as UB*B. From q=UB B B0 G* to q= UBnew B0 G*'

        t1.SetToolTipString(tp1)
        t2.SetToolTipString(tp2)
        self.comboElem.SetToolTipString(tp2)
        t3.SetToolTipString(tp3)
        self.comboBmatrix.SetToolTipString(tp3)

        t4.SetToolTipString(tp4)
        self.comboMatrix.SetToolTipString(tp4)

        t5.SetToolTipString(tp5)
        self.comboExtinctions.SetToolTipString(tp5)

        b1.SetToolTipString(tpb1)
        b2.SetToolTipString(tpb2)
        b3.SetToolTipString(tpb3)

        self.btn_mergeUB.SetToolTipString(tpsetub)

        tipsportUBs = 'Sort Orientation Matrix name by alphabetical order'
        btn_sortUBsname.SetToolTipString(tipsportUBs)

        tipreloadMat = 'Reload Materials from dict_Materials file'
        btnReloadMaterials.SetToolTipString(tipreloadMat)

        self.btncenteronhkl.SetToolTipString('Orient crystal such as to have hkl at the center of detector frame')

    def onSortUBsname(self, _):
        listrot = list(DictLT.dict_Rot.keys())
        listrot = sorted(listrot, key=str.lower)
        self.comboMatrix.Clear()
        self.comboMatrix.AppendItems(listrot)

    def OnLoadMaterials(self, _):
        # self.mainframe.GetParent().OnLoadMaterials(1)
        # loadedmaterials = self.mainframe.GetParent().dict_Materials

        wcd = "All files(*)|*|dict_Materials files(*.dat)|*.mat"
        _dir = os.getcwd()
        open_dlg = wx.FileDialog(self, message="Choose a file", defaultDir=_dir, defaultFile="",
                                                                                    wildcard=wcd,
                                                                                    style=wx.OPEN)
        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()

            try:
                loadedmaterials = DictLT.readDict(path)

                DictLT.dict_Materials = loadedmaterials

            except IOError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except UnicodeDecodeError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except ValueError as error:
                dlg = wx.MessageDialog(self, "Error opening file: Something went wrong when parsing materials line\n" + str(error))
                dlg.ShowModal()

        open_dlg.Destroy()

        self.mainframe.dict_Materials = loadedmaterials
        self.comboElem.Clear()
        elements_keys = sorted(loadedmaterials.keys())
        self.comboElem.AppendItems(elements_keys)

        if self.mainframe.GetParent():
            self.mainframe.GetParent().dict_Materials = loadedmaterials


class CCDParamPanel(wx.Panel):
    """
    class panel for CCD detector parameters
    """

    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        # print("self.mainframe in CCDParamPanel", self.mainframe)

        self.init_pixelsize = self.mainframe.pixelsize
        self.init_detectordiameter = self.mainframe.detectordiameter

        #         wx.Button(self, 101, 'Set CCD Param.', (5, 5))
        #         self.Bind(wx.EVT_BUTTON, self.mainframe.OnInputParam, id=101)

        txtpixelsize = wx.StaticText(self, -1, "Pixelsize (mm) ")
        txtdetdiam = wx.StaticText(self, -1, "Diameter (mm) ")

        self.pixelsize_txtctrl = wx.TextCtrl(self, -1, str(self.mainframe.pixelsize), size=(150,-1))
        self.detectordiameter_txtctrl = wx.TextCtrl(self, -1, str(self.mainframe.detectordiameter), size=(150,-1))

        btnaccept = wx.Button(self, -1, "Accept Pixel size and diameter values")#, size=(-1, 80))
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        btnaccept.SetFont(font3)

        btnaccept.Bind(wx.EVT_BUTTON, self.onAccept)

        # layout
        h1box=wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add(txtpixelsize, 0, wx.EXPAND|wx.ALL, 10)
        h1box.Add(self.pixelsize_txtctrl, 0, wx.EXPAND|wx.ALL, 10)

        h2box=wx.BoxSizer(wx.HORIZONTAL)
        h2box.Add(txtdetdiam, 0, wx.EXPAND|wx.ALL, 10)
        h2box.Add(self.detectordiameter_txtctrl, 0, wx.EXPAND|wx.ALL, 10)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(5)
        vbox.Add(h1box, 0, wx.EXPAND, 5)
        vbox.Add(h2box, 0, wx.EXPAND, 5)
        vbox.Add(btnaccept, 0, wx.EXPAND, 5)

        self.SetSizer(vbox)

    def onAccept(self, evt):
        print("accept")
        ps = float(self.pixelsize_txtctrl.GetValue())
        detdiam = float(self.detectordiameter_txtctrl.GetValue())

        self.mainframe.pixelsize = ps
        self.mainframe.detectordiameter = detdiam

        print("new self.mainframe.pixelsize", self.mainframe.pixelsize)

        self.mainframe._replot(evt)


class DetectorParametersDisplayPanel(wx.Panel):
    """
    class panel to display and modify CCD parameters
    """
    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.granparent = parent.GetParent()

        self.CCDParam = self.granparent.CCDParam

        sizetxtctrl = wx.Size(70, -1)

        # current values
        self.act_distance = wx.TextCtrl(self, -1, str(self.CCDParam[0]), size=sizetxtctrl,
                                                style=wx.TE_PROCESS_ENTER)
        self.act_Xcen = wx.TextCtrl(self, -1, str(self.CCDParam[1]), size=sizetxtctrl,
                                                style=wx.TE_PROCESS_ENTER)
        self.act_Ycen = wx.TextCtrl(self, -1, str(self.CCDParam[2]), size=sizetxtctrl,
                                                style=wx.TE_PROCESS_ENTER)
        self.act_Ang1 = wx.TextCtrl(self, -1, str(self.CCDParam[3]), size=sizetxtctrl,
                                                style=wx.TE_PROCESS_ENTER)
        self.act_Ang2 = wx.TextCtrl(self, -1, str(self.CCDParam[4]), size=sizetxtctrl,
                                                style=wx.TE_PROCESS_ENTER)

        self.act_distance.Bind(wx.EVT_TEXT_ENTER, self.granparent.OnSetCCDParams)
        self.act_Xcen.Bind(wx.EVT_TEXT_ENTER, self.granparent.OnSetCCDParams)
        self.act_Ycen.Bind(wx.EVT_TEXT_ENTER, self.granparent.OnSetCCDParams)
        self.act_Ang1.Bind(wx.EVT_TEXT_ENTER, self.granparent.OnSetCCDParams)
        self.act_Ang2.Bind(wx.EVT_TEXT_ENTER, self.granparent.OnSetCCDParams)

        currenttxt = wx.StaticText(self, -1, "Current&&Set Value")
        resultstxt = wx.StaticText(self, -1, "Refined Value")

        # values resulting from model refinement
        self.act_distance_r = wx.TextCtrl(self, -1, "", style=wx.TE_READONLY, size=sizetxtctrl)
        self.act_Xcen_r = wx.TextCtrl(self, -1, "", style=wx.TE_READONLY, size=sizetxtctrl)
        self.act_Ycen_r = wx.TextCtrl(self, -1, "", style=wx.TE_READONLY, size=sizetxtctrl)
        self.act_Ang1_r = wx.TextCtrl(self, -1, "", style=wx.TE_READONLY, size=sizetxtctrl)
        self.act_Ang2_r = wx.TextCtrl(self, -1, "", style=wx.TE_READONLY, size=sizetxtctrl)

        if WXPYTHON4:
            grid = wx.GridSizer(6, 2, 2)
        else:
            grid = wx.GridSizer(3, 6)

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        headertext = wx.StaticText(self, -1, "Detector Parameters")
        headertext.SetFont(font3)

        grid.Add(wx.StaticText(self, -1, ""))
        for txt in DictLT.CCD_CALIBRATION_PARAMETERS[:5]:
            grid.Add(wx.StaticText(self, -1, txt), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)

        grid.Add(currenttxt)
        for txtctrl in [self.act_distance, self.act_Xcen, self.act_Ycen, self.act_Ang1, self.act_Ang2]:
            grid.Add(txtctrl, 0,wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
            txtctrl.SetToolTipString("Current and Set new value (press enter)")
            txtctrl.SetSize(sizetxtctrl)

        grid.Add(resultstxt)
        for txtctrl in [self.act_distance_r, self.act_Xcen_r, self.act_Ycen_r, self.act_Ang1_r, self.act_Ang2_r]:
            grid.Add(txtctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_CENTER_HORIZONTAL)
            txtctrl.SetToolTipString("Fit result value")

        vbox=wx.BoxSizer(wx.VERTICAL)
        vbox.Add(headertext,0, wx.ALIGN_CENTER_HORIZONTAL)
        vbox.Add(grid,0)

        self.SetSizer(vbox)

        # tooltips
        resultstxt.SetToolTipString("CCD detector plane parameters resulting from the best "
        "refined model")
        currenttxt.SetToolTipString('Current CCD detector plane parameters. '
        'New parameters value can be entered in the corresponding field and '
        'accepted by pressing the "Accept" button')


class MoveCCDandXtal(wx.Panel):
    """
    class panel to move CCD camera and crystal
    """
    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        # print("self.mainframe in CCDParamPanel", self.mainframe)

        t1 = wx.StaticText(self, -1, "Sample-Detector Distance    Step")
        b10 = wx.Button(self, 10, "-")#, (20, -1))
        b11 = wx.Button(self, 11, "+")#, (20, -1))
        st1 = wx.StaticText(self, -1, "mm")
        self.stepdistance = wx.TextCtrl(self, -1, "0.5")#, (30, -1))
        self.cb_dd = wx.CheckBox(self, -1, "fit")
        self.cb_dd.SetValue(True)

        t2 = wx.StaticText(self, -1, "X center")
        b20 = wx.Button(self, 20, "-")#, (20, -1))
        b21 = wx.Button(self, 21, "+")#, (20, -1))
        st2 = wx.StaticText(self, -1, "pixel")
        self.stepXcen = wx.TextCtrl(self, -1, "20.")#, (30, -1))
        self.cb_Xcen = wx.CheckBox(self, -1, "fit")
        self.cb_Xcen.SetValue(True)

        t3 = wx.StaticText(self, -1, "Y center")
        b30 = wx.Button(self, 30, "-")#, (20, -1))
        b31 = wx.Button(self, 31, "+")#, (20, -1))
        st3 = wx.StaticText(self, -1, "pixel")
        self.stepYcen = wx.TextCtrl(self, -1, "20.")#,(30, -1))
        self.cb_Ycen = wx.CheckBox(self, -1, "fit")
        self.cb_Ycen.SetValue(True)

        t4 = wx.StaticText(self, -1, "Angle xbet")
        b40 = wx.Button(self, 40, "-")#, (20, -1))
        b41 = wx.Button(self, 41, "+")#, (20, -1))
        st4 = wx.StaticText(self, -1, "deg ")
        self.stepang1 = wx.TextCtrl(self, -1, "1.")#, (30, -1))
        self.cb_angle1 = wx.CheckBox(self, -1, "fit")
        self.cb_angle1.SetValue(True)

        t5 = wx.StaticText(self, -1, "Angle xgam")
        b50 = wx.Button(self, 50, "-")#, (20, -1))
        b51 = wx.Button(self, 51, "+")#, (20, -1))
        st5 = wx.StaticText(self, -1, "deg ")
        self.stepang2 = wx.TextCtrl(self, -1, "1.")#, (30, -1))
        self.cb_angle2 = wx.CheckBox(self, -1, "fit")
        self.cb_angle2.SetValue(True)

        # Angles buttons - crystal orientation
        a1 = wx.StaticText(self, -1, "Angle 1 (deg)     Step")
        b1000 = wx.Button(self, 1000, "-")#, (20, -1))
        b1100 = wx.Button(self, 1100, "+")#, (20, -1))
        # wx.StaticText(self, -1, 'step(deg)',(960, 30))
        self.angle1 = wx.TextCtrl(self, -1, "1.", (20, -1))
        self.cb_theta1 = wx.CheckBox(self, -1, "fit", )
        self.cb_theta1.SetValue(True)

        a2 = wx.StaticText(self, -1, "Angle2  (deg)")
        b2000 = wx.Button(self, 2000, "-")#, (20, -1))
        b2100 = wx.Button(self, 2100, "+")#, (20, -1))
        # wx.StaticText(self, -1, 'step(deg)',(960, pos2+20))
        self.angle2 = wx.TextCtrl(self, -1, "1.", (20, -1))
        self.cb_theta2 = wx.CheckBox(self, -1, "fit")
        self.cb_theta2.SetValue(True)

        a3 = wx.StaticText(self, -1, "Angle 3  (deg)")
        b3000 = wx.Button(self, 3000, "-")#, (20, -1))
        b3100 = wx.Button(self, 3100, "+")#, (20, -1))
        # wx.StaticText(self, -1, 'step(deg)',(960, pos3+20))
        self.angle3 = wx.TextCtrl(self, -1, "1.", (20, -1))
        self.cb_theta3 = wx.CheckBox(self, -1, "fit")
        self.cb_theta3.SetValue(True)

        self.EnableRotationLabel = "Select Axis && Rotate"
        self.rotatebtn = wx.Button(self, -1, self.EnableRotationLabel)
        self.rotatebtn.Bind(wx.EVT_BUTTON, self.OnActivateRotation)
        st9 = stepangletxt = wx.StaticText(self, -1, "step (deg)")
        self.stepanglerot = wx.TextCtrl(self, -1, "10.")

        self.listofparamfitctrl = [self.cb_dd, self.cb_Xcen, self.cb_Ycen, self.cb_angle1,
                                    self.cb_angle2,  # detector param
                                    self.cb_theta1,
                                    self.cb_theta2,
                                    self.cb_theta3]  # delta angle of orientation
        # layout --------------------------------------------
        hboxes = [0, 0, 0, 0, 0]
        bminus = [b10, b20, b30, b40, b50]
        bplus = [b11, b21, b31, b41, b51]
        steps = [st1, st2, st3, st4, st5]
        steptxtctrls = [self.stepdistance, self.stepXcen, self.stepYcen, self.stepang1, self.stepang2]
        chckboxes = self.listofparamfitctrl[:5]
        for k in range(5):
            hboxes[k] = wx.BoxSizer(wx.HORIZONTAL)
            hboxes[k].Add(bminus[k], 0, wx.EXPAND, 2)
            hboxes[k].Add(bplus[k], 0, wx.EXPAND, 2)
            hboxes[k].Add(steptxtctrls[k], 0, wx.EXPAND, 2)
            hboxes[k].Add(steps[k], 0, wx.EXPAND, 2)
            hboxes[k].Add(chckboxes[k], 0, wx.EXPAND, 2)

        h2boxes = [0, 0, 0, 0, 0]
        bminus = [b1000, b2000, b3000]
        bplus = [b1100, b2100, b3100]
        steptxtctrls = [self.angle1, self.angle2, self.angle3]
        chckboxes = self.listofparamfitctrl[5:]

        for k in range(3):
            h2boxes[k] = wx.BoxSizer(wx.HORIZONTAL)
            h2boxes[k].Add(bminus[k], 0, wx.EXPAND, 2)
            h2boxes[k].Add(bplus[k], 0, wx.EXPAND, 2)
            h2boxes[k].Add(steptxtctrls[k], 0, wx.EXPAND, 2)
            h2boxes[k].Add(chckboxes[k], 0, wx.EXPAND, 2)

        hrotbox=wx.BoxSizer(wx.HORIZONTAL)
        hrotbox.Add(self.rotatebtn, 0, wx.EXPAND, 10)
        hrotbox.Add(st9, 0, wx.EXPAND, 10)
        hrotbox.Add(self.stepanglerot, 0, wx.EXPAND, 10)



        vdetbox = wx.BoxSizer(wx.VERTICAL)
        vdetbox.Add(t1, 0, wx.EXPAND, 0)
        vdetbox.Add(hboxes[0], 0, wx.EXPAND, 0)
        vdetbox.Add(t2, 0, wx.EXPAND, 0)
        vdetbox.Add(hboxes[1], 0, wx.EXPAND, 0)
        vdetbox.Add(t3, 0, wx.EXPAND|wx.ALL, 0)
        vdetbox.Add(hboxes[2], 0, wx.EXPAND, 0)
        vdetbox2 = wx.BoxSizer(wx.VERTICAL)
        vdetbox2.Add(t4, 0, wx.EXPAND|wx.ALL, 0)
        vdetbox2.Add(hboxes[3], 0, wx.EXPAND, 0)
        vdetbox2.Add(t5, 0, wx.EXPAND|wx.ALL, 0)
        vdetbox2.Add(hboxes[4], 0, wx.EXPAND, 0)

        vangbox = wx.BoxSizer(wx.VERTICAL)
        vangbox.Add(a1, 0, wx.EXPAND, 0)
        vangbox.Add(h2boxes[0], 0, wx.EXPAND, 0)
        vangbox.Add(a2, 0, wx.EXPAND, 0)
        vangbox.Add(h2boxes[1], 0, wx.EXPAND, 0)
        vangbox.Add(a3, 0, wx.EXPAND, 0)
        vangbox.Add(h2boxes[2], 0, wx.EXPAND, 0)
        vangbox.Add(hrotbox, 0, wx.EXPAND, 5)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(vdetbox, 0, wx.EXPAND, 10)
        hbox.Add(vdetbox2, 0, wx.EXPAND, 10)
        hbox.Add(vangbox, 0, wx.EXPAND, 10)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(2)
        vbox.Add(hbox, 0)

        self.SetSizer(vbox)

        # tooltips
        a3.SetToolTipString("Angle 3: angle around z axis")
        a1.SetToolTipString("Angle 1: angle around y axis (horizontal and perp. to incoming beam)")
        a2.SetToolTipString("Angle 2: angle around x axis (// incoming beam")

        rottip = ("[For 2theta/chi coordinates]\n Rotate crystal such as rotating the Laue Pattern around a selected axis.\n")
        rottip += 'Click on a point in plot to select an invariant Laue spot by rotation. Then press "+" or "-" keys to rotation the pattern.\n'
        rottip += "Step angle can be adjusted (default 10 degrees).\n"
        rottip += "Press the Rotate button to disable the rotation and enable other functionnalities."
        self.rotatebtn.SetToolTipString(rottip)
        stepangletxt.SetToolTipString(rottip)
        self.stepanglerot.SetToolTipString(rottip)

    def OnActivateRotation(self, _):

        self.mainframe.RotationActivated = not self.mainframe.RotationActivated

        # clear previous rotation axis
        self.mainframe.SelectedRotationAxis = None

        if self.mainframe.RotationActivated:
            print("Activate Rotation around axis")
            self.rotatebtn.SetLabel("DISABLE\nRotation\naround\nselected Axis")
        else:
            print("Disable Rotation around axis")
            self.rotatebtn.SetLabel(self.EnableRotationLabel)


class StrainXtal(wx.Panel):
    """
    class panel to strain crystal
    """
    def __init__(self, parent):
        """
        """
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        # print("self.mainframe in CCDParamPanel", self.mainframe)

        self.key_material = self.mainframe.crystalparampanel.comboElem.GetValue()
        self.key_material_initparams_in_dict = copy.copy(DictLT.dict_Materials[self.key_material])
        self.lattice_parameters = copy.copy(DictLT.dict_Materials[self.key_material][1])

        if WXPYTHON4:
            grid = wx.FlexGridSizer(6, 10, 1)
            grid2 = wx.FlexGridSizer(6, 10, 1)
        else:
            grid = wx.FlexGridSizer(6, 6)  # correct order ..??
            grid2 = wx.FlexGridSizer(6, 6)

        self.lattice_parameters_key = ["a (Angst.)", "b (Angst.)", "c (Angst.)", "alpha (deg)", "beta (deg)", "gamma (deg)"]

        self.dict_keyparam = {}
        for k, key_param in enumerate(self.lattice_parameters_key):
            self.dict_keyparam[key_param] = k

        self.lattice_parameters_dict = {}
        for k, key_param in enumerate(self.lattice_parameters_key):
            self.lattice_parameters_dict[key_param] = self.lattice_parameters[k]

        headertxt = wx.StaticText(self, -1, "Crystal Unit Cell Lattice Parameters")
        font10 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        headertxt.SetFont(font10)

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "Current"))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "step"))
        grid.Add(wx.StaticText(self, -1, ""))
        
        grid2.Add(wx.StaticText(self, -1, ""))
        grid2.Add(wx.StaticText(self, -1, "Current"))
        grid2.Add(wx.StaticText(self, -1, ""))
        grid2.Add(wx.StaticText(self, -1, ""))
        grid2.Add(wx.StaticText(self, -1, "step"))
        grid2.Add(wx.StaticText(self, -1, ""))
        
        for k, key_param in enumerate(self.lattice_parameters_key):

            minusbtn = wx.Button(self, -1, "-", size=(40, 30))
            plusbtn = wx.Button(self, -1, "+", size=(40, 30))
            stepctrl = wx.TextCtrl(self, -1, "0.05", size=(60, 30))
            fitchckbox = wx.CheckBox(self, -1, "fit")
            fitchckbox.SetValue(True)
            fitchckbox.Disable()
            currentctrl = wx.TextCtrl(self, -1, str(self.lattice_parameters_dict[key_param]),
                size=(60, -1), style=wx.TE_PROCESS_ENTER)

            setattr(self, "minusbtn_%s" % key_param, minusbtn)
            setattr(self, "plusbtn_%s" % key_param, plusbtn)
            setattr(self, "stepctrl_%s" % key_param, stepctrl)
            setattr(self, "fitchckbox_%s" % key_param, fitchckbox)
            setattr(self, "currentctrl_%s" % key_param, currentctrl)

            getattr(self, "minusbtn_%s" % key_param, minusbtn).myname = ("minusbtn_%s" % key_param)
            getattr(self, "plusbtn_%s" % key_param, plusbtn).myname = ("minusbtn_%s" % key_param)
            getattr(self, "currentctrl_%s" % key_param, currentctrl).myname = ("currentctrl_%s" % key_param)

            if k in (0,1,2):
                grid.Add(wx.StaticText(self, -1, key_param), 0)
                grid.Add(currentctrl, 0)
                grid.Add(minusbtn, 0)
                grid.Add(plusbtn, 0)
                grid.Add(stepctrl, 0)
                grid.Add(fitchckbox, 0)
                
            else:
                grid2.Add(wx.StaticText(self, -1, key_param), 0)
                grid2.Add(currentctrl, 0)
                grid2.Add(minusbtn, 0)
                grid2.Add(plusbtn, 0)
                grid2.Add(stepctrl, 0)
                grid2.Add(fitchckbox, 0)

            getattr(self, "plusbtn_%s" % key_param).Bind(
                wx.EVT_BUTTON, lambda event: self.ModifyLatticeParamsStep(event, "+"))
            getattr(self, "minusbtn_%s" % key_param).Bind(
                wx.EVT_BUTTON, lambda event: self.ModifyLatticeParamsStep(event, "-"))
            getattr(self, "currentctrl_%s" % key_param).Bind(wx.EVT_TEXT_ENTER,
                                                            self.ModifyLatticeParams)

        twogridshbox = wx.BoxSizer(wx.HORIZONTAL)
        twogridshbox.Add(grid, 0, wx.EXPAND)
        twogridshbox.AddSpacer(15)
        twogridshbox.Add(grid2, 0, wx.EXPAND)
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(headertxt, 0, wx.ALIGN_CENTER_HORIZONTAL)
        vbox.Add(twogridshbox, 0, wx.EXPAND)

        self.SetSizer(vbox)

    def update_latticeparameters(self):

        self.key_material = self.mainframe.crystalparampanel.comboElem.GetValue()
        self.key_material_initparams_in_dict = copy.copy(DictLT.dict_Materials[self.key_material])
        self.lattice_parameters = copy.copy(DictLT.dict_Materials[self.key_material][1])

        for k, key_param in enumerate(self.lattice_parameters_key):
            self.lattice_parameters_dict[key_param] = self.lattice_parameters[k]
            getattr(self, "currentctrl_%s" % key_param).SetValue(str(self.lattice_parameters[k]))

    def ModifyLatticeParamsStep(self, event, sign_of_step):
        """
        modify lattice parameters according to event.name and sign
        """

        #         print "sign_of_step", sign_of_step
        name = event.GetEventObject().myname
        #         print "name", name

        key_param = name.split("_")[-1]

        self.lattice_parameters_dict[key_param] = float(getattr(self,
                                                        "currentctrl_%s" % key_param).GetValue())

        if sign_of_step == "+":
            stepsign = 1.0
        elif sign_of_step == "-":
            stepsign = -1.0

        #         print "modify lattice parameter: %s and initial value: %.2f" % (key_param, self.lattice_parameters_dict[key_param])
        self.lattice_parameters_dict[key_param] += stepsign * float(getattr(self,
                                                            "stepctrl_%s" % key_param).GetValue())

        # now building or updating an element in dict_Materials
        if "strained" not in self.key_material:
            new_key_material = "strained_%s" % self.key_material
        else:
            new_key_material = self.key_material

        DictLT.dict_Materials[new_key_material] = self.key_material_initparams_in_dict
        # update label
        DictLT.dict_Materials[new_key_material][0] = new_key_material

        if self.mainframe.crystalparampanel.comboElem.FindString(new_key_material) == -1:
            print("adding new material in comboelement list")
            self.mainframe.crystalparampanel.comboElem.Append(new_key_material)

        new_lattice_params = []
        for _key_param in self.lattice_parameters_key:
            new_lattice_params.append(self.lattice_parameters_dict[_key_param])

        #         print "from self.lattice_parameters_dict", self.lattice_parameters_dict

        DictLT.dict_Materials[new_key_material][1] = new_lattice_params

        print("new lattice parameters", new_lattice_params)
        print("for material: %s" % new_key_material)

        getattr(self,
                "currentctrl_%s" % key_param).SetValue(str(self.lattice_parameters_dict[key_param]))

        self.mainframe.crystalparampanel.comboElem.SetValue(new_key_material)
        self.mainframe._replot(1)

    def ModifyLatticeParams(self, event):

        name = event.GetEventObject().myname

        key_param = name.split("_")[-1]

        formulaexpr = getattr(self,"currentctrl_%s" % key_param).GetValue()
        self.lattice_parameters_dict[key_param] = float(eval(formulaexpr))

        if "strained" not in self.key_material:
            new_key_material = "strained_%s" % self.key_material
        else:
            new_key_material = self.key_material

        DictLT.dict_Materials[new_key_material] = self.key_material_initparams_in_dict
        DictLT.dict_Materials[new_key_material][0] = new_key_material

        if self.mainframe.crystalparampanel.comboElem.FindString(new_key_material) == -1:
            print("adding new material in comboelement list")
            self.mainframe.crystalparampanel.comboElem.Append(new_key_material)

        new_lattice_params = []
        for _key_param in self.lattice_parameters_key:
            new_lattice_params.append(self.lattice_parameters_dict[_key_param])

        DictLT.dict_Materials[new_key_material][1] = new_lattice_params

        print("new lattice parameters", new_lattice_params)
        print("for material: %s" % new_key_material)

        # getattr(self, "currentctrl_%s" % key_param).SetValue(str(self.lattice_parameters_dict[key_param])) 
        getattr(self, "currentctrl_%s" % key_param).SetValue(formulaexpr)

        self.mainframe.crystalparampanel.comboElem.SetValue(new_key_material)
        self.mainframe._replot(1)

    def OnActivateRotation(self, _):

        self.mainframe.RotationActivated = not self.mainframe.RotationActivated

        if self.mainframe.RotationActivated:
            print("Activate Rotation around axis")
        else:
            print("Disable Rotation around axis")
            self.mainframe.SelectedRotationAxis = None


class TextFrame(wx.Frame):
    def __init__(self, parent, _id, strexpression, index=0):
        wx.Frame.__init__(self, parent, _id, "Matrix Store and Save", size=(500, 250))

        self.parent = parent
        self.index = index

        panel = wx.Panel(self, -1)
        matrixLabel = wx.StaticText(panel, -1, "Matrix Elements:")
        matrixText = wx.TextCtrl(panel, -1, strexpression, size=(490, 100),
                                                        style=wx.TE_MULTILINE | wx.TE_READONLY)
        #         matrixText.SetInsertionPoint(0)

        storeLabel = wx.StaticText(panel, -1, "Stored Matrix name: ")
        self.storeText = wx.TextCtrl(panel, -1, "storedMatrix_%d" % self.index, size=(175, -1))

        saveLabel = wx.StaticText(panel, -1, "Save Matrix filename: ")
        self.saveText = wx.TextCtrl(panel, -1, "SavedMatrix_%d" % self.index, size=(175, -1))

        btnstore = wx.Button(panel, -1, "Store (in GUI)")
        btnsave = wx.Button(panel, -1, "Save (on Hard Disk)")
        btnquit = wx.Button(panel, -1, "Quit")

        btnstore.Bind(wx.EVT_BUTTON, self.onStore)
        btnsave.Bind(wx.EVT_BUTTON, self.onSave)
        btnquit.Bind(wx.EVT_BUTTON, self.onQuit)

        sizer6 = wx.FlexGridSizer(cols=3, hgap=6, vgap=6)
        sizer6.AddMany([storeLabel, self.storeText, btnstore, saveLabel, self.saveText, btnsave])

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(matrixLabel)
        vbox.Add(matrixText)
        vbox.Add(sizer6)
        vbox.Add(btnquit)
        panel.SetSizer(vbox)

    def onStore(self, _):

        matrix_name = str(self.storeText.GetValue())

        UBmatrix = np.array(self.parent.crystalparampanel.UBmatrix)

        DictLT.dict_Rot[matrix_name] = UBmatrix
        self.parent.crystalparampanel.comboMatrix.Append(matrix_name)

    def onSave(self, _):
        matrixfilename = str(self.storeText.GetValue())

        UBmatrix = np.array(self.parent.crystalparampanel.UBmatrix)

        np.savetxt(matrixfilename, UBmatrix, delimiter=",")

        _file = open(matrixfilename + "_list", "w")
        text = ("[[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f]]"
            % tuple(np.ravel(UBmatrix).tolist()))
        _file.write(text)
        _file.close()

    def onQuit(self, _):
        self.Close()


# --- -----------------  Calibration Board
class MainCalibrationFrame(wx.Frame):
    """
    Class to display calibration tools on data
    """
    def __init__(self, parent, _id, title, _initialParameter,
                file_peaks="Cu_near_28May08_0259.peaks",
                pixelsize=165.0 / 2048,
                datatype="2thetachi",
                dim=(2048, 2048),  # for MARCCD 165,
                kf_direction="Z>0",
                fliprot="no",
                data_added=None):
        """ init

        lot of parameters in _initialParameter dict
        :param file_peaks: fullpath to peaks list file"""

        wx.Frame.__init__(self, parent, _id, title, size=(1200, 830))

        self.parent = parent # LaueToolsGUI.MainFrame

        self.initialParameter = _initialParameter

        # 5 parameters defining Detector Plane and frame
        self.CCDParam = self.initialParameter["CCDParam"]
        # to interact with LaueToolsGUI
        self.defaultParam = self.CCDParam
        self.detectordiameter = self.initialParameter["detectordiameter"]
        self.CCDLabel = self.initialParameter["CCDLabel"]
        self.kf_direction = kf_direction
        self.kf_direction_from_file = kf_direction
        self.filename = file_peaks # could be .dat or .cor file
        # to interact with LaueToolsGUI
        self.DataPlot_filename = self.filename
        self.dirnamepklist = self.initialParameter["dirname"]
        self.writefolder = None
        self.resetwf = False

        self.pixelsize = pixelsize
        self.framedim = dim
        self.fliprot = fliprot

        self.data_theo = data_added
        self.tog = 0
        self.datatype = datatype

        self.dict_Materials = self.initialParameter["dict_Materials"]

        self.points = []  # to store points
        self.selectionPoints = []
        self.twopoints = []
        self.threepoints = []
        self.sixpoints = []
        self.nbclick = 1
        self.nbsuccess = 0
        self.nbclick_dist = 1
        self.nbclick_zone = 1

        self.recognition_possible = True
        self.toshow = []
        self.DEFAULT_UBMATRIX_CALIB = 'GesCMOS_sept2024'
        self.deltamatrix = np.eye(3)
        self.manualmatrixinput = None

        self.inputmatrix = None
        

        # for plot spots annotation
        self.drawnAnnotations_exp = {}
        self.links_exp = []

        self.drawnAnnotations_theo = {}
        self.links_theo = []

        self.RotationActivated = False
        self.SelectedRotationAxis = None

        self.savedindex = 0
        self.storedmatrixindex = 0
        self.savedmatrixindex = 0

        # for fitting procedure  initial model (pairs of simul and exp; spots)----
        self.linkedspots = None
        self.linkExpMiller = []
        self.linkResidues = None

        # highlight spots
        self.plotlinks = None   # exp spot linked to theo spot

        # savings of refined model
        self.linkedspotsAfterFit = None
        self.linkExpMillerAfterFit = None
        self.linkIntensityAfterFit = None
        self.residues_fitAfterFit = None
        self.residues_fit = None

        self.SpotsData = None

        # save previous result to undo goto fit results-----------------
        self.previous_CCDParam = copy.copy(self.CCDParam)
        self.previous_UBmatrix = np.eye(3)

        self.twicetheta = None
        self.chi = None
        self.Data_I, self.data, self.Data_index_expspot = None, None, None
        self.data_x, self.data_y = None, None
        self.filenameCalib = None
        self.linkIntensity = None
        self.linkEnergy = None
        self.UBmatrix = None
        self.Umat2 = None
        self.Bmat_tri = None
        self.HKLxyz_names, self.HKLxyz = None, None
        self.totalintensity = None
        self.p2S, self.p3S = None, None

        self.init_plot = True
        self.Extinctions = None

        self.emin, self.emax = 5, 25
        self.key_material = None
        self.B0matrix = None
        self.Bmatrix = None
        self.Miller_ind = None

        self.sim_gnomonx, self.sim_gnomony = None, None
        self.data_gnomonx, self.data_gnomony = None, None
        self.successfull = False
        self.EXPpoints = None

        self.mat_solution = None
        self.TwicethetaChi_solution = None
        self.data_fromGnomon = None
        self.RecBox = None
        self.centerx, self.centery = None, None
        self.press = None
        self.savedfinaltxt = ''

        self._dataANNOTE_exp, self._dataANNOTE_theo = None, None

        self.setwidgets()

        # read peaks data --------------------------
        self.ReadExperimentData()
        # plot simulated and experimental data
        self._replot(wx.EVT_IDLE)
        self.display_current()

    def setwidgets(self):
        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        self.press = None

        # BUTTONS, PLOT and CONTROL
        #        self.plotPanel = wxmpl.PlotPanel(self, -1, size=Size, autoscaleUnzoom=False)
        self.panel = wx.Panel(self)

        self.nb = wx.Notebook(self.panel, -1, style=0)

        self.plotrangepanel = PlotRangePanel(self.nb)
        self.crystalparampanel = CrystalParamPanel(self.nb)
        self.ccdparampanel = CCDParamPanel(self.nb)
        self.moveccdandxtal = MoveCCDandXtal(self.nb)
        self.strainxtal = StrainXtal(self.nb)

        self.nb.AddPage(self.plotrangepanel, "Plot Range")
        self.nb.AddPage(self.crystalparampanel, "Crystal Param")
        self.nb.AddPage(self.ccdparampanel, "CCD Param")
        self.nb.AddPage(self.moveccdandxtal, "Move CCD and Xtal")
        self.nb.AddPage(self.strainxtal, "Strain Xtal")

        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.figsizex, self.figsizey = 4, 3
        self.fig = Figure((self.figsizex, self.figsizey), dpi=self.dpi)
        self.fig.set_size_inches(self.figsizex, self.figsizey, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        self.init_plot = True

        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas)

        self.tooltip = wx.ToolTip(tip="Welcome on LaueTools Simulation and Calibration Board")
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)

        # Add the custom tools that we created
        self.toolbar.AddTool(123456, 'oneTool',  wx.Bitmap(os.path.join(DictLT.LAUETOOLSFOLDER,"icons",'flipudsmall.png')),  
                                                     wx.Bitmap(os.path.join(DictLT.LAUETOOLSFOLDER,"icons",'flipudsmall.png')),  
                                             kind = wx.ITEM_CHECK, shortHelp ="Flip up/down")
        self.toolbar.Bind(wx.EVT_TOOL, self.OnChangeOrigin, id=123456)

        self.toolbar.Realize() 

        self.sb = self.CreateStatusBar()

        self.cidpress = self.fig.canvas.mpl_connect("button_press_event", self.onClick)
        self.fig.canvas.mpl_connect("key_press_event", self.onKeyPressed)
        self.cidrelease = self.fig.canvas.mpl_connect("button_release_event", self.onRelease)
        self.cidmotion = self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion)

        self.Bind(wx.EVT_BUTTON, self.OnStoreMatrix, id=1011)  # in crystalparampanel

        self.peakpropstxt = wx.StaticText(self.panel, -1, "Draw peak props.   ")
        self.btn_label_theospot = wx.ToggleButton(self.panel, 104, "Exp. spot")
        self.btn_label_expspot = wx.ToggleButton(self.panel, 106, "Simul. spot")

        self.resetAnnotationBtn = wx.Button(self.panel, -1, "Reset")
        self.resetAnnotationBtn.Bind(wx.EVT_BUTTON, self.OnResetAnnotations)

        self.startfit = wx.Button(self.panel, 505, "Start FIT")#, size=(150, 80))
        self.startfit.SetFont(font3)
        self.Bind(wx.EVT_BUTTON, self.StartFit, id=505)

        self.cb_gotoresults = wx.CheckBox(self.panel, -1, "GOTO fit results")
        self.use_weights = wx.CheckBox(self.panel, -1, "use weights")
        self.use_weights.SetValue(False)
        self.cb_gotoresults.SetValue(True)

        self.undogotobtn = wx.Button(self.panel, -1, "Undo GOTO last fit")#, size=(-1, 80))
        self.undogotobtn.Bind(wx.EVT_BUTTON, self.OnUndoGoto)

        self.residualstrainbtn = wx.Button(self.panel, -1, "Assess Resid. Strain")#, size=(-1, 80))
        self.residualstrainbtn.Bind(wx.EVT_BUTTON, self.OnAssessResidualStrain)

        self.resstrainstatsbtn = wx.Button(self.panel, -1, "Resid. Strain Statistics")#, size=(-1, 80))
        self.resstrainstatsbtn.Bind(wx.EVT_BUTTON, self.OnResidualStrainStatistics)

        # replot simul button (one button in two panels)
        self.Bind(wx.EVT_BUTTON, self._replot, id=52)

        self.Bind(wx.EVT_BUTTON, self.OnDecreaseDistance, id=10)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseDistance, id=11)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseXcen, id=20)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseXcen, id=21)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseYcen, id=30)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseYcen, id=31)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseang1, id=40)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseang1, id=41)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseang2, id=50)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseang2, id=51)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseAngle1, id=1000)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseAngle1, id=1100)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseAngle2, id=2000)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseAngle2, id=2100)
        self.Bind(wx.EVT_BUTTON, self.OnDecreaseAngle3, id=3000)
        self.Bind(wx.EVT_BUTTON, self.OnIncreaseAngle3, id=3100)

        self.btnswitchspace = wx.Button(self.panel, 102, "Switch Space")#, size=(150, 80))
        self.btnswitchspace.SetFont(font3)
        self.Bind(wx.EVT_BUTTON, self.OnSwitchPlot, id=102)

        self.btnautolinks = wx.Button(self.panel, -1, "Auto. Links")#, size=(150, 80))
        self.btnautolinks.SetFont(font3)
        self.btnautolinks.Bind(wx.EVT_BUTTON, self.OnLinkSpotsAutomatic)

        self.btnmanuallinks = wx.Button(self.panel, -1, "Manual Links")#, size=(-1, 80))
        self.btnmanuallinks.Bind(wx.EVT_BUTTON, self.OnLinkSpots)
        self.btnmanuallinks.Enable(False)

        self.btnshowlinks = wx.Button(self.panel, -1, "Filter Links")#, size=(-1, 80))
        self.btnshowlinks.Bind(wx.EVT_BUTTON, self.OnShowAndFilter)

        self.txtangletolerance = wx.StaticText(self.panel, -1, "Tolerance Angle\n      (deg)")
        self.AngleMatchingTolerance = wx.TextCtrl(self.panel, -1, "0.5")

        self.btnsaveresults = wx.Button(self.panel, 1013, "Save .fit file")#, size=(-1, 80))  # produces file with results
        self.Bind(wx.EVT_BUTTON, self.OnWriteResults, id=1013)

        self.btnsavecalib = wx.Button(self.panel, 1012, "Save calibration.det file")#, size=(-1, 80))# calibration parameters + orientation UBmatrix
        self.Bind(wx.EVT_BUTTON, self.OnSaveCalib, id=1012)
        self.btnsavecalib.SetFont(font3)

        self.defaultColor = self.GetBackgroundColour()
        # print "self.defaultColor",self.defaultColor
        self.p2S, self.p3S = 0, 0

        self.Bind(wx.EVT_TOGGLEBUTTON, self.ToggleLabelExp, id=104)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.ToggleLabelSimul, id=106)

        self.parametersdisplaypanel = DetectorParametersDisplayPanel(self.panel)
        self.Bind(wx.EVT_BUTTON, self.OnSetCCDParams, id=159)

        self.txtresidues = wx.StaticText(self.panel, -1, "Mean Residues (pix)   ")
        self.txtnbspots = wx.StaticText(self.panel, -1, "Nbspots")
        self.act_residues = wx.TextCtrl(self.panel, -1, "", style=wx.TE_READONLY)
        self.nbspots_in_fit = wx.TextCtrl(self.panel, -1, "", style=wx.TE_READONLY)

        self.incrementfile = wx.CheckBox(self.panel, -1, "increment saved filenameindex")

        self.layout()

        # tooltips -----------------------------------------------------------------
        self.plotrangepanel.SetToolTipString("Set plot and spots display parameters")
        self.moveccdandxtal.SetToolTipString("Change manually and fit (by checking corresponding "
                                        "boxes) the 5 CCD parameters and rotate the crystal "
                                        "around 3 elementary angles")
        self.crystalparampanel.SetToolTipString("set crystal parameters for laue spots simulation")
        self.ccdparampanel.SetToolTipString("Set new CCD camera parameters")

        self.btnmanuallinks.SetToolTipString("Build manually a list of associations or links "
                                                "between close simulated and experimental spots")
        self.btnautolinks.SetToolTipString("Build automatically a list of associations or links "
                                        "between close simulated and experimental spots "
                                        "within 'Angle Tolerance'")

        tp1 = "Maximum separation angle (degree) to associate (link) automatically pair of spots (exp. and theo)"
        self.txtangletolerance.SetToolTipString(tp1)
        self.AngleMatchingTolerance.SetToolTipString(tp1)

        self.btnshowlinks.SetToolTipString('Browse and filter links resulting from "Auto. Links"')

        self.btnswitchspace.SetToolTipString("switch between different spots coordinates: "
                    "2theta,chi ; Gnomonic projection coordinates ; X,Y pixel position on Camera")
        self.btn_label_theospot.SetToolTipString("Display on plot data related to selected (by clicking) experimental spot: index, intensity")
        self.btn_label_expspot.SetToolTipString("Display on plot data related to selected "
                    "(by clicking) theoretical (simulated) spot: #index, hkl miller indices, Energy")
        self.resetAnnotationBtn.SetToolTipString("Reset exp or theo. spot displayed labels on plot")

        self.parametersdisplaypanel.SetToolTipString("View or modifiy current CCD detector plane parameters")

        tpresidues = "Mean residues in pixel over distances between exp. and best refined model spots positions"
        self.txtresidues.SetToolTipString(tpresidues)
        self.act_residues.SetToolTipString(tpresidues)

        tpnb = "Nb of spots associations used for model refinement"
        self.txtnbspots.SetToolTipString(tpnb)
        self.nbspots_in_fit.SetToolTipString(tpnb)

        self.btnsavecalib.SetToolTipString("Save .det file containing current CCD parameters and "
                            "current crystal orientation")
        self.btnsaveresults.SetToolTipString("Save .fit file containing indexed spots used to "
                                "refine the CCD detector plane and pixel frame parameters.")

        tpfit = "Start fitting procedure to refine checked parameters related to crystal orientation and CCD detector plane.\n"
        tpfit += "The model predicts the positions of theoretical spots (red hollow circle).\n"
        tpfit += "Distances between pair of spots (experimental and theoretical) are minimized by a least squares refinement procedures.\n"
        tpfit += 'Spots associations are either build manually ("Manual Links") or automatically ("Auto. Links").'

        self.startfit.SetToolTipString(tpfit)

        self.cb_gotoresults.SetToolTipString("Update CCD parameters and crystal orientation "
        "according to the fit results")
        self.use_weights.SetToolTipString("Refine the model by Weighting each separation distance "
        "between exp. and modeled spots positions by experimental intensity")

        self.incrementfile.SetToolTipString("If Checked, increment filename avoiding overwritten file")

        self.residualstrainbtn.SetToolTipString("Assess the residual strain by fitting orientation and strain only (not detector geometry)")

        self.undogotobtn.SetToolTipString('Undo update of plot from last refined simulated data')

    def layout(self):
        """layout of detectorCalibration GUI"""
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        vbox.Add(self.toolbar, 0, wx.EXPAND)

        hboxlabel = wx.BoxSizer(wx.HORIZONTAL)
        hboxlabel.Add(self.peakpropstxt, 0, wx.ALL, 5)
        hboxlabel.Add(self.btn_label_theospot, 0, wx.ALL, 5)
        hboxlabel.Add(self.btn_label_expspot, 0, wx.ALL, 5)
        hboxlabel.Add(self.resetAnnotationBtn, 0, wx.ALL, 5)
        
        vboxfit2 = wx.BoxSizer(wx.VERTICAL)
        vboxfit2.Add(self.txtresidues, 0, wx.ALL, 0)
        vboxfit2.Add(self.act_residues, 0, wx.ALL, 0)

        vboxfit3 = wx.BoxSizer(wx.VERTICAL)
        vboxfit3.Add(self.txtnbspots, 0, wx.ALL, 0)
        vboxfit3.Add(self.nbspots_in_fit, 0, wx.ALL, 0)

        hboxfit2 = wx.BoxSizer(wx.HORIZONTAL)
        hboxfit2.Add(vboxfit2, 0, wx.ALL, 0)
        hboxfit2.Add(vboxfit3, 0, wx.ALL, 0)
        hboxfit2.Add(wx.StaticText(self.panel, -1, "              "), 0, wx.EXPAND)
        hboxfit2.Add(self.incrementfile, 0, wx.ALL, 0)
        hboxfit2.Add(self.residualstrainbtn, 1, wx.ALL, 5)
        hboxfit2.Add(self.resstrainstatsbtn, 1, wx.ALL, 5)

        vbox2 = wx.BoxSizer(wx.VERTICAL)
        vbox2.Add(hboxlabel, 0, wx.ALL, 0)
        vbox2.Add(self.nb, 0, wx.EXPAND, 0)
        vbox2.Add(self.parametersdisplaypanel, 1, wx.EXPAND, 5)
        # vbox2.Add(wx.StaticLine(self.panel, -1, size=(-1, 10), style=wx.LI_HORIZONTAL),
        #                                                         0, wx.EXPAND|wx.ALL, 5)

        hboxfit = wx.BoxSizer(wx.HORIZONTAL)
        hboxfit.Add(self.startfit, 1, wx.EXPAND|wx.ALL, 5)
        hboxfit.Add(self.undogotobtn, 1, wx.ALL, 5)
        hboxfit.Add(self.cb_gotoresults, 1, wx.ALL, 5)
        hboxfit.Add(self.use_weights, 1, wx.ALL, 5)

        vbox2.AddSpacer(5)
        vbox2.Add(hboxfit, 0, wx.EXPAND, 0)
        vbox2.Add(hboxfit2, 0, wx.EXPAND, 0)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(vbox, 1, wx.EXPAND)
        hbox.Add(vbox2, 1, wx.EXPAND)

        # btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        # btnSizer.AddSpacer(5)
        # btnSizer.Add(self.btnswitchspace, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.btnautolinks, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.btnshowlinks, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.btnmanuallinks, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.txtangletolerance, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.AngleMatchingTolerance, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.btnsaveresults, 1, wx.EXPAND, 10)
        # btnSizer.Add(self.btnsavecalib, 1, wx.EXPAND, 10)
        # btnSizer.AddSpacer(5)


        bottomgrid = wx.GridSizer(10,5,10)
        bottomgrid.Add(self.btnswitchspace, 0, wx.EXPAND, 10)
        bottomgrid.Add(wx.StaticText(self.panel, -1, '   Spots\nSelection ==>'), 0, wx.ALIGN_RIGHT, 0)
        bottomgrid.Add(self.btnautolinks, 0, wx.EXPAND, 2)
        bottomgrid.Add(self.btnshowlinks, 0, wx.EXPAND, 2)
        bottomgrid.Add(self.btnmanuallinks, 0, wx.EXPAND, 0)
        bottomgrid.Add(self.txtangletolerance, 0, wx.ALIGN_RIGHT, 0)
        bottomgrid.Add(self.AngleMatchingTolerance, 0, wx.EXPAND, 10)
        bottomgrid.Add(wx.StaticText(self.panel, -1, '   Save\nResults ==>'), 0, wx.ALIGN_RIGHT, 0)
        bottomgrid.Add(self.btnsaveresults, 0, wx.EXPAND, 10)
        bottomgrid.Add(self.btnsavecalib, 0, wx.EXPAND, 10)

        vboxgeneral = wx.BoxSizer(wx.VERTICAL)
        vboxgeneral.Add(hbox, 1, wx.EXPAND)
        vboxgeneral.AddSpacer(10)
        vboxgeneral.Add(bottomgrid, 0, wx.EXPAND)


        self.panel.SetSizer(vboxgeneral)
        vboxgeneral.Fit(self)
        self.Layout()

    def ReadExperimentData(self):
        """
        - open self.filename (self.dirnamepklist)
        - take into account:
        self.CCDParam
        self.pixelsize
        self.kf_direction

        - set exp. spots attributes:
        self.twicetheta, self.chi = chi
        self.Data_I
        self.filename
        self.data = (self.twicetheta, self.chi, self.Data_I, self.filename)
        self.Data_index_expspot
        self.data_x, self.data_y
        """
        print('\n\nIn ReadExperimentData():  \n\n')

        datfilename = self.filename

        extension = self.filename.split(".")[-1]

        print('extension of self.filename: ', extension)
        
        filepath = os.path.join(self.dirnamepklist, self.filename)
        print('filepath', filepath)

        # print("self.CCDParam in ReadExperimentData()", self.CCDParam)
        # print('self.kf_direction', self.kf_direction)
        # print('self.writefolder', self.writefolder)
        # print('filepath', filepath)

        if not os.access(self.dirnamepklist, os.W_OK):
            if self.writefolder is None:
                self.writefolder = OSLFGUI.askUserForDirname(self)
            print('choosing %s as folder for results  => '%self.writefolder)
        else:
            self.writefolder = self.dirnamepklist

        if extension in ("dat", "DAT"):
            addspotproperties = True
            (twicetheta, chi, dataintensity, data_x, data_y, dict_data_spotsproperties
            ) = F2TC.Compute_data2thetachi(filepath, detectorparams=self.CCDParam,
                                            pixelsize=self.pixelsize,
                                            kf_direction=self.kf_direction,
                                            addspotproperties=addspotproperties)
            self.initialParameter['filename.cor'] = None
            self.initialParameter['filename.dat'] = filepath
            if not filepath.endswith('calib_.dat'):
                self.initialParameter['initialfilename'] = filepath
            print('extension .dat : Reset self.filename to : ', self.filename)
            self.filename = filepath


        elif extension in ("cor",):
            dict_data_spotsproperties = {}
            addspotproperties = False
            nbcolumns_cor = len(IOLT.getcolumnsname_dat(filepath))
            print(f'found {nbcolumns_cor} columns in .cor file :', filepath)
            if nbcolumns_cor>5:
                addspotproperties = True
                #print('There is extra spots properties ...')
                (_, data_theta,
                chi,
                data_x,
                data_y,
                dataintensity,
                _,
                dict_data_spotsproperties) = IOLT.readfile_cor(filepath, output_only5columns=False)

            else:
                addspotproperties = False

                (_, data_theta,
                    chi,
                    data_x,
                    data_y,
                    dataintensity,
                    _) = IOLT.readfile_cor(filepath, output_only5columns=True)
                    
            twicetheta = 2 * data_theta

            if self.filename == 'calib_.cor':
                self.initialParameter['filename.cor'] = self.filename
            else:
                self.initialParameter['initialfilename'] = filepath

            # write a basic temporary calib_.dat file from .cor file
            if addspotproperties:
                print('columnsname', dict_data_spotsproperties['columnsname'])
                if 'Xfiterr' in dict_data_spotsproperties['columnsname']:

                    nbcols = 13
                else:
                    nbcols = 11
            else:
                nbcols = 10  # ???
            print('Preparing array to write in .dat with shape: ', (len(data_theta), nbcols))
            Data_array = np.zeros((len(data_theta), nbcols))
            Data_array[:, 0] = data_x
            Data_array[:, 1] = data_y
            Data_array[:, 2] = dataintensity

            if addspotproperties:
                alldata = dict_data_spotsproperties['data_spotsproperties']
                peakbkg= alldata[:,11]
                Data_array[:, 2] = peakbkg + dataintensity
                Data_array[:, 3] = dataintensity
                if Data_array[:, 4:].shape != alldata[:,4:].shape:
                    print('Data_array, alldata shapes',Data_array[:, 4:].shape, alldata[:,6:].shape)
                Data_array[:, 4:] = alldata[:,6:]

            outputprefix = 'calib_'
            IOLT.writefile_Peaklist(outputprefix, Data_array, overwrite=1,
                                                        initialfilename=self.filename,
                                                        comments=None,
                                                        dirname=self.writefolder)
            self.initialParameter['filename.dat'] = os.path.join(self.dirnamepklist, outputprefix+'.dat')
            # next time in ReadExperimentData  this branch (.cor) won't be used
            self.filename = self.initialParameter['filename.dat']

            print('extension .cor : Reset self.filename to : ', self.filename)

        self.twicetheta = twicetheta
        self.chi = chi
        self.Data_I = dataintensity

        self.data = (self.twicetheta, self.chi, self.Data_I, self.filename)
        # starting X,Y data to plot (2theta , chi)
        self.Data_index_expspot = np.arange(len(self.twicetheta))

        # pixel coordinates of experimental spots
        self.data_x, self.data_y = data_x, data_y

        # peaksearch spots properties
        if len(dict_data_spotsproperties['columnsname'])==dict_data_spotsproperties['data_spotsproperties'].shape[1]:
            self.dict_data_spotsproperties = dict_data_spotsproperties

            print('')
        else:
            print('ERROR in dict_data_spotsproperties !! nb columns', len(dict_data_spotsproperties['columnsname']))

            print('data shape:',  dict_data_spotsproperties['data_spotsproperties'].shape)

    def computeGnomonicExpData(self):
        # compute Gnomonic projection
        (twicetheta, chi, dataintensity) = self.data[:3]
        nbexpspots = len(twicetheta)

        originChi = 0

        if self.plotrangepanel.shiftChiOrigin.GetValue():
            originChi = float(self.plotrangepanel.meanchi.GetValue())

        dataselected = IOLT.createselecteddata((twicetheta, chi + originChi, dataintensity),
                                            np.arange(nbexpspots),
                                            nbexpspots)[0]

        return IIM.ComputeGnomon_2(dataselected)

    def OnSaveCalib(self, _):
        """
        Save detector geometry calibration parameters in .det file
        """
        dlg = wx.TextEntryDialog(self,
            "Enter Calibration File name : \n Current Detector parameters are: \n %s\n Pixelsize and dimensions : %s"
            % (str(self.CCDParam), str([self.pixelsize, self.framedim[0], self.framedim[1]])),
                                                        "Saving Calibration Parameters Entry")
        dlg.SetValue("*.det")
        self.filenameCalib = None
        if dlg.ShowModal() == wx.ID_OK:
            self.filenameCalib = str(dlg.GetValue())
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = np.ravel(self.UBmatrix).round(
                decimals=7)

            dd, xcen, ycen, xbet, xgam = self.CCDParam

            print('chosen   :  self.filenameCalib', self.filenameCalib)

            outputfile = open(os.path.join(self.writefolder,self.filenameCalib), "w")

            text = "%.5f, %.4f, %.4f, %.7f, %.7f, %.8f, %.0f, %.0f\n" % (round(dd, 3),
                                                                        round(xcen, 2),
                                                                        round(ycen, 2),
                                                                        round(xbet, 3),
                                                                        round(xgam, 3),
                                                                        self.pixelsize,
                                                                        round(self.framedim[0], 0),
                                                                        round(self.framedim[1], 0))
            text += "Sample-Detector distance(IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2\n"
            text += "Calibration done with %s at %s with LaueToolsGUI.py\n" % (
                self.crystalparampanel.comboElem.GetValue(), time.asctime())
            text += "Experimental Data file: %s\n" % self.filename
            text += "Orientation Matrix:\n"
            text += "[[%.7f,%.7f,%.7f],[%.7f,%.7f,%.7f],[%.7f,%.7f,%.7f]]\n" % (
                                                    m11, m12, m13, m21, m22, m23, m31, m32, m33)
            #             CCD_CALIBRATION_PARAMETERS = ['dd', 'xcen', 'ycen', 'xbet', 'xgam',
            #                       'xpixelsize', 'ypixelsize', 'CCDLabel',
            #                       'framedim', 'detectordiameter', 'kf_direction']
            vals_list = [round(dd, 3), round(xcen, 2), round(ycen, 2),
                        round(xbet, 3), round(xgam, 3),
                        self.pixelsize, self.pixelsize, self.pixelsize,
                        self.CCDLabel,
                        self.framedim, self.detectordiameter, self.kf_direction]

            key_material = str(self.crystalparampanel.comboElem.GetValue())

            text += "# %s : %s\n" % ("Material", key_material)
            for key, val in zip(DictLT.CCD_CALIBRATION_PARAMETERS, vals_list):
                text += "# %s : %s\n" % (key, val)

            outputfile.write(text[:-1])
            outputfile.close()

        dlg.Destroy()

        if self.filenameCalib is not None:
            fullname = os.path.join(self.writefolder, self.filenameCalib)
            wx.MessageBox("Calibration file written in %s" % fullname, "INFO")

            #             # remove .cor file with old CCD geometry parameters
            #             os.remove(self.initialParameter['filename'])

            # update main GUI CCD geomtrical parameters
            # print("self.parent", self.parent)
            if self.parent:
                self.parent.defaultParam = self.CCDParam
                self.parent.pixelsize = self.pixelsize
                self.parent.kf_direction = self.kf_direction

    def OnStoreMatrix(self, _):
        """
        Store the current UBmatrix in the orientation UBmatrix dictionnary
        """

        tf = TextFrame(self, -1, self.getstringrep(self.crystalparampanel.UBmatrix),
                        self.storedmatrixindex)
        tf.Show(True)
        self.storedmatrixindex += 1
        return

    #         # current UBmatrix  : self.UBmatrix
    #
    #         # dialog for UBmatrix name
    #         dlg = wx.TextEntryDialog(self, 'Enter Matrix Name : \n Current Matrix is: \n %s' % \
    #                                  self.getstringrep(self.crystalparampanel.UBmatrix), 'Storing Matrix Name Entry')
    #         dlg.SetValue('')
    #         if dlg.ShowModal() == wx.ID_OK:
    #             matrix_name = str(dlg.GetValue())
    #
    #             self.crystalparampanel.UBmatrix = np.array(self.crystalparampanel.UBmatrix)
    #             DictLT.dict_Rot[matrix_name] = self.crystalparampanel.UBmatrix.tolist()
    #             self.crystalparampanel.comboMatrix.Append(matrix_name)
    #             dlg.Destroy()

    def getstringrep(self, matrix):
        if isinstance(matrix, np.ndarray):
            listmatrix = matrix.tolist()
        else:
            raise ValueError("matrix is not an array ?")
        strmat = "["
        for row in listmatrix:
            strmat += str(row) + ",\n"

        return strmat[:-2] + "]"

    def OnShowAndFilter(self, _):
        """ on button Filter Links"""
        print('\n In OnShowAndFilter(): \n')


        fields = ["#Spot Exp", "#Spot Theo", "h", "k", "l", "Intensity", "Energy(keV)","residues"]
        # self.linkedspots = dia.listofpairs
        # self.linkExpMiller = dia.linkExpMiller
        # self.linkIntensity = dia.linkIntensity
        # self.linkEnergy = dia.linkEnergy

        indExp = self.linkedspots[:, 0]
        indTheo = self.linkedspots[:, 1]
        _h, _k, _l = np.transpose(np.array(self.linkExpMiller, dtype=np.int16))[1:4]
        intens = self.linkIntensity
        energy = self.linkEnergy
        if self.linkResidues is not None:
            residues = np.array(self.linkResidues)[:, 2]
        else:
            residues = -1 * np.ones(len(indExp))

        to_put_in_dict = [indExp, indTheo, _h, _k, _l, intens, energy, residues]

        if len(self.dict_data_spotsproperties)>0:
            print('Handling extra spots properties')
            ar_alldata = self.dict_data_spotsproperties['data_spotsproperties']
            print('ar_alldata.shape', ar_alldata.shape)
            assert ar_alldata.shape[1]>0

            int_indExp = np.array(indExp, dtype=np.int32)
            print('int_indExp.shape', int_indExp.shape)
            ar_data = np.take(ar_alldata,int_indExp, axis=0).T.tolist()
            
            to_put_in_dict += ar_data

            print('final length of to_put_in_dict', len(to_put_in_dict))
            fields += self.dict_data_spotsproperties['columnsname']
            print('final length of fields', len(fields))


        mySpotData = {}
        for k, ff in enumerate(fields):
            mySpotData[ff] = to_put_in_dict[k]
        dia = LSEditor.SpotsEditor(None, -1, "Spots Editor in Calibration Board",
                                    mySpotData,
                                    func_to_call=self.readdata_fromEditor_Filter,
                                    field_name_and_order=fields)

        dia.Show(True)

    def readdata_fromEditor_Filter(self, data):
        """function to set filtered links from SpotsEditor
        
        set attributes self.link#####"""

        ArrayReturn = np.array(data)

        self.linkedspots = ArrayReturn[:, :2]
        self.linkExpMiller = np.take(ArrayReturn, [0, 2, 3, 4], axis=1)
        self.linkIntensity = ArrayReturn[:, 5]
        self.linkEnergy  = ArrayReturn[:, 6]
        self.linkResidues = np.take(ArrayReturn, [0, 1, 7], axis=1)

        self.plotlinks = self.linkedspots


    def OnLinkSpotsAutomatic(self, _):
        """ create automatically links between currently close experimental
        and theoretical spots in 2theta chi representation

        .. todo::
            use getProximity() ??
        """
        veryclose_angletol = float(self.AngleMatchingTolerance.GetValue())  # in degrees

        # theoretical data
        twicetheta, chi, Miller_ind, posx, posy, energy = self.simulate_theo(removeharmonics=1)
        
        # experimental data (set exp. spots attributes)
        self.ReadExperimentData()

        # print("theo. spots")
        # print("k, x, y, 2theta, theta, chi hkl")
        for k in range(len(twicetheta)):
            print(k, posx[k], posy[k], twicetheta[k], twicetheta[k] / 2, chi[k], Miller_ind[k])

        # print('theo', np.array([twicetheta, chi]).T)
        # print('exp' , np.array([self.twicetheta, self.chi]).T)

        Resi, ProxTable = matchingrate.getProximity(np.array([twicetheta, chi]),  # warning array(2theta, chi)
                                        self.twicetheta / 2.0,
                                        self.chi,  # warning theta, chi for exp
                                        proxtable=1,
                                        angtol=5.0,
                                        verbose=0,
                                        signchi=1)[:2]  # sign of chi is +1 when apparently SIGN_OF_GAMMA=1

        # array theo spot index
        very_close_ind = np.where(Resi < veryclose_angletol)[0]
        # print "In OnLinkSpotsAutomatic() very close indices",very_close_ind
        longueur_very_close = len(very_close_ind)

        List_Exp_spot_close = []
        Miller_Exp_spot = []
        Energy_Spot = []

        # todisplay = ''
        if longueur_very_close > 0:
            for theospot_ind in very_close_ind:  # loop over theo spots index

                List_Exp_spot_close.append(ProxTable[theospot_ind])
                Miller_Exp_spot.append(Miller_ind[theospot_ind])
                Energy_Spot.append(energy[theospot_ind])

                # todisplay += "theo # %d   exp. # %d  Miller : %s \n"%(spot_ind, ProxTable[spot_ind],str(TwicethetaChi[0][spot_ind].Millers))
                # print "theo # %d   exp. # %d  Miller : %s"%(spot_ind, ProxTable[spot_ind],str(TwicethetaChi[0][spot_ind].Millers))
        # print "List_Exp_spot_close",List_Exp_spot_close
        # print "Miller_Exp_spot",Miller_Exp_spot

        # removing exp spot which appears many times(close to several simulated spots of one grain)--------------
        arrayLESC = np.array(List_Exp_spot_close, dtype=float)

        sorted_LESC = np.sort(arrayLESC)

        diff_index = sorted_LESC - np.array(list(sorted_LESC[1:]) + [sorted_LESC[0]])
        toremoveindex = np.where(diff_index == 0)[0]

        # print "List_Exp_spot_close", List_Exp_spot_close
        # print "sorted_LESC", sorted_LESC
        # print "toremoveindex", toremoveindex

        # print "number labelled exp spots", len(List_Exp_spot_close)
        # print "List_Exp_spot_close", List_Exp_spot_close
        # print "Miller_Exp_spot", Miller_Exp_spot

        if len(toremoveindex) > 0:
            # index of exp spot in arrayLESC that are duplicated
            ambiguous_exp_ind = GT.find_closest(
                np.array(sorted_LESC[toremoveindex], dtype=float), arrayLESC, 0.1)[1]
            # print "ambiguous_exp_ind", ambiguous_exp_ind

            # marking exp spots(belonging ambiguously to several simulated grains)
            for ind in ambiguous_exp_ind:
                Miller_Exp_spot[ind] = None
                Energy_Spot[ind] = 0.0

        # -----------------------------------------------------------------------------------------------------
        ProxTablecopy = copy.copy(ProxTable)
        # tag duplicates in ProxTable with negative sign ----------------------
        # ProxTable[index_theo]  = index_exp   closest link

        #ProxTable = list of theo_ind, exp_ind
        for _, exp_ind in enumerate(ProxTable):
            where_th_ind = np.where(ProxTablecopy == exp_ind)[0]
            # print "theo_ind, exp_ind ******** ",theo_ind, exp_ind
            if len(where_th_ind) > 1:
                # exp spot(exp_ind) is close to several theo spots
                # then tag the index with negative sign
                for indy in where_th_ind:
                    ProxTablecopy[indy] = -ProxTable[indy]
                # except that which corresponds to the closest
                closest = np.argmin(Resi[where_th_ind])
                # print "residues = Resi[where_th_ind]",Resi[where_th_ind]
                # print "closest",closest
                # print "where_exp_ind[closest]",where_th_ind[closest]
                # print "Resi[where_th_ind[closest]]", Resi[where_th_ind[closest]]
                ProxTablecopy[where_th_ind[closest]] = -ProxTablecopy[where_th_ind[closest]]

        singleindices = []
        calib_indexed_spots = {}

        for k in range(len(List_Exp_spot_close)):

            exp_index = List_Exp_spot_close[k]
            if not singleindices.count(exp_index):
                # there is not exp_index in singleindices
                singleindices.append(exp_index)

                theo_index = np.where(ProxTablecopy == exp_index)[0]
                # print "theo_index", theo_index

                if len(theo_index) == 1:
                    # fill with expindex,[h,k,l]
                    calib_indexed_spots[exp_index] = [exp_index,
                                                    theo_index,
                                                    Miller_Exp_spot[k],
                                                    Energy_Spot[k]]
                else:  # recent PATCH:
                    print("Resi[theo_index]", Resi[theo_index])
                    closest_theo_ind = np.argmin(Resi[theo_index])
                    # print theo_index[closest_theo_ind]
                    if Resi[theo_index][closest_theo_ind] < veryclose_angletol:
                        calib_indexed_spots[exp_index] = [exp_index,
                                                        theo_index[closest_theo_ind],
                                                        Miller_Exp_spot[k],
                                                        Energy_Spot[k]]
            else:
                print("Experimental spot #%d may belong to several theo. spots!"
                    % exp_index)

        # find theo spot linked to exp spot ---------------------------------

        # calib_indexed_spots is a dictionnary:
        # key is experimental spot index and value is [experimental spot index,h,k,l]
        #print("calib_indexed_spots", calib_indexed_spots)

        listofpairs = []
        linkExpMiller = []
        linkIntensity = []
        linkResidues = []
        linkEnergy = []
        # for val in list(calib_indexed_spots.values()):
        #     if val[2] is not None:
        #         listofpairs.append([val[0], val[1]])  # Exp, Theo,  where -1 for specifying that it came from automatic linking
        #         linkExpMiller.append([float(val[0])] + [float(elem) for elem in val[2]])  # float(val) for further handling as floats array
        #         linkIntensity.append(self.Data_I[val[0]])
        #         linkResidues.append([val[0], val[1], Resi[val[1]]])

        for val in list(calib_indexed_spots.values()):
            if val[2] is not None:
                if not isinstance(val[1], (list, np.ndarray)):
                    closetheoindex = val[1]
                else:
                    closetheoindex = val[1][0]

                listofpairs.append([val[0], closetheoindex])  # Exp, Theo,  where -1 for specifying that it came from automatic linking
                linkExpMiller.append([float(val[0])] + [float(elem) for elem in val[2]])  # float(val) for further handling as floats array
                linkIntensity.append(self.Data_I[val[0]])
                linkResidues.append([val[0], closetheoindex, Resi[closetheoindex]])
                linkEnergy.append(val[3])


        self.linkedspots = np.array(listofpairs)
        self.linkExpMiller = linkExpMiller
        self.linkIntensity = linkIntensity
        self.linkResidues = linkResidues
        self.linkEnergy = linkEnergy

        self.plotlinks = self.linkedspots

        return calib_indexed_spots

    def OnLinkSpots(self, _):  # manual links
        """
        open an editor to link manually spots(exp, theo) for the next fitting procedure
        """
        print("self.linkExpMiller", self.linkExpMiller)

        dia = SLE.LinkEditor(None, -1, "Link between spots Editor", self.linkExpMiller,
                                                                    self.Miller_ind,
                                                                    intensitylist=self.Data_I)

        dia.Show(True)
        #         dia.Destroy()
        self.linkedspots = dia.listofpairs
        self.linkExpMiller = dia.linkExpMiller
        self.linkIntensity = dia.linkIntensity
        self.linkEnergy = dia.linkIntensity

        self.plotlinks = self.linkedspots

    def OnUndoGoto(self, evt):
        self.cb_gotoresults.SetValue(False)

        # updating plot of theo. and exp. spots in calibFrame

        #         print "\n\nUndo Last go to refinement results detector\n"
        #         print "old ccd:", self.CCDParam
        #         print "old UBmatrix", self.UBmatrix

        self.CCDParam = copy.copy(self.previous_CCDParam)
        self.UBmatrix = self.previous_UBmatrix

        #         print "new ccd:", self.CCDParam
        #         print "new UBmatrix", self.UBmatrix
        #
        #         print '\n******\n\n'
        #
        #         if len(arr_indexvaryingparameters) > 1:
        #             for k, val in enumerate(arr_indexvaryingparameters):
        #                 if val < 5:  # only detector params
        #                     self.CCDParam[val] = results[k]
        #         elif len(arr_indexvaryingparameters) == 1:
        #             # only detector params [dd,xcen,ycen,alpha1,alpha2]
        #             if arr_indexvaryingparameters[0] < 5:
        #                 self.CCDParam[arr_indexvaryingparameters[0]] = results[0]
        #         print "New parameters", self.CCDParam
        #
        #         # update orient UBmatrix
        #         print "updating orientation parameters"
        #         # self.UBmatrix = np.dot(deltamat, self.UBmatrix)

        self.crystalparampanel.UBmatrix = self.UBmatrix
        self.deltamatrix = np.eye(3)  # identity
        #         # self.B0matrix is unchanged
        #
        # update exp and theo data
        self.update_data(evt)

    def OnResidualStrainStatistics(self, event):
        dlg = wx.TextEntryDialog(self, "Strain will be estimated 'Nbtrials' times from a set of 'Nspots' randomly chosen spots. Enter: NbTrials, Nspots values",'Residual Strain Statistics', value="20,20")

        if dlg.ShowModal() == wx.ID_OK:
            NbTrials, Nspots = map(int, dlg.GetValue().split(','))
            print('NbTrials', NbTrials)
            print('Nspots', Nspots)
        dlg.Destroy()

        listmaxlevelstrain = []
        listresidues = []
        for ii in range(NbTrials):
            
            maxstrain, residues = self.OnAssessResidualStrain(event,displayresults=False, verbose=0, subsetsize=Nspots)
            print(f' trial {ii+1}/{NbTrials} maxstrain = ', maxstrain)
            listmaxlevelstrain.append(maxstrain)
            listresidues.append(residues)
 
        print('STATISTICS On Residual STRAIN  (10-3 units)')
        ar_maxlevelstrain = np.array(listmaxlevelstrain)
        ar_listresidues = np.array(listresidues)

        # print('listmaxlevelstrain', ar_maxlevelstrain)
        # print('listresidues', listresidues)
        print('STRAIN')
        print('Mean value', np.mean(ar_maxlevelstrain))
        print('Min and Max Value', np.amin(ar_maxlevelstrain), np.amax(ar_maxlevelstrain))
        print('std', np.std(ar_maxlevelstrain))

        print('RESIDUES')
        print('Mean value', np.mean(ar_listresidues))
        print('Min and Max Value', np.amin(ar_listresidues), np.amax(ar_listresidues))
        print('std', np.std(ar_listresidues))

    

    def OnAssessResidualStrain(self, event, displayresults=True, verbose=1, subsetsize = 0):
        """Single Crystal orientation and lattice parameters refinement (to assess residual strain afetr detector geometry calibration refinement.
        """        
        if self.linkedspots is None:
            wx.MessageBox('You need to create first links between experimental and simulated spots '
                            'with the "link spots" button.',
                            "INFO")
            event.Skip()
            return
        
        if verbose:
            print("\nIn OnAssessResidualStrain")
            print('self.initialParameter["dirname"]', self.initialParameter["dirname"])
            print('self.filename', self.filename)
            #print("Pairs of spots used", self.linkedspots)

        if subsetsize > 0:
            if verbose:
                print("\n\n ***************\nIn OnAssessResidualStrain")
                print('self.linkedspots', self.linkedspots)
            ar_ind = np.arange(len(self.linkedspots))
            np.random.shuffle(ar_ind)
            randindices = ar_ind[:subsetsize]
            arraycouples = np.take(np.array(self.linkedspots),randindices, axis=0)
            if verbose:
                print('arraycouples', arraycouples)
                print('***********\n\n')
            exp_indices = np.array(arraycouples[:, 0], dtype=np.int16)
            nb_pairs = len(exp_indices)
            #sim_indices = np.array(arraycouples[:, 1], dtype=np.int16)
            Data_Q = np.take(np.array(self.linkExpMiller)[:, 1:],randindices, axis=0)
            sim_indices = np.arange(nb_pairs)

        else:
            arraycouples = np.array(self.linkedspots)

            Data_Q = np.array(self.linkExpMiller)[:, 1:]
            exp_indices = np.array(arraycouples[:, 0], dtype=np.int16)
            #sim_indices = np.array(arraycouples[:, 1], dtype=np.int16)
            nb_pairs = len(exp_indices)
            sim_indices = np.arange(nb_pairs)

        if verbose: print("Nb of pairs theo-exp spots: ", nb_pairs)
        #print(exp_indices, sim_indices)

        # self.data_theo contains the current simulated spots: twicetheta, chi, Miller_ind, posx, posy
        # Data_Q = self.data_theo[2]  # all miller indices must be entered with sim_indices = arraycouples[:,1]

        #print("DataQ from self.linkExpMiller", Data_Q)

        # experimental spots selection from self.data_x, self.data_y(loaded when initialising calibFrame)
        pixX, pixY = (np.take(self.data_x, exp_indices),
                        np.take(self.data_y, exp_indices))  # pixel coordinates
        # twth, chi = np.take(self.twicetheta, exp_indices),np.take(self.chi, exp_indices)  # 2theta chi coordinates

        # initial parameters of calibration and misorientation from the current orientation UBmatrix
        if verbose: print("detector parameters", self.CCDParam)

        starting_orientmatrix = self.crystalparampanel.UBmatrix

        # starting B0matrix corresponding to the unit cell   -----
        latticeparams = DictLT.dict_Materials[self.key_material][1]
        B0matrix = CP.calc_B_RR(latticeparams)

        # initial distorsion is  1 1 0 0 0  = refined_22,refined_33, 0,0,0
        allparameters = np.array(self.CCDParam + [1, 1, 0, 0, 0] + [0, 0, 0])

        # change ycen if grain is below the surface (NOT ALONG beam direction (ybeam)):
        # depth is counted positively below surface in microns
        #depth = float(self.sampledepthctrl.GetValue())
        depth = 0 
        depth_along_beam = depth / np.sin(40 * np.pi / 180.)
        delta_ycen = depth_along_beam/1000./self.pixelsize
        allparameters[2] += delta_ycen

        # strain & orient
        initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
        arr_indexvaryingparameters = np.arange(5, 13)

        # if self.fitycen.GetValue():
        #     initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, allparameters[2]])
        #     arr_indexvaryingparameters = np.append(np.arange(5, 13), 2)
        if verbose:
            print("\nInitial error--------------------------------------\n")

            print("initial_values, allparameters, arr_indexvaryingparameters")
            print(initial_values, allparameters, arr_indexvaryingparameters)

        residues, deltamat, _ = FitO.error_function_on_demand_strain(
                                                                initial_values,
                                                                Data_Q,
                                                                allparameters,
                                                                arr_indexvaryingparameters,
                                                                sim_indices,
                                                                pixX,
                                                                pixY,
                                                                initrot=starting_orientmatrix,
                                                                Bmat=B0matrix,
                                                                pureRotation=0,
                                                                verbose=1,
                                                                pixelsize=self.pixelsize,
                                                                dim=self.framedim,
                                                                weights=None,
                                                                kf_direction=self.kf_direction)

        if verbose:
            print("\nInitial error--------------------------------------\n")
            print('residues', residues)
            print("mean Initial residues", np.mean(residues))
            print("---------------------------------------------------\n")

        results = FitO.fit_on_demand_strain(initial_values,
                                                    Data_Q,
                                                    allparameters,
                                                    FitO.error_function_on_demand_strain,
                                                    arr_indexvaryingparameters,
                                                    sim_indices,
                                                    pixX,
                                                    pixY,
                                                    initrot=starting_orientmatrix,
                                                    Bmat=B0matrix,
                                                    pixelsize=self.pixelsize,
                                                    dim=self.framedim,
                                                    verbose=verbose,
                                                    weights=None,
                                                    kf_direction=self.kf_direction)


        if verbose:
            print("\n********************\n       Results of Fit        \n********************")
            print("results", results)

        if results is None:
            return

        

        if verbose: print("\nFinal error--------------------------------------\n")
        residues, deltamat, refinedUB = FitO.error_function_on_demand_strain(
                                                                results,
                                                                Data_Q,
                                                                allparameters,
                                                                arr_indexvaryingparameters,
                                                                sim_indices,
                                                                pixX,
                                                                pixY,
                                                                initrot=starting_orientmatrix,
                                                                Bmat=B0matrix,
                                                                pureRotation=0,
                                                                verbose=1,
                                                                pixelsize=self.pixelsize,
                                                                dim=self.framedim,
                                                                weights=None,
                                                                kf_direction=self.kf_direction)

        
        if verbose:
            print("Final residues", residues)
            print("---------------------------------------------------\n")
            print("mean", np.mean(residues))

        # building B mat
        param_strain_sol = results
        varyingstrain = np.array([[1.0, param_strain_sol[2], param_strain_sol[3]],
                                        [0, param_strain_sol[0], param_strain_sol[4]],
                                        [0, 0, param_strain_sol[1]]])
        if verbose:
            print("varyingstrain results")
            print(varyingstrain)

        # if self.fitycen.GetValue():
        #     print("fitted ycen", param_strain_sol[8])
        #     print('calib ref. ycen: ', allparameters[2])
        #     print('delta ycen: ', param_strain_sol[8] - allparameters[2])

        newUmat = np.dot(deltamat, starting_orientmatrix)

        # building UBmat(= newmatrix)
        newUBmat = np.dot(newUmat, varyingstrain)

        # ---------------------------------------------------------------
        # postprocessing of unit cell orientation and strain refinement
        # ---------------------------------------------------------------
        if verbose:
            print("newUBmat", newUBmat)
            print("refinedUB", refinedUB)
            print("self.newUBmat after fitting", newUBmat)
        maxlevelstrain = self.evaluate_strain_display_results(newUBmat,
                                            self.key_material,
                                            residues,
                                            nb_pairs,
                                            constantlength="a",displayresults=displayresults)
        
        return maxlevelstrain, residues
        
    def evaluate_strain_display_results(self,
                                        newUBmat,
                                        key_material,
                                        residues_non_weighted,
                                        nb_pairs,
                                        constantlength="a", displayresults=True):
        """
        evaluate strain and display fitting results
        :param newUBmat: array, 3x3 orientation matrix UBrefined
        :param key_material: str, label for material taht sets the lattice parameters and unit cell shape
        :param residues_non_weighted: array or list,  to estimate the mean pixel deviation
        :param nb_pairs: int, nb of pairs used in the refinement
        :param constantlength: str, "a", "b", or "c" to set the length having been kept during refinement
        """
        # compute new lattice parameters  -----
        latticeparams = DictLT.dict_Materials[key_material][1]
        B0matrix = CP.calc_B_RR(latticeparams)

        UBmat = copy.copy(newUBmat)

        (devstrain, lattice_parameter_direct_strain) = CP.compute_deviatoricstrain(
                                                                UBmat, B0matrix, latticeparams)
        # overwrite and rescale possibly lattice lengthes
        lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(
                                                            UBmat, key_material, constantlength,
                                                            dictmaterials=DictLT.dict_Materials)

        print("final lattice_parameter_direct_strain", lattice_parameter_direct_strain)

        deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(
                                                                        devstrain, UBmat)

        devstrain_sampleframe_round = np.round(deviatoricstrain_sampleframe * 1000, decimals=3)
        devstrain_round = np.round(devstrain * 1000, decimals=3)



        # ADDONS: strain in lauetools frame:
        devstrain_LTframe = np.round(CP.strain_from_crystal_to_LaueToolsframe(devstrain, UBmat)*1000,decimals=3)
        print('====> **** devstrain_LTframe',devstrain_LTframe)
        print('*************************************\n')

        # TODO: to complete ---------------------
        # devstrain_crystal_voigt = np.take(np.ravel(np.array(devstrain)), (0, 4, 8, 5, 2, 1))

        UBB0mat = np.dot(newUBmat, B0matrix)

        Umat = None

        Umat = CP.matstarlab_to_matstarlabOND(matstarlab=None, matLT3x3=np.array(UBmat))
        # TODO to be translated !----------------------
        # conversion in np array is necessary from automatic indexation results, but not necessary from check orientation results

        print("**********test U ****************************")
        print("U matrix = ")
        print(Umat.round(decimals=9))
        print("norms :")
        for i in range(3):
            print(i, GT.norme_vec(Umat[:, i]).round(decimals=5))
        print("scalar products")
        for i in range(3):
            j = np.mod(i + 1, 3)
            print(i, j, np.inner(Umat[:, i], Umat[:, j]).round(decimals=5))
        print("determinant")
        print(np.linalg.det(Umat).round(decimals=5))

        Bmat_triang_up = np.dot(np.transpose(Umat), UBmat)

        print(" Bmat_triang_up= ")
        print(Bmat_triang_up.round(decimals=9))

        (list_HKL_names,
        HKL_xyz) = CP.matrix_to_HKLs_along_xyz_sample_and_along_xyz_lab(
                                                        matstarlab=None,  # OR
                                                        UBmat=UBB0mat,  # LT , UBB0 ici
                                                        omega=None,  # was MG.PAR.omega_sample_frame,
                                                        mat_from_lab_to_sample_frame=None,
                                                        results_in_OR_frames=0,
                                                        results_in_LT_frames=1,
                                                        sampletilt=40.0)
        HKLxyz_names = list_HKL_names
        HKLxyz = HKL_xyz

        initial_filepath = self.initialParameter['initialfilename']
        initialfolder, initialfilename= os.path.split(initial_filepath)

        texts_dict = {}

        txt0 = "Filename: %s\t\t\tDate: %s\t\tPlotRefineGUI.py\n" % (initialfilename,
                                                                    time.asctime())
        txt0 += 'Folder: %s\n'%initialfolder
        txt0 += "Mean Pixel Deviation: %.3f\n" % np.mean(residues_non_weighted)
        txt0 += "Number of refined Laue spots: %d\n" % nb_pairs
        texts_dict["NbspotsResidues"] = txt0

        txt1 = "Deviatoric Strain (10-3 units) in crystal frame (direct space) \n"
        for k in range(3):
            txt1 += "%.3f   %.3f   %.3f\n" % tuple(devstrain_round[k])
        texts_dict["devstrain_crystal"] = txt1

        txt2 = "Deviatoric Strain (10-3 units) in sample frame (tilt=40deg)\n"
        for k in range(3):
            txt2 += "%.3f   %.3f   %.3f\n" % tuple(devstrain_sampleframe_round[k])
        texts_dict["devstrain_sample"] = txt2

        #         txt3 = 'Full Strain (10-3 units) sample frame (tilt=40deg)\n'
        #         txt3 += 'Assumption: cubic material + stress33=0\n'
        #         for k in range(3):
        #             txt3 += '%.3f   %.3f   %.3f\n' % tuple(fullstrain_round[k])
        txt3 = ""
        texts_dict["fullstrain_sample"] = txt3

        txtinitlattice = "Initial Lattice Parameters\n"
        paramcellname = ["  a", "  b", "  c", "alpha", "beta", "gamma"]
        for name, val in zip(paramcellname, latticeparams):
            txtinitlattice += "%s\t\t%.6f\n" % (name, val)

        texts_dict["Initial lattice"] = txtinitlattice

        txtfinallattice = "Refined Lattice Parameters\n"
        paramcellname = ["  a", "  b", "  c", "alpha", "beta", "gamma"]
        for name, val in zip(paramcellname, lattice_parameter_direct_strain):
            txtfinallattice += "%s\t\t%.6f\n" % (name, val)

        texts_dict["Refined lattice"] = txtfinallattice

        

        txtUB = "UB matrix in q = UB B0 G*\n"
        txtUB += "["
        for k in range(3):
            txtUB += "[%.8f, %.8f, %.8f],\n" % tuple(UBmat[k])
        texts_dict["UBmatrix"] = txtUB[:-2] + "]"

        txtB0 = "B0 matrix in q = UB B0 G*\n"
        txtB0 += "["
        for k in range(3):
            txtB0 += "[%.8f, %.8f, %.8f],\n" % tuple(B0matrix[k])
        texts_dict["B0matrix"] = txtB0[:-2] + "]"

        txtHKLxyz_names = "                                 HKL frame coordinates\n"
        listvectors = ["x=[100]_LT :",
                        "y=[010]_LT :",
                        "z=[001]_LT :",
                        "xs=[100]_LTsample :",
                        "ys=[010]_LTsample :",
                        "zs=[001]_LTsample :"]
        for k in range(6):
            txtHKLxyz_names += listvectors[k] + "\t [%.3f, %.3f, %.3f]\n" % tuple(
                HKLxyz[k])
            texts_dict["HKLxyz_names"] = txtHKLxyz_names

        print(txtHKLxyz_names)

        # if 0:
        #     txtHKLxyz = "HKL = \n"
        #     txtHKLxyz += "["
        #     for k in range(6):
        #         txtHKLxyz += "[%.3f, %.3f, %.3f],\n" % tuple(self.HKLxyz[k])
        #         texts_dict["HKLxyz"] = txtHKLxyz[:-2] + "]"
        #     print(txtHKLxyz)
        texts_dict["HKLxyz"] = ""

        if displayresults:
            from LaueTools.GUI import PlotRefineGUI as PRGUI

            frb = PRGUI.FitResultsBoard(self, -1, "REFINEMENT RESULTS", texts_dict)
            frb.ShowModal()

            frb.Destroy()

        return max(np.amax(np.fabs(devstrain_round)),np.amax(np.fabs(devstrain_sampleframe_round)))
        

    def StartFit(self, event):
        """
        StartFit in calib frame

        Single Crystal orientation and detector geometry parameters refinement
        """
        if self.linkedspots is None:
            wx.MessageBox('You need to create first links between experimental and simulated spots '
                            'with the "link spots" button.',
                            "INFO")
            event.Skip()
            return

        print("\nIn StartFit()")
        print('self.initialParameter["dirname"]', self.initialParameter["dirname"])
        print('self.filename', self.filename)
        #print("Pairs of spots used", self.linkedspots)
        arraycouples = np.array(self.linkedspots)

        exp_indices = np.array(arraycouples[:, 0], dtype=np.int16)
        sim_indices = np.array(arraycouples[:, 1], dtype=np.int16)

        nb_pairs = len(exp_indices)
        print("Nb of pairs  theo-exp spots: ", nb_pairs)
        #print(exp_indices, sim_indices)

        # self.data_theo contains the current simulated spots: twicetheta, chi, Miller_ind, posx, posy
        # Data_Q = self.data_theo[2]  # all miller indices must be entered with sim_indices = arraycouples[:,1]

        #print("self.linkExpMiller", self.linkExpMiller)
        Data_Q = np.array(self.linkExpMiller)[:, 1:]

        sim_indices = np.arange(nb_pairs)
        #print("DataQ from self.linkExpMiller", Data_Q)

        # experimental spots selection from self.data_x, self.data_y(loaded when initialising calibFrame)
        pixX, pixY = (np.take(self.data_x, exp_indices),
                        np.take(self.data_y, exp_indices))  # pixel coordinates
        # twth, chi = np.take(self.twicetheta, exp_indices),np.take(self.chi, exp_indices)  # 2theta chi coordinates

        # initial parameters of calibration and misorientation from the current orientation UBmatrix
        print("detector parameters", self.CCDParam)

        allparameters = np.array(self.CCDParam + [0, 0, 0])  # 3 last params = 3 quaternion angles not used here

        # select the parameters that must be fitted
        boolctrl = [ctrl.GetValue() for ctrl in self.moveccdandxtal.listofparamfitctrl]
        varyingparameters = []
        init_values = []
        for k, val in enumerate(boolctrl):
            if val:
                varyingparameters.append(k)
                init_values.append(allparameters[k])

        if not bool(varyingparameters):
            wx.MessageBox("You need to select at least one parameter to fit!!", "INFO")
            return

        listparam = ["distance(mm)",
                    "Xcen(pixel)",
                    "Ycen(pixel)",
                    "Angle1(deg)",
                    "Angle2(deg)",  # detector parameter
                    "theta1(deg)",
                    "theta2(deg)",
                    "theta3(deg)"]  # misorientation with respect to initial UBmatrix(/ elementary axis rotation)

        # start fit
        initial_values = np.array(init_values)  # [dd, xcen, ycen, ang1, ang2, theta1, theta2, theta3]
        arr_indexvaryingparameters = np.array(varyingparameters)  # indices of position of parameters in [dd, xcen, ycen, ang1, ang2, theta1, theta2, theta3]

        self.UBmatrix = self.crystalparampanel.UBmatrix

        print("starting fit of :", [listparam[k] for k in arr_indexvaryingparameters])
        print("With initial values: ", initial_values)
        # print "miller selected ",np.take(self.data_theo[2],sim_indices, axis = 0) ????
        print("allparameters", allparameters)
        print("arr_indexvaryingparameters", arr_indexvaryingparameters)
        print("nb_pairs", nb_pairs)
        print("indices of simulated spots(selection in whole Data_Q list)", sim_indices)
        print("Experimental pixX, pixY", pixX, pixY)
        print("self.UBmatrix", self.UBmatrix)
        print("self.kf_direction", self.kf_direction)

        pureRotation = 0  # OR, was 1

        if self.use_weights.GetValue():
            weights = self.linkIntensity
        else:
            weights = None

        # fitting procedure for one or many parameters
        nb_fittingparams = len(arr_indexvaryingparameters)
        if nb_pairs < nb_fittingparams:
            wx.MessageBox("You need at least %d spots links to fit these %d parameters."
                            % (nb_fittingparams, nb_fittingparams),
                            "INFO")
            event.Skip()
            return

        print("Initial error--------------------------------------\n")
        residues, deltamat, newmatrix = FitO.error_function_on_demand_calibration(
                                            initial_values,
                                            Data_Q,
                                            allparameters,
                                            arr_indexvaryingparameters,
                                            sim_indices,
                                            pixX,
                                            pixY,
                                            initrot=self.UBmatrix,
                                            vecteurref=self.B0matrix,
                                            pureRotation=pureRotation,
                                            verbose=1,
                                            pixelsize=self.pixelsize,
                                            dim=self.framedim,
                                            weights=weights,
                                            kf_direction=self.kf_direction)
        print("Initial residues", residues)
        print("---------------------------------------------------\n")

        diag = None

        # if self.kf_direction in ('X>0', 'X<0'):
        #     diag = [1,1,1,1,10,1,1,1]
        results = FitO.fit_on_demand_calibration(initial_values,
                                                Data_Q,
                                                allparameters,
                                                FitO.error_function_on_demand_calibration,
                                                arr_indexvaryingparameters,
                                                sim_indices,
                                                pixX,
                                                pixY,
                                                initrot=self.UBmatrix,
                                                vecteurref=self.B0matrix,
                                                pureRotation=pureRotation,
                                                pixelsize=self.pixelsize,
                                                dim=self.framedim,
                                                verbose=0,
                                                weights=weights,
                                                kf_direction=self.kf_direction,
                                                diag=diag)

        print("\n********************\n       Results of Fit        \n********************")
        print("results", results)
        allresults = allparameters

        if nb_fittingparams == 1:
            results = [results]

        print("weights = ", weights)

        residues, deltamat, newmatrix = FitO.error_function_on_demand_calibration(
                                        results,
                                        Data_Q,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        sim_indices,
                                        pixX,
                                        pixY,
                                        initrot=self.UBmatrix,
                                        vecteurref=self.B0matrix,
                                        pureRotation=pureRotation,
                                        verbose=1,
                                        pixelsize=self.pixelsize,
                                        dim=self.framedim,
                                        weights=weights,
                                        kf_direction=self.kf_direction)

        residues_nonweighted, _delta, _newmatrix, self.SpotsData = FitO.error_function_on_demand_calibration(results,
                                                Data_Q,
                                                allparameters,
                                                arr_indexvaryingparameters,
                                                sim_indices,
                                                pixX,
                                                pixY,
                                                initrot=self.UBmatrix,
                                                vecteurref=self.B0matrix,
                                                pureRotation=pureRotation,
                                                verbose=1,
                                                pixelsize=self.pixelsize,
                                                dim=self.framedim,
                                                weights=None,
                                                allspots_info=1,
                                                kf_direction=self.kf_direction)

        print("last pixdev table")
        print(residues_nonweighted)
        print("Mean pixdev no weights")
        print(np.mean(residues_nonweighted))
        print("Mean pixdev")
        print(np.mean(residues))
        print("initial UBmatrix")
        print(self.UBmatrix)
        print("New delta UBmatrix")
        print(deltamat)
        print("newmatrix")
        print(newmatrix)
        print(newmatrix.tolist())

        if len(arr_indexvaryingparameters) > 1:
            for k, val in enumerate(arr_indexvaryingparameters):
                allresults[val] = results[k]
        elif len(arr_indexvaryingparameters) == 1:
            allresults[arr_indexvaryingparameters[0]] = results[0]

        self.residues_fit = residues_nonweighted
        # display fit results
        dataresults = (allresults.tolist()
                    + [np.mean(self.residues_fit)]
                    + [len(self.residues_fit)])
        self.display_results(dataresults)

        # updating plot of theo. and exp. spots in calibFrame
        if self.cb_gotoresults.GetValue():
            print("Updating plot with new CCD parameters and crystal orientation detector")

            # saving previous results
            self.previous_CCDParam = copy.copy(self.CCDParam)
            self.previous_UBmatrix = copy.copy(self.UBmatrix)

            if len(arr_indexvaryingparameters) > 1:
                for k, val in enumerate(arr_indexvaryingparameters):
                    if val < 5:  # only detector params
                        self.CCDParam[val] = results[k]
            elif len(arr_indexvaryingparameters) == 1:
                # only detector params [dd,xcen,ycen,alpha1,alpha2]
                if arr_indexvaryingparameters[0] < 5:
                    self.CCDParam[arr_indexvaryingparameters[0]] = results[0]
            print("New parameters", self.CCDParam)

            # update orient UBmatrix
            #             print "updating orientation parameters"
            # self.UBmatrix = np.dot(deltamat, self.UBmatrix)
            self.UBmatrix = newmatrix
            self.crystalparampanel.UBmatrix = newmatrix
            self.deltamatrix = np.eye(3)  # identity
            # self.B0matrix is unchanged

            # start ---  OR   ---------
            UBB0 = np.dot(self.UBmatrix, self.B0matrix)

            Umat = CP.matstarlab_to_matstarlabOND(matstarlab=None, matLT3x3=self.UBmatrix)

            print("**********test U ****************************")
            print("U matrix = ")
            print(Umat.round(decimals=5))
            print("normes :")
            for i in range(3):
                print(i, GT.norme_vec(Umat[:, i]).round(decimals=5))
            print("produit scalaire")
            for i in range(3):
                j = np.mod(i + 1, 3)
                print(i, j, np.inner(Umat[:, i], Umat[:, j]).round(decimals=5))
            print("determinant")
            print(np.linalg.det(Umat).round(decimals=5))

            Bmat_triang_up = np.dot(Umat.T, self.UBmatrix)

            print(" Bmat_triang_up= ")
            print(Bmat_triang_up.round(decimals=5))

            self.Umat2 = Umat
            self.Bmat_tri = Bmat_triang_up

            list_HKL_names, HKL_xyz = CP.matrix_to_HKLs_along_xyz_sample_and_along_xyz_lab(
                matstarlab=None,  # OR
                UBmat=UBB0,  # LT , UBB0 ici
                omega=None,  # was MG.PAR.omega_sample_frame,
                mat_from_lab_to_sample_frame=None,
                results_in_OR_frames=0,
                results_in_LT_frames=1,
                sampletilt=40.0)
            self.HKLxyz_names = list_HKL_names
            self.HKLxyz = HKL_xyz
            # end -------  OR   ------------

            # update exp and theo data
            self.update_data(event)

            #print("self.linkedspots at the end of StartFit ", self.linkedspots)
            self.linkedspotsAfterFit = copy.copy(self.linkedspots)
            self.linkExpMillerAfterFit = copy.copy(self.linkExpMiller)
            self.linkIntensityAfterFit = copy.copy(self.linkIntensity)
            self.residues_fitAfterFit = copy.copy(self.residues_fit)
            # for energy no need ... apparently
            #self.linkEnergyAfterFit = copy.copy(self.linkEnergy)

        # update .cor file  self.initialParameter["filename.cor"]
        print("In StartFit(): after refinement self.defaultParam ", self.CCDParam)
        
        fullpathfilename = os.path.join(self.initialParameter["dirname"],
                                        self.filename)
        print('self.initialParameter["dirname"]', self.initialParameter["dirname"])
        print('self.filename', self.filename)

        (twicetheta, chi, dataintensity, data_x, data_y, dict_data_spotsproperties) = F2TC.Compute_data2thetachi(
                                                            fullpathfilename,
                                                            sorting_intensity="yes",
                                                            detectorparams=self.CCDParam,
                                                            pixelsize=self.pixelsize,
                                                            kf_direction=self.kf_direction,
                                                            addspotproperties=True)

        folder, filename = os.path.split(fullpathfilename)
        prefix = filename.split(".")[0]

        if self.writefolder is None and not os.access(folder, os.W_OK):
            self.writefolder = OSLFGUI.askUserForDirname(self)

        IOLT.writefile_cor(prefix, twicetheta, chi, data_x, data_y,
                        dataintensity,
                        sortedexit=False,
                        param=self.CCDParam + [self.pixelsize],
                        initialfilename=self.filename,
                        dirname_output=self.writefolder,
                        dict_data_spotsproperties=dict_data_spotsproperties)  # check sortedexit = 0 or 1 to have decreasing intensity sorted data
        
        print("In StartFit(): end of fit: %s has been updated" % (prefix + ".cor"))
        self.initialParameter["filename.cor"] = prefix + ".cor"

    def OnWriteResults(self, _):
        """
        write a .fit file from refined orientation and detector calibration CCD geometry
        """
        # print("self.linkedspots in OnWriteResults()", self.linkedspots)

        if self.SpotsData is None or self.linkedspotsAfterFit is None:
            wx.MessageBox("You must have run once a calibration refinement!", "INFO")
            return

        # spotsData = [Xtheo,Ytheo, Xexp, Yexp, Xdev, Ydev, theta_theo]
        spotsData = self.SpotsData

        print("\nIn OnWriteResults(): Writing results in .fit file")
        suffix = ""
        if self.incrementfile.GetValue():
            self.savedindex += 1
            suffix = "_%d" % self.savedindex

        outputfilename = self.filename.split(".")[0] + suffix + ".fit"
        folder, filename = os.path.split(outputfilename)

        print("self.writefolder",self.writefolder)
        print("folder",folder)

        if self.writefolder is None:
            self.writefolder = OSLFGUI.askUserForDirname(self)
            
        outputfilename = os.path.join(self.writefolder,filename)

        indExp = np.array(self.linkedspotsAfterFit[:, 0], dtype=np.int16)
        _h, _k, _l = np.transpose(np.array(self.linkExpMillerAfterFit, dtype=np.int16))[1:4]
        intens = self.linkIntensityAfterFit
        residues_calibFit = self.residues_fitAfterFit
        # for energy: it will recalculated below ...

        # elem = self.crystalparampanel.comboElem.GetValue()

        # latticeparam = DictLT.dict_Materials[str(elem)][1][0] * 1.0
        Data_Q = np.array(self.linkExpMillerAfterFit)[:, 1:]

        dictCCD = {}
        dictCCD["CCDparam"] = self.CCDParam
        dictCCD["dim"] = self.framedim
        dictCCD["pixelsize"] = self.pixelsize
        dictCCD["kf_direction"] = self.kf_direction

        spotsProps = LAUE.calcSpots_fromHKLlist(self.UBmatrix, self.B0matrix, Data_Q, dictCCD)
        # H, K, L, Qx, Qy, Qz, Xtheo, Ytheo, twthe, chi, Energy = spotsProps
        Xtheo, Ytheo, twthe, chi, Energy = spotsProps[-5:]

        # print('self.initialParameter["filename.cor"] in OnWriteResults',
        #         self.initialParameter["filename.cor"])
        print('self.filename',self.filename)
        print('self.initialParameter["filename.cor"]', self.initialParameter["filename.cor"])
        print('self.initialParameter["initialfilename"]', self.initialParameter["initialfilename"])

        initialfile = self.initialParameter["initialfilename"]
        print('initialfile  :', initialfile)

        if initialfile.endswith('dat'):
            data_peak = IOLT.read_Peaklist(initialfile)
            initialfileextension = 'dat'
            _, nbcolumns_dat = data_peak.shape
        elif initialfile.endswith('cor'):
            data_peak,_,_,_,_,_,_,dict_spotsproperties = IOLT.readfile_cor(initialfile,output_only5columns=False)
            initialfileextension = 'cor'
            _, nbcolumns_cor = data_peak.shape
        

        selected_data_peak = np.take(data_peak, indExp, axis=0)

        # print('selected_data_peak.shape',selected_data_peak.shape)
        
        if initialfileextension == 'dat' and initialfile != 'calib_.dat':
            nbcolumns_cor = 0
            if nbcolumns_dat == 11:
                (Xexp, Yexp, _, peakAmplitude,
            peak_fwaxmaj, peak_fwaxmin, peak_inclination,
            Xdev_peakFit, Ydev_peakFit, peak_bkg, IntensityMax) = selected_data_peak.T
            elif nbcolumns_dat == 13:
                (Xexp, Yexp, _, peakAmplitude,
            peak_fwaxmaj, peak_fwaxmin, peak_inclination,
            Xdev_peakFit, Ydev_peakFit, peak_bkg, IntensityMax, XfitErr, YfitErr) = selected_data_peak.T

        elif initialfileextension == 'cor':
            nbcolumns_dat = 0
            if nbcolumns_cor==5:
                (_, _, Xexp, Yexp, peakAmplitude) = selected_data_peak.T
                _ = peakAmplitude
                unknowns = np.zeros(len(Xexp))
                peak_fwaxmaj = unknowns
                peak_fwaxmin = unknowns
                peak_inclination = unknowns
                Xdev_peakFit = unknowns
                Ydev_peakFit, peak_bkg, IntensityMax = unknowns, unknowns, unknowns
            else:
                print('%d of colmuns in initialfile'%nbcolumns_cor, initialfile)
                if nbcolumns_cor== 15:
                    (_,_, Xexp, Yexp, peakAmplitude,
                    I_tot,
                peak_fwaxmaj, peak_fwaxmin, peak_inclination,
                Xdev_peakFit, Ydev_peakFit, peak_bkg, IntensityMax, XfitErr, YfitErr) = selected_data_peak.T

                if nbcolumns_cor== 13:
                    (_,_, Xexp, Yexp, peakAmplitude,
                    I_tot,
                peak_fwaxmaj, peak_fwaxmin, peak_inclination,
                Xdev_peakFit, Ydev_peakFit, peak_bkg, IntensityMax) = selected_data_peak.T


        Xdev_calibFit, Ydev_calibFit = spotsData[4:6]

        # #spot index, peakamplitude, h,k,l, Xtheo, Ytheo, Xexp, Yexp, Xdev,
        # Xdev_calibFit, Ydev_calibFit, sqrt(Xdev_calibFit**2+Ydev_calibFit**2)
        # 2thetaTheo, chiTheo, EnergyTheo, peakamplitude, hottestintensity, localintensitybackground
        # peak_fullwidth_axisminor, peak_fullwidth_axismahor, peak elongation direction angle,
        # Xdev_peakfit, Ydev_peakfit (fit by gaussian 2D shape for example)
        Columns = [indExp, intens, _h, _k, _l, Xtheo, Ytheo, Xexp, Yexp,
                Xdev_calibFit, Ydev_calibFit, residues_calibFit,
                twthe, chi, Energy,
                peakAmplitude, IntensityMax, peak_bkg,
                peak_fwaxmaj, peak_fwaxmin, peak_inclination,
                Xdev_peakFit, Ydev_peakFit]
        
        if nbcolumns_cor == 15 or nbcolumns_dat == 15:
            Columns.append(XfitErr)
            Columns.append(YfitErr)

        datatooutput = np.transpose(np.array(Columns))
        datatooutput = np.round(datatooutput, decimals=5)

        # sort by decreasing intensity
        data = datatooutput[np.argsort(datatooutput[:, 1])[::-1]]

        dict_matrices = {}
        dict_matrices["Element"] = self.key_material

        dict_matrices["UBmat"] = self.UBmatrix
        dict_matrices["B0"] = self.B0matrix
        #         dict_matrices['UBB0'] = self.UBB0mat
        #         dict_matrices['devstrain'] = self.deviatoricstrain

        UBB0_v2 = np.dot(dict_matrices["UBmat"], dict_matrices["B0"])
        euler_angles = ORI.calc_Euler_angles(UBB0_v2).round(decimals=3)
        dict_matrices["euler_angles"] = euler_angles

        # Odile Robach's addition
        dict_matrices["UBB0"] = UBB0_v2
        dict_matrices["Umat2"] = self.Umat2
        dict_matrices["Bmat_tri"] = self.Bmat_tri
        dict_matrices["HKLxyz_names"] = self.HKLxyz_names
        dict_matrices["HKLxyz"] = self.HKLxyz
        dict_matrices["detectorparameters"] = list(np.array(self.CCDParam).round(decimals=3))
        dict_matrices["pixelsize"] = self.pixelsize
        dict_matrices["framedim"] = self.framedim
        dict_matrices["CCDLabel"] = self.CCDLabel

        columnsname = "spot_index Itot h k l Xtheo Ytheo Xexp Yexp XdevCalib YdevCalib pixDevCalib "
        columnsname += "2theta_theo chi_theo Energy PeakAmplitude Imax PeakBkg "
        columnsname += "PeakFwhm1 PeakFwhm2 PeakTilt XdevPeakFit YdevPeakFit"
        if nbcolumns_cor == 15 or nbcolumns_dat == 15:
            columnsname += " XfitEerr YfitErr"
        columnsname +="\n"

        meanresidues = np.mean(residues_calibFit)

        # IOLT.writefitfile(outputfilename,
        #                 data,
        #                 len(indExp),
        #                 dict_matrices=dict_matrices,
        #                 meanresidues=meanresidues,
        #                 PeakListFilename=initialdatfile,
        #                 columnsname=columnsname,
        #                 modulecaller="DetectorCalibration.py",
        #                 refinementtype="CCD Geometry")

        # wx.MessageBox("Fit results saved in %s" % outputfilename, "INFO")

        if self.parent:
            # update main GUI CCD geometrical parameters
            self.parent.defaultParam = self.CCDParam
            self.parent.pixelsize = self.pixelsize
            self.parent.kf_direction = self.kf_direction

        ##################
        dlg = wx.TextEntryDialog(self, "Enter File name with .fit extension: \n Folder: %s"%self.writefolder,"Saving refined peaks list (.fit) file")

        dlg.SetValue("%s" % filename)
        filenamefit = None
        if dlg.ShowModal() == wx.ID_OK:
            filenamefit = str(dlg.GetValue())
            fullpath = os.path.abspath(os.path.join(self.writefolder, filenamefit))

        dlg.Destroy()

        IOLT.writefitfile(fullpath,
                        data,
                        len(indExp),
                        dict_matrices=dict_matrices,
                        meanresidues=meanresidues,
                        PeakListFilename=initialfile,
                        columnsname=columnsname,
                        modulecaller="DetectorCalibration.py",
                        refinementtype="CCD Geometry")



        wx.MessageBox("Fit results saved in %s" % fullpath, "INFO")

    def show_alltogglestate(self, flag):
        if flag:
            # print "self.pointButton.GetValue()",self.pointButton.GetValue()
            print("self.btn_label_theospot.GetValue()", self.btn_label_theospot.GetValue())
            print("self.btn_label_expspot.GetValue()", self.btn_label_expspot.GetValue())

    def ToggleLabelExp(self, _):
        self.show_alltogglestate(0)

        if self.p2S == 0:
            self.btn_label_theospot.SetBackgroundColour("Green")
            self.btn_label_expspot.SetBackgroundColour(self.defaultColor)
            self.btn_label_expspot.SetValue(False)

            #             print "Disable Rotation around axis"
            self.SelectedRotationAxis = None
            self.moveccdandxtal.rotatebtn.SetLabel(self.moveccdandxtal.EnableRotationLabel)
            self.RotationActivated = False

            self.p2S = 1
            self.p3S = 0
        else:
            self.btn_label_theospot.SetBackgroundColour(self.defaultColor)
            self.btn_label_theospot.SetValue(False)

            self.p2S = 0

    def ToggleLabelSimul(self, _):
        self.show_alltogglestate(0)
        if self.p3S == 0:
            self.btn_label_expspot.SetBackgroundColour("Green")
            self.btn_label_theospot.SetBackgroundColour(self.defaultColor)
            self.btn_label_theospot.SetValue(False)

            #             print "Disable Rotation around axis"
            self.SelectedRotationAxis = None
            self.moveccdandxtal.rotatebtn.SetLabel(self.moveccdandxtal.EnableRotationLabel)
            self.RotationActivated = False

            self.p3S = 1
            self.p2S = 0
        else:
            self.btn_label_expspot.SetBackgroundColour(self.defaultColor)
            self.btn_label_expspot.SetValue(False)

            self.p3S = 0

    def display_current(self):
        """display current CCD parameters in txtctrls
        """
        self.parametersdisplaypanel.act_distance.SetValue(str(self.CCDParam[0]))
        self.parametersdisplaypanel.act_Xcen.SetValue(str(self.CCDParam[1]))
        self.parametersdisplaypanel.act_Ycen.SetValue(str(self.CCDParam[2]))
        self.parametersdisplaypanel.act_Ang1.SetValue(str(self.CCDParam[3]))
        self.parametersdisplaypanel.act_Ang2.SetValue(str(self.CCDParam[4]))

    def display_results(self, dataresults):
        """display CCD parameters refinement results in txtctrls
        """
        # for backreflection xbet = 0 is dangerous!
        xbet = float(dataresults[3])+0.0000001

        self.parametersdisplaypanel.act_distance_r.SetValue(str(dataresults[0]))
        self.parametersdisplaypanel.act_Xcen_r.SetValue(str(dataresults[1]))
        self.parametersdisplaypanel.act_Ycen_r.SetValue(str(dataresults[2]))
        self.parametersdisplaypanel.act_Ang1_r.SetValue(str(xbet))
        self.parametersdisplaypanel.act_Ang2_r.SetValue(str(dataresults[4]))
        self.act_residues.SetValue(str(np.round(dataresults[8], decimals=5)))
        self.nbspots_in_fit.SetValue(str(dataresults[9]))

    def close(self, _):
        self.Close(True)

    def OnSetCCDParams(self, event):
        """
        called by goto current button according to CCD parameters value
        """
        try:
            self.CCDParam = [float(self.parametersdisplaypanel.act_distance.GetValue()),
                            float(self.parametersdisplaypanel.act_Xcen.GetValue()),
                            float(self.parametersdisplaypanel.act_Ycen.GetValue()),
                            float(self.parametersdisplaypanel.act_Ang1.GetValue()),
                            float(self.parametersdisplaypanel.act_Ang2.GetValue())]
            print("Actual detector parameters are now default parameters", self.CCDParam)
            self.initialParameter["CCDParam"] = self.CCDParam

            self.update_data(event)
        except ValueError:
            dlg = wx.MessageDialog(self, "Detector Parameters in entry field are not float values! ",
                                    "Incorr",
                                    wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()

        self._replot(event)

    def OnInputParam(self, event):
        """
        in calibration frame
        """
        self.initialParameter["CCDParam"] = self.CCDParam
        self.initialParameter["pixelsize"] = self.pixelsize
        self.initialParameter["framedim"] = self.framedim
        self.initialParameter["kf_direction"] = self.kf_direction
        self.initialParameter["detectordiameter"] = self.detectordiameter

        print("before\n\n", self.initialParameter)

        DPBoard = DP.DetectorParameters(self, -1, "Detector parameters Board", self.initialParameter)

        DPBoard.ShowModal()
        DPBoard.Destroy()

        print("new param", self.CCDParam + [self.pixelsize,
                                            self.framedim[0],
                                            self.framedim[1],
                                            self.detectordiameter,
                                            self.kf_direction])

        self.display_current()
        self.update_data(event)

    #     def OnInputMatrix(self, event):
    #
    #         helptstr = 'Enter Matrix elements : \n [[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]]'
    #         helptstr += 'Or list of Matrices'
    #         dlg = wx.TextEntryDialog(self, helptstr, 'Calibration- Orientation Matrix elements Entry')
    #
    #         _param = '[[1, 0, 0],[0, 1, 0],[0, 0,1]]'
    #         dlg.SetValue(_param)
    #         if dlg.ShowModal() == wx.ID_OK:
    #             paramraw = str(dlg.GetValue())
    #             if paramraw != '1':  # neutral value ?
    #                 try:
    #                     paramlist = paramraw.split(',')
    #                     a11 = float(paramlist[0][2:])
    #                     a12 = float(paramlist[1])
    #                     a13 = float(paramlist[2][:-1])
    #                     a21 = float(paramlist[3][1:])
    #                     a22 = float(paramlist[4])
    #                     a23 = float(paramlist[5][:-1])
    #                     a31 = float(paramlist[6][1:])
    #                     a32 = float(paramlist[7])
    #                     a33 = float(paramlist[8][:-2])
    #
    #                     self.inputmatrix = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    #                     # may think about normalisation
    #
    #                     self.manualmatrixinput = 1
    #                     self.deltamatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #                     dlg.Destroy()
    #
    #                     self._replot(event)
    #                     self.display_current()
    #
    #                 except ValueError:
    #                     txt = "Unable to read the UBmatrix elements !!.\n"
    #                     txt += "There might be entered some strange characters in the entry field. Check it...\n"
    #                     wx.MessageBox(txt, 'INFO')
    #                     return

    def EnterMatrix(self, event):

        helptstr = "Enter Matrix elements : \n [[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]]"
        helptstr += "Or list of Matrices"
        dlg = wx.TextEntryDialog(self, helptstr, "Calibration- Orientation Matrix elements Entry")

        _param = "[[1, 0, 0],[0, 1, 0],[0, 0,1]]"
        dlg.SetValue(_param)
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

                _allm = np.array(ListMatrices[ind_matrix])
                if np.linalg.det(_allm)<0:
                    txt = "Matrix is not direct (det(UB)<0)"
                    print(txt)

                    wx.MessageBox(txt, "ERROR")
                    return
            # save in list of orientation matrix
            # default name
            inputmatrixname = "InputMat_"

            initlength = len(DictLT.dict_Rot)
            for k, mat in enumerate(ListMatrices):
                mname = inputmatrixname + "%d" % k
                DictLT.dict_Rot[mname] = mat
                self.crystalparampanel.comboMatrix.Append(mname)
            print("len dict", len(DictLT.dict_Rot))

            # or combo.Clear  combo.Appenditems(dict.rot)
            #             listrot = DictLT.dict_Rot.keys()
            #             sorted(listrot)
            #             self.crystalparampanel.comboMatrix.choices = listrot
            self.crystalparampanel.comboMatrix.SetSelection(initlength)
            #             self.crystalparampanel.comboMatrix.SetValue(inputmatrixname + '0')

            # update with the first input matrix
            self.inputmatrix = ListMatrices[0]
            # may think about normalisation

            self.manualmatrixinput = 1
            self.deltamatrix = np.eye(3)

            dlg.Destroy()

            self._replot(event)
            self.display_current()

    def OnChangeBMatrix(self, event):
        """
        Bmatrix selected in list
        """
        self._replot(event)
        self.display_current()

    def OnChangeMatrix(self, event):
        """
        UBmatrix selected in list
        """
        self.deltamatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.manualmatrixinput = 0
        self._replot(event)
        self.display_current()

    #     def OnChangeTransforms(self, event):
    #         """
    #         UBmatrix selected in list
    #         """
    #         self._replot(event)
    #         self.display_current()

    def OnChangeExtinc(self, event):
        self._replot(event)
        self.display_current()

    def OnChangeElement(self, event):
        key_material = self.crystalparampanel.comboElem.GetValue()
        self.sb.SetStatusText("Selected Material: %s" % str(DictLT.dict_Materials[key_material]))

        print("I change element to %s" % key_material)

        self.strainxtal.update_latticeparameters()
        # update extinctions rules
        extinc = DictLT.dict_Extinc_inv[DictLT.dict_Materials[key_material][2]]
        self.crystalparampanel.comboExtinctions.SetValue(extinc)
        self._replot(event)
        self.display_current()

    def update_data(self, event):
        """
        update experimental data according to CCD parameters
        and replot simulated data with _replot()
        """
        self.ReadExperimentData()

        print('In update_data():')
        # update theoretical data
        self._replot(event)
        self.display_current()

    # ---   --- Step Movements functions
    def OnDecreaseDistance(self, event):

        self.CCDParam[0] -= float(self.moveccdandxtal.stepdistance.GetValue())
        #         if self.CCDParam[0] < 20.:
        #             print "Distance seems too low..."
        #             self.CCDParam[0] = 20.
        self.update_data(event)

    def OnIncreaseDistance(self, event):

        self.CCDParam[0] += float(self.moveccdandxtal.stepdistance.GetValue())
        if self.CCDParam[0] > 200.0:
            print("Distance seems too high...")
            self.CCDParam[0] = 200.0
        self.update_data(event)

    def OnDecreaseXcen(self, event):

        self.CCDParam[1] -= float(self.moveccdandxtal.stepXcen.GetValue())
        if self.CCDParam[1] < -3000.0:
            print("Xcen seems too low...")
            self.CCDParam[1] = 3000
        self.update_data(event)

    def OnIncreaseXcen(self, event):

        self.CCDParam[1] += float(self.moveccdandxtal.stepXcen.GetValue())
        if self.CCDParam[1] > 6000.0:
            print("Xcen seems too high...")
            self.CCDParam[1] = 6000.0

        self.update_data(event)

    def OnDecreaseYcen(self, event):

        self.CCDParam[2] -= float(self.moveccdandxtal.stepYcen.GetValue())
        if self.CCDParam[2] < -3000.0:
            print("Ycen seems too low...")
            self.CCDParam[2] = -3000.0

        self.update_data(event)

    def OnIncreaseYcen(self, event):

        self.CCDParam[2] += float(self.moveccdandxtal.stepYcen.GetValue())
        if self.CCDParam[2] > 6000.0:
            print("Ycen seems too high...")
            self.CCDParam[2] = 6000.0

        self.update_data(event)

    def OnDecreaseang1(self, event):

        self.CCDParam[3] -= float(self.moveccdandxtal.stepang1.GetValue())
        #         if self.CCDParam[3] < -60.:
        #             print "Ang1 seems too low..."
        #             self.CCDParam[3] = -60.

        self.update_data(event)

    def OnIncreaseang1(self, event):

        self.CCDParam[3] += float(self.moveccdandxtal.stepang1.GetValue())
        self.update_data(event)

    def OnDecreaseang2(self, event):

        self.CCDParam[4] -= float(self.moveccdandxtal.stepang2.GetValue())
        self.update_data(event)

    def OnIncreaseang2(self, event):

        self.CCDParam[4] += float(self.moveccdandxtal.stepang2.GetValue())
        self.update_data(event)

    # incrementing or decrementing orientation elementary angles
    def OnDecreaseAngle1(self, event):
        """
        decrease angle1  (rotation around Y axis LaueTools)
        delta orientation angles around elementary axes"""
        # Xjsm =y Xmas  Yjsm = -Xxmas Zjsm = Zxmas
        a1 = float(self.moveccdandxtal.angle1.GetValue()) * DEG

        mat = np.array([[math.cos(a1), 0, -math.sin(a1)],
                        [0, 1, 0],
                        [math.sin(a1), 0, math.cos(a1)]])  # in XMAS and fitOrient
        self.deltamatrix = mat

        self._replot(event)
        self.display_current()

    def OnIncreaseAngle1(self, event):

        a1 = float(self.moveccdandxtal.angle1.GetValue()) * DEG
        mat = np.array([[math.cos(a1), 0, math.sin(a1)],
                        [0, 1, 0],
                        [-math.sin(a1), 0, math.cos(a1)]])  # in XMAS and fitOrient
        self.deltamatrix = mat

        self._replot(event)
        self.display_current()

    def OnDecreaseAngle2(self, event):
        """decrease angle1  (rotation around X axis LaueTools = incoming beam)"""
        a2 = float(self.moveccdandxtal.angle2.GetValue()) * DEG
        # mat = np.array([[math.cos(a2), 0, math.sin(-a2)],[0, 1, 0],[math.sin(a2), 0, math.cos(a2)]])  #in LaueTools Frame
        mat = np.array([[1, 0, 0],
                        [0, math.cos(a2), -math.sin(a2)],
                        [0, math.sin(a2), math.cos(a2)]])  # in XMAS and fitOrient
        self.deltamatrix = mat

        self._replot(event)
        self.display_current()

    def OnIncreaseAngle2(self, event):

        a2 = float(self.moveccdandxtal.angle2.GetValue()) * DEG
        # mat = np.array([[math.cos(a2), 0, math.sin(a2)],[0, 1, 0],[-math.sin(a2), 0, math.cos(a2)]]) in LaueTools Frame
        mat = np.array([[1, 0, 0],
                        [0, math.cos(a2), math.sin(a2)],
                        [0, math.sin(-a2), math.cos(a2)]])  # in XMAS and fitOrient
        self.deltamatrix = mat

        self._replot(event)
        self.display_current()

    def OnDecreaseAngle3(self, event):
        """decrease angle1  (rotation around Z vertical axis LaueTools) """
        a3 = float(self.moveccdandxtal.angle3.GetValue()) * DEG
        mat = np.array([[math.cos(a3), math.sin(a3), 0],
                        [math.sin(-a3), math.cos(a3), 0],
                        [0.0, 0, 1]])  # XMAS and LaueTools are similar
        self.deltamatrix = mat

        self._replot(event)
        self.display_current()

    def OnIncreaseAngle3(self, event):

        a3 = float(self.moveccdandxtal.angle3.GetValue()) * DEG
        mat = np.array([[math.cos(a3), -math.sin(a3), 0],
                        [math.sin(a3), math.cos(a3), 0],
                        [0, 0, 1]])
        self.deltamatrix = mat

        self._replot(event)
        self.display_current()

    def OnSwitchPlot(self, event):
        self.tog += 1

        if self.tog % 3 == 0:
            self.datatype = "2thetachi"
        elif self.tog % 3 == 1:
            self.datatype = "gnomon"
        elif self.tog % 3 == 2:
            self.datatype = "pixels"

        self.init_plot = True
        self._replot(event)
        self.display_current()

    def OnCheckEmaxValue(self, _):
        # emax = float(self.crystalparampanel.emaxC.GetValue())
        pass

    def OnCheckEminValue(self, _):
        # emin = float(self.crystalparampanel.eminC.GetValue())
        pass

    def define_kf_direction(self):
        """
        define main region of Laue Pattern simulation
        """
        #         print "Define mean region of simulation in MainCalibrationFrame"
        Central2Theta = float(self.plotrangepanel.mean2theta.GetValue())
        CentralChi = float(self.plotrangepanel.meanchi.GetValue())

        # reflection (camera top)
        if (Central2Theta, CentralChi) == (90, 0):
            self.kf_direction = "Z>0"
        # transmission
        elif (Central2Theta, CentralChi) == (0, 0):
            self.kf_direction = "X>0"
        # back reflection
        elif (Central2Theta, CentralChi) == (180, 0):
            self.kf_direction = "X<0"
        # reflection (camera side plus)
        elif (Central2Theta, CentralChi) == (90, 90):
            self.kf_direction = "Y>0"
        # reflection (camera side plus)
        elif (Central2Theta, CentralChi) == (90, -90):
            self.kf_direction = "Y<0"
        else:
            self.kf_direction = [Central2Theta, CentralChi]

        print("kf_direction chosen:", self.kf_direction)

    def onCenterOnhkl(self, evt):
        cppanel = self.crystalparampanel
        h = float(cppanel.tchc.GetValue())
        k = float(cppanel.tckc.GetValue())
        l = float(cppanel.tclc.GetValue())

        cppanel.UBmatrix = GT.propose_orientation_from_hkl([h, k, l], B0matrix=self.B0matrix)

        self._replot(evt)

    def onSetOrientMatrix_with_BMatrix(self, _):
        print("reset orientmatrix by integrating B matrix: OrientMatrix=OrientMatrix*B")
        self.crystalparampanel.UBmatrix = np.dot(self.crystalparampanel.UBmatrix, self.Bmatrix)

        self.crystalparampanel.comboBmatrix.SetValue("Identity")

    def simulate_theo(self, removeharmonics=0):
        """
        in MainCalibrationFrame

        Simulate theoretical Laue spots properties

        :param removeharmonics:  1  keep only lowest hkl (fondamental) for each harmonics spots family
                          0  consider all spots (fond. + harmonics)

        return:
        twicetheta, chi, self.Miller_ind, posx, posy, Energy
        """
        # print('entering simulate_theo() --------\n\n')
        # print('self.kf_direction', self.kf_direction)
        ResolutionAngstrom = None

        self.Extinctions = DictLT.dict_Extinc[self.crystalparampanel.comboExtinctions.GetValue()]

        # default
        # (deltamatrix can be updated step by step by buttons)
        if self.manualmatrixinput is None:
            self.crystalparampanel.UBmatrix = np.dot(
                self.deltamatrix, self.crystalparampanel.UBmatrix)
            # reset deltamatrix
            self.deltamatrix = np.eye(3)

        # from combobox of UBmatrix
        elif self.manualmatrixinput == 0:
            # self.UBmatrix = np.dot(self.deltamatrix, LaueToolsframe.dict_Rot[self.comboMatrix.GetValue()])
            self.crystalparampanel.UBmatrix = DictLT.dict_Rot[self.crystalparampanel.comboMatrix.GetValue()]
            # to keep self.UBmatrix unchanged at this step
            self.manualmatrixinput = None
            print('\nIn simulate_theo(): matrix label',self.crystalparampanel.comboMatrix.GetValue())
        # from manual input
        elif self.manualmatrixinput == 1:
            self.crystalparampanel.UBmatrix = self.inputmatrix
            self.manualmatrixinput = None

        pixelsize = self.pixelsize

        #self.define_kf_direction()

        self.emin = self.crystalparampanel.eminC.GetValue()
        self.emax = self.crystalparampanel.emaxC.GetValue()

        self.key_material = self.crystalparampanel.comboElem.GetValue()

        Grain = CP.Prepare_Grain(self.key_material, self.crystalparampanel.UBmatrix,
                                                    dictmaterials=self.dict_Materials)

        self.B0matrix = Grain[0]

        Bmatrix_key = str(self.crystalparampanel.comboBmatrix.GetValue())
        self.Bmatrix = DictLT.dict_Transforms[Bmatrix_key]

        Grain[2] = np.dot(Grain[2], self.Bmatrix)

        if self.CCDLabel.startswith("sCMOS"):  # squared detector
            diameter_for_simulation = self.detectordiameter * 1.4 * 1.25
        else:
            diameter_for_simulation = self.detectordiameter

        # xbet = 0 is dangerous for back reflection geometry (issue with calculation of pixel X, Y position)!
        if math.fabs(self.CCDParam[3]) <= 1e-7:
            self.CCDParam[3] += 0.000000000001

        SINGLEGRAIN = 1
        if SINGLEGRAIN:  # for single grain simulation
            if self.kf_direction in ("Z>0", "X>0", 'X<0') and removeharmonics == 0:
                # for single grain simulation (WITH HARMONICS   TROUBLE with TRansmission geometry)
                #print('SINGLEGRAIN')
                #print('parameters for SimulateLaue_full_np', self.CCDParam[:5], self.kf_direction, removeharmonics,pixelsize, self.framedim)
                ResSimul = LAUE.SimulateLaue_full_np(Grain,
                                                    self.emin,
                                                    self.emax,
                                                    self.CCDParam[:5],
                                                    kf_direction=self.kf_direction,
                                                    ResolutionAngstrom=False,
                                                    removeharmonics=removeharmonics,
                                                    pixelsize=pixelsize,
                                                    dim=self.framedim,
                                                    detectordiameter=diameter_for_simulation * 1.25,
                                                    force_extinction=self.Extinctions,
                                                    dictmaterials=self.dict_Materials)
            else:  # for autolinks (removeharmonics=1)
                ResSimul = LAUE.SimulateLaue(Grain,
                                            self.emin,
                                            self.emax,
                                            self.CCDParam[:5],
                                            kf_direction=self.kf_direction,
                                            ResolutionAngstrom=ResolutionAngstrom,
                                            removeharmonics=removeharmonics,
                                            pixelsize=pixelsize,
                                            dim=self.framedim,
                                            detectordiameter=diameter_for_simulation * 1.25,
                                            force_extinction=self.Extinctions,
                                            dictmaterials=self.dict_Materials)


            if ResSimul is None:
                return None

            (twicetheta, chi, self.Miller_ind, posx, posy, Energy) = ResSimul
            # print('twicetheta[:5], chi[:5]', twicetheta[:5], chi[:5])
            # print('posx[:5], posy[:5]', posx[:5], posy[:5])
            # print('min max posx, min max posy', np.amin(posx), np.amax(posx),np.amin(posy), np.amax(posy))

            # sort data by increasing twicetheta and print only
            TEST_ONLY = False
            if TEST_ONLY:
                cond = np.logical_and(twicetheta>80,twicetheta<82)
                ix_in = np.where(cond)[0]
                sorted_ix = np.argsort(twicetheta[ix_in])
                select_ix = ix_in[sorted_ix]

                s_twtheta = twicetheta[select_ix]
                s_posx = posx[select_ix]
                s_posy = posy[select_ix]
                s_Energy = Energy[select_ix]
                print('sorted data by increasing 2theta')
                print(np.array([s_twtheta, s_posx,s_posy,s_Energy]).T)

        else:
            # for twinned grains simulation
            print("---------------------------")
            print("Twins simulation mode")
            print("---------------------------")

            Grainparent = Grain
            twins_operators = [DictLT.dict_Transforms["twin010"]]
            #             twins_operators = [DictLT.dict_Transforms['sigma3_1']]

            #             axisrot = [np.cos((103.68 - 90) * DEG), 0, np.sin((103.68 - 90) * DEG)]
            #             rot180 = GT.matRot(axisrot, 180.)
            #             twins_operators = [rot180]

            (twicetheta,
                chi,
                self.Miller_ind,
                posx,
                posy,
                Energy) = LAUE.SimulateLaue_twins(Grainparent,
                                                twins_operators,
                                                self.emin,
                                                self.emax,
                                                self.CCDParam[:5],
                                                only_2thetachi=False,
                                                kf_direction=self.kf_direction,
                                                ResolutionAngstrom=False,
                                                removeharmonics=1,
                                                pixelsize=pixelsize,
                                                dim=self.framedim,
                                                detectordiameter=diameter_for_simulation * 1.25)

            print("nb of spots", len(twicetheta))

        #print('End of simulate_theo() ------------\n\n')

        return twicetheta, chi, self.Miller_ind, posx, posy, Energy

    def OnChangeOrigin(self, _):
        
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        # print('ylim',ylim)

        self.axes.set_ylim(ylim[1],ylim[0])
        self._replot(1)
    
    def _replot(self, _):
        """
        in MainCalibrationFrame
        """
        # simulate theo data
        ResSimul = self.simulate_theo()  # twicetheta, chi, self.Miller_ind, posx, posy
        if ResSimul is None:
            self.deltamatrix = np.eye(3)
            print("reset deltamatrix to identity")
            return

        self.data_theo = ResSimul

        # offsets to match imshow and scatter plot coordinates frames
        if self.datatype == "pixels":
            X_offset = 1
            Y_offset = 1
        else:
            X_offset = 0
            Y_offset = 0

        if not self.init_plot:
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()

        self.axes.clear()
        self.axes.set_autoscale_on(False)  # Otherwise, infinite loop
        #         self.axes.set_autoscale_on(True)

        # to have the data coordinates when pointing with the mouse
        def fromindex_to_pixelpos_x(index, _):
            return index

        def fromindex_to_pixelpos_y(index, _):
            return index

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))

        # plot THEORETICAL SPOTS simulated data ------------------------------------------
        if self.data_theo is not None:  # only in 2theta, chi space
            # self.axes.scatter(self.data_theo[0], self.data_theo[1],s=50, marker='o',alpha=0, edgecolor='r',c='w')

            # laue spot model intensity 2
            Energy = self.data_theo[5]

            Polariz = (1 - (np.sin(self.data_theo[0] * DEG) * np.sin(self.data_theo[1] * DEG))** 2)
            #
            #             sizespot = 150 * np.exp(-Energy * 1. / 10.)  # * Polariz
            #             print "len(Polariz)", len(Polariz)
            #             print "Energy", Energy
            #             print "sizespot", sizespot
            #             print 'len(np.array(self.data_theo[2]))', len(np.array(self.data_theo[2]))
            Fsquare = 50.0 / np.sum(np.array(self.data_theo[2]) ** 2, axis=1)

            #             print "Fsquare", Fsquare[:5]

            sizespot = (100 * GT.CCDintensitymodel2(Energy) * Fsquare * Polariz
                * float(self.plotrangepanel.spotsizefactor.GetValue()))

            #             print "Polariz", Polariz
            #             print "Fsquare", Fsquare

            if self.datatype == "2thetachi":
                # dependent of matplotlib and OS see pickyframe...
                # self.axes.scatter(self.data_theo[0], self.data_theo[1],s=sizespot, marker='o',alpha=0, edgecolor='r',c='w')  # don't work with linux and matplotlib 0.99.1
                # self.axes.scatter(self.data_theo[0], self.data_theo[1],s = sizespot, marker='o',alpha=0, edgecolor='r',c='w')
                self.axes.scatter(self.data_theo[0],
                                    self.data_theo[1],
                                    s=sizespot,
                                    marker="o",
                                    edgecolor="r",
                                    facecolor="None")

            elif self.datatype == "gnomon":
                # compute Gnomonic projection
                nbofspots = len(self.data_theo[0])
                sim_dataselected = IOLT.createselecteddata(
                                        (self.data_theo[0], self.data_theo[1], np.ones(nbofspots)),
                                        np.arange(nbofspots), nbofspots)[0]
                self.sim_gnomonx, self.sim_gnomony = IIM.ComputeGnomon_2(sim_dataselected)
                self.axes.scatter(self.sim_gnomonx,
                                    self.sim_gnomony,
                                    s=sizespot,
                                    marker="o",
                                    edgecolor="r",
                                    facecolor="None")

            elif self.datatype == "pixels":
                # dependent of matplotlib and OS see pickyframe...
                # self.axes.scatter(self.data_theo[0], self.data_theo[1],s=sizespot, marker='o',alpha=0, edgecolor='r',c='w')  # don't work with linux and matplotlib 0.99.1
                # self.axes.scatter(self.data_theo[0], self.data_theo[1],s=sizespot, marker='o',alpha=0, edgecolor='r',c='w')
                self.axes.scatter(self.data_theo[3],
                                    self.data_theo[4],
                                    s=sizespot,
                                    marker="o",
                                    edgecolor="r",
                                    facecolor="None")

        # plot EXPERIMENTAL data ----------------------------------------
        if self.datatype == "2thetachi":
            originChi = 0

            if self.plotrangepanel.shiftChiOrigin.GetValue():
                originChi = float(self.plotrangepanel.meanchi.GetValue())

            self.axes.scatter(self.twicetheta,
                                self.chi + originChi,
                                s=self.Data_I / np.amax(self.Data_I) * 100.0,
                                c=self.Data_I / 50.0,
                                alpha=0.5)

            if self.init_plot:
                amp2theta = float(self.plotrangepanel.range2theta.GetValue())
                mean2theta = float(self.plotrangepanel.mean2theta.GetValue())

                ampchi = float(self.plotrangepanel.rangechi.GetValue())
                meanchi = float(self.plotrangepanel.meanchi.GetValue())

                min2theta = max(0, mean2theta - amp2theta)
                max2theta = min(180.0, mean2theta + amp2theta)

                minchi = max(-180, meanchi - ampchi)
                maxchi = min(180.0, meanchi + ampchi)

                mean2theta = 0.5 * (min2theta + max2theta)
                halfampli2theta = 0.5 * (max2theta - min2theta)

                meanchi = 0.5 * (maxchi + minchi)
                halfamplichi = 0.5 * (maxchi - minchi)

                xlim = (mean2theta - halfampli2theta, mean2theta + halfampli2theta)
                ylim = (meanchi - halfamplichi, meanchi + halfamplichi)

            self.axes.set_xlabel("2theta(deg.)")
            self.axes.set_ylabel("chi(deg)")

        elif self.datatype == "gnomon":

            self.data_gnomonx, self.data_gnomony = self.computeGnomonicExpData()

            self.axes.scatter(self.data_gnomonx,
                                self.data_gnomony,
                                s=self.Data_I / np.amax(self.Data_I) * 100.0,
                                c=self.Data_I / 50.0,
                                alpha=0.5)

            if self.init_plot:
                xmin = np.amin(self.data_gnomonx) - 0.1
                xmax = np.amax(self.data_gnomonx) + 0.1
                ymin = np.amin(self.data_gnomony) - 0.1
                ymax = np.amax(self.data_gnomony) + 0.1

                ylim = (ymin, ymax)
                xlim = (xmin, xmax)

            self.axes.set_xlabel("X gnomon")
            self.axes.set_ylabel("Y gnomon")

        elif self.datatype == "pixels":
            self.axes.scatter(self.data_x - X_offset,
                            self.data_y - Y_offset,
                            s=self.Data_I / np.amax(self.Data_I) * 100.0,
                            c=self.Data_I / 50.0,
                            alpha=0.5)
            if self.init_plot:
                ylim = (-100, self.framedim[0] + 100)
                xlim = (-100, self.framedim[1] + 100)
            self.axes.set_xlabel("X CCD")
            self.axes.set_ylabel("Y CCD")

        # ---------------------------------------------------------------
        # plot experimental spots linked to 1 theo. spot)
        # ---------------------------------------------------------------
        if self.plotlinks is not None:
            exp_indices = np.array(np.array(self.plotlinks)[:, 0], dtype=np.int16)
            #print('exp_indices in yellow links plot ',exp_indices)

            # experimental spots selection -------------------------------------
            if self.datatype == "2thetachi":
                Xlink = np.take(self.twicetheta, exp_indices)
                Ylink = np.take(self.chi, exp_indices)
            elif self.datatype == "pixels":
                pixX = np.take(self.data_x, exp_indices)
                pixY = np.take(self.data_y, exp_indices)

                Xlink = pixX - X_offset
                Ylink = pixY - Y_offset

            self.axes.scatter(Xlink, Ylink, s=100., alpha=0.5, c='yellow')

        self.axes.set_title("%s %d spots" % (os.path.split(self.filename)[-1], len(self.twicetheta)))
        self.axes.grid(True)

        # restore the zoom limits(unless they're for an empty plot)
        if xlim != (0.0, 1.0) or ylim != (0.0, 1.0):
            # print('xlim, ylim', xlim, ylim)
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)

        self.init_plot = False

        # redraw the display
        self.canvas.draw()

    def SelectOnePoint(self, event):
        """
        in MainCalibrationFrame
        """
        toreturn = []
        self.successfull = 0

        if self.nbsuccess == 0:
            self.EXPpoints = []

        xtol = 20
        ytol = 20.0
        """
        self.twicetheta, self.chi, self.Data_I, self.filename = self.data
        self.Data_index_expspot = np.arange(len(self.twicetheta))
        """
        xdata, ydata, annotes = (self.twicetheta, self.chi,
                                    list(zip(self.Data_index_expspot, self.Data_I)))

        _dataANNOTE_exp = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        # print clickX, clickY

        annotes = []
        for x, y, a in _dataANNOTE_exp:
            if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]
            # print "the nearest experimental point is at(%.2f,%.2f)"%(x, y)
            # print "with index %d and intensity %.1f"%(annote[0],annote[1])

            self.EXPpoints.append([annote[0], x, y])
            self.successfull = 1
            self.nbsuccess += 1
            print("# selected points", self.nbsuccess)
            # print "Coordinates(%.3f,%.3f)"%(x, y)

            toreturn = self.EXPpoints

        return toreturn

    def SelectThreePoints(self, event):
        """
        in MainCalibrationFrame
        """
        toreturn = []
        if self.nbclick_zone <= 3:
            if self.nbclick_zone == 1:
                self.threepoints = []
            xtol = 0.5
            ytol = 0.5
            """
            self.twicetheta, self.chi, self.Data_I, self.filename = self.data
            self.Data_index_expspot = np.arange(len(self.twicetheta))
            """
            xdata, ydata, annotes = (self.twicetheta, self.chi,
                                        list(zip(self.Data_index_expspot, self.Data_I)))

            _dataANNOTE_exp = list(zip(xdata, ydata, annotes))

            clickX = event.xdata
            clickY = event.ydata
            # print clickX, clickY
            annotes = []
            for x, y, a in _dataANNOTE_exp:
                if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                    annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

            if annotes:
                annotes.sort()
                _distance, x, y, annote = annotes[0]
                # print "the nearest experimental point is at(%.2f,%.2f)"%(x, y)
                # print "with index %d and intensity %.1f"%(annote[0],annote[1])

            self.threepoints.append([annote[0], x, y])
            print("# selected points", self.nbclick_zone)
            # print "Coordinates(%.3f,%.3f)"%(x, y)
            if len(self.threepoints) == 3:
                toreturn = self.threepoints
                self.nbclick_zone = 0
                print("final triplet", toreturn)

        self.nbclick_zone += 1
        self._replot(event)
        return toreturn

    def SelectSixPoints(self, event):
        """
        in MainCalibrationFrame
        """
        toreturn = []
        if self.nbclick_zone <= 6:
            if self.nbclick_zone == 1:
                self.sixpoints = []
            xtol = 2.0
            ytol = 2.0
            """
            self.twicetheta, self.chi, self.Data_I, self.filename = self.data
            self.Data_index_expspot = np.arange(len(self.twicetheta))
            """
            xdata, ydata, annotes = (self.twicetheta, self.chi,
                                                list(zip(self.Data_index_expspot, self.Data_I)))

            _dataANNOTE_exp = list(zip(xdata, ydata, annotes))

            clickX = event.xdata
            clickY = event.ydata

            # print clickX, clickY

            annotes = []
            for x, y, a in _dataANNOTE_exp:
                if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                    annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

            print("# selected points", self.nbclick_zone)
            if annotes:
                annotes.sort()
                _distance, x, y, annote = annotes[0]
                print("the nearest experimental point is at(%.2f,%.2f)" % (x, y))
                print("with index %d and intensity %.1f" % (annote[0], annote[1]))

            self.sixpoints.append([annote[0], x, y])

            if len(self.sixpoints) == 6:
                toreturn = self.sixpoints
                self.nbclick_zone = 0
                print("final six points", toreturn)

        self.nbclick_zone += 1
        self._replot(event)
        return toreturn

    def OnSelectZoneAxes(self, event):
        """
        in MainCalibrationFrame
        TODO: not implemented yet
        """
        pass

    def textentry(self):
        """
        in MainCalibrationFrame
        TODO: use better SetDetectorParam() of frame
        """
        dlg = wx.TextEntryDialog(self, "Enter Miller indices: [h, k,l]", "Miller indices entry")
        dlg.SetValue("[0, 0,1]")
        if dlg.ShowModal() == wx.ID_OK:
            miller = dlg.GetValue()
        dlg.Destroy()
        Miller = np.array(np.array(miller[1:-1].split(",")), dtype=int)
        return Miller

    def OnInputMiller(self, evt):
        """
        user selects TWO exp and gives corresponding miller indices
        comparison of theo and exp; distances.
        VERY usefull if the reference sample is well known
        """

        pts = self.SelectOnePoint(evt)

        if self.nbsuccess == 1:
            index1, X1, Y1 = pts[0]
            print("selected # exp.spot:", index1, " @(%.3f,%.3f)" % (X1, Y1))
            self.twopoints = [[X1, Y1]]

        if self.nbsuccess == 2:
            index1, X1, Y1 = pts[0]
            index2, X2, Y2 = pts[1]
            print("selected # exp.spot:", index2, " @(%.3f,%.3f)" % (X2, Y2))

            self.twopoints.append([X2, Y2])

            mil1 = self.textentry()
            print(mil1)
            mil2 = self.textentry()
            print(mil2)
            tdist = (np.arccos(np.dot(mil1, mil2) / np.sqrt(np.dot(mil1, mil1) * np.dot(mil2, mil2)))
                * 180.0 / np.pi)
            print("Theoretical distance", tdist)
            _dist = GT.distfrom2thetachi(np.array(self.twopoints[0]), np.array(self.twopoints[1]))
            print("Experimental distance: %.3f deg " % _dist)
            if _dist < 0.0000001:
                print("You may have selected the same theoretical spot ... So the distance is 0!")

            self.nbsuccess = 0

            wx.MessageBox("selected # exp.spot:%d @(%.3f ,%.3f)\nselected # exp.spot:%d @(%.3f ,%.3f)\nTheoretical distance %.3f\nExperimental distance %.3f"
                % (index1, X1, Y1, index2, X2, Y2, tdist, _dist), "Results")

    def allbuttons_off(self):
        """
        in MainCalibrationFrame
        """
        # self.pointButton.SetValue(False)
        self.btn_label_theospot.SetValue(False)
        self.btn_label_expspot.SetValue(False)

    def readlogicalbuttons(self):
        # return [self.pointButton.GetValue(),
        # self.btn_label_theospot.GetValue(),
        # self.btn_label_expspot.GetValue(),
        # self.pointButton3.GetValue()]

        return [self.btn_label_theospot.GetValue(), self.btn_label_expspot.GetValue()]

    def _on_point_choice(self, evt):
        """
        TODO: remove! obsolete
        in MainCalibrationFrame
        """
        if self.readlogicalbuttons() == [True, False]:
            self.allbuttons_off()
            self.btn_label_theospot.SetValue(True)
            self.Annotate_exp(evt)
        if self.readlogicalbuttons() == [False, True]:
            self.allbuttons_off()
            self.btn_label_expspot.SetValue(True)
            self.Annotate_theo(evt)

    def select_2pts(self, evt):  # pick distance
        """
        in MainCalibrationFrame
        """
        toreturn = []
        if self.nbclick_dist <= 2:
            if self.nbclick_dist == 1:
                self.twopoints = []

            self.twopoints.append([evt.xdata, evt.ydata])
            print("# selected points", self.nbclick_dist)
            print("Coordinates(%.3f,%.3f)" % (evt.xdata, evt.ydata))
            print("click", self.nbclick_dist)

            if len(self.twopoints) == 2:
                # compute angular distance:
                spot1 = self.twopoints[0]  # (X, Y) (e.g. 2theta, chi)
                spot2 = self.twopoints[1]
                if self.datatype == "2thetachi":
                    _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))
                    print("angular distance :  %.3f deg " % _dist)
                if self.datatype == "gnomon":
                    tw, ch = IIM.Fromgnomon_to_2thetachi(
                        [np.array([spot1[0], spot2[0]]),
                            np.array([spot1[1], spot2[1]])],
                        0,)[:2]
                    _dist = GT.distfrom2thetachi(np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]]))
                    print("angular distance :  %.3f deg " % _dist)
                toreturn = self.twopoints
                self.nbclick_dist = 0
                # self.twopoints = []
                self.btn_label_theospot.SetValue(False)

        self.nbclick_dist += 1
        self._replot(evt)
        return toreturn

    def Reckon_2pts(self, evt):  # Recognise distance
        """
        in MainCalibrationFrame
        .. todo::
            May be useful to integrate back to the calibration board
        """
        twospots = self.select_2pts(evt)

        if twospots:
            print("twospots", twospots)
            spot1 = twospots[0]
            spot2 = twospots[1]
            print("---Selected points")

            if self.datatype == "2thetachi":
                _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))
                print("(2theta, chi) ")

            elif self.datatype == "gnomon":
                tw, ch = IIM.Fromgnomon_to_2thetachi([np.array([spot1[0], spot2[0]]),
                                                np.array([spot1[1], spot2[1]])], 0)[:2]
                _dist = GT.distfrom2thetachi(np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]]))
                spot1 = [tw[0], ch[0]]
                spot2 = [tw[1], ch[1]]

            print("spot1 [%.3f,%.3f]" % (tuple(spot1)))
            print("spot2 [%.3f,%.3f]" % (tuple(spot2)))
            print("angular distance :  %.3f deg " % _dist)

            # distance recognition -------------------------
            ang_tol = 2.0
            # residues matching angle -------------------------
            ang_match = 5.0

            ind_sorted_LUT_MAIN_CUBIC = [np.argsort(elem) for elem in FindO.LUT_MAIN_CUBIC]
            sorted_table_angle = []
            for k in range(len(ind_sorted_LUT_MAIN_CUBIC)):
                # print len(LUT_MAIN_CUBIC[k])
                # print len(ind_sorted_LUT_MAIN_CUBIC[k])
                sorted_table_angle.append((FindO.LUT_MAIN_CUBIC[k])[ind_sorted_LUT_MAIN_CUBIC[k]])

            sol = INDEX.twospots_recognition([spot1[0] / 2.0, spot1[1]],
                                                [spot2[0] / 2.0, spot2[1]], ang_tol)
            print("sol = ", sol)

            print("\n")
            print("---Planes Recognition---")
            if isinstance(sol, np.ndarray):
                print("planes found ------ for angle %.3f within %.2f deg"% (_dist, ang_tol))
                print("spot 1          spot 2           theo. value(deg)")
                for k in range(len(sol[0])):
                    theodist = (np.arccos(np.dot(sol[0][k], sol[1][k])
                            / np.sqrt(np.dot(sol[0][k], sol[0][k])* np.dot(sol[1][k], sol[1][k])
                            )) * 180.0 / np.pi)
                    # print sol[0][k]
                    # print sol[1][k]
                    print(" %s          %s           %.3f" % (str(sol[0][k]), str(sol[1][k]), theodist))

                res = []
                self.mat_solution = [[] for k in range(len(sol[0]))]
                self.TwicethetaChi_solution = [[] for k in range(len(sol[0]))]

                print("datatype", self.datatype)

                for k in range(len(sol[0])):
                    mymat = FindO.givematorient(sol[0][k], spot1, sol[1][k], spot2, verbose=0)
                    self.mat_solution[k] = mymat
                    emax = 25
                    emin = 5
                    vecteurref = np.eye(3)  # means: a* // X, b* // Y, c* //Z
                    grain = [vecteurref, [1, 1, 1], mymat, "Cu"]

                    # PATCH: redefinition of grain to simulate any unit cell(not only cubic) ---
                    key_material = grain[3]
                    grain = CP.Prepare_Grain(key_material, grain[2], dictmaterials=self.dict_Materials)
                    # -----------------------------------------------------------------------------

                    # array(vec) and array(indices)(here with fastcompute = 0 array(indices) = 0) of spots exiting the crystal in 2pi steradian(Z>0)
                    spots2pi = LAUE.getLaueSpots(DictLT.CST_ENERGYKEV / emax,
                                                DictLT.CST_ENERGYKEV / emin,
                                                [grain],
                                                fastcompute=1,
                                                verbose=0,
                                                dictmaterials=self.dict_Materials)
                    # 2theta, chi of spot which are on camera(with harmonics)
                    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)
                    self.TwicethetaChi_solution[k] = TwicethetaChi

                    if self.datatype == "2thetachi":
                        tout = matchingrate.getProximity(TwicethetaChi,
                                                        np.array(self.data[0]) / 2.0,
                                                        np.array(self.data[1]),
                                                        angtol=ang_match)
                    elif self.datatype == "gnomon":
                        # print "self.data in reckon 2pts",self.data[0][:10]
                        TW, CH = IIM.Fromgnomon_to_2thetachi(self.data[:2], 0)[:2]
                        # print "TW in reckon 2pst",TW[:10]
                        # LaueToolsframe.control.SetValue(str(array(TW, dtype = '|S8'))+'\n'+str(array(CH, dtype = '|S8')))
                        tout = matchingrate.getProximity(TwicethetaChi,
                                                        np.array(TW) / 2.0,
                                                        np.array(CH),
                                                        angtol=ang_match)

                    # print "calcul residues",tout[2:]
                    # print mymat
                    # print "tout de tout",tout
                    res.append(tout[2:])

                # Display results
                if self.datatype == "gnomon":
                    self.data_fromGnomon = (TW, CH, self.Data_I, self.filename)
                    self.RecBox = RecognitionResultCheckBox(
                        self, -1, "Potential solutions", res, self.data_fromGnomon, emax=emax)
                    self.RecBox = str(self.crystalparampanel.comboElem.GetValue())
                    self.RecBox.TwicethetaChi_solution = self.TwicethetaChi_solution
                    self.RecBox.mat_solution = self.mat_solution
                else:  # default 2theta, chi
                    self.RecBox = RecognitionResultCheckBox(
                        self, -1, "Potential solutions", res, self.data, emax=emax)
                    self.RecBox = str(self.crystalparampanel.comboElem.GetValue())
                    self.RecBox.TwicethetaChi_solution = self.TwicethetaChi_solution
                    self.RecBox.mat_solution = self.mat_solution
                print("result", res)
                self.recognition_possible = False

            elif sol == []:
                print("Sorry! No planes found for this angle within angular tolerance %.2f"% ang_tol)
                print("Try to: increase the angular tolerance or be more accurate in clicking!")
                print("Try to extend the number of possible planes probed in recognition, ask the programmer!")
            # distance recognition -------------------------

            self._replot(evt)

    # ---  --- plot Annotations ------------
    def OnResetAnnotations(self, evt):
        self.drawnAnnotations_exp = {}
        self.drawnAnnotations_theo = {}
        self._replot(evt)

    def onKeyPressed(self, event):
        key = event.key
        #         print "key", key
        if key == "escape":
            ret = wx.MessageBox("Are you sure to quit?", "Question", wx.YES_NO | wx.NO_DEFAULT, self)

            if ret == wx.YES:
                self.Close()

        elif key in ("+", "-"):
            angle = float(self.moveccdandxtal.stepanglerot.GetValue())
            if key == "+":
                self.RotateAroundAxis(angle)
            if key == "-":
                self.RotateAroundAxis(-angle)

    def onClick(self, event):
        """ onclick with mouse
        """
        if event.inaxes:

            if event.button == 1:
                self.centerx, self.centery = event.xdata, event.ydata


                if self.datatype=='pixels':
                    tw, chi = self.convertpixels2twotheta(event.xdata, event.ydata)
                    print('X,Y',event.xdata, event.ydata)
                    print('2theta, chi', tw, chi)

            # rotation  around self.centerx, self.centery triggered by button and NOT MOUSE
            if self.RotationActivated:
                # axis is already defined
                if self.SelectedRotationAxis is not None:
                    print("Rotation possible around : ", self.SelectedRotationAxis)
                #                     self.RotateAroundAxis()
                # axis must be defined
                else:
                    self.SelectedRotationAxis = self.selectrotationaxis(event.xdata, event.ydata)

            elif self.toolbar.mode != "":
                print("You clicked on something, but toolbar is in mode %s."% str(self.toolbar.mode))

            elif self.btn_label_theospot.GetValue():
                self.Annotate_exp(event)

            elif self.btn_label_expspot.GetValue():
                self.Annotate_theo(event)

            elif self.toolbar.mode == "":  # dragging and 'rotating' laue pattern
                self.press = event.xdata, event.ydata

    def onRelease(self, event):
        """handle the release of button 1 for rotation axis selection"""

        # need to have previously clicked on a point
        #print('Release  button', event.button)
        if self.press is None:
            #print('self.press is None when Release')
            return

        if event.button == 1:
            #print('event.button == 1 in onrelease()')
            self.centerx, self.centery = self.press

            # define rotation axis from self.centerx, self.centery that must be 2theta and chi angles
            self.SelectedRotationAxis = self.selectrotationaxis(self.centerx, self.centery)
            self._replot(event)

        #print('self.press = None in onrelease\n\n')
        self.press = None


    def onMotion_ToolTip(self, event):
        """tool tip to show data (exp. and theo. spots) when mouse hovers on plot
        """
        
        if len(self.data[0]) == 0:
            return

        collisionFound_exp = False
        collisionFound_theo = False

        if self.datatype == "2thetachi":
            xtol = 5
            ytol = 5
        elif self.datatype == "pixels":
            xtol = 50
            ytol = 50

        # twicetheta, chi, self.Miller_ind, posx, posy, Energy = self.data_theo
        if self.datatype == "2thetachi":
            xdata, ydata, _annotes_exp = (self.data[0],
                                        self.data[1],
                                        list(zip(self.Data_index_expspot, self.Data_I)))
            
            xdata_theo, ydata_theo, _annotes_theo = (self.data_theo[0],
                                                self.data_theo[1],
                                                list(zip(*self.data_theo[2:])))
        elif self.datatype == "pixels":
            xdata, ydata, _annotes_exp = (self.data_XY[0],
                                        self.data_XY[1],
                                        list(zip(self.Data_index_expspot, self.Data_I)))

            xdata_theo, ydata_theo, _annotes_theo = (self.data_theo[3],
                                                self.data_theo[4],
                                                list(zip(*self.data_theo[2:])))

        if event.xdata != None and event.ydata != None:

            evx, evy = event.xdata, event.ydata

            if self.datatype == "pixels":
                tip = "current (X,Y)=(%.2f,%.2f)"%(evx, evy)
            if self.datatype == "2thetachi":
                tip = "current (2theta,chi)=(%.2f,%.2f)"%(evx, evy)

            annotes_exp = []
            for x, y, aexp in zip(xdata, ydata, _annotes_exp):
                if (evx - xtol < x < evx + xtol) and (evy - ytol < y < evy + ytol):
                    #print("got exp. spot!! at x,y", x, y)
                    annotes_exp.append((GT.cartesiandistance(x, evx, y, evy), x, y, aexp))

            annotes_theo = []
            for x, y, atheo in zip(xdata_theo, ydata_theo, _annotes_theo):
                if (evx - xtol < x < evx + xtol) and (evy - ytol < y < evy + ytol):
                    #print("got theo. spot!!")
                    #                     print "with info: ", atheo
                    annotes_theo.append((GT.cartesiandistance(x, evx, y, evy), x, y, atheo))

            if annotes_exp != []:
                collisionFound_exp = True
            if annotes_theo != []:
                collisionFound_theo = True

            if not collisionFound_exp and not collisionFound_theo:
                self.tooltip.SetTip(tip)
                return

            tip_exp = ""
            tip_theo = ""
            if self.datatype == "2thetachi":
                closedistance = 3
            elif self.datatype == "pixels":
                closedistance = 10

            if collisionFound_exp:
                annotes_exp.sort()
                _distanceexp, x, y, annote_exp = annotes_exp[0]

                # if exp. spot is close enough
                if _distanceexp < closedistance:
                    tip_exp = "Closest Exp. spot: index=%d Intensity=%.1f at (%.2f, %.2f) " % (annote_exp[0], annote_exp[1],x,y)
                    print('Closest Exp. found ->  at (%.2f,%.2f)'% (x, y), tip_exp)
                    self.updateStatusBar(x, y, annote_exp, spottype="exp")

                    self.highlightexpspot = annote_exp[0]
                else:
                    self.sb.SetStatusText("",0)
                    tip_exp = ""
                    collisionFound_exp = False
                    self.highlightexpspot = None

            if collisionFound_theo:
                try:
                    annotes_theo.sort()
                    txtharmonics = ''
                except ValueError:
                    txtharmonics = f'Peak with {len(annotes_theo)} harmonics !'
                    # print(txtharmonics)

                _distancetheo, x, y, annote_theo = annotes_theo[0]

                # if theo spot is close enough
                if _distancetheo < closedistance:
                    # print("\nthe nearest theo point is at(%.2f,%.2f)" % (x, y))
                    # print("with info (hkl, other coordinates, energy)", annote_theo)

                    tip_theo = "Theo. [h k l]=%s Energy=%.2f keV" % (str(annote_theo[0]), annote_theo[3])
                    if self.datatype == "pixels":
                        tip_theo += "\nTheo. (X,Y)=(%.2f,%.2f) (2theta,Chi)=(%.2f,%.2f)" % (
                            x, y, annote_theo[1], annote_theo[2])
                    if self.datatype == "2thetachi":
                        tip_theo += "\nTheo. (X,Y)=(%.2f,%.2f) (2theta,Chi)=(%.2f,%.2f)" % (
                            annote_theo[1], annote_theo[2], x, y)

                    tip_theo += txtharmonics

                    self.updateStatusBar(x, y, annote_theo, spottype="theo")

                    # find theo spot index
                    hkl0 = annote_theo[0]
                    #print('hkl0',hkl0)
                    hkls = self.data_theo[2]
                    theoindex = np.where(np.sum(np.hypot(hkls - hkl0, 0), axis=1) < 0.01)[0]
                    #print('theoindex',theoindex)
                    self.highlighttheospot = theoindex
                    hklstr = '[h,k,l]=[%d,%d,%d]'%(annote_theo[0][0], annote_theo[0][1], annote_theo[0][2])
                    finaltxt = 'theo spot index : %d, '%theoindex + hklstr + ' X,Y=(%.2f,%.2f) Energy=%.3f keV'%(annote_theo[1], annote_theo[2], annote_theo[3])
                    if finaltxt != self.savedfinaltxt:
                        print(finaltxt)
                    self.savedfinaltxt = finaltxt
                else:
                    self.sb.SetStatusText("",0)
                    tip_theo = ""
                    collisionFound_theo = False
                    self.highlighttheospot = None
                    self._replot(1)

            if collisionFound_exp or collisionFound_theo:
                if tip_exp != "":
                    fulltip = tip_exp + "\n" + tip_theo
                else:
                    fulltip = tip_theo

                self.tooltip.SetTip(tip + "\n" + fulltip)
                self.tooltip.Enable(True)

                self._replot(1)
                #return

        if not collisionFound_exp and not collisionFound_theo:
            self.tooltip.SetTip("")
            # set to False to avoid blocked tooltips from btns and checkboxes on windows os platforms

    #             self.tooltip.Enable(False)

    def updateStatusBar(self, x, y, annote, spottype="exp"):

        if self.datatype == "2thetachi":
            Xplot = "2theta"
            Yplot = "chi"
        else:
            Xplot = "x"
            Yplot = "y"

        if spottype == "theo":
            self.sb.SetStatusText(("%s= %.2f " % (Xplot, x)
                    + " %s= %.2f " % (Yplot, y)
                    + "  HKL=%s " % str(annote)), 0)

        elif spottype == "exp":

            self.sb.SetStatusText(("%s= %.2f " % (Xplot, x)
                    + " %s= %.2f " % (Yplot, y)
                    + "   Spotindex=%d " % annote[0]
                    + "   Intensity=%.2f" % annote[1]), 0)

    def onMotion(self, event):

        if self.press is None:
            self.onMotion_ToolTip(event)
            #print('self.press is None in OnMotion')
            return
            
        
        if not event.inaxes:
            #print('event.inaxes is False')
            return
               
        #print('self.press',self.press)
        if self.toolbar.mode in ('pan/zoom', 'zoom rect'):
            #print('self.toolbar.mode', self.toolbar.mode)
            return
        
        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        if dx == 0.0 and dy == 0.0:
            return

        # calculate new UBmat
        #print('Going to compute UB mat from GUI and mouse position')
        if self.datatype == "2thetachi":
            twth1, chi1 = self.press
            twth2, chi2 = event.xdata, event.ydata
            axis2theta, axischi = self.centerx, self.centery
        elif self.datatype == "pixels":
            X1, Y1 = self.press
            X2, Y2 = event.xdata, event.ydata

            twth1, chi1 = self.convertpixels2twotheta(X1, Y1)
            #             print 'twth1, chi1', twth1, chi1
            twth2, chi2 = self.convertpixels2twotheta(X2, Y2)
            #             print 'twth2, chi2', twth2, chi2
            axis2theta, axischi = self.convertpixels2twotheta(self.centerx, self.centery)
        else:
            return

        # left mouse button
        if event.button == 1:
            # drag a spot
            self.SelectedRotationAxis, angle = self.computeRotation(twth1, chi1, twth2, chi2)
        # right mouse button
        else:
            # rotate around a spot
            (self.SelectedRotationAxis,
                angle) = self.computeRotation_aroundaxis(axis2theta, axischi,
                                                        twth1, chi1, twth2, chi2)

        #         print "self.SelectedRotationAxis, angle", self.SelectedRotationAxis, angle
        self.RotateAroundAxis(angle)

        if self.datatype == "2thetachi":
            self.press = twth2, chi2
        elif self.datatype == "pixels":
            self.press = X2, Y2

    def convertpixels2twotheta(self, X, Y):

        tws, chs = F2TC.calc_uflab(np.array([X, X]),
                                    np.array([Y, Y]),
                                    self.CCDParam[:5],
                                    pixelsize=self.pixelsize,
                                    kf_direction=self.kf_direction)

        print('for kf_direction',self.kf_direction)
        print('X,Y: ',X,Y)
        print('2theta, chi', tws[0], chs[0])
        return tws[0], chs[0]

    def computeRotation(self, twth1, chi1, twth2, chi2):
        """
        compute rotation (axis, angle) from two points in 2theta chi coordinnates

        q = -2sintheta ( -sintheta,  costheta sin chi, costheta cos chi)
        rotation axis : q1unit^q2unit
        cos anglerot = q1unit.q2unit
        """
        q1 = np.array([-np.sin(twth1 / 2.0 * DEG),
                np.cos(twth1 / 2.0 * DEG) * np.sin(chi1 * DEG),
                np.cos(twth1 / 2.0 * DEG) * np.cos(chi1 * DEG)])
        q2 = np.array([-np.sin(twth2 / 2.0 * DEG),
                np.cos(twth2 / 2.0 * DEG) * np.sin(chi2 * DEG),
                np.cos(twth2 / 2.0 * DEG) * np.cos(chi2 * DEG)])

        qaxis = np.cross(q1, q2)
        angle = np.arccos(np.dot(q1, q2)) / DEG

        return qaxis, angle

    def computeRotation_aroundaxis(self, twthaxis, chiaxis, twth1, chi1, twth2, chi2):
        """
        compute rotation angle from two points in 2theta chi coordinnates around axis

        q = -2sintheta ( -sintheta,  costheta sin chi, costheta cos chi)
        rotation axis : q1unit^q2unit
        cos anglerot = q1unit.q2unit
        """
        # from self.centerx, self.centery
        qaxis = self.selectrotationaxis(twthaxis, chiaxis)

        qaxis = np.array(qaxis)

        q1 = qunit(twth1, chi1)
        q2 = qunit(twth2, chi2)

        beta = np.arccos(np.dot(q1, qaxis)) / DEG

        #         print "beta", beta
        #         print "qaxis", qaxis

        # q1 and q2 projection along qaxis and perpendicular to it
        q1_alongqaxis = np.dot(q1, qaxis)
        q1perp = q1 - q1_alongqaxis * qaxis

        q2_alongqaxis = np.dot(q2, qaxis)

        #         print 'q2_alongqaxis', q2_alongqaxis
        #         print 'q2', q2
        q2perp = q2 - (q2_alongqaxis * qaxis)

        # norms
        #         nq1perp = np.sqrt(np.dot(q1perp, q1perp))
        nq2perp = np.sqrt(np.dot(q2perp, q2perp))

        # q2_tilted coplanar with qaxis and q2, and same angle with qaxis than q1tilted
        #         q2_tilted = np.cos(beta * DEG) * qaxis + np.sin(beta * DEG) * q2perp / nq2perp
        #         q1_tilted = q1

        # q2tilted_perp and q1tilted_perp form a plane perpendicular to qaxis
        # angle between q2tilted_perp and q1tilted_perp is the rotation angle around qaxis
        #         q2tilted_perp = q2_tilted - (np.dot(q2_tilted, qaxis) * qaxis)
        q2tilted_perp = np.sin(beta * DEG) * q2perp / nq2perp
        # ie : np.sin(beta * DEG) * q2perp / nq2perp

        q1tilted_perp = q1perp

        # norm of qtilterperp  (must be equal)
        nq1tilted_perp = np.sqrt(np.dot(q1tilted_perp, q1tilted_perp))
        #         nq2tilted_perp = np.sqrt(np.dot(q2tilted_perp, q2tilted_perp))

        if nq1tilted_perp <= 0.0001:
            angle = 0
        else:
            angle = (1 / DEG * np.arcsin(np.dot(qaxis, np.cross(q1tilted_perp / nq1tilted_perp,
                                                    q2tilted_perp / nq1tilted_perp))))

        return qaxis, angle

    def selectrotationaxis(self, twtheta, chi):
        """
        return 3D vector of rotation axis from twtheta and chi axis coordinates
        """
        if self.datatype == "gnomon":
            RES = IIM.Fromgnomon_to_2thetachi([np.array([twtheta, twtheta]),
                                                    np.array([chi, chi])], 0)[:2]
            twtheta = RES[0][0]
            chi = RES[1][0]
        elif self.datatype == 'pixels':
            wx.MessageBox('Not implement yet','Info')
        #             twthetas, chis = F2TC.calc_uflab(np.array([twtheta, twtheta]),
        #                                          np.array([chi, chi]),
        #                                         self.CCDParam[:5],
        #                                         pixelsize=self.pixelsize,
        #                                         kf_direction=self.kf_direction)
        #             twtheta, chi = twthetas[0], chis[0]

        #         print "twtheta, chi", twtheta, chi
        theta = twtheta / 2.0

        sintheta = np.sin(theta * DEG)
        costheta = np.cos(theta * DEG)

        # q axis
        SelectedRotationAxis = [-sintheta,
                                costheta * np.sin(chi * DEG),
                                costheta * np.cos(chi * DEG)]

        return SelectedRotationAxis

    #         print "self.SelectedRotationAxis", self.SelectedRotationAxis

    def RotateAroundAxis(self, angle):
        """ compute rotation matrix from angle and self.SelectedRotationAxis and replot()"""

        self.deltamatrix = GT.matRot(self.SelectedRotationAxis, angle)

        self._replot(1)
        self.display_current()

    def drawAnnote_exp(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index

        in MainCalibrationFrame
        """
        if (x, y) in self.drawnAnnotations_exp:
            markers = self.drawnAnnotations_exp[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            if self.datatype == "2thetachi":
                t1 = axis.text(x + 1, y + 1, "%d" % (annote[0]), size=8)
                t2 = axis.text(x + 1, y - 1, "%.1f" % (annote[1]), size=8)
            elif self.datatype == "gnomon":
                t1 = axis.text(x + 0.02, y + 0.02, "%d" % (annote[0]), size=8)
                t2 = axis.text(x + 0.02, y - 0.02, "%.1f" % (annote[1]), size=8)
            elif self.datatype == "pixels":
                t1 = axis.text(x + 50, y + 50, "%d" % (annote[0]), size=8)
                t2 = axis.text(x + 50, y - 50, "%.1f" % (annote[1]), size=8)

            if matplotlibversion < "0.99.1":
                m = axis.scatter([x], [y], s=1, marker="d", c="r", zorder=100, faceted=False)
            else:
                m = axis.scatter([x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None")  # matplotlib 0.99.1.1

            self.drawnAnnotations_exp[(x, y)] = (t1, t2, m)
            # self.axis.figure.canvas.draw()
            self.canvas.draw()

    def drawSpecificAnnote_exp(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_exp if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_exp(self.axes, x, y, a)

    def Annotate_exp(self, event):
        """
        in MainCalibrationFrame
        """
        if self.datatype == "2thetachi":
            xtol = 20
            ytol = 20
            xdata, ydata, annotes = (self.twicetheta,
                                        self.chi,
                                        list(zip(self.Data_index_expspot, self.Data_I)))

        elif self.datatype == "gnomon":
            xtol = 0.05
            ytol = 0.05
            xdata, ydata, annotes = (self.data_gnomonx,
                                    self.data_gnomony,
                                    list(zip(self.Data_index_expspot, self.Data_I)))

        elif self.datatype == "pixels":
            xtol = 100
            ytol = 100
            xdata, ydata, annotes = (self.data_x, self.data_y,
                                     list(zip(self.Data_index_expspot, self.Data_I)))

        self._dataANNOTE_exp = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        print(clickX, clickY)

        annotes = []
        for x, y, a in self._dataANNOTE_exp:
            if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]
            print("the nearest experimental point is at(%.2f,%.2f)" % (x, y))
            print("with index %d and intensity %.1f" % (annote[0], annote[1]))
            self.drawAnnote_exp(self.axes, x, y, annote)
            for l in self.links_exp:
                l.drawSpecificAnnote_exp(annote)

    def drawAnnote_theo(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index

        in MainCalibrationFrame
        """
        if (x, y) in self.drawnAnnotations_theo:
            markers = self.drawnAnnotations_theo[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            if self.datatype == "2thetachi":
                t1 = axis.text(x + 1, y + 1,
                    "#%d hkl=%s\nE=%.3f keV" % (annote[0], str(annote[1]), annote[2]),
                    size=8)
            elif self.datatype == "gnomon":
                t1 = axis.text(x + 0.02, y + 0.02,
                    "#%d hkl=%s\nE=%.3f keV" % (annote[0], str(annote[1]), annote[2]),
                    size=8)
            elif self.datatype == "pixels":
                t1 = axis.text(x + 50, y + 50,
                    "#%d hkl=%s\nE=%.3f keV" % (annote[0], str(annote[1]), annote[2]),
                    size=8)

            if matplotlibversion < "0.99.1":
                m = axis.scatter([x], [y], s=1, marker="d", c="r", zorder=100, faceted=False)
            else:
                m = axis.scatter([x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None")  # matplotlib 0.99.1.1

            self.drawnAnnotations_theo[(x, y)] = (t1, m)
            # self.axis.figure.canvas.draw()
            self.canvas.draw()

    def drawSpecificAnnote_theo(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_theo if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_theo(self.axes, x, y, a)

    def Annotate_theo(self, event):
        """
        in MainCalibrationFrame
        """
        if self.datatype == "2thetachi":
            xtol = 20
            ytol = 20
            xdata, ydata, annotes = (self.data_theo[0], self.data_theo[1],
                list(zip(np.arange(len(self.data_theo[0])),
                        self.data_theo[2],
                        self.data_theo[-1])))

        elif self.datatype == "gnomon":
            xtol = 0.05
            ytol = 0.05
            xdata, ydata, annotes = (self.sim_gnomonx, self.sim_gnomony,
                list(zip(np.arange(len(self.data_theo[0])),
                        self.data_theo[2],
                        self.data_theo[-1])))

        elif self.datatype == "pixels":
            xtol = 100
            ytol = 100
            xdata, ydata, annotes = (self.data_theo[3], self.data_theo[4],
                list(zip(np.arange(len(self.data_theo[0])),
                        self.data_theo[2],
                        self.data_theo[-1])))

        self._dataANNOTE_theo = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        # print clickX, clickY

        annotes = []
        for x, y, a in self._dataANNOTE_theo:
            if (clickX - xtol < x < clickX + xtol) and (clickY - ytol < y < clickY + ytol):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]
            print("the nearest theo. point is at (%.2f,%.2f)" % (x, y))
            print("with index %d and Miller indices %s " % (annote[0], str(annote[1])))
            print("with Energy (and multiples) (keV):")
            print(givesharmonics(annote[2], self.emin, self.emax))
            self.drawAnnote_theo(self.axes, x, y, annote)
            for l in self.links_theo:
                l.drawSpecificAnnote_theo(annote)


def qunit(twth, chi):
    th = 0.5 * twth * DEG
    chi = chi * DEG
    sinth = np.sin(th)
    costh = np.cos(th)
    sinchi = np.sin(chi)
    coschi = np.cos(chi)

    return np.array([-sinth, costh * sinchi, costh * coschi])


def givesharmonics(E, _, Emax):
    """
    .. todo:: should consider Emin
    """
    multiples_E = []
    n = 1
    Eh = E
    while Eh <= Emax:
        multiples_E.append(Eh)
        n += 1
        Eh = E * n
    return multiples_E


def start():
    """ start of GUI for module launch"""
    initialParameter = {}
    #initialParameter["CCDParam"] = [71, 1039.42, 1095, 0.0085, -0.981]
    initialParameter["CCDParam"] = [71, 1000, 1000, 0.0, -0.0]
    initialParameter["detectordiameter"] = 165.0
    initialParameter["CCDLabel"] = "MARCCD165"
    initialParameter["filename"] = "Ge0001.dat"
    initialParameter["dirname"] = "/home/micha/LaueToolsPy3/LaueTools/Examples/Ge"
    initialParameter["dict_Materials"] = DictLT.dict_Materials
    kf_direction = 'Z>0'

    filepathname = os.path.join(initialParameter["dirname"], initialParameter["filename"])
    CalibGUIApp = wx.App()
    CalibGUIFrame = MainCalibrationFrame(None, -1, "Detector Calibration Board", initialParameter,
                                                        file_peaks=filepathname,
                                                        pixelsize=0.079142,
                                                        datatype="2thetachi",
                                                        dim=(2048, 2048),
                                                        fliprot="no",
                                                        kf_direction=kf_direction,
                                                        data_added=None)

    CalibGUIFrame.Show()

    CalibGUIApp.MainLoop()

if __name__ == "__main__":

    initialParameter = {}
    initialParameter["CCDParam"] = [71, 1039.42, 1095, 0.0085, -0.981]
    initialParameter["detectordiameter"] = 165.0
    initialParameter["CCDLabel"] = "MARCCD165"
    initialParameter["filename"] = "Ge0001.dat"
    initialParameter["dirname"] = "/home/micha/LaueToolsPy3/LaueTools/Examples/Ge"
    initialParameter["dict_Materials"] = DictLT.dict_Materials

    filepathname = os.path.join(initialParameter["dirname"], initialParameter["filename"])
    #    initialParameter['imagefilename'] = 'SS_0171.mccd'
    #    initialParameter['dirname'] = '/home/micha/lauetools/trunk'

    kf_direction = 'X>0'

    kf_direction = 'Z>0'

    CalibGUIApp = wx.App()
    CalibGUIFrame = MainCalibrationFrame(None, -1, "Detector Calibration Board", initialParameter,
                                                        file_peaks=filepathname,
                                                        pixelsize=165.0 / 2048,
                                                        datatype="2thetachi",
                                                        dim=(2048, 2048),
                                                        fliprot="no",
                                                        data_added=None,
                                                        kf_direction=kf_direction)

    CalibGUIFrame.Show()

    CalibGUIApp.MainLoop()
