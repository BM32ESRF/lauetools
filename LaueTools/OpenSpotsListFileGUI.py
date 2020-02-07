import os
import sys
import copy

import wx

if sys.version_info.major == 3:
    from . import CCDFileParametersGUI as CCDParamGUI
    from . import DetectorParameters as DP
    from . import dict_LaueTools as DictLT
    from . import LaueGeometry as F2TC
    from . import IOLaueTools as IOLT
else:
    import CCDFileParametersGUI as CCDParamGUI
    import DetectorParameters as DP
    import dict_LaueTools as DictLT
    import LaueGeometry as F2TC
    import IOLaueTools as IOLT

DICT_LAUE_GEOMETRIES = DictLT.DICT_LAUE_GEOMETRIES

DICT_LAUE_GEOMETRIES_INFO = DictLT.DICT_LAUE_GEOMETRIES_INFO

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

def defaultFileDialogOptions(dirname):
    """ Return a dictionary with file dialog options that can be
        used in both the save file dialog as well as in the open
        file dialog. """
    wcd = "All files(*)|*|cor file(*.cor)|*.cor|peaks list(*.dat)|*.dat"
    return dict(message="Choose a data file (peaks list)",
                defaultDir=dirname,
                wildcard=wcd)

def askUserForFilename(parent, **dialogOptions):
    """
    provide a dialog to browse the folders and files

    set self.dirname, self.filename
    returns boolean userProvidedFilename
    """
    dialog = wx.FileDialog(parent, **dialogOptions)
    if dialog.ShowModal() == wx.ID_OK:
        userProvidedFilename = True
        # self.filename = dialog.GetFilename()
        # #self.dirname = dialog.GetDirectory()

        allpath = dialog.GetPath()
        print(allpath)

        parent.dirname, parent.filename = os.path.split(allpath)

    else:
        userProvidedFilename = False
    dialog.Destroy()
    return userProvidedFilename

def OpenCorfile(filename, parent):
    """
    Read a .cor file with 5 columns 2theta chi pixX pixY I
    reads detector parameters and set defaultParam according to them
    returns self.data_theta, self.data_chi, self.data_pixX, self.data_pixY, self.data_I

    creates or updates self.Current_peak_data (all columns of .cor file)
    creates or updates self.data_theta, self.data_chi, self.data_I, self.data_pixX,
        self.data_pixY, self.CCDCalibDict

    parent must have attributes: kf_direction_from_file, CCDLabel, detectordiameter
    """
    kf_direction_from_file, CCDLabel = parent.kf_direction_from_file, parent.CCDLabel

    print('Opening %s'%filename)
    (Current_peak_data,
        data_theta,
        data_chi,
        data_pixX,
        data_pixY,
        data_I,
        calib,
        CCDCalibDict,
    ) = IOLT.readfile_cor(filename, output_CCDparamsdict=True)

    print("\nCCDCalibDict after readfile_cor ", CCDCalibDict)

    CheckCCDCalibParameters(CCDCalibDict, kf_direction_from_file, CCDLabel, parent)

    pixelsize_fromfile = IOLT.getpixelsize_from_corfile(filename)

    # update  CCDLabel,   framedim (nb pixels * nb pixels)
    parent.CCDLabel = CCDCalibDict['CCDLabel']
    print('parent.CCDLabel', parent.CCDLabel)
    parent.framedim = DictLT.dict_CCD[parent.CCDLabel][0]
    parent.pixelsize = DictLT.dict_CCD[parent.CCDLabel][0]

    # set parent parameters:

    if pixelsize_fromfile:
        parent.pixelsize = pixelsize_fromfile
        print('reading pixelsize from file: %f mm'%parent.pixelsize)
    else:
        parent.pixelsize = DictLT.dict_CCD[parent.CCDLabel][1]
        print('reading pixelsize from CCDLabel : %f mm'%parent.pixelsize)
    if calib is not None:
        parent.defaultParam = calib

    parent.data_XY = (data_pixX, data_pixY)

    (parent.Current_peak_data,
        parent.data_theta,
        parent.data_chi,
        parent.data_pixX,
        parent.data_pixY,
        parent.data_I,
        calib,
        parent.CCDCalibDict) = (Current_peak_data,
                        data_theta,
                        data_chi,
                        data_pixX,
                        data_pixY,
                        data_I,
                        calib,
                        CCDCalibDict)

    # Spots List to index object ----------------------
    print("self.CCDCalibDict  end ", parent.CCDCalibDict)
    print("self.pixelsize  end", parent.pixelsize)

    return (data_theta, data_chi,
            data_pixX, data_pixY, data_I)

def CheckCCDCalibParameters(CCDCalibDict, kf_direction_from_file, CCDLabel, parent):
    """
    check if all CCD parameters are read from file .cor

    parent must have attributes:
        - kf_direction
    """
    ccp = DictLT.CCD_CALIBRATION_PARAMETERS

    sorted_list_parameters = [ccp[7], ccp[10]]
    for key in sorted_list_parameters:
        if key not in CCDCalibDict:
            missing_param = key

            if (missing_param == "kf_direction" and kf_direction_from_file is None):

                LaueGeomBoard = SetGeneralLaueGeometry(parent, -1, "Select Laue Geometry")
                LaueGeomBoard.ShowModal()
                LaueGeomBoard.Destroy()

            if missing_param == "CCDLabel" or CCDLabel is None:
                DPBoard = CCDParamGUI.CCDFileParameters(parent, -1, "CCD File Parameters Board",
                                                    CCDLabel)
                DPBoard.ShowModal()
                DPBoard.Destroy()

def Launch_DetectorParamBoard(parent):
    """Board to enter manually detector params
    Launch Entry dialog
    """
    Parameters_dict = {}
    Parameters_dict["CCDLabel"] = parent.CCDLabel
    Parameters_dict["CCDParam"] = parent.defaultParam
    Parameters_dict["pixelsize"] = parent.pixelsize
    Parameters_dict["framedim"] = parent.framedim
    Parameters_dict["detectordiameter"] = parent.detectordiameter
    Parameters_dict["kf_direction"] = parent.kf_direction
    # print "old param",self.defaultParam+[self.pixelsize]
    DPBoard = DP.DetectorParameters(parent, -1, "Detector parameters Board", Parameters_dict)
    DPBoard.ShowModal()
    DPBoard.Destroy()


def OnOpenPeakList(parent):
    """
    Load Peak list data (.dat or .cor)

    set parent attributes:
        - dirname
        - filename   .dat file or .cor file (built from .dat file)
        - defaultParam
        - pixelsize
        - framedim
        - detectordiameter
        - kf_direction
        - kf_direction_from_file
        - PeakListDatFileName  .dat file
        - DataPlot_filename
    """
    if askUserForFilename(parent, style=wx.OPEN, **defaultFileDialogOptions(parent.dirname)):
        # print "Current directory in OnOpenPeakList()",self.dirname
        os.chdir(parent.dirname)

        # print String_in_File_Data # in stdout/stderr
        DataPlot_filename = str(parent.filename)
        print("Current file   :", DataPlot_filename)

        prefix, file_extension = DataPlot_filename.rsplit(".", 1)

    if file_extension in ("dat", "DAT"):

        # open .det file to compute 2thea and chi scattering angles and write .cor file
        # will set defaultParam pixelsize framedim detectordiameter kf_direction
        Launch_DetectorParamBoard(parent)


        # will set kf_direction attributr
        LaueGeomBoard = SetGeneralLaueGeometry(parent, -1, "Select Laue Geometry")
        LaueGeomBoard.ShowModal()
        LaueGeomBoard.Destroy()

        # print("kf_direction in OnOpenPeakList", self.kf_direction)

        # compute 2theta and chi according to detector calibration geometry
        (twicetheta, chi, dataintensity,
            data_x, data_y) = F2TC.Compute_data2thetachi(
                                                DataPlot_filename,
                                                (0, 1, 3),
                                                1,
                                                sorting_intensity="yes",
                                                param=parent.defaultParam,
                                                pixelsize=parent.pixelsize,
                                                kf_direction=parent.kf_direction)
        # write .cor file
        IOLT.writefile_cor("dat_" + prefix,
                            twicetheta,
                            chi,
                            data_x,
                            data_y,
                            dataintensity,
                            sortedexit=0,
                            param=parent.defaultParam + [parent.pixelsize],
                            initialfilename=DataPlot_filename,
                        )  # check sortedexit = 0 or 1 to have decreasing intensity sorted data

        print("%s has been created with defaultparameter" % ("dat_" + prefix + ".cor"))
        print("%s" % str(parent.defaultParam))

        parent.PeakListDatFileName = copy.copy(DataPlot_filename)
        parent.kf_direction_from_file = parent.kf_direction

        file_extension = "cor"
        DataPlot_filename = "dat_" + prefix + "." + file_extension
        # WARNING: this file will be read in the next "if" clause

    # for .cor file ------------------------------
    if file_extension == "cor":
        # read peak list and detector calibration parameters
        OpenCorfile(DataPlot_filename, parent)

        parent.DataPlot_filename = DataPlot_filename
        parent.filename = parent.DataPlot_filename



# --- -------------------- general Laue Geometry settings
class SetGeneralLaueGeometry(wx.Dialog):
    """
    Dialog Class to set  general Laue Geometry

    parent must have   kf_direction attribute
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

        self.combogeo = wx.ComboBox(self, -1, str(initialGeo), size=(-1, 40),
                                    choices=["Top Reflection (2theta=90)", "Transmission"],
                                    style=wx.CB_READONLY)

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

    def OnChangeGeom(self, _):
        """change detection geometry
        """
        focus_geom = self.combogeo.GetValue()

        #         print "Laue Geometry info :", DICT_LAUE_GEOMETRIES_INFO[focus_geom]
        self.comments.SetValue(str(DictLT.DICT_LAUE_GEOMETRIES_INFO[focus_geom]))

    #         self.sb.SetStatusText(str(DICT_LAUE_GEOMETRIES_INFO[focus_geom]))

    def OnAccept(self, _):
        """accept geometry
        """
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

    def OnQuit(self, _):
        self.Close()
