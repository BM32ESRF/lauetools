r"""
GUI for microdiffraction Laue Pattern peaks list file loading.

This module belongs to the open source LaueTools project
with a free code repository at github

J. S. Micha Sept 2024
mailto: micha --+at-+- esrf --+dot-+- fr
"""
import os
import sys
import copy

import shutil

import wx

if sys.version_info.major == 3:
    from . import CCDFileParametersGUI as CCDParamGUI
    from . import DetectorParameters as DP
    from .. import dict_LaueTools as DictLT
    from .. import LaueGeometry as F2TC
    from .. import IOLaueTools as IOLT
    from .. import generaltools as GT
    from . import OpenSpotsListFileGUI as OSLFGUI

else:
    import GUI.CCDFileParametersGUI as CCDParamGUI
    import GUI.DetectorParameters as DP
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

def askUserForFilename(parent, filetype = 'peaklist', **dialogOptions):
    """
    provide a dialog to browse the folders and files

    set parent.dirnamepklist, parent.filenamepklist
    
    returns boolean userProvidedFilename
    """
    dialog = wx.FileDialog(parent, **dialogOptions)
    if dialog.ShowModal() == wx.ID_OK:
        userProvidedFilename = True
        # self.filename = dialog.GetFilename()
        # #self.dirname = dialog.GetDirectory()

        _allpath = dialog.GetPath()
        GT.printyellow(f'\n\nIn askUserForFilename() --------- selected file _allpath : {_allpath}')
        allpath = os.path.abspath(_allpath)
        GT.printyellow(f'\n\nIn askUserForFilename() --------- selected file allpath : {allpath}')


        if filetype == 'peaklist':
            parent.dirnamepklist, parent.filenamepklist = os.path.split(allpath)
        elif filetype == 'image':
            parent.imgdirname, parent.imgfilename = os.path.split(allpath)

    else:
        userProvidedFilename = False
    dialog.Destroy()
    return userProvidedFilename

def askUserForDirname(parent, verbose=0):
    """
    provide a dialog to browse the folders and files
    """
    if verbose>0: GT.printyellow('\n\nIn askUserForDirname():------')
    dialog = wx.DirDialog(parent, message="CHOOSE A FOLDER (WITH PERMISSION) TO WRITE RESULTS", defaultPath=parent.dirnamepklist)
    if dialog.ShowModal() == wx.ID_OK:
        # self.filename = dialog.GetFilename()
        # #self.dirname = dialog.GetDirectory()

        allpath = dialog.GetPath()
        if verbose>0: print('selected folder',allpath)
        writefolder = allpath

    dialog.Destroy()

    if verbose>0: print('checking permission for',writefolder)
    try:
        ftest = open(os.path.join(writefolder,"secondtestfile.txt"),"w")
        execOK=os.access(writefolder, mode=os.X_OK)
        print('execOK',execOK)
        if verbose>0:
            GT.printyellow('End of askUserForDirname():------')
            if not execOK: GT.printred('execOK',execOK)
        return writefolder
    except PermissionError:
        wx.MessageBox("You STILL don't have permission to write in this folder\n Select please another folder!", "Error", wx.OK | wx.ICON_ERROR)
        askUserForDirname(parent)


def OpenCorfile(filename, parent, verbose=0):
    """
    Reads a .cor file with spots porperties columns (usually 5: 2theta chi pixX pixY I).
    Reads also detector parameters and set defaultParam according to them
    
    creates or updates parent.Current_peak_data (all columns of .cor file)
    creates or updates parent.data_theta, parent.data_chi, parent.data_I, parent.data_pixX,
        parent.data_pixY, parent.CCDCalibDict

    :param parent: object, with mandatory attributes: kf_direction_from_file, CCDLabel, detectordiameter, dict_spotsproperties
    
    :return: data_theta, data_chi, data_pixX, data_pixY, data_I, dict_spotsproperties where
    dict_spotsproperties = {'columnsname': ..., 'data_spotsproperties': ...}
    
    """
    kf_direction_from_file, CCDLabel = parent.kf_direction_from_file, parent.CCDLabel

    dict_spotsproperties = {}  # potential dict of extra spots properties 

    if verbose>0: print('In OpenCorfile():\nOpening .cor file %s'%filename)
    (Current_peak_data, data_theta, data_chi,
        data_pixX, data_pixY, data_I,
        calib,
        CCDCalibDict, dict_spotsproperties
    ) = IOLT.readfile_cor(filename, output_CCDparamsdict=True, output_only5columns=False)

    #print("\nCCDCalibDict after readfile_cor ", CCDCalibDict)
    if 'CCDLabel' in CCDCalibDict:
        CCDLabel = CCDCalibDict['CCDLabel']

    

    CheckCCDCalibParameters(CCDCalibDict, kf_direction_from_file, CCDLabel, parent)

    pixelsize_fromfile = IOLT.getpixelsize_from_corfile(filename)
    kf_direction_fromfile = IOLT.getkfdirection_from_corfile(filename)
    parent.kf_direction = kf_direction_fromfile


    # update  CCDLabel,   framedim (nb pixels * nb pixels)
    parent.CCDLabel = CCDCalibDict['CCDLabel']
    # print('parent.CCDLabel', parent.CCDLabel)
    parent.framedim = DictLT.dict_CCD[parent.CCDLabel][0]
    parent.pixelsize = DictLT.dict_CCD[parent.CCDLabel][0]
    parent.detectordiameter = IOLT.getdetectordiameter_from_corfile(filename)


    # set parent parameters:

    if pixelsize_fromfile:
        parent.pixelsize = pixelsize_fromfile
        if verbose>0: print('reading pixelsize from file: %f mm'%parent.pixelsize)
    else:
        parent.pixelsize = DictLT.dict_CCD[parent.CCDLabel][1]
        if verbose>0: print('reading pixelsize from CCDLabel : %f mm'%parent.pixelsize)
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
    
    if len(dict_spotsproperties)>0:
        if verbose>0: print('dict_spotsproperties is not empty cool!')
        parent.dict_spotsproperties = dict_spotsproperties

    # Spots List to index object ----------------------
    # print("self.CCDCalibDict  end ", parent.CCDCalibDict)
    # print("self.pixelsize  end", parent.pixelsize)

    return (data_theta, data_chi,
            data_pixX, data_pixY, data_I, dict_spotsproperties)

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


def OpenPeakList(parent, writecorfile=True, verbose=0):
    """
    Load Peak list data (.dat or .cor). If input is a .dat file then a .cor file will be created

    set parent attributes:
        - dirnamepklist   
        - filenamepklist   .dat file or .cor file (built from .dat file)
        - defaultParam
        - pixelsize
        - framedim
        - detectordiameter
        - kf_direction
        - kf_direction_from_file
        - PeakListDatFileName  .dat file
        - DataPlot_filename

    :param parent: GUI object with above 
    """
    if verbose>0: GT.printyellow('\n\nIn OpenPeakList(): -------------------\n')
    if parent.resetwf is True:
        parent.writefolder = parent.dirnamepklist

    lastfolder = parent.dirnamepklist
    if parent.writefolder is not None:
        lastfolder = parent.writefolder

    if askUserForFilename(parent, style=wx.OPEN, **defaultFileDialogOptions(lastfolder)):

        # print String_in_File_Data # in stdout/stderr
        DataPlot_filename = str(parent.filenamepklist)
        fullpathfilename = os.path.join(parent.dirnamepklist, parent.filenamepklist)
        if verbose>0:
            print("Current file   :", DataPlot_filename)
            print("dirname   :", parent.dirnamepklist)

        prefix, file_extension = DataPlot_filename.rsplit(".", 1)

        try:
            # print('os.getlogin()',os.getlogin())
            # print('os.path.expanduser',os.path.expanduser('~'))
            with open(os.path.join(parent.dirnamepklist,"ezrztr.txt"),"w") as ftest:
                txt = " "
                ftest.write(txt)
        except PermissionError:
            if verbose>0:
                print('before:', parent.writefolder)
                GT.printred('Permission Error, select an other folder to write (temporarly or permanent) results')

            wx.MessageBox('Permission Error, select an other folder to write (temporarly or permanent) results','information', wx.OK | wx.ICON_INFORMATION)
            parent.writefolder = OSLFGUI.askUserForDirname(parent)

            if verbose>0:
                print('after:', parent.writefolder)

            source = fullpathfilename

            # Make sure the destination folder exists
            os.makedirs(parent.writefolder, exist_ok=True)

            shutil.copy(source, os.path.join(parent.writefolder,DataPlot_filename))

            fullpathfilename = os.path.join(parent.writefolder, DataPlot_filename)
            parent.filenamepklist = DataPlot_filename

            if verbose>0: print('after: parent.writefolder',parent.writefolder)

        else:
            parent.writefolder = parent.dirnamepklist
            GT.printgreen(f'In OpenPeakList(): I can write in {parent.writefolder}')
        
    if file_extension in ("dat", "DAT"):
        if verbose>0: print("In OpenPeakList(): .dat file branch")
        # open .det file to compute 2thea and chi scattering angles and write .cor file
        # will set defaultParam pixelsize framedim detectordiameter kf_direction
        Launch_DetectorParamBoard(parent)

        # will set kf_direction attributr
        LaueGeomBoard = SetGeneralLaueGeometry(parent, -1, "Select Laue Geometry")
        LaueGeomBoard.ShowModal()
        LaueGeomBoard.Destroy()

        

        # compute 2theta and chi according to detector calibration geometry
        (twicetheta, chi, dataintensity, data_x, data_y,
            dict_data_spotsproperties) = F2TC.Compute_data2thetachi(
                                                fullpathfilename, sorting_intensity="yes",
                                                detectorparams=parent.defaultParam,
                                                pixelsize=parent.pixelsize,
                                                kf_direction=parent.kf_direction,
                                                addspotproperties=True,
                                                verbose=verbose-1)

        # if not os.access(parent.dirnamepklist, os.W_OK):
        #     parent.writefolder = askUserForDirname(parent)
        #     print('choosing %s as folder for results  => ', parent.writefolder)
        # else:
        #     parent.writefolder = parent.dirnamepklist

        # write .cor file
        prefixfilename = "dat_" + prefix

        # ["dd", "xcen", "ycen", "xbet", "xgam", "pixelsize",
        # "xpixelsize", "ypixelsize", "CCDLabel",
        # "framedim", "detectordiameter", "kf_direction"]
        Parameters_dict = {}
        Parameters_dict["CCDLabel"] = parent.CCDLabel
        Parameters_dict["CCDParam"] = parent.defaultParam
        Parameters_dict["pixelsize"] = parent.pixelsize
        Parameters_dict["framedim"] = parent.framedim
        Parameters_dict["detectordiameter"] = parent.detectordiameter
        Parameters_dict["kf_direction"] = parent.kf_direction
        for k,v in zip(["dd", "xcen", "ycen", "xbet", "xgam"],parent.defaultParam[:5]):
            Parameters_dict[k]=v


        if dict_data_spotsproperties is not None:
            pass
            #print('dict_data_spotsproperties',dict_data_spotsproperties)

        if writecorfile:
            try:
                IOLT.writefile_cor(prefixfilename, twicetheta, chi, data_x, data_y,
                                    dataintensity, sortedexit=False,
                                    param=Parameters_dict,#parent.defaultParam + [parent.pixelsize],
                                    initialfilename=DataPlot_filename,
                                    dirname_output=parent.writefolder,
                                    dict_data_spotsproperties=dict_data_spotsproperties,
                                    verbose=verbose-1)
            except PermissionError:
                wx.MessageBox("Permission Error to write .cor file...\n Please, select an other folder to write (temporarly or permanent) results", "INFO")
                return
                
            if verbose>0:
                print("%s has been created\n in folder %s"%("dat_" + prefix + ".cor", parent.writefolder))
                print("with defaultparameter\n %s" % str(parent.defaultParam))

            parent.PeakListDatFileName = copy.copy(DataPlot_filename)
            parent.kf_direction_from_file = parent.kf_direction

            file_extension = "cor"

            fullpathfilename = os.path.join(parent.writefolder, "dat_" + prefix + "." + file_extension)
            parent.filenamepklist = fullpathfilename
            # WARNING: this file will be read just below in the just the following "if" branch !

    # for .cor file ------------------------------
    if file_extension == "cor":
        if verbose>0:
            print("In OpenPeakList(): .cor file branch")
            print('fullpathfilename of .cor file ', fullpathfilename)
        # read peak list and detector calibration parameters
        folder, filen = os.path.split(fullpathfilename)


        # if not os.access(folder, os.W_OK):
        #     print('Permission Error, select an other folder to write (temporarly or permanent) results')
        #     parent.writefolder = askUserForDirname(parent)
        #     print('choosing %s as folder for results  => ', parent.writefolder)
        # else:
        #     parent.writefolder = parent.dirnamepklist

        if verbose>0:
            print("parent.writefolder", parent.writefolder)
            print("before parent.filenamepklist", parent.filenamepklist)
            print("parent.dirnamepklist", parent.dirnamepklist)

        OpenCorfile(fullpathfilename, parent, verbose=verbose-1)

        parent.dict_spotsproperties = {}
        parent.DataPlot_filename = filen
        parent.filenamepklist = parent.DataPlot_filename
        parent.dirnamepklist = folder

        if verbose>0:
            print('parent.filenamepklist', parent.filenamepklist)
            print('parent.dirnamepklist', parent.dirnamepklist)
            print('parent.DataPlot_filename', parent.DataPlot_filename)



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

        initialGeo = DICT_LAUE_GEOMETRIES[parent.kf_direction]

        self.combogeo = wx.ComboBox(self, -1, str(initialGeo), size=(-1, 40),
                                    choices=["Top Reflection (2theta=90)",
                                            "Transmission (2theta=0)",
                                            "Back Reflection (2theta=180)"],
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

        if LaueGeometry == "Transmission (2theta=0)":
            kf_direction = "X>0"
        elif LaueGeometry == "Top Reflection (2theta=90)":
            kf_direction = "Z>0"
        elif LaueGeometry == "Back Reflection (2theta=180)":
            kf_direction = "X<0"

        self.parent.kf_direction = kf_direction

        print("Laue geometry set to: %s" % LaueGeometry)
        print("kf_direction set to: %s" % kf_direction)

        self.Close()

    def OnQuit(self, _):
        """ quit """
        self.Close()
