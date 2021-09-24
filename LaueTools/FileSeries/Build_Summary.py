# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:23:57 2013

@author: micha

from initially T. Cerba
"""
import sys
import os
import copy

sys.path.append("..")

import matplotlib
matplotlib.use("WXAgg")

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        """ rename tooltip function for wxpython4 """
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip

if sys.version_info.major == 3:
    import LaueTools.dict_LaueTools as dictLT
else:
    import dict_LaueTools as dictLT

LAUETOOLSFOLDER = dictLT.LAUETOOLSFOLDER
LaueToolsProjectFolder = os.path.abspath(LAUETOOLSFOLDER)
print("LaueToolProjectFolder", LaueToolsProjectFolder)

from LaueTools.FileSeries import multigrainFS as MGFS

# #import FileSeries.multigrainFS as MGFS

try:
    import tables

    PYTABLES_EXISTS = True
except IOError:
    print("Unable to load tables module from pytables... Try install vitables by \n pip install vitables")
    PYTABLES_EXISTS = False

LIST_TXTPARAM_BS = ["Folder .fit file",
                    "Folder Result file",
                    "prefix .fit file",
                    ".fit suffix",
                    "Nbdigits filename",
                    "startindex",
                    "finalindex",
                    "stepindex",
                    "stiffness file (.stf)",
                    "Material",
                    "file xyz",
                    "nx",
                    "ny",
                    "fast axis",
                    "stepxy"]

TIP_BS = ["Folder containing indexed Peaks List .fit files",
    "Folder to write summary files",
    "Prefix for .fit files filename prefix####.fit",
    'file suffix. Default ".fit"',
    "maximum nb of digits for zero padding of filename index.(e.g. nb of # in prefix####.fit)\n0 for no zero padding.",
    "starting file index (integer)",
    "final file index (integer)",
    "incremental step file index (integer)",
    "full path to stiffness file .stf (for single material stress evaluation)",
    "Material (for single material stress evaluation)",
    "file xyz : full path to file xy with 3 columns (imagefile_index x y)",
    'nb of images along x direction (nb of columns, nb of images per line).\nNumber of images (points) per line along the "fast axis"',
    'nb of images along y direction (nb of lines along x)\n.Number of images (points) per row along the "slow axis"',
    'sample direction for fast motor axis: "x"or "y"',
    "steps (Dx,Dy) along resp. fast and slow axes in micrometer. Dx and Dy can be negative.\nBy increasing image index position along fast axis increases by Dx. Between each line position along slow axis increases by Dy",
]

DICT_TOOLTIP = {}
for key, tip in zip(LIST_TXTPARAM_BS, TIP_BS):
    DICT_TOOLTIP[key] = "%s : %s" % (key, tip)


DICT_TOOLTIP["Material"] = "Material : Material"
DICT_TOOLTIP["file xyz"] = "file xyz : full path to file xy with 3 columns (imagefile_index x y)"

class MainFrame_BuildSummary(wx.Frame):
    def __init__(self, parent, _id, title, _initialparameters):
        wx.Frame.__init__(self, parent, _id, title, wx.DefaultPosition, wx.Size(1000, 700))
        self.parent = parent
        self.initialparameters = _initialparameters
        print('self.initialparameters', self.initialparameters)
        file_xyz = self.initialparameters[10]
        print('file_xyz', file_xyz)

        self.panel = wx.Panel(self)
        if WXPYTHON4:
            grid = wx.FlexGridSizer(3, 10, 10)
        else:
            grid = wx.FlexGridSizer(14, 3)

        grid.SetFlexibleDirection(wx.HORIZONTAL)

        txt_fields = LIST_TXTPARAM_BS[:9] + ["Material", "file xy"]

        val_fields = copy.copy(_initialparameters[:9])
        val_fields.append("Si")
        val_fields.append(file_xyz)

        dict_tooltip = DICT_TOOLTIP
        keys_list_dicttooltip = list(DICT_TOOLTIP.keys())

        self.list_txtctrl = []
        for kk, txt_elem in enumerate(txt_fields):
            txt = wx.StaticText(self.panel, -1, "     %s" % txt_elem)
            grid.Add(txt)
            self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))

            self.txtctrl.SetValue(str(val_fields[kk]))

            if txt_elem in keys_list_dicttooltip:

                txt.SetToolTipString(dict_tooltip[txt_elem])
                self.txtctrl.SetToolTipString(dict_tooltip[txt_elem])

            grid.Add(self.txtctrl)

            self.list_txtctrl.append(self.txtctrl)
            if kk in (0, 1, 2, 8, 10):
                btnbrowse = wx.Button(self.panel, -1, "Browse")
                grid.Add(btnbrowse)
                if kk == 0:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_fitfilesfolder)
                    btnbrowse.SetToolTipString("Select Folder containing .fit files")
                elif kk == 1:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_results_folder)
                    btnbrowse.SetToolTipString(
                        "Select output folder for summary results files")
                elif kk == 2:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout_fit)
                    btnbrowse.SetToolTipString(
                        "Select one .fit file to get (and guess) the generic prefix of all .fit filenames")
                elif kk == 8:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_stiffnessfile)
                    btnbrowse.SetToolTipString("Select stiffness .stf file")
                
                elif kk == 10:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filexy)
                    btnbrowse.SetToolTipString("Select a filexy.dat (3 columnes ascii file: image index, x, y ")

            else:
                nothing = wx.StaticText(self.panel, -1, "")
                grid.Add(nothing)
       
        btn_fabrication_to_hand = wx.Button(
            self.panel, -1, "Build file xy manually", size=(200, -1))

        grid.Add(wx.StaticText(self.panel, -1, ""))
        grid.Add(btn_fabrication_to_hand)
        txt_none = wx.StaticText(self.panel, -1, "")
        grid.Add(txt_none)
        
        btn_fabrication_to_hand.Bind(wx.EVT_BUTTON, self.OnbuildManually)

        grid.Add(wx.StaticText(self.panel, -1, ""))
        self.builddatfile = wx.CheckBox(self.panel, -1, "Build .dat file")
        self.builddatfile.SetValue(True)
        self.buildhdf5 = wx.CheckBox(self.panel, -1, "Build .hdf5 file")
        self.buildhdf5.SetValue(False)

        if not PYTABLES_EXISTS:
            self.buildhdf5.Disable()
        else:
            self.buildhdf5.SetValue(True)

        grid.Add(self.builddatfile)
        grid.Add(self.buildhdf5)

        btnStart = wx.Button(self.panel, -1, "BUILD SUMMARY FILE(s)", size=(-1, 60))

        btnStart.Bind(wx.EVT_BUTTON, self.OnCreateSummary)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0)
        vbox.Add(btnStart, 0, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

        vbox.Fit(self)
        self.Layout()

        # tooltips ----------
        btn_fabrication_to_hand.SetToolTipString("Create a 3 columns file: image_index, x, y")
        btnStart.SetToolTipString("Start reading all .fit files and build summary files in folder results")

        self.builddatfile.SetToolTipString("Build an ASCII summary file with extension .dat")
        self.buildhdf5.SetToolTipString("Build a hdf5 file system file")

    def OnbtnBrowse_fitfilesfolder(self, _):
        folder = wx.DirDialog(self, "Select folder for refined peaklist .fit files")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            self.list_txtctrl[0].SetValue(abspath)

            mainpath = os.path.split(abspath)[0]
            self.list_txtctrl[1].SetValue(mainpath)

    def OnbtnBrowse_results_folder(self, _):
        folder = wx.DirDialog(self, "Select folder for writing summary results")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            self.list_txtctrl[1].SetValue(abspath)

    def OnbtnBrowse_filepathout_fit(self, _):
        folder = wx.FileDialog(self,
            "Select refined Peaklist File .fit for catching prefix",
            wildcard="Refined PeakList (*.fit)|*.fit|All files(*)|*",)

        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()
            filename = os.path.split(abspath)[-1]
            intension, extension = filename.split(".")

            self.list_txtctrl[3].SetValue("." + extension)

            nbdigits = int(self.list_txtctrl[4].GetValue())
            self.list_txtctrl[2].SetValue(intension[:-nbdigits])

    def OnbtnBrowse_stiffnessfile(self, _):
        ff = wx.FileDialog(self, "Select stiffness file .stf",
            wildcard="stiffness file (*.stf)|*.stf|All files(*)|*")
        if ff.ShowModal() == wx.ID_OK:

            self.list_txtctrl[8].SetValue(ff.GetPath())

    def OnbtnBrowse_filexy(self, _):
        ff = wx.FileDialog(self, "Select file xy .dat",
            wildcard="filexy file (*.dat)|*.dat|All files(*)|*")
        if ff.ShowModal() == wx.ID_OK:

            self.list_txtctrl[10].SetValue(ff.GetPath())

    def OnbtnChangeparameters(self, _):

        wx.MessageBox("Sorry! This will be implemented very soon!", "INFO")
        return

    def OnbuildManually(self, _):
        PSboard = Manual_XYZfilecreation_Frame(
            self, -1, "File sample XYZ position: Manual creation Board")
        PSboard.Show(True)

    def Onbtn_fabrication_with_image(self, _):
        wx.MessageBox("Sorry! This will be implemented very soon!", "INFO")
        return

    def OnCreateSummary(self, _):

        if not self.builddatfile.GetValue() and not self.buildhdf5.GetValue():
            wx.MessageBox("Check at least one type of summary file!", "Error")

        startindex = int(self.list_txtctrl[5].GetValue())
        finalindex = int(self.list_txtctrl[6].GetValue())
        stepindex = int(self.list_txtctrl[7].GetValue())

        image_indices = list(range(startindex, finalindex + 1, stepindex))
        prefix = str(self.list_txtctrl[2].GetValue())
        nbdigits_for_zero_padding = int(self.list_txtctrl[4].GetValue())
        suffix = str(self.list_txtctrl[3].GetValue())

        folderfitfiles = str(self.list_txtctrl[0].GetValue())
        folderresult = str(self.list_txtctrl[1].GetValue())

        stiffnessfile = str(self.list_txtctrl[8].GetValue())

        key_material = str(self.list_txtctrl[9].GetValue())

        filexyz = str(self.list_txtctrl[10].GetValue())

        


        if self.builddatfile.GetValue():
            try:
                _, fullpath_summary_filename = MGFS.build_summary(
                                    image_indices,
                                    folderfitfiles,
                                    prefix,
                                    suffix,
                                    filexyz,
                                    startindex=startindex,
                                    finalindex=finalindex,
                                    number_of_digits_in_image_name=nbdigits_for_zero_padding,
                                    folderoutput=folderresult,
                                    default_file=DEFAULT_FILE)

                print("fullpath_summary_filename", fullpath_summary_filename)

                fullpath_summary_filename = MGFS.add_columns_to_summary_file_new(
                                                        fullpath_summary_filename,
                                                        elem_label=key_material,
                                                        filestf=stiffnessfile)

                wx.MessageBox("Operation Successful! \t \t Summary file created here: %s"
                    % fullpath_summary_filename)

                if self.parent is not None:
                    object_to_set = self.parent
                    object_to_set.initialparameters["Map Summary File"] = fullpath_summary_filename
                    object_to_set.initialparameters["File xyz"] = filexyz

            except ValueError as err:
                wx.MessageBox("%s"%str(err))
        if self.buildhdf5.GetValue():
            from Lauehdf5 import Add_allspotsSummary_from_fitfiles

            "\n\n ****** \nBuilding a hdf5 summary file\n*********  \n\n"
            Summary_HDF5_filename = prefix + ".h5"
            Add_allspotsSummary_from_fitfiles(Summary_HDF5_filename,
                                                prefix,
                                                folderfitfiles,
                                                image_indices,
                                                Summary_HDF5_dirname=folderfitfiles,
                                                number_of_digits_in_image_name=nbdigits_for_zero_padding,
                                                filesuffix=".fit",
                                                nb_of_spots_per_image=300)

class Manual_XYZfilecreation_Frame(wx.Frame):
    """
    GUI class for setting parameters to build xyz file
    """
    def __init__(self, parent, _id, title):
        wx.Frame.__init__(self, parent, _id, title, wx.DefaultPosition,
                                                        wx.Size(350, 300))
        self.parent = parent
        self.panel = wx.Panel(self)

        if WXPYTHON4:
            grid = wx.FlexGridSizer(4, 10, 10)
        else:
            grid = wx.FlexGridSizer(4, 4)

        grid.SetFlexibleDirection(wx.HORIZONTAL)

        self.list_txtctrl_manual = []

        proposedfilexyname = '{}_{}_to_{}_xy.dat'.format(
                            self.parent.list_txtctrl[2].GetValue(),
                            self.parent.list_txtctrl[5].GetValue(),
                            self.parent.list_txtctrl[6].GetValue())
        self.parent.initialparameters[10] = os.path.join(self.parent.list_txtctrl[1].GetValue(),
                                                proposedfilexyname)

        for kk, elem in enumerate(LIST_TXTPARAM_BS):
            if kk >= 10:
                grid.Add(wx.StaticText(self.panel, -1, "    %s" % elem))

                self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(200, 25))
                self.txtctrl.SetValue(str(self.parent.initialparameters[kk]))
                self.list_txtctrl_manual.append(self.txtctrl)
                grid.Add(self.txtctrl)
                nothing = wx.StaticText(self.panel, -1, "")
                grid.Add(nothing)

                grid.Add(wx.Button(self.panel, kk + 9, "?", size=(25, 25)))

        self.Bind(wx.EVT_BUTTON, lambda event: self.OnbtnBrowse_filepathout(self), id=12)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filepathout(self), id=13)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_fileprefix(self), id=14)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filesuffix(self), id=15)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nbdigits(self), id=16)

        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_indimg(self), id=17)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onfilexyz_help(self), id=19)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nx(self), id=20)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_ny(self), id=21)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_xfast(self), id=22)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_xstep(self), id=23)

        btnStart_BS_fabricationHand = wx.Button(
            self.panel, -1, "Create File with array Index,X,Y", size=(-1, 60))

        btnStart_BS_fabricationHand.Bind(wx.EVT_BUTTON, self.start_manualXYZ)
        #layout---------------------------
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        vbox.Add(btnStart_BS_fabricationHand, 0, wx.EXPAND)

        self.help = wx.TextCtrl(
            self.panel, -1, "", style=wx.TE_MULTILINE | wx.TE_READONLY, size=(250, 100))
        vbox.Add(self.help, 1, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

    def Onbtnhelp_nbdigits(self, _):
        helpstring = "a remplir"
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_filepathout(self, _):
        helpstring = "a remplir"
        self.help.SetValue(str(helpstring))

    def Onfilexyz_help(self, _):
        helpstring = "file xyz : full path to file xy with 3 columns (imagefile_index x y)"
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_nx(self, _):
        helpstring = 'Number of images (points) per line along the "fast axis" (X (resp Y) axis is "fast axis" is set to "x" resp. "y"))'
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_ny(self, _):
        helpstring = 'Number of images (points) per column along the "slow axis"'
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_xfast(self, _):
        helpstring = 'sample direction for fast motor axis: "x"or "y"'
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_xstep(self, _):
        helpstring = "steps (Dx,Dy) along resp. fast and slow axes in micrometer. Dx and Dy can be negative.\n"
        helpstring += "By increasing image index position along fast axis increases by Dx.\n"
        helpstring += "Between each line position along slow axis increases by Dy"
        self.help.SetValue(str(helpstring))

    #     def Onbtnhelp_ystep(self, event):
    #         helpstring = 'a remplir'
    #         self.help.SetValue(str(helpstring))
    #     def Onbtnhelp_indimg(self, event):
    #         helpstring = 'a remplir'
    #         self.help.SetValue(str(helpstring))
    def OnbtnBrowse_filepathout(self, _):
        folder = wx.DirDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            self.list_txtctrl_manual[2].SetValue(folder.GetPath())

    def start_manualXYZ(self, _):
        """
        read parameters and launch creation of file xy
        """
        #         manag.Activate_BuildSummary()

        check = 1

        nx = int(self.list_txtctrl_manual[1].GetValue())
        ny = int(self.list_txtctrl_manual[2].GetValue())
        fastaxis = str(self.list_txtctrl_manual[3].GetValue())
        stepxy = str(self.list_txtctrl_manual[4].GetValue())

        if fastaxis in ("x", "X"):
            xfast, yfast = 1, 0
        elif fastaxis in ("y", "Y"):
            xfast, yfast = 0, 1

        steplist = stepxy[1:-1].split(",")

        if len(steplist) != 2:
            wx.MessageBox("Wrong typed stepxy !", "Error")
            return

        xstep, ystep = float(steplist[0]), float(steplist[1])

        print(self.parent.list_txtctrl)

        # prefix = str(self.parent.list_txtctrl[2].GetValue())

        outfilename = str(self.list_txtctrl_manual[0].GetValue())

        if check == 1:
            # writing filexyz with xy for map sample description in
            MGFS.build_xy_list_by_hand(outfilename, nx, ny, xfast, yfast, xstep, ystep,
                        startindex=int(self.parent.list_txtctrl[5].GetValue()),
                        lastindex=int(self.parent.list_txtctrl[6].GetValue()), )

            self.parent.list_txtctrl[10].SetValue(outfilename)

            wx.MessageBox(
                "Operation Successful! \t \t Filexyz created: %s" % outfilename)
        else:
            wx.MessageBox("Files's path or input datas are missing!")

        self.Destroy()


class Stock_parameters_BuildSummary_image:
    def __init__(self, list_txtparamBSi, list_valueparamBSi):
        self.list_txtparamBSi = list_txtparamBSi
        self.list_valueparamBSi = list_valueparamBSi


class Stock_parameters_BuildSummary_hand:
    def __init__(self, list_txtparamBS, _list_valueparamBS):
        self.list_txtparamBS = list_txtparamBS
        self.list_valueparamBS = _list_valueparamBS


def fill_list_valueparamBS(initialparameters_dict):
    """
    return a list of default value for buildsummary board from a dict initialparameters
    """
    list_valueparam_BS = [
        initialparameters_dict["IndexRefine PeakList Folder"],
        initialparameters_dict["IndexRefine PeakList Folder"],
        initialparameters_dict["IndexRefine PeakList Prefix"],
        initialparameters_dict["IndexRefine PeakList Suffix"],
        initialparameters_dict["nbdigits"],
        initialparameters_dict["startingindex"],
        initialparameters_dict["finalindex"],
        initialparameters_dict["stepindex"],
        initialparameters_dict["stiffness file"],
        initialparameters_dict["Material"],
        initialparameters_dict["file xyz"],
        initialparameters_dict["Map shape"][1],
        initialparameters_dict["Map shape"][0],
        initialparameters_dict["fast axis: x or y"],
        initialparameters_dict["(stepX, stepY) microns"]]
    return list_valueparam_BS


initialparameters = {}

print("LaueToolProjectFolder", LaueToolsProjectFolder)

MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "CuSi")

print("MainFolder", MainFolder)

initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")

initialparameters["file xyz"] = os.path.join(MainFolder, "fitfiles", "SiCustrain_0_to_5_xy.dat")
initialparameters["IndexRefine PeakList Prefix"] = "SiCustrain"
initialparameters["IndexRefine PeakList Suffix"] = ".fit"

initialparameters["Map shape"] = (5, 3)  # (nb lines, nb images per line)

initialparameters["(stepX, stepY) microns"] = (1.0, 1.0)
initialparameters["Material"] = "Si"
initialparameters["stiffness file"] = os.path.join(MainFolder, "si.stf")

initialparameters["nbdigits"] = 0
initialparameters["startingindex"] = 0
initialparameters["finalindex"] = 5
initialparameters["stepindex"] = 1
initialparameters["fast axis: x or y"] = "x"

list_valueparamBS = fill_list_valueparamBS(initialparameters)

DEFAULT_FILE = os.path.join(initialparameters["IndexRefine PeakList Folder"],
                                            "nanox2_400_0000.fit")

def start():
    BuildSummaryApp = wx.App()
    BSFrame = MainFrame_BuildSummary(None, -1,
                                "Build Summary Parameters Board", list_valueparamBS)
    BSFrame.Show(True)
    BuildSummaryApp.MainLoop()

if __name__ == "__main__":

    start()
