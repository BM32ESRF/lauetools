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

from dict_LaueTools import LAUETOOLSFOLDER
LaueToolsProjectFolder = os.path.abspath(LAUETOOLSFOLDER)
print("LaueToolProjectFolder", LaueToolsProjectFolder)

import FileSeries.multigrainFS as MGFS
import FileSeries.module_graphique as modgraph
import param_multigrain as PAR

try:
    import tables

    PYTABLES_EXISTS = True
except IOError:
    print("Unable to load tables module from pytables...")
    PYTABLES_EXISTS = False

LIST_TXTPARAM_BS = [
    "Folder .fit file",
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
    "stepxy",
]

TIP_BS = [
    "Folder containing indexed Peaks List .fit files",
    "Folder to write summary files",
    "Prefix for .fit files filename prefix####.fit",
    'file suffix. Default ".fit"',
    "maximum nb of digits for zero padding of filename index.(e.g. nb of # in prefix####.fit)\n0 for no zero padding.",
    "starting file index (integer)",
    "final file index (integer)",
    "incremental step file index (integer)",
    "full path to stiffness file .stf for single material",
    "Material",
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
DICT_TOOLTIP[
    "file xyz"
] = "file xyz : full path to file xy with 3 columns (imagefile_index x y)"


class MainFrame_BuildSummary(wx.Frame):
    def __init__(self, parent, _id, title, _initialparameters):
        wx.Frame.__init__(
            self, parent, _id, title, wx.DefaultPosition, wx.Size(1000, 700)
        )

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

        txt_fields = LIST_TXTPARAM_BS[:9] + ["Material", "file xyz"]

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
            if kk in (0, 1, 2, 8):
                btnbrowse = wx.Button(self.panel, -1, "Browse")
                grid.Add(btnbrowse)
                if kk == 0:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_fitfilesfolder)
                    btnbrowse.SetToolTipString("Select Folder containing .fit files")
                elif kk == 1:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_results_folder)
                    btnbrowse.SetToolTipString(
                        "Select output folder for summary results files"
                    )
                elif kk == 2:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filepathout_fit)
                    btnbrowse.SetToolTipString(
                        "Select one .fit file to get (and guess) the generic prefix of all .fit filenames"
                    )
                elif kk == 8:
                    btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_stiffnessfile)
                    btnbrowse.SetToolTipString("Select stiffness .stf file")

            else:
                nothing = wx.StaticText(self.panel, -1, "")
                grid.Add(nothing)

        #         btnChangeparameters = wx.Button(self.panel, -1, "Change parameters")
        #         grid.Add(btnChangeparameters)
        #         btnChangeparameters.Bind(wx.EVT_BUTTON, self.OnbtnChangeparameters)
        #
        #         none8 = wx.StaticText(self.panel, -1, "")
        #         grid.Add(none8)
        #         none9 = wx.StaticText(self.panel, -1, "")
        #         grid.Add(none9)
        #         none10 = wx.StaticText(self.panel, -1, "")
        #         grid.Add(none10)

        txt_fileparameters = wx.StaticText(
            self.panel, -1, "Step 1 : Build (index,x,y) file"
        )
        grid.Add(txt_fileparameters)
        font = wx.Font(12, wx.MODERN, wx.ITALIC, wx.NORMAL)
        txt_fileparameters.SetFont(font)
        txt_none = wx.StaticText(self.panel, -1, "")
        txt_none_2 = wx.StaticText(self.panel, -1, "")
        grid.Add(txt_none)
        grid.Add(txt_none_2)

        btn_fabrication_to_hand = wx.Button(
            self.panel, -1, "Build manually", size=(200, -1)
        )
        grid.Add(btn_fabrication_to_hand)
        btn_fabrication_to_hand.Bind(wx.EVT_BUTTON, self.OnbuildManually)
        btn_fabrication_with_image = wx.Button(
            self.panel, -1, "Build from Images Info", size=(200, -1)
        )
        btn_fabrication_with_image.Disable()
        grid.Add(btn_fabrication_with_image)
        btn_fabrication_with_image.Bind(
            wx.EVT_BUTTON, self.Onbtn_fabrication_with_image
        )
        txt_none_3 = wx.StaticText(self.panel, -1, "")
        grid.Add(txt_none_3)

        txt_buildsummary = wx.StaticText(
            self.panel, -1, "Step 2 : Build summary files "
        )
        grid.Add(txt_buildsummary)

        txt_buildsummary.SetFont(font)
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

        btnStart = wx.Button(self.panel, -1, "CREATE FILE(s)", size=(-1, 60))

        btnStart.Bind(wx.EVT_BUTTON, self.OnCreateSummary)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0)
        vbox.Add(btnStart, 0, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

        vbox.Fit(self)
        self.Layout()

        # tooltips ----------
        btn_fabrication_to_hand.SetToolTipString(
            "Create a 3 columns file: image_index, x, y"
        )
        btnStart.SetToolTipString(
            "Start reading all .fit files and build summary files in folder results"
        )

        self.builddatfile.SetToolTipString("Build an ASCII summary file with extension .dat")
        self.buildhdf5.SetToolTipString("Build a hdf5 file system file")

    def OnbtnBrowse_fitfilesfolder(self, evt):
        folder = wx.DirDialog(self, "Select folder for refined peaklist .fit files")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            self.list_txtctrl[0].SetValue(abspath)

            mainpath = os.path.split(abspath)[0]
            self.list_txtctrl[1].SetValue(mainpath)

    def OnbtnBrowse_results_folder(self, evt):
        folder = wx.DirDialog(self, "Select folder for writing summary results")
        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            self.list_txtctrl[1].SetValue(abspath)

    def OnbtnBrowse_filepathout_fit(self, evt):
        folder = wx.FileDialog(
            self,
            "Select refined Peaklist File .fit for catching prefix",
            wildcard="Refined PeakList (*.fit)|*.fit|All files(*)|*",
        )

        if folder.ShowModal() == wx.ID_OK:

            abspath = folder.GetPath()

            #             print "folder.GetPath()", abspath

            filename = os.path.split(abspath)[-1]
            #             print "filename", filename
            intension, extension = filename.split(".")

            self.list_txtctrl[3].SetValue("." + extension)

            nbdigits = int(self.list_txtctrl[4].GetValue())
            self.list_txtctrl[2].SetValue(intension[:-nbdigits])

    def OnbtnBrowse_stiffnessfile(self, evt):
        folder = wx.FileDialog(
            self,
            "Select stiffness file .stf",
            wildcard="stiffness file (*.stf)|*.stf|All files(*)|*",
        )
        if folder.ShowModal() == wx.ID_OK:

            self.list_txtctrl[8].SetValue(folder.GetPath())

    def OnbtnChangeparameters(self, event, objet_BS, objet_BSi):

        wx.MessageBox("Sorry! This will be implemented very soon!", "INFO")
        return
        # PSboard = SetParametersFrame(self, -1, 'New parameters',
        #                                       objet_BS, objet_BSi, self.list_txtctrl)
        # PSboard.Show(True)

    def OnbuildManually(self, event):
        PSboard = Manual_XYZfilecreation_Frame(
            self, -1, "File sample XYZ position: Manual creation Board"
        )
        PSboard.Show(True)

    def Onbtn_fabrication_with_image(self, event):
        wx.MessageBox("Sorry! This will be implemented very soon!", "INFO")
        return
        # PSboard = SetParameters_BuildSummary_fabricationImage(self, -1,
        #                 'Entryparameters_fabricationimage')
        # PSboard.Show(True)

    def OnCreateSummary(self, event):

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

        #         if not os.path.exists(str(objet_BS.list_valueparamBS[0]) + 'summary_new'):
        #                 os.makedirs(str(objet_BS.list_valueparamBS[0]) + 'summary_new')
        #         modgraph.outfilename = (str(objet_BS.list_valueparamBS[0]) + 'summary_new')

        if self.builddatfile.GetValue():

            allres, fullpath_summary_filename = MGFS.build_summary(
                image_indices,
                folderfitfiles,
                prefix,
                suffix,
                filexyz,
                startindex=startindex,
                finalindex=finalindex,
                number_of_digits_in_image_name=nbdigits_for_zero_padding,
                folderoutput=folderresult,
                default_file=DEFAULT_FILE,
            )

            print("fullpath_summary_filename", fullpath_summary_filename)

            fullpath_summary_filename = MGFS.add_columns_to_summary_file_new(
                fullpath_summary_filename,
                elem_label=key_material,
                filestf=stiffnessfile,
            )

            wx.MessageBox(
                "Operation Successful! \t \t Summary file created here: %s"
                % fullpath_summary_filename
            )
        if self.buildhdf5.GetValue():
            from Lauehdf5 import Add_allspotsSummary_from_fitfiles

            "\n\n ****** \nBuilding a hdf5 summary file\n*********  \n\n"
            Summary_HDF5_filename = prefix + ".h5"
            Add_allspotsSummary_from_fitfiles(
                Summary_HDF5_filename,
                prefix,
                folderfitfiles,
                image_indices,
                Summary_HDF5_dirname=folderfitfiles,
                number_of_digits_in_image_name=4,
                filesuffix=".fit",
                nb_of_spots_per_image=300,
            )


class Manual_XYZfilecreation_Frame(wx.Frame):
    """
    GUI class for setting parameters to build xyz file 
    """
    def __init__(self, parent, _id, title):
        wx.Frame.__init__(
            self, parent, _id, title, wx.DefaultPosition, wx.Size(350, 300)
        )
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
                            self.parent.list_txtctrl[6].GetValue()
                            )
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

        self.Bind(
            wx.EVT_BUTTON, lambda event: self.OnbtnBrowse_filepathout(self), id=12
        )
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filepathout(self), id=13)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_fileprefix(self), id=14)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filesuffix(self), id=15)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nbdigits(self), id=16)
        #         self.Bind(wx.EVT_BUTTON, lambda event: self.OnbtnChange_indimg(self), id=25)

        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_indimg(self), id=17)
        #       id = 16 correspond à la lignée qui saute, donc pas d'association avec bouton
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onfilexyz_help(self), id=19)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nx(self), id=20)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_ny(self), id=21)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_xfast(self), id=22)
        #         self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_yfast(self), id=23)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_xstep(self), id=23)
        #         self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_ystep(self), id=25)

        btnStart_BS_fabricationHand = wx.Button(
            self.panel, -1, "Create File with array Index,X,Y", size=(-1, 60)
        )

        btnStart_BS_fabricationHand.Bind(wx.EVT_BUTTON, self.start_manualXYZ)
        #layout---
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        vbox.Add(btnStart_BS_fabricationHand, 0, wx.EXPAND)

        self.help = wx.TextCtrl(
            self.panel, -1, "", style=wx.TE_MULTILINE | wx.TE_READONLY, size=(250, 100)
        )
        vbox.Add(self.help, 1, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

    def Onbtnhelp_nbdigits(self, event):
        helpstring = "a remplir"
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_filepathout(self, event):
        helpstring = "a remplir"
        self.help.SetValue(str(helpstring))

    def Onfilexyz_help(self, event):
        helpstring = "file xyz : full path to file xy with 3 columns (imagefile_index x y)"
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_nx(self, event):
        helpstring = 'Number of images (points) per line along the "fast axis" (X (resp Y) axis is "fast axis" is set to "x" resp. "y"))'
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_ny(self, event):
        helpstring = 'Number of images (points) per column along the "slow axis"'
        self.help.SetValue(str(helpstring))

    def Onbtnhelp_xfast(self, event):
        helpstring = 'sample direction for fast motor axis: "x"or "y"'
        self.help.SetValue(str(helpstring))

    #     def Onbtnhelp_yfast(self, event):
    #         helpstring = 'sample direction for slow motor axis'
    #         self.help.SetValue(str(helpstring))
    def Onbtnhelp_xstep(self, event):
        helpstring = "steps (Dx,Dy) along resp. fast and slow axes in micrometer. Dx and Dy can be negative.\n"
        helpstring += (
            "By increasing image index position along fast axis increases by Dx.\n"
        )
        helpstring += "Between each line position along slow axis increases by Dy"
        self.help.SetValue(str(helpstring))

    #     def Onbtnhelp_ystep(self, event):
    #         helpstring = 'a remplir'
    #         self.help.SetValue(str(helpstring))
    #     def Onbtnhelp_indimg(self, event):
    #         helpstring = 'a remplir'
    #         self.help.SetValue(str(helpstring))
    def OnbtnBrowse_filepathout(self, event):
        folder = wx.DirDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            self.list_txtctrl_manual[2].SetValue(folder.GetPath())

    def OnbtnChange_indimg(self, event):
        PSboard = New_indimg(None, -1, "New indimg", self.list_txtctrl_manual)
        PSboard.Show(True)

    def start_manualXYZ(self, event):
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

        prefix = str(self.parent.list_txtctrl[2].GetValue())

        outfilename = str(self.list_txtctrl_manual[0].GetValue())

        if check == 1:
            # writing filexyz with xy for map sample description in
            MGFS.build_xy_list_by_hand( outfilename, nx, ny, xfast, yfast, xstep, ystep,
                        startindex=int(self.parent.list_txtctrl[5].GetValue()),
                        lastindex=int(self.parent.list_txtctrl[6].GetValue()), )

            self.parent.list_txtctrl[10].SetValue(outfilename)

            wx.MessageBox(
                "Operation Successful! \t \t Filexyz created: %s" % outfilename
            )
        else:
            wx.MessageBox("Files's path or input datas are missing!")

        self.Destroy()


class New_indimg(wx.Frame):
    def __init__(self, parent, _id, title, list_txtctrl_hand):
        wx.Frame.__init__(
            self, parent, _id, title, wx.DefaultPosition, wx.Size(700, 700)
        )

        self.panel = wx.Panel(self)
        if WXPYTHON4:
            grid = wx.FlexGridSizer(3, 10, 10)
        else:
            grid = wx.FlexGridSizer(5, 3)
        grid.SetFlexibleDirection(wx.HORIZONTAL)

        self.list_txtctrl_new = []
        self.list_indimg = ["Nbpicture1", "Nblastpicture", "increment"]

        for kk, elem in enumerate(self.list_indimg):
            grid.Add(wx.StaticText(self.panel, -1, elem))
            self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))
            self.list_txtctrl_new.append(self.txtctrl)
            grid.Add(self.txtctrl)
            grid.Add(wx.Button(self.panel, kk + 10, "?", size=(25, 25)))

        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nbpicture1(self), id=10)
        self.Bind(
            wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nblastpicture(self), id=11
        )
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_increment(self), id=12)

        btnOK = wx.Button(self.panel, -1, "OK")
        grid.Add(btnOK)
        btnOK.Bind(wx.EVT_BUTTON, lambda event: self.OnbtnOK(self, list_txtctrl_hand))

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        self.help = wx.TextCtrl(
            self.panel, -1, "", style=wx.TE_MULTILINE | wx.TE_READONLY, size=(100, 300)
        )
        vbox.Add(self.help, 0, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

    def Onbtnhelp_nbpicture1(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnhelp_nblastpicture(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnhelp_increment(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def OnbtnOK(self, event, list_txtctrl_hand):
        modgraph.nbpicture1 = int(self.list_txtctrl_new[0].GetValue())
        modgraph.nblastpicture = int(self.list_txtctrl_new[1].GetValue())
        modgraph.increment = int(self.list_txtctrl_new[2].GetValue())
        list_txtctrl_hand[6].SetValue(
            str(
                "range("
                + str(self.list_txtctrl_new[0].GetValue())
                + ","
                + str(self.list_txtctrl_new[1].GetValue())
                + ","
                + str(self.list_txtctrl_new[2].GetValue())
                + ")"
            )
        )
        self.Destroy()


class SetParameters_BuildSummary_fabricationImage(wx.Frame):
    """
    Non used GUI class to extract in a clever way map parameters ...
    """
    def __init__(self, parent, _id, title, objet_BSi, manag, list_txtctrl):
        wx.Frame.__init__(
            self, parent, _id, title, wx.DefaultPosition, wx.Size(1000, 900)
        )

        self.panel = wx.Panel(self)
        if WXPYTHON4:
            grid = wx.FlexGridSizer(4, 10, 10)
        else:
            grid = wx.FlexGridSizer(6, 4)

        grid.SetFlexibleDirection(wx.HORIZONTAL)

        self.list_txtctrl_image = []

        for kk, elem in enumerate(objet_BSi.list_txtparamBSi):
            if kk > 1 and kk < 8:
                if kk != 6:
                    grid.Add(wx.StaticText(self.panel, -1, elem))
                    self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))
                    self.txtctrl.SetValue(str(objet_BSi.list_valueparamBSi[kk]))
                    self.list_txtctrl_image.append(self.txtctrl)
                    grid.Add(self.txtctrl)
                    if kk == 2 or kk == 7:
                        grid.Add(wx.Button(self.panel, kk + 10, "Browse"))
                    else:
                        nothing = wx.StaticText(self.panel, -1, "")
                        grid.Add(nothing)
                    grid.Add(wx.Button(self.panel, kk + 18, "?", size=(25, 25)))
                else:
                    grid.Add(wx.StaticText(self.panel, -1, elem))
                    self.txtctrl = wx.TextCtrl(self.panel, -1, "", size=(500, 25))
                    self.txtctrl.SetValue(str(objet_BSi.list_valueparamBSi[kk]))
                    self.list_txtctrl_image.append(self.txtctrl)
                    grid.Add(self.txtctrl)
                    grid.Add(wx.Button(self.panel, 26, "Change Indimg"))
                    nothing = wx.StaticText(self.panel, -1, "")
                    grid.Add(nothing)
            else:
                self.list_txtctrl_image.append("")

        print("list_txt_image", self.list_txtctrl_image)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filepathout(self), id=20)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_fileprefix(self), id=21)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filesuffix(self), id=22)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_nbdigits(self), id=23)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnhelp_filepathim(self), id=25)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Onbtnchangeindimg(self), id=26)
        self.Bind(
            wx.EVT_BUTTON, lambda event: self.OnbtnBrowse_filepathout(self), id=12
        )
        self.Bind(wx.EVT_BUTTON, lambda event: self.OnbtnBrowse_filepathim(self), id=17)

        btnStart_BS_fabricationImage = wx.Button(self.panel, -1, "START")
        grid.Add(btnStart_BS_fabricationImage)
        btnStart_BS_fabricationImage.Bind(
            wx.EVT_BUTTON,
            lambda event: self.OnbtnStart_BS_fabricationImage(
                self, manag, objet_BSi, list_txtctrl
            ),
        )

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid, 0, wx.EXPAND)
        self.help = wx.TextCtrl(
            self.panel, -1, "", style=wx.TE_MULTILINE | wx.TE_READONLY, size=(100, 300)
        )
        vbox.Add(self.help, 0, wx.EXPAND)

        self.panel.SetSizer(vbox, wx.EXPAND)

    def Onbtnhelp_fileprefix(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnhelp_filesuffix(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnhelp_filepathim(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnhelp_filepathout(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnhelp_nbdigits(self, event):
        txt = "a remplir"
        self.help.SetValue(str(txt))

    def Onbtnchangeindimg(self, event):
        PSboard = New_indimg(None, -1, "Enter indimg", self.list_txtctrl_image)
        PSboard.Show(True)

    def OnbtnBrowse_filepathim(self, event):
        folder = wx.DirDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            self.list_txtctrl_image[7].SetValue(folder.GetPath())

    def OnbtnBrowse_filepathout(self, event):
        folder = wx.DirDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            self.list_txtctrl_image[2].SetValue(folder.GetPath())

    def OnbtnStart_BS_fabricationImage(self, event, manag, objet_BSi, list_txtctrl):
        manag.Activate_BuildSummary()
        manag.matrice_callfunctions[2, 2] = 2

        check = 1
        for k in range(2, 8):
            if k == 5:
                objet_BSi.list_valueparamBSi[k] = int(
                    self.list_txtctrl_image[k].GetValue()
                )
                PAR.number_of_digits_in_image_name = int(
                    self.list_txtctrl_image[k].GetValue()
                )
                list_txtctrl[k].SetValue(str(self.list_txtctrl_image[k].GetValue()))
            elif k != 6:
                objet_BSi.list_valueparamBSi[k] = str(
                    self.list_txtctrl_image[k].GetValue()
                )
            else:
                objet_BSi.list_valueparamBSi[k] = modgraph.indimg
                list_txtctrl[k].SetValue(
                    (
                        "range"
                        + "("
                        + str(modgraph.nbpicture1)
                        + ","
                        + str(modgraph.nblastpicture)
                        + ","
                        + str(modgraph.increment)
                        + ")"
                    )
                )
            if (
                objet_BSi.list_valueparamBSi[k] is None
                or objet_BSi.list_valueparamBSi[k] == "None"
                or objet_BSi.list_valueparamBSi[k] == ""
            ):
                check = 0

        objet_BSi.list_valueparamBSi[9] = objet_BSi.list_valueparamBSi[3]
        objet_BSi.list_valueparamBSi[10] = objet_BSi.list_valueparamBSi[4]

        if not os.path.exists(str(objet_BSi.list_valueparamBSi[2]) + "xyz_"):
            os.makedirs(str(objet_BSi.list_valueparamBSi[2] + "xyz_"))
        modgraph.outfilenamexy = str(objet_BSi.list_valueparamBSi[2]) + "xyz_"
        if check == 1:
            objet_BSi.list_valueparamBSi[8] = MGFS.get_xyzech(
                objet_BSi.list_valueparamBSi[7],
                objet_BSi.list_valueparamBSi[3],
                modgraph.indimg,
                objet_BSi.list_valueparamBSi[4],
                modgraph.outfilenamexy,
            )
            list_txtctrl[1].SetValue(str(objet_BSi.list_valueparamBSi[8]))
            wx.MessageBox(
                "Operation Successful! Files created here: %s"
                % (modgraph.outfilenamexy)
            )
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
        initialparameters_dict["(stepX, stepY) microns"],
    ]
    return list_valueparam_BS


initialparameters = {}

print("LaueToolProjectFolder", LaueToolsProjectFolder)

MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "CuSi")

print("MainFolder", MainFolder)

initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")

initialparameters["file xyz"] = os.path.join(
    MainFolder, "fitfiles", "CuSi_xy_0_to_14.dat"
)
initialparameters["IndexRefine PeakList Prefix"] = "cusi_"
initialparameters["IndexRefine PeakList Suffix"] = ".fit"

initialparameters["Map shape"] = (5, 3)  # (nb lines, nb images per line)

initialparameters["(stepX, stepY) microns"] = (1.0, 1.0)
initialparameters["Material"] = "Si"
initialparameters["stiffness file"] = os.path.join(MainFolder, "si.stf")

initialparameters["nbdigits"] = 4
initialparameters["startingindex"] = 0
initialparameters["finalindex"] = 5
initialparameters["stepindex"] = 1
initialparameters["fast axis: x or y"] = "x"

list_valueparamBS = fill_list_valueparamBS(initialparameters)

DEFAULT_FILE = os.path.join(
    initialparameters["IndexRefine PeakList Folder"], "nanox2_400_0000.fit"
)

if __name__ == "__main__":



    BuildSummaryApp = wx.App()
    BSFrame = MainFrame_BuildSummary(
        None, -1, "Build Summary Parameters Board", list_valueparamBS
    )
    BSFrame.Show(True)
    BuildSummaryApp.MainLoop()
