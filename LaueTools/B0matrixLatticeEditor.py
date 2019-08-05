"""
module containing a GUI class to compute lattice parameters in both spaces (real and reciprocal)
"""
import os
import sys
import re

import numpy as np

import wx
if sys.version_info.major == 3:
    from . import CrystalParameters as CP
else:
    import CrystalParameters as CP

# --- ---------------------   UB Matrix Editor
class B0MatrixEditor(wx.Frame):
    """
    GUI class to compute lattice parameters in both spaces (real and reciprocal)
    """
    
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

        self.panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(715, 995), pos=(5, 5))

        dr = wx.StaticText(self.panel, -1, "Direct Space", (60, 10))
        dr.SetFont(font3)

        self.rbeditor = wx.RadioButton(self.panel, -1, "Text Editor Input", (25, 40), style=wx.RB_GROUP)
        wx.StaticText(self.panel, -1, "[[#,#,#],[#,#,#],[#,#,#]]", (25, 60))
        self.text = wx.TextCtrl(self.panel, 1000, "", pos=(25, 85), size=(250, 90),
                                        style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        self.text.SetFocus()
        
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

        self.rblatticeparam_direct = wx.RadioButton( self.panel, -1, "Real Lattice parameters Input", (25, 320))
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
        # letter 's' at the end of widgets name stands for 'star' corresponding to reciprocal space variable

        hshift = 350
        drs = wx.StaticText(self.panel, -1, "Reciprocal Space", (hshift + 60, 10))
        drs.SetFont(font3)

        self.rbeditors = wx.RadioButton(self.panel, -1, "Text Editor Input Bmatrix", (hshift + 25, 40))
        wx.StaticText(self.panel, -1, "[[#,#,#],[#,#,#],[#,#,#]]", (hshift + 25, 60))
        self.texts = wx.TextCtrl(self.panel, 1003, "", pos=(hshift + 25, 85),
                            size=(250, 90), style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        self.texts.SetFocus()

        self.rbelems = wx.RadioButton(self.panel, -1, "B0 Matrix Elements Input", (hshift + 25, 190))

        self.mat_a11s = wx.TextCtrl(self.panel, -1, "1", (hshift + 20, 220), size=(80, -1))
        self.mat_a12s = wx.TextCtrl(self.panel, -1, "0", (hshift + 120, 220), size=(80, -1))
        self.mat_a13s = wx.TextCtrl(self.panel, -1, "0", (hshift + 220, 220), size=(80, -1))
        self.mat_a21s = wx.TextCtrl(self.panel, -1, "0", (hshift + 20, 250), size=(80, -1))
        self.mat_a22s = wx.TextCtrl(self.panel, -1, "1", (hshift + 120, 250), size=(80, -1))
        self.mat_a23s = wx.TextCtrl(self.panel, -1, "0", (hshift + 220, 250), size=(80, -1))
        self.mat_a31s = wx.TextCtrl(self.panel, -1, "0", (hshift + 20, 280), size=(80, -1))
        self.mat_a32s = wx.TextCtrl(self.panel, -1, "0", (hshift + 120, 280), size=(80, -1))
        self.mat_a33s = wx.TextCtrl(self.panel, -1, "1", (hshift + 220, 280), size=(80, -1))

        self.rblatticeparam_reciprocal = wx.RadioButton( self.panel, -1, "Reciprocal Lattice parameters Input",
                                                                            (hshift + 25, 320))
        wx.StaticText(self.panel, -1, "a*", (hshift + 40, 350))
        self.astar = wx.TextCtrl(self.panel, -1, "1", (hshift + 20, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "b*", (hshift + 120, 350))
        self.bstar = wx.TextCtrl(self.panel, -1, "1", (hshift + 100, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "c*", (hshift + 200, 350))
        self.cstar = wx.TextCtrl(self.panel, -1, "1", (hshift + 180, 380), size=(60, -1))
        wx.StaticText(self.panel, -1, "1/Angst.", (hshift + 260, 385))
        wx.StaticText(self.panel, -1, "alpha*", (hshift + 20, 410))
        self.alphastar = wx.TextCtrl(self.panel, -1, "90", (hshift + 20, 440), size=(60, -1))
        wx.StaticText(self.panel, -1, "beta*", (hshift + 100, 410))
        self.betastar = wx.TextCtrl(self.panel, -1, "90", (hshift + 100, 440), size=(60, -1))
        wx.StaticText(self.panel, -1, "gamma*", (hshift + 180, 410))
        self.gammastar = wx.TextCtrl(self.panel, -1, "90", (hshift + 180, 440), size=(60, -1))
        wx.StaticText(self.panel, -1, "deg.", (hshift + 260, 445))

        self.rbeditors.SetValue(True)

        buttonConvert = wx.Button(self.panel, 105, "Compute && Convert", pos=(150, 480), size=(400, 50))
        # buttonload.SetFont(font3)
        
        self.text.Bind(wx.EVT_TEXT, self.OnTextChanged, id=1000)
        self.text.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.texts.Bind(wx.EVT_TEXT, self.OnTextChanged, id=1003)
        self.texts.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        buttonConvert.Bind(wx.EVT_BUTTON, self.OnConvert, id=105)

        #tips
        tiprealB = 'Columns are real unit cell vector basis, a,b,c expressed in LaueTools frame\n'
        tiprealB += '[ ax   bx  cx]\n'
        tiprealB += '[ ay   by  cy]\n'
        tiprealB += '[ az   bz  cz]\n'
        dr.SetToolTipString(tiprealB)
        self.rbeditor.SetToolTipString(tiprealB)
        self.rbelem.SetToolTipString(tiprealB)

        listtctrl = [self.mat_a11,self.mat_a12,self.mat_a13,
                    self.mat_a11,self.mat_a11,self.mat_a11,
                    self.mat_a11,self.mat_a11,self.mat_a11]

        tiprecipB = 'B0 matrix: Columns are reciprocal unit cell vector basis, a*,b*,c* expressed in LaueTools frame\n'
        tiprecipB += '[ a*x   b*x  c*x]\n'
        tiprecipB += '[ a*y   b*y  c*y]\n'
        tiprecipB += '[ a*z   b*z  c*z]\n'
        drs.SetToolTipString(tiprecipB)
        self.rbeditors.SetToolTipString(tiprecipB)
        self.rbelems.SetToolTipString(tiprecipB)

        buttonConvert.SetToolTipString('Compute lattice parameters and matrices in both spaces')

        #-------------- GUI middle and second part ----------------------------------
        wx.StaticText(self.panel, -1, "_" * 90, (25, 530))

        self.Build_list_of_B()
        vshift = 80
        buttonread = wx.Button( self.panel, 101, "Look", pos=(20, vshift + 500), size=(60, 25))
        # buttonread.SetFont(font3)
        
        self.comboRot = wx.ComboBox( self.panel, 6, "Identity", (100, vshift + 500), choices=self.list_of_Vect)
        
        wx.StaticText(self.panel, -1, "in B0 matrix", (300, vshift + 500 + 5))

        buttonsave = wx.Button(self.panel, 102, "Save", pos=(20, vshift + 540), size=(60, 25))
        # buttonsave.SetFont(font3)
        
        wx.StaticText(self.panel, -1, "B0 matrix in", (110, vshift + 545))
        self.filenamesave = wx.TextCtrl( self.panel, -1, "*.b0mat", (200, vshift + 540), size=(100, 25))
        wx.StaticText(self.panel, -1, "(on hard disk)", (300, vshift + 545))

        buttonstore = wx.Button(self.panel, 103, "Store", pos=(20, vshift + 580), size=(60, 25))
        # buttonstore.SetFont(font3)
        wx.StaticText(self.panel, -1, "B0 matrix in", (110, vshift + 585))
        self.filenamestore = wx.TextCtrl( self.panel, -1, "", (200, vshift + 580), size=(100, 25))
        wx.StaticText(self.panel, -1, "(will appear in a*,b*,c* simulation menu)", (300, vshift + 585))

        buttonload = wx.Button( self.panel, 104, "Load", pos=(20, vshift + 620), size=(60, 25))
        # buttonload.SetFont(font3)
        wx.StaticText(self.panel, -1, "B0 Matrix from simple ASCII file",
                                                                    (100, vshift + 625))

        buttonread.Bind(wx.EVT_BUTTON, self.OnLookMatrix, id=101)
        self.comboRot.Bind(wx.EVT_COMBOBOX, self.EnterComboRot, id=6)
        buttonsave.Bind(wx.EVT_BUTTON, self.OnSaveFile, id=102)
        buttonstore.Bind(wx.EVT_BUTTON, self.OnStoreFile, id=103)
        buttonload.Bind(wx.EVT_BUTTON, self.DoOpenFile, id=104)

        # -------------    Lower and third GUI part   ----------------
        self.vshift2 = vshift + 535 + 60

        wx.StaticText(self.panel, -1, "_" * 90, (25, self.vshift2 + 50))

        st000 = wx.StaticText( self.panel, -1, "Crystal Unit Cell", (250, self.vshift2 + 70))
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
        wx.StaticText(self.panel, -1, "U        .", (pos0x + 1 * self.stepx, self.vshift2 + postext))
        wx.StaticText(self.panel, -1, "B        .", (pos0x + 2 * self.stepx, self.vshift2 + postext))
        wx.StaticText(self.panel, -1, "Dc       .", (pos0x + 3 * self.stepx, self.vshift2 + postext))
        wx.StaticText(self.panel, -1, "G*", (pos0x + 3 * self.stepx + 100, self.vshift2 + postext))
        wx.StaticText(self.panel, -1, "Extinc", (pos0x + 3 * self.stepx + 150, self.vshift2 + postext))

        self.poscombos = 120

        # combos of unit cell reference parameters
        self.DisplayCombosUnitCell()

        posv = (self.poscombos + self.vshift2) + 40

        storeCellbtn = wx.Button(self.panel, -1, "Store Unit as new material", pos=(20, posv), size=(80, 25))
        wx.StaticText(self.panel, -1, "in", (110, posv + 5))
        self.namestore = wx.TextCtrl(self.panel, -1, "", (150, posv), size=(100, 25))
        wx.StaticText(self.panel, -1, "(Will appear in Elem list for classical indexation)", (280, posv + 5))

        quitbtn = wx.Button(self.panel, -1, "Quit", (575, self.vshift2 + 180), size=(120, 100))

        quitbtn.Bind(wx.EVT_BUTTON, self.OnQuit)
        storeCellbtn.Bind(wx.EVT_BUTTON, self.OnStoreRefCell)

        #---tooltips
        tipstore = 'Store in LaueToolsGUI a new element from Real lattice parameter above'
        storeCellbtn.SetToolTipString(tipstore)
        self.namestore.SetToolTipString(tipstore)

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

        self.comboDa = wx.ComboBox(self.panel, -1, "Identity", (50, self.vshift2 + self.poscombos),
                                                choices=self.list_of_Strain, size=(100, -1))
        self.comboU = wx.ComboBox(self.panel, -1, "Identity", (50 + self.stepx, self.vshift2 + self.poscombos),
                                                choices=self.list_of_Rot, size=(100, -1))
        self.comboB = wx.ComboBox(self.panel, -1, "Identity", (50 + 2 * self.stepx, self.vshift2 + self.poscombos),
                                                choices=self.list_of_Vect, size=(100, -1))
        self.comboDc = wx.ComboBox( self.panel, -1, "Identity", (50 + 3 * self.stepx, self.vshift2 + self.poscombos),
                                                choices=self.list_of_Strain, size=(100, -1))
        self.comboExtinc = wx.ComboBox( self.panel, -1, "NoExtinction", (50 + 4 * self.stepx + 40, self.vshift2 + self.poscombos),
                                    choices=list(self.parent.dict_Extinc.keys()), size=(100, -1))

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

        UnitCellParameters = []
        for par_txtctrl in [self.a, self.b, self.c, self.alpha, self.beta,self.gamma]:
            UnitCellParameters.append(float(par_txtctrl.GetValue()))

        print("UnitCellParameters", UnitCellParameters)
        print("Stored key_material", [key_material, UnitCellParameters, struct_extinc])
        self.parent.dict_Materials[key_material] = [key_material, UnitCellParameters, struct_extinc]

    def StatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-5, -2, -1])

    def ToggleStatusBar(self, evt):
        if self.statusbar.IsShown():
            self.statusbar.Hide()
        else:
            self.statusbar.Show()

    def OnTextChanged(self, evt):
        self.modify = True
        evt.Skip()

    def OnKeyDown(self, evt):
        keycode = evt.GetKeyCode()
        if keycode == wx.WXK_INSERT:
            if not self.replace:
                self.statusbar.SetStatusText("INS", 2)
                self.replace = True
            else:
                self.statusbar.SetStatusText("", 2)
                self.replace = False
        evt.Skip()

    def OnSaveFile(self, evt):
        """
        Saves the matrix in ASCII file on Hard Disk from the ASCII editor or the 9 input elements 
        """
        self.last_name_saved = self.filenamesave.GetValue()

        if self.last_name_saved:

            try:
                if self.rbeditor.GetValue():
                    paramraw = self.texts.GetValue()

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
                    if nbval != 9:
                        txt = "Something wrong, I can't read this matrix %s \n."
                        txt += "It doesn't contain 9 elements with float type ..."%paramraw
                        print(txt)

                        wx.MessageBox(txt, "VALUE ERROR")
                        return

                    allm = []
                    ind_elem = 0
                    for i in range(3):
                        for j in range(3):
                            floatval = listelem[ind_elem]
                            allm.append(floatval)
                            ind_elem += 1

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
            wx.MessageBox("Please provide a name for the matrix file to be saved...!", "VALUE ERROR")
            return

    def OnStoreFile(self, evt):
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
                paramraw = str(self.texts.GetValue())

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
                if nbval != 9:
                    txt = "Something wrong, I can't read this matrix %s \n."
                    txt += "It doesn't contain 9 elements with float type ..."%paramraw
                    print(txt)

                    wx.MessageBox(txt, "VALUE ERROR")
                    return

                mat = np.zeros((3, 3))
                ind_elem = 0
                for i in range(3):
                    for j in range(3):
                        floatval = listelem[ind_elem]
                        mat[i][j] = floatval
                        ind_elem += 1

                self.parent.dict_Rot[self.last_name_stored] = mat

            self.statusbar.SetStatusText(
                os.path.basename(self.last_name_stored) + " stored", 0
            )
            self.DisplayCombosUnitCell()

        else:
            print("No name input !!!")

    def DoOpenFile(self, evt):
        wcd = "All files(*)|*|Matrix files(*.b0mat)|*.b0mat"
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

    def EnterComboRot(self, evt):
        """
        in UB matrix editor
        """
        item = evt.GetSelection()
        self.CurrentMat = self.list_of_Rot[item]
        evt.Skip()

    # def EnterComboDa(self, evt):
    # """
    # in UB matrix editor

    # """
    # item = evt.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # evt.Skip()

    # def EnterComboU(self, evt):
    # """
    # in UB matrix editor

    # """
    # item = evt.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # evt.Skip()

    # def EnterComboDc(self, evt):
    # """
    # in UB matrix editor

    # """
    # item = evt.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # evt.Skip()

    # def EnterComboB(self, evt):
    # """
    # in UB matrix editor

    # """
    # item = evt.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # evt.Skip()

    # def EntercomboExtinc(self, evt):
    # """
    # in UB matrix editor

    # """
    # item = evt.GetSelection()
    # self.CurrentMat = self.list_of_Rot[item]
    # evt.Skip()

    def OnLookMatrix(self, evt):
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

    def OnConvert(self, evt):
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

    def OnQuit(self, evt):
        dlg = wx.MessageDialog(
            self, 'To use stored UB Matrices, click on "refresh choices" button before.'
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.Close()
            evt.Skip()
        else:
            evt.Skip()

if __name__ == "__main__":
    import dict_LaueTools as DictLT

    class parentB0Editor(wx.Frame):
        """gui class to test
        """
        def __init__(self, parent, _id, title):

            wx.Frame.__init__(self, parent, _id, title)
            self.dict_Vect = DictLT.dict_Vect
            self.dict_Transforms = DictLT.dict_Transforms
            self.dict_Extinc = DictLT.dict_Extinc
            self.dict_Rot = DictLT.dict_Rot
            self.dict_Materials = DictLT.dict_Materials

            Editorframe = B0MatrixEditor(
                    self, -1, "editor"
                )
            Editorframe.Show(True)

    EditorApp = wx.App()
    GUIFrame = parentB0Editor(
        None,
        -1,
        "BO matrix and lattice parameters computer")

    GUIFrame.Show()

    EditorApp.MainLoop()
