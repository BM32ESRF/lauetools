import os
import sys
import re

import numpy as np

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

if sys.version_info.major == 3:
    from .. import generaltools as GT
else:
    import generaltools as GT
    
DEG = np.pi/180.
# --- ---------------  Matrix Editor
class MatrixEditor_Dialog(wx.Frame):
    """
    class to handle edition of matrices

    parent lust have:
    dict_Rot
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

        self.rbeditor = wx.RadioButton(panel, -1, "Text Editor Input", (25, 10),
                                                                                style=wx.RB_GROUP)
        wx.StaticText(panel, -1, "[[#,#,#],[#,#,#],[#,#,#]]", (40, 25))

        self.text = wx.TextCtrl(panel, -1, "", pos=(50, 45), size=(250, 85),
                                                    style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        self.text.SetFocus()
        self.text.Bind(wx.EVT_TEXT, self.OnTextChanged)
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

        buttoncomputemat_1 = wx.Button(panel, 555, "Compute", pos=(330, 180), size=(80, 25))
        wx.StaticText(panel, -1, "from axis angles", (420, 185))
        buttoncomputemat_1.Bind(wx.EVT_BUTTON, self.OnComputeMatrix_axisangles, id=555)
        buttoncomputemat_2 = wx.Button(panel, 556, "Compute", pos=(330, 210), size=(80, 25))
        wx.StaticText(panel, -1, "from axis vector", (420, 215))
        buttoncomputemat_2.Bind(wx.EVT_BUTTON, self.OnComputeMatrix_axisvector, id=556)

        buttonread = wx.Button(panel, 101, "Look", pos=(20, 300), size=(60, 25))
        # buttonread.SetFont(font3)
        buttonread.Bind(wx.EVT_BUTTON, self.OnLookMatrix, id=101)
        self.comboRot = wx.ComboBox(panel, 6, "Identity", (100, 300), choices=self.list_of_Rot)
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
        wx.StaticText(panel, -1,
                "Matrix from saved file in simple ASCII text editor format", (100, 425))

        buttonXMASload = wx.Button(panel, 105, "Read XMAS", pos=(20, 460), size=(100, 25))
        # buttonload.SetFont(font3)
        buttonXMASload.Bind(wx.EVT_BUTTON, self.OnLoadXMAS_INDfile, id=105)
        wx.StaticText(panel, -1, ".IND file", (130, 465))
        buttonXMASload.Disable()

        buttonconvert = wx.Button(panel, 106, "Convert", pos=(200, 460), size=(70, 25))
        # buttonload.SetFont(font3)
        buttonconvert.Bind(wx.EVT_BUTTON, self.OnConvert, id=106)
        wx.StaticText(panel, -1, "XMAS to LaueTools lab. frame", (290, 465))
        buttonconvert.Disable()

        buttonconvert2 = wx.Button(panel, 107, "Convert", pos=(200, 510), size=(70, 25))
        buttonconvert2.Bind(wx.EVT_BUTTON, self.OnConvertlabtosample, id=107)
        wx.StaticText(panel, -1, "LaueTools: from lab. to sample frame", (290, 515))
        buttonconvert2.Disable()

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

    def ToggleStatusBar(self, _):
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

    def OnComputeMatrix_axisangles(self, _):
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

    def OnComputeMatrix_axisvector(self, _):
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

    def OnSaveFile(self, _):
        """
        Saves the matrix in ASCII editor or 9 entried elements on Hard Disk
        """
        self.last_name_saved = self.filenamesave.GetValue()

        if self.last_name_saved:

            try:
                # radio button on text ascii editor
                if self.rbeditor.GetValue():
                    f = open(self.last_name_saved, "w")
                    text = self.text.GetValue()
                    f.write(text)
                    f.close()
                # 9 elements txtctrl editors
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
                    text = ("[[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f],\n[%.17f,%.17f,%.17f]]"
                        % (m11, m12, m13, m21, m22, m23, m31, m32, m33))
                    _file.write(text)
                    _file.close()

                self.statusbar.SetStatusText(
                    os.path.basename(self.last_name_saved) + " saved", 0)
                self.modify = False
                self.statusbar.SetStatusText("", 1)

                fullname = os.path.join(os.getcwd(), self.last_name_saved)
                wx.MessageBox("Matrix saved in %s" % fullname, "INFO")

            except IOError as error:

                dlg = wx.MessageDialog(self, "Error saving file\n" + str(error))
                dlg.ShowModal()
        else:
            print("No name input")

    def OnStoreMatrix(self, _):
        r"""
        Stores the matrix from the ASCII editor or the 9 entried elements
        in main list of orientation matrix for further simulation
        """
        self.last_name_stored = self.filenamestore.GetValue()

        if self.last_name_stored:

            # read 9 elements
            if not self.rbeditor.GetValue():
                m11 = float(self.mat_a11.GetValue())
                m12 = float(self.mat_a12.GetValue())
                m13 = float(self.mat_a13.GetValue())

                m21 = float(self.mat_a21.GetValue())
                m22 = float(self.mat_a22.GetValue())
                m23 = float(self.mat_a23.GetValue())

                m31 = float(self.mat_a31.GetValue())
                m32 = float(self.mat_a32.GetValue())
                m33 = float(self.mat_a33.GetValue())

                _allm = np.array([
                    [m11, m12, m13],
                    [m21, m22, m23],
                    [m31, m32, m33]])
                if np.linalg.det(_allm)<0:
                    txt = "Matrix is not direct (det(UB)<0)"
                    print(txt)

                    wx.MessageBox(txt, "txt")
                    return

                self.parent.dict_Rot[self.last_name_stored] = _allm.tolist()

            # read ASCII editor
            else:
                paramraw = str(self.text.GetValue())

                listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", paramraw)
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

                _allm = np.array(mat)
                if np.linalg.det(_allm)<0:
                    txt = "Matrix is not direct (det(UB)<0)"
                    print(txt)

                    wx.MessageBox(txt, "ERROR")
                    return

                self.parent.dict_Rot[self.last_name_stored] = mat

            self.statusbar.SetStatusText(
                os.path.basename(self.last_name_stored) + " stored", 0)

        else:
            print("No name input")

    def OnOpenMatrixFile(self, _):
        wcd = "All files(*)|*|Matrix files(*.mat)|*.mat"
        _dir = os.getcwd()
        open_dlg = wx.FileDialog(
                                self,
                                message="Choose a file",
                                defaultDir=_dir,
                                defaultFile="",
                                wildcard=wcd,
                                style=wx.OPEN | wx.CHANGE_DIR)
        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()

            try:
                _file = open(path, "r")
                paramraw = _file.read()
                _file.close()

                listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", paramraw)
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

                    wx.MessageBox(txt, "ERROR")
                    return

                mat = np.zeros((3, 3))
                ind_elem = 0
                for i in range(3):
                    for j in range(3):
                        floatval = listelem[ind_elem]
                        mat[i][j] = floatval
                        ind_elem += 1

                _allm = np.array(mat)
                if np.linalg.det(_allm)<0:
                    txt = "Matrix is not direct (det(UB)<0)"
                    print(txt)

                    wx.MessageBox(txt, "ERROR")
                    return

                self.text.Clear()
                self.text.WriteText(str(mat))
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
        in matrix editor
        """
        item = evt.GetSelection()
        self.CurrentMat = self.list_of_Rot[item]
        evt.Skip()

    def OnLookMatrix(self, _):
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

    def OnLoadXMAS_INDfile(self, _):
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
                                style=wx.OPEN | wx.CHANGE_DIR)
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
                            listlines[posmatrix + 1].split(), dtype=float)
                        secondrow = np.array(
                            listlines[posmatrix + 2].split(), dtype=float)
                        thirdrow = np.array(
                            listlines[posmatrix + 3].split(), dtype=float)
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
                            listlines[posmatrix + 1].split(), dtype=float)
                        secondrow = np.array(
                            listlines[posmatrix + 2].split(), dtype=float)
                        thirdrow = np.array(
                            listlines[posmatrix + 3].split(), dtype=float)
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

    def OnConvert(self, _):
        """ old functionality...
        """
        wx.MessageBox('This conversion has been deleted', 'Info')

        # text = str(self.text.GetValue())
        # tu = text.replace("[", "").replace("]", "")
        # ta = tu.split(",")
        # to = [float(elem) for elem in ta]
        # matrix = np.array([to[:3], to[3:6], to[6:]])

        # dlg = wx.MessageDialog(
        #     self,
        #     "New matrix will be displated in Text Editor Input\nConversion will use current calibration, namely xbet: %.4f \n  \nDo you want to continue ?"
        #     % self.parent.defaultParam[3])
        # if dlg.ShowModal() == wx.ID_OK:

        #     evt.Skip()

        #     # newmatrix = F2TC.matxmas_to_OrientMatrix(matrix, LaueToolsframe.defaultParam)
        #     newmatrix = FXL.convert_fromXMAS_toLaueTools(
        #         matrix, 5.6575, anglesample=40.0, xbet=self.parent.defaultParam[3])
        #     print("Matrix read in editor")
        #     print(matrix.tolist())
        #     print("Matrix as converted to Lauetools by F2TC")
        #     print(newmatrix.tolist())
        #     # matrix UB normalized
        #     # must be multiplied by lattic param*

        #     self.rbeditor.SetValue(True)
        #     self.text.SetValue(str(newmatrix.tolist()))
        # else:
        #     # TODO: advise how to set new calibration parameter(in calibration menu ?)
        #     evt.Skip()

    def OnConvertlabtosample(self, _):
        """
        qs= R ql    q expressed in sample frame(xs,ys,zs) = R * ql
        with ql being q expressed in lab frame(x,y,z)
        G=ha*+kb*+lc*
        ql = UB * G with UB orientation and strain matrix
        ie UB columns are a*, b*, c* expressed in x,y,z frame
        qs = R * UB * G
        R*UB in the orientation matrix in sample frame
        ie columns are a*, b*,c* expressed in xs,ys,zs frame

        Gs = R Gl    G expressed in sample frame(xs,ys,zs)  =  R * Gl
        with Gl being G expressed in lab frame

        qs = R*UB*invR  Gs means that R*UB*invR is the orientation matrix .
        From a*,b*,c* in xs,ys,zs to q in xs,ys,zs
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

        Rot = np.array([[np.cos(anglesample), 0, np.sin(anglesample)],
                        [0, 1, 0],
                        [-np.sin(anglesample), 0, np.cos(anglesample)]])
        invRot = np.linalg.inv(Rot)
        UBs = np.dot(Rot, UB)
        print("UB as read in editor")
        print(UB.tolist())
        print("UB converted in lauetools sample frame(From G=ha*+kb*+lc* with a*,b* and c* "
            "expressed in lab frame to q expressed in sample frame)")
        print(UBs.tolist())
        print("UB converted in XMAS-like sample frame")
        print(np.dot(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), UBs).tolist())
        print("UB converted in lauetools sample frame(From G=ha*+kb*+lc* with a*,b* "
            "and c* expressed in sample frame to q expressed in sample frame)")
        print(np.dot(UBs, invRot).tolist())

        # matrix UB normalized
        # must be multiplied by lattic param*

        self.rbeditor.SetValue(True)
        self.text.SetValue(str(UBs.tolist()))

    def OnQuit(self, evt):
        dlg = wx.MessageDialog(self,
            'To use stored Matrices in simulation boards that do not appear, click '
            'on "refresh choices" button before.')
        if dlg.ShowModal() == wx.ID_OK:
            self.Close()
            evt.Skip()
        else:
            evt.Skip()
