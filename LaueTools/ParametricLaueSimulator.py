
import os
import time, sys
import string

import numpy as np
import wx
import wx.lib.scrolledpanel as scrolled

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    def insertitem(*args):
        return wx.ListCtrl.InsertItem(*args)

    wx.ListCtrl.InsertStringItem = insertitem
    def setitem(*args):
        return wx.ListCtrl.SetItem(*args)

    wx.ListCtrl.SetStringItem = setitem

if sys.version_info.major == 3:
    from . import dict_LaueTools as DictLT
    from . SimulFrame import SimulationPlotFrame
    from . import generaltools as GT
    from . import lauecore as LAUE
    from . import LaueGeometry as F2TC
    from . import CrystalParameters as CP
    from . import ProportionalSplitter as PropSplit

else:
    import dict_LaueTools as DictLT
    from SimulFrame import SimulationPlotFrame
    import generaltools as GT
    import lauecore as LAUE
    import LaueGeometry as F2TC
    import CrystalParameters as CP
    import ProportionalSplitter as PropSplit


DEG = np.pi / 180.0


class TransformPanel(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.granparent = parent.GetParent().GetParent()

        print("granparent of TransformPanel", self.granparent)

        self.SelectGrains = {}

        # list Control for selected grains for SIMULATION
        font3 = self.granparent.font3

        titlemiddle = wx.StaticText(self, -1, "Transformations")
        titlemiddle.SetFont(font3)

        # rotation
        rottext = wx.StaticText(self, -1, "ROTATION              ")
        rottext.SetFont(font3)

        self.rb_rotId = wx.RadioButton(self, -1, "No rotation", style=wx.RB_GROUP)
        self.rb_rotId.SetValue(True)

        # [start, end, nb steps]
        self.trange1 = wx.StaticText(self, -1, "varying t range")

        self.tc_range_rot = wx.TextCtrl(self, -1, "[0, 10, 10]", size=(150, -1))

        self.rb_rotaxis = wx.RadioButton(self, -1, "Axis-angle Variation")

        self.axisrot = wx.StaticText(self, -1, "Axis")
        self.tc_Rot_axis = wx.TextCtrl(self, -1, "a[1, 1,1]", size=(150, -1))
        self.anglerot = wx.StaticText(self, -1, "Angle(deg)")
        self.tc_Rot_ang = wx.TextCtrl(self, -1, "0", size=(150, -1))

        self.rb_rotmatrix = wx.RadioButton(self, -1, "General Transform")

        self.rb_rotmatrix.Bind(wx.EVT_RADIOBUTTON, self.onEnableRotation)
        self.rb_rotaxis.Bind(wx.EVT_RADIOBUTTON, self.onEnableRotation)

        defaultmatrixtransform = "a[[1,0,0],[0,1,0],[0,0,1]]"
        self.tc_rotmatrix = wx.TextCtrl(
            self,
            1000,
            defaultmatrixtransform,
            size=(250, 100),
            style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER,
        )
        self.tc_rotmatrix.SetFocus()
        self.tc_rotmatrix.Bind(wx.EVT_TEXT, self.granparent.OnTextChanged)
        self.tc_rotmatrix.Bind(wx.EVT_KEY_DOWN, self.granparent.OnKeyDown)
        self.modify = False
        self.replace = False

        # Strain
        straintext = wx.StaticText(self, -1, "STRAIN                ")
        straintext.SetFont(font3)

        self.rb_strainId = wx.RadioButton(self, -1, "No strain", style=wx.RB_GROUP)
        self.rb_strainId.SetValue(True)

        trange2 = wx.StaticText(self, -1, "varying t range")
        self.tc_strainrange = wx.TextCtrl(self, -1, "[0, 10, 10]", size=(100, -1))

        self.rb_strainaxes = wx.RadioButton(self, -1, "Tensile axis")

        self.rb_strainaxes.Bind(wx.EVT_RADIOBUTTON, self.onEnableStrain)

        a1 = wx.StaticText(self, -1, "Axis 1")
        self.tc_axe1_axis = wx.TextCtrl(self, -1, "s[0, 0, 1]", size=(60, -1))
        f1 = wx.StaticText(self, -1, "factor")
        self.tc_axe1_factor = wx.TextCtrl(self, -1, "1.000", size=(140, -1))

        a2 = wx.StaticText(self, -1, "Axis 2")
        self.tc_axe2_axis = wx.TextCtrl(self, -1, "s[1, 1, 0]", size=(60, -1))
        f2 = wx.StaticText(self, -1, "factor")
        self.tc_axe2_factor = wx.TextCtrl(self, -1, "1.000", size=(140, -1))

        a3 = wx.StaticText(self, -1, "Axis 3")
        self.tc_axe3_axis = wx.TextCtrl(self, -1, "c[1, 1, 1]", size=(60, -1))
        f3 = wx.StaticText(self, -1, "factor")
        self.tc_axe3_factor = wx.TextCtrl(self, -1, "1.000", size=(140, -1))

        # self.rb_strainmatrix = wx.RadioButton(self, 200, 'Element matrix(a: absolute, s:sample, c:crystal)',(15, posstrain+160))
        # defaultmatrixstrain = 'a[[1, 0,0],[0, 1,0],[0, 0,1]]'
        # self.tc_strainmatrix = wx.TextCtrl(self, 1001, defaultmatrixstrain, pos =(50, posstrain+180),size =(250, 100), style = wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        # self.tc_strainmatrix.SetFocus()
        # self.tc_strainmatrix.Bind(wx.EVT_TEXT, self.OnTextChanged_strain, id = 1001)
        # self.tc_strainmatrix.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown_strain)
        # self.modify_strain = False
        # self.replace_strain = False

        buttontransform = wx.Button(self, -1, "Apply transforms", size=(150, 35))
        buttontransform.Bind(wx.EVT_BUTTON, self.granparent.OnApplytransform)
        buttontransform.SetFont(font3)

        self.granparent.transform_index = 0
        self.granparent.dict_transform = {}

        self.tooltips_transformpanel()

        h1box = wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add(rottext, 0)
        h1box.Add(self.rb_rotId, 0)

        h2box = wx.BoxSizer(wx.HORIZONTAL)
        h2box.Add(self.rb_rotaxis, 0)
        h2box.AddSpacer(15)
        h2box.Add(self.trange1, 0)
        h2box.AddSpacer(10)
        h2box.Add(self.tc_range_rot, 0)

        h4box = wx.BoxSizer(wx.HORIZONTAL)
        h4box.Add(self.axisrot, 0)
        h4box.AddSpacer(10)
        h4box.Add(self.tc_Rot_axis, 0)
        h4box.AddSpacer(15)
        h4box.Add(self.anglerot, 0)
        h4box.AddSpacer(10)
        h4box.Add(self.tc_Rot_ang, 0)

        h5box = wx.BoxSizer(wx.HORIZONTAL)
        h5box.Add(straintext, 0)
        h5box.Add(self.rb_strainId, 0)

        h6box = wx.BoxSizer(wx.HORIZONTAL)
        h6box.Add(self.rb_strainaxes, 0)
        h6box.AddSpacer(15)
        h6box.Add(trange2, 0)
        h6box.AddSpacer(10)
        h6box.Add(self.tc_strainrange, 0)

        haxis1 = wx.BoxSizer(wx.HORIZONTAL)
        haxis1.Add(a1)
        haxis1.Add(self.tc_axe1_axis)
        haxis1.Add(f1)
        haxis1.Add(self.tc_axe1_factor)

        haxis2 = wx.BoxSizer(wx.HORIZONTAL)
        haxis2.Add(a2)
        haxis2.Add(self.tc_axe2_axis)
        haxis2.Add(f2)
        haxis2.Add(self.tc_axe2_factor)

        haxis3 = wx.BoxSizer(wx.HORIZONTAL)
        haxis3.Add(a3)
        haxis3.Add(self.tc_axe3_axis)
        haxis3.Add(f3)
        haxis3.Add(self.tc_axe3_factor)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(titlemiddle)
        vbox.Add(h1box)
        vbox.Add(h2box)
        vbox.Add(h4box)
        vbox.Add(self.rb_rotmatrix)
        vbox.Add(self.tc_rotmatrix)
        vbox.Add(h5box)
        vbox.Add(h6box)
        vbox.Add(haxis1)
        vbox.Add(haxis2)
        vbox.Add(haxis3)
        vbox.Add(buttontransform, 0, wx.EXPAND)

        self.SetBackgroundColour("sky Blue")

        self.SetSizer(vbox)

    def tooltips_transformpanel(self):
        tipaxis = (
            "Rotation axis: letter[a1,a2,a3] where:\nletter refers to the frame \n"
        )
        tipaxis += "(a: for absolute Lauetools frame, s: sample frame\n"
        tipaxis += "c: crystal reciprocal frame (reciprocal unit cell basis vectors a*,b*,c*)\n"
        tipaxis += (
            "d: crystal direct frame (direct real unit cell basis vectors a,b,c))\n"
        )
        tipaxis += "a1,a2,a3 are components along the chosen basis vectors that can be integer, float or mathemical expression involving variable t"

        self.axisrot.SetToolTipString(tipaxis)
        self.tc_Rot_axis.SetToolTipString(tipaxis)

        angletip = "Rotation angle:  mathematical expression involving the variable t, e.g. t/10., cos(t/2.)**2, exp(-t)"
        self.tc_Rot_ang.SetToolTipString(angletip)
        self.anglerot.SetToolTipString(angletip)

        tipaxisangle = "Create a distribution of child grain from parent grain\n"
        tipaxisangle += "by setting by a parametric rotation given its axes and angles"
        self.rb_rotId.SetToolTipString(tipaxisangle)

        tiprange = 'range of variation of parameter "t": [start, end, nb steps]'
        self.trange1.SetToolTipString(tiprange)
        self.tc_range_rot.SetToolTipString(tiprange)

        self.rb_rotId.SetToolTipString("No rotation (default)")

    def onEnableStrain(self, evt):
        if self.rb_strainaxes.GetValue():
            self.rb_rotId.SetValue(True)

    def onEnableRotation(self, evt):
        if self.rb_rotaxis.GetValue() or self.rb_rotmatrix.GetValue():
            self.rb_strainId.SetValue(True)

    def ReadTransform(self):
        """
        reads toggle radio button and rotation and strain parameter
        returns parametric matrix rotation and strain
        """
        anglesample = DictLT.SAMPLETILT * DEG
        # transform matrix from xs, ys, zs sample frame to x, y,z absolute frame
        # vec / abs = R * vec / sample
        matrot_sample = np.array(
            [
                [np.cos(anglesample), 0, -np.sin(anglesample)],
                [0, 1, 0],
                [np.sin(anglesample), 0, np.cos(anglesample)],
            ]
        )
        inv_matrot_sample = np.linalg.inv(matrot_sample)

        # no transform
        if self.rb_rotId.GetValue() and self.rb_strainId.GetValue():
            # RotA = 'Id'
            return ""

        # rotation or general transform but no axial strain defined at the bottom of the board
        elif not self.rb_rotId.GetValue() and self.rb_strainId.GetValue():
            # reads tc_range_rot
            strlinspace = str(self.tc_range_rot.GetValue())[1:-1].split(",")
            try:
                tmin, tmax, step = (
                    float(strlinspace[0]),
                    float(strlinspace[1]),
                    int(strlinspace[2]),
                )
                # print "listrange",tmin, tmax, step
            except ValueError:
                sentence = 'Expression for t variation in ROTATION transform not understood! Check if there are "," and "]" '
                dlg = wx.MessageDialog(
                    self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR
                )
                dlg.ShowModal()
                dlg.Destroy()
                return

            # defines t array: paramtric values
            if step != 1:
                t = np.linspace(tmin, tmax, num=step)
            elif step == 1:
                # take the smallest value
                t = np.array([tmin])

            # rotation along an axis
            if self.rb_rotaxis.GetValue():
                # reads tc_Rot_axis
                # reads tc_Rot_ang
                frame_axis_rot = (
                    self.tc_Rot_axis.GetValue()
                )  # must contain s[exp1(t),exp2(t),exp3(t)]
                angle_rot = str(self.tc_Rot_ang.GetValue())
                framerot = frame_axis_rot[0]
                axisrot = str(frame_axis_rot[2:-1]).split(",")

                if 0:
                    print("framerot", framerot)
                    print("axisrot", axisrot)
                    print("angle_rot", angle_rot)

                # print "eva",eval(angle_rot,{"__builtins__":None},LaueToolsframe.safe_dict)  #http://lybniz2.sourceforge.net/safeeval.html

                # evaluates mathematical expression using or not 't' as parameter
                try:
                    if "t" in angle_rot:
                        evalangle_rot = eval(angle_rot)
                        # print "evaangle",evalangle_rot
                    else:
                        evalangle_rot = eval(angle_rot) * np.ones(len(t))
                except:
                    sentence = 'Expression for t variation in ROTATION axis not understood! Check if there are "," and "]" '
                    dlg = wx.MessageDialog(
                        self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR
                    )
                    dlg.ShowModal()
                    dlg.Destroy()
                    return

                evalaxisrot = []
                for k in range(3):
                    if "t" in axisrot[k]:
                        evalaxisrot.append(eval(axisrot[k]))
                    else:
                        evalaxisrot.append(eval(axisrot[k]) * np.ones(len(t)))

                # print "all axis",np.transpose(array(evalaxisrot))

                if framerot in ("s", "a"):
                    if framerot == "s":
                        evalaxisrot = np.dot(matrot_sample, evalaxisrot)
                    # array of angle, array of axis
                    return "r_axis", evalangle_rot, np.array(evalaxisrot).T

                if framerot in ("c", "d"):
                    # tag for transform , array of angle
                    # NOTE: array of axis  coordinates change is done later
                    # according to the orientation
                    return (
                        "r_axis_%s" % framerot,
                        evalangle_rot,
                        np.array(evalaxisrot).T,
                    )

            # general transform given by input of a matrix and a frame
            if self.rb_rotmatrix.GetValue():
                # reads self.tc_rotmatrix
                strmat = self.tc_rotmatrix.GetValue()

                # frame designation in which geometrical transform is expressed
                framerot = strmat[0]

                # reading matrix elements from strmat string value of tc_rotmatrix editor
                # transform matrix is evalmat and frame is framerot
                try:
                    text = str(strmat[3:-2])
                    tu = text.replace("[", "").replace("]", "").split(",")
                    # print "tu",tu
                    evalmatrot = []
                    for k in range(9):
                        # print "tu[k]",tu[k]
                        # print 't' in tu[k]
                        # if the variable 't' appears in formula, then evaluate the formula
                        if "t" in tu[k]:
                            evalmatrot.append(eval(tu[k]))
                        else:
                            evalmatrot.append(eval(tu[k]) * np.ones(len(t)))
                    # print "evalmatrot",evalmatrot
                    evalmat = np.reshape(np.array(evalmatrot).T, (len(t), 3, 3))
                except ValueError:
                    sentence = 'Expression for general expression in ROTATION transform not understood! Check if there are "," and "]". Mathematical operators may be unknown by numpy'
                    dlg = wx.MessageDialog(
                        self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR
                    )
                    dlg.ShowModal()
                    dlg.Destroy()
                    return

                if framerot in ("s", "a"):

                    # need to convert operator in xs, ys, zs frame into x, y,z frame
                    if framerot == "s":
                        for trans in range(len(t)):
                            # TODO: clarify
                            # computing M transform from a*,b*,c* to x, y,z absolute frame
                            # ie.  Xfinal = MXinitial_a*,b*,c*
                            # from evalmat[trans] matrix input by user in xs, ys, zs frame
                            evalmat[trans] = np.dot(
                                matrot_sample, np.dot(evalmat[trans], inv_matrot_sample)
                            )

                    print("evalmat from s or a", evalmat)
                    # tag: general matrix transform in absolute frame, array of rot matrix
                    return "r_mat", evalmat

                elif framerot in ("c", "d"):
                    # 'c' user have input a transform in crystal frame a*,b*,c*
                    # 'd' user have input a transform in crystal frame a,b,c (real unit cell basis vectors)
                    # as with 'a' but calculation is done later according to matorient

                    # print "evalmat from c",evalmat

                    return "r_mat_%s" % framerot, evalmat  # array of rot matrix

        # three axial strain transform
        elif not self.rb_strainId.GetValue():

            # reads tc_strainrange
            strlinspace = str(self.tc_strainrange.GetValue())[1:-1].split(",")
            try:
                tmin, tmax, step = (
                    float(strlinspace[0]),
                    float(strlinspace[1]),
                    int(strlinspace[2]),
                )
            except ValueError:
                sentence = 'Expression for t variation in STRAIN transform not understood! Check if there are "," and "]" '
                dlg = wx.MessageDialog(
                    self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR
                )
                dlg.ShowModal()
                dlg.Destroy()
                return
            #            print "listrange", tmin, tmax, step
            if step != 1:
                t = np.linspace(tmin, tmax, num=step)
            else:
                t = np.array([tmin])

            strainIDlist = []
            if self.rb_strainaxes.GetValue():  # strain along axes 1, 2, 3

                evalfac_strain_list = []
                evalaxisstrain_list = []

                list_tc_axis = [self.tc_axe1_axis, self.tc_axe2_axis, self.tc_axe3_axis]
                list_tc_factor = [
                    self.tc_axe1_factor,
                    self.tc_axe2_factor,
                    self.tc_axe3_factor,
                ]

                # loop over the three axes defined by user
                for axe in range(3):
                    # reads tc_axe1_axis
                    # reads tc_axe1_factor
                    frame_axis_strain = list_tc_axis[
                        axe
                    ].GetValue()  # must contain s[exp1(t),exp2(t),exp3(t)]
                    fac_strain = str(list_tc_factor[axe].GetValue())
                    framestrain = frame_axis_strain[0]
                    axisstrain = str(frame_axis_strain[2:-1]).split(",")
                    print("framestrain", framestrain)
                    print("axisstrain", axisstrain)
                    print("fac_strain", fac_strain)

                    if "t" in fac_strain:
                        evalfac_strain = eval(fac_strain)
                    else:
                        evalfac_strain = eval(fac_strain) * np.ones(len(t))

                    evalaxisstrain = []
                    for k in range(3):
                        if "t" in axisstrain[k]:
                            evalaxisstrain.append(eval(axisstrain[k]))
                            print("eval(axisstrain[k])", eval(axisstrain[k]))
                        else:
                            evalaxisstrain.append(eval(axisstrain[k]) * np.ones(len(t)))

                    print("array(evalaxisstrain)", np.array(evalaxisstrain).T)
                    # to do now
                    if framestrain == "a":
                        strainID = "s_axis"
                        evalfac_strain_list.append(evalfac_strain)
                        evalaxisstrain_list.append(np.array(evalaxisstrain).T)
                    elif framestrain == "s":
                        strainID = "s_axis"
                        evalfac_strain_list.append(evalfac_strain)
                        evalaxisstrain_list.append(
                            np.transpose(
                                np.dot(matrot_sample, np.array(evalaxisstrain))
                            )
                        )

                    elif (
                        framestrain == "c"
                    ):  # as with 'a' but calculation is done later according to matorient
                        strainID = "s_axis_c"
                        evalfac_strain_list.append(evalfac_strain)
                        evalaxisstrain_list.append(np.array(evalaxisstrain).T)
                    # #                    if framerot == 's' or framerot == 'a':
                    # #                        if framerot == 's': evalaxisrot = np.dot(matrot_sample, evalaxisrot)
                    # #                        return 'r_axis',evalangle_rot, np.transpose(array(evalaxisrot))  # array of angle, array of axis
                    # #                    if framerot == 'c':
                    # #                        # array of angle, array of axis  coordinates change is done later according to the orientation
                    # #                        return 'r_axis_c',evalangle_rot, np.transpose(array(evalaxisrot))
                    # #                    # to finish now
                    strainIDlist.append(strainID)

                    # """
                    # evalfac_strain_list.append(evalfac_strain)
                    # evalaxisstrain_list.append(np.transpose(array(evalaxisstrain)))
                    # """

                # array of strainaxis ID, array of 3 strain factors, array of corresponding 3 axes
                return strainIDlist, evalfac_strain_list, evalaxisstrain_list

        return


class SlipSystemPanel(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.granparent = parent.GetParent().GetParent()

        print("granparent of TransformPanel", self.granparent)

        self.SelectGrains = {}
        self.Bmatrices = {}

        # list Control for selected grains for SIMULATION
        font3 = self.granparent.font3

        titlemiddle = wx.StaticText(self, -1, "slip systems rotation Transformations")
        titlemiddle.SetFont(font3)

        buttontransform = wx.Button(self, -1, "Apply transforms", size=(150, 35))
        buttontransform.Bind(wx.EVT_BUTTON, self.granparent.OnApplytransformSlipSystems)
        buttontransform.SetFont(font3)

        self.granparent.transform_index = 0
        self.granparent.dict_transform = {}

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(titlemiddle)

        vbox.Add(buttontransform)

        self.SetBackgroundColour("sky Blue")

        self.SetSizer(vbox)

    def ReadTransform(self):
        """
        
        """
        print("ReadTransform  slipsystem")
        Bmatrix = self.granparent.Bmatrix_current

        misanglemin = -0.2
        misanglemax = 0.2
        nbsteps = 5

        angle_rot = np.linspace(misanglemin, misanglemax, num=nbsteps)
        nb_angles = len(angle_rot)

        slipsystemsfcc = np.array(DictLT.SLIPSYSTEMS_FCC).reshape((12, 2, 3))

        nbsystems = len(slipsystemsfcc)
        axisrot_list = []

        for slipsystem in slipsystemsfcc:
            #             print slipsystem
            plane_HKL, direction_uvw = slipsystem

            # plane normal coordinates in a b c (direct unit cell) basis
            plane_uvw = CP.fromreciprocalframe_to_realframe(plane_HKL, Bmatrix)
            # roation axis in a b c frame
            axis_uvw = np.cross(plane_uvw, direction_uvw)

            axisrot = np.tile(np.array(axis_uvw), nb_angles).reshape((nb_angles, 3))
            axisrot_list.append(axisrot)

        all_axes = np.concatenate(tuple(axisrot_list))

        all_angles = np.tile(angle_rot, nbsystems)

        return "r_axis_d_slipsystem", all_angles, all_axes


class SimulationPanel(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.granparent = parent.GetParent().GetParent()

        print("granparent of SimulationPanel", self.granparent)

        self.SetBackgroundColour("cyan")

        title1 = wx.StaticText(self, -1, "Spectral Band(keV)")
        title1.SetFont(self.granparent.font3)

        txtemin = wx.StaticText(self, -1, "Energy min: ")
        self.scmin = wx.SpinCtrl(self, -1, "5", size=(60, -1), min=5, max=195)

        txtemax = wx.StaticText(self, -1, "Energy max: ")
        self.scmax = wx.SpinCtrl(self, -1, "25", size=(60, -1), min=6, max=200)

        gridSizer = wx.GridSizer(rows=1, cols=4, hgap=1, vgap=1)
        gridSizer.Add(txtemin, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        # Set the TextCtrl to expand on resize
        gridSizer.Add(self.scmin, 0, wx.EXPAND)
        gridSizer.Add(txtemax, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer.Add(self.scmax, 0, wx.EXPAND)

        title2 = wx.StaticText(self, -1, "Plot Parameters")
        title2.SetFont(self.granparent.font3)
        title25 = wx.StaticText(self, -1, "Detector Parameters")
        title25.SetFont(self.granparent.font3)

        self.showplotBox = wx.CheckBox(self, -1, "Show Plot")
        self.showplotBox.SetValue(True)
        self.rbtop = wx.RadioButton(self, 200, "Reflection mode top", style=wx.RB_GROUP)
        self.rbside = wx.RadioButton(self, 200, "Reflection mode side +")
        self.rbsideneg = wx.RadioButton(self, 200, "Reflection mode side -")
        self.rbtransmission = wx.RadioButton(self, 200, "Transmission mode")

        self.rbtop.SetValue(True)

        self.checkshowExperimenalData = wx.CheckBox(self, -1, "Show Exp. Data")
        self.checkshowExperimenalData.SetValue(False)

        self.checkExperimenalImage = wx.CheckBox(self, -1, "Show Exp. Image")
        self.checkExperimenalImage.SetValue(False)

        self.expimagetxtctrl = wx.TextCtrl(self, -1, "", size=(75, -1))
        self.expimagebrowsebtn = wx.Button(self, -1, "...", size=(50, -1))

        self.expimagebrowsebtn.Bind(wx.EVT_BUTTON, self.onSelectImageFile)

        current_param = self.granparent.initialParameters["CalibrationParameters"]

        txtdd = wx.StaticText(self, -1, "Det.Dist.(mm): ")
        self.detdist = wx.TextCtrl(self, -1, str(current_param[0]), size=(75, -1))
        txtdiam = wx.StaticText(self, -1, "Det. Diam.(mm): ")
        self.detdiam = wx.TextCtrl(self, -1, "165", size=(40, -1))
        txtxcen = wx.StaticText(self, -1, "xcen(pix): ")
        self.xcen = wx.TextCtrl(self, -1, str(current_param[1]), size=(75, -1))
        txtycen = wx.StaticText(self, -1, "ycen(pix): ")
        self.ycen = wx.TextCtrl(self, -1, str(current_param[2]), size=(75, -1))
        txtxbet = wx.StaticText(self, -1, "xbet(deg): ")
        self.xbet = wx.TextCtrl(self, -1, str(current_param[3]), size=(75, -1))
        txtxgam = wx.StaticText(self, -1, "xgam(deg): ")
        self.xgam = wx.TextCtrl(self, -1, str(current_param[4]), size=(75, -1))
        txtpixelsize = wx.StaticText(self, -1, "pixelsize(mm): ")
        self.ctrlpixelsize = wx.TextCtrl(
            self, -1, str(self.granparent.pixelsize), size=(75, -1)
        )

        self.pt_2thetachi = wx.RadioButton(self, 100, "2ThetaChi", style=wx.RB_GROUP)
        self.pt_XYCCD = wx.RadioButton(self, 300, "XYPixel")
        self.pt_XYfit2d = wx.RadioButton(self, 300, "XYfit2d")
        self.pt_2thetachi.SetValue(True)

        # set tooltips
        self.rbtop.SetToolTipString("Camera at 2theta=90 deg on top of sample")
        self.rbside.SetToolTipString("Camera at 2theta=90 deg on side of sample")
        self.rbsideneg.SetToolTipString("Camera at 90 deg on other side of sample")
        self.rbtransmission.SetToolTipString("Camera at 2theta=0 deg")

        self.checkshowExperimenalData.SetToolTipString(
            "Plot markers for current experimental peak list"
        )
        self.checkExperimenalImage.SetToolTipString("Display Laue pattern (2D image)")
        self.expimagetxtctrl.SetToolTipString(
            "Full path for Laue pattern to be superimposed to simulated peaks"
        )
        self.expimagebrowsebtn.SetToolTipString(
            "Browse and select Laue Pattern image file"
        )

        self.pt_2thetachi.SetToolTipString(
            "Peaks Coordinates in scattering angles: 2theta and Chi"
        )
        self.pt_XYCCD.SetToolTipString("Peaks Coordinates in detector frame pixels")
        self.pt_XYfit2d.SetToolTipString(
            "Peaks Coordinates in detector frame pixels (fit2D convention)"
        )

        # set widgets layout
        gridSizer2 = wx.GridSizer(rows=11, cols=3, hgap=1, vgap=1)

        gridSizer2.Add(title2, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        # Set the TextCtrl to expand on resize
        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.EXPAND)
        gridSizer2.Add(title25, 0, wx.EXPAND)

        gridSizer2.Add(self.showplotBox, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        # Set the TextCtrl to expand on resize
        gridSizer2.Add(txtdd, 0, wx.EXPAND)
        gridSizer2.Add(self.detdist, 0, wx.EXPAND)

        gridSizer2.Add(self.rbtop, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtdiam, 0, wx.EXPAND)
        gridSizer2.Add(self.detdiam, 0, wx.EXPAND)

        gridSizer2.Add(self.rbside, 1, wx.ALIGN_LEFT)
        gridSizer2.Add(txtxcen, 0, wx.EXPAND)
        gridSizer2.Add(self.xcen, 0, wx.EXPAND)

        gridSizer2.Add(self.rbsideneg, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtycen, 0, wx.EXPAND)
        gridSizer2.Add(self.ycen, 0, wx.EXPAND)

        gridSizer2.Add(self.rbtransmission, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtxbet, 0, wx.EXPAND)
        gridSizer2.Add(self.xbet, 0, wx.EXPAND)

        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(txtxgam, 0, wx.EXPAND)
        gridSizer2.Add(self.xgam, 0, wx.EXPAND)

        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(txtpixelsize, 0, wx.EXPAND)
        gridSizer2.Add(self.ctrlpixelsize, 0, wx.EXPAND)

        gridSizer2.Add(self.pt_2thetachi, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(self.pt_XYCCD, 0, wx.EXPAND)
        gridSizer2.Add(self.pt_XYfit2d, 0, wx.EXPAND)

        gridSizer2.Add(
            self.checkshowExperimenalData, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER
        )
        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.EXPAND)
        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.EXPAND)

        gridSizer2.Add(self.checkExperimenalImage, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(self.expimagetxtctrl, 0, wx.EXPAND)
        gridSizer2.Add(self.expimagebrowsebtn, 0, wx.EXPAND)

        self.spSizer = wx.BoxSizer(wx.VERTICAL)
        self.spSizer.Add(title1, 0)
        self.spSizer.AddSpacer(10)
        self.spSizer.Add(gridSizer, 0)
        self.spSizer.Add(gridSizer2, 0)
        self.spSizer.AddSpacer(5)

        self.SetSizer(self.spSizer)

    def onSelectImageFile(self, evt):
        self.GetfullpathFile(evt)
        self.expimagetxtctrl.SetValue(self.fullpathimagefile)

    def GetfullpathFile(self, evt):
        myFileDialog = wx.FileDialog(
            self,
            "Choose an image file",
            style=wx.OPEN,
            #                                         defaultDir=self.dirname,
            wildcard="MAR or Roper images(*.mccd)|*.mccd|All files(*)|*",
        )
        dlg = myFileDialog
        dlg.SetMessage("Choose an image file")
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()

            #             self.dirnameBlackList = dlg.GetDirectory()
            self.fullpathimagefile = str(filename)

        else:
            pass


class parametric_Grain_Dialog3(wx.Frame):
    """
    in development new board of parametric 
    Laue Simulation with tabs for ergonomic GUI
    """

    def __init__(self, parent, id, title, initialParameters):
        wx.Frame.__init__(self, parent, id, title)

        self.panel = wx.Panel(self)

        self.dirname = os.getcwd()

        self.parent = parent
        # self.dirname = LaueToolsframe.dirname
        self.initialParameters = initialParameters

        # self.dirname = '.'
        try:
            self.CCDLabel = self.parent.CCDLabel
        except AttributeError:
            self.CCDLabel = self.initialParameters["CCDLabel"]

        current_param = self.initialParameters["CalibrationParameters"]
        self.pixelsize = self.initialParameters["pixelsize"]
        self.framedim = self.initialParameters["framedim"]

        self.font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        self.initialParameters = initialParameters
        if initialParameters["ExperimentalData"] is not None:

            (
                self.data_2theta,
                self.data_chi,
                self.data_pixX,
                self.data_pixY,
                self.data_I,
            ) = initialParameters["ExperimentalData"]

        self.dict_grain_created = {}
        self.SelectGrains = {}

        self.CurrentGrain = [
            "Cu",
            "FaceCenteredCubic",
            "Identity",
            "Identity",
            "Identity",
            "Identity",
            "Grain_0",
            "",
        ]

        self.create_leftpanel()

        # defines self.hboxbottom  sizer
        self.bottompanel()

        self.nb0 = wx.Notebook(self.panel, -1, style=0)

        self.centerpanel = TransformPanel(self.nb0)
        self.centerpanel2 = SlipSystemPanel(self.nb0)
        self.rightpanel = SimulationPanel(self.nb0)

        self.nb0.AddPage(self.centerpanel, "Transforms")
        self.nb0.AddPage(self.centerpanel2, "SlipSystems")
        self.nb0.AddPage(self.rightpanel, "Simulation")

        self.nb0.SetSelection(2)

        self.nb0.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnTabChange_nb0)

        # tooltips
        self.centerpanel.SetToolTipString(
            "Apply parametric transforms (distribution of orientation and strain) on higlighted grain in basket"
        )
        self.rightpanel.SetToolTipString(
            "Simulation parameters board (CCD position, spots coordinates,Energy band pass...)"
        )

        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.buttonsvertSizer, 1, wx.EXPAND)
        self.hbox.Add(self.nb0, 1, wx.EXPAND)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.hbox, 1, wx.EXPAND, 0)
        self.vbox.AddSpacer(10)
        linehoriz = wx.StaticLine(self.panel)
        self.vbox.Add(linehoriz, 0, wx.ALL | wx.EXPAND, 5)
        self.vbox.Add(self.hboxbottom, 0, wx.EXPAND, 0)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.Layout()

    def OnTabChange_nb0(self, event):

        #        print 'tab changed'
        selected_tab = self.nb0.GetSelection()
        print("selected tab:", selected_tab)
        print(self.nb0.GetPage(self.nb0.GetSelection()))
        print(self.nb0.GetPage(self.nb0.GetSelection()).GetName())

        event.Skip()  # patch for windows to update the tab display

    def create_leftpanel(self):

        #         self.panel.SetBackgroundColour('pink')

        titlegrain = wx.StaticText(self.panel, -1, "Grain Definition")
        titlegrain.SetFont(self.font3)

        self.RefreshCombos(1)

        txt1 = wx.StaticText(self.panel, -1, "Material")
        txt2 = wx.StaticText(self.panel, -1, "Extinctions")
        txt3 = wx.StaticText(self.panel, -1, "Transform_a")
        txt4 = wx.StaticText(self.panel, -1, "Rot. Matrix")
        txt5 = wx.StaticText(self.panel, -1, "B matrix")
        txt6 = wx.StaticText(self.panel, -1, "Transform_c")

        self.comboElem = wx.ComboBox(
            self.panel, -1, "Cu", choices=self.list_of_Elem, style=wx.CB_READONLY
        )
        self.comboExtinc = wx.ComboBox(
            self.panel,
            -1,
            "FaceCenteredCubic",
            choices=self.list_of_Extinc,
            style=wx.CB_READONLY,
        )
        self.comboStrain_a = wx.ComboBox(
            self.panel, -1, "Identity", choices=self.list_of_Strain_a
        )
        self.comboRot = wx.ComboBox(
            self.panel, -1, "Identity", choices=self.list_of_Rot
        )
        self.comboVect = wx.ComboBox(
            self.panel, -1, "Identity", choices=self.list_of_Vect
        )
        self.comboStrain_c = wx.ComboBox(
            self.panel, -1, "Identity", choices=self.list_of_Strain_c
        )

        buttonrefresh = wx.Button(self.panel, -1, "Refresh choices")
        buttonrefresh.Bind(wx.EVT_BUTTON, self.RefreshCombos)

        addgrainbtn = wx.Button(self.panel, -1, "Add Grain")
        addgrainbtn.SetFont(self.font3)
        addgrainbtn.Bind(wx.EVT_BUTTON, self._On_AddGrain)

        self.comboElem.Bind(wx.EVT_COMBOBOX, self.EnterComboElem)
        self.comboExtinc.Bind(wx.EVT_COMBOBOX, self.EnterComboExtinc)
        self.comboStrain_a.Bind(wx.EVT_COMBOBOX, self.EnterCombostrain_a)
        self.comboRot.Bind(wx.EVT_COMBOBOX, self.EnterComboRot)
        self.comboVect.Bind(wx.EVT_COMBOBOX, self.EnterComboVect)
        self.comboStrain_c.Bind(wx.EVT_COMBOBOX, self.EnterCombostrain_c)

        txtlist = wx.StaticText(self.panel, -1, "List of grains to be simulated")
        txtlist.SetFont(self.font3)
        # - - - - - - - - - - -  - - - - -
        self.LC = wx.ListCtrl(self.panel, -1, style=wx.LC_REPORT, size=(-1, 150))
        self.LC.InsertColumn(0, "Grain Name")
        self.LC.InsertColumn(1, "Element")
        self.LC.InsertColumn(2, "Extinc.")
        self.LC.InsertColumn(3, "Transform_a")
        self.LC.InsertColumn(4, "Rot. Matrix")
        self.LC.InsertColumn(5, "Transform_c")
        self.LC.InsertColumn(6, "Bmatrix")
        self.LC.InsertColumn(7, "Transform(p)")
        Col_width = 70
        self.LC.SetColumnWidth(0, 80)
        self.LC.SetColumnWidth(1, 60)
        self.LC.SetColumnWidth(2, Col_width)
        self.LC.SetColumnWidth(3, Col_width)
        self.LC.SetColumnWidth(4, Col_width)
        self.LC.SetColumnWidth(5, Col_width)
        self.LC.SetColumnWidth(6, Col_width)
        self.LC.SetColumnWidth(7, Col_width)

        #         self.String_Info = self.userguidestring()
        #         self.infotext = wx.TextCtrl(self.panel,
        #                                     style=wx.TE_MULTILINE | wx.TE_READONLY,  # | wx.HSCROLL,
        #                                     size=(590, 190),
        #                                     pos=(5, 40))
        #
        #         self.infotext.SetValue(self.String_Info)

        deletebutton = wx.Button(self.panel, -1, "Delete")
        deleteallbutton = wx.Button(self.panel, -1, "DeleteAll")
        deletebutton.Bind(wx.EVT_BUTTON, self.DeleteGrain)
        deleteallbutton.Bind(wx.EVT_BUTTON, self.DeleteAllGrain)

        # tooltips
        tipmat = "Material or Crystallographic structure"

        txt1.SetToolTipString(tipmat)
        self.comboElem.SetToolTipString(tipmat)

        tipextinc = "Systematic extinctions rules"

        txt2.SetToolTipString(tipextinc)
        self.comboExtinc.SetToolTipString(tipextinc)

        tipstrain_a = "Operator A in formula: q= A U B C G*"
        txt3.SetToolTipString(tipstrain_a)
        self.comboStrain_a.SetToolTipString(tipstrain_a)

        tipRotationMatrix = "Operator U (Orientation Matrix) in formula: q= A U B C G*"
        txt4.SetToolTipString(tipRotationMatrix)
        self.comboRot.SetToolTipString(tipRotationMatrix)

        tipvect = "Operator B in formula: q= A U B C G*\n"
        tipvect += (
            "Initial set of reciprocal unit cell basis vector in LaueTools lab. frame"
        )

        txt5.SetToolTipString(tipvect)
        self.comboVect.SetToolTipString(tipvect)

        tipstrain_c = "Operator C in formula: q= A U B C G*\n"
        self.comboStrain_c.SetToolTipString(tipstrain_c)
        txt6.SetToolTipString(tipstrain_c)

        addgrainbtn.SetToolTipString("Add a grain in list of grains to simulate")
        buttonrefresh.SetToolTipString("Update list of above selection drop down menus")

        deletebutton.SetToolTipString("Delete the current selected grain")
        deleteallbutton.SetToolTipString("Delete all grains of list")

        self.LC.SetToolTipString("List of grains that will be simulated")

        # layout

        gridSizer = wx.GridSizer(rows=6, cols=2, hgap=1, vgap=1)
        gridSizer.Add(txt1, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        # Set the TextCtrl to expand on resize
        gridSizer.Add(self.comboElem, 0, wx.EXPAND)
        gridSizer.Add(txt2, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer.Add(self.comboExtinc, 0, wx.EXPAND)
        gridSizer.Add(txt3, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer.Add(self.comboStrain_a, 0, wx.EXPAND)
        gridSizer.Add(txt4, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer.Add(self.comboRot, 0, wx.EXPAND)
        gridSizer.Add(txt5, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer.Add(self.comboVect, 0, wx.EXPAND)
        gridSizer.Add(txt6, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer.Add(self.comboStrain_c, 0, wx.EXPAND)

        self.buttonshoriz2Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.buttonshoriz2Sizer.Add(addgrainbtn, 1)
        self.buttonshoriz2Sizer.Add(buttonrefresh, 1)

        horizbtnsizers = wx.BoxSizer(wx.HORIZONTAL)
        horizbtnsizers.Add(deletebutton, 0)
        horizbtnsizers.Add(deleteallbutton, 0)

        self.buttonsvertSizer = wx.BoxSizer(wx.VERTICAL)
        self.buttonsvertSizer.Add(titlegrain, 0, wx.ALIGN_CENTER)
        self.buttonsvertSizer.Add(gridSizer)
        self.buttonsvertSizer.Add(self.buttonshoriz2Sizer, 0, wx.ALIGN_CENTER)
        self.buttonsvertSizer.AddSpacer(10)
        self.buttonsvertSizer.Add(txtlist, 0, wx.ALIGN_CENTER)
        self.buttonsvertSizer.Add(self.LC, 1, wx.EXPAND | wx.ALL)
        self.buttonsvertSizer.Add(horizbtnsizers, 0, wx.ALIGN_CENTER)

    def bottompanel(self):
        """
        fill self.hboxbottom = wx.BoxSizer(wx.HORIZONTAL)
        """

        title3 = wx.StaticText(self.panel, -1, "File Parameters")
        title3.SetFont(self.font3)

        txtdir = wx.StaticText(self.panel, -1, "Directory")
        selectbtn = wx.Button(self.panel, -1, "Select")
        selectbtn.Bind(wx.EVT_BUTTON, self.opendir)
        currbtn = wx.Button(self.panel, -1, "Current")
        currbtn.Bind(wx.EVT_BUTTON, self.showCurrentDir)

        gridSizerdirectory = wx.GridSizer(rows=1, cols=3, hgap=1, vgap=1)
        gridSizerdirectory.Add(txtdir, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizerdirectory.Add(selectbtn, 0, wx.EXPAND)
        gridSizerdirectory.Add(currbtn, 0, wx.EXPAND)

        self.savefileBox = wx.CheckBox(self.panel, -1, "Save File")
        self.savefileBox.SetValue(False)

        self.rb1 = wx.RadioButton(self.panel, 300, "Manual", style=wx.RB_GROUP)
        self.rb2 = wx.RadioButton(self.panel, 300, "Auto. Indexed")
        self.rb2.SetValue(True)

        self.textcontrolfilemanual = wx.TextCtrl(self.panel, -1, "myfilename")
        txtsimext = wx.StaticText(self.panel)

        self.prefixfilenamesimul = self.initialParameters["prefixfilenamesimul"]

        self.textcontrolfileauto = wx.TextCtrl(self.panel, -1, self.prefixfilenamesimul)
        txtsimext2 = wx.StaticText(self.panel, -1, ".sim")

        self.corfileBox = wx.CheckBox(self.panel, -1, "Create .cor file")
        self.corfileBox.SetValue(False)
        self.corcontrolfake = wx.TextCtrl(self.panel, -1, self.prefixfilenamesimul)
        txtcorext = wx.StaticText(self.panel, -1, ".cor")

        # set tool tips
        self.savefileBox.SetToolTipString("Save simulated peaks in file")
        self.rb1.SetToolTipString("Manual set of simulation peaks list Filename  .sim")
        self.rb2.SetToolTipString(
            "Automatic incrementation of simulation peaks list Filename  .sim"
        )
        self.corfileBox.SetToolTipString(
            "Create a fake experimental peaks list (without miller indices)"
        )

        # widgets layout

        gridSizerFile = wx.GridSizer(rows=3, cols=3, hgap=1, vgap=1)
        gridSizerFile.Add(self.rb1, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizerFile.Add(self.textcontrolfilemanual, 0, wx.EXPAND)
        gridSizerFile.Add(txtsimext, 0, wx.EXPAND)

        gridSizerFile.Add(self.rb2, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizerFile.Add(self.textcontrolfileauto, 0, wx.EXPAND)
        gridSizerFile.Add(txtsimext2, 0, wx.EXPAND)

        gridSizerFile.Add(self.corfileBox, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizerFile.Add(self.corcontrolfake, 0, wx.EXPAND)
        gridSizerFile.Add(txtcorext, 0, wx.EXPAND)

        btnSimulate = wx.Button(self.panel, -1, "Simulate", size=(200, 50))
        btnSimulate.Bind(wx.EVT_BUTTON, self.OnSimulate)
        btnSimulate.SetFont(self.font3)

        self.textprocess = wx.StaticText(self.panel, -1, "                     ")
        self.gauge = wx.Gauge(self.panel, -1, 1000, size=(200, 25))

        v1 = wx.BoxSizer(wx.VERTICAL)
        v1.Add(title3, 0)
        v1.Add(self.savefileBox, 0)
        v1.Add(gridSizerdirectory, 0)

        v3 = wx.BoxSizer(wx.VERTICAL)
        v3.Add(btnSimulate, 0)
        v3.Add(self.textprocess, 0)
        v3.Add(self.gauge, 0)

        self.hboxbottom = wx.BoxSizer(wx.HORIZONTAL)
        self.hboxbottom.Add(v1)
        self.hboxbottom.Add(gridSizerFile)
        self.hboxbottom.Add(v3)

    def userguidestring(self):
        self.String_Info = " ***** USER GUIDE ******\n\n"
        self.String_Info += (
            "1- Select crystallographic structure of by Element or structure\n"
        )
        self.String_Info += "          Orientation by some specific orientation of a*,b*,c* or rotation matrix\n"
        self.String_Info += "          Angular strain matrix\n"
        self.String_Info += "2- Add grains as many as you want in the list\n"
        self.String_Info += "          Delete unwanted grains by selecting them and clicking in Delete button\n"
        self.String_Info += "3- OPTIONNALY Select a set of orientation or strain transfrom from a selected parent grain in the list\n"
        self.String_Info += (
            "          Axis rotation and Axes traction can not be combined\n"
        )
        self.String_Info += "          One transform set for one selected grain\n"
        self.String_Info += "          t is the varying parameter given start, end and number of steps values,\n"
        self.String_Info += "               can be put in any place in other field with maths expression\n"
        self.String_Info += "               example: a[1+cos(t/2.)*exp(-t*.1)]\n"
        self.String_Info += "               start, end may be floats\n"
        self.String_Info += "          Frame chosen to express coordinates can be selectiong by choosing the letter a, s or c\n\n"
        self.String_Info += "          ROTATION:\n"
        self.String_Info += (
            "          Choose the frame, axis-vector coordinnates and angle in degree\n"
        )
        self.String_Info += "          STRAIN:\n"
        self.String_Info += "          three axes of traction can be combined,\n"
        self.String_Info += (
            "          with frame, vector direction, and amplitude in real space\n"
        )
        self.String_Info += "                example: factor of 1.1 means 10\% expansion in real space along the chosen direction\n"
        self.String_Info += (
            "4- Select plot/calibration/file parameter and click on simulate button\n"
        )
        return self.String_Info

    def OnTextChanged(self, event):
        self.modify = True
        event.Skip()

    def OnKeyDown(self, event):
        event.Skip()

    def showCurrentDir(self, event):
        dlg = wx.MessageDialog(
            self,
            "Current directory :%s" % self.dirname,
            "Current Directory",
            wx.OK | wx.ICON_INFORMATION,
        )
        dlg.ShowModal()
        dlg.Destroy()

    def Display_combos(self):

        self.comboElem = wx.ComboBox(
            self.toppanel,
            -1,
            "Cu",
            (10, 30),
            size=(60, -1),
            choices=self.list_of_Elem,
            style=wx.CB_READONLY,
        )
        self.comboExtinc = wx.ComboBox(
            self.toppanel,
            -1,
            "FaceCenteredCubic",
            (100, 30),
            size=(60, -1),
            choices=self.list_of_Extinc,
            style=wx.CB_READONLY,
        )
        self.comboStrain_a = wx.ComboBox(
            self.toppanel,
            -1,
            "Identity",
            (190, 30),
            size=(80, -1),
            choices=self.list_of_Strain_a,
        )
        self.comboRot = wx.ComboBox(
            self.toppanel,
            -1,
            "Identity",
            (300, 30),
            size=(80, -1),
            choices=self.list_of_Rot,
        )
        self.comboVect = wx.ComboBox(
            self.toppanel,
            -1,
            "Identity",
            (410, 30),
            size=(80, -1),
            choices=self.list_of_Vect,
        )
        self.comboStrain_c = wx.ComboBox(
            self.toppanel,
            -1,
            "Identity",
            (520, 30),
            size=(80, -1),
            choices=self.list_of_Strain_c,
        )

    def RefreshCombos(self, event):
        # order list element for clarity
        List_Extinc_name = list(DictLT.dict_Extinc.keys())
        List_Extinc_name.remove("NoExtinction")
        List_Extinc_name.sort()

        List_Rot_name = list(DictLT.dict_Rot.keys())
        List_Rot_name.remove("Identity")
        List_Rot_name.sort()

        List_Vect_name = list(DictLT.dict_Vect.keys())
        List_Vect_name.remove("Identity")
        List_Vect_name.sort()

        List_Elem_name = list(DictLT.dict_Materials.keys())
        List_Elem_name.remove("inputB")
        List_Elem_name.sort()

        List_Transform_name = list(DictLT.dict_Transforms.keys())
        List_Transform_name.remove("Identity")
        List_Transform_name.sort()

        self.list_of_Elem = ["inputB"] + List_Elem_name
        self.list_of_Extinc = ["NoExtinction"] + List_Extinc_name
        self.list_of_Strain_a = ["Identity"] + List_Transform_name
        self.list_of_Rot = ["Identity"] + List_Rot_name
        self.list_of_Vect = ["Identity"] + List_Vect_name
        self.list_of_Strain_c = ["Identity"] + List_Transform_name

    #        self.Display_combos()

    def EnterComboElem(self, event):
        item = event.GetSelection()
        key_material = self.list_of_Elem[item]
        # print "item",item
        # print "self.list_of_Elem[item]",self.list_of_Elem[item]
        self.CurrentGrain[0] = key_material

        # set extinction code corresponding to key_material
        extinction_code = DictLT.dict_Materials[key_material][-1]
        #        print "extinction_code", extinction_code
        self.comboExtinc.SetValue(DictLT.dict_Extinc_inv[extinction_code])
        event.Skip()

    def EnterComboExtinc(self, event):
        item = event.GetSelection()
        self.CurrentGrain[1] = self.list_of_Extinc[item]
        event.Skip()

    def EnterCombostrain_a(self, event):
        item = event.GetSelection()
        self.CurrentGrain[2] = self.list_of_Strain_a[item]
        event.Skip()

    def EnterComboRot(self, event):
        item = event.GetSelection()
        self.CurrentGrain[3] = self.list_of_Rot[item]
        event.Skip()

    def EnterComboVect(self, event):
        item = event.GetSelection()
        self.CurrentGrain[4] = self.list_of_Vect[item]
        event.Skip()

    def EnterCombostrain_c(self, event):
        item = event.GetSelection()
        self.CurrentGrain[5] = self.list_of_Strain_c[item]
        event.Skip()

    def _On_AddGrain(self, event):
        """
        parametric_Grain_Dialog
        """
        # read parameter in combos:
        elem = self.comboElem.GetValue()
        extinc = self.comboExtinc.GetValue()
        stra = self.comboStrain_a.GetValue()
        rot = self.comboRot.GetValue()
        B = self.comboVect.GetValue()
        strc = self.comboStrain_c.GetValue()

        # out them in self.CurrentGrain

        for k, val in enumerate([elem, extinc, stra, rot, B, strc]):
            self.CurrentGrain[k] = val

        # inserting parameters in listctrl

        num_items = self.LC.GetItemCount()

        grain_name = "Grain_" + str(num_items)
        self.CurrentGrain[6] = grain_name

        print("self.CurrentGrain", self.CurrentGrain)

        self.LC.InsertStringItem(num_items, str(grain_name))
        self.LC.SetStringItem(num_items, 1, str(self.CurrentGrain[0]))
        self.LC.SetStringItem(num_items, 2, str(self.CurrentGrain[1]))
        self.LC.SetStringItem(num_items, 3, str(self.CurrentGrain[2]))
        self.LC.SetStringItem(num_items, 4, str(self.CurrentGrain[3]))
        self.LC.SetStringItem(num_items, 5, str(self.CurrentGrain[4]))
        self.LC.SetStringItem(num_items, 6, str(self.CurrentGrain[5]))
        self.LC.SetStringItem(num_items, 7, "")

        self.dict_grain_created[grain_name] = self.CurrentGrain[:]

        event.Skip()

    def Select_AllGrain(self, event):
        """
        in parametric
        select all grains in created grain list(LC) t
        """
        num_items_to_select = self.LC.GetItemCount()  # nb of items in LC list(origin)

        for index_item in range(num_items_to_select - 1, -1, -1):
            name = self.LC.GetItemText(index_item)
            print("name in parametric selection", name)
            if name:
                self.SelectGrains[name] = self.dict_grain_created[name]

            else:
                print(
                    "You must select by mouse a least one grain!!"
                )  # TODO better error catching
        # print self.SelectGrains
        event.Skip()

    def DeleteGrain(self, event):
        """
        delete one grain in created grain list
        TODO: some strange behaviour when deleting one grain and 
        grain index in list of grain to simulate
        """
        nametodeselect = self.LC.GetItemText(self.LC.GetFocusedItem())
        # print nametodeletet
        self.LC.DeleteItem(self.LC.GetFocusedItem())
        del self.dict_grain_created[nametodeselect]
        del self.SelectGrains[nametodeselect]
        event.Skip()

    def DeleteAllGrain(self, event):
        """
        delete all grains in created grain list
        """
        self.LC.DeleteAllItems()
        self.dict_grain_created.clear()
        self.SelectGrains.clear()
        event.Skip()

    def OnApplytransform(self, event):
        """
        read and prepare geometrical transforms  according to selected or highlighted grain
        """
        name = self.LC.GetItemText(self.LC.GetFocusedItem())
        num_items = self.LC.GetItemCount()
        selectitem = self.LC.GetFocusedItem()
        print(" OnApplytransformname", name)
        print("total num_items", num_items)
        print("selectitem", selectitem)
        # print "from dict",self.dict_grain_created[name]
        if name:
            colindex_transform_p = 7
            transform_name = "Tr_" + str(self.transform_index)

            self.LC.SetStringItem(selectitem, colindex_transform_p, transform_name)

            self.dict_grain_created[name][colindex_transform_p] = transform_name
            self.SelectGrains[name] = self.dict_grain_created[name]

            # read transform
            alltransforms = self.centerpanel.ReadTransform()

            # create or update transform dictionary
            self.dict_transform[transform_name] = alltransforms
            print("in OnApplytransform -----")
            print("self.SelectGrains", self.SelectGrains)
            print("self.dict_transform", self.dict_transform)
        else:
            print("Please! Select a parent grain to be slightly transformed!")

        self.transform_index += 1
        event.Skip()

    def OnApplytransformSlipSystems(self, event):
        """
        apply geometrical transforms of slip systems on selected or highlighted grain
        """
        # force CCD pixel plot
        self.rightpanel.pt_XYCCD.SetValue(True)

        grainindex = self.LC.GetItemText(self.LC.GetFocusedItem())
        num_items = self.LC.GetItemCount()
        selectitem = self.LC.GetFocusedItem()
        print(" OnApplytransformname", grainindex)
        print("total num_items", num_items)
        print("selectitem", selectitem)
        # print "from dict",self.dict_grain_created[grainindex]
        if grainindex:
            colindex_transform_p = 7
            transform_name = "Tr_%dslipsystem" % self.transform_index

            self.LC.SetStringItem(selectitem, colindex_transform_p, transform_name)

            self.dict_grain_created[grainindex][colindex_transform_p] = transform_name
            self.SelectGrains[grainindex] = self.dict_grain_created[grainindex]

            # B matrix or T B matrix form slected grains

            Bmatrix_key = self.SelectGrains[grainindex][4]
            Transform_C_key = self.SelectGrains[grainindex][5]
            B_matrix = DictLT.dict_Vect[Bmatrix_key]
            Transform_crystalframe = DictLT.dict_Transforms[Transform_C_key]

            self.Bmatrix_current = np.dot(B_matrix, Transform_crystalframe)
            self.ParentGrainname = grainindex

            alltransforms = self.centerpanel2.ReadTransform()

            # create or update transform dictionary
            self.dict_transform[transform_name] = alltransforms
            print("in OnApplytransform -----")
            print("self.SelectGrains", self.SelectGrains)
            print("self.dict_transform", self.dict_transform)
        else:
            print("Please! Select a parent grain to be slightly transformed!")

        self.transform_index += 1
        event.Skip()

    def OnWriteCorFile(self, file_name_fake, data, nbgrains):
        """
        Write a fake  experimental file .cor
        TODO: check if there are not duplicates in other module ... use better readwriteASCII module
        """
        print("\n\n Writing fake .cor file...\n")
        wholestring = Edit_String_SimulData(data).splitlines()
        print('wholestring',wholestring)
        # header = '2theta  chi   x   y   I'
        headerarray = np.array(["2theta", " chi", "   x", "   y", "   I"], dtype="|S11")

        nbgrains = int(wholestring[1].split()[-1])
        print("nbgrains", nbgrains)

        # line position of lines starting with #G
        posG = [3]
        k = 4
        nn = 1

        while nn < nbgrains:
            if wholestring[k].startswith("#G"):
                posG.append(k)
                nn += 1
            k += 1

        print("posG", posG)

        # Read the data in Frame Ctrl

        nbpeaks_per_grain = []
        for st in posG:
            nbpeaks = int(wholestring[st].split()[-1])
            nbpeaks_per_grain.append(nbpeaks)

        print("nbpeaks_per_grain", nbpeaks_per_grain)

        dataarray = []
        gi = 0  # grainindex
        while gi < nbgrains:
            list_of_lines = wholestring[posG[gi] + 1 : nbpeaks_per_grain[gi] + posG[gi] + 1]
            print("list_of_lines",list_of_lines)
            joineddata = " ".join(list_of_lines)
            # print "array(joineddata.split())",array(joineddata.split())
            array_grain = np.reshape(
                np.array(np.array(joineddata.split()), dtype=np.float32),
                (nbpeaks_per_grain[gi], 9),
            )
            #            print "array_grain", array_grain
            dataarray.append(array_grain)
            gi += 1

        WData = np.concatenate(dataarray)
        twothetachi = WData[:, 5:7]
        # intensity model
        intensity = 1000.0 / WData[:, 4]  # inverse of energy
        # intensity=-WData[:,4] # energy
        xy = WData[:, -2:]
        # print "twothetachi",twothetachi
        # print "intensity",intensity
        # print "xy",xy
        Toedit = np.dstack(
            (twothetachi[:, 0], twothetachi[:, 1], xy[:, 0], xy[:, 1], intensity)
        )[0]

        # sort data according to modelled intensity
        # print "Toedit",Toedit[:3]
        Toedit_sorted = Toedit[np.argsort(Toedit[:, 4])[::-1]]
        # print "ghg",array(Toedit_sorted, dtype = '|S11')[:10] # number of digits in file encoded by number of chars in string

        print("Toedit_sorted",Toedit_sorted)
        filename = os.path.join(self.dirname, file_name_fake)

        


        corfileobj = open(filename, "w")

        footer="# Cor file generated by LaueTools Polygrains Simulation Software"
        footer += "\n# File created at %s by by ParametricLaueSimulator.py" % (time.asctime())
        footer += "\n# Calibration parameters"
        for par, value in zip(["dd", "xcen", "ycen", "xbet", "xgam"], self.calib):
            footer+="\n# %s     :   %s" % (par, value)
        footer+="\n# pixelsize    :   %s" % self.pixelsize
        footer+="\n# ypixelsize    :   %s" % self.pixelsize    
        footer+="\n# CCDLabel    :   %s" % self.CCDLabel
        
        header2="2theta   chi    x    y  I"
        np.savetxt(corfileobj,Toedit_sorted,fmt='%.8f',header=header2,footer=footer,comments="")


        # for line in np.vstack((headerarray, np.array(Toedit_sorted, dtype="|S11"))):
        #     print("line",line)
        #     linux = str(np.concatenate(line[0]))
        #     corfileobj.write(linux)
        #     corfileobj.write("\n")

        # corfileobj.write("\n# Cor file generated by LaueTools Polygrains Simulation Software")
        # corfileobj.write(
        #     "\n# File created at %s by by ParametricLaueSimulator.py" % (time.asctime())
        # )
        # corfileobj.write("\n# Calibration parameters")
        # for par, value in zip(["dd", "xcen", "ycen", "xbet", "xgam"], self.calib):
        #     corfileobj.write("\n# %s     :   %s" % (par, value))
        # corfileobj.write("\n# pixelsize    :   %s" % self.pixelsize)
        # corfileobj.write("\n# ypixelsize    :   %s" % self.pixelsize)      
        # corfileobj.write("\n# CCDLabel    :   %s" % self.CCDLabel)

        corfileobj.close()

    def OnSave(self, event, data_res):

        textfile = open(os.path.join(self.dirname, self.simul_filename), "w")

        textfile.write(Edit_String_SimulData(data_res))

        textfile.close()

        fullname = os.path.join(self.dirname, self.simul_filename)
        wx.MessageBox("File saved in %s" % fullname, "INFO")

    def opendir(self, event):
        dlg = wx.DirDialog(
            self,
            "Choose a directory:self.control.SetValue(str(self.indexed_spots).replace('],',']\n'))",
            style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON,
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.dirname = dlg.GetPath()

            print(self.dirname)

        dlg.Destroy()

    def OnQuit(self, event):
        #        print "Current selected grains ", self.SelectGrains
        self.Close()

    def OnSimulate(self, event):
        """
        in parametric transformation simulation parametric_Grain_Dialog3in 
        """
        # select all created grain and build dict self.SelectGrains
        self.Select_AllGrain(event)
        # list of parameters for parent and child grains
        list_param = Construct_GrainsParameters_parametric(self.SelectGrains)

        if 0:
            print("list_param in parametric", list_param)
            print("nb grains", len(list_param))
            print(
                "\n******************\ndict transform\n*************\n",
                self.dict_transform,
            )
            print("self.SelectGrains", self.SelectGrains)

        if len(list_param) == 0:
            dlg = wx.MessageDialog(
                self,
                "You must create and select at least one grain!",
                "Empty Grains list",
                wx.OK | wx.ICON_ERROR,
            )
            dlg.ShowModal()
            dlg.Destroy()
            return True

        self.emin = self.rightpanel.scmin.GetValue()
        self.emax = self.rightpanel.scmax.GetValue()
        if self.rightpanel.rbtop.GetValue():
            cameraposition = "Z>0"
        elif self.rightpanel.rbside.GetValue():
            cameraposition = "Y>0"
        elif self.rightpanel.rbtransmission.GetValue():
            cameraposition = "X>0"
        else:
            cameraposition = "Y<0"

        try:
            self.Det_distance = float(self.rightpanel.detdist.GetValue())
            self.Det_diameter = float(self.rightpanel.detdiam.GetValue())
            self.Xcen = float(self.rightpanel.xcen.GetValue())
            self.Ycen = float(self.rightpanel.ycen.GetValue())
            self.Xbet = float(self.rightpanel.xbet.GetValue())
            self.Xgam = float(self.rightpanel.xgam.GetValue())
            self.pixelsize = float(self.rightpanel.ctrlpixelsize.GetValue())
        except ValueError:
            dlg = wx.MessageDialog(
                self,
                "Detector parameters must be float with dot separator",
                "Bad Input Parameters",
                wx.OK | wx.ICON_ERROR,
            )
            dlg.ShowModal()
            dlg.Destroy()
            return True

        showExperimenalData = self.rightpanel.checkshowExperimenalData.GetValue()
        showExperimentalImage = self.rightpanel.checkExperimenalImage.GetValue()

        # show markers experimental list of peaks
        if showExperimenalData:
            if self.initialParameters["ExperimentalData"] is None:
                dlg = wx.MessageDialog(
                    self,
                    "You must load experimental data(File/Open Menu) before or uncheck Show Exp. Data box",
                    "Experimental Data Missing!",
                    wx.OK | wx.ICON_ERROR,
                )
                dlg.ShowModal()
                dlg.Destroy()
                return True

        # show 2D pixel intensity from image file
        ImageArray = None
        if showExperimentalImage:
            import readmccd as RMCCD

            fullpathimagename = str(self.rightpanel.expimagetxtctrl.GetValue())
            if not os.path.isfile(fullpathimagename):
                dlg = wx.MessageDialog(
                    self,
                    "Image file : %s\n\ndoes not exist!!" % fullpathimagename,
                    "error",
                    wx.OK | wx.ICON_ERROR,
                )
                #                 dlg = wx.MessageDialog(self, 'Detector parameters must be float with dot separator',
                #                                    'Bad Input Parameters',)
                dlg.ShowModal()
                dlg.Destroy()
                return

            ImageArray, framedim, fliprot = RMCCD.readCCDimage(
                fullpathimagename, self.CCDLabel, dirname=None
            )

            self.rightpanel.pt_XYCCD.SetValue(True)

        if self.rightpanel.pt_2thetachi.GetValue():
            plottype = "2thetachi"
        elif self.rightpanel.pt_XYCCD.GetValue():
            plottype = "XYmar"
        else:
            plottype = "XYfit2d"

        self.textprocess.SetLabel("Processing Laue Simulation")
        self.gauge.SetRange(len(list_param) * 10000)

        # simulation in class parametric_Grain_Dialog3

        self.calib = [self.Det_distance, self.Xcen, self.Ycen, self.Xbet, self.Xgam]
        data_res = dosimulation_parametric(
            list_param,
            emax=self.emax,
            emin=self.emin,
            showplot=self.rightpanel.showplotBox.GetValue(),
            showExperimenalData=showExperimenalData,
            plottype=plottype,
            detectordistance=self.Det_distance,
            detectordiameter=self.Det_diameter,
            posCEN=(self.Xcen, self.Ycen),
            cameraAngles=(self.Xbet, self.Xgam),
            gauge=self.gauge,
            kf_direction=cameraposition,
            Transform_params=self.dict_transform,
            SelectGrains=self.SelectGrains,
            pixelsize=self.pixelsize,
            framedim=self.framedim,
        )

        (
            list_twicetheta,
            list_chi,
            list_energy,
            list_Miller,
            list_posX,
            list_posY,
            ListName,
            nb_g_t,
            calib,
            total_nb_grains,
        ) = data_res

        print("len(list_posX)", len(list_posX))
        print("len(list_posY)", len(list_posY))
        print("len(list_posX[0])", len(list_posX[0]))

        # find transform for slip systems
        print("\n\ndata_res[7]", nb_g_t)
        StreakingData = None
        for elem in nb_g_t:
            grainindex, nb_transforms, transformtype = elem
            if transformtype in ("slipsystem",):
                print("there is a slipsystem simulation")
                StreakingData = data_res
        # ------------------------------------------------
        # plot results --------------------------------------
        if self.rightpanel.showplotBox.GetValue():
            # experimental data
            if showExperimenalData:
                experimentaldata_2thetachi = (
                    self.data_2theta,
                    self.data_chi,
                    self.data_I,
                )
                experimentaldata_XYMAR = self.data_pixX, self.data_pixY, self.data_I
                experimentaldata_XYfit2D = (
                    self.data_pixX,
                    self.initialParameters["framedim"][1] - self.data_pixY,
                    self.data_I,
                )  # TODO: to be checked
            else:
                experimentaldata_2thetachi = None
                experimentaldata_XYMAR = None
                experimentaldata_XYfit2D = None

            # theoretical data
            if plottype == "2thetachi":
                simulframe = SimulationPlotFrame(
                    self,
                    -1,
                    "LAUE Pattern simulation visualisation frame",
                    data=(
                        list_twicetheta,
                        list_chi,
                        list_energy,
                        list_Miller,
                        total_nb_grains,
                        plottype,
                        experimentaldata_2thetachi,
                    ),
                    GrainName_for_Streaking=None,
                    list_grains_transforms=nb_g_t,
                    Size=(6, 4),
                    CCDLabel=self.CCDLabel,
                )
            elif plottype == "XYmar":
                simulframe = SimulationPlotFrame(
                    self,
                    -1,
                    "LAUE Pattern simulation visualisation frame",
                    data=(
                        list_posX,
                        list_posY,
                        list_energy,
                        list_Miller,
                        total_nb_grains,
                        "XYMar",
                        experimentaldata_XYMAR,
                    ),
                    ImageArray=ImageArray,
                    GrainName_for_Streaking=StreakingData,
                    list_grains_transforms=nb_g_t,
                    Size=(6, 4),
                    CCDLabel=self.CCDLabel,
                )

            elif plottype == "XYfit2d":
                newlist_posY = [
                    [
                        self.initialParameters["framedim"][1] - positionY
                        for positionY in childlistY
                    ]
                    for childlistY in list_posY
                ]
                simulframe = SimulationPlotFrame(
                    self,
                    -1,
                    "LAUE Pattern simulation visualisation frame",
                    data=(
                        list_posX,
                        newlist_posY,
                        list_energy,
                        list_Miller,
                        total_nb_grains,
                        "pixels",
                        experimentaldata_XYfit2D,
                    ),
                    Size=(6, 4),
                    CCDLabel=self.CCDLabel,
                )

            simulframe.Show(True)

        self.textprocess.SetLabel("Laue Simulation Completed")

        if self.savefileBox.GetValue():
            if self.rb1.GetValue():
                file_name = self.textcontrolfilemanual.GetValue() + ".sim"
            else:
                file_name = (
                    self.textcontrolfileauto.GetValue()
                    + str(self.initialParameters["indexsimulation"])
                    + ".sim"
                )
                self.initialParameters["indexsimulation"] += 1
                print("Next index is %s" % self.initialParameters["indexsimulation"])

            self.simul_filename = file_name
            print("Simulation file saved in %s " % self.simul_filename)
            self.OnSave(event, data_res)

        if self.corfileBox.GetValue():

            file_name_fake = self.corcontrolfake.GetValue() + ".cor"

            print("Fake data in file %s " % file_name_fake)
            self.OnWriteCorFile(file_name_fake, data_res, len(list_param))


def Read_Grainparameter_cont(param):
    """
    Read dictionary of simulation inpu key parameters
    """
    # Elem, Extinc, Transf_a, Rot, Bmatrix, Transf_c, GrainName, transform = param
    key_material, Extinc, Transf_a, Rot, Bmatrix, Transf_c = param[:6]
    GrainName = str(param[6])
    # print "param in Read_Grainparameter_cont",param

    Extinctions = DictLT.dict_Extinc[str(Extinc)]
    Transform_labframe = DictLT.dict_Transforms[Transf_a]
    orientMatrix = DictLT.dict_Rot[Rot]
    B_matrix = DictLT.dict_Vect[Bmatrix]
    Transform_crystalframe = DictLT.dict_Transforms[Transf_c]

    return (
        [
            key_material,
            Extinctions,
            Transform_labframe,
            orientMatrix,
            B_matrix,
            Transform_crystalframe,
        ],
        GrainName,
    )


def Construct_GrainsParameters_parametric(SelectGrains_parametric):
    """
    return list of simulation parameters for each grain set (mother and children grains)
    """
    list_selectgrains_param = []
    # keys from dialogs were in reverse order
    # self.SelectGrains_parametric  == parametric_Grain_Dialog().SelectGrains
    for key_grain in list(SelectGrains_parametric.keys())[::-1]:
        # print "self.SelectGrains_parametric[key_grain]",self.SelectGrains_parametric[key_grain]
        list_selectgrains_param.append(
            Read_Grainparameter_cont(SelectGrains_parametric[key_grain])
        )
    # print list_selectgrains_param
    return list_selectgrains_param


def dosimulation_parametric(
    _list_param,
    Transform_params=None,
    SelectGrains=None,
    emax=25.0,
    emin=5.0,
    detectordistance=68.7,
    detectordiameter=165.0,
    posCEN=(1024.0, 1024.0),
    cameraAngles=(0.0, 0.0),
    showplot=True,
    showExperimenalData=False,
    gauge=None,
    plottype="2thetachi",
    kf_direction="Z>0",
    pixelsize=165.0 / 2048,
    framedim=(2048, 2048),
):
    """
    Simulation of orientation or deformation gradient.
    From parent grain simulate a list of transformations (deduced by a parametric variation)

    _list_param   : list of parameters for each grain  [grain parameters, grain name]

    posCEN =(Xcen, Ycen)
    cameraAngles =(Xbet, Xgam)

    TODO:simulate for any camera position
    TODO: simulate spatial distribution of laue pattern origin
    """
    print("\n\n********* Starting dosimulation_parametric *********\n\n")

    # Number_ParentGrains parent grains
    Number_ParentGrains = len(_list_param)

    # Extracting parent grains  simluation param and name
    # creating list of simulation parameters for parent grains
    ListParam = []
    ParentGrainName_list = []
    for m in range(Number_ParentGrains):
        # print "_list_param[m]", _list_param[m]
        _paramsimul, _grainname = _list_param[m]

        elem = np.shape(np.array(_paramsimul[0]))
        # print "elem parametric",elem

        # convert EULER angles to matrix in case of  3 input elements
        if np.shape(np.array(_paramsimul[2])) == (3,):
            _paramsimul[2] = GT.fromEULERangles_toMatrix(_paramsimul[2])

        ListParam.append(_paramsimul)
        ParentGrainName_list.append(_grainname)

    #            print "ListParam in dosimulation_parametric", ListParam
    #            print "ParentGrainName_list", ParentGrainName_list

    # Calculating Laue spots of each parent grain ----------------------------

    print("Doing simulation with %d parent grains" % Number_ParentGrains)
    list_twicetheta = []
    list_chi = []
    list_energy = []
    list_Miller = []
    list_posX = []
    list_posY = []

    total_nb_grains = 0

    # list of [Parent grain index,Number of corresponding transforms, transform_type]
    list_ParentGrain_transforms = []

    if gauge:
        gaugecount = 0
        gauge.SetValue(gaugecount)
        # gauge count max has been set to 1000*Number_ParentGrains

    # loop over parent grains
    for parentgrain_index in range(Number_ParentGrains):

        # read simulation parameters
        name_of_grain = ParentGrainName_list[parentgrain_index]
        print("\n\n %d, name_of_grain: %s" % (parentgrain_index, name_of_grain))

        Laue_classic_param = ListParam[parentgrain_index]

        key_material, Extinc, Ta, U, B, Tc = Laue_classic_param  # from combos

        # build GrainSimulParam
        GrainSimulParam = [0, 0, 0, 0]

        # user has entered his own B matrix
        if key_material == "inputB":
            # print "\n**************"
            # print "using Bmatrix containing lattice parameter"
            # print "****************\n"

            # take then parameters from combos
            GrainSimulParam[0] = B
            GrainSimulParam[1] = Extinc
            GrainSimulParam[2] = np.dot(np.dot(Ta, U), Tc)
            GrainSimulParam[3] = "inputB"

        # user uses a pre defined B matrix contain in material dictionnary
        # (need to read lattice parameter or element definition
        # then compute B matrix)
        elif key_material != "inputB":

            grain = CP.Prepare_Grain(key_material, np.eye(3))

            # print "grain in dosimulation_parametric() input Element",grain

            B0, Extinc0, U0, key = grain  # U0 is identity

            # new B matrix
            newB = np.dot(Tc, np.dot(B, B0))
            # Extinction is overwritten by value in comboExtinc
            newExtinc = Extinc
            # new U matrix
            newU = np.dot(Ta, np.dot(U, U0))

            GrainSimulParam = [newB, newExtinc, newU, key_material]

            print("Using following parameters from Material Dict.")
            print(DictLT.dict_Materials[key_material])

        # --- Simulate
        print("GrainSimulParam in dosimulation_parametric() input Element")
        print(GrainSimulParam)

        spots2pi = LAUE.getLaueSpots(
            DictLT.CST_ENERGYKEV / emax,
            DictLT.CST_ENERGYKEV / emin,
            [GrainSimulParam],  # bracket because of a list of one grain
            [[""]],
            fastcompute=0,
            fileOK=0,
            verbose=0,
            kf_direction=kf_direction,
        )

        # q vectors in lauetools frame, miller indices
        # print "spots2pi",spots2pi

        # ---------  [list of 3D vectors],[list of corresponding Miller indices]
        Qvectors_ParentGrain, HKLs_ParentGrain = spots2pi

        if gauge:
            gaugecount += 100
            # print "gaugecount += 100",gaugecount
            gauge.SetValue(gaugecount)
            wx.Yield()

        # --- Calculating small deviations(rotations and strain)
        # --- from parent grains according to transform

        # print " in simul Transform_params",Transform_params
        # print " in simul SelectGrains",SelectGrains

        # get transform
        if Transform_params is None:
            Transform_listparam = [""]
        elif Transform_params is not None:
            Transform_params[""] = [""]

            # print "SelectGrains[name_of_grain]",SelectGrains[name_of_grain]
            Transform_listparam = Transform_params[SelectGrains[name_of_grain][7]]
            if Transform_listparam == "":
                Transform_listparam = [""]

        # print "Transform_listparam",Transform_listparam

        nb_transforms = 1
        print("GrainSimulParam", GrainSimulParam)
        matrix_list = [np.eye(3)]

        # matrix giving a*,b*,c* in absolute x, y,z frame
        # this matrix is used for strain a*,b*,c* are NORMALIZED:
        # this is a B matrix used in  q= U B G* formalism
        # this matrix represents also an initial orientation

        # old way
        # InitMat = Bmatrix
        # # matrix of additional dilatation in Reciprocal space of a* b* c* of the lattice giving the proper length of a*,b*,c*
        # mat_dilatinRS = np.array([[Laue_classic_param[1][0],0, 0],[0, Laue_classic_param[1][1],0],[0, 0,Laue_classic_param[1][2]]])  # in a* b* c* frame

        # # Rotation matrix from initial orientation to the given orientation   Newpostion = R oldposition    in x, y,z frame
        # matOrient_pure = np.array(Laue_classic_param[2])
        # # full matrix containing

        # matOrient = np.dot(matOrient_pure, np.dot(InitMat, mat_dilatinRS))  # U * B *(diagonal three elements reciprocal dilatation matrix)
        # print "matOrient in dosimulation_parametric",matOrient

        # Calculates matOrient which is U*B in q = U*B*Gstar
        matOrient = np.dot(GrainSimulParam[2], GrainSimulParam[0])

        if Transform_listparam != "":
            # print "Transform_listparam[0]",Transform_listparam[0]
            if Transform_listparam[0] == "r_axis":
                axis_list = Transform_listparam[2]
                angle_list = Transform_listparam[1]
                nb_transforms = len(angle_list)

            elif Transform_listparam[0] in ("r_axis_d", "r_axis_d_slipsystem"):
                axis_list = Transform_listparam[2]
                #                 print "axis_list before orientation in d frame", axis_list
                angle_list = Transform_listparam[1]
                nb_transforms = len(angle_list)
                # axis coordinate change from abc frame(direct crystal) to a*b*c* frame( reciprocal crystal)
                axis_list_c = np.array(
                    [
                        CP.fromrealframe_to_reciprocalframe(ax, GrainSimulParam[0])
                        for ax in axis_list
                    ]
                )
                #                 print "axis_list_c", axis_list_c
                # axis coordinate change from a*b*c* frame(crystal) to absolute frame
                axis_list = np.dot(matOrient, axis_list_c.T).T
            #                 print "axis_list in absolute frame from d frame", axis_list

            # general transform expressed in absolute lauetools frame
            elif Transform_listparam[0] == "r_axis_c":
                axis_list = Transform_listparam[2]
                # print "axis_list before orientation in c frame",axis_list
                angle_list = Transform_listparam[1]
                nb_transforms = len(angle_list)
                # axis coordinate change from hkl frame(crystal) to absolute frame
                axis_list = np.dot(matOrient, axis_list.T).T
            #                 print "axis_list in absolute frame from c frame", axis_list

            # general transform expressed in absolute lauetools frame
            elif Transform_listparam[0] == "r_mat":
                matrix_list = Transform_listparam[1]
                nb_transforms = len(matrix_list)

                # general transform expressed in crystal frame

            elif Transform_listparam[0] == "r_mat_d":
                raise ValueError(
                    "r_mat_d matrix transform with d frame is not implemented yet"
                )

            elif Transform_listparam[0] == "r_mat_c":
                # print "using r_mat_c"
                matrix_list = Transform_listparam[1]
                nb_transforms = len(matrix_list)
                # then convert transform in absolute lauetools frame
                for k in range(nb_transforms):
                    matrix_list[k] = np.dot(
                        matOrient, np.dot(matrix_list[k], np.linalg.inv(matOrient))
                    )
                    # matrix_list[k] = np.dot(inv(matOrient),np.dot(matrix_list[k],matOrient))

                # transform is a list of tensile transforms
            elif type(Transform_listparam[0]) == type(["1", "2", "3"]):
                # is a list of 's_axis' or 's_axis_c
                liststrainframe = Transform_listparam[0]
                # list of 3 arrays, each array contains the nb_transforms axis
                # (array of three elements)
                _axis_list = Transform_listparam[2]
                # list of 3 arrays, each array contains the nb_transforms strain factor
                factor_list = Transform_listparam[1]

                # print "axis_list in c frame",_axis_list
                # print "factor_list",factor_list
                nb_transforms = len(factor_list[0])

                axis_list = [np.ones(3) for k in range(3)]
                # loop over the three proposed axial strain in simulation board
                for mm in range(3):
                    if liststrainframe[mm] == "s_axis_c":
                        axis_list[mm] = np.dot(
                            matOrient, np.transpose(_axis_list[mm])
                        ).T
                    else:
                        axis_list[mm] = _axis_list[mm]
                # print "axis_list in a frame",axis_list
                # print "Transform_listparam[2]",Transform_listparam[2]

        #         print "HKLs_ParentGrain", HKLs_ParentGrain
        #         print "nb_transforms", nb_transforms
        #         print "Transform_listparam[0]", Transform_listparam[0]

        # -----------------------------------------------------
        # loop over child grains derived from transformation of a single parent grain
        for ChildGrain_index in range(nb_transforms):
            # Qvectors_ParentGrain is used to create Qvectors_ParentGrain for each chold grain
            # according to the transform

            # print "Qvectors_ParentGrain", Qvectors_ParentGrain

            # Geometrical transforms for each case
            # loop over reciprocal lattice vectors is done with numpy array functions

            # for rotation around axis expressed in any frame
            if Transform_listparam[0] in (
                "r_axis",
                "r_axis_c",
                "r_axis_d",
                "r_axis_d_slipsystem",
            ):
                # print "angle, axis",angle_list[ChildGrain_index],axis_list[ChildGrain_index]
                qvectors_ChildGrain = GT.rotate_around_u(
                    Qvectors_ParentGrain[0],
                    angle_list[ChildGrain_index],
                    u=axis_list[ChildGrain_index],
                )
                # list of spot which are on camera(without harmonics)
                # hkl are common to all child grains
                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain

            # for general transform expressed in any frame
            elif (
                Transform_listparam[0] == "r_mat"
                or Transform_listparam[0] in ("r_mat_c", "r_mat_d")
                or Transform_listparam == ""
                or Transform_listparam == [""]
            ):

                # general transformation is applied to q vector
                # expressed in lauetools absolute frame
                qvectors_ChildGrain = np.dot(
                    matrix_list[ChildGrain_index], Qvectors_ParentGrain[0].T
                ).T

                if 0:
                    print(
                        " 10 first transpose(Qvectors_ParentGrain[0])",
                        Qvectors_ParentGrain[0].T[:, :10],
                    )
                    print("%d / %d" % (ChildGrain_index, nb_transforms))
                    print("current matrix", matrix_list[ChildGrain_index])
                    print(np.shape(qvectors_ChildGrain))
                    print("GrainSimulParam", GrainSimulParam)
                    print("qvectors_ChildGrain", qvectors_ChildGrain[:10])
                # list of spot which are on camera(without harmonics)
                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain

            # for the three consecutive axial strains
            elif type(Transform_listparam[0]) == type(["1", "2", "3"]):

                first_traction = GT.tensile_along_u(
                    Qvectors_ParentGrain[0],
                    factor_list[0][ChildGrain_index],
                    u=axis_list[0][ChildGrain_index],
                )
                second_traction = GT.tensile_along_u(
                    first_traction,
                    factor_list[1][ChildGrain_index],
                    u=axis_list[1][ChildGrain_index],
                )
                qvectors_ChildGrain = GT.tensile_along_u(
                    second_traction,
                    factor_list[2][ChildGrain_index],
                    u=axis_list[2][ChildGrain_index],
                )
                # list of spots for a child grain (on camera + without harmonics)
                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain

                if 1:
                    print(
                        " 10 first transpose(Qvectors_ParentGrain[0])",
                        Qvectors_ParentGrain[0][:10],
                    )
                    print(np.shape(qvectors_ChildGrain))
                    print("GrainSimulParam", GrainSimulParam)
                    print("qvectors_ChildGrain", qvectors_ChildGrain[:10])

            else:
                # no transformation
                #                 print "\n No trandformation \n"
                pass

            # test whether there is at least one Laue spot in the camera
            for elem in spots2pi[0]:
                if len(elem) == 0:
                    print(
                        "There is at least one child grain without peaks on CCD camera for ChildGrain_index= %.3f"
                        % ChildGrain_index
                    )
                    break

            # ---------------------------------
            # filter spots to keep those in camera, filter harmonics
            try:
                print("kf_direction = ", kf_direction)
                if kf_direction == "Z>0" or isinstance(
                    kf_direction, list
                ):  # or isinstance(kf_direction, np.array):
                    Laue_spot_list = LAUE.filterLaueSpots(
                        spots2pi,
                        fileOK=0,
                        fastcompute=0,
                        detectordistance=detectordistance,
                        detectordiameter=detectordiameter,  # * 1.2, # avoid losing some spots in large transformation
                        kf_direction=kf_direction,
                        HarmonicsRemoval=1,
                        pixelsize=pixelsize,
                    )

                    # for elem in Laue_spot_list[0][:10]:
                    # print elem

                    # print "Laue_spot_list[0][0].Twicetheta"
                    # print Laue_spot_list[0][0].Twicetheta

                    if gauge:
                        gaugecount = gaugecount + 900 / nb_transforms
                        gauge.SetValue(gaugecount)
                        wx.Yield()
                        # print "ChildGrain_index 900%nb_transforms",ChildGrain_index, gaugecount

                    twicetheta = [spot.Twicetheta for spot in Laue_spot_list[0]]
                    chi = [spot.Chi for spot in Laue_spot_list[0]]
                    energy = [
                        spot.EwaldRadius * DictLT.CST_ENERGYKEV
                        for spot in Laue_spot_list[0]
                    ]
                    Miller_ind = [list(spot.Millers) for spot in Laue_spot_list[0]]

                    calib = [
                        detectordistance,
                        posCEN[0],
                        posCEN[1],
                        cameraAngles[0],
                        cameraAngles[1],
                    ]

                    print("calib parameters in dosimulation_parametric")
                    print(calib)
                    print("pixelsize", pixelsize)
                    print("framedim", framedim)

                    posx, posy = F2TC.calc_xycam_from2thetachi(
                        twicetheta,
                        chi,
                        calib,
                        pixelsize=pixelsize,
                        signgam=DictLT.SIGN_OF_GAMMA,
                        dim=framedim,
                        kf_direction=kf_direction,
                    )[:2]
                    # posx, posy, theta0 = F2TC.calc_xycam_from2thetachi(twicetheta, chi, calib, pixelsize = self.pixelsize, signgam = SIGN_OF_GAMMA)

                    #                    vecRR = [spot.Qxyz for spot in Laue_spot_list[0]] #uf_lab in JSM LaueTools frame

                    # print "twicetheta",twicetheta
                    # print "2*th0",(2*theta0).tolist()

                    list_twicetheta.append(twicetheta)
                    list_chi.append(chi)
                    list_energy.append(energy)
                    list_Miller.append(Miller_ind)
                    list_posX.append(posx.tolist())
                    list_posY.append(posy.tolist())

                    success = 1

                elif kf_direction in ("Y<0", "Y>0"):
                    # TODO: patch for test: detectordistance = 126.5

                    Laue_spot_list = LAUE.filterLaueSpots(
                        spots2pi,
                        fileOK=0,
                        fastcompute=0,
                        detectordistance=detectordistance,
                        detectordiameter=detectordiameter,  # * 1.2, # avoid losing some spots in large transformation
                        kf_direction=kf_direction,
                        HarmonicsRemoval=1,
                        pixelsize=pixelsize,
                    )

                    # for elem in Laue_spot_list[0][:10]:
                    # print elem

                    # print "Laue_spot_list[0][0].Twicetheta"
                    # print Laue_spot_list[0][0].Twicetheta
                    if gauge:
                        gaugecount = gaugecount + 900 / nb_transforms
                        gauge.SetValue(gaugecount)
                        wx.Yield()
                        # print "ChildGrain_index 900%nb_transforms",ChildGrain_index, gaugecount

                    twicetheta = [spot.Twicetheta for spot in Laue_spot_list[0]]
                    chi = [spot.Chi for spot in Laue_spot_list[0]]
                    energy = [
                        spot.EwaldRadius * DictLT.CST_ENERGYKEV
                        for spot in Laue_spot_list[0]
                    ]
                    Miller_ind = [list(spot.Millers) for spot in Laue_spot_list[0]]

                    calib = [
                        detectordistance,
                        posCEN[0],
                        posCEN[1],
                        cameraAngles[0],
                        cameraAngles[1],
                    ]

                    posx, posy = F2TC.calc_xycam_from2thetachi(
                        twicetheta,
                        chi,
                        calib,
                        pixelsize=pixelsize,
                        signgam=DictLT.SIGN_OF_GAMMA,
                        kf_direction=kf_direction,
                    )[:2]
                    # posx, posy, theta0 = F2TC.calc_xycam_from2thetachi(twicetheta, chi, calib, pixelsize = self.pixelsize, signgam = SIGN_OF_GAMMA)

                    posx = [spot.Xcam for spot in Laue_spot_list[0]]
                    posy = [spot.Ycam for spot in Laue_spot_list[0]]

                    vecRR = [
                        spot.Qxyz for spot in Laue_spot_list[0]
                    ]  # uf_lab in JSM frame

                    # print "twicetheta",twicetheta
                    # print "2*th0",(2*theta0).tolist()

                    list_twicetheta.append(twicetheta)
                    list_chi.append(chi)
                    list_energy.append(energy)
                    list_Miller.append(Miller_ind)
                    list_posX.append(posx)
                    list_posY.append(posy)

                    success = 1

                elif kf_direction in ("X>0",):  # transmission mode
                    print("\n*****\nSimulation in transmission mode\n*****\n")

                    Laue_spot_list = LAUE.filterLaueSpots(
                        spots2pi,
                        fileOK=0,
                        fastcompute=0,
                        detectordistance=detectordistance,
                        detectordiameter=detectordiameter,  # * 1.2, # avoid losing some spots in large transformation
                        kf_direction=kf_direction,
                        HarmonicsRemoval=1,
                        pixelsize=pixelsize,
                    )

                    # for elem in Laue_spot_list[0][:10]:
                    # print elem

                    # print "Laue_spot_list[0][0].Twicetheta"
                    # print Laue_spot_list[0][0].Twicetheta
                    if gauge:
                        gaugecount = gaugecount + 900 / nb_transforms
                        gauge.SetValue(gaugecount)
                        wx.Yield()
                        # print "ChildGrain_index 900%nb_transforms",ChildGrain_index, gaugecount

                    twicetheta = [spot.Twicetheta for spot in Laue_spot_list[0]]
                    chi = [spot.Chi for spot in Laue_spot_list[0]]
                    energy = [
                        spot.EwaldRadius * DictLT.CST_ENERGYKEV
                        for spot in Laue_spot_list[0]
                    ]
                    Miller_ind = [list(spot.Millers) for spot in Laue_spot_list[0]]

                    calib = [
                        detectordistance,
                        posCEN[0],
                        posCEN[1],
                        cameraAngles[0],
                        cameraAngles[1],
                    ]

                    posx, posy = F2TC.calc_xycam_from2thetachi(
                        twicetheta,
                        chi,
                        calib,
                        pixelsize=pixelsize,
                        signgam=DictLT.SIGN_OF_GAMMA,
                        kf_direction=kf_direction,
                    )[:2]

                    posx = posx.tolist()
                    posy = posy.tolist()
                    # posx, posy, theta0 = F2TC.calc_xycam_from2thetachi(twicetheta, chi, calib, pixelsize = self.pixelsize, signgam = SIGN_OF_GAMMA)

                    #                     posx = [spot.Xcam for spot in  Laue_spot_list[0]]
                    #                     posy = [spot.Ycam for spot in  Laue_spot_list[0]]

                    vecRR = [
                        spot.Qxyz for spot in Laue_spot_list[0]
                    ]  # uf_lab in JSM frame

                    # print "twicetheta",twicetheta
                    # print "2*th0",(2*theta0).tolist()

                    list_twicetheta.append(twicetheta)
                    list_chi.append(chi)
                    list_energy.append(energy)
                    list_Miller.append(Miller_ind)
                    list_posX.append(posx)
                    list_posY.append(posy)

                    success = 1

            except UnboundLocalError:
                txt = "With theses parameters, there are no peaks in the CCD frame!!\n"
                txt += "for transform with t = %.3f\n" % ChildGrain_index
                txt += "It may seem that the transform you have designed has a too large amplitude\n"
                txt += "-Try then to reduce the variation range of t\n"
                txt += "-Or reduce ratio between extrema in input matrix transform\n\n"

                #                dlg = wx.MessageDialog(self, txt, 'Pattern without peaks!', wx.OK | wx.ICON_ERROR)
                #                dlg.ShowModal()
                #                dlg.Destroy()

                success = 0
                break

            # end of loop over transforms(or children grains)

        transform_type = "parametric"
        print("Transform_listparam", Transform_listparam)
        for elem in Transform_listparam[0]:
            if elem.endswith("slipsystem"):
                transform_type = "slipsystem"
        list_ParentGrain_transforms.append(
            [parentgrain_index, nb_transforms, transform_type]
        )
        total_nb_grains += nb_transforms

        if gauge:
            gaugecount += (parentgrain_index + 1) * 1000
            gauge.SetValue(gaugecount)
            wx.Yield()
            # print "(parentgrain_index+1)*1000",gaugecount

    # end of loop over parent grains

    print("total_nb_grains", total_nb_grains)

    # 1 grain data lists are listed i.e. use list_twicetheta[0] etc.
    # polygrain use list_twicetheta

    print("List of Grain Name", ParentGrainName_list)
    print("Number_ParentGrains of parent grains", Number_ParentGrains)
    print("Number_ParentGrains of spots in grain0", len(list_twicetheta[0]))

    data = (
        list_twicetheta,
        list_chi,
        list_energy,
        list_Miller,
        list_posX,
        list_posY,
        ParentGrainName_list,
        list_ParentGrain_transforms,
        calib,
        total_nb_grains,
    )

    return data


# end dosimulation_parametric
def Edit_String_SimulData(data):
    """
    return string object made of lines which contains laue spots properties 
    
    Called to be Edited in LaueToolsframe.control

    params:

    data tuple of spots properties    
    data =(list_twicetheta, list_chi, list_energy, list_Miller, list_posX, list_posY,
           ListName, nb of(parent) grains,
            calibration parameters list, total nb of grains)
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
        #        self.control.SetValue(lines)
        return lines

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
        #        self.control.SetValue(lines)
        return lines


if __name__ == "__main__":
    title = "test"

    title2 = "test2"

    data_X = np.arange(20, 90)
    data_Y = np.random.randint(100, size=len(data_X))

    dataXY = np.array([data_X, data_Y])

    initialParameters = {}
    initialParameters["CalibrationParameters"] = [70, 1024, 1024, -1.2, 0.89]
    initialParameters["prefixfilenamesimul"] = "Ge_blanc_"
    initialParameters["indexsimulation"] = 0
    initialParameters["ExperimentalData"] = None
    initialParameters["pixelsize"] = 165 / 2048.0
    initialParameters["framedim"] = (2048, 2048)
    initialParameters["CCDLabel"] = "MARCCD165"

    GUIApp = wx.App()
    GUIframe = parametric_Grain_Dialog3(None, -1, title, initialParameters)
    #    GUIframe = parametric_Grain_Dialog(None, -1, title, initialParameters)
    GUIframe.Show()

    GUIApp.MainLoop()
