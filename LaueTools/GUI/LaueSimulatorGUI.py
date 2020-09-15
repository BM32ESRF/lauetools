# -*- coding: utf-8 -*-
r"""
GUI Module to simulate Laue Patterns from several crystals in various geometry

Main author is J. S. Micha:   micha [at] esrf [dot] fr

version July 2019
from LaueTools package for python2 hosted in

http://sourceforge.net/projects/lauetools/

or for python3 and 2 in

https://gitlab.esrf.fr/micha/lauetools

To be launched (to deal with relative imports)
> python -m LaueTools.LaueSimulatorGUI
"""
import os
import sys
import time

import numpy as np
import wx

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
    from .. import dict_LaueTools as DictLT
    from . SimulFrame import SimulationPlotFrame #, getindices_StreakingData
    from .. import CrystalParameters as CP
    from .. import multigrainsSimulator as MGS
    from .. import IOimagefile as IOimage
    from .. import indexingImageMatching as IMM

else:
    import dict_LaueTools as DictLT
    from GUI.SimulFrame import SimulationPlotFrame #, getindices_StreakingData
    import CrystalParameters as CP
    import multigrainsSimulator as MGS
    import IOimagefile as IOimage
    import indexingImageMatching as IMM


class TransformPanel(wx.Panel):
    """
    GUI class to set parameters to define a set of geometrical transforms (strain, orientatio)
    and store it in dict_transform to be used by the simulator
    """
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        # print("mainframe of TransformPanel", self.mainframe)

        self.SelectGrains = {}

        # list Control for selected grains for SIMULATION
        font3 = self.mainframe.font3

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
        self.tc_rotmatrix = wx.TextCtrl(self, 1000, defaultmatrixtransform,
                                    size=(250, 100), style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        self.tc_rotmatrix.SetFocus()
        self.tc_rotmatrix.Bind(wx.EVT_TEXT, self.mainframe.OnTextChanged)
        self.tc_rotmatrix.Bind(wx.EVT_KEY_DOWN, self.mainframe.OnKeyDown)
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
        buttontransform.Bind(wx.EVT_BUTTON, self.mainframe.OnApplytransform)
        buttontransform.SetFont(font3)

        self.mainframe.transform_index = 0
        self.mainframe.dict_transform = {}

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
        tipaxis = "Rotation axis: letter[a1,a2,a3] where:\nletter refers to the frame \n"
        tipaxis += "(a: for absolute Lauetools frame, s: sample frame\n"
        tipaxis += "c: crystal reciprocal frame (reciprocal unit cell basis vectors a*,b*,c*)\n"
        tipaxis += "d: crystal direct frame (direct real unit cell basis vectors a,b,c))\n"
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

    def onEnableStrain(self, _):
        if self.rb_strainaxes.GetValue():
            self.rb_rotId.SetValue(True)

    def onEnableRotation(self, _):
        if self.rb_rotaxis.GetValue() or self.rb_rotmatrix.GetValue():
            self.rb_strainId.SetValue(True)

    def ReadTransform(self):
        r"""
        core function of this GUI panel
        reads toggle radio button and rotation and strain parameter
        returns a list of matrix rotation and strain
        """
        DEG = np.pi / 180.0
        anglesample = DictLT.SAMPLETILT * DEG
        # transform matrix from xs, ys, zs sample frame to x, y,z absolute frame
        # vec / abs = R * vec / sample
        matrot_sample = np.array([[np.cos(anglesample), 0, -np.sin(anglesample)],
                                    [0, 1, 0],
                                    [np.sin(anglesample), 0, np.cos(anglesample)]])
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
                tmin, tmax, step = (float(strlinspace[0]), float(strlinspace[1]), int(strlinspace[2]))
                # print "listrange",tmin, tmax, step
            except ValueError:
                sentence = 'Expression for t variation in ROTATION transform not understood! '
                'Check if there are "," and "]" '
                dlg = wx.MessageDialog(self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR)
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
                frame_axis_rot = self.tc_Rot_axis.GetValue()  # must contain s[exp1(t),exp2(t),exp3(t)]
                angle_rot = str(self.tc_Rot_ang.GetValue())
                framerot = frame_axis_rot[0]
                axisrot = str(frame_axis_rot[2:-1]).split(",")

                # if 0:
                #     print("framerot", framerot)
                #     print("axisrot", axisrot)
                #     print("angle_rot", angle_rot)

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
                    dlg = wx.MessageDialog(self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR)
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
                    return ("r_axis_%s" % framerot, evalangle_rot, np.array(evalaxisrot).T)

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
                        # if the variable 't' appears in formula, then evaluate the formula
                        if "t" in tu[k]:
                            evalmatrot.append(eval(tu[k]))
                        else:
                            evalmatrot.append(eval(tu[k]) * np.ones(len(t)))
                    # print "evalmatrot",evalmatrot
                    evalmat = np.reshape(np.array(evalmatrot).T, (len(t), 3, 3))
                except ValueError:
                    sentence = 'Expression for general expression in ROTATION transform not understood! Check if there are "," and "]". Mathematical operators may be unknown by numpy'
                    dlg = wx.MessageDialog(self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR)
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
                            evalmat[trans] = np.dot(matrot_sample, np.dot(evalmat[trans],
                                                    inv_matrot_sample))

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
                tmin, tmax, step = (float(strlinspace[0]), float(strlinspace[1]), int(strlinspace[2]))
            except ValueError:
                sentence = 'Expression for t variation in STRAIN transform not understood! Check if there are "," and "]" '
                dlg = wx.MessageDialog(self, sentence, "Wrong expression", wx.OK | wx.ICON_ERROR)
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
                list_tc_factor = [self.tc_axe1_factor, self.tc_axe2_factor, self.tc_axe3_factor]

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
                            np.transpose(np.dot(matrot_sample, np.array(evalaxisstrain))))

                    elif framestrain == "c":  # as with 'a' but calculation is done later according to matorient
                        strainID = "s_axis_c"
                        evalfac_strain_list.append(evalfac_strain)
                        evalaxisstrain_list.append(np.array(evalaxisstrain).T)

                    strainIDlist.append(strainID)

                # array of strainaxis ID, array of 3 strain factors, array of corresponding 3 axes
                return strainIDlist, evalfac_strain_list, evalaxisstrain_list

        return


class SlipSystemPanel(wx.Panel):
    """GUI class to set slip systems to be simulated to interpret Laue spots elongation
    """
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.mainframe = parent.GetParent().GetParent()

        self.SelectGrains = {}
        self.Bmatrices = {}

        # list Control for selected grains for SIMULATION
        font3 = self.mainframe.font3

        titlemiddle = wx.StaticText(self, -1, "slip systems rotation Transformations")
        titlemiddle.SetFont(font3)

        buttontransform = wx.Button(self, -1, "Apply transforms", size=(150, 35))
        buttontransform.Bind(wx.EVT_BUTTON, self.mainframe.OnApplytransformSlipSystems)
        buttontransform.SetFont(font3)

        self.mainframe.transform_index = 0
        self.mainframe.dict_transform = {}

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(titlemiddle)

        vbox.Add(buttontransform)

        self.SetBackgroundColour("sky Blue")

        self.SetSizer(vbox)

    def ReadTransform(self):
        """
        build lists of parameters for the simulation of set of grains
        """
        print("ReadTransform  slipsystem")
        Bmatrix = self.mainframe.Bmatrix_current

        # slip system settings
        misorientationangleMAX = 0.5
        nbsteps = 11

        misanglemin = -misorientationangleMAX
        misanglemax = misorientationangleMAX

        angle_rot = np.linspace(misanglemin, misanglemax, num=nbsteps)
        nb_angles = len(angle_rot)

        slipsystemsfcc = DictLT.SLIPSYSTEMS_FCC

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

        self.mainframe = parent.GetParent().GetParent()

        self.fullpathimagefile = None

        # widgets -----------------------------------
        self.SetBackgroundColour("cyan")

        title1 = wx.StaticText(self, -1, "Spectral Band(keV)")
        title1.SetFont(self.mainframe.font3)

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
        title2.SetFont(self.mainframe.font3)
        title25 = wx.StaticText(self, -1, "Detector Parameters")
        title25.SetFont(self.mainframe.font3)

        self.rbtop = wx.RadioButton(self, 200, "Reflection mode top", style=wx.RB_GROUP)
        self.rbside = wx.RadioButton(self, 200, "Reflection mode side +")
        self.rbsideneg = wx.RadioButton(self, 200, "Reflection mode side -")
        self.rbtransmission = wx.RadioButton(self, 200, "Transmission mode")
        self.rbbackreflection = wx.RadioButton(self, 200, "Back Reflection mode")

        self.rbtop.SetValue(True)

        current_param = self.mainframe.initialParameters["CalibrationParameters"]

        txtdd = wx.StaticText(self, -1, "Det.Dist. (mm): ")
        self.detdist = wx.TextCtrl(self, -1, str(current_param[0]), size=(75, -1))
        txtdiam = wx.StaticText(self, -1, "Det. Diam. (mm): ")
        self.detdiam = wx.TextCtrl(self, -1, "165", size=(40, -1))
        txtxcen = wx.StaticText(self, -1, "xcen (pix): ")
        self.xcen = wx.TextCtrl(self, -1, str(current_param[1]), size=(75, -1))
        txtycen = wx.StaticText(self, -1, "ycen (pix): ")
        self.ycen = wx.TextCtrl(self, -1, str(current_param[2]), size=(75, -1))
        txtxbet = wx.StaticText(self, -1, "xbet (deg): ")
        self.xbet = wx.TextCtrl(self, -1, str(current_param[3]), size=(75, -1))
        txtxgam = wx.StaticText(self, -1, "xgam (deg): ")
        self.xgam = wx.TextCtrl(self, -1, str(current_param[4]), size=(75, -1))
        txtpixelsize = wx.StaticText(self, -1, "pixelsize (mm): ")
        self.ctrlpixelsize = wx.TextCtrl(self, -1, str(self.mainframe.pixelsize), size=(75, -1))
        title4 = wx.StaticText(self, -1, "Display Parameters")
        title4.SetFont(self.mainframe.font3)

        self.checkshowExperimenalData = wx.CheckBox(self, -1, "Show Exp. Data")
        self.checkshowExperimenalData.SetValue(False)
        self.checkshowFluoFrame = wx.CheckBox(self, -1, "Show Fluo. Det. Frame")
        self.checkshowFluoFrame.SetValue(False)
        self.checkExperimenalImage = wx.CheckBox(self, -1, "Show Exp. Image")
        self.checkExperimenalImage.SetValue(False)
        self.expimagetxtctrl = wx.TextCtrl(self, -1, "", size=(75, -1))
        self.expimagebrowsebtn = wx.Button(self, -1, "...", size=(50, -1))
        self.expimagebrowsebtn.Bind(wx.EVT_BUTTON, self.onSelectImageFile)

        self.pt_2thetachi = wx.RadioButton(self, -1, "2ThetaChi", style=wx.RB_GROUP)
        self.pt_XYCCD = wx.RadioButton(self, -1, "XYPixel")
        self.pt_gnomon = wx.RadioButton(self, -1, "Gnomon")
        self.pt_2thetachi.SetValue(True)

        # set tooltips--------------------------
        self.rbtop.SetToolTipString("Camera at 2theta=90 deg on top of sample")
        self.rbside.SetToolTipString("Camera at 2theta=90 deg on side of sample")
        self.rbsideneg.SetToolTipString("Camera at 90 deg on other side of sample")
        self.rbtransmission.SetToolTipString("Camera at 2theta=0 deg")
        self.rbbackreflection.SetToolTipString("Camera at 2theta=180 deg")

        self.checkshowExperimenalData.SetToolTipString("Plot markers for current experimental peak list")
        self.checkExperimenalImage.SetToolTipString("Display Laue pattern (2D image)")
        self.expimagetxtctrl.SetToolTipString("Full path for Laue pattern to be superimposed to simulated peaks")
        self.expimagebrowsebtn.SetToolTipString("Browse and select Laue Pattern image file")

        self.pt_2thetachi.SetToolTipString("Peaks Coordinates in scattering angles: 2theta and Chi")
        self.pt_XYCCD.SetToolTipString("Peaks Coordinates in detector frame pixels")

        # set widgets layout---------------------------
        gridSizer2 = wx.GridSizer(rows=13, cols=3, hgap=1, vgap=1)

        gridSizer2.Add(title2, 0, wx.ALIGN_CENTER | wx.ALIGN_CENTER)
        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.EXPAND)
        gridSizer2.Add(title25, 0, wx.EXPAND)

        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.EXPAND)
        gridSizer2.Add(txtdd, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.detdist, 0, wx.EXPAND)

        gridSizer2.Add(self.rbtop, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtdiam, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.detdiam, 0, wx.EXPAND)

        gridSizer2.Add(self.rbside, 1, wx.ALIGN_LEFT)
        gridSizer2.Add(txtxcen, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.xcen, 0, wx.EXPAND)

        gridSizer2.Add(self.rbsideneg, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtycen, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.ycen, 0, wx.EXPAND)

        gridSizer2.Add(self.rbtransmission, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtxbet, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.xbet, 0, wx.EXPAND)

        gridSizer2.Add(self.rbbackreflection, 0, wx.ALIGN_LEFT)
        gridSizer2.Add(txtxgam, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.xgam, 0, wx.EXPAND)

        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(txtpixelsize, 0, wx.ALIGN_RIGHT)
        gridSizer2.Add(self.ctrlpixelsize, 0, wx.EXPAND)

        gridSizer2.Add(wx.StaticLine(self, -1, size=(-1, 10), style=wx.LI_HORIZONTAL), 0, wx.EXPAND|wx.ALL, 5)
        gridSizer2.Add(wx.StaticLine(self, -1, size=(-1, 10), style=wx.LI_HORIZONTAL), 0, wx.EXPAND|wx.ALL, 5)
        gridSizer2.Add(wx.StaticLine(self, -1, size=(-1, 10), style=wx.LI_HORIZONTAL), 0, wx.EXPAND|wx.ALL, 5)

        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(title4, 0, wx.EXPAND)
        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)

        gridSizer2.Add(self.pt_2thetachi, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(self.pt_XYCCD, 0, wx.EXPAND)
        gridSizer2.Add(self.pt_gnomon, 0, wx.EXPAND)

        gridSizer2.Add(self.checkshowExperimenalData, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(self.checkshowFluoFrame, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(wx.StaticText(self, -1, ""), 0, wx.EXPAND)

        gridSizer2.Add(self.checkExperimenalImage, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        gridSizer2.Add(self.expimagetxtctrl, 0, wx.EXPAND)
        gridSizer2.Add(self.expimagebrowsebtn, 0, wx.EXPAND)

        spSizer = wx.BoxSizer(wx.VERTICAL)
        spSizer.Add(title1, 0, wx.ALIGN_CENTER)
        spSizer.AddSpacer(10)
        spSizer.Add(gridSizer, 0)
        spSizer.Add(gridSizer2, 0)
        spSizer.AddSpacer(5)

        self.SetSizer(spSizer)

    def onSelectImageFile(self, evt):
        """ open a File Dialog for an image to set self.expimagetxtctrl """
        self.GetfullpathFile(evt)
        self.expimagetxtctrl.SetValue(self.fullpathimagefile)

    def GetfullpathFile(self, _):
        """ open File Dialog and set self.fullpathimagefile"""
        myFileDialog = wx.FileDialog(self, "Choose an image file", style=wx.OPEN,
                                    wildcard="MAR or Roper images(*.mccd)|*.mccd|All files(*)|*")
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
    board of parametric
    Laue Simulation with tabs for ergonomic GUI
    """
    def __init__(self, parent, _id, title, initialParameters):
        wx.Frame.__init__(self, parent, _id, title)

        self.panel = wx.Panel(self)

        self.dirname = os.getcwd()

        self.parent = parent
        # self.dirname = LaueToolsframe.dirname
        self.initialParameters = initialParameters

        self.dict_Materials = initialParameters['dict_Materials']

        # self.dirname = '.'
        try:
            self.CCDLabel = self.parent.CCDLabel
        except AttributeError:
            self.CCDLabel = self.initialParameters["CCDLabel"]

        # current_param = self.initialParameters["CalibrationParameters"]
        self.pixelsize = self.initialParameters["pixelsize"]
        self.framedim = self.initialParameters["framedim"]

        self.font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)

        if initialParameters["ExperimentalData"] is not None:

            (self.data_2theta,
                self.data_chi,
                self.data_pixX,
                self.data_pixY,
                self.data_I,
            ) = initialParameters["ExperimentalData"]

        self.dict_grain_created = {}
        self.SelectGrains = {}

        self.CurrentGrain = ["Cu", "FaceCenteredCubic", "Identity",
                            "Identity", "Identity", "Identity", "Grain_0", "", ]
        self.modify = None
        self.list_of_Elem = None
        self.list_of_Extinc = None
        self.list_of_Strain_a = None
        self.list_of_Rot = None
        self.list_of_Vect = None
        self.list_of_Strain_c = None
        self.Bmatrix_current = None
        self.ParentGrainname = None
        self.emin, self.emax = None, None
        self.Det_distance, self.Det_diameter = None, None
        self.Xcen, self.Ycen, self.Xbet, self.Xgam = None, None, None, None
        self.calib = None
        self.simul_filename = None

        # widgets --------------------------------
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
        self.centerpanel.SetToolTipString("Apply parametric transforms (distribution of "
        "orientation and strain) on higlighted grain in basket")
        self.rightpanel.SetToolTipString("Simulation parameters board (CCD position, spots "
        "coordinates,Energy band pass...)")

        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.buttonsvertSizer, 1, wx.EXPAND)
        self.hbox.Add(self.nb0, 1, wx.EXPAND)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.hbox, 1, wx.EXPAND, 0)
        vbox.AddSpacer(10)
        linehoriz = wx.StaticLine(self.panel)
        vbox.Add(linehoriz, 0, wx.ALL | wx.EXPAND, 5)
        vbox.Add(self.hboxbottom, 0, wx.EXPAND, 0)

        self.panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

    def OnTabChange_nb0(self, event):
        #        print 'tab changed'
        selected_tab = self.nb0.GetSelection()
        print("selected tab:", selected_tab)
        print(self.nb0.GetPage(self.nb0.GetSelection()))
        print(self.nb0.GetPage(self.nb0.GetSelection()).GetName())
        event.Skip()  # patch for windows to update the tab display

    def create_leftpanel(self):

        titlegrain = wx.StaticText(self.panel, -1, "Grain Definition")
        titlegrain.SetFont(self.font3)

        txt1 = wx.StaticText(self.panel, -1, "Material")
        txt2 = wx.StaticText(self.panel, -1, "Extinctions")
        txt3 = wx.StaticText(self.panel, -1, "Transform_a")
        txt4 = wx.StaticText(self.panel, -1, "Rot. Matrix")
        txt5 = wx.StaticText(self.panel, -1, "B matrix")
        txt6 = wx.StaticText(self.panel, -1, "Transform_c")

        self.RefreshCombosChoices()

        self.comboElem = wx.ComboBox(self.panel, -1, "Cu",
                                choices=self.list_of_Elem, style=wx.CB_READONLY)
        self.comboExtinc = wx.ComboBox(self.panel, -1, "FaceCenteredCubic",
                                choices=self.list_of_Extinc, style=wx.CB_READONLY)
        self.comboStrain_a = wx.ComboBox(self.panel, -1, "Identity",
                                choices=self.list_of_Strain_a)
        self.comboRot = wx.ComboBox(self.panel, -1, "Identity", choices=self.list_of_Rot)
        self.comboVect = wx.ComboBox(self.panel, -1, "Identity", choices=self.list_of_Vect)
        self.comboStrain_c = wx.ComboBox(self.panel, -1, "Identity", choices=self.list_of_Strain_c)

        buttonrefresh = wx.Button(self.panel, -1, "Refresh choices")
        buttonrefresh.Bind(wx.EVT_BUTTON, self.updatecombosmenus)

        loadMaterialsbtn = wx.Button(self.panel, -1, "Reload Materials")
        loadMaterialsbtn.Bind(wx.EVT_BUTTON, self.onLoadMaterials)

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

        deletebutton = wx.Button(self.panel, -1, "Delete")
        deleteallbutton = wx.Button(self.panel, -1, "DeleteAll")
        deletebutton.Bind(wx.EVT_BUTTON, self.DeleteGrain)
        deleteallbutton.Bind(wx.EVT_BUTTON, self.DeleteAllGrain)

        # tooltips --------------------------------
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
        tipvect += "Initial set of reciprocal unit cell basis vector in LaueTools lab. frame"

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

        buttonshoriz2Sizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonshoriz2Sizer.Add(addgrainbtn, 1)
        buttonshoriz2Sizer.Add(buttonrefresh, 1)
        buttonshoriz2Sizer.Add(loadMaterialsbtn, 1)

        horizbtnsizers = wx.BoxSizer(wx.HORIZONTAL)
        horizbtnsizers.Add(deletebutton, 0)
        horizbtnsizers.Add(deleteallbutton, 0)

        self.buttonsvertSizer = wx.BoxSizer(wx.VERTICAL)
        self.buttonsvertSizer.Add(titlegrain, 0, wx.ALIGN_CENTER)
        self.buttonsvertSizer.Add(gridSizer)
        self.buttonsvertSizer.Add(buttonshoriz2Sizer, 0, wx.ALIGN_CENTER)
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

        self.rb1 = wx.RadioButton(self.panel, -1, "Manual", style=wx.RB_GROUP)
        self.rb2 = wx.RadioButton(self.panel, -1, "Auto. Indexed")
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
        self.rb2.SetToolTipString("Automatic incrementation of simulation peaks list Filename  .sim")
        self.corfileBox.SetToolTipString("Create a fake experimental peaks list (without miller indices)")

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
        String_Info = " ***** USER GUIDE ******\n\n"
        String_Info += "1- Select crystallographic structure of by Element or structure\n"
        String_Info += "          Orientation by some specific orientation of a*,b*,c* or rotation matrix\n"
        String_Info += "          Angular strain matrix\n"
        String_Info += "2- Add grains as many as you want in the list\n"
        String_Info += "          Delete unwanted grains by selecting them and clicking in Delete button\n"
        String_Info += "3- OPTIONNALY Select a set of orientation or strain transfrom from a selected parent grain in the list\n"
        String_Info += "          Axis rotation and Axes traction can not be combined\n"
        String_Info += "          One transform set for one selected grain\n"
        String_Info += "          t is the varying parameter given start, end and number of steps values,\n"
        String_Info += "               can be put in any place in other field with maths expression\n"
        String_Info += "               example: a[1+cos(t/2.)*exp(-t*.1)]\n"
        String_Info += "               start, end may be floats\n"
        String_Info += "          Frame chosen to express coordinates can be selectiong by choosing the letter a, s or c\n\n"
        String_Info += "          ROTATION:\n"
        String_Info += "          Choose the frame, axis-vector coordinnates and angle in degree\n"
        String_Info += "          STRAIN:\n"
        String_Info += "          three axes of traction can be combined,\n"
        String_Info += "          with frame, vector direction, and amplitude in real space\n"
        String_Info += "                example: factor of 1.1 means 10\% expansion in real space along the chosen direction\n"
        String_Info += "4- Select plot/calibration/file parameter and click on simulate button\n"
        return String_Info

    def OnTextChanged(self, evt):
        self.modify = True
        evt.Skip()

    def OnKeyDown(self, event):
        event.Skip()

    def showCurrentDir(self, _):
        dlg = wx.MessageDialog(self, "Current directory :%s" % self.dirname,
                                                "Current Directory", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def onLoadMaterials(self, _):
        self.parent.OnLoadMaterials(1)
        self.updatecombosmenus(1)

    def updatecombosmenus(self, _):

        self.RefreshCombosChoices()

        self.comboElem.Clear()
        self.comboElem.AppendItems(self.list_of_Elem)
        self.comboExtinc.Clear()
        self.comboExtinc.AppendItems(self.list_of_Extinc)
        self.comboStrain_a.Clear()
        self.comboStrain_a.AppendItems(self.list_of_Strain_a)
        self.comboRot.Clear()
        self.comboRot.AppendItems(self.list_of_Rot)
        self.comboVect.Clear()
        self.comboVect.AppendItems(self.list_of_Vect)
        self.comboStrain_c.Clear()
        self.comboStrain_c.AppendItems(self.list_of_Strain_c)

    def RefreshCombosChoices(self):
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

        # List_Elem_name = list(DictLT.dict_Materials.keys())
        List_Elem_name = list(self.parent.dict_Materials.keys())
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
            # print('k,val', k, val)
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
                print("You must select by mouse a least one grain!!")
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

        num_items = self.LC.GetItemCount()

        if num_items < 1:
            wx.MessageBox("You must select a grain before calculating some geometrical transforms!", "ERROR")

        grainindex = self.LC.GetItemText(self.LC.GetFocusedItem())
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
        print('wholestring', wholestring)
        # header = '2theta  chi   x   y   I'
        # headerarray = np.array(["2theta", " chi", "   x", "   y", "   I"], dtype="|S11")

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
            print("list_of_lines", list_of_lines)
            joineddata = " ".join(list_of_lines)
            # print "array(joineddata.split())",array(joineddata.split())
            array_grain = np.reshape(
                np.array(np.array(joineddata.split()), dtype=np.float32),
                (nbpeaks_per_grain[gi], 9))
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
            (twothetachi[:, 0], twothetachi[:, 1], xy[:, 0], xy[:, 1], intensity))[0]

        # sort data according to modelled intensity
        # print "Toedit",Toedit[:3]
        Toedit_sorted = Toedit[np.argsort(Toedit[:, 4])[::-1]]
        # print "ghg",array(Toedit_sorted, dtype = '|S11')[:10] # number of digits in file encoded by number of chars in string

        print("Toedit_sorted", Toedit_sorted)
        filename = os.path.join(self.dirname, file_name_fake)

        corfileobj = open(filename, "w")

        footer = "# Cor file generated by LaueTools Polygrains Simulation Software"
        footer += "\n# File created at %s by by ParametricLaueSimulator.py" % (time.asctime())
        footer += "\n# Calibration parameters"
        for par, value in zip(["dd", "xcen", "ycen", "xbet", "xgam"], self.calib):
            footer += "\n# %s     :   %s" % (par, value)
        footer += "\n# pixelsize    :   %s" % self.pixelsize
        footer += "\n# ypixelsize    :   %s" % self.pixelsize
        footer += "\n# CCDLabel    :   %s" % self.CCDLabel

        header2 = "2theta   chi    x    y  I"
        np.savetxt(corfileobj, Toedit_sorted, fmt='%.8f', header=header2, footer=footer, comments="")

        corfileobj.close()

    def OnSave(self, _, data_res):

        textfile = open(os.path.join(self.dirname, self.simul_filename), "w")

        textfile.write(Edit_String_SimulData(data_res))

        textfile.close()

        fullname = os.path.join(self.dirname, self.simul_filename)
        wx.MessageBox("File saved in %s" % fullname, "INFO")

    def opendir(self, _):
        dlg = wx.DirDialog(
                            self,
                            "Choose a directory:self.control.SetValue(str(self.indexed_spots).replace('],',']\n'))",
                            style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
        if dlg.ShowModal() == wx.ID_OK:
            self.dirname = dlg.GetPath()

            print(self.dirname)

        dlg.Destroy()

    def OnQuit(self, _):
        self.Close()

    def OnSimulate(self, evt):
        """
        in parametric transformation simulation parametric_Grain_Dialog3
        """
        # select all created grain and build dict self.SelectGrains
        self.Select_AllGrain(evt)
        # list of parameters for parent and child grains
        list_param = MGS.Construct_GrainsParameters_parametric(self.SelectGrains)

        # print('list_param',list_param)

        if not list_param:
            dlg = wx.MessageDialog(self,
                                    "You must create and select at least one grain!",
                                    "Empty Grains list",
                                    wx.OK | wx.ICON_ERROR)
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
        elif self.rightpanel.rbbackreflection.GetValue():
            cameraposition = "X<0"
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
            dlg = wx.MessageDialog(self,
                                        "Detector parameters must be float with dot separator",
                                        "Bad Input Parameters",
                                        wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return True

        showExperimenalData = self.rightpanel.checkshowExperimenalData.GetValue()
        showExperimentalImage = self.rightpanel.checkExperimenalImage.GetValue()
        showFluoFrame = self.rightpanel.checkshowFluoFrame.GetValue()

        # show markers experimental list of peaks
        if showExperimenalData:
            if self.initialParameters["ExperimentalData"] is None:
                dlg = wx.MessageDialog(self,
                    "You must load experimental data(File/Open Menu) before or uncheck Show Exp. Data box", "Experimental Data Missing!", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return True

        # show 2D pixel intensity from image file
        ImageArray = None
        if showExperimentalImage:

            fullpathimagename = str(self.rightpanel.expimagetxtctrl.GetValue())
            if not os.path.isfile(fullpathimagename):
                dlg = wx.MessageDialog(self, "Image file : %s\n\ndoes not exist!!" % fullpathimagename,
                    "FILE ERROR", wx.OK | wx.ICON_ERROR, )
                dlg.ShowModal()
                dlg.Destroy()
                return

            ImageArray = IOimage.readCCDimage(fullpathimagename, self.CCDLabel, dirname=None)[0]

            self.rightpanel.pt_XYCCD.SetValue(True)

        if self.rightpanel.pt_2thetachi.GetValue():
            plottype = "2thetachi"
        elif self.rightpanel.pt_XYCCD.GetValue():
            plottype = "XYmar"
        elif self.rightpanel.pt_gnomon.GetValue():
            plottype = "gnomon"
        else:
            raise ValueError('plottype "%s" in OnSimulate() is unknown...!'%plottype)

        if showFluoFrame:
            plottype = "XYmar_fluo"

        self.textprocess.SetLabel("Processing Laue Simulation")
        self.gauge.SetRange(len(list_param) * 10000)

        # simulation in class parametric_Grain_Dialog3

        self.calib = [self.Det_distance, self.Xcen, self.Ycen, self.Xbet, self.Xgam]
        data_res = MGS.dosimulation_parametric(list_param,
                                                emax=self.emax,
                                                emin=self.emin,
                                                detectordistance=self.Det_distance,
                                                detectordiameter=self.Det_diameter,
                                                posCEN=(self.Xcen, self.Ycen),
                                                cameraAngles=(self.Xbet, self.Xgam),
                                                gauge=self.gauge,
                                                kf_direction=cameraposition,
                                                Transform_params=self.dict_transform,
                                                SelectGrains=self.SelectGrains,
                                                pixelsize=self.pixelsize,
                                                dictmaterials=self.parent.dict_Materials)

        # (list_twicetheta, list_chi,
        # list_energy, list_Miller,
        # list_posX, list_posY, ListName, nb_g_t, calib, total_nb_grains) = data_res
        (list_twicetheta, list_chi,
        list_energy, list_Miller,
        list_posX, list_posY, _, nb_g_t, _, total_nb_grains) = data_res

        # compute gnomonic coordinates
        list_xgnomon, list_ygnomon = [], []
        nblists = len(list_twicetheta)
        for k in range(nblists):
            xgs, ygs = IMM.ComputeGnomon_2((np.array(list_twicetheta[k]), np.array(list_chi[k])))
            list_xgnomon.append(xgs.tolist())
            list_ygnomon.append(ygs.tolist())

        print("len(list_posX)", len(list_posX))
        print("len(list_posY)", len(list_posY))
        print("len(list_posX[0])", len(list_posX[0]))

        # for subgrainposX in list_posX:
        #     print('len(subgrainposX)',len(subgrainposX))

        #-----  slip system handling ------------------

        GrainParent_list = []
        TransformType_list = []
        Nbspots_list = []
        SpotIndexAccum_list = [] # list of last spot index belonging to the subgrain
        subgrainindex = 0
        accumNb = -1
        for par in nb_g_t:
            parGrainIndex, nbtransfroms, transform_type = par
            for _ in range(nbtransfroms):
                GrainParent_list.append(parGrainIndex)
                TransformType_list.append(transform_type)
                nbLaueSpots = len(list_posX[subgrainindex])
                accumNb += nbLaueSpots
                Nbspots_list.append(nbLaueSpots)
                SpotIndexAccum_list.append(accumNb)
                subgrainindex += 1

        print('SpotIndexAccum_list', SpotIndexAccum_list)
        print('GrainParent_list', GrainParent_list)
        print('TransformType_list', TransformType_list)

        # ------  setting StreakingData   for grains distribution or slips system or single crystals
        # StreakingData = data_res, SpotIndexAccum_list, GrainParent_list, TransformType_list, slipsystemsfcc
        # -------------------------------------------------------
        print("\n\ndata_res[7]  nb_g_t", nb_g_t)
        StreakingData = None
        slipsystemsfcc = None
        for elem in nb_g_t:
            _, _, transformtype = elem
            print('transformtype', transformtype)
            print('elem transform', elem)
            if 'slip' in transformtype:
                print("there is a slipsystem simulation")
                slipsystemsfcc = DictLT.SLIPSYSTEMS_FCC
                plottype += 'XYmar_SlipsSystem'

        StreakingData = data_res, SpotIndexAccum_list, GrainParent_list, TransformType_list, slipsystemsfcc

        print('StreakingData[1]', StreakingData[1])

        # ------------------------------------------------
        # -------   plot results -------------------------
        #-------------------------------------------------

        # experimental data--------------------------------------
        if showExperimenalData:
            experimentaldata_2thetachi = (self.data_2theta, self.data_chi, self.data_I)
            experimentaldata_XYMAR = self.data_pixX, self.data_pixY, self.data_I
            xgexp, ygexp = IMM.ComputeGnomon_2((self.data_2theta, self.data_chi))
            experimentaldata_gnomon = (xgexp, ygexp, self.data_I)
        else:
            experimentaldata_2thetachi = None
            experimentaldata_XYMAR = None
            experimentaldata_gnomon = None

        # theoretical data--------------------------------------
        print('plottype in LaueSimulatorGUI : %s  \n\n'%plottype)
        if plottype == "2thetachi":
            totalnbspots = 0
            for slist in list_twicetheta:
                totalnbspots += len(slist)
            if totalnbspots == 0:
                wx.MessageBox('No Laue spots on the detector defined by the current position, distance, diameter, ... . Change the simulation parameters!', 'Info')
            simulframe = SimulationPlotFrame(self, -1, "LAUE Pattern simulation visualisation frame",
                            data=(list_twicetheta, list_chi, list_energy, list_Miller,
                            total_nb_grains, plottype, experimentaldata_2thetachi,),
                            StreakingData=StreakingData,
                            list_grains_transforms=nb_g_t,
                            CCDLabel=self.CCDLabel)

        elif "XYmar" in plottype: # XYPixel
            totalnbspots = 0
            for slist in list_posX:
                totalnbspots += len(slist)
            if totalnbspots == 0:
                wx.MessageBox('No Laue spots on the detector defined by the current position, distance, diameter, ... . Change the simulation parameters!', 'Info')
            simulframe = SimulationPlotFrame(self, -1, "LAUE Pattern simulation visualisation frame",
                        data=(list_posX, list_posY, list_energy, list_Miller,
                        total_nb_grains, plottype, experimentaldata_XYMAR,),
                        ImageArray=ImageArray,
                        StreakingData=StreakingData,
                        list_grains_transforms=nb_g_t,
                        CCDLabel=self.CCDLabel)

        elif plottype == "gnomon":
            totalnbspots = 0
            for slist in list_xgnomon:
                totalnbspots += len(slist)
            if totalnbspots == 0:
                wx.MessageBox('No Laue spots on the detector defined by the current position, distance, diameter, ... . Change the simulation parameters!', 'Info')
            simulframe = SimulationPlotFrame(self, -1, "LAUE Pattern simulation visualisation frame",
                            data=(list_xgnomon, list_ygnomon, list_energy, list_Miller,
                            total_nb_grains, plottype, experimentaldata_gnomon,),
                            StreakingData=StreakingData,
                            list_grains_transforms=nb_g_t,
                            CCDLabel=self.CCDLabel)

        simulframe.Show(True)

        self.textprocess.SetLabel("Laue Simulation Completed")

        if self.savefileBox.GetValue():
            if self.rb1.GetValue():
                file_name = self.textcontrolfilemanual.GetValue() + ".sim"
            else:
                file_name = (self.textcontrolfileauto.GetValue()
                            + str(self.initialParameters["indexsimulation"])
                            + ".sim")
                self.initialParameters["indexsimulation"] += 1
                print("Next index is %s" % self.initialParameters["indexsimulation"])

            self.simul_filename = file_name
            print("Simulation file saved in %s " % self.simul_filename)
            self.OnSave(evt, data_res)

        if self.corfileBox.GetValue():

            file_name_fake = self.corcontrolfake.GetValue() + ".cor"

            print("Fake data in file %s " % file_name_fake)
            self.OnWriteCorFile(file_name_fake, data_res, len(list_param))
#-----    End of GUI class------------------------


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
    if isinstance(nb, int):  # multigrains simulations without transformations
        nb_grains = data[7]
        TWT, CHI, ENE, MIL, XX, YY = data[:6]
        NAME = data[6]
        calib = data[8]

        for index_grain in range(nb_grains):
            nb_of_simulspots = len(TWT[index_grain])
            startgrain = "#G %d\t%s\t%d\n" % (index_grain, NAME[index_grain], nb_of_simulspots)

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
                    YY[index_grain][data_index])
                lines += linedata
        lines += "#calibration parameters\n"
        for param in calib:
            lines += "# %s\n" % param
        # print "in edit",lines
        #        self.control.SetValue(lines)
        return lines

    if isinstance(nb, list):  # nb= list of [grain index, nb of transforms]
        print("nb in Edit_String_SimulData", nb)
        gen_i = 0
        TWT, CHI, ENE, MIL, XX, YY = data[:6]
        NAME = data[6]
        calib = data[8]
        for grain_ind in range(len(nb)):  # loop over parent grains
            for tt in range(nb[grain_ind][1]):
                nb_of_simulspots = len(TWT[gen_i])
                startgrain = "#G %d\t%s\t%d\t%d\n" % (grain_ind,
                                                        NAME[grain_ind],
                                                        tt,
                                                        nb_of_simulspots)

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
                                                                YY[gen_i][data_index])
                    lines += linedata
                gen_i += 1

        lines += "#calibration parameters\n"
        for param in calib:
            lines += "# %s\n" % param
        # print "in edit",lines
        #        self.control.SetValue(lines)
        return lines


def start():
    title = "Multiple Grains Laue Pattern simulator"

    initialParameters = {}
    initialParameters["CalibrationParameters"] = [70, 1024, 1024, -1.2, 0.89]
    initialParameters["prefixfilenamesimul"] = "Ge_blanc_"
    initialParameters["indexsimulation"] = 0
    initialParameters["ExperimentalData"] = None
    initialParameters["pixelsize"] = 165 / 2048.0
    initialParameters["framedim"] = (2048, 2048)
    initialParameters["CCDLabel"] = "MARCCD165"
    initialParameters["dict_Materials"] = DictLT.dict_Materials

    GUIApp = wx.App()
    GUIframe = parametric_Grain_Dialog3(None, -1, title, initialParameters)
    GUIframe.Show()

    GUIApp.MainLoop()


if __name__ == "__main__":
    start()
