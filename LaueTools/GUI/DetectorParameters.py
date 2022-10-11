from __future__ import division
import time
import os
import copy
import sys

import wx

if sys.version_info.major == 3:
    from .. import dict_LaueTools as DictLT
else:
    import dict_LaueTools as DictLT


class DetectorParameters(wx.Dialog):
    """
    Class GUI to set the detector parameters

    parent needs to have following attributes
    defaultParam (ie "CCDParam")
    pixelsize
    framedim
    detectordiameter
    kf_direction
    """
    def __init__(self, parent, _id, title, Parameters_dict):

        wx.Dialog.__init__(self, parent, -1, title, size=(600, 600))

        self.parent = parent

        self.panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(590, 650), pos=(5, 5))

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.paramdetector = ["Distance",
                            "xcen",
                            "ycen",
                            "betangle",
                            "gammaangle",
                            "pixelsize",
                            "dim1",
                            "dim2",
                            "detectordiameter",
                            "kf_direction"]
        self.units = ["mm",
                    "pixel",
                    "pixel",
                    "deg",
                    "deg",
                    "mm",
                    "pixel",
                    "pixel",
                    "mm",
                    "ascii"]

        self.initialParameters_dict = copy.copy(Parameters_dict)

        self.Parameters_dict = Parameters_dict

        self.params_values_list = self.Parameters_dict["CCDParam"]

        if len(self.Parameters_dict["CCDParam"]) == 5:

            self.params_values_list = self.Parameters_dict["CCDParam"] + [self.Parameters_dict["pixelsize"],
                                                                self.Parameters_dict["framedim"][0],
                                                                self.Parameters_dict["framedim"][1],
                                                                self.Parameters_dict["detectordiameter"],
                                                                self.Parameters_dict["kf_direction"]]

        print("self.params_values_list", self.params_values_list)
        print("self.paramdetector", self.paramdetector)

        self.currentvalues = copy.copy(self.params_values_list)

        self.newparam = []

        self.controltext = []

        a1 = wx.StaticText(self.panel, -1, "parameter", (15, 10))
        a2 = wx.StaticText(self.panel, -1, "current value", (150, 10))
        a3 = wx.StaticText(self.panel, -1, "initial value", (340, 10))
        a4 = wx.StaticText(self.panel, -1, "unit", (540, 10))

        for text in [a1, a2, a3, a4]:
            text.SetFont(font3)
        for kk, paramVal in enumerate(self.params_values_list):
            print("kk,paramVal", kk, paramVal)
            wx.StaticText(self.panel, -1, self.paramdetector[kk], (15, 45 + 30 * kk))
            self.controltext.append(
                wx.TextCtrl(
                    self.panel, -1, str(paramVal), (150, 40 + 30 * kk), (150, -1)))
            wx.StaticText(self.panel, -1, str(paramVal), (340, 45 + 30 * kk))
            wx.StaticText(self.panel, -1, self.units[kk], (540, 45 + 30 * kk))

        sizey = 100
        posbuttons = sizey + 110

        com = wx.StaticText(self.panel, -1, "Comments", (5, 340))
        com.SetFont(font3)
        self.comments = wx.TextCtrl(
            self.panel, style=wx.TE_MULTILINE, size=(590, sizey), pos=(5, 360))

        loadbtn = wx.Button(self.panel, 1, "Load", (160, 260 + posbuttons), (100, 40))
        loadbtn.Bind(wx.EVT_BUTTON, self.OnLoadCalib)

        wx.Button(self.panel, 2, "Save", (280, 260 + posbuttons), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnSaveCalib, id=2)

        wx.Button(self.panel, 3, "Accept", (40, 260 + posbuttons), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnAcceptCalib, id=3)

        wx.Button(self.panel, 4, "Cancel", (400, 260 + posbuttons), (100, 40))
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id=4)

        self.keepon = False

        # tooltips
        loadbtn.SetToolTipString("Load Detector Parameters")

    def OnLoadCalib(self, _):
        """
        Load calibration detector geometry (in DetectorParameters)

        only the first 8 parameters are set

        .. warning:: diameter and kf direction are not set!
        """
        wcd = "Calibration file(*.det)|*.det|All files(*)|*"
        _dir = os.getcwd()
        if self.parent is not None:
            if self.parent.dirnamepklist is not None:
                _dir = self.parent.dirnamepklist
        open_dlg = wx.FileDialog(self,
                                message="Choose a file",
                                defaultDir=_dir,
                                defaultFile="",
                                wildcard=wcd,
                                style=wx.OPEN | wx.CHANGE_DIR)
        if open_dlg.ShowModal() == wx.ID_OK:
            path = open_dlg.GetPath()
            try:
                _file = open(path, "r")
                text = _file.readlines()
                _file.close()

                # first line contains parameters
                parameters = [float(elem) for elem in str(text[0]).split(",")]
                self.currentvalues[:8] = parameters
                # others are comments
                comments = text[1:]

                for kk, controller in enumerate(self.controltext):
                    controller.SetValue(str(self.currentvalues[kk]))

                allcomments = ""
                for line in comments:
                    allcomments += line

                self.comments.SetValue(str(allcomments))

            except IOError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

            except UnicodeDecodeError as error:
                dlg = wx.MessageDialog(self, "Error opening file\n" + str(error))
                dlg.ShowModal()

        open_dlg.Destroy()
        #os.chdir(_dir)

    def OnSaveCalib(self, _):
        """
        in DetectorParameters
        """
        # update values
        i = 0
        for controller in self.controltext:
            if i < 9:
                self.newparam.append(float(controller.GetValue()))
            else:
                self.newparam.append(str(controller.GetValue()))
            i = i + 1

        txt = ""
        for par in self.newparam:
            txt += str(par) + "\n"

        dlg = wx.TextEntryDialog(self,
                    "Enter Calibration File name : \n Current Calibration parameters are: \n" + txt,
                        "Saving Calibration Parameters Entry")
        dlg.SetValue("*.det")
        if dlg.ShowModal() == wx.ID_OK:
            filenameCalib = str(dlg.GetValue())

            dd, xcen, ycen, xbet, xgam, pixsize, dim1, dim2 = self.newparam[:8]

            comments = self.comments.GetValue()

            _file = open(filenameCalib, "w")
            text = "%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f,%.17f\n" % (dd,
                                                                        xcen,
                                                                        ycen,
                                                                        xbet,
                                                                        xgam,
                                                                        pixsize,
                                                                        dim1,
                                                                        dim2)
            text += "Sample-Detector distance(IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2 \n"
            text += "Saved at %s with LaueToolsGUI.py\n" % time.asctime()
            text += comments
            _file.write(text)
            _file.close()

        dlg.Destroy()

        fullname = os.path.join(os.getcwd(), filenameCalib)
        wx.MessageBox("Calibration parameters saved in %s" % fullname, "INFO")
        self.newparam = []

    #         self.OnCancel(event)

    def OnAcceptCalib(self, _):
        """
        in DetectorParameters
        """
        if not self.getcurrentParams():
            return

        print("self.newparam in OnAcceptCalib()", self.newparam)

        Parameter = {}
        Parameter["CCDParam"] = self.newparam[:5]
        Parameter["pixelsize"] = self.newparam[5]
        Parameter["framedim"] = self.newparam[6:8]
        Parameter["detectordiameter"] = self.newparam[8]
        Parameter["kf_direction"] = self.newparam[9]

        print("param to set", Parameter["CCDParam"])
        # new parameters can be called from outside parent frame
        if self.parent is not None:

            try:
                print("Parameter['kf_direction']", Parameter["kf_direction"])
                print("detectordiameter", Parameter["detectordiameter"])

                self.parent.defaultParam = Parameter["CCDParam"]
                self.parent.pixelsize = Parameter["pixelsize"]
                self.parent.framedim = Parameter["framedim"]
                self.parent.detectordiameter = Parameter["detectordiameter"]
                self.parent.kf_direction = Parameter["kf_direction"]

                print("self.parent.defaultParam", self.parent.defaultParam)
            except AttributeError:
                print("you must define an attribute 'Parameters_dict'")
                print("of the calling parent object to collect the new parameters")

        self.Close()

    def getcurrentParams(self):
        """
        get current values from fields and return True if all is correct
        """
        self.newparam = []
        for ii, controller in enumerate(self.controltext):
            if ii < 9:
                self.newparam.append(float(controller.GetValue()))
            else:
                val_kf_direction = str(controller.GetValue())
                if val_kf_direction not in DictLT.DICT_LAUE_GEOMETRIES:
                    wx.MessageBox("Value of kf_direction (Laue Geometry) is unknown.\nMust be in %s"
                                            % str(list(DictLT.DICT_LAUE_GEOMETRIES.keys())),
                                            "Error")
                    return False
                else:
                    self.newparam.append(val_kf_direction)
        return True

    def OnCancel(self, event):
        print("Detector Parameters are unchanged, Still: ")
        print((self.Parameters_dict["CCDLabel"],
                self.Parameters_dict["CCDParam"],
                self.Parameters_dict["pixelsize"],
                self.Parameters_dict["framedim"],
                self.Parameters_dict["detectordiameter"],
                self.Parameters_dict["kf_direction"]))

        self.Close()
        event.Skip()


def autoDetectDetectorType(autoDetectDetectorType):
    CCDlabel = None
    if autoDetectDetectorType in (".tif", "tif"):
        CCDlabel = "sCMOS"

    return CCDlabel


if __name__ == "__main__":

    Parameters_dict = {}
    Parameters_dict["CCDLabel"] = "MARCCD165"
    Parameters_dict["CCDParam"] = [100, 1024, 2048, 0.01, -0.2]
    Parameters_dict["pixelsize"] = 165.0 / 2048
    Parameters_dict["framedim"] = (2048, 8956)
    Parameters_dict["detectordiameter"] = 165.0
    Parameters_dict["kf_direction"] = "Z>0"

    DetectorParamGUIApp = wx.App()
    DetectorParamGUIFrame = DetectorParameters(None, -1, "Detector Calibration Board",
                                                                                    Parameters_dict)

    DetectorParamGUIFrame.Show()

    DetectorParamGUIApp.MainLoop()
