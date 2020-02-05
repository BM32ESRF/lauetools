"""
GUI Module of LaueTools package for choosing Detector File Parameters 
"""

import sys

import wx

if sys.version_info.major == 3:
    from . import dict_LaueTools as DictLT
    from . import generaltools as GT
else:
    import dict_LaueTools as DictLT
    import generaltools as GT


# --- ---------------  Binary image File parameters
class CCDFileParameters(wx.Dialog):
    """
    Class to set image file parameters
    """
    def __init__(self, parent, _id, title, CCDlabel):
        """
        initialize board window
        """
        wx.Dialog.__init__(self, parent, _id, title, size=(660, 440), style=wx.RESIZE_BORDER)
        self.parent = parent
        #print("self.parent", self.parent)

        txt = wx.StaticText(self, -1, "Choose Readout parameters of CCD image file", (50, 10))
        font = wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        txt.SetFont(font)
        txt.SetForegroundColour((255, 0, 0))

        self.panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER, size=(650, 440), pos=(5, 40))

        dict_CCD = DictLT.dict_CCD
        self.allCCD_names = list(dict_CCD.keys())
        self.allCCD_names.sort()

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.paramdetector = ["CCD label",
                            "header offset",
                            "dataformat",
                            "fliprot",
                            "saturation Value",
                            "file extension"]
        self.units = ["ascii",
                        "byte",
                        "python format",
                        "operator tag",
                        "integer",
                        "ascii"]

        self.CCDLabel = CCDlabel
        self.readCCDparams()

        self.newparam = []

        self.controltext = []

        posy = 15

        a0 = wx.StaticText(self.panel, -1, "CCD Image File type", (15, posy))
        a0.SetFont(font3)

        self.allCCD_names = GT.put_on_top_list(("MARCCD165",
                                            "sCMOS",
                                            "----------",
                                            "PRINCETON",
                                            "VHR_Mar13",
                                            "----------"),
                                            self.allCCD_names,
                                            forceinsertion=True)

        self.comboCCD = wx.ComboBox(self.panel,
                                    -1,
                                    self.CCDLabel,
                                    (320, posy - 5),
                                    size=(180, -1),
                                    choices=self.allCCD_names,
                                    style=wx.CB_READONLY)

        self.comboCCD.Bind(wx.EVT_COMBOBOX, self.EnterComboCCD)

        a1 = wx.StaticText(self.panel, -1, "parameter", (15, posy + 45))
        a2 = wx.StaticText(self.panel, -1, "current value", (150, posy + 45))
        a3 = wx.StaticText(self.panel, -1, "initial value", (340, posy + 45))
        a4 = wx.StaticText(self.panel, -1, "unit", (540, posy + 45))

        for text in [a1, a2, a3, a4]:
            text.SetFont(font3)

        com = wx.StaticText(self.panel, -1, "Comments", (5, posy + 245))
        com.SetFont(font3)

        self.posyvalues = 80

        for kk, param in enumerate(self.paramdetector):
            wx.StaticText(self.panel, -1, param, (15, self.posyvalues + 5 + 30 * kk))
            wx.StaticText(self.panel,
                            -1,
                            str(self.value[kk]),
                            (340, self.posyvalues + 5 + 30 * kk))
            wx.StaticText(self.panel, -1, self.units[kk], (540, self.posyvalues + 5 + 30 * kk))

        self.comments = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE, size=(580, 50), pos=(10, 280))
        self.comments.SetValue(DictLT.dict_CCD[self.CCDLabel][-1])

        self.DisplayValues(1)

        btnaccept = wx.Button(self.panel, 3, "Accept", (40, 340), (300, 40))
        self.Bind(wx.EVT_BUTTON, self.OnAccept, id=3)
        btnaccept.SetDefault()

        wx.Button(self.panel, 4, "Cancel", (400, 340), (150, 40))
        self.Bind(wx.EVT_BUTTON, self.OnCancel, id=4)

        self.keepon = False

        # tooltip
        btnaccept.SetToolTipString("Accept")

    #        self.Show(True)
    #        self.Centre()

    def readCCDparams(self):
        CCDparams = DictLT.dict_CCD[str(self.CCDLabel)]

        # TODO add framedim and pixelsize as CCD parameters better than as Detector parameters
        (self.parent.framedim,
            self.parent.pixelsize,
            self.saturationvalue,
            self.fliprot,
            self.headeroffset,
            self.dataformat,
            self.commentstxt,
            self.file_extension) = CCDparams

        self.value = (self.CCDLabel,
                    self.headeroffset,
                    self.dataformat,
                    self.fliprot,
                    self.saturationvalue,
                    self.file_extension)
        #print("self.value", self.value)

    def DisplayValues(self, _):
        """
        display and set values of parameters
        """
        for kk, _ in enumerate(self.paramdetector):
            self.controltext.append(
                wx.TextCtrl(self.panel,
                                -1,
                                str(self.value[kk]),
                                (150, self.posyvalues + 30 * kk),
                                (150, -1)))

    def DeleteValues(self, _):
        """
        set parameters to zero or None 
        """
        for kk, _ in enumerate(self.paramdetector):
            self.controltext.append(
                wx.TextCtrl(self.panel,
                            -1,
                            str(None),
                            (150, self.posyvalues + 30 * kk),
                            (150, -1)))

    def EnterComboCCD(self, event):
        item = event.GetSelection()
        self.CCDLabel = self.allCCD_names[item]
        print("item", item)
        print("CCDLabel", self.CCDLabel)

        if self.CCDLabel.startswith("--"):
            self.DeleteValues(event)
            self.comments.SetValue("")
            return

        self.readCCDparams()

        #print("self.value", self.value)
        self.comments.SetValue(self.commentstxt)
        self.DisplayValues(event)
        event.Skip()

    def OnAccept(self, _):
        if not self.CCDLabel.startswith("--"):
            # file parameters
            self.parent.headeroffset = self.headeroffset
            self.parent.dataformat = self.dataformat
            self.parent.fliprot = self.fliprot
            self.parent.saturationvalue = self.saturationvalue
            self.parent.CCDLabel = self.CCDLabel
            self.parent.file_extension = self.file_extension
            # geometrical parameters
            self.parent.framedim, self.parent.pixelsize = DictLT.dict_CCD[str(self.CCDLabel)][0:2]

            #print("self.parent", self.parent)
            print("self.parent.framedim", self.parent.framedim)

            self.parent.detectordiameter = (max(self.parent.framedim) * self.parent.pixelsize)
            self.Close()
        else:
            wx.MessageBox("Please select on type of CCD camera or Cancel", "Error")

    def OnCancel(self, _):
        self.Close()
