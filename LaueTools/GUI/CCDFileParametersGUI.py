"""
GUI Module of LaueTools package for choosing Detector File Parameters
"""
import sys

import wx

if sys.version_info.major == 3:
    from .. import dict_LaueTools as DictLT
    from .. import generaltools as GT
else:
    import dict_LaueTools as DictLT
    import generaltools as GT

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR

    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself, wx.ToolTip(strtip))

    wx.Window.SetToolTipString = sttip


# --- ---------------  Binary image File parameters
class CCDFileParameters(wx.Dialog):
    """
    Class to set binary image file parameters
    """
    def __init__(self, parent, _id, title, CCDlabel):
        """
        initialize board window
        """
        wx.Dialog.__init__(self, parent, _id, title, size=(800, 600), style=wx.RESIZE_BORDER)
        self.parent = parent
        #print("self.parent", self.parent)

        txt = wx.StaticText(self, -1, "Choose readout parameters of CCD image file")
        font = wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        txt.SetFont(font)
        txt.SetForegroundColour((255, 0, 0))

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

        a0 = wx.StaticText(self, -1, "CCD Image File type")
        a0.SetFont(font3)

        self.allCCD_names = GT.put_on_top_list(("MARCCD165",
                                            "sCMOS",
                                            "----------",
                                            "PRINCETON",
                                            "VHR_Mar13",
                                            "----------"),
                                            self.allCCD_names,
                                            forceinsertion=True)

        self.comboCCD = wx.ComboBox(self,
                                    -1,
                                    self.CCDLabel,
                                    size=(180, -1),
                                    choices=self.allCCD_names,
                                    style=wx.CB_READONLY)

        self.comboCCD.Bind(wx.EVT_COMBOBOX, self.EnterComboCCD)

        a1 = wx.StaticText(self, -1, "parameter")
        a2 = wx.StaticText(self, -1, "current value")
        a3 = wx.StaticText(self, -1, "initial value")
        a4 = wx.StaticText(self, -1, "unit")

        for text in [a1, a2, a3, a4]:
            text.SetFont(font3)

        com = wx.StaticText(self, -1, "Comments")
        com.SetFont(font3)

        self.comments = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(580, 50))
        self.comments.SetValue(DictLT.dict_CCD[self.CCDLabel][-1])

        self.DisplayValues(1)

        btnaccept = wx.Button(self, -1, "Accept", (300, 40))
        btnaccept.Bind(wx.EVT_BUTTON, self.OnAccept)
        btnaccept.SetDefault()

        btncancel = wx.Button(self, -1, "Cancel", (150, 40))
        btncancel.Bind(wx.EVT_BUTTON, self.OnCancel)

        self.keepon = False

        if WXPYTHON4:
            grid = wx.FlexGridSizer(4, 7, 10)
        else:
            grid = wx.FlexGridSizer(4, 7)

        grid.Add(a1)
        grid.Add(a2)
        grid.Add(a3)
        grid.Add(a4)
        for krow in range(6):
            grid.Add(wx.StaticText(self, -1, self.paramdetector[krow]))
            grid.Add(self.controltext[krow])
            grid.Add(wx.StaticText(self, -1, str(self.value[krow])))
            grid.Add(wx.StaticText(self, -1, self.units[krow]))

        hboxcombo = wx.BoxSizer(wx.HORIZONTAL)
        hboxcombo.Add(a0)
        hboxcombo.Add(self.comboCCD)

        hboxbtn = wx.BoxSizer(wx.HORIZONTAL)
        hboxbtn.Add(btnaccept)
        hboxbtn.Add(btncancel)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt)
        vbox.Add(hboxcombo)
        vbox.Add(grid)
        vbox.Add(com)
        vbox.Add(self.comments)
        vbox.Add(hboxbtn)

        self.SetSizer(vbox)

        # tooltip
        btnaccept.SetToolTipString("Accept")

    def readCCDparams(self):
        """ read CCD parameters from self.CCDLabel
        
        set self.value"""
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

        set self.controltext list
        """
        if self.controltext == []:
            for kk, _ in enumerate(self.paramdetector):
                self.controltext.append(wx.TextCtrl(self, -1, str(self.value[kk])))
        else:
            for kk, _ in enumerate(self.paramdetector):
                self.controltext[kk].SetValue(str(self.value[kk]))
    def DeleteValues(self, _):
        """
        set parameters to zero or None
        """
        for kk, _ in enumerate(self.paramdetector):
            self.controltext.append(wx.TextCtrl(self, -1, str(None)))

    def EnterComboCCD(self, event):
        """ select detector item """
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
        """ accept current parameters and set self.parent attributes:
        headeroffset, dataformat, fliprot, saturationvalue, CCDLabel, file_extension
        framedim, pixelsize, detectordiameter """
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
            print("self.parent.pixelsize", self.parent.pixelsize)

            self.parent.detectordiameter = (max(self.parent.framedim) * self.parent.pixelsize)
            self.Close()
        else:
            wx.MessageBox("Please select on type of CCD camera or Cancel", "Error")

    def OnCancel(self, _):
        self.Close()
