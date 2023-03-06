import copy

import matplotlib

matplotlib.use("WXAgg")

from matplotlib import __version__ as matplotlibversion

import wx

if wx.__version__ < "4.":
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN

# --- ---------------  Plot limits board  parameters
class PlotLimitsBoard(wx.Dialog):
    """
    Class to set limits parameters of plot

    parent class must have
    xlim, ylim,  flipyaxis attributes
    _replot(), getlimitsfromplot() methods
    """
    def __init__(self, parent, _id, title, data_dict):
        """
        initialize board window
        """
        wx.Dialog.__init__(self, parent, _id, title, size=(400, 250))

        self.parent = parent

        self.data_dict = data_dict

        xlim = self.data_dict["xlim"]
        ylim = self.data_dict["ylim"]

        self.init_xlim = copy.copy(xlim)
        self.init_ylim = copy.copy(ylim)

        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None
        self.xlim, self.ylim = None, None

        font3 = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
        txt1 = wx.StaticText(self, -1, "X and Y limits controls")
        txt2 = wx.StaticText(self, -1, "Data type: %s" % self.data_dict["datatype"])

        txt1.SetFont(font3)

        self.txtctrl_xmin = wx.TextCtrl(self, -1, str(xlim[0]), style=wx.TE_PROCESS_ENTER)
        self.txtctrl_xmax = wx.TextCtrl(self, -1, str(xlim[1]), style=wx.TE_PROCESS_ENTER)
        self.txtctrl_ymin = wx.TextCtrl(self, -1, str(ylim[0]), style=wx.TE_PROCESS_ENTER)
        self.txtctrl_ymax = wx.TextCtrl(self, -1, str(ylim[1]), style=wx.TE_PROCESS_ENTER)

        self.txtctrl_xmin.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)
        self.txtctrl_xmax.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)
        self.txtctrl_ymin.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)
        self.txtctrl_ymax.Bind(wx.EVT_TEXT_ENTER, self.onEnterValue)

        fittodatabtn = wx.Button(self, -1, "Fit to Data")
        fittodatabtn.Bind(wx.EVT_BUTTON, self.onFittoData)

        acceptbtn = wx.Button(self, -1, "Accept")
        cancelbtn = wx.Button(self, -1, "Cancel")

        acceptbtn.Bind(wx.EVT_BUTTON, self.onAccept)
        cancelbtn.Bind(wx.EVT_BUTTON, self.onCancel)

        if WXPYTHON4:
            grid = wx.GridSizer(5, 10, 10)
        else:
            grid = wx.GridSizer(6, 5)

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "Y"), wx.ALIGN_CENTER_HORIZONTAL)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "MAX"), wx.ALIGN_CENTER_HORIZONTAL)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(self.txtctrl_ymax)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, "X    min"), wx.ALIGN_RIGHT)
        grid.Add(self.txtctrl_xmin)
        grid.Add(fittodatabtn)
        grid.Add(self.txtctrl_xmax)
        grid.Add(wx.StaticText(self, -1, "MAX"))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(self.txtctrl_ymin)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, "min"), wx.ALIGN_CENTER_HORIZONTAL)
        grid.Add(wx.StaticText(self, -1, ""))
        grid.Add(wx.StaticText(self, -1, ""))

        btnssizer = wx.BoxSizer(wx.HORIZONTAL)
        btnssizer.Add(acceptbtn, 0, wx.ALL)
        btnssizer.Add(cancelbtn, 0, wx.ALL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt1)
        vbox.Add(txt2)
        vbox.Add(grid)
        vbox.Add(btnssizer)

        self.SetSizer(vbox)

    def onEnterValue(self, _):
        self.readvalues()
        self.updateplot()

    def onFittoData(self, _):

        xmin = self.data_dict["dataXmin"]
        xmax = self.data_dict["dataXmax"]
        ymin = self.data_dict["dataYmin"]
        ymax = self.data_dict["dataYmax"]

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.txtctrl_xmin.SetValue(str(self.xmin))
        self.txtctrl_xmax.SetValue(str(self.xmax))
        self.txtctrl_ymin.SetValue(str(self.ymin))
        self.txtctrl_ymax.SetValue(str(self.ymax))

        self.setxylim()

        self.updateplot()

    def updateplot(self):
        self.parent.xlim = self.xlim
        self.parent.ylim = self.ylim
        self.parent.getlimitsfromplot = False
        self.parent._replot()

    def readvalues(self):
        self.xmin = float(self.txtctrl_xmin.GetValue())
        self.xmax = float(self.txtctrl_xmax.GetValue())
        self.ymin = float(self.txtctrl_ymin.GetValue())
        self.ymax = float(self.txtctrl_ymax.GetValue())

        self.setxylim()

    def setxylim(self):
        self.xlim = (self.xmin, self.xmax)

        if self.parent.flipyaxis is not None:
            if not self.parent.flipyaxis:
                self.ylim = (self.ymin, self.ymax)
            else:
                # flip up for marccd roper...
                self.ylim = (self.ymax, self.ymin)
        else:
            self.ylim = (self.ymin, self.ymax)

    def onAccept(self, _):

        self.readvalues()
        self.updateplot()

        self.Close()

    def onCancel(self, _):

        self.parent.xlim = self.init_xlim
        self.parent.ylim = self.init_ylim
        self.Close()
