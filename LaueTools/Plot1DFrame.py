# --- ------------  1D plot class
import wx
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

from matplotlib.ticker import FuncFormatter


class Plot1DFrame(wx.Frame):
    """
    Class for plotting 1D data
    """

    def __init__(
        self,
        parent,
        id,
        title,
        title2,
        dataarray,
        figsize=5,
        dpi=100,
        radius=1.0,
        logscale=1,
        size=(500, 500),
        **kwds
    ):

        wx.Frame.__init__(self, parent, id, title, size=size)

        self.dataX, self.dataY = dataarray
        self.plot_kwds = kwds

        self.title = title
        self.title2 = title2

        self.figsize = figsize
        self.dpi = dpi

        self.radius = radius

        self.create_main_panel()

        self.scaletype = logscale
        self.init_figure_draw()

    def create_main_panel(self):
        """ main panel
        """
        self.panel = wx.Panel(self)

        if isinstance(self.figsize, int):
            self.figsizeh, self.figsizew = self.figsize, self.figsize
        else:
            self.figsizeh, self.figsizew = self.figsize

        #         self.fig = Figure((self.figsizeh, self.figsizew), dpi=self.dpi)
        self.fig = Figure((self.figsizeh, self.figsizew))
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        adjustprops = dict(
            left=0.12, bottom=0.1, right=0.9, top=0.82, wspace=0.2, hspace=0.2
        )
        self.fig.subplots_adjust(**adjustprops)

        self.tooltip = wx.ToolTip(
            tip="tip with a long %s line and a newline\n" % (" " * 100)
        )
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.toolbar = NavigationToolbar(self.canvas)
        #        self.toolbar = CustomNavToolbar(self.canvas)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.Layout()

    def on_key(self, event):
        """
        press any key to switch between log and linear y scale

        Need to click at least once before on plot
        """
        if 1:  # event.xdata != None and event.ydata != None: # mouse is inside the axes
            #            print('you pressed', event.key, event.xdata, event.ydata)
            self.scaletype += 1
            self.scaletype = self.scaletype % 2

            self.set_yscale()
            self.canvas.draw()

    def set_yscale(self):
        if self.scaletype == 0:
            self.axes.set_yscale("linear")
        else:
            self.axes.set_yscale("log")

    def updatedata(self, dataarray, title=None):
        self.dataX, self.dataY = dataarray
        if title:
            self.axes.set_title(title)

    def updateplot_from_newdata(self):

        self.line.set_data(self.dataX, self.dataY)
        self.axes.relim()
        self.axes.autoscale_view(True, True, True)
        self.canvas.draw()

    def onMotion_ToolTip(
        self, event
    ):  # tool tip to show data when mouse hovers on plot

        if self.dataX is None:
            return

        collisionFound = False

        if event.xdata != None and event.ydata != None:  # mouse is inside the axes
            tip = "x=%f\ny=%f" % (event.xdata, event.ydata)
            for i in range(len(self.dataX)):
                if (
                    abs(event.xdata - self.dataX[i]) < self.radius
                    and abs(event.ydata - self.dataY[i]) < self.radius
                ):
                    tip = "x=%f\ny=%f" % (
                        event.xdata,
                        event.ydata,
                    ) + "\nxdata = %f\nydata = %f" % (self.dataX[i], self.dataY[i])
                    #            for i in xrange(dims * dimf):
                    #                X, Y = self.Xin[0, i % dimf], self.Yin[i % dimf, 0]

                    collisionFound = True
            #            break
            self.tooltip.SetTip(tip)
            self.tooltip.Enable(True)
            return

        if not collisionFound:
            pass

    def init_figure_draw(self):
        """ init the figure
        """
        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        #        self.axes.set_autoscale_on(False) # Otherwise, infinite loop
        self.axes.set_autoscale_on(True)

        #         print "self.plot_kwds", self.plot_kwds
        self.line, = self.axes.plot(self.dataX, self.dataY, "bo-", **self.plot_kwds)
        self.axes.set_title(self.title2)

        def fromindex_to_pixelpos_x(index, pos):
            return index

        def fromindex_to_pixelpos_y(index, pos):
            return index

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))

        self.axes.grid(True)

        self.set_yscale()
        self.canvas.draw()


if __name__ == "__main__":
    title = "test"

    title2 = "test2"

    data_X = np.arange(20, 90)
    data_Y = np.random.randint(100, size=len(data_X))

    dataXY = np.array([data_X, data_Y])

    GUIApp = wx.App()
    GUIframe = Plot1DFrame(None, -1, title, title2, dataXY)
    GUIframe.Show()

    GUIApp.MainLoop()
