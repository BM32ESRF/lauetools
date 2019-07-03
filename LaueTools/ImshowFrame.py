""" 2D imshow plot class

"""
import sys
import wx

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter

if sys.version_info.major == 3:
    from . import generaltools as GT
else:
    import generaltools as GT


class ImshowFrame(wx.Frame):
    """

    """
    def __init__(
        self,
        parent,
        _id,
        title,
        dataarray,
        Size=(4, 3),
        center=(100, 100),
        boxsize=(30, 30),
        logscale=0,
        fitfunc=None,
        fitresults=None,
        **kwds
    ):
        wx.Frame.__init__(self, parent, _id, title, size=(500, 500))

        self.dpi = 100
        self.figsize = 5
        self.title = title

        self.data = dataarray
        print("data.shape", self.data.shape)
        self.center = center
        self.boxsize = boxsize
        self.imshow_kwds = kwds
        self.fitfunc = fitfunc
        self.fitresults = fitresults
        self.title = title
        self.logscale = logscale

        self.create_main_panel()

        self.init_figure_draw()

    def create_main_panel(self):
        """

        """
        self.panel = wx.Panel(self)

        self.dpi = 100
        self.fig = Figure((self.figsize, self.figsize), dpi=self.dpi)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        self.tooltip = wx.ToolTip(
            tip="tip with a long %s line and a newline\n" % (" " * 100)
        )
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)

        self.toolbar = NavigationToolbar(self.canvas)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.Layout()

    def onMotion_ToolTip(
        self, event
    ):  # tool tip to show data when mouse hovers on plot

        if self.data is None:
            return

        collisionFound = False

        dims, dimf = self.data.shape[:2]
        #        print "self.data_2D.shape onmotion", self.data_2D.shape
        radius = 0.5
        if (
            event.xdata is not None and event.ydata is not None
        ):  # mouse is inside the axes
            #            for i in xrange(len(self.dataX)):
            #                radius = 1
            #                if abs(event.xdata - self.dataX[i]) < radius and abs(event.ydata - self.dataY[i]) < radius:
            #                    top = tip = 'x=%f\ny=%f' % (event.xdata, event.ydata)
            #            for i in xrange(dims * dimf):
            #                X, Y = self.Xin[0, i % dimf], self.Yin[i % dimf, 0]
            rx = int(np.round(event.xdata))
            ry = int(np.round(event.ydata))

            if (
                abs(rx - (dimf - 1) / 2) <= (dimf - 1) / 2
                and abs(ry - (dims - 1) / 2) <= (dims - 1) / 2
            ):
                #                print X, Y
                #                print event.xdata, event.ydata

                zvalue = self.data[ry, rx]

                tip = "X=%d\nY=%d\n(x,y):(%d %d)\nI=%.5f" % (
                    self.center[0] - self.boxsize[0] + rx,
                    self.center[1] - self.boxsize[1] + ry,
                    rx,
                    ry,
                    zvalue,
                )

                self.tooltip.SetTip(tip)
                self.tooltip.Enable(True)
                collisionFound = True
                #            break
                return
        if not collisionFound:
            pass

    def init_figure_draw(self):
        """ init the figure
        """

        def fromindex_to_pixelpos_x(index, pos):
            return self.center[0] - self.boxsize[0] + index

        def fromindex_to_pixelpos_y(index, pos):
            return self.center[1] - self.boxsize[1] + index

        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        #        self.axes.set_autoscale_on(False) # Otherwise, infinite loop
        self.axes.set_autoscale_on(True)

        if self.logscale == 1:
            if not np.any(self.data == 0):
                self.axes.imshow(np.log(self.data), **self.imshow_kwds)
        else:
            self.axes.imshow(self.data, **self.imshow_kwds)

        # plotting fitting function
        if self.fitfunc is not None:
            # print "ya quelqun? --------------"
            self.axes.contour(
                self.fitfunc(*np.indices(self.data.shape)), cmap=GT.COPPER
            )
            # self.axes.scatter(array([5]),array([5]),marker='x',edgecolor='r') # simply to play with data

        self.axes.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))
        font0 = FontProperties()
        font0.set_size("x-small")

        if self.fitresults is not None:
            if isinstance(self.fitresults, list):
                listparams_title = self.fitresults
            else:
                listparams_title = self.fitresults.tolist()
            sentence = "%s\n Intbkg= %.1f Int-Intbkg=%.1f \n(X,Y)=(%.2f,%.2f) (std1,std2)=(%.3f,%.3f) rotAng=%.1f"
            self.axes.set_title(sentence % tuple([self.title] + listparams_title))
        else:
            self.axes.set_title("%s\n" % self.title)
        self.axes.grid(True)

        self.canvas.draw()


if __name__ == "__main__":
    title = "test"

    dataarray = np.random.randint(65000, size=(301, 301))

    kwds = {"interpolation": "nearest"}

    PSGUIApp = wx.App()
    PSGUIframe = ImshowFrame(
        None,
        -1,
        title,
        dataarray,
        Size=(4, 3),
        center=(150, 150),
        boxsize=(150, 150),
        fitfunc=None,
        fitresults=None,
        **kwds
    )
    PSGUIframe.Show()

    PSGUIApp.MainLoop()
