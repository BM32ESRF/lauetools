# --- ------------  1D bar plot class
import wx

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas


class HistogramPlot(wx.Frame):
    """

    """
    def __init__( self, parent, _id, title, title2, dataarray, Size=(4, 3),
                                                                        logscale=0,
                                                                        dpi=100,
                                                                        figsize=5):
        wx.Frame.__init__(self, parent, _id, title, size=(600, 600))

        self.dpi = dpi
        self.figsize = figsize
        self.title = title

        self.data = dataarray

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
        self.bbox = (0, 200, 0, 200)

    def init_figure_draw(self):
        """ init the figure
        """
        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        #        self.axes.set_autoscale_on(False) # Otherwise, infinite loop
        self.axes.set_autoscale_on(True)

        if self.logscale:
            self.axes.set_title("pixel intensity log10(frequency)")
        else:
            self.axes.set_title("pixel intensity frequency")

        y, bins = self.data
        print(len(y), len(bins))

        if self.logscale:
            y = np.log10(y)

        self.myplot = self.axes.bar(bins[:-1], y, color="r", width=bins[1] - bins[0], log=False)
        self.axes.grid(True)

        self.canvas.draw()


if __name__ == "__main__":
    title = "test"

    title2 = "test2"

    histo = np.histogram(np.random.randint(100, size=200))

    PSGUIApp = wx.App()
    PSGUIframe = HistogramPlot(None, -1, title, title2, histo, Size=(4, 3), logscale=0,
                                                                            dpi=100, figsize=5)
    PSGUIframe.Show()

    PSGUIApp.MainLoop()
