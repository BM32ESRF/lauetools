"""

module of lauetools project

to annotate the plot of matplotlib
"""
import pylab

from numpy import sqrt


class AnnoteFinder:
    """
    callback for matplotlib to display an annotation when points are clicked on.  The
    point which is closest to the click and within xtol and ytol is identified.
      
    Register this function like this:
      
    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotes, axis=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata)) / float(len(xdata))) / 2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata)) / float(len(ydata))) / 2
        self.xtol = xtol
        self.ytol = ytol
        if axis is None:
            self.axis = pylab.gca()
        else:
            self.axis = axis
        self.drawnAnnotations = {}
        self.links = []

    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if (self.axis is None) or (self.axis == event.inaxes):
                annotes = []
                for x, y, a in self.data:
                    if (clickX - self.xtol < x < clickX + self.xtol) and (
                        clickY - self.ytol < y < clickY + self.ytol
                    ):
                        annotes.append((pointsdistance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, axis, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.axis.figure.canvas.draw()
        else:
            # t = axis.text(x,y, "(%3.2f, %3.2f) - %s"%(x,y,annote), ) # par defaut
            t = axis.text(x, y, "%s" % (str(annote)))
            m = axis.scatter(
                [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False
            )
            self.drawnAnnotations[(x, y)] = (t, m)
            self.axis.figure.canvas.draw()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.axis, x, y, a)


def pointsdistance(x1, x2, y1, y2):
    """
	return the distance between two points
	"""
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
