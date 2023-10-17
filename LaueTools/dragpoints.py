import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

import numpy as np


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        print("click", event)
        if event.inaxes != self.line.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


class DraggablePoints(object):
    def __init__(self, artists, tolerance=5):
        for artist in artists:
            artist.set_picker(tolerance)
        self.artists = artists
        self.currently_dragging = False
        self.current_artist = None
        self.offset = (0, 0)

        for canvas in set(artist.figure.canvas for artist in self.artists):
            canvas.mpl_connect("button_press_event", self.on_press)
            canvas.mpl_connect("button_release_event", self.on_release)
            canvas.mpl_connect("pick_event", self.on_pick)
            canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        self.currently_dragging = True

    def on_release(self, event):
        self.currently_dragging = False
        self.current_artist = None

    def on_pick(self, event):
        if self.current_artist is None:
            self.current_artist = event.artist
            x0, y0 = event.artist.center
            x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
            self.offset = (x0 - x1), (y0 - y1)

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_artist is None:
            return
        dx, dy = self.offset
        self.current_artist.center = event.xdata + dx, event.ydata + dy
        self.current_artist.figure.canvas.draw()


class DraggableLine(object):
    def __init__(
        self,
        artists,
        connectingline,
        tolerance=5,
        parent=None,
        framedim=(2048, 2048),
        datatype="pixels",
    ):
        for artist in artists:
            artist.set_picker(tolerance)
        self.artists = artists
        self.currently_dragging = False
        self.current_artist = None
        self.connectingline = connectingline
        self.parent = parent
        self.framedim = framedim
        self.offset = (0, 0)

        self.datatype = datatype

        canvaslist = [artist.figure.canvas for artist in self.artists]
        #         print "available canvas", canvaslist

        for canvas in set(canvaslist):
            canvas.mpl_connect("button_press_event", self.on_press)
            canvas.mpl_connect("button_release_event", self.on_release)
            canvas.mpl_connect("pick_event", self.on_pick)
            canvas.mpl_connect("motion_notify_event", self.on_motion)
            canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        print("evt in drag points", event)

        key = event.key

        if key in ("+", "-"):
            if key == "+":
                self.increase_line_length = True
                self.decrease_line_length = False

            elif key == "-":
                self.decrease_line_length = True
                self.increase_line_length = False

            self.change_line_length()
            return

        if key in ("p", "m"):

            if key == "p":
                self.increaseradius = 1
            elif key == "m":
                self.increaseradius = -1

            self.changecircleradius()

        else:
            return

    def on_press(self, event):
        #         print "I m pressing !"
        self.currently_dragging = True

    def on_release(self, event):
        self.currently_dragging = False
        self.current_artist = None

    def on_pick(self, event):
        #         print "picking in dragpoint"
        if self.current_artist is None:
            self.current_artist = event.artist
            if event.artist != self.connectingline:
                #                 print "touching a circle at"
                x0, y0 = event.artist.center
                #                 print 'center=', x0, y0
                x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
                self.offset = (x0 - x1), (y0 - y1)

                #                 print "self.offset", self.offset
                return True

    def on_motion(self, event):
        #         print "I m dragging in dragpoint!"
        if not self.currently_dragging:
            return
        if self.current_artist is None:
            return

        if self.datatype == "pixels":
            limited_to_positive_integers = True
            xylimits = None
        elif self.datatype == "gnomon":
            limited_to_positive_integers = False
            xylimits = (-10, 10, -10, 10)
        elif self.datatype == "2thetachi":
            # draw lines is useless
            return

        dx, dy = self.offset

        #         print 'self.offset in onMotion', self.offset

        # first extreme point
        if self.current_artist == self.artists[0]:

            if event.xdata is None:
                X0 = self.artists[0].center[0]
            else:
                X0 = event.xdata + dx
            if event.ydata is None:
                Y0 = self.artists[0].center[1]
            else:
                Y0 = event.ydata + dy

            evtx, evty = keep_in_imagearray(
                X0,
                Y0,
                self.framedim,
                limited_to_positive_integers=limited_to_positive_integers,
                xylimits=xylimits,
            )

            self.artists[0].center = evtx, evty
            #             self.current_artist.figure.canvas.draw()

            #             print 'moving pt 0'
            xs, ys = self.get_listpts_line()
            #             print "xs, ys", xs, ys
            #             print "xs0, ys0", xs[0], ys[0]

            if event is not None:
                if event.button in (1, "1"):
                    ChangeLineLength = False
                else:
                    ChangeLineLength = True

            xs[0] = self.artists[0].center[0]
            ys[0] = self.artists[0].center[1]

            if not ChangeLineLength:
                #             print "xs0, ys0", xs[0], ys[0]
                xs[1], ys[1] = center_pts([xs[0], ys[0]], [xs[2], ys[2]])

                self.artists[1].center = xs[1], ys[1]

            # middle pt constant, second point moves
            else:
                xs[2] = xs[0] + 2 * (xs[1] - xs[0])
                ys[2] = ys[0] + 2 * (ys[1] - ys[0])

                self.artists[2].center = xs[2], ys[2]

            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()

        # second extreme point
        if self.current_artist == self.artists[2]:

            if event.xdata is None:
                X2 = self.artists[2].center[0]
            else:
                X2 = event.xdata + dx
            if event.ydata is None:
                Y2 = self.artists[2].center[1]
            else:
                Y2 = event.ydata + dy

            evtx, evty = keep_in_imagearray(
                X2,
                Y2,
                self.framedim,
                limited_to_positive_integers=limited_to_positive_integers,
                xylimits=xylimits,
            )

            self.artists[2].center = evtx, evty

            #             print 'moving pt 2'
            xs, ys = self.get_listpts_line()
            #             print "xs, ys ", xs, ys

            if event is not None:
                if event.button in (1, "1"):
                    ChangeLineLength = False
                else:
                    ChangeLineLength = True

            xs[2] = self.artists[2].center[0]
            ys[2] = self.artists[2].center[1]

            if not ChangeLineLength:

                xs[1], ys[1] = center_pts([xs[0], ys[0]], [xs[2], ys[2]])
                self.artists[1].center = xs[1], ys[1]

            # middle pt constant
            else:
                xs[0] = xs[2] - 2 * (xs[2] - xs[1])
                ys[0] = ys[2] - 2 * (ys[2] - ys[1])

                self.artists[0].center = xs[0], ys[0]

            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()
        #             self.current_artist.figure.canvas.draw()

        # middle point
        if self.current_artist == self.artists[1]:

            if event.xdata is None:
                X1 = self.artists[1].center[0]
            else:
                X1 = event.xdata + dx
            if event.ydata is None:
                Y1 = self.artists[1].center[1]
            else:
                Y1 = event.ydata + dy

            xold, yold = self.artists[1].center

            evtx, evty = keep_in_imagearray(
                X1,
                Y1,
                self.framedim,
                limited_to_positive_integers=limited_to_positive_integers,
                xylimits=xylimits,
            )

            self.artists[1].center = evtx, evty

            dxc = self.artists[1].center[0] - xold
            dyc = self.artists[1].center[1] - yold

            #             print 'moving pt central'
            xs, ys = self.get_listpts_line()

            xs[1] = self.artists[1].center[0]
            ys[1] = self.artists[1].center[1]

            xs[2] = xs[2] + dxc
            ys[2] = ys[2] + dyc
            pos2 = list(self.artists[2].center)

            pos2[0] += dxc
            pos2[1] += dyc

            self.artists[2].center = pos2

            xs[0] = xs[0] + dxc
            ys[0] = ys[0] + dyc
            pos0 = list(self.artists[0].center)

            pos0[0] += dxc
            pos0[1] += dyc

            self.artists[0].center = pos0

            #             print "self.connectingline.figure.canvas", self.connectingline.figure.canvas

            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()

        if self.parent is not None:
            # TODO to replace by  self.parent.viewingLUTpanel.myfunction(local parameters: artist center)
            #             print "self.parent in DraggableLine", self.parent
            try:
                self.parent.viewingLUTpanel.x0, self.parent.viewingLUTpanel.y0 = self.artists[
                    0
                ].center
                self.parent.viewingLUTpanel.x2, self.parent.viewingLUTpanel.y2 = self.artists[
                    2
                ].center
                self.parent.viewingLUTpanel.updateLineProfile()
            except AttributeError:
                return

    def get_listpts_line(self):
        xs = np.array(list(self.connectingline.get_xdata()))
        ys = np.array(list(self.connectingline.get_ydata()))

        return xs, ys

    def changecircleradius(self):

        for artist in self.artists:
            artist.radius = int(artist.radius + self.increaseradius * 5.0)

    def change_line_length(self):

        print("change_line_lengths")
        xs, ys = self.get_listpts_line()

        print("xs, ys", xs, ys)

        half_length_x = 0.5 * (xs[2] - xs[0])
        half_length_y = 0.5 * (ys[2] - ys[0])

        factor = 0.5

        if self.decrease_line_length == True:
            factor = 0.5
        elif self.increase_line_length == True:
            factor = 1.5

        xs[0] = xs[1] - int(half_length_x * factor)
        ys[0] = ys[1] - int(half_length_y * factor)

        xs[2] = xs[1] + int(half_length_x * factor)
        ys[2] = ys[1] + int(half_length_y * factor)

        self.artists[0].center = xs[0], ys[0]
        self.artists[2].center = xs[2], ys[2]

        self.decrease_line_length = False
        self.increase_line_length = False

        self.connectingline.set_data(xs, ys)
        self.connectingline.figure.canvas.draw()


def keep_in_imagearray(
    x, y, framedim, limited_to_positive_integers=True, xylimits=None
):

    # for image pixel data
    if limited_to_positive_integers:
        if x < 0:
            x = 0
        if x >= framedim[1]:
            x = framedim[1] - 1

        if y < 0:
            y = 0
        if y >= framedim[0]:
            y = framedim[0] - 1
    # for other float with float x y coordinates
    else:
        if xylimits is not None:
            xmin, xmax, ymin, ymax = xylimits
        else:
            xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0

        epsilonshift = (xmax - xmin) / 200.0

        if x < xmin:
            x = xmin + epsilonshift
        if x >= xmax:
            x = xmax - epsilonshift

        if y < ymin:
            y = ymin + epsilonshift
        if y >= ymax:
            y = ymax - epsilonshift

    return x, y


def center_pts(pt1, pt2):
    return [(pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0]


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set(xlim=[-1, 2], ylim=[-1, 2])

    pt1 = [0.0, 0.0]
    pt2 = [1, 1]
    ptcenter = center_pts(pt1, pt2)

    circles = [
        patches.Circle(pt1, 0.1, fc="b", alpha=0.5),
        patches.Circle(ptcenter, 0.1, fc="r", alpha=0.5),
        patches.Circle(pt2, 0.1, fc="g", alpha=0.5),
    ]

    line, = ax.plot(
        [pt1[0], ptcenter[0], pt2[0]], [pt1[1], ptcenter[1], pt2[1]], picker=1
    )
    #     linebuilder = LineBuilder(line)

    for circ in circles:
        ax.add_patch(circ)

    print("ax.patches", ax.patches)

    dr = DraggableLine(circles, line)

    plt.show()
