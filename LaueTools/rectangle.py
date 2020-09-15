import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


# import matplotlib.backend_bases as aa

def twoindices_positive_up_to(n, m):
    """
    build  2D integer indices up to n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("%s is not a positive integer" % str(n))
        return None

    nbpos_n = n + 1
    nbpos_m = m + 1

    gripos = np.mgrid[:n:nbpos_n * 1j, :m:nbpos_m * 1j]
    indices_pos = np.reshape(gripos.T, (nbpos_n * nbpos_m, 2))

    return indices_pos


class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()

        self.readdata()
        self.ax.imshow(self.data)

#         self.rect = Rectangle((0, 0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
#         self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.currentselection = False
        self.rectangleexists = False

    def on_press(self, event):
        if not self.rectangleexists:
            print('press')
            self.x0 = event.xdata
            self.y0 = event.ydata

            self.currentselection = True
            self.selectrect = Rectangle((self.x0, self.y0), 0.02, 0.02, facecolor='yellow', edgecolor='black')
            self.selectrect.set_linestyle('dashed')
            self.selectrect.set_alpha(0.2)
            self.ax.add_artist(self.selectrect)

    def on_release(self, event):
        if not self.rectangleexists:
            print('release')
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.selectrect.set_width(self.x1 - self.x0)
            self.selectrect.set_height(self.y1 - self.y0)
            self.selectrect.set_xy((self.x0, self.y0))
            self.selectrect.set_linestyle('solid')
            self.ax.figure.canvas.draw()

            self.currentselection = False
            self.rectangleexists = True

    def on_motion(self, event):
        if not self.rectangleexists:
            print('on_motion')
            if self.currentselection is True and event.inaxes:
                self.x1 = event.xdata
                self.y1 = event.ydata
                self.selectrect.set_width(self.x1 - self.x0)
                self.selectrect.set_height(self.y1 - self.y0)
                self.selectrect.set_xy((self.x0, self.y0))
                self.ax.figure.canvas.draw()

    def readdata(self):
        self.data = np.random.rand(20, 20)


class DraggableRectangle(object):
    def __init__(self, artists, connectingline, arrows,
                 line_segments_vert=None,
                 line_segments_hor=None,
                 nbsteps_w_h=None,
                 texts=None,
                 tolerance=5, parent=None, framedim=(2048, 2048),
                 datatype='pixels'):
        for artist in artists:
            artist.set_picker(tolerance)
        self.artists = artists
        self.currently_dragging = False
        self.current_artist = None
        self.connectingline = connectingline
        self.arrows = arrows

        self.line_segments_vert = line_segments_vert
        self.line_segments_hor = line_segments_hor
        self.nbsteps_w_h = nbsteps_w_h

        self.texts = texts

        self.parent = parent
        self.framedim = framedim
        self.offset = (0, 0)

        self.datatype = datatype

        self.keyshiftpressed = False

        canvaslist = [artist.figure.canvas for artist in self.artists]
#         print "available canvas", canvaslist

        for canvas in set(canvaslist):
            canvas.mpl_connect('button_press_event', self.on_press)
            canvas.mpl_connect('button_release_event', self.on_release)
            canvas.mpl_connect('pick_event', self.on_pick)
            canvas.mpl_connect('motion_notify_event', self.on_motion)
            canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        print("evt in drag points", event)

        key = event.key

        print("key", key)

        if key in ('shift',):
            self.keyshiftpressed = not self.keyshiftpressed

            return

        if key in ('up', 'down'):
            if key == 'down':
                self.nbsteps_w_h[0] -= 1

                if self.nbsteps_w_h[0] <= 2:
                    self.nbsteps_w_h[0] = 2

            elif key == 'up':
                self.nbsteps_w_h[0] += 1

            # gridlines management
            pt_lb = self.artists[0].center
            pt_lt = self.artists[1].center
            pt_rb = self.artists[3].center
            nbsteps_w_h = self.nbsteps_w_h
            segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, self.nbsteps_w_h[0], self.nbsteps_w_h[1])

            self.line_segments_vert.set_verts(segv)
            self.line_segments_hor.set_verts(_segh)

            # update press pt  and plot
            self.press = event.xdata, event.ydata

            self.connectingline.figure.canvas.draw()


            return


        if key in ('left', 'right'):
            if key == 'left':
                self.nbsteps_w_h[1] -= 1

                if self.nbsteps_w_h[1] <= 2:
                    self.nbsteps_w_h[1] = 2

            elif key == 'right':
                self.nbsteps_w_h[1] += 1

            # gridlines management
            pt_lb = self.artists[0].center
            pt_lt = self.artists[1].center
            pt_rb = self.artists[3].center
            nbsteps_w_h = self.nbsteps_w_h
            segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, self.nbsteps_w_h[0], self.nbsteps_w_h[1])

            self.line_segments_vert.set_verts(segv)
            self.line_segments_hor.set_verts(_segh)

            # update press pt  and plot
            self.press = event.xdata, event.ydata

            self.connectingline.figure.canvas.draw()


            return

        if key in ('p', 'm'):

            if key == 'p':
                self.increaseradius = 1
            elif key == 'm':
                self.increaseradius = -1

            self.changecircleradius()

        else:
            return

    def on_press(self, event):

        # don t mix pan/zoom with right click dragging
        toolbar = self.connectingline.figure.canvas.toolbar
        if toolbar.mode != '':
            return

        self.currently_dragging = True

        if event.button not in (1, '1',):
            print("pressing button %d" % event.button)
            self.press = event.xdata, event.ydata

            if not self.keyshiftpressed:
                self.rotatingrectangle = True
            else:
                self.rotatingrectangle = False

        else:
            self.rotatingrectangle = False

    def on_release(self, event):
        self.currently_dragging = False

        if event.button not in (1, '1',):
            self.rotatingrectangle = False
        else:
            self.current_artist = None

    def on_pick(self, event):
#         print "picking in dragpoint"
        if self.current_artist is None:
            self.current_artist = event.artist
            if event.artist != self.connectingline:
                print("touching a circle at")
                x0, y0 = event.artist.center
                print('center=', x0, y0)
                x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
                self.offset = (x0 - x1), (y0 - y1)

                print("self.offset", self.offset)
                return True

    def on_motion(self, event):
#         print "I m dragging in dragpoint!"
        if not self.currently_dragging:
            return
        if self.current_artist is None and not self.rotatingrectangle and not self.keyshiftpressed:
            return

        if self.datatype == 'pixels':
            limited_to_positive_integers = True
            xylimits = None
        elif self.datatype == 'gnomon':
            limited_to_positive_integers = False
            xylimits = (-1, 1, -1, 1)
        elif self.datatype == '2thetachi':
            # draw lines is useless
            return

#         print 'self.offset in onMotion', self.offset

        # handle rectangle rotation by moving right btn of mouse
        if event.button not in (1, '1',):

            if not self.keyshiftpressed:
                xpress, ypress = self.press
                print(" self.press", self.press)
                dx = event.xdata - xpress
                dy = event.ydata - ypress

                print("dx,dy", dx, dy)

                xMnew, yMnew = event.xdata, event.ydata
                xM, yM = xpress, ypress

                xs, ys = self.get_listpts_line()

                xC, yC = center_pts((xs[0], ys[0]), (xs[2], ys[2]))

                print("xC,yC", xC, yC)

                norme_dCM = 1.0 * math.sqrt((xC - xM) ** 2 + (yC - yM) ** 2)
                norme_dCMnew = 1.0 * math.sqrt((xC - xMnew) ** 2 + (yC - yMnew) ** 2)

                angle = math.acos(((xMnew - xC) * (xM - xC) + (yMnew - yC) * (yM - yC)) / norme_dCM / norme_dCMnew)
                cosangle = math.cos(angle)
                sinangle = math.sqrt(1 - cosangle ** 2)

                scal = (-dx * (yMnew - yC) + dy * (xMnew - xC))
                if scal < 0:
                    sign_angle = -1
                else:
                    sign_angle = 1

                MatRot = np.array([[cosangle, -sinangle * sign_angle],
                                   [sinangle * sign_angle, cosangle]])

                # last elem is the same that the fi#             angle = math.atan(0.04 * ((-dx * (yMnew - yC) + dy * (xMnew - xC)) / norme_dperpCM / norme_dCM))
    #             sinangle = math.sin(angle)
    #             cosangle = math.sqrt(1 - sinangle ** 2)rst to draw a line
                x_sc = np.array(xs[:-1]) - xC
                y_sc = np.array(ys[:-1]) - yC

                Corners = (np.array([x_sc, y_sc]).T)

                print("Corners", Corners)

                XCorners, YCorners = np.dot(MatRot, Corners.T)

                xs[:-1] = XCorners + xC
                ys[:-1] = YCorners + yC
                xs[-1] = xs[0]
                ys[-1] = ys[0]

                self.artists[0].center = xs[0], ys[0]
                self.artists[1].center = xs[1], ys[1]
                self.artists[2].center = xs[2], ys[2]
                self.artists[3].center = xs[3], ys[3]

                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)


                # manage dimension texts
                pt_rt = self.artists[2].center
                posw, wlen, posh, hlen = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)

#                 print "sedgfsd", dir(self.texts[0])
                self.texts[0].set_x(posw[0])
                self.texts[0].set_y(posw[1])

                self.texts[0].set_text('%.2f microns' % wlen)

                self.texts[1].set_x(posh[0])
                self.texts[1].set_y(posh[1])

                self.texts[1].set_text('%.2f microns' % hlen)

                # update press pt  and plot
                self.press = event.xdata, event.ydata

                self.connectingline.set_data(xs, ys)
                self.connectingline.figure.canvas.draw()

                return
            # handle rectangle size  by moving right btn of mouse
            else:
                xs, ys = self.get_listpts_line()

                h_vector = np.array([xs[1] - xs[0], ys[1] - ys[0]])
                w_vector = np.array([xs[3] - xs[0], ys[3] - ys[0]])

                wnorme = np.sqrt(w_vector[0] ** 2 + w_vector[1] ** 2)
                hnorme = np.sqrt(h_vector[0] ** 2 + h_vector[1] ** 2)

                xpress, ypress = self.press
                print(" self.press", self.press)
                dx = event.xdata - xpress
                dy = event.ydata - ypress

                print("dx,dy", dx, dy)
                vecMMprime = [dx, dy]

                vec_dw = np.dot(vecMMprime, w_vector) * w_vector / wnorme ** 2
                vec_dh = np.dot(vecMMprime, h_vector) * h_vector / hnorme ** 2

                xcorners, ycorners = xs[:-1], ys[:-1]

                corners = np.array([xcorners, ycorners]).T

                corners[0] = corners[0] - vec_dw - vec_dh
                corners[1] = corners[1] - vec_dw + vec_dh
                corners[2] = corners[2] + vec_dw + vec_dh
                corners[3] = corners[3] + vec_dw - vec_dh

                for k in range(4):
                    xs[k], ys[k] = corners[k]
                xs[4], ys[4] = xs[0], ys[0]

                self.artists[0].center = xs[0], ys[0]
                self.artists[1].center = xs[1], ys[1]
                self.artists[2].center = xs[2], ys[2]
                self.artists[3].center = xs[3], ys[3]

                self.press = event.xdata, event.ydata

                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)

                # manage dimension texts
                pt_rt = self.artists[2].center
                posw, wlen, posh, hlen = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)

#                 print "sedgfsd", dir(self.texts[0])
                self.texts[0].set_x(posw[0])
                self.texts[0].set_y(posw[1])

                self.texts[0].set_text('%.2f microns' % wlen)

                self.texts[1].set_x(posh[0])
                self.texts[1].set_y(posh[1])

                self.texts[1].set_text('%.2f microns' % hlen)

                # update press pt  and plot
                self.press = event.xdata, event.ydata


                self.connectingline.set_data(xs, ys)
                self.connectingline.figure.canvas.draw()

                return

        # hangle reaction of rectangle by moving either center, or one the 4 corners
        dx, dy = self.offset

        # central point
        if self.current_artist == self.artists[4]:

            if event.xdata is None:
                X4 = self.artists[4].center[0]
            else:
                X4 = event.xdata + dx
            if event.ydata is None:
                Y4 = self.artists[4].center[1]
            else:
                Y4 = event.ydata + dy

            evtx, evty = keep_in_imagearray(X4, Y4, self.framedim,
                                            limited_to_positive_integers=limited_to_positive_integers,
                                            xylimits=xylimits
                                            )

            self.artists[4].center = evtx, evty
#             self.current_artist.figure.canvas.draw()

#             print 'moving pt 0'
            xs, ys = self.get_listpts_line()
            print("xs, ys", xs, ys)
#             print "xs0, ys0", xs[0], ys[0]

            h_vector = (xs[1] - xs[0], ys[1] - ys[0])
            w_vector = (xs[3] - xs[0], ys[3] - ys[0])

            print("h_vector", h_vector)
            print("w_vector", w_vector)

            if event is not None:
                # left click
                if event.button in (1, '1',):
                    ChangeLineLength = False
                else:
                    print("ChangeLineLength is True")
                    ChangeLineLength = True

            xcenter = self.artists[4].center[0]
            ycenter = self.artists[4].center[1]

            if not ChangeLineLength:
    #             print "xs0, ys0", xs[0], ys[0]
                xs[0], ys[0] = xcenter - h_vector[0] / 2. - w_vector[0] / 2., ycenter - h_vector[1] / 2. - w_vector[1] / 2.
                xs[1], ys[1] = xcenter - h_vector[0] / 2. + w_vector[0] / 2., ycenter - h_vector[1] / 2. + w_vector[1] / 2.
                xs[2], ys[2] = xcenter + h_vector[0] / 2. + w_vector[0] / 2., ycenter + h_vector[1] / 2. + w_vector[1] / 2.
                xs[3], ys[3] = xcenter + h_vector[0] / 2. - w_vector[0] / 2., ycenter + h_vector[1] / 2. - w_vector[1] / 2.

                xs[4] = xs[0]
                ys[4] = ys[0]

                self.artists[0].center = xs[0], ys[0]
                self.artists[1].center = xs[1], ys[1]
                self.artists[2].center = xs[2], ys[2]
                self.artists[3].center = xs[3], ys[3]

                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)

                # manage dimension texts
                pt_rt = self.artists[2].center
                posw, wlen, posh, hlen = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)

#                 print "sedgfsd", dir(self.texts[0])
                self.texts[0].set_x(posw[0])
                self.texts[0].set_y(posw[1])

                self.texts[0].set_text('%.2f microns' % wlen)

                self.texts[1].set_x(posh[0])
                self.texts[1].set_y(posh[1])

                self.texts[1].set_text('%.2f microns' % hlen)


#
            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()


        # first  point  left bottom xs[0], ys[0]
        elif self.current_artist == self.artists[0]:

            if event.xdata is None:
                X0 = self.artists[0].center[0]
            else:
                X0 = event.xdata + dx
            if event.ydata is None:
                Y0 = self.artists[0].center[1]
            else:
                Y0 = event.ydata + dy

            evtx, evty = keep_in_imagearray(X0, Y0, self.framedim,
                                            limited_to_positive_integers=limited_to_positive_integers,
                                            xylimits=xylimits
                                            )

            self.artists[0].center = evtx, evty
#             self.current_artist.figure.canvas.draw()

#             print 'moving pt 0'
            xs, ys = self.get_listpts_line()
#             print "xs, ys", xs, ys
#             print "xs0, ys0", xs[0], ys[0]

            if event is not None:
                # left click
                if event.button in (1, '1',):
                    ChangeLineLength = False
                else:
                    ChangeLineLength = True

            xs[0] = self.artists[0].center[0]
            ys[0] = self.artists[0].center[1]

            # xs[2] fixed
            if not ChangeLineLength:
    #             print "xs0, ys0", xs[0], ys[0]

                # old frame  : h = 01 or 32  w = 12 or 03  m 02/2
                h_vector = (xs[2] - xs[3], ys[2] - ys[3])
                w_vector = (xs[2] - xs[1], ys[2] - ys[1])

#                 old_O = pt2 - (h_vector+w_vector)

                xO, yO = xs[2] - (h_vector[0] + w_vector[0]), ys[2] - (h_vector[1] + w_vector[1])

                x_newcenter, y_newcenter = center_pts((xs[0], ys[0]), (xs[2], ys[2]))

                # vec 0'1' = h_vector(1-(OO'.h_vector)/h_vector**2) = h_vector*fac1
                # vec 0'3' = w_vector(1-(OO'.w_vector)/w_vector**2) = w_vector*fac2

                vecOOprime = [xs[0] - xO, ys[0] - yO]

                fac1 = 1 - np.dot(vecOOprime, h_vector) / np.dot(h_vector, h_vector)
                fac2 = 1 - np.dot(vecOOprime, w_vector) / np.dot(w_vector, w_vector)

                xs[1] = xs[0] + fac1 * h_vector[0]
                ys[1] = ys[0] + fac1 * h_vector[1]

                xs[3] = xs[0] + fac2 * w_vector[0]
                ys[3] = ys[0] + fac2 * w_vector[1]

                xs[4] = xs[0]
                ys[4] = ys[0]


                self.artists[4].center = x_newcenter, y_newcenter
                self.artists[1].center = xs[1], ys[1]
#                 self.artists[2].center = xs[2], ys[2]
                self.artists[3].center = xs[3], ys[3]
#             # other  pt 0  move effect
#             else:
#                 # TODO
#                 xs[2] = xs[0] + 2 * (xs[1] - xs[0])
#                 ys[2] = ys[0] + 2 * (ys[1] - ys[0])
#
#                 self.artists[2].center = xs[2], ys[2]

                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)

                # manage dimension texts
                pt_rt = self.artists[2].center
                posw, wlen, posh, hlen = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)

#                 print "sedgfsd", dir(self.texts[0])
                self.texts[0].set_x(posw[0])
                self.texts[0].set_y(posw[1])

                self.texts[0].set_text('%.2f microns' % wlen)

                self.texts[1].set_x(posh[0])
                self.texts[1].set_y(posh[1])

                self.texts[1].set_text('%.2f microns' % hlen)


            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()

        # 2nd  point  left top xs[1], ys[1]
        elif self.current_artist == self.artists[1]:
#             print "moving self.artists[1]"
            if event.xdata is None:
                X1 = self.artists[1].center[0]
            else:
                X1 = event.xdata + dx
            if event.ydata is None:
                Y1 = self.artists[1].center[1]
            else:
                Y1 = event.ydata + dy

            evtx, evty = keep_in_imagearray(X1, Y1, self.framedim,
                                            limited_to_positive_integers=limited_to_positive_integers,
                                            xylimits=xylimits
                                            )

            self.artists[1].center = evtx, evty
#             self.current_artist.figure.canvas.draw()

#             print 'moving pt 0'
            xs, ys = self.get_listpts_line()
#             print "xs, ys", xs, ys
#             print "xs0, ys0", xs[0], ys[0]

            if event is not None:
                # left click
                if event.button in (1, '1',):
                    ChangeLineLength = False
                else:
                    ChangeLineLength = True

            xs[1] = self.artists[1].center[0]
            ys[1] = self.artists[1].center[1]

            # xs[3] fixed
            if not ChangeLineLength:
    #             print "xs0, ys0", xs[0], ys[0]

                # old frame  : h = 01 or 32  w = 12 or 03  m 02/2
                h_vector = (xs[2] - xs[3], ys[2] - ys[3])
                w_vector = (xs[3] - xs[0], ys[3] - ys[0])

#                 old_1 = pt3 + h_vector-w_vector

                x1, y1 = xs[3] - w_vector[0] + h_vector[0], ys[3] - w_vector[1] + h_vector[1]

                x_newcenter, y_newcenter = center_pts((xs[1], ys[1]), (xs[3], ys[3]))

                # vec 1'0' = -h_vector(1+(11'.h_vector)/h_vector**2) = h_vector*fac1
                # vec 1'2' = w_vector(1-(11'.w_vector)/w_vector**2) = w_vector*fac2

                vec11prime = [xs[1] - x1, ys[1] - y1]

                fac1 = -(1 + np.dot(vec11prime, h_vector) / np.dot(h_vector, h_vector))
                fac2 = 1 - np.dot(vec11prime, w_vector) / np.dot(w_vector, w_vector)

                xs[0] = xs[1] + fac1 * h_vector[0]
                ys[0] = ys[1] + fac1 * h_vector[1]

                xs[2] = xs[1] + fac2 * w_vector[0]
                ys[2] = ys[1] + fac2 * w_vector[1]

                xs[4] = xs[0]
                ys[4] = ys[0]

                self.artists[4].center = x_newcenter, y_newcenter
                self.artists[0].center = xs[0], ys[0]
#                 self.artists[2].center = xs[2], ys[2]
                self.artists[2].center = xs[2], ys[2]

#             # other  pt 0  move effect
#             else:
#                 # TODO
#                 xs[2] = xs[0] + 2 * (xs[1] - xs[0])
#                 ys[2] = ys[0] + 2 * (ys[1] - ys[0])
#
#                 self.artists[2].center = xs[2], ys[2]
                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)

                # manage dimension texts
                pt_rt = self.artists[2].center
                posw, wlen, posh, hlen = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)

#                 print "sedgfsd", dir(self.texts[0])
                self.texts[0].set_x(posw[0])
                self.texts[0].set_y(posw[1])

                self.texts[0].set_text('%.2f microns' % wlen)

                self.texts[1].set_x(posh[0])
                self.texts[1].set_y(posh[1])

                self.texts[1].set_text('%.2f microns' % hlen)


            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()

        # 3rd  point  right top xs[2], ys[2]
        elif self.current_artist == self.artists[2]:
#             print "moving self.artists[2]"

            if event.xdata is None:
                X2 = self.artists[2].center[0]
            else:
                X2 = event.xdata + dx
            if event.ydata is None:
                Y2 = self.artists[2].center[1]
            else:
                Y2 = event.ydata + dy

            evtx, evty = keep_in_imagearray(X2, Y2, self.framedim,
                                            limited_to_positive_integers=limited_to_positive_integers,
                                            xylimits=xylimits
                                            )

            self.artists[2].center = evtx, evty
#             self.current_artist.figure.canvas.draw()

#             print 'moving pt 0'
            xs, ys = self.get_listpts_line()
#             print "xs, ys", xs, ys
#             print "xs0, ys0", xs[0], ys[0]

            if event is not None:
                # left click
                if event.button in (1, '1',):
                    ChangeLineLength = False
                else:
                    ChangeLineLength = True

            xs[2] = self.artists[2].center[0]
            ys[2] = self.artists[2].center[1]

            # xs[0] fixed
            if not ChangeLineLength:
    #             print "xs0, ys0", xs[0], ys[0]

                # old frame  : h = 01 or 32  w = 12 or 03  m 02/2
                h_vector = (xs[1] - xs[0], ys[1] - ys[0])
                w_vector = (xs[3] - xs[0], ys[3] - ys[0])

#                 old_2 = pt0 + h_vector+w_vector

                x2, y2 = xs[0] + w_vector[0] + h_vector[0], ys[0] + w_vector[1] + h_vector[1]

                x_newcenter, y_newcenter = center_pts((xs[0], ys[0]), (xs[2], ys[2]))

                # vec 2'3' = -h_vector(1+(22'.h_vector)/h_vector**2) = h_vector*fac1
                # vec 2'1' = -w_vector(1+(22'.w_vector)/w_vector**2) = w_vector*fac2

                vec22prime = [xs[2] - x2, ys[2] - y2]
                print("vec22prime", vec22prime)

                fac1 = -(1 + np.dot(vec22prime, h_vector) / np.dot(h_vector, h_vector))
                fac2 = -(1 + np.dot(vec22prime, w_vector) / np.dot(w_vector, w_vector))

                print("fac1 and fac2", fac1, fac2)

                xs[3] = xs[2] + fac1 * h_vector[0]
                ys[3] = ys[2] + fac1 * h_vector[1]

                xs[1] = xs[2] + fac2 * w_vector[0]
                ys[1] = ys[2] + fac2 * w_vector[1]

                xs[4] = xs[0]
                ys[4] = ys[0]

                self.artists[4].center = x_newcenter, y_newcenter
                self.artists[3].center = xs[3], ys[3]
#                 self.artists[2].center = xs[2], ys[2]
                self.artists[1].center = xs[1], ys[1]

                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)

                # manage dimension texts
                pt_rt = self.artists[2].center
                posw, wlen, posh, hlen = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)

#                 print "sedgfsd", dir(self.texts[0])
                self.texts[0].set_x(posw[0])
                self.texts[0].set_y(posw[1])

                self.texts[0].set_text('%.2f microns' % wlen)

                self.texts[1].set_x(posh[0])
                self.texts[1].set_y(posh[1])

                self.texts[1].set_text('%.2f microns' % hlen)

            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()

        # 4th  point  right bottom xs[3], ys[3]
        elif self.current_artist == self.artists[3]:

            if event.xdata is None:
                X3 = self.artists[3].center[0]
            else:
                X3 = event.xdata + dx
            if event.ydata is None:
                Y3 = self.artists[3].center[1]
            else:
                Y3 = event.ydata + dy

            evtx, evty = keep_in_imagearray(X3, Y3, self.framedim,
                                            limited_to_positive_integers=limited_to_positive_integers,
                                            xylimits=xylimits
                                            )

            self.artists[3].center = evtx, evty
#             self.current_artist.figure.canvas.draw()

#             print 'moving pt 0'
            xs, ys = self.get_listpts_line()
#             print "xs, ys", xs, ys
#             print "xs0, ys0", xs[0], ys[0]

            if event is not None:
                # left click
                if event.button in (1, '1',):
                    ChangeLineLength = False
                else:
                    ChangeLineLength = True

            xs[3] = self.artists[3].center[0]
            ys[3] = self.artists[3].center[1]

            # xs[1] fixed
            if not ChangeLineLength:

                # old frame  : h = 01 or 32  w = 12 or 03  m 02/2
                h_vector = (xs[1] - xs[0], ys[1] - ys[0])
                w_vector = (xs[2] - xs[1], ys[2] - ys[1])

#                 old_3 = pt1 - h_vector+w_vector

                x3, y3 = xs[1] + w_vector[0] - h_vector[0], ys[1] + w_vector[1] - h_vector[1]

                x_newcenter, y_newcenter = center_pts((xs[3], ys[3]), (xs[1], ys[1]))

                # vec 3'2' = h_vector(1-(33'.h_vector)/h_vector**2) = h_vector*fac1
                # vec 3'0' = -w_vector(1+(33'.w_vector)/w_vector**2) = w_vector*fac2

                vec33prime = [xs[3] - x3, ys[3] - y3]

                fac1 = 1 - np.dot(vec33prime, h_vector) / np.dot(h_vector, h_vector)
                fac2 = -(1 + np.dot(vec33prime, w_vector) / np.dot(w_vector, w_vector))

                xs[2] = xs[3] + fac1 * h_vector[0]
                ys[2] = ys[3] + fac1 * h_vector[1]

                xs[0] = xs[3] + fac2 * w_vector[0]
                ys[0] = ys[3] + fac2 * w_vector[1]

                xs[4] = xs[0]
                ys[4] = ys[0]

                self.artists[4].center = x_newcenter, y_newcenter
                self.artists[2].center = xs[2], ys[2]
                self.artists[0].center = xs[0], ys[0]

                # arrows management
                xybase, xytip = self.getnewarrowparameters(xs, ys)
#                 print dir(self.arrows[0])

                self.arrows[0].xytip = xytip[0]
                self.arrows[1].xytip = xytip[1]
                self.arrows[0].xybase = xybase[0]
                self.arrows[1].xybase = xybase[1]

                # gridlines management
                pt_lb = self.artists[0].center
                pt_lt = self.artists[1].center
                pt_rb = self.artists[3].center
                nbsteps_w_h = self.nbsteps_w_h
                segv, _segh = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, nbsteps_w_h[0], nbsteps_w_h[1])

                self.line_segments_vert.set_verts(segv)
                self.line_segments_hor.set_verts(_segh)

#             # other  pt 0  move effect
#             else:
#                 # TODO
#                 xs[2] = xs[0] + 2 * (xs[1] - xs[0])
#                 ys[2] = ys[0] + 2 * (ys[1] - ys[0])
#
#                 self.artists[2].center = xs[2], ys[2]

            self.connectingline.set_data(xs, ys)
            self.connectingline.figure.canvas.draw()


        if self.parent is not None:
            # TODO to replace by  self.parent.viewingLUTpanel.myfunction(local parameters: artist center)
#             print "self.parent in DraggableLine", self.parent
            try:
                self.parent.viewingLUTpanel.x0, self.parent.viewingLUTpanel.y0 = self.artists[0].center
                self.parent.viewingLUTpanel.x2, self.parent.viewingLUTpanel.y2 = self.artists[2].center
                self.parent.viewingLUTpanel.updateLineProfile()
            except AttributeError:
                return

    def getnewarrowparameters(self, xs, ys):
        """
        from xs 's positions (line points)
        """
        xytip = [[xs[3], ys[3]], [xs[1], ys[1]]]
        xybase = [[xs[0], ys[0]], [xs[0], ys[0]]]
        return xybase, xytip

    def get_listpts_line(self):
        xs = np.array(list(self.connectingline.get_xdata()))
        ys = np.array(list(self.connectingline.get_ydata()))

        return xs, ys

    def changecircleradius(self):

        for artist in self.artists:
            artist.radius = int(artist.radius + self.increaseradius * 5.)

    def change_line_length(self):

        print("change_line_lengths")
        xs, ys = self.get_listpts_line()

        print("xs, ys", xs, ys)

        half_length_x = 0.5 * (xs[2] - xs[0])
        half_length_y = 0.5 * (ys[2] - ys[0])

        factor = 0.5

        if self.decrease_line_length == True:
            factor = .5
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

def keep_in_imagearray(x, y, framedim,
                       limited_to_positive_integers=True, xylimits=None):

        # for image pixel data
        if limited_to_positive_integers:
            if x < 0: x = 0
            if x >= framedim[1]: x = framedim[1] - 1

            if y < 0: y = 0
            if y >= framedim[0]: y = framedim[0] - 1
        # for other float with float x y coordinates
        else:
            if xylimits is not None:
                xmin, xmax, ymin, ymax = xylimits
            else:
                xmin, xmax, ymin, ymax = -1., 1., -1., 1.

            epsilonshift = (xmax - xmin) / 200.

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
    return [(pt1[0] + pt2[0]) / 2., (pt1[1] + pt2[1]) / 2.]


def getsegs_forlines_2(pt_lb, pt_rb, pt_lt, w_nbsteps, h_nbsteps):
    """
    return vertical and horizontal lines collections
    """
    w_vector = [pt_rb[0] - pt_lb[0], pt_rb[1] - pt_lb[1]]
    w_length = np.sqrt(np.dot(w_vector, w_vector))
    w_unit = w_vector / w_length

    w_steplength = 1.0 * w_length / w_nbsteps

    h_vector = [pt_lt[0] - pt_lb[0], pt_lt[1] - pt_lb[1]]
    h_length = np.sqrt(np.dot(h_vector, h_vector))
    h_unit = h_vector / h_length

    h_steplength = 1.0 * h_length / h_nbsteps

    print("w_length", w_length)
    print("h_length", h_length)

    print("w_steplength", w_steplength)
    print("h_steplength", h_steplength)

    nodes_uv = twoindices_positive_up_to(w_nbsteps, 1)

    NodesUV = nodes_uv.reshape((2, w_nbsteps + 1, 2)).transpose((1, 0, 2)).reshape((2 * (w_nbsteps + 1), 2))

    print("nodes_uv", NodesUV)

    nodes_xy = np.dot(NodesUV, np.array([w_unit * w_steplength, h_vector])) + pt_lb

    print("nodes_xy.shape", nodes_xy.shape)

    segs_vert = nodes_xy.reshape((w_nbsteps + 1, 2, 2))

    #  horizontal lines
    nodes_uv = twoindices_positive_up_to(h_nbsteps, 1)

    NodesUV = nodes_uv.reshape((2, h_nbsteps + 1, 2)).transpose((1, 0, 2)).reshape((2 * (h_nbsteps + 1), 2))

    print("nodes_uv", NodesUV)

    nodes_xy = np.dot(NodesUV, np.array([h_unit * h_steplength, w_vector])) + pt_lb

    print("nodes_xy.shape", nodes_xy.shape)

    segs_hor = nodes_xy.reshape((h_nbsteps + 1, 2, 2))

    return segs_vert, segs_hor

def get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb):
    h_vector = np.array([pt_lt[0] - pt_lb[0], pt_lt[1] - pt_lb[1]])
    w_vector = np.array([pt_rb[0] - pt_lb[0], pt_rb[1] - pt_lb[1]])
    ptcenter = np.array([0.5 * (pt_lb[0] + pt_rt[0]),
                         0.5 * (pt_lb[1] + pt_rt[1])])

    text_w_pos = ptcenter - .85 * h_vector
    text_h_pos = ptcenter + .85 * w_vector
    h_length = np.sqrt(np.dot(h_vector, h_vector))
    w_length = np.sqrt(np.dot(w_vector, w_vector))



    return text_w_pos, w_length, text_h_pos, h_length



if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set(xlim=[-1, 21], ylim=[-1, 21])

    pt_lb = [10., 10.]
    pt_lt = [10., 12.]
    pt_rt = [15., 12.]
    pt_rb = [15., 10.]

    from matplotlib.collections import LineCollection

    w_nbsteps = 3
    h_nbsteps = 5

    segs_vert, segs_hor = getsegs_forlines_2(pt_lb, pt_rb, pt_lt, w_nbsteps, h_nbsteps)
# 
    line_segments_vert = LineCollection(segs_vert,
                                linestyle='solid')
#     ax.add_collection(line_segments_vert)
# 
    line_segments_hor = LineCollection(segs_hor,
                                linestyle='solid', colors='r')
#     ax.add_collection(line_segments_hor)
# 
# 
    ptcenter = center_pts(pt_lb, pt_rt)
# 
    circles = [patches.Circle(pt_lb, 0.3, fc='g', alpha=0.5),
               patches.Circle(pt_lt, 0.3, fc='g', alpha=0.5),
               patches.Circle(pt_rt, 0.3, fc='g', alpha=0.5),
               patches.Circle(pt_rb, 0.3, fc='g', alpha=0.5),
               patches.Circle(ptcenter, 0.3, fc='r', alpha=0.5)]
# 
# #     arrows = [patches.Arrow(pt_lb[0], pt_lb[1], pt_lt[0] - pt_lb[0], pt_lt[1] - pt_lb[1], fc='g', width=2.0, alpha=0.5),
# #                patches.Arrow(pt_lb[0], pt_lb[1], pt_rb[0] - pt_lb[0], pt_rb[1] - pt_lb[1], fc='g', width=0.5, alpha=0.5)]
# 
#   # TODO use FancyArrow see SimulFrame.py
    arrows = [patches.YAArrow(fig, (pt_rb[0], pt_rb[1]), (pt_lb[0], pt_lb[1]),
                              fc='b', width=0.1, headwidth=.3, linestyle='dashed', alpha=0.5),
               patches.YAArrow(fig, (pt_lt[0], pt_lt[1]), (pt_lb[0], pt_lb[1]),
                               fc='b', width=0.3, headwidth=.9, alpha=0.5)]
# 
    drectangle, = ax.plot([pt_lb[0], pt_lt[0], pt_rt[0], pt_rb[0], pt_lb[0]],
                          [pt_lb[1], pt_lt[1], pt_rt[1], pt_rb[1], pt_lb[1]], picker=1.5)
# #     linebuilder = LineBuilder(line)
# 
    for circ in circles:
        ax.add_patch(circ)
    #
    for ar in arrows:
        ax.add_patch(ar)
# 
    text_w_pos, w_length, text_h_pos, h_length = get_dimensionstext(pt_lb, pt_lt, pt_rt, pt_rb)
# 
    textw = ax.text(text_w_pos[0], text_w_pos[1], '%.2f microns' % w_length)
    texth = ax.text(text_h_pos[0], text_h_pos[1], '%.2f microns' % h_length)
# 
# 
#     print "ax.patches", ax.patches

    dr = DraggableRectangle(circles, drectangle, arrows,
                            line_segments_vert=line_segments_vert,
                            line_segments_hor=line_segments_hor,
                            nbsteps_w_h=[w_nbsteps, h_nbsteps],
                            texts=[textw, texth],
                            tolerance=2.)
     


    plt.show()
#     a = Annotate()

