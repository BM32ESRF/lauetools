# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:16:17 2012

@author: micha
"""
import os
import sys
import wx
import numpy as np
import pylab

from matplotlib import __version__ as matplotlibversion
from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg as FigCanvas,
                                    NavigationToolbar2WxAgg as NavigationToolbar)
import matplotlib as mpl
from matplotlib.figure import Figure

if sys.version_info.major == 3:
    from . import generaltools as GT
    from . PlotRefineGUI import IntensityScaleBoard
else:
    import generaltools as GT
    from PlotRefineGUI import IntensityScaleBoard


SIZE_PLOTTOOLS = (6, 6)
DEG = np.pi / 180.0
CA = np.cos(40.0 * DEG)
SA = np.sin(40.0 * DEG)


class SimulationPlotFrame(wx.Frame):
    r"""
    class to plot simulated Laue pattern of one or several grains
    """
    def __init__(
        self,
        parent,
        _id,
        title,
        data=(1, 1, 1, 1, 1, "2thetachi", None),
        ImageArray=None,
        StreakingData=None,
        list_grains_transforms=None,
        dirname=os.curdir,
        CCDLabel="MARCCD165",
        **kwds
    ):
        r"""
        Class for 2D plot of spot location and get info from data

        Each element of data contains list composed of data set (e.g. corresponding to one grain)
        - [0] list of list of first coordinate of spots
        - [1] list of list of second coordinate of spots
        - [2] list of list of scalar that can be spot intensity or spot Energy
        - [3] list of list of 3d vector (3 integers defining miller indices)
        - [4] scalar giving the number of data set in the lists above (e.g. number of grains)
        - [5] string defining the type of data and coordinates (angular or cartesian) ('2thetachi','XY')
        - [6] 3-tuple for additional data to plot (e.g. experimental data):
            - [0] list of 1rst coordinates spots
            - [1] list of 2nd coordinates spots
            - [2] scalar (e.g. spot intensities)
        """
        wx.Frame.__init__(self, parent, _id, title, **kwds)

        self.SetSize(wx.Size(600, 1200))

        self.data = data

        # nb_grains = self.data[4]

        self.showFluoDetectorFrame = False
        self.showFluoDetectorFrameTools = False
        self.datatype = self.data[5]
        if self.datatype.endswith('fluo'):
            self.showFluoDetectorFrameTools = True

        self.ImageArray = ImageArray
        self.data_dict = {}
        self.data_dict["Imin"] = 1
        self.data_dict["Imax"] = 1000
        self.data_dict["vmin"] = 1
        self.data_dict["vmax"] = 1000
        self.data_dict["lut"] = "jet"
        self.data_dict["logscale"] = True
        self.data_dict["markercolor"] = "b"
        self.data_dict["CCDLabel"] = CCDLabel

        self.X_offsetfluoframe = 0
        self.Y_offsetfluoframe = 0

        self.ScatterPlot_ParentGrain = {}
        self.ScatterPlot_Grain = {}
        transformindex = 0
        self.list_grains_transforms = list_grains_transforms
        nb_ParentGrains = len(self.list_grains_transforms)
        print("nb_ParentGrains", nb_ParentGrains)
        for parentgrainindex in range(nb_ParentGrains):
            print("self.list_grains_transforms", self.list_grains_transforms[parentgrainindex])
            _, nb_transforms, transform_type = self.list_grains_transforms[parentgrainindex]
            if transform_type in ("slipsystem",):
                self.ScatterPlot_ParentGrain[parentgrainindex] = False
            else:
                self.ScatterPlot_ParentGrain[parentgrainindex] = True
            # nb_transforms  = nb of subgrains
            # with slipsystem simulation:
            # nb_transforms = nb of steps (or subgrains) / slip * nb of slips 
            for k in range(nb_transforms):
                # print( "parentgrainindex,transformindex", parentgrainindex, transformindex )
                if transform_type in ("slipsystem",):
                    self.ScatterPlot_Grain[transformindex] = False
                else:
                    self.ScatterPlot_Grain[transformindex] = True
                transformindex += 1

        # print("ScatterPlot_ParentGrain", self.ScatterPlot_ParentGrain)
        # print("self.ScatterPlot_Grain", self.ScatterPlot_Grain)

        # StreakingData = data_res, SpotIndexAccum_list, GrainParent_list, TransformType_list, slipsystemsfcc
        self.StreakingData = StreakingData

        self.CCDLabel = CCDLabel

        self.init_plot = True

        # in plotRefineGUI
        #         self.data_theo = [Twicetheta, Chi, Miller_ind]
        #         self.data_theo_pixXY = [posx, posy, Miller_ind]

        self.dirname = dirname

        self.pick_distance_mode = False
        self.points = []  # to store points
        self.selectionPoints = []
        self.twopoints = []
        self.nbclick = 1
        self.nbclick_dist = 0

        self.panel = wx.Panel(self)

        self.dpi = 100
        self.figsizex, self.figsizey = 5, 3
        self.fig = Figure((self.figsizex, self.figsizey), dpi=self.dpi)
        self.fig.set_size_inches(self.figsizex, self.figsizey, forward=True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        self.axes = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas)

        self.angulardist_btn = wx.Button(self.panel, -1, "GetAngularDistance")
        self.pixeldist_btn = wx.Button(self.panel, -1, "GetPixelDistance")
        # self.pointButton3 = wx.ToggleButton(self.panel, 3, 'Show indices')
        self.drawindicesBtn = wx.ToggleButton(self.panel, 4, "Draw indices")
        self.setImageScalebtn = wx.Button(self.panel, -1, "Set Image && Scale")
        #  self.pointButton5 = wx.Button(self.panel, 5, 'Save Plot')
        self.pointButton6 = wx.Button(self.panel, -1, "Quit")
        self.defaultColor = self.GetBackgroundColour()
        # self.Bind(wx.EVT_BUTTON, self.OnSavePlot, id=5)
        self.pointButton6.Bind(wx.EVT_BUTTON, self.OnQuit)
        self.setImageScalebtn.Bind(wx.EVT_BUTTON, self.onSetImageScale)
        self.angulardist_btn.Bind(wx.EVT_BUTTON, self.GetAngularDistance)
        self.pixeldist_btn.Bind(wx.EVT_BUTTON, self.GetCartesianDistance)

        self.txtctrldistance_angle = wx.StaticText(self.panel, -1, "==> %s deg" % "", size=(80, -1))
        self.txtctrldistance_pixel = wx.StaticText(self.panel, -1, "==> %s pixel" % "", size=(80, -1))
        if self.datatype in ("pixels",) or self.showFluoDetectorFrameTools:
            self.sidefluodetector_btn = wx.Button(self.panel, -1, "FluoDetectorFrame")
            self.txtoffset = wx.StaticText(self.panel, -1, "Fluo frame origin")
            self.txtoffsetX = wx.StaticText(self.panel, -1, "X", size=(80, -1))
            self.offsetXtxtctrl = wx.TextCtrl(self.panel, -1, "0.0", size=(75, -1))

            self.txtoffsetY = wx.StaticText(self.panel, -1, "Y", size=(80, -1))
            self.offsetYtxtctrl = wx.TextCtrl(self.panel, -1, "0.0", size=(75, -1))

            self.sidefluodetector_btn.Bind(wx.EVT_BUTTON, self.OnSideFluoDetector)
            self.offsetXtxtctrl.Bind(wx.EVT_TEXT_ENTER, self.OnEnterOffsetX)
            self.offsetYtxtctrl.Bind(wx.EVT_TEXT_ENTER, self.OnEnterOffsetY)

        self.cidpress = self.fig.canvas.mpl_connect("button_press_event", self.onClick)

        self.tooltip = wx.ToolTip(
            tip="Welcome on LaueTools Laue Pattern simulation frame"
        )
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onMotion_ToolTip)

        self.slider = wx.Slider(self.panel, -1, 50, 0, 100, size=(120, -1))
        self.slidertxt = wx.StaticText(self.panel, -1, "spot size", (5, 5))
        self.Bind(wx.EVT_SLIDER, self.sliderUpdate)

        self.statusBar = self.CreateStatusBar()

        self.readdata()

        print("factor spot size = %f " % self.factorsize)

        self._layout()

        self._replot()

        # for spot annotation
        self.drawnAnnotations = {}
        self.links = []

    def _layout(self):
        """arrange widgets
        """
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)

        bottombarsizer = wx.BoxSizer(wx.HORIZONTAL)
        bottombarsizer.Add(self.slider)
        bottombarsizer.Add(self.slidertxt)

        btnSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer.Add(self.angulardist_btn, 0, wx.BOTTOM | wx.LEFT)
        btnSizer.Add(self.txtctrldistance_angle, 0, wx.BOTTOM | wx.LEFT)
        btnSizer.Add(self.pixeldist_btn, 0, wx.BOTTOM | wx.LEFT)
        btnSizer.Add(self.txtctrldistance_pixel, 0, wx.BOTTOM | wx.LEFT)
        # btnSizer.Add(self.pointButton3, 0, wx.BOTTOM | wx.LEFT)
        btnSizer.Add(self.drawindicesBtn, 0, wx.BOTTOM | wx.LEFT)
        # btnSizer.Add(self.pointButton5, 0, wx.BOTTOM | wx.LEFT)

        btnSizer.Add(self.setImageScalebtn, 0, wx.BOTTOM | wx.LEFT)
        btnSizer.Add(bottombarsizer, 0, wx.BOTTOM | wx.LEFT)
        btnSizer.Add(self.pointButton6, 0, wx.BOTTOM | wx.LEFT)
        if self.datatype in ("pixels",) or self.showFluoDetectorFrameTools:
            btnSizer.Add(self.sidefluodetector_btn, 0, wx.BOTTOM | wx.LEFT)

            btnSizer.Add(self.txtoffset, 0, wx.BOTTOM | wx.LEFT)

            b1 = wx.BoxSizer(wx.HORIZONTAL)
            b1.Add(self.txtoffsetX, 0, wx.BOTTOM | wx.LEFT)
            b1.Add(self.offsetXtxtctrl, 0, wx.BOTTOM | wx.LEFT)

            b2 = wx.BoxSizer(wx.HORIZONTAL)
            b2.Add(self.txtoffsetY, 0, wx.BOTTOM | wx.LEFT)
            b2.Add(self.offsetYtxtctrl, 0, wx.BOTTOM | wx.LEFT)

            btnSizer.Add(b1, 0, wx.BOTTOM | wx.LEFT)
            btnSizer.Add(b2, 0, wx.BOTTOM | wx.LEFT)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.vbox, 1, wx.EXPAND)
        sizer.Add(btnSizer, 0, wx.EXPAND)

        self.panel.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()

    def sliderUpdate(self, _):
        # self.factorsize = np.exp(int(self.slider.GetValue()) / 500.) - 1.
        self.factorsize = 10 ** (
            int(self.slider.GetValue()) / 50.0 - 1.0 + np.log10(self.central_value)
        )
        self._replot()
        # print "factor spot size = %f " % self.factorsize

    def onClick(self, event):
        """ onclick
        """
        print("self.nbclick_dist start onClick", self.nbclick_dist)
        #        print 'clicked on mouse'
        if event.inaxes:
            #            print("inaxes", event)
            #             print("inaxes x,y", event.x, event.y)
            print(("inaxes  xdata, ydata", event.xdata, event.ydata))
            self.centerx, self.centery = event.xdata, event.ydata

            if self.pick_distance_mode:
                self.nbclick_dist += 1
                print("self.nbclick_dist", self.nbclick_dist)

                if self.nbclick_dist == 1:

                    x, y = event.xdata, event.ydata

                    self.twopoints = [(x, y)]

                if self.nbclick_dist == 2:

                    x, y = event.xdata, event.ydata

                    self.twopoints.append((x, y))

                    spot1 = self.twopoints[0]
                    spot2 = self.twopoints[1]

                    if self.datatype == "2thetachi":
                        _dist = GT.distfrom2thetachi(np.array(spot1), np.array(spot2))
                        print("angular distance (q1,q2):  %.3f deg " % _dist)

                    if self.datatype == "gnomon":
                        from indexingImageMatching import Fromgnomon_to_2thetachi

                        tw, ch = Fromgnomon_to_2thetachi(
                            [
                                np.array([spot1[0], spot2[0]]),
                                np.array([spot1[1], spot2[1]]),
                            ],
                            0,
                        )[:2]
                        _dist = GT.distfrom2thetachi(
                            np.array([tw[0], ch[0]]), np.array([tw[1], ch[1]])
                        )
                        print("angular distance (q1,q2):  %.3f deg " % _dist)

                    if self.datatype == "pixels":
                        # dist in pixel when data are in pixel
                        _dist = np.sqrt(
                            (spot1[0] - spot2[0]) ** 2 + (spot1[1] - spot2[1]) ** 2
                        )
                        # TODO: dist in pixel when data are in 2thetachi
                    #                         last_index = self.clicked_indexSpot[-1]
                    #                         print "last clicked", last_index
                    #                         if len(self.clicked_indexSpot) > 1:
                    #                             last_last_index = self.clicked_indexSpot[-2]
                    #                             print "former clicked", last_last_index
                    #
                    #                         spot1 = [self.tth[last_last_index], self.chi[last_last_index]]
                    #                         spot2 = [self.tth[last_index], self.chi[last_index]]
                    #
                    #                         _dist = GT.distfrom2thetachi(np.array(spot1),
                    #                                                      np.array(spot2))
                    #                         print "Angular distance (q1,q2):  %.3f deg " % _dist

                    # Reset for next pick request and display results
                    self.nbclick_dist = 0
                    # self.twopoints = []
                    self.pick_distance_mode = False
                    if self.datatype == "2thetachi":
                        print("RES =", _dist)
                        sentence = "Corresponding lattice planes angular distance"
                        sentence += (
                            "\n between two scattered directions : %.2f " % _dist
                        )
                        print(sentence)

                        self.angulardist_btn.SetBackgroundColour(self.defaultColor)
                        self.txtctrldistance_angle.SetLabel(
                            "==> %s deg" % str(np.round(_dist, decimals=3))
                        )
                    if self.datatype == "pixels":
                        print("RES =", _dist)
                        sentence = "pixel distance"
                        sentence += (
                            "\n between two scattered directions : %.2f " % _dist
                        )
                        print(sentence)
                        self.pixeldist_btn.SetBackgroundColour(self.defaultColor)
                        self.txtctrldistance_pixel.SetLabel(
                            "==> %s pixels" % str(np.round(_dist, decimals=3))
                        )

                    self.statusBar.SetStatusText(sentence, 0)

                    # DOES NOT WORK  stuck the computer!!
            #                     wx.MessageBox(sentence, 'INFO')

            #draw indices button !
            elif self.drawindicesBtn.GetValue():
                self.OnDrawIndices(event)

    def readdata(self):
        """
        read input parameter 'data'to be plotted
        """
        # if nb grains = 1

        # print self.data
        self.Data_X = self.data[0]
        self.Data_Y = self.data[1]
        self.Data_I = self.data[2]  # this column may contain energy
        self.Data_Miller = self.data[3]

        self.datatype = self.data[5]

        self.nbGrains = self.data[4]
        self.Data_index_expspot = np.arange(len(self.Data_X))

        self.experimentaldata = self.data[6]

        #         print "self.Data_I", self.Data_I
        #         print "self.Data_X", self.Data_X
        #         print "self.Data_Y", self.Data_Y

        # for many grains annotations, pixel X, pixel Y, intensity, Miller indices
        self.Xdat = []
        self.Ydat = []
        self.Idat = []
        self.Mdat = []
        self.grainindexdat = []
        self.parentgrainindexdat = []
        self.spotindex_in_grain = []
        # build data and list of spot index borders for each grain
        firstindex = 0
        lastindex = 0

        # defining limits of plot
        if self.nbGrains > 1:
            self.Xmin = min(list(map(min, self.Data_X)))
            self.Xmax = max(list(map(max, self.Data_X)))
            self.Ymin = min(list(map(min, self.Data_Y)))
            self.Ymax = max(list(map(max, self.Data_Y)))

            for k in range(self.nbGrains):
                #                 print "self.Data_X[k] type", type(self.Data_X[k])
                #                 print "len(self.data[0][k])", len(self.data[0][k])
                self.Xdat += self.Data_X[k]
                self.Ydat += self.Data_Y[k]
                self.Idat += self.Data_I[k]  # intensity or Energy
                self.Mdat += self.Data_Miller[k]  # miller indices
                lastindex = len(self.Data_X[k]) - 1
                self.spotindex_in_grain.append(firstindex)
                self.spotindex_in_grain.append(lastindex + firstindex)
                # print "firstindex",firstindex
                # print "lastindex+firstindex",lastindex+firstindex

                firstindex += lastindex + 1

            # tables containing infos on Laue spots / subgrain
            self.allspotindex = np.arange(len(self.Xdat))
            self.spotindex_in_grain = np.array(self.spotindex_in_grain)
            self.mini = self.spotindex_in_grain[::2]
            self.maxi = self.spotindex_in_grain[1::2]

            # spot size scale factor
            self.central_value = 100.0 / max(self.Idat) / 5.0

        elif self.nbGrains == 1:
            # print "rgtrg", self.Data_X[0]
            datX = self.Data_X[0]
            datY = self.Data_Y[0]
            self.Xmin = min(datX)
            self.Xmax = max(datX)
            self.Ymin = min(datY)
            self.Ymax = max(datY)

            self.Xdat = datX
            self.Ydat = datY
            self.Idat = self.Data_I[0]  # intensity or Energy
            self.Mdat = self.Data_Miller[0]

            #            self.Data_X = datX
            #            self.Data_Y = datY

            self.allspotindex = np.arange(len(self.Xdat))
            self.mini = 0
            self.maxi = len(self.Xdat)

            # spot size scale factor
            # print "self.Idat", self.Idat
            self.central_value = 100.0 / max(self.Idat) / 5.0

        self.currentbounds = ((self.Xmin, self.Xmax), (self.Ymin, self.Ymax))
        self.factorsize = self.central_value
        # print "self.currentbounds", self.currentbounds

    def onMotion_ToolTip(self, event):
        """
        tool tip to show data when mouse hovers on plot on simulFrame
        """
        ExperimentalSpots = False

        if self.pick_distance_mode:
            return

        if len(self.data[0]) == 0:
            return

        collisionFound_exp = False
        collisionFound_theo = False

        if self.datatype in ("2thetachi",):
            xtol = 2
            ytol = 2
        else:
            xtol = 50
            ytol = 50

        if ExperimentalSpots:
            xdata, ydata, _annotes_exp = (self.Data_X,
                                            self.Data_Y,
                                            list(zip(self.Data_index_expspot, self.Data_I)))

        xdata, ydata, infos = self.Xdat, self.Ydat, list(zip(self.Idat, self.Mdat))
        #         print "print xdata", xdata
        #         print "print ydata", ydata
        #         print self.Idat
        #         print self.Mdat
        # print infos
        _dataANNOTE = list(zip(xdata, ydata, infos, self.allspotindex))

        clickX = event.xdata
        clickY = event.ydata

        #         print "clickX, clickY", clickX, clickY

        if event.xdata != None and event.ydata != None:

            clickX = event.xdata
            clickY = event.ydata

            #             print 'clickX,clickY in onMotion_ToolTip', clickX, clickY

            dataabscissa_name, dataordinate_name = "X", "Y"
            if self.datatype in ("2thetachi",):
                dataabscissa_name, dataordinate_name = "2theta", "chi"

            sttext = "(%s,%s)=(%.1f,%.1f) " % (dataabscissa_name, dataordinate_name, clickX, clickY)
            if self.showFluoDetectorFrame:
                sttext += "(ydet,zdet)= (%.1f,%.1f) " % self.convertXY2ydetzdet(clickX, clickY)

            self.statusBar.SetStatusText((sttext), 0)

            if ExperimentalSpots:
                annotes_exp = []
                for x, y, a in zip(xdata, ydata, _annotes_exp):
                    if (clickX - xtol < x < clickX + xtol) and (
                        clickY - ytol < y < clickY + ytol
                    ):
                        annotes_exp.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

                if annotes_exp != []:
                    collisionFound_exp = True

            list_close_pts = []
            for x, y, a, ind in _dataANNOTE:
                if (clickX - xtol < x < clickX + xtol) and (
                    clickY - ytol < y < clickY + ytol
                ):
                    list_close_pts.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a, ind))

            if list_close_pts:
                list_close_pts.sort()
                # closest pt
                _distance, x, y, annote, ind = list_close_pts[0]

                global_spot_index = ind

                infostuple = self.getsubgraininfos(global_spot_index)
                print("the nearest simulated point is at (%.2f,%.2f)" % (x, y))
                print("with E= %.3f keV and Miller indices %s" % (annote[0], annote[1:]))
                # print("index : %d" % ind)
                # print('infostuple', str(infostuple))

                if not isinstance(self.maxi, int):
                    grain_index = np.searchsorted(self.maxi, global_spot_index)
                    first_grainindex = self.mini[grain_index]
                else:
                    grain_index = 0
                    first_grainindex = 0

                local_spot_index = global_spot_index - first_grainindex

            if list_close_pts != []:
                collisionFound_theo = True

            if not collisionFound_exp and not collisionFound_theo:
                return

            tip_exp = ""
            tip_theo = ""
            if collisionFound_exp:
                annotes_exp.sort()
                #                 print 'annotes_exp', annotes_exp
                _distance, x, y, annote_exp = annotes_exp[0]
                #             print "the nearest experimental point is at(%.2f,%.2f)" % (x, y)
                #             print "with index %d and intensity %.1f" % (annote[0], annote[1])

                # if exp. spot is close enough
                if _distance < xtol:
                    tip_exp = "spot index=%d. Intensity=%.1f" % (annote_exp[0], annote_exp[1])
                    self.updateStatusBar_theo_exp(x, y, annote_exp, spottype="exp")
                else:
                    self.sb.SetStatusText("", 1)
                    collisionFound_exp = False

            if collisionFound_theo:
                # if theo spot is close enough
                if _distance < xtol:
                    E = annote[0]
                    HKL = annote[1:4]

                    tip_theo = " SPOT @ x= %.2f y= %2f E= %.3f [h,k,l]=%s" % (x, y, E, HKL)

                    subgrainindex, parentgrainindex, transform_type = infostuple
                    #starting subgrains index:
                    stindex = self.getstartingsubgrainindex(subgrainindex) # val[0]

                    tip_theo += ' Grain_%d, subgrain_%d'%(parentgrainindex, subgrainindex)
                    tip_theo += " spotindex in grain: %d " % local_spot_index

                    if 'Slips' in self.datatype:
                        # print("Data for slips")
                        # Assuming a single slip system simulation
                        list_ParentGrain_transforms = self.StreakingData[0][7]
                        # print("list_ParentGrain_transforms",list_ParentGrain_transforms)

                        nbsteps = 11
                        slipindex = (grain_index - stindex) // nbsteps

                        if list_ParentGrain_transforms[parentgrainindex][2] == 'slipsystem':
                            plane, direction = self.StreakingData[4][slipindex]
                            tip_theo += '\nslipsystem infos: index %d' % slipindex
                            tip_theo += ': plane %s, direction %s'%(str(plane), str(direction))

                    sttext += tip_theo

                    self.statusBar.SetStatusText((sttext), 0)

                else:
                    self.statusBar.SetStatusText("")
                    collisionFound_theo = False

            if collisionFound_exp or collisionFound_theo:
                if tip_exp is not "":
                    fulltip = tip_exp + "\n" + tip_theo
                else:
                    fulltip = tip_theo

                self.tooltip.SetTip(fulltip)
                self.tooltip.Enable(True)
                return

        if not collisionFound_exp and not collisionFound_theo:
            pass

    def getsubgraininfos(self, spotindex):
        """
        from spotindex return subgrainindex, grainparentindex, transform_type
        StreakingData = data_res, SpotIndexAccum_list, GrainParent_list, TransformType_list, slipsystemsfcc
        """
        subg = np.array(self.StreakingData[1]).searchsorted(spotindex)
        GrPar = self.StreakingData[2][subg]
        transformtype = self.StreakingData[3][subg]
        return subg, GrPar, transformtype

    def getstartingsubgrainindex(self, subgrainindex):
        """ return first subgrain index of a set of grains which contains subgrain 'subgrainindex'
        """

        GrParList = self.StreakingData[2]
        GrPar = GrParList[subgrainindex]
        
        return GrParList.index(GrPar)




    def func_size_energy(self, val, factor):
        return 400.0 * factor / (val + 1.0)

    def func_size_intensity(self, val, factor, offset, lin=1):
        if lin:
            return factor * val + offset
        else:  # log scale
            return factor * np.log(np.clip(val, 0.000000001, 1000000000000)) + offset

    def fromindex_to_pixelpos_x(self, index, _):
        """
        x ticks format
        """
        return index

    def fromindex_to_pixelpos_y(self, index, _):
        """
        y ticks format
        """
        return index

    def setplotlimits_fromcurrentplot(self):
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
        print("new limits x", self.xlim)
        print("new limits y", self.ylim)

    def OnSideFluoDetector(self, _):
        """
        change frame of pixel to be see x ray like side fluorescence detector
        
        but first show frame of ydet and zdet
        """
        self.showFluoDetectorFrame = not self.showFluoDetectorFrame
        # in side Y+ frame
        if self.showFluoDetectorFrame:
            self.X_offsetfluoframe = float(self.offsetXtxtctrl.GetValue())
            self.Y_offsetfluoframe = float(self.offsetYtxtctrl.GetValue())
        self._replot()

    def OnEnterOffsetX(self, _):
        """
        set the origin of ydet zdet fluodetector frame in pixel x,y frame
        """
        self.X_offsetfluoframe = float(self.offsetXtxtctrl.GetValue())  # in pixel
        self._replot()

    def OnEnterOffsetY(self, _):
        """
        set the origin of ydet zdet fluodetector frame in pixel x,y frame
        """
        self.Y_offsetfluoframe = float(self.offsetYtxtctrl.GetValue())  # in pixel
        self._replot()

    def convertXY2ydetzdet(self, X, Y):

        #         RotYm40 = np.array([[ca, sa], [-sa, ca]])
        Xo = X - self.X_offsetfluoframe
        Yo = Y - self.Y_offsetfluoframe
        ydet = -CA * Xo + -SA * Yo
        zdet = -(SA * Xo - CA * Yo)

        return ydet, zdet

    def _replot(self):
        """
        plot spots in SimulationPlotFrame
        """
        print("self.datatype", self.datatype)
        print("self.init_plot", self.init_plot)
        # offsets to match imshow and scatter plot coordinates frames
        # coordinates in pixels
        if 'XYMAR' in self.datatype:
            self.X_offset = 1
            self.Y_offset = 1
        #coordinates 2theta chi
        else:
            self.X_offset = 0
            self.Y_offset = 0

        if not self.init_plot:
            self.setplotlimits_fromcurrentplot()

        self.axes.clear()
        self.axes.set_autoscale_on(False)  # Otherwise, infinite loop
        
        self.axes.xaxis.set_major_formatter(pylab.FuncFormatter(self.fromindex_to_pixelpos_x))
        self.axes.yaxis.set_major_formatter(pylab.FuncFormatter(self.fromindex_to_pixelpos_y))

        if self.showFluoDetectorFrame:
            print("show Fluodetector frame in side")
            # not in Marccd big nb of pixels convention
            self.X_offset = 0
            self.Y_offset = 0

            center_detframe = [self.X_offsetfluoframe, self.Y_offsetfluoframe]

            # range rectangle in ydet et zdet frame
            pt_lb = [-40.0, 0.0]
            pt_lt = [-40.0, 30.0]
            pt_rt = [40.0, 30.0]
            pt_rb = [40.0, 0.0]
            # 40 deg tilt of detector motors frame
            RotY40 = np.array([[CA, -SA], [SA, CA]])

            rot_pts = np.dot(RotY40, np.array([pt_lb, pt_lt, pt_rt, pt_rb]).T)
            trans_rot_pts = rot_pts.T + np.array(center_detframe)

            pt_lb_prime, pt_lt_prime, pt_rt_prime, pt_rb_prime = trans_rot_pts

            pt_zaxis_prime = (pt_lt_prime + pt_rt_prime) / 2.0

            from matplotlib.collections import LineCollection
            import matplotlib.patches as patches
            if sys.version_info.major == 3:
                from . import rectangle as rect
            else:
                import rectangle as rect

            w_nbsteps = 40
            h_nbsteps = 15

            segs_vert, segs_hor = rect.getsegs_forlines_2(pt_lb_prime, pt_rb_prime, pt_lt_prime,
                                                            w_nbsteps, h_nbsteps)

            line_segments_vert = LineCollection(segs_vert, linestyle="solid")
            self.axes.add_collection(line_segments_vert)

            line_segments_hor = LineCollection(segs_hor, linestyle="solid", colors="r")
            self.axes.add_collection(line_segments_hor)

            arrows = [patches.YAArrow(self.fig, (pt_lb_prime[0], pt_lb_prime[1]),
                            (center_detframe[0], center_detframe[1]), fc="g",
                            width=0.3, headwidth=0.9, alpha=0.5),
                    patches.YAArrow(self.fig, (pt_zaxis_prime[0], pt_zaxis_prime[1]),
                            (center_detframe[0], center_detframe[1]), fc="r",
                            width=0.3, headwidth=0.9, alpha=0.5)]

            for ar in arrows:
                self.axes.add_patch(ar)

        if self.ImageArray is not None:

            print("self.ImageArray", self.ImageArray.shape)
            self.myplot = self.axes.imshow(self.ImageArray, interpolation="nearest")

            if not self.data_dict["logscale"]:
                norm = mpl.colors.Normalize(vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"])
            else:
                norm = mpl.colors.LogNorm(vmin=self.data_dict["vmin"], vmax=self.data_dict["vmax"])

            self.myplot.set_norm(norm)
            self.myplot.set_cmap(self.data_dict["lut"])
            kwords = {"marker": "o", "facecolor": "None"}

        else:
            kwords = {"edgecolor": "None"}

        if self.nbGrains > 1:
            colors = GT.JET(np.arange(self.nbGrains) * 1.0 / ((self.nbGrains - 1)))
        elif self.nbGrains == 1:
            colors = [list(GT.JET(0))]

        #---------------------------------------
        # loop over grains => scatter plot
        #---------------------------------------
        print('self.nbGrains',self.nbGrains)
        for grainindex in range(self.nbGrains):

            # slip systems ---------------------
            print('grainindex %d , self.ScatterPlot_Grain[grainindex]'%grainindex,self.ScatterPlot_Grain[grainindex])
            if not self.ScatterPlot_Grain[grainindex]:
                continue
            
            # print "self.Data_X[grainindex] in plot", self.Data_X[grainindex]
            colors[grainindex][3] = 0.0  # set alpha to 0  i.e. full transparency

            if self.ImageArray is not None:
                kwords["edgecolors"] = tuple(colors[grainindex])
            else:
                kwords["c"] = tuple(colors[grainindex])

            # theo Laue spots scatter plot
            self.axes.scatter(
                np.array(self.Data_X[grainindex]) - self.X_offset,
                np.array(self.Data_Y[grainindex]) - self.Y_offset,
                s=self.func_size_energy(
                    np.array(self.Data_I[grainindex]), self.factorsize
                ),
                #                        s=self.func_size_intensity(np.array(self.Data_I[k]), self.factorsize, 0),
                #                         c=spotcolors,
                #                     edgecolors='None',
                alpha=1.0,
                **kwords
            )
        
        # plot lines to connect Laue spots for each slip
        if 'Slips' in self.datatype:# and self.isSingleStreakingPlot():
            print('\n**************\nEntering Slips line plot\n************\n')

            # self.StreakingData[0] = (
            #list_twicetheta, list_chi, list_energy, list_Miller,
            #list_posX, list_posY,
            #ParentGrainName_list, list_ParentGrain_transforms, calib, total_nb_grains, )

            ParentGrainName_list = self.StreakingData[0][6]
            list_ParentGrain_transforms = self.StreakingData[0][7]

            print('ParentGrainName_list', ParentGrainName_list)
            print('list_ParentGrain_transforms', list_ParentGrain_transforms)

            dictindicesStreakingData = getindices_StreakingData(list_ParentGrain_transforms)
            print('dictindicesStreakingData', dictindicesStreakingData)

            for elem in list_ParentGrain_transforms:
                parentgrainindex, nbtransforms, _ = elem
                print('parentgrainindex: %d, self.ScatterPlot_ParentGrain[parentgrainindex]'%parentgrainindex,
                                            self.ScatterPlot_ParentGrain[parentgrainindex])

                if self.ScatterPlot_ParentGrain[parentgrainindex]:
                    continue

                allrawX = self.StreakingData[0][4]
                allrawY = self.StreakingData[0][5]

                print('len(allrawX)', len(allrawX))

                sindex, findex = dictindicesStreakingData[parentgrainindex]
                rawX = np.array(allrawX[sindex:findex])
                rawY = np.array(allrawY[sindex:findex])

                print('rawX.shape', rawX.shape)
                nbsubgrains, nbLauespots = rawX.shape

                slipsystem = self.StreakingData[4]
                nbslips = len(slipsystem)
                nbsteps = nbsubgrains//nbslips

                print("nbLauespots", nbLauespots)
                print('nbsubgrains', nbsubgrains)
                print('nbsteps = nbsubgrains/slip', nbsteps)
                print('nbslips', nbslips)

                # print('sahpe shape',len(self.Data_X))
                # print(len(self.Data_X[0]))
                # print(self.Data_X[0])
                # print(self.Data_X[1])
                # print(self.Data_X[10])
                # print(self.Data_X[11])
                # print(self.Data_X[12])
                # XX = np.array(self.Data_X).T.reshape((nbLauespots,nbslips,nbsteps))
                # YY = np.array(self.Data_Y).T.reshape((nbLauespots,nbslips,nbsteps))

                try:
                    XX = rawX.T.reshape((nbLauespots, nbslips, nbsteps))
                    YY = rawY.T.reshape((nbLauespots, nbslips, nbsteps))
                except ValueError:
                    wx.MessageBox('Sorry!\nYou still cannot mix a slip system simulation with a single crystal one','In DEVELOPEMENT')


                # print(XX[0])
                # print(XX[0,0])
                
                colorsslip = GT.JET(np.arange(nbslips) * 1.0 / ((nbslips-1)))
                # colorsslip[parentgrainindex][3] = 0.0  # set alpha to 0  i.e. full transparency
                kwords_slip = {}
                
                # --- add lines between extreme Laue spots for each slip
                 
                for spotindex in range(nbLauespots):
                    for slipindex in range(nbslips):
                        
                        xx = XX[spotindex][slipindex]
                        yy = YY[spotindex][slipindex]
                        kwords_slip["c"] = tuple(colorsslip[slipindex])
                        s = 0
                        mid = nbsteps//2
                        e = -1
                        if spotindex==0:
                            print('[xx[s], xx[mid], xx[e]]',[xx[s], xx[mid], xx[e]])
                            print('slipindex',slipindex)
                        self.axes.plot(np.array([xx[s], xx[mid], xx[e]]) - self.X_offset,
                                        np.array([yy[s], yy[mid], yy[e]]) - self.Y_offset,
                                        '-o', **kwords_slip)
        #---------------------------------

        if self.init_plot:
            print('self.datatype',self.datatype)
            if self.datatype == "2thetachi":
                self.axes.set_xlabel("2theta (deg.)")
                self.axes.set_ylabel("chi (deg)")
                self.ylim = (-50, 50)
                self.xlim = (40, 130)
            elif "XYmar" in self.datatype:
                self.axes.set_xlabel("X (pixel)")
                self.axes.set_ylabel("Y (pixel)")
                # marccd and roper convention
                if self.CCDLabel in ("MARCCD165", "PRINCETON"):
                    self.ylim = (2048, 0)
                    self.xlim = (0, 2048)
                elif self.CCDLabel in ("sCMOS", "sCMOS_fliplr"):
                    self.ylim = (2100, 0)
                    self.xlim = (0, 2100)
                elif self.CCDLabel in ("VHR_PSI",):
                    self.ylim = (3000, 0)
                    self.xlim = (0, 4000)
                elif self.CCDLabel in ("EIGER_4M",):
                    self.ylim = (2200, 0)
                    self.xlim = (0, 2200)
                elif self.CCDLabel in ("EIGER_1M",):
                    self.ylim = (1100, 0)
                    self.xlim = (0, 1100)
                elif self.CCDLabel in ("EDF",):
                    self.ylim = (1100, 0)
                    self.xlim = (0, 1100)
                else:
                    wx.MessageBox('The camera with label "%s" is not implemented yet' % self.CCDLabel,
                                    "info",)

        self.axes.set_title("number of grain(s) : %s" % self.nbGrains)
        self.axes.grid(True)

        if self.experimentaldata:  # plot experimental data
            self.axes.scatter(
                self.experimentaldata[0] - self.X_offset,
                self.experimentaldata[1] - self.Y_offset,
                s=self.func_size_intensity(
                    np.array(self.experimentaldata[2]), 0.02, 0.0, lin=1
                ),
                facecolor="r",
                edgecolor="r",
                marker="+",
            )

        # restore the zoom limits (unless they're for an empty plot)
        if self.xlim != (0.0, 1.0) or self.ylim != (0.0, 1.0):
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)

        self.init_plot = False
        # redraw the display
        self.canvas.draw()

    def isSingleStreakingPlot(self):
        flag = True
        ParentGrainName_list = self.StreakingData[0][6]
        list_ParentGrain_transforms = self.StreakingData[0][7]

        # print('ParentGrainName_list', ParentGrainName_list)
        print('list_ParentGrain_transforms', list_ParentGrain_transforms)

        # disable slip system in case of mixture: single crystals and slip system simulation
        for elem in list_ParentGrain_transforms:
            parentgrainindex, nbtransforms, transform_type = elem
            if 'param' in transform_type:
                flag = False
                break
        return flag

    def OnSavePlot(self, _):
        dlg = wx.FileDialog(
            self,
            "Saving in png format. Choose a file",
            self.dirname,
            "",
            "*.*",
            wx.SAVE | wx.OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            # Open the file for write, write, close
            filename = dlg.GetFilename()
            # dirname = dlg.GetDirectory()

            if len(str(filename).split(".")) > 1:  # filename has a extension
                Pre, Ext = str(filename).split(".")
                if Ext != "png":
                    filename = Pre + ".png"
            else:
                filename = filename + ".png"

            self.fig.savefig(os.path.join(dirname, filename), dpi=300)

        dlg.Destroy()

    def OnQuit(self, _):
        self.Destroy()

    def updateStatusBar(self, x, y, annote, grain_index, local_spot_index):

        E = annote[0]
        HKL = annote[1:4]

        self.statusBar.SetStatusText(
            (
                "grainindex: %d local spotindex: %d " % (grain_index, local_spot_index)
                + "x= %.2f " % x
                + "y= %2f " % y
                + "E=%.5f " % E
                + "HKL=%s" % HKL
            ),
            0,
        )

    def OnDrawIndices(self, event):
        """
        OnDrawIndices in SimulationPlotFrame
        """
        xtol = 20
        ytol = 20

        #         print "self.axes.viewLim.bounds", self.axes.viewLim.bounds
        #         print "self.axes.get_xbound", self.axes.get_xbound()
        #         print "self.axes.get_ybound", self.axes.get_ybound()

        self.currentbounds = (self.axes.get_xbound(), self.axes.get_ybound())

        xdata, ydata, infos = self.Xdat, self.Ydat, list(zip(self.Idat, self.Mdat))
        #         print "print xdata", xdata
        #         print "print ydata", ydata
        #         print self.Idat
        #         print self.Mdat
        # print infos
        self._dataANNOTE = list(zip(xdata, ydata, infos, self.allspotindex))

        clickX = event.xdata
        clickY = event.ydata

        #         print "clickX, clickY", clickX, clickY

        list_close_pts = []
        for x, y, a, ind in self._dataANNOTE:
            if (clickX - xtol < x < clickX + xtol) and (
                clickY - ytol < y < clickY + ytol
            ):
                list_close_pts.append(
                    (GT.cartesiandistance(x, clickX, y, clickY), x, y, a, ind)
                )

        if list_close_pts:
            list_close_pts.sort()
            # closest pt
            _distance, x, y, annote, ind = list_close_pts[0]
            print("the nearest simulated point is at (%.2f,%.2f)" % (x, y))
            print("with E= %.3f keV and Miller indices %s" % (annote[0], annote[1:]))
            print("index : %d" % ind)

            global_spot_index = ind

            #             print 'self.maxi', self.maxi
            #             print 'self.mini', self.mini

            if not isinstance(self.maxi, int):
                grain_index = np.searchsorted(self.maxi, global_spot_index)
                first_grainindex = self.mini[grain_index]
            else:
                grain_index = 0
                first_grainindex = 0

            local_spot_index = global_spot_index - first_grainindex

            self.updateStatusBar(x, y, annote, grain_index, local_spot_index)
            self.drawAnnote(self.axes, x, y, annote)
            for l in self.links:
                l.drawSpecificAnnote(annote)
        else:
            print("you clicked too far from any spot!")

    def drawAnnote(self, axis, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())

        else:
            coef_text_offset = 5
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), ) # par defaut
            t1 = axis.text(
                x + coef_text_offset * 1,
                y + coef_text_offset * 3,
                "%.3f" % (annote[0]),
                size=8,
            )
            t2 = axis.text(
                x + coef_text_offset * 1,
                y - coef_text_offset * 3,
                "%d %d %d" % (int(annote[1][0]), int(annote[1][1]), int(annote[1][2])),
                size=8,
            )

            m = axis.scatter(
                [x],
                [y],
                s=1,
                marker="d",
                c="r",
                zorder=100,
                edgecolors="None",
                alpha=0.5,
            )  # matplotlib 0.99.1.1

            self.drawnAnnotations[(x, y)] = (t1, t2, m)

        self.canvas.draw()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.axes, x, y, a)

    def drawAnnote_exp(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index
        #from Plot_RefineFrame
        """
        if (x, y) in self.drawnAnnotations_exp:
            markers = self.drawnAnnotations_exp[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            t1 = axis.text(x + 1, y + 1, "#spot %d" % (annote[0]), size=8)
            t2 = axis.text(x + 1, y - 1, "Intens. %.1f" % (annote[1]), size=8)
            if matplotlibversion <= "0.99.1":
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False
                )
            else:
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None"
                )  # matplotlib 0.99.1.1
            self.drawnAnnotations_exp[(x, y)] = (t1, t2, m)
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()

    def drawSpecificAnnote_exp(self, annote):
        """
        from Plot_RefineFrame
        """
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_exp if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_exp(self.axes, x, y, a)

    def Annotate_exp(self, event):
        """
        from Plot_RefineFrame
        """
        xtol = 20
        ytol = 20
        # """
        # self.Data_X, self.Data_Y, self.Data_I, self.File_NAME = self.data
        # self.Data_index_expspot = np.arange(len(self.Data_X))
        # """
        xdata, ydata, annotes = (
            self.Data_X,
            self.Data_Y,
            list(zip(self.Data_index_expspot, self.Data_I)),
        )
        # print self.Idat
        # print self.Mdat
        # print annotes
        self._dataANNOTE_exp = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        # print clickX, clickY

        annotes = []
        for x, y, a in self._dataANNOTE_exp:
            if (clickX - xtol < x < clickX + xtol) and (
                clickY - ytol < y < clickY + ytol
            ):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]
            print("the nearest experimental point is at(%.2f,%.2f)" % (x, y))
            print("with index %d and intensity %.1f" % (annote[0], annote[1]))
            self.drawAnnote_exp(self.axes, x, y, annote)
            for l in self.links_exp:
                l.drawSpecificAnnote_exp(annote)

            self.updateStatusBar_theo_exp(x, y, annote, spottype="exp")

    def drawAnnote_theo(self, axis, x, y, annote):
        """
        Draw the annotation on the plot here it s exp spot index

        from Plot_RefineFrame
        """
        if (x, y) in self.drawnAnnotations_theo:
            markers = self.drawnAnnotations_theo[(x, y)]
            # print markers
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()
        else:
            # t = axis.text(x, y, "(%3.2f, %3.2f) - %s"%(x, y,annote), )  # par defaut
            t1 = axis.text(x - 7, y, "%s" % (str(annote)), size=8, color="r")

            if matplotlibversion <= "0.99.1":
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, faceted=False
                )
            else:
                m = axis.scatter(
                    [x], [y], s=1, marker="d", c="r", zorder=100, edgecolors="None"
                )  # matplotlib 0.99.1.1

            self.drawnAnnotations_theo[(x, y)] = (t1, m)
            # self.axis.figure.canvas.draw()
            #            self.plotPanel.draw()
            self.canvas.draw()

    def drawSpecificAnnote_theo(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self._dataANNOTE_theo if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote_theo(self.axes, x, y, a)

    def Annotate_theo(self, event):
        """ Display Miller indices of user clicked spot

        Add annotation in plot for the theoretical spot that is closest to user clicked point

        #from Plot_RefineFrame
        """
        xtol = 20.0
        ytol = 20.0

        if self.datatype == "2thetachi":
            xdata, ydata, annotes = self.data_theo  # 2theta chi miller

        elif self.datatype == "gnomon":
            xdata, ydata, annotes = (
                self.data_theo
            )  # sim_gnomonx, sim_gnomony, Miller_ind

        self._dataANNOTE_theo = list(zip(xdata, ydata, annotes))

        clickX = event.xdata
        clickY = event.ydata

        #         print "clickX, clickY", clickX, clickY

        annotes = []
        for x, y, a in self._dataANNOTE_theo:
            if (clickX - xtol < x < clickX + xtol) and (
                clickY - ytol < y < clickY + ytol
            ):
                annotes.append((GT.cartesiandistance(x, clickX, y, clickY), x, y, a))

        if annotes:
            annotes.sort()
            _distance, x, y, annote = annotes[0]  # this the best of annotes
            # print("the nearest theo point is at(%.2f,%.2f)" % (x, y))
            # print("with index %s " % (str(annote)))
            self.drawAnnote_theo(self.axes, x, y, annote)
            for l in self.links_theo:
                l.drawSpecificAnnote_theo(annote)

            self.updateStatusBar_theo_exp(x, y, annote)

    def updateStatusBar_theo_exp(self, x, y, annote, spottype="theo"):

        if self.datatype == "2thetachi":
            Xplot = "2theta"
            Yplot = "chi"
        else:
            Xplot = "x"
            Yplot = "y"

        if spottype == "theo":
            self.sb.SetStatusText(
                (
                    "%s= %.2f " % (Xplot, x)
                    + " %s= %.2f " % (Yplot, y)
                    + "  HKL=%s " % str(annote)
                ),
                0,
            )

        elif spottype == "exp":

            self.sb.SetStatusText(
                (
                    "%s= %.2f " % (Xplot, x)
                    + " %s= %.2f " % (Yplot, y)
                    + "   Spotindex=%d " % annote[0]
                    + "   Intensity=%.2f" % annote[1]
                ),
                1,
            )

    def GetAngularDistance(self, _):
        """
        computes angle (in deg) between the NORMALS of TWO ATOMIC PLANES corresponding of
        the two clicked spots (NOT the angular distance between the spots)
        """
        if self.datatype != "2thetachi":
            sentence = "data are not in angular coordinate!"
            GT.printred(sentence)
            self.statusBar.SetForegroundColour(wx.RED)
            self.statusBar.SetStatusText("ERROR: %s" % sentence, 0)
            return

        self.pick_distance_mode = True
        self.angulardist_btn.SetBackgroundColour("Green")

        return

    def GetCartesianDistance(self, _):
        """
        computes angle (in deg) between the normals of atomic planes corresponding of
        the two clicked spot
        """
        if self.datatype != "pixels":
            print("data are not in pixel or cartesian coordinates!")
            return

        self.pick_distance_mode = True
        self.pixeldist_btn.SetBackgroundColour("Green")

        return

    def onSetImageScale(self, _):
        """
        open a board to open image and play with its intensity scale
        """
        if self.datatype == "2thetachi":
            wx.MessageBox('Please, open a new simulation window by restarting the simulation with XYPixel "in Display parameters" checked','Info')
            return

        IScaleBoard = IntensityScaleBoard(self, -1, "Image scale setting Board", self.data_dict)

        IScaleBoard.Show(True)

def getindices_StreakingData(list_ParentGrain_transforms):
    """read list_ParentGrain_transforms and return for each element
    the positions indices to extract date for straeking plot

    # assuming that first elements are in correct order and contiguous ...

    >>> getindices_StreakingData([[0, 132, 'slipsystem'], [1, 1, 'parametric']])
    >>> {0:[0,132],1:None}
    
    >>> getindices_StreakingData([[0,15,'parametric'],[1,132,'slipsystem'],
    [2,1,'parametric'],[3,500,'slipsystem']])
    >>> {0: None, 1: [15, 147], 2: None, 3: [148, 648]}


    """
    nbparentgrains = len(list_ParentGrain_transforms)
    dictindices = {}

    accum_nb = 0
    for k in range(nbparentgrains):
        gindex, nbtransforms, transform_type = list_ParentGrain_transforms[k]
        if transform_type == 'slipsystem':
            dictindices[gindex] = [accum_nb,accum_nb + nbtransforms]
        if transform_type == 'parametric':
            dictindices[gindex] = None
        accum_nb += nbtransforms
    
    return dictindices


if __name__ == "__main__":

    import os

    dirname = os.curdir

    data_xyI_1 = np.random.rand(3, 20).tolist()
    x_1, y_1, I_1 = data_xyI_1

    data_xyI_2 = np.random.rand(3, 35).tolist()
    x_2, y_2, I_2 = data_xyI_2

    miller_1 = np.array(np.random.rand(20, 3) * 10, dtype=np.uint8).tolist()
    miller_2 = np.array(np.random.rand(35, 3) * 10, dtype=np.uint8).tolist()
    nb = 2
    plottype = "2thetachi"
    exp_data = None

    data = [
        [x_1, x_2],
        [y_1, y_2],
        [I_1 * 10000, I_2 * 10000],
        [miller_1, miller_2],
        nb,
        plottype,
        exp_data,
    ]

    # data = [x_1, y_1, I_1 * 1000, miller_1, 1, datatype, exp_data]

    # print "data", data

    class App(wx.App):
        def OnInit(self):
            """Create the main window and insert the custom frame"""

            frame = SimulationPlotFrame(
                None, -1, "Laue Simulation Frame", data=data, dirname=dirname
            )
            self.SetTopWindow(frame)
            frame.Show(True)
            return True

    app = App(0)
    app.MainLoop()
