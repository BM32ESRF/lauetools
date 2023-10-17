# -*- coding: utf-8 -*-
"""
module of lauetools project

purposes:

- gnomonic projection
- hough transform (in development, feasability demonstrated)
- image matching (in development, feasability demonstrated)
- zone axes recongition (in development)

js micha March 2012
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import os
import time
import sys
import pickle

import scipy.ndimage as NDI

try:

    import Image
except:
    print("-- warning: module Image or PIL is not installed, but only used for templateimagematching")
import pylab as p


import numpy as np

try:
    import ImageFilter  # should contain JSM filter in the future

except ImportError:
    import PIL.ImageFilter as ImageFilter

if sys.version_info.major == 3:
    from . import lauecore as LAUE
    from . import CrystalParameters as CP
    from . import LaueGeometry as F2TC
    from . import dict_LaueTools as DictLT
    from . import generaltools as GT
    from . import readmccd as RMCCD
    from . import IOLaueTools as IOLT
    from .annot import AnnoteFinder
    from . import IOimagefile as IOimage
    from . import imageprocessing as ImProc

else:
    import lauecore as LAUE
    import CrystalParameters as CP
    import LaueGeometry as F2TC
    import dict_LaueTools as DictLT
    import generaltools as GT
    import readmccd as RMCCD
    import IOLaueTools as IOLT
    from annot import AnnoteFinder
    import IOimagefile as IOimage
    import imageprocessing as ImProc

# --- ------------ CONSTANTS
DEG = np.pi / 180.0
CST_ENERGYKEV = DictLT.CST_ENERGYKEV

# --- -------------  PROCEDURES


def toviewgnomonofMARCCD():
    """ to plot the gnomonic projection of MARCCD chip
    """
    nbpixels = 2048 * 2048
    nbsteps = 500
    # TODO: missing args
    tata = F2TC.Compute_data2thetachi()  # missing args
    bill = ComputeGnomon_2((np.ravel(tata[0]), np.ravel(tata[1])))[0]
    p.scatter(np.take(bill[0], np.arange(0, nbpixels, nbsteps)),
                np.take(bill[1], np.arange(0, nbpixels, nbsteps)))
    p.show()


def CreateArgumentTable(_gnomonx, _gnomony):
    """
    compute table of mutual angular argument (vector lying two spots) for spots
    whose coordinates are in gnomonic projection
    TODO: seems to be used for zones axis recongnition

    """
    print(" --------------------------------------------")
    print("Calculating points alignement table")
    deltax = _gnomonx - np.reshape(_gnomonx, (len(_gnomonx), 1))
    deltay = _gnomony - np.reshape(_gnomony, (len(_gnomony), 1))

    # print "deltax",deltax

    # --- Angular argument table
    _argumtable = np.arctan2(deltay, deltax) / DEG

    print(" --------------------------------------------")
    print("\n")
    return _argumtable


def plotgnomondata(gnomonx, gnomony, X, Y, savefilename=None, maxIndexIoPlot=None, filename_data=None):
    """
    raw plots with labels of spots in gnomonic and in 2theta, chi coordinates

    """
    #    font = {'fontname'   : 'Courier', 'color'      : 'k','fontweight' : 'normal','fontsize'   : 7}

    if isinstance(maxIndexIoPlot, int):
        if maxIndexIoPlot >= 0:
            mostintense = maxIndexIoPlot + 1
        else:
            mostintense = len(gnomonx) + 1
    else:
        raise TypeError("Need integer")

    p.subplot(211)
    p.scatter(X[:mostintense], Y[:mostintense])
    p.title("exp. data (n*theta,chi)")
    p.xlabel("n*theta")
    p.ylabel("chi")

    p.subplot(212)
    p.scatter(gnomonx[:mostintense], gnomony[:mostintense])
    p.xlabel("X")
    p.ylabel("Y")
    p.xlim(-0.8, 0.8)
    p.title("gnomonic projection of spots of %s" % filename_data)
    p.grid(True)

    if savefilename:
        p.savefig(savefilename)

    mystrlabel = np.array(np.arange(len(X)), dtype="S11")

    af = AnnoteFinder(gnomonx, gnomony, mystrlabel)
    p.connect("button_press_event", af)
    p.show()


def Plot_compare_gnomon(Angles, Xgnomon_data, Ygnomon_data, key_material=14,
                                                            EULER=0,
                                                            dictmaterials=DictLT.dict_Materials):
    """
    plot data and simulation (given by list of 3 angles for orientation) in gnomonic space
    TODO: to update
    """
    emax = 25
    emin = 5
    #    nb_of_peaks=35 # nb of hough pixel featuring the laue pattern information in hough space

    angle_X, angle_Y, angle_Z = Angles

    if EULER == 0:
        mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])

    # PATCH to use correctly getLaueSpots() of laue6
    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    # array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=1,
                                verbose=0,
                                dictmaterials=dictmaterials)
    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    # Enlever les harmonics a ce niveau la!!
    # print TwicethetaChi
    lon = len(TwicethetaChi[0])
    # print "nb spots for harmonics filtering",lon

    # Removing pts that are very close deltax and delta y<0.002
    toextract = np.where(np.ravel(np.triu(np.fromfunction(GT.fct_j, (lon, lon)), k=1)) != 0)[0]
    distancetolere = 0.0002
    xxtab = TwicethetaChi[0]
    yytab = TwicethetaChi[1]
    x_ind = np.where(abs(np.ravel(xxtab - np.reshape(xxtab, (lon, 1)))[toextract]) < distancetolere)
    y_ind = np.where(abs(np.ravel(yytab - np.reshape(yytab, (lon, 1)))[toextract]) < distancetolere)
    ravx_dup = toextract[x_ind]
    ravy_dup = toextract[y_ind]
    X_dup_I, X_dup_J = ravx_dup / lon, ravx_dup % lon
    Y_dup_I, Y_dup_J = ravy_dup / lon, ravy_dup % lon

    # print [X_dup_I],[X_dup_J]
    # print [Y_dup_I],[Y_dup_J]
    # toconserve=[]
    toremove = []
    for XI_elem, XJ_elem, YI_elem, YJ_elem in zip(X_dup_I, X_dup_J, Y_dup_I, Y_dup_J):
        if XI_elem == YI_elem and XJ_elem == YJ_elem:
            # print "hourra"
            # duplicates for points index XI_elem and XJ_elem
            # toconserve.append(XI_elem)
            toremove.append(XJ_elem)
        else:
            # print "bof"
            pass
    # set() allows to remove duplicates..."
    tokeep = np.array(list(set(np.arange(lon)) - set(toremove)))
    # print "to keep",tokeep
    # print "nb of removed harmonics",(lon-len(tokeep))

    # for elem in oncam_sansh[0]:
    #    print elem._indice,elem.Twicetheta/2.,elem.Chi

    # if PlotOK:
    #    Plot_Laue(emin,emax,oncam_sansh, TwicethetaChi[:,0],TwicethetaChi[:,1], 'test', oncamwoharmonics=1,Plot_Data=1, Display_label=0,What_to_plot='2thetachi',saveplotOK=0,WriteLogFile=0)

    VIPintensity = None
    xgnomon, ygnomon = ComputeGnomon_2((xxtab[tokeep], yytab[tokeep]))

    p.title("Euler Angles [%.1f,%.1f,%.1f]" % (tuple(Angles)))
    p.scatter(Xgnomon_data, Ygnomon_data, s=50, c="w", marker="o", faceted=True, alpha=0.5)
    p.scatter(xgnomon, ygnomon, c="r", faceted=False)
    p.show()


def Plot_compare_gnomondata(Angles,
                            twicetheta_data,
                            chi_data,
                            verbose=1,
                            key_material="Si",
                            emax=25,
                            emin=5,
                            EULER=0,
                            exp_spots_list_selection=None,
                            dictmaterials=DictLT.dict_Materials):
    """
    plot data and simulation (given by list of 3 angles for orientation) in 2theta chi space (kf vector angles)

    """
    angle_X, angle_Y, angle_Z = Angles
    # TODO: branching not clear
    if type(EULER) != type(np.array([1, 2, 3])):
        if EULER == 0:
            mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
        elif EULER == 1:
            mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        if verbose:
            print("Using orientation Matrix for plotting")
        mymat = EULER

    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=1,
                                verbose=0,
                                dictmaterials=dictmaterials)

    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    xgnomon_theo, ygnomon_theo = ComputeGnomon_2((TwicethetaChi[0], TwicethetaChi[1]))

    #    print "nb of spots in CCD frame", len(TwicethetaChi[0])

    if exp_spots_list_selection is not None:  # to plot only selected list of exp. spots
        sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
        sel_chi = np.array(chi_data)[exp_spots_list_selection]

        # compute gnomonic coordinates:
        xgnomon, ygnomon = ComputeGnomon_2((sel_2theta, sel_chi))

    else:
        # compute gnomonic coordinates:
        xgnomon, ygnomon = ComputeGnomon_2((twicetheta_data, chi_data))

    p.title("Euler Angles [%.1f,%.1f,%.1f]" % (tuple(Angles)))

    # plot exp.spots
    p.scatter(xgnomon, ygnomon, s=40, c="w", marker="o", faceted=True, alpha=0.5)

    # simulated scattered spots
    p.scatter(xgnomon_theo, ygnomon_theo, c="r", faceted=False)

    p.show()


def Plot_compare_2thetachi(Angles,
                            twicetheta_data,
                            chi_data,
                            verbose=1,
                            key_material="Si",
                            emax=25,
                            emin=5,
                            EULER=0,
                            exp_spots_list_selection=None,
                            dictmaterials=DictLT.dict_Materials):
    """
    plot data and simulation (given by list of 3 angles for orientation) in 2theta chi space (kf vector angles)

    """

    angle_X, angle_Y, angle_Z = Angles
    # TODO: branching not clear
    if type(EULER) != type(np.array([1, 2, 3])):
        if EULER == 0:
            mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
        elif EULER == 1:
            mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        if verbose:
            print("Using orientation Matrix for plotting")
        mymat = EULER

    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=1,
                                verbose=0,
                                dictmaterials=dictmaterials)

    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    #    print "nb of spots in CCD frame", len(TwicethetaChi[0])

    if exp_spots_list_selection is not None:  # to plot only selected list of exp. spots
        sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
        sel_chi = np.array(chi_data)[exp_spots_list_selection]

    p.title("Euler Angles [%.1f,%.1f,%.1f]" % (tuple(Angles)))
    if exp_spots_list_selection is not None:
        p.scatter(sel_2theta, sel_chi, s=40, c="w", marker="o", faceted=True, alpha=0.5)
    else:
        p.scatter(twicetheta_data, chi_data, s=40, c="w", marker="o", faceted=True, alpha=0.5)
    # simulated scattered spots
    p.scatter(TwicethetaChi[0], TwicethetaChi[1], c="r", faceted=False)

    p.show()


def Plot_compare_2thetachi_multi(list_Angles,
                                twicetheta_data,
                                chi_data,
                                verbose=1,
                                emax=25,
                                emin=5,
                                key_material=14,
                                EULER=0,
                                exp_spots_list_selection=None,
                                title_plot="default",
                                figsize=(6, 6),
                                dpi=80,
                                dictmaterials=DictLT.dict_Materials):
    """ up to 9
    only for test or development
    Warning: blindly corrected
    """
    fig = p.figure(figsize=figsize, dpi=dpi)  # ? mouais mais dans savefig c'est ok!

    nb_of_orientations = len(list_Angles)
    if nb_of_orientations == 1:
        codefigure = 111
    if nb_of_orientations == 2:
        codefigure = 211
    if nb_of_orientations in (3, 4):
        codefigure = 221
    if nb_of_orientations in (5, 6):
        codefigure = 321
    if nb_of_orientations in (7, 8, 9):
        codefigure = 331
    index_fig = 0
    for orient_index in list_Angles:
        if type(EULER) != type(np.array([1, 2, 3])):
            if EULER == 0:
                mymat = GT.fromelemangles_toMatrix(list_Angles[orient_index])
            elif EULER == 1:
                mymat = GT.fromEULERangles_toMatrix(list_Angles[orient_index])
        else:
            mymat = EULER[orient_index]
            if verbose:
                print("Using orientation Matrix for plotting")
                print("mymat", mymat)

        # PATCH to use correctly getLaueSpots() of laue6
        grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

        # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                        CST_ENERGYKEV / emin,
                                        [grain],
                                        fastcompute=1,
                                        verbose=0,
                                        dictmaterials=dictmaterials)
        # 2theta,chi of spot which are on camera (with harmonics)
        TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

        if exp_spots_list_selection is None:  # to plot all exp. spots
            sel_2theta = np.array(twicetheta_data)
            sel_chi = np.array(chi_data)
        elif type(exp_spots_list_selection) == type(np.array([1, 2, 3])):
            sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
            sel_chi = np.array(chi_data)[exp_spots_list_selection]
        elif type(exp_spots_list_selection) == type(5):  # it is a number # for plotting
            if exp_spots_list_selection > 1:
                ind_max = min(len(twicetheta_data) - 1, exp_spots_list_selection)
                sel_2theta = np.array(twicetheta_data)[:exp_spots_list_selection]
                sel_chi = np.array(chi_data)[:exp_spots_list_selection]

        ax = fig.add_subplot(codefigure)
        if type(title_plot) == type([1, 2]):
            sco = title_plot[index_fig]
            p.title("nb close,<0.5deg: %d,%d  mean ang %.2f" % tuple(sco))
        else:
            if type(EULER) != type(np.array([1, 2, 3])):
                if EULER == 1:
                    p.title(
                        "Euler Angles [%.1f,%.1f,%.1f]"
                        % (tuple(list_Angles[orient_index]))
                    )
            else:
                p.title("Orientation Matrix #%d" % orient_index)

        ax.set_xlim((35, 145))
        ax.set_ylim((-45, 45))
        # exp spots
        ax.scatter(
            sel_2theta, sel_chi, s=40, c="w", marker="o", faceted=True, alpha=0.5
        )
        # theo spots
        ax.scatter(TwicethetaChi[0], TwicethetaChi[1], c="r", faceted=False)
        if index_fig < nb_of_orientations:
            index_fig += 1
            codefigure += 1

    p.show()


def correctangle(angle):
    """
    shift angle in between 0 and 360
    TODO: useful for zone axis recognition ?
    """
    res = angle
    if angle < 0.0:
        res = 180.0 + angle
    return res


def gen_Nuplets(items, n):
    """
    generator taking n-uplet from items, and
    """
    if n == 0:
        yield []
    else:
        for i in list(range(len(items) - n + 1)):
            for cc in gen_Nuplets(items[i + 1 :], n - 1):
                yield [items[i]] + cc


def getArgmin(tab_angulardist):
    """
    temporarly doc
    from matrix of mutual angular distances return index of closest neighbour

    TODO: to explicit documentation as a function of tab_angulardist property only
    """
    return np.argmin(tab_angulardist, axis=1)


def find_key(mydict, num):
    """ only for single value
    (if value is a list : type if num in mydict[k] )
    """
    for k in list(mydict.keys()):
        if num == mydict[k]:
            return k


# --- ------------  Gnomonic Projection

# def ComputeGnomon(_dataselected,_nbmax):
#
#    print " -------------------------------------------"
#    print "Calculating Gnomonic coordinates"
#
#
#
#    _lat=(_dataselected[0][:_nbmax]+90.)*3.14159265/180
#    _longit=(_dataselected[1][:_nbmax])*3.14159265/180
#
#    lat0=3.14159265/2. # coordinates of the tangent point # lat0=3.14159265/2. long0=0.
#    long0=0.
#
#    cosanguldist=sin(_longit)*sin(long0)+cos(_longit)*cos(long0)*cos(_lat-lat0) # cosine of angular distance from the tangent point (lat0,long0)
#
#    _gnomonx = cos(long0)*sin(lat0-_lat)/cosanguldist
#
#    _gnomony = ( cos(_longit)*sin(long0)-sin(_longit)*cos(long0)*cos(_lat-lat0) )/cosanguldist
#
#    print " -------------------------------------------"
#    print "\n"
#    return _gnomonx,_gnomony


def ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG)):
    """ compute gnomonic projection coordinates of spot's kf vector defined
    by 2theta chi.

    From an array:
    [0] array 2theta
    [1] array chi

    returns:
    array:
    [0] gnomonic X
    [1] gnomonic Y
    """
    # ---------  Computing gnomonic coordinnates

    data_theta = TwiceTheta_Chi[0] / 2.0
    data_chi = TwiceTheta_Chi[1]

    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)

    # print "lat",lat[:]
    # print min(lat),max(lat)
    # print "long",longit[:]
    # print min(longit),max(longit)

    """
    # center the gnomonic projection on particular position (e.g. peak position)
    peaklat0=arcsin(sin(72/2.*DEG)*cos(20*DEG))
    peaklongit0=sin(20.*DEG)*tan(72/2.*DEG)
    """

    centerlat, centerlongit = CenterProjection

    slat0 = np.ones(len(data_chi)) * np.sin(centerlat)
    clat0 = np.ones(len(data_chi)) * np.cos(centerlat)
    longit0 = np.ones(len(data_chi)) * centerlongit

    # print "lat0",lat0[0]
    # print "longit0",longit0[0]

    slat = np.sin(lat)
    clat = np.cos(lat)

    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    # print "cosanguldist",cosanguldist

    # print "distang",arccos(cosanguldist[:])*180./3.14159265

    _gnomonx = clat * np.sin(longit0 - longit) / cosanguldist
    _gnomony = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist

    # print "sqrt(x^2+y^2)",sqrt(_gnomonx**2+_gnomony**2)
    # print "arctan",arctan2(_gnomony,_gnomonx)
    # print "cos(arctan)",cos(arctan(_gnomony/_gnomonx))
    # print "_gnomonx",_gnomonx
    # print "_gnomony",_gnomony

    return _gnomonx, _gnomony


def InverseGnomon(_gnomonX, _gnomonY):
    """ from x,y in gnomonic projection gives lat and long
    return theta and chi of Q (direction of Q)   in radians

    WARNING: assume that center of projection is centerlat, centerlongit = 45 deg, 0
    """
    lat0 = np.ones(len(_gnomonX)) * np.pi / 4
    longit0 = np.zeros(len(_gnomonX))
    Rho = np.sqrt(_gnomonX ** 2 + _gnomonY ** 2) * 1.0
    CC = np.arctan(Rho)

    # the sign should be - !!
    lalat = np.arcsin(np.cos(CC) * np.sin(lat0) + _gnomonY / Rho * np.sin(CC) * np.cos(lat0))
    lonlongit = longit0 + np.arctan2(_gnomonX * np.sin(CC),
        Rho * np.cos(lat0) * np.cos(CC) - _gnomonY * np.sin(lat0) * np.sin(CC))

    Theta = np.arcsin(np.cos(lalat) * np.cos(lonlongit))
    Chi = np.arctan(np.sin(lonlongit) / np.tan(lalat))

    return Theta, Chi  # in radians


def Fromgnomon_to_2thetachi(gnomonicXY, _dataI):
    """ From an array: with at least 2 elements !
    [0] gnomonic X
    [1] gnomonic Y
    returns:
    array IN DEG:
    [0] array 2theta
    [1] array chi

    TODO: there is an other procedure InverseGnomon(), look and compare...
    """
    # ---------  Computing 2theta and chi coordinnates

    data_x = gnomonicXY[0]
    data_y = gnomonicXY[1]

    rho = np.sqrt(data_x ** 2 + data_y ** 2)
    argc = np.arctan(rho)

    # print "cos(argc)",cos(argc)
    # print "argc in deg",argc/DEG
    # print "rho",rho
    #
    #    lat0 = np.ones(len(data_x))*(np.pi/4) # in rads
    longit0 = np.zeros(len(data_x))  # in rads

    slat0 = np.ones(len(data_x)) * np.sin(np.pi / 4)
    clat0 = np.ones(len(data_x)) * np.cos(np.pi / 4)

    sargc = np.sin(argc)
    cargc = np.cos(argc)

    lat = np.arcsin(cargc * slat0 + (data_y * sargc * clat0) * np.nan_to_num(1 / rho))
    longit = longit0 + np.arctan2(data_x * sargc, rho * clat0 * cargc - data_y * slat0 * sargc)

    # print "lat",lat
    # print "longit",longit

    # TODO: sign of chi must be checked
    _chi = np.arctan(np.sin(longit) / np.tan(lat)) / DEG
    _twicetheta = 2 * np.arcsin(np.cos(lat) * np.cos(longit)) / DEG

    return _twicetheta, _chi, _dataI


def accum_one_point2(col_arrayrho, _pointintensity, long_arraydeg, bibins, binnedarraydeg):
    """
    compute the sinogram in hough space for one spot taking into account its intensity

    col_arrayrho
    bibins    :    1D array of rho values
    TODO: to finish and integrate to ComputeHough
    """

    intermediate_accum = np.zeros(long_arraydeg * len(bibins))
    binnedrho = np.clip(np.digitize(col_arrayrho, bibins), 0, len(bibins) - 1)
    onepointcontribution_list_index = binnedarraydeg + long_arraydeg * binnedrho
    # print "col_arrayrho",col_arrayrho
    #            p.figure(1)
    #            p.plot(col_arrayrho)
    # print "binnedarraydeg",binnedarraydeg
    # print "binnedrho",binnedrho
    # print "onepointcontribution_list_index",onepointcontribution_list_index

    if np.__version__ < "1.3.0":
        np.put(intermediate_accum, onepointcontribution_list_index, _pointintensity)
    else:
        intermediate_accum[onepointcontribution_list_index] = _pointintensity * np.ones(
            len(onepointcontribution_list_index))

    # print "binned rho",len(binnedrho)
    # print "_pointintensity",_pointintensity
    # print "onepointcontribution_list_index",onepointcontribution_list_index[onepointcontribution_list_index>140000]
    # print "intermediate_accum",intermediate_accum[intermediate_accum!=1]

    # print "where non nul",np.where(intermediate_accum!=0)
    # print np.where(intermediate_accum>0)
    # print intermediate_accum[60030:60040]
    # print "len(intermediate_accum)",len(intermediate_accum)
    return intermediate_accum


# --- ----------------  Hough Transform
def ComputeHough(_datagnomonx, _datagnomony, Intensity_table=None,
                                            stepdeg=0.5,
                                            steprho=0.002,
                                            lowresolution=0):
    """ Compute hough transform from gnomonic data with tangent plane
    at lat=45 deg anf longit=0 deg

    Returns three arrays:

    rho and theta of straight line equation are digitized into bins
    len(bibins) in rho axe
    long_arraydeg=len(arraydeg) in theta axe

    TODO:generalize for any centerProjection as a function
    of the tangent plane position used in computegnomon()

    TODO: there is annoying artefact intensity for lowest rho boundary (not for the highest one)  
    """
    if lowresolution:
        steprho = 0.004
        stepdeg = 1.0
    else:
        # using input args
        pass

    steprad = stepdeg * DEG

    # center of mass of MARCCDcamera gnomonic projection
    centerProjection = (-0.06, -0.09)

    arraydeg = np.arange(0, 2 * np.pi, steprad)
    long_arraydeg = len(arraydeg)
    # print "Number of angle bins",long_arraydeg
    binnedarraydeg = np.arange(long_arraydeg)
    #    print "len(arraydeg): nb of bins of theta", len(arraydeg)
    # print arraydeg
    _deg = np.reshape(arraydeg, (long_arraydeg, 1))

    arrayrho = (np.array(_datagnomonx) - centerProjection[0]) * np.cos(_deg)
    ycos = (np.array(_datagnomony) - centerProjection[1]) * np.sin(_deg)

    np.add(arrayrho, ycos, arrayrho)
    # arrayrho=(_datagnomonx+.06)*cos(_deg)+(_datagnomony+.09)*sin(_deg)

    # columns are rho versus deg (sinusoide) (nb of columns = nb of points)
    # line are values of rho for all the deg value

    # print "arrayrho[:,6]",arrayrho[:,6] # contient une sinusoide
    # t=arange(len(arrayrho[:,0]))
    # plot(t,arrayrho[:,0],t,arrayrho[:,6])

    if Intensity_table is None:
        bibins = np.concatenate((np.arange(-1.0 + 0.4, 1.0 - 0.4, steprho), np.array([10])))
        # print "bibins",bibins[:20],bibins[-5:]
        # print "len(bibins)",len(bibins)
        def bincountatfixedangle(_tablerho_line):
            """
            bincount for _tablerho_line
            """
            return np.bincount(np.concatenate((np.digitize(_tablerho_line, bibins), np.arange(len(bibins)))))

        resus = np.array(list(map(bincountatfixedangle, arrayrho)))

        return resus[:, :-1]
    else:
        # For example for a 3x3 binning:
        # mask = ones((3,3))
        # binned = convolve2d(data,mask,'same')[1::3,1::3]

        bibins = np.arange(-1.0 + 0.4, 1.0 - 0.4, steprho)

        #        indexedbibins=arange(len(bibins))
        print("len(bibins) ie nb of bins of rho", len(bibins))
        # print "bibins",bibins[:20],bibins[-5:]
        # print indexedbibins[:20],indexedbibins[-5:]

        # create an array white accumulation of each contribution of the p=x*cos deg + y*sin deg
        accumHough = np.zeros(long_arraydeg * len(bibins))
        print("nb of elements in the Hough table", len(accumHough))

        # line on arrayrho is a deg constant; column on arrayrho is a rho constant, this last contribution correspond to 1 pt in gnomonic projction and a single intensity
        # A(deg,rho)= Intensity(rho cst=1 column in arrayrho) at position binned (deg,rho) !!!!

        # for one point (1 arrayrho column)

        def accum_one_point(col_arrayrho, _pointintensity):
            """
            compute the sinogram in hough space for one spot taking into account its intensity

            uses bibins
            """

            intermediate_accum = np.zeros(long_arraydeg * len(bibins))
            binnedrho = np.clip(np.digitize(col_arrayrho, bibins), 0, len(bibins) - 1)
            onepointcontribution_list_index = binnedarraydeg + long_arraydeg * binnedrho
            # print "col_arrayrho",col_arrayrho
            #            p.figure(1)
            #            p.plot(col_arrayrho)
            # print "binnedarraydeg",binnedarraydeg
            # print "binnedrho",binnedrho
            # print "onepointcontribution_list_index",onepointcontribution_list_index

            if np.__version__ < "1.3.0":
                np.put(intermediate_accum, onepointcontribution_list_index, _pointintensity)
            else:
                va = _pointintensity * np.ones(len(onepointcontribution_list_index))
                intermediate_accum[onepointcontribution_list_index] = va

            # print "binned rho",len(binnedrho)
            # print "_pointintensity",_pointintensity
            # print "onepointcontribution_list_index",onepointcontribution_list_index[onepointcontribution_list_index>140000]
            # print "intermediate_accum",intermediate_accum[intermediate_accum!=1]

            # print "where non nul",np.where(intermediate_accum!=0)
            # print np.where(intermediate_accum>0)
            # print intermediate_accum[60030:60040]
            # print "len(intermediate_accum)",len(intermediate_accum)
            return intermediate_accum

        def rhotable_from_one_point(pointcoordinate):
            """
            rhotable_from_one_point pointcoordinate
            """
            # print "pointcoordinate",pointcoordinate
            # print "arraydeg",arraydeg
            ptx = np.ones(len(arraydeg)) * (pointcoordinate[0] - centerProjection[0])  # be careful of the offsets
            pty = np.ones(len(arraydeg)) * (pointcoordinate[1] - centerProjection[1])

            np.multiply(ptx, np.cos(arraydeg), ptx)
            np.multiply(pty, np.sin(arraydeg), pty)
            np.add(ptx, pty, ptx)
            # print "ptx",ptx
            return ptx

        tra = np.array([_datagnomonx, _datagnomony]).T
        # TwicethetaChi=map(rhotable_from_one_point,tra)

        # ---TOO LONG !!!
        for k in list(range(len(_datagnomonx))):  # JSM 21 OCt 2010   10 what for ???
            # print "tra",tra[k]
            # print "intiti",Intensity_table[k]
            # print "%d/%d"%(k,len(_datagnomonx)-1)
            accumHough = accumHough + accum_one_point(rhotable_from_one_point(tra[k]), Intensity_table[k])
        # ---TOO LONG !!! -

        # print "rho digitized for point #0",np.where(onepointcontri!=0)
        # print "arrayrho[:,0]",arrayrho[:,0]

        return np.reshape(accumHough, (len(bibins), long_arraydeg))


"""def One_line_binned(index_of_line): #for a given deg value, we binned in rho and add corresponding intensities
            position_line_digi = digitize( arrayrho[index_of_line] , bibins ) # give an array where each element is the corresponding index in the bins

            def addintensity_in_bin(binning_value):
                # give for only one binning index the sum of intensities that come from one or many pixels
                return np.sum(np.where(position_line_digi==binning_value,1,0)*Intensity_line)
            # array where each element is the net intensity corresponding to binning index
            return line_with_addedintensities=map(addintensity_in_bin(binning_value),indexedbibins)
"""


def InverseHough(rhotheta, halfsemiaxes=(1.2, 0.6), centerofproj=(0.0, 0.0)):
    """ from rho and theta in hough space
    gives extreme points of a straight line where must lie all spots
    belonging to the same zone axis
    In a rigourous manner, we must find the intersection of the straight line with an ellipse.
    Here, we consider simply a rectangle!
    """
    pass


def Hough_peak_position(Angles,
                        verbose=1,
                        saveimages=1,
                        saveposition=1,
                        prefixname="tata",
                        key_material=14,
                        PlotOK=0,
                        arraysize=(600, 720),
                        returnXYgnomonic=0,
                        EULER=0,
                        dictmaterials=DictLT.dict_Materials):
    """
    peak in seach in Hough space representation of Laue data projected in gnomonic plane
    WARNING: need to retrieve a filter called JSM contained in ImageFilter of PIL Image module

    TODO: use numerical filter of ndimage rather than PIL. Should be faster

    TODO: to be removed soon
    """
    emax = 25
    emin = 5
    nb_of_peaks = 35 # nb of hough pixel featuring the laue pattern information in hough space

    angle_X, angle_Y, angle_Z = Angles
    nn = 1  # nn-1= number of digits in angles to put in the outputfilename

    fullname = (prefixname + str(int(nn * angle_Z)) + str(int(nn * angle_Y)) + str(int(nn * angle_X)))
    # print fullname

    if EULER == 0:
        mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])

    # PATCH to use correctly getLaueSpots() of laue6
    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    # array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=1,
                                verbose=0,
                                dictmaterials=dictmaterials)
    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    # Enlever les harmonics a ce niveau la!!
    # print TwicethetaChi
    lon = len(TwicethetaChi[0])
    # print "nb spots for harmonics filtering",lon

    toextract = np.where(np.ravel(np.triu(np.fromfunction(GT.fct_j, (lon, lon)), k=1)) != 0)[0]
    distancetolere = 0.0002
    xxtab = TwicethetaChi[0]
    yytab = TwicethetaChi[1]
    x_ind = np.where(
        abs(np.ravel(xxtab - np.reshape(xxtab, (lon, 1)))[toextract]) < distancetolere)
    y_ind = np.where(
        abs(np.ravel(yytab - np.reshape(yytab, (lon, 1)))[toextract]) < distancetolere)
    ravx_dup = toextract[x_ind]
    ravy_dup = toextract[y_ind]
    X_dup_I, X_dup_J = ravx_dup / lon, ravx_dup % lon
    Y_dup_I, Y_dup_J = ravy_dup / lon, ravy_dup % lon

    # print [X_dup_I],[X_dup_J]
    # print [Y_dup_I],[Y_dup_J]
    # toconserve=[]
    toremove = []
    for XI_elem, XJ_elem, YI_elem, YJ_elem in zip(X_dup_I, X_dup_J, Y_dup_I, Y_dup_J):
        if XI_elem == YI_elem and XJ_elem == YJ_elem:
            # print "hourra"
            # duplicates for points index XI_elem and XJ_elem
            # toconserve.append(XI_elem)
            toremove.append(XJ_elem)
        else:
            # print "bof"
            pass
    # set() allows to remove duplicates..."
    tokeep = np.array(list(set(np.arange(lon)) - set(toremove)))
    # print "to keep",tokeep
    # print "nb of removed harmonics",(lon-len(tokeep))

    # for elem in oncam_sansh[0]:
    #    print elem._indice,elem.Twicetheta/2.,elem.Chi

    # if PlotOK:
    #    Plot_Laue(emin,emax,oncam_sansh, TwicethetaChi[:,0],TwicethetaChi[:,1], 'test', oncamwoharmonics=1,Plot_Data=1, Display_label=0,What_to_plot='2thetachi',saveplotOK=0,WriteLogFile=0)

    VIPintensity = None
    xgnomon, ygnomon, = ComputeGnomon_2((xxtab[tokeep], yytab[tokeep]))

    # scatter(xgnomon,ygnomon)
    # show()
    if verbose:
        print("Computing %d selected MARCCD pixels in Gnomonic space for Hough transform"
            % len(xgnomon))
        print("Now all pixels are considered to have the same intensity")

    tagreso = 0
    if arraysize == (300, 360):
        tagreso = 1
    bigHoughcollector = ComputeHough(xgnomon, ygnomon, Intensity_table=None,
                                    stepdeg=0.5,
                                    steprho=0.002,
                                    lowresolution=tagreso)
    # argsort(ravel(bigHoughcollector))[-30:] for taking the first 30 most intense peak

    # some filtering to extract main peaks
    mike = np.ravel(bigHoughcollector)
    # mikemax=max(mike)
    # print "raw histogram",histogram(mike,range(mikemax))

    if saveimages:
        mikeraw = Image.new("L", arraysize)
        mikeraw.putdata(mike)
        mikeraw.save(fullname + "raw" + ".TIFF")

    # cutoffintensity defined as the intensity of the 200th most intense pixel
    cutoffintensity = mike[np.argsort(mike)[-200]]
    mike[mike <= cutoffintensity] = 1

    #  filter mike array
    mikeimage = Image.new("L", arraysize)
    mikeimage.putdata(mike)
    # TODO: this filter is in an old module, I guess it is a lagrangian filter size 5?
    mikeclipjsm = mikeimage.filter(ImageFilter.JSM)
    # mikeclipjsm=mikeclipjsm.filter(ImageFilter.MaxFilter) # suitable for Si ?
    if saveimages:
        mikeclipjsm.save(fullname + "clipfilt" + ".TIFF")
    # ---------------------

    #    popos = ImProc.LocalMaxima_ndimage(mike, peakVal=4,
    #                                    boxsize=5,
    #                                    central_radius=2,
    #                                    threshold=10,
    #                                    connectivity=1)

    jsmdata = np.array(list(mikeclipjsm.getdata()))
    indices_peak = np.where(jsmdata >= 32)[0]  # intensity threshold for hough pixel to be selected
    # nb of hough pixels above the intensity threshold
    nbpeaks = len(indices_peak)
    # we have enough points to select the nb_of_peaks most intense
    if nb_of_peaks <= nbpeaks:
        indices_peak = indices_peak[np.argsort(jsmdata[indices_peak])[-nb_of_peaks:]]
        nbpeaks = nb_of_peaks
    if verbose:
        print("cutoff intensity", cutoffintensity)
        print("clipped histogram", np.histogram(mike))
        print("clipped and filtered histogram", np.histogram(jsmdata))
        print("nbpeaks", nbpeaks)
        print("indice_peaks", indices_peak)

    if verbose:
        print("nb of selected peak", nbpeaks)
        # print "indices",indices_peak

    if saveposition:
        # saving the list indices
        outputname = fullname + "pick"
        filepickle = open(outputname, "w")
        pickle.dump(indices_peak, filepickle)
        filepickle.close()

    if returnXYgnomonic == 1:
        return nbpeaks, indices_peak, xgnomon, ygnomon
    else:
        return nbpeaks, indices_peak


def removeClosePoints(Twicetheta, Chi, dist_tolerance=0.0002):
    """
    remove very close spots within dist_tolerance

    dist_tolerance (in deg) is a crude criterium since angular distance are computed
    as if 2theta,chi coordinates where cartesian ones !

    NOTE: this can be used for harmonics removal (harmonics enhance too much some sinusoids and
    leads to artefact when searching zone axes by gnomonic-hough transform
    and digital image processing methods)
    """

    lon = len(Twicetheta)
    if lon != len(Chi):
        raise TypeError("Twicetheta and Chi array haven't the same length!")

    # print "nb spots for harmonics filtering",lon

    toextract = GT.indices_in_flatTriuMatrix(lon)

    x_ind = np.where(abs(np.ravel(Twicetheta - np.reshape(Twicetheta, (lon, 1)))[toextract])
        < dist_tolerance)
    y_ind = np.where(abs(np.ravel(Chi - np.reshape(Chi, (lon, 1)))[toextract]) < dist_tolerance)
    ravx_dup = toextract[x_ind]
    ravy_dup = toextract[y_ind]
    X_dup_I, X_dup_J = ravx_dup / lon, ravx_dup % lon
    Y_dup_I, Y_dup_J = ravy_dup / lon, ravy_dup % lon

    # print [X_dup_I],[X_dup_J]
    # print [Y_dup_I],[Y_dup_J]
    # toconserve=[]
    toremove = []
    for XI_elem, XJ_elem, YI_elem, YJ_elem in zip(X_dup_I, X_dup_J, Y_dup_I, Y_dup_J):
        if XI_elem == YI_elem and XJ_elem == YJ_elem:
            # print "hourra"
            # duplicates for points index XI_elem and XJ_elem
            # toconserve.append(XI_elem)
            toremove.append(XJ_elem)
        else:
            # print "bof"
            pass
    # set() allows to remove duplicates..."
    tokeep = np.array(list(set(np.arange(lon)) - set(toremove)))
    # print "to keep",tokeep
    # print "nb of removed harmonics",(lon-len(tokeep))

    TTH = Twicetheta[tokeep]
    CHI = Chi[tokeep]

    # return filtered twicetheta and chi arrays
    return TTH, CHI, tokeep


def Hough_peak_position_fast(Angles,
                            Nb=50,
                            verbose=1,
                            pos2D=0,
                            removedges=2,
                            key_material="Si",
                            returnfilterarray=0,
                            arraysize=(300, 360),
                            EULER=0,
                            blur_radius=0.5,
                            printOrientMatrix=0,
                            emax=25,
                            dictmaterials=DictLT.dict_Materials):
    """
    Simulate for 3 Euler angles Laue pattern
    peak search in Hough space representation of Laue data projected in gnomonic plane

    """
    emin = 5

    angle_X, angle_Y, angle_Z = Angles

    if EULER == 0:
        mymat = GT.fromelemangles_toMatrix(
            [angle_X * 1.0, angle_Y * 1.0, angle_Z * 1.0]
        )
    else:
        mymat = GT.fromEULERangles_toMatrix(
            [angle_X * 1.0, angle_Y * 1.0, angle_Z * 1.0]
        )

    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    if printOrientMatrix:
        print("grainparameters")
        print(grain)

    # simulation + ad hoc method for harmonics removal
    if 1:
        # array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                    CST_ENERGYKEV / emin,
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    dictmaterials=dictmaterials)

        # 2theta,chi of spot which are on camera (BUT with harmonics)
        TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1, detectordistance=70.0)

        #        print "TwicethetaChi"
        #        print len(TwicethetaChi[0])

        TTH, CHI, tokeep = removeClosePoints(TwicethetaChi[0], TwicethetaChi[1])
    #        TTH, CHI = TwicethetaChi[0], TwicethetaChi[1]

    #        print "TwicethetaChifilteredold method"
    #        print len(np.array([TTH, CHI]).T)

    #  simulation + a more accurate harmonics removal method (a bit longer than the previous one)
    else:

        #         TwicethetaChi = LAUE.get_2thetaChi_withoutHarmonics(grain, emin, emax,
        #                                detectordistance=70.,
        #                                detectordiameter=165.,
        #                                pixelsize=165 / 2048.)

        simulparameters = {}
        simulparameters["detectordiameter"] = 165.0
        simulparameters["kf_direction"] = "Z>0"
        simulparameters["detectordistance"] = 70.0
        simulparameters["pixelsize"] = 165.0 / 2048

        TwicethetaChi = LAUE.SimulateResult(
            grain, emin, emax, simulparameters, fastcompute=1, ResolutionAngstrom=False)

        #    print "TwicethetaChifiltered"
        #        print len(TwicethetaChi[0])

        TTH = TwicethetaChi[0]
        CHI = TwicethetaChi[1]

    # compute gnomonic projection

    VIPintensity = None
    xgnomon, ygnomon = ComputeGnomon_2((TTH, CHI))

    if verbose:
        print("Computing %d selected MARCCD pixels in Gnomonic space for Hough transform"
            % len(xgnomon))
        print("Now all pixels are considered to have the same intensity")

    tagreso = 0
    if arraysize == (300, 360):
        tagreso = 1

    bigHoughcollector = ComputeHough(xgnomon,
                                    ygnomon,
                                    Intensity_table=None,
                                    stepdeg=0.5,
                                    steprho=0.002,
                                    lowresolution=tagreso)

    #    popul , bins = np.histogram(bigHoughcollector)
    #    print "intensity frequency bigHoughcollector", popul
    #    print "intensity bins", bins

    # pre image processing
    if 1:
        #        minimum_number_per_zone_axis = np.round(bins[1])
        minimum_number_per_zone_axis = 2
        # #        print "thresholding from :", minimum_number_per_zone_axis
        #        bigHoughcollectorfiltered = np.where(bigHoughcollector < minimum_number_per_zone_axis,
        #                                             0, bigHoughcollector)
        #    bigHoughcollectorfiltered = enhancefilter2(bigHoughcollectorfiltered)
        bigHoughcollectorfiltered = bigHoughcollector
        bigHoughcollectorfiltered = NDI.gaussian_filter(bigHoughcollectorfiltered, (1, 0.7))
        bigHoughcollectorfiltered = np.where(bigHoughcollectorfiltered <= 1, 0, bigHoughcollectorfiltered)
    #    background = NDI.uniform_filter(bigHoughcollectorfiltered, 10)
    #        background = NDI.minimum_filter(bigHoughcollectorfiltered, 20)
    #        bigHoughcollectorfiltered -= background

    #    bigHoughcollectorfiltered = NDI.gaussian_filter(bigHoughcollector, (blur_radius, blur_radius))
    #    bigHoughcollectorfiltered = meanfilter(bigHoughcollector)
    #    bigHoughcollectorfiltered = enhancefilter2(bigHoughcollectorfiltered)
    #    bigHoughcollectorfiltered = tophatfilter(bigHoughcollectorfiltered,
    #                                            peakVal=7,
    #                                            boxsize=10,
    #                                            central_radius=1)

    #    maxint = np.amax(bigHoughcollectorfiltered)

    else:

        popul, bins = np.histogram(bigHoughcollector)
        print("intensity frequency bigHoughcollector", popul)
        print("intensity bins", bins)
        minimum_number_per_zone_axis = np.round(bins[1])
        bigHoughcollectorfiltered = np.where(bigHoughcollector < minimum_number_per_zone_axis,
                                                            0, bigHoughcollector)
        # larger integration along theta axis (slow index of bigHoughcollector
        #        bigHoughcollectorfiltered = NDI.gaussian_filter(bigHoughcollectorfiltered, sigma=(1, .1))

        bigHoughcollectorfiltered = NDI.maximum_filter(bigHoughcollectorfiltered, size=5)

    # ---   get peaks
    # most intense pixel
    if 0:

        poshighest_most = get_mostintensepeaks(bigHoughcollectorfiltered, Nb, pos2D=1,
                                                                    removedges=removedges)

        #        print "poshighest", poshighest
        #
        #        poshighest = convert1Dindices_fromcroppedarray(bigHoughcollectorfiltered.shape,
        #                                                       poshighest,
        #                                                       removedges=removedges)
        nrow, ncol = bigHoughcollectorfiltered.shape
        #    print "array_to_filter.shape", array_to_filter.shape
        poshighest = poshighest_most[:, 0] * ncol + poshighest_most[:, 1]

        if returnfilterarray == 0:
            return poshighest
        elif returnfilterarray == 1:
            return poshighest, bigHoughcollectorfiltered
        elif returnfilterarray == 2:
            return poshighest, bigHoughcollectorfiltered, bigHoughcollector
    # blob search
    else:
        # 2D center of mass of peak -------------------------
        # must find Nb most intense zone axes
        array_to_filter = bigHoughcollectorfiltered

        popul, bins = np.histogram(array_to_filter)
        #        print "intensity frequency", popul
        #        print "intensity bins", bins

        nbiter = 0
        while True:
            THRESHOLD = bins[1]
            poshighest_2D = getblobCOM(array_to_filter,
                                    threshold=THRESHOLD,
                                    connectivity=1,
                                    returnfloatmeanpos=0,
                                    removedges=2)
            if len(poshighest_2D) >= Nb:
                poshighest_2D_final = poshighest_2D[:Nb]
                break
            else:
                print("reducing the threshold")
                THRESHOLD -= 0.5
                nbiter += 1
                if nbiter > 10:
                    print("WARNING: starting threshold is really not well defined")
                    print("in Hough_peak_position_fast()")
                    poshighest = None
                    if returnfilterarray == 0:
                        return poshighest
                    elif returnfilterarray == 1:
                        return poshighest, bigHoughcollectorfiltered
                    elif returnfilterarray == 2:
                        return poshighest, bigHoughcollectorfiltered, bigHoughcollector

        #        print "poshighest_2D.len", len(poshighest_2D)
        #        print "10 highest zone axes"
        #        print poshighest_2D_final[:10]

        # conversion to 1D indices (without edges removal)
        nrow, ncol = array_to_filter.shape
        #    print "array_to_filter.shape", array_to_filter.shape
        poshighest_blob = poshighest_2D_final[:, 0] * ncol + poshighest_2D_final[:, 1]
        poshighest = poshighest_blob
        # ----------------------------------------------

        if returnfilterarray == 0:
            return poshighest
        elif returnfilterarray == 1:
            return poshighest, bigHoughcollectorfiltered
        elif returnfilterarray == 2:
            return poshighest, bigHoughcollectorfiltered, bigHoughcollector


def get_mostintensepeaks(data_array, Nb, removedges=2, pos2D=0):
    """
    return the Nb most intense pixels in data_array

    """
    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges : nbrow - removedges, removedges : nbcol - removedges]

    posbest = np.argsort(np.ravel(clipdata_array))[-Nb:]

    # return index from an 2D array
    if pos2D:
        return convertindices1Dto2D(data_array.shape, posbest, removedges=removedges)
    # return index from an 1D array
    else:
        return posbest


def tophatfilter(data_array, removedges=2, peakVal=4, boxsize=6, central_radius=3):
    """
    apply an top hat filter to array without edges

    """
    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[
        removedges : nbrow - removedges, removedges : nbcol - removedges]

    enhancedarray = ImProc.ConvolvebyKernel(
        clipdata_array, peakVal=peakVal, boxsize=boxsize, central_radius=central_radius)
    newarray = np.zeros_like(data_array)
    newarray[removedges : nbrow - removedges, removedges : nbcol - removedges] = enhancedarray

    return newarray


def meanfilter(data_array, removedges=2, n=3):
    """
    apply an top hat filter to array without edges

    """
    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges : nbrow - removedges, removedges : nbcol - removedges]
    kernel = np.ones((n, n)) / (1.0 * n ** 2)
    enhancedarray = NDI.filters.convolve(clipdata_array, kernel)

    newarray = np.zeros_like(data_array)
    newarray[removedges : nbrow - removedges, removedges : nbcol - removedges] = enhancedarray

    return newarray


def getblobCOM(data_array, threshold=0, connectivity=1, returnfloatmeanpos=0, removedges=0):
    """
    return center of mass of blob in data_array
    blobs are sorted regarding their intensity

    """

    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges : nbrow - removedges, removedges : nbcol - removedges]

    thraa = np.where(clipdata_array > threshold, 1, 0)

    if connectivity == 0:
        star = np.eye(3)
        ll, nf = NDI.label(thraa, structure=star)
    elif connectivity == 1:
        ll, nf = NDI.label(thraa, structure=np.ones((3, 3)))
    elif connectivity == 2:
        ll, nf = NDI.label(thraa, structure=np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]]))

    range_nf = np.array(np.arange(1, nf + 1), dtype=np.int16)

    meanpos = np.array(
        NDI.measurements.center_of_mass(thraa, ll, range_nf), dtype=np.float32)

    #    maximumvalues = np.array(NDI.measurements.maximum(thraa, ll, range_nf),
    #                                                                dtype=np.float32)

    sumofvalues = np.array(NDI.measurements.sum(thraa, ll, range_nf), dtype=np.float32)

    #    meanofvalues = np.array(NDI.measurements.sum(thraa, ll, np.arange(1, nf + 1)),
    #                                                                dtype=np.float32)

    criterium = sumofvalues

    SortedPos = np.take(meanpos, np.argsort(criterium)[::-1], axis=0)

    # take into account edges removal
    SortedPos += removedges * np.array([1, 1])

    if returnfloatmeanpos:
        return SortedPos
    else:
        return np.round(SortedPos)


def enhancefilter2(data_array, removedges=2):
    """
    apply a filter to array without edges
    """
    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges : nbrow - removedges, removedges : nbcol - removedges]

    #    kernel = np.array([[0, 1, 1, 1, 0],
    #                       [-2, 1, 1, 1, -2],
    #                       [-3, -1, 4, -1, -3],
    #                       [-2, 1, 1, 1, -2],
    #                       [0, 1, 1, 1, 0]])
    #
    #    kernel2 = np.array([[1, 1, 1, 1, 1],
    #                       [0, 1, 1, 1, 0],
    #                       [-2, -1, 20, -1, -2],
    #                       [0, 1, 1, 1, 0],
    #                       [1, 1, 1, 1, 1]])
    #
    #    kernel3 = np.array([[1, 1, 1, ],
    #                       [-1, 6, -1],
    #                       [1, 1, 1]])

    kernel4 = np.array([[1, 1, 1, 1, 1],
                        [-1, 2, 2, 2, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, 2, 2, 2, -1],
                        [1, 1, 1, 1, 1]])

    enhancedarray = NDI.filters.convolve(clipdata_array, kernel4)
    newarray = np.zeros_like(data_array)
    newarray[removedges : nbrow - removedges, removedges : nbcol - removedges] = enhancedarray

    return newarray


def convertindices1Dto2D(data_array_shape, indices1D, removedges=0):
    """
    convert indices from 1D raveled array
    coming from an array ((data_array_shape) which edges have been removed

    in corresponding indices of 2d array

    example : removeedges = 1 from [[-,-,-,-,-],
                                [-,0,1,2,-],
                                [-,3,4,5,-],
                                [-,6,7,8,-],
                                [-,-,-,-,-]]
    and 1D indices raveled [[1,2,3,4,5],
                            [6,7,8,9,10],
                            [11,12,13,14,15],
                            [16,17,18,19,20],
                            [21,22,23,24,25]]

    find the correspondance 0,1,2,3 etc. to 7,8,9,12
    """
    nbrow, nbcol = data_array_shape

    row_slow = indices1D / (nbcol - 2 * removedges) + removedges
    col_fast = indices1D % (nbcol - 2 * removedges) + removedges

    return np.array([row_slow, col_fast]).T


def convertindices1Dto3D(data_array_shape, indices1D):
    """
    convert indices from 1D raveled array
    coming from an array ((data_array_shape) which edges have been removed
    
    in corresponding indices of 2d array
    example:

    from indices in [[[1,2,3,4],[4,5,6,7],[8,9,10,11]],[[12,13,14,15],[16,17,18,19],[20,21,22,23]]]
    return i,j,k:

    4 -> 0,1,0
    13 -> 1,0,1
    23 -> 1,2,3
    """
    nbrow, nbcol, nbdim3 = data_array_shape

    row_slow = indices1D / nbdim3 / nbcol
    col_fast = indices1D / nbdim3 % nbcol
    lastindex_veryfast = indices1D % nbdim3

    return np.array([row_slow, col_fast, lastindex_veryfast]).T


def convert1Dindices_fromcroppedarray(data_array_shape, indices1D_croppedarray, removedges=0):
    """
    convert 1D indices from raveled array
    coming from an array ((data_array_shape) ) which edges have been removed

    in corresponding indices of 1d array
    TODO: give example
    """
    nbrow, nbcol = data_array_shape

    offset = removedges * (nbcol + 1)

    original_index = (offset
        + indices1D_croppedarray / (nbcol - 2 * removedges)
        + indices1D_croppedarray % (nbcol - 2 * removedges))

    return original_index


def StickLabel_on_exp_peaks(Angles,
                        twicetheta_data,
                        chi_data,
                        angulartolerance,
                        emax,
                        verbose=1,
                        key_material=14,
                        arraysize=(600, 720),
                        EULER=1,
                        dictmaterials=DictLT.dict_Materials):

    """
    it seems that this function aims at linking 
    experimental spot to simulated ones
    (given the 3 orientation angles)

    TODO: really useful?
    """
    emin = 5
    angle_X, angle_Y, angle_Z = Angles

    if EULER == 0:
        mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])

    # PATCH to use correctly getLaueSpots() of laue6
    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    # fastcompute=0 => array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=0,
                                verbose=0,
                                dictmaterials=dictmaterials)

    # fastcompute=0 => result is list of spot instances which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=0)

    theta_and_chi_theo = np.array([[elem.Twicetheta / 2.0, elem.Chi] for elem in TwicethetaChi[0]])
    theta_and_chi_exp = np.array([twicetheta_data / 2.0, chi_data]).T

    # from the experimental point of view (how far are the theoritical point from 1 one exp point)

    table_distance = GT.calculdist_from_thetachi(theta_and_chi_theo, theta_and_chi_exp)
    prox_index = getArgmin(table_distance)

    # creation of label allocation with angular vicinity condition:
    peak_dict_ind = {}  # key is the exp spot index, value is the theo spot index
    peak_dict_dist = {}  # idem, value is the theo spot angular dist (from exp spot)
    nb_of_exp_peaks = len(theta_and_chi_exp)
    if verbose:
        print("nb of experimenal spots to labelled", nb_of_exp_peaks)
    for k in np.arange(nb_of_exp_peaks):
        dist_from_nearest = table_distance[k][prox_index[k]]
        if dist_from_nearest <= angulartolerance:  # angular tolerance for labelling
            peak_dict_ind[k] = prox_index[k]
            peak_dict_dist[k] = dist_from_nearest

    # selection of the nearest of the neighbors
    p_d_i = {}  # dictionnary, key is the theo index, value is the exp index
    p_d_d = {}  # idem but with value= distance
    for cle_theo in peak_dict_ind.values():  # cle_theo is an theoritical spot index
        # index of exp spots which have the same neighbouring theo spot (whose index is cle_theo)
        neighbors = np.where(prox_index == cle_theo)[0]

        list_dist_neighbors = []
        list_ind_neighbors = []
        # print "experiment spot ",cle_theo," has theoritical spots",neighbors
        # print "with distances"

        # Now do some selection in order not to do labelling mistakes
        for ind_exp in neighbors:  # index of exp spots
            if (ind_exp in peak_dict_dist):  # key is defined if previous angular tolerance for labelling was True

                # list of all theoritical neighboring spots and distances
                list_dist_neighbors.append(peak_dict_dist[ind_exp])
                list_ind_neighbors.append(ind_exp)
                # p_d_i[cle_theo]=list_ind_neighbors
                # p_d_d[cle_theo]=list_dist_neighbors

                # sort for selecting only the nearest theoritical spot
                Nearest_ind = list_ind_neighbors[np.argmin(np.array(list_dist_neighbors))]
                Nearest_dist = min(list_dist_neighbors)
                # p_d_i[cle_theo]=Nearest_ind
                # p_d_d[cle_theo]=Nearest_dist

        # selecting only exp. spots with only 1 neighboring theo. spots
        # (case of two neighboring theo spots in angular resolution range from exp. spots is excluded)
        if len(list_dist_neighbors) == 1:
            p_d_i[cle_theo] = list_ind_neighbors[0]
            p_d_d[cle_theo] = list_dist_neighbors[0]

        # if cle_theo<10: print cle_theo,"dffd",list_dist_neighbors
    # print "p_d_i",p_d_i
    # print "p_d_d",p_d_d
    if verbose:
        print("nb of labelled exp. spots ", len(p_d_d))

    # from the remaining theor. point of view (theoritical points must be far away to avoid ambiguity)
    theo_spot_index_list = np.array(list(p_d_i.keys()))
    # print "theo_spot_index_list",theo_spot_index_list
    theta_and_chi_theo_Select = theta_and_chi_theo[theo_spot_index_list]
    table_distance_T = GT.calculdist_from_thetachi(theta_and_chi_theo_Select, theta_and_chi_theo_Select)
    prox_index_T = getArgmin(table_distance_T)
    # print prox_index_T
    # print "theta_and_chi_theo_Select[23]",theta_and_chi_theo_Select[23]
    # print table_distance_T[23:,23:]
    aa, bb = np.where(
        np.logical_and(table_distance_T <= 1.5 * angulartolerance, table_distance_T != 0.0))

    theo_to_remove = np.array(list(set(aa[aa < bb]).union(set(bb[aa < bb]))))  # to remove some spurious numerical imprecision: e.g. finding angle between the same point  !=0 !!!
    if verbose:
        print("index to remove in theo_spot_index_list", theo_to_remove)
        print("True index of theo spots to be removed",
            theo_spot_index_list[theo_to_remove])
    # => remove from dictionnary, see at the end

    # from the remaining exp. point of view (exp. points must be far away to avoid ambiguity)
    exp_spot_index_list = np.array(list(p_d_i.values()))
    theta_and_chi_exp_Select = np.array(theta_and_chi_exp)[exp_spot_index_list]
    # print "theta_and_chi_exp_Select[5]",theta_and_chi_exp_Select[5]
    table_distance_E = GT.calculdist_from_thetachi(np.array(theta_and_chi_exp_Select),
                                            np.array(theta_and_chi_exp_Select))
    prox_index_E = getArgmin(table_distance_E)
    # print prox_index_E
    # print table_distance_E[5:,5:]
    cc, dd = np.where(
        np.logical_and(table_distance_E <= 1.5 * angulartolerance, table_distance_E != 0.0))
    exp_to_remove = np.array(list(set(cc[cc < dd]).union(set(dd[cc < dd]))))  # to remove some spurious numerical imprecision: e.g. finding angle between the same point  !=0 !!!
    if verbose:
        print("index to remove in exp_spot_index_list", exp_to_remove)
        print(
            "True index of exp spots to be removed", exp_spot_index_list[exp_to_remove])
    # => remove from dictionnary, see just below

    key_theo_to_remove = []
    for dict_val in exp_spot_index_list[exp_to_remove]:
        something = find_key(p_d_i, dict_val)
        if something is not None:
            key_theo_to_remove.append(something)

    whole_couple_the_exp_to_remove = set(key_theo_to_remove).union(set(theo_spot_index_list[theo_to_remove]))
    if verbose:
        print("key_theo_to_remove", whole_couple_the_exp_to_remove)

    for theo in whole_couple_the_exp_to_remove:
        del p_d_i[theo]
        del p_d_d[theo]
    # p_d_i is a dict: key = theo index, value = exp index
    if verbose:
        print("Final number of spots pair (theo,exp) ", len(p_d_d))
    return (theta_and_chi_exp, theta_and_chi_theo, peak_dict_ind, prox_index, p_d_d, p_d_i)


# --- --------------- IMAGE MATCHING INDEXING

# --- --very old method

def Houghcorrel(exp_1d_data, simul_database):
    """ from an 1D experimental array
    for each element, we calculate the sum of intensity from the experimental Hough 1d array
    according to these "nb_of_indices" indices

    Next we create a 2d array (row,col=angZ,angX) where each element is couple (pos=best(angX),intensity) mentionned above.
    We may locate then in this 2d intensity angular, the regions of (angX,angY,angZ) where exp and simulated patterns fit
    TODO: to clarify
    TODO: at this step provide a 3D array instead of a 2D array with corresponding argument
    where value is the highest for each element
    """

    # for 1 deg step resolution EULER angles; nd3 for angX
    # (in this axis, angle is scanned for selecting only the max)

    nd1, nd2, nd3 = 90, 180, 90  # angZ, angY, angX

    print("in Houghcorrel()")
    print("exp_1d_data.shape", exp_1d_data.shape)
    # shape of simul_database must be (nb of orientations, nb_of_indices)
    print("simul_database.shape", simul_database.shape)

    # integration of all intensities picked up in exp_1d_data
    listint = np.reshape(np.sum(exp_1d_data[simul_database], axis=1), (nd1 * nd2, nd3))

    # shape of listint must be (90*180,90)
    #    print "listint.shape", listint.shape

    # "angX" argmax index for the 90*180 elements
    listarg = np.argmax(listint, axis=1)
    #    print "listarg", listarg

    # matching intensity
    listmax = np.amax(listint, axis=1)
    tab2dmaxint = np.reshape(listmax, (nd1, nd2))

    return tab2dmaxint, listarg


def findGrain_in_orientSpace(CorrelIntensTab, nbfirstpt):
    """ from a 2d intensity table
    locate local maxima in correlation intensity space

    # TODO: replace by a robust local maxima method in other lauetools module
    peaksearch with ndimage. measurements (see readmccd.py)?

    TODO: to be removed soon
    """
    n1, n2 = np.shape(CorrelIntensTab)
    _rara = np.ravel(CorrelIntensTab)
    ind_i = np.argmax(CorrelIntensTab, axis=1)  # find max in one direction for each value of i
    ind_j = np.argmax(CorrelIntensTab, axis=0)  # find max in one direction for each value of j
    gluck = np.argsort(np.ravel(_rara))[-nbfirstpt:][::-1]  # indices in descending order of the first most intense

    # print "first %d orientations"%nbfirstpt
    # print gluck

    diffgluck_i = gluck - np.reshape(gluck, (nbfirstpt, 1))  # difference at i cst
    pos1 = np.where(gluck - np.reshape(gluck, (nbfirstpt, 1)) == 1)  # next right or left distance indice 1

    (toeleminate_1, toeleminate_2, toeleminate_3, toeleminate_4) = (np.array([]),
                                                                    np.array([]),
                                                                    np.array([]),
                                                                    np.array([]))
    if len(pos1[0]) > 0:
        # print "pos1",pos1
        toeleminate_1 = gluck[np.amax(np.array(pos1))]
        # print "elimi h",toeleminate_1

    # ----too long to remove along column ..
    # TODO: what is the 23 !!
    newgluck = (gluck % 23) * 23 + gluck / 23  # indexation by permuting col and row
    # print newgluck
    # diffgluck_j = newgluck - np.reshape(newgluck, (nbfirstpt, 1)) # difference at j cst
    pos2 = np.where(
        newgluck - np.reshape(newgluck, (nbfirstpt, 1)) == 1)  # next top or bottom
    if len(pos2[0]) > 0:
        # print "pos2",pos2
        toelem_2 = np.amax(np.array(pos2))
        # print "raw elim index",toelem_2
        # print "raw abs index",newgluck[np.array(  toelem_2)]
        toeleminate_2 = (newgluck[toelem_2] % 23) * 23 + newgluck[toelem_2] / 23
        # print "elimi v good indice",toeleminate_2

    pos22 = np.where(
        gluck - np.reshape(gluck, (nbfirstpt, 1)) == 22)  # close left and top distance indice 22 et inv.
    pos24 = np.where(
        gluck - np.reshape(gluck, (nbfirstpt, 1)) == 24)  # close left and top distance indice 24 et inv.

    if len(pos22[0]) > 0:
        # print "pos22",pos22
        toeleminate_3 = gluck[np.amax(np.array(pos22))]
        # print "elimi diag1",toeleminate_3

    if len(pos24[0]) > 0:
        # print "pos24",pos24
        toeleminate_4 = gluck[np.amax(np.array(pos24))]
        # print "elimi diag1",toeleminate_4

    torem = set(
        np.concatenate((toeleminate_1, toeleminate_2, toeleminate_3, toeleminate_4)))

    if len(torem) > 0:
        print("removing some orientations")
        # print torem
        print("Remaining orientations")
        remindices = np.array(list(set(gluck) - torem))
        remelem = []
        for k in list(range(len(gluck))):
            if gluck[k] in remindices:
                remelem.append(gluck[k])
        return remelem
    else:
        print("No contiguous orientations found")
        return gluck


def convert_Orientindex_to_angles(array_of_index, dimensions, resolution="high"):
    """ 
    dimensions is a tuple (dim1,dim2)
    """
    if resolution == "low":
        nz, ny = dimensions

        _angZ = array_of_index / nz
        _angY = array_of_index % nz

        R_Z = np.arange(0.0, 45.0, 2.0)
        R_Y = np.arange(0.0, 45.0, 2.0)
    elif resolution == "high":
        nz, ny = 90, 90

        _angZ = array_of_index / nz
        _angY = array_of_index % nz

        R_Z = np.arange(0.0, 90.0, 1.0)
        R_Y = np.arange(0.0, 90.0, 1.0)

    return R_Z[_angZ], R_Y[_angY]


def best_orientations(Database, bigHoughcollector, nb_orientations=20):
    """
    from Database and experimental data (bigHoughcollector)
    return array of the "nb_orientations" best orientations
    (1 Orientation = 3 angles)

    """
    print("bigHoughcollector.shape", bigHoughcollector.shape)
    #    rara, argrara = Houghcorrel(np.ravel(bigHoughcollector), Database)
    rara, argrara = Houghcorrel(np.ravel(np.reshape(bigHoughcollector, (300, 360))), Database)

    print("rara.shape", rara.shape)
    print("argrara.shape", argrara.shape)

    bestindices = np.array(findGrain_in_orientSpace(rara, nb_orientations))
    print("bestindices in best_orientations", bestindices)

    best_Az, best_Ay = convert_Orientindex_to_angles(bestindices, (90, 90))
    best_Ax = np.arange(0.0, 90.0, 1.0)[argrara[bestindices]]

    print(best_Ax, "\n", best_Ay, "\n", best_Az)

    return np.transpose(np.array([best_Ax, best_Ay, best_Az]))


def give_best_orientations(prefixname,
                            file_index,
                            nbofpeaks_max,
                            _fromfiledata,
                            col_I=1,
                            nb_orientations=20,
                            dirname=".",
                            plotHough=0):
    """
    read peak list and by image matching propose a list of orientation for peaks indexation

    High resolution (1 deg step)
    _fromfiledata is the databank: huge array containing indices to pick up in transformed data p
    """
    filename_data = prefixname + IOimage.stringint(file_index, 4) + ".cor"
    print("filename_data", filename_data)
    #    nblines_to_skip=1 # nb lines of header
    #    col_2theta=0
    #    col_chi=1
    # col_I=1 # Intensity  column

    # nbofpeaks_max=500 # for recognition

    mydata = IOLT.ReadASCIIfile(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]), listofselectedpts, -1)

    print(" ******   Finding best orientations ****************")
    print("Raw data in %s have %d spots" % (filename_data, len(data_theta)))
    print("Number of selected (or not recongnised) spots from raw data: ",
        len(listofselectedpts))
    print("Looking only from these last spots, only the first %d spots" % nbmax)
    print("Last spot index probed in rawdata: %d" % listofselectedpts[nbmax - 1])
    print("\n")

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(dataselected)

    #    print len(gnomonx)

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(
        gnomonx, gnomony, Intensity_table=None, lowresolution=1)
    print(bigHoughcollector)
    print(np.shape(bigHoughcollector))
    print(np.shape(_fromfiledata[100]))  # nb of fingerprint indices

    if plotHough:
        p.imshow(bigHoughcollector, interpolation="nearest")
        p.show()

    AAA = best_orientations(
        _fromfiledata, bigHoughcollector, nb_orientations=nb_orientations)

    return np.transpose(AAA), dataselected, [gnomonx, gnomony]


# --- -old method


def findGrain_in_orientSpace_new(CorrelIntensTab, nbfirstpt):
    """ from a 2d intensity table
    locate local maxima in correlation intensity space
    """
    rankN = 60
    #
    #    maxint = np.amax(CorrelIntensTab)
    #
    #    posintense = np.argsort(np.ravel(CorrelIntensTab))[::-1][:rankN]
    #    pos_ij = convertindices1Dto2D(CorrelIntensTab.shape, posintense)
    #
    #    intensity_rank0 = CorrelIntensTab[pos_ij[0][0], pos_ij[0][1]]
    #    intensity_rankN = CorrelIntensTab[pos_ij[-1][0], pos_ij[-1][1]]
    #
    #    print "maxint", maxint
    #    print "intensity_rank0", intensity_rank0
    #    print "intensity_rankN", intensity_rankN

    #    CorrelIntensTab = np.where(CorrelIntensTab < intensity_rankN, 0, CorrelIntensTab - intensity_rankN)

    #    print np.histogram(CorrelIntensTab)

    print("in findGrain_in_orientSpace_new() --------------------")

    CorrelIntensTab = NDI.filters.gaussian_filter(CorrelIntensTab, sigma=(1.0, 1.0))
    print("histogram(convolvedCorrelIntensTab)")
    print(np.histogram(CorrelIntensTab))

    maxint = np.amax(CorrelIntensTab)

    posintense = np.argsort(np.ravel(CorrelIntensTab))[::-1][:rankN]
    pos_ij = convertindices1Dto2D(CorrelIntensTab.shape, posintense)

    intensity_rank0 = CorrelIntensTab[pos_ij[0][0], pos_ij[0][1]]
    intensity_rankN = CorrelIntensTab[pos_ij[-1][0], pos_ij[-1][1]]

    print("maxint", maxint)
    print("intensity_rank0", intensity_rank0)
    print("intensity_rankN", intensity_rankN)

    thraa = np.where(CorrelIntensTab > intensity_rankN, CorrelIntensTab, 0)

    maxtab = NDI.filters.maximum_filter(thraa, size=3)
    ll, nf = NDI.label(maxtab)

    maxpos = np.array(NDI.measurements.maximum_position(maxtab, ll, np.arange(1, nf + 1)),
        dtype=np.float32)

    #    meanvalues = np.array(NDI.measurements.sum(thraa,
    #                                                    ll,
    #                                                    np.arange(1, nf + 1)),
    #                                                    dtype=np.float32)

    meanvalues2 = np.array(NDI.measurements.mean(thraa, ll, np.arange(1, nf + 1)), dtype=np.float32)

    maximumvalues = np.array(NDI.measurements.maximum(CorrelIntensTab, ll, np.arange(1, nf + 1)),
        dtype=np.float32)

    minimumvalues = np.array(NDI.measurements.minimum(CorrelIntensTab, ll, np.arange(1, nf + 1)),
        dtype=np.float32)

    #    std_values = np.array(NDI.measurements.standard_deviation(CorrelIntensTab,
    #                                                      ll,
    #                                                      np.arange(1, nf + 1)),
    #                                                      dtype=np.float32)

    # position variable
    var_pos = maxpos

    # intensity criterium
    var_int = maximumvalues

    SortedOrients = np.take(var_pos, np.argsort(var_int)[::-1], axis=0)

    rpos = np.round(var_pos)
    rSortedOrients = np.round(SortedOrients)
    print("position", var_pos)
    print("GrainOrient :", rpos)
    print("intensity criterium", var_int)
    print("meanvalues2", meanvalues2)
    print("peak_to_valley", maximumvalues - minimumvalues)
    print("SortedOrients", rSortedOrients)

    #    print "in findGrain_in_orientSpace_new() --------------------"
    #
    #    convolvedCorrelIntensTab = ImProc.ConvolvebyKernel(CorrelIntensTab,
    #                        peakVal=3,
    #                        boxsize=4,
    #                        central_radius=2)
    #    print "histogram(convolvedCorrelIntensTab)"
    #    print np.histogram(convolvedCorrelIntensTab)
    #
    #    maxint = np.amax(convolvedCorrelIntensTab)
    #
    #    posintense = np.argsort(np.ravel(convolvedCorrelIntensTab))[::-1][:rankN]
    #    pos_ij = convertindices1Dto2D(convolvedCorrelIntensTab.shape, posintense)
    #
    #    intensity_rank0 = convolvedCorrelIntensTab[pos_ij[0][0], pos_ij[0][1]]
    #    intensity_rankN = convolvedCorrelIntensTab[pos_ij[-1][0], pos_ij[-1][1]]
    #
    #    print "maxint", maxint
    #    print "intensity_rank0", intensity_rank0
    #    print "intensity_rankN", intensity_rankN
    #
    #    thraa = np.where(convolvedCorrelIntensTab > intensity_rankN, convolvedCorrelIntensTab , 0)
    #

    # method for looking for large blobs
    #    thraa = np.where(convolvedCorrelIntensTab > intensity_rankN, 1 , 0)
    #    connectivity = 0
    #    if connectivity == 0:
    #        star = np.eye(3)
    #        ll, nf = NDI.label(thraa, structure=star)
    #    elif connectivity == 1:
    #        ll, nf = NDI.label(thraa, structure=np.ones((3, 3)))
    #    elif connectivity == 2:
    #        star = np.eye(3)
    #        ll, nf = NDI.label(thraa, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))
    #
    # #    meanpos = np.array(NDI.measurements.center_of_mass(thraa,
    # #                                                        ll,
    # #                                                        np.arange(1, nf + 1)),
    # #                                                        dtype=np.float32)
    #    print "ll", np.transpose(ll)[8:15, 40:60]
    #    maxpos = np.array(NDI.measurements.maximum_position(thraa,
    #                                                        ll,
    #                                                        np.arange(1, nf + 1)),
    #                                                        dtype=np.float32)
    #
    #    maximumvalues = np.array(NDI.measurements.maximum(CorrelIntensTab,
    #                                                      ll,
    #                                                      np.arange(1, nf + 1)),
    #                                                      dtype=np.float32)
    #    #position variable
    #    var_pos = maxpos
    #
    #    SortedOrients = np.take(var_pos, np.argsort(maximumvalues)[::-1], axis=0)
    #
    #    rpos = np.round(var_pos)
    #    rSortedOrients = np.round(SortedOrients)
    #    print "meanpos", var_pos
    #    print "GrainOrient :", rpos
    #    print "maximumvalues", maximumvalues
    #    print "SortedOrients", rSortedOrients

    return rSortedOrients


def best_orientations_new(Database, bigHoughcollector, nb_orientations=20):
    """
    from Database and experimental data (bigHoughcollector)
    return array of the "nb_orientations" best orientations
    (1 Orientation = 3 angles)

    """
    #    print "bigHoughcollector.shape", bigHoughcollector.shape
    #    print "Database.shape", Database.shape

    #    rara, argrara = Houghcorrel(np.ravel(bigHoughcollector), Database)
    rara, argrara = Houghcorrel(np.ravel(np.reshape(bigHoughcollector, (300, 360))), Database)

    N3, N2 = rara.shape

    #    print "rara.shape", rara.shape
    #    print "argrara.shape", argrara.shape

    bestindices = np.array(findGrain_in_orientSpace_new(rara, nb_orientations), dtype=np.int16)

    # TODO: generalize to other angles sampling
    #    best_Az, best_Ay = convert_Orientindex_to_angles(bestindices, (90, 90))

    best_Az, best_Ay = bestindices.T  # indices are directly angles

    #    best_Ax = np.arange(0., 90., 1.)[argrara[bestindices]]
    best_Ax = argrara[bestindices[:, 0] * N2 + bestindices[:, 1]]

    bestAngles = np.array([best_Ax, best_Ay, best_Az]).T

    print("best angles")
    print(bestAngles)

    return bestAngles


def readnselectdata(filename_data, nbofpeaks_max, dirname=None, verbose=0):
    """
    return  3 arrays of 2theta , chi , intensity data of experimental spots in file

    TODO place in IOLT module...
    """

    if dirname == None:
        dirname = os.curdir

    mydata = IOLT.readfile_cor(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata[1], mydata[2], mydata[5]

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)

    # selection of points in the three arrays in first argument
    TwiceTheta_Chi_Int, nbmax = IOLT.createselecteddata(
        (data_theta * 2, data_chi, data_I), listofselectedpts, -1)

    if verbose:
        print(" ******   Finding best orientations ****************")
        print("Raw data in %s have %d spots" % (filename_data, len(data_theta)))
        print("Number of selected (or not recongnised) spots from raw data: ",
            len(listofselectedpts))
        print("Looking only from these last spots, only the first %d spots" % nbmax)
        print("Last spot index probed in rawdata: %d" % listofselectedpts[nbmax - 1])
        print("\n")

    return TwiceTheta_Chi_Int


def give_best_orientations_new(filename_data,
                            nbofpeaks_max,
                            DataBase_array,
                            tolerancedistance=0,
                            maxindex=40,
                            dirname=None,
                            plotHough=0,
                            Hough_init_sigmas=(1, 0.7),
                            Hough_init_Threshold=1,
                            useintensities=0,
                            rank_n=20,
                            sigmagaussian=(1.0, 1.0, 1.0),
                            addnoise=0,
                            verbose=0):
    """
    from peak list, index by image matching propose a list of orientation for peaks indexation

    High resolution (1 deg step)
    DataBase_array is the databank: huge array containing indices to pick up in transformed data p
    """
    TwiceTheta_Chi_Int = readnselectdata(filename_data, nbofpeaks_max,
                                            dirname=dirname, verbose=verbose)

    if addnoise:
        amplitude = 0.1
        noisefraction = 0.5

        nb_spots = len(TwiceTheta_Chi_Int[0])

        modified_spots = np.arange(nb_spots)

        np.random.shuffle(modified_spots)

        ind_noisy = modified_spots[: int(noisefraction * nb_spots)]

        tomodify = True * np.ones(nb_spots)

        np.put(tomodify, ind_noisy, False)

        tomodify = np.array(tomodify, dtype=np.bool)

        print("tomodify", tomodify)

        tomodify = np.tile(tomodify, 3).reshape((3, len(tomodify)))

        # add some noise in 2theta chi and intensity (crude manner):
        highlimit = amplitude / 2.0
        lowlimit = -amplitude / 2.0
        noise = (highlimit - lowlimit) * np.random.random((3, nb_spots)) + lowlimit

        noise = np.where(tomodify, noise, 0)

        print("noise", noise)

        TwiceTheta_Chi_Int += noise

    return getOrientationfromDatabase(TwiceTheta_Chi_Int,
                                        DataBase_array,
                                        maxindex=maxindex,
                                        plotHough=plotHough,
                                        rank_n=rank_n,
                                        sigmagaussian=sigmagaussian,
                                        verbose=verbose,
                                        Hough_init_sigmas=Hough_init_sigmas,
                                        Hough_init_Threshold=Hough_init_Threshold,
                                        useintensities=useintensities,
                                        tolerancedistance=tolerancedistance)


def getOrientationfromDatabase(TwiceTheta_Chi,
                                DataBase_array,
                                maxindex=40,
                                plotHough=0,
                                rank_n=20,
                                sigmagaussian=(1.0, 1.0, 1.0),
                                verbose=0,
                                Hough_init_sigmas=(1, 0.7),
                                Hough_init_Threshold=1,
                                useintensities=0,
                                tolerancedistance=0):
    """
    from peak list, index by image matching propose a list of orientation for peaks indexation

    TwiceTheta_Chi         : arrays of exp. data (TTH, CHI)   2theta and Chi angles of kf 

    High resolution (1 deg step)
    DataBase_array is the databank: huge array containing indices to pick up in transformed data p
    """

    if tolerancedistance:
        print("TwiceTheta_Chi.shape", TwiceTheta_Chi.shape)
        TTHkept, CHIkept, tokeep = GT.removeClosePoints_2(TwiceTheta_Chi[0], TwiceTheta_Chi[1],                                                                         dist_tolerance=tolerancedistance)

        Intensitykept = TwiceTheta_Chi[2][tokeep]

        TwiceTheta_Chi = np.array([TTHkept, CHIkept, Intensitykept])

        print("TwiceTheta_Chi.shape after close points removal")
        print(TwiceTheta_Chi.shape)

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(TwiceTheta_Chi[:2])

    if useintensities:
        Intensity_table = TwiceTheta_Chi[2]
    else:
        Intensity_table = None

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(gnomonx, gnomony, Intensity_table=Intensity_table, lowresolution=1)

    if verbose:
        print("histogram bigHoughcollector", np.histogram(bigHoughcollector))
        #    print bigHoughcollector
        print("bigHoughcollector.shape", bigHoughcollector.shape)
        print("nb of indices in pattern fingerprint", DataBase_array[100].shape)  # nb of fingerprint indices

    # experimental data pre processing
    #    minimum_number_per_zone_axis = 2
    #
    #    bigHoughcollectorfiltered = bigHoughcollector

    if plotHough:
        plottab(bigHoughcollector)

    if Hough_init_Threshold == -1:
        bigHoughcollector = NDI.maximum_filter(bigHoughcollector, size=3)

    elif Hough_init_Threshold == -2:
        bigHoughcollector = NDI.median_filter(bigHoughcollector, size=5)

    else:
        bigHoughcollector = NDI.gaussian_filter(bigHoughcollector, Hough_init_sigmas)
        bigHoughcollector = np.where(bigHoughcollector <= Hough_init_Threshold, 0, bigHoughcollector)

    #    bigHoughcollector = np.where(bigHoughcollector < 3, 0, bigHoughcollector)

    if plotHough:
        plottab(bigHoughcollector)

    AAA = best_orientations_new_3D(DataBase_array,
                                bigHoughcollector,
                                maxindex=maxindex,
                                rank_n=rank_n,
                                sigmagaussian=sigmagaussian)

    if AAA is None:
        return None, TwiceTheta_Chi, [gnomonx, gnomony]

    return np.transpose(AAA), TwiceTheta_Chi, [gnomonx, gnomony]


def Houghcorrel_3D(exp_1d_data, database_array, maxindex=40, verbose=0):
    """ Calculate correlation intensity from experimental "exp_1d_data"
    and image matchingdatabase "database_array"
    """

    # for 1 deg step resolution EULER angles; nd3 for angX
    # (in this axis, angle is scanned for selecting only the max)

    nd1, nd2, nd3 = 90, 180, 90  # angZ, angY, angX

    if maxindex > 40:
        raise ValueError("Database contains only 40 indices and not %s" % str(maxindex))

    if verbose:
        print("in Houghcorrel_3D()")
        print("exp_1d_data.shape", exp_1d_data.shape)
        # shape of database_array must be (nb of orientations, nb_of_indices)
        print("database_array.shape", database_array.shape)

    # integration of all intensities picked up in exp_1d_data

    correl3D = np.reshape(np.sum(exp_1d_data[database_array[:, :maxindex]], axis=1), (nd1, nd2, nd3))

    # shape of listint must be (90,180,90)
    #    print "listint.shape", correl3D.shape

    return correl3D


def findGrain_in_orientSpace_new_3D(CorrelIntensTab, rank_n=20, sigmagaussian=(1.0, 1.0, 1), verbose=0):
    """ from a 2d intensity table
    locate local maxima in correlation intensity space

    CorrelIntensTab    :  array of sum of correlation intensity for all orientations

    rank_n        : number of pixel to threshold intensity CorrelIntensTab
                    (if too large, blobs in 3D orientation space will large and deformed)
                    (if too small, only one blob will be found)

    sigmagaussian    : 3D gaussian averaging sigma values to smooth
                        the intensity of CorrelIntensTab

    """
    RANK_N = rank_n
    SIGMA_FOR_GAUSSIAN = sigmagaussian

    CorrelIntensTab = NDI.filters.gaussian_filter(CorrelIntensTab, sigma=SIGMA_FOR_GAUSSIAN)

    if verbose:
        print("in findGrain_in_orientSpace_new_3D() --------------------")
        print("histogram(convolvedCorrelIntensTab)")
        print(np.histogram(CorrelIntensTab))
        print("CorrelIntensTab.shape", CorrelIntensTab.shape)

    posintense = np.argsort(np.ravel(CorrelIntensTab))[::-1][:RANK_N]

    pos_ij = convertindices1Dto3D(CorrelIntensTab.shape, posintense)

    intensity_rank0 = CorrelIntensTab[pos_ij[0][0], pos_ij[0][1], pos_ij[0][2]]
    intensity_rankN = CorrelIntensTab[pos_ij[-1][0], pos_ij[-1][1], pos_ij[-1][2]]

    if verbose:
        print("RANK_N  :", RANK_N)
        print("intensity_rank0", intensity_rank0)
        print("intensity_rankN", intensity_rankN)

    thraa = np.where(CorrelIntensTab > intensity_rankN, CorrelIntensTab, 0)

    maxtab = NDI.filters.maximum_filter(thraa, size=3)
    ll, nf = NDI.label(maxtab)

    maxpos = np.array(NDI.measurements.maximum_position(maxtab, ll, np.arange(1, nf + 1)),
        dtype=np.float32)

    meanpos = np.array(NDI.measurements.center_of_mass(maxtab, ll, np.arange(1, nf + 1)),
        dtype=np.float32)

    #

    #    meanvalues = np.array(NDI.measurements.sum(thraa,
    #                                                    ll,
    #                                                    np.arange(1, nf + 1)),
    #                                                    dtype=np.float32)
    #
    #    meanvalues2 = np.array(NDI.measurements.mean(thraa,
    #                                                    ll,
    #                                                    np.arange(1, nf + 1)),
    #                                                    dtype=np.float32)

    maximumvalues = np.array(NDI.measurements.maximum(CorrelIntensTab, ll, np.arange(1, nf + 1)),
        dtype=np.float32)

    #    minimumvalues = np.array(NDI.measurements.minimum(CorrelIntensTab,
    #                                                      ll,
    #                                                      np.arange(1, nf + 1)),
    #                                                      dtype=np.float32)
    #

    #    std_values = np.array(NDI.measurements.standard_deviation(CorrelIntensTab,
    #                                                      ll,
    #                                                      np.arange(1, nf + 1)),
    #                                                      dtype=np.float32)

    # if rank_N is large, blob are no longer symetric,
    # then it would be better to use maxpos than meanpos
    # position variable
    var_pos = meanpos

    # intensity criterium
    var_int = maximumvalues

    SortedOrients = np.take(var_pos, np.argsort(var_int)[::-1], axis=0)

    #    rpos = np.round(var_pos)

    if verbose:
        print("position", var_pos)
        print("intensity criterium", var_int)
        print("SortedOrients", SortedOrients)

    return SortedOrients


def best_orientations_new_3D(Database, bigHoughcollector, maxindex=40,
                                                            rank_n=20,
                                                            sigmagaussian=(1.0, 1.0, 1.0)):
    """
    from Database and experimental data (bigHoughcollector)
    return array of the "nb_orientations" best orientations
    (1 Orientation = 3 angles)

    """

    #    print "bigHoughcollector.shape", bigHoughcollector.shape
    #    print "Database.shape", Database.shape

    Correl3D = Houghcorrel_3D(np.ravel(bigHoughcollector), Database, maxindex=maxindex)
    #    Correl3D = Houghcorrel_3D(np.ravel(np.reshape(bigHoughcollector, (300, 360))), Database)

    N3, N2, N1 = Correl3D.shape

    #    print "rara.shape", rara.shape
    #    print "argrara.shape", argrara.shape

    bestindices = findGrain_in_orientSpace_new_3D(Correl3D, rank_n=rank_n, sigmagaussian=sigmagaussian)
    #    print "bestindices", bestindices

    if len(bestindices) == 0:
        return None

    # indices are directly angles if sampling step is 1 deg
    best_Az, best_Ay, best_Ax = bestindices.T

    bestAngles = np.array([best_Ax, best_Ay, best_Az]).T

    return bestAngles


# --- plot procedures


def plotHough_Exp(prefixname, file_index, nbofpeaks_max, dirname="."):
    """
    read peak list

    High resolution (1 deg step)
    """
    filename_data = prefixname + IOimage.stringint(file_index, 4) + ".cor"
    print("filename_data", filename_data)

    mydata = IOLT.ReadASCIIfile(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]), listofselectedpts, -1)

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(dataselected)

    #    print len(gnomonx)

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(gnomonx, gnomony, Intensity_table=None, lowresolution=1)
    print(bigHoughcollector)
    print(np.shape(bigHoughcollector))
    print(np.histogram(bigHoughcollector))

    bigHoughcollectorfiltered = np.where(bigHoughcollector < 3, -10, 10 * bigHoughcollector)

    p.subplot(121)
    p.title("raw")
    p.imshow(bigHoughcollector, interpolation="nearest", cmap=GT.SPECTRAL_R)

    p.subplot(122)
    p.title("filtered")
    p.imshow(bigHoughcollectorfiltered, interpolation="nearest", cmap=GT.SPECTRAL_R)

    p.show()

    return bigHoughcollector


def plotHough_Simul(Angles, showplot=1, blur_radius=0.5, emax=25, NB=60, pos2D=0, removedges=2):
    t0 = time.time()

    poshighest1D, Houghfiltered, rawHough = Hough_peak_position_fast(Angles,
                                                                    Nb=NB,
                                                                    pos2D=pos2D,
                                                                    removedges=removedges,
                                                                    verbose=0,
                                                                    key_material="Si",
                                                                    EULER=1,
                                                                    arraysize=(300, 360),
                                                                    returnfilterarray=2,
                                                                    blur_radius=blur_radius,
                                                                    emax=emax)

    print("histo", np.histogram(rawHough))

    print("execution time %.3f sec" % (time.time() - t0))
    p.subplot(121)
    p.title("filtered")
    p.imshow(Houghfiltered, interpolation="nearest", cmap=GT.REDS)

    #        #indices from an edges removed array
    #        pospeak = convertindices1Dto2D(datafiltered.shape, poshighest1D, removedges=2)
    #        Y1, X1 = pospeak.T
    #        p.scatter(X1, Y1, c='r')

    # indices from an edges unremoved array
    pospeak = np.zeros(2)
    if poshighest1D is not None:
        pospeak = convertindices1Dto2D(Houghfiltered.shape, poshighest1D, removedges=0)
        Y2, X2 = pospeak.T
        p.scatter(X2, Y2, c="g")
    else:
        print("\nSorry! No peaks found!\n")

    p.subplot(122)
    p.title("raw")
    p.imshow(rawHough, interpolation="nearest", cmap=GT.REDS)
    if poshighest1D is not None:
        p.scatter(X2, Y2, c="g")

    p.show()

    return np.fliplr(pospeak), Houghfiltered, rawHough


def plotHough_compare(Angles, filename_data, nbofpeaks_max, dirname=".", EULER=0):
    """
    plot exp. hough data and simulated ones for Euler angles set

    see plotHough_Exp()
    """
    POS2D = 0
    REMOVEDGES = 2
    poshighest1D, datafiltered, rawsimulHough = Hough_peak_position_fast(Angles,
                                                                        Nb=60,
                                                                        pos2D=POS2D,
                                                                        removedges=REMOVEDGES,
                                                                        verbose=0,
                                                                        key_material="Si",
                                                                        EULER=EULER,
                                                                        arraysize=(300, 360),
                                                                        returnfilterarray=2)

    #  experimental data
    mydata = IOLT.ReadASCIIfile(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]), listofselectedpts, -1)

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(dataselected)

    #    print len(gnomonx)

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(gnomonx, gnomony, Intensity_table=None, lowresolution=1)
    print(bigHoughcollector)
    print(np.shape(bigHoughcollector))

    # plots
    pospeak = convertindices1Dto2D(datafiltered.shape, poshighest1D, removedges=0)
    Y1, X1 = pospeak.T

    p.subplot(131)
    p.imshow(bigHoughcollector, interpolation="nearest", cmap=GT.REDS)
    p.title("experimental")
    p.scatter(X1, Y1, c="b")
    p.subplot(132)
    p.imshow(datafiltered, interpolation="nearest", cmap=GT.REDS)
    p.title("Simulation [%.0f, %.0f, %.0f]" % tuple(Angles))
    p.scatter(X1, Y1, c="b")
    p.subplot(133)
    p.imshow(rawsimulHough, interpolation="nearest", cmap=GT.REDS)
    p.title("raw Simulation [%.0f, %.0f, %.0f]" % tuple(Angles))
    p.scatter(X1, Y1, c="b")

    p.show()


def BrowseHoughCorrel(Database, filename_data, nbofpeaks_max, dirname="."):
    """
    read peak list, transform it into hough space, make correlation with Database

    High resolution (1 deg step)
    """
    mydata = IOLT.ReadASCIIfile(os.path.join(dirname, filename_data))

    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))
    length_of_data = len(data_theta)

    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]), listofselectedpts, -1)

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(dataselected)

    #    print len(gnomonx)

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(gnomonx, gnomony, Intensity_table=None, lowresolution=1)

    #    rara, argrara = Houghcorrel(np.ravel(np.reshape(bigHoughcollector, (300, 360))), Database)
    nd1, nd2, nd3 = 90, 180, 90

    rav_bigHoughcollector = np.ravel(bigHoughcollector)

    print("in BrowseHoughCorrel()")
    print("exp_1d_data.shape", rav_bigHoughcollector.shape)
    # shape of simul_database must be (nb of orientations, nb_of_indices)
    print("simul_database.shape", Database.shape)

    # integration of all intensities picked up in exp_1d_data
    listint = np.reshape(np.sum(rav_bigHoughcollector[Database], axis=1), (nd1 * nd2, nd3))

    # shape of listint must be (90*180,90)
    print("listint.shape", listint.shape)

    listarg = np.argmax(listint, axis=1)  # 1d argmax array
    print("listarg", listarg)
    print("listarg.shape", listarg.shape)

    return listint, listarg


def plotHoughCorrel(Database, filename_data, nbofpeaks_max, dirname=".", returnCorrelArray=0):
    """
    read peak list, transform it into hough space, make correlation with Database

    High resolution (1 deg step)
    """
    mydata = IOLT.ReadASCIIfile(os.path.join(dirname, filename_data))

    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))
    length_of_data = len(data_theta)

    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]), listofselectedpts, -1)

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(dataselected)

    #    print len(gnomonx)

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(gnomonx, gnomony, Intensity_table=None, lowresolution=1)

    rara, argrara = Houghcorrel(np.ravel(np.reshape(bigHoughcollector, (300, 360))), Database)

    if returnCorrelArray:
        return rara, argrara
    else:
        # plot
        plottab(rara)


def plottab(array_to_plot):
    """
    function to plot an array with x,y,intensity values displayed when hovering on array
    """
    fig = p.figure()
    axes = fig.gca()

    # clear the axes and replot everything
    axes.cla()

    axes.imshow(array_to_plot, interpolation="nearest")  # , cmap=GT.GREYS)
    numrows, numcols = array_to_plot.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = array_to_plot[row, col]
            return "x=%1.4f, y=%1.4f, z=%1.4f" % (x, y, z)
        else:
            return "x=%1.4f, y=%1.4f" % (x, y)

    axes.format_coord = format_coord

    p.show()


def plot2tabs(tab1, tab2):
    """
    function to plot 2 arrays
    """

    p.subplot(121)
    p.imshow(tab1, interpolation="nearest")  # , cmap=GT.GREYS)

    p.subplot(122)
    p.imshow(tab2, interpolation="nearest")  # , cmap=GT.GREYS)

    p.show()


def plot1Darray(tab):
    """
    function to plot 2 arrays
    """

    p.plot(tab)
    p.show()


# --- ------------  image matching DATABASE OLD version


def unpickle_olddatabase():
    """ load the database
    lowresolution 2 degrees step ?
    """

    # resultat base de donnees= largeRefFCCpos et largeRefFCCnb
    # np.where(np.array(  largeRefFCCnb)!=40)
    # gives (np.array(  [ 398, 8303]),) ils ont 38 au lieu de 40

    # np.where(bigHoughcollector==0)
    # (np.array(  [], dtype=int32), np.array(  [], dtype=int32))
    # >>> np.where(bigHoughcollector==1)
    # (np.array(  [  0,   0,   0, ..., 359, 359, 359]), array([  1,   2,   3, ..., 296, 297, 298]))
    # a la main:
    # largeRefFCCpos[398]=np.array(  [  3457,  15474,  15475,  15774,  15775,  28931,  33477,  33478,
    #        33479,  33777,  33778,  33779,  38477,  51453,  51454,  51455,
    #        51753,  51754,  51755,  57444,  69426,  69427,  69726,  69727,
    #        82970,  87422,  87423,  87424,  87722,  87723,  87724,  92624,
    #       105446, 105447, 105448, 105746, 105747, 105748,1,1])
    # et
    # largeRefFCCpos[8303]=np.array(  [  3457,  15474,  15475,  15774,  15775,  28931,  33477,  33478,
    #        33479,  33777,  33778,  33779,  38477,  51453,  51454,  51455,
    #        51753,  51754,  51755,  57444,  69426,  69427,  69726,  69727,
    #        82970,  87422,  87423,  87424,  87722,  87723,  87724,  92624,
    #       105446, 105447, 105448, 105746, 105747, 105748,1,1])
    # tabpos=np.array(  largeRefFCCpos)
    # >>> shape(tabpos)
    # (31740, 40)

    frou = open("lowresFCC_pos.dat", "r")
    tabdata = pickle.load(frou)
    frou.close()
    tabdata[398] = np.array(
        [
            3457,
            15474,
            15475,
            15774,
            15775,
            28931,
            33477,
            33478,
            33479,
            33777,
            33778,
            33779,
            38477,
            51453,
            51454,
            51455,
            51753,
            51754,
            51755,
            57444,
            69426,
            69427,
            69726,
            69727,
            82970,
            87422,
            87423,
            87424,
            87722,
            87723,
            87724,
            92624,
            105446,
            105447,
            105448,
            105746,
            105747,
            105748,
            1,
            1,
        ]
    )
    tabdata[8303] = np.array(
        [
            3457,
            15474,
            15475,
            15774,
            15775,
            28931,
            33477,
            33478,
            33479,
            33777,
            33778,
            33779,
            38477,
            51453,
            51454,
            51455,
            51753,
            51754,
            51755,
            57444,
            69426,
            69427,
            69726,
            69727,
            82970,
            87422,
            87423,
            87424,
            87722,
            87723,
            87724,
            92624,
            105446,
            105447,
            105448,
            105746,
            105747,
            105748,
            1,
            1,
        ]
    )
    return np.array(tabdata)


def unpickle_2databases():
    """ Low resolution databank (2 deg step) with 35 indices per orientation
    """
    frou = open("lowres2FCC_pos_0.dat", "r")
    tabdata_0 = pickle.load(frou)
    frou.close()
    frou = open("lowres2FCC_pos_1.dat", "r")
    tabdata_1 = pickle.load(frou)
    frou.close()
    return np.concatenate((np.array(tabdata_0), np.array(tabdata_1)))


def unpickle_part_databases(data_index, prefixname="FCC_pos_25keV_", databasefolder="."):
    """
    -high resolution databank (1 deg step) in EULER angles
    with 35 featuring indices per orientation
    -index of database ranges from 1 to 12
    - last file contains less (16200) than the previous databank (64800 elements)
    """
    Globalname = prefixname + str(data_index) + ".dat"
    # binary mode , data were created with cPickle.dum(...,...,protocol=2
    frou = open(os.path.join(databasefolder, Globalname), "rb")  # could be 'r' only instead of 'rb' !!
    tabdata = pickle.load(frou)
    frou.close()
    return np.array(tabdata)


def create_database_step1deg(short="No", singleindex=1, databasefolder="."):
    """ create 1d databank
    729000  *    35 indices  =  25515000 elements
    """
    toreturn = []
    wholerange = list(range(1, 13))
    if short == "Yes":
        wholerange = (singleindex,)
    for k in wholerange:
        print("loading databank #%d/12" % k)
        toreturn.append(
            unpickle_part_databases(k, prefixname="FCC_pos_25keV_", databasefolder=databasefolder))
    # print "Databank is now in list called: databankHoughIndices"
    return np.concatenate(toreturn)


def create_database_step1deg_12keV(short="No"):
    """ create 1d databank
    729000  *    35 indices  =  25515000 elements
    """
    toreturn = []
    wholerange = list(range(0, 9))
    if short == "Yes":
        wholerange = (1,)
    for k in wholerange:
        print("loading databank #%d/8" % k)
        toreturn.append(unpickle_part_databases(k, prefixname="Ref_fcc_pos_"))
    # print "Databank is now in list called: databankHoughIndices"

    return np.concatenate(toreturn)


def BuildDataBaseImageMatching():
    """
    create old imagematching database
    with Hough_peak_position
    """
    largeRefFCCpos = []
    largeRefFCCnb = []
    index_ = 0
    t0 = time.time()
    for angZ in np.arange(48.0, 90.0, 1.0):
        print("-------------------------------------")
        print("          angZ", angZ)
        for angY in np.arange(0.0, 90.0, 1.0):
            print("  angY", angY)
            for angX in np.arange(0.0, 90.0, 1.0):
                # if (angX%30)==0: print "angX",angX
                nb, pos = Hough_peak_position([angX, angY, angZ],
                                                key_material="Si",
                                                returnXYgnomonic=0,
                                                arraysize=(300, 360),
                                                verbose=0,
                                                EULER=1,
                                                saveimages=0,
                                                saveposition=0,
                                                prefixname="Ref_H_FCC_")
                largeRefFCCpos.append(pos)
                largeRefFCCnb.append(nb)
                index_ += 1
    tf = time.time()
    print("index", index_)
    print("time ", tf - t0)

    print("nb elem", len(largeRefFCCpos))

    file_pos = open("FCC_pos_7.dat", "w")
    pickle.dump(largeRefFCCpos, file_pos)
    file_pos.close()

    file_nb = open("FCC_nb_7.dat", "w")
    pickle.dump(largeRefFCCnb, file_nb)
    file_nb.close()


# --- -----------  image matching DATABASE NEW version

# --- to build
def BuildDataBaseImageMatching_fast(FROMIND, UPTOIND, fileindex):
    """
    building database for image matching indexing
    """

    NBPIXELS_HOUGH = 40
    EMAX = 25
    EULER = 1
    ELEMENT = "Si"

    index_ = 0

    samplingAngZ = np.linspace(FROMIND, UPTOIND, 1)
    DATABANK = np.zeros((len(samplingAngZ) * 180 * 90, NBPIXELS_HOUGH), dtype=np.uint32)
    print("DATABANK", DATABANK.shape)
    t00 = time.time()
    for angZ in samplingAngZ:
        print("-------------------------------------")
        print("          angZ: %f                   " % angZ)
        print("-------------------------------------")
        for angY in np.arange(0.0, 180.0, 1.0):
            if int(angY) % 2 == 0:
                print("  angY ----> ", angY)
            t0 = time.time()
            for angX in np.arange(0.0, 90.0, 1.0):
                #                    if (angX % 30) == 0: print "angX", angX
                pos = Hough_peak_position_fast([angX, angY, angZ],
                                            Nb=NBPIXELS_HOUGH,
                                            pos2D=0,
                                            removedges=2,
                                            blur_radius=0.5,
                                            key_material=ELEMENT,
                                            arraysize=(300, 360),
                                            verbose=0,
                                            EULER=EULER,
                                            emax=EMAX)

                if pos is not None:
                    DATABANK[index_] = pos
                else:
                    raise ValueError("no peak found for [%.f,%.f,%.f]" % (angX, angY, angZ))

                index_ += 1

            tf = time.time()
            print("loop time ", tf - t0)

    tf = time.time()
    print("index", index_)
    print("total time ", tf - t00)

    file_pos = open("Si_REF_%d" % fileindex, "w")
    pickle.dump(DATABANK, file_pos)
    file_pos.close()


def builddatabaseRange(startAngle, endAngle):
    """
    see builddatabase.py

    on crg3 launch on 8 xterms (eight procs)
    by dispatching angle1 angle2 values: from 0 to 89

    prompt> /usr/bin/python builddatabase.py angle1 angle2
    """
    for ind in list(range(startAngle, endAngle + 1)):
        BuildDataBaseImageMatching_fast(ind, ind + 1, ind)


# --- to read


def Read_One_DataBaseImageMatching(file_index, prefixname="Si_REF_", databasefolder="."):
    """
    Read one part of database (internal purpose)
    -high resolution databank (1 deg step) in EULER angles
    with 40 featuring indices per orientation
    -file_index of database ranges from 0 to 89 (every 1 deg in AngZ)
    """
    fullname = prefixname + "%d" % file_index
    # binary mode , data were created with cPickle.dum(...,...,protocol=2
    f = open(os.path.join(databasefolder, fullname), "rb")  # could be 'r' only instead of 'rb' !!
    tabdata = np.array(pickle.load(f))
    f.close()
    print(tabdata.shape)
    return tabdata


def CreateDataBasestep1deg(databasefolder=None, prefixname="Si_REF_"):
    """
    stack every parts of database (internal purpose)
    create 1d database new version
    90*180*90  *    40 indices  =  58 320 000 elements
    """
    if databasefolder is None:
        databasefolder = os.curdir
    toreturn = []

    for k in list(range(90)):
        print("loading databank #%d/89" % k)
        toreturn.append(Read_One_DataBaseImageMatching(
                k, prefixname=prefixname, databasefolder=databasefolder))

    allDatabase = np.concatenate(toreturn)

    f = open("Si_REF_25keV_40", "w")
    pickle.dump(allDatabase, f)
    f.close()
    print("Database has been saved in %s" % ("Si_REF_25keV_40"))


def DataBaseImageMatchingSi():
    """
    Read full image matching database file as an array
    read 1d databank of Si
    90*180*90  *    40 indices  =  58 320 000 elements
    """
    DBname = "Si_REF_25keV_40"
    dirname = os.path.join(os.curdir, "ImageMatchingDatabase")
    g = open(os.path.join(dirname, DBname), "r")
    data = pickle.load(g)
    g.close()
    return data
