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
__version__ = '$Revision$'

import os, time, copy
import pickle
import pickle

import scipy.ndimage as NDI
try:
    import Image
except:
    print("module Image / PIL is not installed")
import pylab as p
from annot import AnnoteFinder

import numpy as np
try:
    import ImageFilter  # should contain JSM filter in the future
except ImportError:
    import PIL.ImageFilter as ImageFilter

import lauecore as LAUE
import CrystalParameters as CP
import LaueGeometry as F2TC

import dict_LaueTools as DictLT
import generaltools as GT

import readmccd as RMCCD
import IOLaueTools as IOLT
# import indexingSpotsSet as ISS

#--- ------------ CONSTANTS
DEG = np.pi / 180.
CST_ENERGYKEV = DictLT.CST_ENERGYKEV

#--- -------------  PROCEDURES


def toviewgnomonofMARCCD():
    """ to plot the gnomonic projection of MARCCD chip
    """
    nbpixels = 2048 * 2048
    nbsteps = 500
    # TODO: missing args
    tata = F2TC.Compute_data2thetachi()  # missing args
    bill = ComputeGnomon_2((np.ravel(tata[0]), np.ravel(tata[1])))[0]
    p.scatter(np.take(bill[0],
            np.arange(0, nbpixels, nbsteps)),
            np.take(bill[1],
            np.arange(0, nbpixels, nbsteps)))
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


def stringint(k, n):
    """ returns string of k by placing zeros before to have n characters
    ex: 1 -> '0001'
    15 -> '0015'
    
    # TODO: replace by string format %04d
    """
    strint = str(k)
    res = '0' * (n - len(strint)) + strint
    return res


def plotgnomondata(gnomonx, gnomony, X, Y,
                   savefilename=None, maxIndexIoPlot=None, filename_data=None):
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
    p.title('exp. data (n*theta,chi)')
    p.xlabel('n*theta')
    p.ylabel('chi')

    p.subplot(212)
    p.scatter(gnomonx[:mostintense], gnomony[:mostintense])
    p.xlabel('X')
    p.ylabel('Y')
    p.xlim(-.8, .8)
    p.title('gnomonic projection of spots of %s' % filename_data)
    p.grid(True)

    if savefilename:
        p.savefig(savefilename)

    mystrlabel = np.array(np.arange(len(X)), dtype='S11')

    af = AnnoteFinder(gnomonx, gnomony, mystrlabel)
    p.connect('button_press_event', af)
    p.show()


def Plot_compare_gnomon(Angles, Xgnomon_data, Ygnomon_data, key_material=14, EULER=0):
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
    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)

    # array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1, fastcompute=1, fileOK=0, verbose=0)
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

    p.title('Euler Angles [%.1f,%.1f,%.1f]' % (tuple(Angles)))
    p.scatter(Xgnomon_data, Ygnomon_data, s=50, c='w', marker='o', faceted=True, alpha=.5)
    p.scatter(xgnomon, ygnomon, c='r', faceted=False)
    p.show()


def Plot_compare_gnomondata(Angles, twicetheta_data, chi_data,
                            verbose=1,
                            key_material='Si',
                            emax=25, emin=5,
                            EULER=0,
                            exp_spots_list_selection=None):
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

    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)


    # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1,
                                 fastcompute=1, fileOK=0, verbose=0)

    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    xgnomon_theo , ygnomon_theo = ComputeGnomon_2((TwicethetaChi[0], TwicethetaChi[1]))

#    print "nb of spots in CCD frame", len(TwicethetaChi[0])

    if exp_spots_list_selection is not None:  # to plot only selected list of exp. spots
        sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
        sel_chi = np.array(chi_data)[exp_spots_list_selection]

        # compute gnomonic coordinates:
        xgnomon , ygnomon = ComputeGnomon_2((sel_2theta, sel_chi))

    else:
        # compute gnomonic coordinates:
        xgnomon , ygnomon = ComputeGnomon_2((twicetheta_data, chi_data))

    p.title('Euler Angles [%.1f,%.1f,%.1f]' % (tuple(Angles)))

    # plot exp.spots
    p.scatter(xgnomon, ygnomon, s=40, c='w', marker='o', faceted=True, alpha=.5)


    # simulated scattered spots
    p.scatter(xgnomon_theo, ygnomon_theo, c='r', faceted=False)

    p.show()


def Plot_compare_2thetachi(Angles, twicetheta_data, chi_data,
                            verbose=1,
                            key_material='Si',
                            emax=25, emin=5,
                            EULER=0,
                            exp_spots_list_selection=None):
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

    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)


    # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1,
                                 fastcompute=1, fileOK=0, verbose=0)

    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

#    print "nb of spots in CCD frame", len(TwicethetaChi[0])

    if exp_spots_list_selection is not None:  # to plot only selected list of exp. spots
        sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
        sel_chi = np.array(chi_data)[exp_spots_list_selection]

    p.title('Euler Angles [%.1f,%.1f,%.1f]' % (tuple(Angles)))
    if exp_spots_list_selection is not None:
        p.scatter(sel_2theta, sel_chi, s=40, c='w', marker='o', faceted=True, alpha=.5)
    else:
        p.scatter(twicetheta_data, chi_data, s=40, c='w', marker='o', faceted=True, alpha=.5)
    # simulated scattered spots
    p.scatter(TwicethetaChi[0], TwicethetaChi[1], c='r', faceted=False)

    p.show()


def Plot_compare_2thetachi_multi(list_Angles, twicetheta_data, chi_data,
                                verbose=1,
                                emax=25, emin=5,
                                key_material=14,
                                EULER=0,
                                exp_spots_list_selection=None,
                                title_plot='default',
                                figsize=(6, 6),
                                dpi=80):
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
        grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)


        # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1, fastcompute=1, fileOK=0, verbose=0)
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
            p.title('nb close,<0.5deg: %d,%d  mean ang %.2f' % tuple(sco))
        else:
            if type(EULER) != type(np.array([1, 2, 3])):
                if EULER == 1:
                    p.title('Euler Angles [%.1f,%.1f,%.1f]' % (tuple(list_Angles[orient_index])))
            else:
                p.title('Orientation Matrix #%d' % orient_index)

        ax.set_xlim((35, 145))
        ax.set_ylim((-45, 45))
        # exp spots
        ax.scatter(sel_2theta, sel_chi, s=40, c='w', marker='o', faceted=True, alpha=.5)
        # theo spots
        ax.scatter(TwicethetaChi[0], TwicethetaChi[1], c='r', faceted=False)
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
    if angle < 0.:
        res = 180. + angle
    return res


def gen_Nuplets(items, n):
    """
    generator taking n-uplet from items, and
    """
    if n == 0:
        yield []
    else:
        for i in range(len(items) - n + 1):
            for cc in gen_Nuplets(items[i + 1:], n - 1):
                yield [items[i]] + cc


def ReadASCIIfile(_filename_data, col_2theta=0, col_chi=1, col_Int=-1, nblineskip=1):
    """ from a file
    return 3 arrays of columns located at index given by
    col_2theta=0, col_chi=1, col_Int=-1:
    [0] theta
    [1] chi
    [2] intensity
    
    TODO: to move to readwriteASCII module
    """

    _tempdata = np.loadtxt(_filename_data, skiprows=nblineskip)
    # _tempdata = scipy.io.array_import.read_array(_filename_data, lines = (nblineskip,-1))

    _data_theta = _tempdata[nblineskip - 1:, col_2theta] / 2.
    _data_chi = _tempdata[nblineskip - 1:, col_chi]

    try:
        _data_I = _tempdata[nblineskip - 1:, col_Int]
    except IndexError:
        print("there are not 5 columns in data.cor file")
        print("I create then a uniform intensity data!")
        _data_I = np.ones(len(_data_theta))
    if (np.array(_data_I) < 0.).any():
        print("Strange ! I don't like negative intensity...")
        print("I create then a uniform intensity data!")
        _data_I = np.ones(len(_data_theta))

    return (_data_theta, _data_chi, _data_I)

# keep track of old name for external import
createrawdata = ReadASCIIfile





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

#--- ------------  Gnomonic Projection

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

    data_theta = TwiceTheta_Chi[0] / 2.
    data_chi = TwiceTheta_Chi[1]

    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(-np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)

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
    return theta and chi of Q (direction of Q)
    """
    lat0 = np.ones(len(_gnomonX)) * np.pi / 4
    longit0 = np.zeros(len(_gnomonX))
    Rho = np.sqrt(_gnomonX ** 2 + _gnomonY ** 2) * 1.
    CC = np.arctan(Rho)

    # the sign should be - !!
    lalat = np.arcsin(np.cos(CC) * np.sin(lat0) + _gnomonY / Rho * np.sin(CC) * np.cos(lat0))
    lonlongit = longit0 + np.arctan2(_gnomonX * np.sin(CC),
                                     Rho * np.cos(lat0) * np.cos(CC) - _gnomonY * np.sin(lat0) * np.sin(CC))

    Theta = np.arcsin(np.cos(lalat) * np.cos(lonlongit))
    Chi = np.arctan(np.sin(lonlongit) / np.tan(lalat))
    return Theta, Chi


def Fromgnomon_to_2thetachi(gnomonicXY, _dataI):
    """ From an array:
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
    longit = longit0 + np.arctan2(data_x * sargc,
                               rho * clat0 * cargc - data_y * slat0 * sargc)

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


    if np.__version__ < '1.3.0':
        np.put(intermediate_accum, onepointcontribution_list_index, _pointintensity)
    else:
        intermediate_accum[onepointcontribution_list_index] = _pointintensity * \
                                        np.ones(len(onepointcontribution_list_index))

    # print "binned rho",len(binnedrho)
        # print "_pointintensity",_pointintensity
        # print "onepointcontribution_list_index",onepointcontribution_list_index[onepointcontribution_list_index>140000]
        # print "intermediate_accum",intermediate_accum[intermediate_accum!=1]

    # print "where non nul",np.where(intermediate_accum!=0)
        # print np.where(intermediate_accum>0)
        # print intermediate_accum[60030:60040]
        # print "len(intermediate_accum)",len(intermediate_accum)
    return intermediate_accum

#--- ----------------  Hough Transform
def ComputeHough(_datagnomonx, _datagnomony,
                 Intensity_table=None,
                 stepdeg=.5,
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
        stepdeg = 1.
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
        bibins = np.concatenate((np.arange(-1. + .4, 1. - .4, steprho), np.array([10])))
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

        bibins = np.arange(-1. + .4, 1. - .4, steprho)

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

            if np.__version__ < '1.3.0':
                np.put(intermediate_accum, onepointcontribution_list_index, _pointintensity)
            else:
                intermediate_accum[onepointcontribution_list_index] = _pointintensity * \
                                                np.ones(len(onepointcontribution_list_index))

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
        for k in range(len(_datagnomonx)):  # JSM 21 OCt 2010   10 what for ???
            # print "tra",tra[k]
            # print "intiti",Intensity_table[k]
            # print "%d/%d"%(k,len(_datagnomonx)-1)
            accumHough = accumHough + accum_one_point(rhotable_from_one_point(tra[k]),
                                                      Intensity_table[k])
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


def InverseHough(rhotheta, halfsemiaxes=(1.2, .6), centerofproj=(0., 0.)):
    """ from rho and theta in hough space
    gives extreme points of a straight line where must lie all spots
    belonging to the same zone axis
    In a rigourous manner, we must find the intersection of the straight line with an ellipse.
    Here, we consider simply a rectangle!
    """
    pass


def Hough_peak_position(Angles, verbose=1, saveimages=1, saveposition=1,
                        prefixname="tata", key_material=14, PlotOK=0,
                        arraysize=(600, 720), returnXYgnomonic=0, EULER=0):
    """
    peak in seach in Hough space representation of Laue data projected in gnomonic plane
    WARNING: need to retrieve a filter called JSM contained in ImageFilter of PIL Image module 
    
    TODO: use numerical filter of ndimage rather than PIL. Should be faster
    
    TODO: to be removed soon
    """
    emax = 25
    emin = 5
    nb_of_peaks = 35  # nb of hough pixel featuring the laue pattern information in hough space

    angle_X, angle_Y, angle_Z = Angles
    nn = 1  # nn-1= number of digits in angles to put in the outputfilename

    fullname = prefixname + str(int(nn * angle_Z)) + str(int(nn * angle_Y)) + str(int(nn * angle_X))
    # print fullname

    if EULER == 0:
        mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])

    # PATCH to use correctly getLaueSpots() of laue6
    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)


    # array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1, fastcompute=1, fileOK=0, verbose=0)
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
    xgnomon, ygnomon, = ComputeGnomon_2((xxtab[tokeep], yytab[tokeep]))

    # scatter(xgnomon,ygnomon)
    # show()
    if verbose:
        print("Computing %d selected MARCCD pixels in Gnomonic space for Hough transform" % len(xgnomon))
        print("Now all pixels are considered to have the same intensity")

    tagreso = 0
    if arraysize == (300, 360):
        tagreso = 1
    bigHoughcollector = ComputeHough(xgnomon, ygnomon, Intensity_table=None, stepdeg=.5,
                                     steprho=0.002, lowresolution=tagreso)
    # argsort(ravel(bigHoughcollector))[-30:] for taking the first 30 most intense peak

    # some filtering to extract main peaks
    mike = np.ravel(bigHoughcollector)
    # mikemax=max(mike)
    # print "raw histogram",histogram(mike,range(mikemax))


    if saveimages:
        mikeraw = Image.new("L", arraysize)
        mikeraw.putdata(mike)
        mikeraw.save(fullname + 'raw' + '.TIFF')

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
        mikeclipjsm.save(fullname + 'clipfilt' + '.TIFF')
    # ---------------------

#    popos = RMCCD.LocalMaxima_ndimage(mike, peakVal=4,
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
        outputname = fullname + 'pick'
        filepickle = open(outputname, 'w')
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

    x_ind = np.where(abs(np.ravel(Twicetheta - np.reshape(Twicetheta, (lon, 1)))[toextract]) < dist_tolerance)
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




def Hough_peak_position_fast(Angles, Nb=50, verbose=1, pos2D=0, removedges=2,
                        key_material='Si', returnfilterarray=0,
                        arraysize=(300, 360), EULER=0, blur_radius=0.5,
                        printOrientMatrix=0, emax=25):
    """
    Simulate for 3 Euler angles Laue pattern
    peak search in Hough space representation of Laue data projected in gnomonic plane

    """
    emin = 5

    angle_X, angle_Y, angle_Z = Angles

    if EULER == 0:
        mymat = GT.fromelemangles_toMatrix([angle_X * 1., angle_Y * 1., angle_Z * 1.])
    else:
        mymat = GT.fromEULERangles_toMatrix([angle_X * 1., angle_Y * 1., angle_Z * 1.])

    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)

    if printOrientMatrix:
        print("grainparameters")
        print(grain)

    # simulation + ad hoc method for harmonics removal
    if 1:
        # array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain],
                                     1, fastcompute=1, fileOK=0, verbose=0)

        # 2theta,chi of spot which are on camera (BUT with harmonics)
        TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1, detectordistance=70.)

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
        simulparameters['detectordiameter'] = 165.
        simulparameters['kf_direction'] = 'Z>0'
        simulparameters['detectordistance'] = 70.
        simulparameters['pixelsize'] = 165. / 2048

        TwicethetaChi = LAUE.SimulateResult(grain, emin, emax,
                   simulparameters,
                   fastcompute=1,
                   ResolutionAngstrom=False)

    #    print "TwicethetaChifiltered"
#        print len(TwicethetaChi[0])

        TTH = TwicethetaChi[0]
        CHI = TwicethetaChi[1]

    # compute gnomonic projection

    VIPintensity = None
    xgnomon, ygnomon = ComputeGnomon_2((TTH, CHI))

    if verbose:
        print("Computing %d selected MARCCD pixels in Gnomonic space for Hough transform" % len(xgnomon))
        print("Now all pixels are considered to have the same intensity")

    tagreso = 0
    if arraysize == (300, 360):
        tagreso = 1


    bigHoughcollector = ComputeHough(xgnomon, ygnomon, Intensity_table=None, stepdeg=.5,
                                     steprho=0.002, lowresolution=tagreso)

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
        bigHoughcollectorfiltered = NDI.gaussian_filter(bigHoughcollectorfiltered, (1, .7))
        bigHoughcollectorfiltered = np.where(bigHoughcollectorfiltered <= 1,
                                             0, bigHoughcollectorfiltered)
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

        popul , bins = np.histogram(bigHoughcollector)
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

        poshighest_most = get_mostintensepeaks(bigHoughcollectorfiltered, Nb,
                                          pos2D=1, removedges=removedges)

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

        popul , bins = np.histogram(array_to_filter)
#        print "intensity frequency", popul
#        print "intensity bins", bins

        nbiter = 0
        while(True):
            THRESHOLD = bins[1]
            poshighest_2D = getblobCOM(array_to_filter,
                                   threshold=THRESHOLD, connectivity=1, returnfloatmeanpos=0,
                                   removedges=2)
            if len(poshighest_2D) >= Nb:
                poshighest_2D_final = poshighest_2D[:Nb]
                break
            else:
                print("reducing the threshold")
                THRESHOLD -= .5
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
    clipdata_array = data_array[removedges:nbrow - removedges, removedges: nbcol - removedges]

    posbest = np.argsort(np.ravel(clipdata_array))[-Nb:]

    # return index from an 2D array
    if pos2D:
        return convertindices1Dto2D(data_array.shape, posbest, removedges=removedges)
    # return index from an 1D array
    else:
        return posbest

def tophatfilter(data_array, removedges=2, peakVal=4,
                        boxsize=6,
                        central_radius=3):
    """
    apply an top hat filter to array without edges
    
    """
    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges:nbrow - removedges, removedges: nbcol - removedges]

    enhancedarray = RMCCD.ConvolvebyKernel(clipdata_array,
                        peakVal=peakVal,
                        boxsize=boxsize,
                        central_radius=central_radius
                        )
    newarray = np.zeros_like(data_array)
    newarray[removedges:nbrow - removedges, removedges: nbcol - removedges] = enhancedarray

    return newarray

def meanfilter(data_array, removedges=2, n=3):
    """
    apply an top hat filter to array without edges
    
    """
    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges:nbrow - removedges, removedges: nbcol - removedges]
    kernel = np.ones((n, n)) / (1.*n ** 2)
    enhancedarray = NDI.filters.convolve(clipdata_array, kernel)

    newarray = np.zeros_like(data_array)
    newarray[removedges:nbrow - removedges, removedges: nbcol - removedges] = enhancedarray

    return newarray


def getblobCOM(data_array, threshold=0, connectivity=1, returnfloatmeanpos=0, removedges=0):
    """
    return center of mass of blob in data_array
    
    blobs are sorted regarding their intensity 
    
    """

    nbrow, nbcol = data_array.shape
    clipdata_array = data_array[removedges:nbrow - removedges, removedges: nbcol - removedges]

    thraa = np.where(clipdata_array > threshold, 1 , 0)

    if connectivity == 0:
        star = np.eye(3)
        ll, nf = NDI.label(thraa, structure=star)
    elif connectivity == 1:
        ll, nf = NDI.label(thraa, structure=np.ones((3, 3)))
    elif connectivity == 2:
        ll, nf = NDI.label(thraa, structure=np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]]))

    range_nf = np.array(np.arange(1, nf + 1), dtype=np.int16)

    meanpos = np.array(NDI.measurements.center_of_mass(thraa, ll, range_nf),
                                                                dtype=np.float)

#    maximumvalues = np.array(NDI.measurements.maximum(thraa, ll, range_nf),
#                                                                dtype=np.float)

    sumofvalues = np.array(NDI.measurements.sum(thraa, ll, range_nf),
                                                                dtype=np.float)

#    meanofvalues = np.array(NDI.measurements.sum(thraa, ll, np.arange(1, nf + 1)),
#                                                                dtype=np.float)

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
    clipdata_array = data_array[removedges:nbrow - removedges, removedges: nbcol - removedges]

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
    newarray[removedges:nbrow - removedges, removedges: nbcol - removedges] = enhancedarray

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

    original_index = offset + indices1D_croppedarray / (nbcol - 2 * removedges) + \
                                indices1D_croppedarray % (nbcol - 2 * removedges)

    return original_index


def StickLabel_on_exp_peaks(Angles,
                            twicetheta_data, chi_data,
                            angulartolerance,
                            emax,
                            verbose=1,
                            key_material=14,
                            arraysize=(600, 720),
                            EULER=1):

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
    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)


    # fastcompute=0 => array(vec) and array(indices) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1, fastcompute=0, fileOK=0, verbose=0)

    # fastcompute=0 => result is list of spot instances which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=0)

    theta_and_chi_theo = np.array([[elem.Twicetheta / 2., elem.Chi] for elem in TwicethetaChi[0]])
    theta_and_chi_exp = np.array([twicetheta_data / 2., chi_data]).T

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
            if ind_exp in peak_dict_dist:  # key is defined if previous angular tolerance for labelling was True

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
    aa, bb = np.where(np.logical_and(table_distance_T <= 1.5 * angulartolerance, table_distance_T != 0.0))

    theo_to_remove = np.array(list(set(aa[aa < bb]).union(set(bb[aa < bb]))))  # to remove some spurious numerical imprecision: e.g. finding angle between the same point  !=0 !!!
    if verbose:
        print("index to remove in theo_spot_index_list", theo_to_remove)
        print("True index of theo spots to be removed", theo_spot_index_list[theo_to_remove])
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
    cc, dd = np.where(np.logical_and(table_distance_E <= 1.5 * angulartolerance, table_distance_E != 0.0))
    exp_to_remove = np.array(list(set(cc[cc < dd]).union(set(dd[cc < dd]))))  # to remove some spurious numerical imprecision: e.g. finding angle between the same point  !=0 !!!
    if verbose:
        print("index to remove in exp_spot_index_list", exp_to_remove)
        print("True index of exp spots to be removed", exp_spot_index_list[exp_to_remove])
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
    return theta_and_chi_exp, theta_and_chi_theo, peak_dict_ind, prox_index, p_d_d, p_d_i


#--- --------------- IMAGE MATCHING INDEXING

#--- --very old method

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

    (toeleminate_1, toeleminate_2,
     toeleminate_3, toeleminate_4) = (np.array([]), np.array([]),
                                      np.array([]), np.array([]))
    if len(pos1[0]) > 0:
        # print "pos1",pos1
        toeleminate_1 = gluck[np.amax(np.array(pos1))]
        # print "elimi h",toeleminate_1


    # ----too long to remove along column ..
    # TODO: what is the 23 !!
    newgluck = (gluck % 23) * 23 + gluck / 23  # indexation by permuting col and row
    # print newgluck
    # diffgluck_j = newgluck - np.reshape(newgluck, (nbfirstpt, 1)) # difference at j cst
    pos2 = np.where(newgluck - np.reshape(newgluck, (nbfirstpt, 1)) == 1)  # next top or bottom
    if len(pos2[0]) > 0:
        # print "pos2",pos2
        toelem_2 = np.amax(np.array(pos2))
        # print "raw elim index",toelem_2
        # print "raw abs index",newgluck[np.array(  toelem_2)]
        toeleminate_2 = (newgluck[toelem_2] % 23) * 23 + newgluck[toelem_2] / 23
        # print "elimi v good indice",toeleminate_2

    pos22 = np.where(gluck - np.reshape(gluck, (nbfirstpt, 1)) == 22)  # close left and top distance indice 22 et inv.
    pos24 = np.where(gluck - np.reshape(gluck, (nbfirstpt, 1)) == 24)  # close left and top distance indice 24 et inv.

    if len(pos22[0]) > 0:
        # print "pos22",pos22
        toeleminate_3 = gluck[np.amax(np.array(pos22))]
        # print "elimi diag1",toeleminate_3

    if len(pos24[0]) > 0:
        # print "pos24",pos24
        toeleminate_4 = gluck[np.amax(np.array(pos24))]
        # print "elimi diag1",toeleminate_4

    torem = set(np.concatenate((toeleminate_1, toeleminate_2, toeleminate_3, toeleminate_4)))

    if len(torem) > 0:
        print("removing some orientations")
        # print torem
        print("Remaining orientations")
        remindices = np.array(list(set(gluck) - torem))
        remelem = []
        for k in range(len(gluck)):
            if gluck[k] in remindices:
                remelem.append(gluck[k])
        return remelem
    else:
        print("No contiguous orientations found")
        return gluck


def convert_Orientindex_to_angles(array_of_index, dimensions, resolution='high'):
    """ 
    dimensions is a tuple (dim1,dim2)
    """
    if resolution == 'low':
        nz, ny = dimensions

        _angZ = array_of_index / nz
        _angY = array_of_index % nz

        R_Z = np.arange(0., 45., 2.)
        R_Y = np.arange(0., 45., 2.)
    elif resolution == 'high':
        nz, ny = 90, 90

        _angZ = array_of_index / nz
        _angY = array_of_index % nz

        R_Z = np.arange(0., 90., 1.)
        R_Y = np.arange(0., 90., 1.)

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
    best_Ax = np.arange(0., 90., 1.)[argrara[bestindices]]


    print(best_Ax, "\n", best_Ay, "\n", best_Az)

    return np.transpose(np.array([best_Ax, best_Ay, best_Az]))


def give_best_orientations(prefixname,
                            file_index,
                            nbofpeaks_max,
                            _fromfiledata,
                            col_I=1,
                            nb_orientations=20,
                            dirname='.',
                            plotHough=0):
    """
    read peak list and by image matching propose a list of orientation for peaks indexation
    
    High resolution (1 deg step)
    _fromfiledata is the databank: huge array containing indices to pick up in transformed data p
    """
    filename_data = prefixname + stringint(file_index, 4) + '.cor'
    print("filename_data", filename_data)
#    nblines_to_skip=1 # nb lines of header
#    col_2theta=0
#    col_chi=1
    # col_I=1 # Intensity  column

    # nbofpeaks_max=500 # for recognition

    mydata = ReadASCIIfile(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]),
                                             listofselectedpts, -1)

    print(" ******   Finding best orientations ****************")
    print("Raw data in %s have %d spots" % (filename_data, len(data_theta)))
    print("Number of selected (or not recongnised) spots from raw data: ", len(listofselectedpts))
    print("Looking only from these last spots, only the first %d spots" % nbmax)
    print("Last spot index probed in rawdata: %d" % listofselectedpts[nbmax - 1])
    print("\n")

    # compute gnomonic coordinnates of a discrete number of peak
    # (from 2theta,chi coordinates)
    gnomonx, gnomony = ComputeGnomon_2(dataselected)

#    print len(gnomonx)

    # lowresolution=1  : stepdeg=1,steprho=0.004
    bigHoughcollector = ComputeHough(gnomonx, gnomony, Intensity_table=None, lowresolution=1)
    print(bigHoughcollector)
    print(np.shape(bigHoughcollector))
    print(np.shape(_fromfiledata[100]))  # nb of fingerprint indices

    if plotHough:
        p.imshow(bigHoughcollector, interpolation='nearest')
        p.show()

    AAA = best_orientations(_fromfiledata, bigHoughcollector, nb_orientations=nb_orientations)

    return np.transpose(AAA), dataselected, [gnomonx, gnomony]


#--- -old method


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

    CorrelIntensTab = NDI.filters.gaussian_filter(CorrelIntensTab, sigma=(1., 1.))
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

    thraa = np.where(CorrelIntensTab > intensity_rankN, CorrelIntensTab , 0)

    maxtab = NDI.filters.maximum_filter(thraa, size=3)
    ll, nf = NDI.label(maxtab)

    maxpos = np.array(NDI.measurements.maximum_position(maxtab,
                                                        ll,
                                                        np.arange(1, nf + 1)),
                                                        dtype=np.float)

#    meanvalues = np.array(NDI.measurements.sum(thraa,
#                                                    ll,
#                                                    np.arange(1, nf + 1)),
#                                                    dtype=np.float)

    meanvalues2 = np.array(NDI.measurements.mean(thraa,
                                                    ll,
                                                    np.arange(1, nf + 1)),
                                                    dtype=np.float)

    maximumvalues = np.array(NDI.measurements.maximum(CorrelIntensTab,
                                                      ll,
                                                      np.arange(1, nf + 1)),
                                                      dtype=np.float)


    minimumvalues = np.array(NDI.measurements.minimum(CorrelIntensTab,
                                                      ll,
                                                      np.arange(1, nf + 1)),
                                                      dtype=np.float)


#    std_values = np.array(NDI.measurements.standard_deviation(CorrelIntensTab,
#                                                      ll,
#                                                      np.arange(1, nf + 1)),
#                                                      dtype=np.float)

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
#                        convolvedCorrelIntensTab = RMCCD.ConvolvebyKernel(CorrelIntensTab,
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
# #                                                        dtype=np.float)
#    print "ll", np.transpose(ll)[8:15, 40:60]
#    maxpos = np.array(NDI.measurements.maximum_position(thraa,
#                                                        ll,
#                                                        np.arange(1, nf + 1)),
#                                                        dtype=np.float)
#
#    maximumvalues = np.array(NDI.measurements.maximum(CorrelIntensTab,
#                                                      ll,
#                                                      np.arange(1, nf + 1)),
#                                                      dtype=np.float)
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

    bestindices = np.array(findGrain_in_orientSpace_new(rara, nb_orientations), dtype=np.int)

    # TODO: generalize to other angles sampling
#    best_Az, best_Ay = convert_Orientindex_to_angles(bestindices, (90, 90))

    best_Az, best_Ay = bestindices.T  # indices are directly angles

#    best_Ax = np.arange(0., 90., 1.)[argrara[bestindices]]
    best_Ax = argrara[bestindices[:, 0] * N2 + bestindices[:, 1]]

    bestAngles = np.array([best_Ax, best_Ay, best_Az]).T

    print("best angles")
    print(bestAngles)

    return bestAngles

#--- ---current method


def bestorient_from_2thetachi(tth_chi_int, database,
                              dictparameters=None):
    """
    return sorted list of 3eulers angles and corresponding matching rate
    
    dictparameters is a dictionary of parameters for imagematching technique
    keys = 'maxindex','plotHough','rank_n','sigmagaussian','Hough_init_sigmas',
            'Hough_init_Threshold','useintensities','tolerancedistance'
    """
    # if Nonee then use default parameter
    if dictparameters == None:
        dictparameters = {}

    TBO, dataselected, gno_dataselected = getOrientationfromDatabase(tth_chi_int,
                                                                     database, **dictparameters)

    if TBO is None:
        print("Image Matching technique has not found any potential orientations!")
        return None

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)


    return bestEULER


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
    TwiceTheta_Chi_Int, nbmax = IOLT.createselecteddata((data_theta * 2, data_chi, data_I),
                                             listofselectedpts, -1)

    if verbose:
        print(" ******   Finding best orientations ****************")
        print("Raw data in %s have %d spots" % (filename_data, len(data_theta)))
        print("Number of selected (or not recongnised) spots from raw data: ", len(listofselectedpts))
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
                            Hough_init_sigmas=(1, .7),
                            Hough_init_Threshold=1,
                            useintensities=0,
                            rank_n=20,
                            sigmagaussian=(1., 1., 1.),
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
        amplitude = .1
        noisefraction = .5

        nb_spots = len(TwiceTheta_Chi_Int[0])

        modified_spots = np.arange(nb_spots)

        np.random.shuffle(modified_spots)

        ind_noisy = modified_spots[:int(noisefraction * nb_spots)]

        tomodify = True * np.ones(nb_spots)

        np.put(tomodify, ind_noisy, False)

        tomodify = np.array(tomodify, dtype=np.bool)

        print("tomodify", tomodify)

        tomodify = np.tile(tomodify, 3).reshape((3, len(tomodify)))

        # add some noise in 2theta chi and intensity (crude manner):
        highlimit = amplitude / 2.
        lowlimit = -amplitude / 2.
        noise = (highlimit - lowlimit) * np.random.random((3, nb_spots)) + lowlimit

        noise = np.where(tomodify, noise, 0)

        print("noise", noise)

        TwiceTheta_Chi_Int += noise


    return getOrientationfromDatabase(TwiceTheta_Chi_Int, DataBase_array,
                                      maxindex=maxindex,
                                    plotHough=plotHough,
                                    rank_n=rank_n,
                                    sigmagaussian=sigmagaussian,
                                    verbose=verbose,
                                    Hough_init_sigmas=Hough_init_sigmas,
                                    Hough_init_Threshold=Hough_init_Threshold,
                                    useintensities=useintensities,
                                    tolerancedistance=tolerancedistance)

def getOrientationfromDatabase(TwiceTheta_Chi, DataBase_array,
                               maxindex=40,
                            plotHough=0,
                            rank_n=20,
                            sigmagaussian=(1., 1., 1.),
                            verbose=0,
                            Hough_init_sigmas=(1, .7),
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
        TTHkept, CHIkept, tokeep = GT.removeClosePoints_2(TwiceTheta_Chi[0], TwiceTheta_Chi[1],
                                             dist_tolerance=tolerancedistance)

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
    bigHoughcollector = ComputeHough(gnomonx, gnomony,
                                     Intensity_table=Intensity_table,
                                     lowresolution=1)

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
        bigHoughcollector = np.where(bigHoughcollector <= Hough_init_Threshold,
                                         0, bigHoughcollector)

#    bigHoughcollector = np.where(bigHoughcollector < 3, 0, bigHoughcollector)

    if plotHough:
        plottab(bigHoughcollector)

    AAA = best_orientations_new_3D(DataBase_array, bigHoughcollector,
                                   maxindex=maxindex,
                                rank_n=rank_n, sigmagaussian=sigmagaussian)

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


def findGrain_in_orientSpace_new_3D(CorrelIntensTab, rank_n=20,
                                    sigmagaussian=(1., 1., 1),
                                    verbose=0):
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

    thraa = np.where(CorrelIntensTab > intensity_rankN, CorrelIntensTab , 0)

    maxtab = NDI.filters.maximum_filter(thraa, size=3)
    ll, nf = NDI.label(maxtab)

    maxpos = np.array(NDI.measurements.maximum_position(maxtab,
                                                        ll,
                                                        np.arange(1, nf + 1)),
                                                        dtype=np.float)

    meanpos = np.array(NDI.measurements.center_of_mass(maxtab,
                                                        ll,
                                                        np.arange(1, nf + 1)),
                                                        dtype=np.float)

    #

#    meanvalues = np.array(NDI.measurements.sum(thraa,
#                                                    ll,
#                                                    np.arange(1, nf + 1)),
#                                                    dtype=np.float)
#
#    meanvalues2 = np.array(NDI.measurements.mean(thraa,
#                                                    ll,
#                                                    np.arange(1, nf + 1)),
#                                                    dtype=np.float)

    maximumvalues = np.array(NDI.measurements.maximum(CorrelIntensTab,
                                                      ll,
                                                      np.arange(1, nf + 1)),
                                                      dtype=np.float)


#    minimumvalues = np.array(NDI.measurements.minimum(CorrelIntensTab,
#                                                      ll,
#                                                      np.arange(1, nf + 1)),
#                                                      dtype=np.float)
#

#    std_values = np.array(NDI.measurements.standard_deviation(CorrelIntensTab,
#                                                      ll,
#                                                      np.arange(1, nf + 1)),
#                                                      dtype=np.float)

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


def best_orientations_new_3D(Database, bigHoughcollector,
                             maxindex=40,
                             rank_n=20, sigmagaussian=(1., 1., 1.)):
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



#--- plot procedures


def plotHough_Exp(prefixname, file_index, nbofpeaks_max, dirname='.'):
    """
    read peak list
    
    High resolution (1 deg step)
    """
    filename_data = prefixname + stringint(file_index, 4) + '.cor'
    print("filename_data", filename_data)

    mydata = ReadASCIIfile(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]),
                                             listofselectedpts, -1)

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
    p.title('raw')
    p.imshow(bigHoughcollector, interpolation='nearest', cmap=GT.SPECTRAL_R)

    p.subplot(122)
    p.title('filtered')
    p.imshow(bigHoughcollectorfiltered, interpolation='nearest', cmap=GT.SPECTRAL_R)

    p.show()

    return bigHoughcollector


def plotHough_Simul(Angles, showplot=1, blur_radius=0.5,
                    emax=25, NB=60, pos2D=0, removedges=2):
    t0 = time.time()

    poshighest1D, Houghfiltered, rawHough = Hough_peak_position_fast(Angles,
                                    Nb=NB, pos2D=pos2D, removedges=removedges,
                                    verbose=0, key_material='Si',
                                    EULER=1,
                                    arraysize=(300, 360),
                                    returnfilterarray=2,
                                    blur_radius=blur_radius,
                                    emax=emax
                                    )

    print("histo", np.histogram(rawHough))

    print("execution time %.3f sec" % (time.time() - t0))
    p.subplot(121)
    p.title('filtered')
    p.imshow(Houghfiltered, interpolation='nearest', cmap=GT.REDS)

#        #indices from an edges removed array
#        pospeak = convertindices1Dto2D(datafiltered.shape, poshighest1D, removedges=2)
#        Y1, X1 = pospeak.T
#        p.scatter(X1, Y1, c='r')

    # indices from an edges unremoved array
    pospeak = np.zeros(2)
    if poshighest1D is not None:
        pospeak = convertindices1Dto2D(Houghfiltered.shape, poshighest1D, removedges=0)
        Y2, X2 = pospeak.T
        p.scatter(X2, Y2, c='g')
    else:
        print("\nSorry! No peaks found!\n")


    p.subplot(122)
    p.title('raw')
    p.imshow(rawHough, interpolation='nearest', cmap=GT.REDS)
    if poshighest1D is not None:
        p.scatter(X2, Y2, c='g')

    p.show()

    return np.fliplr(pospeak), Houghfiltered, rawHough


def plotHough_compare(Angles, filename_data,
                            nbofpeaks_max,
                            dirname='.',
                            EULER=0
                            ):
    """
    plot exp. hough data and simulated ones for Euler angles set
    
    see plotHough_Exp()
    """
    POS2D = 0
    REMOVEDGES = 2
    poshighest1D, datafiltered, rawsimulHough = Hough_peak_position_fast(Angles,
                                    Nb=60, pos2D=POS2D, removedges=REMOVEDGES,
                                    verbose=0,
                                    key_material='Si',
                                    EULER=EULER,
                                    arraysize=(300, 360),
                                    returnfilterarray=2
                                    )

    #  experimental data
    mydata = ReadASCIIfile(os.path.join(dirname, filename_data))
    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))

    length_of_data = len(data_theta)
    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]),
                                             listofselectedpts, -1)

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
    p.imshow(bigHoughcollector, interpolation='nearest', cmap=GT.REDS)
    p.title('experimental')
    p.scatter(X1, Y1, c='b')
    p.subplot(132)
    p.imshow(datafiltered, interpolation='nearest', cmap=GT.REDS)
    p.title('Simulation [%.0f, %.0f, %.0f]' % tuple(Angles))
    p.scatter(X1, Y1, c='b')
    p.subplot(133)
    p.imshow(rawsimulHough, interpolation='nearest', cmap=GT.REDS)
    p.title('raw Simulation [%.0f, %.0f, %.0f]' % tuple(Angles))
    p.scatter(X1, Y1, c='b')

    p.show()


def BrowseHoughCorrel(Database, filename_data,
                            nbofpeaks_max,
                            dirname='.'):
    """
    read peak list, transform it into hough space, make correlation with Database
    
    High resolution (1 deg step)
    """
    mydata = ReadASCIIfile(os.path.join(dirname, filename_data))

    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))
    length_of_data = len(data_theta)

    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]),
                                             listofselectedpts, -1)

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


def plotHoughCorrel(Database, filename_data,
                            nbofpeaks_max,
                            dirname='.',
                            returnCorrelArray=0):
    """
    read peak list, transform it into hough space, make correlation with Database
    
    High resolution (1 deg step)
    """
    mydata = ReadASCIIfile(os.path.join(dirname, filename_data))

    data_theta, data_chi, data_I = mydata
    print("shape(mydata)", np.shape(mydata))
    length_of_data = len(data_theta)

    nb_to_extract = min(nbofpeaks_max, length_of_data)
    listofselectedpts = np.arange(nb_to_extract)
    # selection of points in the three arrays in first argument
    dataselected, nbmax = IOLT.createselecteddata((mydata[0] * 2, mydata[1], mydata[2]),
                                             listofselectedpts, -1)

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

    axes.imshow(array_to_plot, interpolation='nearest')  # , cmap=GT.GREYS)
    numrows, numcols = array_to_plot.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = array_to_plot[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    axes.format_coord = format_coord

    p.show()


def plot2tabs(tab1, tab2):
    """
    function to plot 2 arrays
    """

    p.subplot(121)
    p.imshow(tab1, interpolation='nearest')  # , cmap=GT.GREYS)

    p.subplot(122)
    p.imshow(tab2, interpolation='nearest')  # , cmap=GT.GREYS)

    p.show()


def plot1Darray(tab):
    """
    function to plot 2 arrays
    """

    p.plot(tab)
    p.show()


#--- ------------  image matching DATABASE OLD version

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

    frou = open('lowresFCC_pos.dat', 'r')
    tabdata = pickle.load(frou)
    frou.close()
    tabdata[398] = np.array([  3457, 15474, 15475, 15774, 15775, 28931, 33477, 33478,
            33479, 33777, 33778, 33779, 38477, 51453, 51454, 51455,
            51753, 51754, 51755, 57444, 69426, 69427, 69726, 69727,
            82970, 87422, 87423, 87424, 87722, 87723, 87724, 92624,
        105446, 105447, 105448, 105746, 105747, 105748, 1, 1])
    tabdata[8303] = np.array([  3457, 15474, 15475, 15774, 15775, 28931, 33477, 33478,
            33479, 33777, 33778, 33779, 38477, 51453, 51454, 51455,
            51753, 51754, 51755, 57444, 69426, 69427, 69726, 69727,
            82970, 87422, 87423, 87424, 87722, 87723, 87724, 92624,
        105446, 105447, 105448, 105746, 105747, 105748, 1, 1])
    return np.array(tabdata)


def unpickle_2databases():
    """ Low resolution databank (2 deg step) with 35 indices per orientation
    """
    frou = open('lowres2FCC_pos_0.dat', 'r')
    tabdata_0 = pickle.load(frou)
    frou.close()
    frou = open('lowres2FCC_pos_1.dat', 'r')
    tabdata_1 = pickle.load(frou)
    frou.close()
    return np.concatenate((np.array(tabdata_0), np.array(tabdata_1)))


def unpickle_part_databases(data_index, prefixname='FCC_pos_25keV_', databasefolder='.'):
    """
    -high resolution databank (1 deg step) in EULER angles
    with 35 featuring indices per orientation
    -index of database ranges from 1 to 12
    - last file contains less (16200) than the previous databank (64800 elements)
    """
    Globalname = prefixname + str(data_index) + '.dat'
    # binary mode , data were created with cPickle.dum(...,...,protocol=2
    frou = open(os.path.join(databasefolder, Globalname), 'rb')  # could be 'r' only instead of 'rb' !!
    tabdata = pickle.load(frou)
    frou.close()
    return np.array(tabdata)


def create_database_step1deg(short='No', singleindex=1, databasefolder='.'):
    """ create 1d databank
    729000  *    35 indices  =  25515000 elements
    """
    toreturn = []
    wholerange = list(range(1, 13))
    if short == 'Yes':
        wholerange = (singleindex,)
    for k in wholerange:
        print("loading databank #%d/12" % k)
        toreturn.append(unpickle_part_databases(k, prefixname='FCC_pos_25keV_', databasefolder=databasefolder))
    # print "Databank is now in list called: databankHoughIndices"
    return np.concatenate(toreturn)


def create_database_step1deg_12keV(short='No'):
    """ create 1d databank
    729000  *    35 indices  =  25515000 elements
    """
    toreturn = []
    wholerange = list(range(0, 9))
    if short == 'Yes':
        wholerange = (1,)
    for k in wholerange:
        print("loading databank #%d/8" % k)
        toreturn.append(unpickle_part_databases(k, prefixname='Ref_fcc_pos_'))
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
    for angZ in np.arange(48., 90., 1.):
        print("-------------------------------------")
        print("          angZ", angZ)
        for angY in np.arange(0., 90., 1.):
            print("  angY", angY)
            for angX in np.arange(0., 90., 1.):
                # if (angX%30)==0: print "angX",angX
                nb, pos = Hough_peak_position([angX, angY, angZ], key_material='Si',
                                              returnXYgnomonic=0, arraysize=(300, 360), verbose=0,
                                              EULER=1, saveimages=0, saveposition=0,
                                              prefixname='Ref_H_FCC_')
                largeRefFCCpos.append(pos)
                largeRefFCCnb.append(nb)
                index_ += 1
    tf = time.time()
    print("index", index_)
    print("time ", tf - t0)

    print("nb elem", len(largeRefFCCpos))

    file_pos = open('FCC_pos_7.dat', 'w')
    pickle.dump(largeRefFCCpos, file_pos)
    file_pos.close()

    file_nb = open('FCC_nb_7.dat', 'w')
    pickle.dump(largeRefFCCnb, file_nb)
    file_nb.close()

#--- -----------  image matching DATABASE NEW version

#--- to build
def BuildDataBaseImageMatching_fast(FROMIND, UPTOIND, fileindex):
    """
    building database for image matching indexing
    """

    NBPIXELS_HOUGH = 40
    EMAX = 25
    EULER = 1
    ELEMENT = 'Si'

    index_ = 0

    samplingAngZ = np.linspace(FROMIND, UPTOIND, 1)
    DATABANK = np.zeros((len(samplingAngZ) * 180 * 90, NBPIXELS_HOUGH), dtype=np.uint32)
    print("DATABANK", DATABANK.shape)
    t00 = time.time()
    for angZ in samplingAngZ:
        print("-------------------------------------")
        print("          angZ: %f                   " % angZ)
        print("-------------------------------------")
        for angY in np.arange(0., 180., 1.):
            if (int(angY) % 2 == 0):
                print("  angY ----> ", angY)
            t0 = time.time()
            for angX in np.arange(0., 90., 1.):
#                    if (angX % 30) == 0: print "angX", angX
                pos = Hough_peak_position_fast([angX, angY, angZ], Nb=NBPIXELS_HOUGH,
                                               pos2D=0, removedges=2, blur_radius=0.5,
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

    file_pos = open('Si_REF_%d' % fileindex, 'w')
    pickle.dump(DATABANK, file_pos)
    file_pos.close()


def builddatabaseRange(startAngle, endAngle):
    """
    see builddatabase.py
    
    on crg3 launch on 8 xterms (eight procs)
    by dispatching angle1 angle2 values: from 0 to 89
    
    prompt> /usr/bin/python builddatabase.py angle1 angle2
    """
    for ind in range(startAngle, endAngle + 1):
        BuildDataBaseImageMatching_fast(ind, ind + 1, ind)

#--- to read

def Read_One_DataBaseImageMatching(file_index, prefixname='Si_REF_', databasefolder='.'):
    """
    Read one part of database (internal purpose)
    -high resolution databank (1 deg step) in EULER angles
    with 40 featuring indices per orientation
    -file_index of database ranges from 0 to 89 (every 1 deg in AngZ)
    """
    fullname = prefixname + '%d' % file_index
    # binary mode , data were created with cPickle.dum(...,...,protocol=2
    f = open(os.path.join(databasefolder, fullname), 'rb')  # could be 'r' only instead of 'rb' !!
    tabdata = np.array(pickle.load(f))
    f.close()
    print(tabdata.shape)
    return tabdata


def CreateDataBasestep1deg(databasefolder=None, prefixname='Si_REF_'):
    """
    stack every parts of database (internal purpose)
    create 1d database new version
    90*180*90  *    40 indices  =  58 320 000 elements
    """
    if databasefolder is None:
        databasefolder = os.curdir
    toreturn = []

    for k in range(90):
        print("loading databank #%d/89" % k)
        toreturn.append(Read_One_DataBaseImageMatching(k, prefixname=prefixname,
                                                       databasefolder=databasefolder))

    allDatabase = np.concatenate(toreturn)

    f = open('Si_REF_25keV_40', 'w')
    pickle.dump(allDatabase, f)
    f.close()
    print("Database has been saved in %s" % ('Si_REF_25keV_40'))


def DataBaseImageMatchingSi():
    """
    Read full image matching database file as an array
    read 1d databank of Si 
    90*180*90  *    40 indices  =  58 320 000 elements
    """
    DBname = 'Si_REF_25keV_40'
    dirname = os.path.join(os.curdir, 'ImageMatchingDatabase')
    g = open(os.path.join(dirname, DBname), 'r')
    data = pickle.load(g)
    g.close()
    return data


#--- ----------  TESTs image Matching


def test_speedbuild():
    """
    test to be launched with python (not ipython)
    """

    import profile
    profile.run('Hough_peak_position_fast([12., 51., 37], Nb=40, pos2D=0, removedges=2, key_material="Si", verbose=0, EULER=1)', 'mafonction.profile')
    import pstats
    pstats.Stats('mafonction.profile').sort_stats('time').print_stats()


def show_solution_existence():
    matsol = GT.randomRotationMatrix()
    print("matsol")
    print(matsol)

    detectorparameters = [70, 1024, 1024, 0, 0]
    grainSi = [np.eye(3), 'dia', matsol, 'Si']
    Twicetheta, Chi, Miller_ind, posx, posy, Energy = LAUE.SimulateLaue(grainSi, 5, 25,
                                                                        detectorparameters,
                                                                        removeharmonics=1)
    dataintensity = 20000 * np.ones(len(Twicetheta))

    IOLT.writefile_cor('testmatch', Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)

    from .indexingSpotsSet import getallcubicMatrices
    EULER = 1
    for mat in getallcubicMatrices(matsol):
        if EULER == 0:
            angles = GT.fromMatrix_to_elemangles(mat)
        elif EULER == 1:
            angles = GT.fromMatrix_to_EulerAngles(mat)

        if angles[0] >= 0 and angles[1] >= 0 and angles[2] >= 0:
            print(angles)
            plotHough_compare(angles, 'testmatch.cor', 100000, EULER=EULER)

    bestangles = [37, 7, 89]
    Plot_compare_2thetachi(bestangles, Twicetheta, Chi, verbose=0, key_material='Si', emax=25,
                           EULER=EULER)

def randomdetectorparameters():
    """
    generate random detector parameters
    """
    randomdet = (np.random.rand() - .5) * 2 * 3 + 70.
    randomxcen = (np.random.rand() - .5) * 2 * 300 + 1024
    randomycen = (np.random.rand() - .5) * 2 * 300 + 1024
    randomxbet = (np.random.rand() - .5) * 2 * 2 + 0
    randomxgam = (np.random.rand() - .5) * 2 * 4 + 0

    detectorparameters = [randomdet,
                          randomxcen,
                          randomycen,
                          randomxbet,
                          randomxgam]

    return detectorparameters

def test_ImageMatching(database=None):
    """
    Image Matching Test with single crystal Si randomly oriented
    
    No intensity used
    
    """
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # generate .cor file
    matsol = GT.randomRotationMatrix()
    print("matsol")
    print(matsol)

    detectorparameters = randomdetectorparameters()

#    detectorparameters = [68.2, 920, 880, -0.1, 0.8]

    print("detectorparameters", detectorparameters)

    grainSi = [np.eye(3), 'dia', matsol, 'Si']
    Twicetheta, Chi, Miller_ind, posx, posy, Energy = LAUE.SimulateLaue(grainSi, 5, 25,
                                                                        detectorparameters)
    dataintensity = 20000 * np.ones(len(Twicetheta))

    IOLT.writefile_cor('testSirandom', Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)

    # Find the best orientations and gives a table of results
    TBO, dataselected, gno_dataselected = give_best_orientations_new('testSirandom.cor',
                                                                1000000,
                                                                database,
                                                                dirname=None,
                                                                plotHough=0)



    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    print("bestEuler")
    print(bestEULER)

#    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
#                                                 dirname='.', returnCorrelArray=1)
#    print "min", np.amin(correlintensity)
#    print "max", np.amax(correlintensity)
#    print "mean", np.mean(correlintensity)
#    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)

    # Look at the results: --------------------------------------
#    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
#                            key_material='Si',EULER=1)

    p.close()
    for k in range(len(bestEULER)):
        p.close()
        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
                           verbose=1, key_material='Si', emax=22,
                           EULER=1)
        p.close()


def test_ImageMatching_2(database=None):
    """
    Image Matching Test with two crystal Si randomly oriented
    
    No intensity used
    
    """

    from . import indexingSpotsSet as ISS
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # generate .cor file
    matsol1 = GT.randomRotationMatrix()
    print("matsol1")
    print(matsol1)

    matsol2 = GT.randomRotationMatrix()
    print("matsol2")
    print(matsol2)

#    detectorparameters = [70, 1024, 1024, 0, 0]

    detectorparameters = randomdetectorparameters()

    grainSi1 = [np.eye(3), 'dia', matsol1, 'Si']
    grainSi2 = [np.eye(3), 'dia', matsol2, 'Si']


    Twicetheta1, Chi1, Miller_ind1, posx1, posy1, Energy1 = LAUE.SimulateLaue(grainSi1, 5, 25,
                                                                        detectorparameters)
    dataintensity1 = 20000 * np.ones(len(Twicetheta1))

    Twicetheta2, Chi2, Miller_ind2, posx2, posy2, Energy2 = LAUE.SimulateLaue(grainSi2, 5, 25,
                                                                        detectorparameters)
    dataintensity2 = 20000 * np.ones(len(Twicetheta2))

    Twicetheta, Chi, posx, posy, dataintensity = (np.concatenate((Twicetheta1, Twicetheta2)),
                                                   np.concatenate((Chi1, Chi2)),
                                                   np.concatenate((posx1, posx2)),
                                                   np.concatenate((posy1, posy2)),
                                                   np.concatenate((dataintensity1, dataintensity2)))

    IOLT.writefile_cor('testSirandom', Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)

    sigmagauss = .5
    # Find the best orientations and gives a table of results
    TBO, dataselected, gno_dataselected = give_best_orientations_new('testSirandom.cor',
                                                                1000000,
                                                                database,
                                                                dirname=None,
                                                                plotHough=0,
                                                                rank_n=20,
                                                                sigmagaussian=(sigmagauss, sigmagauss, sigmagauss))


    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    print("bestEuler")
    print(bestEULER)

    twicetheta_data, chi_data, intensity_data = dataselected


    ISS.getStatsOnMatching(bestEULER, twicetheta_data, chi_data, 'Si')

    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
                                                 dirname='.', returnCorrelArray=1)
#    print "min", np.amin(correlintensity)
#    print "max", np.amax(correlintensity)
#    print "mean", np.mean(correlintensity)
#    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)
#
#    # Look at the results:
# #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
# #                            key_material='Si',EULER=1)
#
    p.close()
    for k in range(len(bestEULER)):
        p.close()
        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
                           verbose=1, key_material='Si', emax=25,
                           EULER=1)
        p.close()




    plottab(correlintensity)

#    print "matsol"
#    print matsol.tolist()
#
#    for mat in INDEX.getallcubicMatrices(matsol):
#        angles = GT.fromMatrix_to_EulerAngles(mat)
#        bestangles = np.round(angles)
#        Plot_compare_2thetachi(bestangles, dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=22,
#                           EULER=1)
def test_ImageMatching_ngrains(database=None, nbgrains=5):
    """
    Image Matching Test with nb crystals of Si randomly oriented
    
    No intensity used
    
    """
    from .indexingSpotsSet import getStatsOnMatching, comparematrices
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------


    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    for k in range(nbgrains):
        orientmat = GT.randomRotationMatrix()
#        print "orientmat %d"%k
#        print orientmat.tolist()
        simul_matrices.append(orientmat)

        grains.append([np.eye(3), 'dia', orientmat, 'Si'])

        Twicetheta1, Chi1, Miller_ind1, posx1, posy1, Energy1 = LAUE.SimulateLaue(grains[k], 5, 25,
                                                                        detectorparameters)
        dataintensity1 = 20000 * np.ones(len(Twicetheta1))

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    IOLT.writefile_cor('testSirandom', Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)


    # Find the best orientations and gives a table of results
    SIGMAGAUSS = .5
    TBO, dataselected, gno_dataselected = give_best_orientations_new('testSirandom.cor',
                                                                1000000,
                                                                database,
                                                                dirname=None,
                                                                plotHough=0,
                                                                rank_n=20,
                                                                sigmagaussian=(SIGMAGAUSS, SIGMAGAUSS, SIGMAGAUSS))



    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

#    print "bestEuler"
#    print bestEULER

    twicetheta_data, chi_data, intensity_data = dataselected

    # sort solution by matching rate
    sortedindices, matchingrates = getStatsOnMatching(bestEULER, twicetheta_data, chi_data, 'Si', verbose=0)

#    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
#                                                 dirname='.', returnCorrelArray=1)
#    print "min", np.amin(correlintensity)
#    print "max", np.amax(correlintensity)
#    print "mean", np.mean(correlintensity)
#    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)
#
#    # Look at the results:
# #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
# #                            key_material='Si',EULER=1)
#
    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    verybestmat = GT.fromEULERangles_toMatrix(bestEULER[0])

    grain_indexed = []
    ind_euler = 0
    for eulers in bestEULER:
        bestmat = GT.fromEULERangles_toMatrix(eulers)
        kk = 0
#        print "ind_euler",ind_euler
#        print "eulers",eulers
#        print bestmat.tolist()
        for mat in simul_matrices:
            if comparematrices(bestmat, mat, tol=0.1)[0]:
                print("\nindexation succeeded with grain #%d !! with Euler Angles #%d\n" % (kk, ind_euler))
                angx, angy, angz = eulers
                print("[ %.1f, %.1f, %.1f]" % (angx, angy, angz))
                print("matching rate  ", bestmatchingrates[ind_euler])
                if kk not in grain_indexed:
                    grain_indexed.append(kk)
            kk += 1

        ind_euler += 1

    # summary
    print("nb of grains indexed  :%d" % len(grain_indexed))

    p.close()
#    for k in range(len(bestEULER)):
#        p.close()
#        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=25,
#                           EULER=1)
#        p.close()

    p.close()
    Plot_compare_2thetachi(bestEULER[0], dataselected[0], dataselected[1],
                           verbose=1, key_material='Si', emax=25,
                           EULER=1)



#    plottab(correlintensity)

#    print "matsol"
#    print matsol.tolist()
#
#    for mat in INDEX.getallcubicMatrices(matsol):
#        angles = GT.fromMatrix_to_EulerAngles(mat)
#        bestangles = np.round(angles)
#        Plot_compare_2thetachi(bestangles, dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=22,
#                           EULER=1)


def test_ImageMatching_otherelement(database=None, nbgrains=3):
    """
    Image Matching Test with other element than that used for building the database
    
    No intensity used
    
    """

    from .indexingSpotsSet import (comparematrices, getStatsOnMatching,
                                  initIndexationDict, getIndexedSpots, updateIndexationDict)

    ELEMENT = 'Cu'
    EXTINCTION = 'fcc'
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------


    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    for k in range(nbgrains):
        orientmat = GT.randomRotationMatrix()
#        print "orientmat %d"%k
#        print orientmat.tolist()
        simul_matrices.append(orientmat)

        grains.append([np.eye(3), EXTINCTION, orientmat, ELEMENT])

        Twicetheta1, Chi1, Miller_ind1, posx1, posy1, Energy1 = LAUE.SimulateLaue(grains[k], 5, 25,
                                                                        detectorparameters)
        dataintensity1 = 20000 * np.ones(len(Twicetheta1))

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    IOLT.writefile_cor('testSirandom', Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)

    # create a dictionary of indexed spots
    indexed_spots_dict = initIndexationDict((Twicetheta, Chi, dataintensity, posx, posy))

#    print "indexed_spots_dict", indexed_spots_dict


    # Find the best orientations and gives a table of results
    SIGMAGAUSS = .5
    TBO, dataselected, gno_dataselected = give_best_orientations_new('testSirandom.cor',
                                                                1000000,
                                                                database,
                                                                dirname=None,
                                                                plotHough=0,
                                                                rank_n=20,
                                                                sigmagaussian=(SIGMAGAUSS, SIGMAGAUSS, SIGMAGAUSS))


    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

#    print "bestEuler"
#    print bestEULER

    twicetheta_data, chi_data, intensity_data = dataselected

    # sort solution by matching rate
    sortedindices, matchingrates = getStatsOnMatching(bestEULER, twicetheta_data, chi_data, ELEMENT, verbose=0)

#    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
#                                                 dirname='.', returnCorrelArray=1)
#    print "min", np.amin(correlintensity)
#    print "max", np.amax(correlintensity)
#    print "mean", np.mean(correlintensity)
#    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)
#
#    # Look at the results:
# #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
# #                            key_material='Si',EULER=1)
#
    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)



    grain_indexed = []
    ind_euler = 0
    for eulers in bestEULER:
        bestmat = GT.fromEULERangles_toMatrix(eulers)
        kk = 0
#        print "ind_euler",ind_euler
#        print "eulers",eulers
#        print bestmat.tolist()
        for mat in simul_matrices:
            if comparematrices(bestmat, mat, tol=0.1)[0]:
                print("\nindexation succeeded with grain #%d !! with Euler Angles #%d\n" % (kk, ind_euler))
                angx, angy, angz = eulers
                print("[ %.1f, %.1f, %.1f]" % (angx, angy, angz))
                print("matching rate  ", bestmatchingrates[ind_euler])
                if kk not in grain_indexed:
                    grain_indexed.append(kk)
            kk += 1

        ind_euler += 1

    # summary
    print("nb of grains indexed  :%d" % len(grain_indexed))


    # handle spot indexation for one grain

    verybestmat = GT.fromEULERangles_toMatrix(bestEULER[0])

    indexation_res, nbtheospots = getIndexedSpots(verybestmat,
                       (twicetheta_data, chi_data, intensity_data),
                       ELEMENT,
                       detectorparameters,
                       veryclose_angletol=.5,
                       emin=5,
                       emax=25,
                       verbose=0,
                       detectordiameter=165.)

    nb_of_indexed_spots = len(indexation_res[5])

#    print indexation_res
    print("nb of indexed spots fo this matrix: %d" % nb_of_indexed_spots)

    grain_index = 0
    indexed_spots_dict, nb_updates = updateIndexationDict(indexation_res,
                                                          indexed_spots_dict,
                                                          grain_index)


#    p.close()
#    for k in range(len(bestEULER)):
#        p.close()
#        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=25,
#                           EULER=1)
#        p.close()

    p.close()
    Plot_compare_2thetachi(bestEULER[0], dataselected[0], dataselected[1],
                           verbose=1, key_material=ELEMENT, emax=25,
                           EULER=1)

    return indexed_spots_dict

#    plottab(correlintensity)

#    print "matsol"
#    print matsol.tolist()
#
#    for mat in INDEX.getallcubicMatrices(matsol):
#        angles = GT.fromMatrix_to_EulerAngles(mat)
#        bestangles = np.round(angles)
#        Plot_compare_2thetachi(bestangles, dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=22,
#                           EULER=1)


def test_ImageMatching_dev(database=None):
    """
    Image Matching Test with single crystal Si randomly oriented
    
    No intensity used
    """
    from . import indexingSpotsSet as ISS
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # generate .cor file
    matsol = GT.randomRotationMatrix()
    print("matsol")
    print(matsol)

    detectorparameters = detectorparameters = randomdetectorparameters()
#    detectorparameters = [68.2, 920, 880, -0.1, 0.8]

    print("detectorparameters", detectorparameters)

    grainSi = [np.eye(3), 'dia', matsol, 'Si']
    Twicetheta, Chi, Miller_ind, posx, posy, Energy = LAUE.SimulateLaue(grainSi, 5, 25,
                                                                        detectorparameters)
    dataintensity = 20000 * np.ones(len(Twicetheta))

    IOLT.writefile_cor('testSirandom', Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)

    # Find the best orientations and gives a table of results
    TBO, dataselected, gno_dataselected = give_best_orientations_new('testSirandom.cor',
                                                                1000000,
                                                                database,
                                                                dirname=None,
                                                                plotHough=0)


    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    print("bestEuler")
    print(bestEULER)


#    print dataselected
    twicetheta_data, chi_data, intensity_data = dataselected


    ISS.getStatsOnMatching(bestEULER, twicetheta_data, chi_data, 'Si')

#    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
#                                                 dirname='.', returnCorrelArray=1)
#    print "min", np.amin(correlintensity)
#    print "max", np.amax(correlintensity)
#    print "mean", np.mean(correlintensity)
#    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)

    # Look at the results: --------------------------------------
#    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
#                            key_material='Si',EULER=1)

    p.close()
    for k in range(len(bestEULER)):
        p.close()
        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
                           verbose=1, key_material='Si', emax=25,
                           EULER=1)
        p.close()


def test_ImageMatching_twins(database=None):
    """
    Image Matching Test with other element than that used for building the database
    
    No intensity used
    """
    from . import indexingSpotsSet as ISS
    DictLT.dict_Materials['DIAs'] = ['DIAs', [3.16, 3.16, 3.16, 90, 90, 90], 'dia']

    if 1:

        ELEMENT = 'Si'

        MatchingRate_Threshold = 60  # percent

        SIGMAGAUSS = .5
        Hough_init_sigmas = (1, .7)
        Hough_init_Threshold = 1  # threshold for a kind of filter
        rank_n = 20
        plotHough = 1
        useintensities = 0
        maxindex = 40

    # 1 ,2, 3 grains with diamond structure small unit cell
    if 0:

        ELEMENT = 'DIAs'

        MatchingRate_Threshold = 50  # percent

        SIGMAGAUSS = .5
        Hough_init_sigmas = (1, .7)

        Hough_init_Threshold = -1  # -1 : maximum filter
        rank_n = 40
        plotHough = 0
        useintensities = 0
        maxindex = 40

    # 4 grains with diamond structure small unit cell
    # testSirandom_7.cor   :  zones axes overlap
    # 6 grains with diamond structure small unit cell
    # testSirandom_8.cor   :  zones axes overlap
    if 0:

        ELEMENT = 'DIAs'

        MatchingRate_Threshold = 30  # percent

        Hough_init_sigmas = (1, .7)
        Hough_init_Threshold = -1
        rank_n = 20
        SIGMAGAUSS = .5
        plotHough = 1
        useintensities = 1
        maxindex = 20



    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal

    outputfilename = 'testtwins'
    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    EXTINCTION = DictLT.dict_Materials[ELEMENT][2]

    orientmat = GT.randomRotationMatrix()
#        print "orientmat %d"%k
#        print orientmat.tolist()
    simul_matrices.append(orientmat)

#        grain = CP.Prepare_Grain(ELEMENT, OrientMatrix=orientmat)
#        print "grain", grain
    grains.append([np.eye(3), EXTINCTION, orientmat, ELEMENT])

    grains.append([np.eye(3), EXTINCTION, np.dot(DictLT.dict_Vect['sigma3_1'], orientmat), ELEMENT])

    for k in range(2):

        Twicetheta1, Chi1, Miller_ind1, posx1, posy1, Energy1 = LAUE.SimulateLaue(grains[k], 5, 25,
                                                                        detectorparameters,
                                                                        removeharmonics=1)
        dataintensity1 = (20000 + k) * np.ones(len(Twicetheta1))

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    IOLT.writefile_cor(outputfilename, Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters)

    file_to_index = outputfilename + '.cor'



#    print "indexed_spots_dict", indexed_spots_dict

    # Find the best orientations and gives a table of results

    sigmas = (SIGMAGAUSS, SIGMAGAUSS, SIGMAGAUSS)
    TBO, dataselected, gno_dataselected = give_best_orientations_new(file_to_index,
                                                                1000000,
                                                                database,
                                                                maxindex=maxindex,
                                                                dirname=None,
                                                                plotHough=plotHough,
                                                                Hough_init_sigmas=Hough_init_sigmas,
                                                                Hough_init_Threshold=Hough_init_Threshold,
                                                                useintensities=useintensities,
                                                                rank_n=rank_n,  # 20
                                                                sigmagaussian=sigmas)

    if TBO is None:
        print("Image  Matching has not found any potential orientations!")
        return

    # TODO:to simplify
    alldata, data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(file_to_index)
    Twicetheta = 2.*data_theta

    # create a dictionary of indexed spots
    indexed_spots_dict = ISS.initIndexationDict((Twicetheta, Chi, dataintensity, posx, posy))


    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

#    print "bestEuler"
#    print bestEULER

    twicetheta_data, chi_data, intensity_data = dataselected

    # sort solution by matching rate
    sortedindices, matchingrates = ISS.getStatsOnMatching(bestEULER,
                                                      twicetheta_data, chi_data,
                                                      ELEMENT,
                                                      verbose=0)

    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    print("\nnb of potential grains %d" % len(bestEULER))
    print("bestmatchingrates")
    print(bestmatchingrates)

    bestEULER_0, bestmatchingrates_0 = ISS.filterEquivalentMatrix(bestEULER, bestmatchingrates)


    bestEULER, bestmatchingrates = ISS.filterMatrix_MinimumRate(bestEULER_0, bestmatchingrates_0,
                                                          MatchingRate_Threshold)
    print("After filtering (cubic permutation, matching threshold %.2f)" % MatchingRate_Threshold)
    print("%d matrices remain\n" % len(bestEULER))


    # first indexation of spots with raw (un refined) matrices
    AngleTol = 1.
    dict_grain_matrix = {}
    dict_matching_rate = {}
    (indexed_spots_dict,
     dict_grain_matrix,
     dict_matching_rate) = ISS.rawMultipleIndexation(bestEULER, indexed_spots_dict,
                                                  ELEMENT, detectorparameters,
                                                  AngleTol=AngleTol, emax=25)

    print("dict_grain_matrix", dict_grain_matrix)

    if dict_grain_matrix is not None:
        ISS.plotgrains(dict_grain_matrix, ELEMENT, detectorparameters, 25,
                            exp_data=dataselected)
    else:
        print("plot the best orientation matrix candidate")
        Plot_compare_2thetachi(bestEULER_0[0], dataselected[0], dataselected[1],
                           verbose=1, key_material=ELEMENT, emax=25,
                           EULER=1)

        Plot_compare_gnomondata(bestEULER_0[0], dataselected[0], dataselected[1],
                           verbose=1, key_material=ELEMENT, emax=25,
                           EULER=1)
        return


    # ---    refine matrix
    print("\n\n Refine first matrix")

    grain_index = 0
#    print "initial matrix", dict_grain_matrix[grain_index]
    refinedMatrix, devstrain = ISS.refineUBSpotsFamily(indexed_spots_dict, grain_index,
                        dict_grain_matrix[grain_index], ELEMENT, detectorparameters,
                        use_weights=1)

#    print "refinedMatrix"
#    print refinedMatrix

    if refinedMatrix is not None:

        AngleTol = .5
        # redo the spots links
        indexation_res, nbtheospots = ISS.getIndexedSpots(refinedMatrix,
                                   (twicetheta_data, chi_data, intensity_data),
                                   ELEMENT,
                                   detectorparameters,
                                   removeharmonics=1,
                                   veryclose_angletol=AngleTol,
                                   emin=5,
                                   emax=25,
                                   verbose=0,
                                   detectordiameter=165.)

#        print 'nb of links', len(indexation_res[1])

        if indexation_res is None:
            return

        indexed_spots_dict, nb_updates = ISS.updateIndexationDict(indexation_res,
                                                              indexed_spots_dict,
                                                              grain_index,
                                                              overwrite=1)

        print("with refined matrix")
        print("nb of indexed spots for this matrix # %d: %d / %d" % (grain_index, nb_updates, nbtheospots))
        print("with tolerance angle : %.2f deg" % AngleTol)

        dict_grain_matrix[grain_index] = refinedMatrix
        dict_matching_rate[grain_index] = [nb_updates, 100.*nb_updates / nbtheospots]

        ISS.plotgrains(dict_grain_matrix, ELEMENT, detectorparameters, 25,
                            exp_data=ISS.getSpotsData(indexed_spots_dict)[:, 1:3].T)

        # one more time with less tolerance in spotlinks

        # ---    refine matrix
        print("\n\n Refine first matrix")

        grain_index = 0
#        print "initial matrix", dict_grain_matrix[grain_index]
        refinedMatrix, devstrain = ISS.refineUBSpotsFamily(indexed_spots_dict, grain_index,
                            dict_grain_matrix[grain_index], ELEMENT, detectorparameters,
                            use_weights=1,
                            pixelsize=165. / 2048,
                            dim=(2048, 2048))

#        print "refinedMatrix"
#        print refinedMatrix

        if refinedMatrix is None:
            return

        AngleTol = .1
        indexation_res, nbtheospots = ISS.getIndexedSpots(refinedMatrix,
                                   (twicetheta_data, chi_data, intensity_data),
                                   ELEMENT,
                                   detectorparameters,
                                   removeharmonics=1,
                                   veryclose_angletol=AngleTol,
                                   emin=5,
                                   emax=25,
                                   verbose=0,
                                   detectordiameter=165.)

#        print 'nb of links', len(indexation_res[1])
        if indexation_res is None:
            return

        indexed_spots_dict, nb_updates = ISS.updateIndexationDict(indexation_res,
                                                              indexed_spots_dict,
                                                              grain_index,
                                                              overwrite=1)

        print("with refined matrix")
        print("nb of indexed spots for this matrix # %d: %d / %d" % (grain_index, nb_updates, nbtheospots))
        print("with tolerance angle : %.2f deg" % AngleTol)

        dict_grain_matrix[grain_index] = refinedMatrix
        dict_matching_rate[grain_index] = [nb_updates, 100.*nb_updates / nbtheospots]

        ISS.plotgrains(dict_grain_matrix, ELEMENT, detectorparameters, 25,
                            exp_data=ISS.getSpotsData(indexed_spots_dict)[:, 1:3].T)


    return indexed_spots_dict, dict_grain_matrix


def test_ImageMatching_index(database=None, nbgrains=3, readfile=None):
    """
    Image Matching Test with other element than that used for building the database

    No intensity used
    """
    from .indexingSpotsSet import (createFakeData,
                                  initIndexationDict,
                                  rawMultipleIndexation,
                                  plotgrains,
                                  filterEulersList)

    DictLT.dict_Materials['DIAs'] = ['DIAs', [3.16, 3.16, 3.16, 90, 90, 90], 'dia']
    firstmatchingtolerance = 1.
    nb_of_peaks = 1000000  # all peaks are used in hough transform for recognition
    tolerancedistance = 0  # no close peaks removal
    maxindex = 40
    useintensities = 0
    plotHough = 0

    emax = 25
    if 1:

        key_material = 'Si'

        MatchingRate_Threshold = 60  # percent

        SIGMAGAUSS = .5
        Hough_init_sigmas = (1, .7)
        Hough_init_Threshold = 1  # threshold for a kind of filter
        rank_n = 20
        useintensities = 0

        dictimm = {'sigmagaussian':(0.5, 0.5, 0.5),
                   'Hough_init_sigmas' : (1, .7),
                   'Hough_init_Threshold':1,
                   'rank_n':20,
                   'useintensities':0}

    # 1 ,2, 3 grains with diamond structure small unit cell
    if 0:

        key_material = 'DIAs'

        MatchingRate_Threshold = 50  # percent

        SIGMAGAUSS = .5
        Hough_init_sigmas = (1, .7)

        Hough_init_Threshold = -1  # -1 : maximum filter
        rank_n = 40
        plotHough = 0
        useintensities = 0
        maxindex = 40

    # 4 grains with diamond structure small unit cell
    # testSirandom_7.cor   :  zones axes overlap
    # 6 grains with diamond structure small unit cell
    # testSirandom_8.cor   :  zones axes overlap
    if 0:

        key_material = 'DIAs'

        MatchingRate_Threshold = 15  # percent

        Hough_init_sigmas = (1, .7)
        Hough_init_Threshold = 1
        rank_n = 40
        SIGMAGAUSS = .5
        plotHough = 1
        useintensities = 0
        maxindex = 40
        firstmatchingtolerance = .5
        tolerancedistance = 1.
#        nb_of_peaks = 100


    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'test%srandom' % key_material
        file_to_index = createFakeData(key_material, nbgrains, outputfilename=outputfilename)
    else:
        file_to_index = readfile


    # read spots data: 2theta, chi, intensity
    TwiceTheta_Chi_Int = readnselectdata(file_to_index, nb_of_peaks)

    twicetheta_data, chi_data = TwiceTheta_Chi_Int[:2]
    # TODO:to simplify
    alldata, data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(file_to_index)
    Twicetheta = 2.*data_theta

    # create an initial dictionary of indexed spots
    indexed_spots_dict = initIndexationDict((Twicetheta, Chi, dataintensity, posx, posy))

    # Find the best orientations and gives a table of results

    bestEULER = bestorient_from_2thetachi((Twicetheta, Chi), database, dictparameters=dictimm)

    bestEULER, bestmatchingrates = filterEulersList(bestEULER, (Twicetheta, Chi), key_material, emax,
                                                    rawAngularmatchingtolerance=firstmatchingtolerance,
                                                    MatchingRate_Threshold=MatchingRate_Threshold,
                                                    verbose=1)

    # first indexation of spots with raw (un refined) matrices
    AngleTol = 1.
    dict_grain_matrix = {}
    dict_matching_rate = {}
    (indexed_spots_dict,
     dict_grain_matrix,
     dict_matching_rate) = rawMultipleIndexation(bestEULER, indexed_spots_dict,
                                                  key_material, detectorparameters,
                                                  AngleTol=AngleTol, emax=emax)

    print("dict_grain_matrix", dict_grain_matrix)

    if dict_grain_matrix is not None:
        plotgrains(dict_grain_matrix, key_material, detectorparameters, emax,
                            exp_data=(Twicetheta, Chi))
    else:
        print("plot the best orientation matrix candidate")
        Plot_compare_2thetachi(bestEULER[0], twicetheta_data, chi_data,
                           verbose=1, key_material=key_material, emax=emax,
                           EULER=1)

        Plot_compare_gnomondata(bestEULER[0], twicetheta_data, chi_data,
                           verbose=1, key_material=key_material, emax=emax,
                           EULER=1)
        return


# ------------  Main            ------------------------------

if __name__ == '__main__':

    database = DataBaseImageMatchingSi()

    test_ImageMatching_index(database=database, nbgrains=3)
    test_ImageMatching_index(database=database, nbgrains=3, readfile='testSirandom.cor')


#    test_speedbuild() # to be used only with python (not ipython)

#
#    plotHough_Simul([5, 37, 9])
#    plotHough_Exp('dat_Ge', 1, 100, col_I=4, dirname='./Examples/Ge/')

#    plotHough_compare([20, 10, 50], 'euler201050_', 0, 100)

#    position = Hough_peak_position_fast([0, 0, 0], Nb=60,
#                                               pos2D=0, removedges=2, blur_radius=0.5,
#                                               key_material='Si',
#                                                arraysize=(300, 360),
#                                                verbose=0,
#                                                EULER=1,
#                                                returnfilterarray=0)

#    pos, fH, raw = plotHough_Simul([20, 10, 50], emax=25, NB=60, pos2D=0, removedges=2)

#    plotHough_Exp('euler201050_', 2, 10000000)
#    pos, fH, raw = plotHough_Simul([20, 10, 50], emax=25, NB=80, pos2D=0, removedges=2)

#    plotHough_compare([20, 10, 50], 'euler201050_0002.cor', 100000)

#    plotHough_compare([20, 10, 50], 'testmatch.cor', 100000)








#    correlintensity, argcorrel = plotHoughCorrel(database, 'euler201050_0002.cor',
#                                                 100000,dirname='.', returnCorrelArray=1)
#
#    plottab(correlintensity)



    """
    # loading the best_orientation for a serie of file (containing 2theta, chi ,I and intensity sorted)
    # list of element: [index_file,array of 3 Euler Angles (sorted by correlation degree)]
    filou=open('best_orient_UO2_He_103to1275_300p','r')
    TBO=pickle.load(filou)
    filou.close
    print "Table of best orientation loaded in TBO"
    print "contains %d elements"%len(TBO)
    # each element in TBO has:
    #[0] file index
    #[1]  has :
    #   [0] array of three Euler angles (sorted by correlation importance)
    #   [1] (2theta array,chi array, Intensity array(possibly))
    # tip: transpose(TBO[index_in_TBO][1][0])[orient_rank] are the 3 Euler Angles of orientation ranled by orient_rank

    Very_Angles=[[] for k in range(len(TBO))]
    #Decision_threshold=0.0023 # 1000p
    Decision_threshold=0.003 # 300p 0.8 deg et 20 keV
    verbose=0

    for index_in_TBO in range(0,len(TBO)): # index of the file in table of best orientation frame (from 0 to len(TBO)-1)
        std_res=1
        nb_of_spots=100
        orient_rank=0
        nb_orient=len(transpose(TBO[index_in_TBO][1][0]))
        print "-------------------------------------"
        print "File index",index_in_TBO
        print "Nb of exp.spots",1000 # nb of exp spots used for finding the 20 first orientations
        print "nb of orientations",nb_orient
        while (orient_rank<nb_orient): # threshold to pursue the search of a correct orientation (good catch)
        #while (1.*std_res/nb_of_spots>=0.0023) and (nb_of_spots>=10) and (orient_rank<nb_orient): # threshold to pursue the search of a correct orientation (good catch)
        #while (std_res>=0.08) and (nb_of_spots>=10) and (orient_rank<nb_orient): # threshold to pursue the search of a correct orientation (good catch)
            if verbose:
                print "-------------------------------------"
                print "orient_rank",orient_rank
                print "EULER angles",transpose(TBO[index_in_TBO][1][0])[orient_rank]
            john=StickLabel_on_exp_peaks(transpose(TBO[index_in_TBO][1][0])[orient_rank],
                                        TBO[index_in_TBO][1][1][0],TBO[index_in_TBO][1][1][1],
                                        0.8,
                                        20,
                                        verbose=0,
                                        key_material='UO2',
                                        arraysize=(300,360),
                                        EULER=1)

            residues_angles=array(john[-2].values())
            mean_res_1=mean(residues_angles)
            maxi_res_1=max(residues_angles)
            std_res_1=std(residues_angles)
            nb_of_spots_1=len(residues_angles)

            if verbose:
                print "max ",maxi_res_1," mean ",mean_res_1," std_res",std_res_1," nb_of_spots",nb_of_spots_1
                print "std/nbspots",1.*std_res_1/nb_of_spots_1
            #removing far spots
            anormalous_spots_key=[]
            for k,v in john[-2].items():
                if v>=mean_res_1+1.5*sqrt(std_res_1):
                    anormalous_spots_key.append(k)
            for cle in anormalous_spots_key:
                print john[-1][cle]
                del john[-2][cle]
                del john[-1][cle]

            residues_angles=array(john[-2].values())
            mean_res=mean(residues_angles)
            maxi_res=max(residues_angles)
            std_res=std(residues_angles)
            nb_of_spots=len(residues_angles)
            if verbose:
                print "max ",maxi_res," mean ",mean_res," std_res",std_res," nb_of_spots",nb_of_spots
                print "std/nbspots",1.*std_res/nb_of_spots

            #print "cond",((std_res>=0.08) and (nb_of_spots>=10) and (orient_rank<20))
            if ((1.*std_res/nb_of_spots>=Decision_threshold)  and (nb_of_spots>=10) and (orient_rank<nb_orient))==False:
            #if ((std_res>=0.08) and (nb_of_spots>=10) and (orient_rank<nb_orient))==False:
                if verbose:
                    print "********-----------------------------------------"
                    print "Good! I ve found an orientation!"
                    print "********-----------------------------------------"
                Very_Angles[index_in_TBO].append(transpose(TBO[index_in_TBO][1][0])[orient_rank])


            orient_rank+=1
        print "Number of found orientations",len(Very_Angles[index_in_TBO])



        orient_EULER_angles_rank=orient_rank-1
        if nb_orient==orient_rank: # use avec while (1.*std_res/nb_of_spots>=0.0023) and (nb_of_spots>=10) and (orient_rank<nb_orient)
            #print "display by default the best found in TBO"
            orient_EULER_angles_rank=0

        # plot only exp labelled spots


        Plot_compare_2thetachi(transpose(TBO[index_in_TBO][1][0])[orient_EULER_angles_rank],
                            TBO[index_in_TBO][1][1][0],TBO[index_in_TBO][1][1][1],
                            verbose=1,key_material='UO2',EULER=1,
                            exp_spots_list_selection=array(john[-1].values()))

        # plot all exp spots

        Plot_compare_2thetachi(transpose(TBO[index_in_TBO][1][0])[orient_EULER_angles_rank],
                            TBO[index_in_TBO][1][1][0],TBO[index_in_TBO][1][1][1],
                            verbose=1,key_material='UO2',EULER=1,exp_spots_list_selection=None)


        # Working now only with remaining exp spots (those not labelled in previous sticking)
        #give_best_orientations('sUrHe',TBO[index_in_TBO][0],300, listselection=######) # 300 = most intense peaks
    """



    """

    # building the map
    # nb elem = 51*23

    list_of_pointsinstance=[]
    list_of_signs=[]
    dic_ID_orient={}
    grain_counter=0
    for m in range(51*23):
        XX=m/51
        YY=m%51
        p=Map_point([XX,YY])
        p.orients=refine_orient(Main_Orient[m],array([3,3,3]))
        list_of_pointsinstance.append(p)

        nb_orient=len(p.orients)
        #print m,nb_orient
        if nb_orient>0:
            point_sign=[]
            for g in range(nb_orient):
                #print g,list(p.orients[g])
                signature_g=from3angles_to_int(list(p.orients[g]))
                point_sign.append(signature_g)
                if signature_g not in dic_ID_orient.values():
                    dic_ID_orient[grain_counter]=signature_g
                    grain_counter+=1
            list_of_signs.append(point_sign)
            p.orients_ID=point_sign
        else:
            list_of_signs.append([-1])
            p.orients_ID=[-1]

    print "I ve found %d grain in the map"%len(dic_ID_orient)
    #adding -1 = None
    dic_ID_orient[-1]=-1
    print dic_ID_orient

    print "Now put the general ID grains in all the point of the map"
    # inverser le dictionnaire...
    def invdict(adict):

        return dict([(val, key) for key, val in adict.items()])

    dic_orient_ID=invdict(dic_ID_orient)

    # lists grain id for each points
    grain_per_point=[map(lambda elem: dic_orient_ID[elem],ls) for ls in list_of_signs]

    # gives for each grain ID, the point instances where they were found ----------------------------
    localisation_grain={}
    for ID in dic_ID_orient.keys():
        localisation_grain[ID]=[]
    for p in list_of_pointsinstance:
        map(lambda cle_ID: localisation_grain[dic_orient_ID[cle_ID]].append(p.pos),p.orients_ID)
    table_pos=[array(localisation_grain[ID]) for ID in dic_ID_orient.keys()]

    """



    # -----------------------------------------------------------------------------------------------
    def make_colormarkerdict():
        """
        build a dictionary of color and marker to distinguish contribution of grains in scatter plot
        """
        # faudrait rendre aleatoire l'indice i_dict pour un meilleur rendu
        colorlist = ['k', (1.0, 0.0, 0.5), 'b', 'g', 'y', 'r', 'c', 'm', '0.75', (.2, .2, .2), (0.0, 0.5, 1.0)]
        markerlist = ['s', 'v', 'h', 'd', 'o', '^', '8']
        colormarkerdict = {}
        i_dict = 0
        for col in colorlist:
            for mark in markerlist:
                colormarkerdict[i_dict] = [col, mark]
                i_dict = i_dict + 1
        return colormarkerdict


    def plot_mapmajor(listoflocations, xmin, xmax, ymin, ymax):
        """
        example of map plot given list of coordinates ?? ( I do not remember actually)
        
        """
        p.close()
        colormarkerdict = make_colormarkerdict()

        tableofxcoord = []
        tableofycoord = []

        # move the last elem in first position the first grain ID corresponds now to unknown grain
        listoflocations.insert(0, listoflocations.pop(-1))

        for elem in (listoflocations[0],
                     listoflocations[8],
                     listoflocations[16],
                      listoflocations[24],
                       listoflocations[32]):
            tableofxcoord.append(elem[:, 0])
            tableofycoord.append(elem[:, 1])

        # print "tableofxcoord",tableofxcoord

        p.figure(figsize=(10, 7), dpi=80)

        kk = 0
        for xdata, ydata in zip(tuple(tableofxcoord), tuple(tableofycoord)):
            p.scatter(ydata, xdata, c=colormarkerdict[kk][0], marker=colormarkerdict[kk][1], s=500, alpha=0.5)
            # permutation of x y / array index ?...
            kk = kk + 1


        ticks_x = np.arange(xmin, xmax, 1)
        ticks_y = np.arange(ymin, ymax, 1)
        p.yticks(ticks_x)  # permutation of x y / array index ?...
        p.xticks(ticks_y)
        p.ylim(-1, xmax + 1)
        p.xlim(-1, ymax + 1)
        p.xlabel('x', fontsize=20)
        p.ylabel('y', fontsize=20)
        p.title('UO2_He_103_1275: Grain map. Unknown grains are black squares')
        p.grid(True)
        p.show()

    # plot_mapmajor(table_pos,0,22,0,50)



#    def corro(central_angles, deltaangle):
#        d_angle = np.array(deltaangle)
#        elemangle = np.array(central_angles) + d_angle
#        #elemangle=array(fromMatrix_to_elemangles(mamasol))
#        #elemangle=-array(fromMatrix_to_elemangles(dot(array(mamasol),permu5)))
#        print elemangle
#
#        nb, pos = Hough_peak_position(list(elemangle), key_material=29, returnXYgnomonic=0, arraysize=(300, 360), verbose=0, saveimages=0, saveposition=0, EULER=1, prefixname='Ref_H_FCC_')
#        #scatter(xgnomon,ygnomon,c='r') #theory
#        #scatter(gnomonx,gnomony,c='b',alpha=.5,s=40) #data
#        #show()
#
#        #sumy=np.sum(blurredHough[pos])
#        #sumy = np.sum(filteredHough[pos])
#        #print sumy
#        return sumy

    # corro([20.69,-11.23,9.26],[0,0,0])

    # intcorrel=[[    max([corro([20.69,-11.23,9.26],[l,j,k]) for l in arange(-10,10,1)])    for k in arange(-2,3,1)] for j in arange(-2,3,1)] #result optim_1
    # intcorrel=[corro([20.69,-11.23,9.26],[l,-1.,0]) for l in arange(-10,10,1)] # =>



    # 1024 pts et angle d'EULER
    # intcorrel=[[    max([corro([0,0,0],[l,j,k]) for l in arange(0,90,1)])    for k in arange(0,90,2)] for j in arange(0,90,2)]
    # intcorrel=[  corro([0,74,46],[k,0,0])  for k in arange(0,90,1.)] #=> k = 68
    # intcorrel=[  corro([0,80,18],[k,0,0])  for k in arange(0,90,1.)] #=> k = 56


    # 1024 pts et angle d'EULER

    # intcorrel=[[    max([corro([0,74,46],[l,j,k]) for l in arange(0,90,.5)])    for k in arange(-1,2,.5)] for j in arange(-1,2,.5)]


    """
    from np.random import *
    totalsizeHougharray=432000
    nboforientations=45*45*360/1.
    t0=time.time()
    bigpos=random_integers(0,totalsizeHougharray-1,(nboforientations,100))
    tr=time.time()
    print "time array creation",tr-t0
    john=map(lambda elem: np.sum(filteredHough[elem]),bigpos)

    tf=time.time()
    print size(bigpos)
    print "nboforientations",nboforientations
    print "time",tf-tr
    """


    """ VERY LONG
    def mysum(arra):
        if len(arra)>0:
            return np.sum(arra)
        else:
            return 0

    rhovalue=1.
    toleranceonrho=.1
    RHO=linspace(-1.05,1.05,42)
    ANGLE=linspace(0.,2*np.pi,100)
    angleflat=ANGLE.flat
    shapearray=shape(RHO)
    ACCum=np.zeros((42,100))
    i_accum=0
    for _rho in RHO.flat:
        print "i_accum =",i_accum
        j_accum=0
        for _angle in angleflat:
            #print "_angle",_angle
            #print "j_accum",j_accum
            summy=mysum(intensitytable[np.where(abs(xgnomon*cos(_angle)+ygnomon*sin(_angle)-_rho)<.005)])
            #print "summy",summy
            ACCum[i_accum,j_accum]=summy
            j_accum+=1
        i_accum+=1

    glu=open('Hough','r')
    import pickle
    glot=pickle.load(glu)
    glu.close()

    APRES COUP: rho est de maniere interessante entre bin 8 et 35
    """


