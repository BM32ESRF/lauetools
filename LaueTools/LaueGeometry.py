# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Module of lauetools project to compute Laue spots position on CCD camera.
It handles detection and source geometry. 

JS Micha August 2014

http://sourceforge.net/projects/lauetools/

*Outgoing scattered beam  computations*
    - **q** momentum transfer vector from resp. incoming and outgoing wave vector **ki** and **kf**:
    :math:`q=kf-ki`

    - When a Laue spot exists, **q** is equal to the one node of the reciprocal lattice given by **G*** vector
    **G*** is perpendicular to atomic planes defined by the three Miller indices h,k,l such as:
    **G***=h**a*** + k**b*** +l**c*** where **a***, **b***, and **c*** are the unit cell lattice basis vectors.

    - **kf**: scattering vector whose corresponding unit vector is **uf**
    - **ki** incoming beam vector, **ui** corresponding unit vector

* Laboratory Frame LT2:
    - I: origin
    - z vertical up perpendicular to CCD plane (top camera geometry)
    - y along X-ray horizontal
    - x towards wall behind horizontal
    - O: origin of pixel CCD frame in detecting plane
    - **j** // **ui** incoming beam unit vector
    - z axis is defined by the CCD camera position. z axis is perpendicular to CCD plane
    such as IO belongs to the plane Iyz
    - bet: angle between **IO** and **k**
    - **i**= **j**^**k** (when entering the BM32 hutch **i** is approximately towards the wall
    (in CCD on top geometry and beam coming from the right)
    - M: point lying in CCD plane corresponding to Laue spot
    - **uf** is the unit vector relative to vector **IM**

**kf** is also a vector collinear to **IM** with a length of R=1/wavelength=E/12.398 [keV]
with wavelength and Energy of the corresponding bragg's reflections.

I is the point from which calibration parameters (CCD position) are deduced (from a perfectly known crystal structure Laue pattern) 
Iprime is an other source of emission (posI or offset in functions)

2theta is the scattering angle between **ui** and **uf**: :math:`cos(2\the)=ui.uf`

**kf**=( -sin 2theta sin chi,cos 2theta  , sin 2theta cos chi)
**ki** = ( 0, 1, 0)
Energy= 12.398*  q**2/(2* **q**.**ui**)=12.398 * q**2/ (-2 sin theta) 

*Calibration paramters (CCD position and detection geometry)*
    - calib: list of the 5 calibration parameters [dd,xcen,ycen,xbet,xgam]
    - dd: norm of **IO**  [mm]
    - xcen,ycen [pixel unit]: pixels values in CCD frame of point O with respect to Oprime where
    Oprime is the origin of CCD pixels frame (at a corner of the CCD array)
    - xbet: angle between **IO** and **k** [degree]
    - xgam: azimutal rotation angle around z axis. Angle between CCD array axes
    and (**i**,**j**) after rotation by xbet [degree].

*sample frame*

Origin is I and unit frame vectors (**is**,**js**,**ks**) are derived
from absolute frame by the rotation (axis= -i, angle= wo) where wo is the angle between **js** and **j**
"""

__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import sys, os

import numpy as np
import pylab as P

import findorient as FindO
import generaltools as GT
import LaueGeometry as LTGeo
import CrystalParameters as CP

import dict_LaueTools as DictLT

# -----------  CONSTANTS ------------------
RECTPIX = DictLT.RECTPIX  # see above  camera skewness

PI = np.pi
DEG = PI / 180.

CST_CONV_LAMBDA_KEV = DictLT.CST_ENERGYKEV

# sign of CCD camera angle =1 to mimic XMAS convention
SIGN_OF_GAMMA = 1

#--- -----   old function  ---------------
norme = GT.norme_vec

#--- -------- geometrical functions relating 2theta, chi, pixel X, pixel Y, detector plane ---- 
def calc_uflab(xcam, ycam, CCDcalibrationparameters,offset=0,
               returnAngles=1,
               verbose=0,
               pixelsize=165. / 2048,
               signgam=SIGN_OF_GAMMA,
               rectpix=RECTPIX,
               kf_direction='Z>0'):
    r"""
    compute 2theta and chi scattering angles or scattered unit vector uf and kf vectors
    from lists of X and Y Laue spots pixels positions on detector

    :param xcam: list of pixel X position
    :type xcam: list of floats
    :param ycam: list of pixel Y position
    :type ycam: list of floats
    :param calib: list of 5 calibration parameters
    
    :param offset: float, offset in position along incoming beam of source of scattered rays
                if positive: offset in sample depth
                units: mm

    :returns:
        - if returnAngles=1   : twicetheta, chi   *(default)*
        - if returnAngles!=1  : uflab, IMlab
    """
    calib = CCDcalibrationparameters[:5]
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.
#    print "pixelsize in calc_uflab ", pixelsize

    # transmission geometry
    if kf_direction in ('X>0',):
        return calc_uflab_trans(xcam, ycam, calib,
               returnAngles=returnAngles,
               verbose=verbose,
               pixelsize=pixelsize,
               signgam=signgam,
               rectpix=rectpix)
    # 2theta=90 deg reflection geometry (top side+ and side -)
    elif kf_direction in ('Z>0', 'Y>0', 'Y<0'):
        cosbeta = np.cos(PI / 2. - xbet * DEG)
        sinbeta = np.sin(PI / 2. - xbet * DEG)

    else:
        raise ValueError("kf_direction = %s not implemented in calc_uflab" % \
                                            str(kf_direction))

    cosgam = np.cos(-signgam * xgam * DEG)
    singam = np.sin(-signgam * xgam * DEG)

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I impact point on sample (location of x-ray scattering or emission)
    # O centre of origin of pixel CCD
    # M belong to CCD plane
    # IM is parallel to kf
    # frame is not Lauetools'one
    # since here:  y // ki

    # IOlab = detect * array([0.0, cosbeta, sinbeta])
    xO, yO, zO = detect * np.array([0.0, cosbeta, sinbeta])

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = -yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1. * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame
    # print nIMlab #distance source pt on CCD (mm)

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)
    
#     print "uflab w/o source offset",uflab
    
    if offset not in (None, 0,0.0):
        # with source offset along y (>0 if along the beam and in sample depth)
        #ufprimelab = unit(IpMlab) = unit(IpIlab+IMlab)
        IpMlab=np.array([0,offset,0])+IMlab
        normedIpM = np.sqrt(np.sum(IpMlab**2,axis=1)).reshape((len(IpMlab),1))
        
        ufprime = 1.*IpMlab/normedIpM
        
        print("ufprime, uflab with source offset",ufprime)
        
        uflab=ufprime

    # calculus of scattering angles
    EPS = 1E-17
    chi = np.arctan(-uflab[:, 0] / (uflab[:, 2] + EPS)) / DEG  # JSM convention
#     chiXMAS = np.arctan(uflab[:, 0] / np.sqrt(uflab[:, 1] ** 2 + uflab[:, 2] ** 2)) / DEG
#     chiXMAS2 = np.arctan(np.sqrt(uflab[:, 0] ** 2 + uflab[:, 1] ** 2) / uflab[:, 2]) / DEG

    twicetheta = np.arccos(uflab[:, 1]) / DEG

    if verbose:
        print("chi_JSM", chi)
#         print "chi_XMAS", chiXMAS
#         print "chi_XMAS2", chiXMAS2
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi


def calc_uflab_trans(xcam, ycam, calib,
               returnAngles=1,
               verbose=0,
               pixelsize=165. / 2048,
               signgam=SIGN_OF_GAMMA,
               rectpix=RECTPIX):
    r"""
    compute 2theta and chi scattering angles or uf and kf vectors
    from lists of X and Y Laue spots positions
    in TRANSMISSION geometry

    :param xcam: list of pixel X position
    :type xcam: list of floats
    :param ycam: list of pixel Y position
    :type ycam: list of floats
    :param calib: list of 5 calibration parameters

    :returns:
        - if returnAngles=1   : twicetheta, chi   *(default)*
        - if returnAngles!=1  : uflab, IMlab
        
    # TODO: add offset like in reflection geometry
    """
    print('transmission GEOMETRY')
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.
#    print "pixelsize in calc_uflab ", pixelsize

    cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)

    cosgam = np.cos(-signgam * xgam * DEG)
    singam = np.sin(-signgam * xgam * DEG)

    xcam1 = (np.array(xcam) - xcen) * pixelsize
    ycam1 = (np.array(ycam) - ycen) * pixelsize * (1.0 + rectpix)

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    # I centre
    # O centre of origin of pixel CCD
    # M belong to CCD plane
    # IM is parallel to kf

    # IOlab = detect * array([0.0, cosbeta, sinbeta])
    xO, yO, zO = detect * np.array([0.0, cosbeta, sinbeta])

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xOM = xca0
    yOM = yca0 * sinbeta
    zOM = -yca0 * cosbeta

    # IMlab = IOlab + OMlab
    xM = xO + xOM
    yM = yO + yOM
    zM = zO + zOM
    IMlab = np.array([xM, yM, zM]).T

    # norm of IM vector
    # nIMlab=sqrt(dot(IMlab,IMlab))
    nIMlab = 1. * np.sqrt(xM ** 2 + yM ** 2 + zM ** 2)

    # print transpose(array([xM,yM,zM])) # vector joining source and pt on CCD in abs frame
    # print nIMlab #distance source pt on CCD (mm)

    uflab = np.transpose(np.array([xM, yM, zM]) / nIMlab)
    # print "uflab",uflab
    EPS = 1E-17

    print("transmission mode ", -uflab[:, 2], (uflab[:, 0] + EPS))

    chi = np.arctan2(-xM, zM) / DEG

#     chiXMAS = np.arctan(uflab[:, 0] / np.sqrt(uflab[:, 1] ** 2 + uflab[:, 2] ** 2)) / DEG
#     chiXMAS2 = np.arctan(np.sqrt(uflab[:, 0] ** 2 + uflab[:, 1] ** 2) / uflab[:, 2]) / DEG

    twicetheta = np.arccos(uflab[:, 1]) / DEG

    if verbose:
        print("chi_JSM", chi)
#         print "chi_XMAS", chiXMAS
#         print "chi_XMAS2", chiXMAS2
        print("2theta", twicetheta)

    if returnAngles != 1:
        return uflab, IMlab
    else:  # default return
        return twicetheta, chi


def OM_from_uf(uflab, calib, signgam=SIGN_OF_GAMMA, energy=0, offset=None, verbose=0):
    """
    2D vector position of point OM in detector frame plane in pixels
    alias function to calc_xycam
    """
    return calc_xycam(uflab, calib, signgam=signgam, energy=energy, offset=offset, verbose=verbose)


def IprimeM_from_uf(uflab, posI, calib, signgam=SIGN_OF_GAMMA, verbose=0):
    """
    from:
    uflab
    posI= IIprime = position (3elemts vector) of source with respect to I (calibrated emission source) in millimeter
    
    returns:
    IprimeM vector joining shifted source emission to point M lying on CCD
    """

    return calc_xycam(uflab, calib,
                      signgam=signgam,
                      energy=0,
                      offset=posI,
                      verbose=verbose,
                      returnIpM=True)


def calc_xycam(uflab,
                calib,
                signgam=SIGN_OF_GAMMA,
                energy=0,
                offset=None,
                verbose=0,
                returnIpM=False,
                pixelsize=165. / 2048,
                dim=(2048, 2048),
                rectpix=RECTPIX
                ):
    r"""
    compute Laue spots position x and y in pixels units in CCD frame from scattering vector q

    computes coordinates of point M on CCD from point source and **uflab**.
    Point Ip (source Iprime of x-ray scattered beams)
    (for each Laue spot **uflab** is the unit vector of **IpM**)
    Point Ip is shifted by offset (if not None) from the default point I
    (used to calibrate the CCD camera and 2theta chi determination)

    th0 (theta in degrees)
    Energy (energy in keV)

    :param uflab: list or array of [qx,qy,qz] (q vector)
    :type uflab: list or array (length must > 1)

    :param calib: list 5 detector calibration parameters
    :type calib: list of floats

    :param offset: offset (in mm) in the scattering source (origin of Laue spots)
            position with respect to the position which has been used
            for the calibration of  the CCD detector plane. Offset is positive when in the same
            direction as incident beam (i.e. in sample depth)
            (incident beam direction remains constant)
    :type offset: list of floats ([x,y,z])

    :returns:
        - xcam: list of pixel X coordinates
        - ycam: list of pixel Y coordinates
        - theta: list half scattering angle "theta" (in degree)

        - optionally energy=1: add in output list of spot energies (in keV) 

        - if returnIpM and offset not None: return list of vectors **IprimeM** 
    """
    detect, xcen, ycen, xbet, xgam = np.array(calib) * 1.

    # beta = PI/2 - xbet*DEG
    # xbet angle between IO and z axis
    # beta angle between y and IO
    # cosbeta= sin xbet
    # sinbeta = cos xbet

    cosbeta = np.cos(PI / 2. - xbet * DEG)
    sinbeta = np.sin(PI / 2. - xbet * DEG)

#    print "cosbeta", cosbeta
#    print "sinbeta", sinbeta

    # IOlab: vector joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    IOlab = detect * np.array([0.0, cosbeta, sinbeta])

#    print "IOlab", IOlab

    # unitary normal vector of CCD plane
    # joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    unlab = IOlab / np.sqrt(np.dot(IOlab, IOlab))

    # normalization of all input uflab
    norme_uflab = np.sqrt(np.sum(uflab ** 2, axis=1))
    uflab = uflab / np.reshape(norme_uflab, (len(norme_uflab), 1))

    # un is orthogonal to any vector joining O and a point M lying in the CCD frame plane
    scal = np.dot(uflab, unlab)
    normeIMlab = detect / scal

    # IMlab = normeIMlab*uflab
    IMlab = uflab * np.reshape(normeIMlab, (len(normeIMlab), 1))

    OMlab = IMlab - IOlab

#    print "OMlab", OMlab

    if offset not in (None,0,0.0):  # offset input in millimeter
        # OO'=II'-(II'.un)un  # 1 vector
        # dd'=  dd - II'.un # scalar
        # I'M= dd'/(uf.un) uf # n vector
        # I'O'= dd' un # 1 vectorin
        # O'M=I'M - I'O' # n vector
        # OM = OO' + O'M # n vector
        IIprime = offset*np.array([1,0,0])
        IIprime_un = np.dot(IIprime, unlab)
        OOprime = IIprime - IIprime_un * unlab
        ddprime = detect + IIprime_un
        IprimeM_norm = ddprime / scal
        IprimeM = uflab * np.reshape(IprimeM_norm, (len(uflab), 1))
        IprimeOprime = ddprime * unlab
        OMlab = OOprime + IprimeM - IprimeOprime

        if verbose:
            print("IIprime", IIprime)
            print("IIprime_un", IIprime_un)
            print("OOprime", OOprime)
            print("IprimeM_norm", IprimeM_norm)
            print("IprimeM", IprimeM)
            print("dd", detect)
            print("dd'", ddprime)
            print("OM", OMlab)

        if returnIpM:
            return IprimeM

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xca0 = OMlab[:, 0]
    if sinbeta != 0.:
        yca0 = OMlab[:, 1] / sinbeta
    else:
        yca0 = -OMlab[:, 2] / cosbeta
    # zca0 = 0

    cosgam = np.cos(-signgam * xgam * DEG)
    singam = np.sin(-signgam * xgam * DEG)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

#    print "xcam1", xcam1
#    print "ycam1", ycam1

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / (pixelsize * (1.0 + rectpix))

    twicetheta = (1 / DEG) * np.arccos(uflab[:, 1])
    th0 = twicetheta / 2.0

    # q = kf - ki
    qlab = uflab - np.array([0., 1., 0.])
    norme_qlab = np.sqrt(np.sum(qlab ** 2, axis=1))

    Energy = CST_CONV_LAMBDA_KEV * norme_qlab ** 2 / (2. * np.sin(th0 * DEG))

    if energy:
        return xcam, ycam, th0, Energy
    else:
        return xcam, ycam, th0


def calc_xycam_transmission(uflab,
                calib,
                signgam=SIGN_OF_GAMMA,
                energy=0,
                offset=None,
                verbose=0,
                returnIpM=False,
                pixelsize=165. / 2048,
                dim=(2048, 2048),
                rectpix=RECTPIX
                ):
    """
    Compute Laue spots position x and y in pixels units (in CCD frame) from scattering vector q

    As calc_xycam() but in TRANSMISSION geometry
    """

    distance_IO, xcen, ycen, xbet, xgam = np.array(calib) * 1.

    # beta = PI/2 - xbet*DEG
    # xbet angle between IO and z axis
    # beta angle between y and IO
    # cosbeta= sin xbet
    # sinbeta = cos xbet

#     cosbeta = np.cos(PI / 2. - xbet * DEG)
#     sinbeta = np.sin(PI / 2. - xbet * DEG)

    cosbeta = np.cos(-xbet * DEG)
    sinbeta = np.sin(-xbet * DEG)
    # if xbet positive bottom part of CCD is closer to sample than top part

#    print "cosbeta", cosbeta
#    print "sinbeta", sinbeta

    # IOlab: vector joining O nearest point of CCD plane and I (origin of lab frame and emission source)

    IOlab = distance_IO * np.array([0.0, cosbeta, sinbeta])

    # sinbeta negative for xbet positive

#    print "IOlab", IOlab

    # unitary normal vector of CCD plane
    # joining O nearest point of CCD plane and I (origin of lab frame and emission source)
    unlab = IOlab / np.sqrt(np.dot(IOlab, IOlab))

    # normalization of all input uflab
    norme_uflab = np.sqrt(np.sum(uflab ** 2, axis=1))
    uflab = uflab / np.reshape(norme_uflab, (len(norme_uflab), 1))

    # un is orthogonal to any vector joining O and a point M lying in the CCD frame plane
    scal = np.dot(uflab, unlab)
    normeIMlab = distance_IO / scal

    # IMlab = normeIMlab*uflab
    IMlab = uflab * np.reshape(normeIMlab, (len(normeIMlab), 1))

    OMlab = IMlab - IOlab

#    print "OMlab", OMlab

    if offset not in (None,0,0.0):  # offset input in millimeter
        # OO'=II'-(II'.un)un  # 1 vector
        # dd'=  dd - II'.un # scalar
        # I'M= dd'/(uf.un) uf # n vector
        # I'O'= dd' un # 1 vector
        # O'M=I'M - I'O' # n vector
        # OM = OO' + O'M # n vector
        IIprime = offset
        IIprime_un = np.dot(IIprime, unlab)
        OOprime = IIprime - IIprime_un * unlab
        ddprime = distance_IO + IIprime_un
        IprimeM_norm = ddprime / scal
        IprimeM = uflab * np.reshape(IprimeM_norm, (len(uflab), 1))
        IprimeOprime = ddprime * unlab
        OMlab = OOprime + IprimeM - IprimeOprime

        if verbose:
            print("IIprime", IIprime)
            print("IIprime_un", IIprime_un)
            print("OOprime", OOprime)
            print("IprimeM_norm", IprimeM_norm)
            print("IprimeM", IprimeM)
            print("dd", distance_IO)
            print("dd'", ddprime)
            print("OM", OMlab)

        if returnIpM:
            return IprimeM

    # OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    xca0 = OMlab[:, 0]
    if sinbeta != 0.:
        yca0 = OMlab[:, 1] / sinbeta
    else:
        yca0 = -OMlab[:, 2] / cosbeta
    # zca0 = 0

    cosgam = np.cos(-signgam * xgam * DEG)
    singam = np.sin(-signgam * xgam * DEG)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

#    print "xcam1", xcam1
#    print "ycam1", ycam1

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / (pixelsize * (1.0 + rectpix))

    twicetheta = (1 / DEG) * np.arccos(uflab[:, 1])
    th0 = twicetheta / 2.0

    # q = kf - ki
    qf = uflab - np.array([0., 1., 0.])
    norme_qflab = np.sqrt(np.sum(qf ** 2, axis=1))

    Energy = CST_CONV_LAMBDA_KEV * norme_qflab ** 2 / (2. * np.sin(th0 * DEG))

    if energy:
        return xcam, ycam, th0, Energy
    else:
        return xcam, ycam, th0


def calc_xycam_from2thetachi(twicetheta, chi, calib,
                             offset = 0,
                             verbose=0,
                             pixelsize=165. / 2048,
                             dim=(2048, 2048),
                             signgam=SIGN_OF_GAMMA,
                             kf_direction='Z>0'):
    r"""
    calculate spots coordinates in pixel units in detector plane
    from 2theta, chi angles (kf)
    
    :param offset: offset (in mm) in the scattering source (origin of Laue spots)
            position with respect to the position which has been used
            for the calibration of  the CCD detector plane. Offset is positive when in the same
            direction as incident beam (i.e. in sample depth)
            (incident beam direction remains constant)
    :type offset: list of floats ([x,y,z])
    """
    # scattered vector not in Lauetools frame (y//ki)
    uflab = uflab_from2thetachi(twicetheta, chi, verbose=0)

    if verbose:
        print("uflab", uflab)

    if kf_direction in ('Z>0',):  # , '[90.0, 45.0]'):
        return calc_xycam(uflab, calib, offset = offset, pixelsize=pixelsize, dim=dim,
                          signgam=signgam)
    elif kf_direction in ('Y>0', 'Y<0',):
        print("CAUTION: not checked yet")
        # TODO raise ValueError, print "not checked yet"
        return calc_xycam(uflab, calib, offset=offset, pixelsize=pixelsize, dim=dim,
                          signgam=signgam)
    elif kf_direction in ('X>0',):  # transmission
        return calc_xycam_transmission(uflab, calib, offset = offset, pixelsize=pixelsize,
                                       dim=dim, signgam=signgam)
    else:
        sentence = "kf_direction = %s is not implemented yet " % kf_direction
        sentence += "in calc_xycam_from2thetachi() in find2thetachi"
        raise ValueError(sentence)


def uflab_from2thetachi(twicetheta, chi, verbose=0):
    r"""
    Compute **uf** vectors coordinates in lauetools LT2 frame
    from **kf** scattering angles 2theta and chi angles
    
    :param twicetheta: (list) 2theta angle(s) ( in degree)
    :param chi: (list) chi angle(s) ( in degree)
    
    :return: (list) [uf_x,uf_y,uf_z]
    
    chi convention: (XMAS convention)
    y along Xray horizontal
    x towards wall behind horizontal
    z vertical up

    kf=( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi)
    """

    ctw = np.cos(np.array(twicetheta) * DEG)
    stw = np.sin(np.array(twicetheta) * DEG)
    cchi = np.cos(np.array(chi) * DEG)
    schi = np.sin(np.array(chi) * DEG)

    xuflab = -stw * schi
    yuflab = ctw
    zuflab = stw * cchi

    uflab = np.array([xuflab, yuflab, zuflab]).T

    if verbose:
        print("uflab", uflab)

    return uflab


def q_unit_XYZ(twicetheta, chi):
    """
    from kf angles return unit vector of q
    # TODO: useful ? check with from_twchi_to_qunit()
    lauetools frame 
    #in deg
    """
    THETA = twicetheta / 2. * DEG
    CHI = chi * DEG
    return np.array([-np.sin(THETA),
                     np.cos(THETA) * np.sin(CHI),
                     np.cos(THETA) * np.cos(CHI)])


def q_unit_2thetachi(vec):
    """
    compute kf angles from a vector
    #result in deg
    # TODO: useful ? check with from_qunit_to_twchi()
    lauetools frame ?
    """
    X, Y, Z = tuple(vec)
    # TODO: sign of chi must be checked
    chi = np.arctan2(Y, Z) / DEG
    theta = -np.arcsin(X) / DEG
    return np.array([2. * theta, chi])


def from_twchi_to_qunit(Angles):

    """
    from kf 2theta,chi to q unit in lab frame (xx// ki) q=kf-ki
    returns array = (all x's, all y's, all z's)
    
    Angles in degrees !! 
    Angles[0] 2theta deg values,
    Angles[1] chi values in deg
    
    this is the inverse function of from_qunit_to_twchi(), useful to check it
    """

    twthe = np.array(Angles[0]) * DEG
    chi = np.array(Angles[1]) * DEG
    no = 2. * np.sin(twthe / 2.)
    qx = np.cos(twthe) - 1
    qy = np.sin(twthe) * np.sin(chi)
    qz = np.sin(twthe) * np.cos(chi)
    return np.array([qx, qy, qz]) / no


def from_qunit_to_twchi(arrayXYZ, labXMAS=0):
    """
    from a q unit vector (defining a direction) (-sin the,costhesinchi,costhe coschi)
    in lab frame (xx// ki) q=kf-ki
    returns 2the chi 

    for kf = (cos2the,sin2the sinchi,sin2the coschi) and
    q= 2sinthe(-sin the,costhe sinchi,costhe coschi)

    In XMAS Frame   labXMAS=1

    for kf = (-sin2the sinchi,cos2the,sin2the coschi) and
    q= 2sinthe(-costhe sinchi,-sin the,costhe coschi)
    """
    X, Y, Z = arrayXYZ

    if labXMAS:
        chi = np.arctan2(-X * 1., Z)
        twthe = 2 * np.arcsin(-Y * 1.)
    else:  # labXMAS=0  lauetools
        chi = np.arctan2(Y * 1., Z)
        twthe = 2 * np.arcsin(-X * 1.)

    return np.array([twthe, chi]) / DEG


def qvector_from_xy_E(xcamList,ycamList,energy,CCDcalibrationparameters,pixelsize):

    """
    return q vectors in Lauetools frame given x and y pixel positions on detector
    for a given Energy (keV)
    
    """
    # in LT's frame (x// ki)
    
#     print "xcamList",xcamList
#     print "ycamList",ycamList
    twtheta,chi = calc_uflab(xcamList, ycamList, CCDcalibrationparameters,
               returnAngles=1,
               verbose=0,
               pixelsize=pixelsize,
               signgam=SIGN_OF_GAMMA,
               rectpix=RECTPIX,
               kf_direction='Z>0')
    
    thetarad = twtheta * DEG / 2.
    chirad = chi * DEG


    qx = -np.sin(thetarad)
    qy = np.cos(thetarad) * np.sin(chirad)
    qz = np.cos(thetarad) * np.cos(chirad)

    newq = np.array([qx, qy, qz])
    normnewq = np.sqrt(qx**2+qy**2+qz**2)
    
    qvec=newq*(1./normnewq)
#     print "qvec",qvec
    

#     print "energy",energy
    
    
    qvector_Lauetoolsframe = qvec*np.sin(thetarad)*(2*energy/12.398)
    
#     print "qvector_Lauetoolsframe",qvector_Lauetoolsframe
    
    
    return qvector_Lauetoolsframe

def unit_q(ttheta, chi, frame='lauetools', anglesample=40.):
    """
    returns unit q vector from 2theta,chi coordinates

    three possible frames: lauetools , XMASlab, XMASsample
    """
    thetarad = ttheta * DEG / 2.
    chirad = chi * DEG

    if frame == 'lauetools':
        qx = -np.sin(thetarad)
        qy = np.cos(thetarad) * np.sin(chirad)
        qz = np.cos(thetarad) * np.cos(chirad)

        newq = np.array([qx, qy, qz])
        normnewq = np.sqrt(np.dot(newq, newq))

        return newq / normnewq
    # LT2 frame
    elif frame == 'XMASlab':
        # chi convention:
        # y along Xray horizontal
        # x towards wall behind horizontal
        # z vertical up

        # kf=( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi) XMAS convention
        # ki=( 0, 1  , 0) XMAS convention
        # q = 2 sin theta (- costheta sinchi , - sintheta  ,  costheta coschi)
        #  unitkf  =  unitki  +   2sintheta unitq

        qx = np.cos(thetarad) * np.sin(chirad)
        qy = -np.sin(thetarad)
        qz = np.cos(thetarad) * np.cos(chirad)

        newq = np.array([qx, qy, qz])
        normnewq = np.sqrt(np.dot(newq, newq))

        return newq / normnewq
    # LT2 sample frame
    elif frame == 'XMASsample':

        # kf=( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi) XMAS convention
        # ki=( 0, 1  , 0) XMAS convention
        # q = 2 sin theta (- costheta sinchi , - sintheta  ,  costheta coschi)
        #  unitkf  =  unitki  +   2sintheta unitq

        angrad = anglesample * np.pi / 180.  # Must include -xbet/2 correction ???
        ca = np.cos(angrad)
        sa = np.sin(angrad)

        matrot = np.array([[1, 0, 0.],
                           [0., ca, sa],
                           [0, -sa, ca]])

        qx = np.cos(thetarad) * np.sin(chirad)
        qy = -np.sin(thetarad)
        qz = np.cos(thetarad) * np.cos(chirad)

        newq = np.dot(matrot, np.array([qx, qy, qz]))
        normnewq = np.sqrt(np.dot(newq, newq))

        return newq / normnewq


def plotXY2thetachi(datX, datY, dat2the, datchi, mostintense=None):
    """
    old script to combine plot of pixel x,y and 2theta chi plot
    """
    if mostintense is not None and mostintense < len(datX):
        data_x = datX[:mostintense]
        data_y = datY[:mostintense]
        twicetheta = dat2the[:mostintense]
        chi = datchi[:mostintense]
    else:
        data_x = datX
        data_y = datY
        twicetheta = dat2the
        chi = datchi

    plot1 = P.subplot(121)
    plot1.set_aspect(aspect='equal')
    P.xlabel('X')
    P.ylabel('Y')
    plot1.scatter(tuple(data_x), tuple(data_y))


    plot2 = P.subplot(122)
    plot2.set_aspect(aspect=.5)
    P.xlabel('chi')
    P.ylabel('2theta')
    plot2.scatter(tuple(chi), tuple(twicetheta), c='r', marker='d')

    P.show()


# ---------------    Frame Matrix conversion
def matxmas_to_OrientMatrix(satocr, calib):
    """
    thanks to Odile robach's reverse engineering hard work

    From XMAS matrices in IND file to matrix in lauetools frame
    convert matrix from XMAS sample axes to lab axes + normalize by astar

    - calib with last angles in degrees

    - satocrs = transposee de la matrice numero 2 du .STR :   matrice UB 
    "coordinates of a*, b*, c* in X, Y, Z"
    - satocru = matrice numero 2 du .IND :   matrice U
    "matrix hkl => XYZ"
    """
    astar = np.sqrt(np.sum(satocr[:, 0] ** 2))
    # print "satocr \n", satocr
    satocrnorm = satocr / astar  # sample to crystal: satocr
    # print "satocrnorm \n", satocrnorm

    omega0 = 40.0  # deg
    xbet = calib[3] * DEG  # rad
    # print "xbet in rad" , xbet
    omega = omega0 * DEG - xbet / 2.0
    # print "omega" , omega*180.0/np.pi

    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(omega), np.sin(omega)],
                      [0.0, -np.sin(omega), np.cos(omega)]])
    # print "matrot \n" , matrot

    labtocr = np.dot(matrot, satocrnorm)
    astarlab = labtocr[:, 0]
    bstarlab = labtocr[:, 1]
    cstarlab = labtocr[:, 2]

    matstarlab1 = np.hstack((astarlab, bstarlab, cstarlab))
    changesign = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])

    matstarlab2 = np.multiply(matstarlab1, changesign)
    # print "matstarlab from satocr \n", matstarlab2

    # mm=matstarlab1
    mm = matstarlab2

    # matstarlabLaueTools= array([[mm[1],-mm[4],-mm[7]],[mm[0],-mm[6],-mm[3]],[-mm[2],mm[5],mm[8]]])
    matstarlabLaueTools = np.array([[mm[1], mm[4], mm[7]],
                                    [-mm[0], -mm[3], -mm[6]],
                                    [mm[2], mm[5], mm[8]]])

    # resulting matrix contains orientation + strain

    return matstarlabLaueTools


def matstarlabLaueTools_to_matstarlabOR(UBmat):
    """
    convert matrix from lauetools frame: ki//x, z towards CCD (top), y = z^x
                    to ORobach or XMAS's frame: ki//y, z towards CCD (top), y = z^x

    convert the so called UBmat to matstarlab
    (matstarlab stands for 'matrix of reciprocal unit cell basis vectors in lab. frame')

    see the reciprocal function: matstarlabOR_to_matstarlabLaueTools
    """
    mm = UBmat

    astar1 = mm[:, 0]
    bstar1 = mm[:, 1]
    cstar1 = mm[:, 2]

    # print "check matrix before normalizing"
    # print "normes astar1 bstar1 cstar1 = ", GT.norme_vec(astar1),GT.norme_vec(astar1),GT.norme_vec(astar1)
    # print "inner products astar1.bstar1, bstar1.cstar1, cstar1.astar1 \n",\
    #      inner(astar1,bstar1), inner(bstar1,cstar1), inner(cstar1,astar1)
    # print "cross products sign(astar1xbstar1).cstar1", sign(inner(cross(astar1,bstar1),cstar1))

    # matstarlab = array([-mm[1,0],mm[0,0],mm[2,0],mm[1,1],-mm[0,1],-mm[2,1],mm[1,2],-mm[0,2],-mm[2,2]])
    matstarlab = np.array([-mm[1, 0], mm[0, 0], mm[2, 0],
                           - mm[1, 1], mm[0, 1], mm[2, 1],
                           - mm[1, 2], mm[0, 2], mm[2, 2]])

    matstarlab = matstarlab / GT.norme_vec(matstarlab[:3])

    return matstarlab


def matstarlabOR_to_matstarlabLaueTools(matstarlab):
    """
    reciprocal function of matstarlabLaueTools_to_matstarlabOR
    """
    mm = matstarlab

    astar1 = mm[0:3]
    bstar1 = mm[3:6]
    cstar1 = mm[6:]

    # print "check matrix before normalizing"
    # print "normes astar1 bstar1 cstar1 = ", GT.norme_vec(astar1),GT.norme_vec(astar1),GT.norme_vec(astar1)
    # print "inner products astar1.bstar1, bstar1.cstar1, cstar1.astar1 \n",\
    #      inner(astar1,bstar1), inner(bstar1,cstar1), inner(cstar1,astar1)
    # print "cross products sign(astar1xbstar1).cstar1", sign(inner(cross(astar1,bstar1),cstar1))

    UBmat = np.array([[mm[1], mm[4], mm[7]],
                    [-mm[0], -mm[3], -mm[6]],
                    [mm[2], mm[5], mm[8]]])

    return UBmat

def matstarlab_to_matwithlatpar(matstarlab, dlatu_rad):

    norm_vec0 =np.sqrt(np.inner(matstarlab[0:3],matstarlab[0:3])) 
    matnorm = matstarlab / norm_vec0
    rlatsr = CP.matrix_to_rlat(GT.matline_to_mat3x3(matnorm), angles_in_deg=0)
    dlatsr = CP.dlat_to_rlat(rlatsr, angles_in_deg=0)

    # print "matstarlab = \n", matstarlab
    dil = CP.dlat_to_dil(dlatu_rad, dlatsr, angles_in_deg=0)
    # print "dilatation =", dil
    rlats1 = np.hstack((rlatsr[0:3] * (1.0 + dil), rlatsr[3:6]))
    # print "rlats1 = ", rlats1
    mat = matnorm * rlats1[0]
    #dlats1 = CP.dlat_to_rlat(rlats1, angles_in_deg=0)
    # print "dlats1 = ", dlats1

    return(mat)


def readlt_det(filedet, returnmatLT=False, min_matLT=False):

    print("reading info from LaueTools det file : \n", filedet)
    print("calibration, orientation matrix")
    print("convert matrix to matstarlabOR")

    calib, mat_line = LTGeo.readfile_det(filedet)

    matLT3x3 = (GT.matline_to_mat3x3(mat_line)).T

    if min_matLT == True :
        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        matLT3x3 = matmin

    matstarlab = matstarlabLaueTools_to_matstarlabOR(matLT3x3)

    print("matstarlab = \n", matstarlab.round(decimals=6))


    if returnmatLT == False :
        return(calib, matstarlab)
    else :
        return(calib, matstarlab, matLT3x3)


def readlt_fit(filefit,
            returnmatLT=False, min_matLT=False,
            readmore=False, verbose=1,
            verbose2=0, readmore2=False):
    """
    modif 03Aug12 : genfromtxt removed (problem with skip_footer)
    add transfo of HKL's if matmin_LT  == True
    """

    if verbose :
        print("reading info from LaueTools fit file : \n", filefit)
        print("strained orientation matrix, peak list")
        print("convert matrix to matstarlabOR")

    matLT3x3 = np.zeros((3, 3), dtype=np.float)
    strain = np.zeros((3, 3), dtype=np.float)
    f = open(filefit, 'r')
    i = 0
    matrixfound = 0
    calibfound = 0
    pixdevfound = 0
    strainfound = 0
    eulerfound = 0
    linecalib = 0
    linepixdev = 0
    linestrain = 0
    lineeuler = 0
    list1 = []
    linestartspot = 10000
    lineendspot = 10000

    try:
        for line in f:
            i = i + 1
            # print i
            if line[:5] == "spot#" :
                linecol = line.rstrip('\n')
                linestartspot = i + 1
            if line[:3] == "#UB" :
                # print line
                matrixfound = 1
                linestartmat = i
                lineendspot = i
                j = 0
                # print "matrix found"
            if line[:3] == "#Sa" :
                # print line
                calibfound = 1
                linecalib = i + 1
            if line[:3] == "#pi" :
                # print line
                pixdevfound = 1
                linepixdev = i + 1
            if line[:3] == "#de" :
                # print line
                strainfound = 1
                linestrain = i
                j = 0
            if line[:3] == "#Eu" :
                # print line
                eulerfound = 1
                lineeuler = i + 1
            if matrixfound :
                if i in (linestartmat + 1, linestartmat + 2, linestartmat + 3) :
                    toto = line.rstrip('\n').replace('[', '').replace(']', '').split()
                    # print toto
                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    j = j + 1
            if strainfound :
                if i in (linestrain + 1, linestrain + 2, linestrain + 3) :
                    toto = line.rstrip('\n').replace('[', '').replace(']', '').split()
                    # print toto
                    strain[j, :] = np.array(toto, dtype=float)
                    j = j + 1
            if calibfound & (i == linecalib):
                calib = np.array(line.split(',')[:5], dtype=float)
                # print "calib = ", calib
            if eulerfound & (i == lineeuler):
                euler = np.array(line.replace('[', '').replace(']', '').split()[:3], dtype=float)
                # print "euler = ", euler
            if pixdevfound & (i == linepixdev):
                pixdev = float(line.rstrip('\n'))
                # print "pixdev = ", pixdev
            if (i >= linestartspot) & (i < lineendspot) :
                list1.append(line.rstrip('\n').replace('[', '').replace(']', '').split())
    finally:
        f.close()
        linetot = i

    # print "linetot = ", linetot

    data_fit = np.array(list1, dtype=float)

    if verbose :
        print(np.shape(data_fit))
        print(data_fit[0, :])
        print(data_fit[-1, :])

    # print "UB matrix = \n", matLT3x3.round(decimals=6)

    if verbose2 :
        print("before transfo")
        print(data_fit[0, 2:5])
        print(data_fit[-1, 2:5])
        q0 = np.dot(matLT3x3, data_fit[0, 2:5])
        print("q0 = ", q0.round(decimals=4))
        qm1 = np.dot(matLT3x3, data_fit[-1, 2:5])
        print("qm1 = ", qm1.round(decimals=4))

    if min_matLT == True :
        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3, verbose=verbose)
        matLT3x3 = matmin
        if verbose :
            print("transfmat \n", list(transfmat))
        # transformer aussi les HKL pour qu'ils soient coherents avec matmin
        hkl = data_fit[:, 2:5]
        data_fit[:, 2:5] = np.dot(transfmat, hkl.transpose()).transpose()

    if verbose2 :
        print("after transfo")
        print(data_fit[0, 2:5])
        print(data_fit[-1, 2:5])
        q0 = np.dot(matLT3x3, data_fit[0, 2:5])
        print("q0 = ", q0.round(decimals=4))
        qm1 = np.dot(matLT3x3, data_fit[-1, 2:5])
        print("qm1 = ", qm1.round(decimals=4))

    matstarlab = matstarlabLaueTools_to_matstarlabOR(matLT3x3)

    if verbose : print("matstarlab = \n", matstarlab.round(decimals=6))

    if readmore2 == True : readmore = False

    # xx yy zz yz xz xy
    strain6 = np.array([strain[0, 0], strain[1, 1], strain[2, 2], strain[1, 2], strain[0, 2], strain[0, 1]])

    if returnmatLT == False :
        if readmore == True :
            return(matstarlab, data_fit, calib, pixdev)
        elif readmore2 == True :
            return(matstarlab, data_fit, calib, pixdev, strain6, euler)
        else :
            return(matstarlab, data_fit)
    else :
        if readmore == True :
            return(matstarlab, data_fit, matLT3x3, calib, pixdev)
        elif readmore2 == True :
            return(matstarlab, data_fit, matLT3x3, calib, pixdev, strain6, euler)
        else :
            return(matstarlab, data_fit, matLT3x3)

def readall_str(grain_index, filemane_str,
                returnmatLT=False, min_matLT=False):

    data_str, matstr, calib, dev_str = LTGeo.readfile_str(filemane_str, grain_index)

    # postprocessing

    data_str[:, 2:5] = -data_str[:, 2:5]

    satocrs = matstr.transpose()
    # print "strained orientation matrix (satocrs) = \n", satocrs

    matstarlab = matxmas_to_matstarlab(satocrs, calib)

    if min_matLT == True :
        matLT3x3 = matstarlabOR_to_matstarlabLaueTools(matstarlab)
        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        matLT3x3 = matmin
        matstarlab = matstarlabLaueTools_to_matstarlabOR(matLT3x3)
        # transfo des HKL a verifier
        hklmin = np.dot(transfmat, data_str[:, 2:5].transpose()).T
        data_str[:, 2:5] = hklmin

    if returnmatLT == False :
        return(data_str, matstarlab, calib, dev_str)
    else :
        return(data_str, matstarlab, calib, dev_str, matLT3x3)


def matxmas_to_matstarlab(satocr, calib):
    """
    Original function to correctly use matrix from STR or IND

    # modif 04 Mar 2010 xbet en degres au lieu de radians

    # satocrs = transposee de la matrice numero 2 du .STR :
    # "coordinates of a*, b*, c* in X, Y, Z"
    # satocru = matrice numero 2 du .IND :
    #  "matrix hkl => XYZ"
    #print "convert matrix from XMAS sample axes to lab axes + normalize by astar"
    """
    astar = GT.norme_vec(satocr[:, 0])
    # print "satocr \n", satocr
    satocrnorm = satocr / astar
    # print "satocrnorm \n", satocrnorm

    omega0 = 40.0
    xbetrad = calib[3] * DEG

    # print "xbetrad = " , xbetrad

    omega = omega0 * DEG - xbetrad / 2.0

    # print "omega" , omega*180.0/np.pi

    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(omega), np.sin(omega)],
                       [0.0, -np.sin(omega), np.cos(omega)]])
    # print "matrot \n" , matrot

    labtocr = np.dot(matrot, satocrnorm)
    astarlab = labtocr[:, 0]
    bstarlab = labtocr[:, 1]
    cstarlab = labtocr[:, 2]

    matstarlab1 = np.hstack((astarlab, bstarlab, cstarlab))
    changesign = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    matstarlab2 = np.multiply(matstarlab1, changesign)
    # print "matstarlab from satocr \n", matstarlab2
    return matstarlab2


def Compute_data2thetachi(filename,
                        tuple_column_X_Y_I,
                        _nblines_headertoskip,
                        sorting_intensity='yes',
                        param=None,
                        kf_direction='Z>0',
                        verbose=1,
                        signgam=SIGN_OF_GAMMA,
                        pixelsize=165. / 2048,
                        dim=(2048, 2048),  # only for peaks coming from fit2d doing an y direction inversion
                        saturation=0,
                        forceextension_lines_to_extract=None,
                        col_isbadspot=None,
                        alpha_xray_incidence_correction=None):
    """
    Convert spot positions x,y to scattering angles 2theta, chi from a list of peaks

    :param filename: fullpath to peaks list ASCII file
    :type filename: string

    :param tuple_column_X_Y_I: tuple with column indices of spots X, Y (pixels on CCD) and intensity  
    :type tuple_column_X_Y_I: 3 elements

    :param _nblines_headertoskip: nb of line to skip before reading
    an array of data in ascii file

    :param param: list of CCD calibration parameters [det, xcen, ycen, xbet, xgam]
    :param pixelsize: pixelsize in mm
    :param dim: (nb pixels x, nb pixels y)

    :param kf_direction: label of detection geometry (CCD position): 'Z>0','X>0',...
    :type kf_direction: string


    :param sorting_intensity: 'yes' sort spots list by decreasing intensity


    saturation = 0 : do not read Ipixmax column of DAT file from LaueTools peaksearch
    saturation > 0 : read Ipixmax column and create data_sat list
    data_sat[i] = 1 if Ipixmax[i]> saturation, =0 otherwise

    Note: _nblines_headertoskip =0 for .pik file (no header at all)
            _nblines_headertoskip =1 for .peaks coming from fit2d

    col_Ipixmax = 10 for .dat from LT peak search using method "Local Maxima"
    (TODO : bug in Ipixmax for method "convolve")
    """
    col_X, col_Y, col_I = tuple_column_X_Y_I

    extension = filename.split('.')[-1]

    if forceextension_lines_to_extract is not None:
        extension = 'forcedextension'

    if extension == 'pik':  # no header
        nbline = 0
        data_xyI = np.loadtxt(filename, usecols=(col_X, col_Y, col_I),
                              skiprows=nbline)
    elif extension == 'peaks':  # single line header
        data_xyI = np.loadtxt(filename, usecols=(col_X, col_Y, col_I),
                              skiprows=1)

    elif extension in ('dat', 'DAT'):  # peak list single line header
        data_xyI = np.loadtxt(filename, usecols=(col_X, col_Y, col_I),
                              skiprows=1)
        print("nb of spots and columns in .dat file", data_xyI.shape)

        if saturation :
            data_Ipixmax = np.loadtxt(filename, usecols=-1, skiprows=1)
            # print "Ipixmax ",data_Ipixmax
            indsat = np.where(data_Ipixmax >= saturation)
            # print indsat
            data_sat = np.zeros(len(data_Ipixmax), dtype=np.int)
            data_sat[indsat[0]] = 1
            # print data_sat

            if col_isbadspot is not None :
                data_isbadspot = np.loadtxt(filename, usecols=col_isbadspot,
                                            skiprows=1)
                print(data_isbadspot)

        # mike.close()

    elif extension == 'forcedextension':
        # mike=scipy.io.array_import.get_open_file(filename)
        # mike.readline()
        # data_xyI=scipy.io.array_import.read_array(filename,columns=(col_X,col_Y,col_I),lines=forceextension_lines_to_extract)
        data_xyI = np.loadtxt(filename, usecols=(col_X, col_Y, col_I), skiprows=1)
        # mike.close()
    elif extension == 'cor':  # single line header
        try:
            data_xyI = np.loadtxt(filename, usecols=(2, 3, 4),
                              skiprows=1)
        except:
            raise ValueError('%s does contain just one header line'%filename)
    else:
        raise ValueError('Unknown file extension for %s'%filename)

    sha = data_xyI.shape

    # manage if there is a single spot
    if len(sha) == 1:
        nb_peaks = 1
    else:
        nb_peaks = sha[0]

    if param is None:
        raise ValueError('Missing param arg in Compute_data2thetachi() of find2thetachi module')
    else:
        param_det = param

    if verbose:
        print("file :%s" % filename)
        print("containing %d peaks" % nb_peaks)
        # print data_xyI

    # default
    # data_x=data_xyI[:,0]
    # data_y=data_xyI[:,1]
    # data_I=data_xyI[:,2]

    if filename.split('.')[-1] in ('pik', 'peaks'):
        data_x = data_xyI[:, 0]  # + 0.5  # 0.5 for being closer to XMAS peaks position
        data_y = dim[1] - data_xyI[:, 1]  # + 0.5 # 0.5 for being closer to XMAS peaks position
        data_I = data_xyI[:, 2]  # for fit2d pixels convention

    elif filename.split('.')[-1] in ('dat', 'DAT'):
        if nb_peaks > 1:
            data_x = data_xyI[:, 0]
            data_y = data_xyI[:, 1]
            data_I = data_xyI[:, 2]
        elif nb_peaks == 1:
            data_x = [data_xyI[0], data_xyI[0]]
            data_y = [data_xyI[1], data_xyI[1]]
            data_I = [data_xyI[2], data_xyI[2]]

    if extension in ('forcedextension','cor'):
        data_x = data_xyI[:, 0]
        data_y = data_xyI[:, 1]
        data_I = data_xyI[:, 2]
        
    # 21Jul14  O. Robach---------------
    if alpha_xray_incidence_correction != None :

        print("Using alpha_xray_incidence_correction = ", alpha_xray_incidence_correction)
        xystart = np.column_stack((data_x, data_y))
#        print "xystart = ", xystart
        npics = np.shape(xystart)[0]
        xynew = np.zeros((npics, 2), float)
        xycen = np.array([param_det[1], param_det[2]])
#        print "xycen = ", xycen
        dxy = xystart - xycen
        dxynorm2 = (np.multiply(dxy, dxy)).sum(axis=1)
#        print "dxynorm2 = ", dxynorm2
        dxynorm = np.power(dxynorm2, 0.5)
#        print "dxynorm =", dxynorm
        dxynorminv = 1. / dxynorm
        scale_factor = pixelsize / param_det[0]
        scale_factor = scale_factor * scale_factor
#        print "dd = ", param_det[0]
#        print "pixelsize = ", pixelsize
#        print "scale_factor = ", scale_factor
        for i in range(npics):
            xynew[i, :] = xystart[i, :] \
                + alpha_xray_incidence_correction * scale_factor * dxy[i, :] * dxynorm[i]

        delta_xy = xynew - xystart
#        print "delta_xy = ", delta_xy
        print("maximum spot displacement |dx| |dy| : ", (abs(delta_xy)).max(axis=0).round(decimals=3))

        data_x = xynew[:, 0]
        data_y = xynew[:, 1]
    #-----------------------------------

    twicethetaraw, chiraw = calc_uflab(data_x, data_y,
                                       param_det[:5],
                                       returnAngles=1,
                                       pixelsize=pixelsize,
                                       signgam=signgam,
                                       kf_direction=kf_direction)

    # print chi,twicetheta
    if nb_peaks > 1 and sorting_intensity == 'yes':
        listsorted = np.argsort(data_I)[::-1]
        chi = np.take(chiraw, listsorted)
        twicetheta = np.take(twicethetaraw, listsorted)
        data_x = np.take(data_x, listsorted)
        data_y = np.take(data_y, listsorted)
        dataintensity = np.take(data_I, listsorted)
        if saturation :
            data_sat = np.take(data_sat, listsorted)
        if col_isbadspot != None :
            data_isbadspot = np.take(data_isbadspot, listsorted)

    else:
        dataintensity = data_I
        chi = chiraw
        twicetheta = twicethetaraw

    if nb_peaks == 1:
        dataintensity = [dataintensity[0]]
        chi = [chi[0]]
        twicetheta = [twicetheta[0]]
        data_x = [data_x[0]]
        data_y = [data_y[0]]

    if saturation :
        print("adding flag column for saturated peaks")
        return twicetheta, chi, dataintensity, data_x, data_y, data_sat
    if col_isbadspot != None :
        return twicetheta, chi, dataintensity, data_x, data_y, data_sat, data_isbadspot

    else :
        if col_isbadspot != None :
            return twicetheta, chi, dataintensity, data_x, data_y, data_isbadspot
        else :
            return twicetheta, chi, dataintensity, data_x, data_y


def convert2corfile(filename, calibparam,
                    dirname_in=None, dirname_out=None,
                    signgam=SIGN_OF_GAMMA,
                    pixelsize=165. / 2048,
                    CCDCalibdict=None):
    """
    create a .cor file (ascii peaks list (2theta chi X Y int ...))
    from a peak list file (x,y,I,...)

    :param calibparam: list of 5 CCD cakibration parameters
    (used if CCDCalibdict is None or  CCDCalibdict['CCDCalibPameters'] is missing
    :param pixelsize: CCD pixelsize (in mm)
    (used if CCDCalibdict is None or CCDCalibdict['pixelsize'] is missing)

    :param CCDCalibdict: dictionary of CCD file and calibration parameters
    """
    if dirname_in != None:
        filename_in = os.path.join(dirname_in, filename)
    else:
        filename_in = filename

    if CCDCalibdict is not None:
        if 'CCDCalibParameters' in CCDCalibdict:
            calibparam = CCDCalibdict['CCDCalibParameters']

        if 'xpixelsize' in CCDCalibdict:
            pixelsize = CCDCalibdict['xpixelsize']

    (twicetheta, chi,
     dataintensity,
     data_x, data_y) = Compute_data2thetachi(filename_in,
                                       (0, 1, 3), 1,  # 2 for centroid intensity, 3 for integrated  intensity
                                        sorting_intensity='yes',
                                        param=calibparam,
                                        signgam=signgam,
                                        pixelsize=pixelsize)

    # TODO: handle windowsOS path syntax
    filename_wo_path = filename.split('/')[-1]

    file_extension = filename_wo_path.split('.')[-1]

    prefix_outputname = filename_wo_path[:-len(file_extension) - 1]

    if dirname_out != None:
        filename_out = os.path.join(dirname_out, prefix_outputname)
    else:
        filename_out = prefix_outputname

#     print "filename_out", filename_out

    if CCDCalibdict is not None:
        for kk, key in enumerate(DictLT.CCD_CALIBRATION_PARAMETERS[:5]):
            CCDCalibdict[key] = calibparam[kk]

        CCDCalibdict['xpixelsize'] = pixelsize
        CCDCalibdict['ypixelsize'] = pixelsize

        param = CCDCalibdict

        # update dict according to values in file .cor
        f = open(filename_in, 'r')
        param = LTGeo.readCalibParametersInFile(f, Dict_to_update=CCDCalibdict)
        f.close()

    else:
        param = calibparam + [pixelsize]

    LTGeo.writefile_cor(filename_out, twicetheta, chi, data_x, data_y, dataintensity,
                          sortedexit=0,
                          param=param,
                          initialfilename=filename)


def convert2corfile_fileseries(fileindexrange, filenameprefix, calibparam,
                               suffix='', nbdigits=4,
                               dirname_in=None, outputname=None, dirname_out=None,
                               signgam=SIGN_OF_GAMMA, pixelsize=165. / 2048, fliprot='no'):
    """
    convert a serie of peaks list ascii files to .cor files (adding scattering angles)
    
    filename is decomposed as following for incrementing file index in ####:
    prefix####suffix
    example: myimage_0025.myccd => prefix=myimage_ nbdigits=4 suffix=.myccd
    
    :param nbdigits: nb of digits of file index in filename (with zero padding)
    (example: for myimage_0002.ccd nbdigits = 4
    
    :param calibparam: list of 5 CCD cakibration parameters
    """
    encodingdigits = '%%0%dd' % nbdigits

    if suffix == '':
        suffix = '.dat'

    for fileindex in range(fileindexrange[0], fileindexrange[1] + 1):
        filename_in = filenameprefix + encodingdigits % fileindex + suffix
        print("filename_in", filename_in)
        convert2corfile(filename_in, calibparam,
                        dirname_in=dirname_in, outputname=outputname,
                        dirname_out=dirname_out, signgam=signgam,
                        pixelsize=pixelsize)


def convert2corfile_multiprocessing(fileindexrange, filenameprefix, calibparam,
                                    dirname_in=None,
                                    suffix='', nbdigits=4,
                    outputname=None, dirname_out=None, signgam=SIGN_OF_GAMMA,
                    pixelsize=165. / 2048, fliprot='no', nb_of_cpu=6):
    """
    launch several processes in parallel to convert .dat file to .cor file
    """
    import multiprocessing

    try:
        index_start, index_final = fileindexrange
    except:
        raise ValueError("Need 2 file indices integers in fileindexrange=(indexstart, indexfinal)")
        return

    fileindexdivision = GT.getlist_fileindexrange_multiprocessing(index_start, index_final, nb_of_cpu)

#    t00 = time.time()
    jobs = []
    for ii in range(nb_of_cpu):
        proc = multiprocessing.Process(target=convert2corfile_fileseries,
                                    args=(fileindexdivision[ii], filenameprefix, calibparam,
                               suffix, nbdigits,
                               dirname_in, outputname, dirname_out,
                               signgam, pixelsize))
        jobs.append(proc)
        proc.start()
#    t_mp = time.time() - t00
#    print "Execution time : %.2f" % t_mp


def fromlab_tosample(UB, anglesample_deg=40):  # in deg
    """
    compute UBs

    qs = UBs G   with G =ha*+kb*+lc* a*,b*,c* are aligned with lab frame x,y,z
    qs is q in sample frame deduced from lab by a 40* rotation around y (lauetools convention)

    lauetools convention: x // ki ie. ki = 2pi/lambda(1,0,0), z perpendicular to x and contained 
    in the plane defined by x and dd* u where u is a unit vector normal to the CCD plane
    and dd is the shortest distance between the CCD plane and the emission source of scattered beams

    """
    anglesample = anglesample_deg * DEG  # in rad
    Rot = np.array([[ np.cos(anglesample), 0, np.sin(anglesample)],
                    [0, 1, 0],
                    [-np.sin(anglesample), 0, np.cos(anglesample)]])

    # = GT.matRot([0,1,0], 40)
    # invRot = np.linalg.inv(Rot)
    UBs = np.dot(Rot, UB)
    return UBs


def vec_normalTosurface(mat_labframe):
    """
    solve Mat * X = (0,0,1) for X
    for pure rotation invMat = transpose(Mat)

    TODO: add option sample angle and axis
    """
    # last row of matrix in sample frame
    return fromlab_tosample(mat_labframe)[2]


def vec_onsurface_alongys(mat_labframe):
    """
    solve Mat * X = (0,1,0) for X
    for pure rotation invMat = transpose(Mat)
    """
    return fromlab_tosample(mat_labframe)[1]


# ---------------------------------------------------------------------------
# ---------------------------WIRE TECHNIQUE ---------------------------------
# ---------------------------------------------------------------------------


# Following functions are for dealing with in depth  x-ray emission source

def find_yzsource_from_IM_uf(IM, uf, depth_z=0, anglesample=40.):
    """
    from vector IM in absolute frame  I origin, M point in CCD plane
    uf: unit vector in absolute frame joining Iprime (source) and M
    depth_z: in microns known vertical offset

    returns x and y position of emission source
    """
    ux, uy, uz = uf.T
    x, y, z = IM.T

    deltaZ = z - depth_z * 1000.  # in millimeters
    ratio = deltaZ / uz

    xsource = x - ratio * ux
    ysource = y - ratio * uy
    zsource = depth_z

    # rotation matrix from absolute to sample frame
    # Xs= R Xabs
    anglerad = anglesample * DEG
    ca = np.cos(anglerad)
    sa = np.sin(anglerad)
    R = np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]])
    # source position in sample frame
    tsource = np.array(xsource, ysource, zsource)
    xsource_s, ysource_s, zsource_s = np.dot(R, tsource)

    IIprime = tsource.T
    IIprime_s = np.array([xsource_s, ysource_s, zsource_s]).T


    return IIprime, IIprime_s


def IMlab_from_xycam(xcam, ycam, calib, verbose=0, signgam=SIGN_OF_GAMMA):
    """
    returns list of vector position of M (on CCD camera) in absolute frame 
    from pixels position vector in CCD frame
    """
    uflab_not_used, IMlab = calc_uflab(xcam, ycam, calib, returnAngles='uflab', signgam=signgam)

    if verbose:
        print("IMlab", IMlab)

    return IMlab


def IW_from_IM_onesource(IIprime, IM, depth_wire, anglesample=40., anglewire=40.):
    """
    from:
    II': single vector II' (2 elements= y,z) source position in absolute frame
    IM: array of vectors IM (3 elements= x,y,z) point on CCD in absolute frame
    depth_wire: height normal to the surface of the wire
    I origin of absolute frame and calibrated source emission

    returns:
    array of vectors wire position (y,z)  (hypothesis: wire parallel to Ox)
    """
    try:
        yIp, zIp = IIprime
    except ValueError:
        print("Next time, please use a 2elements source position (y,z)")
        xIp, yIp, zIp = IIprime

    angs = anglesample * DEG
    angw = anglewire * DEG

    x, y, z = (IM * 1.).T

    IH = np.array([0, -depth_wire * np.sin(angs), depth_wire * np.cos(angs)])

    slopeM = (z - zIp) / (y - yIp)
    slopefil = -np.tan(angw)

    cstM = slopeM * yIp + zIp
    cstH = slopefil * IH[1] + IH[2]

    yw = (cstM - cstH) / (slopeM - slopefil)
    zw = cstH - slopefil * yw

    return np.array([yw, zw])


def IW_from_source_oneIM(IIprime, IM, depth_wire, anglesample=40.):
    """
    TODO : MAY BE FALSE
    from:
    II': array of  vectors II' (2 elements= y,z) source position in absolute frame
    IM: SINGLE vector IM (3 elements= x,y,z) point on CCD in absolute frame
    depth_wire: height normal to the surface of the wire
    I origin of absolute frame and calibrated source emission

    returns:
    array of vectors wire position (y,z)  (hypothesis: wire parallel to Ox)
    """

    yIp, zIp = (IIprime * 1.).T
    x, y, z = IM

    slope = (zIp - z) / (yIp - y)

    ang = anglesample * DEG
    tw0 = np.tan(ang)

    yw = (yIp * slope + zIp + depth_wire) / (slope - tw0)
    zw = tw0 * yw + depth_wire


    return np.array([yw, zw])


def find_yzsource_from_xycam_uf(OM, uf, calib, depth_z=0, anglesample=40.):
    """
    from:
    OM: list of vectors OM (2 elements) in CCD plane in CCD frame (pixels unit)
    uf: list of unit vectors (3 elements) in absolute frame
    depth_z:  known vertical offset of the beam with respect in microns
                default value = 0 (source is along the line passing through I origin for CCD calibration)


    returns:
    list of position [y,z] of emission source in absolute frame
    list of position [y,z] of emission source in sample frame
    """
    xcam, ycam = OM.T

    IMlab = IMlab_from_xycam(xcam, ycam, calib)

    IMlab_yz = IMlab[:, 1:]

    return find_yzsource_from_IM_uf(IMlab_yz, uf, depth_z=depth_z, anglesample=anglesample)


def find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib, anglesample=40.):
    """
    from:
    OMs: array of 2 vectors OM (2 elements) in CCD plane in CCD frame (pixels unit): array([OM1,OM2])
    IWs: array of 2 position vectors (2 elements= [y,z]) in absolute frame of wire (which is parallel to Ox): array([IW1,IW2]
    
    returns:
    y,z position of source of emission (hypothesis x=0)
    """
    xcam, ycam = OMs.T

    IMlab = IMlab_from_xycam(xcam, ycam, calib)

    X_IM_notused, Y_IM, Z_IM = IMlab.T
    y1, y2 = Y_IM
    z1, z2 = Z_IM

    YW, ZW = IWs.T
    u1, u2 = YW
    v1, v2 = ZW

    A = u1 - y1
    B = v1 - z1
    C = u2 - y2
    D = v2 - z2

    E = -y1 * B + z1 * A
    F = -y2 * D + z2 * C

    determ = A * D - B * C

    ysource = (C * E - A * F) / determ
    zsource = (D * E - B * F) / determ

    return np.array([ysource, zsource])


def find_yzsource_from_2xycam_2yzwire_version2(OMs, IWs, calib, anglesample=40., verbose=0):
    """
    from:
    OMs: array of 2 vectors OM (2 elements) in CCD plane in CCD frame (pixels unit): array([OM1,OM2])
    IWs: array of 2 position vectors (2 elements= [y,z]) in absolute frame of wire (which is parallel to Ox): array([IW1,IW2]

    assumption: xsource = 0

    returns:
    position of source of emission

    """
    xcam, ycam = OMs.T

    IMlab = IMlab_from_xycam(xcam, ycam, calib)

    IM_1, IM_2 = IMlab[:2, 1:]  # y,z of IMs

    IW_1, IW_2 = IWs[:2]  # y,z of IW

    A, B = GT.ShortestLine(IM_1, IW_1, IM_2, IW_2)

    if verbose:
        print("line1 points", IM_1, IW_1)
        print("line2 points", IM_2, IW_2)
        print("two points", A, B)
        print("distance ", np.sqrt(np.dot(A - B, A - B)))

    return (A + B) / 2.


def find_multiplesourcesyz_from_multiplexycam_multipleyzwire(OMs, Wire_abscissae, calib, anglesample=40., wire_height=0.3, verbose=0):
    """
    from:
    OMs: array of n vectors OM (2 elements) in CCD plane in CCD frame (pixels unit): array([OM1,OM2, ..., OMn])
    IWs: array of n wire abscissae of wire (which is parallel to Ox): array([W1,W2,...,Wn]
    Wire (strictly parallel to x) travels strictly at anglesample from horizontal plane (defined by ui,xbet and CCD plane in calibration). Wire abscissa is zero when wire is on top of point I of calibration at height wire_height. Wire abscissa increases in same direction as y (and Xray beam).

    assumption: 1) xsource = 0
                2) sources lying in yOz plane may only differ from incident direction in this plane

    returns:
    all positions of source of emission from all pairs [spot,Wire_abscissa]

    """

    IWs = IW_from_wireabscissa(Wire_abscissae, wire_height, anglesample=anglesample)  # array of [y,z]
    IWs = IWs.T
    pairs = GT.pairs_of_indices(len(IWs))

    if verbose:
        print(Wire_abscissae)
        print(OMs)
        print(IWs)
        print(pairs)

    results = []
    for k in range(len(pairs)):
        p = pairs[k]
        if verbose:
            print("p", p)
            print("oms", np.take(OMs, p, axis=0))
            print("iws", np.take(IWs, p, axis=0))

        results.append(find_yzsource_from_2xycam_2yzwire_version2(np.take(OMs, p, axis=0),
                                                                  np.take(IWs, p, axis=0),
                                                                  calib,
                                                                  anglesample=anglesample,
                                                                  verbose=0))

    return np.array(results)


def IW_from_wireabscissa(abscissa, wire_height, anglesample=40.):
    """
    from:
    abscissa of wire and wire height from sample surface inclined by anglesample (deg)

    returns:
    absolute coordinate of wire (hypothesis x is undetermined)
    """
    ang = anglesample * DEG
    sw0 = np.sin(ang)
    cw0 = np.cos(ang)
    y = abscissa * cw0 - wire_height * sw0
    z = abscissa * sw0 + wire_height * cw0
    return np.array([y, z])


def Wireabscissa_from_IW(IWy, IWz, wire_height, anglesample=40.):
    """
    from:
    absolute coordinate of wire (hypothesis x is undetermined)
    wire height from sample surface inclined by anglesample (deg)

    returns:
    abscissa of wire along ysample inclined by anglesample from point at IIw0 distance from I
    """
    angs = anglesample * DEG

    IH = np.array([0, -wire_height * np.sin(angs), wire_height * np.cos(angs)])

    # WH = array([0, IWy, IWz]) -  IH

    WHy = IWy - (-wire_height * np.sin(angs))
    WHz = IWz - wire_height * np.cos(angs)

    # add sign simply for angle wire  around 40 deg
    signe = np.where(IWz > wire_height * np.cos(angs), 1, -1)


    return signe * np.sqrt(WHy ** 2 + WHz ** 2)


def twotheta_from_wire_and_source(ysource, Height_wire, Abscissa_wire, anglesample=40):
    """
    point moving parallel to sample surface in Oyz plane

    ysource= II'  abscissa of spots emission (I') from calibration origin source (I)
    Height_wire= height of moving point Iw from sample surface
    Abscissa_wire= abscissa of moving point along its straight trajectory from point Iw0
            which is in between I and microscope  (normal at sample surface)

    returns scattering angle 2theta from y direction to I'Iw (no x component)
    """

    lambda_angle = np.arctan(Height_wire * 1. / Abscissa_wire) / DEG + anglesample
    coslambda = np.cos(lambda_angle * DEG)
    yprime = ysource * 1. / Abscissa_wire

    return np.arccos((coslambda - yprime) / np.sqrt(1 + yprime ** 2 - 2 * yprime * coslambda)) / DEG


def convert_xycam_from_sourceshift(OMs, IIp, calib, verbose=0, signgam=SIGN_OF_GAMMA):
    """
    From x,y on CCD camera (OMs) and source shift (IIprime)
    compute modified x,y values for the SAME calibration (calib)(for further analysis)
    
    return new value of x,y 
    """
    xcam, ycam = OMs.T
    uflab, IM = calc_uflab(xcam, ycam, calib, returnAngles=0, verbose=0, pixelsize=165. / 2048, signgam=signgam)  # IM normalized
    # IM vectors
    # IM=IprimeM_from_uf(uflab,array([0,0,0]),calib,verbose=0)
    OM = calc_xycam(uflab, calib,
                    signgam=signgam, energy=1, offset=None,
                    verbose=0, returnIpM=False, pixelsize=165. / 2048)
    x0cam, y0cam, th0, E0 = OM
    # IpM=IpI + IM= IM-IIP
    IpM = IM - IIp
    nor = np.sqrt(np.sum(IpM ** 2, axis=1))
    # new uf prime
    ufp = IpM * 1. / np.reshape(nor, (len(nor), 1))
    OpMp = calc_xycam(ufp, calib,
                      signgam=signgam, energy=1, offset=None,
                      verbose=0, returnIpM=False, pixelsize=165. / 2048)
    xpcam, ypcam, thp, Ep = OpMp

    if verbose:
        print("\n for source at I and Iprime")
        print("X in pixel")
        print(x0cam, xpcam)
        print("Y in pixel")
        print(y0cam, ypcam)
        print("2theta in deg")
        print(2.*th0, 2.*thp)
        print("Energy")
        print(E0, Ep)
        # print uflab,ufp
        print("\n for IIprime (mm)", IIp)
        print("And calibration ", calib)
        print("\n shift to add to X")
        print(xpcam - x0cam)
        print("shift to add to Y")
        print(ypcam - y0cam)
        print("shift to add to 2theta in deg")
        print(2.*(thp - th0))
        print("shift to add to Energy (IN eV)")
        print(1000.*(Ep - E0))


def absorbprofile(x, R, mu, x0):

    cond = (np.fabs(x - x0) <= R)
    print(cond)
    print(cond.dtype)
    def Absorbfunction(x, radius=R, coef=mu, center=x0):
            return np.exp(-2 * coef * np.sqrt(radius ** 2 - (x - center) ** 2))

    yabs = list(map(Absorbfunction, x))
    print(yabs)
    # y=np.piecewise(x,[cond],[Absorbfunction,1.],radius=R,coef=mu,center=x0)
    y = np.select([cond], [yabs], 1.)
    return y


def lengthInSample(depth, twtheta, chi, omega):
    """ compute geometrical lengthes in sample from impact point (I) at the surface to a point (B) where xray are scattered
    (or fluorescence is emitted) and finally to a point (C) lying at the sample surface
    (intersection of line with unit vector u with sample surface plane tilted by omega)
    
    WARNING:
    twtheta and chi can be misleading. they correspond simply to angles defining a direction
    like kf is the unit vector describing the scattered direction
    Assumption ius made that angles of unit vector from B to C (or to detector frame pixel) are 2theta and chi
    For large depth D, unit vector direction is not given by 2theta and chi angles
    as they are used for describing the scattering direction from point I and a given detector frame position
    (you should then compute the two angles correction , actually chi is unchanged, and is 2theta change is approx
    d/ distance .i.e. 3 10-4 for d=20 µm and CCD at 70 mm)
    
    (incoming beam coming from the right positive x direction
    IB = (-D,0,0)
    BC =(xc+D,yc,zc)
    
    legnth BC is proportional to the depth D
    
    """
    D=depth*1.
    c2theta = np.cos(twtheta*DEG)
    s2theta = np.sin(twtheta*DEG)
    cchi = np.cos(chi*DEG)
    schi = np.sin(chi*DEG)
    comega=np.cos(omega*DEG)
    somega = np.sin(omega*DEG)
    
    factor = D*somega/(-c2theta*somega+s2theta*cchi*comega)
    xc= factor*(-c2theta)
    yc= factor*(-s2theta*schi)
    zc= factor*(s2theta*cchi)
    
    BC = np.sqrt((xc+D)**2+yc**2+zc**2)
    
    print('[x,y,z] of BC')
    print(np.array([twtheta,chi,xc,yc,zc]).T)
    
    Ratio_BC_over_D = BC/D
    
    return D+BC, BC,Ratio_BC_over_D, D
    

# --------------------   TEST & EXAMPLES functions ---------------------------------
def test_1():
    """
    test
    """
    abscissa = np.array([5, -5., 6., 0.001, 2.35, 52, 0.])
    IIw0 = 10.
    yoyo = IW_from_wireabscissa(abscissa, IIw0, anglesample=40.)
    tata = Wireabscissa_from_IW(yoyo[0], yoyo[1], IIw0, anglesample=40.)
    print("abscissa", abscissa)
    print("tata", tata)
    print("This two arrays must be equal!")


def test_2():
    """
    test
    """
    calib = [70, 1024, 1024, .0, -0.]
    uflab = np.array([[0.5, -.5, 0.5], [0., -0., 1.], [0.0, -.5, 0.5], [0.5, 0.5, .5]])
    xcam, ycam, th = OM_from_uf(uflab, calib, energy=0, offset=None, verbose=0)
    uflab_2, IMlab_2 = calc_uflab(xcam, ycam, calib, returnAngles='uflab', verbose=0)
    print("****************")
    print("start", uflab)
    print("final", uflab_2)
    print("This two arrays must be equal!")


def test_3(anglesample, calib, IIprime):
    """
    test
    """
    IIprime = np.array(IIprime)
    IM = np.array([[1, 1, 1], [1, 0, 1], [1, -1, 1], [0, 1, 1], [0, 0, 5], [10, 5, 4]])
    nor = np.sqrt(np.sum(IM ** 2, axis=1))
    uf = IM * 1. / np.reshape(nor, (len(nor), 1))
    IpM = IprimeM_from_uf(uf, IIprime, calib, verbose=0)
    print("IpM", IpM)

    height_wire = 10.
    IW = IW_from_IM_onesource(IIprime[1:], IpM, height_wire, anglesample=anglesample)
    print("IW", IW)

    print("finding source origin from two reflectionx")
    OM = OM_from_uf(uf, calib, energy=0, offset=None, verbose=0)[:2]
    print("OM", end=' ')
    IWs = np.transpose(IW)[:2]
    OMs = np.transpose(OM)[:2]
    print("Using 2 reflections")
    print("2 OMs", OMs)
    print("2 IWs", IWs)
    print("------------")
    ysource, zsource = find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib, anglesample=anglesample)
    print("ysource,zsource", ysource, zsource)


def test_4(IIprime, height_wire, errorOM1, errorOM2):
    """
    test
    """
    anglesample = 40.
    calib = [70, 1024, 1024, .0, -0.]
    IIprime = np.array(IIprime)
    IM = np.array([[1, 1, 1], [1, 0, 1], [1, -1, 1], [0, 1, 1], [0, 0, 5], [10, 5, 4]])
    nor = np.sqrt(np.sum(IM ** 2, axis=1))
    uf = IM * 1. / np.reshape(nor, (len(nor), 1))
    IpM = IprimeM_from_uf(uf, IIprime, calib, verbose=0)
    print("IpM", IpM)

    IW = IW_from_IM_onesource(IIprime[1:], IpM, height_wire, anglesample=anglesample)
    print("IW", IW)

    print("finding source origin from two reflectionx")
    OM = OM_from_uf(uf, calib, energy=0, offset=None, verbose=0)[:2]
    print("OM", end=' ')

    print("Using 2 reflections")
    IWs = np.transpose(IW)[:2]
    OMs = np.transpose(OM)[:2]
    errorOM1, errorOM2 = np.array(errorOM1), np.array(errorOM2)
    OMs = OMs + np.array([errorOM1, errorOM1])
    print("2 OMs", OMs)
    print("2 IWs", IWs)
    print("------------")
    ysource, zsource = find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib, anglesample=anglesample)
    print("ysource,zsource", ysource, zsource)

def test_5(IIprime, height_wire, arrayindex, errorWabscissa1, errorWabscissa2):
    """

    Simulate some reflections of a shifted source / calibrated emission point
    and corresponding wire abscissa

    wire is assumed to be parallel to sample surface (inclined by anglesample, see below) and strictly along 0x axis

    IIprime: 3 elements vector of source position (mm)
    height_wire: height of the wire (mm)
    errorWabscissa1,errorWabscissa2: error in measuring wireabscissa where the reflection is apparently extinguished

    Then retrieve the source position from 2 reflections and 2 measured wireabscissa 
    
    """

    anglesample = 40.
    calib = [100, 1024, 1024, .0, -0.]
    IIprime = np.array(IIprime)
    IM = np.array([[1, 1, 3], [1, 0, 3], [1, -1, 3], [0, 1, 3], [0, 0, 5], [4, 5, 10]])
    nor = np.sqrt(np.sum(IM ** 2, axis=1))
    uf = IM * 1. / np.reshape(nor, (len(nor), 1))
    IpM = IprimeM_from_uf(uf, IIprime, calib, verbose=0)
    print("IpM", IpM)

    IW = IW_from_IM_onesource(IIprime[1:], IpM, height_wire, anglesample=anglesample)
    print("IW", IW)


    OM = OM_from_uf(uf, calib, energy=0, offset=None, verbose=0)[:2]
    print("positions on CCD: OMs", OM)

    _twtheta, _chi = calc_uflab(OM[0], OM[1], calib, returnAngles=1, verbose=0, pixelsize=165. / 2048)

    print("finding source origin from two reflections")
    print("Using 2 reflections of index:", arrayindex)
    # this is where two reflections can be chosen among others
    IWs = np.take(IW.T, np.array(arrayindex), axis=0)
    OMs = np.take(OM.T, np.array(arrayindex), axis=0)

    twthe = np.take(_twtheta, np.array(arrayindex))

    chi = np.take(_chi, np.array(arrayindex))


    # simulated wire abscissa
    W1, W2 = Wireabscissa_from_IW(IWs[:, 0], IWs[:, 1], height_wire, anglesample=anglesample)
    # introducing abscissa errors
    W1 = W1 + errorWabscissa1
    W2 = W2 + errorWabscissa2
    print("W1,W2", W1, W2)
    IWs = (IW_from_wireabscissa(np.array([W1, W2]), height_wire, anglesample=anglesample)).T
    print("2 OMs", OMs)
    print("2 IWs", IWs)
    print("2 2theta and 2 chi", twthe, chi)
    print("2 Wire abscissae", W1, W2)
    print("------------")
    ysource, zsource = find_yzsource_from_2xycam_2yzwire_version2(OMs, IWs, calib, anglesample=anglesample)
    print("\n*******************\nRetrieving source position\n")
    print("With wire's height (mm):  ", height_wire)
    print("and 2 measured abscissa errors", errorWabscissa1, errorWabscissa2)
    print("\nSource found at")
    print("ysource,zsource", ysource, zsource)
    print("Source simulated at [y,z] (mm)")
    print(IIprime)
    print("Source errors [y,z] (mm)")
    print(np.array([ysource, zsource]) - IIprime[1:])
    print("\n ***********\n")


def test_offset_xraysource():


    calib = [70, 1024, 1024, .0, -0.]
    uflab = np.array([[0.5, -.5, 0.5], [0., -0., 1.], [0.0, -.5, 0.5], [0.5, 0.5, .5]])

    print("Without offset")
    X, Y, th0, E = calc_xycam(uflab, calib, energy=1, offset=[0, 0, 0.])

    print("Xcam (mm)", X * 165 / 2048.)
    print("Ycam (mm)", Y * 165 / 2048.)
    print("Xcam (pixel)", X)
    print("Ycam (pixel)", Y)
    print("theta ", th0)
    print("2theta ", 2 * th0)
    print("energy ", E)

    print("\n With offset\n")
    X, Y, th0, E = calc_xycam(uflab, calib, energy=1, offset=[0., .01, 0.])
    print("Xcam (mm)", X * 165 / 2048.)
    print("Ycam (mm)", Y * 165 / 2048.)
    print("Xcam (pixel)", X)
    print("Ycam (pixel)", Y)
    print("theta ", th0)
    print("2theta ", 2 * th0)
    print("energy ", E)


def test_sourcetriangulation():
    calib = [68., 1024, 1024, 1.0, -2.]
    twicetheta = np.array([90., 80., 90., 80., 150., 60])
    chi = np.array([0., 0., 25., -25., 37., -23])
    uflab = uflab_from2thetachi(twicetheta, chi, verbose=0)

    print("Simulation of data ---")
    print("\nWithout offset\n")
    posI = np.array([0, 0, 0.])
    IM0 = IprimeM_from_uf(uflab, posI, calib, verbose=0)
    X0, Y0, th0 = calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print(X0, Y0)
    print("IM0", IM0)
    depth_wire = 0.01
    IW0y, IW0z = IW_from_IM_onesource(posI[1:], IM0, depth_wire, anglesample=40.)

    print("\nWith offset\n")
    posI = np.array([0, 0.01, -0.01])
    print("posI", posI)
    IpM = IprimeM_from_uf(uflab, posI, calib, verbose=0)
    IM1 = IpM + posI
    X1, Y1, th1 = calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print("X1", X1)
    print("Y1", Y1)
    print("IM1", IM1)

    H_wire = 0.01
    IW1y, IW1z = IW_from_IM_onesource(posI[1:], IM1, H_wire, anglesample=40.)

    print("IW1y,IW1z", IW1y, IW1z)
    Wireabscisa_1 = Wireabscissa_from_IW(IW1y, IW1z, H_wire, anglesample=40.)
    print("Wireabscisa_1", Wireabscisa_1)

    print("\n-------------------------------------")
    print("finding source origin from two reflectionx")
    OMs = np.transpose(np.array([X1, Y1]))[:2]
    IWs = np.transpose(np.array([IW1y, IW1z]))[:2]
    print("Using 2 reflections")
    print("OMs", OMs.tolist())
    print("IWs", IWs.tolist())
    print("------------")


    ysource, zsource = find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib, anglesample=40.)

    print("ysource,zsource", ysource, zsource)


def test_sourcefinding():
    calib = [68., 1024, 1024, 1.0, -2.]
    twicetheta = np.array([90., 80., 90., 80., 150., 60])
    chi = np.array([0., 0., 25., -25., 37., -23])
    uflab = uflab_from2thetachi(twicetheta, chi, verbose=0)

    print("Simulation of data ---")
    print("\nWithout offset\n")
    posI = np.array([0, 0, 0.])
    IM0 = IprimeM_from_uf(uflab, posI, calib, verbose=0)
    X0, Y0, th0 = calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print(X0, Y0)
    print("IM0", IM0)
    depth_wire = 0.01
    IW0y, IW0z = IW_from_IM_onesource(posI[1:], IM0, depth_wire, anglesample=40.)

    print("\nWith offset\n")
    posI = np.array([0, 0.01, -0.01])
    print("posI", posI)
    IpM = IprimeM_from_uf(uflab, posI, calib, verbose=0)
    IM1 = IpM + posI
    # coordinates on CCD for this source and the same ufs
    X1, Y1, th1 = calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print("X1", X1)
    print("Y1", Y1)
    print("IM1", IM1)

    H_wire = 0.3
    IW1y, IW1z = IW_from_IM_onesource(posI[1:], IM1, H_wire, anglesample=40.)

    print("IW1y,IW1z", IW1y, IW1z)
    Wireabscisa_1 = Wireabscissa_from_IW(IW1y, IW1z, H_wire, anglesample=40.)
    print("Wireabscisa_1", Wireabscisa_1)

    print("\n\n-------------------------------------")
    print("finding source origin all reflections masking")
    print("originally: posI: ", posI, "  Hwire: ", H_wire)
    OMs = np.transpose(np.array([X1, Y1]))
    Wire_abscissae = Wireabscisa_1
    sourcepos = find_multiplesourcesyz_from_multiplexycam_multipleyzwire(OMs, Wire_abscissae, calib, anglesample=40., wire_height=H_wire)
    print("all results", sourcepos)

    largey = np.where(abs(sourcepos[:, 0]) > 1)[0]
    largez = np.where(abs(sourcepos[:, 1]) > 1)[0]
    badpoints_indices = set(largey).union(set(largez))
    print(badpoints_indices)
    list(badpoints_indices)
    # put -1 numericql tqg
#    print "\n mean source position", mean(tt, axis=0)
#    print "------------"
#
#    print "adding some noise in"


def test_correction_1():
    """
    TEST: Reading experimental points=(x,y)
    """
    print("TEST: Reading experimental points=(x,y)")
    param = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201]  # Nov 09 J. Villanova BM32
    peaksfilename = 'SS_0170.peaks'
    twicetheta, chi, dataintensity, data_x, data_y = Compute_data2thetachi(peaksfilename,
                                                                     (0, 1, 2),
                                                                     1,  # 1 for .peaks
                                                                     sorting_intensity='yes',
                                                                     param=param  # None
                                                                     )

    print(twicetheta)
    LTGeo.writefile_cor('polyZrO2_test', twicetheta, chi, data_x, data_y, dataintensity, param=param, initialfilename=peaksfilename)


def test_correction_2():
    """
    TEST: Reading experimental points=(x,y)
    """
    print("TEST: Reading experimental points=(x,y)")
    param = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201]  # Nov 09 J. Villanova BM32
    peaksfilename = 'Ge.peaks'
    twicetheta, chi, dataintensity, data_x, data_y = Compute_data2thetachi(peaksfilename,
                                                                     (0, 1, 2),
                                                                     1,  # 1 for .peaks
                                                                     sorting_intensity='yes',
                                                                     param=param  # None
                                                                     )

    print(twicetheta)
    LTGeo.writefile_cor('Ge_test', twicetheta, chi, data_x, data_y, dataintensity, param=param, initialfilename=peaksfilename)


def test_correction_3():
    """
    TEST: Reading experimental points=(x,y)
    """
    print("TEST: Reading experimental points=(x,y)")
    param = [69.66055, 895.27118, 960.77417, 0.8415, -.31818]  # Nov 09 J. Villanova BM32
    peaksfilename = 'Ge_run41_1_0003.peaks'
    twicetheta, chi, dataintensity, data_x, data_y = Compute_data2thetachi(peaksfilename,
                                                                     (0, 1, 2),
                                                                     1,  # 1 for .peaks
                                                                     sorting_intensity='yes',
                                                                     param=param  # None
                                                                     )

    print(twicetheta)
    LTGeo.writefile_cor('Ge_run41_1_0003', twicetheta, chi, data_x, data_y, dataintensity, param=param, initialfilename=peaksfilename)


def find_referencepicture(anglesample=40,
                          penetration=0,
                          calib=np.array([69.1219, 1074.11, 1109.11, 0.32857, 0.00817]),
                          combination=0,
                          falling_or_rising=0,
                          wire_height=0.3,
                          verbose=0, veryverbose=0):
    """
    Return the picture corresponding to the reference picture
    (in find_multiplesourcesyz_from_multiplexycam_multipleyzwire())
    according to the 'good ylab' (like 0mm at the sample surface,
    and "penetration" [mm] if the reference source point (I) is not at the surface).  
    """
    # TODO (object way): put step in argument of this function. Transforme test_Gec() as a generic function or put OMs, IWs, etc as arguments of find_referencepicture()
    ylab_at_the_good_depth = 9999
    k_at_the_good_depth = 9999
    xbet = calib[3]
    step_temp_array = test_Gec(anglesample=anglesample, referencepicture=0, wire_height=wire_height)  # Just to know the step. TODO : put in argument
    step = step_temp_array[2]

    if verbose :
        print(calib)
        print(xbet)
        print(step)

    for k in range(0, 700):
        temp = test_Gec(anglesample=anglesample, referencepicture=k, wire_height=wire_height)
        ylab = temp[falling_or_rising][combination][0]
        "falling_or_rising : 0 for falling edge (first edge); 1 for rising edge (second edge)"
        if veryverbose :
            print(ylab)
            print(k)
        if abs(ylab - penetration) < ylab_at_the_good_depth :
            ylab_at_the_good_depth = ylab
            k_at_the_good_depth = k

    """
    From the reference taken by find_multiplesourcesyz_from_multiplexycam_multipleyzwire()
    """
    k_cut_the_direct_beam = k_at_the_good_depth - (wire_height / np.tan(anglesample * DEG)) / step
    yf_cut_the_direct_beam = k_cut_the_direct_beam * step

    # TODO: k_at_90deg_under_sample=

    return ["k_cut_the_direct_beam=", k_cut_the_direct_beam,
            "yf_cut_the_direct_beam=", yf_cut_the_direct_beam,
            "k_at_the_good_depth=", k_at_the_good_depth,
            "ylab_at_the_good_depth=", ylab_at_the_good_depth]


def test_Gec(anglesample=40.,
             referencepicture=245,
             wire_height=0.3):

    """ Test function for Gec_XXXX.mccd
    Desctiption of the scan (to know the step allowing the convertion: picture number <-> yf)
    """

    yf_start = -1.66076
    yf_end = -0.960765
    nb_interval = 700  # nb_inteval = nb_of_step - 1

    step = abs((yf_end - yf_start) / nb_interval)

    """
    Position in pixels of each studied peak:
    """
    OM1 = np.array([1071.2, 1238.2])
    OM2 = np.array([1546.0, 943.1])
    OM3 = np.array([558.6, 523.2])
    OM4 = np.array([992.5, 144.5])
    OM5 = np.array([1023.4, 493.2])
    OMs = np.array([OM1, OM2, OM3, OM4, OM5])

    """
    yf for the quenching of each peak:
    With the falling edge of the intensity (when the wire begins to shadow the peak):
    """
    W1_falling = step * (531.66 - referencepicture)
    W2_falling = step * (358.37 - referencepicture)
    W3_falling = step * (205.6 - referencepicture)
    W4_falling = step * (125.7 - referencepicture)
    W5_falling = step * (199.95 - referencepicture)
    Wire_abscissae_falling = np.array([W1_falling, W2_falling, W3_falling, W4_falling, W5_falling])

    """
    With the rising edge (second edge):
    """
    W1_rising = step * (605.65 - referencepicture)
    W2_rising = step * (415.17 - referencepicture)
    W3_rising = step * (255.95 - referencepicture)
    W4_rising = step * (176.6 - referencepicture)
    W5_rising = step * (249.7 - referencepicture)
    Wire_abscissae_rising = np.array([W1_rising, W2_rising, W3_rising, W4_rising, W5_rising])

    """
    Calibration:
    """
    calib = np.array([69.1219, 1074.11, 1109.11, 0.32857, 0.00817])

    """
    Calculation: 
    """
    res_falling = find_multiplesourcesyz_from_multiplexycam_multipleyzwire(OMs, Wire_abscissae_falling, calib, anglesample, wire_height, 0)
    res_rising = find_multiplesourcesyz_from_multiplexycam_multipleyzwire(OMs, Wire_abscissae_rising, calib, anglesample, wire_height, 0)

    res = [res_falling, res_rising, step]


    return res

# ------------------------------------------------------------
# --------------------------  MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    calib1 = [70, 1000., 1100, -.2, .3]

    anglesample = 40.

    height_wire = 0.10  # mm
    IIprime = np.array([0., 0, 0])

    ycam = 1.*np.arange(0, 2048, 40)[::-1]
    xcam = 1024.*np.ones(len(ycam))

    IMlab1 = IMlab_from_xycam(xcam, ycam, calib1, verbose=0)

    IWy1, IWz1 = IW_from_IM_onesource(IIprime[1:], IMlab1, height_wire, anglesample=anglesample, anglewire=anglesample)

    Wireabscissa1 = Wireabscissa_from_IW(IWy1, IWz1, height_wire, anglesample=anglesample)

    #-------------------------------------

    calib2 = [60, 1000., 1100, -.2, .3]
    anglesample = 40.

    height_wire = 0.10  # mm
    IIprime = np.array([0., 0, 0])

    ycam = 1.*np.arange(0, 2048, 40)[::-1]
    xcam = 1024.*np.ones(len(ycam))

    IMlab2 = IMlab_from_xycam(xcam, ycam, calib2, verbose=0)

    IWy2, IWz2 = IW_from_IM_onesource(IIprime[1:], IMlab2, height_wire, anglesample=anglesample, anglewire=anglesample)

    Wireabscissa2 = Wireabscissa_from_IW(IWy2, IWz2, height_wire, anglesample=anglesample)

    import pylab as p

    p.plot(ycam, Wireabscissa1, ycam, Wireabscissa2)

    p.show()

    raise ValueError("End of example")

    if len(sys.argv) == 1:
        # test_correction_1()
        # test_correction_2()
        # test_correction_3()
        # test_offset_xraysource()
        test_sourcetriangulation()
        sys.exit()


    print("\n *************\n\nfrom file:", sys.argv[1])

    filename = sys.argv[1]
    # filename="NbSe3_11Mar07_0012_936pics.dat"
    # filename="I832a0325.DAT"
    # filename="CdTe_I832_0325_peak.dat"
    prefix = 'sUrHe'
    indexfile = '0103'
    suffix = '.pik'
    col_X = 0
    col_Y = 1
    col_I = 2  # index (starting from 0) of the intensity column
    nblines_headertoskip = 0
    Intensitysorted = 0  # =1 if intensity sorting must be done for the outputfile, =0 means that sorting already done in input file or sorting not needed

    # filename=prefix+indexfile+suffix

    # twicetheta,chi,dataintensity,data_x,data_y=Compute_data2thetachi(filename,(col_X,col_Y,col_I),nblines_headertoskip)

    # plotXY2thetachi(data_x,data_y,twicetheta,chi,mostintense=200)
    # LTGeo.writefile_cor(prefix+indexfile,twicetheta,chi,data_x,data_y,sortedexit=Intensitysorted)


    # for doing a files serie
    def series():
        """
        doing a files serie
        """
        for index in range(620, 1276):
            filename = prefix + '%04d' % index + suffix
            twicetheta, chi, dataintensity, data_x, data_y = Compute_data2thetachi(filename, (col_X, col_Y, col_I), nblines_headertoskip)
            LTGeo.writefile_cor(prefix + '%04d' % index, twicetheta, chi, data_x, data_y, dataintensity, sortedexit=Intensitysorted)

    # series()


