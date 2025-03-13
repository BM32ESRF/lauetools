# -*- coding: utf-8 -*-
r"""
IOimagefile module is made for reading data contained in binary image file
fully or partially.

More tools can be found in LaueTools package at sourceforge.net and gitlab.esrf.fr
March 2020
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

# built-in modules
import sys
import os
import copy
import struct
from scipy import ndimage as scind
import h5py 


try:
    SKIMAGE = True
    from skimage.external import tifffile
except:
    SKIMAGE = False

import LaueTools.generaltools as GT

# third party modules
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        import fabio

    FABIO_EXISTS = True
except ImportError:
    print("Missing fabio module. Please install it if you need open some tiff images "
            "from the sCMOS camera. However, libtiff or PIL can do the job!")
    FABIO_EXISTS = False

try:
    from libtiff import TIFF, libtiff_ctypes

    libtiff_ctypes.suppress_warnings()
    LIBTIFF_EXISTS = True
except (ImportError, ValueError, AttributeError, NameError):
    print("-- warning: Missing library libtiff, Please install: pylibtiff if you need open some tiff images. However, Fabio or PIL can do the job!")
    LIBTIFF_EXISTS = False

try:
    from PIL import Image

    PIL_EXISTS = True
except ImportError:
    print("Missing python module called PIL. Please install it if you need open some tiff. However, Fabio or libtiff can do the job!  "
            "images from vhr camera")
    PIL_EXISTS = False
import numpy as np


# lauetools modules
if sys.version_info.major == 3:
    from . import dict_LaueTools as DictLT
    from . import imageprocessing as ImProc
else:
    import dict_LaueTools as DictLT
    import imageprocessing as ImProc

listfile = os.listdir(os.curdir)

def stringint(k, n):
    r""" returns string of integer k with n zeros padding
    (by placing zeros before to have n characters)

    :param k: integer to convert
    :param n: nb of digits for zero padding

    :return: string of length n containing integer k

    Example: 1 -> '0001'
    15 -> '0015'
    """
    #     strint = str(k)
    #     res = '0' * (n - len(strint)) + strint

    encodingdigits = "{" + ":0{}".format(int(n)) + "}"
    res = encodingdigits % k

    return res

# --- ---------------- read images functions
def setfilename(imagefilename, imageindex, nbdigits=4, CCDLabel=None, verbose=0):
    r"""
    reconstruct filename string from imagefilename and update filename index with imageindex

    :param imagefilename: filename string (full path or not)
    :param imageindex: index in filename
    :type imageindex: string

    :return filename: input filename with index replaced by input imageindex
    :rtype: string
    """
    if nbdigits in (0,'0','None'):
        nbdigits = None
    #     print "imagefilename",imagefilename
    if imagefilename.endswith("mccd"):
        lenext = 5 #length of extension including '.'
        imagefilename = imagefilename[: -(lenext + nbdigits)] + "{:04d}.mccd".format(imageindex)

    elif CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        #default file extension for sCMOS camera
        ext = "tif"
        lenext = 4  #length of extension including '.'

        if imagefilename.endswith("tiff"):
            ext = "tiff"
            lenext = 5
        
        prepart = str(imagefilename).rsplit('.',1)[0]
        prefix0, strindex = str(prepart).rsplit('_', 1)

        if nbdigits is not None or nbdigits == 0:  # zero padded   (4 digits)
            #imagefilename = prefix0 + "_{:04d}.{}".format(imageindex, ext)
            # spec or BLISS 
            encodingdigits = "%%0%dd" % nbdigits
            imagefilename = prefix0 +'_'+encodingdigits%imageindex + '.'+ ext

        else:  # bliss psl control of sCMOS
            imagefilename = prefix0 + "_{}.{}".format(imageindex, ext)


    elif CCDLabel in ("EIGER_4Mstack",):
        # only stackimageindex is changed not imagefilename
        pass

    #     print "imagefilename archetype", imagefilename

    elif CCDLabel in ("MaxiPIXCdTe","EIGER_4MCdTe",'EIGER_1M'):
        
        #default file extension for sCMOS camera
        ext = "h5"
        lenext = 3  #length of extension including '.'
        
        prepart = str(imagefilename).rsplit('.',1)[0]
        prefix0, strindex = str(prepart).rsplit('_', 1)

        if prepart.startswith(('mpxcdte_', 'eiger1_')):

            if nbdigits is not None or nbdigits == 0:  # zero padded   (4 digits)
                #imagefilename = prefix0 + "_{:04d}.{}".format(imageindex, ext)
                # spec or BLISS 
                encodingdigits = "%%0%dd" % nbdigits
                imagefilename = prefix0 +'_'+encodingdigits%imageindex + '.'+ ext

            else:  # bliss psl control of sCMOS
                imagefilename = prefix0 + "_{}.{}".format(imageindex, ext)

        else: # same file but stackingindex will change
            print('same file : %s but stackingindex inside file will change!'%prepart)
            pass 
    elif imagefilename.endswith("mar.tiff"):

        imagefilename = imagefilename[: -(9 + nbdigits)] + "{:04d}_mar.tiff".format(imageindex)

    elif imagefilename.endswith("mar.tif"):
        # requires two underscores with number in between
        # map1_105_mar.tif
        sname = imagefilename.split("_")
        if len(sname) >= 4:  # pathfilename contain more than 1 underscore

            sname2 = sname[-2] + "_" + sname[-1]
            prefix = imagefilename.rstrip(sname2)
        else:
            prefix = imagefilename.rsplit("_")[0]
        imagefilename = prefix + "_{}_mar.tif".format(imageindex)

    # special case for image id15 frelon corrected form distorsion
    elif imagefilename.endswith((".tif", ".edf")):
        if CCDLabel in ("ImageStar_raw", ):
            prefixwihtindex = imagefilename[:-4]
            prefixwithindex_list = list(prefixwihtindex)
            indexnodigit = 0
            while len(prefixwithindex_list) > 0:
                lastelem = prefixwithindex_list.pop(-1)
                if verbose>0: print("lastelem", lastelem)
                if not lastelem.isdigit():
                    break
                indexnodigit += 1
            prefix = prefixwihtindex[:-(indexnodigit)]
            if verbose > 0: print("prefix", prefix)
            imagefilename = prefix + "{}.tif".format(imageindex)

        else:
            suffix = imagefilename[-4:]
            prefix = imagefilename[: -(4 + nbdigits)]

            imagefilename = prefix + "{:04d}{}".format(imageindex, suffix)

    elif imagefilename.endswith("mccd"):

        imagefilename = imagefilename[: -(5 + nbdigits)] + "{:04d}.mccd".format(imageindex)

    elif imagefilename.endswith("edf"):

        imagefilename = imagefilename[: -(4 + nbdigits)] + "{:04d}.edf".format(imageindex)

    elif imagefilename.endswith("unstacked"):

        imagefilename = imagefilename[: -(10 + nbdigits)] + "{:04d}.unstacked".format(imageindex)

    #     print "set filename to:", imagefilename
    return imagefilename


def getIndex_fromfilename(imagefilename, nbdigits=4, CCDLabel=None, stackimageindex=-1, verbose=0):
    r"""
    get integer index from imagefilename string if stackimageindex=-1 (default)

    if stackimageindex != -1, 

    :param imagefilename: filename string (full path or not)

    :return: file index
    """
    imageindex = None
    if CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        #default file extension for sCMOS camera
        ext = "tif"
        lenext = 4 #length of extension including '.'

        if imagefilename.endswith("tiff"):
            ext = "tiff"
            lenext = 5
        if verbose > 0: print('imagefilename', imagefilename)
        if nbdigits is not None:
            if imagefilename.endswith(ext):
                #imageindex = int(imagefilename[-(lenext + nbdigits): - (lenext)])
                prepart = str(imagefilename).rsplit('.',1)[0]
                digitpart = str(prepart).rsplit('_',1)[-1]
                # print("prepart", prepart)
                # print("digitpart",digitpart)
                imageindex = int(digitpart)
            elif imagefilename.endswith(ext+".gz"):
                imageindex = int(imagefilename[-(lenext+3 + nbdigits) : -(lenext+3)])
        else:
            if imagefilename.endswith(ext):
                prefix, _ = imagefilename.rsplit(".")
                imageindex = int(prefix.rsplit("_")[1])

    # for stacked images we return the position of image data in the stack as imagefileindex
    elif CCDLabel in ("EIGER_4Mstack","MaxiPIXCdTe","EIGER_4MCdTe", 'EIGER_1M'):
        _filename = os.path.split(imagefilename)[-1]
        if _filename.startswith(('mpxcdte_','eiger1_')):
            imageindex= int(_filename.split('.')[0].split('_')[-1])
        else:
            print('CCDLabel in getIndex_fromfilename',CCDLabel)
            imageindex = stackimageindex
            print('imageindex',imageindex)

    elif imagefilename.endswith("mar.tiff"):
        imageindex = int(imagefilename[-(9 + nbdigits) : -9])
    elif imagefilename.endswith("tiff"):
        imageindex = int(imagefilename[-(5 + nbdigits) : -5])
    elif imagefilename.endswith("mar.tif"):
        imageindex = int(imagefilename.split("_")[-2])
    elif imagefilename.endswith((".tif", "bmp")):
        # TODO: treat the case of HN56.tif  without underscore (see setfilename)
        imageindex = int(imagefilename.split("_")[-1][:-4])
    elif imagefilename.endswith("mccd"):
        imageindex = int(imagefilename[-(5 + nbdigits) : -5])
    elif imagefilename.endswith("edf"):
        imageindex = int(imagefilename[-(4 + nbdigits) : -4])
    elif imagefilename.endswith("unstacked"):
        imageindex = int(imagefilename[-(10 + nbdigits) : -10])

    if imageindex is None:
        raise ValueError(f'Detector label {CCDLabel} does not probably correspond to the selected file {imagefilename}')
    return imageindex


def getfilename(dirname, imfileprefix, imfilesuffix=None, numim=None,
                nbdigits_filename=4):
    """
    to get the global file name (name+path) for given components of the name
    put %4d instead of stringint
    """

    fileim = imfileprefix + stringint(numim, nbdigits_filename) + imfilesuffix
    filename = os.path.join(dirname, fileim)

    return filename

def getwildcardstring(CCDlabel):
    r"""  return smart wildcard to open binary CCD image file with priority of CCD type of CCDlabel

    :param CCDlabel: string label defining the CCD type
    :type CCDlabel: str
    :return: string from concatenated strings to be used in wxpython open file dialog box
    :rtype: str

    .. see also::
        - :func:`getIndex_fromfilename`

        - LaueToolsGUI.AskUserfilename

        - wx.FileDialog

    .. example::
        >>> from readmccd import getwildcardstring
        >>> getwildcardstring('MARCCD165')
        'MARCCD, ROPER(*.mccd)|*mccd|mar tif(*.tif)|*_mar.tiff|tiff(*.tiff)|*tiff|Princeton(*.spe)|*spe|Frelon(*.edf)|*edf|tif(*.tif)|*tif|All files(*)|*'
    """
    ALL_EXTENSIONS = ["mccd", "_mar.tiff", "tiff", "spe", "edf", "tif", "h5", ""]
    INFO_EXTENSIONS = ["MARCCD, ROPER(*.mccd)", "mar tif(*.tif)", "tiff(*.tiff)",
                    "Princeton(*.spe)", "Frelon(*.edf)", "tif(*.tif)", "hdf5(*.h5)",
                    "All files(*)"]

    extensions = copy.copy(ALL_EXTENSIONS)
    infos = copy.copy(INFO_EXTENSIONS)
    chosen_extension = DictLT.dict_CCD[CCDlabel][7]

    if chosen_extension in ALL_EXTENSIONS:
        index = ALL_EXTENSIONS.index(chosen_extension)
        ce = extensions.pop(index)
        extensions.insert(0, ce)

        inf = infos.pop(index)
        infos.insert(0, inf)

    wcd = ""
    for inf, ext in zip(infos, extensions):
        wcd += "{}|*{}|".format(inf, ext)

    wildcard_extensions = wcd[:-1]

    return wildcard_extensions

def pixelvalat(imagefilename, xy=None, sortpeaks=False, CCDLabel='sCMOS'):
    """ return values of pixel intensity xy pixel position
    
    :param sortpeaks: True or False, to sort peaks by increasing y value
    .. note: assume nb  of peaks > 1
    """
    if not isinstance(xy[0,0],np.int16):
        xy=np.int_(xy)
    if not sortpeaks: # already sorted
        xx,yy = np.array(xy).T
        nbpeaks = len(xx)
    else: # sort by y
        X,Y=np.array(xy).T
        nbpeaks=len(xy)
        
        sp = np.argsort(Y)
        xx=X[sp]
        yy=Y[sp]

    assert nbpeaks>1

    vals = None
    with open(imagefilename, 'rb') as f:
        filesize = os.path.getsize(imagefilename)
        framedim = DictLT.dict_CCD[CCDLabel][0]
        offset = filesize - np.prod(framedim) * 2
        vals = np.zeros(nbpeaks, dtype=np.int32)
        for k in range(nbpeaks):
            x=xx[k]
            y=yy[k]
            if CCDLabel == 'sCMOS':
                #x = 2018 - x  # fliplr
                f.seek(offset + 2 * (2016 * y + x))
                vals[k]=struct.unpack("H", f.read(2))[0]
            elif CCDLabel == 'IMSTAR_bin2':
                #x = 2018 - x  # fliplr
                f.seek(offset + 2 * (3035 * y + x))
                vals[k]=struct.unpack("H", f.read(2))[0]
            elif CCDLabel == 'MARCCD165':
                f.seek(offset + 2 * (2048 * y + x))
                vals[k]=struct.unpack("H", f.read(2))[0]
    return vals

def getroismax(imagefilename, roicenters=None, halfboxsize=(10,10), CCDLabel='sCMOS'):
    dataimage= None
    framedim=None

    selectedcounters = ["Imax",]
    ndivisions = (4,20)
    CountersData={}
    CountersData["Imax"]=np.zeros(len(roicenters))
    with fabio.open(imagefilename) as img:  # ok for sCMOS
        dataimage = img.data
        framedim = dataimage.shape
        for roi_index, roi in enumerate(roicenters):
            center_pixel = (round(roi[0]), round(roi[1]))

            indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                    (halfboxsize[0], halfboxsize[1]),
                                                    framedim,
                                                    flipxycenter=0)
            imin, imax, jmin, jmax = indicesborders

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                                    framedim=framedim)

            piece_dat = dataimage[imin:imax, jmin:jmax]
            piece_dat_m = None

            for counter in selectedcounters:

                if 'multiple' in counter:  #split array into several regular subarrays (tiled)
                    if piece_dat_m is None:
                        piece_dat_m, _, (box1, box2) = GT.splitarray(piece_dat, ndivisions)

                if counter.startswith("I"):
                    if 'multiple' not in counter:
                        if counter == "Imean":
                            CountersData["Imean"][roi_index] = np.mean(piece_dat)
                        elif counter == "Imax":
                            CountersData["Imax"][roi_index] = np.amax(piece_dat)
                        elif counter == "Iptp":
                            CountersData["Iptp"][roi_index] = np.ptp(piece_dat)
                    else:
                        if counter == "Imean_multiple":
                            CountersData["Imean_multiple"][roi_index] = np.mean(piece_dat_m, axis=(1, 2))
                        elif counter == "Imax_multiple":
                            CountersData["Imax_multiple"][roi_index] = np.amax(piece_dat_m, axis=(1, 2))
                        elif counter == "Iptp_multiple":
                            CountersData["Iptp_multiple"][roi_index] = np.ptp(piece_dat_m, axis=(1, 2))

                # TODO this branch is useless ?!
                elif counter.startswith("pos"):

                    if 'multiple' in counter:
                        if counter == "posmax_multiple":
                            nbrois = ndivisions[0] * ndivisions[1]
                            labels = np.repeat(np.arange(nbrois), box1 * box2).reshape((nbrois, box1, box2))
                            #print('labels.shape',labels.shape)
                            index = np.arange(nbrois)
                            datmaximumpos = np.array(scind.measurements.maximum_position(piece_dat_m, labels=labels, index=index))
                            CountersData["posmax_multiple"][roi_index] = datmaximumpos[:, 1:]

                    else:
                        datminimum = scind.measurements.maximum(piece_dat)
                        # center of mass without background removal
                        datcenterofmass = np.array(scind.measurements.center_of_mass(piece_dat))
                        # remove baseline level set to minimum pixel intensity
                        datcenterofmass2 = np.array(scind.measurements.center_of_mass(piece_dat - datminimum))

                        centerofmass = datcenterofmass2 + np.array([jmin, imin])

                        datmaximumpos = scind.measurements.maximum_position(piece_dat)
                        datmaximumpos = np.array(datmaximumpos, dtype="uint32")

                        posmax = datmaximumpos + np.array([jmin, imin])

                        # position monitors selection
                        XY = posmax
                        XY = centerofmass

                        xDATA, yDATA = XY

                        CountersData["posX"][roi_index] = xDATA
                        CountersData["posY"][roi_index] = yDATA
    
    return CountersData["Imax"]

def getroissum(imagefilename, roicenters=None, halfboxsize=(10,10), CCDLabel='sCMOS'):
    dataimage= None
    framedim=None

    CountersData={}
    CountersData["Isum"]=np.zeros(len(roicenters))
    with fabio.open(imagefilename) as img:  # ok for sCMOS
        dataimage = img.data
        framedim = dataimage.shape
        for roi_index, roi in enumerate(roicenters):
            center_pixel = (round(roi[0]), round(roi[1]))

            indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                    (halfboxsize[0], halfboxsize[1]),
                                                    framedim,
                                                    flipxycenter=0)
            imin, imax, jmin, jmax = indicesborders

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                                    framedim=framedim)

            piece_dat = dataimage[imin:imax, jmin:jmax]

            CountersData["Isum"][roi_index] = np.sum(piece_dat)
    
    return CountersData["Isum"]

def getroisXYmax(imagefilename, roicenters=None, halfboxsize=(10,10), CCDLabel='sCMOS'):
    dataimage= None
    framedim=None

    selectedcounters = ["posX","posY"]
    CountersData={}
    CountersData["posXY"]=np.zeros((len(roicenters),2))
    with fabio.open(imagefilename) as img:  # ok for sCMOS
        dataimage = img.data
        framedim = dataimage.shape
        i_idx = 0
        for roi_index, roi in enumerate(roicenters):
            center_pixel = (round(roi[0]), round(roi[1]))

            indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                    (halfboxsize[0], halfboxsize[1]),
                                                    framedim,
                                                    flipxycenter=0)
            imin, imax, jmin, jmax = indicesborders

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                                    framedim=framedim)

            piece_dat = dataimage[imin:imax, jmin:jmax]

            for counter in selectedcounters:

                if counter.startswith("I"):
                    if counter == "Imean":
                        CountersData["Imean"][roi_index] = np.mean(piece_dat)
                    elif counter == "Imax":
                        CountersData["Imax"][roi_index] = np.amax(piece_dat)
                    elif counter == "Iptp":
                        CountersData["Iptp"][roi_index] = np.ptp(piece_dat)
                    

                elif counter.startswith("pos"):
                    # method 2  ------ position of maximum --------
                    datmaximumpos = scind.measurements.maximum_position(piece_dat)
                    datmaximumpos = np.array(datmaximumpos, dtype="uint32")

                    posmax = datmaximumpos + np.array([jmin, imin])

                    # position monitors selection
                    XY = posmax

                    xDATA, yDATA = XY

                    CountersData["posXY"][i_idx] = [xDATA,yDATA]
            i_idx+=1
    return CountersData["posXY"]

def getroiscenterofmass(imagefilename, roicenters=None, halfboxsize=(10,10), CCDLabel='sCMOS'):
    dataimage= None
    framedim=None

    selectedcounters = ["posX","posY"]
    CountersData={}
    CountersData["posXY"]=np.zeros((len(roicenters),2))
    with fabio.open(imagefilename) as img:  # ok for sCMOS
        dataimage = img.data
        framedim = dataimage.shape
        i_idx = 0
        for roi_index, roi in enumerate(roicenters):
            center_pixel = (round(roi[0]), round(roi[1]))

            indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                    (halfboxsize[0], halfboxsize[1]),
                                                    framedim,
                                                    flipxycenter=0)
            imin, imax, jmin, jmax = indicesborders

            # avoid to wrong indices when slicing the data
            imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                                    framedim=framedim)

            piece_dat = dataimage[imin:imax, jmin:jmax]

            for counter in selectedcounters:

                if counter.startswith("I"):
                    if counter == "Imean":
                        CountersData["Imean"][roi_index] = np.mean(piece_dat)
                    elif counter == "Imax":
                        CountersData["Imax"][roi_index] = np.amax(piece_dat)
                    elif counter == "Iptp":
                        CountersData["Iptp"][roi_index] = np.ptp(piece_dat)
                    

                elif counter.startswith("pos"):
                    # method 1  ------  center of mass  without a flat baseline-----
                    datminimum = scind.measurements.minimum(piece_dat)
                    # center of mass without background removal
                    datcenterofmass = np.array(scind.measurements.center_of_mass(piece_dat))
                    # remove baseline level set to minimum pixel intensity
                    datcenterofmass2 = np.array(scind.measurements.center_of_mass(piece_dat - datminimum))

                    centerofmass = datcenterofmass2 + np.array([jmin, imin])

                    XY = centerofmass

                    xDATA, yDATA = XY

                    CountersData["posXY"][i_idx] = [xDATA,yDATA]
            i_idx+=1
    return CountersData["posXY"]

def getpixelValue(filename, x, y, ccdtypegeometry="edf"):
    r"""return pixel value at x,y

    .. warning::
        Very old function. To be checked. Use better readpixelvalue in plotdip.py

    :param filename: path to image file
    :type filename: str
    :param x: x pixel value
    :type x: int
    :param y: y pixel value
    :type y: int
    :param ccdtypegeometry: CCD label, defaults to "edf"
    :type ccdtypegeometry: str, optional
    :return: pixel intensity
    :rtype: int
    """
    if ccdtypegeometry == "edf":
        # frelon camera as mounted on BM32 Oct2012
        y = 2047 - y
    if ccdtypegeometry in ("mccd",):
        pass

    f = open(filename, "rb")
    f.seek(1024 + 2 * (2048 * y + x))
    val = struct.unpack("H", f.read(2))
    f.close()
    return val[0]


def readheader(filename, offset=4096, CCDLabel="MARCCD165"):
    r"""
    return header in a raw format

    default offset for marccd image
    """
    if CCDLabel.startswith("sCMOS"):
        filesize = os.path.getsize(filename)
        framedim = DictLT.dict_CCD[CCDLabel][0]
        offset = filesize - np.prod(framedim) * 2
    elif CCDLabel.startswith("MARCCD165"):
        offset = 4096
    with open(filename, "rb") as f:
        myheader = f.read(offset)
        #myheader.replace("\x00", " ")

    return myheader


def read_header_marccd(filename):
    r"""
    return string of parameters found in header in marccd image file .mccd

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    f = open(filename, "rb")
    f.seek(2048)
    posbyte = 0
    allsentences = ""
    for _ in list(range(32)):
        tt = f.read(32)
        s1 = tt.strip("\x00")
        if s1 != "":
            allsentences += s1 + "\n"
        # print posbyte, s1
        posbyte += 32
    tt = f.read(1024)
    s1 = tt.strip("\x00")
    if s1 != "":
        allsentences += s1 + "\n"

    f.close()
    return allsentences


def read_header_marccd2(filename):
    r"""
    return string of parameters comments and exposure time
    found in header in marccd image file .mccd

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    f = open(filename, "rb")
    f.seek(3072)
    tt = f.read(512)
    # from beamline designed input
    dataset_comments = str(tt).strip("\x00")

    f.seek(1024 + 2 * 256 + 128 + 12)
    s = struct.Struct("I I I")
    unpacked_data = s.unpack(f.read(3 * 4))
    #     print 'Unpacked Values:', unpacked_data
    #integration_time, expo_time, readout_time = unpacked_data
    _, expo_time, _ = unpacked_data

    f.close()
    return dataset_comments, expo_time


def read_header_scmos(filename, verbose=0):
    r"""
    return string of parameters comments and exposure time
    found in header in scmos image file .tif

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    if not PIL_EXISTS:
        return {}

    img = Image.open(filename)


    # img.tag.keys()
    # tag[270]   = (u'0 (thf=-50.4850 mon=24923 exposure=0.400)',)  or xech=-9.89771 yech=2.8205850000000012

    dictpar = {}
    strcom = img.tag[270][0]
    if verbose>0: print('read_header_scmos', strcom, type(strcom))
    try:
        si = strcom.index("(")
        fi = strcom.index(")")
        listpar = strcom[si + 1 : fi].split()
    except:
        listpar = strcom.split(' ')
    else:
        return {}
    #     print "listpar",listpar
    for elem in listpar:
        if "=" in elem:
            key, val = elem.split("=")
            dictpar[key] = float(val)

    return dictpar

def read_scmos_Artisttag(filename, allquads = False, verbose=0):
    r"""
    read list of sCMOS parameters written in Artist tag in tiff header file

    :param allquads: bool, if True, infos for 4 quads is returned, defaull a single quad info

    :return: QuadsParamsDict, dllversion, imagedate, fps

    .. note: imagedate   not sure... maybe dll date 
    """
    if not PIL_EXISTS:
        return {}

    img = Image.open(filename)

    Artisttag = img.tag[315][0]
    if verbose>0: print('Artisttag', Artisttag, type(Artisttag))

    quad,dll, G_a,G_b, G_c,G_d = Artisttag.split('GSense')
    qvf, imagedate, _ = (quad+dll).split('[')
    dllversion, fps = qvf.split("@")
    QuadsParamsDict = {}
    if allquads:
        scanquads = (G_a,G_b, G_c,G_d)
    else:
        scanquads = (G_a,)
    for gg in scanquads:
        quadindex= int(gg[0])
        
        quaddict = {}
        datquad = gg.split(':')
        datquad[0]='dummyvalue,'+datquad[0]
        try:
            del datquad[datquad.index('r')]
        

            for k in range(0,len(datquad)-1):
                _, key = datquad[k].rsplit(',',1)
                val, _ = datquad[k+1].rsplit(',',1)
                quaddict[key]= val
            
            QuadsParamsDict[quadindex]=quaddict

        except ValueError:
            return datquad, imagedate

    return QuadsParamsDict, dllversion, imagedate, fps

def readheadertiff(fullpathimage):
    ''' return artist tag of tiff image generated by pslviewer including exposure time, and unit'''
    if not SKIMAGE:
        print('SKIMAGE is missing ...')
        return None

    with tifffile.TiffFile(fullpathimage) as tif:
        imgs = [page.asarray() for page in tif.pages] # image(s)
        kkeys = [kk for page in tif.pages for kk in page.tags.keys() ]
        artist = [page.tags['artist'].value for page in tif.pages]
        listprops = artist[0].decode('UTF-8').split(',')
        gotExposureFlag=False
        for prop in listprops:
            if gotExposureFlag:
                expo_unit = prop[:-1].strip()
                print('expo_unit', expo_unit)
                break
            if prop.startswith('Exposure'):
                gotExposureFlag=True
                _, exp = prop.split(':')
                exposuretime = int(exp[1:])
                continue
        return artist, exposuretime, expo_unit


def read_motorsposition_fromheader(filename, CCDLabel="MARCCD165"):
    r""" return xyzpositions, expo_time from image file header
    available for "MARCCD165", "sCMOS", "sCMOS_fliplr"

    .. note: todo for hdf5 file
    """

    if CCDLabel not in ("MARCCD165","sCMOS", "sCMOS_fliplr"):
        if filename.endswith('.h5'):
            # TODO read some groups in hd5 file 
            pass
        return None, None
    
    if CCDLabel in ("MARCCD165",):
        dataset_comments, expo_time = read_header_marccd2(filename)
        xyz = dataset_comments.split(" ")[:3]
        xyzpositions = []
        for pos in xyz:
            xyzpositions.append(float(pos))
    elif CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        dictpar = {}
        import re

        img = Image.open(filename)
        ff = img.tag[315]
        aa = re.findall(r"([^:]+)=([^:]+)(?:,|$)", ff[0])
        listeq = aa[0][0].split(" ")
        for elem in listeq:
            if "Xech" in elem or "Yech" in elem or "Zech" in elem:
                dictpar[elem.lstrip("(")] = float(elem.rstrip(")"))

        headerdict = read_header_scmos(filename)
        if headerdict:
            expo_time = headerdict["exposure"]
        xyzpositions = []
        for motorname in ("Xech", "Yech", "Zech"):
            xyzpositions.append(dictpar[motorname])

    return xyzpositions, expo_time


def readoneimage_full(filename, frametype="mccd", dirname=None):
    r"""
    too SLOW!
    reads 1 entire image (marCCD format)
    :return: PILimage, image object of PIL module (16 bits integer) and
    arrayofdata: 2D array of intensity
    #TODO: manage framedim like readoneimage() just below"""

    if frametype == "mccd":
        shapeCCD = (2048, 2048)
    elif frametype == "martiff":
        # opposite of fit2d displays in loading data
        shapeCCD = (2591, 2751)
    elif frametype == "spe":
        # opposite of fit2d displays in loading data
        shapeCCD = (2048, 2048)

    if dirname is None:
        dirname = os.curdir

    pilimage = Image.open(os.path.join(dirname, filename))
    ravdata = np.array(pilimage.getdata())

    return pilimage, np.reshape(ravdata, shapeCCD)


def readCCDimage(filename, CCDLabel="MARCCD165", dirname=None, stackimageindex=-1, verbose=0):
    r"""general function to read raw data binary (Laue pattern) image file recorder on 2D detector.

    Read raw data binary image file and return pixel intensity 2D array such as
    to fit the data (2theta, chi) scattering angles representation convention.

    :param filename: path to image file (fullpath if ` dirname` =None)
    :type filename: str
    :param CCDLabel: label, defaults to "MARCCD165"
    :type CCDLabel: str, optional
    :param dirname: folder path, defaults to None
    :type dirname: str, optional
    :param stackimageindex: index of images bunch, defaults to -1
    :type stackimageindex: int, optional
    :param verbose: 0 or 1, defaults to 0
    :type verbose: int, optional
    :raises ValueError: if data format and CCD parameters from label are not compatible
    :return:
        - dataimage, 2D array image data pixel intensity properly oriented
        - framedim, iterable of 2 integers shape of dataimage
        - fliprot : string, key for CCD frame transform to orient image
    :rtype: tuple of 3 elements
    """
    (framedim, _, _, fliprot, offsetheader, formatdata, _, _) = DictLT.dict_CCD[CCDLabel]

    USE_RAW_METHOD = False

    if verbose > 1:
        print("CCDLabel in readCCDimage", CCDLabel)
        print('FABIO_EXISTS',FABIO_EXISTS)
        print('LIBTIFF_EXISTS',LIBTIFF_EXISTS)
        print('PIL_EXISTS',PIL_EXISTS)

    #    if extension != extension:
    #        print "warning : file extension does not match CCD type set in Set CCD File Parameters"
    if FABIO_EXISTS and CCDLabel not in ('Alban','psl_IN_bmp','MaxiPIXCdTe', 'EIGER_4MCdTestack'):#,'EIGER_4M'):
        if CCDLabel in ('MARCCD165', "EDF", "EIGER_4M", "EIGER_4MCdTe","EIGER_1M","IMSTAR_bin2","IMSTAR_bin1","RXO",
                        "sCMOS", "sCMOS_fliplr", "sCMOS_fliplr_16M", "sCMOS_16M","sCMOS_9M",
                        "Rayonix MX170-HS", 'psl_weiwei', 'ImageStar_dia_2021',
                        'ImageStar_dia_2021_2x2','psl_IN_tif', 'Alexiane'):#, 'Alban'):

            if verbose > 1:
                print('----> Using fabio ... to open %s\n'%filename)
            # warning import Image  # for well read of header only

            if dirname is not None:
                pathfile=os.path.join(dirname, filename)
            else:
                pathfile=filename

            with fabio.open(pathfile) as img:
                dataimage = img.data
            
                framedim = dataimage.shape

                # pythonic way to change immutable tuple...
                # initframedim = list(DictLT.dict_CCD[CCDLabel][0])
                # initframedim[0] = framedim[0]
                # initframedim[1] = framedim[1]
                # initframedim = tuple(initframedim)
                if CCDLabel in ("EIGER_4M",):
                    dataimage = np.ma.masked_where(dataimage<0, dataimage)
                    print('min fo dataimage: ',np.amin(dataimage))

                # elif CCDLabel in ('EIGER_4MCdTe',):
                #     print('raw framedim EIGER_4MCdTe', framedim)
        else:
            USE_RAW_METHOD = True
    # special treatment: because fabio open only the fisrt image of stacked images hdf5 file!....
    elif CCDLabel in ("EIGER_4Mstack"):  #made by PSL software

        import tables as Tab

        if dirname is not None:
            pathtofile = os.path.join(dirname, filename)
        else:
            pathtofile = filename

        # TODO check to correct version transition
        if Tab.__version__ >= "3.4.2":
            hdf5file = Tab.open_file(pathtofile)
        else:
            hdf5file = Tab.openFile(pathtofile)

        if verbose > 0: print("opening hdf5 stacked data table")
        alldata = hdf5file.root.entry.data.data
        if verbose > 0: print("alldata.shape", alldata.shape)

        dataimage = alldata[stackimageindex]
        framedim = dataimage.shape

    elif CCDLabel in ("EIGER_4MCdTestack"):  #made by ESRF BLISS software
        if dirname is not None:
            pathfile=os.path.join(dirname, filename)
        else:
            pathfile=filename

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with h5py.File(pathfile, 'r') as f:
            try:
                if stackimageindex < 0:
                    stackimageindex = 0
                dataimage = f['entry_0000']['measurement']['data'][()][stackimageindex]
            except IndexError:
                stackedimagesshape = f['entry_0000']['measurement']['data'][()].shape
                nbmaximages = stackedimagesshape[0]
                print('\n---------------\nWARNING !! For this file : %s'%filename)
                print(f'Requested stackindex {stackimageindex} exceeds the number of stacked images: {nbmaximages}')
                print("---------------\n")
                dataimage=np.zeros((stackedimagesshape[1], stackedimagesshape[2]))

            framedim = dataimage.shape
            print('framedim EIGER_4MCdTestack',framedim) 
            fliprot = 'no'

    elif filename.endswith('h5'):  #  maxipix  hdf5 file
        print('MAXIPIXCDTE ***********!\n\n')  # no more eiger4MCdTe

        import LaueTools.logfile_reader as iohdf5      

        if CCDLabel in ('MaxiPIXCdTe',):
            """there are two types of h5 file:
                case 1, raw data only 
                case 2,  master h5 file with meta data (namely folder) pointing (link) to h5 file of case 1
                case 3, stacked images raw data"""
            
            if dirname is not None:
                pathfile=os.path.join(dirname, filename)
            else:
                pathfile=filename

            # case 1 or 3
            if os.path.split(pathfile)[-1].startswith(('mpxcdte_','eiger1_')):  
                h5filetype = 'h5puredata'
            # case 2
            else:
                h5filetype = 'h5master'  
            
            # case 2
            if h5filetype == 'h5master':
                """framedim  =  nbframes,   dim1 , dim2  with nbframes >=1"""
                print('----> Reading hdf5 file [pointing to hdf5 file] %s\n'%filename)
                dirname = '/home/micha/LaueProjects/MaxiPIX_Laue'
                filename = 'align_0001.h5'
                key = 'align_0001_84'

                if verbose > 0: print("opening hdf5 stacked data table")
                alldata = iohdf5.Scan_hdf5(pathfile, key, verbose=0).mpxcdte
                if verbose > 0: print("alldata.shape", alldata.shape)

                dataimage = alldata[stackimageindex]
                framedim = alldata.shape  # nb images, dim1, dim2
                fliprot = 'fliplr_h5master'
            # case 1 or 3
            elif h5filetype == 'h5puredata':
                """framedim  = dim1 , dim2 """
                os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
                with h5py.File(pathfile, 'r') as f:
                    dataimage = f['entry_0000']['measurement']['data'][()][0]
                    framedim = dataimage.shape
                    print('framedim',framedim) 
                    fliprot = 'fliplr_h5pure'


    elif LIBTIFF_EXISTS or CCDLabel in ("Alban",):
        
        if CCDLabel in ("sCMOS", "MARCCD165", "TIFF Format", "EIGER_4M","FRELONID15_corrected", "VHR_PSI",
                            "VHR_DLS", "MARCCD225", "Andrea", "pnCCD_Tuba", "sCMOS_16M", "Alban", ):
            if 1: #verbose > 1:
                print("----> Using libtiff...")
                print(f'for {CCDLabel} camera \n')
            if dirname is not None:
                tifimage = TIFF.open(os.path.join(dirname, filename), mode="r")
            else:
                tifimage = TIFF.open(filename, mode="r")

            dataimage = tifimage.read_image()
            framedim = (int(tifimage.GetField("ImageLength")),
                        int(tifimage.GetField("ImageWidth")))
            
            if tifimage.IsByteSwapped():
                dataimage = dataimage.byteswap()
                
            if CCDLabel in ("Alban",):  # float from summation in imageJ
                print("amin()", np.amin(dataimage))
                print('pixel val',dataimage[455,590])
                ampli = np.amax(dataimage)-np.amin(dataimage)
                dataimage = (dataimage-np.amin(dataimage))/ampli*60000
                print("amin(), amax()", np.amin(dataimage), np.amax(dataimage))
                a2=np.round(dataimage, decimals=0)
                dataimage = np.array(a2, dtype=np.int16)
                print('pixel val',dataimage[455,590])
                print("amin(), amax()", np.amin(dataimage), np.amax(dataimage))

                dataimage=np.where(dataimage<0,2*65535-dataimage, dataimage)
                print('pixel val',dataimage[455,590])
                print("amin(), amax()", np.amin(dataimage), np.amax(dataimage))

            if CCDLabel in ("EIGER_4M",):
                #dataimage = np.ma.masked_values(dataimage, -1)
                np.clip(dataimage, a_min=0, a_max=200000, out=dataimage)
                dataimage = np.ma.masked_where(dataimage<0, dataimage)
                print('min of dataimage: ',np.amin(dataimage))
                print('image type: ',dataimage.dtype)
        else:
            USE_RAW_METHOD = True

    elif PIL_EXISTS and CCDLabel not in ('EIGER_4M',):
        if verbose > 1:
            print("using PIL's module Image")
        if CCDLabel in ("sCMOS", "MARCCD165", "sCMOS_16M"):
            if verbose > 1: print('PIL is too slow. Better install libtiff or fabio. Meanwhile I will use PIL...')
            USE_RAW_METHOD = True
        elif CCDLabel in ("VHR_PSI", "VHR_DLS", "MARCCD225", "Andrea", "pnCCD_Tuba", ):
            # data are compressed!

            if dirname is not None:
                fullpath = os.path.join(dirname, filename)
            else:
                fullpath = filename

            im = Image.open(fullpath, "r")
            dataimage = np.array(im.getdata()).reshape(framedim)

        elif CCDLabel in ("psl_IN_bmp",):
            # data are compressed!

            if dirname is not None:
                fullpath = os.path.join(dirname, filename)
            else:
                fullpath = filename

            im = Image.open(fullpath, "r")
            dataimageraw = np.asarray(im)
            dataimage = np.amax(dataimageraw)-dataimageraw

    # RAW method knowing or deducing offsetheader and dataformat
    if USE_RAW_METHOD:
        print("----> !!! not using libtiff, nor fabio, nor PIL!!!  ")
        if CCDLabel in ("MARCCD165",):
            print("for MARCCD not using libtiff, raw method ...")
            # offsetheader may change ...
            #filesize=getfilesize(dirname, filename)
            offsetheader = 4096 #filesize - 2048*2048 * 2
        # offset header varying
        elif CCDLabel.startswith("ImageStar_raw"):
            filesize=getfilesize(dirname, filename)
            bytes_per_pixels = 2
            if CCDLabel.endswith("32bits"):
                bytes_per_pixels = 4

            nbpixels = 1500
            if CCDLabel in ("ImageStar_1528x1528",):
                nbpixels = 1528
            offsetheader = filesize - nbpixels * nbpixels * bytes_per_pixels

        elif CCDLabel == "ImageStar_dia_2021":
            filesize=getfilesize(dirname, filename)
            bytes_per_pixels = 2
            nbpixels = 3056

            offsetheader = filesize - nbpixels * nbpixels * bytes_per_pixels
        elif CCDLabel in ("sCMOS",):
            if verbose > 0: print("for sCMOS not using libtiff, raw method ...")
            # offsetheader may change ...
            filesize = getfilesize(dirname, filename)
            offsetheader = filesize - 2016*2018 * 2

        dataimage = readoneimage(filename,
                                framedim=framedim,
                                dirname=dirname,
                                offset=offsetheader,
                                formatdata=formatdata)

        #print("dataimage", dataimage)

        if CCDLabel in ("FRELONID15_corrected",):

            dataimage = dataimage.byteswap()

        if CCDLabel in ("EIGER_4Munstacked",):
            dataimage = np.ma.masked_where(dataimage > 4000000, dataimage)
            if verbose > 0:
                print("framedim", framedim)
                print("offsetheader", offsetheader)
                print("formatdata", formatdata)
                print("dataimage", dataimage)

    if verbose:
        print("CCDLabel: ", CCDLabel)
        print("nb of pixels", np.shape(dataimage))

    # need to reshape data from 1D to 2D
    try:
        if len(dataimage.shape) == 1:
            if verbose:
                print("nb elements", len(dataimage))
                print("framedim", framedim)
                print("framedim nb of elements", framedim[0] * framedim[1])
            dataimage = np.reshape(dataimage, framedim)
    except ValueError:
        raise ValueError(
            "Selected CCD type :{} may be wrong (or nb of pixels, dimensions...)".format(
                CCDLabel))

    # some array transformations if needed depending on the CCD mounting
    # print('fliprot', fliprot)

    if fliprot == "spe":
        dataimage = np.rot90(dataimage, k=1)

    elif fliprot == "EIGER SLS":
        dataimage = np.rot90(dataimage, k=-1)

    elif fliprot == "Alexiane":
        dataimage = np.rot90(dataimage, k=1)

    elif fliprot in ("VHR_Feb13", "sCMOS_fliplr", "fliplr") or fliprot.startswith('fliplr'):  # for h5 from maxipixof stacked link to images 'h5master'
        #print('fliprot  2 ', fliprot)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhr":  # july 2012 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhrdiamond":  # july 2012 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot in ("Imagestar_dia_2021","ImageStar_dia_2021"):  # starting Feb july 2021 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=1)
        dataimage = np.fliplr(dataimage)

    elif fliprot in  ("frelon2", "flipud"):
        dataimage = np.flipud(dataimage)

    return dataimage, framedim, fliprot

def getfilesize(dirname,filename):
    """get size in byte of a file

    :param dirname: folder
    :type dirname: str, pathlike
    :param filename: filename
    :type filename: str, pathlike
    :return: file size
    :rtype: int
    """
    if dirname is not None:
        filesize = os.path.getsize(os.path.join(dirname, filename))
    else:
        filesize = os.path.getsize(filename)
    return filesize

def readoneimage(filename, framedim=(2048, 2048), dirname=None, offset=4096, formatdata="uint16"):
    r""" crude way to open binary image. it returns a 1d array of integers from a binary image file (full data)

    :param filename: image file name (full path if dirname=0)
    :type filename: str
    :param framedim: detector dimensions, defaults to (2048, 2048)
    :type framedim: tuple of 2 integers, optional
    :param dirname: folder path, defaults to None
    :type dirname: str, optional
    :param offset: file header in byte (octet), defaults to 4096
    :type offset: int, optional
    :param formatdata: numpy format of raw binary image pixel value, defaults to "uint16"
    :type formatdata: str, optional
    :return: dataimage : image data pixel intensity
    :rtype: 1D array
    """

    nb_elem = framedim[0] * framedim[1]

    if dirname is None:
        dirname = os.curdir

    f = open(os.path.join(dirname, filename), "rb")
    f.seek(offset)

    # d=scipy.io.fread(f,2048*2048,np.oldnumeric.Int16)
    # d = scipy.io.fread(f,nb_elem,np.oldnumeric.UInt16)
    d = np.fromfile(f, dtype=formatdata, count=nb_elem)
    f.close()
    return d


def readoneimage_band(filename,
                        framedim=(2048, 2048),
                        dirname=None,
                        offset=4096,
                        line_startindex=0,
                        line_finalindex=2047,
                        formatdata="uint16"):
    r"""
    returns a 1d array of integers from a binary image file. Data located in band according shape of data (framedim)

    :param filename: string
               path to image file (fullpath if `dirname`=None)
    :param offset: integer
             nb of file header bytes
    :param framedim: iterable of 2 integers
               shape of expected 2D data
    :param formatdata: string
                 key for numpy dtype to decode binary file

    :return: dataimage, 1D array, image data pixel intensity
    """
    if dirname is None:
        dirname = os.curdir

    if formatdata in ("uint16",):
        nbBytesPerElement = 2
    if formatdata in ("uint32",):
        nbBytesPerElement = 4

    nbElems = (line_finalindex - line_startindex + 1) * framedim[1]

    f = open(os.path.join(dirname, filename), "rb")
    f.seek(offset + line_startindex * framedim[1] * nbBytesPerElement)

    # d=scipy.io.fread(f,2048*2048,np.oldnumeric.Int16)
    # d = scipy.io.fread(f,nb_elem,np.oldnumeric.UInt16)
    #     print "line_finalindex-line_startindex",line_finalindex-line_startindex
    #     print "line_startindex*framedim[1]*nbBytesPerElement",line_startindex*framedim[1]*nbBytesPerElement
    #     print "line_startindex*framedim[1]",line_startindex*framedim[1]
    #     print "nbElems",nbElems

    band = np.fromfile(f, dtype=formatdata, count=nbElems)
    f.close()
    return band


def readoneimage_crop_fast(filename, dirname=None, CCDLabel="MARCCD165",
                                                    firstElemIndex=0, lastElemIndex=2047, verbose=0):
    r""" Returns a 2d array of integers from a binary image file. Data are taken only from a rectangle

    with respect to firstElemIndex and lastElemIndex.

    :param filename: string, path to image file (fullpath if ` dirname`=None)
    :param offset: integer, nb of file header bytes
    :param framedim: iterable of 2 integers, shape of expected 2D data
    :param formatdata: string, key for numpy dtype to decode binary file

    :return: dataimage : 1D array image data pixel intensity
    """
    (framedim, _, _, fliprot, offsetheader, formatdata, _, _) = DictLT.dict_CCD[CCDLabel]
    if verbose > 0:
        print("framedim read from DictLT.dict_CCD in readoneimage_crop_fast()", framedim)
        print("formatdata", formatdata)
        print("offsetheader", offsetheader)

    if dirname is None:
        dirname = os.curdir

    # if formatdata in ("uint16",):
    #     nbBytesPerElement = 2
    # if formatdata in ("uint32",):
    #     nbBytesPerElement = 4

    dataimage2D = np.zeros(framedim)

    # colFirstElemIndex = firstElemIndex % framedim[1]
    lineFirstElemIndex = firstElemIndex // framedim[1]
    # colLastElemIndex = lastElemIndex % framedim[1]
    lineLastElemIndex = lastElemIndex // framedim[1]

    band1D = readoneimage_band(filename, framedim=framedim, dirname=dirname, offset=offsetheader,
                                                            line_startindex=lineFirstElemIndex,
                                                            line_finalindex=lineLastElemIndex,
                                                            formatdata=formatdata)

    #     print "band1D.shape",band1D.shape
    #     print "(lineLastElemIndex-lineFirstElemIndex,framedim[1])",(lineLastElemIndex-lineFirstElemIndex,framedim[1])
    #     print "((lineLastElemIndex-lineFirstElemIndex)*framedim[1])",((lineLastElemIndex-lineFirstElemIndex)*framedim[1])

    band2D = band1D.reshape((lineLastElemIndex - lineFirstElemIndex + 1, framedim[1]))

    dataimage2D[lineFirstElemIndex : lineLastElemIndex + 1, :] = band2D

    return dataimage2D, framedim, fliprot


def readrectangle_in_image(filename, pixx, pixy, halfboxx, halfboxy, dirname=None,
                                                                            CCDLabel="MARCCD165",
                                                                            verbose=0,
                                                                            stackimageindex=-1):
    r"""
    returns a 2d array of integers from a binary image file. Data are taken only from a rectangle
    centered on pixx, pixy

    :return: dataimage : 2D array, image data pixel intensity
    """
    # if CCDLabel == "EIGER_4MCdTestack":
    #     raise NotImplementedError(f'Mosaic and roi counters is not yet implemented for {CCDLabel}. In progress')

    (framedim, _, _, fliprot, offsetheader, formatdata, _, _) = DictLT.dict_CCD[CCDLabel]

    if verbose > 0:
        print("framedim read from DictLT.dict_CCD in readrectangle_in_image()", framedim)
        print("formatdata", formatdata)
        print("offsetheader", offsetheader)
    # recompute headersize
    if dirname is not None:
        fullpathfilename = os.path.join(dirname, filename)
    else:
        dirname = os.curdir
        fullpathfilename = filename

    if formatdata in ("uint16",):
        nbBytesPerElement = 2
    if formatdata in ("uint32",):
        nbBytesPerElement = 4

    if verbose > 0:
        print("fullpathfilename", fullpathfilename)
    try:
        filesize = os.path.getsize(fullpathfilename)
    except OSError:
        print("missing file {}\n".format(fullpathfilename))
        return None

    # uint16
    offsetheader = filesize - (framedim[0] * framedim[1]) * nbBytesPerElement

    if CCDLabel in ('EIGER_4MCdTe'):  # not useful ?
        oneimagesize = (framedim[0] * framedim[1]) * nbBytesPerElement
        offsetheader = filesize % oneimagesize

    if verbose > 0:
        print("calculated offset of header from file size...", offsetheader)

    x = int(pixx)
    y = int(pixy)

    if fliprot in ("sCMOS_fliplr",):
        x = framedim[1] - x

    boxx = int(halfboxx)
    boxy = int(halfboxy)

    xpixmin = x - boxx
    xpixmax = x + boxx

    ypixmin = y - boxy
    ypixmax = y + boxy

    lineFirstElemIndex = ypixmin
    lineLastElemIndex = ypixmax

    if verbose > 0:
        print("lineFirstElemIndex", lineFirstElemIndex)
        print("lineLastElemIndex", lineLastElemIndex)

    if CCDLabel in ('EIGER_4MCdTe','EIGER_1M'):  # not tested with EIGER_1M
        _data, _dims, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname, stackimageindex=-1, verbose=0)
        band2D = _data[ypixmin:ypixmax+1]

        #print('shape of band2D', band2D.shape)

    elif CCDLabel in ('EIGER_4MCdTestack'):
        print('stackimageindex', stackimageindex)
        print('filename',filename)
        _data, _dims, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname, stackimageindex=stackimageindex, verbose=0)
        band2D = _data[ypixmin:ypixmax+1]

    elif CCDLabel not in ('EIGER_4MCdTe', 'EIGER_4MCdTestack','EIGER_1M'):
        band = readoneimage_band(fullpathfilename,
                                framedim=framedim,
                                dirname=None,
                                offset=offsetheader,
                                line_startindex=lineFirstElemIndex,
                                line_finalindex=lineLastElemIndex,
                                formatdata=formatdata)
        
        nblines = lineLastElemIndex - lineFirstElemIndex + 1

        band2D = np.reshape(band, (nblines, framedim[1]))

    rectangle2D = band2D[:, xpixmin : xpixmax + 1]

    if verbose > 0:
        print("band2D.shape", band2D.shape)
        print("rectangle2D.shape", rectangle2D.shape)

    return rectangle2D


def readoneimage_crop(filename, center, halfboxsize, CCDLabel="PRINCETON", dirname=None):
    r"""
    return a cropped array of data read in an image file

    :param filename: string, path to image file (fullpath if ` dirname`=None)
    :param center: iterable of 2 integers, (x,y) pixel coordinates
    :param halfboxsize: integer or iterable of 2 integers, ROI half size in both directions

    :return: dataimage : 1D array, image data pixel intensity

    .. todo:: useless?
    """
    if dirname is None:
        dirname = os.curdir
    else:
        filename = os.path.join(dirname, filename)

    if isinstance(halfboxsize, int):
        boxsizex, boxsizey = halfboxsize, halfboxsize
    elif len(halfboxsize) == 2:
        boxsizex, boxsizey = halfboxsize

    xpic, ypic = center

    dataimage, framedim, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=None)

    x1 = np.maximum(0, xpic - boxsizex)
    x2 = np.minimum(framedim[1], xpic + boxsizex + 1)  # framedim[0]
    y1 = np.maximum(0, ypic - boxsizey)
    y2 = np.minimum(framedim[0], ypic + boxsizey + 1)  # framedim[1]

    return None, dataimage[y1:y2, x1:x2]


def readoneimage_manycrops(filename, centers, boxsize, stackimageindex=-1, CCDLabel="MARCCD165",
                                                                        addImax=False,
                                                                        use_data_corrected=None,
                                                                        verbose=0):
    r"""
    reads 1 image and extract many regions
    centered on center_pixel with xyboxsize dimensions in pixel unit

    :param filename: string,fullpath to image file
    :param centers: list or array of [int,int] centers (x,y) pixel coordinates
    :param use_data_corrected: enter data instead of reading data from file
                         must be a tuple of 3 elements:
                         fulldata, framedim, fliprot
                         where fulldata is a numpy.ndarray
                         as output by :func:`readCCDimage`
    :param boxsize: iterable 2 elements or integer
              boxsizes [in x, in y] direction or integer to set a square ROI

    :return: Data, list of 2D array pixel intensity if addImax is False
            or (Data,Imax) if addImax is True
    """
    # use alternate data  (for instance for data from filename without background)
    if use_data_corrected is not None:
        if isinstance(use_data_corrected, tuple):
            if len(use_data_corrected) == 3:
                fulldata, framedim, _ = use_data_corrected
    # use data by reading file
    else:
        fulldata, framedim, _ = readCCDimage(filename, stackimageindex=stackimageindex,
                                                            CCDLabel=CCDLabel,
                                                            dirname=None)

    if isinstance(boxsize, int):
        boxsizex, boxsizey = boxsize, boxsize
    elif len(boxsize) == 2:
        boxsizex, boxsizey = boxsize

    # xpic, ypic = np.array(centers).T

    #    x1 = np.array(np.maximum(0, xpic - boxsizex), dtype=np.int16)
    #    x2 = np.array(np.minimum(framedim[0], xpic + boxsizex), dtype=np.int16)
    #    y1 = np.array(np.maximum(0, ypic - boxsizey), dtype=np.int16)
    #    y2 = np.array(np.minimum(framedim[1], ypic + boxsizey), dtype=np.int16)

    Data = []

    Imax = []
        
    if verbose > 0:
        print("framedim in readoneimage_manycrops", framedim)
    # if CCDLabel in ('MaxiPIXCdTe',):  #stacking of images in scan  or h5 pure data
    #     framedim = framedim[-1], framedim[-2]
    # else:
    #     framedim = framedim[1], framedim[0]

    framedim = framedim[-1], framedim[-2]

    for center in centers:
        i1, i2, j1, j2 = ImProc.getindices2cropArray(center, (boxsizex, boxsizey), framedim)
        #        print "i1, i2, j1, j2-----", i1, i2, j1, j2
        cropdata = fulldata[i1:i2, j1:j2]
        #         # print "cropdata.shape", cropdata.shape
        #         print "i2-i1",i2-i1
        #         print "boxsizey*2+1",boxsizey*2+1
        #         print "j2-j1",j2-j1
        #         print "boxsizex*2+1",boxsizex*2+1

        # for spot near border, replace by zeros array
        if i2 - i1 != boxsizey * 2 or j2 - j1 != boxsizex * 2:
            cropdata = np.zeros((boxsizey * 2 + 1, boxsizex * 2 + 1))

        Data.append(cropdata)
        if addImax:
            Imax.append(np.amax(cropdata))

    #        print "max in cropped data", np.amax(cropdata)
    if addImax:
        return Data, Imax
    else:
        return Data

def writeimage(outputname, _header, data, dataformat=np.uint16, verbose=0):
    r"""
    from data 1d array of integers
    with header coming from a f.open('imagefile'); f.read(headersize);f.close()
    .. warning:: header contain dimensions for subsequent data. Check before the compatibility of
    data with header infos(nb of byte per pixel and array dimensions
    """
    newfile = open(outputname, "wb")
    newfile.write(_header)
    data = np.array(data, dtype=dataformat)
    data.tofile(newfile)
    newfile.close()
    if verbose > 0: print("image written in ", outputname)


def write_rawbinary(outputname, data, dataformat=np.uint16, verbose=0):
    r"""
    write a binary file without header of a 2D array

    used ?
    """
    newfile = open(outputname, "wb")
    data = np.array(data, dtype=dataformat)
    data.tofile(newfile)

    newfile.close()
    if verbose > 0: print("image written in ", outputname)


def SumImages(prefixname, suffixname, ind_start, ind_end, dirname=None,
                                                            plot=0,
                                                            output_filename=None,
                                                            CCDLabel=None,
                                                            nbdigits=0):
    r"""
    sum images and write image with 32 bits per pixel format (4 bytes)

    used?
    """
    #     prefixname = 'HN08_'
    #     suffixname = '.tif'
    #     CCDLabel = 'ImageStar_raw'
    #     dirname = '/home/micha/LaueProjects/Vita'

    output_filename = "mean_{}_{:04d}_{}{}".format(prefixname, ind_start, ind_end, suffixname)

    filename = "{}{:04d}{}".format(prefixname, ind_start, suffixname)

    data, shape, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname)

    if CCDLabel == "ImageStar_raw":
        # Add addition of 32 bits image => replace 2 by 4  nb of bytes per pixel
        filesize = os.path.getsize(os.path.join(dirname, filename))
        offsetheader = filesize - 1500 * 1500 * 2
    if CCDLabel == "EIGER_1M":
        filesize = os.path.getsize(os.path.join(dirname, filename))
        offsetheader = filesize - 1065 * 1030 * 4
    else:
        # Add addition of 32 bits image => replace 2 by 4  nb of bytes per pixel
        filesize = os.path.getsize(os.path.join(dirname, filename))
        offsetheader = filesize - 2048 * 2048 * 2
    # print "shape = ", np.shape(datastart)
    datasum = np.zeros(shape, dtype=np.uint32)

    #     nb_images = ind_end - ind_start + 1

    indexscanlist = list(range(ind_start, ind_end + 1, 1))
    for k in indexscanlist:
        # print k
        filename = "{}{:04d}{}".format(prefixname, k, suffixname)
        # print filename1
        data, shape, _ = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname)
        # print max(data1), np.argmax(data1)
        datasum = data + datasum

    if output_filename:
        outputfilename = output_filename
        header = readheader(os.path.join(dirname, filename), offset=offsetheader)
        writeimage(os.path.join(dirname, outputfilename), header, datasum, dataformat=np.uint32)

        print("Added images with prefix {} from {} to {} written in {}".format(
                prefixname, ind_start, ind_end, outputfilename))
    if plot:
        print("later")

    return datasum


def Add_Images2(prefixname,
                ind_start,
                ind_end,
                plot=False,
                writefilename=None,
                CCDLabel="MARCCD165",
                average=True):
    """
    Add several Images

    Notes
    -----
    in dev 

    Parameters
    ----------
    prefixname
        str, prefix filename
    ind_start
        int, first file index
    ind_end
        int, laqt file index
    plot, optional
        bool, plot image (not implemented), by default False
    writefilename, optional
        str, output filename, by default None
    CCDLabel, optional
        _description_, by default "MARCCD165"
    average, optional
        _description_, by default True

    Returns
    -------
        numpy array, sum of image data arrays
    """
    suffixname = "." + DictLT.dict_CCD[CCDLabel][-1]

    filename = prefixname + stringint(ind_start, 4) + suffixname
    datastart, _, _ = readCCDimage(filename, CCDLabel)

    # print "shape = ", np.shape(datastart)
    datastart = np.array(datastart, dtype=float)

    indexscanlist = list(range(ind_start + 1, ind_end + 1, 1))
    for k in indexscanlist:
        # print k
        filename1 = prefixname + stringint(k, 4) + suffixname
        # print filename1
        data1, _, _ = readCCDimage(filename1, CCDLabel)
        # print max(data1), np.argmax(data1)
        datastart = np.array(data1, dtype=float) + datastart
    # datastart= datastart/((len(indexscanlist)+1)*1.)
    # print "final"
    # print max(datastart), np.argmax(datastart)
    if average:
        datastart = datastart / (float(len(indexscanlist)) + 1.0)
    # print max(datastart), np.argmax(datastart)
    datastart = np.array(datastart, dtype=np.uint16)
    # print max(datastart), np.argmax(datastart)
    if writefilename:
        outputfilename = writefilename
        header = readheader(filename1, CCDLabel=CCDLabel)
        writeimage(outputfilename, header, datastart)
        print("written in ", outputfilename)
    if plot:
        print("later")

    return datastart


def Add_Images(prefixname, ind_start, ind_end, plot=0, writefilename=None):
    r"""
    Add continuous sequence of images

    .. note::
        Add_Images2 exists

    :param prefixname: string, prefix common part of name of files

    :param ind_start: int, starting image index

    :param ind_end: int, final image index

    :param writefilename: string, new image filename where to write datastart (with last image file header read)

    :return:  datastart, array accumulation of 2D data from each image
    """
    suffixname = ".mccd"

    datastart = readoneimage(prefixname + stringint(ind_start, 4) + suffixname)
    # print "shape = ", np.shape(datastart)
    datastart = np.array(datastart, dtype=float)

    indexscanlist = list(range(ind_start + 1, ind_end + 1, 1))
    for k in indexscanlist:
        # print k
        filename1 = prefixname + stringint(k, 4) + suffixname
        # print filename1
        data1 = readoneimage(filename1)
        # print max(data1), np.argmax(data1)
        datastart = np.array(data1, dtype=float) + datastart
    # datastart= datastart/((len(indexscanlist)+1)*1.)
    # print "final"
    # print max(datastart), np.argmax(datastart)
    datastart = datastart / (float(len(indexscanlist)) + 1.0)
    # print max(datastart), np.argmax(datastart)
    datastart = np.array(datastart, dtype=np.uint16)
    # print max(datastart), np.argmax(datastart)
    if writefilename:
        outputfilename = writefilename
        header = readheader(filename1)
        writeimage(outputfilename, header, datastart)
        print("written in ", outputfilename)
    if plot:
        print("later")

    return datastart

def get_imagesize(framedim, nbbits_per_pixel, headersize_bytes):
    r"""
    return size of image in byte (= 1 octet = 8 bits)
    """
    return (framedim[0] * framedim[1] * nbbits_per_pixel + headersize_bytes * 8) // 8


if __name__=='__main__':
    pass
