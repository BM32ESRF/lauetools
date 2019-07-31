# -*- coding: utf-8 -*-
"""
readmccd module is made for reading data contained in binary image file
 fully or partially.
 It can process a peak or blob search by various methods
 and refine the peak by a gaussian or lorentzian 2D model

 More tools can be found in LaueTools package at sourceforge.net and gitlab.esrf.fr
"""

__version__ = "$Revision: 975 $"
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

# built-in modules
import sys
import os
import copy
import time as ttt
import struct
import math

# third party modules

import scipy.interpolate as sci
import scipy.ndimage as ndimage
import scipy.signal
import scipy.spatial.distance as ssd

try:
    import fabio

    FABIO_EXISTS = True
except ImportError:
    print(
        "Missing fabio module. Please install it if you need open some tiff images from the sCMOS camera"
    )
    FABIO_EXISTS = False

try:
    from libtiff import TIFF, libtiff_ctypes

    libtiff_ctypes.suppress_warnings()
    LIBTIFF_EXISTS = True
except ImportError:
    print(
        "Missing library libtiff, Please install: pylibtiff if you need open some tiff images"
    )
    LIBTIFF_EXISTS = False

try:
    from PIL import Image

    PIL_EXISTS = True
except ImportError:
    print(
        "Missing python module called PIL. Please install it if you need open some tiff images from vhr camera"
    )
    PIL_EXISTS = False
import numpy as np
import pylab as pp

# lauetools modules
if sys.version_info.major == 3:
    from . import fit2Dintensity as fit2d
    from . import fit2Dintensity_Lorentz as fit2d_l
    from . import generaltools as GT
    from . import IOLaueTools as IOLT
    from . import dict_LaueTools as DictLT
else:
    import fit2Dintensity as fit2d
    import fit2Dintensity_Lorentz as fit2d_l
    import generaltools as GT
    import IOLaueTools as IOLT
    import dict_LaueTools as DictLT

listfile = os.listdir(os.curdir)

# Default dictionary peak search parameters:

PEAKSEARCHDICT_Convolve = {
    "PixelNearRadius": 10,
    "thresholdConvolve": 500,
    "IntensityThreshold": 10,
    "removeedge": 2,
    "local_maxima_search_method": 2,
    "boxsize": 15,
    "position_definition": 1,
    "verbose": 0,
    "fit_peaks_gaussian": 1,
    "xtol": 0.001,
    "FitPixelDev": 2.0,
    "return_histo": 0,
    "write_execution_time": 0,
    "Data_for_localMaxima": "auto_background",
    "NumberMaxofFits": 5000,
}
# 'MaxPeakSize':3.0,
# 'MinPeakSize':0.01
# }


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
def setfilename(imagefilename, imageindex, nbdigits=4, CCDLabel=None):
    r"""
    reconstruct filename string from imagefilename and update filename index with imageindex

    :param imagefilename: filename string (full path or not)
    :param imageindex: index in filename 
    :type imageindex: string

    :return filename: input filename with index replaced by input imageindex
    :rtype filename: string
    """

    #     print "imagefilename",imagefilename
    if imagefilename.endswith("mccd"):
        lenext = 5 #length of extension including '.'
        imagefilename = imagefilename[: -(lenext + nbdigits)] + "{:04d}.mccd".format(
            imageindex
        )

    elif CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        #default file extension for sCMOS camera
        ext= "tif"
        lenext = 4 #length of extension including '.'
        if imagefilename.endswith("tiff"):
            ext = "tiff"
            lenext = 5 
        # zero padded index for filename
        
        if nbdigits is not None:
            if imagefilename.endswith(ext):
                imagefilename = imagefilename[: -(lenext + nbdigits)] + "{:04d}.{}".format(
                    imageindex,ext
                )
            elif imagefilename.endswith(ext+".gz"):
                imagefilename = imagefilename[
                    : -(lenext+3 + nbdigits)
                ] + "{:04d}.{}.gz".format(imageindex,ext)
        # no zero padded index for filename
        else:
            if imagefilename.endswith(ext):
                prefix, extension = imagefilename.split(".")
                prefix0 = prefix.split("_")[0]
                if imageindex > 9999:
                    imagefilename = prefix0 + "_{}.{}".format(imageindex,ext)
                else:
                    imagefilename = prefix0 + "_{:04d}.{}".format(imageindex,ext)

    elif CCDLabel in ("EIGER_4Mstack",):
        # only stackimageindex is changed not imagefilename
        pass

    #     print "imagefilename archetype", imagefilename
    elif imagefilename.endswith("mar.tiff"):

        imagefilename = imagefilename[: -(9 + nbdigits)] + "{:04d}_mar.tiff".format(
            imageindex
        )

    elif imagefilename.endswith("mar.tif"):
        # requires two underscores with number in between
        # map1_105_mar.tif
        sname = imagefilename.split("_")
        if len(sname) >= 4:  # pathfilename contain more than 1 underscore

            sname2 = sname[-2] + "_" + sname[-1]
            prefix = imagefilename.rstrip(sname2)
        else:
            prefix = imagefilename.split("_")[0]
        imagefilename = prefix + "_{}_mar.tif".format(imageindex)

    # special case for image id15 frelon corrected form distorsion
    elif imagefilename.endswith((".tif", ".edf")):
        if CCDLabel in ("ImageStar_raw"):
            prefixwihtindex = imagefilename[:-4]
            prefixwithindex_list = list(prefixwihtindex)
            indexnodigit = 0
            while len(prefixwithindex_list) > 0:
                lastelem = prefixwithindex_list.pop(-1)
                print("lastelem", lastelem)
                if not lastelem.isdigit():
                    break
                indexnodigit += 1
            prefix = prefixwihtindex[:-(indexnodigit)]
            print("prefix", prefix)
            imagefilename = prefix + "{}.tif".format(imageindex)

        else:
            suffix = imagefilename[-4:]
            prefix = imagefilename[: -(4 + nbdigits)]

            imagefilename = prefix + "{:04d}{}".format(imageindex, suffix)

    elif imagefilename.endswith("mccd"):

        imagefilename = imagefilename[: -(5 + nbdigits)] + "{:04d}.mccd".format(
            imageindex
        )

    elif imagefilename.endswith("edf"):

        imagefilename = imagefilename[: -(4 + nbdigits)] + "{:04d}.edf".format(
            imageindex
        )

    elif imagefilename.endswith("unstacked"):

        imagefilename = imagefilename[: -(10 + nbdigits)] + "{:04d}.unstacked".format(
            imageindex
        )

    #     print "set filename to:", imagefilename
    return imagefilename


def getIndex_fromfilename(imagefilename, nbdigits=4, CCDLabel=None, stackimageindex=-1):
    """
    get integer index from imagefilename string

    :param imagefilename: filename string (full path or not)

    :return: file index
    """
    #     print "CCDLabel",CCDLabel
    #     print "imagefilename",imagefilename

    
    if CCDLabel in ("sCMOS", "sCMOS_fliplr"):
        #default file extension for sCMOS camera
        ext= "tif"
        lenext = 4 #length of extension including '.'
        
        if imagefilename.endswith("tiff"):
            ext = "tiff"
            lenext = 5
        print(imagefilename)
        if nbdigits is not None:
            if imagefilename.endswith(ext):
                imageindex = int(imagefilename[-(lenext + nbdigits) : -(lenext)])
            elif imagefilename.endswith(ext+".gz"):
                imageindex = int(imagefilename[-(lenext+3 + nbdigits) : -(lenext+3)])
        else:
            if imagefilename.endswith(ext):
                prefix, extension = imagefilename.split(".")
                imageindex = int(prefix.split("_")[1])

    # for stacked images we return the position of image data in the stack as imagefileindex
    elif CCDLabel in ("EIGER_4Mstack",):
        imageindex = stackimageindex

    elif imagefilename.endswith("mar.tiff"):
        imageindex = int(imagefilename[-(9 + nbdigits) : -9])
    elif imagefilename.endswith("tiff"):
        imageindex = int(imagefilename[-(5 + nbdigits) : -5])
    elif imagefilename.endswith("mar.tif"):
        imageindex = int(imagefilename.split("_")[-2])
    elif imagefilename.endswith(".tif"):
        # TODO: treat the case of HN56.tif  without underscore (see setfilename)
        imageindex = int(imagefilename.split("_")[-1][:-4])
    elif imagefilename.endswith("mccd"):
        imageindex = int(imagefilename[-(5 + nbdigits) : -5])
    elif imagefilename.endswith("edf"):
        imageindex = int(imagefilename[-(4 + nbdigits) : -4])
    elif imagefilename.endswith("unstacked"):
        imageindex = int(imagefilename[-(10 + nbdigits) : -10])

    return imageindex


def docstringsexample():
    r"""
    docstrings format such as use numpydoc extension in conf.py of sphinx

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array. We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    Long_variable_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc :   Relationship (optional), which could be fairly long, in which
                case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_. You may also cite these
    references in the notes section above.
    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a=[1,2,3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """
    pass


def getwildcardstring(CCDlabel):
    r"""
    return smart wildcard to open binary CCD image file with priority of CCD type of CCDlabel

    Parameters
    ----------
    CCDlabel : string
        label defining the CCD type

    Returns
    -------
    wildcard_extensions : string
        string from concatenated strings to be used in wxpython open file dialog box

    See Also
    ----------

    :func:`getIndex_fromfilename`

    LaueToolsGUI.AskUserfilename

    wx.FileDialog

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.
    
    >>> from readmccd import getwildcardstring
    >>> getwildcardstring('MARCCD165')
    'MARCCD, ROPER(*.mccd)|*mccd|mar tif(*.tif)|*_mar.tiff|tiff(*.tiff)|*tiff|Princeton(*.spe)|*spe|Frelon(*.edf)|*edf|tif(*.tif)|*tif|All files(*)|*'

    """
    ALL_EXTENSIONS = ["mccd", "_mar.tiff", "tiff", "spe", "edf", "tif", "h5", ""]
    INFO_EXTENSIONS = [
        "MARCCD, ROPER(*.mccd)",
        "mar tif(*.tif)",
        "tiff(*.tiff)",
        "Princeton(*.spe)",
        "Frelon(*.edf)",
        "tif(*.tif)",
        "hdf5(*.h5)",
        "All files(*)",
    ]

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


def getpixelValue(filename, x, y, ccdtypegeometry="edf"):
    """
    x,y on display mapcanvas of lauetools are swaped by respect to array = d[y,x]

    for .edf file there is 2047-y on top of that
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
    """
    return header in a raw format

    default offset for marccd image
    """
    if CCDLabel.startswith("sCMOS"):
        filesize = os.path.getsize(filename)
        framedim = DictLT.dict_CCD[CCDLabel][0]
        offset = filesize - np.prod(framedim) * 2
    f = open(filename, "rb")
    myheader = f.read(offset)
    myheader.replace("\x00", " ")

    f.close()
    return myheader


def read_header_marccd(filename):
    """
    return string of parameters found in header in marccd image file .mccd

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    f = open(filename, "rb")
    f.seek(2048)
    posbyte = 0
    allsentences = ""
    for k in list(range(32)):
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
    """
    return string of parameters comments and exposure time
    found in header in marccd image file .mccd

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    f = open(filename, "rb")
    f.seek(3072)
    tt = f.read(512)
    # from beamline designed input
    dataset_comments = tt.strip("\x00")

    f.seek(1024 + 2 * 256 + 128 + 12)
    s = struct.Struct("I I I")
    unpacked_data = s.unpack(f.read(3 * 4))
    #     print 'Unpacked Values:', unpacked_data
    integration_time, expo_time, readout_time = unpacked_data

    f.close()
    return dataset_comments, expo_time


def read_header_scmos(filename):
    """
    return string of parameters comments and exposure time
    found in header in scmis image file .tif

    - print allsentences  displays the header
    - use allsentences.split('\n') to get a list
    """
    if not PIL_EXISTS:
        return {}

    img = Image.open(filename)

    # img.tag.keys()
    # tag[270]   = (u'0 (thf=-50.4850 mon=24923 exposure=0.400)',)

    dictpar = {}
    strcom = img.tag[270]
    si = strcom.index("(")
    fi = strcom.index(")")
    listpar = strcom[si + 1 : fi].split()
    #     print "listpar",listpar
    for elem in listpar:
        if "=" in elem:
            key, val = elem.split("=")
            dictpar[key] = float(val)

    return dictpar


def read_motorsposition_fromheader(filename, CCDLabel="MARCCD165"):
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
    """
    too SLOW!
    reads 1 entire image (marCCD format)
    returns:
    PILimage: image object of PIL module (16 bits integer)
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

    if dirname == None:
        dirname = os.curdir

    pilimage = Image.open(os.path.join(dirname, filename))
    ravdata = np.array(pilimage.getdata())

    return pilimage, np.reshape(ravdata, shapeCCD)


def readCCDimage(
    filename, CCDLabel="PRINCETON", dirname=None, stackimageindex=-1, verbose=0
):
    """
    read raw data image file and return pixel intensity 2D array
    such as to fit the data (2theta, chi) scattering angles representation convention

    Parameters
    -------------
    filename : string
               path to image file (fullpath if `dirname`=None)

    Returns
    ----------
    dataimage : 2D array
                image data pixel intensity properly oriented
    framedim : iterable of 2 integers
               shape of dataimage
    fliprot : string
              key for CCD frame transform to orient image
    """

    (
        framedim,
        pixelsize,
        saturationvalue,
        fliprot,
        offsetheader,
        formatdata,
        comments,
        extension,
    ) = DictLT.dict_CCD[CCDLabel]

    #     print "CCDLabel in readCCDimage", CCDLabel

    #    if extension != extension:
    #        print "warning : file extension does not match CCD type set in Set CCD File Parameters"
    if (
        CCDLabel
        in (
            "EDF",
            "EIGER_4M",
            "EIGER_1M",
            "sCMOS",
            "sCMOS_fliplr",
            "sCMOS_fliplr_16M",
            "sCMOS_16M",
            "Rayonix MX170-HS",
        )
        and FABIO_EXISTS
    ):
        # warning import Image  # for well read of header only

        if dirname is not None:
            img = fabio.open(os.path.join(dirname, filename))
        else:
            img = fabio.open(filename)

        dataimage = img.data
        framedim = dataimage.shape

        # pythonic way to change immutable tuple...
        initframedim = list(DictLT.dict_CCD[CCDLabel][0])
        initframedim[0] = framedim[0]
        initframedim[1] = framedim[1]
        initframedim = tuple(initframedim)

    elif CCDLabel in ("EIGER_4Mstack"):

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

        print("opening hdf5 stacked data table")
        alldata = hdf5file.root.entry.data.data
        print("alldata.shape", alldata.shape)

        dataimage = alldata[stackimageindex]
        framedim = dataimage.shape

    elif CCDLabel in (
        "TIFF Format",
        "FRELONID15_corrected",
        "VHR_PSI",
        "VHR_DLS",
        "MARCCD225",
        "Andrea",
        "pnCCD_Tuba",
    ):
        if LIBTIFF_EXISTS and CCDLabel not in ("Andrea",):
            print("using libtiff...")
            #         print "tiff format", CCDLabel
            #             print "dirname, filename", dirname, filename
            if dirname is not None:
                tifimage = TIFF.open(os.path.join(dirname, filename), mode="r")
            else:
                tifimage = TIFF.open(filename, mode="r")

            dataimage = tifimage.read_image()
            framedim = (
                int(tifimage.GetField("ImageLength")),
                int(tifimage.GetField("ImageWidth")),
            )
            if tifimage.IsByteSwapped():
                dataimage = dataimage.byteswap()
        else:
            print("not using libtiff")
            if CCDLabel in ("FRELONID15_corrected",):
                # TODO robust with dirname =None or?
                dataimage = readoneimage(
                    filename,
                    framedim=framedim,
                    dirname=dirname,
                    offset=offsetheader,
                    formatdata=formatdata,
                )

                dataimage = dataimage.byteswap()
            if (
                CCDLabel in ("VHR_PSI", "VHR_DLS", "MARCCD225", "Andrea", "pnCCD_Tuba")
                and PIL_EXISTS
            ):
                # data are compressed!
                print("using PIL's module Image")

                if dirname is not None:
                    fullpath = os.path.join(dirname, filename)
                else:
                    fullpath = filename

                im = Image.open(fullpath, "r")
                dataimage = np.array(im.getdata()).reshape(framedim)

    else:
        # offset header varying
        if CCDLabel.startswith("ImageStar_raw"):
            filesize = os.path.getsize(os.path.join(dirname, filename))
            bytes_per_pixels = 2
            if CCDLabel.endswith("32bits"):
                bytes_per_pixels = 4

            nbpixels = 1500
            if CCDLabel in ("ImageStar_1528x1528",):
                nbpixels = 1528
            offsetheader = filesize - nbpixels * nbpixels * bytes_per_pixels

        # almost very general case
        print(
            "\n\n\n WARNING: A very basic way was used to open image. Image is likely to be not well loaded ...\n\n\n"
        )
        dataimage = readoneimage(
            filename,
            framedim=framedim,
            dirname=dirname,
            offset=offsetheader,
            formatdata=formatdata,
        )

        if CCDLabel in ("EIGER_4Munstacked",):
            print("framedim", framedim)
            print("offsetheader", offsetheader)
            print("formatdata", formatdata)

            dataimage = np.ma.masked_where(dataimage > 4000000, dataimage)

            print("dataimage", dataimage)

    if verbose:
        print("CCDLabel: ", CCDLabel)
        print("nb of pixels", np.shape(dataimage))

    # need to reshape data from 1D to 2D
    try:
        if len(dataimage.shape) == 1:
            print("nb elements", len(dataimage))
            print("framedim", framedim)
            print("framedim nb of elements", framedim[0] * framedim[1])
            dataimage = np.reshape(dataimage, framedim)
    except ValueError:
        raise ValueError(
            "Selected CCD type :{} may be wrong (or nb of pixels, dimensions...)".format(
                CCDLabel
            )
        )

    # some array transformations if needed depending on the CCD mounting
    if fliprot == "spe":
        dataimage = np.rot90(dataimage, k=1)

    elif fliprot == "VHR_Feb13":
        #            self.dataimage_ROI = np.rot90(self.dataimage_ROI, k=3)
        # TODO: do we need this left and right flip ?
        dataimage = np.fliplr(dataimage)

    elif fliprot == "sCMOS_fliplr":
        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhr":  # july 2012 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhrdiamond":  # july 2012 close to diamond monochromator crystal
        dataimage = np.rot90(dataimage, k=3)
        dataimage = np.fliplr(dataimage)

    elif fliprot == "frelon2":
        dataimage = np.flipud(dataimage)

    #     print "framedim",framedim, fliprot
    return dataimage, framedim, fliprot


def readoneimage(
    filename, framedim=(2048, 2048), dirname=None, offset=4096, formatdata="uint16"
):
    """
    returns a 1d array of integers from a binary image file (full data)

    Parameters
    -------------
    filename : string
               path to image file (fullpath if `dirname`=None)
    offset : integer
             nb of file header bytes
    framedim : iterable of 2 integers
               shape of expected 2D data
    formatdata : string
                 key for numpy dtype to decode binary file

    Returns
    ----------
    dataimage : 1D array
                image data pixel intensity
    """
    nb_elem = framedim[0] * framedim[1]

    if dirname == None:
        dirname = os.curdir

    f = open(os.path.join(dirname, filename), "rb")
    f.seek(offset)

    # d=scipy.io.fread(f,2048*2048,np.oldnumeric.Int16)
    # d = scipy.io.fread(f,nb_elem,np.oldnumeric.UInt16)
    d = np.fromfile(f, dtype=formatdata, count=nb_elem)
    f.close()
    return d


def readoneimage_band(
    filename,
    framedim=(2048, 2048),
    dirname=None,
    offset=4096,
    line_startindex=0,
    line_finalindex=2047,
    formatdata="uint16",
):
    """
    returns a 1d array of integers from a binary image file. Data located in band according shape of data (framedim)

    Parameters
    -------------
    filename : string
               path to image file (fullpath if `dirname`=None)
    offset : integer
             nb of file header bytes
    framedim : iterable of 2 integers
               shape of expected 2D data
    formatdata : string
                 key for numpy dtype to decode binary file

    Returns
    ----------
    dataimage : 1D array
                image data pixel intensity
    """
    if dirname == None:
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


def readoneimage_crop_fast(
    filename, dirname=None, CCDLabel="MARCCD165", firstElemIndex=0, lastElemIndex=2047
):
    """
    returns a 2d array of integers from a binary image file. Data are taken only from a rectangle
    with respect to firstElemIndex
    and lastElemIndex

    Parameters
    -------------
    filename : string
               path to image file (fullpath if `dirname`=None)
    offset : integer
             nb of file header bytes
    framedim : iterable of 2 integers
               shape of expected 2D data
    formatdata : string
                 key for numpy dtype to decode binary file

    Returns
    ----------
    dataimage : 1D array
                image data pixel intensity
    """
    (
        framedim,
        pixelsize,
        saturationvalue,
        fliprot,
        offsetheader,
        formatdata,
        comments,
        extension,
    ) = DictLT.dict_CCD[CCDLabel]

    print("framedim read from DictLT.dict_CCD in readoneimage_crop_fast()", framedim)
    print("formatdata", formatdata)
    print("offsetheader", offsetheader)

    if dirname == None:
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

    band1D = readoneimage_band(
        filename,
        framedim=framedim,
        dirname=dirname,
        offset=offsetheader,
        line_startindex=lineFirstElemIndex,
        line_finalindex=lineLastElemIndex,
        formatdata=formatdata,
    )

    #     print "band1D.shape",band1D.shape
    #     print "(lineLastElemIndex-lineFirstElemIndex,framedim[1])",(lineLastElemIndex-lineFirstElemIndex,framedim[1])
    #     print "((lineLastElemIndex-lineFirstElemIndex)*framedim[1])",((lineLastElemIndex-lineFirstElemIndex)*framedim[1])

    band2D = band1D.reshape((lineLastElemIndex - lineFirstElemIndex + 1, framedim[1]))

    dataimage2D[lineFirstElemIndex : lineLastElemIndex + 1, :] = band2D

    return dataimage2D, framedim, fliprot


def readrectangle_in_image(
    filename,
    pixx,
    pixy,
    halfboxx,
    halfboxy,
    dirname=None,
    CCDLabel="MARCCD165",
    verbose=True,
):
    """
    returns a 2d array of integers from a binary image file. Data are taken only from a rectangle
    centered on pixx, pixy 

    
    Returns
   -------
    dataimage : 2D array
                image data pixel intensity
    """
    (
        framedim,
        pixelsize,
        saturationvalue,
        fliprot,
        offsetheader,
        formatdata,
        comments,
        extension,
    ) = DictLT.dict_CCD[CCDLabel]

    if verbose:
        print(
            "framedim read from DictLT.dict_CCD in readrectangle_in_image()", framedim
        )
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

    if verbose:
        print("fullpathfilename", fullpathfilename)
    try:
        filesize = os.path.getsize(fullpathfilename)
    except OSError:
        print("missing file {}\n".format(fullpathfilename))
        return None

    # uint16
    offsetheader = filesize - (framedim[0] * framedim[1]) * nbBytesPerElement

    if verbose:
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

    #     ymin= (y-boxy)*framedim[1]
    #     ymax= (y+boxy)*framedim[1]

    lineFirstElemIndex = ypixmin
    lineLastElemIndex = ypixmax

    if verbose:
        print("lineFirstElemIndex", lineFirstElemIndex)
        print("lineLastElemIndex", lineLastElemIndex)

    band = readoneimage_band(
        fullpathfilename,
        framedim=framedim,
        dirname=None,
        offset=offsetheader,
        line_startindex=lineFirstElemIndex,
        line_finalindex=lineLastElemIndex,
        formatdata=formatdata,
    )

    nblines = lineLastElemIndex - lineFirstElemIndex + 1

    band2D = np.reshape(band, (nblines, framedim[1]))

    dataimage2D = np.zeros(framedim)

    if verbose:
        print("band2D.shape", band2D.shape)

    rectangle2D = band2D[:, xpixmin : xpixmax + 1]

    if verbose:
        print("rectangle2D.shape", rectangle2D.shape)

    return rectangle2D


def readoneimage_crop(
    filename, center, halfboxsize, CCDLabel="PRINCETON", dirname=None
):
    """
    return a cropped array of data read in an image file
    
    Parameters
    -------------
    filename : string
               path to image file (fullpath if `dirname`=None)
    center : iterable of 2 integers
             (x,y) pixel coordinates
    halfboxsize : integer or iterable of 2 integers
                  ROI half size in both directions
    
    Returns
    ----------
    dataimage : 1D array
                image data pixel intensity
                
    #TODO: useless?
    """
    if dirname == None:
        dirname = os.curdir
    else:
        filename = os.path.join(dirname, filename)

    if isinstance(halfboxsize, int):
        boxsizex, boxsizey = halfboxsize, halfboxsize
    elif len(halfboxsize) == 2:
        boxsizex, boxsizey = halfboxsize

    xpic, ypic = center

    dataimage, framedim, fliprot = readCCDimage(
        filename, CCDLabel=CCDLabel, dirname=None
    )

    x1 = np.maximum(0, xpic - boxsizex)
    x2 = np.minimum(framedim[1], xpic + boxsizex + 1)  # framedim[0]
    y1 = np.maximum(0, ypic - boxsizey)
    y2 = np.minimum(framedim[0], ypic + boxsizey + 1)  # framedim[1]

    return None, dataimage[y1:y2, x1:x2]


def readoneimage_multi_barycenters(filename, centers, boxsize, offsetposition=0):
    """
    SLOW !!
    TODO :   clip data if intensity is below 0.05*(max-min) in ROI
    TODO : too slow use numpy instead of PIL
    """

    Images, Data = readoneimage_manycrops_old(filename, centers, boxsize)

    BoxSize = np.ones((len(Images), 2)) * boxsize
    TabCentroid = []
    shapy = Data[0].shape
    for iii, dd in enumerate(Data):
        max_in_roi, min_in_roi = np.amax(dd), np.amin(dd)
        # print max_in_roi,min_in_roi
        # print centers[iii]

        # flat background approximation for small ROI
        bkg = min_in_roi * np.ones(shapy)
        clipped_dd = np.clip(dd - bkg, 0.5 * (max_in_roi - min_in_roi), max_in_roi * 2)
        # imean,jmean = scipy.ndimage.measurements.center_of_mass(dd-bkg)
        imean, jmean = ndimage.measurements.center_of_mass(clipped_dd)
        TabCentroid.append([imean + centers[iii][0], jmean + centers[iii][1]])

    return np.array(TabCentroid) - BoxSize


def readoneimage_manycrops(
    filename,
    centers,
    boxsize,
    stackimageindex=-1,
    CCDLabel="MARCCD165",
    addImax=False,
    use_data_corrected=None,
):
    """
    reads 1 image and extract many regions
    centered on center_pixel with xyboxsize dimensions in pixel unit
    
    Parameters
    -------------
    filename : string
               fullpath to image file
    centers : list or array of [int,int]
              centers (x,y) pixel coordinates
    use_data_corrected : enter data instead of reading data from file
                         must be a tuple of 3 elements:
                         fulldata, framedim, fliprot
                         where fulldata is a numpy.ndarray
                         as output by :func:`readCCDimage`
    boxsize : iterable 2 elements or integer
              boxsizes [in x, in y] direction or integer to set a square ROI

    Returns
    ----------
    Data : list of 2D array pixel intensity

    Imax :
    returns:
    array of data: list of 2D array of intensity
    """
    #     print "input parameters of readoneimage_manycrops"
    #     print (filename, centers, boxsize,
    #                            CCDLabel,
    #                            addImax,
    #                            use_data_corrected
    #                            )

    # if isinstance(boxsize, (int, float)):
    #     if boxsize > 0.0:
    #         xboxsize, yboxsize = int(boxsize), int(boxsize)
    # else:
    #     xboxsize, yboxsize = boxsize

    # use alternate data  (for instance for data from filename without background)
    if use_data_corrected is not None:
        if isinstance(use_data_corrected, tuple):
            if len(use_data_corrected) == 3:
                fulldata, framedim, fliprot = use_data_corrected
    # use data by reading file
    else:
        fulldata, framedim, fliprot = readCCDimage(
            filename, stackimageindex=stackimageindex, CCDLabel=CCDLabel, dirname=None
        )

    if type(boxsize) == type(5):
        boxsizex, boxsizey = boxsize, boxsize
    elif type(boxsize) == type((10, 20)):
        boxsizex, boxsizey = boxsize

    xpic, ypic = np.array(centers).T

    #    x1 = np.array(np.maximum(0, xpic - boxsizex), dtype=np.int)
    #    x2 = np.array(np.minimum(framedim[0], xpic + boxsizex), dtype=np.int)
    #    y1 = np.array(np.maximum(0, ypic - boxsizey), dtype=np.int)
    #    y2 = np.array(np.minimum(framedim[1], ypic + boxsizey), dtype=np.int)

    Data = []

    Imax = []

    print("framedim in readoneimage_manycrops", framedim)
    framedim = framedim[1], framedim[0]
    #    print "framedim in readoneimage_manycrops", framedim

    #     print "centers",centers
    for center in centers:
        i1, i2, j1, j2 = getindices2cropArray(center, (boxsizex, boxsizey), framedim)
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


def readoneimage_manycrops_old(filename, centers, boxsize):
    """
    !! SLOW !!
    old  and slow version retrieving data with PIL module

    reads 1 image in marCCD format and extract many regions
    centered on center_pixel with xyboxsizedimensions in pixel unit

    returns:
    PILimage: list image objects of PIL module
    arrayofdata: list of 2D array of intensity

    TODO: to be deleted
    """
    dimMCCD = (2047, 2047)  # MarCCD max value in two directions in array index unit
    dim = dimMCCD

    if isinstance(boxsize, (float, int)):
        xboxsize, yboxsize = int(boxsize), int(boxsize)
    else:
        xboxsize, yboxsize = boxsize

    f = Image.open(filename)  # file is only opened once

    if type(boxsize) == type(5):
        boxsizex, boxsizey = boxsize, boxsize
    elif type(boxsize) == type((10, 20)):
        boxsizex, boxsizey = boxsize

    xpic, ypic = np.array(centers).T

    x1 = np.maximum(0, xpic - boxsizex)
    x2 = np.minimum(dim[0], xpic + boxsizex)
    y1 = np.maximum(0, ypic - boxsizey)
    y2 = np.minimum(dim[1], ypic + boxsizey)

    Images = []
    Data = []
    for box in zip(x1, y1, x2, y2):
        _x1, _y1, _x2, _y2 = box
        cf = f.crop(box)
        Images.append(cf)
        ravdata = np.array(cf.getdata())
        Data.append(np.reshape(ravdata, (_y2 - _y1, _x2 - _x1)))
    return Images, Data


def LoadMARfile(filename):
    """
    load marCCD file and create a 1d array
    TODO: obsolete use readoneimage() directly
    """
    d = readoneimage(
        filename, framedim=(2048, 2048), dirname=None, offset=4096, formatdata="uint16"
    )

    return d


def readoneimage_multiROIfit(
    filename,
    centers,
    boxsize,
    stackimageindex=-1,
    CCDLabel="PRINCETON",
    baseline="auto",
    startangles=0.0,
    start_sigma1=1.0,
    start_sigma2=1.0,
    position_start="max",
    fitfunc="gaussian",
    showfitresults=1,
    offsetposition=0,
    verbose=0,
    xtol=0.00000001,
    addImax=False,
    use_data_corrected=None,
):
    r"""
    Fit several peaks in one image

    Parameters
    -------------
    filename : string
              full path to image file
    centers : list or array like with shape=(n,2)
              list of centers of selected ROI
    boxsize : (Truly HALF boxsize: fuill boxsize= 2 *halfboxsize +1)
                iterable 2 elements or integer
              boxsizes [in x, in y] direction or integer to set a square ROI

    Optional parameters
    ---------------------
    baseline : string
               'auto' (ie minimum intensity in ROI) or array of floats
    startangles : float or iterable of 2 floats
                  elliptic gaussian angle (major axis with respect to X direction),
                  one value or array of values
    start_sigma1, start_sigma2: floats
                                gaussian standard deviation (major and minor axis) in pixel,
    position_start : string
                     starting gaussian center:'max' (position of maximum intensity in ROI),
                     'centers' (centre of each ROI)

    offsetposition : integer
        0 for no offset
        1  XMAS compatible, since XMAS consider first pixel as index 1 (in array, index starts with 0)
        2  fit2d, since fit2d for peaksearch put pixel labelled n at the position n+0.5 (between n and n+1)

    use_data_corrected : tuple of 3 elements    
                         Enter data instead of reading data from file:
                         fulldata, framedim, fliprot
                         where fulldata is a ndarray

    returns
    -------------
    params_sol : list of results
                 bkg,  amp  (gaussian height-bkg), X , Y ,
                major axis standard deviation ,minor axis standard deviation,
                major axis tilt angle / Ox

    # TODO: setting list of initial guesses can be improve with
    scipy.ndimages of a concatenate array of multiple slices?
    """
    print("addImax", addImax)
    #     print "entrance of readoneimage_multiROIfit", (filename,
    #                              centers,
    #                              boxsize,
    #                              CCDLabel,
    #                              baseline,
    #                              startangles,
    #                              start_sigma1,
    #                              start_sigma2,
    #                              position_start,
    #                              fitfunc,
    #                              showfitresults,
    #                              offsetposition,
    #                              verbose,
    #                              xtol,
    #                              addImax,
    #                              use_data_corrected)

    # read data (several arrays)
    ResData = readoneimage_manycrops(
        filename,
        centers,
        boxsize,
        stackimageindex,
        CCDLabel=CCDLabel,
        addImax=addImax,
        use_data_corrected=use_data_corrected,
    )
    if addImax:
        Datalist, Imax = ResData
    else:
        Datalist = ResData

    framedim = DictLT.dict_CCD[CCDLabel][0]
    saturation_value = DictLT.dict_CCD[CCDLabel][2]

    #     print "len(Datalist)",Datalist
    # for elem in Datalist:
    #     print("nb 2D intensities arrays",len(elem))
    #     print("2D array intensity shape",elem.shape)
    Data = np.array(Datalist)
    #
    #     condition=Data>=saturation_value
    #     np.ma.masked_where(condition,Data,copy=True)
    #
    #     print "Data",Data

    # setting initial guessed values for each center
    nb_Images = len(Data)
    #     print "nb of images to fitdata ... in  readoneimage_multiROIfit()", nb_Images
    if baseline in ("auto", None):  # background height or baseline level
        list_min = []
        for k, dd in enumerate(Data):
            #            print "k, dd.shape", k, dd.shape
            list_min.append(np.amin(dd))
        start_baseline = list_min
    else:  # input numerical value array
        start_baseline = baseline

    if isinstance(startangles, (int, float)):
        start_anglerot = startangles * np.ones(nb_Images)
    else:
        start_anglerot = startangles

    if isinstance(start_sigma1, (int, float)):
        start_sigma1 = start_sigma1 * np.ones(nb_Images)
    if isinstance(start_sigma2, (int, float)):
        start_sigma2 = start_sigma2 * np.ones(nb_Images)

    if isinstance(boxsize, (int, float)):
        xboxsize, yboxsize = int(boxsize), int(boxsize)
    else:
        xboxsize, yboxsize = boxsize

    Xboxsize = xboxsize * np.ones(nb_Images)
    Yboxsize = yboxsize * np.ones(nb_Images)

    start_j = []
    start_i = []
    start_amplitude = []

    if position_start in ("centers", "center"):  # starting position  from input center
        start_j, start_i = yboxsize, xboxsize
        start_amplitude = []
        d = 0
        for dd in Data:
            start_amplitude.append(
                dd[int(start_j[d]), int(start_i[d])] - start_baseline[d]
            )
            d += 1
    elif position_start == "max":  # starting position  from maximum intensity in dat

        d = 0
        for dd in Data:
            start_j.append(np.argmax(dd) // dd.shape[1])
            start_i.append(np.argmax(dd) % dd.shape[1])
            start_amplitude.append(np.amax(dd) - start_baseline[d])
            d += 1

    startingparams_zip = np.array(
        [
            start_baseline,
            start_amplitude,
            start_j,
            start_i,
            start_sigma1,
            start_sigma2,
            start_anglerot,
        ]
    )

    RES_params = []
    RES_cov = []
    RES_infodict = []
    RES_errmsg = []

    if verbose:
        print("startingparams_zip", startingparams_zip.T)

    # consider that ROi shape will be constant over all ROIs,
    # so no need to recompute np.indices in gaussfit
    ROIshape = Data[0].shape
    ijindices_array = np.indices(ROIshape)

    k_image = 0
    for startingparams in startingparams_zip.T:
        # if (k_image%25) == 0: print "%d/%d"%(k_image,nb_Images)
        if verbose:
            print("startingparams", startingparams)
        if fitfunc == "gaussian":
            if Data[k_image].shape != ROIshape:
                ijindices_array = None

            ROIdata = Data[k_image]
            #             print 'ROIdata', ROIdata
            #             print 'np.amax(ROIdata)', np.amax(ROIdata)
            #             print 'np.amin (ROIdata)', np.amin(ROIdata)
            #             print 'np.argmax (ROIdata)', np.argmax(ROIdata)

            params, cov, infodict, errmsg = fit2d.gaussfit(
                Data[k_image],
                err=None,
                params=startingparams,
                autoderiv=1,
                return_all=1,
                circle=0,
                rotate=1,
                vheight=1,
                xtol=xtol,
                Acceptable_HighestValue=saturation_value,
                Acceptable_LowestValue=0,
                ijindices_array=ijindices_array,
            )

        elif fitfunc == "lorentzian":
            params, cov, infodict, errmsg = fit2d_l.lorentzfit(
                Data[k_image],
                err=None,
                params=startingparams,
                autoderiv=1,
                return_all=1,
                circle=0,
                rotate=1,
                vheight=1,
                xtol=xtol,
            )

        if showfitresults:
            # print "startingparams"
            # print startingparams
            print("\n *****fitting results ************\n")
            print(params)
            print(
                "background intensity:                        {:.2f}".format(params[0])
            )
            print(
                "Peak amplitude above background              {:.2f}".format(params[1])
            )
            print(
                "pixel position (X)                   {:.2f}".format(
                    params[3] - Xboxsize[k_image] + centers[k_image][0]
                )
            )  # WARNING Y and X are exchanged in params !
            print(
                "pixel position (Y)                   {:.2f}".format(
                    params[2] - Yboxsize[k_image] + centers[k_image][1]
                )
            )
            print(
                "std 1,std 2 (pix)                    ( {:.2f} , {:.2f} )".format(
                    params[4], params[5]
                )
            )
            print(
                "e=min(std1,std2)/max(std1,std2)              {:.3f}".format(
                    min(params[4], params[5]) / max(params[4], params[5])
                )
            )
            print("Rotation angle (deg)                 {:.2f}".format(params[6] % 360))
            print("************************************\n")
        bkg_sol, amp_sol, Y_sol, X_sol, std1_sol, std2_sol, ang_sol = params

        RES_cov.append(cov)
        RES_infodict.append(infodict)
        RES_errmsg.append(errmsg)

        params_sol = np.array(
            [
                bkg_sol,
                amp_sol,
                X_sol - Xboxsize[k_image] + centers[k_image][0],
                Y_sol - Yboxsize[k_image] + centers[k_image][1],
                std1_sol,
                std2_sol,
                ang_sol,
            ]
        )  # now X,Y in safest order

        if offsetposition == 1:
            # PATCH: To match peak positions given by XMAS
            # (confusion coming from counting array indices from 0 or 1...)
            # array fitted by python module see pixels at lower position
            params_sol[3] = params_sol[3] + 1.0
            params_sol[2] = params_sol[2] + 1.0
            # End of PATCH

        elif offsetposition == 2:  # see Compute_data2thetachi() in find2thetachi.py
            # PATCH: To match peak positions given by peaksearch of fit2D
            # in fit2D graphics window first pixel labelled 1 is for the
            # peaksearch located at position in  between 0 and 1 (ie 0.5)
            params_sol[3] = params_sol[3] + 0.5
            params_sol[2] = (
                framedim[0] - params_sol[2]
            ) + 0.5  # TODO: tocheck dim[0] or dim[1]
            # End of PATCH

        RES_params.append(params_sol)

        k_image += 1

    if addImax:
        return RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline, Imax
    else:
        return RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline


def fitPeakMultiROIs(
    Data, centers, FittingParametersDict, showfitresults=True, verbose=False
):

    # MUST BE  2n+1 odd

    boxsize = FittingParametersDict["boxsize"]
    print("bosize", boxsize)
    if (boxsize[0] % 2 == 0) or (boxsize[1] % 2 == 0):
        print("boxsize", boxsize)
        raise ValueError("boxsizes are not odd !!")

    framedim = FittingParametersDict["framedim"]
    saturation_value = FittingParametersDict["saturation_value"]
    baseline = FittingParametersDict["baseline"]

    startangles = FittingParametersDict["startangles"]
    position_start = FittingParametersDict["position_start"]
    start_sigma1 = FittingParametersDict["start_sigma1"]
    start_sigma2 = FittingParametersDict["start_sigma2"]

    fitfunc = FittingParametersDict["fitfunction"]
    xtol = FittingParametersDict["xtol"]
    offsetposition = FittingParametersDict["offsetposition"]

    # setting initial guessed values for each center
    nb_Images = len(Data)
    print("nb of images fitPeakMultiROIs", nb_Images)
    print("shape of Data", Data.shape)

    if baseline in ("auto", None):  # background height or baseline level
        list_min = []
        for k, dd in enumerate(Data):
            #            print "k, dd.shape", k, dd.shape
            list_min.append(np.amin(dd))
        start_baseline = list_min
    else:  # input numerical value array
        start_baseline = baseline

    if isinstance(startangles, (int, float)):
        start_anglerot = startangles * np.ones(nb_Images)
    else:
        start_anglerot = startangles

    if isinstance(start_sigma1, (int, float)):
        start_sigma1 = start_sigma1 * np.ones(nb_Images)
    if isinstance(start_sigma2, (int, float)):
        start_sigma2 = start_sigma2 * np.ones(nb_Images)

    if isinstance(boxsize, (int, float)):
        xboxsize, yboxsize = int(boxsize), int(boxsize)
    else:
        xboxsize, yboxsize = boxsize

    halfxboxsize = int(xboxsize / 2.0)
    halfyboxsize = int(yboxsize / 2.0)

    Xhalfboxsize = halfxboxsize * np.ones(nb_Images)
    Yhalfboxsize = halfyboxsize * np.ones(nb_Images)

    start_j = []
    start_i = []
    start_amplitude = []

    if position_start in ("centers", "center"):  # starting position  from input center
        start_j, start_i = halfyboxsize, halfxboxsize
        start_amplitude = []
        d = 0
        for dd in Data:
            start_amplitude.append(
                dd[int(start_j[d]), int(start_i[d])] - start_baseline[d]
            )
            d += 1
    elif position_start == "max":  # starting position  from maximum intensity in dat

        d = 0
        for dd in Data:
            start_j.append(np.argmax(dd) // dd.shape[1])
            start_i.append(np.argmax(dd) % dd.shape[1])
            start_amplitude.append(np.amax(dd) - start_baseline[d])
            d += 1

    #     print "parame",(start_baseline,
    #                                    start_amplitude,
    #                                    start_j, start_i,
    #                                    start_sigma1, start_sigma2,
    #                                    start_anglerot)

    startingparams_zip = np.array(
        [
            start_baseline,
            start_amplitude,
            start_j,
            start_i,
            start_sigma1,
            start_sigma2,
            start_anglerot,
        ]
    )

    RES_params = []
    RES_cov = []
    RES_infodict = []
    RES_errmsg = []

    if verbose:
        print("startingparams_zip", startingparams_zip.T)

    # consider that ROi shape will be constant over all ROIs,
    # so no need to recompute np.indices in gaussfit
    ROIshape = Data[0].shape
    ijindices_array = np.indices(ROIshape)

    k_image = 0
    for startingparams in startingparams_zip.T:
        # if (k_image%25) == 0: print "%d/%d"%(k_image,nb_Images)
        if verbose:
            print("startingparams", startingparams)
        if fitfunc == "gaussian":
            if Data[k_image].shape != ROIshape:
                ijindices_array = None

            ROIdata = Data[k_image]
            #             print 'ROIdata', ROIdata
            #             print 'np.amax(ROIdata)', np.amax(ROIdata)
            #             print 'np.amin (ROIdata)', np.amin(ROIdata)
            #             print 'np.argmax (ROIdata)', np.argmax(ROIdata)

            params, cov, infodict, errmsg = fit2d.gaussfit(
                Data[k_image],
                err=None,
                params=startingparams,
                autoderiv=1,
                return_all=1,
                circle=0,
                rotate=1,
                vheight=1,
                xtol=xtol,
                Acceptable_HighestValue=saturation_value,
                Acceptable_LowestValue=0,
                ijindices_array=ijindices_array,
            )

        elif fitfunc == "lorentzian":
            params, cov, infodict, errmsg = fit2d_l.lorentzfit(
                Data[k_image],
                err=None,
                params=startingparams,
                autoderiv=1,
                return_all=1,
                circle=0,
                rotate=1,
                vheight=1,
                xtol=xtol,
            )

        if showfitresults:
            print("\n *****fitting results ************\n")
            print("  for k_image = ", k_image)
            print(params)
            print(
                "background intensity:                        {:.2f}".format(params[0])
            )
            print(
                "Peak amplitude above background              {:.2f}".format(params[1])
            )
            print(
                "pixel position (X)                   {:.2f}".format(
                    params[3] - Xhalfboxsize[k_image] + centers[k_image][0]
                )
            )  # WARNING Y and X are exchanged in params !
            print(
                "pixel position (Y)                   {:.2f}".format(
                    params[2] - Yhalfboxsize[k_image] + centers[k_image][1]
                )
            )
            print(
                "std 1,std 2 (pix)                    ( {:.2f} , {:.2f} )".format(
                    params[4], params[5]
                )
            )
            print(
                "e=min(std1,std2)/max(std1,std2)              {:.3f}".format(
                    min(params[4], params[5]) / max(params[4], params[5])
                )
            )
            print("Rotation angle (deg)                 {:.2f}".format(params[6] % 360))
            print("- Xboxsize[k_image]", -Xhalfboxsize[k_image])
            print("centers[k_image][0]", centers[k_image][0])
            print("************************************\n")
        bkg_sol, amp_sol, Y_sol, X_sol, std1_sol, std2_sol, ang_sol = params

        RES_cov.append(cov)
        RES_infodict.append(infodict)
        RES_errmsg.append(errmsg)

        params_sol = np.array(
            [
                bkg_sol,
                amp_sol,
                X_sol - Xhalfboxsize[k_image] + centers[k_image][0],
                Y_sol - Yhalfboxsize[k_image] + centers[k_image][1],
                std1_sol,
                std2_sol,
                ang_sol,
            ]
        )  # now X,Y in safest order

        if offsetposition == 1:
            # PATCH: To match peak positions given by XMAS
            # (confusion coming from counting array indices from 0 or 1...)
            # array fitted by python module see pixels at lower position
            params_sol[3] = params_sol[3] + 1.0
            params_sol[2] = params_sol[2] + 1.0
            # End of PATCH

        elif offsetposition == 2:  # see Compute_data2thetachi() in find2thetachi.py
            # PATCH: To match peak positions given by peaksearch of fit2D
            # in fit2D graphics window first pixel labelled 1 is for the
            # peaksearch located at position in  between 0 and 1 (ie 0.5)
            params_sol[3] = params_sol[3] + 0.5
            params_sol[2] = (
                framedim[0] - params_sol[2]
            ) + 0.5  # TODO: tocheck dim[0] or dim[1]
            # End of PATCH

        RES_params.append(params_sol)

        k_image += 1

    return RES_params, RES_cov, RES_infodict, RES_errmsg, start_baseline


def crop_fit2d(PILimage, fit2dlimits, plot=0, returndata=0):
    """
    old function to crop image read by PIL and
    boundaries given by fit2d display
    #TODO: to delete
    """
    xmin, ymin, xmax, ymax = fit2dlimits
    ymin_PIL = 2048 - ymax
    ymax_PIL = 2048 - ymin
    xmin_PIL = xmin
    xmax_PIL = xmax
    tuppy = xmin_PIL, ymin_PIL, xmax_PIL, ymax_PIL
    print(tuppy)

    PILcropimage = PILimage.crop(tuppy)
    if plot:
        PILcropimage.show()
    if returndata:  # in fit2d convention
        rawdata = np.array(PILcropimage.getdata())
        rraw = np.reshape(rawdata, (ymax_PIL - ymin_PIL, xmax_PIL - xmin_PIL))
        newdata_fit2d = np.transpose(np.flipud(rraw))
        # newdata[0,0] = fit2data [xmin+1,ymin+1]

        return PILcropimage, newdata_fit2d
    else:
        return None


def getindices2cropArray(center, halfboxsizeROI, arrayshape, flipxycenter=False):
    """
    return array indices limits to crop array data

    Parameters
    ------------
    center : iterable of 2 elements
             (x,y) pixel center of the ROI 
    halfboxsizeROI : integer or iterable of 2 elements
                     half boxsize ROI in two dimensions
    arrayshape : iterable of 2 integers
                 maximal number of pixels in both directions

    Options
    ------------
    flipxycenter : boolean
                   True: swap x and y of center with respect to others
                   parameters that remain fixed

    Return
    ------------
    imin, imax, jmin, jmax : 4 integers
                             4 indices allowing to slice a 2D np.ndarray
    """
    xpic, ypic = center
    if flipxycenter:
        ypic, xpic = center

    xpic, ypic = int(xpic), int(ypic)

    if isinstance(halfboxsizeROI, int):
        boxsizex, boxsizey = halfboxsizeROI, halfboxsizeROI
    else:
        boxsizex, boxsizey = halfboxsizeROI

    x1 = np.maximum(0, xpic - boxsizex)
    x2 = np.minimum(arrayshape[0], xpic + boxsizex)
    y1 = np.maximum(0, ypic - boxsizey)
    y2 = np.minimum(arrayshape[1], ypic + boxsizey)

    imin, imax, jmin, jmax = y1, y2, x1, x2

    return imin, imax, jmin, jmax


def check_array_indices(imin, imax, jmin, jmax, framedim=None):
    """
    Return 4 indices for array slice compatible with framedim

    Parameters
    -----------
    imin, imax, jmin, jmax: 4 integers
                            mini. and maxi. indices in both directions
    framedim : iterable of 2 integers
               shape of the array to be sliced by means of the 4 indices

    Return
    -------
    imin, imax, jmin, jmax: 4 integers
                            mini. and maxi. indices in both directions

    """
    if framedim is None:
        print("framedim is empty in check_array_indices()")
        return
    imin = max(imin, 0)
    jmin = max(jmin, 0)
    imax = min(framedim[0], imax)
    jmax = min(framedim[1], jmax)

    return imin, imax, jmin, jmax


# --- --------------  Modify images
def to8bits(PILimage, normalization_value=None):
    """
    convert PIL image (16 bits) in 8 bits PIL image
    returns:
    [0]  8 bits image
    [1] corresponding pixels value array
    
    TODO: since not used, may be deleted
    """

    imagesize = PILimage.size
    image8bits = Image.new("L", imagesize)
    rawdata = np.array(PILimage.getdata())
    if not normalization_value:
        normalization_value = 1.0 * np.amax(rawdata)
    datatoput = np.array(rawdata / normalization_value * 255, dtype="uint8")
    image8bits.putdata(datatoput)

    return image8bits, datatoput


def writeimage(outputname, _header, data, dataformat=np.uint16):
    """
    from data 1d array of integers
    with header coming from a f.open('imagefile'); f.read(headersize);f.close()
    WARNING: header contain dimensions for subsequent data. Check before the compatibility of
    data with header infos(nb of byte per pixel and array dimensions
    """
    newfile = open(outputname, "wb")
    newfile.write(_header)
    data = np.array(data, dtype=dataformat)
    data.tofile(newfile)
    newfile.close()
    print("image written in ", outputname)


def write_rawbinary(outputname, data, dataformat=np.uint16):
    """
    write a binary file without header of a 2D array

    """
    newfile = open(outputname, "wb")
    data = np.array(data, dtype=dataformat)
    data.tofile(newfile)

    newfile.close()
    print("image written in ", outputname)


def SumImages(
    prefixname,
    suffixname,
    ind_start,
    ind_end,
    dirname=None,
    plot=0,
    output_filename=None,
    CCDLabel=None,
    nbdigits=0,
):
    """
    sum images and write image with 32 bits per pixel format (4 bytes)
    """
    #     prefixname = 'HN08_'
    #     suffixname = '.tif'
    #     CCDLabel = 'ImageStar_raw'
    #     dirname = '/home/micha/LaueProjects/Vita'

    output_filename = "mean_{}_{:04d}_{}{}".format(
        prefixname, ind_start, ind_end, suffixname
    )

    filename = "{}{:04d}{}".format(prefixname, ind_start, suffixname)

    data, shape, fliprot = readCCDimage(filename, CCDLabel=CCDLabel, dirname=dirname)

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
        data, shape, fliprot = readCCDimage(
            filename, CCDLabel=CCDLabel, dirname=dirname
        )
        # print max(data1), np.argmax(data1)
        datasum = data + datasum

    if output_filename:
        outputfilename = output_filename
        header = readheader(os.path.join(dirname, filename), offset=offsetheader)
        writeimage(
            os.path.join(dirname, outputfilename), header, datasum, dataformat=np.uint32
        )

        print(
            "Added images with prefix {} from {} to {} written in {}".format(
                prefixname, ind_start, ind_end, outputfilename
            )
        )
    if plot:
        print("later")

    return datasum


def Add_Images2(
    prefixname,
    ind_start,
    ind_end,
    plot=0,
    writefilename=None,
    CCDLabel="MARCCD165",
    average=True,
):
    """


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
    """
    Add continuous sequence of images

    Parameters
    ----------
    prefixname : string
                 prefix common part of name of files

    ind_start : int
                starting image index

    ind_end : int
              final image index

    Optional Parameters
    ----------
    writefilename: string
                   new image filename where to write datastart (with last image file header read)

    Returns
    ---------
    datastart : array
                accumulation of 2D data from each image 

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


def rebin(a, *args):
    """rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:

    Examples
    -----------

    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) // np.asarray(args)
    evList = (
        ["a.reshape("]
        + ["args[{:d}],factor[{:d}],".format(i, i) for i in list(range(lenShape))]
        + [")"]
        + [".sum({:d})".format(i + 1) for i in list(range(lenShape))]
        + ["/factor[{:d}]".format(i) for i in list(range(lenShape))]
    )
    return (
        a.reshape(args[0], factor[0], args[1], factor[1]).sum(1).sum(2)
        / factor[0]
        / factor[1]
    )
    return eval("".join(evList))


def fromfit2D_to_array(peaklist):
    """ pixel frame conversion from fit2d to numpy
    #TODO(JSM): to be deleted
    """
    x, y = peaklist.T
    i = 2048 - y
    j = x - 1
    return np.array([i, j]).T


def fromarray_to_fit2d(peaklist):
    """ pixel frame conversion from numpy to fit2d
    """
    # i slow index, j fast index
    i, j = peaklist.T
    y = 2048 - i
    x = j + 1
    return np.array([x, y]).T


# --- -------------  getting data from images or ROI
def diff_pix(pix, array_pix, radius=1):
    """
    returns
    index in array_pix which is the closest to pix if below the tolerance radius

    array_pix: array of 2d pixel points
    pix: one 2elements pixel point
    """
    dist2 = sum((pix - array_pix) ** 2, axis=1)
    closepix = np.argmin(dist2)
    if closepix:
        if dist2[closepix] <= radius ** 2:
            return closepix
        else:
            return None


def mean_index(data):
    """
    returns barycenter of weights in data
    (slow index mean position, fast index mean position)
    mean(i) =  sum (i,j) data(i,j)i  / sum(i,j) data(i,j)
    mean(j) =  sum (i,j) dyata(i,j)j  / sum(i,j) data(i,j)

    #TODO: to BE replaced by : scipy.ndimage.measurements.center_of_mass(data)
    """
    SLow, Fast = np.indices(data.shape)
    total_I = 1.0 * np.sum(data)
    return np.array([np.sum(SLow * data), np.sum(Fast * data)]) / total_I


def minmax(D_array, center, boxsize, framedim=(2048, 2048), withmaxpos=False):
    """
    extract min and max from a 2d array in a ROI

    Parameters
    -----------
    D_array : 2D array
              data array
    center : iterable of 2 integers
             (x,y) pixel center
    boxsize : integer or iterable of 2 integers
              full boxsize defined in both directions
    framedim : iterable of 2 integers
               shape of D_array

    Return
    --------
    [min, max]: minimium and maximum pixel internsity in ROI
    [min, max],absolute_max_pos : if withmaxpos is True  add in output
                                  the absolute position of the largest pixel

    #TODO: replace by scipy.ndimage.extrema
    # see next functions below
    # framedim = from dictionary of CCDs
    D_array shape is flip(framedim)
    """
    if not isinstance(boxsize, int):
        boxsize = (boxsize, boxsize)
    #    print "framedim in minmax", framedim
    #    print "D_array.shape", D_array.shape

    # halfbox = int(boxsize / 2)

    #    xc, yc = center
    #    imin, imax, jmin, jmax = max(0, yc - halfbox), \
    #                            min(yc + halfbox, framedim[0]), \
    #                            max(0, xc - halfbox), \
    #                            min(framedim[1], xc + halfbox)
    #
    #    print "imin, imax, jmin, jmax", imin, imax, jmin, jmax
    framedim = framedim[1], framedim[0]
    imin, imax, jmin, jmax = getindices2cropArray(
        center, boxsize, framedim, flipxycenter=1
    )

    #    print "imin, imax, jmin, jmax", imin, imax, jmin, jmax

    fulldata = D_array
    array_short = fulldata[imin:imax, jmin:jmax]

    mini_in_ROI = np.amin(array_short)
    maxi_in_ROI = np.amax(array_short)

    absolute_max_pos = ndimage.maximum_position(array_short) + np.array([imin, jmin])

    if withmaxpos:
        return [mini_in_ROI, maxi_in_ROI], absolute_max_pos

    else:
        return [mini_in_ROI, maxi_in_ROI]


def getExtrema(data2d, center, boxsize, framedim, ROIcoords=0, flipxycenter=True):
    """
    return  min max XYposmin, XYposmax values in ROI

    Parameters
    ------------
    ROIcoords : 1 in local array indices coordinates
                0 in X,Y pixel CCD coordinates
    flipxycenter : boolean like
                   swap input center coordinates
    data2d : 2D array
             data array as read by :func:`readCCDimage`

    Return
    --------
    min, max, XYposmin, XYposmax:
        - min : minimum pixel intensity
        - max : maximum pixel intensity
        - XYposmin : list of absolute pixel coordinates of lowest pixel
        - XYposmax : list of absolute pixel coordinates of largest pixel

    """
    if center is None or len(center) == 0:
        raise ValueError("center (peak list) in getExtrema is empty")

    indicesborders = getindices2cropArray(
        center, [boxsize, boxsize], framedim, flipxycenter=flipxycenter
    )
    imin, imax, jmin, jmax = indicesborders

    print("imin, imax, jmin, jmax", imin, imax, jmin, jmax)
    datacropped = data2d[imin:imax, jmin:jmax]

    # mini, maxi, posmin, posmax

    mini, maxi, posmin, posmax = ndimage.measurements.extrema(datacropped)

    if ROIcoords:
        return mini, maxi, posmin, posmax
    else:
        max_i, max_j = posmax
        min_i, min_j = posmin

        if flipxycenter:
            centery, centerx = center
        else:
            centerx, centery = center
        #        print "local position of maximum i,j", max_i, max_j

        globalXmax = max_j + centerx - boxsize
        globalYmax = max_i + centery - boxsize

        globalXmin = min_j + centerx - boxsize
        globalYmin = min_i + centery - boxsize

        #        print "Highest intensity %.f at (X,Y): (%d,%d) " % (maxi, globalX, globalY)
        if flipxycenter:
            return mini, maxi, [globalYmin, globalXmin], [globalYmax, globalXmax]
        else:
            return mini, maxi, [globalXmin, globalYmin], [globalXmax, globalYmax]


def getIntegratedIntensities(
    fullpathimagefile,
    list_centers,
    boxsize,
    CCDLabel="MARCCD165",
    thresholdlevel=0.2,
    flipxycenter=True,
):
    """
    read binary image file and compute integrated intensities of peaks whose center is given in list_centers 
    
    return
    ----------
    array of
    column 0: integrated intensity
    column 1: absolute minimum intensity threshold
    column 2: nb of pixels composing the peak
    """
    dataimage, framedim, fliprot = readCCDimage(fullpathimagefile, CCDLabel, None, 0)
    res = []
    for center in list_centers:
        res.append(
            getIntegratedIntensity(
                dataimage, center, boxsize, framedim, thresholdlevel, flipxycenter
            )
        )
    return np.array(res)


def getIntegratedIntensity(
    data2d, center, boxsize, framedim, thresholdlevel=0.2, flipxycenter=True
):
    """
    return  crude estimate of integrated intensity of peak above a given relative threshold
    
    # TODO: center is a only a single center, need to extend to several centers...  

    Parameters
    ------------
    ROIcoords : 1 in local array indices coordinates
                0 in X,Y pixel CCD coordinates
    flipxycenter : boolean like
                   swap input center coordinates
    data2d : 2D array
             data array as read by :func:`readCCDimage`
             
    Thresholdlevel  :  relative level above which pixel intensity must be taken into account
                I(p)- minimum> Thresholdlevel* (maximum-minimum)

    Return
    --------
    integrated intensity, minimum absolute intensity, nbpixels used for the summation

    """
    if center is None or len(center) == 0:
        raise ValueError("center (peak list) in getExtrema is empty")

    indicesborders = getindices2cropArray(
        center, [boxsize, boxsize], framedim, flipxycenter=flipxycenter
    )
    imin, imax, jmin, jmax = indicesborders

    #     print "imin, imax, jmin, jmax", imin, imax, jmin, jmax
    datacropped = data2d[imin:imax, jmin:jmax]

    # mini, maxi, posmin, posmax

    mini, maxi, posmin, posmax = ndimage.measurements.extrema(datacropped)

    minimum_amplitude = thresholdlevel * (maxi - mini) + mini
    print("integration for pixel intensity higher than: ", minimum_amplitude)
    pixelsabove = datacropped[datacropped > minimum_amplitude]
    nbpixels = len(pixelsabove)
    print("nb pixels above threshold  ", nbpixels)
    return ndimage.measurements.sum(pixelsabove), minimum_amplitude, nbpixels


def getMinMax(data2d, center, boxsize, framedim):
    r"""
    return min and max values in ROI

    Parameters:
    --------------
    data2d : 2D array
             array as read by readCCDimage
    """
    return getExtrema(data2d, center, boxsize, framedim)[:2]


def minmax_fast(D_array, centers, boxsize=(25, 25)):
    """
    extract min (considered as background in boxsize) and intensity at center
    from a 2d array at different places (centers)

    centers is tuple a two array (  array([slow indices]),  array([fast indices]))

    return:

    [0] background values
    [1] intensity value
    """

    min_array = ndimage.minimum_filter(D_array, size=boxsize)

    return [min_array[centers], D_array[centers]]


def get_imagesize(framedim, nbbits_per_pixel, headersize_bytes):
    """
    return size of image in byte (= 1 octet = 8 bits)
    """
    return (framedim[0] * framedim[1] * nbbits_per_pixel + headersize_bytes * 8) // 8


# --- ------------- Mexican Hat 2D kernel
def myfromfunction(f, s, t):
    return np.fromfunction(f, s).astype(t)


def normalize_shape(shape):
    """
    return shape
    in case a scalar was given:
    return (shape,)
    """
    try:
        len(shape)
        return shape
    except TypeError:
        return (shape,)


def LoG(r, sigma=None, dim=1, r0=None, peakVal=None):
    """note:
         returns *negative* Laplacian-of-Gaussian (aka. mexican hat)
         zero-point will be at sqrt(dim)*sigma
         integral is _always_ 0
         if peakVal is None:  uses "mathematical" "gaussian derived" norm
         if r0 is not None: specify radius of zero-point (IGNORE sigma !!)
    """
    r2 = r ** 2

    if sigma is None:
        if r0 is not None:
            sigma = float(r0) / np.sqrt(dim)
        else:
            raise ValueError("One of sigma or r0 have to be non-None")
    else:
        if r0 is not None:
            raise ValueError("Only one of sigma or r0 can be non-None")
    s2 = sigma ** 2
    dsd = dim * sigma ** dim

    if peakVal is not None:
        norm = peakVal / dsd
    else:
        norm = 1.0 / (s2 * (2.0 * np.pi * sigma) ** (dim / 2.0))
    return np.exp(-r2 / (2.0 * s2)) * (dsd - r2) * norm


def LoGArr(
    shape=(256, 256),
    r0=None,
    sigma=None,
    peakVal=None,
    orig=None,
    wrap=0,
    dtype=np.float32,
):
    """returns n-dim Laplacian-of-Gaussian (aka. mexican hat)
    if peakVal   is not None
         result max is peakVal
    if r0 is not None: specify radius of zero-point (IGNORE sigma !!)

    credits: "Sebastian Haase <haase@msg.ucsf.edu>"
    """
    shape = normalize_shape(shape)
    dim = len(shape)
    return radialArr(
        shape,
        lambda r: LoG(r, sigma=sigma, dim=dim, r0=r0, peakVal=peakVal),
        orig,
        wrap,
        dtype,
    )


def radialArr(shape, func, orig=None, wrap=False, dtype=np.float32):
    """generates and returns radially symmetric function sampled in volume(image) of shape shape
    if orig is None the origin defaults to the center
    func is a 1D function with 1 paramater: r

    if shape is a scalar uses implicitely `(shape,)`
    wrap tells if functions is continued wrapping around image boundaries
    wrap can be True or False or a tuple same length as shape:
       then wrap is given for each axis sperately
    """
    shape = normalize_shape(shape)
    try:
        len(shape)
    except TypeError:
        shape = (shape,)

    if orig is None:
        orig = (np.array(shape, dtype=np.float) - 1) / 2.0
    else:
        try:
            oo = float(orig)
            orig = np.ones(shape=len(shape)) * oo
        except:
            pass

    if len(shape) != len(orig):
        raise ValueError("shape and orig not same dimension")

    try:
        if len(wrap) != len(shape):
            raise ValueError("wrap tuple must be same length as shape")
    except TypeError:
        wrap = (wrap,) * len(shape)

    def wrapIt(ax, q):
        if wrap[ax]:
            nq = shape[ax]
            return np.where(q > nq // 2, q - nq, q)
        else:
            return q

    #     if wrap:
    #         def wrapIt(q, nq):
    #             return np.where(q>nq/2,q-nq, q)
    #     else:
    #         def wrapIt(q, nq):
    #             return q

    #     if len(shape) == 1:
    #         x0 = orig[0]  # 20060606: [0] prevents orig (as array) promoting its dtype (e.g. Float64) into result
    #         nx = shape[0]
    #         return myfromfunction(lambda x: func(wrapIt(np.absolute(x-x0),nx)), shape, dtype)
    #     elif len(shape) == 2:
    #         y0,x0 = orig
    #         ny,nx=shape
    #         return myfromfunction(lambda y,x: func(
    #                                              np.sqrt( \
    #             (wrapIt((x-x0),nx))**2 + (wrapIt((y-y0),ny))**2 ) ), shape, dtype)
    #     elif len(shape) == 3:
    #         z0,y0,x0 = orig
    #         nz,ny,nx=shape
    #         return myfromfunction(lambda z,y,x: func(
    #                                              np.sqrt( \
    #             (wrapIt((x-x0),nx))**2 + (wrapIt((y-y0),ny))**2 + (wrapIt((z-z0),nz))**2 ) ), shape, dtype)
    if len(shape) == 1:
        x0 = orig[
            0
        ]  # 20060606: [0] prevents orig (as array) promoting its dtype (e.g. Float64) into result
        return myfromfunction(
            lambda x: func(wrapIt(0, np.absolute(x - x0))), shape, dtype
        )
    elif len(shape) == 2:
        y0, x0 = orig
        return myfromfunction(
            lambda y, x: func(
                np.sqrt((wrapIt(-1, x - x0)) ** 2 + (wrapIt(-2, y - y0)) ** 2)
            ),
            shape,
            dtype,
        )
    elif len(shape) == 3:
        z0, y0, x0 = orig
        return myfromfunction(
            lambda z, y, x: func(
                np.sqrt(
                    (wrapIt(-1, x - x0)) ** 2
                    + (wrapIt(-2, y - y0)) ** 2
                    + (wrapIt(-3, z - z0)) ** 2
                )
            ),
            shape,
            dtype,
        )
    else:
        raise ValueError("only defined for dim < 3 (#TODO)")


# --- --------------------  Local Maxima or Local Hot pixels search
def LocalMaxima_ndimage(
    Data,
    peakVal=4,
    boxsize=5,
    central_radius=2,
    threshold=1000,
    connectivity=1,
    returnfloatmeanpos=0,
    autothresholdpercentage=None,
):

    r"""
    returns (float) i,j positions in array of each blob
    (peak, spot, assembly of hot pixels or whatever)

    input:

    peakVal, boxsize, central_radius    :
        parameters for numerical convolution with a mexican-hat-like kernel

    threshold :
        intensity threshold of filtered Data (by convolution with the kernel)
        above which blob signal will be considered
        if = 0 : take all blobs at the expense of processing time

    connectivity :
        1 for filled square 3*3 connectivity
        0 for 3*3 star like connectivity
        
    autothresholdpercentage :
        threshold in filtered image with respect to the maximum intensity in filtered image 

    output:
    array (n,2): array of 2 indices
    """
    aa = ConvolvebyKernel(
        Data, peakVal=peakVal, boxsize=boxsize, central_radius=central_radius
    )

    print("Histogram after convolution with Mexican Hat")
    print(np.histogram(aa))

    if autothresholdpercentage is None:
        thraa = np.where(aa > threshold, 1, 0)
    else:
        thraa = np.where(aa > (autothresholdpercentage / 100.0) * np.amax(aa), 1, 0)

    if connectivity == 0:
        star = np.eye(3)
        ll, nf = ndimage.label(thraa, structure=star)
    elif connectivity == 1:
        ll, nf = ndimage.label(thraa, structure=np.ones((3, 3)))
    elif connectivity == 2:
        ll, nf = ndimage.label(
            thraa, structure=np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]])
        )
    elif connectivity == 3:

        ll, nf = ndimage.label(
            thraa, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )

    #     meanpos = np.array(ndimage.measurements.center_of_mass(thraa,
    #                                                     ll,
    #                                                     np.arange(1, nf + 1)),
    #                                                     dtype=np.float)

    meanpos = np.array(
        ndimage.measurements.maximum_position(thraa, ll, np.arange(1, nf + 1)),
        dtype=np.float,
    )

    if returnfloatmeanpos:
        return meanpos
    else:
        return np.array(meanpos, dtype=np.int)


def ConvolvebyKernel(Data, peakVal=4, boxsize=5, central_radius=2):
    """
    Convolve Data array witn mexican-hat kernel

    inputs:
    Data                            : 2D array containing pixel intensities
    peakVal > central_radius        : defines pixel distance from box center where weights are positive
                                    (in the middle) and negative farther to converge back to zero
    boxsize                            : size of the box

    ouput:
    array  (same shape as Data)
    """

    # from scipy import ndimage

    # outa = np.zeros(Data.shape)
    # ndimage.filters.gaussian_laplace(d,(5,5),output=outa)

    # whole_structure= createstructure(10, 10)-2*createstructure(10, 7)+4*createstructure(10, 5)
    # bb= ndimage.convolve(d,whole_structure)

    # bb=ndimage.morphology.white_tophat(Data,(boxsize,boxsize))
    # mexicanhat = array(LoGArr((10,10),r0=6,peakVal=4),dtype= int16)

    mexicanhat = LoGArr((boxsize, boxsize), r0=central_radius, peakVal=peakVal)
    mexicanhat = mexicanhat - sum(mexicanhat) / mexicanhat.size
    bb = ndimage.convolve(np.array(Data, dtype=np.float32), mexicanhat)

    return bb


def LocalMaxima_KernelConvolution(
    Data,
    framedim=(2048, 2048),
    peakValConvolve=4,
    boxsizeConvolve=5,
    central_radiusConvolve=2,
    thresholdConvolve=1000,
    connectivity=1,
    IntensityThreshold=500,
    boxsize_for_probing_minimal_value_background=30,
    return_nb_raw_blobs=0,
    peakposition_definition="max",
):  # full side length

    r"""
    return local maxima (blobs) position and amplitude in Data by using
    convolution with a mexican hat like kernel.
    Two Thresholds are used sequently:
        - thresholdConvolve : level under which intensity of kernel-convolved array is discarded
        - IntensityThreshold : level under which blob whose local intensity amplitude in raw array is discarded  

    Parameters
    ------------
    
    Data : 2D array containing pixel intensities

    peakValConvolve, boxsizeConvolve, central_radiusConvolve : convolution kernel parameters
    
    thresholdConvolve : minimum threshold (expressed in unit of convolved array intensity)
                        under which convoluted blob is rejected.It can be zero
                        (all blobs are accepted but time consuming)
    connectivity : shape of connectivity pattern to consider pixels belonging to the
                   same blob.
                       - 1: filled square  (1 pixel connected to 8 neighbours)
                       - 0: star (4 neighbours in vertical and horizontal direction)

    IntensityThreshold : minimum local blob amplitude to accept
    
    boxsize_for_probing_minimal_value_background : boxsize to evaluate the background and the blob amplitude 

    peakposition_definition : string ('max' or 'center')
                              key to assign to the blob position its hottest pixel position
                              or its center (no weight)
    
    Returns:
    ---------
    
    peakslist : array like (n,2)
                list of peaks position (pixel)
    Ipixmax : array like (n,1) of integer
             list of highest pixel intensity in the vicinity of each peak
    npeaks : integer
             nb of peaks (if return_nb_raw_blobs =1)
    
    """
    print("framedim in LocalMaxima_KernelConvolution", framedim)
    print("Data.shape", Data.shape)
    dataimage_ROI = Data

    peak = LocalMaxima_ndimage(
        dataimage_ROI,
        peakVal=peakValConvolve,
        boxsize=boxsizeConvolve,
        central_radius=central_radiusConvolve,
        threshold=thresholdConvolve,
        connectivity=connectivity,
        returnfloatmeanpos=0,
    )

    #     print "peak in LocalMaxima_KernelConvolution", peak
    if len(peak) == 0:
        return None

    peak = (peak[:, 0], peak[:, 1])

    intensity_localmaxima = dataimage_ROI[peak]

    peaki = peak[0]
    peakj = peak[1]

    # building an array of hot pixels (2 coordinates)
    Yarray = peakj
    Xarray = peaki
    peaklist = np.array([Xarray, Yarray]).T

    #    print "peaklist", peaklist
    #     print "%d local maxima have been found from convolution method" % len(peaklist)
    #    print "peaklist[:3]", peaklist[:3]

    # probing background and maximal intensity in boxsize
    #
    ptp_boxsize = boxsize_for_probing_minimal_value_background

    # first method ---------------------------
    tabptp = []  # tab of min and max around each peak
    tabposmax = (
        []
    )  # # tab of position of hottest pixel close to that found after convolution
    for k in list(range(len(peaklist))):
        #        print "k in LocalMaxima_KernelConvolution", k
        #        print "dataimage_ROI.shape", dataimage_ROI.shape
        minimaxi, maxpos = minmax(
            dataimage_ROI, peaklist[k], ptp_boxsize, framedim=framedim, withmaxpos=1
        )
        tabptp.append(minimaxi)
        #         if minimaxi[1] > 4000:
        #             print "k, peaklist[k]", k, peaklist[k]
        #             print maxpos, minimaxi[1]

        #        mini, maxi, minpos, maxpos = getExtrema(dataimage_ROI, peaklist[k], ptp_boxsize, framedim,
        #                                                ROIcoords=0,
        #                                                flipxycenter=1)
        #
        #        tabptp.append([mini, maxi])
        #
        #        if maxi > 4000:
        #            print "k, peaklist[k]", k, peaklist[k]
        #            print maxpos, maxi

        tabposmax.append(maxpos)  # # new

    #    # to test peaks position / framedim
    #    print minmax(dataimage_ROI,
    #                                 [2580, 2750],
    #                                ptp_boxsize,
    #                                framedim=framedim,
    #                                withmaxpos=1)

    ar_ptp = np.array(tabptp)
    ar_posmax = np.array(tabposmax)  # # new

    #     print ' ar_posmax[:3]', ar_posmax[:10]
    #     print ' ar_posmax[:3]', ar_posmax[10:20]
    #     print ' ar_posmax[:3]', ar_posmax[20:30]

    #     print "saturation at", np.where(ar_ptp > 65000.)
    # -------------------------------------------

    # second method ---------------
    # tabptp = minmax_fast(dataimage_ROI,
    #                 tuple(transpose(peaklist)),
    #                    boxsize=(boxsize_for_probing_minimal_value_background,
    #                                boxsize_for_probing_minimal_value_background))
    # ar_ptp = array(tabptp).T
    # ---------------------------------

    # ar_amp = np.subtract(ar_ptp[:,1],ar_ptp[:,0])
    ar_amp = np.subtract(intensity_localmaxima, ar_ptp[:, 0])
    amp_rank = np.argsort(ar_amp)[::-1]

    peaklist_sorted = peaklist[amp_rank]
    ptp_sorted = ar_ptp[amp_rank]
    amp_sorted = ar_amp[amp_rank]
    posmax_sorted = ar_posmax[amp_rank]  # # new
    # thresholding on peak-to-peak amplitude
    threshold_amp = IntensityThreshold

    cond = np.where(amp_sorted > threshold_amp)
    th_peaklist = peaklist_sorted[cond]
    th_ar_ptp = ptp_sorted[cond]
    th_ar_amp = amp_sorted[cond]
    th_ar_pos = posmax_sorted[cond]  # # new

    #     print "th_ar_ptp", th_ar_ptp[:10]

    ##### peak positions that will be returned are the hottest pixels
    if peakposition_definition == "max":
        th_peaklist = th_ar_pos
    else:
        # using th_peaklist which is float position
        pass

    print(
        "{} local maxima found after thresholding above {} (amplitude above local background)".format(
            len(th_ar_amp), threshold_amp
        )
    )

    # NEW --- from method array shift!
    # remove duplicates (close points), the most intense pixel is kept
    # minimum distance between hot pixel
    # it corresponds both to distance between peaks and peak size ...
    pixeldistance = 10  # pixeldistance_remove_duplicates

    purged_pklist, index_todelete = GT.purgeClosePoints2(th_peaklist, pixeldistance)

    purged_amp = np.delete(th_ar_amp, index_todelete)
    purged_ptp = np.delete(th_ar_ptp, index_todelete, axis=0)

    #     print 'shape of purged_ptp method conv.', purged_ptp.shape
    print(
        "{} local maxima found after removing duplicates (minimum intermaxima distance = {})".format(
            len(purged_amp), pixeldistance
        )
    )

    # print "purged_pklist", purged_pklist
    #     print "shape(purged_pklist)", np.shape(purged_pklist)

    npeaks = np.shape(purged_pklist)[0]
    # print np.shape(Data)

    Ipixmax = purged_ptp[:, 1]
    # print "Ipixmax = ", Ipixmax

    peakslist = np.fliplr(purged_pklist)

    if return_nb_raw_blobs:
        return peakslist, Ipixmax, npeaks
    else:
        return peakslist, Ipixmax

    # NEW --- !
    # -----------------------


#     npeaks = np.shape(th_peaklist)[0]
#     Ipixmax = np.zeros(npeaks, dtype=int)
#     # print np.shape(Data)
#     for i in list(range(npeaks)):
#         # Ipixmax[i]=Data[th_peaklist[i,0],th_peaklist[i,1]]
#         Ipixmax[i] = th_ar_ptp[i][1]
#
#     if return_nb_raw_blobs == 1:
#         return np.fliplr(th_peaklist), Ipixmax, len(peaklist)
#     else:
#         return np.fliplr(th_peaklist), Ipixmax


def LocalMaxima_ShiftArrays(
    Data,
    framedim=(2048, 2048),
    IntensityThreshold=500,
    Saturation_value=65535,
    boxsize_for_probing_minimal_value_background=30,  # full side length
    nb_of_shift=25,
    pixeldistance_remove_duplicates=25,
    verbose=0,
):

    try:
        import networkx as NX
    except ImportError:
        print("\n***********************************************************")
        print(
            "networkx module is missing! Some functions may not work...\nPlease install it at http://networkx.github.io/"
        )
        print("***********************************************************\n")

    # time_0 = ttt.time()
    # pilimage,dataimage=readoneimage_full(filename)

    xminfit2d, xmaxfit2d, yminfit2d, ymaxfit2d = (1, framedim[1], 1, framedim[0])

    # warning i corresponds to y
    # j corresponds to x
    # change nom xminf2d => xminfit2d pour coherence avec le reste

    # imin,imax,jmin,jmax=2048-ymaxfit2d,2048-yminfit2d,xminfit2d,xmaxfit2d
    imin, imax, jmin, jmax = (
        framedim[0] - ymaxfit2d,
        framedim[0] - yminfit2d,
        xminfit2d - 1,
        xmaxfit2d - 1,
    )

    # dataimage_ROI=dataimage[imin:imax,jmin:jmax]# array index   i,j
    # # fit2d index:  X=j Y=2048-i

    dataimage_ROI = Data

    print("searching local maxima for non saturated consecutive pixels")

    peak = localmaxima(dataimage_ROI, nb_of_shift, diags=1)

    print("Done...!")
    print(peak)
    # print "execution time : %f  secondes"%(ttt.time()-time_0)

    intensity_localmaxima = dataimage_ROI[peak]
    # print intensity_localmaxima

    # SATURATION handling ------------------------------
    # if the top of the local maximum has at least two pixels with the same intensity, this maximum is not detected
    # this generally the case for saturated peaks
    # saturation value : saturation above which we will take into account the pixel
    # this value may be lower than the 2^n bits value to handle unfortunately very flat weak peak with 2 neighbouring pixels
    # Saturation_value = 65535 for mccd
    Size_of_pixelconnection = 20
    print("Saturation value for flat top peak handling", Saturation_value)
    sat_pix = np.where(dataimage_ROI >= Saturation_value)

    if verbose:
        print("positions of saturated pixels \n", sat_pix)
    print("nb of saturated pixels", len(sat_pix[0]))
    sat_pix_mean = None

    # there is at least one peak above or equal to the Saturation_value threshold
    # loop over saturated pixels
    if len(sat_pix[0]) > 0:

        if verbose:
            print("positions of saturated pixels \n", sat_pix)

        if 1:  # use of graph algorithms
            sat_pix = np.column_stack(sat_pix)

            disttable_sat = ssd.pdist(sat_pix, "euclidean")
            sqdistmatrix_sat = ssd.squareform(disttable_sat)
            # building adjencymat

            a, b = np.indices(sqdistmatrix_sat.shape)
            indymat = np.triu(b) + np.tril(a)
            cond2 = np.logical_and(
                sqdistmatrix_sat < Size_of_pixelconnection, sqdistmatrix_sat > 0
            )
            adjencymat = np.where(cond2, indymat, 0)

            # print "before networkx"
            print("networkx version", NX.__version__)
            GGraw = NX.to_networkx_graph(adjencymat, create_using=NX.Graph())
            list_of_cliques = NX.find_cliques(GGraw)
            # print "after networkx"

            # now find average pixel of each clique
            sat_pix_mean = []
            for clique in list_of_cliques:
                ii, jj = np.mean(sat_pix[clique], axis=0)
                sat_pix_mean.append([int(ii), int(jj)])

            sat_pix_mean = np.array(sat_pix_mean)
            print("Mean position of saturated pixels blobs = \n", sat_pix_mean)

        if 0:  # of scipy.ndimage

            df = ndimage.gaussian_filter(dataimage_ROI, 10)

            # histo = np.histogram(df)
            # print "histogram",histo
            # print "maxinten",np.amax(df)
            threshold_for_measurements = (
                np.amax(df) / 10.0
            )  # histo[1][1]# 1000  pour CdTe # 50 pour Ge

            tG = np.where(df > threshold_for_measurements, 1, 0)
            ll, nf = ndimage.label(tG)  # , structure = np.ones((3,3)))
            meanpos = np.array(
                ndimage.measurements.center_of_mass(tG, ll, np.arange(1, nf + 1)),
                dtype=float,
            )
            # meanpos = np.fliplr(meanpos)  # this done later

            # print "meanpos",meanpos

            sat_pix_mean = meanpos
    else:
        print("No pixel saturation")
    # SATURATION handling -(End) --------------------------------------------------------

    # x,y from localmaxima is a matter of convention

    peaki = peak[0] + imin
    peakj = peak[1] + jmin

    # building an array of hot pixels (2 coordinates)
    Yarray = peakj
    Xarray = peaki
    peaklist = np.array([Xarray, Yarray]).T

    # print peaklistfit2D
    # print peaklist[100:150]
    print("{} local maxima have been found".format(len(peaklist)))

    # probing background and maximal intensity in boxsize
    #
    ptp_boxsize = boxsize_for_probing_minimal_value_background

    tabptp = []
    for k in list(range(len(peaklist))):
        tabptp.append(
            minmax(dataimage_ROI, peaklist[k], ptp_boxsize, framedim=framedim)
        )

    ar_ptp = np.array(tabptp)
    # ar_amp = np.subtract(ar_ptp[:,1],ar_ptp[:,0])
    ar_amp = np.subtract(intensity_localmaxima, ar_ptp[:, 0])
    amp_rank = np.argsort(ar_amp)[::-1]

    peaklist_sorted = peaklist[amp_rank]
    # ptp_sorted = ar_ptp[amp_rank]
    amp_sorted = ar_amp[amp_rank]
    # thresholding on peak-to-peak amplitude
    threshold_amp = IntensityThreshold

    cond = np.where(amp_sorted > threshold_amp)
    th_peaklist = peaklist_sorted[cond]
    th_ar_amp = amp_sorted[cond]

    print(
        "{} local maxima found after thresholding above {} amplitude above local background".format(
            len(th_ar_amp), threshold_amp
        )
    )

    # remove duplicates (close points), the most intense pixel is kept
    # minimum distance between hot pixel
    # it corresponds both to distance between peaks and peak size ...
    pixeldistance = pixeldistance_remove_duplicates

    purged_pklist, index_todelete = GT.purgeClosePoints2(th_peaklist, pixeldistance)

    purged_amp = np.delete(th_ar_amp, index_todelete)
    print(
        "{} local maxima found after removing duplicates (minimum intermaxima distance = {})".format(
            len(purged_amp), pixeldistance
        )
    )

    # print "execution time : %f  secondes"%( ttt.time() - time_0)

    # merging different kind of peaks
    if sat_pix_mean is not None:
        print("Merging saturated and normal peaks")
        print("number of saturated peaks : ", np.shape(sat_pix_mean)[0])
        purged_pklist = np.vstack((sat_pix_mean, purged_pklist))

    if 0:  # check if there are still close hot pixels
        disttable_c = ssd.pdist(purged_pklist, "euclidean")
        maxdistance_c = np.amax(disttable_c)
        sqdistmatrix_c = ssd.squareform(disttable_c)
        distmatrix_c = sqdistmatrix_c + np.eye(sqdistmatrix_c.shape[0]) * maxdistance_c

        print(
            "close hotpixels", np.where(distmatrix_c < pixeldistance)
        )  # must be (array([], dtype=int64), array([], dtype=int64))

    # print "purged_pklist", purged_pklist
    print("shape(purged_pklist)", np.shape(purged_pklist))
    npeaks = np.shape(purged_pklist)[0]
    Ipixmax = np.zeros(npeaks, dtype=int)
    # print np.shape(Data)

    for i in list(range(npeaks)):
        Ipixmax[i] = Data[purged_pklist[i, 0], purged_pklist[i, 1]]
        # print "Ipixmax = ", Ipixmax

    return np.fliplr(purged_pklist), Ipixmax


def LocalMaxima_from_thresholdarray(Data, IntensityThreshold=400):
    """
    return center of mass of each blobs composes by pixels above 
    IntensityThreshold

    !warning!: center of mass of blob where all intensities are set to 1
    """
    thrData_for_label = np.where(Data > IntensityThreshold, 1, 0)

    #     thrData = np.where(Data > IntensityThreshold, Data, 0)

    #    star = array([[0,1,0],[1,1,1],[0,1,0]])
    # ll, nf = ndimage.label(thrData_for_label, structure=np.ones((3,3)))
    # ll, nf = ndimage.label(thrData_for_label, structure=star)
    ll, nf = ndimage.label(thrData_for_label)

    #     print "nb of blobs in LocalMaxima_from_thresholdarray()", nf

    if nf == 0:
        return None
    # meanpos = np.zeros((nf,2))
    # for k in range(nf):
    # meanpos[k] = np.mean(np.where((ll == k),axis=1)

    # ndimage.find_objects(ll)
    #     meanpos = \
    #     np.array(ndimage.measurements.center_of_mass(Data,
    #                                                ll,
    #                                                np.arange(1, nf + 1)),
    #                                                 dtype=float)

    meanpos = np.array(
        ndimage.measurements.maximum_position(Data, ll, np.arange(1, nf + 1)),
        dtype=float,
    )

    if len(np.shape(meanpos)) > 1:
        meanpos = np.fliplr(meanpos)
    else:
        meanpos = np.roll(meanpos, 1)

    return meanpos


def shiftarrays(Data_array, n, dimensions=1):
    """
    1D
    returns 3 arrays corresponding to shifted arrays by n in two directions and original one
    2D
    returns 5 arrays corresponding to shifted arrays by n in two directions and original one

    these arrays are ready for comparison with eg np.greater
    """
    if n > 0:
        if dimensions == 2:
            shift_zero = Data_array[n:-n, n:-n]

            shift_left = Data_array[: -2 * n, n:-n]
            shift_right = Data_array[2 * n :, n:-n]
            shift_up = Data_array[n:-n, : -2 * n]
            shift_down = Data_array[n:-n, 2 * n :]

            return shift_zero, shift_left, shift_right, shift_up, shift_down

        if dimensions == 1:
            shift_zero = Data_array[n:-n]

            shift_left = Data_array[: -2 * n]
            shift_right = Data_array[2 * n :]

            return shift_zero, shift_left, shift_right


def shiftarrays_accum(Data_array, n, dimensions=1, diags=0):
    """
    idem than shiftarrays() but with all intermediate shifted arrays
    1D
    returns 3 arrays corresponding to shifted arrays
    by n in two directions and original one
    2D
    returns 5 arrays corresponding to shifted arrays
    by n in two directions and original one

    these arrays are ready for comparison with eg np.greater

    Data_array must have shape (slowdim,fastdim) so that
    slowdim-2*n>=1 and fastdim-2*n>=1
    (ie central array with zero shift has some elements)

    TODO: replace append by a pre allocated array
    """
    if n <= 0:
        raise ValueError("shift value must be positive")

    if dimensions == 2:
        if diags:
            shift_zero = Data_array[n:-n, n:-n]

            allleft = []
            allright = []
            allup = []
            alldown = []
            alldiagleftdown = []  # diag "y=x"
            alldiagrightup = []  # diag "y=x"
            alldiagrightdown = []  #  diah "y=-x"
            alldiagleftup = []  #  diah "y=-x"

            for k in np.arange(1, n + 1)[::-1]:

                allleft.append(Data_array[n - k : -(n + k), n:-n])
                alldown.append(Data_array[n:-n, n - k : -(n + k)])
                alldiagrightdown.append(Data_array[n - k : -(n + k), n - k : -(n + k)])

                if (n - k) != 0:
                    allright.append(Data_array[k + n : -(n - k), n:-n])
                    allup.append(Data_array[n:-n, k + n : -(n - k)])
                    alldiagleftdown.append(
                        Data_array[k + n : -(n - k), n - k : -(n + k)]
                    )
                    alldiagleftup.append(Data_array[k + n : -(n - k), k + n : -(n - k)])
                    alldiagrightup.append(
                        Data_array[n - k : -(n + k), k + n : -(n - k)]
                    )

                else:  # correct python array slicing at the end :   a[n:0]  would mean a[n:]

                    allright.append(Data_array[k + n :, n:-n])
                    allup.append(Data_array[n:-n, k + n :])
                    alldiagleftdown.append(Data_array[k + n :, n - k : -(n + k)])
                    alldiagleftup.append(Data_array[k + n :, k + n :])
                    alldiagrightup.append(Data_array[n - k : -(n + k), k + n :])

            return (
                shift_zero,
                allleft,
                allright,
                alldown,
                allup,
                alldiagleftdown,
                alldiagrightup,
                alldiagrightdown,
                alldiagleftup,
            )

        else:
            shift_zero = Data_array[n:-n, n:-n]

            allleft = []
            allright = []
            allup = []
            alldown = []

            allleft.append(Data_array[: -2 * n, n:-n])
            alldown.append(Data_array[n:-n, : -2 * n])

            for k in np.arange(1, n)[::-1]:
                allleft.append(Data_array[n - k : -(n + k), n:-n])
                allright.append(Data_array[k + n : -(n - k), n:-n])
                alldown.append(Data_array[n:-n, n - k : -(n + k)])
                allup.append(Data_array[n:-n, k + n : -(n - k)])

            allright.append(Data_array[2 * n :, n:-n])
            allup.append(Data_array[n:-n, 2 * n :])

            return shift_zero, allleft, allright, alldown, allup

    elif dimensions == 1:
        shift_zero = Data_array[n:-n]
        allleft = []
        allright = []
        allleft.append(Data_array[: -2 * n])
        for k in np.arange(1, n)[::-1]:
            allright.append(Data_array[k + n : -(n - k)])
            allleft.append(Data_array[n - k : -(n + k)])
        allright.append(Data_array[2 * n :])

        return shift_zero, allleft, allright


def localmaxima(DataArray, n, diags=1, verbose=0):
    """
    from DataArray 2D  returns (array([i1,i2,...,ip]),array([j1,j2,...,jp]))
    of indices where pixels value is higher in two direction up to n pixels

    this tuple can be easily used after in the following manner:
    DataArray[tupleresult] is an array of the intensity of the hottest pixels in array

    in similar way with only four cardinal directions neighbouring (found in the web):
    import numpy as N
    def local_minima(array2d):
        return ((array2d <= np.roll(array2d,  1, 0)) &
                (array2d <= np.roll(array2d, -1, 0)) &
                (array2d <= np.roll(array2d,  1, 1)) &
                (array2d <= np.roll(array2d, -1, 1)))

    WARNING: flat top peak are not detected !!
    """
    dim = len(np.shape(DataArray))

    if diags:
        c, alll, allr, alld, allu, diag11, diag12, diag21, diag22 = shiftarrays_accum(
            DataArray, n, dimensions=dim, diags=diags
        )
        flag = np.greater(c, alll[0])
        for elem in alll[1:] + allr + alld + allu + diag11 + diag12 + diag21 + diag22:
            flag = flag * np.greater(c, elem)
    else:
        c, alll, allr, alld, allu = shiftarrays_accum(
            DataArray, n, dimensions=dim, diags=diags
        )
        flag = np.greater(c, alll[0])
        for elem in alll[1:] + allr + alld + allu:
            flag = flag * np.greater(c, elem)

    peaklist = np.nonzero(flag)  # in c frame index

    if verbose:
        print("value local max", c[peaklist])
        print("value from original array ", DataArray[tuple(np.array(peaklist) + n)])
        print(
            "positions of local maxima in original frame index",
            tuple(np.array(peaklist) + n),
        )

    # first slow index array , then second fast index array
    return tuple(np.array(peaklist) + n)


def Find_optimal_thresholdconvolveValue(
    filename, IntensityThreshold, CCDLabel="PRINCETON"
):
    """
    give the lowest value for thresholdconvolve according to IntensityThreshold
    in order not to miss any blob detection in the first filtering
    (by convolution)
    that would be accepted in the second filtering
    (thresholding by IntensityThreshold)
    """
    # image = scipy.misc.lena().astype(float32)
    #    d = readoneimage(filename).reshape((2048,2048))

    res = []
    for tc in (0, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000):
        # tstart = ttt.time()
        Isorted, fitpeak, localpeak, nbrawblobs = PeakSearch(
            filename,
            CCDLabel=CCDLabel,
            PixelNearRadius=20,
            removeedge=2,
            IntensityThreshold=IntensityThreshold,
            thresholdConvolve=tc,
            boxsize=15,
            position_definition=1,
            verbose=0,
            fit_peaks_gaussian=0,
            xtol=0.001,
            return_histo=2,
        )

        # res.append([tc,IntensityThreshold, nbrawblobs,len(fitpeak), ttt.time()-tstart])
        res.append([tc, IntensityThreshold, nbrawblobs, len(fitpeak)])

    Res = np.array(res)
    cond = (
        Res[:, 2] > Res[:, 3]
    )  # if False  ==> threshold after convolution is too high, some blob may have been rejected before intensity thresholding
    indices_False = np.where(cond == False)[0]
    optim_value = 0
    if len(indices_False) > 1:

        if indices_False[0] != 0:
            optim_value = Res[indices_False[0] - 1, 0]

    print("optim value for thresholdConvolve", optim_value)

    return optim_value, Res


def writepeaklist(
    tabpeaks, output_filename, outputfolder=None, comments=None, initialfilename=None
):
    """
    write peaks properties and comments in file with extension .dat added
    """
    outputfilefullpath = IOLT.writefile_Peaklist(
        output_filename,
        tabpeaks,
        dirname=outputfolder,
        overwrite=1,
        initialfilename=initialfilename,
        comments=comments,
    )

    return outputfilefullpath


def fitoneimage_manypeaks(
    filename,
    peaklist,
    boxsize,
    stackimageindex=-1,
    CCDLabel="PRINCETON",
    dirname=None,
    position_start="max",
    type_of_function="gaussian",
    guessed_peaksize=(1.0, 1.0),
    xtol=0.001,
    FitPixelDev=2.0,
    Ipixmax=None,
    MaxIntensity=100000000000,
    MinIntensity=0,
    PeakSizeRange=(0, 200),
    verbose=0,
    position_definition=1,
    NumberMaxofFits=500,
    ComputeIpixmax=False,
    use_data_corrected=None,
    reject_negative_baseline=True,
    purgeDuplicates=True,
):

    """ 
    fit multiple ROI data to get peaks position in a single image

    Ipixmax  :  highest intensity above background in every ROI centered on element of peaklist

    use_data_corrected   :  enter data instead of reading data from file
                        must be a tuple of 3 elements:
                        fulldata, framedim, fliprot
                        where fulldata is an ndarray

    purgeDuplicates    : True   remove duplicates that are close within pixel distance of 'boxsize' and keep the most intense peak
    
    use_data_corrected   :  enter data instead of reading data from file
                        must be a tuple of 3 elements:
                        fulldata, framedim, fliprot
                        where fulldata  ndarray
    
    """

    #     print 'Ipixmax in fitoneimage_manypeaks', Ipixmax
    if len(peaklist) >= NumberMaxofFits:
        print("TOO MUCH peaks to fitdata.")
        print("(in fitoneimage_manypeaks) It may stuck the computer.")
        print(
            "Try to reduce the number of Local Maxima or reduce NumberMaxofFits in fitoneimage_manypeaks()"
        )
        return

    if dirname is not None:
        filename = os.path.join(dirname, filename)

    start_sigma1, start_sigma2 = guessed_peaksize

    tstart = ttt.time()

    ResFit = readoneimage_multiROIfit(
        filename,
        peaklist,
        boxsize,
        stackimageindex,
        CCDLabel=CCDLabel,
        baseline="auto",  # min in ROI box
        startangles=0.0,
        start_sigma1=start_sigma1,
        start_sigma2=start_sigma2,
        position_start=position_start,  # 'centers' or 'max'
        showfitresults=0,
        offsetposition=0,  # offset are applied after fit
        fitfunc=type_of_function,
        xtol=xtol,
        addImax=ComputeIpixmax,
        use_data_corrected=use_data_corrected,
    )

    print(
        "fitting time for {} peaks is : {:.4f}".format(
            len(peaklist), ttt.time() - tstart
        )
    )
    print("nb of results: ", len(ResFit[0]))

    if ComputeIpixmax == True:
        params, cov, info, message, baseline, Ipixmax = ResFit
    else:
        params, cov, info, message, baseline = ResFit

    par = np.array(params)

    #    print "par in fitoneimage_manypeaks", par

    if par == []:
        print("no fitted peaks")
        return

    peak_bkg = par[:, 0]
    peak_I = par[:, 1]
    peak_X = par[:, 2]
    peak_Y = par[:, 3]
    peak_fwaxmaj = par[:, 4]
    peak_fwaxmin = par[:, 5]
    peak_inclination = par[:, 6] % 360

    # pixel deviations from guessed initial position before fitting
    Xdev = peak_X - peaklist[:, 0]
    Ydev = peak_Y - peaklist[:, 1]

    #     print "peak_X",peak_X
    #     print "peaklist[:, 0]",peaklist[:, 0]
    #     print 'Xdev', Xdev
    #     print "Ydev", Ydev

    # --- --- PEAKS REJECTION -------------------------------
    # print "peaklist[:20]",peaklist[:]
    # number of iteration screening
    to_reject = []
    k = 0
    for inf in info:
        if inf["nfev"] > 1550:
            if verbose:
                print("k= {}   too much iteration".format(k))
            to_reject.append(k)
        k += 1

    if CCDLabel == "FRELONID15_corrected":
        reject_negative_baseline = False
    # negative intensity rejection
    if reject_negative_baseline:
        to_reject2 = np.where((peak_bkg - baseline) < 0)[0]
    else:
        to_reject2 = []

    # too far found peak rejection
    to_reject3 = np.where(np.sqrt(Xdev ** 2 + Ydev ** 2) > FitPixelDev)[0]
    #     print 'to_reject3', to_reject3

    #     print "peak_I",peak_I

    # too intense compared to given threshold (saturation)
    to_reject4 = np.where(peak_I >= MaxIntensity)[0]

    # too weak compared to given threshold
    to_reject5 = np.where(peak_I <= MinIntensity)[0]

    maxpeaksize = np.maximum(peak_fwaxmaj, peak_fwaxmin)

    # too small peak compared to given threshold
    to_reject6 = np.where(maxpeaksize <= PeakSizeRange[0])[0]

    # too large peak compared to given threshold
    to_reject7 = np.where(maxpeaksize >= PeakSizeRange[1])[0]

    if verbose:
        print("to_reject", type(to_reject))
        print("to_reject ...(len)", len(to_reject))
        print(np.take(peaklist, to_reject, axis=0))
        print("to_reject2 ...(len)", len(to_reject2))
        print(np.take(peaklist, to_reject2, axis=0))
        print(np.take(peaklist, to_reject3, axis=0))

    print(
        "After fitting, {}/{} peaks have been rejected\n due to (final - initial position)> FitPixelDev = {}".format(
            len(to_reject3), len(peaklist), FitPixelDev
        )
    )
    print(
        "{} spots have been rejected\n due to negative baseline".format(len(to_reject2))
    )
    print(
        "{} spots have been rejected\n due to much intensity ".format(len(to_reject4))
    )
    print(
        "{} spots have been rejected\n due to weak intensity ".format(len(to_reject5))
    )
    print(
        "{} spots have been rejected\n due to small peak size".format(len(to_reject6))
    )
    print(
        "{} spots have been rejected\n due to large peak size".format(len(to_reject7))
    )

    # spots indices to reject
    ToR = (
        set(to_reject)
        | set(to_reject2)
        | set(to_reject3)
        | set(to_reject4)
        | set(to_reject5)
        | set(to_reject6)
        | set(to_reject7)
    )  # to reject

    # spot indices to take
    ToTake = set(np.arange(len(peaklist))) - ToR

    print("ToTake", ToTake)
    print("len(ToTake)", len(ToTake))
    if len(ToTake) < 1:
        return None, par, peaklist

    #     print "Ipixmax",Ipixmax
    if Ipixmax is None:
        Ipixmax = peak_I
    else:
        # ask for maximum intensity in ROI, see
        pass

    # all peaks list building
    tabpeak = np.array(
        [
            peak_X,
            peak_Y,
            peak_I,
            peak_fwaxmaj,
            peak_fwaxmin,
            peak_inclination,
            Xdev,
            Ydev,
            peak_bkg,
            Ipixmax,
        ]
    ).T

    # print("Results of all fits in tabpeak", tabpeak)

    tabpeak = np.take(tabpeak, list(ToTake), axis=0)

    #     print "tabpeak.shape",tabpeak.shape
    if len(tabpeak.shape) > 1:  # several peaks
        intense_rank = np.argsort(tabpeak[:, 2])[
            ::-1
        ]  # sort by decreasing intensity-bkg
        #    print "intense_rank", intense_rank

        tabIsorted = tabpeak[intense_rank]
    #         print "tabIsorted.shape case 1",tabIsorted.shape

    else:  # single peak
        #         tabIsorted = np.array(tabpeak)[:,0]
        print("tabIsorted.shape case 2", tabIsorted.shape)

    if position_definition == 1:  # XMAS offset
        tabIsorted[:, :2] = tabIsorted[:, :2] + np.array([1, 1])

    if verbose:
        print("\n\nIntensity sorted\n\n")
        print(tabIsorted[:10])
        print("X,Y", tabIsorted[:10, :2])

    print("\n{} fitted peak(s)\n".format(len(tabIsorted)))

    if purgeDuplicates and len(tabIsorted) > 2:
        print("Removing duplicates from fit")

        # remove duplicates (close points), the most intense pixel is kept
        # minimum distance fit solutions
        pixeldistance = boxsize

        tabXY, index_todelete = GT.purgeClosePoints2(tabIsorted[:, :2], pixeldistance)

        #         print tabXY
        #         print index_todelete

        tabIsorted = np.delete(tabIsorted, index_todelete, axis=0)

        print(
            "\n{} peaks found after removing duplicates (minimum intermaxima distance = {})".format(
                len(tabIsorted), pixeldistance
            )
        )

    return tabIsorted, par, peaklist


def PeakSearch(
    filename,
    stackimageindex=-1,
    CCDLabel="PRINCETON",
    center=None,
    boxsizeROI=(200, 200),  # use only if center != None
    PixelNearRadius=5,
    removeedge=2,
    IntensityThreshold=400,
    thresholdConvolve=200,
    paramsHat=(4, 5, 2),
    boxsize=15,
    verbose=0,
    position_definition=1,
    local_maxima_search_method=1,
    peakposition_definition="max",
    fit_peaks_gaussian=1,
    xtol=0.00001,
    return_histo=1,
    FitPixelDev=25,  # to_reject3 parameter
    write_execution_time=1,
    Saturation_value=65535,  # to be merged in CCDLabel
    Saturation_value_flatpeak=65535,
    MinIntensity=0,
    PeakSizeRange=(0, 200),
    oldversion=False,  # to be removed
    Data_for_localMaxima=None,
    Fit_with_Data_for_localMaxima=False,
    Remove_BlackListedPeaks_fromfile=None,
    maxPixelDistanceRejection=15.0,
    maxDistanceRejection=15,
    NumberMaxofFits=5000,
    reject_negative_baseline=True,
    formulaexpression="A-1.1*B",
):
    """
    Find local intensity maxima as starting position for fittinng and return peaklist.
    
    Parameters
    -------------------
    
    filename : string
               full path to image data file
               
    stackimageindex : integer
                index corresponding to the position of image data on a stacked images file
                if -1  means single image data w/o stacking

    CCDLabel : string
               label for CCD 2D detector used to read the image data file see dict_LaueTools.py

    center : #TODO: to be removed: position of the ROI center in CCD frame

    boxsizeROI : dimensions of the ROI to crop the data array
                    only used if center != None

    boxsize : half length of the selected ROI array centered on each peak:
                for fitting a peak
                for estimating the background around a peak
                for shifting array in second method of local maxima search (shifted arrays)

    IntensityThreshold : integer
                         pixel intensity level above which potential peaks are kept for fitting position procedure
                        for local maxima method 0 and 1, this level is relative to zero intensity
                        for local maxima method 2, this level is relative to lowest intensity in the ROI (local background)
                        start with high value
                        If too high, few peaks are found (only the most important)
                        If too low, too many local maxima are found leading to time consuming fitting procedure

    thresholdConvolve : integer
                        pixel intensity level in convolved image above which potential peaks are kept for fitting position procedure
                        This threshold step on convolved image is applied prior to the local threshold step with IntensityThreshold on initial image (with respect to the local background)

    paramsHat :  mexican hat kernel parameters (see :func:`LocalMaxima_ndimage`)

    PixelNearRadius: integer
                     pixel distance between two regions considered as peaks
                    start rather with a large value.
                    If too low, there are very much peaks duplicates and
                    this is very time consuming

    local_maxima_search_method : integer
                                 Select method for find the local maxima, each of them will fitted
                            : 0   extract all pixel above intensity threshold
                            : 1   find pixels are highest than their neighbours in horizontal, vertical
                                    and diagonal direction (up to a given pixel distance)
                            : 2   find local hot pixels which after numerical convolution give high intensity
                                above threshold (thresholdConvolve)
                                then threshold (IntensityThreshold) on raw data

    peakposition_definition    : 'max' or 'center'  for local_maxima_search_method == 2
                                to assign to the blob position its hottest pixel position
                                                            or its center (no weight)

    Saturation_value_flatpeak        :  saturation value of detector for local maxima search method 1

    Remove_BlackListedPeaks_fromfile    : None or full file path to a peaklist file containing peaks
                                            that will be deleted in peak list resulting from
                                            the local maxima search procedure (prior to peak refinement)
                                            
    maxPixelDistanceRejection   : maximum distance between black listed peaks and current peaks
                                    (found by peak search) to be rejected

    NumberMaxofFits            : highest acceptable number of local maxima peak to be refined with a 2D modelPeakSearch

    fit_peaks_gaussian      :    0  no position and shape refinement procedure performed from local maxima (or blob) results
                            :    1  2D gaussian peak refinement 
                            :    2  2D lorentzian peak refinement 

    xtol  : relative error on solution (x vector)  see args for leastsq in scipy.optimize
    FitPixelDev            :  largest pixel distance between initial (from local maxima search) and refined peak position  

    position_definition: due to various conventional habits when reading array, add some offset to fitdata XMAS or fit2d peak search values
                         = 0    no offset (python numpy convention)
                         = 1   XMAS offset
                         = 2   fit2d offset

    return_histo        : 0   3 output elements
                        : 1   4 elemts, last one is histogram of data
                        : 2   4 elemts, last one is the nb of raw blob found after convolution and threshold

    Data_for_localMaxima   :  object to be used only for initial step of finding local maxima (blobs) search
                                (and not necessarly for peaks fitting procedure):
                              -  ndarray     = array data
                              - 'auto_background'  = calculate and remove background computed from image data itself (read in file 'filename')
                              - path to image file (string)  = B image to be used in a mathematical operation with Ato current image

    Fit_with_Data_for_localMaxima    : use 'Data_for_localMaxima' object as image when refining peaks position and shape
                                       with initial peak position guess from local maxima search

    formulaexpression    : string containing A (raw data array image) and B (other data array image)
                            expressing mathematical operation,e.g:
                            'A-3.2*B+10000'
                            for simple background substraction (with B as background data):
                            'A-B' or 'A-alpha*B' with alpha > 1.

    reject_negative_baseline        :  True  reject refined peak result if intensity baseline (local background) is negative
                                        (2D model is maybe not suitable)

    returns:

    peak list sorted by decreasing (integrated intensity - fitted bkg)
    peak_X,peak_Y,peak_I,peak_fwaxmaj,peak_fwaxmin,peak_inclination,Xdev,Ydev,peak_bkg

    for fit_peaks_gaussian == 0 (no fitdata) and local_maxima_search_method==2 (convolution)
        if peakposition_definition ='max' then X,Y,I are from the hottest pixels
        if peakposition_definition ='center' then X,Y are blob center and I the hottest blob pixel

    nb of output elements depends on 'return_histo' argument
    """

    if return_histo in (0, 1):
        return_nb_raw_blobs = 0
    if return_histo in (2,):
        return_nb_raw_blobs = 1
    if write_execution_time:
        t0 = ttt.time()

    # user input its own shaped Data array
    if isinstance(Data_for_localMaxima, np.ndarray):
        print("Using 'Data_for_localMaxima' ndarray for finding local maxima")
        Data = Data_for_localMaxima

        #         print "min, max intensity", np.amin(Data), np.amax(Data)
        # TODO to test with VHR
        framedim = Data.shape
        ttread = ttt.time()

    # Data are read from image file
    elif isinstance(Data_for_localMaxima, str) or Data_for_localMaxima is None:

        Data, framedim, fliprot = readCCDimage(
            filename,
            stackimageindex=stackimageindex,
            CCDLabel=CCDLabel,
            dirname=None,
            verbose=1,
        )
        print("image from filename {} read!".format(filename))

        # peak search in a particular region of image
        if center is not None:

            #        imin, imax, jmin, jmax = getindices2cropArray(center, boxsizeROI, framedim)
            #        Data = Data[imin: imax, jmin: jmax]

            framedim = (framedim[1], framedim[0])
            imin, imax, jmin, jmax = getindices2cropArray(center, boxsizeROI, framedim)
            Data = Data[jmin:jmax, imin:imax]

        if write_execution_time:
            dtread = ttt.time() - t0
            ttread = ttt.time()
            print("Read Image. Execution time : {:.3f} seconds".format(dtread))

        if return_histo:
            # from histogram, deduces
            min_intensity = max(np.amin(Data), 1)  # in case of 16 integer
            max_intensity = min(np.amax(Data), Saturation_value)
            print("min_intensity", min_intensity)
            print("max_intensity", max_intensity)
            histo = np.histogram(
                Data,
                bins=np.logspace(
                    np.log10(min_intensity), np.log10(max_intensity), num=30
                ),
            )

    if isinstance(Data_for_localMaxima, str):
        print(
            "Using Data_for_localMaxima for local maxima search: --->",
            Data_for_localMaxima,
        )
        # compute and remove background from this image
        if Data_for_localMaxima == "auto_background":
            print("computing background from current image ", filename)
            backgroundimage = compute_autobackground_image(Data, boxsizefilter=10)
            # basic substraction
            usemask = True
        # path to a background image file
        else:
            if stackimageindex == -1:
                raise ValueError("Use stacked images as background is not implement")
            path_to_bkgfile = Data_for_localMaxima
            print("Using image file {} as background".format(path_to_bkgfile))
            try:
                backgroundimage, framedim_bkg, fliprot_bkg = readCCDimage(
                    path_to_bkgfile, CCDLabel=CCDLabel
                )
            except IOError:
                raise ValueError(
                    "{} does not seem to be a path file ".format(path_to_bkgfile)
                )

            usemask = False

        print("Removing background for local maxima search")
        Data = computefilteredimage(
            Data,
            backgroundimage,
            CCDLabel,
            usemask=usemask,
            formulaexpression=formulaexpression,
        )

    print("Data.shape for local maxima", Data.shape)

    # --- PRE SELECTION OF HOT PIXELS as STARTING POINTS FOR FITTING ---------
    # first method ---------- "Basic Intensity Threshold"
    if local_maxima_search_method in (0, "0"):

        print("Using simple intensity thresholding to detect local maxima (method 1/3)")
        peaklist = LocalMaxima_from_thresholdarray(
            Data, IntensityThreshold=IntensityThreshold
        )

        if peaklist is not None:
            print("len(peaklist)", len(peaklist))

            Ipixmax = np.ones(len(peaklist)) * IntensityThreshold

            ComputeIpixmax = False

    # second method ----------- "Local Maxima in a box by shift array method"
    if local_maxima_search_method in (1, "1"):
        # flat top peaks (e.g. saturation) are NOT well detected
        print("Using shift arrays to detect local maxima (method 2/3)")
        peaklist, Ipixmax = LocalMaxima_ShiftArrays(
            Data,
            framedim=framedim,
            IntensityThreshold=IntensityThreshold,
            Saturation_value=Saturation_value_flatpeak,
            boxsize_for_probing_minimal_value_background=boxsize,  # 30
            pixeldistance_remove_duplicates=PixelNearRadius,  # 25
            nb_of_shift=boxsize,
        )  # 25

        ComputeIpixmax = True

    # third method: ------------ "Convolution by a gaussian kernel"
    if local_maxima_search_method in (2, "2"):

        print("Using mexican hat convolution to detect local maxima (method 3/3)")

        peakValConvolve, boxsizeConvolve, central_radiusConvolve = paramsHat

        Candidates = LocalMaxima_KernelConvolution(
            Data,
            framedim=framedim,
            peakValConvolve=peakValConvolve,
            boxsizeConvolve=boxsizeConvolve,
            central_radiusConvolve=central_radiusConvolve,
            thresholdConvolve=thresholdConvolve,  # 600 for CdTe
            connectivity=1,
            IntensityThreshold=IntensityThreshold,
            boxsize_for_probing_minimal_value_background=PixelNearRadius,
            return_nb_raw_blobs=return_nb_raw_blobs,
            peakposition_definition=peakposition_definition,
        )

        if Candidates is None:
            print("No local maxima found, change peak search parameters !!!")
            return None

        if return_nb_raw_blobs == 1:
            peaklist, Ipixmax, nbrawblobs = Candidates
        else:
            peaklist, Ipixmax = Candidates

        #         print "len(peaklist)", peaklist.shape
        #         print "Ipixmax", Ipixmax.shape
        ComputeIpixmax = True
    #         print "Ipixmax after convolution method", Ipixmax
    # -------------------------------------------------------------
    # --- END of blobs search methods calls

    if (
        peaklist is None
        or peaklist is []
        or peaklist is np.array([])
        or (len(peaklist) == 0)
    ):
        print("No local maxima found, change peak search parameters !!!")
        return None
    # pixel origin correction due to ROI croping
    if center is not None:
        x1, y1 = center  # TODO: to ne checked !!
        peaklist = peaklist + np.array([x1, y1])

    if write_execution_time:
        dtsearch = ttt.time() - float(ttread)

        print("Local maxima search. Execution time : {:.3f} seconds".format(dtsearch))

    # removing some duplicates ------------
    if len(peaklist) >= 2:
        nb_peaks_before = len(peaklist)
        #         print "%d peaks in peaklist before purge" % nb_peaks_before
        #         print 'peaklist',in peaklist before purge

        if len(peaklist) >= NumberMaxofFits:
            print("TOO MUCH peaks to handle.")
            print("(in PeakSearch) It may stuck the computer.")
            print(
                "Try to reduce the number of Local Maxima or\n reduce NumberMaxofFits in PeakSearch()"
            )
            return None

        Xpeaklist, Ypeaklist, tokeep = GT.removeClosePoints(
            peaklist[:, 0], peaklist[:, 1], dist_tolerance=2
        )

        peaklist = np.array([Xpeaklist, Ypeaklist]).T
        Ipixmax = np.take(Ipixmax, tokeep)

        print(
            "Keep {} from {} initial peaks (ready for peak positions and shape fitting)".format(
                len(peaklist), nb_peaks_before
            )
        )
    # -----------------------------------------------

    # remove black listed peaks option
    if Remove_BlackListedPeaks_fromfile is not None:

        data_peak_blacklisted = IOLT.read_Peaklist(
            Remove_BlackListedPeaks_fromfile, dirname=None
        )

        #         print "data_peak_blacklisted", data_peak_blacklisted

        if len(peaklist) > 1 and len(data_peak_blacklisted) > 1:

            XY_blacklisted = data_peak_blacklisted[:, :2].T

            X, Y = peaklist[:, :2].T

            (peakX, peakY, tokeep) = GT.removeClosePoints_two_sets(
                [X, Y],
                XY_blacklisted,
                dist_tolerance=maxPixelDistanceRejection,
                verbose=0,
            )

            npeak_before = len(X)
            npeak_after = len(peakX)

            print(
                "\n Removed {} (over {}) peaks belonging to the blacklist {}\n".format(
                    npeak_before - npeak_after,
                    npeak_before,
                    Remove_BlackListedPeaks_fromfile,
                )
            )

            #             print "peaklist before", peaklist
            #             print "peakX, peakY blacklisted", XY_blacklisted

            peaklist = np.take(peaklist, tokeep, axis=0)
            Ipixmax = Ipixmax[tokeep]

    #             print "peaklist after blacklist removal", peaklist

    # ---- ----------- no FITTING ----------------------------

    # NO FIT  and return raw list of local maxima
    if fit_peaks_gaussian == 0:

        if position_definition == 1:  # XMAS like offset
            peaklist[:, :2] = peaklist[:, :2] + np.array([1, 1])

        if position_definition == 2:  # fit2D offset
            peaklist[:, 0] = peaklist[:, 0] + 0.5
            peaklist[:, 1] = framedim[0] - peaklist[:, 1] + 0.5

        if verbose:
            print("{} local maxima found".format(len(peaklist)))
            print("20 first peaks", peaklist[:20])

        # tabpeak mimics the array built after fitting procedures
        tabpeak = np.zeros((len(peaklist[:, 0]), 10))
        tabpeak[:, 0] = peaklist[:, 0]
        tabpeak[:, 1] = peaklist[:, 1]
        tabpeak[:, 2] = Ipixmax
        # return tabpeak, peaklist, peaklist, peaklist  # no fitdata return raw list of local maxima
        lastelem = peaklist
        if return_nb_raw_blobs == 1:
            lastelem = nbrawblobs

        return tabpeak, peaklist, peaklist, lastelem

    # ----  ---------------FITTING ----------------------------
    # gaussian fitdata
    elif fit_peaks_gaussian == 1:
        type_of_function = "gaussian"

    # lorentzian fitdata
    elif fit_peaks_gaussian == 2:
        type_of_function = "lorentzian"

    else:
        raise ValueError(
            "optional fit_peaks_gaussian value is not understood! Must be 0,1 or 2"
        )

    print("\n*****************")
    print("{} local maxima found".format(len(peaklist)))
    print("\n Fitting of each local maxima\n")

    #    print "framedim", framedim
    #    print "offset", offset
    #    print "formatdata", formatdata
    #    print "fliprot", fliprot

    if center is not None:
        position_start = "centers"
    else:
        position_start = "max"

    # if Data_for_localMaxima will be used for refining peak positions
    if Fit_with_Data_for_localMaxima:
        Data_to_Fit = (Data, framedim, fliprot)
    else:
        Data_to_Fit = None

    return fitoneimage_manypeaks(
        filename,
        peaklist,
        boxsize,
        stackimageindex,
        CCDLabel=CCDLabel,
        dirname=None,
        position_start=position_start,
        type_of_function=type_of_function,
        xtol=xtol,
        FitPixelDev=FitPixelDev,
        Ipixmax=Ipixmax,
        MaxIntensity=Saturation_value,
        MinIntensity=MinIntensity,
        PeakSizeRange=PeakSizeRange,
        verbose=verbose,
        position_definition=position_definition,
        NumberMaxofFits=NumberMaxofFits,
        ComputeIpixmax=ComputeIpixmax,
        use_data_corrected=Data_to_Fit,
        reject_negative_baseline=reject_negative_baseline,
    )


def peaksearch_on_Image(
    filename_in,
    pspfile,
    background_flag="no",
    blacklistpeaklist=None,
    dictPeakSearch={},
    CCDLabel="MARCCD165",
    outputfilename=None,
    KF_DIRECTION="Z>0",
    psdict_Convolve=PEAKSEARCHDICT_Convolve,
):
    """
    Perform a peaksearch by using .psp file

    # not very used ?
    # missing dictPeakSearch   as function argument for formulaexpression  or dict_param??
    """

    dict_param = readPeakSearchConfigFile(pspfile)

    Data_for_localMaxima, formulaexpression = read_background_flag(background_flag)
    blacklistedpeaks_file = read_blacklist_filepath(blacklistpeaklist)

    dict_param["Data_for_localMaxima"] = Data_for_localMaxima
    dict_param["formulaexpression"] = formulaexpression
    dict_param["Remove_BlackListedPeaks_fromfile"] = blacklistedpeaks_file

    # create a data considered as background from an imagefile
    BackgroundImageCreated = False
    flag_for_backgroundremoval = dict_param["Data_for_localMaxima"]

    # flag_for_backgroundremoval is a file path to an imagefile
    # create background data: dataimage_bkg
    if flag_for_backgroundremoval not in ("auto_background", None) and not isinstance(
        flag_for_backgroundremoval, np.ndarray
    ):

        fullpath_backgroundimage = psdict_Convolve["Data_for_localMaxima"]

        #         print "fullpath_backgroundimage ", fullpath_backgroundimage

        dirname_bkg, imagefilename_bkg = os.path.split(fullpath_backgroundimage)

        CCDlabel_bkg = CCDLabel

        (dataimage_bkg, framedim_bkg, fliprot_bkg) = readCCDimage(
            imagefilename_bkg, CCDLabel=CCDlabel_bkg, dirname=dirname_bkg
        )

        BackgroundImageCreated = True

        print(
            "consider dataimagefile {} as background".format(fullpath_backgroundimage)
        )
        (dataimage_raw, framedim_raw, fliprot_raw) = readCCDimage(
            filename_in, CCDLabel=CCDLabel, dirname=None
        )

        if "formulaexpression" in dictPeakSearch:
            formulaexpression = dictPeakSearch["formulaexpression"]
        else:
            raise ValueError(
                'Missing "formulaexpression" to operate on images before peaksearch in peaksearch_fileseries()'
            )

        saturationlevel = DictLT.dict_CCD[CCDLabel][2]

        dataimage_corrected = applyformula_on_images(
            dataimage_raw,
            dataimage_bkg,
            formulaexpression=formulaexpression,
            SaturationLevel=saturationlevel,
            clipintensities=True,
        )

        print("using {} in peaksearch_fileseries".format(formulaexpression))

        # for finding local maxima in image from formula
        psdict_Convolve["Data_for_localMaxima"] = fullpath_backgroundimage

        # for fitting peaks in image from formula
        psdict_Convolve["reject_negative_baseline"] = False
        psdict_Convolve["formulaexpression"] = formulaexpression
        psdict_Convolve["Fit_with_Data_for_localMaxima"] = True

    Res = PeakSearch(
        filename_in,
        CCDLabel=CCDLabel,
        Saturation_value=DictLT.dict_CCD[CCDLabel][2],
        Saturation_value_flatpeak=DictLT.dict_CCD[CCDLabel][2],
        **psdict_Convolve
    )

    if Res in (False, None):
        print("No peak found for image file: ", filename_in)
        return None
    # write file with comments
    Isorted, fitpeak, localpeak = Res[:3]

    if outputfilename:

        params_comments = "Peak Search and Fit parameters\n"

        params_comments += "# {}: {}\n".format("CCDLabel", CCDLabel)

        for key, val in list(psdict_Convolve.items()):
            if not BackgroundImageCreated or key not in ("Data_for_localMaxima",):
                params_comments += "# " + key + " : " + str(val) + "\n"

        if BackgroundImageCreated:
            params_comments += (
                "# "
                + "Data_for_localMaxima"
                + " : {} \n".format(fullpath_backgroundimage)
            )
        # .dat file extension is done in writefile_Peaklist()

        IOLT.writefile_Peaklist(
            "{}".format(outputfilename),
            Isorted,
            overwrite=1,
            initialfilename=filename_in,
            comments=params_comments,
        )

    return Isorted


# -------------------  CONFIG file functions (.psp)
import configparser as CONF

# --- ---- Local maxima and fit parameters
CONVERTKEY_dict = {
    "fit_peaks_gaussian": "fit_peaks_gaussian",
    "position_definition": "position_definition",
    "local_maxima_search_method": "local_maxima_search_method",
    "intensitythreshold": "IntensityThreshold",
    "thresholdconvolve": "thresholdConvolve",
    "boxsize": "boxsize",
    "pixelnearradius": "PixelNearRadius",
    "xtol": "xtol",
    "fitpixeldev": "FitPixelDev",
    "maxpixeldistancerejection": "maxPixelDistanceRejection",
    "maxpeaksize": "MaxPeakSize",
    "minpeaksize": "MinPeakSize",
}

LIST_OPTIONS_PEAKSEARCH = [
    "local_maxima_search_method",
    "IntensityThreshold",
    "thresholdConvolve",
    "boxsize",
    "PixelNearRadius",
    "fit_peaks_gaussian",
    "xtol",
    "FitPixelDev",
    "position_definition",
    "maxPixelDistanceRejection",
    "MinPeakSize",
    "MaxPeakSize",
]

LIST_OPTIONS_TYPE_PEAKSEARCH = [
    "integer flag",
    "integer count",
    "float count",
    "integer pixel",
    "integer pixel",
    "integer flag",
    "float",
    "float pixel",
    "integer flag",
    "float",
    "float",
    "float",
]

LIST_OPTIONS_VALUESPARAMS = [1, 1000, 5000, 15, 10, 1, 0.001, 2.0, 1, 15.0, 0.01, 3.0]

if (
    len(CONVERTKEY_dict)
    != len(LIST_OPTIONS_PEAKSEARCH)
    != LIST_OPTIONS_TYPE_PEAKSEARCH
    != LIST_OPTIONS_VALUESPARAMS
):
    raise ValueError(
        "Lists of parameters for config .psp file do not have the same length (readmccd.py)"
    )


def savePeakSearchConfigFile(dict_param, outputfilename=None):
    # save peaksearch parameter in config file
    config = CONF.RawConfigParser()
    config.add_section("PeakSearch")

    params_comments = "Peak Search and Fit parameters\n"

    for key, val in list(dict_param.items()):
        params_comments += "# " + key + " : " + str(val) + "\n"
        config.set("PeakSearch", key, str(val))

    if outputfilename is None:
        outputfilename = "PeakSearch.psp"

    if not outputfilename.endswith(".psp"):
        if outputfilename.count(".") > 0:

            outputfilename = "".join(outputfilename.split(".")[:-1] + ".psp")
        else:
            outputfilename += ".psp"

    # Writing configuration file to 'PeakSearch.cfg'
    with open(outputfilename, "w") as configfile:
        config.write(configfile)

    return outputfilename


def readPeakSearchConfigFile(filename):
    config = CONF.RawConfigParser()
    config.optionxform = str
    #    config = MyCasePreservingConfigParser()

    config.read(filename)

    section = config.sections()[0]

    if section not in ("PeakSearch",):
        raise ValueError(
            "wrong section name in config file {}. Must be in {}".format(
                filename, "IndexRefine"
            )
        )

    #     print "section", section

    dict_param = {}

    list_options = config.options(section)

    for option in list_options:

        #         print "\n option\n", option
        for option_ref, option_type in zip(
            LIST_OPTIONS_PEAKSEARCH, LIST_OPTIONS_TYPE_PEAKSEARCH
        ):

            #             print "option_ref, option_type", option_ref, option_type

            if option_ref == option or option_ref.lower() == option:

                #                 print "BINGO! I m able to read %s" % option_ref
                #                 print "data type should be: %s" % option_type

                try:
                    optionkey = CONVERTKEY_dict[option_ref]
                except KeyError:
                    optionkey = option_ref

                #                 print 'optionkey', optionkey

                option_lower = option_ref.lower()

                try:
                    if option_type.startswith("int"):
                        dict_param[optionkey] = int(
                            config.getint(section, option_lower)
                        )
                    elif option_type.startswith(("float",)):
                        dict_param[optionkey] = float(
                            config.getfloat(section, option_lower)
                        )
                    else:
                        dict_param[optionkey] = config.get(section, option_lower)

                #                     print "matindex", matindex
                #                     print "optionkey", optionkey
                #                     print "option_lower", option_lower
                except ValueError:
                    print(
                        "Value of option '{}' has not the correct type".format(option)
                    )
                    return None

                break

    #     print "Finally, I ve read these parameters"
    #     print dict_param
    return dict_param


def read_background_flag(background_flag):
    """
    interpret the background flag (field used in FileSeries/Peak_Search.py)
    
    return two values to put in dict_param of peaksearch_series
    """
    formulaexpression = "A-B"

    if background_flag in ("auto", "AUTO", "yes", "YES", "y"):
        Data_for_localMaxima = "auto_background"
    elif background_flag in ("n", "no", "NO", None, "None", "NONE"):
        Data_for_localMaxima = None
    else:
        # should be a path to a single imagefile
        # (plus optionally a mathematical formula expression)

        ressplit = background_flag.split(";")

        if len(ressplit) != 2:
            filepath = background_flag
        elif len(ressplit) == 2:
            filepath, formulaexpression = ressplit

        if not os.path.exists(filepath):
            wx.MessageBox(
                "{} does not exist. Check filename and path.".format(filepath), "Error"
            )
            return

        Data_for_localMaxima = filepath

        print("Image file path used for background", Data_for_localMaxima)

    return Data_for_localMaxima, formulaexpression


def read_blacklist_filepath(blacklistpeaklist):
    if blacklistpeaklist == "None":
        Remove_BlackListedPeaks_fromfile = None
    # fullpath
    else:
        Remove_BlackListedPeaks_fromfile = blacklistpeaklist
    return Remove_BlackListedPeaks_fromfile


# --- -------------- Plot image and peaks
def plot_image_markers(image, markerpos, position_definition=1):
    """
    plot 2D array (image) with markers at first two columns of (markerpos)

    """
    fig = pp.figure()
    ax = fig.add_subplot(111)

    numrows, numcols = image.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = image[row, col]
            return "x = {:.4f}, y = {:.4f}, z = {:.4f}".format(x, y, z)
        else:
            return "x = {:.4f}, y = {:.4f}".format(x, y)

    ax.format_coord = format_coord

    from matplotlib.patches import Circle

    # correction only to fitdata peak position to the display
    if position_definition == 1:
        PointToPlot = np.zeros(markerpos.shape)
        PointToPlot[:, :2] = markerpos[:, :2] - np.array([1, 1])

    for po in PointToPlot[:, :2]:

        large_circle = Circle(po, 7, fill=False, color="b")
        center_circle = Circle(po, 0.5, fill=True, color="r")
        ax.add_patch(large_circle)
        ax.add_patch(center_circle)

    ax.imshow(np.log(image), interpolation="nearest")

    pp.show()


def getPixeltest(x, y, filename="test_0072.edf", ccdtypegeometry="edf"):
    """
    x,y on display mapcanvas of lauetools are swaped by respect to array   = d[y,x]

    TODO: only for frelon? camera
    """

    f = open(filename, "rb")
    f.seek(1024 + 2 * (2048 * (2047 - y) + x))
    val = struct.unpack("H", f.read(2))

    print("val", val)
    f.close()
    return val[0]


def applyformula_on_images(
    A, B, formulaexpression="A-B", SaturationLevel=None, clipintensities=True
):
    """
    calculate image data array from math expression

    A, B            : ndarray  of the same shape

    SaturationLevel  :  saturation level of intensity 

    clipintensities    :   clip resulting intensities to zero and saturation value
    """
    if A.shape != B.shape:
        raise ValueError(
            "input arrays in applyformula_on_images() have not the same shape."
        )

    A = np.array(A, dtype="float32")
    B = np.array(B, dtype="float32")

    #        nbpix = A.shape[0] * A.shape[1]
    #        resformula = np.zeros_like(A)
    #        resformula.dtype = 'int32'

    #        A.dtype = 'int32'
    #        B.dtype = 'int32'
    resformula = eval(formulaexpression)
    #        print resformula.dtype
    newarray = resformula

    if clipintensities:
        if SaturationLevel is not None:
            newarray = np.clip(resformula, 0, SaturationLevel)
        else:
            raise ValueError(
                "saturation level is unknown to clip data for large intensity!\n Missing argument in applyformula_on_images()"
            )

    return newarray


# --- -------------- multiple file peak search
def peaksearch_fileseries(
    fileindexrange,
    filenameprefix,
    suffix="",
    nbdigits=4,
    dirname_in="/home/micha/LaueProjects/AxelUO2",
    outputname=None,
    dirname_out=None,
    CCDLABEL="MARCCD165",
    KF_DIRECTION="Z>0",  # not used yet
    dictPeakSearch=None,
):
    """
    peaksearch function to be called for multi or single processing
    """
    # peak search Parameters update from .psp file
    if isinstance(dictPeakSearch, dict):
        for key, val in list(dictPeakSearch.items()):
            PEAKSEARCHDICT_Convolve[key] = val

    if not "MinPeakSize" in PEAKSEARCHDICT_Convolve:
        PEAKSEARCHDICT_Convolve["MinPeakSize"] = 0.65
        PEAKSEARCHDICT_Convolve["MaxPeakSize"] = 3.
        print("Default values for minimal and maximal peaksize are used!. Resp. 0.65 and 3 pixels.")

    PEAKSEARCHDICT_Convolve["PeakSizeRange"] = (
        copy.copy(PEAKSEARCHDICT_Convolve["MinPeakSize"]),
        copy.copy(PEAKSEARCHDICT_Convolve["MaxPeakSize"]),
    )
    del PEAKSEARCHDICT_Convolve["MinPeakSize"]
    del PEAKSEARCHDICT_Convolve["MaxPeakSize"]

    # ----handle reading of filename
    # special case for _mar.tif files...
    if nbdigits in ("varying",):
        DEFAULT_DIGITSENCODING = 4
        encodingdigits = "{" + ":0{}".format(DEFAULT_DIGITSENCODING) + "}"
    # normal case
    else:
        encodingdigits = "{" + ":0{}".format(int(nbdigits)) + "}"
        nbdigits = int(nbdigits)

    if suffix == "":
        suffix = ".mccd"

    if dirname_in != None:
        filenameprefix_in = os.path.join(dirname_in, filenameprefix)
    else:
        filenameprefix_in = filenameprefix

    filename_wo_path = filenameprefix_in.split("/")[-1]

    if outputname != None:
        prefix_outputname = outputname
    else:
        prefix_outputname = (
            filenameprefix
        )  # filename_wo_path[:-len(file_extension) - 1]

    if dirname_out != None:
        prefix_outputname = os.path.join(dirname_out, prefix_outputname)

    if len(fileindexrange) == 2:
        fileindexrange = fileindexrange[0], fileindexrange[1], 1

    # create a data considered as background from an imagefile
    BackgroundImageCreated = False
    flag_for_backgroundremoval = PEAKSEARCHDICT_Convolve["Data_for_localMaxima"]

    # flag_for_backgroundremoval is a file path to an imagefile
    # create background data: dataimage_bkg
    if flag_for_backgroundremoval not in ("auto_background", None) and not isinstance(
        flag_for_backgroundremoval, np.ndarray
    ):

        fullpath_backgroundimage = PEAKSEARCHDICT_Convolve["Data_for_localMaxima"]

        #         print "fullpath_backgroundimage ", fullpath_backgroundimage

        dirname_bkg, imagefilename_bkg = os.path.split(fullpath_backgroundimage)

        CCDlabel_bkg = CCDLABEL

        (dataimage_bkg, framedim_bkg, fliprot_bkg) = readCCDimage(
            imagefilename_bkg, CCDLabel=CCDlabel_bkg, dirname=dirname_bkg
        )

        BackgroundImageCreated = True

    for fileindex in list(
        range(fileindexrange[0], fileindexrange[1] + 1, fileindexrange[2])
    ):
        # TODO to move this branching elsewhere (readmccd)
        if suffix.endswith("_mar.tif"):
            filename_in = setfilename(
                filenameprefix_in + "{}".format(fileindex) + suffix, fileindex
            )
        else:
            #             filename_in = filenameprefix_in + encodingdigits % fileindex + suffix
            filename_in = filenameprefix_in + str(fileindex).zfill(nbdigits) + suffix

        tirets = "-" * 15
        print(
            "\n\n {} PeakSearch on filename {}\n{}\n{}{}{}n\n".format(
                tirets, tirets, filename_in, tirets, tirets, tirets
            )
        )

        if not os.path.exists(filename_in):
            raise ValueError(
                "\n\n*******\nSomething wrong with the filename: {}. Please check carefully the filename!".format(
                    filename_in
                )
            )

        # remove a single image (considered as background) to current image
        if BackgroundImageCreated:

            print(
                "consider dataimagefile {} as background".format(
                    fullpath_backgroundimage
                )
            )
            (dataimage_raw, framedim_raw, fliprot_raw) = readCCDimage(
                filename_in, CCDLabel=CCDLABEL, dirname=None
            )

            if "formulaexpression" in dictPeakSearch:
                formulaexpression = dictPeakSearch["formulaexpression"]
            else:
                raise ValueError(
                    'Missing "formulaexpression" to operate on images before peaksearch in peaksearch_fileseries()'
                )

            saturationlevel = DictLT.dict_CCD[CCDLABEL][2]

            dataimage_corrected = applyformula_on_images(
                dataimage_raw,
                dataimage_bkg,
                formulaexpression=formulaexpression,
                SaturationLevel=saturationlevel,
                clipintensities=True,
            )

            print("using {} in peaksearch_fileseries".format(formulaexpression))

            #             print 'Imin Imax dataimage_raw', np.amin(A), np.amax(A)
            #             print 'Imin Imax dataimage_bkg', np.amin(B), np.amax(B)

            # for finding local maxima in image from formula
            PEAKSEARCHDICT_Convolve["Data_for_localMaxima"] = fullpath_backgroundimage

            # for fitting peaks in image from formula
            PEAKSEARCHDICT_Convolve["reject_negative_baseline"] = False
            PEAKSEARCHDICT_Convolve["formulaexpression"] = formulaexpression
            PEAKSEARCHDICT_Convolve["Fit_with_Data_for_localMaxima"] = True

        #             print 'Imin Imax dataimage_corrected', np.amin(dataimage_corrected), np.amax(dataimage_corrected)

        # --------------------------
        # launch peaksearch
        # ------------------------
        Res = PeakSearch(
            filename_in,
            CCDLabel=CCDLABEL,
            Saturation_value=DictLT.dict_CCD[CCDLABEL][2],
            Saturation_value_flatpeak=DictLT.dict_CCD[CCDLABEL][2],
            **PEAKSEARCHDICT_Convolve
        )

        if Res in (False, None):
            print("No peak found for image file: ", filename_in)
        #             Isorted, fitpeak, localpeak = None, None, None
        else:  # write file with comments
            Isorted, fitpeak, localpeak = Res[:3]

            params_comments = "Peak Search and Fit parameters\n"

            params_comments += "# {}: {}\n".format("CCDLabel", CCDLABEL)

            for key, val in list(PEAKSEARCHDICT_Convolve.items()):
                if not BackgroundImageCreated or key not in ("Data_for_localMaxima",):
                    params_comments += "# " + key + " : " + str(val) + "\n"

            if BackgroundImageCreated:
                params_comments += (
                    "# "
                    + "Data_for_localMaxima"
                    + " : {} \n".format(fullpath_backgroundimage)
                )
            # .dat file extension is done in writefile_Peaklist()
            # filename_out = prefix_outputname + encodingdigits % fileindex
            # TODO valid whatever
            filename_out = prefix_outputname + str(fileindex).zfill(nbdigits)
            IOLT.writefile_Peaklist(
                "{}".format(filename_out),
                Isorted,
                overwrite=1,
                initialfilename=filename_in,
                comments=params_comments,
            )

    print("\n\n\n*******************\n\n\n task of peaksearch COMPLETED!")


def peaksearch_multiprocessing(
    fileindexrange,
    filenameprefix,
    suffix="",
    nbdigits=4,
    dirname_in="/home/micha/LaueProjects/AxelUO2",
    outputname=None,
    dirname_out=None,
    CCDLABEL="MARCCD165",
    KF_DIRECTION="Z>0",
    dictPeakSearch=None,
    nb_of_cpu=2,
):
    """
    launch several processes in parallel
    """
    import multiprocessing

    try:
        if len(fileindexrange) > 2:
            print("\n\n STEP INDEX is SET to 1 \n\n")
        index_start, index_final = fileindexrange[:2]
    except:
        raise ValueError(
            "Need 2 file indices integers in fileindexrange=(indexstart, indexfinal)"
        )
        return

    fileindexdivision = GT.getlist_fileindexrange_multiprocessing(
        index_start, index_final, nb_of_cpu
    )
    #
    #    fileindexrange, filenameprefix,
    #                          suffix='', nbdigits=4,
    #                          dirname_in='/home/micha/LaueProjects/AxelUO2',
    #                          outputname=None, dirname_out=None,
    #                            CCDLABEL='MARCCD165',
    #                            fileextension='mccd',
    #                            dictPeakSearch=None

    #    t00 = ttt.time()
    jobs = []
    for ii in list(range(nb_of_cpu)):
        proc = multiprocessing.Process(
            target=peaksearch_fileseries,
            args=(
                fileindexdivision[ii],
                filenameprefix,
                suffix,
                nbdigits,
                dirname_in,
                outputname,
                dirname_out,
                CCDLABEL,
                KF_DIRECTION,
                dictPeakSearch,
            ),
        )
        jobs.append(proc)
        proc.start()


#    t_mp = ttt.time() - t00
#    print "Execution time : %.2f" % t_mp


def peaklist_dict(prefixfilename, startindex, finalindex, dirname=None):
    dict_peaks = {}
    for k in list(range(startindex, finalindex + 1)):
        filename = prefixfilename + "{:04d}.dat".format(k)

        array_peaks = IOLT.read_Peaklist(filename, dirname=dirname)
        dict_peaks[k] = array_peaks

    return dict_peaks


# --- ----  Filtering and background removal function
def gauss_kern(size, sizey=None):
    """
    Returns a normalized 2D gauss kernel array for convolutions
    """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size : size + 1, -sizey : sizey + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(sizey)))
    return g / g.sum()


def blur_image(im, n, ny=None):
    """
    blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im, g, mode="valid")
    return improc


def blurCCD(im, n):
    """
    apply a blur filter to image ndarray
    """
    framedim = im.shape
    minipix = np.amin(im)
    tab = minipix * np.ones(framedim)

    blurredpart = blur_image(im, n)
    tab[n:-n, n:-n] = blurredpart
    return tab


def circularMask(center, radius, arrayshape):
    """
    return a boolean ndarray of elem in array inside a mask
    """
    II, JJ = np.mgrid[0 : arrayshape[0], 0 : arrayshape[1]]

    cond = (II - center[0]) ** 2 + (JJ - center[1]) ** 2 <= radius ** 2
    return cond


def compute_autobackground_image(dataimage, boxsizefilter=10):
    """
    return 2D array of filtered data array
    :param dataimage: array of image data
    :type dataimage: 2D array
    """

    bkgimage = filter_minimum(dataimage, boxsize=boxsizefilter)

    return bkgimage


def computefilteredimage(
    dataimage, bkg_image, CCDlabel, kernelsize=5, formulaexpression="A-B", usemask=True
):
    """
    return 2D array of initial image data without background given by bkg_image data

    usemask        : True  then substract bkg image on masked raw data
                    False  apply formula on all pixels (no mask)

    :param dataimage: array of image data
    :type dataimage: 2D array
    :param bkg_image: array of filtered image data (background)
    :type bkg_image: 2D array
    :param CCDlabel: key for CCD dictionary
    :type CCDlabel: string
    """

    framedim = DictLT.dict_CCD[CCDlabel][0]
    SaturationLevel = DictLT.dict_CCD[CCDlabel][2]
    dataformat = DictLT.dict_CCD[CCDlabel][5]

    print("framedim in computefilteredimage ", framedim)
    print("CCDlabel in computefilteredimage ", CCDlabel)
    #
    #     if CCDlabel in ('EDF',):
    #         return dataimage

    #     if framedim not in ((2048, 2048), [2048, 2048]):
    #         raise ValueError, "Background removal still implemented for non squared camera "
    # computing substraction on whole array
    if CCDlabel in ("sCMOS", "sCMOS_fliplr"):
        usemask = False

    if usemask:
        # mask parameter to avoid high intensity steps at border:
        # TODO: to compute for all CCD types
        #     center, mask_radius, minvalue = (1024, 1024), 1010, 0

        # radius of the mask a bit smaller than real radius avoiding circular intensity step

        mask_radius = min(framedim[0] // 2, framedim[1] // 2) - 15
        center, minvalue = (framedim[0] // 2, framedim[1] // 2), 0

        print("mask_radius", mask_radius)
        print("center", center)

        dataarray2D_without_background = filterimage(
            dataimage,
            framedim,
            blurredimage=bkg_image,
            kernelsize=kernelsize,
            mask_parameters=(center, mask_radius, minvalue),
            clipvalues=(0, SaturationLevel),
            imageformat=dataformat,
        )

    else:
        dataarray2D_without_background = applyformula_on_images(
            dataimage,
            bkg_image,
            formulaexpression=formulaexpression,
            SaturationLevel=SaturationLevel,
            clipintensities=True,
        )

    return dataarray2D_without_background


def filterimage(
    image_array,
    framedim,
    blurredimage=None,
    kernelsize=5,
    mask_parameters=None,
    clipvalues=None,
    imageformat=np.uint16,
):
    """
    compute a difference of images inside a region defined by a mask

    blurredimage:    ndarray image to substract to image_array
    kernelsize:    pixel size of gaussian kernel if blurredimage is None

    mask_parameters: circular mask parameter: center=(x,y), radius, value outside mask
    """
    if blurredimage is None:
        dblur = blurCCD(image_array, kernelsize)
    else:
        dblur = blurredimage

    if clipvalues is not None:
        minival, maxival = clipvalues
        tab = np.clip(image_array - dblur, minival, maxival)
    else:
        tab = image_array - dblur

    if mask_parameters is not None:
        center, radius, minvalue = mask_parameters
        if minvalue == "minvalue":
            minivalue = np.amin(image_array)
        else:
            minivalue = 0
        maskcd = np.where(circularMask(center, radius, framedim), tab, minivalue)

    return np.array(maskcd, dtype=imageformat)


def rebin2Darray(inputarray, bin_dims=(2, 2), operator="mean"):
    """
    rebin 2D array by applying an operator to define the value of one element from the other

    operator: mean, min, max, sum
    bin_dims: side sizes of binning. (2,3) means 2X3
    """
    rows, cols = inputarray.shape
    binr, binc = bin_dims
    if operator == "mean":
        op = np.mean
    if operator == "max":
        op = np.max
    if operator == "min":
        op = np.max
    if operator == "sum":
        op = np.sum
    if rows % binr == 0 and cols % binc == 0:
        return op(
            op(inputarray.reshape(rows // binr, binr, cols // binc, binc), axis=3),
            axis=1,
        )
    else:
        print("array and binning size are not compatible")
        return None


def blurCCD_with_binning(im, n, binsize=(2, 2)):
    """
    blur the array by rebinning before and after aplying the filter
    """
    framedim = im.shape
    imrebin = rebin2Darray(im, bin_dims=binsize, operator="min")
    if imrebin is None:
        return None

    dblur = blurCCD(imrebin, n)

    return np.repeat(np.repeat(dblur, binsize[0], axis=0), binsize[1], axis=1)


def filter_minimum(im, boxsize=10):
    return ndimage.filters.minimum_filter(im, size=boxsize)


def remove_minimum_background(im, boxsize=10):
    """
    remove to image array the array resulting from minimum_filter
    """
    return im - filter_minimum(im, boxsize=boxsize)


def purgePeaksListFile(filename1, blacklisted_XY, dist_tolerance=0.5, dirname=None):
    """
    remove in peaklist .dat file peaks that are in blacklist

    blacklisted_XY:         [X1,Y1],[X2,Y2]
    """
    data_peak = IOLT.read_Peaklist(filename1, dirname=dirname)

    XY = data_peak[:, 0:2].T

    blacklisted_XY = np.array(blacklisted_XY).T

    peakX, peakY, tokeep = GT.removeClosePoints_two_sets(
        XY, blacklisted_XY, dist_tolerance=dist_tolerance, verbose=0
    )

    return peakX, peakY, tokeep


def write_PurgedPeakListFile(
    filename1, blacklisted_XY, outputfilename, dist_tolerance=0.5, dirname=None
):
    """
    write a new .dat file where peaks in blacklist are omitted
    """
    peakX, peakY, tokeep = purgePeaksListFile(
        filename1, blacklisted_XY, dist_tolerance=0.5, dirname=dirname
    )

    data_peak = IOLT.read_Peaklist(filename1, dirname=dirname)

    new_data_peak = np.take(data_peak, tokeep, axis=0)

    if dirname is not None:
        outputfilename = os.path.join(dirname, outputfilename)

    IOLT.writefile_Peaklist(
        outputfilename,
        new_data_peak,
        overwrite=1,
        initialfilename=filename1,
        comments="Some peaks have been removed by write_PurgedPeakListFile",
        dirname=dirname,
    )

    print("New peak list file {} has been written".format(outputfilename))


def removePeaks_inPeakList(
    PeakListfilename,
    BlackListed_PeakListfilename,
    outputfilename,
    dist_tolerance=0.5,
    dirname=None,
):
    """
    read peaks PeakListfilename and remove those in BlackListed_PeakListfilename
    and write a new peak list file
    """
    data_peak_blacklisted = IOLT.read_Peaklist(
        BlackListed_PeakListfilename, dirname=dirname
    )

    XY_blacklisted = data_peak_blacklisted[:, 0:2].T

    write_PurgedPeakListFile(
        PeakListfilename,
        XY_blacklisted,
        outputfilename,
        dist_tolerance=0.5,
        dirname=dirname,
    )


def merge_2Peaklist(
    filename1, filename2, dist_tolerance=5, dirname1=None, dirname2=None, verbose=0
):
    """
    return merge spots data from two peaklists and removed duplicates within dist_tolerance (pixel)
    
    """
    data_peak_1 = IOLT.read_Peaklist(filename1, dirname=dirname1)
    data_peak_2 = IOLT.read_Peaklist(filename2, dirname=dirname2)

    XY1 = data_peak_1[:, 0:2]
    XY2 = data_peak_2[:, 0:2]

    XY, ind_delele_1, ind_delele_2 = GT.mergelistofPoints(
        XY1, XY2, dist_tolerance=dist_tolerance, verbose=verbose
    )

    data1 = np.delete(data_peak_1, ind_delele_1, axis=0)
    data2 = np.delete(data_peak_2, ind_delele_2, axis=0)

    return np.concatenate((data1, data2), axis=0)


def writefile_mergedPeaklist(
    filename1,
    filename2,
    outputfilename,
    dist_tolerance=5,
    dirname1=None,
    dirname2=None,
    verbose=0,
):
    """
    write peaklist file from the merge of spots data from two peaklists
    (and removed duplicates within dist_tolerance (pixel))
    """
    merged_data = merge_2Peaklist(
        filename1, filename2, dist_tolerance, dirname1, dirname2, verbose
    )
    comments = "Peaks from merging {} and {} with pixel tolerance {:.3f}".format(
        filename1, filename2, dist_tolerance
    )
    IOLT.writefile_Peaklist(outputfilename, merged_data, 1, None, comments, None)

    print("Merged peak lists written in file: ", outputfilename)
