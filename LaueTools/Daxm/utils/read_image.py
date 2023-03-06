#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os
import re

import numpy as np

import LaueTools.IOimagefile as rmccd
import LaueTools.dict_LaueTools as dlt

from LaueTools.Daxm.utils.num import is_int

def split_filename(filename):
    rule = re.compile(u'([a-zA-Z_0-9]+)\.(\d+)(\S+)')#,add argument flags=re.LOCALE  for python2!!

    parts = rule.findall(filename[::-1])[0]

    return str(parts[2][::-1]), str(parts[1][::-1]), str(parts[0][::-1])


def sprintf_filename(prefix, idx, extension, ndigits=4):

    return prefix + str(idx).zfill(ndigits) + '.' + extension


def test_filename(prefix, idx, extension='tif', ndigits=4, folder=''):

    filename = sprintf_filename(prefix, idx, extension, ndigits)

    return os.path.isfile(os.path.join(folder, filename))


def ccd_to_extension(CCDLabel):

    return dlt.dict_CCD[CCDLabel][-1]


def split_linesubfolder(folder):
    """folder  must contain the str   'line' or 'row'
    """

    subfolder = os.path.basename(folder)

    rule = re.compile("(\d+)(\S+)")#, flags=re.LOCALE)

    parts = rule.findall(subfolder[::-1])

    result = None

    if len(parts) and len(parts[0]) == 2 and is_int(parts[0][0]) and parts[0][1][::-1] in ("line", "row"):
        subname = parts[0][1][::-1]
        subindx = int(parts[0][0][::-1])
        result = subname, subindx, os.path.dirname(folder)

    return result

def read_image_rectangle(filename,  xlim, ylim, dirname=None, CCDLabel='MARCCD165'):
    """
    returns a 2d array of integers from a binary image file. 
    Data are taken only from a rectangle defined by xlim, ylim

    
    Returns
    ----------
    dataimage : 2D array
                image data pixel intensity
    """
    (framedim,
     pixelsize,
    saturationvalue,
    fliprot,
    offsetheader,
    formatdata,
    comments,
    extension) = dlt.dict_CCD[CCDLabel]
    
    #print "framedim read from DictLT.dict_CCD in readrectangle_in_image()",framedim
    #print "formatdata",formatdata
    #print 'offsetheader',offsetheader
    # recompute headersize
    if dirname is not None:
        fullpathfilename = os.path.join(dirname,filename)
    else:
        dirname = os.curdir
        fullpathfilename = filename
        
    if formatdata in ("uint16",):
        nbBytesPerElement=2
    if formatdata in ("uint32",):
        nbBytesPerElement=4
    
    try:
        filesize = os.path.getsize(fullpathfilename)
    except OSError:
        print('missing file %s'%fullpathfilename)
        return None

    # uint16
    offsetheader = filesize-(framedim[0]*framedim[1])*nbBytesPerElement
    
    #print 'calculated offset of header from file size...',offsetheader
     
    xpixmin, xpixmax = xlim
    ypixmin, ypixmax = ylim
    
    lineFirstElemIndex = ypixmin
    lineLastElemIndex = ypixmax
    
    band = rmccd.readoneimage_band(fullpathfilename, framedim=framedim, dirname=None,
                 offset=offsetheader,
                 line_startindex=lineFirstElemIndex,
                 line_finalindex=lineLastElemIndex,
                 formatdata="uint16")
    
    nblines = lineLastElemIndex-lineFirstElemIndex+1
    
    band2D=np.reshape(band,(nblines,framedim[1]))    
    
    rectangle2D = band2D[:,xpixmin:xpixmax+1]
    
    return rectangle2D
