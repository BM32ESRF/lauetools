#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os, sys

sys.path.append("../..")


import numpy as np

import LaueTools.IOimagefile as rmccd
import LaueTools.dict_LaueTools as dlt

from fabio.TiffIO import TiffIO


def write_image(data, filename, CCDLabel='MARCCD165', dirname=None, verbose=0, header=None):
    """Main function to write image files in desired format."""
    if dirname is not None:
        
        filename = os.path.join(dirname, filename)

    dims, _, _, fliprot, _, dtype, _, ext = dlt.dict_CCD[CCDLabel]

    if CCDLabel in ('MARCCD165', 'sCMOS', 'sCMOS_fliplr'):

        #preparing
        data = treat_fliprot(data, fliprot)

        check_dimensions(data, dims)

        filename = add_extension(filename, ext)

        if header is None:
            header = get_header(CCDLabel)

        dataformat = get_format(dtype)

        #writing
        # newfile = open(filename, 'wb')
        # newfile.write(header)
        # data = np.array(data, dtype=dataformat)
        # data.tofile(newfile)
        # newfile.close()

        tif = TiffIO(filename, mode='w')
        data = np.array(data, dtype=dataformat)
        tif.writeImage(data)

        print_msg("Wrote {}x{} image with {} format in {}.".format(dims[0], dims[1], CCDLabel, filename), verbose)

    else:
        sys.exit("Required CCD format '{}' is not available in writeImage!".format(CCDLabel))


def treat_fliprot(dataimage, fliprot):

    if fliprot == "spe":
        dataimage = np.rot90(dataimage, k=3)

    elif fliprot == "VHR_Feb13":
#            self.dataimage_ROI = np.rot90(self.dataimage_ROI, k=3)
        # TODO: do we need this left and right flip ?
        dataimage = np.fliplr(dataimage)

    elif fliprot == "sCMOS_fliplr":

        dataimage = np.fliplr(dataimage)

    elif fliprot == "vhr":  # july 2012 close to diamond monochromator crystal
        dataimage = np.fliplr(dataimage)
        dataimage = np.rot90(dataimage, k=1)

    elif fliprot == "vhrdiamond":  # july 2012 close to diamond monochromator crystal
        dataimage = np.fliplr(dataimage)
        dataimage = np.rot90(dataimage, k=1)

    elif fliprot == "frelon2":
        dataimage = np.flipud(dataimage)

    else:
        pass

    return dataimage


def check_dimensions(data, dims):

    if data.shape[0] != dims[0] or data.shape[1] != dims[1]:

        sys.exit("Image dims ({}x{}) do not match required format dims ({}x{})!".format(data.shape[0],
                                                                                        data.shape[1],
                                                                                        *dims))


def calc_ndigits(img_qty):

    return int(np.floor(np.log10(img_qty)) + 1)


def add_extension(filedir, ext):

    return filedir + "." + ext


def get_header(CCDLabel):

    if CCDLabel == 'MARCCD165':

        header =  header_marccd

    elif CCDLabel in ('sCMOS', 'sCMOS_fliplr'):

        header =  header_scmos

    else:

        header = []

    return header


def get_format(dtype):

    return np.dtype(dtype)

# prepare headers (done once at the import in case of several files to be generated)
__path__ = os.path.dirname(__file__)


def readheader_scmos(filename, framedim=None):

    if framedim is None:
        framedim = dlt.dict_CCD['sCMOS'][0]

    filesize = os.path.getsize(filename)
    offsetheader = filesize-(np.prod(framedim))*2

    return rmccd.readheader(filename, offset=offsetheader)

header_marccd = rmccd.readheader(os.path.join(__path__, "header_marccd.txt"))
header_scmos  = readheader_scmos(os.path.join(__path__, "header_scmos.txt"))        
        
#handle printing and verbosity 
def print_msg(msg, verbose):
    
    if verbose:
        print(msg)
