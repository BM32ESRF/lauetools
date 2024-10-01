# -*- coding: utf-8 -*-
r"""
imagescollector module is made for reading data contained in binary image file
fully or partially.

"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

# built-in modules
import os

import LaueTools.generaltools as GT
import LaueTools.IOimagefile as IOimage  
import LaueTools.readmccd as RMCCD

def setfilename(prefix,index,CCDLabel='sCMOS'):
    if CCDLabel in ('sCMOS',):
        return prefix+'%04d'%index + '.tif'
    elif CCDLabel in ('MARCCD165',):
        return prefix+'%04d'%index + '.mccd'

def collectpixelvalue_singlefile(index, peaklist=None, prefix=None, folder =None,CCDLabel='sCMOS'):
    """ collect single pixel intensities located in several locations (defined by peaklist) in 1 image
    
    :param index: int, image file index 
    :param peaklist: list or array of 2d pixels position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param CCDLabel: label of detector
    
    :return: array of pixel values with shape = (nbimages, nbpeaks)

    """
    filename = setfilename(prefix,index,CCDLabel)   
    imagefilename = os.path.join(folder, filename)

    pixvals = IOimage.pixelvalat(imagefilename, xy=peaklist, sortpeaks=False, CCDLabel=CCDLabel)
    return pixvals



def collectroismax_singlefile(index, roicenters=None, prefix=None, folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None):
    """ collect max intensity in some given rois centered on peaks defined in peaklist in 1 image
    
    
        
    :param index: int, image file index
    :param roicenters: list or array of 2d PIXELS position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param folder: str, path to folder containing all images
    :param boxsize_row, boxsize_line: half boxsize along x and y of the roi (can be a list?)
    :param CCDLabel: label of detector
    
    :return: array of max intensities: shape = (nbimages, nbpeaks)
    """
    filename = setfilename(prefix,index,CCDLabel)   
    imagefilename = os.path.join(folder, filename)

    pixvals = IOimage.getroismax(imagefilename, roicenters=roicenters,
                                 halfboxsize=(boxsize_row,boxsize_line), CCDLabel=CCDLabel)
    return pixvals

def collectroisXYmax_singlefile(index, roicenters=None, prefix=None, folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None):
    """ collect pixel XY position of max intensity in some given rois centered on peaks defined in peaklist in 1 image

    :param index: int, image file index
    :param peaklist: list or array of 2d pixels position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param folder: str, path to folder containing all images
    :param boxsize_row, boxsize_line: half boxsize along x and y of the roi (can be a list?)
    :param CCDLabel: label of detector
    
    :return: array of max intensities: shape = (nbimages, nbpeaks)
    """
    filename = setfilename(prefix,index,CCDLabel)   
    imagefilename = os.path.join(folder, filename)

    pixvals = IOimage.getroisXYmax(imagefilename, roicenters=roicenters,
                                   halfboxsize=(boxsize_row,boxsize_line),
                                   CCDLabel=CCDLabel)
    return pixvals

def collectroisXYcenterofmass_singlefile(index, roicenters=None, prefix=None, folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None):
    """ collect pixel XY position of max intensity in some given rois centered on peaks defined in peaklist in 1 image

    :param index: int, image file index
    :param peaklist: list or array of 2d pixels position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param folder: str, path to folder containing all images
    :param boxsize_row, boxsize_line: half boxsize along x and y of the roi (can be a list?)
    :param CCDLabel: label of detector
    
    :return: array of max intensities: shape = (nbimages, nbpeaks)
    """
    filename = setfilename(prefix,index,CCDLabel)   
    imagefilename = os.path.join(folder, filename)

    pixvals = IOimage.getroiscenterofmass(imagefilename, roicenters=roicenters,
                                   halfboxsize=(boxsize_row,boxsize_line),
                                   CCDLabel=CCDLabel)
    return pixvals

def collectroisfitpeak_singlefile(index, roicenters=None, prefix=None,
                                  folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None,
                            computerrorbars=False):
    """ collect pixel XY position of max intensity in some given rois centered on peaks defined in peaklist in 1 image

    :param index: int, image file index
    :param peaklist: list or array of 2d pixels position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param folder: str, path to folder containing all images
    :param boxsize_row, boxsize_line: half boxsize along x and y of the roi (can be a list?)
    :param CCDLabel: str, label of detector
    :param computerrorbars: bool, compute fit parameter error bars
    
    :return: array of max intensities: shape = (nbimages, nbpeaks)
    """
    filename = setfilename(prefix,index,CCDLabel)   
    imagefilename = os.path.join(folder, filename)
    
    resfit = RMCCD.readoneimage_multiROIfit(imagefilename,
                                        roicenters,
                                        [boxsize_row,boxsize_line],
                                        stackimageindex=-1,
                                        CCDLabel=CCDLabel,
                                        baseline="auto",  # min in ROI box
                                        startangles=0.0,
                                        start_sigma1=1.,
                                        start_sigma2=1.,
                                        position_start='max',  # 'centers' or 'max'
                                        showfitresults=0,
                                        offsetposition=1,
                                        fitfunc="gaussian",
                                        xtol=0.00001,
                                        addImax=False,
                                        use_data_corrected=None,
                                        computerrorbars=computerrorbars)
    #  bkg, amp, x, y std1, std2, angle
    #print('resfit',resfit)
    toreturn = resfit[0]
    if computerrorbars:
        # infodict["pfit_leastsq"]=pfit_leastsq
        # infodict["perr_leastsq"]=perr_leastsq
        infodict = resfit[2]
        toreturn = resfit[0], infodict['pfit_leastsq'], infodict['perr_leastsq']
        
    return toreturn