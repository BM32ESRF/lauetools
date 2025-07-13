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
import LaueTools.imageprocessing as Improc
import LaueTools.dict_LaueTools as DictLT


def setfilename(prefix:str,index:int,CCDLabel='sCMOS')->str:
    if CCDLabel in ('sCMOS',):
        return prefix+'%04d'%index + '.tif'
    elif CCDLabel in ('MARCCD165',):
        return prefix+'%04d'%index + '.mccd'
    
def collectroiarray_singlefile(imageindex, roicenter=None, prefix=None, folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None):
    """ collect pixel intensity array in given SINGLE roi 
        
    :param index: int, image file index
    :param roicenters: list or array of 2d PIXELS position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param folder: str, path to folder containing all images
    :param boxsize_row, boxsize_line: half boxsize along x and y of the roi (can be a list?)
    :param CCDLabel: label of detector
    
    :return: array of max intensities: shape = (nbimages, nbpeaks)
    """
    
    filename = prefix+'%04d'%imageindex + '.tif'   
    imagefilename = os.path.join(folder, filename)
    
    halfboxsizes = boxsize_row, boxsize_line  # along X, along Y
    
    xpic, ypic = roicenter

#     if ccdlabel in ('EIGER_4MCdTestack'): # or other stack images detector
#         print('CCDLabel ', CCDLabel)
#         filename = filename_representative
#         stackimageindex = imageindex
#         print("filename",filename)
#         print("stackimageindex",stackimageindex)
    
    stackimageindex = -1

    framedimraw = DictLT.dict_CCD[CCDLabel][0]
    fliprot = DictLT.dict_CCD[CCDLabel][3]
    if 0:
        print("framedim of CCDLabel", framedimraw, CCDLabel)
        print("fliprot", fliprot)

    center_pixel = xpic, ypic
    if fliprot in ("sCMOS_fliplr",):
        center_pixel = (framedimraw[1] - xpic, ypic)
    # if ccdlabel in ('EIGER_4MCdTestack'):
    #     center_pixel = (ypic, xpic)
    #     _fdim = DictLT.dict_CCD[ccdlabel][0]
    #     framedimraw = _fdim[1], _fdim[0]

    indicesborders = Improc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                (halfboxsizes[0], halfboxsizes[1]),
                                                framedimraw,
                                                flipxycenter=0)
    imin, imax, jmin, jmax = indicesborders

    if 0:
        print("indicesborders", indicesborders)

    # avoid to wrong indices when slicing the data
    imin, imax, jmin, jmax = Improc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                framedim=framedimraw)
    if 0:
        print("imin, imax, jmin, jmax", imin, imax, jmin, jmax)
    # new fast way to read specific area in file directly
    #print('imagefilename', imagefilename)
    datacrop = IOimage.readrectangle_in_image(imagefilename,
                                            xpic,
                                            ypic,
                                            halfboxsizes[0],
                                            halfboxsizes[1],
                                            dirname=None,
                                            CCDLabel=CCDLabel,
                                            stackimageindex=stackimageindex)

    return datacrop

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

def collectroisptp_singlefile(index, roicenters=None, prefix=None, folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None):
    """ collect max - min  (peak to peak) in some given rois centered on peaks defined in peaklist in 1 image  """
    pass



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
    filename = setfilename(prefix, index, CCDLabel)   
    imagefilename = os.path.join(folder, filename)

    maxvals = IOimage.getroismax(imagefilename, roicenters=roicenters,
                                 halfboxsize=(boxsize_row,boxsize_line), CCDLabel=CCDLabel)
    return maxvals

def collectroissum_singlefile(index, roicenters=None, prefix=None, folder =None,
                            boxsize_row=10,boxsize_line=10,CCDLabel=None):
    """ collect integrated intensity (sum of intensities)in some given rois centered on peaks defined in peaklist in 1 image
    
    
        
    :param index: int, image file index
    :param roicenters: list or array of 2d PIXELS position
    :param prefix: str, prefix of imagefilename (excluding folder path) excluding digits number of the index
    ('Cu_' for images named  'Cu_0020.tif')
    :param folder: str, path to folder containing all images
    :param boxsize_row, boxsize_line: half boxsize along x and y of the roi (can be a list?)
    :param CCDLabel: label of detector
    
    :return: array of max intensities: shape = (nbimages, nbpeaks)
    """
    filename = prefix+'%04d'%index + '.tif'   
    imagefilename = os.path.join(folder, filename)

    pixsums = IOimage.getroissum(imagefilename, roicenters=roicenters,
                                 halfboxsize=(boxsize_row,boxsize_line), CCDLabel=CCDLabel)
    return pixsums

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

    XYvals = IOimage.getroisXYmax(imagefilename, roicenters=roicenters,
                                   halfboxsize=(boxsize_row,boxsize_line),
                                   CCDLabel=CCDLabel)
    return XYvals

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

    XYvals = IOimage.getroiscenterofmass(imagefilename, roicenters=roicenters,
                                   halfboxsize=(boxsize_row,boxsize_line),
                                   CCDLabel=CCDLabel)
    return XYvals

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