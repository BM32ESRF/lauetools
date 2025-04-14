# -*- coding: utf-8 -*-
r"""
List of functions for Laue data (ASCII file and laue pattern image)  data visualization from laue analysis of large number of images during map
- PlotImage                : plot Laue pattern recorded in .tif file
- PlotPeakPos              : scatter plot of data in .dat or .cor file 
- PlotIndexedPeakPos       : scatter data in .fit file (indexed peaks position)
- PlotIndexedPeakPosWlabels: scatter data in .fit file (indexed peaks position) with hkl
- PeakShiftMap             : 2D plot of X and Y position of a reflection with given 'h k l'
- PeakIntensityMap
- EulerAngles              : 
- EulerAngles2D            : 
- StrainMap            : plot the 6 strain components for a scan
- StrainMapHistogram  :
- LatticeParamsMap

this module has been originally written by S. Bongiornio, S. Tardif, and B. De Goes Foschiani
in 2024-2025

Revised by J.S. Micha March 2025
"""
import os, sys
import fabio
#import h5py
from scipy.optimize import curve_fit

from typing import List, Tuple, Union
class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import LaueTools.IOLaueTools as IOLT
import LaueTools.generaltools as GT
from LaueTools.fitfilereader import parsed_fitfile, parsed_fitfileseries

def __gaussian__(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))



#with h5py.File('path/to/h5') as f:
#    xech = f['scan_nb/measurement/xech'][:]
#    yech = f['scan_nb/measurement/yech'][:]
#    
#    xech = ((xech - xech[0]) * 1e3).reshape(nb_rows,nb_cols)
#    yech = ((yech - yech[0]) * 1e3).reshape(nb_rows,nb_cols)


def PlotImage(imagepath: str, ROI: tuple = None, **kwargs) -> None:
    # "ROI = (Xcen, Ycen, boxsizex, boxsizey)"
    imdata = fabio.open(imagepath).data
    
    if ROI is not None:
        xindex1 = ROI[0] - (ROI[2] // 2)
        xindex2 = ROI[0] + (ROI[2] // 2)
        yindex1 = ROI[1] - (ROI[3] // 2)
        yindex2 = ROI[1] + (ROI[3] // 2)
                
        imdata = imdata[xindex1:xindex2, yindex1:yindex2]
    
    fig, ax = plt.subplots()
        
    im = ax.imshow(imdata, **kwargs)
    
    ax.set_xlabel('Pixel x')
    ax.set_ylabel('Pixel y')
    ax.set_title(imagepath.split('/')[-1])
    plt.colorbar(im)


def PlotPeakPos(filepathdat: str, size: tuple = None, frame:str='pixel', showindex:bool=False, figax=None,  **kwargs) -> tuple:
    with HidePrint():
        peaklist = IOLT.readfile_dat(filepathdat)
        
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
        ax.invert_yaxis()
    if size is not None:
        fig.set_size_inches(size[0], size[1])

    if frame=='pixel':
        x,y = peaklist[:,2:4].T
        xlabel, ylabel = 'pixel X','pixel Y'
        ax.set_ylim(2050,-50)
    elif frame == 'angles':
        x,y = peaklist[:,:2].T
        xlabel, ylabel = r'2$\theta$ (deg)',r'$\chi$ (deg)'
        
    ax.scatter(x,y, label = 'Peak position', cmap = plt.cm.inferno, c = np.arange(len(x), 0,-1), **kwargs)
    if showindex:
        for _i, (_x, _y) in enumerate(zip(x,y)):
            #ax.annote(x + 1, y + 1, "%d" % _i)
            ax.annotate("%d" % _i, xy=(_x, _y), xycoords='data')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_aspect('equal')
    ax.legend(loc = 'upper right')
    
    return fig, ax
    
def PlotIndexedPeakPos(filepathfit: str, ax = None, size: tuple = None, **kwargs) -> object:
    
    peak_dict =  parsed_fitfile(filepathfit).peak
    peak_pos  = np.array([[peak_dict[key][7], peak_dict[key][8]] for key in peak_dict.keys()], dtype = float)
    
    if ax is None:
        fig, ax = plt.subplots()
        if size is not None:
            fig.set_size_inches(size[0], size[1])
        # axis is to be inverted only when a new one is created
        # otherwise I expect the passed axis to be already inverted from PlotPeakPos function
        ax.invert_yaxis()
    
    ax.scatter(peak_pos[:,0], peak_pos[:,1], label = 'Indexed peak position', **kwargs)
    
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_aspect('equal')
    ax.legend(loc = 'upper right')
    
    return ax

def PlotIndexedPeakPosWlabels(peaks: dict, fontsize: int = 6, size: tuple = (8, 8), **kwargs):
    fig, ax = plt.subplots()
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    for key in peaks.keys():
        idx_posx = float(peaks[key][7])
        idx_posy = float(peaks[key][8])
        
        ax.scatter(idx_posx, idx_posy, **kwargs)
        
        #TO DO: IMPROVE THE DECISION BASED ON THE FRAMESIZE FROM CCDlabel IN DICTLAUETOOLS
        bottom_left_quadrant  = idx_posx <  1024 and idx_posy >= 1024
        bottom_right_quadrant = idx_posx >= 1024 and idx_posy >= 1024
        top_left_quadrant     = idx_posx <  1024 and idx_posy <  1024
        top_right_quadrant    = idx_posx >= 1024 and idx_posy <  1024
        
        xshift, yshift = 0, 0
        
        if bottom_left_quadrant:
            ha, va = 'right', 'bottom'
            xshift = -5
            yshift = -5
        elif bottom_right_quadrant:
            ha, va = 'left', 'bottom'
            xshift =  5
            yshift = -5
        elif top_left_quadrant:
            ha, va = 'right', 'top'
            xshift = -5
            yshift = +5
        elif top_right_quadrant:
            ha, va = 'left', 'top'
            xshift =  5
            yshift = +5
        
        ax.text(idx_posx + xshift, idx_posy + yshift, f'[{key}]', ha = ha, va = va, fontsize = fontsize)
        #t.set_bbox(dict(facecolor='white', alpha=0, linewidth=0, boxstyle='square,pad=1'))
    ax.invert_yaxis()
    ax.set_aspect('equal')
    

def PeakShiftMap(xech: np.ndarray, yech: np.ndarray, 
                 indexed_fileseries: parsed_fitfileseries, 
                 miller_index: str, 
                 space: str = 'pixel',
                 rel_pos: bool = False,
                 size: tuple = (13, 5), **kwargs) -> tuple:

    fig, ax = plt.subplots(1,2)
    
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    if space == 'pixel':
        data0 = indexed_fileseries.Xpos(miller_index).reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
        data1 = indexed_fileseries.Ypos(miller_index).reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
        
        title0, title1 = 'X pixel', 'Y pixel'
        
    elif space == 'twothetachi':
        data0 = indexed_fileseries.chi(miller_index).reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
        data1 = indexed_fileseries.twotheta(miller_index).reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
        
        title0, title1 = 'chi', '2theta'
    else:
        raise NameError("Invalid plot type. Accepted type arguments for 'space' are either 'pixel' or 'twothetachi'")
        
    if rel_pos:
        data0 -= np.nanmean(data0)
        data1 -= np.nanmean(data1)
        
    for axis, data, title in zip(ax, [data0, data1], [title0, title1]):
        im = axis.pcolormesh(xech, yech, data, **kwargs)
        axis.set_xlabel('Position [µm]')
        axis.set_ylabel('Position [µm]')
        axis.set_aspect('equal')
        axis.set_title(title)
        
        # create axis just for colorbar to give it the same height of the plot 
        cbarax = fig.add_axes([axis.get_position().x1 + 0.01, axis.get_position().y0, 0.025, axis.get_position().height])
        fig.colorbar(im, cax=cbarax)

    fig.suptitle(f'Peak shift map of the [{miller_index}] spot')
    
    return fig, ax

def PeakIntensityMap(xech: np.ndarray, yech: np.ndarray, 
                 indexed_fileseries: parsed_fitfileseries, 
                 miller_index: str, 
                 size: tuple = None, **kwargs) -> tuple:
    
    fig, ax = plt.subplots()
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    data = indexed_fileseries.intensity(miller_index).reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
    
    im = ax.pcolormesh(xech, yech, data, **kwargs)
    ax.set_xlabel('Position [µm]')
    ax.set_ylabel('Position [µm]')
    ax.set_aspect('equal')
    ax.set_title(f'Intensity map of the [{miller_index}] peak')
    
    # create axis just for colorbar to give it the same height of the plot 
    cbarax = fig.add_axes([ax.get_position().x1 + 0.015, ax.get_position().y0, 0.025, ax.get_position().height])
    fig.colorbar(im, cax=cbarax)
    
    
def EulerAngles(indexed_fileseries: parsed_fitfileseries, size: tuple = (21,6), **kwargs) -> tuple:
    fig, ax = plt.subplots(1,3)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    phi   = indexed_fileseries.EulerAngles[:,0]
    theta = indexed_fileseries.EulerAngles[:,1]
    psi   = indexed_fileseries.EulerAngles[:,2]
    
    titles = ['Phi', 'Theta', 'Psi']
    
    for axis, data, title in zip(ax, [phi, theta, psi], titles):
        axis.scatter(np.arange(0, indexed_fileseries.nb_files), data, **kwargs)
        axis.set_xlabel('Image index')
        axis.set_ylabel(title)
    
    fig.suptitle('Euler Angles')
    
    return fig, ax

def EulerAngles2D(xech: np.ndarray, yech: np.ndarray, 
                  indexed_fileseries: parsed_fitfileseries, 
                  size: tuple = (10,6), **kwargs) -> None:
    
    fig, ax = plt.subplots(1,3)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    phi   = indexed_fileseries.EulerAngles[:,0].reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
    theta = indexed_fileseries.EulerAngles[:,1].reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
    psi   = indexed_fileseries.EulerAngles[:,2].reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
    
    titles = ['Phi', 'Theta', 'Psi']
    
    for axis, data, title in zip(ax, [phi, theta, psi], titles):
        im = axis.pcolormesh(xech, yech, data, **kwargs)
        axis.set_xlabel('Position [µm]')
        axis.set_ylabel('Position [µm]')
        axis.set_aspect('equal')
        axis.set_title(title)
        
        # create axis just for colorbar to give it the same height of the plot 
        #cbarax = fig.add_axes([axis.get_position().x1 + 0.01, axis.get_position().y0, 0.025, axis.get_position().height])
        fig.colorbar(im)#, cax=cbarax)
    
    fig.suptitle('Euler Angles')

def PlotNumberIndexedSpots(indexed_fileseries: parsed_fitfileseries, 
                  size: tuple = (10,6), maskingcondition=None, **kwargs) -> None:
    
    fig, ax = plt.subplots()
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    data = indexed_fileseries.NumberOfIndexedSpots.reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)

    if maskingcondition is not None:
        if maskingcondition.shape != data.shape:
            print("Be careful! maskingcondition shape does not match with data shape", maskingcondition.shape, data.shape)
            return None, None
        data = ma.masked_where(maskingcondition, data)
    
    im = ax.imshow(data, **kwargs)
    ax.set_xlabel('Position [µm]')
    ax.set_ylabel('Position [µm]')
    ax.set_aspect('equal')
    ax.set_title(f'Nb of indexed spots')
    
    # create axis just for colorbar to give it the same height of the plot 
    cbarax = fig.add_axes([ax.get_position().x1 + 0.015, ax.get_position().y0, 0.025, ax.get_position().height])
    fig.colorbar(im, cax=cbarax)
    return fig, ax


def PlotMeanDevPixel(indexed_fileseries: parsed_fitfileseries, 
                  size: tuple = (10,6), maskingcondition=None, **kwargs) -> None:
    
    fig, ax = plt.subplots()
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    data = indexed_fileseries.MeanDevPixel.reshape(indexed_fileseries.nb_rows, indexed_fileseries.nb_cols)
    
    if maskingcondition is not None:
        if maskingcondition.shape != data.shape:
            print("Be careful! maskingcondition shape does not match with data shape", maskingcondition.shape, data.shape)
            return None, None
        data = ma.masked_where(maskingcondition, data)

    im = ax.imshow(data, **kwargs)
    ax.set_xlabel('Position [µm]')
    ax.set_ylabel('Position [µm]')
    ax.set_aspect('equal')
    ax.set_title(f'Mean Deviation (pix)')
    
    # create axis just for colorbar to give it the same height of the plot 
    cbarax = fig.add_axes([ax.get_position().x1 + 0.015, ax.get_position().y0, 0.025, ax.get_position().height])
    fig.colorbar(im, cax=cbarax)

    return fig, ax

def StrainMap(xech: np.ndarray, yech: np.ndarray, indexed_fileseries: parsed_fitfileseries, frame:str='crystal',
              multiplier: float = 1e4, scale: str = 'default', size: tuple = (21,10), maskingcondition=None, **kwargs) -> tuple:
    
    if frame == 'crystal':
        exx = indexed_fileseries.exx * multiplier
        eyy = indexed_fileseries.eyy * multiplier
        ezz = indexed_fileseries.ezz * multiplier
        exy = indexed_fileseries.exy * multiplier
        exz = indexed_fileseries.exz * multiplier
        eyz = indexed_fileseries.eyz * multiplier
    elif frame == 'sample':
        exx = indexed_fileseries.exx_sample * multiplier
        eyy = indexed_fileseries.eyy_sample * multiplier
        ezz = indexed_fileseries.ezz_sample * multiplier
        exy = indexed_fileseries.exy_sample * multiplier
        exz = indexed_fileseries.exz_sample * multiplier
        eyz = indexed_fileseries.eyz_sample * multiplier

    if maskingcondition is not None:
        if maskingcondition.shape != exx.shape:
            print('Be careful! maskingcondition has not the expected shape :', exx.shape)
            return None, None
        
        exx = ma.masked_where(maskingcondition, exx)
        eyy = ma.masked_where(maskingcondition, eyy)
        ezz = ma.masked_where(maskingcondition, ezz)
        exy = ma.masked_where(maskingcondition, exy)
        exz = ma.masked_where(maskingcondition, exz)
        eyz = ma.masked_where(maskingcondition, eyz)
    
    if scale == 'default':
        plotvmin = np.repeat(None, 6)
        plotvmax = np.repeat(None, 6)
    
    elif scale == 'mean3sigma':
        plotvmin = np.array([np.nanmean(exx) - 3*np.nanstd(exx),
                             np.nanmean(eyy) - 3*np.nanstd(eyy),
                             np.nanmean(ezz) - 3*np.nanstd(ezz),
                             np.nanmean(exy) - 3*np.nanstd(exy),
                             np.nanmean(exz) - 3*np.nanstd(exz),
                             np.nanmean(eyz) - 3*np.nanstd(eyz)])
        
        plotvmax = np.array([np.nanmean(exx) + 3*np.nanstd(exx),
                             np.nanmean(eyy) + 3*np.nanstd(eyy),
                             np.nanmean(ezz) + 3*np.nanstd(ezz),
                             np.nanmean(exy) + 3*np.nanstd(exy),
                             np.nanmean(exz) + 3*np.nanstd(exz),
                             np.nanmean(eyz) + 3*np.nanstd(eyz)])
        
    elif scale == 'uniform':
        plotvmin = np.nanmin([np.nanmin(exx), np.nanmin(eyy), np.nanmin(ezz), np.nanmin(exy), np.nanmin(exz), np.nanmin(eyz)])
        plotvmax = np.nanmax([np.nanmax(exx), np.nanmax(eyy), np.nanmax(ezz), np.nanmax(exy), np.nanmax(exz), np.nanmax(eyz)])
        
        plotvmax = np.max([np.abs(plotvmin), np.abs(plotvmax)])
        plotvmin = - plotvmax
        
        plotvmin = np.repeat(plotvmin, 6)
        plotvmax = np.repeat(plotvmax, 6)
    
    elif scale == 'other':
        try:
            plotvmin = kwargs.pop('vmin')
            plotvmax = kwargs.pop('vmax')
            
            plotvmin = np.repeat(plotvmin, 6)
            plotvmax = np.repeat(plotvmax, 6)
        except KeyError:
            raise KeyError("If you select scale = 'other' you must specify vmin and vmax")
    
    else:
        raise Exception("scale must be in the list ['default', 'mean3sigma', 'uniform', 'other']")
    
    titles = ['ε$_{xx}$', 'ε$_{yy}$', 'ε$_{zz}$', 'ε$_{xy}$', 'ε$_{xz}$', 'ε$_{yz}$']
    strain = [exx, eyy, ezz, exy, exz, eyz]
    
    fig, ax = plt.subplots(2, 3)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    xechstep = np.fabs(xech[1]-xech[0])
    yechstep = np.fabs(yech[1]-yech[0])

    print('xechstep',xechstep)
    print('yechstep',yechstep)
    print('deviatoric strain xx shape', exx.shape)
    fmtcoordfuncs = []
    for kk, _data in enumerate(strain):
        def make_fmtcoordfunc(data2D):  # function factory to ensure "early binding"
            def fmtcoord(x,y):
                return GT.format_getimageindex_pcolormesh(x, y, data2D=data2D, mapdims=data2D.shape,
                                                      xech_stepsize=xechstep,yech_stepsize=yechstep)
            return fmtcoord
        fmtcoordfuncs.append(make_fmtcoordfunc(_data))

    kk = 0
    for axidx, data, title, plotmin, plotmax in zip(np.ndindex(ax.shape), strain, titles, plotvmin, plotvmax):
        im = ax[axidx].pcolormesh(xech, yech, data, vmin = plotmin, vmax = plotmax, **kwargs)
        ax[axidx].set_xlabel('Position [µm]')
        ax[axidx].set_ylabel('Position [µm]')
        ax[axidx].set_title(title+f' (x{1/multiplier:.0E})')
        ax[axidx].set_aspect('equal')
        ax[axidx].title.set_size(12)
        ax[axidx].format_coord = fmtcoordfuncs[kk]
        kk+=1
        plt.colorbar(im, ax=ax[axidx])
    fig.suptitle(f'Deviatoric strain components in {frame} frame')
    fig.tight_layout()

    return fig, ax

def StrainMapHistogram(indexed_fileseries: parsed_fitfileseries, frame:str='crystal',
                       multiplier: float = 1e4, maskingcondition:bool=None, 
                       size: tuple = (18,10),
                       fit: bool = False, **kwargs) -> tuple:
    
    if frame == 'crystal':
        exx = (indexed_fileseries.exx).reshape(indexed_fileseries.nb_files) * multiplier
        eyy = (indexed_fileseries.eyy).reshape(indexed_fileseries.nb_files) * multiplier
        ezz = (indexed_fileseries.ezz).reshape(indexed_fileseries.nb_files) * multiplier
        exy = (indexed_fileseries.exy).reshape(indexed_fileseries.nb_files) * multiplier
        exz = (indexed_fileseries.exz).reshape(indexed_fileseries.nb_files) * multiplier
        eyz = (indexed_fileseries.eyz).reshape(indexed_fileseries.nb_files) * multiplier
    if frame == 'sample':
        exx = (indexed_fileseries.exx_sample).reshape(indexed_fileseries.nb_files) * multiplier
        eyy = (indexed_fileseries.eyy_sample).reshape(indexed_fileseries.nb_files) * multiplier
        ezz = (indexed_fileseries.ezz_sample).reshape(indexed_fileseries.nb_files) * multiplier
        exy = (indexed_fileseries.exy_sample).reshape(indexed_fileseries.nb_files) * multiplier
        exz = (indexed_fileseries.exz_sample).reshape(indexed_fileseries.nb_files) * multiplier
        eyz = (indexed_fileseries.eyz_sample).reshape(indexed_fileseries.nb_files) * multiplier

    if maskingcondition is not None:
        if maskingcondition.shape != exx.shape:
            print('Be careful! maskingcondition has not the expected shape :', exx.shape)
            return None, None
        
        exx = ma.masked_where(maskingcondition, exx)
        eyy = ma.masked_where(maskingcondition, eyy)
        ezz = ma.masked_where(maskingcondition, ezz)
        exy = ma.masked_where(maskingcondition, exy)
        exz = ma.masked_where(maskingcondition, exz)
        eyz = ma.masked_where(maskingcondition, eyz)
    
    strain  = [exx, eyy, ezz, exy, exz, eyz]
    xlabels = ['ε$_{xx}$', 'ε$_{yy}$', 'ε$_{zz}$', 'ε$_{xy}$', 'ε$_{xz}$', 'ε$_{yz}$']
    
    fig, ax = plt.subplots(2, 3)
    fig.subplots_adjust(hspace = 0.25, wspace = 0.2)

    fitgaussianresults= []
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    for axidx, data, xlabel in zip(np.ndindex(ax.shape), strain, xlabels):
        counts, bins, patches = ax[axidx].hist(data, **kwargs)
        
        titleplot = xlabel
        if fit:       
            # Compute the bins centers. Used when evaluating the fitting function (__gaussian__)
            bin_centers = bins[:-1] + np.diff(bins)/2
            # Compute the fit_params [A, mu, sigma], returned covariance is trashed   
            fit_params, _ = curve_fit(__gaussian__, bin_centers, counts, p0 = (50, 0, 2))
            
            # Plot result
            xlims = ax[axidx].get_xlim()
            xvals = np.linspace(xlims[0], xlims[1], 200)

            fitgaussianresults.append(fit_params)
            
            ax[axidx].plot(xvals, __gaussian__(xvals, *fit_params), color = 'red', linewidth = 2)
            titleplot += f'\nA = {fit_params[0]:.3f}, μ = {fit_params[1]:.3f}, σ = {fit_params[2]:.3f}'
            
    
        ax[axidx].set_xlabel(xlabel+f' (x{1/multiplier:.0E})')
        ax[axidx].set_ylabel('Counts')
        ax[axidx].set_title(titleplot)
        ax[axidx].title.set_size(10)
        
    fig.suptitle(f'Distribution strain components in {frame} frame')
    
    return fig, ax, fitgaussianresults

def LatticeParamsMap(xech: np.ndarray, yech: np.ndarray, indexed_fileseries: parsed_fitfileseries, 
              scale: str = 'default', size = (21,10), **kwargs) -> tuple:
    
    a = indexed_fileseries.a_prime
    b = indexed_fileseries.a_prime
    c = indexed_fileseries.a_prime
    alpha = indexed_fileseries.a_prime
    beta  = indexed_fileseries.a_prime
    gamma = indexed_fileseries.a_prime
    
    if scale == 'default':
        plotvmin = np.repeat(None, 6)
        plotvmax = np.repeat(None, 6)

    elif scale == 'other':
        try:
            plotvmin = kwargs.pop('vmin')
            plotvmax = kwargs.pop('vmax')
            
            plotvmin = np.repeat(plotvmin, 6)
            plotvmax = np.repeat(plotvmax, 6)
        except KeyError:
            raise KeyError("If you select scale = 'other' you must specify vmin and vmax")
    
    else:
        raise Exception("scale must be in the list ['default', 'mean3sigma', 'uniform', 'other']")
    
    titles = ['$a$', '$b$', '$c$', '$\alpha$', '$\beta$', '$\gamma$']
    
    latparams = [a,b,c, alpha, beta, gamma]
    
    fig, ax = plt.subplots(2, 3)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    
    for axidx, data, title, plotmin, plotmax in zip(np.ndindex(ax.shape), latparams, titles, plotvmin, plotvmax):
        im = ax[axidx].pcolormesh(xech, yech, data, vmin = plotmin, vmax = plotmax, **kwargs)
        ax[axidx].set_xlabel('Position [µm]')
        ax[axidx].set_ylabel('Position [µm]')
        ax[axidx].set_title(title)
        ax[axidx].set_aspect('equal')
        ax[axidx].title.set_size(16)

        plt.colorbar(im)
    
    return fig, ax