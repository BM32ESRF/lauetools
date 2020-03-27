import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import time as ttt
import numpy as np

import scipy.interpolate as sci

import scipy.ndimage as ndimage
import scipy.signal

import fit2Dintensity as fit2d
import readmccd as RMCCD
import IOLaueTools as IOLT
import generaltools as GT
import dict_LaueTools as DictLT

from .. import IOimagefile as IOimage
from .. import imageprocessing as ImProc

def Simulate_gaussianpeak(height,
                            center_x_slowdir,
                            center_y_fastdir,
                            width_x_slowdir,
                            width_y_fastdir,
                            slowdirection_range=(0, 20),
                            fastdirection_range=(0, 20)):
    """
    returns a 2D array with dimensions
    shape=(slow direction value, fast direction value)

    if pixel values are considered arranged as in a matrix,
    as pylab imshow plot

    -------->  fast direction  (2nd array indice)
    |
    |
    |
    |
    V  slow direction  (1rst array indice)

    """
    f1, f2 = fastdirection_range
    s1, s2 = slowdirection_range

    X, Y = np.mgrid[s1:s2, f1:f2]

    Z = fit2d.gaussian(
        height, center_x_slowdir, center_y_fastdir, width_x_slowdir, width_y_fastdir
    )(X, Y)

    #    print "X,shape", np.shape(X)
    #    print "Z.shape", np.shape(Z)

    return Z


def test_addimage():
    """  sum up three contiguous images
    newimage name will be prefixname+S+000futureindex+

    Au_mono_70N_P_ter_0100.mccd -> 0201
    """
    prefixname = "Au_mono_70N_P_ter_"
    suffixname = ".mccd"
    #    histo = np.histogram(data)

    futureindex = 0  # index for the new image
    # for k in range(0,383,3):  # 0,3,6
    for k in list(range(100, 201, 2)):
        
        filename1 = prefixname + IOimage.stringint(k, 4) + suffixname
        filename2 = prefixname + IOimage.stringint(k + 1, 4) + suffixname
        print("Sum of ", filename1, filename2)  # ,filename3
        data1 = IOimage.readoneimage(filename1)
        data2 = IOimage.readoneimage(filename2)
        # data3=readoneimage(filename3)
        datasum = data1 + data2  # +data3
        header = IOimage.readheader(filename1)
        outputfilename = prefixname + "S" + IOimage.stringint(futureindex, 4) + suffixname
        print("written in ", outputfilename)
        IOimage.writeimage(outputfilename, header, datasum)
        futureindex += 1


def test_VHR_ROI():
    """"reading of mar tiff format as produced by vhr photonics science
    + home conversion in marccd alike format
    """

    # shapeCCD=(n,m)  # opposite of fit2d displays in loading data  X 1 m  and Y 1 n
    framedim = (2671, 4008)
    filename = "Au_0290_mar.tiff"
    dataimage = IOimage.readoneimage(filename, framedim=framedim)

    dataimage = np.reshape(dataimage, framedim)

    print("Full frame peak search")

    # p.subplot(211)
    # p.imshow(dataimage-83,aspect='equal',
    # interpolation='nearest',
    # norm = LogNorm(vmin=200, vmax= 4096))

    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(filename,
                                                            CCDLabel="VHR_full",
                                                            PixelNearRadius=20,
                                                            removeedge=2,
                                                            IntensityThreshold=500,
                                                            boxsize=15,
                                                            position_definition=1,
                                                            verbose=1,
                                                            fit_peaks_gaussian=1,
                                                            xtol=0.001,
                                                            return_histo=1,
                                                            Saturation_value=4095,
                                                            Saturation_value_flatpeak=4095)

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist("Au_0290_mar_test_full",
                            Isorted,
                            overwrite=1,
                            initialfilename=filename,
                            comments="test, test")

    print(
        "\n************************************\npartial frame peak search\n***********************\n"
    )
    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(
        filename,
        CCDLabel="VHR_full",
        center=(2000, 1000),
        boxsizeROI=(200, 200),
        PixelNearRadius=20,
        removeedge=2,
        IntensityThreshold=500,
        boxsize=15,
        position_definition=1,
        verbose=1,
        fit_peaks_gaussian=1,
        xtol=0.001,
        return_histo=1,
        Saturation_value=4095,
        Saturation_value_flatpeak=4095,
    )

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist(
        "Au_0290_mar_test_ROI",
        Isorted,
        overwrite=1,
        initialfilename=filename,
        comments="test, test",
    )


def test_peaksearch_ROI_martiff():
    """
    test of peak search with ROI on martiff format

    """
    prefixfilename = "CdTe_I999_03Jul06_0200"
    suffix = ".mccd"

    inputfilename = prefixfilename + suffix
    print("Full frame peak search")
    Isorted, _, _, histo = RMCCD.PeakSearch(
        inputfilename,
        CCDLabel="MARCCD165",
        PixelNearRadius=10,
        removeedge=2,
        IntensityThreshold=1000,
        boxsize=15,
        position_definition=1,
        verbose=0,
        xtol=0.001,
        fit_peaks_gaussian=1,
        FitPixelDev=25,
    )
    p.bar(histo[1][: len(histo[0])], histo[0])
    # p.show()
    print("Isorted", Isorted)
    IOLT.writefile_Peaklist(
        prefixfilename + "full",
        Isorted,
        overwrite=1,
        initialfilename=inputfilename,
        comments="blah, blah",
    )

    print(
        "\n************************************\npartial frame peak search\n***********************\n"
    )
    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(
        inputfilename,
        CCDLabel="MARCCD165",
        center=(992, 985),
        boxsizeROI=(50, 160),
        PixelNearRadius=10,
        removeedge=2,
        IntensityThreshold=1000,
        boxsize=15,
        position_definition=1,
        verbose=1,
        xtol=0.001,
        fit_peaks_gaussian=1,
        FitPixelDev=25,
    )

    # p.bar(histo[1][:len(histo[0])],histo[0])
    # p.show()
    print("Isorted", Isorted)
    IOLT.writefile_Peaklist(
        prefixfilename + "ROI",
        Isorted,
        overwrite=1,
        initialfilename=inputfilename,
        comments="blah, blah",
    )


def example_Use_of_Bispline():

    X, Y = np.mgrid[0:10, 0:10]

    Z = fit2d.gaussian(100, 10, 10, 2, 2)(X, Y)
    tck, _, _, _ = sci.bisplrep(np.ravel(X), np.ravel(Y), np.ravel(Z), kx=4, ky=4, full_output=1)

    Zfit = sci.bisplev(np.arange(10), np.arange(10), tck)

    for k in np.arange(9):
        p.subplot(3, 3, k + 1)
        p.plot(Z[k], "bo - ")
        p.plot(Zfit[k], "ro - ")

    p.show()


def test_Approximate_Background():
    """
    test_Approximate_Background
    """
    X, Y = np.mgrid[0:10, 0:20]
    import pylab as p

    Z = fit2d.gaussian(100, 2, 13, 3, 5)(X, Y)
    tck, _, _, _ = sci.bisplrep(np.ravel(X), np.ravel(Y), np.ravel(Z), kx=4, ky=4, full_output=1)
    print(Z.shape)
    print(X.shape)
    print(Y.shape)

    Zfit = sci.bisplev(np.arange(10), np.arange(20), tck)

    # for k in np.arange(9):
    # p.subplot(3,3,k+1)
    # p.plot(Z[k],'bo - ')
    # p.plot(Zfit[k],'ro - ')

    p.imshow(Z)
    p.contour(Zfit, cmap=GT.COPPER)

    p.show()


def test_Approximate_Background_mccdimage_spline():
    """
    with spline: does not work

    """
    filename = "CdTe_I999_03Jul06_0200.mccd"

    _, dataimage = IOimage.readoneimage_full(filename)

    mini = np.amin(dataimage[dataimage > 0])

    sampling = 50

    Xin, Yin = np.mgrid[0:2048:sampling, 0:2048:sampling]
    # data_gauss = fit2d.gaussian(100, 1024, 1024, 500, 500)(Xin, Yin)

    dataimage = dataimage[::sampling, ::sampling]

    cond_circle = (Xin - 1023) ** 2 + (Yin - 1023) ** 2 <= 1024 ** 2

    dataimage_bis = np.where(cond_circle, dataimage, mini)

    #    Z=fit2d.gaussian(100,25,50,30,30)(Xin,Yin)

    print(dataimage_bis.shape)
    print(Xin.shape)
    print(Yin.shape)
    tck, _, _, msg = sci.bisplrep(np.ravel(Xin),
                                    np.ravel(Yin),
                                    np.ravel(dataimage_bis),
                                    kx=3,
                                    ky=3,
                                    nxest=100,
                                    nyest=100,
                                    s=10000,
                                    full_output=1)

    # print "tck",tck
    print("msg", msg)

    Zfit = sci.bisplev(np.arange(0, 2048, sampling), np.arange(0, 2048, sampling), tck)

    # for k in np.arange(9):
    # p.subplot(3,3,k+1)
    # p.plot(Z[k],'bo - ')
    # p.plot(Zfit[k],'ro - ')

    p.imshow(dataimage)

    p.contour(Zfit, cmap=GT.COPPER)

    p.show()


def test_Approximate_Background_mccdimage_gaussian():

    """
    with spline: does not work

    """
    filename = "CdTe_I999_03Jul06_0200.mccd"
    # filename = 'Wmap_blanc_11Sep08_d0_5MPa_0000.mccd'
    _, dataimage = IOimage.readoneimage_full(filename)

    mini = np.amin(dataimage[dataimage > 0])
    print("non zero minimum value", mini)

    histo = np.histogram(dataimage, bins=50, range=(1, 200))
    print(histo)
    print(len(histo[0]))
    print(len(histo[1]))
    p.subplot(211)
    p.bar(histo[1][1:-1], histo[0][1:], width=5)
    # implies mini = 50

    mini = 10
    sampling = 30

    Xin, Yin = np.mgrid[0:2048:sampling, 0:2048:sampling]
    # data_gauss = fit2d.gaussian(100, 1024, 1024, 500, 500)(Xin, Yin)
    print("Xin", Xin)

    dataimage = dataimage[::sampling, ::sampling]

    print("dataimage.shape", dataimage.shape)

    cond_circle = (Xin - 1023) ** 2 + (Yin - 1023) ** 2 <= 1024 ** 2

    dataimage_bis = np.where(cond_circle, dataimage, mini)

    print("dataimage_bis.shape", dataimage_bis.shape)

    start_baseline = mini
    start_amplitude = 100
    start_j = 1024 // sampling
    start_i = 1024 // sampling
    start_sigma1 = 300 // sampling
    start_sigma2 = 300 // sampling
    start_anglerot = 0

    startingparams = [start_baseline,
                    start_amplitude,
                    start_j,
                    start_i,
                    start_sigma1,
                    start_sigma2,
                    start_anglerot]

    params, _, _, _ = fit2d.gaussfit(dataimage_bis,
                                                err=None,
                                                params=startingparams,
                                                autoderiv=1,
                                                return_all=1,
                                                circle=0,
                                                rotate=1,
                                                vheight=1)

    print("\n *****fitting results ************\n")
    print(params)
    print("background intensity:            %.2f" % params[0])
    print("Peak amplitude above background        %.2f" % params[1])
    print("pixel position (X)            %.2f" % (params[3] * sampling))
    print("pixel position (Y)            %.2f" % (params[2] * sampling))
    print("std 1,std 2 (pix)            ( %.2f , %.2f )"
        % (params[4] * sampling, params[5] * sampling))
    print("e=min(std1,std2)/max(std1,std2)        %.3f"
        % (min(params[4], params[5]) / max(params[4], params[5])))
    print("Rotation angle (deg)            %.2f" % (params[6] % 360))
    print("************************************\n")
    print(params)
    inpars_res = params
    fitdata = fit2d.twodgaussian(inpars_res, 0, 1, 1)
    p.subplot(212)
    p.imshow(dataimage_bis, interpolation="nearest")

    p.contour(fitdata(*np.indices(dataimage_bis.shape)), cmap=GT.COPPER)

    p.show()

    return dataimage_bis


def test_filtereffect():

    filename = "CdTe_I999_03Jul06_0200.mccd"
    pilimage, _ = IOimage.readoneimage_full(filename)

    im8bit = ImProc.to8bits(pilimage)[0]
    im1 = im8bit.filter(ImageFilter.MinFilter)
    im2 = im8bit.filter(ImageFilter.BLUR)
    im3 = im8bit.filter(ImageFilter.SMOOTH_MORE)
    im4 = im8bit.filter(ImageFilter.SMOOTH)

    im1.save("im1.TIFF")
    im2.save("im2.TIFF")
    im3.save("im3.TIFF")
    im4.save("im4.TIFF")


def shiftarrays(Data_array, n, dimensions=1):
    """
    1D
    returns 3 arrays corresponding to shifted arrays by n in two directions and original one
    2D
    returns 5 arrays corresponding to shifted arrays by n in two directions and original one

    these arrays are ready for comparison with eg np.greater

    .. note:: readmccd.localmaxima is better
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



def test_localmaxima():
    bb = np.array([[0, 1, 2, 0, 5, 1, 0, 2, 0],
                    [1, 3, 0, 2, 2, 6, 5, 8, 2],
                    [2, 4, 0, 1, 6, 3, 4, 2, 1],
                    [4, 5, 6, 9, 4, 6, 5, 3, 0],
                    [2, 2, 5, 6, 4, 3, 4, 5, 1],
                    [2, 0, 2, 2, 2, 2, 2, 2, 1],
                    [4, 2, 0, 1, 3, 2, 5, 2, 2]])

    aa = np.array([0, 1, 3, 7, 12, 25, 18, 20, 10, 16, 19, 12, 6, 9, 5, 3, 2])
    c, alll, allr = ImProc.shiftarrays_accum(aa, 2, dimensions=1, diags=0)

    flag = np.greater(c, alll[0])

    for elem in allr + alll[1:]:
        flag = flag * np.greater(c, elem)

    for elem in alll + [c] + allr:
        print(elem)

    peaklist = np.nonzero(flag)

    print("value local max", c[peaklist])

    ImProc.localmaxima(bb, 2, verbose=1, diags=0)

    dd = np.array(
        [
            [0, 1, 2, 0, 5, 1, 0, 2, 0],
            [1, 3, 0, 2, 2, 6, 5, 8, 2],
            [2, 4, 0, 50, 50, 3, 4, 2, 1],
            [4, 5, 6, 50, 50, 6, 5, 3, 0],
            [2, 2, 5, 50, 50, 3, 4, 5, 1],
            [2, 0, 2, 2, 2, 2, 2, 2, 1],
            [4, 2, 0, 1, 3, 2, 5, 2, 2],
        ]
    )
    print(
        "testing saturation... central peak is not detected because the top has at least two equal pixel"
    )
    ImProc.localmaxima(dd, 2, verbose=1, diags=0)

    dd = np.array(
        [
            [0, 1, 2, 0, 5, 1, 0, 2, 0],
            [1, 3, 0, 2, 2, 6, 5, 8, 2],
            [2, 4, 0, 49, 48, 3, 4, 2, 1],
            [4, 5, 6, 50, 49, 6, 5, 3, 0],
            [2, 2, 5, 48, 49, 3, 4, 5, 1],
            [2, 0, 2, 2, 2, 2, 2, 2, 1],
            [4, 2, 0, 1, 3, 2, 5, 2, 2],
        ]
    )
    print(
        "testing saturation... central peak is detected (equal pixel are not the most intense)"
    )
    ImProc.localmaxima(dd, 2, verbose=1, diags=1)

    dd = np.array(
        [
            [0, 1, 2, 0, 5, 1, 0, 2, 0],
            [1, 3, 0, 2, 2, 6, 5, 8, 2],
            [2, 4, 0, 49, 48, 46, 48, 2, 1],
            [4, 5, 50, 50, 49, 46, 49, 3, 0],
            [2, 2, 5, 48, 49, 44, 48, 5, 1],
            [2, 0, 2, 2, 2, 2, 2, 2, 1],
            [4, 2, 0, 1, 3, 2, 5, 2, 2],
        ]
    )
    print(
        "testing saturation... central peak is detected (equal pixel are not the most intense)"
    )
    ImProc.localmaxima(dd, 4, verbose=1, diags=1)


def test_average_Images():
    """
    average images in windows OS
    """
    filepath = "D:\\Documents and Settings\\or208865\\Bureau\\AAA\\"
    fileprefix = "Ge_WB_14sep_d0_500MPa_"

    prefixname = filepath + fileprefix

    outfile = prefixname + "ave_0to9.mccd"

    IOimage.Add_Images(prefixname, 0, 9, plot=0, writefilename=outfile)


def test_VHR():
    # reading of mar tiff format as produced by vhr photonics science + home conversion in marccd alike format

    # shapeCCD=(n,m)  # opposite of fit2d displays in loading data  X 1 m  and Y 1 n
    framedim = (2671, 4008)
    filename = "Au_0290_mar.tiff"
    dataimage = IOimage.readoneimage(filename, framedim=framedim)

    dataimage = np.reshape(dataimage, framedim)

    p.subplot(211)
    p.imshow(dataimage - 83,
        aspect="equal",
        interpolation="nearest",
        norm=LogNorm(vmin=200, vmax=4096))

    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(
        filename,
        CCDLabel="VHR_full",
        PixelNearRadius=20,
        removeedge=2,
        IntensityThreshold=500,
        boxsize=15,
        position_definition=1,
        verbose=1,
        fit_peaks_gaussian=1,
        xtol=0.001,
        return_histo=1,
        Saturation_value=4095,
        Saturation_value_flatpeak=4095,
    )

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist(
        "Au_0290_mar_test",
        Isorted,
        overwrite=1,
        initialfilename=filename,
        comments="test, test",
    )


def test_VHR_crop():

    framedim = (2594, 3764)
    offsetheader = 4096
    fliprotvhr = "vhr"
    filename = "dia10_0500_mar.tiff"

    centers = np.array([[1920, 3032], [1024, 1024]])
    boxsize = 10

    dataimages = IOimage.readoneimage_manycrops(
        filename, centers, boxsize, CCDLabel="VHR_diamond"
    )

    dataimage = dataimages[0]

    from matplotlib.colors import LogNorm

    p.subplot(111)
    p.imshow(
        dataimage - 83,
        aspect="equal",
        interpolation="nearest",
        norm=LogNorm(vmin=200, vmax=4096),
    )

    p.show()


def test_VHR_2():
    # reading of mar tiff format as produced by vhr photonics science + home conversion in marccd alike format

    # shapeCCD=(n,m)  # opposite of fit2d displays in loading data  X 1 m  and Y 1 n
    framedim = (2594, 3764)
    offsetheader = 4096
    fliprotvhr = "vhr"
    filename = "dia10_0500_mar.tiff"
    dataimage = IOimage.readoneimage(filename, framedim=framedim, offset=offsetheader)

    dataimage = np.reshape(dataimage, framedim)

    dataimage = np.rot90(dataimage, k=3)

    #    from matplotlib.colors import LogNorm
    #    p.subplot(111)
    #    p.imshow(dataimage - 83, aspect='equal',
    #                        interpolation='nearest',
    #                        norm=LogNorm(vmin=200, vmax=4096))
    #
    #    p.show()

    #    Isorted, fitpeak, localpeak, histo = PeakSearch(filename,CCDLabel ='VHR_diamond',
    #                                            PixelNearRadius=20,
    #                                            FitPixelDev=25,
    #                                            removeedge=2,
    #                                            local_maxima_search_method=0,
    #                                            IntensityThreshold=500,
    #                                            boxsize=15,
    #                                            position_definition=1,
    #                                            verbose=1,
    #                                            fit_peaks_gaussian=1,
    #                                            xtol=0.001,
    #                                            return_histo=1,
    #                                            Saturation_value=4095,
    #                                            Saturation_value_flatpeak=4095)

    #    # local_maxima_search_method=1  not implemented for non squared CCD frame
    #    Isorted, fitpeak, localpeak, histo = PeakSearch(filename,CCDLabel ='VHR_diamond',
    #                                            PixelNearRadius=20,
    #                                            FitPixelDev=25,
    #                                            removeedge=2,
    #                                            local_maxima_search_method=1,
    #                                            IntensityThreshold=2000,
    #                                            boxsize=15,
    #                                            position_definition=1,
    #                                            verbose=1,
    #                                            fit_peaks_gaussian=0,
    #                                            xtol=0.001,
    #                                            return_histo=1,
    #                                            Saturation_value=4095,
    #                                            Saturation_value_flatpeak=4095)

    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(
        filename,
        CCDLabel="VHR_diamond",
        PixelNearRadius=20,
        FitPixelDev=25,
        removeedge=2,
        local_maxima_search_method=2,
        IntensityThreshold=200,
        thresholdConvolve=1500,
        paramsHat=(6, 10, 3),
        boxsize=60,
        position_definition=1,
        verbose=1,
        fit_peaks_gaussian=0,
        xtol=0.001,
        return_histo=1,
        Saturation_value=4095,
        Saturation_value_flatpeak=4095,
    )

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist(
        "VHR2test",
        Isorted,
        overwrite=1,
        initialfilename=filename,
        comments="test, test",
    )


def test_PSL_LAUE_INES():
    # reading image made by PSL LaueImaging camera at CEA-INAC  Bat D5

    # shapeCCD=(n,m)  # opposite of fit2d displays in loading data  X 1 m  and Y 1 n
    offsetheader = 110
    # fliprotvhr = 'no'
    #    framedim = (1970, 1290)
    #    filename = 'bidon - 5.tif'
    LAUEIMAGING_DATA_FORMAT = "uint8"

    framedim = (985, 645)  # as seen by general viewer
    framedim = (645, 985)
    filename = "bidon - 8.tif"

    dataimage = IOimage.readoneimage(filename,
                                    framedim=framedim,
                                    offset=offsetheader,
                                    formatdata=LAUEIMAGING_DATA_FORMAT)

    dataimage = np.reshape(dataimage, framedim)

    from matplotlib.colors import LogNorm

    ax = p.subplot(111)
    ax.imshow(dataimage, aspect="equal", interpolation="nearest", norm=LogNorm(vmin=0.0001, vmax=255))

    numrows, numcols = dataimage.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = dataimage[row, col]
            return "x = % 1.4f, y = % 1.4f, z = % 1.4f" % (x, y, z)
        else:
            return "x = % 1.4f, y = % 1.4f" % (x, y)

    ax.format_coord = format_coord

    p.show()

    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(filename,
                                                        CCDLabel="LaueImaging",
                                                        PixelNearRadius=20,
                                                        FitPixelDev=5,
                                                        removeedge=2,
                                                        local_maxima_search_method=0,
                                                        IntensityThreshold=120,
                                                        boxsize=15,
                                                        position_definition=1,
                                                        verbose=1,
                                                        fit_peaks_gaussian=1,
                                                        xtol=0.001,
                                                        return_histo=1,
                                                        Saturation_value=65000,
                                                        Saturation_value_flatpeak=65000)

    #    # local_maxima_search_method=1  not implemented for non squared CCD frame
    #    Isorted, fitpeak, localpeak, histo = PeakSearch(filename,CCDLabel='LaueImaging',
    #                                            framedim=framedim,
    #                                            offset=offsetheader,
    #                                            formatdata='uint8'
    #                                            fliprot=fliprotvhr,
    #                                            PixelNearRadius=20,
    #                                            FitPixelDev=25,
    #                                            removeedge=2,
    #                                            local_maxima_search_method=1,
    #                                            IntensityThreshold=2000,
    #                                            boxsize=15,
    #                                            position_definition=1,
    #                                            verbose=1,
    #                                            fit_peaks_gaussian=0,
    #                                            xtol=0.001,
    #                                            return_histo=1,
    #                                            Saturation_value=4095,
    #                                            Saturation_value_flatpeak=4095)

    #    Isorted, fitpeak, localpeak, histo = PeakSearch(filename,CCDLabel='LaueImaging',
    #                                            framedim=framedim,
    #                                            offset=offsetheader,
    #                                            formatdata=LAUEIMAGING_DATA_FORMAT,
    #                                            fliprot=fliprotvhr,
    #                                            PixelNearRadius=10,
    #                                            FitPixelDev=2,
    #                                            removeedge=2,
    #                                            local_maxima_search_method=2,
    #                                            IntensityThreshold=50,
    #                                            thresholdConvolve=1400,
    #                                            paramsHat=(4, 5, 2),
    #                                            boxsize=15,
    #                                            position_definition=1,
    #                                            verbose=1,
    #                                            fit_peaks_gaussian=1,
    #                                            xtol=0.001,
    #                                            return_histo=1,
    #                                            Saturation_value=255,
    #                                            Saturation_value_flatpeak=255)

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist("VHR2test", Isorted, overwrite=1, initialfilename=filename,
                                                                            comments="test, test")


def test_PSL_LAUE_INES_16bits():
    # reading image made by PSL LaueImaging camera at CEA-INAC  Bat D5

    # shapeCCD=(n,m)  # opposite of fit2d displays in loading data  X 1 m  and Y 1 n
    offsetheader = 110
    # fliprotvhr = 'no'
    framedim = (1970, 1290)  # as seen by general viewer
    framedim = (1290, 1970)
    filename = "bidon - 5.tif"
    LAUEIMAGING_DATA_FORMAT = "uint16"

    dataimage = IOimage.readoneimage(filename, framedim=framedim, offset=offsetheader,
                                                            formatdata=LAUEIMAGING_DATA_FORMAT)

    dataimage = np.reshape(dataimage, framedim)

    from matplotlib.colors import LogNorm

    ax = p.subplot(111)
    ax.imshow(dataimage, aspect="equal", interpolation="nearest",
                                                                norm=LogNorm(vmin=0.0001, vmax=255))

    numrows, numcols = dataimage.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = dataimage[row, col]
            return "x = % 1.4f, y = % 1.4f, z = % 1.4f" % (x, y, z)
        else:
            return "x = % 1.4f, y = % 1.4f" % (x, y)

    ax.format_coord = format_coord

    p.show()

    Isorted, _, _, _ = RMCCD.PeakSearch(filename,
                                    CCDLabel="LaueImaging",
                                    PixelNearRadius=10,
                                    FitPixelDev=2,
                                    removeedge=2,
                                    local_maxima_search_method=2,
                                    IntensityThreshold=10,
                                    thresholdConvolve=12000,
                                    paramsHat=(6, 8, 4),
                                    boxsize=15,
                                    position_definition=1,
                                    verbose=1,
                                    fit_peaks_gaussian=1,
                                    xtol=0.01,
                                    return_histo=1,
                                    Saturation_value=65000,
                                    Saturation_value_flatpeak=65000)

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist("VHR2test", Isorted, overwrite=1, initialfilename=filename,
                                                                            comments="test, test")


def test_peak_search_file_series():

    print("real sample : peak search on file series")

    filepath = "E:\\2008\\Sep08\\14sep08\\"
    fileprefix = "Wmap_WB_14sep_d0_500MPa_"
    filesuffix = ".mccd"
    imstart, imend = 45, 45
    indimg = list(range(imstart, imend + 1, 1))
    #    numimg = len(indimg)
    #    toto = np.zeros((numimg,3), dtype=float)
    prefixname = filepath + fileprefix

    #    j = 0
    for kk in indimg:
        filename = prefixname + IOimage.stringint(kk, 4) + filesuffix
        print(filename)
        prefix = ("D:\Documents and Settings\or208865\Bureau\AAA\AA\\toto_"
            + IOimage.stringint(kk, 4))
        print("prefix ", prefix)
        commentaire = "LT rev 437 \n# PixelNearRadius=5, removeedge=2, IntensityThreshold=500, boxsize=5,\n# \
            position_definition=1, fit_peaks_gaussian=1, xtol=0.001, FitPixelDev=2.0 \n"
        # print commentaire

        time_0 = ttt.time()

        Isorted, fitpeak, localpeak = RMCCD.PeakSearch(filename,
                                                        CCDLabel="MARCCD165",
                                                        PixelNearRadius=5,
                                                        removeedge=2,
                                                        IntensityThreshold=500,
                                                        boxsize=5,
                                                        position_definition=1,
                                                        verbose=1,
                                                        fit_peaks_gaussian=1,
                                                        xtol=0.001,
                                                        return_histo=0,
                                                        FitPixelDev=2.0)

        IOLT.writefile_Peaklist(
            prefix, Isorted, overwrite=1, initialfilename=filename, comments=commentaire)

        print("peak list written in %s" % (prefix + ".dat"))
        print("execution time: %.2f sec" % (ttt.time() - time_0))


def test_Peaksearch_Ge(filename="Ge_blanc_0000.mccd"):

    time_0 = ttt.time()

    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(filename,
                                                        CCDLabel="MARCCD165",
                                                        PixelNearRadius=20,
                                                        removeedge=2,
                                                        IntensityThreshold=200,
                                                        boxsize=15,
                                                        position_definition=1,
                                                        verbose=0,
                                                        fit_peaks_gaussian=1,
                                                        xtol=0.001,
                                                        return_histo=1)

    prefix, file_extension = filename.split(".")
    IOLT.writefile_Peaklist(
        prefix + "_c", Isorted, overwrite=1, initialfilename=filename, comments="test")

    print("peak list written in %s" % (prefix + "_c.dat"))
    print("execution time: %.2f sec" % (ttt.time() - time_0))


def test_fast_peaksearch(filename="Ge_blanc_0000.mccd", t=1000):
    """
    test fast peak search on Ge by adding a hot pixel in (1024,1024)
    
    """
    import pylab as p

    time_0 = ttt.time()

    # ----------------------
    f = open(filename, "rb")
    myheader = f.read(4096)
    f.close()

    # adding a single hot pixel -------------------------
    f = open("fakemccd_0000.mccd", "wb")
    # f.write(myheader)
    # d = readoneimage(filename)
    # d[2048*2048/2]=65000
    # f.write(d)
    f.write(myheader)
    d = RMCCD.readoneimage(filename)
    d[2048 * 1024 + 1024] = 65000
    #    scipy.io.numpyio.fwrite(f, 2048 * 2048, d)
    d.tofile(f)  # this method

    f.close()
    # ---------------------------------------------------

    # d = readoneimage(filename).reshape((2048,2048))
    d = d.reshape((2048, 2048))

    t1 = ttt.time()
    print("Read frame, execution time: %.2f sec" % (t1 - time_0))
    aa = ImProc.ConvolvebyKernel(d, peakVal=4, boxsize=5, central_radius=2)

    print("result of convolve shape", aa.shape)

    thraa = np.where(aa > t, 1, 0)

    t2 = ttt.time()
    print("Convolve, execution time: %.2f sec" % (t2 - t1))
    ll, nf = ndimage.label(thraa, structure=np.ones((3, 3)))

    t3 = ttt.time()
    print("Label cluster, execution time: %.2f sec" % (t3 - t2))
    print("nb of peaks", nf)

    meanpos = np.array(
        ndimage.measurements.center_of_mass(thraa, ll, np.arange(1, nf + 1)),
        dtype=float,
    )
    meanpos = np.fliplr(meanpos)

    print("meanpos", meanpos)
    t4 = ttt.time()
    print("Mean pos of cluster, execution time: %.2f sec" % (t4 - t3))
    ax = p.subplot(111)
    ax.imshow(np.log(d), interpolation="nearest")

    from matplotlib.patches import Circle

    for po in meanpos:

        large_circle = Circle(po, 7, fill=False, color="b")
        center_circle = Circle(po, 0.5, fill=True, color="r")
        ax.add_patch(large_circle)
        ax.add_patch(center_circle)

    numrows, numcols = d.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = d[row, col]
            return "x = % 1.4f, y = % 1.4f, z = % 1.4f" % (x, y, z)
        else:
            return "x = % 1.4f, y = % 1.4f" % (x, y)

    ax.format_coord = format_coord

    p.show()

    return meanpos, d


def test_filter_convolve(filename="CdTe_I999_03Jul06_0200.mccd"):
    """
    read data, filter, find blobs (no fitdata) and display 

    """
    import pylab as p

    import numpy as np
    import scipy.ndimage
    import scipy.signal

    filename = "CdTe_I999_03Jul06_0200.mccd"
    # filename = 'Ge_blanc_0000.mccd'

    # image = scipy.misc.lena().astype(float32)
    time_0 = ttt.time()

    d = RMCCD.readoneimage(filename).reshape((2048, 2048))

    t1 = ttt.time()
    print("Read frame, execution time: %.2f sec" % (t1 - time_0))

    # image  = d[1400:1500,700:800]
    image = d

    # A very simple and very narrow highpass filter
    # kernel = np.array([[-1, -1, -1],
    # [-1,  8, -1],
    # [-1, -1, -1]])
    # highpass_3x3 = scipy.ndimage.convolve(image, kernel)

    # # A slightly "wider", but sill very simple highpass filter
    kernel = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 2, 1, -1],
            [-1, 2, 4, 2, -1],
            [-1, 1, 2, 1, -1],
            [-1, -1, -1, -1, -1],
        ]
    )
    # highpass_5x5 = scipy.ndimage.convolve(image, kernel)

    # Another way of making a highpass filter is to simply subtract a lowpass
    # filtered image from the original. Here, we'll use a simple gaussian filter
    # to "blur" (i.e. a lowpass filter) the original.
    lowpass = scipy.ndimage.gaussian_filter(image, 11)
    gauss_highpass = image - lowpass

    print("high pass filtering completed")

    lowpass2 = scipy.ndimage.convolve(gauss_highpass, kernel)

    Imax = 60000
    Imin = 2000

    print("removing noise")
    cond = np.logical_and(lowpass2 > Imin, lowpass2 < Imax)

    matmeas = np.where(cond, 1, 0)

    t2 = ttt.time()
    print("convolution, execution time: %.2f sec" % (t2 - t1))

    ll, nf = ndimage.label(matmeas)  # , structure=np.ones((3,3)))
    meanpos = np.array(
        scipy.ndimage.measurements.center_of_mass(matmeas, ll, np.arange(1, nf + 1)),
        dtype=float,
    )
    meanpos = np.fliplr(meanpos)

    t3 = ttt.time()
    print("finding blobs, measurements, execution time: %.2f sec" % (t3 - t2))

    print("Found %d blobs" % len(meanpos))
    print("Total execution time: %.2f sec" % (t3 - time_0))

    if len(meanpos) < 4000:
        if 0:
            # plotting ----------to find procedure-------------------------
            fig = p.figure()
            ax = fig.add_subplot(221)
            ax.imshow(
                gauss_highpass, interpolation="nearest"
            )  # , r'Gaussian Highpass, $\sigma = 3 pixels$')

            numrows, numcols = gauss_highpass.shape

            def format_coord(x, y):
                col = int(x + 0.5)
                row = int(y + 0.5)
                if col >= 0 and col < numcols and row >= 0 and row < numrows:
                    z = gauss_highpass[row, col]
                    return "x=%1.4f, y=%1.4f, z=%1.4f" % (x, y, z)
                else:
                    return "x=%1.4f, y=%1.4f" % (x, y)

            ax.format_coord = format_coord

            ax2 = fig.add_subplot(222)
            cl_data = np.where(lowpass2 > Imax, 0, lowpass2)
            ax2.imshow(cl_data, interpolation="nearest")

            def format_coord2(x, y):
                col = int(x + 0.5)
                row = int(y + 0.5)
                if col >= 0 and col < numcols and row >= 0 and row < numrows:
                    z = cl_data[row, col]
                    return "x=%1.4f, y=%1.4f, z=%1.4f" % (x, y, z)
                else:
                    return "x=%1.4f, y=%1.4f" % (x, y)

            ax2.format_coord = format_coord2

            from matplotlib.patches import Circle

            ax3 = fig.add_subplot(223)
            ax3.imshow(np.log(image), interpolation="nearest")
            # PointToPlot[:,:2]= self.peaklistPixels[:,:2] - np.array([1,1])
            for po in meanpos:

                large_circle = Circle(po, 7, fill=False, color="b")
                center_circle = Circle(po, 0.5, fill=True, color="r")
                ax3.add_patch(large_circle)
                ax3.add_patch(center_circle)

            p.show()
        if 1:  # plotting final results
            fig = p.figure()
            ax = fig.add_subplot(111)

            numrows, numcols = image.shape

            def format_coord(x, y):
                col = int(x + 0.5)
                row = int(y + 0.5)
                if col >= 0 and col < numcols and row >= 0 and row < numrows:
                    z = image[row, col]
                    return "x=%1.4f, y=%1.4f, z=%1.4f" % (x, y, z)
                else:
                    return "x=%1.4f, y=%1.4f" % (x, y)

            ax.format_coord = format_coord

            from matplotlib.patches import Circle

            ax.imshow(np.log(image), interpolation="nearest")
            # PointToPlot[:,:2]= self.peaklistPixels[:,:2] - np.array([1,1])
            for po in meanpos:

                large_circle = Circle(po, 7, fill=False, color="b")
                center_circle = Circle(po, 0.5, fill=True, color="r")
                ax.add_patch(large_circle)
                ax.add_patch(center_circle)

            p.show()


def test_image_singlepeak(filename="Ge_blanc_0000.mccd"):
    """
    read marccd file add a peak and perform a peak search
    """
    CCDLABEL = "MARCCD165"

    fliprot = DictLT.dict_CCD[CCDLABEL][3]
    framedim = DictLT.dict_CCD[CCDLABEL][0]
    formatdata = DictLT.dict_CCD[CCDLABEL][5]
    offset = DictLT.dict_CCD[CCDLABEL][4]

    dataimage = IOimage.readoneimage(
        filename, framedim=framedim, offset=offset, formatdata=formatdata
    )

    header = IOimage.readheader(filename, offset=offset)

    dataimage = np.reshape(dataimage, framedim)

    # build a peak
    mypeak = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 10, 15, 10, 0],
            [0, 15, 20, 15, 0],
            [0, 10, 15, 10, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    pospeak = [1023, 600]

    j, i = pospeak

    dataimage[i - 2 : i + 3, j - 2 : j + 3] = 1000 * mypeak

    newfilename = "Ge_blanc_0000_test.mccd"

    IOimage.writeimage(newfilename, header, np.ravel(dataimage), dataformat=np.uint16)


def test_edf(filename="test_0058.edf"):
    """
    read edf file from ESRF frelon camera, add a peak and perform a peak search
    """
    CCDLABEL = "FRELON"

    fliprot = DictLT.dict_CCD[CCDLABEL][3]
    framedim = DictLT.dict_CCD[CCDLABEL][0]
    formatdata = DictLT.dict_CCD[CCDLABEL][5]
    offset = DictLT.dict_CCD[CCDLABEL][4]

    dataimage = IOimage.readoneimage(
        filename, framedim=framedim, offset=offset, formatdata=formatdata
    )

    header = IOimage.readheader(filename, offset=offset)

    dataimage = np.reshape(dataimage, framedim)

    # build a peak
    mypeak = np.array([[1, 5, 6, 1, 4],
                        [2, 10, 50, 60, 2],
                        [6, 80, 90, 60, 2],
                        [2, 55, 60, 60, 3],
                        [1, 23, 3, 9, 9]])

    dataimage[1200:1205, 600:605] = 20 * mypeak

    dataimage[:, 950:1050] = 1000 * np.ones((2048, 100))

    newfilename = "testpeak_0058.edf"

    IOimage.writeimage(newfilename, header, np.ravel(dataimage), dataformat=np.uint16)

    from matplotlib.colors import LogNorm

    p.imshow(dataimage, aspect="equal", interpolation="nearest")  # ,
    # norm=LogNorm(vmin=200, vmax=4096))

    p.show()

    Isorted, fitpeak, localpeak = RMCCD.PeakSearch(
        newfilename,
        CCDLabel="FRELON",
        PixelNearRadius=10,
        removeedge=2,
        IntensityThreshold=500,
        local_maxima_search_method=2,
        thresholdConvolve=1000,
        boxsize=15,
        position_definition=1,
        verbose=0,
        fit_peaks_gaussian=1,
        xtol=0.001,
        FitPixelDev=2.0,
        return_histo=0,
        Saturation_value=DictLT.dict_CCD[CCDLABEL][2],
        Saturation_value_flatpeak=DictLT.dict_CCD[CCDLABEL][2],
        write_execution_time=1,
    )[:3]

    print("Isorted", Isorted)

    IOLT.writefile_Peaklist(
        "frelon_test",
        Isorted,
        overwrite=1,
        initialfilename=filename,
        comments="test,test",
    )

    return dataimage


def test_edf_onepix(filename="test_0072.edf"):
    """
    read edf file from ESRF frelon camera, add a peak and perform a peak search
    """
    CCDLABEL = "FRELON"

    fliprot = DictLT.dict_CCD[CCDLABEL][3]
    framedim = DictLT.dict_CCD[CCDLABEL][0]
    formatdata = DictLT.dict_CCD[CCDLABEL][5]
    offset = DictLT.dict_CCD[CCDLABEL][4]

    dataimage = IOimage.readoneimage(
        filename, framedim=framedim, offset=offset, formatdata=formatdata)

    dataimage = np.reshape(dataimage, framedim)

    dataimage = np.flipud(dataimage)

    #    from matplotlib.colors import LogNorm
    #
    #    p.imshow(dataimage, aspect='equal',
    #                        interpolation='nearest')#,
    #                        #norm=LogNorm(vmin=200, vmax=4096))

    return dataimage


def test_peaksearch(filenameindex=23, dirname="/home/micha/LaueProjects/AxelUO2", mp=1):
    """
    peaksearch function to be called for testing multiprocessing
    """
    CCDLABEL = "MARCCD165"
    framedim = (2048, 2048)
    offset = 4096
    formatdata = "uint16"
    fliprot = "no"

    filename = "UO2_Kr_1_%04d.mccd" % filenameindex
    filename = os.path.join(dirname, filename)
    Isorted, fitpeak, localpeak = RMCCD.PeakSearch(
        filename,
        CCDLabel="MARCCD165",
        PixelNearRadius=10,
        removeedge=2,
        IntensityThreshold=10,
        local_maxima_search_method=2,
        thresholdConvolve=500,
        boxsize=15,
        position_definition=1,
        verbose=1,
        fit_peaks_gaussian=1,
        xtol=0.001,
        FitPixelDev=2.0,
        return_histo=0,
        Saturation_value=DictLT.dict_CCD[CCDLABEL][2],
        Saturation_value_flatpeak=DictLT.dict_CCD[CCDLABEL][2],
        write_execution_time=1,
    )[:3]

    print("Isorted", Isorted)
    IOLT.writefile_Peaklist(
        "%s_testmp%d" % (filename[:-5], mp),
        Isorted,
        overwrite=1,
        initialfilename=filename,
        comments="test,test",
    )


def test_readSCMOS_crop_fast():
    folder = "/home/micha/LaueProjects/testscmos/25Oct/raw"
    filename = "scan_vg3_0001.tif"
    fullpathfilename = os.path.join(folder, filename)
    filesize = os.path.getsize(fullpathfilename)
    offsetheader = filesize - (2018 * 2016) * 2

    print("offsetheader", offsetheader)

    linestart_ypix_ifast = 1009
    linefinal_ypix_ifast = 1308

    ysize = linefinal_ypix_ifast - linestart_ypix_ifast + 1

    band = IOimage.readoneimage_band(
        fullpathfilename,
        framedim=(2018, 2016),
        dirname=None,
        offset=offsetheader,
        line_startindex=linestart_ypix_ifast,
        line_finalindex=linefinal_ypix_ifast,
        formatdata="uint16",
    )

    print("band shape", band.shape)
    print(len(band) / 2016.0)

    band2D = np.reshape(band, (ysize, 2016))
    import pylab as p

    p.subplot(141)
    p.imshow(band2D, vmin=1000, vmax=4000, origin="upper", interpolation="nearest")

    #
    dataimage, framedim, fliprot = IOimage.readCCDimage(
        fullpathfilename, CCDLabel="sCMOS_fliplr"
    )
    p.subplot(142)
    p.imshow(dataimage, vmin=1000, vmax=4000, origin="upper", interpolation="nearest")

    x = 839
    y = 1091
    boxx = 50
    boxy = 50

    xpixmin = x - boxx
    xpixmax = x + boxx

    ypixmin = y - boxy
    ypixmax = y + boxy

    ymin = (y - boxy) * 2016
    ymax = (y + boxy) * 2016

    lineFirstElemIndex = ypixmin
    lineLastElemIndex = ypixmax

    print("lineFirstElemIndex", lineFirstElemIndex)
    print("lineLastElemIndex", lineLastElemIndex)

    band = IOimage.readoneimage_band(
        fullpathfilename,
        framedim=(2018, 2016),
        dirname=None,
        offset=offsetheader,
        line_startindex=lineFirstElemIndex,
        line_finalindex=lineLastElemIndex,
        formatdata="uint16",
    )

    nblines = lineLastElemIndex - lineFirstElemIndex + 1

    band2D = np.reshape(band, (nblines, 2016))

    dataimage2D = np.zeros((2018, 2016))

    print("band2D.shape", band2D.shape)

    dataimage2D[lineFirstElemIndex : lineLastElemIndex + 1, :] = band2D
    p.subplot(143)
    p.imshow(dataimage2D, vmin=1000, vmax=4000, origin="upper", interpolation="nearest")

    p.subplot(144)
    #     p.imshow(dataimage2D[lineFirstElemIndex:lineLastElemIndex+1,xpixmin:xpixmax+1], vmin = 1000, vmax=4000, origin='upper')
    p.imshow(
        band2D[:, xpixmin : xpixmax + 1],
        vmin=1000,
        vmax=4000,
        origin="upper",
        interpolation="nearest",
    )

    p.show()


def test_EIGER4Munstacked_fast():
    # test EIGER4Munstacked
    folder = "/home/micha/LaueProjects/TarikSadat_hdf5"
    filename = "Si_reference_y3_2_224_data_000001_scan_1234_0008.unstacked"
    fullpathfilename = os.path.join(folder, filename)
    filesize = os.path.getsize(fullpathfilename)

    framedim = (2167, 2070)
    # uint16
    offsetheader = filesize - (framedim[0] * framedim[1]) * 2

    print("offsetheader", offsetheader)

    linestart_ypix_ifast = 500
    linefinal_ypix_ifast = 600

    ysize = linefinal_ypix_ifast - linestart_ypix_ifast + 1

    band = IOimage.readoneimage_band(
        fullpathfilename,
        framedim=framedim,
        dirname=None,
        offset=0,
        line_startindex=linestart_ypix_ifast,
        line_finalindex=linefinal_ypix_ifast,
        formatdata="uint32",
    )

    print("band shape", band.shape)
    print(len(band) // framedim[1])

    band2D = np.reshape(band, (ysize, framedim[1]))
    import pylab as p

    p.subplot(141)
    p.imshow(band2D, vmin=10, vmax=50000, origin="upper", interpolation="nearest")

    #
    dataimage, framedim, fliprot = IOimage.readCCDimage(
        fullpathfilename, CCDLabel="EIGER_4Munstacked"
    )
    p.subplot(142)
    p.imshow(dataimage, vmin=10, vmax=50000, origin="upper", interpolation="nearest")

    x = 635
    y = 550
    boxx = 50
    boxy = 100

    xpixmin = x - boxx
    xpixmax = x + boxx

    ypixmin = y - boxy
    ypixmax = y + boxy

    ymin = (y - boxy) * framedim[1]
    ymax = (y + boxy) * framedim[1]

    lineFirstElemIndex = ypixmin
    lineLastElemIndex = ypixmax

    print("lineFirstElemIndex", lineFirstElemIndex)
    print("lineLastElemIndex", lineLastElemIndex)

    band = IOimage.readoneimage_band(
        fullpathfilename,
        framedim=framedim,
        dirname=None,
        offset=0,
        line_startindex=lineFirstElemIndex,
        line_finalindex=lineLastElemIndex,
        formatdata="uint32",
    )

    nblines = lineLastElemIndex - lineFirstElemIndex + 1

    band2D = np.reshape(band, (nblines, framedim[1]))

    dataimage2D = np.zeros(framedim)

    print("band2D.shape", band2D.shape)

    dataimage2D[lineFirstElemIndex : lineLastElemIndex + 1, :] = band2D
    p.subplot(143)
    p.imshow(dataimage2D, vmin=10, vmax=50000, origin="upper", interpolation="nearest")

    p.subplot(144)
    #     p.imshow(dataimage2D[lineFirstElemIndex:lineLastElemIndex+1,xpixmin:xpixmax+1], vmin = 1000, vmax=4000, origin='upper')
    p.imshow(
        band2D[:, xpixmin : xpixmax + 1],
        vmin=10,
        vmax=50000,
        origin="upper",
        interpolation="nearest",
    )

    p.show()


# ----------                    MAIN  and tryouts            -----------------

#     folder = '/home/micha/LaueProjects/testscmos/25Oct/raw'
#     filename = 'scan_vg3_0001.tif'
#     fullpathfilename = os.path.join(folder,filename)
#     filesize = os.path.getsize(fullpathfilename)
#     framedim = (2018,2016)
#     offsetheader = filesize-(framedim[0]*framedim[1])*2
#     # uint16
#     print 'offsetheader',offsetheader


folder = "/home/micha/LaueProjects/testscmos/GeMar"
filename = "Gestd_0001.mccd"
fullpathfilename = os.path.join(folder, filename)
filesize = os.path.getsize(fullpathfilename)
framedim = (2048, 2048)
# uint16
offsetheader = filesize - (framedim[0] * framedim[1]) * 2

print("offsetheader", offsetheader)

linestart_ypix_ifast = 1009
linefinal_ypix_ifast = 1308

ysize = linefinal_ypix_ifast - linestart_ypix_ifast + 1

band = IOimage.readoneimage_band(
    fullpathfilename,
    framedim=framedim,
    dirname=None,
    offset=offsetheader,
    line_startindex=linestart_ypix_ifast,
    line_finalindex=linefinal_ypix_ifast,
    formatdata="uint16",
)

print("band shape", band.shape)
print(len(band) / framedim[1])

band2D = np.reshape(band, (ysize, framedim[1]))
import pylab as p


p.subplot(141)
p.imshow(band2D, vmin=10, vmax=4000, origin="upper", interpolation="nearest")


#
dataimage, framedim, fliprot = IOimage.readCCDimage(
    fullpathfilename, CCDLabel="sCMOS_fliplr"
)
p.subplot(142)
p.imshow(dataimage, vmin=10, vmax=4000, origin="upper", interpolation="nearest")


x = 1168
y = 978
boxx = 50
boxy = 70

xpixmin = x - boxx
xpixmax = x + boxx

ypixmin = y - boxy
ypixmax = y + boxy

ymin = (y - boxy) * framedim[1]
ymax = (y + boxy) * framedim[1]

lineFirstElemIndex = ypixmin
lineLastElemIndex = ypixmax

print("lineFirstElemIndex", lineFirstElemIndex)
print("lineLastElemIndex", lineLastElemIndex)

band = IOimage.readoneimage_band(
    fullpathfilename,
    framedim=framedim,
    dirname=None,
    offset=offsetheader,
    line_startindex=lineFirstElemIndex,
    line_finalindex=lineLastElemIndex,
    formatdata="uint16",
)

nblines = lineLastElemIndex - lineFirstElemIndex + 1

band2D = np.reshape(band, (nblines, framedim[1]))


dataimage2D = np.zeros(framedim)

print("band2D.shape", band2D.shape)

dataimage2D[lineFirstElemIndex : lineLastElemIndex + 1, :] = band2D
p.subplot(143)
p.imshow(dataimage2D, vmin=10, vmax=4000, origin="upper", interpolation="nearest")


p.subplot(144)
#     p.imshow(dataimage2D[lineFirstElemIndex:lineLastElemIndex+1,xpixmin:xpixmax+1], vmin = 1000, vmax=4000, origin='upper')
p.imshow(band2D[:, xpixmin : xpixmax + 1],
                    vmin=10,
                    vmax=4000,
                    origin="upper",
                    interpolation="nearest")


if 0:

    background_flag = "auto"
    blacklistpeaklist = None
    outputfilename = "testpsearch.dat"
    pspfile = "PeakSearch_Ge_blanc_0000_LT_0.psp"
    RMCCD.peaksearch_on_Image("Ge_blanc_0000.mccd",
                                pspfile,
                                background_flag,
                                blacklistpeaklist,
                                "MARCCD165",
                                outputfilename)


if 0:
    import pylab as p

    dirname = "/home/micha/LaueProjects/NW"
    filename = "NW_curve2_000078.mccd"

    d, framedim, fliprot = IOimage.readCCDimage(os.path.join(dirname, filename))

    p.figure(1)

    ax = p.subplot(221)
    ax.imshow(d, interpolation="nearest")

    dmin = ndimage.filters.minimum_filter(d, size=(11, 11))

    ax2 = p.subplot(222)
    ax2.imshow(dmin, interpolation="nearest")

    ax3 = p.subplot(223)
    ax3.imshow(d - dmin, interpolation="nearest")

    ax4 = p.subplot(224)
    dblur = ImProc.blurCCD(d, 5)
    ax4.imshow(dblur, interpolation="nearest")

    cd = np.clip(d - dblur, 0, 57500)
    maskcd = np.where(ImProc.circularMask((1024, 1024), 1020, (2048, 2048)), cd, 0)
    overbkg = np.array(maskcd, dtype=np.uint16)

    numrows, numcols = d.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = d[row, col]
            zmin = dmin[row, col]
            zblur = dblur[row, col]
            return "x=%1.4f, y=%1.4f, z=%1.4f zblur=%1.4f z-zblur=%1.4f" % (
                x,
                y,
                z,
                zblur,
                z - zblur)
        else:
            return "x=%1.4f, y=%1.4f" % (x, y)

    ax.format_coord = format_coord
    #
    #    p.show()

    _header = IOimage.readheader(os.path.join(dirname, filename))

    print("shape")

    IOimage.writeimage(os.path.join(dirname, "NW_curve2_001000.mccd"), _header, np.ravel(overbkg))

if 0:

    filename = "Ge_blanc_0000.mccd"
    t = 1000
    time_0 = ttt.time()
    d = IOimage.readoneimage("Ge_blanc_0000.mccd")
    # , framedim = (2048,2048), dirname = None, offset = 4100, formatdata = "uint16").reshape((2048,2048))

    # d = rot90(d,k=3)

    Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(filename,
                                            CCDLabel="MARCCD165",
                                            center=None,
                                            boxsizeROI=(200, 200),
                                            PixelNearRadius=5,
                                            removeedge=2,
                                            IntensityThreshold=400,
                                            thresholdConvolve=200,
                                            boxsize=15,
                                            verbose=0,
                                            position_definition=0,
                                            fit_peaks_gaussian=1,
                                            xtol=0.0000001,
                                            return_histo=1,
                                            FitPixelDev=25,  # to_reject3 parameter
                                            write_execution_time=1,
                                            Saturation_value=65535,
                                            Saturation_value_flatpeak=65535)

    print("meanpos", Isorted)
    t4 = ttt.time()
    print("Mean pos of cluster, execution time: %.2f sec" % (t4 - time_0))
    ax = p.subplot(111)
    ax.imshow(np.log(d), interpolation="nearest")

    from matplotlib.patches import Circle

    for po in Isorted:

        large_circle = Circle(po, 7, fill=False, color="b")
        center_circle = Circle(po, 0.5, fill=True, color="r")
        ax.add_patch(large_circle)
        ax.add_patch(center_circle)

    numrows, numcols = d.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = d[row, col]
            return "x=%1.4f, y=%1.4f, z=%1.4f" % (x, y, z)
        else:
            return "x=%1.4f, y=%1.4f" % (x, y)

    ax.format_coord = format_coord

    p.show()

    raise ValueError("end of example")

    filename = "CdTe_I999_03Jul06_0200.mccd"
    filename = "Ge_blanc_0000.mccd"
    # filename = 'SS_0171.mccd'

    val, res = RMCCD.Find_optimal_thresholdconvolveValue(filename, 200)
    print("val", val)

    time_0 = ttt.time()

    d = IOimage.readoneimage(filename).reshape((2048, 2048))

    t1 = ttt.time()
    print("Read frame, execution time: %.2f sec" % (t1 - time_0))

    res = []
    IntensityThreshold = 400
    for tc in (0, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000):
        # tstart = ttt.time()
        Isorted, fitpeak, localpeak, nbrawblobs = RMCCD.PeakSearch(
            filename,
            CCDLabel="MARCCD165",
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

    # plot_image_markers(d, Isorted[:,:2])

    raise ValueError("end of example")
if 0:
    import scipy

    image = scipy.misc.lena().astype(np.float32)
    # Convolve with a Gaussian Gt for t=2 ------------------------------------------------------
    [X, Y] = np.mgrid[-6:7, -6:7]
    A = X + 1j * Y
    R = np.abs(A)
    t = 2.0
    Gt = np.exp(-R * R / 4.0 / t) / 4.0 / np.pi / t
    convImage = scipy.signal.convolve(image, Gt)

    # Gradient
    Grdnt = np.gradient(convImage)
    # Magnitude of the gradient
    magn = np.sqrt(Grdnt[0] * Grdnt[0] + Grdnt[1] * Grdnt[1])
    # Laplacian
    Lplz = ndimage.laplace(np.float32(convImage))

if 0:
    Gx = ndimage.sobel(d, axis=0)
    Gy = ndimage.sobel(d, axis=1)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    # p.imshow(G, interpolation = 'nearest')
    # p.show()

    # d= d[1400:1500,700:800]

    df = ndimage.gaussian_filter(d, 10)

    tG = np.where(df > 1000, 1, 0)
    ll, nf = ndimage.label(tG)  # , structure = np.ones((3,3)))
    meanpos = np.array(
        ndimage.measurements.center_of_mass(tG, ll, np.arange(1, nf + 1)), dtype=float
    )
    meanpos = np.fliplr(meanpos)

    # test_multiROIfit_thread()
    # test_average_Images()

    # > python readmccd.py myimage.mccd

    # test_peak_search_file_series()

    # test_fast_peaksearch()

    # t = 500
    # import pylab as p
    # time_0 = ttt.time()
    # d = IOimage.readoneimage('CdTe_I999_03Jul06_0200.mccd').reshape((2048,2048))

    # t1 = ttt.time()
    # print "Read frame, execution time: %.2f sec"%( t1 - time_0)
    # aa = ConvolvebyKernel(d,
    # framedim = (2048,2048),
    # peakVal = 4,
    # boxsize = 5,
    # central_radius = 2,
    # verbose = 0)

    # thraa = np.where((aa>t,1,0)

    # t2 = ttt.time()
    # print "Convolve, execution time: %.2f sec"%( t2 - t1)
    # ll,nf = ndimage.label(thraa, structure = np.ones((3,3)))

    # t3 = ttt.time()
    # print "Label cluster, execution time: %.2f sec"%( t3 - t2)
    # print "nb of peaks",nf

    # #meanpos = np.zeros((nf,2))
    # #for k in range(nf):
    # #meanpos[k] = np.mean(np.where((ll == k),axis=1)

    # #ndimage.find_objects(ll)
    # meanpos = np.array(scipy.ndimage.measurements.center_of_mass(thraa, ll,np.arange(1,nf+1)),dtype=float)
    # meanpos = np.fliplr(meanpos)
    # t4 = ttt.time()
    # print "Mean pos of cluster, execution time: %.2f sec"%( t4 - t3)
    # p.imshow(thraa, interpolation = 'nearest')
    # p.show()

    if len(sys.argv) > 1:

        filename = sys.argv[1]
        time_0 = ttt.time()

        Isorted, fitpeak, localpeak, histo = RMCCD.PeakSearch(
            filename,
            CCDLabel="MARCCD165",
            PixelNearRadius=20,
            removeedge=2,
            IntensityThreshold=100,
            boxsize=15,
            position_definition=1,
            verbose=0,
            fit_peaks_gaussian=1,
            xtol=0.001,
            return_histo=1,
        )

        prefix, file_extension = filename.split(".")
        IOLT.writefile_Peaklist(
            prefix + "_c",
            Isorted,
            overwrite=1,
            initialfilename=filename,
            comments="test",
        )

        print("peak list written in %s" % (prefix + "_c.dat"))
        print("execution time: %.2f sec" % (ttt.time() - time_0))

        sys.exit()

    # # PERFORMANCE PROFILER
    # import profile
    # #profile.run('test_Peaksearch_Ge()','peaksearch.profile')
    # d = IOimage.readoneimage('Ge_blanc_0000.mccd', framedim = (2048,2048))
    # Data = np.reshape(d, (2048,2048))
    # profile.run('localmaxima(Data, 25, diags=1)','peaksearch.profile')

    # import pstats
    # pstats.Stats('peaksearch.profile').sort_stats('time').print_stats()

    # using PIL and filtering

    import pylab as p
    try:
        import ImageFilter  
    except ImportError:
        import PIL.ImageFilter as ImageFilter
    # test_multiROIfit()

    from matplotlib.colors import LogNorm

    # test_peaksearch_ROI()
    # test_VHR_ROI()

if 0:

    if 0:  # TODO: finish that ...
        filename = "Ge_blanc_0000.mccd"
        centers = [[621, 1656], [1242, 1661]]
        boxsize = 15
        taby = IOimage.readoneimage_multi_barycenters(
            filename, centers, boxsize, offsetposition=0
        )

    if 1:  # reading of marccd image #peak search with convolve or shifted arrays
        filename = "Ge_blanc_0000.mccd"
        # filename='CdTe_I999_03Jul06_0200.mccd'
        # filename = 'SS_0171.mccd'
        time_0 = ttt.time()
        pilimage, dataimage = IOimage.readoneimage_full(filename)

        xminf2d, xmaxfit2d, yminfit2d, ymaxfit2d = 1, 2048, 1, 2048

        imin, imax, jmin, jmax = 2048 - ymaxfit2d, 2048 - yminfit2d, xminf2d, xmaxfit2d

        dataimage_ROI = dataimage[imin:imax, jmin:jmax]  # array index   i,j
        # fit2d index:  X=j Y=2048-i

        # WARNING LocalMaxima_ShiftArrays returns 2 args
        purged_pklist = ImProc.LocalMaxima_ShiftArrays(
            dataimage_ROI,
            IntensityThreshold=500,
            Saturation_value=65535,
            boxsize_for_probing_minimal_value_background=30,
            pixeldistance_remove_duplicates=25)

        purged_pklist = purged_pklist - np.array([2, 1])
        purged_pklist = np.fliplr(purged_pklist)

        # this is for plotting image and peak markers

        if 0:
            imgplot = p.imshow(
                np.log(dataimage_ROI),
                aspect="equal",
                interpolation="nearest",
                vmin=0.0,
                vmax=0.80 * np.log(dataimage_ROI).max(),
            )  # extent ?
            # imgplot=p.imshow(dataimage_ROI,interpolation='nearest')
            imgplot.set_cmap("spectral")

            # imgplot.set_clim=(0.5,1.)
            p.colorbar()

        if 0:
            imgplot = p.imshow(
                dataimage_ROI,
                aspect="equal",
                interpolation="nearest",
                norm=LogNorm(vmin=1, vmax=65000),
            )  # extent ?
            # imgplot=p.imshow(dataimage_ROI,interpolation='nearest')
            imgplot.set_cmap("spectral")

            # imgplot.set_clim=(0.5,1.)
            p.colorbar()

        if 1:  # the GOOD ONE
            fig = p.figure()
            ax = fig.add_subplot(111)

            # trying to remove a gaussian background (only for now for display)
            # Create the gaussian data
            Xin, Yin = np.mgrid[0:2047, 0:2047]
            data_gauss = fit2d.gaussian(100, 1024, 1024, 500, 500)(Xin, Yin)

            cond_circle = (Xin - 1023) ** 2 + (Yin - 1023) ** 2 <= 1024 ** 2
            dataimage_ROI_wo_bkg = np.where(
                cond_circle, dataimage_ROI - data_gauss, dataimage_ROI
            )

            imgplot = ax.imshow(
                dataimage_ROI_wo_bkg,  # dataimage_ROI,
                aspect="equal",
                interpolation="nearest",
                norm=LogNorm(vmin=1, vmax=65000),
            )  # extent ?
            # imgplot=p.imshow(dataimage_ROI,interpolation='nearest')
            imgplot.set_cmap("spectral")

            PointToPlot = np.zeros(purged_pklist.shape)
            PointToPlot = (
                purged_pklist
            )  # - array([2,1])  # this crude correction only to fitdata with the display

            from matplotlib.patches import Circle

            for po in np.fliplr(PointToPlot):

                large_circle = Circle(po, 7, fill=False, color="b")
                center_circle = Circle(po, 0.5, fill=True, color="r")
                ax.add_patch(large_circle)
                ax.add_patch(center_circle)

            # imgplot.set_clim=(0.5,1.)
            # fig.Figure.colorbar()  # does not work

            def onclick(event):
                print(
                    "mouse button=%d, x=%d, y=%d, xdata=%f, ydata=%f, intensity=%f"
                    % (
                        event.button,
                        event.x,
                        event.y,
                        event.xdata,
                        event.ydata,
                        dataimage_ROI[int(event.ydata), int(event.xdata)],
                    )
                )

            cid = fig.canvas.mpl_connect("button_press_event", onclick)

            p.show()
