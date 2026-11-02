# -*- coding: utf-8 -*-
"""
Data side of the Mosaic & Monitor tools (GUI/mosaic.py, PeakSearchGUI): no GUI dependency.

- ROI counters over a series of images: CollectData(), CollectData_oneImage(), peak fits FitPeakOnMap(),
  filter_fitted_peaks() (default fit parameters DEFAULT_DICTfittingparameters)
- results files: text map files (map_file_header(), read_map_file(), list_map_files()) and
  roicounters_*.h5 files of workflows.ROICountersWorkflow (read_counters_results(), counters_of_results_file())

GUI/mosaic.py imports these names, so LaueTools.GUI.mosaic.read_map_file etc. still work.
"""
import os
import re

import numpy as np
from scipy import ndimage as scind

from . import dict_LaueTools as DictLT
from . import generaltools as GT
from . import readmccd as RMCCD
from . import IOimagefile as IOimage
from . import imageprocessing as ImProc

DEFAULT_DICTfittingparameters = {
                'modelFunction':"gaussian",
                'positionStart':"max",
                'peaksizeStart': 4,
                'MinimumPeakAmplitude':25,
                'reject_negative_baseline':True,
                'reject_large_PixelDeviation':True,
                'reject_weakPeaks':False,
                'FitPixelDev':None,
            }

def results_writers(save=True):
    """return np.savetxt and a function opening a file for writing, or functions writing nothing
    if save is False (results files of buildMosaic3() and plot_counters_results())"""
    if save:
        return np.savetxt, lambda path: open(path, "w")
    return (lambda *args, **kwargs: None), lambda path: open(os.devnull, "w")


# plotted quantity of GUI counters (title and colorbar label of plots), %s: X or Y
COUNTERS_QUANTITY = {"mosaic": "ROI pixel intensity",
                     "mean": "mean intensity in ROI",
                     "max": "max intensity in ROI",
                     "ptp": "peak to peak intensity in ROI (max - min)",
                     "Position Centroid": "centroid %s position (pixel)",
                     "Position MAX": "highest intensity pixel %s position (pixel)",
                     "Position XY": "fitted peak %s position (pixel)",
                     "Displacement": "peak displacement from mean position (pixel)",
                     "DisplacementVector": "peak displacement vector, color: norm (pixel)",
                     "Amplitude": "fitted peak amplitude (above background)",
                     "Shape": "fitted peak size: largest gaussian width (pixel)"}


def counter_quantity(counter, dict_param, component=None):
    """name of quantity plotted for GUI counter (component: 'X' or 'Y' for positions)"""
    quantity = COUNTERS_QUANTITY.get(counter, counter)
    if "%s" in quantity:
        quantity = quantity % (component or "")
    if dict_param.get("NormalizeWithMonitor", False) and counter in ("mosaic", "mean", "max", "ptp",
                                                                       "Amplitude"):
        quantity += " / monitor"
    return quantity


# ============================================================================
# RESULTS FILES: replot of maps and mosaics (show_map_file(), show2DmapGUI)
# ============================================================================

# keys of dict_param written in the header of text map files (see map_file_header())
MAPFILE_DICT_PARAM_KEYS = ("pixelX_center", "pixelY_center", "pixelboxsize_X", "pixelboxsize_Y", "imagesfolder",
                           "filename_representative", "CCDLabel", "nbdigits", "xlabel", "ylabel",
                           "NormalizeWithMonitor")
# text map files without header (before Oct. 2026): regex of file name -> datatype of ImshowFrame
# (group 1: counter, group 2: X or Y component)
LEGACY_MAPFILE_PATTERNS = ((r"^(mean|max|ptp)_2D_", "Intensity"),
                           (r"^(Position XY|Position MAX|Position Centroid)_2D_([XY])_", "Position"),
                           (r"^(Displacement)_2D_radial_", "scalar"),
                           (r"^(Shape)_2D_size_", "scalar"),
                           (r"^(Amplitude)_2D_Amplitude_", "scalar"))
# GUI counters plotted from results of workflows.ROICountersWorkflow (see GUI_TO_WORKFLOW_COUNTERS)
WORKFLOW_TO_GUI_COUNTERS = {"mosaic": ["mosaic"], "mean": ["mean"], "max": ["max"], "ptp": ["ptp"],
                            "XYcentroid": ["Position Centroid"], "XYmax": ["Position MAX"],
                            "fit": ["Position XY", "Displacement", "Amplitude", "Shape"]}


def map_file_header(datatype, quantity, Imageindices, nb_col, nb_lines, boxsize_col=0, boxsize_line=0,
                    imagename="", dict_param=None, mosaic=0, absolutecornerindices=None):
    """header ('# key: value' lines, title last) of text files of maps and mosaics plotted by ImshowFrame
    (same arguments), read by read_map_file(). np.loadtxt() skips it."""
    lines = ["datatype: %s" % ("MOSAIC" if mosaic else datatype),
             "quantity: %s" % quantity,
             "Imageindices: " + " ".join("%d" % index for index in np.ravel(Imageindices)),
             "nb_col: %d" % nb_col, "nb_lines: %d" % nb_lines,
             "boxsize_col: %d" % boxsize_col, "boxsize_line: %d" % boxsize_line]
    if absolutecornerindices is not None:
        lines.append("absolutecornerindices: %d %d" % tuple(absolutecornerindices))
    for key in MAPFILE_DICT_PARAM_KEYS:
        if (dict_param or {}).get(key) is not None:
            lines.append("%s: %s" % (key, dict_param[key]))
    lines.append("title: %s" % str(imagename).replace("\n", " "))
    return "\n".join("# " + line for line in lines)


def _header_value(text):
    """int, float, bool or str of a header value"""
    if text in ("True", "False"):
        return text == "True"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def read_map_file(path):
    """read a text map file of mosaic.py: MOSAIC_image_Plot_*, <counter>_2D_* or a file saved by
    File/Save Data of a map window. Files without header (before Oct. 2026) are recognized by their name:
    their image indices are read in the 1D file of the same date (e.g. max_1D_date_...), if any.

    :return: dict of the arguments of ImshowFrame (title, dataarray, dict_param, datatype, ...)
    """
    folder, filename = os.path.split(path)
    header = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
    if "datatype" not in header:
        for pattern, datatype in LEGACY_MAPFILE_PATTERNS:
            match = re.match(pattern, filename)
            if match:
                break
        else:
            raise ValueError("%s is not a map file of mosaic.py (no header, unknown file name)" % filename)
    data = np.loadtxt(path, ndmin=2)
    nb_lines, nb_col = data.shape

    if "datatype" in header:
        mosaic = int(header["datatype"] == "MOSAIC")
        datatype = "Intensity" if mosaic else header["datatype"]
        dict_param = {key: _header_value(header[key]) for key in MAPFILE_DICT_PARAM_KEYS if key in header}
        Imageindices = np.array(header["Imageindices"].split(), dtype=int)
        quantity = header.get("quantity", counter_quantity("mosaic", {}) if mosaic else "")
        corner = header.get("absolutecornerindices")
        frameargs = dict(nb_col=int(header["nb_col"]), nb_lines=int(header["nb_lines"]),
                         boxsize_row=int(header.get("boxsize_col", 0)),
                         boxsize_line=int(header.get("boxsize_line", 0)),
                         absolutecornerindices=tuple(int(v) for v in corner.split()) if corner else None)
        imagename = header.get("title", filename)
    else:
        counter = match.group(1)
        component = match.group(2) if match.re.groups > 1 else None
        if datatype == "Position":
            datatype += component
        mosaic = 0
        dict_param = {}
        if component:  # ROI centre unknown: background level of ImshowFrame
            dict_param["pixel%s_center" % component] = float(np.nanmean(data))
        Imageindices = legacy_image_indices(folder, filename, data.size)
        quantity = counter_quantity(counter, {}, component)
        frameargs = dict(nb_col=nb_col, nb_lines=nb_lines, boxsize_row=int(datatype == "Intensity"),
                         boxsize_line=int(datatype == "Intensity"))
        imagename = filename
    if datatype in ("RadialPosition", "Vector"):  # X and Y maps needed to change the centre are not in file
        datatype = "scalar"
    return dict(title=filename, dataarray=data, dict_param=dict_param, datatype=datatype, quantity=quantity,
                Imageindices=Imageindices, imagename=imagename, mosaic=mosaic, stepindex=1, **frameargs)


def legacy_image_indices(folder, filename, nbpoints):
    """image indices of a map file without header (see read_map_file()), read in the first column of the
    1D file of the same date (<counter>_1D_*, Position XYXpos_*, AmplitudeAmplitude_*), or 0 to nbpoints - 1"""
    date = re.search(r"date_\d{8}_\d{6}", filename)
    if date is not None:
        for name in sorted(os.listdir(folder)):
            if (date.group() not in name or "_2D" in name
                    or not any(word in name for word in ("_1D_", "pos_", "Amplitude_"))):
                continue
            try:
                values = np.loadtxt(os.path.join(folder, name), ndmin=2)
            except ValueError:
                continue
            if values.shape[1] in (2, 4) and len(values) <= nbpoints:
                return values[:, 0].astype(int)
    return np.arange(nbpoints)


def read_counters_results(path):
    """results of workflows.ROICountersWorkflow (roicounters_*.h5 written by the parallel computation of
    counters) and dict_param to plot them with plot_counters_results(). Only the first ROI is plotted.

    :return: results, dict_param
    """
    import LaueTools.scripts.workflows as WF

    results = WF.load_results(path)
    params = results["params"]
    xpic, ypic = params["roicenters"][0]
    dict_param = {"imagesfolder": params["folder"],
                  "filename_representative": params["filename_representative"],
                  "CCDLabel": params["CCDLabel"],
                  "nbdigits": params.get("nbdigits", 4),
                  "pixelX_center": float(xpic), "pixelY_center": float(ypic),
                  "pixelboxsize_X": params["boxsize_X"], "pixelboxsize_Y": params["boxsize_Y"],
                  "NormalizeWithMonitor": params.get("NormalizeWithMonitor", False),
                  "monitoroffset": params.get("monitoroffset", 0.0),
                  "selectedcounters": [guicounter for counter in params["counters"]
                                       for guicounter in WORKFLOW_TO_GUI_COUNTERS.get(counter, [])],
                  # replot only: no results files
                  "save_results": False, "save_plots": False}
    return results, dict_param


def counters_of_results_file(path):
    """GUI counters that can be plotted from a roicounters_*.h5 file (its parameters only are read)"""
    import json
    import h5py

    with h5py.File(path, "r") as f:
        params = json.loads(f.attrs["params"])
    return [guicounter for counter in params["counters"] for guicounter in WORKFLOW_TO_GUI_COUNTERS.get(counter, [])]


def list_map_files(folder):
    """results files of mosaic.py in folder (roicounters_*.h5 and text map files), most recent first"""
    names = [name for name in os.listdir(folder)
             if (name.startswith("roicounters_") and name.endswith(".h5")) or name.startswith("MOSAIC_image_Plot")
             or any(re.match(pattern, name) for pattern, _ in LEGACY_MAPFILE_PATTERNS)]
    return sorted((os.path.join(folder, name) for name in names), key=os.path.getmtime, reverse=True)


def FitPeakOnMap(mosaic,
                ROIcenter,
                reject_negative_baseline=True,
                reject_large_PixelDeviation=True,
                reject_weakPeaks=True,
                FitPixelDev=None,
                MinimumPeakAmplitude=25,
                modelFunction="gaussian",
                framedimensions=(2048, 2048),
                positionStart="max",
                peaksizeStart=0.5,
                verbose=1):
    """
    Fit peak on series of 2D pixel intensities array
    """

    print('\n\n****!! Fitting a peak on a several images (e.g. map)  !!****\n\n')
    n0, n1, n2, n3 = mosaic.shape
    xpic, ypic = ROIcenter

    reject_negative_baseline = 1
    if FitPixelDev is None:
        # limited by boxsize
        FitPixelDev = max(n2, n3)

    FittingParametersDict = {}
    FittingParametersDict["boxsize"] = (n2, n3)
    FittingParametersDict["framedim"] = framedimensions
    FittingParametersDict["saturation_value"] = 65000
    FittingParametersDict["baseline"] = "auto"

    FittingParametersDict["startangles"] = 0
    FittingParametersDict["position_start"] = positionStart
    FittingParametersDict["start_sigma1"] = peaksizeStart
    FittingParametersDict["start_sigma2"] = peaksizeStart

    FittingParametersDict["fitfunction"] = modelFunction
    FittingParametersDict["xtol"] = 0.0001
    FittingParametersDict["offsetposition"] = 1

    centers = xpic * np.ones(n0 * n1), ypic * np.ones(n0 * n1)
    CentralPeaks = np.array(centers).T
    #     print "CentralPeaks",CentralPeaks

    Resfit = RMCCD.fitPeakMultiROIs(mosaic.reshape((n0 * n1, n2, n3)),
                                    CentralPeaks,
                                    FittingParametersDict,
                                    showfitresults=False)
    # filter fit results
    params, cov, info, message, baseline = Resfit

    return filter_fitted_peaks(np.array(params), [inf["nfev"] for inf in info], baseline, CentralPeaks,
                                reject_negative_baseline=reject_negative_baseline,
                                reject_large_PixelDeviation=reject_large_PixelDeviation,
                                reject_weakPeaks=reject_weakPeaks,
                                FitPixelDev=FitPixelDev,
                                MinimumPeakAmplitude=MinimumPeakAmplitude,
                                verbose=verbose)


def filter_fitted_peaks(par, nfev, baseline, CentralPeaks,
                        reject_negative_baseline=True,
                        reject_large_PixelDeviation=True,
                        reject_weakPeaks=True,
                        FitPixelDev=10,
                        MinimumPeakAmplitude=25,
                        verbose=1):
    """
    filter results of peak fits (see FitPeakOnMap())

    :param par: array of fit results (nb fits, 7): background, amplitude, X, Y, sigma1, sigma2, angle
    :param nfev: nb of function evaluations of each fit
    :param baseline: starting baseline of each fit
    :param CentralPeaks: array (nb fits, 2) of ROIs centers
    :return: XY (masked array (nb fits, 2) of positions, rejected fits are masked),
        FilteredfittedPeaksData (masked array (nb fits, 10), see below)
    """
    par = np.asarray(par, dtype=float)
    nfev = np.asarray(nfev, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if FitPixelDev is None:
        FitPixelDev = np.inf

    peak_bkg = par[:, 0]
    peak_I = par[:, 1]
    peak_X = par[:, 2]
    peak_Y = par[:, 3]
    peak_fwaxmaj = par[:, 4]
    peak_fwaxmin = par[:, 5]
    peak_inclination = par[:, 6] % 360

    # pixel deviations from guessed initial position before fitting
    Xdev = peak_X - CentralPeaks[:, 0]
    Ydev = peak_Y - CentralPeaks[:, 1]
    #     print 'Xdev', Xdev
    #     print "Ydev", Ydev
    #     print "peak_X",peak_X
    #     print "peak_Y",peak_Y

    # --- --- PEAKS REJECTION -------------------------------

    to_reject = list(np.where(nfev > 1550)[0])
    if verbose and to_reject:
        print("too much iterations for", to_reject)
    # no fit (missing image)
    to_reject += list(np.where(np.isnan(peak_X) | np.isnan(peak_Y))[0])

    #                 if CCDLabel == 'FRELONID15_corrected':
    #                     reject_negative_baseline = False
    if reject_weakPeaks:
        cond = peak_I - peak_bkg < MinimumPeakAmplitude
        to_reject0 = np.where(cond)[0]
        print("to_reject0 peak amplitude < %d" % MinimumPeakAmplitude, to_reject0)
    else:
        to_reject0 = []

    # negative peak width
    if 1:
        cond = np.logical_or(peak_fwaxmaj <= 0, peak_fwaxmin <= 0)
        to_reject1 = np.where(cond)[0]
        print("to_reject1 negative peak width", to_reject1)
    else:
        to_reject1 = []

    # negative intensity rejection
    if reject_negative_baseline:
        to_reject2 = np.where((peak_bkg - baseline) < 0)[0]
        print("to_reject2 negative baseline", to_reject2)
    else:
        to_reject2 = []

    # too far found peak rejection
    if reject_large_PixelDeviation:
        to_reject3 = np.where(np.sqrt(Xdev ** 2 + Ydev ** 2) > FitPixelDev)[0]
        if len(to_reject3) > 0:
            print("to_reject3  too far from initial guess minimum pixel distance = %f"
                % FitPixelDev,
                to_reject3)
    else:
        to_reject3 = []

    ToR = (set(to_reject)
        | set(to_reject0)
        | set(to_reject1)
        | set(to_reject2)
        | set(to_reject3))  # to reject

    print("After fitting, %d/%d peaks have been rejected" % (len(ToR), len(CentralPeaks)))
    print("to reject indices: ", ToR)

    ToTake = set(np.arange(len(CentralPeaks))) - ToR

    #print("ToTake", ToTake)

    BoolToTake = False * np.ones(len(CentralPeaks))
    BoolToTake[list(ToTake)] = True

    BoolToMask = True * np.ones(len(CentralPeaks))
    BoolToMask[list(ToTake)] = False

    #print("BoolToMask", BoolToMask)

    #     print "where Masked", np.where(BoolToMask==True)
    maskedrows = np.where(BoolToMask == True)[0]

    # all peaks list building
    fittedPeaksData = np.array([peak_X,
                                peak_Y,
                                peak_I,
                                peak_fwaxmaj,
                                peak_fwaxmin,
                                peak_inclination,
                                Xdev,
                                Ydev,
                                peak_bkg,
                                maskedrows], dtype='object').T
    #                 DEFAULT_WRONG_X = np.nan
    #                 DEFAULT_WRONG_Y = np.nan
    #                 FilterX = np.where(BoolToTake, peak_X, DEFAULT_WRONG_X)
    #                 FilterY = np.where(BoolToTake, peak_Y, DEFAULT_WRONG_Y)

    FilterX0 = np.ma.array(peak_X)
    FilterY0 = np.ma.array(peak_Y)
    FilterX = np.ma.masked_where(BoolToMask, FilterX0)
    FilterY = np.ma.masked_where(BoolToMask, FilterY0)

    # all peaks list building
    fittedPeaksData = np.ma.array([peak_X,
                                    peak_Y,
                                    peak_I,
                                    peak_fwaxmaj,
                                    peak_fwaxmin,
                                    peak_inclination,
                                    Xdev,
                                    Ydev,
                                    peak_bkg,
                                    FilterX]).T
    #                 DEFAULT_WRONG_X = np.nan
    #                 DEFAULT_WRONG_Y = np.nan
    #                 FilterX = np.where(BoolToTake, peak_X, DEFAULT_WRONG_X)
    #                 FilterY = np.where(BoolToTake, peak_Y, DEFAULT_WRONG_Y)

    #     m =False*np.zeros_like(fittedPeaksData)
    #     m[maskedrows,:]=True*ones(10)
    #
    #     print "m",m
    #
    FilteredfittedPeaksData = np.ma.mask_rowcols(fittedPeaksData, axis=0)

    XY = np.ma.array([FilterX, FilterY]).T
    return XY, FilteredfittedPeaksData


def CollectData(param, outputfolder, ccdlabel="MARCCD165"):
    """

    """
    (dirname, filename, startind, endind, nb_lines, nb_images_per_line,
        peaklist, boxsize_row, boxsize_line, selectedcounters) = param

    print("selectedcounters", selectedcounters)

    nbpeaks = len(peaklist)

    #print("nbpeaks", nbpeaks)
    print("peaklist", peaklist)

    startind, endind = int(startind), int(endind)
    nb_lines, boxsize_row, boxsize_line = list(map(int, [nb_lines, boxsize_row, boxsize_line]))

    halfboxsize = boxsize_row, boxsize_line

    #    print "boxsize_row, boxsize_line", halfboxsize

    #        nb_image = endind - startind + 1
    nb_col = nb_images_per_line

    dic_carto = {}

    if nb_col > 0:
        for ind in range(startind, endind + 1, 1):
            _ind = ind - startind

            dic_carto[ind] = [_ind, _ind / nb_col, _ind % nb_col]
    else:
        print("!***!****!****!")
        print("something wrong with startind,endind,nb_lines")
        print("Cannot build map between sample position and image index")
        print("!***!****!****!")

    boxpixelsize = (2 * halfboxsize[0] + 1, 2 * halfboxsize[1] + 1)

    #        print "boxpixelsize", boxpixelsize

    try:
        mosaic = np.zeros((nb_lines, nb_col, boxpixelsize[0], boxpixelsize[1]))
    except MemoryError:
        print("!***!****!****!")
        print("something wrong with pixel center or boxsize")
        print("Cannot Prepare Image")
        print("!***!****!****!")

    tabindices = np.arange(startind, endind + 1, 1)
    nbimages = len(tabindices)

    CountersData = {"mean1D": np.zeros((nbpeaks, nbimages)),
                    "max1D": np.zeros((nbpeaks, nbimages)),
                    "ptp1D": np.zeros((nbpeaks, nbimages)),
                    "posX": np.zeros((nbpeaks, nbimages)),
                    "posY": np.zeros((nbpeaks, nbimages))}

    for k, imageindex in enumerate(tabindices):
        filename = IOimage.setfilename(filename, imageindex)

        filename = os.path.join(dirname, filename)
        #            print "filename", filename

        for grain_index, peak in enumerate(peaklist):
            try:
                framedim = DictLT.dict_CCD[ccdlabel][0]
                dataimage, framedim, fliprot = IOimage.readCCDimage(filename, CCDLabel=ccdlabel,
                                                                                dirname=None)

                center_pixel = (round(peak[0]), round(peak[1]))

                indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                            (halfboxsize[0], halfboxsize[1]),
                                                            framedim,
                                                            flipxycenter=0)
                imin, imax, jmin, jmax = indicesborders

                # avoid to wrong indices wHen slicing the data
                imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                            framedim=framedim)

                datcrop = dataimage[imin:imax, jmin:jmax]

            except IOError:
                print("!***!****!****!")
                print("something wrong with image. cannot find %s" % filename)
                print("!***!****!****!")
                break

            if k % 10 == 1:
                print(filename, "up to", IOimage.setfilename(filename, endind))

            piece_dat = datcrop

            title = "indexrange [%06d-%06d]" % (startind, endind)

            print("selectedcounters", selectedcounters)

            for counter in selectedcounters:
                print("counter", counter)

                if counter in ("mean", "max", "ptp"):

                    if counter == "mean":
                        CountersData["mean1D"][grain_index, k] = np.mean(piece_dat)
                    elif counter == "max":
                        CountersData["max1D"][grain_index, k] = np.amax(piece_dat)
                    elif counter == "ptp":
                        CountersData["ptp1D"][grain_index, k] = np.ptp(piece_dat)

                elif counter in ("Position XY",):

                    datminimum = scind.measurements.maximum(piece_dat)
                    # center of mass without background removal
                    datcenterofmass = np.array(scind.measurements.center_of_mass(piece_dat))
                    # remove baseline level set to minimum pixel intensity
                    datcenterofmass2 = np.array(scind.measurements.center_of_mass(piece_dat - datminimum))

                    print("datcenterofmass2", datcenterofmass2)

                    centerofmass = datcenterofmass2 + np.array([jmin, imin])

                    datmaximumpos = scind.measurements.maximum_position(piece_dat)
                    datmaximumpos = np.array(datmaximumpos, dtype="uint32")

                    posmax = datmaximumpos + np.array([jmin, imin])

                    nbimages = endind - startind + 1

                    # position monitors selection
                    XY = posmax
                    XY = centerofmass

                    xDATA, yDATA = XY

                    CountersData["posX"][grain_index, k] = xDATA
                    CountersData["posY"][grain_index, k] = yDATA

    return CountersData


def CollectData_oneImage(param, outputfolder, ccdlabel="MARCCD165",
                selectedcounters=("Imean", "Imax", "Iptp", "posX", "posY"), ndivisions=(1,15), imageheaderinfos=False):
    """
    return dictionary of counters

    if selectedcounters have  Imean_multiple, Imax_multiple, Iptp_multiple, then array will subdivided
    according to nbdivsions = (n1,n2) ie n1*n2 subarrays.
    n1 divisions along slow axis (Y, vert), n2 along fast axis (X, horiz)

    param  = (dirname, filename, imageindex, peaklist, boxsize_row, boxsize_line)

    peaklist : list of pixel X pixel Y (horiz, vert)   . For multiple detector : box around [[X, Y]] will be split
    boxsize_row = half boxsize (in pixel) // pixel X axis horiz
    boxsize_line = half boxsize (in pixel) // pixel Y axis vert

    # todo add sCMOS 9M 4M ?? maybe this function is obsolete / datacollector.py
    """
    (dirname, filename, imageindex, peaklist, boxsize_row, boxsize_line) = param

    #    print "selectedcounters", selectedcounters

    nbpeaks = len(peaklist)

    #print("nbpeaks", nbpeaks)

    boxsize_row, boxsize_line = list(map(int, [boxsize_row, boxsize_line]))

    halfboxsize = boxsize_row, boxsize_line

    dic_carto = {}

    boxpixelsize = (2 * halfboxsize[0] + 1, 2 * halfboxsize[1] + 1)

    CountersData = {}
    for counter in selectedcounters:
        if 'multiple' not in counter:
            CountersData[counter] = np.zeros(nbpeaks)
        else:
            ny, nx = ndivisions
            CountersData[counter] = np.zeros((nbpeaks, nx*ny))

            CountersData["posmax_multiple"] = np.zeros((nbpeaks, nx*ny, 2))
    CountersData["Monitor"] = 1.0
    CountersData["ExposureTime"] = 1000.0  # milliseconds

    filename = IOimage.setfilename(filename, imageindex)

    filename = os.path.join(dirname, filename)

    try:
        framedim = DictLT.dict_CCD[ccdlabel][0]
        dataimage, framedim, fliprot = IOimage.readCCDimage(filename, CCDLabel=ccdlabel,
                                                                                dirname=None)
    except IOError:
        print("!***!****!****!")
        print("Missing image file : %s" % filename)
        print("!***!****!****!")
        raise IOError

    for peak_index, peak in enumerate(peaklist):
        if imageheaderinfos:
            if ccdlabel == "MARCCD165":
                comments, expo_time = IOimage.read_header_marccd2(filename)
                I0 = float(comments.split()[3])
                CountersData["Monitor"] = I0
                CountersData["ExposureTime"] = expo_time
            elif ccdlabel == "sCMOS":
                dictpar = IOimage.read_header_scmos(filename)
                if dictpar is not {}:
                    CountersData["Monitor"] = dictpar.get("mon",None)
                    CountersData["ExposureTime"] = dictpar.get("exposure", None)  # in milliseconds
        
        center_pixel = (round(peak[0]), round(peak[1]))

        indicesborders = ImProc.getindices2cropArray((center_pixel[0], center_pixel[1]),
                                                (halfboxsize[0], halfboxsize[1]),
                                                framedim,
                                                flipxycenter=0)
        imin, imax, jmin, jmax = indicesborders

        # avoid to wrong indices when slicing the data
        imin, imax, jmin, jmax = ImProc.check_array_indices(imin, imax + 1, jmin, jmax + 1,
                                                                                framedim=framedim)

        piece_dat = dataimage[imin:imax, jmin:jmax]

        #print('piece_dat',piece_dat)

        if 'multiple' in counter:  #split array into several subarrays
            piece_dat, _, (box1, box2) = GT.splitarray(piece_dat, ndivisions)

        for counter in selectedcounters:

            if counter.startswith("I"):
                if 'multiple' not in counter:
                    if counter == "Imean":
                        CountersData["Imean"][peak_index] = np.mean(piece_dat)
                    elif counter == "Imax":
                        CountersData["Imax"][peak_index] = np.amax(piece_dat)
                    elif counter == "Iptp":
                        CountersData["Iptp"][peak_index] = np.ptp(piece_dat)
                else:
                    if counter == "Imean_multiple":
                        CountersData["Imean_multiple"][peak_index] = np.mean(piece_dat, axis=(1, 2))
                    elif counter == "Imax_multiple":
                        CountersData["Imax_multiple"][peak_index] = np.amax(piece_dat, axis=(1, 2))
                    elif counter == "Iptp_multiple":
                        CountersData["Iptp_multiple"][peak_index] = np.ptp(piece_dat, axis=(1, 2))

            elif counter.startswith("pos"):

                if 'multiple' in counter:
                    if counter == "posmax_multiple":
                        nbrois = ndivisions[0] * ndivisions[1]
                        labels = np.repeat(np.arange(nbrois), box1 * box2).reshape((nbrois, box1, box2))
                        #print('labels.shape',labels.shape)
                        index = np.arange(nbrois)
                        datmaximumpos = np.array(scind.measurements.maximum_position(piece_dat, labels=labels, index=index))
                        CountersData["posmax_multiple"][peak_index] = datmaximumpos[:, 1:]

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

                    CountersData["posX"][peak_index] = xDATA
                    CountersData["posY"][peak_index] = yDATA

    return CountersData
