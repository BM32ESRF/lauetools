# -*- coding: utf-8 -*-
"""
Created on Fri May 03 18:18:53 2013

initial @author: odile32

revised and refactored by JS Micha in September 2019 for python3 compatibility

.. todo::
    still a lot of refactorings to do
"""

import os
import sys
import time
import struct
import pickle
import copy
import gzip

print(("diamond module located in", __file__))

import matplotlib.pyplot as p

import numpy as np
#from numpy import round
import scipy.signal as sg
import scipy.ndimage as snd
import scipy.optimize as so

if sys.version_info.major == 3:
    from .. import LaueGeometry as F2TC
    from .. import readmccd as rmccd
    from .. import dict_LaueTools as DictLT
    from .. import findorient as FindO
    from .. import CrystalParameters as CP
    from ..generaltools import norme_vec as norme
    from .. import generaltools as GT
    from .. import tifffile as TIFF
    from . import spec_reader as SR
    from .. import IOimagefile as IOimage

    # WARNING  looped imports
    from . import multigrain as MG

else:
    import LaueGeometry as F2TC
    import readmccd as rmccd
    import dict_LaueTools as DictLT
    import findorient as FindO
    import CrystalParameters as CP
    from generaltools import norme_vec as norme
    import generaltools as GT
    from . import multigrain as MG
    from . import spec_reader as SR
    from .. import tifffile as TIFF
    from .. import IOimagefile as IOimage


def fromimageindexTothf(imageindex, scan):
    """ convert image index to diamond thf angle according to scan object
    """
    return np.interp(imageindex, scan.img, scan.thf)


def fromthfToimageindex(thf, scan):
    """ convert diamond thf angle to image index according to scan object
    """
    return np.interp(thf, scan.thf, scan.img)


# 1D case   for Intensity
def normalizeprofile(Intensity, pedestal, mon_in_scan, mon_offset):

    yy = ((Intensity - pedestal)
        / (Intensity[0] - pedestal)
        * (mon_in_scan[0] - mon_offset)
        / (mon_in_scan[:] - mon_offset))
    return yy


# Multiple profiles normalization
def normalizeprofile_multipledips(data_Dips, pedestal, mon_in_scan, mon_offset):
    """
    compute normalized (by monitor) intensity profiles

    :param data_Dips:    raw intensity profiles
    :type data_Dips: array (n,m) with n = nb of profiles (or peak or pixel at (X, Y)) and m nb of points in scan
    :param pedestal:    added intensity offset by detector manufacturer
    :type pedestal: float, integer
    :param mon_in_scan: monitor (incoming flux) values during scan, length = nb of points in scan
    :type mon_in_scan: list, array
    :param mon_offset: offset of monitor (i.e. residual monitor counts without incoming x ray)
    :type mon_offset: float, int

    :return: normalized intensity profiles
    :rtype: array, shape (n,m)
    """

    IntensityProfiles = (data_Dips[:, 1:]).T

    shapeprof = np.shape(IntensityProfiles)

    print("shapeprof", shapeprof)

    ndips = np.zeros(shapeprof)

    for profindex in range(len(IntensityProfiles)):
        Intensity = IntensityProfiles[profindex]
        ndips[profindex] = ((Intensity - pedestal) / (Intensity[0] - pedestal)
            * (mon_in_scan[0] - mon_offset)
            / (mon_in_scan[:] - mon_offset))
    return ndips


def dipSearch(data_Dips, stdmax=0.03, scan=None, DipDepthThreshold=0.06, pedestal=1000, Navg1=150, Navg2=5):
    """ detect dips from raw data array of profiles

    return:
    list of :
    profileindex, nbDipsFound, dipamplitude, dippositionimagindex, [dippositionthf]

    """
    # 8 16 18 12 66 69

    # DipDepthThreshold = 0.06  # 6% of attenuation

    # fdips=smooth(ndata_Dips[kprof],window_len=301)

    # We create a Butterworth low-pass filter.  (no dip, no noise)  for avgdip
    b, a = sg.butter(2, 1.0 / Navg1)

    # We create a Butterworth low-pass filter. (dip , no noise) for avgdip2
    b2, a2 = sg.butter(3, 1.0 / Navg2)

    DipsArray = []

    _, m = np.shape(data_Dips)
    nbprofiles = m - 1
    # normaize data with monitor and first element of intensity profile
    ndata_Dips = normalizeprofile_multipledips(data_Dips, pedestal, scan.Monitor, scan.monitoroffset)

    dictProfileswithdips = {}

    for kprof in range(nbprofiles):

        mydip = ndata_Dips[kprof]

        # low freq signal baseline higly smoothed
        avgdip = sg.filtfilt(b, a, mydip)

        # low freq signal (with dips , small noise) = slightly smoothed
        avgdip2 = sg.filtfilt(b2, a2, mydip)

        # deviation from low freq signal (filtered signal)
        fdip = 1 - (avgdip2 / avgdip)

        Std = np.std(avgdip2 - mydip)

        print(("\n**********\nkprof", kprof))
        print("std error", Std)
        cprofile = fdip
        #         tcprofile=np.where(cprofile>max(1.5*Std,DipDepthThreshold), 1,0)
        tcprofile = np.where(cprofile > max(1.5 * Std, DipDepthThreshold), 1, 0)

        offset_pos = 0
        # print "cprofile",cprofile
        # print "tcprofile",tcprofile
        ll, nf = snd.label(tcprofile)
        # print "ll,nf",ll, nf

        meanpos = np.array(snd.measurements.center_of_mass(cprofile, ll, np.arange(1, nf + 1)),
                                                            dtype=np.float)
        peakpos = []
        dipamplitude = []
        nbDipsFound = 0
        if Std >= stdmax:
            print("Signal is too noisy to find dips! So")
            meanpos = []

        if len(meanpos) > 0:

            peakpos = np.array(meanpos[:, 0], dtype=int)
            dipamplitude = 1 - avgdip2[peakpos + offset_pos] / avgdip[peakpos + offset_pos]

            nbDipsFound = len(peakpos)

            print("peak position (imageindex)", peakpos + offset_pos)
            print("dip amplitude %", dipamplitude * 100)

            if scan is not None:
                DipsArray.append([kprof, nbDipsFound, dipamplitude, peakpos,
                                                                fromimageindexTothf(peakpos, scan)])
            else:
                DipsArray.append([kprof, nbDipsFound, dipamplitude, peakpos])

        else:
            print("no dip found !!!")

        dictProfileswithdips[kprof] = [mydip, avgdip, avgdip2, fdip, peakpos]

    return DipsArray, dictProfileswithdips


def readDiamondScan(spec_file_path, scanindex, MonitorOffset=0, exportcountersdata=None,
                                                                    add_str_file_Ipix_vs_img=""):
    """
    read scan of rotating diamond (thf angle)

    param:
    MonitorOffset  :  offset of monitor per second (without Xray)

    :return:
        -scanObject
        -scanDict
    :rtype: dict, dict

    """
    sf = SR.SpecFile(spec_file_path, verbose=False)

    scan = SR.Scan(sf, scanindex)

    if not hasattr(scan, "thf"):
        print("this scan has not a varying motor thf !!")
        return None

    Epoch_0 = scan.Epoch[0]
    print("scan = ", scanindex)
    print("command = ", scan.command)
    #    print "yech = ", scan.motors["yech"]
    print("first img = ", scan.M)
    print("scan.counters", scan.counters)

    Epoch_in_scans = scan.Epoch - Epoch_0

    #    yech_in_scans_as_line = np.renp.shape(yech_in_scans_as_matrix, (nscans*npts_per_scan, 1))

    npts = len(scan.Epoch)
    img_in_scan = np.array(scan.img + 0.0001, int)

    if img_in_scan[-1] - img_in_scan[0] < 1:
        first_img = int(scan.M[0][-8:-4])
        print(first_img)
        img_in_scan = np.arange(npts) + first_img

        print("recalculated img_in_scan")
        print(img_in_scan)

    _ = scan.Monitor

    thf_in_scan = scan.thf

    print("thf_in_scan = ", thf_in_scan)

    thf_cen = (thf_in_scan[-1] + thf_in_scan[0]) / 2.0
    #    thf_cen = thf_in_scan[0]
    print("thf_cen (center of scan) = ", thf_cen)

    delta_thf_center_end = (thf_in_scan[-1] - thf_in_scan[0]) / 2.0
    delta_thf_center_end = int(np.round(delta_thf_center_end, 1))

    mon_offset = MonitorOffset * scan.ct

    add_str_file_Ipix_vs_img = add_str_file_Ipix_vs_img + "_scan" + str(scanindex)

    print("add_str_file_Ipix_vs_img = ", add_str_file_Ipix_vs_img)

    scanDatadict = {"thf": scan.thf,
                    "imageindex": scan.img,
                    "Monitor": scan.Monitor,
                    "Epoch": scan.Epoch}
    if exportcountersdata is not None:
        for counter in exportcountersdata:
            if counter in scan.counters:
                scanDatadict[counter] = getattr(scan, counter)

    scanDict = {"Epoch_0": Epoch_0,
            "Epoch_in_scans": Epoch_in_scans,
            "ExposureTime": scan.ct,
            "img_in_scan": img_in_scan,
            "thf_in_scan": thf_in_scan,
            "thf_cen": thf_cen,
            "half_thfscan_apmlitude": delta_thf_center_end,
            "mon_offset_scan": mon_offset,
            "string_suffix": add_str_file_Ipix_vs_img,
            "scanDatadict": scanDatadict}

    return scan, scanDict


def correctIncomingdirection(geometry_corrections, ui=np.array([0.0, 0.0, 0.0])):
    """ apply geometrical corrections to incoming beam direction ui
    """
    ui_pitch, ui_yaw = geometry_corrections[0], geometry_corrections[1]
    # dirv dirh : 1e-4 rad
    dirv = ui_pitch * 1.0e-4
    dirh = ui_yaw * 1.0e-4
    cdirv = np.cos(dirv)
    sdirv = np.sin(dirv)
    cdirh = np.cos(dirh)
    sdirh = np.sin(dirh)
    uilab = np.array([cdirv * sdirh, cdirv * cdirh, sdirv])
    #     print "uilab = ", uilab
    return uilab


def plotDiamondBranchAbacus(diamondbranchIndex, abacusfilepath,
                                                geometry_corrections=np.array([0.0, 0.0, 0.0, 0.0]),
                                                thf_cen=-47.75,
                                                thf_half_range=22,
                                                plot_thf_ref=-48,
                                                verbose=1,
                                                plotgraphs=True,
                                                figsize=(5, 5)):
    """
    plot single diamond branch properties

    """
    CST_CONV_LAMBDA_KEV = 12.398

    thf_ref = plot_thf_ref
    dthf = thf_half_range

    #     p.rcParams["figure.subplot.right"] = 0.9
    #     p.rcParams["figure.subplot.left"] = 0.1
    #     p.rcParams["figure.subplot.top"] = 0.9
    #     p.rcParams["figure.subplot.bottom"] = 0.1

    dict_Edia, dict_values_names, thf_list = read_dict_Edia_vs_thf(abacusfilepath)

    ind_hkldia = dict_values_names.index("hkldia")
    ind_uq0 = dict_values_names.index("uqlab_0")
    ind_Edia = dict_values_names.index("Edia_list")

    npts = len(thf_list)
    branchPropsValues = dict_Edia[diamondbranchIndex]
    #     print "branchPropsValues",branchPropsValues
    print("table limited in thf to : [", thf_cen - dthf, thf_cen + dthf, "]")
    print("diamondbranchIndex (ndia_ref) = ", diamondbranchIndex)
    print("HKLdia = ", branchPropsValues[ind_hkldia])
    HKLdia = np.array(branchPropsValues[ind_hkldia].round(decimals=0), dtype=int)
    if verbose:
        print("thf (°), Edia (keV), EnergyCorrected, [uqxlab,uqylav,uqzlab), tth (°), chi (°)")
    ind1 = []

    uilab = correctIncomingdirection(geometry_corrections)
    print("correct uilab", uilab)

    chi_list = np.zeros(npts, float)
    tth_list = np.zeros(npts, float)
    sign_uflabz = np.ones(npts, int)
    for i in range(npts):
        if ((thf_list[i] > thf_cen - dthf)
            & (thf_list[i] < thf_cen + dthf)
            & (dict_Edia[diamondbranchIndex][ind_Edia][i] > 1.0)):

            uqlab = np.array([branchPropsValues[ind_uq0][i],
                            branchPropsValues[ind_uq0 + 1][i],
                            branchPropsValues[ind_uq0 + 2][i]])
            sintheta = -1.*np.inner(uqlab, uilab)
            uflab = uilab + 2 * sintheta * uqlab
            if uflab[2] < 0.0:
                sign_uflabz[i] = -1
            chi_list[i], tth_list[i] = MG.uflab_to_2thetachi(uflab)

            #             Energy = CST_CONV_LAMBDA_KEV * norme_qflab ** 2 / (2. * np.sin(th0 * DEG))
            norme_qlab = np.sqrt(np.sum(uqlab ** 2))
            EnergyCorrected = CST_CONV_LAMBDA_KEV * norme_qlab ** 2 / (2.0 * sintheta)

            if verbose:
                print(thf_list[i],
                        branchPropsValues[ind_Edia][i],
                        EnergyCorrected,
                        uqlab,
                        np.round(tth_list[i], 2),
                        np.round(chi_list[i], 2))
            ind1.append(i)
    #    print ind1
    #    print dict_Edia[ndia_ref][2][ind1]

    # ----------  DISPLAY   --------------------
    plotkwargs = {}
    plotkwargs["figsize"] = figsize
    if plotgraphs:
        _, ax1 = p.subplots(figsize=figsize)
        title1 = "Diamond Branch = " + str(diamondbranchIndex) + ", HKL diamond = " + str(HKLdia) + "\n"

        ax1.plot(thf_list[ind1], branchPropsValues[ind_Edia][ind1] / 1000.0, "ko-", ms=5)
        ax1.set_xlabel("thf (deg)")
        ax1.set_ylabel("Edia (keV)")
        ax1.axvline(x=thf_ref)
        ax1.set_ylim(5.0, 22.0)
        ax1.set_title(title1, fontsize=14)
        ax1.grid()

        _, ax2 = p.subplots()

        ax2.plot(thf_list[ind1], tth_list[ind1], "ro-", ms=5)
        ax2.set_ylabel("tth (deg)")
        ax2.set_xlabel("thf (deg)")
        ax2.axvline(x=thf_ref)

        ax2.axhline(y=90.0)
        ax2.set_title(title1, fontsize=14)
        ax2.grid()

        _, ax3 = p.subplots()

        ax3.plot(thf_list[ind1], chi_list[ind1], "ro-", ms=5)
        ax3.set_ylabel("chi (deg)")
        ax3.set_xlabel("thf (deg)")
        ax3.axvline(x=thf_ref)

        ax3.axhline(y=0.0)
        ax3.set_title(title1, fontsize=14)
        ax3.grid()

        _, ax4 = p.subplots()

        ax4.plot(thf_list[ind1], sign_uflabz[ind1], "ro-", ms=5)
        ax4.set_ylabel("signe uflab_z")
        ax4.set_xlabel("thf (deg)")
        ax4.axvline(x=thf_ref)
        ax4.set_ylim(-1.1, 1.1)
        ax4.axhline(y=0.0)
        ax4.set_title(title1, fontsize=14)
        ax4.grid()


def getthftheo(dictDipsTheo, profileindex, diamondbranchindex):
    profileindex = int(profileindex)
    diamondbranchindex = int(diamondbranchindex)
    #     print 'dictDipsTheo.iloc[profileindex]',dictDipsTheo.iloc[profileindex]

    branchlocation = np.where(dictDipsTheo.iloc[profileindex]["#branchDia"] == diamondbranchindex)[0]
    #     print branchlocation
    thf_theo = dictDipsTheo.iloc[profileindex]["thf_theo"][branchlocation]
    print('thf_theo', thf_theo)
    if isinstance(thf_theo, np.ndarray):
        thf_theo = thf_theo[0]
    return thf_theo


def getHKLdiamondBranch(dictDipsTheo, profileindex, diamondbranchindex):
    diamondbranchindex = int(diamondbranchindex)
    profileindex = int(profileindex)

    branchlocation = np.where(dictDipsTheo.iloc[profileindex]["#branchDia"] == diamondbranchindex)[0]
    hkldiamondbranch = dictDipsTheo.iloc[profileindex]["HKLDia_branches"][branchlocation]
    return hkldiamondbranch


def computeDeltaThfs(selectedDips, dictDipsTheo):

    listdeltathf = []
    for diplinkprops in selectedDips:
        print("diplinkprops", diplinkprops)
        profileindex, thf_exp = diplinkprops[1:3]
        branchindex = diplinkprops[-1]
        #     print branchindex
        thf_theo = getthftheo(dictDipsTheo, profileindex, branchindex)

        deltathf = thf_theo - thf_exp
        #         print "deltathf : %f  thf: theo :  %f,  exp: %f"%(deltathf,thftheo,thf_exp)
        listdeltathf.append(deltathf)

    return listdeltathf


def computeDeltaEnergys(selectedDips, dictDipsTheo, calibdia):
    #     print "selectedDips in computeDeltaEnergys",selectedDips
    listdeltaenergy = []
    for diplinkprops in selectedDips:
        profileindex, thf_exp, E_exp = diplinkprops[1:4]
        branchindex = int(diplinkprops[-1])
        print('branchindex', branchindex)
        hkldia = getHKLdiamondBranch(dictDipsTheo, profileindex, branchindex)[0]
        print("hkldia", hkldia)
        E_theo = 1.2398 / thf_dip_to_lambda_dia_v2(hkldia, thf_exp, calibdia) * 1000

        deltaE = E_theo - E_exp
        #         print "deltaE : %f  E: theo :  %f,  exp: %f"%(deltaE,E_theo,E_exp)
        listdeltaenergy.append(deltaE)

    return listdeltaenergy


def plotSingleProfilewithDips(dictProfileswithdips, pdspotwh, profileindex):
    """
    plot intensity profile and dips found

    param:

    profileindex  : int   intensity profile index (with harmonics)

    """
    k_prof = profileindex
    spotindex = pdspotwh.iloc[k_prof]["spotindex"]

    mydip, avgdip, avgdip2, fdip, peakpos = dictProfileswithdips[k_prof]

    _, ax = p.subplots(figsize=(12, 3))
    ax.plot(mydip, label="normalized")

    ax.vlines(peakpos, np.amin(mydip), np.amax(mydip), color="g")
    ax.set_xlabel("imageindex")
    ax.legend()
    ax.set_title("#profile (k_prof) %d #spot (spotindex): %d" % (k_prof, spotindex))
    ax.grid()

    _, ax = p.subplots(figsize=(12, 3))
    ax.plot(mydip, label="norm.")
    ax.plot(avgdip, label="avg.")
    ax.plot(avgdip2, label="no noise")

    ax.set_xlabel("imageindex")
    ax.legend()
    ax.grid()

    _, ax = p.subplots(figsize=(12, 3))
    ax.plot(fdip, label="diff")
    # ax.scatter(peakpos,cprofile[peakpos], c='r')

    ax.vlines(peakpos, np.amin(fdip), np.amax(fdip), color="r")

    ax.set_xlabel("imageindex")
    ax.legend()
    ax.grid()


# str1 = "." + os.path.sep + "FileSeries"
# sys.path.append(str1)
# from multigrain_jsm import readfitfile_multigrains

# from FileSeries.multigrain import readfitfile_multigrains

# warning : matstarlab (matline) is in OR's lab frame, matLT (mat3x3) in LaueTools lab frame
# zLT = zOR, xLT = yOR, xOR = - yLT
# incident beam along +y in OR lab frame, along +x in LT lab frame

# dans matwithlatpar le parametre de maille est en nm

# needed for calc_diamond_rotation_axis
MG.init_numbers_for_crystal_opsym_and_first_stereo_sector(elem_label="DIAs")


def calc_calibdia(matref_center, vlab_start_end, thf_ref, geometry_corrections):

    dlatu_angstroms_deg = DictLT.dict_Materials["DIAs"][1]
    dlatu_nm_rad = MG.deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
    print("dlatu_nm_rad = ", dlatu_nm_rad)
    matref_withlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matref_center, dlatu_nm_rad)
    print("matref with latpar", matref_withlatpar_inv_nm)
    print("normes of matrix columns should be 2.804")
    print(MG.norme(matref_withlatpar_inv_nm[0:3]),
        MG.norme(matref_withlatpar_inv_nm[3:6]),
        MG.norme(matref_withlatpar_inv_nm[6:9]))

    ui_pitch = geometry_corrections[0]
    ui_yaw = geometry_corrections[1]
    axis_yaw = geometry_corrections[2]
    axis_roll = geometry_corrections[3]

    # normalisation:
    vlab_s_e = vlab_start_end / MG.norme(vlab_start_end)

    aya = axis_yaw * 1.0e-4
    aro = axis_roll * 1.0e-4
    vlab = vlab_s_e + np.array([0.0, aya, aro])
    vlab_corr = vlab / MG.norme(vlab)
    print("vlab_corr = ", vlab_corr)

    # dirv dirh : 1e-4 rad
    dirv = ui_pitch * 1.0e-4
    dirh = ui_yaw * 1.0e-4
    cdirv = np.cos(dirv)
    sdirv = np.sin(dirv)
    cdirh = np.cos(dirh)
    sdirh = np.sin(dirh)
    uilab_corr = np.array([cdirv * sdirh, cdirv * cdirh, sdirv])
    print("uilab_corr = ", uilab_corr)

    # calibdia

    #    mat_range = range(0,9)
    #    vlab_range = range(9,12)
    #    uilab_range = range(12,15)
    #    thfref_range = 15
    #
    calibdia = np.hstack((matref_withlatpar_inv_nm, vlab_corr, uilab_corr, thf_ref))

    return calibdia


# ---   --------- build_spotlistref_sample -------
def build_spotlistref_sample(filefit_sample_max_npeaks=None,
                            filefit_sample_min_pixdev=None,
                            filedat_sample=None,  # verifier derniere colonne Ipixmax dans dat file
                            skip_unindexed_spots=0,
                            elem_label="Ge",
                            fitfile_type="GUI_strain",
                            fileextensionmarker=".cor",  # no more used, kept for compatibility,
                            pixelsize=DictLT.dict_CCD[MG.PAR.CCDlabel][1],
                            min_matLT=0):  # MG multigrain, GUI, FS fileseries
    """Write a file with exhaustive list of spots properties with harmonics from .fit and .dat file.
    Outputname is filedat_sample with addition of '_spotlistref_with_harmonics.dat' """

    if filefit_sample_max_npeaks is None:
        filefit_sample_max_npeaks = filefit_sample_min_pixdev
    if filefit_sample_min_pixdev is None:
        filefit_sample_min_pixdev = filefit_sample_max_npeaks

    ind_Ipixmax_in_datfile = -1
    ind_xy_in_datfile = list(range(2))
    ind_dxy_in_datfile = list(range(7, 9))

    res1 = MG.read_any_fitfitfile_multigrain(filefit_sample_max_npeaks,
                                            verbose=1,
                                            fitfile_type=fitfile_type,
                                            check_Etheor=1,
                                            elem_label=elem_label,
                                            check_pixdev=1,
                                            pixelsize=pixelsize,
                                            min_matLT=min_matLT,
                                            check_pixdev_JSM=0)

    print("res1 = ", res1)

    (_, _, _, _,
    data_fit_all, _, _, _, _,
    ind_h_x_int_pixdev_Etheor) = res1

    res2 = MG.read_any_fitfitfile_multigrain(filefit_sample_min_pixdev,
                                            verbose=1,
                                            fitfile_type=fitfile_type,
                                            check_Etheor=1,
                                            elem_label=elem_label,
                                            check_pixdev=1,
                                            pixelsize=pixelsize,
                                            min_matLT=min_matLT,
                                            check_pixdev_JSM=0)

    (_, _, _, matstarlab_all2,
    _, calib_all2, _, _, _,
    _) = res2

    indh = ind_h_x_int_pixdev_Etheor[0]
    indx = ind_h_x_int_pixdev_Etheor[1]
    ind_Etheor_in_data_fit = ind_h_x_int_pixdev_Etheor[4]

    print("Efit = ", data_fit_all[:, ind_Etheor_in_data_fit])

    ind_hkl_in_data_fit = list(range(indh, indh + 3))
    ind_xy_in_data_fit = list(range(indx, indx + 2))

    matstarlab2 = matstarlab_all2[0]

    calib2 = calib_all2[0]

    data_fit = np.array(data_fit_all, float)

    print("list of spots from filefit_sample_max_npeaks")
    print(data_fit)

    # la on utilise le .fit avec max npeaks
    nfit = np.shape(data_fit)[0]

    data_xyfit = np.array(data_fit[:, ind_xy_in_data_fit], dtype=float)

    #    data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype="LT")

    data_dat0 = np.loadtxt(filedat_sample, skiprows=1)

    data_dat0 = np.array(data_dat0, dtype=float)

    data_dat = MG.sort_list_decreasing_column(data_dat0, ind_Ipixmax_in_datfile)

    # data_dat = sorted, with final order

    ndat = np.shape(data_dat)[0]

    data_new = np.zeros((ndat, 13), dtype=float)  # modif 10Feb14

    data_xy = data_dat[:, ind_xy_in_datfile]
    data_xy_integer = data_xy - data_dat[:, ind_dxy_in_datfile]

    # print data_xy_integer
    data_new[:, 0] = list(range(ndat))
    data_new[:, 1:3] = data_xy_integer * 1.0
    data_new[:, 3:5] = data_xy * 1.0
    data_new[:, 8] = data_dat[:, ind_Ipixmax_in_datfile]

    dxy_tol = 0.2

    for i in range(ndat):
        for k in range(nfit):
            dxy = norme(data_xy[i] - data_xyfit[k])
            if dxy < dxy_tol:
                data_new[i, 5:8] = data_fit[k, ind_hkl_in_data_fit] * 1.0
                data_new[i, 9] = data_fit[k, ind_Etheor_in_data_fit]
    #    print data_new[:,5:8]

    # calcul Etheor avec matrice avec min pixdev

    dlatu_angstroms_deg = DictLT.dict_Materials[elem_label][1]
    dlatu_nm_rad = MG.deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
    print("dlatu_nm_rad = ", dlatu_nm_rad)
    matwithlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matstarlab2, dlatu_nm_rad)
    print("mat with latpar", matwithlatpar_inv_nm)

    cryst_struct = DictLT.dict_Materials[elem_label][2]
    print("cryst_struct = ", cryst_struct)

    showall = 1

    spotlistnew = MG.spotlist_gen(5.0,
                                    MG.PAR.energy_max,
                                    "top",
                                    matwithlatpar_inv_nm,
                                    cryst_struct,
                                    showall,
                                    calib2,
                                    remove_harmonics="no")

    spotlistnew = np.array(spotlistnew, dtype=float)
    nspots_theor = np.shape(spotlistnew)[0]
    hkltheor = spotlistnew[:, :3]
    Etheor = spotlistnew[:, 9]

    # add 10Feb14 colonnes pour uqlab
    uqlab1 = np.zeros((nspots_theor, 3), float)
    uilab = np.array([0.0, 1.0, 0.0])
    for i in range(nspots_theor):
        uqlab = spotlistnew[i, 3:6] - uilab
        uqlab1[i, :] = uqlab / MG.norme(uqlab)

    if 0:  # add single spot
        hklref = np.array([-9.0, -5.0, 5.0])
        # print "hklref = ", hklref
        uqcr_ref = hklref / norme(hklref)
        for k in range(nspots_theor):  # test de tous les spots theor harmon incluse
            uqcr = hkltheor[k, :] / norme(hkltheor[k, :])
            duq = uqcr - uqcr_ref
            if norme(duq) < 1.0e-3:
                print("hkltheor = ", hkltheor[k, :])
                print(Etheor[k])
                raise ValueError

    firsttime = 1
    nadd = 0
    for i in range(ndat):
        tt = norme(data_new[i, 5:8])
        if tt > 1.0e-3:  # pics du .dat non indexes avec max npeaks ont garde hkl = 000
            # print "i =", i
            hklref = data_new[i, 5:8]
            # print "hklref = ", hklref
            uqcr_ref = data_new[i, 5:8] / tt
            for k in range(nspots_theor):  # test de tous les spots theor harmon incluse
                uqcr = hkltheor[k, :] / norme(hkltheor[k, :])
                duq = uqcr - uqcr_ref
                if norme(duq) < 1.0e-3:
                    # print "hkltheor = ", hkltheor[k,:]
                    if norme(hkltheor[k, :] - hklref) < 1.0e-3:  # harmon deja presente dans .fit
                        if Etheor[k] > 0.1:
                            data_new[i, 9] = Etheor[k]
                            data_new[i, 10:13] = uqlab1[k, :]
                        # print "already in list from .fit"
                    else:  # harmonique absente dans .fit
                        if firsttime:
                            # print "first time"
                            data_add = data_new[i] * 1.0
                            data_add[9] = Etheor[k]
                            data_add[5:8] = hkltheor[k, :]
                            data_add[10:13] = uqlab1[k, :]
                            firsttime = 0
                            # print "np.shape(data_add) =", np.shape(data_add)
                            # print data_add[0]
                        else:
                            data_add = np.row_stack((data_add, data_new[i]))
                            data_add[-1, 9] = Etheor[k]
                            data_add[-1, 5:8] = hkltheor[k, :]
                            data_add[-1, 10:13] = uqlab1[k, :]
                            # print "next"
                            # print "np.shape(data_add) =", np.shape(data_add)
                            # print data_add[:,0]
                        nadd += 1

    if skip_unindexed_spots:
        print("skip unindexed spots")
        dhkl = abs(data_new[:, 5:8]).sum(axis=1)
        ind0 = np.where(dhkl > 0.99)
        if len(ind0[0]) > 0:
            print("keep ", len(ind0[0]), "spots over ", np.shape(data_new)[0])
            data_new = data_new[ind0[0], :]
            ndat = np.shape(data_new)[0]

    print("data_new :")
    print("numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9, uqlab 10:13")
    for i in range(ndat):
        print(int(np.round(data_new[i, 0], 0)),
            "\t",
            np.array(data_new[i, 1:3].round(decimals=0), dtype=int),
            data_new[i, 3:5],
            np.array(data_new[i, 5:8].round(decimals=0), dtype=int),
            "\t",
            int(np.round(data_new[i, 8], 0)),
            "\t",
            np.round(data_new[i, 9], 1),
            "\t",
            np.array(data_new[i, 10:13].round(decimals=3)))

    print("data_add : ")
    print("numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9, uqlab 10:13")
    for i in range(nadd):
        print(int(np.round(data_add[i, 0], 0)),
            "\t",
            np.array(data_add[i, 1:3].round(decimals=0), dtype=int),
            data_add[i, 3:5],
            np.array(data_add[i, 5:8].round(decimals=0), dtype=int),
            "\t",
            int(np.round(data_add[i, 8], 0)),
            "\t",
            np.round(data_add[i, 9], 1),
            "\t",
            np.array(data_add[i, 10:13].round(decimals=3)))

    # print data_new

    if nadd == 0:
        tt = data_new
    else:
        tt = np.row_stack((data_new, data_add))
    data_new = MG.sort_list_decreasing_column(tt, 8)

    print("total number of spots after adding harmonics : ", np.shape(data_new)[0])

    header = "# spotlistref sample after adding harmonics \n"
    header += "# orientation matrix from : %s \n" % (filefit_sample_min_pixdev)
    header += "# spot indexation from : %s \n" % (filefit_sample_max_npeaks)
    header += "# spot list from : %s \n" % (filedat_sample)
    header += "numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9, uqlab 10:13 \n"

    outfilename = filedat_sample.split(".dat")[0] + "_spotlistref_with_harmonics.dat"
    print("spotlistref saved in :", outfilename)
    outputfile = open(outfilename, "w")
    outputfile.write(header)
    np.savetxt(outputfile, data_new, fmt="%.3f")
    outputfile.close()

    return outfilename


def fitprof(xdat, ydat, p0, titre, xtitre=None, ytitre=None, plotlin=0, plotlog=0, printp1=0):

    xexp = np.array(xdat, dtype=float)
    yexp = np.array(ydat, dtype=float)

    # gaussian + line
    # same parameters as in newplot for background and gaussian

    fitfunc = lambda pp, x: pp[0] * np.exp(-(x - pp[1]) ** 2 / (2 * (pp[2] / 2.35) ** 2)) + pp[3] + x * pp[4]

    # Distance to the target function
    # errfuncstat = lambda pp, x, y: ((fitfunc(pp,x) -y)/sqrt(abs(y)+equal(abs(y),0)))**2
    _ = lambda pp, x, y: ((fitfunc(pp, x) - y) / np.sqrt(abs(y) + np.equal(abs(y), 0))) ** 2
    errfunc = lambda pp, x, y: ((fitfunc(pp, x) - y)) ** 2

    p1, _ = so.leastsq(errfunc, p0, args=(xexp, yexp), xtol=0.00001)

    residuals = (errfunc(p1, xexp, yexp)).sum() / (yexp ** 2).sum()

    if printp1:
        print("height, position, width, constant, slope")
        print("p1 = ", p1)
        print("residuals =", residuals)

    if plotlin:
        p.figure()
        p.plot(xexp, yexp, "-ro")
        p.plot(xexp, fitfunc(p1, xexp), "-")
        p.text(xexp[0] + 5, max(yexp), titre)
        if xtitre is not None:
            p.xlabel(xtitre)
        if ytitre is not None:
            p.ylabel(ytitre)
        # p.axvline(x=952.89)

    if plotlog:
        Yexp = np.log10(np.clip(yexp, 1.0, 1e6))
        Ytheor = np.log10(np.clip(fitfunc(p1, xexp), 1.0, 1e6))

        p.figure()
        p.plot(xexp, Yexp, "-ro")
        p.plot(xexp, Ytheor, "-")
        p.text(xexp[0] + 5, max(Yexp), titre)
        if xtitre is not None:
            p.xlabel(xtitre)
        if ytitre is not None:
            p.ylabel("log(" + ytitre + ")")
        # p.axvline(x=952.89)

    return (p1, residuals)


def plot_spot_displacement_map(filexyz,
                                img_list,
                                xyfit_list,
                                imgref,
                                datlist,
                                dat_min_forplot=None,
                                dat_max_forplot=None,
                                remove_ticklabels_titles=0,
                                titre="",
                                color_grid="r",
                                zoom="yes",
                                xylim=None,
                                color_above_max=np.array([1.0, 0.0, 0.0]),
                                color_below_min=np.array([1.0, 1.0, 1.0]),
                                color_for_missing_data=np.array([1.0, 0.8, 0.8]),
                                arrow_scale=20.0):

    map_imgnum, dxystep, _, _ = MG.calc_map_imgnum(filexyz)

    tt = np.loadtxt(filexyz, skiprows=1)
    xy = tt[:, 1:3]
    imgxy = np.array(tt[:, 0].round(decimals=0), dtype=int)

    nlines = np.shape(map_imgnum)[0]
    ncol = np.shape(map_imgnum)[1]
    plotdat = color_for_missing_data * np.ones((nlines, ncol, 3), dtype=float)

    if dat_min_forplot is None:
        list_plot_min = min(datlist)
    else:
        list_plot_min = dat_min_forplot
    if dat_max_forplot is None:
        list_plot_max = max(datlist)
    else:
        list_plot_max = dat_max_forplot
    print("min, max for plot = ", list_plot_min, list_plot_max)
    print("mean, min, max = ", np.round(np.mean(datlist), 3), np.round(min(datlist), 3), np.round(max(datlist), 3))

    numimg = len(img_list)

    if zoom == "yes":
        listxj = []
        listyi = []

    _ = abs(dxystep)

    ind1 = np.where(img_list == imgref)
    xypic_ref = xyfit_list[ind1[0]]
    dxyfit_list = xyfit_list - xypic_ref

    print("dxyfit_list :")
    #        print dxyfit_list.round(decimals=1)
    print("mean :", dxyfit_list.np.mean(axis=0).round(decimals=1))
    print("std :", dxyfit_list.std(axis=0).round(decimals=1))
    print("min :", dxyfit_list.min(axis=0).round(decimals=1))
    print("max :", dxyfit_list.max(axis=0).round(decimals=1))

    dxynorme_list = np.zeros(numimg, float)
    for i in range(numimg):
        dxynorme_list[i] = norme(dxyfit_list[i, :])

    print("dxynorme_list :")
    #        print dxyfit_list.round(decimals=1)
    print("mean, std, max :", np.round(np.mean(dxynorme_list), 1),
                        np.round(np.std(dxynorme_list), 1),
                        np.round(max(dxynorme_list), 1))

    ind1 = np.argmax(dxynorme_list)
    print("img at max :", img_list[ind1])

    xysample_list = np.zeros((numimg, 2), float)
    for i in range(numimg):
        ind1 = np.where(imgxy == img_list[i])
        xysample_list[i, :] = xy[ind1[0]]

    ind1 = np.where(imgxy == imgref)
    xysample_ref = xy[ind1[0][0], :]

    for i in range(numimg):
        ind2 = np.where(map_imgnum == img_list[i])
        iref, jref = ind2[0][0], ind2[1][0]
        if zoom == "yes":
            ind1 = np.where(imgxy == img_list[i])
            listxj.append(xy[ind1[0], 0])
            listyi.append(xy[ind1[0], 1])

        if datlist[i] > list_plot_max:
            plotdat[iref, jref, 0:3] = color_above_max
        elif datlist[i] < list_plot_min:
            plotdat[iref, jref, 0:3] = color_below_min
        else:
            for j in range(3):
                plotdat[iref, jref, j] = (list_plot_max - datlist[i]) / (list_plot_max - list_plot_min)

    xrange1 = np.array([0.0, ncol * dxystep[0]])
    yrange1 = np.array([0.0, nlines * dxystep[1]])
    xmin, xmax = min(xrange1), max(xrange1)
    ymin, ymax = min(yrange1), max(yrange1)
    extent = xmin, xmax, ymin, ymax
    print(extent)
    if zoom == "yes":
        listxj = np.array(listxj, dtype=float)
        listyi = np.array(listyi, dtype=float)
        minxj = listxj.min()
        maxxj = listxj.max()
        minyi = listyi.min()
        maxyi = listyi.max()
        print("zoom : minxj, maxxj, minyi, maxyi : ", minxj, maxxj, minyi, maxyi)

    _ = p.figure(1, figsize=(15, 10))
    ax = p.subplot(111)
    #            print p.setp(fig1)
    #            print p.getp(fig1)
    _ = p.imshow(plotdat[:, :, :], interpolation="nearest", extent=extent)
    #            print p.setp(imrgb)

    p.rcParams["lines.markersize"] = 2
    p.rcParams["savefig.bbox"] = None
    p.plot(xysample_list[:, 0], xysample_list[:, 1], "bo")
    p.quiver(xysample_list[:, 0],
                xysample_list[:, 1],
                dxyfit_list[:, 0] * arrow_scale,
                -dxyfit_list[:, 1] * arrow_scale,
                units="dots",
                angles="uv",
                scale_units=None,
                scale=1,
                width=2,
                color="b")

    print(xysample_ref)
    p.text(xysample_ref[0], xysample_ref[1], "x", va="center", ha="center")

    if remove_ticklabels_titles == 0:
        p.title(titre)
    ax.grid(color=color_grid, linestyle="-", linewidth=2)

    if MG.PAR.cr_string == "\n":
        ax.locator_params("x", tight=True, nbins=5)
        ax.locator_params("y", tight=True, nbins=5)
    if remove_ticklabels_titles == 0:
        p.xlabel("dxech (microns)")
        p.ylabel("dyech (microns)")
    if zoom == "yes":
        p.xlim(minxj, maxxj)
        p.ylim(minyi, maxyi)
    elif xylim is not None:
        p.xlim(xylim[0], xylim[1])
        p.ylim(xylim[2], xylim[3])

    return 0


def plot_map_simple(filexyz,
                    img_list,
                    datlist,
                    dat_min_forplot=None,
                    dat_max_forplot=None,
                    color_above_max=np.array([1.0, 0.0, 0.0]),
                    color_below_min=np.array([1.0, 1.0, 1.0]),
                    remove_ticklabels_titles=0,
                    titre="",
                    color_grid="r",
                    zoom="yes",
                    xylim=None,
                    color_for_missing_data=np.array([1.0, 0.8, 0.8])):

    map_imgnum, dxystep, _, _ = MG.calc_map_imgnum(filexyz)

    tt = np.loadtxt(filexyz, skiprows=1)
    xy = tt[:, 1:3]
    imgxy = np.array(tt[:, 0].round(decimals=0), dtype=int)

    nlines = np.shape(map_imgnum)[0]
    ncol = np.shape(map_imgnum)[1]
    plotdat = color_for_missing_data * np.ones((nlines, ncol, 3), dtype=float)

    if dat_min_forplot is None:
        list_plot_min = min(datlist)
    else:
        list_plot_min = dat_min_forplot
    if dat_max_forplot is None:
        list_plot_max = max(datlist)
    else:
        list_plot_max = dat_max_forplot
    print("min, max for plot = ", list_plot_min, list_plot_max)
    print("mean, min, max = ", np.round(np.mean(datlist), 3), np.round(min(datlist), 3), np.round(max(datlist), 3))

    numimg = len(img_list)

    if zoom == "yes":
        listxj = []
        listyi = []

    _ = abs(dxystep)

    for i in range(numimg):
        ind2 = np.where(map_imgnum == img_list[i])
        iref, jref = ind2[0][0], ind2[1][0]
        if zoom == "yes":
            ind1 = np.where(imgxy == img_list[i])
            listxj.append(xy[ind1[0], 0])
            listyi.append(xy[ind1[0], 1])

        if datlist[i] > list_plot_max:
            plotdat[iref, jref, 0:3] = color_above_max
        elif datlist[i] < list_plot_min:
            plotdat[iref, jref, 0:3] = color_below_min
        else:
            for j in range(3):
                plotdat[iref, jref, j] = (list_plot_max - datlist[i]) / (list_plot_max - list_plot_min)

    xrange1 = np.array([0.0, ncol * dxystep[0]])
    yrange1 = np.array([0.0, nlines * dxystep[1]])
    xmin, xmax = min(xrange1), max(xrange1)
    ymin, ymax = min(yrange1), max(yrange1)
    extent = xmin, xmax, ymin, ymax
    print(extent)
    if zoom == "yes":
        listxj = np.array(listxj, dtype=float)
        listyi = np.array(listyi, dtype=float)
        minxj = listxj.min()
        maxxj = listxj.max()
        minyi = listyi.min()
        maxyi = listyi.max()
        print("zoom : minxj, maxxj, minyi, maxyi : ", minxj, maxxj, minyi, maxyi)

    _ = p.figure(1, figsize=(15, 10))
    ax = p.subplot(111)

    _ = p.imshow(plotdat[:, :, :], interpolation="nearest", extent=extent)
    #            print p.setp(imrgb)

    if remove_ticklabels_titles == 0:
        p.title(titre)
    ax.grid(color=color_grid, linestyle="-", linewidth=2)

    if MG.PAR.cr_string == "\n":
        ax.locator_params("x", tight=True, nbins=5)
        ax.locator_params("y", tight=True, nbins=5)
    if remove_ticklabels_titles == 0:
        p.xlabel("dxech (microns)")
        p.ylabel("dyech (microns)")
    if zoom == "yes":
        p.xlim(minxj, maxxj)
        p.ylim(minyi, maxyi)
    elif xylim is not None:
        p.xlim(xylim[0], xylim[1])
        p.ylim(xylim[2], xylim[3])

    return 0


def plot_spot_traj(img_list,
                    Intensity_list,
                    threshold_factor_for_traj,
                    xyfit_list,
                    xboxsize,
                    yboxsize,
                    xpic,
                    ypic,
                    titre,
                    overlay="no"):

    p.rcParams["figure.subplot.right"] = 0.85
    p.rcParams["figure.subplot.left"] = 0.15
    p.rcParams["font.size"] = 20

    print("plot peak trajectory for images with high peak intensity")
    index_high_int2 = np.where(
        (Intensity_list > threshold_factor_for_traj * max(Intensity_list))
        & (abs(xyfit_list[:, 0] - float(xpic)) < xboxsize)
        & (abs(xyfit_list[:, 1] - float(ypic)) < yboxsize))
    #        print "index_high_int2 =", index_high_int2
    xyfit_list_high_int2 = xyfit_list[index_high_int2]
    #        xymax_list_high_int2 = xymax_list[index_high_int2]
    Intensity_list_high_int2 = Intensity_list[index_high_int2]
    #        print "xyfit_list_high_int2 = \n",xyfit_list_high_int2.round(decimals=2)
    #        print "Intensity_list_high_int2 = ", Intensity_list_high_int2
    xymoy2 = np.zeros(2, float)

    for i in range(2):
        xymoy2[i] = np.average(xyfit_list_high_int2[:, i], weights=Intensity_list_high_int2)
    print("average on points with Intensity_list > threshold_factor_for_traj*max(Intensity_list)")
    print("number of points : selected / total", len(index_high_int2[0]), len(Intensity_list))
    print("xymoy2 =", xymoy2.round(decimals=2))
    print("threshold_factor_for_traj =", threshold_factor_for_traj)

    if titre is not None:
        titre2 = titre + "\n" + "x-y peak position for high intensity points only"

    if overlay == "no":
        p.figure()
        color1 = "bo-"
    elif overlay == "yes":
        color1 = "rs-"
    # p.plot(xymax_list_high_int[:,0],xymax_list_high_int[:,1],'ro-',label = 'xy max')
    p.plot(xyfit_list_high_int2[:, 0], -xyfit_list_high_int2[:, 1], color1)
    # p.text(min(xyfit_list_high_int2[:,0]),min(xyfit_list_high_int2[:,1]),titre2)
    p.xlabel("xpix")
    p.ylabel("ypix")
    for i in index_high_int2[0]:
        x = xyfit_list[i, 0]
        y = -xyfit_list[i, 1]
        p.text(x, y, str(img_list[i]), fontsize=16)

    if overlay == "no":
        if titre is not None:
            p.title(titre2)
    # p.axvline(x=xymoy2[0])
    # p.axhline(y=xymoy2[1])

    return 0


def build_mosaic_and_fit_spot_position(indimg,
                                        imfile_path,
                                        imfile_prefix,
                                        xpic,
                                        ypic,
                                        xboxsize,
                                        yboxsize,
                                        mosaic_fast_size,
                                        mosaic_slow_size,
                                        filepathout,
                                        bad_img=np.array([]),
                                        xypic_from_LaueTools_Peak_Search_datfile=1,
                                        xypic_from_Imagej=0,
                                        imfile_suffix=".mccd",
                                        build_mosaic=1,
                                        mosaic_order="1234",  # m00 m01 m10 m11
                                        ycam_orientation_in_mosaic="vertical down",
                                        plot_spot_trajectory=1,
                                        threshold_factor_for_traj=0.2,
                                        calc_sum_image=0,
                                        subtract_background=0,
                                        fit_xyprofiles_of_sumimage=0,
                                        plot_curve_Ibox=1,  # pour les scans lineaires
                                        plot_curve_Imax=1,  # pour les scans lineaires
                                        plot_map_Ibox=0,
                                        filexyz=None,  # pour plot_map_Ibox = 1
                                        calib=np.array([69.0, 1024.0, 1024.0, 0.0, 0.0]),
                                        CCDlabel=MG.PAR.CCDlabel,
                                        return_mosaic_filename=0,
                                        mon_in_scan=None):
    """
    calib : to convert xyfit of spot into 2sintheta
    """

    pixelsize = DictLT.dict_CCD[CCDlabel][1]

    # adapte de mono4c, 17Dec13

    #    mosaicsize = ((2*xboxsize+1)*mosaic_fast_size, (2*yboxsize+1)*mosaic_slow_size)

    mosaicsize = ((2 * yboxsize + 1) * mosaic_slow_size, (2 * xboxsize + 1) * mosaic_fast_size)

    #    mosaic = Image.new("I;16",mosaicsize)

    #    mosaic2 = ImageChops.invert(mosaic)

    mosaic = np.zeros(mosaicsize, dtype="uint16")

    numim = len(indimg) - len(bad_img)

    print("numim =", numim)

    Ibox_list = np.zeros(numim, float)
    Imax_list = np.zeros(numim, float)
    img_list = np.arange(numim)
    xyfit_list = np.zeros((numim, 2), float)
    xymax_list = np.zeros((numim, 2), int)
    height_list = np.zeros(numim, float)
    xyheight_list = np.zeros((numim, 2), float)
    xywidth_list = np.zeros((numim, 2), float)
    twosintheta_list = np.zeros(numim, float)

    uilab = np.array([0.0, 1.0, 0.0])

    sumimage = np.zeros((2 * xboxsize, 2 * yboxsize), int)

    Ipixmax = 0

    kx = 0
    ky = 0
    kim = 0

    _ = 2

    if xypic_from_LaueTools_Peak_Search_datfile:
        xshift = -1
        yshift = -1
    elif xypic_from_Imagej:
        xshift = 0
        yshift = 0

    print("xpic, ypic, xboxsize, yboxsize : ", xpic, ypic, xboxsize, yboxsize)
    x1 = xpic - xboxsize + xshift
    x2 = xpic + xboxsize + xshift
    y1 = ypic - yboxsize + yshift
    y2 = ypic + yboxsize + yshift

    print("box x1 y1 x2 y2 : ", x1, y1, x2, y2)
    _ = (x1, y1, x2, y2)

    # verifier si je fais le -1 sur xpic ypic ou non

    if build_mosaic:
        # np.rot90(m,k=1)  Rotate an array by 90 degrees in the counter-clockwise direction.

        # key = mosaic_order  m00 m01 m10 m11
        # box_transform_list
        # mosaic_transform_list
        dict_fliprot = {"1234": [None, None],
                        "1324": [["fliplr", "rot90left"], ["rot90right", "fliplr"]],
                        "3412": [["flipud"], ["flipud"]],
                        "2413": [["rot90right"], ["rot90left"]],
                        "2143": [["fliplr"], ["fliplr"]],
                        "3142": [["rot90left"], ["rot90right"]],
                        "4321": [["rot180"], ["rot180"]],
                        "4231": [["fliplr", "rot90right"], ["rot90left", "fliplr"]]}

    for kk in indimg:

        print("\nkk = ", kk)

        if kk in bad_img:
            print("image belongs to list of bad images")
            continue

        img_list[kim] = kk

        fileim = (imfile_path
                    + imfile_prefix
                    + MG.imgnum_to_str(kk, MG.PAR.number_of_digits_in_image_name)
                    + imfile_suffix)

        print("image file : ", fileim)

        dataimage, _, _ = IOimage.readCCDimage(fileim, CCDLabel=CCDlabel, dirname=None)

        #        print np.shape(dataimage)

        databox = dataimage[y1:y2, x1:x2]

        #        ind1 = np.where(databox > 40000)
        #        databox[ind1[0]] = 0
        #
        #        print np.shape(databox)

        #        f = open(fileim,'r')
        #        im = Image.open(fileim)
        #        #print "original image :", im.format, im.size, im.mode
        #
        #        region = im.crop(box)
        ##         region2 = ImageChops.invert(region)
        ##         region.show()
        ##         la convention pour l'orientation de y est la meme que dans XMAS
        #        print "cropped image : ", region.format, region.size, region.mode
        #        print "cropped image pixel values"
        #
        #
        #        data1= list(region.getdata())
        #
        #        data5 = np.array(region.getdata(), dtype = "uint16")
        #
        #        #data2 = str(data1).replace(',','')
        #        #data3 = data2.strip('[]')
        #        #data4 = data3.split(' ')
        #        # ligne
        #        #data5 = np.array(data4)
        #
        #        #tableau
        #        databox = data5.renp.shape(2*yboxsize,2*xboxsize)
        #
        # print "databox \n", databox
        # data7 = np.array(databox, dtype=int)
        # print data7

        if calc_sum_image:

            print("summing images")
            # print "image = \n", data7
            sumimage = sumimage + databox
            # print "sumimage = \n", sumimage

        xprof = np.sum(databox, axis=0)
        yprof = np.sum(databox, axis=1)
        # print "sum over all lines, axis=0, xprofile \n" , xprof
        # print "sum over all columns, axis=1, yprofile \n" , yprof

        xx0 = list(range(x1 + 1, x2 + 1))
        xx = np.array(xx0, dtype=float)
        yy0 = list(range(y1 + 1, y2 + 1))
        yy = np.array(yy0, dtype=float)

        guess_xpic = xx[np.argmax(xprof)]
        guess_height = float(max(xprof) - min(xprof))
        guess_constant = float(min(xprof))

        px0 = [guess_height, guess_xpic, 2.0, guess_constant, 0.0]

        #        print "px0 =", px0

        px, _ = fitprof(
            xx, xprof, px0, "x-profile " + str(kk), "xpix", "intensity", plotlin=0, plotlog=0, printp1=0)

        guess_ypic = yy[np.argmax(yprof)]
        guess_height = float(max(yprof) - min(yprof))
        guess_constant = float(min(yprof))

        py0 = [guess_height, guess_ypic, 2.0, guess_constant, 0.0]
        # print "py0 =", py0

        py, _ = fitprof(
            yy, yprof, py0, "y-profile " + str(kk), "ypix", "intensity", plotlin=0, plotlog=0, printp1=0)

        xyfit_list[kim, :] = np.array([px[1], py[1]])

        #        xymax_list[kim,:] = np.array([xx[np.argmax(xprof)],yy[np.argmax(yprof)]])

        height_list[kim] = np.sqrt(px[0] * py[0])

        xyheight_list[kim] = np.array([px[0], py[0]])

        xywidth_list[kim] = np.array([abs(px[2]), abs(py[2])])

        uqlab = MG.xycam_to_uqlab(xyfit_list[kim, :], calib, pixelsize=pixelsize)

        sintheta = -1. * np.inner(uqlab, uilab)

        twosintheta_list[kim] = 2.0 * sintheta

        #        Imax = max(np.array(data5, dtype=int))
        size1 = np.shape(databox)[0] * np.shape(databox)[1]
        data5 = np.reshape(databox, size1)
        Imax = max(data5)
        print("Imax : ", Imax)

        xymax = np.where(databox == Imax)

        #        print "i,j of max in box : ",  xymax[0][0], xymax[1][0]
        #        print "xymax :", xx[xymax[1][0]], yy[xymax[0][0]]

        xymax_list[kim, 0] = xx[xymax[1][0]]
        xymax_list[kim, 1] = yy[xymax[0][0]]

        # 11Mar14 : je prends le xy du max de la boite et pas le max de chaque profil

        print("img, xymax, xyfit = ", kk, xymax_list[kim, :], xyfit_list[kim, :].round(decimals=2))

        Imax_list[kim] = Imax

        if Imax > Ipixmax:
            Ipixmax = Imax * 1

        Itot = (np.array(data5, dtype=int)).sum(axis=0)
        Ibox_list[kim] = Itot

        if subtract_background:

            # print "first line "
            firstline = np.array(databox[0, :], dtype=int)
            # print firstline
            sum1 = firstline.sum()
            # print sum1
            # print "last line "
            lastline = np.array(databox[2 * yboxsize - 1, :], dtype=int)
            # print lastline
            sum2 = lastline.sum()
            # print sum2
            # print "first colon without first and last line"
            firstcoltrunc = np.array(databox[1 : 2 * yboxsize - 1, 0], dtype=int)
            # print firstcoltrunc
            sum3 = firstcoltrunc.sum()
            # print sum3
            # print "last colon without first and last line"
            lastcoltrunc = np.array(databox[1 : 2 * yboxsize - 1, 2 * xboxsize - 1], dtype=int)
            # print lastcoltrunc
            sum4 = lastcoltrunc.sum()
            # print sum4

            sumframe = sum1 + sum2 + sum3 + sum4
            npixframe = len(firstline) + len(lastline) + len(firstcoltrunc) + len(lastcoltrunc)

            print("intensity sum over frame pixels : ", sumframe)
            print("nb of frame pixels : ", npixframe)

            pixbackgnd = float(sumframe) / float(npixframe)
            print("background per pixel : ", pixbackgnd)
            totbackgnd = 2 * xboxsize * 2 * yboxsize * pixbackgnd
            print("background to subtract to integrated intensity : ", totbackgnd)

            Ibox = float(Itot) - totbackgnd
            print("frame-background subtracted integrated intensity : ", Ibox)

            Ibox_list[kim] = Ibox

        if build_mosaic:

            print("pasting image in mosaic")

            box_transform_list = dict_fliprot[mosaic_order][0]
            # print "box_transform_list = ", box_transform_list

            # TODO : rajouter les transfo sur la mosaique apres remplissage

            if box_transform_list is not None:
                print("mosaic_order = ", mosaic_order)
                print("box_transform_list = ", box_transform_list)
                for transform_op in box_transform_list:
                    if transform_op == "fliplr":
                        databox = np.fliplr(databox)
                    if transform_op == "flipud":
                        databox = np.flipud(databox)
                    if transform_op == "rot90left":
                        databox = np.rot90(databox, k=1)
                    if transform_op == "rot90right":
                        databox = np.rot90(databox, k=3)
                    if transform_op == "rot180":
                        databox = np.rot90(databox, k=2)

            mosaic[ky * 2 * yboxsize : (ky + 1) * 2 * yboxsize, kx * 2 * xboxsize : (kx + 1) * 2 * xboxsize] = databox

        #            mosaic2.paste(region,(kx*2*xboxsize,ky*2*yboxsize,(kx+1)*2*xboxsize,(ky+1)*2*yboxsize))
        # if kx == 0 :

        # elif kx == 1 :
        # dxy = listxyech[kim,:]-listxyech[kim-1,:]

        # mosaic2.paste(region,(kcol*2*xboxsize,kline*2*yboxsize,(kcol+1)*2*xboxsize,(kline+1)*2*yboxsize))

        lastkx = kx
        lastky = ky

        # faire l'inversion a la fin apres ajout des differentes cases dans la mosaique

        kx = kx + 1
        if kx > mosaic_fast_size - 1:
            kx = 0
            ky = ky + 1

        kim = kim + 1

    titre = (imfile_prefix
                + " img "
                + str(min(img_list))
                + " to "
                + str(max(img_list))
                + " xpic ypic "
                + str(xpic)
                + " "
                + str(ypic))

    print("image list \n", img_list)
    if len(bad_img) > 0:
        print("bad_img \n", bad_img)
    # print "Ibox_list \n", Ibox_list
    # print "xyfit_list \n",xyfit_list
    # print "xywidth_list \n",xywidth_list
    # print "xymax_list \n",xymax_list
    print(titre)
    print("max pixel intensity in box in all images = ", Ipixmax)
    print("xpic, ypic, xboxsize, yboxsize : ", xpic, ypic, xboxsize, yboxsize)
    # print "np.argmax(Ibox_list) = ", np.argmax(Ibox_list)
    print("integrated intensity Ibox :")
    if subtract_background == 1:
        print("subtract background from box frame")
    print("max box-integrated intensity in all images = ", max(Ibox_list))
    print("image at max = ", img_list[np.argmax(Ibox_list)])
    print("spot position at max = ", xyfit_list[np.argmax(Ibox_list)].round(decimals=2))

    outfilename = (filepathout + imfile_prefix + str(min(img_list)) + "to" + str(max(img_list))
                    + "_" + str(xpic) + "_" + str(ypic) + "_box" + str(xboxsize)
                    + "x" + str(yboxsize) + "y" + ".dat")

    print("filename for output results", outfilename)

    tt = np.column_stack(
        (img_list, Ibox_list, xyfit_list, xywidth_list, Imax_list, xymax_list, twosintheta_list * 1000.0))
    # print tt
    header = "img 0, Ibox 1, xyfit 2:4, widthxy 4:6 Imax 6, xymax 7:9 twosintheta_x1000 9 "
    header2 = "img Ibox xfit yfit xwidth ywidth Imax xmax ymax twosintheta_x1000 "
    if mon_in_scan is not None:
        tt = np.column_stack((tt, mon_in_scan))
        header += "Monitor 10 "
        header2 += "Monitor "
    header += "\n"
    header2 += "\n"

    outputfile = open(outfilename, "w")
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, tt, fmt="%6.2f")
    outputfile.close()

    if fit_xyprofiles_of_sumimage:

        print("fitting x and y projected profiles of sum image")

        xprof = np.sum(sumimage, axis=0)
        yprof = np.sum(sumimage, axis=1)

        guess_xpic = xx[np.argmax(xprof)]
        guess_height = float(max(xprof) - min(xprof))
        guess_constant = float(min(xprof))

        px0 = [guess_height, guess_xpic, 2.0, guess_constant, 0.0]

        # print "px0 =", px0
        px, _ = fitprof(
            xx, xprof, px0, "x-profile sumimage", "xpix", "intensity", plotlin=1, plotlog=1, printp1=1)

        guess_ypic = yy[np.argmax(yprof)]
        guess_height = float(max(yprof) - min(yprof))
        guess_constant = float(min(yprof))

        py0 = [guess_height, guess_ypic, 2.0, guess_constant, 0.0]
        # print "py0 =", py0

        py, _ = fitprof(
            yy, yprof, py0, "y-profile sumimage", "ypix", "intensity", plotlin=1, plotlog=1, printp1=1)

    if plot_map_Ibox:
        titre1 = titre + " integrated intensity"
        plot_map_simple(filexyz,
                        img_list,
                        Ibox_list,
                        dat_min_forplot=None,
                        dat_max_forplot=None,
                        color_above_max=np.array([1.0, 0.0, 0.0]),
                        color_below_min=np.array([1.0, 1.0, 1.0]),
                        remove_ticklabels_titles=0,
                        titre="Ibox")
    if plot_curve_Ibox:
        titre1 = titre + " integrated intensity"
        p.figure()
        p.plot(img_list, Ibox_list, "ro-")
        p.xlabel("img")
        p.ylabel("Ibox")
        p.title(titre1)

    if plot_curve_Imax:
        titre1 = titre + " max intensity"
        p.figure()
        p.plot(img_list, Imax_list, "ro-")
        p.xlabel("img")
        p.ylabel("Imax")
        p.title(titre1)
        if 0:  # map1 Mar13
            for k in img_list:
                if not (k % 17):
                    #                    print k
                    p.axvline(x=k, ymax=0.5)

    if plot_spot_trajectory:

        plot_spot_traj(img_list, Imax_list, threshold_factor_for_traj,
                    xyfit_list, xboxsize, yboxsize, xpic, ypic, titre)

    print("warning : plot_spot_traj : intensity thresholding now done on Imax_list")

    if build_mosaic:

        print("save mosaic image to file")

        mosaic2 = mosaic[: (lastky + 1) * 2 * yboxsize, : (mosaic_fast_size) * 2 * xboxsize]

        mosaic_transform_list = dict_fliprot[mosaic_order][1]

        #            TODO : rajouter les transfo sur la mosaique apres remplissage

        if mosaic_transform_list is not None:
            print("mosaic_order = ", mosaic_order)
            print("mosaic_transform_list = ", mosaic_transform_list)
            for transform_op in mosaic_transform_list:
                if transform_op == "fliplr":
                    mosaic2 = np.fliplr(mosaic2)
                if transform_op == "flipud":
                    mosaic2 = np.flipud(mosaic2)
                if transform_op == "rot90left":
                    mosaic2 = np.rot90(mosaic2, k=1)
                if transform_op == "rot90right":
                    mosaic2 = np.rot90(mosaic2, k=3)
                if transform_op == "rot180":
                    mosaic2 = np.rot90(mosaic2, k=2)

        p.figure()
        print("plotting mosaic")
        #        print np.shape(mosaic2)

        p.imshow(mosaic2, vmin=0.0, vmax=100.0, cmap=GT.ORRD)

        titre1 = titre + " mosaic, box" + str(xboxsize) + "," + str(yboxsize)

        p.title(titre1)

        #        box2 = (0,0,(mosaic_fast_size)*2*xboxsize,(lastky+1)*2*yboxsize)
        #
        #        out = mosaic2.crop(box2)
        #
        #        imagefilename = filepathout + imfile_prefix + str(min(img_list))+'to'+str(max(img_list))+ '_' + str(xpic) + '_' + str(ypic)+ '.TIFF'
        imagefilename = (filepathout
                        + imfile_prefix
                        + str(min(img_list))
                        + "to"
                        + str(max(img_list))
                        + "_"
                        + str(xpic)
                        + "_"
                        + str(ypic)
                        + "_box"
                        + str(xboxsize)
                        + "x"
                        + str(yboxsize)
                        + "y"
                        + "_order"
                        + mosaic_order
                        + ".TIFF")

        print("filename for output mosaic image", imagefilename)

        TIFF.imsave(imagefilename, mosaic2)

    if return_mosaic_filename:
        return (outfilename, imagefilename)
    else:
        return outfilename


# --- --- build_Ipix_vs_img_table  -----
def build_Ipix_vs_img_table(indimg,
                            imfile_path,
                            imfile_prefix,
                            imfile_suffix,
                            filepathout,
                            user_comment="",
                            add_str_in_outfilename=None,
                            filespotlist_sample=None,
                            filedat=None,
                            xypic=None,
                            xylist=None,
                            ccdlabel=None,
                            nbdigits=4,
                            xypic_from_LaueTools_Peak_Search_datfile=1,
                            xypic_from_Imagej=0):
    """ read pixel intensity values of several images
    indimg:  list of image indices
    imfile_path: path to image file
    imfile_prefix: prefix of image filename (filename without index and file extension)
    filespotlist_sample: peaks list file
    return:

    DipsData, xy_LT, outfilename, filenamepckl
    """

    # utilise xypic sortis du peak search LaueTools,
    # decales de 1 pixel en x y par rapport au xypic affiches par Imagej

    if filespotlist_sample is not None:
        data_dat = np.loadtxt(filespotlist_sample, skiprows=5)
        #    print data_dat
        xy_LT = np.array(data_dat[:, 1:3].round(decimals=1), dtype="int")
        #    print xy_LT
        npics = np.shape(data_dat)[0]
    #        xypic_from_LaueTools_Peak_Search_datfile = 1
    #        xypic_from_Imagej = 0

    elif filedat is not None:
        data_dat = np.loadtxt(filedat, skiprows=1)
        # data_xy_LT = data_dat[:, :2]
        # data_xy_LT_integer = data_xy_LT - data_dat[:, 7:9]
        # xy_LT = np.array(data_xy_LT_integer.round(decimals=1), dtype="int")
        data_xy_LT_integer = data_dat[:, :2]
        xy_LT = np.array(data_xy_LT_integer.round(decimals=1), dtype="int")
        npics = np.shape(data_dat)[0]
    #        xypic_from_LaueTools_Peak_Search_datfile = 1
    #        xypic_from_Imagej = 0

    elif xypic is not None:
        if xypic_from_LaueTools_Peak_Search_datfile:
            xy_LT = np.array(xypic, dtype=int)
        elif xypic_from_Imagej:
            xy_LT = np.array(xypic, dtype=int) + np.ones(2, int)
        xy_LT = xy_LT.renp.shape(1, 2)
        npics = 1

    elif xylist is not None:
        if xypic_from_LaueTools_Peak_Search_datfile:
            xy_LT = np.array(xylist, dtype=int)
        elif xypic_from_Imagej:
            xy_LT = np.array(xylist, dtype=int) + np.ones(2, int)
        npics = np.shape(xy_LT)[0]

    # add fliprot for rotated cameras

    fliprot = DictLT.dict_CCD[ccdlabel][3]
    offset = DictLT.dict_CCD[ccdlabel][4]
    ncol = DictLT.dict_CCD[ccdlabel][0][1]
    print("offset =", offset)
    print("ncol = ", ncol)

    if fliprot == "no":
        xy_raw = xy_LT * 1
    elif fliprot == "VHR_Feb13":
        # flip left right
        # le flip sur fait sur x-1, y-1, puis je rajoute +1 +1 au resultat
        # les valeurs vont de 0 a ncol-1
        xy_raw = np.array([ncol - 1 - (xy_LT[:, 0] - 1) + 1, xy_LT[:, 1]]).T
    else:
        print("camera not yet implemented : ")
        print("this can be corrected by editing function build_Ipix_vs_img_table in diamond.py")
        print("and adding the proper elif fliprot == ... :")

    print("xy_raw = ", xy_raw)

    print(npics)

    numim = len(indimg)

    listI = np.zeros((numim, npics), int)

    fileim = ""

    kim = 0

    framedim = DictLT.dict_CCD[ccdlabel][0]
    datasizebytes = np.prod(framedim) * 2

    for kk in indimg:

        fileim = os.path.join(imfile_path, imfile_prefix + MG.imgnum_to_str(kk, nbdigits) + imfile_suffix)
        if kim == 0:
            firstimfile = fileim

        print("image file : \n", fileim)
        #        print "img peak xypix intensity"

        # from readmccd Loic

        if ccdlabel.startswith("sCMOS"):
            filesize = os.path.getsize(fileim)
            offset = filesize - datasizebytes

        if fileim.endswith('gz'):
            with gzip.open(fileim, 'rb') as f:

                for k in range(npics):
                    xpic = xy_raw[k, 0] - 1
                    ypic = xy_raw[k, 1] - 1

                    f.seek(offset + 2 * (ypic * ncol + xpic))
                    val = struct.unpack("H", f.read(2))
                    listI[kim, k] = val[0]
        else:
            f = open(fileim, "rb")
            for k in range(npics):
                xpic = xy_raw[k, 0] - 1
                ypic = xy_raw[k, 1] - 1

                f.seek(offset + 2 * (ypic * ncol + xpic))
                val = struct.unpack("H", f.read(2))
                listI[kim, k] = val[0]
            # print kim, k, xpic, ypic, listI[kim,k]

        kk = kk + 1
        kim = kim + 1

    DipsData = np.column_stack((indimg, listI))
    header = "img"

    if add_str_in_outfilename is None:
        add_str_in_outfilename = ""
    if filepathout is not None:
        if xypic is None:
            outfilename = os.path.join(filepathout,
                                        imfile_prefix
                                        + '%d'%indimg[0]
                                        + "to"
                                        + '%d'%indimg[-1]
                                        + "_img_Ipix"
                                        + add_str_in_outfilename
                                        + ".dat")
        else:
            outfilename = os.path.join(filepathout,
                                        imfile_prefix
                                        + '%d'%indimg[0]
                                        + "to"
                                        + '%d'%indimg[-1]
                                        + "_img_Ipix"
                                        + add_str_in_outfilename
                                        + ".dat")

    print(DipsData[:10, :])
    for i in range(npics):
        header = header + " Ipix" + str(i)
    header = header + "\n"
    print(header)
    print("filename for output results : \n", outfilename)

    outputfile = open(outfilename, "w")
    outputfile.write(header)
    np.savetxt(outputfile, DipsData, fmt="%6.0f")
    tailer = "#first image file :" + firstimfile + "\n"
    outputfile.write(tailer)
    str1 = "#warning : xpic ypic from LaueTools peak search, shifted by +1 +1 from Imagej display \n"
    outputfile.write(str1)
    str1 = " ".join(str(e) for e in xy_LT[:, 0])
    outputfile.write("#xpic_list : " + str1 + "\n")
    str1 = " ".join(str(e) for e in xy_LT[:, 1])
    outputfile.write("#ypic_list : " + str1 + "\n")
    str1 = " ".join(str(e) for e in xy_raw[:, 0])
    outputfile.write("#xpic_raw_list : " + str1 + "\n")
    str1 = " ".join(str(e) for e in xy_raw[:, 1])
    outputfile.write("#ypic_raw_list : " + str1 + "\n")

    print(user_comment)

    if user_comment is not None:
        outputfile.write(user_comment)

    outputfile.close()

    # pickle
    data_pickled = (DipsData, xy_LT)
    filenamepckl = os.path.join(filepathout, 'intensityprofiles_'
                                        + imfile_prefix
                                        + '%d'%indimg[0]
                                        + "to"
                                        + '%d'%indimg[-1]
                                        + "_img_Ipix"
                                        + add_str_in_outfilename
                                        + ".pckl")
    with open(filenamepckl, "wb") as f:
        pickle.dump(data_pickled, f)

    return DipsData, xy_LT, outfilename, filenamepckl


def serial_Ipix_vs_img(imfile_path,
                        imfile_prefix,
                        imfile_suffix,
                        filepathout,
                        user_comment_list=None,
                        CCDlabel=MG.PAR.CCDlabel,
                        img_reflist=None,
                        dict_xy_E=None,
                        scan_list=None,
                        img_in_scan=None,
                        ngood_imgref=None):

    # img_ref = 1ere image du thf scan
    # 0 xyint_ref
    # 1 hkl_ref
    # 2 Etheor_ref
    # 3 xint_list
    # 4 yint_list
    # 5 Etheor_list

    if ngood_imgref is not None:
        img_reflist = img_reflist[:ngood_imgref]
    nthfscans = len(img_reflist)

    dimg = img_reflist[1] - img_reflist[0]

    file_Ipix_vs_img_list = []

    if len(scan_list) == 1:  # single mesh scan
        add_str_in_outfilename = "_scan" + str(scan_list[0])
        user_comment = user_comment_list[0][0]

    #    full_img_list = np.arange(imgref_list[0], imgref_list[-1]+ dimg)
    #    Elist = np.zeros(len(dict_xy_E.keys()))

    for i in range(nthfscans):

        if len(scan_list) == nthfscans:  # one scan per thf scan
            add_str_in_outfilename = "_scan" + str(scan_list[i])
            user_comment = user_comment_list[i][0]

        indimg = np.arange(img_reflist[i], img_reflist[i] + dimg)
        if (i == nthfscans - 1) & (ngood_imgref is None):
            indimg = np.arange(img_reflist[i], img_in_scan[-1])

        xlist = []
        ylist = []

        for _, value in dict_xy_E.items():
            xlist.append(value[3][i])
            ylist.append(value[4][i])

        xlist = np.array(xlist, dtype=int)
        ylist = np.array(ylist, dtype=int)
        xylist = np.column_stack((xlist, ylist))

        filename = build_Ipix_vs_img_table(indimg,
                                        imfile_path,
                                        imfile_prefix,
                                        imfile_suffix,
                                        filepathout,
                                        user_comment,
                                        add_str_in_outfilename=add_str_in_outfilename,
                                        filespotlist_sample=None,
                                        filedat=None,
                                        xypic=None,
                                        xylist=xylist,
                                        ccdlabel=CCDlabel)

        file_Ipix_vs_img_list.append(filename)

    print(file_Ipix_vs_img_list)
    return file_Ipix_vs_img_list


def read_xypic_in_file_Ipix_vs_img(file_Ipix_vs_img):
    """
    read X, Y pixel coordinates of pixels corresponding to intensity profiles

    """

    f = open(file_Ipix_vs_img, "r")
    i = 0
    linepos_list = np.zeros(2, int)
    #    endlinechar = '\r\n' # unix
    #    endlinechar = '\n' # dos
    _ = MG.PAR.cr_string
    try:
        for line in f:
            if line[0] == "#":
                # print line.rstrip(endlinechar)
                if line[1:10] == "xpic_list":
                    print(line)
                    linepos_list[0] = i
                #                     print "xpic_LT", linepos_list[0]
                if line[1:10] == "ypic_list":
                    print(line)
                    linepos_list[1] = i
            #                     print "ypic_LT", linepos_list[1]
            # print line.rstrip(endlinechar)
            i = i + 1
    finally:
        f.close()

    f = open(file_Ipix_vs_img, "r")
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)

    f.seek(0)
    #     print "f.readline() in read_xypic_in_file_Ipix_vs_img",f.readline()

    for j in range(2):
        n = linepos_list[j]
        #        print "n = ", n
        f.seek(line_offset[n])
        tt = f.readline()
        # tt1 = tt.rstrip(MG.PAR.cr_string)
        tt1 = tt.rstrip("\n")

        # print tt1.split(' ')[2:]
        tt2 = np.array(tt1.split(" ")[2:], dtype=int)
        if j == 0:
            allres = tt2 * 1
        else:
            allres = np.row_stack((allres, tt2))
    xypic = allres.transpose()

    #     print "xypic list in file_Ipix_vs_img :\n", xypic

    return xypic


# ---  ----plot_Ipix_vs_img  --------------
def plot_Ipix_vs_img(file_Ipix_vs_img=None,
                    file_exp_dia=None,
                    save_figures=0,
                    npics_max=None,  # limit nbIprofiles
                    mon_in_scan=None,  # normalize to monitor
                    thf_in_scan=None,  # plot vs thf instead of img
                    img_in_scan=None,
                    mon_offset=0,  # monitor offset per ctime
                    filedipstheor=None,
                    filediplinks=None,
                    figsize=(6, 8),
                    dip_depth_min=1.0,
                    xy_single_spot=None,
                    refpoint=0,  # overwritten by imgref
                    file_Ipix_vs_img_single_spot=None,
                    remove_saturated=65535,
                    imgref=None,
                    use_npoint_as_x=None,
                    pedestal=0.0,
                    scanObject=None,
                    indexedspotlistwithharmonics=None):
    """
    params:
    ------------

    npics_max   : number of intensity profiles to plot

    # WARNING: if file_dips_theor != None,
    # then will produce a file with links with dips from theory and experiments in a narrow thf range

    """

    p.rcParams["figure.subplot.right"] = 0.75

    # if filedipstheor is not None
    # cette fonction prepare un fichier de diplinks
    # a remplir ensuite d'apres les positions en img des dips experimentaux
    # en mettant le "confidence" a 1 au lieu de 0 pour ces dips

    if thf_in_scan is not None:
        thf_min = thf_in_scan[0]
        thf_max = thf_in_scan[-1]

    add_str = ""
    ncol_in_listI1 = None

    if file_Ipix_vs_img is not None:
        print("reading file_Ipix_vs_img from:")
        print(file_Ipix_vs_img)
        IntProfilesData = np.loadtxt(file_Ipix_vs_img, skiprows=1)
        inputfile = file_Ipix_vs_img
        nbIntProfiles = np.shape(IntProfilesData)[1] - 1
        xyIntProfiles = read_xypic_in_file_Ipix_vs_img(file_Ipix_vs_img)

    if filedipstheor is not None:
        print("reading filedipstheor from:")
        print(filedipstheor)
        list_diplinks_theor = []
        dict_dips_theor, dict_values_names2 = read_dict_dips_theor(filedipstheor)

        ind_thf_dip = dict_values_names2.index("thf_list_theor")
        ind_depth = dict_values_names2.index("dip_depth_theor")
        ind_ndia = dict_values_names2.index("ndia_list")
        ind_xy = dict_values_names2.index("xypix_xyfit")

        npics_from_dict_dips_theor = len(list(dict_dips_theor.keys()))

        print(" npics_from_dict_dips_theor = ", npics_from_dict_dips_theor)
        print("more peaks than in dat file because harmonics are included")
        add_str = add_str + "_with_theor_dips"

        if npics_from_dict_dips_theor != nbIntProfiles:
            print("oups : npics_from_dict_dips_theor differs from nbIntProfiles")
            ncol_in_listI1 = -1 * np.ones(npics_from_dict_dips_theor, int)
            for key, value in dict_dips_theor.items():
                xypic_integer_d = np.array(value[ind_xy][:2].round(decimals=1), dtype=int)
                #                print xypic_integer_d
                dxy = abs(xyIntProfiles - xypic_integer_d).sum(axis=1)
                #                print dxy
                ind0 = np.where(dxy == 0)
                if len(ind0[0] > 1):
                    ncol_in_listI1[key] = ind0[0][0]
            print("ncol_in_listI1 =", ncol_in_listI1)

            nbIprofiles = npics_from_dict_dips_theor

        else:
            nbIprofiles = nbIntProfiles

    else:
        nbIprofiles = nbIntProfiles

    if filediplinks is not None:
        print("reading filediplinks from:")
        print(filediplinks)
        diplinks = np.loadtxt(filediplinks, skiprows=1)
        diplinks = np.array(diplinks.round(decimals=0), dtype=int)
        ind0 = np.where(diplinks[:, 4] >= 1)
        if len(ind0[0]) > 0:
            diplinks = diplinks[ind0[0]]
        # print diplinks
        _ = np.shape(diplinks)[0]
        add_str = add_str + "_with_exp_dips"

    if file_exp_dia is not None:
        print("reading fileexpdia from :")
        print(file_exp_dia)
        dict_exp = read_dict_diaexp(file_exp_dia)
        IntProfilesData = np.array(dict_exp[0], dtype=float)
        inputfile = file_exp_dia

    # Getting x, y h, k,l, energy info from  indexedspotlistwithharmonics file
    # generated by build_spotlistref_sample()
    if indexedspotlistwithharmonics is not None:
        # spot list from : /home/micha/LaueProjects/DiamondPesciAug2018/diam/Gediam_0000.dat
        # numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9, uqlab 10:13
        DataExpSpots = np.loadtxt(indexedspotlistwithharmonics, skiprows=5)

    # single intensity profile case
    if xy_single_spot is not None:
        if file_Ipix_vs_img_single_spot is None:
            xy_single_spot = np.array(xy_single_spot, dtype=int)
            xyIntProfiles = read_xypic_in_file_Ipix_vs_img(file_Ipix_vs_img)
            dxy = abs(xyIntProfiles - xy_single_spot).sum(axis=1)
            ind0 = np.where(dxy == 0)
            if len(ind0[0] > 1):
                ncol_in_listI1 = [ind0[0][0]]
            nbIprofiles = 1
        else:
            nbIprofiles = 1
            ncol_in_listI1 = [0]

    imageindiceslist = IntProfilesData[:, 0]
    nbimages = len(imageindiceslist)

    # array of intensity profiles
    # shape = (nbimages, nbIntensityProfiles)
    intensitiesArray = IntProfilesData[:, 1:]

    # preparation of intensity profiles
    yy = np.zeros(nbimages, dtype=np.float)

    if imgref != None:
        ind1 = np.where(imageindiceslist == imgref)
        if len(ind1[0]) > 0:
            refpoint = ind1[0][0]
        else:
            print("DIA.plot_Ipix_vs_img : ")
            print("imgref not found in imageindiceslist")
            exit()

    if npics_max is not None:
        if npics_max < nbIprofiles:
            nbIprofiles = npics_max

    #     if npics_max is None:
    #         indicesIprofiles = np.arange(nbIprofiles)
    #
    #     elif npics_max is not None:
    #         if isinstance(npics_max,int):
    #             indicesIprofiles = np.arange(npics_max)
    #         else:
    #             indicesIprofiles = npics_max

    # ---   PLOTTING PART ------------
    firsttime = 1
    figlist = []
    figureindex = 0
    # print "k_profile_w_harmonics indexExpProfile Ipixref Ipixstart"

    print("\n\n ncol_in_listI1", ncol_in_listI1)

    print("k_profile_w_harmonics, xypix, yy.max-yy.min, yy.np.mean(), Ipix min, Ipix max ")
    #     for k_profile_w_harmonics, expspotindex in enumerate(indicesIprofiles):
    for k_profile_w_harmonics in range(nbIprofiles):
        if ncol_in_listI1 is None:
            good_col = k_profile_w_harmonics
        else:
            good_col = ncol_in_listI1[k_profile_w_harmonics]

        if filedipstheor is not None:

            indexExpProfile = int(np.round(dict_dips_theor[k_profile_w_harmonics][0], 0))

        if file_exp_dia is not None:
            indexExpProfile = int(np.round(dict_exp[1][k_profile_w_harmonics][0], 1))

        # if profileindex is a multiple of 10
        # then create figure
        def tick_function(X):
            V = fromimageindexTothf(X, scanObject)
            return ["%.2f" % z for z in V]

        if not k_profile_w_harmonics % 10:

            fig = p.figure(figsize=figsize)
            figlist.append(fig)
            ax1 = fig.add_subplot(111)

            ax1.set_xlabel("image index")
            ax1.set_ylabel("intensity profile")

            ax1.grid(color="b")

            ax2 = ax1.twiny()

            new_tick_locations = np.linspace(scanObject.img[0], scanObject.img[-1], 5)
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(tick_function(new_tick_locations))
            ax2.set_xlabel(r"thf")
            #print("xxthf", xxthf)
            #ax2.plot(xxthf, np.ones(len(scanObject.img)))  # Create a dummy plot
            # ax2.cla()

            if not firsttime:
                if save_figures:
                    figfilename = inputfile.rstrip(".dat") + add_str + "_fig" + str(figureindex) + ".png"
                    print("saving figure in ", figfilename)
                    fig.savefig(figfilename, transparent=False, bbox_inches="tight")

            print("k_profile_w_harmonics", k_profile_w_harmonics)
            ax1.set_ylim(k_profile_w_harmonics / 2, k_profile_w_harmonics / 2 + 7)
            firsttime = 0
            figureindex = +1

        if nbIprofiles == 1:
            vertical_shift = 0.0
        else:
            vertical_shift = (float(k_profile_w_harmonics) + 1.0) * 0.5

        # build vertically shifted intensity profiles
        if mon_in_scan is not None:  # monitor normalization
            yy = (intensitiesArray[:, good_col] - pedestal) / (
                intensitiesArray[refpoint, good_col] - pedestal
            ) * (mon_in_scan[refpoint] - mon_offset) / (mon_in_scan[:] - mon_offset) + vertical_shift
        else:  # no monitor normalization
            yy = (intensitiesArray[:, good_col] - pedestal) / (
                intensitiesArray[refpoint, good_col] - pedestal
            ) + vertical_shift

        if (thf_in_scan is not None) & (filedipstheor is None):
            xx = thf_in_scan

        elif use_npoint_as_x is not None:
            xx = np.arange(nbimages)

        else:
            xx = imageindiceslist

        # p.plot(xx, yy, "bo-", ms = 3)
        if remove_saturated is not None:
            maxI = intensitiesArray[:, good_col].max()
            if maxI < remove_saturated:
                ax1.plot(xx, yy, "b-")
        else:
            ax1.plot(xx, yy, "b-")

        if (thf_in_scan is not None) & (filedipstheor is None):
            ax1.set_xlim(thf_in_scan[0], thf_in_scan[-1])

        # plot annotations on right hand side
        # several intensity profiles case
        if file_Ipix_vs_img_single_spot is None:
            # plot diamond branches info
            if (file_exp_dia is not None) | (filedipstheor is not None):
                label1 = str(k_profile_w_harmonics) + " , spot#" + str(indexExpProfile)

            else:
                label1 = str(k_profile_w_harmonics) + " : "
                label1 += str(xyIntProfiles[k_profile_w_harmonics, 0]) + ","
                label1 += str(xyIntProfiles[k_profile_w_harmonics, 1])

            if indexedspotlistwithharmonics is not None:
                label1 += " E=%0.f" % DataExpSpots[k_profile_w_harmonics][9]
                label1 += " (X,Y)=(%0.f,%.0f)" % tuple(DataExpSpots[k_profile_w_harmonics][1:3])

        # single intensity profile case
        else:
            label1 = str(xy_single_spot[0]) + "," + str(xy_single_spot[1])

        ax1.text(xx[-1], yy[-1], label1, fontsize=9)

        if imgref is not None:
            _ = "Ipix/Ipix(imgref =" + str(imgref) + ")"
            if use_npoint_as_x is not None:
                ax1.axvline(x=refpoint, color="k_profile_w_harmonics")
            else:
                ax1.axvline(x=imgref, color="k_profile_w_harmonics")
        else:
            _ = "Ipix/Ipix(refpoint =" + str(refpoint) + ")"

        nptmax = len(yy)

        _ = yy[:nptmax].max() - yy[:nptmax].min()

        # statistics on normalized and stacked relative intensity variation
        #         firsttime2 = 1
        #
        #         if range2 > 0.1 :
        #             print "k_profile_w_harmonics, xypix, yy.max-yy.min, yy.np.mean(), Ipix min, Ipix max "
        #             print k_profile_w_harmonics, xyIntProfiles[k_profile_w_harmonics,:], np.round(range2,4), np.round(yy[:nptmax].np.mean(),1), int(intensitiesArray[:nptmax,good_col].min()), int(intensitiesArray[:nptmax,good_col].max())
        #             p.figure(num = figureIndex)
        #             histo = np.histogram(yy[:nptmax], bins = 20)
        #     #    print "histogram data : ", histo[0]
        #     #    print "bin edges :",  histo[1]
        #     #    print np.shape(histo[0])
        #     #    print np.shape(histo[1])
        #             barwidth = histo[1][1]-histo[1][0]
        #     #    print "bar width = ", barwidth
        #             p.bar(histo[1][:-1], histo[0], width = barwidth)
        #             if firsttime2 :
        #                 p.xlabel("yy")
        #                 p.ylabel("frequency")
        #                 firsttime2 = 0
        # #            label2 = str(xyIntProfiles[k_profile_w_harmonics,0]) + "," + str(xyIntProfiles[k_profile_w_harmonics,1])
        #         figureIndex=+1

        #         p.figure(num = figureIndex)
        #        if imgref is not None :
        #            p.axvline(x=imgref,color = "k_profile_w_harmonics")
        #            xmin1 = imgref -10
        #            xmax1 = imgref +10
        #            p.axvline(x=imgref-51,color = "k_profile_w_harmonics")
        #            p.axvline(x=imgref+51,color = "k_profile_w_harmonics")
        #            p.xlim(xmin1,xmax1)

        ### add dip_theor
        # 0 indexExpProfile 0
        # 1 HKL_sample 0
        # 2 xypix_xyfit 1
        # 3 Ipixmax 0
        # 4 Etheor_sample 1
        # 5 uqlab_sample 1
        # 6 ndia_list 0
        # 7 HKL_dia_list_theor 2
        # 8 thf_list_theor 1
        # 9 dip_depth_theor 1
        # 10 slope_thf_vs_Edia 1
        # 11 uqlab_dia_0 1
        # 12 uqlab_dia_1 1
        # 13 uqlab_dia_2 1
        # 14 inner_uq_dia_xz_uq_sample_xz 1
        # 15 uq_dia_z 1

        if filedipstheor is not None:

            ndia_list = np.array(dict_dips_theor[k_profile_w_harmonics][ind_ndia], dtype=int)
            # print ndia_list
            ndips_theor = len(ndia_list)

            if (ndips_theor == 1) & (ndia_list[0] == -1):
                continue

            dip_depth_theor = np.array(dict_dips_theor[k_profile_w_harmonics][ind_depth], dtype=float)
            thf_dip_theor = np.array(dict_dips_theor[k_profile_w_harmonics][ind_thf_dip], dtype=float)

            img_dips = img_in_scan[0] + (thf_dip_theor - thf_in_scan[0]) * (
                img_in_scan[-1] - img_in_scan[0]) / (thf_in_scan[-1] - thf_in_scan[0])

            colordia = []
            for i in range(ndips_theor):
                if dip_depth_theor[i] > 75.0:
                    color1 = "r"
                elif dip_depth_theor[i] > 10.0:
                    color1 = "m"
                elif dip_depth_theor[i] > 3.0:
                    color1 = "b"
                elif dip_depth_theor[i] > 0.5:
                    color1 = "g"
                else:
                    color1 = "y"
                colordia.append(color1)

            _ = 0.0

            for i in range(ndips_theor):
                # print dip_depth_theor[i]
                if (thf_dip_theor[i] > thf_min) & (thf_dip_theor[i] < thf_max):

                    xx4 = np.array([img_dips[i], img_dips[i] + 0.001])
                    yymean = (yy[0] + yy[-1]) / 2.0
                    yy4 = np.array([yymean - 0.15, yymean + 0.15])
                    color1 = "-" + colordia[i]
                    ax1.plot(xx4, yy4, color1, lw=2)
                    tt = ndia_list[i]
                    # print tt
                    str1 = str(tt)
                    ax1.text(xx4[0] + 5, yy4[0], str1)

                    #                    "k_profile_w_harmonics 0, indexExpProfile 1, ndia 2, img 3, confidence 4 dip_depth_theor 5"
                    tt = np.array(
                        [
                            k_profile_w_harmonics,
                            indexExpProfile,
                            ndia_list[i],
                            int(np.round(img_dips[i], 1)),
                            0,
                            int(np.round(dip_depth_theor[i], 1)),
                        ],
                        dtype=int,
                    )

                    if dip_depth_min is None:
                        list_diplinks_theor.append(tt)
                    else:
                        if dip_depth_theor[i] > dip_depth_min:
                            list_diplinks_theor.append(tt)


        if filediplinks is not None:
            if indexExpProfile in diplinks[:, 1]:
                ind0 = np.where(diplinks[:, 1] == indexExpProfile)
                #                 print ind0[0]
                if len(ind0[0]) > 0:
                    xx = diplinks[ind0[0], 3]
                    yy2 = np.ones(len(xx), float) * (yy[0] - 0.25)
                    for i in range(len(xx)):
                        xx1 = np.array([xx[i], xx[i] + 0.001])
                        yy3 = np.array([yy[0] - 0.3, yy[0] - 0.1])
                        ax1.plot(xx1, yy3, "k_profile_w_harmonics-")
                        label1 = " " + str(xx[i])
                        ax1.text(xx[i], yy2[i], label1, color="r")

    if save_figures & (nbIprofiles > 1):
        figfilename = inputfile.rstrip(".dat") + add_str + "_fig" + str(figureindex) + ".png"
        print("saving figure in ", figfilename)
        fig.savefig(figfilename, transparent=False, bbox_inches="tight")

    outputfilename_filediplinks = None
    if filedipstheor is not None:
        header = "k_profile_w_harmonics 0, indexExpProfile 1, ndia 2, img 3, confidence 4, depth 5 \n"
        list_diplinks_theor = np.array(list_diplinks_theor, dtype=int)
        print("list_diplinks_theor : \n", list_diplinks_theor)
        print(np.shape(list_diplinks_theor))
        outputfilename_filediplinks = filedipstheor.rstrip(".dat") + "_diplinks_theor.dat"
        print("diplinks theor saved in : ", outputfilename_filediplinks)
        outputfile = open(outputfilename_filediplinks, "w")
        outputfile.write(header)
        np.savetxt(outputfile, list_diplinks_theor, fmt="%d")
        outputfile.close()

        # pickle file
        data_pickled = (header, list_diplinks_theor)
        picklefilename_filediplinks = filedipstheor.rstrip(".dat") + "_diplinks_theor.pckl"

        with open(picklefilename_filediplinks, "wb") as f:
            pickle.dump(data_pickled, f)

    return outputfilename_filediplinks


def calc_diamond_rotation_axis(dict_diafitfiles, fitfile_type="MGnew", ref_axis=None):

    # xlab is close to the diamond rotation axis
    # so it is the lab axis that moves the less with respect to the crystal
    # when changing thf
    # so it is good as a reference for choosing stereo_mat
    # xlab_in_sample_coord = xlab

    #    xlab = np.array([1.,0.,0.])

    #    tt = 40.0*math.pi/180.
    #    ylab_in_sample_coord = np.cos(tt)*np.array([0.,1.,0.])-sin(tt)*np.array([0.,0.,1.])
    # print "ylab in sample coord :", axis_pole_sample_ylab

    #    ref_axis = xlab
    #    ref_axis = ylab_in_sample_coord
    #    ref_axis = None

    vhkl, vlab, ang1, matstarlabOND1 = MG.twofitfiles_to_rotation(
        filefit1=dict_diafitfiles["start"][0],
        matref1=None,
        filefit2=dict_diafitfiles["end"][0],
        apply_cubic_opsym_to_mat2_to_have_min_angle_with_mat1="no",
        apply_cubic_opsym_to_both_mat_to_have_one_sample_axis_in_first_stereo_triangle=ref_axis,
        fitfile_type=fitfile_type)

    print(dict_diafitfiles["start"])
    print("matstarlabOND1 \n", matstarlabOND1.round(decimals=7))
    print("matstarlabOND1 \n", list(matstarlabOND1))
    print(dict_diafitfiles["end"])
    print("use matrixes with +/- ref axis in first stereo triangle")
    print("ref_axis :", ref_axis)
    print("rotation :")
    print("axis_HKL  axis_xyz angle(deg)")
    print(vhkl.round(decimals=6), vlab.round(decimals=6), np.round(ang1, 5))
    tt = max(abs(vhkl))
    print("max(abs(vhkl))", tt)
    print("vhkl normalized", (vhkl / tt).round(decimals=2))

    ang_start_end = ang1

    vlab_start_end = vlab.round(decimals=8)

    vhkl, vlab, ang1, matstarlabOND1 = MG.twofitfiles_to_rotation(
        filefit1=dict_diafitfiles["center"][0],
        matref1=None,
        filefit2=dict_diafitfiles["end"][0],
        apply_cubic_opsym_to_mat2_to_have_min_angle_with_mat1="no",
        apply_cubic_opsym_to_both_mat_to_have_one_sample_axis_in_first_stereo_triangle=ref_axis,
        fitfile_type=fitfile_type)

    print(dict_diafitfiles["center"])
    print("matstarlabOND1 \n", matstarlabOND1.round(decimals=7))
    print("matstarlabOND1 \n", list(matstarlabOND1))
    print(dict_diafitfiles["end"])
    print("use matrixes with +/- ref axis in first stereo triangle")
    print("ref_axis :", ref_axis)
    print("rotation :")
    print("axis_HKL  axis_xyz angle(deg)")
    print(vhkl.round(decimals=6), vlab.round(decimals=6), np.round(ang1, 5))
    tt = max(abs(vhkl))
    print("max(abs(vhkl))", tt)
    print("vhkl normalized", (vhkl / tt).round(decimals=2))

    ang_center_end = ang1

    matref_center = matstarlabOND1.round(decimals=8)

    # take the printout and stick it in Param_your_exp.py
    # adding np.array() around each list

    print("matref_center = ", list(matref_center))
    print("vlab_start_end = ", list(vlab_start_end))
    print("ang_start_end, ang_center_end = ", np.round(ang_start_end, 5), np.round(ang_center_end, 5))

    return (matref_center, vlab_start_end)


def save_spot_results(
    filename, spotlist, column_list, user_comment, full_list="yes", normalized_intensity_lim=0.001):

    header = "#Spotlist360 : "
    if full_list == "yes":
        header += "full spot list - keep all spots \n"
    else:
        header += "short spot list - keep only spots with intensity larger than %s \n" % (
            normalized_intensity_lim
        )
    header += "#File created at %s with diamond.py \n" % (time.asctime())
    header += "#" + column_list + "\n"
    outputfile = open(filename, "w")
    outputfile.write(header)
    np.savetxt(outputfile, spotlist, fmt="%.4f")
    outputfile.write(user_comment)
    outputfile.close()

    #    print "spotlist saved in :"
    #    print filename

    return filename


def build_diamond_spotlist_360_v2(thf_list,
                                filepathdia,
                                calibdia,
                                Emin=5.0,
                                Emax=22.0,
                                showall=0,
                                dict_diafitfiles=None,  # for comments only
                                vlab_start_end=None,  # for comments only
                                dlatu_nm_rad=None,  # for comments only
                            ):
    mat_range = list(range(0, 9))
    vlab_range = list(range(9, 12))
    uilab_range = list(range(12, 15))
    thfref_range = 15

    vlab = calibdia[vlab_range]
    thf_ref = calibdia[thfref_range]
    matref_line = calibdia[mat_range]
    matref_center = GT.matline_to_mat3x3(matref_line)
    uilab = calibdia[uilab_range]

    cryst_struct = "dia"

    dxytol = 0.01

    npts = len(thf_list)
    for k in range(npts):
        thf = thf_list[k]
        matrot = MG.from_axis_vecangle_to_mat(vlab, thf_ref - thf)  # start
        matnew = np.dot(matrot, matref_center)
        mat = GT.mat3x3_to_matline(matnew)
        print("matrix : \n", mat)

        spotlist2 = MG.spotlist_360(Emin, Emax, mat, cryst_struct, showall, uilab=uilab)
        hkl2 = spotlist2[:, 1:4]
        if k == 0:
            hklref = hkl2
            nspots = np.shape(spotlist2)[0]
            tt = np.ones(nspots, float) * thf
            spotlistref = np.column_stack((spotlist2, tt))
        else:
            _, _, iscommon2 = MG.find_common_peaks(
                hklref, hkl2, dxytol=dxytol, verbose=0)
            ind2 = np.where(iscommon2 == 0)
            if len(ind2[0]) > 0:
                nspots = len(ind2[0])
                tt = thf * np.ones(nspots, float)
                spotlistnew = np.column_stack((spotlist2[ind2[0]], tt))
                spotlistref = np.row_stack((spotlistref, spotlistnew))
                hklnew = hkl2[ind2[0]]
                hklref = np.row_stack((hklref, hklnew))

        print("k =", k)
        print("nspots = ", np.shape(spotlistref)[0])

    outputfilename = filepathdia + "spotlistref_all.dat"
    print("spot list saved in : ", outputfilename)

    column_list = "E 0, hkl 1:4, uflab 4:7, tth 7, chi 8, thf 9 "  # , fpol 9, thf 10'

    user_comment = "# Description of parameters : " + "\n"
    user_comment += "#- vlab : corrected rotation axis, in lab coord" + "\n"
    user_comment += "#- vlab_start_end : initial rotation axis, in lab coord" + "\n"
    user_comment += "#- uilab : corrected incident beam, in lab coord" + "\n"
    user_comment += "# NB : usual uilab = [0.,1.,0.]" + "\n"
    user_comment += "# correcting uilab is a mathematical trick to avoid correcting matref" + "\n"
    user_comment += "# matref = reference orientation matrix at thf=thf_ref, astar bstar cstar as one 9-elements line on xyzlab \n"
    user_comment += "# Lab frame : ylab axis along beam (ui) , zlab axis in (ui,zcam) plane (approx. vertical) , zcam perp. to detector screen \n"
    user_comment += "# Step 1 (DIA.build_diamond_spotlist_360_v2) :" + "\n"
    user_comment += "# Description of columns in spotlistref_all.dat :" + "\n"
    user_comment += "# thf = thf at which spot first appears when scanning thf_list \n"
    user_comment += "# E, uflab, tth, chi valid only at this thf \n"

    if dict_diafitfiles is not None:
        user_comment += (
            "# List of diamond Laue patterns used to calculate orientation matrix and rotation axis :" + "\n"
        )
        user_comment += "# start : " + dict_diafitfiles["start"][0] + "\n"
        user_comment += "# thf_start : " + str(dict_diafitfiles["start"][1]) + "\n"
        user_comment += "# center : " + dict_diafitfiles["center"][0] + "\n"
        user_comment += "# thf_center : " + str(dict_diafitfiles["center"][1]) + "\n"
        user_comment += "# end : " + dict_diafitfiles["end"][0] + "\n"
        user_comment += "# thf_end : " + str(dict_diafitfiles["end"][1]) + "\n"

    user_comment += "# Values of parameters used for step 1 :" + "\n"
    user_comment += "#thf_ref step 1 : " + str(thf_ref) + "\n"

    str1 = " ".join(str(e) for e in thf_list)
    # print str1
    user_comment += "#thf_list step 1: " + str1 + "\n"

    str1 = " ".join(str(e) for e in matref_line)
    # print str1
    user_comment += "#matref step 1: " + str1 + "\n"
    if vlab_start_end is not None:
        user_comment += "#vlab_start_end step 1:" + str(vlab_start_end) + "\n"
    user_comment += "#vlab step 1: " + str(vlab) + "\n"
    user_comment += "#uilab step 1: " + str(uilab) + "\n"
    user_comment += "#direct lattice parameters (nm,rad) : " + str(dlatu_nm_rad) + "\n"
    user_comment += "#Emin (keV): " + str(Emin) + "\n"
    user_comment += "#Emax (keV): " + str(Emax) + "\n"

    print("user_comment :")
    print(user_comment)

    save_spot_results(outputfilename, spotlistref, column_list, user_comment, full_list="yes")

    return outputfilename


def remove_low_intensity_diamond_spots_from_list(
    filespot, normalized_intensity_lim, filepathdia, file_structfact_dia):

    spotlist1 = np.loadtxt(filespot)
    nspots_start = np.shape(spotlist1)[0]
    struct_fact0 = np.loadtxt(file_structfact_dia, skiprows=1)

    struct_fact = MG.sort_list_decreasing_column(struct_fact0, 3)
    # print struct_fact
    # print np.shape(struct_fact)

    nstruc = np.shape(struct_fact)[0]
    qnorm = np.zeros(nstruc, float)
    sf1 = struct_fact[:, 3]

    print("diamond structure factors from cctbx web service")
    print("hkl qnorm sf")
    for i in range(nstruc):
        qnorm[i] = norme(struct_fact[i, :3])
        print(struct_fact[i, :3], np.round(qnorm[i], 4), np.round(sf1[i], 10))

    print("................ 1 ...............")

    nspots1 = np.shape(spotlist1)[0]
    hkl1 = spotlist1[:, 1:4]
    qnorm2 = np.zeros(nspots1, float)
    for i in range(nspots1):
        qnorm2[i] = norme(hkl1[i, :])

    spotlist2 = np.column_stack((hkl1, qnorm2))
    spotlist2 = MG.sort_list_decreasing_column(spotlist2, -1)
    print("diamond spotlist")
    print("hkl qnorm")
    for i in range(nspots1):
        print(spotlist2[i, :3], np.round(spotlist2[i, 3], 4))
    print("............... 2 ..................")

    sf2 = np.zeros(nspots1, float)
    sf_sqr = np.zeros(nspots1, float)
    isbadpeak1 = np.ones(nspots1, int)
    for i in range(nspots1):
        for j in range(nstruc):
            dq = abs(spotlist2[i, 3] - qnorm[j])
            if dq < 1.0e-2:
                sf2[i] = sf1[j]
                sf_sqr[i] = sf2[i] * sf2[i]
                isbadpeak1[i] = 0

    max_sf_sqr = max(sf_sqr)
    sf_sqr_norm = 100.0 * sf_sqr / max_sf_sqr
    print("hkl qnorm sf sf_sqr 100*sf_sqr_norm")
    for i in range(nspots1):
        print(spotlist2[i, :3],
                np.round(spotlist2[i, 3], 2),
                np.round(sf2[i], 3),
                np.round(sf_sqr[i], 3),
                np.round(sf_sqr_norm[i], 5))

    spotlist3 = np.column_stack((spotlist2[:, :3], sf_sqr_norm))

    ind2 = np.where(isbadpeak1 == 0)
    ind1 = np.where(isbadpeak1 == 1)
    print("bad spots i.e. not included in cctbx list : ", ind1[0])
    spotlist3 = spotlist3[ind2[0], :]

    # filtrage spots diamant
    ind1 = np.where(spotlist3[:, -1] > normalized_intensity_lim * 100.0)
    print("intensity threshold (% of max) : ", normalized_intensity_lim)
    print("number of spots remaining : ", np.shape(ind1[0])[0])
    # print ind1[0]

    spotlistref2 = spotlist3[ind1[0], :]

    ind2 = np.where(abs(spotlistref2[:, -1] - 100.0) < 0.01)
    print("most intense lines :")
    tt = spotlistref2[ind2[0], :]
    print(tt)

    nspots_end = np.shape(spotlistref2)[0]

    column_list = "hkl 0:3, sf_sqr_norm*100 3"
    print(column_list)

    outputfilename = filepathdia + "spotlistref_short.dat"
    print("spot list saved in : \n ", outputfilename)
    print("nspots start / end : ", nspots_start, nspots_end)

    # recuperation des user_comments
    i = 0
    user_comment = ""
    linepos_start_user_comments = 1e6

    f = open(filespot, "r")
    try:
        for line in f:
            if line.startswith("# Description of parameters : "):
                linepos_start_user_comments = i
            if i >= linepos_start_user_comments:
                user_comment += line
            i = i + 1
    finally:
        f.close()

    #    print user_comment
    user_comment += "# Step 2 (DIA.remove_low_intensity_diamond_spots_from_list) :" + "\n"
    user_comment += "#Values of parameters for step 2 :" + "\n"
    user_comment += "#keep only spots with normalized intensity larger than normalized_intensity_lim" + "\n"
    user_comment += "#normalized_intensity_lim :" + str(normalized_intensity_lim) + "\n"

    save_spot_results(outputfilename,
                    spotlistref2,
                    column_list,
                    user_comment,
                    full_list="no",
                    normalized_intensity_lim=normalized_intensity_lim)

    return outputfilename


def save_dict_Edia_vs_thf(outputfilename, dict_Edia, user_comment, thf_list):

    header = "#Edia vs thf curves for diamond spots (dictionnary, key = spotnum) \n"
    header += "#File created at %s with diamond.py \n" % (time.asctime())
    outputfile = open(outputfilename, "w")
    outputfile.write(header)
    #    outputfile.write("#User comments from filespot : \n")
    outputfile.write(user_comment)
    #    outputfile.write("#New comments : \n")
    #    # needs thf_list without # for reading
    #    outputfile.write("#Geometry corrections used for Edia calculation : \n")
    #    str1 = ' '.join(str(e) for e in geometry_corrections)
    #    outputfile.write('#ui_pitch ui_yaw axis_yaw axis_roll (0.1 mrad units) : ' + str1 +'\n')
    #    outputfile.write('#Incident beam in lab coordinates (corrected) : '+ str(uilab)+'\n')
    #    outputfile.write('#diamond rotation axis in lab coordinates (corrected) : '+ str(vlab)+'\n')
    #    # needs thf_list without # for reading
    #    str1 = ' '.join(str(e) for e in thf_list)
    #    outputfile.write('thf_list\n')
    #    outputfile.write(str1 +'\n')
    # key = dia spot num
    dict_values_names = ["hkldia",
                        "intensity_norm",
                        "Edia_list",
                        "fpol_list",
                        "Edia_mean_min_max",
                        "thf_mean_min_max",
                        "uqlab_0",
                        "uqlab_1",
                        "uqlab_2"]

    ndict = len(dict_values_names)
    for i in range(ndict):
        outputfile.write(dict_values_names[i] + "\n")
        for key, value in dict_Edia.items():
            if i in [1]:
                str1 = str(value[i])
            else:
                str1 = " ".join(str(e) for e in value[i])
            outputfile.write(str(key) + " : " + str1 + "\n")

    outputfile.close()

    return 0


# --- ----------  build_dict_Edia_vs_thf ---------
def build_dict_Edia_vs_thf(filespot,
                            thf_list,
                            filepathdia,
                            calibdia,
                            Emin,
                            Emax,
                            geometry_corrections,  # for comments only
                            verbose=0):
    """
    Calculation of branch diamond properties
    filespot      spot list ref_short.dat     h,k,l F**2

    calibdia
    """

    mat_range = list(range(0, 9))
    vlab_range = list(range(9, 12))
    uilab_range = list(range(12, 15))
    thfref_range = 15

    vlab = calibdia[vlab_range]
    thf_ref = calibdia[thfref_range]
    matref_line = calibdia[mat_range]
    matref_center = GT.matline_to_mat3x3(matref_line)
    uilab = calibdia[uilab_range]

    # recuperation des user_comments
    i = 0
    user_comment = ""
    linepos_start_user_comments = 1e6

    f = open(filespot, "r")
    try:
        for line in f:
            if line.startswith("# Description of parameters : "):
                linepos_start_user_comments = i
            if i >= linepos_start_user_comments:
                user_comment += line
            i = i + 1
    finally:
        f.close()

    user_comment += "# Step 3 (build_dict_Edia_vs_thf) : " + "\n"
    user_comment += "#Values of parameters used for step 3 : " + "\n"
    user_comment += "#Geometry corrections :" + "\n"
    str1 = " ".join(str(e) for e in geometry_corrections)
    user_comment += "#ui_pitch ui_yaw axis_yaw axis_roll (0.1 mrad units) step 3: " + str1 + "\n"
    user_comment += "#thf_ref step 3 : " + str(thf_ref) + "\n"
    str1 = " ".join(str(e) for e in matref_line)
    # print str1
    user_comment += "#matref step 3 : " + str1 + "\n"
    user_comment += "#vlab step 3 : " + str(vlab) + "\n"
    user_comment += "#uilab step 3 : " + str(uilab) + "\n"
    # needs thf_list without # for reading
    str1 = " ".join(str(e) for e in thf_list)
    user_comment += "thf_list" + "\n"
    user_comment += str1 + "\n"

    # modif 10Feb14 add uqlab

    npts = len(thf_list)

    print("spot list from : ", filespot)
    spotlist1 = np.loadtxt(filespot)

    spotlist2 = MG.sort_list_decreasing_column(spotlist1, -1)
    nspots = np.shape(spotlist2)[0]
    print("number of spots :", nspots)

    # nspots = 4

    #     print "hkl sf_sqr_norm"
    #     for i in range(nspots):
    #         print spotlist2[i,:3], spotlist2[i,-1]

    dict_Edia = {}

    for i in range(nspots):
        dict_Edia[i] = [spotlist2[i, :3], spotlist2[i, -1]]

    dict_values_names = ["hkldia",
                        "intensity_norm",
                        "Edia_list",
                        "fpol_list",
                        "Edia_mean_min_max",
                        "thf_mean_min_max"]
    _ = len(dict_values_names)

    xlab = np.array([1.0, 0.0, 0.0])

    matall = np.zeros((npts, 9), float)

    Edia = np.zeros((nspots, npts), float)
    fpol = np.zeros((nspots, npts), float)
    uqlab1 = np.zeros((nspots, npts, 3), float)

    # near thf = -45
    restricted_range = list(range(npts))  # full range
    # thf_range = range(80,90)

    for k in restricted_range:
        thf = thf_list[k]

        matrot = MG.from_axis_vecangle_to_mat(vlab, thf_ref - thf)  # v1
        matnew = np.dot(matrot, matref_center)
        mat = GT.mat3x3_to_matline(matnew)
        if verbose:
            print("thf =", thf)
            print("matrix : \n", mat)
        matall[k, :] = mat * 1.0

    Emin_eV = Emin * 1000.0
    Emax_eV = Emax * 1000.0

    for k in restricted_range:
        mat = matall[k, :] * 1.0
        # print "k = ", k
        for i in range(nspots):
            hkldia = np.array(dict_Edia[i][0], dtype=float)
            # print "HKLdia = ", hkldia
            H = hkldia[0]
            K = hkldia[1]
            L = hkldia[2]
            qlab = H * mat[0:3] + K * mat[3:6] + L * mat[6:]
            uqlab = qlab / norme(qlab)
            uqlab1[i, k] = uqlab * 1.0
            # print "uqlab = ", uqlab
            sintheta = -1.0 * np.inner(uqlab, uilab)
            # print "sintheta = ", sintheta
            if sintheta > 0.0:
                # print "reachable reflection"
                tt = DictLT.E_eV_fois_lambda_nm * norme(qlab) / (2.0 * sintheta)
                # print "Edia = ", tt
                if (tt > Emin_eV) & (tt < Emax_eV):
                    Edia[i, k] = np.round(tt, 2)
                    # print Edia[i,k]
                    uflab = uilab + 2 * sintheta * uqlab

                    if 0:  # ancienne version du calcul de fpol
                        _, tth = MG.uflab_to_2thetachi(uflab)
                        un = np.cross(uilab, uqlab)
                        un = un / norme(un)
                        fsig = np.inner(un, xlab)
                        fpi = np.sqrt(1 - fsig * fsig)
                        cos2theta = np.cos(tth * np.pi / 180.0)
                        tt = np.sqrt(fsig * fsig + fpi * fpi * cos2theta * cos2theta)
                        fpol[i, k] = np.round(tt, 3)
                    if 1:  # version GR du calcul de fpol
                        uflab_xy = np.array([uflab[0], uflab[1], 0.0])
                        uflab_xy = uflab_xy / norme(uflab_xy)
                        tt = np.inner(uilab, uflab_xy)
                        fpol[i, k] = np.round(tt, 3)

            # else : print "unreachable reflection"

    Edia_mean_min_max = np.zeros((nspots, 3), float)
    thf_mean_min_max = np.zeros((nspots, 3), float)
    for i in range(nspots):
        ind1 = np.where(Edia[i, :] > 0.1)
        if verbose:
            print(ind1[0])
        if len(ind1[0]) > 0:
            Edia_min = Edia[i, ind1[0]].min()
            Edia_max = Edia[i, ind1[0]].max()
            Edia_mean = Edia[i, ind1[0]].mean()
            Edia_mean_min_max[i] = np.array([Edia_mean, Edia_min, Edia_max]).round(decimals=2)
            thf_min = thf_list[ind1[0]].min()
            thf_max = thf_list[ind1[0]].max()
            thf_mean = thf_list[ind1[0]].mean()
            thf_mean_min_max[i] = np.array([thf_mean, thf_min, thf_max]).round(decimals=3)

    for i in range(nspots):
        dict_Edia[i].append(Edia[i, :])
        dict_Edia[i].append(fpol[i, :])
        dict_Edia[i].append(Edia_mean_min_max[i, :])
        dict_Edia[i].append(thf_mean_min_max[i, :])
        dict_Edia[i].append(uqlab1[i, :, 0].round(decimals=3))
        dict_Edia[i].append(uqlab1[i, :, 1].round(decimals=3))
        dict_Edia[i].append(uqlab1[i, :, 2].round(decimals=3))

    #    # version array
    #     for i in range(ndict):
    #         print dict_values_names[i]
    #         for key, value in dict_Edia.iteritems():
    #             print key,value[i]

    outputfilename = filepathdia + "dict_Edia_with_uqlab.dat"

    print("dict_Edia saved in :", outputfilename)

    save_dict_Edia_vs_thf(outputfilename, dict_Edia, user_comment, thf_list)

    return outputfilename


def read_dict_Edia_vs_thf(fileEdia, verbose=0, read_calibdia=None):  # "old", "new"
    """
    read Abacus File : diamond branch (spots) Energy vs thf   (dictionnary file)

    read_calibdia:   'old' or 'new' to account for calibration Diamond monochromator setup parameters

    return:
     (dict_Edia, dict_values_names, thf_list)
     (dict_Edia, dict_values_names, thf_list, calibdia) if calibdia is not None

     dict_Edia:   dictionary  key = branch index, value = list with 9 following elements

    0 hkldia
    1 intensity_norm     scalar
    2 Edia_list
    3 fpol_list
    4 Edia_mean_min_max
    5 thf_mean_min_max
    6 uqlab_0
    7 uqlab_1
    8 uqlab_2

    .. todo::
        Need to be simplified
    """
    #    read_calibdia = "old" for fileEdia before 26Jan17

    print("reading Abacus dict_Edia from :")
    print(fileEdia)

    listint = []
    listfloat = list(range(0, 6))

    dict_values_names = ["hkldia",
                        "intensity_norm",
                        "Edia_list",
                        "fpol_list",
                        "Edia_mean_min_max",
                        "thf_mean_min_max"]

    dict_values_names = dict_values_names + ["uqlab_0", "uqlab_1", "uqlab_2"]
    listfloat = listfloat + list(range(6, 9))

    nbDicts = len(dict_values_names)
    linepos_list = np.zeros(nbDicts + 1, dtype=int)

    f = open(fileEdia, "r")

    endlinechar = MG.PAR.cr_string
    endlinechar = '\n\r'

    dict_string = {
        "old": ["#thf_ref", "#Matref at thf_ref",
            "#Incident beam in lab coordinates (corrected)",
            "#diamond rotation axis in lab coordinates (corrected)"],
        "new": ["#thf_ref step 3", "#matref step 3", "#uilab step 3", "#vlab step 3"]}

    if read_calibdia is not None:
        string_thf_ref = dict_string[read_calibdia][0]
        string_matref = dict_string[read_calibdia][1]
        string_uilab = dict_string[read_calibdia][2]
        string_vlab = dict_string[read_calibdia][3]

    i = 0
    try:
        for line in f:
            # print line.rstrip(endlinechar)
            for j_dict in range(nbDicts):
                if line.rstrip(endlinechar) == dict_values_names[j_dict]:
                    linepos_list[j_dict] = i
                    if verbose:
                        print("reading", j_dict, linepos_list[j_dict], line.rstrip(endlinechar))
            if (line.rstrip(endlinechar))[:8] == "thf_list":
                linepos_thf = i
                if verbose:
                    print("thf", linepos_thf)
            if read_calibdia is not None:
                #                 print "line",line
                if line.startswith("#normalized_intensity_lim"):
                    print("gfddgdfgdg")
                if line.startswith(string_thf_ref):
                    print("bingo !", line, i)
                    linepos_thf_ref = i
                if line.startswith(string_matref):
                    linepos_matref = i
                if line.startswith(string_uilab):
                    linepos_uilab = i
                if line.startswith(string_vlab):
                    linepos_vlab = i
            #             print "i",i
            i += 1
    finally:
        f.close()

    linepos_list[-1] = i

    nspots = linepos_list[1] - linepos_list[0] - 1

    if verbose:
        print("linepos_thf =", linepos_thf)
        print("linepos_list = ", linepos_list)
        if read_calibdia is not None:
            print("linepos_thf_ref =", linepos_thf_ref)
            print("linepos_matref = ", linepos_matref)
            print("linepos_uilab = ", linepos_uilab)
            print("linepos_vlab = ", linepos_vlab)
        print("nspots = ", nspots)

    f = open(fileEdia, "r")
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)

    f.seek(0)
    if verbose:
        print(f.readline())

    dict_Edia = {}

    if read_calibdia is not None:
        print("new : reading calibdia from fileEdia")
        # lecture thf_ref

        f.seek(line_offset[linepos_thf_ref])
        tt = f.readline()
        tt1 = tt.rstrip(MG.PAR.cr_string).split(":")
        #    print tt1
        thf_ref = float(tt1[1])
        print("thf_ref = ", thf_ref)

        f.seek(line_offset[linepos_matref])
        tt = f.readline()
        tt1 = tt.rstrip(MG.PAR.cr_string).split(":")
        tt2 = tt1[1].split(" ")
        matref = np.array(tt2[1:], dtype=float)  # il y a un espace qui traine au debut
        print("matref = ", matref)

        f.seek(line_offset[linepos_uilab])
        tt = f.readline()
        tt1 = tt.rstrip(MG.PAR.cr_string).split(":")
        tt2 = tt1[1].replace("[", "").replace("]", "").split()
        uilab = np.array(tt2, dtype=float)
        print("uilab = ", uilab)

        f.seek(line_offset[linepos_vlab])
        tt = f.readline()
        tt1 = tt.rstrip(MG.PAR.cr_string).split(":")
        tt2 = tt1[1].replace("[", "").replace("]", "").split()
        vlab = np.array(tt2, dtype=float)
        print("vlab = ", vlab)

        if read_calibdia == "old":
            dlatu_angstroms_deg = DictLT.dict_Materials["DIAs"][1]
            dlatu_nm_rad = MG.deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
            print("dlatu_nm_rad = ", dlatu_nm_rad)
            matref_withlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matref, dlatu_nm_rad)
            print("matref with latpar", matref_withlatpar_inv_nm)
            print("normes of matrix columns should be 2.804")
            print(MG.norme(matref_withlatpar_inv_nm[0:3]),
                MG.norme(matref_withlatpar_inv_nm[3:6]),
                MG.norme(matref_withlatpar_inv_nm[6:9]))

        elif read_calibdia == "new":
            matref_withlatpar_inv_nm = matref * 1.0

        gapnorm = abs(MG.norme(matref_withlatpar_inv_nm[0:3]) - 2.804)
        if gapnorm > 0.1:
            raise ValueError("problem with lattice parameters from matrix in DIA.read_dict_Edia_vs_thf")

        #        mat_range = range(0,9)
        #        vlab_range = range(9,12)
        #        uilab_range = range(12,15)
        #        thfref_range = 15

        calibdia = np.hstack((matref_withlatpar_inv_nm, vlab, uilab, thf_ref))

        print("calibdia = ", calibdia)

    # lecture thf
    n = linepos_thf
    #    if verbose : print "n = ", n
    f.seek(line_offset[n])
    if verbose:
        print(f.readline())
    f.seek(line_offset[n + 1])
    tt = f.readline()
    tt1 = tt.rstrip(MG.PAR.cr_string)
    #    print tt1
    tt2 = np.array(tt1.split(" "), dtype=float)
    thf_list = tt2
    #     if verbose : print 'thf_list',thf_list

    # reading dicts
    for j_dict in range(nbDicts):

        n = linepos_list[j_dict]
        if verbose:
            print("n = ", n)
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        if verbose:
            print(f.readline())
        f.seek(line_offset[n + 1])

        i_spot = 0
        while i_spot < nspots:
            tt = f.readline()
            # print('tt line -1> ',tt)
            tt1 = (tt.rstrip(MG.PAR.cr_string).split(": "))[1]

            # version array
            if j_dict in listint:
                tt2 = np.array(tt1.split(" "), dtype=int)
            elif j_dict in listfloat:
                tt2 = np.array(tt1.split(" "), dtype=float)
            # print tt2
            if j_dict == 0:
                dict_Edia[i_spot] = [tt2]
            elif j_dict == 1:
                dict_Edia[i_spot].append(tt2[0])
            else:
                dict_Edia[i_spot].append(tt2)
            i_spot += 1

    f.close()

    if read_calibdia:
        return (dict_Edia, dict_values_names, thf_list, calibdia)
    else:
        return (dict_Edia, dict_values_names, thf_list)


# ----    -----  plot_Edia_vs_thf
def plot_Edia_vs_thf(fileEdia, Emin=5.0, Emax=22.0, thf_in_scan=None, ndia_max=None, dylim=0.1, list_ndia=None,
                    filespotlist_sample=None,
                    nsample_max=None,
                    filediplinks=None,  # not used any more
                    img_in_scan=None,
                    dict_dips_exp=None,  # not used any more
                    figfile_prefix=None,
                    savefig=0,
                    show_ndia=0,
                    show_lines=0,
                    figsize=(10, 6),
                    ExpAbacus=None,
                    read_calibdia="new",
                    verbose=0,
                    newfigureandax=None):
    """
    read and plot abacus File of Energy vs thf for several Diamond branches

    return (diabranchesdata, fig, figfilename)
    """
    p.rcParams["figure.subplot.right"] = 0.85
    p.rcParams["figure.subplot.left"] = 0.15
    p.rcParams["figure.subplot.top"] = 0.85
    p.rcParams["figure.subplot.bottom"] = 0.15

    #    p.rcParams['font.size']=30     # article
    p.rcParams["font.size"] = 20  # article

    if thf_in_scan is not None:
        thf_min = thf_in_scan[0]
        thf_max = thf_in_scan[-1]

    res = read_dict_Edia_vs_thf(fileEdia, read_calibdia=read_calibdia, verbose=verbose)

    if read_calibdia is not "new":
        dict_Edia, dict_values_names, thf_list = res
    else:
        dict_Edia, dict_values_names, thf_list, _ = res

    ndict = len(dict_values_names)

    for j_dict in range(ndict):
        print(dict_values_names[j_dict])

    ind_hkldia = dict_values_names.index("hkldia")
    ind_Edia = dict_values_names.index("Edia_list")
    ind_fpol = dict_values_names.index("fpol_list")
    ind_intensity = dict_values_names.index("intensity_norm")

    print('ind_intensity', ind_intensity)

    nspots_dia = len(list(dict_Edia.keys()))
    print("nspots_dia = ", nspots_dia)

    if ndia_max is not None:
        nspots_dia = ndia_max

    # take polarization factor at thf_ref = center of scan
    # or at point in thf_list next just below thf_ref

    if list_ndia is None:
        list_ndia = list(range(nspots_dia))
    else:
        list_ndia = np.array(list_ndia, dtype=int)
        nspots_dia = len(list_ndia)

    # DISPLAY

    if newfigureandax is None:
        fig = p.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig, ax = newfigureandax
    ax.grid(True)
    yy = thf_list

    CHECK_FPOL = 0

    #    # color coding d'apres sf*sf*fpol*fpol
    color_list = ["r", "m", "b", "g", "y"]
    intensity_threshold_list = np.array([101.0, 75.0, 10.0, 3.0, 0.5, 0.0])

    # loop over branch index
    diabranchesdata = []
    for i_branch in list_ndia:
        #        print "i = ", i
        fpol_list = np.array(dict_Edia[i_branch][ind_fpol], dtype=float)
        Edia_list = np.array(dict_Edia[i_branch][ind_Edia], dtype=float) / 1000.0
        ind0 = np.where(Edia_list > 0.1)
        #        print "ind0 = ", ind0
        if len(ind0[0]) > 0:
            normdip = float(dict_Edia[i_branch][ind_intensity])
            int_list = normdip * np.multiply(fpol_list[ind0[0]], fpol_list[ind0[0]])
            #        p.plot(thf_list[ind0[0]], int_list, "ro", ms = 5)
            list_color = ["y"] * len(int_list)
            #        print list_color
            for j in range(5):
                #            print color_list[j]
                #            print intensity_threshold_list[j+1], intensity_threshold_list[j]
                ind1 = np.where(
                    (int_list > intensity_threshold_list[j + 1]) & (int_list < intensity_threshold_list[j]))
                if len(ind1[0]) > 0:
                    #                print int_list[ind1[0]]
                    #                print ind1[0]
                    for k in range(len(ind1[0])):
                        list_color[ind1[0][k]] = color_list[j]
            #        print list_color
            ind2 = np.argmax(int_list)

            if CHECK_FPOL:
                ax.scatter(thf_list[ind0[0]], int_list, marker="o", c=list_color, s=100)

            else:
                ax.scatter(Edia_list[ind0[0]], thf_list[ind0[0]], marker="o", c=list_color, s=50)
                if show_lines:
                    _, = ax.plot(Edia_list[ind0[0]], thf_list[ind0[0]], "-",
                                                                        color=list_color[ind2],
                                                                        label=str(i_branch),
                                                                        picker=0.5)

                hkl = np.array(dict_Edia[i_branch][ind_hkldia].round(decimals=0), dtype=int)
                _ = str(hkl[0]) + str(hkl[1]) + str(hkl[2]) + " #" + str(i_branch)

                yy = thf_list[ind0[0]]
                xx = Edia_list[ind0[0]]

                # p.text(xx[-1],2.7, label1, rotation = 'vertical', color = color1, fontsize = 20)
                # p.text(xx1[-1], 2.0, label1, rotation = 'vertical', verticalalignment='bottom', color = colordia[i] , fontsize = 16)
                # add label for diamond lines
                #
                # ind0 = np.where(yy1> ymin1)
                # print yy1
                #            ind1 = np.where( yy <= thf_min)  #old
                ind1 = np.where((yy > thf_min) & (yy < thf_max))
                # print yy1
                # print ind0

                if show_ndia:
                    if len(ind1[0]) > 0:
                        xx1 = xx[ind1[0]]
                        pos1 = np.argmax(xx1)
                        #                    pos1 = ind1[0][-1]  #old
                        # print pos1, xx1[pos1],yy1[pos1]
                        ax.plot(xx[pos1], yy[pos1], "s", ms=20, mec="k", mfc="w", mew=3)
                        ax.text(xx[pos1], yy[pos1], str(i_branch), fontsize=16, color="k",
                                                            horizontalalignment="center",
                                                            verticalalignment="center")
                else:
                    p.title(str(list_ndia), fontsize=8)

            if len(ind0[0]) > 1:
                diabranchesdata.append(np.concatenate((xx, yy, i_branch * np.ones(len(xx)))))

    if CHECK_FPOL:
        ax.set_xlabel("thf (deg)")
        ax.set_ylabel("sf*sf*fpol*fpol")
        ax.set_title(str(list_ndia))

    else:
        ax.set_xlabel("diffracted photon energy (keV)")
        ax.set_ylabel("filter angle (degrees)")
        ax.set_xlim(Emin, Emax)
        # p.ylim(-47.5,-42.5) # Mar13
        ax.set_ylim(thf_min - dylim, thf_max + dylim)

    if filespotlist_sample is not None:

        print(filespotlist_sample)
        data_sample = np.loadtxt(filespotlist_sample, skiprows=5)
        print("numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9")
        _ = np.shape(data_sample)[0]
        _ = np.array(data_sample[:, 0], dtype=int)

        if nsample_max is not None:
            _ = nsample_max
        _ = data_sample[:, 9] / 1000.0

        # if (dict_dips_exp is None) & (filediplinks is None):
        #     for i_spot in range(nspots_sample):
        #         ax.axvline(x=Etheor_sample[i_spot], color="k")
        #         label1 = str(i_spot) + "," + str(numdat[ii_spot])
        #         ax.text(Etheor_sample[i_spot],
        #             thf_max + 0.1,
        #             label1,
        #             rotation="vertical",
        #             verticalalignment="bottom",
        #             fontsize=16)

        #     print("vertical lines labelled by numHKL, numdat")
        #     print("several numHKL per numdat if harmonics")

        ax.set_xlim(Emin, Emax)
        ax.set_ylim(thf_min - dylim, thf_max + dylim)

    if ExpAbacus is not None:
        selectedDips = []
        AB = np.array(ExpAbacus)
        y_thf = AB[:, 2]
        x_energy = AB[:, 3] / 1000.0
        s = AB[:, 4] * 4000
        harmoniclevel = AB[:, 10]

        ax.scatter(x_energy, y_thf, s, marker="*", c=harmoniclevel, edgecolor="None", picker=3)
        ax.set_xlim(5, 23)
        ax.set_ylim(-52, -45)

        # text=ax.text(0,0, -41., " ", va="bottom", ha="left",fontsize = 12,transform=ax.transAxes)

        def onclick(event):

            xdata = x_energy
            ydata = y_thf
            _ = 5
            d = np.sqrt((xdata - event.xdata) ** 2.0 + (ydata - event.ydata) ** 2.0)
            #             ind = np.nonzero(np.less_equal(d, maxd))
            ind = np.argmin(d)

            spotindex, profileindex, thf, Energy = ExpAbacus[ind][:4]
            h, k, l, X, Y = ExpAbacus[ind][5:10]

            selectedDips.append([spotindex, profileindex, thf, Energy, h, k, l, X, Y])
            ty = "#spot: %d #profile: %d  thf: %.2f E: %.0f (%d,%d,%d) X,Y=[%d, %d]" % (
                                                        spotindex,
                                                        profileindex,
                                                        thf,
                                                        Energy,
                                                        h,
                                                        k,
                                                        l,
                                                        X,
                                                        Y)

            #     text.set_text(ty)
            ax.set_title(ty, fontsize=13)

        _ = fig.canvas.mpl_connect("button_press_event", onclick)

    figfilename = ""
    if savefig:
        #         figfilename = figfile_prefix + "_Edia_vs_thf.png"
        figfilename = "Abacus" + "_Edia_vs_thf.png"
        p.savefig(figfilename, bbox_inches="tight")
        print("Abacus saved in", figfilename)

    return (diabranchesdata, fig, figfilename)


def save_dict_dips_theor(outputfilename, dict_dips_theor, fileEdia, file_spolistref_sample,
                            include_exp="no",
                            filediplinks=None):
    # 0 numdat
    # 1 HKL_sample
    # 2 xypix_xyfit
    # 3 Ipixmax
    # 4 Etheor_sample
    # 5 uqlab_sample
    # 6 ndia_list
    # 7 HKL_dia_list_theor
    # 8 thf_list_theor
    # 9 dip_depth_theor
    # 10 slope_thf_vs_Edia
    # 11 uqlab_dia_0
    # 12 uqlab_dia_1
    # 13 uqlab_dia_2
    # 14 inner_uq_dia_xz_uq_sample_xz
    # 15 uq_dia_z

    dict_values_names = ["numdat",
                        "HKL_sample",
                        "xypix_xyfit",
                        "Ipixmax",
                        "Etheor_sample",
                        "uqlab_sample",
                        "ndia_list",
                        "HKL_dia_list_theor",
                        "thf_list_theor",
                        "dip_depth_theor",
                        "slope_thf_vs_Edia",
                        "uqlab_dia_0",
                        "uqlab_dia_1",
                        "uqlab_dia_2",
                        "inner_uq_dia_xz_uq_sample_xz",
                        "uq_dia_z"]

    #    dict_values_names = ["numdat","HKL_sample", "xypix_xyfit", "Ipixmax", "Etheor_sample",  \
    #                          "ndia_list", "HKL_dia_list_theor", "thf_list_theor", "dip_depth_theor", \
    #                          "slope_thf_vs_Edia"]

    if include_exp == "yes":
        dict_values_names = dict_values_names + ["img_exp",
                                                "thf_exp",
                                                "dEsE_interpol",
                                                "dEsE_recalc_withcorr"]

    header = "#Theoretical dips \n"
    header += "#Edia vs thf dictionnary file = %s \n" % (fileEdia)
    header += "#sample spotlistref file = %s \n" % (file_spolistref_sample)

    if include_exp == "yes":
        header += "#dip links file = %s \n" % (filediplinks)
    header += "#File created at %s with diamond.py \n" % (time.asctime())
    outputfile = open(outputfilename, "w")
    outputfile.write(header)
    # key = dia spot num

    ndict = len(dict_values_names)

    for i in range(ndict):
        outputfile.write(dict_values_names[i] + "\n")
        print(i, dict_values_names[i])
        for key, value in dict_dips_theor.items():
            #            print i, key
            #            if i == 11 : print len(value[i])
            if i in [0, 3, 4]:
                if include_exp == "yes":
                    str1 = str(value[i][0])
                else:
                    str1 = str(value[i])
            else:
                str1 = " ".join(str(e) for e in value[i])
            outputfile.write(str(key) + " : " + str1 + "\n")

    outputfile.close()

    return outputfilename


# ---  -----  build_dict_dips_theor
def build_dict_dips_theor(filespotlist_sample,
                                fileEdia,
                                outfile_prefix,
                                verbose=0,
                                accurate_calc_for_thf_dip=1,
                                read_calibdia="new",  # only for accurate_calc_for_thf_dip = 1
                            ):
    """Build theoretical dips position dict 'dict_dips_theor' from indexed peak list
    `filespotlist_sample` and save in ascii file 'file_dips_theor'

    :param filespotlist_sample: full path to ascii file of spots properties generated by build_spotlistref_sample
    :type filespotlist_sample: str
    :param fileEdia: full path to ascii file of dictionary(ies?) energy-thf abacus
    :type fileEdia: str
    :param outfile_prefix: prefix for outputfile name in including folder path.
                It is recommended to use the same prefix than the thf-scan images file name
    :type outfile_prefix: str
    :param verbose: [description], defaults to 0
    :type verbose: int, optional
    :param accurate_calc_for_thf_dip: flag to perform accurate computation, defaults to 1
    :type accurate_calc_for_thf_dip: int, optional
    :param read_calibdia:   'old' or 'new' to account for calibration Diamond monochromator setup parameters, defaults to "new" when using `accurate_calc_for_thf_dip=1`
    :type read_calibdia: str, optional
    :return:
        - dict_dips_theor
        - dict_values_names2 (list of str corresponding to element of dict_dips_theor values)
        - outputfilename
    :rtype: dict, dict, str

    .. todo:: improve readibility
    """

    # 24Jan17 : full computation for thf_dip
    # instead of piecewise linear Abacus

    # increment for taylor expansion of first order of Abacus
    # used for accurate computation (when accurate_calc_for_thf_dip = 1)
    dthf = 0.01  # deg

    if accurate_calc_for_thf_dip:
        (dict_Edia,
        dict_values_names, thf_list, calibdia) = read_dict_Edia_vs_thf(fileEdia, read_calibdia=read_calibdia)
    else:
        dict_Edia, dict_values_names, thf_list = read_dict_Edia_vs_thf(fileEdia,
                                                                        read_calibdia=read_calibdia)

    ndict = len(dict_values_names)

    #    print dict_values_names[6:9]

    if verbose:
        print(filespotlist_sample)

    data_sample = np.loadtxt(filespotlist_sample, skiprows=5)
    # if verbose : print "numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9, uqlab 10:13"
    nspots_sample = np.shape(data_sample)[0]
    # nspots_sample = 3
    #    Etheor_sample = data_sample[:,9]/1000.

    #    p.figure(figsize = (8,8))
    #    p.plot(data_sample[:,1],-data_sample[:,2],'ko')
    #    p.xlim(0.,2048.)
    #    p.ylim(-2048., 0.)
    #    jdqlqs

    # key = sample spot num in spotlisref , may be different from numdat
    dict_dips_theor = {}

    dict_values_names2 = ["numdat", "HKL_sample", "xypix_xyfit", "Ipixmax", "Etheor_sample",
                        "uqlab_sample", "ndia_list", "HKL_dia_list_theor", "thf_list_theor",
                        "dip_depth_theor", "slope_thf_vs_Edia", "uqlab_dia_0",
                        "uqlab_dia_1", "uqlab_dia_2", "inner_uq_dia_xz_uq_sample_xz", "uq_dia_z"]

    ndict2 = len(dict_values_names2)

    # nspots_sample = 3

    for i_spot in range(nspots_sample):
        dict_dips_theor[i_spot] = [int(np.round(data_sample[i_spot, 0], 0)),
                            np.array(data_sample[i_spot, 5:8].round(decimals=0), dtype=int),
                            data_sample[i_spot, 1:5],
                            int(np.round(data_sample[i_spot, 8], 0)),
                            float(np.round(data_sample[i_spot, 9], 2)),
                            data_sample[i_spot, 10:13].round(decimals=3)]

    if verbose:
        for key, value in dict_dips_theor.items():
            print(key, value)
        # print("dict_Edia")
        for j_dict in range(ndict):
            print(j_dict, dict_values_names[j_dict])
        # print("\n")
        # print("dict_dips_theor")
        for j_dict2 in range(ndict2):
            print(j_dict2, dict_values_names2[j_dict2])

    nspotsdia = len(list(dict_Edia.keys()))
    # nspotsdia = 20

    #    0 hkldia
    #    1 intensity_norm
    #    2 Edia_list
    #    3 fpol_list
    #    4 Edia_mean_min_max
    #    5 thf_mean_min_max

    #   depuis 10Feb14

    #    6 uqlab_0
    #    7 uqlab_1
    #    8 uqlab_2

    # reperage des courbes Edia vs thf avec changement de pente

    is_double_valued = np.zeros(nspotsdia, int)
    double_range = np.zeros((nspotsdia, 2), float)
    argmin_short = np.zeros(nspotsdia, int)
    for j in range(nspotsdia):
        # print "j = ", j
        Edia_list = dict_Edia[j][2]
        ind1 = np.where(Edia_list > 0.1)
        # print('Edia_list',Edia_list)
        Edia_list2 = Edia_list[ind1[0]]
        if len(ind1[0]) > 0:
            ind2 = np.argmin(Edia_list2)
            indmax = len(Edia_list2) - 1
            # print "argmin, indmax = ", ind2, indmax
            tt = (Edia_list2[0], Edia_list2[-1], Edia_list2.min(), Edia_list2.max())
            tt1 = np.sort(tt)
            if (ind2 != 0) & (ind2 != indmax):
                # print "risk of two dips from same HKLdia : "
                is_double_valued[j] = 1
                double_range[j, :] = tt1[:2]
                argmin_short[j] = ind2
                # print "double_range , argmin_short = ", double_range[j,:], argmin_short[j]
    # print "is_double_valued = ", is_double_valued
    if verbose:
        print("nb of diamond lines : total / with double_valued thf vs Edia")
    ind1 = np.where(is_double_valued == 1)
    if verbose:
        print(nspotsdia, len(ind1[0]))

    xz_lab = np.array([1.0, 0.0, 1.0])
    for i in range(nspots_sample):
        Esample = data_sample[i, 9]
        print("i (spotindex)= ", i, "Etheor_sample = ", Esample)
        _ = 0
        ndip = 0
        ndia_list = []
        hkldia_list = []
        thf_dip_list = []
        dip_depth_list = []
        slope_list = []
        uqlab0_list = []
        uqlab1_list = []
        uqlab2_list = []
        inner_list = []
        inner2_list = []

        uq_sample_xz = np.multiply(data_sample[i, 10:13], xz_lab)
        uq_sample_xz = uq_sample_xz / MG.norme(uq_sample_xz)

        for j in range(nspotsdia):
            Ediamin = float(dict_Edia[j][4][1])
            Ediamax = float(dict_Edia[j][4][2])
            HKLdia = np.array(dict_Edia[j][0].round(decimals=0), dtype=int)
            #print("nsample, ndia, Ediamin, Ediamax", i, j, Ediamin, Ediamax)
            if (Esample < Ediamax) & (Esample > Ediamin):
                #                dip_found = 1
                # print "possible dip"
                Edia_list = dict_Edia[j][2]
                ind1 = np.where(Edia_list > 0.1)
                # print ind1[0]
                # if j == 23 : kljkldsa
                if len(ind1[0]) > 0:
                    #print("len(ind1[0]) =", len(ind1[0]))
                    Edia_list2 = Edia_list[ind1[0]]
                    thf_list2 = thf_list[ind1[0]]
                    fpol_list2 = dict_Edia[j][3][ind1[0]]

                    uqlab0_list2 = dict_Edia[j][6][ind1[0]]
                    uqlab1_list2 = dict_Edia[j][7][ind1[0]]
                    uqlab2_list2 = dict_Edia[j][8][ind1[0]]

                    xylist = np.column_stack((Edia_list2, thf_list2, fpol_list2,
                                            uqlab0_list2, uqlab1_list2, uqlab2_list2))

                    if ((is_double_valued[j] == 1)
                        & (Esample > double_range[j, 0])
                        & (Esample < double_range[j, 1])):
                        _ = 1
                        # print "two dips"
                        # print "double_range = ", double_range[j]
                        # print "Esample = ", Esample
                        npts1 = np.shape(xylist)[0]
                        # print "npts1 =", npts1
                        # print "argmin_short =", argmin_short[j]
                        # print Edia_list2[0], Edia_list2[1],Edia_list2[argmin_short[j]], Edia_list2[-2],Edia_list2[-1]
                        range1 = list(range(0, argmin_short[j] + 1))
                        # print "range1 = ", range1
                        xylist1 = xylist[range1]
                        newy1, slope1 = interpol_new(xylist1, Esample)  # thf_dip, fpol_dip, dthf/dEdia, dfpol/dEdia
                        dip_depth1 = dict_Edia[j][1] * newy1[1] * newy1[1]  # sf*sf*fpol*fpol

                        uqlab_dia = np.array([newy1[2], newy1[3], newy1[4]], dtype=float)
                        #                        inner1 = np.inner(uqlab_dia, data_sample[i,10:13])
                        #                        inner2 = (np.cross(uqlab_dia, data_sample[i,10:13]))[0]
                        uqlab_dia_xz = np.multiply(uqlab_dia, xz_lab)
                        uqlab_dia_xz = uqlab_dia_xz / MG.norme(uqlab_dia_xz)
                        inner1 = np.inner(uqlab_dia_xz, uq_sample_xz)
                        inner2 = uqlab_dia[2]

                        ndia_list.append(j)
                        hkldia_list.append(HKLdia)

                        # 24Jan17
                        thf_dip1 = newy1[0]
                        if accurate_calc_for_thf_dip:
                            # analytical inverse function for  thf_dip_to_lambda_dia_v2  not yet available !
                            lambda_dia_nm = thf_dip_to_lambda_dia_v2(HKLdia, thf_dip1, calibdia)
                            Edia2_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
                            thf_dip3 = thf_dip1 + dthf
                            lambda_dia_nm = thf_dip_to_lambda_dia_v2(HKLdia, thf_dip3, calibdia)
                            Edia3_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
                            thf_dip1b = thf_dip1 + dthf * (Esample - Edia2_eV) / (Edia3_eV - Edia2_eV)

                            thf_dip_list.append(np.round(thf_dip1b, 4))
                        else:
                            thf_dip_list.append(np.round(thf_dip1, 4))

                        #                        thf_dip_list.append(np.round(newy1[0],4))

                        dip_depth_list.append(np.round(dip_depth1, 4))
                        slope_list.append(np.round(slope1[0] * 1000.0, 2))
                        uqlab0_list.append(np.round(newy1[2], 3))
                        uqlab1_list.append(np.round(newy1[3], 3))
                        uqlab2_list.append(np.round(newy1[4], 3))
                        inner_list.append(np.round(inner1, 3))
                        inner2_list.append(np.round(inner2, 3))

                        assert len(ndia_list) <= 10000

                        range2 = list(range(argmin_short[j], npts1))
                        # print "range2 =", range2
                        xylist2 = xylist[range2]
                        newy2, slope2 = interpol_new(xylist2, Esample)
                        # print('dict_Edia[j][1]',dict_Edia[j][1])
                        dip_depth2 = dict_Edia[j][1] * newy2[1] * newy2[1]

                        #                        uqlab_dia = np.array([newy2[2],newy2[3],newy2[4]], dtype = float)
                        #                        inner1 = np.inner(uqlab_dia, data_sample[i,10:13])
                        #                        inner2 = (np.cross(uqlab_dia, data_sample[i,10:13]))[0]
                        uqlab_dia_xz = np.multiply(uqlab_dia, xz_lab)
                        uqlab_dia_xz = uqlab_dia_xz / MG.norme(uqlab_dia_xz)
                        inner1 = np.inner(uqlab_dia_xz, uq_sample_xz)
                        inner2 = uqlab_dia[2]

                        ndia_list.append(j)
                        hkldia_list.append(HKLdia)

                        # 24Jan17
                        thf_dip1 = newy2[0]
                        if accurate_calc_for_thf_dip:
                            lambda_dia_nm = thf_dip_to_lambda_dia_v2(HKLdia, thf_dip1, calibdia)
                            Edia2_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
                            thf_dip3 = thf_dip1 + dthf
                            lambda_dia_nm = thf_dip_to_lambda_dia_v2(HKLdia, thf_dip3, calibdia)
                            Edia3_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
                            thf_dip1b = thf_dip1 + dthf * (Esample - Edia2_eV) / (Edia3_eV - Edia2_eV)

                            thf_dip_list.append(np.round(thf_dip1b, 4))
                        else:
                            thf_dip_list.append(np.round(thf_dip1, 4))

                        dip_depth_list.append(np.round(dip_depth2, 4))
                        slope_list.append(np.round(slope2[0] * 1000.0, 2))
                        uqlab0_list.append(np.round(newy2[2], 3))
                        uqlab1_list.append(np.round(newy2[3], 3))
                        uqlab2_list.append(np.round(newy2[4], 3))
                        inner_list.append(np.round(inner1, 3))
                        inner2_list.append(np.round(inner2, 3))

                        assert len(ndia_list) <= 10000
                        ndip = ndip + 2

                    else:
                        # print "one dip"
                        _ = 0
                        newy1, slope1 = interpol_new(xylist, Esample)
                        dip_depth1 = dict_Edia[j][1] * newy1[1] * newy1[1]

                        uqlab_dia = np.array([newy1[2], newy1[3], newy1[4]], dtype=float)
                        #                        inner1 = np.inner(uqlab_dia, data_sample[i,10:13])
                        #                        inner2 = (np.cross(uqlab_dia, data_sample[i,10:13]))[0]
                        uqlab_dia_xz = np.multiply(uqlab_dia, xz_lab)
                        uqlab_dia_xz = uqlab_dia_xz / MG.norme(uqlab_dia_xz)
                        inner1 = np.inner(uqlab_dia_xz, uq_sample_xz)
                        inner2 = uqlab_dia[2]

                        ndia_list.append(j)
                        hkldia_list.append(HKLdia)

                        # 24Jan17
                        thf_dip1 = newy1[0]
                        if accurate_calc_for_thf_dip:
                            lambda_dia_nm = thf_dip_to_lambda_dia_v2(HKLdia, thf_dip1, calibdia)
                            Edia2_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
                            thf_dip3 = thf_dip1 + dthf
                            lambda_dia_nm = thf_dip_to_lambda_dia_v2(HKLdia, thf_dip3, calibdia)
                            Edia3_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
                            thf_dip1b = thf_dip1 + dthf * (Esample - Edia2_eV) / (Edia3_eV - Edia2_eV)

                            thf_dip_list.append(np.round(thf_dip1b, 4))
                        else:
                            thf_dip_list.append(np.round(thf_dip1, 4))

                        #                        thf_dip_list.append(round(newy1[0],4))
                        dip_depth_list.append(np.round(dip_depth1, 4))
                        slope_list.append(np.round(slope1[0] * 1000.0, 2))
                        uqlab0_list.append(np.round(newy1[2], 4))
                        uqlab1_list.append(np.round(newy1[3], 4))
                        uqlab2_list.append(np.round(newy1[4], 4))
                        inner_list.append(np.round(inner1, 3))
                        inner2_list.append(np.round(inner2, 3))

                        assert len(ndia_list) <= 10000

                        ndip = ndip + 1

        if ndip == 0:
            dict_dips_theor[i].append([-1])
            dict_dips_theor[i].append([0])  # HKLdia
            dict_dips_theor[i].append([0.0])  # thf_dip
            dict_dips_theor[i].append([0.0])
            dict_dips_theor[i].append([0.0])  # slope dthf/dEdia deg/keV
            dict_dips_theor[i].append([0.0])
            dict_dips_theor[i].append([0.0])
            dict_dips_theor[i].append([0.0])
            dict_dips_theor[i].append([0.0])
            dict_dips_theor[i].append([0.0])
        else:
            dict_dips_theor[i].append(ndia_list)  # ndia
            dict_dips_theor[i].append(np.array(hkldia_list))  # HKLdia
            dict_dips_theor[i].append(thf_dip_list)  # thf_dip
            dict_dips_theor[i].append(dip_depth_list)
            dict_dips_theor[i].append(slope_list)  # slope dthf/dEdia deg/keV
            dict_dips_theor[i].append(uqlab0_list)
            dict_dips_theor[i].append(uqlab1_list)
            dict_dips_theor[i].append(uqlab2_list)
            dict_dips_theor[i].append(inner_list)
            dict_dips_theor[i].append(inner2_list)

        print("i =", i, "ndip = ", ndip)

    #    0 numdat
    #    1 HKL_sample
    #    2 xypix_xyfit
    #    3 Ipixmax
    #    4 Etheor_sample
    #    5 ndia_list
    #    6 HKL_dia_list_theor
    #    7 thf_list_theor
    #    8 dip_depth_theor
    #    9 slope_thf_vs_Edia

    for i in range(ndict2):
        print(dict_values_names2[i])
    #        for key, value in dict_dips_theor.iteritems():
    #            print key,value[i]
    #            if key > 10 : break

    #    for key, value in dict_dips_theor.iteritems():
    #        print "\n", key, ":",
    #        for i in range(5,ndict2):
    #            print len(value[i]),

    outputfilename = outfile_prefix + "_dict_dips_theor_with_harmonics.dat"
    print("dict_dips_theor saved in : ", outputfilename)

    save_dict_dips_theor(outputfilename, dict_dips_theor, fileEdia, filespotlist_sample)

    filenamepckl = outputfilename[:-4]+'.pckl'
    data_pickled = dict_dips_theor

    with open(filenamepckl, "wb") as f:
        pickle.dump(data_pickled, f)

    return dict_dips_theor, dict_values_names2, outputfilename


def read_dict_dips_theor(filedipstheor, include_exp="no", verbose=1):
    """
    read theoretical dips dictionnary

    # TODO: to repair
    """
    #    listint = [0,1,3,5,10]
    #    listfloat = [2,4,7,8,9,11,12,13]
    #    listmix = [6,]

    # numdat
    # HKL_sample
    # xypix_xyfit
    # Ipixmax
    # Etheor_sample
    # uqlab_sample
    # ndia_list
    # HKL_dia_list_theor
    # thf_list_theor
    # dip_depth_theor
    # slope_thf_vs_Edia
    # uqlab_dia_0
    # uqlab_dia_1
    # uqlab_dia_2
    # inner_uq_dia_xz_uq_sample_xz
    # uq_dia_z

    #    10 img_exp
    #    11 thf_exp
    #    12 dEsE_interpol
    #    13 dEsE_recalc_withcorr
    dict_data_type = {0: int, 1: float, 2: "mixint", 3: "mixfloat"}

    dict_values_names = ["numdat", "HKL_sample", "xypix_xyfit", "Ipixmax", "Etheor_sample",
                        "uqlab_sample", "ndia_list", "HKL_dia_list_theor", "thf_list_theor",
                        "dip_depth_theor", "slope_thf_vs_Edia", "uqlab_dia_0", "uqlab_dia_1",
                        "uqlab_dia_2", "inner_uq_dia_xz_uq_sample_xz", "uq_dia_z"]

    ind_hkldia = dict_values_names.index("HKL_dia_list_theor")

    # dict of data type for echa dict
    # 0 : int, 1 : float, 2 : mixint
    dict_type = [0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1]

    if include_exp == "yes":
        dict_values_names = dict_values_names + ["img_exp",
                                                    "thf_exp",
                                                    "dEsE_interpol",
                                                    "dEsE_recalc_withcorr"]
        # TODO: shoud have 4 elements !
        dict_type2 = [1, 1, 1]
        dict_type = dict_type + dict_type2

    ndict = len(dict_values_names)

    print("variable , type")
    for j_dict in range(ndict):
        print(j_dict, dict_values_names[j_dict])  # , dict_type[i]

    linepos_list = np.zeros(ndict + 1, dtype=int)

    f = open(filedipstheor, "r")

    endlinechar = '\n\r'

    try:
        i = 0
        for line in f:
            # print line.rstrip(endlinechar)
            for j_dict in range(ndict):
                if line.startswith(dict_values_names[j_dict]):
                    linepos_list[j_dict] = i
            i += 1
    finally:
        f.close()

    linepos_list[-1] = i

    nspots = linepos_list[1] - linepos_list[0] - 1

    if verbose:
        print("linepos_list = ", linepos_list)
        print("nspots = ", nspots)

    f = open(filedipstheor, "r")
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)

    f.seek(0)
    print(f.readline())

    #--  filling main dict for theoretical dips properties which contains subdicts
    dict_dips_theor = {}

    # reading procedure for all dicts
    for j_dict in range(ndict):
        data_type = dict_data_type[dict_type[j_dict]]
        #        print j_dict
        #        print dict_values_names[j_dict]
        #        print data_type
        n = linepos_list[j_dict]
        if verbose:
            print("line postion of field n = ", n)
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        if verbose:
            print(" ", f.readline())
        f.seek(line_offset[n + 1])

        # loop over sub dicts of dict_dips_theor
        i_spot = 0
        while i_spot < nspots:
            # if j_dict == 5 : print "spot = ", i
            tt = f.readline()
            tt1 = (tt.rstrip(endlinechar).split(": "))[1]

            # print('i_spot',i_spot)
            # print('tt',tt)
            if data_type != "mixint":
                #patch   for str = [val1] [val2]
                if j_dict in (9, ): # 9==dip_depth_theor
                    tt2 = np.array(tt1.replace("[", "").replace("]", "").split(), dtype=float)
                else:
                    tt2 = np.array(tt1.split(), dtype=data_type)
            else:
                tt2 = np.array(tt1.replace("[", "").replace("]", "").split(), dtype=int)
            # print tt2
            if j_dict in (0,):
                # dict value is a list with the first value filled
                dict_dips_theor[i_spot] = [tt2]
            elif j_dict in (3, 4):
                # dict_dips_theor[i] = [tt2]
                dict_dips_theor[i_spot].append(tt2)
            else:
                dict_dips_theor[i_spot].append(tt2)
            i_spot += 1

    f.close()

    # reshape du tableau de HKLdia_list
    for k in list(dict_dips_theor.keys()):
        HKLdia_a = np.array(dict_dips_theor[k][ind_hkldia], dtype=float)
        if len(HKLdia_a) > 1:
            nhkl = len(HKLdia_a) // 3
            # print nhkl
            dict_dips_theor[k][ind_hkldia] = HKLdia_a.reshape(nhkl, 3)

    return (dict_dips_theor, dict_values_names)


def dcalibdia_or_dthf_to_dE_surE(hkldia,
                                calibdia,
                                ncorr=0,  # numero de la composante de geometry_corrections qui varie
                                dcorr=0.0,  # amplitude de la correction en unites 1e-4
                                thf=-50.0,  # degres
                                dthf=0.0,  # degres
                                verbose=0):

    mat_range = list(range(0, 9))
    vlab_range = list(range(9, 12))
    uilab_range = list(range(12, 15))
    thfref_range = 15

    vlab_start = calibdia[vlab_range]
    thf_ref = calibdia[thfref_range]
    matref_line = calibdia[mat_range]
    matref_center = GT.matline_to_mat3x3(matref_line)
    uilab_start = calibdia[uilab_range]

    if verbose:
        print("thf_ref = ", thf_ref)

    geometry_corrections = np.zeros(4, float)

    geometry_corrections[ncorr] = dcorr

    if verbose:
        print("geometry_corrections = ", geometry_corrections)

    ui_pitch = geometry_corrections[0]
    ui_yaw = geometry_corrections[1]
    axis_yaw = geometry_corrections[2]
    axis_roll = geometry_corrections[3]

    vlab_corr = vlab_start * 1.0

    if ncorr in [2, 3]:
        aya = axis_yaw * 1.0e-4
        aro = axis_roll * 1.0e-4
        vlab = vlab_start + np.array([0.0, aya, aro])
        vlab_corr = vlab / MG.norme(vlab)
        if verbose:
            print("vlab_corr = ", vlab_corr)

    uilab_corr = uilab_start * 1.0

    if ncorr in [0, 1]:
        # dirv dirh : 1e-4 rad
        dirv = ui_pitch * 1.0e-4
        dirh = ui_yaw * 1.0e-4
        uilab = uilab_start + np.array([dirh, 0.0, dirv])
        uilab_corr = uilab / MG.norme(uilab)
        if verbose:
            print("uilab_corr = ", uilab_corr)

    if verbose:
        print("thf =", thf)
    if verbose:
        print("dthf = ", dthf)

    matrot1 = MG.from_axis_vecangle_to_mat(vlab_start, thf_ref - thf)  # v1
    matnew1 = np.dot(matrot1, matref_center)
    mat1 = GT.mat3x3_to_matline(matnew1)
    if verbose:
        print("matrix 1 : \n", mat1)

    matrot2 = MG.from_axis_vecangle_to_mat(vlab_corr, thf_ref - thf - dthf)  # v1
    matnew2 = np.dot(matrot2, matref_center)
    mat2 = GT.mat3x3_to_matline(matnew2)
    if verbose:
        print("matrix 2 : \n", mat2)

    H = hkldia[0]
    K = hkldia[1]
    L = hkldia[2]
    qlab1 = H * mat1[0:3] + K * mat1[3:6] + L * mat1[6:]
    uqlab1 = qlab1 / MG.norme(qlab1)
    # print "uqlab = ", uqlab
    sintheta1 = -1.0 * np.inner(uqlab1, uilab_start)

    qlab2 = H * mat2[0:3] + K * mat2[3:6] + L * mat2[6:]
    uqlab2 = qlab2 / MG.norme(qlab2)
    # print "uqlab = ", uqlab
    sintheta2 = -1.0*np.inner(uqlab2, uilab_corr)

    dE_sur_E = -(sintheta2 - sintheta1) / sintheta1 * 1.0e4

    if verbose:
        print("dE_sur_E (1e-4 units)= ", dE_sur_E)

    return dE_sur_E


# --- ------ Calibration refinement procedure ------------
def calc_dEsE_list_v2(filediplinks,
                        filedipstheor,
                        fileEdia,
                        img_in_scan,
                        thf_in_scan,
                        calibdia,
                        Emin=5.0,
                        Emax=22.0,
                        filepathout=None,
                        save_new_dict_dips_theor=0,
                        filespotlist_sample=None,
                        confidence_min=1):

    mat_range = list(range(0, 9))
    vlab_range = list(range(9, 12))
    uilab_range = list(range(12, 15))
    thfref_range = 15

    uilab = calibdia[uilab_range]
    vlab = calibdia[vlab_range]
    _ = calibdia[mat_range]
    thf_ref = calibdia[thfref_range]

    #    xlab = np.array([1.,0.,0.])

    # print "k 0, numdat 1, ndia 2, img 3, confidence 4"

    print("reading filediplinks from :")
    print(filediplinks)

    diplinks = np.loadtxt(filediplinks, skiprows=1)

    diplinks = np.array(diplinks.round(decimals=0), dtype=int)
    # print diplinks

    ind0 = np.where(diplinks[:, 4] >= confidence_min)
    if len(ind0[0]) > 0:
        diplinks = diplinks[ind0[0]]
    else:
        print("no dips found with confidence >= confidence_min")
        return 0

    print(diplinks)

    ndiplinks = np.shape(diplinks)[0]

    print("reading filedipstheor from:")
    print(filedipstheor)

    dict_dips_theor, dict_values_names2 = read_dict_dips_theor(filedipstheor, verbose=0)

    #    ind_inner = dict_values_names2.index("inner_uq_dia_xz_uq_sample_xz")
    ind_thf_dip = dict_values_names2.index("thf_list_theor")
    #    ind_depth = dict_values_names2.index("dip_depth_theor")
    #    ind_hkldia = dict_values_names2.index("HKL_dia_list_theor")
    ind_ndia = dict_values_names2.index("ndia_list")
    #    ind_inner2 = dict_values_names2.index("uq_dia_z")
    ind_slope = dict_values_names2.index("slope_thf_vs_Edia")
    ind_Etheor = dict_values_names2.index("Etheor_sample")
    #    ind_xypic = dict_values_names2.index("xypix_xyfit")
    #    ind_hklsample = dict_values_names2.index("HKL_sample")

    npics = len(list(dict_dips_theor.keys()))

    dict_Edia, _, _ = read_dict_Edia_vs_thf(fileEdia, verbose=0)

    allres = np.zeros((ndiplinks, 5), float)

    nfields2 = len(dict_values_names2)

    dict_dips_theor2 = dict_dips_theor

    for k in range(npics):  # ajout des nouveaux champs dans dict_dips_theor
        ndia_list = np.array(dict_dips_theor[k][ind_ndia], dtype=int)
        ndips_theor = len(ndia_list)
        img_exp = np.ones(ndips_theor, dtype=int) * (-1)
        dict_dips_theor2[k].append(img_exp)
        thf_exp = np.zeros(ndips_theor, dtype=float)
        dict_dips_theor2[k].append(thf_exp)
        dEsE_dict = np.zeros(ndips_theor, dtype=float)
        dict_dips_theor2[k].append(dEsE_dict)
        dEsE_recalc = np.zeros(ndips_theor, dtype=float)
        dict_dips_theor2[k].append(dEsE_recalc)

    list_ndia_dips = []

    dthf_list_by_dip = np.zeros(ndiplinks, float)

    dthf_sur_dimg = (thf_in_scan[-1] - thf_in_scan[0]) / (img_in_scan[-1] - img_in_scan[0])

    for i in range(ndiplinks):
        k = diplinks[i, 0]  # numero du spot dans le dico dict_dips_theor

        img_exp = diplinks[i, 3]
        thf_exp = thf_in_scan[0] + (img_exp - img_in_scan[0]) * dthf_sur_dimg
        thf_dip_theor = np.array(dict_dips_theor[k][ind_thf_dip], dtype=float)

        ndia_exp = diplinks[i, 2]
        if ndia_exp not in list_ndia_dips:
            list_ndia_dips.append(ndia_exp)

        # print "k = ", k
        # print diplinks[i]
        ndia_list = np.array(dict_dips_theor[k][ind_ndia], dtype=int)

        # print diplinks[i,2]
        ind0 = np.where(ndia_list == ndia_exp)
        #        print "**********************************************"
        #        print ndia_list
        #        print ndia_exp
        #        print ind0[0]
        #        print thf_dip_theor[ind0[0]].round(decimals=3)
        if len(ind0[0]) == 1:
            #            print "yoho"
            goodind = ind0[0][0]
        elif len(ind0[0]) > 1:
            dthf = abs(thf_dip_theor[ind0[0]] - thf_exp)
            ind5 = np.argmin(dthf)
            goodind = ind0[0][ind5]
        #        print thf_dip_theor[goodind]

        ndips_theor = len(ndia_list)
        Etheor = dict_dips_theor[k][ind_Etheor][0]
        slope_theor = np.array(dict_dips_theor[k][ind_slope], dtype=float)
        #        print "slope_theor = ",  slope_theor

        #        dip_depth_theor = np.array(dict_dips_theor[k][ind_depth], dtype = float)
        #        img_dip_theor = img_in_scan[0] + (thf_dip_theor-thf_in_scan[0])*(img_in_scan[-1]-img_in_scan[0])/(thf_in_scan[-1] - thf_in_scan[0])
        # print "theor values : thf deg, img, dip_depth, slope deg/keV, E eV"
        # print thf_dip_theor[goodind],round(img_dip_theor[goodind],2), dip_depth_theor[goodind], slope_theor[goodind], Etheor

        dict_dips_theor2[k][nfields2][goodind] = img_exp

        dict_dips_theor2[k][nfields2 + 1][goodind] = np.round(thf_exp, 4)

        dthf_list_by_dip[i] = thf_exp - thf_ref

        dE = (thf_exp - thf_dip_theor[goodind]) / slope_theor[goodind] * 1000
        dEsE = dE / Etheor * 1.0e4
        dict_dips_theor2[k][nfields2 + 2][goodind] = np.round(dEsE, 2)
        # print "exp values : thf, img, dE eV, dEsE *1e4"
        # print np.round(thf_exp,3), img_exp, np.round(dE,2), np.round(dEsE,2)
        allres[i, :] = np.array([Etheor, np.round(dE, 2), np.round(dEsE, 2), 0.0, 0.0])

        hkldia = np.array(dict_Edia[diplinks[i, 2]][0], dtype=float)

        # sensitivity tests

        lambda_dia_nm = thf_dip_to_lambda_dia_v2(hkldia, thf_exp, calibdia)

        lambda_dia_nm_at_imgexp_plus_1 = thf_dip_to_lambda_dia_v2(hkldia, thf_exp + dthf_sur_dimg, calibdia)

        Edia_eV = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
        Edia_corr = np.round(Edia_eV, 2)

        Edia_eV2 = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm_at_imgexp_plus_1
        Edia_corr2 = np.round(Edia_eV2, 2)

        dEsEcorr = (Edia_corr - Etheor) / Etheor * 1.0e4
        dEsEcorr2 = (Edia_corr2 - Etheor) / Etheor * 1.0e4
        dict_dips_theor2[k][nfields2 + 3][goodind] = np.round(dEsEcorr, 2)
        allres[i, 3] = np.round(dEsEcorr, 2)
        allres[i, 4] = np.round(dEsEcorr2 - dEsEcorr, 2)

    print("list_ndia_dips = ", list_ndia_dips)
    #    print "k 0, numdat 1, ndia 2, img 3, confidence 4, depth 5 , Etheor 6, dE(eV) 7, dE/E start(*1e4) 8 dE/E corr(*1e4) 9"
    print("E en eV, dE∕E en 1e-4")
    print("[k, numdat, ndia, img , confidence, depth] , Etheor, dE, (dE/E)_start, (dE/E)_corr, d(dE∕E)/dimg ")
    for i in range(ndiplinks):
        print(diplinks[i], allres[i, 0], allres[i, 1], allres[i, 2], allres[i, 3], allres[i, 4])
    dEsEall = allres[:, 3]

    mean1 = dEsEall.np.mean().round(decimals=2)
    std1 = dEsEall.std().round(decimals=3)
    range1 = np.round((max(dEsEall) - min(dEsEall)), 2)

    dev_list_start_by_dip = dEsEall * 1.0

    print("dE/Ecorr, units = 0.1 mrad")
    print("statistics on all dips :")
    print("mean std range min max npts")
    print(mean1, std1, range1, np.round(min(dEsEall), 2), np.round(max(dEsEall), 2), len(dEsEall))

    dev_list_start_grouped_by_ndia = np.zeros(len(list_ndia_dips), float)
    k = 0
    print("statistics on dips with the same ndia : ")
    print("ndia mean std range min max npts")
    for ndia in list_ndia_dips:
        ind1 = np.where(diplinks[:, 2] == ndia)
        tt = dEsEall[ind1[0]]
        if len(ind1[0]) > 1:
            print(ndia, tt.np.mean().round(decimals=2), tt.std().round(decimals=3),
                                                    np.round((max(tt) - min(tt)), 2),
                                                    np.round(min(tt), 2),
                                                    np.round(max(tt), 2),
                                                    len(ind1[0]))
            dev_list_start_grouped_by_ndia[k] = tt.np.mean()
        else:
            print(ndia, tt[0].round(decimals=2))
            dev_list_start_grouped_by_ndia[k] = tt[0]

        k = k + 1

    # fit de corr0 et corr1 (uilab pour thf = thf_ref)
    #    en gros pour le fit il me faut une fonction à minimiser
    # somme de carres de termes de type
    # a_i0 * corr0 + a_i1 * corr1 - dev_i
    #
    # avec a_ij = d(dE∕E (ndia = i, thf=thf_ref)) / d(corr_j)
    # et dev_i = dE/E np.mean(ndia = i, corr0_start, corr1_start)

    mat_corr01 = np.zeros((len(list_ndia_dips), 2), float)
    mat_corr23 = np.zeros((ndiplinks, 2), float)

    thf_test = thf_ref + 1.0
    print("thf_ref = ", thf_ref)
    print("thf_test pour corr0 corr1 dthf_sur_dimg = ", thf_ref)
    print("thf_test pour corr2 corr3= ", thf_test)
    print("NB : vlab a un effet nul a thf = thf_ref")
    print("dthf_sur_dimg (deg) = ", dthf_sur_dimg)
    print("d(dE/E)/d(corr) pour corr = 1e-4 ou corr = dthf_sur_dimg ")
    print("ndia corr0 corr1 corr2 corr3 dthf_sur_dimg")
    k = 0
    for ndia in list_ndia_dips:
        hkldia = np.array(dict_Edia[ndia][0], dtype=float)
        dE_sur_E_list = np.zeros(5, float)
        ind0 = np.where(diplinks[:, 2] == ndia)

        for i in range(4):
            if i in [0, 1]:
                thf_test2 = thf_ref
            elif i in [2, 3]:
                thf_test2 = thf_test
            dE_sur_E_list[i] = dcalibdia_or_dthf_to_dE_surE(hkldia,
                                                            calibdia,
                                                            ncorr=i,  # numero de la composante de geometry_corrections qui varie
                                                            dcorr=1.0,  # amplitude de la correction en unites 1e-4
                                                            thf=thf_test2,  # degres
                                                            dthf=0.0,  # degres
                                                            verbose=0)
        mat_corr01[k, :] = dE_sur_E_list[:2]
        mat_corr23[ind0[0], :] = dE_sur_E_list[2:4]

        dE_sur_E_list[4] = dcalibdia_or_dthf_to_dE_surE(hkldia,
                                                        calibdia,
                                                        ncorr=0,  # numero de la composante de geometry_corrections qui varie
                                                        dcorr=0.0,  # amplitude de la correction en unites 1e-4
                                                        thf=thf_ref,  # degres
                                                        dthf=dthf_sur_dimg,  # degres
                                                        verbose=0)
        print(ndia, dE_sur_E_list.round(decimals=2))
        k = k + 1

    print("mat_corr01 =\n", mat_corr01.round(decimals=2))
    print("mat_corr23 =\n", mat_corr23.round(decimals=4))

    diff_corr = np.zeros(4, float)
    corr1, success = so.leastsq(
        err_func_corr01, diff_corr[:2], args=(mat_corr01, dev_list_start_grouped_by_ndia), xtol=0.00001
    )

    print("corr1, success = ", corr1, success)

    diff_corr = np.ones(4, float)
    print("err01 =", err_func_corr01(diff_corr, mat_corr01, dev_list_start_grouped_by_ndia))
    print("err23 = ", err_func_corr23(diff_corr, mat_corr23, dthf_list_by_dip, dev_list_start_by_dip))
    #    print "diplinks[:,2] = \n", diplinks[:,2]
    #    str1 = ' '.join(str(e) for e in geometry_corrections)
    #    print '#Geometry corrections ui_pitch ui_yaw axis_yaw axis_roll (0.1 mrad units) : '
    #    print str1
    if 0:
        print("corrected vectors :")
        print("incident beam : uilab = ", uilab.round(decimals=5))
        print("rotation axis : vlab = ", vlab.round(decimals=5))

    if save_new_dict_dips_theor:
        outputfilename = filepathout + "dict_dips_theor_with_harmonics_with_exp.dat"
        print("saving dips theor + exp + links in : ", outputfilename)
        save_dict_dips_theor(
            outputfilename,
            dict_dips_theor2,
            fileEdia,
            filespotlist_sample,
            include_exp="yes",
            filediplinks=filediplinks,
        )

    return (mean1, std1, range1)


#    fitfunc = lambda pp, x: pp[0]*np.exp(-(x-pp[1])**2/(2*(pp[2]/2.35)**2))+pp[3]+x*pp[4]
#
#    # Distance to the target function
#    #errfuncstat = lambda pp, x, y: ((fitfunc(pp,x) -y)/sqrt(abs(y)+equal(abs(y),0)))**2
#    errfunc2 = lambda pp, x, y: ((fitfunc(pp,x) -y)/sqrt(abs(y)+equal(abs(y),0)))**2
#    errfunc = lambda pp, x, y: ((fitfunc(pp,x) -y))**2
#
#    p1, success = optimize.leastsq(errfunc, p0, args = (xexp,yexp), xtol = 0.00001)


def err_func_corr01(corr, mat_corr01, dev_start_ndia):

    # mat_corr : lignes : differents ndia,
    # colonnes : differentes comp de geometry_corrections

    tt = np.dot(mat_corr01, corr[:2]) - dev_start_ndia
    tt2 = tt ** 2
    err_value = tt2.sum() / len(tt)

    return err_value


def err_func_corr23(corr, mat_corr23, dthf_list, dev_start_dip):

    # mat_corr : lignes : differents dips,
    # colonnes : differentes comp de geometry_corrections

    tt = np.dot(mat_corr23, corr[2:4])
    tt2 = np.multiply(tt, dthf_list) - dev_start_dip
    tt3 = tt2 ** 2
    err_value = tt3.sum() / len(tt)

    return err_value


# 0 numdat
# 1 HKL_sample
# 2 xypix_xyfit
# 3 Ipixmax
# 4 Etheor_sample
# 5 ndia_list
# 6 HKL_dia_list_theor
# 7 thf_list_theor
# 8 dip_depth_theor
# 9 slope_thf_vs_Edia
#    10 img_exp
#    11 thf_exp
#    12 dEsE_interpol
#    13 dEsE_recalc_withcorr


def interpol_new(xylist, newx):
    # only for monotonous curves
    # yy may contain several columns

    xx = xylist[:, 0]
    yy = xylist[:, 1:]
    ind_inf = np.where(xx < newx)
    ind_sup = np.where(xx > newx)
    # print xx[0], xx[-1], newx
    # print ind_inf[0]
    # print ind_sup[0]
    # if (len(ind_inf[0])>0)&((len(ind_sup[0])>0):
    tt = np.array([ind_inf[0][0], ind_inf[0][-1], ind_sup[0][0], ind_sup[0][-1]])
    tt1 = np.sort(tt)
    # print tt1
    pos1 = tt1[1]
    pos2 = tt1[2]
    # elif (len(ind_inf[0])
    nycol = np.shape(yy)[1]
    newy = np.zeros(nycol, float)
    slope1 = np.zeros(nycol, float)
    # print pos1, pos2
    # print newx, xx[pos1], xx[pos2]
    for k in range(nycol):
        slope1[k] = (yy[pos2, k] - yy[pos1, k]) / (xx[pos2] - xx[pos1])
        newy[k] = yy[pos1, k] + slope1[k] * (newx - xx[pos1])
        # print np.round(newy[k],4), yy[pos1,k], yy[pos2,k]
        # print slope1[k]

    return (newy, slope1)


def convert_JSM_fitfile_into_OR_fitfile(filefit, min_matLT=True, verbose=0,
                                    verbose2=0, check_Etheor=0, elem_label="Ge"):

    if verbose:
        print("convert fit file from LaueTool24.py to fit file from multigrain.py: \n", filefit)

    UBmatLT3x3 = np.zeros((3, 3), dtype=np.float)
    B0matLT3x3 = np.zeros((3, 3), dtype=np.float)
    strain = np.zeros((3, 3), dtype=np.float)
    f = open(filefit, "r")
    i = 0

    UBmatrixfound = 0
    B0matrixfound = 0
    calibfound = 0
    pixdevfound = 0
    strainfound = 0
    linecalib = 0
    linepixdev = 0
    linestrain = 0
    list1 = []
    linestartspot = 10000
    lineendspot = 10000
    ccdlabelfound = 0
    lineccdlabel = 0

    try:
        for line in f:
            i = i + 1
            # print i
            if line[:5] == "#spot":
                _ = line.rstrip("\n")
                linestartspot = i + 1
                if verbose:
                    print(line)
            if line[:5] == "#UB m":
                if verbose:
                    print(line)
                B0matrixfound = 1
                linestartB0mat = i
                lineendspot = i
                j = 0
                if verbose:
                    print("UB matrix found")
            if line[:3] == "#B0":
                if verbose:
                    print(line)
                UBmatrixfound = 1
                linestartUBmat = i
                j = 0
                if verbose:
                    print("B0 matrix found")
            if line[:3] == "#De":
                if verbose:
                    print(line)
                calibfound = 1
                linecalib = i + 1
            if line[:4] == "# Me":
                if verbose:
                    print(line)
                pixdevfound = 1
                linepixdev = i
            if line[:3] == "#de":
                if verbose:
                    print(line)
                strainfound = 1
                linestrain = i
                j = 0
            if line[:4] == "#CCD":
                if verbose:
                    print(line)
                ccdlabelfound = 1
                lineccdlabel = i + 1

            if UBmatrixfound:
                if i in (linestartUBmat + 1, linestartUBmat + 2, linestartUBmat + 3):
                    tt = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    if verbose:
                        print(tt)
                    UBmatLT3x3[j, :] = np.array(tt, dtype=float)
                    j = j + 1
            if B0matrixfound:
                if i in (linestartB0mat + 1, linestartB0mat + 2, linestartB0mat + 3):
                    tt = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    if verbose:
                        print(tt)
                    B0matLT3x3[j, :] = np.array(tt, dtype=float)
                    j = j + 1
            if strainfound:
                if i in (linestrain + 1, linestrain + 2, linestrain + 3):
                    tt = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    if verbose:
                        print(tt)
                    strain[j, :] = np.array(tt, dtype=float)
                    j = j + 1
            if calibfound & (i == linecalib):
                calib = np.array(
                    line.rstrip("\n").replace("[", "").replace("]", "").split(",")[:5], dtype=float)
                if verbose:
                    print("calib = ", calib)
            if pixdevfound & (i == linepixdev):
                pixdev = float(line.rstrip("\n").split(":")[-1])
                if verbose:
                    print("pixdev = ", pixdev)
            if (i >= linestartspot) & (i < lineendspot):
                list1.append(line.rstrip("\n").replace("[", "").replace("]", "").split())
            if ccdlabelfound & (i == lineccdlabel):
                CCDlabel = line.rstrip(MG.PAR.cr_string)
                if verbose:
                    print(CCDlabel)

    finally:
        f.close()
        _ = i

    # print "linetot = ", linetot

    # OR = spot# Intensity h k l  pixDev xexp yexp Etheor(eV)
    # JSM = spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz

    #    if verbose : print list1
    data_fit_JSM = np.array(list1, dtype=float)
    if verbose:
        print("data_fit_JSM :")
        print(data_fit_JSM)

    matLT3x3 = np.dot(UBmatLT3x3, B0matLT3x3)

    if verbose:
        print(np.shape(data_fit_JSM))

    data_fit = np.column_stack((data_fit_JSM[:, :6], data_fit_JSM[:, 7:9], data_fit_JSM[:, 6] * 1000.0))

    if verbose:
        print("data_fit : ")
        print("shape :", np.shape(data_fit))
        print("first line : \n", data_fit[0, :])
        print("last line : \n", data_fit[-1, :])

    # print "UB matrix = \n", matLT3x3.round(decimals=6)
    if verbose2:
        print("before transfo")
        print(data_fit[0, 2:5])
        print(data_fit[-1, 2:5])
        q0 = np.dot(matLT3x3, data_fit[0, 2:5])
        print("q0 = ", q0.round(decimals=4))
        qm1 = np.dot(matLT3x3, data_fit[-1, 2:5])
        print("qm1 = ", qm1.round(decimals=4))

    if min_matLT == True:
        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        matLT3x3 = matmin
        print("transfmat \n", list(transfmat))
        # transformer aussi les HKL pour qu'ils soient coherents avec matmin
        hkl = data_fit[:, 2:5]
        data_fit[:, 2:5] = np.dot(transfmat, hkl.transpose()).transpose()

    if verbose2:
        print("after transfo")
        print(data_fit[0, 2:5])
        print(data_fit[-1, 2:5])
        q0 = np.dot(matLT3x3, data_fit[0, 2:5])
        print("q0 = ", q0.round(decimals=4))
        qm1 = np.dot(matLT3x3, data_fit[-1, 2:5])
        print("qm1 = ", qm1.round(decimals=4))

    data_fit_sorted = MG.sort_peaks_decreasing_int(data_fit, 1)
    nfit = np.shape(data_fit_sorted)[0]

    if check_Etheor:
        print("checking Etheor using elem_label = ", elem_label)

        Etheor = np.zeros(nfit, float)
        uilab = np.array([0.0, 1.0, 0.0])
        latticeparam = DictLT.dict_Materials[elem_label][1][0] * 1.0
        print(latticeparam)
        dlatu = np.array([latticeparam, latticeparam, latticeparam, MG.PI / 2.0, MG.PI / 2.0, MG.PI / 2.0])
        matstarlab = F2TC.matstarlabLaueTools_to_matstarlabOR(matLT3x3)
        mat = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu)
        for i in range(nfit):
            qlab = (
                data_fit_sorted[i, 2] * mat[0:3]
                + data_fit_sorted[i, 3] * mat[3:6]
                + data_fit_sorted[i, 4] * mat[6:]
            )
            uqlab = qlab / MG.norme(qlab)
            sintheta = -np.inner(uqlab, uilab)
            if sintheta > 0.0:
                # print "reachable reflection"
                Etheor[i] = DictLT.E_eV_fois_lambda_nm * MG.norme(qlab) / (2.0 * sintheta) * 10.0
                print(data_fit_sorted[i, 2:5], np.round(Etheor[i], 2), np.round(data_fit_sorted[i, 8], 1))

        dE = Etheor - data_fit_sorted[:, 8]
        if dE.max() > 2.0:  # 2 eV
            data_fit_sorted[:, 8] = Etheor

    filesuffix = "_from_GUI"
    filefit_OR = MG.save_fit_results(
        filefit, data_fit_sorted, matLT3x3, strain, filesuffix, pixdev, 0, calib, CCDlabel
    )

    return filefit_OR


def read_dict_diaexp(filediaexp):

    """
    read file with Ipix vs img from mono4c + other info

    """
    listint = [0]
    listfloat = [1]

    dict_values_names = ["Ipix_vs_img", "spotlistref", "spec_scan_header"]

    dict_skip_rows = [0, 5, 0]

    dict_next_lines_are_comments = [0, 0, 0]

    ndict = len(dict_values_names)
    linepos_list = np.zeros(ndict, dtype=int)
    linepos_end_list = np.zeros(ndict, dtype=int)

    f = open(filediaexp, "r")
    i = 0

    # endlinechar = '\r\n' # unix
    endlinechar = "\n"  # dos
    current_j = 10000
    first_time = 1
    try:
        for line in f:
            # print line.rstrip(endlinechar)
            for j in range(ndict):
                if line.rstrip(endlinechar) == dict_values_names[j]:
                    linepos_list[j] = i + dict_skip_rows[j]
                    current_j = j
                    first_time = 0
                    if j > 0:
                        if dict_next_lines_are_comments[j - 1] == 0:
                            linepos_end_list[j - 1] = i
            if (line[0] == "#") & (first_time == 0):
                if dict_next_lines_are_comments[current_j] == 1:
                    linepos_end_list[current_j] = i
                    first_time = 1
            i = i + 1
    finally:
        f.close()

    print("linepos_list = ", linepos_list)
    print("linepos_end_list = ", linepos_end_list)

    f = open(filediaexp, "r")
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)

    f.seek(0)
    print(f.readline())

    dict_exp = {}

    # lecture dictionnaire spots
    for j in range(ndict - 1):

        n = linepos_list[j]
        print("n = ", n)
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        print(f.readline())
        f.seek(line_offset[n + 1])
        i = 0
        nlines = linepos_end_list[j] - linepos_list[j] - 1
        while i < nlines:
            tt = f.readline()
            # print tt,
            tt1 = tt.rstrip("\r\n")
            # version string plus lisible pour verif initiale
            # if n == 0 : dict_grains[i] = "[" + tt1 + "]"
            # else : dict_grains[i] = dict_grains[i] + "[" + tt1 + "]"
            # version array
            if j in listint:
                tt2 = np.array(tt1.split(), dtype=int)
            elif j in listfloat:
                # print tt1.split()
                tt2 = np.array(tt1.split(), dtype=float)
            # print tt2
            if i == 0:
                dict_exp[j] = [tt2]
            else:
                dict_exp[j].append(tt2)
            i = i + 1
    f.close()

    # version string
    # print dict_exp

    for j in range(ndict - 1):
        print(dict_values_names[j], np.shape(dict_exp[j]))

    return dict_exp


def hkl_to_xycam(matstarlab, hkl, calib, CCDlabel=MG.PAR.CCDlabel):

    uflab_cen = np.array([0.0, 0.0, 1.0])
    uilab = np.array([0.0, 1.0, 0.0])
    mat = np.array(matstarlab, dtype=float)
    qlab = hkl[0] * mat[0:3] + hkl[1] * mat[3:6] + hkl[2] * mat[6:]
    uqlab = qlab / norme(qlab)
    sintheta = -np.inner(uqlab, uilab)
    uflab = uilab + 2.0 * sintheta * uqlab
    xycam = MG.uflab_to_xycam_gen(uflab, calib, uflab_cen, pixelsize=DictLT.dict_CCD[CCDlabel][1])

    return xycam


# def hkl_to_Etheor(matstarlab,
#                 hkl, elem_label = "Ge"):
#
#    # attention seulement pour cristaux cubiques
#    # pas verifie matstarlab_to_matwithlatpar pour cristaux non cubiques
#    uilab = np.array([0.0,1.0,0.0])
#    latticeparam_angstroms = float(DictLT.dict_Materials[elem_label][1][0])
#    print latticeparam_angstroms
#    dlatu_nm = np.array([latticeparam_angstroms/10., latticeparam_angstroms/10., latticeparam_angstroms/10., MG.PI / 2.0, MG.PI / 2.0, MG.PI / 2.0])
#    print dlatu_nm
#    mat = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu_nm)
#    print "mat with latpar", mat
#
#    qlab = hkl[0]*mat[0:3]+ hkl[1]*mat[3:6]+ hkl[2]*mat[6:]
#    uqlab = qlab/norme(qlab)
#    sintheta = -np.inner(uqlab,uilab)
#    Etheor = DictLT.E_eV_fois_lambda_nm*norme(qlab)/(2.*sintheta)
#
#    return(Etheor)


def build_dict_spec_mesh(spec_file,
                        scan_list,
                        motor_names=["xs"],
                        dict_mot_pos_in_header={"xs": [1, 4, 0.0], "ys": [1, 5, 0.0], "zech": [1, 3, 0.0]},
                        img_counter="img",
                        list_colnames=None):
    # output :
    # dict_spec_all_scans : toutes les colonnes de tous les scans
    # ctime_list : counting time
    # npts_reflist : seulement utile pour les mesh, numero du point du debut de chaque thf scan
    # scan_title_list : seulement utile pour liste de scan (pas pour les mesh)
    # img_reflist : numero d'image du debut de chaque thf scan
    # user_comment_list : liste des headers de tous les scans
    #    dict_mot_pos_in_header_all_scans : liste des positions des "autres" moteurs pour tous les scans
    #                liste des autres moteurs donnee dans dict_mot_pos_in_header
    # input :
    # scan_list : liste de longueur 1 (pour mesh scan) ou N (pour liste de scans)
    # motor_names : utile seulement pour mesh scan
    # dict_mot_pos_in_header : noms des moteurs dont il faut aller chercher la position dans le header des scans
    #         et position ligne / colonne dans les lignes de type #O du header

    if len(scan_list) == 1:  # mesh thf + autre moteur

        if list_colnames is None:
            list_colnames = motor_names + [img_counter, "thf", "Monitor", "fluo"]

        dict_spec, ctime, user_comment1, dict_mot_pos_in_header, scan_title = MG.read_scan_in_specfile(
            spec_file,
            scan_list[0],
            list_colnames=list_colnames,
            verbose=1,
            user_comment="",
            dict_mot_pos_in_header=dict_mot_pos_in_header)

        user_comment_list = [user_comment1]
        thf_in_scan = dict_spec["thf"]
        img_in_scan = dict_spec["img"]
        thf_min = min(thf_in_scan)
        ind0 = np.where(abs(thf_in_scan - thf_min) < 0.001)
        npts_reflist = ind0[0]
        print("point number for img ref :", npts_reflist)
        img_reflist = img_in_scan[ind0[0]]
        motor_in_scan = np.array(dict_spec[motor_names[0]][npts_reflist], float)
        print(motor_names[0], ":", motor_in_scan.round(decimals=4))
        print("img_reflist :", img_reflist)

        ctime_list = ctime
        scan_title_list = scan_title

        return (img_reflist,
            user_comment_list,
            dict_spec,
            dict_mot_pos_in_header,
            npts_reflist,
            ctime_list,
            scan_title_list)

    else:  # series of thf scans

        if list_colnames is None:
            list_colnames = [img_counter, "thf", "Monitor", "fluo"]

        dict_spec_all_scans = {}
        dict_mot_pos_in_header_all_scans = {}
        img_reflist = []
        user_comment_list = []
        ctime_list = []
        scan_title_list = []
        firsttime = 1
        for spec_scan_num in scan_list:

            dict_spec, ctime, user_comment1, dict_mot_pos_in_header, scan_title = read_scan_in_specfile(
                spec_file,
                spec_scan_num,
                list_colnames=list_colnames,
                verbose=1,
                user_comment="",
                dict_mot_pos_in_header=dict_mot_pos_in_header)

            user_comment_list.append([user_comment1])
            img_in_scan = dict_spec[img_counter]
            img_reflist.append(img_in_scan[0])
            ctime_list.append(ctime)
            scan_title_list.append(scan_title)
            if firsttime:
                for key, value in dict_spec.items():
                    dict_spec_all_scans[key] = [value]
                for key, value in dict_mot_pos_in_header.items():
                    dict_mot_pos_in_header_all_scans[key] = [value[2]]
                firsttime = 0
            else:
                for key, value in dict_spec.items():
                    dict_spec_all_scans[key].append(value)
                for key, value in dict_mot_pos_in_header.items():
                    dict_mot_pos_in_header_all_scans[key].append(value[2])

        for key, value in dict_spec_all_scans.items():
            print(key, value)
        for key, value in dict_mot_pos_in_header_all_scans.items():
            print(key, value)

        if img_counter == "img2":  # VHR
            print("warning : VHR camera : using image number = img2 + 1")
        print("img_reflist :", img_reflist)
        print("ctime_list : ", ctime_list)
        print("scan_title_list : ", scan_title_list)

        #        print user_comment_list

        npts_reflist = np.zeros(len(scan_list), int)  # utile seulement pour les mesh

        return (img_reflist,
                    user_comment_list,
                    dict_spec_all_scans,
                    dict_mot_pos_in_header_all_scans,
                    npts_reflist,
                    ctime_list,
                    scan_title_list)


def build_dict_xy_E(img_list,
                    filespotlist_sample,
                    filepathfit,
                    filepathdat,
                    fileprefix,
                    list_spot_keys=[0],
                    datfile_suffix=".dat",
                    fitfile_suffix="_t_UWN.fit",
                    CCDlabel=MG.PAR.CCDlabel,
                    elem_label="Ge",
                    min_matLT=0,
                    fitfile_type="MGnew",
                    pixelsize=DictLT.dict_CCD[MG.PAR.CCDlabel][1]):

    dxy_tol = 1.0
    #   pour un dmesh thf xs ou une serie de scans en thf a differents xs ys zech

    #    list_spot_keys : keys des spots a traiter

    # utilise 2 fitfiles, un avec min pixdev et un avec max npeaks
    # 1ere image de chaque scan en thf

    #    numdat 0, xy_integer 1:3, xy_fit 3:5, hkl 5:8, Ipixmax 8, Etheor 9 \n"

    data_ref = np.loadtxt(filespotlist_sample, skiprows=5)

    xyint_ref = np.array(data_ref[:, 1:3].round(decimals=1), dtype=int)
    hkl_ref = np.array(data_ref[:, 5:8].round(decimals=1), dtype=int)
    xyfit_ref = np.array(data_ref[:, 3:5], dtype=float)
    Etheor_ref = np.array(data_ref[:, 9], dtype=float)

    dlatu_angstroms_deg = DictLT.dict_Materials[elem_label][1]
    dlatu_nm_rad = deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
    #            print "dlatu_nm_rad = ", dlatu_nm_rad

    # rajouter img_list en commentaire dans fichier

    dict_xy_E = {}
    # 0 xyint_ref
    # 1 hkl_ref
    # 2 Etheor_ref
    # 3 xint_list
    # 4 yint_list
    # 5 Etheor_list

    dict_values_names = ["xyint_ref", "hkl_ref", "Etheor_ref", "xint_list", "yint_list", "Etheor_list"]

    print("reference spots in spotlistref :")
    for key in list_spot_keys:
        print(key, xyint_ref[key], hkl_ref[key], xyfit_ref[key])
        dict_xy_E[key] = [xyint_ref[key], hkl_ref[key], Etheor_ref[key], [], [], []]  # 0

    _ = len(list_spot_keys)
    _ = len(img_list)

    kk = 0
    print("img, key, xypic, Etheor, dxypic, dEtheor")
    for img in img_list:

        filedat = (filepathdat
            + fileprefix
            + MG.imgnum_to_str(img, MG.PAR.number_of_digits_in_image_name)
            + datfile_suffix)

        data_dat = np.loadtxt(filedat, skiprows=1)

        data_dat = np.array(data_dat, dtype=float)

        xyfit_dat = data_dat[:, :2]

        xyint_dat = xyfit_dat - data_dat[:, 7:9]

        filefit_nopath = (
            fileprefix + MG.imgnum_to_str(img, MG.PAR.number_of_digits_in_image_name) + fitfile_suffix)

        filefit = filepathfit + filefit_nopath

        if filefit_nopath not in os.listdir(filepathfit):
            for key in list_spot_keys:
                for j in [3, 4]:
                    dict_xy_E[key][j].append(0)
                dict_xy_E[key][5].append(0.0)
            continue

        res1 = MG.read_any_fitfitfile_multigrain(filefit,
                                                verbose=1,
                                                fitfile_type=fitfile_type,
                                                check_Etheor=1,
                                                elem_label=elem_label,
                                                check_pixdev=1,
                                                pixelsize=pixelsize,
                                                min_matLT=min_matLT,
                                                check_pixdev_JSM=0)

        _, _, _, matstarlab_all, data_fit_all, calib_all, pixdev_all, _, _, ind_h_x_int_pixdev_Etheor = res1

        indh = ind_h_x_int_pixdev_Etheor[0]
        indx = ind_h_x_int_pixdev_Etheor[1]
        ind_Etheor_in_data_fit = ind_h_x_int_pixdev_Etheor[4]

        ind_hkl_in_data_fit = list(range(indh, indh + 3))
        ind_xy_in_data_fit = list(range(indx, indx + 2))

        matstarlab = matstarlab_all[0]
        data_fit = np.array(data_fit_all, float)
        calib = calib_all[0]
        _ = pixdev_all[0]

        #        matstarlab, data_fit, calib, pixdev = \
        #                        F2TC.readlt_fit(filefit,  min_matLT = True,
        #                                                          readmore = True,
        #                                                          verbose = 0,
        #                                                          verbose2 = 0)
        #      spot# Intensity h k l  pixDev xexp yexp Etheor
        _ = np.array(data_fit[:, ind_xy_in_data_fit], dtype=float)

        hkl_fit = np.array(data_fit[:, ind_hkl_in_data_fit].round(decimals=1), dtype=int)

        numdat_fit = np.array(data_fit[:, 0].round(decimals=1), dtype=int)

        Etheor_fit = np.array(data_fit[:, ind_Etheor_in_data_fit], dtype=float)

        matwithlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu_nm_rad)

        for key in list_spot_keys:

            dhkl = abs(hkl_fit - hkl_ref[key]).sum(axis=1)
            ind0 = np.where(dhkl < 1)
            if len(ind0[0]) > 0:
                #                print "spot already indexed"
                Etheor_new = Etheor_fit[ind0[0][0]]
                numdat = numdat_fit[ind0[0][0]]
                xyint_new = xyint_dat[numdat]

            else:
                print("unindexed spot : look for missing refl.")
                xytheor = hkl_to_xycam(matstarlab, hkl_ref[key], calib, CCDlabel=CCDlabel)

                dxy = abs(xyfit_dat - xytheor).sum(axis=1)
                ind1 = np.where(dxy < dxy_tol)
                if len(ind1[0]) > 0:
                    print("missing spot found in dat file")
                    xyint_new = xyint_dat[ind1[0][0]]

                    Etheor_new, _, _ = MG.mat_and_hkl_to_Etheor_ththeor_uqlab(
                        matwithlatpar_inv_nm=matwithlatpar_inv_nm, hkl=hkl_ref[key])

                #                    Etheor_new = hkl_to_Etheor(matstarlab, hkl_ref[key],
                #                                    elem_label = elem_label)
                else:
                    print("missing spot not found")
                    xyint_new = np.zeros(2, float)
                    Etheor_new = 0.0

            dxy = xyint_new - xyint_ref[key]
            dEtheor = Etheor_new - Etheor_ref[key]
            #            print "dxyint to reference = ", dxy
            #            print "dEtheor to reference = ", dEtheor
            dict_xy_E[key][3].append(xyint_new[0])
            dict_xy_E[key][4].append(xyint_new[1])
            dict_xy_E[key][5].append(np.round(Etheor_new, 2))

            print(img, key, xyint_new, np.round(Etheor_new, 2), dxy.round(decimals=0), np.round(dEtheor, 2))

        kk = kk + 1
    #        tt = np.column_stack((img_list[key], xyint_list[key], Etheor_list[key]))
    ngood_img = kk
    print("ngood_img =", ngood_img)

    ndict = len(dict_values_names)
    for i in range(ndict):
        print(dict_values_names[i])
        for key, value in dict_xy_E.items():
            print(key, ":", value[i])

    return dict_xy_E


#    Etheor2 = Etheor_list.renp.shape(nimg, nkey)
# non il faut deux procedures : une qui part des .dat et cree les Ipix vs img

# et une qui part des .fit et cree les spotlistref avec les Etheor

# et ensuite il faut harmoniser les numeros des spots entre les differentes
# spotlistref pour faire une spotlistref generale
# pour savoir quelle colonne utiliser dans quel file_Ipix_vs_img


def print_dips_theor_single_hkl_sample(filedipstheor, key, thf_cen, dthf, depth_min):
    # PRINT_DIPS_THEOR_SINGLE_HKL_SAMPLE

    print("warning : harmonics may add dips :")
    print("check dictionnary file for harmonics of the chosen spot")

    dict_dips_theor, dict_values_names2 = read_dict_dips_theor(filedipstheor)

    ind_inner = dict_values_names2.index("inner_uq_dia_xz_uq_sample_xz")
    ind_thf_dip = dict_values_names2.index("thf_list_theor")
    ind_depth = dict_values_names2.index("dip_depth_theor")
    ind_hkldia = dict_values_names2.index("HKL_dia_list_theor")
    ind_ndia = dict_values_names2.index("ndia_list")
    ind_inner2 = dict_values_names2.index("uq_dia_z")
    ind_slope = dict_values_names2.index("slope_thf_vs_Edia")
    ind_Etheor = dict_values_names2.index("Etheor_sample")
    ind_xypic = dict_values_names2.index("xypix_xyfit")
    ind_hklsample = dict_values_names2.index("HKL_sample")

    ndia_list = np.array(dict_dips_theor[key][ind_ndia], dtype=int)
    print("table limited in thf to : [", thf_cen - dthf, thf_cen + dthf, "]")
    print("table limited to dips with depth > ", depth_min)
    print("key = ", key)
    print("HKL_sample = ", dict_dips_theor[key][ind_hklsample])
    print("Etheor_sample = ", dict_dips_theor[key][ind_Etheor])
    print("xypic = ", dict_dips_theor[key][ind_xypic][:2])
    print("ndia, HKLdia, dip_depth, slope, inner, inner2, thf_dip")

    for i in range(len(ndia_list)):
        thf_dip = float(dict_dips_theor[key][ind_thf_dip][i])
        dip_depth = float(dict_dips_theor[key][ind_depth][i])
        HKLdia = dict_dips_theor[key][ind_hkldia][i]
        inner1 = dict_dips_theor[key][ind_inner][i]
        inner2 = dict_dips_theor[key][ind_inner2][i]
        slope1 = dict_dips_theor[key][ind_slope][i]
        if (thf_dip > thf_cen - dthf) & (thf_dip < thf_cen + dthf) & (dip_depth > depth_min):
            print(ndia_list[i], HKLdia, np.round(dip_depth, 1), slope1, inner1, inner2, thf_dip)

    return ()


def print_dips_theor_single_hkl_dia(filedipstheor, ndia_ref, thf_cen, dthf):

    first_time = 1
    dict_dips_theor, dict_values_names2 = read_dict_dips_theor(filedipstheor)

    print("table limited in thf to : [", thf_cen - dthf, thf_cen + dthf, "]")
    print("ndia_ref = ", ndia_ref)

    ind_slope = dict_values_names2.index("slope_thf_vs_Edia")
    ind_inner = dict_values_names2.index("inner_uq_dia_xz_uq_sample_xz")
    ind_inner2 = dict_values_names2.index("uq_dia_z")
    ind_Etheor = dict_values_names2.index("Etheor_sample")
    ind_thf_dip = dict_values_names2.index("thf_list_theor")
    ind_depth = dict_values_names2.index("dip_depth_theor")
    ind_hkldia = dict_values_names2.index("HKL_dia_list_theor")
    ind_ndia = dict_values_names2.index("ndia_list")
    ind_xypic = dict_values_names2.index("xypix_xyfit")

    first_time = 1

    list_thf_dip = []
    list_xy = []
    list_nsample = []

    for key, value in dict_dips_theor.items():
        ndia_list = np.array(value[ind_ndia], dtype=int)
        ind0 = np.where(ndia_list == ndia_ref)
        if len(ind0[0]) > 0:
            if first_time:
                print("HKLdia", value[ind_hkldia][ind0[0][0]])
                print("key, xypic, Etheor_sample, dip_depth, slope, inner, inner2, thf_dip")
                HKLdia = np.array(value[ind_hkldia][ind0[0][0]].round(decimals=0), dtype=int)
                first_time = 0
            for i in ind0[0]:
                thf_dip = float(value[ind_thf_dip][i])
                if (thf_dip > thf_cen - dthf) & (thf_dip < thf_cen + dthf):
                    print(key,
                        value[ind_xypic][:2],
                        value[ind_Etheor],
                        np.round(value[ind_depth][i], 1),
                        value[ind_slope][i],
                        value[ind_inner][i],
                        value[ind_inner2][i],
                        value[ind_thf_dip][i])
                    list_thf_dip.append(np.round(thf_dip, 4))
                    list_xy.append(value[ind_xypic][:2])
                    list_nsample.append(key)

    print("list_thf_dip : mean, std, min, max")
    list_thf_dip = np.array(list_thf_dip, dtype=float)
    print(np.round(list_thf_dip.np.mean(), 2),
        np.round(list_thf_dip.std(), 2),
        np.round(list_thf_dip.min(), 4),
        np.round(list_thf_dip.max(), 4))
    histo = np.histogram(list_thf_dip, bins=90, range=(-90.0, 0.0))
    #    print "histogram data : ", histo[0]
    #    print "bin edges :",  histo[1]
    #    print np.shape(histo[0])
    #    print np.shape(histo[1])
    p.figure()
    barwidth = histo[1][1] - histo[1][0]
    #    print "bar width = ", barwidth
    p.bar(histo[1][:-1], histo[0], width=barwidth)
    p.xlabel("thf (deg)")
    p.ylabel("number of dips")
    title1 = "ndia = " + str(ndia_ref) + ", HKLdia = " + str(HKLdia) + "\n"
    p.title(title1, fontsize=14)
    #    p.text(-90., max(histo[0])+1., filedipstheor, fontsize = 8)
    p.xlim(-90.0, 0.0)

    list_xy = np.array(list_xy, dtype=float)
    list_nsample = np.array(list_nsample, dtype=int)

    return (list_thf_dip, list_xy, list_nsample)


def write_ndia_macro(filepathout, thf_dip_list, ndia_ref, list_xy):

    outfilename = filepathout + "ndia" + str(ndia_ref) + ".mac"

    print("writing macro file to : ", outfilename)

    npts = len(thf_dip_list)

    outputfile = open(outfilename, "w")
    str1 = "unglobal THF_TAB \n"
    outputfile.write(str1)
    str1 = "unglobal icast \n"
    outputfile.write(str1)
    str1 = "global THF_TAB \n"
    outputfile.write(str1)
    str1 = "global icast \n"
    outputfile.write(str1)

    for i in range(npts):
        str1 = "THF_TAB[" + str(i) + "]=" + str(np.round(thf_dip_list[i], 3)) + "\n"
        outputfile.write(str1)
        str1 = ("com icast "
            + str(i)
            + " xpic "
            + str(list_xy[i, 0])
            + " ypic "
            + str(list_xy[i, 1])
            + " thf_dip "
            + str(np.round(thf_dip_list[i], 3))
            + "\n")
        outputfile.write(str1)

    str1 = "for (icast=0;icast<" + str(npts) + ';icast++) {comment "icast =%g" icast ;'
    str1 = str1 + " mv thf THF_TAB[icast] ; wm thf ; sleep(1) ; dscan thf -0.2 0.2 80 0.4 } \n"
    outputfile.write(str1)
    outputfile.close()

    return outfilename


def write_mesh_thf_macro(filepathout, motor_name, motor_list, thf_list):

    outfilename = filepathout + "mesh_thf_" + motor_name + ".mac"

    print("writing macro file to : ", outfilename)

    npts = len(motor_list)

    outputfile = open(outfilename, "w")
    str1 = "unglobal MOT_TAB \n"
    outputfile.write(str1)
    str1 = "unglobal THF_TAB \n"
    outputfile.write(str1)
    str1 = "unglobal icast \n"
    outputfile.write(str1)
    str1 = "global MOT_TAB \n"
    outputfile.write(str1)
    str1 = "global THF_TAB \n"
    outputfile.write(str1)
    str1 = "global icast \n"
    outputfile.write(str1)

    for i in range(npts):
        str1 = "MOT_TAB[" + str(i) + "]=" + str(np.round(motor_list[i], 4)) + "\n"
        outputfile.write(str1)
        str1 = "THF_TAB[" + str(i) + "]=" + str(np.round(thf_list[i], 2)) + "\n"
        outputfile.write(str1)

    str1 = ("for (icast=0;icast<"
        + str(npts)
        + ';icast++) {comment "icast =%g" icast ; mv '
        + motor_name
        + " MOT_TAB[icast]")
    str1 = (str1 + "; mv thf THF_TAB[icast] ; wm thf ; wech ; sleep(1) ; dscan thf -0.1 0.1 2 1 ; ascan rien 0 1 1 0.5} \n")
    outputfile.write(str1)
    outputfile.close()

    return outfilename


def dxycam_to_dthf(xycam_list, xycam_ref, slope_thf_vs_Edia, HKLspot, calib, elem_label,
                                                    Edia_ref,
                                                    thf_ref,
                                                    pixelsize=DictLT.dict_CCD[MG.PAR.CCDlabel][1]):

    uilab = np.array([0.0, 1.0, 0.0])
    # cubic lattice only, zero strain

    latticeparam_angstroms = float(DictLT.dict_Materials[elem_label][1][0])
    print("latticeparam_angstroms = ", latticeparam_angstroms)
    dlatu_nm = np.array([latticeparam_angstroms / 10.0,
                        latticeparam_angstroms / 10.0,
                        latticeparam_angstroms / 10.0,
                        MG.PI / 2.0,
                        MG.PI / 2.0,
                        MG.PI / 2.0])
    print("dlatu_nm = ", dlatu_nm)

    dHKL_nm = dlatu_nm[0] / norme(HKLspot)

    print("dHKL_nm = ", np.round(dHKL_nm, 4))

    uqlab_ref = MG.xycam_to_uqlab(xycam_ref, calib, pixelsize=pixelsize)

    sintheta_ref = -np.inner(uqlab_ref, uilab)

    Etheor_ref = DictLT.E_eV_fois_lambda_nm / (2 * sintheta_ref * dHKL_nm)

    print("Etheor_ref = ", np.round(Etheor_ref, 2))  # eV

    thf_ref2 = thf_ref + (Etheor_ref - Edia_ref) * slope_thf_vs_Edia / 1000.0  # slope en deg/keV

    print("thf_ref2 = ", np.round(thf_ref2, 3))

    npts = np.shape(xycam_list)[0]
    Etheor_new = np.zeros(npts, float)
    thf_new = np.zeros(npts, float)

    print("xy, Etheor, thf")
    for i in range(npts):
        uqlab_new = MG.xycam_to_uqlab(xycam_list[i, :], calib, pixelsize=pixelsize)
        sintheta_new = -np.inner(uqlab_new, uilab)
        Etheor_new[i] = DictLT.E_eV_fois_lambda_nm / (2 * sintheta_new * dHKL_nm)
        thf_new[i] = thf_ref + (Etheor_new[i] - Edia_ref) * slope_thf_vs_Edia / 1000.0
        print(xycam_list[i, :].round(decimals=2), np.round(Etheor_new[i], 1), np.round(thf_new[i], 2))

    return (Etheor_new, thf_new)


# ----   -------  thf_dip_to_lambda_dia_v2
def thf_dip_to_lambda_dia_v2(hkldia, thf_dip, calibdia):
    """
    return Energy of diamond branch at thf considering calibration diamond parameters

    """
    mat_range = list(range(0, 9))
    vlab_range = list(range(9, 12))
    uilab_range = list(range(12, 15))
    thfref_range = 15

    uilab = calibdia[uilab_range]
    vlab = calibdia[vlab_range]
    matref1 = calibdia[mat_range]
    thf_ref = calibdia[thfref_range]

    matref = GT.matline_to_mat3x3(matref1)

    thf = thf_dip
    # print "thf =", thf
    matrot = MG.from_axis_vecangle_to_mat(vlab, thf_ref - thf)
    matnew = np.dot(matrot, matref)
    mat = GT.mat3x3_to_matline(matnew)
    # print "matrix : \n", mat

    # print "HKLdia = ", hkldia
    H = hkldia[0]
    K = hkldia[1]
    L = hkldia[2]
    qlab = H * mat[0:3] + K * mat[3:6] + L * mat[6:]
    uqlab = qlab / norme(qlab)
    # print "uqlab = ", uqlab
    sintheta = -np.inner(uqlab, uilab)
    # print "sintheta = ", sintheta

    lambda_dia_nm = (2.0 * sintheta) / norme(qlab)

    _ = DictLT.E_eV_fois_lambda_nm / lambda_dia_nm
    #    print "Edia (eV) = ", np.round(Edia,2)

    return lambda_dia_nm


# def thf_dip_to_lambda_dia( hkldia,
#                             thf_dip,
#                             matref,
#                             thf_ref,
#                             vlab_start,
#                             geometry_corrections):
#
#    ui_pitch = geometry_corrections[0]
#    ui_yaw = geometry_corrections[1]
#    axis_yaw = geometry_corrections[2]
#    axis_roll = geometry_corrections[3]
#
#    aya = axis_yaw * 1.0e-4
#    aro = axis_roll * 1.0e-4
#    vlab = vlab_start + np.array([0., aya, aro])
#    vlab = vlab/norme(vlab)
#    print "vlab = ", vlab
#
#    # dirv dirh : 1e-4 rad
#    dirv = ui_pitch * 1.0e-4
#    dirh = ui_yaw * 1.0e-4
#    cdirv = np.cos(dirv)
#    sdirv = sin(dirv)
#    cdirh = np.cos(dirh)
#    sdirh = sin(dirh)
#    uilab = np.array([cdirv*sdirh,cdirv*cdirh,sdirv])
#    print "uilab = ", uilab
#
#    thf = thf_dip
#    #print "thf =", thf
#    matrot = MG.from_axis_vecangle_to_mat(vlab, thf_ref-thf)
#    matnew = np.dot(matrot, matref)
#    mat = GT.mat3x3_to_matline(matnew)
#    #print "matrix : \n", mat
#
#    #print "HKLdia = ", hkldia
#    H = hkldia[0]
#    K = hkldia[1]
#    L = hkldia[2]
#    qlab = H*mat[0:3]+ K*mat[3:6]+ L*mat[6:]
#    uqlab = qlab/norme(qlab)
#    #print "uqlab = ", uqlab
#    sintheta = -np.inner(uqlab,uilab)
#    #print "sintheta = ", sintheta
#
#    lambda_dia_nm = (2.*sintheta)/norme(qlab)
#
#    Edia = DictLT.E_eV_fois_lambda_nm/lambda_dia_nm
#    print "Edia (eV) = ", np.round(Edia,2)
#
#    return(lambda_dia_nm)


def matstarlab_and_hkl_to_amean_over_dHKL(matstarlab, hkl):

    qlab = hkl[0] * matstarlab[0:3] + hkl[1] * matstarlab[3:6] + hkl[2] * matstarlab[6:]
    dHKL = 1.0 / norme(qlab)
    rlat = MG.mat_to_rlat(matstarlab)
    # print rlat
    Vcell = CP.vol_cell(rlat, angles_in_deg=0)
    amean = Vcell ** (1.0 / 3.0)

    amean_over_dHKL = amean / dHKL

    return amean_over_dHKL


def matstarlab_and_hkl_to_uqsample(matstarlab, hkl, omega=MG.PAR.omega_sample_frame):

    matstarsample3x3 = MG.matstarlab_to_matstarsample3x3(matstarlab, omega=omega)

    qsample = (hkl[0] * matstarsample3x3[:, 0] + hkl[1] * matstarsample3x3[:, 1] + hkl[2] * matstarsample3x3[:, 2]
    )

    ind0 = np.argmax(abs(qsample))

    qsample_norm = qsample / qsample[ind0]

    return qsample_norm

if __name__ == '__main__':
    # test1
    fileEdia = "/home/micha/LaueToolsPy3/LaueTools/Diamond/dict_Edia_with_uqlab_m160_m20_22keV_27Oct17_v2.dat"

    #fileEdia ='/home/micha/LaueToolsPy3/LaueTools/Diamond/CorrectedAbacus_dict_Edia_with_uqlab.dat'
    dd, ddname, thfs = read_dict_Edia_vs_thf(fileEdia)
    print('nb branches', len(dd))
    for i in range(200):
        print('Edia_list = dd[0][%d]'%i, dd[0][i])
        if i != 2:
            print('len(Edia_list = dd[0][%d])'%i, len(dd[0][i]))
    print('ddname', ddname)
    dgff
    # test2
    filespotlist_sample = '/home/micha/LaueProjects/DiamondOct18/data/Ge40_0000_spotlistref_with_harmonics.dat'
    fullpathNewAbacus = fileEdia
    outfile_prefix = 'tototest_'

    (dd,
    ddnames,
    file_dips) = build_dict_dips_theor(filespotlist_sample,
                                              fullpathNewAbacus,
                                              outfile_prefix,
                                              accurate_calc_for_thf_dip=1,
                                              read_calibdia='new')
    print('dd[0]', dd[0])
