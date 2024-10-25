# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:54:18 2013

@author: odile32
"""
#print("In param_multigrain module: in LaueTools/FileSeries")
# invisible parameters for serial_peak_search

CCDlabel = (
    "MARCCD165"
)  # OK for MAR and ROPER except few experiments with ROPER Jul11-Sep11
# voir dict_CCD dans dict_LaueTools.py pour la liste des cameras / orientations de cameras

omega_sample_frame = 40.0  # DEG

number_of_digits_in_image_name = 4

LT_REV_peak_search = "1168"  # LaueTools revision - for info only

overwrite_peak_search = 1

local_maxima_search_method = 2

thresholdConvolve = 900

CDTE, GE, W, C_VHR_May12, C_VHR_Nov12, NOFIT, SI_VHR_Feb13, CU = 0, 1, 0, 0, 0, 0, 0, 0

if CDTE:
    PixelNearRadius = 10
    IntensityThreshold = 100
    boxsize = 5
    position_definition = 1
    fit_peaks_gaussian = 1
    xtol = 0.001
    FitPixelDev = 2.0
    local_maxima_search_method = 1

if CU:  # Dec09
    PixelNearRadius = 10
    # IntensityThreshold = 250
    IntensityThreshold = 300
    boxsize = 10
    position_definition = 1
    fit_peaks_gaussian = 1
    xtol = 0.001
    FitPixelDev = 5.0

if NOFIT:

    PixelNearRadius = 10
    # IntensityThreshold = 500
    IntensityThreshold = 300
    boxsize = 5
    # XMAS offset
    position_definition = 1
    fit_peaks_gaussian = 0
    xtol = 0.001
    # FitPixelDev = 0.7
    FitPixelDev = 2.0

if GE:
    PixelNearRadius = 10
    # IntensityThreshold = 500
    IntensityThreshold = 300
    boxsize = 5
    # XMAS offset
    position_definition = 1
    fit_peaks_gaussian = 1
    xtol = 0.001
    # FitPixelDev = 0.7
    FitPixelDev = 2.0

if W:  # Sep08
    PixelNearRadius = 5
    # IntensityThreshold = 250
    IntensityThreshold = 300
    boxsize = 5
    position_definition = 1
    fit_peaks_gaussian = 1
    xtol = 0.001
    FitPixelDev = 2.0

if C_VHR_May12:
    #             'VHR_diamond':((2594, 3764), 0.031, 4095, "vhr", 4096, "uint16",
    #                            "first vhr settings of Jun 12 close to diamond 2theta axis displayed is vertical, still problem with fit from PeakSearchGUI", "tiff"),
    CCD_label = "VHR_diamond"
    local_maxima_search_method = 0
    PixelNearRadius = 100
    # IntensityThreshold = 250
    IntensityThreshold = 300
    boxsize = 20
    position_definition = 1
    fit_peaks_gaussian = 1
    xtol = 0.001
    FitPixelDev = 25.0

if C_VHR_Nov12:
    #                 'VHR_small':((2594, 2748), 0.031, 4095, "vhr", 4096, "uint16",
    #                          "vhr close to diamond Nov12 frame size is lower than VHR_diamond", "tiff"),
    CCD_label = "VHR_small"
    # basic method
    local_maxima_search_method = 0
    PixelNearRadius = 100
    IntensityThreshold = 190
    # IntensityThreshold = 300
    boxsize = 20
    position_definition = 1
    fit_peaks_gaussian = 0
    xtol = 0.001
    FitPixelDev = 25.0


if SI_VHR_Feb13:  # VHR for sample
    #             'VHR_Feb13':((2594, 2774), 0.031, 4095, 'VHR_Feb13', 4096, "uint16",
    #                          "vhr settings of Feb13 close to sample 2theta axis displayed is vertical, no problem with fit from PeakSearchGUI", "tiff"),

    # TODO : rajouter if res == 0 pour si en dehors de l'echantillon
    CCDlabel = "VHR_Feb13"
    local_maxima_search_method = 2
    PixelNearRadius = 20
    IntensityThreshold = 20.0
    boxsize = 30
    position_definition = 1
    fit_peaks_gaussian = 1
    xtol = 0.001
    FitPixelDev = 25.0
    thresholdConvolve = 2050

# invisible parameters for
# index_refine_multigrain_one_image
# serial_index_refine_multigrain
# index_refine_calib_one_image
# serial_index_refine_calib

filter_peaks_index_refine_calib = 1

maxpixdev_filter_peaks_index_refine_calib = 0.7

elem_label_index_refine_calib = "Ge"

elem_label_index_refine = "Ge"
# voir dict_Materials dans dict_LaueTools.py pour la liste des structures cristallines
# pas besoin de parametres de maille tres precis sauf si mesures de Espot

ngrains_index_refine = 4  # try to index up to "ngrains_index_refine" grains

overwrite_index_refine = 1  # overwrite existing fit files

add_str_index_refine = "_t_UWN_mg"  # string to add : UWN = "use weights no"

# invisible parameters in index_refine_one_image

check_grain_presence = 1


remove_sat = (
    0
)  # remove saturated spots with Ipixmax = Saturation value defined by CCDlabel
# = 1 : keep saturated peaks (Ipixmax = 65565) for indexation but remove them for refinement

remove_sat_calib = 0

elim_worst_pixdev = (
    1
)  # after first strain refinement, eliminate spots with pixdev > maxpixdev
# only one iteration i.e. sometimes a few spots with pixdev > maxpixdev remain

elim_worst_pixdev_calib = 1

maxpixdev = 1.0

maxpixdev_calib = 1.0

spot_index_central = [
    7,
    8,
    9,
]  # 1,2]  # spot(s) used a first spot when testing doublets for indexation
#    spot_index_central = [0,1,2,3,4,5]  # central spot or list of spots

spot_index_central_calib = 0

nbmax_probed = 10  #     # 'Recognition spots set Size (RSSS): '
# number of spots to be used as second spot when testing doublets for indexation
# selected spots = first nbmax_probed spots from spotlist
# spots in spotlist are sorted according to decreasing intensity

nbmax_probed_calib = 10

energy_max = 22  # keV

energy_max_calib = 22

rough_tolangle = (
    0.5
)  #    # 'Dist. Recogn. Tol. Angle (deg)'      # for testing doublets

rough_tolangle_calib = 0.5

fine_tolangle = (
    0.2
)  #   # 'Matching Tolerance Angle (deg)'       # for spotlink exp - theor

fine_tolangle_calib = 0.2

Nb_criterium = (20,)  #   # 'Minimum Number Matched Spots: '

Nb_criterium_calib = 20

NBRP = 1  #    NBRP = 1 # number of best result for each element of spot_index_central

NBRP_calib = 1

mark_bad_spots = 1
#      mark_bad_spots = 1 : for multigrain indexation, at step n+1 eliminate from the starting set all the spots
#                              more intense than the most intense indexed spot of step n
#                              (trying to exclude "grouped intense spots" from starting set)


# parameters for get_xyzech

xech_offset = 0xE00
yech_offset = 0xE07
zech_offset = 0xE0E

# TODO : rajouter mon_offset

# parameters for build_summary

nbtopspots = (
    10
)  # mean local grain intensity is taken over the most intense ntopspots spots

# parameters for add_columns_to_summary_file

filestf = "CdTe.stf"

# parameters for class_data_into_grainnum

rgb_tol_class_grains = 0.03


# organisation en dictionnaires par fonctions d'appel, valeurs à initialiser, peut etre creer une nouvelle variable "materiau"

# dict_PeakSearch= { "LT_REV_peak_search": ,
#                   "PixelNearRadius": ,
#                   "IntensityThreshold": ,
#                   "boxsize": ,
#                   "position_definition": ,
#                   "fit_peaks_gaussian": ,
#                   "xtol": ,
#                   "FitPixelDev": ,
#                   "local_maxima_search_method" : ,
#                   "thresholdConvolve" : ,
#                   "number_of_digits_in_image_name": ,
#                   "overwrite_peak_search" : ,
#                   "CCDlabel" :
#                  }
#
#
# dict_IndexRefine = { "remove_sat" : ,
#                     "check_grain_presence": ,
#                     "elim_worst_pixdev": ,
#                     "maxpixdev" : ,
#                     "spot_index_central": ,
#                     "nbmax_probed": ,
#                     "energy_max": ,
#                     "rough_tolangle": ,
#                     "fine_tolangle": ,
#                     "Nb_criterium": ,
#                     "NBRP": ,
#                     "mark_bad_spots": ,
#                     "CCDlabel": ,
#                     "ngrains_index_refine": ,
#                     "number_of_digits_in_image_name": ,
#                     "add_str_index_refine": ,
#                     "overwrite_index_refine": ,
#                     "elem_label_index_refine": ,
#                     "elem_label_index_refine_calib": ,
#                     "remove_sat_calib": ,
#                     "elim_worst_pixdev_calib" : ,
#                     "maxpixdev_calib": ,
#                     "spot_index_central_calib": ,
#                     "nbmax_probed_calib": ,
#                     "energy_max_calib": ,
#                     "rough_tolangle_calib": ,
#                     "fine_tolangle_calib": ,
#                     "Nb_criterium_calib" : ,
#                     "NBRP_calib" : ,
#                     "maxpixdev_filter_peaks_index_refine_calib" :
#                  }
#
#
# dict_BuildSummary = { "number_of_digits_in_image_name" : ,
#                      "xech_offset": ,
#                      "yech_offset": ,
#                      "zech_offset" : ,
#                      "nbtopspots": ,
#                      "filestf": ,
#                      "elem_label_index_refine": ,
#                    }
#
# dict_PlotMap = {}
#
# dict_SortGrain = {"rgb_tol_class_grains": }
#
# dict_PlotGrain = {}


# carriage return string
# for unix
cr_string = "\r\n"

# for windows
# cr_string = "\n"
#
