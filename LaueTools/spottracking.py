from __future__ import print_function
"""
module of lauetools project

JS Micha November 2019

spot tracking allows to follow pixel positions of spot in peaks list file
"""

import sys
import numpy as np

if sys.version_info.major == 3:
    from . import generaltools as GT
    from . import IOLaueTools as IOLT
else:
    import generaltools as GT
    import IOLaueTools as IOLT


def getspotindex(XY,
                spotslist_XY,
                maxdistancetolerance=5,
                minimum_seconddistance=10,
                predictedshift_X=None,
                predictedshift_Y=None):
    """
    get spot index of spot in the spot list located closest to target XY=[X,Y]

    maxdistancetolerance: largest acceptable distance to consider the spots association
    minimum_seconddistance : minimum distance for a second spot in spotslist_XY close to the spot
    at target_XY to validate the association. Otherwise the association is ambiguous.

    if predictedshift_X and predictedshift_Y are given in pixels
    then target XY is shifted accordingly allowing a finer tolerance in maxdistancetolerance


    return None:
        if closest spot is farther than 'maxdistancetolerance' from target XY
        or
        if second closest is at least at 'minimum_seconddistance' from target XY


    """
    target_XY = XY

    if (predictedshift_X, predictedshift_Y) != (None, None):
        target_XY = [XY[0] + predictedshift_X, XY[1] + predictedshift_Y]

    indices, distances = GT.FindTwoClosestPoints(spotslist_XY, target_XY)

    first, _ = indices
    first_dist, second_dist = distances

    if first_dist > maxdistancetolerance:
        return None
    if second_dist < minimum_seconddistance:
        return None

    return first, first_dist


def getSpotsAssociations(spotlist_XY,
                            ref_list_XY,
                            maxdistancetolerance=5,
                            minimum_seconddistance=10,
                            list_predictedshift_X=None,
                            list_predictedshift_Y=None):
    """
    return spot association list from spots in two lists

    input:
    spotlist_XY: list of [X,Y]
    ref_list_XY: list of [X,Y]

    maxdistancetolerance: largest acceptable distance to consider the spots association
    minimum_seconddistance : minimum distance for a second spot in ref_list_XY close to a spot
    in spotlist_XY to validate the association. Otherwise the association is ambiguous.

    list_predictedshift_X,list_predictedshift_Y
        list of guessed shift in X and Y spot wise for spot in spotlist_XY
        allowing a finer tolerance in maxdistancetolerance


    return:
    list of correspondences
    [index in spotlist_XY, index in ref_list_XY,pixel distance between associated spots]
    list of spots index in spotlist_XY without close association or with ambiguous association (two spots in ref list)
    """
    correspondence = []
    nocorrespondence = []
    kk = 0
    for XY in range(len(spotlist_XY)):
        predictedshift_X = None
        predictedshift_Y = None
        if list_predictedshift_X is not None and list_predictedshift_Y is not None:
            predictedshift_X = list_predictedshift_X[kk]
            predictedshift_Y = list_predictedshift_Y[kk]

        XY = spotlist_XY[kk]

        res = getspotindex(XY,
                            ref_list_XY,
                            maxdistancetolerance=maxdistancetolerance,
                            minimum_seconddistance=minimum_seconddistance,
                            predictedshift_X=predictedshift_X,
                            predictedshift_Y=predictedshift_Y)
        if res == None:
            nocorrespondence.append(kk)
        else:
            spotindex_in_ref_list, distance = res
            correspondence.append([kk, spotindex_in_ref_list, distance])
        kk += 1

    return correspondence, nocorrespondence


def sortSpotsDataCor(data_theta, Chi, posx, posy, dataintensity, referenceList):
    """
    change order of spots data (data_theta, Chi, posx, posy, dataintensity)
    according their pixel position (posx, posy) in  referenceList

    referenceList = list or array of [X,Y]  or string for full path to file .cor

    #TODO accept also .dat

    return:
    - rearranged 5 elements of data, nb of elements = nb of common pts 
    - isolated_spots_in_spotlist: indices of spot in data without association,
    - isolated_spots_in_reflist: indices of spot in referenceList without association
    """
    if isinstance(referenceList, str):
        # file path to ref peaklist
        data_ref = IOLT.readfile_cor(referenceList)

        posx_ref, posy_ref = data_ref[3:5]

        referenceList = np.array([posx_ref, posy_ref]).T

    if isinstance(referenceList, (np.ndarray, list)):

        spotlist_XY = np.array([posx, posy]).T

        corresp, isolated_spots_in_spotlist = getSpotsAssociations(spotlist_XY,
                                                                    referenceList,
                                                                    maxdistancetolerance=5,
                                                                    minimum_seconddistance=10,
                                                                    list_predictedshift_X=None,
                                                                    list_predictedshift_Y=None)

        print("corresp", corresp)
        # isolated spots in spotlist
        print("isolated_spots_in_spotlist", isolated_spots_in_spotlist)

    data = np.array([data_theta, Chi, posx, posy, dataintensity]).T

    corresp_array = np.array(corresp)
    # according to index in ref list
    arg_ind = np.argsort(corresp_array[:, 1])

    print("arg_ind", arg_ind)

    # spot index in current list sorted according to order in ref. list
    new_order_spotindices = np.array(corresp_array[arg_ind][:, 0], dtype=np.int)

    print("new_order_spotindices", new_order_spotindices)

    print("data.shape", data.shape)

    resorted_data = np.take(data, new_order_spotindices, axis=0)

    # spot in ref without association
    associated_spot_in_ref = set(corresp_array[:, 1].tolist())
    all_spot_in_ref = set(range(len(corresp_array)))

    isolated_spots_in_reflist = all_spot_in_ref - associated_spot_in_ref

    (data_theta, Chi, posx, posy, dataintensity) = resorted_data.T

    return (data_theta, Chi, posx, posy, dataintensity,
            isolated_spots_in_spotlist,
            isolated_spots_in_reflist)

