# -*- coding: utf-8 -*-
"""
module of lauetools project

purposes:

- classical method to index laue spots
- gnomonic projection
- hough transform (in development, feasability demonstrated)
- image matching (in development, feasability demonstrated)
- zone axes recognition (in development)

js micha June  2014
"""

__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os

import pylab as p

import numpy as np

import lauecore as LAUE
import CrystalParameters as CP
import findorient as FindO

import dict_LaueTools as DictLT
import generaltools as GT
import IOLaueTools as IOLT

import indexingSpotsSet as ISS

try:
    import angulardist
    USE_CYTHON = True
except ImportError:
    print("Cython compiled module for fast computation of angular distance is not installed!")
    USE_CYTHON = False

try:
    import wx
except:
    print("wx is not installed! Could be some trouble from this lack...")
    pass

#--- ------------ CONSTANTS
DEG = np.pi / 180.
CST_ENERGYKEV = DictLT.CST_ENERGYKEV

#--- -------------  PROCEDURES
def stringint(k, n):
    """ returns string of k by placing zeros before to have n characters
    ex: 1 -> '0001'
    15 -> '0015'
    
    # TODO: replace by string format %04d
    """
    strint = str(k)
    res = '0' * (n - len(strint)) + strint
    return res


def Plot_compare_2thetachi(Angles, twicetheta_data, chi_data,
                            verbose=1,
                            key_material=14,
                            emax=25, emin=5,
                            EULER=0,
                            exp_spots_list_selection=None):
    """
    plot data and simulation (given by list of 3 angles for orientation)
    in 2theta chi space (kf vector angles)
    """

    angle_X, angle_Y, angle_Z = Angles

    if type(EULER) != type(np.array([1, 2, 3])):
        if EULER == 0:
            mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
        elif EULER == 1:
            mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        if verbose:
            print("Using orientation Matrix for plotting")
        mymat = EULER

    # PATCH to use correctly getLaueSpots() of laue6
    grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)


    # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1,
                                 fastcompute=1, fileOK=0, verbose=0)

    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    print("nb of spots in CCD frame", len(TwicethetaChi[0]))

    if exp_spots_list_selection is not None:  # to plot only selected list of exp. spots
        sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
        sel_chi = np.array(chi_data)[exp_spots_list_selection]

    p.title('Euler Angles [%.1f,%.1f,%.1f]' % (tuple(Angles)))
    if exp_spots_list_selection is not None:
        p.scatter(sel_2theta, sel_chi, s=40, c='w', marker='o', faceted=True, alpha=.5)
    else:
        p.scatter(twicetheta_data, chi_data, s=40, c='w', marker='o', faceted=True, alpha=.5)
    p.scatter(TwicethetaChi[0], TwicethetaChi[1], c='r', faceted=False)

    p.show()


def Plot_compare_2thetachi_multi(list_Angles, twicetheta_data, chi_data,
                                verbose=1,
                                emax=25, emin=5,
                                key_material=14,
                                arraysize=(600, 720),
                                EULER=0,
                                exp_spots_list_selection=None,
                                title_plot='default',
                                figsize=(6, 6),
                                dpi=80):
    """ up to 9
    only for test or development
    Warning: blindly corrected 
    """
    fig = p.figure(figsize=figsize, dpi=dpi)  # ? mouais mais dans savefig c'est ok!

    nb_of_orientations = len(list_Angles)
    if nb_of_orientations == 1:
        codefigure = 111
    if nb_of_orientations == 2:
        codefigure = 211
    if nb_of_orientations in (3, 4):
        codefigure = 221
    if nb_of_orientations in (5, 6):
        codefigure = 321
    if nb_of_orientations in (7, 8, 9):
        codefigure = 331
    index_fig = 0
    for orient_index in list_Angles:
        if type(EULER) != type(np.array([1, 2, 3])):
            if EULER == 0:
                mymat = GT.fromelemangles_toMatrix(list_Angles[orient_index])
            elif EULER == 1:
                mymat = GT.fromEULERangles_toMatrix(list_Angles[orient_index])
        else:
            mymat = EULER[orient_index]
            if verbose:
                print("Using orientation Matrix for plotting")
                print("mymat", mymat)

        # PATCH to use correctly getLaueSpots() of laue6
        grain = CP.Prepare_Grain(key_material, OrientMatrix=mymat)

        # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grain], 1, fastcompute=1, fileOK=0, verbose=0)
        # 2theta,chi of spot which are on camera (with harmonics)
        TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

        if exp_spots_list_selection is None:  # to plot all exp. spots
            sel_2theta = np.array(twicetheta_data)
            sel_chi = np.array(chi_data)
        elif type(exp_spots_list_selection) == type(np.array([1, 2, 3])):
            sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
            sel_chi = np.array(chi_data)[exp_spots_list_selection]
        elif type(exp_spots_list_selection) == type(5):  # it is a number # for plotting
            if exp_spots_list_selection > 1:
                ind_max = min(len(twicetheta_data) - 1, exp_spots_list_selection)
                sel_2theta = np.array(twicetheta_data)[:exp_spots_list_selection]
                sel_chi = np.array(chi_data)[:exp_spots_list_selection]

        ax = fig.add_subplot(codefigure)
        if type(title_plot) == type([1, 2]):
            sco = title_plot[index_fig]
            p.title('nb close,<0.5deg: %d,%d  mean ang %.2f' % tuple(sco))
        else:
            if type(EULER) != type(np.array([1, 2, 3])):
                if EULER == 1:
                    p.title('Euler Angles [%.1f,%.1f,%.1f]' % (tuple(list_Angles[orient_index])))
            else:
                p.title('Orientation Matrix #%d' % orient_index)

        ax.set_xlim((35, 145))
        ax.set_ylim((-45, 45))
        # exp spots
        ax.scatter(sel_2theta, sel_chi, s=40, c='w', marker='o',
                                    faceted=True, alpha=.5)
        # theo spots
        ax.scatter(TwicethetaChi[0], TwicethetaChi[1], c='r', faceted=False)
        if index_fig < nb_of_orientations:
            index_fig += 1
            codefigure += 1

    p.show()


def correctangle(angle):
    """
    shift angle in between 0 and 360
    TODO: useful for zone axis recognition ?
    """
    res = angle
    if angle < 0.:
        res = 180. + angle
    return res


def findInArray1D(array1D, val, tol):
    """
    return array of index where val has been found in input_array within tol as
    tolerance value
    #TODO: to put outside in general module to deal with list and array ?
    """
    return np.where(abs(array1D - val) <= tol)[0]


def gen_Nuplets(items, n):
    """
    generator taking n-uplet from items, and
    """
    if n == 0:
        yield []
    else:
        for i in list(range(len(items) - n + 1)):
            for cc in gen_Nuplets(items[i + 1:], n - 1):
                yield [items[i]] + cc


def ReadASCIIfile(_filename_data, col_2theta=0, col_chi=1, col_Int=-1, nblineskip=1):
    """ from a file
    return 3 arrays of columns located at index given by
    col_2theta=0, col_chi=1, col_Int=-1:
    [0] theta
    [1] chi
    [2] intensity

    TODO: to move into readwriteASCII module
    """
    _tempdata = np.loadtxt(_filename_data, skiprows=nblineskip)
    # _tempdata = scipy.io.array_import.read_array(_filename_data, lines = (nblineskip,-1))

    _data_theta = _tempdata[nblineskip - 1:, col_2theta] / 2.
    _data_chi = _tempdata[nblineskip - 1:, col_chi]

    try:
        _data_I = _tempdata[nblineskip - 1:, col_Int]
    except IndexError:
        print("there are not 5 columns in data.cor file")
        print("I create then a uniform intensity data!")
        _data_I = np.ones(len(_data_theta))
    if (np.array(_data_I) < 0.).any():
        print("Strange ! I don't like negative intensity...")
        print("I create then a uniform intensity data!")
        _data_I = np.ones(len(_data_theta))

    return (_data_theta, _data_chi, _data_I)

# keep track of old name for external import
createrawdata = ReadASCIIfile


def calculdist_from_thetachi(listpoints1, listpoints2):
    """
    mutual angular distance from two arrays of spots:

    listpoints1  : array of pairs (THETA, CHI)
    listpoints2  : array of pairs (THETA, CHI)

    return:

    tab_angulardist:
    matrix of all mutual angular distance

    NOTE: It would be necessary later to remove duplicates 
    since there are pairs that have the same second value

    TODO: still used, see similar function in generaltools module
    """
    data1 = np.array(listpoints1)
    data2 = np.array(listpoints2)
    # print "data1",data1
    # print "data2",data2
    longdata1 = data1[:, 0]  # theta
    latdata1 = data1[:, 1]  # chi

    longdata2 = data2[:, 0]  # theta
    latdata2 = data2[:, 1]  # chi

    deltalat = latdata1 - np.reshape(latdata2, (len(latdata2), 1))
    longdata2new = np.reshape(longdata2, (len(longdata2), 1))
    prodcos = np.cos(longdata1 * DEG) * np.cos(longdata2new * DEG)
    prodsin = np.sin(longdata1 * DEG) * np.sin(longdata2new * DEG)

    arccos_arg = np.around(prodsin + prodcos * np.cos(deltalat * DEG), decimals=9)

    tab_angulardist = (1. / DEG) * np.arccos(arccos_arg)

    return tab_angulardist


def getArgmin(tab_angulardist):
    """
    temporarly doc
    from matrix of mutual angular distances return index of closest neighbour

    TODO: to explicit documentation as a function of tab_angulardist properties only
    """
    return np.argmin(tab_angulardist, axis=1)


def find_key(mydict, num):
    """ only for single value
    (if value is a list : type if num in mydict[k] )
    """
    for k in list(mydict.keys()):
        if num == mydict[k]:
            return k


#--- -------------------  ANGLES LUT INDEXING
def return_index(tup):
    """
    test if second element of tup is not empty
    """
    toreturn = False
    if len(tup[1]) > 0:
        toreturn = True
    return toreturn


def Possible_planes(angles_value, tole=0.2, verbose=1, onlyclosest=1):
    """
    for a given angular distance and tolerance (in degrees)
    returns possible pairs of planes
    (among the first most important 100 110 111 210 211 221 310 311 321)

    TODO: possiblity of rejecting some plane type according the structure fcc dia...)
    """

    # LUT_MAIN_CUBIC comes from findorient

    ind_sorted_LUT_MAIN_CUBIC = [np.argsort(elem) for elem in FindO.LUT_MAIN_CUBIC]
    sorted_table_angle = []
    for k in list(range(len(ind_sorted_LUT_MAIN_CUBIC))):
        # print len(LUT_MAIN_CUBIC[k])
        # print len(ind_sorted_LUT_MAIN_CUBIC[k])
        sorted_table_angle.append((FindO.LUT_MAIN_CUBIC[k])[ind_sorted_LUT_MAIN_CUBIC[k]])

    if onlyclosest:
        one_value = [GT.find_closest(np.array(elem), np.array([angles_value]), tole)[:2] for elem in sorted_table_angle]
        # print "gjhgjh",one_value

        sol_one_value = []
        for k in list(range(len(one_value))):
            if return_index(one_value[k]):
                if verbose:
                    print("k", k, "  sortedindex ", one_value[k][0][0])
                sol_one_value.append([ k, ind_sorted_LUT_MAIN_CUBIC[k][one_value[k][0][0]]])
        # LUT_MAIN_CUBIC[a,b] donne l'angle avec sol_one_value=[[a,b],[a2,b2],...]
        # a est l'indice du plan central, b est l'indice du plan

        # reading the solution
        # print sol_one_value
        planes_sol = []
        for m in list(range(len(sol_one_value))):
            first_plane_sol = FindO.convplanetypetoindice(FindO.INVDICOPLANE[sol_one_value[m][0]])
            second_plane_sol = FindO.DICOLISTNEIGHBOURS[FindO.INVDICOPLANE[sol_one_value[m][0]]][sol_one_value[m][1]]
            if verbose:
                print("----------------------------------------")
                print("sol", sol_one_value[m])
                print("index of central plane", sol_one_value[m][0])
                print("corresponding type", first_plane_sol)
                print("index of the type of neighbouring planes", sol_one_value[m][1])
                print("corresponding type", second_plane_sol)
                # print "all possibles", DICOLISTNEIGHBOURS[INVDICOPLANE[sol_one_value[m][0]]]
            planes_sol.append([first_plane_sol, second_plane_sol])

    else:  # Not only closest angle in LUT is found but many distances can be found within tolerance

        values = [findInArray1D(np.array(elem), angles_value, tole) for elem in sorted_table_angle]
        sol_one_value = []
        for k in list(range(len(values))):
            if len(values[k]):
                for ind in values[k]:
                    sol_one_value.append([k, ind_sorted_LUT_MAIN_CUBIC[k][ind]])

        planes_sol = []
        for m in list(range(len(sol_one_value))):
            first_plane_sol = FindO.convplanetypetoindice(FindO.INVDICOPLANE[sol_one_value[m][0]])
            second_plane_sol = FindO.DICOLISTNEIGHBOURS[FindO.INVDICOPLANE[sol_one_value[m][0]]][sol_one_value[m][1]]
            if verbose:
                print("----------------------------------------")
                print("sol", sol_one_value[m])
                print("index of central plane", sol_one_value[m][0])
                print("corresponding type", first_plane_sol)
                print("index of the type of neighbouring planes", sol_one_value[m][1])
                print("corresponding type", second_plane_sol)
                # print "all possibles", DICOLISTNEIGHBOURS[FindO.INVDICOPLANE[sol_one_value[m][0]]]
            planes_sol.append([first_plane_sol, second_plane_sol])

    return planes_sol


def plane_type_attribution(spot_index_1, angulartolerance, table_angdist):
    """
    from 1 spot (given by its index as defined in angulardisttable[1])
    returns:
    [0] : list of [spots_index_1,spot_index_2]
                    where spot_index_2 is the index spot lying
                    at the tabulated distance of spot 1
    [1] : list of corresponding plane type: [plane type 1,plane 2]

    NB: plane type 1 is a family type (Miller indices always positive and arbitrarly ordered)
    instead of plane 2 which is an accurate plane type (order and sign is important with respect to the famiiy type 1)
    """

    from_spot = [Possible_planes(angle,
                                                  tole=angulartolerance,
                                                  verbose=0) for angle in table_angdist[spot_index_1]]

    couples_index = []
    couples_type = []

    for m in list(range(len(from_spot))):
        nbcouples = len(from_spot[m])
        if nbcouples > 0:
            couples_index.append([spot_index_1, m])
            couples_type.append(from_spot[m])

    return couples_index, couples_type


def plane_type_attribution_twospots(spot_index_1, spot_index_2,
                                    angulartolerance, table_angdist):
    """
    from 2 spots (given by their index as defined in angulardisttable[1])
    returns:
    array:
    [0] list of possible plane type for spot1 (family type)
    [1] list of possible plane type for spot2  (true type)

    NB: plane type 1 is a family type 
        (Miller indices always positive and arbitrarly ordered)
    instead of plane 2 which is an accurate plane type
        (order and sign is important with respect to the famiiy type 1)
        
    TODO: apparently NOT used, to delete ?
    """
    angle = table_angdist[spot_index_1][spot_index_2]
    from_spot = Possible_planes(angle, tole=angulartolerance, verbose=0)
    couples_type = []
    for m in list(range(len(from_spot))):
        nbcouples = len(from_spot[m])
        if nbcouples > 0:
            couples_type.append(from_spot[m])
    # print shape(np.array(  couples_type))
    # print couples_type
    # print np.transpose(np.array(  couples_type),(1,0,2))
    return np.transpose(np.array(couples_type), (1, 0, 2))


def twospots_recognition(spot_1, spot_2, angulartolerance, onlyclosest=1):
    """
    from 2 spots (given by their theta,chi coordinates)
    returns:
    array:
    [0] list of possible plane type for spot1 (family type)
    [1] list of possible plane type for spot2  (true type)

    NB: plane type 1 is a family type (Miller indices always positive and arbitrarly ordered)
    instead of plane 2 which is an accurate plane type
    (order and sign is important with respect to the family type 1, to have the correct angle)
    """
    listspot = np.array([spot_1, spot_2])
    angle = calculdist_from_thetachi(listspot, listspot)[0, 1]
    from_spot = Possible_planes(angle, tole=angulartolerance, verbose=1, onlyclosest=onlyclosest)
    couples_type = []
    for m in list(range(len(from_spot))):
        nbcouples = len(from_spot[m])
        if nbcouples > 0:
            couples_type.append(from_spot[m])
    # print shape(np.array(  couples_type))
    # print couples_type
    # print np.transpose(np.array(  couples_type),(1,0,2))
    if len(couples_type) > 0:
        return np.transpose(np.array(couples_type), (1, 0, 2))
    else:
        return []


def matrices_from_onespot(spot_index, ang_tol, table_angdist, Theta, Chi,
                          verbose=0):
    """
    from one spot of index spot_index,
    ang_tol angular tolerance (deg) for look up table recognition,
    table_angdist all experimental distances
    Theta and Chi are the experimental Theta Chi spots coordinates
    """
    pairspots = []
    matrix_list = []
    pairplanes = []

    possible_couplespots, possible_coupleplanes = plane_type_attribution(spot_index,
                                                                            ang_tol,
                                                                            table_angdist)
    # print possible_couplespots
    # print possible_coupleplanes

    nb_couplespots = len(possible_couplespots)

    for cs_index in list(range(nb_couplespots)):

        spot_index_2 = possible_couplespots[cs_index][1]
        if verbose:
            print(" ***********   cs_nindex", cs_index)
            print("couple spots", possible_couplespots[cs_index])
            print("spot2 index", spot_index_2)
        nb_coupleplanes_for_this_cs = len(possible_coupleplanes[cs_index])

        for pp_index in list(range(nb_coupleplanes_for_this_cs)):

            plane_1 = possible_coupleplanes[cs_index][pp_index][0]
            plane_2 = possible_coupleplanes[cs_index][pp_index][1]
            matrix = FindO.givematorient(plane_1, [2 * Theta[spot_index], Chi[spot_index]], plane_2, [2 * Theta[spot_index_2], Chi[spot_index_2]], verbose=0)

            matrix_list.append(matrix)
            pairplanes.append([plane_1, plane_2])
            pairspots.append(possible_couplespots[cs_index])

            if verbose:
                print("in matrices_from_onespot")
                print("pp_index", pp_index)
                print("couple planes ", plane_1, plane_2)
                print([2 * Theta[spot_index], Chi[spot_index]], [2 * Theta[spot_index_2], Chi[spot_index_2]])
                print("spot_index_2", spot_index_2)
                print("matrix", matrix)

    return matrix_list, pairplanes, pairspots


def matrices_from_onespot_hkl(spot_index, LUT_tol_angle, table_angdist,
                                twiceTheta_exp, Chi_exp,
                                n, key_material, MaxRadiusHKL=False,
                                hkl1=FindO.HKL_CUBIC_UP3,
                                hkl2=None,
                                LUT=None,
                                allow_restrictedLUT=False,
                                verbose=1):
    """
    get all possibles UBs from one central spot and given its hkl1 miller indices

    spot_index_central      : integer index
    hkl1                    :  array or list of 3 elements
    table_angdist        : mutual angular distances between spots n*n symetric table 
    LUT_tol_angle          : angular tolerance for computing the number of matching simulated
                        spots with exp. spots

    twiceTheta_exp, Chi_exp  : experimental spots angles
    n                        : angles reference LUT table order
    key_material            : string label for material

    hkl2                    : list of hkl2 to compute the LUT with hkl1

    allow_restrictedLUT           : flag to restrict LUT (removed hkl1 with negative l)

    return: - like matrices_from_onespot_new() -
    (list_orient_matrix,
         planes,
         pairspots)
    """
    Distances_from_central_spot = table_angdist[spot_index]

    PPs_list = []

    latticeparams = DictLT.dict_Materials[key_material][1]
    B = CP.calc_B_RR(latticeparams)

    if hkl1 is not None:
        hkl1 = np.array(hkl1)
        print("hkl1 in matrices_from_onespot_hkl()", hkl1)

    if allow_restrictedLUT:
        # LUT restriction given by crystal structure
        allow_restrictedLUT = CP.isCubic(latticeparams)

    if LUT is None:
        if hkl2 is None:
            print("Computing hkl2 list for specific or cubic LUT in matrices_from_onespot_hkl()")
            # compute hkl2 outside loop
            hkl_all = GT.threeindices_up_to(n, remove_negative_l=allow_restrictedLUT)

            if 1:  # filterharmonics:
                hkl_all = FindO.FilterHarmonics(hkl_all)

            hkl2 = hkl_all
    else:
        # LUT will be used in next calculations and not recomputed
        print("Using specific LUT in matrices_from_onespot_hkl()")
        pass

    for spotindex_2, query_angle in enumerate(Distances_from_central_spot):
        if verbose:
            print("\n-*****----------------------------------------------------")
            print("k,angle = ", spotindex_2, query_angle)
            print("-*****----------------------------------------------------\n")
        # hkls, LUT = PlanePairs_from2sets(angle, ang_tol,
                                    # GT.Positiveindices_up_to(n), GT.Positiveindices_up_to(n), Gstar,
                                    # onlyclosest = 1, filterharmonics = 1,verbose = 1)
        # hkls = PlanePairs(angle,
                                        # ang_tol,
                                        # Gstar,
                                        # n,
                                        # onlyclosest = 1, filterharmonics = 1,
                                        # verbose = 0) # LUT is computed inside !

        hkls, LUT = FindO.PlanePairs_from2sets(query_angle, LUT_tol_angle, hkl1,
                                          hkl2, key_material,
                                          LUT=LUT,
                                          onlyclosest=0,
                                          filterharmonics=1,
                                          verbose=verbose)

        if hkls is not None and (spot_index != spotindex_2):
            nbpairs = len(hkls)
            PPs_list.append([hkls, spotindex_2, nbpairs])
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spotindex_2, nbpairs)

    coords_exp = np.array([twiceTheta_exp, Chi_exp]).T
    coord_central_spot = coords_exp[spot_index]

#     print "PPs_list in matrices_from_onespot_hkl", PPs_list

    Matrices_Res = Loop_on_PlanesPairs_and_Get_Matrices(PPs_list, spot_index, coord_central_spot,
                                                coords_exp, B,
                                                verbose=verbose)

    return Matrices_Res, hkl2, LUT


def matrices_from_onespot_new(spot_index,
                              ang_tol,
                              table_angdist,
                              twiceTheta,
                              Chi,
                              n,
                              B,
                              LUT=None,
                              MaxRadiusHKL=False,
                              verbose=0):
    """
    returns list of pair of planes and exp pairs of spots that match an angle in a reference LUT.
    LUT is computed from B (Gstar)
    
    USED in automatic indexation
    Used AutoIndexation module


    spot_index            : index of spot considered (must be lower than len(table_angdist) )
    ang_tol                : angular tolerance (deg) for look up table matching
    table_angdist         : all experimental distances square matrix
    2Theta and Chi         : the experimental 2Theta Chi spots coordinates

    For building the angles reference LUT:

    n                    : integer value corresponding the maximum miller index considered in angle value LUT
    B                    : Bmatrix of reference unit cell used TRIANGULAR UP
                            for extracting lattice parameter and building LUT

    orientation matrix is then given with respect the frame in which is expressed B
    """
    # possible_couplespots, possible_coupleplanes = plane_type_attribution(spot_index,
                                                                            # ang_tol,
                                                                            # table_angdist)

    Distances_from_central_spot = table_angdist[spot_index]

    PPs_list = []

    if LUT is None:
        print("LUT build in matrices_from_onespot_new()")
        print("cubicSymmetry is False for an exhaustive LUT")
        LUT = build_AnglesLUT(B, n, MaxRadiusHKL=MaxRadiusHKL,
                              cubicSymmetry=False)

    for spotindex_2, angle in enumerate(Distances_from_central_spot):
        if verbose:
            print("\n-*-*-*----------------------------------------------------")
            print("k,angle = ", spotindex_2, angle)
            print("-*-*-*----------------------------------------------------\n")
        # hkls, LUT = PlanePairs_from2sets(angle, ang_tol,
                                    # GT.Positiveindices_up_to(n), GT.Positiveindices_up_to(n), Gstar,
                                    # onlyclosest = 1, filterharmonics = 1,verbose = 1)
        # hkls = PlanePairs(angle,
                                        # ang_tol,
                                        # Gstar,
                                        # n,
                                        # onlyclosest = 1, filterharmonics = 1,
                                        # verbose = 0) # LUT is computed inside !

        hkls = FindO.PlanePairs_2(angle,
                                  ang_tol,
                                  LUT,
                                  onlyclosest=0,
                                  verbose=verbose)  # LUT is provided !

        if hkls is not None and (spot_index != spotindex_2):
            nbpairs = len(hkls)
            PPs_list.append([hkls, spotindex_2, nbpairs])
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spotindex_2, nbpairs)

    coords_exp = np.array([twiceTheta, Chi]).T
    coord_central_spot = coords_exp[spot_index]

    return Loop_on_PlanesPairs_and_Get_Matrices(PPs_list, spot_index, coord_central_spot,
                                                coords_exp, B, verbose=verbose)


def getUBs_and_MatchingRate(spot_index_1, spot_index_2, ang_tol_LUT,
                            angdist, coords_1, coords_2,
                            n, B,
                            twiceTheta_exp, Chi_exp,
                            set_hkl_1=None,
                            key_material=None,
                            emax=None,
                            ResolutionAngstrom=None,
                            ang_tol_MR=0.5,
                            detectorparameters=None,
                            LUT=None,
                            MaxRadiusHKL=False,
                            verbose=0,
                            verbosedetails=True,
                            Minimum_Nb_Matches=6,
                            worker=None):
    """
    from two spots only
    USED in manual indexation
    """

    MAX_NB_SOLUTIONS = 30

    List_UBs = []  # matrix list
    List_Scores = []  # hall of fame BestScores_per_centralspot list

    (list_orient_matrix,
     planes,
     pairspots) = UBs_from_twospotsdistance(spot_index_1,
                              spot_index_2,
                              ang_tol_LUT,
                              angdist,
                              coords_1,
                              coords_2,
                              n,
                              B,
                              LUT=LUT,
                              set_hkl_1=set_hkl_1,
                              key_material=key_material,
                              MaxRadiusHKL=MaxRadiusHKL,
                              verbose=verbose)

    solutions_matorient_index = []
    solutions_spotscouple = []
    solutions_hklcouple = []
    solutions_matchingscores = []
    solutions_matchingrate = []

    if verbose:
        print("len(list_orient_matrix)", len(list_orient_matrix))
        # print "key_material",key_material
        # print "\n"
        print("#mat nb<%.2f       nb. theo. spots     mean       max    nb**2/nb_theo*mean     plane indices" % \
                                                    (ang_tol_MR))
    WORKEREXIST = 0
    if worker is not None:
        WORKEREXIST = 1
    # loop over orient matrix given from LUT recognition for one central spot
    nb_UB_matrices = len(list_orient_matrix)
    for mat_ind in list(range(nb_UB_matrices)):
        if WORKEREXIST:
#             print "there is a worker !!"
            if worker._want_abort:
                print("\n\n!!!!!!! Indexation Aborted \n\n!!!!!!!\n")
                BestScores_per_centralspot = np.array([])
                worker.callbackfct(None)
                return
        if (mat_ind%10)==0:       
            print('Calculating matching with exp. data for matrix #%d / %d' % (mat_ind, nb_UB_matrices))
        # compute matching rate and store if high
        AngRes = ISS.Angular_residues_np(list_orient_matrix[mat_ind],
                                    twiceTheta_exp, Chi_exp,
                                    key_material=key_material,
                                    emax=emax,
                                    ResolutionAngstrom=ResolutionAngstrom,
                                    ang_tol=ang_tol_MR,
                                    detectorparameters=detectorparameters)

        if AngRes is None:
            continue

        (allres, resclose, nbclose,
         nballres,
         mean_residue, max_residue) = AngRes

        if nbclose > Minimum_Nb_Matches:
            std_closematch = np.std(allres[allres < ang_tol_MR])
            if verbosedetails:
#                     print "%d        %d              %d             %.3f       %.3f        %.3f" % \
#                             (mat_ind, nbclose, nballres, mean_residue, max_residue, nbclose ** 2 * 1. / nballres * mean_residue), "    ", str(planes[mat_ind]), \
#                             "  ", pairspots[mat_ind]

                mean_residue_closematch = np.mean(allres[allres < ang_tol_MR])
                max_residue_closematch = np.max(allres[allres < ang_tol_MR])
                print("mat_ind      nbclose      fullnb      std_closematch     mean_resid  max_resid  figmerit")
                print("%d        %d       %d       %.3f      %.3f       %.3f        %.3f" % \
                        (mat_ind, nbclose, nballres,
                         std_closematch, mean_residue_closematch, max_residue,
                         nbclose ** 2 * 1. / nballres / std_closematch), "    ", str(planes[mat_ind]), \
                        "  ", pairspots[mat_ind])

#             print "mat_ind: %d" % mat_ind
#             print "AngRes", AngRes
            solutions_matorient_index.append(mat_ind)
            solutions_spotscouple.append(pairspots[mat_ind])
            solutions_hklcouple.append(planes[mat_ind])
            solutions_matchingscores.append([nbclose, nballres, std_closematch])
#             solutions_matchingscores.append([nbclose, nballres, mean_residue])
            solutions_matchingrate.append(100. * nbclose / nballres)
            # print list_orient_matrix[mat_ind]

    BestScores_per_centralspot = np.array(solutions_matchingscores)

    # for one central spot if there are at least one potential solution
    if len(BestScores_per_centralspot) > 0:

#             print "Got one solution for k_centspot_index: %d" % k_centspot_index

        # sort results
        rank = np.lexsort(keys=\
              (BestScores_per_centralspot[:, 2],
               BestScores_per_centralspot[:, 0] * 1. / \
                    BestScores_per_centralspot[:, 1]))[::-1]

        hall_of_fame = BestScores_per_centralspot[rank]

        orient_index_fame = np.array(solutions_matorient_index)[rank]
        psb = np.array(solutions_spotscouple)[rank]
        ppb = np.array(solutions_hklcouple)[rank]

        # only for plot
        list_UBs_for_plot = np.array(list_orient_matrix)

#             print "orient_index_fame", orient_index_fame

        # print "list_UBs_for_plot",list_UBs_for_plot

        # for search from many spots (<=9)
        # one keeps only the best matrix

        for mm in list(range(min(MAX_NB_SOLUTIONS, len(orient_index_fame)))):
            bestmatrix = list_UBs_for_plot[orient_index_fame[mm]]
            bestscores = list(hall_of_fame[mm]) + list([psb[mm]]) + list([ppb[mm]])
#                 print "bestmatrix", bestmatrix
#                 print "bestscores", bestscores
            List_UBs.append(bestmatrix)
            List_Scores.append(bestscores)

    else:
        print("No orientation matrix found with nb of matches larger than %d" % \
                            Minimum_Nb_Matches)
        if verbose:
            print("Try to:")
            print("- decrease Ns (MNMS: minimum number of matched spots)")
            print("- increase angular tolerances (distance recognition and/or matching)")
            print("- increase nbofpeaks (ISSS: Intense spot set size)")
            print("- increase Energy max\n")
            
    if WORKEREXIST:
        worker.fctOutputResults = List_UBs, List_Scores

        print("finished!")
        print("setting worker.fctOutputResults to", worker.fctOutputResults)
        worker.callbackfct('COMPLETED')

    return List_UBs, List_Scores


def UBs_from_twospotsdistance(spot_index_1,
                              spot_index_2,
                              angle_tol,
                              exp_angular_dist,
                              coords_1,
                              coords_2,
                              n,
                              B,
                              LUT=None,
                              set_hkl_1=None,
                              set_hkls_2=None,
                              key_material=None,
                              MaxRadiusHKL=False,
                              allow_restrictedLUT=True,
                              verbose=0):
    """
    returns list of pair of planes and exp pairs of spots that match an angle in a reference LUT.
    LUT is computed from B (Gstar)
    
    USED in manual indexation


    spot_index            : index of spot considered (must be lower than len(table_angdist) )
    angle_tol                : angular tolerance (deg) for look up table matching
    exp_angular_dist         : experimental distance between q1,q2 (lattice planes normals)
    coords_exp         : the experimental 2Theta Chi spots coordinates

    For building the angles reference LUT:

    n                    : integer value corresponding the maximum miller index considered in angle value LUT
    B                    : Bmatrix of reference unit cell used TRIANGULAR UP
                            for extracting lattice parameter and building LUT

    orientation matrix is then given with respect the frame in which is expressed B

    set_hkl_1     : if not None, [h,k,l] of spot #1
    """

    print("using UBs_from_twospotsdistance()")
    print("coords_1", coords_1)
    print("coords_2", coords_2)
    PPs_list = []

    if set_hkl_1 is None:

        if LUT is None:
            print("LUT build in UBs_from_twospotsdistance()")
            LUT = build_AnglesLUT(B, n, MaxRadiusHKL=MaxRadiusHKL,
                                cubicSymmetry=CP.hasCubicSymmetry(key_material))
#                                   cubicSymmetry=False)

        hkls = FindO.PlanePairs_2(exp_angular_dist,
                                  angle_tol,
                                  LUT,
                                  onlyclosest=0,
                                  verbose=verbose)  # LUT is provided !

#         print "nb of hkls found in LUT:", len(hkls)
#         print "hkls:", hkls

    # when hkl1 is guessed (and set)
    elif set_hkl_1 is not None:
        latticeparams = DictLT.dict_Materials[key_material][1]
        B = CP.calc_B_RR(latticeparams)

        if allow_restrictedLUT:
            # LUT restriction given by crystal structure
            allow_restrictedLUT = CP.isCubic(latticeparams)

        if set_hkls_2 is None:
            print("Computing hkl2 list for specific or cubic LUT in UBs_from_twospotsdistance()")
            # compute hkl2 outside loop:
            hkl_all = GT.threeindices_up_to(n, remove_negative_l=allow_restrictedLUT)

            if 1:  # filterharmonics:
                hkl_all = FindO.FilterHarmonics(hkl_all)

            hkl2 = hkl_all
            
            print("new calculated hkl2")

        hkls, LUTspecific = FindO.PlanePairs_from2sets(exp_angular_dist, angle_tol, set_hkl_1,
                                          hkl2, key_material,
                                          LUT=None,
                                          onlyclosest=0,
                                          filterharmonics=1,
                                          verbose=verbose)
        
        print("found planes pairs")

    if hkls is not None and (spot_index_1 != spot_index_2):
            nbpairs = len(hkls)
            PPs_list.append([hkls, spot_index_2, nbpairs])
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spot_index_2, nbpairs)

    #     print "PPs_list", PPs_list

    return Loop_on_PlanesPairs_and_Get_Matrices(PPs_list, spot_index_1, coords_1,
                                                    coords_2, B, verbose=verbose,
                                                    single_coords_2=True)



def Loop_on_PlanesPairs_and_Get_Matrices(PP_list, spot_index, coord1, coords, B,
                                         verbose=0, single_coords_2=False):
    """
    loop on possible planes couples (PP) from recognised distances from a single spot (spotindex)

    USED in manual indexation

    coords  :  2theta chi spots coordinates 
    """
    pairspots = []
    matrix_list = []
    pairplanes = []

    currentspotindex = -1

    # loop over all possible pairs of planes found in LUT
    for k, PP in enumerate(PP_list):

        hkls, spotindex_2, nn = PP

        if spotindex_2 != currentspotindex:
            print("Looking up planes pairs in LUT from exp. spots (%d, %d): "%(spot_index,spotindex_2))
            currentspotindex = spotindex_2

        if single_coords_2:
            coord2 = coords
        else:
            coord2 = coords[spotindex_2]

        hlks_shape = hkls.shape

        if len(hlks_shape) == 2:
            nb_pairs = hlks_shape[0] / 2
        elif len(hlks_shape) == 3:
            nb_pairs = hlks_shape[0]

        if nb_pairs == 1:
            hkls = [hkls]

        if verbose:
            print("\n************** k", k)
            print("PP[%d] = " % k, PP)
            print('hlks_shape', hlks_shape)
            print('hkls', hkls)
            print("nb_pairs", nb_pairs)

        for planepair in hkls:

            hkl1, hkl2 = planepair

#             print "hkl1, hkl2 ", hkl1, hkl2
            # print "coord1,coord2", coord1,coord2
            # print "spot_index, spotindex_2",spot_index, spotindex_2

            matrix = FindO.OrientMatrix_from_2hkl(hkl1, coord1,
                                            hkl2, coord2,
                                            B,
                                            verbose='no',
                                            frame='lauetools')

            # print "matrix",matrix

            # matrix=givematorient(plane_1,[2*Theta[spot_index],Chi[spot_index]],plane_2,[2*Theta[spot_index_2],Chi[spot_index_2]],verbose=0)

            matrix_list.append(matrix)
            pairplanes.append([hkl1, hkl2])
            pairspots.append([spot_index, spotindex_2])

            # ---compute matrix by swaping hkl1 and hkl2

            matrix = FindO.OrientMatrix_from_2hkl(hkl2, coord1,
                                                hkl1, coord2,
                                                B,
                                                verbose='no',
                                                frame='lauetools')

            # print "matrix",matrix

            # matrix=givematorient(plane_1,[2*Theta[spot_index],Chi[spot_index]],plane_2,[2*Theta[spot_index_2],Chi[spot_index_2]],verbose=0)

            matrix_list.append(matrix)
            pairplanes.append([hkl2, hkl1])
            pairspots.append([spot_index, spotindex_2])

            if verbose:
                print("in matrices_from_onespot_new")
                print("pair of lattice planes ", hkl1, hkl2)
                print([coord1, coord2])
                print("spot_index_2", spotindex_2)
                print("matrix", matrix)

    return matrix_list, pairplanes, pairspots


def getOrientMatrix_from_onespot(spot_index,
                              ang_tol,  # angular toleance for distance recognition
                              table_angdist,
                              twiceTheta,
                              Chi,
                              n,
                              B,
                              cubicSymmetry=True,
                              LUT=None,
                              ResolutionAngstrom=False,
                              MatchingThresholdStop=100,
                              key_material=None,
                              emax=None,
                              MatchingRate_Angle_Tol=None,  # angular tolerance for matching rate calculation
                              detectorparameters=None,
                              verbose=0):
    """
    TODO: to delete only used in multigrain.py

    returns list of pair of planes and exp pairs of spots that match an angle in a reference LUT.
    LUT is computed from Gstar


    spot_index            : index of spot considered (must be lower than len(table_angdist) )
    ang_tol                : angular tolerance (deg) for look up table recognition 
    table_angdist         : all experimental distances square matrix
    Theta and Chi         : the experimental Theta Chi spots coordinates

    For building the angles reference LUT:

    n                    : integer value corresponding the maximum miller index considered in angle value LUT
    B                    : Bmatrix of reference unit cell used TRIANGULAR UP
                            for extracting lattice parameter and building LUT

    orientation matrix is then given with respect the frame in which is expressed B


    detectorparameters      : dictionary of detector parameters (key, value) which must contain
                            'kf_direction' , general position of detector plane
                            'detectordistance', detector distance (mm)
                            'detectordiameter', detector diameter (mm)

    TODO: to make it more compact !! there are too much copy-paste
    TODO: in the main loop should start at spot_index +1   !!!
    """
    # test inputs
    if key_material is None:
        raise ValueError("need key_material to simulate data")
    if emax is None:
        raise ValueError("Highest Energy is not defined!")
    if MatchingRate_Angle_Tol is None:
        raise ValueError("Need a tolerance angle to compute matching rate!")
    if MatchingThresholdStop in (None, 0., 100., 0, 100):
        AngTol_LUTmatching = 0.5
        res_onespot = getOrientMatrices(spot_index,
                        emax,
                        table_angdist,
                        twiceTheta / 2., Chi,
                        n=n,
                        B=B,
                        cubicSymmetry=cubicSymmetry,
                        LUT=LUT,
                        ResolutionAngstrom=ResolutionAngstrom,
                        LUT_tol_angle=AngTol_LUTmatching,
                        MR_tol_angle=MatchingRate_Angle_Tol,
                        Minimum_Nb_Matches=1,
                        key_material=key_material,
                        detectorparameters=detectorparameters,
                        plot=0,
                        nbbestplot=1,
                        nbspots_plot='all',  # nb exp spots to display if plot = 1
                        addMatrix=None,
                        verbose=1)

        print("res_onespot", res_onespot)
        matrix = res_onespot[0][0]
        infos = res_onespot[1][0]

        nbmatch, nbtheo, meanresidue, best_spotscouple_indices, best_hkl_couple = infos

        matching_rate = 100. * nbmatch / nbtheo
        hkl1, hkl2 = best_hkl_couple

        spot_index, spotindex_2 = best_spotscouple_indices

        Res = matrix, matching_rate, [hkl1, hkl2], [spot_index, spotindex_2]
        return Res

#    print "table_angdist.shape in getOrientMatrix_from_onespot", table_angdist.shape
#    print "spot_index in getOrientMatrix_from_onespot", spot_index

    Res = None
    BestRecordedRes = None, 0, None, None
    exitloop = False

    if spot_index >= len(table_angdist):
        return BestRecordedRes

    Distances_from_central_spot = table_angdist[spot_index]

    coord = np.array([twiceTheta, Chi]).T
    coord1 = coord[spot_index]

    if LUT is None:
        print("build LUT in getOrientMatrix_from_onespot()")
        LUT = build_AnglesLUT(B, n,
                              MaxRadiusHKL=False,
                              cubicSymmetry=CP.hasCubicSymmetry(key_material))

    # main loop over all distances from central and each spots2
    for spotindex_2, angle in enumerate(Distances_from_central_spot):
        if verbose:
            print("\n----------------------------------------------------------")
            print("k,angle = ", spotindex_2, angle)
            print("-----------------------------------------------------------\n")

        hkls = FindO.PlanePairs_2(angle,
                                  ang_tol,
                                  LUT,
                                  onlyclosest=1,
                                  verbose=0)

        if hkls is not None and (spot_index != spotindex_2):
            nbpairs = len(hkls)
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spotindex_2, nbpairs)

            # handle hkls to compute a matrix and a matching rate
            if nbpairs == 1:

                hkl1, hkl2 = hkls[0].tolist()

                coord2 = coord[spotindex_2]

                matrix = FindO.OrientMatrix_from_2hkl(hkl1, coord1,
                                                    hkl2, coord2,
                                                    B,
                                                    verbose='no',
                                                    frame='lauetools')

                AngRes = ISS.Angular_residues(matrix,
                                        twiceTheta, Chi,
                                        key_material=key_material,
                                        emax=emax,
                                        ang_tol=MatchingRate_Angle_Tol,
                                        detectorparameters=detectorparameters)

                if AngRes is None:
                    continue

                (allres, resclose, nbclose,
                 nballres,
                 mean_residue, max_residue) = AngRes

                matching_rate = 100. * nbclose / nballres
                if verbose:
                    print("1P_A matching_rate %.1f" % matching_rate)

                if matching_rate > BestRecordedRes[1]:
                    BestRecordedRes = (matrix,
                                       matching_rate,
                                       [hkl1, hkl2],
                                       [spot_index, spotindex_2])
                    if verbose:
                        print("best matching rate record!")
                        print(BestRecordedRes, "\n")

                if matching_rate >= MatchingThresholdStop:
                    if verbose:
                        print("in matrices_from_onespot_new")
                        print("couple planes ", hkl1, hkl2)
                        print([coord1, coord2])
                        print("spot_index_2", spotindex_2)
                        print("matrix", matrix)

                    Res = (matrix,
                           matching_rate,
                           [hkl1, hkl2],
                           [spot_index, spotindex_2])
                    exitloop = True
                    break

                # ---compute matrix by swaping hkl1 and hkl2
                matrix = FindO.OrientMatrix_from_2hkl(hkl2, coord1,
                                                hkl1, coord2,
                                                B,
                                                verbose='no',
                                                frame='lauetools')

                AngRes = ISS.Angular_residues(matrix,
                                            twiceTheta, Chi,
                                            key_material=key_material,
                                            emax=emax,
                                            ang_tol=MatchingRate_Angle_Tol,
                                            detectorparameters=detectorparameters)

                if AngRes is None:
                    continue

                (allres, resclose, nbclose,
                 nballres,
                 mean_residue, max_residue) = AngRes

                matching_rate = 100.*nbclose / nballres
                if verbose:
                    print("1P_B matching_rate %.1f" % matching_rate)

                if matching_rate > BestRecordedRes[1]:
                    BestRecordedRes = matrix, matching_rate, [hkl1, hkl2], [spot_index, spotindex_2]

                if matching_rate >= MatchingThresholdStop:
                    if verbose:
                        print("in matrices_from_onespot_new")
                        print("couple planes ", hkl1, hkl2)
                        print([coord1, coord2])
                        print("spot_index_2", spotindex_2)
                        print("matrix", matrix)

                    Res = matrix, matching_rate, [hkl1, hkl2], [spot_index, spotindex_2]
                    exitloop = True
                    break

            if exitloop == True:
                break

            else:  # several pairs of solutions

                for m in list(range(nbpairs)):

                    # ---compute matrix by using hkl1 and hkl2
                    hkl1, hkl2 = hkls[m].tolist()

                    coord2 = coord[spotindex_2]

                    # print "hkl1, hkl2 ", hkl1, hkl2
                    # print "coord1,coord2", coord1,coord2
                    # print "spot_index, spotindex_2",spot_index, spotindex_2

                    matrix = FindO.OrientMatrix_from_2hkl(hkl1, coord1,
                                                    hkl2, coord2,
                                                    B,
                                                    verbose='no',
                                                    frame='lauetools')

                    AngRes = ISS.Angular_residues(matrix,
                                            twiceTheta, Chi,
                                            key_material=key_material,
                                            emax=emax,
                                            ang_tol=MatchingRate_Angle_Tol,
                                            detectorparameters=detectorparameters)

                    if AngRes is None:
                        continue

                    (allres, resclose, nbclose,
                     nballres,
                     mean_residue, max_residue) = AngRes

                    matching_rate = 100. * nbclose / nballres
                    if verbose:
                        print("nP_A matching_rate %.1f" % matching_rate)

                    if matching_rate > BestRecordedRes[1]:
                        BestRecordedRes = matrix, matching_rate, [hkl1, hkl2], [spot_index, spotindex_2]
                        if verbose:
                            print("best matching rate record!")
                            print(BestRecordedRes, "\n")

#                    #for debug
#                    if spotindex_2 == 8:
#                        print "in matrices_from_onespot_new"
#                        print "couple planes ", hkl1, hkl2
#                        print [coord1, coord2]
#                        print "spot_index_2", spotindex_2
#                        print "matrix", matrix
#
#                        print "matching_rate", matching_rate

                    if matching_rate >= MatchingThresholdStop:
                        if verbose:
                            print("in matrices_from_onespot_new")
                            print("couple planes ", hkl1, hkl2)
                            print([coord1, coord2])
                            print("spot_index_2", spotindex_2)
                            print("matrix", matrix)

                        Res = matrix, matching_rate, [hkl1, hkl2], [spot_index, spotindex_2]
                        exitloop = True
                        break

                    # ---compute matrix by using swaped hkl1 and hkl2

                    matrix = FindO.OrientMatrix_from_2hkl(hkl2, coord1,
                                                        hkl1, coord2,
                                                        B,
                                                        verbose='no',
                                                        frame='lauetools')

                    AngRes = ISS.Angular_residues(matrix,
                                        twiceTheta, Chi,
                                        key_material=key_material,
                                        emax=emax,
                                        ang_tol=MatchingRate_Angle_Tol,
                                        detectorparameters=detectorparameters)

                    if AngRes is None:
                        continue

                    (allres, resclose, nbclose,
                     nballres,
                     mean_residue, max_residue) = AngRes

                    matching_rate = 100.*nbclose / nballres
                    if verbose:
                        print("nP_B matching_rate %.1f" % matching_rate)

                    if matching_rate > BestRecordedRes[1]:
                        BestRecordedRes = (matrix, matching_rate,
                                           [hkl1, hkl2], [spot_index, spotindex_2])
                        if verbose:
                            print("best matching rate record!")
                            print(BestRecordedRes, "\n")

#                    #for debug
#                    if spotindex_2 == 8:
#                        print "in matrices_from_onespot_new"
#                        print "couple planes ", hkl1, hkl2
#                        print [coord1, coord2]
#                        print "spot_index_2", spotindex_2
#                        print "matrix", matrix
#
#                        print "matching_rate", matching_rate


                    if matching_rate >= MatchingThresholdStop:
                        if verbose:
                            print("in matrices_from_onespot_new")
                            print("couple planes ", hkl1, hkl2)
                            print([coord1, coord2])
                            print("spot_index_2", spotindex_2)
                            print("matrix", matrix)

                        Res = (matrix, matching_rate,
                               [hkl1, hkl2], [spot_index, spotindex_2])
                        exitloop = True
                        break

            # exit the main for loop
            if exitloop == True:
                break

    return BestRecordedRes


def getProximity(TwicethetaChi, data_theta, data_chi,
                        angtol=0.5, proxtable=0, verbose=0, signchi=1):
    """
    TwicethetaChi is an array made of 2 arrays 2theta and chi

    data_theta array of exp spot
    data_chi array of exp spot

    WARNING: TwicethetaChi contains 2theta instead of data_theta contains theta !
    TODO: change this input to 2theta, chi for every arguments

    signchi = 1 fixed old convention
    TODO: remove this option

    TODO: improve documentation
    """
    print("in getProximity()*****************************************************")
    # simul data
    theodata = np.array([TwicethetaChi[0] / 2., signchi * TwicethetaChi[1]]).T
    # exp data
    sorted_data = np.array([data_theta, data_chi]).T
    
#     table_dist = calculdist_from_thetachi(sorted_data, theodata)
#     print "table_dist_old", table_dist[:5, :5]
#     print "table_dist_old", table_dist[-5:, -5:]
#     print "table_dist_old", table_dist.shape
    
    if not USE_CYTHON:
        table_dist = calculdist_from_thetachi(sorted_data, theodata)
#         print "table_dist", table_dist[:5, :5]
    else:
        # TODO to be improved by not preparing array
        import angulardist
#         print "using cython"
        array_twthetachi_theo = np.array([TwicethetaChi[0], TwicethetaChi[1]]).T
        array_twthetachi_exp = np.array([data_theta * 2., data_chi]).T
        
#         print "flags", array_twthetachi_theo.flags
#         print "flags", array_twthetachi_exp.flags
        table_dist = angulardist.calculdist_from_2thetachi(array_twthetachi_theo.copy(order='c'),
                                                           array_twthetachi_exp.copy(order='c'))
    
#     print "table_dist_new", table_dist[:5, :5]
#     print "table_dist_new", table_dist[-5:, -5:]
#     print "table_dist_new", table_dist.shape    
    
    # table_dist has shape (len(theo),len(exp))
    # tab[0] has len(exp) elements
    # tab[i] with i index of theo spot contains all distance from theo spot #i and all the exprimental spots
    # prox_table has len(theo) elements
    # prox_table[i] is the index of the exp. spot closest to theo spot (index i)
    # table_dist[i][prox_table[i]] is the minimum angular distance separating
    # theo spot i from the nearest exp spot whose index is given by prox_table[i]
    prox_table = getArgmin(table_dist)

    # print "shape table_dist",shape(table_dist)

    if verbose:
#        print "/0", np.where(table_dist[:, 0] < 1.)
#        print np.argmin(table_dist[:, 0])
#        print "/4 exp", np.where(table_dist[:, 4] < 1.)
#        print np.argmin(table_dist[:, 4])
#        print np.shape(table_dist)
#         print table_dist[:, 4]
        print("table_dist",table_dist)
        print("prox_table",prox_table)

    pos_closest_1d = np.array(np.arange(len(theodata)) * len(data_chi) * np.ones(len(theodata)) + \
                               prox_table,
                               dtype=np.int32)
    # 1D array of smallest distance from an exp. spots for each theo. spots
    # len(allresidues)  = len(theo)
    allresidues = np.ravel(table_dist)[pos_closest_1d]

#     print "theodata", theodata
#     print 'len(allresidues)', len(allresidues)
    if proxtable == 0:
        whereclose = np.where(allresidues < angtol)
        res = allresidues[whereclose]
        longueur_res = len(whereclose[0])
        print("allresidues",allresidues)
        # print res
        # print longueur_res
        if longueur_res <= 1:
            nb_in_res = longueur_res
            if len(allresidues)==0:
                maxi = -1
            else:
                maxi = -np.min(allresidues)
            meanres = -1
        else:
            nb_in_res = len(res)
            maxi = np.max(res)
            meanres = np.mean(res)

        return allresidues, res, nb_in_res, len(allresidues), meanres, maxi

    elif proxtable == 1:
        return allresidues, prox_table, table_dist


def getOrientMatrices(spot_index_central,
                        energy_max,
                        Tab_angl_dist,
                        Theta_exp, Chi_exp,
                        n=3,  # ie. up  to (332)
                        ResolutionAngstrom=False,
                        B=np.eye(3),  # for cubic
                        cubicSymmetry=False,
                        LUT=None,
                        LUT_tol_angle=0.5,
                        MR_tol_angle=0.2,
                        Minimum_Nb_Matches=15,
                        key_material='',
                        plot=0,
                        nbbestplot=1,
                        nbspots_plot='all',  # nb exp spots to display if plot = 1
                        addMatrix=None,
                        verbose=1,
                        detectorparameters=None,
                        set_central_spots_hkl=None,
                        verbosedetails=True,
                        gauge=None
                        ):
    """
    Return all matrices that have a matching rate Minimum_Nb_Matches.
    Distances between two spots are compared to a reference
    angles look up table (LUT).

    Used AutoIndexation module (in LaueToolsGUI.py Classical Angular indexation)
    USED in FileSeries

    From:
    spot_index_central:         :    Integer or list of integer  corresponding to index of exp. spot from which distances
                                will be calculated with the rest of the exp. spots
    energy_max :                 :    Maximum energy used in simulation of the Laue Pattern (the higher this value the larger
                                the number of theo. spots)
    Tab_angl_dist :             :    Symetric matrix whose elements are angular distances (deg) between exp. spots as if they
                                were correspond to lattice planes (ie angle between lattice plane normals)
    Theta_exp,Chi_exp :        :    experimental 2theta/2 and chi two 1d arrays
    n                          :  integer for the maximum index of probed hkl when computing the LUT
    B                          :  Triangular up matrix defining the reciprocal unit cell
    LUT                            : if LUT is provided   the LUT won't be recomputed. It can save some times...
    LUT_tol_angle :             :    Angular tolerance (deg) below which exp. distance can be considered as recognised
                                in reference Look Uo Table
    MR_tol_angle:                :    Angular tolerance below which one exp spot can be linked to a single theo. spot
                                from simulation without any ambiguity
    Minimum_Nb_Matches:                :    Minimum nb of matches (nb of links between exp and theo spots) above which the corresponding
                                orientation matrix will be returned as a likely good candidate.
    key_material:            :    Element or structure label to the simulate the proper Laue Pattern (e.g.: 'Cu', 'Ge')
    Plot                        :    flag for plotting results
    nbbestplot:                    :    Maximum number of plot to display
    nbspots_plot:                :    'all' or int , nb exp spots to display if plot = 1
    addMatrix                    :    Matrix or list of Matrix that have to be processed in determining matching rate.
    set_central_spots_hkl    : list of hkls to set hkl for central spots (otherwise None)
                                to set one element

    Output:
    [0]  candidate Matrices
    [1]  corresponding score (matching rate, nb of theo. Spots, mean angular deviation over exp and theo. links)

    if spot_index_central is a list of spot index, then only one successful result is return per spot
    """
    if verbosedetails:
        print("print details")

    nbofpeaks = len(Tab_angl_dist) - 1  # nbmaxprobed-1
    
#     print "Tab_angl_dist", Tab_angl_dist

    if isinstance(spot_index_central, (list, np.ndarray)):
        list_spot_central_indices = spot_index_central
    elif isinstance(spot_index_central, int):
        if spot_index_central < 0:
            raise ValueError("spot_index_central is negative")
        list_spot_central_indices = (spot_index_central,)
    else:
        print("looking from spot # 0 (default settings)")
        list_spot_central_indices = (0,)

    if max(list_spot_central_indices) > nbofpeaks:
        raise ValueError("Tab_angl_dist of size %d is too small and does not contain distance with spot #%d" % \
            (nbofpeaks, max(list_spot_central_indices)))

    if key_material == '':
        raise ValueError("Warning! key_material is not defined in getOrientMatrices()")

    BestScores_per_centralspot = [[] for k in list(range(len(list_spot_central_indices)))]

    List_UBs = []  # matrix list
    List_Scores = []  # hall of fame BestScores_per_centralspot list

    if LUT is not None:
        print("LUT is not None when entering getOrientMatrices()")
    else:
        print("LUT is None when entering getOrientMatrices()")

    twiceTheta_exp = 2 * np.array(Theta_exp)

    # ---------------------------------
    # filling hkl central spots list
    set_central_spots_hkl_list = [None for ll in list(range(len(list_spot_central_indices)))]
    
    print("set_central_spots_hkl",set_central_spots_hkl)

    if set_central_spots_hkl not in (None, 'None'):
        print("set_central_spots_hkl is not None in getOrientMatrices()")
        set_central_spots_hkl = np.array(set_central_spots_hkl)
        print("set_central_spots_hkl", set_central_spots_hkl)
        print("set_central_spots_hkl.shape", set_central_spots_hkl.shape)
        if set_central_spots_hkl.shape == (3,):
            print('case: 1a')
            set_central_spots_hkl_list = np.tile(set_central_spots_hkl,
                                                 (len(list_spot_central_indices), 1))
        elif set_central_spots_hkl.shape == (1, 3):
            set_central_spots_hkl_list = np.tile(set_central_spots_hkl[0],
                                                 (len(list_spot_central_indices), 1))
            print('case: 1b')
        else:
            print('case: 2')
            for ll, hkl_elem in enumerate(list_spot_central_indices):
                if ll < len(set_central_spots_hkl):
                    set_central_spots_hkl_list[ll] = set_central_spots_hkl[ll]

    print("set_central_spots_hkl_list", set_central_spots_hkl_list)
    print("cubicSymmetry", cubicSymmetry)

    if detectorparameters['kf_direction'] == 'X>0':
        allow_restrictedLUT = False
    else:
        allow_restrictedLUT = cubicSymmetry

#     if cubicSymmetry:
#         hkl1s = FindO.HKL_CUBIC

    print("LUT_tol_angle", LUT_tol_angle)

    if gauge:
        import wx
        gaugecount = 0
        gauge.SetValue(gaugecount)

    # set of hkl for computing specific LUT when hkl is given for central spot
    hkl2 = None
    LUTcubic = None
    LUTspecific = None

    #--- loop over central spots -------------------------------------------------------
    for k_centspot_index, spot_index_central in enumerate(list_spot_central_indices):
        print("*---****------------------------------------------------*")
        print("Calculating all possible matrices from exp spot #%d and the %d other(s)" % \
                                (spot_index_central, nbofpeaks))
        # list_orient_matrix, planes, pairspots = matrices_from_onespot(spot_index_central,
                                                                        # LUT_tol_angle,
                                                                        # Tab_angl_dist,
                                                                        # Theta_exp,Chi_exp,
                                                                        # verbose=0)
        # read hkl for central spots defined in a list
        if set_central_spots_hkl_list is not None:
            hkl = set_central_spots_hkl_list[k_centspot_index]

        # hkl for central spots IS defined
        print("hkl in getOrientMatrices",hkl, type(hkl))
        if hkl is 'None': hkl = None
        if hkl is not None:

            # find some potential matrices from recognised distances
            # in LUT by assuming hkl of spot_index_central

            # compute a specific LUT from 1 hkl and hkls up to n
            print("using LUTspecific")
            if LUTspecific is not None:
                print("LUTspecific is not None for k_centspot_index %d in getOrientMatrices()" % k_centspot_index)
            else:
                print("LUTspecific is None for k_centspot_index %d in getOrientMatrices()" % k_centspot_index)

            ((list_orient_matrix,
             planes,
             pairspots),
             hkl2,
             LUTspecific) = matrices_from_onespot_hkl(spot_index_central, LUT_tol_angle,
                                            Tab_angl_dist,
                                            twiceTheta_exp, Chi_exp,
                                            n, key_material, MaxRadiusHKL=False,
                                            hkl1=hkl,
                                            hkl2=hkl2,
                                            LUT=LUTspecific,
                                            allow_restrictedLUT=allow_restrictedLUT,
                                            verbose=verbose)

        # hkl for central spots IS NOT defined
        else:
            # TODO: retrieve cubic LUT if already calculated
            if cubicSymmetry:
                # compute a specific LUT from 1 hkl and hkls up to n
                print("using LUTcubic")
                if LUTcubic is not None:
                    print("LUTcubic is not None for k_centspot_index %d in getOrientMatrices()" % k_centspot_index)
                else:
                    print("LUTcubic is None for k_centspot_index %d in getOrientMatrices()" % k_centspot_index)

                if n == 3:
                    hkl1 = FindO.HKL_CUBIC_UP3
                else:
                    hkl1 = GT.threeindicesfamily(n)
                
                (list_orient_matrix,
                 planes,
                 pairspots), hkl2, LUTcubic = matrices_from_onespot_hkl(spot_index_central,
                                                LUT_tol_angle,
                                                Tab_angl_dist,
                                                twiceTheta_exp, Chi_exp,
                                                n, key_material, MaxRadiusHKL=False,
                                                hkl1=hkl1,
                                                hkl2=hkl2,
                                                LUT=LUTcubic,
                                                allow_restrictedLUT=allow_restrictedLUT,
                                                verbose=verbose)
            # general crystallographic case
            else:
                #--- building angles reference Look up table (LUT) from B and n
                if LUT is None:
                    # use following LUT
                    print("building LUT in getOrientMatrices()")
                    LUT = build_AnglesLUT(B, n, MaxRadiusHKL=False,
                                          cubicSymmetry=cubicSymmetry)

                print("using general non cubic LUT")
                # find some potential matrices from recognised distances in LUT
                (list_orient_matrix,
                 planes,
                 pairspots) = matrices_from_onespot_new(spot_index_central,
                                                        LUT_tol_angle,
                                                        Tab_angl_dist,
                                                        twiceTheta_exp,
                                                        Chi_exp,
                                                        n, B,
                                                        LUT=LUT,
                                                        verbose=verbose)

        if gauge:
            gaugecount = (k_centspot_index + 1) * nbofpeaks
            # print "gaugecount += 100",gaugecount
            gauge.SetValue(gaugecount)
            wx.Yield()

        solutions_matorient_index = []
        solutions_spotscouple = []
        solutions_hklcouple = []
        solutions_matchingscores = []
        solutions_matchingrate = []

        if verbose:
            print("len(list_orient_matrix)", len(list_orient_matrix))
            # print "key_material",key_material
            # print "\n"
            print("#mat nb<%.2f       nb. theo. spots     mean       max    nb**2/nb_theo*mean     plane indices" % \
                                                        (MR_tol_angle))

        #--- loop over orient matrix given from LUT recognition for one central spot
        currentspotindex2 = -1
        for mat_ind in list(range(len(list_orient_matrix))):
            
#             print "calculating matching with exp. Data for matrix condidate index=%d" % mat_ind

            # compute matching (indexation) success and store if high
            AngRes = ISS.Angular_residues_np(list_orient_matrix[mat_ind],
                                        twiceTheta_exp, Chi_exp,
                                        key_material=key_material,
                                        emax=energy_max,
                                        ResolutionAngstrom=ResolutionAngstrom,
                                        ang_tol=MR_tol_angle,
                                        detectorparameters=detectorparameters)

            if AngRes is None:
                continue

            (allres, resclose, nbclose,
             nballres,
             mean_residue, max_residue) = AngRes

            if nbclose >= Minimum_Nb_Matches:
                std_closematch = np.std(allres[allres < MR_tol_angle])
                if verbosedetails:
#                     print "%d        %d              %d             %.3f       %.3f        %.3f" % \
#                             (mat_ind, nbclose, nballres, mean_residue, max_residue, nbclose ** 2 * 1. / nballres * mean_residue), "    ", str(planes[mat_ind]), \
#                             "  ", pairspots[mat_ind]

                    print("%d        %d       %d       %.3f      %.3f       %.3f        %.3f" % \
                            (mat_ind, nbclose, nballres,
                             std_closematch, mean_residue, max_residue,
                             nbclose ** 2 * 1. / nballres / std_closematch), "    ", str(planes[mat_ind]), \
                            "  ", pairspots[mat_ind])

                spotindex2 = pairspots[mat_ind][1]
                if currentspotindex2 != spotindex2:
                    print("calculating matching rates of solutions for exp. spots", pairspots[mat_ind])
                    currentspotindex2 = spotindex2

                solutions_matorient_index.append(mat_ind)
                solutions_spotscouple.append(pairspots[mat_ind])
                solutions_hklcouple.append(planes[mat_ind])
#                 solutions_matchingscores.append([nbclose, nballres, mean_residue])
                solutions_matchingscores.append([nbclose, nballres, std_closematch])
                solutions_matchingrate.append(100. * nbclose / nballres)
                # print list_orient_matrix[mat_ind]

        BestScores_per_centralspot[k_centspot_index] = np.array(solutions_matchingscores)

        # for one central spot if there are at least one potential solution
        if len(BestScores_per_centralspot[k_centspot_index]) > 0:

#             print "Got one solution for k_centspot_index: %d" % k_centspot_index

            # sort results
            rank = np.lexsort(keys=\
                  (BestScores_per_centralspot[k_centspot_index][:, 2],
                   BestScores_per_centralspot[k_centspot_index][:, 0] * 1. / \
                        BestScores_per_centralspot[k_centspot_index][:, 1]))[::-1]

            hall_of_fame = BestScores_per_centralspot[k_centspot_index][rank]

            orient_index_fame = np.array(solutions_matorient_index)[rank]
            psb = np.array(solutions_spotscouple)[rank]
            ppb = np.array(solutions_hklcouple)[rank]

            # only for plot
            list_UBs_for_plot = np.array(list_orient_matrix)

#             print "orient_index_fame", orient_index_fame

            # print "list_UBs_for_plot",list_UBs_for_plot

            # for search from many spots (<=9)
            # one keeps only the best matrix

            for mm in list(range(min(nbbestplot, len(orient_index_fame)))):
                bestmatrix = list_UBs_for_plot[orient_index_fame[mm]]
                bestscores = list(hall_of_fame[mm]) + list([psb[mm]]) + list([ppb[mm]])
#                 print "bestmatrix", bestmatrix
#                 print "bestscores", bestscores
                List_UBs.append(bestmatrix)
                List_Scores.append(bestscores)

        else:
            print("No orientation matrix found with nb of matches larger than %d" \
                                            % Minimum_Nb_Matches)
            if verbose:
                print("Try to:")
                print("- decrease Ns (MNMS: minimum number of matched spots)")
                print("- increase angular tolerances (distance recognition and/or matching)")
                print("- increase nbofpeaks (ISSS: Intense spot set size)")
                print("- increase Energy max")
        print("\n")

    # Consider immediately (if any) a list of a priori known matrices
    if addMatrix is not None:
        orient_index_best_add = []
        pair_spots_best_add = []
        pair_planes_best_add = []
        score_best_add = []

        # # restart indexation from the lastindex = len(BestScores_per_centralspot) - 1
        # newindex = len(list_UBs_for_plot) # for a single central spot ... # TODO !!: implement many central spots

        # calculate residues from additional matrix
        # loop over added matrices
        print("loop over added matrices")
        for mat_ind, added_mat in enumerate(addMatrix):
            AngRes = ISS.Angular_residues_np(added_mat,
                                        2 * Theta_exp, Chi_exp,
                                        key_material=key_material,
                                        emax=energy_max,
                                        ResolutionAngstrom=ResolutionAngstrom,
                                        ang_tol=MR_tol_angle)

            if AngRes is None:
                continue

            (allres, resclose, nbclose,
             nballres,
             mean_residue, max_residue) = AngRes

            if nbclose > Minimum_Nb_Matches:
                if verbose:
                    print("%d        %d                   %d                %.3f       %.3f " % \
                    (mat_ind, nbclose, nballres, mean_residue, max_residue), "unknown              unknown ")
                std_closematch = np.std(allres[allres < MR_tol_angle])
                orient_index_best_add.append(mat_ind)
                pair_spots_best_add.append(np.array([0, 0]))  # not defined
                pair_planes_best_add.append(np.array([[0, 0, 0],
                                                      [0, 0, 0]]))  # not defined
#                 score_best_add.append([nbclose, nballres, mean_residue])
                score_best_add.append([nbclose, nballres, std_closematch])
                # print list_orient_matrix[mat_ind]

        fame_add = np.array(score_best_add)
        # nb of BestScores_per_centralspot a priori matrices
        nbpriorimatrices = len(fame_add)

        # if there is a least a interesting added matrix
        # that has hit good rank in previous angular residues test
        if len(fame_add) > 0:

            for mm in list(range(nbpriorimatrices)):
                List_UBs.append(addMatrix[orient_index_best_add[mm]])
                List_Scores.append(list(score_best_add[mm]) + \
                           list([pair_spots_best_add[mm]]) + \
                           list([pair_planes_best_add[mm]]))

    # results and plot for only one central spot
    if len(list_spot_central_indices) == 1:

        if plot:
            if nbspots_plot == 'all':
                _nbspots_plot = len(Chi_exp)
            else:
                _nbspots_plot = nbspots_plot

            Plot_compare_2thetachi_multi(orient_index_fame[:9],
                                            2 * Theta_exp, Chi_exp,
                                            EULER=list_UBs_for_plot,
                                            key_material=key_material,
                                            emax=energy_max,
                                            verbose=0,
                                            exp_spots_list_selection=_nbspots_plot,
                                            title_plot=list(hall_of_fame),
                                            figsize=(8, 8),
                                            dpi=70)
            # savefig('spot_'+str(spot_index_central[0])+'.png')

        print('return best matrix and matching scores for the one central_spot')
        # returned results for only one central spot
        return List_UBs, List_Scores

    # results and plot for many central spots
    elif len(list_spot_central_indices) > 1:
        # Taking only the best matrix for each central spot

        MATOS = np.array(List_UBs)
        nbmatrixfound = len(List_UBs)
        # print "MATOS",MATOS
        HF = List_Scores

        if plot:
            if nbspots_plot == 'all':
                _nbspots_plot = len(Chi_exp)
            else:
                _nbspots_plot = nbspots_plot

            Plot_compare_2thetachi_multi(np.arange(min(nbmatrixfound, 9)),
                                            2 * Theta_exp, Chi_exp,
                                            key_material=key_material,
                                            emax=energy_max,
                                            verbose=0,
                                            EULER=MATOS,
                                            exp_spots_list_selection=_nbspots_plot,
                                            title_plot=list(HF))
            # savefig('spot_'+str(spot_index_central[0])+'.png')

        # returned results for several central spots
        return MATOS, HF


def build_AnglesLUT(Bmatrix, n, MaxRadiusHKL=False, cubicSymmetry=False):
    """
    build a Look-up-table from a B Matrix up to index n
    (higher miller plane [n,n,n])

    cubicSymmetry   : flag to restrict LUT to use only HKL with L>=0
    """
    latticeparameters = CP.directlatticeparameters_fromBmatrix(Bmatrix)
    return build_AnglesLUT_fromlatticeparameters(latticeparameters, n,
                                                 MaxRadiusHKL=MaxRadiusHKL,
                                                 cubicSymmetry=cubicSymmetry)


def build_AnglesLUT_fromlatticeparameters(latticeparameters, n,
                                          MaxRadiusHKL=False,
                                          cubicSymmetry=False):
    """
    build a Look-up-table from the 6 lattice parameters (a,b,c,alpha,beta,gamma) up to index n
    (higher miller plane [n,n,n])

    cubicSymmetry   : True to restrict LUT to have only HKL with L>=0

    MaxRadiusHKL    : False or largest value of sqrt(H**2+k**2+l**2) to keep HKL in LUT 
    """
    a, b, c, AA, BB, CC = latticeparameters

    print("Build angles LUT with latticeparameters")
    print(latticeparameters)
    print("and n=%d" % n)
    print("MaxRadiusHKL", MaxRadiusHKL)
    print("cubicSymmetry", cubicSymmetry)

    # metric tensor
    Gstar_metric = CP.Gstar_from_directlatticeparams(a, b, c, AA, BB, CC)

    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n, remove_negative_l=cubicSymmetry)

    if 1:  # filterharmonics:
#         hkl_all = FindO.FilterHarmonics(hkl_all)
        hkl_all = CP.FilterHarmonics_2(hkl_all)

    if MaxRadiusHKL not in (False, 0, 0.0):
        HKLnorm = np.sqrt(np.sum(hkl_all ** 2, axis=1))

        H, K, L = hkl_all.T
        Condit = HKLnorm < (1. / MaxRadiusHKL)

        # print "Condit",Condit
        H = np.compress(Condit, H)
        K = np.compress(Condit, K)
        L = np.compress(Condit, L)

        hkl_all = np.array([H, K, L]).T

    # GenerateLookUpTable
    LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)

#     if not cubicSymmetry:
#         LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)
#     else:
#         LUT = FindO.Generate_LUT_for_Cubic(hkl_all, Gstar_metric)

#     print "LUT", LUT

    return LUT


def getOrients_AnglesLUT(spot_index_central,
                        Tab_angl_dist,
                        TwiceTheta,
                        Chi,
                        Matching_Threshold_Stop=40.,
                        angleTolerance_LUT=0.5,
                        MatchingRate_Angle_Tol=0.5,
                        n=3,
                        B=np.eye(3),
                        LUT=None,
                        key_material=None,
                        emax=25.,
                        absoluteindex=None,
                        detectorparameters=None,
                        verbose=1):
    """
    TODO: obsolete not used anymore ?

    Return all matrices that have a matching rate Nb_criterium.
    Distances between two spots are compared to a reference
    angles look up table.

    used by indexingspotsSet.py

    From:
    spot_index_central:         :    Integer or list of integer  corresponding to index of exp. spot from which distances
                                will be calculated with the rest of the exp. spots
    energy_max :                 :    Maximum energy used in simulation of the Laue Pattern (the higher this value the larger
                                the number of theo. spots)
    Tab_angl_dist :             :    Symetric matrix whose elements are angular distances (deg) between exp. spots as if they
                                were correspond to lattice planes (ie angles between lattice planes normals)
    TwiceTheta,Chi :           :    experimental 2theta and chi two 1d arrays
    n                            integer for the maximum index of probed hkl when computing the LUT
    B                            Triangular up matrix defining the unit cell
    LUT                            : if LUT is provided   the LUT won't be recomputed. It can save some times...
    angleTolerance_LUT :             :    Angular angleTolerance_LUT (deg) below which exp. distance can be considered as recognised
                                in reference Look Uo Table
    MatchingRate_Angle_Tol:                :    Angular angleTolerance_LUT below which one exp spot can be linked to a single theo. spot
                                from simulation without any ambiguity
    Nb_criterium:                :    Minimum matching rate (nb of links between exp and theo spots) above which the corresponding
                                orientation matrix will be returned as a likely good candidate.
    structure_label:            :    Element or structure label to the simulate the proper Laue Pattern (e.g.: 'Cu', 'Ge')
    Plot                        :    flag for plotting results
    nbbestplot:                    :    Maximum number of plot to display
    nbspots_plot:                :    'all' or int , nb exp spots to display if plot = 1
    addMatrix                    :    Matrix or list of Matrix that have to be processed in determining matching rate.

    detectorparameters      : dictionary of detector parameters (key, value)
                            'kf_direction' , general position of detector plane
                            'detectordistance', detector distance (mm)
                            'detectordiameter', detector diameter (mm)

    Output:
    [0]  candidate Matrices
    [1]  corresponding score (matching rate, nb of theo. Spots, mean angular deviation over exp and theo. links)

    if spot_index_central is a list of spot index, then only one successful result is return per spot
    """
    nbofpeaks = len(Tab_angl_dist) - 1

    if type(spot_index_central) == type(np.array([1, 2, 3])) or type(spot_index_central) == type(list([1, 2, 3])):
        scan_over_index = spot_index_central
    elif isinstance(spot_index_central, int):
        if spot_index_central < 0:
            raise ValueError("spot_index_central is negative")
        scan_over_index = [spot_index_central]
    else:
        print("looking from spot # 0")
        scan_over_index = [0]

    if key_material is None:
        raise ValueError("Warning! StructureLabel is not defined in getOrients_AnglesLUT()")

    #--- building angles reference Look up table (LUT) from B and n
    if LUT is None:
        LUT = build_AnglesLUT(B, n,
                              MaxRadiusHKL=False,
                              cubicSymmetry=CP.hasCubicSymmetry(key_material))

    twiceTheta = np.array(TwiceTheta)

    ResMatrix, ResScore = None, None

    AllRes = []
    AllMatrixres = []
    Threshold_reached = False

    for spot_index_central in scan_over_index:

        print("\n*-------------------------------------------------------*")
        print("Calculating all possible matrices from exp spot #%d and the %d others" % \
                                                    (spot_index_central, nbofpeaks))
        if absoluteindex is not None:
            print("central spot absolute index : %d" % \
                    absoluteindex[spot_index_central])

        Res = getOrientMatrix_from_onespot(spot_index_central,
                                           angleTolerance_LUT,
                                  Tab_angl_dist,
                                  twiceTheta, Chi,
                                  n, B,
                                  LUT=LUT,
                                  MatchingThresholdStop=Matching_Threshold_Stop,
                                  key_material=key_material,
                                  emax=emax,
                                  MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                                  verbose=0,
                                  detectorparameters=detectorparameters)

        print("Res", Res)
        print("for spot central relative index %d" % spot_index_central)

        # matching rate corresponds to a high matching
        if Res[1] >= Matching_Threshold_Stop:
            print("spot_index_central", spot_index_central)
            if absoluteindex is not None:
                print("central spot absolute index : %d" % \
                        absoluteindex[spot_index_central])
#            print "Res", Res[1]
            print("with %d exp spots" % len(twiceTheta))

            (matrix,
             matching_rate,
             pair_hkl,
             pair_spots) = Res
            if verbose:
                print("Res Angles LUT matching")
                print(Res)
                print("pair hkls", pair_hkl)
                print("pair spots", pair_spots)
                if absoluteindex is not None:
                    print("pair spots absolute index : [%d,%d]" % \
                            (absoluteindex[pair_spots[0]],
                             absoluteindex[pair_spots[1]]))

            ResMatrix, ResScore = matrix, matching_rate

            Threshold_reached = True
            break

        # best matching rate is not so high
        else:
            AllRes.append([spot_index_central, Res[1]])
            AllMatrixres.append(Res[0])
            # if matching rate is quite close to Matching_Threshold_Stop
#            if Res[1]<=0.90*Matching_Threshold_Stop:
#                break

            # continue the for loop if best matching rate is too poor
            pass

    if Threshold_reached == False:
        ArRes = np.array(AllRes)
        bestres = np.argmax(ArRes[:, 1])

        ResMatrix, ResScore = AllMatrixres[bestres], ArRes[bestres]

    return ResMatrix, ResScore, Threshold_reached


#--- ------------  TESTs   Angles LUT
def test_GetProximity():
    
    nexp = 3
    
    shufflenexp = np.random.permutation(nexp)
    
    
    TwicethetaChi = np.ascontiguousarray(np.random.randint(40, high=140, size=(2, 20)),
                                         dtype=np.double)

    
    data_theta = TwicethetaChi[0] / 2.
    data_chi = TwicethetaChi[1] + 0.4
    
    exp_theta, exp_chi = data_theta[shufflenexp], data_chi[shufflenexp]
    
    res = getProximity(TwicethetaChi, exp_theta, exp_chi,
                        angtol=0.5, proxtable=0, verbose=0, signchi=1)
    
    # allresidues, res, nb_in_res, len(allresidues), meanres, maxi
    
    print(res)
    return res, TwicethetaChi[0] / 2., exp_theta, TwicethetaChi[1], exp_chi


def test_GetProximity2():
    
    simuldata = np.array([[  59.2164, -66.6041],
       [  54.7398, -88.3875],
       [  43.0763, -83.0983],
       [  58.895 , -88.7721],
       [  39.3337, -65.6132],
       [  52.4121, -58.2399],
       [  47.6515, -100.4466],
       [  47.9575, -83.2749],
       [  44.0163, -64.8536],
       [  56.4791, -57.2717],
       [  48.434 , -115.549 ],
       [  52.2976, -101.5099],
       [  28.9731, -75.0536],
       [  53.191 , -83.4718],
       [  49.0293, -64.0107],
       [  41.0373, -47.0408],
       [  52.3338, -117.1192],
       [  37.1386, -98.1813],
       [  58.7918, -83.6928],
       [  54.3839, -63.0704],
       [  55.9659, -41.1327],
       [  39.5682, -118.0816],
       [  56.3583, -118.8114],
       [  41.6482, -74.3025],
       [  49.9809, -43.6082],
       [  53.2592, -130.741 ],
       [  44.4802, -120.2579],
       [  49.9426, -100.9654],
       [  38.9334, -47.8105],
       [  54.7471, -41.6504],
       [  56.4465, -132.6251],
       [  49.6174, -122.6474],
       [  21.8155, -95.1273],
       [  57.3701, -73.2746],
       [  46.408 , -26.3132],
       [  55.4884, -25.5038],
       [  46.0081, -136.7264],
       [  59.6475, -134.6084],
       [  28.2351, -126.1327],
       [  54.9433, -125.2734],
       [  31.5495, -97.0398],
       [  22.4182, -53.4955],
       [  52.3414, -42.6513],
       [  36.0373, -24.6759],
       [  50.0743, -24.0159],
       [  58.489 , -23.5838],
       [  35.0685, -129.9977],
       [  43.26  , -99.4789],
       [  32.9485, -49.9361],
       [  40.7146, -21.6059],
       [  53.7612, -21.5865],
       [  32.8305, -146.1445],
       [  53.4871, -141.8828],
       [  42.2965, -134.3526],
       [  57.1994, -102.6829],
       [  45.4058, -45.3993],
       [  21.3158, -17.0592],
       [  45.4556, -18.3182],
       [  57.4384, -19.0229],
       [  37.2146, -149.6308],
       [  49.749 , -139.236 ],
       [  27.7533, -12.0696],
       [  50.1926, -14.8127],
       [  50.5507, -152.2943],
       [  41.5523, -153.2806],
       [  57.1871, -144.6661],
       [  34.2669, -6.6991],
       [  54.8492, -11.096 ],
       [  53.1766, -154.6529],
       [  45.7737, -157.0741],
       [  12.2809, -166.6089],
       [  40.6313, -1.0068],
       [  59.3447, -7.1823],
       [  52.0084, -4.6259],
       [  55.7231, -157.0654],
       [  49.8098, -160.985 ],
       [  22.0916, -176.2917],
       [  46.612 , 4.9165],
       [  43.6835, 1.9328],
       [  54.5417, -2.2204],
       [  42.4339, -165.4599],
       [  58.1746, -159.525 ],
       [  53.5979, -164.9811],
       [  30.6904, 174.3774],
       [  52.0051, 10.9533],
       [  56.9817, 0.2302],
       [  45.1098, -168.3128],
       [  26.5826, 178.968 ],
       [  57.086 , -169.0252],
       [  37.6256, 165.8261],
       [  15.6917, 39.6224],
       [  56.6694, 16.9722],
       [  32.5476, 15.6023],
       [  49.3926, 7.9288],
       [  59.3144, 2.7178],
       [  47.6427, -171.1734],
       [  42.8967, 158.285 ],
       [  22.5114, 46.4513],
       [  35.8165, 19.2694],
       [  50.0185, -174.0275],
       [  34.3748, 169.985 ],
       [  46.7627, 151.7992],
       [  27.9587, 52.2656],
       [  38.8316, 22.8625],
       [  54.4341, 13.9733],
       [  52.2268, -176.8611],
       [  49.55  , 146.2952],
       [  55.2312, 100.1347],
       [  32.2566, 57.1803],
       [  41.5781, 26.3568],
       [  54.2617, -179.6607],
       [  40.4564, 161.9228],
       [  51.5498, 141.6466],
       [  35.6456, 61.3316],
       [  44.0522, 29.7315],
       [  58.7058, 19.9338],
       [  56.121 , 177.5861],
       [  52.9876, 137.7171],
       [  38.334 , 64.8498],
       [  46.2592, 32.9705],
       [  57.8061, 174.8905],
       [  44.9849, 154.9128],
       [  54.0273, 134.3813],
       [  40.4874, 67.8483],
       [  48.2113, 36.0624],
       [  59.3215, 172.2623],
       [  54.7844, 131.5323],
       [  42.2315, 70.4209],
       [  49.9255, 39.0003],
       [  41.3115, 160.6805],
       [  48.2716, 148.9317],
       [  55.3394, 129.0823],
       [  43.6605, 72.6439],
       [  51.4215, 41.7811],
       [  55.7485, 126.9602],
       [  44.8447, 74.5785],
       [  52.7206, 44.4051],
       [  44.3254, 156.0077],
       [  50.6327, 143.8726],
       [  56.0512, 125.1091],
       [  45.8366, 76.2738],
       [  53.8438, 46.875 ],
       [  45.6099, 153.8468],
       [  56.2755, 123.4835],
       [  46.6758, 77.7691],
       [  54.8115, 49.1957],
       [  52.3274, 139.6002],
       [  56.4416, 122.0467],
       [  47.3926, 79.0961],
       [  55.6426, 51.3733],
       [  56.5638, 120.7692],
       [  48.01  , 80.2805],
       [  56.3545, 53.4149],
       [  53.549 , 135.9821],
       [  56.6529, 119.6269],
       [  48.5461, 81.3433],
       [  56.9627, 55.3281],
       [  56.7166, 118.6003],
       [  49.0149, 82.3015],
       [  57.4811, 57.1208],
       [  54.4355, 132.9019],
       [  56.761 , 117.6731],
       [  49.4277, 83.1695],
       [  57.9219, 58.8009],
       [  56.7903, 116.8321],
       [  49.7933, 83.9591],
       [  58.2956, 60.376 ],
       [  55.0832, 130.2622],
       [  56.8081, 116.066 ],
       [  50.1191, 84.6802],
       [  58.6116, 61.8535],
       [  56.8168, 115.3656],
       [  50.4108, 85.3411],
       [  58.8779, 63.2404],
       [  55.5594, 127.9841],
       [  56.8185, 114.7228],
       [  59.1012, 64.5432],
       [  56.8147, 114.1311],
       [  56.8067, 113.5847]])
    
    
    expdata = np.ascontiguousarray(np.array([[  45.44929 , -18.34126 ],
       [  45.157076, -137.083181],
       [  44.92114 , 154.84827 ],
       [  27.931473, 52.277335],
       [  35.129301, -129.958303],
       [  22.506704, 46.525391],
       [  45.805636, -157.105433],
       [  27.76149 , -12.146467],
       [  22.099332, -176.308319],
       [  21.854819, -95.128858]]), dtype=np.double)
    
    TwicethetaChi = np.ascontiguousarray(simuldata, dtype=np.double).T
    
    exp_theta, exp_chi = expdata[:, 0] / 2., expdata[:, 1]
    
    res = getProximity(TwicethetaChi, exp_theta, exp_chi,
                        angtol=0.5, proxtable=0, verbose=0, signchi=1)
    
    # allresidues, res, nb_in_res, len(allresidues), meanres, maxi
    
    print(res)
    return res, TwicethetaChi[0] / 2., exp_theta / 2., TwicethetaChi[1], exp_chi


def Test_Build_and_Query_LUT():
    """
    test building and query of an angular look up table from a crystal
    given its lattice parameters

    """
    # test
    n = 3
    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n)

    if 1:  # filterharmonics:
        hkl_all = FindO.FilterHarmonics(hkl_all)

#    Gstar_metric = CP.Gstar_from_directlatticeparams(2, 1, 4, 75, 90, 120)
    Gstar_metric = CP.Gstar_from_directlatticeparams(5.43, 5.43, 5.43, 90, 90, 90)
    # GenerateLookUpTable
    LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)

    print(FindO.QueryLUT(LUT, 14.01, 1))


def test_indexation_with_apriori_Matrices():
    """
    test of indexation + adding some selected matrices
    """
    folderdata = './Examples/UO2/'
    filename_data = 'dat_UO2_A163_2_0028_LT_0.cor'
    filename_data = os.path.join(folderdata, filename_data)
    StructureLabel = 'UO2'

    nb_of_spots = 2000  # nb of spots to select for recognition
    nbspots_plot = 300  # nb of spots to select for plotting

    # READ .cor file
    mydata = ReadASCIIfile(filename_data,
                        col_2theta=0,
                        col_chi=1,
                        col_Int=-1,
                        nblineskip=1)
    data_theta, data_chi, data_I = mydata

    print(data_I)
    print(data_theta)

    nbp = len(data_theta)
    # listofselectedpts=arange(nbofpeaks-1) # 1 for 1 header line
    # print data_I

    # --- Selection of peaks
    if nb_of_spots > 0:
        upto = min(nb_of_spots, nbp)
        sorted_int_index = np.argsort(data_I)[::-1][:upto + 1]
        print("Considering only %d most intense spots" % upto)

    elif nb_of_spots == -1:
        print("Considering all spots")
        sorted_int_index = np.argsort(data_I)[::-1]

#    listofselectedpts = np.arange(len(sorted_int_index))
#    indicespotmax=len(listofselectedpts)

    Theta = data_theta[sorted_int_index]
    Chi = data_chi[sorted_int_index]
    Intens = data_I[sorted_int_index]
    sorted_data = np.array([Theta, Chi, Intens]).T

    print(sorted_data[:10])

    # --- Rough interangular distance recognition

    # array of interangular distance of all points
    print("Calculating all angular distances\n")
    Tabledistance = calculdist_from_thetachi(sorted_data[:, 0:2], sorted_data[:, 0:2])

    ind_sorted_LUT_MAIN_CUBIC = [np.argsort(elem) for elem in FindO.LUT_MAIN_CUBIC]
    sorted_table_angle = []
    for k in list(range(len(ind_sorted_LUT_MAIN_CUBIC))):
        # print len(LUT_MAIN_CUBIC[k])
        # print len(ind_sorted_LUT_MAIN_CUBIC[k])
        sorted_table_angle.append((FindO.LUT_MAIN_CUBIC[k])[ind_sorted_LUT_MAIN_CUBIC[k]])

    # --- getOrientMatrices   with (spot_index_central,energy_max)
    spot_index = [0, 1, 21, 22, 24]  # one integer (starting from 0) or an array:array([1,8,9,5])
    energymax = 25
    nbmax_probed = 25  # size of set of spots to calculate distance from central spot #spot_index
    nbspots_plot = 200  # for plot exp. data
    print("Using getOrientMatrices()")

    TrialsMatrix = (
                    np.array([[-0.99251179, 0.07104589, 0.09936213],
                              [-0.01590782, -0.88248953, 0.47006294],
                              [ 0.12126142 , 0.46522732, 0.87684617]]),
                    np.array([[-0.49751201, -0.64141178, 0.58401432],
                              [ 0.19466032, -0.7385631, -0.6454703 ],
                              [ 0.84535393, -0.20737935, 0.49231143]]),
                    np.eye(3)
                    )

    TrialsMatrix = None

    # with nbbestplot very high  mama contain all matrices with matching rate above Nb_criterium

    n = 3  # for LUT
    latticeparams = DictLT.dict_Materials[StructureLabel][1]
    B = CP.calc_B_RR(latticeparams)

    mama, hhh = getOrientMatrices(spot_index,
                        energymax,
                        Tabledistance[:nbmax_probed, :nbmax_probed],
                        Theta,
                        Chi,
                        n,
                        B,
                        LUT_tol_angle=0.5,
                        MR_tol_angle=0.2,
                        Minimum_Nb_Matches=40,
                        key_material=StructureLabel,
                        nbspots_plot=nbspots_plot,
                        nbbestplot=20,
                        plot=0,
                        addMatrix=TrialsMatrix)

    print("All %d matrices solutions" % len(mama))
    print(mama)
    print("Corresponding Scores")
    print(hhh)

    MAMA, SCO = ISS.MergeSortand_RemoveDuplicates(mama, hhh, 80, tol=0.0001, keep_only_equivalent=True)

    print("%d Filtered matrix solutions with %s possible axes permutation" % (len(MAMA), PERMU))
    print(MAMA)
    print(SCO)


def test_indexation_generalRefunitCell_1():
    """
    test of indexation and unit cell refinement

    # lattice parameter : 2 1 4 75 90 120
    """
    B_input = np.array([[0.58438544724018371, 0.62634243462603045, -0.040527337709274316],
                    [0.0, 1.0352761804100832, -0.06698729810778066],
                    [0.0, 0.0, 0.24999999999999997]])

    data_filecor = np.array([[
    94.67479705, -1.77030003, 921.5841064, 877.0715942, 72.34869384], [
    73.69339752, 20.61630058, 572.9110107, 1218.955322, 72.01788330], [
    102.4393997, 8.985199928, 756.9523925, 755.0565795, 71.78106689], [
    80.51010131, 30.81780052, 381.7903137, 1118.804809, 69.76594543], [
    83.94760131, -11.9162998, 1077.943359, 1040.604248, 68.99935150], [  # 1 1 3
    63.29959869, 9.497099876, 754.0991210, 1386.598754, 68.80802154], [  # 0 1 3
    107.3254013, 19.63260078, 583.6207275, 661.9633789, 68.60917663], [
    84.57730102, 39.66809844, 180.0175933, 1058.541748, 64.89128112], [
    109.8495025, 29.49519920, 401.0426940, 589.9348754, 64.09487915], [
    111.0460968, -11.4846000, 1070.015625, 605.5197753, 62.07760238], [
    117.7855987, -1.76250004, 919.4577026, 488.4346923, 61.40348434], [
    70.78790283, -20.9498996, 1226.293212, 1267.149902, 61.09899902], [  # 1 1 2
    102.2435989, -20.2021999, 1213.381591, 745.6085205, 60.87683105], [
    122.3706970, 8.517900466, 761.3325805, 389.1836853, 59.37057495], [
    86.79679870, 47.08269882, -33.2585983, 1024.272338, 59.23816299], [
    110.6781005, 38.15859985, 208.5700988, 533.8281860, 59.13156890], [
    49.21139907, -1.77890002, 925.8856811, 1685.104736, 57.60239410], [  # 0 1 2
    91.72329711, -27.7292995, 1349.845825, 916.0936279, 57.46092987], [  # 2 1 3
    124.9682998, 18.72879981, 595.1906127, 304.0439147, 56.49592971]
    ])

    mat_tofind = DictLT.dict_Rot['OrientSurf111']

    a, b, c, alpha, beta, gamma = 2., 1., 4., 75., 90., 120.
#    latticeparams = a,b,c,alpha,beta,gamma
    B = CP.calc_B_RR([a, b, c, alpha, beta, gamma])
    print("B", B)
    # --- metric tensor
#    Gstar_metric = CP.Gstar_from_directlatticeparams(a,b,c,alpha,beta,gamma)

    twiceTheta, Chi = data_filecor[:, :2].T

    thechi_exp = np.array([twiceTheta / 2., Chi]).T

    table_angdist = calculdist_from_thetachi(thechi_exp, thechi_exp)

    ang_tol = 1
    n = 3
    spot_index = 5

    mat = matrices_from_onespot_new(spot_index, ang_tol,
                                table_angdist,
                                twiceTheta, Chi,
                                n, B,
                                verbose=1)

    index_matching = []
    for k, mat_sol in enumerate(mat[0]):
        if np.allclose(mat_sol, mat_tofind, atol=.2):
            index_matching.append([mat[0][k], mat[1][k], mat[2][k]])

    print("These are the solutions !...")
    print(index_matching)


def test_indexation_generalRefunitCell_2():
    """
    test of indexation and unit cell refinement

    [2,1,4,90,90,90] lattice parameter oriented with orientsurface111
    """
    data_filecor = np.array([[
    79.98930358, -0.01720000, 896.3380737, 1100.323608, 59.85998535], [  # 2 1 4
    72.14089965, -7.87809991, 1015.881286, 1227.356567, 59.33337783], [  # 2 1 3
    85.72489929, 7.331699848, 784.4506835, 1013.795471, 58.13841629], [  # 2 1 5
    61.93239974, -15.9895000, 1143.677978, 1422.501708, 55.33495330], [  # 2 1 2
    61.84360122, 15.91269969, 653.3759155, 1426.837646, 55.26354980], [  # 1 1 4
    89.83450317, -14.0176000, 1111.113281, 949.4226074, 55.25106430], [  # 3 1 4
    89.75630187, 14.00380039, 679.5944213, 953.0634155, 55.21326446], [  # 2 1 6
    95.69380187, -6.89139986, 999.4022216, 860.4885864, 54.51536941], [  # 3 1 5
    82.30560302, -21.1537990, 1229.820800, 1071.203369, 54.36821365], [  # 3 1 3
    68.30660247, 22.44009971, 542.4238891, 1319.939941, 54.00218200], [
    52.94110107, 8.390600204, 772.7620849, 1601.591430, 53.40956115]
    ])

    mat_tofind = DictLT.dict_Rot['OrientSurf111']

    a, b, c, alpha, beta, gamma = 2., 1., 4., 90., 90., 90.
#    latticeparams = a,b,c,alpha,beta,gamma
    B = CP.calc_B_RR([a, b, c, alpha, beta, gamma])
    # --- metric tensor
#    Gstar_metric = CP.Gstar_from_directlatticeparams(a,b,c,alpha,beta,gamma)

    twiceTheta, Chi = data_filecor[:, :2].T

    thechi_exp = np.array([twiceTheta / 2., Chi]).T

    table_angdist = calculdist_from_thetachi(thechi_exp, thechi_exp)

    ang_tol = 1
    n = 4
    spot_index = 1

    mat = matrices_from_onespot_new(spot_index, ang_tol,
                                table_angdist,
                                twiceTheta, Chi,
                                n, B,
                                verbose=0)

    print("mat", mat)

    index_matching = []
    for k, mat_sol in enumerate(mat[0]):
        if np.allclose(mat_sol, mat_tofind, atol=.2):
            index_matching.append([mat[0][k], mat[1][k], mat[2][k]])

    print("These are the solutions !...")
    print(index_matching)


def Test_getOrients_AnglesMatching():
    """
    test for the classical method implemented in Common_index_Method()
    """
    readfile = None

    spot_index_central = [0, 1, 2]
    key_material = 'UO2'
    emax = 22

    nbmax_probed = 8000000

    angleTolerance_LUT = .1
    MatchingRate_Angle_Tol = 0.001
    Matching_Threshold_Stop = 50.  # minimum to stop the loop for searching potential solution

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'toto'
        nbgrains = 3
        removespots = [0, 0, 0]  # , 0, 3, 10, 2, 1, 5]
        addspots = 0
#        removespots = None
#        addspots = None
        file_to_index = ISS.createFakeData(key_material, nbgrains,
                                       outputfilename=outputfilename,
                                       removespots=removespots,
                                       addspots=addspots)
    else:
        file_to_index = readfile

    # read data
    data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(file_to_index)[1:]
    tth = 2 * data_theta

    exp_data = np.array([tth, Chi])

    thechi_exp = np.array([data_theta, Chi]).T

    table_angdist = calculdist_from_thetachi(thechi_exp, thechi_exp)


    latticeparams = DictLT.dict_Materials[key_material][1]
    B = CP.calc_B_RR(latticeparams)

    nbmax_probed = min(nbmax_probed, len(data_theta))

    matrix, score, Threshold_reached = getOrients_AnglesLUT(spot_index_central,
                        table_angdist[:nbmax_probed, :nbmax_probed],
                        tth, Chi,
                        Matching_Threshold_Stop=Matching_Threshold_Stop,
                        n=3,  # up  to (332)
                        B=B,  # for cubic
                        LUT=None,
                        angleTolerance_LUT=angleTolerance_LUT,
                        MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                        key_material=key_material,
                        emax=emax,
                        verbose=0)
#
#    bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(bestmat, stats_res, Nb_criterium,
#                                                           tol=0.0001, allpermu=True)

    print("B", B)
    print("matrix", matrix)
    return matrix, score, detectorparameters  # , exp_data


def plotgrains(listmat, key_material, emax, file_to_index='toto.cor'):
    """
    plot grains spots and data from orientmatrix list up to 9
    """
    # read data
    data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(file_to_index)[1:]
    tth_exp = 2 * data_theta
    Chi_exp = Chi

    EXTINCTION = DictLT.dict_Materials[key_material][2]

    nb_matrices = 0
    all_tthchi = []
    for mat in listmat:

        print("mat in plotgrains", mat)

        grain = [np.eye(3), EXTINCTION, mat, key_material]

        (Twicetheta_theo, Chi_theo, Miller_ind_theo,
         posx_theo, posy_theo, Energy_theo) = LAUE.SimulateLaue(grain, 5, emax,
                                                                detectorparameters)

        all_tthchi.append([Twicetheta_theo, Chi_theo])

        nb_matrices += 1
        if nb_matrices == 9:
            break

    print("nb_of_matrices to plot", nb_matrices)
    if nb_matrices == 1:
        codefigure = 111
    if nb_matrices == 2:
        codefigure = 211
    if nb_matrices in (3, 4):
        codefigure = 221
    if nb_matrices in (5, 6):
        codefigure = 321
    if nb_matrices in (7, 8, 9):
        codefigure = 331

    index_fig = 0

#    ax.set_xlim((35, 145))
#    ax.set_ylim((-45, 45))

    dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
    nbcolors = len(dicocolor)

    # theo spots
    for i_mat in list(range(nb_matrices)):
        p.subplot(codefigure)

        if file_to_index is not None:
            # all exp spots
            p.scatter(tth_exp, Chi_exp,
                      s=40, c='w', marker='o', faceted=True, alpha=.5)

        # simul spots
        p.scatter(all_tthchi[i_mat][0], all_tthchi[i_mat][1],
               c=dicocolor[(i_mat + 1) % nbcolors], faceted=False)

        if index_fig < nb_matrices:
            index_fig += 1
            codefigure += 1
            if nb_matrices == 9:
                break

    p.show()

# ------------  Main            ------------------------------


if __name__ == '__main__':
    
    res = test_GetProximity2()

    # test_indexation_with_apriori_Matrices()
#     Test_Common_index_Method()
    # test_indexation_generalRefunitCell_1()
    # test_indexation_generalRefunitCell_2()

    # Test_Common_index_Method()

    """
    # building the map
    # nb elem = 51*23

    list_of_pointsinstance=[]
    list_of_signs=[]
    dic_ID_orient={}
    grain_counter=0
    for m in range(51*23):
        XX=m/51
        YY=m%51
        p=Map_point([XX,YY])
        p.orients=refine_orient(Main_Orient[m],array([3,3,3]))
        list_of_pointsinstance.append(p)

        nb_orient=len(p.orients)
        #print m,nb_orient
        if nb_orient>0:
            point_sign=[]
            for g in range(nb_orient):
                #print g,list(p.orients[g])
                signature_g=from3angles_to_int(list(p.orients[g]))
                point_sign.append(signature_g)
                if signature_g not in dic_ID_orient.values():
                    dic_ID_orient[grain_counter]=signature_g
                    grain_counter+=1
            list_of_signs.append(point_sign)
            p.orients_ID=point_sign
        else:
            list_of_signs.append([-1])
            p.orients_ID=[-1]

    print "I ve found %d grain in the map"%len(dic_ID_orient)
    #adding -1 = None
    dic_ID_orient[-1]=-1
    print dic_ID_orient

    print "Now put the general ID grains in all the point of the map"
    # inverser le dictionnaire...
    def invdict(adict):

        return dict([(val, key) for key, val in adict.items()])

    dic_orient_ID=invdict(dic_ID_orient)

    # lists grain id for each points
    grain_per_point=[map(lambda elem: dic_orient_ID[elem],ls) for ls in list_of_signs]

    # gives for each grain ID, the point instances where they were found ----------------------------
    localisation_grain={}
    for ID in dic_ID_orient.keys():
        localisation_grain[ID]=[]
    for p in list_of_pointsinstance:
        map(lambda cle_ID: localisation_grain[dic_orient_ID[cle_ID]].append(p.pos),p.orients_ID)
    table_pos=[array(localisation_grain[ID]) for ID in dic_ID_orient.keys()]

    """

    # -----------------------------------------------------------------------------------------------
    def make_colormarkerdict():
        """
        build a dictionary of color and marker to distinguish contribution of grains in scatter plot
        """
        # faudrait rendre aleatoire l'indice i_dict pour un meilleur rendu
        colorlist = ['k', (1.0, 0.0, 0.5), 'b', 'g', 'y', 'r', 'c', 'm', '0.75', (.2, .2, .2), (0.0, 0.5, 1.0)]
        markerlist = ['s', 'v', 'h', 'd', 'o', '^', '8']
        colormarkerdict = {}
        i_dict = 0
        for col in colorlist:
            for mark in markerlist:
                colormarkerdict[i_dict] = [col, mark]
                i_dict = i_dict + 1
        return colormarkerdict


    def plot_mapmajor(listoflocations, xmin, xmax, ymin, ymax):
        """
        example of map plot given list of coordinates ?? ( I do not remember actually)

        """
        p.close()
        colormarkerdict = make_colormarkerdict()

        tableofxcoord = []
        tableofycoord = []

        # move the last elem in first position the first grain ID corresponds now to unknown grain
        listoflocations.insert(0, listoflocations.pop(-1))

        for elem in (listoflocations[0],
                     listoflocations[8],
                     listoflocations[16],
                      listoflocations[24],
                       listoflocations[32]):
            tableofxcoord.append(elem[:, 0])
            tableofycoord.append(elem[:, 1])

        # print "tableofxcoord",tableofxcoord

        p.figure(figsize=(10, 7), dpi=80)

        kk = 0
        for xdata, ydata in zip(tuple(tableofxcoord), tuple(tableofycoord)):
            p.scatter(ydata, xdata, c=colormarkerdict[kk][0], marker=colormarkerdict[kk][1], s=500, alpha=0.5)
            # permutation of x y / array index ?...
            kk = kk + 1


        ticks_x = np.arange(xmin, xmax, 1)
        ticks_y = np.arange(ymin, ymax, 1)
        p.yticks(ticks_x)  # permutation of x y / array index ?...
        p.xticks(ticks_y)
        p.ylim(-1, xmax + 1)
        p.xlim(-1, ymax + 1)
        p.xlabel('x', fontsize=20)
        p.ylabel('y', fontsize=20)
        p.title('UO2_He_103_1275: Grain map. Unknown grains are black squares')
        p.grid(True)
        p.show()

    # plot_mapmajor(table_pos,0,22,0,50)




