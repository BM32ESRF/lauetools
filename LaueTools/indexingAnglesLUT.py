# -*- coding: utf-8 -*-
"""
module of lauetools project

purposes:

- classical method to index laue spots
- gnomonic projection
- hough transform (in development, feasability demonstrated)
- image matching (in development, feasability demonstrated)
- zone axes recognition (in development)

js micha May  2019
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import sys

import pylab as p
import numpy as np

if sys.version_info.major == 3:
    from . import lauecore as LAUE
    from . import CrystalParameters as CP
    from . import findorient as FindO
    from . import dict_LaueTools as DictLT
    from . import generaltools as GT
    from . import indexingSpotsSet as ISS  # for test only
    from . import matchingrate
else:
    import lauecore as LAUE
    import CrystalParameters as CP
    import findorient as FindO
    import dict_LaueTools as DictLT
    import generaltools as GT
    import indexingSpotsSet as ISS  # for test only
    import matchingrate

try:
    if sys.version_info.major == 3:
        from . import angulardist
    else:
        import angulardist
    USE_CYTHON = True
except ImportError:
    #print("-- warning. Cython compiled module for fast computation of angular distance is not installed!")
    USE_CYTHON = False

try:
    import wx
except ImportError:
    print("wx is not installed! Could be some trouble from this lack...")

# --- ------------ CONSTANTS
DEG = np.pi / 180.0
CST_ENERGYKEV = DictLT.CST_ENERGYKEV

# --- -------------  PROCEDURES
def Plot_compare_2thetachi(Angles,
                        twicetheta_data,
                        chi_data,
                        verbose=1,
                        key_material=14,
                        emax=25,
                        emin=5,
                        EULER=0,
                        exp_spots_list_selection=None,
                        dictmaterials=DictLT.dict_Materials):
    """
    plot data and simulation (given by list of 3 angles for orientation)
    in 2theta chi space (kf vector angles)
    """

    angle_X, angle_Y, angle_Z = Angles

    if not isinstance(EULER, (np.ndarray,)):
        if EULER == 0:
            mymat = GT.fromelemangles_toMatrix([angle_X, angle_Y, angle_Z])
        elif EULER == 1:
            mymat = GT.fromEULERangles_toMatrix([angle_X, angle_Y, angle_Z])
    else:
        if verbose:
            print("Using orientation Matrix for plotting")
        mymat = EULER

    # PATCH to use correctly getLaueSpots() of lauecore
    grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

    # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=1,
                                verbose=0,
                                dictmaterials=dictmaterials)

    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

    print("nb of spots in CCD frame", len(TwicethetaChi[0]))

    if exp_spots_list_selection is not None:  # to plot only selected list of exp. spots
        sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
        sel_chi = np.array(chi_data)[exp_spots_list_selection]

    p.title("Euler Angles [%.1f,%.1f,%.1f]" % (tuple(Angles)))
    if exp_spots_list_selection is not None:
        p.scatter(sel_2theta, sel_chi, s=40, c="w", marker="o", faceted=True, alpha=0.5)
    else:
        p.scatter(twicetheta_data, chi_data, s=40, c="w", marker="o", faceted=True, alpha=0.5)
    p.scatter(TwicethetaChi[0], TwicethetaChi[1], c="r", faceted=False)

    p.show()


def Plot_compare_2thetachi_multi(list_Angles,
                                twicetheta_data,
                                chi_data,
                                verbose=1,
                                emax=25,
                                emin=5,
                                key_material=14,
                                EULER=0,
                                exp_spots_list_selection=None,
                                title_plot="default",
                                figsize=(6, 6),
                                dpi=80,
                                dictmaterials=DictLT.dict_Materials):
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
        if not isinstance(EULER, (np.ndarray,)):
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
        grain = CP.Prepare_Grain(key_material, mymat, dictmaterials=dictmaterials)

        # array(vec) and array(indices) (here with fastcompute=1 array(indices)=0)
        # of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                    CST_ENERGYKEV / emin,
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    dictmaterials=dictmaterials)
        # 2theta,chi of spot which are on camera (with harmonics)
        TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1)

        if exp_spots_list_selection is None:  # to plot all exp. spots
            sel_2theta = np.array(twicetheta_data)
            sel_chi = np.array(chi_data)
        if isinstance(exp_spots_list_selection, (np.ndarray,)):
            sel_2theta = np.array(twicetheta_data)[exp_spots_list_selection]
            sel_chi = np.array(chi_data)[exp_spots_list_selection]
        elif isinstance(exp_spots_list_selection, int):  # it is a number # for plotting
            if exp_spots_list_selection > 1:
                # ind_max = min(len(twicetheta_data) - 1, exp_spots_list_selection)
                sel_2theta = np.array(twicetheta_data)[:exp_spots_list_selection]
                sel_chi = np.array(chi_data)[:exp_spots_list_selection]

        ax = fig.add_subplot(codefigure)
        if isinstance(title_plot, list):
            sco = title_plot[index_fig]
            p.title("nb close,<0.5deg: %d,%d  mean ang %.2f" % tuple(sco))
        else:
            if not isinstance(EULER, (np.ndarray,)):
                if EULER == 1:
                    p.title("Euler Angles [%.1f,%.1f,%.1f]" % (tuple(list_Angles[orient_index])))
            else:
                p.title("Orientation Matrix #%d" % orient_index)

        ax.set_xlim((35, 145))
        ax.set_ylim((-45, 45))
        # exp spots
        ax.scatter(sel_2theta, sel_chi, s=40, c="w", marker="o", faceted=True, alpha=0.5)
        # theo spots
        ax.scatter(TwicethetaChi[0], TwicethetaChi[1], c="r", faceted=False)
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
    if angle < 0.0:
        res = 180.0 + angle
    return res


def findInArray1D(array1D, val, tol):
    """
    return array of index where val has been found in input_array within tol as
    tolerance value
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
            for cc in gen_Nuplets(items[i + 1 :], n - 1):
                yield [items[i]] + cc


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


# --- -------------------  ANGLES LUT INDEXING
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

    #TODO: possiblity of rejecting some plane type according the structure fcc dia...)
    """

    # LUT_MAIN_CUBIC comes from findorient

    ind_sorted_LUT_MAIN_CUBIC = [np.argsort(elem) for elem in FindO.LUT_MAIN_CUBIC]
    sorted_table_angle = []
    for k in list(range(len(ind_sorted_LUT_MAIN_CUBIC))):
        # print len(LUT_MAIN_CUBIC[k])
        # print len(ind_sorted_LUT_MAIN_CUBIC[k])
        sorted_table_angle.append(
            (FindO.LUT_MAIN_CUBIC[k])[ind_sorted_LUT_MAIN_CUBIC[k]])

    if onlyclosest:
        one_value = [GT.find_closest(np.array(elem), np.array([angles_value]), tole)[:2]
            for elem in sorted_table_angle]
        # print "gjhgjh",one_value

        sol_one_value = []
        for k in list(range(len(one_value))):
            if return_index(one_value[k]):
                if verbose:
                    print("k", k, "  sortedindex ", one_value[k][0][0])
                sol_one_value.append([k, ind_sorted_LUT_MAIN_CUBIC[k][one_value[k][0][0]]])
        # LUT_MAIN_CUBIC[a,b] donne l'angle avec sol_one_value=[[a,b],[a2,b2],...]
        # a est l'indice du plan central, b est l'indice du plan

        # reading the solution
        # print sol_one_value
        planes_sol = []
        for m in list(range(len(sol_one_value))):
            first_plane_sol = FindO.convplanetypetoindice(FindO.INVDICOPLANE[sol_one_value[m][0]])
            second_plane_sol = FindO.DICOLISTNEIGHBOURS[
                FindO.INVDICOPLANE[sol_one_value[m][0]]][sol_one_value[m][1]]
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

        values = [findInArray1D(np.array(elem), angles_value, tole)
            for elem in sorted_table_angle]
        sol_one_value = []
        for k in list(range(len(values))):
            if len(values[k]):
                for ind in values[k]:
                    sol_one_value.append([k, ind_sorted_LUT_MAIN_CUBIC[k][ind]])

        planes_sol = []
        for m in list(range(len(sol_one_value))):
            first_plane_sol = FindO.convplanetypetoindice(FindO.INVDICOPLANE[sol_one_value[m][0]])
            second_plane_sol = FindO.DICOLISTNEIGHBOURS[
                FindO.INVDICOPLANE[sol_one_value[m][0]]
            ][sol_one_value[m][1]]
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
    from 1 spot (given by its index as defined in table_angdist[1])
    returns:
    [0] : list of [spots_index_1,spot_index_2]
                    where spot_index_2 is the index spot lying
                    at the tabulated distance of spot 1
    [1] : list of corresponding plane type: [plane type 1,plane 2]

    NB: plane type 1 is a family type (Miller indices always positive and arbitrarly ordered)
    instead of plane 2 which is an accurate plane type (order and sign is important with respect to the famiiy type 1)
    """

    from_spot1 = [
        Possible_planes(angle, tole=angulartolerance, verbose=0)
        for angle in table_angdist[spot_index_1]]

    couples_index = []
    couples_type = []

    for m in list(range(len(from_spot1))):
        nbcouples = len(from_spot1[m])
        if nbcouples > 0:
            couples_index.append([spot_index_1, m])
            couples_type.append(from_spot1[m])

    return couples_index, couples_type


def plane_type_attribution_twospots(spot_index_1, spot_index_2, angulartolerance, table_angdist):
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
    angle = GT.calculdist_from_thetachi(listspot, listspot)[0, 1]
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


def matrices_from_onespot(spot_index, ang_tol, table_angdist, Theta, Chi, verbose=0):
    """
    from one spot of index spot_index,
    ang_tol angular tolerance (deg) for look up table recognition,
    table_angdist all experimental distances
    Theta and Chi are the experimental Theta Chi spots coordinates
    """
    pairspots = []
    matrix_list = []
    pairplanes = []

    possible_couplespots, possible_coupleplanes = plane_type_attribution(spot_index, ang_tol,
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
            matrix = FindO.givematorient(plane_1,
                                            [2 * Theta[spot_index], Chi[spot_index]],
                                            plane_2,
                                            [2 * Theta[spot_index_2], Chi[spot_index_2]],
                                            verbose=0)

            matrix_list.append(matrix)
            pairplanes.append([plane_1, plane_2])
            pairspots.append(possible_couplespots[cs_index])

            if verbose:
                print("in matrices_from_onespot")
                print("pp_index", pp_index)
                print("couple planes ", plane_1, plane_2)
                print([2 * Theta[spot_index], Chi[spot_index]],
                    [2 * Theta[spot_index_2], Chi[spot_index_2]])
                print("spot_index_2", spot_index_2)
                print("matrix", matrix)

    return matrix_list, pairplanes, pairspots


def matrices_from_onespot_hkl(spot_index, LUT_tol_angle, table_angdist, twiceTheta_exp, Chi_exp,
                                            n, key_material, MaxRadiusHKL=False,
                                            hkl1=None,#FindO.HKL_CUBIC_UP3,
                                            hkl2=None,
                                            LUT=None,
                                            allow_restrictedLUT=False,
                                            verbose=1,
                                            dictmaterials=DictLT.dict_Materials,
                                            LUT_with_rules=True,
                                            excludespotspairs=None):
    """
    get all possibles UBs from one central spot and given its hkl1 miller indices

    spot_index_central      : integer index
    hkl1                    :  array or list of 3 elements  (shape = (n,3))
    table_angdist        : mutual angular distances between spots n*n symetric table
    LUT_tol_angle          : angular tolerance for computing the number of matching simulated
                        spots with exp. spots

    twiceTheta_exp, Chi_exp  : experimental spots angles
    n                        : angles reference LUT table order
    key_material            : string label for material

    hkl2                    : list of hkl2 to compute the LUT with hkl1

    allow_restrictedLUT           : flag to restrict LUT (removed hkl1 with negative l)

    LUT_with_rules:   True, apply Extinctions Rules when computing LUT

    return: - like matrices_from_onespot_new() -
    (list_orient_matrix,
         planes,
         pairspots)

    .. note::
        * Used in FileSeries
        * Used in AutoIndexation  (if max setA < max setB)
    """
    
    # print('input param',spot_index, LUT_tol_angle, table_angdist, twiceTheta_exp, Chi_exp,
    #                                         n, key_material, MaxRadiusHKL,
    #                                         hkl1, hkl2, LUT, allow_restrictedLUT,
    #                                         verbose, dictmaterials,
    #                                         LUT_with_rules,
    #                                         excludespotspairs)
    #verbose = 1
    if verbose:
        print("---- mode verbose = 1 --------")
        print('--------*****   matrices_from_onespot_hkl()   ******-----------')
    if verbose:
        print('\n\n***  in matrices_from_onespot_hkl\n\n')
        print("LUT_tol_angle", LUT_tol_angle)
        print('table_angdist.shape', table_angdist.shape)
        print('n', n)
        print('hkl1', hkl1)
        print('hkl2', hkl2)
        print('LUT', LUT)

    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    Distances_from_central_spot = table_angdist[spot_index]

    PPs_list = []

    latticeparams = dictmaterials[key_material][1]
    B = CP.calc_B_RR(latticeparams)

    if hkl1 is not None:
        hkl1 = np.array(hkl1)
        #print("hkl1 in matrices_from_onespot_hkl()", hkl1)

    if allow_restrictedLUT:
        # LUT restriction given by crystal structure
        allow_restrictedLUT = CP.isCubic(latticeparams)

    if LUT is None:
        if hkl2 is None:
            print("Computing hkl2 list for specific or cubic LUT in matrices_from_onespot_hkl()")
            # compute hkl2 outside loop
            hkl_all = GT.threeindices_up_to(n, remove_negative_l=allow_restrictedLUT)

            if LUT_with_rules:
                Rules = dictmaterials[key_material][2]
                hkl_all = CP.ApplyExtinctionrules(hkl_all, Rules)

            # filterharmonics:
            hkl_all = FindO.FilterHarmonics(hkl_all)

            hkl2 = hkl_all
    # default LUT or entered LUT as argument in function call
    else:
        # LUT will be used in next calculations and not recomputed
        if verbose: print("Using specific LUT in matrices_from_onespot_hkl()")

    # in case of hkl is a single 3D vector
    if len(hkl1.shape) == 1:
        hkl1 = np.array([hkl1])
    if len(hkl2.shape) == 1:
        hkl2 = np.array([hkl2])

    for spotindex_2, query_angle in enumerate(Distances_from_central_spot):
        if verbose:
            print("\n-*****----------------------------------------------------")
            print("< spot_index, spotindex_2 >  angle = ", spot_index, spotindex_2, query_angle)
            print("-*****----------------------------------------------------\n")

        # print('hkl1', hkl1)
        # print('hkl2', hkl2)

        # during the loop LUT is calculated just once
        (hkls, angles), LUT = FindO.PlanePairs_from2sets(query_angle,
                                            LUT_tol_angle,
                                            hkl1,
                                            hkl2,
                                            key_material,
                                            LUT=LUT,
                                            onlyclosest=0,
                                            verbose=verbose,
                                            dictmaterials=dictmaterials,
                                            LUT_with_rules=LUT_with_rules)

        if hkls is not None and (spot_index != spotindex_2):
            nbpairs = len(hkls)
            PPs_list.append([hkls, spotindex_2, nbpairs])
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spotindex_2, nbpairs)
                print('angles', angles)

    coords_exp = np.array([twiceTheta_exp, Chi_exp]).T
    coord_central_spot = coords_exp[spot_index]

    # if spot_index >=0:
    #     print('\nspot_index of matrices_from_onespot_hkl() !!!!!')
    #     print("len PPs_list in matrices_from_onespot_hkl", len(PPs_list))
    #     print('spot_index', spot_index)
    #     print('excludespotspairs',excludespotspairs)
        #print('coord_central_spot, coords_exp, B,excludespotspairs', coord_central_spot, coords_exp, B,excludespotspairs)
    Matrices_Res = Loop_on_PlanesPairs_and_Get_Matrices(PPs_list, spot_index,
                                                            coord_central_spot, coords_exp, B,
                                                            verbose=verbose,
                                                            excludespotspairs=excludespotspairs)
    #if spot_index == 4: print('Matrices_Res', Matrices_Res)
    if verbose:
        print('LUT in matrices_from_onespot_hkl', LUT)

    return Matrices_Res, hkl2, LUT


def matrices_from_onespot_new(spot_index, ang_tol, table_angdist, twiceTheta, Chi, n, B0,
                                                                        LUT=None,
                                                                        MaxRadiusHKL=False,
                                                                        applyExtinctionRules=None,
                                                                        verbose=0,
                                                                        excludespotspairs=None):
    r"""
    returns list of pair of planes and exp pairs of spots that match an angle in a reference LUT.
    LUT is computed from B (Gstar)

    .. note::
        - USED in automatic indexation
        - Used AutoIndexation module
        - Used in FileSeries

    :param spot_index: index of spot considered (must be lower than len(table_angdist) )
    :param ang_tol: angular tolerance (deg) for look up table matching
    :param table_angdist: all experimental distances square matrix
    :param 2Theta and Chi: the experimental 2Theta Chi spots coordinates

    For building the angles reference LUT:

    :param n: integer value corresponding the maximum miller index considered in angle value LUT
    :param B0: B0matrix of reference unit cell used TRIANGULAR UP
                            for extracting lattice parameter and building LUT

    orientation matrix is then given with respect the frame in which is expressed B

    :param applyExtinctionRules: apply hkl filtering when building the LUT (if LUT =None)
    """
    # possible_couplespots, possible_coupleplanes = plane_type_attribution(spot_index,
    # ang_tol,
    # table_angdist)

    Distances_from_central_spot = table_angdist[spot_index]

    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    PPs_list = []

    if LUT is None:
        print("LUT build in matrices_from_onespot_new()")
        print("cubicSymmetry is False for an exhaustive LUT")
        LUT = build_AnglesLUT(B0, n, MaxRadiusHKL=MaxRadiusHKL, cubicSymmetry=False,
                                                        applyExtinctionRules=applyExtinctionRules)

    for spotindex_2, angle in enumerate(Distances_from_central_spot):
        if angle ==0.0:
            continue
            
        if verbose:
            print("\n-*-*-*----------------------------------------------------")
            print("k,angle = ", spotindex_2, angle)
            print("-*-*-*----------------------------------------------------\n")

        hkls = FindO.PlanePairs_2(angle, ang_tol, LUT, onlyclosest=0, verbose=verbose)  # LUT is provided !

        if hkls is not None:
            nbpairs = len(hkls)
            PPs_list.append([hkls, spotindex_2, nbpairs])
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spotindex_2, nbpairs)

    coords_exp = np.array([twiceTheta, Chi]).T
    coord_central_spot = coords_exp[spot_index]

    return Loop_on_PlanesPairs_and_Get_Matrices(PPs_list, spot_index, coord_central_spot,
                                            coords_exp, B0, verbose=verbose,
                                            excludespotspairs=excludespotspairs)


def getUBs_and_MatchingRate(spot_index_1, spot_index_2, ang_tol_LUT, angdist, coords_1, coords_2,
                            n, B, twiceTheta_exp, Chi_exp, set_hkl_1=None,
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
                                                            worker=None,
                                                            dictmaterials=DictLT.dict_Materials,
                                                            LUT_with_rules=True,
                                                            excludespotspairs=None):
    """
    angdist   : scalar
    coords_1   2theta, chi
    twiceTheta_exp :  all 2theta values (to compute matching rate of Laue Patterns)

    LUT: look-up-table , if None, a LUT will be built

    LUT_with_rules:

    :return: (List_UBs, List_Scores), LUT

    from two spots only
    USED in manual indexation
    """

    print("\n\n ---------------------------- \n getUBs_and_MatchingRate() between\n spots pair "
        "[%d, %d] \n ---------------------------- \n\n " % (spot_index_1, spot_index_2))
        
    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    MAX_NB_SOLUTIONS = 30

    List_UBs = []  # matrix list
    List_Scores = []  # hall of fame BestScores_per_centralspot list

    #print('set_hkl_1 in getUBs_and_MatchingRate', set_hkl_1)

    (list_orient_matrix, planes, pairspots), LUT = UBs_from_twospotsdistance(spot_index_1,
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
                                                                verbose=verbose,
                                                                dictmaterials=dictmaterials,
                                                                LUT_with_rules=LUT_with_rules,
                                                                excludespotspairs=excludespotspairs)

    solutions_matorient_index = []
    solutions_spotscouple = []
    solutions_hklcouple = []
    solutions_matchingscores = []
    solutions_matchingrate = []

    if verbose:
        print("len(list_orient_matrix)", len(list_orient_matrix))
        print("#mat nb<%.2f       nb. theo. spots     mean       max    nb**2/nb_theo*mean     plane indices"
            % (ang_tol_MR))

    WORKEREXIST = 0
    if worker is not None:
        WORKEREXIST = 1
    # loop over orient matrix given from LUT recognition for one central spot
    nb_UB_matrices = len(list_orient_matrix)
    # TODO: use multiprocessing if nb_UB_matrices > 1000
    for mat_ind in list(range(nb_UB_matrices)):
        if WORKEREXIST:
            #             print "there is a worker !!"
            if worker._want_abort:
                print("\n\n!!!!!!! Indexation Aborted \n\n!!!!!!!\n")
                BestScores_per_centralspot = np.array([])
                worker.callbackfct(None)
                return
        if (mat_ind % 20) == 0:
            print("Calculating matching with exp. data for matrix #%d / %d" % (mat_ind,
                                                                                nb_UB_matrices))
        # compute matching rate and store if high
        AngRes = matchingrate.Angular_residues_np(list_orient_matrix[mat_ind],
                                                    twiceTheta_exp,
                                                    Chi_exp,
                                                    key_material=key_material,
                                                    emax=emax,
                                                    ResolutionAngstrom=ResolutionAngstrom,
                                                    ang_tol=ang_tol_MR,
                                                    detectorparameters=detectorparameters,
                                                    dictmaterials=dictmaterials)

        if AngRes is None:
            continue

        (allres, _, nbclose, nballres, _, max_residue) = AngRes

        if nbclose > Minimum_Nb_Matches:
            std_closematch = np.std(allres[allres < ang_tol_MR])
            if verbosedetails:

                mean_residue_closematch = np.mean(allres[allres < ang_tol_MR])
                # max_residue_closematch = np.max(allres[allres < ang_tol_MR])
                print("mat_ind      nbclose      fullnb      std_closematch     "
                                "mean_resid  max_resid  figmerit")
                print("%d        %d       %d       %.3f      %.3f       %.3f        %.3f"
                    % (mat_ind, nbclose, nballres, std_closematch, mean_residue_closematch,
                        max_residue,
                        nbclose ** 2 * 1.0 / nballres / std_closematch), "    ",
                        str(planes[mat_ind]), "  ", pairspots[mat_ind])

            #             print "mat_ind: %d" % mat_ind
            #             print "AngRes", AngRes
            # mat_ind, pairspots[mat_ind], planes[mat_ind], [nbclose, nballres, std_closematch], 100.0 * nbclose / nballres are the output for each evaluated matrix
            
            solutions_matorient_index.append(mat_ind)
            solutions_spotscouple.append(pairspots[mat_ind])
            solutions_hklcouple.append(planes[mat_ind])
            solutions_matchingscores.append([nbclose, nballres, std_closematch])
            solutions_matchingrate.append(100.0 * nbclose / nballres)

    BestScores_per_centralspot = np.array(solutions_matchingscores)

    # for one central spot if there are at least one potential solution
    if len(BestScores_per_centralspot) > 0:

        #             print "Got one solution for k_centspot_index: %d" % k_centspot_index

        # sort results
        rank = np.lexsort(keys=(BestScores_per_centralspot[:, 2],
                                BestScores_per_centralspot[:, 0]
                                * 1.0 / BestScores_per_centralspot[:, 1]))[::-1]

        hall_of_fame = BestScores_per_centralspot[rank]

        orient_index_fame = np.array(solutions_matorient_index)[rank]
        psb = np.array(solutions_spotscouple)[rank]
        ppb = np.array(solutions_hklcouple)[rank]

        # only for plot
        list_UBs_for_plot = np.array(list_orient_matrix)

        # one keeps only the best matrix

        for mm in list(range(min(MAX_NB_SOLUTIONS, len(orient_index_fame)))):
            bestmatrix = list_UBs_for_plot[orient_index_fame[mm]]
            bestscores = list(hall_of_fame[mm]) + list([psb[mm]]) + list([ppb[mm]])
            #                 print "bestmatrix", bestmatrix
            #                 print "bestscores", bestscores
            List_UBs.append(bestmatrix)
            List_Scores.append(bestscores)

    else:
        print("Zero orientation matrix has been found with nb of matches larger than the given threshold "
                                            "'Minimum_Nb_Matches' = %d" % Minimum_Nb_Matches)

    if WORKEREXIST:
        worker.fctOutputResults = (List_UBs, List_Scores), LUT

        print("finished!")
        #print("setting worker.fctOutputResults to", worker.fctOutputResults)
        worker.callbackfct("COMPLETED")

    return (List_UBs, List_Scores), LUT


def UBs_from_twospotsdistance(spot_index_1, spot_index_2, angle_tol, exp_angular_dist,
                                coords_1, coords_2,
                                n,
                                B,
                                LUT=None,
                                set_hkl_1=None,
                                set_hkls_2=None,
                                key_material=None,
                                MaxRadiusHKL=False,
                                allow_restrictedLUT=True,
                                verbose=0,
                                dictmaterials=DictLT.dict_Materials,
                                LUT_with_rules=True,
                                excludespotspairs=None):
    r"""
    returns list of pair of planes and exp pairs of spots that match an angle in a reference LUT.
    (a LUT if not given, is computed (and returned). LUT building uses B (Gstar)
    and set_hkl_1 and set_hkls_2)



    :param spot_index: index of spot considered (must be lower than len(table_angdist) )
    :param angle_tol: angular tolerance (deg) for look up table matching
    :param exp_angular_dist: experimental distance between q1,q2 (lattice planes normals)
    :param coords_exp: the experimental 2Theta Chi spots coordinates

    For building the angles reference LUT:

    :param n: integer value corresponding the maximum miller index considered in angle value LUT
    :param B: Bmatrix of reference unit cell used TRIANGULAR UP
                            for extracting lattice parameter and building LUT

    orientation matrix is then given with respect the frame in which is expressed B

    set_hkl_1     : if not None, [h,k,l] of spot #1

    LUT_with_rules: True, apply extinctions rules when computing LUT


    :return: (matrix_list, pairplanes, pairspots), LUT

    .. note:: USED in manual indexation

    .. todo::
        conflict between key_material and B ...
    """
    if verbose:
        print("using UBs_from_twospotsdistance()")
        print("coords_1", coords_1)
        print("coords_2", coords_2)

    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    PPs_list = []

    if set_hkl_1 is None:

        if LUT is None:
            print("LUT build in UBs_from_twospotsdistance()")
            hascubicSymmetry = CP.hasCubicSymmetry(key_material, dictmaterials=dictmaterials)

            if LUT_with_rules:
                Rules = dictmaterials[key_material][2]
            else:
                Rules = None

            LUT = build_AnglesLUT(B, n, MaxRadiusHKL=MaxRadiusHKL, cubicSymmetry=hascubicSymmetry,
                                        applyExtinctionRules=Rules)

        # LUT is provided now for sure!
        hkls = FindO.PlanePairs_2(exp_angular_dist, angle_tol, LUT, onlyclosest=0, verbose=verbose)

    #         print "nb of hkls found in LUT:", len(hkls)
    #         print "hkls:", hkls

    # when hkl1 is guessed (and set)
    elif set_hkl_1 is not None:
        latticeparams = dictmaterials[key_material][1]
        B = CP.calc_B_RR(latticeparams)

        if allow_restrictedLUT:
            # LUT restriction given by crystal structure
            allow_restrictedLUT = CP.isCubic(latticeparams)

        if set_hkls_2 is None:
            print("Computing hkl2 list for specific or cubic LUT in UBs_from_twospotsdistance()")
            # compute hkl2 outside loop:
            hkl_all = GT.threeindices_up_to(n, remove_negative_l=allow_restrictedLUT)

            if LUT_with_rules:
                Rules = dictmaterials[key_material][2]
                hkl_all = CP.ApplyExtinctionrules(hkl_all, Rules)

            # filterharmonics:
            hkl_all = FindO.FilterHarmonics(hkl_all)

            hkl2 = hkl_all

            print("new calculated hkl2", hkl2)

        (hkls, _), LUT = FindO.PlanePairs_from2sets(exp_angular_dist, angle_tol,
                                                set_hkl_1, hkl2,
                                                key_material, LUT=None, onlyclosest=0,
                                                verbose=verbose,
                                                LUT_with_rules=LUT_with_rules)

        if hkls is not None:
            print("nb found planes pairs :" % len(hkls))

    if hkls is not None and (spot_index_1 != spot_index_2):
        nbpairs = len(hkls)
        PPs_list.append([hkls, spot_index_2, nbpairs])
        if verbose:
            print("hkls, plane_indices spotindex_2, nbpairs", hkls, spot_index_2, nbpairs)

    #     print "PPs_list", PPs_list

    matrix_list, pairplanes, pairspots = Loop_on_PlanesPairs_and_Get_Matrices(PPs_list,
                                                                spot_index_1,
                                                                coords_1, coords_2,
                                                                B, verbose=verbose,
                                                                single_coords_2=True,
                                                                excludespotspairs=excludespotspairs)

    return (matrix_list, pairplanes, pairspots), LUT


def Loop_on_PlanesPairs_and_Get_Matrices(PP_list, spot_index, coord1, coords, B,
                                                    verbose=0, single_coords_2=False,
                                                    excludespotspairs=None):
    r"""
    loop on possible planes couples (PP) from recognised distances from a single spot (of index spot_index)

    :param excludespotspairs:  list of pair of spot indices already taken into account

    :return: matrix_list, pairplanes, pairspots

    .. note::
        USED in FileSeries
        USED in manual indexation
        USED in autoindexation

    coords  :  2theta chi spots coordinates
    """
    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    if verbose: print("** --Loop_on_PlanesPairs_and_Get_Matrices() --**")

    pairspots = []
    matrix_list = []
    pairplanes = []

    if PP_list == []:
        print('in Loop_on_PlanesPairs_and_Get_Matrices(), PP_list is empty')
        return matrix_list, pairplanes, pairspots

    # loop over all possible pairs of planes found in LUT
    for k, PP in enumerate(PP_list):

        hkls, spotindex_2, nbplanepairs = PP

        # skip already probed pair of spots
        if [spotindex_2, spot_index] in excludespotspairs or [spot_index, spotindex_2] in excludespotspairs:
            continue

        if spotindex_2 == spot_index:
            continue

        if verbose:
            print("** --\nLooking up for the %d planes pairs in LUT from exp. spots "
                        "(%d, %d): " % (nbplanepairs, spot_index, spotindex_2))

        excludespotspairs.append([spotindex_2, spot_index])
        excludespotspairs.append([spot_index, spotindex_2])

        if single_coords_2:
            coord2 = coords
        else:
            coord2 = coords[spotindex_2]

        hlks_shape = hkls.shape

        #print('hlks_shape',hlks_shape)
        if len(hlks_shape) == 2:
            nb_pairs = hlks_shape[0] / 2
        elif len(hlks_shape) == 3:
            nb_pairs = hlks_shape[0]

        # if nb_pairs == 1:
        #     hkls = [hkls]

        if verbose:
            print("\n************** k", k)
            print("PP[%d] = " % k, PP)
            print("hlks_shape", hlks_shape)
            print("hkls", hkls)
            print("nb_pairs", nb_pairs)
            print('spot_index, spotindex_2', spot_index, spotindex_2)

        # loop over the plane pairs
        for planepair in hkls:

            hkl1, hkl2 = planepair
            #print('planepair: ', planepair)
            # print("hkl1, hkl2 ", hkl1, hkl2)
            # print "coord1,coord2", coord1,coord2
            # print "spot_index, spotindex_2",spot_index, spotindex_2

            if np.sum(np.abs(np.cross(hkl1, hkl2))) < 0.00000001:
                #print("parallel planes ...")
                continue

            matrix = FindO.OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B,
                                                        verbose="no", frame="lauetools")

            #print("matrix",matrix)

            # matrix=givematorient(plane_1,[2*Theta[spot_index],Chi[spot_index]],plane_2,[2*Theta[spot_index_2],Chi[spot_index_2]],verbose=0)

            matrix_list.append(matrix)
            pairplanes.append([hkl1, hkl2])
            pairspots.append([spot_index, spotindex_2])

            # ---compute matrix by swaping hkl1 and hkl2

            matrix = FindO.OrientMatrix_from_2hkl(hkl2, coord1, hkl1, coord2, B,
                                                        verbose="no", frame="lauetools")

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
                                verbose=0,
                                dictmaterials=DictLT.dict_Materials,
                                LUT_with_rules=True):
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

    LUT_with_rules :

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
    if MatchingThresholdStop in (None, 0.0, 100.0, 0, 100):
        AngTol_LUTmatching = 0.5
        res_onespot = getOrientMatrices(spot_index,
                                        emax,
                                        table_angdist,
                                        twiceTheta / 2.0,
                                        Chi,
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
                                        nbspots_plot="all",  # nb exp spots to display if plot = 1
                                        addMatrix=None,
                                        verbose=1,
                                        dictmaterials=dictmaterials,
                                        LUT_with_rules=LUT_with_rules)

        print("res_onespot", res_onespot)
        matrix = res_onespot[0][0]
        infos = res_onespot[1][0]

        nbmatch, nbtheo, _, best_spotscouple_indices, best_hkl_couple = infos

        matching_rate = 100.0 * nbmatch / nbtheo
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
        if LUT_with_rules:
            Rules = dictmaterials[key_material][2]
        else:
            Rules = None
        LUT = build_AnglesLUT(B, n, MaxRadiusHKL=False,
                    cubicSymmetry=CP.hasCubicSymmetry(key_material, dictmaterials=dictmaterials),
                                                    applyExtinctionRules=Rules)

    # main loop over all distances from central and each spots2
    for spotindex_2, angle in enumerate(Distances_from_central_spot):
        if verbose:
            print("\n----------------------------------------------------------")
            print("k,angle = ", spotindex_2, angle)
            print("-----------------------------------------------------------\n")

        hkls = FindO.PlanePairs_2(angle, ang_tol, LUT, onlyclosest=1, verbose=0)

        if hkls is not None and (spot_index != spotindex_2):
            nbpairs = len(hkls)
            if verbose:
                print("hkls, plane_indices spotindex_2, nbpairs", hkls, spotindex_2, nbpairs)

            # handle hkls to compute a matrix and a matching rate
            if nbpairs == 1:

                hkl1, hkl2 = hkls[0].tolist()

                coord2 = coord[spotindex_2]

                matrix = FindO.OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B,
                                                            verbose="no", frame="lauetools")

                AngRes = matchingrate.Angular_residues(matrix,
                                                        twiceTheta,
                                                        Chi,
                                                        key_material=key_material,
                                                        emax=emax,
                                                        ang_tol=MatchingRate_Angle_Tol,
                                                        detectorparameters=detectorparameters)

                if AngRes is None:
                    continue

                (_, _, nbclose, nballres, _, _) = AngRes

                matching_rate = 100.0 * nbclose / nballres
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

                    Res = (matrix, matching_rate, [hkl1, hkl2], [spot_index, spotindex_2])
                    exitloop = True
                    break

                # ---compute matrix by swaping hkl1 and hkl2
                matrix = FindO.OrientMatrix_from_2hkl(
                    hkl2, coord1, hkl1, coord2, B, verbose="no", frame="lauetools")

                AngRes = matchingrate.Angular_residues(matrix,
                                                    twiceTheta,
                                                    Chi,
                                                    key_material=key_material,
                                                    emax=emax,
                                                    ang_tol=MatchingRate_Angle_Tol,
                                                    detectorparameters=detectorparameters)

                if AngRes is None:
                    continue

                (_, _, nbclose, nballres, _, _, ) = AngRes

                matching_rate = 100.0 * nbclose / nballres
                if verbose:
                    print("1P_B matching_rate %.1f" % matching_rate)

                if matching_rate > BestRecordedRes[1]:
                    BestRecordedRes = (matrix,
                                        matching_rate,
                                        [hkl1, hkl2],
                                        [spot_index, spotindex_2])

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

            if exitloop:
                break

            else:  # several pairs of solutions

                for m in list(range(nbpairs)):

                    # ---compute matrix by using hkl1 and hkl2
                    hkl1, hkl2 = hkls[m].tolist()

                    coord2 = coord[spotindex_2]

                    # print "hkl1, hkl2 ", hkl1, hkl2
                    # print "coord1,coord2", coord1,coord2
                    # print "spot_index, spotindex_2",spot_index, spotindex_2

                    matrix = FindO.OrientMatrix_from_2hkl(
                        hkl1, coord1, hkl2, coord2, B, verbose="no", frame="lauetools")

                    AngRes = matchingrate.Angular_residues(matrix,
                                                            twiceTheta,
                                                            Chi,
                                                            key_material=key_material,
                                                            emax=emax,
                                                            ang_tol=MatchingRate_Angle_Tol,
                                                            detectorparameters=detectorparameters)

                    if AngRes is None:
                        continue

                    (_, _, nbclose, nballres, _, _) = AngRes

                    matching_rate = 100.0 * nbclose / nballres
                    if verbose:
                        print("nP_A matching_rate %.1f" % matching_rate)

                    if matching_rate > BestRecordedRes[1]:
                        BestRecordedRes = (matrix,
                                        matching_rate,
                                        [hkl1, hkl2],
                                        [spot_index, spotindex_2])
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

                        Res = (matrix,
                                matching_rate,
                                [hkl1, hkl2],
                                [spot_index, spotindex_2])
                        exitloop = True
                        break

                    # ---compute matrix by using swaped hkl1 and hkl2

                    matrix = FindO.OrientMatrix_from_2hkl(
                        hkl2, coord1, hkl1, coord2, B, verbose="no", frame="lauetools")

                    AngRes = matchingrate.Angular_residues(matrix,
                                                        twiceTheta,
                                                        Chi,
                                                        key_material=key_material,
                                                        emax=emax,
                                                        ang_tol=MatchingRate_Angle_Tol,
                                                        detectorparameters=detectorparameters)

                    if AngRes is None:
                        continue

                    (_, _, nbclose, nballres, _, _) = AngRes

                    matching_rate = 100.0 * nbclose / nballres
                    if verbose:
                        print("nP_B matching_rate %.1f" % matching_rate)

                    if matching_rate > BestRecordedRes[1]:
                        BestRecordedRes = (matrix,
                                            matching_rate,
                                            [hkl1, hkl2],
                                            [spot_index, spotindex_2])
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

                        Res = (matrix,
                                matching_rate,
                                [hkl1, hkl2],
                                [spot_index, spotindex_2])
                        exitloop = True
                        break

            # exit the main for loop
            if exitloop:
                break

    return BestRecordedRes


def flatnestedlist(list_of_lists):
    """can be found in graingraph.py

    """
    return [y for x in list_of_lists for y in x]

def getOrientMatrices_SubSpotsSets(selectedspots_ind, emax, Theta_exp, Chi_exp, nLUT,
                                            key_material, LUT_tol_angle, detectorparameters,
                                            minimumNbMatches=15,
                                            LUT_with_rules=True):
    """find orientation matrices and scores from the mutual angles recognition
    in a subset of spots

    LUT_with_rules:

    .. note:: used in Autoindexation  if max setA > max setB
    """

    print("\n\n ----------------\n ---  getOrientMatrices_SubSpotsSets  --- \n --------------------\n\n")

    # set of mutual distances -------
    Theta = Theta_exp[selectedspots_ind]
    Chi = Chi_exp[selectedspots_ind]
    sorted_data = np.transpose(np.array([Theta, Chi]))

    print("Calculating all mutual angular distances of selected spots...")
    Tabledistance = GT.calculdist_from_thetachi(
            sorted_data, sorted_data)
    #-----------------------

    # Building LUT of reference angles
    dictmaterials = DictLT.dict_Materials
    latticeparams = dictmaterials[key_material][1]
    if LUT_with_rules:
        Rules = dictmaterials[key_material][2]
    else:
        Rules = None
    B = CP.calc_B_RR(latticeparams)
    LUT = build_AnglesLUT(B, nLUT, MaxRadiusHKL=False,
                cubicSymmetry=CP.hasCubicSymmetry(key_material, dictmaterials=dictmaterials),
                applyExtinctionRules=Rules)

    keep_only_equivalent = CP.isCubic(latticeparams)

    # ---------   START of INDEXATION  -------------
    bestmatList = []
    stats_resList = []

    #spot_index 1 and 2  = absolute index
    # i1,i2 local index to scan selectedspots_ind

    SolutionsFound = False
    # loop over all possible spots pairs in selected set of spots
    for i1, i2 in GT.allpairs_in_set(range(len(selectedspots_ind))):

        spot_index_1 = selectedspots_ind[i1]
        spot_index_2 = selectedspots_ind[i2]

        # print("\n**getOrientMatrices_SubSpotsSets *\n\ni1,i2, "
        #     "local_spotindex1,local_spotindex2", i1, i2, spot_index_1, spot_index_2)

        All_2thetas = Theta_exp*2.
        All_Chis = Chi_exp

        coords_1 = All_2thetas[spot_index_1], All_Chis[spot_index_1]
        coords_2 = All_2thetas[spot_index_2], All_Chis[spot_index_2]

        # Table of distances is very small (nb selected spots**2)
        expdistance_2spots = Tabledistance[i1, i2]

        # print('expdistance_2spots  = ', expdistance_2spots)

        UBS_MRS = getUBs_and_MatchingRate(spot_index_1,
                                            spot_index_2,
                                            LUT_tol_angle,
                                            expdistance_2spots,
                                            coords_1,
                                            coords_2,
                                            nLUT,
                                            B,
                                            All_2thetas,
                                            All_Chis,
                                            LUT=LUT,
                                            key_material=key_material,
                                            emax=emax,
                                            ang_tol_MR=LUT_tol_angle,
                                            detectorparameters=detectorparameters,
                                            verbosedetails=False)

        # print('UBS_MRS',UBS_MRS)

        # no matrices found for this pair i1,i2
        if len(UBS_MRS[0]) == 0:
            print('nb of UBs for this pair [%d, %d]: '%(spot_index_1, spot_index_2), len(UBS_MRS[0]))
            continue

        bestmat, stats_res = UBS_MRS

        #print("Merging matrices for this pair")
        # print("keep_only_equivalent = %s" % keep_only_equivalent)
        bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(bestmat,
                                                        stats_res,
                                                        minimumNbMatches,
                                                        tol=0.0001,
                                                        keep_only_equivalent=keep_only_equivalent)

        # print("final stats_res of this pair", stats_res)
        nb_sol = len(bestmat)
        print('final nb of UBs for this pair [%d, %d]'% (spot_index_1, spot_index_2), nb_sol)
        if nb_sol > 0:
            bestmatList.append(bestmat)
            stats_resList.append(stats_res)
            SolutionsFound = True


    if not SolutionsFound:
        return [], []

    BestMatrices = flatnestedlist(bestmatList)
    BestStats = flatnestedlist(stats_resList)

    nbsol = len(BestMatrices)
    print('nb solutions', nbsol)

    if nbsol > 1:
        BestMatrices, BestStats = ISS.MergeSortand_RemoveDuplicates(BestMatrices,
                                                                BestStats,
                                                                minimumNbMatches,
                                                                tol=0.0001,
                                                                keep_only_equivalent=keep_only_equivalent)

    print('final nb solutions', len(BestMatrices))
    print('BestMatrices\n')
    print(BestMatrices)

    return BestMatrices, BestStats


def getOrientMatrices_fromTwoSets(selectedspots_ind1, selectedspots_ind2,
                                    emax, Theta_exp, Chi_exp, nLUT,
                                    key_material, LUT_tol_angle, detectorparameters,
                                    set_hkl_1=None,
                                    minimumNbMatches=15,
                                    LUT_with_rules=True,
                                    excludespotspairs=None):
    """find orientation matrices and scores from the mutual angles recognition
    from two Sets of spots

    Return all matrices that have a least a 'Minimum_Nb_Matches'.
    Distances between two spots are compared to a reference
    angles look up table (LUT).

    .. note::
        - Used in AutoIndexation module (in LaueToolsGUI.py Classical Angular indexation)
        - USED in FileSeries

    :param selectedspots_ind1: list of exp. spot indices
    :param selectedspots_ind2: list of exp. spot indices. Spots indices corresponds to spots data location in Theta_exp and Chi_exp
    :param emax:    Maximum energy used in simulation of the Laue Pattern (the higher this value the larger
                                the number of theo. spots)
    :param Theta_exp: experimental 2theta/2  1D array
    :param Chi_exp: experimental chi  1d array
    :param nLUT:  integer for the maximum index of probed hkl when computing the LUT
    :param B: Triangular up matrix defining the reciprocal unit cell
    :param LUT_tol_angle: Angular tolerance (deg) below which exp. distance can be considered as recognised
                                in reference Look Uo Table
    MR_tol_angle:                :    Angular tolerance below which one exp spot can be linked to a single theo. spot
                                from simulation without any ambiguity
    :param minimumNbMatches: Minimum nb of matches (nb of links between exp and theo spots) above which the corresponding
                                orientation matrix will be returned as a likely good candidate.
    :param key_material: Element or structure label to the simulate the proper Laue Pattern (e.g.: 'Cu', 'Ge')

    :param detectorparameters: dictionary of detector parameters (key, value) which must contain
                            'kf_direction' , general position of detector plane
                            'detectordistance', sample to detector distance (mm).
                            (Could be approximate since only used to compute Matchiing Rate)
                            'detectordiameter', detector diameter (mm)

    :param set_hkl_1:

    :param LUT_with_rules:


    :return:
        * [0]  list of potential Orientation Matrices (UB)
        * [1]  list of corresponding scores (matching rate, nb of theo. Spots, mean angular deviation over exp and theo. links)
    """

    print("\n\n ----------------\n ---  getOrientMatrices_fromTwoSets  --- \n --------------------\n\n")

    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    # set of mutual distances -------
    if isinstance(selectedspots_ind1, int):
        selectedspots_ind1 = [selectedspots_ind1]
    if isinstance(selectedspots_ind2, int):
        selectedspots_ind2 = [selectedspots_ind2]

    Theta1 = Theta_exp[selectedspots_ind1]
    Chi1 = Chi_exp[selectedspots_ind1]
    Theta2 = Theta_exp[selectedspots_ind2]
    Chi2 = Chi_exp[selectedspots_ind2]
    sorted_data1 = np.transpose(np.array([Theta1, Chi1]))
    sorted_data2 = np.transpose(np.array([Theta2, Chi2]))

    # print("sorted_data1", sorted_data1)
    # print("sorted_data2", sorted_data2)

    # maybe useless
    if len(sorted_data1.shape) == 1:
        sorted_data1 = [sorted_data1]
    if len(sorted_data2.shape) == 1:
        sorted_data2 = [sorted_data2]

    print("Calculating all mutual angular distances of selected spots...")
    Tabledistance = np.transpose(GT.calculdist_from_thetachi(sorted_data1, sorted_data2))

    print("Tabledistance.shape", Tabledistance.shape)
    #-----------------------

    # Building LUT of reference angles
    dictmaterials = DictLT.dict_Materials
    latticeparams = dictmaterials[key_material][1]
    B = CP.calc_B_RR(latticeparams)
    keep_only_equivalent = CP.isCubic(latticeparams)

    LUT = None
    if set_hkl_1 is None:
        if LUT_with_rules:
            Rules = dictmaterials[key_material][2]
        else:
            Rules = None
        LUT = build_AnglesLUT(B, nLUT, MaxRadiusHKL=False,
                    cubicSymmetry=CP.hasCubicSymmetry(key_material, dictmaterials=dictmaterials),
                    applyExtinctionRules=Rules)

    # ---------   START of INDEXATION  -------------
    bestmatList = []
    stats_resList = []

    #spot_index 1 and 2  = absolute index
    # i1,i2 local index to scan selectedspots_ind

    SolutionsFound = False
    probed_pairs = []
    # loop over all possible spots pairs in selected set of spots
    for i1, i2 in GT.mutualpairs(range(len(selectedspots_ind1)),
                                range(len(selectedspots_ind2))):

        spot_index_1 = selectedspots_ind1[i1]
        spot_index_2 = selectedspots_ind2[i2]

        probedpair = [spot_index_1, spot_index_2]

        if spot_index_1 == spot_index_2:
            continue
        if probedpair in probed_pairs:
            continue

        # print("\n**getOrientMatrices_fromTwoSets *\n\ni1,i2, "
        #     "local_spotindex1,local_spotindex2", i1, i2, spot_index_1, spot_index_2)

        All_2thetas = Theta_exp*2.
        All_Chis = Chi_exp

        coords_1 = All_2thetas[spot_index_1], All_Chis[spot_index_1]
        coords_2 = All_2thetas[spot_index_2], All_Chis[spot_index_2]

        # Table of distances is very small (nb selected spots**2)
        expdistance_2spots = Tabledistance[i1, i2]

        #print('expdistance_2spots  = ', expdistance_2spots)

        UBS_MRS, LUT = getUBs_and_MatchingRate(spot_index_1, spot_index_2, LUT_tol_angle,
                                                    expdistance_2spots,
                                                    coords_1,
                                                    coords_2,
                                                    nLUT,
                                                    B,
                                                    All_2thetas,
                                                    All_Chis,
                                                    LUT=LUT,
                                                    key_material=key_material,
                                                    emax=emax,
                                                    ang_tol_MR=LUT_tol_angle,
                                                    set_hkl_1=set_hkl_1,
                                                    detectorparameters=detectorparameters,
                                                    Minimum_Nb_Matches=minimumNbMatches,
                                                    verbosedetails=False,
                                                    LUT_with_rules=LUT_with_rules)

        print('UBS_MRS', UBS_MRS)

        excludespotspairs.append([spot_index_1, spot_index_2])
        excludespotspairs.append([spot_index_2, spot_index_1])

        # no matrices found for this pair i1,i2
        if len(UBS_MRS[0]) == 0:
            print('nb of UBs for this pair [%d, %d]: '%(spot_index_1, spot_index_2), len(UBS_MRS[0]))
            continue

        bestmat, stats_res = UBS_MRS

        if len(bestmat) > 1:

            # print("Merging matrices for this pair")
            # print("keep_only_equivalent = %s" % keep_only_equivalent)
            bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(bestmat,
                                                            stats_res,
                                                            minimumNbMatches,
                                                            tol=0.0001,
                                                            keep_only_equivalent=keep_only_equivalent)

        print("final stats_res of this pair", stats_res)
        nb_sol = len(bestmat)
        print('final nb of UBs for this spots pair [%d, %d]   : '%(spot_index_1, spot_index_2), nb_sol)
        if nb_sol > 0:
            bestmatList.append(bestmat)
            stats_resList.append(stats_res)
            SolutionsFound = True


    if not SolutionsFound:
        return [], []

    BestMatrices = flatnestedlist(bestmatList)
    BestStats = flatnestedlist(stats_resList)

    nbsol = len(BestMatrices)
    print('nb solutions', nbsol)

    if nbsol > 1:
        BestMatrices, BestStats = ISS.MergeSortand_RemoveDuplicates(BestMatrices,
                                                                BestStats,
                                                                minimumNbMatches,
                                                                tol=0.0001,
                                                                keep_only_equivalent=keep_only_equivalent)

    print('final nb solutions', len(BestMatrices))
    print('BestMatrices\n')
    print(BestMatrices)

    return BestMatrices, BestStats


def getOrientMatrices(spot_index_central, energy_max, Tab_angl_dist, Theta_exp, Chi_exp, n=3,
                                            ResolutionAngstrom=False,
                                            B=np.eye(3),  # for cubic
                                            cubicSymmetry=False,
                                            LUT=None,
                                            LUT_tol_angle=0.5,
                                            MR_tol_angle=0.2,
                                            Minimum_Nb_Matches=15,
                                            key_material="",
                                            plot=0,
                                            nbbestplot=1,
                                            nbspots_plot="all",  # nb exp spots to display if plot = 1
                                            addMatrix=None,
                                            verbose=1,
                                            detectorparameters=None,
                                            set_central_spots_hkl=None,
                                            verbosedetails=True,
                                            gauge=None,
                                            dictmaterials=DictLT.dict_Materials,
                                            MaxRadiusHKL=False,
                                            LUT_with_rules=True,
                                            excludespotspairs=None):
    """
    Return all matrices that have a matching rate Minimum_Nb_Matches.
    Distances between two spots are compared to a reference
    angles look up table (LUT).

    .. note::
        - Used in AutoIndexation module (in LaueToolsGUI.py Classical Angular indexation)
        - USED in FileSeries

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
    LUT                            : if LUT is provided   the LUT won't be recomputed. It can save some time...
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
    LUT_with_rules:

    Output:
    [0]  candidate Matrices
    [1]  corresponding score (matching rate, nb of theo. Spots, mean angular deviation over exp and theo. links)

    if spot_index_central is a list of spot index, then only one successful result is return per spot
    """
    if verbosedetails:
        print('\n  -------   mode verbosedetails  ---------------')
        print("print details\n")

    if excludespotspairs is None:
        excludespotspairs = [[0, 0]]

    nbofpeaks = len(Tab_angl_dist) - 1  # nbmaxprobed-1

    #     print "Tab_angl_dist", Tab_angl_dist

    if isinstance(spot_index_central, (tuple,)):
        spot_index_central = list(spot_index_central)
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
        raise ValueError(
            "Tab_angl_dist of size %d is too small and does not contain distance with spot #%d"
            % (nbofpeaks, max(list_spot_central_indices)))

    if key_material == "":
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

    if verbosedetails: print("set_central_spots_hkl", set_central_spots_hkl)

    if set_central_spots_hkl not in (None, "None"):
        print("set_central_spots_hkl is not None in getOrientMatrices()")
        set_central_spots_hkl = np.array(set_central_spots_hkl)
        print("set_central_spots_hkl", set_central_spots_hkl)
        print("set_central_spots_hkl.shape", set_central_spots_hkl.shape)
        if set_central_spots_hkl.shape == (3,):
            print("case: 1a")
            set_central_spots_hkl_list = np.tile(
                set_central_spots_hkl, (len(list_spot_central_indices), 1))
        elif set_central_spots_hkl.shape == (1, 3):
            set_central_spots_hkl_list = np.tile(
                set_central_spots_hkl[0], (len(list_spot_central_indices), 1))
            print("case: 1b")
        else:
            print("case: 2")
            for ll, _ in enumerate(list_spot_central_indices):
                if ll < len(set_central_spots_hkl):
                    set_central_spots_hkl_list[ll] = set_central_spots_hkl[ll]

    if verbosedetails:
        print("set_central_spots_hkl_list", set_central_spots_hkl_list)
        print("cubicSymmetry", cubicSymmetry)

    if detectorparameters["kf_direction"] == "X>0":
        allow_restrictedLUT = False
    else:
        allow_restrictedLUT = cubicSymmetry

    #     if cubicSymmetry:
    #         hkl1s = FindO.HKL_CUBIC

    if verbosedetails: print("LUT_tol_angle", LUT_tol_angle)

    if gauge:
        gaugecount = 0
        gauge.SetValue(gaugecount)

    # set of hkl for computing specific LUT when hkl is given for central spot
    hkl2 = None
    LUTcubic = None
    LUTspecific = None

    first_cspot_idx = list_spot_central_indices[0]
    excludespotspairs = [[first_cspot_idx, first_cspot_idx]]

    # --- loop over central spots -------------------------------------------------------
    for k_centspot_index, spot_index_central in enumerate(list_spot_central_indices):
        if verbose:
            print("*---****-----getOrientMatrices() ------------------------------------*")
            print("Calculating all possible matrices from exp spot #%d and the %d other(s)"
                % (spot_index_central, nbofpeaks))
        # list_orient_matrix, planes, pairspots = matrices_from_onespot(spot_index_central,
        # LUT_tol_angle,
        # Tab_angl_dist,
        # Theta_exp,Chi_exp,
        # verbose=0)
        # read hkl for central spots defined in a list
        if set_central_spots_hkl_list is not None:
            hkl = set_central_spots_hkl_list[k_centspot_index]

        # if hkl for central spots IS defined
        #print("hkl in getOrientMatrices", hkl, type(hkl))
        if hkl is "None":
            hkl = None

        if hkl is not None:

            # find some potential matrices from recognised distances
            # in LUT by assuming hkl of spot_index_central

            # compute a specific LUT from 1 hkl and hkls up to n
            if verbosedetails:
                print("using LUTspecific")
                if LUTspecific is not None:
                    print("LUTspecific is not None for k_centspot_index %d in getOrientMatrices()"
                        % k_centspot_index)
                else:
                    print("LUTspecific is None for k_centspot_index %d in getOrientMatrices()"
                        % k_centspot_index)

            ((list_orient_matrix, planes, pairspots),
            hkl2,
            LUTspecific) = matrices_from_onespot_hkl(spot_index_central,
                                                        LUT_tol_angle,
                                                        Tab_angl_dist,
                                                        twiceTheta_exp,
                                                        Chi_exp,
                                                        n,
                                                        key_material,
                                                        MaxRadiusHKL=False,
                                                        hkl1=hkl,
                                                        hkl2=hkl2,
                                                        LUT=LUTspecific,
                                                        allow_restrictedLUT=allow_restrictedLUT,
                                                        verbose=verbose,
                                                        dictmaterials=dictmaterials,
                                                        LUT_with_rules=LUT_with_rules,
                                                        excludespotspairs=excludespotspairs)

        # if hkl for central spots IS NOT known  (si is None...)
        else:
            # TODO: retrieve cubic LUT if already calculated
            if cubicSymmetry:
                # compute a specific LUT from 1 hkl and hkls up to n
                if verbosedetails:
                    print("using LUTcubic")
                    if LUTcubic is not None:
                        print("LUTcubic is not None for k_centspot_index %d in getOrientMatrices()"
                            % k_centspot_index)
                    else:
                        print("LUTcubic is None for k_centspot_index %d in getOrientMatrices()" % k_centspot_index)

                if n == 3:
                    hkl1 = FindO.HKL_CUBIC_UP3
                elif n == 4:
                    hkl1 = FindO.HKL_CUBIC_UP4
                elif n == 5:
                    hkl1 = FindO.HKL_CUBIC_UP5
                elif n == 6:
                    hkl1 = FindO.HKL_CUBIC_UP6
                elif n == 7:
                    hkl1 = FindO.HKL_CUBIC_UP7
                else:
                    hkl1 = GT.threeindicesfamily(n)

                # Rules = dictmaterials[key_material][2]
                # hkl1 = CP.ApplyExtinctionrules(np.array(hkl1), Rules)

                (list_orient_matrix, planes, pairspots,
                ), hkl2, LUTcubic = matrices_from_onespot_hkl(spot_index_central,
                                                            LUT_tol_angle,
                                                            Tab_angl_dist,
                                                            twiceTheta_exp,
                                                            Chi_exp,
                                                            n,
                                                            key_material,
                                                            MaxRadiusHKL=False,
                                                            hkl1=hkl1,
                                                            hkl2=hkl2,
                                                            LUT=LUTcubic,
                                                            allow_restrictedLUT=allow_restrictedLUT,
                                                            verbose=verbose,
                                                            dictmaterials=dictmaterials,
                                                            LUT_with_rules=LUT_with_rules,
                                                            excludespotspairs=excludespotspairs)
            # general crystallographic structures cases (excluding then cubic structure)
            else:
                # --- building angles reference Look up table (LUT) from B and n
                if LUT is None:
                    # use following LUT
                    print("building LUT in getOrientMatrices()")

                    if LUT_with_rules:
                        Rules = dictmaterials[key_material][2]
                    else:
                        Rules = None
                    LUT = build_AnglesLUT(B, n, MaxRadiusHKL=MaxRadiusHKL,
                                                    cubicSymmetry=cubicSymmetry,
                                                    applyExtinctionRules=Rules)

                if verbose: print("using general non cubic LUT")
                # find some potential matrices from recognised distances in LUT
                (list_orient_matrix, planes, pairspots) = matrices_from_onespot_new(
                                                                    spot_index_central,
                                                                    LUT_tol_angle,
                                                                    Tab_angl_dist,
                                                                    twiceTheta_exp,
                                                                    Chi_exp,
                                                                    n,
                                                                    B,
                                                                    LUT=LUT,
                                                                    verbose=verbose,
                                                                    excludespotspairs=excludespotspairs)
        if len(list_orient_matrix) == 0:
            print('\n\n ----- len(list_orient_matrix)  is 0 \n\n', len(list_orient_matrix))

        if gauge:
            # TODO: VERY DIRTY...
            gaugecount = (k_centspot_index + 1) * nbofpeaks
            # print "gaugecount += 100",gaugecount
            gauge.SetValue(gaugecount)
            wx.Yield()

        solutions_matorient_index = []
        solutions_spotscouple = []
        solutions_hklcouple = []
        solutions_matchingscores = []
        solutions_matchingrate = []

        if verbosedetails:
            print("len(list_orient_matrix)", len(list_orient_matrix))
            # print "key_material",key_material
            # print "\n"
            print("#mat nb<%.2f       nb. theo. spots     mean       max    nb**2/nb_theo*mean     plane indices"
                % (MR_tol_angle))

        # --- loop over orient matrix given from LUT recognition for one central spot
        currentspotindex2 = -1
        for mat_ind in list(range(len(list_orient_matrix))):
            # compute matching (indexation) rate
            AngRes = matchingrate.Angular_residues_np(list_orient_matrix[mat_ind],
                                                        twiceTheta_exp,
                                                        Chi_exp,
                                                        key_material=key_material,
                                                        emax=energy_max,
                                                        ResolutionAngstrom=ResolutionAngstrom,
                                                        ang_tol=MR_tol_angle,
                                                        detectorparameters=detectorparameters,
                                                        dictmaterials=dictmaterials)

            if AngRes is None:
                continue

            (allres, _, nbclose, nballres, mean_residue, max_residue) = AngRes
            # store matching rate  if it is high
            if nbclose >= Minimum_Nb_Matches:
                std_closematch = np.std(allres[allres < MR_tol_angle])
                if verbosedetails:
                    print("%d        %d       %d       %.3f      %.3f       %.3f        %.3f"
                        % (mat_ind, nbclose, nballres, std_closematch, mean_residue,
                                    max_residue, nbclose ** 2 * 1.0 / nballres / std_closematch),
                        "    ", str(planes[mat_ind]), "  ", pairspots[mat_ind])
                    print("matrix ==", list_orient_matrix[mat_ind])

                spotindex2 = pairspots[mat_ind][1]
                if currentspotindex2 != spotindex2:
                    if verbose:
                        print("calculating matching rates of solutions for exp. spots", pairspots[mat_ind])
                    currentspotindex2 = spotindex2

                solutions_matorient_index.append(mat_ind)
                solutions_spotscouple.append(pairspots[mat_ind])
                solutions_hklcouple.append(planes[mat_ind])
                # solutions_matchingscores.append([nbclose, nballres, mean_residue])
                solutions_matchingscores.append([nbclose, nballres, std_closematch])
                solutions_matchingrate.append(100.0 * nbclose / nballres)
                # print list_orient_matrix[mat_ind]
            else:
                # print("Poor matching rate for exp. spots",pairspots[mat_ind])
                pass

        BestScores_per_centralspot[k_centspot_index] = np.array(solutions_matchingscores)

        # for one central spot if there are at least one potential solution
        if len(BestScores_per_centralspot[k_centspot_index]) > 0:

            # print "Got one solution for k_centspot_index: %d" % k_centspot_index
            # sort results
            # rank = np.lexsort(keys=(BestScores_per_centralspot[k_centspot_index][:, 2],
            #         BestScores_per_centralspot[k_centspot_index][:, 0]
            #         * 1.0 / BestScores_per_centralspot[k_centspot_index][:, 1]))[::-1]

            rank = np.lexsort(keys=(BestScores_per_centralspot[k_centspot_index][:, 2],
                   1./np.array(solutions_matchingrate)))

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
                List_UBs.append(bestmatrix)
                List_Scores.append(bestscores)

        else:
            print("Sorry! No orientation matrix found with nb of matches larger than %d"
                % Minimum_Nb_Matches)
            if verbose:
                print("Try to:")
                print("- decrease Ns (MNMS: minimum number of matched spots)")
                print("- increase angular tolerances (distance recognition and/or matching)")
                print("- increase nbofpeaks (ISSS: Intense spot set size)")
                print("- increase Energy max")

    # Consider immediately (if any) a list of a priori known matrices
    if addMatrix is not None:
        orient_index_best_add = []
        pair_spots_best_add = []
        pair_planes_best_add = []
        score_best_add = []

        # # restart indexation from the lastindex = len(BestScores_per_centralspot) - 1
        # newindex = len(list_UBs_for_plot) # for a single central spot ...
        # # TODO !!: implement many central spots

        # calculate residues from additional matrix
        # loop over added matrices
        print("loop over added matrices")
        for mat_ind, added_mat in enumerate(addMatrix):
            AngRes = matchingrate.Angular_residues_np(added_mat,
                                                    2 * Theta_exp,
                                                    Chi_exp,
                                                    key_material=key_material,
                                                    emax=energy_max,
                                                    ResolutionAngstrom=ResolutionAngstrom,
                                                    ang_tol=MR_tol_angle)

            if AngRes is None:
                continue

            (allres, _, nbclose, nballres, mean_residue, max_residue) = AngRes

            if nbclose > Minimum_Nb_Matches:
                if verbose:
                    print("%d        %d                   %d                %.3f       %.3f "
                        % (mat_ind, nbclose, nballres, mean_residue, max_residue),
                        "unknown              unknown ")
                std_closematch = np.std(allres[allres < MR_tol_angle])
                orient_index_best_add.append(mat_ind)
                pair_spots_best_add.append(np.array([0, 0]))  # not defined
                pair_planes_best_add.append(np.array([[0, 0, 0], [0, 0, 0]]))  # not defined
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
                List_Scores.append(list(score_best_add[mm])
                                + list([pair_spots_best_add[mm]])
                                + list([pair_planes_best_add[mm]]))

    # results and plot for only one central spot
    if len(list_spot_central_indices) == 1:

        if plot:
            if nbspots_plot == "all":
                _nbspots_plot = len(Chi_exp)
            else:
                _nbspots_plot = nbspots_plot

            Plot_compare_2thetachi_multi(orient_index_fame[:9],
                                        2 * Theta_exp,
                                        Chi_exp,
                                        EULER=list_UBs_for_plot,
                                        key_material=key_material,
                                        emax=energy_max,
                                        verbose=0,
                                        exp_spots_list_selection=_nbspots_plot,
                                        title_plot=list(hall_of_fame),
                                        figsize=(8, 8),
                                        dpi=70)
            # savefig('spot_'+str(spot_index_central[0])+'.png')

        print("return best matrix and matching scores for the one central_spot")
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
            if nbspots_plot == "all":
                _nbspots_plot = len(Chi_exp)
            else:
                _nbspots_plot = nbspots_plot

            Plot_compare_2thetachi_multi(np.arange(min(nbmatrixfound, 9)),
                                            2 * Theta_exp,
                                            Chi_exp,
                                            key_material=key_material,
                                            emax=energy_max,
                                            verbose=0,
                                            EULER=MATOS,
                                            exp_spots_list_selection=_nbspots_plot,
                                            title_plot=list(HF))
            # savefig('spot_'+str(spot_index_central[0])+'.png')

        # returned results for several central spots
        return MATOS, HF


def build_AnglesLUT(B0matrix, n, MaxRadiusHKL=False, cubicSymmetry=False, applyExtinctionRules=None):
    """
    build a Look-up-table LUT from a B0 Matrix up to index n
    (higher miller plane [n,n,n])

    cubicSymmetry   : flag to restrict LUT to use only HKL with L>=0
    """
    latticeparameters = CP.directlatticeparameters_fromBmatrix(B0matrix)
    return build_AnglesLUT_fromlatticeparameters(latticeparameters, n,
                                                MaxRadiusHKL,
                                                cubicSymmetry,
                                                applyExtinctionRules)


def build_AnglesLUT_fromlatticeparameters(latticeparameters, n,
                                    MaxRadiusHKL=False,
                                    cubicSymmetry=False,
                                    applyExtinctionRules=None,
                                    verbose=0):
    r"""
    build a Look-up-table from the 6 lattice parameters (a,b,c,alpha,beta,gamma) up to index n
    (higher miller plane [n,n,n])

    :param cubicSymmetry: True to restrict LUT to have only HKL with L>=0

    :param MaxRadiusHKL: False or largest value of sqrt(H**2+k**2+l**2) to keep HKL in LUT

    :return: LUT  (5 elements tuple: sorted_ind, sorted_angles, indy, tab_side_size, hkl_all
     see FindOrient.GenerateLookUpTable)

    .. todo::
        * change name of MaxRadiusHKL
        * add extinction rule
    """
    a, b, c, AA, BB, CC = latticeparameters

    
    print("\n------ build_AnglesLUT_fromlatticeparameters -------")
    if verbose:
        print('latticeparameters', latticeparameters)
        print("and n=%d" % n)
        print("MaxRadiusHKL", MaxRadiusHKL)
        print("cubicSymmetry", cubicSymmetry)

    # metric tensor
    Gstar_metric = CP.Gstar_from_directlatticeparams(a, b, c, AA, BB, CC)

    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n, remove_negative_l=cubicSymmetry)
    #hkl_all = GT.threeindices_up_to(n)


    if applyExtinctionRules is not None:
        nb0 = len(hkl_all)
        hkl_all = CP.ApplyExtinctionrules(hkl_all, applyExtinctionRules)
        nb1 = len(hkl_all)
        if verbose: print('%d hkls have been removed over %d.\nNow LUT contains %d hkls' % (nb0-nb1, nb0, nb1))

        # print('hkl_all')
        # print(hkl_all)

    # filterharmonics
    hkl_all = CP.FilterHarmonics_2(hkl_all)

    if MaxRadiusHKL not in (False, 0, 0.0):
        B0matrix = CP.calc_B_RR(latticeparameters)
        H, K, L = hkl_all.T
        xs, ys, zs = np.dot(B0matrix, hkl_all.T) # vector G* in x,y,z Angst-1

        if verbose: print('nb of nodes', len(xs))
        maxxs, maxys, maxzs = np.amax(xs), np.amax(ys), np.amax(zs)
        mindist = min(maxxs, maxys, maxzs)

        cond = (xs**2 + ys**2 + zs**2 <= mindist**2)
        Hf = H[cond]
        Kf = K[cond]
        Lf = L[cond]

        if verbose: print('nb of nodes after resolution filtering %d\n'%len(Hf))

        # HKLnorm = np.sqrt(np.sum(hkl_all ** 2, axis=1))

        # H, K, L = hkl_all.T
        # Condit = HKLnorm < (1.0 / MaxRadiusHKL)

        # # print "Condit",Condit
        # H = np.compress(Condit, H)
        # K = np.compress(Condit, K)
        # L = np.compress(Condit, L)

        # hkl_all = np.array([H, K, L]).T

        hkl_all = np.array([Hf, Kf, Lf]).T

    # GenerateLookUpTable
    LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)

    return LUT


def getOrients_AnglesLUT(spot_index_central,
                        Tab_angl_dist,
                        TwiceTheta,
                        Chi,
                        Matching_Threshold_Stop=40.0,
                        angleTolerance_LUT=0.5,
                        MatchingRate_Angle_Tol=0.5,
                        n=3,
                        B=np.eye(3),
                        LUT=None,
                        key_material=None,
                        emax=25.0,
                        absoluteindex=None,
                        detectorparameters=None,
                        verbose=1,
                        dictmaterials=DictLT.dict_Materials,
                        applyExtinctionRules=None,
                        LUT_with_rules=True):
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

    LUT_with_rules:

    Output:
    [0]  candidate Matrices
    [1]  corresponding score (matching rate, nb of theo. Spots, mean angular deviation over exp and theo. links)

    if spot_index_central is a list of spot index, then only one successful result is return per spot
    """
    nbofpeaks = len(Tab_angl_dist) - 1

    if isinstance(spot_index_central, (list, np.ndarray)):
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

    # --- building angles reference Look up table (LUT) from B and n
    if LUT is None:
        LUT = build_AnglesLUT(B, n, MaxRadiusHKL=False,
                                cubicSymmetry=CP.hasCubicSymmetry(key_material,
                                dictmaterials=dictmaterials),
                                applyExtinctionRules=applyExtinctionRules)

    twiceTheta = np.array(TwiceTheta)

    ResMatrix, ResScore = None, None

    AllRes = []
    AllMatrixres = []
    Threshold_reached = False

    for spot_index_central in scan_over_index:

        print("\n*-------------getOrients_AnglesLUT()--------------------------------*")
        print("Calculating all possible matrices from exp spot #%d and the %d others"
            % (spot_index_central, nbofpeaks))
        if absoluteindex is not None:
            print("central spot absolute index : %d" % absoluteindex[spot_index_central])

        Res = getOrientMatrix_from_onespot(spot_index_central,
                                        angleTolerance_LUT,
                                        Tab_angl_dist,
                                        twiceTheta,
                                        Chi,
                                        n,
                                        B,
                                        LUT=LUT,
                                        MatchingThresholdStop=Matching_Threshold_Stop,
                                        key_material=key_material,
                                        emax=emax,
                                        MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                                        verbose=0,
                                        detectorparameters=detectorparameters,
                                        LUT_with_rules=LUT_with_rules)

        print("Res", Res)
        print("for spot central relative index %d" % spot_index_central)

        # matching rate corresponds to a high matching
        if Res[1] >= Matching_Threshold_Stop:
            print("spot_index_central", spot_index_central)
            if absoluteindex is not None:
                print("central spot absolute index : %d" % absoluteindex[spot_index_central])
            #            print "Res", Res[1]
            print("with %d exp spots" % len(twiceTheta))

            (matrix, matching_rate, pair_hkl, pair_spots) = Res
            if verbose:
                print("Res Angles LUT matching")
                print(Res)
                print("pair hkls", pair_hkl)
                print("pair spots", pair_spots)
                if absoluteindex is not None:
                    print("pair spots absolute index : [%d,%d]"
                        % (absoluteindex[pair_spots[0]], absoluteindex[pair_spots[1]]))

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

    if not Threshold_reached:
        ArRes = np.array(AllRes)
        bestres = np.argmax(ArRes[:, 1])

        ResMatrix, ResScore = AllMatrixres[bestres], ArRes[bestres]

    return ResMatrix, ResScore, Threshold_reached
