# -*- coding: utf-8 -*-
"""
Modules that creates and uses lookuptable of distance between pairs of 
planes to propose a approximate orientation matrix
ONLY FOR CUBIC structure

Lookuptable is divided with respect to the a known plane [1,0,0],[1,1,0],...,up to [3,2,1] (central plane)
For each family of plane (hkl) all equivalent planes of this family are listed [h,k,l],[-h,k,l], ...
that is to say all permutations and sign inversions.

LUT_MAIN_CUBIC sums up the distances list for each central plane to neighbouring planes (below 90 deg)

From two identified planes, calculation of orientation matrix U can be done
There is no refinement of U according to other spots

TODO: generalise to other structure and include symetry considerations in selecting part of the lookuptable

"""
__version__ = '$Revision$'
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import string

import numpy as np
from scipy.linalg.basic import lstsq
from numpy import linalg as LA

import CrystalParameters as CP
import find2thetachi as F2TC
import generaltools as GT

from dict_LaueTools import dict_Materials

#--- ---------  CONSTANTS
DEG = np.pi / 180.


#--- ------ Cubic ANGULAR DISTANCE
DICOPLANE = {100:0, 110:1, 111:2, 210:3, 211:4, 221:5, 310:6, 311:7, 321:8}
INVDICOPLANE = {0:100, 1:110, 2:111, 3:210, 4:211, 5:221, 6:310, 7:311, 8:321}

LISTNEIGHBORS_100 = [[0, 1, 0],

                   [1, 1, 0],
                   [0, 1, 1],

                   [1, 1, 1],

                   [2, 1, 0],
                   [1, 0, 2],
                   [0, 1, 2],

                   [2, 1, 1],
                   [1, 1, 2],

                   [2, 2, 1],
                   [1, 2, 2],

                   [1, 0, 3],
                   [0, 1, 3],
                   [3, 1, 0],

                   [3, 1, 1],
                   [1, 1, 3],

                   [1, 2, 3],
                   [2, 1, 3],
                   [3, 1, 2]
                   ]
LISTCUT100 = [0, 1, 3, 4, 7, 9, 11, 14, 16, 19]
SLICING100 = [[0, 1], [1, 3], [3, 4], [4, 7], [7, 9], [9, 11], [11, 14], [14, 16], [16, 19]]

LISTNEIGHBORS_110 = [[1, 0, 0],
                   [0, 0, 1],

                   [-1, 1, 0],
                   [0, 1, 1],

                   [1, 1, 1],
                   [-1, 1, 1],

                   [2, 1, 0],
                    [2, 0, 1],
                   [1, 0, 2],

                   [2, 1, 1],
                   [-1, 2, 1],
                   [1, 1, 2],
                   [-1, 1, 2],

                   [2, 2, 1],
                   [2, 1, 2],
                   [2, -1, 2],
                   [2, -2, 1],

                   [1, 0, 3],
                   [1, 3, 0],
                   [0, 3, 1],
                   [-1, 3, 0],

                   [3, 1, 1],
                   [1, 1, 3],
                   [-1, 1, 3],

                   [3, 2, 1],
                   [3, 1, 2],
                   [1, 2, 3],
                   [-1, 3, 2],
                   [-2, 3, 1]
                   ]
LISTCUT110 = [0, 2, 4, 6, 9, 13, 17, 21, 24, 29]
SLICING110 = [[0, 2], [2, 4], [4, 6], [6, 9], [9, 13], [13, 17], [17, 21], [21, 24], [24, 29]]

LISTNEIGHBORS_111 = [[1, 0, 0],

                   [1, 1, 0],
                   [-1, 1, 0],

                   [-1, 1, 1],

                   [1, 2, 0],
                   [-1, 2, 0],

                   [1, 1, 2],
                   [-1, 1, 2],
                   [-2, 1, 1],

                   [2, 2, 1],
                   [-1, 2, 2],
                   [-2, 1, 2],

                   [1, 0, 3],
                   [-1, 0, 3],

                   [1, 1, 3],
                   [-1, 1, 3],
                   [3, -1, -1],

                   [1, 2, 3],
                   [-1, 2, 3],
                   [-2, 1, 3],
                   [-3, 1, 2]
                   ]
LISTCUT111 = [0, 1, 3, 4, 6, 9, 12, 14, 17, 21]
SLICING111 = [[0, 1], [1, 3], [3, 4], [4, 6], [6, 9], [9, 12], [12, 14], [14, 17], [17, 21]]

LISTNEIGHBORS_210 = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],

                   [1, 1, 0],
                   [1, 0, 1],
                   [1, -1, 0],

                   [1, 1, 1],
                   [1, -1, 1],

                   [2, -1, 0],
                   [1, 2, 0],
                   [1, 0, 2],
                   [0, 1, 2],
                   [-1, 2, 0],

                   [2, 1, 1],
                   [1, 2, 1],
                   [1, 1, 2],
                   [1, -1, 2],
                   [1, -2, 1],

                   [2, 2, 1],
                   [2, 1, 2],
                   [1, 2, 2],
                   [2, -1, 2],
                   [2, -2, 1],
                   [-1, 2, 2],

                   [3, 1, 0],
                   [3, 0, 1],
                   [1, 3, 0],
                   [0, 3, 1],
                   [1, 0, 3],
                   [0, 1, 3],

                   [3, 1, 1],
                   [1, 3, 1],
                   [1, 1, 3],
                   [1, -1, 3],

                   [3, 2, 1],
                   [2, 3, 1],
                   [1, 3, 2],
                   [1, 2, 3],
                   [2, -1, 3],
                   [-1, 3, 2],
                   [1, -2, 3]
                   ]
LISTCUT210 = [0, 3, 6, 8, 13, 18, 24, 30, 34, 42]
SLICING210 = [[0, 3], [3, 6], [6, 8], [8, 13], [13, 18], [18, 24], [24, 30], [30, 34], [34, 42]]

LISTNEIGHBORS_211 = [[1, 0, 0],
                   [0, 1, 0],

                   [1, 1, 0],
                   [0, 1, 1],
                   [1, -1, 0],
                   [0, 1, -1],

                   [1, 1, 1],
                   [1, -1, 1],
                   [1, -1, -1],

                   [2, 1, 0],
                   [1, 2, 0],
                   [2, -1, 0],
                   [0, -1, 2],
                   [1, 0, -2],

                   [1, 2, 1],
                   [2, 1, -1],
                   [1, 2, -1],
                   [2, -1, -1],
                   [1, -2, 1],

                   [2, 2, 1],
                   [1, 2, 2],
                   [2, 2, -1],
                   [2, -2, 1],
                   [1, 2, -2],
                   [2, -1, -2],

                   [3, 1, 0],
                   [3, -1, 0],
                   [0, 1, 3],
                   [0, -1, 3],
                   [-1, 3, 0],

                   [3, 1, 1],
                   [3, 1, -1],
                   [3, -1, -1],
                   [-1, 1, 3],
                   [-1, -1, 3],

                   [3, 2, 1],
                   [2, 3, 1],
                   [3, 2, -1],
                   [2, 3, -1],
                   [3, -2, 1],
                   [3, -2, -1],
                   [2, -3, 1],
                   [2, -3, -1]
                   ]
LISTCUT211 = [0, 2, 6, 9, 14, 19, 25, 30, 35, 43]
SLICING211 = [[0, 2], [2, 6], [6, 9], [9, 14], [14, 19], [19, 25], [25, 30], [30, 35], [35, 43]]

LISTNEIGHBORS_221 = [[1, 0, 0],
                   [0, 0, 1],

                   [1, 1, 0],
                   [1, 0, 1],
                   [1, 0, -1],
                   [1, -1, 0],

                   [1, 1, 1],
                   [1, 1, -1],
                   [1, -1, 1],

                   [2, 1, 0],
                   [2, 0, 1],
                   [1, 0, 2],
                   [2, 0, -1],
                   [2, -1, 0],
                   [1, 0, -2],

                   [2, 1, 1],
                   [1, 1, 2],
                   [2, 1, -1],
                   [2, -1, 1],
                   [1, 1, -2],
                   [2, -1, -1],

                    [2, 1, 2],
                   [2, 2, -1],
                   [2, 1, -2],
                   [2, -2, 1],
                   [2, -1, -2],

                   [3, 1, 0],
                   [3, 0, 1],
                   [3, 0, -1],
                   [3, -1, 0],
                   [-1, 0, 3],

                   [3, 1, 1],
                   [3, 1, -1],
                   [3, -1, 1],
                   [3, -1, -1],
                   [1, 1, -3],

                   [3, 2, 1],
                   [3, 1, 2],
                   [3, 2, -1],
                   [3, 1, -2],
                   [2, -1, 3],
                    [3, -2, 1],
                   [3, -1, -2],
                   [3, -2, -1]
                   ]
LISTCUT221 = [0, 2, 6, 9, 15, 21, 26, 31, 36, 44]
SLICING221 = [[0, 2], [2, 6], [6, 9], [9, 15], [15, 21], [21, 26], [26, 31], [31, 36], [36, 44]]

LISTNEIGHBORS_310 = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],

                   [1, 1, 0],
                   [1, 0, 1],
                   [1, -1, 0],
                   [0, 1, 1],

                   [1, 1, 1],
                   [1, -1, 1],

                   [2, 1, 0],
                   [2, 0, 1],
                   [1, 2, 0],
                   [1, 0, 2],
                   [0, 2, 1],
                   [0, 1, 2],

                   [2, 1, 1],
                   [2, -1, 1],
                   [1, 1, 2],
                   [1, -1, 2],
                   [1, -2, 1],

                   [2, 2, 1],
                   [2, 1, 2],
                   [2, -1, 2],
                   [2, -2, 1],
                   [1, -2, 2],

                   [3, 0, 1],
                   [3, -1, 0],
                   [1, 3, 0],
                   [1, 0, 3],
                   [0, 1, 3],
                   [1, -3, 0],

                    [3, 1, 1],
                   [3, -1, 1],
                   [1, 3, 1],
                   [1, 1, 3],
                   [1, -1, 3],
                   [1, -3, 1],

                   [3, 2, 1],
                   [3, 1, 2],
                   [2, 3, 1],
                   [3, -1, 2],
                   [3, -2, 1],
                   [1, 3, 2],
                   [2, -1, 3],
                   [2, -3, 1],
                   [1, -2, 3],
                   [1, -3, 2]
                   ]
LISTCUT310 = [0, 3, 7, 9, 15, 20, 25, 31, 37, 47]
SLICING310 = [[0, 3], [3, 7], [7, 9], [9, 15], [15, 20], [20, 25], [25, 31], [31, 37], [37, 47]]

LISTNEIGHBORS_311 = [[1, 0, 0],
                   [0, 1, 0],

                   [1, 1, 0],
                   [0, 1, 1],
                   [0, 1, -1],

                   [1, 1, 1],
                   [1, -1, 1],
                   [1, -1, -1],

                   [2, 1, 0],
                   [1, 2, 0],
                   [0, 1, 2],
                   [0, -1, 2],

                   [2, 1, 1],
                   [2, 1, -1],
                   [2, -1, -1],
                   [1, 1, -2],
                   [1, -1, -2],

                   [2, 2, 1],
                   [2, 2, -1],
                   [2, -2, 1],
                   [2, -2, -1],
                   [-1, 2, 2],

                   [3, 1, 0],
                   [3, -1, 0],
                   [1, 3, 0],
                   [0, 1, 3],
                   [0, -1, 3],
                   [-1, 3, 0],

                   [3, 1, -1],
                   [3, -1, -1],
                   [1, 3, -1],
                   [1, -3, 1],

                   [3, 2, 1],
                   [3, 2, -1],
                   [3, -2, 1],
                   [3, -2, -1],
                   [2, -3, 1],
                   [2, -3, -1]
                   ]
LISTCUT311 = [0, 2, 5, 8, 12, 17, 22, 28, 32, 38]
SLICING311 = [[0, 2], [2, 5], [5, 8], [8, 12], [12, 17], [17, 22], [22, 28], [28, 32], [32, 38]]

LISTNEIGHBORS_321 = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],

                   [1, 1, 0],
                   [1, 0, 1],
                   [0, 1, 1],
                   [1, 0, -1],
                   [1, -1, 0],

                   [1, 1, 1],
                   [1, 1, -1],
                   [1, -1, 1],
                   [-1, 1, 1],

                   [2, 1, 0],
                   [1, 2, 0],
                   [1, 0, 2],
                   [0, 1, 2],
                   [0, 2, -1],
                   [1, 0, -2],
                   [0, -1, 2],

                   [2, 1, 1],
                   [1, 2, 1],
                   [2, 1, -1],
                   [1, 2, -1],
                   [2, -1, 1],
                   [2, -1, -1],
                   [-1, 2, 1],
                   [-1, 2, -1],

                   [2, 2, 1],
                   [2, 1, 2],
                   [2, 2, -1],
                   [2, 1, -2],
                   [1, 2, -2],
                   [2, -2, 1],
                   [2, -1, -2],
                   [2, -2, -1],

                   [3, 1, 0],
                   [3, 0, 1],
                   [1, 3, 0],
                   [3, 0, -1],
                   [3, -1, 0],
                   [1, 0, 3],
                   [0, 3, -1],
                   [-1, 3, 0],
                   [0, -1, 3],
                   [1, 0, -3],

                   [3, 1, 1],
                   [3, 1, -1],
                   [3, -1, 1],
                   [3, -1, -1],
                   [-1, 3, 1],
                   [-1, 3, -1],

                   [3, 1, 2],
                   [3, 2, -1],
                   [3, 1, -2],
                   [1, 2, 3],
                   [3, -1, 2],
                   [2, -1, 3],
                   [3, -2, 1],
                   [3, -1, -2],
                   [3, -2, -1],
                   [1, -2, 3],
                   [2, -3, 1]

                   ]
LISTCUT321 = [0, 3, 8, 12, 19, 27, 35, 45, 51, 62]
SLICING321 = [[0, 3], [3, 8], [8, 12], [12, 19], [19, 27], [27, 35], [35, 45], [45, 51], [51, 62]]

DICOLISTNEIGHBOURS = {100:LISTNEIGHBORS_100,
                    110:LISTNEIGHBORS_110,
                    111:LISTNEIGHBORS_111,
                    210:LISTNEIGHBORS_210,
                    211:LISTNEIGHBORS_211,
                    221:LISTNEIGHBORS_221,
                    310:LISTNEIGHBORS_310,
                    311:LISTNEIGHBORS_311,
                    321:LISTNEIGHBORS_321}

DICOSLICING = {100:SLICING100,
                    110:SLICING110,
                    111:SLICING111,
                    210:SLICING210,
                    211:SLICING211,
                    221:SLICING221,
                    310:SLICING310,
                    311:SLICING311,
                    321:SLICING321}

DICOINDICE = {100:[1, 0, 0],
                     110:[1, 1, 0],
                     111:[1, 1, 1],
                     210:[2, 1, 0],
                     211:[2, 1, 1],
                     221:[2, 2, 1],
                     310:[3, 1, 0],
                     311:[3, 1, 1],
                     321:[3, 2, 1]}


#--- ------------------  END of tables and dictionnaries
def anglebetween(lista, listb):
    """
    Returns angle (deg) between two 3-elements lists

    """
    veca = np.array(lista)
    vecb = np.array(listb)
    norma = np.sqrt(np.dot(veca, veca))
    normb = np.sqrt(np.dot(vecb, vecb))
    return np.arccos(np.dot(veca, vecb) / (norma * normb)) / DEG


def prodtab_100():
    """
    produces angular distance from [1,0,0] plane with other planes up to (321) type

    """
    listangles_100 = [anglebetween([1, 0, 0], elemref) for elemref in LISTNEIGHBORS_100]
    return np.array(listangles_100)


def prodtab_110():
    """
    produces angular distance from [1,1,0] plane with other planes up to (321) type

    """
    listangles_110 = [anglebetween([1, 1, 0], elemref) for elemref in LISTNEIGHBORS_110]
    return np.array(listangles_110)


def prodtab_111():
    """
    produces angular distance from [1,1,1] plane with other planes up to (321) type

    """
    listangles_111 = [anglebetween([1, 1, 1], elemref) for elemref in LISTNEIGHBORS_111]
    return np.array(listangles_111)


def prodtab_210():
    """
    produces angular distance from [2,1,0] plane with other planes up to (321) type

    """
    listangles_210 = [anglebetween([2, 1, 0], elemref) for elemref in LISTNEIGHBORS_210]
    return np.array(listangles_210)


def prodtab_211():
    """
    produces angular distance from [2,1,1] plane with other planes up to (321) type

    """
    listangles_211 = [anglebetween([2, 1, 1], elemref) for elemref in LISTNEIGHBORS_211]
    return np.array(listangles_211)


def prodtab_221():
    """
    produces angular distance from [2,2,1] plane with other planes up to (321) type

    """
    listangles_221 = [anglebetween([2, 2, 1], elemref) for elemref in LISTNEIGHBORS_221]
    return np.array(listangles_221)


def prodtab_310():
    """
    produces angular distance from [3,1,0] plane with other planes up to (321) type

    """
    listangles_310 = [anglebetween([3, 1, 0], elemref) for elemref in LISTNEIGHBORS_310]
    return np.array(listangles_310)


def prodtab_311():
    """
    produces angular distance from [3,1,1] plane with other planes up to (321) type

    """
    listangles_311 = [anglebetween([3, 1, 1], elemref) for elemref in LISTNEIGHBORS_311]
    return np.array(listangles_311)


def prodtab_321():
    """
    produces angular distance from [3,2,2] plane with other planes up to (321) type

    """
    listangles_321 = [anglebetween([3, 2, 1], elemref) for elemref in LISTNEIGHBORS_321]
    return np.array(listangles_321)
#--- --------------  LUT for Cubic
LUT_MAIN_CUBIC = [prodtab_100(),
               prodtab_110(),
               prodtab_111(),
               prodtab_210(),
               prodtab_211(),
               prodtab_221(),
               prodtab_310(),
               prodtab_311(),
               prodtab_321()]


#--- ----------   FUNCTIONS
def convplanetypetoindice(myinteger):
    """

    Converts plane type to list of integer
    ex: 320 -> [3,2,0]
    CAUTION: valip only for abc with a,b,c<10 !!

    """
    stri = str(myinteger)
    return [string.atoi(elem) for elem in stri]


def convindicetoplanetype(listindice):
    """
    Converts miller plane (3 indices) into plane type for recognition
    where indices are sorted in decreasing order
    ex: [1,-2,-1]   -> 211
    """
    listpositiveinteger = [abs(elem) for elem in listindice]
    listpositiveinteger.sort(reverse=True)
    resint = 100 * listpositiveinteger[0] + 10 * listpositiveinteger[1] + 1 * listpositiveinteger[2]
    return resint


def findneighbours(central_type, neighbour_type, angle, tolerance, verbose='yes'):
    """
    Returns plane(s) solution of the matching between angular distance (angle) and lookup table 
    between plane central type and neighbour type plane)
    with reespect to angular tolerance (tolerance)
    """
    investigatedlistneighbours = DICOLISTNEIGHBOURS[central_type]
    slicing = DICOSLICING[central_type]
    extractindiceplane = DICOPLANE[neighbour_type]

    # select all equivalent neighbours planes of the family given by  neighbour_type in lookuptable
    extracted = investigatedlistneighbours[slicing[extractindiceplane][0]:slicing[extractindiceplane][1]]
    # compute the distance matrix from central plane and just above selected planes
    extractprodtab = [anglebetween(DICOINDICE[central_type], elemref) for elemref in extracted]

    if verbose == 'yes':
        print("possible ang. dist. between (%s,%s) planes:" % (central_type, neighbour_type))
        print(extractprodtab)

    # position in extracprodtab where there is a match
    winnerindicefound = GT.find_closest(np.array([angle * 1.]), np.array(extractprodtab), tolerance)[1]
    # matching planes list
    possibleplanes = np.take(extracted, winnerindicefound, axis=0)
    if len(possibleplanes) == 0:
        if verbose == 'yes':
            print("No planes of type ", str(neighbour_type), " is found around ", str(central_type), " at ", angle, " within ", tolerance, " deg")
        return None
    elif len(possibleplanes) > 1:  # very rare case for family with small multiplicity and small ang. tolerance
        if verbose == 'yes':
            print("Ambiguity! I found more than two planes! You may reduce the angular tolerance")
        return possibleplanes
    else:  # matching planes list generally contain 1 solution
        errortheoexp = angle - np.take(extractprodtab, winnerindicefound)[0]
        listindicesecondplane = possibleplanes[0].tolist()
        if verbose == 'yes':
            print("-------************-------------*************-------")
            print("Excellent! I've found an unique solution")
            print("First plane is ", convplanetypetoindice(central_type))
            print("Second plane is ", listindicesecondplane, " (ang. error of ", errortheoexp, " deg.)")
            print("------------------")
        return listindicesecondplane


def proposematrix(indpt1, indpt2, recogn, angulartolerance, verbose='yes'):
    """

    Gives orientation matrix from two points in recogn

    recogn: list of [2theta,chi,planetype,spot index in rawdata]
    indpt1,indpt2: index of spots in recogn
    angulartolerance: angular tolerance between experimental 
    distance and distance in lookup table between two known indexed planes

    """
    _verbose = verbose

    indexpoint1 = indpt1
    indexpoint2 = indpt2
    coord1 = np.array(recogn[indexpoint1][:2])
    coord2 = np.array(recogn[indexpoint2][:2])
    didist = GT.distfrom2thetachi(coord1, coord2)
    if verbose == 'yes':
        print("distance between the two recognised spots: ", didist, " deg.")

    hkl1 = convplanetypetoindice(recogn[indexpoint1][2])
    # find a matching pair (planetype 1,planetype 2)
    hkl2 = findneighbours(recogn[indexpoint1][2], recogn[indexpoint2][2], didist, angulartolerance, verbose=_verbose)
    # print "hkl2 in findorient",hkl2
    if hkl2:
        return givematorient(hkl1, coord1, hkl2, coord2, verbose=_verbose)
    else:
        if verbose == 'yes':
            print("Unfortunately, one spot has not been well recognised !!")
        return None


def constructMat(matrice_P, qq1, qq2, qq3prod):
    """
    Construction of rotation matrix from:
    matrice_P  columns = (G1,G2,G1^G2)     in a*,b*,c* frame
    qq1,qq2,qq3prod  respectively three vectors in  x,y,z space

    """
    qq3 = qq3prod / np.sqrt(np.dot(qq3prod, qq3prod))

    Xq = np.array([qq1[0], qq2[0], qq3[0]])
    Yq = np.array([qq1[1], qq2[1], qq3[1]])
    Zq = np.array([qq1[2], qq2[2], qq3[2]])

    firstline = lstsq(matrice_P, Xq)
    secondline = lstsq(matrice_P, Yq)
    thirdline = lstsq(matrice_P, Zq)

    matrixorient = np.array([firstline[0] / np.sqrt(np.dot(firstline[0], firstline[0])),
                        secondline[0] / np.sqrt(np.dot(secondline[0], secondline[0])),
                        thirdline[0] / np.sqrt(np.dot(thirdline[0], thirdline[0]))])  # second line may be negative because of wrong Y sign chi ???


    return matrixorient


def constructMat_new(matrice_P, qq1, qq2, qq3prod):
    """
    Construction of rotation matrix from:
    matrice_P  columns = (G1,G2,G1^G2)     in a*,b*,c* frame
    qq1,qq2,qq3prod  respectively three vectors in  x,y,z space

    """
    qq3 = qq3prod / np.sqrt(np.dot(qq3prod, qq3prod))

    Xq = np.array([qq1[0], qq2[0], qq3[0]])
    Yq = np.array([qq1[1], qq2[1], qq3[1]])
    Zq = np.array([qq1[2], qq2[2], qq3[2]])

    firstline = lstsq(matrice_P, Xq)
    secondline = lstsq(matrice_P, Yq)
    thirdline = lstsq(matrice_P, Zq)

    matrixorient = np.array([firstline[0] / np.sqrt(np.dot(firstline[0], firstline[0])),
                        secondline[0] / np.sqrt(np.dot(secondline[0], secondline[0])),
                        thirdline[0] / np.sqrt(np.dot(thirdline[0], thirdline[0]))])  # second line may be negative because of wrong Y sign chi ???

    # print "matrixorient in constructMat_new()",matrixorient


    # # more general lstsq
    # big_matrice_P = np.zeros((9,9))
    # big_matrice_P[:3,:3] = matrice_P
    # big_matrice_P[3:6,3:6] = matrice_P
    # big_matrice_P[6:,6:] = matrice_P

    # threelines = lstsq(big_matrice_P,np.ravel(np.array([Xq,Yq,Zq])))
    # print "threelines",threelines
    # matrixorient = np.reshape(threelines[0],(3,3))
    return matrixorient


def givematorient(hkl1, coord1, hkl2, coord2, verbose='yes', frame='lauetools'):
    """
    Returns orientation matrix in chosen frame
    from:
    hkl: 3-elements list of hkl Miller indices of spot
    coord: [2theta, chi] coordinates of spot

    PROBLEM: hint: lAUETOOLS use frame = 'lauetools' for recognition from pattern simulated in lauetools frame...
    three possible frames: lauetools , XMASlab, XMASsample

    WARNING! only valid for cubic unit cell!

    """
    h1 = hkl1[0]
    k1 = hkl1[1]
    l1 = hkl1[2]

    h2 = hkl2[0]
    k2 = hkl2[1]
    l2 = hkl2[2]

    G1 = np.array([h1, k1, l1])
    G2 = np.array([h2, k2, l2])
    G3 = np.cross(G1, G2)

    normG1 = np.sqrt(np.dot(G1, G1))
    normG2 = np.sqrt(np.dot(G2, G2))
    normG3 = np.sqrt(np.dot(G3, G3))

    matrice_P = np.array([G1 / normG1, G2 / normG2, G3 / normG3])
    # TODO: OK for almost undistorted cubic structure, to generalize to arbitrary triclinic structure

    # similar matrix but with 2theta,chi coordinate
    twicetheta_1 = coord1[0]
    chi_1 = coord1[1]
    twicetheta_2 = coord2[0]
    chi_2 = coord2[1]

    # expression of unit q in chosen frame
    qq1 = F2TC.unit_q(twicetheta_1 , chi_1, frame=frame)
    qq2 = F2TC.unit_q(twicetheta_2 , chi_2, frame=frame)

    qq3prod = np.cross(qq1, qq2)  # can be negative, we want to have finally a matrix with one eigenvalue of +1

    matou = constructMat(matrice_P, qq1, qq2, qq3prod)

    valeurpropres = np.linalg.eigvals(matou)  # eigen values
    # check number of eigen values that are close to 1.0 (+- 0.05)
    # (we want to have finally a matrix with one eigenvalue of +1, corresponding eigen vector is the rotation axis
    if len(GT.find_closest(np.array([1.0]), np.real(valeurpropres), .05)[1]) == 1:  # we have an axis: OK
        matorient = matou
    else:  # matou is not a rotation matrix
        # print "I need to construct an other matrix"
        matorient = constructMat(matrice_P, qq1, qq2, -qq3prod)

    if verbose == 'yes':
        print("Estimated Orientation Matrix")
        print(matorient)
        print("--------------------------------------------")
        # print "sol",sol
    return matorient


def OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B,
                           verbose=0, frame='lauetools'):
    """
    Upgrade of just above givematorient()
    take into account distorted structure by using Gstar (metric tensor of unit cell)

    Returns orientation matrix in chosen frame

    inputs:
    hkl1, hkl2            : two list of hkl Miller indices
    coord                : [2theta, chi] coordinates of spot

    B            : triangular up matrix B = CP.calc_B_RR([direct lattice parameters])

    PROBLEM: hint: lAUETOOLS use frame = 'lauetools' for recognition from pattern simulated in lauetools frame...
    three possible frames: lauetools , XMASlab, XMASsample

    """
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2

#     print "hkl1", hkl1
#     print "hkl2", hkl2
#     print "B", B

    G1 = np.dot(B, np.array(hkl1))
    G2 = np.dot(B, np.array(hkl2))
    G3 = np.cross(G1, G2)

    normG1 = np.sqrt(np.dot(G1, G1))
    normG2 = np.sqrt(np.dot(G2, G2))
    normG3 = np.sqrt(np.dot(G3, G3))

    # print normG1,normG2,normG3
    matrice_P = np.array([G1 / normG1,
                          G2 / normG2,
                          G3 / normG3])

#     print "Gs", G1, G2, G3
#     print "matrice_P", matrice_P
    # print "Angle G1,G2",np.arccos(np.dot(matrice_P[0],matrice_P[1]))*180./np.pi

    # similar matrix but with 2theta,chi coordinate
    twicetheta_1 = coord1[0]
    chi_1 = coord1[1]
    twicetheta_2 = coord2[0]
    chi_2 = coord2[1]

    # expression of unit q in chosen frame
    qq1 = F2TC.unit_q(twicetheta_1, chi_1, frame=frame)
    qq2 = F2TC.unit_q(twicetheta_2, chi_2, frame=frame)

    qq3prod = np.cross(qq1, qq2)  # can be negative, we want to have finally a matrix with one eigenvalue of +1
    qq3n = np.sqrt(np.dot(qq3prod, qq3prod)) * 1.

    # print "qs",qq1,qq2,qq3prod/qq3n
    # print "Angle qq1,qq2",np.arccos(np.dot(qq1,qq2))*180./np.pi
    # print "det of Gs",np.linalg.det(matrice_P)
    # print "det of unit qs", np.linalg.det(np.array([qq1,qq2,qq3prod/qq3n]))

    # lstsq
    matou = constructMat_new(matrice_P, qq1, qq2, qq3prod / qq3n)

    # print "matou",matou
    # print "determinant",np.linalg.det(matou)

    # U,s,V= np.linalg.svd(matou)
    # print "singular values of matou",s

    valeurpropres = np.linalg.eigvals(matou)  # eigen values
    # print "valeurpropres",valeurpropres

    # check number of eigen values that are close to 1.0 (+- 0.05)
    # (we want to have finally a matrix with one eigenvalue of +1, corresponding eigen vector is the rotation axis
    if len(GT.find_closest(np.array([1.0]), np.real(valeurpropres), .05)[1]) in (1, 3):  # we have an axis: OK
        matorient = matou
    else:  # matou is not a rotation matrix
        # print "I need to construct an other matrix"
        matorient = constructMat(matrice_P, qq1, qq2, -qq3prod)

    if verbose in ('yes', 1):
        print("Estimated Orientation Matrix ---------------")
        print(matorient)
        print("--------------------------------------------")
        # print "sol",sol
    return matorient


def Allproposedmatrix(listrecogn, tolang):
    """
    Screens all pair or elems in listrecogn, find pairs that match a solution in lookup table
    """
    print("------- All orientation matrices from picked file ----------")
    listofmatrix = []
    for k in range(len(listrecogn)):
        for j in range(k, len(listrecogn)):
            mama = proposematrix(k, j, listrecogn, tolang, verbose='no')
            if mama is not None:
                listofmatrix.append([[k, j], mama])
    return listofmatrix


def computeUnique_UBmatrix(UBmat):
    print("matrix with lowest Euler angles", end=' ')
    matLTmin, transfmat = find_lowest_Euler_Angles_matrix(UBmat)
    return matLTmin, transfmat


def find_lowest_Euler_Angles_matrix(mat):
    """
        # corrected 08 Nov 11 by O. Robach
        # mat : gives columns of a* b* c* on x y z
    """
    if LA.det(mat) < 0.0 :
        raise ValueError("warning : det < 0 in input of find_lowest_Euler_Angles_matrix")

    sign3 = np.ones(3)
    ind6 = np.zeros(6)
    ind4 = np.zeros(4)
    sign3m = -1.0 * sign3
    matm = mat * -1.0
    mat6 = np.hstack((mat, matm))
    sign6 = np.hstack((sign3, sign3m))

    # print "find lowest euler angles matrix"
    # print mat6
    # print sign6

    # find which vector a b c -a -b -c has largest z component  => new vector c'
    ic = np.argmax(mat6[2, :])
    # print ic
    # print mat6[:,ic]
    icm = np.mod(ic + 3, 6)
    # print icm
    # ind6 = 1 pour c' et -c', 0 pour a', b', -a', -b'
    ind6[ic] = 1
    ind6[icm] = 1
    ind2 = (np.where(ind6 == 0))[0]
    # print ind2
    # mat4 : a', b', -a', -b'
    mat4 = mat6[:, ind2]
    # print mat4
    sign4 = sign6[ind2]
    # print sign4
    # find which vector a' b' -a' -b' has largest x component => new vector a''
    ia = np.argmax(mat4[0, :])
    # print ia
    # print mat4[:,ia]
    iam = np.mod(ia + 2, 4)
    # print iam
    # ind4 = 1 pour a'' et -a''
    ind4[ia] = 1
    ind4[iam] = 1
    ind2 = (np.where(ind4 == 0))[0]
    # print ind2
    # mat2 : b'', -b''
    mat2 = mat4[:, ind2]
    # print mat2
    sign2 = sign4[ind2]
    # find which vector b" -b" gives direct triedre => new vector b'''
    csec = mat6[:, ic]
    asec = mat4[:, ia]
    b1 = np.cross(csec, asec)
    if np.inner(b1, mat2[:, 0]) > 0.0 :
        bsec = mat2[:, 0]
    else :
        bsec = mat2[:, 1]

    matfinal = np.column_stack((asec, bsec, csec))
# #    print sign(inner(cross(mat[:,0], mat[:,1]),mat[:,2]))
# #    sign0 = sign(inner(cross(matfinal[:,0], matfinal[:,1]),matfinal[:,2]))
# #        print sign0
    print("transform matrix to matrix with lowest Euler Angles")
    print("start \n", mat)
    print("final \n", matfinal)

    # matrix to transform hkl's
    transfmat = LA.inv((np.dot(LA.inv(mat), matfinal).round(decimals=1)))
# #    print "transfmat = \n ", transfmat
# #        hkl = array([5.0,-7.0,3.0])
# #        print shape(hkl)
# #        # hkl needs to be as column vector
# #        qxyz = dot(mat,hkl)
# #        print "qxyz =", qxyz.round(decimals = 4)
# #        hkl2 = dot(transfmat,hkl)
# #        qxyz2 = dot(matfinal,dot(transfmat,hkl))
# #        print "qxyz2 =", qxyz2.round(decimals = 4)
# #        print "hkl = ", hkl
# #        print "hkl2 = ", hkl2

    if LA.det(matfinal) < 0.0 :
        # print "warning : det < 0 in output of find_lowest_Euler_Angles_matrix"
        raise ValueError("warning : det < 0 in output of find_lowest_Euler_Angles_matrix")

    return matfinal, transfmat


def GenerateLookUpTable(hkl_all, Gstar):
    """
    Generate Look Up Table of angles between hkl directions
    for an unit cell defined by (reciprocal) metric tensor Gstar

    inputs:

    hkl_all            :    array of [h,k,l]
    Gstar            :    3*3 array

    outputs:

    sorted_angles        : angles between all pairs of hkl in hkl_all sorted in increasing order
    sorted_ind            : array of indices of original indy array when sorting the array of angles
    indy                : array of indices where angle between hkls are taken in the flattened pairs angles matrix (originally square)
    tab_side_size        : size of the square pairs angles matrix
    """

    # compute square matrix containing angles
    tab_angulardist = CP.AngleBetweenNormals(hkl_all, hkl_all, Gstar)

    # # place value higher than any possible angle value in diagonal elements
    # np.putmask(tab_angulardist,tab_angulardist<0.001,400)
    # np.putmask(tab_angulardist,np.isnan(tab_angulardist),400)

    # proxtable = np.argmin(tab_angulardist,axis=1)

    # from square interangles matrix (from the same set of hkl)
    tab_side_size = tab_angulardist.shape[0]
    indy = GT.indices_in_flatTriuMatrix(tab_side_size)
    angles_set = np.take(tab_angulardist, indy)  # 1D array (flatten automatically tab_angulardist)

    # sort indices (from 1D array) from angle values
    sorted_ind = np.argsort(angles_set)
    sorted_angles = angles_set[sorted_ind]

    return sorted_ind, sorted_angles, indy, tab_side_size, hkl_all


def Generate_selectedLUT(hkl1, hkl2, key_material, verbose=0):
    """
    Generate Look Up Table of angles between hkl1 and hkl2 directions
    for an unit cell defined in dict_LaueTools.py in material dictionnary

    see doc of GenerateLookUpTable_from2sets()
    """
    latticeparams = dict_Materials[key_material][1]
    Gstar = CP.Gstar_from_directlatticeparams(*latticeparams)

    return GenerateLookUpTable_from2sets(hkl1, hkl2, Gstar, verbose=verbose)

HKL_CUBIC_UP3 = [[1, 0, 0], [1, 1, 0], [1, 1, 1],
           [2, 1, 0], [2, 1, 1], [2, 2, 1],
           [3, 1, 0], [3, 1, 1], [3, 2, 1], [3, 2, 2], [3, 3, 1], [3, 3, 2]]


def Generate_LUT_for_Cubic(hkl2, Gstar, verbose=0):
    """
    Generate Look Up Table of angles between HKL_CUBIC_UP3 (up to order 3) and hkl2 directions
    for an unit cell defined by (reciprocal) metric tensor Gstar

    see doc of GenerateLookUpTable_from2sets()
    """
    hkl1 = np.array(HKL_CUBIC_UP3)

    return GenerateLookUpTable_from2sets(hkl1, hkl2, Gstar, verbose=verbose)


def GenerateLookUpTable_from2sets(hkl1, hkl2, Gstar, verbose=0):
    """
    Generate Look Up Table of angles between hkl1 and hkl2 directions
    for an unit cell defined by (reciprocal) metric tensor Gstar

    inputs:
    hkl1            :    array of [h,k,l]
    hkl2            :    array of [h,k,l]
    Gstar            :    3*3 array

    outputs:
    sorted_angles        : angles between all pairs of hkl in hkl_all sorted in increasing order
    sorted_ind            : array of indices of original indy array when sorting the array of angles
    indy                : array of indices where angle between hkls are taken in the flattened pairs angles matrix (originally square)
    tab_side_size        : size of the square pairs angles matrix
    """
    # compute square matrix containing angles
    tab_angulardist = CP.AngleBetweenNormals(hkl1, hkl2 , Gstar)
    # shape of tab_angulardist  (len(hkl1), len(hkl2))
#     print "tab_angulardist.shape", tab_angulardist.shape

    # to exclude parallel hkls for further purpose (putting very high angle value)
    np.putmask(tab_angulardist, np.abs(tab_angulardist) < 0.001, 400)

    angles_set = np.ravel(tab_angulardist)  # 1D array

    if verbose:
        print("tab_angulardist.shape", tab_angulardist.shape)
        print("tab_angulardist", tab_angulardist)
        print("angles_set", angles_set)

    # sort indices (from 1D array) from angle values
    sorted_ind = np.argsort(angles_set)
    sorted_angles = angles_set[sorted_ind]

    sorted_ind_ij = GT.convert2indices(sorted_ind, tab_angulardist.shape)
#     print "finished GenerateLookUpTable_from2sets"
    return sorted_ind, sorted_angles, sorted_ind_ij, tab_angulardist.shape


def QueryLUT(LUT, query_angle, tolerance_angle, verbose=0):
    """
    Query the LUT and return the atomic planes pairs solutions
    ( that form an angle close to query_angle within tolerance_angle)

    LUT has been  built from a square interdistance angles from one hkl list with itself
    """
    # in sorted angles
    indices = np.where((np.abs(LUT[1] - query_angle)) < tolerance_angle)[0]

    angles_close = np.take(LUT[1], indices)
    # in absolute indice in 1d triangular up wo diagonal frame
    index_in_triu = np.take(LUT[0], indices)
    index_in_triu_orig = np.take(LUT[2], index_in_triu)

    IJ_indices = GT.indices_in_TriuMatrix(index_in_triu_orig, LUT[3])

    planes_pairs = np.take(LUT[4], IJ_indices, axis=0)

    if verbose:
        print("indices", indices)
        print("index_in_triu", index_in_triu)
        print("index_in_triu_orig", index_in_triu_orig)
        print("IJ_indices", IJ_indices)

    return planes_pairs, angles_close


def buildLUT_fromMaterial(key_material, n, CheckAndUseCubicSymmetry=True):
    """
    build reference angles LUT from all mutual angular distances
    between hkls of two different sets

    n    : highest hkls order

    key_material        : element, material or structure label (string)

    CheckAndUseCubicSymmetry  : False  to not restrict the LUT
                                True   to restrict LUT (allowed only for cubic crystal)
    """
    from . import dict_LaueTools as DictLT
    latticeparams = DictLT.dict_Materials[key_material][1]

    return buildLUT_fromLatticeParams(latticeparams, n,
                                      CheckAndUseCubicSymmetry=CheckAndUseCubicSymmetry)


def buildLUT_fromMaterial_nm(key_material, n, m, CheckAndUseCubicSymmetry=True):
    """
    build reference angles LUT from all mutual angular distances
    between hkls of two different sets

    n    : highest hkls order of set 1
    m    : highest hkls order of set 2

    key_material        : element, material or structure label (string)

    CheckAndUseCubicSymmetry  : False  to not restrict the LUT
                                True   to restrict LUT (allowed only for cubic crystal)
    """
    from . import dict_LaueTools as DictLT
    latticeparams = DictLT.dict_Materials[key_material][1]

    restrictLUT = False
    if CheckAndUseCubicSymmetry:
        # LUT restriction given by crystal structure
        restrictLUT = CP.isCubic(latticeparams)

    hkls_1 = GT.threeindices_up_to(n, remove_negative_l=restrictLUT)
    hkls_2 = GT.threeindices_up_to(m, remove_negative_l=restrictLUT)

    if 1:  # filterharmonics:
#        hkl_all = CP.FilterHarmonics_2(hkl_all)
        hkls_1 = FilterHarmonics(hkls_1)
        hkls_2 = FilterHarmonics(hkls_2)

    Gstar_metric = CP.Gstar_from_directlatticeparams(*latticeparams)
    # GenerateLookUpTable
    LUT = GenerateLookUpTable_from2sets(hkls_1, hkls_2, Gstar_metric, verbose=0)

    return LUT


def buildLUT_fromLatticeParams(latticeparams, n, CheckAndUseCubicSymmetry=True):
    """
    build reference angles LUT from all mutual angular distances
    between hkls of two different sets

    n    : highest hkls order

    latticeparams        : 6 [direct space] lattice parameters of 
                        element, material or structure label
                        [a,b,c,alpha, beta,gamma] (angles in degrees)

    CheckAndUseCubicSymmetry  : False  to not restrict the LUT
                                True   to restrict LUT (allowed only for cubic crystal)
    """
    restrictLUT = False
    if CheckAndUseCubicSymmetry:
        # LUT restriction given by crystal structure
        restrictLUT = CP.isCubic(latticeparams)

    hkl_all = GT.threeindices_up_to(n, remove_negative_l=restrictLUT)

    if 1:  # filterharmonics:
#        hkl_all = CP.FilterHarmonics_2(hkl_all)
        hkl_all = FilterHarmonics(hkl_all)

    Gstar_metric = CP.Gstar_from_directlatticeparams(*latticeparams)
    # GenerateLookUpTable
    LUT = GenerateLookUpTable(hkl_all, Gstar_metric)

    return LUT


def RecogniseAngle(angle, tol, nLUT, latticeparams_or_material):
    """
    Return hlk couples and corresponding angle that match the input angle within the tolerance angle

    nLUT   :  order of the LUT

    latticeparams_or_material  : either string key for material or list of 6 lattice parameters
    """
    from . import dict_LaueTools as DictLT
    if isinstance(latticeparams_or_material, str):
        latticeparams = DictLT.dict_Materials[latticeparams_or_material][1]
    else:
        latticeparams = latticeparams_or_material

    LUT = buildLUT_fromLatticeParams(latticeparams, nLUT)

    sol = QueryLUT(LUT, angle, tol)

    print("solutions", sol)

    return sol


def PlanePairs_2(query_angle, angle_tol, LUT, onlyclosest=1, verbose=0):
    """
    return pairs of lattice hkl planes
    whose mutual angles between normals are the closest to the given query_angle within tolerance

    USED in manual indexation
    
    LUT is an input argument !! then main part is the same as in PlanePairs

    input:
    query_angle        : angle in deg to look up in the generated angle table
    angle_tol            : angular tolerance when look up in the generated reference angle table

    Gstar                : metric tensor of the unit cell structure
    n                    : maximum index for generating reference angle table from Gstar

    onlyclosest            : 1 for considering only one angle value closest to query_angle
                            (only planes pairs corresponding to one matched angle are returned)
                        : 0 for considering all angle close to query_angle within angle_tol

    TODO: many target angles
    """
    sorted_ind, sorted_angles, indy, tab_side_size, hkl_all = LUT

    angular_tolerance_Recognition = angle_tol
    angle_query = query_angle

    # if angle_query is a tuple,array,list
    if type(query_angle) not in (type(5), type(5.5), type(np.arange(0., 2., 3.)[0])):
        angle_query = query_angle[0]

    # Find matching

    # only planes pairs corresponding to one matched angle are returned
    # taking the first value of angle_target
    if onlyclosest:
        closest_index_in_sorted_angles_raw = GT.find_closest(sorted_angles,
                                                             np.array([angle_query]),
                                                             angular_tolerance_Recognition)[0]

        closest_angle = sorted_angles[closest_index_in_sorted_angles_raw][0]
        # print "Closest_angle",closest_angle

        if abs(closest_angle - angle_query) <= angle_tol:

            # in case of many similar angles...
            close_angles_duplicates = np.where(sorted_angles == closest_angle)[0]

            if len(close_angles_duplicates) > 1:

                # print "\nThere are several angles in structure from different lattice plane pairs that are equal\n"
                closest_index_in_sorted_angles_raw = close_angles_duplicates

            # in sorted triu
            index_in_triu = np.take(sorted_ind, closest_index_in_sorted_angles_raw)
            # in original trio
            index_in_triu_orig = np.take(indy, index_in_triu)
            IJ_indices = GT.indices_in_TriuMatrix(index_in_triu_orig, tab_side_size)

            planes_pairs = np.take(hkl_all, IJ_indices, axis=0)

            if verbose:
                print("\n within %.3f and close to %.9f deg" % \
                        (angular_tolerance_Recognition, angle_query))
                for pair in planes_pairs:
                    print("< ", pair[0], "  ,  ", pair[1], " > = %.6f " % closest_angle)

            return planes_pairs

        else:
            if angle_query > 0.5:
                print("\nthere is no angle close to %.2f within %.2f deg" % (angle_query, angular_tolerance_Recognition))
                print("Nearest angle found is %.2f deg\n" % closest_angle)
            return None

    # planes pairs corresponding to all matched angles in angle range
    # (defined by query_angle and angle_tol) are returned
    else:  # onlyclosest = 0

        condition = np.abs(sorted_angles - angle_query) < angular_tolerance_Recognition
        closest_indices_in_sorted_angles_raw = np.where(condition)

        if len(closest_indices_in_sorted_angles_raw[0]) > 1:

            # in sorted triu
            index_in_triu = np.take(sorted_ind, closest_indices_in_sorted_angles_raw[0])
            # in original trio
            index_in_triu_orig = np.take(indy, index_in_triu)
            IJ_indices = GT.indices_in_TriuMatrix(index_in_triu_orig, tab_side_size)

            planes_pairs = np.take(hkl_all, IJ_indices, axis=0)

            if verbose:
                print("\n within %.3f and close to %.9f deg" % \
                        (angular_tolerance_Recognition, angle_query))
                for k, pair in enumerate(planes_pairs):
                    print("< ", pair[0], "  ,  ", pair[1], " > = %.6f " % sorted_angles[closest_indices_in_sorted_angles_raw[0][k]])
            return planes_pairs

        else:
            if angle_query > 0.5:
#                 print "\nthere is no angle close to %.2f within %.2f deg" % (angle_query, angular_tolerance_Recognition)
#                 print "Nearest angle found is %s deg\n" % str(sorted_ind)
                pass
            return None


def PlanePairs(query_angle, angle_tol, Gstar, n,
               onlyclosest=1, filterharmonics=1, verbose=0):
    """
    return pairs of lattice hkl planes
    whose angle between normals are the closest to the given query_angle within tolerance

    LUT is computed during the function

    input:
    query_angle        : angle in deg to look up in the generated angle table
    angle_tol            : angular tolerance when look up in the generated reference angle table

    Gstar                : metric tensor of the unit cell structure
    n                    : maximum index for generating reference angle table from Gstar

    onlyclosest            : 1 for considering only one angle value closest to query_angle
                            (only planes pairs corresponding to one matched angle are returned)
                        : 0 for considering all angle close to query_angle within angle_tol

    TODO: many target angles
    """
    hkl_all = GT.threeindices_up_to(n)

    if filterharmonics:
#         hkl_all = FilterHarmonics(hkl_all)
        hkl_all = CP.FilterHarmonics_2(hkl_all)

    # GenerateLookUpTable
    LUT = GenerateLookUpTable(hkl_all, Gstar)

    sorted_ind, sorted_angles, indy, tab_side_size, hkl_all = LUT

    angular_tolerance_Recognition = angle_tol
    angle_query = query_angle

    # if angle_query is a tuple,array,list
    if type(query_angle) not in (type(5), type(5.5), type(np.arange(0., 2., 3.)[0])):
        angle_query = query_angle[0]

    # Find matching

    # only planes pairs corresponding to one matched angle are returned
    # taking the first value of angle_target
    if onlyclosest:

        closest_index_in_sorted_angles_raw = GT.find_closest(sorted_angles, np.array([angle_query]), angular_tolerance_Recognition)[0]
        closest_angle = sorted_angles[closest_index_in_sorted_angles_raw][0]
        # print "Closest_angle",closest_angle

        if abs(closest_angle - angle_query) <= angle_tol:

            # in case of many similar angles...
            close_angles_duplicates = np.where(sorted_angles == closest_angle)[0]

            if len(close_angles_duplicates) > 1:

                # print "\nThere are several angles in structure from different lattice plane pairs that are equal\n"
                closest_index_in_sorted_angles_raw = close_angles_duplicates

            # in sorted triu
            index_in_triu = np.take(sorted_ind, closest_index_in_sorted_angles_raw)
            # in original trio
            index_in_triu_orig = np.take(indy, index_in_triu)
            IJ_indices = GT.indices_in_TriuMatrix(index_in_triu_orig, tab_side_size)


            planes_pairs = np.take(hkl_all, IJ_indices, axis=0)

            if verbose:
                print("\n within %.3f and close to %.9f deg" % (angular_tolerance_Recognition, angle_query))
                for pair in planes_pairs:
                    print("< ", pair[0], "  ,  ", pair[1], " > = %.6f " % closest_angle)

            return planes_pairs

        else:
            if angle_query > 0.5:
#                 print "\nthere is no angle close to %.2f within %.2f deg" % (angle_query, angular_tolerance_Recognition)
#                 print "Nearest angle found is %.2f deg\n" % closest_angle
                pass
            return None

    # planes pairs corresponding to all matched angles in angle range
    # (defined by query_angle and angle_tol) are returned
    else:  # onlyclosest = 0
        closest_indices_in_sorted_angles_raw = np.where(np.abs(sorted_angles - angle_query) < angular_tolerance_Recognition / 2.)

        if len(closest_indices_in_sorted_angles_raw[0]) > 1:

            # in sorted triu
            index_in_triu = np.take(sorted_ind, closest_indices_in_sorted_angles_raw[0])
            # in original trio
            index_in_triu_orig = np.take(indy, index_in_triu)
            IJ_indices = GT.indices_in_TriuMatrix(index_in_triu_orig, tab_side_size)


            planes_pairs = np.take(hkl_all, IJ_indices, axis=0)

            if verbose:
                print("\n within %.3f and close to %.9f deg" % (angular_tolerance_Recognition, angle_query))
                for k, pair in enumerate(planes_pairs):
                    print("< ", pair[0], "  ,  ", pair[1], " > = %.6f " % sorted_angles[closest_indices_in_sorted_angles_raw[0][k]])
            return planes_pairs

        else:
            if angle_query > 0.5:
#                 print "\nthere is no angle close to %.2f within %.2f deg" % (angle_query, angular_tolerance_Recognition)
#                 print "Nearest angle found is %.2f deg\n" % sorted_ind
                pass
            return None


def PlanePairs_from2sets(query_angle, angle_tol,
                            hkl1, hkl2, key_material,
                            onlyclosest=1, filterharmonics=1,
                            LUT=None,
                            verbose=0):
    """
    Generate a particular LUT from hkl1 and hkl2
    and return pairs of lattice hkl planes
    whose angle between normals are the closest to the given query_angle within tolerance

    input:

    query_angle            : angle in deg to look up in the generated angle table
    angle_tol            : angular tolerance when look up in the generated reference angle table

    key_material                : string label for material

    onlyclosest            : 1 for considering only one angle value closest to query_angle
                            (only planes pairs corresponding to one matched angle are returned)
                        : 0 for considering all angle close to query_angle within angle_tol

    TODO: many target angles in this function
    """
    if isinstance(hkl1, list):
        hkl1 = np.array(hkl1)
    if isinstance(hkl2, list):
        hkl2 = np.array(hkl2)
        
#     print "hkl1", hkl1
#     print "hkl2", hkl2
    
    if filterharmonics:
#         hkl1 = FilterHarmonics(hkl1)
#         hkl2 = FilterHarmonics(hkl2)

        if hkl1.shape != (3,):
            hkl1 = CP.FilterHarmonics_2(hkl1)
        if hkl2.shape != (3,):
            hkl2 = CP.FilterHarmonics_2(hkl2)
            
    if hkl1.shape == (3,):
        hkl1 = np.array([hkl1])
    if hkl2.shape == (3,):
        hkl2 = np.array([hkl2])

    if LUT is None:
        # GenerateLookUpTable
        print("Calculating LUT in PlanePairs_from2sets()")
        LUT = Generate_selectedLUT(hkl1, hkl2, key_material)

    (sorted_ind,
     sorted_angles,
     sorted_ind_ij,
     tab_angulardist_shape) = LUT

#     print "sorted_ind", sorted_ind
#     print "sorted_ind_ij", sorted_ind_ij

    angular_tolerance_Recognition = angle_tol
    angle_query = query_angle

#     if type(query_angle) not in (type(5), type(5.5), type(np.arange(0., 2., 3.)[0])):  # if angle_query is a tuple,array,list
    if isinstance(query_angle, (list, np.ndarray, tuple)):
        angle_query = query_angle[0]

    array_angledist = np.abs(sorted_angles - angle_query)

    pos_min = np.argmin(array_angledist)

    closest_angle = sorted_angles[pos_min]
    
#     print "closest_angle ok ===>", closest_angle

    if np.abs(closest_angle - query_angle) > angular_tolerance_Recognition:
        if angle_query > 0.5:
#             print "\nThere is no angle close to %.2f within %.2f deg" % \
#                             (angle_query, angular_tolerance_Recognition)
#             print "Nearest angle found is %.2f deg\n" % closest_angle
            pass
        return None, LUT

    condition = array_angledist <= angular_tolerance_Recognition
    closest_indices_in_sorted_angles_raw = np.where(condition)

    closest_index_in_sorted_angles_raw = closest_indices_in_sorted_angles_raw[0]

    if onlyclosest:
        # in case of many similar angles...
        close_angles_duplicates = np.where(sorted_angles == closest_angle)[0]

        if len(close_angles_duplicates) > 1:
            if verbose:
                print("\nThere are several angles in structure from different lattice plane pairs that are equal\n")
            closest_index_in_sorted_angles_raw = close_angles_duplicates
        else:
            pass

    closest_angles_values = np.take(sorted_angles, closest_index_in_sorted_angles_raw)
    AngDev = np.take(array_angledist, closest_index_in_sorted_angles_raw)

    IJ_indices = np.take(sorted_ind_ij, closest_index_in_sorted_angles_raw, axis=0)

    plane_1 = np.take(hkl1, IJ_indices[:, 0], axis=0)
    plane_2 = np.take(hkl2, IJ_indices[:, 1], axis=0)
    
#     print "plane_1", plane_1
#     print "plane_2", plane_2

    if len(plane_1) > 1:
#         print 'nb sol >1'
        planes_pairs = np.hstack((plane_1, plane_2)).reshape((len(plane_1), 2, 3))
    else:
        planes_pairs = np.array([plane_1[0], plane_2[0]])

    if verbose:
        print("tab_angulardist_shape", tab_angulardist_shape)
        print("query_angle", query_angle)
        print("sorted_angles", sorted_angles)
        print("Closest_angle", closest_angle)
        print("closest_index_in_sorted_angles", closest_index_in_sorted_angles_raw)
        print("closest_angles_values", closest_angles_values)
        print("Angles Deviation from query angle", AngDev)
        print("IJ_indices", IJ_indices)
        print("plane_1", plane_1)
        print("plane_2", plane_2)
        print('len(plane_1)', len(plane_1))
        print('len(plane_2)', len(plane_2))
        print("planes_pairs", planes_pairs)
        print("\n Lattice Planes pairs found within %.3f and close to %.9f deg" % \
                    (angular_tolerance_Recognition, angle_query))
        if len(plane_1) > 1:
            for k, pair in enumerate(planes_pairs):
                print("< ", pair[0], " , ", pair[1], " > = %.2f, AngDev %.2f" % \
                (closest_angles_values[k], AngDev[k]))
        else:
            print("< ", planes_pairs[0], "  ,  ", planes_pairs[1], " > = %.6f " % closest_angles_values)
    return planes_pairs, LUT


def FilterHarmonics(hkl):
    """
    keep only hkl 3d vectors that are representative of direction nh,nk,nl
    for any h,k,l signed integers

    TODO: not working ??? see corresponding test: test_FilterHarmonics
    See FilterHarmonics_2 in CrystalParameters

    NOTE: this function used to build angles LUT seems correct
    """
#     print "np.array(hkl) in FilterHarmonics", np.array(hkl)
    if np.array(hkl).shape[0] == 1:
#         print "input array has only one element! So Nothing to filter..."
        return hkl
    elif np.array(hkl).shape == (3,):
        return np.array([hkl])
    # square array with element[i,j] = cross(HKLs[i],HKLs[j])
    a = GT.find_parallel_hkl(hkl)
    SA = a.shape
    fa = np.reshape(a, (SA[0] * SA[1], 3))
    ind_in_flat_a = GT.indices_in_flatTriuMatrix(SA[0])
    # 1D array 3d vectors (inter cross products)
    crosspair = np.take(fa, ind_in_flat_a, axis=0)

    # print "crosspair",crosspair

    # index of zero 3D vectors
    pos_zeros = np.where(np.all(crosspair == 0, axis=1) == True)[0]

    # print "pos_zeros",pos_zeros

    if len(pos_zeros) > 0:
        hkls_pairs_index = GT.indices_in_TriuMatrix(ind_in_flat_a[pos_zeros], SA[0])
        hkls_pairs = np.take(hkl, hkls_pairs_index, axis=0)

        # print "hkls_pairs_index",hkls_pairs_index
        # print "hkls_pairs",hkls_pairs
        # create dict of equivalent hkls (same direction), same irreductible representation
        dictsets, intermediary_dict = GT.Set_dict_frompairs(hkls_pairs_index, verbose=0)

        # print "dictsets, intermediary_dict",dictsets, intermediary_dict

        toremove = []
        for key, val in list(dictsets.items()):
            toremove += sorted(val)[1:]
        toremove.sort()

        return np.delete(hkl, toremove, axis=0)
    else:
        return hkl


def HKL2string(hkl):
    """
    convert hkl into string
    [-10.0,-0.0,5.0]  -> '-10,0,5'
    """
    res = ''
    for elem in hkl:
        ind = int(elem)
        strind = str(ind)
        if strind == '-0':  # removing sign before 0
            strind = '0'
        res += strind + ','

    return res[:-1]


def HKLs2strings(hkls):
    """
    convert a list a hkl into string using HKL2string()

    """
    res = []
    if np.shape(np.array(hkls)) == (3,):
        return HKL2string(hkls)
    else:
        for hkl in hkls:
            res.append(HKL2string(hkl))
        return res


def FilterEquivalentPairsHKL(pairs_hkl):
    """
    remove pairs of hkls that are equivalent by inversion
    [h,k,l]1 [h,k,l]2  and [-h,-k,-l]1 [-h,-k,-l]2  are equivalent
    """

    if pairs_hkl is None:
        print("pairs_hkl is empty !!")
        return None
    if type(pairs_hkl) != type(np.ones(2)):
        print("Wrong input! Array with shape (#,2,3) is needed!")
        return None

    if len(pairs_hkl) > 1:

        dict_diff = {}

        # mutual pair addition
        # shape = ( len(pairs_hkl), len(pairs_hkl) )
        sumpp = pairs_hkl + pairs_hkl[:, np.newaxis]

        # logical operations
        true_in_vectors = np.all(sumpp == 0, axis=3)
        true_in_pairs_of_vectors = np.all(true_in_vectors, axis=2)
        truepos = np.where(true_in_pairs_of_vectors == True)
        # i,j position of
        truepos_ij = np.array([truepos[0], truepos[1]]).T

        dictsets_pairs, intermediary_dict_pairs = GT.Set_dict_frompairs(truepos_ij, verbose=0)

        toremove_pairs = []
        for key, val in list(dictsets_pairs.items()):
            toremove_pairs += sorted(val)[1:]
        toremove_pairs.sort()

        purged_pp = np.delete(pp, toremove_pairs, axis=0)

        return purged_pp
    else:
        print("Purge is not needed, since there is only one pair of hkls ..!")

        return purged_pp


#--- ---TEST ----------------
def Test_Build_and_Query_LUT(angle=69, tol=1, n=3, latticeparams=(2, 1, 4, 75, 90, 120)):
    """
    Test_Build_and_Query_LUT
    """
    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n, remove_negative_l=CP.isCubic(latticeparams))

    if 1:  # filterharmonics:
#        hkl_all = CP.FilterHarmonics_2(hkl_all)
        hkl_all = FilterHarmonics(hkl_all)

    Gstar_metric = CP.Gstar_from_directlatticeparams(*latticeparams)
    # GenerateLookUpTable
    LUT = GenerateLookUpTable(hkl_all, Gstar_metric)

    sol69 = QueryLUT(LUT, angle, tol)

    print("solutions", sol69)

    return sol69


def test_FilterHarmonics():
    """
    test harmonics removal
    """
    a = np.arange(3 * 8).reshape((8, 3))
    a[-1] = np. array([-6, -8, -10])
    a[-3] = np. array([9, 12, 15])
    fila = FilterHarmonics(a)
    print("a", a)
    print("filtered a")
    print(fila)

    print("\n other test")
    testhkl = np.array([[ -2, -4, 2], [  0, -2, 2], [ -3, -11, 3], [ -1, -11, 3], [ -5, -9, 3],
       [ -3, -9, 3], [ -1, -9, 3], [  1, -9, 3], [ -5, -7, 3], [ -3, -7, 3],
       [ -1, -7, 3], [  1, -7, 3], [ -5, -5, 3], [ -3, -5, 3], [ -1, -5, 3],
       [  1, -5, 3], [ -3, -3, 3], [ -1, -3, 3], [  1, -3, 3], [ -2, -10, 4],
       [ -4, -8, 4], [  0, -8, 4], [ -6, -6, 4], [ -2, -6, 4], [  2, -6, 4],
       [  0, -4, 4], [ -2, -2, 4], [ -5, -7, 5]])
    print("testhkl")
    print(testhkl)
    print("one of  [  0,  -2,   2] and [  0,  -4,   4] must be removed")

    print(FilterHarmonics(testhkl))


def test():
    """
    test
    """
    # GT.find_closest(input_array, target_array, tol)

    # datafrompickle=open('recognised.rec','r')
    # reconnu = pickle.load(datafrompickle)
    # datafrompickle.close()

    recognised = [[72.263978989400002, 19.5486315279, 111, 12],
    [109.63069847600001, -24.477560408599999, 110, 16],
    [95.941622918999997, 10.5435782548, 210, 21],
    [82.3329445217, 4.2357560583199998, 311, 31],
    [85.062674650700004, -3.4110763979200001, 210, 32],
    [97.118855973999999, -0.87674472922199997, 211, 64],
    [106.209401235, -15.1071780961, 311, 93],
    [49.406572447000002, -12.219836147100001, 311, 133],
    [93.0858676231, 3.5826456916599998, 221, 147]]

    recog = recognised  # reconnu

    print("-----------------------------------------")
    print("First two recognised spots Matrix proposition")
    indexpoint1 = 0  # default = 0
    indexpoint2 = 1  # default = 1
    ang = 0.1
    print("For spots #%d and #%d in data" % (recog[indexpoint1][-1], recog[indexpoint2][-1]))
    print("Angular tolerance: %.2f deg" % ang)
    proposematrix(indexpoint1, indexpoint2, recog, ang)
    print("------------------------\n")

    print("Now looking at all spots pairs distances and compare with lookup table")
    toleranceang = 0.05
    print("-----------------------------------------")
    print("Angular tolerance for multiple search: %.2f" % toleranceang)
    result = Allproposedmatrix(recog, toleranceang)
    # print result

    if len(result) > 0:
        print("Cool, I can propose %s matrix" % len(result))
        for elem in result:
            spotindices = elem[0]
            print("With spots #%d and #%d in data" % (recog[spotindices[0]][-1], recog[spotindices[1]][-1]))
            print(elem[1])


        print("Now removing similar matrix if needed")
        singlematrixlist = [result[0][1]]
    else:
        print("Argh no matrix found!")

    for j in range(1, len(result)):
        listcorrel = [np.corrcoef(np.ravel(result[j][1]), np.ravel(singlematrixlist[k]))[0][1] for k in range(len(singlematrixlist))]

        booltable = (np.array(listcorrel) < 0.995)  # True if matrices are uncorrelated

        if all(booltable):  # if matrix proposed is different from the others

            singlematrixlist.append(result[j][1])

    print("Similar matrices removed")
    print(singlematrixlist)

    print("TODO: looking at same matrix by frame permutation")


def example_Neighbourslist():
    """ 
    give list of planes that are at 25 deg +- tol from a the 210 plane
    """
    print("Planes that are distant from [2,1,0] plane from 20 to 30 degree")
    tol = 5.
    table_210 = prodtab_210()
    pos_close = GT.find_closest(array([25.]), table_210, tol)
    plane_sol = np.take(LISTNEIGHBORS_210, pos_close[1], axis=0)
    distance_sol = np.take(table_210, pos_close[1])
    for plane, dist in zip(plane_sol, distance_sol):
        print("plane %s at %.2f degree from plane [2,1,0]" % (plane, dist))


def test_PlanePairs_from2sets():
    latticeparams = dict_Materials['Cu'][1]
    n = 3
    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n, remove_negative_l=CP.isCubic(latticeparams))

    if 1:  # filterharmonics:
#        hkl_all = CP.FilterHarmonics_2(hkl_all)
        hkl_all = FilterHarmonics(hkl_all)

    Gstar = CP.Gstar_from_directlatticeparams(*latticeparams)

    hkl1 = [[1, 0, 0], [2, 2, 1]]
    hkl2 = hkl_all

    PlanePairs_from2sets(25.35, 5,
                            hkl1, hkl2, Gstar,
                            onlyclosest=1, filterharmonics=1, verbose=1)

    PlanePairs_from2sets(25.35, 5,
                            hkl1, hkl2, Gstar,
                            onlyclosest=0, filterharmonics=1, verbose=1)


#--- ------------------  MAIN
if __name__ == "__main__":
    pass

#     Test_Build_and_Query_LUT(angle=69, tol=1, n=4, latticeparams=(2, 1, 4, 75, 90, 120))


    # test() # old test with data from recognition.py



#     PlanePairs_from2sets(25.1, .5,
#                             hkl1, hkl2, Gstar,
#                             onlyclosest=1, filterharmonics=1, verbose=1)

    if 0:  # some tests to play with angles LUT

        # these parameters cannot be randomly chosen (they are related each other)
        # But by setting one angle to 90. you may play a lot with the other lattice parameters
        a, b, c, alpha, beta, gamma = 5., 2.3, 4.86, 89.56, 68.25, 90.
        # a,b,c,alpha,beta,gamma = 4.99,4.99,17.061,90,90,120 # trigonal, 'hexagonal' lattice

        # compute metric tensor
        Gstar = CP.Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma)

        # HKL_list = np.array([[0,0,1],[1,0,4],[-1,1,4],[1,2,0]])
        # for hkl1 in HKL_list[:-1]:
            # for hkl2 in HKL_list[1:]:
                # angle = CP.AngleBetweenTwoNormals(hkl1, hkl2 ,Gstar)
                # print "<%s , %s> = %.5f"%(str(hkl1),str(hkl2),angle)


        # some hkl lists
        hkl1 = GT.Positiveindices_up_to(1)
        hkl2 = GT.Positiveindices_up_to(1)[4:5]
        # build LUT
        LUT = GenerateLookUpTable_from2sets(hkl1, hkl2, Gstar)


        # test_hkl = np.array([[5,-5,3],[203,77,91],[-4,2,1],[6,3,3],[8,4,2],[1,1,1],[2,2,2],[0,2,8]])
        # GCDs = GT.GCD(np.abs(test_hkl))

        # res = GT.reduceHKL(test_hkl)

        pp, LUT = PlanePairs_from2sets(60, 1,
                                    hkl1, hkl2, Gstar,
                                    onlyclosest=1, filterharmonics=1, verbose=1)



        # Gstar = np.eye(3)
        # pp = PlanePairs(56.6, 2, Gstar, 3, onlyclosest = 0, filterharmonics = 0, verbose = 0)

        # test_hkl = GT.Positiveindices_up_to(3)
        # purgedhkl = FilterHarmonics(test_hkl)

        # Gstar = np.eye(3)
        # pp = PlanePairs(56.6, 3, Gstar,1, onlyclosest = 1, verbose = 0)
        # purged_pp = FilterEquivalentPairsHKL(pp)

    if 0:  # test of orient matrix from distorted unit cell

        from numpy import *
        a, b, c, alpha, beta, gamma = 3.2, 2.5, 5., 86, 90., 120.
        latticeparams = a, b, c, alpha, beta, gamma
        B = CP.calc_B_RR([a, b, c, alpha, beta, gamma])
        # metric tensor ---------
        Gstar_metric = CP.Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma)

        # hkl = np.array([[-2,0,2],[-6,0,2], [-3,1,1],[-2,4,2]])
        # matorient = eye(3)

        hkl = np.array([[3, 1, 7], [1, 1, 3], [2, 2, 2], [5, 1, 1]])
        matorient = array([[-0.6848285 , 0.2544487, -0.6828365],
                            [-0.7064087, -0.0017916 , 0.7078019],
                            [ 0.1788759 , 0.9670846 , 0.1809717]])  # surf 111

        # Digest of Laue6.py --------------------
        # G* taken into account distorsion
        Gstarprime = dot(B, hkl.T).T


        print("matrorient for simulation", matorient)

        q = dot(matorient, Gstarprime.T).T


        qn = sqrt(sum(q ** 2, axis=1))

        uq = q / reshape(qn, (len(qn), 1))

        # ui in Lauetools frame
        ui = array([1, 0, 0.])
        # uf = ui -2(ui.uq)uq

        coef = -2.*dot(ui, uq.T)
        uf = ui + uq * reshape(coef, (len(qn), 1))

        # kf = ( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi) XMAS
        # kf = ( cos 2theta, sin 2theta sin chi,   , sin 2theta cos chi) LT

        EPS = 0.00000001
        PI = np.pi
        # in LT
        chi = (180.0 / PI) * arctan(uf[:, 1] / (uf[:, 2] + EPS))
        twtheta = (180.0 / PI) * arccos(uf[:, 0])

        coord = array([ twtheta, chi]).T
        # ---------------------------------------------------------------

        # coord = [[90,0],[143.1301,0],[139.1214,    26.5651],[70.5288,    45.0000]]

        hkl1 = hkl[0]
        coord1 = coord[0] + array([0.1, -1])  # add some noise

        hkl2 = hkl[2]
        coord2 = coord[2] + array([-.05, .7])  # add some noise

        mat = OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B, verbose='yes', frame='lauetools')

        # --------------------------------------------------
        # check from all kinds of situation

        # res_mat = []

        # for i in range(len(hkl)-1):
            # for j in range(i+1,len(hkl)):


                # hkl1 = hkl[i]
                # hkl2 = hkl[j]

                # coord1 = coord[i] +array([0.1,-1]) # add some noise
                # coord2 = coord[j] +array([-.05,.7])

                # mat = OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B, verbose='yes', frame = 'lauetools')

                # res_mat.append(mat)
                # print "\n\nmatorient",mat
                # print "*****************\n"

        # --------------------------------------------------


        if 0:
            # build linear matrix equation to solve with pinv
            # for m = 1,2,3
            # we look for operator that solve
            # qm = (U) Gm*
            # size(qm) = size(Gm*) = 3
            # shape (U) = (3,3)
            # let U = A whose element are a ij
            # the three equations (qix,qiy,qiz) = ((a11,a12,a13),(a21,a22,a23),(a31,a32,a33))(Gi*x,Gi*y,Gi*z)
            # can be rearranged like M P = Q
            # where P is 9 element of A: [a11,a12,a13,a21,a22,a23,a31,a32,a33]
            #         Q is a stack of the sthree components of each qm
            #       Q = [q1x,q1y,q1z,q2x,q2y,q2z,q3x,q3y,q3z]
            # and M is a quite sparse matrix containing row of Gm* elements
            #     M is a 3 times vertically stacked pattern:
            #  patternM =[[G1x*,G1y*,G1z*,0,0,0,0,0,0],[0,0,0,G2x*,G2y*,G2z*,0,0,0],[0,0,0,0,0,0,G3x*,G3y*,G3z*]]

            # in case of distorted unit cell: Gm* are unit vector from B Gm*
            # qm are also unit vectors


            G1 = np.dot(B, np.array(hkl1))
            G2 = np.dot(B, np.array(hkl2))
            G3 = np.cross(G1, G2)

            normG1 = np.sqrt(np.dot(G1, G1))
            normG2 = np.sqrt(np.dot(G2, G2))
            normG3 = np.sqrt(np.dot(G3, G3))

            print("norms of G1,G2,G3", normG1, normG2, normG3)

            matrice_P = np.array([G1 / normG1, G2 / normG2, G3 / normG3])

            # building M
            patternM = np.zeros((3, 9))
            patternM[0, :3] = matrice_P[0]
            patternM[1, 3:6] = matrice_P[1]
            patternM[2, 6:] = matrice_P[2]

            M = np.vstack((patternM, patternM, patternM))

            # similar matrix but with 2theta,chi coordinate
            twicetheta_1 = coord1[0]
            chi_1 = coord1[1]
            twicetheta_2 = coord2[0]
            chi_2 = coord2[1]

            # expression of unit q in chosen frame
            frame = 'lauetools'
            qq1 = F2TC.unit_q(twicetheta_1 , chi_1, frame=frame)
            qq2 = F2TC.unit_q(twicetheta_2 , chi_2, frame=frame)

            qq3prod = np.cross(qq1, qq2)  # can be negative, we want to have finally a matrix with one eigenvalue of +1
            qq3n = np.sqrt(np.dot(qq3prod, qq3prod)) * 1.

            Q = np.ravel(np.array([qq1, qq2, qq3prod / qq3n]))

            # solving for P : MP= Q -----------------------
            # 1-  use SVD:
            U, s, V = np.linalg.svd(M)
            # then P   =  V diag(1/s) U.T Q
            # with diag(1/s) is a diagonal matrix made of inverse of singular values
            # if one singular values is very close to 0 then the inverse of this value must replace by zero!!
            print("singular values", s)

            threshold = 0.001
            invs = np.where(s < threshold, 0, 1. / s)

            print("inverse singular value", invs)
            invS = np.diag(invs)

            invM = np.dot(V, np.dot(invS, U.T))

            P = np.dot(invM, Q)


