# -*- coding: utf-8 -*-
"""
#  laue6.py programm to compute Laue Pattern in various geometry
#  J. S. Micha   micha [at] esrf [dot] fr
# version August 2014
#  from LaueTools package
#  http://sourceforge.net/projects/lauetools/


Look at the main part of this code to have details and examples on how laue6
can be used

"""

__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
__version__ = '$Revision: 1717$'

import time, math, pickle
import builtins

# import annot
import numpy as np
np.set_printoptions(precision=15)
# http://www.scipy.org/Cookbook/Matplotlib/Transformations
from pylab import figure, show, connect, title
from matplotlib.transforms import offset_copy as offset

# LaueTools modules
import CrystalParameters as CP
import generaltools as GT
import IOLaueTools as IOLT
from dict_LaueTools import dict_Rot, dict_Materials, dict_Vect, dict_Extinc, CST_ENERGYKEV, SIGN_OF_GAMMA
# TODO: LTGeo to be removed
import LaueGeometry as LTGeo

try:
    import generatehkl
    USE_CYTHON = True
except ImportError:
    print("Cython compiled module for fast computation of Laue spots is not installed!")
    USE_CYTHON = False
    
DEG = np.pi / 180.

# Default constant
DEFAULT_DETECTOR_DISTANCE = 70.  # mm
DEFAULT_DETECTOR_DIAMETER = 165.  # mm
DEFAULT_TOP_GEOMETRY = 'Z>0'

#--- ---------- Spot class
class spot:
    """
    Laue Spot class
    """
    def __init__(self, indice):
        self.Millers = indice
        self.Qxyz = None
        self.EwaldRadius = None
        self.Xcam = None
        self.Ycam = None
        self.Twicetheta = None
        self.Chi = None
#        self._psilatitude = None
#        self._philongitude = None

#     def isfcc(self):
#         """ return True if spot indices are compatible with fcc struture
#         (all even or all odd)
#         """
#         listtst = [x % 2 == 0 for x in self.Millers]
#         if listtst == [1, 1, 1] or listtst == [0, 0, 0]:
#             return True
#         else:
#             return False

    def __lt__(self, autre):
        """ renvoie True si la norme de self est plus petite que celle de autre
        """
        vvself = np.array(self.Millers)
        vvautre = np.array(autre.Millers)
        return np.dot(vvself, vvself) < np.dot(vvautre, vvautre)

    def __le__(self, autre):
        """ renvoie True si la norme de self est plus petite ou egale a celle de autre
        """
        vvself = np.array(self.Millers)
        vvautre = np.array(autre.Millers)
        return np.dot(vvself, vvself) <= np.dot(vvautre, vvautre)

    def __eq__(self, autre):
        """ renvoie True si self et autre sont egaux (coordonnees strictement egales)
        """
        return self.Millers == autre.Millers

        # def __hash__(self):
        #    return reduce(lambda x, y:x + y, self.have_fond().Millers, 0)

    def __hash__(self):
        """
        to used indices for defining a key and using set() sorted()  ... object
        comparison
        In our case used for eliminating spots which equal fondamental key
        Warning: Indice must not be changed , intead troubles:
            see http://wiki.python.org/moin/DictionaryKeys
        """
        # listint = tuple(self.have_fond().Millers)
        # still have -2 -4 -2   =  -2 -2 -4
        III = self.have_fond_indices()
        listint = tuple(np.array([100000, 1000, 1]) + III)
        return listint.__hash__()

#     def have_fond(self):
#         """
#         veco = np.array(self.Millers)
#         absvec = abs(veco[veco != 0])
#         return spot(1.*veco / pgcdl(absvec))
#         """
#         interlist = filter(lambda x:x != 0, list(self.Millers))
#         inter = map(lambda elem:int(math.fabs(elem)), interlist)
#         # print "interlist ",interlist
#         # print "inter ", inter
#         return spot((np.array(self.Millers) / GT.pgcdl(inter)))

    def have_fond_indices(self):
        """
        return prime indices (fondamental direction)
        """
        interlist = [x for x in list(self.Millers) if x != 0]
        inter = [int(math.fabs(elem)) for elem in interlist]
        # print "interlist ",interlist
        # print "inter ", inter
        return np.array(self.Millers) / GT.pgcdl(inter)

#     def is_insidesphere(self, wave):
#         """ Renvoie True si self est dans la sphere d'ewald
#         de rayon 1 / wavelength (en angstrom-1) et centre en (-r, 0, 0).
#         l'argument radius est une longueur d'onde       """
#         # print math.sqrt((self.Qxyz[0]-1. / radius) ** 2+self.Qxyz[1] ** 2+self.Qxyz[2] ** 2),1. / radius
#         recip_pos_trans = np.array(self.Qxyz) + np.array([1. / wave, 0., 0.])
#         # return (self.Qxyz[0]+1. / wave) ** 2+self.Qxyz[1] ** 2+self.Qxyz[2] ** 2  <= 1. / wave ** 2
#         return np.dot(recip_pos_trans, recip_pos_trans) <= 1. / wave ** 2
#
#     def is_insidespherecentered(self, wave):
#         """ Renvoie True si self est dans la sphere d'ewald
#         de rayon 1 / wavelength (en angstrom-1) et centre en (0, 0, 0).
#         l'argument radius est une longueur d'onde       """
#         # print math.sqrt((self.Qxyz[0]-1. / radius) ** 2+self.Qxyz[1] ** 2+self.Qxyz[2] ** 2),1. / radius
#         return self.Qxyz[0] ** 2 + \
#                 self.Qxyz[1] ** 2 + \
#                 self.Qxyz[2] ** 2 <= 1. / wave ** 2

#     def sepvector(self, autre):
#         """ donne le tuple difference de self - autre
#         """
#         return (self.Millers[0] - autre.Millers[0],
#                 self.Millers[1] - autre.Millers[1],
#                 self.Millers[2] - autre.Millers[2])

#     def is_1rstneighbor_of(self, autre):
#         """ renvoie le vecteur difference (true avec if) si le vecteur separant self et autre est un vecteur definissant la maille, sinon 0 (false avec if)
#         Dans le cas cubique fcc..."""
#         res = 0
#         if self.sepvector(autre) in [(1, 1, 1), (-1, 1, 1), (1, -1, 1),
#                                     (1, 1, -1), (-1, -1, -1), (1, -1, -1),
#                                     (-1, 1, -1), (-1, -1, 1)]:
#             res = self.sepvector(autre)
#         return res

#     def is_2ndneighbor_of(self, autre):
#         """ renvoie le vecteur difference (true avec if) si le vecteur
#         separant self et autre est un vecteur definissant la maille,
#         sinon 0 (false avec if)
#         """
#         res = 0
#         if self.sepvector(autre) in [(2, 0, 0), (-2, 0, 0), (0, 2, 0),
#                                     (0, -2, 0), (0, 0, 2), (0, 0, -2),
#                                     (1, 1, 1), (-1, 1, 1), (1, -1, 1),
#                                     (1, 1, -1), (-1, -1, -1), (1, -1, -1),
#                                         (-1, 1, -1), (-1, -1, 1)]:
#             res = self.sepvector(autre)
#         return res


#--- ---------------   PROCEDURES
def Quicklist(OrientMatrix, ReciprocBasisVectors, listRSnorm, lambdamin, verbose=0):
    """
    return 6 indices min and max boundary values for each Miller index h, k, l to probe
        in order to find h, k, l spots belonging
        to the largest Ewald Sphere (wavelength min or energy maximum)

    :param OrientMatrix:  orientation matrix (3*3 matrix)
    :param ReciprocBasisVectors:      list of the three vectors a*,b*,c* in the lab frame
                            before rotation with OrientMatrix
    :param listRSnorm:      : list of the three reciprocal space lengthes of a*,b*,c*
    :param lambdamin:       : lambdamin corresponding to energy max

    :return: [[hmin,hmax],[kmin,kmax],[lmin,lmax]]
    """
#     print "OrientMatrix in Quicklist", OrientMatrix
    
    # lengthes of a*,b*,c*
    astarnorm, bstarnorm, cstarnorm = listRSnorm[:3]

    if verbose:
        print("norms of input a*,b*,c* : in Angstrom-1")
        print([astarnorm, bstarnorm, cstarnorm])

    # vec a*,b*,c* input
    vecastar, vecbstar, veccstar = np.array(ReciprocBasisVectors)

    # unit vectors
    # input vec a* normalized , ie vec a* direction
    vecastar_n = vecastar / math.sqrt(sum(vecastar ** 2))
    vecbstar_n = vecbstar / math.sqrt(sum(vecbstar ** 2))
    veccstar_n = veccstar / math.sqrt(sum(veccstar ** 2))

    # vec a* with appropriate length
    astar = vecastar_n * astarnorm
    bstar = vecbstar_n * bstarnorm
    cstar = veccstar_n * cstarnorm

    if verbose:
        print("(a*,b*,c*) in Angstrom-1")
        print(astar)
        print(bstar)
        print(cstar)

    normastar = math.sqrt(sum(astar ** 2))
    normbstar = math.sqrt(sum(bstar ** 2))
    normcstar = math.sqrt(sum(cstar ** 2))

    if verbose:
        print("norms of a*,b*,c* : (must be equal to input norms) in Angstrom-1")
        print(normastar, normbstar, normcstar)

    # rotation matrix or orientation matrix
    matrice_rot = np.array(OrientMatrix)

    if verbose:
        print(matrice_rot)

    rotAstar = np.dot(matrice_rot, astar)  # R(vec a* with appropriate length)
    rotBstar = np.dot(matrice_rot, bstar)
    rotCstar = np.dot(matrice_rot, cstar)

    if verbose:
        print("R(a*, b*, c*) in Angstrom-1")
        print(rotAstar)
        print(rotBstar)
        print(rotCstar)

    # transfer matrix from R(a*),R(b*),R(c*) to X, Y, Z
    matricerotvecstar = np.array([rotAstar, rotBstar, rotCstar]).transpose()
    # transfer matrix from X, Y, Z to R(a*),R(b*),R(c*)
    # inv_matricerotstar = scipy.linalg.basic.inv(matricerotvecstar)
    inv_matricerotstar = np.linalg.inv(matricerotvecstar)

    if verbose:
        print(" ------------------------")
        print("transfer matrix from R(a*),R(b*),R(c*) to X, Y, Z")
        print(matricerotvecstar)

        print(" ------------------------")
        print("transfer matrix from X, Y, Z to R(a*),R(b*),R(c*)")
        print(inv_matricerotstar)

        print("1 / lambdamin:", 1. / lambdamin, " Angstrom-1")

    OC = np.array([-1. / lambdamin, 0., 0.])  # centre of the largest Ewald sphere in X, Y, Z frame

    OCrot = np.dot(inv_matricerotstar, OC)  # same centre in the R(a*),R(b*),R(c*) frame

    if verbose:
        print("center of largest Ewald Sphere in the R(a*),R(b*),R(c*) frame")
        print(OCrot)

        print("1 / lambdamin in corresponding R(a*),R(b*),R(c*) units")
        print([1. / lambdamin / normastar,
               1. / lambdamin / normbstar,
               1. / lambdamin / normcstar])

    # for non alpha = beta = gamma = 90 deg case
    # Calculate the crossproduct of rotated vector to correct properly the range on each rotated vector

    crossastar = np.cross(rotBstar, rotCstar) / \
            math.sqrt(sum(np.cross(rotBstar, rotCstar) ** 2))  # normalised cross vector for reference
    crossbstar = np.cross(rotCstar, rotAstar) / \
            math.sqrt(sum(np.cross(rotCstar, rotAstar) ** 2))
    crosscstar = np.cross(rotAstar, rotBstar) / \
            math.sqrt(sum(np.cross(rotAstar, rotBstar) ** 2))

    if verbose:
        print("crossedvector", crossastar, crossbstar, crosscstar)

    cosanglea = np.dot(rotAstar / normastar, crossastar)
    cosangleb = np.dot(rotBstar / normbstar, crossbstar)
    cosanglec = np.dot(rotCstar / normcstar, crosscstar)

    if verbose:
        print("cosangle", [cosanglea, cosangleb, cosanglec])

    hmin = GT.properinteger(OCrot[0] - 1. / lambdamin / normastar / cosanglea)
    kmin = GT.properinteger(OCrot[1] - 1. / lambdamin / normbstar / cosangleb)
    lmin = GT.properinteger(OCrot[2] - 1. / lambdamin / normcstar / cosanglec)

    hmax = GT.properinteger(OCrot[0] + 1. / lambdamin / normastar / cosanglea)
    kmax = GT.properinteger(OCrot[1] + 1. / lambdamin / normbstar / cosangleb)
    lmax = GT.properinteger(OCrot[2] + 1. / lambdamin / normcstar / cosanglec)

    # print "hmM, kmM, lmM",[hmin, hmax, kmin, kmax, lmin, lmax]
    # SECURITY -1
    Hmin = min(hmin, hmax) - 1
    Kmin = min(kmin, kmax) - 1
    Lmin = min(lmin, lmax) - 1

    # SECURITY +1 and +1 because of exclusion convention for slicing in python
    # (superior limit in range is excluded)
    Hmax = max(hmin, hmax) + 2
    Kmax = max(kmin, kmax) + 2
    Lmax = max(lmin, lmax) + 2

    # print "Hmin, Kmin, Lmin",Hmin, Kmin, Lmin
    # print "Hmax, Kmax, Lmax",Hmax, Kmax, Lmax

    try:
        list_hkl_limits = [[int(Hmin), int(Hmax)],
             [int(Kmin), int(Kmax)],
              [int(Lmin), int(Lmax)]]
        return list_hkl_limits
    except ValueError:
        return None


def both_even_or_odd(x, y):
    """
    return True is x and y are both even or odd
    """
    return ((x - y) % 2) == 0


def iseven(x):
    """
    return True is x is even
    """
    return (x % 2) == 0


def isodd(x):
    """
    return True is x is odd
    """
    return not iseven(x)


def issumeven(x, y, z):
    """
    return True is x + y + z is even
    """
    return iseven(x + y + z)


def is4n(x):
    """
    return True if x can be divided by 4 (x = 4n)
    """
    return (x % 4) == 0


def is4nplus2(x):
    """
    return True if the remainder of x divided by 4 is 2 (x = 4n + 2)
    """
    return (x % 4) == 2


def genHKL_np(listn, Extinc):
    """
    generate all Miller indices from indices limits on h k l
    and taking into account for systematic exctinctions

    :param listn: Miller indices limits (warning: these lists are used in python range (last index is excluded))
    :type listn: [[hmin,hmax],[kmin,kmax],[lmin,lmax]]

    :param Extinc: label corresponding to systematic exctinction 
            rules on h k and l('fcc', 'bcc', 'dia') or 'no' for any rules 
    :type Extinc: string

    :return: array of [h,k,l]

    Note: node [0,0,0] is excluded
    """
    if listn is None:
        raise ValueError("hkl ranges are undefined")
        return None

#    print "inside genHKL_np"
    if isinstance(listn, list) and len(listn) == 3:
        try:
            n_h_min, n_h_max = listn[0]
            n_k_min, n_k_max = listn[1]
            n_l_min, n_l_max = listn[2]
        except:
            raise TypeError("arg #1 has not the shape (3, 2)")
            return None
    else:
        raise TypeError("arg #1 is not a list or has not 3 elements")
        return None

    nbelements = (n_h_max - n_h_min) * \
            (n_k_max - n_k_min) * \
            (n_l_max - n_l_min)

    if (not isinstance(nbelements, int)) or nbelements <= 0.:
        raise ValueError("Needs (3,2) list of sorted integers")
        return None

    if (Extinc not in list(dict_Extinc.values())):
        raise ValueError('Could not understand extinction code: " %s " ' % \
                                                                str(Extinc))

    HKLraw = np.mgrid[n_h_min:n_h_max, n_k_min:n_k_max, n_l_min:n_l_max]
    HKLs = HKLraw.shape
#    print "HKLs", HKLs
    HKL = HKLraw.T.reshape((HKLs[1] * HKLs[2] * HKLs[3], HKLs[0]))
    
    return ApplyExtinctionrules(HKL, Extinc)
    

def ApplyExtinctionrules(HKL, Extinc, verbose=0):
    """
    apply selection rules to HKL (n,3) ndarray
    """
    H, K, L = HKL.T

    if verbose:
        print("nb of reflections before extinctions %d" % len(HKL))
        
    # 'dia' adds selection rules to those of 'fcc'
    if Extinc in ('fcc', 'dia'):
        cond1 = ((H - K) % 2 == 0) * ((H - L) % 2 == 0)
        if Extinc == 'dia':
            conddia = (((H + K + L) % 4) != 2)
            cond1 = cond1 * conddia

        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)

    elif Extinc == 'bcc':
        cond1 = ((H + K + L) % 2 == 0)
        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)
        
    elif Extinc == 'h+k=2n':  # group space 12  I2/m
        cond1 = ((H + K) % 2 == 0)
        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)
        
    elif Extinc == 'h+k+l=2n':  # group space 139 indium
        cond1 = ((H + K +L) % 2 == 0)
        array_hkl = np.take(HKL, np.where(cond1 == True)[0], axis=0)

    elif Extinc == 'Al2O3':
        cond1 = ((-H + K + L) % 3 == 0)
        cond2 = ((L) % 2 == 0)
        cond = cond1 * cond2
        array_hkl = np.take(HKL, np.where(cond == True)[0], axis=0)
        
    elif Extinc == 'Ti2AlN':
        
        # wurtzite condition
        condfirst = (H == K) * ((L % 2) != 0)
        array_hkl_1 = np.delete(HKL, np.where(condfirst == True)[0], axis=0)

        H, K, L = array_hkl_1.T
        
        # other conditions due to symmetries
        cond_lodd = ((L) % 2 == 1)
        cond1 = ((H - K) % 3 == 0)
        
        cond = cond_lodd * cond1
        array_hkl = np.delete(array_hkl_1, np.where(cond == True)[0], axis=0)


    elif Extinc == 'wurtzite':
        cond1 = (H - K == 0)
        cond2 = ((L) % 2 != 0)
        cond = cond1 * cond2
        array_hkl = np.delete(HKL, np.where(cond == True)[0], axis=0)

#         H, K, L = array_hkl_1.T

#         cond3 = ((L) % 2 != 0)
#         array_hkl_2 = np.delete(array_hkl_1, np.where(cond3 == True)[0], axis=0)
#
#         H, K, L = array_hkl_2.T

#         cond4 = ((L) % 2 == 0)
#         cond5 = (((H - K) % 3) != 0)
#
#         cond6 = cond4 * cond5
#
#         array_hkl = np.take(array_hkl_1, np.where(cond6 == True)[0], axis=0)

    elif Extinc == 'magnetite':  # GS 227
        # TODO not working ...
        cond = ((H + K) % 2 == 0) * ((H + L) % 2 == 0) * ((K + L) % 2 == 0)

        array_hkl_1 = np.take(HKL, np.where(cond == True)[0], axis=0)

        print("len", len(array_hkl_1))

        H, K, L = array_hkl_1.T

        cond0kl_1 = ((K) % 2 == 0) * ((L) % 2 == 0) * ((K + L) % 4 != 0)
        cond0kl_2 = ((H) % 2 == 0) * ((L) % 2 == 0) * ((H + L) % 4 != 0)
        cond0kl_3 = ((H) % 2 == 0) * ((K) % 2 == 0) * ((H + K) % 4 != 0)

        cond0kl = (H == 0) * cond0kl_1 + (K == 0) * cond0kl_2 + (L == 0) * cond0kl_3

        array_hkl_2 = np.delete(array_hkl_1, np.where(cond0kl == True)[0], axis=0)

        print("len", len(array_hkl_2))

        H, K, L = array_hkl_2.T

        condhhl_1 = (H == K) * ((H + L) % 2 != 0)
        condhhl_2 = (K == L) * ((L + H) % 2 != 0)
        condhhl_3 = (L == H) * ((H + K) % 2 != 0)

        condhhl = condhhl_1 + condhhl_2 + condhhl_3

        array_hkl_3 = np.delete(array_hkl_2, np.where(condhhl == True)[0], axis=0)

        print("len 3", len(array_hkl_3))

        H, K, L = array_hkl_3.T

        condh00_1 = (K == 0) * (L == 0) * (H % 4 != 0)
        condh00_2 = (L == 0) * (H == 0) * (K % 4 != 0)
        condh00_3 = (H == 0) * (K == 0) * (L % 4 != 0)

        condh00 = condh00_1 + condh00_2 + condh00_3

        array_hkl_0 = np.delete(array_hkl_3, np.where(condh00 == True)[0], axis=0)

        print("len f", len(array_hkl_0))

        H, K, L = array_hkl_0.T

        cond8a = ((H % 2 == 1) + (K % 2 == 1) + (L % 2 == 1)) + ((H + K + L) % 4 == 0)

        array_hkl_00 = np.take(array_hkl_0, np.where(cond8a == True)[0], axis=0)

        print("len", len(array_hkl_00))

        H, K, L = array_hkl_00.T

        cond16d = ((H % 2 == 1) + (K % 2 == 1) + (L % 2 == 1)) + \
                ((H % 4 == 2) * (K % 4 == 2) * (L % 4 == 2)) + \
                ((H % 4 == 0) * (K % 4 == 0) * (L % 4 == 0))

        array_hkl = np.take(HKL, np.where(cond16d == True)[0], axis=0)

        print("len", len(array_hkl))

    elif Extinc == 'Al2O3_rhombo':
        cond1 = ((H - K) == 0)
        cond2 = ((L) % 2 == 0)
        cond3 = ((H + K + L) % 2 == 0)
        cond = cond1 * cond2 * cond3
        array_hkl = np.take(HKL, np.where(cond == True)[0], axis=0)

    elif Extinc == 'VO2_mono':
        cond1a = (K == 0)
        cond1b = ((L) % 2 != 0)
        cond1 = cond1a * cond1b

        array_hkl_1 = np.delete(HKL, np.where(cond1 == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond2a = (H == 0)
        cond2b = (L == 0)
        cond2c = ((K) % 2 != 0)

        cond2 = cond2a * cond2b * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)

        cond3a = (H == 0)
        cond3b = (K == 0)
        cond3c = ((L) % 2 != 0)
        cond3 = cond3a * cond3b * cond3c

        array_hkl = np.delete(array_hkl_2, np.where(cond3 == True)[0], axis=0)
        
    elif Extinc =='VO2_mono2':
        
 
        cond1 = ((H+K) % 2 != 0)


        array_hkl_1 = np.delete(HKL, np.where(cond1 == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond2a = (K == 0)
        cond2c = ((H) % 2 != 0)

        cond2 = cond2a * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)

        cond3a = (H == 0)
        cond3c = ((K) % 2 != 0)
        cond3 = cond3a * cond3c

        array_hkl_3 = np.delete(array_hkl_2, np.where(cond3 == True)[0], axis=0)
        
        cond4a = (L == 0)
        cond4c = ((H+K) % 2 != 0)
        cond4 = cond4a * cond4c

        array_hkl_4 = np.delete(array_hkl_3, np.where(cond4 == True)[0], axis=0)
        
        cond5a = (H == 0)
        cond5b = (L == 0)
        cond5c = ((K) % 2 != 0)
        cond5 = cond5a * cond5b*cond5c

        array_hkl_5 = np.delete(array_hkl_4, np.where(cond5 == True)[0], axis=0)
        
        cond6a = (K == 0)
        cond6b = (L == 0)
        cond6c = ((H) % 2 != 0)
        cond6 = cond6a * cond6b*cond6c

        array_hkl = np.delete(array_hkl_5, np.where(cond6 == True)[0], axis=0)
        
    elif Extinc =='rutile':
        
        cond1a = (H == 0)
        cond1b = ((K+L) % 2 != 0)
        cond1 = cond1a * cond1b

        array_hkl_1 = np.delete(HKL, np.where(cond1 == True)[0], axis=0)

        H, K, L = array_hkl_1.T

        cond2a = (H == 0)
        cond2b = (K == 0)
        cond2c = ((L) % 2 != 0)

        cond2 = cond2a * cond2b * cond2c
        array_hkl_2 = np.delete(array_hkl_1, np.where(cond2 == True)[0], axis=0)

        cond3a = (K == 0)
        cond3b = (L == 0)
        cond3c = ((H) % 2 != 0)
        cond3 = cond3a * cond3b * cond3c

        array_hkl = np.delete(array_hkl_2, np.where(cond3 == True)[0], axis=0)
        

    # no extinction rules
    else:
        array_hkl = HKL

    # removing the node 000
    pos000 = np.where(np.all((array_hkl == 0), axis=1) == True)[0]
#     print 'pos000', pos000
    array_hkl = np.delete(array_hkl, pos000, axis=0)
    
    if verbose:
        print("nb of reflections after extinctions %d" % len(array_hkl))

#     print "shape array_hkl", np.shape(array_hkl)
#     print array_hkl[:5]
    return array_hkl


#--- -----------------------  Main procedures
def readinputparameters(SingleCrystalParams):
    """
    extract and test type of the 4 elements of SingleCrystalParams
    """
    try:
        Bmatrix, Extinc, Orientmatrix, key_for_dict = SingleCrystalParams
    except IndexError:
        raise IndexError("List of input parameters for crystal Laue pattern simulation "
                         "has %d elements instead of 4 !" % len(SingleCrystalParams))

    if Extinc not in list(dict_Extinc.values()):
        raise ValueError("wrong code %s for extinction !" % Extinc)
    if np.array(Bmatrix).shape != (3, 3):
        raise ValueError("wrong B matrix format!")
    if np.array(Orientmatrix).shape != (3, 3):
        raise ValueError("wrong Orientmatrix format!")
    return Bmatrix, Extinc, Orientmatrix, key_for_dict


def getLaueSpots(wavelmin, wavelmax,
                    crystalsParams,
                    linestowrite,
                    extinction=None,
                    kf_direction=DEFAULT_TOP_GEOMETRY,
                    OpeningAngleCollection=22.,
                    fastcompute=0,
                    ResolutionAngstrom=False,
                    fileOK=1,
                    verbose=1
                    ):
    r"""
    build a list of laue spots

    :param wavelmin:   smallest wavelength in Angstrom
    :param wavelmax:  largest wavelength in Angstrom

    :param crystalsParams:     list of *SingleCrystalParams*, each of them being a list
                        of 4 elements for crystal orientation and strain properties:

                        SingleCrystalParams[0]: is the B matrix a*,b*,c* vectors are expressed in column
                        in LaueTools frame in reciprocal angstrom units

                        SingleCrystalParams[1]: peak Extinction rules ('no','fcc','dia', etc...)

                        SingleCrystalParams[2]: orientation matrix

                        SingleCrystalParams[3]: key for material element

    :param linestowrite:   list of [string] that can be write in file or display in
                    stdout. Example: [[""]] or [["**********"],["lauetools"]]

    :param kf_direction:   string defining the average geometry, mean value of
                    exit scattered vector:
                        'Z>0'   top spots

                        'Y>0'   one side spots (towards hutch door)

                        'Y<0'   other side spots

                        'X>0'   transmission spots

                        'X<0'   backreflection spots
    :param fastcompute:  1 compute reciprocal space (RS) vector
                            BUT NOT the Miller indices
                         0 returns both RS vectors (normalised) and Miller indices

    :param ResolutionAngstrom: smallest interplanar distance ordered in crystal
                        in angstrom. If None: all reflections will be calculated
                        that can be time-consuming for large unit cell

    :param fileOK: 0 or 1 to write a file


    :return: wholelistvecfiltered, wholelistindicesfiltered (fastcompute = 0)
             wholelistvecfiltered, None                     (fastcompute = 1)
            where
            wholelistvecfiltered        : list of array of q 3D vectors corresponding to G* nodes
                                        (1 array per grain)
            wholelistindicesfiltered    : list of array of 3 miller indices
                                        (1 array per grain)

    .. caution::
        this method doesn't create spot instances.
        This is done in filterLaueSpots with fastcompute = 0
    .. caution::
        finer selection of nodes : on camera , without harmonics can be
        done later with filterLaueSpots()

    .. note:: lauetools laboratory frame is in this case:
        x// ki (center of ewald sphere has neagtive x component)
        z perp to x and belonging to the plane defined by x and dd vectors
        (where dd vector is the smallest vector joining sample impact point and points on CCD plane)
        y is perpendicular to x and z

    """
    if isinstance(wavelmin, (float, int)) and \
        isinstance(wavelmax, (float, int)):
        if wavelmin < wavelmax and wavelmin > 0:
            wlm = wavelmin * 1.
            wlM = wavelmax * 1.
        else:
            raise ValueError("wavelengthes must be positive and ordered")
    else:
        raise ValueError("wavelengthes must have numerical values")

    if isinstance(linestowrite, (int, float, str)):
        linestowrite = [[""]]

    if not isinstance(linestowrite, list) or \
        not isinstance(linestowrite[-1], list) or \
        not isinstance(linestowrite[-1][0], str):
        raise ValueError("Missing list of string")

    nb_of_grains = len(crystalsParams)

    if verbose:
        print("energy range: %.2f - %.2f keV" % \
                        (CST_ENERGYKEV / wlM, CST_ENERGYKEV / wlm))
        print("Number of grains: ", nb_of_grains)

    # loop over grains
    for i in range(nb_of_grains):
        try:
            Elem = dict_Materials[crystalsParams[i][3]][0]
        except (IndexError, TypeError, KeyError):
            smsg = "wrong type of input paramters: must be a list of 4 elements"
            raise ValueError(smsg)
        if verbose:
            print("# grain:  ", i, " made of ", Elem)
            print(" crystal parameters:", crystalsParams[i])
        if fileOK:
            linestowrite.append(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'])
            linestowrite.append(['grain no ', str(i), ' made of ', Elem, ' default lattice parameter ', str(dict_Materials[crystalsParams[i][3]][1])])
            linestowrite.append(['(vec* basis in lab. frame, real lattice lengthes expansion, orientation matrix, atomic number):'])
            linestowrite.append(['(orientation angles have no meanings here since orienation is given by a quaternion extracted from openGL, see below)'])
            for elem in crystalsParams[i]:
                linestowrite.append([str(elem)])
        # print dict_Materials[crystalsParams[i][3]]

    if fileOK:
        linestowrite.append(['************------------------------------------***************'])
        linestowrite.append(['energy range: ', str(CST_ENERGYKEV / wlM), ' - ', str(CST_ENERGYKEV / wlm), ' keV'])

    wholelistvecfiltered = []
    wholelistindicesfiltered = []

    # calculation of RS lattice nodes in lauetools laboratory frame and indices

    # loop over grains
    for i in range(nb_of_grains):

        Bmatrix, Extinc, Orientmatrix, key_for_dict = readinputparameters(crystalsParams[i])

        if verbose:
            print("\nin getLaueSpots()")
            print("Bmatrix", Bmatrix)
            print("Orientmatrix", Orientmatrix)
            print("key_for_dict", key_for_dict)
            print("Extinc", Extinc)
            print("")

        Bmatrix = np.array(Bmatrix)
        
        # generation of hkl nodes from Bmatrix

        listvecstarlength = np.sqrt(np.sum(Bmatrix ** 2, axis=0))

        # print "listvecstarlength",listvecstarlength

        # in Bmatrix.T,  a*, b* ,c* are rows of this argument
        #  B matrix in q= Orientmatrix B G formula
        # limitation of probed h k and l ranges
        list_hkl_limits = Quicklist(Orientmatrix,
                                        Bmatrix.T,
                                        listvecstarlength,
                                        wlm, verbose=0)
        
        # Loop over h k l 
        # ----cython optimization 
        global USE_CYTHON
        if USE_CYTHON:
            hlim, klim, llim = list_hkl_limits
            hmin, hmax = hlim
            kmin, kmax = klim
            lmin, lmax = llim
            dict_extinc = {'no':0, 'fcc':1, 'dia':2}
            try:
                ExtinctionCode = dict_extinc[Extinc]
                SPECIAL_EXTINC = False
            except KeyError:
                ExtinctionCode = 0
                SPECIAL_EXTINC = True
            
            
#             print "\n\n*******\nUsing Cython optimization\n********\n\n"
            hkls, counter = generatehkl.genHKL(hmin, hmax, kmin, kmax, lmin, lmax, ExtinctionCode)
        
            table_vec = hkls[:counter]
            
            # TODO need to remove element [0,0,0] or naturally removed in ?
            if SPECIAL_EXTINC:
#                 print "special extinction"
                table_vec = ApplyExtinctionrules(table_vec, Extinc)
                
        if not USE_CYTHON:
            table_vec = genHKL_np(list_hkl_limits, Extinc)

#         print 'len(table_vec(', len(table_vec)

        Orientmatrix = np.array(Orientmatrix)

        listrotvec = np.dot(Orientmatrix, np.dot(Bmatrix, table_vec.T))

        listrotvec_X = listrotvec[0]
        listrotvec_Y = listrotvec[1]
        listrotvec_Z = listrotvec[2]

        arraysquare = listrotvec_X ** 2 + \
                        listrotvec_Y ** 2 + \
                        listrotvec_Z ** 2

        # removing all spots that have positive X component
        cond_Xnegativeonly = listrotvec_X <= 0
        listrotvec_X = listrotvec_X[cond_Xnegativeonly]
        listrotvec_Y = listrotvec_Y[cond_Xnegativeonly]
        listrotvec_Z = listrotvec_Z[cond_Xnegativeonly]
        arraysquare = arraysquare[cond_Xnegativeonly]

        table_vec = table_vec[cond_Xnegativeonly]

#         print "len(table_vec)", len(table_vec)

        # Kf direction selection
        # top reflection 2theta = 90
        if kf_direction == 'Z>0':
            KF_condit = listrotvec_Z > 0.
        # side reflection  2theta = 90
        elif kf_direction == 'Y>0':
            KF_condit = listrotvec_Y > 0
        # side reflection  2theta = 90
        elif kf_direction == 'Y<0':
            KF_condit = listrotvec_Y < 0
        # x > -R transmission 2theta = 0
        elif kf_direction == 'X>0':
            KF_condit = listrotvec_X + \
                        1. / (2. * np.abs(listrotvec_X) / arraysquare) > 0
        # x < -R back reflection 2theta = 180
        elif kf_direction == 'X<0':
            KF_condit = listrotvec_X + \
                        1. / (2. * np.abs(listrotvec_X) / arraysquare) < 0
#        all spots in the two ewald's sphere'
        elif kf_direction == '4PI':
            KF_condit = np.ones_like(listrotvec_X) * True
        # user's definition of mean kf vector
        # [2theta , chi]= / kf = [cos2theta,sin2theta*sinchi,sin2theta*coschi]
        elif isinstance(kf_direction, (builtins.list, np.array)):
            print("\nUSING user defined LauePattern Region\n")
            if len(kf_direction) != 2:
                raise ValueError("kf_direction must be defined by a list of two angles !")
            else:
                kf_2theta, kf_chi = kf_direction
                kf_2theta, kf_chi = kf_2theta * DEG, kf_chi * DEG

                qmean_theta, qmean_chi = kf_2theta / 2., kf_chi

                print("central q angles", qmean_theta, qmean_chi)

                # q.qmean >0
#                KF_condit = (-listrotvec_X * np.sin(qmean_theta) + \
#                            listrotvec_Y * np.cos(qmean_theta) * np.sin(qmean_chi) + \
#                            listrotvec_Z * np.cos(qmean_theta) * np.cos(qmean_chi)) > 0

#                # angle(q, qmean) < 45 deg
#                AngleMax = OpeningAngleCollection * DEG
#                KF_condit = np.arccos(((-listrotvec_X * np.sin(qmean_theta) + \
#                            listrotvec_Y * np.cos(qmean_theta) * np.sin(qmean_chi) + \
#                            listrotvec_Z * np.cos(qmean_theta) * np.cos(qmean_chi))) / np.sqrt(arraysquare)) < AngleMax

                # angle(kf, kfmean) < 45 deg
                AngleMax = OpeningAngleCollection * DEG

                Rewald = arraysquare / (2 * np.fabs(listrotvec_X))
                kfsquare = Rewald ** 2
                KF_condit = np.arccos(((listrotvec_X + Rewald) * np.cos(kf_2theta) + \
                            listrotvec_Y * np.sin(kf_2theta) * np.sin(kf_chi) + \
                            listrotvec_Z * np.sin(kf_2theta) * np.cos(kf_chi)) / np.sqrt(kfsquare)) < AngleMax
        else:
            raise ValueError("kf_direction '%s' is not understood!" \
                                            % str(kf_direction))

#        print "nb of KF_condit", len(np.where(KF_condit == True)[0])

        # vectors inside the two Ewald's spheres
        # last condition could be perhaps put before on
        # listrotvec_X[listrotvec_Z > 0]...
        Condit = np.logical_and(\
                np.logical_and(\
                ((listrotvec_X * 2. / wlm + arraysquare) <= 0.),
                ((listrotvec_X * 2. / wlM + arraysquare) > 0.)),
                (KF_condit))

#         print "nb of spots in spheres and in towards CCD region", len(np.where(Condit == True)[0])

        if fastcompute == 0:
#             print 'Using detailled computation mode'

            if ResolutionAngstrom:

                # #crude resolution limitation 2Angstrom for lattice [20 4.8 49]
                # Hcond = np.abs(table_vec[:,0])<10
                # Kcond = np.abs(table_vec[:,1])<3
                # Lcond = np.abs(table_vec[:,2])<25
                # ConditResolution = np.logical_and(np.logical_and((Hcond) ,
                #                                                  (Kcond)),
                #                                                   (Lcond))

                ConditResolution = arraysquare < (1. / ResolutionAngstrom) ** 2

                Condit = np.logical_and(Condit, ConditResolution)

            # print "Condit",Condit
            fil_X = np.compress(Condit, listrotvec_X)
            fil_Y = np.compress(Condit, listrotvec_Y)
            fil_Z = np.compress(Condit, listrotvec_Z)

            # print "fil_Z",fil_Z
            fil_H = np.compress(Condit, table_vec[:, 0])
            fil_K = np.compress(Condit, table_vec[:, 1])
            fil_L = np.compress(Condit, table_vec[:, 2])

            listvecfiltered = np.transpose(np.array([fil_X, fil_Y, fil_Z]))
            listindicesfiltered = np.transpose(np.array([fil_H, fil_K, fil_L]))

#             print "listindicesfiltered", len(listindicesfiltered)

            # if 0:
                # print "listrotvec[0]",listrotvec[0]
                # print "fgfghfg",fil_X
                # Energyarray = CST_ENERGYKEV*2*np.abs(listrotvec_X) / arraysquare
                # Efiltered = np.compress(Condit, Energyarray)
                # print "energyfiltered",Efiltered
                # print "listindicesfiltered",listindicesfiltered

            # to return
            wholelistvecfiltered.append(listvecfiltered)
            wholelistindicesfiltered.append(listindicesfiltered)

            if verbose:
                print("Orientation matrix", Orientmatrix)
                print("# grain: ", i)
                print("Number of spots for # grain ", \
                                                i, ": ", len(listvecfiltered))

            if fileOK:
                linestowrite.append(['# grain :', str(i)])
                linestowrite.append(['Orientation matrix given as input :', str(Orientmatrix)])
                linestowrite.append(['Number of spots for # grain  ', str(i), ': ', str(len(listvecfiltered))])
                linestowrite.append(['*****-------------*****'])

        # only q vectors are calculated, not Miller indices
        elif fastcompute == 1:
#            print 'Using fastcompute mode'

            if ResolutionAngstrom:

                # #crude resolution limitation 2Angstrom for lattice [20 4.8 49]
                # Hcond = np.abs(table_vec[:,0])<10
                # Kcond = np.abs(table_vec[:,1])<3
                # Lcond = np.abs(table_vec[:,2])<25
                # ConditResolution = np.logical_and(np.logical_and((Hcond) ,
                #                                                  (Kcond)),
                #                                                   (Lcond))

                ConditResolution = arraysquare < (1. / ResolutionAngstrom) ** 2

                Condit = np.logical_and(Condit, ConditResolution)

            # print "Condit", Condit
            fil_X = np.compress(Condit, listrotvec_X)
            fil_Y = np.compress(Condit, listrotvec_Y)
            fil_Z = np.compress(Condit, listrotvec_Z)

            # to return
            listvecfiltered = np.transpose(np.array([fil_X, fil_Y, fil_Z]))
            wholelistvecfiltered.append(listvecfiltered)
            # wholelistindicesfiltered is empty

            if verbose:
                print("Orientation matrix", Orientmatrix)
                print("# grain: ", i)
                print("Rotating all spots")
                print("Nb of spots for # grain  ", \
                            i, ": ", len(listvecfiltered))

            if fileOK:
                linestowrite.append(['# grain :', str(i)])
                linestowrite.append(['Orientation matrix given as input :', str(Orientmatrix)])
                linestowrite.append(['Number of spots for # grain  ', str(i), ': ', str(len(listvecfiltered))])
                linestowrite.append(['*****-------------*****'])

    return wholelistvecfiltered, wholelistindicesfiltered


def create_spot(pos_vec, miller, detectordistance,
                allattributes=False,
                pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot instance (on top camera geometry)

    :param pos_vec: 3D vector
    :type pos_vec: list of 3 float
    :param  miller: list of 3 miller indices
    :param detectordistance: approximate distance detector sample (to compute complementary spots attributes)
    :param  allattributes: False or 0  not to compute complementary spot attributes
    :param  allattributes: boolean

    :return: spot instance

    spot.Qxyz is a vector expressed in lauetools frame
    X along x-ray Z towards CCD when CCD on top and
    y towards experimental hutch door
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2. * math.fabs(vecres[0]))
    if spotty.Qxyz[2] > 0:
        normkout = math.sqrt((spotty.Qxyz[0] + spotty.EwaldRadius) ** 2 + \
                             spotty.Qxyz[1] ** 2 + \
                             spotty.Qxyz[2] ** 2)

        if allattributes not in (False, 0):
            X = detectordistance * (spotty.Qxyz[0] + spotty.EwaldRadius) / spotty.Qxyz[2]
            Y = detectordistance * (spotty.Qxyz[1]) / spotty.Qxyz[2]
            spotty.Xcam = -X / pixelsize + dim[0] / 2
            spotty.Ycam = -Y / pixelsize + dim[1] / 2

        spotty.Twicetheta = math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG
#        spotty.Chi = math.atan(spotty.Qxyz[1] * 1. / spotty.Qxyz[2]) / DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1., spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_np(Qxyz, miller, detectordistance,
                allattributes=False,
                pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot instance (on top camera geometry)

    :param pos_vec: 3D vector
    :type pos_vec: list of 3 float
    :param  miller: list of 3 miller indices
    :param detectordistance: approximate distance detector sample (to compute complementary spots attributes)
    :param  allattributes: False or 0  not to compute complementary spot attributes
    :param  allattributes: boolean

    :return: spot instance

    Qxyz is a vector expressed in lauetools frame
    X along x-ray Z towards CCD when CCD on top and
    y towards experimental hutch door
    """
#     print "Qxyz", Qxyz
    EwaldRadius = np.sum(Qxyz ** 2, axis=1) / (2. * np.abs(Qxyz[:, 0]))
    ki = np.zeros((len(Qxyz), 3))
    ki[:, 0] = EwaldRadius
    
    kout = ki + Qxyz
#     if spotty.Qxyz[2] > 0:
    normkout = np.sqrt(np.sum(kout ** 2, axis=1))
    
#     print "kout[:, 0]", kout[:, 0]
#     print "len(kout[:, 0])", len(kout[:, 0])
#     print "normkout", normkout
#     print "len(normkout)", len(normkout)
    

    Twicetheta = np.arccos((kout[:, 0]) / normkout) / DEG
#        spotty.Chi = math.atan(spotty.Qxyz[1] * 1. / spotty.Qxyz[2]) / DEG
    Chi = np.arctan2(kout[:, 1] * 1., kout[:, 2]) / DEG
    
    Energy = EwaldRadius * CST_ENERGYKEV

    return Twicetheta, Chi, Energy, miller



def create_spot_4pi(pos_vec, miller,
                    detectordistance, allattributes=0,
                pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot scattered in 4pi steradian no camera position

    spot.Qxyz is a vector expressed in lauetools frame
    X along x-ray Z towards plane defined by CCD frame and
    y = z ^ x
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2. * math.fabs(vecres[0]))

    normkout = math.sqrt((spotty.Qxyz[0] + spotty.EwaldRadius) ** 2 + \
                         spotty.Qxyz[1] ** 2 + \
                         spotty.Qxyz[2] ** 2)

    spotty.Twicetheta = math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG
#    spotty.Chi = math.atan(spotty.Qxyz[1] * 1. / spotty.Qxyz[2]) / DEG
    spotty.Chi = math.atan2(spotty.Qxyz[1] * 1., spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_side_pos(pos_vec, miller,
                        detectordistance, allattributes=0,
                        pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot on side camera
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2. * math.fabs(vecres[0]))
    if spotty.Qxyz[1] > 0:
        normkout = math.sqrt((spotty.Qxyz[0] + spotty.EwaldRadius) ** 2 + spotty.Qxyz[1] ** 2 + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = detectordistance * (spotty.Qxyz[0] + spotty.EwaldRadius) / spotty.Qxyz[1]
            Y = detectordistance * (spotty.Qxyz[2]) / spotty.Qxyz[1]
#             spotty.Xcam = X / pixelsize + dim[0] / 2
#             spotty.Ycam = Y / pixelsize + dim[1] / 2
            spotty.Xcam = X / pixelsize
            spotty.Ycam = Y / pixelsize
            
        spotty.Twicetheta = math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1., spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_side_neg(pos_vec, miller,
                         detectordistance, allattributes=0,
                         pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot on neg side camera
    TODO: update with dim as other create_spot()
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2. * math.fabs(vecres[0]))

    if spotty.Qxyz[1] < 0.:
        normkout = math.sqrt((spotty.Qxyz[0] + spotty.EwaldRadius) ** 2 + spotty.Qxyz[1] ** 2 + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = -detectordistance * (spotty.Qxyz[0] + spotty.EwaldRadius) / spotty.Qxyz[1]
            Y = detectordistance * (spotty.Qxyz[2]) / spotty.Qxyz[1]
            spotty.Xcam = X / pixelsize + dim[0] / 2.
            spotty.Ycam = Y / pixelsize + dim[1] / 2.

        spotty.Twicetheta = math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1., spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_front(pos_vec, miller, detectordistance,
                      allattributes=0,
                      pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot on forward direction transmission geometry
    """
#     print "use create_spot_front"
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2. * math.fabs(vecres[0]))
    
    if miller[0] == 0 and miller[1] == 0 and miller[2] == 0:
        print("MILLER == MILER ==0")

    if spotty.Qxyz[0] < 0.:
        abskx = math.fabs(spotty.Qxyz[0] + spotty.EwaldRadius)
        normkout = 1.*math.sqrt(abskx ** 2 + spotty.Qxyz[1] ** 2 + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = -detectordistance * (spotty.Qxyz[1]) / abskx
            Y = -detectordistance * (spotty.Qxyz[2]) / abskx
            spotty.Xcam = X / pixelsize + dim[0] / 2.
            spotty.Ycam = Y / pixelsize + dim[1] / 2.

        spotty.Twicetheta = math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1., spotty.Qxyz[2]) / DEG

    return spotty


def create_spot_back(pos_vec, miller, detectordistance,
                     allattributes=0, pixelsize=165. / 2048, dim=(2048, 2048)):
    """ From reciprocal space position and 3 miller indices
    create a spot on backward direction back reflection geometry
    """
    spotty = spot(miller)
    spotty.Qxyz = pos_vec
    vecres = np.array(pos_vec)
    spotty.EwaldRadius = np.dot(vecres, vecres) / (2. * math.fabs(vecres[0]))
    if spotty.Qxyz[0] < 0.:
        abskx = math.fabs(spotty.Qxyz[0] + spotty.EwaldRadius)
        normkout = math.sqrt(abskx ** 2 + spotty.Qxyz[1] ** 2 + spotty.Qxyz[2] ** 2)

        if not allattributes:
            X = detectordistance * (spotty.Qxyz[1]) / abskx
            Y = detectordistance * (spotty.Qxyz[2]) / abskx
            spotty.Xcam = X / pixelsize + dim[0] / 2.
            spotty.Ycam = Y / pixelsize + dim[1] / 2.

        spotty.Twicetheta = math.acos((spotty.Qxyz[0] + spotty.EwaldRadius) / normkout) / DEG
        # spotty.Chi = math.atan(spotty.Qxyz[1]*1. / spotty.Qxyz[2])/DEG
        spotty.Chi = math.atan2(spotty.Qxyz[1] * 1., spotty.Qxyz[2]) / DEG

    return spotty


def Calc_spot_on_cam_sansh(listoflistofspots, fileOK=0, linestowrite=[[""]]):

    """ Calculates list of grains spots on camera w / o harmonics,
    from liligrains (list of grains spots instances scattering in 2pi steradians)
    [[spots grain 0],[spots grain 0],etc] = >
    [[spots grain 0],[spots grain 0],etc] w / o harmonics and on camera  CCD

    TODO: useless  only used in Plot_Laue(), need to be replace by filterLaueSpots
    """

    nbofgrains = len(listoflistofspots)

    _oncam = emptylists(nbofgrains)  # list of spot on the camera (harmonics included)
    _oncamsansh = emptylists(nbofgrains)  # list of spot on the camera (harmonics EXCLUDED) only fondamental (but still with E > Emin)

    if fileOK:
        linestowrite.append(['\n'])
        linestowrite.append(['--------------------------------------------------------'])
        linestowrite.append(['------------- Simulation Data --------------------------'])
        linestowrite.append(['--------------------------------------------------------'])
        linestowrite.append(['\n'])
        linestowrite.append(['#grain, h, k, l, energy(keV), 2theta (deg), chi (deg), X_Xmas, Y_Xmas, X_JSM, Y_JSM, Xtest, Ytest'])

    for i in range(nbofgrains):
        condCam = elem.Xcam ** 2 + elem.Ycam ** 2 <= (DEFAULT_DETECTOR_DIAMETER / 2) ** 2
        _oncam[i] = [elem for elem in listoflistofspots[i] if (elem.Xcam is not None and condCam)]

        # Creating list of spot without harmonics
        _invdict = {}  # for each grain, use of _invdict to remove harmonics present in _oncam

        for elem in sorted(_oncam[i], reverse=True):
            _invdict[elem.__hash__()] = elem
            _oncamsansh[i] = [_invdict[cle] for cle in list(_invdict.keys())]

        if fileOK:
            IOLT.Writefile_data_log(_oncamsansh[i], i, linestowrite=linestowrite)

    return _oncamsansh


def filterLaueSpots(vec_and_indices,
                    HarmonicsRemoval=1,
                    fastcompute=0,
                    kf_direction=DEFAULT_TOP_GEOMETRY,
                    fileOK=0,
                    detectordistance=DEFAULT_DETECTOR_DISTANCE,
                    detectordiameter=DEFAULT_DETECTOR_DIAMETER,
                    pixelsize=165. / 2048,
                    dim=(2048, 2048),
                    linestowrite=[[""]]
                    ):
    """ Calculates list of grains spots on camera and without harmonics
    and on CCD camera
    from [[spots grain 0],[spots grain 1],etc] =>
    returns [[spots grain 0],[spots grain 1],etc] w / o harmonics and on camera  CCD

    :param vec_and_indices : list of elements corresponding to 1 grain,
                    each element is composed by: [0] array of vector
                            [1] array of indices

    :param HarmonicsRemoval: 1 remove harmonics according to their miller indices
                            (only for fastcompute = 0)

    :param fastcompute:  1 output a list for each grain of 2theta spots and a list for each grain of chi spots
                            (HARMONICS spots are still HERE!)
                           0 output list for each grain of spots with


    :param kf_direction: label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type kf_direction: string

    :return: list of spot instances if fastcompute=0
    
                2theta, chi          if fastcompute=1

    #TODO: add dim in create_spot in various geometries
    #TODO: complete the doc
    """
    try:
        Qvectors_list, HKLs_list = vec_and_indices
    except ValueError:
        raise ValueError("vec_and_indices has not two elements")

    try:
        nbofgrains = len(Qvectors_list)
    except TypeError:
        raise TypeError("Qvectors_list is not a iterable !")

    if not isinstance(Qvectors_list[0], np.ndarray):
        raise TypeError("first element of Qvectors_list is not surprisingly an array")

    # preparing list of results
    if fastcompute == 0:
        ListSpots_Oncam_wo_harmonics = emptylists(nbofgrains)
    elif fastcompute == 1:
        Oncam2theta = emptylists(nbofgrains)
        Oncamchi = emptylists(nbofgrains)

    # loop over grains
    for grainindex in range(nbofgrains):
        try:
            Qx = Qvectors_list[grainindex][:, 0] * 1.
            Qy = Qvectors_list[grainindex][:, 1] * 1.
            Qz = Qvectors_list[grainindex][:, 2] * 1.
            if np.shape(Qvectors_list[grainindex])[1] != 3:
                raise TypeError
        except IndexError:
            raise IndexError("vec_and_indices has not the proper shape")

        Qsquare = Qx ** 2 + Qy ** 2 + Qz ** 2

#        print "Qx", Qx

        # correspondinf Ewald sphere radius for each spots
        # (proportional to photons Energy)
        Rewald = Qsquare / 2. / np.abs(Qx)

        # Kf direction selection
        if kf_direction == 'Z>0':  # top reflection geometry
            ratiod = detectordistance / Qz
            Ycam = ratiod * (Qx + Rewald)
            Xcam = ratiod * (Qy)
        elif kf_direction == 'Y>0':  # side reflection geometry (for detector between the GMT hutch door and the sample (beam coming from right to left)
            ratiod = detectordistance / Qy
            Xcam = ratiod * (Qx + Rewald)
            Ycam = ratiod * (Qz)
            
        elif kf_direction == 'Y<0':  # other side reflection
            ratiod = detectordistance / np.abs(Qy)
            Xcam = -ratiod * (Qx + Rewald)
            Ycam = ratiod * (Qz)
        elif kf_direction == 'X>0':  # transmission geometry
            ratiod = detectordistance / np.abs(Qx + Rewald)
            Xcam = -1. * ratiod * (Qy)
            Ycam = -ratiod * (Qz)
        elif kf_direction == 'X<0':  # back reflection geometry
            ratiod = detectordistance / np.abs(Qx + Rewald)
            Xcam = ratiod * (Qy)
            Ycam = ratiod * (Qz)
        elif kf_direction == '4PI':  # to keep all scattered spots
            Xcam = np.zeros_like(Qy)
            Ycam = np.zeros_like(Qy)

        elif isinstance(kf_direction, (list, np.array)):
            if len(kf_direction) != 2:
                raise ValueError("kf_direction must be defined by a list of two angles !")
            else:
                Xcam = np.zeros_like(Qy)
                Ycam = np.zeros_like(Qy)
        else:
            raise ValueError("Unknown laue geometry code for kf_direction parameter")

        # print "Xcam, Ycam",Xcam
        # print Ycam
        # print "******************"
        # On camera filter
        # print "detectordiameter", detectordiameter
        halfCamdiametersquare = (detectordiameter / 2.) ** 2
        # TODO: should contain Xcam-Xcamcen (mm) and Ycam-Ycamcen with Xcamcen, Ycamcen
        # given by user
        onCam_cond = Xcam ** 2 + Ycam ** 2 <= halfCamdiametersquare 
        # resulting arrays
        oncam_Qx = np.compress(onCam_cond, Qx)
        oncam_Qy = np.compress(onCam_cond, Qy)
        oncam_Qz = np.compress(onCam_cond, Qz)

        oncam_R = np.compress(onCam_cond, Rewald)
        oncam_XplusR = oncam_Qx + oncam_R

        # compute 2theta, chi
        oncam_2theta = np.arccos(oncam_XplusR / np.sqrt(oncam_XplusR ** 2 + \
                                            oncam_Qy ** 2 + oncam_Qz ** 2))
        # be careful of the of sign
        oncam_chi = np.arctan(1. * oncam_Qy / oncam_Qz)
#         oncam_chi = np.arctan2(1. * oncam_Qy, oncam_Qz)
        # TODO: replace by arctan2(1. * oncam_Qy ,oncam_Qz) ??

#        print "oncam_2theta", oncam_2theta
#        print "oncam_chi", oncam_chi
#        print "len(oncam)", len(oncam_2theta)

        # creates spot instances which takes some times...
        # will compute spots on the camera
        # (if HarmonicsRemoval = 0 harmonics still exist)
        # (if HarmonicsRemoval = 1 harmonics are removed, fundamentals are kept)
        # (those of lowest energy)
        if fastcompute == 0:
            try:
                indi_H = HKLs_list[grainindex][:, 0]
                indi_K = HKLs_list[grainindex][:, 1]
                indi_L = HKLs_list[grainindex][:, 2]
                if np.shape(HKLs_list[grainindex])[1] != 3:
                    raise TypeError
            except TypeError:
                emsg = "HKLs_list must be a list of array of 3D vectors"
                raise TypeError(emsg)

            oncam_vec = np.array([oncam_Qx, oncam_Qy, oncam_Qz]).T

            oncam_H = np.compress(onCam_cond, indi_H)
            oncam_K = np.compress(onCam_cond, indi_K)
            oncam_L = np.compress(onCam_cond, indi_L)
            oncam_HKL = np.transpose(np.array([oncam_H, oncam_K, oncam_L]))

            # build list of spot objects
            listspot = get2ThetaChi_geometry(oncam_vec, oncam_HKL,
                                             detectordistance=detectordistance,
                                             pixelsize=pixelsize,
                                             dim=dim,
                                             kf_direction=kf_direction)

#             print "oncam_HKL", oncam_HKL.tolist()
            # Creating list of spot with or without harmonics
            if HarmonicsRemoval and listspot:
#                ListSpots_Oncam_wo_harmonics[grainindex] = RemoveHarmonics(listspot)
                (oncam_HKL_filtered,
                 toremove) = CP.FilterHarmonics_2(oncam_HKL,
                                                  return_indices_toremove=1)
                listspot = np.delete(np.array(listspot), toremove).tolist()

            # feeding final list of spots
            ListSpots_Oncam_wo_harmonics[grainindex] = listspot

            if fileOK:
                IOLT.Writefile_data_log(ListSpots_Oncam_wo_harmonics[grainindex],
                                           grainindex,
                                           linestowrite=linestowrite)

            # print "Number of spot in camera w / o harmonics",len(ListSpots_Oncam_wo_harmonics[grainindex])

        # (fastcompute = 1) no instantiation of spot object
        # will return 2theta, chi for each grain
        # no harmonics removal
        elif fastcompute == 1:
            Oncam2theta[grainindex] = oncam_2theta
            Oncamchi[grainindex] = oncam_chi

    # outputs and returns
    if fileOK:
        linestowrite.append(['\n'])
        linestowrite.append(['--------------------------------------------------------'])
        linestowrite.append(['------------- Simulation Data --------------------------'])
        linestowrite.append(['--------------------------------------------------------'])
        linestowrite.append(['\n'])
        linestowrite.append(['#grain, h, k, l, energy(keV), 2theta (deg), chi (deg), X_Xmas, Y_Xmas, X_JSM, Y_JSM, Xtest, Ytest'])

    if fastcompute == 0:
        # list of elements which are list of objects of spot class (1 element / grain)
        return ListSpots_Oncam_wo_harmonics
    elif fastcompute == 1:
        # list of elements which are composed by two arrays (2theta, chi) (1 element / grain)
        return (np.concatenate(Oncam2theta) / DEG,
                np.concatenate(Oncamchi) / DEG)


def filterLaueSpots_full_np(veccoord, indicemiller,
                    HarmonicsRemoval=1,
                    fastcompute=0,
                    kf_direction=DEFAULT_TOP_GEOMETRY,
                    fileOK=0,
                    detectordistance=DEFAULT_DETECTOR_DISTANCE,
                    detectordiameter=DEFAULT_DETECTOR_DIAMETER,
                    pixelsize=165. / 2048,
                    dim=(2048, 2048),
                    linestowrite=[[""]],
                    grainindex=0
                    ):
    """ Calculates spots data for an individual grain
    on camera and without harmonics
    and on CCD camera

    :param vec_and_indices : list of elements corresponding to 1 grain,
                    each element is composed by: [0] array of q vector
                                                [1] array of indices

    :param HarmonicsRemoval: 1 remove harmonics according to their miller indices
                            (only for fastcompute = 0)

    :param fastcompute:  1 output a list for each grain of 2theta spots and a list for each grain of chi spots
                            (HARMONICS spots are still HERE!)
                           0 output list for each grain of spots with


    :param kf_direction: label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type kf_direction: string

    :return: tuple of lists of Twtheta Chi Energy Millers if fastcompute=0
    
                tuple of lists 2theta, chi          if fastcompute=1

    #TODO: add dim in create_spot in various geometries
    #TODO: complete the doc
    """
    VecX = veccoord[:, 0] * 1.
    VecY = veccoord[:, 1] * 1.
    VecZ = veccoord[:, 2] * 1.

    Vecsquare = VecX ** 2 + VecY ** 2 + VecZ ** 2

#        print "VecX", VecX

    # correspondinf Ewald sphere radius for each spots
    # (proportional to photons Energy)
    Rewald = Vecsquare / 2. / np.abs(VecX)

    # Kf direction selection
    if kf_direction == 'Z>0':  # top reflection geometry
        # VecZ is >0
        ratiod = detectordistance / VecZ
        Ycam = ratiod * (VecX + Rewald)
        Xcam = ratiod * (VecY)
    elif kf_direction == 'Y>0':  # side reflection geometry (for detector between the GMT hutch door and the sample (beam coming from right to left)
        ratiod = detectordistance / VecY
        Xcam = ratiod * (VecX + Rewald)
        Ycam = ratiod * (VecZ)
    elif kf_direction == 'Y<0':  # other side reflection
        ratiod = detectordistance / np.abs(VecY)
        Xcam = -ratiod * (VecX + Rewald)
        Ycam = ratiod * (VecZ)
    elif kf_direction == 'X>0':  # transmission geometry
        ratiod = detectordistance / np.abs(VecX + Rewald)
        Xcam = -1. * ratiod * (VecY)
        Ycam = -ratiod * (VecZ)
    elif kf_direction == 'X<0':  # back reflection geometry
        ratiod = detectordistance / np.abs(VecX + Rewald)
        Xcam = ratiod * (VecY)
        Ycam = ratiod * (VecZ)
    elif kf_direction == '4PI':  # to keep all scattered spots
        Xcam = np.zeros_like(VecY)
        Ycam = np.zeros_like(VecY)

    elif isinstance(kf_direction, (list, np.array)):
        if len(kf_direction) != 2:
            raise ValueError("kf_direction must be defined by a list of two angles !")
        else:
            Xcam = np.zeros_like(VecY)
            Ycam = np.zeros_like(VecY)
    else:
        raise ValueError("Unknown laue geometry code for kf_direction parameter")

    # print "Xcam, Ycam",Xcam
    # print Ycam
    #----------------------------------------------------------
    # Approximate selection of spots in camera accroding to Xcam and Ycam
    # onCam_cond  conditions
    #------------------------------------------------------------
    
    # print "******************"
    # On camera filter
    # print "detectordiameter", detectordiameter
    halfCamdiametersquare = (detectordiameter / 2.) ** 2
    # TODO: should contain Xcam-Xcamcen (mm) and Ycam-Ycamcen with Xcamcen, Ycamcen
    # given by user
    onCam_cond = Xcam ** 2 + Ycam ** 2 <= halfCamdiametersquare 
    # resulting arrays
    oncam_vecX = np.compress(onCam_cond, VecX)
    oncam_vecY = np.compress(onCam_cond, VecY)
    oncam_vecZ = np.compress(onCam_cond, VecZ)

    # creates spot instances which takes some times...
    # will compute spots on the camera
    # (if HarmonicsRemoval = 0 harmonics still exist)
    # (if HarmonicsRemoval = 1 harmonics are removed, fundamentals are kept)
    # (those of lowest energy)
    if fastcompute == 0:

        indi_H = indicemiller[:, 0]
        indi_K = indicemiller[:, 1]
        indi_L = indicemiller[:, 2]

        oncam_vec = np.array([oncam_vecX, oncam_vecY, oncam_vecZ]).T

        oncam_H = np.compress(onCam_cond, indi_H)
        oncam_K = np.compress(onCam_cond, indi_K)
        oncam_L = np.compress(onCam_cond, indi_L)
        oncam_HKL = np.transpose(np.array([oncam_H, oncam_K, oncam_L]))

        # build list of spot objects
        TwthetaChiEnergyMillers_list_one_grain = get2ThetaChi_geometry_full_np(oncam_vec, oncam_HKL,
                                         detectordistance=detectordistance,
                                         pixelsize=pixelsize,
                                         dim=dim,
                                         kf_direction=kf_direction)

#         print 'TwthetaChiEnergy_list_one_grain', TwthetaChiEnergy_list_one_grain
        # Creating list of spot with or without harmonics
        if HarmonicsRemoval and True:  # listspot:
            raise ValueError("Harmonic removal is not implemented and listspot does not exist anymore")
#                ListSpots_Oncam_wo_harmonics[i] = RemoveHarmonics(listspot)
            (oncam_HKL_filtered,
             toremove) = CP.FilterHarmonics_2(oncam_HKL,
                                              return_indices_toremove=1)
            listspot = np.delete(np.array(listspot), toremove).tolist()

        if fileOK:
            IOLT.Writefile_data_log(TwthetaChiEnergyMillers_list_one_grain,
                                       grainindex,
                                       linestowrite=linestowrite)

        # print "Number of spot in camera w / o harmonics",len(ListSpots_Oncam_wo_harmonics[i])

    # (fastcompute = 1) no instantiation of spot object
    # will return 2theta, chi for each grain
    # no harmonics removal
    elif fastcompute == 1:
        
        oncam_R = np.compress(onCam_cond, Rewald)
        oncam_XplusR = oncam_vecX + oncam_R
        # compute 2theta, chi
        oncam_2theta = np.arccos(oncam_XplusR / np.sqrt(oncam_XplusR ** 2 + \
                                            oncam_vecY ** 2 + oncam_vecZ ** 2)) / DEG
        # be careful of the of sign
        oncam_chi = np.arctan(1. * oncam_vecY / oncam_vecZ) / DEG
        #         oncam_chi = np.arctan2(1. * oncam_vecY, oncam_vecZ)
        # TODO: replace by arctan2(1. * oncam_vecY ,oncam_vecZ) ??
        
        #        print "oncam_2theta", oncam_2theta
        #        print "oncam_chi", oncam_chi
        #        print "len(oncam)", len(oncam_2theta)


    # outputs and returns
    if fileOK:
        linestowrite.append(['\n'])
        linestowrite.append(['--------------------------------------------------------'])
        linestowrite.append(['------------- Simulation Data --------------------------'])
        linestowrite.append(['--------------------------------------------------------'])
        linestowrite.append(['\n'])
        linestowrite.append(['#grain, h, k, l, energy(keV), 2theta (deg), chi (deg), X_Xmas, Y_Xmas, X_JSM, Y_JSM, Xtest, Ytest'])

    if fastcompute == 0:
        # list spots data
        return TwthetaChiEnergyMillers_list_one_grain
    elif fastcompute == 1:
        # list of elements which are composed by two arrays (2theta, chi) (1 element / grain)
        return oncam_2theta, oncam_chi


def get2ThetaChi_geometry(oncam_vec, oncam_HKL,
                          detectordistance=DEFAULT_DETECTOR_DISTANCE,
                          pixelsize=165. / 2048,
                          dim=(2048, 2048),
                          kf_direction=DEFAULT_TOP_GEOMETRY):
    """
    compute list of spots instances from oncam_vec (q 3D vectors)
    and oncam_HKL (miller indices 3D vectors)

    :param oncam_vec: q vectors [qx,qy,qz] (corresponding to kf collected on camera)
    :type oncam_vec: array with 3D elements (shape = (n,3))

    :param dim: CCD frame dimensions (nb pixels, nb pixels)
    :type dim: list or tuple of 2 integers

    :param detectordistance: approximate distance detector sample
    :param detectordistance: float or integer

    :param: kf_direction : label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type: kf_direction: string

    :param pixelsize: pixel size in mm
    :type pixelsize: float
    
    :return: list of spot

    TODO: to be replaced by something not using spot class
    TODO: put this function obviously in find2thetachi ?
    """
    if len(oncam_vec) != len(oncam_HKL):
        raise ValueError("Wrong input for get2ThetaChi_geometry()")

    listspot = []
    options_createspot = {'allattributes': 0,
                          'pixelsize': pixelsize,
                          'dim': dim}

    dictcase = {'Z>0': create_spot,
                'Y>0': create_spot_side_pos,
                'Y<0': create_spot_side_neg,
                'X>0': create_spot_front,
                'X<0': create_spot_back,
                '4PI': create_spot_4pi}

    for position, indices in zip(oncam_vec, oncam_HKL):
        try:
            function_create_spot = dictcase[kf_direction]
        except TypeError:
            function_create_spot = create_spot_4pi

        listspot.append(function_create_spot(position,
                                                indices,
                                                detectordistance,
                                                **options_createspot))
    return listspot


def get2ThetaChi_geometry_full_np(oncam_vec, oncam_HKL,
                          detectordistance=DEFAULT_DETECTOR_DISTANCE,
                          pixelsize=165. / 2048,
                          dim=(2048, 2048),
                          kf_direction=DEFAULT_TOP_GEOMETRY):
    """
    compute 2theta chi from oncam_vec (q 3D vectors) and oncam_HKL (miller indices 3D vectors)
    for all spots of one grain

    :param oncam_vec: q vectors [qx,qy,qz] (corresponding to kf collected on camera)
    :type oncam_vec: array with 3D elements (shape = (n,3))

    :param dim: CCD frame dimensions (nb pixels, nb pixels)
    :type dim: list or tuple of 2 integers

    :param detectordistance: approximate distance detector sample
    :param detectordistance: float or integer

    :param: kf_direction : label for detection geometry
                    (CCD plane with respect to
                    the incoming beam and sample)
    :type: kf_direction: string

    :param pixelsize: pixel size in mm
    :type pixelsize: float
    
    :return: list of spot

    TODO: put this function obviously in find2thetachi ?
    """
    if len(oncam_vec) != len(oncam_HKL):
        raise ValueError("Wrong input for get2ThetaChi_geometry()")

    TwthetaChiEnergy_list = []
    options_createspot = {'allattributes': 0,
                          'pixelsize': pixelsize,
                          'dim': dim}

    dictcase = {'Z>0': create_spot_np,
                'Y>0': create_spot_side_pos,
                'Y<0': create_spot_side_neg,
                'X>0': create_spot_np,
                'X<0': create_spot_back,
                '4PI': create_spot_4pi}

    try:
        function_create_spot = dictcase[kf_direction]
    except TypeError:
        function_create_spot = create_spot_4pi

    TwthetaChiEnergyMillers_list = function_create_spot(oncam_vec,
                                                oncam_HKL,
                                                detectordistance,
                                                **options_createspot)
    return TwthetaChiEnergyMillers_list


def RemoveHarmonics(listspot):
    """
    remove harmonics present in listspot (list of objects of spot class)
    """
    _invdict = {}

    # print "len after sorting wtih respect to __hash__()",len(sorted(listspot, reverse = True))
    for elem in sorted(listspot, reverse=True):
        # print "elem, elem.__hash__()",elem.Millers, elem.__hash__()

        _invdict[elem.__hash__()] = elem
        _oncamsansh = [_invdict[cle] for cle in list(_invdict.keys())]

    # print "Number of fundamental spots (RS directions): %d"%len(_oncamsansh)
    return _oncamsansh


def calcSpots_fromHKLlist(UB, B0, HKL, dictCCD):
    """
    computes all Laue Spots properties on CCD camera from list of hkl

    :param UB: orientation matrix (rotation -and if any- strain)
    :type UB: 3x3 array (or list)

    :param B0: initial a*,b*,c* reciprocal unit cell basis vector in Lauetools frame (x// ki))
    :type B0: 3x3 array (or list)

    :param HKL: array of Miller indices
    :type HKL: array with shape = (n,3)

    :param dictCCD: dictionnary of CCD properties (with key 'CCDparam', 'pixelsize','dim')
    for 'ccdparam' 5 CCD calibration parameters [dd,xcen,ycen,xbet,xgam], pixelsize in mm, and (dim1, dim2)
    :param dictCCD: dict object

    Fundamental equation
    q = UB*B0 * Gstar
    with Gstar = h*astar+k*bstar+l*cstar
    """

    detectorparam = dictCCD['CCDparam']
    pixelsize = dictCCD['pixelsize']
    dim = dictCCD['dim']
    if 'kf_direction' in dictCCD:
        kf_direction = dictCCD['kf_direction']
    else:
        kf_direction = 'Z>0'

    # H,K,L
    tHKL = np.transpose(HKL)

    # initial lattice rotation and distorsion

    tQ = np.dot(np.dot(UB, B0), tHKL)
    # results are qx,qy,qz

    # Q**2
    Qsquare = np.sum(tQ ** 2, axis=0)
    # norms of Q vectors
    Qn = 1. * np.sqrt(Qsquare)

    twthe, chi = LTGeo.from_qunit_to_twchi(tQ / Qn, labXMAS=0)

    X, Y, theta = LTGeo.calc_xycam_from2thetachi(twthe,
                                            chi, detectorparam,
                                            verbose=0,
                                            pixelsize=pixelsize,
                                            dim=dim,
                                            signgam=SIGN_OF_GAMMA,
                                            kf_direction=kf_direction)

    # E = (C)*(-q**2/qx/2)
    Energy = (CST_ENERGYKEV) * (-0.5 * Qsquare / tQ[0])

    # theoretical values
    H, K, L = tHKL
    Qx, Qy, Qz = tQ

    return H, K, L, Qx, Qy, Qz, X, Y, twthe, chi, Energy


def emptylists(n):
    """
    build a list of empty lists : [[],[],[], ...,[]]

    Warning: [[]]*n  leads to something dangerous in python!!
    xx=[[]]*2 ; xx[0]=2 => [2,[]]
    but
    xx=[[]]*2 ; xx[0].append((2) => [[2],[2]] instead of [[2],[]] !!
    """
    return [[] for k in range(n)]


def Plot_Laue(_emin, _emax,
              list_of_spots,
              _data_2theta, _data_chi,
              _data_filename,
              Plot_Data=1,
              Display_label=1,
              What_to_plot='2thetachi',
              saveplotOK=1,
            WriteLogFile=1,
            removeharmonics=0,
            kf_direction=DEFAULT_TOP_GEOMETRY,
            linestowrite=[[""]]):
    """
    basic function to plot LaueData for showcase
    """
    from . import annot
    # creation donnee pour etiquette sur figure
    # if plotOK:
    font = {'fontname': 'Courier',
            'color': 'k', 'fontweight': 'normal',
            'fontsize': 7}
    fig = figure(figsize=(6., 6.), dpi=80)  # ? small but ok for savefig!
    ax = fig.add_subplot(111)

    title('Laue pattern %.1f-%.1f keV \n %s' % (_emin, _emax, _data_filename),
                                                  font, fontsize=12)
        # text(0.5, 2.5, 'a line', font, color = 'k')

    if removeharmonics == 0:
        oncam_sansh = Calc_spot_on_cam_sansh(list_of_spots,
                                             fileOK=WriteLogFile,
                                             linestowrite=linestowrite)
    elif removeharmonics == 1:
        print("harmonics have been already removed")
        oncam_sansh = list_of_spots

    nb_of_grains = len(list_of_spots)
    print("Number of grains in Plot_Laue", nb_of_grains)

    xxx = emptylists(nb_of_grains)
    yyy = emptylists(nb_of_grains)

    energy = emptylists(nb_of_grains)
    trans = emptylists(nb_of_grains)
    indy = emptylists(nb_of_grains)

    # calculation for a list of spots object of spots belonging to the camera direction

    if WriteLogFile:
        linestowrite.append(["-" * 60])
        linestowrite.append([" Number of spots w / o harmonics: "])

    # building  simul data
    # loop over grains
    for i in range(nb_of_grains):

        print("Number of spots on camera (w / o harmonics): GRAIN number ", i , \
                                        " : ", len(oncam_sansh[i]))
        if WriteLogFile:
            linestowrite.append(["--- GRAIN number ", str(i), " : ", \
                            str(len(oncam_sansh[i])), " ----------"])
        facscale = 1.
        # loop over spots
        for elem in oncam_sansh[i]:
            if What_to_plot == '2thetachi' and (isinstance(kf_direction, list) or kf_direction in ('Z>0', 'Y>0', 'Y<0')):
                xxx[i].append(elem.Twicetheta)
                yyy[i].append(elem.Chi)
            elif What_to_plot == 'XYcam' or kf_direction in ('X>0', 'X<0'):
                facscale = 2.  # for bigger spot on graph
                xxx[i].append(elem.Xcam)
                yyy[i].append(elem.Ycam)
            elif What_to_plot == 'projgnom':  # essai pas fait
                xxx[i].append(elem.Xcam)
                yyy[i].append(elem.Ycam)
#            elif What_to_plot == 'directionq':
#                xxx[i].append(elem._philongitude)
#                yyy[i].append(elem._psilatitude)
            else:
                xxx[i].append(elem.Xcam)
                yyy[i].append(elem.Ycam)
            energy[i].append(elem.EwaldRadius * CST_ENERGYKEV)
            #if etiquette: # maintenant avec la souris!!!
            indy[i].append(elem.Millers)

        print(len(xxx[i]))

    # plot of simulation and or not data
    # affichage avec ou sans etiquettes
    dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
#        dicosymbol = {0:'k.',1:'r.',2:'g.',3:'b.',4:'c.',5:'m.'}

    # exp data
    if _data_2theta is not None and _data_chi is not None and Plot_Data:
        xxx_data = tuple(_data_2theta)
        yyy_data = tuple(_data_chi)

        # PLOT EXT DATA
        # print "jkhhkjhkjhjk**********",xxx_data[:20]
        ax.scatter(xxx_data, yyy_data,
                   s=40, c='w', marker='o', faceted=True, alpha=0.5)
        # END PLOT EXP DATA

    # Plot Simulated laue patterns
    for i in range(nb_of_grains):  # loop over grains
        # ax.plot(tuple(xxx[i]),tuple(yyy[i]),dicosymbol[i])

        # display label to the side of the point
        if nb_of_grains >= 3:
#            trans[i] = offset(ax, fig, 10, -5 * (nb_of_grains - 2) + 5 * i)
            trans[i] = offset(ax.transData, fig, 10, -5 * (nb_of_grains - 2) + 5 * i, units='dots')
        else:
#            trans[i] = offset(ax, fig, 10, 5 * (-1) ** (i % 2))
            trans[i] = offset(ax.transData, fig, 10, 5 * (-1) ** (i % 2), units='dots')

        print("nb of spots for grain # %d : %d" % (i, len(xxx[i])))
#        print tuple(xxx[i])
#        print tuple(yyy[i])
        ax.scatter(tuple(xxx[i]), tuple(yyy[i]),
                        s=[facscale * int(200. / en) for en in energy[i]],
                        c=dicocolor[(i + 1) % (len(dicocolor))],
                        marker='o',
                        faceted=False,
                        alpha=0.5)

        if Display_label:  # displaying labels c and d at position a, b
            for a, b, c, d in zip(tuple(xxx[i]), tuple(yyy[i]),
                                  tuple(indy[i]), tuple(energy[i])):
                ax.text(a, b, '%s,%.1f' % (str(c), d), font,
                        fontsize=7,
                        color=dicocolor[i % (len(dicocolor))],
                        transform=trans[i])

    ax.set_aspect(aspect='equal')
    if What_to_plot == 'XYcam':
        # ax.set_axis_off()
        ax.set_xlim((0., 2030.))
        ax.set_ylim((0., 2030.))
#    elif What_to_plot == 'directionq':
#        #ax.set_axis_off()
#        ax.set_xlim((120.,240.))
#        ax.set_ylim((20.,70.))

    # CCD circle contour border drawing

    t = np.arange(0., 6.38, 0.1)

    if What_to_plot == '2thetachi' and kf_direction in ('Y>0', 'Y<0'):
        if kf_direction == 'Y>0':
            si = 1
        else:
            si = -1

        circY = si * (-90 + 45 * np.sin(t))
        circX = (90 + 45 * np.cos(t))
        Xtol = 5.
        Ytol = 5.

    elif What_to_plot == '2thetachi' and (isinstance(kf_direction, list) or kf_direction in ('Z>0')):

        circX = 90 + 45 * np.cos(t)
        circY = 45 * np.sin(t)
        Xtol = 5.
        Ytol = 5.
#    elif What_to_plot == 'directionq': #essai coordonnee du cercle camera???
#        circX = map(lambda tutu: 180 + 60 * math.cos(tutu), t)
#        circY = map(lambda tutu: 45 + 25 * math.sin(tutu), t)
#        Xtol = 5.
#        Ytol = 5.
    # TODO embellished with numpy like above
    elif kf_direction in ('X>0', 'X<0'):
        circX = [1024 + 1024 * math.cos(tutu) for tutu in t]
        circY = [1024 + 1024 * math.sin(tutu) for tutu in t]
        Xtol = 20
        Ytol = 20
    else:
        circX = [1024 + 1024 * math.cos(tutu) for tutu in t]
        circY = [1024 + 1024 * math.sin(tutu) for tutu in t]
        Xtol = 20
        Ytol = 20
    ax.plot(circX, circY, '')

    if saveplotOK:
        fig.savefig('figlaue.png', dpi=300, facecolor='w',
                    edgecolor='w', orientation='portrait')

    if not Display_label:
        whole_xxx = xxx[0]
        whole_yyy = yyy[0]
        whole_indy = indy[0]
        whole_energy = energy[0]
        for i in range(nb_of_grains - 1):
            whole_xxx += xxx[i + 1]
            whole_yyy += yyy[i + 1]
            whole_indy += indy[i + 1]
            whole_energy += energy[i + 1]

        whole_energy_r = np.around(np.array(whole_energy), decimals=2)

        todisplay = np.vstack((np.array(whole_indy, dtype=np.int8).T,
                               whole_energy_r)).T

        # af =  annot.AnnoteFinder(xxx[0] + xxx[1],yyy[0] + yyy[1],indy[0] + indy[1])
        # af =  annot.AnnoteFinder(whole_xxx, whole_yyy, whole_indy, xtol = 5.,ytol = 5.)
        # af =  annot.AnnoteFinder(whole_xxx, whole_yyy, whole_energy_r, xtol = 10.,ytol = 10.)
        af = annot.AnnoteFinder(whole_xxx,
                                 whole_yyy,
                                 todisplay,
                                 xtol=Xtol,
                                 ytol=Ytol)
        # af =  AnnoteFinder(xxx[0],yyy[0],energy[0],indy[0])
        connect('button_press_event', af)
    show()


def CreateData_(list_spots, outputname='fakedatapickle', pickledOK=0):
    """
    return list de (2theta, chi)
    
    :param list_spots: list of list of instances of class 'spot'
    
    Data may be pickled if needed
    """
    mydata = [[elem.Twicetheta, elem.Chi] for elem in list_spots[0]]
    if outputname != None:
        if pickledOK:
            filepickle = open(outputname, 'w')
            pickle.dump(mydata, filepickle)
            filepickle.close()
        else:
            line_to_write = []
            line_to_write.append(['2theta   chi h k l E spotnumber'])
            spotnumber = 0
            for i in range(len(list_spots)):

                for elem in list_spots[i]:

                    line_to_write.append(
                        [
                        str(elem.Twicetheta),
                        str(elem.Chi),
                        str(elem.Millers[0]),
                        str(elem.Millers[1]),
                        str(elem.Millers[2]),
                        str(elem.EwaldRadius * CST_ENERGYKEV),
                        str(spotnumber)
                        ])
                    spotnumber += 1

            filetowrite = open(outputname, 'w')
            aecrire = line_to_write
            for line in aecrire:
                lineData = '\t'.join(line)
                filetowrite.write(lineData)
                filetowrite.write('\n')
            filetowrite.close()

    return mydata


def SimulateLaue_merge(grains, emin, emax, detectorparameters,
                       only_2thetachi=True,
                       output_nb_spots=False,
                 kf_direction=DEFAULT_TOP_GEOMETRY,
                 ResolutionAngstrom=False,
                 removeharmonics=0,
                 pixelsize=165 / 2048.,
                 dim=(2048, 2048),
                 detectordiameter=None):

    """
    simulate Laue pattern full data from a list of grains and concatenate results data 

    :param grains: list of 4 elements grain parameters

    :param only_2thetachi: True, return only concatenated grains data 2theta and chi,
                           False, return All_Twicetheta, All_Chi, All_Miller_ind,
                                    All_posx, All_posy, All_Energy

    :param output_nb_spots: True, output a second element (in addition to data)
                            with list of partial nb of spots per grain
                            (to know the grain origin of spots)
    """
    # use SimulateLaue
    if not only_2thetachi:
        All_Twicetheta = []
        All_Chi = []
        All_Miller_ind = []
        All_posx = []
        All_posy = []
        All_Energy = []
        All_nb_spots = []

        for grain in grains:
            (Twicetheta, Chi,
             Miller_ind,
             posx, posy,
             Energy) = SimulateLaue(grain, emin, emax, detectorparameters,
                                 kf_direction=kf_direction,
                                 ResolutionAngstrom=ResolutionAngstrom,
                                 removeharmonics=removeharmonics,
                                 pixelsize=pixelsize,
                                 dim=dim,
                                 detectordiameter=detectordiameter)
            nb_spots = len(Twicetheta)

            All_Twicetheta.append(Twicetheta)
            All_Chi.append(Chi)
            All_Miller_ind.append(Miller_ind)
            All_posx.append(posx)
            All_posy.append(posy)
            All_Energy.append(Energy)
            All_nb_spots.append(nb_spots)

        All_Twicetheta = np.concatenate(All_Twicetheta)
        All_Chi = np.concatenate(All_Chi)
        All_Miller_ind = np.concatenate(All_Miller_ind)
        All_posx = np.concatenate(All_posx)
        All_posy = np.concatenate(All_posy)
        All_Energy = np.concatenate(All_Energy)

        toreturn = All_Twicetheta, All_Chi, All_Miller_ind, All_posx, All_posy, All_Energy

    # Use SimulateResult
    else:
        simulparameters = {}
        simulparameters['detectordiameter'] = detectordiameter
        simulparameters['kf_direction'] = kf_direction
        simulparameters['detectordistance'] = detectorparameters[0]
        simulparameters['pixelsize'] = pixelsize

        All_TwicethetaChi = []
        All_nb_spots = []
        for grain in grains:
            TwicethetaChi = SimulateResult(grain, emin, emax,
                   simulparameters,
                   fastcompute=1,
                   ResolutionAngstrom=False)
            nb_spots = len(TwicethetaChi[0])

            All_TwicethetaChi.append(TwicethetaChi)
            All_nb_spots.append(nb_spots)

        All_TwicethetaChi = np.concatenate(All_TwicethetaChi)

        toreturn = All_TwicethetaChi

    if output_nb_spots:
        return toreturn, All_nb_spots
    else:
        return toreturn


def SimulateLaue_twins(grainparent, twins_operators, emin, emax, detectorparameters,
                       only_2thetachi=True,
                       output_nb_spots=False,
                 kf_direction=DEFAULT_TOP_GEOMETRY,
                 ResolutionAngstrom=False,
                 removeharmonics=0,
                 pixelsize=165 / 2048.,
                 dim=(2048, 2048),
                 detectordiameter=None):

    """
    simulate Laue pattern full data for twinned grain

    input:

    grainparent
    twins_operators     : list of 3*3 matrices corresponding of Matrices

    output_nb_spots  : True    output a second element with list of partial nb of spots per grain
    """
    nb_twins = len(twins_operators)

    Bmat, dilat, Umat, extinction = grainparent

    grains = [grainparent]

    for k, twin_op in enumerate(twins_operators):
        twinUmat = np.dot(Umat, twin_op)
        grains.append([Bmat, dilat, twinUmat, extinction])

    return SimulateLaue_merge(grains, emin, emax, detectorparameters,
                       only_2thetachi=only_2thetachi,
                       output_nb_spots=output_nb_spots,
                 kf_direction=kf_direction,
                 ResolutionAngstrom=ResolutionAngstrom,
                 removeharmonics=removeharmonics,
                 pixelsize=pixelsize,
                 dim=dim,
                 detectordiameter=detectordiameter)


def SimulateLaue(grain, emin, emax, detectorparameters,
                 kf_direction=DEFAULT_TOP_GEOMETRY,
                 ResolutionAngstrom=False,
                 removeharmonics=0,
                 pixelsize=165 / 2048.,
                 dim=(2048, 2048),
                 detectordiameter=None,
                 force_extinction=None):
    """Compute Laue Pattern spots positions, scattering angles, miller indices
                            for a SINGLE grain or Xtal

    :param grain:        : crystal parameters : 4 elements list
    :param emin, emax: energy bandpass limits

    :param removeharmonics: 1, remove harmonics spots and keep fondamental spots
                            (with lowest Miller indices)

    :return: single grain data: Twicetheta, Chi, Miller_ind, posx, posy, Energy

    TODO: to update to accept kf_direction not in reflection geometry 
    """

    if detectordiameter is None:
        DETECTORDIAMETER = pixelsize * dim[0]
    else:
        DETECTORDIAMETER = detectordiameter
    # use DEFAULT_TOP_GEOMETRY <=> kf_direction = 'Z>0'
    
#     print "grain", grain

    key_material = grain[3]

    grain = CP.Prepare_Grain(key_material, OrientMatrix=grain[2], force_extinction=force_extinction)
    
#     print "grain", grain



#     print "grain in SimulateResult()", grain

    Spots2pi = getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                [[""]],
                                fastcompute=0,
                                fileOK=0,
                                verbose=0,
                                kf_direction=kf_direction,
                                ResolutionAngstrom=ResolutionAngstrom)

#     print "len Spots2pi", len(Spots2pi[0][0])

    # list of spot which are on camera (with harmonics)
    ListofListofSpots = filterLaueSpots(Spots2pi,
                                    fileOK=0,
                                    fastcompute=0,
                                    detectordistance=detectorparameters[0],
                                    detectordiameter=DETECTORDIAMETER,
                                    kf_direction=kf_direction,
                                    HarmonicsRemoval=removeharmonics,
                                    pixelsize=pixelsize
                                    )

    ListofSpots = ListofListofSpots[0]

    Twicetheta = np.array([spot.Twicetheta for spot in ListofSpots])
    Chi = np.array([spot.Chi for spot in ListofSpots])
    Miller_ind = np.array([list(spot.Millers) for spot in ListofSpots])
    Energy = np.array([spot.EwaldRadius * CST_ENERGYKEV for spot in ListofSpots])

    posx, posy = LTGeo.calc_xycam_from2thetachi(Twicetheta, Chi,
                                            detectorparameters,
                                            verbose=0,
                                            signgam=LTGeo.SIGN_OF_GAMMA,
                                            pixelsize=pixelsize,
                                            dim=dim,
                                            kf_direction=kf_direction)[:2]

    return Twicetheta, Chi, Miller_ind, posx, posy, Energy


def SimulateLaue_full_np(grain, emin, emax, detectorparameters,
                 kf_direction=DEFAULT_TOP_GEOMETRY,
                 ResolutionAngstrom=False,
                 removeharmonics=0,
                 pixelsize=165 / 2048.,
                 dim=(2048, 2048),
                 detectordiameter=None,
                 force_extinction=None):
    """Compute Laue Pattern spots positions, scattering angles, miller indices
                            for a SINGLE grain or Xtal

    :param grain:        : crystal parameters : 4 elements list
    :param emin, emax: energy bandpass limits

    :param removeharmonics: 1, remove harmonics spots and keep fondamental spots
                            (with lowest Miller indices)

    :return: single grain data: Twicetheta, Chi, Miller_ind, posx, posy, Energy

    TODO: to update to accept kf_direction not in reflection geometry 
    """

    if detectordiameter is None:
        DETECTORDIAMETER = pixelsize * dim[0]
    else:
        DETECTORDIAMETER = detectordiameter
    # use DEFAULT_TOP_GEOMETRY <=> kf_direction = 'Z>0'

    key_material = grain[3]
    grain = CP.Prepare_Grain(key_material, OrientMatrix=grain[2],
                             force_extinction=force_extinction)

    print("grain in SimulateResult()", grain)
    
    Qxyz, HKL = getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                [[""]],
                                fastcompute=0,
                                fileOK=0,
                                verbose=0,
                                kf_direction=kf_direction,
                                ResolutionAngstrom=ResolutionAngstrom)

#     print "Qxyz", Qxyz
#     print "HKL", HKL

    # list of spot which are on camera (with harmonics)
    TwthetaChiEnergyMillers_list_one_grain_wo_harmonics = filterLaueSpots_full_np(Qxyz[0], HKL[0],
                                    fileOK=0,
                                    fastcompute=0,
                                    detectordistance=detectorparameters[0],
                                    detectordiameter=DETECTORDIAMETER,
                                    kf_direction=kf_direction,
                                    HarmonicsRemoval=removeharmonics,
                                    pixelsize=pixelsize
                                    )

#     print "TwthetaChiEnergyMillers_list_one_grain_wo_harmonics", TwthetaChiEnergyMillers_list_one_grain_wo_harmonics

    Twicetheta, Chi, Energy, Miller_ind = TwthetaChiEnergyMillers_list_one_grain_wo_harmonics[:4]

#     print 'len(Twicetheta)', len(Twicetheta)
#     print 'len(Chi)', len(Chi)
#     print 'len(Energy)', len(Energy)
#     print 'len(Miller_ind)', len(Miller_ind)

    posx, posy = LTGeo.calc_xycam_from2thetachi(Twicetheta, Chi,
                                            detectorparameters,
                                            verbose=0,
                                            signgam=LTGeo.SIGN_OF_GAMMA,
                                            pixelsize=pixelsize,
                                            dim=dim,
                                            kf_direction=kf_direction)[:2]
                                            
    print("nb of spots theo. ", len(Twicetheta))

    return Twicetheta, Chi, Miller_ind, posx, posy, Energy


def SimulateResult(grain, emin, emax,
                   simulparameters,
                   fastcompute=1,
                   ResolutionAngstrom=False):
    """Simulate 2theta chi of Laue Pattern spots for ONE SINGLE grain

    Need of approximate detector distance and diameter
    to restrict simulation to a limited solid angle

    Returns 2theta and chi array only
    """

    detectordiameter = simulparameters['detectordiameter']
    kf_direction = simulparameters['kf_direction']
    detectordistance = simulparameters['detectordistance']
    pixelsize = simulparameters['pixelsize']

    # PATCH: redefinition of grain to simulate any unit cell(not only cubic)
    key_material = grain[3]
    grain = CP.Prepare_Grain(key_material, OrientMatrix=grain[2])
    #-----------------------------------------------------------------------------

    # print "grain in SimulateResult()",grain

    spots2pi = getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin,
                                        [grain],
                                        [[""]],
                                        fastcompute=fastcompute,
                                        ResolutionAngstrom=ResolutionAngstrom,
                                        fileOK=0,
                                        verbose=0,
                                        kf_direction=kf_direction)
    # ---------------------------------------------------------------------------

    # array(vec) and array(indices)  of spots exiting the crystal in 2pi steradian
    # if fastcompute = 0 array(indices) = 0 and TwicethetaChi is a list of spot object
    # k_direction =(Z>0)
    # TODO: to be argument if camera is far from preset kf_direction!!
    # spots2pi = LAUE.generalfabriquespot_fromMat_veryQuick(CST_ENERGYKEV/emax, CST_ENERGYKEV/emin,[grain],1,
                                        # fastcompute = fastcompute, fileOK = 0, verbose = 0, kf_direction = 'Z>0')

    # 2theta, chi of spot which are on camera(without harmonics)
    TwicethetaChi = filterLaueSpots(spots2pi,
                                    fileOK=0,
                                    fastcompute=fastcompute,
                                    detectordistance=detectordistance,
                                    detectordiameter=detectordiameter * 1.2,
                                    kf_direction=kf_direction,
                                    pixelsize=pixelsize)

    return TwicethetaChi


def simulatepattern(grain,
                    emin, emax,
                    kf_direction,
                    data_filename,
                    PlotLaueDiagram=1,
                    Plot_Data=0,
                    verbose=0,
                    detectordistance=DEFAULT_DETECTOR_DISTANCE,
                    ResolutionAngstrom=False,
                    Display_label=1,
                    HarmonicsRemoval=1):

    whatplot = '2thetachi'


    print("kf_direction", kf_direction)

    Starting_time = time.time()
    linestowrite = [['*********'],
                ['file .log generated by laue6.py'],
                ['at %s' % (time.asctime())],
                [' ==' * 20]]
    # vec and indices of spots Z > 0 (vectors [x, y, z] array, miller indices [H, K, L] array)

    vecind = getLaueSpots(CST_ENERGYKEV / emax,
                                                CST_ENERGYKEV / emin,
                                                grain,
                                                linestowrite,
                                                fileOK=0,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                verbose=verbose,
                                                ResolutionAngstrom=ResolutionAngstrom)
    pre_filename = 'laue6simul'

    print("len(vecind[0])", len(vecind[0][0]))

    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)
    try:
        oncam_sansh = filterLaueSpots(vecind,
                                                fileOK=1,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                detectordistance=detectordistance,
                                                HarmonicsRemoval=HarmonicsRemoval,
                                                linestowrite=linestowrite)
#        print "oncam_sansh",oncam_sansh
    except UnboundLocalError:
        raise UnboundLocalError("Empty list of spots or vector (variable: vecind)")

    # plot and save and write file of simulated Data
    # third input is a list whose each element corresponds to one grain list of spots object

    data_2theta, data_chi = None, None  # exp. data

    if data_filename != None:
        # loadind experimental data:
        try:
            res = IOLT.readfile_cor(data_filename)

        except IOError:
            print("You must enter the name of experimental datafile (similar to e.g 'Ge_test.cor')")
            print("or set Plot_Data=0")
            return

        else:
            alldata, data_2theta, data_chi, data_x, data_y, data_I, detparam = res

    else:
        print("You must enter the name of experimental datafile (similar to e.g 'Ge_test.cor')")

    # ----------  Time consumption information
    finishing_time = time.time()
    duration = finishing_time - Starting_time
    print("Time duration for computation %.2f sec." % duration)

    if PlotLaueDiagram:

        Plot_Laue(emin, emax, oncam_sansh, data_2theta, data_chi, data_filename,
                            removeharmonics=1,
                            kf_direction=kf_direction,
                            Plot_Data=Plot_Data,
                            Display_label=Display_label,
                            What_to_plot=whatplot,
                            saveplotOK=0,
                            WriteLogFile=1,
                            linestowrite=linestowrite)

#     # logbook file edition
#     IOLT.writefile_log(output_logfile_name=pre_filename + '.log',
#                  linestowrite=linestowrite)

#     return CreateData_(oncam_sansh, outputname='laue6table', pickledOK=0), \
#             oncam_sansh
    return True


def simulatepurepattern(grain,
                    emin, emax,
                    kf_direction,
                    data_filename,
                    PlotLaueDiagram=1,
                    Plot_Data=0,
                    verbose=0,
                    detectordistance=DEFAULT_DETECTOR_DISTANCE,
                    ResolutionAngstrom=False,
                    Display_label=1,
                    HarmonicsRemoval=1):

    vecind = getLaueSpots(CST_ENERGYKEV / emax,
                                                CST_ENERGYKEV / emin,
                                                grain,
                                                1,
                                                fileOK=0,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                verbose=verbose,
                                                ResolutionAngstrom=ResolutionAngstrom)
    

    print("len(vecind[0])", len(vecind[0][0]))

    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)

    oncam_sansh = filterLaueSpots(vecind,
                                                fileOK=0,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                detectordistance=detectordistance,
                                                HarmonicsRemoval=HarmonicsRemoval)

    return True


def StructureFactorCubic(h,k,l,extinctions='dia'):
    pi=np.pi
    F=( 1+np.exp(-1.j*pi/2.*(h+k+l)) ) * ( 1+np.exp(-1.j*pi*(k+l))+np.exp(-1.j*pi*(h+l))+np.exp(-1.j*pi*(h+k)) )
    return F


def StructureFactorUO2(h,k,l,qvector):
    #CaF2 structure
    pi=np.pi
    fu=atomicformfactor(qvector,'U')
    fo=atomicformfactor(qvector,'O')
    F=(fu+ 2*fo*np.cos(pi/(2*(h+k+l)))) * (1+ np.exp(-1.j*pi*(k+l)) + np.exp(-1.j*pi*(l+h)) + np.exp(-1.j*pi*(h+k)) ) 
    return F

def atomicformfactor(qvector,element='Ge'):
    """
    qvector in Angst -1

    http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    """
    if (element=='Ge'):
        p=(16.0816, 2.8509, 6.3747, 0.2516, 3.7068, 11.4468 ,3.683, 54.7625,2.1313)

    elif (element=='U'):
        p=(5.3715, 0.516598, 22.5326, 3.05053, 12.0291, 12.5723, 4.79840, 23.4582, 13.2671)

    elif (element=='O'):
        p=(3.04850, 13.2771, 2.28680, 5.70110, 1.54630, 0.323900, 0.867000, 32.9089, 0.250800)

    val=0
    for k in range(4):
        val+= p[2*k]*np.exp(-p[2*k+1]*(qvector/4/np.pi)**2)
    val+=p[-1]
    return val 
    


def simulatepurepattern_np(grain,
                    emin, emax,
                    kf_direction,
                    data_filename,
                    PlotLaueDiagram=1,
                    Plot_Data=0,
                    verbose=0,
                    detectordistance=DEFAULT_DETECTOR_DISTANCE,
                    ResolutionAngstrom=False,
                    Display_label=1,
                    HarmonicsRemoval=1):

    vecind = getLaueSpots(CST_ENERGYKEV / emax,
                                                CST_ENERGYKEV / emin,
                                                grain,
                                                1,
                                                fileOK=0,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                verbose=verbose,
                                                ResolutionAngstrom=ResolutionAngstrom)
    

    print("len(vecind[0])", len(vecind[0][0]))

    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)

    oncam_sansh = filterLaueSpots_full_np(vecind,
                                                fileOK=0,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                detectordistance=detectordistance,
                                                HarmonicsRemoval=HarmonicsRemoval)

    return True

#--- -------------------  TESTS & EXAMPLES
def test_simulation():
    """
    test scenario
    """
    linestowrite = [['*********'],
                ['file .log generated by laue6.py'],
                ['at %s' % (time.asctime())],
                [' ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  == ']]

    #  SIMULATION Compute list of spots for each grains

    emin = 5
    emax = 25
    # What kind of 2D plot must be display#
    # 'xy' plot (xcam, ycam)  (X horizontal + along the x-ray ; Y horizontal + towards the GMT door; Z vertical + towards the ceilling)
    # 'XYcam' plot (xcam, ycam) Xmas
    # 'xytest' plot pour test orientation cam
    # '2thetachi' plot (2theta, chi)
    # 'directionq' plot (longitude, latitude) de q
    whatplot = '2thetachi'
    # detection geometry:
    # 'Z>0' top camera 'Y>0' side + 'Y<0' side - 'X<0' backreflection  'X>0' transmission
    kf_direction = 'Z>0'  # 'Z>0' 'Y>0' 'Y<0' 'X<0'

    # mat_UO2_He_103_1 = GT.OrientMatrix_fromGL(filename = 'UO2_He_103_1L.dat')
    mat_Id = np.eye(3)
    # mymat1 = fromelemangles_toMatrix([  1.73284101e-01 ,  3.89021368e-03 ,  4.53663032e+01])
    # mymat2 = fromelemangles_toMatrix([  0. ,  35.19 ,45.])
    # mattest= [[-0.72419551 , 0.09344237, -0.6832345 ], [-0.59410999 , 0.41294652,  0.69029595], [ 0.3465515 ,  0.90488673, -0.24714785]]
    # #mat_Cu = GT.OrientMatrix_fromGL(filename = 'matrixfromopenGL_311_c2.dat')
    # mat_Cu = np.eye(3)
    # matfromHough = fromelemangles_toMatrix([85., 68., 76.])
    # matfromHough = fromEULERangles_toMatrix([42., 13., 20.])
    grain1 = [mat_Id, 'dia', dict_Rot['Si_001'], 'Si']
    # grain2 = [mat_Id, 'dia', mymat2,'Si']
    # #grainUO2_103 = [vecteurref, [1, 1, 1],mat_UO2_He_103_1,'UO2']
    # grainHough = [mat_Id, 'fcc', matfromHough,'UO2']
    # graintest = [mat_Id, 'fcc', mattest,'UO2']

    # GRAIN Must Be Prepared !!! see CP.Prepare_Grain() or fill before dict_Material

    starting_time = time.time()

    # vec and indices of spots Z > 0 (vectors [x, y, z] array, miller indices [H, K, L] array)
    vecind = getLaueSpots(CST_ENERGYKEV / emax,
                           CST_ENERGYKEV / emin,
                           [grain1],
                            linestowrite,
                            fileOK=0,
                            fastcompute=0,
                            kf_direction=kf_direction)
    pre_filename = 'totobill'

    # print "vecind",vecind
    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)
    oncam_sansh = filterLaueSpots(vecind,
                                fileOK=1,
                                fastcompute=0,
                                kf_direction=kf_direction,
                                detectordistance=DEFAULT_DETECTOR_DISTANCE,
                                HarmonicsRemoval=1,
                                linestowrite=linestowrite)
    # print "oncam_sansh",oncam_sansh

    # plot and save and write file of simulated Data
    # third input is a list whose each element corresponds to one grain list of spots object

    # loadind experimental data:
    data_filename = 'Ge_test.cor'
    data_2theta, data_chi, data_x, data_y, data_I = IOLT.readfile_cor(data_filename)

    # ----------  Time consumption information
    finishing_time = time.time()
    duration = finishing_time - starting_time
    print("Time duration for computation %.2f sec." % duration)

    Plot_Laue(emin, emax, oncam_sansh, data_2theta, data_chi, data_filename,
                            removeharmonics=1,
                            kf_direction=kf_direction,
                            Plot_Data=1,
                            Display_label=0,
                            What_to_plot=whatplot,
                            saveplotOK=0,
                            WriteLogFile=1,
                            linestowrite=linestowrite)

    # logbook file edition
    IOLT.writefile_log(output_logfile_name=pre_filename + '.log',
                 linestowrite=linestowrite)

    return CreateData_(oncam_sansh, outputname='tototable', pickledOK=0), \
            oncam_sansh


def test_speed():
    # ----------  Time consumption information
    Starting_time = time.time()

    Array2thetachi, Oncam = test_simulation()


#--- ------------  MAIN
if __name__ == "__main__":

    emin = 5
    emax = 15
    kf_direction = 'Z>0'
    data_filename = 'Ge_test.cor'  # experimental data

    Id = np.eye(3)

    if 0:
        grainSi = CP.Prepare_Grain('Si', OrientMatrix=dict_Rot['Si_001'])
        linestowrite = [[""]]

        grainSi[2] = GT.fromEULERangles_toMatrix([20., 10., 50.])

        print("\n*******************\nSimulation with fastcompute = 0")
        vecind = getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grainSi],
                                                linestowrite,
                                                fileOK=0,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                verbose=1)

        print("nb of spots with harmonics %d\n\n" % len(vecind[0][0]))

        print("\n*********\nSimulation with fastcompute = 1")

        vecindfast = getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, [grainSi],
                                                1,
                                                fileOK=0,
                                                fastcompute=1,
                                                kf_direction=kf_direction,
                                                verbose=1)


        print("nb of spots with harmonics fast method", len(vecindfast[0][0]))


        print("\n*******************\n --------  Harmonics removal -----------\n")

        oncam_sansh00 = filterLaueSpots(vecind,
                                                fileOK=1,
                                                fastcompute=0,
                                                kf_direction=kf_direction,
                                                HarmonicsRemoval=1,
                                                linestowrite=linestowrite)

        print("after harm removal 00", len(oncam_sansh00[0]))

        oncam_sansh01 = filterLaueSpots(vecind,
                                                fileOK=1,
                                                fastcompute=1,
                                                kf_direction=kf_direction,
                                                HarmonicsRemoval=1,
                                                linestowrite=linestowrite)

        print("after harm removal 01", len(oncam_sansh01[0]))

        oncam_sansh11 = filterLaueSpots(vecindfast,
                                                fileOK=1,
                                                fastcompute=1,
                                                kf_direction=kf_direction,
                                                HarmonicsRemoval=1,
                                                linestowrite=linestowrite)

        print("after harm removal 11", len(oncam_sansh11[0]))


        # ------------------------
        print("\n*****************************\n rax compute\n")
        grainSi[2] = GT.fromEULERangles_toMatrix([20., 10., 50.])

        res = get_2thetaChi_withoutHarmonics(grainSi, 5, 15)

    if 1:  # some grains example
        inittime = time.time()
        
        emax = 50
        emin = 5
        
        elem1 = 'Ti'
        elem2 = 'Cu'

        kf_direction = 'Z>0'

        BmatTi = CP.calc_B_RR(dict_Materials['Ti'][1])

        BmatCu = CP.calc_B_RR(dict_Materials['Cu'][1])

        Umat = dict_Rot['OrientSurf111']

        grainTi = [BmatTi, 'no', Umat, 'Ti']

        grainCu = [BmatCu, 'fcc', Id, 'Cu']
        grainCu__ = [BmatCu, 'no', Id, 'Cu']

        dict_Materials['Cu2'] = ['Cu2', [5, 3.6, 3.6, 90, 90, 90], 'fcc']
        BmatCu2 = CP.calc_B_RR(dict_Materials['Cu2'][1])

        dict_Materials['Cu3'] = ['Cu3', [Id, Umat, Id, BmatCu2], 'fcc']  # Da, U, Dc, B

        grainCu2 = CP.Prepare_Grain('Cu2', OrientMatrix=Id)
        grainCu3 = CP.Prepare_Grain('Cu3', OrientMatrix='ihjiĥiohio')

        grainSi1 = CP.Prepare_Grain('Cu', OrientMatrix=Id)
        grainSi2 = CP.Prepare_Grain('Si', OrientMatrix=dict_Rot['OrientSurf111'])
        
        grainAl2o3 = CP.Prepare_Grain('Al2O3', OrientMatrix=Id)

        # grains= [grainSi,grainCu3]
        grains = [grainSi1, grainCu3, grainAl2o3]

        print(grains)

        simulatepattern(grains, emin, emax,
                        kf_direction, data_filename, Plot_Data=0, verbose=1,
                        HarmonicsRemoval=0)
        
        finaltime = time.time()
        
        print('computation time is ', finaltime - inittime)

    if 0:
        # strain study: ---------------------------------------
        dict_Materials['Cu2'] = ['Cu2', [5, 3.6, 3.6, 90, 90, 90], 'fcc']
        BmatCu2 = CP.calc_B_RR(dict_Materials['Cu2'][1])

        Umat = dict_Rot['OrientSurf111']
        # Da, U, Dc, B
        dict_Materials['Cu3'] = ['Cu3', [Id, Umat, Id , BmatCu2], 'fcc']

        grains = []
        for k in range(-5, 6):
            dict_Materials['Cu_%d' % k] = dict_Materials['Cu3']
            # set material label
            dict_Materials['Cu_%d' % k][0] = 'Cu_%d' % k
            # set deformation in absolute space
            dict_Materials['Cu_%d' % k][1][0] = [[1 + 0.01 * k, 0, 0],
                                               [0, 1, 0],
                                                [0, 0, 1.]]
            grains.append(CP.Prepare_Grain('Cu_%d' % k))

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                        Plot_Data=0)
        # -----------------------------------------------

    if 0:

        Bmat = dict_Vect['543_909075']
        Umat = dict_Rot['OrientSurf001']
        Dc = [[1.02, 0.01, 0],
              [-.01, 0.98, 0.005],
                [0.001, -0.02, 1.01]]
        Dc = [[1.00, 0.00, 0],
              [-.00, 1.00, 0.000],
            [0.000, -0.00, 1.03]]

        dict_Materials['mycell'] = ['mycell', [Id, Umat, Id, Bmat], 'fcc']
        dict_Materials['mycell_strained'] = ['mycell_strained',
                                            [Id, Umat, Dc, Bmat],
                                            'fcc']

        mygrain = CP.Prepare_Grain('mycell')
        mygrain_s = CP.Prepare_Grain('mycell_strained')

        grains = [mygrain, mygrain_s]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                        Plot_Data=0)

    if 0:

        Bmat = dict_Vect['543_909075']
        Umat = dict_Rot['mat311c1']
        Dc = dict_Vect['shear4']

        dict_Materials['mycell_s'] = ['mycell_s', [Id, Umat, Dc, Bmat], 'fcc']
        dict_Materials['mycell'] = ['mycell', [Id, Umat, Id, Bmat], 'fcc']

        mygrain_s = CP.Prepare_Grain('mycell_s')
        mygrain = CP.Prepare_Grain('mycell')

        grains = [mygrain_s]


        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1)

    if 0:
        emin = 5
        emax = 22
        kf_direction = 'X>0'  # transmission
        kf_direction = 'Z>0'  # reflection

        ResolutionAngstrom = 2.

        # overwriting dict_Materials['smallpro']
        dict_Materials['smallpro'] = ['smallpro',
                                        [20, 4.8, 49, 90, 90, 90],
                                        'no']
        mygrain = CP.Prepare_Grain('smallpro', OrientMatrix=dict_Rot['mat311c1'])
        mygrain = CP.Prepare_Grain('smallpro', OrientMatrix=Id)

        grains = [mygrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1, detectordistance=69, ResolutionAngstrom=ResolutionAngstrom)

    if 0:
        emin = 5
        emax = 30
        kf_direction = 'X>0'  # transmission
#        kf_direction = 'Z>0' # reflection

        ResolutionAngstrom = False

        # overwriting dict_Materials['smallpro']

        mat111alongx = np.dot(GT.matRot([1, 0, 0], -45.), GT.matRot([0, 0, 1], 45.))

        print("mat111alongx", mat111alongx)

        matmono = np.dot(GT.matRot([1, 0, 0], -1), mat111alongx)

        mygrain = CP.Prepare_Grain('Cu', OrientMatrix=matmono)

        grains = [mygrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1, detectordistance=10,
                            ResolutionAngstrom=ResolutionAngstrom)

    if 0:
        emin = 8
        emax = 25
        kf_direction = 'X<0'  # back reflection
#        kf_direction = 'Z>0' # reflection

        ResolutionAngstrom = False

        # overwriting dict_Materials['smallpro']





        matmono = np.dot(GT.matRot([0, 0, 1], 20), np.eye(3))

        mygrain = CP.Prepare_Grain('Si', OrientMatrix=matmono)

        grains = [mygrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1, detectordistance=55,
                            ResolutionAngstrom=ResolutionAngstrom,
                            Display_label=0)

    if 0:
        emin = 8
        emax = 25
        kf_direction = 'X<0'  # back reflection
#        kf_direction = 'Z>0' # reflection

        ResolutionAngstrom = False

        # overwriting dict_Materials['smallpro']




        matmother = np.dot(GT.matRot([0, 0, 1], -5), np.eye(3))
        matmisorient = np.dot(GT.matRot([-1, 1, 1], .1), matmother)

        maingrain = CP.Prepare_Grain('Si', OrientMatrix=matmother)

        mygrain = CP.Prepare_Grain('Si', OrientMatrix=matmisorient)

        grains = [maingrain, mygrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1, detectordistance=55,
                            ResolutionAngstrom=ResolutionAngstrom,
                            Display_label=0)

    if 0:
        emin = 5
        emax = 20
        kf_direction = [90, 45]
#        kf_direction = 'Z>0' # reflection

        ResolutionAngstrom = False

        # overwriting dict_Materials['smallpro']

        matmother = np.dot(GT.matRot([0, 0, 1], 0), np.eye(3))
#        matmisorient = np.dot(GT.matRot([-1, 1, 1], .1), matmother)

        maingrain = CP.Prepare_Grain('Si', OrientMatrix=matmother)

        grains = [maingrain, maingrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1, detectordistance=70,
                            ResolutionAngstrom=ResolutionAngstrom,
                            Display_label=0)

    if 0:
        emin = 5
        emax = 20
#        kf_direction = [0, 0]
        kf_direction = 'X>0'  #

        ResolutionAngstrom = False

        # overwriting dict_Materials['smallpro']

        matmother = np.dot(GT.matRot([0, 0, 1], 0), np.eye(3))
#        matmisorient = np.dot(GT.matRot([-1, 1, 1], .1), matmother)

        maingrain = CP.Prepare_Grain('Si', OrientMatrix=matmother)

        grains = [maingrain, maingrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1, detectordistance=70,
                            ResolutionAngstrom=ResolutionAngstrom,
                            Display_label=0)



    if 0:
        emin = 5
        emax = 22

        Detpos = 0  # 1 = 'top' 0 = 'trans'

        if Detpos == 1:
            # on top
            kf_direction = 'Z>0'  # reflection
            Detdist = 70  # mm
        elif Detpos == 0:
            # transmission
            kf_direction = 'X>0'  # transmission
            Detdist = 100  # mm

        ResolutionAngstrom = None

        # overwriting dict_Materials['smallpro']
        dict_Materials['smallpro'] = ['smallpro', [20, 4.8, 49, 90, 90, 90], 'no']
        mygrain = CP.Prepare_Grain('smallpro', OrientMatrix=dict_Rot['mat311c1'])
        mygrain = CP.Prepare_Grain('smallpro', OrientMatrix=Id)
        ResolutionAngstrom = 2.


        # mygrain = Prepare_Grain('Cu', OrientMatrix=Id)
        # ResolutionAngstrom = None

        grains = [mygrain]

        simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1,
                            detectordistance=Detdist,
                            ResolutionAngstrom=ResolutionAngstrom)

    if 0:
        emin = 5
        emax = 22

        Detpos = 0  # 1 = 'top' 0 = 'trans'

        if Detpos == 1:
            # on top
            kf_direction = 'Z>0'  # reflection
            Detdist = 70  # mm
        elif Detpos == 0:
            # transmission
            kf_direction = 'X>0'  # transmission
            Detdist = 100.  # mm

        ResolutionAngstrom = None

        # overwriting dict_Materials['smallpro']
        dict_Materials['smallpro'] = ['smallpro', [20, 4.8, 49, 90, 90, 90], 'no']
        ResolutionAngstrom = 2.

        for ori in list(dict_Rot.keys())[2:3]:
            mygrain = CP.Prepare_Grain('smallpro', OrientMatrix=dict_Rot[ori])

            grains = [mygrain]

            mydata = simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1,
                            detectordistance=Detdist,
                            ResolutionAngstrom=ResolutionAngstrom)


        ard = np.array(mydata[0])

        import pylab as pp

        pp.figure(1)
        pp.scatter(ard[:, 0], ard[:, 1])

        xyd = np.array([[elem.Xcam, elem.Ycam] for elem in mydata[1][0]])

        pp.figure(2)
        pp.scatter(xyd[:, 0], xyd[:, 1])

        calib = [100., 1024., 1024., 90, 0.]

        SIGN_OF_GAMMA = 1
        xyd_fromfind2 = LTGeo.calc_xycam_from2thetachi(ard[:, 0], ard[:, 1], calib,
                             verbose=0,
                             pixelsize=165. / 2048,
                             dim=(2048, 2048),
                             signgam=SIGN_OF_GAMMA,
                             kf_direction=kf_direction)

        X, Y, theta = xyd_fromfind2


        pp.scatter(X, Y, c='r')

        pp.show()


    if 0:
        emin = 50
        emax = 120

        Detpos = 0  # 1 = 'top' 0 = 'trans'

        if Detpos == 1:
            # on top
            kf_direction = 'Z>0'  # reflection
            Detdist = 70  # mm
        elif Detpos == 0:
            # transmission
            kf_direction = 'X>0'  # transmission
            Detdist = 100.  # mm

        ResolutionAngstrom = .5


        for ori in list(dict_Rot.keys())[2:3]:
            mygrain = CP.Prepare_Grain('Ni', OrientMatrix=dict_Rot[ori])

            grains = [mygrain]

            mydata = simulatepattern(grains, emin, emax, kf_direction, data_filename,
                            Plot_Data=0, verbose=1,
                            detectordistance=Detdist,
                            ResolutionAngstrom=ResolutionAngstrom)


        ard = np.array(mydata[0])

        print("mydata", mydata)

        import pylab as pp

        pp.figure(1)
        pp.scatter(ard[:, 0], ard[:, 1])

        xyd = np.array([[elem.Xcam, elem.Ycam] for elem in mydata[1][0]])

        miller = [elem.Millers for elem in mydata[1][0]]

        print(miller)

        pp.figure(2)
        pp.scatter(xyd[:, 0], xyd[:, 1])

        calib = [105.624, 1017.50, 996.62, -0.027, -116.282]
        pixelsize = 0.048

        SIGN_OF_GAMMA = 1
        xyd_fromfind2 = LTGeo.calc_xycam_from2thetachi(ard[:, 0], ard[:, 1], calib,
                             verbose=0,
                             pixelsize=pixelsize,
                             dim=(2048, 2048),
                             kf_direction=kf_direction,
                             signgam=SIGN_OF_GAMMA)

        X, Y, theta = xyd_fromfind2


        pp.scatter(X, Y, c='r')

        pp.show()

        f = open('Ni_fake_transmission.dat', 'w')
        f.write('X Y I from simulation\n')
        for k in range(len(X)):
            f.write('%s %s %s %s %s %s %s %s %s %s %s\n' % (X[k], Y[k], 65000 * (1 - .8 * k / len(X)), 65000 * (1 - .8 * k / len(X)), 1.47, 1.81, 82.460, -1.07, 1.78, 624.74, 64740))
        f.write('# %s pixelsize %s' % (str(calib), pixelsize))
        f.close()

    if 0:  # Si 111 in transmission on ID15
        emin = 80
        emax = 100

        Detpos = 0  # 1 = 'top' 0 = 'trans'

        if Detpos == 1:
            # on top
            kf_direction = 'Z>0'  # reflection
            Detdist = 100  # mm
        elif Detpos == 0:
            # transmission
            kf_direction = 'X>0'  # transmission
            Detdist = 100.  # mm

        mattrans3 = [[0.998222982873332, 0.04592237603705288, -0.037973831023250665],
                     [0.0036229001244100726, 0.5893105150537007, 0.8078985031808321],
                     [0.05947899678171664, -0.8066004291012069, 0.5880969279936651]]
        mygrain = CP.Prepare_Grain('Si', OrientMatrix=dict_Rot['mat111alongx'])
        mygrain = CP.Prepare_Grain('Si', OrientMatrix=mattrans3)

        grains = [mygrain]

        mydata = simulatepattern(grains, emin, emax, kf_direction, data_filename,
                                 PlotLaueDiagram=0,
                        Plot_Data=0, verbose=1,
                        detectordistance=Detdist,
                        ResolutionAngstrom=None)

        # two theta chi
        ard = np.array(mydata[0])

        print("ard", ard)

        import pylab as pp

        pp.figure(1)
        pp.scatter(ard[:, 0], ard[:, 1])

        # pixel X Y position with default camera settings ...
#         xyd = np.array([[elem.Xcam, elem.Ycam] for elem in mydata[1][0]])
#
#         pp.figure(2)
#         pp.scatter(xyd[:, 0], xyd[:, 1])

        calib = [105., 1024., 1024., 0., 0.]

        SIGN_OF_GAMMA = 1
        xyd_fromfind2 = LTGeo.calc_xycam_from2thetachi(ard[:, 0], ard[:, 1], calib,
                             verbose=0,
                             pixelsize=0.048,
                             dim=(2048, 2048),
                             signgam=SIGN_OF_GAMMA,
                             kf_direction=kf_direction)

        X, Y, theta = xyd_fromfind2

        intensity = np.ones_like(X)

        IOLT.writefile_Peaklist("ID15transSi111", np.array([X, Y, intensity,
                                                                   intensity,
                                                                   intensity,
                                                                   intensity,
                                                                   intensity,
                                                                   intensity,
                                                                   intensity,
                                                                   intensity,
                                                                   intensity]).T)

        pp.figure(3)
        pp.scatter(X, Y, c='r')

        pp.show()



