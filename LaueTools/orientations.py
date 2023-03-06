from __future__ import print_function
"""
module of lauetools project

JS Micha May 2019
"""
import sys
#import os

# sys.path.insert(0, os.path.abspath('../..'))
# print('sys.path in CrystalParameters', sys.path)

import numpy as np

if sys.version_info.major == 3:
    from . import dict_LaueTools as DictLT
    from . import generaltools as GT
    from . import LaueGeometry as LaueGeom
    from . CrystalParameters import matstarlab_to_matstarlabOND, norme
    from . import findorient as FO

else:
    import generaltools as GT
    import dict_LaueTools as DictLT
    import LaueGeometry as LaueGeom
    import findorient as FO

DEG = np.pi / 180.0

NORMAL_TO_SAMPLE_AXIS = np.array([-np.sin(40.0 * DEG), 0, np.cos(40.0 * DEG)])


# --- -------------------  Mapping visualization
def calc_Euler_angles(mat3x3):
    r"""
    Calculates unique 3 euler angles representation of mat3x3

    .. note::
        from O Robach

    .. todo:: to be placed in generaltools
    """
    # mat3x3 = matrix "minimized" in LT lab frame
    # see F2TC.find_lowest_Euler_Angles_matrix
    # phi 0, theta 1, psi 2

    mat = GT.matline_to_mat3x3(
        matstarlab_to_matstarlabOND(GT.mat3x3_to_matline(mat3x3)))

    RAD = 180.0 / np.pi

    euler = np.zeros(3, float)
    euler[1] = RAD * np.arccos(mat[2, 2])

    if np.abs(np.abs(mat[2, 2]) - 1.0) < 1e-5:
        # if theta is zero, phi+psi is defined, if theta is pi, phi-psi is defined */
        # put psi = 0 and calculate phi */
        # psi */
        euler[2] = 0.0
        # phi */
        euler[0] = RAD * np.arccos(mat[0, 0])
        if mat[0, 1] < 0.0:
            euler[0] = -euler[0]
    else:
        # psi */
        toto = np.sqrt(1 - mat[2, 2] * mat[2, 2])  # sin theta - >0 */
        euler[2] = RAD * np.arccos(mat[1, 2] / toto)
        # phi */
        euler[0] = RAD * np.arccos(-mat[2, 1] / toto)
        if mat[2, 0] < 0.0:
            euler[0] = 360.0 - euler[0]

        # print "Euler angles phi theta psi (deg)"
        # print euler.round(decimals = 3)

        return euler


def myRGB_3(mat):
    r"""
    propose a RGB (red green blue) vector to represent a matrix

    .. note::
        from O Robach

    .. todo:: to be placed in generaltools
    """
    allpermu = DictLT.OpSymArray
    allpermudet1 = np.array([kkk for kkk in allpermu if np.linalg.det(kkk) == 1.0])

    vec = LaueGeom.vec_normalTosurface(mat)
    # allvec = np.dot(allpermudet1,vec)
    # allposvec = allvec[np.all(allvec>=0,axis=1)]
    # # pos of row in allposvec where last element is the higher element of each row
    # hpos = np.where(np.argmax(allposvec, axis=1)==2)[0][0]
    # #print hpos
    # return allposvec[hpos].tolist()

    myframe = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    invf = np.linalg.inv(myframe)

    # vec expressed in myframe basis
    vec_in_frame = np.dot(invf, vec)

    # find permutation that lead to 3 positive components
    # TODO: do the permutation before expressing the myframe basis ??
    allvec = np.dot(allpermudet1, vec_in_frame)
    allposvec = allvec[np.all(allvec >= 0, axis=1)]

    if len(np.shape(allposvec)) > 1:
        vecmaxfirstcolumn = np.argmax(allposvec[:, 0])
        allposvec = allposvec[vecmaxfirstcolumn]

    return allposvec


def getMisorientation(
    mat, refAxis=NORMAL_TO_SAMPLE_AXIS, followVector=np.array([0, 0, 1])):
    r"""
    compute an angle of a reflection / an axis

    # default axis normal to surface tilted by  40 degrees from horizontal

    .. todo:: to be placed in generaltools

    """

    # angle between mat.G* and refAxis
    #    norm = np.sqrt(np.dot(followVector, followVector))
    rotGstar = np.dot(mat, followVector)

    norm_rotGstar = np.sqrt(np.dot(rotGstar, rotGstar))

    angle = np.arccos(np.dot(rotGstar, refAxis) / (norm_rotGstar)) / DEG

    return angle


def Matrix_to_RGB_2(mat):
    r"""
    use unique representation of orientation matrix with its euler angles

    .. note::
        from O Robach

    .. todo:: to be placed in generaltools
    """
    normalisation_angles = np.array([180.0, 360, 180.0])
    # normalisation_angles = np.ones(3)

    return calc_Euler_angles(FO.find_lowest_Euler_Angles_matrix(mat)[0]
                                                ) / normalisation_angles + np.array([0.1, 0.1, 0.1])
