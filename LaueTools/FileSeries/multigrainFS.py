# -*- coding: cp1252 -*-

import site

from time import time, asctime
import numpy as np
from numpy import *
import matplotlib.pylab as p
from scipy import *  # pour charger les donnees

# import scipy.io.array_import # pour charger les donnees
import os, sys

# print sys.path

sys.path.append("..")

import LaueGeometry as F2TC
import readmccd as rmccd
import LaueAutoAnalysis as LAA
import indexingAnglesLUT as INDEX

import findorient as FindO

import CrystalParameters as CP
from generaltools import norme_vec as norme
import generaltools as GT
import IOLaueTools as IOLT

import dict_LaueTools as DictLT
from mosaic import ImshowFrameNew, ImshowFrame_Scalar, ImshowFrame

from numpy.linalg import inv

# set invisible parameters for serial_peak_search, serial_index_refine_multigrain
import param_multigrain as PAR

# carriage return string
# TODO test MAC OS
if sys.platform.startswith("lin"):
    # for unix
    cr_string = "\r\n"
else:
    # for windows
    cr_string = "\n"

stock1 = 0
stock2 = 0
stock3 = 0
# warning : matstarlab (matline) is in OR's lab frame, matLT (mat3x3) in LaueTools lab frame
# zLT = zOR, xLT = yOR, xOR = - yLT
# incident beam along +y in OR lab frame, along +x in LT lab frame

pi1 = math.pi

p.rcParams["lines.markersize"] = 12
p.rcParams["lines.linewidth"] = 1.5
p.rcParams["font.size"] = 12
p.rcParams["axes.labelsize"] = "large"
p.rcParams["figure.subplot.bottom"] = 0.2
p.rcParams["figure.subplot.left"] = 0.2
p.rcParams["xtick.major.size"] = 8
p.rcParams["xtick.major.pad"] = 8
p.rcParams["ytick.major.size"] = 8
p.rcParams["ytick.major.pad"] = 8

#########


def pc():
    p.close("all")


def print_calib(calib):

    # modif 04 Mar 2010 xbet et xgam en degres au lieu de radians

    calib3 = zeros(5, float)
    calib3 = calib * 1.0

    print(
        calib3[0].round(decimals=3),
        calib3[1].round(decimals=2),
        calib3[2].round(decimals=2),
        calib3[3].round(decimals=3),
        calib3[4].round(decimals=3),
    )

    return 0


# calcul reseau reciproque


def dlat_to_rlat(dlat):

    rlat = rand(6)
    """
    # Compute reciprocal lattice parameters. The convention used is that
    # a[i]*b[j] = d[ij], i.e. no 2PI's in reciprocal lattice.
    """

    # compute volume of real lattice cell

    volume = (
        dlat[0]
        * dlat[1]
        * dlat[2]
        * sqrt(
            1
            + 2 * cos(dlat[3]) * cos(dlat[4]) * cos(dlat[5])
            - cos(dlat[3]) * cos(dlat[3])
            - cos(dlat[4]) * cos(dlat[4])
            - cos(dlat[5]) * cos(dlat[5])
        )
    )

    # compute reciprocal lattice parameters

    rlat[0] = dlat[1] * dlat[2] * sin(dlat[3]) / volume
    rlat[1] = dlat[0] * dlat[2] * sin(dlat[4]) / volume
    rlat[2] = dlat[0] * dlat[1] * sin(dlat[5]) / volume
    rlat[3] = arccos(
        (cos(dlat[4]) * cos(dlat[5]) - cos(dlat[3])) / (sin(dlat[4]) * sin(dlat[5]))
    )
    rlat[4] = arccos(
        (cos(dlat[3]) * cos(dlat[5]) - cos(dlat[4])) / (sin(dlat[3]) * sin(dlat[5]))
    )
    rlat[5] = arccos(
        (cos(dlat[3]) * cos(dlat[4]) - cos(dlat[5])) / (sin(dlat[3]) * sin(dlat[4]))
    )

    return rlat


def mat_to_rlat(matstarlab):

    rlat = zeros(6, float)

    astarlab = matstarlab[0:3]
    bstarlab = matstarlab[3:6]
    cstarlab = matstarlab[6:9]
    rlat[0] = norme(astarlab)
    rlat[1] = norme(bstarlab)
    rlat[2] = norme(cstarlab)
    rlat[5] = arccos(inner(astarlab, bstarlab) / (rlat[0] * rlat[1]))
    rlat[4] = arccos(inner(cstarlab, astarlab) / (rlat[2] * rlat[0]))
    rlat[3] = arccos(inner(bstarlab, cstarlab) / (rlat[1] * rlat[2]))

    # print "rlat = ",rlat

    return rlat


def rad_to_deg(dlat):

    dlatdeg = hstack((dlat[0:3], dlat[3:6] * 180.0 / math.pi))
    return dlatdeg


def deg_to_rad(dlat):

    dlatrad = hstack((dlat[0:3], dlat[3:6] * math.pi / 180.0))
    return dlatrad


def dlat_to_dlatr(dlat):

    dlatr = zeros(6)
    for i in range(0, 3):
        dlatr[i] = dlat[i] / dlat[0]
    for i in range(3, 6):
        dlatr[i] = dlat[i]

    return dlatr


def epsline_to_epsmat(epsline):  # 29May13
    """
    # deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12
    """
    epsmat = identity(3, float)

    epsmat[0, 0] = epsline[0]
    epsmat[1, 1] = epsline[1]
    epsmat[2, 2] = epsline[2]

    epsmat[1, 2] = epsline[3]
    epsmat[0, 2] = epsline[4]
    epsmat[0, 1] = epsline[5]

    epsmat[2, 1] = epsline[3]
    epsmat[2, 0] = epsline[4]
    epsmat[1, 0] = epsline[5]

    return epsmat


def epsmat_to_epsline(epsmat):  # 29May13
    """
    # deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12
    """
    epsline = zeros(6, float)

    epsline[0] = epsmat[0, 0]
    epsline[1] = epsmat[1, 1]
    epsline[2] = epsmat[2, 2]

    epsline[3] = epsmat[1, 2]
    epsline[4] = epsmat[0, 2]
    epsline[5] = epsmat[0, 1]

    return epsline


def dlat_to_Bstar(dlat):  # 29May13

    """
        # Xcart = Bstar*Xcrist_rec
        # changement de coordonnees pour le vecteur X entre 
        # le repere de la maille reciproque Rcrist_rec
        # et le repere OND Rcart associe a Rcrist_rec
        # dlat  direct lattice parameters
        # rlat  reciprocal lattice parameters
        # en radians
    """

    Bstar = zeros((3, 3), dtype=float)
    rlat = dlat_to_rlat(dlat)

    Bstar[0, 0] = rlat[0]
    Bstar[0, 1] = rlat[1] * cos(rlat[5])
    Bstar[1, 1] = rlat[1] * sin(rlat[5])
    Bstar[0, 2] = rlat[2] * cos(rlat[4])
    Bstar[1, 2] = -rlat[2] * sin(rlat[4]) * cos(dlat[3])
    Bstar[2, 2] = 1.0 / dlat[2]

    return Bstar


def rlat_to_Bstar(rlat):  # 29May13

    """
        # Xcart = Bstar*Xcrist_rec
        # changement de coordonnees pour le vecteur X entre 
        # le repere de la maille reciproque Rcrist_rec
        # et le repere OND Rcart associe a Rcrist_rec
        # rlat  reciprocal lattice parameters
        # dlat  direct lattice parameters
        # en radians
        """
    Bstar = zeros((3, 3), dtype=float)
    dlat = dlat_to_rlat(rlat)

    Bstar[0, 0] = rlat[0]
    Bstar[0, 1] = rlat[1] * cos(rlat[5])
    Bstar[1, 1] = rlat[1] * sin(rlat[5])
    Bstar[0, 2] = rlat[2] * cos(rlat[4])
    Bstar[1, 2] = -rlat[2] * sin(rlat[4]) * cos(dlat[3])
    Bstar[2, 2] = 1.0 / dlat[2]

    return Bstar


def matstarlab_to_deviatoric_strain_crystal(
    matstarlab, version=2, reference_element_for_lattice_parameters="Ge"
):
    # 29May13
    """
    # version = 1 : simplified calculation for initially cubic unit cell
    # version = 2 : full calculation for unit cell with any symmetry
    # formulas from Tamura's XMAS chapter in Barabash 2013 book
    # = same as Chung and Ice 1999 (but clearer explanation)
    # needs angles in radians
    # dlat[0] can be any value, not necessarily 1.0
    """

    rlat = mat_to_rlat(matstarlab)
    dlat = dlat_to_rlat(rlat)
    # print "dlat = ", dlat  # - np.array([1.,1.,1.,math.pi/2.,math.pi/2.,math.pi/2.])
    dlatrdeg = rad_to_deg(dlat_to_dlatr(dlat))

    if version == 1:  # only for initially cubic unit cell

        epsp = zeros(6, float)

        tr3 = (dlat[0] + dlat[1] + dlat[2]) / 3.0

        epsp[0] = (dlat[0] - tr3) * 1000.0 / dlat[0]
        epsp[1] = (dlat[1] - tr3) * 1000.0 / dlat[0]
        epsp[2] = (dlat[2] - tr3) * 1000.0 / dlat[0]

        epsp[3] = -1000.0 * (dlat[3] - math.pi / 2.0) / 2.0
        epsp[4] = -1000.0 * (dlat[4] - math.pi / 2.0) / 2.0
        epsp[5] = -1000.0 * (dlat[5] - math.pi / 2.0) / 2.0

    elif version == 2:  # for any symmetry of unit cell

        # reference lattice parameters with angles in degrees
        dlat0_deg = np.array(
            DictLT.dict_Materials[reference_element_for_lattice_parameters][1],
            dtype=float,
        )
        dlat0 = deg_to_rad(dlat0_deg)

        # print dlat0.round(decimals = 4)
        # print dlat.round(decimals = 4)

        # matstarlab construite pour avoir norme(astar) = 1
        Bdir0 = rlat_to_Bstar(dlat0)
        Bdir0 = Bdir0 / dlat0[0]

        Bdir = rlat_to_Bstar(dlat)
        Bdir = Bdir / dlat[0]

        # print Bdir0.round(decimals=4)
        # print Bdir.round(decimals=4)

        # Rmat = inv(Bdir) et T = dot(inv(Rmat), Rmat0)

        Tmat = dot(Bdir, inv(Bdir0))

        eps1 = 0.5 * (Tmat + Tmat.transpose()) - eye(3)
        # print eps1.round(decimals=2)
        # print np.trace(eps1)

        # la normalisation du premier vecteur de Bdir a 1
        # ne donne pas le meme volume pour les deux mailles
        # => il faut soustraire la partie dilatation

        epsp1 = 1000.0 * (eps1 - (np.trace(eps1) / 3.0) * eye(3))
        # print epsp1.round(decimals=1)
        # print np.trace(epsp1)

        epsp = epsmat_to_epsline(epsp1)

    # print "deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12  *1e3 \n", epsp.round(decimals=1)

    # print "dlatrdeg = \n", dlatrdeg

    return (epsp, dlatrdeg)


def read_stiffness_file(filestf):  # 29May13
    """
    # units = 1e11 N/m2
    # dans les fichiers stf de XMAS les cij sont en 1e11 N/m2 
    # voir http://www.docstoc.com/docs/45454109/Elastic-Constants-of-Single-Crystals
    """
    c_tensor = loadtxt(filestf, skiprows=1)
    c_tensor = np.array(c_tensor, dtype=float)
    print(filestf)
    print(shape(c_tensor))
    print("stiffness tensor C, 1e11 N/m2 units")
    print(c_tensor)

    return c_tensor


def deviatoric_strain_crystal_to_stress_crystal(c_tensor, eps_crystal_line):  # 29May13

    """
    Voigt Notation 
    C = 6x6 matrix 
    (C is not a second rank tensor => rule of passing from one frame to an other does not apply to  C)
    (notation Pedersen is required cf mail Consonni to build a correct tensor from C)
    sigma = dot (C, gamma)
    gam1 = eps11
    gam2 = eps22
    gam3 = eps33
    gam4 = 2 eps23
    gam5 = 2 eps13
    gam6 = 2 eps12    
    
    cij in 1e11 N/m2 = 100 GPa units
    epsij in 1e-3 units
    sigma in 0.1 GPa = 100 MPa units
    """
    fact1 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    gam_cryst = multiply(eps_crystal_line, fact1)
    sigma_crystal_line = dot(c_tensor, gam_cryst)
    # print eps_crystal_line
    # print gam_cryst
    # print sigma_crystal_line

    return sigma_crystal_line


def glide_systems_to_schmid_tensors(
    n_ref=array([1.0, 1.0, 1.0]), b_ref=array([1.0, -1.0, 0.0]), verbose=0
):
    # 29May13
    """
    only for cubic systems
    coordonnees cartesiennes dans le repere OND obtenu en orthonormalisant le repere cristal
    cf thesis Gael Daveau p 16
    """
    nop = 24

    allop = DictLT.OpSymArray
    indgoodop = array(
        [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            25,
            27,
            29,
            31,
            33,
            35,
            37,
            39,
            41,
            43,
            45,
            47,
        ]
    )
    goodop = allop[indgoodop]

    hkl_2 = row_stack((n_ref, b_ref))
    normehkl = zeros(2, float)

    uqref = zeros((2, 3), float)
    for i in range(2):
        normehkl[i] = norme(hkl_2[i, :])
        uqref[i, :] = hkl_2[i, :] / normehkl[i]

    uqall = zeros((2, nop, 3), float)
    for j in range(2):  # j=0 : n, j=1 : b
        for k in range(nop):
            uqall[j, k, :] = dot(goodop[k], uqref[j, :])

    isdouble = zeros(nop, int)
    for k in range(nop):
        # print "k = ", k
        un_ref = uqall[0, k, :]
        ub_ref = uqall[1, k, :]
        for j in range(k + 1, nop):
            # print "j = ", j
            dun = norme(cross(un_ref, uqall[0, j, :]))
            dub = norme(cross(ub_ref, uqall[1, j, :]))
            dun_dub = dun + dub
            if dun_dub < 0.01:
                isdouble[j] = 1
    print(isdouble)

    ind0 = where(isdouble == 0)
    print(ind0[0])
    uqall = uqall[:, ind0[0], :]

    nop2 = 12

    st1 = zeros((nop2, 3, 3), float)
    print("n b schmid_tensor [line1, line2, line3]")
    for k in range(nop2):
        un_colonne = uqall[0, k, :].reshape(3, 1)
        ub_ligne = uqall[1, k, :].reshape(1, 3)
        st1[k, :, :] = dot(un_colonne, ub_ligne)
        # print  uqall[0,k,:]*normehkl[0], uqall[1,k,:]*normehkl[1], st1[k,:,:].reshape(1,9).round(decimals=3)

    return st1


def deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(
    sigma_crystal_line, schmid_tensors
):
    # 29May13
    nop2 = shape(schmid_tensors)[0]
    sigma_crystal_3x3 = epsline_to_epsmat(sigma_crystal_line)
    tau_all = zeros(nop2, float)
    for k in range(nop2):
        tau_all[k] = (multiply(schmid_tensors[k], sigma_crystal_3x3)).sum()

    # print tau_all
    return tau_all


def deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal_line):
    # 29May13
    sig = sigma_crystal_line * 1.0
    von_mises = (
        (sig[0] - sig[1]) * (sig[0] - sig[1])
        + (sig[1] - sig[2]) * (sig[1] - sig[2])
        + (sig[2] - sig[0]) * (sig[2] - sig[0])
        + 6.0 * (sig[3] * sig[3] + sig[4] * sig[4] + sig[5] * sig[5])
    )
    von_mises = von_mises / 2.0
    von_mises = sqrt(von_mises)
    return von_mises


def uflab_to_xycam(uflab, calib):

    # modif 04 Mar 2010 xbet xgam en degres au lieu de radians

    # XMAS PCIF6 changer le signe de xgam
    # laptop OR garder le meme signe pour xgam

    detect = calib[0] * 1.0
    xcen = calib[1] * 1.0
    ycen = calib[2] * 1.0
    xbet = calib[3] * 1.0
    xgam = calib[4] * 1.0

    # #    print "Correcting the data according to the parameters"
    # #    print "xcam, ycam in XMAS convention"
    # #
    # #    print "detect in mm" , detect
    # #    print "xcen in pixels" , xcen
    # #    print "ycen in pixels" , ycen
    # #    print "xbet in degrees" , xbet
    # #    print "xgam in degrees" , xgam

    PI = math.pi

    xbetrad = xbet * PI / 180.0
    xgamrad = xgam * PI / 180.0

    cosbeta = cos(PI / 2.0 - xbetrad)
    sinbeta = sin(PI / 2.0 - xbetrad)
    cosgam = cos(-xgamrad)
    singam = sin(-xgamrad)

    IOlab = detect * array([0.0, cosbeta, sinbeta])

    unlab = IOlab / norme(IOlab)

    normeIMlab = detect / inner(uflab, unlab)

    # uflab1 = array([-uflab[0],uflab[1],uflab[2]])

    uflab1 = uflab * 1.0

    IMlab = normeIMlab * uflab1

    OMlab = IMlab - IOlab

    xca0 = OMlab[0]
    yca0 = OMlab[1] / sinbeta

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    xcam = xcen + xcam1 * 2048.0 / 165.0
    ycam = ycen + ycam1 * 2048.0 / 165.0

    uflabyz = array([0.0, uflab1[1], uflab1[2]])
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab

    chi = (180.0 / PI) * arctan(uflab1[0] / norme(uflabyz))
    twicetheta = (180.0 / PI) * arccos(uflab1[1])
    th0 = twicetheta / 2.0

    # print "2theta, theta, chi en deg", twicetheta , chi, twicetheta/2.0
    # print "xcam, ycam = ", xcam, ycam

    return (xcam, ycam, th0)


def uqlab_to_xycam(uqlab, calib):

    uflab = zeros(3, float)
    xycam = zeros(2, float)

    uilab = array([0.0, 1.0, 0.0])
    sintheta = -inner(uqlab, uilab)
    uflab = uilab + 2 * sintheta * uqlab

    xycam[0], xycam[1], th0 = uflab_to_xycam(uflab, calib)

    return xycam


def matstarlab_to_matstarlabOND(matstarlab):

    astar1 = matstarlab[:3]
    bstar1 = matstarlab[3:6]
    cstar1 = matstarlab[6:]

    astar0 = astar1 / norme(astar1)
    cstar0 = cross(astar0, bstar1)
    cstar0 = cstar0 / norme(cstar0)
    bstar0 = cross(cstar0, astar0)

    matstarlabOND = hstack((astar0, bstar0, cstar0)).transpose()

    # print matstarlabOND

    return matstarlabOND


def matstarlab_to_matdirlab3x3(matstarlab):  # 29May13

    rlat = mat_to_rlat(matstarlab)
    # print rlat
    vol = CP.vol_cell(rlat, angles_in_deg=0)

    astar1 = matstarlab[:3]
    bstar1 = matstarlab[3:6]
    cstar1 = matstarlab[6:]

    adir = cross(bstar1, cstar1) / vol
    bdir = cross(cstar1, astar1) / vol
    cdir = cross(astar1, bstar1) / vol

    matdirlab3x3 = column_stack((adir, bdir, cdir))

    # print " matdirlab3x3 =\n", matdirlab3x3.round(decimals=6)

    return (matdirlab3x3, rlat)


def matstarlab_to_matdirONDsample3x3(matstarlab, omega0=40.0):  # 29May13

    # uc unit cell
    # dir direct
    # uc_dir_OND : cartesian frame obtained by orthonormalizing direct unit cell

    matdirlab3x3, rlat = matstarlab_to_matdirlab3x3(matstarlab)
    # dir_bmatrix = uc_dir on uc_dir_OND

    dir_bmatrix = dlat_to_Bstar(rlat)

    # matdirONDlab3x3 = uc_dir_OND on lab

    matdirONDlab3x3 = dot(matdirlab3x3, np.linalg.inv(dir_bmatrix))

    omega = omega0 * math.pi / 180.0

    # rotation de -omega autour de l'axe x pour repasser dans Rsample
    matrot = array(
        [[1.0, 0.0, 0.0], [0.0, cos(omega), sin(omega)], [0.0, -sin(omega), cos(omega)]]
    )

    # matdirONDsample3x3 = uc_dir_OND on sample
    # rsample = matdirONDsample3x3 * ruc_dir_OND

    matdirONDsample3x3 = dot(matrot, matdirONDlab3x3)

    return matdirONDsample3x3


def transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(
    matstarlab, tensor_crystal_line, omega0=40.0
):
    # 29May13
    """
    start from stress or strain tensor
    as 6 coord vector
    """
    tensor_crystal_3x3 = epsline_to_epsmat(tensor_crystal_line)

    matdirONDsample3x3 = matstarlab_to_matdirONDsample3x3(matstarlab, omega0=omega0)

    # changement de base pour tenseur d'ordre 2

    toto = dot(tensor_crystal_3x3, matdirONDsample3x3.transpose())

    tensor_sample_3x3 = dot(matdirONDsample3x3, toto)

    tensor_sample_line = epsmat_to_epsline(tensor_sample_3x3)

    return tensor_sample_line


def matstarlab_to_deviatoric_strain_sample(
    matstarlab,
    omega0=40.0,
    version=2,
    returnmore=False,
    reference_element_for_lattice_parameters="Ge",
):
    # 29May13
    epsp_crystal, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarlab,
        version=version,
        reference_element_for_lattice_parameters=reference_element_for_lattice_parameters,
    )

    epsp_sample = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(
        matstarlab, epsp_crystal, omega0=omega0
    )
    if returnmore == False:
        return epsp_sample

    else:
        return (epsp_sample, epsp_crystal)  # add epsp_crystal


def read_all_grains_str(filestr, min_matLT=True):  # 29May13

    print("reading info from STR file : \n", filestr)
    # print "peak list, calibration, strained orientation matrix, deviations"
    # print "change sign of HKL's"
    f = open(filestr, "r")
    i = 0
    grainfound = 0
    calib = zeros(5, float)
    # x(exp)  y(exp)  h  k  l xdev  ydev  energy  dspace  intens   integr
    # xwidth   ywidth  tilt  rfactor   pearson  xcentroid  ycentroid
    # 0 x(exp)    # 1 y(exp)
    # 2 3 4 h  k  l
    # 5 6 xdev  ydev
    # 7 energy
    # 8 dspace
    # 9 10 intens   integr
    grainnumstr = 1
    is_good_grain = []
    try:
        for line in f:
            i = i + 1
            if line[0] == "N":
                # print line
                # print line.split()
                gnumtot = np.array((line.split())[-1], dtype=int)
                print("Number of grains : ", gnumtot)
                linestartspot = zeros(gnumtot, int)
                lineendspot = zeros(gnumtot, int)
                linemat = zeros(gnumtot, int)
                dlatstr = zeros((gnumtot, 6), float)
                dev_str = zeros((gnumtot, 3), float)

            if line[0] == "G":
                gnumloc = np.array((line.split())[2], dtype=int)
                # print gnumloc
                gnum = grainnumstr - 1
                if gnumloc == grainnumstr:
                    print("grain ", grainnumstr)
                    # print  " indexed peaks list starts at line : ", i
                    linestartspot[gnum] = i + 1
                    grainfound = 1

                    list1 = []
            if grainfound == 1:
                # if i == linestart :
                # print line.rstrip("\n")
                if line[0:3] == "lat":
                    # print "lattice parameters at line : ", i
                    dlatstr[gnum, :] = np.array(line[18:].split(), dtype=float)
                    print("lattice parameters : \n", dlatstr[gnum, :])
                    # print "indexed peaks list ends at line = ", i
                    if norme(dlatstr[gnum, :]) > 0.1:
                        is_good_grain.append(1)
                    else:
                        is_good_grain.append(0)

                    lineendspot[gnum] = i - 1
                if gnum == 0:
                    if line[0:2] == "dd":
                        # print "calib starts at line : ", i
                        calib[:3] = np.array(line[17:].split(), dtype=float)
                    if line[0:4] == "xbet":
                        # print "calib line 2 at line = ", i
                        calib[3:] = np.array(line[11:].split(), dtype=float)
                if line[0:4] == "dev1":
                    dev_str[gnum, :] = np.array(line.split()[3:], dtype=float)
                    print("deviations : \n", dev_str[gnum, :])
                if line[15:17] == "a*":
                    # print "matrix starts at line : ", i
                    linemat[gnum] = int(i)
                    grainfound = 0
                    if gnum < gnumtot:
                        grainnumstr = grainnumstr + 1
    finally:
        linetot = i
        f.seek(0)

    print("calib :")
    print_calib(calib)

    matstarlab = zeros((gnumtot, 9), float)
    npeaks = zeros(gnumtot, int)
    print("linemat = ", linemat)

    is_good_grain = np.array(is_good_grain, dtype=int)
    print("is_good_grain =", is_good_grain)

    # f = open(filestr, 'r')
    listspot = []
    listmat = []
    gnum = 0
    i = 0
    print("grain number : ", gnum)
    try:
        for line in f:
            if (i >= linestartspot[gnum]) & (i < lineendspot[gnum]):
                # print line
                listspot.append(
                    line.rstrip("\n").replace("[", "").replace("]", "").split()
                )

            if (i >= linemat[gnum]) & (i < linemat[gnum] + 3):
                # print line
                listmat.append(
                    line.rstrip("\n").replace("[", "").replace("]", "").split()
                )

            if ((i > linemat[gnum] + 2) & (line[0] == "G")) | (i == linetot - 1):
                if is_good_grain[gnum] == 1:
                    matstr = np.array(listmat, dtype=float)
                    satocrs = matstr.transpose()
                    print("strained orientation matrix (satocrs) = \n", satocrs)
                    matstarlab[gnum, :] = F2TC.matxmas_to_matstarlab(satocrs, calib)
                    data_str = np.array(listspot, dtype=float)
                    print("first / last lines of data_str :")
                    print(data_str[0, :2])
                    print(data_str[-1, :2])
                    data_str[:, 2:5] = -data_str[:, 2:5]

                    if min_matLT == True:
                        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(
                            matstarlab[gnum, :]
                        )
                        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(
                            matLT3x3
                        )
                        matLT3x3 = matmin * 1.0
                        matstarlab[gnum, :] = F2TC.matstarlabLaueTools_to_matstarlabOR(
                            matLT3x3
                        )
                        # transfo des HKL a verifier
                        hklmin = dot(
                            transfmat, data_str[:, 2:5].transpose()
                        ).transpose()
                        data_str[:, 2:5] = hklmin

                    # data_str_sorted = sort_peaks_decreasing_int(data_str, 10)
                    # data_str = data_str_sorted * 1.0

                    if gnum == 0:
                        data_str_all = data_str * 1.0
                    else:
                        data_str_all = vstack((data_str_all, data_str))

                    # print data_str
                    npeaks[gnum] = shape(data_str)[0]
                    print("number of indexed peaks :", npeaks[gnum])
                    print("first peak : ", data_str[0, 2:5])
                    print("last peak : ", data_str[-1, 2:5])

                    gnum = gnum + 1
                    listspot = []
                    listmat = []
                    if i < linetot - 1:
                        print("grain number : ", gnum)
                else:
                    print("bad grain")
                    gnum = gnum + 1
                    listspot = []
                    listmat = []
                    if i < linetot - 1:
                        print("grain number : ", gnum)

            i = i + 1
    finally:
        f.close()

    # print "return(data_str, satocrs, calib, dev_str)"
    # print "data_str :  xy(exp) 0:2 hkl 2:5 xydev 5:7 energy 7 dspacing 8  intens 9 integr 10"

    print("gnumtot = ", gnumtot)
    print("npeaks = ", npeaks)
    print("shape(data_str_all) =", shape(data_str_all))
    # print data_str_all[:,2]

    return (data_str_all[:, :11], matstarlab, calib, dev_str, npeaks)


def read_xmas_txt_file_from_seq_file(
    filexmas,
    read_all_cols="yes",
    list_column_names=[
        "GRAININDICE",
        "ASIMAGEINDICE",
        "STRNINDEX",
        "ASSPIXDEV",
        "ASVONMISES",
        "RSS",
    ],
):
    # 29May13
    nameline = "ASIMAGEINDICE GRAININDICE ASXSTAGE ASYSTAGE ASDD ASXCENT ASYCENT ASXBET ASXGAM ASXALFD ASXBETD\
        ASNINDEX ASDEV1 ASDEV2 ASPIXDEV STRNINDEX ASSTRAUXX ASSTRAUXY ASSTRAUXZ ASSTRAUYY ASSTRAUYZ ASSTRAUZZ\
        ASSTRAXXX ASSTRAXXY ASSTRAXXZ ASSTRAXYY ASSTRAXYZ ASSTRAXZZ\
        ASSTREUXX ASSTREUXY ASSTREUXZ ASSTREUYY ASSTREUYZ ASSTREUZZ\
        ASSTREXXX ASSTREXXY ASSTREXXZ ASSTREXYY ASSTREXYZ ASSTREXZZ ASSDEV1 ASSDEV2\
        ASSPIXDEV ASVONMISES ASAX ASAY ASAZ ASBX ASBY ASBZ ASCX ASCY ASCZ\
        ASASX ASASY ASASZ ASBSX ASBSY ASBSZ ASCSX ASCSY ASCSZ ORUND ORUNR ORSND ORSNR ORUID ORUIR ORSID ORSIR\
        ASTOTINTENS2 ASTOTINTENS ASTOTALINT ASTOTALINT2 IZERO AVINTPIXEL AVINTPIXSUB AVERAGEINT\
        FLUOA FLUOB FLUOC FLUOD FLUOE EXPOSURE RSS DELTA1 DELTA2 AVERPEAKWIDTH TOTSTRAXX TOTSTRAYY\
        TOTSTRAZZ TOTSTREXX TOTSTREYY TOTSTREZZ DELTA3 DELTA4 DELTA5\
        UAX UAY UAZ UBX UBY UBZ UCX UCY UCZ UASX UASY UASZ UBSX UBSY UBSZ UCSX UCSY UCSZ\
        ORSIRXZ ORSIRYZ DEFDX DEFDY DEFDZ COMPDATA5 UASU UASV UASW UBSU UBSV UBSW UCSU UCSV UCSW\
        SASU SASV SASW SBSU SBSV SBSW SCSU SCSV SCSW QUATW QUATX QUATY QUATZ RODR1 RODR2 RODR3 ROTANGLE\
        EULERPHI EULERTHETA EULERPSI MISANGLE MISVECX MISVECY MISVECZ COSANGR COSANGG COSANGB UNUSED19 UNUSED20"

    print("reading summary file")
    print("first two lines :")
    f = open(filexmas, "r")
    i = 0
    try:
        for line in f:
            if i == 0:
                npts = line.rstrip("  \n")
            if i == 1:
                nameline1 = line.rstrip("\n")
            i = i + 1
            if i > 2:
                break
    finally:
        f.close()

    print(npts)
    print(nameline1)
    listname = nameline1.split()

    data_sum = loadtxt(filexmas, skiprows=2)

    if read_all_cols == "yes":
        print("shape(data_sum) = ", shape(data_sum))
        return (data_sum, listname, npts)

    else:
        print(len(listname))
        ncol = len(list_column_names)

        ind0 = zeros(ncol, int)

        for i in range(ncol):
            ind0[i] = listname.index(list_column_names[i])

        print(ind0)

        data_sum_select_col = data_sum[:, ind0]

        print(shape(data_sum))
        print(shape(data_sum_select_col))
        print(filesum)
        print(list_column_names)
        print(data_sum_select_col[:5, :])

        return (data_sum_select_col, list_column_names, npts)


if 0:  # find symmetry operations with det > 1
    allop = DictLT.OpSymArray
    ind0 = []
    for j in range(48):
        det1 = np.linalg.det(allop[j])
        print(j, det1)
        if det1 > 0.0:
            ind0.append(j)

    print(ind0)
    ind0 = np.array(ind0, int)
    jksqld

    allop = DictLT.OpSymArray
    indgoodop = array(
        [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            25,
            27,
            29,
            31,
            33,
            35,
            37,
            39,
            41,
            43,
            45,
            47,
        ]
    )
    goodop = allop[indgoodop]
    # print shape(goodop)


def matstarlab_to_matstarsample3x3(matstarlab, omega=40.0):

    matstarlab3x3 = GT.matline_to_mat3x3(matstarlab)
    omega = omega * math.pi / 180.0  # -xbetrad/2.0
    # rotation de -omega autour de l'axe x pour repasser dans Rsample
    matrot = array(
        [[1.0, 0.0, 0.0], [0.0, cos(omega), sin(omega)], [0.0, -sin(omega), cos(omega)]]
    )
    matstarsample3x3 = dot(matrot, matstarlab3x3)

    # print  "matstarsample3x3 =\n" , matstarsample3x3.round(decimals=6)
    return matstarsample3x3


def matstarsample3x3_to_matstarlab(matstarsample3x3, omega=40.0):  # 29May13

    # rotation de -omega autour de l'axe x pour repasser dans Rsample
    omega = omega * math.pi / 180.0
    matrot = array(
        [[1.0, 0.0, 0.0], [0.0, cos(omega), -sin(omega)], [0.0, sin(omega), cos(omega)]]
    )
    matstarlab3x3 = dot(matrot, matstarsample3x3)

    matstarlab = GT.mat3x3_to_matline(matstarlab3x3)
    # print  "matstarsample3x3 =\n" , matstarsample3x3.round(decimals=6)

    return matstarlab


def fromMatrix_toQuat(matrix):
    # print "matrix \n", matrix
    mat = array(
        [
            matrix[0, 0],
            matrix[1, 0],
            matrix[2, 0],
            0.0,
            matrix[0, 1],
            matrix[1, 1],
            matrix[2, 1],
            0.0,
            matrix[0, 2],
            matrix[1, 2],
            matrix[2, 2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )

    # print shape(mat)

    toto = 1.0 + mat[0] + mat[5] + mat[10]
    if toto < 0.0:
        print("warning : negative toto in fromMatrix_toQuat", toto)
        toto = abs(toto)
        # print matrix

    soso = sqrt(toto) * 2.0
    qx = (mat[9] - mat[6]) / soso
    qy = (mat[2] - mat[8]) / soso
    qz = (mat[4] - mat[1]) / soso
    qw = 0.25 * soso
    return np.array([qx, qy, qz, qw])


def fromQuat_to_vecangle(quat):
    """ from quat = [vec,scalar] = [sin angle/2 (unitvec(x,y,z)), cos angle/2]
    gives unitvec and angle of rotation around unitvec
    """
    normvectpart = math.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
    # print "nor",normvectpart
    angle = math.acos(quat[3] / normvectpart) * 2.0  # in radians
    unitvec = array(quat[:3]) / math.sin(angle / 2.0) / normvectpart
    return unitvec, angle


def fromvecangle_to_Quat(unitvec, angle):
    # angle in radians
    quat = zeros(4, float)
    quat[3] = cos(angle / 2.0)
    quat[:3] = sin(angle / 2.0) * unitvec
    return quat


def from_axis_vecangle_to_mat(vlab, ang1):
    # ang1 in degrees
    vlab = vlab / norme(vlab)
    ang1 = ang1 * math.pi / 180.0
    rotv = array(
        [[0.0, -vlab[2], vlab[1]], [vlab[2], 0.0, -vlab[0]], [-vlab[1], vlab[0], 0.0]]
    )
    matrot = cos(ang1) * eye(3) + (1 - cos(ang1)) * outer(vlab, vlab) + sin(ang1) * rotv
    # print "matrot = ", matrot
    return matrot


if 0:  # variables needed in calc_cosines_first_stereo_triangle

    omega = 40.0
    omega = omega * math.pi / 180.0
    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = array(
        [[1.0, 0.0, 0.0], [0.0, cos(omega), -sin(omega)], [0.0, sin(omega), cos(omega)]]
    )

    hkl_3 = array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    uqref_cr = zeros((3, 3), float)
    # uqref_cr 3 vecteurs 001 101 111 en colonnes
    for i in range(3):
        uqref_cr[:, i] = hkl_3[i, :] / norme(hkl_3[i, :])

    print("cosines between extremities of first triangle : ")
    cos01 = inner(uqref_cr[:, 0], uqref_cr[:, 1])
    cos02 = inner(uqref_cr[:, 0], uqref_cr[:, 2])
    cos12 = inner(uqref_cr[:, 1], uqref_cr[:, 2])

    print(round(cos01, 3), round(cos02, 3), round(cos12, 3))

    cos0 = min(cos01, cos02)
    cos1 = min(cos01, cos12)
    cos2 = min(cos02, cos12)

    print("minimum cos with 001 101 and 111 : ")
    print(round(cos0, 3), round(cos1, 3), round(cos2, 3))

    # vectors normal to frontier planes of stereographic triangle

    uqn_b = cross(uqref_cr[:, 0], uqref_cr[:, 1])
    uqn_b = uqn_b / norme(uqn_b)
    uqn_g = cross(uqref_cr[:, 0], uqref_cr[:, 2])
    uqn_g = uqn_g / norme(uqn_g)
    uqn_r = cross(uqref_cr[:, 1], uqref_cr[:, 2])
    uqn_r = uqn_r / norme(uqn_r)


def calc_cosines_first_stereo_triangle(
    matstarlab, axis_pole_sample
):  # , matrot, uqref_cr) : # return_matrix = "yes", return_cosines = "no") : #, xyz_sample_azimut):

    # modified 15Nov12 : return RGBx RGBz instead of cosines
    # dependance sous-entendue vis a vis des variables :
    # matrot, uqref_cr, uqn_r , uqn_g, uqn_b
    # 06Aug12
    # cf calc_matrix_for_stereo2 dans multigrain_OR.py laptop, apres correction
    # avec operations de symetrie de det = 1 et de det = -1
    # axe axis_pole ou -axis_pole dans le premier triangle stereo
    # entre (001), (101), (111)

    # garder la liste de variables globales

    # pas encore verifie si deformations coherentes dans Rsample entre avant / apres opsym

    verbose = 0

    allop = DictLT.OpSymArray
    nop = 48
    # opsym avec det>1
    indgoodop = array(
        [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            25,
            27,
            29,
            31,
            33,
            35,
            37,
            39,
            41,
            43,
            45,
            47,
        ]
    )

    # ------------------ xyz-sample convention OR
    omega = 40.0
    omega = omega * math.pi / 180.0
    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = array(
        [[1.0, 0.0, 0.0], [0.0, cos(omega), -sin(omega)], [0.0, sin(omega), cos(omega)]]
    )

    hkl_3 = array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    uqref_cr = zeros((3, 3), float)
    # uqref_cr 3 vecteurs 001 101 111 en colonnes
    for i in range(3):
        uqref_cr[:, i] = hkl_3[i, :] / norme(hkl_3[i, :])

    cos01 = inner(uqref_cr[:, 0], uqref_cr[:, 1])
    cos02 = inner(uqref_cr[:, 0], uqref_cr[:, 2])
    cos12 = inner(uqref_cr[:, 1], uqref_cr[:, 2])

    #         print "cosines between extremities of first triangle : "
    #         print round(cos01, 3), round(cos02, 3), round(cos12, 3)

    cos0 = min(cos01, cos02)
    cos1 = min(cos01, cos12)
    cos2 = min(cos02, cos12)

    #         print "minimum cos with 001 101 and 111 : "
    #         print round(cos0, 3), round(cos1, 3), round(cos2, 3)

    # vectors normal to frontier planes of stereographic triangle

    uqn_b = cross(uqref_cr[:, 0], uqref_cr[:, 1])
    uqn_b = uqn_b / norme(uqn_b)
    uqn_g = cross(uqref_cr[:, 0], uqref_cr[:, 2])
    uqn_g = uqn_g / norme(uqn_g)
    uqn_r = cross(uqref_cr[:, 1], uqref_cr[:, 2])
    uqn_r = uqn_r / norme(uqn_r)
    # --   end of preliminary calculations

    upole_sample = axis_pole_sample / norme(axis_pole_sample)
    upole_lab = dot(matrot, upole_sample)
    # print "pole axis - sample coord : ", upole_sample
    # print "pole axis - lab coord : ", upole_lab.round(decimals=3)

    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)

    mat = GT.matline_to_mat3x3(matstarlabOND)

    Bstar = rlat_to_Bstar(mat_to_rlat(matstarlab))
    matdef = GT.matline_to_mat3x3(matstarlab)

    # #        # test
    # #        matdef = GT.matline_to_mat3x3(matstarlab)
    # #        print "matdef"
    # #        print matdef.round(decimals=4)
    # #        print "matdef recalc"
    # #        print dot(mat,Bstar).round(decimals=4)

    cosangall = zeros((2 * nop, 3), float)
    ranknum = list(range(2 * nop))
    opsym = 2 * list(range(nop))
    # print opsym
    matk_lab = zeros((2 * nop, 3, 3), float)

    for k in range(nop):
        matk_lab[k] = dot(mat, allop[k])
        # retour a un triedre direct si indirect
        if k not in indgoodop:
            # print "yoho"
            matk_lab[k, :, 2] = -matk_lab[k, :, 2]
        uqrefk_lab = dot(matk_lab[k], uqref_cr)
        for j in range(3):
            cosangall[k, j] = inner(upole_lab, uqrefk_lab[:, j])
            cosangall[k + 48, j] = inner(-upole_lab, uqrefk_lab[:, j])

    data1 = column_stack((opsym, cosangall, ranknum))
    # print shape(data1)

    # priorites 001 101 111
    np1 = 1
    np2 = 2
    np3 = 3

    # print "opsym  cos001 cos101 cos111 ranknum"
    # print data1[:,1:4].round(decimals = 3)
    data1_sorted = sort_list_decreasing_column(data1, np1)

    # print data1_sorted.round(decimals = 3)

    ind1 = where(abs(data1_sorted[:, np1] - data1_sorted[0, np1]) < 1e-3)

    # print ind1

    data2_sorted = sort_list_decreasing_column(data1_sorted[ind1[0], :], np2)

    # print data2_sorted.round(decimals = 3)

    ind2 = where(abs(data2_sorted[:, np2] - data2_sorted[0, np2]) < 1e-3)

    # print ind2

    data3_sorted = sort_list_decreasing_column(data2_sorted[ind2[0], :], np3)

    # print data3_sorted.round(decimals = 3)

    ind3 = where(abs(data3_sorted[:, np3] - data3_sorted[0, np3]) < 1e-3)

    # print ind3

    # print "initial matrix abcstar_on_xyzlab"

    # print GT.matline_to_mat3x3(matstarlab).round(decimals=4)

    # print "pole axis in Rsample"
    # print axis_pole_sample

    # print "new matrix abcstar_on_xyzlab with polar axis in first stereo triangle :"
    opsymres = []
    rankres = []
    for i in ind3[0]:
        op1 = int(round(data3_sorted[i, 0], 1))
        rank1 = int(round(data3_sorted[i, -1], 1))
        opsymres.append(op1)
        rankres.append(rank1)
        # print "opsym =" , op1
        # print matk_lab[op1].round(decimals=4)

    opsymres = np.array(opsymres, dtype=int)
    # print opsymres

    if 0 in opsymres:
        op1 = 0
    else:
        op1 = opsymres[0]

    matONDnew = matk_lab[op1]
    opres = dot(np.linalg.inv(mat), matONDnew)
    # print "opres \n", opres.round(decimals=1)
    # print "det(opres)", np.linalg.det(opres)
    toto = dot(opres.transpose(), Bstar)
    Bstarnew = dot(toto, opres)
    matdef2 = dot(matONDnew, Bstarnew)
    matstarlabnew = GT.mat3x3_to_matline(matdef2)

    abcstar_on_xyzsample = matstarlab_to_matstarsample3x3(matstarlabnew)
    xyzsample_on_abcstar = np.linalg.inv(abcstar_on_xyzsample)

    transfmat = np.linalg.inv((dot(np.linalg.inv(matdef), matdef2).round(decimals=1)))

    # print "transfmat \n", transfmat

    # print "final : xyzsample_on_abcstar"
    # print xyzsample_on_abcstar.round(decimals=4)

    # print "matrix"
    # print "initial" , matstarlab.round(decimals=4)
    # print "final ", matstarlabnew.round(decimals=4)

    if verbose:
        print(
            "op sym , rank, cos ",
            op1,
            ranknum[rankres[0]],
            cosangall[rankres[0]].round(decimals=3),
        )

    if ranknum[rankres[0]] < 48:
        cos_end = cosangall[rankres[0]]
    else:
        cos_end = -cosangall[rankres[0]]

    cos_end_abs = abs(cos_end)
    if (cos_end_abs[0] < cos0) | (cos_end_abs[1] < cos1) | (cos_end_abs[2] < cos2):
        print("problem : pole axis not in first triangle")
        exit()
    # else : print "cosines OK"

    # print "new crystal coordinates of axis_pole :"
    uq = np.dot(xyzsample_on_abcstar, upole_sample)
    if verbose:
        print("uq :")
        print(uq.round(decimals=4))
        print("uqref_cr :")
        print(uqref_cr.round(decimals=3))

    # RGB coordinates
    rgb_pole = zeros(3, float)
    # blue : distance in q space between M tq OM = uq et le plan 001 101 passant par O
    rgb_pole[2] = abs(inner(uq, uqn_b)) / abs(inner(uqref_cr[:, 2], uqn_b))
    rgb_pole[1] = abs(inner(uq, uqn_g)) / abs(inner(uqref_cr[:, 1], uqn_g))
    rgb_pole[0] = abs(inner(uq, uqn_r)) / abs(inner(uqref_cr[:, 0], uqn_r))

    # convention OR
    rgb_pole = rgb_pole / max(rgb_pole)
    # convention Tamura
    # rgb_pole = rgb_pole / norme(rgb_pole)

    # print "rgb_pole :"
    # print rgb_pole

    return (matstarlabnew, transfmat, rgb_pole)


def sort_list_decreasing_column(data_str, colnum):

    # print "sort list, decreasing values of column ", colnum

    npics = shape(data_str)[0]
    # print "nlist = ", npics
    index2 = zeros(npics, int)

    index1 = argsort(data_str[:, colnum])
    for i in range(npics):
        index2[i] = index1[npics - i - 1]
    # print "index2 =", index2
    data_str2 = data_str[index2]

    return data_str2


def hkl_to_xystereo(
    hkl0, polar_axis=[0.0, 0.0, 1.0], down_axis=[1.0, 0.0, 0.0], return_more=None
):

    uq = hkl0 / norme(hkl0)
    uz = polar_axis / norme(polar_axis)
    udown = down_axis / norme(down_axis)
    uright = cross(uz, udown)
    uqz = inner(uq, uz)
    change_sign = 1
    if uqz < 0.0:
        print("warning : uq.uz < 0 in hkl_to_xystereo, change sign of uq")
        uqz = -uqz
        uq = -uq
        if return_more != None:
            change_sign = -1
    qs = (uq - uqz * uz) / (1.0 + uqz)
    # print qs.round(decimals=3)
    # print norme(qs)
    qsxy = np.array([inner(qs, uright), -inner(qs, udown)])
    # print qsxy.round(decimals=3)

    if return_more == None:
        return qsxy
    else:
        return (qsxy, change_sign)


def remove_duplicates(data, dtol=0.01):
    ndat = shape(data)[0]
    nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(
        data, data, dxytol=dtol, verbose=1
    )
    ind1 = where(iscommon1 == 1)
    # duplicates give iscommon1 > 1
    datanew = data[ind1[0], :]
    return datanew


if 0:  # ## test 1  #29May13

    # mat3x3 = array([[1.0,0.01,0.0],[0.0,0.99995,0.0],[0.0,0.0,1.0]])
    # mat3x3 = array([[1.,0.01,0.02],[0.,1.,0.03],[0.,0.,1.]])
    mat3x3 = array([[1.0, 0.01, 0.02], [0.0, 1.05, 0.03], [0.0, 0.0, 1.03]])
    print("mat3x3")
    print(mat3x3)

    matstarlab = GT.mat3x3_to_matline(mat3x3)
    print("matstarlab")
    print(matstarlab)

    print("strain in crystal frame")
    print("version 1 = linearize")
    print("version 2 = use B matrix")

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarlab, version=1, reference_element_for_lattice_parameters="Ge"
    )
    print("version 1 :")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarlab, version=2, reference_element_for_lattice_parameters="Ge"
    )

    print("version 2 :")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab)
    print("matLT3x3")
    print(matLT3x3)

    matstarlabLT = GT.mat3x3_to_matline(matLT3x3)
    # print "matstarlabLT"
    # print matstarlabLT

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarlabLT, version=2, reference_element_for_lattice_parameters="Ge"
    )

    print("version 2 : using matstarlabLT")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    matstarsample3x3 = matstarlab_to_matstarsample3x3(matstarlab)
    print("matstarsample3x3")
    print(matstarsample3x3)

    matstarsample = GT.mat3x3_to_matline(matstarsample3x3)

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarsample, version=2, reference_element_for_lattice_parameters="Ge"
    )

    print("version 2 : using matstarsample")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    print("strain in sample frame")
    print("version 2 :")
    epsp_sample_v1 = matstarlab_to_deviatoric_strain_sample(
        matstarlab,
        omega0=40.0,
        version=2,
        reference_element_for_lattice_parameters="Ge",
    )
    print("deviatoric strain xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-4 units)")
    print((epsp_sample_v1 * 10.0).round(decimals=2))

    jklsqdjkl

if 0:  # ## test 2 #29May13
    mat3x3 = array([[0.01, 1.01, 0.02], [0.0, 0.03, 1.02], [1.03, 0.0, 0.0]])
    print("mat3x3")
    print(mat3x3)

    print("det(mat3x3)")
    print(np.linalg.det(mat3x3))

    matstarlab = GT.mat3x3_to_matline(mat3x3)
    print("matstarlab")
    print(matstarlab)

    matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(mat3x3)
    print("matmin")
    print(matmin)

    matstarlab2 = GT.mat3x3_to_matline(matmin)
    print(matstarlab2)
    print("matstarlab2")

    kdlmsqdsqd

if 0:  # ## test 3
    mat3x3 = array([[1.01, 0.01, 0.02], [0.0, 1.02, 0.03], [0.0, 0.0, 1.03]])
    mat3x3 = array([[0.01, 1.01, 0.02], [0.0, 0.03, 1.02], [1.03, 0.0, 0.0]])
    print("mat3x3")
    print(mat3x3)

    matstarlab = GT.mat3x3_to_matline(mat3x3)
    print("matstarlab")
    print(matstarlab)

    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
    print("matstarlabOND")
    print(matstarlabOND.round(decimals=6))

    mat3x3OND = GT.matline_to_mat3x3(matstarlabOND)
    print("mat3x3OND")
    print(mat3x3OND.round(decimals=6))

    klmk

##********************************************************************************************************************************


def test_index_refine(
    filedat,
    paramdetector_top,
    use_weights=False,
    proposed_matrix=None,
    check_grain_presence=None,
    paramtofit="strain",
    elem_label="UO2",
    grainnum=1,
    remove_sat=0,
    elim_worst_pixdev=1,
    maxpixdev=1.0,
    spot_index_central=[0, 1, 2, 3, 4, 5],
    nbmax_probed=20,
    energy_max=22,
    rough_tolangle=0.5,
    fine_tolangle=0.2,
    Nb_criterium=15,
    NBRP=1,
    mark_bad_spots=0,
    boolctrl="1" * 8,
    pixelsize=165.0 / 2048,
    dim=(2048, 2048),
    LUT=None,
):

    # 29May13
    """
    data treatment sequence:
    
    indexation
    strain refinement
    
    keep this : definition of parameters

    spot_index_central = [0,1,2,3,4,5]                   # central spot or list of spots
    nbmax_probed = 20              # 'Recognition spots set Size (RSSS): '
    
    energy_max = 22

    rough_tolangle = 0.5            # 'Dist. Recogn. Tol. Angle (deg)'
    fine_tolangle = 0.2             # 'Matching Tolerance Angle (deg)'
    Nb_criterium = 15                # 'Minimum Number Matched Spots: '
    NBRP = 1 # number of best result for each element of spot_index_central
          remove_sat = 0   # = 1 : keep saturated peaks (Ipixmax = 65565) for indexation but remove them for refinement
          elim_worst_pixdev = 1 : for 2nd refinement : eliminate spots with pixdev > maxpixdev after first refinement
          mark_bad_spots = 1 : for multigrain indexation, at step n+1 eliminate from the starting set all the spots
                                  more intense than the most intense indexed spot of step n
                                  (trying to exclude "grouped intense spots" from starting set)
                                  
    30Nov12 : add new column Etheor
"""
    SIGN_OF_GAMMA = 1

    dim1 = dim

    # MAR ou Roper
    # #    pixelsize = 165. / 2048
    # #    dim1 = (2048,2048)
    # VHR
    # pixelsize = 0.031
    # dim1 = (2594, 3764)

    # Automation of indexation + strain refinement
    calib = np.array(paramdetector_top, dtype=float)
    data_dat = np.loadtxt(filedat, skiprows=1)
    nspots = shape(data_dat)[0]

    numcolint = 3  # adapter au format du .dat
    data_dat = sort_peaks_decreasing_int(data_dat, numcolint)

    if remove_sat:
        col_Ipixmax = 5  # adapter au format du .dat
        saturation = 65535
        data_sat = zeros(nspots, int)
        data_Ipixmax = np.array(data_dat[:, col_Ipixmax], dtype=int)
        ind0 = where(data_Ipixmax == saturation)
        data_sat[ind0[0]] = 1
        print(data_sat)

    if mark_bad_spots:
        col_isbadspot = 4  # adapter au format du .dat
        data_isbadspot = np.array(data_dat[:, col_isbadspot], dtype=int)

    data_pixX = data_dat[:, 0] * 1.0
    data_pixY = data_dat[:, 1] * 1.0
    data_I = data_dat[:, numcolint] * 1.0

    data_twotheta, data_chi = F2TC.calc_uflab(
        data_pixX,
        data_pixY,
        calib,
        returnAngles=1,
        pixelsize=pixelsize,
        signgam=SIGN_OF_GAMMA,
    )

    data_theta = data_twotheta / 2.0

    if proposed_matrix == None:  # only for paramtofit = "strain"

        if paramtofit == "calib":
            print("need proposed_matrix to index before fitting calib")
            return 0

        if mark_bad_spots:
            ind1 = where(data_isbadspot == 0)
            data_theta1 = data_theta[ind1[0]]
            data_chi1 = data_chi[ind1[0]]
            data_x1 = data_pixX[ind1[0]]
            data_y1 = data_pixY[ind1[0]]
            print(shape(data_theta))
            print(shape(data_theta1))

        else:
            data_theta1 = data_theta * 1.0
            data_chi1 = data_chi * 1.0
            data_x1 = data_pixX * 1.0
            data_y1 = data_pixY * 1.0

        # #                istart = 0
        # #                angmean = 0.0
        # #
        # #                while (angmean < 18.0) :
        # #
        # #                        print istart
        # #                        listcouple = np.transpose(np.array([data_theta1[istart:istart+nbmax_probed], data_chi1[istart:istart+nbmax_probed]]))
        # #
        # #                        Tabledistance = INDEX.calculdist_from_thetachi(listcouple, listcouple)
        # #
        # #
        # #                        #print shape(listcouple)
        # #

        # #
        # #
        # #                        print angmean
        # #
        # #                        istart = istart + 1
        # #
        # #                istart = istart-1
        # #        data_theta1 = data_theta1[istart:]
        # #        data_chi1 = data_chi1[istart:]
        # #        data_x1 = data_x1[istart:]
        # #        data_y1 = data_y1[istart:]

        listcouple = np.transpose(
            np.array([data_theta1[:nbmax_probed], data_chi1[:nbmax_probed]])
        )

        Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        # classical indexation parameters:
        data = (2 * data_theta, data_chi, data_I)  # , filecor)

        print("rough_tolangle ", rough_tolangle)
        print("fine_tolangle ", fine_tolangle)

        # indexation procedure

        start_tab_distance = Tabledistance

        if LUT == None:
            n = 3
            latticeparams = DictLT.dict_Materials[elem_label][1]
            Bmatrix = CP.calc_B_RR(latticeparams)
            LUT = INDEX.build_AnglesLUT(Bmatrix, n)

        nLUT = 3
        detectorparameters = {}
        detectorparameters["kf_direction"] = "Z>0"
        detectorparameters["detectorparameters"] = calib
        detectorparameters["detectordiameter"] = 165.0
        detectorparameters["pixelsize"] = pixelsize
        detectorparameters["dim"] = dim

        for spotnum in spot_index_central:
            print(data_x1[spotnum], data_y1[spotnum])

            RES_onespot = INDEX.getOrientMatrix_from_onespot(
                spotnum,
                rough_tolangle,
                start_tab_distance,
                data[0],
                data[1],
                nLUT,
                Bmatrix,
                LUT=LUT,
                MatchingThresholdStop=99.999,
                key_material=elem_label,
                emax=energy_max,
                MatchingRate_Angle_Tol=fine_tolangle,
                verbose=0,
                detectorparameters=detectorparameters,
            )

            print("RES_onespot", RES_onespot)
            print(shape(RES_onespot))

            bestmat, matchingrate, latticeplanes_pairs, spots_pairs = RES_onespot

            if bestmat is not None and matchingrate >= 0:
                orientmatrix = bestmat
            else:
                raise ValueError("Orientation Matrix not found")

    #            bestmat, stats_res = INDEX.getOrientMatrices(
    #                                                        spotnum,
    #                                                        energy_max,
    #                                                        start_tab_distance,
    #                                                        data_theta1, data_chi1,
    #                                                        B = Bmatrix,
    #                                                        LUT = LUT,
    #                                                        rough_tolangle=rough_tolangle,
    #                                                        fine_tolangle=fine_tolangle,
    #                                                        Nb_criterium=Nb_criterium,
    #                                                        plot=0,
    #                                                        structure_label = elem_label,
    #                                                        nbbestplot=NBRP
    #                                                        )
    # #def getOrientMatrices(spot_index_central,
    # #                        energy_max,
    # #                        Tab_angl_dist,
    # #                        Theta, Chi,
    # #                        n=3, # up  to (332)
    # #                        B=None, # for cubic
    # #                        LUT=None,
    # #                        rough_tolangle=0.5,
    # #                        fine_tolangle=0.2,
    # #                        Nb_criterium=15,
    # #                        structure_label='',
    # #                        plot=0,
    # #                        nbbestplot=1,
    # #                        nbspots_plot='all', # nb exp spots to display if plot = 1
    # #                        addMatrix=None,
    # #                        verbose=1):
    #
    #            # bestmat contains: NBRP * len(spot_index_central) matrix
    #            print bestmat
    #            print stats_res
    #            print len(stats_res)
    #            if len(stats_res) == 0 :
    #                continue
    #            else :
    #                break
    #
    #        if len(stats_res) == 0 :
    #            return(0)
    #
    #        # Look for the very best matrix candidates: -----to be improved in readability ------------------------
    #        ar = []
    #        for elem in stats_res:
    #                ar.append([elem[0], -elem[2]])
    #
    #        tabstat = np.array(ar)#, dtype = [('x', '<i4'), ('y', '<i4')])
    #        rankmat = np.argsort(tabstat[:, 0])[::-1]#,order=('x','y'))
    #
    #        verybestmat = bestmat[rankmat[0]]
    #
    #        # -----------------------------------------------------------------------------------------------------
    #        # very best matrix
    #        orientmatrix = verybestmat

    else:
        orientmatrix = proposed_matrix
        # UBmat = proposed_matrix
        if paramtofit == "strain":
            data = (2.0 * data_theta, data_chi, data_I)  # , filecor)
        elif paramtofit == "calib":
            data = (2.0 * data_theta, data_chi, data_I, data_pixX, data_pixY)

    if check_grain_presence != None:

        # then resimulate to output miller indices
        emax_simul = 16
        veryclose_angletol = 0.2
        Bmatrix = eye(3)

        res1 = LAA.GetStrainOrient(
            orientmatrix,
            Bmatrix,
            elem_label,
            emax_simul,
            veryclose_angletol,
            paramdetector_top,
            SIGN_OF_GAMMA,
            data,
            addoutput=1,
        )

        if (res1 == 0) | (res1 == None):
            return 0
        else:
            UBmat, deviatoricstrain, RR, spotnum_i_pixdev, pixdev, nfit, spotnum_hkl = (
                res1
            )
    else:

        UBmat = proposed_matrix
        print(UBmat)

    emax_simul = energy_max
    veryclose_angletol = 0.1

    Bmatrix = eye(3)
    if paramtofit == "strain":

        print("start strain refinement")

        if remove_sat:
            saturated = data_sat
        else:
            saturated = None

        res2 = LAA.GetStrainOrient(
            UBmat,
            Bmatrix,
            elem_label,
            emax_simul,
            veryclose_angletol,
            paramdetector_top,
            SIGN_OF_GAMMA,
            data,
            addoutput=1,
            saturated=saturated,
            use_weights=use_weights,
            addoutput2=1,
        )

        if (res2 == 0) | (res2 == None):
            return 0
        else:
            UBmat_2, deviatoricstrain_2, RR_2, spotnum_i_pixdev_2, pixdev_2, nfit_2, spotnum_hkl_2, spotexpnum_spotsimnum_2, Intensity_list_2 = (
                res2
            )

        # print spotnum_i_pixdev_2
        # print spotexpnum_spotsimnum_2
        # print Intensity_list_2

        if elim_worst_pixdev:
            pixdevlist = spotnum_i_pixdev_2[:, 2]
            print(pixdevlist)
            ind0 = np.where(pixdevlist > maxpixdev)
            print(ind0[0])
            nbad = len(ind0[0])
            if nbad > 0:
                print("MAXPIXDEV = ", maxpixdev)
                print("found ", nbad, " peaks with pixdev larger than ", maxpixdev)
                ind1 = np.where(pixdevlist < maxpixdev)
                ngood = len(ind1[0])
                if ngood > 10:
                    # print spotexpnum_spotsimnum_2
                    # print spotnum_hkl_2
                    # print Intensity_list_2
                    print("fit again after removing bad peaks")

                    spotexpnum_spotsimnum_2 = spotexpnum_spotsimnum_2[ind1[0]]
                    spotnum_hkl_2 = (np.array(spotnum_hkl_2, dtype=int))[ind1[0]]
                    Intensity_list_2 = (np.array(Intensity_list_2, dtype=float))[
                        ind1[0]
                    ]
                    spotnum_hkl_2 = list(spotnum_hkl_2)
                    Intensity_list_2 = list(Intensity_list_2)

                    res_refinement = LAA.RefineUB(
                        spotexpnum_spotsimnum_2,
                        spotnum_hkl_2,
                        Intensity_list_2,
                        paramdetector_top,
                        data,
                        UBmat_2,
                        Bmatrix,
                        pixelsize=pixelsize,
                        dim=dim1,
                        signgam=SIGN_OF_GAMMA,
                        use_weights=use_weights,
                    )

                    UBmat_2, newUmat_2, newBmat_2, deviatoricstrain_2, RR_2, spotnum_i_pixdev_2, pixdev_2 = (
                        res_refinement
                    )

                    nfit_2 = len(Intensity_list_2)
                    print("nfit_2 = ", nfit_2)
                else:
                    print("not enough peaks with pixdev small enough : ngood = ", ngood)
                    return 0

    elif paramtofit == "calib":

        veryclose_angletol = 1.0

        print(paramdetector_top)

        res = LAA.GetCalibParameter(
            emax_simul,
            veryclose_angletol,
            elem_label,
            UBmat,
            paramdetector_top,
            data,
            pixelsize=pixelsize,
            dim=dim1,
            signgam=SIGN_OF_GAMMA,
            boolctrl=boolctrl,
            use_weights=use_weights,
            addoutput=1,
        )

        newparam, UBmat_2, pixdev_2, nfit_2, pixdev_list, spotnum_hkl_2 = res

        if elim_worst_pixdev:
            pixdevlist = pixdev_list
            print(pixdevlist)
            ind0 = np.where(pixdevlist > maxpixdev)
            print(ind0[0])
            nbad = len(ind0[0])
            if nbad > 0:
                print("MAXPIXDEV = ", maxpixdev)
                print("found ", nbad, " peaks with pixdev larger than ", maxpixdev)
                ind1 = np.where(pixdevlist < maxpixdev)
                ngood = len(ind1[0])
                if ngood > 10:
                    # print spotexpnum_spotsimnum_2
                    # print spotnum_hkl_2
                    # print Intensity_list_2
                    print("fit again after removing bad peaks")
                    print(shape(data))
                    data1 = (np.array(data, dtype=float)).transpose()
                    print(data1[0])
                    datanew = zeros(shape(data1), float)
                    print(shape(spotnum_hkl_2))
                    n_hkl = np.array(spotnum_hkl_2, dtype=float)
                    print(n_hkl[:, 0])
                    numdat_ind = np.array(n_hkl[:, 0], dtype=int)
                    numdat_ind_good = numdat_ind[ind1[0]]
                    datanew[numdat_ind_good] = data1[numdat_ind_good] * 1.0
                    datanew_tuple = (
                        datanew[:, 0],
                        datanew[:, 1],
                        datanew[:, 2],
                        datanew[:, 3],
                        datanew[:, 4],
                    )

                    res = LAA.GetCalibParameter(
                        emax_simul,
                        veryclose_angletol,
                        elem_label,
                        UBmat,
                        paramdetector_top,
                        datanew_tuple,
                        pixelsize=pixelsize,
                        dim=dim1,
                        signgam=SIGN_OF_GAMMA,
                        boolctrl=boolctrl,
                        use_weights=use_weights,
                        addoutput=1,
                    )

                    if res > 0:
                        newparam, UBmat_2, pixdev_2, nfit_2, pixdev_list, spotnum_hkl_2 = (
                            res
                        )
                        print("nfit_2 = ", nfit_2)
                    else:
                        print(
                            "not enough peaks with pixdev small enough : ngood = ",
                            ngood,
                        )
                        return 0

                    # , \
                    # addoutput=1, saturated = data_sat, use_weights = use_weights )

    # #                print newparam
    # #                print nfit_2
    # #                print pixdevlist

    # #        print "UBmat_2 \n", UBmat_2
    # #        print "deviatoricstrain_2 \n", deviatoricstrain_2
    # #        print "RR_2 \n", RR_2
    # #        print "spotnum_i_pixdev_2 \n", spotnum_i_pixdev_2
    # #        print "pixdev_2 \n" , pixdev_2
    # #        print "nfit_2 \n", nfit_2
    # #        print "spotnum_hkl_2 \n", spotnum_hkl_2

    print(np.shape(spotnum_hkl_2))

    spotnum_hkl = np.array(spotnum_hkl_2, dtype=int)

    print("indexed peak list :")
    print("npeak xexp yexp H K L intensity chi theta ")
    for i in range(nfit_2):
        npeak = spotnum_hkl[i, 0]
        # print "npeak = ", npeak
        hkl = spotnum_hkl[i, 1:]
        print(
            npeak,
            data_pixX[npeak],
            data_pixY[npeak],
            np.array(hkl, dtype=int),
            data_I[npeak],
            data_chi[npeak],
            data_theta[npeak],
        )

    numlist = spotnum_hkl[:, 0]
    xyind = np.column_stack((data_pixX[numlist], data_pixY[numlist]))
    # print xyind

    print("matrix with lowest Euler angles")
    matLTmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(UBmat_2)
    print("matLTmin \n", matLTmin.round(decimals=6))
    print("original UB matrix \n", UBmat_2.round(decimals=6))
    print("transfmat \n", list(transfmat))

    epsp1, dlatsrdeg1 = matstarlab_to_deviatoric_strain_crystal(
        GT.mat3x3_to_matline(matLTmin),
        version=2,
        reference_element_for_lattice_parameters=elem_label,
    )
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)")
    print(epsp1.round(decimals=2))
    deviatoricstrain = GT.epsline_to_epsmat(epsp1).round(decimals=2)
    print("deviatoric strain matrix \n", deviatoricstrain)

    print("pixdev mean = ", pixdev_2)
    print("use weights = ", use_weights)

    # transformer aussi les HKL pour qu'ils soient coherents avec matmin
    # return transfmat dans find_lowest_Euler_Angles_matrix
    hkl = spotnum_hkl[:, 1:]
    hklmin = dot(transfmat, hkl.transpose()).transpose()

    intensity_list = data_I[numlist]

    if paramtofit == "strain":
        pixdev_list = np.array(spotnum_i_pixdev_2)[:, 2]

    # calcul Etheor : cubique uniquement
    Etheor = zeros(nfit_2, float)
    uilab = array([0.0, 1.0, 0.0])
    latticeparam = DictLT.dict_Materials[elem_label][1][0] * 1.0
    print(latticeparam)
    dlatu = np.array(
        [latticeparam, latticeparam, latticeparam, pi1 / 2.0, pi1 / 2.0, pi1 / 2.0]
    )
    matstarlab = F2TC.matstarlabLaueTools_to_matstarlabOR(matLTmin)
    mat = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu)
    for i in range(nfit_2):
        H = hklmin[i, 0]
        K = hklmin[i, 1]
        L = hklmin[i, 2]
        qlab = float(H) * mat[0:3] + float(K) * mat[3:6] + float(L) * mat[6:]
        uqlab = qlab / norme(qlab)
        sintheta = -inner(uqlab, uilab)
        if sintheta > 0.0:
            # print "reachable reflection"
            Etheor[i] = (
                DictLT.E_eV_fois_lambda_nm * norme(qlab) / (2.0 * sintheta) * 10.0
            )

    print(shape(numlist))
    print(shape(xyind))
    print(shape(hklmin))

    data_fit = column_stack(
        (
            numlist,
            intensity_list,
            hklmin[:, 0],
            hklmin[:, 1],
            hklmin[:, 2],
            pixdev_list,
            xyind[:, 0],
            xyind[:, 1],
            Etheor,
        )
    )

    data_fit_sorted = sort_peaks_decreasing_int(data_fit, 1)

    filecor = filedat

    # UBmat = matLTmin

    if use_weights == False:
        filesuffix = "_UWN"
    else:
        filesuffix = "_UWY"

    if paramtofit == "strain":
        filefit = save_fit_results(
            filecor,
            data_fit_sorted,
            matLTmin,
            deviatoricstrain,
            filesuffix,
            pixdev_2,
            grainnum,
            paramdetector_top,
            pixelsize,
            dim,
        )
    elif paramtofit == "calib":
        filefit = save_fit_results(
            filecor,
            data_fit_sorted,
            matLTmin,
            deviatoricstrain,
            filesuffix,
            pixdev_2,
            grainnum,
            newparam,
            pixelsize,
            dim,
        )
        # filedet = save_det_results(filedat, filecor, matLTmin, filesuffix, newparam, elem_label)
        # filecal = save_cal_results(filecor, numlist, hkl , intensity_list, pixdev_list , matLTmin, filesuffix, pixdev_2,newparam)

    PLOTRESULTS = 0

    if PLOTRESULTS:
        xyind = data_cor[numlist, 2:4]
        xyall = data_cor[:, 2:4]
        print(shape(xyind))
        # print xy_list
        p.figure(figsize=(8, 8))
        p.plot(xyall[:, 0], xyall[:, 1], "bo", markersize=8)
        p.plot(
            xyind[:, 0], xyind[:, 1], "rx", label="LT", markersize=12, markeredgewidth=2
        )
        p.xlim(0, 2048)
        p.ylim(2048, 0)

    if paramtofit == "strain":
        return (filefit, filecor, nfit_2, pixdev_2, matLTmin)
    elif paramtofit == "calib":
        return (filefit, filecor, nfit_2, pixdev_2, matLTmin, newparam)
        # return(filedet, filecor, nfit_2, pixdev_2, matLTmin)


def save_fit_results(
    filename,
    data_fit,
    matmin,
    deviatoricstrain,
    filesuffix,
    pixdev,
    grainnum,
    calib,
    pixelsize=165.0 / 2048.0,
    dim=(2048.0, 2048.0),
):

    extension = filename.split(".")[-1]
    extension = "." + extension
    # print extension

    outputfilename = (
        filename.split(extension)[0] + filesuffix + "_" + str(grainnum) + ".fit"
    )
    # print outputfilename

    datatooutput = data_fit.round(decimals=4)
    euler = calc_Euler_angles(matmin).round(decimals=3)

    header = "# Strain and Orientation Refinement of: %s\n" % (filename)
    header += "# File created at %s with multigrain_new2.py \n" % (asctime())
    header += (
        "# Number of indexed spots : "
        + str(shape(data_fit)[0])
        + " pixdev :"
        + str(round(pixdev, 4))
        + "\n"
    )
    header += "spot# Intensity h k l  pixDev xexp yexp Etheor\n"
    outputfile = open(outputfilename, "w")
    outputfile.write(header)
    np.savetxt(outputfile, datatooutput, fmt="%.4f")
    outputfile.write("#UB matrix in reciprocal space\n")
    outputfile.write(str(matmin) + "\n")
    outputfile.write("#deviatoric strain (1e-3 units)\n")
    outputfile.write(str(deviatoricstrain) + "\n")
    outputfile.write("#Euler angles phi theta psi (deg)\n")
    outputfile.write(str(euler) + "\n")
    outputfile.write("#grain number\n")
    outputfile.write(str(grainnum) + "\n")
    text = "#Number of indexed spots \n"
    text += str(shape(data_fit)[0]) + "\n"
    outputfile.write(text)
    text = "#pixdev\n"
    text += str(round(pixdev, 4)) + "\n"
    outputfile.write(text)
    # MAR ou Roper
    # pixelsize = 165. / 2048
    # dim = (2048.0, 2048.0)
    # VHR
    # #    pixelsize = 0.031
    # #    dim = (2594.0, 3764.0)
    text = "#Sample-Detector distance (IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2\n"
    dd, xcen, ycen, xbet, xgam = calib[0], calib[1], calib[2], calib[3], calib[4]
    text += "%.3f, %.2f, %.2f, %.3f, %.3f, %.5f, %.0f, %.0f\n" % (
        round(dd, 3),
        round(xcen, 2),
        round(ycen, 2),
        round(xbet, 3),
        round(xgam, 3),
        pixelsize,
        round(dim[0], 0),
        round(dim[1], 0),
    )
    outputfile.write(text)

    outputfile.close()

    return outputfilename


def save_det_results(filedat, filecor, matLTmin, filesuffix, newparam, elem_label):

    pixelsize = 165.0 / 2048
    dim = (2048.0, 2048.0)

    outputfilename = filecor.split(".")[0] + filesuffix + ".det"

    m11, m12, m13, m21, m22, m23, m31, m32, m33 = np.ravel(matLTmin).round(decimals=7)

    dd, xcen, ycen, xbet, xgam = (
        newparam[0],
        newparam[1],
        newparam[2],
        newparam[3],
        newparam[4],
    )

    text = "%.3f, %.2f, %.2f, %.3f, %.3f, %.5f, %.0f, %.0f\n" % (
        round(dd, 3),
        round(xcen, 2),
        round(ycen, 2),
        round(xbet, 3),
        round(xgam, 3),
        pixelsize,
        round(dim[0], 0),
        round(dim[1], 0),
    )
    text += (
        "Sample-Detector distance (IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2\n"
    )
    text += "Calibration done with %s at %s with batchOR.py\n" % (elem_label, asctime())
    text += "Experimental Data file: %s\n" % filedat
    text += "Orientation Matrix:\n"
    text += "[[%.7f,%.7f,%.7f],[%.7f,%.7f,%.7f],[%.7f,%.7f,%.7f]]" % (
        m11,
        m12,
        m13,
        m21,
        m22,
        m23,
        m31,
        m32,
        m33,
    )
    outputfile = open(outputfilename, "w")
    outputfile.write(text)
    outputfile.close()


def speed_index_refine(
    filedet,
    filepath,
    fileprefix,
    filesuffix,
    indimg,
    use_weights=False,
    proposed_matrix=None,
    paramtofit="strain",
    elem_label="UO2",
):

    numfiles = len(indimg)
    nlist = zeros(numfiles, dtype=int)
    pixdevlist = zeros(numfiles, dtype=float)

    paramdetector_top, matstarlab = F2TC.readlt_det(filedet)

    i = 0
    mat_ref = proposed_matrix

    for kk in indimg:

        peakfile = filepath + fileprefix + rmccd.stringint(kk, 4) + filesuffix

        print(peakfile)

        filefit, filecor, nlist[i], pixdevlist[i], matLTmin = test_index_refine(
            peakfile,
            list(paramdetector_top),
            use_weights=use_weights,
            proposed_matrix=mat_ref,
            paramtofit=paramtofit,
            elem_label=elem_label,
        )

        mat_ref = matLTmin

        if paramtofit == "strain":
            listfile = [filefit, filecor, filedet]
            mergefiles(listfile)

        i = i + 1

    print(nlist)
    print(pixdevlist.round(decimals=4))

    return (nlist, pixdevlist)


def sort_peaks_decreasing_int(data_str, colnum):

    print("tri des pic par intensite decroissante")

    npics = shape(data_str)[0]
    print(npics)
    index2 = zeros(npics, int)

    index1 = argsort(data_str[:, colnum])
    for i in range(npics):
        index2[i] = index1[npics - i - 1]
    # print "index2 =", index2
    data_str2 = data_str[index2]

    return data_str2


from math import acos

# #    /* phi = GRAINIMAGELIST[NGRAINIMAGE].EULER[0] / RAD;
# #    theta = GRAINIMAGELIST[NGRAINIMAGE].EULER[1]/ RAD;
# #    psi = GRAINIMAGELIST[NGRAINIMAGE].EULER[2]/ RAD;
# #    b[0][0] = cos(psi)*cos(phi)- cos(theta)*sin(phi)*sin(psi);
# #    b[0][1] = cos(psi)*sin(phi)+ cos(theta)*cos(phi)*sin(psi);
# #    b[0][2] = sin(psi)*sin(theta);
# #    b[1][0] = -sin(psi)*cos(phi)-cos(theta)*sin(phi)*cos(psi);
# #    b[1][1] = -sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi);
# #    b[1][2] = cos(psi)*sin(theta);
# #    b[2][0] = sin(theta)*sin(phi);
# #    b[2][1] = -sin(theta)*cos(phi);
# #    b[2][2] = cos(theta);


def calc_Euler_angles(mat3x3):

    # mat3x3 = matrix "minimized" in LT lab frame
    # see FindO.find_lowest_Euler_Angles_matrix
    # phi 0, theta 1, psi 2

    mat = GT.matline_to_mat3x3(
        matstarlab_to_matstarlabOND(GT.mat3x3_to_matline(mat3x3))
    )

    RAD = 180.0 / math.pi

    euler = zeros(3, float)
    euler[1] = RAD * acos(mat[2, 2])

    if abs(abs(mat[2, 2]) - 1.0) < 1e-5:
        # if theta is zero, phi+psi is defined, if theta is pi, phi-psi is defined */
        # put psi = 0 and calculate phi */
        # psi */
        euler[2] = 0.0
        # phi */
        euler[0] = RAD * acos(mat[0, 0])
        if mat[0, 1] < 0.0:
            euler[0] = -euler[0]
    else:
        # psi */
        toto = sqrt(1 - mat[2, 2] * mat[2, 2])  # sin theta - >0 */
        euler[2] = RAD * acos(mat[1, 2] / toto)
        # phi */
        euler[0] = RAD * acos(-mat[2, 1] / toto)
        if mat[2, 0] < 0.0:
            euler[0] = 360.0 - euler[0]

        # print "Euler angles phi theta psi (deg)"
        # print euler.round(decimals = 3)

    return euler


def merge_fit_files_multigrain(filelist, removefiles=1):
    """
    .fit with all grains
    """

    if len(filelist) == 0:
        filedum = "toto.fit"
        dat1 = zeros(6, float)
        savetxt(filedum, dat1)
        return filedum

    filefitallgrains = filelist[0].split("_1.")[0] + "_mg.fit"
    print("merged fit file : ", filefitallgrains)
    outputfile = open(filefitallgrains, "w")
    for filename in filelist:
        outputfile.write(filename + "\n")
        f = open(filename, "r")
        try:
            for line in f:
                outputfile.write(line)
        finally:
            f.close()
        outputfile.write("\n")
    outputfile.close()
    if removefiles:
        for filename in filelist:
            os.remove(filename)

    return filefitallgrains


def convert_xmas_str_to_LT_fit(
    filestr, elem_label="W", stiffness_c_tensor=None, schmid_tensors=None
):

    # 29May12
    print("convert XMAS .str file into LaueTools .fit file (multigrain)")
    print(
        "warning : compatibility OK with XMAS 2006 version, not OK for later XMAS versions"
    )

    numcolint = (
        9
    )  # 10  # intens = column used by XMAS for intensity sorting in peak search
    # min_matLT = True
    min_matLT = False
    # data_str_all :  xy(exp) 0:2 hkl 2:5 xydev 5:7 energy 7 dspacing 8  intens 9 integr 10"
    data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(
        filestr, min_matLT=min_matLT
    )

    indstart = 0

    print("sorting whole spot list by intensities")

    npics = shape(data_str_all)[0]
    toto = list(range(npics))
    toto1 = list(range(npics, 0, -1))
    data_str_all2 = column_stack((data_str_all, toto1))
    data_str_all2_sorted = sort_peaks_decreasing_int(data_str_all2, numcolint)
    index1 = np.array(data_str_all2_sorted[:, -1], dtype=int)
    # print index1
    data_str_all3 = column_stack((data_str_all2_sorted, toto))
    data_str_all3_unsorted = sort_peaks_decreasing_int(data_str_all3, -2)
    # data_str_all3_unsorted = data_str_all3[index1,:]
    index2 = np.array(data_str_all3_unsorted[:, -1], dtype=int)

    # print data_str_all[:,numcolint]
    # print data_str_all3_unsorted[:,numcolint]
    # print index2
    # print data_str_all2_sorted[:,numcolint]

    pixdev_list = zeros(npics, float)
    Etheor_list = zeros(npics, float)
    for i in range(npics):
        pixdev_list[i] = sqrt(
            data_str_all[i, 5] * data_str_all[i, 5]
            + data_str_all[i, 6] * data_str_all[i, 6]
        )

    data_str_all = column_stack((data_str_all, index2, pixdev_list, Etheor_list))
    # data_str_all : index1 11, pixdev 12
    # print data_str_all
    filelist = []

    for i in range(len(npeaks)):
        print("grain number : ", i)
        if npeaks[i] > 0:
            filecor = filestr
            filesuffix = "_from_xmas"
            indend = indstart + npeaks[i]
            toto = data_str_all[indstart:indend, :]
            # data_fit = vstack((toto[:,11],toto[:,10],toto[:,2],toto[:,3],toto[:,4],toto[:,12],toto[:,0],toto[:,1])).transpose()
            data_fit = column_stack(
                (
                    toto[:, 11],
                    toto[:, numcolint],
                    toto[:, 2:5],
                    toto[:, 12],
                    toto[:, :2],
                    toto[:, -1],
                )
            )
            # data_fit = vstack((numlist, intensity_list, hklmin[:,0], hklmin[:,1], hklmin[:,2] ,
            # pixdev_list, xyind[:,0],xyind[:,1])).transpose()
            data_fit_sorted = sort_peaks_decreasing_int(data_fit, 1)
            matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab[i, :])
            # GT.mat3x3_to_matline(matLTmin)

            epsp_sample, epsp_crystal = matstarlab_to_deviatoric_strain_sample(
                matstarlab[i, :],
                omega0=40.0,
                version=2,
                returnmore=True,
                reference_element_for_lattice_parameters=elem_label,
            )

            print(
                "deviatoric strain crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)"
            )
            print(epsp_crystal.round(decimals=2))

            print(
                "deviatoric strain sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-3 units)"
            )
            print(epsp_sample.round(decimals=2))

            if stiffness_c_tensor != None:

                sigma_crystal = deviatoric_strain_crystal_to_stress_crystal(
                    c_tensor, epsp_crystal
                )
                sigma_sample = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(
                    matstarlab[i, :], sigma_crystal, omega0=40.0
                )

                print(
                    "deviatoric stress crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (100 MPa units)"
                )
                print(sigma_crystal.round(decimals=2))

                print(
                    "deviatoric stress sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (100 MPa units)"
                )
                print(sigma_sample.round(decimals=2))

                von_mises = deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal)
                print(
                    "Von Mises equivalent Stress (100 MPa units)", round(von_mises, 3)
                )

                if schmid_tensors != None:
                    tau1 = deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(
                        sigma_crystal, schmid_tensors
                    )
                    print(
                        "RSS resolved shear stresses on glide planes (100 MPa units) : "
                    )
                    print(tau1.round(decimals=3))
                    print("Max RSS : ", round(abs(tau1).max(), 3))

            deviatoricstrain = GT.epsline_to_epsmat(epsp_crystal).round(decimals=2)
            # here the recalculated eps_yz and eps_xy are opposite to those of XMAS
            # while all the epsilon_crystal components are equal to those of XMAS is min_matLT = False
            # y_sample_XMAS = - y_sample_OR ? (i.e. xyz_sample_XMAS est un triedre indirect ?)

            filefit = save_fit_results(
                filecor,
                data_fit_sorted,
                matLTmin,
                deviatoricstrain,
                filesuffix,
                dev_str[i, 2],
                i + 1,
                calib,
            )
            filelist.append(filefit)
            indstart = indstart + npeaks[i]
            print("indstart =", indstart)
        else:
            print("bad grain")

    if i > 1:
        merge_fit_files_multigrain(filelist)
    return 0


def find_common_peaks(xy1, xy2, dxytol=0.01, verbose=1):

    # print "look for peaks common to two grains"
    ndat1 = shape(xy1)[0]
    ndat2 = shape(xy2)[0]
    if verbose:
        print("ndat1, ndat2 = ", ndat1, ndat2)

    # print xy1

    iscommon1 = zeros(ndat1, int)
    iscommon2 = zeros(ndat2, int)

    nb_common_peaks = 0
    # print "common peaks : "
    for j in range(ndat1):
        # print "j= ", j
        for k in range(ndat2):
            # print "k= ", k
            dxy = norme(xy1[j, :] - xy2[k, :])
            if dxy < dxytol:
                # print xy1[j,:],xy2[k,:]
                nb_common_peaks = nb_common_peaks + 1
                iscommon1[j] = iscommon1[j] + 1
                iscommon2[k] = iscommon2[k] + 1

    if verbose:
        print("nb_common_peaks =", nb_common_peaks)
    ind1 = where(iscommon1 == 0)
    ind2 = where(iscommon2 == 0)
    # #    print iscommon1
    # #    print iscommon2
    # #    print ind1
    # #    print ind2

    if verbose:
        if nb_common_peaks != 0:  # & (nb_common_peaks < max(ndat1,ndat2)):
            print("nb / indexes / xy of non-common peaks : ")
            if shape(ind1)[1] > 0:
                print("list 1 :", len(ind1[0]), " : ", ind1[0])
                print(xy1[ind1[0]])
            if shape(ind2)[1] > 0:
                print("list 2 :", len(ind2[0]), " : ", ind2[0])
                print(xy2[ind2[0]])

    return (nb_common_peaks, iscommon1, iscommon2)


def readlt_fit_mg(
    fitfilename, verbose=1, readmore=False, fileextensionmarker=".fit"
):  # 29May13
    """
    read a single .fit file containing data for several grains
    """
    print("reading multigrain fit file: %s" % fitfilename)

    f = open(fitfilename, "r")

    # search for each start of grain dat
    nbgrains = 0
    linepos_grain_list = []
    lineindex = 0
    try:
        for line in f:
            _line = line.rstrip("\n")
            if _line.endswith(fileextensionmarker):
                nbgrains += 1
                linepos_grain_list.append(lineindex)
            lineindex += 1
    finally:
        linepos_grain_list.append(lineindex)
        f.close()

    if verbose:
        print("nbgrains = ", nbgrains)
        print("linepos_grain_list = ", linepos_grain_list)
        print("linepos_grain_list", linepos_grain_list)

    # nothing has been indexed
    if nbgrains == 0:
        return 0

    f = open(fitfilename, "rb")

    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)

    gnumlist = list(range(nbgrains))

    matstarlab = zeros((nbgrains, 9), float)
    strain6 = zeros((nbgrains, 6), float)
    calib = zeros((nbgrains, 5), float)
    euler = zeros((nbgrains, 3), float)
    npeaks = zeros(nbgrains, int)
    indstart = zeros(nbgrains, int)
    pixdev = zeros(nbgrains, float)

    # read .fit file for each grain
    for k in range(nbgrains):
        matLT3x3 = np.zeros((3, 3), dtype=np.float)
        strain = np.zeros((3, 3), dtype=np.float)
        i = 0
        n = linepos_grain_list[k]
        # print "n = ", n
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        # print f.readline()
        f.seek(line_offset[n + 1])
        matrixfound = 0
        calibfound = 0
        pixdevfound = 0
        strainfound = 0
        eulerfound = 0
        linecalib = 0
        linepixdev = 0
        linestrain = 0
        lineeuler = 0
        list1 = []
        linestartspot = 10000
        lineendspot = 10000

        while i < (linepos_grain_list[k + 1] - linepos_grain_list[k]):
            line = f.readline()
            i = i + 1
            # print i
            if line.startswith(("spot#", "#spot")):
                #                 linecol = line.rstrip('\n')
                linestartspot = i + 1
            if line.startswith("#UB"):
                # print line
                matrixfound = 1
                linestartmat = i
                lineendspot = i
                j = 0
                # print "matrix found"
            if line.startswith("#Sample"):
                # print line
                calibfound = 1
                linecalib = i + 1
            if line.startswith("#pixdev"):
                # print line
                pixdevfound = 1
                linepixdev = i + 1
            if line.startswith("#deviatoric"):
                # print line
                strainfound = 1
                linestrain = i
                j = 0
            if line.startswith("#Euler"):
                # print line
                eulerfound = 1
                lineeuler = i + 1
            if matrixfound:
                if i in (linestartmat + 1, linestartmat + 2, linestartmat + 3):
                    toto = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    # print toto
                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    j = j + 1
            if strainfound:
                if i in (linestrain + 1, linestrain + 2, linestrain + 3):
                    toto = line.rstrip("\n").replace("[", "").replace("]", "").split()
                    # print toto
                    strain[j, :] = np.array(toto, dtype=float)
                    j = j + 1
            if calibfound & (i == linecalib):
                calib[k, :] = np.array(line.split(",")[:5], dtype=float)
                # print "calib = ", calib[k,:]
            if eulerfound & (i == lineeuler):
                euler[k, :] = np.array(
                    line.replace("[", "").replace("]", "").split()[:3], dtype=float
                )
                # print "euler = ", euler[k,:]
            if pixdevfound & (i == linepixdev):
                pixdev[k] = float(line.rstrip("\n"))
                # print "pixdev = ", pixdev[k]
            if (i >= linestartspot) & (i < lineendspot):
                print(line, i)
                list1.append(
                    line.rstrip("\n").replace("[", "").replace("]", "").split()
                )

        data_fit = np.array(list1, dtype=float)
        npeaks[k] = shape(data_fit)[0]

        verbose2 = 0
        if verbose2:
            print(np.shape(data_fit))
            print(data_fit[0, :2])
            print(data_fit[-1, :2])

        #        if min_matLT == True :
        #            matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        #            matLT3x3 = matmin
        #            print "transfmat \n", list(transfmat)
        #            # transformer aussi les HKL pour qu'ils soient coherents avec matmin
        #            hkl = data_fit[:, 2:5]
        #            data_fit[:, 2:5] = np.dot(transfmat, hkl.transpose()).transpose()

        matstarlab[k, :] = F2TC.matstarlabLaueTools_to_matstarlabOR(matLT3x3)

        # xx yy zz yz xz xy
        # voigt notation
        strain6[k, :] = np.array(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                strain[1, 2],
                strain[0, 2],
                strain[0, 1],
            ]
        )

        if k == 0:
            data_fit_all = data_fit * 1.0
        else:
            data_fit_all = row_stack((data_fit_all, data_fit))

    f.close()

    for k in range(1, nbgrains):
        indstart[k] = indstart[k - 1] + npeaks[k - 1]

    if verbose:
        print("gnumlist = ", gnumlist)
        print("matstarlab = ")
        print(matstarlab)
        print("npeaks = ", npeaks)
        print("indstart = ", indstart)
        print("pixdev = ", pixdev.round(decimals=4))
        print("strain6 = \n", strain6.round(decimals=2))
        print("euler = \n", euler.round(decimals=3))

    try:
        import module_graphique as modgraph

        modgraph.euler = euler
    except (ImportError):
        print("You need module_graphique.py")

    if readmore == False:
        return (gnumlist, npeaks, indstart, matstarlab, data_fit_all, calib, pixdev)
    elif readmore == True:
        return (
            gnumlist,
            npeaks,
            indstart,
            matstarlab,
            data_fit_all,
            calib,
            pixdev,
            strain6,
            euler,
        )


def readfitfile_multigrains(
    fitfilename,
    verbose=0,
    readmore=False,
    fileextensionmarker=".fit",
    default_file=None,
):  # 29May13
    """
    JSM version of readlt_fit_mg()
    read a single .fit file containing data for several grains

    spots data are excluded

    fileextensionmarker :  '.fit' extension at the end of the line 
                            stating that a new grain data starts
    """
    print("reading multigrain fit file: %s" % fitfilename)

    if not os.path.exists(fitfilename):
        f = open(default_file, "r")
    else:
        f = open(fitfilename, "r")

    # search for each start of grain dat
    nbgrains = 0
    linepos_grain_list = []
    lineindex = 1
    try:
        for line in f:
            _line = line.rstrip("\n")
            if _line.endswith(fileextensionmarker) and not _line.startswith(
                "# Unindexed and unrefined"
            ):
                nbgrains += 1
                linepos_grain_list.append(lineindex)
            lineindex += 1
    finally:
        linepos_grain_list.append(lineindex)
        f.close()

    if verbose:
        print("nbgrains = ", nbgrains)
        print("linepos_grain_list = ", linepos_grain_list)

    # nothing has been indexed
    if nbgrains == 0:
        return 0

    if not os.path.exists(fitfilename):
        f = open(default_file, "rb")
    else:
        f = open(fitfilename, "rb")

    # Read in the file once and build a list of line offsets
    #     line_offset = []
    #     offset = 0
    #     for line in f:
    #         line_offset.append(offset)
    #         offset += len(line)

    gnumlist = list(range(nbgrains))

    matstarlab = zeros((nbgrains, 9), float)
    strain6 = zeros((nbgrains, 6), float)
    calib = zeros((nbgrains, 5), float)
    calibJSM = zeros((nbgrains, 7), float)
    euler = zeros((nbgrains, 3), float)
    npeaks = zeros(nbgrains, int)
    indstart = zeros(nbgrains, int)
    pixdev = zeros(nbgrains, float)

    Material_list = []
    GrainName_list = []
    PixDev_list = []

    # read .fit file for each grain

    matLT3x3 = np.zeros((3, 3), dtype=np.float)
    strain = np.zeros((3, 3), dtype=np.float)

    #     n = linepos_grain_list[grain_index]
    #         # print "n = ", n
    #         # Now, to skip to line n (with the first line being line 0), just do
    #         f.seek(line_offset[n])
    #         # print f.readline()
    #         f.seek(line_offset[n + 1])
    matrixfound = 0
    calibfound = 0
    calibfoundJSM = 0
    pixdevfound = 0
    strainfound = 0
    eulerfound = 0
    linecalib = 0
    linepixdev = 0
    linestrain = 0
    lineeuler = 0
    list1 = []
    linestartspot = 10000
    lineendspot = 10000

    for grain_index in range(nbgrains):
        print("changing grain_index now = ", grain_index)
        print(linepos_grain_list[grain_index + 1])
        print(linepos_grain_list[grain_index])
        iline = linepos_grain_list[grain_index]

        nb_indexed_spots = 0

        while iline < linepos_grain_list[grain_index + 1]:

            line = f.readline()
            #             print "iline =%d line" % iline, line
            # print i
            if line.startswith("# Number of indexed spots"):
                nb_indexed_spots = int(line.split(":")[-1])

            elif line.startswith("# Number of unindexed spots"):
                nb_indexed_spots = 0
                nb_UNindexed_spots = int(line.split(":")[-1])

            elif line.startswith(
                ("# Mean Pixel Deviation", "# Mean Deviation(pixel):")
            ):
                meanpixdev = float(line.split(":")[-1])
                print("meanpixdev", meanpixdev)
                PixDev_list.append(meanpixdev)

            elif line.startswith("#Element"):
                line = f.readline()
                Material_list.append(line.rstrip("\n"))
                #                 print "Material_list", Material_list
                iline += 1
            elif line.startswith("#grainIndex"):
                line = f.readline()
                GrainName_list.append(line.rstrip("\n"))
                #                 print "GrainName_list", GrainName_list

                iline += 1
            elif line.startswith(("spot#", "#spot")):
                if nb_indexed_spots > 0:
                    #                     print "nb of indexed spots", nb_indexed_spots
                    nbspots = nb_indexed_spots

                    dataspots = []
                    for kline in range(nbspots):
                        line = f.readline()
                        iline += 1
                        dataspots.append(
                            line.rstrip("\n").replace("[", "").replace("]", "").split()
                        )

                    dataspots = np.array(dataspots, dtype=np.float)
                #                     print "got dataspots!"
                #                     print "shape", dataspots.shape

                elif nb_UNindexed_spots > 0:
                    #                     print "nb of UNindexed spots", nb_UNindexed_spots
                    nbspots = nb_UNindexed_spots

                    dataspots_Unindexed = []
                    for kline in range(nbspots):
                        line = f.readline()
                        iline += 1
                        dataspots_Unindexed.append(
                            line.rstrip("\n").replace("[", "").replace("]", "").split()
                        )

                    dataspots_Unindexed = np.array(dataspots_Unindexed, dtype=np.float)
            #                     print "got dataspots_Unindexed!"
            #                     print "shape", dataspots_Unindexed.shape

            elif line.startswith("#UB"):
                matrixfound = 1

                lineendspot = iline - 1

                # print "matrix found"
            elif line.startswith("#Sample") | line.startswith("#DetectorParameters"):
                # print line
                calibfound = 1
                linecalib = iline + 1
            elif line.startswith("# Calibration"):
                # print line
                calibfoundJSM = 1
                linecalib = iline + 1
            elif line.startswith("#pixdev"):
                # print line
                pixdevfound = 1
                linepixdev = iline + 1
            elif line.startswith("#deviatoric"):
                # print line
                strainfound = 1

            elif line.startswith("#Euler"):
                # print line
                eulerfound = 1
                lineeuler = iline + 1

            if matrixfound:
                for jline_matrix in range(3):
                    line = f.readline()
                    #                     print "line in matrix", line
                    lineval = (
                        line.rstrip("\n").replace("[", "").replace("]", "").split()
                    )
                    # print toto
                    matLT3x3[jline_matrix, :] = np.array(lineval, dtype=float)
                    iline += 1
                #                 print "got UB matrix:", matLT3x3
                matrixfound = 0
            if strainfound:
                for jline_matrix in range(3):
                    line = f.readline()
                    #                     print "line in matrix", line
                    lineval = (
                        line.rstrip("\n").replace("[", "").replace("]", "").split()
                    )
                    # print toto
                    strain[jline_matrix, :] = np.array(lineval, dtype=float)
                    iline += 1
                #                 print "got strain matrix:", strain
                strainfound = 0
            if calibfoundJSM:
                calibparam = []
                for jline_calib in range(7):
                    line = f.readline()
                    #                     print "line in matrix", line
                    val = float(line.split(":")[-1])
                    # print toto
                    calibparam.append(val)
                    iline += 1
                #                 print "got calibration parameters:", calibparam
                calibJSM[grain_index, :] = calibparam
                calibfoundJSM = 0

            if calibfound & (iline == linecalib):
                calibJSM[grain_index, :5] = np.array(
                    line.rstrip("\n").replace("[", "").replace("]", "").split(",")[:5],
                    dtype=float,
                )
                print("calib = ", calibJSM[grain_index, :])
            if eulerfound & (iline == lineeuler):
                euler[grain_index, :] = np.array(
                    line.replace("[", "").replace("]", "").split()[:3], dtype=float
                )
                # print "euler = ", euler[grain_index,:]
            if pixdevfound & (iline == linepixdev):
                pixdev[grain_index] = float(line.rstrip("\n"))
                # print "pixdev = ", pixdev[grain_index]
            #             if (iline >= linestartspot) & (iline < lineendspot):
            # #                 print line, iline
            #                 list1.append(line.rstrip('\n').replace('[', '').replace(']', '').split())

            iline += 1

        npeaks[grain_index] = np.shape(dataspots)[0]

        #        if min_matLT == True :
        #            matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
        #            matLT3x3 = matmin
        #            print "transfmat \n", list(transfmat)
        #            # transformer aussi les HKL pour qu'ils soient coherents avec matmin
        #            hkl = data_fit[:, 2:5]
        #            data_fit[:, 2:5] = np.dot(transfmat, hkl.transpose()).transpose()

        matstarlab[grain_index, :] = F2TC.matstarlabLaueTools_to_matstarlabOR(matLT3x3)

        # xx yy zz yz xz xy
        # voigt notation
        strain6[grain_index, :] = np.array(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                strain[1, 2],
                strain[0, 2],
                strain[0, 1],
            ]
        )

        if grain_index == 0:
            data_fit_all = dataspots * 1.0
        elif grain_index:
            data_fit_all = row_stack((data_fit_all, dataspots))

    f.close()

    for grain_index in range(1, nbgrains):
        indstart[grain_index] = indstart[grain_index - 1] + npeaks[grain_index - 1]

    pixdev = np.array(PixDev_list, dtype=np.float)

    if verbose:
        print("gnumlist = ", gnumlist)
        print("matstarlab (OR frame) ")
        print(matstarlab)
        print("npeaks = ", npeaks)
        print("indstart = ", indstart)
        print("pixdev = ", pixdev.round(decimals=4))
        print("strain6 = \n", strain6.round(decimals=2))
        print("euler = \n", euler.round(decimals=3))

    try:
        import module_graphique as modgraph

        modgraph.euler = euler
    except (ImportError):
        print("You need module_graphique.py")

    if readmore == False:
        return (
            gnumlist,
            npeaks,
            indstart,
            matstarlab,
            data_fit_all,
            calibJSM[:, :5],
            pixdev,
        )
    elif readmore == True:
        return (
            gnumlist,
            npeaks,
            indstart,
            matstarlab,
            data_fit_all,
            calibJSM[:, :5],
            pixdev,
            strain6,
            euler,
        )


def calc_pixdev(matstarlab, calib, data_xy, data_hkl):

    Npics = len(data_xy[:, 0])
    xytheor = zeros((Npics, 2), float)
    xydev = zeros((Npics, 2), float)
    pixdev = zeros(Npics, float)
    uilab = array([0.0, 1.0, 0.0])

    mat1 = matstarlab

    for i in range(Npics):
        # print "hkl = ", hkl0[i,:]
        qlab = (
            data_hkl[i, 0] * mat1[0:3]
            + data_hkl[i, 1] * mat1[3:6]
            + data_hkl[i, 2] * mat1[6:]
        )
        # print "qlab = ", qlab
        uqlab = qlab / norme(qlab)
        sintheta = -inner(uqlab, uilab)
        if sintheta > 0.0:
            xydev[i, :] = uqlab_to_xycam(uqlab, calib) - data_xy[i, :]
            pixdev[i] = norme(xydev[i, :])
            # print np.array(data_hkl[i,:], dtype=int), xydev[i].round(decimals=3), \
            #    round(pixdev[i],3)
        else:
            print(data_hkl[i, :], "unreachable reflection")

    pixdevmean = pixdev.mean()
    # print "pixdev : mean ", round(pixdevmean,3)

    return pixdevmean


def compare_multigrain_fit(
    filefitmg1, filefitmg2, mat_tol=1.0e-3, dxytol=0.05, compare_calib=0, elem_label="W"
):

    # 29May13
    gnumlist1, npeaks1, indstart1, matstarlab1, data_fit1, calib1, pixdev1 = readlt_fit_mg(
        filefitmg1
    )
    gnumlist2, npeaks2, indstart2, matstarlab2, data_fit2, calib2, pixdev2 = readlt_fit_mg(
        filefitmg2
    )
    ng1 = len(gnumlist1)
    ng2 = len(gnumlist2)
    ng = min(ng1, ng2)
    # couple : gind1 gind2 gnum1 gnum2
    grain_couples = zeros((ng, 4), int)

    k = 0
    for i in range(ng1):
        matching_grain_found = 0
        mat1 = matstarlab1[i, :]
        for j in range(ng2):
            mat2 = matstarlab2[j, :]
            dmat = mat2 - mat1
            # print "gnum1, gnum2, norme(dmat) = ",  gnumlist1[i], gnumlist2[j], norme(dmat)
            if norme(dmat) < mat_tol:
                # print "dmat = ", dmat
                print(
                    "gnum1, gnum2, norme(dmat) = ",
                    gnumlist1[i],
                    gnumlist2[j],
                    norme(dmat),
                )
                grain_couples[k, 0] = i
                grain_couples[k, 1] = j
                grain_couples[k, 2] = gnumlist1[i]
                grain_couples[k, 3] = gnumlist2[j]
                print(grain_couples[k, :])
                k = k + 1
                matching_grain_found = 1
        if matching_grain_found == 0:
            print("no match for grain gind, gnum :", i, gnumlist1[i])
            if 1:
                range1 = arange(indstart1[i], indstart1[i] + npeaks1[i])
                xy1 = data_fit1[range1, 6:8]
                for j in range(ng2):
                    if j not in grain_couples[:k, 1]:
                        print("gnum1, gnum2 = ", gnumlist1[i], gnumlist2[j])
                        range2 = arange(indstart2[j], indstart2[j] + npeaks2[j])
                        xy2 = data_fit2[range2, 6:8]
                        find_common_peaks(xy1, xy2, dxytol=dxytol)
    # print grain_couples
    ncouples = k
    print(filefitmg1)
    print(filefitmg2)
    print("number of couples found : ", k)
    print(grain_couples[:k, :])
    print("common peaks")
    print("couple : gind1 gind2 gnum1 gnum2")
    color1 = ("ko", "ro", "go", "bo", "mo", "ks")
    for k in range(ncouples):
        print("###########################################")
        print(grain_couples[k, :])
        i = grain_couples[k, 0]
        j = grain_couples[k, 1]
        if len(indstart1) > 1:
            range1 = arange(indstart1[i], indstart1[i] + npeaks1[i])
        else:
            range1 = arange(0, npeaks1)
        if len(indstart2) > 1:
            range2 = arange(indstart2[j], indstart2[j] + npeaks2[j])
        else:
            range2 = arange(0, npeaks2)
        xy1 = data_fit1[range1, 6:8]
        xy2 = data_fit2[range2, 6:8]
        find_common_peaks(xy1, xy2, dxytol=dxytol)

        matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1[i, :])
        # GT.mat3x3_to_matline(matLTmin)

        epsp1, dlatsrdeg1 = matstarlab_to_deviatoric_strain_crystal(
            matstarlab1[i, :],
            version=2,
            reference_element_for_lattice_parameters=elem_label,
        )

        euler1 = calc_Euler_angles(matLTmin).round(decimals=3)

        matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab2[j, :])

        epsp2, dlatsrdeg1 = matstarlab_to_deviatoric_strain_crystal(
            matstarlab2[j, :],
            version=2,
            reference_element_for_lattice_parameters=elem_label,
        )

        euler2 = calc_Euler_angles(matLTmin).round(decimals=3)
        deps = (epsp2 - epsp1) * 10.0
        print("difference of deviatoric strain eps2 - eps1 (1e-4 units)")
        print(deps.round(decimals=1))
        deuler = (euler2 - euler1) * 1.0e4 * math.pi / 180.0
        print("difference of euler angles euler2 - euler1 (0.1 mrad)")
        print(deuler.round(decimals=1))
        hkl1 = data_fit1[range1, 2:5]
        hkl2 = data_fit2[range2, 2:5]
        print("pixdev1, pixdev2 (from file)")
        print(round(pixdev1[i], 4), round(pixdev2[j], 4))
        print("pixdev1, pixdev2 (recalculated from xy hkl mat)")
        pixdev1b = calc_pixdev(matstarlab1[i, :], calib1, xy1, hkl1)
        pixdev2b = calc_pixdev(matstarlab2[j, :], calib2, xy2, hkl2)
        print(round(pixdev1b, 4), round(pixdev2b, 4))
        if compare_calib:
            dcalibn = zeros(5, float)
            pixelsize = 165.0 / 2048.0
            RAD = math.pi / 180.0
            dcalib = calib2 - calib1
            print("difference of calibration")
            print("standard units mm pix pix deg deg")
            print_calib(dcalib)
            print(
                "angular units 0.1 mrad  d(dd)/dd, d(xcen)/dd, d(ycen)/dd, dxbet, dxgam"
            )
            dcalibn[0] = dcalib[0] / calib1[0]
            dcalibn[1] = dcalib[1] * pixelsize / calib1[0]
            dcalibn[2] = dcalib[2] * pixelsize / calib1[0]
            dcalibn[3] = dcalib[3] * RAD
            dcalibn[4] = dcalib[4] * RAD
            dcalibn = dcalibn * 1.0e4
            print(dcalibn.round(decimals=1))

        if 0:
            p.figure(figsize=(8, 8))
            p.plot(xy1[:, 0], -xy1[:, 1], "ro")
            p.plot(xy2[:, 0], -xy2[:, 1], "kx", markersize=20, markeredgewidth=3)
            text1 = "G" + str(grain_couples[k, 2])
            p.text(100.0, -150.0, text1, fontsize=20)
            p.xlim(0.0, 2048.0)
            p.ylim(-2048.0, 0.0)
            p.xlabel("xcam XMAS - toward back")
            p.ylabel("-ycamXMAS - upstream")
        if 0:
            if k == 0:
                p.figure(num=1, figsize=(8, 8))
            else:
                p.figure(1)
            p.plot(xy1[:, 0], -xy1[:, 1], color1[k])
            p.xlim(0.0, 2048.0)
            p.ylim(-2048.0, 0.0)
            p.xlabel("xcam XMAS - toward back")
            p.ylabel("-ycamXMAS - upstream")

    return 0


def read_dat(filedat, filetype="XMAS", flip_xaxis="no"):

    if filetype == "XMAS":
        print("reading peak list from XMAS DAT file : (fit case) \n", filedat)
    elif filetype == "LT":
        print("reading peak list from LaueTools DAT file : (fit case) \n", filedat)

    # lecture liste positions .DAT de XMAS
    # attribution des colonnes (cas du fit)
    # 0 , 1 : xfit, yfit
    # 2, 3  :intens max, integr
    # 4 , 5, 6  : widthx, widthy, tilt
    # 7 : Rfactor du fit
    # 8 : rien
    # 9, 10 : x(centroid), y(centroid)

    # .DAT de LT
    # peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax

    # data_dat = scipy.io.array_import.read_array(filedat, columns=((0, -1)), lines=(1, -1))
    data_dat = loadtxt(filedat, skiprows=1)

    data_xyexp = data_dat[:, 0:2]
    data_int = data_dat[:, 2:4]

    # for VHR images Jun12
    if flip_xaxis == "yes":
        data_xyexp[:, 0] = 2594.0 - data_xyexp[:, 0]

    if filetype == "LT":
        data_Ipixmax = data_dat[:, -1]

    if filetype == "XMAS":
        # exchange the two intensity column for consistency with xmas spot sorting by intensities
        data_int2 = data_int * 1.0
        data_int2[:, 0] = data_int[:, 1]
        data_int2[:, 1] = data_int[:, 0]
        return (data_xyexp, data_int2)
    elif filetype == "LT":
        return (data_xyexp, data_int, data_Ipixmax)


def test_LT(filestr, elem_label):

    # use full peak list (multigrain) from XMAS STR file
    # use guess matrixes from STR file for multigrain indexation, after orthonormalizing

    min_matLT = True
    data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(
        filestr, min_matLT=min_matLT
    )

    xyII = (
        vstack(
            (
                data_str_all[:, 0],
                data_str_all[:, 1],
                data_str_all[:, 9],
                data_str_all[:, 10],
            )
        )
    ).transpose()
    print(shape(xyII))
    # xyII = xyII[:npeaks[0],:]
    # print shape(xyII)
    # ajout d'une ligne de zeros pour simuler le header du .DAT classique
    toto = zeros((1, 4), float)
    xyII = vstack((toto, xyII))
    filedat = filestr.rstrip(".str") + "_from_str.dat"
    savetxt(filedat, xyII, fmt="%.4f")

    ngrains = len(npeaks)
    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(ngrains):
        if npeaks[i] > 10:
            matstarlabOND = matstarlab_to_matstarlabOND(matstarlab[i, :])
            matstarlab1 = matstarlabOND * 1.0
            # matstarlab1 =matstarlab[i,:]*1.0
            matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
            paramdetector_top = list(calib)
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = test_index_refine(
                filedat,
                paramdetector_top,
                use_weights=False,
                proposed_matrix=matLT3x3,
                check_grain_presence=None,
                paramtofit="strain",
                elem_label=elem_label,
                grainnum=i + 1,
            )
            filelist.append(filefit)

    merge_fit_files_multigrain(filelist)
    print("min_matLT = ", min_matLT)
    print("orthonormalize matstarlab before indexation")
    print(npeaks)
    print(npeaks_LT)
    print(dev_str[:, 2])
    print(pixdev_LT)

    return 0


def test_LT_2(filestr, elem_label):

    # use full peak list (multigrain) from XMAS STR file
    # no initial guess for indexation

    min_matLT = True
    data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(
        filestr, min_matLT=min_matLT
    )
    header = "xexp yexp integr intens\n"
    filedat = filestr.rstrip(".str") + "_from_str.dat"

    data_xy = data_str_all[:, :2]
    data_int2 = data_str_all[:, 9:11]
    data_int = data_int2 * 1.0
    data_int[:, 0] = data_int2[:, 1]
    data_int[:, 1] = data_int2[:, 0]

    if 1:
        # use if 0 : to avoid reindexing already indexed grains
        xyII = column_stack((data_xy, data_int))
        xyII = sort_peaks_decreasing_int(xyII, 3)
        print(shape(xyII))
        outputfile = open(filedat, "w")
        outputfile.write(header)
        np.savetxt(outputfile, xyII, fmt="%.4f")
        outputfile.close()
    else:
        xyII = loadtxt(filedat, skiprows=1)

    ngrains = 10
    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(
        ngrains
    ):  # adjust range to keep .fit files of already indexed grains
        # for i in range(6,10):
        paramdetector_top = list(calib)
        res1 = test_index_refine(
            filedat,
            paramdetector_top,
            use_weights=False,
            proposed_matrix=None,
            check_grain_presence=1,
            paramtofit="strain",
            elem_label=elem_label,
            grainnum=i + 1,
        )

        if res1 != 0:
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
            filelist.append(filefit)
            matstarlab_LT[i, :], data_fit = F2TC.readlt_fit(filefit)
            nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(
                xyII[:, :2], data_fit[:, -2:]
            )
            ind1 = where(iscommon1 == 0)

            xyII = xyII[ind1[0], :]
            outputfile = open(filedat, "w")
            outputfile.write(header)
            np.savetxt(outputfile, xyII, fmt="%.4f")
            outputfile.close()
            print(shape(xyII))
        else:
            break

    merge_fit_files_multigrain(filelist)
    print(npeaks)
    print(npeaks_LT)
    print(dev_str[:, 2])
    print(pixdev_LT)

    return 0


# #def test_LT_3(filedat1, elem_label, filestr = None, filefit = None, filedattype = "XMAS"):
# #
# #        # use peak list from peak search (XMAS or LT)
# #        # use calib either from XMAS STR file or from LT fit file
# #        # no initial guess for indexation
# #
# #        min_matLT = True
# #        if filestr != None :
# #                data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(filestr, min_matLT = min_matLT)
# #        elif filefit != None :
# #                matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefit, readmore = True)
# #
# #        if filedattype == "XMAS" :
# #                data_xy, data_int = read_dat(filedat1, filetype = "XMAS")
# #                header = "xexp yexp integr intens \n"
# #        elif filedattype == "LT" :
# #                data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype = "LT")
# #                header = "xexp yexp Ipeak Isub \n"
# #
# #
# #        filedat = filedat1.rstrip(".dat") + "_from_dat.dat"
# #
# #        if 1:
# #                # use if 0 : to avoid reindexing already indexed grains
# #                xyII = column_stack((data_xy, data_int))
# #                xyII =  sort_peaks_decreasing_int(xyII,3)
# #                print shape(xyII)
# #                outputfile = open(filedat,'w')
# #                outputfile.write(header)
# #                np.savetxt(outputfile,xyII,fmt = '%.4f')
# #                outputfile.close()
# #        else :
# #                xyII = loadtxt(filedat, skiprows = 1)
# #
# #        ngrains = 10
# #        npeaks_LT = zeros(ngrains,int)
# #        pixdev_LT = zeros(ngrains,float)
# #        matstarlab_LT = zeros((ngrains,9), float)
# #        filelist = []
# #        for i in range(ngrains):
# #        #for i in range(5,10):    # adjust range to keep .fit files of already indexed grains
# #                paramdetector_top = list(calib)
# #                res1 = test_index_refine(filedat, paramdetector_top,\
# #                                use_weights = False, proposed_matrix = None, \
# #                              check_grain_presence = 1, paramtofit = "strain",\
# #                                        elem_label = elem_label, grainnum = i+1,
# #                                      remove_sat = 0, elim_worst_pixdev = 1, maxpixdev = 1.0,
# #                                      spot_index_central = [0,1,2,3,4,5,6,7,8,9,10], nbmax_probed = 20, energy_max = 22,
# #                                      rough_tolangle = 0.5 ,fine_tolangle = 0.2, Nb_criterium = 20,
# #                                      NBRP = 1)
# #
# #                if res1 != 0 :
# #                        filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
# #                        filelist.append(filefit)
# #                        matstarlab_LT[i,:], data_fit = F2TC.readlt_fit(filefit)
# #                        nb_common_peaks,iscommon1,iscommon2 = find_common_peaks(xyII[:,:2], data_fit[:,-2:])
# #                        ind1 = where(iscommon1 == 0)
# #
# #                        xyII = xyII[ind1[0],:]
# #                        outputfile = open(filedat,'w')
# #                        outputfile.write(header)
# #                        np.savetxt(outputfile,xyII,fmt = '%.4f')
# #                        outputfile.close()
# #                        print shape(xyII)
# #                else :
# #                        break
# #
# #        merge_fit_files_multigrain(filelist, removefiles = 0)
# #        if filestr != None :
# #                print npeaks
# #        print npeaks_LT
# #        if filestr != None :
# #                print dev_str[:,2]
# #        print pixdev_LT
# #
# #    return(0)


def test_LT_calib(filestr, elem_label, filedat1=None, filedattype="LT"):

    # use peak list either from XMAS STR file or from XMAS / LT dat file
    # use guess calib from XMAS STR file
    # use guess matrix from STR file for indexation, after orthonormalizing

    min_matLT = True
    data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(
        filestr, min_matLT=min_matLT
    )
    matstarlab = matstarlab[0, :]
    print(shape(matstarlab))

    if filedat1 == None:
        filedat = filestr.rstrip(".str") + "_from_str.dat"
        xyII = column_stack((data_str_all[:, :2], data_str_all[:, 9:11]))
    else:
        filedat = filedat1.rstrip(".dat") + "_from_dat.dat"
        if filedattype == "XMAS":
            data_xy, data_int = read_dat(filedat1, filetype="XMAS")
            header = "xexp yexp integr intens \n"
        elif filedattype == "LT":
            data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype="LT")
            header = "xexp yexp Ipeak Isub \n"
        xyII = column_stack((data_xy, data_int))

    print(shape(xyII))

    outputfile = open(filedat, "w")
    outputfile.write(header)
    np.savetxt(outputfile, xyII, fmt="%.4f")
    outputfile.close()

    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
    matstarlab1 = matstarlabOND * 1.0
    # matstarlab1 =matstarlab[i,:]*1.0
    matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
    paramdetector_top = list(calib)
    filefit, filecor, npeaks_LT, pixdev_LT, matLTmin, calibLT = test_index_refine(
        filedat,
        paramdetector_top,
        use_weights=False,
        proposed_matrix=matLT3x3,
        check_grain_presence=None,
        paramtofit="calib",
        elem_label=elem_label,
        grainnum=1,
    )

    print("min_matLT = ", min_matLT)
    print("orthonormalize matstarlab before indexation")
    print_calib(calib)
    print_calib(calibLT)
    print(npeaks)
    print(npeaks_LT)
    print(dev_str[:, 2])
    print(pixdev_LT)
    return 0


def test_LT_4(filedat1, elem_label, filestr=None, filefit=None, filedattype="XMAS"):

    # use peak list from peak search (XMAS or LT)
    # use calib either from XMAS STR file or from LT fit file
    # no initial guess for indexation
    # add column isbadspot at end of .dat and .cor files to eliminate "intense grouped spots" from starting set
    # in test_index_refine

    # look for a maximum of ngrains grains
    ngrains = 10

    min_matLT = True
    if filestr != None:
        data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(
            filestr, min_matLT=min_matLT
        )
    elif filefit != None:
        matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefit, readmore=True)

    if filedattype == "XMAS":
        data_xy, data_int = read_dat(filedat1, filetype="XMAS")
        header = "xexp yexp integr intens isbadspot \n"
    elif filedattype == "LT":
        data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype="LT")
        header = "xexp yexp Ipeak Isub isbadspot \n"

    filedat = filedat1.rstrip(".dat") + "_from_dat.dat"

    nspots = shape(data_xy)[0]
    isbadspot = zeros(nspots, int)
    if 1:
        # use if 0 : to avoid reindexing already indexed grains
        xyII = column_stack((data_xy, data_int, isbadspot))
        xyII = sort_peaks_decreasing_int(xyII, 3)
        print(shape(xyII))
        outputfile = open(filedat, "w")
        outputfile.write(header)
        np.savetxt(outputfile, xyII, fmt="%.4f")
        outputfile.close()
    else:
        xyII = loadtxt(filedat, skiprows=1)

    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(ngrains):
        # for i in range(7,10):    # adjust range to keep .fit files of already indexed grains
        paramdetector_top = list(calib)
        res1 = test_index_refine(
            filedat,
            paramdetector_top,
            use_weights=False,
            proposed_matrix=None,
            check_grain_presence=1,
            paramtofit="strain",
            elem_label=elem_label,
            grainnum=i + 1,
            remove_sat=0,
            elim_worst_pixdev=1,
            maxpixdev=1.0,
            spot_index_central=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            nbmax_probed=20,
            energy_max=22,
            rough_tolangle=0.5,
            fine_tolangle=0.2,
            Nb_criterium=20,
            NBRP=1,
            mark_bad_spots=1,
        )

        if res1 != 0:
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
            filelist.append(filefit)
            matstarlab_LT[i, :], data_fit = F2TC.readlt_fit(filefit)
            first_indexed = int(data_fit[0, 0])
            # spots before first_indexed marked as bad
            xyII[:first_indexed, -1] = 1
            nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(
                xyII[:, :2], data_fit[:, -2:]
            )
            ind1 = where(iscommon1 == 0)
            xyII = xyII[ind1[0], :]
            xyII = sort_peaks_decreasing_int(xyII, 3)
            outputfile = open(filedat, "w")
            outputfile.write(header)
            np.savetxt(outputfile, xyII, fmt="%.4f")
            outputfile.close()
            print(shape(xyII))
        else:
            break

    merge_fit_files_multigrain(filelist, removefiles=0)
    if filestr != None:
        print(npeaks)
    print(npeaks_LT)
    if filestr != None:
        print(dev_str[:, 2])
    print(pixdev_LT)

    return 0


if 0:  # # test 4
    mat = array(
        [
            [0.17393303, 0.53947968, 0.82594746],
            [-0.78117538, 0.58454443, -0.21760663],
            [-0.59964349, -0.60638337, 0.52285059],
        ]
    )

    FindO.find_lowest_Euler_Angles_matrix(mat)

#### tests for comparing XMAS and Lauetools results for multigrain indexation and strain refinement

filepath = ".\\Examples\\strain_calib_test2\\"

if 0:  # ## test 5
    filestr = filepath + "W_v3c.str"
    convert_xmas_str_to_LT_fit(filestr)

    filestr = filepath + "Ge_v1b.str"
    convert_xmas_str_to_LT_fit(filestr)

    jkldsq


if 0:  # ## test 6
    filestr = filepath + "Ge_v1b.str"
    test_LT_calib(filestr, "Ge")
    hjsdqkqs

if 0:  # ## test 6b
    filefitGe1 = filepath + "Ge_v1b_from_str_UWN_1.fit"
    filefitGe2 = filepath + "Ge_v1b_from_xmas_1.fit"
    compare_multigrain_fit(
        filefitGe1, filefitGe2, mat_tol=5.0e-3, dxytol=0.01, compare_calib=1
    )
    jkldqs

if 0:  # ## test 7
    filestr = filepath + "Ge_v1b.str"
    filedatLTGe = filepath + "Ge_WB_14sep_d0_500MPa_ave_0to9_LT_0.dat"
    test_LT_calib(filestr, "Ge", filedat1=filedatLTGe, filedattype="LT")
    jkldsq

if 0:  # ## test 7b
    filefitGe1 = filepath + "Ge_WB_14sep_d0_500MPa_ave_0to9_LT_0_from_dat_UWN_1.fit"
    filefitGe2 = filepath + "Ge_v1b_from_xmas_1.fit"
    compare_multigrain_fit(
        filefitGe1, filefitGe2, mat_tol=5.0e-3, dxytol=1.0, compare_calib=1
    )
    jklqsdqs

if 0:  # ## test 8
    filestr = filepath + "W_v3c.str"
    test_LT(filestr, "W")
    filefitW1 = filepath + "W_v3c_from_str_UWN_mg.fit"
    filefitW1_new = filepath + "W_v3c_from_str_UWN_mg_guess.fit"
    os.rename(filefitW1, filefitW1_new)
    jklsdq

if 0:  # ## test 8b
    filefitW1 = filepath + "W_v3c_from_str_UWN_mg_guess.fit"
    filefitW2 = filepath + "W_v3c_from_xmas_mg.fit"
    compare_multigrain_fit(
        filefitW1, filefitW2, mat_tol=5.0e-3, dxytol=0.01, compare_calib=0
    )
    jdslqqds
# #
if 0:  # ## test 9
    filestr = filepath + "W_v3c.str"
    test_LT_2(filestr, "W")
    filefitW1 = filepath + "W_v3c_from_str_UWN_mg.fit"
    filefitW1_new = filepath + "W_v3c_from_str_UWN_mg_noguess.fit"
    os.rename(filefitW1, filefitW1_new)
    jklsdq

if 0:  # ## test 9b
    filefitW1 = filepath + "W_v3c_from_str_UWN_mg_noguess.fit"
    filefitW2 = filepath + "W_v3c_from_xmas_mg.fit"
    compare_multigrain_fit(
        filefitW1, filefitW2, mat_tol=5.0e-3, dxytol=0.1, compare_calib=0
    )
    jdslqqds

if 0:  # test 10
    filestr = filepath + "W_v3c.str"
    filedatxmas = filepath + "W_v3.dat"
    test_LT_4(filedatxmas, "W", filestr=filestr, filedattype="XMAS")
    jkldqssqd
# #
if 0:  # ## test 10b
    filefitW1 = filepath + "W_v3_from_dat_UWN_mg.fit"
    filefitW2 = filepath + "W_v3c_from_xmas_mg.fit"
    compare_multigrain_fit(
        filefitW1, filefitW2, mat_tol=5.0e-3, dxytol=0.1, compare_calib=0
    )
    jkldsq

if 0:  # ## test 11
    filestr = filepath + "W_v3c.str"
    filedatLT = filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0.dat"
    test_LT_4(filedatLT, "W", filestr=filestr, filedattype="LT")
    filefitW1 = filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0_from_dat_UWN_mg.fit"
    filefitW1_new = (
        filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0f_from_dat_UWN_mg_calibXMAS.fit"
    )
    os.rename(filefitW1, filefitW1_new)
    jklsdqjsqdl

if 0:  # ## test 11b
    filefitW1 = (
        filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0f_from_dat_UWN_mg_calibXMAS.fit"
    )
    filefitW2 = filepath + "W_v3c_from_xmas_mg.fit"
    compare_multigrain_fit(
        filefitW1, filefitW2, mat_tol=5.0e-3, dxytol=1.0, compare_calib=0
    )
    jklfdsjkl

if 0:  # ## test 12
    filefitGe1 = filepath + "Ge_WB_14sep_d0_500MPa_ave_0to9_LT_0_from_dat_UWN_1.fit"
    filedatLT = filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0.dat"
    test_LT_4(filedatLT, "W", filefit=filefitGe1, filedattype="LT")
    filefitW1 = filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0_from_dat_UWN_mg.fit"
    filefitW1_new = (
        filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0f_from_dat_UWN_mg_calibLT.fit"
    )
    os.rename(filefitW1, filefitW1_new)
    jklsdqjsqdl

if 0:  # ## test 12b
    filefitW1 = (
        filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0f_from_dat_UWN_mg_calibLT.fit"
    )
    filefitW2 = filepath + "W_v3c_from_xmas_mg.fit"
    compare_multigrain_fit(
        filefitW1, filefitW2, mat_tol=5.0e-3, dxytol=1.0, compare_calib=1
    )
    jklfdsjkl

if 0:  # ## test 12c
    filedatLT2 = filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0_from_dat.dat"
    data_xy, data_int, data_Ipixmax = read_dat(filedatLT2, filetype="LT")
    p.figure(figsize=(8, 8))
    p.plot(data_xy[:, 0], -data_xy[:, 1], "ro")
    p.plot(data_xy[:20, 0], -data_xy[:20, 1], "kx", markersize=20, markeredgewidth=3)
    p.xlim(0.0, 2048.0)
    p.ylim(-2048.0, 0.0)
    p.xlabel("xcam XMAS - toward back")
    p.ylabel("-ycamXMAS - upstream")

    jklfsd

# use p.show() in ipython to display plots
# use pc() to close plots


if 0:  # to merge files after multi-grain indexation "by hand"
    filelist = []
    for i in range(1, 8):
        filelist.append(
            filepath
            + "Wmap_WB_14sep_d0_500MPa_0045_LT_0_from_dat_UWN_"
            + str(i)
            + ".fit"
        )
    print(filelist)
    merge_fit_files_multigrain(filelist, removefiles=0)
    jksqld

# uses a number of invisible parameters set in param.py
def serial_peak_search(filepathim, fileprefix, indimg, filesuffix, filepathout):

    print("peak search in series of images (or single image)")

    npeaks = zeros(len(indimg), int)

    commentaire = (
        "LT rev "
        + PAR.LT_REV_peak_search
        + "\n# PixelNearRadius = "
        + str(PAR.PixelNearRadius)
        + "\n# IntensityThreshold = "
        + str(PAR.IntensityThreshold)
        + "\n# boxsize = "
        + str(PAR.boxsize)
        + "\n# position_definition = "
        + str(PAR.position_definition)
        + "\n# fit_peaks_gaussian = "
        + str(PAR.fit_peaks_gaussian)
        + "\n# xtol = "
        + str(PAR.xtol)
        + "\n# FitPixelDev = "
        + str(PAR.FitPixelDev)
        + "\n# local_maxima_search_method = "
        + str(PAR.local_maxima_search_method)
        + "\n# Threshold Convolve = "
        + str(PAR.thresholdConvolve)
        + "\n"
    )

    print("peak search parameters :")
    print(commentaire)

    k = 0
    for i in indimg:
        print("i = ", i)
        # filename = filelist[i]
        filename = os.path.join(
            filepathim,
            fileprefix
            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
            + filesuffix,
        )
        print("image in :")
        print(filename)
        print("saving peak list in :")
        fileprefix1 = os.path.join(
            filepathout,
            fileprefix + rmccd.stringint(i, PAR.number_of_digits_in_image_name),
        )
        filedat = (
            fileprefix + rmccd.stringint(i, PAR.number_of_digits_in_image_name) + ".dat"
        )
        import module_graphique as modgraph

        modgraph.savdatpeak = os.path.join(filepathout, filedat)
        # print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_peak_search:
            while filedat in os.listdir(filepathout):
                print("warning : change name to avoid overwrite")
                fileprefix2 = (
                    fileprefix
                    + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
                    + "_new_"
                    + str(j)
                )
                filedat = fileprefix2 + ".dat"
                import module_graphique as modgraph

                modgraph.savdatpeak = os.path.join(filepathout, filedat)
                print(filepathout + filedat)
                j = j + 1

        if j > 0:
            fileprefix1 = filepathout + fileprefix2
        else:
            print(filepathout + filedat)

        Isorted, fitpeak, localpeak, to_reject3 = rmccd.PeakSearch(
            filename,
            CCDLabel=PAR.CCDlabel,
            PixelNearRadius=PAR.PixelNearRadius,
            IntensityThreshold=PAR.IntensityThreshold,
            boxsize=PAR.boxsize,
            position_definition=PAR.position_definition,
            verbose=1,
            fit_peaks_gaussian=PAR.fit_peaks_gaussian,
            xtol=PAR.xtol,
            return_histo=0,
            FitPixelDev=PAR.FitPixelDev,
            local_maxima_search_method=PAR.local_maxima_search_method,
            thresholdConvolve=PAR.thresholdConvolve,
            Saturation_value=DictLT.dict_CCD[PAR.CCDlabel][2],
            Saturation_value_flatpeak=DictLT.dict_CCD[PAR.CCDlabel][2],
        )
        npeaks[k] = shape(Isorted)[0]

        if shape(Isorted)[0] > 0:

            IOLT.writefile_Peaklist(
                fileprefix1,
                Isorted,
                overwrite=1,
                initialfilename=filename,
                comments=commentaire,
            )
        k = k + 1

    print("indimg ", indimg)
    print("npeaks ", npeaks)
    return Isorted, localpeak, to_reject3


if 0:  #### test 13 serial peak search

    filesuffix = ".mccd"
    indimg = [45]
    fileprefix = "Wmap_WB_14sep_d0_500MPa_"
    filepathout = filepath
    filepathim = filepath
    serial_peak_search(filepathim, fileprefix, indimg, filesuffix, filepathout)
    jklsdq

# uses a number of invisible parameters set in param.py
def index_refine_multigrain_one_image(
    filedat1, elem_label, filefitcalib, ngrains=10, proposed_matrix=None
):

    # use peak list from peak search (LT)
    # use calib from LT fit file
    # no initial guess for indexation
    # add column isbadspot at end of .dat and .cor files to eliminate "intense grouped spots" from starting set
    # in test_index_refine
    # look for a maximum of ngrains grains

    matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(
        filefitcalib, readmore=True, verbose=1
    )

    data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype="LT")
    header = "xexp yexp Ipeak Isub isbadspot \n"

    filedat = filedat1.rstrip(".dat") + "_t.dat"

    nspots = shape(data_xy)[0]
    isbadspot = zeros(nspots, int)
    if 1:
        # use if 0 : to avoid reindexing already indexed grains
        xyII = column_stack((data_xy, data_int, isbadspot))
        xyII = sort_peaks_decreasing_int(xyII, 3)
        print(shape(xyII))
        outputfile = open(filedat, "w")
        outputfile.write(header)
        np.savetxt(outputfile, xyII, fmt="%.4f")
        outputfile.close()
    else:
        xyII = loadtxt(filedat, skiprows=1)

    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(ngrains):
        # for i in range(2,ngrains):    # adjust range to keep .fit files of already indexed grains
        paramdetector_top = list(calib)
        res1 = test_index_refine(
            filedat,
            paramdetector_top,
            use_weights=False,
            proposed_matrix=proposed_matrix,
            check_grain_presence=PAR.check_grain_presence,
            paramtofit="strain",
            elem_label=elem_label,
            grainnum=i + 1,
            remove_sat=PAR.remove_sat,
            elim_worst_pixdev=PAR.elim_worst_pixdev,
            maxpixdev=PAR.maxpixdev,
            spot_index_central=PAR.spot_index_central,
            nbmax_probed=PAR.nbmax_probed,
            energy_max=PAR.energy_max,
            rough_tolangle=PAR.rough_tolangle,
            fine_tolangle=PAR.fine_tolangle,
            Nb_criterium=PAR.Nb_criterium,
            NBRP=PAR.NBRP,
            mark_bad_spots=PAR.mark_bad_spots,
            pixelsize=DictLT.dict_CCD[PAR.CCDlabel][1],
            dim=DictLT.dict_CCD[PAR.CCDlabel][0],
        )
        if res1 != 0:
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
            filelist.append(filefit)
            print(filelist)
            print(filefit)
            matstarlab_LT[i, :], data_fit = F2TC.readlt_fit(filefit)
            print("numbers of indexed spots : ", data_fit[:, 0])
            first_indexed = int(data_fit[0, 0])
            # spots before first_indexed marked as bad
            xyII[:first_indexed, -1] = 1
            #            print "xyII"
            #            print xyII[0,:]
            #            print xyII[:, :2]
            #            print "data_fit"
            #            print data_fit[0,:]
            #            print  data_fit[:, 6:8]
            nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(
                xyII[:, :2], data_fit[:, 6:8], verbose=0
            )
            ind1 = where(iscommon1 == 0)
            #            print ind1
            if ngrains > 1:
                # print xyII[:, :2]
                # print data_fit[:, -2:]
                # print ind1
                xyII = xyII[ind1[0], :]
                xyII = sort_peaks_decreasing_int(xyII, 3)
                outputfile = open(filedat, "w")
                outputfile.write(header)
                np.savetxt(outputfile, xyII, fmt="%.4f")
                outputfile.close()
                print(shape(xyII))
        else:
            break

    ngrains_found = i

    filefitmg = merge_fit_files_multigrain(filelist, removefiles=1)

    # if ngrains==1 : pixdev_LT.reshape(len(pixdev_LT),1)
    print(npeaks_LT)
    print(pixdev_LT)
    # print filefitmg, ngrains_found, npeaks_LT, pixdev_LT
    return (filefitmg, ngrains_found, npeaks_LT, pixdev_LT)


# uses a number of invisible parameters set in param.py
def serial_indexerefine_multigrain(
    filepathdat,
    fileprefix,
    indimg,
    filesuffix,
    filefitcalib,
    filepathout,
    filefitref=None,
):

    nimg = len(indimg)
    ngrains_found = zeros(nimg, int)
    npeaks = zeros((nimg, PAR.ngrains_index_refine), int)
    pixdev = zeros((nimg, PAR.ngrains_index_refine), float)

    proposed_matrix = None

    if filefitref != None:
        matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefitref, readmore=True)
        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarlab1 = matstarlabOND * 1.0

        # matstarlab1 =matstarlab[i,:]*1.0
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
        proposed_matrix = matLT3x3

    k = 0
    for i in indimg:
        print("i = ", i)
        # filename = filelist[i]
        filedat1 = os.path.join(
            filepathdat,
            fileprefix
            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
            + filesuffix,
        )
        print("image in :")
        print(filedat1)
        # print "saving fit in :"
        filefit = (
            fileprefix
            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
            + PAR.add_str_index_refine
            + ".fit"
        )
        import module_graphique as modgraph

        modgraph.savindexfit = os.path.join(filepathout, filefit)
        # print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_index_refine:
            while filefit in os.listdir(filepathout):
                print("warning : change name to avoid overwrite")
                filefit = (
                    fileprefix
                    + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
                    + PAR.add_str_index_refine
                    + "_new_"
                    + str(j)
                    + ".fit"
                )
                print(filepathout + filefit)
                import module_graphique as modgraph

                modgraph.savindexfit = os.path.join(filepathout, filefit)
                j = j + 1

        filefit_withdir = filepathout + filefit
        print(PAR.elem_label_index_refine)

        filefitmg, ngrains_found[k], npeaks[k, :], pixdev[
            k, :
        ] = index_refine_multigrain_one_image(
            filedat1,
            PAR.elem_label_index_refine,
            filefitcalib,
            ngrains=PAR.ngrains_index_refine,
            proposed_matrix=proposed_matrix,
        )

        if filepathout != filepathdat:
            print("filefitmg", filefitmg)
            print("filefit_withdir", filefit_withdir)
            os.rename(filefitmg, filefit_withdir)

        k = k + 1

        print("indimg ", indimg[:k])
        print("ngrains_found ", ngrains_found[:k])
        print("npeaks ", npeaks[:k, :])
        print("pixdev ", pixdev[:k, :])

    return (ngrains_found, npeaks, pixdev)


def serial_index_refine_multigrain_v2(
    filepathdat,
    fileprefix,
    indimg,
    filesuffix,
    filefitcalib,
    filepathout,
    filefitref=None,
    filefitmgref=None,
    gnumlocmgref=None,
    nLUT=3,
):

    # speed up : single calculation of LUT + single reading of filefitcalib

    latticeparams = DictLT.dict_Materials[PAR.elem_label_index_refine][1]
    Bmatrix = CP.calc_B_RR(latticeparams)
    LUT = INDEX.build_AnglesLUT(Bmatrix, nLUT)

    matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(
        filefitcalib, readmore=True, verbose=1
    )

    proposed_matrix = None

    if filefitref != None:
        matstarlab, data_fit, calib1, pixdev = F2TC.readlt_fit(
            filefitref, readmore=True
        )
        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarlab1 = matstarlabOND * 1.0

        # matstarlab1 =matstarlab[i,:]*1.0
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
        proposed_matrix = matLT3x3

    if (filefitmgref != None) & (gnumlocmgref != None):

        res1 = readlt_fit_mg(filefitmgref, verbose=1, readmore=True)
        #                print res1
        if res1 != 0:
            gnumlist, npeaks1, indstart, matstarlab_all, data_fit1, calib1, pixdev1, strain6, euler = (
                res1
            )

        matstarlab = matstarlab_all[gnumlocmgref]
        print(matstarlab)
        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarlab1 = matstarlabOND * 1.0

        # matstarlab1 =matstarlab[i,:]*1.0
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
        proposed_matrix = matLT3x3

    nimg = len(indimg)
    ngrains_found = zeros(nimg, int)
    npeaks = zeros((nimg, PAR.ngrains_index_refine), int)
    pixdev = zeros((nimg, PAR.ngrains_index_refine), float)

    k = 0
    for i in indimg:
        print("i = ", i)
        # filename = filelist[i]
        filedat1 = (
            filepathdat
            + fileprefix
            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
            + filesuffix
        )
        print("image in :")
        print(filedat1)
        # print "saving fit in :"
        filefit = (
            fileprefix
            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
            + PAR.add_str_index_refine
            + ".fit"
        )
        # print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_index_refine:
            while filefit in os.listdir(filepathout):
                print("warning : change name to avoid overwrite")
                filefit = (
                    fileprefix
                    + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
                    + PAR.add_str_index_refine
                    + "_new_"
                    + str(j)
                    + ".fit"
                )
                print(filepathout + filefit)
                j = j + 1

        filefit_withdir = filepathout + filefit
        print(PAR.elem_label_index_refine)

        res1 = index_refine_multigrain_one_image_with_twins(
            filedat1,
            calib,
            ngrains=PAR.ngrains_index_refine,
            proposed_matrix=proposed_matrix,
        )

        print("k = ", k)
        if PAR.ngrains_index_refine > 1:
            filefitmg, ngrains_found[k], npeaks[k, :], pixdev[k, :] = res1
        else:
            filefitmg, ngrains_found[k], npeaks[k], pixdev[k] = res1

        if filefitmg != None:
            if filepathout != filepathdat:
                os.rename(filefitmg, filefit_withdir)

        k = k + 1

        if nimg < 50:
            print("indimg ", indimg[:k])
            print("ngrains_found ", ngrains_found[:k])
            print("npeaks ", npeaks[:k])
            print("pixdev ", pixdev[:k])

    return (ngrains_found, npeaks, pixdev)


if 0:  # ## test 14 : serial index refine multigrain
    indimg = list(range(5))
    # indimg = [45,]
    fileprefix = "Wmap_WB_14sep_d0_500MPa_"
    filepathdat = filepath
    filepathout = filepath
    filefitcalib = filepath + "Ge_WB_14sep_d0_500MPa_ave_0to9_LT_0_from_dat_UWN_1.fit"
    filesuffix = ".dat"

    ngrains_found, npeaks, pixdev = serial_indexerefine_multigrain(
        filepathdat, fileprefix, indimg, filesuffix, filefitcalib, filepathout
    )

    jkldasdsa

# uses a number of invisible parameters set in param.py
def index_refine_calib_one_image(
    filedat1,
    filedet=None,
    filefitcalib=None,
    boolctrl="1" * 8,
    fixedcalib=None,
    elim_worst_pixdev=1,
    maxpixdev=0.7,
    elem_label="Ge",
    CCD_label="MARCCD165",
):

    if filedet != None:
        calib, matstarlab = F2TC.readlt_det(filedet)
    if filefitcalib != None:
        matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(
            filefitcalib, readmore=True
        )

    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
    matstarlab1 = matstarlabOND * 1.0
    # matstarlab1 =matstarlab[i,:]*1.0
    matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
    if fixedcalib == None:
        paramdetector_top = list(calib)
    else:
        paramdetector_top = list(fixedcalib)
    filefit, filecor, npeaks_LT, pixdev_LT, matLTmin, calibLT = (
        res1
    ) = test_index_refine(
        filedat1,
        paramdetector_top,
        use_weights=False,
        proposed_matrix=matLT3x3,
        check_grain_presence=None,
        paramtofit="calib",
        elem_label=elem_label,
        grainnum=1,
        remove_sat=PAR.remove_sat_calib,
        elim_worst_pixdev=elim_worst_pixdev,
        maxpixdev=maxpixdev,
        spot_index_central=PAR.spot_index_central_calib,
        nbmax_probed=PAR.nbmax_probed_calib,
        energy_max=PAR.energy_max_calib,
        rough_tolangle=PAR.rough_tolangle_calib,
        fine_tolangle=PAR.fine_tolangle_calib,
        Nb_criterium=PAR.Nb_criterium_calib,
        NBRP=PAR.NBRP_calib,
        boolctrl=boolctrl,
        CCD_label=CCD_label,
    )

    return (filefit, npeaks_LT, pixdev_LT)


# uses a number of invisible parameters set in param.py
def serial_index_refine_calib(
    filepathdat,
    fileprefix,
    indimg,
    filesuffix,
    filefitcalib,
    filepathout,
    boolctrl="1" * 8,
    fixedcalib=None,
    fixed_filefitcalib="yes",
    number_of_digits_in_image_name=4,
    add_str_index_refine="_t_UWN",
):  # string to add : UWN = "use weights no")
    nimg = len(indimg)
    npeaks = zeros(nimg, int)
    pixdev = zeros(nimg, float)

    k = 0
    filefitcalib_local = filefitcalib

    for i in indimg:
        print("i = ", i)
        # filename = filelist[i]
        filedat1 = (
            filepathdat
            + fileprefix
            + rmccd.stringint(i, number_of_digits_in_image_name)
            + filesuffix
        )
        print("image in :")
        print(filedat1)
        # data_dat = loadtxt(filedat1, skiprows = 1)
        # print "saving fit in :"
        filefit = (
            fileprefix
            + rmccd.stringint(i, number_of_digits_in_image_name)
            + PAR.add_str_index_refine
            + ".fit"
        )
        # print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_index_refine:
            while filefit in os.listdir(filepathout):
                print("warning : change fit file name to avoid overwrite")
                filefit = (
                    fileprefix
                    + rmccd.stringint(i, number_of_digits_in_image_name)
                    + add_str_index_refine
                    + "_new_"
                    + str(j)
                    + ".fit"
                )
                print(filepathout + filefit)
                j = j + 1

        filefit_withdir = filepathout + filefit

        filefit, npeaks[k], pixdev[k] = index_refine_calib_one_image(
            filedat1,
            filefitcalib=filefitcalib_local,
            boolctrl=boolctrl,
            fixedcalib=fixedcalib,
        )
        print(filefit_withdir)
        if filepathout != filepathdat:
            os.rename(filefit, filefit_withdir)

        if fixed_filefitcalib == "no":
            # use result of fit as guess for next indexation
            filefitcalib_local = filefit_withdir

        k = k + 1

        print("indimg ", indimg[:k])
        print("npeaks ", npeaks[:k])
        print("pixdev ", pixdev[:k].round(decimals=3))

    return (npeaks, pixdev)


if 0:  # # test 15
    filepath = "/users/gonio/dataVHR/32.2.748/21Feb13/Si1g_50N//"
    # filedet = filepath + "Ge//Ge1_0000_mar_LT_1.det"
    # filedat1 = filepath + "Ge//Ge1_0000_mar_LT_1.dat"
    # index_refine_calib_one_image(filedat1, "Ge", filedet=filedet)

    filepathdat = os.path.join(filepath, "scan//datfiles//")
    filepathout = os.path.join(filepath, "scan//fitfiles//")
    fileprefix = "S1gscan_"
    # indimg = arange(7,291,1)
    indimg = arange(15, 17, 1)
    filesuffix = ".dat"
    filefitcalib = os.path.join(filepath, "Ge//Ge1_0000_mar_LT_1_t_UWN_1.fit")

    filefitref = os.path.join(filepathout, "S1gscan_0008_t_UWN_mg_ref.fit")

    ngrains_found, npeaks, pixdev = serial_indexerefine_multigrain(
        filepathdat,
        fileprefix,
        indimg,
        filesuffix,
        filefitcalib,
        filepathout,
        filefitref=None,
    )

    jkldas


def filter_peaks(filedat, maxpixdev=0.7):

    k = 0
    filedat2 = filedat.split(".dat")[0] + "_mpd.dat"

    outputfile = open(filedat2, "w")
    f = open(filedat, "r")
    try:
        for line in f:
            if (line[0] == "p") | (line[0] == "#"):
                outputfile.write(line)
            else:
                # print line
                toto = np.array(line.rstrip(PAR.cr_string).split(), dtype=float)
                if norme(toto[7:9]) < maxpixdev:
                    outputfile.write(line)
                else:
                    print("reject line : \n", line)
                    k = k + 1

    finally:
        f.close()
    outputfile.write("\n")
    outputfile.close()

    print("input dat file : \n", filedat)
    print("filtered dat file: \n", filedat2)
    print("maxpixdev : ", maxpixdev)
    print("number of rejected lines : ", k)

    return filedat2


if 0:  # test filter_peaks
    filepath = (
        "D:\\Documents and Settings\\or208865\\Bureau\\mono_inverse\\Nov12\\Gedia1\\"
    )
    # filedat1 = filepath + "Ge1dia1_000000_LT_1.dat"
    # filedat1 = filepath + "Ge2dia1_002000_LT_0.dat"
    filedat1 = os.path.join(filepath, "Ge3dia1_000000_LT_0.dat")
    filter_peaks(filedat1)
    jsqdl


import struct

# uses invisible parameters from param.py
def get_xyzech(filepathim, fileprefix, indimg, filesuffix, filepathout):

    nimg = len(indimg)
    data_list = zeros((nimg, 6), float)
    fileim = ""

    # ROPER only - use hexedit to find hexadecimal location of stored floats

    kk = 0
    for k in indimg:

        data_list[kk, 0] = k
        print(filepathim)
        print(fileprefix)
        print(filesuffix)
        fileim = os.path.join(
            filepathim,
            fileprefix
            + rmccd.stringint(k, PAR.number_of_digits_in_image_name)
            + filesuffix,
        )
        print(fileim)

        f = open(fileim, "rb")

        # #            toto1 =""
        # #            f.seek(0x9B4)
        # #            for i in range(7) :
        # #                toto = struct.unpack("c",f.read(1))
        # #                toto1 = toto1 + toto[0]
        # #
        # #            print toto1
        # #            data_list[kk,5] = float(toto1)

        toto1 = ""
        print(type(toto1))
        f.seek(PAR.xech_offset)
        for i in range(7):
            toto = struct.unpack("c", f.read(1))
            toto1 = toto1 + toto[0]
        print(toto1)
        print(type(toto1))
        data_list[kk, 1] = float(toto1)

        toto1 = ""
        f.seek(PAR.yech_offset)
        for i in range(7):
            toto = struct.unpack("c", f.read(1))
            toto1 = toto1 + toto[0]
        print(toto1)
        data_list[kk, 2] = float(toto1)

        toto1 = ""
        f.seek(PAR.zech_offset)
        for i in range(7):
            toto = struct.unpack("c", f.read(1))
            toto1 = toto1 + toto[0]
        print(toto1)
        data_list[kk, 3] = float(toto1)

        toto1 = ""
        f.seek(PAR.xech_offset)
        for i in range(40):
            toto = struct.unpack("c", f.read(1))
            toto1 = toto1 + toto[0]
        print(toto1)
        print(toto1.split()[3])
        data_list[kk, 4] = float(toto1.split()[3])
        data_list[kk, 5] = float(toto1.split()[4])

        print("img, xech, yech, zech, mon4, lambda = \n", data_list[kk, :])

        kk = kk + 1

    print(data_list)

    header = "img 0 , xech 1, yech 2, zech 3, mon4 4, lambda 5 \n"

    try:
        import module_graphique as modgraph

        print(modgraph.outfilenamexyz)
        outfilename = os.path.join(
            modgraph.outfilenamexyz,
            fileprefix
            + str(modgraph.indimg[0])
            + "_to_"
            + str(modgraph.indimg[-1])
            + ".dat",
        )
        print(outfilename)
        outputfile = open(outfilename, "w")
        outputfile.write(header)
        np.savetxt(outputfile, data_list, fmt="%.4f")
        outputfile.close()
    except ImportError:
        print("Miss module_graphique.py")

    return outfilename


try:
    import module_graphique as modgraph
except ImportError:
    print("You have to import module_graphique.py")


def build_xy_list_by_hand(
    fileprefix,
    nx,
    ny,
    xfast,
    yfast,
    xstep,
    ystep,
    dirname=str(modgraph.outfilenamexy),
    startindex=modgraph.indimg[0],
    lastindex=modgraph.indimg[-1],
):
    """
    write a file with image index and x and y sample positions
    """
    nx = nx - 1
    ny = ny - 1

    if yfast:
        xylist = zeros((nx + 1, ny + 1, 2), float)
        nx = int(round(nx))
        ny = int(round(ny))
        for i in range(nx + 1):
            xylist[i, :, 0] = float(i) * xstep
        for j in range(ny + 1):
            xylist[:, j, 1] = float(j) * ystep

    if xfast:
        xylist = zeros((ny + 1, nx + 1, 2), float)
        nx = int(round(nx))
        ny = int(round(ny))
        for i in range(nx + 1):
            xylist[:, i, 0] = float(i) * xstep
        for j in range(ny + 1):
            xylist[j, :, 1] = float(j) * ystep

    xylist_new = xylist.reshape((nx + 1) * (ny + 1), 2)
    indimg = arange((nx + 1) * (ny + 1)) + startindex
    data_list = column_stack((indimg, xylist_new))

    print(data_list)

    header = "img 0 , xech 1, yech 2 \n"

    outputfilename = os.path.join(
        dirname, fileprefix + "%s_to_%s.dat" % (str(startindex), str(lastindex))
    )

    print("writing image index x,y sample position in:", outputfilename)

    outputfile = open(outputfilename, "w")
    outputfile.write(header)
    np.savetxt(outputfile, data_list, fmt="%.4f")
    outputfile.close()

    return outputfilename


def build_summary(
    fileindex_list,
    filepathfit,
    fileprefix,
    filesuffix,
    filexyz,
    startindex=modgraph.indimg[0],
    finalindex=modgraph.indimg[-1],
    number_of_digits_in_image_name=4,
    nbtopspots=10,
    outputprefix="_SUMMARY_",
    folderoutput=modgraph.outfilename,
    default_file=None,
):  # 29May13
    """
    write a file containing the sumary of results from a set .fit file
    fileindex_list: list of file index

    # mean local grain intensity is taken over the most intense ntopspots spots
    nbtopspots = 10 
    
    number_of_digits_in_image_name :  nb of 0 padded integer formatting
                                    example: for 4  , then 56 => 0056
                                    0 to simpliest integer formatting (not zero padding)
    """
    # filexyz : img 0 , xech 1, yech 2, zech 3, mon4 4, lambda 5
    total_nb_cols = 25

    list_col_names = ["dxymicrons", "matstarlab", "strain6_crystal", "euler3"]
    number_col_list = array([2, 9, 6, 3])

    list_col_names2 = ["img", "gnumloc", "npeaks", "pixdev", "intensity"]

    for k in range(len(list_col_names)):
        for nbcol in range(number_col_list[k]):
            lcn = list_col_names[k] + "_" + str(nbcol)
            list_col_names2 += [lcn]

    # print list_col_names2
    header2 = ""
    for i in range(total_nb_cols):
        header2 = header2 + list_col_names2[i] + " "
    header2 += "\n"
    print(header2)

    iloop = 0

    # read xyz position file
    posxyz = loadtxt(filexyz, skiprows=1)
    xy = posxyz[:, 1:3]
    imgxy = posxyz[:, 0]
    dxy = xy - xy[0, :]  # *1000.0

    list_files_in_folder = os.listdir(filepathfit)
    import re

    test = re.compile("\.fit$", re.IGNORECASE)
    list_fitfiles_in_folder = list(filter(test.search, list_files_in_folder))

    encodingdigits = "%%0%dd" % int(number_of_digits_in_image_name)
    # loop for reading each .fit file
    for fileindex in fileindex_list:
        ind0 = where(imgxy == fileindex)
        print("dxy = ", dxy[ind0[0], :])
        _filename = fileprefix + encodingdigits % fileindex + filesuffix

        if _filename not in list_fitfiles_in_folder:
            print("Warning! missing .fit file: %s" % _filename)
            res = zeros(total_nb_cols, float)
            res[0] = fileindex

            if iloop == 0:
                #                 print "res", res
                allres = res
            else:
                #                 print "res iloop diff 0",res
                allres = row_stack((allres, res))
            continue

        filefitmg = os.path.join(filepathfit, _filename)
        # print filefitmg

        # read .fit file
        #         res1 = readlt_fit_mg(filefitmg, verbose=1, readmore=True)

        res1 = readfitfile_multigrains(
            filefitmg,
            verbose=1,
            readmore=True,
            fileextensionmarker=".cor",
            default_file=default_file,
        )

        #         print "res1", res1

        if res1 != 0:
            (
                gnumlist,
                npeaks,
                indstart,
                matstarlab,
                data_fit,
                calib,
                pixdev,
                strain6,
                euler,
            ) = res1

            if len(pixdev) == 0:
                pixdev = np.zeros_like(gnumlist)

            ngrains = len(gnumlist)
            # print indstart
            intensity = zeros(ngrains, float)
            if ngrains > 1:
                for j in range(ngrains):
                    range1 = arange(indstart[j], indstart[j] + npeaks[j])
                    data_fit1 = data_fit[range1, :]
                    intensity[j] = data_fit1[:nbtopspots, 1].mean()
            else:
                intensity[0] = data_fit[:nbtopspots, 1].mean()
                strain6 = strain6.reshape(1, 6)
                euler = euler.reshape(1, 3)

            imnumlist = ones(ngrains, int) * fileindex
            dxylist = multiply(ones((ngrains, 2), float), dxy[ind0[0], :])
            # print dxylist

            # print shape(strain6)
            # print shape(imnumlist)
            # print shape(gnumlist)
            # print shape(dxylist)
            res = column_stack(
                (
                    imnumlist,
                    gnumlist,
                    npeaks,
                    pixdev,
                    intensity,
                    dxylist,
                    matstarlab,
                    strain6,
                    euler,
                )
            )

            # print imnumlist
            print("intensity in build_summary()", intensity)
            # print res
        else:
            res = zeros(total_nb_cols, float)
            res[0] = fileindex

            print("something is empty")

        if iloop == 0:
            allres = res
        else:
            allres = row_stack((allres, res))

        iloop += 1

    print("shape allres")
    print(shape(allres))
    print(folderoutput)
    print(
        fileprefix + "%s%s_to_%s.dat" % (outputprefix, str(startindex), str(finalindex))
    )

    header = "img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, strain6_crystal 16:22, euler 22:25  \n"

    try:
        import module_graphique as modgraph

        fullpath_summary_filename = os.path.join(
            folderoutput,
            fileprefix
            + "%s%s_to_%s.dat" % (outputprefix, str(startindex), str(finalindex)),
        )

        print("fullpath_summary_filename", fullpath_summary_filename)
        modgraph.filesumbeforecolumn = fullpath_summary_filename
        outputfile = open(fullpath_summary_filename, "w")
        outputfile.write(header)
        outputfile.write(header2)
        np.savetxt(outputfile, allres, fmt="%.6f")
        outputfile.close()

    except ImportError:
        print("Missing module_graphique.py")

    return allres, fullpath_summary_filename


def read_summary_file(
    filesum,
    read_all_cols="yes",
    list_column_names=[
        "img",
        "gnumloc",
        "npeaks",
        "pixdev",
        "intensity",
        "dxymicrons_0",
        "dxymicrons_1",
        "matstarlab_0",
        "matstarlab_1",
        "matstarlab_2",
        "matstarlab_3",
        "matstarlab_4",
        "matstarlab_5",
        "matstarlab_6",
        "matstarlab_7",
        "matstarlab_8",
        "strain6_crystal_0",
        "strain6_crystal_1",
        "strain6_crystal_2",
        "strain6_crystal_3",
        "strain6_crystal_4",
        "strain6_crystal_5",
        "euler3_0",
        "euler3_1",
        "euler3_2",
        "strain6_sample_0",
        "strain6_sample_1",
        "strain6_sample_2",
        "strain6_sample_3",
        "strain6_sample_4",
        "strain6_sample_5",
        "rgb_x_sample_0",
        "rgb_x_sample_1",
        "rgb_x_sample_2",
        "rgb_z_sample_0",
        "rgb_z_sample_1",
        "rgb_z_sample_2",
        "stress6_crystal_0",
        "stress6_crystal_1",
        "stress6_crystal_2",
        "stress6_crystal_3",
        "stress6_crystal_4",
        "stress6_crystal_5",
        "stress6_sample_0",
        "stress6_sample_1",
        "stress6_sample_2",
        "stress6_sample_3",
        "stress6_sample_4",
        "stress6_sample_5",
        "res_shear_stress_0",
        "res_shear_stress_1",
        "res_shear_stress_2",
        "res_shear_stress_3",
        "res_shear_stress_4",
        "res_shear_stress_5",
        "res_shear_stress_6",
        "res_shear_stress_7",
        "res_shear_stress_8",
        "res_shear_stress_9",
        "res_shear_stress_10",
        "res_shear_stress_11",
        "max_rss",
        "von_mises",
    ],
):

    # 29May13
    print("reading summary file")
    print("first two lines :")
    f = open(filesum, "r")
    i = 0
    try:
        for line in f:
            if i == 0:
                nameline0 = line.rstrip("  \n")
            if i == 1:
                nameline1 = line.rstrip("\n")
            i = i + 1
            if i > 2:
                break
    finally:
        f.close()

    print(nameline0)
    print(nameline1)
    listname = nameline1.split()

    data_sum = loadtxt(filesum, skiprows=2)

    if read_all_cols == "yes":
        print("shape(data_sum) = ", shape(data_sum))
        return (data_sum, listname, nameline0)

    else:
        print(len(listname))
        ncol = len(list_column_names)

        ind0 = zeros(ncol, int)

        for i in range(ncol):
            ind0[i] = listname.index(list_column_names[i])

        print(ind0)

        data_sum_select_col = data_sum[:, ind0]

        print(shape(data_sum))
        print(shape(data_sum_select_col))
        print(filesum)
        print(list_column_names)
        print(data_sum_select_col[:5, :])

        return (data_sum_select_col, list_column_names, nameline0)


def twomat_to_rotation_Emeric(matstarlab1, matstarlab2, omega0=40.0):

    # utilise matstarlab

    # version Emeric nov 13
    matref = matstarlab_to_matdirONDsample3x3(matstarlab1, omega0=omega0)
    matmes = matstarlab_to_matdirONDsample3x3(matstarlab2, omega0=omega0)

    # ATTENTION : Orthomormalisation avant de faire le calcul
    # matmisor = dot(np.linalg.inv(matref.transpose()),matmes.transpose())
    matmisor = dot(matref, matmes.T)  # cf cas CK

    toto = (matmisor[0, 0] + matmisor[1, 1] + matmisor[2, 2] - 1.0) / 2.0
    # 2 + 2* toto = 2 + trace - 1 =  1 + trace

    # theta en rad
    theta = np.arccos(toto)

    # Cas pathologique de theta=0 => vecteur == 1 0 0
    # to complete

    # Sinon

    toto1 = 2.0 * (1.0 + toto)
    rx = (matmisor[1, 2] - matmisor[2, 1]) / toto1
    ry = (matmisor[2, 0] - matmisor[0, 2]) / toto1
    rz = (matmisor[0, 1] - matmisor[1, 0]) / toto1

    vecRodrigues_sample = np.array(
        [rx, ry, rz]
    )  # axe de rotation en coordonnees sample

    theta = theta * 180.0 / np.pi

    return (vecRodrigues_sample, theta)


def add_columns_to_summary_file_new(
    filesum,
    elem_label="Ge",
    filestf=None,
    omega_sample_frame=40.0,
    verbose=0,
    include_misorientation=0,
    filefitref_for_orientation=None,  # seulement pour include_misorientation = 1
    include_strain=1,  # 0 seulement pour mat2spots ou fit calib ou EBSD
    # les 4 options suivantes seulement pour
    #  include_misorientation = 1
    # et filefitref_for_orientation = None
    filter_mean_matrix_by_pixdev_and_npeaks=1,
    maxpixdev_for_mean_matrix=0.25,
    minnpeaks_for_mean_matrix=20,
    filter_mean_matrix_by_intensity=0,
    minintensity_for_mean_matrix=20000.0,
):  # 29May13

    """
    filesum previously generated with build_summary
    strain in 1e-3 units
    stress in 100 MPa units
    add :
        cosines rgb_x and rgb_z for orientation maps with color scale of first stereo triangle
        reference x and z for rgb are in sample frame
        strain in sample frame
        stress in crystal frame
        stress in sample frame
        von mises stress
        resolved shear stress RSS on glide planes
        max RSS
        
        if include_misorientation :  # seulement pour les analyses mono-grain
            add  :
                misorientation angle
                w23 w13 w12 tires du vecteur de Rodrigues vector en coordonnees sample rx ry rz
                par w23 = 2*rx,  w13 = 2*ry,  w12 = 2*rz,
                calcul par Emeric Plancher inspire de Romain Quey doc Orilib
                
       08Jan14 
       add rgby and wx wy wz
       09Jan14
       add rgbxyz_lab  - utile pour departager macles avec axe de maclage suivant x, y, ou z sample         
       24Jan14 : enleve strain columns
    """
    #

    data_1, list_column_names, nameline0 = read_summary_file(filesum)

    data_1 = np.array(data_1, dtype=float)

    list_col_names2 = list_column_names

    list_col_names_orient = [
        "rgb_x_sample",
        "rgb_y_sample",
        "rgb_z_sample",
        "rgb_x_lab",
        "rgb_y_lab",
        "rgb_z_lab",
    ]

    number_col_orient = array([3, 3, 3, 3, 3, 3])

    for k in range(len(number_col_orient)):
        for i in range(number_col_orient[k]):
            toto = list_col_names_orient[k] + "_" + str(i)
            list_col_names2.append(toto)

    if include_strain:

        list_col_names_strain = [
            "strain6_crystal",
            "strain6_sample",
            "stress6_crystal",
            "stress6_sample",
            "res_shear_stress",
            "max_rss",
            "von_mises",
        ]
        number_col_strain = array([6, 6, 6, 6, 12])

        for k in range(len(number_col_strain)):
            for i in range(number_col_strain[k]):
                toto = list_col_names_strain[k] + "_" + str(i)
                list_col_names2.append(toto)

        for k in range(len(number_col_strain), len(number_col_strain) + 2):
            list_col_names2.append(list_col_names_strain[k])

    # print list_col_names2
    header2 = ""
    for i in range(len(list_col_names2)):
        header2 = header2 + list_col_names2[i] + " "

    header = (
        nameline0
        + ", rgb_x_sample, rgb_y_sample, rgb_z_sample, rgb_x_lab, rgb_y_lab, rgb_z_lab"
    )

    if include_strain:
        header = (
            header
            + ", strain6_crystal,  strain6_sample, stress6_crystal, stress6_sample, res_shear_stress_12, max_rss, von_mises"
        )

    if include_misorientation:
        header = header + ", misorientation_angle, w_mrad_0, w_mrad_1, w_mrad_2 \n"
        header2 = header2 + "misorientation_angle w_mrad_0 w_mrad_1 w_mrad_2 \n"
    else:
        header = header + "\n"
        header2 = header2 + "\n"

    print(header)
    print(header2)
    print(header2.split())

    schmid_tensors = glide_systems_to_schmid_tensors(verbose=0)

    if filestf != None:
        c_tensor = read_stiffness_file(filestf)

    xsample_sample_coord = array([1.0, 0.0, 0.0])
    ysample_sample_coord = array([0.0, 1.0, 0.0])
    zsample_sample_coord = array([0.0, 0.0, 1.0])

    omegarad = omega_sample_frame * np.pi / 180.0
    ylab_sample_coord = array([0.0, cos(omegarad), -sin(omegarad)])
    zlab_sample_coord = array([0.0, sin(omegarad), cos(omegarad)])
    print(
        "x y z sample - sample coord : ",
        xsample_sample_coord,
        ysample_sample_coord,
        zsample_sample_coord,
    )
    print("y z lab - sample coord : ", ylab_sample_coord, zlab_sample_coord)

    numig = shape(data_1)[0]

    # numig = 10

    rgb_x = zeros((numig, 3), float)
    rgb_y = zeros((numig, 3), float)
    rgb_z = zeros((numig, 3), float)
    rgb_xlab = zeros((numig, 3), float)
    rgb_ylab = zeros((numig, 3), float)
    rgb_zlab = zeros((numig, 3), float)
    if include_strain:
        epsp_crystal = zeros((numig, 6), float)
        epsp_sample = zeros((numig, 6), float)
        sigma_crystal = zeros((numig, 6), float)
        sigma_sample = zeros((numig, 6), float)
        tau1 = zeros((numig, 12), float)
        von_mises = zeros(numig, float)
        maxrss = zeros(numig, float)

    #    img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, strain6_crystal 16:22, euler 22:25
    indimg = list_column_names.index("img")  # 3
    indpixdev = list_column_names.index("pixdev")  # 3
    indnpeaks = list_column_names.index("npeaks")  # 2
    indmatstart = list_column_names.index("matstarlab_0")  # 7
    indintensity = list_column_names.index("intensity")  # 4
    print(indpixdev, indnpeaks, indmatstart, indintensity)
    indmat = np.arange(indmatstart, indmatstart + 9)
    img_list = data_1[:, indimg]
    pixdev_list = data_1[:, indpixdev]
    npeaks_list = data_1[:, indnpeaks]
    intensity_list = data_1[:, indintensity]
    mat_list = data_1[:, indmat]

    if include_misorientation:
        indfilt2 = where(npeaks_list > 0.0)  # pour raccourcir le summary a la fin
        misorientation_angle = zeros(numig, float)
        omegaxyz = zeros((numig, 3), float)

        if filefitref_for_orientation == None:

            if filter_mean_matrix_by_pixdev_and_npeaks:
                print("filter matmean by pixdev and npeaks")
                indfilt = where(
                    (pixdev_list < maxpixdev_for_mean_matrix)
                    & (npeaks_list > minnpeaks_for_mean_matrix)
                )
                matstarlabref = (mat_list[indfilt[0]]).mean(axis=0)
                print("number of points used to calculate matmean", len(indfilt[0]))

            elif filter_mean_matrix_by_intensity:
                print("filter matmean by intensity")
                indfilt = where(
                    (intensity_list > minintensity_for_mean_matrix)
                    & (npeaks_list > 0.0)
                )
                matstarlabref = (mat_list[indfilt[0]]).mean(axis=0)
                print("number of points used to calculate matmean", len(indfilt[0]))

            else:
                matstarlabref = (mat_list[indfilt2[0]]).mean(axis=0)

            # TO REMOVE
        #            matmean = ((mat_list[indfilt[0]])[-10:]).mean(axis=0)   # test pour data Keckes

        else:
            matstarlabref, data_fit, calib, pixdev = F2TC.readlt_fit(
                filefitref_for_orientation, readmore=True
            )

    #        matmean3x3 = GT.matline_to_mat3x3(matmean)

    k = 0
    for i in range(numig):
        print("ig : ", i, "img : ", img_list[i])
        if npeaks_list[i] > 0.0:
            matstarlab = mat_list[i, :]
            # print "x"
            matstarlabnew, transfmat, rgb_x[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, xsample_sample_coord
            )
            rgb_xlab[i, :] = rgb_x[i, :] * 1.0
            # print "y"
            matstarlabnew, transfmat, rgb_y[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, ysample_sample_coord
            )
            matstarlabnew, transfmat, rgb_ylab[
                i, :
            ] = calc_cosines_first_stereo_triangle(matstarlab, ylab_sample_coord)
            # print "z"
            matstarlabnew, transfmat, rgb_z[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, zsample_sample_coord
            )
            matstarlabnew, transfmat, rgb_zlab[
                i, :
            ] = calc_cosines_first_stereo_triangle(matstarlab, zlab_sample_coord)

            if include_strain:
                epsp_sample[i, :], epsp_crystal[
                    i, :
                ] = matstarlab_to_deviatoric_strain_sample(
                    matstarlab,
                    omega0=omega_sample_frame,
                    version=2,
                    returnmore=True,
                    reference_element_for_lattice_parameters=elem_label,
                )

                sigma_crystal[i, :] = deviatoric_strain_crystal_to_stress_crystal(
                    c_tensor, epsp_crystal[i, :]
                )
                sigma_sample[
                    i, :
                ] = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(
                    matstarlab, sigma_crystal[i, :], omega0=omega_sample_frame
                )

                von_mises[i] = deviatoric_stress_crystal_to_von_mises_stress(
                    sigma_crystal[i, :]
                )

                tau1[
                    i, :
                ] = deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(
                    sigma_crystal[i, :], schmid_tensors
                )
                maxrss[i] = abs(tau1[i, :]).max()

            if include_misorientation:
                #                mat2 = GT.matline_to_mat3x3(matstarlab)
                #                vec_crystal, vec_lab, misorientation_angle[i] = twomat_to_rotation(matmean3x3,mat2, verbose = 0)

                (
                    vecRodrigues_sample,
                    misorientation_angle[i],
                ) = twomat_to_rotation_Emeric(
                    matstarlabref, matstarlab, omega0=omega_sample_frame
                )
                omegaxyz[i, :] = vecRodrigues_sample * 2.0 * 1000.0  # unites = mrad
                # misorientation_angle : unites = degres
                print(
                    round(misorientation_angle[i], 3), omegaxyz[i, :].round(decimals=2)
                )
            #                if k == 5 : return()

            if verbose:
                print(matstarlab)
                if include_strain:
                    print(
                        "deviatoric strain crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)"
                    )
                    print(epsp_crystal.round(decimals=2))
                    print(
                        "deviatoric strain sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-3 units)"
                    )
                    print(epsp_sample[i, :].round(decimals=2))

                    print(
                        "deviatoric stress crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (100 MPa units)"
                    )
                    print(sigma_crystal[i, :].round(decimals=2))

                    print(
                        "deviatoric stress sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (100 MPa units)"
                    )
                    print(sigma_sample[i, :].round(decimals=2))

                    print(
                        "Von Mises equivalent Stress (100 MPa units)",
                        round(von_mises[i], 3),
                    )
                    print(
                        "RSS resolved shear stresses on glide planes (100 MPa units) : "
                    )
                    print(tau1[i, :].round(decimals=3))
                    print("Max RSS : ", round(maxrss[i], 3))
        k = k + 1

    # numig here for debug with smaller numig
    data_list = column_stack(
        (data_1[:numig, :], rgb_x, rgb_y, rgb_z, rgb_xlab, rgb_ylab, rgb_zlab)
    )

    if include_strain:
        data_list = column_stack(
            (
                data_list,
                epsp_crystal,
                epsp_sample,
                sigma_crystal,
                sigma_sample,
                tau1,
                maxrss,
                von_mises,
            )
        )

    if include_misorientation:
        data_list = column_stack((data_list, misorientation_angle, omegaxyz))
        data_list = data_list[
            indfilt2[0], :
        ]  # enleve les images avec zero grain indexe

    add_str = "_add_columns"
    if filefitref_for_orientation != None:
        add_str = add_str + "_use_orientref"
        # TO REMOVE
    #    add_str = add_str + "_use_mean_10_points"

    outfilesum = filesum.rstrip(".dat") + add_str + ".dat"
    print(outfilesum)
    outputfile = open(outfilesum, "w")
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, data_list, fmt="%.6f")
    outputfile.close()

    return outfilesum


def add_columns_to_summary_file(
    filesum, elem_label="Ge", filestf=None, verbose=0
):  # 29May13

    """
    filesum previously generated with build_summary
    strain in 1e-3 units
    stress in 100 MPa units
    add :
        cosines rgb_x and rgb_z for orientation maps with color scale of first stereo triangle
        reference x and z for rgb are in sample frame
        strain in sample frame
        stress in crystal frame
        stress in sample frame
        von mises stress
        resolved shear stress RSS on glide planes
        max RSS
    """

    data_1, list_column_names, nameline0 = read_summary_file(filesum)

    data_1 = np.array(data_1, dtype=float)

    list_col_names = [
        "strain6_sample",
        "rgb_x_sample",
        "rgb_z_sample",
        "stress6_crystal",
        "stress6_sample",
        "res_shear_stress",
        "max_rss",
        "von_mises",
    ]

    list_col_names2 = list_column_names

    number_col = array([6, 3, 3, 6, 6, 12])

    for k in range(6):
        for i in range(number_col[k]):
            toto = list_col_names[k] + "_" + str(i)
            list_col_names2.append(toto)

    for k in range(6, 8):
        list_col_names2.append(list_col_names[k])

    # print list_col_names2
    header2 = ""
    for i in range(len(list_col_names2)):
        header2 = header2 + list_col_names2[i] + " "
    header2 = header2 + "\n"
    print(header2)

    print(header2.split())

    header = (
        nameline0
        + ", strain_6sample 25:31, rgb_x_sample 31:34, rgb_z_sample 34:37, \
stress6_crystal 37:43 , stress6_sample 43:49, res_shear_stress 49:61, \
max_rss 61, von_mises 62 \n"
    )

    print(header)

    schmid_tensors = glide_systems_to_schmid_tensors()

    print("filestf")
    print(filestf)
    if filestf != None:
        c_tensor = read_stiffness_file(filestf)

    axis_pole_sample_z = array([0.0, 0.0, 1.0])
    axis_pole_sample_x = array([1.0, 0.0, 0.0])
    print("pole axes 1, 2 - sample coord : ", axis_pole_sample_x, axis_pole_sample_z)

    numig = shape(data_1)[0]

    # numig = 10

    rgb_z = zeros((numig, 3), float)
    rgb_x = zeros((numig, 3), float)
    epsp_sample = zeros((numig, 6), float)
    sigma_crystal = zeros((numig, 6), float)
    sigma_sample = zeros((numig, 6), float)
    tau1 = zeros((numig, 12), float)
    von_mises = zeros(numig, float)
    maxrss = zeros(numig, float)

    for i in range(numig):
        print(i)
        if data_1[i, 2] > 0.0:
            matstarlab = data_1[i, 7:16]
            # print "z"
            matstarlabnew, transfmat, rgb_z[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, axis_pole_sample_z
            )
            # print "x"
            matstarlabnew, transfmat, rgb_x[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, axis_pole_sample_x
            )

            epsp_sample[i, :], epsp_crystal = matstarlab_to_deviatoric_strain_sample(
                matstarlab,
                omega0=40.0,
                version=2,
                returnmore=True,
                reference_element_for_lattice_parameters=elem_label,
            )

            sigma_crystal[i, :] = deviatoric_strain_crystal_to_stress_crystal(
                c_tensor, epsp_crystal
            )
            sigma_sample[
                i, :
            ] = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(
                matstarlab, sigma_crystal[i, :], omega0=40.0
            )

            von_mises[i] = deviatoric_stress_crystal_to_von_mises_stress(
                sigma_crystal[i, :]
            )

            tau1[
                i, :
            ] = deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(
                sigma_crystal[i, :], schmid_tensors
            )
            maxrss[i] = abs(tau1[i, :]).max()

            if verbose:
                print(matstarlab)
                print(
                    "deviatoric strain crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)"
                )
                print(epsp_crystal.round(decimals=2))
                print(
                    "deviatoric strain sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-3 units)"
                )
                print(epsp_sample[i, :].round(decimals=2))

                print(
                    "deviatoric stress crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (100 MPa units)"
                )
                print(sigma_crystal[i, :].round(decimals=2))

                print(
                    "deviatoric stress sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (100 MPa units)"
                )
                print(sigma_sample[i, :].round(decimals=2))

                print(
                    "Von Mises equivalent Stress (100 MPa units)",
                    round(von_mises[i], 3),
                )
                print("RSS resolved shear stresses on glide planes (100 MPa units) : ")
                print(tau1[i, :].round(decimals=3))
                print("Max RSS : ", round(maxrss[i], 3))

    data_list = column_stack(
        (
            data_1[:numig, :],
            epsp_sample,
            rgb_x,
            rgb_z,
            sigma_crystal,
            sigma_sample,
            tau1,
            maxrss,
            von_mises,
        )
    )
    print("filesum", filesum)
    outfilesum = filesum.rstrip(".dat") + "_add_columns.dat"
    import module_graphique as modgraph

    modgraph.stockfilesumcolumn = outfilesum
    print(outfilesum)
    outputfile = open(outfilesum, "w")
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, data_list, fmt="%.6f")
    outputfile.close()

    return outfilesum


def plot_orientation_triangle_color_code():

    # plot orientation color scale in stereographic projection

    p.figure(figsize=(8, 8))

    numrand1 = 50
    range001 = list(range(numrand1 + 1))
    range001 = np.array(range001, dtype=float) / numrand1

    angrange = range001 * 1.0

    for i in range(numrand1 + 1):
        for j in range(numrand1 + 1):
            col1 = zeros(3, float)
            uq = (1.0 - range001[i]) * uqref_cr[:, 0] + range001[i] * (
                angrange[j] * uqref_cr[:, 1] + (1.0 - angrange[j]) * uqref_cr[:, 2]
            )
            uq = uq / norme(uq)

            qsxy = hkl_to_xystereo(uq, down_axis=[0.0, -1.0, 0.0])
            # RGB coordinates
            rgb_pole = zeros(3, float)

            # blue : distance in q space between M tq OM = uq et le plan 001 101 passant par O
            rgb_pole[2] = abs(inner(uq, uqn_b)) / abs(inner(uqref_cr[:, 2], uqn_b))
            rgb_pole[1] = abs(inner(uq, uqn_g)) / abs(inner(uqref_cr[:, 1], uqn_g))
            rgb_pole[0] = abs(inner(uq, uqn_r)) / abs(inner(uqref_cr[:, 0], uqn_r))

            # normalize
            # convention OR LT
            rgb_pole = rgb_pole / max(rgb_pole)
            # convention Tamura XMAS
            # rgb_pole = rgb_pole / norme(rgb_pole)
            # print "rgb_pole :"
            # print rgb_pole
            rgb_pole = rgb_pole.clip(min=0.0, max=1.0)

            p.plot(
                qsxy[0],
                qsxy[1],
                marker="o",
                markerfacecolor=rgb_pole,
                markeredgecolor=rgb_pole,
                markersize=5,
            )

    p.xlim(-0.1, 0.5)
    p.ylim(-0.1, 0.5)

    return 0


def calc_map_imgnum(filexyz):  # 31May13

    # setup location of images in map based on xech yech + map pixel size
    # permet pixels rectangulaires
    # permet cartos incompletes

    print("\n\n  HELLO \n\n")

    data_1 = loadtxt(filexyz, skiprows=1)
    nimg = shape(data_1)[0]
    imglist = data_1[:, 0]
    print("first line :", data_1[0, :])
    print("last line : ", data_1[-1, :])

    xylist = data_1[:, 1:3] - data_1[0, 1:3]

    dxyfast = xylist[1, :] - xylist[0, :]
    print("dxyfast = ", dxyfast)
    dxymax = xylist[-1, :] - xylist[0, :]
    print("dxymax = ", dxymax)

    print("fast axis")
    indfast = where(abs(dxyfast) > 0.0)
    fast_axis = indfast[0][0]
    print(fast_axis)
    nintfast = dxymax[fast_axis] / dxyfast[fast_axis]
    # print nintfast
    nintfast = int(round(nintfast, 0))
    print(nintfast)
    nptsfast = nintfast + 1

    dxyslow = xylist[nptsfast, :] - xylist[0, :]
    print("dxyslow = ", dxyslow)
    print("slow axis")
    slow_axis = int(abs(fast_axis - 1))
    print(slow_axis)
    nintslow = dxymax[slow_axis] / dxyslow[slow_axis]
    # print nintslow
    nintslow = int(round(nintslow, 0))
    print(nintslow)
    nptsslow = nintslow + 1

    print("nptstot from file : ", data_1[-1, 0] - data_1[0, 0] + 1)
    print("npstot from nptsslow*nptsfast : ", nptsslow * nptsfast)

    # BM32 : maps with dxech > 0 and dyech >0 : start at lower left corner on sample

    # d2scan xy not allowed only dscan x or dscan y

    dxystep = dxyfast + dxyslow

    print("axis : fast , slow ", fast_axis, slow_axis)
    print("nb of points : fast, slow ", nptsfast, nptsslow)
    print("dxy step", dxystep)

    abs_step = abs(dxystep)
    print("dxy step abs ", abs_step)
    largestep = max(abs_step)
    smallstep = min(abs_step)
    pix_r = largestep / smallstep
    if pix_r != 1.0:
        print("|dx| and |dy| steps not equal : will use rectangular pixels in map")
        print("aspect ratio : ", pix_r)
    else:
        print("equal |dx| and |dy| steps")
        pix1 = pix2 = 1

    if float(int(round(pix_r, 1))) < (pix_r - 0.01):
        print("non integer aspect ratio")
        for nmult in (2, 3, 4, 5):
            toto = float(nmult) * pix_r
            # print toto
            # print int(round(toto,1))
            if abs(float(int(round(toto, 1))) - toto) < 0.01:
                # print nmult
                break
        pix1 = nmult
        pix2 = int(round(float(nmult) * pix_r, 1))
        # print "map pixel size will be ", pix1, pix2
    else:
        pix1 = 1
        pix2 = int(round(pix_r, 1))

    print("pixel size for map (pix1= small, pix2= large):", pix1, pix2)

    large_axis = argmax(abs_step)
    small_axis = argmin(abs_step)

    # print large_axis, small_axis
    if large_axis == 1:
        pixsize = array([pix1, pix2], dtype=int)
    else:
        pixsize = array([pix2, pix1], dtype=int)
    print("pixel size for map dx dy", pixsize)

    # dx => columns, dy => lines
    if fast_axis == 0:
        nximg, nyimg = nptsfast, nptsslow
    else:
        nximg, nyimg = nptsslow, nptsfast

    map_imgnum = zeros((nyimg, nximg), int)

    print("map raw size ", shape(map_imgnum))

    impos_start = zeros(2, int)
    if dxymax[0] > 0.0:
        startx = "left"
        impos_start[1] = 0
    else:
        startx = "right"
        impos_start[1] = nximg - 1
    if dxymax[1] > 0.0:
        starty = "lower"
        impos_start[0] = nyimg - 1
    else:
        starty = "upper"
        impos_start[0] = 0
    startcorner = starty + " " + startx + " corner"
    print("map starts in : ", startcorner)
    print("impos_start = ", impos_start)
    # print dxyfast[fast_axis]
    # print dxyslow[slow_axis]
    # print dxystep

    impos = zeros(2, int)  # y x

    # tableau normal : y augmente vers le bas, x vers la droite
    # xech yech : xech augmente vers la droite, yech augmente vers le hut

    # tester orientation avec niveaux gris = imgnum

    for i in range(nimg):
        # for i in range(200) :
        imnum = int(round(imglist[i], 0))
        impos[1] = xylist[i, 0] / abs(dxystep[0])
        impos[0] = -xylist[i, 1] / abs(dxystep[1])
        # print impos
        impos = impos_start + impos
        # print impos
        map_imgnum[impos[0], impos[1]] = imnum

    # print "map_imgnum"
    # print map_imgnum

    return (map_imgnum, dxystep, pixsize, impos_start)


# from matplotlib import mpl
# cmap = mpl.cm.PiYG

import matplotlib.cm as mpl

cmap = mpl.get_cmap("PiYG")


def plot_map(
    filesum,
    filexyz=None,
    maptype="fit",
    filetype="LT",
    subtract_mean="no",
    grain_index=1,
    filter_on_pixdev_and_npeaks=1,
    maxpixdev_forfilter=0.3,
    minnpeaks_forfilter=20,
    strainmin_forplot=-0.2,  # pour strain, stress, rss, von Mises
    strainmax_forplot=0.2,  # use None for autoscale
    pixdevmin_forplot=0.0,
    pixdevmax_forplot=0.3,
    npeaksmin_forplot=6.0,
    npeaksmax_forplot=70.0,
):  # 29May13

    """
        # grain_index  = "indexing rank" of grain selected for mapping (for multigrain Laue patterns)
        first grain = grain with most intense spot  
        first grain has grain_index = 0  (LT summary files)
        first grain has grain_index = 1 (XMAS summary files (rebuilt))

        filetype = "LT" or "XMAS"
        maptype =  "fit" 
                or "euler3" or "rgb_x_sample"     
                or "strain6_crystal" or "strain6_sample" 
                or "stress6_crystal" or "stress6_sample"
                or "res_shear_stress"
                or 'max_rss'
                or 'von_mises'

        min/max = -/+ strainscale for strain plots
        (and quantities derived from strain)

        """

    list_column_names = [
        "img",
        "gnumloc",
        "npeaks",
        "pixdev",
        "intensity",
        "dxymicrons_0",
        "dxymicrons_1",
        "matstarlab_0",
        "matstarlab_1",
        "matstarlab_2",
        "matstarlab_3",
        "matstarlab_4",
        "matstarlab_5",
        "matstarlab_6",
        "matstarlab_7",
        "matstarlab_8",
        "strain6_crystal_0",
        "strain6_crystal_1",
        "strain6_crystal_2",
        "strain6_crystal_3",
        "strain6_crystal_4",
        "strain6_crystal_5",
        "euler3_0",
        "euler3_1",
        "euler3_2",
        "strain6_sample_0",
        "strain6_sample_1",
        "strain6_sample_2",
        "strain6_sample_3",
        "strain6_sample_4",
        "strain6_sample_5",
        "rgb_x_sample_0",
        "rgb_x_sample_1",
        "rgb_x_sample_2",
        "rgb_z_sample_0",
        "rgb_z_sample_1",
        "rgb_z_sample_2",
        "stress6_crystal_0",
        "stress6_crystal_1",
        "stress6_crystal_2",
        "stress6_crystal_3",
        "stress6_crystal_4",
        "stress6_crystal_5",
        "stress6_sample_0",
        "stress6_sample_1",
        "stress6_sample_2",
        "stress6_sample_3",
        "stress6_sample_4",
        "stress6_sample_5",
        "res_shear_stress_0",
        "res_shear_stress_1",
        "res_shear_stress_2",
        "res_shear_stress_3",
        "res_shear_stress_4",
        "res_shear_stress_5",
        "res_shear_stress_6",
        "res_shear_stress_7",
        "res_shear_stress_8",
        "res_shear_stress_9",
        "res_shear_stress_10",
        "res_shear_stress_11",
        "max_rss",
        "von_mises",
    ]

    #        list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev']

    data_list, listname, nameline0 = read_summary_file(filesum)

    data_list = np.array(data_list, dtype=float)

    numig = shape(data_list)[0]
    print(numig)
    ndata_cols = shape(data_list)[1]
    print(ndata_cols)

    indimg = listname.index("img")
    indgnumloc = listname.index("gnumloc")
    indnpeaks = listname.index("npeaks")
    indpixdev = listname.index("pixdev")

    # maptype , ncolplot, nb_values
    # ncolplot = nb of columns for these data
    # nb_values = 3 per rgb color map
    # nb_plots = number of graphs
    # ngraphline, ngraphcol = subplots
    # ngraphlabels = subplot number -1 for putting xlabel and ylabel on axes
    dict_nplot = {
        "euler3": [3, 3, 1, 1, 1, 0, ["rgb_euler"]],
        "rgb_x_sample": [6, 6, 2, 1, 2, 0, ["x_sample", "z_sample"]],
        "strain6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "strain6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "stress6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "stress6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "res_shear_stress": [
            12,
            36,
            12,
            3,
            4,
            8,
            [
                "rss0",
                "rss1",
                "rss2",
                "rss3",
                "rss4",
                "rss5",
                "rss6",
                "rss7",
                "rss8",
                "rss9",
                "rss10",
                "rss11",
            ],
        ],
        "max_rss": [1, 3, 1, 1, 1, 0, ["max_rss"]],
        "von_mises": [1, 3, 1, 1, 1, 0, ["von Mises stress"]],
        "fit": [2, 6, 2, 1, 2, 0, ["npeaks", "pixdev"]],
    }

    ncolplot = dict_nplot[maptype][0]
    if maptype != "fit":
        if maptype in ["max_rss", "von_mises"]:
            map_first_col_name = maptype
        else:
            print(maptype)
            print(listname)
            map_first_col_name = maptype + "_0"
        ind_first_col = listname.index(map_first_col_name)
        indcolplot = list(range(ind_first_col, ind_first_col + ncolplot))

    gnumlist = np.array(data_list[:, indgnumloc], dtype=int)
    pixdevlist = data_list[:, indpixdev]
    npeakslist = np.array(data_list[:, indnpeaks], dtype=int)

    if filexyz == None:
        # creation de filexyz a partir des colonnes de filesum
        indxy = listname.index("dxymicrons_0")
        imgxy = column_stack((data_list[:, indimg], data_list[:, indx : indx + 2]))
        ind1 = where(gnumlist == 0)
        imgxynew = imgxy[ind1[0], :]
        print("min, max : img x y ", imgxynew.min(axis=0), imgxynew.max(axis=0))
        print("first, second, last point : img x y :")
        print(imgxynew[0, :])
        print(imgxynew[1, :])
        print(imgxynew[-1, :])
        filexyz = "filexyz.dat"
        header = "img 0 , xech 1, yech 2 \n"
        outputfile = open(filexyz, "w")
        outputfile.write(header)
        np.savetxt(outputfile, imgxynew, fmt="%.4f")
        outputfile.close()

    map_imageindex_array, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)

    nb_lines, nb_cols = shape(map_imageindex_array)
    nb_values = dict_nplot[maptype][1]

    plotdat = zeros((nb_lines, nb_cols, nb_values), float)

    print("grain_index : ", grain_index)
    if filter_on_pixdev_and_npeaks:
        print("filtering :")
        print("maxpixdev ", maxpixdev_forfilter)
        print("minnpeaks ", minnpeaks_forfilter)
        ind1 = where(
            (gnumlist == grain_index)
            & (pixdevlist < maxpixdev_forfilter)
            & (npeakslist > minnpeaks_forfilter)
        )
    else:
        ind1 = where((gnumlist == grain_index) & (npeakslist > 0))

    print("position of grain_index:", ind1[0])

    # filtered data
    data_list2 = data_list[ind1[0], :]

    if maptype == "euler3":
        euler3 = data_list2[:, indcolplot]
        ang0 = 360.0
        ang1 = arctan(sqrt(2.0)) * 180.0 / math.pi
        ang2 = 180.0
        ang012 = array([ang0, ang1, ang2])
        print(euler3[0, :])
        euler3norm = euler3 / ang012
        print(euler3norm[0, :])
        # print min(euler3[:,0]), max(euler3[:,0])
        # print min(euler3[:,1]), max(euler3[:,1])
        # print min(euler3[:,2]), max(euler3[:,2])

    elif maptype == "rgb_x_sample":
        rgbxz = data_list2[:, indcolplot]

    elif maptype == "fit":
        for j in range(ncolplot):
            plotdat[:, :, 3 * j] = 1.0
        print(pixdevlist)
        pixdevlist2 = pixdevlist[ind1[0]]
        npeakslist2 = npeakslist[ind1[0]]
        pixdevmin2 = pixdevlist2.min()
        pixdevmax2 = pixdevlist2.max()
        pixdevmean2 = pixdevlist2.mean()
        npeaksmin2 = npeakslist2.min()
        npeaksmax2 = npeakslist2.max()
        npeaksmean2 = npeakslist2.mean()
        print(filesum)

        print("pixdev : mean, min, max")
        print(round(pixdevmean2, 3), round(pixdevmin2, 3), round(pixdevmax2, 3))
        print("npeaks : mean min max")
        print(round(npeaksmean2, 1), npeaksmin2, npeaksmax2)

        if pixdevmin_forplot == None:
            pixdevmin_forplot = pixdevmin2
        if pixdevmax_forplot == None:
            pixdevmax_forplot = pixdevmax2
        if npeaksmin_forplot == None:
            npeaksmin_forplot = npeaksmin2
        if npeaksmax_forplot == None:
            npeaksmax_forplot = npeaksmax2

        print("for color map : ")
        print("pixdev : min, max : ", pixdevmin_forplot, pixdevmax_forplot)
        print("npeaks : min, max : ", npeaksmin_forplot, npeaksmax_forplot)
        print("black = min for npeaks")
        print("black = max for pixdev")
        print("red = missing")

    else:  # for strain and derived quantities
        if maptype in ["max_rss", "von_mises"]:
            for j in range(ncolplot):
                plotdat[:, :, 3 * j] = 1.0
        else:
            for j in range(ncolplot):
                plotdat[:, :, 3 * j : 3 * (j + 1)] = 0.5
        strain62 = data_list2[:, indcolplot]
        stramean = strain62.mean(axis=0)
        if subtract_mean == "yes":
            strain62 = strain62 - stramean

        stramin = strain62.min(axis=0)
        stramax = strain62.max(axis=0)

        print("xx xy xz yy yz zz")
        print("min : ", stramin.round(decimals=2))
        print("max : ", stramax.round(decimals=2))
        print("mean : ", stramean.round(decimals=2))

        if strainmin_forplot != None:
            stramin = strainmin_forplot * ones(ncolplot, float)
        if strainmax_forplot != None:
            stramax = strainmax_forplot * ones(ncolplot, float)

        print("for color map :")
        print("min : ", stramin.round(decimals=2))
        print("max : ", stramax.round(decimals=2))

    # feeding data for plot
    imglist = np.array(data_list2[:, 0], dtype=int)
    numig2 = shape(data_list2)[0]
    for i in range(numig2):
        ind2 = where(map_imageindex_array == imglist[i])
        iref, jref = ind2[0][0], ind2[1][0]

        if maptype == "euler3":
            plotdat[iref, jref, :] = euler3norm[i, :]

        elif maptype == "rgb_x_sample":
            plotdat[iref, jref, :] = rgbxz[i, :] * 1.0

        elif maptype == "fit":
            plotdat[iref, jref, 0:3] = (
                (npeakslist2[i] - npeaksmin_forplot)
                / (npeaksmax_forplot - npeaksmin_forplot)
            ).clip(min=0.0, max=1.0)
            plotdat[iref, jref, 3:6] = (
                (pixdevmax_forplot - pixdevlist2[i])
                / (pixdevmax_forplot - pixdevmin_forplot)
            ).clip(min=0.0, max=1.0)

        elif maptype in ["max_rss", "von_mises"]:
            plotdat[iref, jref, 0:3] = (strain62[i, j] - stramin[j]) / (
                stramax[j] - stramin[j]
            ).clip(min=0.0, max=1.0)
        else:
            for j in range(ncolplot):
                toto = (strain62[i, j] - stramin[j]) / (stramax[j] - stramin[j]).clip(
                    min=0.0, max=1.0
                )
                plotdat[iref, jref, 3 * j : 3 * j + 3] = np.array(cmap(toto))[:3]
                if strain62[i, j] > stramax[j]:
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = [1.0, 1.0, 0.0]  # yellow
                elif strain62[i, j] < stramin[j]:
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = [1.0, 0.0, 0.0]  # red

        # reperage de l'ordre des images dans la carto
    # #                if imglist[i]==min(imglist) :
    # #                    plotdat[iref, jref, :] = array([1., 0., 0.])
    # #                if imglist[i]==min(imglist)+nb_cols-1 :
    # #                    plotdat[iref, jref, :] = array([0., 1., 0.])
    # #                if imglist[i]==max(imglist) :
    # #                    plotdat[iref, jref, :] = array([0., 0., 1.])
    # plotdat[iref, jref, :] = array([1.0, 1.0, 1.0])*float(imglist[i])/max(imglist)
    # print plotdat[iref, jref, :]

    # extent corrected 06Feb13
    xrange1 = array([0.0, nb_cols * dxystep[0]])
    yrange1 = array([0.0, nb_lines * dxystep[1]])
    xmin, xmax = min(xrange1), max(xrange1)
    ymin, ymax = min(yrange1), max(yrange1)
    extent = xmin, xmax, ymin, ymax
    print(extent)

    nb_plots = dict_nplot[maptype][2]
    ngraphline = dict_nplot[maptype][3]
    ngraphcol = dict_nplot[maptype][4]
    ngraphlabels = dict_nplot[maptype][5]
    print("nb_plots, ngraphline, ngraphcol, ngraphlabels")
    print(nb_plots, ngraphline, ngraphcol, ngraphlabels)
    print("shape(plotdat)")
    print(shape(plotdat))

    sys.path.append(os.path.abspath(".."))
    from mosaic import ImshowFrameNew

    for j in range(nb_plots):
        p.figure(1, figsize=(15, 10))
        ax = p.subplot(ngraphline, ngraphcol, j + 1)
        imrgb = p.imshow(
            plotdat[:, :, 3 * j : 3 * (j + 1)], interpolation="nearest", extent=extent
        )
        strname = dict_nplot[maptype][6][j]
        p.title(strname)

        ax.locator_params("x", tight=True, nbins=5)
        ax.locator_params("y", tight=True, nbins=5)
        if j == ngraphlabels:
            p.xlabel("dxech (microns)")
            p.ylabel("dyech (microns)")

        plo = ImshowFrameNew(
            None,
            -1,
            strname,
            plotdat[:, :, 3 * j : 3 * (j + 1)],
            extent=extent,
            xylabels=("dxech (microns)", "dyech (microns)"),
            Imageindices=map_imageindex_array,
        )
        plo.Show(True)

    return 0


cmap = mpl.get_cmap("RdBu_r")
# cmap = mpl.cm.RdBu_r


DEFAULT_PLOTMAPS_PARAMETERS_DICT = {
    "Map Summary File": None,
    "File xyz": None,
    "maptype": "fit",
    "filetype": "LT",
    "subtract_mean": "no",
    "probed_grainindex": 0,
    "filter_on_pixdev_and_npeaks": 1,
    "maxpixdev_forfilter": 20.0,  # only for filter_on_pixdev_and_npeaks : 1
    "minnpeaks_forfilter": 1.0,  # only for filter_on_pixdev_and_npeaks : 1
    "min_forplot": -0.2,  # pour strain, stress, rss, von Mises
    "max_forplot": 0.2,  # use None for autoscale
    "pixdevmin_forplot": 0.0,  # only for maptype : "fit"
    "pixdevmax_forplot": 10.0,  # only for maptype : "fit"
    "npeaksmin_forplot": 6.0,  # only for maptype : "fit"
    "npeaksmax_forplot": 70.0,  # only for maptype : "fit"
    "zoom": "no",
    "xylim": None,
    "filter_mean_strain_on_misorientation": 0,
    "max_misorientation": 0.15,  # only for filter_mean_strain_on_misorientation : 1
    "change_sign_xy_xz": 0,
    "subtract_constant": None,
    "remove_ticklabels_titles": 0,
    "col_for_simple_map": None,
    "low_npeaks_as_missing": None,
    "low_npeaks_as_red_in_npeaks_map": None,  # only for maptype : "fit"
    "low_pixdev_as_green_in_pixdev_map": None,  # only for maptype : "fit"
    "use_mrad_for_misorientation": "no",  # only for maptype : "misorientation_angle"
    "color_for_duplicate_images": None,  # [0.,1.,0.]
    "color_for_missing": None,
    "high_pixdev_as_blue_and_red_in_pixdev_map": None,  # only for maptype : "fit"
    "filter_on_intensity": 0,
    "min_intensity_forfilter": 20000.0,  # only for filter_on_intensity : 1
    "color_for_max_strain_positive": array(
        [1.0, 0.0, 0.0]
    ),  # red  # [1.0,1.0,0.0]  # yellow
    "color_for_max_strain_negative": array([0.0, 0.0, 1.0]),  # blue
    "plot_grid": 1,
    "map_rotation": 0,
}


def plot_map_new2(dict_params, maptype, grain_index, App_parent=None):  # JSM May 2017
    """
        grain_index  = "indexing rank" of grain selected for mapping (for multigrain Laue patterns)
        first grain = grain with most intense spot  
        first grain has gnumloc = 0  (LT summary files)



        maptype =  "fit" 
                or "euler3" or "rgb_x_sample"     
                or "strain6_crystal" or "strain6_sample" 
                or "stress6_crystal" or "stress6_sample"
                or "res_shear_stress"
                or 'max_rss'
                or 'von_mises'


        """

    d = DEFAULT_PLOTMAPS_PARAMETERS_DICT

    d.update(dict_params)

    print("\n\nENTERING plot_map_new()\n\n")

    print(d["Map Summary File"], d["File xyz"], d["maptype"], d["filetype"])
    print(d["subtract_mean"], d["probed_grainindex"], d["filter_on_pixdev_and_npeaks"])

    list_column_names = [
        "img",
        "probed_grainindex",
        "npeaks",
        "pixdev",
        "intensity",
        "dxymicrons_0",
        "dxymicrons_1",
        "matstarlab_0",
        "matstarlab_1",
        "matstarlab_2",
        "matstarlab_3",
        "matstarlab_4",  # 7:16
        "matstarlab_5",
        "matstarlab_6",
        "matstarlab_7",
        "matstarlab_8",
        "strain6_crystal_0",
        "strain6_crystal_1",
        "strain6_crystal_2",
        "strain6_crystal_3",  # 16:22
        "strain6_crystal_4",
        "strain6_crystal_5",
        "euler3_0",
        "euler3_1",
        "euler3_2",  # 22:25
        "strain6_sample_0",
        "strain6_sample_1",
        "strain6_sample_2",
        "strain6_sample_3",
        "strain6_sample_4",
        "strain6_sample_5",
        "rgb_x_sample_0",
        "rgb_x_sample_1",
        "rgb_x_sample_2",  # 25:31
        "rgb_z_sample_0",
        "rgb_z_sample_1",
        "rgb_z_sample_2",
        "stress6_crystal_0",
        "stress6_crystal_1",
        "stress6_crystal_2",  # 31:37
        "stress6_crystal_3",
        "stress6_crystal_4",
        "stress6_crystal_5",
        "stress6_sample_0",
        "stress6_sample_1",
        "stress6_sample_2",  # 37:43
        "stress6_sample_3",
        "stress6_sample_4",
        "stress6_sample_5",
        "res_shear_stress_0",
        "res_shear_stress_1",
        "res_shear_stress_2",
        "res_shear_stress_3",  # 43:15
        "res_shear_stress_4",
        "res_shear_stress_5",
        "res_shear_stress_6",
        "res_shear_stress_7",
        "res_shear_stress_8",
        "res_shear_stress_9",
        "res_shear_stress_10",
        "res_shear_stress_11",
        "max_rss",
        "von_mises",  # 58 and # 59
        "misorientation_angle",
        "dalf",
    ]

    #               list_column_names=['img', 'gnumloc', 'npeaks', 'pixdev',
    #         'intensity', 'dxymicrons_0', 'dxymicrons_1',
    # 'matstarlab_0', 'matstarlab_1', 'matstarlab_2',
    # 'matstarlab_3', 'matstarlab_4', 'matstarlab_5',
    # 'matstarlab_6', 'matstarlab_7', 'matstarlab_8',
    #  'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2',
    #  'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5',
    #  'euler3_0', 'euler3_1', 'euler3_2',
    #  'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2',
    #  'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5',
    #  'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2',
    #  'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2',
    #  'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2',
    #  'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5',
    #  'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2',
    #  'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5',
    #  'res_shear_stress_0', 'res_shear_stress_1', 'res_shear_stress_2',
    #  'res_shear_stress_3', 'res_shear_stress_4', 'res_shear_stress_5',
    #  'res_shear_stress_6', 'res_shear_stress_7', 'res_shear_stress_8',
    #  'res_shear_stress_9', 'res_shear_stress_10', 'res_shear_stress_11',
    #  'max_rss',
    #  'von_mises']):

    # key = maptype , nb_values, nplot
    # nb_values = nb of columns for these data
    # nplot = 3 per rgb color map
    # nb_plots = number of graphs
    # ngraphline, ngraphcol = subplots
    # ngraphlabels = subplot number -1 for putting xlabel and ylabel on axes
    dict_nplot = {
        "euler3": [3, 3, 1, 1, 1, 0, ["rgb_euler"]],
        "rgb_x_sample": [
            9,
            9,
            3,
            1,
            3,
            0,
            [
                "x_sample",
                "y_sample",
                "z_sample",
                "x_sample",
                "y_sample",
                "z_sample",
                "x_sample",
                "y_sample",
                "z_sample",
            ],
        ],
        "orientation": [
            9,
            9,
            3,
            1,
            3,
            0,
            [
                "x_sample",
                "y_sample",
                "z_sample",
                "x_sample",
                "y_sample",
                "z_sample",
                "x_sample",
                "y_sample",
                "z_sample",
            ],
        ],
        "rgb_x_lab": [9, 9, 3, 1, 3, 0, ["x_lab", "y_lab", "z_lab"]],
        "strain6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "strain6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "stress6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "stress6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "w_mrad": [3, 9, 3, 1, 3, 0, ["WX", "WY", "WZ"]],
        "res_shear_stress": [
            12,
            36,
            12,
            3,
            4,
            8,
            [
                "rss0",
                "rss1",
                "rss2",
                "rss3",
                "rss4",
                "rss5",
                "rss6",
                "rss7",
                "rss8",
                "rss9",
                "rss10",
                "rss11",
            ],
        ],
        "max_rss": [1, 3, 1, 1, 1, 0, ["max_rss"]],
        "von_mises": [1, 3, 1, 1, 1, 0, ["von Mises stress"]],
        "misorientation_angle": [1, 3, 1, 1, 1, 0, ["misorientation angle"]],
        "intensity": [1, 3, 1, 1, 1, 0, ["intensity"]],
        "maxpixdev": [1, 3, 1, 1, 1, 0, ["maxpixdev"]],
        "stdpixdev": [1, 3, 1, 1, 1, 0, ["stdpixdev"]],
        "fit": [2, 6, 2, 1, 2, 0, ["npeaks", "pixdev"]],
        "dalf": [1, 3, 1, 1, 1, 0, ["delta_alf exp-theor"]],
    }

    #  NB : misorientation_angle column seulement pour analyse mono-grain
    # NB : dalf column seulement pour mat2spots ou fit calib

    #        list_column_names =  ['img', 'probed_grainindex', 'npeaks', 'pixdev']

    #         color_grid = "k"

    #         if d['col_for_simple_map'] != None:
    #             filter_on_pixdev_and_npeaks = 0
    #             filter_mean_strain_on_misorientation = 0

    #         grain_index = 1
    #         maptype = 'strain6_crystal'
    #         maptype = 'fit'

    data, listname, nameline0 = read_summary_file(d["Map Summary File"])

    data_list = np.array(data, dtype=float)

    print("Data of strain  \n\n************\n")
    print("data.shape", data_list.shape)
    print("shape = ((nb images)* nb grains , nb of data columns)")
    nbgrains = int(np.amax(data_list[:, 1]) + 1)
    nb_images = int(data_list.shape[0] / nbgrains)

    print("nb of grains per image", nbgrains)
    print("nb of images", nb_images)

    grains_data = []
    for grainindex in range(nbgrains):
        grains_data.append(data_list[grainindex::nbgrains, :])

    print("first image of grains_data[0]", grains_data[0][0])
    print("len(first image of grains_data[0])", len(grains_data[0][0]))
    print("grains_data[0].shape", grains_data[0].shape)

    #         if maptype in ('strain6_crystal','strain6_sample','stress6_crystal','stress6_sample'):
    #             nb_components = 6
    #
    #         elif maptype in ('fit',):
    #             nb_components = 2
    #
    #         elif maptype in ('euler3'):
    #             nb_components = 3

    if maptype in ("orientation", "rgb_x_sample"):
        maptype = "orientation"

    plot_maptype_list = dict_nplot[maptype][6]

    datatype = "scalar"

    if maptype == "fit":
        colmin, nbdatacolumns = 2, 2
        datatype = "scalar"
        datasigntype = "positive"
    elif maptype == "orientation":
        colmin, nbdatacolumns = 7, 9
        datasigntype = "relative"
    elif maptype == "strain6_crystal":
        colmin, nbdatacolumns = 16, 6
        datatype = "symetricscalar"
        datasigntype = "relative"
    elif maptype == "euler3":
        colmin, nbdatacolumns = 22, 3
        datasigntype = "relative"
    elif maptype == "rgb_x_sample":
        colmin, nbdatacolumns = 25, 9
        datatype = "RGBvector"

    elif maptype == "rgb_x_lab":
        colmin, nbdatacolumns = 34, 9
        datasigntype = "positive"

    elif maptype == "strain6_crystal":
        colmin, nbdatacolumns = 43, 6
        datatype = "symetricscalar"
        datasigntype = "relative"
    elif maptype == "strain6_sample":
        colmin, nbdatacolumns = 49, 6
        datatype = "symetricscalar"
        datasigntype = "relative"
    elif maptype == "stress6_crystal":
        colmin, nbdatacolumns = 55, 6
    elif maptype == "stress6_sample":
        datatype = "symetricscalar"
        datasigntype = "relative"
        colmin, nbdatacolumns = 60, 6
    elif maptype == "res_shear_stress":
        colmin, nbdatacolumns = 67, 12
        datasigntype = "relative"

    elif maptype == "max_rss":
        colmin, nbdatacolumns = 79, 1
        datasigntype = "relative"
    elif maptype == "von_mises":
        colmin, nbdatacolumns = 80, 1
        datasigntype = "relative"

    #         elif maptype == 'misorientation_angle':
    #             colmin, nbdatacolumns=63,1
    #         elif maptype == 'dalf':
    #             colmin, nbdatacolumns=64,1

    zvalues_Ncomponents = grains_data[grain_index][:, colmin : colmin + nbdatacolumns]

    if maptype == "orientation":
        nbdatacolumns = 9
        datatype = "scalar"

        UBmatrices = zvalues_Ncomponents
        nbmatrices = len(UBmatrices)
        rUBs = UBmatrices.reshape(nbmatrices, 3, 3)
        print("UBmatrices[0]", UBmatrices[0])
        print("rUBs[0]", rUBs[0])
        cosines_array, list_vecs = GT.getdirectbasiscosines(rUBs)

        print("cosines_array [0]", cosines_array[0])
        print("cosines_array.shape", cosines_array.shape)

    filexyz = d["File xyz"]
    map_imageindex_array, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)

    #         print "map_imageindex_array",map_imageindex_array
    print("map_imageindex_array.shape", map_imageindex_array.shape)

    # Normal convention
    map_imageindex_array = np.flipud(map_imageindex_array)

    nlines = shape(map_imageindex_array)[0]
    ncol = shape(map_imageindex_array)[1]

    print("nlines,ncol", nlines, ncol)
    #         print "z_values.shape",z_values.shape

    for index_component in range(nbdatacolumns):

        columnname = plot_maptype_list[index_component]

        if maptype == "orientation":
            z_values = cosines_array.reshape((nlines, ncol, 9))[:, :, index_component]
            print("z_values.shape", z_values.shape)
            plot_maptype_list[index_component] = (
                str(list_vecs[index_component / 3]) + plot_maptype_list[index_component]
            )
            colorbar_label = plot_maptype_list[index_component]
        elif datatype in ("scalar", "symetricscalar"):
            print("consider datatype=", datatype)
            colorbar_label = columnname
            zvalues = zvalues_Ncomponents[:, index_component]
            z_values = zvalues.reshape((nlines, ncol))

        elif datatype == "RGBvector":
            zvalues = zvalues_Ncomponents[
                :, index_component * 3 : (index_component + 1) * 3
            ]
            z_values = zvalues.reshape((nlines, ncol, 3))

        print("scalar columnname", columnname)
        print("index_component", index_component)
        print("z_values.shape", z_values.shape)

        # to fit with odile's conventions
        #             z_values = flipud(z_values)
        ncol = int(ncol)
        nlines = int(nlines)
        i_index, j_index = GT.twoindices_positive_up_to(ncol - 1, nlines - 1).T

        posmotor_i = i_index * dxystep[0]
        posmotor_j = j_index * dxystep[1]

        ar_posmotor = np.array([posmotor_i, posmotor_j]).T

        ar_posmotor = reshape(ar_posmotor, (nlines, ncol, 2))

        #             plotobjet = ImshowFrame_Scalar(App_parent, -1,
        #                                      '%s %s'%(maptype,plot_maptype_list[index_component]),
        #                                      z_values,
        #                                      dataarray_info=ar_posmotor,
        #                                      datatype=datatype,
        #                                      xylabels=("dxech (microns)", "dyech (microns)"),
        #                                      posmotorname=('Xsample', 'Ysample'),
        #                                      Imageindices=map_imageindex_array,
        #                                      absolute_motorposition_unit='micron',
        #                                      colorbar_label=colorbar_label,
        #                                      maptype = maptype)

        print("in multigrain.py zvalues.shape", z_values.shape)
        nb_lines, nb_col = z_values.shape  # tocheck
        dict_param = {"datasigntype": datasigntype}
        Tabindices1D = np.ravel(map_imageindex_array)
        plotobjet = ImshowFrame(
            App_parent,
            -1,
            "%s %s" % (maptype, columnname),
            z_values,
            cmap=GT.ORRD,
            interpolation="nearest",
            origin="upper",
            Imageindices=Tabindices1D,
            nb_row=nb_col,
            nb_lines=nb_lines,
            stepindex=1,
            boxsize_row=1,
            boxsize_line=1,
            imagename=columnname,
            mosaic=0,
            datatype=None,
            dict_param=dict_param,
        )

        plotobjet.Show(True)

        if App_parent is not None:
            if App_parent.list_of_windows not in ([],):
                App_parent.list_of_windows.append(plotobjet)
            else:
                App_parent.list_of_windows = [plotobjet]


def plot_map_new(dict_params, App_parent=None):  # 29May13
    """
        # gnumloc  = "indexing rank" of grain selected for mapping (for multigrain Laue patterns)
        first grain = grain with most intense spot  
        first grain has gnumloc = 0  (LT summary files)
        first grain has gnumloc = 1 (XMAS summary files (rebuilt))

        filetype = "LT" or "XMAS"
        maptype =  "fit" 
                or "euler3" or "rgb_x_sample"     
                or "strain6_crystal" or "strain6_sample" 
                or "stress6_crystal" or "stress6_sample"
                or "res_shear_stress"
                or 'max_rss'
                or 'von_mises'

        min/max = -/+ strainscale for strain plots
        (and quantities derived from strain)
        """

    d = DEFAULT_PLOTMAPS_PARAMETERS_DICT

    d.update(dict_params)

    print("\n\nENTERING plot_map_new()\n\n")

    print(d["Map Summary File"], d["File xyz"], d["maptype"], d["filetype"])
    print(d["subtract_mean"], d["probed_grainindex"], d["filter_on_pixdev_and_npeaks"])

    list_column_names = [
        "img",
        "probed_grainindex",
        "npeaks",
        "pixdev",
        "intensity",
        "dxymicrons_0",
        "dxymicrons_1",
        "matstarlab_0",
        "matstarlab_1",
        "matstarlab_2",
        "matstarlab_3",
        "matstarlab_4",
        "matstarlab_5",
        "matstarlab_6",
        "matstarlab_7",
        "matstarlab_8",
        "strain6_crystal_0",
        "strain6_crystal_1",
        "strain6_crystal_2",
        "strain6_crystal_3",
        "strain6_crystal_4",
        "strain6_crystal_5",
        "euler3_0",
        "euler3_1",
        "euler3_2",
        "strain6_sample_0",
        "strain6_sample_1",
        "strain6_sample_2",
        "strain6_sample_3",
        "strain6_sample_4",
        "strain6_sample_5",
        "rgb_x_sample_0",
        "rgb_x_sample_1",
        "rgb_x_sample_2",
        "rgb_z_sample_0",
        "rgb_z_sample_1",
        "rgb_z_sample_2",
        "stress6_crystal_0",
        "stress6_crystal_1",
        "stress6_crystal_2",
        "stress6_crystal_3",
        "stress6_crystal_4",
        "stress6_crystal_5",
        "stress6_sample_0",
        "stress6_sample_1",
        "stress6_sample_2",
        "stress6_sample_3",
        "stress6_sample_4",
        "stress6_sample_5",
        "res_shear_stress_0",
        "res_shear_stress_1",
        "res_shear_stress_2",
        "res_shear_stress_3",
        "res_shear_stress_4",
        "res_shear_stress_5",
        "res_shear_stress_6",
        "res_shear_stress_7",
        "res_shear_stress_8",
        "res_shear_stress_9",
        "res_shear_stress_10",
        "res_shear_stress_11",
        "max_rss",
        "von_mises",
        "misorientation_angle",
        "dalf",
    ]

    #  NB : misorientation_angle column seulement pour analyse mono-grain
    # NB : dalf column seulement pour mat2spots ou fit calib

    #        list_column_names =  ['img', 'probed_grainindex', 'npeaks', 'pixdev']

    color_grid = "k"

    if d["col_for_simple_map"] != None:
        filter_on_pixdev_and_npeaks = 0
        filter_mean_strain_on_misorientation = 0

    data, listname, nameline0 = read_summary_file(d["Map Summary File"])

    data_list = np.array(data, dtype=float)

    print("Data of strain  \n\n************\n")
    print("data.shape", data_list.shape)
    print("shape = ((nb images)* nb grains , nb of data columns)")
    nbgrains = int(np.amax(data_list[:, 1]) + 1)
    nb_images = data_list.shape[0] / nbgrains

    print("nb of grains per image", type(nbgrains))
    print("nb of images", nb_images)

    grains_data = []
    for grain_index in range(nbgrains):
        grains_data.append(data_list[grain_index : grain_index + nb_images, :])

    print(grains_data[0])
    print(grains_data[0].shape)

    strain6_crystal = []
    strain6_sample = []
    stress6_crystal = []
    stress6_sample = []

    for grain_index in range(nbgrains):
        strain6_crystal.append(grains_data[grain_index][:, 16 : 16 + 6])
        strain6_sample.append(grains_data[grain_index][:, 25 : 25 + 6])
        stress6_crystal.append(grains_data[grain_index][:, 38 : 38 + 6])
        stress6_sample.append(grains_data[grain_index][:, 44 : 44 + 6])

    print("second grain, first image strain6_crystal", strain6_crystal[1][0])
    print("second grain, last image strain6_crystal", strain6_crystal[1][-1])

    numig = shape(data_list)[0]
    print(numig)
    ndata_cols = shape(data_list)[1]
    print(ndata_cols)

    indimg = listname.index("img")

    print("data_list", data_list[:, 3])

    if d["filter_on_intensity"]:
        indintensity = listname.index("intensity")
        intensitylist = np.array(data_list[:, indintensity], dtype=float)

    if d["col_for_simple_map"] == None:
        print("filling data")
        indgnumloc = listname.index("gnumloc")
        indnpeaks = listname.index("npeaks")
        indpixdev = listname.index("pixdev")
        indxech = listname.index("dxymicrons_0")
        if "misorientation_angle" in listname:
            indmisor = listname.index("misorientation_angle")
            if filter_mean_strain_on_misorientation:
                misor_list = np.array(data_list[:, indmisor], dtype=float)
                if d["use_mrad_for_misorientation"] == "yes":
                    print("converting misorientation angle into mrad")
                    misor_list = misor_list * math.pi / 180.0 * 1000.0
                indm = where(misor_list < d["max_misorientation"])
                print(
                    "filtering out img with large misorientation > ",
                    d["max_misorientation"],
                )
                print("nimg with low misorientation : ", shape(indm)[1])

        gnumlist = np.array(data_list[:, indgnumloc], dtype=int)
        pixdevlist = data_list[:, indpixdev]
        npeakslist = np.array(data_list[:, indnpeaks], dtype=int)
    else:
        gnumlist = zeros(numig, int)
        pixdevlist = zeros(numig, int)
        npeakslist = ones(numig, int) * 25

    # key = maptype , nb_values, nplot
    # nb_values = nb of columns for these data
    # nplot = 3 per rgb color map
    # nb_plots = number of graphs
    # ngraphline, ngraphcol = subplots
    # ngraphlabels = subplot number -1 for putting xlabel and ylabel on axes
    dict_nplot = {
        "euler3": [3, 3, 1, 1, 1, 0, ["rgb_euler"]],
        "rgb_x_sample": [9, 9, 3, 1, 3, 0, ["x_sample", "y_sample", "z_sample"]],
        "rgb_x_lab": [9, 9, 3, 1, 3, 0, ["x_lab", "y_lab", "z_lab"]],
        "strain6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "strain6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "stress6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "stress6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "w_mrad": [3, 9, 3, 1, 3, 0, ["WX", "WY", "WZ"]],
        "res_shear_stress": [
            12,
            36,
            12,
            3,
            4,
            8,
            [
                "rss0",
                "rss1",
                "rss2",
                "rss3",
                "rss4",
                "rss5",
                "rss6",
                "rss7",
                "rss8",
                "rss9",
                "rss10",
                "rss11",
            ],
        ],
        "max_rss": [1, 3, 1, 1, 1, 0, ["max_rss"]],
        "von_mises": [1, 3, 1, 1, 1, 0, ["von Mises stress"]],
        "misorientation_angle": [1, 3, 1, 1, 1, 0, ["misorientation angle"]],
        "intensity": [1, 3, 1, 1, 1, 0, ["intensity"]],
        "maxpixdev": [1, 3, 1, 1, 1, 0, ["maxpixdev"]],
        "stdpixdev": [1, 3, 1, 1, 1, 0, ["stdpixdev"]],
        "fit": [2, 6, 2, 1, 2, 0, ["npeaks", "pixdev"]],
        "dalf": [1, 3, 1, 1, 1, 0, ["delta_alf exp-theor"]],
    }

    nb_values = dict_nplot[d["maptype"]][0]
    if d["maptype"] != "fit":
        #            if maptype in ['max_rss','von_mises','misorientation_angle', 'dalf', "intensity"]:
        if nb_values == 1:
            map_first_col_name = d["maptype"]
            if d["col_for_simple_map"] != None:
                map_first_col_name = d["col_for_simple_map"]
        else:
            map_first_col_name = d["maptype"] + "_0"
            if d["col_for_simple_map"] != None:
                map_first_col_name = d["col_for_simple_map"]
        ind_first_col = listname.index(map_first_col_name)
        print("ind_first_col", ind_first_col)

        indcolplot = np.arange(ind_first_col, ind_first_col + nb_values)

    if d["zoom"] == "yes":
        listxj = []
        listyi = []

    filexyz = d["File xyz"]
    if filexyz == None:
        # creation de filexyz a partir des colonnes de filesum
        indxy = listname.index("dxymicrons_0")
        imgxy = column_stack((data_list[:, indimg], data_list[:, indxy : indxy + 2]))
        ind1 = where(gnumlist == 0)
        imgxynew = imgxy[ind1[0], :]
        print("min, max : img x y ", imgxynew.min(axis=0), imgxynew.max(axis=0))
        print("first, second, last point : img x y :")
        print(imgxynew[0, :])
        print(imgxynew[1, :])
        print(imgxynew[-1, :])
        filexyz = "filexyz.dat"
        header = "img 0 , xech 1, yech 2 \n"
        outputfile = open(filexyz, "w")
        outputfile.write(header)
        np.savetxt(outputfile, imgxynew, fmt="%.4f")
        outputfile.close()

    filexyz_new = filexyz
    xylim_new = d["xylim"]

    if abs(d["map_rotation"]) > 0.1:
        print("rotating map clockwise by : ", d["map_rotation"], "degrees")
        filexyz_new, xylim_new = rotate_map(
            filexyz, d["map_rotation"], xylim=d["xylim"]
        )

    map_imageindex_array, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz_new)

    nlines = shape(map_imageindex_array)[0]
    ncol = shape(map_imageindex_array)[1]
    nplot = dict_nplot[d["maptype"]][1]
    plotdat = zeros((nlines, ncol, nplot), float)
    datarray_info = zeros((nlines, ncol, nplot), float)
    ARRAY_INFO_FILLED = False

    print("grain : ", d["probed_grainindex"])
    print("npeakslist", npeakslist)
    if d["filter_on_pixdev_and_npeaks"]:
        print("filter_on_pixdev_and_npeaks")
        print("filtering :")
        print("maxpixdev ", d["maxpixdev_forfilter"])
        print("minnpeaks ", d["minnpeaks_forfilter"])
        indf = where(
            (gnumlist == d["probed_grainindex"])
            & (pixdevlist < d["maxpixdev_forfilter"])
            & (npeakslist > d["minnpeaks_forfilter"])
        )
    elif d["filter_on_intensity"]:
        print("filter_on_intensity")
        indf = where(
            (gnumlist == d["probed_grainindex"])
            & (npeakslist > 0)
            & (intensitylist > d["min_intensity_forfilter"])
        )
    else:
        print("default filtering")
        indf = where((gnumlist == d["probed_grainindex"]) & (npeakslist > 0))

    # print 'indf', indf
    # print 'indf[0]', indf[0]

    # filtered data
    data_list2 = data_list[indf[0], :]

    if d["maptype"] == "euler3":
        euler3 = data_list2[:, indcolplot]
        ang0 = 360.0
        ang1 = arctan(sqrt(2.0)) * 180.0 / np.pi
        ang2 = 180.0
        ang012 = array([ang0, ang1, ang2])
        print(euler3[0, :])
        euler3norm = euler3 / ang012
        print(euler3norm[0, :])
        # print min(euler3[:,0]), max(euler3[:,0])
        # print min(euler3[:,1]), max(euler3[:,1])
        # print min(euler3[:,2]), max(euler3[:,2])

    elif d["maptype"][:5] == "rgb_x":
        rgbxyz = data_list2[:, indcolplot]

    elif d["maptype"] == "fit":
        default_color_for_missing = array([1.0, 0.8, 0.8])
        if d["color_for_missing"] == None:
            color0 = default_color_for_missing
        else:
            color0 = d["color_for_missing"]
        for j in range(nb_values):
            plotdat[:, :, 3 * j : 3 * (j + 1)] = color0

        print("pixdevlist", pixdevlist)
        print("indf[0]", indf[0])

        pixdevlist2 = pixdevlist[indf[0]]
        npeakslist2 = npeakslist[indf[0]]
        pixdevmin2 = pixdevlist2.min()
        pixdevmax2 = pixdevlist2.max()
        pixdevmean2 = pixdevlist2.mean()
        npeaksmin2 = npeakslist2.min()
        npeaksmax2 = npeakslist2.max()
        npeaksmean2 = npeakslist2.mean()

        print("npeakslist2", npeakslist2)

        print("filesum", d["Map Summary File"])

        print("pixdev : mean, min, max")
        print(round(pixdevmean2, 3), round(pixdevmin2, 3), round(pixdevmax2, 3))
        print("npeaks : mean min max")
        print(round(npeaksmean2, 1), npeaksmin2, npeaksmax2)

        if d["pixdevmin_forplot"] == None:
            pixdevmin_forplot = pixdevmin2
        else:
            pixdevmin_forplot = d["pixdevmin_forplot"]
        if d["pixdevmax_forplot"] == None:
            pixdevmax_forplot = pixdevmax2
        else:
            pixdevmax_forplot = d["pixdevmax_forplot"]
        if d["npeaksmin_forplot"] == None:
            npeaksmin_forplot = npeaksmin2
        else:
            npeaksmin_forplot = d["npeaksmin_forplot"]
        if d["npeaksmax_forplot"] == None:
            npeaksmax_forplot = npeaksmax2
        else:
            npeaksmax_forplot = d["npeaksmax_forplot"]

        print("for color map : ")
        print("pixdev : min, max : ", pixdevmin_forplot, pixdevmax_forplot)
        print("npeaks : min, max : ", npeaksmin_forplot, npeaksmax_forplot)
        print("black = min for npeaks")
        print("black = max for pixdev")
        print("pink = missing")
        if d["low_npeaks_as_red_in_npeaks_map"] != None:
            print("npeaks : red < ", d["low_npeaks_as_red_in_npeaks_map"])

        color_grid = "k"

    # for strain and derived quantities
    else:
        maptype = d["maptype"]
        print("maptype", maptype)
        #            if maptype in ['max_rss','von_mises','misorientation_angle', "intensity"]:
        if nb_values == 1:
            default_color_for_missing = array(
                [1.0, 0.8, 0.8]
            )  # pink = color for missing data
            if d["color_for_missing"] == None:
                color0 = default_color_for_missing
            else:
                color0 = d["color_for_missing"]
            plotdat[:, :, 0:3] = color0
            color_filtered = np.array([0.0, 1.0, 0.0])
            color_grid = "k"
            if d["low_npeaks_as_missing"]:
                color_filtered = np.array([1.0, 0.8, 0.8])
        else:
            for j in range(nb_values):
                plotdat[
                    :, :, 3 * j : 3 * (j + 1)
                ] = 0.0  # black = color for missing data
            if maptype != "dalf":
                if maptype != "w_mrad":
                    print("xx xy xz yy yz zz")
                else:
                    print("wx wy wz")
            #                color_filtered = np.array([0.5,0.5,0.5])
            color_filtered = zeros(3, float)
            color_grid = "w"
            if d["low_npeaks_as_missing"]:
                color_filtered = np.array([0.0, 0.0, 0.0])

        imglist1 = np.array(data_list[:, indimg], dtype=int)
        for i in range(numig):
            if i not in indf[0]:
                ind2 = where(map_imageindex_array == imglist1[i])
                iref, jref = ind2[0][0], ind2[1][0]

                #                    if maptype in ['max_rss','von_mises','misorientation_angle', "intensity"]:
                if nb_values == 1:
                    plotdat[iref, jref, 0:3] = color_filtered
                else:
                    for j in range(nb_values):
                        plotdat[iref, jref, 3 * j : 3 * j + 3] = color_filtered

        list_plot = data_list2[:, indcolplot]
        if (maptype == "misorientation_angle") & (
            d["use_mrad_for_misorientation"] == "yes"
        ):
            print("converting misorientation angle into mrad")
            list_plot = list_plot * np.pi / 180.0 * 1000.0
        print(shape(list_plot))

        if d["change_sign_xy_xz"] & (maptype == "strain6_sample"):
            list_plot[:, 4] = -list_plot[:, 4]
            list_plot[:, 5] = -list_plot[:, 5]
        if d["filter_mean_strain_on_misorientation"]:
            list_plot_mean = list_plot[indm[0]].mean(axis=0)
        else:
            list_plot_mean = list_plot.mean(axis=0)
        if d["subtract_mean"] == "yes":
            print("subtract mean")
            list_plot = list_plot - list_plot_mean
        print("subtract_constant", d["subtract_constant"])
        if d["subtract_constant"] != None:
            list_plot = list_plot - d["subtract_constant"]

        list_plot_min = list_plot.min(axis=0)
        list_plot_max = list_plot.max(axis=0)

        print("min : ", list_plot_min.round(decimals=2))
        print("max : ", list_plot_max.round(decimals=2))
        print("mean : ", list_plot_mean.round(decimals=2))

        if d["min_forplot"] != None:
            list_plot_min = d["min_forplot"] * ones(nb_values, float)
        if d["max_forplot"] != None:
            list_plot_max = d["max_forplot"] * ones(nb_values, float)

        print("for color map :")
        print("min : ", list_plot_min.round(decimals=2))
        print("max : ", list_plot_max.round(decimals=2))

    maptype = d["maptype"]

    imglist = np.array(data_list2[:, 0], dtype=int)

    numig2 = shape(data_list2)[0]
    if d["col_for_simple_map"] == None:
        npeakslist = np.array(data_list2[:, indnpeaks], dtype=int)
        xylist = np.array(data_list2[:, indxech : indxech + 2], dtype=float)
    else:
        npeakslist = ones(numig2, int) * 25
        xylist = zeros((numig2, 2), float)

    dxystep_abs = abs(dxystep)

    #        if (maptype != "fit")&(maptype[:2] != "rgb"):
    #            print maptype
    #            print list_plot_min
    #            print list_plot_max
    #            list_plot_cen = (list_plot_max+list_plot_min)/2.0

    # -----------------------------------------------
    # filling array of data to plot  'plotdat'
    for i in range(numig2):
        ind2 = where(map_imageindex_array == imglist[i])
        #                print imglist[i]
        #                print ind2
        iref, jref = ind2[0][0], ind2[1][0]
        if (d["zoom"] == "yes") & (npeakslist[i] > 0):
            listxj.append(xylist[i, 0])
            listyi.append(xylist[i, 1])

        if maptype == "euler3":
            plotdat[iref, jref, :] = euler3norm[i, :]

            val_euler = euler3norm[i, :]
            datarray_info[iref, jref, :] = val_euler
            ARRAY_INFO_FILLED = True

        elif maptype[:5] == "rgb_x":
            plotdat[iref, jref, :] = rgbxyz[i, :] * 1.0

            val_rgb_x = rgbxyz[i, :]
            datarray_info[iref, jref, :] = val_rgb_x
            ARRAY_INFO_FILLED = True

        elif maptype == "fit":
            #                 print "npeaksmax_forplot", npeaksmax_forplot
            #                 print "npeaksmin_forplot", npeaksmin_forplot
            plotdat[iref, jref, 0:3] = (npeakslist2[i] - npeaksmin_forplot) / (
                npeaksmax_forplot - npeaksmin_forplot
            )

            # print 'pixdevlist2[i]',pixdevlist2[i]

            if d["low_npeaks_as_red_in_npeaks_map"] != None:
                if npeakslist2[i] < d["low_npeaks_as_red_in_npeaks_map"]:
                    plotdat[iref, jref, 0:3] = array([1.0, 0.0, 0.0])
            else:
                if npeakslist2[i] < npeaksmin_forplot:
                    plotdat[iref, jref, 0:3] = array([1.0, 0.0, 0.0])

            plotdat[iref, jref, 3:6] = (pixdevmax_forplot - pixdevlist2[i]) / (
                pixdevmax_forplot - pixdevmin_forplot
            )

            if d["high_pixdev_as_blue_and_red_in_pixdev_map"] != None:
                if pixdevlist2[i] > 0.25:
                    plotdat[iref, jref, 3:6] = array([0.0, 0.0, 1.0])
                if pixdevlist2[i] > 0.5:
                    plotdat[iref, jref, 3:6] = array([1.0, 0.0, 0.0])
            else:
                if pixdevlist2[i] > pixdevmax_forplot:
                    plotdat[iref, jref, 3:6] = array([1.0, 0.0, 0.0])
                if d["low_pixdev_as_green_in_pixdev_map"] != None:
                    if (pixdevlist2[i] < 0.25) & (npeakslist2[i] > 20):
                        plotdat[iref, jref, 3:6] = array([0.0, 1.0, 0.0])

            valnbpeaks = npeakslist2[i]
            valpixdev = pixdevlist2[i]
            datarray_info[iref, jref, :] = [
                valnbpeaks,
                valnbpeaks,
                valnbpeaks,
                valpixdev,
                valpixdev,
                valpixdev,
            ]
            ARRAY_INFO_FILLED = True

        #                elif maptype in ['max_rss','von_mises', "misorientation_angle", "intensity"]:
        elif nb_values == 1:
            if list_plot[i] > list_plot_max:
                plotdat[iref, jref, 0:3] = np.array([1.0, 0.0, 0.0])
            elif list_plot[i] < list_plot_min:
                plotdat[iref, jref, 0:3] = np.array([1.0, 1.0, 1.0])
            else:
                for j in range(3):
                    plotdat[iref, jref, j] = (list_plot_max - list_plot[i]) / (
                        list_plot_max - list_plot_min
                    )

            val_singlevalue = list_plot[i]
            datarray_info[iref, jref, :] = [
                val_singlevalue,
                val_singlevalue,
                val_singlevalue,
            ]

            ARRAY_INFO_FILLED = True

        else:
            for j in range(nb_values):
                if list_plot[i, j] > list_plot_max[j]:
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = d[
                        "color_for_max_strain_positive"
                    ]
                elif list_plot[i, j] < list_plot_min[j]:
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = d[
                        "color_for_max_strain_negative"
                    ]
                else:
                    toto = (list_plot[i, j] - list_plot_min[j]) / (
                        list_plot_max[j] - list_plot_min[j]
                    )
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = np.array(cmap(toto))[:3]

                val = list_plot[i, j]
                datarray_info[iref, jref, 3 * j : 3 * j + 3] = [val, val, val]

                ARRAY_INFO_FILLED = True

        if d["color_for_duplicate_images"] != None:
            if i > 0:
                dimg = imglist[i] - imglist[i - 1]
                if dimg == 0.0:
                    print("warning : two grains on img ", imglist[i])
                    plotdat[iref, jref, 0:3] = d["color_for_duplicate_images"]

        # reperage de l'ordre des images dans la carto
    # #                if imglist[i]==min(imglist) :
    # #                    plotdat[iref, jref, :] = array([1., 0., 0.])
    # #                if imglist[i]==min(imglist)+ncol-1 :
    # #                    plotdat[iref, jref, :] = array([0., 1., 0.])
    # #                if imglist[i]==max(imglist) :
    # #                    plotdat[iref, jref, :] = array([0., 0., 1.])
    # plotdat[iref, jref, :] = array([1.0, 1.0, 1.0])*float(imglist[i])/max(imglist)
    # print plotdat[iref, jref, :]

    # extent corrected 06Feb13
    xrange1 = array([0.0, ncol * dxystep[0]])
    yrange1 = array([0.0, nlines * dxystep[1]])
    xmin, xmax = min(xrange1), max(xrange1)
    ymin, ymax = min(yrange1), max(yrange1)
    extent = xmin, xmax, ymin, ymax
    print(extent)

    nb_plots = dict_nplot[maptype][2]
    ngraphline = dict_nplot[maptype][3]
    ngraphcol = dict_nplot[maptype][4]
    ngraphlabels = dict_nplot[maptype][5]
    print("nb_plots, ngraphline, ngraphcol, ngraphlabels")
    print(nb_plots, ngraphline, ngraphcol, ngraphlabels)
    print("shape(plotdat)")
    print(shape(plotdat))

    if d["zoom"] == "yes":
        listxj = np.array(listxj, dtype=float)
        listyi = np.array(listyi, dtype=float)
        minxj = listxj.min() - 2 * dxystep_abs[0]
        maxxj = listxj.max() + 2 * dxystep_abs[0]
        minyi = listyi.min() - 2 * dxystep_abs[1]
        maxyi = listyi.max() + 2 * dxystep_abs[1]
        print("zoom : minxj, maxxj, minyi, maxyi : ", minxj, maxxj, minyi, maxyi)

    sys.path.append(os.path.abspath(".."))
    from mosaic import ImshowFrameNew, ImshowFrame_Scalar

    p.rcParams["figure.subplot.right"] = 0.9
    p.rcParams["figure.subplot.left"] = 0.1
    p.rcParams["figure.subplot.bottom"] = 0.1
    p.rcParams["figure.subplot.top"] = 0.9

    #        p.rcParams['savefig.bbox'] = "tight"
    for j in range(nb_plots):
        #             fig1 = p.figure(1, figsize=(15, 10))
        # #            print p.setp(fig1)
        # #            print p.getp(fig1)
        #             ax = p.subplot(ngraphline, ngraphcol, j + 1)
        #             imrgb = p.imshow(plotdat[:, :, 3 * j:3 * (j + 1)], interpolation='nearest', extent=extent)
        # #            print p.setp(imrgb)
        if d["col_for_simple_map"] == None:
            strname = dict_nplot[maptype][6][j]
        else:
            strname = d["col_for_simple_map"]
        #             if remove_ticklabels_titles == 0 :
        #                 p.title(strname)
        #             if remove_ticklabels_titles:
        # #                print p.getp(ax)
        #                 p.subplots_adjust(wspace=0.05, hspace=0.05)
        #                 p.setp(ax, xticklabels=[])
        #                 p.setp(ax, yticklabels=[])
        #             if plot_grid :
        #                 ax.grid(color=color_grid, linestyle='-', linewidth=2)
        #
        #             if PAR.cr_string == "\n":
        #                 ax.locator_params('x', tight=True, nbins=5)
        #                 ax.locator_params('y', tight=True, nbins=5)
        #             if remove_ticklabels_titles == 0 :
        #                 if (j == ngraphlabels) :
        #                     p.xlabel("dxech (microns)")
        #                     p.ylabel("dyech (microns)")
        #             if zoom == "yes" :
        #                 p.xlim(minxj, maxxj)
        #                 p.ylim(minyi, maxyi)
        #             if xylim_new != None :
        #                 p.xlim(xylim_new[0], xylim_new[1])
        #                 p.ylim(xylim_new[2], xylim_new[3])

        if ARRAY_INFO_FILLED:
            AddedArrayInfo = datarray_info[:, :, 3 * j : 3 * (j + 1)]
            print("AddedArrayInfo.shape", AddedArrayInfo.shape)
        else:
            AddedArrayInfo = None

        datatype = None

        print("\n\nmaptype:%s" % maptype)
        if maptype in ("fit", "von_mises", "max_rss"):
            datatype = "scalar"

            if maptype == "fit":
                if j == 0:
                    col_data = 0
                    colorbar_label = "Nb peaks"
                elif j == 1:
                    col_data = 3
                    colorbar_label = "PixDev"
                z_values = datarray_info[:, :, col_data]

            if maptype == "von_mises":
                col_data = 0
                colorbar_label = "von_mises"
                z_values = datarray_info[:, :, col_data]

            if maptype == "max_rss":
                col_data = 0
                colorbar_label = "max_rss"
                z_values = datarray_info[:, :, col_data]

        elif maptype.startswith(("rgb_", "strain", "stress", "res_shear")):
            datatype = "RGBvector"
            if maptype.startswith("rgb_x"):
                colorbar_label = "rgb_x"
                z_values = datarray_info[:, :, 3 * j : 3 * (j + 1)]
            elif maptype.startswith("strain"):
                colorbar_label = "strain"
                #                     z_values = datarray_info[:, :, 3 * j:3 * (j + 1)]
                z_values = plotdat[:, :, 3 * j : 3 * (j + 1)]
            elif maptype.startswith("stress"):
                colorbar_label = "stress"
                z_values = plotdat[:, :, 3 * j : 3 * (j + 1)]
            elif maptype.startswith("res_shear"):
                colorbar_label = "res_shear"
                z_values = plotdat[:, :, 3 * j : 3 * (j + 1)]

        # to fit with odile's conventions
        z_values = flipud(z_values)
        ncol = int(ncol)
        nlines = int(nlines)
        i_index, j_index = GT.twoindices_positive_up_to(ncol - 1, nlines - 1).T

        posmotor_i = i_index * dxystep[0]
        posmotor_j = j_index * dxystep[1]

        ar_posmotor = np.array([posmotor_i, posmotor_j]).T

        ar_posmotor = reshape(ar_posmotor, (nlines, ncol, 2))

        plo = ImshowFrame_Scalar(
            App_parent,
            -1,
            strname,
            z_values,
            dataarray_info=ar_posmotor,
            datatype=datatype,
            xylabels=("dxech (microns)", "dyech (microns)"),
            posmotorname=("Xsample", "Ysample"),
            Imageindices=map_imageindex_array,
            absolute_motorposition_unit="micron",
            colorbar_label=colorbar_label,
        )

        #             plo = ImshowFrameNew(App_parent, -1, strname, plotdat[:, :, 3 * j:3 * (j + 1)],
        #                                  dataarray_info=AddedArrayInfo,
        #                                  datatype=datatype,
        #                                  extent=extent, xylabels=("dxech (microns)", "dyech (microns)"),
        #                                  Imageindices=map_imageindex_array)

        if App_parent is not None:
            if App_parent.list_of_windows not in ([],):
                App_parent.list_of_windows.append(plo)
            else:
                App_parent.list_of_windows = [plo]
        plo.Show(True)

    return 0


def rotate_map(filexyz, map_rotation, xylim=None):

    data_1 = loadtxt(filexyz, skiprows=1)
    data_1 = np.array(data_1, dtype=float)
    nimg = shape(data_1)[0]

    xylist = data_1[:, 1:3]
    print(xylist[:3, :])

    sin_rot = sin(map_rotation * PI / 180.0)
    cos_rot = cos(map_rotation * PI / 180.0)
    sin_rot = int(round(sin_rot, 0))
    cos_rot = int(round(cos_rot, 0))
    if (abs(sin_rot) != 1) & (sin_rot != 0):
        print("map rotation limited to 90, 180 or -90 deg")
        return 0
    matrotmap = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])

    xylist_new = (dot(matrotmap, xylist.transpose())).transpose()

    print(xylist_new[:3, :])

    xylim_new = None

    if xylim != None:
        # xmin xmax ymin ymax
        print("xylim = ", xylim)
        # print xylim.reshape(2,2)
        toto = xylim.reshape(2, 2)
        toto1 = dot(matrotmap, toto)
        # print toto1.reshape(4,)
        toto2 = toto1.reshape(4)
        xylim_new = np.array(
            [min(toto2[0:2]), max(toto2[0:2]), min(toto2[2:4]), max(toto2[2:4])]
        )
        print("xylim_new = ", xylim_new)

    data_1_new = column_stack((data_1[:, 0], xylist_new, data_1[:, 3:]))

    filexyz_new = filexyz.rstrip(".dat") + "_new.dat"

    header = "img 0 , xech_new 1, yech_new 2, zech 3, mon4 4, lambda 5 \n"
    outfilename = filexyz_new
    print(outfilename)
    outputfile = open(outfilename, "w")
    outputfile.write(header)
    np.savetxt(outputfile, data_1_new, fmt="%.4f")
    outputfile.close()
    return (filexyz_new, xylim_new)


def class_data_into_grainnum(
    filesum, filepathout, tol1=0.1, test_mode="yes"
):  # 29May13

    data_list, listname, nameline0 = read_summary_file(filesum)

    data_list = np.array(data_list, dtype=float)

    indimg = listname.index("img")
    indgnumloc = listname.index("gnumloc")
    indnpeaks = listname.index("npeaks")
    indrgb = listname.index("rgb_x_sample_0")

    local_gnumlist = np.array(data_list[:, indgnumloc], dtype=int)
    npeakslist = np.array(data_list[:, indnpeaks], dtype=int)

    ind1 = where(npeakslist > 1)

    print(ind1[0])
    data_list2 = data_list[ind1[0], :]
    numig2 = shape(data_list2)[0]
    print(numig2)

    indrgbxz = list(range(indrgb, indrgb + 6))

    local_gnumlist2 = local_gnumlist[ind1[0]]
    rgbxz = data_list2[:, indrgbxz]
    imglist = data_list2[:, indimg]
    print(rgbxz[:2, :])
    print(np.shape(rgbxz))

    # norme(rgb) = 1
    for i in range(numig2):
        rgbxz[i, :3] = rgbxz[i, :3] / norme(rgbxz[i, :3])
        rgbxz[i, 3:] = rgbxz[i, 3:] / norme(rgbxz[i, 3:])

    toto = column_stack((list(range(numig2)), imglist, local_gnumlist2, rgbxz))

    #        print toto[:10,:]
    #        print toto.transpose()[:,:10]

    toto1 = toto.transpose()

    # nested sort
    # Sort on last row, then on 2nd last row, etc.
    ind2 = np.lexsort(toto1)

    print(shape(ind2))
    print(toto1.take(ind2[:10], axis=-1))

    if test_mode == "yes":
        nmax = 250
        verbose = 1
        teststr = "test"
    else:
        nmax = numig2
        verbose = 0
        teststr = "all2"

    sorted_list = toto1.take(ind2[:nmax], axis=-1).transpose()

    print(shape(sorted_list))

    dict_grains = {}
    num_one_pixel_grains = 0
    ig_list = np.array(sorted_list[:, 0], dtype=int)
    img_list = np.array(sorted_list[:, 1], dtype=int)
    gnumloc_list = np.array(sorted_list[:, 2], dtype=int)

    has_grain_num = zeros(nmax, int)
    is_ref = zeros(nmax, int)
    grain_size = zeros(nmax, int)
    global_gnum = zeros(nmax, int)
    gnum = 0
    for i in range(nmax):
        if has_grain_num[i] == 0:
            print("i, gnum = ", i, gnum)
            rgbref = sorted_list[i, -6:]
            global_gnum[i] = gnum
            is_ref[i] = 1
            has_grain_num[i] = 1
            grain_size[i] = 1
            ind_in_grain_list = [i]
            # ig_in_grain_list = [ig_list[i],]
            for j in range(i + 1, nmax):
                if has_grain_num[j] == 0:
                    drgb = norme(sorted_list[j, -6:] - rgbref)
                    if verbose:
                        print("j, drgb = ", j, round(drgb, 3))
                    if drgb < tol1:
                        has_grain_num[j] = 1
                        global_gnum[j] = gnum
                        if verbose:
                            print("gnum =", gnum)
                        grain_size[i] = grain_size[i] + 1
                        ind_in_grain_list.append(j)
                        # ig_in_grain_list.append(ig_list[j])

                    else:
                        if verbose:
                            print(" ")
            if grain_size[i] == 1:
                num_one_pixel_grains = num_one_pixel_grains + 1
                global_gnum[i] = -1
                print(" ")
            else:
                grain_size[ind_in_grain_list] = grain_size[i]
                short_rgb = sorted_list[ind_in_grain_list, -6:]
                mean_rgb = short_rgb.mean(axis=0)
                std_rgb = short_rgb.std(axis=0) * 1000.0
                range_rgb = (short_rgb.max(axis=0) - short_rgb.min(axis=0)) * 1000.0

                ig_in_grain_list = ig_list[ind_in_grain_list]
                img_in_grain_list = img_list[ind_in_grain_list]
                gnumloc_in_grain_list = gnumloc_list[ind_in_grain_list]

                print("grain_size = ", grain_size[i])
                # print "ind_in_grain_list ", ind_in_grain_list
                print("ig_in_grain_list ", ig_in_grain_list)
                print("img_in_grain_list", img_in_grain_list)
                print("gnumloc_in_grain_list", gnumloc_in_grain_list)
                print("rgb in grain :")
                print("mean", mean_rgb.round(decimals=3))
                print("std*1000 ", std_rgb.round(decimals=3))
                print("range*1000 ", range_rgb.round(decimals=3))
                print("\n")

                dict_grains[gnum] = [
                    grain_size[i],
                    ind_in_grain_list,
                    ig_in_grain_list,
                    img_in_grain_list,
                    gnumloc_in_grain_list,
                    mean_rgb.round(decimals=3),
                    std_rgb.round(decimals=3),
                    range_rgb.round(decimals=3),
                ]

                gnum = gnum + 1

    print("##########################################################")
    print("gnum = ", gnum)
    print("num_one_pixel_grains =", num_one_pixel_grains)

    gnumtot = gnum

    # print dict_grains
    dict_values_names = [
        "grain size",
        "ind_in_grain_list",
        "ig_in_grain_list",
        "img_in_grain_list",
        "gnumloc_in_grain_list",
        "mean_rgb",
        "std_rgb *1000",
        "range_rgb *1000",
    ]

    ndict = len(dict_values_names)

    # renumerotation des grains pour classement par taille decroissante May13
    gnumlist = list(range(gnumtot))
    gsizelist = zeros(gnumtot, int)

    for key, value in dict_grains.items():
        gsizelist[key] = value[0]

    gnum_gsize_list = column_stack((gnumlist, gsizelist))
    sorted_gnum_gsize_list = sort_list_decreasing_column(gnum_gsize_list, 1)
    # print gnum_gsize_list
    # print sorted_gnum_gsize_list
    dict_grains2 = {}
    for gnum2 in range(gnumtot):
        #        print gnum2
        #        print sorted_gnum_gsize_list[gnum2,0]
        dict_grains2[gnum2] = dict_grains[sorted_gnum_gsize_list[gnum2, 0]]

    #    for i in range(ndict):
    #        print dict_values_names[i]
    #        for key, value in dict_grains2.iteritems():
    #            print key,value[i]
    #    klmdqs

    ig_list = column_stack(
        (
            np.array(sorted_list[:, :3], dtype=int),
            is_ref,
            has_grain_num,
            global_gnum,
            grain_size,
            list(range(nmax)),
        )
    )

    header = "ig 0, img 1, local_gnum 2, is_ref 3, has_grain_num 4, global_gnum 5, grain_size 6, igsort 7"
    print(header)

    sorted_ig_list = sort_list_decreasing_column(ig_list, 6)

    # print sorted_ig_list

    # nouveaux numeros de grain dans liste ig
    for i in range(nmax):
        if sorted_ig_list[i, 5] != -1:
            ind1 = where(sorted_gnum_gsize_list[:, 0] == sorted_ig_list[i, 5])
            sorted_ig_list[i, 5] = ind1[0]

    print(sorted_ig_list)

    if 1:
        outfilegnum = os.path.join(filepathout, "grain_num2_" + teststr + ".txt")
        print("column results saved in :")
        print(outfilegnum)
        outputfile = open(outfilegnum, "w")
        outputfile.write(header + "\n")
        np.savetxt(outputfile, sorted_ig_list, fmt="%d")
        outputfile.close()

    if 1:
        import module_graphique as modgraph

        outfilegtog = os.path.join(filepathout, "gtog3_" + teststr + ".txt")
        modgraph.filegrain_1 = outfilegtog
        print("dictionnary results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog, "w")
        for i in range(ndict):
            outputfile.write(dict_values_names[i] + "\n")
            for key, value in dict_grains2.items():
                if i == 0:
                    str1 = str(value[i])
                else:
                    str1 = " ".join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 + "\n")
        outputfile.close()

    # taille de grains moyenne
    list1 = []
    for key, value in dict_grains2.items():
        # print key,value[0]
        list1.append(value[0])

    toto = np.array(list1, dtype=float)
    print("mean grain size (units = map pixels) ", round(toto.mean(), 2))

    # mosaique de grain moyenne
    list1 = []
    for key, value in dict_grains2.items():
        # print key,value[6]
        list1.append(value[6])

    toto = np.array(list1, dtype=float)
    print("mean grain std_rgb * 1000 ", toto.mean(axis=0).round(decimals=2))

    return dict_grains2


def read_dict_grains(
    filegrains, dict_with_edges="no", dict_with_all_cols="no", dict_with_all_cols2="no"
):  # 29May13
    """
    read grain dictionnary file created by class_data_into_grainnum
    or appended by find_grain_edges
    or appended by fill_dict_grains
    or appended by add_intragrain_rotations_to_dict_grains
    """

    listint = [0, 1, 2, 3, 4, 8, 9, 10, 12, 13]
    listfloat = [5, 6, 7, 11]
    toto = list(range(14, 50))
    listfloat = listfloat + toto
    print(listfloat)

    # 0 1 2 3 4 int
    # 5 6 7 float
    dict_values_names = [
        "grain size",
        "ind_in_grain_list",
        "ig_in_grain_list",
        "img_in_grain_list",
        "gnumloc_in_grain_list",
        "mean_rgb",
        "std_rgb *1000",
        "range_rgb *1000",
    ]

    if dict_with_edges == "yes":
        # 8 9 10 int
        # 11 float
        # 12 int
        # pixels des frontieres etendues, pixel_line_position pixel_column_position pixel_edge_type
        toto = [
            "list_line",
            "list_col",
            "list_edge",
            "gnumloc_mean",
            "list_edge_restricted",
        ]
        dict_values_names = dict_values_names + toto

    if dict_with_all_cols == "yes":
        # 13 int
        # 14 : 48 float
        toto = [
            "npeaks",
            "pixdev",
            "intensity",
            "strain6_crystal_0",
            "strain6_crystal_1",
            "strain6_crystal_2",
            "strain6_crystal_3",
            "strain6_crystal_4",
            "strain6_crystal_5",
            "strain6_sample_0",
            "strain6_sample_1",
            "strain6_sample_2",
            "strain6_sample_3",
            "strain6_sample_4",
            "strain6_sample_5",
            "rgb_x_sample_0",
            "rgb_x_sample_1",
            "rgb_x_sample_2",
            "rgb_z_sample_0",
            "rgb_z_sample_1",
            "rgb_z_sample_2",
            "stress6_crystal_0",
            "stress6_crystal_1",
            "stress6_crystal_2",
            "stress6_crystal_3",
            "stress6_crystal_4",
            "stress6_crystal_5",
            "stress6_sample_0",
            "stress6_sample_1",
            "stress6_sample_2",
            "stress6_sample_3",
            "stress6_sample_4",
            "stress6_sample_5",
            "max_rss",
            "von_mises",
        ]
        dict_values_names = dict_values_names + toto

    if dict_with_all_cols2 == "yes":
        # 48,49 : float
        toto = ["matstarlab_mean", "misorientation_angle"]
        dict_values_names = dict_values_names + toto

    ndict = len(dict_values_names)
    linepos_list = zeros(ndict + 1, dtype=int)

    f = open(filegrains, "r")
    i = 0
    try:
        for line in f:
            for j in range(ndict):
                if line.rstrip("\n") == dict_values_names[j]:
                    linepos_list[j] = i
            i = i + 1
    finally:
        f.close()

    linepos_list[-1] = i

    print(linepos_list)

    ngrains = linepos_list[1] - linepos_list[0] - 1

    print("ngrains = ", ngrains)

    f = open(filegrains, "rb")
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)

    f.seek(0)
    print(f.readline())

    dict_grains = {}

    for j in range(ndict):

        n = linepos_list[j]
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        print(f.readline())
        f.seek(line_offset[n + 1])
        i = 0
        while i < ngrains:
            toto = f.readline()
            # print toto,
            toto1 = (toto.rstrip("\r\n").split(": "))[1]
            # version string plus lisible pour verif initiale
            # if n == 0 : dict_grains[i] = "[" + toto1 + "]"
            # else : dict_grains[i] = dict_grains[i] + "[" + toto1 + "]"
            # version array
            if n == 0:
                dict_grains[i] = [int(toto1)]
            else:
                if j in listint:
                    toto2 = np.array(toto1.split(" "), dtype=int)
                elif j in listfloat:
                    toto2 = np.array(toto1.split(" "), dtype=float)
                dict_grains[i].append(toto2)
            i = i + 1

    f.close()

    # version string
    # print dict_grains

    # version array
    #    for i in range(ndict):
    #        print dict_values_names[i]
    #        for key, value in dict_grains.iteritems():
    #            print key,value[i]

    print(dict_values_names)

    return (dict_grains, dict_values_names)


def neighbors_list(img, map_imgnum, verbose=0):  # 29May13

    # 8 positions particulieres
    # bords : droit 1 gauche 2 haut 4 bas 8
    # coins  : haut droit 5 haut gauche 6 bas droit 9 bas gauche 10

    #    dict_pixtype = { 0:"center",
    #                     1:"right",
    #                     2:"left"}

    # input : un numero d'image
    # output :
    # ligne colonne pour les pixels cen right left top bottom
    # type de pixel pour cette image : centre 0 / bord 1,2,4,8 / coin 5,6,9,10
    # img num pour les pixels cen right left top bottom
    # avec conditions aux limites periodiques en bord de carto

    if verbose:
        print(shape(map_imgnum))
    mapsize = np.array(shape(map_imgnum), dtype=int)

    # img = 122
    pixtype = 0
    ind1 = where(map_imgnum == img)
    if verbose:
        print("img =", img)
        print(shape(ind1)[1])

    if shape(ind1)[1] == 0:
        print("img not in map")
        uoezazae
    else:
        # print ind1
        ind_cen = np.array([ind1[0][0], ind1[1][0]])
        if verbose:
            print("cen", ind_cen)

        ind_right = ind_cen + array([0, 1])
        ind_left = ind_cen + array([0, -1])
        ind_top = ind_cen + array([-1, 0])
        ind_bottom = ind_cen + array([1, 0])

        listpix = [ind_cen, ind_right, ind_left, ind_top, ind_bottom]
        if ind_right[1] > (mapsize[1] - 1):
            print("img at right edge of map")
            pixtype = pixtype + 1
            ind_right[1] = ind_right[1] - mapsize[1]
        if ind_left[1] < 0:
            print("img at left edge of map")
            pixtype = pixtype + 2
        if ind_top[0] < 0:
            print("img at top edge of map")
            pixtype = pixtype + 4
        if ind_bottom[0] > (mapsize[0] - 1):
            print("img at bottom edge of map")
            pixtype = pixtype + 8
            ind_bottom[0] = ind_bottom[0] - mapsize[0]

        # print listpix

        listpix2 = np.array(listpix, dtype=int)
        list_neighbors = zeros(5, int)
        for i in range(shape(listpix2)[0]):
            list_neighbors[i] = map_imgnum[listpix2[i, 0], listpix2[i, 1]]
        if verbose:
            # ligne colonne pour les pixels cen right left top bottom
            print(listpix2)
            # type de pixel centre 0 / bord 1 2 4 8 / coin 5 6 9 10
            print("pixtype = ", pixtype)
            # img num pour les pixels cen right left top bottom
            print(list_neighbors)
        return (listpix2, pixtype, list_neighbors)


def find_grain_edges(filegrains, filexyz):  # 29May13

    # modifie 28Feb13 : ajout frontiere restreinte

    #    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
    #                         "img_in_grain_list","gnumloc_in_grain_list",\
    #                         "mean_rgb", "std_rgb *1000", "range_rgb *1000"]
    #
    dict_grains, dict_values_names = read_dict_grains(filegrains)

    print(dict_values_names[3])
    for key, value in dict_grains.items():
        print(key, value[3])

    map_imgnum, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)
    # img = 6481
    # listpix, pixtype, list_neighbors = neighbors_list(img, map_imgnum, verbose = 1)
    # jkldsq

    ngrains = len(list(dict_grains.keys()))

    # pixtype, indtest
    dict_neigh = {
        0: [1, 2, 3, 4],
        1: [2, 3, 4],
        2: [1, 3, 4],
        4: [1, 2, 4],
        8: [1, 2, 3],
        5: [2, 4],
        6: [1, 4],
        9: [2, 3],
        10: [1, 3],
    }

    # test
    # ngrains = 2
    dict_grains2 = {}
    # gnum0 = 106

    for gnum in range(ngrains):
        # for gnum in [gnum0,]:

        dict_grains2[gnum] = dict_grains[gnum]

        list_img = dict_grains[gnum][3]
        nimg = len(list_img)
        list_edge = zeros(nimg, dtype=int)
        list_edge_restricted = zeros(nimg, dtype=int)
        bitwise = array([1, 2, 4, 8])

        print("gnum = ", gnum)
        print(dict_values_names[3])
        print(dict_grains[gnum][3])
        print(dict_values_names[4])
        print(dict_grains[gnum][4])

        list_line = zeros(nimg, dtype=int)
        list_col = zeros(nimg, dtype=int)

        gnumloc_list = np.array(dict_grains[gnum][4], dtype=float)

        gnumloc_min = gnumloc_list.min()

        gnumloc_min = int(round(gnumloc_min, 0))
        print("gnumloc_min = ", gnumloc_min)
        gnumloc_mean = gnumloc_list.mean()
        print("gnumloc_mean = ", gnumloc_mean)
        # gnumloc_mean_int = int(gnumloc_mean + 0.5)
        # print "gnumloc_mean_int = ", gnumloc_mean_int

        for i in range(nimg):
            edge1 = zeros(4, dtype=int)
            edge2 = zeros(4, dtype=int)
            img = list_img[i]
            gnumloc = dict_grains[gnum][4][i]

            listpix, pixtype, list_neighbors = neighbors_list(
                img, map_imgnum, verbose=0
            )
            indtest = dict_neigh[pixtype]
            # list_neighbors[indtest] donne les img voisines a tester
            # print pixtype, indtest
            for j in indtest:
                if list_neighbors[j] not in list_img:
                    edge1[j - 1] = 1

            if gnumloc != gnumloc_min:
                edge2[j - 1] = 0
            else:
                for j in indtest:
                    if list_neighbors[j] not in list_img:
                        edge2[j - 1] = 1
                    else:  # list_neighbors[j] in list_img
                        ind0 = where(list_img == list_neighbors[j])
                        # print ind0[0][0]
                        gnumloc_neighbor = dict_grains[gnum][4][ind0[0][0]]
                        if gnumloc_neighbor > gnumloc_min:
                            edge2[j - 1] = 1

            # print edge1
            list_edge[i] = inner(edge1, bitwise)
            list_edge_restricted[i] = inner(edge2, bitwise)
            # print "img, gnumloc, edge, edge_restricted ", img, gnumloc, list_edge[i] , list_edge_restricted[i]

            list_line[i] = listpix[0, 0]
            list_col[i] = listpix[0, 1]
        # print "gnum : ", gnum
        # print "gnumloc : ", dict_grains[gnum][4]
        print("list_edge :", list_edge)
        print("list_edge_restricted :", list_edge_restricted)

        dict_grains2[gnum].append(list_line)
        dict_grains2[gnum].append(list_col)
        dict_grains2[gnum].append(list_edge)
        dict_grains2[gnum].append(round(gnumloc_mean, 2))
        dict_grains2[gnum].append(list_edge_restricted)

    # liste de pixels du grain
    # line_pix col_pix edge_type_pix
    # edge_type_pix : pour frontiere etendue
    # edge_type_pix = 0 pour pixel pas sur frontiere
    # edge_type_pix = 1 a 15 pour pixel sur frontiere
    # 1 2 4 8 code les frontieres right left top bottom
    # somme bitwise des codes si plusieurs bords du pixel sont frontiere simultanement
    toto = [
        "list_line",
        "list_col",
        "list_edge",
        "gnumloc_mean",
        "list_edge_restricted",
    ]
    dict_values_names = dict_values_names + toto

    ndict = len(dict_values_names)

    if 1:
        import module_graphique as modgraph

        outfilegtog = filegrains.rstrip(".txt") + "_with_edges" + ".txt"
        modgraph.filegrain_2 = outfilegtog
        print("results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog, "w")
        for i in range(ndict):
            outputfile.write(dict_values_names[i] + "\n")
            for key, value in dict_grains2.items():
                if (dict_values_names[i] == "grain size") | (
                    dict_values_names[i] == "gnumloc_mean"
                ):
                    str1 = str(value[i])
                else:
                    str1 = " ".join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 + "\n")
        outputfile.close()

    return dict_grains2


def fill_dict_grains(filesum, filegrains):  # 29May13

    dict_grains, dict_values_names = read_dict_grains(filegrains, dict_with_edges="yes")

    list_column_names_to_add = [
        "npeaks",
        "pixdev",
        "intensity",
        "strain6_crystal_0",
        "strain6_crystal_1",
        "strain6_crystal_2",
        "strain6_crystal_3",
        "strain6_crystal_4",
        "strain6_crystal_5",
        "strain6_sample_0",
        "strain6_sample_1",
        "strain6_sample_2",
        "strain6_sample_3",
        "strain6_sample_4",
        "strain6_sample_5",
        "rgb_x_sample_0",
        "rgb_x_sample_1",
        "rgb_x_sample_2",
        "rgb_z_sample_0",
        "rgb_z_sample_1",
        "rgb_z_sample_2",
        "stress6_crystal_0",
        "stress6_crystal_1",
        "stress6_crystal_2",
        "stress6_crystal_3",
        "stress6_crystal_4",
        "stress6_crystal_5",
        "stress6_sample_0",
        "stress6_sample_1",
        "stress6_sample_2",
        "stress6_sample_3",
        "stress6_sample_4",
        "stress6_sample_5",
        "max_rss",
        "von_mises",
    ]

    print(len(list_column_names_to_add))

    data_list, listname, nameline0 = read_summary_file(filesum)

    data_list = np.array(data_list, dtype=float)

    indimg = listname.index("img")
    indgnumloc = listname.index("gnumloc")

    img_list = np.array(data_list[:, indimg], dtype=int)
    gnumloc_list = np.array(data_list[:, indgnumloc], dtype=int)

    #    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
    #                         "img_in_grain_list","gnumloc_in_grain_list",\
    #                         "mean_rgb", "std_rgb *1000", "range_rgb *1000",\
    #                         "list_line", "list_col", "list_edge"]

    print(dict_values_names[0], dict_values_names[3], dict_values_names[4])

    indimg_d = dict_values_names.index("img_in_grain_list")
    indgnumloc_d = dict_values_names.index("gnumloc_in_grain_list")
    indgrainsize_d = dict_values_names.index("grain size")

    dict_grains2 = dict_grains

    # print listname

    for col_name in list_column_names_to_add:
        indcoladd = listname.index(col_name)
        dict_values_names = dict_values_names + [col_name]
        print(col_name)
        if col_name == "npeaks":  # int
            data_col_add = np.array(
                data_list[:, indcoladd].round(decimals=0), dtype=int
            )
            for key, value in dict_grains.items():
                print(key, value[indgrainsize_d])  # , "\n", value[3],"\n", value[4]
                list1 = []
                nimg = value[indgrainsize_d]
                for i in range(nimg):
                    ind1 = where(
                        (img_list == value[indimg_d][i])
                        & (gnumloc_list == value[indgnumloc_d][i])
                    )
                    # print ind1[0][0]
                    j = ind1[0][0]
                    list1.append(data_col_add[j])
                # print list1
                dict_grains2[key].append(list1)
        else:  # float
            data_col_add = np.array(data_list[:, indcoladd], dtype=float)
            # nb of decimals for storage
            if (col_name[:3] == "rgb") | (col_name == "pixdev"):
                ndec = 3
            else:
                ndec = 2
            for key, value in dict_grains.items():
                print(key, value[indgrainsize_d])  # , "\n", value[3],"\n", value[4]
                list1 = []
                nimg = value[indgrainsize_d]
                for i in range(nimg):
                    ind1 = where(
                        (img_list == value[indimg_d][i])
                        & (gnumloc_list == value[indgnumloc_d][i])
                    )
                    # print ind1[0][0]
                    j = ind1[0][0]
                    list1.append(round(data_col_add[j], ndec))
                # print list1
                dict_grains2[key].append(list1)

    ndict = len(dict_values_names)
    print(ndict)

    if 1:
        import module_graphique as modgraph

        outfilegtog = filegrains.rstrip(".txt") + "_filled" + ".txt"
        modgraph.finalfilegrain = outfilegtog
        print("dictionnary results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog, "w")
        for i in range(ndict):
            outputfile.write(dict_values_names[i] + "\n")
            for key, value in dict_grains2.items():
                if i == 0:
                    str1 = str(value[i])
                else:
                    str1 = " ".join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 + "\n")
        outputfile.close()

    return dict_grains2


def add_intragrain_rotations_to_dict_grains(filesum, filegrains):  # 29May13

    dict_grains, dict_values_names = read_dict_grains(
        filegrains, dict_with_edges="yes", dict_with_all_cols="yes"
    )

    print(dict_values_names)

    list_column_names_to_add = ["matstarlab_mean", "misorientation_angle"]

    print(len(list_column_names_to_add))

    data_list, listname, nameline0 = read_summary_file(filesum)

    data_list = np.array(data_list, dtype=float)

    indimg = listname.index("img")
    indgnumloc = listname.index("gnumloc")

    img_list = np.array(data_list[:, indimg], dtype=int)
    gnumloc_list = np.array(data_list[:, indgnumloc], dtype=int)

    #    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
    #                         "img_in_grain_list","gnumloc_in_grain_list",\
    #                         "mean_rgb", "std_rgb *1000", "range_rgb *1000",\
    #                         "list_line", "list_col", "list_edge"]

    # print dict_values_names[0],  dict_values_names[3], dict_values_names[4]

    indimg_d = dict_values_names.index("img_in_grain_list")
    indgnumloc_d = dict_values_names.index("gnumloc_in_grain_list")
    indgrainsize_d = dict_values_names.index("grain size")

    print(indimg_d, indgnumloc_d, indgrainsize_d)

    dict_grains2 = dict_grains

    # print listname

    indmat = listname.index("matstarlab_0")
    dict_values_names = dict_values_names + list_column_names_to_add

    matstarlab_all = np.array(data_list[:, indmat : indmat + 9], dtype=float)

    indangle_d = dict_values_names.index("misorientation_angle")
    indmat_d = dict_values_names.index("matstarlab_mean")

    for key, value in dict_grains.items():
        nimg = value[indgrainsize_d]
        print("gnum ", key, "gsize", nimg)  # , "\n", value[3],"\n", value[4]
        matstarlab_ig = zeros((nimg, 9), dtype=float)
        img_list_d = np.array(value[indimg_d], dtype=int)
        gnumloc_list_d = np.array(value[indgnumloc_d], dtype=int)

        for i in range(nimg):
            ind1 = where(
                (img_list == img_list_d[i]) & (gnumloc_list == gnumloc_list_d[i])
            )
            # print ind1
            # print ind1[0][0]
            j = ind1[0][0]
            matstarlab_ig[i, :] = matstarlab_all[j, :]
            # print matstarlab_1

        matstarlab_mean = matstarlab_ig.mean(axis=0)
        # print "matmean = ", matstarlab_mean

        dict_grains2[key].append(matstarlab_mean.round(decimals=6))

        vec_crystal = zeros((nimg, 3), float)
        vec_lab = zeros((nimg, 3), float)
        angle1 = zeros(nimg, float)
        matmean3x3 = GT.matline_to_mat3x3(matstarlab_mean)
        for k in range(nimg):
            mat2 = GT.matline_to_mat3x3(matstarlab_ig[k, :])
            vec_crystal[k, :], vec_lab[k, :], angle1[k] = twomat_to_rotation(
                matmean3x3, mat2, verbose=0
            )
            # if k == 5 : return()

        dict_grains2[key].append(angle1.round(decimals=3))
        print("angle1 : mean, std, min, max")
        print(
            round(angle1.mean(), 3),
            round(angle1.std(), 3),
            round(angle1.min(), 3),
            round(angle1.max(), 3),
        )

    #        print "new dict entries"
    #        print dict_grains2[key][indmat_d]
    #        print dict_grains2[key][indangle_d]
    #        if key== 2 : return()

    ndict = len(dict_values_names)
    print(ndict)
    print(dict_values_names)

    if 1:
        import module_graphique as modgraph

        outfilegtog = filegrains.rstrip(".txt") + "_with_rotations" + ".txt"
        modgraph.finalfilegrain = outfilegtog
        print("dictionnary results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog, "w")
        for i in range(ndict):
            outputfile.write(dict_values_names[i] + "\n")
            for key, value in dict_grains2.items():
                if i == 0:
                    str1 = str(value[i])
                else:
                    str1 = " ".join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 + "\n")
        outputfile.close()

    return dict_grains2


def testBit(int_type, offset):  # 29May13
    mask = 1 << offset
    return int_type & mask


if 0:  # test testBit function  #29May13
    print(testBit(4, 3))
    print(testBit(1, 1))
    klmdfs


def list_edge_lines(pixel_edge_code):  # 29May13
    # x y cad col line
    # key = bit number
    # 0 : 1, 1 : 2, 2 : 4, 3 : 8
    dict_edge_lines = {
        0: [[1, 0], [1, 1]],
        1: [[0, 0], [0, 1]],
        2: [[0, 1], [1, 1]],
        3: [[0, 0], [1, 0]],
    }

    list_edge_lines = []
    for bit1 in range(4):
        # print testBit(pixel_edge_code,bit1)
        if testBit(pixel_edge_code, bit1) > 0:
            list_edge_lines.append(dict_edge_lines[bit1])

    print(pixel_edge_code)
    print(": ")
    print(list_edge_lines)
    print(",")
    # print shape(list_edge_lines)[0]

    return list_edge_lines


if 0:  # segments for pixel edges for frontiers     #29May13
    for i in range(16):
        list_edge_lines(i)
    jkdlqs

if 1:  # dict_edge_lines  #29May13
    dict_edge_lines = {
        0: [],
        1: [[[1, 0], [1, 1]]],
        2: [[[0, 0], [0, 1]]],
        3: [[[1, 0], [1, 1]], [[0, 0], [0, 1]]],
        4: [[[0, 1], [1, 1]]],
        5: [[[1, 0], [1, 1]], [[0, 1], [1, 1]]],
        6: [[[0, 0], [0, 1]], [[0, 1], [1, 1]]],
        7: [[[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 1], [1, 1]]],
        8: [[[0, 0], [1, 0]]],
        9: [[[1, 0], [1, 1]], [[0, 0], [1, 0]]],
        10: [[[0, 0], [0, 1]], [[0, 0], [1, 0]]],
        11: [[[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 0], [1, 0]]],
        12: [[[0, 1], [1, 1]], [[0, 0], [1, 0]]],
        13: [[[1, 0], [1, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]]],
        14: [[[0, 0], [0, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]]],
        15: [[[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]]],
    }


def plot_all_grain_maps(
    filegrains,
    filexyz,
    zoom="no",
    mapcolor="gnumloc_in_grain_list",
    filter_on_pixdev_and_npeaks=1,
    maxpixdev=0.3,
    minnpeaks=20,
    map_prefix="z_gnumloc_gmap_2edges_",
    test1="yes",
    savefig_map=0,
    number_of_graphs=None,
    number_of_graphs_per_figure=9,
    grains_to_plot="all",
    gnumloc_min=0,
    gnumloc_max=3,
    maxvalue_for_plot=None,
    minvalue_for_plot=None,
    single_gnumloc=None,
    subtract_mean="no",
    filepathout=None,
):
    # 30May13

    """
#        cartos grain par grain avec frontieres entre les grains
#        deux types de frontieres :
#            edge : en noir
            = limite entre (img contenant le grain gnum, independamment du gnumloc) 
            et (img ne contenant pas le grain gnum) 
#            edge_restricted : en jaune
            = limite entre (img contenant le grain gnum avec gnumloc = gnumloc_min(gnum))
#            et (img contenant le grain gnum et gnumloc != gnumloc_min(gnum))
#            ou (img ne contenant pas le grain gnum)
#               
#        le fichier filegrains contient un dictionnaire dict_grains 
#        il a ete construit a partir du fichier colonnes filesum 
#        en utilisant successivement les fonctions :
#        - class_data_into_grainnum(filesum, filepathout, tol1 = 0.1, test_mode = "no")
#        - find_grain_edges(filegrains, filexyz)
#        - fill_dict_grains(filesum, filegrains2)
#        - add_intragrain_rotations_to_dict_grains(filesum, filegrains2)
#        
#        le fichier filesum a ete construit a partir d'une serie de fichiers .fit issus
#        en utilisant successivement les fonctions :
#        - build_summary(indimg, filepathfit, fileprefix, filesuffix, filexyz)
#        - add_columns_to_summary_file(filesum, elem_label = "Ge", filestf = None, verbose = 0)
#        
#        le fichier filesum peut aussi etre construit a partir d'un fichier XMAS
#        en utilisant la fonction :
#        read_xmas_txt_file_from_seq_file(filexmas, 
#                                         read_all_cols = "yes",
#                                         list_column_names = ["GRAININDICE", "ASIMAGEINDICE", "STRNINDEX", "ASSPIXDEV", "ASVONMISES","RSS"])
#        avec l'option : read_all_cols = "yes"
#        (rajouter une fonction qui reorganise les colonnes pour arriver a un filesum
#        semblable a celui issui de build_summary)
#        
#        les fichiers .fit sont issus de serial_index_refine_multigrain applique a une serie
#        de fichiers .dat
#
#        les fichiers .dat sont issus de serial_peak_search applique a une serie d'images
#        (= diagrammes de Laue)        
#        
#        le fichier filexyz est un fichier avec les colonnes "img" "x" "y"
#        qui donne la position x y de chaque image numero img de la carto
#        
#        gnumloc_min = 0 pour fichiers LT, 1 pour fichiers XMAS   
#        gnumloc_max = parametre ngrains du serial_index_refine_multigrain (nb max de grains indexes par image)
#        
#        options de plot_all_grain_maps :
#            
#            - un graphe par grain, N = 4 ou 9 graphes par figure
#          ou  
#            - un graphe pour plusieurs grains dans une liste 
#
#            - sauvegarde des figures dans des .png
#            map_prefix = prefixe pour le nom des figures
#            
#            - filtrage des points de mesure pour garder seulement ceux avec
#            pixdev < maxpixdev et npeaks > minnpeaks
#            
#            dans le mode "un grain par graphe" :
#            - zoom = "no" : pour voir l'emplacement du grain dans la carto
#            - zoom = "yes" : pour zoomer sur chaque grain 
#            
#
#        valeurs possibles pour mapcolor :  (= contenu de dict_values_names)
#        
#        les valeurs entre parentheses sont des quantites moyennes par grain (non cartographiees ici)
#        
#        les valeurs entre doubles parentheses peuvent etre cartographiees mais sans grand interet
#
##    ('grain size'), (('ind_in_grain_list', 'ig_in_grain_list', 'img_in_grain_list')), 
##                         'gnumloc_in_grain_list', 
##    ('mean_rgb', 'std_rgb *1000', 'range_rgb *1000'), (('list_line', 'list_col', 
##    'list_edge')), ('gnumloc_mean'), (('list_edge_restricted')), 
#
##    'npeaks', 'pixdev', 'intensity', 
##    'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 
##    'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
##    'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 
##    'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
##    'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
##    'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
##    'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 
##    'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
##    'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 
##    'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
##    'max_rss', 'von_mises', 'matstarlab_mean', 'misorientation_angle'
#
#        options speciales pour mapcolor :
#            "rgb_x" "rgb_z" 'mean_rgb': orientations en rgb
#            
    """
    # key = number_of_graphs_per_figure
    # nb of subplots lines
    # nb of subplots columns
    # numero du subplot pour mettre les labels des axes
    dict_graph = {9: [3, 3, 7], 4: [2, 2, 3]}

    min_nimg = 2
    max_nimg = 10
    # ind_nimg = range(min_nimg, max_nimg)

    # ind_gnum = [226,139,254,177,140,154,175,193,46,89]
    # ind_gnum = [120,170,256,127,151,163]
    # macles
    # ind_gnum = [201,254,139,127,249,165,180,154,163,130,89,177,140,226,231,148,224,253,30,110,222,196]
    # print len(ind_gnum)
    #    ind_gnum = [228,256,170,175,232,193,202,178,115,168,120,203,151,244,245,102,195,255,58,10,247,181]
    #    print len(ind_gnum)
    # jkldqs

    # definition of map type
    # rgb color for missing data
    # rgb color for filtered data
    dict_maptype = {
        0: [
            "grey scale for positive quantities (black = min)",
            [1.0, 0.8, 0.8],
            [0.0, 1.0, 0.0],
        ],
        1: [
            "inverted grey scale for positive quantities (black = max)",
            [1.0, 0.8, 0.8],
            [0.0, 1.0, 0.0],
        ],
        2: [
            "strain-like color scale for signed quantities (red < min - magenta <0 - white 0 - green > 0 - yellow > max)",
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        3: ["rgb color scale for orientations", [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
        4: ["rgb color from dict_rgb_gnumloc", [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
    }

    # 0 rouge, 1 vert, 2 bleu clair
    # adapter dict_rgb_gnumloc en fonction du gnumloc_max
    if mapcolor == "gnumloc_in_grain_list":
        if gnumloc_min == 0:  # XMAS
            dict_rgb_gnumloc = {
                0: [1.0, 0.0, 0.0],
                1: [0.0, 1.0, 0.0],
                2: [0.5, 0.5, 1.0],
                3: [0.5, 1.0, 0.5],
                4: [0.5, 0.5, 0.8],
            }
        elif gnumloc_min == 1:  # LT
            dict_rgb_gnumloc = {
                1: [1.0, 0.0, 0.0],
                2: [0.0, 1.0, 0.0],
                3: [0.5, 0.5, 1.0],
                4: [0.5, 0.5, 0.5],
            }

    # number of decimals to display
    # maptype
    dict_dec = {
        "pixdev": [4, 1],
        "npeaks": [1, 0],
        "intensity": [1, 0],
        "gnumloc_in_grain_list": [2, 4],
    }

    for key in [
        "strain6_crystal_0",
        "strain6_crystal_1",
        "strain6_crystal_2",
        "strain6_crystal_3",
        "strain6_crystal_4",
        "strain6_crystal_5",
        "strain6_sample_0",
        "strain6_sample_1",
        "strain6_sample_2",
        "strain6_sample_3",
        "strain6_sample_4",
        "strain6_sample_5",
        "stress6_crystal_0",
        "stress6_crystal_1",
        "stress6_crystal_2",
        "stress6_crystal_3",
        "stress6_crystal_4",
        "stress6_crystal_5",
        "stress6_sample_0",
        "stress6_sample_1",
        "stress6_sample_2",
        "stress6_sample_3",
        "stress6_sample_4",
        "stress6_sample_5",
    ]:
        dict_dec[key] = [3, 2]

    for key in ["rgb_x", "rgb_z"]:
        dict_dec[key] = [3, 3]
    for key in ["max_rss", "von_mises", "misorientation_angle"]:
        dict_dec[key] = [3, 0]

    if filter_on_pixdev_and_npeaks == "yes":
        filter_str = "_filter"
    else:
        filter_str = ""
    map_prefix = map_prefix + mapcolor + filter_str + "_gmap_"

    p.rcParams["figure.subplot.bottom"] = 0.1
    p.rcParams["figure.subplot.left"] = 0.1

    dict_grains, dict_values_names = read_dict_grains(
        filegrains,
        dict_with_edges="yes",
        dict_with_all_cols="yes",
        dict_with_all_cols2="yes",
    )

    #    dict_grains, dict_values_names = read_dict_grains(filegrains, dict_with_edges = "yes")

    ngrains = len(list(dict_grains.keys()))

    print("dict_grains.keys()", list(dict_grains.keys()))

    if test1 == "yes":
        ngrains = 12

    print(dict_values_names)

    # grain sizes
    list_gnum = []
    list_grain_size = []
    print(dict_values_names[0])
    for key, value in dict_grains.items():
        # print key,value[0]
        list_gnum.append(key)
        list_grain_size.append(value[0])
    print("list of grain sizes", list_grain_size)
    # jskqdl
    list_grainsize = np.array(list_grain_size, dtype=float)

    indgnumloc = dict_values_names.index("gnumloc_in_grain_list")
    ind_pixdev = dict_values_names.index("pixdev")
    ind_npeaks = dict_values_names.index("npeaks")

    indline = dict_values_names.index("list_line")
    indcol = dict_values_names.index("list_col")
    indedge = dict_values_names.index("list_edge")
    indedge_restricted = dict_values_names.index("list_edge_restricted")

    if mapcolor in ["rgb_x", "rgb_z"]:
        first_col_name = mapcolor + "_sample_0"
        indfirstcol = dict_values_names.index(first_col_name)
        indrgbxz = list(range(indfirstcol, indfirstcol + 3))
    else:
        indplot = dict_values_names.index(mapcolor)

        # calcul des valeurs moyennes de mapcolor sur toute la carto sans filtrage
        list1 = []
        # print dict_values_names[indplot]
        for key, value in dict_grains.items():
            # print key,value[indplot]
            list1 = list1 + list(value[indplot])
        # print list1
        # print len(list1)
        toto = np.array(list1, dtype=float)
        maxvalue = max(toto)
        minvalue = min(toto)
        meanvalue = mean(toto)
        stdvalue = std(toto)

        # print mapcolor[:-1]

        print("statistics on all data points in grains with grain_size > 1 :")
        print(mapcolor)
        ndec = dict_dec[mapcolor][0]
        if mapcolor in ["rgb_x", "rgb_z"]:
            print(
                "mean, std, min, max ",
                meanvalue.round(decimals=ndec),
                stdvalue.round(decimals=ndec),
                minvalue,
                maxvalue,
            )
        else:
            print(
                "mean, std, min, max ",
                round(meanvalue, ndec),
                round(stdvalue, ndec),
                minvalue,
                maxvalue,
            )

        if minvalue_for_plot == None:
            minvalue_for_plot = minvalue
        if maxvalue_for_plot == None:
            maxvalue_for_plot = maxvalue
        print("min, max for plot", minvalue_for_plot, maxvalue_for_plot)

    maptype = dict_dec[mapcolor][1]
    print("type of color map : ")
    print(dict_maptype[maptype][0])
    color_for_missing_data = dict_maptype[maptype][1]
    print("rgb color for missing data :", color_for_missing_data)

    if filter_on_pixdev_and_npeaks == "yes":
        print("filtered map")
        print("maxpixdev, minnpeaks = ", maxpixdev, minnpeaks)
        color_for_filtered_data = dict_maptype[maptype][2]
        print("rgb color for filtered data :", color_for_filtered_data)

    # jkldqs

    #    print dict_values_names[3]
    #    for key, value in dict_grains.iteritems():
    #        print key,value[3]
    #

    map_imgnum, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)

    nlines = shape(map_imgnum)[0]
    ncol = shape(map_imgnum)[1]

    xrange1 = array([0.0, ncol * dxystep[0]])
    yrange1 = array([0.0, nlines * dxystep[1]])
    xmin, xmax = min(xrange1), max(xrange1)
    ymin, ymax = min(yrange1), max(yrange1)
    extent = xmin, xmax, ymin, ymax
    print(extent)
    dxystep_abs = abs(dxystep)

    shape_rgb = (nlines, ncol, 3)

    # decalage entre x y entier et centre pixel suivant position origine x y dans carto
    # haut / bas, droite / gauche
    # impos_start = 0 0 haut gauche, 0 80 haut droit, 100 0 bas gauche, 100 80 bas droit

    if impos_start[1] == 0:
        x_coin_carto = "gauche"
    else:
        x_coin_carto = "droit"
    if impos_start[0] == 0:
        y_coin_carto = "haut"
    else:
        y_coin_carto = "bas"

    if x_coin_carto == "gauche":
        dx_corner_to_center = 0.5
    else:
        dx_corner_to_center = -0.5
    if y_coin_carto == "haut":
        dy_corner_to_center = -0.5
    else:
        dy_corner_to_center = 0.5

    dxy_corner_to_center_microns = [
        dx_corner_to_center * dxystep_abs[0],
        dy_corner_to_center * dxystep_abs[1],
    ]

    if number_of_graphs == 1:
        p.figure(figsize=(10, 10))

    rgb1 = color_for_missing_data * ones(shape_rgb, dtype=float)

    print("grains_to_plot", grains_to_plot)

    if grains_to_plot == "all":
        indgnum = list(range(ngrains))
    else:
        indgnum = grains_to_plot

    print(indgnum)

    listx = []
    listy = []
    k = 0
    kk = 0
    for gnum in indgnum:

        rgb_rand = rand(3)
        rgb_rand = rgb_rand / max(rgb_rand)

        if ((zoom == "yes") & (number_of_graphs == None)) | (number_of_graphs == 1):
            listx = []
            listy = []

        list_img = dict_grains[gnum][3]
        nimg = len(list_img)

        #        if nimg in ind_nimg :

        list_line = np.array(dict_grains[gnum][indline], dtype=int)
        list_col = np.array(dict_grains[gnum][indcol], dtype=int)
        list_edge = np.array(dict_grains[gnum][indedge], dtype=int)
        list_edge_restricted = np.array(
            dict_grains[gnum][indedge_restricted], dtype=int
        )
        if mapcolor in ["rgb_x", "rgb_z"]:
            list_plot = np.array(
                [
                    dict_grains[gnum][indrgbxz[0]],
                    dict_grains[gnum][indrgbxz[1]],
                    dict_grains[gnum][indrgbxz[2]],
                ],
                dtype=float,
            )
            list_plot = list_plot.transpose()
            meanvalue = list_plot.mean(axis=0).round(decimals=3)
        #            stdvalue = list_plot.std(axis=0).round(decimals=3)
        #            minvalue = list_plot.min(axis=0).round(decimals=3)
        #            maxvalue = list_plot.max(axis=0).round(decimals=3)
        #            print "mean, std, min, max"
        #            print meanvalue
        #            print stdvalue
        #            print minvalue
        #            print maxvalue
        # print list_plot
        else:
            list_plot = np.array(dict_grains[gnum][indplot], dtype=float)
            meanvalue = round(list_plot.mean(), 3)
            # print "list_plot ", list_plot

        if subtract_mean == "yes":
            list_plot = list_plot - meanvalue
            # print "list_plot - mean ", list_plot

        if filter_on_pixdev_and_npeaks == "yes":
            list_pixdev = np.array(dict_grains[gnum][ind_pixdev], dtype=float)
            list_npeaks = np.array(dict_grains[gnum][ind_npeaks], dtype=float)

        if impos_start[1] == 0:
            list_x = list_col * dxystep[0]
        else:
            list_x = (impos_start[1] - list_col) * dxystep[0]
        if impos_start[0] == 0:
            list_y = list_line * dxystep[1]
        else:
            list_y = (impos_start[0] - list_line) * dxystep[1]

        # print dict_values_names[5]
        # print dict_grains[gnum][5]
        # pour rajouter l'info d'orientation
        # rgb1[0,0,:]= dict_grains[gnum][5][:3]   # rgb x
        # rgb1[0,1,:]= dict_grains[gnum][5][3:]   # rgb z

        print("gnum = ", gnum)
        #        print dict_values_names[3]
        #        print dict_grains[gnum][3]
        #        print dict_values_names[4]
        #        print dict_grains[gnum][4]
        #
        #        #list_edge_for_plot = zeros((nimg,3), dtype = int)
        #        list_line = zeros(nimg, dtype=int)
        #        list_col = zeros(nimg, dtype=int)

        if single_gnumloc != None:
            gnumloc_list = dict_grains[gnum][indgnumloc]
            gnumloc_min1 = min(gnumloc_list)
            if gnumloc_min1 != single_gnumloc:
                print(
                    "skip grain : gnumloc_min1 , single_gnumloc : ",
                    gnumloc_min1,
                    single_gnumloc,
                )
                continue

        if number_of_graphs == None:
            rgb1 = color_for_missing_data * ones(shape_rgb, dtype=float)

        for i in range(nimg):
            img = list_img[i]
            gnumloc = dict_grains[gnum][indgnumloc][i]

            if single_gnumloc != None:
                if gnumloc != single_gnumloc:
                    continue

            ind1 = where(list_img == img)

            if maptype == 0:
                rgb1[list_line[i], list_col[i], :] = (
                    (list_plot[i] - minvalue_for_plot)
                    / (maxvalue_for_plot - minvalue_for_plot)
                ).clip(min=0.0, max=1.0)
            elif maptype == 1:
                rgb1[list_line[i], list_col[i], :] = (
                    (maxvalue_for_plot - list_plot[i])
                    / (maxvalue_for_plot - minvalue_for_plot)
                ).clip(min=0.0, max=1.0)
            elif maptype == 2:
                toto = (
                    (list_plot[i] - minvalue_for_plot)
                    / (maxvalue_for_plot - minvalue_for_plot)
                ).clip(min=0.0, max=1.0)
                rgb1[list_line[i], list_col[i], :] = np.array(cmap(toto))[:3]
                if list_plot[i] > maxvalue_for_plot:
                    rgb1[list_line[i], list_col[i], :] = [1.0, 1.0, 0.0]  # yellow
                elif list_plot[i] < minvalue_for_plot:
                    rgb1[list_line[i], list_col[i], :] = [1.0, 0.0, 0.0]  # red
            elif maptype == 3:
                rgb1[list_line[i], list_col[i], :] = list_plot[i, :] * 1.0
            elif maptype == 4:
                if single_gnumloc != None:
                    if gnumloc == single_gnumloc:
                        rgb1[list_line[i], list_col[i], :] = rgb_rand
                else:
                    # marque en jaune les doublons
                    # = deux gnumloc pour un meme couple (img, gnum)
                    if shape(ind1)[1] == 1:
                        rgb1[list_line[i], list_col[i], :] = dict_rgb_gnumloc[gnumloc]
                    else:
                        rgb1[list_line[i], list_col[i], :] = [1.0, 1.0, 0.0]  # jaune

            if filter_on_pixdev_and_npeaks == "yes":
                if (list_pixdev[i] > maxpixdev) | (list_npeaks[i] < minnpeaks):
                    rgb1[list_line[i], list_col[i], :] = color_for_filtered_data

        if 1:  # all grains - all maps but last
            if number_of_graphs_per_figure > 1:
                if not (kk % number_of_graphs_per_figure):
                    if kk > 0:
                        kkstart = kk - number_of_graphs_per_figure
                        kkend = kk - 1
                        figfilename = os.path.join(
                            filepathout,
                            str(map_prefix)
                            + str(k)
                            + "_g_"
                            + str(kkstart)
                            + "_to_"
                            + str(kkend)
                            + ".png",
                        )
                        try:
                            import module_graphique as modgraph

                            modgraph.outgraphgrain = figfilename
                        except ImportError:
                            pass

                        print(figfilename)
                        if savefig_map:
                            p.savefig(figfilename, bbox_inches="tight")
                    p.figure(figsize=(10, 10))
                    k = k + 1

                gnum1to9 = 1 + kk % number_of_graphs_per_figure
                # print "gnum1to9 = ", gnum1to9
                ax = p.subplot(
                    dict_graph[number_of_graphs_per_figure][0],
                    dict_graph[number_of_graphs_per_figure][1],
                    gnum1to9,
                )
                p.subplots_adjust(hspace=0.01, wspace=0.01)

            imrgb = p.imshow(rgb1, interpolation="nearest", extent=extent)

            if number_of_graphs_per_figure > 1:
                if gnum1to9 != dict_graph[number_of_graphs_per_figure][2]:
                    ax.axes.get_xaxis().set_ticks([])
                    ax.axes.get_yaxis().set_ticks([])

            # trace des frontieres

            for i in range(nimg):
                gnumloc = dict_grains[gnum][indgnumloc][i]
                if single_gnumloc != None:
                    if gnumloc != single_gnumloc:
                        continue

                # print list_col[i], list_line[i], list_x[i], list_y[i], list_edge[i]
                xcen1 = list_x[i] + dxy_corner_to_center_microns[0]
                ycen1 = list_y[i] + dxy_corner_to_center_microns[1]
                #                p.text(xcen1, ycen1, \
                #                str(list_edge[i]),fontsize = 10, ha = 'center', va = 'center' )

                if single_gnumloc == None:
                    # extended frontiers
                    list_edge_lines = dict_edge_lines[list_edge[i]]
                    #                print list_edge_lines
                    #                print list_edge_lines[0]
                    #                print list_edge_lines[0][0]
                    #                print list_edge_lines[0][0][0]
                    num_segments = shape(list_edge_lines)[0]

                    for j in range(num_segments):
                        first_point_x = (
                            xcen1 + (list_edge_lines[j][0][0] - 0.5) * dxystep_abs[0]
                        )
                        first_point_y = (
                            ycen1 + (list_edge_lines[j][0][1] - 0.5) * dxystep_abs[1]
                        )
                        last_point_x = (
                            xcen1 + (list_edge_lines[j][1][0] - 0.5) * dxystep_abs[0]
                        )
                        last_point_y = (
                            ycen1 + (list_edge_lines[j][1][1] - 0.5) * dxystep_abs[1]
                        )
                        listx.append(first_point_x)
                        listx.append(last_point_x)
                        listy.append(first_point_y)
                        listy.append(last_point_y)
                        xx = [first_point_x, last_point_x]
                        yy = [first_point_y, last_point_y]
                        # strcolor = rgb_rand
                        strcolor = "k"
                        p.plot(xx, yy, color=strcolor, linestyle="-")

                # restricted frontiers for gnumloc = gnumloc_min
                if ((zoom == "yes") & (number_of_graphs == None)) | (
                    number_of_graphs == 1
                ):
                    list_edge_lines = dict_edge_lines[list_edge_restricted[i]]
                    #                print list_edge_lines
                    #                print list_edge_lines[0]
                    #                print list_edge_lines[0][0]
                    #                print list_edge_lines[0][0][0]
                    num_segments = shape(list_edge_lines)[0]

                    for j in range(num_segments):
                        first_point_x = (
                            xcen1 + (list_edge_lines[j][0][0] - 0.5) * dxystep_abs[0]
                        )
                        first_point_y = (
                            ycen1 + (list_edge_lines[j][0][1] - 0.5) * dxystep_abs[1]
                        )
                        last_point_x = (
                            xcen1 + (list_edge_lines[j][1][0] - 0.5) * dxystep_abs[0]
                        )
                        last_point_y = (
                            ycen1 + (list_edge_lines[j][1][1] - 0.5) * dxystep_abs[1]
                        )
                        listx.append(first_point_x)
                        listx.append(last_point_x)
                        listy.append(first_point_y)
                        listy.append(last_point_y)
                        xx = [first_point_x, last_point_x]
                        yy = [first_point_y, last_point_y]
                        # strcolor = rgb_rand
                        #                        #if single_gnumloc != None : strcolor = 'k'
                        #                        if single_gnumloc != None : strcolor = 'r'
                        #                        else :
                        strcolor = array([1.0, 1.0, 0.0])
                        p.plot(xx, yy, color=strcolor, linestyle="-")

            if number_of_graphs_per_figure == 1:
                xmean1 = mean(listx)
                ymean1 = mean(listy)
                p.text(xmean1, ymean1, str(gnum), ha="center", va="center", color="r")
                p.title(mapcolor)

            if number_of_graphs_per_figure > 1:
                if zoom == "yes":
                    # print listx, listy
                    xmin1 = min(listx) - dxystep_abs[0]
                    xmax1 = max(listx) + dxystep_abs[0]
                    ymin1 = min(listy) - dxystep_abs[1]
                    ymax1 = max(listy) + dxystep_abs[1]
                    # xceng = (xmin1+xmax1)/2.0
                    # yceng = (ymin1+ymax1)/2.0
                    p.text(
                        xmin1 + dxystep_abs[0] / 2.0,
                        ymin1 + dxystep_abs[1] / 2.0,
                        str(gnum),
                        ha="center",
                        va="center",
                    )
                    p.xlim(xmin1, xmax1)
                    p.ylim(ymin1, ymax1)
                if zoom == "no":
                    # p.text(2.5,-2.5,str(gnum))  # UO2
                    p.text(-50, -50, str(gnum), ha="center", va="center")  # CdTe

                if gnum1to9 == dict_graph[number_of_graphs_per_figure][2]:
                    p.xlabel("dxech (microns)")
                    p.ylabel("dyech (microns)")
                    if zoom == "no":
                        p.xlim(xmin, xmax)
                        p.ylim(ymin, ymax)
                    ax.locator_params("x", tight=True, nbins=5)
                    ax.locator_params("y", tight=True, nbins=5)
            else:
                p.xlabel("dxech (microns)")
                p.ylabel("dyech (microns)")
                p.xlim(xmin, xmax)
                p.ylim(ymin, ymax)

        kk = kk + 1
    # fin du if nimg in ind_nimg

    if number_of_graphs_per_figure > 1:
        #    if 1 : # all grains - last map
        kkstart = kk - 1 - int(kk / number_of_graphs_per_figure)
        kkend = kk - 1
        figfilename = (
            filepathout
            + map_prefix
            + str(k)
            + "_g_"
            + str(kkstart)
            + "_to_"
            + str(kkend)
            + ".png"
        )
        try:
            import module_graphique as modgraph

            modgraph.outgraphgrain = figfilename
        except ImportError:
            pass
        print(figfilename)
        if savefig_map:
            p.savefig(figfilename, bbox_inches="tight")

    if 0:  # test with only one grain
        fig = p.plt.figure(frameon=True, figsize=(8, 8))
        imrgb = p.imshow(rgb1, interpolation="nearest", extent=extent)
        for i in range(nimg):
            print(list_col[i], list_line[i], list_x[i], list_y[i], list_edge[i])
            xcen1 = list_x[i] + dxy_corner_to_center_microns[0]
            ycen1 = list_y[i] + dxy_corner_to_center_microns[1]
            p.text(
                xcen1, ycen1, str(list_edge[i]), fontsize=10, ha="center", va="center"
            )
            list_edge_lines = dict_edge_lines[list_edge[i]]
            #                print list_edge_lines
            #                print list_edge_lines[0]
            #                print list_edge_lines[0][0]
            #                print list_edge_lines[0][0][0]
            num_segments = shape(list_edge_lines)[0]
            for j in range(num_segments):
                first_point_x = (
                    xcen1 + (list_edge_lines[j][0][0] - 0.5) * dxystep_abs[0]
                )
                first_point_y = (
                    ycen1 + (list_edge_lines[j][0][1] - 0.5) * dxystep_abs[1]
                )
                last_point_x = xcen1 + (list_edge_lines[j][1][0] - 0.5) * dxystep_abs[0]
                last_point_y = ycen1 + (list_edge_lines[j][1][1] - 0.5) * dxystep_abs[1]
                xx = [first_point_x, last_point_x]
                yy = [first_point_y, last_point_y]
                p.plot(xx, yy, "k-")

        p.xlabel("dxech (microns)")
        p.ylabel("dyech (microns)")
        p.text(2.5, -2.5, str(gnum))

    return 0


def plot_strain_stress_color_bar(bar_legend="strain"):

    # Make a colorbar as a separate figure. (for strain maps)
    cdict = {
        "red": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 0.0)),
        "green": ((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "blue": ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
    }

    """
    Make a colorbar as a separate figure.
    """
    # cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    # Make a figure and axes with dimensions as desired.
    # fig = pyplot.figure(figsize=(3,8)) # vertical bar
    fig = pyplot.figure(figsize=(8, 3))  # horizontal bar
    # ax1 = fig.add_axes([0.05, 0.05, 0.15, 0.5])  # vertical bar
    # ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    ax1 = fig.add_axes([0.05, 0.3, 0.5, 0.15])  # horizontal bar

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    # cmap = mpl.cm.Greys
    cmap = mpl.cm.PiYG

    # norm = mpl.colors.Normalize(vmin=0, vmax=0.25)
    # norm = mpl.colors.Normalize(vmin=0.15, vmax=0.45)
    norm = mpl.colors.Normalize(vmin=-0.2, vmax=0.2)
    # cmap.set_over(color = "r")
    cmap.set_over(color=[1.0, 1.0, 0.0])  # yellow
    cmap.set_under(color="r")

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(
        ax1,
        cmap=cmap,
        norm=norm,
        extend="both",
        # ticks = [0.,0.1,0.2, 0.25],
        ticks=[-0.2, 0.0, 0.2],
        spacing="proportional",
        orientation="horizontal",
    )

    # cb1.set_label('rotation angle (degrees)', fontsize = 20)
    if bar_legend == "strain":
        cb1.set_label("strain (0.1%)", fontsize=20)
    elif bar_legend == "stress":
        cb1.set_label("strain (100 MPa)", fontsize=20)
    # c0 = mpl.artist.getp(cb1.ax, 'ymajorticklabels') # vertical bar
    c0 = mpl.artist.getp(cb1.ax, "xmajorticklabels")  # horizontal bar
    mpl.artist.setp(c0, fontsize=20)

    return ()


def tworefl_to_mat(hkl, xydat, calib):

    # modif 04 Mar 2010

    # reseau cubique uniquement

    print("calculate orientation matrix from two reflections")
    print("hkl = \n", hkl)
    print("xy = \n", xydat)

    uqlab1 = xycam_to_uqlab(xydat[0, :], calib)
    uqlab2 = xycam_to_uqlab(xydat[1, :], calib)

    uqcr1 = hkl[0, :] / norme(hkl[0, :])
    uqcr2 = hkl[1, :] / norme(hkl[1, :])

    qcr3 = cross(uqcr1, uqcr2)
    uqcr3 = qcr3 / norme(qcr3)

    qlab3 = cross(uqlab1, uqlab2)
    uqlab3 = qlab3 / norme(qlab3)

    uqlab2b = cross(uqlab3, uqlab1)

    uqcr2b = cross(uqcr3, uqcr1)

    # print "normes :", norme(uqlab1), norme(uqlab2), norme(uqcr1), norme(uqcr2)
    # print norme(uqcr3), norme(uqlab3), norme(uqlab2b)

    # print "inner products :", inner(uqlab1, uqlab2b), inner(uqlab2b, uqlab3), inner(uqlab3,uqlab1)
    # print inner(uqcr1,uqcr2b), inner (uqcr2b,uqcr3), inner(uqcr3, uqcr1)

    crtoRq = vstack((uqcr1, uqcr2b, uqcr3))
    # print "crtoRq = \n", crtoRq
    Rqtocr = np.linalg.inv(crtoRq)
    # print "Rqtocr = \n", Rqtocr

    astarlab = Rqtocr[0, 0] * uqlab1 + Rqtocr[0, 1] * uqlab2b + Rqtocr[0, 2] * uqlab3
    bstarlab = Rqtocr[1, 0] * uqlab1 + Rqtocr[1, 1] * uqlab2b + Rqtocr[1, 2] * uqlab3
    cstarlab = Rqtocr[2, 0] * uqlab1 + Rqtocr[2, 1] * uqlab2b + Rqtocr[2, 2] * uqlab3

    matstarlab = hstack((astarlab, bstarlab, cstarlab)) / norme(astarlab)

    print("matstarlab = \n", matstarlab)

    return matstarlab


def twomat_to_rotation(mat1, mat2, verbose=1):

    # from 3x3 matrixes
    # to get vector in crystal coordinates
    # no need to say which crystal because HKL of rotation axis is the same in the two crystal
    # (only axis that stays fixed during rotation)
    # abc2 en colonnes sur abc1

    mat12 = dot(np.linalg.inv(mat1), mat2)  # version OR
    quat12 = fromMatrix_toQuat(mat12)
    unitvec, angle = fromQuat_to_vecangle(quat12)
    ind1 = argmax(abs(unitvec))
    vec_crystal = unitvec / unitvec[ind1]

    angle1 = angle * 180.0 / math.pi

    mat12 = dot(mat1, np.linalg.inv(mat2))  # version CK
    quat12 = fromMatrix_toQuat(mat12)
    unitvec, angle = fromQuat_to_vecangle(quat12)
    ind1 = argmax(abs(unitvec))
    vec_lab = unitvec / unitvec[ind1]

    if verbose:
        print("rotation")
        print("vector (crystal)   vector (lab)   angle (deg)")
        print(
            vec_crystal.round(decimals=6),
            vec_lab.round(decimals=6),
            "\t",
            round(angle1, 5),
        )  # ,round(angle2,3)

    return (vec_crystal / norme(vec_crystal), vec_lab / norme(vec_lab), angle1)


if 0:  # creation du fichier resume des positions des 4 pics pour toutes les images

    hkl4C = array(
        [[4.0, 0.0, -2.0], [2.0, 0.0, 0.0], [5.0, -1.0, -1.0], [4.0, -2.0, -4.0]]
    )
    xy4C = array(
        [
            [878.241, 1039.591],
            [197.199, 1196.644],
            [458.248, 677.17],
            [1479.827, 549.604],
        ]
    )

    imgstart, imgend = 2828, 6362  # grain C
    xytol = 20.0
    # xytol = 30.0

    filepathdat = "D:\\LT\\Gael\\dossier_Odile\\tri_et1+et1bis_40\\"
    fileprefix = "tri_et1_40_"
    filesuffix = ".DAT"

    imgnum = list(range(imgstart, imgend + 1))

    numim = imgend - imgstart + 1

    xy4all = zeros((numim, 8), float)

    for i in range(numim):
        filedat1 = filepathdat + fileprefix + rmccd.stringint(imgnum[i], 4) + filesuffix
        data_xyexp, data_int = read_dat(filedat1, filetype="XMAS")
        npics = shape(data_xyexp)[0]
        if np.isscalar(data_xyexp[0]):
            continue
        for j in range(4):
            # dxy = data_xyexp-xy4A[j,:]
            # dxy = data_xyexp-xy4B[j,:]
            dxy = data_xyexp - xy4C[j, :]
            for k in range(npics):
                if norme(dxy[k, :]) < xytol:
                    xy4all[i, j * 2 : j * 2 + 2] = data_xyexp[k, :]
                    break
        # #                print "i = ", i
        # #                print "j = ", j
        # #                print "k = ", k
        print(xy4all[i, :])

    print(xy4all)

    toto = column_stack((imgnum, xy4all))
    fileout = filepath + "et1_xy4C_img" + str(imgstart) + "_" + str(imgend) + ".dat"

    savetxt(fileout, toto, fmt="%.4f")
    jkldsqsd

if 0:  # matrices a partir des positions de 2 pics de HKL connus

    fileout = filepath + "et1_xy4C_img2828_6362.dat"

    data_all = loadtxt(fileout)
    xy4all = data_all[:, 1:]
    imgall = data_all[:, 0]
    nimg = shape(data_all)[0]

    # nimg = 50

    # npic1, npic2 = 0, 2
    npic1, npic2 = 1, 3

    # hkl = array([hkl4A[npic1,:],hkl4A[npic2,:]])
    # hkl = array([hkl4B[npic1,:],hkl4B[npic2,:]])
    hkl = array([hkl4C[npic1, :], hkl4C[npic2, :]])
    matstarlab_all = zeros((nimg, 9), float)

    xy2 = column_stack(
        (xy4all[:, npic1 * 2 : npic1 * 2 + 2], xy4all[:, npic2 * 2 : npic2 * 2 + 2])
    )

    isbadimg = zeros(nimg, int)
    for i in range(nimg):
        xydat = xy2[i, :].reshape(2, 2)
        if xydat.all() != 0.0:
            print(xydat)
            matstarlab_all[i, :] = tworefl_to_mat(hkl, xydat, calib)
        else:
            isbadimg[i] = 1

    print(matstarlab_all.round(decimals=6))
    print(isbadimg)
    ind1 = where(isbadimg == 0)
    print(ind1[0])

    toto = column_stack((imgall[ind1[0]], matstarlab_all[ind1[0], :].round(decimals=6)))
    fileout = (
        filepath + "et1_mat2C_img2828_6362_p" + str(npic1) + "_p" + str(npic2) + ".dat"
    )
    savetxt(fileout, toto, fmt="%.6f")

    jkldsq

if 0:  # desorientation par rapport a la matrice moyenne

    fileout = filepath + "et1_mat2C_img2828_6362_p1_p3.dat"

    data_all = loadtxt(fileout)
    matall = data_all[:, 1:]
    imgall = data_all[:, 0]
    nimg = shape(data_all)[0]

    print(nimg)

    # nimg = 10

    vec_crystal = zeros((nimg, 3), float)
    vec_lab = zeros((nimg, 3), float)
    angle1 = zeros(nimg, float)
    matmean = matall.mean(axis=0)

    print("matmean = ", matmean)
    matmean3x3 = F2TC.matline_to_mat3x3(matmean)

    for k in range(nimg):
        mat2 = F2TC.matline_to_mat3x3(matall[k, :])
        vec_crystal[k, :], vec_lab[k, :], angle1[k] = twomat_to_rotation(
            matmean3x3, mat2
        )

    # print shape(imgall), shape(vec_crystal), shape(vec_lab), shape(angle1)

    toto = column_stack((imgall[:nimg], vec_crystal, vec_lab, angle1))

    fileout = filepath + "et1_C_desorient_img2828_6362_p1_p3.dat"
    savetxt(fileout, toto, fmt="%.6f")

    jklqsd


def uflab_to_2thetachi(uflab):

    # 23May11 : go to JSM convention for chi

    uflabyz = array([0.0, uflab[1], uflab[2]])
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab
    # chi2 = (180.0/math.pi)*arctan(uflab[0]/norme(uflabyz))

    # JSM convention : angle dans le plan xz entre les projections de uflab suivant x et suivant z
    # OR change sign of chi
    EPS = 1e-17
    chi2 = (180.0 / math.pi) * arctan(uflab[0] / (uflab[2] + EPS))  # JSM convention

    twicetheta2 = (180.0 / math.pi) * arccos(uflab[1])

    # #    chi3 = (180.0/PI)*arccos(inner(uflab,uflabyz)/norme(uflabyz))*sign(uflab[0])

    # #    print "uflab =", uflab
    # #    print "2theta, theta, chi en deg", twicetheta2 , chi2, twicetheta2/2.0
    # #    print "chi3 = ", chi3

    return (chi2, twicetheta2)


def uflab_to_xycam_gen(uflab, calib, uflab_cen, pixelsize=0.08056640625):

    # 08Jun12 add variable uflab_cen
    # modif 04 Mar 2010 xbet xgam en degres au lieu de radians

    # XMAS PCIF6 changer le signe de xgam
    # laptop OR garder le meme signe pour xgam

    detect = calib[0] * 1.0
    xcen = calib[1] * 1.0
    ycen = calib[2] * 1.0
    xbet = calib[3] * 1.0
    xgam = calib[4] * 1.0

    # #    print "Correcting the data according to the parameters"
    # #    print "xcam, ycam in XMAS convention"
    # #
    # #    print "detect in mm" , detect
    # #    print "xcen in pixels" , xcen
    # #    print "ycen in pixels" , ycen
    # #    print "xbet in degrees" , xbet
    # #    print "xgam in degrees" , xgam

    PI = math.pi

    uilab = array([0.0, 1.0, 0.0])

    xbetrad = xbet * PI / 180.0
    xgamrad = xgam * PI / 180.0

    cosbeta = cos(PI / 2.0 - xbetrad)
    sinbeta = sin(PI / 2.0 - xbetrad)
    cosgam = cos(-xgamrad)
    singam = sin(-xgamrad)

    uflab_cen2 = zeros(3, float)
    tthrad0 = acos(uflab_cen[1])
    tthrad = tthrad0 - xbetrad
    uflab_cen2[1] = cos(tthrad)
    uflab_cen2[0] = uflab_cen[0] / sin(tthrad0) * sin(tthrad)
    uflab_cen2[2] = uflab_cen[2] / sin(tthrad0) * sin(tthrad)

    # print "norme(uflab_cen2) = ", norme(uflab_cen2)

    # IOlab = detect * array([0.0, cosbeta, sinbeta])
    IOlab = detect * uflab_cen2

    # unlab = IOlab/norme(IOlab)

    # normeIMlab = detect / inner(uflab,unlab)

    normeIMlab = detect / inner(uflab, uflab_cen2)

    # uflab1 = array([-uflab[0],uflab[1],uflab[2]])

    # uflab1 = uflab*1.0

    # IMlab = normeIMlab*uflab1

    IMlab = normeIMlab * uflab

    OMlab = IMlab - IOlab

    # print "inner(OMlab,uflab_cen2) = ", inner(OMlab,uflab_cen2)

    # jusqu'ici on definissait xlab = xcam0 (avant rotation xgam) par la perpendiculaire au plan ui, uflab_cen
    # ici on change

    uxcam0 = cross(uilab, uflab_cen2)
    uxcam0 = uxcam0 / norme(uxcam0)

    xca0 = inner(OMlab, uxcam0)

    # xca0 = OMlab[0]

    # calculer en dehors : IOlab, uflab_cen2, uxcam0, uycam0 pour eviter de repeter ces calculs
    uycam0 = cross(uflab_cen2, uxcam0)
    uycam0 = uycam0 / norme(uycam0)

    # yca0 = OMlab[1]/sinbeta

    yca0 = inner(OMlab, uycam0)

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / pixelsize

    # uflabyz = array([0.0, uflab1[1],uflab1[2]])
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab

    # chi = (180.0/PI)*arctan(uflab1[0]/norme(uflabyz))
    # twicetheta = (180.0/PI)*arccos(uflab1[1])
    # th0 = twicetheta/2.0

    # print "2theta, theta, chi en deg", twicetheta , chi, twicetheta/2.0
    # print "xcam, ycam = ", xcam, ycam

    xycam = array([xcam, ycam])

    return xycam


def spotlist_gen(
    Emin,
    Emax,
    diagr,
    matwithlatpar,
    cryst_struct,
    showall,
    calib,
    pixelsize=0.08056640625,
    remove_harmonics="yes",
):

    # 08Jun12 uflab_to_xycam_gen for more general calculation
    # modif 04 Mar 2010
    # 20Oct10 : nouveau coeff E_eV_fois_lambda_nm plus precis
    # 21Oct10 : input matrix avec parametres de maille integres
    # 23May11 : add returnmore for 2theta chi

    # structures traitees ici : FCC, BCC, diamant (pour les extinctions)
    # maille deformee OK
    # Emin , Emax en KeV

    nmaxspots = 500
    limangle = 70
    cosangle = cos(limangle * math.pi / 180.0)

    mat = matwithlatpar

    # Rlab z vers le haut, x vers le back, y vers l'aval

    if diagr == "side":
        uflab_cen = array([-1.0, 0.0, 0.0])
    if diagr == "top":
        uflab_cen = array([0.0, 0.0, 1.0])
    if diagr == "halfback":  # 2theta = 118
        # 0 -sin28 cos28
        tth = 28 * math.pi / 180.0
        uflab_cen = array([0.0, -sin(tth), cos(tth)])

    uflab_cen = uflab_cen / norme(uflab_cen)

    uilab = array([0.0, 1.0, 0.0])

    uqlab_cen = uflab_cen - uilab
    uqlab_cen = uqlab_cen / norme(uqlab_cen)

    if showall:
        print("calculate theoretical Laue pattern from orientation matrix")
        print("use matrix (with strain) \n", mat)
        print("energy range :", Emin, Emax)
        print("max angle between uflab and uflab_cen (deg) : ", limangle)
        print("uflab_cen =", uflab_cen)
        print("diagram : ", diagr)
        print("structure :", cryst_struct)

    # print "cosangle = ", cosangle

    hkl = zeros((nmaxspots, 3), int)
    uflab = zeros((nmaxspots, 3), float)
    xy = zeros((nmaxspots, 2), float)
    Etheor = zeros(nmaxspots, float)
    ththeor = zeros(nmaxspots, float)
    tth = zeros(nmaxspots, float)
    chi = zeros(nmaxspots, float)

    dlatapprox = 1.0 / norme(mat[0:3])
    print("dlatapprox = ", dlatapprox)

    Hmax = int(dlatapprox * 2 * Emax / 1.2398)

    if showall:
        print("Hmax = ", Hmax)

    nspot = 0

    for H in range(-Hmax, Hmax):
        for K in range(-Hmax, Hmax):
            if (not (K - H) % 2) | (cryst_struct == "BCC"):
                for L in range(-Hmax, Hmax):
                    if (not (L - H) % 2) | (cryst_struct == "BCC"):
                        if (
                            (cryst_struct == "FCC")
                            | (
                                (cryst_struct == "diamond")
                                & ((H % 2) | ((not H % 2) & (not (H + K + L) % 4)))
                            )
                            | ((cryst_struct == "BCC") & (not (H + K + L) % 2))
                        ):
                            # print "hkl =", H,K,L
                            qlab = (
                                float(H) * mat[0:3]
                                + float(K) * mat[3:6]
                                + float(L) * mat[6:]
                            )
                            if norme(qlab) > 1.0e-5:
                                uqlab = qlab / norme(qlab)
                                cosangle2 = inner(uqlab, uqlab_cen)
                                sintheta = -inner(uqlab, uilab)
                                if (sintheta > 0.0) & (cosangle2 > cosangle):
                                    # print "reachable reflection"
                                    Etheor[nspot] = (
                                        DictLT.E_eV_fois_lambda_nm
                                        * norme(qlab)
                                        / (2 * sintheta)
                                    )
                                    ththeor[nspot] = (180.0 / math.pi) * arcsin(
                                        sintheta
                                    )
                                    # print "Etheor = ", Etheor[nspot]
                                    if (Etheor[nspot] > (Emin * 1000.0)) & (
                                        Etheor[nspot] < (Emax * 1000.0)
                                    ):
                                        uflabtheor = uilab + 2 * sintheta * uqlab
                                        chi[nspot], tth[nspot] = uflab_to_2thetachi(
                                            uflabtheor
                                        )
                                        if (diagr == "side") & (chi[nspot] > 0.0):
                                            chi[nspot] = chi[nspot] - 180.0
                                        test = inner(uflabtheor, uflab_cen)
                                        # print "hkl =", H,K,L
                                        # print "uflabtheor.uflab_cen = ",test
                                        if test > cosangle:
                                            hkl[nspot, :] = array([H, K, L])
                                            uflab[nspot, :] = uflabtheor
                                            # top diagram use xbet xgam close to zero
                                            xy[nspot, :] = uflab_to_xycam_gen(
                                                uflab[nspot, :],
                                                calib,
                                                uflab_cen,
                                                pixelsize=pixelsize,
                                            )
                                            nspot = nspot + 1

    if remove_harmonics == "yes":
        hkl2, uflab2, xy2, nspots2, isbadpeak2 = remove_harmonic(
            hkl[0:nspot, :], uflab[0:nspot, :], xy[0:nspot, :]
        )

        index_goodpeak = where(isbadpeak2 == 0)
        Etheor2 = Etheor[index_goodpeak]
        ththeor2 = ththeor[index_goodpeak]
        chi2 = chi[index_goodpeak]
        tth2 = tth[index_goodpeak]
    else:
        range1 = list(range(0, nspot))
        hkl2, uflab2, xy2, nspots2, Etheor2, ththeor2, chi2, tth2 = (
            hkl[range1],
            uflab[range1],
            xy[range1],
            nspot,
            Etheor[range1],
            ththeor[range1],
            chi[range1],
            tth[range1],
        )

    if showall:
        print("list of theoretical peaks")
        if remove_harmonics == "yes":
            print("after removing harmonics")
        else:
            print("keeping all harmonics")
        print("hkl 0:3, uflab 3:6, xy 6:8, th 8, Etheor 9")
        for i in range(nspots2):
            print(
                hkl2[i, :],
                uflab2[i, :].round(decimals=3),
                round(xy2[i, 0], 2),
                round(xy2[i, 1], 2),
                round(ththeor2[i], 4),
                round(Etheor2[i], 1),
            )
        print("nb of peaks :", nspots2)
        print("keep spots with over/under range pixel positions")

    print("hkl 0:3, uflab 3:6, xy 6:8, th 8, Etheor 9, chi 10, tth 11 ")

    spotlist2 = column_stack((hkl2, uflab2, xy2, ththeor2, Etheor2, chi2, tth2))

    if diagr == "side":
        print("conversion to ydet downstream and zdet upwards :")
        print("ydet = ycam, zdet = xcam")

    print(shape(spotlist2))

    return spotlist2


def spotlist_360(
    Emin, Emax, matwithlatpar, cryst_struct, showall, uilab=array([0.0, 1.0, 0.0])
):

    # modif de spotlist de calcdef_include.py
    # tous les spots avec q a moins de x degres d'un certain plan ici plan yzlab.
    # pas de calcul de xycam car pas de limitation sur position detecteur
    # on garde les harmoniques

    # modif 04 Mar 2010
    # 20Oct10 : nouveau coeff E_eV_fois_lambda_nm plus precis
    # 21Oct10 : input matrix avec parametres de maille integres
    # 23May11 : add returnmore for 2theta chi

    # structures traitees ici : FCC, BCC, diamant (pour les extinctions)
    # maille deformee OK
    # Emin , Emax en KeV

    nmaxspots = 800
    limangle = 90.0
    sinangle = sin(limangle * math.pi / 180.0)

    mat = matwithlatpar

    uqlab_perp = array([1.0, 0.0, 0.0])

    if showall:
        print("calculate theoretical Laue pattern from orientation matrix")
        print("use matrix (with strain) \n", mat)
        print("energy range :", Emin, Emax)
        print(
            "max angle between uq and plane perpendicular to uqlab_perp (deg) : ",
            limangle,
        )
        print("uqlab_perp =", uqlab_perp)
        print("structure :", cryst_struct)

    # print "cosangle = ", cosangle
    yz = zeros((nmaxspots, 2), float)
    hkl = zeros((nmaxspots, 3), int)
    uflab = zeros((nmaxspots, 3), float)
    Etheor = zeros(nmaxspots, float)
    ththeor = zeros(nmaxspots, float)
    #    fpol = zeros(nmaxspots, float)

    tth = zeros(nmaxspots, float)
    chi = zeros(nmaxspots, float)

    dlatapprox = 1.0 / norme(mat[0:3])
    print("dlatapprox in nanometers = ", dlatapprox)

    Hmax = int(dlatapprox * 2 * Emax / 1.2398)

    if showall:
        print("Hmax = ", Hmax)

    # xlab = array([1.,0.,0.]) # utile pour fpol

    nspot = 0

    # Rlab z vers le haut, x vers le back, y vers l'aval

    for H in range(-Hmax, Hmax):
        for K in range(-Hmax, Hmax):
            if (not (K - H) % 2) | (cryst_struct == "BCC"):
                for L in range(-Hmax, Hmax):
                    if (not (L - H) % 2) | (cryst_struct == "BCC"):
                        if (
                            (cryst_struct == "FCC")
                            | (
                                (cryst_struct == "diamond")
                                & ((H % 2) | ((not H % 2) & (not (H + K + L) % 4)))
                            )
                            | ((cryst_struct == "BCC") & (not (H + K + L) % 2))
                        ):
                            # print "hkl =", H,K,L
                            qlab = (
                                float(H) * mat[0:3]
                                + float(K) * mat[3:6]
                                + float(L) * mat[6:]
                            )
                            if (
                                norme(qlab) > 1.0e-5
                            ):  # j'avais oublie de virer HKL = 000
                                uqlab = qlab / norme(qlab)
                                sinangle2 = abs(inner(uqlab, uqlab_perp))
                                sintheta = -inner(uqlab, uilab)
                                if (sintheta > 0.0) & (sinangle2 < sinangle):
                                    # print "reachable reflection"
                                    Etheor[nspot] = (
                                        DictLT.E_eV_fois_lambda_nm
                                        * norme(qlab)
                                        / (2 * sintheta)
                                    )
                                    ththeor[nspot] = (180.0 / math.pi) * arcsin(
                                        sintheta
                                    )
                                    # print "Etheor = ", Etheor[nspot]
                                    if (Etheor[nspot] > (Emin * 1000.0)) & (
                                        Etheor[nspot] < (Emax * 1000.0)
                                    ):
                                        uflabtheor = uilab + 2 * sintheta * uqlab
                                        chi[nspot], tth[nspot] = uflab_to_2thetachi(
                                            uflabtheor
                                        )
                                        hkl[nspot, :] = array([H, K, L])
                                        uflab[nspot, :] = uflabtheor
                                        # calcul fpol
                                        #                                        un = cross(uilab,uqlab)
                                        #                                        un = un / norme(un)
                                        #                                        fsig = inner(un,xlab)
                                        #                                        fpi = sqrt(1-fsig*fsig)
                                        #                                        cos2theta = cos(tth[nspot]*math.pi/180.0)
                                        #                                        fpol[nspot] = sqrt(fsig*fsig + fpi*fpi*cos2theta*cos2theta)
                                        nspot = nspot + 1
    #                            else :
    #                                print "warning : norme(qlab) < 1e-5 in spotlist_360"
    #                                print "H K L = ", H, K ,L

    spotlist2 = column_stack((Etheor, hkl, uflab, tth, chi))  # , fpol))

    print("E 0, hkl 1:4, uflab 4:7, tth 7, chi 8")  # , fpol 9"

    spotlist2 = spotlist2[:nspot, :]

    spotlist2_sorted = sort_list_decreasing_column(spotlist2, 0)

    if showall:
        print("list of theoretical peaks keeping harmonics")
        print("Etheor, hkl, uflab, 2theta, chi ")  # , fpol "
        for i in range(nspot):
            print(
                round(spotlist2_sorted[i, 0], 1),
                spotlist2_sorted[i, 1:4],
                spotlist2_sorted[i, 4:7].round(decimals=3),
                round(spotlist2_sorted[i, 7], 4),
                round(spotlist2_sorted[i, 8], 2),
            )  # ,round(spotlist2_sorted[i,9],2)
        print("nb of peaks :", nspot)

    print(shape(spotlist2_sorted))

    toto = list(range(nspot - 1, -1, -1))
    # print toto
    spotlist2_sorted = spotlist2_sorted[toto, :]

    # savetxt("toto.txt",spotlist2_sorted, fmt = "%.6f")

    return spotlist2_sorted


def twofitfiles_to_rotation(
    filefit1=None,
    matref1=None,
    filefit2=None,
    use_opsym="yes",
    axis_in_first_stereo_triangle=None,
):

    if filefit1 != None:
        print(filefit1)
        matstarlab1, data_fit, calib, pixdev = F2TC.readlt_fit(filefit1, readmore=True)
    else:
        matstarlab1 = matref1

    matstarlab2, data_fit, calib, pixdev = F2TC.readlt_fit(filefit2, readmore=True)

    if axis_in_first_stereo_triangle != None:
        matstarlab1, transfmat, rgb_axis = calc_cosines_first_stereo_triangle(
            matstarlab1, axis_in_first_stereo_triangle
        )
        matstarlab2, transfmat, rgb_axis = calc_cosines_first_stereo_triangle(
            matstarlab2, axis_in_first_stereo_triangle
        )

    matstarlabOND1 = matstarlab_to_matstarlabOND(matstarlab1)
    matstarlabOND2 = matstarlab_to_matstarlabOND(matstarlab2)
    mat1 = GT.matline_to_mat3x3(matstarlabOND1)
    mat2_start = GT.matline_to_mat3x3(matstarlabOND2)

    print(filefit2)

    if use_opsym == "yes":
        nop = 24
        allop = DictLT.OpSymArray
        indgoodop = array(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                25,
                27,
                29,
                31,
                33,
                35,
                37,
                39,
                41,
                43,
                45,
                47,
            ]
        )
        goodop = allop[indgoodop]
        vec_crystal = zeros((nop, 3), float)
        vec_lab = zeros((nop, 3), float)
        angle1 = zeros(nop, float)

        # print mat2_start.round(decimals=3)
        for k in range(nop):
            mat2 = dot(mat2_start, goodop[k])
            # print mat2.round(decimals=3)
            vec_crystal[k, :], vec_lab[k, :], angle1[k] = twomat_to_rotation(mat1, mat2)

        ind1 = argmin(angle1)
        print("opsym for minimal rotation : ", ind1)
        print("opsym matrix \n", goodop[k])
        print("minimal angle : ", round(angle1[ind1], 3))

        return (vec_crystal[ind1], vec_lab[ind1], angle1[ind1], matstarlabOND1)

    elif use_opsym == "no":

        mat2 = mat2_start
        vec_crystal, vec_lab, angle1 = twomat_to_rotation(mat1, mat2)
        return (vec_crystal, vec_lab, angle1, matstarlabOND1)

    else:
        return 1
