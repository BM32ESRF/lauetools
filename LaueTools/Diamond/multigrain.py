# -*- coding: utf-8 -*-
print(("multigrain module located in",__file__))

import site

from time import time, asctime
import numpy as np
from numpy import *
import matplotlib.pylab as p
from scipy import * # pour charger les donnees
#import scipy.io.array_import # pour charger les donnees
import os

from numpy.linalg import inv, det

from math import atan
from math import acos
from math import asin

import sys
import os

sys.path.append("..")

if sys.version_info.major == 3:
    from .. import LaueGeometry as F2TC
    from .. import readmccd as rmccd
    from .. import LaueAutoAnalysis as LAA
    from .. import indexingAnglesLUT as INDEX
    from .. import findorient as FindO
    from .. import FitOrient as FitO
    from .. import CrystalParameters as CP
    from .. generaltools import norme_vec as norme
    from .. import generaltools as GT
    from .. import IOLaueTools as RWASCII
    from .. import dict_LaueTools as DictLT
    # set invisible parameters for serial_peak_search, serial_index_refine_multigrain
    from .. import param_multigrain as PAR
    from .. import lauecore as LAUE

else:

    import find2thetachi as F2TC
    import readmccd as rmccd
    import LaueAutoAnalysis as LAA
    import indexingAnglesLUT as INDEX
    import findorient as FindO
    import FitOrient as FitO
    import CrystalParameters as CP
    from generaltools import norme_vec as norme
    import generaltools as GT
    import IOLaueTools as RWASCII
    import dict_LaueTools as DictLT
    # set invisible parameters for serial_peak_search, serial_index_refine_multigrain
    import param_multigrain as PAR
    import lauecore as LAUE


# warning : matstarlab (matline) is in OR's lab frame, matLT (mat3x3) in LaueTools lab frame
# zLT = zOR, xLT = yOR, xOR = - yLT
# incident beam along +y in OR lab frame, along +x in LT lab frame


#******************************
### note about reference frames :
#
#The reference frame (O, xcam, ycam, zcam) is attached to the detector, as the detector remains fixed during the experiment. The origin O of this reference frame is taken in a wedge at the detector surface. 
#
#zcam is perpendicular to the detector screen, pointing from sample to detector.
#xcam points rightwards, and ycam points downwards, for an observer located at the sample position, looking at the detector while receiving the x-ray beam in the top of the head.
#In the BM32 setup, the incident beam (unit vector ui) is close to ycam.
#
#When loading into LaueTools a camera image not fitting with this convention, proper flipping of axes and/or image rotation is applied so that the new image respects the convention.
#
#Two correction angles beta and gamma define the exact orientation of ui in the detector frame
#
# see JSR Petit 2012 for details
#
#The (approximate) sample geometry is defined by the beta and gamma angles, and by the mean position of the diffracting volume described by its three coordinates xcam = xcen (in pixels), ycam = ycen (in pixels) and zcam = DD (in mm). This is only approximate as the ycam position of the diffracting volume (along the incident beam, i.e. its position in depth) varies from spot to spot, due to both absorption effects (linked to electronic density and photon energy) and to extinction effects (linked to crystal quality and HKL plane) [ref Lonsdale].
#
#
#K. Lonsdale (1947), "Extinction in X-ray crystallography", Mineralogical Magazine;
#March 1947 v. 28; no. 196; p. 14-25; DOI: 10.1180/minmag.1947.028.196.04 , http://www.minersoc.org/pages/Archive-MM/Volume_28/28-196-14.pdf
#
#
#see slide 20. in O. Robach's lecture at :
#http://www.small-scale-plasticity.cemes.fr/index.php/lectures-available
#or in
#"Laue MicroDiffraction : a local metrology tool", O. Robach et al. in "Rayons X et Matiere : RX 2013" eds. P. Goudeau, R. Guinebretiere, Lavoisier, in preparation.
#
#The "laboratory" frame Rlab, a somehow misleading term as it is NOT attached to the laboratory, is defined from zcam and ui, by :
#ylab = ui,
#xlab = cross(ui, zcam)
#zlab = cross(xlab, ylab)
#On BM32 Rlab and Rcam are almost coincident within 2 degrees.
#The "sample" frame Rsample, again a somehow misleading term as it is NOT attached to the sample, is obtained from Rlab by positive rotation of 40 degrees around the x axis.
#On BM32 the motorized linear displacement stages of the sample xech yech zech (sample = echantillon in French, hence the "ech") are almost coincident in direction (but not in sign) with the xsample, ysample and zsample axes, within 2 degrees.
#
#
#In the summary files created by multigrain.py, with columns
#
#img gnumloc npeaks pixdev intensity dxymicrons_0 dxymicrons_1 matstarlab_0 matstarlab_1 matstarlab_2 matstarlab_3 matstarlab_4 matstarlab_5 matstarlab_6 matstarlab_7 matstarlab_8 euler3_0 euler3_1 euler3_2 maxpixdev stdpixdev rgb_x_sample_0 rgb_x_sample_1 rgb_x_sample_2 rgb_y_sample_0 rgb_y_sample_1 rgb_y_sample_2 rgb_z_sample_0 rgb_z_sample_1 rgb_z_sample_2 rgb_x_lab_0 rgb_x_lab_1 rgb_x_lab_2 rgb_y_lab_0 rgb_y_lab_1 rgb_y_lab_2 rgb_z_lab_0 rgb_z_lab_1 rgb_z_lab_2 strain6_crystal_0 strain6_crystal_1 strain6_crystal_2 strain6_crystal_3 strain6_crystal_4 strain6_crystal_5 strain6_sample_0 strain6_sample_1 strain6_sample_2 strain6_sample_3 strain6_sample_4 strain6_sample_5 stress6_crystal_0 stress6_crystal_1 stress6_crystal_2 stress6_crystal_3 stress6_crystal_4 stress6_crystal_5 stress6_sample_0 stress6_sample_1 stress6_sample_2 stress6_sample_3 stress6_sample_4 stress6_sample_5 res_shear_stress_0 res_shear_stress_1 res_shear_stress_2 res_shear_stress_3 res_shear_stress_4 res_shear_stress_5 res_shear_stress_6 res_shear_stress_7 res_shear_stress_8 res_shear_stress_9 res_shear_stress_10 res_shear_stress_11 max_rss von_mises misorientation_angle w_mrad_0 w_mrad_1 w_mrad_2
#
#the reference frames are the following (odile32 convention) :
#
#for stress and strain :
#
#0 1 2 3 4 5 = xx yy zz yz xz xy
#
#
#(it would be better to use 1 2 3 4 5 6 instead, to match the "Voigt" shortcut convention
#used to express elastic constants as a 6x6 matrix and the strain and stress as 6-components vectors)
#
#Rlab, Rsample, Rcrystal are reference frames in which the tensor is calculated
#
#Rlab :
#ylab along incident beam (downstream),
#xlab along cross(ylab, zcam) where zcam is the normal to the detector screen (almost vertical upwards)
#zlab along cross(xlab, ylab)
#
#Rsample : obtained from Rlab by a rotation
#of MG.PAR.omega_sample_frame (set in param_multigrain.py) around xlab

# added 27Jan2016 : more flexibility in the definition of the sample frame
# any cartesian frame (orthonorme direct) may be used
# new associated variable :
#    MG.PAR.mat_from_lab_to_sample_frame
# this 3x3 matrix gives as lines
# the coordinates of the vectors of the sample frame
# on the vectors of the lab frame
#  y_sample = mat[1,0] * x_lab + mat[1,1] * y_lab + mat[1,2] * z_lab
# new option in functions previously using MG.PAR.omega_sample_frame :
#    mat_from_lab_to_sample_frame = None or MG.PAR.mat_from_lab_to_sample_frame

#
#Rcrystal : obtained by orthonormalizing the a,b,c frame of the measured crystal unit cell
#
#
#for RGB :
#
#0 1 2 = amounts of red, green and blue
#
#rgb_x_sample gives orientation of x_sample / -x_sample axis in the first stereo triangle of the crystal
#
#Warning :
#
#In the code, two conventions coexist for the lab frame :
#the definition of zlab is always the same
#but the definition of xlab and ylab change depending on the coding person :
#jsmicha uses xlab along the incident beam
#odile32 uses ylab along the incident beam
#
#matrices using the jsmicha convention typically have a name like "UBmat"
#
#In multigrain.py and diamond.py :
#
#Matrices with the jsmicha convention have a "LT" in their names.
#Matrices using the odile32 convention typically have a name like "matstarlab", "matdirlab", "matstarsample".
#
#matstarlab is a 9-components vectors giving the coordinates of astar, bstar and cstar in Rlab :
#
#astar_lab = m[0:3]
#bstar_lab = m[3:6]
#cstar_lab = m[6:9]
#
#Matrices in a 3x3 shape have a "3x3" in their names.
#
#FileSeries is a new "graphical interface" version of multigrain.py
#which also incorporates new features
#such as searching for several crystal structures in the same multi-grain Laue pattern.
#
#The summary files generated by FileSeries use the same Rlab and Rsample conventions as the summary files generated bymultigrain.py 
#

# Note : FileSeries uses a "modified" multigrain.py located in trunk/FileSeries/

#***************************************
### note : 3 ways of defining the sample reference frame    # 29Jan2016
# the corresponding block should be added at the beginning of the script
# after the "import" sequence

#    if 1 :  # "classical" sample frame

#        MG.PAR.omega_sample_frame = 40.
#        MG.omega_sample_frame_to_mat_from_lab_to_sample_frame()
#        MG.PAR.omega_sample_frame = None
#
#    if 1 :  # "classical" sample frame,  slower - only for retrocompatibility

#        MG.PAR.omega_sample_frame = 40
#        MG.PAR.mat_from_lab_to_sample_frame = None
#
#    if 1 :  # user-defined sample frame : here obtained from classical sample frame by rotation of -28.5 degrees around z_sample

#        MG.PAR.omega_sample_frame = None
#        ang1 = -28.5
#        vsample1 = np.array([0.,0.,1.])
#        matrot1 = MG.from_axis_vecangle_to_mat(vsample1, ang1)
#        print "sample_new vectors as columns on sample vectors"
#        print "matrot1 = \n", matrot1
#        ang2 = 40.
#        vlab2 = np.array([1.,0.,0.])
#        matrot2 = MG.from_axis_vecangle_to_mat(vlab2, ang2)
#        print "sample vectors as columns on lab vectors"
#        print "matrot2 = \n", matrot2
#        MG.PAR.mat_from_lab_to_sample_frame = np.dot(matrot1.transpose(), matrot2.transpose())
#        print "sample_new vectors as lines on lab vectors"
#        print "MG.PAR.mat_from_lab_to_sample_frame = \n", MG.PAR.mat_from_lab_to_sample_frame
#

# ******************************************************************************
### note : typical import sequence at the beginning of a script using multigrain.py

#import matplotlib.pylab as p
#import matplotlib.pyplot as plt
#
#def pc():
#    p.close('all')
#
#import multigrain as MG
#
#import diamond as DIA
#import readmccd as rmccd
#import find2thetachi as F2TC
#import generaltools as GT
#import dict_LaueTools as DictLT
#import readwriteASCII as RWASCII
#import findorient as FindO
#from math import cos, sin
#import shutil as SH
#
#import os
#
#import time
#
#import numpy as np
#
#from FileSeries.multigrain import readfitfile_multigrains
#
#if sys.platform.startswith('lin'):
#    # for unix
#    MG.PAR.cr_string = "\r\n"
#else:
#    # for windows
#    MG.PAR.cr_string = "\n"

#******************************************************************************
### note : below is the typical script block for changing the parameters initialized in param_multigrain.py
# the corresponding block should be added at the beginning of the script
# after the "import" sequence

#MGPARAM = 1
#
#if MGPARAM :
#
#    MG.PAR.elem_label_index_refine = "quartz_alpha"
#    # voir dict_Materials dans dict_LaueTools.py pour la liste des structures cristallines
#    # pas besoin de parametres de maille tres precis sauf si mesures de Espot
#    MG.PAR.ngrains_index_refine = 1  # try to index up to "ngrains_index_refine" grains
#    MG.PAR.overwrite_index_refine = 1 # overwrite existing fit files
#    MG.PAR.add_str_index_refine = "_t_UWN" # carto_rough fitfiles   string to add : UWN = "use weights no"
#  etc ...
#**********************************************************************************
### note : known bug : fewer indexed spots from MG compared to LaueTools GUI
# solution : increase detectordiameter in LAA.simulate_theo
# ***********************************************************

PI = math.pi

p.rcParams['lines.markersize'] = 12
p.rcParams['lines.linewidth'] = 1.5
p.rcParams['font.size'] = 12
p.rcParams['axes.labelsize'] = 'large'
p.rcParams['figure.subplot.bottom'] = 0.2
p.rcParams['figure.subplot.left'] = 0.2
p.rcParams['xtick.major.size'] = 8
p.rcParams['xtick.major.pad'] = 8
p.rcParams['ytick.major.size'] = 8
p.rcParams['ytick.major.pad'] = 8
p.rcParams['figure.facecolor'] = 'w'

#########


def pc():
    p.close('all')

def print_calib(calib, verbose = False, pixelsize = None) :

    # modif 04 Mar 2010 xbet et xgam en degres au lieu de radians

    calib3 = zeros(5, float)
    calib3 = calib * 1.0

    if verbose:
        print("calib :")
        print("dd(mm) xcen(pixel) ycen(pixel) xbet(deg) xgam(deg)")
        if pixelsize is not None:
            print("pixelsize (mm)")

    if pixelsize  is  not None: print(calib3[0].round(decimals=3), calib3[1].round(decimals=2), calib3[2].round(decimals=2), \
        calib3[3].round(decimals=3), calib3[4].round(decimals=3), pixelsize)

    else: print(calib3[0].round(decimals=3), calib3[1].round(decimals=2), calib3[2].round(decimals=2), \
        calib3[3].round(decimals=3), calib3[4].round(decimals=3))

    if verbose:
        print(calib3[0].round(decimals=3), ",", calib3[1].round(decimals=2), "," ,calib3[2].round(decimals=2), \
            "," , calib3[3].round(decimals=3), ",", calib3[4].round(decimals=3))
    return 0

# calcul reseau reciproque

def dlat_to_rlat(dlat) :

    rlat = rand(6)
    """
    # Compute reciprocal lattice parameters. The convention used is that
    # a[i]*b[j] = d[ij], i.e. no 2PI's in reciprocal lattice.
    """
    # compute volume of real lattice cell
    volume = dlat[0]*dlat[1]*dlat[2]*np.sqrt(1+2*np.cos(dlat[3])*np.cos(dlat[4])*np.cos(dlat[5])
             -np.cos(dlat[3])*np.cos(dlat[3])
             -np.cos(dlat[4])*np.cos(dlat[4])
             -np.cos(dlat[5])*np.cos(dlat[5]))

    # compute reciprocal lattice parameters
    rlat[0] = dlat[1]*dlat[2]*np.sin(dlat[3])/volume
    rlat[1] = dlat[0]*dlat[2]*np.sin(dlat[4])/volume
    rlat[2] = dlat[0]*dlat[1]*np.sin(dlat[5])/volume
    rlat[3] = np.arccos((np.cos(dlat[4])*np.cos(dlat[5])-np.cos(dlat[3]))
                   /(np.sin(dlat[4])*np.sin(dlat[5])))
    rlat[4] = np.arccos((np.cos(dlat[3])*np.cos(dlat[5])-np.cos(dlat[4]))
                   /(np.sin(dlat[3])*np.sin(dlat[5])))
    rlat[5] = np.arccos((np.cos(dlat[3])*np.cos(dlat[4])-np.cos(dlat[5]))
                   /(np.sin(dlat[3])*np.sin(dlat[4])))

    return rlat

def mat_to_rlat(matstarlab):

    rlat = zeros(6, float)

    astarlab = matstarlab[0:3]
    bstarlab = matstarlab[3:6]
    cstarlab = matstarlab[6:9]
    rlat[0] = norme(astarlab)
    rlat[1] = norme(bstarlab)
    rlat[2] = norme(cstarlab)
    rlat[5] = np.arccos(inner(astarlab, bstarlab) / (rlat[0] * rlat[1]))
    rlat[4] = np.arccos(inner(cstarlab, astarlab) / (rlat[2] * rlat[0]))
    rlat[3] = np.arccos(inner(bstarlab, cstarlab) / (rlat[1] * rlat[2]))

    return rlat

def rad_to_deg(dlat):

    dlatdeg = hstack((dlat[0:3], dlat[3:6] * 180.0 / PI))
    return dlatdeg

def deg_to_rad(dlat):

    dlatrad = hstack((dlat[0: 3], dlat[3: 6] * PI / 180.0))
    return dlatrad

def deg_to_rad_angstroms_to_nm(dlat_angstroms_deg):

    print(dlat_angstroms_deg)
    dlat_angstroms_deg = np.array(dlat_angstroms_deg, float)
    dlat_nm_rad = hstack((dlat_angstroms_deg[0: 3] / 10., dlat_angstroms_deg[3: 6] * PI / 180.))
    return dlat_nm_rad

def dlat_to_dlatr(dlat):

    dlatr = zeros(6)
    for i in range(0, 3):
        dlatr[i] = dlat[i] / dlat[0]
    for i in range(3, 6):
        dlatr[i] = dlat[i]

    return dlatr

def epsline_to_epsmat(epsline): #29May13
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

def epsmat_to_epsline(epsmat): #29May13
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

def dlat_to_Bstar(dlat): #29May13
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
    Bstar[0, 1] = rlat[1] * np.cos(rlat[5])
    Bstar[1, 1] = rlat[1] * np.sin(rlat[5])
    Bstar[0, 2] = rlat[2] * np.cos(rlat[4])
    Bstar[1, 2] = -rlat[2] * np.sin(rlat[4])*np.cos(dlat[3])
    Bstar[2, 2] = 1.0 / dlat[2]

    return Bstar

def rlat_to_Bstar(rlat): #29May13

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
    Bstar[0, 1] = rlat[1]*np.cos(rlat[5])
    Bstar[1, 1] = rlat[1]*np.sin(rlat[5])
    Bstar[0, 2] = rlat[2]*np.cos(rlat[4])
    Bstar[1, 2] = -rlat[2]*np.sin(rlat[4])*np.cos(dlat[3])
    Bstar[2, 2] = 1.0/dlat[2]

    return Bstar

def matstarlab_to_deviatoric_strain_crystal(matstarlab, 
                                            version=2,
                                            elem_label="Ge"):
    #29May13
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
#    print "dlat = ", dlat  # - np.array([1.,1.,1.,PI/2.,PI/2.,PI/2.])
    dlatrdeg = rad_to_deg(dlat_to_dlatr(dlat))

    if version == 1: # only for initially cubic unit cell

        epsp = zeros(6,float)

        tr3 = (dlat[0]+dlat[1]+dlat[2])/3.

        epsp[0] = (dlat[0] - tr3)*1000./dlat[0]
        epsp[1] = (dlat[1] - tr3)*1000./dlat[0]
        epsp[2] = (dlat[2] - tr3)*1000./dlat[0]

        epsp[3] = -1000.*(dlat[3]-PI/2.)/2.
        epsp[4] = -1000.*(dlat[4]-PI/2.)/2.
        epsp[5] = -1000.*(dlat[5]-PI/2.)/2.

    elif version == 2: # for any symmetry of unit cell

        # reference lattice parameters with angles in degrees
        dlat0_deg = np.array(DictLT.dict_Materials[elem_label][1], dtype=float)
        dlat0 = deg_to_rad(dlat0_deg)

#        print "dlat0, angles in rad", dlat0.round(decimals = 4)
#        print "dlat, angles in rad", dlat.round(decimals = 4)

        # matstarlab construite pour avoir norme(astar) = 1
        Bdir0 = rlat_to_Bstar(dlat0)
        Bdir0 = Bdir0 / dlat0[0]

        Bdir = rlat_to_Bstar(dlat)
        Bdir = Bdir/dlat[0]
        #print Bdir0.round(decimals=4)
        #print Bdir.round(decimals=4)

        # Rmat = inv(Bdir) et T = dot(inv(Rmat), Rmat0)

        Tmat = dot(Bdir, inv(Bdir0))

        eps1 = 0.5 * (Tmat + Tmat.transpose()) - eye(3)
        #print eps1.round(decimals=2)
        #print np.trace(eps1)

        # la normalisation du premier vecteur de Bdir a 1
        # ne donne pas le meme volume pour les deux mailles
        # => il faut soustraire la partie dilatation

        epsp1 = 1000.0 * (eps1 - (np.trace(eps1) / 3.0)*eye(3))
        #print epsp1.round(decimals=1)
        #print np.trace(epsp1)

        epsp = epsmat_to_epsline(epsp1)

    print("deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12  *1e3 \n", epsp.round(decimals=1))

    print("dlatrdeg = \n", dlatrdeg)

    return epsp, dlatrdeg

def read_stiffness_file(filestf): #29May13
    """
    # units = 1e11 N/m2
    # dans les fichiers stf de XMAS les cij sont en 1e11 N/m2
    """
    c_tensor = loadtxt(filestf, skiprows=1)
    c_tensor = np.array(c_tensor, dtype=float)
    print(filestf)
    print(shape(c_tensor))
    print("stiffness tensor C, 1e11 N/m2 (100 GPa) units")
    print(c_tensor)

    return c_tensor

def deviatoric_strain_crystal_to_stress_crystal(c_tensor, eps_crystal_line): #29May13
    """
    Voigt Notation
    C = 6x6 matrix
    (C n'est pas un tenseur d'ordre 2 => regle de changement de repere ne s'applique pas a C)
    (il faut la notation Pedersen cf mail Consonni pour fabriquer un tenseur d'ordre 2 a partir de C)
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
    fact1 = np.array([1., 1., 1., 2., 2., 2.])
    gam_cryst = multiply(eps_crystal_line, fact1)
    sigma_crystal_line = dot(c_tensor, gam_cryst)
    #print eps_crystal_line
    #print gam_cryst
    #print sigma_crystal_line
    return sigma_crystal_line

def glide_systems_to_schmid_tensors(n_ref=array([1., 1., 1.]), 
                                    b_ref=array([1., -1., 0.]),
                                    verbose=0,
                                    returnmore=0):
    #29May13
    """
    only for cubic systems
    coordonnees cartesiennes dans le repere OND obtenu en orthonormalisant le repere cristal
    cf these Gael Daveau p 16
    """
    nop = 24

    allop = DictLT.OpSymArray
    indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])
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
        #print "k = ", k
        un_ref = uqall[0, k, :]
        ub_ref = uqall[1, k, :]
        for j in range(k + 1, nop):
            #print "j = ", j
            dun = norme(cross(un_ref, uqall[0, j, :]))
            dub = norme(cross(ub_ref, uqall[1, j, :]))
            dun_dub = dun + dub
            if dun_dub < 0.01: isdouble[j] = 1
    print(isdouble)

    ind0 = where(isdouble == 0)
    print(ind0[0])
    uqall = uqall[:, ind0[0], :]

    nop2 = 12

    st1 = zeros((nop2, 3, 3),float)
    hkl_n = uqall[0, :, :]*normehkl[0]
    hkl_b = uqall[1, :, :]*normehkl[1]
    if verbose: print("n b schmid_tensor [line1, line2, line3]")
    for k in range(nop2):
        un_colonne = uqall[0, k, :].reshape(3, 1)
        ub_ligne = uqall[1, k, :].reshape(1, 3)
        st1[k, :, :] = dot(un_colonne,ub_ligne)        
        if verbose: print(uqall[0, k, :]*normehkl[0], uqall[1,k,:]*normehkl[1], st1[k,:,:].reshape(1,9).round(decimals=3))

    if returnmore == 0:
        return st1
    else:
        return st1, hkl_n, hkl_b

def deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(sigma_crystal_line, schmid_tensors):
    #29May13
    nop2 = shape(schmid_tensors)[0]
    sigma_crystal_3x3 = epsline_to_epsmat(sigma_crystal_line)
    tau_all = zeros(nop2, float)
    for k in range(nop2):
        tau_all[k] = (np.multiply(schmid_tensors[k], sigma_crystal_3x3)).sum()

    #print tau_all
    return tau_all

def deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal_line):
    #29May13
    # cf formula (4.17) in book chapter by N. Tamura p 143
    # book "strain and dislocation gradients from diffraction"
    # eds R.I. Barabash and G.E. Ice
    sig = sigma_crystal_line*1.0
    von_mises = (sig[0]-sig[1])*(sig[0]-sig[1]) + \
                (sig[1]-sig[2])*(sig[1]-sig[2]) +\
                (sig[2]-sig[0])*(sig[2]-sig[0]) + \
                6.* (sig[3]*sig[3] + sig[4]*sig[4] + sig[5]*sig[5])
    von_mises = von_mises / 2.
    von_mises = np.sqrt(von_mises)
    return von_mises

def deviatoric_strain_crystal_to_equivalent_strain(epsilon_crystal_line):
    #23May16
    # formula (1) from Chen et al. Geology 2014
    # cf formula (4.14) in book chapter by N. Tamura p 142
    # book "strain and dislocation gradients from diffraction"
    # eds R.I. Barabash and G.E. Ice

    eps = epsilon_crystal_line*1.
    toto = (eps[0]-eps[1])*(eps[0]-eps[1]) + \
                (eps[1]-eps[2])*(eps[1]-eps[2]) +\
                (eps[2]-eps[0])*(eps[2]-eps[0]) + \
                6.* (eps[3]*eps[3] + eps[4]*eps[4] + eps[5]*eps[5])
    toto = toto / 2.
    eq_strain = (2./3.) * np.sqrt(toto)
    return eq_strain

def uflab_to_xycam(uflab, 
                   calib,
                   pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]):

    # modif 04 Mar 2010 xbet xgam en degres au lieu de radians

    # XMAS PCIF6 changer le signe de xgam
    # laptop OR garder le meme signe pour xgam

    detect = calib[0] * 1.0
    xcen = calib[1] * 1.0
    ycen = calib[2] * 1.0
    xbet = calib[3] * 1.0
    xgam = calib[4] * 1.0

##    print "Correcting the data according to the parameters"
##    print "xcam, ycam in XMAS convention"
##
##    print "detect in mm" , detect
##    print "xcen in pixels" , xcen
##    print "ycen in pixels" , ycen
##    print "xbet in degrees" , xbet
##    print "xgam in degrees" , xgam

    xbetrad = xbet * PI / 180.0
    xgamrad = xgam * PI / 180.0

    cosbeta = np.cos(PI / 2. - xbetrad)
    sinbeta = np.sin(PI / 2. - xbetrad)
    cosgam = np.cos(-xgamrad)
    singam = np.sin(-xgamrad)

    IOlab = detect * np.array([0.0, cosbeta, sinbeta])

    unlab = IOlab / norme(IOlab)

    normeIMlab = detect / inner(uflab, unlab)

    #uflab1 = array([-uflab[0],uflab[1],uflab[2]])

    uflab1 = uflab * 1.0

    IMlab = normeIMlab * uflab1

    OMlab = IMlab - IOlab

    xca0 = OMlab[0]
    yca0 = OMlab[1] / sinbeta

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / pixelsize

    uflabyz = array([0.0, uflab1[1], uflab1[2]])
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab

    chi = (180.0 / PI) * np.arctan(uflab1[0] / norme(uflabyz))
    twicetheta = (180.0 / PI) * np.arccos(uflab1[1])
    th0 = twicetheta / 2.0

    #print "2theta, theta, chi en deg", twicetheta , chi, twicetheta/2.0
    #print "xcam, ycam = ", xcam, ycam

    return(xcam, ycam, th0)

def uqlab_to_xycam(uqlab,
                   calib,
                   pixelsize=DictLT.dict_CCD[PAR.CCDlabel][1]):

    uflab = zeros(3, float)
    xycam = zeros(2, float)

    uilab = array([0.0, 1.0, 0.0])
    sintheta = -inner(uqlab, uilab)
    uflab = uilab + 2 * sintheta * uqlab

    xycam[0], xycam[1], th0 = uflab_to_xycam(uflab,
                                            calib,
                                            pixelsize=pixelsize)

    return xycam

def matstarlab_to_matstarlabOND(matstarlab):

    astar1 = matstarlab[: 3]
    bstar1 = matstarlab[3: 6]
    cstar1 = matstarlab[6:]

    astar0 = astar1 / norme(astar1)
    cstar0 = cross(astar0, bstar1)
    cstar0 = cstar0 / norme(cstar0)
    bstar0 = cross(cstar0, astar0)

    matstarlabOND = hstack((astar0, bstar0, cstar0)).transpose()

    #print matstarlabOND

    return matstarlabOND

def matstarlab_to_matdirlab3x3(matstarlab): #29May13

    rlat = mat_to_rlat(matstarlab)
    #print rlat
    vol = CP.vol_cell(rlat, angles_in_deg=0)

    astar1 = matstarlab[: 3]
    bstar1 = matstarlab[3: 6]
    cstar1 = matstarlab[6:]

    adir = cross(bstar1, cstar1) / vol
    bdir = cross(cstar1, astar1) / vol
    cdir = cross(astar1, bstar1) / vol

    matdirlab3x3 = column_stack((adir, bdir, cdir))

    #print " matdirlab3x3 =\n", matdirlab3x3.round(decimals=6)

    return(matdirlab3x3, rlat)

def omega_sample_frame_to_mat_from_lab_to_sample_frame() :

    omega0 = PAR.omega_sample_frame

    omega = omega0 * PI / 180.0

    # rotation de -omega autour de l'axe x pour repasser dans Rsample
    PAR.mat_from_lab_to_sample_frame = array([[1.0, 0.0, 0.0],
                                        [0.0, np.cos(omega), np.sin(omega)]
                                        [0.0, -np.sin(omega), np.cos(omega)]])

    print("MG.PAR.mat_from_lab_to_sample_frame = \n", PAR.mat_from_lab_to_sample_frame)

    return 1

def matstarlab_to_matdirONDsample3x3(matstarlab, 
                                     omega0=None, # was PAR.omega_sample_frame
                                     mat_from_lab_to_sample_frame=PAR.mat_from_lab_to_sample_frame
                                     ): #29May13

    # uc unit cell
    # dir direct
    # uc_dir_OND : cartesian frame obtained by orthonormalizing direct unit cell

    matdirlab3x3, rlat = matstarlab_to_matdirlab3x3(matstarlab)
    # dir_bmatrix = uc_dir on uc_dir_OND

    dir_bmatrix = dlat_to_Bstar(rlat)

    # matdirONDlab3x3 = uc_dir_OND on lab

    matdirONDlab3x3 = dot(matdirlab3x3, np.linalg.inv(dir_bmatrix))

    if (omega0 is  not None)&(mat_from_lab_to_sample_frame is None): # deprecated - only for retrocompatibility
        omega = omega0 * PI / 180.0
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        mat_from_lab_to_sample_frame = array([[1.0, 0.0, 0.0],
                                        [0.0, np.cos(omega), np.sin(omega)]
                                        [0.0, -np.sin(omega), np.cos(omega)]])

    # matdirONDsample3x3 = uc_dir_OND on sample
    # rsample = matdirONDsample3x3 * ruc_dir_OND

    matdirONDsample3x3 = dot(mat_from_lab_to_sample_frame, matdirONDlab3x3)

    return matdirONDsample3x3

def transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab,
                                          tensor_crystal_line,
                                          omega0=None, # was PAR.omega_sample_frame,
                                          mat_from_lab_to_sample_frame=PAR.mat_from_lab_to_sample_frame
                                    ):
    #29May13
    """
    start from stress or strain tensor
    as 6 coord vector
    """
    tensor_crystal_3x3 = epsline_to_epsmat(tensor_crystal_line)

    matdirONDsample3x3 =  matstarlab_to_matdirONDsample3x3(matstarlab, 
                                                           omega0=omega0,
                                                           mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame)

    # changement de base pour tenseur d'ordre 2

    toto = dot(tensor_crystal_3x3, matdirONDsample3x3.transpose())

    tensor_sample_3x3 = dot(matdirONDsample3x3, toto)

    tensor_sample_line = epsmat_to_epsline(tensor_sample_3x3)

    return tensor_sample_line

def transform_2nd_order_tensor_from_sample_frame_to_crystal_frame(matstarlab,
                                          tensor_sample_3x3,
                                          omega0=None, # was PAR.omega_sample_frame,
                                          mat_from_lab_to_sample_frame=PAR.mat_from_lab_to_sample_frame
                                          ):
    #29May13
    """
    start from stress or strain tensor
    as 3x3 matrix
    """
    matdirONDsample3x3 = matstarlab_to_matdirONDsample3x3(matstarlab,
                                                           omega0=omega0,
                                                           mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame)

    # changement de base pour tenseur d'ordre 2
#    toto = dot(tensor_crystal_3x3 , matdirONDsample3x3.transpose())
#    tensor_sample_3x3 = dot(matdirONDsample3x3,toto)

    toto = dot(tensor_sample_3x3, matdirONDsample3x3)

    tensor_crystal_3x3 = dot(matdirONDsample3x3.transpose(),toto)    

    return tensor_crystal_3x3

def matstarlab_to_deviatoric_strain_sample(matstarlab, 
                                           omega0=None, # was PAR.omega_sample_frame,
                                           mat_from_lab_to_sample_frame=PAR.mat_from_lab_to_sample_frame,
                                           version=2,
                                           returnmore=False,
                                           elem_label="Ge"):
    #29May13
    epsp_crystal, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(matstarlab, 
                                            version=version, 
                                            elem_label=elem_label)

    epsp_sample =  transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab,
                                                                  epsp_crystal,
                                                                  omega0=omega0,
                                                                  mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame)
    if returnmore == False: return epsp_sample

    else: return epsp_sample, epsp_crystal   # add epsp_crystal


def read_all_grains_str(filestr, 
                        min_matLT=False): #29May13

    print("reading info from STR file : \n", filestr)
    #print "peak list, calibration, strained orientation matrix, deviations"
    #print "change sign of HKL's"
    f = open(filestr, 'r')
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
            if line[0] == 'N':
                #print line
                #print line.split()
                gnumtot = np.array((line.split())[-1], dtype=int)
                print("Number of grains : ", gnumtot)
                linestartspot = zeros(gnumtot, int)
                lineendspot = zeros(gnumtot, int)
                linemat = zeros(gnumtot, int)
                linestrain = zeros(gnumtot, int)
                linestrain2 = zeros(gnumtot, int)
                dlatstr = zeros((gnumtot, 6), float)
                dev_str = zeros((gnumtot, 3), float)

            if line[0] == 'G':
                gnumloc = np.array((line.split())[2], dtype=int)
                #print gnumloc
                gnum = grainnumstr - 1
                if gnumloc == grainnumstr:
                    print("grain ", grainnumstr)
                    #print  " indexed peaks list starts at line : ", i
                    linestartspot[gnum] = i + 1
                    grainfound = 1

                    list1 = []
            if grainfound == 1:
                #if i == linestart :
                    #print line.rstrip("\n")
                if line[0:3] == 'lat':
                    #print "lattice parameters at line : ", i
                    dlatstr[gnum, :] = np.array(line[18:].split(), dtype=float)
                    print("lattice parameters : \n", dlatstr[gnum, :])
                    #print "indexed peaks list ends at line = ", i
                    if norme(dlatstr[gnum, :]) > 0.1: is_good_grain.append(1)
                    else: is_good_grain.append(0)

                    lineendspot[gnum] = i - 1
                if gnum == 0:
                    if line[0:2] == 'dd':
                        #print "calib starts at line : ", i
                        calib[:3] = np.array(line[17:].split(), dtype=float)              
                    if line[0:4] == 'xbet':
                        #print "calib line 2 at line = ", i
                        calib[3:] = np.array(line[11:].split(), dtype=float)
                if line[0:4] == 'dev1':
                    dev_str[gnum, :] = np.array(line.split()[3:], dtype=float)
                    print("deviations : \n", dev_str[gnum, :])
                if line[:28] == "deviatoric strain in crystal":
                    #print "strain crystal starts at line : ", i
                    linestrain[gnum] = int(i)                    
                if line[:24] == "deviatoric strain in lab":
                    #print "strain sample starts at line : ", i
                    linestrain2[gnum] = int(i)
                if line[:36] == "coordinates of a*, b*, c* in X, Y, Z":
                    #print "matrix starts at line : ", i
                    linemat[gnum] = int(i)
                    grainfound = 0
                    if gnum < gnumtot :
                        grainnumstr = grainnumstr + 1
    finally:
        linetot = i
        f.seek(0)

    print("calib :")
    print_calib(calib)
    
    matstarlab = zeros((gnumtot, 9), float)
    strain6 = zeros((gnumtot, 6), float)
    strain6sample = zeros((gnumtot, 6), float)
    npeaks = zeros(gnumtot, int)
    print("linemat = ", linemat)

    is_good_grain = np.array(is_good_grain, dtype=int)
    print("is_good_grain =", is_good_grain)

    #f = open(filestr, 'r')
    listspot = []
    listmat = []
    liststrain = []
    liststrain2 = []
    gnum = 0
    i = 0
    print("grain number : ", gnum)
    try:
        for line in f:
            if (i >= linestartspot[gnum]) & (i < lineendspot[gnum]):
                #print line
                listspot.append(line.rstrip(PAR.cr_string).split())
            if (i >= linemat[gnum]) & (i < linemat[gnum]+3) :
                #print line
                listmat.append(line.rstrip(PAR.cr_string).split()) 
            if i >= linestrain[gnum] & (i < linestrain[gnum]+3):
                #print line
                liststrain.append(line.rstrip(PAR.cr_string).split())                 
            if i >= linestrain2[gnum] & (i < linestrain2[gnum]+3):
                liststrain2.append(line.rstrip(PAR.cr_string).split())   
            if (i>linemat[gnum]+2) & (line[0] == 'G') | (i==linetot-1):
                if is_good_grain[gnum]==1:
                    matstr = np.array(listmat, dtype=float)
                    satocrs = matstr.transpose()  
                    print("strained orientation matrix (satocrs) = \n", satocrs)
                    matstarlab[gnum, :] = F2TC.matxmas_to_matstarlab(satocrs, calib)
                    data_str = np.array(listspot, dtype=float)
                    print("first / last lines of data_str :")
                    print(data_str[0, :2])
                    print(data_str[-1, :2])
                    data_str[:, 2:5] = -data_str[:, 2:5]   # erreur dans XMAS version 2006

                    strain3x3 = np.array(liststrain, dtype=float)
                    print("strain crystal : \n", strain3x3.round(decimals=2))
                    strain6[gnum, :] = epsmat_to_epsline(strain3x3)
#                    print liststrain2
                    strain2_3x3 = np.array(liststrain2, dtype=float)
                    print("strain sample : \n", strain2_3x3.round(decimals=2))
                    strain6sample[gnum, :] = epsmat_to_epsline(strain2_3x3)
                    toto = np.array([1., 1., 1., -1., 1., -1.])
                    strain6sample[gnum,:] = np.multiply(strain6sample[gnum,:], toto)  # change sign of epsxy and epsyz
                    if min_matLT == True:
                        print("warning : min_matLT = True in read_all_grains_str")
                        print("will need recalculate_strain_from_matrix = 1 in convert_xmas_str_to_LT_fit")
                        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab[gnum, :])
                        matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
                        matLT3x3 = matmin * 1.0
                        matstarlab[gnum, :] = F2TC.matstarlabLaueTools_to_matstarlabOR(matLT3x3)
                        # transfo des HKL a verifier
                        hklmin = dot(transfmat, data_str[:, 2:5].transpose()).transpose()
                        data_str[:, 2:5] = hklmin

                    #data_str_sorted = sort_peaks_decreasing_int(data_str, 10)
                    #data_str = data_str_sorted * 1.0

                    if gnum == 0:
                        data_str_all = data_str * 1.0
                    else:
                        data_str_all = vstack((data_str_all, data_str))

                    #print data_str
                    npeaks[gnum] = shape(data_str)[0]
                    print("number of indexed peaks :", npeaks[gnum])
                    print("first peak : ", data_str[0, 2:5])
                    print("last peak : ", data_str[-1, 2:5])                

                    gnum = gnum + 1
                    listspot = []
                    listmat = []
                    liststrain = []
                    liststrain2 = []
                    if i < linetot-1:
                        print("grain number : ", gnum)
                else:
                    print("bad grain")
                    gnum = gnum + 1
                    listspot = []
                    listmat = []
                    liststrain = []
                    liststrain2 = []
                    if i < linetot-1:
                        print("grain number : ", gnum)

            i = i+1
    finally:
        f.close()

    #print "return(data_str, satocrs, calib, dev_str)"
    #print "data_str :  xy(exp) 0:2 hkl 2:5 xydev 5:7 energy 7 dspacing 8  intens 9 integr 10"


    print("gnumtot = ", gnumtot)
    print("npeaks = ", npeaks)
    print("shape(data_str_all) =", shape(data_str_all))
    #print data_str_all[:,2]

    return(data_str_all[:, :11], matstarlab, calib, dev_str, npeaks, strain6, strain6sample)

def read_xmas_txt_file_from_seq_file(filexmas, 
                                read_all_cols="yes",     
                                list_column_names=["GRAININDICE", "ASIMAGEINDICE", "STRNINDEX", "ASSPIXDEV", "ASVONMISES","RSS"]):
    #29May13
    nameline =  "ASIMAGEINDICE GRAININDICE ASXSTAGE ASYSTAGE ASDD ASXCENT ASYCENT ASXBET ASXGAM ASXALFD ASXBETD\
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
    f = open(filexmas, 'r')
    i = 0
    try:
        for line in f:
            if i == 0: npts = line.rstrip("  " + PAR.cr_string)
            if i == 1: nameline1 = line.rstrip(PAR.cr_string)
            i = i+1
            if i>2: break
    finally:
        f.close() 

    print(npts)   
    print(nameline1)
    listname = nameline1.split()

    data_sum = loadtxt(filexmas, skiprows=2)   

    if read_all_cols == "yes":
        print("shape(data_sum) = ", shape(data_sum))
        return(data_sum, listname, npts)
        
    else :     
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

        return data_sum_select_col, list_column_names, npts

# if 0 : # find symmetry operations with det > 1
#     allop = DictLT.OpSymArray
#     ind0 = []
#     for j in range(48):
#         det1 = np.linalg.det(allop[j])
#         print(j, det1)
#         if det1 > 0. : ind0.append(j)

#     print(ind0)
#     ind0 = np.array(ind0,int)
#     jksqld

#     allop = DictLT.OpSymArray
#     indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])
#     goodop = allop[indgoodop]
#     #print shape(goodop)
           
def matstarlab_to_matstarsample3x3(matstarlab,
                                   omega=None, # was PAR.omega_sample_frame
                                   mat_from_lab_to_sample_frame=PAR.mat_from_lab_to_sample_frame
                                   ):

    matstarlab3x3 =  GT.matline_to_mat3x3(matstarlab)
    
    if (omega  is  not None)&(mat_from_lab_to_sample_frame  is None) : # deprecated - only for retrocompatibility
        omega = omega * PI / 180.0
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        mat_from_lab_to_sample_frame = array([[1.0, 0.0, 0.0],
                                        [0.0, np.cos(omega), np.sin(omega)],
                                        [0.0, -np.sin(omega), np.cos(omega)]])

    matstarsample3x3 = dot(mat_from_lab_to_sample_frame, matstarlab3x3)

    #print  "matstarsample3x3 =\n" , matstarsample3x3.round(decimals=6)
    return(matstarsample3x3)
    
def matstarsample3x3_to_matstarlab(matstarsample3x3, 
                                   omega = None, # was PAR.omega_sample_frame
                                   mat_from_lab_to_sample_frame=PAR.mat_from_lab_to_sample_frame): #29May13


    if (omega  is  not None)&(mat_from_lab_to_sample_frame  is None) : # deprecated - only for retrocompatibility
        omega = omega * PI / 180.0
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        mat_from_lab_to_sample_frame = array([[1.0, 0.0, 0.0],
                                        [0.0, np.cos(omega), np.sin(omega)],
                                        [0.0, -np.sin(omega), np.cos(omega)]])

    mat_from_sample_to_lab_frame = transpose(mat_from_lab_to_sample_frame)

#    # rotation de -omega autour de l'axe x pour repasser dans Rsample
#    omega = omega*PI/180.0
#    matrot = array([[1.0,0.0,0.0],[0.0,np.cos(omega),-np.sin(omega)],[0.0,np.sin(omega),np.cos(omega)]])

    matstarlab3x3 = dot(mat_from_sample_to_lab_frame, matstarsample3x3)

    matstarlab =  GT.mat3x3_to_matline(matstarlab3x3)
    #print  "matstarsample3x3 =\n" , matstarsample3x3.round(decimals=6)

    return(matstarlab)    
    
def fromMatrix_toQuat(matrix):
    #print "matrix \n", matrix
    mat = array([matrix[0, 0], matrix[1, 0], matrix[2, 0], 0.0,
                matrix[0, 1], matrix[1, 1], matrix[2, 1], 0.0,
                matrix[0, 2], matrix[1, 2], matrix[2, 2], 0.0,
                0.0, 0.0, 0.0, 1.0])

    #print shape(mat)
    
    toto = 1.0+mat[0]+mat[5]+mat[10]
    if toto < 0.0 :
        print("warning : negative toto in fromMatrix_toQuat",toto)
        toto = abs(toto)
        #print matrix
        
    soso = np.sqrt(toto) * 2.0
    qx = (mat[9] - mat[6])/soso
    qy = (mat[2] - mat[8])/soso
    qz = (mat[4] - mat[1])/soso
    qw = 0.25 * soso
    return np.array([qx,qy,qz,qw])
   
def fromQuat_to_vecangle(quat):
    """ from quat = [vec,scalar] = [sin angle/2 (unitvec(x,y,z)), cos angle/2]
    gives unitvec and angle of rotation around unitvec
    """
    normvectpart = np.sqrt(quat[0]**2 + quat[1]**2+quat[2]**2+quat[3]**2)
    #print "nor",normvectpart
    angle = np.arccos(quat[3] / normvectpart) * 2.0 # in radians
    unitvec = array(quat[:3]) / np.sin(angle / 2.0) / normvectpart
    return unitvec, angle

def fromvecangle_to_Quat(unitvec,angle):
    # angle in radians
    quat = zeros(4, float)
    quat[3] = np.cos(angle/2.0)
    quat[:3] = np.sin(angle/2.0)*unitvec
    return quat

def from_axis_vecangle_to_mat(vlab, ang1):

    # matrot gives the vectors of R1 as columns on the vectors of R0
    # R1 obtained from R0 by rotation of ang1 around v
    # coordinates of v are in R0

    # xR1 = m00 * xR0 + m10 * yR0 + m20 * zR0

    # ang1 in degrees

    vlab = vlab/norme(vlab)
    ang1 = ang1 * PI/180.0
    rotv = array([[0.,-vlab[2],vlab[1]],[vlab[2], 0., -vlab[0]], [-vlab[1], vlab[0], 0.0]])
    matrot = np.cos(ang1) * eye(3) + (1 - np.cos(ang1)) * outer(vlab,vlab)+ np.sin(ang1) * rotv
    #print "matrot = ", matrot
    return matrot

def from_axis_to_reflection_matrix(axis):

    ulab = axis / norme(axis)

    mat = np.zeros((3,3),float)
    mat[0, 0]= 1. -2.* ulab[0] * ulab[0]
    mat[1, 1]= 1. -2.* ulab[1] * ulab[1]
    mat[2, 2]= 1. -2.* ulab[2] * ulab[2]

    mat[0, 1] = -2.* ulab[0] * ulab[1]    
    mat[1, 0] = mat[0,1] * 1.
    
    mat[0, 2] = -2. * ulab[0] * ulab[2]    
    mat[2, 0] = mat[0, 2] * 1.
    
    mat[1, 2] = -2. * ulab[1] * ulab[2]    
    mat[2, 1] = mat[1, 2] * 1.
    
    return mat


def init_numbers_for_crystal_opsym_and_first_stereo_sector(elem_label = "Si") :

     # invisible arguments in matstarlab_to_orientation_color_rgb

#    allop = DictLT.OpSymArray

    if test_if_cubic(elem_label) or 1:   # PATCH TO BE REMOVED

        PAR.struct1 = "cubic"
        PAR.nop = 48
        #opsym avec det>1
        PAR.indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])

        hkl_3 = array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.]])    

        PAR.uqref_cr = zeros((3, 3), float)
        # uqref_cr 3 vecteurs 001 101 111 en colonnes
        for i in range(3):
                PAR.uqref_cr[:, i] = hkl_3[i, :] / norme(hkl_3[i, :])

    #    print "cosines between extremities of first triangle : "
        cos01 = inner(PAR.uqref_cr[:, 0], PAR.uqref_cr[:, 1])
        cos02 = inner(PAR.uqref_cr[:, 0], PAR.uqref_cr[:, 2])
        cos12 = inner(PAR.uqref_cr[:, 1], PAR.uqref_cr[:, 2])

    #    print round(cos01,3), round(cos02,3), round(cos12,3)

        PAR.cos0 = min(cos01, cos02)
        PAR.cos1 = min(cos01, cos12)
        PAR.cos2 = min(cos02, cos12)

    #    print "minimum cos with 001 101 and 111 : "
    #    print round(cos0,3), round(cos1,3), round(cos2,3)

        # vectors normal to frontier planes of stereographic triangle

        uqn_b = cross(PAR.uqref_cr[:, 0],PAR.uqref_cr[:, 1])
        PAR.uqn_b = uqn_b / norme(uqn_b)
        uqn_g = cross(PAR.uqref_cr[:, 0],PAR.uqref_cr[:, 2])
        PAR.uqn_g = uqn_g / norme(uqn_g)
        uqn_r = cross(PAR.uqref_cr[:, 1],PAR.uqref_cr[:, 2])
        PAR.uqn_r = uqn_r / norme(uqn_r)

    if test_if_monoclinic(elem_label):

        PAR.struct1 = "monoclinic"
        PAR.nop = 2
        PAR.indgoodop = array([0, 4])      

        print("still pending : monoclinic")

    return 1

def uq_cr_to_orientation_color_rgb(uq,
                                   cryst_struct = "cubic"):
        
    # RGB coordinates
    rgb_pole = zeros(3,float)
    # blue : distance in q space between M tq OM = uq et le plan 001 101 passant par O
    
    if cryst_struct == "cubic" :
        
        rgb_pole[2] = abs(inner(uq,PAR.uqn_b))/abs(inner(PAR.uqref_cr[:,2], PAR.uqn_b))
        rgb_pole[1] = abs(inner(uq,PAR.uqn_g))/abs(inner(PAR.uqref_cr[:,1], PAR.uqn_g))
        rgb_pole[0] = abs(inner(uq,PAR.uqn_r))/abs(inner(PAR.uqref_cr[:,0], PAR.uqn_r))
        
    elif cryst_struct == "monoclinic" :
        
        # la il faut un udir pas un urec      
        
        xdir = uq

        # red
        dir010 = np.array([0.,1.,0.])
        
        rgb_pole[0] = inner(xdir,dir010)   

        rec100 = np.array([1.,0.,0.])
 
        # green
        rgb_pole[1] = (acos(inner(xdir,-rec100))/PI)*(1.-rgb_pole[0])
       
        # blue      
        rgb_pole[2] = (acos(inner(xdir,rec100))/PI)*(1.-rgb_pole[0])
                
    # convention OR
#    if (abs(uq[0])< 0.01):
#        print "uq =", uq, "rgb_pole = ", rgb_pole #, max(rgb_pole)
    rgb_pole = rgb_pole / max(rgb_pole)
    # convention Tamura
#    rgb_pole = rgb_pole / norme(rgb_pole)

    #print "rgb_pole :"
    #print rgb_pole
    
    return(rgb_pole)
    
def plot_orientation_half_circle_color_code(elem_label = "m_zirconia"):
    
    # for monoclinic
    
    # 23Dec15
        
    # plot orientation color scale in stereographic projection
#    p.rcParams['savefig.bbox'] = None # "tight"
    p.figure(figsize = (6,12))

    numrand1 = 50
    range001 = np.arange(numrand1+1)
    range001 = np.array(range001, dtype=float)/numrand1

    angrange = range001*1.
    
    dir001 = np.array([0.,0.,1.])        
    dir010 = np.array([0.,1.,0.])
    rec100 = np.array([1.,0.,0.])    

    for i in range(numrand1+1):
        for j in range(numrand1+1):

            uq1 = (1.-range001[i])*rec100 + range001[i]* (angrange[j]*dir010 + (1.-angrange[j])*dir001)
            uq1 = uq1/norme(uq1)            

            qsxy = hkl_to_xystereo(uq1, down_axis = [1.,0.,0.])
            
            # RGB coordinates
            
            rgb_pole = uq_cr_to_orientation_color_rgb(uq1, cryst_struct = "monoclinic")

            rgb_pole = rgb_pole.clip(min=0.0,max=1.0)
            
            p.plot(qsxy[0],qsxy[1],marker = 'o', markerfacecolor = rgb_pole, markeredgecolor = rgb_pole, markersize = 5, clip_on = False)     

            uq1 = (1.-range001[i])*(-rec100) + range001[i]* (angrange[j]*dir010 + (1.-angrange[j])*dir001)
            uq1 = uq1/norme(uq1)            

            qsxy = hkl_to_xystereo(uq1, down_axis = [1.,0.,0.])
            
            # RGB coordinates
            
            rgb_pole = uq_cr_to_orientation_color_rgb(uq1, cryst_struct = "monoclinic")

            rgb_pole = rgb_pole.clip(min=0.0,max=1.0)
            
            p.plot(qsxy[0],qsxy[1],marker = 'o', markerfacecolor = rgb_pole, markeredgecolor = rgb_pole, markersize = 5, clip_on = False)     

    dict_vectors_rec = {"a*" : [1.,0.,0.], "-a*" : [-1.,0.,0.], "c*" : [0.,0.,1.], "a*+b*" : [1.,1.,0.],
                        "a*+c*" : [1.,0.,1.], "-a*+b*" : [-1.,1.,0.], "-a*+c*" : [-1.,0.,1.], "b*+c*" : [0.,1.,1.],
                        "a*+b*+c*" : [1.,1.,1.], "-a*+b*+c*" : [-1.,1.,1.]}
                    
    dlatdeg = np.array(DictLT.dict_Materials[elem_label][1], dtype = float)
    print("direct lattice parameters angstroms degrees = \n ", dlatdeg)    
    
    dlatrad = deg_to_rad(dlatdeg)
    Bstar = dlat_to_Bstar(dlatrad) 
    
    # ici repere labo = repere Rcart_rec a* b c OND
    matstarlab = GT.mat3x3_to_matline(Bstar)
    # matstarlab = a* b* c* sur Rcart_rec
    # matdirlab3x3 = a b c en colonnes sur Rcart_rec
    matdirlab3x3, rlat1 = matstarlab_to_matdirlab3x3(matstarlab)

    print("adding black circles for reciprocal vectors")      

    for key, value in dict_vectors_rec.items():
        print(key)
        uq_cr_rec = np.array(value)
        uq_cr_rec_cart = dot(Bstar,uq_cr_rec)
        qsxy = hkl_to_xystereo(uq_cr_rec_cart, down_axis = [1.,0.,0.])    
        p.plot(qsxy[0],qsxy[1],marker = 'o', mfc = "w", mec = "k", mew = 2, ms = 8, clip_on = False)           

    dict_vectors_dir = {"a" : [1.,0.,0.],
                        "-a" : [-1.,0.,0.],
                        "b" : [0.,1.,0.],
                        "c" : [0.,0.,1.],
                        "a+b" : [1.,1.,0.],
                        "a+c" : [1.,0.,1.],
                        "-a+b" : [-1.,1.,0.],
                        "-a+c" : [-1.,0.,1.],
                        "b+c" : [0.,1.,1.],
                        "a+b+c" : [1.,1.,1.],
                        "-a+b+c" : [-1.,1.,1.],
                    }
                    
    print("adding black squares for direct vectors")      

    for key, value in dict_vectors_dir.items():
        print(key)
        uq_cr_dir = np.array(value)
#        print "matdirlab3x3 = ", matdirlab3x3
#        print "uq_cr_dir = ", uq_cr_dir
        uq_cr_rec_cart = dot(matdirlab3x3,uq_cr_dir)
        qsxy = hkl_to_xystereo(uq_cr_rec_cart, down_axis = [1.,0.,0.])    
        p.plot(qsxy[0],qsxy[1],marker = 's', mfc = "w", mec = "k", mew = 2, ms = 8, clip_on = False)           
                                       
    p.xlim(0.,1.)
    p.ylim(-1.,1.)

    return(0)
    
def matstarlab_to_orientation_color_rgb(matstarlab, 
                                       axis_pole_sample,
                                       omega = None, # was PAR.omega_sample_frame
                                       mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame,
                                       elem_label = "Si") : #, matrot, uqref_cr) : # return_matrix = "yes", return_cosines = "no") : #, xyz_sample_azimut):

    # for cubic system only 
    # first apply crystal symmetry operations
    # to calculate the stereo_mat orientation matrix from the initial orientation matrix
     # stereo_mat has one particular sample axis 
     # in the first triangle of stereographic projection 100 - 101 - 111
     # then use_stereo_mat
     # to calculate RGB color code of orientation
     # this ensures that all data points in the same crystallite 
     # end up with almost the same color in the orientation map

        # 09Dec15
        # replaces calc_cosines_first_stereo_triangle
        # which was restricted to cubic
        # test if unit cell is cubic from elem_label
        # and allow for monoclinic (i.e. only one symmetry operation = rotation 180 degrees around b)
        # monoclinic means beta different from 90 deg here

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

        if (omega  is  not None)&(mat_from_lab_to_sample_frame  is None) : # deprecated - only for retrocompatibility
            omega = omega*PI/180.0
            # rotation de -omega autour de l'axe x pour repasser dans Rsample
            mat_from_lab_to_sample_frame = array([[1.0,0.0,0.0],[0.0,cos(omega),np.sin(omega)],[0.0,-np.sin(omega),np.cos(omega)]])

        mat_from_sample_to_lab_frame = transpose(mat_from_lab_to_sample_frame)
                        
#        omega = omega*PI/180.0 
#        # rotation de omega autour de l'axe x pour repasser dans Rlab
#        matrot = array([[1.0,0.0,0.0],[0.0,np.cos(omega),-np.sin(omega)],[0.0,np.sin(omega),np.cos(omega)]])

        verbose = 0
#
        allop = DictLT.OpSymArray
#
#        if test_if_cubic(elem_label) : # PATCH TO BE REMOVED
#            struct1 = "cubic"
#            PAR.nop = 48
#            #opsym avec det>1        
#            indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])
#
#        elif test_if_monoclinic(elem_label) :
#                struct1 = "monoclinic"
#                PAR.nop = 2
#                indgoodop = array([0,4])
#        else :
#            print "warning : color code not defined for this crystal symmetry"
#            print "crystal will be treated as cubic"
#            indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])

        # xyz-sample convention OR
        
        upole_sample = axis_pole_sample / norme(axis_pole_sample)
        upole_lab = dot(mat_from_sample_to_lab_frame,upole_sample)
        #print "pole axis - sample coord : ", upole_sample
        #print "pole axis - lab coord : ", upole_lab.round(decimals=3)

        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        # 11Dec15 : dans le cas du monoclinique le repere OND est fabrique avec a* b* et c
        
        mat = GT.matline_to_mat3x3(matstarlabOND)

        Bstar = rlat_to_Bstar(mat_to_rlat(matstarlab))
        matdef = GT.matline_to_mat3x3(matstarlab)

##        # test
##        matdef = GT.matline_to_mat3x3(matstarlab)
##        print "matdef"
##        print matdef.round(decimals=4)
##        print "matdef recalc"
##        print dot(mat,Bstar).round(decimals=4)
        
        cosangall = zeros((2*PAR.nop,3), float)
        ranknum = np.arange(2*PAR.nop)
        opsym = 2*list(range(PAR.nop))   # la il faut garder le range et pas np.arange car liste 0 - 48 puis 0 - 48
        if 0 :
            print(shape(opsym))
            print(opsym[0], opsym[47], opsym[48], opsym[-1])
            print(shape(ranknum))
            print(shape(cosangall))
            jkdasl

        matk_lab = zeros((2*PAR.nop,3,3),float)

        if PAR.struct1 == "cubic" :
            for k in range(PAR.nop):
                matk_lab[k] = dot(mat,allop[k])
                # retour a un triedre direct si indirect
                if k not in PAR.indgoodop :
                        #print "yoho"
                        matk_lab[k,:,2]= -matk_lab[k,:,2]
                uqrefk_lab = dot(matk_lab[k], PAR.uqref_cr)
                for j in range(3):
                        cosangall[k,j] = inner(upole_lab, uqrefk_lab[:,j])
                        cosangall[k+PAR.nop,j] = inner(-upole_lab, uqrefk_lab[:,j])
            

        data1 = column_stack((opsym,cosangall,ranknum))
        #print shape(data1)

        # priorites 001 101 111
        np1 = 1
        np2 = 2
        np3 = 3

        #print "opsym  cos001 cos101 cos111 ranknum"
        #print data1[:,1:4].round(decimals = 3)
        data1_sorted = sort_list_decreasing_column(data1, np1)

        #print data1_sorted.round(decimals = 3)

        ind1 = where(abs(data1_sorted[:,np1]-data1_sorted[0,np1])<1e-3)
        #print ind1
        data2_sorted = sort_list_decreasing_column(data1_sorted[ind1[0],:], np2)
        #print data2_sorted.round(decimals = 3)
        ind2 = where(abs(data2_sorted[:,np2]-data2_sorted[0,np2])<1e-3)
        #print ind2
        data3_sorted = sort_list_decreasing_column(data2_sorted[ind2[0],:], np3)
        #print data3_sorted.round(decimals = 3)
        ind3 = where(abs(data3_sorted[:,np3]-data3_sorted[0,np3])<1e-3)
        #print ind3

        #print "initial matrix abcstar_on_xyzlab"

        #print GT.matline_to_mat3x3(matstarlab).round(decimals=4)

        #print "pole axis in Rsample"
        #print axis_pole_sample

        #print "new matrix abcstar_on_xyzlab with polar axis in first stereo triangle :"
        opsymres = []
        rankres = []
        for i in ind3[0]:
                op1 = int(round(data3_sorted[i,0],1))
                rank1 = int(round(data3_sorted[i,-1],1))
                opsymres.append(op1)
                rankres.append(rank1)
                #print "opsym =" , op1
                #print matk_lab[op1].round(decimals=4)

        opsymres = np.array(opsymres,dtype = int)
        #print opsymres

        if 0 in opsymres : op1 = 0
        else : op1 = opsymres[0]

        matONDnew = matk_lab[op1]
        opres = dot(transpose(mat),matONDnew)
        #print "opres \n", opres.round(decimals=1)
        #print "det(opres)", np.linalg.det(opres)
        toto = dot(opres.transpose(),Bstar)
        Bstarnew = dot(toto,opres)
        matdef2 = dot(matONDnew,Bstarnew) 
        matstarlabnew = GT.mat3x3_to_matline(matdef2)

        abcstar_on_xyzsample = matstarlab_to_matstarsample3x3(matstarlabnew, 
                                                              omega = PAR.omega_sample_frame)
        xyzsample_on_abcstar = np.linalg.inv(abcstar_on_xyzsample)

        transfmat = np.linalg.inv((dot(np.linalg.inv(matdef), matdef2).round(decimals = 1)))

        #print "transfmat \n", transfmat

        #print "final : xyzsample_on_abcstar"
        #print xyzsample_on_abcstar.round(decimals=4)

        #print "matrix"
        #print "initial" , matstarlab.round(decimals=4)
        #print "final ", matstarlabnew.round(decimals=4)

        if verbose :
            print("op sym , rank, cos ", op1, ranknum[rankres[0]], cosangall[rankres[0]].round(decimals=3))

        if ranknum[rankres[0]]< 48 : cos_end = cosangall[rankres[0]]
        else : cos_end = -cosangall[rankres[0]]
        
        cos_end_abs = abs(cos_end)
#        print cos0, cos1, cos2
#        print cos_end_abs
        if (cos_end_abs[0] < PAR.cos0)|(cos_end_abs[1] < PAR.cos1)|(cos_end_abs[2] < PAR.cos2):
            print("problem : pole axis not in first triangle")
            rgb_pole = zeros(3, float)
            return(matstarlabnew, transfmat, rgb_pole)
#            exit()
        #else : print "cosines OK"

        #print "new crystal coordinates of axis_pole :"
        uq = np.dot(xyzsample_on_abcstar, upole_sample)
        if verbose :
            print("uq :")
            print(uq.round(decimals=4))
            print("PAR.uqref_cr :")
            print(PAR.uqref_cr.round(decimals=3))
            
        rgb_pole = uq_cr_to_orientation_color_rgb(uq, cryst_struct = PAR.struct1)

        return(matstarlabnew, transfmat, rgb_pole)
        
def sort_list_decreasing_column(data_str, colnum):
                              
    #print "sort list, decreasing values of column ", colnum
    
    npics = shape(data_str)[0]
    #print "nlist = ", npics
    index2 = zeros(npics, int)
    
    index1 = argsort(data_str[:,colnum])
    for i in range(npics):
        index2[i]=index1[npics-i-1]
    #print "index2 =", index2
    data_str2 = data_str[index2]

    return(data_str2)

def hkl_to_xystereo(hkl0,polar_axis = [0.,0.,1.], down_axis = [1.,0.,0.], return_more = None) :

    uq = hkl0/norme(hkl0)
    uz = polar_axis /norme(polar_axis)
    udown = down_axis / norme(down_axis)
    uright = cross(uz,udown)
    uqz = inner(uq,uz)
    change_sign = 1
    if uqz < 0.0 :
            print("warning : uq.uz < 0 in hkl_to_xystereo, change sign of uq")
            uqz = -uqz
            uq = -uq
            if return_more  is  not None :
                    change_sign = -1
    qs = (uq - uqz*uz)/(1.0+uqz)
    #print qs.round(decimals=3)
    #print norme(qs)
    qsxy = np.array([inner(qs,uright),-inner(qs,udown)])
    #print qsxy.round(decimals=3)

    if return_more  is None :  return(qsxy)
    else : return(qsxy, change_sign)
        
def remove_duplicates(data, dtol = 0.01):
        ndat = shape(data)[0]
        nb_common_peaks,iscommon1,iscommon2 = find_common_peaks(data, data, dxytol = dtol, verbose = 1)
        ind1 = where(iscommon1 == 1)
        # duplicates give iscommon1 > 1
        datanew = data[ind1[0],:]
        return(datanew)
        

##********************************************************************************************************************************

def test_if_cubic(elem_label) :
    
    latticeparams = np.array(DictLT.dict_Materials[elem_label][1], dtype = float)    
#    print "test_if_cubic"
#    print "latticeparams (lengths_in_Angstroems, angles_in_deg) = \n", latticeparams

    # angles_in_deg
    
    all_ang = latticeparams[3:]
    dang = all_ang-np.array([90.,90.,90.])
    if norme(dang)> 0.01 : 
#        print "not cubic"
        return(0)
    all_lengths = latticeparams[0:3]
    dlength = all_lengths - all_lengths[0]
    if norme(dlength) > 0.01 : 
#        print "not cubic"
        return(0)
    
#    print "cubic"
    return(1)
    
def test_if_monoclinic(elem_label) :
    
    latticeparams = np.array(DictLT.dict_Materials[elem_label][1], dtype = float)    
    print("latticeparams (lengths_in_Angstroems, angles_in_deg) = \n", latticeparams)

    # angles_in_deg
    
    all_ang = latticeparams[3:]
    dang = all_ang-np.array([90.,90.,90.])
    if (dang[0]< 0.01)&(dang[2]<0.01)&(dang[1]> 0.01) : return(1)
    else : return(0)


def test_index_refine(filedat, paramdetector_top, proposed_matrix=None, 
                      check_grain_presence=None, paramtofit="strain", elem_label="UO2", 
                      grainnum=1, remove_sat=0, elim_worst_pixdev=1, 
                      maxpixdev=1.0, spot_index_central=[0, 1, 2, 3, 4, 5], nbmax_probed=20, 
                      energy_max=22, rough_tolangle=0.5 , fine_tolangle=0.2, 
                      Nb_criterium=15, NBRP=1, mark_bad_spots=0, 
                      boolctrl = '1' * 8, CCDlabel = "MARCCD165",
                      use_weights=False,  # les parametres marques par * sont muets dans serial_index_refine
                      nLUT = 3,  #*
                      LUT = None, #*
                      min_matLT = True, #* 
                      stereo_mat = False, #* 
                      axis_pole_sample = None,
                      verbose = 0,
                      xmin_xmax_ymin_ymax = None): #* 
    
    #29May13
    """
    data treatment sequence:
    
    indexation
    strain refinement
    
    keep this : definition of parameters

    spot_index_central = [0,1,2,3,4,5]                   # central spot or list of spots
    nbmax_probed = 20              # 'Recognition spots set Size (RSSS): '
    
    energy_max = 22

    rough_tolangle = 0.5            # 'Dist. Recogn. Tol. Angle (deg)'  # used only if proposed_matrix  is None
    fine_tolangle = 0.2             # 'Matching Tolerance Angle (deg)'  # spotlink exp-theor tolerance
    Nb_criterium = 15                # 'Minimum Number Matched Spots: '
    NBRP = 1 # number of best result for each element of spot_index_central
    remove_sat = 0   # = 1 : keep saturated peaks (Ipixmax = saturation value) for indexation but remove them for refinement
    elim_worst_pixdev = 1 : for 2nd refinement : eliminate spots with pixdev > maxpixdev after first refinement
    mark_bad_spots = 1 : for multigrain indexation, at step n+1 eliminate from the starting set all the spots
                                  more intense than the most intense indexed spot of step n
                                  (trying to exclude "grouped intense spots" from starting set)

      min_matLT = True : final orientation matrix minimizes Euler Angles in LT laboratory frame
      stereo_mat = True, axis_pole_sample = [1.0,0.0,0.0] :
      final orientation matrix has +/-xsample axis in first triangle of stereographic projection 100 - 101 - 111
                                  
    30Nov12 : add new column Etheor
"""
    pixelsize = DictLT.dict_CCD[CCDlabel][1]
    dim = DictLT.dict_CCD[CCDlabel][0]
    
    latticeparams = DictLT.dict_Materials[elem_label][1]
        
    Bmatrix = CP.calc_B_RR(latticeparams)  
    
    if test_if_cubic(elem_label) : print("cubic material")
    else :
        print("non-cubic material")
        min_matLT = False
        stereo_mat = False
    
    # MAR ou Roper
##    pixelsize = 165. / 2048
##    dim1 = (2048,2048)
    # VHR
    #pixelsize = 0.031
    #dim1 = (2594, 3764)
    
    # Automation of indexation + strain refinement
    calib = np.array(paramdetector_top, dtype=float)
    data_dat = np.loadtxt(filedat, skiprows=1)
    nspots = shape(data_dat)[0]

    numcolint = 3  # adapter au format du .dat
    data_dat = sort_peaks_decreasing_int(data_dat, numcolint)
    
    if remove_sat :
        col_Ipixmax = -1  # adapter au format du .dat
        saturation = DictLT.dict_CCD[PAR.CCDlabel][2]
        data_sat = zeros(nspots, int)
        data_Ipixmax = np.array(data_dat[:, col_Ipixmax], dtype=int)
        ind0 = where(data_Ipixmax == saturation)
        data_sat[ind0[0]] = 1
        print(data_sat)
        
    if mark_bad_spots :
        col_isbadspot = 4  # adapter au format du .dat
        data_isbadspot = np.array(data_dat[:, col_isbadspot], dtype=int)

    data_pixX = data_dat[:, 0] * 1.0
    data_pixY = data_dat[:, 1] * 1.0
    data_I = data_dat[:, numcolint] * 1.0 
    
    if xmin_xmax_ymin_ymax != None :
        cond_xmin = (data_pixX > xmin_xmax_ymin_ymax[0])
        cond_xmax = (data_pixX < xmin_xmax_ymin_ymax[1])
        cond_ymin = (data_pixY > xmin_xmax_ymin_ymax[2])
        cond_ymax = (data_pixY < xmin_xmax_ymin_ymax[3])
        cond_total = cond_xmin * cond_xmax * cond_ymin * cond_ymax
        indfilt = np.where(cond_total > 0)
        data_pixX = data_pixX[indfilt[0]]
        data_pixY = data_pixY[indfilt[0]]
        data_I = data_I[indfilt[0]]
        print("filtering on x y")
        print("xmin_xmax_ymin_ymax = ", xmin_xmax_ymin_ymax)
        print("number of spots remaining = ", len(indfilt[0]))
        
    data_twotheta, data_chi = F2TC.calc_uflab(data_pixX, 
                                              data_pixY, 
                                              calib, 
                                              returnAngles=1, 
                                              pixelsize=pixelsize)   

    data_theta = data_twotheta / 2.0

    detectorparameters = {}
    detectorparameters['kf_direction'] = 'Z>0'
    detectorparameters['detectorparameters'] = calib
    detectorparameters['detectordiameter'] = 165.
    detectorparameters['pixelsize'] = pixelsize
    detectorparameters['dim'] = dim


    if proposed_matrix  is None :   # only for paramtofit = "strain"
    
        print("find matrix from scratch using (qi,qj) angles")

        if paramtofit == "calib" :
            print("calib fitting needs guess matrix : please set proposed_matrix first")
            return(0)

        if mark_bad_spots :
            ind1 = where(data_isbadspot == 0)
            data_theta1 = data_theta[ind1[0]]
            data_chi1 = data_chi[ind1[0]]
            data_x1 = data_pixX[ind1[0]]
            data_y1 = data_pixY[ind1[0]]
            print(shape(data_theta))
            print(shape(data_theta1))

        else :
            data_theta1 = data_theta * 1.0
            data_chi1 = data_chi * 1.0
            data_x1 = data_pixX * 1.0
            data_y1 = data_pixY * 1.0

        listcouple = np.transpose(np.array([data_theta1[:nbmax_probed], data_chi1[:nbmax_probed]]))
        
        Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        # classical indexation parameters:
        data = (2 * data_theta, data_chi, data_I)#, filecor)

        print("rough_tolangle ", rough_tolangle)
        print("fine_tolangle ", fine_tolangle)

        # indexation procedure

        start_tab_distance = Tabledistance
        
        if LUT  is None :                  
            LUT = INDEX.build_AnglesLUT(Bmatrix, nLUT)

        
        for spotnum in spot_index_central :
            print(data_x1[spotnum], data_y1[spotnum])

            RES_onespot = INDEX.getOrientMatrix_from_onespot(spotnum, 
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
                                              detectorparameters=detectorparameters)

            print("RES_onespot", RES_onespot)
            print(shape(RES_onespot))

            bestmat, matchingrate, latticeplanes_pairs, spots_pairs = RES_onespot

            if bestmat  is  not None and matchingrate >= 0:
                UBmat = bestmat
#                print "line 2021 in multigrain test_index_refine"
#                print "UBmat = ", UBmat
            else:
                raise ValueError("Orientation Matrix not found")
                        
    else :  # proposed_matrix  is  not None
    
        print("use proposed matrix")
        orientmatrix = proposed_matrix
        #UBmat = proposed_matrix
               
        if check_grain_presence  is  not None :
            
    #l'interet du check_grain_presence dans le cas d'une proposed_matrix pas tres precise
    #c'est de faire tourner le fit deux fois :
    #une premiere fois avec la matrice guess et peu de spots indexes
    #et une deuxieme fois avec une matrice plus precise et beaucoup de spots indexes       
    
            print("warning : check_grain_presence  is  not None")
            print("perform first approximate spotlink / refine using : finetol_angle = 0.2, emax_simul = 16")
            
            # then resimulate to output miller indices
            emax_simul = 16
#            emax_simul = energy_max
            veryclose_angletol = 0.2
            
            print("GetStrainOrient in test_index_refine")
    
            res1 = LAA.GetStrainOrient(orientmatrix, Bmatrix, elem_label,
                                               emax_simul,
                                               veryclose_angletol,
                                               data,
                                               addoutput=1,
                                               detectorparameters = detectorparameters)
    
            if (res1 == 0) | (res1  is None) :
                return(0)
            else : 
                UBmat, deviatoricstrain, RR, spotnum_i_pixdev, pixdev , nfit, spotnum_hkl = res1
        else :
    
            UBmat = proposed_matrix
            if verbose : print("starting UBmat = ", UBmat)
        
    if paramtofit == "strain" :
        data = (2.0 * data_theta, data_chi, data_I)#, filecor)
        
    elif paramtofit == "calib" :
        data = (2.0 * data_theta, data_chi, data_I, data_pixX, data_pixY)
        
    emax_simul = energy_max
    
    veryclose_angletol = fine_tolangle
    
    if paramtofit == "strain" :
        print("start strain refinement")
        
        if remove_sat :  
            saturated = data_sat
        else :
            saturated = None
            
        print("GetStrainOrient in test_index_refine")
        
        res2 = LAA.GetStrainOrient(UBmat,
                                    Bmatrix,
                                    elem_label,
                                    emax_simul,
                                    veryclose_angletol,
                                    data,
                                    addoutput=1,
                                    saturated = saturated,
                                    use_weights=use_weights,
                                    addoutput2=1,
                                    detectorparameters = detectorparameters)
                                                        
        if (res2 == 0) | (res2  is None) :
            return(0)
        else :
            UBmat_2, deviatoricstrain_2, RR_2, spotnum_i_pixdev_2, pixdev_2 , nfit_2, spotnum_hkl_2,\
                spotexpnum_spotsimnum_2, Intensity_list_2 = res2
                                                        
        #print spotnum_i_pixdev_2
            #print spotexpnum_spotsimnum_2
            #print Intensity_list_2
    
            if elim_worst_pixdev :
                pixdevlist = spotnum_i_pixdev_2[:, 2]
                if verbose : print(pixdevlist)
                ind0 = np.where(pixdevlist > maxpixdev)
                print(ind0[0])
                nbad = len(ind0[0])
                if nbad > 0 :
                    print("MAXPIXDEV = ", maxpixdev)
                    print("found ", nbad, " peaks with pixdev larger than ", maxpixdev)
                    ind1 = np.where(pixdevlist < maxpixdev)
                    ngood = len(ind1[0])
                    if ngood > 10 :
                        #print spotexpnum_spotsimnum_2
                        #print spotnum_hkl_2
                        #print Intensity_list_2
                        print("fit again after removing bad peaks")
                        
                        spotexpnum_spotsimnum_2 = spotexpnum_spotsimnum_2[ind1[0]]
                        spotnum_hkl_2 = (np.array(spotnum_hkl_2, dtype=int))[ind1[0]]
                        Intensity_list_2 = (np.array(Intensity_list_2, dtype=float))[ind1[0]]
                        spotnum_hkl_2 = list(spotnum_hkl_2)
                        Intensity_list_2 = list(Intensity_list_2)
                        
                        res_refinement = LAA.RefineUB(spotexpnum_spotsimnum_2,
                                                spotnum_hkl_2,
                                                Intensity_list_2,
                                                paramdetector_top, 
                                                data,
                                                UBmat_2,
                                                Bmatrix,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                use_weights=use_weights)
    
                        UBmat_2, newUmat_2, newBmat_2, deviatoricstrain_2, RR_2, spotnum_i_pixdev_2, pixdev_2 = res_refinement
    
                        nfit_2 = len(Intensity_list_2)
                        print("nfit_2 = ", nfit_2)
                    else :
                        print("not enough peaks with pixdev small enough : ngood = ", ngood)
                        return(0)

    elif paramtofit == "calib" :

        veryclose_angletol = fine_tolangle

        print(paramdetector_top)
        
        res = LAA.GetCalibParameter(emax_simul, veryclose_angletol, elem_label,
                                UBmat, paramdetector_top, data, 
                                pixelsize=pixelsize, dim=dim, boolctrl=boolctrl,
                                use_weights=use_weights, addoutput=1)
        
        newparam, UBmat_2, pixdev_2, nfit_2, pixdev_list, spotnum_hkl_2 = res
        
        if elim_worst_pixdev :
            pixdevlist = pixdev_list
            if verbose : print(pixdevlist)
            ind0 = np.where(pixdevlist > maxpixdev)
            print(ind0[0])
            nbad = len(ind0[0])
            if nbad > 0 :
                print("MAXPIXDEV = ", maxpixdev)
                print("found ", nbad, " peaks with pixdev larger than ", maxpixdev)
                ind1 = np.where(pixdevlist < maxpixdev)
                ngood = len(ind1[0])
                if ngood > 10 :
                    #print spotexpnum_spotsimnum_2
                    #print spotnum_hkl_2
                    #print Intensity_list_2
                    print("fit again after removing bad peaks")
                    print(shape(data))
                    data1 = (np.array(data,dtype=float)).transpose()
                    print(data1[0])
                    datanew = zeros(shape(data1),float)
                    print(shape(spotnum_hkl_2))
                    n_hkl = np.array(spotnum_hkl_2,dtype = float)
                    print(n_hkl[:,0])
                    numdat_ind = np.array(n_hkl[:,0],dtype=int)
                    numdat_ind_good = numdat_ind[ind1[0]]
                    datanew[numdat_ind_good] = data1[numdat_ind_good]*1.0
                    datanew_tuple = (datanew[:,0],datanew[:,1],datanew[:,2],datanew[:,3],datanew[:,4])

                    res = LAA.GetCalibParameter(emax_simul, veryclose_angletol, elem_label, \
                        UBmat, paramdetector_top, datanew_tuple, \
                        pixelsize=pixelsize, dim=dim, boolctrl=boolctrl, \
                        use_weights=use_weights, addoutput=1)
                    
                    if res is not None :
                        newparam, UBmat_2, pixdev_2, nfit_2, pixdev_list, spotnum_hkl_2 = res
                        print("nfit_2 = ", nfit_2)
                    else :
                        print("not enough peaks with pixdev small enough : ngood = ", ngood)
                        return(0)
        
        
                    # , \
                    # addoutput=1, saturated = data_sat, use_weights = use_weights )

##                print newparam
##                print nfit_2
##                print pixdevlist

##        print "UBmat_2 \n", UBmat_2
##        print "deviatoricstrain_2 \n", deviatoricstrain_2
##        print "RR_2 \n", RR_2
##        print "spotnum_i_pixdev_2 \n", spotnum_i_pixdev_2
##        print "pixdev_2 \n" , pixdev_2 
##        print "nfit_2 \n", nfit_2
##        print "spotnum_hkl_2 \n", spotnum_hkl_2
                    
    print(np.shape(spotnum_hkl_2))

    spotnum_hkl = np.array(spotnum_hkl_2, dtype=int)

    if verbose : 
        print("indexed peak list :")
        print("npeak xexp yexp H K L intensity chi theta ")
    for i in range(nfit_2):
        npeak = spotnum_hkl[i, 0]
        #print "npeak = ", npeak
        hkl = spotnum_hkl[i, 1:]
        if verbose : print(npeak, data_pixX[npeak], data_pixY[npeak], np.array(hkl, dtype=int), data_I[npeak], data_chi[npeak], data_theta[npeak])

    numlist = spotnum_hkl[:, 0]
    xyind = np.column_stack((data_pixX[numlist], data_pixY[numlist]))
    #print xyind

    if min_matLT == True :
            print("matrix with lowest Miller angles", end=' ') 
            matLTmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(UBmat_2)
            print("matLTmin \n", matLTmin.round(decimals=6))
            print("original UB matrix \n", UBmat_2.round(decimals=6))
            print("transfmat \n", list(transfmat))
            # transformer aussi les HKL pour qu'ils soient coherents avec matmin
            # return transfmat dans find_lowest_Euler_Angles_matrix
            hkl = spotnum_hkl[:,1:]
            hklmin = dot(transfmat,hkl.transpose()).transpose()
            
    elif stereo_mat == True : 
            MG.init_numbers_for_crystal_opsym_and_first_stereo_sector(elem_label = elem_label)
            if axis_pole_sample  is  not None :
                    matstarlab = F2TC.matstarlabLaueTools_to_matstarlabOR(UBmat_2)
                    matstarlabnew, transfmat, rgb_pole = matstarlab_to_orientation_color_rgb(matstarlab, axis_pole_sample, elem_label = elem_label)
                    matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlabnew)
                    hkl = spotnum_hkl[:,1:]
                    hklmin = dot(transfmat,hkl.transpose()).transpose()
            else :
                    print("need to set axis_pole_sample when using stereo_mat = True")
    else :
            hklmin = spotnum_hkl[:,1:]
            matLTmin = UBmat_2*1.
            

    matLT_UB = matLTmin
    matLT_UBB0 = dot(matLT_UB, Bmatrix)
    
    epsp1, dlatsrdeg1 = matstarlab_to_deviatoric_strain_crystal(GT.mat3x3_to_matline(matLT_UBB0), 
                                            version = 2, 
                                            elem_label = elem_label)
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)")
    print(epsp1.round(decimals=2))
    deviatoricstrain = GT.epsline_to_epsmat(epsp1).round(decimals=2)
    print("deviatoric strain matrix \n", deviatoricstrain)

    print("pixdev mean = " , pixdev_2)
    print("use weights = ", use_weights)
        
    intensity_list = data_I[numlist]
    
    if paramtofit == "strain" :
        pixdev_list = np.array(spotnum_i_pixdev_2)[:, 2]

    # calcul Etheor : structure quelconque

    Etheor = zeros(nfit_2,float)
    dlatu_angstroms_deg = DictLT.dict_Materials[elem_label][1]
    dlatu_nm_rad = deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
    print("dlatu_nm_rad = ", dlatu_nm_rad)
        
    matstarlab = F2TC.matstarlabLaueTools_to_matstarlabOR(matLT_UBB0)
    matwithlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu_nm_rad)
    
#    mat = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu)
    for i in range(nfit_2):
        
        Etheor_eV, ththeor_deg, uqlab = mat_and_hkl_to_Etheor_ththeor_uqlab(matwithlatpar_inv_nm = matwithlatpar_inv_nm,
                                                                            hkl = hklmin[i,:])
        Etheor[i] = Etheor_eV                                                                   
                                                                             
    print(shape(numlist))
    print(shape(xyind))
    print(shape(hklmin))
        
    data_fit = column_stack((numlist, intensity_list, hklmin[:, 0], hklmin[:, 1], hklmin[:, 2] , pixdev_list, xyind[:, 0], xyind[:, 1], Etheor))

    data_fit_sorted = sort_peaks_decreasing_int(data_fit, 1)

    filecor = filedat
    
    #UBmat = matLT_UBB0

    if use_weights == False :
        filesuffix = "_UWN"
    else :
        filesuffix = "_UWY"
        
    if paramtofit == "strain" :
        filefit = save_fit_results(filecor, data_fit_sorted , matLT_UBB0, deviatoricstrain, filesuffix,\
                                   pixdev_2, grainnum, paramdetector_top, CCDlabel, elem_label = elem_label,
                                   matLT_UB = matLT_UB, Bmatrix = Bmatrix)
    elif paramtofit == "calib" :
        filefit = save_fit_results(filecor, data_fit_sorted , matLT_UBB0, deviatoricstrain, filesuffix,\
                                   pixdev_2, grainnum, newparam,  CCDlabel, elem_label = elem_label,
                                   matLT_UB = matLT_UB, Bmatrix = Bmatrix)
        #filedet = save_det_results(filedat, filecor, matLTmin, filesuffix, newparam, elem_label)
        #filecal = save_cal_results(filecor, numlist, hkl , intensity_list, pixdev_list , matLTmin, filesuffix, pixdev_2,newparam)

    PLOTRESULTS = 0
    
    if PLOTRESULTS :
        xyind = data_cor[numlist, 2:4]
        xyall = data_cor[:, 2:4]
        print(shape(xyind))
        #print xy_list
        p.figure(figsize=(8, 8))
        p.plot(xyall[:, 0], xyall[:, 1], 'bo', markersize=8)
        p.plot(xyind[:, 0], xyind[:, 1], 'rx', label='LT', markersize=12, markeredgewidth=2)
        p.xlim(0, 2048)
        p.ylim(2048, 0)

    if paramtofit == "strain" :
        return(filefit, filecor, nfit_2, pixdev_2, matLT_UBB0)
    elif paramtofit == "calib" :
        return(filefit, filecor, nfit_2, pixdev_2, matLT_UBB0, newparam)
        #return(filedet, filecor, nfit_2, pixdev_2, matLTmin)

def imgnum_to_str(imgnum, number_of_digits_in_image_name):
                      
    if number_of_digits_in_image_name  is None :
        str1 = str(imgnum)
    else :
        encodingdigits = '%%0%dd' % number_of_digits_in_image_name
        str1 =  encodingdigits % imgnum
    return(str1)

def save_fit_results(filename, 
                     data_fit , 
                     matLT_UBB03x3, 
                     deviatoricstrain, 
                     filesuffix, 
                     pixdev, 
                     grainnum, 
                     calib, 
                     CCDlabel,
                     filepathout = None,
                     fileprefix = None,
                     imgnum = None, 
                     deviatoricstrainsample = None,
                     elem_label = "Ge", # only for deviatoricstrainsample = None,
                     matLT_UB = eye(3),
                     Bmatrix = eye(3)   
                     ):
                         
    pixelsize = DictLT.dict_CCD[CCDlabel][1]
    dim = DictLT.dict_CCD[CCDlabel][0]
    
    if filepathout  is None :
        extension = filename.split('.')[-1]
        extension = '.' + extension
        #print extension
        outputfilename = filename.split(extension)[0] + filesuffix + "_" + str(grainnum) + '.fit'
    else :
        outputfilename = filepathout + fileprefix + imgnum_to_str(imgnum,PAR.number_of_digits_in_image_name)+ filesuffix + "_" + str(grainnum) + '.fit'
        filename = filename + "1"
        
    print(outputfilename)
    
    datatooutput = data_fit.round(decimals=4)
    euler = calc_Euler_angles(matLT_UBB03x3).round(decimals=3)
    
    if deviatoricstrainsample  is None :
        matstarlab = F2TC.matstarlabLaueTools_to_matstarlabOR(matLT_UBB03x3)       
        epsp_sample, epsp_crystal = matstarlab_to_deviatoric_strain_sample(matstarlab, 
                                                        omega0 = PAR.omega_sample_frame, 
                                                        version = 2,
                                                        returnmore = True,
                                                        elem_label = elem_label)
        deviatoricstrainsample = GT.epsline_to_epsmat(epsp_sample).round(decimals=2)

    header = '# Strain and Orientation Refinement of: %s\n' % (filename)
    header += '# File created at %s with multigrain.py \n' % (asctime()) 
    header += '# Number of indexed spots : %s \n'%str(shape(data_fit)[0])
    header += '# Mean Deviation(pixel): : %s \n' %str(round(pixdev, 4))
    header += '# pixdev : %s \n' %str(round(pixdev, 4))
    
    if shape(datatooutput)[1] == 9 :
        header += 'spot# Intensity h k l  pixDev xexp yexp Etheor\n'
    elif shape(datatooutput)[1] == 17 :
        header += 'spot# Intensity h k l  pixDev xexp yexp Etheor htwin ktwin ltwin is_from_crystal_5col \n'
    outputfile = open(outputfilename, 'w')
    outputfile.write(header)
    np.savetxt(outputfile, datatooutput, fmt='%.4f')    
    outputfile.write("#UB matrix in q= (UB) B0 G*\n")
    outputfile.write(str(matLT_UB) + '\n')
    outputfile.write("#B0 matrix in q= UB (B0) G*\n")
    outputfile.write(str(Bmatrix) + '\n')
    outputfile.write('#UBB0 matrix in q= (UB B0) G*\n')
    outputfile.write(str(matLT_UBB03x3) + '\n')
    outputfile.write('#deviatoric strain crystal (1e-3 units)\n')
    outputfile.write(str(deviatoricstrain) + '\n') 
    outputfile.write('#deviatoric strain sample (1e-3 units)\n')
    outputfile.write(str(deviatoricstrainsample) + '\n')
    text = "#omega_sample_frame \n"
    text += str(round(PAR.omega_sample_frame, 2)) + '\n'
    outputfile.write(text)        
    outputfile.write('#Euler angles phi theta psi (deg)\n')
    outputfile.write(str(euler) + '\n')
    outputfile.write('#grain number\n')
    outputfile.write(str(grainnum) + '\n')
    text = "#Number of indexed spots \n"
    text += str(shape(data_fit)[0]) + "\n"
    outputfile.write(text)
    text = "#pixdev\n"
    text += str(round(pixdev, 4)) + '\n'
    outputfile.write(text)
    # MAR
    # pixelsize = 165. / 2048
    # dim = (2048.0, 2048.0)
    # VHR
##    pixelsize = 0.031
##    dim = (2594.0, 3764.0)
    text = "#Sample-Detector distance (IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2\n"    
    dd, xcen, ycen, xbet, xgam = calib[0], calib[1], calib[2], calib[3], calib[4]
    text += "%.3f, %.2f, %.2f, %.3f, %.3f, %.5f, %.0f, %.0f\n" % (round(dd, 3), round(xcen, 2), round(ycen, 2), \
                        round(xbet, 3), round(xgam, 3), pixelsize, round(dim[0], 0), round(dim[1], 0))
    outputfile.write(text)
 
    text = "# Notes on reference frames" + '\n'
    text += "# UBB0 : astar, bstar, cstar as columns on Rlab_LT " + '\n'
    text += "# norme(astar) = norme(first column of B0)" + '\n'
    text += "# Rlab_LT :" + '\n'
    text += "# xlab along incident beam, " + '\n'
    text += "# ylab along cross(zcam,xlab)" + '\n'
    text += "# zcam along inner normal to camera screen (upwards)" + '\n'
    text += "# B0 : Bstar matrix of undeformed reciprocal crystal unit cell" + '\n'
    text += "# (direct unit cell is defined in DictLaueTools.py)" + '\n'
    text += "# B0 = astar, bstar, cstar as columns on astarOND bstarOND cstarOND" + '\n'
    text += "# astarOND, bstarOND, cstarOND = cartesian frame attached to reciprocal crystal unit cell" + '\n'
    text += "# OND  = orthonorme direct" + '\n'
    text += "# deviatoric strain crystal : strain in cartesian frame attached to direct crystal unit cell" + '\n'
    text += "# deviatoric strain sample : strain in Rsample_MG frame" + '\n'
    text += "# Rlab_MG :" + '\n' 
    text += "# ylab along incident beam," + '\n' 
    text += "# xlab along cross(ylab, zcam)" + '\n'
    text += "# Rsample_MG :" + '\n'
    text += "# almost aligned with -xech -yech zech sample scanning linear stages for BM32ESRF" + '\n'
    text += "# zech along outer normal to sample surface" + '\n'
    text += "# deduced from Rlab_MG by positive rotation around xlab" + '\n'
    text += "# with an angle omega_sample_frame in degrees" + '\n' 
    
    outputfile.write(text)    

    outputfile.close()

    return(outputfilename)


def save_det_results(filedat, filecor, matLTmin, filesuffix, newparam, elem_label) :

#    pixelsize = 165. / 2048
    pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]
    dim = (2048.0, 2048.0)

    outputfilename = filecor.split('.')[0] + filesuffix + '.det'

    m11, m12, m13, m21, m22, m23, m31, m32, m33 = np.ravel(matLTmin).round(decimals=7)

    dd, xcen, ycen, xbet, xgam = newparam[0], newparam[1], newparam[2], newparam[3], newparam[4]

    text = "%.3f, %.2f, %.2f, %.3f, %.3f, %.5f, %.0f, %.0f\n" % (round(dd, 3), round(xcen, 2), round(ycen, 2), \
                        round(xbet, 3), round(xgam, 3), pixelsize, round(dim[0], 0), round(dim[1], 0))
    text += "Sample-Detector distance (IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2\n"
    text += "Calibration done with %s at %s with batchOR.py\n" % (elem_label, asctime())
    text += "Experimental Data file: %s\n" % filedat
    text += "Orientation Matrix:\n"
    text += "[[%.7f,%.7f,%.7f],[%.7f,%.7f,%.7f],[%.7f,%.7f,%.7f]]" % (m11, m12, m13, m21, m22, m23, m31, m32, m33)
    outputfile = open(outputfilename, 'w')
    outputfile.write(text)
    outputfile.close()  
                            
def speed_index_refine(filedet, filepath, fileprefix, filesuffix, indimg, \
               use_weights=False, proposed_matrix=None, paramtofit="strain", elem_label="UO2"):

    numfiles = len(indimg)
    nlist = zeros(numfiles, dtype=int)
    pixdevlist = zeros(numfiles, dtype=float)

    paramdetector_top, matstarlab = F2TC.readlt_det(filedet)

    i = 0
    mat_ref = proposed_matrix
    
    for kk in indimg :
        
        peakfile = filepath + fileprefix + imgnum_to_str(kk, 4) + filesuffix
        
        print(peakfile)
        
        filefit, filecor, nlist[i], pixdevlist[i], matLTmin = test_index_refine(peakfile,
                                                                                list(paramdetector_top),
                                                                                use_weights=use_weights,
                                                                                proposed_matrix=mat_ref,
                                                                                paramtofit=paramtofit,
                                                                                elem_label=elem_label)

        mat_ref = matLTmin
        
        if paramtofit == "strain" :
            listfile = [filefit, filecor, filedet]
            mergefiles(listfile)

        i = i + 1

    print(nlist)
    print(pixdevlist.round(decimals=4))

    return(nlist, pixdevlist)

def sort_peaks_decreasing_int(data_str, colnum):
                              
    print("tri des pic par intensite decroissante")

    npics = shape(data_str)[0]
    print(npics)
    index2 = zeros(npics, int)
    
    index1 = argsort(data_str[:, colnum])
    for i in range(npics):
        index2[i] = index1[npics - i - 1]
    #print "index2 =", index2
    data_str2 = data_str[index2]

    return(data_str2)


##    /* phi = GRAINIMAGELIST[NGRAINIMAGE].EULER[0] / RAD;
##    theta = GRAINIMAGELIST[NGRAINIMAGE].EULER[1]/ RAD;
##    psi = GRAINIMAGELIST[NGRAINIMAGE].EULER[2]/ RAD;
##    b[0][0] = np.cos(psi)*np.cos(phi)- np.cos(theta)*np.sin(phi)*np.sin(psi);
##    b[0][1] = np.cos(psi)*np.sin(phi)+ np.cos(theta)*np.cos(phi)*np.sin(psi);
##    b[0][2] = np.sin(psi)*np.sin(theta);
##    b[1][0] = -np.sin(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.cos(psi);
##    b[1][1] = -np.sin(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.cos(psi);
##    b[1][2] = np.cos(psi)*np.sin(theta);
##    b[2][0] = np.sin(theta)*np.sin(phi);
##    b[2][1] = -np.sin(theta)*np.cos(phi);
##    b[2][2] = np.cos(theta);

def calc_Euler_angles(mat3x3):
        
        # mat3x3 = matrix "minimized" in LT lab frame
        # see FindO.find_lowest_Euler_Angles_matrix
        # phi 0, theta 1, psi 2

    mat = GT.matline_to_mat3x3(matstarlab_to_matstarlabOND(GT.mat3x3_to_matline(mat3x3)))
    
    RAD = 180.0 / PI
    
    euler = zeros(3, float)
    euler[1] = RAD * acos(mat[2, 2])
    
    if (abs(abs(mat[2, 2]) - 1.0) < 1e-5):
        # if theta is zero, phi+psi is defined, if theta is pi, phi-psi is defined */
        # put psi = 0 and calculate phi */
        # psi */
        euler[2] = 0.0
        # phi */
        euler[0] = RAD * acos(mat[0, 0]) 
        if (mat[0, 1] < 0.0) :
                        euler[0] = -euler[0]
    else :
        # psi */ 
        toto = np.sqrt(1 - mat[2, 2] * mat[2, 2]) # sin theta - >0 */
        euler[2] = RAD * acos(mat[1, 2] / toto)
        # phi */    
        euler[0] = RAD * acos(-mat[2, 1] / toto)
        if (mat[2, 0] < 0.0) :
            euler[0] = 360.0 - euler[0]
                        
        #print "Euler angles phi theta psi (deg)"
        #print euler.round(decimals = 3)

    return(euler)

def calc_Euler_angles_Emeric(mat3x3):
       
        # mat3x3 = matrix "minimized" in LT lab frame
        # see FindO.find_lowest_Euler_Angles_matrix
        # phi 0, theta 1, psi 2
        
#    mail d'Emeric du 23Oct12
# 
#Salut,
#
#Voici un document issu de la bibliotheque de Romain Quey qui utilise les memes equations que dans la fonction calc_Euler_angles (). Cependant la matrice " g " de sa bibliotheque correspond a la transposee de matLT ou mat3x3 fournie en argument
#{ i.e.   g <=> (matLT)T <=> [a* b* c*]}.
#
#Je propose donc dans calc_Euler_angles() de rajouter :
#
#    mat=mat.transpose()  
#
#et pour une question de signe non reglee avec phi2 :
#
#    if (mat[0, 2] < 0.0) :
#    euler[2] = 360.0 - euler[2]    
#
#
#Voila voila,
#Emeric

# ici je vais calculer les angles d'Euler en partant de la matrice "sample"
# et pas de la matrice "labo"
    

    mat = GT.matline_to_mat3x3(matstarlab_to_matstarlabOND(GT.mat3x3_to_matline(mat3x3)))
   
    #modif EMeric 21 octobre 2013
    mat=mat.transpose()   
   
    RAD = 180.0 / PI
   
    euler = zeros(3, float)
    euler[1] = RAD * acos(mat[2, 2])
   
    if (abs(abs(mat[2, 2]) - 1.0) < 1e-5):
        # if theta is zero, phi+psi is defined, if theta is pi, phi-psi is defined */
        # put psi = 0 and calculate phi */
        # psi */
        euler[2] = 0.0
        # phi */
        euler[0] = RAD * acos(mat[0, 0])
        if (mat[0, 1] < 0.0) :
                        euler[0] = -euler[0]
    else :
        # psi */
        toto = np.sqrt(1 - mat[2, 2] * mat[2, 2]) # sin theta - >0 */
        euler[2] = RAD * acos(mat[1, 2] / toto)
        # phi */   
        euler[0] = RAD * acos(-mat[2, 1] / toto)
        if (mat[2, 0] < 0.0) :
            euler[0] = 360.0 - euler[0]
        if (mat[0, 2] < 0.0) :
            euler[2] = 360.0 - euler[2]              
        #print "Euler angles phi theta psi (deg)"
        #print euler.round(decimals = 3)

    return(euler)
    
def calc_matrix_from_Euler_angles_EBSD(euler3_rad) :
    
    # d'apres doc Romain Quey p 11
    # notation de Bunge
    # Column 1-3: phi1, PHI, phi2 (orientation of point in radians)
    phi1 = euler3_rad[0]
    PHI = euler3_rad[1]
    phi2 = euler3_rad[2]
    
    g11 = np.cos(phi1)*np.cos(phi2) - np.sin(phi1)*np.sin(phi2)*np.cos(PHI)
    g12 = np.sin(phi1)*np.cos(phi2) + np.cos(phi1)*np.sin(phi2)*np.cos(PHI)
    g13 = np.sin(phi2)*np.sin(PHI)
    g21 = -np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(PHI)
    g22 = -np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(PHI)
    g23 = np.cos(phi2)*np.sin(PHI)
    g31 = np.sin(phi1)*np.sin(PHI)
    g32 = -np.cos(phi1)*np.sin(PHI)
    g33 = np.cos(PHI)

    # g = abc en ligne sur xyz
    
    g_matrix = np.array([[g11,g12,g13],[g21,g22,g23],[g31,g32,g33]])
    
    # matrice OND donc abc alignes avec abcstar
    
    matstarsampleOND3x3 = g_matrix.transpose()
    
    # new 27Jan14 : minimisation angles d'Euler dans repere labo et pas repere sample
    matmin3x3, transfmat = FindO.find_lowest_Euler_Angles_matrix(GT.matline_to_mat3x3(matstarsample3x3_to_matstarlab(matstarsampleOND3x3)))

    matstarlab = GT.mat3x3_to_matline(matmin3x3)
#    matmin3x3, transfmat = FindO.find_lowest_Euler_Angles_matrix(matstarsampleOND3x3)  # removed 27Jan14
    
#    matstarlab =  matstarsample3x3_to_matstarlab(matmin3x3, omega = PAR.omega_sample_frame) # removed 27Jan14

    return(matstarlab)

#def calc_matrix_from_Euler_angles_EBSD_v2(euler3_rad, 
#                                                  elem_label = "m_zirconia",
#                                                  verbose = 1,
#                                                  Euler_angles_give_Udir_or_Ustar = "Ustar") :
#    
#    # d'apres doc Romain Quey p 11
#    # notation de Bunge
#    # Column 1-3: phi1, PHI, phi2 (orientation of point in radians)
#    phi1 = euler3_rad[0]
#    PHI = euler3_rad[1]
#    phi2 = euler3_rad[2]
#    
#    g11 = np.cos(phi1)*np.cos(phi2) - np.sin(phi1)*np.sin(phi2)*np.cos(PHI)
#    g12 = np.sin(phi1)*np.cos(phi2) + np.cos(phi1)*np.sin(phi2)*np.cos(PHI)
#    g13 = np.sin(phi2)*np.sin(PHI)
#    g21 = -np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(PHI)
#    g22 = -np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(PHI)
#    g23 = np.cos(phi2)*np.sin(PHI)
#    g31 = np.sin(phi1)*np.sin(PHI)
#    g32 = -np.cos(phi1)*np.sin(PHI)
#    g33 = np.cos(PHI)
#
#    # g = abc en ligne sur xyz, matrice orthonormee
#    
#    g_matrix = np.array([[g11,g12,g13],[g21,g22,g23],[g31,g32,g33]])
#    
#    dlatdeg = np.array(DictLT.dict_Materials[elem_label][1], dtype = float)
#    if verbose : print "direct lattice parameters angstroms degrees = \n ", dlatdeg    
#    
#    dlatrad = deg_to_rad(dlatdeg)
#    
#    if Euler_angles_give_Udir_or_Ustar == "Udir" :
#        
#        if verbose : print "Euler_angles_give_Udir_or_Ustar = Udir"
#        
#        Udirsample = g_matrix.transpose()
#        
#        if verbose : print "Udirsample, abc as columns on xyzsample = \n", Udirsample
#        
#    #    print DictLT.dict_Materials[elem_label][1]
#    
#        Bdir = rlat_to_Bstar(dlatrad)
#    
#        if verbose : print "Bdir = \n", Bdir    
#        
#        matdirsample3x3 = dot(Udirsample, Bdir)
#        
#        if verbose :  print "matdirsample3x3 = \n ", matdirsample3x3
#        
#        matdirsample3x3_new = matdirsample3x3/norme(matdirsample3x3[:,0])
#        
#        if verbose :  print "matdirsample3x3_new = \n ", matdirsample3x3_new
#        
#        matdirsample = GT.mat3x3_to_matline(matdirsample3x3_new)
#        
#        if verbose : print "matdirsample = \n", matdirsample
#    
#        matstarsample3x3, dummy = matstarlab_to_matdirlab3x3(matdirsample)
#    
##        if verbose : print "matstarsample3x3 = \n ", matstarsample3x3
#        
#    elif Euler_angles_give_Udir_or_Ustar == "Ustar" :   
#        
#        if verbose : print "Euler_angles_give_Udir_or_Ustar = Ustar"
#       
#        Ustarsample = g_matrix.transpose()
#        
#        if verbose : print "Ustarsample, astar bstar c as columns on xyzsample = \n", Ustarsample
#
#        Bstar = dlat_to_Bstar(dlatrad)
#    
#        if verbose : print "Bstar = \n", Bstar    
#        
#        matstarsample3x3 = dot(Ustarsample, Bstar)
#        
##        if verbose :  print "matstarsample3x3 = \n ", matstarsample3x3
#                
#    elif Euler_angles_give_Udir_or_Ustar == "Ustar_rotated_180_around_c" :   
#        
#        if verbose : print "Euler_angles_give_Udir_or_Ustar = Ustar_rotated_180_around_c"
#       
#        toto = g_matrix.transpose()
#        
#        Ustarsample = np.zeros((3,3), float)
#        
#        Ustarsample[:,0] = - toto[:,0]
#        Ustarsample[:,1] = - toto[:,1]
#        Ustarsample[:,2] = toto[:,2]       
#        
#        if verbose : print "Ustarsample, astar bstar c as columns on xyzsample = \n", Ustarsample
#
#        Bstar = dlat_to_Bstar(dlatrad)
#    
#        if verbose : print "Bstar = \n", Bstar    
#        
#        matstarsample3x3 = dot(Ustarsample, Bstar)
#        
##        if verbose :  print "matstarsample3x3 = \n ", matstarsample3x3   
#        
#    elif Euler_angles_give_Udir_or_Ustar == "Ustar_rotated_m60_around_c" :   
#        
#        # en chantier 09Feb17
#        
#        if verbose : print "Euler_angles_give_Udir_or_Ustar = Ustar_rotated_m60_around_c"
#       
#        toto = g_matrix.transpose()
#
#        vlab = np.array([0.,0.,1.])
#        ang1 = -60.
#        
#        matrot = from_axis_vecangle_to_mat(vlab, ang1)
#        
#        Ustarsample = dot(toto, matrot)        
#        
#        if verbose : print "Ustarsample, astar bstar c as columns on xyzsample = \n", Ustarsample
#
#        Bstar = dlat_to_Bstar(dlatrad)
#    
#        if verbose : print "Bstar = \n", Bstar    
#        
#        matstarsample3x3 = dot(Ustarsample, Bstar)
#        
#    if verbose :  print "matstarsample3x3 = \n ", matstarsample3x3   
#    
#    if verbose : print "omega = ",  PAR.omega_sample_frame   
#    
#    matstarlab =  matstarsample3x3_to_matstarlab(matstarsample3x3, omega = PAR.omega_sample_frame)
#
#    if verbose : print "matstarlab = \n", matstarlab
#    
#    matstarlab = matstarlab / norme(matstarlab[0:3])
#    
#    if verbose : print "matstarlab (norme(astar)=1) = \n", matstarlab  
#    
#    klmsdq
#    
##    jklaze
#
##    matmin3x3, transfmat = FindO.find_lowest_Euler_Angles_matrix(matstarsampleOND3x3)  # removed 27Jan14
#    
##    matstarlab =  matstarsample3x3_to_matstarlab(matmin3x3, omega = PAR.omega_sample_frame) # removed 27Jan14
#
#    return(matstarlab)
#


def merge_fit_files_multigrain(filelist, removefiles=1):
        
#    if len(filelist) == 0 :
#        filedum = "toto.fit"
#        dat1 = zeros(6, float)
#        savetxt(filedum, dat1)
#        return(filedum)
        
    filefitallgrains = filelist[0].split('_1.')[0] + '_mg.fit'
    print("merged fit file : ", filefitallgrains)
    outputfile = open(filefitallgrains, 'w')
    for filename in filelist :    
        outputfile.write(filename + '\n')
        f = open(filename, 'r')
        try:
            for line in f:
                outputfile.write(line)        
        finally:
            f.close()
        outputfile.write('\n')
    outputfile.close()
    if removefiles :
                for filename in filelist :
                        os.remove(filename)
                        
    return(filefitallgrains)

def convert_xmas_str_to_LT_fit(filestr, 
                               elem_label = "W",
                               stiffness_c_tensor = None, 
                               schmid_tensors = None,
                               CCDlabel = PAR.CCDlabel,
                               recalculate_strain_from_matrix = 0,
                               min_matLT = False):
    
    # attention si min_matLT = True et recalculate_strain_from_matrix = 0
    # mettre un warning    
    
    #29May12    
    print("convert XMAS .str file into LaueTools .fit file (multigrain)")
    print("warning : compatibility OK with XMAS 2006 version, not OK for later XMAS versions")

    numcolint = 9  # 10  # intens = column used by XMAS for intensity sorting in peak search
    #min_matLT = True
    
    
    #data_str_all :  xy(exp) 0:2 hkl 2:5 xydev 5:7 energy 7 dspacing 8  intens 9 integr 10"
    data_str_all, matstarlab, calib, dev_str, npeaks, strain, strain2 = \
            read_all_grains_str(filestr, min_matLT=min_matLT)
    
    indstart = 0
    
    print("sorting whole spot list by intensities")

    npics = shape(data_str_all)[0]
    toto = np.arange(npics)
    toto1 = np.arange(npics, 0, -1)
    data_str_all2 = column_stack((data_str_all, toto1))
    data_str_all2_sorted = sort_peaks_decreasing_int(data_str_all2, numcolint)
    index1 = np.array(data_str_all2_sorted[:, -1], dtype=int)
    #print index1
    data_str_all3 = column_stack((data_str_all2_sorted, toto))
    data_str_all3_unsorted = sort_peaks_decreasing_int(data_str_all3, -2)
    #data_str_all3_unsorted = data_str_all3[index1,:]
    index2 = np.array(data_str_all3_unsorted[:, -1], dtype=int)
    
    #print data_str_all[:,numcolint]
    #print data_str_all3_unsorted[:,numcolint]
    #print index2
    #print data_str_all2_sorted[:,numcolint]
    
    pixdev_list = zeros(npics, float)
    Etheor_list = zeros(npics, float)
    for i in range(npics):
        pixdev_list[i] = np.sqrt(data_str_all[i, 5] * data_str_all[i, 5] + data_str_all[i, 6] * data_str_all[i, 6])

    data_str_all = column_stack((data_str_all, index2, pixdev_list, Etheor_list))
    #data_str_all : index1 11, pixdev 12
    #print data_str_all
    filelist = []

    for i in range(len(npeaks)):
        print("grain number : ", i)
        if npeaks[i]> 0 :           
            filecor = filestr
            filesuffix = "_from_xmas"
            indend = indstart + npeaks[i]
            toto = data_str_all[indstart:indend, :]
            #data_fit = vstack((toto[:,11],toto[:,10],toto[:,2],toto[:,3],toto[:,4],toto[:,12],toto[:,0],toto[:,1])).transpose()
            data_fit = column_stack((toto[:, 11], toto[:, numcolint], toto[:, 2:5], toto[:, 12], toto[:, :2], toto[:,-1]))
            #data_fit = vstack((numlist, intensity_list, hklmin[:,0], hklmin[:,1], hklmin[:,2] ,
                                          # pixdev_list, xyind[:,0],xyind[:,1])).transpose()
            data_fit_sorted = sort_peaks_decreasing_int(data_fit, 1)
            matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab[i, :])
            #GT.mat3x3_to_matline(matLTmin)
            
            if recalculate_strain_from_matrix :
                filesuffix = filesuffix + "_recalc_strain"
                epsp_sample, epsp_crystal = matstarlab_to_deviatoric_strain_sample(matstarlab[i, :], 
                                                                    omega0 = PAR.omega_sample_frame, 
                                                                    version = 2,
                                                                    returnmore = True,
                                                                    elem_label = elem_label)
            else :
                if min_matLT == True :                    
                    print("warning : strain in crystal frame will be wrong")
                    print("min_matLT == True and recalculate_strain_from_matrix = 0")
                    exit()
                else :
                    epsp_crystal = np.array(strain[i,:], dtype=float)
                    epsp_sample = np.array(strain2[i,:], dtype=float)
                    
            print("deviatoric strain crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)")
            print(epsp_crystal.round(decimals=2))
                                                                    
            print("deviatoric strain sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-3 units)")
            print(epsp_sample.round(decimals=2))    
                    
            if stiffness_c_tensor  is  not None :
                
                sigma_crystal = deviatoric_strain_crystal_to_stress_crystal(c_tensor, epsp_crystal)
                sigma_sample = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab[i, :],
                                                                      sigma_crystal,
                                                                      omega0 = PAR.omega_sample_frame)
                
                print("pseudo-deviatoric stress crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (100 MPa units)")
                print(sigma_crystal.round(decimals=2))
                                                                        
                print("pseudo-deviatoric stress sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (100 MPa units)")
                print(sigma_sample.round(decimals=2))   
                
                von_mises = deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal)
                print("pseudo Von Mises equivalent Stress (100 MPa units)", round(von_mises,3))
                
                if schmid_tensors  is  not None :
                    tau1 = deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(sigma_crystal, schmid_tensors)
                    print("RSS resolved shear stresses on glide planes (100 MPa units) : ")
                    print(tau1.round(decimals = 3))
                    print("Max RSS : ", round(abs(tau1).max(),3))
            
            deviatoricstrain = GT.epsline_to_epsmat(epsp_crystal).round(decimals=2)  
            deviatoricstrainsample = GT.epsline_to_epsmat(epsp_sample).round(decimals=2) 
            # here the recalculated eps_yz and eps_xy are opposite to those of XMAS
            # while all the epsilon_crystal components are equal to those of XMAS is min_matLT = False
            # y_sample_XMAS = - y_sample_OR ? (i.e. xyz_sample_XMAS est un triedre indirect ?)
            
            filefit = save_fit_results(filecor, data_fit_sorted, matLTmin, deviatoricstrain, filesuffix, dev_str[i, 2], i + 1, calib, CCDlabel, deviatoricstrainsample = deviatoricstrainsample)
            filelist.append(filefit)
            indstart = indstart + npeaks[i]
            print("indstart =", indstart)
        else :
            print("bad grain")
           
    if i > 1 :    
        merge_fit_files_multigrain(filelist)
    return(0)


def find_common_peaks(xy1, 
                      xy2, 
                      dxytol=0.01, 
                      verbose=1, 
                      returnmore = 0, 
                      only_one_pair_per_peak = "no"):


    #print "look for peaks common to two grains"
    ndat1 = shape(xy1)[0]
    ndat2 = shape(xy2)[0]
    if verbose : print("ndat1, ndat2 = ", ndat1, ndat2)
    
    if len(shape(xy1)) > 1 :
        ncols = shape(xy1)[1]
    else : ncols = 1

    #print xy1

    iscommon1 = zeros(ndat1, int)
    iscommon2 = zeros(ndat2, int)
    
    if returnmore == 1 :
        ndat = min(ndat1, ndat2)
        list_pairs = zeros((ndat,2),int)

    nb_common_peaks = 0
    #print "common peaks : " 
    for j in range(ndat1):
        #print "j= ", j
#        print xy1[j]
#        print xy2
        dxyall = xy1[j]-xy2
#        print dxyall
        dxyall2 = np.multiply(dxyall, dxyall)
#        print dxyall2
        if ncols > 1 :
            dxyall2sum = dxyall2.sum(axis=1)
        else :
            dxyall2sum = dxyall2
#        print dxyall2sum
#        print shape(dxyall2sum)
#        print shape(xy2)
        dxyallnorme = np.sqrt(dxyall2sum)
#        print dxyallnorme
#        print shape(dxyallnorme)

        ind0 = where(dxyallnorme < dxytol)
#        print ind0[0]
        ncom = len(ind0[0])
        if ncom > 0 :
            if only_one_pair_per_peak == "no" :
                nb_common_peaks = nb_common_peaks + ncom
                iscommon1[j]=iscommon1[j]+ ncom
                iscommon2[ind0[0]] = iscommon2[ind0[0]]+1
                if returnmore == 1 :
                    list_pairs[nb_common_peaks-ncom : nb_common_peaks, 0] = j
                    list_pairs[nb_common_peaks-ncom : nb_common_peaks, 1] = ind0[0] 
            else :
                if iscommon1[j]== 1 : 
                    print("more than one xy2 matching this xy1 : keep first xy2") 
                    break
                else :
                    if iscommon2[ind0[0][0]] == 1 :
                        print("more than one xy1 matching this xy2 : keep first xy1") 
                        break
                    else :                       
                        nb_common_peaks = nb_common_peaks + 1
                        iscommon1[j]= 1
                        iscommon2[ind0[0][0]] = 1
                        if returnmore == 1 :
                            list_pairs[nb_common_peaks-1, 0] = j
                            list_pairs[nb_common_peaks-1, 1] = ind0[0][0]                     
                            
    if verbose : print("nb_common_peaks =", nb_common_peaks)
    ind1 = where(iscommon1 == 0)
    ind2 = where(iscommon2 == 0)
    if returnmore == 1 :
        list_single1 = ind1[0]
        list_single2 = ind2[0]
        nsingle1 = len(ind1[0])
        nsingle2 = len(ind2[0])
    if verbose :
        print(iscommon1)
        print(iscommon2)
        print(ind1)
        print(ind2)

    if verbose :
        if nb_common_peaks != 0 : #& (nb_common_peaks < max(ndat1,ndat2)):
            print("nb / indexes / xy of non-common peaks : ")
            if shape(ind1)[1] > 0 :    
                print("list 1 :", len(ind1[0]), " : ", ind1[0]) 
                print("xy1 : ", xy1[ind1[0]])
            if shape(ind2)[1] > 0 :
                print("list 2 :", len(ind2[0]), " : ", ind2[0]) 
                print("xy2 : ", xy2[ind2[0]])
            if returnmore == 1 :
                print("list pairs :", nb_common_peaks , " : ") 
                print("index 1" , list_pairs[:nb_common_peaks,0])
                print("index 2" , list_pairs[:nb_common_peaks,0])
                print("xy : ", xy1[list_pairs[:nb_common_peaks,0]])
    
    if returnmore == 0 :    
        return(nb_common_peaks, iscommon1, iscommon2)
    else :
        if nb_common_peaks > 0 :
            # utile de retourner aussi iscommon1 et iscommon2 pour ncom > 1
            return(nb_common_peaks, list_pairs[:nb_common_peaks,:], \
                   nsingle1, nsingle2, list_single1, list_single2, \
                   iscommon1, iscommon2)
        else : return(0)

def readlt_fit_mg(filefitmg, 
                  verbose=1, 
                  readmore = False, # read strain crystal + euler, 
                  first_line_ends_with = ".dat",
                  readmore2 = False # read strain sample
                  ): # 29May13
            
    extension = first_line_ends_with

    if verbose :    
        print("reading multigrain fit file")
        
#        print extension

    print(filefitmg) 
    
    f = open(filefitmg, 'r')
    nbgrains = 0
    linepos_list = []
    i = 0
    try:
        for line in f:
            toto = line.rstrip(PAR.cr_string)
#            print toto
#            print toto[-4:]
            if toto[-4:]== extension :
                nbgrains = nbgrains + 1
                linepos_list.append(i)
            i = i+1
    finally:
        linepos_list.append(i)
        f.close()
        
    if verbose : 
        print("nbgrains = ", nbgrains)
        print("linepos_list = ", linepos_list)
    
    if nbgrains == 0 : return(0)
    
    f = open(filefitmg, 'r')
    
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)
        
    gnumlist = np.arange(nbgrains)
    
    matstarlab = zeros((nbgrains, 9), float)
    strain6 = zeros((nbgrains, 6), float)
    strain6sample = zeros((nbgrains, 6), float)
    calib = zeros((nbgrains, 5), float)
    euler = zeros((nbgrains, 3), float)
    npeaks = zeros(nbgrains, int)
    indstart = zeros(nbgrains, int)
    pixdev = zeros(nbgrains, float)
    
    # lecture .fit pour chaque grain    
    for k in range(nbgrains) : 
        matLT3x3 = np.zeros((3, 3), dtype=np.float)
        strain = np.zeros((3, 3), dtype=np.float)
        strain2 = np.zeros((3, 3), dtype=np.float)
        i = 0
        n = linepos_list[k]
        #print "n = ", n
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        #print f.readline()
        f.seek(line_offset[n+1])
        matrixfound = 0
        calibfound = 0
        pixdevfound = 0
        strainfound = 0
        strainfound2 = 0
        eulerfound = 0
        linecalib = 0
        linepixdev = 0
        linestrain = 0
        linestrain2 = 0
        lineeuler = 0
        list1 = []
        linestartspot = 10000
        lineendspot = 10000    
            
        while (i<(linepos_list[k+1]-linepos_list[k])):
            line = f.readline()
            i = i + 1
            #print i
            if line[:5] == "spot#" :
                linecol = line.rstrip(PAR.cr_string)
                linestartspot = i + 1
            if line[:3] == "#UB" :
                #print line
                matrixfound = 1
                linestartmat = i
                lineendspot = i
                j = 0
                #print "matrix found"
            if line[:3] == "#Sa" :
                #print line
                calibfound = 1
                linecalib = i + 1
            if line[:3] == "#pi" :
                #print line
                pixdevfound = 1
                linepixdev = i + 1
            if line[:3] == "#de" :
                #print line
                if readmore2 :
                    if line[:26] == "#deviatoric strain crystal":
                        strainfound = 1
                        linestrain = i  
                    elif line[:25] == "#deviatoric strain sample":
                        strainfound2 = 1
                        linestrain2 = i
                else :
                    strainfound = 1
                    linestrain = i
                j = 0
            if line[:3] == "#Eu" :
                #print line
                eulerfound = 1
                lineeuler = i + 1
            if matrixfound :
                if i in (linestartmat + 1, linestartmat + 2, linestartmat + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    j = j + 1
            if strainfound :
                if i in (linestrain + 1, linestrain + 2, linestrain + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
                    strain[j, :] = np.array(toto, dtype=float)
                    j = j + 1                    
            if strainfound2 :
                if i in (linestrain2 + 1, linestrain2 + 2, linestrain2 + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
                    strain2[j, :] = np.array(toto, dtype=float)
                    j = j + 1
            if calibfound & (i == linecalib):
                calib[k,:] = np.array(line.split(',')[:5], dtype=float)
                #print "calib = ", calib[k,:]
            if eulerfound & (i == lineeuler):
                euler[k,:] = np.array(line.replace('[', '').replace(']', '').split()[:3], dtype=float)
                #print "euler = ", euler[k,:]
            if pixdevfound & (i == linepixdev):
                pixdev[k] = float(line.rstrip(PAR.cr_string))
                #print "pixdev = ", pixdev[k]
            if (i >= linestartspot) & (i < lineendspot) :
                list1.append(line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split())

        data_fit = np.array(list1, dtype=float)
        npeaks[k]=shape(data_fit)[0]
        
        verbose2 = 0
        if verbose2 :
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
            
        matstarlab[k,:] =F2TC.matstarlabLaueTools_to_matstarlabOR(matLT3x3)

        # xx yy zz yz xz xy
        strain6[k,:] = np.array([strain[0, 0], strain[1, 1], strain[2, 2], strain[1, 2], strain[0, 2], strain[0, 1]])
        strain6sample[k,:] = np.array([strain2[0, 0], strain2[1, 1], strain2[2, 2], strain2[1, 2], strain2[0, 2], strain2[0, 1]])

        if k == 0 :
           data_fit_all = data_fit * 1.0
        else :
           data_fit_all = row_stack((data_fit_all, data_fit))

    f.close()
    
    for k in range(1,nbgrains): indstart[k]=indstart[k-1]+npeaks[k-1]

    #04Dec13
    matstarlab = matstarlab.round(decimals=6)
    data_fit_all = data_fit_all.round(decimals=4)
    calib = calib.round(decimals=3)
    pixdev = pixdev.round(decimals=4)
    strain6 = strain6.round(decimals=2)
    strain6sample = strain6sample.round(decimals=2)    
    euler =  euler.round(decimals=3)
         
    if verbose :
        print("gnumlist = ", gnumlist)
        print("matstarlab = ")        
        print(matstarlab)
        print("npeaks = ", npeaks)
        print("indstart = ", indstart)
        print("pixdev = ", pixdev)
        print("strain6 = \n", strain6)
        if readmore2 :
            print("strain6sample = \n", strain6sample)
        print("euler = \n", euler)
    
    if readmore2 == True :     
        return(gnumlist, npeaks, indstart, matstarlab, data_fit_all, calib, pixdev, strain6, euler, strain6sample)   
    else :        
        if readmore == True :
            return(gnumlist, npeaks, indstart, matstarlab, data_fit_all, calib, pixdev, strain6, euler)   
        else :
            return(gnumlist, npeaks, indstart, matstarlab, data_fit_all, calib, pixdev)
            

def read_any_fitfitfile_multigrain(filefitmg, 
                  verbose=1, 
                  fitfile_type = "MG", # "MGnew"  # "GUI_strain", "FileSeries", "GUI_calib" ,
                  check_Etheor = 0,  # any structure
                  elem_label = PAR.elem_label_index_refine,   # seulement pour check_Etheor = 1
                  check_pixdev = 0 ,
                  pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1], # seulement pour check_pixdev = 1
                  min_matLT = 1,   # seulement pour cubique
                  check_pixdev_JSM = 1
                  ): # 21Oct14,  25Nov14
    
    if verbose : print("min_matLT = ", min_matLT)

    if elem_label  is  not None :
        if not test_if_cubic(elem_label) :  
            min_matLT = 0
            print("min_matLT only works for cubic")
    
    # pour fitfile_type = "GUI"       
    # strain crystal non lu (au cas ou matrice LT non minimale)
    
    # 09Dec14 calcul pixdev pour maille non cubique
    
    filetype = fitfile_type

    spot_columns_MG ="spot# Intensity h k l  pixDev xexp yexp Etheor"
    spot_columns_GUI_strain = "#spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz"

    spot_columns_GUI_old = "#spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev"
    spot_columns_GUI_calib = "#spot_index Itot h k l Xtheo Ytheo Xexp Yexp XdevCalib YdevCalib pixDevCalib 2theta_theo chi_theo Energy PeakAmplitude Imax PeakBkg PeakFwhm1 PeakFwhm2 PeakTilt XdevPeakFit YdevPeakFit"
      
#    print spot_columns_MG.split()
    
    list_data_MG =\
    [ "#UB matrix in reciprocal space",
    "#pixdev",
    "#Sample-Detector distance (IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2",
    "#UB matrix in reciprocal space",
     "#deviatoric strain (1e-3 units)",
     "#Euler angles phi theta psi (deg)"]
    
    #  0 : value at end of text line
    # N > 0 : value on N lines after text line 
    line_codes_MG = [3, 1, 1, 3, 3 ,1]


    list_data_MGnew =\
    ["#UBB0 matrix in q= (UB B0) G*",
    "#pixdev",
    "#Sample-Detector distance (IM), xO, yO, angle1, angle2, pixelsize, dim1, dim2",
    "#UB matrix in q= (UB) B0 G*",
     "#deviatoric strain crystal (1e-3 units)",
     "#Euler angles phi theta psi (deg)"]
    
    #  0 : value at end of text line
    # N > 0 : value on N lines after text line 
    line_codes_MGnew = [3, 1, 1, 3, 3 ,1]

        
#    list_data_GUI_strain =\
#    ["#UBB0 matrix in q= (UB B0) G*", # mat
#     "# Mean Deviation(pixel):",
#     "#DetectorParameters",
#     "#UB matrix in q= (UB) B0 G*"] # orientmat
#     
#    line_codes_GUI_strain = [3,0,1]


    list_data_GUI_strain =\
    ["#UB matrix in q= (UB) B0 G*", # orient mat
#      "pixelsize",  #bricolage  
      "# Mean Deviation(pixel):",  # 02Feb16
     "#DetectorParameters",
     "#B0 matrix in q= UB (B0) G*"] # B0
     
    line_codes_GUI_strain = [3,0,1,3]


    list_data_GUI_calib =\
    ["#UB matrix in q= (UB) B0 G*", # orient mat
#      "pixelsize",  #bricolage  
      "# Mean Deviation(pixel):",  # 02Feb16
     "#DetectorParameters",
     "#B0 matrix in q= UB (B0) G*"] # B0
     
    line_codes_GUI_calib = [3,0,1,3]
    
    list_data_FileSeries =\
    ["#UBB0 matrix in q= (UB B0) G*",
     "# Mean Pixel Deviation:",
     "# Calibration parameters",
     "#UB matrix in q= (UB) B0 G*"]    
    
    line_codes_FileSeries= [3,0,7]

    # key = fitfile_type
    # first_line_ends_with
    # npeaks syntax
    dict_filetype = {"MG" : [".dat", spot_columns_MG.split(), list_data_MG, line_codes_MG ], 
                     "MGnew" : [".dat", spot_columns_MG.split(), list_data_MGnew, line_codes_MGnew ], 
                      "GUI_strain" : [".cor", spot_columns_GUI_strain.split(), list_data_GUI_strain, line_codes_GUI_strain],
                       "GUI_calib" : [".dat", spot_columns_GUI_calib.split(), list_data_GUI_calib, line_codes_GUI_calib],
                   "FileSeries" : [".cor", spot_columns_GUI_strain.split(), list_data_FileSeries, line_codes_FileSeries],
                                    
                     }

    listnames =  dict_filetype[filetype][1]                   
    indh = listnames.index("h")
    if verbose : print("indh = ", indh)
    if "Xexp" in listnames : indx = listnames.index("Xexp")
    if "xexp" in listnames : indx = listnames.index("xexp")

    if "Intensity" in listnames : indint = listnames.index("Intensity")
    if "intensity" in listnames : indint = listnames.index("intensity")
    if "Itot" in listnames : indint = listnames.index("Itot")

    if "pixDev" in listnames : indpixdev = listnames.index("pixDev")
    if "PixDev" in listnames : indpixdev = listnames.index("PixDev")  
    if "pixDevCalib" in listnames : indpixdev = listnames.index("pixDevCalib")   
    
    if "Etheor" in listnames : indEtheor = listnames.index("Etheor")
    if "Energy" in listnames : indEtheor = listnames.index("Energy")
    if "energy(keV)" in listnames : indEtheor = listnames.index("energy(keV)")
    
    indtth = None
    if "2theta" in listnames : indtth = listnames.index("2theta")
    if "2theta_exp" in listnames : indtth = listnames.index("2theta_exp")
    if "2theta_theo" in listnames : indtth = listnames.index("2theta_theo")
    
    ind_h_x_int_pixdev_Etheor = np.array([indh, indx, indint, indpixdev, indEtheor])
    
    extension = dict_filetype[filetype][0]
     
    if verbose:
        print("read_any_fitfitfile_multigrain")
        print(filefitmg)
        print("filetype = ", filetype)
    
    matpos = 0   
    
    B0_name = "yoho"
    
    mat_name = dict_filetype[filetype][2][matpos]
#    mat_code =  dict_filetype[filetype][2][matpos] 
    orient_mat_name = dict_filetype[filetype][2][3]
    
    pixdevpos = 1
    pixdev_name = dict_filetype[filetype][2][pixdevpos]
    pixdev_code =  dict_filetype[filetype][3][pixdevpos] 
    
    calibpos = 2
    calib_name = dict_filetype[filetype][2][calibpos]
    calib_code =  dict_filetype[filetype][3][calibpos] 
    
    first_col_spot = dict_filetype[filetype][1][0]

    if filetype in ["MG", "MGnew"] :
#        print dict_filetype[filetype][2]
        strainpos = 4
        strain_name = dict_filetype[filetype][2][strainpos]
#        strain_code =  dict_filetype[filetype][3][strainpos]         
        
        eulerpos = 5
        euler_name = dict_filetype[filetype][2][eulerpos]
#        euler_code =  dict_filetype[filetype][3][eulerpos]           
#        print extension

    else :
        strain_name = "yoho"
        euler_name = "yoho"
        
    if filetype in ["GUI_calib", "GUI_strain"] :
#    if filetype in ["GUI_calib",] :
        orient_mat_name = dict_filetype[filetype][2][0]
        B0_name = dict_filetype[filetype][2][3]
        mat_name = "yoho"

    pixelsize_name = "#pixelsize"

    if verbose : print(filefitmg) 
    
    f = open(filefitmg, 'r')
    nbgrains = 0
    linepos_list = []
    i = 0
    try:
        for line in f:
            toto = line.rstrip(PAR.cr_string)
#            print toto
#            print toto[-4:]
            if ((toto.startswith("# Strain") or toto.startswith("# CCD")) and (toto.endswith(extension))) :
                nbgrains = nbgrains + 1
                linepos_list.append(i)
            i = i+1
    finally:
        linepos_list.append(i)
        f.close()
        
    if verbose : 
        print("nbgrains = ", nbgrains)
        print("linepos_list = ", linepos_list)
    
    if nbgrains == 0 : return(0)
    
    f = open(filefitmg, 'r')
    
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)
        
    gnumlist = np.arange(nbgrains)
    
    matstarlab = zeros((nbgrains, 9), float)
    strain6 = zeros((nbgrains, 6), float)
#    strain6sample = zeros((nbgrains, 6), float)
    calib = zeros((nbgrains, 5), float)
    euler = zeros((nbgrains, 3), float)
    npeaks = zeros(nbgrains, int)
    indstart = zeros(nbgrains, int)
    pixdev = zeros(nbgrains, float)
    
    # lecture .fit pour chaque grain    
    for k in range(nbgrains) : 
        matLT3x3 = np.zeros((3, 3), dtype=np.float)
        orientmat = np.zeros((3, 3), dtype=np.float)
        B0mat = np.zeros((3, 3), dtype=np.float)
        strain = np.zeros((3, 3), dtype=np.float)
        strain2 = np.zeros((3, 3), dtype=np.float)
        i = 0
        n = linepos_list[k]
        #print "n = ", n
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        #print f.readline()
        f.seek(line_offset[n+1])
        matrixfound = 0
        calibfound = 0
        pixdevfound = 0
        strainfound = 0
        strainfound2 = 0
        eulerfound = 0
        orient_mat_found = 0
        B0found = 0
        pixelsizefound = 0
          
        linecalib = 0
        linepixelsize = 0
        linepixdev = 0
        linestrain = 0
        linestrain2 = 0
        lineeuler = 0
        list1 = []
        linestartspot = 10000
        lineendspot = 10000    
            
        while (i<(linepos_list[k+1]-linepos_list[k])):
            line = f.readline()
            i = i + 1
            #print i
            if line.startswith(first_col_spot):
                linecol = line.rstrip(PAR.cr_string)
                linestartspot = i + 1
            if line.startswith(orient_mat_name) :
                lineendspot = i                
            if line.startswith(mat_name):
                #print line
                matrixfound = 1
                linestartmat = i
                j = 0
                if verbose : print("matrix found")
            if line.startswith(orient_mat_name):
                #print line
                orient_mat_found = 1
                linestartorientmat = i
                j = 0
                if verbose : print("orientation matrix found")                    
            if line.startswith(B0_name):
                #print line
                B0found = 1
                linestartB0 = i
                j = 0
                if verbose : print("B0 matrix found")
            if line.startswith(calib_name) :
                #print line
                calibfound = 1
                linecalib = i + 1
                j = 0               
            if line.startswith(pixelsize_name) :
                #print line
                pixelsizefound = 1
                linepixelsize = i + 1
                j = 0
            if line.startswith(pixdev_name) :
                #print line
                if pixdev_code == 0 :
                    pixdev[k] = float(line.rstrip(PAR.cr_string).split()[-1])
                    if verbose : print("pixdev =", pixdev)
                else :
                    pixdevfound = 1
                    linepixdev = i + 1
            if line.startswith(strain_name) :
                #print line
                strainfound = 1
                linestrain = i
                j = 0
            if line.startswith(euler_name) :
                #print line
                eulerfound = 1
                lineeuler = i + 1
            if matrixfound :
                if i in (linestartmat + 1, linestartmat + 2, linestartmat + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
#                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    j = j + 1                                        
            if B0found :
                 if i in (linestartB0 + 1, linestartB0 + 2, linestartB0 + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
#                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    B0mat[j, :] = np.array(toto, dtype=float)
                    j = j + 1  
            if orient_mat_found :  
                 if i in (linestartorientmat + 1, linestartorientmat + 2, linestartorientmat + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
#                    matLT3x3[j, :] = np.array(toto, dtype=float)
                    orientmat[j, :] = np.array(toto, dtype=float)
                    j = j + 1  
                        
            if strainfound :
                if i in (linestrain + 1, linestrain + 2, linestrain + 3) :
                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
                    #print toto
                    strain[j, :] = np.array(toto, dtype=float)
                    j = j + 1                    
#            if strainfound2 :
#                if i in (linestrain2 + 1, linestrain2 + 2, linestrain2 + 3) :
#                    toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split()
#                    #print toto
#                    strain2[j, :] = np.array(toto, dtype=float)
#                    j = j + 1
            if calibfound :
#                print "calib_code = ", calib_code
#                print line
                if (calib_code == 1) & (i == linecalib) :
                    calib[k,:] = np.array(line.replace('[', '').replace(']', '').split(',')[:5], dtype=float)
                elif (calib_code == 7) and (i in range(linecalib, linecalib+5)) :
                    calib[k,j] = float(line.rstrip(PAR.cr_string).split()[-1])
                    j = j+1                    
#                print "calib = ", calib[k,:]

            if pixelsizefound & (i == linepixelsize) and check_pixdev :
                pixelsize_fit = float(line.rstrip(PAR.cr_string))
                if verbose : print("pixelsize from fitfile = ", pixelsize_fit)
                toto = abs(pixelsize_fit-pixelsize)
                if toto > 0.0001 :
                    print("does not match pixelsize from MG.PAR.CCDlabel", pixelsize)
                    exhjkqsdq
            if eulerfound & (i == lineeuler):
                euler[k,:] = np.array(line.replace('[', '').replace(']', '').split()[:3], dtype=float)
                #print "euler = ", euler[k,:]
            if pixdevfound & (i == linepixdev):
                pixdev[k] = float(line.rstrip(PAR.cr_string))
                #print "pixdev = ", pixdev[k]
            if (i >= linestartspot) & (i < lineendspot) :
                list1.append(line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split())

#        print "list1 = ", list1
        data_fit = np.array(list1, dtype=float)
        npeaks[k]=shape(data_fit)[0]
        
        verbose2 = 0
        if verbose2 :
            print(np.shape(data_fit))
            print(data_fit[0, :])
            print(data_fit[-1, :])
            
        if calibfound :
            if verbose : 
                print("calib found")
                print("current calib = ", calib)
        else :
            print("calib not found in file")
            exit()
            
#        TODO : re-ranger xy hkl pour

        # xx yy zz yz xz xy
        strain6[k,:] = np.array([strain[0, 0], strain[1, 1], strain[2, 2], strain[1, 2], strain[0, 2], strain[0, 1]])
#        strain6sample[k,:] = np.array([strain2[0, 0], strain2[1, 1], strain2[2, 2], strain2[1, 2], strain2[0, 2], strain2[0, 1]])

#        print "orient_mat_found = ", orient_mat_found
#        print "B0found = ", B0found
#        jklqsd

        if orient_mat_found and B0found :
            matLT3x3 = dot(orientmat, B0mat*10.)
#            matLT3x3 = orientmat
            if verbose : print("B0 = \n", B0mat)
            if 0 :
                print("test B0mat :")
                rlat = mat_to_rlat(GT.mat3x3_to_matline(B0mat))       
                print("reciprocal lattice parameters (inv_ang, rad) from B0 mat only :")
                print(rlat)
                print(rlat[0:3]/rlat[0],     rlat[3:6]*180./PI)              
                dlat = dlat_to_rlat(rlat)
                print("direct lattice parameters (angstroms, rad) from B0 mat only :")
                print(dlat)
                print(dlat[0:3]/dlat[0],     dlat[3:6]*180./PI)            
            
            if verbose : print("orientmat = \n", orientmat)
            if 0 :
                print("test orientmat :")
                rlat = mat_to_rlat(GT.mat3x3_to_matline(orientmat))
                print("reciprocal lattice parameters (inv_ang, rad) from orientmat UB only :")
                print(rlat)
                print(rlat[0:3]/rlat[0],     rlat[3:6]*180./PI) 
            
        if verbose : print("matLT3x3 =\n", matLT3x3)
        
        if min_matLT :  
            matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3)
            matLT3x3 = matmin*1.
            if verbose : print("transfmat \n", list(transfmat))
            # transformer aussi les HKL pour qu'ils soient coherents avec matmin
            hkl = data_fit[:, indh:indh+3]*1.
            data_fit[:, indh:indh+3] = np.dot(transfmat, hkl.transpose()).transpose()
            
        matstarlab[k,:] =F2TC.matstarlabLaueTools_to_matstarlabOR(matLT3x3) 

        if verbose : print("matstarlab[k,:] = \n", matstarlab[k,:])
        
#        matstarlab[k,:] = np.array([0.823564, 0.511495, 0.245185, -0.198049, -0.145751, 0.969295, 0.531526, -0.846835, -0.018734 ])
#   0.833701 0.507272 0.218216 -0.161827 -0.153384 0.974826 0.527973 -0.848026 -0.045786           
#        m11 -m01 -m21  m12 -m02 m22  m10 -m00 -m20 #
 
        if filetype == "GUI_strain" or filetype == "GUI_calib" :
            print("strain not read")  # pour eviter strain incorrect dans cas matLT_min = 1
            
        hkl = data_fit[:, indh:indh+3] 
        xy = data_fit[:, indx:indx+2]
        
        if check_pixdev_JSM :
            
            print("**********************************************************")
            print("checking pixdev, using JSM calculation from laue6.py")

#            H, K, L, Qx, Qy, Qz, X, Y, twthe, chi, Energy = LAUE.calcSpots_fromHKLlist(UB, B0, HKL, dictCCD)

            dictCCD = {}
            dictCCD['CCDparam'] = calib[0]
            dictCCD['pixelsize'] = pixelsize
            dictCCD['dim'] = DictLT.dict_CCD[PAR.CCDlabel][0]
            if verbose :
                for key, value in dictCCD.items():
                    print(key, value)
            H, K, L, Qx, Qy, Qz, X, Y, twthe, chi, Energy = LAUE.calcSpots_fromHKLlist(orientmat, B0mat, hkl, dictCCD)
#            print "yoho"
            
            xy_new = np.column_stack((X,Y))
            
            npics = np.shape(hkl)[0]
            pixdev1 = np.zeros(npics,float)            
            
            if verbose : print("i, hkl, xy_from_fitfile, xy_recalculated, pixdev_recalculated")
            for i in range(npics):
                pixdev1[i] = norme(xy_new[i,:]-xy[i,:])
                if verbose : print(i, hkl[i,:],  xy[i,:], xy_new[i,:].round(decimals=2), round(pixdev1[i],3))

            print("pixdev_from_fitfile, pixdev_recalculated")
            print(pixdev[k], pixdev1.mean())            

        if check_pixdev :
            
            print("*****************************************************************")
            print("checking pixdev from MG calculation")
            
            if verbose : 
                print("hkl from fitfile = \n ", hkl)
                print("xy from fitfile = \n", xy)
                print("calib[k,:] = ", calib[k,:])
                print("pixelsize =" , pixelsize)
                
            pixdev_new = calc_pixdev(matstarlab[k, :], 
                                     calib[k,:], 
                                        xy,
                                        hkl,
                                        pixelsize = pixelsize,
                                        verbose = 1)
            
            print("pixdev : from_fitfile, recalculated")
            print(round(pixdev[k],3), round(pixdev_new,3))
            dpixdev = abs(pixdev_new - pixdev[k])
            
            if (dpixdev > 0.01) or isnan(pixdev_new) :
                threshold1 = 0.03
                print("warning : recalculated pixdev does not match pixdev from fitfile")
                print("recalculating HKLs")
                print("max value of norme([H,K,L] - [H,K,L].round()) : ", threshold1)
                print("hkl_from_fitfile, uqcr_from_mat_and_xy")
                hkl_new = zeros((npeaks[k],3), float)
                for ii in range(npeaks[k]):
                    uqlab_exp = xycam_to_uqlab(xy[ii,:],calib[k,:], pixelsize = pixelsize)
                    invmat = inv(GT.matline_to_mat3x3(matstarlab[k, :]))
                    uqcr_exp = dot(invmat,uqlab_exp)

                    toto = max(abs(uqcr_exp))
                    uqcr_exp = uqcr_exp / toto
#                    print uqcr_exp.round(decimals=3)
                    if 1 :
                        min1 = 20.
                        test_integer = norme(uqcr_exp - uqcr_exp.round(decimals=0))
                        uqcr_best = np.zeros(3,float)
                        for jj in range(1,20) :
                            uqcr_exp_new = jj * uqcr_exp
#                            print uqcr_exp_new.round(decimals=3)
                            test_integer = norme(uqcr_exp_new - uqcr_exp_new.round(decimals=0))
#                            print jj, test_integer
                            if test_integer < min1 :
                                min1 = test_integer * 1.
                                uqcr_best = uqcr_exp_new *1.
                        if min1 < threshold1 : # 0.03 :
                            hkl_new[ii,:] = uqcr_best.round(decimals=0)
#                            uqcr_best = uqcr_exp_new *1.
#                                break
                            
                    if 0 :
                        uqcr_exp_new = uqcr_exp / norme(uqcr_exp) * norme(hkl[ii,:]) 
                        hkl_new[ii,:] = uqcr_exp_new.round(decimals=0)                           
                    
                    test_str = "OK"
                    str1 = ""
                    if norme(hkl_new[ii,:])< 0.01 :
                        test_str = "bad"
                        str1 = str(round(min1,3))
                        
                    if test_str == "OK" :                    
                        print(ii, hkl[ii,:], uqcr_best.round(decimals = 2))
                    else :
                        print(ii, hkl[ii,:],  uqcr_best.round(decimals = 3), test_str, str1)

#                data_fit[:, indh:indh+3] = hkl_new    # A REMETTRE

                print("recalculating pixdev with new HKL's")
                pixdev_new2 = calc_pixdev(matstarlab[k, :], 
                                         calib[k,:], 
                                            xy,
                                            hkl_new,
                                            pixelsize = pixelsize,
                                            verbose = 1)
                
#                print "pixdev : from_fitfile, recalculated with new hkl"
                print("pixdev_from_fitfile, pixdev_recalculated_with_new_HKLs")
                print(round(pixdev[k],3), round(pixdev_new2,3))

                    
        if check_Etheor :
            
            print("*****************************************************************")
            print("checking Etheor values :")
            print("using elem_label = ", elem_label)   
            print("warning : matrix used to calculate Etheor values depends on author")
            print("for GUI and FileSeries fit files :")
            print("use UBB0 matrix with norme(astar) = norme(astar0)")
            print("for MG fit files :")
            print("use UBB0 matrix with vol_cell = vol_cell0")
            print("grain : ", k)

            hkl = data_fit[:, indh:indh+3]
            print("hkl from fitfile = \n", hkl)
            print("xy from_fitfile = \n", xy)        
  
            
#            print norme(matstarlab[k,0:3])
           
            Etheor = zeros(npeaks[k],float)
#            uilab = array([0.,1.,0.])
#            latticeparam = DictLT.dict_Materials[elem_label][1][0] * 1.
#            print "lattice parameter = ", latticeparam
#            dlatu = np.array([latticeparam, latticeparam, latticeparam, PI/2., PI/2., PI/2.])
#            dlatu_deg =  np.array(DictLT.dict_Materials[elem_label][1])
#            dlatu_rad = deg_to_rad(dlatu_deg)
#            print "lattice parameters from elem_label angstroms - rad"
#            print dlatu_rad
#            mat = F2TC.matstarlab_to_matwithlatpar(matstarlab[k,:], dlatu)
#            print "1./ a"
#            print 1./dlatu[0]
#            print "norme(matwithlatpar)"
#            print norme(mat[:3]), norme(mat[3:6]), norme(mat[6:])
#            rlat = mat_to_rlat(mat)
#            vol = CP.vol_cell(rlat, angles_in_deg=0)
#            print "astarmean = root3(reciprocal unit cell volume) from mat"
#            print pow(vol,1./3.)
#                        
            dlatu_angstroms_deg = DictLT.dict_Materials[elem_label][1]
            dlatu_nm_rad = deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
            print("dlatu_nm_rad = ", dlatu_nm_rad)                    
            matwithlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matstarlab[k,:], dlatu_nm_rad)

            if indtth  is  not None :        
                print("hkl, Etheor_recalc, Etheor_from_fitfile, 2theta_recalc, 2theta_from_fitfile, dE/E(1e-3), dtth / tth")
            else :
                print("hkl, Etheor_recalc, Etheor_from_fitfile, 2theta_recalc, dE/E(1e-3)")
               
            for ii in range(npeaks[k]):
                
                Etheor_eV, ththeor_deg, uqlab = mat_and_hkl_to_Etheor_ththeor_uqlab(matwithlatpar_inv_nm = matwithlatpar_inv_nm,
                                                                            hkl = hkl[ii,:])
                                                                            
                Etheor[ii] = Etheor_eV
                tth = 2.* ththeor_deg
                if fitfile_type[:3] == "GUI" :
                    Etheor_fromfitfile = data_fit[ii,indEtheor]*1000.
                else :
                    Etheor_fromfitfile = data_fit[ii,indEtheor]
                    
                dEsE = (Etheor[ii]/Etheor_fromfitfile-1.)*1000.                 
                
                if indtth  is  not None :
                    dtthstth = tth / data_fit[ii,indtth] - 1.
                    print(ii, hkl[ii,:], round(Etheor[ii],1), round(Etheor_fromfitfile, 1), round(tth,3), round(data_fit[ii,indtth], 3), round(dEsE,4), round(dtthstth,4))
                else :
                    print(ii, hkl[ii,:], round(Etheor[ii],1), round(Etheor_fromfitfile, 1), round(tth,3), round(dEsE,4))

            dE = Etheor-data_fit[:,indEtheor]
            if dE.max() > 2.0 :  # 2 eV
                print("warning : recalculated Etheor inconsistent with Etheor in fitfile")
                data_fit[:,indEtheor] = Etheor
            
        if k == 0 :
           data_fit_all = data_fit * 1.0
        else :
           data_fit_all = row_stack((data_fit_all, data_fit))
                      
    f.close()
    
    for k in range(1,nbgrains): indstart[k]=indstart[k-1]+npeaks[k-1]

    #04Dec13
    matstarlab = matstarlab.round(decimals=6)
    data_fit_all = data_fit_all.round(decimals=4)
    calib = calib.round(decimals=3)
    pixdev = pixdev.round(decimals=4)
    strain6 = strain6.round(decimals=2)
#    strain6sample = strain6sample.round(decimals=2)    
    euler =  euler.round(decimals=3)
         
    if verbose :
        print("gnumlist = ", gnumlist)
        print("matstarlab = ")        
        print(matstarlab)
        print("npeaks = ", npeaks)
        print("indstart = ", indstart)
        print("pixdev = ", pixdev)
        print("strain6 = \n", strain6)
        print("euler = \n", euler)


    return(gnumlist, npeaks, indstart, matstarlab, data_fit_all, calib, pixdev, strain6, euler, ind_h_x_int_pixdev_Etheor)   

def calc_pixdev(matstarlab, 
                calib, 
                data_xy, 
                data_hkl,
                pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1],
                verbose = 0,
                return_xydev_list = None):
    
    Npics = len(data_xy[:, 0])
#    xytheor = zeros((Npics, 2), float)
    xydev = zeros((Npics, 2), float)
    pixdev = zeros(Npics, float)
    isbadspot = zeros(Npics, int)
    uilab = np.array([0., 1., 0.])

    mat1 = matstarlab
    
    if verbose : print("pixelsize (in calc_pixdev) = ", pixelsize)
    
    if verbose : print("hkl , uqlab, xydev, pixdev")
    for i in range(Npics):
        qlab = data_hkl[i, 0] * mat1[0:3] + data_hkl[i, 1] * mat1[3:6] + data_hkl[i, 2] * mat1[6:] 
        if norme(qlab)> 0.01 :
            uqlab = qlab / norme(qlab)
            sintheta = -inner(uqlab, uilab)
            if (sintheta > 0.0) :
                xydev[i, :] = uqlab_to_xycam(uqlab, calib, pixelsize = pixelsize) - data_xy[i, :]
                pixdev[i] = norme(xydev[i, :]) 
                if verbose : 
                    print(i, np.array(data_hkl[i,:], dtype=int), \
                            uqlab.round(decimals=3),\
                            xydev[i].round(decimals=3), \
                            round(pixdev[i],3))
            else :
                print(i, data_hkl[i, :], "unreachable reflection in calc_pixdev")
                print("exclude spot from pixdev calculation")
                isbadspot[i] = 1

        else :
            print(i, data_hkl[i, :], "unreachable reflection in calc_pixdev ")
            print("exclude spot from pixdev calculation")
            isbadspot[i] = 1            
            
    ind1 = where(isbadspot == 0)
    pixdev_short = pixdev[ind1[0]]
    pixdevmean = pixdev_short.mean()      
    if verbose : print("pixdev : mean ", round(pixdevmean,3))
    
    if return_xydev_list == None : return(pixdevmean)
    else : return(pixdevmean, xydev, pixdev)
    
def compare_multigrain_fit(filefitmg1, 
                           filefitmg2, 
                           mat_tol=1.0e-3, 
                           dxytol=0.05, 
                           compare_calib=0, 
                           elem_label = "W",
                           first_line_ends_with1 = ".dat",
                           first_line_ends_with2 = ".str",
                           compare_strain_sample = 0,
                           min_matLT = False):
 
    #29May13       
    gnumlist1, npeaks1, indstart1, matstarlab1, data_fit1, calib1, pixdev1, strain61, eulerall1, strain6sample1 = \
        readlt_fit_mg(filefitmg1,
                      first_line_ends_with = first_line_ends_with1,
                      readmore2 = True)
    gnumlist2, npeaks2, indstart2, matstarlab2, data_fit2, calib2, pixdev2, strain62, eulerall2, strain6sample2 =\
        readlt_fit_mg(filefitmg2,
                      first_line_ends_with = first_line_ends_with2,
                      readmore2 = True)
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
            dmat = (mat2 - mat1)
            #print "gnum1, gnum2, norme(dmat) = ",  gnumlist1[i], gnumlist2[j], norme(dmat)
            if norme(dmat) < mat_tol :
                #print "dmat = ", dmat
                print("gnum1, gnum2, norme(dmat) = ", gnumlist1[i], gnumlist2[j], norme(dmat))
                grain_couples[k, 0] = i
                grain_couples[k, 1] = j
                grain_couples[k, 2] = gnumlist1[i]
                grain_couples[k, 3] = gnumlist2[j]
                print(grain_couples[k, :])
                k = k + 1
                matching_grain_found = 1
        if matching_grain_found == 0 :
            print("no match for grain gind, gnum :", i, gnumlist1[i])
            if 1 :
                range1 = arange(indstart1[i], indstart1[i] + npeaks1[i])
                xy1 = data_fit1[range1, 6:8]
                for j in range(ng2):
                    if j not in grain_couples[:k, 1]:
                        print("gnum1, gnum2 = ", gnumlist1[i], gnumlist2[j])
                        range2 = arange(indstart2[j], indstart2[j] + npeaks2[j])
                        xy2 = data_fit2[range2, 6:8]
                        find_common_peaks(xy1, xy2, dxytol=dxytol)                             
    #print grain_couples
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
        else :
            range1 = arange(0, npeaks1)
        if len(indstart2) > 1:
            range2 = arange(indstart2[j], indstart2[j] + npeaks2[j])
        else :
            range2 = arange(0, npeaks2)
        xy1 = data_fit1[range1, 6:8]
        xy2 = data_fit2[range2, 6:8]
        find_common_peaks(xy1, xy2, dxytol=dxytol)
        
        if min_matLT == True :
            matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1[i, :])
            #GT.mat3x3_to_matline(matLTmin)
            
            epsp1, dlatsrdeg1 = matstarlab_to_deviatoric_strain_crystal(matstarlab1[i, :], 
                                    version = 2, 
                                    elem_label = elem_label)
            
            euler1 = calc_Euler_angles(matLTmin).round(decimals=3)
            
            matLTmin = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab2[j, :])
            
            epsp2, dlatsrdeg1 = matstarlab_to_deviatoric_strain_crystal(matstarlab2[j, :], 
                                    version = 2, 
                                    elem_label = elem_label)        
            
            euler2 = calc_Euler_angles(matLTmin).round(decimals=3)
            
        else :
            epsp1 = strain61[i,:]
            epsp2 = strain62[j,:,]
            epspsample1 = strain6sample1[i,:] 
            epspsample2 = strain6sample2[j,:] 
            euler1 = eulerall1[i,:]
            euler2 = eulerall2[j,:]
        
        print("deviatoric strain crystal :")
        deps = (epsp2 - epsp1) * 10.0
        print("difference eps2 - eps1 (1e-4 units)")
        print(deps.round(decimals=1))
        print("eps1 (1e-4 units)")
        print((epsp1*10.).round(decimals=1))
        print("eps2 (1e-4 units)")
        print((epsp2*10.).round(decimals=1))
        print("deviatoric strain sample :")
        deps = (epspsample2 - epspsample1) * 10.0
        print("difference eps2 - eps1 (1e-4 units)")
        print(deps.round(decimals=1))
        print("eps1 (1e-4 units)")
        print((epspsample1*10.).round(decimals=1))
        print("eps2 (1e-4 units)")
        print((epspsample2*10.).round(decimals=1))


        deuler = (euler2 - euler1) * 1.0e4 * PI / 180.0
        print("difference of euler angles euler2 - euler1 (0.1 mrad)")
        print(deuler.round(decimals=1))
        hkl1 = data_fit1[range1, 2:5]
        hkl2 = data_fit2[range2, 2:5]
        print("pixdev1, pixdev2 (from file)")
        print(round(pixdev1[i], 4), round(pixdev2[j], 4))
        print("pixdev1, pixdev2 (recalculated from xy hkl mat)")
#        print calib1
#        print calib2
        pixdev1b = calc_pixdev(matstarlab1[i, :], calib1[i], xy1, hkl1)
        pixdev2b = calc_pixdev(matstarlab2[j, :], calib2[j], xy2, hkl2)
        print(round(pixdev1b, 4), round(pixdev2b, 4))
        if compare_calib :
            dcalibn = zeros(5, float)
#            pixelsize = 165.0 / 2048.0
            pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]
            RAD = PI / 180.0
            dcalib = calib2[j] - calib1[i]
            print("difference of calibration")
            print("standard units : mm pix pix deg deg")
            print_calib(dcalib, pixelsize = pixelsize)
            print("angular units 0.1 mrad : d(dd)/dd, d(xcen)/dd, d(ycen)/dd, dxbet, dxgam")
            dcalibn[0] = dcalib[0] / calib1[i,0]
            dcalibn[1] = dcalib[1] * pixelsize / calib1[i,0]
            dcalibn[2] = dcalib[2] * pixelsize / calib1[i,0]
            dcalibn[3] = dcalib[3] * RAD
            dcalibn[4] = dcalib[4] * RAD
            dcalibn = dcalibn * 1.0e4
            print(dcalibn.round(decimals=1))
                
        if 0 :
            p.figure(figsize=(8, 8))
            p.plot(xy1[:, 0], -xy1[:, 1], "ro")
            p.plot(xy2[:, 0], -xy2[:, 1], "kx", markersize=20, markeredgewidth=3)
            text1 = "G" + str(grain_couples[k, 2])
            p.text(100.0, -150.0, text1, fontsize=20)
            p.xlim(0.0, 2048.0)
            p.ylim(-2048.0, 0.0)
            p.xlabel('xcam XMAS - toward back')
            p.ylabel('-ycamXMAS - upstream')
        if 0 :
            if k == 0 :
                    p.figure(num=1, figsize=(8, 8))
            else :
                    p.figure(1)
            p.plot(xy1[:, 0], -xy1[:, 1], color1[k])     
            p.xlim(0.0, 2048.0)
            p.ylim(-2048.0, 0.0)
            p.xlabel('xcam XMAS - toward back')
            p.ylabel('-ycamXMAS - upstream')                    

    return(0)

def read_dat(filedat, filetype="XMAS", flip_xaxis = "no"):

    if not os.path.isfile(filedat) :
        if filetype == "XMAS" :  return([0.,0.],0.)
        elif filetype == "LT" :  return([0.,0.],0.,0.)
        
    if filetype == "XMAS" :
        print("reading peak list from XMAS DAT file : (fit case) \n", filedat)
    elif filetype == "LT" :
        print("reading peak list from LaueTools DAT file : (fit case) \n", filedat)

    # lecture liste positions .DAT de XMAS
    # attribution des colonnes (cas du fit)
    # 0 , 1 : xfit, yfit
    # 2, 3  :intens max - background, integr
    # 4 , 5, 6  : widthx, widthy, tilt
    # 7 : Rfactor du fit
    # 8 : rien
    # 9, 10 : x(centroid), y(centroid)

    # .DAT de LT
    #peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax

    fileminsize = 700
    filesize = os.path.getsize(filedat)
    print("file size", filesize)

    if filesize > fileminsize :
        #data_dat = scipy.io.array_import.read_array(filedat, columns=((0, -1)), lines=(1, -1))
        data_dat = loadtxt(filedat, skiprows = 1)  
#        print shape(data_dat)
#        if (shape(data_dat)[0]< 2) :
#            data_dat = zeros((4,5),float)
#        if np.isscalar(data_dat[0]):
#            toto = zeros(2,float)
#            if filetype == "LT"  : return(toto,toto,0.)
#            elif filetype == "XMAS" : return(toto,toto)


        if filetype == "LT" :  # 23Jun14 sort by decreasing Ipixmax for later removal of duplicates
            data_dat = sort_peaks_decreasing_int(data_dat, -1)    
            data_Ipixmax = data_dat[:, -1]
        
        data_xyexp = data_dat[:, 0:2]
        data_int = data_dat[:, 2:4]
    
        # for VHR images Jun12
        if flip_xaxis =="yes" :
            data_xyexp[:,0] = 2594.0 - data_xyexp[:,0]
                
        if filetype == "XMAS" :
            # exchange the two intensity column for consistency with xmas spot sorting by intensities
            data_int2 = data_int * 1.0
            data_int2[:, 0] = data_int[:, 1]
            data_int2[:, 1] = data_int[:, 0]
            return(data_xyexp, data_int2)
        elif filetype == "LT" :
            return(data_xyexp, data_int, data_Ipixmax)
            
    else :
        if filetype == "XMAS" : return([0.,0.],0.)
        elif filetype == "LT" : return([0.,0.],0.,0.)

def test_LT(filestr, elem_label, min_matLT = True):

    # use full peak list (multigrain) from XMAS STR file
    # use guess matrixes from STR file for multigrain indexation, after orthonormalizing
    
    data_str_all, matstarlab, calib, dev_str, npeaks, strain, strain2 = \
        read_all_grains_str(filestr, min_matLT=min_matLT)

    xyII = (vstack((data_str_all[:, 0], data_str_all[:, 1], data_str_all[:, 9], data_str_all[:, 10]))).transpose()
    print(shape(xyII))
    # xyII = xyII[:npeaks[0],:]
    # print shape(xyII)
    # ajout d'une ligne de zeros pour simuler le header du .DAT classique
    toto = zeros((1, 4), float)
    xyII = vstack((toto, xyII))
    filedat = filestr.rstrip(".str") + "_from_str.dat"
    savetxt(filedat, xyII, fmt='%.4f')

    ngrains = len(npeaks)
    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(ngrains):
        if npeaks[i] > 10 :
            matstarlabOND = matstarlab_to_matstarlabOND(matstarlab[i, :])
            matstarlab1 = matstarlabOND * 1.0
            #matstarlab1 =matstarlab[i,:]*1.0
            matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
            paramdetector_top = list(calib)
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = \
                  test_index_refine(filedat, paramdetector_top, \
                            use_weights=False, proposed_matrix=matLT3x3, \
                          check_grain_presence=None, paramtofit="strain", \
                                    elem_label=elem_label, grainnum=i + 1,
                                    min_matLT = min_matLT)
            filelist.append(filefit)
            
    merge_fit_files_multigrain(filelist)        
    print("min_matLT = ", min_matLT)
    print("orthonormalize matstarlab before indexation")
    print("npeaks from str file")
    print(npeaks)
    print("npeaks for LT fit")
    print(npeaks_LT)
    print("pixdev from str file")
    print(dev_str[:, 2])
    print("pixdev after LT fit")
    print(pixdev_LT)
        
    return(0)

def test_LT_2(filestr, elem_label):

    # use full peak list (multigrain) from XMAS STR file
    # no initial guess for indexation
    
    min_matLT = True
    data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(filestr, min_matLT=min_matLT)
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
        outputfile = open(filedat, 'w')
        outputfile.write(header)
        np.savetxt(outputfile, xyII, fmt='%.4f')
        outputfile.close()
    else :
        xyII = loadtxt(filedat, skiprows=1)
            
    ngrains = 10
    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(ngrains):   # adjust range to keep .fit files of already indexed grains
    #for i in range(6,10):
        paramdetector_top = list(calib)
        res1 = test_index_refine(filedat, paramdetector_top, \
                        use_weights=False, proposed_matrix=None, \
                      check_grain_presence=1, paramtofit="strain", \
                                elem_label=elem_label, grainnum=i + 1)

        if res1 != 0 :                   
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
            filelist.append(filefit)
            matstarlab_LT[i, :], data_fit = F2TC.readlt_fit(filefit)
            nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(xyII[:, :2], data_fit[:, 6:8])
            ind1 = where(iscommon1 == 0)
            
            xyII = xyII[ind1[0], :]
            outputfile = open(filedat, 'w')
            outputfile.write(header)
            np.savetxt(outputfile, xyII, fmt='%.4f')
            outputfile.close()
            print(shape(xyII))
        else :
            break
            
    merge_fit_files_multigrain(filelist)        
    print(npeaks)
    print(npeaks_LT)
    print(dev_str[:, 2])
    print(pixdev_LT)
        
    return(0)


##def test_LT_3(filedat1, elem_label, filestr = None, filefit = None, filedattype = "XMAS"):
##
##        # use peak list from peak search (XMAS or LT) 
##        # use calib either from XMAS STR file or from LT fit file
##        # no initial guess for indexation
##        
##        min_matLT = True
##        if filestr  is  not None :
##                data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(filestr, min_matLT = min_matLT)
##        elif filefit  is  not None :
##                matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefit, readmore = True)
##
##        if filedattype == "XMAS" :
##                data_xy, data_int = read_dat(filedat1, filetype = "XMAS")
##                header = "xexp yexp integr intens \n"
##        elif filedattype == "LT" :
##                data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype = "LT")
##                header = "xexp yexp Ipeak Isub \n"
##                
##        
##        filedat = filedat1.rstrip(".dat") + "_from_dat.dat"
##
##        if 1:
##                # use if 0 : to avoid reindexing already indexed grains
##                xyII = column_stack((data_xy, data_int))
##                xyII =  sort_peaks_decreasing_int(xyII,3)
##                print shape(xyII)
##                outputfile = open(filedat,'w')
##                outputfile.write(header)
##                np.savetxt(outputfile,xyII,fmt = '%.4f')
##                outputfile.close()
##        else :
##                xyII = loadtxt(filedat, skiprows = 1)
##
##        ngrains = 10
##        npeaks_LT = zeros(ngrains,int)
##        pixdev_LT = zeros(ngrains,float)
##        matstarlab_LT = zeros((ngrains,9), float)
##        filelist = []
##        for i in range(ngrains):   
##        #for i in range(5,10):    # adjust range to keep .fit files of already indexed grains
##                paramdetector_top = list(calib)
##                res1 = test_index_refine(filedat, paramdetector_top,\
##                                use_weights = False, proposed_matrix = None, \
##                              check_grain_presence = 1, paramtofit = "strain",\
##                                        elem_label = elem_label, grainnum = i+1,
##                                      remove_sat = 0, elim_worst_pixdev = 1, maxpixdev = 1.0,
##                                      spot_index_central = [0,1,2,3,4,5,6,7,8,9,10], nbmax_probed = 20, energy_max = 22,
##                                      rough_tolangle = 0.5 ,fine_tolangle = 0.2, Nb_criterium = 20,
##                                      NBRP = 1)
##
##                if res1 != 0 :                   
##                        filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
##                        filelist.append(filefit)
##                        matstarlab_LT[i,:], data_fit = F2TC.readlt_fit(filefit)
##                        nb_common_peaks,iscommon1,iscommon2 = find_common_peaks(xyII[:,:2], data_fit[:,-2:])
##                        ind1 = where(iscommon1 == 0)
##                        
##                        xyII = xyII[ind1[0],:]
##                        outputfile = open(filedat,'w')
##                        outputfile.write(header)
##                        np.savetxt(outputfile,xyII,fmt = '%.4f')
##                        outputfile.close()
##                        print shape(xyII)
##                else :
##                        break
##                
##        merge_fit_files_multigrain(filelist, removefiles = 0)
##        if filestr  is  not None :
##                print npeaks
##        print npeaks_LT
##        if filestr  is  not None :        
##                print dev_str[:,2]
##        print pixdev_LT
##        
##    return(0)

def test_LT_calib(filestr, 
                  elem_label, 
                  filedat1=None, 
                  filedattype="LT",
                  min_matLT = True):

    # use peak list either from XMAS STR file or from XMAS / LT dat file
    # use guess calib from XMAS STR file
    # use guess matrix from STR file for indexation, after orthonormalizing
    
#    data_str_all, matstarlab, calib, dev_str, npeaks 
    data_str_all, matstarlab, calib, dev_str, npeaks, strain, strain2 =\
        read_all_grains_str(filestr, min_matLT=min_matLT)
    matstarlab = matstarlab[0, :]
    print(shape(matstarlab))

    if filedat1  is None :
        filedat = filestr.rstrip(".str") + "_from_str.dat"
        xyII = column_stack((data_str_all[:, :2], data_str_all[:, 9:11]))
        header = "xexp yexp Ipeak Isub \n"
    else :
        filedat = filedat1.rstrip(".dat") + "_from_dat.dat"
        if filedattype == "XMAS" :
            data_xy, data_int = read_dat(filedat1, filetype="XMAS")
            header = "xexp yexp integr intens \n"
        elif filedattype == "LT" :
            data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype="LT")
            header = "xexp yexp Ipeak Isub \n"
        xyII = column_stack((data_xy, data_int))
            
    print(shape(xyII))

    outputfile = open(filedat, 'w')
    outputfile.write(header)
    np.savetxt(outputfile, xyII, fmt='%.4f')
    outputfile.close()

    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
    matstarlab1 = matstarlabOND * 1.0
    #matstarlab1 =matstarlab[i,:]*1.0
    matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
    paramdetector_top = list(calib)
    filefit, filecor, npeaks_LT, pixdev_LT, matLTmin, calibLT = \
            test_index_refine(filedat, paramdetector_top, \
                    use_weights=False, proposed_matrix=matLT3x3, \
                  check_grain_presence=None, paramtofit="calib", \
                            elem_label=elem_label, grainnum=1,
                            min_matLT = min_matLT)

    print("min_matLT = ", min_matLT)
    print("orthonormalize matstarlab before indexation")
    print("calib from str file")
    print_calib(calib)
    print("calib after LT fit")
    print_calib(calibLT)
    print("npeaks from str file")
    print(npeaks)
    print("npeaks after LT fit")
    print(npeaks_LT)
    print("pixdev from str file")
    print(dev_str[:, 2])
    print("pixdev after LT fit")
    print(pixdev_LT)
    return(0)

def test_LT_4(filedat1, elem_label, filestr=None, filefit=None, filedattype="XMAS"):

    # use peak list from peak search (XMAS or LT) 
    # use calib either from XMAS STR file or from LT fit file
    # no initial guess for indexation
    # add column isbadspot at end of .dat and .cor files to eliminate "intense grouped spots" from starting set
    # in test_index_refine

    # look for a maximum of ngrains grains
    ngrains = 10
    
    min_matLT = True
    if filestr  is  not None :
        data_str_all, matstarlab, calib, dev_str, npeaks = read_all_grains_str(filestr, min_matLT=min_matLT)
    elif filefit  is  not None :
        matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefit, readmore=True)

    if filedattype == "XMAS" :
        data_xy, data_int = read_dat(filedat1, filetype="XMAS")
        header = "xexp yexp integr intens isbadspot \n"
    elif filedattype == "LT" :
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
        outputfile = open(filedat, 'w')
        outputfile.write(header)
        np.savetxt(outputfile, xyII, fmt='%.4f')
        outputfile.close()
    else :
        xyII = loadtxt(filedat, skiprows=1)

    
    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filelist = []
    for i in range(ngrains):   
    #for i in range(7,10):    # adjust range to keep .fit files of already indexed grains
        paramdetector_top = list(calib)
        res1 = test_index_refine(filedat, paramdetector_top, \
                        use_weights=False, proposed_matrix=None, \
                      check_grain_presence=1, paramtofit="strain", \
                                elem_label=elem_label, grainnum=i + 1,
                              remove_sat=0, elim_worst_pixdev=1, maxpixdev=1.0,
                              spot_index_central=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nbmax_probed=20, energy_max=22,
                              rough_tolangle=0.5 , fine_tolangle=0.2, Nb_criterium=20,
                              NBRP=1, mark_bad_spots=1)

        if res1 != 0 :                   
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
            filelist.append(filefit)
            matstarlab_LT[i, :], data_fit = F2TC.readlt_fit(filefit)
            first_indexed = int(data_fit[0, 0])
            # spots before first_indexed marked as bad
            xyII[:first_indexed, -1] = 1
            nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(xyII[:, :2], data_fit[:, 6:8])
            ind1 = where(iscommon1 == 0)  
            xyII = xyII[ind1[0], :]
            xyII = sort_peaks_decreasing_int(xyII, 3)
            outputfile = open(filedat, 'w')
            outputfile.write(header)
            np.savetxt(outputfile, xyII, fmt='%.4f')
            outputfile.close()
            print(shape(xyII))
        else :
            break
        
    merge_fit_files_multigrain(filelist, removefiles=0)
    if filestr  is  not None :
            print(npeaks)
    print(npeaks_LT)
    if filestr  is  not None :        
            print(dev_str[:, 2])
    print(pixdev_LT)
        
    return(0)

if 0 : # to merge files after multi-grain indexation "by hand" 
    filelist = []
    for i in range(1,8):
            filelist.append(filepath + "Wmap_WB_14sep_d0_500MPa_0045_LT_0_from_dat_UWN_" + str(i) + ".fit")
    print(filelist)
    merge_fit_files_multigrain(filelist, removefiles = 0)
    jksqld

# uses a number of invisible parameters set in param.py
def serial_peak_search(filepathim, 
                       fileprefix, 
                       indimg, 
                       filesuffix, 
                       filepathout,
                       LT_VERSION = PAR.LT_REV_peak_search,
                       PixelNearRadius = PAR.PixelNearRadius,
                       IntensityThreshold = PAR.IntensityThreshold,
                       boxsize = PAR.boxsize,
                       position_definition = PAR.position_definition,
                       fit_peaks_gaussian = PAR.fit_peaks_gaussian,
                       xtol = PAR.xtol,
                       FitPixelDev = PAR.FitPixelDev,
                       local_maxima_search_method = PAR.local_maxima_search_method,
                       thresholdConvolve = PAR.thresholdConvolve,
                       number_of_digits_in_image_name = PAR.number_of_digits_in_image_name,
                       overwrite_peak_search = PAR.overwrite_peak_search,
                       CCDlabel=PAR.CCDlabel) :

    print("peak search in series of images (or single image)")

    
    npeaks = zeros(len(indimg), int)

    commentaire = "LT rev " + LT_VERSION + "\n# PixelNearRadius = " + str(PixelNearRadius) + \
            "\n# IntensityThreshold = " + str(IntensityThreshold) + \
            "\n# boxsize = " + str(boxsize) + "\n# position_definition = " + str(position_definition) + \
            "\n# fit_peaks_gaussian = " + str(fit_peaks_gaussian) + \
            "\n# xtol = " + str(xtol) + "\n# FitPixelDev = " + str(FitPixelDev) + \
            "\n# local_maxima_search_method = " + str(local_maxima_search_method) + \
            "\n# Threshold Convolve = " + str(thresholdConvolve) + "\n"
        
    print("peak search parameters :")
    print(commentaire)
    
    k = 0
    for i in indimg :
        print("k = ", k)
        print("i = ", i)
        #filename = filelist[i]
        filename = filepathim + fileprefix + imgnum_to_str(i, number_of_digits_in_image_name) + filesuffix
        print("image in :")
        print(filename)
        print("saving peak list in :")
        fileprefix1 = filepathout + fileprefix + imgnum_to_str(i, number_of_digits_in_image_name)
        filedat = fileprefix + imgnum_to_str(i, number_of_digits_in_image_name) + ".dat"
        #print os.listdir(filepathout)
        j = 0
        if not overwrite_peak_search :
            while (filedat in os.listdir(filepathout)):
                print("warning : change name to avoid overwrite")
                fileprefix2 = fileprefix + imgnum_to_str(i, number_of_digits_in_image_name) + "_new_" + str(j)
                filedat = fileprefix2 + ".dat"
                print(filepathout + filedat)
                j = j + 1

        if j > 0 :
            fileprefix1 = filepathout + fileprefix2
        else :
            print(filepathout + filedat)
            
        res1 = rmccd.PeakSearch(filename,
                                                    CCDLabel=CCDlabel,
                                                    PixelNearRadius=PixelNearRadius ,
                                                    IntensityThreshold=IntensityThreshold,
                                                    boxsize=boxsize,
                                                    position_definition=position_definition,
                                                    verbose=1,
                                                    fit_peaks_gaussian=fit_peaks_gaussian,
                                                    xtol=xtol,
                                                    return_histo=0,
                                                    FitPixelDev=FitPixelDev,
                                                    local_maxima_search_method = local_maxima_search_method,
                                                    thresholdConvolve=thresholdConvolve,
                                                    Saturation_value=DictLT.dict_CCD[CCDlabel][2],
                                                    Saturation_value_flatpeak=DictLT.dict_CCD[CCDlabel][2]
                                                    )
                                                    
        if res1 != False :
            Isorted, fitpeak, localpeak = res1   
                                                 
            npeaks[k] = shape(Isorted)[0]
    
            if shape(Isorted)[0] > 0 :
                
                RWASCII.writefile_Peaklist(fileprefix1,
                                        Isorted,
                                        overwrite=1,
                                        initialfilename=filename,
                                        comments=commentaire)
        k = k + 1

    print("indimg ", indimg)
    print("npeaks ", npeaks)
    return(0)

# uses a number of invisible parameters set in param.py
def index_refine_multigrain_one_image(filedat1, 
                                      elem_label, 
                                      filefitcalib = None, 
                                      ngrains=10, 
                                      proposed_matrix = None,
                                      datfile_type = "LT",
                                      check_grain_presence = PAR.check_grain_presence,
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
                                     CCDlabel = PAR.CCDlabel,
                                     calib = None,
                                     use_weights = False):

    # use peak list from peak search (LT) 
    # use calib from LT fit file
    # no initial guess for indexation
    # add column isbadspot at end of .dat and .cor files to eliminate "intense grouped spots" from starting set
    # in test_index_refine
    # look for a maximum of ngrains grains
    
    if filefitcalib  is  not None :
        matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefitcalib, 
                                                          readmore=True, 
                                                          verbose = 1)                                                         

    if datfile_type == "LT":
        data_xy, data_int, data_Ipixmax = read_dat(filedat1, 
                                                   filetype=datfile_type)
    else :
        data_xy, data_int = read_dat(filedat1, 
                                     filetype=datfile_type)
        
    if np.isscalar(data_xy[0]) : return(0)
    
    if shape(data_xy)[0]< 8 : return(0)
    
    header = "xexp yexp Ipeak Isub isbadspot \n"
    
    filedat = filedat1.rstrip(".dat") + "_t.dat"

    nspots = shape(data_xy)[0]
    isbadspot = zeros(nspots, int)
    
#    dxytol_duplicates = 0.1  # removed 23Jun14
    pixeldistance_remove_duplicates = 0.1 # added 23Jun14
    
    if 1 : # nettoyage des doublons 
    
        # use if 0 : pour continuer sur un _t.dat existant cad eviter de reindexer les grains deja indexes
        # remove duplicates
        print("remove duplicates in dat file")
        
        if 0 :  # removed 23Jun14
            nb_common_peaks,iscommon1,iscommon2 = \
                find_common_peaks(data_xy, data_xy, dxytol = dxytol_duplicates, verbose = 1)
            ind1 = where(iscommon1 == 1)
            # duplicates give iscommon1 > 1
            xyII = column_stack((data_xy[ind1[0]], data_int[ind1[0]], isbadspot[ind1[0]]))
            
        if 1 : # added 23Jun14
            purged_pklist, index_todelete = GT.purgeClosePoints2(data_xy, pixeldistance_remove_duplicates)
            xyII_all = column_stack((data_xy, data_int, isbadspot))
            xyII = np.delete(xyII_all, index_todelete, axis=0)
        
        xyII = sort_peaks_decreasing_int(xyII, 3)
#        print "xyII : \n", xyII
         
        print(shape(xyII))
        outputfile = open(filedat, 'w')
        outputfile.write(header)
        np.savetxt(outputfile, xyII, fmt='%.4f')
        outputfile.close()
    else :
        xyII = loadtxt(filedat, skiprows=1)

    npeaks_LT = zeros(ngrains, int)
    pixdev_LT = zeros(ngrains, float)
    matstarlab_LT = zeros((ngrains, 9), float)
    filefitmg = None
    filelist = []
    ngrains_found = 0
    for i in range(ngrains):   
    #for i in range(2,ngrains):    # adjust range to keep .fit files of already indexed grains
        paramdetector_top = list(calib)
        res1 = test_index_refine(filedat, 
                                 paramdetector_top,
                                 use_weights=use_weights, 
                                 proposed_matrix=proposed_matrix,
                                 check_grain_presence=check_grain_presence, 
                                 paramtofit="strain",
                                 elem_label=elem_label, 
                                 grainnum=i + 1,
                                 remove_sat=remove_sat, 
                                 elim_worst_pixdev=elim_worst_pixdev, 
                                 maxpixdev=maxpixdev,
                                 spot_index_central=spot_index_central,
                                 nbmax_probed=nbmax_probed,
                                 energy_max=energy_max,
                                 rough_tolangle=rough_tolangle,
                                 fine_tolangle=fine_tolangle,
                                 Nb_criterium=Nb_criterium,
                                 NBRP=NBRP, 
                                 mark_bad_spots=mark_bad_spots,
                                 CCDlabel = CCDlabel)
        if res1 != 0 :  
            ngrains_found = ngrains_found + 1                 
            filefit, filecor, npeaks_LT[i], pixdev_LT[i], matLTmin = res1
            filelist.append(filefit)
            print(filelist)
            print(filefit)

            if ngrains>1 :
                # TODO : adapt read
                matstarlab_LT[i, :], data_fit = read_any_fitfile_multigrain(filefit,
                                                    filetype = "MG_new")
                print("numbers of indexed spots : ", data_fit[:,0])
                first_indexed = int(data_fit[0, 0])
                # spots before first_indexed marked as bad
                xyII[:first_indexed, -1] = 1
    #            print "xyII"
    #            print xyII[0,:]
    #            print xyII[:, :2]
    #            print "data_fit"
    #            print data_fit[0,:]
    #            print  data_fit[:, 6:8]
                nb_common_peaks, iscommon1, iscommon2 = find_common_peaks(xyII[:, :2], data_fit[:, 6:8], verbose=0)
                ind1 = where(iscommon1 == 0)
#            print ind1
                #print xyII[:, :2]
                #print data_fit[:, -2:]
                #print ind1
                xyII = xyII[ind1[0], :]
                xyII = sort_peaks_decreasing_int(xyII, 3)
                outputfile = open(filedat, 'w')
                outputfile.write(header)
                np.savetxt(outputfile, xyII, fmt='%.4f')
                outputfile.close()
                print(shape(xyII))
        else :
            break

    print(ngrains_found)
    
    if (ngrains_found > 0) : 
        if ngrains > 1 :    
            filefitmg = merge_fit_files_multigrain(filelist, removefiles=1)
            #if ngrains==1 : pixdev_LT.reshape(len(pixdev_LT),1)
            print(npeaks_LT)
            print(pixdev_LT)
        else :
            filefitmg = filefit
    #print filefitmg, ngrains_found, npeaks_LT, pixdev_LT
    return(filefitmg, ngrains_found, npeaks_LT, pixdev_LT)
    
def spotlink_OR(matwithlatpar_inv_nm, calib, xyexp, cryst_struct, showall, dxytol):
    
    Emin = 5.0
    Emax = PAR.energy_max

    
    print("look for reflections in DAT file corresponding to given orientation matrix")
    print("pixel tolerance =", dxytol)
    print("matwithlatpar_inv_nm :\n", matwithlatpar_inv_nm)
    
    spotlist1 =  spotlist_gen(Emin, Emax, "top", matwithlatpar_inv_nm, cryst_struct, showall, \
             calib, pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1], remove_harmonics = "yes")
             
    hkl1 = spotlist1[:,:3]
    
    xytheor = spotlist1[:,6:8]
    
    res1 = find_common_peaks(xytheor, xyexp, dxytol=dxytol, \
                              verbose = 1, returnmore = 1, only_one_pair_per_peak = "yes")
    if res1!= 0 :                          
        nb_common_peaks, list_pairs, nsingle1, nsingle2, \
            list_single1, list_single2, iscommon1, iscommon2  = res1
        npairs = shape(list_pairs)[0]
        pixdev = zeros(npairs, float)
        print("ntheor, nexp, hkl, xy, pixdev :")
        for i in range(npairs):
            pixdev[i] = norme(xytheor[list_pairs[i,0]]-xyexp[list_pairs[i,1]])
            print(list_pairs[i,:], hkl1[list_pairs[i,0]], xyexp[list_pairs[i,1]], round(pixdev[i],3))

        return(list_pairs,hkl1[list_pairs[i,0]], xyexp[list_pairs[i,1]],pixdev)
        
    else : return(0)

# uses a number of invisible parameters set in param.py
def serial_index_refine_multigrain(filepathdat, 
                                   fileprefix, 
                                   indimg, 
                                   datfile_suffix,        
                                   filepathout, 
                                   filefitcalib = None, 
                                   filefitref = None,
                                   filefitmgref = None,
                                   gnumlocmgref = None,
                                   datfile_type = "LT",
                                   skip_existing_files = "no",
                                   ngrains_index_refine = PAR.ngrains_index_refine,
                                   number_of_digits_in_image_name = PAR.number_of_digits_in_image_name,
                                   add_str_index_refine = PAR.add_str_index_refine,
                                   overwrite_index_refine = PAR.overwrite_index_refine,
                                   elem_label_index_refine = PAR.elem_label_index_refine,
                                   check_grain_presence = PAR.check_grain_presence,
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
                                     CCDlabel = PAR.CCDlabel,
                                     proposed_matstarlab = None,
                                     calib = None,
                                     verbose = 1
                                   ) : 
   
    proposed_matrix = None
    
    if filefitref  is  not None :
        matstarlab, data_fit, calibref, pixdev = F2TC.readlt_fit(filefitref, readmore=True)
        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarlab1 = matstarlabOND * 1.0
        
        #matstarlab1 =matstarlab[i,:]*1.0
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
        proposed_matrix = matLT3x3
        
    if (filefitmgref  is  not None)&(gnumlocmgref  is  not None) :
        
        res2 = readlt_fit_mg(filefitmgref, verbose = 1, readmore = True)
#                print res1
        if res2 != 0 :
                gnumlist, npeaks1, indstart, matstarlab_all, data_fit1, calib1, pixdev1, strain6, euler = res2

        matstarlab = matstarlab_all[gnumlocmgref]
        print(matstarlab)
        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarlab1 = matstarlabOND * 1.0
        
        #matstarlab1 =matstarlab[i,:]*1.0
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
        proposed_matrix = matLT3x3   

    if proposed_matstarlab  is  not None : 
        matstarlabOND = matstarlab_to_matstarlabOND(proposed_matstarlab)
        matstarlab1 = matstarlabOND * 1.0
        
        #matstarlab1 =matstarlab[i,:]*1.0
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
        proposed_matrix = matLT3x3

    nimg = len(indimg)
    ngrains_found = zeros(nimg, int)
    npeaks = zeros((nimg, ngrains_index_refine), int)
    pixdev = zeros((nimg, ngrains_index_refine), float)
    
    k = 0
    for i in indimg :
        str_i =  imgnum_to_str(i, number_of_digits_in_image_name)
        #filename = filelist[i]
        filedat_nodir = fileprefix + str_i + datfile_suffix
        if not(filedat_nodir in os.listdir(filepathdat)) : 
            if verbose : print("img = %d : datfile does not exist" %(i))
            continue
        filedat1 = filepathdat  + filedat_nodir
#        print "i = ", i
        print("peak list in :")
        print(filedat1)
        #print "saving fit in :"
        filefit = fileprefix + str_i + add_str_index_refine + ".fit"
        #print os.listdir(filepathout)
        j = 0
        if not overwrite_index_refine :  # overwrite = 0
            while (filefit in os.listdir(filepathout)):
                print("warning : change name to avoid overwrite")
                filefit = fileprefix + str_i + add_str_index_refine + "_new_" + str(j) + ".fit"
                print(filepathout + filefit)
                j = j + 1
    
        else :
            if (filefit in os.listdir(filepathout)) :
                if skip_existing_files == "yes" :
                    print("file already exists, skip", filefit)
                    k = k+1
                    continue

        filefit_withdir = filepathout + filefit
        print(elem_label_index_refine)
        
        res1 = \
        index_refine_multigrain_one_image(filedat1, 
                                          elem_label_index_refine, 
                                          filefitcalib = filefitcalib, 
                                          ngrains= ngrains_index_refine, 
                                          proposed_matrix = proposed_matrix,
                                          datfile_type = datfile_type,
                                          check_grain_presence = check_grain_presence,
                                          remove_sat=remove_sat, 
                                         elim_worst_pixdev=elim_worst_pixdev, 
                                         maxpixdev=maxpixdev,
                                         spot_index_central=spot_index_central,
                                         nbmax_probed=nbmax_probed,
                                         energy_max=energy_max,
                                         rough_tolangle=rough_tolangle,
                                         fine_tolangle=fine_tolangle,
                                         Nb_criterium=Nb_criterium,
                                         NBRP=NBRP, 
                                         mark_bad_spots=mark_bad_spots,
                                         CCDlabel = CCDlabel,
                                         calib = calib)        
        
        print("k = ", k)    
        print(res1)   

        if res1 != 0 :
                   
            if PAR.ngrains_index_refine > 1 :
                filefitmg, ngrains_found[k], npeaks[k, :], pixdev[k, :] = res1
            else :
                filefitmg, ngrains_found[k], npeaks[k], pixdev[k] = res1    
    
            if filefitmg  is  not None :
                if filepathout != filepathdat :
                    os.rename(filefitmg, filefit_withdir)

        k = k + 1

        if nimg < 200 :
            print("indimg ", indimg[:k])
            print("ngrains_found ", ngrains_found[:k])
            if PAR.ngrains_index_refine > 1 :
                print("npeaks ", npeaks[:k])
                print("pixdev ", pixdev[:k])
            else :
                print("npeaks ", npeaks[:k,0])
                print("pixdev ", pixdev[:k,0])                

    return(ngrains_found, npeaks, pixdev)

# temporary comment out

def serial_index_refine_monograin_use_filemat(filepathdat, 
                                   fileprefix, 
                                   indimg, 
                                   datfile_suffix,        
                                   filepathout, 
#                                   filefitcalib = None, 
                                   filemat = None,  # list of mat from 2-spots calc
                                   datfile_type = "LT",
                                   skip_existing_files = "no",
#                                   ngrains_index_refine = 1,
                                   number_of_digits_in_image_name = PAR.number_of_digits_in_image_name,
                                   add_str_index_refine = PAR.add_str_index_refine,
                                   overwrite_index_refine = PAR.overwrite_index_refine,
                                   elem_label_index_refine = PAR.elem_label_index_refine,
#                                   check_grain_presence = PAR.check_grain_presence,
                                      remove_sat=PAR.remove_sat, 
                                     elim_worst_pixdev=PAR.elim_worst_pixdev, 
                                     maxpixdev=PAR.maxpixdev,
#                                     spot_index_central = None,
#                                     nbmax_probed= None,
                                     energy_max=PAR.energy_max,
#                                     rough_tolangle= None,
                                     fine_tolangle=PAR.fine_tolangle,
                                     Nb_criterium=PAR.Nb_criterium,
                                     NBRP=PAR.NBRP, 
#                                     mark_bad_spots=PAR.mark_bad_spots,
                                     CCDlabel = PAR.CCDlabel,
                                     calib = None,
                                     verbose = 1,
                                     use_weights = False
                                   ) : 
   
    data_list, listname, nameline0 = read_summary_file(filemat, 
                                                              verbose = 0)          
    data_list = np.array(data_list, dtype=float)
    
    nimg2 = np.shape(data_list)[0]               
    print("nimg2 = ", nimg2)
    indimg2 = listname.index("img") 
    img_list = np.array(data_list[:,indimg2],int)
#        print "img_list = ", img_list

    indmat = listname.index("matstarlab_0") 
    mat_list = np.array(data_list[:,indmat:indmat+9],float)
    
    dlat0_deg = np.array(DictLT.dict_Materials[elem_label_index_refine][1], dtype = float)
    dlat0_rad = deg_to_rad(dlat0_deg)

    #print dlat0.round(decimals = 4)
    #print dlat.round(decimals = 4)
    
    Bstar0 = dlat_to_Bstar(dlat0_rad)
    
    print("Bstar0 = \n", Bstar0.round(decimals = 4))
    
    invBstar = inv(Bstar0)

    nimg = len(indimg)
    ngrains_found = zeros(nimg, int)
    npeaks = zeros(nimg, int)
    pixdev = zeros(nimg, float)
    
    k = 0
    for i in indimg :
        str_i =  imgnum_to_str(i, number_of_digits_in_image_name)
        #filename = filelist[i]
        filedat_nodir = fileprefix + str_i + datfile_suffix
        if not(filedat_nodir in os.listdir(filepathdat)) : 
            if verbose : print("img = %d : datfile does not exist" %(i))
            continue
        filedat1 = filepathdat  + filedat_nodir
#        print "i = ", i
        print("peak list in :")
        print(filedat1)
        #print "saving fit in :"
        filefit = fileprefix + str_i + add_str_index_refine + ".fit"
        #print os.listdir(filepathout)
        j = 0
        if not overwrite_index_refine :  # overwrite = 0
            while (filefit in os.listdir(filepathout)):
                print("warning : change name to avoid overwrite")
                filefit = fileprefix + str_i + add_str_index_refine + "_new_" + str(j) + ".fit"
                print(filepathout + filefit)
                j = j + 1
    
        else :
            if (filefit in os.listdir(filepathout)) :
                if skip_existing_files == "yes" :
                    print("file already exists, skip", filefit)
                    k = k+1
                    continue

        filefit_withdir = filepathout + filefit
        print(elem_label_index_refine)
        
        ind1 = np.where(img_list == i)
        
        print(ind1[0])
        if len(ind1[0]) < 1 : 
            print("img %d not found in filemat" %(i))
            continue        
        
        matstarlabref = mat_list[ind1[0][0],:] 

        matLT3x3_UBB0 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlabref)
        
        matLT3x3_UB = dot(matLT3x3_UBB0, invBstar) 
        
        proposed_matrix = matLT3x3_UB
        
        proposed_matrix = proposed_matrix / norme(proposed_matrix[:,0]) # important
              
        res1 = \
        index_refine_multigrain_one_image(filedat1, 
                                          elem_label_index_refine, 
                                          filefitcalib = None, 
                                          ngrains= 1, 
                                          proposed_matrix = proposed_matrix,
                                          datfile_type = datfile_type,
                                          check_grain_presence = 1,
                                          remove_sat=remove_sat, 
                                         elim_worst_pixdev=elim_worst_pixdev, 
                                         maxpixdev=maxpixdev,
                                         spot_index_central= None,
                                         nbmax_probed= None,
                                         energy_max=energy_max,
                                         rough_tolangle=None,
                                         fine_tolangle=fine_tolangle,
                                         Nb_criterium=Nb_criterium,
                                         NBRP=NBRP, 
                                         mark_bad_spots= None,
                                         CCDlabel = CCDlabel,
                                         calib = calib,
                                         use_weights = use_weights)        
        
        print("k = ", k)    
        print(res1)   

        if res1 != 0 :
                   
            filefitmg, ngrains_found[k], npeaks[k], pixdev[k] = res1    
    
            if filefitmg  is  not None :
                if filepathout != filepathdat :
                    os.rename(filefitmg, filefit_withdir)

        k = k + 1
        
#        kmqsd

        if nimg < 200 :
            print("indimg ", indimg[:k])
            print("ngrains_found ", ngrains_found[:k])
            print("npeaks ", npeaks[:k])
            print("pixdev ", pixdev[:k])                

    return(ngrains_found, npeaks, pixdev)


# uses a number of invisible parameters set in param.py
def index_refine_calib_one_image(filedat1,
                                 filedet = None, 
                                 filefitcalib= None, \
                                 boolctrl = '1' * 8, 
                                 fixedcalib = None,
                                 elim_worst_pixdev=1, 
                                 maxpixdev=0.7,
                                 elem_label = "Ge",
                                 CCDlabel = "MARCCD165",
                                 remove_sat=PAR.remove_sat_calib, 
                                 spot_index_central=PAR.spot_index_central_calib,
                                 nbmax_probed=PAR.nbmax_probed_calib,
                                 energy_max=PAR.energy_max_calib,
                                 rough_tolangle=PAR.rough_tolangle_calib,
                                 fine_tolangle=PAR.fine_tolangle_calib,
                                 Nb_criterium=PAR.Nb_criterium_calib,
                                 NBRP=PAR.NBRP_calib
                                 ):

    if filedet  is  not None :
        calib, matstarlab = F2TC.readlt_det(filedet)
    if filefitcalib  is  not None :
        matstarlab, data_fit, calib, pixdev = F2TC.readlt_fit(filefitcalib, readmore=True)

    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
    matstarlab1 = matstarlabOND * 1.0
    #matstarlab1 =matstarlab[i,:]*1.0
    matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab1)
    if fixedcalib  is None :
        paramdetector_top = list(calib)
    else :
        paramdetector_top = list(fixedcalib)
    filefit, filecor, npeaks_LT, pixdev_LT, matLTmin, calibLT = \
            test_index_refine(filedat1, 
                                 paramdetector_top,
                                 use_weights=False, 
                                 proposed_matrix=matLT3x3,
                                 check_grain_presence=None, 
                                 paramtofit="calib",
                                 elem_label=elem_label, 
                                 grainnum=1,
                                 remove_sat=remove_sat, 
                                 elim_worst_pixdev=elim_worst_pixdev, 
                                 maxpixdev=maxpixdev,
                                 spot_index_central=spot_index_central,
                                 nbmax_probed=nbmax_probed,
                                 energy_max=energy_max,
                                 rough_tolangle=rough_tolangle,
                                 fine_tolangle=fine_tolangle,
                                 Nb_criterium=Nb_criterium,
                                 NBRP=NBRP, 
                                 boolctrl = boolctrl,
                                 CCDlabel = CCDlabel)
    
    return(filefit, npeaks_LT, pixdev_LT)

# uses a number of invisible parameters set in param_multigrain.py
def serial_index_refine_calib(filepathdat, 
                              fileprefix, 
                              indimg, 
                              filesuffix,
                              filefitcalib, 
                              filepathout, 
                              boolctrl = '1' * 8, 
                              fixedcalib = None, 
                              fixed_filefitcalib = "yes"):
    nimg = len(indimg)
    npeaks = zeros(nimg, int)
    pixdev = zeros(nimg, float)
    
    k = 0
    filefitcalib_local = filefitcalib
    
    for i in indimg :
        print("i = ", i)
        #filename = filelist[i]
        filedat1 = filepathdat + fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name) + filesuffix
        print("image in :")
        print(filedat1)
        #data_dat = loadtxt(filedat1, skiprows = 1)
        #print "saving fit in :"
        filefit = fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name) + PAR.add_str_index_refine + ".fit"
        #print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_index_refine :
            while (filefit in os.listdir(filepathout)):
                print("warning : change fit file name to avoid overwrite")
                filefit = fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name) + PAR.add_str_index_refine + "_new_" + str(j) + ".fit"
                print(filepathout + filefit)
                j = j + 1

        filefit_withdir = filepathout + filefit

        filefit, npeaks[k], pixdev[k] = index_refine_calib_one_image(filedat1,
                                                                    filefitcalib = filefitcalib_local,
                                                                     boolctrl = boolctrl,
                                                                     fixedcalib = fixedcalib)
        print(filefit_withdir)
        if filepathout != filepathdat :
            os.rename(filefit, filefit_withdir)

        if fixed_filefitcalib == "no" :
            # use result of fit as guess for next indexation
            filefitcalib_local = filefit_withdir
        
        k = k + 1

        print("indimg ", indimg[:k])
        print("npeaks ", npeaks[:k])
        print("pixdev ", pixdev[:k].round(decimals = 3))

    return(npeaks, pixdev)

def filter_peaks(filedat, 
                 maxpixdev = 0.7):
    
    k = 0    
    filedat2 = filedat.split('.dat')[0] + '_mpd.dat'
    
    outputfile = open(filedat2, 'w')
    f = open(filedat, 'r')
    try:
        for line in f:
            if (line[0] == "p")|(line[0] == "#"):
                outputfile.write(line)
            else :
                #print line
                toto = np.array(line.rstrip(PAR.cr_string).split(),dtype=float)
                if norme(toto[7:9])< maxpixdev : outputfile.write(line)
                else :
                    print("reject line : \n", line)
                    k = k+1
                
    finally:
        f.close()
    outputfile.write('\n')
    outputfile.close()

    print("input dat file : \n", filedat)
    print("filtered dat file: \n", filedat2)
    print("maxpixdev : ", maxpixdev)
    print("number of rejected lines : ", k)
                        
    return(filedat2)




import struct

# uses invisible parameters from param.py
def get_xyzech(filepathim, fileprefix, indimg, filesuffix, filepathout):

    nimg = len(indimg)
    data_list = zeros((nimg,6),float)
    fileim = ""

    # ROPER only - use hexedit to find hexadecimal location of stored floats

    kk = 0
    for k in indimg:

            data_list[kk, 0] = k
    
            fileim = filepathim + fileprefix + imgnum_to_str(k,PAR.number_of_digits_in_image_name) + filesuffix
            print(fileim)    
    
            f =open(fileim,'rb')

##            toto1 =""
##            f.seek(0x9B4)
##            for i in range(7) :
##                toto = struct.unpack("c",f.read(1)) 
##                toto1 = toto1 + toto[0]
##                
##            print toto1
##            data_list[kk,5] = float(toto1)

            toto1 =""
            f.seek(PAR.xech_offset)
            for i in range(7) :
                toto = struct.unpack("c",f.read(1)) 
                toto1 = toto1 + toto[0]    
            print(toto1)
            data_list[kk, 1] = float(toto1)

            toto1 =""
            f.seek(PAR.yech_offset)
            for i in range(7) :
                toto = struct.unpack("c",f.read(1)) 
                toto1 = toto1 + toto[0]    
            print(toto1)
            data_list[kk, 2] = float(toto1)

            toto1 =""
            f.seek(PAR.zech_offset)
            for i in range(7) :
                toto = struct.unpack("c",f.read(1)) 
                toto1 = toto1 + toto[0]    
            print(toto1)
            data_list[kk, 3] = float(toto1)
            
            toto1 =""
            f.seek(PAR.xech_offset)
            for i in range(40) :
                toto = struct.unpack("c",f.read(1)) 
                toto1 = toto1 + toto[0]    
            print(toto1)
            print(toto1.split()[3])
            data_list[kk, 4] = float(toto1.split()[3])
            data_list[kk, 5] = float(toto1.split()[4])    

            print("img, xech, yech, zech, mon4, lambda = \n", data_list[kk,:])

            kk = kk+1
            
    print(data_list)

    header = "img 0 , xech 1, yech 2, zech 3, mon4 4, lambda 5 \n" 
    outfilename = filepathout + "xyz_" + fileprefix + str(indimg[0]) + "_to_" + str(indimg[-1]) + ".dat"
    print(outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)
    np.savetxt(outputfile, data_list, fmt = "%.4f")
    outputfile.close()    

    return(0)
            
def build_xy_list_by_hand(filepathout, 
                          fileprefix, 
                          nx = 60, # number of intervals, cf spec scan
                          ny = 60, # number of intervals, cf spec scan
                          xfast = 1, # = 1 : scan x for successive y
                          yfast = 0, 
                          xstep = 0.5, 
                          ystep = -1,
                          shift_img = None,
                          imgnum_jump_at_start_of_each_linescan = None, # for hand-built mesh with successive dscans,
                          xystart = np.array([0.,0.])
                          ):

    # nx, ny : number of intervals (as in spec) 
    
        if imgnum_jump_at_start_of_each_linescan  is  not None :
            if xfast : 
                ncol = nx+1
                nlines = ny+1
            elif yfast :
                ncol = ny+1
                nlines = nx+1               
            n1 = imgnum_jump_at_start_of_each_linescan
            toto = np.arange(n1,ncol+n1)
            indimg = toto
            for k in range(1, nlines):
                toto = toto + ncol + n1
                indimg = np.row_stack((indimg, toto))
            nimg = np.shape(indimg)[0] * np.shape(indimg)[1]    
            indimg = indimg.reshape(nimg,)   
        else :
            indimg = arange((nx+1)*(ny+1))

        if yfast :
            xylist = zeros((nx+1,ny+1,2), float)
            for i in range(nx+1):
                xylist[i,:,0] = float(i)*xstep
            for j in range(ny+1):
                xylist[:,j,1] = float(j)*ystep
                
        if xfast :
            xylist = zeros((ny+1,nx+1,2), float)
            for i in range(nx+1):
                xylist[:,i,0] = float(i)*xstep
            for j in range(ny+1):
                xylist[j,:,1] = float(j)*ystep
        
        xylist_new = xylist.reshape((nx+1)*(ny+1),2)
        
        xylist_new = xylist_new + xystart

        if shift_img  is  not None :
            indimg = indimg + shift_img
        data_list = column_stack((indimg,xylist_new))
        
        print(data_list)
        
        header = "img 0 , xech 1, yech 2 \n" 
        outfilename = filepathout + "xy_" + fileprefix + str(indimg[0]) + "_to_" + str(indimg[-1]) + ".dat"
        print(outfilename)
        outputfile = open(outfilename,'w')
        outputfile.write(header)
        np.savetxt(outputfile, data_list, fmt = "%.4f")
        outputfile.close()
        
        return(outfilename)
 
def build_summary(indimg, 
                  filepathfit, 
                  fileprefix, 
                  filesuffix, 
                  filexyz, 
                  include_calib = "no",
                  fitfile_type = "MG",
                  subtract_xystart = "yes",
                  add_str = "",
                  simple_numbering = 0,
                  verbose = 1): #29May13

        # filexyz : img 0 , xech 1, yech 2, zech 3, mon4 4, lambda 5
        
        ncolstot = 21   # added maxpidev  04Dec13  # added stdpixdev 09Jan14
       
        list_col_names = ["dxymicrons", "matstarlab", "euler3"]
        number_col = array([2,9,3])
        
        list_col_names2 = ["img" , "gnumloc", "npeaks", "pixdev", "intensity"]

        for k in range(3) :        
            for i in range(number_col[k]):
                toto = list_col_names[k] + "_" + str(i)
                toto1 = [toto,]
                list_col_names2 = list_col_names2 + toto1
        
        list_col_names2 = list_col_names2 + ['maxpixdev','stdpixdev']
        
        if include_calib == "yes" :
            list_col_names2 = list_col_names2 + ["dd", 'xcen', "ycen", "xbet", "xgam"]
            ncolstot = ncolstot + 5
            filexyz = None
              
        #print list_col_names2
        header2 = ""
        for i in range(ncolstot):
            header2 = header2 +  list_col_names2[i] + " "
        header2 = header2 + "\n"
        print(header2)
        
        i = 0

        if filexyz  is  not None :
            toto = loadtxt(filexyz, skiprows = 1)
            xy = toto[:,1:3]
            imgxy = np.array(toto[:,0].round(decimals=0), dtype = int)
   
            if subtract_xystart == "yes" :  dxy = (xy - xy[0,:]) #*1000.0
            else :  dxy = xy * 1.
            
        else :  # on va reutiliser filexyz de toute facon dans plot_map et plot_all_grains_maps
            imgxy = np.array(indimg,dtype = int)
            dxy = zeros((len(indimg),2), float)
            print(imgxy)
            print(dxy)

        for kk in indimg :
                ind0 = where(imgxy == kk)
#                print "dxy = ", dxy[ind0[0],:]

                if simple_numbering :
                    strnum = str(kk)
                else :
                    strnum = imgnum_to_str(kk,PAR.number_of_digits_in_image_name)
                filefitmg_nopath = fileprefix + strnum + filesuffix
                filefitmg = filepathfit + filefitmg_nopath
#                print filefitmg
                if (filefitmg_nopath in os.listdir(filepathfit)) :
#                    print "yo"
                    
#                    res1 = readlt_fit_mg(filefitmg, 
#                                         verbose = 1, 
#                                         readmore = True,
#                                         first_line_ends_with = first_line_ends_with)

                    res1 = read_any_fitfitfile_multigrain(filefitmg, 
                                                          verbose=verbose, 
                                                          fitfile_type = fitfile_type) 
#                    print "res1 = " , res1

                    if res1 != 0 :
                            gnumlist, npeaks, indstart, matstarlab, data_fit, calib, meanpixdev, strain6, euler, ind_h_x_int_pixdev_Etheor = res1  

#                            gnumlist, npeaks, indstart, matstarlab, data_fit, calib, meanpixdev, strain6, euler = res1
    
#                            print type(gnumlist)
#                            print type(npeaks)

                            ind_h_x_int_pixdev_Etheor = np.array(ind_h_x_int_pixdev_Etheor,dtype = int)
                            
                            indint = ind_h_x_int_pixdev_Etheor[2]
                            indpixdev = ind_h_x_int_pixdev_Etheor[3]
                            
                            ngrains = len(gnumlist)
                            print("number of grains", ngrains)
                            #print indstart
                            intensity = zeros(ngrains,float)
                            maxpixdev = zeros(ngrains,float)
                            stdpixdev = zeros(ngrains,float)
                            if ngrains > 1 :
                                    for j in range(ngrains):
                                            range1 = arange(indstart[j],indstart[j]+npeaks[j])
                                            data_fit1 = data_fit[range1,:]
                                            intensity[j] = data_fit1[:PAR.nbtopspots,indint].mean()
                                            maxpixdev[j]= data_fit1[:,indpixdev].max()
                                            stdpixdev[j]= data_fit1[:,indpixdev].std()

                            else :
                                    intensity[0] = data_fit[:PAR.nbtopspots,indint].mean()
                                    maxpixdev[0]= data_fit[:,indpixdev].max()
                                    stdpixdev[0]= data_fit[:,indpixdev].std()
                                    euler = euler.reshape(1,3)
                                    calib = calib.reshape(1,5)
                                    
                            imnumlist = np.ones(ngrains,int)*kk
                            #                            print type(imnumlist)

                            dxylist = multiply(ones((ngrains,2),float), dxy[ind0[0],:])
#                            print dxylist
                            if 0 :    
                                print("shapes")
                                print(shape(imnumlist))
                                print(shape(gnumlist))
                                print(shape(dxylist))
                                print(shape(npeaks))
                                print(shape(meanpixdev))
                                print(shape(intensity))
                                print(shape(matstarlab))
                                print(shape(euler))
                                print(shape(calib))
                                
                            if 0 :
                                print(imnumlist[0])
                                print(gnumlist[0])
                                print(npeaks[0])
                                print(type(meanpixdev))
                                print(type(intensity))
                                print(type(dxylist))   
                                print(type(matstarlab))
                                print(type(euler))
                                kljkldsa
                                
                            toto = np.column_stack((imnumlist, gnumlist, \
                                                 npeaks, meanpixdev.round(decimals = 4), \
                                                 intensity.round(decimals = 2), dxylist.round(decimals = 5),\
                                                 matstarlab.round(decimals = 6), euler.round(decimals=3), 
                                                maxpixdev.round(decimals=4),stdpixdev.round(decimals=4) ))
                            if include_calib == "yes" :
                                toto = column_stack((toto,calib.round(decimals = 3)))   
                               
                            #print imnumlist
#                            print intensity
                            #print toto
                    else :
                            toto = zeros(ncolstot, float)
                            toto[0] = kk
    
                    if i == 0 :
                            allres = toto
                    else :
                            allres = row_stack((allres, toto))                   
                    i = i+1
                    
                else :                    
                    continue

        print(shape(allres))
        

        header = "img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, euler 16:19, maxpixdev 19, stdpixdev 20" 
        if include_calib == "yes" :
            header = header + ", calib 21:26 \n"
        else : header = header + "\n"
    
        outfilename = filepathfit + "summary_new_" + fileprefix + "img" + str(indimg[0]) + "to" + str(indimg[-1]) + add_str + ".dat"
        print(outfilename)
        outputfile = open(outfilename,'w')
        outputfile.write(header)
        outputfile.write(header2)
        np.savetxt(outputfile, allres, fmt = "%.6f")
        outputfile.close()
        
        allres = None
        toto = None
      
    #print toto

        return(outfilename)
 
def build_summary_2or4spots(filemat, 
                            filexyz, 
                            filepathout, 
                            fileprefix, 
                            calib = None, 
                            nspots = 2,
                            add_str = ""): 

        # filexyz : img 0 , xech 1, yech 2, zech 3, mon4 4, lambda 5
        
        # filemat : 
#            map_4spots  : img 0, xy1 1:3, xy2 3:5, xy3 5:7, xy4 7:9, matstarlab 9:18
#            map_2spots : img 0, xy1 1:3, xy2 3:5,  matstarlab 5:14, delta_alf 14

# img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, euler 16:19
 

        if nspots == 2 :
            data_mat = loadtxt(filemat, skiprows = 2)
        else :
            data_mat = loadtxt(filemat, skiprows = 1)
        
        print(shape(data_mat))

        imglist = np.array(data_mat[:,0], dtype = int)
        if nspots == 2 : 
            matstarlab = np.array(data_mat[:,5:14], dtype = float)
            dalf = np.array(data_mat[:,14], dtype = float)
        elif nspots == 4 : matstarlab = np.array(data_mat[:,9:18], dtype = float)
        
        nimg = len(imglist)
        
        ncolstot = 19
        
        list_col_names = ["dxymicrons", "matstarlab", "euler3"]
        number_col = array([2,9,3])
        
        list_col_names2 = ["img" , "gnumloc", "npeaks", "pixdev", "intensity"]

        for k in range(3) :        
            for i in range(number_col[k]):
                toto = list_col_names[k] + "_" + str(i)
                toto1 = [toto,]
                list_col_names2 = list_col_names2 + toto1
                
        if calib  is  not None :
            list_col_names2 = list_col_names2 + ["dd", 'xcen', "ycen", "xbet", "xgam"]
            ncolstot = ncolstot + 5
            
        if nspots == 2 :
            list_col_names2 = list_col_names2 + ["dalf",]
            ncolstot = ncolstot + 1
              
        #print list_col_names2
        header2 = ""
        for i in range(ncolstot):
            header2 = header2 +  list_col_names2[i] + " "
        header2 = header2 + "\n"
        print(header2)
        
        toto = loadtxt(filexyz, skiprows = 1)
        xy = toto[:,1:3]
        imgxy = toto[:,0]
        dxy = (xy - xy[0,:]) #*1000.0
        
        gnumlist = zeros(nimg, int)
        npeaks = ones(nimg, int)*25
        pixdev = zeros(nimg, float)
        intensity = zeros(nimg, float)
        calib1 = zeros((nimg,5),float)
        euler3 = zeros((nimg,3),float)
        dxylist = zeros((nimg,2),float)
        
        i = 0
        for kk in imglist :
            ind0 = where(imgxy == kk)
            print("dxy = ", dxy[ind0[0],:])
            dxylist[i,:] = dxy[ind0[0],:]
            if calib  is  not None :
                calib1[i,:]= calib*1.
            i = i+1
            
        allres = column_stack((imglist, gnumlist, npeaks, pixdev, intensity, dxylist, matstarlab, euler3))
        
        if calib  is  not None : allres = column_stack((allres,calib1))  
        
        if nspots == 2 : allres = column_stack((allres,dalf)) 

        print(shape(allres))
        

        header = "img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, euler 16:19" 
        if calib  is  not None :  
            header = header + ", calib 19:24"
            if nspots == 2 :  header = header + ", dalf 24" 
        else :
            if nspots == 2 :  header = header + ", dalf 19"
        header = header + "\n"
    
        if nspots == 2 :
            outfilename = filepathout + "summary_" + str(nspots) + "spots_" + fileprefix + "img" + str(imglist[0]) + "to" + str(imglist[-1]) + "_dalf"+ add_str + ".dat"
        elif nspots == 4 :
             outfilename = filepathout + "summary_" + str(nspots) + "spots_" + fileprefix + "img" + str(imglist[0]) + "to" + str(imglist[-1]) + add_str + ".dat"
           
        print(outfilename)
        outputfile = open(outfilename,'w')
        outputfile.write(header)
        outputfile.write(header2)
        np.savetxt(outputfile, allres, fmt = "%.6f")
        outputfile.close()
        
    #print toto

        return(outfilename)
       
def read_summary_file(filesum, 
                      read_all_cols = "yes",
                      list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev', 'intensity', 
                                            'dxymicrons_0', 'dxymicrons_1', 
 'matstarlab_0', 'matstarlab_1', 'matstarlab_2', 'matstarlab_3', 'matstarlab_4','matstarlab_5', 'matstarlab_6', 'matstarlab_7', 'matstarlab_8', 
 'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
 'euler3_0', 'euler3_1', 'euler3_2', 
 'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
 'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
 'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
 'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
 'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
 'res_shear_stress_0', 'res_shear_stress_1', 'res_shear_stress_2', 'res_shear_stress_3', 'res_shear_stress_4', 'res_shear_stress_5', 'res_shear_stress_6', 'res_shear_stress_7', 'res_shear_stress_8', 'res_shear_stress_9', 'res_shear_stress_10', 'res_shear_stress_11',
 'max_rss', 'von_mises'],
                     verbose = 1):

    #29May13
    if verbose :
        print("reading summary file")
        print("first two lines :")
    f = open(filesum, 'r')
    i = 0
    try:
        for line in f:
            if i == 0 : nameline0 = line.rstrip("  "+PAR.cr_string)
            if i == 1 : nameline1 = line.rstrip(PAR.cr_string)
            i = i+1
            if i>2 : break
    finally:
        f.close() 
    
    if verbose :    
        print(nameline0)   
        print(nameline1)
    listname = nameline1.split()
    
    data_sum = loadtxt(filesum, skiprows = 2)   
    
    if read_all_cols == "yes" :
        if verbose :
            print("shape(data_sum) = ", shape(data_sum))
        if len(listname) != shape(data_sum)[1] :
            print("list of column names does not match number of columns !")
            fjkldsf    
            
        return(data_sum, listname, nameline0)
        
    else :     
        if verbose : print(len(listname))
        ncol = len(list_column_names)
        
        ind0 = zeros(ncol, int)
    
        for i in range(ncol):    
            ind0[i] = listname.index(list_column_names[i])
    
        if verbose : print(ind0)
        
        data_sum_select_col = data_sum[:, ind0]
        
        if verbose :
            print(shape(data_sum))
            print(shape(data_sum_select_col))
            print(filesum)
            print(list_column_names)
            print(data_sum_select_col[:5,:])
        
        print(len(list_column_names))
        
        return(data_sum_select_col, list_column_names, nameline0)

##### ********************************************************************
def read_filexyz(filexyz) :
    
    print("enter read_filexyz") #, asctime()
    
    f = open(filexyz, 'r')
    i = 0
    try:
        for line in f:
            if line.startswith( "#grid_order") :
                toto = line.rstrip(PAR.cr_string).split(":")
                grid_order = str(toto[1])
                print("grid_order = ", grid_order)
            if line.startswith( "#dxystep") :
                toto = line.rstrip(PAR.cr_string).replace('[', '').replace(']', '').split(":")
                toto1 = toto[1].split()
                dxystep = np.array(toto1, float)
                print("dxystep= ", dxystep)
            if line.startswith( "#nlines") :
                toto = line.rstrip(PAR.cr_string).split(":")
                nlines = int(toto[1])
                print("nlines= ", nlines)     
            if line.startswith( "#ncol") :
                toto = line.rstrip(PAR.cr_string).split(":")
                ncol = int(toto[1])
                print("ncol= ", ncol)
            i = i+1
    finally:
        f.close()
    imgxyz = np.loadtxt(filexyz)
    
#    print "exit read_filexyz", asctime()
    
    return(imgxyz,grid_order,dxystep,nlines,ncol)
      
def add_columns_to_summary_file(filesum, 
                                elem_label = "Ge", 
                                filestf = None,
                                verbose = 0, 
                                single_grain = 0, 
                                filefitref_for_orientation = None, # seulement pour single_grain = 1
                                fitfile_type = "MG", # seulement pour filefitref_for_orientation  is  not None
                                include_strain = 1, # 0 seulement pour mat2spots ou fit calib ou EBSD
                                # les 4 options suivantes seulement pour
                                #  single_grain = 1 
                                # et filefitref_for_orientation = None
                                filter_mean_matrix_by_pixdev_and_npeaks = 0, 
                                maxpixdev_for_mean_matrix = 0.25,
                                minnpeaks_for_mean_matrix = 20,
                                filter_mean_matrix_by_intensity = 0, 
                                minintensity_for_mean_matrix = 20000.,
                                filter_mean_matrix_by_matrix_components = 0,
                                component_num_for_mean_matrix = 0,
                                component_range_for_mean_matrix = [-1.,1.],
                                include_rgb = 1,
                                imax = 1e7,
                                filexyz_5col = None
                                ): #29May13
    
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
        
        if single_grain :  # seulement pour les analyses mono-grain
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

    if single_grain :
        include_rgb = 0
    
    print("inside add_columns_to_summary_file")
    print("PAR.omega_sample_frame = ", PAR.omega_sample_frame)
    print("PAR.mat_from_lab_to_sample_frame = \n", PAR.mat_from_lab_to_sample_frame)
    
    if elem_label  is  not None :
        init_numbers_for_crystal_opsym_and_first_stereo_sector(elem_label = elem_label)
    
    data_1, list_column_names, nameline0 = read_summary_file(filesum)
    
    data_1 = np.array(data_1, dtype=float)

    list_col_names2 = list_column_names
    
    list_col_names_orient = ["rgb_x_sample", "rgb_y_sample", "rgb_z_sample",
                      "rgb_x_lab", "rgb_y_lab", "rgb_z_lab"]
                                           
    number_col_orient = array([3,3,3,3,3,3])
    
    if include_rgb :
        for k in range(len(number_col_orient)) :        
            for i in range(number_col_orient[k]):
                toto = list_col_names_orient[k] + "_" + str(i)
                list_col_names2.append(toto)
            
    if include_strain :
        
        list_col_names_strain = ["strain6_crystal","strain6_sample",
                      "stress6_crystal", "stress6_sample", "res_shear_stress",
                      "max_rss", "von_mises", "eq_strain"]    
        number_col_strain = array([6,6,6,6,12])
       
        for k in range(len(number_col_strain)) :        
            for i in range(number_col_strain[k]):
                toto = list_col_names_strain[k] + "_" + str(i)
                list_col_names2.append(toto)
    
        for k in range(len(number_col_strain),len(number_col_strain)+3):
            list_col_names2.append(list_col_names_strain[k])
        
    #print list_col_names2
    header2 = ""
    for i in range(len(list_col_names2)):
        header2 = header2 +  list_col_names2[i] + " "
    
    header = nameline0 
    
    if include_rgb :
        header += ", rgb_x_sample, rgb_y_sample, rgb_z_sample, rgb_x_lab, rgb_y_lab, rgb_z_lab" 
    
    if include_strain :    
        header += \
", strain6_crystal,  strain6_sample, stress6_crystal, stress6_sample, res_shear_stress_12, max_rss, von_mises, eq_strain" 

    if single_grain :
        header +=  ", misorientation_angle, w_mrad "
        header2 += "misorientation_angle w_mrad_0 w_mrad_1 w_mrad_2 "

    if filexyz_5col  is  not None :
        header += ", ijpos, ind_link_img"
        header2 += "ijpos_0, ijpos_1, ind_link_img "

    header = header + "\n"
    header2 = header2 + "\n"
        
    print(header)
    print(header2)
    print(header2.split())
    
    add_str = "_add_columns"
    
    if elem_label  is  not None :        
        schmid_tensors = glide_systems_to_schmid_tensors(verbose = 0)
    
    if filestf  is  not None : 
        c_tensor = read_stiffness_file(filestf)
        
    xsample_sample_coord = array([1.,0.,0.])
    ysample_sample_coord = array([0.,1.,0.])
    zsample_sample_coord = array([0.,0.,1.])
    
    if (PAR.omega_sample_frame  is  not None)&(PAR.mat_from_lab_to_sample_frame  is None) : # deprecated - only for retrocompatibility
        omega =  PAR.omega_sample_frame *PI/180.0
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        mat_from_lab_to_sample_frame = array([[1.0,0.0,0.0],[0.0,np.cos(omega),np.sin(omega)],[0.0,-np.sin(omega),np.cos(omega)]])
    
    else :   mat_from_lab_to_sample_frame =  PAR.mat_from_lab_to_sample_frame 
    
    mat_from_sample_to_lab_frame = transpose(mat_from_lab_to_sample_frame)
      
#    omegarad = PAR.omega_sample_frame * PI/180.
#    ylab_sample_coord = array([0.,np.cos(omegarad),-np.sin(omegarad)])
#    zlab_sample_coord = array([0.,np.sin(omegarad),np.cos(omegarad)])

    xlab_sample_coord = mat_from_sample_to_lab_frame[0,:]
    ylab_sample_coord = mat_from_sample_to_lab_frame[1,:]
    zlab_sample_coord = mat_from_sample_to_lab_frame[2,:]
    
    print("x y z sample - sample coord : ", xsample_sample_coord, ysample_sample_coord, zsample_sample_coord)
    print("x y z lab - sample coord : \n", xlab_sample_coord, "\n", ylab_sample_coord, "\n", zlab_sample_coord)
    
    numig = shape(data_1)[0]
    
    #numig = 10
    
    if include_rgb :
        rgb_x = zeros((numig,3),float)
        rgb_y = zeros((numig,3),float)
        rgb_z = zeros((numig,3),float)
        rgb_xlab = zeros((numig,3),float)
        rgb_ylab = zeros((numig,3),float)
        rgb_zlab = zeros((numig,3),float)
        
    if include_strain :
        epsp_crystal = zeros((numig,6),float)
        epsp_sample = zeros((numig,6),float)
        sigma_crystal = zeros((numig,6),float)
        sigma_sample = zeros((numig,6),float)
        tau1 = zeros((numig,12),float)
        von_mises = zeros(numig,float)
        maxrss = zeros(numig,float)
        eq_strain = zeros(numig,float)
    
#    img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, strain6_crystal 16:22, euler 22:25
    indimg = list_column_names.index("img") # 3
    indpixdev = list_column_names.index("pixdev") # 3
    indnpeaks = list_column_names.index("npeaks") # 2
    indmatstart = list_column_names.index("matstarlab_0") # 7
    indintensity = list_column_names.index("intensity") # 4
    print(indpixdev, indnpeaks, indmatstart, indintensity)
    indmat = np.arange(indmatstart,indmatstart + 9)
    img_list = data_1[:,indimg] 
    pixdev_list = data_1[:,indpixdev]
    npeaks_list = data_1[:,indnpeaks]
    intensity_list = data_1[:,indintensity]
    mat_list = data_1[:,indmat]
    
    if single_grain :
        cond_npeaks_0 = (npeaks_list > 0.)
        cond_total = cond_npeaks_0 * 1
#        indfilt2 = where(npeaks_list > 0.0) # pour raccourcir le summary a la fin
        misorientation_angle = zeros(numig, float)
        omegaxyz = zeros((numig,3), float)
        
        if filefitref_for_orientation  is None :
            print("start filtering mean matrix", asctime())
                                
            if filter_mean_matrix_by_pixdev_and_npeaks :     
                print("filter mean matrix by pixdev and npeaks")
                print("maxpixdev_for_mean_matrix = ", maxpixdev_for_mean_matrix)
                print("minnpeaks_for_mean_matrix = ", minnpeaks_for_mean_matrix)
                cond_pixdev = (pixdev_list < maxpixdev_for_mean_matrix)
                cond_npeaks_min = (npeaks_list > minnpeaks_for_mean_matrix)
                cond_total = cond_total * cond_pixdev * cond_npeaks_min
            
            if filter_mean_matrix_by_intensity :     
                print("filter mean matrix by intensity")
                print("minintensity_for_mean_matrix = ", minintensity_for_mean_matrix)
                cond_intensity = (intensity_list > minintensity_for_mean_matrix)
                cond_total = cond_total * cond_intensity
            
            if filter_mean_matrix_by_matrix_components :
                print("filter mean matrix by matrix component")
                print("component_num_for_mean_matrix = ", component_num_for_mean_matrix)
                print("component_range_for_mean_matrix = ", component_range_for_mean_matrix)
                matcomp_list = data_1[:,indmat[component_num_for_mean_matrix]]
                cond_matcomp_min = (matcomp_list > component_range_for_mean_matrix[0])
                cond_matcomp_max = (matcomp_list < component_range_for_mean_matrix[1])
                cond_total = cond_total * cond_matcomp_min * cond_matcomp_max
                                     
            indfilt = where(cond_total > 0) 
            matstarlabref = (mat_list[indfilt[0]]).mean(axis=0) 
            print("shape(data_1)[0] = ", shape(data_1)[0])
            print("number of points used to calculate matmean", len(indfilt[0]))  
            
            print("end filtering mean matrix", asctime())
            print("matstarlabref = ", list(matstarlabref))
#            jlqsdqsd
            # TO REMOVE
#            matmean = ((mat_list[indfilt[0]])[-10:]).mean(axis=0)   # test pour data Keckes
            
        elif filefitref_for_orientation  is  not None :
#            matstarlabref, data_fit, calib, pixdev = F2TC.readlt_fit(filefitref_for_orientation, readmore = True)
            res1 = read_any_fitfitfile_multigrain(filefitref_for_orientation, 
                                                  verbose=verbose, 
                                                  fitfile_type = fitfile_type) 
#                    print "res1 = " , res1

            gnumlist, npeaks, indstart, matstarlab_all, data_fit_all, calib, meanpixdev, strain6, euler, ind_h_x_int_pixdev  = res1  
            matstarlabref = matstarlab_all[0,:]    
        
#        matmean3x3 = GT.matline_to_mat3x3(matmean) 

    numig2 = min(numig, imax)


    k = 0
    
    indfilt2 = where(cond_npeaks_0[:numig2] > 0)
    
    for i in indfilt2[0] :
        print("ig : ", i) #,  "img : ", img_list[i]
        matstarlab = mat_list[i,:]
        
        if include_rgb :
            #print "x"
            matstarlabnew, transfmat, rgb_x[i,:] = \
            matstarlab_to_orientation_color_rgb(matstarlab, xsample_sample_coord, elem_label = elem_label)
#            rgb_xlab[i,:] = rgb_x[i,:]*1.
            matstarlabnew, transfmat, rgb_xlab[i,:] = \
            matstarlab_to_orientation_color_rgb(matstarlab, xlab_sample_coord, elem_label = elem_label)
            #print "y"
            matstarlabnew, transfmat, rgb_y[i,:] = \
            matstarlab_to_orientation_color_rgb(matstarlab, ysample_sample_coord, elem_label = elem_label)
            matstarlabnew, transfmat, rgb_ylab[i,:] = \
            matstarlab_to_orientation_color_rgb(matstarlab, ylab_sample_coord, elem_label = elem_label)
            #print "z"
            matstarlabnew, transfmat, rgb_z[i,:] = \
            matstarlab_to_orientation_color_rgb(matstarlab, zsample_sample_coord, elem_label = elem_label)
            matstarlabnew, transfmat, rgb_zlab[i,:] = \
            matstarlab_to_orientation_color_rgb(matstarlab, zlab_sample_coord, elem_label = elem_label)

        if include_strain :  
            epsp_sample[i,:], epsp_crystal[i,:] = matstarlab_to_deviatoric_strain_sample(matstarlab, 
                                                                    omega0 = None, # was PAR.omega_sample_frame, 
                                                                    mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame,
                                                                    version = 2,
                                                                    returnmore = True,
                                                                    elem_label = elem_label)
             
            sigma_crystal[i,:] = deviatoric_strain_crystal_to_stress_crystal(c_tensor, epsp_crystal[i,:])
            sigma_sample[i,:] = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab,
                                                                  sigma_crystal[i,:],
                                                                  omega0 = None, # was PAR.omega_sample_frame,
                                                                  mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)
                                                                  
            von_mises[i] = deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal[i,:])
                                                                                   
            tau1[i,:] = deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(sigma_crystal[i,:], schmid_tensors)
            maxrss[i]=abs(tau1[i,:]).max()
            
            eq_strain[i] = deviatoric_strain_crystal_to_equivalent_strain(epsp_crystal[i,:])
        
        if single_grain :  
#                mat2 = GT.matline_to_mat3x3(matstarlab)
#                vec_crystal, vec_lab, misorientation_angle[i] = twomat_to_rotation(matmean3x3,mat2, verbose = 0)
   
            if 0 :
                vecRodrigues_sample, misorientation_angle[i] = twomat_to_rotation_Emeric(matstarlabref,matstarlab)
                omegaxyz[i,:] = vecRodrigues_sample * 2. * 1000. # unites = mrad 
 
           # misorientation_angle : unites = degres
            
            if 1 : # calcul sans orthonormalisation                    
#                
                RxRyRz_mrad, dRxRyRz_mrad, ang1_mrad, dang1_mrad, dLxLyLz_mrad = \
                    twomat_to_RxRyRz_sample_large_strain(matstarlabref, 
                                                         matstarlab,
                                                         omega = None, # was PAR.omega_sample_frame
                                                         mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame
                                                        )
#                
#                # TODO : add new columns differential strain pour single grain
##                epsp_sample_diff[i,0:3] = dLxLyLz_mrad
##                epsp_sample_diff[i,3:6] = dRxRyRz_mrad                
#                    
                omegaxyz[i,:] = RxRyRz_mrad
                
                misorientation_angle[i] = ang1_mrad / 1000. * 180./PI
   
#                print round(misorientation_angle[i], 3), omegaxyz[i,:].round(decimals = 2)
#                if k == 5 : return()


                   
        if verbose : 
            print(matstarlab)
            if include_strain :
                print("deviatoric strain crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)")
                print(epsp_crystal.round(decimals=2))                                                                    
                print("deviatoric strain sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-3 units)")
                print(epsp_sample[i,:].round(decimals=2))
                
                print("deviatoric stress crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (100 MPa units)")
                print(sigma_crystal[i,:].round(decimals=2))
                                                                        
                print("deviatoric stress sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (100 MPa units)")
                print(sigma_sample[i,:].round(decimals=2))   
            
                print("Von Mises equivalent Stress (100 MPa units)", round(von_mises[i],3))                 
                print("RSS resolved shear stresses on glide planes (100 MPa units) : ")
                print(tau1[i,:].round(decimals = 3))
                print("Max RSS : ", round(maxrss[i],3))
                print("Equivalent Strain (1e-3 units)", round(eq_strain[i],3)) 
    k = k+1
        
    # numig here for debug with smaller numig
    
    data_list = data_1[:numig2,:]*1.
    
    if include_rgb :
        data_list = column_stack((data_list, \
                                  rgb_x[:numig2,:], rgb_y[:numig2,:], rgb_z[:numig2,:], rgb_xlab[:numig2,:], rgb_ylab[:numig2,:], rgb_zlab[:numig2,:]))
    
    if include_strain :
        data_list = column_stack((data_list, epsp_crystal[:numig2,:], epsp_sample[:numig2,:], \
                    sigma_crystal[:numig2,:],sigma_sample[:numig2,:],tau1[:numig2,:],maxrss[:numig2],von_mises[:numig2],eq_strain[:numig2]))
                    
    if single_grain : 
        data_list = column_stack((data_list, misorientation_angle[:numig2], omegaxyz[:numig2,:])) 
        data_list = data_list[indfilt2[0],:]  # enleve les images avec zero grain indexe
    
        if filefitref_for_orientation  is  not None :
            add_str = add_str + "_use_ref_matrix"
        else :
            add_str = add_str + "_use_mean_matrix"
            # TO REMOVE
#    add_str = add_str + "_use_mean_10_points"
    
    if filexyz_5col  is  not None :
        data_list = column_stack((data_list, ij_in_filesum, ind_link_img))
    
    outfilesum = filesum.rstrip(".dat") + add_str + ".dat"
    print(outfilesum)
    outputfile = open(outfilesum,'w')
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, data_list, fmt = "%.6f")
    outputfile.close()

    return(outfilesum)
        
def plot_orientation_triangle_color_code():
    
    # for cubic structure
        
    # plot orientation color scale in stereographic projection
#    p.rcParams['savefig.bbox'] = None # "tight"
    p.figure(figsize = (8,8))

    numrand1 = 50
    range001 = np.arange(numrand1+1)
    range001 = np.array(range001, dtype=float)/numrand1

    angrange = range001*1.
    
    for i in range(numrand1+1):
        for j in range(numrand1+1):
            col1 = zeros(3,float)
            uq = (1.-range001[i])*PAR.uqref_cr[:,0] + range001[i]* (angrange[j]*PAR.uqref_cr[:,1] + (1.-angrange[j])*PAR.uqref_cr[:,2])
            uq = uq/norme(uq)            

            qsxy = hkl_to_xystereo(uq, down_axis = [0.,-1.,0.])
            
            # RGB coordinates
            
            rgb_pole = uq_cr_to_orientation_color_rgb(uq, cryst_struct = "cubic")

            rgb_pole = rgb_pole.clip(min=0.0,max=1.0)
            
            p.plot(qsxy[0],qsxy[1],marker = 'o', markerfacecolor = rgb_pole, markeredgecolor = rgb_pole, markersize = 5)     

    p.xlim(-0.1,0.5)
    p.ylim(-0.1,0.5)

    return(0)
 
#from matplotlib import pyplot, mpl

import matplotlib as mpl
import matplotlib.pyplot as pyplot

def plot_negative_positive_colorbar(bar_legend = "strain (0.1 %)", 
                                    minbar = -0.2, 
                                    maxbar = 0.2,
                                    orientation = "horizontal",
                                    aspect_ratio = "thin") :
    
    # strain (0.1%)
    #'stress (100 MPa)'
    
    
    # Make a colorbar as a separate figure. (for strain maps)
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 1.0, 0.0)),
             'green': ((0.0, 0.0, 1.0),
                       (0.5, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
              'blue': ((0.0, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                       (1.0, 0.0, 0.0))}
    
    '''
    Make a colorbar as a separate figure.
    '''
    #cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    # Make a figure and axes with dimensions as desired.

    if aspect_ratio == "thin" : length1 = 0.5
    elif aspect_ratio == "thick" : length1 = 0.3

    if orientation == "vertical" :
        fig = pyplot.figure(figsize=(3,8)) # vertical bar
        ax1 = fig.add_axes([0.05, 0.05, 0.15, length1])  # vertical bar
    #ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    elif orientation == "horizontal" :
        fig = pyplot.figure(figsize=(8,3)) # horizontal bar
        ax1 = fig.add_axes([0.05, 0.3, length1, 0.15])  # horizontal bar

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    #cmap = mpl.cm.Greys
#    cmap = mpl.cm.PiYG
    cmap = mpl.cm.RdBu_r
    
    #norm = mpl.colors.Normalize(vmin=0, vmax=0.25)
    #norm = mpl.colors.Normalize(vmin=0.15, vmax=0.45)
    norm = mpl.colors.Normalize(vmin=minbar, vmax=maxbar)
    #cmap.set_over(color = "r")
    cmap.set_over(color = "r") # 
    cmap.set_under(color = "b")

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                       norm=norm,
                                        extend = 'both',
                                        #ticks = [0.,0.1,0.2, 0.25],
                                        ticks = [minbar, minbar/2., 0.,maxbar/2., maxbar],
                                        spacing = "proportional",
                                        orientation=orientation)#'horizontal')

    #cb1.set_label('rotation angle (degrees)', fontsize = 20)

    if bar_legend  is  not None :
        cb1.set_label(bar_legend, fontsize = 20)
        
    if orientation == "vertical" :  
        c0 = mpl.artist.getp(cb1.ax, 'ymajorticklabels') # vertical bar
    elif orientation == "horizontal" :     
        c0 = mpl.artist.getp(cb1.ax, 'xmajorticklabels') # horizontal bar
        
    mpl.artist.setp(c0, fontsize=20)

    return()
    
def plot_positive_colorbar(vmin = 0.0, 
                                    vmax = 0.15, 
                                    ticks = [0.,0.05,0.1,0.15], 
                                    bar_legend = 'intensity',
                                    orientation = "vertical",
                                    aspect_ratio = "thin"
                                    ):   
    '''
    Make a colorbar as a separate figure.
    '''
    # Make a figure and axes with dimensions as desired.
    
    if aspect_ratio == "thin" : length1 = 0.5
    else : length1 = 0.3
       
    if orientation == "vertical" :
        fig = pyplot.figure(figsize=(3,8))        
        ax1 = fig.add_axes([0.05, 0.05, 0.15, length1])
    elif orientation == "horizontal":
        fig = pyplot.figure(figsize=(8,3)) # horizontal bar
        ax1 = fig.add_axes([0.05, 0.3, length1, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.Greys
    #norm = mpl.colors.Normalize(vmin=0, vmax=0.25)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap.set_over(color = 'r')
    
    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                       norm=norm,
                                        extend = "max",
                                        #ticks = [0.,0.1,0.2, 0.25],
                                        ticks = ticks,
                                        spacing = "proportional",
                                       orientation= orientation)

    if bar_legend  is  not None :
        cb1.set_label(bar_legend, fontsize = 20)
    
    if orientation == "vertical" :    
        c0 = mpl.artist.getp(cb1.ax, 'ymajorticklabels') # vertical bar
    elif orientation == "horizontal" :
        c0 = mpl.artist.getp(cb1.ax, 'xmajorticklabels') # horizontal bar
        
    mpl.artist.setp(c0, fontsize=20)


    return(0)   
    
def plot_positive_inverted_colorbar(vmin = 0.0, 
                                    vmax = 0.15, 
                                    ticks = [0.,0.05,0.1,0.15], 
                                    bar_legend = 'rotation angle (degrees)',
                                    orientation = "vertical",
                                    aspect_ratio = "thin",
                                    show_under = 0                                    
                                    ):   
    '''
    Make a colorbar as a separate figure.
    '''
    # Make a figure and axes with dimensions as desired.
    
    if aspect_ratio == "thin" : length1 = 0.5
    else : length1 = 0.3
       
    if orientation == "vertical" :
        fig = pyplot.figure(figsize=(3,8))        
        ax1 = fig.add_axes([0.05, 0.05, 0.15, length1])
    elif orientation == "horizontal":
        fig = pyplot.figure(figsize=(8,3)) # horizontal bar
        ax1 = fig.add_axes([0.05, 0.3, length1, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.Greys
    #norm = mpl.colors.Normalize(vmin=0, vmax=0.25)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap.set_over(color = 'r')
    extend = "max"
    if show_under :
        cmap.set_under(color = 'b')
        extend = "both"
        
    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                       norm=norm,
                                        extend = extend,
                                        #ticks = [0.,0.1,0.2, 0.25],
                                        ticks = ticks,
                                        spacing = "proportional",
                                       orientation= orientation)

    if bar_legend  is  not None :
        cb1.set_label(bar_legend, fontsize = 20)
    
    if orientation == "vertical" :    
        c0 = mpl.artist.getp(cb1.ax, 'ymajorticklabels') # vertical bar
    elif orientation == "horizontal" :
        c0 = mpl.artist.getp(cb1.ax, 'xmajorticklabels') # horizontal bar
        
    mpl.artist.setp(c0, fontsize=20)


    return(0)


def calc_map_imgnum(filexyz) :  #31May13               

        # setup location of images in map based on xech yech + map pixel size
        # permet pixels rectangulaires
        # permet cartos incompletes

        data_1 = loadtxt(filexyz, skiprows = 1)
        nimg = shape(data_1)[0]
        imglist = data_1[:,0]
        print("first line :", data_1[0,:])
        print("last line : ", data_1[-1,:])

        xylist =  data_1[:,1:3]- data_1[0,1:3]

        dxyfast = xylist[1,:] - xylist[0,:]
        print("dxyfast = " , dxyfast)
        dxymax = xylist[-1,:] - xylist[0,:]
        print("dxymax = ", dxymax)

        print("fast axis")
        indfast = where(abs(dxyfast)>0.0)
        fast_axis = indfast[0][0]
        print(fast_axis)
        nintfast = dxymax[fast_axis]/dxyfast[fast_axis] 
        #print nintfast
        nintfast = int(round(nintfast,0))
        print(nintfast)
        nptsfast = nintfast+1

        dxyslow = xylist[nptsfast,:]-xylist[0,:]
        print("dxyslow = ", dxyslow)
        print("slow axis")
        slow_axis = int(abs(fast_axis-1))
        print(slow_axis)
        nintslow = dxymax[slow_axis]/dxyslow[slow_axis]
        #print nintslow
        nintslow = int(round(nintslow, 0))
        print(nintslow)
        nptsslow = nintslow + 1

        print("(img_last-img_first + 1) from file : ", data_1[-1,0]-data_1[0,0]+1)
        print("npstot from nptsslow*nptsfast : ", nptsslow*nptsfast)

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
        if pix_r != 1.0 :
                print("|dx| and |dy| steps not equal : will use rectangular pixels in map")
                print("aspect ratio : ", pix_r)
        else :
                print("equal |dx| and |dy| steps")
                pix1 = pix2 = 1
                
        if float(int(round(pix_r,1))) < (pix_r -0.01) :
                print("non integer aspect ratio")
                for nmult in (2,3,4,5):
                        toto = float(nmult)*pix_r
                        #print toto
                        #print int(round(toto,1))
                        if abs(float(int(round(toto,1))) - toto)< 0.01 :
                                #print nmult
                                break
                pix1 = nmult
                pix2 = int(round(float(nmult)*pix_r,1))
                #print "map pixel size will be ", pix1, pix2
        else :
                pix1 = 1
                pix2 = int(round(pix_r,1))

        print("pixel size for map (pix1= small, pix2= large):" , pix1, pix2)

        large_axis = argmax(abs_step)
        small_axis = argmin(abs_step)

        #print large_axis, small_axis
        if large_axis == 1 :
                pixsize = array([pix1, pix2], dtype=int)
        else :
                pixsize = array([pix2, pix1], dtype = int)
        print("pixel size for map dx dy" , pixsize)
        
        # dx => columns, dy => lines
        if fast_axis == 0 :
                nximg, nyimg = nptsfast, nptsslow
        else :
                nximg, nyimg = nptsslow, nptsfast 

        map_imgnum = zeros((nyimg,nximg),int)

        print("map raw size ", shape(map_imgnum))
        
        impos_start = zeros(2,int)
        if (dxymax[0]> 0.0) :
            startx = "left"
            impos_start[1] = 0
        else :
            startx = "right"
            impos_start[1] = nximg-1            
        if (dxymax[1]> 0.0) :
            starty = "lower"
            impos_start[0] = nyimg-1 
        else :
            starty = "upper"
            impos_start[0] = 0
        startcorner = starty + " " + startx + " corner" 
        print("map starts in : ", startcorner)
        print("impos_start = ", impos_start)
        #print dxyfast[fast_axis]
        #print dxyslow[slow_axis]
        #print dxystep

        impos = zeros(2,float) # y x   # 22Jan14 changed from int to float

        # tableau normal : y augmente vers le bas, x vers la droite
        # xech yech : xech augmente vers la droite, yech augmente vers le hut

        # tester orientation avec niveaux gris = imgnum
#        print type(xylist[0,0])
#        print type(dxystep[0])
        for i in range(nimg) :
        #for i in range(200) :
                imnum = int(round(imglist[i],0))
                impos[1] = xylist[i,0]/ abs(dxystep[0])
                impos[0] = -xylist[i,1]/ abs(dxystep[1]) 
#                print type(impos[0])
                #print impos
                impos = impos_start+impos
#                print i, imnum, impos
                impos1 = np.array((impos+0.01).round(decimals=0), dtype = int)
#                print i, imnum, impos1
                #print impos
                map_imgnum[impos1[0], impos1[1]] = imnum
#                if i == 1000 : jklds
        print("map_imgnum")
        print(map_imgnum)

        return(map_imgnum, dxystep, pixsize, impos_start)


def map_stats(filesum, 
                 variable_name = "npeaks",   
                 filter_by_pixdev_and_npeaks = 1,
                 pixdev_range_for_stat = array([0.,0.25]),
                 minnpeaks_for_stat = 20,
                 verbose = 1,
                 print_minmax = 1,
                 minmax_instead_of_range = 0,
                 scale_strain_1em4 = 0,
                 scale_stress_1MPa = 0,
                 subtract_mean = None
                 ):

    data_list, listname, nameline0 = read_summary_file(filesum, verbose = verbose)  

    data_list = np.array(data_list, dtype=float)
    
    numig = shape(data_list)[0]
    if verbose : print(numig)
    ndata_cols = shape(data_list)[1]
    if verbose : print(ndata_cols)
    
    indimg = listname.index('img')
    indgnumloc = listname.index('gnumloc')
    indnpeaks = listname.index('npeaks')
    indpixdev = listname.index('pixdev')

    
    if verbose : print(indimg, indgnumloc, indpixdev, indnpeaks)
    
    img_list = np.array(data_list[:,indimg],dtype = int)
    gnumloc_list = np.array(data_list[:,indgnumloc],dtype = int)
    pixdev_list = data_list[:,indpixdev]
    npeaks_list = np.array(data_list[:,indnpeaks],dtype = int) 
    
    nimg = len(img_list)
    
    print("mapsize : ", nimg, end=' ')
    
    if filter_by_pixdev_and_npeaks :         
        indfilt = where((pixdev_list < pixdev_range_for_stat[1])&(pixdev_list > pixdev_range_for_stat[0])&(npeaks_list > minnpeaks_for_stat)) 
        indf = indfilt[0] 
    else :
        indf = np.arange(nimg) 
        
    npts = len(indf)
    print("mapsize_filter ", npts)   

    ndec = 3    
    if subtract_mean  is  not None :
        print("subtract mean = ",  subtract_mean)
        print("1e-3 or 100 MPa units")
        
    if (variable_name[:6] == "stress")|(variable_name[:6] == "strain"):
        first_col_name = variable_name + "_0"
        indfirstcol = listname.index(first_col_name)
        indvar = np.arange(indfirstcol,indfirstcol+6)
    else :
        indvar = [listname.index(variable_name),]
        
    var_list = data_list[:,indvar]
    if subtract_mean  is  not None :
        var_list = var_list - subtract_mean


    print(variable_name)          
    if (variable_name[:6] == "stress")&(scale_stress_1MPa == 1):
        ndec = 0
        var_list = var_list * 100.
        print("1 MPa units")
    if (variable_name[:6] == "strain")&(scale_strain_1em4 == 1):
        ndec = 2
        var_list = var_list * 10.   
        print("1e-4 units")
       
    if npts > 1 :
        
        var_mean = var_list[indf].mean(axis=0)
        var_std = var_list[indf].std(axis=0)
        var_min = var_list[indf].min(axis=0)
        var_max = var_list[indf].max(axis=0)
        var_range = var_max - var_min
        
        if print_minmax :  # only for single variables
            indmin = argmin(var_list[indf])
            indmax = argmax(var_list[indf])
            print(indmin)
            img_list_short = img_list[indf]
            gnumloc_list_short = gnumloc_list[indf]
            
            imgmin = img_list_short[indmin]
            gnumlocmin = gnumloc_list_short[indmin]
            imgmax = img_list_short[indmax]
            gnumlocmax = gnumloc_list_short[indmax]
        
            if print_minmax :
                print("min/imgmin/gnumlocmin, max/imgmax/gnumlocmax")
                print(round(var_min,3), imgmin, gnumlocmin, "\t", round(var_max,3), imgmax, gnumlocmax)            

        if len(indvar) == 1 :
            if minmax_instead_of_range == 0 :
                print("mean, std, range :")
                print(round(var_mean,ndec), " ",  round(var_std, ndec), " ", round(var_range,ndec))
            else :
                print("mean, std, min, max :")
                if ndec > 0 :
                    print(round(var_mean,ndec), " ",  round(var_std, ndec), " ", round(var_min,ndec), " ", round(var_max,ndec))                
                else :
                   print(int(round(var_mean,ndec)), " ",  int(round(var_std, ndec)), " ", int(round(var_min,ndec)), " ", int(round(var_max,ndec)))                
        else :
            if minmax_instead_of_range == 0 :
                print("mean, std, range :")
                if ndec > 0 :
                    print(var_mean.round(decimals = ndec), " ",  var_std.round(decimals = ndec), " ", var_range.round(decimals = ndec))
                else :
                    print(np.array(var_mean.round(decimals = ndec), dtype = int), " ", \
                        np.array(var_std.round(decimals = ndec), dtype = int), " ", \
                        np.array(var_range.round(decimals = ndec), dtype = int))
                
            else :
                print("mean, std, min, max :")
                if ndec > 0 :
                    print(var_mean.round(decimals = ndec), " ",  var_std.round(decimals = ndec), " ", var_min.round(decimals = ndec), " ", var_max.round(decimals = ndec))                
                else :
                    print(np.array(var_mean.round(decimals = ndec), dtype = int), " ", \
                        np.array(var_std.round(decimals = ndec), dtype = int), " ", \
                        np.array(var_min.round(decimals = ndec), dtype = int), " ", \
                        np.array(var_max.round(decimals = ndec), dtype = int))                
           
                             
    if filter_by_pixdev_and_npeaks : 
        print("filter by pixdev and npeaks : ", end=' ') 
        print("pixdev_range_for_stat : ", pixdev_range_for_stat) 
        print("minnpeaks_for_stat : ", minnpeaks_for_stat)
    else : 
        print("no filtering")
    
    return(0) 
    
def rotate_map(filexyz, xylim = None):
    
    data_1 = loadtxt(filexyz, skiprows = 1)
    data_1 = np.array(data_1, dtype = float)
    nimg = shape(data_1)[0]

    xylist =  data_1[:,1:3]
    print(xylist[:3,:])
    
    sin_rot = np.sin(PAR.map_rotation*PI/180.0)
    cos_rot = np.cos(PAR.map_rotation*PI/180.0)
    sin_rot = int(round(sin_rot,0))
    cos_rot = int(round(cos_rot,0))
    if (abs(sin_rot) != 1)&(sin_rot != 0) :
        print("map rotation limited to 90, 180 or -90 deg")
        return(0)
    matrotmap = np.array([[cos_rot, sin_rot],[-sin_rot, cos_rot]])
    
    xylist_new = (dot(matrotmap, xylist.transpose())).transpose()
    
    print(xylist_new[:3,:])
    
    xylim_new = None
    
    if xylim  is  not None :
        # xmin xmax ymin ymax
        print("xylim = ", xylim)
        #print xylim.reshape(2,2)
        toto = xylim.reshape(2,2)
        toto1 = dot(matrotmap, toto)
        #print toto1.reshape(4,)
        toto2 = toto1.reshape(4,)
        xylim_new = np.array([min(toto2[0:2]),max(toto2[0:2]),min(toto2[2:4]),max(toto2[2:4])])
        print("xylim_new = ", xylim_new)
    
    data_1_new = column_stack((data_1[:,0], xylist_new, data_1[:,3:]))
    
    filexyz_new = filexyz.rstrip(".dat") + "_new.dat"
    
    header = "img 0 , xech_new 1, yech_new 2, zech 3, mon4 4, lambda 5 \n" 
    outfilename = filexyz_new
    print(outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)
    np.savetxt(outputfile, data_1_new, fmt = "%.4f")
    outputfile.close()
    return(filexyz_new, xylim_new)

def plot_curve(filesum, 
               xcol_name = "img" ,
               ycol_name = None,
               maptype = None, # "strain6_sample",
               saveplotname = None,
               savedatafilename = None,
               remove_ticklabels_titles = 0,
               ymin_max = [-0.3,0.3],
                xmin_max = None,
                xvline = 5.0,
                overlay = "no"
               ) :
        
        p.rcParams['lines.markersize'] = 5 
        
        
        # added saveplotname  JSM 12 Sept 2013
        list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev', 'intensity', 
                                        'dxymicrons_0', 'dxymicrons_1', 
 'matstarlab_0', 'matstarlab_1', 'matstarlab_2', 'matstarlab_3', 'matstarlab_4','matstarlab_5', 'matstarlab_6', 'matstarlab_7', 'matstarlab_8', 
 'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
 'euler3_0', 'euler3_1', 'euler3_2', 
 'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
 'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
 'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
 'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
 'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
 'res_shear_stress_0', 'res_shear_stress_1', 'res_shear_stress_2', 'res_shear_stress_3', 'res_shear_stress_4', 'res_shear_stress_5', 'res_shear_stress_6', 'res_shear_stress_7', 'res_shear_stress_8', 'res_shear_stress_9', 'res_shear_stress_10', 'res_shear_stress_11',
 'max_rss', 'von_mises', 'misorientation_angle', 'dalf']
 
        # key = maptype , ncolplot, nplot
        # ncolplot = nb of columns for these data
        # nplot = 3 per rgb color map
        # ngraph = number of graphs
        # ngraphline, ngraphcol = subplots
        # ngraphlabels = subplot number -1 for putting xlabel and ylabel on axes
        dict_nplot = {
            "euler3" : [3,3,1,1,1,0,["rgb_euler",]],
            "rgb_x_sample" : [6,6,2, 1,2,0,["x_sample","z_sample"]],
            "strain6_crystal" : [6,18,6,2,3,3,["aa","bb","cc","ca","bc","ab"]], 
            "strain6_sample" : [6,18,6,2,3,3, ["XX","YY","ZZ","YZ","XZ","XY"]], 
            "stress6_crystal" : [6,18,6,2,3,3, ["aa","bb","cc","ca","bc","ab"]], 
            "stress6_sample" : [6,18,6,2,3,3,["XX","YY","ZZ","YZ","XZ","XY"]],
            "res_shear_stress": [12,36,12,3,4,8,["rss0", "rss1","rss2", "rss3","rss4", "rss5","rss6", "rss7","rss8", "rss9","rss10", "rss11"]],
            'max_rss': [1,3,1,1,1, 0,["max_rss",]],
            'von_mises': [1,3,1,1,1,0,["von Mises stress",]],
            'misorientation_angle': [1,3,1,1,1,0,["misorientation angle",]],
            "fit" : [2,6,2,1,2,0,["npeaks", "pixdev"]],
            "dalf" : [1,3,1,1,1,0,["delta_alf exp-theor"]],
            "w_mrad" : [3,9,3,1,3,0,["WX","WY","WZ"]],
            "vecRod" : [3,9,3,1,3,0,["RX","RY","RZ"]],
            }

        data_list, listname, nameline0 = read_summary_file(filesum)  
    
        data_list = np.array(data_list, dtype=float)
        
        numig = shape(data_list)[0]
        print(numig)
        ndata_cols = shape(data_list)[1]
        print(ndata_cols)                   

        if maptype  is None :        
            indx = listname.index(xcol_name)
            indy = listname.index(ycol_name)
                   
            xx = np.array(data_list[:,indx],dtype = float)
            yy = np.array(data_list[:,indy],dtype = float)
            
            imgmin = np.amin(xx)
            imgmax = np.amax(xx)
            nbimages = numig 
            
            if overlay == "no" : 
                p.figure()
                color1 = "ro-"
            else :  color1 = "bs-"
                
            p.plot(xx,yy, color1)
            p.xlabel(xcol_name)
            p.ylabel(ycol_name)
            
#            p.xlim(imgmin-0.05*nbimages,imgmax+0.05*nbimages)
            
            if xvline  is  not None : p.axvline(x=xvline)
            #p.axvline(x = 5., color = "r")
            #p.axhline(y = 0, color = "r")
            
            if saveplotname:
                p.savefig(saveplotname)
            
            if savedatafilename:
                np.savetxt(savedatafilename, np.array([xx,yy]).T, delimiter=" ", fmt="%s") 
                                            
        if maptype  is  not None :
            

            ncolplot = dict_nplot[maptype][0]
            if maptype != "fit" :
                map_first_col_name = maptype + "_0"  
                if maptype == "vecRod" :
                    map_first_col_name = "vecRod0"
                ind_first_col = listname.index(map_first_col_name)           
                indcolplot = np.arange(ind_first_col, ind_first_col + ncolplot)
            else :
                indcolplot = [listname.index(dict_nplot[maptype][6][0]),listname.index(dict_nplot[maptype][6][1])]
                
            ngraph = dict_nplot[maptype][2]
            ngraphline = dict_nplot[maptype][3]
            ngraphcol = dict_nplot[maptype][4]
            ngraphlabels = dict_nplot[maptype][5]
            print("ngraph, ngraphline, ngraphcol, ngraphlabels")
            print(ngraph, ngraphline, ngraphcol, ngraphlabels)
        
            indx = listname.index(xcol_name)
                   
            xx = np.array(data_list[:,indx],dtype = float)
            yy = np.array(data_list[:,indcolplot],dtype = float)
            

           
            for j in range(ngraph):
                
                if overlay == "no" :
                    fig1 = p.figure(1, figsize=(15,10))
                    color1 = "ro-"
                else : color1 = "bs-"
    #            print p.setp(fig1)
    #            print p.getp(fig1)
                ax = p.subplot(ngraphline, ngraphcol, j+1)
               
                p.plot(xx,yy[:,j],color1)
    #            print p.setp(imrgb)
                strname = dict_nplot[maptype][6][j]
                if remove_ticklabels_titles == 0 : p.title(strname)
                if remove_ticklabels_titles :
    #                print p.getp(ax)
                    p.subplots_adjust(wspace = 0.05,hspace = 0.05)
                    p.setp(ax,xticklabels = [])
                    p.setp(ax,yticklabels = [])
#                ax.grid(color=color_grid, linestyle='-', linewidth=2)
                else :
                    p.subplots_adjust(wspace = 0.2,hspace = 0.2)
                if PAR.cr_string == "\n":                    
                    ax.locator_params('x', tight=True, nbins=5)
                    ax.locator_params('y', tight=True, nbins=5)

                if (j == ngraphlabels) :
                    p.xlabel(xcol_name)
                    p.ylabel(maptype)
                    
                if ymin_max  is  not None : p.ylim(ymin_max[0],ymin_max[1])
                else :
                    yymin = yy[:,j].min()
                    yymax = yy[:,j].max()
                    p.ylim(yymin - 0.05*(yymax-yymin),yymax + 0.05*(yymax-yymin))
                p.axhline(y=0)
                if xvline  is  not None : p.axvline(x=xvline)
                if xmin_max  is  not None : p.xlim(xmin_max[0],xmin_max[1])
                else : 
                    xxmin = xx.min()
                    xxmax = xx.max()
                    p.xlim(xxmin - 0.05*(xxmax-xxmin),xxmax + 0.05*(xxmax-xxmin))
                if  PAR.cr_string == "\n" :  # windows    
                    ax.locator_params('x', tight=True, nbins=5)
                    ax.locator_params('y', tight=True, nbins=5)
        

        return(0)
 
#from matplotlib import mpl
#cmap = mpl.cm.PiYG
cmap = mpl.cm.RdBu_r


def plot_map(filesum, 
             filexyz = None, 
             maptype = "fit", 
             filetype = "LT", 
             subtract_mean = "no",
             gnumloc = 1,
             filter_on_pixdev_and_npeaks = 1,
             maxpixdev_forfilter = 0.3, # only for filter_on_pixdev_and_npeaks = 1
             minnpeaks_forfilter = 20, # only for filter_on_pixdev_and_npeaks = 1
             min_forplot = -0.2,   # pour strain, stress, rss, von Mises
             max_forplot = 0.2, # use None for autoscale           
             pixdevmin_forplot = 0.0,   # only for maptype = "fit"
             pixdevmax_forplot = 0.3,   # only for maptype = "fit"
             npeaksmin_forplot = 6.0,  # only for maptype = "fit"
             npeaksmax_forplot = 70.,  # only for maptype = "fit"
             zoom = "no",
             xylim = None,
             filter_mean_strain_on_misorientation = 0,
             max_misorientation = 0.15, # only for filter_mean_strain_on_misorientation = 1
             change_sign_xy_xz = 0,
             subtract_constant = None,
             remove_ticklabels_titles = 0,
             col_for_simple_map = None,
             low_npeaks_as_missing = None,
             low_npeaks_as_red_in_npeaks_map = None, # only for maptype = "fit"
             low_pixdev_as_green_in_pixdev_map = None, # only for maptype = "fit"
             use_mrad_for_misorientation = "no", # only for maptype = "misorientation_angle"
             color_for_duplicate_images = None, # [0.,1.,0.]
             color_for_missing = None,
             high_pixdev_as_blue_and_red_in_pixdev_map = None, # only for maptype = "fit"
             filter_on_intensity = 0,
             min_intensity_forfilter = 20000., # only for filter_on_intensity = 1
             color_for_max_strain_positive = array([1.,0.,0.]),  # red  # [1.0,1.0,0.0]  # yellow
             color_for_max_strain_negative = array([0.,0.,1.]),   # blue
             plot_grid = 1,
             numfig = 1,
             savefig = 0,
             fileprefix = "",
             add_symbols = 0,
             add_symbols_for_filtered = 0
                ):  #29May13
        
        print(time())
        
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
        
        list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev', 'intensity', 
                                        'dxymicrons_0', 'dxymicrons_1', 
 'matstarlab_0', 'matstarlab_1', 'matstarlab_2', 'matstarlab_3', 'matstarlab_4','matstarlab_5', 'matstarlab_6', 'matstarlab_7', 'matstarlab_8', 
 'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
 'euler3_0', 'euler3_1', 'euler3_2', 
 'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
 'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
 'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
 'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
 'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
 'res_shear_stress_0', 'res_shear_stress_1', 'res_shear_stress_2', 'res_shear_stress_3', 'res_shear_stress_4', 'res_shear_stress_5', 'res_shear_stress_6', 'res_shear_stress_7', 'res_shear_stress_8', 'res_shear_stress_9', 'res_shear_stress_10', 'res_shear_stress_11',
 'max_rss', 'von_mises', 'misorientation_angle', 'dalf'] 
 
 
#  NB : misorientation_angle column seulement pour analyse mono-grain
# NB : dalf column seulement pour mat2spots ou fit calib

#        list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev']

        color_grid = "k"
        
        if col_for_simple_map  is  not None :
            filter_on_pixdev_and_npeaks = 0
            filter_mean_strain_on_misorientation = 0
            
        data_list, listname, nameline0 = read_summary_file(filesum)      
        data_list = np.array(data_list, dtype=float)        
        numig = shape(data_list)[0]
        print(numig)
        ndata_cols = shape(data_list)[1]
        print(ndata_cols)
        
        indimg = listname.index('img')
        pixdevlist = zeros(numig, float)   
        gnumlist = zeros(numig, int)       
        npeakslist = ones(numig, int)*25
        xylist = zeros((numig,2),float)
        
        if filter_on_intensity :
            indintensity = listname.index('intensity')
            intensitylist = np.array(data_list[:,indintensity],dtype = float)
        if "gnumloc" in listname : 
            indgnumloc = listname.index('gnumloc')
            gnumlist = np.array(data_list[:,indgnumloc],dtype = int)           
        if 'npeaks' in listname :    
            indnpeaks = listname.index('npeaks')
            npeakslist = np.array(data_list[:,indnpeaks],dtype = int)        
        if 'pixdev' in listname :
            indpixdev = listname.index('pixdev')
            pixdevlist = data_list[:,indpixdev]        
        if 'dxymicrons_0' in listname :
            indxech = listname.index('dxymicrons_0')
            xylist = np.array(data_list[:,indxech:indxech+2],dtype = float)                     
        if 'misorientation_angle' in listname :
            indmisor = listname.index('misorientation_angle')
            if filter_mean_strain_on_misorientation :
                misor_list = np.array(data_list[:,indmisor],dtype = float)
                if use_mrad_for_misorientation == "yes":
                    print("converting misorientation angle into mrad")
                    misor_list = misor_list * math.pi/180.*1000.
                indm = where(misor_list < max_misorientation)
                print("filtering out img with large misorientation > ", max_misorientation)
                print("nimg with low misorientation : ", shape(indm)[1])

        
        # key = maptype , ncolplot, nplot
        # ncolplot = nb of columns for these data
        # nplot = 3 per rgb color map
        # ngraph = number of graphs
        # ngraphline, ngraphcol = subplots
        # ngraphlabels = subplot number -1 for putting xlabel and ylabel on axes
        dict_nplot = {
            "euler3" : [3,3,1,1,1,0,["rgb_euler",]],
            "rgb_x_sample" : [9,9,3,1,3,0,["x_sample","y_sample", "z_sample"]],
            "rgb_x_lab" : [9,9,3,1,3,0,["x_lab","y_lab", "z_lab"]],
            "strain6_crystal" : [6,18,6,2,3,3,["aa","bb","cc","ca","bc","ab"]], 
            "strain6_sample" : [6,18,6,2,3,3, ["XX","YY","ZZ","YZ","XZ","XY"]], 
            "stress6_crystal" : [6,18,6,2,3,3, ["aa","bb","cc","ca","bc","ab"]], 
            "stress6_sample" : [6,18,6,2,3,3,["XX","YY","ZZ","YZ","XZ","XY"]],
            "w_mrad" : [3,9,3,1,3,0,["WX","WY","WZ"]],
            "res_shear_stress": [12,36,12,3,4,8,["rss0", "rss1","rss2", "rss3","rss4", "rss5","rss6", "rss7","rss8", "rss9","rss10", "rss11"]],
            'max_rss': [1,3,1,1,1, 0,["max_rss",]],
            'von_mises': [1,3,1,1,1,0,["von Mises stress",]],
            'eq_strain': [1,3,1,1,1,0,["equivalent strain",]],
            'misorientation_angle': [1,3,1,1,1,0,["misorientation angle",]],
            'intensity': [1,3,1,1,1,0,["intensity",]],
            'maxpixdev': [1,3,1,1,1,0,["maxpixdev",]],
            'stdpixdev': [1,3,1,1,1,0,["stdpixdev",]],
            "fit" : [2,6,2,1,2,0,["npeaks", "pixdev"]],
            "dalf" : [1,3,1,1,1,0,["delta_alf exp-theor"]]
            }
        
        list_of_positive_quantities = ['intensity',
                                       'maxpixdev',
                                       'stdpixdev',
                                       'max_rss',
                                       'misorientation_angle',
                                       'von_mises',
                                       "eq_strain"]
        list_of_positive_or_negative_quantities = ["strain6_crystal",
                                        "strain6_sample" ,
                                        "stress6_crystal",
                                        "stress6_sample" ,
                                        "w_mrad",
                                        "dalf",
                                        "res_shear_stress"]

        ncolplot = dict_nplot[maptype][0]

        if maptype != "fit":
#            if maptype in ['max_rss','von_mises','misorientation_angle', 'dalf', "intensity"]:
            if ncolplot == 1 :
                map_first_col_name = maptype
                if col_for_simple_map  is  not None :
                    map_first_col_name = col_for_simple_map
            else :
                map_first_col_name = maptype + "_0"  
                if col_for_simple_map  is  not None :
                    map_first_col_name = col_for_simple_map                
            ind_first_col = listname.index(map_first_col_name)    
            print(ind_first_col)
            
            indcolplot = np.arange(ind_first_col, ind_first_col + ncolplot)
        
        if zoom == "yes" :
            listxj = []
            listyi = []

        if filexyz  is None :
            # creation de filexyz a partir des colonnes de filesum
            indxy = listname.index('dxymicrons_0')
            imgxy = column_stack((data_list[:,indimg],data_list[:,indx:indx+2]))
            ind1 = where(gnumlist == 0)
            imgxynew = imgxy[ind1[0],:]
            print("min, max : img x y ", imgxynew.min(axis=0) ,imgxynew.max(axis=0))
            print("first, second, last point : img x y :")
            print(imgxynew[0,:])
            print(imgxynew[1,:])
            print(imgxynew[-1,:])
            filexyz = "filexyz.dat"
            header = "img 0 , xech 1, yech 2 \n" 
            outputfile = open(filexyz,'w')
            outputfile.write(header)
            np.savetxt(outputfile, imgxynew, fmt = "%.4f")
            outputfile.close()

        filexyz_new = filexyz
        xylim_new = xylim
        
        if abs(PAR.map_rotation)> 0.1 :
            print("rotating map clockwise by : ", PAR.map_rotation, "degrees")
            filexyz_new, xylim_new = rotate_map(filexyz, xylim = xylim)            
        
        map_imgnum, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz_new)
        
        imgxyz = loadtxt(filexyz_new, skiprows = 1)
        img_in_filexyz = np.array(imgxyz[:,0], int)
        xy_in_filexyz = np.array(imgxyz[:,1:3], float)
            
#            print "img_in_filexyz = ", img_in_filexyz

        nlines = shape(map_imgnum)[0]
        ncol = shape(map_imgnum)[1]
        nplot = dict_nplot[maptype][1]
        plotdat = zeros((nlines,ncol,nplot),float)
        
        print("grain : ", gnumloc)
        
        if filter_on_pixdev_and_npeaks :
            print("filtering :")
            print("maxpixdev ", maxpixdev_forfilter)
            print("minnpeaks ", minnpeaks_forfilter)
            indf = where((gnumlist == gnumloc)&(pixdevlist < maxpixdev_forfilter)&(npeakslist > minnpeaks_forfilter))
        elif filter_on_intensity :
            indf = where((gnumlist == gnumloc)&(npeakslist > 0)&(intensitylist > min_intensity_forfilter))
        else :
            print("****************************************************")
            indf = where((gnumlist == gnumloc)&(npeakslist > 0))

        imglist = np.array(data_list[:,0],dtype = int)
        
        print("numig total", numig)
        print("numig remaining after filtering", len(indf[0]))
        
        if add_symbols_for_filtered and filter_on_pixdev_and_npeaks :
            
            ind_filtered = where((gnumlist == gnumloc)&((pixdevlist >= maxpixdev_forfilter)|(npeakslist <=minnpeaks_forfilter)))
            
            print("numig filtered out", len(ind_filtered[0]))
            first_time = 1
            for i in ind_filtered[0] :
                ind0 = where(img_in_filexyz == imglist[i])
                toto = xy_in_filexyz[ind0[0][0],:] 
                if first_time :
                    list_xy_filtered = toto
                    first_time = 0
                else :
                    list_xy_filtered = row_stack((list_xy_filtered,toto))
        
            if len(ind_filtered[0])>0 :
                list_xy_filtered = list_xy_filtered + dxystep/2.0 
            
#        indf = where((gnumlist == gnumloc)&(npeakslist > 0))   
                             
        #print indf[0]

        # filtered data        
        data_list2 = data_list[indf[0],:]
        
        imglist = np.array(data_list2[:,0],dtype = int)
        
        if maptype == "euler3" : 
            euler3 = data_list2[:,indcolplot]
            ang0 = 360.0
            ang1 = np.arctan(np.sqrt(2.0))*180.0/PI
            ang2 = 180.0
            ang012 = array([ang0, ang1, ang2])
            print(euler3[0,:])
            euler3norm= euler3/ ang012
            print(euler3norm[0,:])
            #print min(euler3[:,0]), max(euler3[:,0])
            #print min(euler3[:,1]), max(euler3[:,1])
            #print min(euler3[:,2]), max(euler3[:,2])

        elif maptype[:5] == "rgb_x": rgbxyz = data_list2[:,indcolplot]
            
        elif maptype == "fit":   
            default_color_for_missing = array([1.0,0.8,0.8])
            if color_for_missing  is None : color0 = default_color_for_missing
            else : color0 = color_for_missing                
            for j in range(ncolplot) : plotdat[:,:,3*j:3*(j+1)] = color0 
            pixdevlist2 = pixdevlist[indf[0]]
            npeakslist2 = npeakslist[indf[0]]
            pixdevmin2 = pixdevlist2.min()
            pixdevmax2 = pixdevlist2.max()
            pixdevmean2 = pixdevlist2.mean()
            npeaksmin2 = npeakslist2.min()
            npeaksmax2 = npeakslist2.max()
            npeaksmean2 = npeakslist2.mean()
            print(filesum)

            print("pixdev : mean, min, max")
            print(round(pixdevmean2,3), round(pixdevmin2,3), round(pixdevmax2,3))
            print("npeaks : mean min max")
            print(round(npeaksmean2,1), npeaksmin2, npeaksmax2)

            if pixdevmin_forplot  is None : pixdevmin_forplot = pixdevmin2
            if pixdevmax_forplot  is None : pixdevmax_forplot = pixdevmax2    
            if npeaksmin_forplot  is None : npeaksmin_forplot = npeaksmin2
            if npeaksmax_forplot  is None : npeaksmax_forplot = npeaksmax2    
                
            print("for color map : ")
            print("pixdev : min, max : " , pixdevmin_forplot, pixdevmax_forplot)
            print("npeaks : min, max : ", npeaksmin_forplot , npeaksmax_forplot)
            print("black = min for npeaks")
            print("black = max for pixdev")
            print("pink = missing")
            if low_npeaks_as_red_in_npeaks_map  is  not None :
                        print("red : npeaks < ",  low_npeaks_as_red_in_npeaks_map)
            
            color_grid = "k"
            
        else :  # for strain and derived quantities 
                  
            print("maptype = ", maptype)
#            if maptype in ['max_rss','von_mises','misorientation_angle', "intensity"]:
            if maptype in list_of_positive_quantities :
#            if ncolplot == 1 :
                default_color_for_missing = array([1.0,0.8,0.8]) # pink = color for missing data
                if color_for_missing  is None : color0 = default_color_for_missing
                else : color0 = color_for_missing   
                plotdat[:,:,0:3] = color0
                color_filtered = np.array([0.,1.,0.])
                color_grid = "k"
                if low_npeaks_as_missing :
                    color_filtered = np.array([1.0,0.8,0.8])
            elif maptype in list_of_positive_or_negative_quantities :
#                color_filtered = np.array([0.5,0.5,0.5])
#                color_filtered = zeros(3,float)
#                color_filtered = np.array([0.,1.,0.])
                color_filtered = zeros(3,float)
                color_grid = "w"
                if add_symbols_for_filtered :
                    color_filtered = zeros(3,float)
                if low_npeaks_as_missing :
                    color_filtered = np.array([0.,0.,0.])    
                if add_symbols :  
                    color_filtered =  array([1.0,0.8,0.8]) # array([1.,1.,1.]) #
                    for j in range(ncolplot) : plotdat[:,:,3*j:3*(j+1)] = array([1.0,0.8,0.8]) # array([1.,1.,1.]) #                
                else :  
                    for j in range(ncolplot) : plotdat[:,:,3*j:3*(j+1)] = NaN #  0. # black = color for missing data
                if maptype != "dalf" : 
                    if maptype != "w_mrad" :
                        print("xx xy xz yy yz zz")
                    else : print("wx wy wz")
            
                
            imglist1 =  np.array(data_list[:,indimg], dtype = int)
            
            for i in range(numig):
                if i not in indf[0]:
                    ind2 = where(map_imgnum ==imglist1[i])
                    iref, jref = ind2[0][0], ind2[1][0]
                    
#                    if maptype in ['max_rss','von_mises','misorientation_angle', "intensity"]:
                    if ncolplot == 1 :
                        plotdat[iref, jref, 0:3] = color_filtered
                    else :
                        for j in range(ncolplot): plotdat[iref,jref,3*j:3*j+3] = color_filtered
                
            list_plot = data_list2[:,indcolplot]

            if (maptype == 'misorientation_angle')&(use_mrad_for_misorientation == "yes"):
                print("converting misorientation angle into mrad")
                list_plot = list_plot * math.pi/180. *1000.               
            print(shape(list_plot))
            
            if 0 : # check sign
                xech_list = data_list2[:,indxech]
                print(list_plot[:,0])
                p.figure()
                p.plot(xech_list,  list_plot[:,0], 'ro-')
                p.ylim(-3,3)
            
            if change_sign_xy_xz & (maptype == "strain6_sample"):
                list_plot[:,4] = -list_plot[:,4]
                list_plot[:,5] = -list_plot[:,5]
            if filter_mean_strain_on_misorientation :
                list_plot_mean = list_plot[indm[0]].mean(axis = 0)
            else :    
                list_plot_mean = list_plot.mean(axis = 0)
            if subtract_mean == "yes" :
                print("subtract mean")
                list_plot = list_plot - list_plot_mean
            if subtract_constant  is  not None :
                if isscalar(subtract_constant) :
                    imgref_for_subtract = subtract_constant
                    ind4 = where(imglist == imgref_for_subtract)
                    print("subtracting value at ref image, image = ", imgref_for_subtract)
                    print("subtracted value = ", (list_plot[ind4[0]]).round(decimals=2))
                    print("ind4 = ", ind4)
                    list_plot = list_plot - list_plot[ind4[0]]
                elif len(subtract_constant) == ngraph :
                    print("subtracting constant = ", subtract_constant)
                    list_plot = list_plot - subtract_constant

                    
            list_plot_min = list_plot.min(axis = 0)
            list_plot_max = list_plot.max(axis = 0)
           
            print("min : ", list_plot_min.round(decimals=2))
            print("max : ", list_plot_max.round(decimals=2))
            print("mean : ", list_plot_mean.round(decimals=2))
            
            if min_forplot  is  not None : list_plot_min = min_forplot*ones(ncolplot,float)
            if max_forplot  is  not None : list_plot_max = max_forplot*ones(ncolplot,float)
            
            print("for color map :")
            print("min : ", list_plot_min.round(decimals=2))
            print("max : ", list_plot_max.round(decimals=2))
            
#            print "type of color map : "
#            print "rgb color for missing data :", color_for_missing
#                
#            if filter_on_pixdev_and_npeaks :
#                print "map filtered by npeaks / pixdev"
#                print "max_pixdev, min_npeaks = ", maxpixdev_forfilter, minnpeaks_forfilter  
#                print "rgb color for filtered data :", color_filtered       
#        
#            if filter_on_intensity :
#                print "map filtered by intensity"
#                print "min_fraction_of_max_intensity_for_filter =", min_intensity_forfilter
#                print "rgb color for filtered data :", color_filtered   
            
            if add_symbols :
 
                ind_strain_below_min_negative = where(list_plot < min_forplot)             
                first_time = 1
                for i in ind_strain_below_min_negative[0] :
                    ind0 = where(img_in_filexyz == imglist[i])
                    toto = xy_in_filexyz[ind0[0][0],:] 
                    if first_time :
                        list_xy_strain_below_min_negative = toto
                        first_time = 0
                    else :
                        list_xy_strain_below_min_negative = row_stack((list_xy_strain_below_min_negative,toto))
            
                if len(ind_strain_below_min_negative[0])>0 :
                    list_xy_strain_below_min_negative = list_xy_strain_below_min_negative + dxystep/2.0 

               
                ind_strain_above_max_positive = where(list_plot > max_forplot)
#                print ind_strain_above_max_positive               
#                print ind_strain_above_max_positive[0]               
                first_time = 1
                for i in ind_strain_above_max_positive[0] :
    #                print "i = ", i
    #                print "img =", imglist[i]
                    ind0 = where(img_in_filexyz == imglist[i])
    #                print ind0[0][0]
                    toto = xy_in_filexyz[ind0[0][0],:] 
                    if first_time :
                        list_xy_strain_above_max_positive = toto
                        first_time = 0
                    else :
                        list_xy_strain_above_max_positive = row_stack((list_xy_strain_above_max_positive,toto))
            
                list_xy_strain_above_max_positive = list_xy_strain_above_max_positive + dxystep/2.0 


                ind_strain_positive = where(list_plot > 0.)             
                first_time = 1
                for i in ind_strain_positive[0] :
                    ind0 = where(img_in_filexyz == imglist[i])
                    toto = xy_in_filexyz[ind0[0][0],:] 
                    if first_time :
                        list_xy_strain_positive = toto
                        first_time = 0
                    else : 
                        list_xy_strain_positive = row_stack((list_xy_strain_positive,toto))            
                list_xy_strain_positive = list_xy_strain_positive + dxystep/2.0 
                
                
                ind_strain_negative = where(list_plot < 0.)             
                first_time = 1
                for i in ind_strain_negative[0] :
                    ind0 = where(img_in_filexyz == imglist[i])
                    toto = xy_in_filexyz[ind0[0][0],:] 
                    if first_time :
                        list_xy_strain_negative = toto
                        first_time = 0
                    else : 
                        list_xy_strain_negative = row_stack((list_xy_strain_negative,toto))            
                list_xy_strain_negative = list_xy_strain_negative + dxystep/2.0 
                
                
        numig2 = shape(data_list2)[0]

        xylist = xylist[indf[0],:]  
            
        # add 08Oct14 
        if add_symbols & (maptype == "fit") :
                        
#            level1 = 0.25
            level2 = 0.5
            ind_pixdev_above_level2 = where(pixdevlist2 > level2)
#            print "ind_pixdev_above_level2 = ", ind_pixdev_above_level2[0]
           
            level3 = 20.
            ind_npeaks_below_level3 = where(npeakslist2 < level3)
         
            first_time = 1
            for i in ind_pixdev_above_level2[0] :
#                print "i = ", i
#                print "img =", imglist[i]
                ind0 = where(img_in_filexyz == imglist[i])
#                print ind0[0][0]
                toto = xy_in_filexyz[ind0[0][0],:] 
                if first_time :
                    list_xy_pixdev_above_level2 = toto
                    first_time = 0
                else :
                    list_xy_pixdev_above_level2 = row_stack((list_xy_pixdev_above_level2,toto))

            first_time = 1
            for i in ind_npeaks_below_level3[0] :
#                print "i = ", i
#                print "img =", imglist[i]
                ind0 = where(img_in_filexyz == imglist[i])
#                print ind0[0][0]
                toto = xy_in_filexyz[ind0[0][0],:] 
                if first_time :
                    list_xy_npeaks_below_level3 = toto
                    first_time = 0
                else :
                    list_xy_npeaks_below_level3 = row_stack((list_xy_npeaks_below_level3,toto))

            print("dxystep =", dxystep) 
                         
            list_xy_pixdev_above_level2 = list_xy_pixdev_above_level2 + dxystep/2.0
            list_xy_npeaks_below_level3 = list_xy_npeaks_below_level3 + dxystep/2.0
        
                
#        jklfds                
                
    
        dxystep_abs = abs(dxystep)
        
#        if (maptype != "fit")&(maptype[:2] != "rgb"):
#            print maptype
#            print list_plot_min
#            print list_plot_max
#            list_plot_cen = (list_plot_max+list_plot_min)/2.0

#        if add_symbols :
#            color_circles = zeros((numig2,nplot), float)

        if maptype in list_of_positive_quantities :
            print("color scale : black = min, white = max")

        if low_pixdev_as_green_in_pixdev_map  is  not None :
            print("green : pixdev < ", low_pixdev_as_green_in_pixdev_map)
        if high_pixdev_as_blue_and_red_in_pixdev_map  is  not None :
            print("red : pixdev > ", high_pixdev_as_blue_and_red_in_pixdev_map[1])


#        ***********************************************
    # ************************************************
    # filling of map   #MAINLOOP

        for i in range(numig2) :
                ind2 = where(map_imgnum == imglist[i])
#                print imglist[i]
                print(time())
                iref, jref = ind2[0][0], ind2[1][0]
                if (zoom == "yes")&(npeakslist[i]> 0) :
                    listxj.append(xylist[i,0])
                    listyi.append(xylist[i,1])
                
                if maptype == "euler3": plotdat[iref, jref, :] = euler3norm[i,:]
                
                elif maptype[:5] == "rgb_x":  plotdat[iref, jref, :] = rgbxyz[i,:]*1.0
                
                elif maptype == "fit":
                    toto = ((npeakslist2[i]-npeaksmin_forplot)/(npeaksmax_forplot-npeaksmin_forplot))
                    plotdat[iref,jref,0:3]= clip(toto, 0., 1.)
#                    print toto, clip(toto, 0., 1.)
                    if low_npeaks_as_red_in_npeaks_map  is  not None : 
                        if (npeakslist2[i]< low_npeaks_as_red_in_npeaks_map) : plotdat[iref,jref,0:3] = array([1.,0.,0.])
                        else :
                            if (npeakslist2[i]< npeaksmin_forplot) : plotdat[iref,jref,0:3] = array([1.,0.,0.])
                        
                    toto = ((pixdevmax_forplot - pixdevlist2[i])/(pixdevmax_forplot-pixdevmin_forplot))
                    plotdat[iref,jref,3:6]= clip(toto, 0., 1.) 
                    
                    if high_pixdev_as_blue_and_red_in_pixdev_map  is  not None :
#                        if pixdevlist2[i] > high_pixdev_as_blue_and_red_in_pixdev_map[0] : plotdat[iref,jref,3:6] = array([0.,0.,1.])
                        if pixdevlist2[i] > high_pixdev_as_blue_and_red_in_pixdev_map[1] : plotdat[iref,jref,3:6] = array([1.,0.,0.])
                    elif pixdevlist2[i] > pixdevmax_forplot : plotdat[iref,jref,3:6] =  array([0.,0.,0.])# array([1.,0.,0.])
                    if low_pixdev_as_green_in_pixdev_map  is  not None :
                        if (pixdevlist2[i] < low_pixdev_as_green_in_pixdev_map)& (npeakslist2[i]>20): plotdat[iref,jref,3:6] = array([0.,1.,0.])
#                elif maptype in ['max_rss','von_mises', "misorientation_angle", "intensity"]:
#                elif ncolplot == 1 :
                elif maptype in list_of_positive_quantities :
                    if list_plot[i]> list_plot_max : plotdat[iref,jref, 0:3]= np.array([1.,0.,0.])
                    elif list_plot[i] < list_plot_min : plotdat[iref,jref, 0:3]= np.array([1.,1.,1.])
                    else :
#                        for j in range(3): 
#                        plotdat[iref,jref, :]= (list_plot_max-list_plot[i])/(list_plot_max-list_plot_min)
                        plotdat[iref,jref, :]= (list_plot[i]-list_plot_min)/(list_plot_max-list_plot_min)


                elif maptype in list_of_positive_or_negative_quantities :   
                    for j in range(ncolplot):
                        if list_plot[i,j]> list_plot_max[j] :
                            plotdat[iref,jref,3*j:3*j+3] = color_for_max_strain_positive  
                        elif list_plot[i,j]< list_plot_min[j] :
                            plotdat[iref,jref,3*j:3*j+3] = color_for_max_strain_negative
                        else :   
                            if add_symbols :
                                toto = (list_plot_max[j]-abs(list_plot[i,j]))/list_plot_max[j]
                                plotdat[iref,jref,3*j:3*j+3] = clip(toto, 0., 1.)
#                                color_circles[i,3*j:3*j+3] = clip(toto, 0., 1.)
                            else : 
                                toto = (list_plot[i,j]-list_plot_min[j])/(list_plot_max[j]-list_plot_min[j])
                                plotdat[iref,jref,3*j:3*j+3] = np.array(cmap(toto))[:3]
#                            plotdat[iref,jref,3*j:3*j+3] = np.array([0.019607843831181526, 0.18823529779911041, 0.3803921639919281]) # MG.cmap(0.)[:3]
#                            plotdat[iref,jref,3*j:3*j+3] = np.array([0.40392157435417175, 0.0, 0.12156862765550613])  # MG.cmap(1.)[:3]

#>>> MG.cmap(0.)[:3]
#(0.019607843831181526, 0.18823529779911041, 0.3803921639919281)
#>>> MG.cmap(1.)[:3]
#(0.40392157435417175, 0.0, 0.12156862765550613)
                           

                if color_for_duplicate_images  is  not None :
                    if i>0 :
                        dimg = imglist[i]-imglist[i-1]
                        if dimg == 0. :
                            print("warning : two grains on img ", imglist[i])
                            plotdat[iref,jref,0:3] = color_for_duplicate_images

              # reperage de l'ordre des images dans la carto
##                if imglist[i]==min(imglist) :
##                    plotdat[iref, jref, :] = array([1., 0., 0.])
##                if imglist[i]==min(imglist)+ncol-1 :
##                    plotdat[iref, jref, :] = array([0., 1., 0.])
##                if imglist[i]==max(imglist) :
##                    plotdat[iref, jref, :] = array([0., 0., 1.])            
                #plotdat[iref, jref, :] = array([1.0, 1.0, 1.0])*float(imglist[i])/max(imglist)
                #print plotdat[iref, jref, :]

        # extent corrected 06Feb13
        xrange1 = array([0.0,ncol*dxystep[0]])
        yrange1 = array([0.0, nlines*dxystep[1]])
        xmin, xmax = min(xrange1), max(xrange1)
        ymin, ymax = min(yrange1), max(yrange1)
        extent = xmin, xmax, ymin, ymax
        print(extent)
               
        ngraph = dict_nplot[maptype][2]
        ngraphline = dict_nplot[maptype][3]
        ngraphcol = dict_nplot[maptype][4]
        ngraphlabels = dict_nplot[maptype][5]
        print("ngraph, ngraphline, ngraphcol, ngraphlabels")
        print(ngraph, ngraphline, ngraphcol, ngraphlabels)
        print("shape(plotdat)")
        print(shape(plotdat))
        
        if zoom == "yes" :         
            listxj = np.array(listxj, dtype = float)
            listyi = np.array(listyi, dtype = float)
            minxj = listxj.min()-2*dxystep_abs[0]
            maxxj = listxj.max()+2*dxystep_abs[0]
            minyi = listyi.min()-2*dxystep_abs[1]
            maxyi = listyi.max()+2*dxystep_abs[1]
            print("zoom : minxj, maxxj, minyi, maxyi : ", minxj, maxxj, minyi, maxyi)
         
        p.rcParams['figure.subplot.right'] = 0.9
        p.rcParams['figure.subplot.left'] = 0.1
        p.rcParams['figure.subplot.bottom'] = 0.1
        p.rcParams['figure.subplot.top'] = 0.9
#        p.rcParams['savefig.bbox'] = "tight"       
        for j in range(ngraph):
            fig1 = p.figure(num = numfig, figsize=(15,10))
#            print p.setp(fig1)
#            print p.getp(fig1)
            ax = p.subplot(ngraphline, ngraphcol, j+1)
#            print "plotdat = ", plotdat[:,:,3*j:3*(j+1)]
            imrgb = p.imshow(plotdat[:,:,3*j:3*(j+1)], interpolation='nearest', extent=extent)
#            print p.setp(imrgb)
            if col_for_simple_map  is None : strname = dict_nplot[maptype][6][j]
            else : strname = col_for_simple_map
            if remove_ticklabels_titles == 0 : p.title(strname)
            if remove_ticklabels_titles :
#                print p.getp(ax)
                p.subplots_adjust(wspace = 0.05,hspace = 0.05)
                p.setp(ax,xticklabels = [])
                p.setp(ax,yticklabels = [])
            if plot_grid :
                ax.grid(color=color_grid, linestyle='-', linewidth=2)
                                
            if PAR.cr_string == "\n":                    
                ax.locator_params('x', tight=True, nbins=5)
                ax.locator_params('y', tight=True, nbins=5)
            if remove_ticklabels_titles == 0 :
                if (j == ngraphlabels) :
                    p.xlabel("dxech (microns)")
                    p.ylabel("dyech (microns)")
            if zoom == "yes" :
                p.xlim(minxj, maxxj)
                p.ylim(minyi, maxyi)
            if xylim_new  is  not None :
                p.xlim(xylim_new[0], xylim_new[1])
                p.ylim(xylim_new[2], xylim_new[3])

            if add_symbols_for_filtered and filter_on_pixdev_and_npeaks :

                ms1 = 8
                ms1 = 3
                xysymb = list_xy_filtered
                if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "x", ms = ms1, mew = 1, mec = "w", mfc = "None") 
                else : p.plot(xysymb[:,0], xysymb[:,1], "x", ms = ms1, mew = 1, mec = "w", mfc = "None")                          
            
            if add_symbols :
                if (maptype == "fit") :
                    if j == 0 :  # npeaks map
                        xysymb = list_xy_npeaks_below_level3
                        if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "wx", mew = 2) 
                        else : p.plot(xysymb[:,0], xysymb[:,1], "wx", mew = 2)      # "_"          
                    if j == 1 :  # pixdev map
#                        xysymb = list_xy_pixdev_above_level1
#                        p.plot(xysymb[:,0], xysymb[:,1], 'o', mec = "k", mfc = "None")
                        xysymb = list_xy_pixdev_above_level2
                        if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "wx", mew = 2) 
                        else : p.plot(xysymb[:,0], xysymb[:,1], 'w+', mew = 2)
                
                elif maptype in list_of_positive_or_negative_quantities :
                    
                    if 0 :
                        ind2 = where(ind_strain_above_max_positive[1] == j)
    #                    print ind2
    #                    print len(ind2[0])
                        if len(ind2[0]) > 0 :
                            xysymb = list_xy_strain_above_max_positive[ind2[0],:]
                            if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "w+", mew = 1 )#, mec = "w", mfc = "None") 
                            else : p.plot(xysymb[:,0], xysymb[:,1], "w+", mew = 1 ) # , mec = "w", mfc = "None")                          

                    ind2 = where(ind_strain_positive[1] == j)
#                    print ind2
#                    print len(ind2[0])
                    if len(ind2[0]) > 0 :
                        xysymb = list_xy_strain_positive[ind2[0],:]
                        if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "o", mew = 1, mec = "w", mfc = "None") 
                        else : p.plot(xysymb[:,0], xysymb[:,1], "o", mew = 1, mec = "w", mfc = "None")                          

                    if 0 :
                        ind2 = where(ind_strain_negative[1] == j)
    #                    print ind2
    #                    print len(ind2[0])
                        if len(ind2[0]) > 0 :
                            xysymb = list_xy_strain_negative[ind2[0],:]
                            if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "s", mew = 2, mec = "w", mfc = "None") 
                            else : p.plot(xysymb[:,0], xysymb[:,1], "s", mew = 2, mec = "w", mfc = "None")                          
    
                    if 0 :
                        ind2 = where(ind_strain_below_min_negative[1] == j)
    #                    print ind2
    #                    print len(ind2[0])
                        if len(ind2[0]) > 0 :
                            xysymb = list_xy_strain_below_min_negative[ind2[0],:]
                            if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "w_", mew = 1 )#, mec = "w", mfc = "None") 
                            else : p.plot(xysymb[:,0], xysymb[:,1], "w_", mew = 1 ) # , mec = "w", mfc = "None")                          

                
        if savefig :
            figfilename = fileprefix + "_" + maptype + ".png"
            print(figfilename)
            p.savefig(figfilename, bbox_inches='tight')                
                    
        return(0)


def class_data_into_grainnum(filesum, 
                             filepathout, 
                             tol1 = 0.1, 
                             test_mode = "yes",
                             prefix_filesum_with_gnum = "summary_with_gnum_",
                             prefix_filegrains = "gtog_",
                             use_rgb_xz_not_xyz = "no",
                             use_rgb_lab = "no"): #29May13
       
    data_list, listname, nameline0 = read_summary_file(filesum)  
    
    data_list = np.array(data_list, dtype=float)
    
    indimg = listname.index('img')
    indgnumloc = listname.index('gnumloc')
    indnpeaks = listname.index('npeaks')
    if use_rgb_lab == "yes" :
        indrgb = listname.index("rgb_x_lab_0")
        rgbstr = "use_rgb_lab_"
    else :
        indrgb = listname.index("rgb_x_sample_0")
        rgbstr = ""
    
    local_gnumlist = np.array(data_list[:,indgnumloc],dtype = int)       
    npeakslist = np.array(data_list[:,indnpeaks],dtype = int)
    
    ind1 = where(npeakslist > 1)
             
    print(ind1[0])
    data_list2 = data_list[ind1[0],:]
    numig2 = shape(data_list2)[0]
    print(numig2)
    
    if use_rgb_xz_not_xyz == "no" :
        indrgbxyz = np.arange(indrgb, indrgb+9)
        ncolrgb = 9
    elif use_rgb_xz_not_xyz == "yes":
        indrgbxyz = np.array([indrgb, indrgb+1, indrgb+2, indrgb+6, indrgb+7, indrgb+8])
        ncolrgb = 6
    
    local_gnumlist2 = local_gnumlist[ind1[0]]
    rgbxyz = data_list2[:,indrgbxyz]
    imglist = data_list2[:,indimg]
    print(rgbxyz[:2,:])
    print(np.shape(rgbxyz))
    
    # norme(rgb) = 1
    for i in range(numig2):
        # la je renormalise chaque rgb
        rgbxyz[i,:3]=rgbxyz[i,:3]/norme(rgbxyz[i,:3])
        rgbxyz[i,3:6]=rgbxyz[i,3:6]/norme(rgbxyz[i,3:6])
        if use_rgb_xz_not_xyz == "no" :
            rgbxyz[i,6:9]=rgbxyz[i,6:9]/norme(rgbxyz[i,6:9])
        
    toto = column_stack((np.arange(numig2), imglist, local_gnumlist2, rgbxyz))
    
    #        print toto[:10,:]
    #        print toto.transpose()[:,:10]
    
    toto1 = toto.transpose()
    
    # nested sort 
    # Sort on last row, then on 2nd last row, etc.
    ind2 = np.lexsort(toto1)
    
    print(shape(ind2))
    print(toto1.take(ind2[:10], axis=-1))
    
    if test_mode == "yes" :
        nmax = 1000
        verbose = 1
        teststr = "test"
    else :
        nmax = numig2
        verbose = 0
        teststr = "all2"
    
    sorted_list = toto1.take(ind2[:nmax], axis=-1).transpose()
    
    print(shape(sorted_list))
    
    dict_grains = {}
    num_one_pixel_grains = 0
    ig_list = np.array(sorted_list[:,0],dtype=int)
    img_list = np.array(sorted_list[:,1],dtype=int)
    gnumloc_list = np.array(sorted_list[:,2],dtype=int)
    
    has_grain_num = zeros(nmax,int)
    is_ref = zeros(nmax,int)
    grain_size = zeros(nmax, int)
    global_gnum = zeros(nmax,int)
    gnum = 0
    for i in range(nmax):
        if has_grain_num[i] == 0 :
            print("i, gnum = ", i, gnum)
            rgbref = sorted_list[i,-ncolrgb:]
#            print shape(rgbref)
            global_gnum[i] = gnum
            is_ref[i] = 1
            has_grain_num[i] = 1
            grain_size[i] = 1
            ind_in_grain_list = [i,]
            #ig_in_grain_list = [ig_list[i],]
            for j in range(i+1,nmax):
                if has_grain_num[j] == 0 :
                    drgb = norme(sorted_list[j,-ncolrgb:]-rgbref)
                    if verbose : print("j, drgb = ", j, round(drgb,3), end=' ')
                    if drgb < tol1 :
                        has_grain_num[j] = 1
                        global_gnum[j] = gnum
                        if verbose : print("gnum =", gnum)
                        grain_size[i]=grain_size[i]+1
                        ind_in_grain_list.append(j)
                        #ig_in_grain_list.append(ig_list[j])
                        
                    else : 
                        if verbose : print(" ")
            if grain_size[i] == 1 :
                num_one_pixel_grains = num_one_pixel_grains + 1
                global_gnum[i] = -1
                print(" ")
            else :
                grain_size[ind_in_grain_list]=grain_size[i]
                short_rgb = sorted_list[ind_in_grain_list,-ncolrgb:]
                mean_rgb = short_rgb.mean(axis=0)
                std_rgb = short_rgb.std(axis=0)*1000.
                range_rgb = (short_rgb.max(axis=0)-short_rgb.min(axis=0))*1000.
                
                ig_in_grain_list = ig_list[ind_in_grain_list]
                img_in_grain_list = img_list[ind_in_grain_list]
                gnumloc_in_grain_list = gnumloc_list[ind_in_grain_list]
                
                print("grain_size = ", grain_size[i])
                #print "ind_in_grain_list ", ind_in_grain_list
                print("ig_in_grain_list ", ig_in_grain_list)
                print("img_in_grain_list", img_in_grain_list)
                print("gnumloc_in_grain_list", gnumloc_in_grain_list)
                print("rgb in grain :")
                print("mean", mean_rgb.round(decimals=3))
                print("std*1000 ", std_rgb.round(decimals=3))
                print("range*1000 ", range_rgb.round(decimals=3))
                print("\n")
                
                dict_grains[gnum] = [grain_size[i],ind_in_grain_list,ig_in_grain_list,\
                        img_in_grain_list,gnumloc_in_grain_list,\
                mean_rgb.round(decimals=3),std_rgb.round(decimals=3),range_rgb.round(decimals=3)]
                
                gnum = gnum+1
    
    print("##########################################################")
    print("gnum = ", gnum)
    print("num_one_pixel_grains =", num_one_pixel_grains)
    
    gnumtot = gnum
    
    #print dict_grains
    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",
                         "img_in_grain_list","gnumloc_in_grain_list",
                         "mean_rgb", "std_rgb *1000", "range_rgb *1000"]
                    
    ndict = len(dict_values_names)
    
    # renumerotation des grains pour classement par taille decroissante May13
    gnumlist = np.arange(gnumtot)
    gsizelist = zeros(gnumtot, int)
    
    for key, value in dict_grains.items():
        gsizelist[key] = value[0]
    
    gnum_gsize_list = column_stack((gnumlist,gsizelist))
    sorted_gnum_gsize_list = sort_list_decreasing_column(gnum_gsize_list, 1)
    #print gnum_gsize_list
    #print sorted_gnum_gsize_list
    dict_grains2 = {}
    for gnum2 in range(gnumtot):
#        print gnum2
#        print sorted_gnum_gsize_list[gnum2,0]
        dict_grains2[gnum2] = dict_grains[sorted_gnum_gsize_list[gnum2,0]]
          
#    for i in range(ndict):
#        print dict_values_names[i]
#        for key, value in dict_grains2.iteritems():
#            print key,value[i]
#    klmdqs
    
    ig_list = column_stack((np.array(sorted_list[:,:3],dtype=int),\
                         is_ref, has_grain_num, global_gnum, grain_size, np.arange(nmax)))
    
    header = "ig 0, img 1, local_gnum 2, is_ref 3, has_grain_num 4, global_gnum 5, grain_size 6, igsort 7"
    print(header)
    
    sorted_ig_list = sort_list_decreasing_column(ig_list, 6)
    
    #print sorted_ig_list
    
    # nouveaux numeros de grain dans liste ig
    for i in range(nmax):
        if sorted_ig_list[i,5] != -1 :
            ind1 = where(sorted_gnum_gsize_list[:,0]==sorted_ig_list[i,5])
            sorted_ig_list[i,5] = ind1[0]
            
    print(sorted_ig_list)
    
    if 1 :
        outfilegnum = filepathout + prefix_filesum_with_gnum + rgbstr + teststr + ".txt"
        print("column results saved in :")
        print(outfilegnum)
        outputfile = open(outfilegnum,'w')
        outputfile.write(header+"\n")
        np.savetxt(outputfile, sorted_ig_list, fmt = "%d")
        outputfile.close()
    
    if 1 :
        outfilegtog = filepathout + prefix_filegrains + rgbstr + teststr +".txt"
        print("dictionnary results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog,'w')
        for i in range(ndict):
            outputfile.write(dict_values_names[i]+"\n")
            for key, value in dict_grains2.items():
                if i == 0 : str1 = str(value[i])
                else : str1 = ' '.join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 +"\n")
        outputfile.close()
    
    # taille de grains moyenne
    list1 = []
    for key, value in dict_grains2.items():
        #print key,value[0]
        list1.append(value[0])
        
    toto = np.array(list1,dtype = float)
    print("mean grain size (units = map pixels) ", round(toto.mean(),2))
    
    # mosaique de grain moyenne
    list1 = []
    for key, value in dict_grains2.items():
        #print key,value[6]
        list1.append(value[6])
        
    toto = np.array(list1,dtype = float)
    print("mean grain std_rgb * 1000 ", toto.mean(axis = 0).round(decimals =2))
    
    return(dict_grains2, outfilegnum, outfilegtog)

def read_dict_grains(filegrains, \
                     dict_with_edges = "no", \
                     dict_with_all_cols = "no",\
                     dict_with_all_cols2 = "no",\
                     dict_with_all_cols3 = "no",
                     old_version = 0,
                     include_strain = 1,
                     N_spots_fit = 0): #29May13
    """
    read grain dictionnary file created by class_data_into_grainnum
    or appended by find_grain_edges
    or appended by fill_dict_grains
    or appended by add_intragrain_rotations_to_dict_grains
    or appended by add_twins_to_dict_grains
    """
    print("read_dict_grains")                 
##    listint = [0,1,2,3,4,8,9,10,12,13,50]
##    listfloat = [5,6,7,11]
#    toto = range(14,50)  # la il faut garder une liste et pas un ndarray
#    listfloat = listfloat + toto
#    print listfloat
#    listmixint = [51,]
#    listmixfloat = [52,]
    
    # 0 1 2 3 4 int
    # 5 6 7 float    
    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
                         "img_in_grain_list","gnumloc_in_grain_list",\
                         "mean_rgb", "std_rgb *1000", "range_rgb *1000"]
    # 0 : int, 1 : float                     
    dict_type = [0,0,0,0,
                 0,1,1,1]         # list            
                         
    if dict_with_edges == "yes" : 
        # 8 9 10 int 
        # 11 float 
        # 12 int
        # pixels des frontieres etendues, pixel_line_position pixel_column_position pixel_edge_type

        toto = ["list_line", "list_col", "list_edge", "gnumloc_mean", "list_edge_restricted"]
        dict_values_names = dict_values_names + toto
        dict_type2 = [0,0,0,1,0]
        
        dict_type = dict_type + dict_type2  # list
        
        print(dict_type)     
        print(len(dict_type))
        print(len(dict_values_names))
        
    if dict_with_all_cols == "yes" : 
        # 13 int
        # 14 : 48 float

        if include_strain :                  
            if old_version :
                toto = ['npeaks', 'pixdev', 'intensity', 
         'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
         'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
         'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
         'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
         'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
         'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
         'max_rss', 'von_mises']
                dict_type2 = [0,1,1,
                              1,1,1,1,1,1,
                              1,1,1,1,1,1,
                              1,1,1,
                              1,1,1,
                              1,1,1,1,1,1,
                              1,1,1,1,1,1,
                              1,1]
            else :
                if N_spots_fit :  # Odile's version
                    toto = ['npeaks',    'pixdev',   'intensity', 'maxpixdev',  'stdpixdev',
         'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 
         'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
         'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 
         'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
         'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2',
          'rgb_y_sample_0', 'rgb_y_sample_1', 'rgb_y_sample_2', 
         'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
         "rgb_x_lab_0", "rgb_x_lab_1", "rgb_x_lab_2", 
         "rgb_y_lab_0", "rgb_y_lab_1", "rgb_y_lab_2", 
         "rgb_z_lab_0", "rgb_z_lab_1", "rgb_z_lab_2",  
         'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 
         'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
         'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 
         'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
         'max_rss', 'von_mises' ]
                    dict_type2 = [0,1,1,1,1,  
                                   1,1,1,1,1,1,  
                                   1,1,1,1,1,1,   
                                   1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,1,1,1,
                                  1,1,1,1,1,1,
                                  1,1]
                              
                else :  # JSM's mixed version 
                    toto = ['npeaks',    'pixdev',   'intensity',
                     'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 
                     'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
                     'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 
                     'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
                     'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2',
                      'rgb_y_sample_0', 'rgb_y_sample_1', 'rgb_y_sample_2', 
                     'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
                     "rgb_x_lab_0", "rgb_x_lab_1", "rgb_x_lab_2", 
                     "rgb_y_lab_0", "rgb_y_lab_1", "rgb_y_lab_2", 
                     "rgb_z_lab_0", "rgb_z_lab_1", "rgb_z_lab_2",  
                     'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 
                     'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
                     'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 
                     'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
                     'max_rss', 'von_mises' ]
                    dict_type2 = [0,1,1,
                                   1,1,1,1,1,1,  
                                   1,1,1,1,1,1,   
                                   1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,
                                  1,1,1,1,1,1,
                                  1,1,1,1,1,1,
                                  1,1]                              
        else :  # include_strain = 0
            toto = ['npeaks', 'pixdev', 'intensity', 
                    'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
                    'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2']    
            dict_type2 = [0,1,1,
                          1,1,1,
                          1,1,1]    
                          
        dict_values_names = dict_values_names + toto
        dict_type = dict_type + dict_type2
        
        print(dict_type)     
        print(len(dict_type))
        print(len(dict_values_names))

    if dict_with_all_cols2 == "yes" : 
        # 48,49 : float
        if old_version :
            toto = ['matstarlab_mean', 'misorientation_angle']
            dict_type2 = [1,1]            
        else :
            toto = ['matstarlab_mean', 'misorientation_angle', "w_mrad_0", "w_mrad_1", "w_mrad_2"]
            dict_type2 = [1,1,1,1,1]
        dict_values_names = dict_values_names + toto
        dict_type = dict_type + dict_type2        
        print(dict_type)     
        print(len(dict_type))
        print(len(dict_values_names))        
        
    if dict_with_all_cols3 == "yes" : 
        # 50 : int, 51 : mixint, 52 : mixfloat
        toto = ['gnum_twin_list', 'HKL_twin_axis_list', 'xyz_twin_axis_list']
        dict_values_names = dict_values_names + toto
        dict_type2 = [0,2,3]
        dict_type = dict_type + dict_type2        
        print(dict_type)     
        print(len(dict_type))
        print(len(dict_values_names)) 
    
#    print dict_values_names                  
    ndict = len(dict_values_names)
    print("ndict = ", ndict)
#    print dict_values_names
    linepos_list = zeros(ndict+1, dtype = int)
    
    f = open(filegrains, 'r')
    i = 0              
    try:
        for line in f:
            for j in range(ndict):
                if line.rstrip(PAR.cr_string) == dict_values_names[j] : 
                    linepos_list[j] = i          
#                    print dict_values_names[j], linepos_list[j]
            i = i + 1
    finally:
        f.close()
        
    linepos_list[-1] = i
        
    print(linepos_list)
    print("len(linepos_list) = ", len(linepos_list))

    ngrains = linepos_list[1]-linepos_list[0]-1
    
    print("ngrains = ", ngrains)
    
    f = open(filegrains, 'r')
    # Read in the file once and build a list of line offsets
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)
    
    f.seek(0)
    print("line at offset 0")
    print(f.readline())
    
    dict_grains = {}
    
    dict_data_type = {0: int,
                  1 : float,
                  2 : "mixint",
                  3 : "mixfloat"}
    
    for j in range(ndict) :
        data_type = dict_data_type[dict_type[j]] 
        n = linepos_list[j]
        # Now, to skip to line n (with the first line being line 0), just do
        f.seek(line_offset[n])
        print("line at offset : ", n)
        print(f.readline())
        print(data_type)
        print()
#        print "yoho"
        f.seek(line_offset[n+1])
        i = 0
        while (i<ngrains):
            toto = f.readline()
#            if j> 49 : print toto
            if toto[-4:] == "\r\n": 
                toto1 = (toto.rstrip('\r\n').split(": "))[1]
            else :
                toto1 = (toto.rstrip('\n').split(": "))[1]
            # version string plus lisible pour verif initiale
            #if n == 0 : dict_grains[i] = "[" + toto1 + "]"
            #else : dict_grains[i] = dict_grains[i] + "[" + toto1 + "]"
            # version array
#            print len(toto1)
            if n == 0 : dict_grains[i] =[int(toto1),]          
            else : 
                if len(toto1) == 1 : # pour gnum_twin_list et HKL_twin_list
                    toto2 = None
                else :
                    if (data_type != "mixint")&(data_type != "mixfloat") :
                        toto2 = np.array(toto1.split(' '), dtype=data_type)
                    elif data_type == "mixint" : 
                        toto2 = np.array(toto1.replace('[', '').replace(']', '').split(), dtype=int)
                    elif data_type == "mixfloat" : 
                        toto2 = np.array(toto1.replace('[', '').replace(']', '').split(), dtype=float)
 
                dict_grains[i].append(toto2)
            i = i+1
    
    f.close()
    
    # version string
    #print dict_grains
    
    # version array
#    for i in range(ndict):
#    for i in range(50,53):
#        print dict_values_names[i]
#        for key, value in dict_grains.iteritems():
#            print key,value[i]

    print(dict_values_names)
            
    return(dict_grains, dict_values_names)

def neighbors_list(img, map_imgnum, verbose = 0): #29May13
    
    #8 positions particulieres
    #bords : droit 1 gauche 2 haut 4 bas 8
    #coins  : haut droit 5 haut gauche 6 bas droit 9 bas gauche 10
    
#    dict_pixtype = { 0:"center",
#                     1:"right",
#                     2:"left"}

    # input : un numero d'image
    # output : 
    # ligne colonne pour les pixels cen right left top bottom
    # type de pixel pour cette image : centre 0 / bord 1,2,4,8 / coin 5,6,9,10
    # img num pour les pixels cen right left top bottom
    # avec conditions aux limites periodiques en bord de carto
    
        
    if verbose :  print(shape(map_imgnum))
    mapsize = np.array(shape(map_imgnum), dtype = int)
        
    #img = 122
    pixtype = 0
    ind1 = where(map_imgnum == img)
    if verbose :  
        print("img =", img)
        print(shape(ind1)[1])
    
    if shape(ind1)[1] == 0 : 
        print("img not in map")
        uoezazae
    else :
        #print ind1
        ind_cen = np.array([ind1[0][0], ind1[1][0]])
        if verbose : print("cen" , ind_cen)
        
        ind_right = ind_cen + array([0,1])
        ind_left = ind_cen + array([0,-1])
        ind_top = ind_cen + array([-1,0])
        ind_bottom = ind_cen + array([1,0])
        
        listpix = [ind_cen,ind_right,ind_left, ind_top, ind_bottom]
        if ind_right[1] > (mapsize[1]-1) : 
            print("img at right edge of map") 
            pixtype = pixtype + 1
            ind_right[1] = ind_right[1]-mapsize[1]
        if ind_left[1]< 0 :
            print("img at left edge of map") 
            pixtype = pixtype + 2
        if ind_top[0]<0 : 
            print("img at top edge of map")
            pixtype = pixtype + 4
        if ind_bottom[0]> (mapsize[0]-1) :
            print("img at bottom edge of map")
            pixtype = pixtype + 8
            ind_bottom[0] = ind_bottom[0]-mapsize[0]
            
        #print listpix
        
        listpix2 = np.array(listpix, dtype=int)
        list_neighbors = zeros(5, int)
        for i in range(shape(listpix2)[0]):
                list_neighbors[i]= map_imgnum[listpix2[i,0],listpix2[i,1]]
        if verbose :
            # ligne colonne pour les pixels cen right left top bottom
            print(listpix2)
            # type de pixel centre 0 / bord 1 2 4 8 / coin 5 6 9 10
            print("pixtype = ", pixtype)
            # img num pour les pixels cen right left top bottom
            print(list_neighbors)
        return(listpix2, pixtype, list_neighbors)


def find_grain_edges(filegrains, filexyz): #29May13
    
    # modifie 28Feb13 : ajout frontiere restreinte

#    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
#                         "img_in_grain_list","gnumloc_in_grain_list",\
#                         "mean_rgb", "std_rgb *1000", "range_rgb *1000"]
#    
    dict_grains, dict_values_names = read_dict_grains(filegrains)
    
    print(dict_values_names[3])
    for key, value in dict_grains.items():
        print(key,value[3])
    
    map_imgnum, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)
    #img = 6481
    #listpix, pixtype, list_neighbors = neighbors_list(img, map_imgnum, verbose = 1)
    #jkldsq
    
    ngrains = len(list(dict_grains.keys()))
    
    # pixtype, indtest
    dict_neigh ={0:[1,2,3,4], 1:[2,3,4], 2:[1,3,4], 4:[1,2,4],
                 8:[1,2,3], 5:[2,4], 6:[1,4], 9:[2,3], 10 : [1,3]} 
    
    # test
    #ngrains = 2    
    dict_grains2 = {}    
    #gnum0 = 106
    
    for gnum in range(ngrains):
    #for gnum in [gnum0,]:
        
        dict_grains2[gnum] = dict_grains[gnum]
    
        list_img = dict_grains[gnum][3]
        nimg = len(list_img)
        list_edge = zeros(nimg,dtype = int)
        list_edge_restricted = zeros(nimg,dtype = int)
        bitwise = array([1,2,4,8])
    
        print("gnum = ", gnum)
        print(dict_values_names[3])
        print(dict_grains[gnum][3])
        print(dict_values_names[4])
        print(dict_grains[gnum][4])
        
        list_line = zeros(nimg, dtype=int)
        list_col = zeros(nimg, dtype=int)
        
        gnumloc_list = np.array(dict_grains[gnum][4], dtype = float)
        
        gnumloc_min = gnumloc_list.min()
       
        gnumloc_min = int(round(gnumloc_min,0))
        print("gnumloc_min = ", gnumloc_min)
        gnumloc_mean = gnumloc_list.mean()
        print("gnumloc_mean = ", gnumloc_mean)
        #gnumloc_mean_int = int(gnumloc_mean + 0.5)
        #print "gnumloc_mean_int = ", gnumloc_mean_int
        
        for i in range(nimg):
            edge1 = zeros(4,dtype = int)
            edge2 = zeros(4,dtype = int)
            img = list_img[i]
            gnumloc = dict_grains[gnum][4][i]
            
            listpix, pixtype, list_neighbors = neighbors_list(img, map_imgnum, verbose = 0)
            indtest = dict_neigh[pixtype]
            # list_neighbors[indtest] donne les img voisines a tester 
            #print pixtype, indtest
            for j in indtest :
                if list_neighbors[j] not in list_img : 
                    edge1[j-1] = 1
                    
            if gnumloc != gnumloc_min : edge2[j-1] = 0
            else :  
                for j in indtest :
                    if list_neighbors[j] not in list_img : 
                        edge2[j-1] = 1
                    else : # list_neighbors[j] in list_img
                        ind0 = where(list_img ==list_neighbors[j] )
                        #print ind0[0][0]
                        gnumloc_neighbor = dict_grains[gnum][4][ind0[0][0]]
                        if gnumloc_neighbor > gnumloc_min : edge2[j-1] = 1                 
                    
            #print edge1
            list_edge[i] = inner(edge1,bitwise)
            list_edge_restricted[i] = inner(edge2,bitwise)
            #print "img, gnumloc, edge, edge_restricted ", img, gnumloc, list_edge[i] , list_edge_restricted[i]  
                
            list_line[i] =  listpix[0,0]
            list_col[i] = listpix[0,1] 
        #print "gnum : ", gnum
        #print "gnumloc : ", dict_grains[gnum][4]
        print("list_edge :",list_edge)
        print("list_edge_restricted :",list_edge_restricted)
            
        dict_grains2[gnum].append(list_line)
        dict_grains2[gnum].append(list_col)
        dict_grains2[gnum].append(list_edge)
        dict_grains2[gnum].append(round(gnumloc_mean,2))
        dict_grains2[gnum].append(list_edge_restricted)
    
    # liste de pixels du grain 
    # line_pix col_pix edge_type_pix
    # edge_type_pix : pour frontiere etendue
    # edge_type_pix = 0 pour pixel pas sur frontiere 
    # edge_type_pix = 1 a 15 pour pixel sur frontiere
    # 1 2 4 8 code les frontieres right left top bottom 
    # somme bitwise des codes si plusieurs bords du pixel sont frontiere simultanement     
    toto = ["list_line", "list_col", "list_edge", "gnumloc_mean", "list_edge_restricted"]
    dict_values_names = dict_values_names + toto
    
    ndict = len(dict_values_names)
        
    if 1 :
        outfilegtog = filegrains.rstrip(".txt") + "_with_edges" +".txt"
        print("results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog,'w')
        for i in range(ndict):
            outputfile.write(dict_values_names[i]+"\n")
            for key, value in dict_grains2.items():
                if (dict_values_names[i]== "grain size")|( dict_values_names[i] == "gnumloc_mean") : str1 = str(value[i])
                else : str1 = ' '.join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 +"\n")
        outputfile.close()
        
    return(dict_grains2, outfilegtog)
    
def fill_dict_grains(filesum, 
                     filegrains,                     
                     include_strain = 1,
                     N_spots_fit = 1): #29May13
    
    dict_grains, dict_values_names = read_dict_grains(filegrains, 
                                                      dict_with_edges = "yes",
                                                      N_spots_fit = N_spots_fit)
 

# img gnumloc npeaks pixdev intensity dxymicrons_0 dxymicrons_1 
# matstarlab_0 matstarlab_1 matstarlab_2 matstarlab_3 matstarlab_4 matstarlab_5 matstarlab_6
# matstarlab_7 matstarlab_8 
# euler3_0 euler3_1 euler3_2 
# maxpixdev stdpixdev 
# strain6_crystal_0 strain6_crystal_1 strain6_crystal_2 
# strain6_crystal_3 strain6_crystal_4 strain6_crystal_5 
# strain6_sample_0 strain6_sample_1 strain6_sample_2 
# strain6_sample_3 strain6_sample_4 strain6_sample_5 
#rgb_x_sample_0 rgb_x_sample_1 rgb_x_sample_2 
# rgb_y_sample_0 rgb_y_sample_1 rgb_y_sample_2 
# rgb_z_sample_0 rgb_z_sample_1 rgb_z_sample_2 
# rgb_x_lab_0 rgb_x_lab_1 rgb_x_lab_2 
# rgb_y_lab_0 rgb_y_lab_1 rgb_y_lab_2 
# rgb_z_lab_0 rgb_z_lab_1 rgb_z_lab_2 
# stress6_crystal_0 stress6_crystal_1 stress6_crystal_2 
# stress6_crystal_3 stress6_crystal_4 stress6_crystal_5 
# stress6_sample_0 stress6_sample_1 stress6_sample_2 
# stress6_sample_3 stress6_sample_4 stress6_sample_5 
# res_shear_stress_0 res_shear_stress_1 res_shear_stress_2 
# res_shear_stress_3 res_shear_stress_4 res_shear_stress_5 
# res_shear_stress_6 res_shear_stress_7 res_shear_stress_8 
# res_shear_stress_9 res_shear_stress_10 res_shear_stress_11 
# max_rss von_mises 

    list_column_names_to_add =  ['npeaks', 
                                     'pixdev', 
                                     'intensity']
    if N_spots_fit :
        list_column_names_to_add = list_column_names_to_add +  ['maxpixdev','stdpixdev']
 
    list_column_names_to_add = list_column_names_to_add + \
             ['rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2',
              'rgb_y_sample_0', 'rgb_y_sample_1', 'rgb_y_sample_2', 
             'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
             "rgb_x_lab_0", "rgb_x_lab_1", "rgb_x_lab_2", 
             "rgb_y_lab_0", "rgb_y_lab_1", "rgb_y_lab_2", 
             "rgb_z_lab_0", "rgb_z_lab_1", "rgb_z_lab_2"]

    if include_strain:
        list_column_names_to_add = list_column_names_to_add + \
                    [ 'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 
                 'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
                 'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 
                 'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
                 'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 
                 'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
                 'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 
                 'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
                 'max_rss', 'von_mises' ]

    print(len(list_column_names_to_add))

    data_list, listname, nameline0 = read_summary_file(filesum)  
    
    data_list = np.array(data_list, dtype=float)
    
    indimg = listname.index('img')
    indgnumloc = listname.index('gnumloc')
        
    img_list = np.array(data_list[:,indimg],dtype = int)
    gnumloc_list = np.array(data_list[:,indgnumloc],dtype = int)
        
#    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
#                         "img_in_grain_list","gnumloc_in_grain_list",\
#                         "mean_rgb", "std_rgb *1000", "range_rgb *1000",\
#                         "list_line", "list_col", "list_edge"]
                         
    print(dict_values_names[0],  dict_values_names[3], dict_values_names[4])
    
    indimg_d = dict_values_names.index('img_in_grain_list')
    indgnumloc_d = dict_values_names.index('gnumloc_in_grain_list')
    indgrainsize_d = dict_values_names.index('grain size')
    
    dict_grains2 = dict_grains 

    #print listname       
    
    for col_name in list_column_names_to_add :
        indcoladd = listname.index(col_name)
        dict_values_names = dict_values_names + [col_name,]
        print(col_name)
        if col_name == 'npeaks':  # int
            data_col_add = np.array(data_list[:,indcoladd].round(decimals=0), dtype = int)
            for key, value in dict_grains.items():
                print(key,value[indgrainsize_d]) #, "\n", value[3],"\n", value[4]               
                list1 = []           
                nimg = value[indgrainsize_d]           
                for i in range(nimg):
                    ind1 = where((img_list == value[indimg_d][i])&(gnumloc_list == value[indgnumloc_d][i]))
                    #print ind1[0][0]
                    j = ind1[0][0]
                    list1.append(data_col_add[j])          
                #print list1
                dict_grains2[key].append(list1)
        else :   # float
            data_col_add = np.array(data_list[:,indcoladd], dtype = float)
            # nb of decimals for storage
            if (col_name[:3] == "rgb")|(col_name == 'pixdev') : ndec = 3
            elif col_name == 'stdpixdev' : ndec = 4
            else : ndec = 2
            for key, value in dict_grains.items():
                print(key,value[indgrainsize_d]) #, "\n", value[3],"\n", value[4]                
                list1 = []           
                nimg = value[indgrainsize_d]           
                for i in range(nimg):
                    ind1 = where((img_list == value[indimg_d][i])&(gnumloc_list == value[indgnumloc_d][i]))
                    #print ind1[0][0]
                    j = ind1[0][0]
                    list1.append(round(data_col_add[j], ndec))            
                #print list1
                dict_grains2[key].append(list1)
    
    ndict = len(dict_values_names)
    print(ndict)
        
    if 1 :
        outfilegtog = filegrains.rstrip(".txt") + "_filled" +".txt"
        print("dictionnary results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog,'w')
        for i in range(ndict):
            outputfile.write(dict_values_names[i]+"\n")
            for key, value in dict_grains2.items():
                if i == 0 : str1 = str(value[i])
                else : str1 = ' '.join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 +"\n")
        outputfile.close()
            
    return(dict_grains2, outfilegtog)

def add_intragrain_rotations_to_dict_grains(filesum, 
                                            filegrains,
                                            filter_mean_matrix_by_pixdev_and_npeaks = 1, 
                                            maxpixdev_for_mean_matrix = 0.25,
                                            minnpeaks_for_mean_matrix = 20,
                                            filter_mean_matrix_by_intensity = 0, 
                                            min_fraction_of_max_intensity_for_mean_matrix = 0.4,
                                            old_version = 0,
                                            include_strain = 1,
                                            N_spots_fit = 1): 
    
    dict_grains, dict_values_names = read_dict_grains(filegrains, 
                                                      dict_with_edges = "yes", 
                                                      dict_with_all_cols = "yes",
                                                      old_version = old_version,
                                                      include_strain = include_strain,
                                                      N_spots_fit = N_spots_fit)
    
    print(dict_values_names)
    
    list_column_names_to_add =  ["matstarlab_mean", 'misorientation_angle', "w_mrad_0", "w_mrad_1", "w_mrad_2"]

    print(len(list_column_names_to_add))

    data_list, listname, nameline0 = read_summary_file(filesum)  
    
    data_list = np.array(data_list, dtype=float)
    
    indimg = listname.index('img')
    indgnumloc = listname.index('gnumloc')
          
    img_list = np.array(data_list[:,indimg],dtype = int)
    gnumloc_list = np.array(data_list[:,indgnumloc],dtype = int)
         
#    dict_values_names = ["grain size", "ind_in_grain_list","ig_in_grain_list",\
#                         "img_in_grain_list","gnumloc_in_grain_list",\
#                         "mean_rgb", "std_rgb *1000", "range_rgb *1000",\
#                         "list_line", "list_col", "list_edge"]
                         
    #print dict_values_names[0],  dict_values_names[3], dict_values_names[4]
    
    indimg_d = dict_values_names.index('img_in_grain_list')
    indgnumloc_d = dict_values_names.index('gnumloc_in_grain_list')
    indgrainsize_d = dict_values_names.index('grain size')
    indpixdev_d = dict_values_names.index('pixdev')
    indnpeaks_d = dict_values_names.index('npeaks')
    indintensity_d = dict_values_names.index('intensity')
    
    print(indimg_d, indgnumloc_d,  indgrainsize_d, indpixdev_d, indnpeaks_d)
    
    dict_grains2 = dict_grains 

    #print listname       
    
    indmat = listname.index("matstarlab_0")
    dict_values_names = dict_values_names + list_column_names_to_add

    matstarlab_all = np.array(data_list[:,indmat:indmat+9], dtype = float)

    indangle_d = dict_values_names.index('misorientation_angle')
    indmat_d = dict_values_names.index("matstarlab_mean")
    
    listratio = []
    listdiff = []
    listbad = []

    for key, value in dict_grains.items():
        nimg = value[indgrainsize_d] 
        print("gnum ", key, "gsize", nimg) #, "\n", value[3],"\n", value[4]                
        matstarlab_ig = zeros((nimg,9), dtype=float) 
        img_list_d =  np.array(value[indimg_d],dtype = int)
        gnumloc_list_d = np.array(value[indgnumloc_d],dtype=int)
        pixdev_list_d = np.array(value[indpixdev_d],dtype=float)
        npeaks_list_d = np.array(value[indnpeaks_d],dtype=int)
        intensity_list_d = np.array(value[indintensity_d],dtype=float)
        
        min_intensity = min_fraction_of_max_intensity_for_mean_matrix * max(intensity_list_d)
        
        for i in range(nimg):
            #print "i = ", i
            ind1 = where((img_list == img_list_d[i])&(gnumloc_list == gnumloc_list_d[i]))
            #print ind1
            #print ind1[0][0]
            j = ind1[0][0]
            #print "j = ", j
            matstarlab_ig[i,:]= matstarlab_all[j,:]
    
#        print matstarlab_ig[:10,:]
#        print pixdev_list_d[:10]
#        print npeaks_list_d[:10]
        
        indfilt = np.arange(nimg)
        if filter_mean_matrix_by_pixdev_and_npeaks :     
            ind1 = where((pixdev_list_d < maxpixdev_for_mean_matrix)&(npeaks_list_d > minnpeaks_for_mean_matrix))      
            indfilt = ind1[0]
        elif filter_mean_matrix_by_intensity :     
            ind1 = where(intensity_list_d > min_intensity)
            indfilt = ind1[0]                     
#        indfilt = where((pixdev_list_d < maxpixdev_for_matmean)&(npeaks_list_d > minnpeaks_for_mean_matrix)) 
                       
        #print indfilt
        #print indfilt[0]
        gsize_filter = len(indfilt)
        print("number of points used to calculate matmean :")
        print("gsize_filter : ", gsize_filter)
        
        ratio1 = float(gsize_filter)/float(nimg)
        print("gsize_filter/gsize ratio : ", round(ratio1, 3))
        listratio.append(round(ratio1, 3))
               
        indfilt0 = where(gnumloc_list_d == 0)
        if len(indfilt0[0])> 0 :
            pixdev_mean_0 = pixdev_list_d[indfilt0[0]].mean()
            print("pixdevmean 0 :", round(pixdev_mean_0,3))

        indfilt1 = where(gnumloc_list_d == 1)
        if len(indfilt1[0])> 0 :
            pixdev_mean_1 = pixdev_list_d[indfilt1[0]].mean()        
            print("pixdevmean 1 :", round(pixdev_mean_1,3))
       
#        if nimg < 50 : 
#            kmsqd
#            print len(listratio)
#            ratio2 = np.array(listratio, dtype = float)
#            print "gsize_filter/gsize ratio : mean std min max range"
#            print round(ratio2.mean(),3), round(ratio2.std(),3), ratio2.min(), ratio2.max(),  ratio2.max()-ratio2.min()
#            diff2 = np.array(listdiff,dtype=float)
#            print "gnumloc_mean_filter-gnumloc_mean_all : mean std min max range"
#            print round(diff2.mean(),3), round(diff2.std(),3), diff2.min(), diff2.max(), diff2.max()-diff2.min()
#
#            hkjdsq
        
        if gsize_filter > 1 :
            matstarlab_mean = matstarlab_ig[indfilt].mean(axis=0)
            gnumloc_mean_all = gnumloc_list_d.mean()
            gnumloc_mean_filter = gnumloc_list_d[indfilt].mean()
            diff1 = gnumloc_mean_filter-gnumloc_mean_all
            print("gnumloc_mean : all (filtered-all)", round(gnumloc_mean_all,3), round(diff1,3))
            listdiff.append(round(diff1,3))     
            
        else :
            matstarlab_mean = matstarlab_ig.mean(axis=0)
            listbad.append(key)
        print("matmean = ", matstarlab_mean)
    
        dict_grains2[key].append(matstarlab_mean.round(decimals= 6))
        
        omegaxyz = zeros((nimg,3),float)
        angle1 = zeros(nimg,float)           

        for k in range(nimg):
#            mat2 = GT.matline_to_mat3x3(matstarlab_ig[k,:])
#            vec_crystal[k,:], vec_lab[k,:],angle1[k] = twomat_to_rotation(matmean3x3,mat2, verbose = 0)
            vecRodrigues_sample, angle1[k] = twomat_to_rotation_Emeric(matstarlab_mean, matstarlab_ig[k,:])
            omegaxyz[k,:] = vecRodrigues_sample * 2. * 1000. # unites = mrad 
            # misorientation_angle : unites = degres
#            print round(angle1[k], 3), omegaxyz[k,:].round(decimals = 2)
#            if k == 5 : return()

        dict_grains2[key].append(angle1.round(decimals=3))
        for j in range(3):
            dict_grains2[key].append(omegaxyz[:,j].round(decimals=2))
            
        print("angle1 : mean, std, min, max")
        print(round(angle1.mean(),3), round(angle1.std(),3), round(angle1.min(),3), round(angle1.max(),3))
        
#        print "new dict entries"
#        print dict_grains2[key][indmat_d]
#        print dict_grains2[key][indangle_d]
#        if key== 2 : return()
    
    ndict = len(dict_values_names)
    print(ndict)
    print(dict_values_names)
        
    if 1 :
        outfilegtog = filegrains.rstrip(".txt") + "_with_rotations" +".txt"
        print("dictionnary results saved in :")
        print(outfilegtog)
        outputfile = open(outfilegtog,'w')
        for i in range(ndict):
            outputfile.write(dict_values_names[i]+"\n")
            for key, value in dict_grains2.items():
                if i == 0 : str1 = str(value[i])
                else : str1 = ' '.join(str(e) for e in value[i])
                outputfile.write(str(key) + " : " + str1 +"\n")
        outputfile.close()
        
    if filter_mean_matrix_by_pixdev_and_npeaks :     
        print("filter matmean by pixdev and npeaks")
        print("maxpixdev_for_mean_matrix :", maxpixdev_for_mean_matrix)
        print("minnpeaks_for_mean_matrix :", minnpeaks_for_mean_matrix)    
    elif filter_mean_matrix_by_intensity :     
        print("filter matmean by intensity")
        print("min_fraction_of_max_intensity_for_mean_matrix : ", min_fraction_of_max_intensity_for_mean_matrix)

    print("list of bad grains : (gsize_filter < 2) ", listbad)       

    return(dict_grains2)
    
        
def grain_stats_from_dict_grains(filegrains, 
                                 gnum_list = [0,], 
                                     variable_name = "misorientation_angle", 
                                     filter_by_pixdev_and_npeaks = 1,
                                     maxpixdev_for_stat = 0.25,
                                     minnpeaks_for_stat = 20, 
                                     filter_by_gnumloc = 0,
                                     gnumloc_for_stat = 0,
                                     dict_with_edges = "yes", 
                                     dict_with_all_cols = "yes",
                                     dict_with_all_cols2 = "yes"):
    
    dict_grains, dict_values_names = read_dict_grains(filegrains, 
                                                  dict_with_edges = dict_with_edges,
                                                  dict_with_all_cols = dict_with_all_cols,
                                                  dict_with_all_cols2 = dict_with_all_cols2 )
                                                  
    print(dict_values_names)
    
    indimg_d = dict_values_names.index('img_in_grain_list')
    indgnumloc_d = dict_values_names.index('gnumloc_in_grain_list')
    indgrainsize_d = dict_values_names.index('grain size')
    indpixdev_d = dict_values_names.index('pixdev')
    indnpeaks_d = dict_values_names.index('npeaks')
    
    indvar_d = dict_values_names.index(variable_name)
    
    print(indimg_d, indgnumloc_d,  indgrainsize_d, indpixdev_d, indnpeaks_d, indvar_d)
    
    if gnum_list == "all" :
        gnum_list = []
        for key, value in dict_grains.items(): gnumlist.append(key)
        gnum_list = np.array(gnum_list,dtype = int)
        
    print(variable_name)    
    for gnum in gnum_list :        
        value = dict_grains[gnum]
        nimg = value[indgrainsize_d] 
        print("gnum ", gnum, "gsize", nimg, end=' ') # "\n", value[3],"\n", value[4]                
        img_list_d =  np.array(value[indimg_d],dtype = int)
        gnumloc_list_d = np.array(value[indgnumloc_d],dtype=int)
        pixdev_list_d = np.array(value[indpixdev_d],dtype=float)
        npeaks_list_d = np.array(value[indnpeaks_d],dtype=int)
        var_list_d = np.array(value[indvar_d],dtype=float)
#        print var_list_d
#        print var_list_d[:10]

        if filter_by_pixdev_and_npeaks :         
            indfilt = where((pixdev_list_d < maxpixdev_for_stat)&(npeaks_list_d > minnpeaks_for_stat)) 
            indf = indfilt[0] 
        elif filter_by_gnumloc :
            indfilt = where(gnumloc_list_d == gnumloc_for_stat)
            indf = indfilt[0] 
        else :
            indf = np.arange(nimg) 
    
        npts = len(indf)
        print("gsize_filter ", npts)
#        print variable_name[-5:]
        
        if (npts > 1)&(variable_name[-5:]!="_mean") :
            var_mean = var_list_d[indf].mean()
            var_std = var_list_d[indf].std()
            var_min = var_list_d[indf].min()
            var_max = var_list_d[indf].max()
            var_range = var_max - var_min
            
            indmin = argmin(var_list_d[indf])
            indmax = argmax(var_list_d[indf])
            #print indmin
            img_list_short = img_list_d[indf]
            gnumloc_list_short = gnumloc_list_d[indf]
            
            imgmin = img_list_short[indmin]
            gnumlocmin = gnumloc_list_short[indmin]
            imgmax = img_list_short[indmax]
            gnumlocmax = gnumloc_list_short[indmax]
        
            print("min/imgmin/gnumlocmin, max/imgmax/gnumlocmax")
            print(round(var_min,3), imgmin, gnumlocmin, "\t", round(var_max,3), imgmax, gnumlocmax)            
            print("mean, std, range :")
            print(round(var_mean,3), " ",  round(var_std, 3), " ", round(var_range,3))

        elif variable_name == "matstarlab_mean" :
#            print "omega_sample_frame = ", PAR.omega_sample_frame
            matstarlab = var_list_d                
            matstarsample3x3 = matstarlab_to_matstarsample3x3(matstarlab,
                                                  omega = None, # was PAR.omega_sample_frame,
                                                  mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame)
            matstarsample = GT.mat3x3_to_matline(matstarsample3x3)   
    
#            mat, rlat = matstarlab_to_matdirlab3x3(matstarsample)
            
        #        print det(mat)
            
        #        print "a b c as columns on x y z sample :"
        #        print mat.round(decimals = 4)
        #        print "columns normalized to largest component :"
        #        for i in range(3):
        #            print (mat[:,i]/abs(mat[:,i]).max()).round(decimals = 4)
                
            mat1 = GT.matline_to_mat3x3(matstarsample)
        #        print "abcstar on xyzsample \n", mat1.round(decimals=6)
            mat2 = mat1.transpose()
        #        print "xyzsample on abcstar \n", mat2.round(decimals=6)
            strlist = ["x","y","z"]
            for i in range(3) :
                print("HKL"+ strlist[i] + "_sample = ", (mat2[:,i]/max(abs(mat2[:,i]))).round(decimals=2))
    
    if filter_by_pixdev_and_npeaks : 
        print("filter by pixdev and npeaks")
        print("maxpixdev_for_stat : ", maxpixdev_for_stat)
        print("minnpeaks_for_stat : ", minnpeaks_for_stat)
    elif filter_by_gnumloc : 
        print("filter by gnumloc")
        print("fixed gnumloc : ", gnumloc_for_stat)
    else : 
        print("no filtering")
    if variable_name == "matstarlab_mean" :
        print("omega_sample_frame = ", PAR.omega_sample_frame)
        print("mat_from_lab_to_sample_frame = ", PAR.mat_from_lab_to_sample_frame)
    
    return(0)
    
    


def testBit(int_type, offset):  #29May13
    mask = 1 << offset
    return(int_type & mask)

if 0 :    # test testBit function  #29May13
    print(testBit(4,3))
    print(testBit(1,1))
    klmdfs

def list_edge_lines(pixel_edge_code): #29May13
    # x y cad col line
    # key = bit number 
    # 0 : 1, 1 : 2, 2 : 4, 3 : 8
    dict_edge_lines = {0: [[1,0],[1,1]],                     
                       1: [[0,0],[0,1]], 
                        2 : [[0,1],[1,1]],
                        3 :[[0,0],[1,0]]
                       }

    list_edge_lines = []
    for bit1 in range(4):
        #print testBit(pixel_edge_code,bit1)
        if testBit(pixel_edge_code,bit1) > 0 : list_edge_lines.append(dict_edge_lines[bit1])
    
    print(pixel_edge_code, end=' ')
    print(": ", end=' ')
    print(list_edge_lines, end=' ')
    print(",")
    #print shape(list_edge_lines)[0]
    
    return(list_edge_lines)    

if 0 : # segments for pixel edges for frontiers     #29May13
    for i in range(16):
         list_edge_lines(i)
    jkdlqs

if 1 : # dict_edge_lines  #29May13
    dict_edge_lines = {
        0 :  [] ,
        1 :  [[[1, 0], [1, 1]]] ,
        2 :  [[[0, 0], [0, 1]]] ,
        3 :  [[[1, 0], [1, 1]], [[0, 0], [0, 1]]] ,
        4 :  [[[0, 1], [1, 1]]] ,
        5 :  [[[1, 0], [1, 1]], [[0, 1], [1, 1]]] ,
        6 :  [[[0, 0], [0, 1]], [[0, 1], [1, 1]]] ,
        7 :  [[[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 1], [1, 1]]] ,
        8 :  [[[0, 0], [1, 0]]] ,
        9 :  [[[1, 0], [1, 1]], [[0, 0], [1, 0]]] ,
        10 :  [[[0, 0], [0, 1]], [[0, 0], [1, 0]]] ,
        11 :  [[[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 0], [1, 0]]] ,
        12 :  [[[0, 1], [1, 1]], [[0, 0], [1, 0]]] ,
        13 :  [[[1, 0], [1, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]]] ,
        14 :  [[[0, 0], [0, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]]] ,
        15 :  [[[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]]] 
    }

def plot_all_grain_maps(filegrains, 
                        filexyz, 
                        zoom = "no", 
                        mapcolor = "gnumloc_in_grain_list", 
                        filter_on_pixdev_and_npeaks = 0,
                        maxpixdev = 0.3,
                        minnpeaks = 20,
                        map_prefix = "z_gnumloc_gmap_2edges_",
                        test1 = "yes",
                        savefig_map = 0,
                        number_of_graphs = None,
                        number_of_graphs_per_figure = 9,
                        grains_to_plot = "all",
                        gnumloc_min = 0, # 0 pour fichiers LT, 1 pour fichiers XMAS
                        gnumloc_max = 3,
                        min_forplot = None,
                        max_forplot = None,
                        single_gnumloc = None,
                        subtract_mean = "no",
                        filepathout = None,
                        gnum_xypos = array([50,50]),
                        xylim = None,
                        remove_ticklabels_titles = 0,
                        filter_on_intensity = 0,
                        min_fraction_of_max_intensity_for_filter = 0.4,
                        dict_grains_type = "with_edges_filled_with_rotations",
                        old_version = 0,
                        include_strain = 1,
                        add_symbols = 0,
                        show_grain_number = 1):
    #30May13
                        
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
#            "rgb_x_sample" "rgb_y_sample" "rgb_z_sample" 'mean_rgb': orientations en rgb
#            "rgb_x_lab" "rgb_y_lab" "rgb_z_lab"
#            
    """
    # key = number_of_graphs_per_figure
    # nb of subplots lines
    # nb of subplots columns
    # numero du subplot pour mettre les labels des axes
    dict_graph = {9:[3,3,7],
                    4:[2,2,3]}
            
    min_nimg = 2
    max_nimg = 10
    #ind_nimg = np.arange(min_nimg, max_nimg)
    
    #ind_gnum = [226,139,254,177,140,154,175,193,46,89]
    #ind_gnum = [120,170,256,127,151,163]
    # macles
    #ind_gnum = [201,254,139,127,249,165,180,154,163,130,89,177,140,226,231,148,224,253,30,110,222,196]
    #print len(ind_gnum)
#    ind_gnum = [228,256,170,175,232,193,202,178,115,168,120,203,151,244,245,102,195,255,58,10,247,181]
#    print len(ind_gnum)
    #jkldqs
    

    # definition of map type    
    # rgb color for missing data  
    # rgb color for filtered data 
    dict_maptype = { 0 : ["grey scale for positive quantities (black = min)",[1.,0.8,0.8],[0.,1.,0.]],
                    1 : ["inverted grey scale for positive quantities (black = max)",[1.,0.8,0.8],[0.,1.,0.]],
                    2 : ["strain-like color scale for signed quantities (red < min - magenta <0 - white 0 - green > 0 - yellow > max)",[0.,0.,0.],[0.5,0.5,0.5]],
                    3 : ["rgb color scale for orientations",[1.,1.,1.],[0.5,0.5,0.5]],
                    4 : ["rgb color from dict_rgb_gnumloc",[1.,1.,1.],[0.5,0.5,0.5]]     
                    }
                    
    # 0 rouge, 1 vert, 2 bleu clair
    # adapter dict_rgb_gnumloc en fonction du gnumloc_max
    
    CdTe = 1
    
    if mapcolor == "gnumloc_in_grain_list" :
        if gnumloc_min == 0 : # LT
            if CdTe :
                dict_rgb_gnumloc = {0:[0.,0.,0.], 1: [0.5, 0.5, 0.5], 2:[0.7,0.7,0.7], 3:[0.9,0.9,0.9] }  # greys
            else :                
                dict_rgb_gnumloc = {0:[1.,0.,0.], 1: [0., 1., 0.], 2:[0.5,0.5,1.], 3:[0.5,1.,0.5], 4:[0.5,0.5,0.8] } 

        elif gnumloc_min == 1 : # XMAS
            dict_rgb_gnumloc = {1:[1.,0.,0.], 2: [0., 1., 0.], 3:[0.5,0.5,1.], 4:[0.5,0.5,0.5] }  # red green blue grey
#            dict_rgb_gnumloc = {1:[1.,1.,1.], 2: [0.5, 0.5, 0.5], 3:[0.2,0.2,0.2], 4:[0.1,0.1,0.1] }  # greys

                    
    # number of decimals to display
    # maptype    
    dict_dec = {"pixdev" : [4,1],
                "npeaks" : [1,0],
                "intensity": [1,0],
                'gnumloc_in_grain_list':[2,4],
                "maxpixdev" : [4,1],
                "stdpixdev" : [4,1]
                }
                
    for key in ['strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 
    'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
    'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 
    'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
    'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 
    'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
    'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 
    'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5',
    'w_mrad_0', 'w_mrad_1', 'w_mrad_2'] : dict_dec[key] = [3,2]

    for key in ["rgb_x_sample", "rgb_y_sample", "rgb_z_sample","rgb_x_lab", "rgb_y_lab", "rgb_z_lab"] : dict_dec[key] = [3,3] 
    for key in ['max_rss', 'von_mises','misorientation_angle'] : dict_dec[key] = [3,1]
    
    
    if filter_on_pixdev_and_npeaks : filter_str = "_filter"
    else : filter_str = ''
    map_prefix = map_prefix + mapcolor + filter_str + "_gmap_"

    p.rcParams['figure.subplot.bottom'] = 0.1
    p.rcParams['figure.subplot.left'] = 0.1   
    
    if dict_grains_type == "with_edges_filled_with_rotations" : 
        dict_with_edges = "yes"
        dict_with_all_cols = "yes"
        dict_with_all_cols2 = "yes"
    elif dict_grains_type == "with_edges_filled" :
        dict_with_edges = "yes"
        dict_with_all_cols = "yes"
        dict_with_all_cols2 = "no"
    elif dict_grains_type == "with_edges" :   
        dict_with_edges = "yes"
        dict_with_all_cols = "no"
        dict_with_all_cols2 = "no"
    elif dict_grains_type  is None :   
        dict_with_edges = "no"
        dict_with_all_cols = "no"
        dict_with_all_cols2 = "no"    
        
    dict_grains, dict_values_names = read_dict_grains(filegrains, 
                                                      dict_with_edges = dict_with_edges,
                                                      dict_with_all_cols = dict_with_all_cols,
                                                      dict_with_all_cols2 = dict_with_all_cols2,
                                                      old_version = old_version,
                                                      include_strain = include_strain
                                                      )
                                                      
#    dict_grains, dict_values_names = read_dict_grains(filegrains, dict_with_edges = "yes")                                                    
    
    ngrains = len(list(dict_grains.keys()))   

    if test1 == "yes" : 
        if ngrains > 15 : ngrains = 15 
        if grains_to_plot != "all" :
            grains_to_plot = grains_to_plot[:15]
                                                     
    print(dict_values_names)

    # grain sizes
    list_gnum = []
    list_grain_size = []    
    print(dict_values_names[0])
    for key, value in dict_grains.items():
        #print key,value[0]
        list_gnum.append(key) 
        list_grain_size.append(value[0])        
    print("list of grain sizes", list_grain_size)   
    #jskqdl
    list_grainsize = np.array(list_grain_size, dtype = float)      
    
    indgnumloc = dict_values_names.index("gnumloc_in_grain_list")            
    ind_pixdev = dict_values_names.index("pixdev")
    ind_npeaks = dict_values_names.index("npeaks") 
    ind_intensity = dict_values_names.index("intensity")            
      
    indline = dict_values_names.index("list_line")
    indcol = dict_values_names.index("list_col")
    indedge = dict_values_names.index("list_edge")
    indedge_restricted = dict_values_names.index("list_edge_restricted")
    
#    if mapcolor in ["rgb_x", "rgb_z"]:
    if mapcolor[:3] == "rgb" :
        first_col_name = mapcolor + "_0"
        indfirstcol = dict_values_names.index(first_col_name)
        indrgbxz = np.arange(indfirstcol, indfirstcol + 3)
    else :
        indplot =  dict_values_names.index(mapcolor)
            
        # calcul des valeurs moyennes de mapcolor sur toute la carto sans filtrage        
        list1 = []        
        #print dict_values_names[indplot]
        for key, value in dict_grains.items():
            #print key,value[indplot]
            list1 = list1 + list(value[indplot])            
        #print list1
        #print len(list1)
        toto = np.array(list1, dtype = float)
        maxvalue = max(toto)
        minvalue = min(toto)
        meanvalue = mean(toto)
        stdvalue = std(toto)
    
        #print mapcolor[:-1]
           
        print("statistics on all data points in grains with grain_size > 1 :")
        print(mapcolor)
        ndec = dict_dec[mapcolor][0]
        if mapcolor[:3] == "rgb" : #: in ["rgb_x", "rgb_z"]:
            print("mean, std, min, max ", meanvalue.round(decimals= ndec), stdvalue.round(decimals= ndec), minvalue, maxvalue)    
        else :
            print("mean, std, min, max ", round(meanvalue, ndec), round(stdvalue, ndec), minvalue, maxvalue)

        
        if min_forplot  is None : min_forplot = minvalue
        if max_forplot  is None : max_forplot = maxvalue
        print("min, max for plot", min_forplot, max_forplot)
    
#    if mapcolor[0]== "w" : colortext = "w"
#    else : 
    colortext = "k"    
    
    maptype = dict_dec[mapcolor][1]
    print("type of color map : ")
    print(dict_maptype[maptype][0])
    color_for_missing_data = dict_maptype[maptype][1]
    print("rgb color for missing data :", color_for_missing_data)
        
    if filter_on_pixdev_and_npeaks :
        print("map filtered by npeaks / pixdev")
        print("maxpixdev, minnpeaks = ", maxpixdev, minnpeaks)  
        color_for_filtered_data = dict_maptype[maptype][2]
        print("rgb color for filtered data :", color_for_filtered_data)       

    if filter_on_intensity :
        print("map filtered by intensity")
        print("min_fraction_of_max_intensity_for_filter =", min_fraction_of_max_intensity_for_filter)
        color_for_filtered_data = dict_maptype[maptype][2]
        print("rgb color for filtered data :", color_for_filtered_data)   
                        
    #jkldqs                                          
    
#    print dict_values_names[3]
#    for key, value in dict_grains.iteritems():
#        print key,value[3]
#    
#    TODO : remettre la rotation mais avec modif des frontieres    
#    filexyz_new = filexyz
#    xylim_new = None
#    xylim = None
#    
#    if abs(PAR.map_rotation)> 0.1 :
#        print "rotating map clockwise by : ", PAR.map_rotation, "degrees"
#        filexyz_new, xylim_new = rotate_map(filexyz, xylim = xylim)  
#            
#    map_imgnum, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz_new)

    map_imgnum, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)   
    nlines = shape(map_imgnum)[0]
    ncol = shape(map_imgnum)[1]
    
    xrange1 = array([0.0,ncol*dxystep[0]])
    yrange1 = array([0.0, nlines*dxystep[1]])
    xmin, xmax = min(xrange1), max(xrange1)
    ymin, ymax = min(yrange1), max(yrange1)
    extent = xmin, xmax, ymin, ymax
    print(extent)
    dxystep_abs = abs(dxystep)
            
    shape_rgb = (nlines,ncol,3)
     
    # decalage entre x y entier et centre pixel suivant position origine x y dans carto
    # haut / bas, droite / gauche
    # impos_start = 0 0 haut gauche, 0 80 haut droit, 100 0 bas gauche, 100 80 bas droit
    
    if impos_start[1]==0 : x_coin_carto = "gauche"
    else : x_coin_carto = "droit" 
    if impos_start[0]==0 : y_coin_carto = "haut"
    else : y_coin_carto = "bas" 
    
    if x_coin_carto == "gauche" : dx_corner_to_center = 0.5
    else : dx_corner_to_center = -0.5
    if y_coin_carto == "haut" : dy_corner_to_center = -0.5
    else : dy_corner_to_center = 0.5 
    
    dxy_corner_to_center_microns = [dx_corner_to_center*dxystep_abs[0],dy_corner_to_center*dxystep_abs[1]]        
   
    if number_of_graphs == 1 : p.figure(figsize=(10,10))
    
    rgb1 = color_for_missing_data * ones(shape_rgb, dtype = float)
       
    if grains_to_plot == "all" :  indgnum = np.arange(ngrains)  
    else : indgnum = grains_to_plot
    
    print(indgnum)
    
    listx = []
    listy = []     
    k=0
    kk = 0
    for gnum in indgnum: 
    
        rgb_rand = rand(3)
        rgb_rand = rgb_rand/max(rgb_rand)

        if ((zoom == "yes")&(number_of_graphs is None))|(number_of_graphs==1):
            listx = []
            listy = []     
        
        list_img = dict_grains[gnum][3]
        nimg = len(list_img)
        
#        if nimg in ind_nimg : 
    
#        print gnum
#        print indline
#        print dict_grains[gnum][indline]
        list_line = np.array(dict_grains[gnum][indline], dtype = int)
        list_col = np.array(dict_grains[gnum][indcol], dtype = int)
        list_edge = np.array(dict_grains[gnum][indedge], dtype = int)
        list_edge_restricted = np.array(dict_grains[gnum][indedge_restricted], dtype = int)
#        if mapcolor in ["rgb_x", "rgb_z"]: 
        if mapcolor[:3] == "rgb" :
            list_plot = np.array([dict_grains[gnum][indrgbxz[0]],\
                                  dict_grains[gnum][indrgbxz[1]],\
                                  dict_grains[gnum][indrgbxz[2]]], dtype = float)
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
            #print list_plot
        else :
            list_plot = np.array(dict_grains[gnum][indplot], dtype = float)
            meanvalue = round(list_plot.mean(),3)
            stdvalue = round(list_plot.std(),3)
            minvalue = round(list_plot.min(),3)
            maxvalue = round(list_plot.max(),3)
            print("mean, std, min, max")
            print(meanvalue, stdvalue, minvalue, maxvalue)           
#            print "list_plot ", list_plot

            if add_symbols :
                
                imgxyz = loadtxt(filexyz, skiprows = 1)
                img_in_filexyz = np.array(imgxyz[:,0], int)
                xy_in_filexyz = np.array(imgxyz[:,1:3], float)
                print("dxystep =", dxystep) 
            
#            print "img_in_filexyz = ", img_in_filexyz
                if mapcolor == "npeaks" :
                    level3 = 20.
                    ind_npeaks_below_level3 = where(list_plot < level3)
                    first_time = 1
                    for i in ind_npeaks_below_level3[0] :
        #                print "i = ", i
        #                print "img =", imglist[i]
                        ind0 = where(img_in_filexyz == list_img[i])
        #                print ind0[0][0]
                        toto = xy_in_filexyz[ind0[0][0],:] 
                        if first_time :
                            list_xy_npeaks_below_level3 = toto
                            first_time = 0
                        else :
                            list_xy_npeaks_below_level3 = row_stack((list_xy_npeaks_below_level3,toto))
        

                    list_xy_npeaks_below_level3 = list_xy_npeaks_below_level3 + dxystep/2.0
#    
                if mapcolor == "pixdev" :
                                
                    level2 = 0.5
                    ind_pixdev_above_level2 = where(pixdevlist2 > level2)
        #            print "ind_pixdev_above_level2 = ", ind_pixdev_above_level2[0]
        
                    first_time = 1
                    for i in ind_pixdev_above_level2[0] :
        #                print "i = ", i
        #                print "img =", imglist[i]
                        ind0 = where(img_in_filexyz == list_img[i])
        #                print ind0[0][0]
                        toto = xy_in_filexyz[ind0[0][0],:] 
                        if first_time :
                            list_xy_pixdev_above_level2 = toto
                            first_time = 0
                        else :
                            list_xy_pixdev_above_level2 = row_stack((list_xy_pixdev_above_level2,toto))
         
                    list_xy_pixdev_above_level2 = list_xy_pixdev_above_level2 + dxystep/2.0
                         

        if subtract_mean == "yes" :
            list_plot = list_plot - meanvalue
            #print "list_plot - mean ", list_plot
        
        if filter_on_pixdev_and_npeaks :
            list_pixdev = np.array(dict_grains[gnum][ind_pixdev], dtype = float)
            list_npeaks = np.array(dict_grains[gnum][ind_npeaks], dtype = float)
            
        if filter_on_intensity :
            list_intensity = np.array(dict_grains[gnum][ind_intensity], dtype = float)
            min_intensity = max(list_intensity) * min_fraction_of_max_intensity_for_filter
        
        if impos_start[1]==0 : list_x = list_col*dxystep[0]
        else : list_x = (impos_start[1]-list_col)*dxystep[0]
        if impos_start[0]==0 : list_y = list_line*dxystep[1]
        else : list_y = (impos_start[0]-list_line)*dxystep[1] 
        
        #print dict_values_names[5]
        #print dict_grains[gnum][5]
        #pour rajouter l'info d'orientation
        #rgb1[0,0,:]= dict_grains[gnum][5][:3]   # rgb x
        #rgb1[0,1,:]= dict_grains[gnum][5][3:]   # rgb z
        
        print("gnum = ", gnum)
#        print dict_values_names[3]
#        print dict_grains[gnum][3]
#        print dict_values_names[4]
#        print dict_grains[gnum][4]
#        
#        #list_edge_for_plot = zeros((nimg,3), dtype = int)
#        list_line = zeros(nimg, dtype=int)
#        list_col = zeros(nimg, dtype=int)

        if single_gnumloc  is  not None : 
            gnumloc_list = dict_grains[gnum][indgnumloc]
            gnumloc_min1 = min(gnumloc_list)
            if gnumloc_min1 != single_gnumloc : 
                print("skip grain : gnumloc_min1 , single_gnumloc : ", gnumloc_min1, single_gnumloc)
                continue

        if number_of_graphs  is None : rgb1 =  color_for_missing_data * ones(shape_rgb, dtype = float)
        
        for i in range(nimg):
            img = list_img[i]
            gnumloc = dict_grains[gnum][indgnumloc][i]  
            
            if single_gnumloc  is  not None :
                if gnumloc != single_gnumloc : continue
            
            ind1 = where(list_img == img)
            
            if filter_on_pixdev_and_npeaks :
                cond1 = (list_pixdev[i]>maxpixdev)|(list_npeaks[i]< minnpeaks)
            else : cond1 = False
            if filter_on_intensity :
                cond2 = list_intensity[i] < min_intensity
            else : cond2 = False     
            
            if number_of_graphs  is  not None :
                if cond1 | cond2 : continue
      
            if maptype == 0 :
                if list_plot[i] > max_forplot : rgb1[list_line[i],list_col[i],:] = [1.,1.,1.] # [1.0,0.0,0.0]  # red
                elif list_plot[i] < min_forplot : rgb1[list_line[i],list_col[i],:] = [0.,0.,0.]  #black
                else :
                    rgb1[list_line[i],list_col[i],:]= ((list_plot[i] - min_forplot)/(max_forplot-min_forplot))                
            elif maptype == 1 :
                if list_plot[i] > max_forplot : rgb1[list_line[i],list_col[i],:] = [0.,0.,0.] # [1.0,0.0,0.0]  # red
                elif list_plot[i] < min_forplot : rgb1[list_line[i],list_col[i],:] = [1.0,1.0,1.0]  #white
                else :
                    rgb1[list_line[i],list_col[i],:]= ((max_forplot - list_plot[i])/(max_forplot-min_forplot))                

            elif maptype == 2 :
                if list_plot[i] > max_forplot :  rgb1[list_line[i],list_col[i],:] = [1.,0.,0.]
                elif list_plot[i] < min_forplot : rgb1[list_line[i],list_col[i],:] = [0.,0.,1.]
                else :
                    toto = ((list_plot[i] - min_forplot)/(max_forplot-min_forplot))            
                    rgb1[list_line[i],list_col[i],:]= np.array(cmap(toto))[:3]
            elif maptype == 3 :
                 rgb1[list_line[i],list_col[i],:] = list_plot[i,:]*1.0
            elif maptype == 4 :
                if single_gnumloc  is  not None :
                    if gnumloc == single_gnumloc : rgb1[list_line[i],list_col[i],:] = rgb_rand
                else :
                    # marque en jaune les doublons 
                    # = deux gnumloc pour un meme couple (img, gnum)
                    if shape(ind1)[1] == 1 : rgb1[list_line[i],list_col[i],:] =  dict_rgb_gnumloc[gnumloc]
                    else :  rgb1[list_line[i],list_col[i],:] = [1.,1.,0.0]  # jaune        
                    
            if number_of_graphs  is None :        
                if cond1 | cond2 :
#                if (list_pixdev[i]>maxpixdev)|(list_npeaks[i]< minnpeaks) :
                    rgb1[list_line[i],list_col[i],:] = color_for_filtered_data  

        if 1 : # all grains - all maps but last
            if number_of_graphs_per_figure > 1 :                
                if not(kk%number_of_graphs_per_figure) :    
                    if kk > 0 :
                        kkstart = kk-number_of_graphs_per_figure
                        kkend = kk-1
                        figfilename = filepathout + map_prefix + str(k) +"_g_"+ str(kkstart)+"_to_"+str(kkend)+".png"
                        print(figfilename)
                        if savefig_map : p.savefig(figfilename, bbox_inches='tight')            
                    p.figure(figsize=(10,10))
                    k = k+1
                    
                gnum1to9 = 1+ kk%number_of_graphs_per_figure
#                print "gnum1to9 = ", gnum1to9
                gnum1to9_new = gnum1to9 * 1
                if abs(PAR.map_rotation)> 0.1 : # TODO : calcul correct
                    new_order = array([7,4,1,8,5,2,9,6,3])
                    gnum1to9_new = new_order[gnum1to9-1]
                
                ax = p.subplot(dict_graph[number_of_graphs_per_figure][0],dict_graph[number_of_graphs_per_figure][1],gnum1to9_new)
                p.subplots_adjust(hspace=0.01,wspace=0.01) 
                
            imrgb= p.imshow(rgb1, interpolation='nearest', extent=extent)
            
            if number_of_graphs_per_figure > 1 :            
                if gnum1to9_new != dict_graph[number_of_graphs_per_figure][2] :
                    ax.axes.get_xaxis().set_ticks([])
                    ax.axes.get_yaxis().set_ticks([])

            # trace des frontieres 
            
            for i in range(nimg):
                gnumloc = dict_grains[gnum][indgnumloc][i]            
                if single_gnumloc  is  not None :
                    if gnumloc != single_gnumloc : continue
                
                #print list_col[i], list_line[i], list_x[i], list_y[i], list_edge[i] 
                xcen1 = list_x[i]+dxy_corner_to_center_microns[0]
                ycen1 = list_y[i]+dxy_corner_to_center_microns[1]
#                p.text(xcen1, ycen1, \
#                str(list_edge[i]),fontsize = 10, ha = 'center', va = 'center' )

                if single_gnumloc  is None  :
                    # extended frontiers
                    list_edge_lines = dict_edge_lines[list_edge[i]]
        #                print list_edge_lines
        #                print list_edge_lines[0]
        #                print list_edge_lines[0][0]
        #                print list_edge_lines[0][0][0]
                    num_segments = shape(list_edge_lines)[0]
    
                    for j in range(num_segments):                    
                        first_point_x = xcen1 + (list_edge_lines[j][0][0]-0.5)*dxystep_abs[0]
                        first_point_y = ycen1 + (list_edge_lines[j][0][1]-0.5)*dxystep_abs[1]
                        last_point_x = xcen1 + (list_edge_lines[j][1][0]-0.5)*dxystep_abs[0]
                        last_point_y =  ycen1 + (list_edge_lines[j][1][1]-0.5)*dxystep_abs[1]
                        listx.append(first_point_x)
                        listx.append(last_point_x)
                        listy.append(first_point_y)
                        listy.append(last_point_y)
                        xx = [first_point_x, last_point_x] 
                        yy = [first_point_y, last_point_y] 
                        #strcolor = rgb_rand
                        strcolor = "k"
                        if maptype == 0 : strcolor = 'r'
                        p.plot(xx,yy,color = strcolor , linestyle = '-') 
                    
                # restricted frontiers for gnumloc = gnumloc_min  
                if ((zoom == "yes")&(number_of_graphs  is None))|(number_of_graphs==1) :                  
                    list_edge_lines = dict_edge_lines[list_edge_restricted[i]]
        #                print list_edge_lines
        #                print list_edge_lines[0]
        #                print list_edge_lines[0][0]
        #                print list_edge_lines[0][0][0]
                    num_segments = shape(list_edge_lines)[0]
    
                    for j in range(num_segments):                    
                        first_point_x = xcen1 + (list_edge_lines[j][0][0]-0.5)*dxystep_abs[0]
                        first_point_y = ycen1 + (list_edge_lines[j][0][1]-0.5)*dxystep_abs[1]
                        last_point_x = xcen1 + (list_edge_lines[j][1][0]-0.5)*dxystep_abs[0]
                        last_point_y =  ycen1 + (list_edge_lines[j][1][1]-0.5)*dxystep_abs[1]
                        listx.append(first_point_x)
                        listx.append(last_point_x)
                        listy.append(first_point_y)
                        listy.append(last_point_y)
                        xx = [first_point_x, last_point_x] 
                        yy = [first_point_y, last_point_y] 
                        #strcolor = rgb_rand
#                        #if single_gnumloc  is  not None : strcolor = 'k'
#                        if single_gnumloc  is  not None : strcolor = 'r'
#                        else :
                        strcolor = array([1.0,1.0,0.0])
                        p.plot(xx,yy,color = strcolor , linestyle = '-')                    
            
            if add_symbols :
                if mapcolor == "npeaks" :
                    xysymb = list_xy_npeaks_below_level3
                    if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "wx", mew = 2) 
                    else : p.plot(xysymb[:,0], xysymb[:,1], "wx", mew = 2)      # "_"          
                if mapcolor == "pixdev" :  
                    xysymb = list_xy_pixdev_above_level2
                    p.plot(xysymb[:,0], xysymb[:,1], 'w+', mew = 2)                
 
                       
            if number_of_graphs_per_figure == 1 :
                xmean1 = mean(listx)
                ymean1 = mean(listy)
                ax = p.subplot(111)
                if show_grain_number :
                    p.text(xmean1, ymean1, str(gnum), ha = "center", va="center", color = colortext)#, rotation = "vertical")
                
                if remove_ticklabels_titles == 0 :                    
                    p.title(mapcolor)
                else :
                    ax.axes.get_xaxis().set_ticks([])
                    ax.axes.get_yaxis().set_ticks([])
                    p.setp(ax,xticklabels = [])
                    p.setp(ax,yticklabels = [])
                    
            if number_of_graphs_per_figure > 1:              
                if zoom =="yes":   
                    #print listx, listy
                    xmin1 = min(listx)-dxystep_abs[0]
                    xmax1 = max(listx)+dxystep_abs[0]
                    ymin1 = min(listy)-dxystep_abs[1]
                    ymax1 = max(listy)+dxystep_abs[1]
                    #xceng = (xmin1+xmax1)/2.0
                    #yceng = (ymin1+ymax1)/2.0
                    p.text(xmin1+dxystep_abs[0]/2.0,ymin1+dxystep_abs[1]/2.0,str(gnum),
                           ha = "center", va="center", color = colortext)
                    p.xlim(xmin1,xmax1)
                    p.ylim(ymin1,ymax1)
                if zoom == "no":
                    if xylim  is None :
                        # p.text(2.5,-2.5,str(gnum))  # UO2
                        #p.text(-50,-50,str(gnum), ha = "center", va="center") # CdTe
                        p.text(xmin + gnum_xypos[0], ymin +gnum_xypos[1],str(gnum), 
                               ha = "center", va="center", rotation = PAR.map_rotation,
                               color = colortext) # CdTe
                    elif xylim  is  not None :
                        p.xlim(xylim[0], xylim[1])
                        p.ylim(xylim[2], xylim[3])
                        p.text(xylim[0] + gnum_xypos[0], xylim[2] +gnum_xypos[1],str(gnum), 
                               ha = "center", va="center", rotation = PAR.map_rotation,
                               color = colortext)                        
             
                if gnum1to9_new == dict_graph[number_of_graphs_per_figure][2] :
                    if remove_ticklabels_titles == 0 :
                        p.xlabel("dxech (microns)")
                        p.ylabel("dyech (microns)")
                    else :
                        p.subplots_adjust(wspace = 0.05,hspace = 0.05)
                        p.setp(ax,xticklabels = [])
                        p.setp(ax,yticklabels = [])
                    if zoom == "no" :
                        if xylim  is None :
                            p.xlim(xmin,xmax)
                            p.ylim(ymin,ymax)
                        elif xylim  is  not None :
                            p.xlim(xylim[0], xylim[1])
                            p.ylim(xylim[2], xylim[3])                        
                    ax.locator_params('x', tight=True, nbins=5)
                    ax.locator_params('y', tight=True, nbins=5)
            else :
                if remove_ticklabels_titles == 0 :
                    p.xlabel("dxech (microns)")
                    p.ylabel("dyech (microns)")                   
                p.xlim(xmin,xmax)
                p.ylim(ymin,ymax)  
                if xylim  is  not None :
                    p.xlim(xylim[0], xylim[1])
                    p.ylim(xylim[2], xylim[3])                    
                
        kk = kk+1                
# fin du if nimg in ind_nimg
            
    if number_of_graphs_per_figure > 1:            
#    if 1 : # all grains - last map
        kkstart = kk-1-int(kk/number_of_graphs_per_figure)
        kkend = kk-1
        figfilename = filepathout + map_prefix + str(k) +"_g_"+ str(kkstart)+"_to_"+str(kkend)+".png"
        print(figfilename)
        if savefig_map : p.savefig(figfilename, bbox_inches='tight')
            
    if 0 : # test with only one grain
        fig = p.plt.figure(frameon=True, figsize = (8,8))
        imrgb= p.imshow(rgb1, interpolation='nearest', extent=extent)
        for i in range(nimg):
            print(list_col[i], list_line[i], list_x[i], list_y[i], list_edge[i]) 
            xcen1 = list_x[i]+dxy_corner_to_center_microns[0]
            ycen1 = list_y[i]+dxy_corner_to_center_microns[1]
            p.text(xcen1, ycen1, \
            str(list_edge[i]),fontsize = 10, ha = 'center', va = 'center' )
            list_edge_lines = dict_edge_lines[list_edge[i]]
#                print list_edge_lines
#                print list_edge_lines[0]
#                print list_edge_lines[0][0]
#                print list_edge_lines[0][0][0]
            num_segments = shape(list_edge_lines)[0]
            for j in range(num_segments):
                first_point_x = xcen1 + (list_edge_lines[j][0][0]-0.5)*dxystep_abs[0]
                first_point_y = ycen1 + (list_edge_lines[j][0][1]-0.5)*dxystep_abs[1]
                last_point_x = xcen1 + (list_edge_lines[j][1][0]-0.5)*dxystep_abs[0]
                last_point_y =  ycen1 + (list_edge_lines[j][1][1]-0.5)*dxystep_abs[1]
                xx = [first_point_x, last_point_x] 
                yy = [first_point_y, last_point_y] 
                p.plot(xx,yy,'k-')

        p.xlabel("dxech (microns)")
        p.ylabel("dyech (microns)")
        p.text(2.5,-2.5,str(gnum))

    return(map_imgnum, np.array(listx), np.array(listy))


    
def xycam_to_uflab(xycam,
                   calib, 
                   pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]):

    # modif 04 Mar 2010 xbet xgam en degres au lieu de radians

    # XMAS PCIF6 changer le signe de xgam  
    # laptop OR garder le meme signe pour xgam

    detect = calib[0]*1.0
    xcen = calib[1]*1.0
    ycen = calib[2]*1.0
    xbet = calib[3]*1.0
    xgam = calib[4]*1.0
    
##    print "Correcting the data according to the parameters"
##    print "xcam, ycam in XMAS convention"
##    
##    print "detect in mm" , detect
##    print "xcen in pixels" , xcen
##    print "ycen in pixels" , ycen
##    print "xbet in degrees" , xbet
##    print "xgam in degrees" , xgam

    xbetrad = xbet * PI/180.0
    xgamrad = xgam * PI/180.0

    cosbeta=np.cos(PI/2.-xbetrad)
    sinbeta=np.sin(PI/2.-xbetrad)
    cosgam=np.cos(-xgamrad)
    singam=np.sin(-xgamrad)

    xcam1=(xycam[0]-xcen)*pixelsize
    
    ycam1=(xycam[1]-ycen)*pixelsize
    xca0=cosgam*xcam1-singam*ycam1
    yca0=singam*xcam1+cosgam*ycam1    

    IOlab = detect * array([0.0, cosbeta, sinbeta])
    OMlab = array([xca0, yca0*sinbeta, -yca0*cosbeta])
    IMlab = IOlab + OMlab
    uflab = IMlab/norme(IMlab)
    
    #uflab[0]=-1.0*uflab[0]

    uflabyz = array([0.0, uflab[1],uflab[2]])    
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab
    
    chi2 = (180.0/PI)*np.arctan(uflab[0]/norme(uflabyz))
    twicetheta2 = (180.0/PI)*np.arccos(uflab[1])
    
##    chi3 = (180.0/PI)*arccos(inner(uflab,uflabyz)/norme(uflabyz))*sign(uflab[0])

    #print "uflab =", uflab
    #print "2theta, theta, chi en deg", twicetheta2 ,  twicetheta2/2.0, chi2
##    print "chi3 = ", chi3
    
    return(uflab)
    
def xycam_to_uqlab(xycam, 
                   calib, 
                   pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]):

    # modif 04 Mar 2010
    uqlab = zeros(3,float)
    uflab = zeros(3,float)
    
    uilab = array([0.0,1.0,0.0])
    uflab = xycam_to_uflab(xycam,
                           calib, 
                           pixelsize = pixelsize)

    # print uflab
    
#    twotheta = arccos(uflab[1])

    #print "xycam, twotheta = ", xycam, "  ",  twotheta * 180/PI
    
#    uqlab= (uflab-uilab)/(2.0*sin(twotheta/2.0))
    
    uqlab= (uflab-uilab)/(2.0*np.sqrt( (1.0-inner(uflab,uilab))/2.0 ))

    return(uqlab)
    
    
def two_spots_to_mat_gen(hkl,
                     xycam,
                     calib, 
                     pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1],
                    elem_label = "Ge",
                    verbose = 1):

    # modif 08Dec14
    # extension du calcul a une maille quelconque 
    
    # ajout calcul difference d'angle exp-theor
    
    uqcr = zeros((2,3), float)
    uqlab = zeros((2,3), float)
    
    if verbose :
        print("pixelsize = ", pixelsize)
#    jklsdf

        print("calculate orientation matrix from two reflections")
        print("hkl = \n", hkl)
        print("xy = \n", xycam)
    
    hkl = np.array(hkl, dtype = float)
    xycam = np.array(xycam, dtype = float)
    
    uqlab[0,:] = xycam_to_uqlab(xycam[0,:],calib, pixelsize = pixelsize)
    uqlab[1,:] = xycam_to_uqlab(xycam[1,:],calib, pixelsize = pixelsize)
    
    qlab3 = cross(uqlab[0,:],uqlab[1,:])
    uqlab3 = qlab3 / norme(qlab3)

    uqlab2b = cross(uqlab3,uqlab[0,:])
    
    mat123_exp_OND_lab = np.column_stack((uqlab[0,:], uqlab2b, uqlab3))

    #print "normes :", norme(uqlab[0,:]), norme(uqlab2), norme(uqcr[0,:]), norme(uqcr2)
    #print norme(uqcr3), norme(uqlab3), norme(uqlab2b)

    #print "inner products :", inner(uqlab[0,:], uqlab2b), inner(uqlab2b, uqlab3), inner(uqlab3,uqlab[0,:])
    #print inner(uqcr[0,:],uqcr2b), inner (uqcr2b,uqcr3), inner(uqcr3, uqcr[0,:])

    dlat0_deg = np.array(DictLT.dict_Materials[elem_label][1], dtype = float)
    dlat0_rad = deg_to_rad(dlat0_deg)

    #print dlat0.round(decimals = 4)
    #print dlat.round(decimals = 4)
    
    # matstarlab construite pour avoir norme(astar) = 1 
    Bstar0 = dlat_to_Bstar(dlat0_rad)
    
    if verbose : print("Bstar0 = ", Bstar0.round(decimals = 4))

    uq1_cart_star = dot(Bstar0,hkl[0,:])
    uq1_cart_star = uq1_cart_star / norme(uq1_cart_star)
    
    uq2_cart_star = dot(Bstar0,hkl[1,:])
    uq2_cart_star = uq2_cart_star / norme(uq2_cart_star)
    
    uq3_cart_star = cross(uq1_cart_star,uq2_cart_star)
    uq3_cart_star = uq3_cart_star / norme(uq3_cart_star)
    
    uq2b_cart_star = cross(uq3_cart_star,uq1_cart_star)    

    mat123_theor_OND_star = np.column_stack((uq1_cart_star, uq2b_cart_star, uq3_cart_star))
        
    invMtheor = inv(mat123_theor_OND_star)
    
    matUstar = dot(mat123_exp_OND_lab,invMtheor)
    
    matstarlab3x3 = dot(matUstar,Bstar0)
    
    matstarlab3x3 =  matstarlab3x3 / norme(matstarlab3x3[:,0])
    
    matstarlab = GT.mat3x3_to_matline(matstarlab3x3)
        
#    uilab = array([0.0,1.0,0.0]) 

    if verbose :  
        print("calib") 
        print_calib(calib)
    
#    for i in range(2) :
#        uqcr[i,:]= hkl[i,:]/norme(hkl[i,:])
#        uflab = xycam_to_uflab(xycam[i,:],
#                               calib,
#                               pixelsize = pixelsize)
#
#        #print "uflab =", uflab
#        #thlab = arccos(inner(uflab,uilab))*90.0/PI
#        #print "th from uflab, uilab in deg", thlab
#
#        uqlab[i,:]= (uflab-uilab)/(2.0*np.sqrt( (1.0-inner(uflab,uilab))/2.0 ))


    if verbose :
        print("uq1_cart_star = \n" , uq1_cart_star.round(decimals = 4))
        print("uq2_cart_star = \n" , uq2_cart_star.round(decimals = 4))
        print("uqlab = \n", uqlab.round(decimals = 4))
        
        print("inner(uqlab[0,:],uqlab[1,:]) = ", inner(uqlab[0,:],uqlab[1,:]))
        print("inner(uq1_cart_star,uq2_cart_star) = ", inner(uq1_cart_star,uq2_cart_star))
    
    alfexp=np.arccos(inner(uqlab[0,:],uqlab[1,:]))
    alftheor=np.arccos(inner(uq1_cart_star,uq2_cart_star))
    delta_alf = (alfexp-alftheor)*1000.0
    
    if verbose :
        print("alftheor = ", alftheor)
        print("alftheor (deg) = ", alftheor*180./PI)
        print("delta_alf mrad = ", round(delta_alf,2))
    
    return(matstarlab, delta_alf)    
        
def two_spots_to_mat(hkl,
                     xycam,
                     calib, 
                     pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1],
                        verbose = 1):

    # modif 04 Mar 2010

    # reseau cubique uniquement
    
    # ajout calcul difference d'angle exp-theor
    uqcr = zeros((2,3), float)
    uqlab = zeros((2,3), float)
    
    if verbose :
        print("pixelsize = ", pixelsize)
    #    jklsdf
    
        print("calculate orientation matrix from two reflections")
        print("hkl = \n", hkl)
        print("xy = \n", xycam)
    
    hkl = np.array(hkl, dtype = float)
    xycam = np.array(xycam, dtype = float)
    
    uqlab[0,:] = xycam_to_uqlab(xycam[0,:],calib, pixelsize = pixelsize)
    uqlab[1,:] = xycam_to_uqlab(xycam[1,:],calib, pixelsize = pixelsize)

    uqcr[0,:] = hkl[0,:]/norme(hkl[0,:])
    uqcr[1,:] = hkl[1,:]/norme(hkl[1,:])

    qcr3 = cross(uqcr[0,:],uqcr[1,:])
    uqcr3 = qcr3 / norme(qcr3)
    
    qlab3 = cross(uqlab[0,:],uqlab[1,:])
    uqlab3 = qlab3 / norme(qlab3)

    uqlab2b = cross(uqlab3,uqlab[0,:])

    uqcr2b = cross(uqcr3,uqcr[0,:])

    #print "normes :", norme(uqlab[0,:]), norme(uqlab2), norme(uqcr[0,:]), norme(uqcr2)
    #print norme(uqcr3), norme(uqlab3), norme(uqlab2b)

    #print "inner products :", inner(uqlab[0,:], uqlab2b), inner(uqlab2b, uqlab3), inner(uqlab3,uqlab[0,:])
    #print inner(uqcr[0,:],uqcr2b), inner (uqcr2b,uqcr3), inner(uqcr3, uqcr[0,:])

    crtoRq= vstack((uqcr[0,:], uqcr2b, uqcr3))                        
    #print "crtoRq = \n", crtoRq
    Rqtocr = np.linalg.inv(crtoRq)
    #print "Rqtocr = \n", Rqtocr
    
    astarlab = Rqtocr[0,0]*uqlab[0,:] + Rqtocr[0,1]*uqlab2b + Rqtocr[0,2]*uqlab3
    bstarlab = Rqtocr[1,0]*uqlab[0,:] + Rqtocr[1,1]*uqlab2b + Rqtocr[1,2]*uqlab3
    cstarlab = Rqtocr[2,0]*uqlab[0,:] + Rqtocr[2,1]*uqlab2b + Rqtocr[2,2]*uqlab3

    matstarlab = hstack((astarlab,bstarlab,cstarlab))/norme(astarlab)

    if verbose :
        print("matstarlab = \n", matstarlab)
            
    #    uilab = array([0.0,1.0,0.0])   
        print("calib") 
        print_calib(calib, pixelsize = pixelsize)
    
#    for i in range(2) :
#        uqcr[i,:]= hkl[i,:]/norme(hkl[i,:])
#        uflab = xycam_to_uflab(xycam[i,:],
#                               calib,
#                               pixelsize = pixelsize)
#
#        #print "uflab =", uflab
#        #thlab = arccos(inner(uflab,uilab))*90.0/PI
#        #print "th from uflab, uilab in deg", thlab
#
#        uqlab[i,:]= (uflab-uilab)/(2.0*np.sqrt( (1.0-inner(uflab,uilab))/2.0 ))


        print("uqcr = \n" , uqcr.round(decimals = 4))
        print("uqlab = \n", uqlab.round(decimals = 4))
        
        print("inner(uqlab[0,:],uqlab[1,:]) = ", inner(uqlab[0,:],uqlab[1,:]))
        print("inner(uqcr[0,:],uqcr[1,:]) = ", inner(uqcr[0,:],uqcr[1,:]))
        
    alfexp=np.arccos(inner(uqlab[0,:],uqlab[1,:]))
    alftheor=np.arccos(inner(uqcr[0,:],uqcr[1,:]))
    delta_alf = (alfexp-alftheor)*1000.0   
    
    if verbose :
        print("alftheor = ", alftheor)
        print("alftheor (deg) = ", alftheor*180./PI)
        print("delta_alf * 1000 = ", round(delta_alf,2))
    
    return(matstarlab, delta_alf)

def twomat_to_dyz_ux_dxz_uy_dxy_uz_sample(matstarlabref, 
                                          matstarlab,
                                          verbose = 0,
                                          omega = None, # was PAR.omega_sample_frame,
                                          mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame) :
                                              
    # from calcdef6b preparation for plot_hist_orient                                          
    
    mat1_ref = matstarlab_to_matstarsample3x3(matstarlabref, 
                                              omega = omega,
                                              mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)
    inv_mat1_ref = inv(mat1_ref)
    mat1 = matstarlab_to_matstarsample3x3(matstarlab, 
                                          omega = omega,
                                          mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)
    dmat = np.dot(mat1,inv_mat1_ref)
    ux = dmat[:,0]/norme(dmat[:,0])
    uy = dmat[:,1]/norme(dmat[:,1])
    uz = dmat[:,2]/norme(dmat[:,2])
    if verbose :
        print("ux =", ux)
        print("uy =", uy)
        print("uz =", uz)
            
    dyz_ux_dxz_uy_dxy_uz_sample = zeros(6,float)
            
    dyz_ux_dxz_uy_dxy_uz_sample[0:2]= ux[1:3]*1000.0
    dyz_ux_dxz_uy_dxy_uz_sample[2] = uy[0]*1000.0
    dyz_ux_dxz_uy_dxy_uz_sample[3] = uy[2]*1000.0
    dyz_ux_dxz_uy_dxy_uz_sample[4:6]= uz[0:2]*1000.0    
    
    return(dyz_ux_dxz_uy_dxy_uz_sample)

def twomat_to_RxRyRz_sample_large_strain(matstarlabref, 
                                          matstarlab,
                                          verbose = 0,
                                          omega = None, # was PAR.omega_sample_frame
                                          mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame):
                                                
    # from twomat_to_dyz_ux_dxz_uy_dxy_uz_sample                                        
    # for Leineweber / Fonovic nitride sample 
    # calculate both rotation + shear
    
    # NB : Rz = 40 mrad add 0.5 mrad uncertainty on Rx and Ry
    # due to working on vectors not successive rotations
    
    matdirlabref = GT.mat3x3_to_matline(matstarlab_to_matdirlab3x3(matstarlabref)[0])
    matdirlab = GT.mat3x3_to_matline(matstarlab_to_matdirlab3x3(matstarlab)[0])     
        
    mat1_ref = matstarlab_to_matstarsample3x3(matdirlabref, 
                                              omega = omega,
                                              mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)
    inv_mat1_ref = inv(mat1_ref)
    mat1 = matstarlab_to_matstarsample3x3(matdirlab, 
                                          omega = omega,
                                          mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)
    dmat = np.dot(mat1,inv_mat1_ref)
    ux = dmat[:,0] #/norme(dmat[:,0])
    uy = dmat[:,1]#/norme(dmat[:,1])
    uz = dmat[:,2]#/norme(dmat[:,2])
    if verbose :
        print("ux =", ux)
        print("uy =", uy)
        print("uz =", uz)
    
    if 0 :        
        dyz_ux_dxz_uy_dxy_uz_sample = zeros(6,float)
                
        dyz_ux_dxz_uy_dxy_uz_sample[0:2]= ux[1:3]*1000.0
        dyz_ux_dxz_uy_dxy_uz_sample[2] = uy[0]*1000.0
        dyz_ux_dxz_uy_dxy_uz_sample[3] = uy[2]*1000.0
        dyz_ux_dxz_uy_dxy_uz_sample[4:6]= uz[0:2]*1000.0  
    
#    dyz_ux_dxz_uy_dxy_uz_sample = twomat_to_dyz_ux_dxz_uy_dxy_uz_sample(matstarlabref, 
#                                          matstarlab,
#                                          verbose = verbose)

    if 0 :    
        dy_ux = dyz_ux_dxz_uy_dxy_uz_sample[0]
        dz_ux = dyz_ux_dxz_uy_dxy_uz_sample[1]
        dx_uy = dyz_ux_dxz_uy_dxy_uz_sample[2]
        dz_uy = dyz_ux_dxz_uy_dxy_uz_sample[3]
        dx_uz = dyz_ux_dxz_uy_dxy_uz_sample[4]
        dy_uz = dyz_ux_dxz_uy_dxy_uz_sample[5]
        
    if 1 : # un peu plus precis : correction 0.02 mrad pour rotation 35 mrad
            
        dx_ux = (norme(ux)-1.)*1000.
        dy_uy = (norme(uy)-1.)*1000.
        dz_uz = (norme(uz)-1.)*1000.
        
        trace1 = (dx_ux + dy_uy + dz_uz)
        
        dx_ux = dx_ux - trace1/3.
        dy_uy = dy_uy - trace1/3.
        dz_uz = dz_uz - trace1/3.
        
        dy_ux = atan(ux[1]/ux[0])*1000.
        dz_ux = atan(ux[2]/ux[0])*1000.
        dx_uy = atan(uy[0]/uy[1])*1000.
        dz_uy = atan(uy[2]/uy[1])*1000.
        dx_uz = atan(uz[0]/uz[2])*1000.
        dy_uz = atan(uz[1]/uz[2])*1000.   

    Rz = (dy_ux - dx_uy)/2.
#    dRz = dy_ux + dx_uy
    dRz = -(acos(inner(ux,uy))-PI/2.)*1000./2.

    Ry = (dx_uz - dz_ux)/2.
#    dRy = dx_uz + dz_ux
    dRy = -(acos(inner(uz,ux))-PI/2.)*1000./2.

    Rx = (dz_uy - dy_uz)/2.
#    dRx = dz_uy + dy_uz
    dRx = -(acos(inner(uy,uz))-PI/2.)*1000./2.

    RxRyRz_mrad = np.array([Rx, Ry, Rz])
    dRxRyRz_mrad = np.array([dRx, dRy, dRz]) 
    dLxLyLz_mrad = np.array([dx_ux, dy_uy, dz_uz])
    ang1_mrad = norme(RxRyRz_mrad)       
    dang1_mrad = norme(dRxRyRz_mrad)

    if verbose :
        print("RxRyRz_mrad = ", RxRyRz_mrad.round(decimals=2))
        print("ang1_mrad = ", round(ang1_mrad,2))
        print("dRxRyRz_mrad = ", dRxRyRz_mrad.round(decimals=2))
        print("dang1_mrad = ", round(dang1_mrad,2))
        print("dLxLyLz_mrad = ", dLxLyLz_mrad.round(decimals=2))

    
    return(RxRyRz_mrad, dRxRyRz_mrad, ang1_mrad, dang1_mrad, dLxLyLz_mrad) 

def twomat_to_rotation_Emeric(matstarlab1,matstarlab2):
    
    # utilise matstarlab
    # pas OK pour grandes deformations : sensible aux permutations de abc

    #version Emeric nov 13
    matref=matstarlab_to_matdirONDsample3x3(matstarlab1)
    matmes=matstarlab_to_matdirONDsample3x3(matstarlab2)
    
  
   #ATTENTION : Orthomormalisation avant de faire le calcul 
    #matmisor = dot(np.linalg.inv(matref.transpose()),matmes.transpose())
    matmisor = dot(matref,matmes.transpose())   # cf cas CK
    
    toto = (matmisor[0,0]+matmisor[1,1]+matmisor[2,2]-1.)/2.
    # 2 + 2* toto = 2 + trace - 1 =  1 + trace
    
    #theta en rad
    theta=np.arccos(toto)
    
    #Cas pathologique de theta=0 => vecteur == 1 0 0
    #a completer
    
    #Sinon
    
    toto1 = 2.*(1.+ toto)
    rx=(matmisor[1,2]-matmisor[2,1])/toto1
    ry=(matmisor[2,0]-matmisor[0,2])/toto1
    rz=(matmisor[0,1]-matmisor[1,0])/toto1

    vecRodrigues_sample=np.array([rx,ry,rz])   # axe de rotation en coordonnees sample
           
    theta=theta*180.0/PI
    
    return(vecRodrigues_sample, theta)
    
def twomat_to_rotation(mat1,mat2,verbose = 1):

    # from 3x3 matrixes
    # to get vector in crystal coordinates
    # no need to say which crystal because HKL of rotation axis is the same in the two crystal
    # (only axis that stays fixed during rotation)
    # abc2 en colonnes sur abc1
    
#    print "MG.twomat_to_rotation line 10218"
#    print "mat1 = ", mat1
#    print "mat2 = ", mat2
    
    mat12 = dot(np.linalg.inv(mat1),mat2)  # version OR
    quat12 = fromMatrix_toQuat(mat12)
    unitvec, angle = fromQuat_to_vecangle(quat12)
    ind1 = argmax(abs(unitvec))
    vec_crystal = unitvec/unitvec[ind1]
    
    angle1 = angle*180.0/PI
    
    mat12 = dot(mat1, np.linalg.inv(mat2))  # version CK
    quat12 = fromMatrix_toQuat(mat12)
    unitvec, angle = fromQuat_to_vecangle(quat12)
    ind1 = argmax(abs(unitvec))
    vec_lab = unitvec/unitvec[ind1]
    
    if verbose :
        print("rotation")
        print("vector (crystal)   vector (lab)   angle (deg)")
        print(vec_crystal.round(decimals = 6), vec_lab.round(decimals = 6), "\t", round(angle1,5)) #,round(angle2,3)
        
    return(vec_crystal/norme(vec_crystal), vec_lab/norme(vec_lab), angle1)
    
def uflab_to_2thetachi(uflab):

    #23May11 : go to JSM convention for chi
    
    uflabyz = array([0.0, uflab[1],uflab[2]])    
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab
    #chi2 = (180.0/PI)*arctan(uflab[0]/norme(uflabyz))

    # JSM convention : angle dans le plan xz entre les projections de uflab suivant x et suivant z
    # OR change sign of chi
    EPS = 1E-17
    chi2 = (180.0/PI)*np.arctan( uflab[0]/(uflab[2]+EPS)) # JSM convention
    
    twicetheta2 = (180.0/PI)*np.arccos(uflab[1])
    
##    chi3 = (180.0/PI)*arccos(inner(uflab,uflabyz)/norme(uflabyz))*sign(uflab[0])

##    print "uflab =", uflab
##    print "2theta, theta, chi en deg", twicetheta2 , chi2, twicetheta2/2.0
##    print "chi3 = ", chi3
    
    return(chi2, twicetheta2)
    
def uflab_to_xycam_gen(uflab, 
                       calib, 
                       uflab_cen, 
                       pixelsize = 0.08056640625):

    # 08Jun12 add variable uflab_cen
    # modif 04 Mar 2010 xbet xgam en degres au lieu de radians

    # XMAS PCIF6 changer le signe de xgam  
    # laptop OR garder le meme signe pour xgam

    detect = calib[0]*1.0
    xcen = calib[1]*1.0
    ycen = calib[2]*1.0
    xbet = calib[3]*1.0
    xgam = calib[4]*1.0
    
##    print "Correcting the data according to the parameters"
##    print "xcam, ycam in XMAS convention"
##    
##    print "detect in mm" , detect
##    print "xcen in pixels" , xcen
##    print "ycen in pixels" , ycen
##    print "xbet in degrees" , xbet
##    print "xgam in degrees" , xgam


    uilab = array([0.,1.,0.])

    xbetrad = xbet * PI/180.0
    xgamrad = xgam * PI/180.0
    
    cosbeta=np.cos(PI/2.-xbetrad)
    sinbeta=np.sin(PI/2.-xbetrad)
    cosgam=np.cos(-xgamrad)
    singam=np.sin(-xgamrad)

    uflab_cen2 = zeros(3,float)
    tthrad0 = acos(uflab_cen[1])
    tthrad = tthrad0 - xbetrad
    uflab_cen2[1] = np.cos(tthrad)
    uflab_cen2[0] = uflab_cen[0]/np.sin(tthrad0)*np.sin(tthrad)
    uflab_cen2[2] = uflab_cen[2]/np.sin(tthrad0)*np.sin(tthrad)
    
    #print "norme(uflab_cen2) = ", norme(uflab_cen2)
   
    #IOlab = detect * array([0.0, cosbeta, sinbeta])
    IOlab = detect * uflab_cen2
    
    #unlab = IOlab/norme(IOlab)

    #normeIMlab = detect / inner(uflab,unlab)
    
    normeIMlab = detect / inner(uflab,uflab_cen2) 

    #uflab1 = array([-uflab[0],uflab[1],uflab[2]])

    #uflab1 = uflab*1.0
    
    #IMlab = normeIMlab*uflab1
    
    IMlab = normeIMlab*uflab
    
    OMlab = IMlab - IOlab

    #print "inner(OMlab,uflab_cen2) = ", inner(OMlab,uflab_cen2)

    # jusqu'ici on definissait xlab = xcam0 (avant rotation xgam) par la perpendiculaire au plan ui, uflab_cen
    # ici on change

    uxcam0 = cross(uilab, uflab_cen2)
    uxcam0 = uxcam0 / norme(uxcam0)
    
    xca0 = inner(OMlab,uxcam0)
    
    # xca0 = OMlab[0]

    # calculer en dehors : IOlab, uflab_cen2, uxcam0, uycam0 pour eviter de repeter ces calculs
    uycam0 = cross(uflab_cen2,uxcam0)
    uycam0 = uycam0 / norme(uycam0)

    # yca0 = OMlab[1]/sinbeta

    yca0 = inner(OMlab, uycam0)

    xcam1 = cosgam*xca0 + singam*yca0
    ycam1 = -singam*xca0 + cosgam*yca0

    xcam = xcen + xcam1 / pixelsize
    ycam = ycen + ycam1 / pixelsize
   
    #uflabyz = array([0.0, uflab1[1],uflab1[2]])
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab
    
    #chi = (180.0/PI)*arctan(uflab1[0]/norme(uflabyz))
    #twicetheta = (180.0/PI)*arccos(uflab1[1])
    #th0 = twicetheta/2.0
    
    #print "2theta, theta, chi en deg", twicetheta , chi, twicetheta/2.0
    #print "xcam, ycam = ", xcam, ycam

    xycam = array([xcam, ycam])

    return(xycam)
    
def remove_harmonic(hkl,uflab):

    # modif 23May11 keep lowest harmonic

    #print "removing harmonics from theoretical peak list"
    nn = shape(uflab)[0]
    isbadpeak = zeros(nn,int)
    toluf = 0.001
    
    print("remove harmonics")
#    print "nn = ", nn
    
    for i in range(nn-1):
#        print "hkl_i = ", hkl[i]
        if (isbadpeak[i]==0):           
            duf = uflab[i+1:]-uflab[i]
#            print "duf =\n", duf.round(decimals=3)
            duf = abs(duf).sum(axis=1)
#            print "duf = ", duf.round(decimals=3)
            ind0 = where(duf < toluf)
            nrefl0 = len(ind0[0])
            if nrefl0 > 0 :
                print("i = ", i)
                print("hkl_i = ", hkl[i])  
                print("nrefl0 = ", nrefl0)
                range0 = ind0[0]+i+1
                print("range0 = ", range0)              
                print("hkl_j = \n", hkl[range0])
                print("duf = ", duf[ind0[0]].round(decimals=3))
                dnorme = (abs(hkl[range0])-abs(hkl[i])).sum(axis = 1)
                print("dnorme = ", dnorme)
    
                ind1 = where(dnorme > 0.)
                if len(ind1[0]) > 0 : 
                    print("isbadpeak[j] = 1 for indices", range0[ind1[0]])
                    isbadpeak[range0[ind1[0]]] = 1
                ind2 = where(dnorme < 0.)
                if len(ind2[0]) > 0 :
                    print("isbadpeak[i] = 1 for indice", i)
                    isbadpeak[i] = 1

    print("isbadpeak = ", isbadpeak)
    index_goodpeak = where(isbadpeak == 0)
#    print "index_goodpeak[0] =", index_goodpeak[0]
    nspots2 = len(index_goodpeak[0])
    print("nspots2 = ", nspots2)
    
    return(nspots2,isbadpeak)

def mat_and_hkl_to_Etheor_ththeor_uqlab(matstarlab = None, 
                         dlatu_angstroms_deg = DictLT.dict_Materials[PAR.elem_label_index_refine][1],
                         elem_label = None,
                         matwithlatpar_inv_nm = None,
                         hkl = np.ones(3,float),
                         uilab = np.array([0.,1.,0.])
                         ) :
    # d'abord on passe les angles en radians et les longueurs en nm
    
    if matwithlatpar_inv_nm  is  not None :
        
        mat = matwithlatpar_inv_nm * 1.
        
    elif (matstarlab  is  not None) :
        if (elem_label  is  not None) :
            dlatu_angstroms_deg = DictLT.dict_Materials[elem_label][1]
         
        dlatu_nm_rad = deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
        
#        print "dlatu_nm_rad = ", dlatu_nm_rad
        
        mat = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu_nm_rad)
    
#    print "mat with latpar, inverse nanometers", mat  
        
    qlab = hkl[0]*mat[0:3]+ hkl[1]*mat[3:6]+ hkl[2]*mat[6:]
    if norme(qlab)>1.0e-5 :
        uqlab = qlab/norme(qlab)
        sintheta = -inner(uqlab,uilab)
        if (sintheta > 0.0):
            #print "reachable reflection"
            Etheor_eV = DictLT.E_eV_fois_lambda_nm*norme(qlab)/(2*sintheta)
            ththeor_deg = (180./PI)*np.arcsin(sintheta)
        else :
            Etheor_eV = 0.
            ththeor_deg = 0.
    else :
        Etheor_eV = 0.
        ththeor_deg = 0.
        uqlab = np.zeros(3,float)
        
    return(Etheor_eV, ththeor_deg, uqlab)        
            
def spotlist_gen(Emin_keV, 
                 Emax_keV, 
                 diagr, 
                 matwithlatpar_inv_nm, 
                 cryst_struct, 
                 showall,
                 calib, 
                 CCDlabel = "MARCCD165", 
                 remove_harmonics = "yes"):
    """ Compute array with spots properties stacked in columns
    hkl 0:3, uflab 3:6, xy 6:8, th 8, Etheor 9, chi 10, tth 11 """

    # attention 07Sep13 retour aux nanometres pour mat
    
    # 08Jun12 uflab_to_xycam_gen for more general calculation  
    # modif 04 Mar 2010
    # 20Oct10 : nouveau coeff E_eV_fois_lambda_nm plus precis
    # 21Oct10 : input matrix avec parametres de maille integres
    # 23May11 : add returnmore for 2theta chi 

    # structures traitees ici : FCC, BCC, diamant (pour les extinctions)
    # maille deformee OK
    # Emin , Emax en KeV
    
    # pour tous les types de diagrammes
    # les x et y des spots sont calcules avec la convention suivante
    # l'observateur couche a la place de l'echantillon 
    # le corps parallele au faisceau incident
    # et recevant le faisceau incident par le sommet de la tete
    # voit sur la camera les spots de coordonnees x y
    # avec l'axe x vers la droite et l'axe y vers le bas du diagramme
    # et l'axe 2theta vers le haut du diagramme    
    
    pixelsize = DictLT.dict_CCD[CCDlabel][1]
    
    nmaxspots = 1000    
    limangle = 70
    cosangle = np.cos(limangle*PI/180.0)

    mat = matwithlatpar_inv_nm
    
    # Rlab z vers le haut, x vers le back, y vers l'aval
    
    if diagr == 'side':
        uflab_cen = array([-1.0, 0.0, 0.0])
    if diagr == 'top' :
        uflab_cen = array([0.0, 0.0, 1.0])
    if diagr == 'halfback':  # 2theta = 118
        # 0 -sin28 cos28
        tth = 28 * PI/180.0
        uflab_cen = array([0.0,-np.sin(tth), np.cos(tth)])

    uflab_cen = uflab_cen / norme(uflab_cen)

    uilab = array([0.0,1.0,0.0])

    uqlab_cen = uflab_cen - uilab
    uqlab_cen = uqlab_cen / norme(uqlab_cen)
    
    if (showall):
        print("calculate theoretical Laue pattern from orientation matrix")
        print("use matrix (with strain) \n", matwithlatpar_inv_nm)
        print("energy range (keV):", Emin_keV, Emax_keV)
        print("max angle between uflab and uflab_cen (deg) : ", limangle)
        print("uflab_cen =", uflab_cen)
        print("diagram : ", diagr)
        print("structure :", cryst_struct)
    
    #print "cosangle = ", cosangle 

    hkl = zeros((nmaxspots,3), int)
    uflab = zeros((nmaxspots,3), float)
    xy = zeros((nmaxspots,2), float)
    Etheor = zeros(nmaxspots, float)
    ththeor = zeros(nmaxspots, float)
    tth = zeros(nmaxspots, float)
    chi = zeros(nmaxspots, float)
    uqlab = zeros((nmaxspots,3), float)
    
    dlatapprox = 1.0/norme(mat[0:3])
    print("dlatapprox (nm) = ", dlatapprox)
    
    Emin_eV = Emin_keV*1000.
    Emax_eV = Emax_keV*1000.    
    
    latpar_nm = dlatapprox * 1.0
        
    Hmax = int(latpar_nm*2.*Emax_eV/DictLT.E_eV_fois_lambda_nm)

    if (showall):
        print("Hmax = ", Hmax)

    nspot=0
            
    for H in range(-Hmax, Hmax):
        for K in range(-Hmax, Hmax):
            if ((not (K-H)%2)|(cryst_struct=='bcc')|(cryst_struct=='no')):
                for L in range(-Hmax,Hmax):
                    if ((not (L-H)%2)|(cryst_struct=='bcc')|(cryst_struct=='no')) :
                        if ((cryst_struct=='fcc')|
                            ((cryst_struct=='dia')&((H%2)|((not H%2)&(not (H+K+L)%4 ))))|
                            ((cryst_struct=='bcc')&(not (H+K+L)%2))|
                            (cryst_struct=='no')
                            ):
#                            print "hkl =", H,K,L
                            hkl1 = np.array([H,K,L], float)
                            Etheor_eV, ththeor_deg, uqlab[nspot] = mat_and_hkl_to_Etheor_ththeor_uqlab(matwithlatpar_inv_nm =matwithlatpar_inv_nm,
                                                                                         hkl = hkl1)
                            if Etheor_eV > 0.01 : 
                                cosangle2 = inner(uqlab[nspot],uqlab_cen)
                                if cosangle2 > cosangle :
                                    Etheor[nspot] = Etheor_eV
                                    ththeor[nspot] = ththeor_deg
                                    sintheta = np.sin(ththeor_deg*PI/180.)
                                    #print "Etheor = ", Etheor[nspot]
                                    if (Etheor[nspot] > (Emin_eV))&(Etheor[nspot] < (Emax_eV)):                            
                                        uflabtheor = uilab + 2*sintheta*uqlab[nspot]
                                        chi[nspot], tth[nspot] = uflab_to_2thetachi(uflabtheor)
                                        if (diagr == "side") & (chi[nspot]> 0.0) :
                                            chi[nspot] = chi[nspot]-180.0
                                        test = inner(uflabtheor,uflab_cen)
                                        #print "hkl =", H,K,L
                                        #print "uflabtheor.uflab_cen = ",test
                                        if (test>cosangle):
                                            hkl[nspot,:] = array([H,K,L])
                                            uflab[nspot,:] = uflabtheor
                                            # top diagram use xbet xgam close to zero
                                            xy[nspot,:] = uflab_to_xycam_gen(uflab[nspot,:],\
                                                                                  calib, uflab_cen, pixelsize = pixelsize)
                                            nspot = nspot + 1

    nspots2 = nspot
    
    if remove_harmonics == "yes" :
        nspots2, isbadpeak2 = remove_harmonic(hkl[0:nspot,:],uflab[0:nspot,:])
    
        index_goodpeak = where(isbadpeak2 ==0)
        hkl2 =  hkl[index_goodpeak]
        uflab2 = uflab[index_goodpeak]   
        uqlab2 = uqlab[index_goodpeak]
        xy2 =  xy[index_goodpeak]
        Etheor2 = Etheor[index_goodpeak]
        ththeor2 = ththeor[index_goodpeak]
        chi2 = chi[index_goodpeak]
        tth2 = tth[index_goodpeak]
    else : 
        range1 = np.arange(0,nspot)
        hkl2, uflab2, xy2, nspots2, Etheor2, ththeor2, chi2, tth2, uqlab2  = hkl[range1], uflab[range1], xy[range1], \
                                    nspot, Etheor[range1], ththeor[range1], chi[range1], tth[range1], uqlab[range1]
    
    if showall: 
        print("list of theoretical peaks")
        if remove_harmonics == "yes" : print("after removing harmonics")
        else : print("keeping all harmonics")
        print("hkl 0:3, uflab 3:6, uqlab, xy 6:8, th 8, Etheor 9")
        for i in range(nspots2):
            print(hkl2[i,:], uflab2[i,:].round(decimals=3), uqlab2[i,:].round(decimals=3),\
                round(xy2[i,0],2),round(xy2[i,1],2),round(ththeor2[i],4), round(Etheor2[i],1))
        print("nb of peaks :", nspots2)
        print("keep spots with over/under range pixel positions")

    print("hkl 0:3, uflab 3:6, xy 6:8, th 8, Etheor 9, chi 10, tth 11 ")
    
    spotlist2 = column_stack((hkl2,uflab2,xy2,ththeor2,Etheor2, chi2, tth2))

    if diagr == "side" :
        print("conversion to ydet downstream and zdet upwards :")
        print("ydet = ycam, zdet = xcam")
    
    #print(shape(spotlist2))

    return spotlist2
    

def spotlist_360(Emin_keV, 
                 Emax_keV, 
                 matwithlatpar_inv_nm, 
                 cryst_struct, 
                 showall, 
                 uilab = array([0.,1.,0.])):

    # modif de spotlist de calcdef_include.py
    # tous les spots avec q \E0 moins de x degres d'un certain plan ici plan yzlab.
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
    limangle = 90.
    sinangle = np.sin(limangle*PI/180.)

    mat = matwithlatpar_inv_nm

    uqlab_perp = array([1.,0.,0.])
    
    Emin_eV = Emin_keV*1000.
    Emax_eV = Emax_keV*1000.
        
    if (showall):
        print("calculate theoretical Laue pattern from orientation matrix")
        print("use matrix (with strain) - inverse nanometers \n", mat)
        print("energy range (keV):", Emin_keV, Emax_keV)
        print("max angle between uq and plane perpendicular to uqlab_perp (deg) : ", limangle)
        print("uqlab_perp =", uqlab_perp)
        print("structure :", cryst_struct)
    
    #print "cosangle = ", cosangle 
    yz = zeros((nmaxspots,2), float)
    hkl = zeros((nmaxspots,3), int)
    uflab = zeros((nmaxspots,3), float)
    Etheor = zeros(nmaxspots, float)
    ththeor = zeros(nmaxspots, float)
#    fpol = zeros(nmaxspots, float)

    tth = zeros(nmaxspots, float)
    chi = zeros(nmaxspots, float)
    
    dlatapprox = 1.0/norme(mat[0:3])
    print("dlatapprox in nanometers = ", dlatapprox)
    
    Hmax = int(dlatapprox*2*Emax_eV/DictLT.E_eV_fois_lambda_nm)

    if (showall):
        print("Hmax = ", Hmax)
        
    #xlab = array([1.,0.,0.]) # utile pour fpol
        
    nspot=0
    
    # Rlab z vers le haut, x vers le back, y vers l'aval
        
    for H in range(-Hmax, Hmax+1):
        for K in range(-Hmax, Hmax+1):
            if ((not (K-H)%2)|(cryst_struct=='BCC')):
                for L in range(-Hmax,Hmax+1):
                    if ((not (L-H)%2)|(cryst_struct=='BCC')) :
                        if ((cryst_struct=='FCC')|
                            ((cryst_struct=='dia')&((H%2)|((not H%2)&(not (H+K+L)%4))))|
                            ((cryst_struct=='BCC')&(not (H+K+L)%2)) 
                            ):
                            #print "hkl =", H,K,L
                            hkl1 = np.array([H,K,L], float)
                            
                            Etheor_eV, ththeor_deg, uqlab = mat_and_hkl_to_Etheor_ththeor_uqlab(matwithlatpar_inv_nm =matwithlatpar_inv_nm,
                                                             hkl = hkl1,
                                                             uilab = uilab)
                            if Etheor_eV > 0.01 :                                                                                              
                                sinangle2 = abs(inner(uqlab,uqlab_perp))
                                sintheta = -inner(uqlab,uilab)
                                if sinangle2 < sinangle :
                                    #print "reachable reflection"
                                    Etheor[nspot] = Etheor_eV
                                    ththeor[nspot] = ththeor_deg
                                    #print "Etheor = ", Etheor[nspot]
                                    if (Etheor[nspot] > Emin_eV)&(Etheor[nspot] < Emax_eV):                            
                                        uflabtheor = uilab + 2*sintheta*uqlab
                                        chi[nspot], tth[nspot] = uflab_to_2thetachi(uflabtheor)
                                        hkl[nspot,:] = array([H,K,L])
                                        uflab[nspot,:] = uflabtheor
                                        # calcul fpol
#                                        un = cross(uilab,uqlab)
#                                        un = un / norme(un)
#                                        fsig = inner(un,xlab)
#                                        fpi = np.sqrt(1-fsig*fsig)
#                                        cos2theta = np.cos(tth[nspot]*PI/180.0)
#                                        fpol[nspot] = np.sqrt(fsig*fsig + fpi*fpi*cos2theta*cos2theta)
                                        nspot = nspot + 1
#                            else :
#                                print "warning : norme(qlab) < 1e-5 in spotlist_360"
#                                print "H K L = ", H, K ,L

    spotlist2 = column_stack((Etheor, hkl, uflab, tth, chi)) #, fpol))

    print("E 0, hkl 1:4, uflab 4:7, tth 7, chi 8") #, fpol 9"

    spotlist2 = spotlist2[:nspot,:]

    spotlist2_sorted = sort_list_decreasing_column(spotlist2, 0)

    if (showall): 
        print("list of theoretical peaks keeping harmonics")
        print("Etheor, hkl, uflab, 2theta, chi ") #, fpol "
        for i in range(nspot):
            print(round(spotlist2_sorted[i,0],1), spotlist2_sorted[i,1:4], spotlist2_sorted[i,4:7].round(decimals=3),\
                round(spotlist2_sorted[i,7],4),round(spotlist2_sorted[i,8],2)) #,round(spotlist2_sorted[i,9],2)
        print("nb of peaks :", nspot)
    
    print(shape(spotlist2_sorted))

    toto=np.arange(nspot-1,-1,-1)
    #print toto
    spotlist2_sorted = spotlist2_sorted[toto,:]

    #savetxt("toto.txt",spotlist2_sorted, fmt = "%.6f")
    
    return(spotlist2_sorted)
    
def twofitfiles_to_rotation(filefit1 = None,
                            matref1 = None, # only for filefit1 = None
                            filefit2 = None, 
                            apply_cubic_opsym_to_mat2_to_have_min_angle_with_mat1 = "yes", 
                            apply_cubic_opsym_to_both_mat_to_have_one_sample_axis_in_first_stereo_triangle = None, # np.array([1.,0.,0.])
                            fitfile_type = "MGnew" # ,
                             ):
                                 
    verbose = 0
                                 
    if filefit1  is  not None :
#        print filefit1                                 
        res1 = read_any_fitfitfile_multigrain(filefit1, 
                                              verbose=verbose, 
                                              fitfile_type = fitfile_type,
                                              min_matLT = 0,   # seulement pour cubique
                                              check_pixdev_JSM = 0) 
#                    print "res1 = " , res1

        if res1 != 0 :
            gnumlist, npeaks, indstart, matstarlab_all1, data_fit, calib, meanpixdev, strain6, euler, ind_h_x_int_pixdev_Etheor = res1  

            matstarlab1 = matstarlab_all1[0]*1.
            
#            print "MG line 10802 : matstarlab1 = " , matstarlab1
    else :
        matstarlab1 = matref1
        
#    print filefit2                                 
    res2 = read_any_fitfitfile_multigrain(filefit2, 
                                          verbose=verbose, 
                                          fitfile_type = fitfile_type,
                                          min_matLT = 0,   # seulement pour cubique
                                          check_pixdev_JSM = 0) 
#                    print "res1 = " , res1

    if res2 != 0 :
        gnumlist, npeaks, indstart, matstarlab_all2, data_fit, calib, meanpixdev, strain6, euler, ind_h_x_int_pixdev_Etheor = res2  

        matstarlab2 = matstarlab_all2[0]*1.
        
#        print "MG line 10819 : matstarlab2 = ", matstarlab2
                
#    if filefit1  is  not None :
#        print filefit1
#        if origin_of_fitfile == "MG" :
#            matstarlab1, data_fit, calib, pixdev = F2TC.readlt_fit(filefit1, readmore = True)
#        elif origin_of_fitfile == "GUI" :
#            gnumlist, npeaks, indstart, matstarlab_all, data_fit, calib_all, \
#             pixdev, strain6, euler = readfitfile_multigrains(filefit1,
#                                                              verbose=1,
#                                                            readmore=True,
#                                                            fileextensionmarker=fileextensionmarker,
#                                                            default_file=None)
#            matstarlab1 = matstarlab_all[0]                                                
#    
#    else :
#        matstarlab1 = matref1
#    
#    if origin_of_fitfile == "MG" :    
#        matstarlab2, data_fit, calib, pixdev = F2TC.readlt_fit(filefit2, readmore = True)
#    elif origin_of_fitfile == "GUI" :
#        gnumlist, npeaks, indstart, matstarlab_all, data_fit, calib_all, \
#         pixdev, strain6, euler = readfitfile_multigrains(filefit2,
#                                                          verbose=1,
#                                                        readmore=True,
#                                                        fileextensionmarker=fileextensionmarker,
#                                                        default_file=None)
#        matstarlab2 = matstarlab_all[0]                                                
   

    if apply_cubic_opsym_to_both_mat_to_have_one_sample_axis_in_first_stereo_triangle  is  not None :
        axis1 = apply_cubic_opsym_to_both_mat_to_have_one_sample_axis_in_first_stereo_triangle
        matstarlab1, transfmat, rgb_axis =  matstarlab_to_orientation_color_rgb(matstarlab1, axis1)
        matstarlab2, transfmat, rgb_axis =  matstarlab_to_orientation_color_rgb(matstarlab2, axis1)
    
    matstarlabOND1 = matstarlab_to_matstarlabOND(matstarlab1)
    matstarlabOND2 = matstarlab_to_matstarlabOND(matstarlab2)
    mat1 = GT.matline_to_mat3x3(matstarlabOND1)
    mat2_start = GT.matline_to_mat3x3(matstarlabOND2)

#    print filefit2

    if apply_cubic_opsym_to_mat2_to_have_min_angle_with_mat1 == "yes" : 
        nop = 24
        allop = DictLT.OpSymArray
        indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])
        goodop = allop[indgoodop]
        vec_crystal = zeros((nop,3),float)
        vec_lab = zeros((nop,3),float)
        angle1 = zeros(nop,float)
        
        #print mat2_start.round(decimals=3)
        for k in range(nop):
            mat2 = dot(mat2_start, goodop[k])
            #print mat2.round(decimals=3)
            vec_crystal[k,:], vec_lab[k,:],angle1[k] = twomat_to_rotation(mat1,mat2)
                
        ind1 = argmin(angle1)
        print("opsym for minimal rotation : ", ind1)
        print("opsym matrix \n", goodop[k])
        print("minimal angle : ", round(angle1[ind1],3))

        return(vec_crystal[ind1], vec_lab[ind1], angle1[ind1], matstarlabOND1)
    
    elif apply_cubic_opsym_to_mat2_to_have_min_angle_with_mat1 == "no":
        
        mat2 = mat2_start
        vec_crystal, vec_lab, angle1 = twomat_to_rotation(mat1,mat2)        
        return(vec_crystal, vec_lab, angle1,  matstarlabOND1)

    else :
        return(1)
        
def plot_diagr2(xy, hkl = None, 
                plotmode = "firstplot", 
                plotlabel = None, 
                xmin0=0., 
                xmax0=2048., 
                ymin0 = 0., 
                ymax0 = 2048., 
                diagr = "top", 
                xs = "cross",
                add_label = None, 
                add_label2 = None,
                print_shortlist = "no",
                title = None,
                numfig = 1,
                savefig = 0,
                fileprefix = "",
                view_orientation = "observer_at_sample_xcam_right_ycam_down",
                color1 = "k",
                markersize = 12,
                marker1 = "o"
                ):

#spotlist_gen sort les coordonnees xcam ycam des spots
#
#ces coordonnees sont calculees pour que le diagramme trace 
#avec l'axe xcam vers la droite 
#et l'axe ycam vers le bas
#soit celui vu par un observateur 
#allonge parallelement au faisceau incident
#situe a la place de l'echantillon
#et qui recoit le faisceau incident par le sommet de la tete
#avec 2theta qui augmente vers le haut
#
#plot_diagr2 trace le graphe de ce diagramme "top"
#(en axes habituels xgraph vers la droite ygraph vers le haut)
#en utilisant 
#xgraph = xcam
#ygraph = -ycam
# c'est l'option view_orientation = "observer_at_sample_xcam_right_ycam_down"
#
#pour representer les diagrammes side 
#et les utiliser pour positionner le detecteur de fluo
#il est plus pratique d'avoir
#l'observateur debout devant le setup
#(donc cote "porte", ou cote gauche de l'observateur allonge precedent),
#qui regarde vers l'echantillon
#avec la tete droite
#ce qui revient a mettre
#l'axe xcam vers le haut
#et l'axe ycam vers la gauche
#
#plot_diagr2 trace ce graphe
#(en axes habituels xgraph vers la droite ygraph vers le haut)
#en utilisant 
#xgraph = -ycam
#ygraph = xcam
# c'est l'option view_orientation == "observer_in_camera_xcam_up_ycam_left"

    titre = None
    p.rcParams['lines.markersize'] = 10
#    p.rcParams['savefig.bbox'] = None

    matrot = PAR.mat_from_lab_to_sample_frame[1:,1:].transpose()
    
#    print "************************************"
#    print "printout from MG.plot_diagr2"

    if (diagr == 'side') :
        
        # ydet upstream, zdet upwards        
        yzdet_0deg = np.column_stack((-xy[:,1], xy[:,0]))
        
        xmin0 = -150.0
        xmax0 = +150.0
        ymin0 = -150.0
        ymax0 = 150.0
        
        if view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech" :
            xmin0 = -100.
            xmax0 = +100.
            ymin0 = -10.
            ymax0 = +70.     
#    print "color1 = ", color1
    if plotmode == "firstplot" :
        p.figure(num=numfig, figsize=(10,10))  # was 8 8        
        ax = p.subplot(111)
        if (view_orientation == "observer_at_sample_xcam_right_ycam_down"):
                p.plot(xy[:,0],-xy[:,1], marker = marker1, ms = markersize, mfc = color1, ls = "None")                
        elif (view_orientation == "observer_in_camera_xcam_up_ycam_left") :
            p.plot(-xy[:,1],xy[:,0], marker = marker1, label = titre, mfc = color1, ms = markersize, ls = "None")             
        elif (diagr == 'side') and (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
#            print "yzdet_0deg = ", yzdet_0deg
            yzdet_40deg = (dot(matrot,yzdet_0deg.transpose())).transpose()
            p.plot(yzdet_40deg[:,0], yzdet_40deg[:,1], marker = marker1, mfc = color1, ms = markersize , ls = "None")
            
    if diagr == "side" :   
        titre = 'side diagram : \n' + view_orientation
        p.axvline(x=0)
        p.axhline(y=0)
        tg40 = np.tan(40.*PI/180.)
        yy = np.array([ymin0, ymax0])
        xx =  yy*tg40
        if (view_orientation == "observer_at_sample_xcam_right_ycam_down"):
            p.plot(xx,-yy,"b-")
        elif (view_orientation == "observer_in_camera_xcam_up_ycam_left"):
            p.plot(-yy, xx,"b-")
        elif (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
            p.axhline(y = 0.)
           
    if plotmode == "overlay" :
        ax = p.subplot(111)
        if xs == "cross" :
            p.plot(xy[:,0],-xy[:,1], marker = "x", ms = markersize, mew = 1, mec = color1, ls = "None")
        else :
            p.plot(xy[:,0],-xy[:,1], marker = marker1, ms = markersize, mec = color1, mfc = "None", mew = 3, ls = "None")
    
#    print "yoho"
    ax.axis("equal")
    
    if hkl  is  not None :
            hkl1 = np.array(hkl.round(decimals=1), dtype = int)

    if (diagr == "top")|(diagr == "topvhr")|(diagr == "topvhrsmall") :
        p.xlabel('xcamXMAS - toward back')
        p.ylabel('-ycamXMAS - upstream')
          
    elif diagr == "side" :        
        if (view_orientation == "observer_at_sample_xcam_right_ycam_down"):
            p.xlabel('xcamXMAS - upwards')
            p.ylabel('-ycamXMAS - upstream')
        elif (view_orientation == "observer_in_camera_xcam_up_ycam_left"):        
            p.xlabel('-ydet = -ycam - upstream')
            p.ylabel('zdet = xcam user - upwards')
        elif (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
            p.xlabel('ydet downwards upstream along yech')
            p.ylabel('zdet upwards upstream along zech')          
                    
    elif diagr == "halfback":
        p.xlabel('xcam - toward back')
        p.ylabel('-ycam - upstream downwards')
    if (diagr == "topvhr")|(diagr == "halfback"):
        xmin0 = 0.0
        xmax0 = 2594.0
        ymin0 = 580.0
        ymax0 = 3420.0
    if (diagr == "topvhrsmall"):
        xmin0 = 0.0
        xmax0 = 2594.0
        ymin0 = 0.
        ymax0 = 2748.0
        
    #p.legend()
    if plotlabel  is  not None :        
        if diagr != "side" :
            shift_label = -20.
        else :
            shift_label = -5.
        
        if diagr == "side" :
            if (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
                print("yoho")
                ind0 = where((yzdet_40deg[:,0]> xmin0)&(yzdet_40deg[:,0]<xmax0)&(yzdet_40deg[:,1]> ymin0)&(yzdet_40deg[:,1]< ymax0))
            else :
               ind0 = where((yzdet_0deg[:,0]> xmin0)&(yzdet_0deg[:,0]<xmax0)&(yzdet_0deg[:,1]> ymin0)&(yzdet_0deg[:,1]< ymax0))
        else :
            ind0 = where((xy[:,0]> xmin0)&(xy[:,0]<xmax0)&(xy[:,1]> ymin0)&(xy[:,1]< ymax0))
            
        for i in range(len(xy[:,0])):
            if i in ind0[0]:
                x = xy[i,0]
                y = xy[i,1] + shift_label
                if hkl  is  not None :
                    label1 = str(hkl1[i,0])+ str(hkl1[i,1])+str(hkl1[i,2])
                if add_label  is  not None :
                    label1 = str(add_label[i])
                if add_label2  is  not None :
                    label1 = str(add_label2[i])  
                if (view_orientation == "observer_at_sample_xcam_right_ycam_down"):    
                    p.text(x, -y, label1)
                elif (view_orientation == "observer_in_camera_xcam_up_ycam_left"):
                    p.text(-y, x, label1)
                elif (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
                   x = yzdet_40deg[i,0]
                   y = yzdet_40deg[i,1] + shift_label
                   p.text(x, y, label1) 
                    
    p.xlim(xmin0, xmax0)
    if (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
        p.ylim(ymin0, ymax0)             
    else :
        p.ylim(-ymax0, -ymin0)
    
    if (print_shortlist == "yes") and (hkl  is  not None) :
        print("spot list truncated to camera frame")
        print("view orientation = ", view_orientation)
        str1 = "HKL xcam ycam"
        if diagr == "side" :
            if (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
               str1 = str1 + " ydet_40deg_upstream zdet_40deg_upwards"
            else :
               str1 = str1 + " ydet_0deg_upstream zdet_0deg_upwards"  
        print(str1)
#        ind0 = where((xy[:,0]> xmin0)&(xy[:,0]<xmax0)&(xy[:,1]> ymin0)&(xy[:,1]< ymax0))
        #print ind0[0]
        for j in ind0[0]:
            if diagr == "side" :
                if (view_orientation == "observer_in_camera_ydet40deg_along_yech_zdet40deg_along_zech") :
                    print(hkl[j,:], xy[j,:].round(decimals=2),yzdet_40deg[j,:].round(decimals=2))                  
                else :
                    print(hkl[j,:], xy[j,:].round(decimals=2),yzdet_0deg[j,:].round(decimals=2))
            else :
                print(hkl[j,:], xy[j,:].round(decimals=2))
        
    if title  is  not None :
        p.title(title)
    elif titre  is  not None :
        p.title(titre)
    
    if savefig :
        figfilename = fileprefix + "_LauePattern.png"
        p.savefig(figfilename, bbox_inches='tight')
        
#    print "view orientation = ", view_orientation
        
    return (1)
    
def select_4_spots(filepathfit, filepathdat, fileprefix, imgref, 
                   Emean = None, delta_E = None, 
                   list_spots = None, plot_spots = np.arange(4), ind_4spots = None) : 
    
    filefitmg1 = filepathfit + fileprefix + imgnum_to_str(imgref, PAR.number_of_digits_in_image_name) + PAR.add_str_fitfile + ".fit"
    gnumlist1, npeaks1, indstart1, matstarlab1, data_fit1, calib1, pixdev1 = readlt_fit_mg(filefitmg1)
    #print data_fit1
    #spot# Intensity h k l  pixDev xexp yexp Etheor
    indEtheor = 8
    indspotnum = 0
    indhkl = np.arange(2,5)
    indxy = np.arange(6,8)

    # get xy hkl Etheor pixdev from fitfile   
    data_fit1 = np.array(data_fit1, dtype=float)
    Etheor_list = data_fit1[:,indEtheor]
    if (Emean  is  not None)&(delta_E  is  not None) :
        Emin = (Emean-delta_E/2.0)*1000.
        Emax = (Emean + delta_E/2.0)*1000.
        ind0 = where((Etheor_list >Emin) &(Etheor_list < Emax))
        ind01 = ind0[0]
    elif list_spots  is  not None :
        ind01 = list_spots
    else :
        ind01 = np.arange(npeaks1)
        
#    print ind0[0]
    spotnum_list = np.array(data_fit1[:,indspotnum], dtype = int)
#    print "spotnum :"
#    print spotnum_list[ind0[0]]
    hkl_list = np.array(data_fit1[:,indhkl], dtype = int)
#    print "hkl"
#    print hkl_list[ind0[0]]
    xy_list = data_fit1[:,indxy]
#    print "xy"
#    print xy_list[ind0[0]]
    
    # get Ipixmax from datfile       
    filedat1 = filepathdat + fileprefix + imgnum_to_str(imgref, PAR.number_of_digits_in_image_name) + PAR.add_str_datfile + ".dat"
    data_xy, data_int, data_Ipixmax = read_dat(filedat1, filetype="LT")
    data_Ipixmax = np.array(data_Ipixmax, dtype = int)  
     
    if (Emean  is  not None)&(delta_E  is  not None) : print("spots within +/- ", delta_E/2.0, "from Emean = ", Emean) 
    print("spotnum_fit spotnum_dat hkl xy Etheor pixdev Ipixmax")
    for i in ind01 :
        print(i, spotnum_list[i], hkl_list[i], xy_list[i], round(Etheor_list[i],1), round(data_fit1[i,5],3), data_Ipixmax[spotnum_list[i]]) 


    if ind_4spots  is  not None : # attention spotnum_fit pas spotnum_dat 
        print("selected spots : ")
        print("spotnum_fit spotnum_dat hkl xy Etheor pixdev Ipixmax")
        for i in ind_4spots :
            print(i, spotnum_list[i], hkl_list[i], xy_list[i], round(Etheor_list[i],1), round(data_fit1[i,5],3), data_Ipixmax[spotnum_list[i]]) 

        limvij = 0.2
        limaijk = 0.2
        quad = np.arange(4)
        hkl_4spots = np.array(hkl_list[ind_4spots], dtype = float)
        if not(test_quad(quad, hkl_4spots,limvij, limaijk)):
            print("bad HKL quadruplet")    
            
    # check if spots are well distributed on camera screen
    
    if plot_spots  is  not None :
        print(ind_4spots)
        print(plot_spots)
        plot_diagr2(xy_list[ind01],  plotlabel = 1, add_label = ind01 )
#        plot_diagr2(xy_list[ind0[0]], hkl_list[ind0[0]], 
#                       plotlabel = 1, add_label = ind0[0], add_label2 = spotnum_list[ind0[0]] )
                                              
        if ind_4spots  is  not None : # attention spotnum_fit pas spotnum_dat 
            plot_diagr2(xy_list[ind_4spots[plot_spots]], plotmode = "overlay",
                           plotlabel = 1, add_label = ind_4spots[plot_spots])
    
    if ind_4spots  is  not None :                           
        return(xy_list[ind_4spots], hkl_list[ind_4spots])
    else :      
        return(0)
        
def four_spots_to_mat_new(quad, 
                      data_xy, 
                      data_hkl, 
                      calib, 
                      verbose = 1,
                      pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]):

    data_xy = np.array(data_xy, float)        
    data_hkl = np.array(data_hkl, float)
 
    xy_4spots = data_xy[quad]     
    hkl_4spots = data_hkl[quad]
    
    if verbose :
        print("xy_4spots = \n", xy_4spots)
        print("hkl_4spots = \n", hkl_4spots)
 
    for i in range(4):
        hkl_4spots[i,:] = hkl_4spots[i,:] / norme(hkl_4spots[i,:])     
    
    matHKL123 = hkl_4spots[0:3,:].transpose()
    
    det1 = np.linalg.det(matHKL123)
    
    print("det(matHKL123) = ", round(det1,3))

    inv_matHKL123 = inv(matHKL123)

    alpha123 = dot(inv_matHKL123,hkl_4spots[3,:])
    
    if verbose :
        print("alpha123 = coord of q4 in q1 q2 q3 frame : \n", alpha123.round(decimals=3))

    uqlab = zeros((4,3),float)

    if verbose :    
        print("pixelsize = ", pixelsize)
        
    for i in range(4) :
        uqlab[i,:] = xycam_to_uqlab(xy_4spots[i,:],
                               calib,
                               pixelsize = pixelsize)
#        print norme(uqlab[i,:])
    
    # on ramene uq4exp dans le repere OND construit sur uq1 uq2 uq3 exp    
    
    uq123_sur_xyzlab = uqlab[0:3,:].transpose()
    matstarlab123 = GT.mat3x3_to_matline(uq123_sur_xyzlab)
    matstarlab123_OND = matstarlab_to_matstarlabOND(matstarlab123)    
    
    matstarlab123_OND_3x3 = GT.matline_to_mat3x3(matstarlab123_OND)
    xyzlab_suz_uq123OND = matstarlab123_OND_3x3.transpose()
    
    uq4_R123_OND =  dot(xyzlab_suz_uq123OND, uqlab[3,:])

    if verbose :    
        print("uq4_R123_OND = ", uq4_R123_OND.round(decimals=3))
    
    matB123 = zeros((3,3),float)
    
    matB123[0,0] = 1.
    cos_gamma = inner(uqlab[0,:],uqlab[1,:])
    sin_gamma = np.sqrt(1.-cos_gamma*cos_gamma)
    matB123[0,1] = cos_gamma
    matB123[1,1] = sin_gamma
    cos_beta = inner(uqlab[0,:],uqlab[2,:])
    matB123[0,2] = cos_beta
    cos_alpha = inner(uqlab[1,:],uqlab[2,:])
    matB123[1,2] = (cos_alpha - cos_beta * cos_gamma)/sin_gamma
    matB123[2,2] = np.sqrt(1- matB123[0,2]*matB123[0,2] - matB123[1,2]* matB123[1,2])
    
    if verbose :
        print("matB123 sans les longueurs = \n", matB123.round(decimals=3)) 
    
    mat2 = matB123
    for i in range(3):
        mat2[:,i]= mat2[:,i]*alpha123[i]

    if verbose :
        print("mat2 = \n", mat2.round(decimals=3))         

    det1 = np.linalg.det(mat2)
    
    print("det(mat2) = ", round(det1,3))     
        
    inv_mat2 = inv(mat2)
    
    q123_sur_q4 = dot(inv_mat2, uq4_R123_OND)
    
    q123_sur_q4 = q123_sur_q4 / q123_sur_q4[0]
    
    if verbose :
        print("q123_sur_q4 = ", q123_sur_q4.round(decimals=4))
    
    matq = uq123_sur_xyzlab
    for i in range(3):
        matq[:,i]= matq[:,i]*q123_sur_q4[i]
        
    matstarlab3x3 = dot(matq, inv_matHKL123)
    matstarlab3x3 = matstarlab3x3 / norme(matstarlab3x3[:,0])
    
    if verbose :
        print("matstarlab3x3 = \n", matstarlab3x3.round(decimals=6))
    
#    print norme(matstarlab3x3[:,0])

    matstarlab = GT.mat3x3_to_matline(matstarlab3x3)

    return(matstarlab)        
               
        
def four_spots_to_mat(quad, 
                      data_xy, 
                      data_hkl, 
                      calib, 
                      showall = 1,
                      pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1]):

    # recup de calcdef_include apres enlevement des intensites

    alfexp=zeros((4,4),float)
    alftheor =zeros((4,4),float)
    uqcr = zeros((4,3), float)
    uqcam = zeros((4,3), float)
    qqcr = zeros((4,3), float)
    matstarlab = zeros(9,float)
    uqlab =  zeros((4,3), float)
    intwgt = 0.0
    
    uilab = array([0.0,1.0,0.0])

    if showall :
        print("xy = \n", data_xy[quad,:])
        print("hkl = \n", data_hkl[quad,:])
        
    data_hkl = np.array(data_hkl, float)
    data_xy = np.array(data_xy, float)
    
    for i in range(4) :
        iq = quad[i]
        qqcr[i,:] = data_hkl[iq, : ]# array([data_hkl[iq,0],data_hkl[iq,1],data_hkl[iq,2]])
        uqcr[i,:]= qqcr[i,:]/norme(qqcr[i,:])
        uqlab[i,:] = xycam_to_uqlab(data_xy[iq,:],
                               calib,
                               pixelsize = pixelsize)

        #print "uflab =", uflab
        #thlab = arccos(inner(uflab,uilab))*90.0/PI
        #print "th from uflab, uilab in deg", thlab

#        uqlab[i,:]= (uflab-uilab)/(2.0*np.sqrt( (1.0-inner(uflab,uilab))/2.0 ))

    for j in range(4):
        for k in range(4):
            if (k>j):
                alfexp[j,k]=np.arccos(inner(uqlab[j,:],uqlab[k,:]))
                alftheor[j,k]=np.arccos(inner(uqcr[j,:],uqcr[k,:]))
                #print "alfexp/alftheor (j,k) =", j, k, alfexp[j,k]/alftheor[j,k]
#            else :
#                alfexp[j,k]=0.0
#                alftheor[j,k]=0.0
                
    if showall :
        print("alfexp in deg \n", alfexp*180.0/PI)
    #print "alftheor in deg \n", alftheor*180.0/PI
        
    crtoRq= qqcr[0:3,:]                        
    #print "crtoRq = \n", crtoRq
    Rqtocr = inv(crtoRq)
    #print "Rqtocr = \n", Rqtocr
    q4Rq = qqcr[3,0]*Rqtocr[0,:]+qqcr[3,1]*Rqtocr[1,:]+qqcr[3,2]*Rqtocr[2,:]
    #print "q4Rq = ", q4Rq

    h14 = q4Rq[0]
    h24 = q4Rq[1]
    h34 = q4Rq[2]

    if ((abs(h14)<1e-3)|(abs(h24)<1e-3)|(abs(h34)<1e-3)):
        print("pb : h14, h24 or h34 at zero : exiting")
        return(epsp)

    signecostheta3 = sign(inner(cross(qqcr[0,:],qqcr[1,:]),qqcr[2,:]))

    signecostheta4 = sign(inner(cross(qqcr[0,:],qqcr[1,:]),qqcr[3,:]))

    #print "signecostheta3, signecostheta4 = ", signecostheta3, signecostheta4

    a12 = alfexp[0,1]
    a13 = alfexp[0,2]
    a23 = alfexp[1,2]
    a14 = alfexp[0,3]
    a24 = alfexp[1,3]

    C3 = (np.cos(a23)-np.cos(a12)*np.cos(a13))/np.sin(a12)

    C4 = (np.cos(a24)-np.cos(a12)*np.cos(a14))/np.sin(a12)

    costheta3 = signecostheta3*np.sqrt(1.0-np.cos(a13)*np.cos(a13)-C3*C3)

    costheta4 = signecostheta4*np.sqrt(1.0-np.cos(a14)*np.cos(a14)-C4*C4)

    if ((abs(signecostheta3)<1e-3)|(abs(signecostheta4)<1e-3)):
        print("pb : costheta3 or costheta4 at zero : exiting")
        return(epsp)

    #print "costheta3, costheta4 =", costheta3, costheta4
   
    rcos34 = costheta3/costheta4

    #print "C4*rcos34-C3", C4*rcos34-C3

    q3surq2 = h24*np.sin(a12)/(h34*(C4*rcos34-C3))    

    #print "q3surq2", q3surq2

    q1surq3 = (h34/h14)*((np.cos(a12)/np.sin(a12))*(C3-C4*rcos34)+rcos34*np.cos(a14)-np.cos(a13))

    q3surq1 = 1.0/q1surq3

    q2surq1 = q3surq1/q3surq2

    r13theor = norme(qqcr[2,:])/norme(qqcr[0,:])

    r12theor = norme(qqcr[1,:])/norme(qqcr[0,:])

    #print "q2surq1exp/q2surq1theor =", q2surq1/r12theor
   
    #print "q3surq1exp/q3surq1theor = ", q3surq1/r13theor
    
    q1lab = uqlab[0,:]
    q2lab = q2surq1*uqlab[1,:]    
    q3lab = q3surq1*uqlab[2,:]

    astarlab = Rqtocr[0,0]*q1lab + Rqtocr[0,1]*q2lab + Rqtocr[0,2]*q3lab
    bstarlab = Rqtocr[1,0]*q1lab + Rqtocr[1,1]*q2lab + Rqtocr[1,2]*q3lab
    cstarlab = Rqtocr[2,0]*q1lab + Rqtocr[2,1]*q2lab + Rqtocr[2,2]*q3lab

    matstarlab = hstack((astarlab,bstarlab,cstarlab))/norme(astarlab)

    if showall :
        print("pixelsize = ", pixelsize)
        print("matstarlab \n", matstarlab)
       
    return(matstarlab)  

def test_quad(quad, data_hkl,limvij, limaijk):
    
    #print "test d'un quadruplet : 
    #print "bon quadruplet : vecteurs diffraction non colineaires et non coplanaires 3 a 3"
    #print "angle minimal entre deux vecteurs : limvij = ", limvij
    #print "angle minimal entre un vecteur et le plan defini par deux autres vecteurs : limaijk = ", limaijk

    data_hkl = np.array(data_hkl, dtype = float)
    hkl_4spots = zeros((4,3),float)
    
    for i in range(4):
        hkl_4spots[i,:] = data_hkl[quad[i],:]
        hkl_4spots[i,:] = hkl_4spots[i,:] / norme(hkl_4spots[i,:])
 
#    print hkl_4spots    

    list_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    
    vij_list = zeros(6,float)
    vij_list2 = zeros((4,4),float)
    n_list = zeros((4,4,3),float)
    
    angle_list = zeros((4,4),float)    
    k = 0    
    for pair1 in list_pairs :
        n1 = pair1[0]
        n2 = pair1[1]
        angle_list[n1,n2] = math.acos(inner(hkl_4spots[n1,:],hkl_4spots[n2,:]))*180./PI
        angle_list[n2,n1] = angle_list[n1,n2]
    
    print("angles list (deg) = \n", angle_list.round(decimals = 3))
    
    k = 0    
    for pair1 in list_pairs :
        n1 = pair1[0]
        n2 = pair1[1]
        n_list[n1,n2,:] = cross(hkl_4spots[n1,:],hkl_4spots[n2,:])
        vij_list[k] = norme(n_list[n1,n2,:])
        vij_list2[n1,n2] = vij_list[k]
        vij_list2[n2,n1] = vij_list[k]
        
        n_list[n1,n2,:] = n_list[n1,n2,:] / vij_list[k]
        n_list[n2,n1,:] = n_list[n1,n2,:]
        if  vij_list[k] < limvij :
            print("bad quadruplet ", n1, n2, "vij = ",  round(vij_list[k],3))
            return(None)
        k = k+1
    
    if 0 : # old version   
    #    det_list = zeros(12,float)
        list_triplets = np.array([[0,1,2],[0,1,3],[0,2,3], 
                         [1,0,2],[1,0,3],[1,2,3],
                        [2,0,1],[2,0,3],[2,1,3],
                        [3,0,1],[3,0,2],[3,1,2]])
                            
        aijk_list = zeros(12, float)
    #    det_list = zeros(12,float)
    
        k = 0
        for triplet in list_triplets :  
    #        matHKL123 = hkl_4spots[triplet,:].transpose()
    #        det_list[k] = abs(np.linalg.det(matHKL123))
                
            n1 = triplet[0]
            n2 = triplet[1]
            n3 = triplet[2]
            vec1 = cross(n_list[n1,n2,:],n_list[n1,n3,:])
            aijk_list[k] = norme(vec1)
            if (aijk_list[k] < limaijk):
                print("bad quadruplet ", n1, n2, n3, round(aijk_list[k],3))
                return(None)   
            k = k+1

    list_triplets = np.array([[0,1,2],[0,1,3],[0,2,1],[0,2,3], 
                     [0,3,1],[0,3,2],[1,2,0],[1,2,3],
                    [1,3,0],[1,3,2],[2,3,0],[2,3,1]])
                        
    aijk_list = zeros(12, float)

    k = 0
    for triplet in list_triplets :  
            
        n1 = triplet[0]
        n2 = triplet[1]
        n3 = triplet[2]
        aijk_list[k] = abs(inner(n_list[n1,n2,:],hkl_4spots[n3,:]))
        if (aijk_list[k] < limaijk):
            print("bad quadruplet ", n1, n2, n3, round(aijk_list[k],3))
            return(None)   
        k = k+1
                              
    vmin = min(vij_list)
    amin = min(aijk_list)
#    detmin = min(det_list)

#    print "list_pairs =", list_pairs
    print("vij_list = ", vij_list.round(decimals = 3))
#    print "list_triplets =", list_triplets
    print("aijk_list = ", aijk_list.round(decimals = 3))       
#    print "det_list = ", det_list.round(decimals = 3)

#    print "vmin , amin, detmin =", round(vmin,4), round(amin,4), round(detmin,4)
    
    print("vmin , amin =", round(vmin,4), round(amin,4))
    angv =  math.asin(vmin) * 180./PI
    anga =  math.asin(amin) * 180./PI
    print("vmin, amin as angles(deg) = ",  round(angv,2), round(anga,2))
    print("good quadruplet")
    
    
    # arrangement de l'ordre du quadruplet 
    # pour avoir un triedre triedre de grands angles
    # pour q1 q2 q3    
    
    ind0 = argmax(vij_list)
#    print "max of vij_list for ind0 = ", ind0
    print("max of vij_list for pair = ", list_pairs[ind0])
    
    pair1 = list_pairs[ind0]
    
    cross01 = n_list[pair1[0],pair1[1],:] / vij_list2[pair1[0],pair1[1]]
    
    scalar01_i = zeros(2,float)
    ind_others = zeros(2,int)
    k = 0
    for i in range(4) :
        if i not in pair1 :
            ind_others[k] = i
            scalar01_i[k] =  inner(hkl_4spots[i,:],cross01)
            print("i, scalar inner(cross(uq0,uq1),uqi) =", i, round(scalar01_i[k],3)) 
            k = k+1

    toto = abs(scalar01_i)
    ind1 = argmax(toto)
    print(toto[ind1])
    print(ind_others[ind1])
    if scalar01_i[ind1] > 0.0 :
        best_triplet = np.array([pair1[0],pair1[1],ind_others[ind1]])
    else :
        best_triplet = np.array([pair1[1],pair1[0],ind_others[ind1]])
    print("best triplet pour triedre direct de grands angles = ", best_triplet)
    
    for i in range(4) : 
        if i not in best_triplet : k = i
    best_quad = zeros(4,int)
    best_quad[0:3] = best_triplet
    best_quad[3] = k
    print("best quad = ", best_quad)    # permutations de 0 1 2 3
    
    best_quad2 = quad[best_quad]   # permutations sur quad
    print("best_quad2 = ", best_quad2)
    
    # nouveau test : equilibre entre les composantes de q4 sur q1 q2 q3
    for i in range(4):
        hkl_4spots[i,:] = data_hkl[best_quad2[i],:]
        hkl_4spots[i,:] = hkl_4spots[i,:] / norme(hkl_4spots[i,:])
    
    matHKL123 = hkl_4spots[0:3,:].transpose()
    det123 = abs(np.linalg.det(matHKL123))

    vij_shortlist = np.array([vij_list2[best_quad[0],best_quad[3]],
                              vij_list2[best_quad[1],best_quad[3]],
                                vij_list2[best_quad[2],best_quad[3]]])
                                
    vmin2 = min(vij_shortlist)
    
    a412 = abs(inner(hkl_4spots[3,:],n_list[best_quad[0],best_quad[1],:]))
    a413 = abs(inner(hkl_4spots[3,:],n_list[best_quad[0],best_quad[2],:]))
    a423 = abs(inner(hkl_4spots[3,:],n_list[best_quad[1],best_quad[2],:]))
    
    aijk_shortlist = np.array([a412,a413,a423])

    amin2 = min(aijk_shortlist)

    print("vij_shortlist = ",  vij_shortlist.round(decimals=3))                             
    print("aijk_shortlist = ", aijk_shortlist.round(decimals=3))
    
    critere1 = vmin2*amin2*det123
#    critere1 = vmin*amin

    if 0 :
        inv_matHKL123 = inv(matHKL123)
    
        alpha123 = dot(inv_matHKL123,hkl_4spots[3,:])
        
    #    if verbose :
        print("alpha123 = coord of q4 in q1 q2 q3 frame : \n", alpha123.round(decimals=3))    
    
        abs_alpha123 = abs(alpha123)
        std_alpha123 = abs_alpha123.std()
        
        print("std_alpha123 = ", round(std_alpha123,3))
    
    print("best_quad det123 vmin2 amin2 critere1")
    print(best_quad2, round(det123,4), vmin2.round(decimals=3),amin2.round(decimals=3),critere1.round(decimals=4))
    
    return(best_quad2, det123, vmin2, amin2, critere1)

def test_all_quads(hkl,
                   limvij, 
                   limaijk, 
                   newfilename):
                       
    #def test_all_quads(hkl, limvij, limaijk, filename):

    # added 04 Mar 2010

    nquad = 0
    Npics = shape(hkl)[0]
    Ntottrip=Npics*(Npics-1)*(Npics-2)/6
    Ntotquad=Ntottrip*(Npics-3)/4

    quad1 = zeros(4,int)
    allquads = zeros((Ntotquad,9),float)
    
    print("Npics, Ntottrip, Ntotquad", Npics, Ntottrip, Ntotquad)
    if Ntotquad > 2000:
        print("warning : long calculation ahead")

    for ii in range(Npics):
        for jj in range(Npics):
            if (jj>ii):
               for kk in range(Npics):
                   if (kk>jj):
                       for LL in range(Npics):
                           if (LL>kk):
                               quad1 = array([ii,jj,kk,LL])
                               res1 = test_quad(quad1,hkl,limvij, limaijk)
                               if res1  is  not None :
                                   best_quad2, det123, vmin2, amin2, critere1 = res1
                                   print("nquad =", nquad)
                                   print("quad1 = ", quad1)
                                   allquads[nquad,0]= float(nquad)
                                   allquads[nquad,1:5]= best_quad2
                                   allquads[nquad, 5]= det123
                                   allquads[nquad, 6]= vmin2
                                   allquads[nquad, 7]= amin2
                                   allquads[nquad, 8]= critere1
                                   nquad = nquad + 1

    print("nquad = ", nquad)
    if nquad > 0 :

        print("good quads saved to file : ", newfilename)
        savetxt(newfilename,allquads[:nquad,:],fmt='%.4f')                                 

    print("nquad, best_quad, det123, vmin2, amin2, critere1")
    for i in range(nquad):
        print(i, allquads[i,1:5], allquads[i,5:9].round(decimals=4))

    return(allquads[:nquad,:])

def search_for_4_spots(filepathdat, 
                       fileprefix, 
                       filesuffix, 
                       filepathout, 
                       indimg, 
                       xytol, 
                       xy_4spots,
                       filetype = "LT",
                       add_str = ""): 

    nimg = len(indimg)

    xy4all = zeros((nimg,8), float)

    kk = 0
    for i in indimg :
        filedat1 = filepathdat + fileprefix + imgnum_to_str(i,PAR.number_of_digits_in_image_name) + filesuffix
        res1 = read_dat(filedat1, filetype=filetype)
        if filetype == "LT" : data_xy, data_int, data_Ipixmax = res1
        elif filetype == "XMAS" : data_xy, data_int = res1
        data_xy = np.array(data_xy, dtype = float)
        npics = shape(data_xy)[0]
        if not(np.isscalar(data_xy[0])):
            for j in range(4):
                dxy = data_xy-xy_4spots[j,:]
                for k in range(npics):
                    if norme(dxy[k,:])< xytol :
                        xy4all[kk,j*2:j*2+2] = data_xy[k,:]
                        break
        ##                print "i = ", i
        ##                print "j = ", j
        ##                print "k = ", k
        print(xy4all[kk,:])
        kk = kk+1
    
            
    print("xy4all = \n", xy4all)

    toto = column_stack((indimg, xy4all))
    
    outfilename = filepathout + "xy_4spots_" + fileprefix + str(indimg[0]) + "_" + str(indimg[-1]) + add_str + ".dat"    
    header = "img x0 y0 x1 y1 x2 y2 x3 y3 \n"
    print("saving spot positions in : \n", outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)
    np.savetxt(outputfile, toto, fmt = "%.2f")
    outputfile.close()

    return(xy4all)
    
def serial_two_or_four_spots_to_mat(filepathout, 
                                   fileprefix, 
                                   file_xy4all, 
                                   hkl_4spots, 
                                   calib, 
                                   ind_spots_for_mat = [0,1],
                                    test = 0,
                                    no_mat_only_spot_distances = 0,
                                    distance_as_angle = 0,
                                    add_str = ""): 
    
    # matrices non deformee a partir des positions de 2 pics de HKL connus
    # ou matrice deformee a partir des positions de 4 pics de HKL connus
    # xy des 2 ou 4 spots a partir d'un fichier liste : img x0 y0 x1 y1 x2 y2 x3 y3 
    # reseau cubique seulement

    data_all = loadtxt(file_xy4all, skiprows = 1)
    xy4all = data_all[:,1:]
    imgall = np.array(data_all[:,0], dtype = int)
    nimg = shape(data_all)[0]
        
    nspots = len(ind_spots_for_mat)
    
    ndec = 6
    ncol = 9
    fmt1 = "%.6f"
    if nspots == 2 : 
        print("unstrained matrix from two spots + delta_alf epx-theor")
        header = "img 0, xy1 1:3, xy2 3:5, matstarlab 5:14 delta_alf 14 \n"
        header2 = "img"
        for i in range(2):
            label2 = " xy"+str(i+1)+"_0 xy"+str(i+1)+"_1"
            header2 = header2 + label2
        for i in range(ncol) :
            label2 = " matstarlab_" + str(i) 
            header2 = header2 + label2        
        header2 = header2 + " dalf \n"  
        outfilename = filepathout + "mat_2spots_" + fileprefix + "img" +\
            str(imgall[0]) + "to" + str(imgall[-1]) + "_alf" + add_str + ".dat"    
        print(outfilename)
        print(header)
        print(header2)

    elif nspots == 4 : 
        if no_mat_only_spot_distances == 0 :
            print("strained matrix from four spots")
            header = "img 0, xy1 1:3, xy2 3:5, xy3 5:7, xy4 7:9, matstarlab 9:18 \n"
            outfilename = filepathout + "mat_4spots_" + fileprefix +\
                str(imgall[0]) + "_" + str(imgall[-1]) + add_str + ".dat"
            limvij = 0.2
            limaijk = 0.2
            quad = np.arange(4)
            print(hkl_4spots)
            if not(test_quad(quad, hkl_4spots,limvij, limaijk)):
                print("bad HKL quadruplet")
                if test == 0 : return(None)
        else :
            ndec = 2
            ncol = 6
            fmt1 = "%.2f"
            print("distances between pairs of spots for 4 spots")
            list_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype = int)
            outfilename = filepathout + "dist_4spots_" + fileprefix +\
                str(imgall[0]) + "_" + str(imgall[-1]) + ".dat"  
            header = "img 0, xy1 1:3, xy2 3:5, xy3 5:7, xy4 7:9"
            for i in range(ncol) :
                label1 = ", d" + str(list_pairs[i,0]) + str(list_pairs[i,1]) + " " + str(9+i)                   
                header = header + label1
            header = header + "\n"
            header2 = "img"
            for i in range(4):
                label2 = " xy"+str(i+1)+"_0 xy"+str(i+1)+"_1"
                header2 = header2 + label2
            for i in range(ncol) :
                label2 = " d" + str(list_pairs[i,0]) + str(list_pairs[i,1]) 
                header2 = header2 + label2 
            header2 = header2 + "\n"    
            print(outfilename)
            print(header)
            print(header2)

    else : 
        print("wrong ind_spots_for_mat : needs 2 or 4 spots")
        return(0)

    #nimg = 50
    
    hkl = hkl_4spots[ind_spots_for_mat,:]
    print(hkl)
    
    toto = zeros(2*nspots, int)
    k = 0
    for i in ind_spots_for_mat :
        toto[k] = 2*i
        toto[k+1] = 2*i+1
        k = k+2
    print(toto)
    
    xy2or4 = xy4all[:,toto]
    print(shape(xy2or4))
      
    xymin = xy2or4.min(axis = 1)
    print(shape(xymin))
    ind0 = where(xymin > 0.1)
    print(shape(ind0))
    
    nimg2 = len(ind0[0])
    
    print("found %d images with the %d spots present" %(nimg2, nspots))
    
    dat_all = zeros((nimg2,ncol), float)
    dalf_all = zeros(nimg2, float)
        
    for i in range(nspots):
        xy2 =  xy2or4[:,2*i:2*i+2]
        xy2min = xy2.min(axis = 1)
        ind3 = where(xy2min > 0.1)            
        print("spot nimg xymean yxstd xyrange")
        print(i, shape(ind3)[1], xy2[ind3[0],:].mean(axis=0).round(decimals=2),\
            xy2[ind3[0],:].std(axis=0).round(decimals=2), \
            xy2[ind3[0],:].max(axis=0)-xy2[ind3[0],:].min(axis=0))
        print("spot nimg2 xymean yxstd xyrange")
        print(i, shape(ind0)[1], xy2[ind0[0],:].mean(axis=0).round(decimals=2),\
            xy2[ind0[0],:].std(axis=0).round(decimals=2), \
            xy2[ind0[0],:].max(axis=0)-xy2[ind0[0],:].min(axis=0))    

# point d'arret standard
    if test == 1 : jlfsdfs
        
    if no_mat_only_spot_distances == 1 : print(list_pairs)        
    k = 0
    for i in ind0[0]:
        #print "k = ", k
        xycam = xy2or4[i,:].reshape(nspots,2)        
#        print xycam
        if nspots == 2 : 
            dat_all[k,:], dalf_all[k] = two_spots_to_mat(hkl,xycam,calib)
        elif nspots == 4 :
            if no_mat_only_spot_distances == 0 :           
                dat_all[k,:] = four_spots_to_mat(quad, xycam, hkl_4spots, calib, showall = 1)
            else :
                for j in range(ncol):
#                    print xycam[list_pairs[j,0],:], xycam[list_pairs[j,1],:]
                    if distance_as_angle == 0 :
                        dat_all[k,j] = norme(xycam[list_pairs[j,0],:]-xycam[list_pairs[j,1],:])
                    else :
                        uq0 = xycam_to_uqlab(xycam[list_pairs[j,0],:], calib, pixelsize = pixelsize)
                        uq1 = xycam_to_uqlab(xycam[list_pairs[j,1],:], calib, pixelsize = pixelsize)
                        dat_all[k,j] = math.acos(inner(uq0,uq1))*1000.
                        
                print(dat_all[k,:].round(decimals = ndec))
        k = k+1
     
    print(dat_all.round(decimals=ndec))
    
    toto = column_stack((imgall[ind0[0]], xy2or4[ind0[0],:], dat_all.round(decimals=ndec)))

    if nspots == 2 :
        print(dalf_all.round(decimals=2))
        toto = column_stack((toto,dalf_all.round(decimals=3)))
    
    print(shape(toto))
    print(outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)
    if (no_mat_only_spot_distances == 1)|(nspots == 2) : outputfile.write(header2)
    np.savetxt(outputfile, toto, fmt = fmt1)
    outputfile.close()

    return(outfilename)    


def serial_four_spots_to_mat(filepathout, 
                                   fileprefix, 
                                   filespot_list,
                                   hkl_4spots, 
                                   calib, 
                                    test = 0,
                                    add_str = "",
                                    pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1],
                                    threshold_on_Imax_ratio = 0.002,
                                    imgref_for_superimposing_spot_trajectories = None,
                                    test_quad = "no",
                                    limvij = 0.125,
                                    limaijk = 0.125): 
    
    # matrices non deformee a partir des positions de 2 pics de HKL connus
    # ou matrice deformee a partir des positions de 4 pics de HKL connus
    # xy des 2 ou 4 spots a partir d'un fichier liste : img x0 y0 x1 y1 x2 y2 x3 y3 
    # reseau cubique seulement

    data_list1, listname, nameline0 = read_summary_file(filespot_list[0])  
    data_list2, listname, nameline0 = read_summary_file(filespot_list[1]) 
    data_list3, listname, nameline0 = read_summary_file(filespot_list[2])  
    data_list4, listname, nameline0 = read_summary_file(filespot_list[3])  
    
    data_list1 = np.array(data_list1, dtype=float)
    data_list2 = np.array(data_list2, dtype=float)    
    data_list3 = np.array(data_list3, dtype=float)
    data_list4 = np.array(data_list4, dtype=float)   

    nimg = shape(data_list1)[0]
    print(nimg)
    ndata_cols = shape(data_list1)[1]
    print(ndata_cols)
                   
    indimg = listname.index("img") 
    img_list1 = np.array(data_list1[:,indimg],int)
    img_list2 = np.array(data_list2[:,indimg],int)
    img_list3 = np.array(data_list3[:,indimg],int)
    img_list4 = np.array(data_list4[:,indimg],int)
    print("img_list1 = ", img_list1)
    
    indxfit = listname.index("xfit")
    indxyfit = np.array([indxfit, indxfit+1])
    xyfit_list1 = data_list1[:,indxyfit]
    xyfit_list2 = data_list2[:,indxyfit]
    xyfit_list3 = data_list3[:,indxyfit]
    xyfit_list4 = data_list4[:,indxyfit]
    
    indImax = listname.index("Imax") 
    Imax_list1 = data_list1[:,indImax] 
    Imax_list2 = data_list2[:,indImax]
    Imax_list3 = data_list3[:,indImax] 
    Imax_list4 = data_list4[:,indImax]
    Imax_ratio_list1 = Imax_list1 / max(Imax_list1)
    Imax_ratio_list2 = Imax_list2 / max(Imax_list2)
    Imax_ratio_list3 = Imax_list3 / max(Imax_list3)
    Imax_ratio_list4 = Imax_list4 / max(Imax_list4)
    
    ncol = 9

    print("strained matrix from four spots")
    header = "img 0, xy1 1:3, xy2 3:5, xy3 5:7, xy4 7:9, matstarlab 9:18 \n"
    header2 = "img"
    for i in range(4):
        label2 = " xy"+str(i+1)+"_0 xy"+str(i+1)+"_1"
        header2 = header2 + label2
    for i in range(ncol) :
        label2 = " matstarlab_" + str(i) 
        header2 = header2 + label2        
    header2 = header2 + "\n"
    
    outfilename = filepathout + "mat_4spots_" + fileprefix +\
        str(img_list1[0]) + "_" + str(img_list1[-1]) + add_str + ".dat"

    quad = np.arange(4)
    print(hkl_4spots)
    
    if test_quad == "yes" :
        best_quad2 = test_quad(quad, hkl_4spots, limvij, limaijk)
        if isscalar(best_quad2):
            print("bad HKL quadruplet")
            if test == 0 : return(0)
    else :
        best_quad2 = quad
       
    if imgref_for_superimposing_spot_trajectories  is  not None :
        xpic, ypic = 0, 0
        xboxsize, yboxsize = 100, 100
        titre = ""
        ind1 = where(img_list1 == imgref_for_superimposing_spot_trajectories)
        xy1 = xyfit_list1-xyfit_list1[ind1[0],:]
        ind2 = where(img_list2 == imgref_for_superimposing_spot_trajectories)        
        xy2 = xyfit_list2-xyfit_list2[ind2[0],:]
        ind3 = where(img_list3 == imgref_for_superimposing_spot_trajectories)
        xy3 = xyfit_list3-xyfit_list3[ind3[0],:]
        ind4 = where(img_list4 == imgref_for_superimposing_spot_trajectories)        
        xy4 = xyfit_list4-xyfit_list4[ind4[0],:]
        
        p.figure()
        p.plot(xy1[:,0], xy1[:,1], "ro-")
        for i in range(nimg) : p.text(xy1[i,0],xy1[i,1],str(img_list1[i]), fontsize = 16)
        p.plot(xy2[:,0], xy2[:,1], "bs-")
        for i in range(nimg) : p.text(xy2[i,0],xy2[i,1],str(img_list2[i]), fontsize = 16)
        p.plot(xy3[:,0], xy3[:,1], "gv-")
        for i in range(nimg) : p.text(xy3[i,0],xy3[i,1],str(img_list3[i]), fontsize = 16)
        p.plot(xy4[:,0], xy4[:,1], "k^-")
        for i in range(nimg) : p.text(xy4[i,0],xy4[i,1],str(img_list4[i]), fontsize = 16)

        p.xlabel("xpix")
        p.ylabel("ypix")

    xy_4spots = np.column_stack((xyfit_list1, xyfit_list2, xyfit_list3, xyfit_list4))
    Imax_ratio_4spots = np.column_stack((Imax_ratio_list1, Imax_ratio_list2, Imax_ratio_list3, Imax_ratio_list4))
    
    imgall = img_list1
    nimg = shape(xy_4spots)[0]
    
    hkl = hkl_4spots
    print(hkl)
    
    Imax_ratio_min = Imax_ratio_4spots.min(axis=1)
    ind0 = where( Imax_ratio_min > threshold_on_Imax_ratio )  
    print(shape(ind0))
    
    nimg2 = len(ind0[0])
    
    print("found %d images with Imax of 4 spots larger than %.4f" %(nimg2, threshold_on_Imax_ratio))
    
    matstarlab_all = zeros((nimg2,ncol), float)
    
    print("spot nimg xymean xystd xyrange")    
    for i in range(4):           
        print(i, shape(ind0)[1], xy_4spots[ind0[0],2*i:2*i+2].mean(axis=0).round(decimals=2),\
            xy_4spots[ind0[0],2*i:2*i+2].std(axis=0).round(decimals=2), \
            xy_4spots[ind0[0],2*i:2*i+2].max(axis=0)-xy_4spots[ind0[0],2*i:2*i+2].min(axis=0))   

# point d'arret standard
    if test == 1 : jlfsdfs
    
    ndec = 6      
    k = 0
    for i in ind0[0]:
        #print "k = ", k
        xycam = xy_4spots[i,:].reshape(4,2)        
#        print "xycam = ", xycam

        print("img = ", imgall[i])
        matstarlab_all[k,:] = four_spots_to_mat_new(best_quad2, 
                                                xycam, 
                                                hkl_4spots,
                                                calib,
                                                verbose = 1,
                                                pixelsize = pixelsize)
                                                   
        k = k+1
     
#    print matstarlab_all.round(decimals=ndec)

    fmt1 = "%.6f"
    toto = column_stack((imgall[ind0[0]], xy_4spots[ind0[0],:], matstarlab_all.round(decimals=ndec)))
    
    print(shape(toto))
    print(outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, toto, fmt = fmt1)
    outputfile.close()

    return(outfilename)    
    
def serial_two_spots_to_mat(filepathout, 
                                   fileprefix, 
                                   filespotmon1,
                                   filespotmon2,
                                   hkl_2spots, 
                                   calib, 
                                    test = 0,
                                    add_str = "",
                                    pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1],
                                    threshold_on_Imax_ratio = 0.002,
                                    imgref_for_superimposing_spot_trajectories = None,
                                    omega = None, # was PAR.omega_sample_frame,
                                    mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame,
                                    elem_label = None,
                                    point_num_for_superimposing_spot_trajectories = None,
                                    add_last_N_columns_from_filespotmon = 0,
                                    verbose = 1,
                                    use_xy_max_or_xyfit = "xyfit"
                                    ):
    
    # matrices non deformee a partir des positions de 2 pics de HKL connus
    # ou matrice deformee a partir des positions de 4 pics de HKL connus
    # xy des 2 ou 4 spots a partir d'un fichier liste : img x0 y0 x1 y1 x2 y2 x3 y3 

    data_list1, listname, nameline0 = read_summary_file(filespotmon1)  
    data_list2, listname, nameline0 = read_summary_file(filespotmon2)  
    
    data_list1 = np.array(data_list1, dtype=float)
    data_list2 = np.array(data_list2, dtype=float)    

    nimg = shape(data_list1)[0]
    print(nimg)
    ndata_cols = shape(data_list1)[1]
    print(ndata_cols)
                       
    indimg = listname.index("img") 
    img_list1 = np.array(data_list1[:,indimg],int)
    img_list2 = np.array(data_list2[:,indimg],int)
    print("img_list1 = ", img_list1)
    
    point_list1 = np.arange(len(img_list1))
    point_list2 = np.arange(len(img_list2))
    
    if use_xy_max_or_xyfit == "xyfit" :
        indxfit = listname.index("xfit")
    elif use_xy_max_or_xyfit == "xymax" :
        indxfit = listname.index("xmax")
        
    indxyfit = np.array([indxfit, indxfit+1])
    xyfit_list1 = data_list1[:,indxyfit]
    xyfit_list2 = data_list2[:,indxyfit]
    
    indImax = listname.index("Imax") 
    Imax_list1 = data_list1[:,indImax] 
    Imax_list2 = data_list2[:,indImax]
    if "Seconds" in listname :
        indSeconds =  listname.index("Seconds")
        Seconds_list = data_list1[:,indSeconds]
        Imax_list1 = Imax_list1/Seconds_list
        Imax_list2 = Imax_list2/Seconds_list       
    
    Imax_ratio_list1 = Imax_list1 / max(Imax_list1)
    Imax_ratio_list2 = Imax_list2 / max(Imax_list2)
    
    if 1 :
        fig, ax1 = p.subplots()
        ax1.plot(point_list1,Imax_ratio_list1, "ro-" )
        ax1.plot(point_list2,Imax_ratio_list2, "bs-" )
        ax1.set_yscale('log')
        ax1.set_xlabel('npts')
        ax1.set_ylabel("(Imax/Seconds) / max(Imax/Seconds) ")
        ax1.axhline(y = threshold_on_Imax_ratio)
    
    if imgref_for_superimposing_spot_trajectories  is  not None :
        xpic, ypic = 0, 0
        xboxsize, yboxsize = 100, 100
        ind1 = where(img_list1 == imgref_for_superimposing_spot_trajectories)
        print(ind1[0])
        xy1 = xyfit_list1-xyfit_list1[ind1[0],:]
        ind2 = where(img_list2 == imgref_for_superimposing_spot_trajectories)        
        xy2 = xyfit_list2-xyfit_list2[ind2[0],:]
        
    if point_num_for_superimposing_spot_trajectories  is  not None :
        ind1 = where(point_list1 == point_num_for_superimposing_spot_trajectories)
        print(ind1[0])
        xy1 = xyfit_list1-xyfit_list1[ind1[0],:]
        ind2 = where(point_list2 == point_num_for_superimposing_spot_trajectories)        
        xy2 = xyfit_list2-xyfit_list2[ind2[0],:]
            
    xy_2spots = np.column_stack((xyfit_list1,xyfit_list2))
    Imax_ratio_2spots = np.column_stack((Imax_ratio_list1, Imax_ratio_list2))
    
    imgall = img_list1
    nimg = shape(xy_2spots)[0]
    
    ndec = 6
    ncol = 9
    fmt1 = "%.6f"  # doit coller avec ndec
    
    str_names_to_add = ""
    if add_last_N_columns_from_filespotmon > 0 :
        n1 = add_last_N_columns_from_filespotmon
        listnames_to_add = listname[-n1:]
        str_names_to_add = " ".join(listnames_to_add)
        
    print("unstrained matrix from two spots + delta_alf epx-theor")
    header = "img 0, xy1 1:3, xy2 3:5, matstarlab 5:14 delta_alf 14 " + str_names_to_add + "\n"
    header2 = "img"
    for i in range(2):
        label2 = " xy"+str(i+1)+"_0 xy"+str(i+1)+"_1"
        header2 = header2 + label2
    for i in range(ncol) :
        label2 = " matstarlab_" + str(i) 
        header2 = header2 + label2        
    header2 = header2 + " dalf " + str_names_to_add + "\n"  
    outfilename = filepathout + "mat_2spots_" + fileprefix + "img" +\
        str(imgall[0]) + "to" + str(imgall[-1]) + "_alf" + add_str + ".dat"    
    print(outfilename)
    print(header)
    print(header2)

    #nimg = 50
    
    hkl = hkl_2spots
    print(hkl)
    
    Imax_ratio_min = Imax_ratio_2spots.min(axis=1)
    if 0 :
        print("Imax_ratio_2spots =", Imax_ratio_2spots)
        print("Imax_ratio_min = ", Imax_ratio_min)
    ind0 = where( Imax_ratio_min > threshold_on_Imax_ratio )  
    print(shape(ind0))
    
    nimg2 = len(ind0[0])
    
    range1 = ind0[0]
    
    print("found %d images with Imax of 2 spots larger than %.4f" %(nimg2, threshold_on_Imax_ratio))

    if (imgref_for_superimposing_spot_trajectories  is  not None) or (point_num_for_superimposing_spot_trajectories  is  not None) : 
        p.figure()
        xy1_short = xy1[range1,:]
        xy2_short = xy2[range1,:]
        img_list1_short = img_list1[range1]
        img_list2_short = img_list2[range1]
        p.plot(xy1_short[:,0], -xy1_short[:,1], "ro-")
        for i in range(nimg2) : p.text(xy1_short[i,0],-xy1_short[i,1],str(img_list1_short[i]), fontsize = 16)
        p.plot(xy2_short[:,0], -xy2_short[:,1], "bs-")
        for i in range(nimg2) : p.text(xy2_short[i,0],-xy2_short[i,1],str(img_list2_short[i]), fontsize = 16)
        p.xlabel("xpix")
        p.ylabel("ypix")


    if add_last_N_columns_from_filespotmon > 0 :
        n1 = add_last_N_columns_from_filespotmon
        columns_to_add = data_list1[ind0[0],-n1:]
     
    matstarlab_all = zeros((nimg2,ncol), float)
    dalf_all = zeros(nimg2, float)
    
    print("spot nimg xymean xystd xyrange")    
    for i in range(2):           
        print(i, shape(ind0)[1], xy_2spots[ind0[0],2*i:2*i+2].mean(axis=0).round(decimals=2),\
            xy_2spots[ind0[0],2*i:2*i+2].std(axis=0).round(decimals=2), \
            xy_2spots[ind0[0],2*i:2*i+2].max(axis=0)-xy_2spots[ind0[0],2*i:2*i+2].min(axis=0))   

# point d'arret standard
    if test == 1 : jlfsdfs
          
    k = 0
    
    iscubic = test_if_cubic(elem_label)
    for i in ind0[0]:
        #print "k = ", k
        xycam = xy_2spots[i,:].reshape(2,2)        
#        print xycam

        print("img = ", imgall[i])
        if iscubic :  # simplification pour le cas cubique
             matstarlab_all[k,:], dalf_all[k] = two_spots_to_mat(hkl,
                                                    xycam,
                                                    calib,
                                                    pixelsize = pixelsize,
                                                    verbose = verbose)           
        else :
            matstarlab_all[k,:], dalf_all[k] = two_spots_to_mat_gen(hkl,
                                                    xycam,
                                                    calib,
                                                    pixelsize = pixelsize,
                                                    elem_label = elem_label,
                                                    verbose = verbose)
                                                    
        mat1 = matstarlab_to_matstarsample3x3(matstarlab_all[k,:],
                                              omega = omega,
                                              mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)

        if 0 :
            mat2 = mat1.transpose()  # OK seulement pour matrice OND
            print("xyzsample on abcstar :")
            print(mat2.round(decimals = 4))
            print("columns normalized to largest component :")
            str1 = ["HKLx", "HKLy", "HKLz"]
            for i in range(3):
                print(str1[i], (mat2[:,i]/abs(mat2[:,i]).max()).round(decimals = 4))
                                                   
        k = k+1
     
#    print matstarlab_all.round(decimals=ndec)

    toto1 = []
    toto = []    
    
    toto1 = column_stack((imgall[ind0[0]].round(decimals=1),
                            xy_2spots[ind0[0],:].round(decimals=2),
                            matstarlab_all.round(decimals=ndec)))

#    print dalf_all.round(decimals=2)
    if add_last_N_columns_from_filespotmon > 0 :
        toto1 = column_stack((toto1, dalf_all.round(decimals=3), columns_to_add))
    
    print(shape(toto1))
    print(outfilename)
    outputfile = open(outfilename, 'w')
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, toto1, fmt = fmt1)
    outputfile.close()

    return(outfilename)    

    
def RefineUB_from_xycam(
            hkl_spots,
            xyexp_spots,
            intensity_exp_spots,
            numdat_spots,
            calib,
            starting_orientmatrix,
            starting_Bmatrix,  # Bstar pas Bdir
            use_weights=True):
    """

    defaultParam = geometric detector parameters

    UP TO NOW: only strain and orientation simultaneously
    """

    nb_pairs = shape(hkl_spots)[0]

    print("\nStarting fit of strain and orientation from spots links ...\n")

    print("Nb of pairs: ", nb_pairs)

    sim_indices = np.arange(nb_pairs) # for fitting function this must be an arange...

    # initial parameters of calibration ----------------------
    #print "detector parameters", calib
                                    
    pixX, pixY = xyexp_spots[:,0], xyexp_spots[:,1]

    #print "nb_pairs",nb_pairs
    #print "Experimental pixX, pixY",pixX, pixY
    #print "starting_orientmatrix",starting_orientmatrix

    if use_weights:
        weights = intensity_exp_spots
    else:
        weights = None

    results = None
    fitresults = False

    if 1:  # fitting procedure for one or many parameters

        initial_values = np.array([1., 1., 0., 0., 0., 0, .0, 0.])
        allparameters = np.array(calib + [1, 1, 0, 0, 0] + [0, 0, 0])

        nspots = np.arange(nb_pairs)

        arr_indexvaryingparameters = np.arange(5, 13)
        #print "\nInitial error--------------------------------------\n"
        residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(initial_values,
                                        hkl_spots,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        sim_indices,
                                        pixX, pixY,
                                        initrot=starting_orientmatrix,
                                        Bmat=starting_Bmatrix,
                                        pureRotation=0, 
                                        verbose=1, 
                                        pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1] ,
                                        dim = DictLT.dict_CCD[PAR.CCDlabel][0],
                                        weights=weights)
        #print "Initial residues",residues
        #print "---------------------------------------------------\n"

        results = FitO.fit_on_demand_strain(initial_values,
                        hkl_spots,
                        allparameters,
                        FitO.error_function_on_demand_strain,
                        arr_indexvaryingparameters,
                        sim_indices,
                        pixX, pixY,
                        initrot=starting_orientmatrix,
                        Bmat=starting_Bmatrix,
                        pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1] ,
                        dim = DictLT.dict_CCD[PAR.CCDlabel][0],
                        verbose=0,
                        weights=weights)

        #print "\n********************\n       Results of Fit        \n********************"
        #print "results",results

    if results  is  not None:

        fitresults = True

        #print "\nFinal error--------------------------------------\n"
        residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(results,
                                        hkl_spots,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        sim_indices,
                                        pixX, pixY,
                                        initrot=starting_orientmatrix,
                                        Bmat=starting_Bmatrix,
                                        pureRotation=0, verbose=1, 
                                        pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1] ,
                                        dim = DictLT.dict_CCD[PAR.CCDlabel][0],
                                        weights=weights)

        residues_non_weighted = FitO.error_function_on_demand_strain(results,
                                        hkl_spots,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        sim_indices,
                                        pixX, pixY,
                                        initrot=starting_orientmatrix,
                                        Bmat=starting_Bmatrix,
                                        pureRotation=0, verbose=1, 
                                        pixelsize = DictLT.dict_CCD[PAR.CCDlabel][1] ,
                                        dim = DictLT.dict_CCD[PAR.CCDlabel][0],
                                        weights=None) [0]

        #print "Final residues",residues
        #print "---------------------------------------------------\n"
        #print "mean",np.mean(residues)

        #building B mat
        param_strain_sol = results
        varyingstrain = np.array([[1., param_strain_sol[2], param_strain_sol[3]], [0, param_strain_sol[0], param_strain_sol[4]], [0, 0, param_strain_sol[1]]])
        #print "varyingstrain results"
        #print varyingstrain

        # building UBmat (= newmatrix)
        UBmat = np.dot(np.dot(deltamat, starting_orientmatrix), varyingstrain)
        #print "UBmat",UBmat
        # print "newmatrix",newmatrix # must be equal to UBmat

        RR, PP = GT.UBdecomposition_RRPP(UBmat)

        # symetric matrix (strain) in direct real distance
        epsil = GT.epsline_to_epsmat(CP.calc_epsp(CP.dlat_to_rlat(CP.matrix_to_rlat(PP))))
        #print "epsil is already a zero trace symetric matrix ",np.round(epsil*1000, decimals = 2)
        # if the trace is not zero
        epsil = epsil - np.trace(epsil) * np.eye(3) / 3.

        epsil = np.round(epsil * 1000, decimals=2)
        print("\n********************\n       result of Strain and Orientation  Fitting        \n********************")
        #print "last pixdev table non weighted"
        #print residues_non_weighted.round(decimals=3)
        print("Mean pixdev non weighted")
        pixdev = np.mean(residues_non_weighted).round(decimals=5)
        print(pixdev)
        print("\nPure Orientation Matrix")
        #print RR.tolist()
        print(RR.round(decimals=5))


        print("Deviatoric strain with respect to initial crystal unit cell (in 10-3 units)")
        #print epsil.tolist()
        print(epsil)

        #pas = np.array([[0,-1, 0],[0, 0, -1],[1, 0, 0]])
        #invpas = np.linalg.inv(pas)
        #print "good deviatoric strain epsil"  # ???? in other frame like XMAS
        #print np.dot(pas, epsil) # ????

        # update linked spots with residues

        linkResidues_fit = np.array([numdat_spots, sim_indices, residues]).T

        # for further use
        newUBmat = UBmat
#        newUmat = np.dot(deltamat, starting_orientmatrix)
#        newBmat = varyingstrain
        deviatoricstrain = epsil

        return newUBmat, deviatoricstrain, linkResidues_fit, pixdev
    else :
        return(0)
        

def find_single_and_common_spots( filefitmg1,
                                     filefitmg2,
                                     gnumlocref1,
                                     gnumlocref2,
                                     dxytol,
                                     imgref1,
                                     imgref2,
                                     gnum1,
                                     gnum2,
                                     save_hkl_lists = "yes",
                                     filepathout = None,
                                     verbose = 1,
                                     plot_spots = "all"):
                                     
#        print gnum1, gnum2, imgref1, imgref2

        gnumlist1, npeaks1, indstart1, matstarlab1, data_fit1, calib1, pixdev1 = readlt_fit_mg(filefitmg1)
       
        gnumlist2, npeaks2, indstart2, matstarlab2, data_fit2, calib2, pixdev2 = readlt_fit_mg(filefitmg2) 
 
        if len(indstart1) > 1:
            range1 = np.arange(indstart1[gnumlocref1], indstart1[gnumlocref1] + npeaks1[gnumlocref1])
        else :
            range1 = np.arange(0, npeaks1)
        if len(indstart2) > 1:
            range2 = np.arange(indstart2[gnumlocref2], indstart2[gnumlocref2] + npeaks2[gnumlocref2])
        else :
            range2 = np.arange(0, npeaks2)
            
        data_fit1b = np.array(data_fit1[range1, :],dtype = float)
        data_fit2b = np.array(data_fit2[range2, :],dtype = float)
        
        hkl1 = np.array(data_fit1b[:,2:5], dtype = int)
        hkl2 = np.array(data_fit2b[:,2:5], dtype = int)
            
        xy1 = data_fit1b[:, 6:8]
        xy2 = data_fit2b[:, 6:8] 
        
        Etheor1 = data_fit1b[:, -1]
        Etheor2 = data_fit2b[:, -1]
        
        nb_common_peaks, list_pairs, nsingle1, nsingle2, list_single1, list_single2, iscommon1, iscommon2 = \
                find_common_peaks(xy1, xy2, dxytol=dxytol, verbose = 0, returnmore = 1)

        print(filefitmg1)        
        print(filefitmg2)        
        print("gnumlocref1, gnumlocref2 = ", gnumlocref1, gnumlocref2)
        print("npeaks1, npeaks2 = ", npeaks1[gnumlocref1], npeaks2[gnumlocref2])
        print("nb_common_peaks =" , nb_common_peaks)
        #print "list pairs = \n", list_pairs
        print(np.shape(list_pairs))
        print("nsingle1 , nsingle2 = ", nsingle1 , nsingle2)
        
        if verbose :
            print("pairs")
            print("i1 hkl1 xy1 Etheor1 i2 hkl2 xy2") # Etheor2"
            for i in range(nb_common_peaks):
                i1 = list_pairs[i,0]
                i2 = list_pairs[i,1]
                print(i1, hkl1[i1], xy1[i1].round(decimals=2), round(data_fit1b[i1,-1],1), \
                    "\t" , i2, hkl2[i2], xy2[i2].round(decimals=2)) #, round(data_fit2b[i2,-1],1)
                    
            print("singles 1")
            print("i1 hkl1 xy1 Etheor1")
            for i in range(nsingle1):
                i1 = list_single1[i]
                print(i1, hkl1[i1], xy1[i1].round(decimals=2), round(data_fit1b[i1,-1],1))

            print("singles 2")
            print("i2 hkl2 xy2 Etheor2")
            for i in range(nsingle2):
                i2 = list_single2[i]
                print(i2, hkl2[i2], xy2[i2].round(decimals=2), round(data_fit2b[i2,-1],1))

        if plot_spots  is  not None :
            
            if plot_spots == "all" :
                plot_diagr2(xy1, plotlabel = None )
                plot_diagr2(xy2, plotmode = "overlay", plotlabel = None, xs = "o" )
                
            elif plot_spots == "singles" :
                plot_diagr2(xy1[list_single1],  plotlabel = 1, add_label = list_single1  )
                plot_diagr2(xy2[list_single2], plotmode = "overlay", plotlabel = 1, add_label = list_single2 , xs = "o" )

            elif plot_spots == "pairs" :
                plot_diagr2(xy1[list_pairs[:,0]],  plotlabel = 1, add_label = list_pairs[:,0]  )
                
#                plot_diagr2(xy2[list_single2], plotmode = "overlay", plotlabel = 1, add_label = list_single2 , xs = "o" )
       
        print(shape(list_pairs)[0], len(list_single1), len(list_single2))   
        
        if save_hkl_lists == "yes" :
            hkl1_multi = hkl1[list_pairs[:,0]]
            filename = filepathout + "hklcommon_G"+  str(gnum1) \
                        + "_G"+  str(gnum2) + "_img" + str(imgref1) + "_img" + str(imgref2) +".txt"  
            print("list of hkl1's of common spots saved in : \n", filename)
            np.savetxt(filename, hkl1_multi, fmt = "%d")
            
            hkl1_single1 = hkl1[list_single1]
            filename = filepathout + "hklsingle_G" + str(gnum1) + "_img" + str(imgref1) + ".txt"   
            print("list of hkl1's of single spots 1 saved in : \n", filename)
            np.savetxt(filename, hkl1_single1, fmt = "%d")
            hkl2_single2 = hkl2[list_single2]
            filename = filepathout + "hklsingle_G" + str(gnum2) + "_img" + str(imgref2) + ".txt"  
            print("list of hkl2 of single spots 2 saved in : \n", filename)
            np.savetxt(filename, hkl2_single2, fmt = "%d")                                   
        
        return(hkl1, hkl2, xy1, xy2, list_pairs, list_single1, list_single2, Etheor1, Etheor2)
        
def refine_again_one_grain(filepathfit_out, 
                           filepathfit_in,
                           fileprefix, 
                           filesuffix_in,
                           fileindimg, 
                           filehkl = None, 
                           all_spots = "no",
                           test = "no",
                           imgtest = None,
                           CCDlabel = PAR.CCDlabel):
                               
#    latticeparam =np.array(DictLT.dict_Materials[PAR.elem_label_index_refine][1][0], dtype = float)
    #print latticeparam
#    dlatu = np.array([latticeparam, latticeparam, latticeparam, PI / 2.0, PI / 2.0, PI / 2.0])
                       
        
    if all_spots == "no" :
        hkl_list = loadtxt(filehkl)
        hkl_list = np.array(hkl_list, dtype = int)    
        print("max number of spots = ", shape(hkl_list)[0])
    
    indimg = loadtxt(fileindimg)
    indimg = np.array(indimg, dtype = int)

    # test 
    if test == "yes" :
        ind1 = where(indimg == imgtest)
        indimg = indimg[ind1[0]:ind1[0]+5]
    
    for kk in indimg :
#        print "img = ", kk
        filefitmg_nopath = fileprefix + imgnum_to_str(kk,PAR.number_of_digits_in_image_name)+ filesuffix_in
        filefitmg = filepathfit_in + filefitmg_nopath
        #print filefitmg
        if (filefitmg_nopath in os.listdir(filepathfit_in)) :
            
            res1 = readlt_fit_mg(filefitmg, verbose = 0, readmore = True)
            #print res1

            if res1 != 0 :
                gnumlist, npeaks, indstart, matstarlab, data_fit, calib, pixdev, strain6, euler = res1
#                print npeaks[0], pixdev

                if all_spots == "no" :
                    hkl1 = np.array(data_fit[:, 2:5], dtype=float)
                    hkl2 = np.array(hkl_list, dtype = float)
            
                    nb_common_peaks, list_pairs, nsingle1, nsingle2, list_single1, list_single2, iscommon1, iscommon2 = \
                            find_common_peaks(hkl1, hkl2, dxytol=0.1, verbose = 0, returnmore = 1) 
                    print(nb_common_peaks)
                
                    ind1 = list_pairs[:,0] # avec seulement spots HKL de la liste

                    condition1 = eval("nb_common_peaks > 7")
                    
                else :
                    ind1 = np.arange(npeaks[0]) # test avec tous les spots, mettre if 1: au lieu de if nb_common_peaks > 7
                    condition1 = 1    

                nfit = len(ind1)
                
                if condition1 :
                    
                    hkl_spots = hkl1[ind1,:]
                    xyexp_spots = np.array(data_fit[ind1, 6:8], dtype=float)
                    intensity_exp_spots = np.array(data_fit[ind1, 1], dtype=float) 
                    numdat_spots = np.array(data_fit[ind1, 0], dtype=int)
                    #print shape(matstarlab[0])
                    matstarlabOND = matstarlab_to_matstarlabOND(matstarlab[0])
                    starting_orientmatrix = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlabOND)
#                    starting_Bmatrix = MG.rlat_to_Bstar(MG.mat_to_rlat(matstarlab[0])) # non
 
#                    starting_Bmatrix = np.eye(3, dtype = float)
                    dlatu_angstroms_deg = DictLT.dict_Materials[elem_label][1]                      
                    dlatu_nm_rad = deg_to_rad_angstroms_to_nm(dlatu_angstroms_deg)
                    starting_Bmatrix = CP.calc_B_RR(dlatu_angstroms_deg)    
                       
                    if 0 :
                        nmax = min(nb_common_peaks, 10)
                        for i in range(nmax):
                            print(numdat_spots[i], hkl_spots[i,:], xyexp_spots[i,:], intensity_exp_spots[i]) 
                        print(Bmatrix)
                        print(starting_orientmatrix)
                        
                    newUBmat, deviatoricstrain, spotnum_i_pixdev_2, pixdev_2 = \
                        RefineUB_from_xycam(
                                        hkl_spots,
                                        xyexp_spots,
                                        intensity_exp_spots,
                                        numdat_spots,
                                        list(calib[0]),
                                        starting_orientmatrix,
                                        starting_Bmatrix,
                                        use_weights=False)
                                        
                    pixdev_list = np.array(spotnum_i_pixdev_2, dtype = float)[:, 2]
                                        
                    # calcul Etheor : cubique uniquement
                    Etheor = zeros(nfit,float)
                    uilab = array([0.,1.,0.])
                    matstarlab = F2TC.matstarlabLaueTools_to_matstarlabOR(newUBmat)
                    matwithlatpar_inv_nm = F2TC.matstarlab_to_matwithlatpar(matstarlab, dlatu_nm_rad)
                    for i in range(nfit):
                        Etheor_eV, ththeor_deg, uqlab = mat_and_hkl_to_Etheor_ththeor_uqlab(matwithlatpar_inv_nm =matwithlatpar_inv_nm,
                                                                                         hkl = hkl_spots[i,:])                        
                        
#                        qlab = hkl_spots[i,0]*mat[0:3]+ hkl_spots[i,1]*mat[3:6]+ hkl_spots[i,2]*mat[6:]
#                        uqlab = qlab/norme(qlab)
#                        sintheta = -inner(uqlab,uilab)
#                        if (sintheta > 0.0) :
                            #print "reachable reflection"
#                            Etheor[i] = DictLT.E_eV_fois_lambda_nm*norme(qlab)/(2.*sintheta)*10.  
                        Etheor[i] = Etheor_eV
    
    # verifie Jul13 que le calcul JSM donne bien la meme chose que mon recalcul                                              
    #                epsp1, dlatsrdeg1 = MG.matstarlab_to_deviatoric_strain_crystal(GT.mat3x3_to_matline(newUBmat), 
    #                                            version = 2, 
    #                                            elem_label = PAR.elem_label_index_refine)
    #                print "deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)"
    #                print epsp1.round(decimals=2)
    #                deviatoricstrain = GT.epsline_to_epsmat(epsp1).round(decimals=2)
    #                print "deviatoric strain matrix \n", deviatoricstrain
                        
                    data_fit_new = column_stack((data_fit[ind1,:5], pixdev_list, data_fit[ind1,6:8], Etheor))
                
                    filesuffix_out = "_UWN"
                    grainnum = 1
    
                    filefit = save_fit_results(filefitmg, data_fit_new , 
                                                  newUBmat, deviatoricstrain, filesuffix_out,\
                                                   pixdev_2, grainnum, calib[0], CCDlabel,
                                filepathout = filepathfit_out, fileprefix = fileprefix, imgnum = kk)                     
                    print("img = ", kk)
                    print("npeaks start/end = ", npeaks[0], nfit)
                    print("pixdev start/end = ", pixdev[0], pixdev_2)

                else :
                    print("not enough spots left in img ", kk)
    
    return(0)
    
def serial_peak_search_monograin(filepathim, 
                                 fileprefix, 
                                 indimg, 
                                 filesuffix, 
                                 filepathout,
                                 filedatref,
                                 guess_from_previous_image = 1) :

    print("peak search in series of images (or single image)")
    
    npeaks = zeros(len(indimg), int)

    commentaire = "LT rev " + PAR.LT_REV_peak_search + "\n# PixelNearRadius = " + str(PAR.PixelNearRadius) + \
            "\n# IntensityThreshold = " + str(PAR.IntensityThreshold) + \
            "\n# boxsize = " + str(PAR.boxsize) + "\n# position_definition = " + str(PAR.position_definition) + \
            "\n# fit_peaks_gaussian = " + str(PAR.fit_peaks_gaussian) + \
            "\n# xtol = " + str(PAR.xtol) + "\n# FitPixelDev = " + str(PAR.FitPixelDev) + \
            "\n# local_maxima_search_method = " + str(PAR.local_maxima_search_method) + \
            "\n# Threshold Convolve = " + str(PAR.thresholdConvolve) + "\n"
        
    print("peak search parameters :")
    print(commentaire)
    
    k = 0
    for i in indimg :
        print("i = ", i)
        #filename = filelist[i]
        filename = filepathim + fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name) + filesuffix
        print("image in :")
        print(filename)
        print("saving peak list in :")
        fileprefix1 = filepathout + fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name)
        filedat = fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name) + ".dat"
        #print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_peak_search :
            while (filedat in os.listdir(filepathout)):
                print("warning : change name to avoid overwrite")
                fileprefix2 = fileprefix + imgnum_to_str(i, PAR.number_of_digits_in_image_name) + "_new_" + str(j)
                filedat = fileprefix2 + ".dat"
                print(filepathout + filedat)
                j = j + 1

        if j > 0 :
            fileprefix1 = filepathout + fileprefix2
        else :
            print(filepathout + filedat)
            
        Isorted, fitpeak, localpeak = rmccd.PeakSearch(filename,
                                                    CCDLabel=PAR.CCDlabel,
                                                    PixelNearRadius=PAR.PixelNearRadius ,
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
                                                    Saturation_value_flatpeak=DictLT.dict_CCD[PAR.CCDlabel][2]
                                                    )
        npeaks[k] = shape(Isorted)[0]

        if shape(Isorted)[0] > 0 :
            
            RWASCII.writefile_Peaklist(fileprefix1,
                                    Isorted,
                                    overwrite=1,
                                    initialfilename=filename,
                                    comments=commentaire)
        k = k + 1

    print("indimg ", indimg)
    print("npeaks ", npeaks)
    return(0)

def plot_spot_displacement_field(xy, dxy, arrow_scale = 100.,
        xmin0=0., xmax0=2048., ymin0 = 0., ymax0 = 2048.):
    
    p.rcParams['lines.markersize'] = 2
#    p.rcParams['savefig.bbox'] = None    
    p.figure(num=1, figsize=(8,8))
    p.plot(xy[:,0],-xy[:,1],'bo')
    p.quiver(xy[:,0],-xy[:,1],dxy[:,0]*arrow_scale,-dxy[:,1]*arrow_scale, \
             units = "xy", angles = "xy", scale_units = 'xy', scale = 1,\
             width = 10, color = "b")
    p.xlabel('xcamXMAS - toward back')
    p.ylabel('-ycamXMAS - upstream')
    p.xlim(xmin0, xmax0)
    p.ylim(-ymax0, -ymin0)    
    
    return(0)

def xyz_sample_to_hkl(matstarlab, 
                      xyz_sample, 
                      omega = None, # was PAR.omega_sample_frame
                      mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame):

        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarsample3x3 = matstarlab_to_matstarsample3x3(matstarlabOND, 
                                                          omega = omega,
                                                          mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)
        
        udir = xyz_sample/norme(xyz_sample)
        udircr = dot(matstarsample3x3.transpose(),udir)
        return(udircr)

def hkl_to_xyz_sample(matstarlab, hkl):

        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarsample3x3 = matstarlab_to_matstarsample3x3(matstarlabOND, 
                                                          omega = None, # was PAR.omega_sample_frame,
                                                          mat_from_lab_to_sample_frame = PAR.mat_from_lab_to_sample_frame)
        
        udircr = hkl/norme(hkl)
        udir = dot(matstarsample3x3,udircr)
        return(udir)

def calc_all_Schmid(matstarlab, 
                    load_axis_xyzsample,   # direction d'application de la force ?
                    hkl_spot = None, 
                    calib = None, 
                    plot_elong_dir = None, 
                    load_axis_hkl = None):
                        
        # force uniaxiale donnee dans le repere sample 
        # orientation cristal donnee par matstarlab

        nop = 24

        allop = DictLT.OpSymArray
        indgoodop = array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])
        goodop = allop[indgoodop]
        
        if nop == 48 :
                goodop = allop*1.0

        # elongation of hkl spot

        if hkl_spot  is  not None :
                uq_spot = hkl_spot / norme(hkl_spot)
        else :
                uz0 = array([0.,0.,1.])
                uq_spot = xyz_sample_to_hkl(matstarlab, uz0)
                uz_spot = uq_spot/norme(uq_spot)
                
        # load_axis_xyzsample convention OR
        # glide plane, glide dir
        hkl_2 = array([[1.,1.,1.],[1.,-1.,0.]])
        
        matstarlabOND = matstarlab_to_matstarlabOND(matstarlab)
        matstarsample3x3 = matstarlab_to_matstarsample3x3(matstarlabOND)
        matstarlab3x3 = GT.matline_to_mat3x3(matstarlab)
        
        # udircr = coord cristal de direction d'application de la force
        if load_axis_hkl  is None :
                udir = load_axis_xyzsample / norme(load_axis_xyzsample)
                udircr = dot(matstarsample3x3.transpose(),udir)
        else :
                udircr = load_axis_hkl/norme(load_axis_hkl)

        normehkl = zeros(2,float)
        
        uqref = zeros((2,3), float)
        for i in range(2):
                normehkl[i] = norme(hkl_2[i,:])
                uqref [i,:] = hkl_2[i,:] / normehkl[i]
                
        uqall = zeros((2,nop,3),float)
        for j in range(2):
                for k in range(nop):
                        uqall[j,k,:] = dot(goodop[k],uqref[j,:])

        uq112 = zeros((nop,3),float)
        sinalf = zeros(nop,float)
        dq_axis = zeros((nop,3),float)
        
        for k in range(nop):
                
                uq112[k,:] = cross(uqall[0,k,:],uqall[1,k,:])
                cosalf = inner(uq112[k,:],uq_spot)
                #print cosalf
                sinalf[k] = np.sqrt(1.0 - cosalf*cosalf)
                #print sinalf[k]
                dq_axis[k,:] = cross(uq112[k,:],uq_spot)
                #print dq_axis[k,:]
                if dq_axis[k,0] != 0.0 :
                        dq_axis[k,:] = dq_axis[k,:]/dq_axis[k,0]
                elif dq_axis[k,1] != 0.0 :
                        dq_axis[k,:] = dq_axis[k,:]/dq_axis[k,1]
                elif dq_axis[k,2] != 0.0 :
                        dq_axis[k,:] = dq_axis[k,:]/dq_axis[k,2]

        #print dq_axis

        tilt_dysdx = zeros(nop, float)
        dxy = zeros(nop, float)
        dxycam =zeros((nop,2), float)

        # probe volume
        s_lab = array([1.,7.,1.])

        if calib  is  not None :
                RAD = 180.0/math.pi
                mat1 = matstarlab
                uilab = array([0.0,1.0,0.0])
                uq = uq_spot
                qlab = uq[0]*mat1[0:3]+ uq[1]*mat1[3:6]+ uq[2]*mat1[6:]
                uqlab = qlab/norme(qlab)
                sintheta = - inner(uqlab,uilab)
                if (sintheta > 0.0) :
                    xycam0 = uqlab_to_xycam(uqlab,calib)
                else :
                    print("reflection not accessible")
                for k in range(nop):
                        if norme(dq_axis[k,:])!= 0.0 :
                                uq = uq_spot + 0.01 * dq_axis[k,:]/norme(dq_axis[k,:])
                                qlab = uq[0]*mat1[0:3]+ uq[1]*mat1[3:6]+ uq[2]*mat1[6:]
                                uqlab = qlab/norme(qlab)
                                xycam1 = uqlab_to_xycam(uqlab,calib)
                                dxycam[k,:] = xycam1 - xycam0
                                dxy[k] = norme(dxycam[k,:])
                                if dxycam[k,0] !=0.0 :
                                        tilt_dysdx[k] = -np.arctan(dxycam[k,1]/dxycam[k,0])*RAD
                                else :
                                        tilt_dysdx[k] = 90.0
                                       
        uq112 = uq112 * np.sqrt(6.0)
                
        schmid = zeros(nop, float)
        ubs = zeros(nop,float)
        opsym = list(range(nop))

        for k in range(nop):
                schmid[k] = abs(inner(udircr,uqall[0,k,:])*inner(udircr,uqall[1,k,:]))
                ub = uqall[1,k,:]
                #print ub.round(decimals=2)
                ublab = dot(matstarlab3x3,ub)
                #print ublab.round(decimals=2)             
                ubs[k] = np.sqrt((s_lab[0]*ublab[0])**2 + (s_lab[1]*ublab[1])**2 + (s_lab[2]*ublab[2])**2)
                #print round(ubs[k],3)

        print("opsym 0, HKLplane 1:4, HKLdir 4:7, HKL112 7:10")
        print("dq_axis 10:13, dxy : 13, tilt_dysdx : 14, dxycam 15:17, sinalf -2, schmid -1")
                
        data1 = column_stack((opsym,uqall[0,:,:]*normehkl[0], uqall[1,:,:]*normehkl[1], uq112, dq_axis, dxy, tilt_dysdx, dxycam, ubs, sinalf, schmid))
        print(shape(data1))
                             
        hkl2_schmid_sorted = sort_list_decreasing_column(data1, -1)
        
        print("matstarlab", matstarlab.round(decimals=4))
        print("HKL spot", hkl_spot)
        print("load_axis_xyzsample", load_axis_xyzsample)
        print("opsym HKLplane HKLdir HKL112 dq_axis sinalf Schmid_factor")
        for i in range(nop):
            print(int(hkl2_schmid_sorted[i,0].round(decimals=2)), hkl2_schmid_sorted[i,1:4].round(decimals=2),\
                hkl2_schmid_sorted[i,4:7].round(decimals=2),hkl2_schmid_sorted[i,7:10].round(decimals=2),\
                hkl2_schmid_sorted[i,10:13].round(decimals=2),\
                hkl2_schmid_sorted[i,-2].round(decimals=3), hkl2_schmid_sorted[i,-1].round(decimals=3))

##        toto = column_stack((hkl2_schmid_sorted[:,10],hkl2_schmid_sorted[:,10]))
##        print toto
##        nb_common_peaks,iscommon1,iscommon2 = find_common_peaks(toto, toto, dxytol = 0.0001, verbose = 1)
##        print iscommon1
##        print iscommon2

        ind0 = list(range(12))
        ind0 = np.array(ind0, dtype=float)*2.0
        ind0 = np.array(ind0, dtype = int)
        #print ind0

        data_schmid = hkl2_schmid_sorted[ind0]
        
        print("matstarlab", matstarlab.round(decimals=4))
        print("HKL spot", hkl_spot)
        print("load_axis_xyzsample", load_axis_xyzsample)
        print("opsym HKLplane HKLdir HKL112 dq_axis sinalf Schmid_factor")
        for i in range(12):
            print(i+1, int(data_schmid[i,0].round(decimals=2)), data_schmid[i,1:4].round(decimals=2),\
                data_schmid[i,4:7].round(decimals=2),data_schmid[i,7:10].round(decimals=2),\
                data_schmid[i,10:13].round(decimals=2),\
                data_schmid[i,-2].round(decimals=3), data_schmid[i,-1].round(decimals=3))
        
        bcom = zeros(12)
        for k in range(12):
                if abs(inner(data_schmid[0,4:7],data_schmid[k,4:7])/2.0) == 1.0 :
                       bcom[k] = 1
        print("Schmid factors")
        print(data_schmid[:,-1].round(decimals=3))
        print("FWHM factors")
        print(data_schmid[:,-2].round(decimals=3))
        #print "opsym 0, HKLplane 1:4, HKLdir 4:7, HKL112 7:10"
        #print "dq_axis 10:13, dxy : 13, tilt_dysdx : 14, dxycam 15:17, ubs : 17, sinalf -2, schmid -1"
        print("HKL spot", hkl_spot)
        print("tilt = (-dycam/dxcam)(deg)")
        print("dxy for dq/q = 0.01")
        print("bcom = 1 if b common with system 1")
        print("ubs = sqrt((s_lab[0]*ublab[0])**2 + (s_lab[1]*ublab[1])**2 + (s_lab[2]*ublab[2])**2)")
        print("for s_lab = [1,7,1]")
##        print "system  sinalf  dxy     tilt      ubs Schmid_factor dq_axis bcom"
##        for i in range(12):
##                print i+1, "\t", data_schmid[i,-2].round(decimals=2),"\t", data_schmid[i,13].round(decimals=1),\
##                "\t", data_schmid[i,14].round(decimals=1),"\t", data_schmid[i,17].round(decimals=1),\
##                "\t", data_schmid[i,-1].round(decimals=3),"\t", data_schmid[i,10:13].round(decimals=2),\
##                int(bcom[i].round(decimals=0))
                
        print("system  sinalf  dxy     tilt      ubs Schmid_factor HKL112 bcom")
        for i in range(12):
                print(i+1, "\t", data_schmid[i,-2].round(decimals=2),"\t", data_schmid[i,13].round(decimals=1),\
                "\t", data_schmid[i,14].round(decimals=1),"\t", data_schmid[i,17].round(decimals=1),\
                "\t", data_schmid[i,-1].round(decimals=3),"\t", data_schmid[i,7:10].round(decimals=2),\
                int(bcom[i].round(decimals=0)))
                
        print("udircr = ", udircr.round(decimals=3))


        # plot elong also for axes z1 = b1^x and x

        if plot_elong_dir == 2 :
                nelong = 4
                x_axis = array([1.,0.,0.])
                hkl = xyz_sample_to_hkl(matstarlab, x_axis)
                uqx = hkl / norme(hkl)
                print("uqx = ", uqx.round(decimals=4))
                hkl_b1 = data_schmid[0,4:7]
                print("hkl_b1 = ", hkl_b1.round(decimals=2))
                uz1b = cross(uqx,hkl_b1)
                uz1b = uz1b / norme(uz1b)
                print("uz1b = ", uz1b.round(decimals=4))
                hkl_n1 = data_schmid[0,1:4]
                print("hkl_n1 = ", hkl_n1.round(decimals=2))
                un1 = hkl_n1 / norme(hkl_n1)
                uz1n = cross(uqx,hkl_n1)
                uz1n = uz1n / norme(uz1n)
                print("uz1n = ", uz1n.round(decimals=4))
##                uy1 = cross(uqx,uz1b)
##                uy1 = uy1 / norme(uy1)
##                print "uy1 = ", uy1.round(decimals=2)
                #urot = array([  1.,   -17.79,  11.76])
                #urot = array( [ 1.,    9.34, -9.03])
                #urot = array( [ 1.,    9.34, -9.03])
                urot = array([ 1. ,  -0.06, -0.68]) # 768-846
                #urot = array([ 1.,    0.11, -0.98]) # 768-794
                urot = urot/norme(urot)
                
                dq_axis2 = zeros((nelong,3),float)
                dq_axis2[0,:] = cross(uqx,uq_spot)
                dq_axis2[1,:] = cross(uz1b,uq_spot)
                dq_axis2[2,:] = cross(uz1n,uq_spot)
                dq_axis2[3,:] = cross(un1,uq_spot)
                #dq_axis2[2,:] = cross(uy1,uq_spot)
                #dq_axis2[3,:] = cross(urot,uq_spot)
                dxycam2 = zeros((nelong,2),float)
                tilt_dysdx2 = zeros(nelong,float)

                if calib  is  not None :
                        RAD = 180.0/math.pi
                        mat1 = matstarlab
                        uilab = array([0.0,1.0,0.0])
                        uq = uq_spot
                        qlab = uq[0]*mat1[0:3]+ uq[1]*mat1[3:6]+ uq[2]*mat1[6:]
                        uqlab = qlab/norme(qlab)
                        sintheta = - inner(uqlab,uilab)
                        if (sintheta > 0.0) :
                            xycam0 = uqlab_to_xycam(uqlab,calib)
                        for k in range(nelong):
                                uq = uq_spot + 0.01 * dq_axis2[k,:]/norme(dq_axis2[k,:])
                                qlab = uq[0]*mat1[0:3]+ uq[1]*mat1[3:6]+ uq[2]*mat1[6:]
                                uqlab = qlab/norme(qlab)
                                xycam1 = uqlab_to_xycam(uqlab,calib)
                                dxycam2[k,:] = xycam1 - xycam0
                                if dxycam2[k,0] !=0.0 :
                                        tilt_dysdx2[k] = -np.arctan(dxycam2[k,1]/dxycam2[k,0])*RAD
                                else :
                                        tilt_dysdx2[k] = 90.0
                print("elongation for other rotations : axes : x, z1b, z1n")
                print("dxycam")
                print(dxycam2.round(decimals = 2))
                print("tilt = (-dycam/dxcam)(deg)")
                print(tilt_dysdx2.round(decimals = 2))
                
        if plot_elong_dir  is  not None :
                
                #ind1 = [7,9,11]   # pour 0 4 -2 img 768
                #ind1 = [7,8,11]   # pour 0 4 -2 img 846
                #ind1 = [7,10,11]   # pour 0 4 -2 img 236
                #ind1 = [4,6,7,10,11,12]  # pour 0 4 0 base img 768
                #ind1 = [3,4,7,10,11,12]  # pour 0 4 0 centre img 846
                #ind1 = [3,5,7,9,11,12]  # pour 0 4 0 centre img 236
                #ind1 = [5,6,7,8,12]  # pour 0 2 -2 centre img 236
                #ind2 = [9,10]  # pour 0 2 -2 centre img 236
                #ind1 = [5,6,7,8,12]  # pour 0 2 -2 centre img 236
                #ind2 = [1,3,9,10]  # pour 0 2 -2 centre img 236
                #ind1 = [3,5,6,8,9,11] # GD 1262
                ind2 = [5,12] # GD 1262
                multf = ones(12,float)
                multf = multf * 1.2
                #ind1 = [3,6,11,12] # GD 2673
                #ind2 = [5]
                ind1 = [4,11,12] # GD 4873
                ind2 = []
                p.figure(figsize=(9,9))
                pix_scale = 1.0
                xx = zeros(2,float)
                yy = zeros(2,float)
                p.axhline(y=0, color = 'k')
                p.axvline(x=0, color = 'k')
                for i in range(12):
                        xx[1] = data_schmid[i,15]
                        yy[1] = -data_schmid[i,16]
                        p.rcParams['lines.markersize'] = 10
                        p.plot(xx,yy,'bo-')
                        if i+1 in ind1 : multf[i] = 1.5
                        if i+1 in ind2 : multf[i] = 1.9
                        p.rcParams['lines.markersize'] = 25
                        p.plot(xx[1]*multf[i],yy[1]*multf[i],"wo")
                        p.text(xx[1]*multf[i],yy[1]*multf[i], str(i+1),fontsize = 20,horizontalalignment='center', verticalalignment='center')

                if plot_elong_dir == 2 :
                        #xz1_label = ["x","z1","y1","v"]
                        xz1_label = ["x","z1b","z1n","n1"]
                        #ind3 = [] # 0 4 0 768
                        #ind3, ind2 = [],[3,] # 0 4 0 846
                        #multf = [1.2, 1.25, 1.25, 1.5] # 0 4 0 846
                        multf = [1.2, 1.25, 1.6, 1.25]
                        #ind3, ind2 = [2,],[] # 0 4 -2 768
                        #ind3, ind2 = [], [2,] # 0 4 -2 846
                        for i in range(nelong):
                                xx[1] = dxycam2[i,0]
                                yy[1] = -dxycam2[i,1]
                                #multf = 1.25
                                if ((i == 1)|(i==2)) & (xx[1]>0.0) :
                                    xx[1] = -xx[1]
                                    yy[1] = -yy[1]
                                if (i == 3) & (xx[1]<0.0) :
                                    xx[1] = -xx[1]
                                    yy[1] = -yy[1]
                                    
##                                if i in ind2 :
##                                    multf = 1.4 # 0 4 -2 768  
##                                    #multf = 1.5 # 0 4 -2 846 et 0 4 0 846
##                                if i in ind3 :
##                                        multf = 1.7 # 0 4 -2 768
##                                        #multf = 1.7
##                                        #multf = 1.4 # fig01
##                                        #multf = 1.3 # fig11
                                p.rcParams['lines.markersize'] = 10
                                p.plot(xx,yy,'ro-')
                                p.rcParams['lines.markersize'] = 35
                                p.plot(xx[1]*multf[i],yy[1]*multf[i],"wo")
                                p.text(xx[1]*multf[i],yy[1]*multf[i],xz1_label[i],fontsize = 20,horizontalalignment='center', verticalalignment='center')                                

                p.plot([-10.0,0.0],[10.0,10.0],"k-")
                p.plot([-10.0,-10.001],[0.0,10.0],"k-")
                p.xlim(-30.,30.) # 0 4 -2
                p.ylim(-30.,30.)
                #p.xlim(-40.,40.) # 0 4 0
                #p.ylim(-40.,40.)


        return(data_schmid)

def plot_elong_theor(filefit, 
                     hkl_rot_axis = None, 
                     hkl_ctr_axis = None, 
                     label2 = None):

        xyz_sample_pole = array([1.,0.,0.])
        matstarlab1, data_fit, calib, pixdev = F2TC.readlt_fit(filefit, readmore = True)
        matstarlabnew1, transfmat = calc_matrix_for_stereo2(matstarlab1, xyz_sample_pole)

        npics = shape(data_fit)[0]
        data_xyexp = data_fit[:,-2:]
        data_hkl = data_fit[:,2:5]
        hklnew = dot(transfmat,data_hkl.transpose()).transpose()
        # faut changer aussi les HKL

        dxycam = zeros((npics,2),float)
        xy2 = zeros((npics,2),float)

        fmult = 2.0

        if hkl_rot_axis  is  not None :
                uq_rot = hkl_rot_axis/norme(hkl_rot_axis)
        if hkl_ctr_axis  is  not None :
                uq_ctr = hkl_ctr_axis/norme(hkl_ctr_axis)
                dq_axis = uq_ctr
                
        RAD = 180.0/math.pi
        mat1 = matstarlabnew1
        uilab = array([0.0,1.0,0.0])
        for i in range(npics) :
                uq_spot = hklnew[i,:]/norme(hklnew[i,:])
                qlab = uq_spot[0]*mat1[0:3]+ uq_spot[1]*mat1[3:6]+ uq_spot[2]*mat1[6:]
                uqlab = qlab/norme(qlab)
                sintheta = - inner(uqlab,uilab)
                if (sintheta > 0.0) :
                    xycam0 = uqlab_to_xycam(uqlab,calib)
                else :
                    print("reflection not accessible")
                    
                if hkl_rot_axis  is  not None :
                        dq_axis = cross(uq_rot,uq_spot)
                        
                if norme(dq_axis)!= 0.0 :
                        uq = uq_spot + 0.01 * dq_axis/norme(dq_axis)
                        qlab = uq[0]*mat1[0:3]+ uq[1]*mat1[3:6]+ uq[2]*mat1[6:]
                        uqlab = qlab/norme(qlab)
                        xycam1 = uqlab_to_xycam(uqlab,calib)
                        dxycam[i,:] = xycam1 - xycam0
##                        dxy[k] = norme(dxycam[k,:])
##                        if dxycam[k,0] !=0.0 :
##                                tilt_dysdx[k] = -arctan(dxycam[k,1]/dxycam[k,0])*RAD
##                        else :
##                                tilt_dysdx[k] = 90.0
                xy2[i,:] = data_xyexp[i,:] + fmult * dxycam[i,:]
                        
        ind0 = list(range(npics))
        
        xy1 = data_xyexp
        p.rcParams['lines.markersize'] = 8
        p.figure(figsize=(8,8))            
        p.plot(xy1[ind0,0],-xy1[ind0,1],'bo')
        p.plot(xy2[ind0,0],-xy2[ind0,1],'ro')
##        for i in ind0 :
##                spotlabel = 
##                p.text(xy2[i,0]+70,-xy2[i,1]-50, spotlabel, fontsize = 16,horizontalalignment='center', verticalalignment='center')
        p.xlabel('xcam XMAS - toward back')
        p.ylabel('-ycamXMAS - upstream')
        hkl1 = np.array(hklnew.round(decimals=1),dtype=int)
        for i in range(npics):
                x = xy2[i,0]
                y = -xy2[i,1] + 20.0
                label1 = str(hkl1[i,0])+ str(hkl1[i,1])+str(hkl1[i,2])
                p.text(x, y, label1)
        p.xlim(0.0, 2048.0)
        p.ylim(-2048.0, 0.0)
        if hkl_rot_axis  is  not None :
                    #hklrot = np.array(hkl_rot_axis.round(decimals=1),dtype=int)
                    hklrot = hkl_rot_axis.round(decimals=2)
                    label1 = "rotation axis :"+ str(hklrot[0])+ " "+ str(hklrot[1])+ " " + str(hklrot[2])
        if hkl_ctr_axis  is  not None :
                    hklrot = np.array(hkl_ctr_axis.round(decimals=1),dtype=int)
                    label1 = "ctr axis :"+ str(hklrot[0])+ str(hklrot[1])+str(hklrot[2])        
        p.text(500,-100,label1,horizontalalignment='center', verticalalignment='center')
        if label2  is  not None :
                p.text(1800,-1800,label2,horizontalalignment='center', verticalalignment='center') 
        return(0)

def plot_GB(matstarlab, 
            vector_perp_to_GB_xyzsample , 
            y_axis_of_plot_xyzsample, 
            data_schmid, 
            plotmode = "firstplot", 
            plotcolor = 'k', 
            plotlabel = None, 
            add_b = None,
            numfig = 1):

        hkl_L = data_schmid[:,7:10]
        hkl_n = data_schmid[:,1:4]
        hkl_b = data_schmid[:,4:7]
        
        p.rcParams['lines.markersize'] = 20
        un4 = array([[1.,1.,1.],[1.,-1.,1.],[-1.,1.,1.],[1.,1.,-1.]])    
        step_hgt = zeros(12,float)
        gt_hkl = zeros((12,3),float)
        gt_xyz = zeros((12,3),float)
        gt_xyznew = zeros((12,3),float)
        gt_tilt = zeros(12,float)
        ub = zeros((12,3),float)
        ub_xyz = zeros((12,3),float)
        ub_xyznew = zeros((12,3),float)
        f_edge = zeros(12,float)
        uGBxyz = vector_perp_to_GB_xyzsample / norme(vector_perp_to_GB_xyzsample)
        hkl = xyz_sample_to_hkl(matstarlab, vector_perp_to_GB_xyzsample)
        #print "hkl", hkl.round(decimals = 3)
        #toto = hkl_to_xyz_sample(matstarlab,hkl)
        #toto = toto / toto[0]
        #print "toto", toto.round(decimals = 3)  
        uhkl = hkl/norme(hkl)
        uGBcr = uhkl*1.0
        if y_axis_of_plot_xyzsample  is  not None :
                urefy = y_axis_of_plot_xyzsample / norme(y_axis_of_plot_xyzsample)
                urefx = cross(urefy,uGBxyz)
                #print norme(urefx)
                mat1 = column_stack((urefx,urefy,uGBxyz))
                mat2 = mat1.transpose()
                
        for j in range(12):
                ub[j,:] = hkl_b[j]/norme(hkl_b[j])
                step_hgt[j] = abs(inner(uGBcr,ub[j]))
                un = hkl_n[j]/norme(hkl_n[j])
                gt_hkl[j,:] = cross(uGBcr,un)
                gt_hkl[j,:] = gt_hkl[j,:] / norme(gt_hkl[j,:])
                uL = hkl_L[j]/norme(hkl_L[j])
                f_edge[j] = 100.0 * (90.0 - 180.0/math.pi * np.arccos(abs(inner(gt_hkl[j,:],uL))))/90.0
                gt_xyz[j] = hkl_to_xyz_sample(matstarlab,gt_hkl[j])
                ub_xyz[j] = hkl_to_xyz_sample(matstarlab,ub[j])
                if y_axis_of_plot_xyzsample  is  not None :
                        gt_xyznew[j] = dot(mat2,gt_xyz[j])
                        ub_xyznew[j] = dot(mat2,ub_xyz[j])
                        if gt_xyznew[j,0] != 0.0 :
                                gt_tilt[j] = 180.0/math.pi * np.arctan(gt_xyznew[j,1]/gt_xyznew[j,0])
                        else :
                                gt_tilt[j] = 0.0
        
        print("grain boundary normal to (sample coord.)", uGBxyz)
        print("uGB (crystal coord.) =", (norm_h(uGBcr)).round(decimals = 3))
        print("gt = glide trace")
        print("xyz = sample coord.")       
        if y_axis_of_plot_xyzsample  is  not None :  print("ref for tilt (sample coord.) = ", urefx.round(decimals = 2)) 
        print("system       step_hgt     gt_hkl      gt_xyz       tilt(deg)     f_edge(%)    ub_xyz ")
        for j in range(12):
                print(j+1, round(step_hgt[j],2), gt_hkl[j].round(decimals=2), \
                    gt_xyz[j].round(decimals=2),gt_tilt[j].round(decimals=1), round(f_edge[j],1),\
                    ub_xyz[j].round(decimals=2))

        if plotmode == "firstplot":  p.figure(num = numfig, figsize = (8,8)) # 5.6))
        #elif plotmode == "overlay" : p.figure(num = 1)

##        #print gt_xyznew[:,:2].round(decimals=2)
##        ind1 = where((xx< 0.0) & (yy <0.0))
##        #print ind1
##        xx[ind1[0]] = -xx[ind1[0]]
##        yy[ind1[0]] = -yy[ind1[0]]
##        for j in range(4):
##                print round(xx[j],3),round(yy[j],3)
                
        toto = zeros(12,float)
        for j in range(12):
                if (gt_xyznew[j,1]< 0.0) : gt_xyznew[j] = - gt_xyznew[j]  
                if (inner(ub_xyznew[j,:2],gt_xyznew[j,:2])< 0.0): ub_xyznew[j] = - ub_xyznew[j]
                
        xx = gt_xyznew[:,0]
        yy = gt_xyznew[:,1]
        dxx = ub_xyznew[:,0]
        dyy = ub_xyznew[:,1]
        
#        print shape(xx)
#        print shape(yy)
        
        fmult = 1.2
        color1 = plotcolor + "o-"

        if add_b  is None : jmax = 12
        else : jmax = 4
        for k in range(4):
                for j in range(jmax):
                        #print un4[k], hkl_n[j]
                        if norme(cross(un4[k],hkl_n[j]))< 0.01 :
                                p.plot([0.,xx[j]],[0.,yy[j]],color1, label = plotlabel)
                                p.text(xx[j]*fmult,yy[j]*fmult,str(j+1),color=plotcolor, fontsize = 16, horizontalalignment='center', verticalalignment='center')
                                break
        fmult = 1.4
        p.rcParams['lines.markersize'] = 12
        color1 = plotcolor + "s-"
        for j in range(4):
                text1 = "b"+str(j+1)
                p.plot([xx[j],xx[j]+dxx[j]],[yy[j],yy[j]+dyy[j]],color1)
                p.text(xx[j]+dxx[j]*fmult,yy[j]+dyy[j]*fmult,text1, color = plotcolor,fontsize = 16, horizontalalignment='center', verticalalignment='center')   
                
        labelx = str(urefx.round(decimals = 2))+ " (sample coord.)"                    
        labely = str(urefy.round(decimals = 2))+ " (sample coord.)"
        p.xlabel(labelx)
        p.ylabel(labely)
        p.axis("equal")
        
        print("Schmid factor", end=' ')
        print(data_schmid[:,-1].round(decimals=3))
        print("step height", end=' ')
        print(step_hgt.round(decimals=2))
        print("glide trace tilt", end=' ')
        print(gt_tilt.round(decimals=1))
#        scale1 = 2.5
#        p.xlim(-scale1,scale1)
#        p.ylim(-1.0,scale1)
        #if plotmode == "overlay" : p.legend(loc=2)

        return(0)

def norm_h(uq):
        uqnew = uq * 1.0
        if uq[0]!= 0.0 : uqnew = uq / uq[0]
        elif uq[1]!= 0.0 : uqnew = uq / uq[1]
        elif uq[2]!= 0.0 : uqnew = uq / uq[2]
        return(uqnew)

def convert_jsm_summary_file_to_or_summary_file(filesum):
    
    # pour minimiser les matrices
    
    data_list, listname, nameline0 = read_summary_file(filesum) 
    
    header = nameline0 + "\n"
    header2 = ""
    for name1 in listname :
        header2 = header2 + name1 + " "
        
    header2 = header2 + "\n"
    print(header2)
    
    data_list = np.array(data_list, dtype=float)
    
    numig = shape(data_list)[0]
    print(numig)
    ndata_cols = shape(data_list)[1]
    print(ndata_cols) 
    
    indmatstart = listname.index("matstarlab_0") # 7
    indmat = np.arange(indmatstart,indmatstart + 9)
    mat_list = data_list[:,indmat]
    
    other_cols_before = data_list[:,:indmatstart]
    other_cols_after = data_list[:,indmatstart+9:]
    
    matnew = zeros((numig,9),float)
    indgoodmat = []
    
    for i in range(numig):
        matstarlab = mat_list[i,:]
        matLT3x3 = F2TC.matstarlabOR_to_matstarlabLaueTools(matstarlab)
    #            print det(matLT3x3)
        if abs(det(matLT3x3))< 0.01 :
            print("bad matrix", i)
        else :
            indgoodmat.append(i)
            matmin, transfmat = FindO.find_lowest_Euler_Angles_matrix(matLT3x3, verbose = 0)
            matnew[i,:] = F2TC.matstarlabLaueTools_to_matstarlabOR(matmin) 
    
    indgoodmat = np.array(indgoodmat, dtype = int)
    
    toto = column_stack((other_cols_before[indgoodmat,:], matnew[indgoodmat,:], other_cols_after[indgoodmat,:]))

    if 1 :
        outfilesum = filesum.rstrip(".dat") + "_converted" + ".dat"
        print(outfilesum)
        outputfile = open(outfilesum,'w')
        outputfile.write(header)
        outputfile.write(header2)
        np.savetxt(outputfile, toto, fmt = "%.6f")
        outputfile.close()

    return(outfilesum)    

def ReadSpec(fname,scan):
   """read spec file . Better reuse the ReadSpec in logfile_reader.py (spec_reader)"""
   f=open(fname,'r')
   s="#S %i"%scan
   title=0
   fileheader = ""
   firsttime = 1
   nlines_in_fileheader = 0
   while 1:
      title=f.readline()    
      if (title[:2]!="#S")&(firsttime) :
          fileheader += title
          nlines_in_fileheader += 1
      if (title[:2]=="#S")&(firsttime) : firsttime = 0                   
      if s == title[0:len(s)]:
         break;
      if len(title)==0:
         nlines_in_fileheader -= 1
         break;

   print("nlines_in_fileheader = ", nlines_in_fileheader)
   print("*******************")
   print("scan title from specfile : ")          
   print(title.rstrip(PAR.cr_string))
   print("*********************")
   header  = ""
   header = header + title
   s="#L"
   coltit=0
   while 1:
      coltit=f.readline()
      header = header + coltit
      if s == coltit[0:len(s)]:
         break;
      if len(coltit)==0:
         break;
   d={}
   print("list of column names in scan from specfile : ")
   print(coltit)
   print("************************")
   coltit=coltit.split()
   for i in range(1,len(coltit)):
    d[coltit[i]]=[]
   while 1:
      l=f.readline()
      if len(l)<5:
         break;
      if l[0]=="#":
         continue
      l=l.split()
      for i in range(1,len(coltit)):
         d[coltit[i]].append(float(l[i-1]))
   nb=len(d[coltit[1]])
   for i in range(1,len(coltit)):
      a=zeros(nb,'f')
      for j in range(nb):
        a[j]=d[coltit[i]][j]
#      d[coltit[i]]=copy.deepcopy(a)
   d[coltit[i]]=copy(a)
   f.close()
   return title,d, header, fileheader

import collections
   
def read_scan_in_specfile(spec_file, 
                          spec_scan_num,
                          list_colnames = ['img', "thf", "xechcnt", "yechcnt", "zech", "Monitor", "fluo"],
                          verbose = 1,
                          user_comment = "",
                          dict_mot_pos_in_header = { "xech":[1,1,0.],
                          "yech": [1,2,0.],
                            "zech" :[1,3,0.]                    
                            },   
                          CCDlabel = PAR.CCDlabel):    
 
# example of list of motors in specfile header                             
#O0     rien     light       vg3       hg3       vo3       ho3      pfoc      hfoc
#O1     expo      xech      yech      zech        xs        ys        xm        ym
#O2       zm        zc      zoom      tdet       tx1       ty1       tz1       rx1
#O3      ry1       rz1       tx2       ty2       tz2       rx2       ry2       rz2
#O4      vof     yshut     focus     xcomp     ycomp       rot       thf       hof
#O5       yf        zf      ycam       ho4       vo4       hg4       vg4      four
#O6     volt                          
# the default dict_mot_pos_in_header corresponds to this specfile header   
                                    
    img_counter = list_colnames[0]     
                        
    if verbose :
        print("spec file = ", spec_file)
        
    scan_title, d, scan_header, file_header = ReadSpec(spec_file,spec_scan_num)
    print("*************************************************")
    print("file header")
    print(file_header)
    print("*************************************************")    
    
    dict_mot_pos_in_header_ordered =  collections.OrderedDict(sorted(dict_mot_pos_in_header.items()))
        
    print("check consistency between file header and dict_mot_pos_in_header")
    lines_of_header = file_header.split("\n")
    for key in list(dict_mot_pos_in_header_ordered.keys()) :
        n_line_mot = 0
        for line in lines_of_header :
            if line[:2]== "#O" :
               listname = line.split()
               if key in listname : 
                   n_col_mot = listname.index(key)-1
                   print(key, "found at nline, ncol = ", n_line_mot,  n_col_mot, "expected at", end=' ')
                   print(dict_mot_pos_in_header_ordered[key][:2])
                   break
               n_line_mot += 1
                
    motors_to_read_in_scan_header = list(dict_mot_pos_in_header_ordered.keys()) # ["xs", "ys", "zech"]
    #print "motors_to_read_in_scan_header :"
    #print motors_to_read_in_scan_header
                
    for motor in motors_to_read_in_scan_header :
        for line in scan_header.split("\n"):
            linenum = dict_mot_pos_in_header[motor][0]
            if line[:3] == "#P" + str(linenum):
#                print "found"
#                print line.split(" ")
                dict_mot_pos_in_header_ordered[motor][2] = float(line.split(" ")[dict_mot_pos_in_header[motor][1]+1] )        
    
    print("motor positions read from scan header (dict_mot_pos_in_header_sorted) :")       
#    print "motor [ line_in_header,  column_in_line,  position ]"            
    for key, value in dict_mot_pos_in_header_ordered.items():
        print(key, value[2])
    print("****************************************")
    
    ctime = float(scan_title.rstrip(PAR.cr_string).split()[-1])
    
    d_sorted = collections.OrderedDict(sorted(d.items()))
        
    if verbose :
        print("ctime = ", ctime)
        print("*************************************")
        print("scan header : ")
        print(scan_header)
        print("****************************************")
        print("column values for first scan point :")
        for key, value in d_sorted.items():
            print(key, value[0])
            
    float_column_names = ["thf", "yf", "xs", "ys", "xech", "yech", 
                          "zech", "rien", "xechcnt", "yechcnt",
                          "xfwhm", "yfwhm", "T_tilt", "ycompct", "Seconds",
                          "xpos", "ypos", "Rstd", "xlt00", "xdt00" ]
                          
    int_column_names = [img_counter, "Monitor", "fluo", "Epoch", 
                        "fluo1", "fluo2", "fluo3", "xicr00", "xocr00", "fluoz"]

#    list_colnames =  ["img", "yech" ,   "Epoch",  "Monitor",  "Rstd",  "fluo",  "xpos", "ypos",
#                      "xfwhm", "yfwhm", "T_tilt", "ycompct", "xechcnt", "yechcnt"
#        
    dict_dtype = {}
    for key in float_column_names : dict_dtype[key] = ["float",]
    for key in int_column_names : dict_dtype[key] = ["int",]
    
    print("list_colnames = ", list_colnames)
    print("img_counter (first name in list_colnames) = ", img_counter)  
    if (img_counter == "img2")&(CCDlabel != "ImageStar") : # VHR    # bizarre
        print("warning : VHR camera : using image number = img2 + 1")   
        
    if verbose :            
        print("***********************************")
        print(scan_title)
        print("counter : start next end step range") 
    
    dict_spec = {}    

    for key in list_colnames :
        if (key not in float_column_names) and (key not in int_column_names) :
            print("warning : in multigrain.py read_scan_in_specfile")
            print("counter type (float or int) is not defined")
            print("use float type", key)
            dict_dtype[key] = ["float",]
            
        if key in list(d_sorted.keys()):
#            print key
            if (key == img_counter)&(img_counter == "img2")&(CCDlabel != "ImageStar") : # VHR
                dict_spec[key] = np.array(d[key], dict_dtype[key][0]) + 1
            else :
                dict_spec[key] = np.array(d[key], dict_dtype[key][0])
            if verbose : 
                step = dict_spec[key][1] - dict_spec[key][0]
                range1 =  dict_spec[key][-1] - dict_spec[key][0]
                print(key, dict_spec[key][0], dict_spec[key][1], dict_spec[key][-1], step, range1)
        else : 
            print("column = ", key, "not present in this scan")
            raise ValueError                        

    dict_spec_ordered = collections.OrderedDict(sorted(dict_spec.items()))  
    
    user_comment1 = user_comment + "#specfile : " + spec_file + '\n'
    user_comment1 += scan_header
    
    return(dict_spec_ordered, ctime, user_comment1, dict_mot_pos_in_header_ordered, scan_title)
       
def read_list_of_scans_from_specfile(        
                       spec_file,
                       scan_list,
                       dict_mot_pos_in_header = { "xech":[1,1,0.],
                          "yech": [1,2,0.],
                            "zech" :[1,3,0.]                    
                            }, 
                        list_colnames = ['img', "xechcnt", "yechcnt",  "Monitor", "fluo"]
                    ):
    # output :                                
    # dict_spec_all_scans : toutes les colonnes de tous les scans
    # ctime_list : counting time                           
    # user_comment_list : liste des headers de tous les scans    
#    dict_mot_pos_in_header_all_scans : liste des positions des "autres" moteurs pour tous les scans
#                liste des autres moteurs donnee dans dict_mot_pos_in_header                  
    # dict_mot_pos_in_header : noms des moteurs dont il faut aller chercher la position dans le header des scans
    #         et position ligne / colonne dans les lignes de type #O du header


    img_counter = list_colnames[0]
    
    dict_spec_all_scans = {}
    dict_mot_pos_in_header_all_scans = {} 
    user_comment_list = []
    ctime_list = []
    scan_title_list = []
    firsttime = 1
    for spec_scan_num in scan_list :

        dict_spec, ctime, user_comment1, dict_mot_pos_in_header, scan_title  = \
                   read_scan_in_specfile(spec_file, 
                              spec_scan_num,
                              list_colnames = list_colnames,
                              verbose = 0,
                              user_comment = "", dict_mot_pos_in_header = dict_mot_pos_in_header)
                              
        user_comment_list.append([user_comment1,])
        ctime_list.append(ctime)
        scan_title_list.append(scan_title)
        if firsttime :
            for key, value in dict_spec.items():
                dict_spec_all_scans[key] = [value,]
            for key, value in dict_mot_pos_in_header.items():
                dict_mot_pos_in_header_all_scans[key] = [value[2],]
            firsttime = 0
        else :
            for key, value in dict_spec.items():
                dict_spec_all_scans[key].append(value)
            for key, value in dict_mot_pos_in_header.items():
                dict_mot_pos_in_header_all_scans[key].append(value[2])

    print("********************************************")
    print("dict_spec_all_scans : ")             
    for key, value in dict_spec_all_scans.items():
        print(key, value)
    print("dict_mot_pos_in_header_all_scans :")
    for key, value in dict_mot_pos_in_header_all_scans.items():
        print(key, value)
    
    if img_counter == "img2" : # VHR
        print("warning : VHR camera : using image number = img2 + 1")
 
    print("ctime_list : ", ctime_list)
    print("scan_title_list : ", scan_title_list)          

#        print user_comment_list
       
    return(user_comment_list, dict_spec_all_scans, dict_mot_pos_in_header_all_scans, ctime_list, scan_title_list)
        
