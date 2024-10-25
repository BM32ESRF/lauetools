# -*- coding: cp1252 -*-
import os, sys
import site
from time import time, asctime
import numpy as np
from numpy.linalg import inv

import math
import matplotlib.pylab as p

sys.path.append("..")

if sys.version_info.major == 3:
    import LaueTools.FileSeries.module_graphique as modgraph
    import LaueTools.LaueGeometry as F2TC
    print('LaueGeometry is from lauetools distribution at :',F2TC.__file__)
    import LaueTools.readmccd as rmccd
    #import LaueTools.LaueAutoAnalysis as LAA
    import LaueTools.indexingAnglesLUT as INDEX
    import LaueTools.findorient as FindO
    import LaueTools.CrystalParameters as CP
    from LaueTools.generaltools import norme_vec as norme
    import LaueTools.generaltools as GT
    import LaueTools.IOLaueTools as IOLT
    import LaueTools.dict_LaueTools as DictLT
    #from mosaic import ImshowFrameNew, ImshowFrame_Scalar, ImshowFrame
    from LaueTools.GUI.mosaic import ImshowFrame_Scalar, ImshowFrame

# TODO   to refactor -------------------
# set invisible parameters for serial_peak_search, serial_index_refine_multigrain
import LaueTools.FileSeries.param_multigrain as PAR

omega_sample_frame = 40.0

if omega_sample_frame != None:
    omegadeg = omega_sample_frame * np.pi / 180.0
    # rotation de -omega autour de l'axe x pour repasser dans Rsample
    mat_from_lab_to_sample_frame = np.array([[1.0, 0.0, 0.0],
                                            [0.0, np.cos(omegadeg), np.sin(omegadeg)],
                                            [0.0, -np.sin(omegadeg), np.cos(omegadeg)]])
else:
    mat_from_lab_to_sample_frame = np.eye(3)  # put
#------------------------------

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

# calcul reseau reciproque

def dlat_to_rlat(dlat):
    """
    # Compute reciprocal lattice parameters. The convention used is that
    # a[i]*b[j] = d[ij], i.e. no 2PI's in reciprocal lattice.
    """

    rlat = np.zeros(6)
    # compute volume of real lattice cell

    volume = (dlat[0] * dlat[1] * dlat[2]
        * np.sqrt(1 + 2 * np.cos(dlat[3]) * np.cos(dlat[4]) * np.cos(dlat[5])
            - np.cos(dlat[3]) * np.cos(dlat[3])
            - np.cos(dlat[4]) * np.cos(dlat[4])
            - np.cos(dlat[5]) * np.cos(dlat[5])))

    # compute reciprocal lattice parameters

    rlat[0] = dlat[1] * dlat[2] * np.sin(dlat[3]) / volume
    rlat[1] = dlat[0] * dlat[2] * np.sin(dlat[4]) / volume
    rlat[2] = dlat[0] * dlat[1] * np.sin(dlat[5]) / volume
    rlat[3] = np.arccos(
        (np.cos(dlat[4]) * np.cos(dlat[5]) - np.cos(dlat[3])) / (np.sin(dlat[4]) * np.sin(dlat[5])))
    rlat[4] = np.arccos(
        (np.cos(dlat[3]) * np.cos(dlat[5]) - np.cos(dlat[4])) / (np.sin(dlat[3]) * np.sin(dlat[5])))
    rlat[5] = np.arccos(
        (np.cos(dlat[3]) * np.cos(dlat[4]) - np.cos(dlat[5])) / (np.sin(dlat[3]) * np.sin(dlat[4])))

    return rlat


def rad_to_deg(dlat):
    """ convert 3 last elements of 6 lattice parameters from rad to deg
    """
    dlatdeg = np.hstack((dlat[0:3], dlat[3:6] * 180.0 / math.pi))
    return dlatdeg


def deg_to_rad(dlat):
    """ convert 3 last elements of 6 lattice parameters from deg to rad
    """
    dlatrad = np.hstack((dlat[0:3], dlat[3:6] * math.pi / 180.0))
    return dlatrad


def dlat_to_dlatr(dlat):

    dlatr = np.zeros(6)
    for i in range(0, 3):
        dlatr[i] = dlat[i] / dlat[0]
    for i in range(3, 6):
        dlatr[i] = dlat[i]

    return dlatr


def epsline_to_epsmat(epsline):  # 29May13
    """
    # deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12
    """
    epsmat = np.identity(3, float)

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
    epsline = np.zeros(6, float)

    epsline[0] = epsmat[0, 0]
    epsline[1] = epsmat[1, 1]
    epsline[2] = epsmat[2, 2]

    epsline[3] = epsmat[1, 2]
    epsline[4] = epsmat[0, 2]
    epsline[5] = epsmat[0, 1]

    return epsline

def matstarlab_to_deviatoric_strain_sample(matstarlab, 
                                           omega0=omega_sample_frame, 
                                           version=2, 
                                           returnmore=False,
                                           reference_element_for_lattice_parameters="Ge"):
    #29May13
    epsp_crystal, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(matstarlab, 
                                            version = version, 
                                            reference_element_for_lattice_parameters = reference_element_for_lattice_parameters)

    epsp_sample =  transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab,
                                                                  epsp_crystal,
                                                                  omega0 = omega0)
    if returnmore == False:
        return(epsp_sample)
    else:
        return(epsp_sample, epsp_crystal)   # add epsp_crystal


def matstarlab_to_deviatoric_strain_crystal(matstarlab, version=2, elem_label="Ge"):
    # 29May13
    """
    # version = 1 : simplified calculation for initially cubic unit cell
    # version = 2 : full calculation for unit cell with any symmetry
    # formulas from Tamura's XMAS chapter in Barabash 2013 book
    # = same as Chung and Ice 1999 (but clearer explanation)
    # needs angles in radians
    # dlat[0] can be any value, not necessarily 1.0
    """

    rlat = CP.mat_to_rlat(matstarlab)
    dlat = dlat_to_rlat(rlat)
    # print "dlat = ", dlat  # - np.array([1.,1.,1.,math.pi/2.,math.pi/2.,math.pi/2.])
    dlatrdeg = rad_to_deg(dlat_to_dlatr(dlat))

    if version == 1:  # only for initially cubic unit cell

        epsp = np.zeros(6, float)

        tr3 = (dlat[0] + dlat[1] + dlat[2]) / 3.0

        epsp[0] = (dlat[0] - tr3) * 1000.0 / dlat[0]
        epsp[1] = (dlat[1] - tr3) * 1000.0 / dlat[0]
        epsp[2] = (dlat[2] - tr3) * 1000.0 / dlat[0]

        epsp[3] = -1000.0 * (dlat[3] - math.pi / 2.0) / 2.0
        epsp[4] = -1000.0 * (dlat[4] - math.pi / 2.0) / 2.0
        epsp[5] = -1000.0 * (dlat[5] - math.pi / 2.0) / 2.0

    elif version == 2:  # for any symmetry of unit cell

        # reference lattice parameters with angles in degrees
        dlat0_deg = np.array(DictLT.dict_Materials[elem_label][1], dtype=float)
        dlat0 = deg_to_rad(dlat0_deg)

        # print dlat0.round(decimals = 4)
        # print dlat.round(decimals = 4)

        # matstarlab construite pour avoir norme(astar) = 1
        Bdir0 = CP.rlat_to_Bstar(dlat0)
        Bdir0 = Bdir0 / dlat0[0]

        Bdir = CP.rlat_to_Bstar(dlat)
        Bdir = Bdir / dlat[0]

        # print Bdir0.round(decimals=4)
        # print Bdir.round(decimals=4)

        # Rmat = inv(Bdir) et T = dot(inv(Rmat), Rmat0)

        Tmat = np.dot(Bdir, inv(Bdir0))

        eps1 = 0.5 * (Tmat + Tmat.transpose()) - np.eye(3)
        # print eps1.ronp.und(decimals=2)
        # print np.trace(eps1)

        # la normalisation du premier vecteur de Bdir a 1
        # ne donne pas le meme volume pour les deux mailles
        # => il faut soustraire la partie dilatation

        epsp1 = 1000.0 * (eps1 - (np.trace(eps1) / 3.0) * np.eye(3))
        # print epsp1.round(decimals=1)
        # print np.trace(epsp1)

        epsp = epsmat_to_epsline(epsp1)

    # print "deviatoric strain 11 22 33 -dalf 23, -dbet 13, -dgam 12  *1e3 \n", epsp.round(decimals=1)

    # print "dlatrdeg = \n", dlatrdeg

    return (epsp, dlatrdeg)


def sort_list_decreasing_column(data_str, colnum):

    # print "sort list, decreasing values of column ", colnum

    npics = np.shape(data_str)[0]
    # print "nlist = ", npics
    index2 = np.zeros(npics, int)

    index1 = np.argsort(data_str[:, colnum])
    for i in range(npics):
        index2[i] = index1[npics - i - 1]
    # print "index2 =", index2
    data_str2 = data_str[index2]

    return data_str2


def hkl_to_xystereo(hkl0, polar_axis=[0.0, 0.0, 1.0], down_axis=[1.0, 0.0, 0.0], return_more=None):

    uq = hkl0 / norme(hkl0)
    uz = polar_axis / norme(polar_axis)
    udown = down_axis / norme(down_axis)
    uright = np.cross(uz, udown)
    uqz = np.inner(uq, uz)
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
    qsxy = np.array([np.inner(qs, uright), -np.inner(qs, udown)])
    # print qsxy.round(decimals=3)

    if return_more == None:
        return qsxy
    else:
        return (qsxy, change_sign)

if 0:  # ## test 1  #29May13

    # mat3x3 = np.array([[1.0,0.01,0.0],[0.0,0.99995,0.0],[0.0,0.0,1.0]])
    # mat3x3 = np.array([[1.,0.01,0.02],[0.,1.,0.03],[0.,0.,1.]])
    mat3x3 = np.array([[1.0, 0.01, 0.02], [0.0, 1.05, 0.03], [0.0, 0.0, 1.03]])
    print("mat3x3")
    print(mat3x3)

    matstarlab = GT.mat3x3_to_matline(mat3x3)
    print("matstarlab")
    print(matstarlab)

    print("strain in crystal frame")
    print("version 1 = linearize")
    print("version 2 = use B matrix")

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarlab, version=1, reference_element_for_lattice_parameters="Ge")
    print("version 1 :")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarlab, version=2, reference_element_for_lattice_parameters="Ge")

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
        matstarlabLT, version=2, reference_element_for_lattice_parameters="Ge")

    print("version 2 : using matstarlabLT")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    matstarsample3x3 = CP.matstarlab_to_matstarsample3x3(matstarlab)
    print("matstarsample3x3")
    print(matstarsample3x3)

    matstarsample = GT.mat3x3_to_matline(matstarsample3x3)

    epsp1, dlatrdeg = matstarlab_to_deviatoric_strain_crystal(
        matstarsample, version=2, reference_element_for_lattice_parameters="Ge")

    print("version 2 : using matstarsample")
    print("deviatoric strain aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-4 units)")
    print((epsp1 * 10.0).round(decimals=2))

    print("strain in sample frame")
    print("version 2 :")
    epsp_sample_v1 = CP.matstarlab_to_deviatoric_strain_sample(matstarlab,
                                                    omega0=40.0,
                                                    version=2,
                                                    reference_element_for_lattice_parameters="Ge")
    print("deviatoric strain xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-4 units)")
    print((epsp_sample_v1 * 10.0).round(decimals=2))

    jklsqdjkl

if 0:  # ## test 2 #29May13
    mat3x3 = np.array([[0.01, 1.01, 0.02], [0.0, 0.03, 1.02], [1.03, 0.0, 0.0]])
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
    mat3x3 = np.array([[1.01, 0.01, 0.02], [0.0, 1.02, 0.03], [0.0, 0.0, 1.03]])
    mat3x3 = np.array([[0.01, 1.01, 0.02], [0.0, 0.03, 1.02], [1.03, 0.0, 0.0]])
    print("mat3x3")
    print(mat3x3)

    matstarlab = GT.mat3x3_to_matline(mat3x3)
    print("matstarlab")
    print(matstarlab)

    matstarlabOND = CP.matstarlab_to_matstarlabOND(matstarlab)
    print("matstarlabOND")
    print(matstarlabOND.round(decimals=6))

    mat3x3OND = GT.matline_to_mat3x3(matstarlabOND)
    print("mat3x3OND")
    print(mat3x3OND.round(decimals=6))

    klmk


def sort_peaks_decreasing_int(data_str, colnum):

    print("tri des pic par intensite decroissante")

    npics = np.shape(data_str)[0]
    print(npics)
    index2 = np.zeros(npics, int)

    index1 = np.argsort(data_str[:, colnum])
    for i in range(npics):
        index2[i] = index1[npics - i - 1]
    # print "index2 =", index2
    data_str2 = data_str[index2]

    return data_str2

# uses a number of invisible parameters set in param.py
def serial_indexerefine_multigrain( filepathdat, fileprefix, indimg, filesuffix,
                                                    filefitcalib,
                                                    filepathout,
                                                    filefitref=None):

    nimg = len(indimg)
    ngrains_found = np.zeros(nimg, int)
    npeaks = np.zeros((nimg, PAR.ngrains_index_refine), int)
    pixdev = np.zeros((nimg, PAR.ngrains_index_refine), float)

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
        filedat1 = os.path.join(filepathdat,
                            fileprefix
                            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
                            + filesuffix)
        print("image in :")
        print(filedat1)
        # print "saving fit in :"
        filefit = (fileprefix
                    + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
                    + PAR.add_str_index_refine
                    + ".fit")
        

        modgraph.savindexfit = os.path.join(filepathout, filefit)
        # print os.listdir(filepathout)
        j = 0
        if not PAR.overwrite_index_refine:
            while filefit in os.listdir(filepathout):
                print("warning : change name to avoid overwrite")
                filefit = (fileprefix
                            + rmccd.stringint(i, PAR.number_of_digits_in_image_name)
                            + PAR.add_str_index_refine
                            + "_new_"
                            + str(j)
                            + ".fit")
                print(filepathout + filefit)
                import module_graphique as modgraph

                modgraph.savindexfit = os.path.join(filepathout, filefit)
                j = j + 1

        filefit_withdir = filepathout + filefit
        print(PAR.elem_label_index_refine)

        filefitmg, ngrains_found[k], npeaks[k, :], pixdev[k, :] = index_refine_multigrain_one_image(
                                                filedat1,
                                                PAR.elem_label_index_refine,
                                                filefitcalib,
                                                ngrains=PAR.ngrains_index_refine,
                                                proposed_matrix=proposed_matrix)

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

def build_xy_list_by_hand(outputfullpath, nx, ny, xfast, yfast, xstep, ystep,
            startindex=modgraph.indimg[0],
            lastindex=modgraph.indimg[-1]):
    """
    write a file with image index and x and y sample positions
    """
    if yfast:
        xylist = np.zeros((nx, ny, 2), float)

        for i in range(nx):
            xylist[i, :, 0] = float(i) * xstep
        for j in range(ny):
            xylist[:, j, 1] = float(j) * ystep

    if xfast:
        xylist = np.zeros((ny, nx, 2), float)

        for i in range(nx):
            xylist[:, i, 0] = float(i) * xstep
        for j in range(ny):
            xylist[j, :, 1] = float(j) * ystep

    # print('xylist.shape',xylist.shape)

    xylist_new = xylist.reshape((nx) * (ny), 2)
    indimg = np.arange((nx) * (ny)) + startindex
    data_list = np.column_stack((indimg, xylist_new))

    print("data_list", data_list)

    header = "img 0, xech 1, yech 2"

    print("writing image index x,y sample position in:", outputfullpath)

    outputfile = open(outputfullpath, "w")
    # print("np.__version__",np.__version__)
    # if np.__version__<'1.7':
    # outputfile.write(header+" \n")
    np.savetxt(outputfullpath, data_list, fmt="%.4f",header=header, comments='')

    outputfile.close()

    return outputfullpath


def build_summary(fileindex_list:list, filepathfit:str, fileprefix:str, filesuffix, filexyz,
                                                        startindex=modgraph.indimg[0],
                                                        finalindex=modgraph.indimg[-1],
                                                        number_of_digits_in_image_name=4,
                                                        nbtopspots=10,
                                                        outputprefix="_SUMMARY_",
                                                        folderoutput=None,
                                                        default_file=None,
                                                        verbose=0):  # 29May13
    """
    write a file containing the summary of results from a set .fit file
    fileindex_list: list of file index

    :param filepathfit: path to folder containing .fit files

    :param filesuffix: extension of files to be scanned 

    # mean local grain intensity is taken over the most intense ntopspots spots
    nbtopspots = 10

    number_of_digits_in_image_name :  nb of 0 padded integer formatting
                                    example: for 4  , then 56 => 0056
                                    0 to simpliest integer formatting (not zero padding)
    """
    if verbose > 0: print('\n In build_summary()  :')
    # filexyz : img 0 , xech 1, yech 2, zech 3, mon4 4, lambda 5
    total_nb_cols = 25

    list_col_names = ["dxymicrons", "matstarlab", "strain6_crystal", "euler3"]
    number_col_list = np.array([2, 9, 6, 3])

    list_col_names2 = ["img", "gnumloc", "npeaks", "pixdev", "intensity"]

    for k in range(len(list_col_names)):
        for nbcol in range(number_col_list[k]):
            lcn = list_col_names[k] + "_" + str(nbcol)
            list_col_names2 += [lcn]

    header2 = ""
    for i in range(total_nb_cols):
        header2 = header2 + list_col_names2[i] + " "
    header2 += "\n"
    if verbose > 0: print(header2)


    # read xyz position file
    posxyz = np.loadtxt(filexyz, skiprows=1)
    xy = posxyz[:, 1:3]
    imgxy = np.array(posxyz[:, 0], dtype=np.int16)
    dxy = xy - xy[0, :]  # *1000.0

    if verbose > 0:
        print("imgxy", imgxy)
        print("dxy", dxy)

    # if nb of elems in file xy is less than nb of images to be compiled for the summary
    if len(dxy) < len(fileindex_list):
        raise ValueError("Map from filexy file has %d elements which is smaller than the %d .fit files to compile !"%(len(dxy), len(fileindex_list)))

    list_files_in_folder = os.listdir(filepathfit)
    import re

    test = re.compile("\.fit$", re.IGNORECASE)
    list_fitfiles_in_folder = list(filter(test.search, list_files_in_folder))

    # loop for reading each .fit file -------------------------
    # encodingdigits = "%%0%dd" % int(number_of_digits_in_image_name)
    if verbose > 0: print("fileindex_list", fileindex_list)

    iloop = 0
    for fileindex in fileindex_list:
        ind0 = np.where(imgxy == fileindex)

        if verbose > 1: print("dxy = ", dxy[ind0[0], :])
        _filename = fileprefix +  str(fileindex).zfill(int(number_of_digits_in_image_name)) + filesuffix

        if _filename not in list_fitfiles_in_folder:
            if verbose > 1: print("Warning! missing .fit file: %s" % _filename)
            res = np.zeros(total_nb_cols, float)
            res[0] = fileindex

            if iloop == 0:
                allres = res
            else:
                allres = np.row_stack((allres, res))
            if verbose > 1: print('allres.shape',allres.shape)
            iloop += 1
            continue

        filefitmg = os.path.join(filepathfit, _filename)

        if verbose > 1:
            print('Starting to read file .fit: ', filefitmg)

        res1 = IOLT.readfitfile_multigrains(filefitmg,
                                            verbose=verbose,
                                            readmore=True,
                                            fileextensionmarker=".cor")

        # print("res1  from readfitfile_multigrains in build_summary()", res1)

        if res1 != 0:
            if verbose > 1:
                print('Nb output elements of readfitfile_multigrains()', len(res1))
                print('We select only 9 first of them ...')
            gnumlist, npeaks, indstart, matstarlab, data_fit, calib, pixdev, strain6, euler = res1[:9]

            if len(pixdev) == 0:
                pixdev = np.zeros_like(gnumlist)

            ngrains = len(gnumlist)
            if verbose > 1: print("ngrains", ngrains)

            intensity = np.zeros(ngrains, float)
            if ngrains > 1:
                for j in range(ngrains):
                    range1 = np.arange(indstart[j], indstart[j] + npeaks[j])
                    data_fit1 = data_fit[range1, :]
                    intensity[j] = data_fit1[:nbtopspots, 1].mean()
            else:
                intensity[0] = data_fit[:nbtopspots, 1].mean()
                strain6 = strain6.reshape(1, 6)
                euler = euler.reshape(1, 3)

            imnumlist = np.ones(ngrains, int) * fileindex
            dxylist = np.multiply(np.ones((ngrains, 2), float), dxy[ind0[0], :])
            if verbose > 1:
                print('dxylist', dxylist)

            res = np.column_stack((imnumlist, gnumlist, npeaks, pixdev, intensity,
                                    dxylist, matstarlab, strain6, euler))

            if verbose > 1: print("intensity in build_summary()", intensity)
        else:
            res = np.zeros(total_nb_cols, float)
            res[0] = fileindex

            if verbose > 1: print("something is empty")

        if iloop == 0:
            allres = res
        else:
            allres = np.row_stack((allres, res))

        iloop += 1

    if verbose > 0:
        print("shape allres")
        print(np.shape(allres))
        #print(folderoutput)
        print("summary file will be saved in :")
        print(fileprefix + "%s%s_to_%s.dat" % (outputprefix, str(startindex), str(finalindex)))

    header = "img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, strain6_crystal 16:22, euler 22:25  \n"

    # write summary file -------------
    try:
        from . import module_graphique as modgraph

        fullpath_summary_filename = os.path.join(folderoutput,
                            fileprefix
                            + "%s%s_to_%s.dat" % (outputprefix, str(startindex), str(finalindex)))

        if verbose > 0: print("fullpath_summary_filename", fullpath_summary_filename)
        modgraph.filesumbeforecolumn = fullpath_summary_filename
        outputfile = open(fullpath_summary_filename, "w")
        outputfile.write(header)
        outputfile.write(header2)
        np.savetxt(outputfile, allres, fmt="%.6f")
        outputfile.close()

    except ImportError:
        print("Missing module_graphique.py")

    return allres, fullpath_summary_filename


def read_summary_file(filesum:str, read_all_cols="yes", verbose=0,
    list_column_names=["img", "gnumloc", "npeaks", "pixdev", "intensity",
        "dxymicrons_0", "dxymicrons_1",
        "matstarlab_0", "matstarlab_1", "matstarlab_2", "matstarlab_3",
        "matstarlab_4", "matstarlab_5", "matstarlab_6", "matstarlab_7", "matstarlab_8",
        "strain6_crystal_0", "strain6_crystal_1", "strain6_crystal_2",
        "strain6_crystal_3", "strain6_crystal_4", "strain6_crystal_5",
        "euler3_0", "euler3_1", "euler3_2",
        "strain6_sample_0", "strain6_sample_1", "strain6_sample_2",
        "strain6_sample_3", "strain6_sample_4", "strain6_sample_5",
        "rgb_x_sample_0", "rgb_x_sample_1", "rgb_x_sample_2",
        "rgb_z_sample_0", "rgb_z_sample_1", "rgb_z_sample_2",
        "stress6_crystal_0", "stress6_crystal_1", "stress6_crystal_2", "stress6_crystal_3", "stress6_crystal_4", "stress6_crystal_5",
        "stress6_sample_0", "stress6_sample_1", "stress6_sample_2", "stress6_sample_3", "stress6_sample_4", "stress6_sample_5",
        "res_shear_stress_0", "res_shear_stress_1", "res_shear_stress_2", "res_shear_stress_3", "res_shear_stress_4", "res_shear_stress_5", "res_shear_stress_6", "res_shear_stress_7", "res_shear_stress_8", "res_shear_stress_9", "res_shear_stress_10", "res_shear_stress_11",
        "max_rss",
        "von_mises"]):
    """
    used by plot_maps2

    :param filesum: str, full path to summary file (.dat)

    :return: data_sum_select_col, list_column_names, firstline
    """
    # 29May13
    if verbose > 0:
        print("In read_summary_file(): summary file is %s"%filesum)
        print("first two lines :")
    f = open(filesum, "r")
    i = 0
    try:
        for line in f:
            if i == 0:
                firstline = line.rstrip("  \n")
            if i == 1:
                nameline1 = line.rstrip("\n")
            i = i + 1
            if i > 2:
                break
    finally:
        f.close()

    listname = nameline1.split()

    data_sum = np.loadtxt(filesum, skiprows=2)

    if verbose > 0: print('data_sum.shape', data_sum.shape)

    if len(data_sum.shape) != 2:
        raise ValueError(f'In read_summary_file(): data_sum.shape is {data_sum.shape} and should be 2 dimension')

    if read_all_cols == "yes":
        return data_sum, listname, firstline

    else:
        print('len(listname)',len(listname))
        ncol = len(list_column_names)

        ind0 = np.zeros(ncol, int)

        for i in range(ncol):
            ind0[i] = listname.index(list_column_names[i])

        print('ind0',ind0)

        data_sum_select_col = data_sum[:, ind0]

        if verbose > 0:
            print('some details in read_summary_file()')
            print(np.shape(data_sum))
            print(np.shape(data_sum_select_col))
            print(filesum)
            print(list_column_names)
            print(data_sum_select_col[:5, :])

        return data_sum_select_col, list_column_names, firstline


def twomat_to_rotation_Emeric(matstarlab1, matstarlab2, omega0=40.0):

    # utilise matstarlab

    # version Emeric nov 13
    matref = matstarlab_to_matdirONDsample3x3(matstarlab1, omega0=omega0)
    matmes = matstarlab_to_matdirONDsample3x3(matstarlab2, omega0=omega0)

    # ATTENTION : Orthomormalisation avant de faire le calcul
    # matmisor = dot(np.linalg.inv(matref.transpose()),matmes.transpose())
    matmisor = np.dot(matref, matmes.T)  # cf cas CK

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

    vecRodrigues_sample = np.array([rx, ry, rz])  # axe de rotation en coordonnees sample

    theta = theta * 180.0 / np.pi

    return vecRodrigues_sample, theta

def glide_systems_to_schmid_tensors(n_ref=np.array([1., 1., 1.]), 
                                    b_ref=np.array([1., -1., 0.]),
                                    verbose=0,
                                    returnmore=0):
    #29May13
    """
    only for cubic systems
    coordonnees cartesiennes dans le repere OND obtenu en orthonormalisant le repere cristal
    cf these Gael Daveau p 16
    """
    if verbose > 0: print('In glide_systems_to_schmid_tensors():')
    nop = 24

    allop = DictLT.OpSymArray
    indgoodop = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47])
    goodop = allop[indgoodop]

    hkl_2 = np.row_stack((n_ref, b_ref))
    normehkl = np.zeros(2, float)

    uqref = np.zeros((2, 3), float)
    for i in range(2):
        normehkl[i] = norme(hkl_2[i, :])
        uqref[i, :] = hkl_2[i, :] / normehkl[i]

    uqall = np.zeros((2, nop, 3), float)
    for j in range(2):  # j=0 : n, j=1 : b
        for k in range(nop):
            uqall[j, k, :] = np.dot(goodop[k], uqref[j, :])

    isdouble = np.zeros(nop, int)    
    for k in range(nop):
        #print "k = ", k
        un_ref = uqall[0, k, :]
        ub_ref = uqall[1, k, :]
        for j in range(k + 1, nop):
            #print "j = ", j
            dun = norme(np.cross(un_ref, uqall[0, j, :]))
            dub = norme(np.cross(ub_ref, uqall[1, j, :]))
            dun_dub = dun + dub
            if dun_dub < 0.01:
                isdouble[j] = 1
    
    if verbose > 0: print(isdouble)

    ind0 = np.where(isdouble == 0)
    if verbose > 0: print(ind0[0])
    uqall = uqall[:, ind0[0], :]

    nop2 = 12

    st1 = np.zeros((nop2, 3, 3), float)
    hkl_n = uqall[0, :, :] * normehkl[0]
    hkl_b = uqall[1, :, :] * normehkl[1]
    if verbose > 0:
        print("n b schmid_tensor [line1, line2, line3]")
    for k in range(nop2):
        un_colonne = uqall[0, k, :].reshape(3, 1)
        ub_ligne = uqall[1, k, :].reshape(1, 3)
        st1[k, :, :] = np.dot(un_colonne, ub_ligne)        
        if verbose > 1:
            print(uqall[0, k, :] * normehkl[0], uqall[1, k, :] * normehkl[1], st1[k, :, :].reshape(1, 9).round(decimals=3))

    if returnmore == 0:
        return st1
    else:
        return st1,hkl_n, hkl_b

def read_stiffness_file(filestf:str, verbose:int=0): #29May13
    """
    # units = 1e11 N/m2
    # dans les fichiers stf de XMAS les cij sont en 1e11 N/m2
    """
    c_tensor = np.loadtxt(filestf, skiprows = 1)
    c_tensor = np.array(c_tensor, dtype = float)
    if verbose > 0:
        print('\nIn read_stiffn:ess_file()')
        print(filestf)
        print(np.shape(c_tensor))
        print("stiffness tensor C, 1e11 N/m2 (100 GPa) units")
        print(c_tensor)

    return c_tensor


def calc_cosines_first_stereo_triangle(matstarlab, axis_pole_sample) :  # , matrot, uqref_cr) : # return_matrix = "yes", return_cosines = "no") : #, xyz_sample_azimut):

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
    indgoodop = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 27,
                                    29, 31, 33, 35, 37, 39, 41, 43, 45, 47])

    #------------------ xyz-sample convention OR
    omega = 40.0
    omega = omega * math.pi / 180.0
    # rotation de omega autour de l'axe x pour repasser dans Rlab
    matrot = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(omega), -np.sin(omega)],
                        [0.0, np.sin(omega), np.cos(omega)]])

    hkl_3 = np.array([[0., 0., 1.], [1., 0., 1.], [1., 1., 1.]])
    uqref_cr = np.zeros((3, 3), float)
    # uqref_cr 3 vecteurs 001 101 111 en colonnes
    for i in range(3):
        uqref_cr[:, i] = hkl_3[i, :] / norme(hkl_3[i, :])

    cos01 = np.inner(uqref_cr[:, 0], uqref_cr[:, 1])
    cos02 = np.inner(uqref_cr[:, 0], uqref_cr[:, 2])
    cos12 = np.inner(uqref_cr[:, 1], uqref_cr[:, 2])

#         print "cosines between extremities of first triangle : "
#         print round(cos01, 3), round(cos02, 3), round(cos12, 3)

    cos0 = min(cos01, cos02)
    cos1 = min(cos01, cos12)
    cos2 = min(cos02, cos12)

#         print "minimum cos with 001 101 and 111 : "
#         print round(cos0, 3), round(cos1, 3), round(cos2, 3)

    # vectors normal to frontier planes of stereographic triangle

    uqn_b = np.cross(uqref_cr[:, 0], uqref_cr[:, 1])
    uqn_b = uqn_b / norme(uqn_b)
    uqn_g = np.cross(uqref_cr[:, 0], uqref_cr[:, 2])
    uqn_g = uqn_g / norme(uqn_g)
    uqn_r = np.cross(uqref_cr[:, 1], uqref_cr[:, 2])
    uqn_r = uqn_r / norme(uqn_r)
    # --   end of preliminary calculations

    upole_sample = axis_pole_sample / norme(axis_pole_sample)
    upole_lab = np.dot(matrot, upole_sample)
    # print "pole axis - sample coord : ", upole_sample
    # print "pole axis - lab coord : ", upole_lab.round(decimals=3)

    matstarlabOND = GT.matstarlab_to_matstarlabOND(matstarlab)

    mat = GT.matline_to_mat3x3(matstarlabOND)

    Bstar = CP.rlat_to_Bstar(CP.mat_to_rlat(matstarlab))
    matdef = GT.matline_to_mat3x3(matstarlab)

# #        # test
# #        matdef = GT.matline_to_mat3x3(matstarlab)
# #        print "matdef"
# #        print matdef.round(decimals=4)
# #        print "matdef recalc"
# #        print dot(mat,Bstar).round(decimals=4)

    cosangall = np.zeros((2 * nop, 3), float)
    ranknum = np.arange(2 * nop)
    opsym = 2 * list(range(nop))
    # print opsym
    matk_lab = np.zeros((2 * nop, 3, 3), float)

    for k in range(nop):
        matk_lab[k] = np.dot(mat, allop[k])
        # retour a un triedre direct si indirect
        if k not in indgoodop:
            matk_lab[k, :, 2] = -matk_lab[k, :, 2]
        uqrefk_lab = np.dot(matk_lab[k], uqref_cr)
        for j in range(3):
            cosangall[k, j] = np.inner(upole_lab, uqrefk_lab[:, j])
            cosangall[k + 48, j] = np.inner(-upole_lab, uqrefk_lab[:, j])

    # print('opsym',opsym)
    # print('cosangall',cosangall)
    # print('ranknum',ranknum)
    # print('lens opsym cosangall ranknum',len(opsym),len(cosangall),len(ranknum))

    data1 = np.column_stack((opsym, cosangall, ranknum))
    # print shape(data1)

    # priorites 001 101 111
    np1 = 1
    np2 = 2
    np3 = 3

    # print "opsym  cos001 cos101 cos111 ranknum"
    # print data1[:,1:4].round(decimals = 3)
    data1_sorted = sort_list_decreasing_column(data1, np1)

    # print data1_sorted.round(decimals = 3)

    ind1 = np.where(abs(data1_sorted[:, np1] - data1_sorted[0, np1]) < 1e-3)

    # print ind1

    data2_sorted = sort_list_decreasing_column(data1_sorted[ind1[0], :], np2)

    # print data2_sorted.round(decimals = 3)

    ind2 = np.where(abs(data2_sorted[:, np2] - data2_sorted[0, np2]) < 1e-3)

    # print ind2

    data3_sorted = sort_list_decreasing_column(data2_sorted[ind2[0], :], np3)

    # print data3_sorted.round(decimals = 3)

    ind3 = np.where(abs(data3_sorted[:, np3] - data3_sorted[0, np3]) < 1e-3)

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

    opsymres = np.array(opsymres, dtype=np.int16)
    # print opsymres

    if 0 in opsymres:
        op1 = 0
    else:
        op1 = opsymres[0]

    matONDnew = matk_lab[op1]
    opres = np.dot(np.linalg.inv(mat), matONDnew)
    # print "opres \n", opres.round(decimals=1)
    # print "det(opres)", np.linalg.det(opres)
    toto = np.dot(opres.T, Bstar)
    Bstarnew = np.dot(toto, opres)
    matdef2 = np.dot(matONDnew, Bstarnew)
    matstarlabnew = GT.mat3x3_to_matline(matdef2)

    abcstar_on_xyzsample = CP.matstarlab_to_matstarsample3x3(matstarlabnew)
    xyzsample_on_abcstar = np.linalg.inv(abcstar_on_xyzsample)

    # print('matdef', matdef)
    # print('np.linalg.inv(matdef)', np.linalg.inv(matdef))
    # print('matdef2', matdef2)
    # print('(np.dot(np.linalg.inv(matdef), matdef2).round(decimals=3))', (np.dot(np.linalg.inv(matdef), matdef2).round(decimals=3)))
    # print('np.dot(np.linalg.inv(matdef), matdef2)', np.dot(np.linalg.inv(matdef), matdef2))

    transfmat = np.linalg.inv(np.dot(np.linalg.inv(matdef), matdef2))

    # print "transfmat \n", transfmat

    # print "final : xyzsample_on_abcstar"
    # print xyzsample_on_abcstar.round(decimals=4)

    # print "matrix"
    # print "initial" , matstarlab.round(decimals=4)
    # print "final ", matstarlabnew.round(decimals=4)

    if verbose:
        print("op sym , rank, cos ", op1, ranknum[rankres[0]], cosangall[rankres[0]].round(decimals=3))

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
    rgb_pole = np.zeros(3, float)
    # blue : distance in q space between M tq OM = uq et le plan 001 101 passant par O
    rgb_pole[2] = abs(np.inner(uq, uqn_b)) / abs(np.inner(uqref_cr[:, 2], uqn_b))
    rgb_pole[1] = abs(np.inner(uq, uqn_g)) / abs(np.inner(uqref_cr[:, 1], uqn_g))
    rgb_pole[0] = abs(np.inner(uq, uqn_r)) / abs(np.inner(uqref_cr[:, 0], uqn_r))

    # convention OR
    rgb_pole = rgb_pole / max(rgb_pole)
    # convention Tamura
    # rgb_pole = rgb_pole / norme(rgb_pole)

    # print "rgb_pole :"
    # print rgb_pole

    return(matstarlabnew, transfmat, rgb_pole)

def mat_to_rlat(matstarlab):

    rlat = np.zeros(6, float)

    astarlab = matstarlab[0:3]
    bstarlab = matstarlab[3:6]
    cstarlab = matstarlab[6:9]
    rlat[0] = norme(astarlab)
    rlat[1] = norme(bstarlab)
    rlat[2] = norme(cstarlab)
    rlat[5] = np.arccos(np.inner(astarlab, bstarlab) / (rlat[0] * rlat[1]))
    rlat[4] = np.arccos(np.inner(cstarlab, astarlab) / (rlat[2] * rlat[0]))
    rlat[3] = np.arccos(np.inner(bstarlab, cstarlab) / (rlat[1] * rlat[2]))

    #print "rlat = ",rlat

    return rlat


def matstarlab_to_matdirlab3x3(matstarlab): #29May13

    rlat = mat_to_rlat(matstarlab)
    #print rlat
    vol = CP.vol_cell(rlat, angles_in_deg=0)

    astar1 = matstarlab[: 3]
    bstar1 = matstarlab[3: 6]
    cstar1 = matstarlab[6:]

    adir = np.cross(bstar1, cstar1) / vol
    bdir = np.cross(cstar1, astar1) / vol
    cdir = np.cross(astar1, bstar1) / vol

    matdirlab3x3 = np.column_stack((adir, bdir, cdir))

    return(matdirlab3x3, rlat)

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
    Bstar = np.zeros((3, 3), dtype=float)
    rlat = dlat_to_rlat(dlat)

    Bstar[0, 0] = rlat[0]
    Bstar[0, 1] = rlat[1] * np.cos(rlat[5])
    Bstar[1, 1] = rlat[1] * np.sin(rlat[5])
    Bstar[0, 2] = rlat[2] * np.cos(rlat[4])
    Bstar[1, 2] = -rlat[2] * np.sin(rlat[4]) * np.cos(dlat[3])
    Bstar[2, 2] = 1.0 / dlat[2]

    return Bstar

def matstarlab_to_matdirONDsample3x3(matstarlab,
                                     omega0=None, # was PAR.omega_sample_frame
                                     mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame): #29May13

    # uc unit cell
    # dir direct
    # uc_dir_OND : cartesian frame obtained by orthonormalizing direct unit cell

    matdirlab3x3, rlat = matstarlab_to_matdirlab3x3(matstarlab)
    # dir_bmatrix = uc_dir on uc_dir_OND

    dir_bmatrix = dlat_to_Bstar(rlat)

    # matdirONDlab3x3 = uc_dir_OND on lab

    matdirONDlab3x3 = np.dot(matdirlab3x3, np.linalg.inv(dir_bmatrix))

    if (omega0 is not None)&(mat_from_lab_to_sample_frame is None): # deprecated - only for retrocompatibility
        omega = omega0*np.pi / 180.0
        # rotation de -omega autour de l'axe x pour repasser dans Rsample
        mat_from_lab_to_sample_frame = np.array([[1.0, 0.0, 0.0],
                                        [0.0, np.cos(omega), np.sin(omega)],
                                        [0.0, -np.sin(omega), np.cos(omega)]])

    # matdirONDsample3x3 = uc_dir_OND on sample
    # rsample = matdirONDsample3x3 * ruc_dir_OND

    matdirONDsample3x3 = np.dot(mat_from_lab_to_sample_frame, matdirONDlab3x3)

    return matdirONDsample3x3

def transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab,
                                          tensor_crystal_line,
                                          omega0=None, # was PAR.omega_sample_frame,
                                          mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame):
    #29May13
    """
    start from stress or strain tensor
    as 6 coord vector
    """
    tensor_crystal_3x3 = epsline_to_epsmat(tensor_crystal_line)

    matdirONDsample3x3 = matstarlab_to_matdirONDsample3x3(matstarlab, omega0=omega0,
                                            mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame)

    # changement de base pour tenseur d'ordre 2

    toto = np.dot(tensor_crystal_3x3, matdirONDsample3x3.transpose())

    tensor_sample_3x3 = np.dot(matdirONDsample3x3, toto)

    tensor_sample_line = epsmat_to_epsline(tensor_sample_3x3)

    return tensor_sample_line

def matstarlab_to_deviatoric_strain_sample(matstarlab, 
                                omega0=None, # was PAR.omega_sample_frame,
                                mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame,
                                version=2,
                                returnmore=False,
                                elem_label="Ge"):
    #29May13
    epsp_crystal, _ = matstarlab_to_deviatoric_strain_crystal(matstarlab, 
                                            version=version,
                                            elem_label=elem_label)

    epsp_sample = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(matstarlab,
                                                epsp_crystal,
                                                omega0=omega0,
                                                mat_from_lab_to_sample_frame=mat_from_lab_to_sample_frame)
    if returnmore == False:
        return epsp_sample

    else: return(epsp_sample, epsp_crystal)   # add epsp_crystal

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
    gam_cryst = np.multiply(eps_crystal_line, fact1)
    sigma_crystal_line = np.dot(c_tensor, gam_cryst)
    #print eps_crystal_line
    #print gam_cryst
    #print sigma_crystal_line

    return sigma_crystal_line

def deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal_line):
    #29May13
    # cf formula (4.17) in book chapter by N. Tamura p 143
    # book "strain and dislocation gradients from diffraction"
    # eds R.I. Barabash and G.E. Ice
    sig = sigma_crystal_line*1.0
    von_mises = (sig[0]-sig[1])*(sig[0]-sig[1]) + \
                (sig[1]-sig[2])*(sig[1]-sig[2]) + \
                (sig[2]-sig[0])*(sig[2]-sig[0]) + \
                6.* (sig[3]*sig[3] + sig[4]*sig[4] + sig[5]*sig[5])
    von_mises = von_mises / 2.
    von_mises = np.sqrt(von_mises)
    return von_mises

def deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(sigma_crystal_line, schmid_tensors):
    #29May13
    nop2 = np.shape(schmid_tensors)[0]
    sigma_crystal_3x3 = epsline_to_epsmat(sigma_crystal_line)
    tau_all = np.zeros(nop2, float)
    for k in range(nop2):
        tau_all[k] = (np.multiply(schmid_tensors[k], sigma_crystal_3x3)).sum()

    return tau_all

def deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal_line):
    #29May13
    # cf formula (4.17) in book chapter by N. Tamura p 143
    # book "strain and dislocation gradients from diffraction"
    # eds R.I. Barabash and G.E. Ice
    sig = sigma_crystal_line*1.0
    von_mises = (sig[0]-sig[1])*(sig[0]-sig[1]) + \
                (sig[1]-sig[2])*(sig[1]-sig[2]) + \
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
    eq_strain = (2. / 3.) * np.sqrt(toto)
    return eq_strain

def add_columns_to_summary_file_new(filesum:str,
                            elem_label:str="Ge",
                            filestf=None,
                            omega_sample_frame:float=40.0,
                            verbose:int=0,
                            include_misorientation=0,
                            filefitref_for_orientation=None,  # seulement pour include_misorientation = 1
                            include_strain=1,  # 0 seulement pour mat2spots ou fit calib ou EBSD
                            # les 4 options suivantes seulement pour
                            #  include_misorientation = 1
                            # et filefitref_for_orientation = None
                            filter_mean_matrix_by_pixdev_and_npeaks=1,
                            maxpixdev_for_mean_matrix:float=0.25,
                            minnpeaks_for_mean_matrix:float=20,
                            filter_mean_matrix_by_intensity=0,
                            minintensity_for_mean_matrix:float=20000.0):  # 29May13

    """
    :param filesum: str, previously generated file (.dat) with build_summary
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
    data1, list_column_names, nameline0 = read_summary_file(filesum, verbose=verbose - 1)

    data_1 = np.array(data1, dtype=float)

    # print("data_1 in add_columns_to_summary_file_new()", data_1)
    # print("data_1.shape", data_1.shape)

    list_col_names2 = list_column_names

    list_col_names_orient = ["rgb_x_sample",
                            "rgb_y_sample",
                            "rgb_z_sample",
                            "rgb_x_lab",
                            "rgb_y_lab",
                            "rgb_z_lab"]

    number_col_orient = np.array([3, 3, 3, 3, 3, 3])

    for k in list(range(len(number_col_orient))):
        for i in list(range(number_col_orient[k])):
            str_column_name = list_col_names_orient[k] + "_" + str(i)
            list_col_names2.append(str_column_name)

    if include_strain:

        list_col_names_strain = ["strain6_crystal",
                                "strain6_sample",
                                "stress6_crystal",
                                "stress6_sample",
                                "res_shear_stress",
                                "max_rss",
                                "von_mises"]
        number_col_strain = np.array([6, 6, 6, 6, 12])

        for k in range(len(number_col_strain)):
            for i in range(number_col_strain[k]):
                toto = list_col_names_strain[k] + "_" + str(i)
                list_col_names2.append(toto)

        for k in range(len(number_col_strain), len(number_col_strain) + 2):
            list_col_names2.append(list_col_names_strain[k])

    header2 = ""
    for i in range(len(list_col_names2)):
        header2 = header2 + list_col_names2[i] + " "

    header = (nameline0+ ", rgb_x_sample, rgb_y_sample, rgb_z_sample, rgb_x_lab, rgb_y_lab, rgb_z_lab")

    if include_strain:
        header = (header
            + ", strain6_crystal,  strain6_sample, stress6_crystal, stress6_sample, res_shear_stress_12, max_rss, von_mises")

    if include_misorientation:
        header = header + ", misorientation_angle, w_mrad_0, w_mrad_1, w_mrad_2 \n"
        header2 = header2 + "misorientation_angle w_mrad_0 w_mrad_1 w_mrad_2 \n"
    else:
        header = header + "\n"
        header2 = header2 + "\n"

    if verbose > 0:
        print('some headers')
        print(header)
        print(header2)
        print(header2.split())

    schmid_tensors = glide_systems_to_schmid_tensors(verbose=verbose-1)

    if filestf != None:
        c_tensor = read_stiffness_file(filestf, verbose - 1)

    xsample_sample_coord = np.array([1.0, 0.0, 0.0])
    ysample_sample_coord = np.array([0.0, 1.0, 0.0])
    zsample_sample_coord = np.array([0.0, 0.0, 1.0])

    omegarad = omega_sample_frame * np.pi / 180.0
    ylab_sample_coord = np.array([0.0, np.cos(omegarad), -np.sin(omegarad)])
    zlab_sample_coord = np.array([0.0, np.sin(omegarad), np.cos(omegarad)])
    if verbose > 0:
        print('Sample frame description')
        print(":: x y z sample in sample frame : ", xsample_sample_coord,
                                                ysample_sample_coord,
                                                zsample_sample_coord)
        print(":: ylab zlab in sample frame : ", ylab_sample_coord, zlab_sample_coord)

    numig = np.shape(data_1)[0]

    if verbose > 0:
        print('data_1.shape', data_1.shape)

    # numig = 10

    rgb_x = np.zeros((numig, 3), float)
    rgb_y = np.zeros((numig, 3), float)
    rgb_z = np.zeros((numig, 3), float)
    rgb_xlab = np.zeros((numig, 3), float)
    rgb_ylab = np.zeros((numig, 3), float)
    rgb_zlab = np.zeros((numig, 3), float)
    if include_strain:
        epsp_crystal = np.zeros((numig, 6), float)
        epsp_sample = np.zeros((numig, 6), float)
        sigma_crystal = np.zeros((numig, 6), float)
        sigma_sample = np.zeros((numig, 6), float)
        tau1 = np.zeros((numig, 12), float)
        von_mises = np.zeros(numig, float)
        maxrss = np.zeros(numig, float)

    #    img 0 , gnumloc 1 , npeaks 2, pixdev 3, intensity 4, dxymicrons 5:7, matstarlab 7:16, strain6_crystal 16:22, euler 22:25
    indimg = list_column_names.index("img")  # 3
    indpixdev = list_column_names.index("pixdev")  # 3
    indnpeaks = list_column_names.index("npeaks")  # 2
    indmatstart = list_column_names.index("matstarlab_0")  # 7
    indintensity = list_column_names.index("intensity")  # 4
    if verbose > 0:
        print('info position')
        print(indpixdev, indnpeaks, indmatstart, indintensity)

    indmat = np.arange(indmatstart, indmatstart + 9)

    assert len(data_1.shape) == 2
    img_list = data_1[:, indimg]
    pixdev_list = data_1[:, indpixdev]
    npeaks_list = data_1[:, indnpeaks]
    intensity_list = data_1[:, indintensity]
    mat_list = data_1[:, indmat]

    if include_misorientation:
        indfilt2 = np.where(npeaks_list > 0.0)  # pour raccourcir le summary a la fin
        misorientation_angle = np.zeros(numig, float)
        omegaxyz = np.zeros((numig, 3), float)

        if filefitref_for_orientation == None:

            if filter_mean_matrix_by_pixdev_and_npeaks:
                print("filter matmean by pixdev and npeaks")
                indfilt = np.where(
                    (pixdev_list < maxpixdev_for_mean_matrix)
                    & (npeaks_list > minnpeaks_for_mean_matrix))
                matstarlabref = (mat_list[indfilt[0]]).mean(axis=0)
                print("number of points used to calculate matmean", len(indfilt[0]))

            elif filter_mean_matrix_by_intensity:
                print("filter matmean by intensity")
                indfilt = np.where((intensity_list > minintensity_for_mean_matrix)
                    & (npeaks_list > 0.0))
                matstarlabref = (mat_list[indfilt[0]]).mean(axis=0)
                print("number of points used to calculate matmean", len(indfilt[0]))

            else:
                matstarlabref = (mat_list[indfilt2[0]]).mean(axis=0)

            # TO REMOVE
        #            matmean = ((mat_list[indfilt[0]])[-10:]).mean(axis=0)   # test pour data Keckes

        else:
            matstarlabref, data_fit, calib, pixdev = F2TC.readlt_fit(filefitref_for_orientation,
                                                                readmore=True)

    #        matmean3x3 = GT.matline_to_mat3x3(matmean)

    k = 0
    for i in range(numig):
        if verbose - 1 > 0: print("i : ", i, "img_list[i] : ", img_list[i], "\r")
        if npeaks_list[i] > 0.0:
            matstarlab = mat_list[i, :]
            # print "x"
            matstarlabnew, transfmat, rgb_x[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, xsample_sample_coord)
            rgb_xlab[i, :] = rgb_x[i, :] * 1.0
            # print "y"
            matstarlabnew, transfmat, rgb_y[i, :] = calc_cosines_first_stereo_triangle(
                matstarlab, ysample_sample_coord)
            matstarlabnew, transfmat, rgb_ylab[i, :] = calc_cosines_first_stereo_triangle(matstarlab, ylab_sample_coord)
            # print "z"
            matstarlabnew, transfmat, rgb_z[i, :] = calc_cosines_first_stereo_triangle(matstarlab, zsample_sample_coord)
            matstarlabnew, transfmat, rgb_zlab[i, :] = calc_cosines_first_stereo_triangle(matstarlab, zlab_sample_coord)

            if include_strain:
                epsp_sample[i, :], epsp_crystal[i, :] = matstarlab_to_deviatoric_strain_sample(
                                                            matstarlab,
                                                            omega0=omega_sample_frame,
                                                            version=2,
                                                            returnmore=True,
                                                            elem_label=elem_label)

                sigma_crystal[i, :] = deviatoric_strain_crystal_to_stress_crystal(
                                                c_tensor, epsp_crystal[i, :])
                sigma_sample[i, :] = transform_2nd_order_tensor_from_crystal_frame_to_sample_frame(
                                        matstarlab, sigma_crystal[i, :], omega0=omega_sample_frame)

                von_mises[i] = deviatoric_stress_crystal_to_von_mises_stress(sigma_crystal[i, :])

                tau1[i, :] = deviatoric_stress_crystal_to_resolved_shear_stress_on_glide_planes(
                    sigma_crystal[i, :], schmid_tensors)
                maxrss[i] = abs(tau1[i, :]).max()

            if include_misorientation:
                #                mat2 = GT.matline_to_mat3x3(matstarlab)
                #                vec_crystal, vec_lab, misorientation_angle[i] = twomat_to_rotation(matmean3x3,mat2, verbose = 0)

                (vecRodrigues_sample, misorientation_angle[i]) = twomat_to_rotation_Emeric(
                                            matstarlabref, matstarlab, omega0=omega_sample_frame)
                omegaxyz[i, :] = vecRodrigues_sample * 2.0 * 1000.0  # unites = mrad
                # misorientation_angle : unites = degres
                if verbose - 1 > 0: print(round(misorientation_angle[i], 3), omegaxyz[i, :].round(decimals=2))

            if verbose > 0:
                print('matstarlab', matstarlab)
                if include_strain:
                    print("deviatoric strain crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (1e-3 units)")
                    print(epsp_crystal.round(decimals=2))
                    print("deviatoric strain sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (1e-3 units)")
                    print(epsp_sample[i, :].round(decimals=2))

                    print("deviatoric stress crystal : aa bb cc -dalf bc, -dbet ac, -dgam ab (100 MPa units)")
                    print(sigma_crystal[i, :].round(decimals=2))

                    print("deviatoric stress sample : xx yy zz -dalf yz, -dbet xz, -dgam xy (100 MPa units)")
                    print(sigma_sample[i, :].round(decimals=2))

                    print("Von Mises equivalent Stress (100 MPa units)",
                        round(von_mises[i], 3))
                    print("RSS resolved shear stresses on glide planes (100 MPa units) : ")
                    print(tau1[i, :].round(decimals=3))
                    print("Max RSS : ", round(maxrss[i], 3))
        k = k + 1

    # numig here for debug with smaller numig
    data_list = np.column_stack((data_1[:numig, :], rgb_x, rgb_y, rgb_z, rgb_xlab, rgb_ylab, rgb_zlab))

    if include_strain:
        data_list = np.column_stack((data_list, epsp_crystal, epsp_sample, sigma_crystal, sigma_sample, tau1, maxrss, von_mises))

    if include_misorientation:
        data_list = np.column_stack((data_list, misorientation_angle, omegaxyz))
        data_list = data_list[indfilt2[0], :]  # enleve les images avec zero grain indexe

    add_str = "_add_columns"
    if filefitref_for_orientation != None:
        add_str = add_str + "_use_orientref"
        # TO REMOVE
    #    add_str = add_str + "_use_mean_10_points"

    outfilesum = filesum.rstrip(".dat") + add_str + ".dat"
    if verbose > 0: print(outfilesum)
    outputfile = open(outfilesum, "w")
    # outputfile.write(header)
    # outputfile.write(header2)
    np.savetxt(outputfile, data_list, fmt="%.6f", header=header + header2, comments='')
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
            col1 = np.zeros(3, float)
            uq = (1.0 - range001[i]) * uqref_cr[:, 0] + range001[i] * (
                angrange[j] * uqref_cr[:, 1] + (1.0 - angrange[j]) * uqref_cr[:, 2])
            uq = uq / norme(uq)

            qsxy = hkl_to_xystereo(uq, down_axis=[0.0, -1.0, 0.0])
            # RGB coordinates
            rgb_pole = np.zeros(3, float)

            # blue : distance in q space between M tq OM = uq et le plan 001 101 passant par O
            rgb_pole[2] = abs(np.inner(uq, uqn_b)) / abs(np.inner(uqref_cr[:, 2], uqn_b))
            rgb_pole[1] = abs(np.inner(uq, uqn_g)) / abs(np.inner(uqref_cr[:, 1], uqn_g))
            rgb_pole[0] = abs(np.inner(uq, uqn_r)) / abs(np.inner(uqref_cr[:, 0], uqn_r))

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
    """
    used by plot_maps2

    # TODO: works well if slow axis dim is > 1 ...
    """

    # setup location of images in map based on xech yech + map pixel size
    # permet pixels rectangulaires
    # permet cartos incompletes
    # # BM32 : maps with dxech > 0 and dyech >0 : start at lower left corner on sample
    # d2scan xy not allowed only dscan x or dscan y

    print("\n\n  HELLO \n\n")

    data_1 = np.loadtxt(filexyz, skiprows=1)
    nimg = np.shape(data_1)[0]
    imglist = data_1[:, 0]
    print("first line :", data_1[0, :])
    print("last line : ", data_1[-1, :])

    xylist = data_1[:, 1:3] - data_1[0, 1:3]

    dxyfast = xylist[1, :] - xylist[0, :]
    print("dxyfast = ", dxyfast)
    dxymax = xylist[-1, :] - xylist[0, :]
    print("dxymax = ", dxymax)

    print("fast axis")
    indfast = np.where(abs(dxyfast) > 0.0)
    fast_axis = indfast[0][0]
    print(fast_axis)
    nintfast = dxymax[fast_axis] / dxyfast[fast_axis]
    print(nintfast)
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

    dxystep = dxyfast + dxyslow

    print("axis : fast , slow ", fast_axis, slow_axis)
    print("nb of points : fast, slow ", nptsfast, nptsslow)
    print("dxy step", dxystep)

    abs_step = np.abs(dxystep)
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

    if float(int(np.round(pix_r, 1))) < (pix_r - 0.01):
        print("non integer aspect ratio")
        for nmult in (2, 3, 4, 5):
            toto = float(nmult) * pix_r
            # print toto
            # print int(round(toto,1))
            if abs(float(int(np.round(toto, 1))) - toto) < 0.01:
                # print nmult
                break
        pix1 = nmult
        pix2 = int(np.round(float(nmult) * pix_r, 1))
        # print "map pixel size will be ", pix1, pix2
    else:
        pix1 = 1
        pix2 = int(round(pix_r, 1))

    print("pixel size for map (pix1= small, pix2= large):", pix1, pix2)

    large_axis = np.argmax(abs_step)
    small_axis = np.argmin(abs_step)

    # print large_axis, small_axis
    if large_axis == 1:
        pixsize = np.array([pix1, pix2], dtype=int)
    else:
        pixsize = np.array([pix2, pix1], dtype=int)
    print("pixel size for map dx dy", pixsize)

    # dx => columns, dy => lines
    if fast_axis == 0:
        nximg, nyimg = nptsfast, nptsslow
    else:
        nximg, nyimg = nptsslow, nptsfast

    map_imgnum = np.zeros((nyimg, nximg), int)

    print("map raw size ", np.shape(map_imgnum))

    impos_start = np.zeros(2, int)
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

    impos = np.zeros(2, int)  # y x

    # tableau normal : y augmente vers le bas, x vers la droite
    # xech yech : xech augmente vers la droite, yech augmente vers le hut

    # tester orientation avec niveaux gris = imgnum

    for i in range(nimg):
        # for i in range(200) :
        imnum = int(np.round(imglist[i], 0))
        impos[1] = xylist[i, 0] / abs(dxystep[0])
        impos[0] = -xylist[i, 1] / abs(dxystep[1])
        # print impos
        impos = impos_start + impos
        # print impos
        map_imgnum[impos[0], impos[1]] = imnum


    return map_imgnum, dxystep, pixsize, impos_start


# from matplotlib import mpl
# cmap = mpl.cm.PiYG

import matplotlib.cm as mpl

cmap = mpl.get_cmap("PiYG")
cmap = mpl.get_cmap("RdBu_r")
# cmap = mpl.cm.RdBu_r



DEFAULT_PLOTMAPS_PARAMETERS_DICT = {  # used by plot_maps2
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
    "color_for_max_strain_positive": np.array(
        [1.0, 0.0, 0.0]
    ),  # red  # [1.0,1.0,0.0]  # yellow
    "color_for_max_strain_negative": np.array([0.0, 0.0, 1.0]),  # blue
    "plot_grid": 1,
    "map_rotation": 0,
}


def plot_map_new2(dict_params, maptype, grain_index, App_parent=None):  # JSM May 2017
    """
    used by plot_maps2

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

    print("\n\nENTERING plot_map_new2()\n\n")

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
        "dalf"]

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
        "rgb_x_sample": [9, 9, 3, 1, 3, 0,
            ["x_sample", "y_sample", "z_sample", "x_sample", "y_sample", "z_sample", "x_sample", "y_sample", "z_sample"]],
        "orientation": [9, 9, 3, 1, 3, 0, ["x_sample",
                                        "y_sample",
                                        "z_sample",
                                        "x_sample",
                                        "y_sample",
                                        "z_sample",
                                        "x_sample",
                                        "y_sample",
                                        "z_sample",
            ]],
        "rgb_x_lab": [9, 9, 3, 1, 3, 0, ["x_lab", "y_lab", "z_lab"]],
        "strain6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "strain6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "stress6_crystal": [6, 18, 6, 2, 3, 3, ["aa", "bb", "cc", "ca", "bc", "ab"]],
        "stress6_sample": [6, 18, 6, 2, 3, 3, ["XX", "YY", "ZZ", "YZ", "XZ", "XY"]],
        "w_mrad": [3, 9, 3, 1, 3, 0, ["WX", "WY", "WZ"]],
        "res_shear_stress": [12, 36, 12, 3, 4, 8,
        ["rss0", "rss1", "rss2", "rss3", "rss4", "rss5", "rss6", "rss7", "rss8",
                "rss9",
                "rss10",
                "rss11"]],
        "max_rss": [1, 3, 1, 1, 1, 0, ["max_rss"]],
        "von_mises": [1, 3, 1, 1, 1, 0, ["von Mises stress"]],
        "misorientation_angle": [1, 3, 1, 1, 1, 0, ["misorientation angle"]],
        "intensity": [1, 3, 1, 1, 1, 0, ["intensity"]],
        "maxpixdev": [1, 3, 1, 1, 1, 0, ["maxpixdev"]],
        "stdpixdev": [1, 3, 1, 1, 1, 0, ["stdpixdev"]],
        "fit": [2, 6, 2, 1, 2, 0, ["npeaks", "pixdev"]],
        "dalf": [1, 3, 1, 1, 1, 0, ["delta_alf exp-theor"]]}

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

    print("data.shape", data_list.shape)
    print("shape = ((nb images)* nb grains , nb of data columns)")
    nbgrains = int(np.amax(data_list[:, 1]) + 1)
    nb_images = int(data_list.shape[0] / nbgrains)

    imgesindices = set()

    print("maximum nb of grains per image", nbgrains)
    #print("nb of images", nb_images)

    # sort data according to their grain number
    grains_data = []
    for g_ix in range(nbgrains):
        posg = np.where(data_list[:,1]==float(g_ix))[0]
        grains_data.append(np.take(data_list, posg, axis =0))

    print('grain 0 data', grains_data[0])
    #print('grain 1 data', grains_data[1])

    # print("first image of grains_data[0]", grains_data[0][0])
    # print("len(first image of grains_data[0])", len(grains_data[0][0]))
    # print("grains_data[0].shape", grains_data[0].shape)

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
        datasigntype = "relative" #  ??
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
    #TODO add mistorientation plot
    #         elif maptype == 'misorientation_angle':
    #             colmin, nbdatacolumns=63,1
    #         elif maptype == 'dalf':
    #             colmin, nbdatacolumns=64,1

    filexyz = d["File xyz"]
    map_imageindex_array, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz)

    # Normal convention
    map_imageindex_array = np.flipud(map_imageindex_array)

    # print('map_imageindex_array',map_imageindex_array)
    # print("map_imageindex_array.shape", map_imageindex_array.shape)

    nlines, ncol = map_imageindex_array.shape

    # print("nlines,ncol", nlines, ncol)
    #         print "z_values.shape",z_values.shape
    # print('nbdatacolumns',nbdatacolumns)
    # print('colmin',colmin)
    zvalues_Ncomponents = np.full((nlines*ncol,nbdatacolumns), np.NaN)

    grainsdata = grains_data[grain_index]

    # print('grainsdata[:4]',grainsdata[:4])

    expimagesindices = grainsdata[:,0]
    exp_ix = 0
    for k in range(nlines*ncol):
        if k in expimagesindices:
            zvalues_Ncomponents[k]=grainsdata[exp_ix][colmin : colmin + nbdatacolumns]
            exp_ix+=1  

    # print('DATA to be plot')
    # print(zvalues_Ncomponents[:4])

    zvalues_Ncomponents = np.ma.masked_invalid(zvalues_Ncomponents)

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

    

    for index_component in range(nbdatacolumns):

        columnname = plot_maptype_list[index_component]

        if maptype == "orientation":
            z_values = cosines_array.reshape((nlines, ncol, 9))[:, :, index_component]
            print("z_values.shape", z_values.shape)
            plot_maptype_list[index_component] = (
                str(list_vecs[index_component // 3]) + plot_maptype_list[index_component])
            colorbar_label = plot_maptype_list[index_component]
        elif datatype in ("scalar", "symetricscalar"):
            print("considered datatype=", datatype)
            colorbar_label = columnname
            zvalues = zvalues_Ncomponents[:, index_component]
            z_values = zvalues.reshape((nlines, ncol))

        elif datatype == "RGBvector":
            zvalues = zvalues_Ncomponents[
                :, index_component * 3 : (index_component + 1) * 3]
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

        ar_posmotor = np.reshape(ar_posmotor, (nlines, ncol, 2))

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
        plotobjet = ImshowFrame(App_parent, -1, "%s %s" % (maptype, columnname),
                                z_values, Imageindices=Tabindices1D,
                                nb_col=nb_col, nb_lines=nb_lines, stepindex=1,
                                boxsize_row=1, boxsize_line=1,
                                imagename=columnname, mosaic=0,
                                datatype=None,
                                dict_param=dict_param)
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
        "dxymicrons_0", "dxymicrons_1",
        "matstarlab_0", "matstarlab_1", "matstarlab_2", "matstarlab_3", "matstarlab_4", "matstarlab_5", "matstarlab_6", "matstarlab_7", "matstarlab_8",
        "strain6_crystal_0", "strain6_crystal_1", "strain6_crystal_2", "strain6_crystal_3", "strain6_crystal_4", "strain6_crystal_5",
        "euler3_0", "euler3_1", "euler3_2",
        "strain6_sample_0", "strain6_sample_1", "strain6_sample_2", "strain6_sample_3", "strain6_sample_4", "strain6_sample_5",
        "rgb_x_sample_0", "rgb_x_sample_1", "rgb_x_sample_2", "rgb_z_sample_0", "rgb_z_sample_1", "rgb_z_sample_2",
        "stress6_crystal_0", "stress6_crystal_1", "stress6_crystal_2", "stress6_crystal_3", "stress6_crystal_4", "stress6_crystal_5",
        "stress6_sample_0", "stress6_sample_1", "stress6_sample_2", "stress6_sample_3", "stress6_sample_4", "stress6_sample_5",
        "res_shear_stress_0", "res_shear_stress_1", "res_shear_stress_2", "res_shear_stress_3", "res_shear_stress_4", "res_shear_stress_5", "res_shear_stress_6", "res_shear_stress_7", "res_shear_stress_8", "res_shear_stress_9", "res_shear_stress_10", "res_shear_stress_11",
        "max_rss",
        "von_mises",
        "misorientation_angle",
        "dalf"]

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
    nb_images = data_list.shape[0] // nbgrains

    print("maximum nb of grains per image", nbgrains)
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

    numig = np.shape(data_list)[0]
    print(numig)
    ndata_cols = np.shape(data_list)[1]
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
                print("filtering out img with large misorientation > ",
                    d["max_misorientation"])
                print("nimg with low misorientation : ", shape(indm)[1])

        gnumlist = np.array(data_list[:, indgnumloc], dtype=int)
        pixdevlist = data_list[:, indpixdev]
        npeakslist = np.array(data_list[:, indnpeaks], dtype=int)
    else:
        gnumlist = np.zeros(numig, int)
        pixdevlist = np.zeros(numig, int)
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
        "res_shear_stress": [12, 36, 12, 3, 4, 8, ["rss0",
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
                                                    "rss11"]],
        "max_rss": [1, 3, 1, 1, 1, 0, ["max_rss"]],
        "von_mises": [1, 3, 1, 1, 1, 0, ["von Mises stress"]],
        "misorientation_angle": [1, 3, 1, 1, 1, 0, ["misorientation angle"]],
        "intensity": [1, 3, 1, 1, 1, 0, ["intensity"]],
        "maxpixdev": [1, 3, 1, 1, 1, 0, ["maxpixdev"]],
        "stdpixdev": [1, 3, 1, 1, 1, 0, ["stdpixdev"]],
        "fit": [2, 6, 2, 1, 2, 0, ["npeaks", "pixdev"]],
        "dalf": [1, 3, 1, 1, 1, 0, ["delta_alf exp-theor"]]}

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
        filexyz_new, xylim_new = rotate_map(filexyz, d["map_rotation"], xylim=d["xylim"])

    map_imageindex_array, dxystep, pixsize, impos_start = calc_map_imgnum(filexyz_new)

    nlines = shape(map_imageindex_array)[0]
    ncol = shape(map_imageindex_array)[1]
    nplot = dict_nplot[d["maptype"]][1]
    plotdat = np.zeros((nlines, ncol, nplot), float)
    datarray_info = np.zeros((nlines, ncol, nplot), float)
    ARRAY_INFO_FILLED = False

    print("grain : ", d["probed_grainindex"])
    print("npeakslist", npeakslist)
    if d["filter_on_pixdev_and_npeaks"]:
        print("filter_on_pixdev_and_npeaks")
        print("filtering :")
        print("maxpixdev ", d["maxpixdev_forfilter"])
        print("minnpeaks ", d["minnpeaks_forfilter"])
        indf = np.where((gnumlist == d["probed_grainindex"])
            & (pixdevlist < d["maxpixdev_forfilter"])
            & (npeakslist > d["minnpeaks_forfilter"]))
    elif d["filter_on_intensity"]:
        print("filter_on_intensity")
        indf = np.where((gnumlist == d["probed_grainindex"])
            & (npeakslist > 0)
            & (intensitylist > d["min_intensity_forfilter"]))
    else:
        print("default filtering")
        indf = where((gnumlist == d["probed_grainindex"]) & (npeakslist > 0))

    # filtered data
    data_list2 = data_list[indf[0], :]

    if d["maptype"] == "euler3":
        euler3 = data_list2[:, indcolplot]
        ang0 = 360.0
        ang1 = arctan(sqrt(2.0)) * 180.0 / np.pi
        ang2 = 180.0
        ang012 = np.array([ang0, ang1, ang2])
        print(euler3[0, :])
        euler3norm = euler3 / ang012
        print(euler3norm[0, :])
        # print min(euler3[:,0]), max(euler3[:,0])
        # print min(euler3[:,1]), max(euler3[:,1])
        # print min(euler3[:,2]), max(euler3[:,2])

    elif d["maptype"][:5] == "rgb_x":
        rgbxyz = data_list2[:, indcolplot]

    elif d["maptype"] == "fit":
        default_color_for_missing = np.array([1.0, 0.8, 0.8])
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
            default_color_for_missing = np.array([1.0, 0.8, 0.8])  # pink = color for missing data
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
                plotdat[:, :, 3 * j : 3 * (j + 1)] = 0.0  # black = color for missing data
            if maptype != "dalf":
                if maptype != "w_mrad":
                    print("xx xy xz yy yz zz")
                else:
                    print("wx wy wz")
            #                color_filtered = np.array([0.5,0.5,0.5])
            color_filtered = np.zeros(3, float)
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
            d["use_mrad_for_misorientation"] == "yes"):
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
        xylist = np.zeros((numig2, 2), float)

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
                npeaksmax_forplot - npeaksmin_forplot)

            # print 'pixdevlist2[i]',pixdevlist2[i]

            if d["low_npeaks_as_red_in_npeaks_map"] != None:
                if npeakslist2[i] < d["low_npeaks_as_red_in_npeaks_map"]:
                    plotdat[iref, jref, 0:3] = np.array([1.0, 0.0, 0.0])
            else:
                if npeakslist2[i] < npeaksmin_forplot:
                    plotdat[iref, jref, 0:3] = np.array([1.0, 0.0, 0.0])

            plotdat[iref, jref, 3:6] = (pixdevmax_forplot - pixdevlist2[i]) / (
                pixdevmax_forplot - pixdevmin_forplot)

            if d["high_pixdev_as_blue_and_red_in_pixdev_map"] != None:
                if pixdevlist2[i] > 0.25:
                    plotdat[iref, jref, 3:6] = np.array([0.0, 0.0, 1.0])
                if pixdevlist2[i] > 0.5:
                    plotdat[iref, jref, 3:6] = np.array([1.0, 0.0, 0.0])
            else:
                if pixdevlist2[i] > pixdevmax_forplot:
                    plotdat[iref, jref, 3:6] = np.array([1.0, 0.0, 0.0])
                if d["low_pixdev_as_green_in_pixdev_map"] != None:
                    if (pixdevlist2[i] < 0.25) & (npeakslist2[i] > 20):
                        plotdat[iref, jref, 3:6] = np.array([0.0, 1.0, 0.0])

            valnbpeaks = npeakslist2[i]
            valpixdev = pixdevlist2[i]
            datarray_info[iref, jref, :] = [valnbpeaks,
                                            valnbpeaks,
                                            valnbpeaks,
                                            valpixdev,
                                            valpixdev,
                                            valpixdev]
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
                        list_plot_max - list_plot_min)

            val_singlevalue = list_plot[i]
            datarray_info[iref, jref, :] = [val_singlevalue,
                                            val_singlevalue,
                                            val_singlevalue]

            ARRAY_INFO_FILLED = True

        else:
            for j in range(nb_values):
                if list_plot[i, j] > list_plot_max[j]:
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = d[
                        "color_for_max_strain_positive"]
                elif list_plot[i, j] < list_plot_min[j]:
                    plotdat[iref, jref, 3 * j : 3 * j + 3] = d[
                        "color_for_max_strain_negative"]
                else:
                    toto = (list_plot[i, j] - list_plot_min[j]) / (
                        list_plot_max[j] - list_plot_min[j])
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
    # #                    plotdat[iref, jref, :] = np.array([1., 0., 0.])
    # #                if imglist[i]==min(imglist)+ncol-1 :
    # #                    plotdat[iref, jref, :] = np.array([0., 1., 0.])
    # #                if imglist[i]==max(imglist) :
    # #                    plotdat[iref, jref, :] = np.array([0., 0., 1.])
    # plotdat[iref, jref, :] = np.array([1.0, 1.0, 1.0])*float(imglist[i])/max(imglist)
    # print plotdat[iref, jref, :]

    # extent corrected 06Feb13
    xrange1 = np.array([0.0, ncol * dxystep[0]])
    yrange1 = np.array([0.0, nlines * dxystep[1]])
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

        plo = ImshowFrame_Scalar(App_parent, -1, strname, z_values,
                                dataarray_info=ar_posmotor,
                                datatype=datatype,
                                xylabels=("dxech (microns)", "dyech (microns)"),
                                posmotorname=("Xsample", "Ysample"),
                                Imageindices=map_imageindex_array,
                                absolute_motorposition_unit="micron",
                                colorbar_label=colorbar_label)

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


def read_dict_grains(filegrains, dict_with_edges="no", dict_with_all_cols="no",
                                dict_with_all_cols2="no"):  # 29May13
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
    dict_values_names = ["grain size",
                        "ind_in_grain_list",
                        "ig_in_grain_list",
                        "img_in_grain_list",
                        "gnumloc_in_grain_list",
                        "mean_rgb",
                        "std_rgb *1000",
                        "range_rgb *1000"]

    if dict_with_edges == "yes":
        # 8 9 10 int
        # 11 float
        # 12 int
        # pixels des frontieres etendues, pixel_line_position pixel_column_position pixel_edge_type
        toto = ["list_line", "list_col", "list_edge", "gnumloc_mean", "list_edge_restricted"]
        dict_values_names = dict_values_names + toto

    if dict_with_all_cols == "yes":
        # 13 int
        # 14 : 48 float
        toto = ["npeaks", "pixdev", "intensity",
            "strain6_crystal_0", "strain6_crystal_1", "strain6_crystal_2", "strain6_crystal_3", "strain6_crystal_4", "strain6_crystal_5",
            "strain6_sample_0", "strain6_sample_1", "strain6_sample_2", "strain6_sample_3", "strain6_sample_4", "strain6_sample_5",
            "rgb_x_sample_0", "rgb_x_sample_1", "rgb_x_sample_2", "rgb_z_sample_0", "rgb_z_sample_1", "rgb_z_sample_2",
            "stress6_crystal_0", "stress6_crystal_1", "stress6_crystal_2", "stress6_crystal_3", "stress6_crystal_4", "stress6_crystal_5",
            "stress6_sample_0", "stress6_sample_1", "stress6_sample_2", "stress6_sample_3", "stress6_sample_4", "stress6_sample_5",
            "max_rss",
            "von_mises"]
        dict_values_names = dict_values_names + toto

    if dict_with_all_cols2 == "yes":
        # 48,49 : float
        toto = ["matstarlab_mean", "misorientation_angle"]
        dict_values_names = dict_values_names + toto

    ndict = len(dict_values_names)
    linepos_list = np.zeros(ndict + 1, dtype=int)

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
        list_neighbors = np.zeros(5, int)
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
    dict_neigh = {0: [1, 2, 3, 4],
                    1: [2, 3, 4],
                    2: [1, 3, 4],
                    4: [1, 2, 4],
                    8: [1, 2, 3],
                    5: [2, 4],
                    6: [1, 4],
                    9: [2, 3],
                    10: [1, 3]}

    # test
    # ngrains = 2
    dict_grains2 = {}
    # gnum0 = 106

    for gnum in range(ngrains):
        # for gnum in [gnum0,]:

        dict_grains2[gnum] = dict_grains[gnum]

        list_img = dict_grains[gnum][3]
        nimg = len(list_img)
        list_edge = np.zeros(nimg, dtype=int)
        list_edge_restricted = np.zeros(nimg, dtype=int)
        bitwise = np.array([1, 2, 4, 8])

        print("gnum = ", gnum)
        print(dict_values_names[3])
        print(dict_grains[gnum][3])
        print(dict_values_names[4])
        print(dict_grains[gnum][4])

        list_line = np.zeros(nimg, dtype=int)
        list_col = np.zeros(nimg, dtype=int)

        gnumloc_list = np.array(dict_grains[gnum][4], dtype=float)

        gnumloc_min = gnumloc_list.min()

        gnumloc_min = int(round(gnumloc_min, 0))
        print("gnumloc_min = ", gnumloc_min)
        gnumloc_mean = gnumloc_list.mean()
        print("gnumloc_mean = ", gnumloc_mean)
        # gnumloc_mean_int = int(gnumloc_mean + 0.5)
        # print "gnumloc_mean_int = ", gnumloc_mean_int

        for i in range(nimg):
            edge1 = np.zeros(4, dtype=int)
            edge2 = np.zeros(4, dtype=int)
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
    toto = ["list_line", "list_col", "list_edge", "gnumloc_mean", "list_edge_restricted"]
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
        "strain6_crystal_0", "strain6_crystal_1", "strain6_crystal_2", "strain6_crystal_3", "strain6_crystal_4", "strain6_crystal_5",
        "strain6_sample_0", "strain6_sample_1", "strain6_sample_2", "strain6_sample_3", "strain6_sample_4", "strain6_sample_5",
        "rgb_x_sample_0", "rgb_x_sample_1", "rgb_x_sample_2", "rgb_z_sample_0", "rgb_z_sample_1", "rgb_z_sample_2",
        "stress6_crystal_0", "stress6_crystal_1", "stress6_crystal_2", "stress6_crystal_3", "stress6_crystal_4", "stress6_crystal_5",
        "stress6_sample_0", "stress6_sample_1", "stress6_sample_2", "stress6_sample_3", "stress6_sample_4", "stress6_sample_5",
        "max_rss",
        "von_mises"]

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
                data_list[:, indcoladd].round(decimals=0), dtype=int)
            for key, value in dict_grains.items():
                print(key, value[indgrainsize_d])  # , "\n", value[3],"\n", value[4]
                list1 = []
                nimg = value[indgrainsize_d]
                for i in range(nimg):
                    ind1 = where(
                        (img_list == value[indimg_d][i])
                        & (gnumloc_list == value[indgnumloc_d][i]))
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
                        & (gnumloc_list == value[indgnumloc_d][i]))
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
        filegrains, dict_with_edges="yes", dict_with_all_cols="yes")

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
        matstarlab_ig = np.zeros((nimg, 9), dtype=float)
        img_list_d = np.array(value[indimg_d], dtype=int)
        gnumloc_list_d = np.array(value[indgnumloc_d], dtype=int)

        for i in range(nimg):
            ind1 = where(
                (img_list == img_list_d[i]) & (gnumloc_list == gnumloc_list_d[i]))
            # print ind1
            # print ind1[0][0]
            j = ind1[0][0]
            matstarlab_ig[i, :] = matstarlab_all[j, :]
            # print matstarlab_1

        matstarlab_mean = matstarlab_ig.mean(axis=0)
        # print "matmean = ", matstarlab_mean

        dict_grains2[key].append(matstarlab_mean.round(decimals=6))

        vec_crystal = np.zeros((nimg, 3), float)
        vec_lab = np.zeros((nimg, 3), float)
        angle1 = np.zeros(nimg, float)
        matmean3x3 = GT.matline_to_mat3x3(matstarlab_mean)
        for k in range(nimg):
            mat2 = GT.matline_to_mat3x3(matstarlab_ig[k, :])
            vec_crystal[k, :], vec_lab[k, :], angle1[k] = twomat_to_rotation(
                matmean3x3, mat2, verbose=0)
            # if k == 5 : return()

        dict_grains2[key].append(angle1.round(decimals=3))
        print("angle1 : mean, std, min, max")
        print(round(angle1.mean(), 3),
            round(angle1.std(), 3),
            round(angle1.min(), 3),
            round(angle1.max(), 3))

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
    dict_edge_lines = {0: [[1, 0], [1, 1]],
                        1: [[0, 0], [0, 1]],
                        2: [[0, 1], [1, 1]],
                        3: [[0, 0], [1, 0]]}

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


def plot_strain_stress_color_bar(bar_legend="strain"):

    # Make a colorbar as a separate figure. (for strain maps)
    cdict = {"red": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 0.0)),
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
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,
                                extend="both",
                                # ticks = [0.,0.1,0.2, 0.25],
                                ticks=[-0.2, 0.0, 0.2],
                                spacing="proportional",
                                orientation="horizontal")

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

    hkl4C = np.array(
        [[4.0, 0.0, -2.0], [2.0, 0.0, 0.0], [5.0, -1.0, -1.0], [4.0, -2.0, -4.0]]
    )
    xy4C = np.array(
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

    xy4all = np.zeros((numim, 8), float)

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

    # hkl = np.array([hkl4A[npic1,:],hkl4A[npic2,:]])
    # hkl = np.array([hkl4B[npic1,:],hkl4B[npic2,:]])
    hkl = np.array([hkl4C[npic1, :], hkl4C[npic2, :]])
    matstarlab_all = np.zeros((nimg, 9), float)

    xy2 = column_stack(
        (xy4all[:, npic1 * 2 : npic1 * 2 + 2], xy4all[:, npic2 * 2 : npic2 * 2 + 2])
    )

    isbadimg = np.zeros(nimg, int)
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

    vec_crystal = np.zeros((nimg, 3), float)
    vec_lab = np.zeros((nimg, 3), float)
    angle1 = np.zeros(nimg, float)
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

    uflabyz = np.array([0.0, uflab[1], uflab[2]])
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

    uilab = np.array([0.0, 1.0, 0.0])

    xbetrad = xbet * PI / 180.0
    xgamrad = xgam * PI / 180.0

    cosbeta = cos(PI / 2.0 - xbetrad)
    sinbeta = sin(PI / 2.0 - xbetrad)
    cosgam = cos(-xgamrad)
    singam = sin(-xgamrad)

    uflab_cen2 = np.zeros(3, float)
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

    # uflab1 = np.array([-uflab[0],uflab[1],uflab[2]])

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

    # uflabyz = np.array([0.0, uflab1[1],uflab1[2]])
    # chi = angle entre uflab et la projection de uflab sur le plan ylab, zlab

    # chi = (180.0/PI)*arctan(uflab1[0]/norme(uflabyz))
    # twicetheta = (180.0/PI)*arccos(uflab1[1])
    # th0 = twicetheta/2.0

    # print "2theta, theta, chi en deg", twicetheta , chi, twicetheta/2.0
    # print "xcam, ycam = ", xcam, ycam

    xycam = np.array([xcam, ycam])

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
        uflab_cen = np.array([-1.0, 0.0, 0.0])
    if diagr == "top":
        uflab_cen = np.array([0.0, 0.0, 1.0])
    if diagr == "halfback":  # 2theta = 118
        # 0 -sin28 cos28
        tth = 28 * math.pi / 180.0
        uflab_cen = np.array([0.0, -sin(tth), cos(tth)])

    uflab_cen = uflab_cen / norme(uflab_cen)

    uilab =  np.array([0.0, 1.0, 0.0])

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

    hkl = np.zeros((nmaxspots, 3), int)
    uflab = np.zeros((nmaxspots, 3), float)
    xy = np.zeros((nmaxspots, 2), float)
    Etheor = np.zeros(nmaxspots, float)
    ththeor = np.zeros(nmaxspots, float)
    tth = np.zeros(nmaxspots, float)
    chi = np.zeros(nmaxspots, float)

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
                        if ((cryst_struct == "FCC")
                            | ((cryst_struct == "diamond")
                                & ((H % 2) | ((not H % 2) & (not (H + K + L) % 4))))
                            | ((cryst_struct == "BCC") & (not (H + K + L) % 2))):
                            # print "hkl =", H,K,L
                            qlab = (float(H) * mat[0: 3]
                                + float(K) * mat[3: 6]
                                + float(L) * mat[6:])
                            if norme(qlab) > 1.0e-5:
                                uqlab = qlab / norme(qlab)
                                cosangle2 = inner(uqlab, uqlab_cen)
                                sintheta = -inner(uqlab, uilab)
                                if (sintheta > 0.0) & (cosangle2 > cosangle):
                                    # print "reachable reflection"
                                    Etheor[nspot] = (DictLT.E_eV_fois_lambda_nm
                                        * norme(qlab)
                                        / (2 * sintheta))
                                    ththeor[nspot] = (180.0 / math.pi) * arcsin(
                                        sintheta)
                                    # print "Etheor = ", Etheor[nspot]
                                    if (Etheor[nspot] > (Emin * 1000.0)) & (
                                        Etheor[nspot] < (Emax * 1000.0)):
                                        uflabtheor = uilab + 2 * sintheta * uqlab
                                        chi[nspot], tth[nspot] = uflab_to_2thetachi(
                                            uflabtheor)
                                        if (diagr == "side") & (chi[nspot] > 0.0):
                                            chi[nspot] = chi[nspot] - 180.0
                                        test = inner(uflabtheor, uflab_cen)
                                        # print "hkl =", H,K,L
                                        # print "uflabtheor.uflab_cen = ",test
                                        if test > cosangle:
                                            hkl[nspot, :] =  np.array([H, K, L])
                                            uflab[nspot, :] = uflabtheor
                                            # top diagram use xbet xgam close to zero
                                            xy[nspot, :] = uflab_to_xycam_gen(
                                                uflab[nspot, :],
                                                calib,
                                                uflab_cen,
                                                pixelsize=pixelsize)
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
