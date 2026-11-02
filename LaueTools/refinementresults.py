# -*- coding: utf-8 -*-
"""
Results of a unit cell refinement (orientation and strain) from the refined UB matrix: computation and text report.

Shared by the GUIs DetectorCalibration and PlotRefineGUI (each displays the report in its own window); usable from
scripts and notebooks: no GUI dependency.

    results = strain_results(UBmat, "Cu")
    texts = results_texts(results, residues, nb_pairs, filename="image_0001.cor", folder="/data")
"""
import time

import numpy as np

from . import CrystalParameters as CP
from . import dict_LaueTools as DictLT
from . import generaltools as GT

LATTICE_PARAMETER_NAMES = ["  a", "  b", "  c", "alpha", "beta", "gamma"]
HKLXYZ_LABELS = ["x=[100]_LT :", "y=[010]_LT :", "z=[001]_LT :",
                 "xs=[100]_LTsample :", "ys=[010]_LTsample :", "zs=[001]_LTsample :"]


def strain_results(UBmat, key_material, constantlength="a", dictmaterials=None, verbose=0):
    """strain, refined lattice parameters and orientation matrices from a refined UB matrix

    :param UBmat: array 3x3, refined UB matrix (q = UB B0 G*)
    :param key_material: str, material label (sets the reference lattice parameters and the elastic constants)
    :param constantlength: str, "a", "b" or "c": lattice length kept fixed during the refinement
    :param dictmaterials: dict of materials (default: dict_LaueTools.dict_Materials)
    :param verbose: int, 0: only the comparison of sample-frame strains (temporary) and the equivalent strain
        are printed
    :return: dict with keys
        latticeparams (reference), B0matrix, UBmat, UBB0mat,
        devstrain (deviatoric strain, crystal frame), devstrain_sampleframe,
        devstrain_sampleframe_old (TEMPORARY-STRAINFIX, value computed before Oct. 2026),
        devstrain_LTframe (x 1000, rounded), equivalentstrain (von Mises, frame independent),
        new_latticeparameters (refined), fullstrain_results (full strain and stress assuming stress_zz = 0),
        fullstrain_text, Umat (orthonormal) and Bmat_tri (upper triangular) such that UBmat = Umat Bmat_tri,
        HKLxyz_names, HKLxyz (HKL coordinates of the lab and sample frame axes),
        UB_new, B0_new, scale, rotdev (UB and B0 referred to the refined lattice parameters)
    """
    if dictmaterials is None:
        dictmaterials = DictLT.dict_Materials
    latticeparams = dictmaterials[key_material][1]
    B0matrix = CP.calc_B_RR(latticeparams)
    UBmat = np.array(UBmat, dtype=np.float64)

    devstrain, _ = CP.compute_deviatoricstrain(UBmat, B0matrix, latticeparams)
    # lattice lengths rescaled so that constantlength keeps its reference value
    new_latticeparameters = CP.computeLatticeParameters_from_UB(UBmat, key_material, constantlength,
                                                                dictmaterials=dictmaterials)
    if verbose > 0:
        print("final lattice_parameter_direct_strain", new_latticeparameters)

    # strain of compute_deviatoricstrain() is in direct crystal frame (x // a)
    devstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(devstrain, UBmat,
                                                                    latticeparams=latticeparams)
    # TEMPORARY-STRAINFIX: OLD deviatoric strain in sample frame as computed before Oct. 2026
    devstrain_sampleframe_old = CP.strain_from_crystal_to_sample_frame_OLD(devstrain, UBmat)
    CP.print_devstrain_sample_comparison(devstrain_sampleframe_old, devstrain_sampleframe,
                                         label=key_material, verbose=1)

    # full strain and stress assuming stress_zz = 0 in sample frame (needs elastic constants of material)
    fullstrain_results = CP.fullstrain_from_deviatoricstrain(devstrain, UBmat, key_material,
                                                             dictmaterials=dictmaterials)
    fullstrain_text = CP.fullstrain_text(fullstrain_results, key_material)
    if CP.PRINT_FULLSTRAIN:
        print(fullstrain_text)

    devstrain_LTframe = np.round(CP.strain_from_crystal_to_LaueToolsframe(devstrain, UBmat,
                                                                         latticeparams=latticeparams) * 1000,
                                 decimals=3)
    if verbose > 0:
        print('====> **** devstrain_LTframe', devstrain_LTframe)
        print('*************************************\n')

    equivalentstrain = CP.equivalent_strain(devstrain)
    print("Equivalent (von Mises) strain (10-3 units, frame independent): %.3f" % (equivalentstrain * 1000))

    UBB0mat = np.dot(UBmat, B0matrix)

    # UBmat = Umat Bmat_tri, Umat orthonormal, Bmat_tri upper triangular
    Umat = CP.matstarlab_to_matstarlabOND(matstarlab=None, matLT3x3=UBmat)
    Bmat_tri = np.dot(np.transpose(Umat), UBmat)
    if verbose > 0:
        print("**********test U ****************************")
        print("U matrix = ")
        print(Umat.round(decimals=9))
        print("norms :")
        for i in range(3):
            print(i, GT.norme_vec(Umat[:, i]).round(decimals=5))
        print("scalar products")
        for i in range(3):
            j = np.mod(i + 1, 3)
            print(i, j, np.inner(Umat[:, i], Umat[:, j]).round(decimals=5))
        print("determinant")
        print(np.linalg.det(Umat).round(decimals=5))
        print(" Bmat_triang_up= ")
        print(Bmat_tri.round(decimals=9))

    HKLxyz_names, HKLxyz = CP.matrix_to_HKLs_along_xyz_sample_and_along_xyz_lab(matstarlab=None,
                                                                                UBmat=UBB0mat,
                                                                                omega=None,
                                                                                mat_from_lab_to_sample_frame=None,
                                                                                results_in_OR_frames=0,
                                                                                results_in_LT_frames=1,
                                                                                sampletilt=40.0,
                                                                                verbose=max(verbose - 1, 0))

    # UB and B0 referred to the refined lattice parameters (new material with these lattice parameters)
    UB_new, B0_new, scale, rotdev = CP.refined_UB_B0(UBmat, B0matrix, new_latticeparameters)

    return {"latticeparams": latticeparams, "B0matrix": B0matrix, "UBmat": UBmat, "UBB0mat": UBB0mat,
            "devstrain": devstrain, "devstrain_sampleframe": devstrain_sampleframe,
            "devstrain_sampleframe_old": devstrain_sampleframe_old, "devstrain_LTframe": devstrain_LTframe,
            "equivalentstrain": equivalentstrain, "new_latticeparameters": new_latticeparameters,
            "fullstrain_results": fullstrain_results, "fullstrain_text": fullstrain_text,
            "Umat": Umat, "Bmat_tri": Bmat_tri, "HKLxyz_names": HKLxyz_names, "HKLxyz": HKLxyz,
            "UB_new": UB_new, "B0_new": B0_new, "scale": scale, "rotdev": rotdev}


def max_strain(results):
    """largest absolute deviatoric strain component (10-3 units, rounded to 1e-6), crystal and sample frames"""
    return max(np.amax(np.fabs(np.round(results["devstrain"] * 1000, decimals=3))),
               np.amax(np.fabs(np.round(results["devstrain_sampleframe"] * 1000, decimals=3))))


def _matrix_text(title, matrix):
    """title line, then the matrix as a python nested list (8 decimals)"""
    txt = title + "["
    for row in np.asarray(matrix):
        txt += "[%.8f, %.8f, %.8f],\n" % tuple(row)
    return txt[:-2] + "]"


def results_texts(results, residues, nb_pairs, filename="", folder="", depth=None, Tsresults=None):
    """report of strain_results() as texts, by section (keys used by the results window FitResultsBoard)

    :param results: dict from strain_results()
    :param residues: array, non weighted residues (pixel deviations) of the refined spots
    :param nb_pairs: int, number of spots used in the refinement
    :param filename, folder: str, peaks list file name and folder
    :param depth: float, depth (microns) of the scattering grain normal to the sample surface, or None (not shown)
    :param Tsresults: refinement results of the transform Ts (Tsresults[2]: 3x3 Ts), or None
    :return: dict of texts
    """
    texts = {}

    txt = "Filename: %s\t\t\tDate: %s\t\tPlotRefineGUI.py\n" % (filename, time.asctime())
    txt += "Folder: %s\n" % folder
    txt += "Mean Pixel Deviation: %.3f\n" % np.mean(residues)
    txt += "Number of refined Laue spots: %d\n" % nb_pairs
    if depth is not None:
        txt += "Normal sample depth (microns): %.1f\n" % depth
    texts["NbspotsResidues"] = txt

    txt = "Deviatoric Strain (10-3 units) in crystal frame (direct space) \n"
    for row in np.round(results["devstrain"] * 1000, decimals=3):
        txt += "%.3f   %.3f   %.3f\n" % tuple(row)
    txt += ("Equivalent (von Mises) strain (10-3 units, frame independent): %.3f\n"
            % (results["equivalentstrain"] * 1000))
    texts["devstrain_crystal"] = txt

    txt = "Deviatoric Strain (10-3 units) in sample frame (tilt=40deg)\n"
    for row in np.round(results["devstrain_sampleframe"] * 1000, decimals=3):
        txt += "%.3f   %.3f   %.3f\n" % tuple(row)
    # TEMPORARY-STRAINFIX: OLD values (before Oct. 2026, wrong for non orthogonal cells)
    txt += "[TEMPORARY] OLD values (before Oct. 2026, wrong for hexagonal, trigonal, monoclinic, triclinic)\n"
    for row in np.round(results["devstrain_sampleframe_old"] * 1000, decimals=3):
        txt += "%.3f   %.3f   %.3f\n" % tuple(row)
    texts["devstrain_sample"] = txt

    texts["fullstrain_sample"] = results["fullstrain_text"]

    for key, title, values in (("Initial lattice", "Initial Lattice Parameters\n", results["latticeparams"]),
                               ("Refined lattice", "Refined Lattice Parameters\n",
                                results["new_latticeparameters"])):
        txt = title
        for name, val in zip(LATTICE_PARAMETER_NAMES, values):
            txt += "%s\t\t%.6f\n" % (name, val)
        texts[key] = txt

    if Tsresults is not None:
        Ts = Tsresults[2]
        Tsvalues = [Ts[0, 0], Ts[0, 1], Ts[0, 2], Ts[1, 1], Ts[1, 2], Ts[2, 2]]
        Tsmat = [[Ts[0, 0], Ts[0, 1], Ts[0, 2]],
                 [0.0, Ts[1, 1], Ts[1, 2]],
                 [0.0, 0.0, Ts[2, 2]]]
        txt = "Final Transform Parameters\n"
        txt += "".join(["  Ts00", "  Ts01", "  Ts02", " Ts11", " Ts12", " Ts22"])
        txt += " = [" + ", ".join("%.6f" % val for val in Tsvalues) + "]\n"
        texts["Ts parameters"] = _matrix_text(txt + "Ts matrix in q = P Ts P-1 U B0 G*\n", Tsmat)

    texts["UBmatrix"] = _matrix_text("UB matrix in q = UB B0 G*\n", results["UBmat"])
    texts["B0matrix"] = _matrix_text("B0 matrix in q = UB B0 G*\n", results["B0matrix"])
    # UB and B0 referred to the refined lattice parameters
    texts["UBmatrix"] += _matrix_text("\n\nUB_new in q = UB_new B0_new G* (refined lattice parameters), "
                                      "UB_new = UB B0 inv(B0_new)\n"
                                      "= scale x rotation: scale %.6f, max |R^T R - I| %.1e\n"
                                      % (results["scale"], results["rotdev"]), results["UB_new"])
    texts["B0matrix"] += _matrix_text("\n\nB0_new in q = UB_new B0_new G* (from the refined lattice parameters)\n",
                                      results["B0_new"])

    txt = "                                 HKL frame coordinates\n"
    for label, hkl in zip(HKLXYZ_LABELS, results["HKLxyz"]):
        txt += label + "\t [%.3f, %.3f, %.3f]\n" % tuple(hkl)
    texts["HKLxyz_names"] = txt
    print(txt)
    texts["HKLxyz"] = ""

    return texts
