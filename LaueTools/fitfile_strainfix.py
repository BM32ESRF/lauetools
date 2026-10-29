# -*- coding: utf-8 -*-
r"""
TEMPORARY-STRAINFIX: rewrite the strain of .fit files written before Oct. 2026

The deviatoric strain in sample frame of .fit files written before Oct. 2026 is wrong for non orthogonal
cells (hexagonal, trigonal, monoclinic, triclinic) and slightly asymmetric for all cells (see
CrystalParameters.strain_from_crystal_to_sample_frame2()). The strain blocks of these files are recomputed
from their UB and B0 matrices (reference lattice parameters from B0, or a given one, e.g. Zr_ref) and written as by the current
IOLaueTools.writefitfile(): deviatoric strain in crystal and sample frames, equivalent strain, full strain and
stress (stress_zz=0, when elastic constants of the material are known). The OLD sample frame deviatoric strain
of the file is kept in the [TEMPORARY] block. The spots table and all other blocks are copied unchanged.

usage:
    import LaueTools.fitfile_strainfix as FSF
    FSF.rewrite_fitfiles_strain("/data/Zr/fitfiles/*_g*.fit")        # results in /data/Zr/fitfiles/strainfix/
    FSF.rewrite_fitfiles_strain("/data/Zr/fitfiles/*_g*.fit", reference={"Zr": "Zr_ref"})   # strain w.r.t. Zr_ref

or from a shell:
    python -m LaueTools.fitfile_strainfix "/data/Zr/fitfiles/*_g*.fit" [--outputdir DIR] [--inplace]
                                          [--reference Zr:Zr_ref] [--reference Cr:Cr_ref]
"""
import os
import sys
import glob
import time
import argparse

import numpy as np

try:
    from . import CrystalParameters as CP
    from . import IOLaueTools as IOLT
except ImportError:
    import CrystalParameters as CP
    import IOLaueTools as IOLT

# labels (without '#') of all the footer blocks written by IOLaueTools.writefitfile() (and before Oct. 2026)
FOOTER_LABELS = ("UB matrix in q=", "Umatrix in q_lab", "Bmatrix in q_lab", "(B-I)*1000", "HKL coord.",
                 "B0 matrix in q=", "UBB0 matrix", "Euler angles", "matstarlab ,", "matstarsample ,",
                 "deviatoric strain in crystal frame", "deviatoric strain in direct crystal frame",
                 "deviatoric strain in sample2 frame", "unstrained reference material of the strain",
                 "equivalent (von Mises) strain", "[TEMPORARY]", "full strain in sample2 frame",
                 "hydrostatic strain", "lattice parameters from full strain", "stress in sample2 frame",
                 "new lattice parameters", "refined UB matrix", "refined B0 matrix", "CCDLabel",
                 "DetectorParameters", "pixelsize", "Frame dimensions", "Refined T transform")

# blocks recomputed by rewrite_fitfile_strain() (written by IOLaueTools.fitfile_strain_footer())
STRAIN_LABELS = ("deviatoric strain in crystal frame", "deviatoric strain in direct crystal frame",
                 "deviatoric strain in sample2 frame", "unstrained reference material of the strain",
                 "equivalent (von Mises) strain", "[TEMPORARY]", "full strain in sample2 frame",
                 "hydrostatic strain", "lattice parameters from full strain", "stress in sample2 frame")

OLD_SAMPLE_LABEL = "[TEMPORARY] OLD deviatoric strain"


def _split_footer_blocks(footerlines):
    """list of [label, value lines] (lines without '#' and end of line) of the footer of a .fit file"""
    blocks = []
    for line in footerlines:
        text = line[1:]
        if text.startswith(FOOTER_LABELS) or not blocks:
            blocks.append([text, []])
        else:
            blocks[-1][1].append(text)
    return blocks


def _values(lines, nb):
    values = " ".join(lines).replace("[", " ").replace("]", " ").split()
    return np.array([float(v) for v in values[:nb]], dtype=np.float64)


def _find_block(blocks, label):
    for block in blocks:
        if block[0].startswith(label):
            return block
    return None


def stiffness_key(key_material, stiffness_material=None):
    r"""
    key of dict_Stiffness of the elastic constants used for the full strain of key_material:
    stiffness_material (str, or dict {key_material: key}) if given, else key_material, else the part of
    key_material before the first '_' (e.g. 'Zr' for 'Zr_ref', 'Zr_ech15_labXRD'). None if no data
    """
    if isinstance(stiffness_material, dict):
        stiffness_material = stiffness_material.get(key_material)
    for key in (stiffness_material, key_material, str(key_material).split("_")[0]):
        if key is not None and key in CP.dict_Stiffness:
            return key
    return None


def rewrite_fitfile_strain(fitfilename, outputfilename=None, sampletilt=40.0, force=False,
                           key_material=None, reference=None, stiffness_material=None,
                           dictmaterials=None, verbose=0):
    r"""
    Rewrite the strain blocks of a .fit file written before Oct. 2026 (one grain per file, e.g. *_g0.fit)

    :param fitfilename: path of the .fit file
    :param outputfilename: path of the rewritten file (default: same name in a 'strainfix' subfolder).
        Can be fitfilename itself (file overwritten)
    :param sampletilt: sample surface tilt (deg) used to compute the sample frame (40 on BM32)
    :param force: rewrite also a file already written with the corrected strain (Oct. 2026 and later)
    :param key_material: reference material when the file gives none (no 'Element' nor reference material)
    :param reference: unstrained reference lattice of the strain instead of the one of B0 of the file:
        key of dictmaterials (e.g. 'Zr_ref'), 6 lattice parameters (Angstrom, deg), or dict {material of the
        file: key or lattice parameters} (e.g. {'Zr': 'Zr_ref'}, other materials keep the lattice of B0).
        The UB and B0 blocks of the file are kept (q = UB B0 G* unchanged); 'new lattice parameters' is
        recomputed with the length a of the reference
    :param stiffness_material: key of dict_Stiffness for the full strain (str or dict {key_material: key}),
        default: see stiffness_key()
    :param dictmaterials: materials dict for the reference keys (default: dict_LaueTools.dict_Materials)
    :return: dict with 'status' ('rewritten', 'already corrected', 'no UB or B0') and, when rewritten,
        'outputfilename', 'devstrain_sample' and 'devstrain_sample_old' (3x3, 10-3 unit), 'maxdiff'
        (max abs difference OLD - corrected, 10-3 unit)
    """
    with open(fitfilename, "r") as f:
        lines = f.read().splitlines()

    # footer: comment lines after the spots table
    lastdataline = max([k for k, line in enumerate(lines) if line.strip() and not line.startswith("#")],
                       default=None)
    if lastdataline is None:
        return {"status": "no spots"}
    headerlines, footerlines = lines[:lastdataline + 1], lines[lastdataline + 1:]
    endlines = []
    while footerlines and footerlines[-1].strip() in ("", "#"):
        endlines.insert(0, footerlines.pop())
    blocks = _split_footer_blocks(footerlines)

    already_corrected = (_find_block(blocks, "equivalent (von Mises) strain") is not None
                         or _find_block(blocks, OLD_SAMPLE_LABEL) is not None)
    if already_corrected and not force:
        return {"status": "already corrected"}

    ub_block = _find_block(blocks, "UB matrix in q=")
    b0_block = _find_block(blocks, "B0 matrix in q=")
    if ub_block is None or b0_block is None:
        return {"status": "no UB or B0"}
    UBmat = _values(ub_block[1], 9).reshape((3, 3))
    B0matrix = _values(b0_block[1], 9).reshape((3, 3))

    # reference material: written since Oct. 2026, else Element of the header, else key_material
    ref_block = _find_block(blocks, "unstrained reference material of the strain")
    if ref_block is not None and ref_block[1]:
        key_material = ref_block[1][0].strip()
    else:
        for k, line in enumerate(headerlines[:-1]):
            if line.strip() == "#Element":
                key_material = headerlines[k + 1][1:].strip()
                break

    # OLD sample frame strain, as written in the file (10-3 unit)
    old_block = _find_block(blocks, OLD_SAMPLE_LABEL)
    if old_block is None:
        old_block = _find_block(blocks, "deviatoric strain in sample2 frame")
    devstrain_sample_old = None if old_block is None else _values(old_block[1], 9).reshape((3, 3))

    # reference lattice: of B0, or given (UB referred to its B0: UB B0 = UB_ref B0_ref)
    ref = reference.get(key_material) if isinstance(reference, dict) else reference
    if ref is None:
        latticeparams = CP.directlatticeparameters_fromBmatrix(B0matrix)
        reftext = "reference lattice parameters from B0"
    else:
        if isinstance(ref, str):
            if dictmaterials is None:
                dictmaterials = CP.dict_Materials
            key_material, latticeparams = ref, list(dictmaterials[ref][1])
        else:
            latticeparams = [float(v) for v in ref]
        reftext = "reference lattice parameters of %s %s" % (key_material, latticeparams)
        B0ref = CP.calc_B_RR(latticeparams)
        UBmat = np.dot(np.dot(UBmat, B0matrix), np.linalg.inv(B0ref))
        B0matrix = B0ref

    # corrected strain (as in CP.evaluate_strain_fromUBmat())
    devstrain, _ = CP.compute_deviatoricstrain(UBmat, B0matrix, latticeparams)
    devstrain_sample = CP.strain_from_crystal_to_sample_frame2(devstrain, UBmat, sampletilt=sampletilt,
                                                               latticeparams=latticeparams)
    newlatticelines = None
    if ref is not None:
        newlattice = CP.computeLatticeParameters_from_UB(UBmat, "ref", "a",
                                                         dictmaterials={"ref": ["ref", latticeparams]})
        newlatticelines = [str(np.round(np.array(newlattice, dtype=np.float64), decimals=7))]
    dict_strain = {"devstrain_crystal": devstrain,
                   "devstrain_sample": devstrain_sample,
                   "equivalent_strain": CP.equivalent_strain(devstrain)}
    stiffkey = None
    if key_material is not None:
        dict_strain["reference_material"] = key_material
        stiffkey = stiffness_key(key_material, stiffness_material)
    if stiffkey is not None:
        res = CP.fullstrain_from_deviatoricstrain(devstrain, UBmat, key_material,
                                                  dictmaterials={key_material: [key_material, latticeparams]},
                                                  Cvoigt=CP.get_stiffness_matrix(stiffkey),
                                                  sampletilt=sampletilt)
        if res is not None:
            dict_strain["fullstrain_sample"] = res["fullstrain_sample"]
            dict_strain["hydrostaticstrain"] = res["hydrostaticstrain"]
            dict_strain["latticeparameters_fullstrain"] = res["latticeparameters"]
            dict_strain["stress_sample"] = res["stress_sample"]
    if devstrain_sample_old is not None:
        dict_strain["devstrain_sample_old"] = devstrain_sample_old * 1e-3

    strainlines = IOLT.fitfile_strain_footer(dict_strain).splitlines()
    strainlines.append("[TEMPORARY] strain blocks rewritten at %s by fitfile_strainfix from %s "
                       "(sample tilt %.1f deg, %s, elastic constants of %s)"
                       % (time.asctime(), os.path.basename(fitfilename), sampletilt, reftext, stiffkey))

    # strain blocks replaced at the place of the old ones (else before 'new lattice parameters')
    newfooter = []
    inserted = False
    for label, values in blocks:
        if label.startswith(STRAIN_LABELS):
            if not inserted:
                newfooter += strainlines
                inserted = True
            continue
        if not inserted and label.startswith(("new lattice parameters", "refined UB matrix", "CCDLabel",
                                              "DetectorParameters", "pixelsize", "Frame dimensions")):
            newfooter += strainlines
            inserted = True
        if newlatticelines is not None and label.startswith("new lattice parameters"):
            values = newlatticelines
        newfooter += [label] + values
    if not inserted:
        newfooter += strainlines

    if outputfilename is None:
        outputfilename = os.path.join(os.path.dirname(os.path.abspath(fitfilename)), "strainfix",
                                      os.path.basename(fitfilename))
    outputdir = os.path.dirname(os.path.abspath(outputfilename))
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    with open(outputfilename, "w") as f:
        f.write("\n".join(headerlines + ["#" + line for line in newfooter] + endlines) + "\n")

    result = {"status": "rewritten", "outputfilename": outputfilename,
              "devstrain_sample": devstrain_sample * 1000.0, "devstrain_sample_old": devstrain_sample_old,
              "maxdiff": None, "key_material": key_material, "stiffness_material": stiffkey,
              "latticeparameters_reference": latticeparams}
    if devstrain_sample_old is not None:
        result["maxdiff"] = float(np.amax(np.fabs(devstrain_sample_old - devstrain_sample * 1000.0)))
    if verbose > 0:
        print("%s -> %s (max |OLD - corrected| = %s 10-3 unit)" % (fitfilename, outputfilename,
              "%.3f" % result["maxdiff"] if result["maxdiff"] is not None else "-"))
    return result


def rewrite_fitfiles_strain(fitfiles, outputdir=None, inplace=False, sampletilt=40.0, force=False,
                            key_material=None, reference=None, stiffness_material=None, dictmaterials=None,
                            verbose=1):
    r"""
    Rewrite the strain blocks of several .fit files written before Oct. 2026 (see rewrite_fitfile_strain())

    :param fitfiles: glob pattern (e.g. '/data/Zr/fitfiles/*_g*.fit'), folder (all its .fit files) or list of paths
    :param outputdir: folder of the rewritten files (default: 'strainfix' subfolder of each file folder)
    :param inplace: True to overwrite the files (outputdir is then ignored)
    :param key_material, reference, stiffness_material, dictmaterials: see rewrite_fitfile_strain()
    :return: dict {fitfile: result of rewrite_fitfile_strain()}
    """
    if isinstance(fitfiles, str):
        if os.path.isdir(fitfiles):
            fitfiles = os.path.join(fitfiles, "*.fit")
        fitfiles = sorted(glob.glob(fitfiles, recursive=True))
    # jupyter checkpoints copies and files already rewritten in 'strainfix' subfolders
    fitfiles = [fitfile for fitfile in fitfiles
                if not {".ipynb_checkpoints", "strainfix"} & set(os.path.normpath(fitfile).split(os.sep)[:-1])]

    results = {}
    for fitfile in fitfiles:
        if inplace:
            outputfilename = fitfile
        elif outputdir is not None:
            outputfilename = os.path.join(outputdir, os.path.basename(fitfile))
        else:
            outputfilename = None
        try:
            results[fitfile] = rewrite_fitfile_strain(fitfile, outputfilename, sampletilt=sampletilt,
                                                      force=force, key_material=key_material, reference=reference,
                                                      stiffness_material=stiffness_material,
                                                      dictmaterials=dictmaterials, verbose=verbose - 1)
        except (ValueError, IndexError, KeyError, OSError) as err:
            results[fitfile] = {"status": "error: %s" % err}

    if verbose > 0:
        statuses = [res["status"] for res in results.values()]
        maxdiffs = [res["maxdiff"] for res in results.values() if res.get("maxdiff") is not None]
        print("%d .fit files: %d rewritten, %d already corrected, %d skipped"
              % (len(statuses), statuses.count("rewritten"), statuses.count("already corrected"),
                 len(statuses) - statuses.count("rewritten") - statuses.count("already corrected")))
        nofull = sorted({str(res.get("key_material")) for res in results.values()
                         if res["status"] == "rewritten" and res.get("stiffness_material") is None})
        if nofull:
            print("no full strain (no elastic constants in dict_Stiffness) for materials: %s" % ", ".join(nofull))
        if maxdiffs:
            print("max |OLD - corrected| deviatoric strain in sample frame (10-3 unit): median %.3f, max %.3f"
                  % (np.median(maxdiffs), np.amax(maxdiffs)))
        for fitfile, res in results.items():
            if res["status"] not in ("rewritten", "already corrected"):
                print("  %s: %s" % (fitfile, res["status"]))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rewrite the strain of .fit files written before Oct. 2026 "
                                     "(OLD sample frame deviatoric strain kept in a [TEMPORARY] block)")
    parser.add_argument("fitfiles", nargs="+", help="glob pattern(s), folder(s) or .fit files")
    parser.add_argument("--outputdir", default=None, help="folder of rewritten files (default: <folder>/strainfix)")
    parser.add_argument("--inplace", action="store_true", help="overwrite the .fit files")
    parser.add_argument("--sampletilt", type=float, default=40.0, help="sample tilt in deg (default 40)")
    parser.add_argument("--force", action="store_true", help="rewrite also files already corrected")
    parser.add_argument("--material", default=None, help="reference material of files without Element")
    parser.add_argument("--reference", action="append", default=None, metavar="MATERIAL:REFERENCE",
                        help="reference lattice of the strain for a material of the files, e.g. Zr:Zr_ref "
                        "(key of dict_Materials), repeatable (default: lattice of B0 of each file)")
    parser.add_argument("--stiffness", default=None, help="key of dict_Stiffness for the full strain "
                        "(default: material, else its part before '_', e.g. Zr for Zr_ref)")
    args = parser.parse_args(argv)

    reference = None
    if args.reference:
        reference = dict(item.split(":", 1) for item in args.reference)

    fitfiles = []
    for item in args.fitfiles:
        if os.path.isdir(item):
            item = os.path.join(item, "*.fit")
        fitfiles += sorted(glob.glob(item, recursive=True))
    rewrite_fitfiles_strain(fitfiles, outputdir=args.outputdir, inplace=args.inplace,
                            sampletilt=args.sampletilt, force=args.force, key_material=args.material, reference=reference,
                            stiffness_material=args.stiffness)


if __name__ == "__main__":
    sys.exit(main())
