# -*- coding: utf-8 -*-
"""
This module belongs to Lauetools project

JS Micha Feb 2012

https://gitlab.esrf.fr/micha/lauetools/

#   this module contains general functions to index and refine data and
#   calibrate detector (on top geometry)

"""
import os
import sys
import time as ttt

import scipy.io
import numpy as np

if sys.version_info.major == 3:
    from . import LaueGeometry as F2TC
    from . import indexingAnglesLUT as INDEX
    from . import indexingSpotsSet as ISS
    from . import lauecore as LAUE
    from . import FitOrient as FitO
    from . import readmccd as rmccd
    from . import IOLaueTools as RWASCII
    from . import generaltools as GT
    from . import CrystalParameters as CP
    from . import dict_LaueTools as DictLT
    from . matchingrate import SpotLinks
else:
    import LaueGeometry as F2TC
    import indexingAnglesLUT as INDEX
    import indexingSpotsSet as ISS
    import lauecore as LAUE
    import FitOrient as FitO
    import readmccd as rmccd
    import IOLaueTools as RWASCII
    import generaltools as GT
    import CrystalParameters as CP
    from matchingrate import SpotLinks
    import dict_LaueTools as DictLT

CST_ENERGYKEV = DictLT.CST_ENERGYKEV


def Non_Indexed_Spots(indexed_spots, n):
    # dictionary of exp spots
    list_nonindexed = []
    for k in range(n):
        if indexed_spots[k][-1] == 0:
            list_nonindexed.append(k)
    return list_nonindexed

def SimulateResult_withMiller(grain, emin, emax, defaultParam, pixelsize=165.0 / 2048):
    """Simulate Laue Pattern for 1 grain

    Returns Twicetheta, Chi, Miller_ind, posx, posy, Energy arrays

    #TODO : use better function from laue6.py
    """
    detectordiameter = 165.0

    # if fastcompute = 0 array(indices) = 0 and TwicethetaChi is a list of spot object
    # k_direction = (Z>0)
    # TODO: to be argument if camera is far from preset kf_direction!!

    key_material = grain[3]
    grain = CP.Prepare_Grain(key_material, grain[2])

    # array(vec) and array(indices)  of spots exiting the crystal in 2pi steradian (Z>0)
    Spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=0,
                                verbose=0,
                                kf_direction="Z>0")

    # 2theta, chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(Spots2pi,
                                        fileOK=0,
                                        fastcompute=0,
                                        detectordistance=defaultParam[0],  # TODO: check if it is really the parameter set we need
                                        detectordiameter=detectordiameter * 1.2,
                                        kf_direction="Z>0",
                                        HarmonicsRemoval=1,
                                        pixelsize=pixelsize)

    Twicetheta = [spot.Twicetheta for spot in TwicethetaChi[0]]
    Chi = [spot.Chi for spot in TwicethetaChi[0]]
    Miller_ind = [list(spot.Millers) for spot in TwicethetaChi[0]]
    Energy = [spot.EwaldRadius * CST_ENERGYKEV for spot in TwicethetaChi[0]]

    posx, posy = F2TC.calc_xycam_from2thetachi(Twicetheta,
                                                Chi,
                                                defaultParam,
                                                verbose=0,
                                                pixelsize=pixelsize)[:2]

    return Twicetheta, Chi, Miller_ind, posx, posy, Energy


def simulate_theo(grain, emax, emin, paramDet, pixelsize, dim):
    """
    simulate theo for calibration purpose

    TODO: merge with SimulateResult_withMiller()
    """

    detectordistance = paramDet[0]
    detectordiameter = dim[0] * pixelsize * 3.0
    print("LAA.simulate_theo detectordiameter = ", detectordiameter)
    if abs(paramDet[3]) > 5.0:
        detectordiameter = 4.0 * detectordiameter
        print("detector diameter = ", detectordiameter)
        print("large xbet angle for diamond camera at 2theta = 118 deg")
        print("detectordiameter increased in LaueAutoAnalysis.py, def simulate_theo")
        print("in order to index all spots")

    kf_direction = "Z>0"

    key_material = grain[3]
    grain = CP.Prepare_Grain(key_material, grain[2])

    # array(vec) and array(indices)  of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grain],
                                fastcompute=0,
                                verbose=0,
                                kf_direction=kf_direction)

    # 2theta, chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi,
                                        fileOK=0,
                                        fastcompute=0,
                                        detectordistance=detectordistance,
                                        detectordiameter=detectordiameter,
                                        kf_direction=kf_direction)

    twicetheta = [spot.Twicetheta for spot in TwicethetaChi[0]]
    chi = [spot.Chi for spot in TwicethetaChi[0]]
    Miller_ind = [list(spot.Millers) for spot in TwicethetaChi[0]]

    posx, posy = F2TC.calc_xycam_from2thetachi(twicetheta,
                                                chi,
                                                paramDet[:5],
                                                verbose=0,
                                                pixelsize=pixelsize)[:2]

    return twicetheta, chi, Miller_ind, posx, posy


def RefineUB(linkedspots_link,
                linkExpMiller_link,
                linkIntensity_link,
                CCDcalib,
                data,
                starting_orientmatrix,
                Bmatrix,
                pixelsize=165.0 / 2048,
                dim=(2048, 2048),
                use_weights=True):
    """
    see OnRefinePicky in class picky_frame of LaueTool24.py

    CCDcalib = geometric detector parameters
    data = (2*select_theta, select_chi, select_I, DataPlot_filename)

    UP TO NOW: only strain and orientation simultaneously
    """
    linkedspots_fit = linkedspots_link
    linkExpMiller_fit = linkExpMiller_link
    linkIntensity_fit = linkIntensity_link
    linkResidues_fit = None

    print("\nStarting fit of strain and orientation from spots links ...\n")
    # print "Pairs of spots used",self.linkedspots
    arraycouples = np.array(linkedspots_fit)

    exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
    sim_indices = np.array(arraycouples[:, 1], dtype=np.int)

    nb_pairs = len(exp_indices)
    print("Nb of pairs: ", nb_pairs)

    # print "exp_indices, sim_indices",exp_indices, sim_indices

    # self.data_theo contains the current simulated spots: twicetheta, chi, Miller_ind, posx, posy
    # Data_Q = self.data_theo[2]  # all miller indices must be entered with sim_indices = arraycouples[:,1]

    # print "self.linkExpMiller",self.linkExpMiller
    Data_Q = np.array(linkExpMiller_fit)[:, 1:]
    sim_indices = np.arange(nb_pairs)  # for fitting function this must be an arange...
    # print "DataQ from self.linkExpMiller",Data_Q

    # initial parameters of calibration ----------------------
    calib = CCDcalib
    # print "detector parameters", calib

    # experimental spots selection -------------------------------------

    _twth, _chi = (np.take(data[0], exp_indices),
                    np.take(data[1], exp_indices))  # 2theta chi coordinates
    # pixel coordinates
    pixX, pixY, _th = F2TC.calc_xycam_from2thetachi(
        _twth, _chi, calib, verbose=0, pixelsize=pixelsize)

    # print "2*_th must be equal to _twth",(2*_th-_twth) < 0.000001 # this a test

    # print "nb_pairs",nb_pairs
    # print "indices of simulated spots (selection in whole Data_Q list)",sim_indices
    # print "Experimental pixX, pixY",pixX, pixY
    # print "starting_orientmatrix",starting_orientmatrix

    if use_weights:
        weights = linkIntensity_fit
    else:
        weights = None

    results = None
    # fitresults = False

    if 1:  # fitting procedure for one or many parameters

        calib = list(calib)

        initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
        allparameters = np.array(calib + [1, 1, 0, 0, 0] + [0, 0, 0])

        # nspots = np.arange(nb_pairs)

        arr_indexvaryingparameters = np.arange(5, 13)
        # print "\nInitial error--------------------------------------\n"
        residues, deltamat, _ = FitO.error_function_on_demand_strain(initial_values,
                                                                        Data_Q,
                                                                        allparameters,
                                                                        arr_indexvaryingparameters,
                                                                        sim_indices,
                                                                        pixX,
                                                                        pixY,
                                                                        initrot=starting_orientmatrix,
                                                                        Bmat=Bmatrix,
                                                                        pureRotation=0,
                                                                        verbose=1,
                                                                        pixelsize=pixelsize,
                                                                        dim=dim,
                                                                        weights=weights)
        # print "Initial residues",residues
        # print "---------------------------------------------------\n"

        results = FitO.fit_on_demand_strain(initial_values,
                                                Data_Q,
                                                allparameters,
                                                FitO.error_function_on_demand_strain,
                                                arr_indexvaryingparameters,
                                                sim_indices,
                                                pixX,
                                                pixY,
                                                initrot=starting_orientmatrix,
                                                Bmat=Bmatrix,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                verbose=0,
                                                weights=weights)

        # print "\n********************\n       Results of Fit        \n********************"
        # print "results",results

    if results is not None:

        # fitresults = True

        # print "\nFinal error--------------------------------------\n"
        residues, deltamat, _ = FitO.error_function_on_demand_strain(results,
                                                                        Data_Q,
                                                                        allparameters,
                                                                        arr_indexvaryingparameters,
                                                                        sim_indices,
                                                                        pixX,
                                                                        pixY,
                                                                        initrot=starting_orientmatrix,
                                                                        Bmat=Bmatrix,
                                                                        pureRotation=0,
                                                                        verbose=1,
                                                                        pixelsize=pixelsize,
                                                                        dim=dim,
                                                                        weights=weights)

        residues_non_weighted = FitO.error_function_on_demand_strain(results,
                                                        Data_Q,
                                                        allparameters,
                                                        arr_indexvaryingparameters,
                                                        sim_indices,
                                                        pixX,
                                                        pixY,
                                                        initrot=starting_orientmatrix,
                                                        Bmat=Bmatrix,
                                                        pureRotation=0,
                                                        verbose=1,
                                                        pixelsize=pixelsize,
                                                        dim=dim,
                                                        weights=None)[0]

        # print "Final residues",residues
        # print "---------------------------------------------------\n"
        # print "mean",np.mean(residues)

        # building B mat
        param_strain_sol = results
        varyingstrain = np.array([[1.0, param_strain_sol[2], param_strain_sol[3]],
                                    [0, param_strain_sol[0], param_strain_sol[4]],
                                    [0, 0, param_strain_sol[1]]])
        # print "varyingstrain results"
        # print varyingstrain

        # building UBmat (= newmatrix)
        UBmat = np.dot(np.dot(deltamat, starting_orientmatrix), varyingstrain)
        # print "UBmat",UBmat
        # print "newmatrix",newmatrix # must be equal to UBmat

        RR, PP = GT.UBdecomposition_RRPP(UBmat)

        # symetric matrix (strain) in direct real distance
        epsil = GT.epsline_to_epsmat(CP.calc_epsp(CP.dlat_to_rlat(CP.matrix_to_rlat(PP))))
        # print "epsil is already a zero trace symetric matrix ",np.round(epsil*1000, decimals = 2)
        # if the trace is not zero
        epsil = epsil - np.trace(epsil) * np.eye(3) / 3.0

        epsil = np.round(epsil * 1000, decimals=2)
        print("\n********************\n       result of Strain and Orientation  Fitting        \n********************")
        # print "last pixdev table non weighted"
        # print residues_non_weighted.round(decimals=3)
        print("Mean pixdev non weighted")
        pixdev = np.mean(residues_non_weighted).round(decimals=5)
        print(pixdev)
        print("\nPure Orientation Matrix")
        # print RR.tolist()
        print(RR)

        print("Deviatoric strain with respect to initial crystal unit cell (in 10-3 units)")
        # print epsil.tolist()
        print(epsil)

        # pas = np.array([[0,-1, 0],[0, 0, -1],[1, 0, 0]])
        # invpas = np.linalg.inv(pas)
        # print "good deviatoric strain epsil"  # ???? in other frame like XMAS
        # print np.dot(pas, epsil) # ????

        # update linked spots with residues

        linkResidues_fit = np.array([exp_indices, sim_indices, residues]).T

        # for further use
        newUBmat = UBmat
        newUmat = np.dot(deltamat, starting_orientmatrix)
        newBmat = varyingstrain
        deviatoricstrain = epsil

        return (newUBmat, newUmat, newBmat, deviatoricstrain, RR, linkResidues_fit, pixdev)


def GetStrainOrient(orientmatrix, Bmatrix, elem_label, emax_simul, veryclose_angletol, data, addoutput=0,
                    saturated=None,
                    use_weights=True,
                    addoutput2=0,
                    Emin=5.0,
                    detectorparameters=None,  # dico cf dans test_index_refine
                ):

    """
    refine UB

    returns:

    newUBmat : refined UB (that can be used later for simulation or more refinement
    deviatoricstrain:
    RR: pure rotation
    linkResidues_fit:  spots indices that have been used

    saturated = sat_flag list : to exclude saturated peaks from refining of UB

    """

    # modif 10Dec14

    #    vecteurref = np.eye(3) # means: a* // X, b* // Y, c* //Z
    #    grain = [vecteurref, [1, 1, 1], orientmatrix, elem_label]

    # PATCH: redefinition of grain to simulate any unit cell (not only cubic) ---
    #    key_material = elem_label # grain[3]
    grain = CP.Prepare_Grain(elem_label, orientmatrix)  # grain[2])

    # attention SimulateLaue attend une OrientMatrix avec norme(astar) = 1
    #  dans la matrice UBB0 de JSM la matrice UB a norme(astar) = 1
    # c'est cette convention qui est utilisee pour le calcul de l'energie des spots
    # donc pas robuste par rapport aux permutations de a b c dans le cubique quand UB est deformee
    # -----------------------------------------------------------------------------

    # theoretical spots
    #    Twicetheta_theo, Chi_theo, Miller_ind_theo, posx_theo, posy_theo, energy_theo = \
    #
    #            SimulateResult_withMiller(grain, 5, emax_simul,
    #                                        CCDcalib,
    #                                        pixelsize=165. / 2048,
    #                                        dim=(2048, 2048)
    #                                        )
    Emax = emax_simul

    (Twicetheta_theo,
        Chi_theo,
        Miller_ind_theo,
        _,
        _,
        energy_theo,
    ) = LAUE.SimulateLaue(grain,
                            Emin,
                            Emax,
                            detectorparameters["detectorparameters"],  # calib
                            kf_direction=detectorparameters["kf_direction"],
                            removeharmonics=1,
                            pixelsize=detectorparameters["pixelsize"],
                            dim=detectorparameters["dim"],
                            ResolutionAngstrom=False,
                            detectordiameter=detectorparameters["detectordiameter"] * 1.25)

    # experimental_spots
    twicetheta_exp, chi_exp, dataintensity_exp = data[:3]

    #    print "len(twicetheta_exp)", len(twicetheta_exp)
    #    print "twicetheta_exp, chi_exp",twicetheta_exp, chi_exp
    #    print "len(twicetheta_exp)", len(Twicetheta_theo)
    #    print "Twicetheta_theo, Chi_theo",Twicetheta_theo, Chi_theo

    #    p.figure()
    #    ax = p.subplot(111)
    #    p.plot(twicetheta_exp, chi_exp, 'ro')
    #    p.plot(Twicetheta_theo, Chi_theo, 'bx')
    #    ax.axis("equal")
    #    klmsdf

    # build automatically theo and exp spots associations (see OnAutoLink(self, evt) of picky_Frame of LaueTool24.py)

    res_assoc = SpotLinks(twicetheta_exp,
                                        chi_exp,
                                        dataintensity_exp,  # experimental data
                                        veryclose_angletol,  # tolerance angle
                                        Twicetheta_theo,
                                        Chi_theo,
                                        Miller_ind_theo,
                                        energy_theo)

    # print "len(assoc)",len(res_assoc[0])
    # print res_assoc[0]

    if not res_assoc:
        return 0
    else:

        # refine UB
        (refine_indexed_spots, linkedspots_link, linkExpMiller_link, linkIntensity_link,
        _, _, _) = res_assoc

        npairs = np.shape(linkedspots_link)[0]
        if 0:
            print("linkedspots_link")
            print(linkedspots_link)
            print(np.shape(linkedspots_link))
            print("refine_indexed_spots")
            for key, value in refine_indexed_spots.items():
                print(key, value)

        if npairs < 8:
            print("Need at least 8 indexed spots to refine the unit cell !!")
            return None

        if saturated != None:
            # print " linkedspots_link ", linkedspots_link
            # print "linkExpMiller_link ", linkExpMiller_link
            # print "linkIntensity_link ", linkIntensity_link
            # print "shapes ", np.shape(linkedspots_link), np.shape(linkExpMiller_link), np.shape(linkIntensity_link)
            print("indexed exp spots : total : ", np.shape(linkedspots_link)[0])
            # indices par rapport a la peak list initiale (celle de data)
            indsat = np.where(saturated == 1)
            # print "indsat[0] ", indsat[0]
            sat_link = np.zeros(np.shape(linkedspots_link)[0], dtype=int)
            for i in indsat[0]:
                # print "i = ", i
                ind2 = np.where(linkedspots_link[:, 0] == i)
                # print "ind2[0] = ", ind2[0]
                if len(ind2[0]) == 1:
                    sat_link[ind2[0][0]] = 1
                else:
                    print("exp peak indexed %d times", len(ind2[0]))
                    raise ValueError("problem ?")
            # indices par rapport a la liste de pics indexes (celle de linkedspots_link)
            # print "sat_link", sat_link
            indnonsat = np.where(sat_link == 0)[0]
            # print "indnonsat = ", indnonsat
            linkedspots_link = linkedspots_link[indnonsat, :]
            linkExpMiller_link = np.array(linkExpMiller_link, dtype=int)[indnonsat, :]
            linkIntensity_link = np.array(linkIntensity_link, dtype=float)[indnonsat]
            # print " linkedspots_link ", linkedspots_link
            # print "new shapes ", np.shape(linkedspots_link), np.shape(linkExpMiller_link), np.shape(linkIntensity_link)
            print("indexed exp spots : non saturated :", np.shape(linkedspots_link)[0])

            # reconvertir linkExpMiller_link et linkIntensity_link en listes et pas array ?  => semble OK sans

        res_refinement = RefineUB(linkedspots_link,
                                    linkExpMiller_link,
                                    linkIntensity_link,
                                    detectorparameters["detectorparameters"],  # CCDcalib,
                                    data,
                                    orientmatrix,
                                    Bmatrix,
                                    pixelsize=detectorparameters["pixelsize"],
                                    dim=detectorparameters["dim"],
                                    use_weights=use_weights)

        newUBmat, newUmat, newBmat, deviatoricstrain, RR, linkResidues_fit, pixdev = res_refinement

        nb_fitted_peaks = len(linkedspots_link)
        print("with %d fitted peaks" % nb_fitted_peaks)
        print("UBmat", newUBmat)
        print("pure rotation ", RR)
        print("deviatoric strain", deviatoricstrain)

        if addoutput2:
            return (newUBmat,
                deviatoricstrain,
                RR,
                linkResidues_fit,
                pixdev,
                nb_fitted_peaks,
                linkExpMiller_link,
                linkedspots_link,
                linkIntensity_link)
        else:
            if addoutput:
                return (newUBmat,
                    deviatoricstrain,
                    RR,
                    linkResidues_fit,
                    pixdev,
                    nb_fitted_peaks,
                    linkExpMiller_link)

            else:
                return (newUBmat,
                    newUmat,
                    newBmat,
                    deviatoricstrain,
                    RR,
                    linkResidues_fit,
                    pixdev,
                    nb_fitted_peaks,
                    linkExpMiller_link)


def test_indexation_refinement_Ge():
    """
    from .dat file and calib parameters do indexation and refinement

    """
    dirname = "./Examples/Ge/"
    DataPlot_filename = "Ge0001.dat"
    # CCDcalib = [69.17369978864516611, 1050.50098288478648101, 1115.53514707010231177,
    #                 0.13049762505535042, -0.23799600745091942]
    pixelsize = 165.0 / 2048

    twicetheta, chi, dataintensity, data_x, data_y = F2TC.Compute_data2thetachi(
                                        os.path.join(dirname, DataPlot_filename),
                                        sorting_intensity="yes",
                                        detectorparams=defaultParam,
                                        pixelsize=pixelsize)

    prefix, _ = DataPlot_filename.split(".")
    # writing .cor file
    RWASCII.writefile_cor(os.path.join(dirname, "C_" + prefix),
                            twicetheta,
                            chi,
                            data_x,
                            data_y,
                            dataintensity,
                            sortedexit=0,
                            param=defaultParam + [pixelsize],
                            initialfilename=DataPlot_filename)

    print("%s has been created with defaultparameter" % (prefix + ".cor"))
    print("%s" % str(defaultParam))
    # a C_#########.cor file is now created
    file_extension = "cor"
    DataPlot_filename = "C_" + prefix + "." + file_extension

    # Reading . cor file
    # TODO: put this function in readmccd for example
    # Current_peak_data = scipy.io.array_import.read_array(os.path.join(dirname, DataPlot_filename), lines=(1, -1))
    np.loadtxt(os.path.join(dirname, DataPlot_filename), skiprows=1)
    # nbcolumns == 5:
    data_theta = Current_peak_data[:, 0] / 2.0
    data_chi, data_pixX, data_pixY, data_I = np.transpose(Current_peak_data)[1:]

    # filename = DataPlot_filename

    # index_foundgrain = 0
    # dictionary of exp spots
    indexed_spots = {}
    for k in range(len(data_theta)):
        indexed_spots[k] = [k,  # index of experimental spot in .cor file
                            data_theta[k] * 2.0,
                            data_chi[k],  # 2theta, chi coordinates
                            data_pixX[k],
                            data_pixY[k],  # pixel coordinates
                            data_I[k],  # intensity
                            0]  # 0 means non indexed yet
    # last_orientmatrix_fromindexation = {}
    # last_Bmatrix_fromindexation = {}
    # last_epsil_fromindexation = {}

    # Running indexation

    # updated exp spot index to be still index
    current_exp_spot_index_list = Non_Indexed_Spots(indexed_spots, len(data_theta))

    # for each grain or indexation step: ******************************
    # ******************************************************************

    # compute angular distance between all exp; spots:

    nbspots_in_data = len(data_theta[current_exp_spot_index_list])

    # Matching spots set Size (MSSS):
    nbspotmaxformatching = nbspots_in_data

    # select 1rstly spots that have not been indexed and 2ndly reduced list by user
    index_to_select = np.take(current_exp_spot_index_list, np.arange(nbspotmaxformatching))

    select_theta = data_theta[index_to_select]
    select_chi = data_chi[index_to_select]
    select_I = data_I[index_to_select]
    # print select_theta
    # print select_chi
    listcouple = np.transpose(np.array([select_theta, select_chi]))
    Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

    # classical indexation parameters:
    data = (2 * select_theta, select_chi, select_I, DataPlot_filename)

    spot_index_central = [0, 1, 2, 3]  # central spot or list of spots
    nbmax_probed = 10  # 'Recognition spots set Size (RSSS): '
    elem_label = "UO2"
    energy_max = 25

    rough_tolangle = 0.5  # 'Dist. Recogn. Tol. Angle (deg)'
    fine_tolangle = 0.2  # 'Matching Tolerance Angle (deg)'
    Nb_criterium = 15  # 'Minimum Number Matched Spots: '
    print("rough_tolangle ", rough_tolangle)
    print("fine_tolangle ", fine_tolangle)
    NBRP = 2  # number of best result for each element of spot_index_central

    n = 3  # for LUT
    latticeparams = DictLT.dict_Materials[elem_label][1]
    B = CP.calc_B_RR(latticeparams)

    # indexation procedure
    bestmat, stats_res = INDEX.getOrientMatrices(spot_index_central,
                                                energy_max,
                                                Tabledistance[:nbmax_probed, :nbmax_probed],
                                                select_theta,
                                                select_chi,
                                                n,
                                                B,
                                                rough_tolangle=rough_tolangle,
                                                fine_tolangle=fine_tolangle,
                                                Nb_criterium=Nb_criterium,
                                                plot=0,
                                                structure_label=elem_label,
                                                nbbestplot=NBRP)

    # bestmat contains: NBRP * len(spot_index_central) matrix

    # Look for the very best matrix candidates: -----to be improved in readability ------------------------
    ar = []
    for elem in stats_res:
        ar.append([elem[0], -elem[2]])

    tabstat = np.array(ar)  # , dtype = [('x', '<i4'), ('y', '<i4')])
    rankmat = np.argsort(tabstat[:, 0])[::-1]  # ,order=('x','y'))

    verybestmat = bestmat[rankmat[0]]
    # -----------------------------------------------------------------------------------------------------
    # very best matrix
    orientmatrix = verybestmat

    # then get strain
    emax_simul = 16
    veryclose_angletol = 0.2

    (newUBmat, _, _, _,
    _, _, _, _) = GetStrainOrient(orientmatrix,
                                                                        B,
                                                                        elem_label,
                                                                        emax_simul,
                                                                        veryclose_angletol,
                                                                        data,
                                                                        detectorparameters=defaultParam,
                                                                        addoutput=0)
    # in this example the previous first step is useless
    emax_simul = 25
    veryclose_angletol = 0.1

    (_,
        _,
        _,
        deviatoricstrain_2,
        _,
        _,
        _,
        _,
    ) = GetStrainOrient(newUBmat, B, elem_label, emax_simul, veryclose_angletol,
                                                                    data,
                                                                    detectorparameters=defaultParam,
                                                                    addoutput=0)

    print("final strain")
    print(deviatoricstrain_2)


def RefineCalibParameters(linkedspots,
                            linkExpMiller,
                            linkIntensity,  # spots associations
                            data_x,
                            data_y,  # X, Y pixel exp data
                            matrix,
                            paramDet,
                            boolctrl,  # initial matrix guess, initial detector params, list of booleans for vary params
                            use_weights=True,
                            dim=(2048, 2048),
                            pixelsize=165.0 / 2048,
                            verbose=0,
                            addoutput=0):
    """
    fit detector parameters

    for varying parameters set True
    # dd, xcen, ycen, xbet, xgam a1,a2,a3
    boolctrl = '11000011'

    """

    arraycouples = np.array(linkedspots)

    exp_indices = np.array(arraycouples[:, 0], dtype=np.int)
    sim_indices = np.array(arraycouples[:, 1], dtype=np.int)

    nb_pairs = len(exp_indices)
    if verbose:
        print("Nb of pairs: ", nb_pairs)
        print(exp_indices, sim_indices)

    # self.data_theo contains the current simulated spots: twicetheta, chi, Miller_ind, posx, posy
    # Data_Q = self.data_theo[2]  # all miller indices must be entered with sim_indices = arraycouples[:,1]

    Data_Q = np.array(linkExpMiller)[:, 1:]

    sim_indices = np.arange(nb_pairs)
    if verbose:
        print("DataQ from self.linkExpMiller", Data_Q)
        print("self.linkExpMiller", linkExpMiller)

    # experimental spots selection from self.data_x, self.data_y (loaded when initialising calibFrame)
    pixX, pixY = (np.take(data_x, exp_indices), np.take(data_y, exp_indices))  # pixel coordinates
    # twth, chi = np.take(self.Data_X, exp_indices),np.take(self.Data_Y, exp_indices) # 2theta chi coordinates

    # initial parameters of calibration and misorientation from the current orientation matrix
    # print "Detector parameters in refine procedure",paramDet

    allparameters = np.array(paramDet + [0, 0, 0])  # 3 last params = 3 quaternion angles not used here

    # select the parameters that must be fitted
    # dd, xcen, ycen, xbet, xgam,  a1,a2,a3
    # boolctrl = '11001100'
    varyingparameters = []
    init_values = []
    for k, val in enumerate(boolctrl):
        if val == "1":
            varyingparameters.append(k)
            init_values.append(allparameters[k])

    if len(varyingparameters) == 0:
        print("You need to select at least one parameter to fit!!")
        return

    listparam = ["distance (mm)",
                "Xcen (pixel)",
                "Ycen (pixel)",
                "Angle1 (deg)",
                "Angle2 (deg)",  # detector parameter
                "theta1 (deg)",
                "theta2 (deg)",
                "theta3 (deg)"]  # misorientation with respect to initial matrix (/ elementary axis rotation)

    # start fit
    initial_values = np.array(init_values)  # [dd, xcen, ycen, ang1, ang2, theta1, theta2, theta3]
    arr_indexvaryingparameters = np.array(varyingparameters)  # indices of position of parameters in [dd, xcen, ycen, ang1, ang2, theta1, theta2, theta3]

    print("\n ***** \n starting fit of :", [listparam[k] for k in arr_indexvaryingparameters])
    print("With initial values: ", initial_values)
    if verbose:
        # print "miller selected ",np.take(self.data_theo[2],sim_indices, axis = 0) ????
        print("allparameters", allparameters)
        print("arr_indexvaryingparameters", arr_indexvaryingparameters)
        print("nb_pairs", nb_pairs)
        print("indices of simulated spots (selection in whole Data_Q list)", sim_indices)
        print("Experimental pixX, pixY", pixX, pixY)
        print("self.matrix", matrix)

    vecteurref = np.eye(3)  # use of Bmat will be handled later

    pureRotation = 1

    if use_weights:
        weights = linkIntensity
    else:
        weights = None

    # fitting procedure for one or many parameters
    nb_fittingparams = len(arr_indexvaryingparameters)
    if nb_pairs < nb_fittingparams:
        print("\n****************************************************************************")
        print("You need at least %d spots links to fit these %d parameters."
            % (nb_fittingparams, nb_fittingparams))
        print("****************************************************************************\n")
        return None, None, None

    # print "Initial error--------------------------------------\n"
    _, _, newmatrix = FitO.error_function_on_demand_calibration(initial_values,
                                                                        Data_Q,
                                                                        allparameters,
                                                                        arr_indexvaryingparameters,
                                                                        sim_indices,
                                                                        pixX,
                                                                        pixY,
                                                                        initrot=matrix,
                                                                        vecteurref=vecteurref,
                                                                        pureRotation=pureRotation,
                                                                        verbose=1,
                                                                        pixelsize=pixelsize,
                                                                        dim=dim,
                                                                        weights=weights)
    # print "Initial residues",residues
    # print "---------------------------------------------------\n"

    # print "Starting fit ------------"
    results = FitO.fit_on_demand_calibration(initial_values,
                                                Data_Q,
                                                allparameters,
                                                FitO.error_function_on_demand_calibration,
                                                arr_indexvaryingparameters,
                                                sim_indices,
                                                pixX,
                                                pixY,
                                                initrot=matrix,
                                                vecteurref=vecteurref,
                                                pureRotation=pureRotation,
                                                pixelsize=pixelsize,
                                                dim=dim,
                                                verbose=0,
                                                weights=weights)

    print("\n********************\n       Results of Fit        \n********************")
    # print "Fit Results",results
    allresults = allparameters

    if nb_fittingparams == 1:
        results = [results]

    # print "\n ----- Computation of fit results residues -------\n"

    # print "\n ----- weighted residues -------\n"
    _, _, newmatrix = FitO.error_function_on_demand_calibration(results,
                                                                        Data_Q,
                                                                        allparameters,
                                                                        arr_indexvaryingparameters,
                                                                        sim_indices,
                                                                        pixX,
                                                                        pixY,
                                                                        initrot=matrix,
                                                                        vecteurref=vecteurref,
                                                                        pureRotation=pureRotation,
                                                                        verbose=1,
                                                                        pixelsize=pixelsize,
                                                                        dim=dim,
                                                                        weights=weights)

    # print "residues",residues

    # print "\n ----- NON weighted residues -------\n"

    (residues_nonweighted,
    _delta, _newmatrix, _) = FitO.error_function_on_demand_calibration(
                                                                        results,
                                                                        Data_Q,
                                                                        allparameters,
                                                                        arr_indexvaryingparameters,
                                                                        sim_indices,
                                                                        pixX,
                                                                        pixY,
                                                                        initrot=matrix,
                                                                        vecteurref=vecteurref,
                                                                        pureRotation=pureRotation,
                                                                        verbose=1,
                                                                        pixelsize=pixelsize,
                                                                        dim=dim,
                                                                        weights=None,
                                                                        allspots_info=1,
                                                                    )

    # print "last pixdev table"
    # print residues_nonweighted
    print("Mean pixdev")
    print(np.mean(residues_nonweighted))
    # print "SpotsData"
    # print SpotsData
    # print "initial matrix"
    # print matrix
    # print "New delta matrix"
    # print deltamat
    # print "newmatrix"
    # print newmatrix
    # print newmatrix.tolist()

    if len(arr_indexvaryingparameters) > 1:
        for k, val in enumerate(arr_indexvaryingparameters):
            allresults[val] = results[k]
    elif len(arr_indexvaryingparameters) == 1:
        allresults[arr_indexvaryingparameters[0]] = results[0]

    residues_fit = residues_nonweighted

    dataresults = allresults.tolist() + [np.mean(residues_fit), len(residues_fit)]

    print("New parameters", allresults)

    # update orient matrix
    # print "updating orientation parameters"

    # matrix = newmatrix
    ##self.deltamatrix = np.eye(3) # identity
    # self.vecteurref is unchanged

    if addoutput:
        return allresults, newmatrix, dataresults, residues_nonweighted
    else:
        return allresults, newmatrix, dataresults


def test_peaksearch_index_refine():
    """
    data treatment sequence:

    peak search
    indexation
    strain refinement

    You may like to change slightly (to allow indexation) defaultParam in order to see how it does affect the resulting Ge strain

    """
    dirname = "./Examples/Ge/"
    imagefilename = "Ge0001.mccd"

    defaultParam = [69.17369978864516611, 1050.50098288478648101, 1115.53514707010231177, 0.13049762505535042, -0.23799600745091942]
    # defaultParam = [69.15, 1050, 1114, 0.1, -0.2]

    pixelsize = 165.0 / 2048

    # Automation of peaksearch + indexation + strain refinement

    # peak search
    time_0 = ttt.time()

    prefix = "autotest_Ge"
    Isorted, _, _ = rmccd.PeakSearch(os.path.join(dirname, imagefilename),
                                                            PixelNearRadius=5,
                                                            removeedge=2,
                                                            IntensityThreshold=500,
                                                            boxsize=5,
                                                            position_definition=1,
                                                            verbose=1,
                                                            fit_peaks_gaussian=1,
                                                            xtol=0.001,
                                                            return_histo=0,
                                                            FitPixelDev=2.0)

    filewritten = RWASCII.writefile_Peaklist(prefix,
                                            Isorted,
                                            position_definition=1,
                                            overwrite=1,
                                            initialfilename=imagefilename,
                                            comments="",
                                            dirname=dirname)  #'.')

    if filewritten:
        print("peak list written in %s" % (prefix + ".dat"))
        print("Working directory is now: ", dirname)
        print("execution time: %.2f sec" % (ttt.time() - time_0))
    else:
        raise ValueError("Please Change directory name")

    dat_filename = prefix + ".dat"

    twicetheta, chi, dataintensity, data_x, data_y = F2TC.Compute_data2thetachi(dat_filename,sorting_intensity="yes", detectorparams=defaultParam, pixelsize=pixelsize)

    prefix, _ = dat_filename.split(".")
    # writing .cor file
    RWASCII.writefile_cor("C_" + prefix,
                        twicetheta,
                        chi,
                        data_x,
                        data_y,
                        dataintensity,
                        sortedexit=0,
                        param=defaultParam + [pixelsize],
                        initialfilename=dat_filename)

    print("%s has been created with defaultparameter" % (prefix + ".cor"))
    print("%s" % str(defaultParam))
    # a C_#########.cor file is now created
    file_extension = "cor"
    DataPlot_filename = "C_" + prefix + "." + file_extension

    # Reading . cor file
    # TODO use loadtxt or readcorfile somewhere
    Current_peak_data = scipy.io.array_import.read_array(DataPlot_filename, lines=(1, -1))

    # nbcolumns == 5:
    data_theta = Current_peak_data[:, 0] / 2.0
    data_chi, data_pixX, data_pixY, data_I = np.transpose(Current_peak_data)[1:]

    # filename = DataPlot_filename

    # index_foundgrain = 0
    # dictionary of exp spots
    indexed_spots = {}
    for k in range(len(data_theta)):
        indexed_spots[k] = [k,  # index of experimental spot in .cor file
                            data_theta[k] * 2.0,
                            data_chi[k],  # 2theta, chi coordinates
                            data_pixX[k],
                            data_pixY[k],  # pixel coordinates
                            data_I[k],  # intensity
                            0]  # 0 means non indexed yet
    # last_orientmatrix_fromindexation = {}
    # last_Bmatrix_fromindexation = {}
    # last_epsil_fromindexation = {}

    # Running indexation

    # updated exp spot index to be still index
    current_exp_spot_index_list = Non_Indexed_Spots(indexed_spots, len(data_theta))

    # for each grain or indexation step: ******************************
    # ******************************************************************

    # compute angular distance between all exp; spots:

    nbspots_in_data = len(data_theta[current_exp_spot_index_list])

    # Matching spots set Size (MSSS):
    nbspotmaxformatching = nbspots_in_data

    # select 1rstly spots that have not been indexed and 2ndly reduced list by user
    index_to_select = np.take(current_exp_spot_index_list, np.arange(nbspotmaxformatching))

    select_theta = data_theta[index_to_select]
    select_chi = data_chi[index_to_select]
    select_I = data_I[index_to_select]
    # print select_theta
    # print select_chi
    listcouple = np.transpose(np.array([select_theta, select_chi]))
    Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

    # classical indexation parameters:
    data = (2 * select_theta, select_chi, select_I, DataPlot_filename)

    spot_index_central = [0, 1, 2, 3]  # central spot or list of spots
    nbmax_probed = 10  # 'Recognition spots set Size (RSSS): '
    elem_label = "UO2"
    energy_max = 25

    rough_tolangle = 0.5  # 'Dist. Recogn. Tol. Angle (deg)'
    fine_tolangle = 0.2  # 'Matching Tolerance Angle (deg)'
    Nb_criterium = 15  # 'Minimum Number Matched Spots: '
    print("rough_tolangle ", rough_tolangle)
    print("fine_tolangle ", fine_tolangle)
    NBRP = 2  # number of best result for each element of spot_index_central

    n = 3  # for LUT
    latticeparams = DictLT.dict_Materials[elem_label][1]
    B = CP.calc_B_RR(latticeparams)

    # indexation procedure
    bestmat, stats_res = INDEX.getOrientMatrices(spot_index_central,
                                                    energy_max,
                                                    Tabledistance[:nbmax_probed, :nbmax_probed],
                                                    select_theta,
                                                    select_chi,
                                                    n,
                                                    B,
                                                    rough_tolangle=rough_tolangle,
                                                    fine_tolangle=fine_tolangle,
                                                    Nb_criterium=Nb_criterium,
                                                    plot=0,
                                                    structure_label=elem_label,
                                                    nbbestplot=NBRP)

    # bestmat contains: NBRP * len(spot_index_central) matrix

    # Look for the very best matrix candidates: -----to be improved in readability ------------------------
    ar = []
    for elem in stats_res:
        ar.append([elem[0], -elem[2]])

    tabstat = np.array(ar)  # , dtype = [('x', '<i4'), ('y', '<i4')])
    rankmat = np.argsort(tabstat[:, 0])[::-1]  # ,order=('x','y'))

    verybestmat = bestmat[rankmat[0]]
    # -----------------------------------------------------------------------------------------------------
    # very best matrix
    orientmatrix = verybestmat

    # the resimulate to output miller indices
    emax_simul = 16
    veryclose_angletol = 0.2

    Bmatrix = np.eye(3)
    (newUBmat, _, _, _,
    _, _, _, _) = GetStrainOrient(orientmatrix,
                                                                    Bmatrix,
                                                                    elem_label,
                                                                    emax_simul,
                                                                    veryclose_angletol,
                                                                    data,
                                                                    detectorparameters=defaultParam,
                                                                    addoutput=0)

    emax_simul = 25
    veryclose_angletol = 0.1

    GetStrainOrient(newUBmat,
                    Bmatrix,
                    elem_label,
                    emax_simul,
                    veryclose_angletol,
                    data,
                    detectorparameters=defaultParam,
                    addoutput=0)


def GetCalibParameter(emax, veryclose_angletol, element, startingmatrix, defaultParam, data_exp,
                                                                        pixelsize=165.0 / 2048,
                                                                        dim=(2048, 2048),
                                                                        boolctrl="1" * 8,
                                                                        use_weights=True,
                                                                        addoutput=0):
    """
    data_exp = twicetheta, chi, dataintensity
    to allow dd, xcen, ycen xbet, xgam, three disorientation angles of crystal
    boolctrl = '11000110'
    """

    matrix = startingmatrix
    Grain = [np.eye(3), [1, 1, 1], matrix, element]

    paramDet = defaultParam
    print("paramDet in LAA.GetCalibParameters :", paramDet)
    if not isinstance(paramDet, list):
        paramDet = paramDet.tolist()

    # simulated theo spots
    data_theo = simulate_theo(Grain, emax, 5, paramDet, pixelsize, dim)
    Twicetheta_theo, Chi_theo, Miller_ind_theo, _, _ = data_theo

    # exp data
    twicetheta, chi, dataintensity, data_x, data_y = data_exp

    # build automatically theo and exp spots associations (see OnAutoLink(self, evt) of picky_Frame of LaueTool24.py)
    energy_theo = 10 * np.ones(len(Twicetheta_theo))
    res_assoc = SpotLinks(twicetheta,
                            chi,
                            dataintensity,  # experimental data
                            veryclose_angletol,  # tolerance angle
                            Twicetheta_theo,
                            Chi_theo,
                            Miller_ind_theo,
                            energy_theo)

    if not res_assoc:
        return None
    else:
        (_,
            linkedspots_link,
            linkExpMiller_link,
            linkIntensity_link,
            _,
            _,
            _,
        ) = res_assoc

        linkedspots = linkedspots_link
        linkExpMiller = linkExpMiller_link
        linkIntensity = linkIntensity_link
        # linkResidues = linkResidues_link

        nb_of_pairs = len(linkedspots)

        # test nb of varying params / nb pairs
        nbvaryingparams = boolctrl.count("1")

        if nbvaryingparams > nb_of_pairs:
            print("There are too few spots pairs to perform the fitting procedure")
            print("nbvaryingparams", nbvaryingparams)
            print("nb of pairs", nb_of_pairs)

        newparam, newmat, dr, pixdevlist = RefineCalibParameters(linkedspots,
                                                                linkExpMiller,
                                                                linkIntensity,
                                                                data_x,
                                                                data_y,
                                                                matrix,
                                                                paramDet,
                                                                boolctrl,
                                                                use_weights=use_weights,
                                                                dim=(2048, 2048),
                                                                pixelsize=pixelsize,
                                                                addoutput=1)

        # spotexpnum_spotsimnum
        # print linkedspots
        # spotexpnum_hkl
        # print linkExpMiller
        # intensitylist
        # print linkIntensity
        if dr != None:
            pixdev = dr[-2]
            if addoutput == 0:
                return newparam, newmat, pixdev, nb_of_pairs
            else:
                return newparam, newmat, pixdev, nb_of_pairs, pixdevlist, linkExpMiller
        else:
            if addoutput == 0:
                return None, None, None, None
            else:
                return None, None, None, None, None, None


def decode_params(string):
    """
    translate string made of 0 1 to resp False True
    """
    b = [True] * len(string)
    for k, s in enumerate(string):
        if s == "0":
            b[k] = False
    return b


def test_calibration_refinement():
    """
    illustrate how to use calibration refinement procedure.

    This converges if initial guess are quite very close to the solution
    (misorientation and unknown detector parameter lead straight to a wrong spots association and then to a wrong )
    """
    dirname = "./Examples/Ge/"
    imagefilename = "Ge0001.mccd"

    # must find
    # defaultParam = [69.190641489891689, 1050.503353382659, 1115.585040635892, 0.14566218092113614, -0.24339580573683331]
    defaultParam = [69.2, 1050.0, 1115, 0, 0]
    element = "Ge"

    pixelsize = 165.0 / 2048
    # dim = (2048, 2048)

    # Automation of peaksearch + indexation + calibration refinement

    # peak search
    time_0 = ttt.time()

    prefix = "autocalibtest_Ge"
    Isorted, _, _ = rmccd.PeakSearch(os.path.join(dirname, imagefilename),
                                                    PixelNearRadius=5,
                                                    removeedge=2,
                                                    IntensityThreshold=500,
                                                    boxsize=5,
                                                    position_definition=1,
                                                    verbose=1,
                                                    fit_peaks_gaussian=1,
                                                    xtol=0.001,
                                                    return_histo=0,
                                                    FitPixelDev=2.0)

    filewritten = RWASCII.writefile_Peaklist(prefix,
                                            Isorted,
                                            position_definition=1,
                                            overwrite=1,
                                            initialfilename=imagefilename,
                                            comments="",
                                            dirname=dirname)  #'.')

    if filewritten:
        print("peak list written in %s" % (prefix + ".dat"))
        print("Working directory is now", dirname)
        print("execution time: %.2f sec" % (ttt.time() - time_0))
    else:
        raise ValueError("Please Change directory name")

    dat_filename = prefix + ".dat"

    print("current directory", os.curdir)

    twicetheta, chi, dataintensity, data_x, data_y = F2TC.Compute_data2thetachi(dat_filename,sorting_intensity="yes", detectorparams=defaultParam,pixelsize=pixelsize)
    # file_peaks = prefix + ".dat"
    # data = (twicetheta, chi, dataintensity, file_peaks)
    # Data_X, Data_Y, Data_I, File_NAME = data # 2theta and chi are now meant as X and Y
    # Data_index_expspot = np.arange(len(Data_X))
    data_exp = (twicetheta, chi, dataintensity, data_x, data_y)

    # starting matrix
    # UB or almost orientation matrix
    startingmatrix = [[0.092608021691080941, -0.97293680654779968, -0.21169793293699135],
                    [0.58953181179698111, 0.22490742744822317, -0.77578363483006851],
                    [0.80239914976161164, -0.052985258144157166, 0.59445649061416561]]

    # add some noise
    # matrot = GT.matRot(np.array([1,1,1]),0.5)
    # startingmatrix = np.dot(startingmatrix,matrot)

    emax = 15
    veryclose_angletol = 1

    newparam, newmat, pixDev, nb_of_pairs = GetCalibParameter(emax,
                                                            veryclose_angletol,
                                                            element,
                                                            startingmatrix,
                                                            defaultParam,
                                                            data_exp,
                                                            pixelsize=165.0 / 2048,
                                                            dim=(2048, 2048),
                                                            boolctrl="1" * 8)

    print("\n**************************************")
    print("pixDev :", pixDev)
    print("nb of peaks : ", nb_of_pairs)
    print("**************************************\n")

    print("\n****************************\n--------------------------2ND Fit ----------------------------\n")

    emax = 18
    veryclose_angletol = 0.5

    # Update
    defaultParam = newparam[:5]
    startingmatrix = newmat

    newparam, newmat, pixDev, nb_of_pairs = GetCalibParameter(emax,
                                                            veryclose_angletol,
                                                            element,
                                                            startingmatrix,
                                                            defaultParam,
                                                            data_exp,
                                                            pixelsize=165.0 / 2048,
                                                            dim=(2048, 2048))

    print("\n**************************************")
    print("pixDev :", pixDev)
    print("nb of peaks : ", nb_of_pairs)
    print("**************************************\n")


def test_SpotLinks():

    twicetheta_exp = np.array([90, 140, 50])
    chi_exp = np.array([70, 45, 85])
    dataintensity_exp = np.array([2400, 2, 2])
    veryclose_angletol = 1

    twicetheta = np.array([90.5, 140.5, -50, 90.5])
    chi = np.array([70, 45, 85, 70.4])
    Miller_ind = np.array([[8, 9, 10], [-5, 6, 0], [3, 2, 1], [6, 5, -5]])

    # posx = np.array([1024, 1000, 2000, 2000.9])
    # posy = np.array([200.6, 600.8, 800.9, 900.9])
    energy = np.array([10.12345, 11.21454796, 12.3179452, 14.547962])

    res = ISS.SpotLinks(twicetheta_exp,
                            chi_exp,
                            dataintensity_exp,  # experimental data
                            veryclose_angletol,  # tolerance angle
                            twicetheta,
                            chi,
                            Miller_ind,
                            energy)  # theoretical data

    if res:
        refine_indexed_spots, linkedspots_link, linkExpMiller_link, linkIntensity_link, linkResidues_link, Energy_Exp_spot, _ = res

    print("refine_indexed_spots", refine_indexed_spots)
    print("linkedspots_link", linkedspots_link)
    print("linkExpMiller_link", linkExpMiller_link)
    print("linkIntensity_link", linkIntensity_link)
    print("linkResidues_link", linkResidues_link)
    print("Energy_Exp_spot", Energy_Exp_spot)


# --- -----------   MAIN


if __name__ == "__main__":

    # test_calibration_refinement()

    # test simulation of strain and then strain retrieval from strain refinement of data

    angsample = 40
    ang = angsample * np.pi / 180.0
    # Ms = np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0 ,np.cos(ang)]])
    Ms = GT.matRot(np.array([0, 1, 0]), 40)
    Msinv = GT.matRot(np.array([0, 1, 0]), -40)

    dirname = "."
    elem_label = "Cu"
    latparam = 3.6

    defaultParam = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201]
    kf_direction = "Z>0"
    emax = 22.0
    emin = 5.0

    pixelsize = 165.0 / 2048

    ## Cu with orientsurf111 stretched by 1 % in bstar direction
    # DataPlot_filename = 'Cu1.cor'

    DataPlot_filename = "pipo.cor"
    B = 1 / latparam * np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.01]])
    Extinc = "fcc"
    Ta = np.eye(3)
    Tc = np.eye(3)
    U = GT.matRot(np.array([1, -2, 0]), 0.0000)

    # if B is expressed in sample frame (x//ki but tilted by 40deg z top, y towards the door
    B1 = B
    B = np.dot(Msinv, B1)
    # U = np.eye(3)

    # BUILDING fake DATA:

    # key_material, Extinc, Ta, U, B, Tc = Laue_classic_param # from combos
    # q = Ta U B Tc G*

    GrainSimulParam = [0, 0, 0, 0]
    # print "\n**************"
    # print "using Bmatrix containing lattice parameter"
    # print "****************\n"

    # take then parameters from combos
    GrainSimulParam[0] = B
    GrainSimulParam[1] = Extinc
    GrainSimulParam[2] = np.dot(np.dot(Ta, U), Tc)
    GrainSimulParam[3] = "inputB"

    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [GrainSimulParam],
                                fastcompute=0,
                                verbose=0,
                                kf_direction=kf_direction)

    # 2theta, chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi,
                                        fileOK=0,
                                        fastcompute=0,
                                        detectordistance=defaultParam[0],  # TODO: check if it is really the parameter set we need
                                        detectordiameter=165 * 1.2,
                                        kf_direction="Z>0",
                                        HarmonicsRemoval=1,
                                        pixelsize=pixelsize)

    Twicetheta = [spot.Twicetheta for spot in TwicethetaChi[0]]
    Chi = [spot.Chi for spot in TwicethetaChi[0]]
    Miller_ind = [list(spot.Millers) for spot in TwicethetaChi[0]]
    Energy = [spot.EwaldRadius * CST_ENERGYKEV for spot in TwicethetaChi[0]]

    posx, posy = F2TC.calc_xycam_from2thetachi(Twicetheta,
                                                Chi,
                                                defaultParam,
                                                verbose=0,
                                                pixelsize=pixelsize)[:2]

    # READING COR file AND INDEXATION ----------------------------------------------------------------------------
    file_extension = "cor"

    data_theta = np.array(Twicetheta) / 2.0
    data_chi = np.array(Chi)
    data_pixX = np.array(posx)
    data_pixY = np.array(posy)
    data_I = np.ones(len(data_theta)) * 1000
    # filename = DataPlot_filename

    index_foundgrain = 0
    # dictionary of exp spots
    indexed_spots = {}
    for k in range(len(data_theta)):
        indexed_spots[k] = [k,  # index of experimental spot in .cor file
                            data_theta[k] * 2.0,
                            data_chi[k],  # 2theta, chi coordinates
                            data_pixX[k],
                            data_pixY[k],  # pixel coordinates
                            data_I[k],  # intensity
                            0]  # 0 means non indexed yet
    last_orientmatrix_fromindexation = {}
    last_Bmatrix_fromindexation = {}
    last_epsil_fromindexation = {}

    # Running indexation

    # updated exp spot index to be still index
    current_exp_spot_index_list = Non_Indexed_Spots(indexed_spots, len(data_theta))

    # for each grain or indexation step: ******************************
    # ******************************************************************

    # compute angular distance between all exp; spots:

    nbspots_in_data = len(data_theta[current_exp_spot_index_list])

    # Matching spots set Size (MSSS):
    nbspotmaxformatching = nbspots_in_data

    # select 1rstly spots that have not been indexed and 2ndly reduced list by user
    index_to_select = np.take(current_exp_spot_index_list, np.arange(nbspotmaxformatching))

    select_theta = data_theta[index_to_select]
    select_chi = data_chi[index_to_select]
    select_I = data_I[index_to_select]
    # print select_theta
    # print select_chi
    listcouple = np.transpose(np.array([select_theta, select_chi]))
    Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

    # classical indexation parameters:
    data = (2 * select_theta, select_chi, select_I, DataPlot_filename)

    spot_index_central = [4, 5, 6, 7]  # central spot or list of spots
    # spot_index_central = 0                  # central spot or list of spots
    nbmax_probed = 10  # 'Recognition spots set Size (RSSS): '

    energy_max = 22.0
    n = 3  # for LUT

    rough_tolangle = 0.5  # 'Dist. Recogn. Tol. Angle (deg)'
    fine_tolangle = 0.2  # 'Matching Tolerance Angle (deg)'
    Nb_criterium = 2  # 'Minimum Number Matched Spots: '
    print("rough_tolangle ", rough_tolangle)
    print("fine_tolangle ", fine_tolangle)
    NBRP = 5  # number of best result for each element of spot_index_central

    latticeparams = DictLT.dict_Materials[elem_label][1]
    B = CP.calc_B_RR(latticeparams)

    # indexation procedure
    bestmat, stats_res = INDEX.getOrientMatrices(spot_index_central,
                                                energy_max,
                                                Tabledistance[:nbmax_probed, :nbmax_probed],
                                                select_theta,
                                                select_chi,
                                                n,
                                                B,
                                                rough_tolangle=rough_tolangle,
                                                fine_tolangle=fine_tolangle,
                                                Nb_criterium=Nb_criterium,
                                                plot=0,
                                                structure_label=elem_label,
                                                nbbestplot=NBRP)

    # bestmat contains: NBRP * len(spot_index_central) matrix

    # Look for the very best matrix candidates: -----to be improved in readability ------------------------
    ar = []
    for elem in stats_res:
        ar.append([elem[0], -elem[2]])

    tabstat = np.array(ar)  # , dtype = [('x', '<i4'), ('y', '<i4')])
    rankmat = np.argsort(tabstat[:, 0])[::-1]  # ,order=('x','y'))

    verybestmat = bestmat[rankmat[0]]
    # -----------------------------------------------------------------------------------------------------
    # very best matrix
    orientmatrix = verybestmat

    # end of READING COR file AND INDEXATION ----------------------------------------------------------------------------

    # STRAIN REFINEMENT -----------------------------------------------
    # the resimulate to output miller indices
    emax_simul = 20
    veryclose_angletol = 0.2

    res_strain = GetStrainOrient(orientmatrix,
                                    B,
                                    elem_label,
                                    emax_simul,
                                    veryclose_angletol,
                                    data,
                                    detectorparameters=defaultParam,
                                    addoutput=0)

    # print "res_strain",res_strain

    newUBmat, newUmat, newBmat, deviatoricstrain, RR, linkResidues_fit, pixdev, nb_fitted_peaks, linkExpMiller_link = res_strain

    latticeparams = DictLT.dict_Materials[elem_label][1]
    deviatoric_strain, latticeparameterstrained = CP.DeviatoricStrain_LatticeParams(
        newUBmat, latticeparams)

    print("\n\ndeviatoric_strain", deviatoric_strain)
    print("latticeparameterstrained", latticeparameterstrained)
    print("newUmat", newUmat)

    def giveeps(mat):
        eps1 = np.dot(Ms, np.dot(mat, Msinv))
        eps2 = np.dot(Msinv, np.dot(mat, Ms))
        return eps1, eps2

    print("\n\nstrain sample frame", giveeps(deviatoricstrain))

    if 0:
        # ----------------------------------------------------------------------------
        # example with Ti
        dirname = "./Examples/Ti/"

        # Ti with orientsurf111 stretched by 1 % in bstar direction
        DataPlot_filename = "Ti_111_0p01_bstar.cor"
        elem_label = "Ti"

        # Ti_s with orientsurf111 lattice param 3 3 4.7 90.5 89.5 120.5
        DataPlot_filename = "Tis.cor"
        elem_label = "Ti"
        # Bmatrix = [[0.3868688469461854, 0.19633627174252782, -0.0010612910304965497],
        # [0.0, 0.33334602612857905, 0.0018567803810125216],
        # [0.0, 0.0, 0.21276595744680851]]

        defaultParam = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201]

        pixelsize = 165.0 / 2048

        # Automation of peaksearch + indexation + strain refinement

        # peak search
        time_0 = ttt.time()

        # a C_#########.cor file is now created
        file_extension = "cor"

        # Reading . cor file
        # TODO use better readcorfile in readmccd.py
        #    Current_peak_data = scipy.io.array_import.read_array(os.path.join(dirname, DataPlot_filename), lines=(1, -1))
        Current_peak_data = np.loadtxt(os.path.join(dirname, DataPlot_filename), skiprows=1)

        # nbcolumns == 5:
        data_theta = Current_peak_data[:, 0] / 2.0
        data_chi, data_pixX, data_pixY, data_I = np.transpose(Current_peak_data)[1:]

        # filename = DataPlot_filename

        index_foundgrain = 0
        # dictionary of exp spots
        indexed_spots = {}
        for k in range(len(data_theta)):
            indexed_spots[k] = [
                k,  # index of experimental spot in .cor file
                data_theta[k] * 2.0,
                data_chi[k],  # 2theta, chi coordinates
                data_pixX[k],
                data_pixY[k],  # pixel coordinates
                data_I[k],  # intensity
                0]  # 0 means non indexed yet
        last_orientmatrix_fromindexation = {}
        last_Bmatrix_fromindexation = {}
        last_epsil_fromindexation = {}

        # Running indexation

        # updated exp spot index to be still index
        current_exp_spot_index_list = Non_Indexed_Spots(indexed_spots, len(data_theta))

        # for each grain or indexation step: ******************************
        # ******************************************************************

        # compute angular distance between all exp; spots:

        nbspots_in_data = len(data_theta[current_exp_spot_index_list])

        # Matching spots set Size (MSSS):
        nbspotmaxformatching = nbspots_in_data

        # select 1rstly spots that have not been indexed and 2ndly reduced list by user
        index_to_select = np.take(current_exp_spot_index_list, np.arange(nbspotmaxformatching))

        select_theta = data_theta[index_to_select]
        select_chi = data_chi[index_to_select]
        select_I = data_I[index_to_select]
        # print select_theta
        # print select_chi
        listcouple = np.transpose(np.array([select_theta, select_chi]))
        Tabledistance = GT.calculdist_from_thetachi(listcouple, listcouple)

        # classical indexation parameters:
        data = (2 * select_theta, select_chi, select_I, DataPlot_filename)

        spot_index_central = [0, 1, 2, 3]  # central spot or list of spots
        spot_index_central = 0  # central spot or list of spots
        nbmax_probed = 10  # 'Recognition spots set Size (RSSS): '

        energy_max = 20.0
        n = 3  # for LUT

        rough_tolangle = 0.5  # 'Dist. Recogn. Tol. Angle (deg)'
        fine_tolangle = 0.2  # 'Matching Tolerance Angle (deg)'
        Nb_criterium = 15  # 'Minimum Number Matched Spots: '
        print("rough_tolangle ", rough_tolangle)
        print("fine_tolangle ", fine_tolangle)
        NBRP = 1  # number of best result for each element of spot_index_central

        latticeparams = DictLT.dict_Materials[elem_label][1]
        B = CP.calc_B_RR(latticeparams)

        # indexation procedure
        bestmat, stats_res = INDEX.getOrientMatrices(spot_index_central,
                                                    energy_max,
                                                    Tabledistance[:nbmax_probed, :nbmax_probed],
                                                    select_theta,
                                                    select_chi,
                                                    n,
                                                    B,
                                                    rough_tolangle=rough_tolangle,
                                                    fine_tolangle=fine_tolangle,
                                                    Nb_criterium=Nb_criterium,
                                                    plot=0,
                                                    structure_label=elem_label,
                                                    nbbestplot=NBRP)

        # bestmat contains: NBRP * len(spot_index_central) matrix

        # Look for the very best matrix candidates: -----to be improved in readability ------------------------
        ar = []
        for elem in stats_res:
            ar.append([elem[0], -elem[2]])

        tabstat = np.array(ar)  # , dtype = [('x', '<i4'), ('y', '<i4')])
        rankmat = np.argsort(tabstat[:, 0])[::-1]  # ,order=('x','y'))

        verybestmat = bestmat[rankmat[0]]
        # -----------------------------------------------------------------------------------------------------
        # very best matrix
        orientmatrix = verybestmat

        # the resimulate to output miller indices
        emax_simul = 16
        veryclose_angletol = 0.2

        res_strain = GetStrainOrient(orientmatrix,
                                        B,
                                        elem_label,
                                        emax_simul,
                                        veryclose_angletol,
                                        data,
                                        detectorparameters=defaultParam,
                                        addoutput=0)

        (newUBmat, newUmat, newBmat, deviatoricstrain, RR,
        linkResidues_fit, pixdev, nb_fitted_peaks) = res_strain

        latticeparams = DictLT.dict_Materials[elem_label][1]
        deviatoric_strain, latticeparameterstrained = CP.DeviatoricStrain_LatticeParams(
            newUBmat, latticeparams)
