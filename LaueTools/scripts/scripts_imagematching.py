import numpy as np

import generaltools as GT
import IOLaueTools as IOLT
import lauecore as LAUE
import dict_LaueTools as DictLT
import indexingImageMatching as IMM
import matchingrate

from .. import indexingSpotsSet as ISS


from . scripts_indexing import createFakeData

# --- ----------  TESTs image Matching
def test_speedbuild():
    """
    test to be launched with python (not ipython)
    """
    import profile

    profile.run('Hough_peak_position_fast([12., 51., 37], Nb=40, pos2D=0, removedges=2, key_material="Si", verbose=0, EULER=1)', "mafonction.profile")
    import pstats

    pstats.Stats("mafonction.profile").sort_stats("time").print_stats()


def show_solution_existence():
    matsol = GT.randomRotationMatrix()
    print("matsol")
    print(matsol)

    detectorparameters = [70, 1024, 1024, 0, 0]
    grainSi = [np.eye(3), "dia", matsol, "Si"]
    Twicetheta, Chi, _, posx, posy, _ = LAUE.SimulateLaue(
        grainSi, 5, 25, detectorparameters, removeharmonics=1)
    dataintensity = 20000 * np.ones(len(Twicetheta))

    IOLT.writefile_cor("testmatch",
                        Twicetheta,
                        Chi,
                        posx,
                        posy,
                        dataintensity,
                        param=detectorparameters)

    from indexingSpotsSet import getallcubicMatrices

    EULER = 1
    for mat in getallcubicMatrices(matsol):
        if EULER == 0:
            angles = GT.fromMatrix_to_elemangles(mat)
        elif EULER == 1:
            angles = GT.fromMatrix_to_EulerAngles(mat)

        if angles[0] >= 0 and angles[1] >= 0 and angles[2] >= 0:
            print(angles)
            IMM.plotHough_compare(angles, "testmatch.cor", 100000, EULER=EULER)

    bestangles = [37, 7, 89]
    IMM.Plot_compare_2thetachi(bestangles, Twicetheta, Chi, verbose=0, key_material="Si",
                                                                            emax=25, EULER=EULER)


def randomdetectorparameters():
    """
    generate random detector parameters
    """
    randomdet = (np.random.rand() - 0.5) * 2 * 3 + 70.0
    randomxcen = (np.random.rand() - 0.5) * 2 * 300 + 1024
    randomycen = (np.random.rand() - 0.5) * 2 * 300 + 1024
    randomxbet = (np.random.rand() - 0.5) * 2 * 2 + 0
    randomxgam = (np.random.rand() - 0.5) * 2 * 4 + 0

    detectorparameters = [randomdet, randomxcen, randomycen, randomxbet, randomxgam]

    return detectorparameters


def test_ImageMatching(database=None):
    """
    Image Matching Test with single crystal Si randomly oriented

    No intensity used

    """
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # generate .cor file
    matsol = GT.randomRotationMatrix()
    print("matsol")
    print(matsol)

    detectorparameters = randomdetectorparameters()
    #    detectorparameters = [68.2, 920, 880, -0.1, 0.8]

    print("detectorparameters", detectorparameters)

    grainSi = [np.eye(3), "dia", matsol, "Si"]
    Twicetheta, Chi, _, posx, posy, _ = LAUE.SimulateLaue(
        grainSi, 5, 25, detectorparameters)
    dataintensity = 20000 * np.ones(len(Twicetheta))

    IOLT.writefile_cor("testSirandom",
                        Twicetheta,
                        Chi,
                        posx,
                        posy,
                        dataintensity,
                        param=detectorparameters)

    # Find the best orientations and gives a table of results
    TBO, dataselected, _ = IMM.give_best_orientations_new(
        "testSirandom.cor", 1000000, database, dirname=None, plotHough=0)

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    print("bestEuler")
    print(bestEULER)

    #    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
    #                                                 dirname='.', returnCorrelArray=1)
    #    print "min", np.amin(correlintensity)
    #    print "max", np.amax(correlintensity)
    #    print "mean", np.mean(correlintensity)
    #    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)

    # Look at the results: --------------------------------------
    #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
    #                            key_material='Si',EULER=1)
    import pylab as p

    p.close()
    for k in list(range(len(bestEULER))):
        p.close()
        IMM.Plot_compare_2thetachi(bestEULER[k],
                                    dataselected[0],
                                    dataselected[1],
                                    verbose=1,
                                    key_material="Si",
                                    emax=22,
                                    EULER=1)
        p.close()


def test_ImageMatching_2(database=None):
    """
    Image Matching Test with two crystal Si randomly oriented

    No intensity used

    """

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # generate .cor file
    matsol1 = GT.randomRotationMatrix()
    print("matsol1")
    print(matsol1)

    matsol2 = GT.randomRotationMatrix()
    print("matsol2")
    print(matsol2)

    #    detectorparameters = [70, 1024, 1024, 0, 0]

    detectorparameters = randomdetectorparameters()

    grainSi1 = [np.eye(3), "dia", matsol1, "Si"]
    grainSi2 = [np.eye(3), "dia", matsol2, "Si"]

    Twicetheta1, Chi1, _, posx1, posy1, _ = LAUE.SimulateLaue(
        grainSi1, 5, 25, detectorparameters)
    dataintensity1 = 20000 * np.ones(len(Twicetheta1))

    Twicetheta2, Chi2, _, posx2, posy2, _ = LAUE.SimulateLaue(
        grainSi2, 5, 25, detectorparameters)
    dataintensity2 = 20000 * np.ones(len(Twicetheta2))

    Twicetheta, Chi, posx, posy, dataintensity = (
        np.concatenate((Twicetheta1, Twicetheta2)),
        np.concatenate((Chi1, Chi2)),
        np.concatenate((posx1, posx2)),
        np.concatenate((posy1, posy2)),
        np.concatenate((dataintensity1, dataintensity2)))

    IOLT.writefile_cor("testSirandom",
                        Twicetheta,
                        Chi,
                        posx,
                        posy,
                        dataintensity,
                        param=detectorparameters)

    sigmagauss = 0.5
    # Find the best orientations and gives a table of results
    TBO, dataselected, _ = IMM.give_best_orientations_new(
                                                "testSirandom.cor",
                                                1000000,
                                                database,
                                                dirname=None,
                                                plotHough=0,
                                                rank_n=20,
                                                sigmagaussian=(sigmagauss, sigmagauss, sigmagauss))

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    print("bestEuler")
    print(bestEULER)

    twicetheta_data, chi_data, _ = dataselected

    matchingrate.getStatsOnMatching(bestEULER, twicetheta_data, chi_data, "Si")

    correlintensity, _ = IMM.plotHoughCorrel(
        database, "testSirandom.cor", 100000, dirname=".", returnCorrelArray=1)
    #    print "min", np.amin(correlintensity)
    #    print "max", np.amax(correlintensity)
    #    print "mean", np.mean(correlintensity)
    #    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)
    #
    #    # Look at the results:
    # #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
    # #                            key_material='Si',EULER=1)
    #
    import pylab as p

    p.close()
    for k in list(range(len(bestEULER))):
        p.close()
        IMM.Plot_compare_2thetachi(bestEULER[k],
                                    dataselected[0],
                                    dataselected[1],
                                    verbose=1,
                                    key_material="Si",
                                    emax=25,
                                    EULER=1)
        p.close()

    IMM.plottab(correlintensity)


#    print "matsol"
#    print matsol.tolist()
#
#    for mat in INDEX.getallcubicMatrices(matsol):
#        angles = GT.fromMatrix_to_EulerAngles(mat)
#        bestangles = np.round(angles)
#        Plot_compare_2thetachi(bestangles, dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=22,
#                           EULER=1)
def test_ImageMatching_ngrains(database=None, nbgrains=5):
    """
    Image Matching Test with nb crystals of Si randomly oriented

    No intensity used

    """
    from indexingSpotsSet import comparematrices

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    for k in list(range(nbgrains)):
        orientmat = GT.randomRotationMatrix()
        #        print "orientmat %d"%k
        #        print orientmat.tolist()
        simul_matrices.append(orientmat)

        grains.append([np.eye(3), "dia", orientmat, "Si"])

        Twicetheta1, Chi1, _, posx1, posy1, _ = LAUE.SimulateLaue(
            grains[k], 5, 25, detectorparameters)
        dataintensity1 = 20000 * np.ones(len(Twicetheta1))

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    IOLT.writefile_cor("testSirandom",
                            Twicetheta,
                            Chi,
                            posx,
                            posy,
                            dataintensity,
                            param=detectorparameters)

    # Find the best orientations and gives a table of results
    SIGMAGAUSS = 0.5
    TBO, dataselected, _ = IMM.give_best_orientations_new(
                                        "testSirandom.cor",
                                        1000000,
                                        database,
                                        dirname=None,
                                        plotHough=0,
                                        rank_n=20,
                                        sigmagaussian=(SIGMAGAUSS, SIGMAGAUSS, SIGMAGAUSS))

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    #    print "bestEuler"
    #    print bestEULER

    twicetheta_data, chi_data, _ = dataselected

    # sort solution by matching rate
    sortedindices, matchingrates = matchingrate.getStatsOnMatching(
        bestEULER, twicetheta_data, chi_data, "Si", verbose=0)

    #    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
    #                                                 dirname='.', returnCorrelArray=1)
    #    print "min", np.amin(correlintensity)
    #    print "max", np.amax(correlintensity)
    #    print "mean", np.mean(correlintensity)
    #    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)
    #
    #    # Look at the results:
    # #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
    # #                            key_material='Si',EULER=1)
    #
    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    # verybestmat = GT.fromEULERangles_toMatrix(bestEULER[0])

    grain_indexed = []
    ind_euler = 0
    for eulers in bestEULER:
        bestmat = GT.fromEULERangles_toMatrix(eulers)
        kk = 0
        #        print "ind_euler",ind_euler
        #        print "eulers",eulers
        #        print bestmat.tolist()
        for mat in simul_matrices:
            if comparematrices(bestmat, mat, tol=0.1)[0]:
                print(
                    "\nindexation succeeded with grain #%d !! with Euler Angles #%d\n"
                    % (kk, ind_euler))
                angx, angy, angz = eulers
                print("[ %.1f, %.1f, %.1f]" % (angx, angy, angz))
                print("matching rate  ", bestmatchingrates[ind_euler])
                if kk not in grain_indexed:
                    grain_indexed.append(kk)
            kk += 1

        ind_euler += 1

    # summary
    print("nb of grains indexed  :%d" % len(grain_indexed))
    import pylab as p

    p.close()
    #    for k in range(len(bestEULER)):
    #        p.close()
    #        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
    #                           verbose=1, key_material='Si', emax=25,
    #                           EULER=1)
    #        p.close()
    import pylab as p

    p.close()
    IMM.Plot_compare_2thetachi(bestEULER[0],
                                dataselected[0],
                                dataselected[1],
                                verbose=1,
                                key_material="Si",
                                emax=25,
                                EULER=1)


#    plottab(correlintensity)

#    print "matsol"
#    print matsol.tolist()
#
#    for mat in INDEX.getallcubicMatrices(matsol):
#        angles = GT.fromMatrix_to_EulerAngles(mat)
#        bestangles = np.round(angles)
#        Plot_compare_2thetachi(bestangles, dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=22,
#                           EULER=1)


def test_ImageMatching_otherelement(database=None, nbgrains=3):
    """
    Image Matching Test with other element than that used for building the database

    No intensity used

    """

    from indexingSpotsSet import (comparematrices, initIndexationDict, getIndexedSpots, updateIndexationDict)

    ELEMENT = "Cu"
    EXTINCTION = "fcc"
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    for k in list(range(nbgrains)):
        orientmat = GT.randomRotationMatrix()
        #        print "orientmat %d"%k
        #        print orientmat.tolist()
        simul_matrices.append(orientmat)

        grains.append([np.eye(3), EXTINCTION, orientmat, ELEMENT])

        Twicetheta1, Chi1, _, posx1, posy1, _ = LAUE.SimulateLaue(
            grains[k], 5, 25, detectorparameters)
        dataintensity1 = 20000 * np.ones(len(Twicetheta1))

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    IOLT.writefile_cor("testSirandom",
                        Twicetheta,
                        Chi,
                        posx,
                        posy,
                        dataintensity,
                        param=detectorparameters)

    # create a dictionary of indexed spots
    indexed_spots_dict, _ = initIndexationDict(
        (Twicetheta, Chi, dataintensity, posx, posy))

    #    print "indexed_spots_dict", indexed_spots_dict

    # Find the best orientations and gives a table of results
    SIGMAGAUSS = 0.5
    TBO, dataselected, _ = IMM.give_best_orientations_new(
        "testSirandom.cor",
        1000000,
        database,
        dirname=None,
        plotHough=0,
        rank_n=20,
        sigmagaussian=(SIGMAGAUSS, SIGMAGAUSS, SIGMAGAUSS))

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    #    print "bestEuler"
    #    print bestEULER

    twicetheta_data, chi_data, intensity_data = dataselected

    # sort solution by matching rate
    sortedindices, matchingrates = matchingrate.getStatsOnMatching(
        bestEULER, twicetheta_data, chi_data, ELEMENT, verbose=0)

    #    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
    #                                                 dirname='.', returnCorrelArray=1)
    #    print "min", np.amin(correlintensity)
    #    print "max", np.amax(correlintensity)
    #    print "mean", np.mean(correlintensity)
    #    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)
    #
    #    # Look at the results:
    # #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
    # #                            key_material='Si',EULER=1)
    #
    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    grain_indexed = []
    ind_euler = 0
    for eulers in bestEULER:
        bestmat = GT.fromEULERangles_toMatrix(eulers)
        kk = 0
        #        print "ind_euler",ind_euler
        #        print "eulers",eulers
        #        print bestmat.tolist()
        for mat in simul_matrices:
            if comparematrices(bestmat, mat, tol=0.1)[0]:
                print(
                    "\nindexation succeeded with grain #%d !! with Euler Angles #%d\n"
                    % (kk, ind_euler))
                angx, angy, angz = eulers
                print("[ %.1f, %.1f, %.1f]" % (angx, angy, angz))
                print("matching rate  ", bestmatchingrates[ind_euler])
                if kk not in grain_indexed:
                    grain_indexed.append(kk)
            kk += 1

        ind_euler += 1

    # summary
    print("nb of grains indexed  :%d" % len(grain_indexed))

    # handle spot indexation for one grain

    verybestmat = GT.fromEULERangles_toMatrix(bestEULER[0])

    indexation_res, _ = getIndexedSpots(verybestmat,
                                        (twicetheta_data, chi_data, intensity_data),
                                        ELEMENT,
                                        detectorparameters,
                                        veryclose_angletol=0.5,
                                        emin=5,
                                        emax=25,
                                        verbose=0,
                                        detectordiameter=165.0)

    nb_of_indexed_spots = len(indexation_res[5])

    #    print indexation_res
    print("nb of indexed spots fo this matrix: %d" % nb_of_indexed_spots)

    grain_index = 0
    indexed_spots_dict, _ = updateIndexationDict(indexation_res, indexed_spots_dict, grain_index)

    #    p.close()
    #    for k in range(len(bestEULER)):
    #        p.close()
    #        Plot_compare_2thetachi(bestEULER[k], dataselected[0], dataselected[1],
    #                           verbose=1, key_material='Si', emax=25,
    #                           EULER=1)
    #        p.close()
    import pylab as p

    p.close()
    IMM.Plot_compare_2thetachi(bestEULER[0],
                                dataselected[0],
                                dataselected[1],
                                verbose=1,
                                key_material=ELEMENT,
                                emax=25,
                                EULER=1)

    return indexed_spots_dict


#    plottab(correlintensity)

#    print "matsol"
#    print matsol.tolist()
#
#    for mat in INDEX.getallcubicMatrices(matsol):
#        angles = GT.fromMatrix_to_EulerAngles(mat)
#        bestangles = np.round(angles)
#        Plot_compare_2thetachi(bestangles, dataselected[0], dataselected[1],
#                           verbose=1, key_material='Si', emax=22,
#                           EULER=1)


def test_ImageMatching_dev(database=None):
    """
    Image Matching Test with single crystal Si randomly oriented

    No intensity used
    """
    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # generate .cor file
    matsol = GT.randomRotationMatrix()
    print("matsol")
    print(matsol)

    detectorparameters = detectorparameters = randomdetectorparameters()
    #    detectorparameters = [68.2, 920, 880, -0.1, 0.8]

    print("detectorparameters", detectorparameters)

    grainSi = [np.eye(3), "dia", matsol, "Si"]
    Twicetheta, Chi, _, posx, posy, _ = LAUE.SimulateLaue(
        grainSi, 5, 25, detectorparameters)
    dataintensity = 20000 * np.ones(len(Twicetheta))

    IOLT.writefile_cor("testSirandom",
                        Twicetheta,
                        Chi,
                        posx,
                        posy,
                        dataintensity,
                        param=detectorparameters)

    # Find the best orientations and gives a table of results
    TBO, dataselected, _ = IMM.give_best_orientations_new(
        "testSirandom.cor", 1000000, database, dirname=None, plotHough=0)

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    print("bestEuler")
    print(bestEULER)

    #    print dataselected
    twicetheta_data, chi_data, _ = dataselected

    matchingrate.getStatsOnMatching(bestEULER, twicetheta_data, chi_data, "Si")

    #    correlintensity, argcorrel = plotHoughCorrel(database, 'testSirandom.cor', 100000,
    #                                                 dirname='.', returnCorrelArray=1)
    #    print "min", np.amin(correlintensity)
    #    print "max", np.amax(correlintensity)
    #    print "mean", np.mean(correlintensity)
    #    print "max/mean", 1.*np.amax(correlintensity) / np.mean(correlintensity)

    # Look at the results: --------------------------------------
    #    Plot_compare_gnomon([ 79, 20,44],gno_dataselected[0],gno_dataselected[1],
    #                            key_material='Si',EULER=1)
    import pylab as p

    p.close()
    for k in list(range(len(bestEULER))):
        p.close()
        IMM.Plot_compare_2thetachi(bestEULER[k],
                                dataselected[0],
                                dataselected[1],
                                verbose=1,
                                key_material="Si",
                                emax=25,
                                EULER=1)
        p.close()


def test_ImageMatching_twins(database=None):
    """
    Image Matching Test with other element than that used for building the database

    No intensity used
    """
    DictLT.dict_Materials["DIAs"] = ["DIAs", [3.16, 3.16, 3.16, 90, 90, 90], "dia"]

    if 1:

        ELEMENT = "Si"

        MatchingRate_Threshold = 60  # percent

        SIGMAGAUSS = 0.5
        Hough_init_sigmas = (1, 0.7)
        Hough_init_Threshold = 1  # threshold for a kind of filter
        rank_n = 20
        plotHough = 1
        useintensities = 0
        maxindex = 40

    # 1 ,2, 3 grains with diamond structure small unit cell
    if 0:

        ELEMENT = "DIAs"

        MatchingRate_Threshold = 50  # percent

        SIGMAGAUSS = 0.5
        Hough_init_sigmas = (1, 0.7)

        Hough_init_Threshold = -1  # -1 : maximum filter
        rank_n = 40
        plotHough = 0
        useintensities = 0
        maxindex = 40

    # 4 grains with diamond structure small unit cell
    # testSirandom_7.cor   :  zones axes overlap
    # 6 grains with diamond structure small unit cell
    # testSirandom_8.cor   :  zones axes overlap
    if 0:

        ELEMENT = "DIAs"

        MatchingRate_Threshold = 30  # percent

        Hough_init_sigmas = (1, 0.7)
        Hough_init_Threshold = -1
        rank_n = 20
        SIGMAGAUSS = 0.5
        plotHough = 1
        useintensities = 1
        maxindex = 20

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal

    outputfilename = "testtwins"
    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    EXTINCTION = DictLT.dict_Materials[ELEMENT][2]

    orientmat = GT.randomRotationMatrix()
    #        print "orientmat %d"%k
    #        print orientmat.tolist()
    simul_matrices.append(orientmat)

    grains.append([np.eye(3), EXTINCTION, orientmat, ELEMENT])

    grains.append([np.eye(3),
            EXTINCTION,
            np.dot(DictLT.dict_Vect["sigma3_1"], orientmat),
            ELEMENT])

    for k in list(range(2)):

        Twicetheta1, Chi1, _, posx1, posy1, _ = LAUE.SimulateLaue(
            grains[k], 5, 25, detectorparameters, removeharmonics=1)
        dataintensity1 = (20000 + k) * np.ones(len(Twicetheta1))

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    IOLT.writefile_cor(outputfilename,
                        Twicetheta,
                        Chi,
                        posx,
                        posy,
                        dataintensity,
                        param=detectorparameters)

    file_to_index = outputfilename + ".cor"

    #    print "indexed_spots_dict", indexed_spots_dict

    # Find the best orientations and gives a table of results

    sigmas = (SIGMAGAUSS, SIGMAGAUSS, SIGMAGAUSS)
    TBO, dataselected, _ = IMM.give_best_orientations_new(
                                                file_to_index,
                                                1000000,
                                                database,
                                                maxindex=maxindex,
                                                dirname=None,
                                                plotHough=plotHough,
                                                Hough_init_sigmas=Hough_init_sigmas,
                                                Hough_init_Threshold=Hough_init_Threshold,
                                                useintensities=useintensities,
                                                rank_n=rank_n,  # 20
                                                sigmagaussian=sigmas)

    if TBO is None:
        print("Image  Matching has not found any potential orientations!")
        return

    # TODO:to simplify
    _, data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(
        file_to_index)
    Twicetheta = 2.0 * data_theta

    # create a dictionary of indexed spots
    indexed_spots_dict, _ = ISS.initIndexationDict(
        (Twicetheta, Chi, dataintensity, posx, posy))

    # table of best 3 Eulerian angles
    bestEULER = np.transpose(TBO)

    #    print "bestEuler"
    #    print bestEULER

    twicetheta_data, chi_data, intensity_data = dataselected

    # sort solution by matching rate
    sortedindices, matchingrates = matchingrate.getStatsOnMatching(
        bestEULER, twicetheta_data, chi_data, ELEMENT, verbose=0)

    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    print("\nnb of potential grains %d" % len(bestEULER))
    print("bestmatchingrates")
    print(bestmatchingrates)

    bestEULER_0, bestmatchingrates_0 = ISS.filterEquivalentMatrix(
        bestEULER, bestmatchingrates)

    bestEULER, bestmatchingrates = ISS.filterMatrix_MinimumRate(
        bestEULER_0, bestmatchingrates_0, MatchingRate_Threshold)
    print("After filtering (cubic permutation, matching threshold %.2f)"
        % MatchingRate_Threshold)
    print("%d matrices remain\n" % len(bestEULER))

    # first indexation of spots with raw (un refined) matrices
    AngleTol = 1.0
    dict_grain_matrix = {}
    dict_matching_rate = {}
    (indexed_spots_dict,
        dict_grain_matrix,
        dict_matching_rate,
    ) = ISS.rawMultipleIndexation(bestEULER,
                                    indexed_spots_dict,
                                    ELEMENT,
                                    detectorparameters,
                                    AngleTol=AngleTol,
                                    emax=25)

    print("dict_grain_matrix", dict_grain_matrix)

    if dict_grain_matrix is not None:
        ISS.plotgrains(dict_grain_matrix, ELEMENT, detectorparameters, 25, exp_data=dataselected)
    else:
        print("plot the best orientation matrix candidate")
        IMM.Plot_compare_2thetachi(bestEULER_0[0],
                                    dataselected[0],
                                    dataselected[1],
                                    verbose=1,
                                    key_material=ELEMENT,
                                    emax=25,
                                    EULER=1)

        IMM.Plot_compare_gnomondata(bestEULER_0[0],
                                    dataselected[0],
                                    dataselected[1],
                                    verbose=1,
                                    key_material=ELEMENT,
                                    emax=25,
                                    EULER=1)
        return

    # ---    refine matrix
    print("\n\n Refine first matrix")

    grain_index = 0
    #    print "initial matrix", dict_grain_matrix[grain_index]
    refinedMatrix, _ = ISS.refineUBSpotsFamily(indexed_spots_dict,
        grain_index,
        dict_grain_matrix[grain_index],
        ELEMENT,
        detectorparameters,
        use_weights=1)

    #    print "refinedMatrix"
    #    print refinedMatrix

    if refinedMatrix is not None:

        AngleTol = 0.5
        # redo the spots links
        indexation_res, nbtheospots = ISS.getIndexedSpots(refinedMatrix,
                                                    (twicetheta_data, chi_data, intensity_data),
                                                    ELEMENT,
                                                    detectorparameters,
                                                    removeharmonics=1,
                                                    veryclose_angletol=AngleTol,
                                                    emin=5,
                                                    emax=25,
                                                    verbose=0,
                                                    detectordiameter=165.0)

        #        print 'nb of links', len(indexation_res[1])

        if indexation_res is None:
            return

        indexed_spots_dict, nb_updates = ISS.updateIndexationDict(
            indexation_res, indexed_spots_dict, grain_index, overwrite=1)

        print("with refined matrix")
        print("nb of indexed spots for this matrix # %d: %d / %d"
            % (grain_index, nb_updates, nbtheospots))
        print("with tolerance angle : %.2f deg" % AngleTol)

        dict_grain_matrix[grain_index] = refinedMatrix
        dict_matching_rate[grain_index] = [nb_updates, 100.0 * nb_updates / nbtheospots]

        ISS.plotgrains(dict_grain_matrix,
            ELEMENT,
            detectorparameters,
            25,
            exp_data=ISS.getSpotsData(indexed_spots_dict)[:, 1:3].T)

        # one more time with less tolerance in spotlinks

        # ---    refine matrix
        print("\n\n Refine first matrix")

        grain_index = 0
        #        print "initial matrix", dict_grain_matrix[grain_index]
        refinedMatrix, _ = ISS.refineUBSpotsFamily(indexed_spots_dict,
            grain_index,
            dict_grain_matrix[grain_index],
            ELEMENT,
            detectorparameters,
            use_weights=1,
            pixelsize=165.0 / 2048,
            dim=(2048, 2048))

        #        print "refinedMatrix"
        #        print refinedMatrix

        if refinedMatrix is None:
            return

        AngleTol = 0.1
        indexation_res, nbtheospots = ISS.getIndexedSpots(refinedMatrix,
                                                        (twicetheta_data, chi_data, intensity_data),
                                                        ELEMENT,
                                                        detectorparameters,
                                                        removeharmonics=1,
                                                        veryclose_angletol=AngleTol,
                                                        emin=5,
                                                        emax=25,
                                                        verbose=0,
                                                        detectordiameter=165.0)

        #        print 'nb of links', len(indexation_res[1])
        if indexation_res is None:
            return

        indexed_spots_dict, nb_updates = ISS.updateIndexationDict(indexation_res,
                                                    indexed_spots_dict, grain_index, overwrite=1)

        print("with refined matrix")
        print("nb of indexed spots for this matrix # %d: %d / %d"
            % (grain_index, nb_updates, nbtheospots))
        print("with tolerance angle : %.2f deg" % AngleTol)

        dict_grain_matrix[grain_index] = refinedMatrix
        dict_matching_rate[grain_index] = [nb_updates, 100.0 * nb_updates / nbtheospots]

        ISS.plotgrains(dict_grain_matrix,
            ELEMENT,
            detectorparameters,
            25,
            exp_data=ISS.getSpotsData(indexed_spots_dict)[:, 1:3].T)

    return indexed_spots_dict, dict_grain_matrix


def test_ImageMatching_index(database=None, nbgrains=3, readfile=None):
    """
    Image Matching Test with other element than that used for building the database

    No intensity used
    """
    from indexingSpotsSet import (
        initIndexationDict,
        rawMultipleIndexation,
        plotgrains,
        filterEulersList)

    DictLT.dict_Materials["DIAs"] = ["DIAs", [3.16, 3.16, 3.16, 90, 90, 90], "dia"]
    firstmatchingtolerance = 1.0
    nb_of_peaks = 1000000  # all peaks are used in hough transform for recognition
    # tolerancedistance = 0  # no close peaks removal
    # maxindex = 40
    # useintensities = 0
    # plotHough = 0

    emax = 25
    if 1:

        key_material = "Si"

        MatchingRate_Threshold = 60  # percent

        # SIGMAGAUSS = 0.5
        # Hough_init_sigmas = (1, 0.7)
        # Hough_init_Threshold = 1  # threshold for a kind of filter
        # rank_n = 20
        # useintensities = 0

        dictimm = {"sigmagaussian": (0.5, 0.5, 0.5),
            "Hough_init_sigmas": (1, 0.7),
            "Hough_init_Threshold": 1,
            "rank_n": 20,
            "useintensities": 0}

    # 1 ,2, 3 grains with diamond structure small unit cell
    if 0:

        key_material = "DIAs"

        MatchingRate_Threshold = 50  # percent

        # SIGMAGAUSS = 0.5
        # Hough_init_sigmas = (1, 0.7)

        # Hough_init_Threshold = -1  # -1 : maximum filter
        # rank_n = 40
        # plotHough = 0
        # useintensities = 0
        # maxindex = 40

    # 4 grains with diamond structure small unit cell
    # testSirandom_7.cor   :  zones axes overlap
    # 6 grains with diamond structure small unit cell
    # testSirandom_8.cor   :  zones axes overlap
    if 0:

        key_material = "DIAs"

        MatchingRate_Threshold = 15  # percent

        # Hough_init_sigmas = (1, 0.7)
        # Hough_init_Threshold = 1
        # rank_n = 40
        # SIGMAGAUSS = 0.5
        # plotHough = 1
        # useintensities = 0
        # maxindex = 40
        firstmatchingtolerance = 0.5
        # tolerancedistance = 1.0
    #        nb_of_peaks = 100

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "test%srandom" % key_material
        file_to_index = createFakeData(key_material, nbgrains, outputfilename=outputfilename)
    else:
        file_to_index = readfile

    # read spots data: 2theta, chi, intensity
    TwiceTheta_Chi_Int = IMM.readnselectdata(file_to_index, nb_of_peaks)

    twicetheta_data, chi_data = TwiceTheta_Chi_Int[:2]
    # TODO:to simplify
    _, data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(
        file_to_index)
    Twicetheta = 2.0 * data_theta

    # create an initial dictionary of indexed spots
    indexed_spots_dict, _ = initIndexationDict(
        (Twicetheta, Chi, dataintensity, posx, posy))

    # Find the best orientations and gives a table of results

    bestEULER = IMM.bestorient_from_2thetachi(
        (Twicetheta, Chi), database, dictparameters=dictimm)

    bestEULER, _ = filterEulersList(bestEULER,
                                (Twicetheta, Chi),
                                key_material,
                                emax,
                                rawAngularmatchingtolerance=firstmatchingtolerance,
                                MatchingRate_Threshold=MatchingRate_Threshold,
                                verbose=1)

    # first indexation of spots with raw (un refined) matrices
    AngleTol = 1.0
    dict_grain_matrix = {}
    # dict_matching_rate = {}
    (indexed_spots_dict, dict_grain_matrix, _) = rawMultipleIndexation(
                                                                        bestEULER,
                                                                        indexed_spots_dict,
                                                                        key_material,
                                                                        detectorparameters,
                                                                        AngleTol=AngleTol,
                                                                        emax=emax)

    print("dict_grain_matrix", dict_grain_matrix)

    if dict_grain_matrix is not None:
        plotgrains(
            dict_grain_matrix,
            key_material,
            detectorparameters,
            emax,
            exp_data=(Twicetheta, Chi))
    else:
        print("plot the best orientation matrix candidate")
        IMM.Plot_compare_2thetachi(
            bestEULER[0],
            twicetheta_data,
            chi_data,
            verbose=1,
            key_material=key_material,
            emax=emax,
            EULER=1)

        IMM.Plot_compare_gnomondata(
            bestEULER[0],
            twicetheta_data,
            chi_data,
            verbose=1,
            key_material=key_material,
            emax=emax,
            EULER=1)
        return


def test_old(database):
    """
    test refinement and multigrains indexing with class
    """
    readfile = "test.cor"

    DictLT.dict_Materials["DIAs"] = ["DIAs", [3.16, 3.16, 3.16, 90, 90, 90], "dia"]
    firstmatchingtolerance = 1.0

    if 1:

        key_material = "Si"
        emin = 5
        emax = 25

        MatchingRate_Threshold = 30  # percent

        dictimm = {
            "sigmagaussian": (0.5, 0.5, 0.5),
            "Hough_init_sigmas": (1, 0.7),
            "Hough_init_Threshold": 1,
            "rank_n": 20,
            "useintensities": 0}

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "testSirandom"
        nbgrains = 3
        file_to_index = createFakeData(key_material, nbgrains, outputfilename=outputfilename)
    else:
        file_to_index = readfile

    # read spots data and init dictionary of indexed spots
    DataSet = ISS.spotsset()
    DataSet.importdatafromfile(file_to_index)

    TwiceTheta_Chi_Int = DataSet.getSpotsExpData()
    totalnbspots = len(TwiceTheta_Chi_Int[0])
    # Find the best orientations and gives a table of results

    DataSet.key_material = key_material
    DataSet.emin = emin
    DataSet.emax = emax

    # potential orientation solutions from image matching
    bestEULER = IMM.bestorient_from_2thetachi(TwiceTheta_Chi_Int, database, dictparameters=dictimm)

    bestEULER, _ = ISS.filterEulersList(bestEULER,
                                                TwiceTheta_Chi_Int,
                                                key_material,
                                                emax,
                                                rawAngularmatchingtolerance=firstmatchingtolerance,
                                                MatchingRate_Threshold=MatchingRate_Threshold,
                                                verbose=1)

    MatrixPile = bestEULER.tolist()

    matrix_to_test = True
    grain_index = 0  # current grain to be indexed

    indexedgrains = []

    selectedspots_index = None

    AngleTol_0 = 1.0
    while matrix_to_test:

        if len(MatrixPile) != 0:
            eulerangles = MatrixPile.pop(0)
            _ = GT.fromEULERangles_toMatrix(eulerangles)
        else:
            print("no more orientation to test (euler angles)")
            print("need to use other indexing techniques, Angles LUT, cliques help...")
            return

        #        #add a matrix in dictionary
        #        DataSet.dict_grain_matrix[grain_index] = trialmatrix

        print("dict_grain_matrix before initial indexing with a raw orientation")
        print(DataSet.dict_grain_matrix)

        # indexation of spots with raw (un refined) matrices & #update dictionaries
        print("\n\n---------------indexing grain  #%d-----------------------"
            % grain_index)
        print("eulerangles", eulerangles)
        #        print "selectedspots_index", selectedspots_index
        #        print "making raw links for grain #%d" % grain_index
        # TODO update
        DataSet.AssignHKL(eulerangles,
                        grain_index,
                        AngleTol_0,
                        use_spots_in_currentselection=selectedspots_index)

        #        print "dict_grain_matrix after raw matching", DataSet.dict_grain_matrix
        #        print "dict_matching_rate", DataSet.dict_grain_matching_rate

        # plot
        if DataSet.dict_grain_matrix is not None:
            DataSet.plotgrains(
                exp_data=TwiceTheta_Chi_Int,
                titlefig="%d Exp. spots data (/%d) after raw matching grain #%d"
                % (len(TwiceTheta_Chi_Int[0]), totalnbspots, grain_index))
        else:
            print("plot the best last orientation matrix candidate")
            IMM.Plot_compare_2thetachi(eulerangles,
                                        TwiceTheta_Chi_Int[0],
                                        TwiceTheta_Chi_Int[1],
                                        verbose=1,
                                        key_material=DataSet.key_material,
                                        emax=DataSet.emax,
                                        EULER=1)

            IMM.Plot_compare_gnomondata(eulerangles,
                                        TwiceTheta_Chi_Int[0],
                                        TwiceTheta_Chi_Int[1],
                                        verbose=1,
                                        key_material=DataSet.key_material,
                                        emax=DataSet.emax,
                                        EULER=1)

        AngleTol_List = [0.5, 0.2, 0.1]

        # TODO: data flow and branching will be improved later
        for k, AngleTol in enumerate(AngleTol_List):

            print("\n\n refining grain #%d step -----%d\n" % (grain_index, k))
            #    print "initial matrix", dict_grain_matrix[grain_index]

            refinedMatrix = DataSet.refineUBSpotsFamily(grain_index,
                                            DataSet.dict_grain_matrix[grain_index], use_weights=1)

            if refinedMatrix is None:
                break

            # extract data not yet indexed or temporarly indexed
            print("\n\n---------------extracting data-----------------------")
            toindexdata = DataSet.getUnIndexedSpotsallData(exceptgrains=indexedgrains)
            absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
            intensity_data = toindexdata[:, 5]
            absoluteindex = np.array(absoluteindex, dtype=np.int)

            TwiceTheta_Chi_Int = np.array([twicetheta_data, chi_data, intensity_data])

            #            print "spots index to index", toindexdata[:20, 0]
            #            print "twicetheta_data", toindexdata[:5, 1]

            print("\n\n---------------extracting data-----------------------")
            indexation_res, nbtheospots, _ = DataSet.getSpotsLinks(
                refinedMatrix,
                exp_data=TwiceTheta_Chi_Int,
                useabsoluteindex=absoluteindex,
                removeharmonics=1,
                veryclose_angletol=AngleTol,
                verbose=0)

            #        print 'nb of links', len(indexation_res[1])

            if indexation_res is None:
                print("no unambiguous close links between exp. and  theo. spots have been found!")
                break

            # TODO: getstatsonmatching before updateindexation... ?
            nb_updates = DataSet.updateIndexationDict(
                indexation_res, grain_index, overwrite=1)

            print("\nnb of indexed spots for this refined matrix: %d / %d"
                % (nb_updates, nbtheospots))
            print("with tolerance angle : %.2f deg" % AngleTol)

            Matching_rate = 100.0 * nb_updates / nbtheospots

            if Matching_rate < 50.0:
                # matching rate too low
                # then remove previous indexed data and launch again imagematching
                print("matching rate too low")
                # there are at least one indexed grain
                # so remove the corresponding data and restart the imagematching
                if len(indexedgrains) > 0:

                    print("\n\n---------------------------------------------")
                    print(
                        "Use again imagematching on purged data from previously indexed spots")
                    print("---------------------------------------------\n\n")

                    #                    print "indexedgrains", indexedgrains

                    # extract data not yet indexed or temporarly indexed
                    toindexdata = DataSet.getUnIndexedSpotsallData(exceptgrains=indexedgrains)
                    absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
                    intensity_data = toindexdata[:, 5]

                    TwiceTheta_Chi_Int = np.array([twicetheta_data, chi_data, intensity_data])

                    # potential orientation solutions from image matching
                    bestEULER2 = IMM.bestorient_from_2thetachi(
                        TwiceTheta_Chi_Int, database, dictparameters=dictimm)

                    bestEULER2, _ = ISS.filterEulersList(
                        bestEULER,
                        TwiceTheta_Chi_Int,
                        key_material,
                        emax,
                        rawAngularmatchingtolerance=firstmatchingtolerance,
                        MatchingRate_Threshold=MatchingRate_Threshold,
                        verbose=1)

                    # overwrite (NOT add in) MatrixPile
                    MatrixPile = bestEULER2.tolist()

                    selectedspots_index = absoluteindex

                    print("working with a new set of trial orientations")

                    # exit the for loop of refinement and start with other set of spots
                    break

                # no data to remove
                else:
                    break

            # matching rate is high than threshold
            else:
                # keep on refining and reducing tolerance angle
                # or for the lowest tolerance angle, consider the refinement-indexation is completed
                DataSet.dict_grain_matrix[grain_index] = refinedMatrix
                DataSet.dict_grain_matching_rate[grain_index] = [nb_updates, Matching_rate]

                DataSet.plotgrains(
                    exp_data=DataSet.getSpotsExpData(),
                    titlefig="%d Exp. spots data (/%d), after refinement of grain #%d at step #%d"
                    % (totalnbspots, totalnbspots, grain_index, k))

                # if this is the last tolerance step
                if k == len(AngleTol_List) - 1:

                    print("\n---------------------------------------------")
                    print("indexing completed for grain #%d with matching rate %.2f "
                        % (grain_index, Matching_rate))
                    print("---------------------------------------------\n")
                    indexedgrains.append(grain_index)
                    grain_index += 1


#                    spots_G0 = DataSet.getSpotsFamily(0)
#                    print "len(G0)", len(spots_G0)
#                    print "G0", spots_G0


def test_index(database):
    """
    test refinement and multigrains indexing with class
    """
    #    readfile = 'test6.cor'
    #    readfile = 'test7.cor'
    #    readfile = 'test10.cor'
    readfile = None

    key_material = "Si"
    emin = 5
    emax = 25

    # MatchingRate_Threshold = 30  # percent

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    dictimm = {"sigmagaussian": (0.5, 0.5, 0.5),
        "Hough_init_sigmas": (1, 0.7),
        "Hough_init_Threshold": 1,
        "rank_n": 20,
        "useintensities": 0}

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "testSirandom"
        nbgrains = 5
        removespots = [9, 8, 15, 6, 2]
        addspots = 6
        #        removespots = None
        #        addspots = None
        file_to_index = createFakeData(key_material,
                                        nbgrains,
                                        outputfilename=outputfilename,
                                        removespots=removespots,
                                        addspots=addspots)
    else:
        file_to_index = readfile

    DataSet = ISS.spotsset()

    DataSet.IndexSpotsSet(file_to_index, key_material, emin, emax, dictimm, database)

    DataSet.plotallgrains()

    return DataSet


def test_compare_IMM_IAM(database, nbgrains, twins=0):
    """
    test refinement and multigrains indexing with class

    comparison Indexing Matching Method and indexing Angles Method
    """
    #    readfile = 'test6.cor'
    #    readfile = 'test7.cor'
    #    readfile = 'test10.cor'
    import time
    import os

    readfile = os.path.join("./Examples/strain_calib_test2",
                                    "dat_Wmap_WB_14sep_d0_500MPa_0045_LT_0.cor")

    key_material = "W"
    emin = 5
    emax = 22

    nbGrainstoFind = "max"  # '1 #'max'

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # MatchingRate_Threshold_Si = 30  # percent for simulated Si
    dictimm_Si = {
        "sigmagaussian": (0.5, 0.5, 0.5),
        "Hough_init_sigmas": (1, 0.7),
        "Hough_init_Threshold": 1,
        "rank_n": 20,
        "useintensities": 0,
    }

    # MatchingRate_Threshold_W = 15  # percent W
    dictimm_W = {
        "sigmagaussian": (0.5, 0.5, 0.5),
        "Hough_init_sigmas": (1, 0.7),
        "Hough_init_Threshold": 1,
        "rank_n": 40,
        "useintensities": 0,
    }

    # MatchingRate_Threshold = MatchingRate_Threshold_W
    dictimm = dictimm_W

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "testSirandom"
        #        nbgrains = 2
        removespots = np.zeros(nbgrains)
        addspots = 0
        #        removespots = None
        #        addspots = None
        file_to_index = createFakeData(key_material,
                                        nbgrains,
                                        outputfilename=outputfilename,
                                        removespots=removespots,
                                        addspots=addspots,
                                        twins=twins)
    else:
        file_to_index = readfile

    #    t0 = time.time()
    #
    #    DataSet = spotsset()
    #
    #    DataSet.IndexSpotsSet(file_to_index, key_material, emin, emax, dictimm, database,
    #                          IMM=True,
    #                          nbGrainstoFind=nbGrainstoFind)
    #
    #    tf1 = time.time()
    #    print "imageMatching execution time %.3f sec." % (tf1 - t0)
    #
    #    DataSet.plotallgrains()

    # IAM technique
    print("\n\n\n\n\n")
    print("ANGLES LUT ---------------------------------------------")
    print("\n\n\n\n\n")

    t0_2 = time.time()
    DataSet_2 = ISS.spotsset()

    DataSet_2.IndexSpotsSet(file_to_index,
                            key_material,
                            emin,
                            emax,
                            dictimm,
                            database,
                            IMM=False,
                            MatchingRate_List=[40, 50, 60],
                            nbGrainstoFind=nbGrainstoFind)

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_2.plotallgrains()
    return DataSet_2  # , DataSet_2


def test_old_imagematching(database):
    """
    test refinement and multigrains indexing with class
    """
    readfile = "test.cor"

    DictLT.dict_Materials["DIAs"] = ["DIAs", [3.16, 3.16, 3.16, 90, 90, 90], "dia"]
    firstmatchingtolerance = 1.0

    if 1:

        key_material = "Si"
        emin = 5
        emax = 25

        MatchingRate_Threshold = 30  # percent

        dictimm = {"sigmagaussian": (0.5, 0.5, 0.5),
                    "Hough_init_sigmas": (1, 0.7),
                    "Hough_init_Threshold": 1,
                    "rank_n": 20,
                    "useintensities": 0}

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "testSirandom"
        nbgrains = 3
        file_to_index = createFakeData(key_material, nbgrains, outputfilename=outputfilename)
    else:
        file_to_index = readfile

    # read spots data and init dictionary of indexed spots
    DataSet = ISS.spotsset()
    DataSet.importdatafromfile(file_to_index)

    TwiceTheta_Chi_Int = DataSet.getSpotsExpData()
    totalnbspots = len(TwiceTheta_Chi_Int[0])
    # Find the best orientations and gives a table of results

    DataSet.key_material = key_material
    DataSet.emin = emin
    DataSet.emax = emax

    # potential orientation solutions from image matching
    bestEULER = IMM.bestorient_from_2thetachi(TwiceTheta_Chi_Int, database, dictparameters=dictimm)

    bestEULER, _ = ISS.filterEulersList(bestEULER,
                                        TwiceTheta_Chi_Int,
                                        key_material,
                                        emax,
                                        rawAngularmatchingtolerance=firstmatchingtolerance,
                                        MatchingRate_Threshold=MatchingRate_Threshold,
                                        verbose=1)

    MatrixPile = bestEULER.tolist()

    matrix_to_test = True
    grain_index = 0  # current grain to be indexed

    indexedgrains = []

    selectedspots_index = None

    AngleTol_0 = 1.0
    while matrix_to_test:

        if len(MatrixPile) != 0:
            eulerangles = MatrixPile.pop(0)
            # trialmatrix = GT.fromEULERangles_toMatrix(eulerangles)
        else:
            print("no more orientation to test (euler angles)")
            print("need to use other indexing techniques, Angles LUT, cliques help...")
            return

        #        #add a matrix in dictionary
        #        DataSet.dict_grain_matrix[grain_index] = trialmatrix

        print("dict_grain_matrix before initial indexing with a raw orientation")
        print(DataSet.dict_grain_matrix)

        # indexation of spots with raw (un refined) matrices & #update dictionaries
        print("\n\n---------------indexing grain  #%d-----------------------"
            % grain_index)
        print("eulerangles", eulerangles)
        #        print "selectedspots_index", selectedspots_index
        #        print "making raw links for grain #%d" % grain_index
        # TODO update
        DataSet.AssignHKL(eulerangles,
                        grain_index,
                        AngleTol_0,
                        use_spots_in_currentselection=selectedspots_index)

        #        print "dict_grain_matrix after raw matching", DataSet.dict_grain_matrix
        #        print "dict_matching_rate", DataSet.dict_grain_matching_rate

        # plot
        if DataSet.dict_grain_matrix is not None:
            DataSet.plotgrains(exp_data=TwiceTheta_Chi_Int,
                titlefig="%d Exp. spots data (/%d) after raw matching grain #%d"
                % (len(TwiceTheta_Chi_Int[0]), totalnbspots, grain_index))
        else:
            print("plot the best last orientation matrix candidate")
            IMM.Plot_compare_2thetachi(eulerangles,
                                    TwiceTheta_Chi_Int[0],
                                    TwiceTheta_Chi_Int[1],
                                    verbose=1,
                                    key_material=DataSet.key_material,
                                    emax=DataSet.emax,
                                    EULER=1)

            IMM.Plot_compare_gnomondata(eulerangles,
                                        TwiceTheta_Chi_Int[0],
                                        TwiceTheta_Chi_Int[1],
                                        verbose=1,
                                        key_material=DataSet.key_material,
                                        emax=DataSet.emax,
                                        EULER=1)

        AngleTol_List = [0.5, 0.2, 0.1]

        # TODO: data flow and branching will be improved later
        for k, AngleTol in enumerate(AngleTol_List):

            print("\n\n refining grain #%d step -----%d\n" % (grain_index, k))
            #    print "initial matrix", dict_grain_matrix[grain_index]

            refinedMatrix = DataSet.refineUBSpotsFamily(grain_index,
                                        DataSet.dict_grain_matrix[grain_index], use_weights=1)

            if refinedMatrix is None:
                break

            # extract data not yet indexed or temporarly indexed
            print("\n\n---------------extracting data-----------------------")
            toindexdata = DataSet.getUnIndexedSpotsallData(exceptgrains=indexedgrains)
            absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
            intensity_data = toindexdata[:, 5]
            absoluteindex = np.array(absoluteindex, dtype=np.int)

            TwiceTheta_Chi_Int = np.array([twicetheta_data, chi_data, intensity_data])

            #            print "spots index to index", toindexdata[:20, 0]
            #            print "twicetheta_data", toindexdata[:5, 1]

            print("\n\n---------------extracting data-----------------------")
            indexation_res, nbtheospots, _ = DataSet.getSpotsLinks(refinedMatrix,
                                                                exp_data=TwiceTheta_Chi_Int,
                                                                useabsoluteindex=absoluteindex,
                                                                removeharmonics=1,
                                                                veryclose_angletol=AngleTol,
                                                                verbose=0)

            #        print 'nb of links', len(indexation_res[1])

            if indexation_res is None:
                print("no unambiguous close links between exp. and  theo. spots have been found!")
                break

            # TODO: getstatsonmatching before updateindexation... ?
            nb_updates = DataSet.updateIndexationDict(
                indexation_res, grain_index, overwrite=1)

            print("\nnb of indexed spots for this refined matrix: %d / %d"
                % (nb_updates, nbtheospots))
            print("with tolerance angle : %.2f deg" % AngleTol)

            Matching_rate = 100.0 * nb_updates / nbtheospots

            if Matching_rate < 50.0:
                # matching rate too low
                # then remove previous indexed data and launch again imagematching
                print("matching rate too low")
                # there are at least one indexed grain
                # so remove the corresponding data and restart the imagematching
                if len(indexedgrains) > 0:

                    print("\n\n---------------------------------------------")
                    print(
                        "Use again imagematching on purged data from previously indexed spots")
                    print("---------------------------------------------\n\n")

                    #                    print "indexedgrains", indexedgrains

                    # extract data not yet indexed or temporarly indexed
                    toindexdata = DataSet.getUnIndexedSpotsallData(
                        exceptgrains=indexedgrains)
                    absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
                    intensity_data = toindexdata[:, 5]

                    TwiceTheta_Chi_Int = np.array(
                        [twicetheta_data, chi_data, intensity_data])

                    # potential orientation solutions from image matching
                    bestEULER2 = IMM.bestorient_from_2thetachi(
                        TwiceTheta_Chi_Int, database, dictparameters=dictimm)

                    bestEULER2, _ = ISS.filterEulersList(bestEULER,
                                                    TwiceTheta_Chi_Int,
                                                    key_material,
                                                    emax,
                                                    rawAngularmatchingtolerance=firstmatchingtolerance,
                                                    MatchingRate_Threshold=MatchingRate_Threshold,
                                                    verbose=1)

                    # overwrite (NOT add in) MatrixPile
                    MatrixPile = bestEULER2.tolist()

                    selectedspots_index = absoluteindex

                    print("working with a new set of trial orientations")

                    # exit the for loop of refinement and start with other set of spots
                    break

                # no data to remove
                else:
                    break

            # matching rate is high than threshold
            else:
                # keep on refining and reducing tolerance angle
                # or for the lowest tolerance angle, consider the refinement-indexation is completed
                DataSet.dict_grain_matrix[grain_index] = refinedMatrix
                DataSet.dict_grain_matching_rate[grain_index] = [nb_updates, Matching_rate]

                DataSet.plotgrains(
                    exp_data=DataSet.getSpotsExpData(),
                    titlefig="%d Exp. spots data (/%d), after refinement of grain #%d at step #%d"
                    % (totalnbspots, totalnbspots, grain_index, k))

                # if this is the last tolerance step
                if k == len(AngleTol_List) - 1:

                    print("\n---------------------------------------------")
                    print("indexing completed for grain #%d with matching rate %.2f "
                        % (grain_index, Matching_rate))
                    print("---------------------------------------------\n")
                    indexedgrains.append(grain_index)
                    grain_index += 1


#                    spots_G0 = DataSet.getSpotsFamily(0)
#                    print "len(G0)", len(spots_G0)
#                    print "G0", spots_G0


def test(database):
    """
    test refinement and multigrains indexing with class
    """

    #    readfile = 'test6.cor'
    #    readfile = 'test7.cor'
    readfile = "test11.cor"
    #    readfile = None

    DictLT.dict_Materials["DIAs"] = ["DIAs", [3.16, 3.16, 3.16, 90, 90, 90], "dia"]

    key_material = "Si"
    emin = 5
    emax = 25

    MatchingRate_Threshold = 30  # percent

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    dictimm = {"sigmagaussian": (0.5, 0.5, 0.5),
        "Hough_init_sigmas": (1, 0.7),
        "Hough_init_Threshold": 1,
        "rank_n": 20,
        "useintensities": 0}

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "testSirandom"
        nbgrains = 5
        removespots = [9, 8, 15, 6, 2]
        addspots = 6
        #        removespots = None
        #        addspots = None
        file_to_index = createFakeData(key_material,
                                        nbgrains,
                                        outputfilename=outputfilename,
                                        removespots=removespots,
                                        addspots=addspots)
    else:
        file_to_index = readfile

    # read spots data and init dictionary of indexed spots
    DataSet = ISS.spotsset()
    DataSet.importdatafromfile(file_to_index)
    totalnbspots = DataSet.nbspots

    DataSet.setMaterial(key_material)
    DataSet.setEnergyBand(emin, emax)

    DataSet.setImageMatchingParameters(dictimm, database)

    grain_index = 0  # current grain to be indexed
    AngleTol_0 = 1.0
    AngleTol_List = [0.5, 0.2, 0.1]
    PLOTRESULTS = 0
    VERBOSE = 0

    provideNewMatrices = True
    indexgraincompleted = False

    while 1:

        nb_remaining_spots = len(DataSet.getUnIndexedSpots())

        print("\n nb of spots to index  : %d\n" % nb_remaining_spots)
        if nb_remaining_spots < 2:
            print("%d spots have been indexed over %d"
                % (totalnbspots - nb_remaining_spots, totalnbspots))
            print("indexing rate is --- : %.1f percents"
                % (100.0 * (totalnbspots - nb_remaining_spots) / totalnbspots))
            print("indexation of %s is completed" % DataSet.filename)
            #            self.dict_grain_matching_rate[grain_index] = [0, 0]
            break

        if 1:
            print("start to index grain #%d" % grain_index)

        if provideNewMatrices:
            # potential orientation solutions from image matching
            print("providing new set of matrices")

            (bestUB, _, nbspotsIMM) = DataSet.getOrients_ImageMatching(
                                                            MatchingRate_Threshold=MatchingRate_Threshold,
                                                            exceptgrains=DataSet.indexedgrains,
                                                            verbose=VERBOSE)

            print("\n working with a new stack of orientation matrices")

            # update (overwrite) candidate orientMatrix object list
            DataSet.UBStack = bestUB

        if DataSet.UBStack is not None and len(DataSet.UBStack) != 0:
            print("%d Matrices are candidates in the Matrix Pile !" % len(DataSet.UBStack))
            if VERBOSE:
                print("\n  -----   Taking a new matrix from the matrices stack  -------")
            UB = DataSet.UBStack.pop(0)

        else:
            print("matrices stack to test is empty")
            print("%d spot(s) have not been indexed" % nb_remaining_spots)

            #            print "You may need to use other indexing techniques, Angles LUT, cliques help..."
            break

        # add a matrix in dictionary
        DataSet.dict_grain_matrix[grain_index] = UB.matrix

        #        print "dict_grain_matrix before initial indexing with a raw orientation"
        #        print DataSet.dict_grain_matrix

        # indexation of spots with raw (un refined) matrices & #update dictionaries
        print("\n\n---------------indexing grain  #%d-----------------------" % grain_index)
        if VERBOSE:
            print("eulerangles", UB.eulers)

        # select data, link spots, update spot dictionary, update matrix dictionary
        DataSet.AssignHKL(UB.eulers, grain_index, AngleTol_0, verbose=VERBOSE)

        # plot
        if PLOTRESULTS:
            if DataSet.dict_grain_matrix is not None:
                DataSet.plotgrains(
                    exp_data=DataSet.TwiceTheta_Chi_Int,
                    titlefig="%d Exp. spots data (/%d) after raw matching grain #%d"
                                    % (nbspotsIMM, totalnbspots, grain_index))

        print("\n\n---------------refining grain orientation #%d-----------------" % grain_index)
        # TODO: data flow and branching will be improved later
        for k, AngleTol in enumerate(AngleTol_List):
            if VERBOSE:
                print("\n\n refining grain #%d step -----%d\n" % (grain_index, k))

            refinedMatrix = DataSet.refineUBSpotsFamily(grain_index,
                                                        DataSet.dict_grain_matrix[grain_index],
                                                        use_weights=1,
                                                        verbose=VERBOSE)

            if refinedMatrix is not None:

                UBrefined = ISS.OrientMatrix(matrix=refinedMatrix)
                #            print "UBrefined", UBrefined.matrix
                DataSet.dict_grain_matrix[grain_index] = UBrefined.matrix
                # select data, link spots, update spot dictionary, update matrix dictionary
                Matching_rate, _, _ = DataSet.AssignHKL(UBrefined,
                                                                            grain_index,
                                                                            AngleTol,
                                                                            verbose=VERBOSE)

            if Matching_rate < 50.0 or refinedMatrix is None:
                # matching rate too low
                # then remove previous indexed data and launch again imagematching
                if 1:
                    print("matching rate too low")

                DataSet.resetSpotsFamily(grain_index)

                # there are at least one indexed grain and this dataset has not been already tested with the matrix stack
                # so remove the corresponding data and restart the imagematching
                if len(DataSet.indexedgrains) > 0 and indexgraincompleted:
                    # exit the 'for' loop of refinement and start with other set of spots
                    provideNewMatrices = True
                    indexgraincompleted = False
                    print("Need to re apply imagematching on purged data")
                    break

                # no data to remove
                # keep on trying matrix in matrix stack
                else:
                    provideNewMatrices = False
                    print("Need to look at the next matrix")
                    break

            # matching rate is higher than threshold
            else:
                if PLOTRESULTS:
                    DataSet.plotgrains(
                        exp_data=DataSet.getSpotsExpData(),
                        titlefig="%d Exp. spots data (/%d), after refinement of grain #%d at step #%d"
                        % (totalnbspots, totalnbspots, grain_index, k))

                # if this is the last tolerance step
                if k == len(AngleTol_List) - 1:

                    if 1:
                        print("\n---------------------------------------------")
                        print("indexing completed for grain #%d with matching rate %.2f "
                            % (grain_index, Matching_rate))
                        print("---------------------------------------------\n")

                    DataSet.indexedgrains.append(grain_index)
                    provideNewMatrices = False
                    indexgraincompleted = True
                    grain_index += 1

    return DataSet


# ------------  Main            ------------------------------

if __name__ == "__main__":

    database = IMM.DataBaseImageMatchingSi()

    test_ImageMatching_index(database=database, nbgrains=3)
    test_ImageMatching_index(database=database, nbgrains=3, readfile="testSirandom.cor")

    #    test_speedbuild() # to be used only with python (not ipython)

    #
    #    plotHough_Simul([5, 37, 9])
    #    plotHough_Exp('dat_Ge', 1, 100, col_I=4, dirname='./Examples/Ge/')

    #    plotHough_compare([20, 10, 50], 'euler201050_', 0, 100)

    #    position = Hough_peak_position_fast([0, 0, 0], Nb=60,
    #                                               pos2D=0, removedges=2, blur_radius=0.5,
    #                                               key_material='Si',
    #                                                arraysize=(300, 360),
    #                                                verbose=0,
    #                                                EULER=1,
    #                                                returnfilterarray=0)

    #    pos, fH, raw = plotHough_Simul([20, 10, 50], emax=25, NB=60, pos2D=0, removedges=2)

    #    plotHough_Exp('euler201050_', 2, 10000000)
    #    pos, fH, raw = plotHough_Simul([20, 10, 50], emax=25, NB=80, pos2D=0, removedges=2)

    #    plotHough_compare([20, 10, 50], 'euler201050_0002.cor', 100000)

    #    plotHough_compare([20, 10, 50], 'testmatch.cor', 100000)

    #    correlintensity, argcorrel = plotHoughCorrel(database, 'euler201050_0002.cor',
    #                                                 100000,dirname='.', returnCorrelArray=1)
    #
    #    plottab(correlintensity)

    """
    # loading the best_orientation for a serie of file (containing 2theta, chi ,I and intensity sorted)
    # list of element: [index_file,array of 3 Euler Angles (sorted by correlation degree)]
    filou=open('best_orient_UO2_He_103to1275_300p','r')
    TBO=pickle.load(filou)
    filou.close
    print "Table of best orientation loaded in TBO"
    print "contains %d elements"%len(TBO)
    # each element in TBO has:
    #[0] file index
    #[1]  has :
    #   [0] array of three Euler angles (sorted by correlation importance)
    #   [1] (2theta array,chi array, Intensity array(possibly))
    # tip: transpose(TBO[index_in_TBO][1][0])[orient_rank] are the 3 Euler Angles of orientation ranled by orient_rank

    Very_Angles=[[] for k in range(len(TBO))]
    #Decision_threshold=0.0023 # 1000p
    Decision_threshold=0.003 # 300p 0.8 deg et 20 keV
    verbose=0

    for index_in_TBO in range(0,len(TBO)): # index of the file in table of best orientation frame (from 0 to len(TBO)-1)
        std_res=1
        nb_of_spots=100
        orient_rank=0
        nb_orient=len(transpose(TBO[index_in_TBO][1][0]))
        print "-------------------------------------"
        print "File index",index_in_TBO
        print "Nb of exp.spots",1000 # nb of exp spots used for finding the 20 first orientations
        print "nb of orientations",nb_orient
        while (orient_rank<nb_orient): # threshold to pursue the search of a correct orientation (good catch)
        #while (1.*std_res/nb_of_spots>=0.0023) and (nb_of_spots>=10) and (orient_rank<nb_orient): # threshold to pursue the search of a correct orientation (good catch)
        #while (std_res>=0.08) and (nb_of_spots>=10) and (orient_rank<nb_orient): # threshold to pursue the search of a correct orientation (good catch)
            if verbose:
                print "-------------------------------------"
                print "orient_rank",orient_rank
                print "EULER angles",transpose(TBO[index_in_TBO][1][0])[orient_rank]
            john=StickLabel_on_exp_peaks(transpose(TBO[index_in_TBO][1][0])[orient_rank],
                                        TBO[index_in_TBO][1][1][0],TBO[index_in_TBO][1][1][1],
                                        0.8,
                                        20,
                                        verbose=0,
                                        key_material='UO2',
                                        arraysize=(300,360),
                                        EULER=1)

            residues_angles=array(john[-2].values())
            mean_res_1=mean(residues_angles)
            maxi_res_1=max(residues_angles)
            std_res_1=std(residues_angles)
            nb_of_spots_1=len(residues_angles)

            if verbose:
                print "max ",maxi_res_1," mean ",mean_res_1," std_res",std_res_1," nb_of_spots",nb_of_spots_1
                print "std/nbspots",1.*std_res_1/nb_of_spots_1
            #removing far spots
            anormalous_spots_key=[]
            for k,v in john[-2].items():
                if v>=mean_res_1+1.5*sqrt(std_res_1):
                    anormalous_spots_key.append(k)
            for cle in anormalous_spots_key:
                print john[-1][cle]
                del john[-2][cle]
                del john[-1][cle]

            residues_angles=array(john[-2].values())
            mean_res=mean(residues_angles)
            maxi_res=max(residues_angles)
            std_res=std(residues_angles)
            nb_of_spots=len(residues_angles)
            if verbose:
                print "max ",maxi_res," mean ",mean_res," std_res",std_res," nb_of_spots",nb_of_spots
                print "std/nbspots",1.*std_res/nb_of_spots

            #print "cond",((std_res>=0.08) and (nb_of_spots>=10) and (orient_rank<20))
            if ((1.*std_res/nb_of_spots>=Decision_threshold)  and (nb_of_spots>=10) and (orient_rank<nb_orient))==False:
            #if ((std_res>=0.08) and (nb_of_spots>=10) and (orient_rank<nb_orient))==False:
                if verbose:
                    print "********-----------------------------------------"
                    print "Good! I ve found an orientation!"
                    print "********-----------------------------------------"
                Very_Angles[index_in_TBO].append(transpose(TBO[index_in_TBO][1][0])[orient_rank])


            orient_rank+=1
        print "Number of found orientations",len(Very_Angles[index_in_TBO])



        orient_EULER_angles_rank=orient_rank-1
        if nb_orient==orient_rank: # use avec while (1.*std_res/nb_of_spots>=0.0023) and (nb_of_spots>=10) and (orient_rank<nb_orient)
            #print "display by default the best found in TBO"
            orient_EULER_angles_rank=0

        # plot only exp labelled spots


        Plot_compare_2thetachi(transpose(TBO[index_in_TBO][1][0])[orient_EULER_angles_rank],
                            TBO[index_in_TBO][1][1][0],TBO[index_in_TBO][1][1][1],
                            verbose=1,key_material='UO2',EULER=1,
                            exp_spots_list_selection=array(john[-1].values()))

        # plot all exp spots

        Plot_compare_2thetachi(transpose(TBO[index_in_TBO][1][0])[orient_EULER_angles_rank],
                            TBO[index_in_TBO][1][1][0],TBO[index_in_TBO][1][1][1],
                            verbose=1,key_material='UO2',EULER=1,exp_spots_list_selection=None)


        # Working now only with remaining exp spots (those not labelled in previous sticking)
        #give_best_orientations('sUrHe',TBO[index_in_TBO][0],300, listselection=######) # 300 = most intense peaks
    """

    """

    # building the map
    # nb elem = 51*23

    list_of_pointsinstance=[]
    list_of_signs=[]
    dic_ID_orient={}
    grain_counter=0
    for m in range(51*23):
        XX=m/51
        YY=m%51
        p=Map_point([XX,YY])
        p.orients=refine_orient(Main_Orient[m],array([3,3,3]))
        list_of_pointsinstance.append(p)

        nb_orient=len(p.orients)
        #print m,nb_orient
        if nb_orient>0:
            point_sign=[]
            for g in range(nb_orient):
                #print g,list(p.orients[g])
                signature_g=from3angles_to_int(list(p.orients[g]))
                point_sign.append(signature_g)
                if signature_g not in dic_ID_orient.values():
                    dic_ID_orient[grain_counter]=signature_g
                    grain_counter+=1
            list_of_signs.append(point_sign)
            p.orients_ID=point_sign
        else:
            list_of_signs.append([-1])
            p.orients_ID=[-1]

    print "I ve found %d grain in the map"%len(dic_ID_orient)
    #adding -1 = None
    dic_ID_orient[-1]=-1
    print dic_ID_orient

    print "Now put the general ID grains in all the point of the map"
    # inverser le dictionnaire...
    def invdict(adict):

        return dict([(val, key) for key, val in adict.items()])

    dic_orient_ID=invdict(dic_ID_orient)

    # lists grain id for each points
    grain_per_point=[map(lambda elem: dic_orient_ID[elem],ls) for ls in list_of_signs]

    # gives for each grain ID, the point instances where they were found ----------------------------
    localisation_grain={}
    for ID in dic_ID_orient.keys():
        localisation_grain[ID]=[]
    for p in list_of_pointsinstance:
        map(lambda cle_ID: localisation_grain[dic_orient_ID[cle_ID]].append(p.pos),p.orients_ID)
    table_pos=[array(localisation_grain[ID]) for ID in dic_ID_orient.keys()]

    """

    # -----------------------------------------------------------------------------------------------
    def make_colormarkerdict():
        """
        build a dictionary of color and marker to distinguish contribution of grains in scatter plot
        """
        # faudrait rendre aleatoire l'indice i_dict pour un meilleur rendu
        colorlist = [
            "k",
            (1.0, 0.0, 0.5),
            "b",
            "g",
            "y",
            "r",
            "c",
            "m",
            "0.75",
            (0.2, 0.2, 0.2),
            (0.0, 0.5, 1.0),
        ]
        markerlist = ["s", "v", "h", "d", "o", "^", "8"]
        colormarkerdict = {}
        i_dict = 0
        for col in colorlist:
            for mark in markerlist:
                colormarkerdict[i_dict] = [col, mark]
                i_dict = i_dict + 1
        return colormarkerdict

    def plot_mapmajor(listoflocations, xmin, xmax, ymin, ymax):
        """
        example of map plot given list of coordinates ?? ( I do not remember actually)

        """
        import pylab as p

        p.close()
        colormarkerdict = make_colormarkerdict()

        tableofxcoord = []
        tableofycoord = []

        # move the last elem in first position the first grain ID corresponds now to unknown grain
        listoflocations.insert(0, listoflocations.pop(-1))

        for elem in (
            listoflocations[0],
            listoflocations[8],
            listoflocations[16],
            listoflocations[24],
            listoflocations[32],
        ):
            tableofxcoord.append(elem[:, 0])
            tableofycoord.append(elem[:, 1])

        # print "tableofxcoord",tableofxcoord
        import pylab as p

        p.figure(figsize=(10, 7), dpi=80)

        kk = 0
        for xdata, ydata in zip(tuple(tableofxcoord), tuple(tableofycoord)):
            p.scatter(
                ydata,
                xdata,
                c=colormarkerdict[kk][0],
                marker=colormarkerdict[kk][1],
                s=500,
                alpha=0.5,
            )
            # permutation of x y / array index ?...
            kk = kk + 1

        ticks_x = np.arange(xmin, xmax, 1)
        ticks_y = np.arange(ymin, ymax, 1)
        p.yticks(ticks_x)  # permutation of x y / array index ?...
        p.xticks(ticks_y)
        p.ylim(-1, xmax + 1)
        p.xlim(-1, ymax + 1)
        p.xlabel("x", fontsize=20)
        p.ylabel("y", fontsize=20)
        p.title("UO2_He_103_1275: Grain map. Unknown grains are black squares")
        p.grid(True)
        p.show()

    # plot_mapmajor(table_pos,0,22,0,50)

    #    def corro(central_angles, deltaangle):
    #        d_angle = np.array(deltaangle)
    #        elemangle = np.array(central_angles) + d_angle
    #        #elemangle=array(fromMatrix_to_elemangles(mamasol))
    #        #elemangle=-array(fromMatrix_to_elemangles(dot(array(mamasol),permu5)))
    #        print elemangle
    #
    #        nb, pos = Hough_peak_position(list(elemangle), key_material=29, returnXYgnomonic=0, arraysize=(300, 360), verbose=0, saveimages=0, saveposition=0, EULER=1, prefixname='Ref_H_FCC_')
    #        #scatter(xgnomon,ygnomon,c='r') #theory
    #        #scatter(gnomonx,gnomony,c='b',alpha=.5,s=40) #data
    #        #show()
    #
    #        #sumy=np.sum(blurredHough[pos])
    #        #sumy = np.sum(filteredHough[pos])
    #        #print sumy
    #        return sumy

    # corro([20.69,-11.23,9.26],[0,0,0])

    # intcorrel=[[    max([corro([20.69,-11.23,9.26],[l,j,k]) for l in arange(-10,10,1)])    for k in arange(-2,3,1)] for j in arange(-2,3,1)] #result optim_1
    # intcorrel=[corro([20.69,-11.23,9.26],[l,-1.,0]) for l in arange(-10,10,1)] # =>

    # 1024 pts et angle d'EULER
    # intcorrel=[[    max([corro([0,0,0],[l,j,k]) for l in arange(0,90,1)])    for k in arange(0,90,2)] for j in arange(0,90,2)]
    # intcorrel=[  corro([0,74,46],[k,0,0])  for k in arange(0,90,1.)] #=> k = 68
    # intcorrel=[  corro([0,80,18],[k,0,0])  for k in arange(0,90,1.)] #=> k = 56

    # 1024 pts et angle d'EULER

    # intcorrel=[[    max([corro([0,74,46],[l,j,k]) for l in arange(0,90,.5)])    for k in arange(-1,2,.5)] for j in arange(-1,2,.5)]

    """
    from np.random import *
    totalsizeHougharray=432000
    nboforientations=45*45*360/1.
    t0=time.time()
    bigpos=random_integers(0,totalsizeHougharray-1,(nboforientations,100))
    tr=time.time()
    print "time array creation",tr-t0
    john=map(lambda elem: np.sum(filteredHough[elem]),bigpos)

    tf=time.time()
    print size(bigpos)
    print "nboforientations",nboforientations
    print "time",tf-tr
    """

    """ VERY LONG
    def mysum(arra):
        if len(arra)>0:
            return np.sum(arra)
        else:
            return 0

    rhovalue=1.
    toleranceonrho=.1
    RHO=linspace(-1.05,1.05,42)
    ANGLE=linspace(0.,2*np.pi,100)
    angleflat=ANGLE.flat
    shapearray=shape(RHO)
    ACCum=np.zeros((42,100))
    i_accum=0
    for _rho in RHO.flat:
        print "i_accum =",i_accum
        j_accum=0
        for _angle in angleflat:
            #print "_angle",_angle
            #print "j_accum",j_accum
            summy=mysum(intensitytable[np.where(abs(xgnomon*cos(_angle)+ygnomon*sin(_angle)-_rho)<.005)])
            #print "summy",summy
            ACCum[i_accum,j_accum]=summy
            j_accum+=1
        i_accum+=1

    glu=open('Hough','r')
    import pickle
    glot=pickle.load(glu)
    glu.close()

    APRES COUP: rho est de maniere interessante entre bin 8 et 35
    """
