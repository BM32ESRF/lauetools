"""
scripts example to be launched from
"""
import os, copy
import numpy as np

import LaueTools
print("Using LaueToolsFolder: ", os.path.abspath(LaueTools.__file__))

import LaueTools.indexingSpotsSet as ISS
import LaueTools.matchingrate
import LaueTools.generaltools as GT
import LaueTools.findorient as FindO
import LaueTools.CrystalParameters as CP
import LaueTools.IOLaueTools as IOLT
import LaueTools.dict_LaueTools as DictLT
import LaueTools.indexingAnglesLUT as IAL
import LaueTools.lauecore as LAUE


def test_GetProximity():

    nexp = 3

    shufflenexp = np.random.permutation(nexp)

    TwicethetaChi = np.ascontiguousarray(
        np.random.randint(40, high=140, size=(2, 20)), dtype=np.float
    )

    data_theta = TwicethetaChi[0] / 2.0
    data_chi = TwicethetaChi[1] + 0.4

    exp_theta, exp_chi = data_theta[shufflenexp], data_chi[shufflenexp]

    res = matchingrate.getProximity(
        TwicethetaChi, exp_theta, exp_chi, angtol=0.5, proxtable=0, verbose=0, signchi=1
    )

    # allresidues, res, nb_in_res, len(allresidues), meanres, maxi

    print(res)
    return res, TwicethetaChi[0] / 2.0, exp_theta, TwicethetaChi[1], exp_chi


def test_GetProximity2():

    simuldata = np.array( [ [59.2164, -66.6041], [54.7398, -88.3875], [43.0763, -83.0983], [58.895, -88.7721], [39.3337, -65.6132], [52.4121, -58.2399], [47.6515, -100.4466], [47.9575, -83.2749], [44.0163, -64.8536], [56.4791, -57.2717], [48.434, -115.549], [52.2976, -101.5099], [28.9731, -75.0536], [53.191, -83.4718], [49.0293, -64.0107], [41.0373, -47.0408], [52.3338, -117.1192], [37.1386, -98.1813], [58.7918, -83.6928], [54.3839, -63.0704],
            [55.9659, -41.1327], [39.5682, -118.0816], [56.3583, -118.8114], [41.6482, -74.3025], [49.9809, -43.6082], [53.2592, -130.741], [44.4802, -120.2579], [49.9426, -100.9654], [38.9334, -47.8105], [54.7471, -41.6504], [56.4465, -132.6251], [49.6174, -122.6474], [21.8155, -95.1273], [57.3701, -73.2746], [46.408, -26.3132], [55.4884, -25.5038], [46.0081, -136.7264], [59.6475, -134.6084], [28.2351, -126.1327], [54.9433, -125.2734], [31.5495, -97.0398], [22.4182, -53.4955], [52.3414, -42.6513], [36.0373, -24.6759],
            [50.0743, -24.0159], [58.489, -23.5838], [35.0685, -129.9977], [43.26, -99.4789], [32.9485, -49.9361], [40.7146, -21.6059], [53.7612, -21.5865], [32.8305, -146.1445], [53.4871, -141.8828], [42.2965, -134.3526], [57.1994, -102.6829], [45.4058, -45.3993], [21.3158, -17.0592], [45.4556, -18.3182], [57.4384, -19.0229], [37.2146, -149.6308], [49.749, -139.236], [27.7533, -12.0696], [50.1926, -14.8127], [50.5507, -152.2943], [41.5523, -153.2806], [57.1871, -144.6661], [34.2669, -6.6991],
            [54.8492, -11.096], [53.1766, -154.6529], [45.7737, -157.0741], [12.2809, -166.6089], [40.6313, -1.0068], [59.3447, -7.1823], [52.0084, -4.6259], [55.7231, -157.0654], [49.8098, -160.985], [22.0916, -176.2917], [46.612, 4.9165], [43.6835, 1.9328], [54.5417, -2.2204], [42.4339, -165.4599], [58.1746, -159.525], [53.5979, -164.9811], [30.6904, 174.3774], [52.0051, 10.9533], [56.9817, 0.2302], [45.1098, -168.3128], [26.5826, 178.968], [57.086, -169.0252], [37.6256, 165.8261], [15.6917, 39.6224], [56.6694, 16.9722], [32.5476, 15.6023], [49.3926, 7.9288], [59.3144, 2.7178], [47.6427, -171.1734], [42.8967, 158.285], [22.5114, 46.4513], [35.8165, 19.2694],
            [50.0185, -174.0275], [34.3748, 169.985], [46.7627, 151.7992], [27.9587, 52.2656], [38.8316, 22.8625], [54.4341, 13.9733], [52.2268, -176.8611], [49.55, 146.2952], [55.2312, 100.1347], [32.2566, 57.1803], [41.5781, 26.3568], [54.2617, -179.6607], [40.4564, 161.9228], [51.5498, 141.6466], [35.6456, 61.3316], [44.0522, 29.7315], [58.7058, 19.9338], [56.121, 177.5861], [52.9876, 137.7171], [38.334, 64.8498], [46.2592, 32.9705], [57.8061, 174.8905], [44.9849, 154.9128], [54.0273, 134.3813], [40.4874, 67.8483], [48.2113, 36.0624],
            [59.3215, 172.2623], [54.7844, 131.5323], [42.2315, 70.4209], [49.9255, 39.0003], [41.3115, 160.6805], [48.2716, 148.9317], [55.3394, 129.0823], [43.6605, 72.6439], [51.4215, 41.7811], [55.7485, 126.9602], [44.8447, 74.5785], [52.7206, 44.4051], [44.3254, 156.0077], [50.6327, 143.8726], [56.0512, 125.1091], [45.8366, 76.2738], [53.8438, 46.875], [45.6099, 153.8468], [56.2755, 123.4835], [46.6758, 77.7691], [54.8115, 49.1957], [52.3274, 139.6002], [56.4416, 122.0467], [47.3926, 79.0961], [55.6426, 51.3733],
            [56.5638, 120.7692], [48.01, 80.2805], [56.3545, 53.4149], [53.549, 135.9821], [56.6529, 119.6269], [48.5461, 81.3433], [56.9627, 55.3281], [56.7166, 118.6003], [49.0149, 82.3015], [57.4811, 57.1208], [54.4355, 132.9019], [56.761, 117.6731], [49.4277, 83.1695], [57.9219, 58.8009], [56.7903, 116.8321], [49.7933, 83.9591], [58.2956, 60.376], [55.0832, 130.2622], [56.8081, 116.066], [50.1191, 84.6802], [58.6116, 61.8535], [56.8168, 115.3656], [50.4108, 85.3411], [58.8779, 63.2404],
            [55.5594, 127.9841], [56.8185, 114.7228], [59.1012, 64.5432], [56.8147, 114.1311], [56.8067, 113.5847], ]
    )

    expdata = np.ascontiguousarray(
        np.array(
            [
                [45.44929, -18.34126],
                [45.157076, -137.083181],
                [44.92114, 154.84827],
                [27.931473, 52.277335],
                [35.129301, -129.958303],
                [22.506704, 46.525391],
                [45.805636, -157.105433],
                [27.76149, -12.146467],
                [22.099332, -176.308319],
                [21.854819, -95.128858],
            ]
        ),
        dtype=np.int32,
    )

    TwicethetaChi = np.ascontiguousarray(simuldata, dtype=np.int32).T

    exp_theta, exp_chi = expdata[:, 0] / 2.0, expdata[:, 1]

    res = matchingrate.getProximity(
        TwicethetaChi, exp_theta, exp_chi, angtol=0.5, proxtable=0, verbose=0, signchi=1
    )

    # allresidues, res, nb_in_res, len(allresidues), meanres, maxi

    print(res)
    return res, TwicethetaChi[0] / 2.0, exp_theta / 2.0, TwicethetaChi[1], exp_chi


def Test_Build_and_Query_LUT():
    """
    test building and query of an angular look up table from a crystal
    given its lattice parameters

    """
    # test
    n = 3
    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n)

    if 1:  # filterharmonics:
        hkl_all = FindO.FilterHarmonics(hkl_all)

    #    Gstar_metric = CP.Gstar_from_directlatticeparams(2, 1, 4, 75, 90, 120)
    Gstar_metric = CP.Gstar_from_directlatticeparams(5.43, 5.43, 5.43, 90, 90, 90)
    # GenerateLookUpTable
    LUT = FindO.GenerateLookUpTable(hkl_all, Gstar_metric)

    print(FindO.QueryLUT(LUT, 14.01, 1))


def test_indexation_with_apriori_Matrices():
    """
    test of indexation + adding some selected matrices
    """
    folderdata = "./Examples/UO2/"
    filename_data = "dat_UO2_A163_2_0028_LT_0.cor"
    filename_data = os.path.join(folderdata, filename_data)
    StructureLabel = "UO2"

    nb_of_spots = 2000  # nb of spots to select for recognition
    nbspots_plot = 300  # nb of spots to select for plotting

    # READ .cor file
    mydata = IOLT.ReadASCIIfile(
        filename_data, col_2theta=0, col_chi=1, col_Int=-1, nblineskip=1
    )
    data_theta, data_chi, data_I = mydata

    print(data_I)
    print(data_theta)

    nbp = len(data_theta)
    # listofselectedpts=arange(nbofpeaks-1) # 1 for 1 header line
    # print data_I

    # --- Selection of peaks
    if nb_of_spots > 0:
        upto = min(nb_of_spots, nbp)
        sorted_int_index = np.argsort(data_I)[::-1][: upto + 1]
        print("Considering only %d most intense spots" % upto)

    elif nb_of_spots == -1:
        print("Considering all spots")
        sorted_int_index = np.argsort(data_I)[::-1]

    #    listofselectedpts = np.arange(len(sorted_int_index))
    #    indicespotmax=len(listofselectedpts)

    Theta = data_theta[sorted_int_index]
    Chi = data_chi[sorted_int_index]
    Intens = data_I[sorted_int_index]
    sorted_data = np.array([Theta, Chi, Intens]).T

    print(sorted_data[:10])

    # --- Rough interangular distance recognition

    # array of interangular distance of all points
    print("Calculating all angular distances\n")
    Tabledistance = GT.calculdist_from_thetachi(
        sorted_data[:, 0:2], sorted_data[:, 0:2]
    )

    ind_sorted_LUT_MAIN_CUBIC = [np.argsort(elem) for elem in FindO.LUT_MAIN_CUBIC]
    sorted_table_angle = []
    for k in list(range(len(ind_sorted_LUT_MAIN_CUBIC))):
        # print len(LUT_MAIN_CUBIC[k])
        # print len(ind_sorted_LUT_MAIN_CUBIC[k])
        sorted_table_angle.append(
            (FindO.LUT_MAIN_CUBIC[k])[ind_sorted_LUT_MAIN_CUBIC[k]]
        )

    # --- getOrientMatrices   with (spot_index_central,energy_max)
    spot_index = [ 0, 1, 21, 22, 24, ]  # one integer (starting from 0) or an array:array([1,8,9,5])
    energymax = 25
    nbmax_probed = (
        25
    )  # size of set of spots to calculate distance from central spot #spot_index
    nbspots_plot = 200  # for plot exp. data
    print("Using getOrientMatrices()")

    TrialsMatrix = (
        np.array(
            [
                [-0.99251179, 0.07104589, 0.09936213],
                [-0.01590782, -0.88248953, 0.47006294],
                [0.12126142, 0.46522732, 0.87684617],
            ]
        ),
        np.array(
            [
                [-0.49751201, -0.64141178, 0.58401432],
                [0.19466032, -0.7385631, -0.6454703],
                [0.84535393, -0.20737935, 0.49231143],
            ]
        ),
        np.eye(3),
    )

    TrialsMatrix = None

    # with nbbestplot very high  mama contain all matrices with matching rate above Nb_criterium

    n = 3  # for LUT
    latticeparams = DictLT.dict_Materials[StructureLabel][1]
    B = CP.calc_B_RR(latticeparams)

    mama, hhh = IAL.getOrientMatrices(
        spot_index,
        energymax,
        Tabledistance[:nbmax_probed, :nbmax_probed],
        Theta,
        Chi,
        n,
        B,
        LUT_tol_angle=0.5,
        MR_tol_angle=0.2,
        Minimum_Nb_Matches=40,
        key_material=StructureLabel,
        nbspots_plot=nbspots_plot,
        nbbestplot=20,
        plot=0,
        addMatrix=TrialsMatrix,
    )

    print("All %d matrices solutions" % len(mama))
    print(mama)
    print("Corresponding Scores")
    print(hhh)

    MAMA, SCO = ISS.MergeSortand_RemoveDuplicates(
        mama, hhh, 80, tol=0.0001, keep_only_equivalent=True
    )

    print(MAMA)
    print(SCO)


def test_indexation_generalRefunitCell_1():
    """
    test of indexation and unit cell refinement

    # lattice parameter : 2 1 4 75 90 120
    """
    B_input = np.array(
        [
            [0.58438544724018371, 0.62634243462603045, -0.040527337709274316],
            [0.0, 1.0352761804100832, -0.06698729810778066],
            [0.0, 0.0, 0.24999999999999997],
        ]
    )

    data_filecor = np.array(
        [
            [94.67479705, -1.77030003, 921.5841064, 877.0715942, 72.34869384],
            [73.69339752, 20.61630058, 572.9110107, 1218.955322, 72.01788330],
            [102.4393997, 8.985199928, 756.9523925, 755.0565795, 71.78106689],
            [80.51010131, 30.81780052, 381.7903137, 1118.804809, 69.76594543],
            [83.94760131, -11.9162998, 1077.943359, 1040.604248, 68.99935150],
            [63.29959869, 9.497099876, 754.0991210, 1386.598754, 68.80802154],  # 1 1 3
            [107.3254013, 19.63260078, 583.6207275, 661.9633789, 68.60917663],  # 0 1 3
            [84.57730102, 39.66809844, 180.0175933, 1058.541748, 64.89128112],
            [109.8495025, 29.49519920, 401.0426940, 589.9348754, 64.09487915],
            [111.0460968, -11.4846000, 1070.015625, 605.5197753, 62.07760238],
            [117.7855987, -1.76250004, 919.4577026, 488.4346923, 61.40348434],
            [70.78790283, -20.9498996, 1226.293212, 1267.149902, 61.09899902],
            [102.2435989, -20.2021999, 1213.381591, 745.6085205, 60.87683105],  # 1 1 2
            [122.3706970, 8.517900466, 761.3325805, 389.1836853, 59.37057495],
            [86.79679870, 47.08269882, -33.2585983, 1024.272338, 59.23816299],
            [110.6781005, 38.15859985, 208.5700988, 533.8281860, 59.13156890],
            [49.21139907, -1.77890002, 925.8856811, 1685.104736, 57.60239410],
            [91.72329711, -27.7292995, 1349.845825, 916.0936279, 57.46092987],  # 0 1 2
            [124.9682998, 18.72879981, 595.1906127, 304.0439147, 56.49592971],  # 2 1 3
        ]
    )

    mat_tofind = DictLT.dict_Rot["OrientSurf111"]

    a, b, c, alpha, beta, gamma = 2.0, 1.0, 4.0, 75.0, 90.0, 120.0
    #    latticeparams = a,b,c,alpha,beta,gamma
    B = CP.calc_B_RR([a, b, c, alpha, beta, gamma])
    print("B", B)
    # --- metric tensor
    #    Gstar_metric = CP.Gstar_from_directlatticeparams(a,b,c,alpha,beta,gamma)

    twiceTheta, Chi = data_filecor[:, :2].T

    thechi_exp = np.array([twiceTheta / 2.0, Chi]).T

    table_angdist = GT.calculdist_from_thetachi(thechi_exp, thechi_exp)

    ang_tol = 1
    n = 3
    spot_index = 5

    mat = IAL.matrices_from_onespot_new(
        spot_index, ang_tol, table_angdist, twiceTheta, Chi, n, B, verbose=1
    )

    index_matching = []
    for k, mat_sol in enumerate(mat[0]):
        if np.allclose(mat_sol, mat_tofind, atol=0.2):
            index_matching.append([mat[0][k], mat[1][k], mat[2][k]])

    print("These are the solutions !...")
    print(index_matching)


def test_indexation_generalRefunitCell_2():
    """
    test of indexation and unit cell refinement

    [2,1,4,90,90,90] lattice parameter oriented with orientsurface111
    """
    data_filecor = np.array(
        [
            [79.98930358, -0.01720000, 896.3380737, 1100.323608, 59.85998535],
            [72.14089965, -7.87809991, 1015.881286, 1227.356567, 59.33337783],  # 2 1 4
            [85.72489929, 7.331699848, 784.4506835, 1013.795471, 58.13841629],  # 2 1 3
            [61.93239974, -15.9895000, 1143.677978, 1422.501708, 55.33495330],  # 2 1 5
            [61.84360122, 15.91269969, 653.3759155, 1426.837646, 55.26354980],  # 2 1 2
            [89.83450317, -14.0176000, 1111.113281, 949.4226074, 55.25106430],  # 1 1 4
            [89.75630187, 14.00380039, 679.5944213, 953.0634155, 55.21326446],  # 3 1 4
            [95.69380187, -6.89139986, 999.4022216, 860.4885864, 54.51536941],  # 2 1 6
            [82.30560302, -21.1537990, 1229.820800, 1071.203369, 54.36821365],  # 3 1 5
            [68.30660247, 22.44009971, 542.4238891, 1319.939941, 54.00218200],  # 3 1 3
            [52.94110107, 8.390600204, 772.7620849, 1601.591430, 53.40956115],
        ]
    )

    mat_tofind = DictLT.dict_Rot["OrientSurf111"]

    a, b, c, alpha, beta, gamma = 2.0, 1.0, 4.0, 90.0, 90.0, 90.0
    #    latticeparams = a,b,c,alpha,beta,gamma
    B = CP.calc_B_RR([a, b, c, alpha, beta, gamma])
    # --- metric tensor
    #    Gstar_metric = CP.Gstar_from_directlatticeparams(a,b,c,alpha,beta,gamma)

    twiceTheta, Chi = data_filecor[:, :2].T

    thechi_exp = np.array([twiceTheta / 2.0, Chi]).T

    table_angdist = GT.calculdist_from_thetachi(thechi_exp, thechi_exp)

    ang_tol = 1
    n = 4
    spot_index = 1

    mat = IAL.matrices_from_onespot_new(
        spot_index, ang_tol, table_angdist, twiceTheta, Chi, n, B, verbose=0
    )

    print("mat", mat)

    index_matching = []
    for k, mat_sol in enumerate(mat[0]):
        if np.allclose(mat_sol, mat_tofind, atol=0.2):
            index_matching.append([mat[0][k], mat[1][k], mat[2][k]])

    print("These are the solutions !...")
    print(index_matching)


def Test_getOrients_AnglesMatching():
    """
    test for the classical method implemented in Common_index_Method()
    """
    readfile = None

    spot_index_central = [0, 1, 2]
    key_material = "UO2"
    emax = 22

    nbmax_probed = 8000000

    angleTolerance_LUT = 0.1
    MatchingRate_Angle_Tol = 0.001
    Matching_Threshold_Stop = (50.0)  # minimum to stop the loop for searching potential solution

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "toto"
        nbgrains = 3
        removespots = [0, 0, 0]  # , 0, 3, 10, 2, 1, 5]
        addspots = 0
        #        removespots = None
        #        addspots = None
        file_to_index = createFakeData(key_material,
                                        nbgrains,
                                        outputfilename=outputfilename,
                                        removespots=removespots,
                                        addspots=addspots)
    else:
        file_to_index = readfile

    # read data
    data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(
        file_to_index)[1:]
    tth = 2 * data_theta

    exp_data = np.array([tth, Chi])

    thechi_exp = np.array([data_theta, Chi]).T

    table_angdist = GT.calculdist_from_thetachi(thechi_exp, thechi_exp)

    latticeparams = DictLT.dict_Materials[key_material][1]
    B = CP.calc_B_RR(latticeparams)

    nbmax_probed = min(nbmax_probed, len(data_theta))

    matrix, score, Threshold_reached = IAL.getOrients_AnglesLUT(spot_index_central,
                                                    table_angdist[:nbmax_probed, :nbmax_probed],
                                                    tth,
                                                    Chi,
                                                    Matching_Threshold_Stop=Matching_Threshold_Stop,
                                                    n=3,  # up  to (332)
                                                    B=B,  # for cubic
                                                    LUT=None,
                                                    angleTolerance_LUT=angleTolerance_LUT,
                                                    MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                                                    key_material=key_material,
                                                    emax=emax,
                                                    verbose=0)
    #
    #    bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(bestmat, stats_res, Nb_criterium,
    #                                                           tol=0.0001, allpermu=True)

    print("B", B)
    print("matrix", matrix)
    return matrix, score, detectorparameters  # , exp_data


def plotgrains(listmat, key_material, emax, file_to_index="toto.cor"):
    """
    plot grains spots and data from orientmatrix list up to 9
    """
    # read data
    data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(
        file_to_index
    )[1:]
    tth_exp = 2 * data_theta
    Chi_exp = Chi

    EXTINCTION = DictLT.dict_Materials[key_material][2]

    nb_matrices = 0
    all_tthchi = []
    for mat in listmat:

        print("mat in plotgrains", mat)

        grain = [np.eye(3), EXTINCTION, mat, key_material]

        (Twicetheta_theo,
            Chi_theo,
            Miller_ind_theo,
            posx_theo,
            posy_theo,
            Energy_theo,
        ) = LAUE.SimulateLaue(grain, 5, emax, detectorparameters)

        all_tthchi.append([Twicetheta_theo, Chi_theo])

        nb_matrices += 1
        if nb_matrices == 9:
            break

    print("nb_of_matrices to plot", nb_matrices)
    if nb_matrices == 1:
        codefigure = 111
    if nb_matrices == 2:
        codefigure = 211
    if nb_matrices in (3, 4):
        codefigure = 221
    if nb_matrices in (5, 6):
        codefigure = 321
    if nb_matrices in (7, 8, 9):
        codefigure = 331

    index_fig = 0

    #    ax.set_xlim((35, 145))
    #    ax.set_ylim((-45, 45))

    dicocolor = {0: "k", 1: "r", 2: "g", 3: "b", 4: "c", 5: "m"}
    nbcolors = len(dicocolor)

    import pylab as p

    # theo spots
    for i_mat in list(range(nb_matrices)):
        p.subplot(codefigure)

        if file_to_index is not None:
            # all exp spots
            p.scatter(tth_exp, Chi_exp, s=40, c="w", marker="o", faceted=True, alpha=0.5)

        # simul spots
        p.scatter(
            all_tthchi[i_mat][0],
            all_tthchi[i_mat][1],
            c=dicocolor[(i_mat + 1) % nbcolors],
            faceted=False,
        )

        if index_fig < nb_matrices:
            index_fig += 1
            codefigure += 1
            if nb_matrices == 9:
                break

    p.show()


# --- ------------ Simulate fake data
def IofE(Energy, Amplitude=50000, powerp=-1, offset=-5):
    """
    returns simulated intensity of bragg spot as a function of Energy only
    """
    return 1.0 * Amplitude * np.power(Energy + offset, powerp * 1.0)


def Grain_to_string(grains):
    """
    convert each 4 elements grain parameters contained in grains list to list of string
    """
    liststr = ["mat, EXTINCTION, orientmat, ELEMENT"]
    for k, grain in enumerate(grains):
        mat, EXTINCTION, orientmat, ELEMENT = grain

        matstr = ISS.matrix_to_string(mat)
        orientmatstr = ISS.matrix_to_string(orientmat)

        liststr.append("Grain G % d" % k)
        for elem in matstr:
            liststr.append(elem)
        liststr.append(EXTINCTION)
        for elem in orientmatstr:
            liststr.append(elem)
        liststr.append(ELEMENT)

    return liststr


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


def createFakeData(
    key_material,
    nbgrains,
    emin=5,
    emax=25,
    outputfilename=None,
    removespots=None,
    addspots=None,
    twins=0,
):
    """
    create fake data with nbgrains grains and random detecgtor parameters

    removespots    : nb of spots to remove (randomly) in each grains
    addspots        : number of random experimental spots to add
    """
    if not outputfilename:
        outputfilename = "testSirandom"
        print("%s.cor has be written" % outputfilename)
    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    EXTINCTION = DictLT.dict_Materials[key_material][2]

    if removespots is not None:
        if len(removespots) != nbgrains:
            raise ValueError("removespots needs as many integers than nbgrains!")

    for k in list(range(nbgrains)):
        orientmat = GT.randomRotationMatrix()
        simul_matrices.append(orientmat)

    if twins >= 1:
        if twins <= nbgrains / 2:
            copysimul_matrices = copy.copy(simul_matrices)
            for twin in list(range(0, twins + 1, 2)):
                print("twin", twin)
                simul_matrices[twin + 1] = np.dot(
                    np.dot(copysimul_matrices[twin], DictLT.dict_Vect["sigma3_1"]),
                    GT.matRot([1, 2, 3], 0),
                )
        else:
            raise ValueError("nb of twins must not be higher than half of nbgrains")

    for k, orientmat in enumerate(simul_matrices):

        grains.append([np.eye(3), EXTINCTION, orientmat, key_material])

        print("orientmat", orientmat)

        # Twicetheta, Chi, Miller_ind, posx, posy, Energy
        (Twicetheta1, Chi1, Miller_ind1, posx1, posy1, Energy1) = LAUE.SimulateLaue(
            grains[k], emin, emax, detectorparameters, removeharmonics=1
        )

        if removespots is not None:
            # TODO : to simplify
            nbspots_init = len(Twicetheta1)
            print("nbspots_init", nbspots_init)
            toremoveind = np.random.randint(nbspots_init, size=removespots[k])

            # remove duplicate
            toremoveind = list(set(toremoveind))
            print("nb of removed spots", len(toremoveind))
            Twicetheta1 = np.delete(Twicetheta1, toremoveind, axis=0)
            Chi1 = np.delete(Chi1, toremoveind, axis=0)
            posx1 = np.delete(posx1, toremoveind, axis=0)
            posy1 = np.delete(posy1, toremoveind, axis=0)
            Energy1 = np.delete(Energy1, toremoveind, axis=0)

            Miller_ind1 = np.delete(Miller_ind1, toremoveind, axis=0)

        #            dataintensity1 = (20000 + k) * np.ones(len(Twicetheta1))
        dataintensity1 = IofE(Energy1)

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

        k += 1

    if twins:
        # need to delete duplicates
        Twicetheta, Chi, tokeep = GT.removeClosePoints_2(
            Twicetheta, Chi, dist_tolerance=0.01
        )
        posx = np.take(posx, tokeep)
        posy = np.take(posy, tokeep)
        dataintensity = np.take(dataintensity, tokeep)

    if addspots is not None:

        posx1 = (2048) * np.random.random_sample(addspots)
        posy1 = (2048) * np.random.random_sample(addspots)
        dataintensity1 = (65000) * np.random.random_sample(addspots)

        from LaueGeometry import calc_uflab

        Twicetheta1, Chi1 = calc_uflab(posx1, posy1, detectorparameters)

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    strgrains = ["simulated pattern with two grains"]
    strgrains += Grain_to_string(grains)

    # sorting data by decreasing intensity:
    intensitysortedind = np.argsort(dataintensity)[::-1]
    arraydata = np.array([Twicetheta, Chi, posx, posy, dataintensity]).T

    sortedarraydata = arraydata[intensitysortedind]

    Twicetheta, Chi, posx, posy, dataintensity = sortedarraydata.T

    IOLT.writefile_cor(
        outputfilename,
        Twicetheta,
        Chi,
        posx,
        posy,
        dataintensity,
        param=detectorparameters,
        comments=strgrains,
    )

    fullname = outputfilename + ".cor"
    print("Fake data have been saved in %s" % fullname)
    return fullname


# --- -----------------  Tests
def example_angularresidues_np(ang_tol):

    detectorparameters = {}
    detectorparameters["kf_direction"] = "Z>0"
    detectorparameters["detectorparameters"] = [70, 1024, 1024, 0, -1]
    detectorparameters["detectordiameter"] = 170
    detectorparameters["pixelsize"] = 0.079
    detectorparameters["dim"] = (2048, 2048)

    test_Matrix = np.eye(3)
    # random data
    expdata = np.array(
        [
            [59.2164, -66.6041],
            [54.7398, -88.3875],
            [43.0763, -83.0983],
            [58.895, -88.7721],
            [39.3337, -65.6132],
            [52.4121, -58.2399],
            [47.6515, -100.4466],
            [47.9575, -83.2749],
            [44.0163, -64.8536],
            [56.4791, -57.2717],
            [48.434, -115.549],
            [52.2976, -101.5099],
            [28.9731, -75.0536],
            [53.191, -83.4718],
            [49.0293, -64.0107],
            [41.0373, -47.0408],
            [52.3338, -117.1192],
            [37.1386, -98.1813],
            [58.7918, -83.6928],
            [54.3839, -63.0704],
            [55.9659, -41.1327],
            [39.5682, -118.0816],
            [56.3583, -118.8114],
            [41.6482, -74.3025],
            [49.9809, -43.6082],
            [53.2592, -130.741],
            [44.4802, -120.2579],
            [49.9426, -100.9654],
            [38.9334, -47.8105],
        ]
    )

    twicetheta_data, chi_data = expdata.T

    AngRes = matchingrate.Angular_residues_np(
        test_Matrix,
        twicetheta_data,
        chi_data,
        ang_tol=ang_tol,
        key_material="CdTe",
        emin=5,
        emax=40,
        ResolutionAngstrom=False,
        detectorparameters=detectorparameters,
    )

    if AngRes is not None:
        allresidues, res, nb_in_res, nb_of_theospots, meanres, maxi = AngRes

        matchringrate = 1.0 * nb_in_res / nb_of_theospots * 100.0
        print("nb of theo spots", nb_of_theospots)
        print("matching rate is %.2f" % matchringrate)


def test_IAM_real():
    """
    test refinement and multigrains indexing with class wit angles LUT indexing technique
    """
    #    readfile = 'test6.cor'
    #    readfile = 'test7.cor'
    #    readfile = 'test10.cor'
    import time, os

    readfile = os.path.join("./Examples/UO2", "dat_UO2_A163_2_0028_LT_0.cor")

    key_material = "UO2"
    emin = 5
    emax = 22

    nbGrainstoFind = 3  # 'max'#'1 #'max'

    MatchingRate_Threshold_Si = 30  # percent for simulated Si
    dictimm_Si = {
        "sigmagaussian": (0.5, 0.5, 0.5),
        "Hough_init_sigmas": (1, 0.7),
        "Hough_init_Threshold": 1,
        "rank_n": 20,
        "useintensities": 0,
    }

    MatchingRate_Threshold_W = 15  # percent W
    dictimm_W = {
        "sigmagaussian": (0.5, 0.5, 0.5),
        "Hough_init_sigmas": (1, 0.7),
        "Hough_init_Threshold": 1,
        "rank_n": 40,
        "useintensities": 0,
    }

    MatchingRate_Threshold = MatchingRate_Threshold_W
    dictimm = dictimm_W

    database = None
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

    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 60,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 10,
        "central spots indices": 10,
    }

    t0_2 = time.time()
    DataSet_2 = spotsset()

    DataSet_2.IndexSpotsSet(
        file_to_index,
        key_material,
        emin,
        emax,
        dict_params,
        database,
        IMM=False,
        MatchingRate_List=[20, 20, 20],
        nbGrainstoFind=nbGrainstoFind,
    )

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_2.plotallgrains()
    return DataSet_2  # , DataSet_2


def testIAM():
    """
    test indexing by angles Matching
    """
    readfile = None

    spot_index_central = [0, 1, 2]
    key_material = "Cu"
    emin = 5
    emax = 22

    nbmax_probed = 8000000
    nbGrainstoFind = 10

    angleTolerance_LUT = 0.1
    MatchingRate_Angle_Tol = 0.001
    Matching_Threshold_Stop = (
        50.0
    )  # minimum to stop the loop for searching potential solution

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = "toto"
        nbgrains = nbGrainstoFind
        removespots = [0] * nbgrains
        addspots = 3
        #        removespots = None
        #        addspots = None
        file_to_index = createFakeData(
            key_material,
            nbgrains,
            emin=emin,
            emax=emax,
            outputfilename=outputfilename,
            removespots=removespots,
            addspots=addspots,
        )
    else:
        file_to_index = readfile

    # read data
    data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(
        file_to_index
    )[1:]
    tth = 2 * data_theta

    exp_data = np.array([tth, Chi])

    thechi_exp = np.array([data_theta, Chi]).T

    table_angdist = GT.calculdist_from_thetachi(thechi_exp, thechi_exp)

    latticeparams = DictLT.dict_Materials[key_material][1]
    B = CP.calc_B_RR(latticeparams)

    nbmax_probed = min(nbmax_probed, len(data_theta))
    # ---------------------------------------------------------------
    import time

    t0_2 = time.time()
    DataSet_2 = spotsset()

    database = None
    dictimm = None
    DataSet_2.IndexSpotsSet(
        file_to_index,
        key_material,
        emin,
        emax,
        dictimm,
        database,
        IMM=False,
        MatchingRate_List=[20, 20, 20],
        nbGrainstoFind=nbGrainstoFind,
        verbose=0,
    )

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_2.plotallgrains()


# --- CuVia analysis
def testtwin(
    fileindex=2301,
    nbGrainstoFind=3,
    dirname="/home/micha/LaueProjects/CuVia/Carto",
    dirtowrite=None,
):
    """
    test twin
    """
    if dirtowrite is None:
        dirtowrite = dirname

    #     spot_index_central = [0, 1, 2]
    key_material = "Si"
    emin = 5
    emax = 25

    file_to_index = os.path.join(dirname, "Si_TSV.cor")

    t0_2 = time.time()
    DataSet_Si = spotsset()

    database = None

    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 60,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 10,
        "central spots indices": 10,
        "AngleTolLUT": 0.5,
        "UseIntensityWeights": True,
        "MinimumNumberMatches": 10,
    }

    DataSet_Si.IndexSpotsSet(
        file_to_index,
        key_material,
        emin,
        emax,
        dict_params,
        database,
        IMM=False,
        MatchingRate_List=[20, 20, 20],
        angletol_list=[0.5, 0.2],
        nbGrainstoFind=1,
        verbose=0,
    )

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_Si.plotallgrains()

    # ind, 2theta, chi, posx, posy, int
    dataSubstrate = DataSet_Si.getSpotsFamilyExpData(0).T

    DataSet_Cu = spotsset()
    DataSet_Cu.importdatafromfile(file_to_index)

    DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

    key_material = "Cu"
    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 40,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 20,
        "central spots indices": 3,
        "AngleTolLUT": 0.5,
        "UseIntensityWeights": True,
        "MinimumNumberMatches": 10,
    }

    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}
    dictRes = (dictMat, dictMR, dictNB, dictstrain)

    nstart = fileindex
    nend = fileindex

    for k in list(range(nstart, nend + 1)):  # 92 - 1708
        file_to_index = os.path.join(dirname, "TSVCU_%04d.cor" % k)

        print("\n\nINDEXING    file : %s\n\n" % file_to_index)

        DataSet_Cu = spotsset()
        DataSet_Cu.importdatafromfile(file_to_index)

        DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

        # init res dict
        DataSet_Cu.dict_grain_matrix = [0, 0, 0]
        DataSet_Cu.dict_grain_matching_rate = [[-1, -1], [-1, -1], [-1, -1]]

        # filling results
        dictMat[k] = [0, 0, 0]
        dictMR[k] = [-1, -1, -1]
        dictNB[k] = [-1, -1, -1]
        dictstrain[k] = [0, 0, 0]
        previousResults = None

        if k > nstart and dictMat[k - 1][0] is not 0:
            addMatrix = dictMat[k - 1][0]
            previousResults = addMatrix, dictMR[k - 1][0], dictNB[k - 1][0]

        DataSet_Cu.IndexSpotsSet(
            file_to_index,
            key_material,
            emin,
            emax,
            dict_params,
            database,
            checkSigma3=True,
            use_file=0,
            IMM=False,
            angletol_list=[0.5, 0.5],
            MatchingRate_List=[10, 10, 10],
            nbGrainstoFind=nbGrainstoFind,
            verbose=0,
            previousResults=previousResults,
        )

        for nbgrain in list(range(nbGrainstoFind)):
            dictMat[k][nbgrain] = DataSet_Cu.dict_grain_matrix[nbgrain]
            dictMR[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][1]
            dictNB[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][0]
            dictstrain[k][nbgrain] = DataSet_Cu.dict_grain_devstrain[nbgrain]

        # intermediate saving
        if (k % 100) == 0:
            #            dictRes = dictMat, dictMR, dictNB
            filepickle = open("dictCuViaaddMatrix%04d_%04d" % (nstart, nend), "w")
            pickle.dump(dictRes, filepickle)
            filepickle.close()

    filepickle = open("dictCuViaaddMatrix%04d_%04d" % (nstart, nend), "w")
    pickle.dump(dictRes, filepickle)
    filepickle.close()

    #        DataSet_Cu.plotallgrains()
    return dictRes


#    return DataSet_Si, DataSet_Cu


def SiCu(nstart, nend):
    """
    shortcut of file serie indexation of Si and Cu grains
    """
    return test_SiCu(nstart=nstart, nend=nend)


def test_SiCu(nstart=92, nend=1707):
    """
    test to index a file series with Si pattern and Cu pattern
    with starting and ending image index as arguments
    """
    import pickle

    spot_index_central = [0, 1, 2]
    key_material = "Si"
    emin = 5
    emax = 25
    nbGrainstoFind = 1

    import time, os

    file_to_index = os.path.join(
        "/home/micha/LaueProjects/CuVia/filecor", "dat_TSVCU_0092.cor"
    )

    t0_2 = time.time()
    DataSet_Si = spotsset()

    database = None

    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 60,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 10,
        "central spots indices": 10,
    }

    DataSet_Si.IndexSpotsSet(
        file_to_index,
        key_material,
        emin,
        emax,
        dict_params,
        database,
        IMM=False,
        MatchingRate_List=[20, 20, 20],
        angletol_list=[0.5, 0.2],
        nbGrainstoFind=nbGrainstoFind,
        verbose=0,
    )

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_Si.plotallgrains()

    # ind, 2theta, chi, posx, posy, int
    dataSubstrate = DataSet_Si.getSpotsFamilyExpData(0).T

    DataSet_Cu = spotsset()
    DataSet_Cu.importdatafromfile(file_to_index)

    DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

    key_material = "Cu"
    nbGrainstoFind = 3
    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 40,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 20,
        "central spots indices": 3,
    }

    dictMat = {}
    dictMR = {}
    dictNB = {}

    for k in list(range(nstart, nend + 1)):  # 92 - 1708
        file_to_index = os.path.join(
            "/home/micha/LaueProjects/CuVia/filecor", "TSVCU_%04d.cor" % k
        )

        print("\n\nINDEXING    file : %s\n\n" % file_to_index)

        DataSet_Cu = spotsset()
        DataSet_Cu.importdatafromfile(file_to_index)

        DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

        # init res dict
        DataSet_Cu.dict_grain_matrix = [0, 0, 0]
        DataSet_Cu.dict_grain_matching_rate = [[-1, -1], [-1, -1], [-1, -1]]

        DataSet_Cu.IndexSpotsSet(
            file_to_index,
            key_material,
            emin,
            emax,
            dict_params,
            database,
            use_file=0,
            IMM=False,
            angletol_list=[0.5, 0.5],
            MatchingRate_List=[10, 10, 10],
            nbGrainstoFind=nbGrainstoFind,
            verbose=0,
        )
        # filling results
        dictMat[k] = [0, 0, 0]
        dictMR[k] = [-1, -1, -1]
        dictNB[k] = [-1, -1, -1]
        for nbgrain in list(range(nbGrainstoFind)):
            dictMat[k][nbgrain] = DataSet_Cu.dict_grain_matrix[nbgrain]
            dictMR[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][1]
            dictNB[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][0]

        # intermediate saving
        dictRes = dictMat, dictMR, dictNB
        if (k % 100) == 0:
            filepickle = open("dictCuVia%04d_%04d" % (nstart, nend), "w")
            pickle.dump(dictRes, filepickle)
            filepickle.close()

    filepickle = open("dictCuVia", "w")
    pickle.dump(dictRes, filepickle)
    filepickle.close()

    #        DataSet_Cu.plotallgrains()
    return dictRes


#    return DataSet_Si, DataSet_Cu


def index_fileseries(
    fileindexrange,
    nbGrainstoFind=1,
    dirname="/home/micha/LaueProjects/CuVia/Carto",
    dirtowrite=None,
    saveObject=0,
):
    """
    This is still an example specific to CuVia (Cu and Si)
    for CuVia Jul 11
    nstart = 1708
    nend = 3323
    """
    nstart, nend = fileindexrange

    if dirtowrite is None:
        dirtowrite = dirname

    spot_index_central = [0, 1, 2]
    key_material = "Si"
    emin = 5
    emax = 25
    nbGrainstoFind_Si = 1

    import time

    file_to_index = os.path.join(dirname, "Si_TSV.cor")

    t0_2 = time.time()
    DataSet_Si = spotsset()

    database = None

    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 60,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 10,
        "central spots indices": 10,
        "AngleTolLUT": 0.5,
        "UseIntensityWeights": True,
        "MinimumNumberMatches": 10,
    }

    DataSet_Si.IndexSpotsSet(
        file_to_index,
        key_material,
        emin,
        emax,
        dict_params,
        database,
        IMM=False,
        MatchingRate_List=[20, 20, 20],
        angletol_list=[0.5, 0.2],
        nbGrainstoFind=nbGrainstoFind_Si,
        verbose=0,
    )

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    # DataSet_Si.plotallgrains() # to comment when mutliprocessing

    # ind, 2theta, chi, posx, posy, int
    dataSubstrate = DataSet_Si.getSpotsFamilyExpData(0).T

    key_material = "Cu"
    nbGrainstoFind_Cu = nbGrainstoFind
    dict_params = {
        "MATCHINGRATE_THRESHOLD_IAL": 40,
        "MATCHINGRATE_ANGLE_TOL": 0.2,
        "NBMAXPROBED": 20,
        "central spots indices": 3,
        "AngleTolLUT": 0.5,
        "UseIntensityWeights": True,
        "MinimumNumberMatches": 10,
    }

    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}
    dictspots = {}

    if saveObject:
        dict_spotssetObj = {}

    #    dictRes = {'dictMat':dictMat,
    #               'dictMR':dictMR,
    #               'dictNB':dictNB,
    #               'dictstrain':dictstrain}
    dictRes = dictMat, dictMR, dictNB, dictstrain, dictspots

    if saveObject:
        todump = dictRes, dict_spotssetObj
    else:
        todump = dictRes

    for k in list(range(nstart, nend + 1)):  # 1708-3323
        file_to_index = os.path.join(dirname, "TSVCU_%04d.cor" % k)

        print("\n\nINDEXING    file : %s\n\n" % file_to_index)

        DataSet_Cu = spotsset()
        DataSet_Cu.importdatafromfile(file_to_index)

        DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

        # TODO usefull ?
        # init res dict
        DataSet_Cu.dict_grain_matrix = [0 for kk in list(range(nbGrainstoFind_Cu))]
        DataSet_Cu.dict_grain_matching_rate = [
            [-1, -1] for kk in list(range(nbGrainstoFind_Cu))
        ]
        DataSet_Cu.dict_grain_devstrain = [0 for kk in list(range(nbGrainstoFind_Cu))]

        # preparing dicts of results
        dictMat[k] = [0 for kk in list(range(nbGrainstoFind_Cu))]
        dictMR[k] = [-1 for kk in list(range(nbGrainstoFind_Cu))]
        dictNB[k] = [-1 for kk in list(range(nbGrainstoFind_Cu))]
        dictstrain[k] = [0 for kk in list(range(nbGrainstoFind_Cu))]
        previousResults = None

        if k > nstart and dictMat[k - 1][0] is not 0:
            addMatrix = dictMat[k - 1][0]
            previousResults = addMatrix, dictMR[k - 1][0], dictNB[k - 1][0]

        DataSet_Cu.IndexSpotsSet(
            file_to_index,
            key_material,
            emin,
            emax,
            dict_params,
            database,
            use_file=0,
            IMM=False,
            angletol_list=[0.5, 0.5],
            MatchingRate_List=[10, 10, 10],
            nbGrainstoFind=nbGrainstoFind_Cu,
            verbose=0,
            previousResults=previousResults,
        )

        dictspots[k] = DataSet_Cu.getSummaryallData()

        if saveObject:
            dict_spotssetObj[k] = DataSet_Cu

        # filling dicts with indexation results
        for nbgrain in list(range(nbGrainstoFind_Cu)):
            dictMat[k][nbgrain] = DataSet_Cu.dict_grain_matrix[nbgrain]
            dictMR[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][1]
            dictNB[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][0]
            dictstrain[k][nbgrain] = DataSet_Cu.dict_grain_devstrain[nbgrain]

        # intermediate saving
        if (k % 10) == 0:
            #            dictRes = dictMat, dictMR, dictNB
            # filepickle = open(os.path.join(dirtowrite, 'dictCu_3g_%04d_%04d' % (nstart, nend)), 'w')
            # pickle.dump(todump, filepickle)
            # filepickle.close()

            with open(
                os.path.join(dirtowrite, "dictCu_3g_%04d_%04d" % (nstart, nend)), "w"
            ) as f:
                pickle.dump(todump, f)

    # filepickle = open(os.path.join(dirtowrite, 'dictCu_3g_%04d_%04d' % (nstart, nend)), 'w')
    # pickle.dump(todump, filepickle)
    # filepickle.close()

    with open(
        os.path.join(dirtowrite, "dictCu_3g_%04d_%04d" % (nstart, nend)), "w"
    ) as f:
        pickle.dump(todump, f)

    #        DataSet_Cu.plotallgrains()

    return todump


#    return DataSet_Si, DataSet_Cu

# ------------  Main            ------------------------------
if __name__ == "__main__":

    res = test_GetProximity2()

    # test_indexation_with_apriori_Matrices()
    #     Test_Common_index_Method()
    # test_indexation_generalRefunitCell_1()
    # test_indexation_generalRefunitCell_2()

    # Test_Common_index_Method()

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
        colorlist = [ "k", (1.0, 0.0, 0.5),
                    "b", "g", "y", "r", "c", "m", "0.75",
                    (0.2, 0.2, 0.2), (0.0, 0.5, 1.0), ]
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

    # from indexing
    if 0:
        initialparameters = {}

        LaueToolsProjectFolder = os.path.abspath(os.curdir)

        print("LaueToolProjectFolder", LaueToolsProjectFolder)

        MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "GeGaN")

        print("MainFolder", MainFolder)

        initialparameters["PeakList Folder"] = os.path.join(MainFolder, "datfiles")
        initialparameters["IndexRefine PeakList Folder"] = os.path.join(
            MainFolder, "fitfiles"
        )
        initialparameters["PeakListCor Folder"] = os.path.join(MainFolder, "corfiles")
        initialparameters["PeakList Filename Prefix"] = "nanox2_400_"
        initialparameters["IndexRefine Parameters File"] = os.path.join(
            MainFolder, "GeGaN.irp"
        )
        initialparameters["Detector Calibration File .det"] = os.path.join(
            MainFolder, "calibGe_nanowMARCCD165.det"
        )
        initialparameters["Detector Calibration File (.dat)"] = os.path.join(
            MainFolder, "nanox2_400_0000_LT_1.dat"
        )
        initialparameters["PeakList Filename Suffix"] = ".dat"

        print(initialparameters["IndexRefine Parameters File"])

        dict_param_list = readIndexRefineConfigFile(
            initialparameters["IndexRefine Parameters File"]
        )

        CCDparams, calibmatrix = IOLT.readfile_det(
            initialparameters["Detector Calibration File .det"], nbCCDparameters=8
        )

        Index_Refine_Parameters_dict = {}
        Index_Refine_Parameters_dict["CCDCalibParameters"] = CCDparams[:5]
        Index_Refine_Parameters_dict["pixelsize"] = CCDparams[5]
        Index_Refine_Parameters_dict["framedim"] = CCDparams[6:8]

        Index_Refine_Parameters_dict["PeakList Folder"] = initialparameters[
            "PeakList Folder"
        ]
        Index_Refine_Parameters_dict["PeakListCor Folder"] = initialparameters[
            "PeakListCor Folder"
        ]
        Index_Refine_Parameters_dict["nbdigits"] = 4
        Index_Refine_Parameters_dict["prefixfilename"] = initialparameters[
            "PeakList Filename Prefix"
        ]
        Index_Refine_Parameters_dict["suffixfilename"] = ".dat"
        Index_Refine_Parameters_dict["prefixdictResname"] = (
            initialparameters["PeakList Filename Prefix"] + "_dict_"
        )

        Index_Refine_Parameters_dict["PeakListFit Folder"] = (
            initialparameters["IndexRefine PeakList Folder"] + "_test"
        )
        Index_Refine_Parameters_dict["Results Folder"] = (
            initialparameters["IndexRefine PeakList Folder"] + "_test"
        )

        Index_Refine_Parameters_dict["dict params list"] = dict_param_list
        Index_Refine_Parameters_dict["Spots Order Reference File"] = None

        print(Index_Refine_Parameters_dict)

        index_fileseries_3(
            (0, 1, 1),
            Index_Refine_Parameters_dict=Index_Refine_Parameters_dict,
            saveObject=0,
            verbose=0,
            nb_materials=None,
            build_hdf5=False,
            prefixfortitle="",
            use_previous_results=False,
            CCDCalibdict=None,
        )

    if 1:
        # E1_13_run2_0171.mccd
        Index_Refine_Parameters_dict = {}

        dict_params = [
            {
                "MATCHINGRATE THRESHOLD IAL": 100.0,
                "MATCHINGRATE ANGLE TOL": 0.5,
                "NBMAXPROBED": 30,
                "central spots indices": 5,
                "key material": "ZrO2Y2O3",
                "emin": 5,
                "emax": 23,
                "nbGrainstoFind": 2,
            }
        ]

        CCDcalib = [
            68.61556108571234347,
            940.77744180377760586,
            1005.14714125015211721,
            0.72531401975651599,
            -0.76181063603636234,
        ]

        CCDCalibdict = {}
        CCDCalibdict["CCDCalibParameters"] = CCDcalib
        CCDCalibdict["framedim"] = (2048, 2048)
        CCDCalibdict["detectordiameter"] = 165.0

        Index_Refine_Parameters_dict["CCDCalibParameters"] = CCDcalib
        # , 0.08056640625000000, 2048.00000000000000000, 2048.00000000000000000]
        Index_Refine_Parameters_dict["pixelsize"] = 0.080566
        # 0.08057, 2048, 2048

        Index_Refine_Parameters_dict["dict params list"] = dict_params

        Index_Refine_Parameters_dict[
            "PeakList Folder"
        ] = "/home/micha/LaueTools/Examples/ZrO2"
        Index_Refine_Parameters_dict[
            "PeakListCor Folder"
        ] = "/home/micha/LaueTools/Examples/ZrO2"
        Index_Refine_Parameters_dict["nbdigits"] = 4
        Index_Refine_Parameters_dict["prefixfilename"] = "E1_13_run2_"
        Index_Refine_Parameters_dict["suffixfilename"] = ".dat"
        Index_Refine_Parameters_dict["prefixdictResname"] = "E1_13_run2_dict_"

        Index_Refine_Parameters_dict[
            "PeakListFit Folder"
        ] = "/home/micha/LaueTools/Examples/ZrO2"
        Index_Refine_Parameters_dict[
            "Results Folder"
        ] = "/home/micha/LaueTools/Examples/ZrO2"

        fileindexrange = (171, 177, 3)

        index_fileseries_3(
            fileindexrange,
            Index_Refine_Parameters_dict=Index_Refine_Parameters_dict,
            saveObject=0,
            CCDCalibdict=CCDCalibdict,
        )
