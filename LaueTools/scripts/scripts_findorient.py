import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import findorient as FO
import generaltools as GT
import CrystalParameters as CP
import LaueGeometry as LTGeo
from dict_LaueTools import dict_Materials


def Test_Build_and_Query_LUT(
    angle=69, tol=1, n=3, latticeparams=(2, 1, 4, 75, 90, 120)):
    """
    Test_Build_and_Query_LUT
    """
    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n, remove_negative_l=CP.isCubic(latticeparams))

    if 1:  # filterharmonics:
        #        hkl_all = CP.FilterHarmonics_2(hkl_all)
        hkl_all = FO.FilterHarmonics(hkl_all)

    Gstar_metric = CP.Gstar_from_directlatticeparams(*latticeparams)
    # GenerateLookUpTable
    LUT = FO.GenerateLookUpTable(hkl_all, Gstar_metric)

    sol69 = FO.QueryLUT(LUT, angle, tol)

    print("solutions", sol69)

    return sol69


def test_FilterHarmonics():
    """
    test harmonics removal
    """
    a = np.arange(3 * 8).reshape((8, 3))
    a[-1] = np.array([-6, -8, -10])
    a[-3] = np.array([9, 12, 15])
    fila = FO.FilterHarmonics(a)
    print("a", a)
    print("filtered a")
    print(fila)

    print("\n other test")
    testhkl = np.array(
        [
            [-2, -4, 2],
            [0, -2, 2],
            [-3, -11, 3],
            [-1, -11, 3],
            [-5, -9, 3],
            [-3, -9, 3],
            [-1, -9, 3],
            [1, -9, 3],
            [-5, -7, 3],
            [-3, -7, 3],
            [-1, -7, 3],
            [1, -7, 3],
            [-5, -5, 3],
            [-3, -5, 3],
            [-1, -5, 3],
            [1, -5, 3],
            [-3, -3, 3],
            [-1, -3, 3],
            [1, -3, 3],
            [-2, -10, 4],
            [-4, -8, 4],
            [0, -8, 4],
            [-6, -6, 4],
            [-2, -6, 4],
            [2, -6, 4],
            [0, -4, 4],
            [-2, -2, 4],
            [-5, -7, 5],
        ]
    )
    print("testhkl")
    print(testhkl)
    print("one of  [  0,  -2,   2] and [  0,  -4,   4] must be removed")

    print(FO.FilterHarmonics(testhkl))


def test():
    """
    test
    """
    # GT.find_closest(input_array, target_array, tol)

    # datafrompickle=open('recognised.rec','r')
    # reconnu = pickle.load(datafrompickle)
    # datafrompickle.close()

    recognised = [
        [72.263978989400002, 19.5486315279, 111, 12],
        [109.63069847600001, -24.477560408599999, 110, 16],
        [95.941622918999997, 10.5435782548, 210, 21],
        [82.3329445217, 4.2357560583199998, 311, 31],
        [85.062674650700004, -3.4110763979200001, 210, 32],
        [97.118855973999999, -0.87674472922199997, 211, 64],
        [106.209401235, -15.1071780961, 311, 93],
        [49.406572447000002, -12.219836147100001, 311, 133],
        [93.0858676231, 3.5826456916599998, 221, 147],
    ]

    recog = recognised  # reconnu

    print("-----------------------------------------")
    print("First two recognised spots Matrix proposition")
    indexpoint1 = 0  # default = 0
    indexpoint2 = 1  # default = 1
    ang = 0.1
    print(
        "For spots #%d and #%d in data"
        % (recog[indexpoint1][-1], recog[indexpoint2][-1])
    )
    print("Angular tolerance: %.2f deg" % ang)
    FO.proposematrix(indexpoint1, indexpoint2, recog, ang)
    print("------------------------\n")

    print("Now looking at all spots pairs distances and compare with lookup table")
    toleranceang = 0.05
    print("-----------------------------------------")
    print("Angular tolerance for multiple search: %.2f" % toleranceang)
    result = FO.Allproposedmatrix(recog, toleranceang)
    # print result

    if len(result) > 0:
        print("Cool, I can propose %s matrix" % len(result))
        for elem in result:
            spotindices = elem[0]
            print(
                "With spots #%d and #%d in data"
                % (recog[spotindices[0]][-1], recog[spotindices[1]][-1])
            )
            print(elem[1])

        print("Now removing similar matrix if needed")
        singlematrixlist = [result[0][1]]
    else:
        print("Argh no matrix found!")

    for j in list(range(1, len(result))):
        listcorrel = [
            np.corrcoef(np.ravel(result[j][1]), np.ravel(singlematrixlist[k]))[0][1]
            for k in range(len(singlematrixlist))
        ]

        booltable = np.array(listcorrel) < 0.995  # True if matrices are uncorrelated

        if all(booltable):  # if matrix proposed is different from the others

            singlematrixlist.append(result[j][1])

    print("Similar matrices removed")
    print(singlematrixlist)

    print("TODO: looking at same matrix by frame permutation")


def example_Neighbourslist():
    """ 
    give list of planes that are at 25 deg +- tol from a the 210 plane
    """
    print("Planes that are distant from [2,1,0] plane from 20 to 30 degree")
    tol = 5.0
    table_210 = FO.prodtab_210()
    pos_close = GT.find_closest(array([25.0]), table_210, tol)
    plane_sol = np.take(FO.LISTNEIGHBORS_210, pos_close[1], axis=0)
    distance_sol = np.take(table_210, pos_close[1])
    for plane, dist in zip(plane_sol, distance_sol):
        print("plane %s at %.2f degree from plane [2,1,0]" % (plane, dist))


def test_PlanePairs_from2sets():
    latticeparams = dict_Materials["Cu"][1]
    n = 3
    # compute LUT outside loop:
    hkl_all = GT.threeindices_up_to(n, remove_negative_l=CP.isCubic(latticeparams))

    if 1:  # filterharmonics:
        #        hkl_all = CP.FilterHarmonics_2(hkl_all)
        hkl_all = FO.FilterHarmonics(hkl_all)

    Gstar = CP.Gstar_from_directlatticeparams(*latticeparams)

    hkl1 = [[1, 0, 0], [2, 2, 1]]
    hkl2 = hkl_all

    FO.PlanePairs_from2sets(
        25.35, 5, hkl1, hkl2, Gstar, onlyclosest=1, filterharmonics=1, verbose=1
    )

    FO.PlanePairs_from2sets(
        25.35, 5, hkl1, hkl2, Gstar, onlyclosest=0, filterharmonics=1, verbose=1
    )


#     Test_Build_and_Query_LUT(angle=69, tol=1, n=4, latticeparams=(2, 1, 4, 75, 90, 120))


# test() # old test with data from recognition.py


#     PlanePairs_from2sets(25.1, .5,
#                             hkl1, hkl2, Gstar,
#                             onlyclosest=1, filterharmonics=1, verbose=1)

if 1:  # some tests to play with angles LUT

    # these parameters cannot be randomly chosen (they are related each other)
    # But by setting one angle to 90. you may play a lot with the other lattice parameters
    a, b, c, alpha, beta, gamma = 5.0, 2.3, 4.86, 89.56, 68.25, 90.0
    # a,b,c,alpha,beta,gamma = 4.99,4.99,17.061,90,90,120 # trigonal, 'hexagonal' lattice

    # compute metric tensor
    Gstar = CP.Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma)

    print("Gstar", Gstar)

    # HKL_list = np.array([[0,0,1],[1,0,4],[-1,1,4],[1,2,0]])
    # for hkl1 in HKL_list[:-1]:
    # for hkl2 in HKL_list[1:]:
    # angle = CP.AngleBetweenTwoNormals(hkl1, hkl2 ,Gstar)
    # print "<%s , %s> = %.5f"%(str(hkl1),str(hkl2),angle)

    # some hkl lists
    hkl1 = GT.Positiveindices_up_to(3)
    hkl2 = GT.Positiveindices_up_to(2)

    print("hkl2", hkl2)
    # build LUT
    LUT = FO.GenerateLookUpTable_from2sets(hkl1, hkl2, Gstar)

    print("LUT", LUT)

    # test_hkl = np.array([[5,-5,3],[203,77,91],[-4,2,1],[6,3,3],[8,4,2],[1,1,1],[2,2,2],[0,2,8]])
    # GCDs = GT.GCD(np.abs(test_hkl))

    # res = GT.reduceHKL(test_hkl)

    pp, LUT = FO.PlanePairs_from2sets(
        12.58, 1.0, hkl1, hkl2, "Cu", onlyclosest=1, filterharmonics=1, verbose=1
    )

if 0:  # test of orient matrix from distorted unit cell

    from numpy import *

    a, b, c, alpha, beta, gamma = 3.2, 2.5, 5.0, 86, 90.0, 120.0
    latticeparams = a, b, c, alpha, beta, gamma
    B = CP.calc_B_RR([a, b, c, alpha, beta, gamma])
    # metric tensor ---------
    Gstar_metric = CP.Gstar_from_directlatticeparams(a, b, c, alpha, beta, gamma)

    # hkl = np.array([[-2,0,2],[-6,0,2], [-3,1,1],[-2,4,2]])
    # matorient = eye(3)

    hkl = np.array([[3, 1, 7], [1, 1, 3], [2, 2, 2], [5, 1, 1]])
    matorient = array(
        [
            [-0.6848285, 0.2544487, -0.6828365],
            [-0.7064087, -0.0017916, 0.7078019],
            [0.1788759, 0.9670846, 0.1809717],
        ]
    )  # surf 111

    # Digest of Laue6.py --------------------
    # G* taken into account distorsion
    Gstarprime = dot(B, hkl.T).T

    print("matrorient for simulation", matorient)

    q = dot(matorient, Gstarprime.T).T

    qn = sqrt(sum(q ** 2, axis=1))

    uq = q / reshape(qn, (len(qn), 1))

    # ui in Lauetools frame
    ui = array([1, 0, 0.0])
    # uf = ui -2(ui.uq)uq

    coef = -2.0 * dot(ui, uq.T)
    uf = ui + uq * reshape(coef, (len(qn), 1))

    # kf = ( - sin 2theta sin chi, cos 2theta  , sin 2theta cos chi) XMAS
    # kf = ( cos 2theta, sin 2theta sin chi,   , sin 2theta cos chi) LT

    EPS = 0.00000001
    PI = np.pi
    # in LT
    chi = (180.0 / PI) * arctan(uf[:, 1] / (uf[:, 2] + EPS))
    twtheta = (180.0 / PI) * arccos(uf[:, 0])

    coord = array([twtheta, chi]).T
    # ---------------------------------------------------------------

    # coord = [[90,0],[143.1301,0],[139.1214,    26.5651],[70.5288,    45.0000]]

    hkl1 = hkl[0]
    coord1 = coord[0] + array([0.1, -1])  # add some noise

    hkl2 = hkl[2]
    coord2 = coord[2] + array([-0.05, 0.7])  # add some noise

    mat = FO.OrientMatrix_from_2hkl(
        hkl1, coord1, hkl2, coord2, B, verbose="yes", frame="lauetools"
    )

    # --------------------------------------------------
    # check from all kinds of situation

    # res_mat = []

    # for i in range(len(hkl)-1):
    # for j in range(i+1,len(hkl)):

    # hkl1 = hkl[i]
    # hkl2 = hkl[j]

    # coord1 = coord[i] +array([0.1,-1]) # add some noise
    # coord2 = coord[j] +array([-.05,.7])

    # mat = OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B, verbose='yes', frame = 'lauetools')

    # res_mat.append(mat)
    # print "\n\nmatorient",mat
    # print "*****************\n"

    # --------------------------------------------------

    if 0:
        # build linear matrix equation to solve with pinv
        # for m = 1,2,3
        # we look for operator that solve
        # qm = (U) Gm*
        # size(qm) = size(Gm*) = 3
        # shape (U) = (3,3)
        # let U = A whose element are a ij
        # the three equations (qix,qiy,qiz) = ((a11,a12,a13),(a21,a22,a23),(a31,a32,a33))(Gi*x,Gi*y,Gi*z)
        # can be rearranged like M P = Q
        # where P is 9 element of A: [a11,a12,a13,a21,a22,a23,a31,a32,a33]
        #         Q is a stack of the sthree components of each qm
        #       Q = [q1x,q1y,q1z,q2x,q2y,q2z,q3x,q3y,q3z]
        # and M is a quite sparse matrix containing row of Gm* elements
        #     M is a 3 times vertically stacked pattern:
        #  patternM =[[G1x*,G1y*,G1z*,0,0,0,0,0,0],[0,0,0,G2x*,G2y*,G2z*,0,0,0],[0,0,0,0,0,0,G3x*,G3y*,G3z*]]

        # in case of distorted unit cell: Gm* are unit vector from B Gm*
        # qm are also unit vectors

        G1 = np.dot(B, np.array(hkl1))
        G2 = np.dot(B, np.array(hkl2))
        G3 = np.cross(G1, G2)

        normG1 = np.sqrt(np.dot(G1, G1))
        normG2 = np.sqrt(np.dot(G2, G2))
        normG3 = np.sqrt(np.dot(G3, G3))

        print("norms of G1,G2,G3", normG1, normG2, normG3)

        matrice_P = np.array([G1 / normG1, G2 / normG2, G3 / normG3])

        # building M
        patternM = np.zeros((3, 9))
        patternM[0, :3] = matrice_P[0]
        patternM[1, 3:6] = matrice_P[1]
        patternM[2, 6:] = matrice_P[2]

        M = np.vstack((patternM, patternM, patternM))

        # similar matrix but with 2theta,chi coordinate
        twicetheta_1 = coord1[0]
        chi_1 = coord1[1]
        twicetheta_2 = coord2[0]
        chi_2 = coord2[1]

        # expression of unit q in chosen frame
        frame = "lauetools"
        qq1 = LTGeo.unit_q(twicetheta_1, chi_1, frame=frame)
        qq2 = LTGeo.unit_q(twicetheta_2, chi_2, frame=frame)

        qq3prod = np.cross(
            qq1, qq2
        )  # can be negative, we want to have finally a matrix with one eigenvalue of +1
        qq3n = np.sqrt(np.dot(qq3prod, qq3prod)) * 1.0

        Q = np.ravel(np.array([qq1, qq2, qq3prod / qq3n]))

        # solving for P : MP= Q -----------------------
        # 1-  use SVD:
        U, s, V = np.linalg.svd(M)
        # then P   =  V diag(1/s) U.T Q
        # with diag(1/s) is a diagonal matrix made of inverse of singular values
        # if one singular values is very close to 0 then the inverse of this value must replace by zero!!
        print("singular values", s)

        threshold = 0.001
        invs = np.where(s < threshold, 0, 1.0 / s)

        print("inverse singular value", invs)
        invS = np.diag(invs)

        invM = np.dot(V, np.dot(invS, U.T))

        P = np.dot(invM, Q)
