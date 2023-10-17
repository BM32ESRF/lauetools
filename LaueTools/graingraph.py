"""
Module to deduce from distances recognition in mutual pair distance matrix
the adjaceny matrix from which graph theory's algorithms can produce
set of spots highly connected to each other, hence spos likely
to belong to the same grain

Tabledistance comes from most important planes (and intense reflection) in cubic structure up to family plane (321)

TODO: take into account the information related to the connectivity between two spots
"""

import sys
import os
# import scipy.io.array_import # pour charger les donnees
import pickle

import numpy as np

import pylab as P

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

try:
    import networkx as NX
except ImportError:
    print("\n***********************************************************")
    print("networkx module is missing! Please install it !")
    print("***********************************************************\n")

if sys.version_info.major == 3:
    from . import findorient as FO
    # from . import indexingAnglesLUT as INDEX
    from . import generaltools as GT
    from . import IOLaueTools as IOLT
else:
    import findorient as FO
    # import indexingAnglesLUT as INDEX
    import generaltools as GT
    import IOLaueTools as IOLT

__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"


def read_LUT(picklefilename):

    return pickle.load(open(picklefilename, 'rb'))


def create_tabdist(pickle_it=0, picklefilename="CubicDistanceList.dat"):
    """
    Creates all distances below 90 degrees between two planes in CUBIC non distorted structure
    from main planes (1,0,0),(1,1,0) up to (3,2,1)  type

    options pickle_it=1 to save table in picklefilename
    """

    whole_interdistance_list = []
    for func in FO.LUT_MAIN_CUBIC:
        whole_interdistance_list = whole_interdistance_list + list(func)

    # print np.sort(array(whole_interdistance_list))

    ar_distances = np.sort(np.array(np.array(list(
                    set(list(np.array(np.sort(np.array(whole_interdistance_list)).round(
                                    decimals=3), dtype="|S6"))))), dtype=np.float32))
    print("Table of distances")
    print(ar_distances)

    if pickle_it:
        distfile = open(picklefilename, "w")
        pickle.dump(ar_distances, distfile, protocol=2)
        # binary mode , data were created with cPickle.dump(obj,filenprotocol=2)
        # frou=open(Globalname,'rb')
        # tabdata=cPickle.load(frou)
        distfile.close()
        print("Table pickled in %s" % picklefilename)
    return ar_distances


def read_disttable(picklefilename=None):
    """
    Reads all sorted distances in cubic structure below 90 degrees
    from main planes (1,0,0),(1,1,0) up to (3,2,1)  type
    """
    if picklefilename is not None:
        ar_distances,_,_,_ = read_LUT(picklefilename)
        print("Cubic distance table read from a pickled file %s"%picklefilename)
    else:
        ar_distances = np.array([8.13000011, 10.02499962, 10.89299965, 11.48999977, 14.76299953, 15.79300022, 17.02400017,
        17.54800034, 17.71500015, 18.43499947, 19.10700035, 19.2859993, 19.47100067, 21.61700058, 21.78700066,
        22.20800018, 24.09499931, 25.23900032, 25.35199928, 25.84199905, 26.56500053, 27.0170002,
        27.26600075, 29.20599937, 29.49600029, 30.0, 31.00300026, 31.48200035, 31.94799995, 32.31200027,
        32.51300049, 33.21099854, 33.55699921, 35.09700012, 35.26399994, 36.31000137, 36.69900131, 36.86999893,
        38.94200134, 39.23199844, 40.20299911, 40.29100037, 40.47900009, 40.89300156, 41.81000137,
        42.39199829, 42.45000076, 43.0890007, 44.41500092, 45.0, 45.28900146, 47.12400055, 47.45899963,
        47.60800171, 47.86999893, 48.18999863, 49.10699844, 49.79700089, 49.86000061, 49.99499893, 50.47900009,
        50.76800156, 51.88700104, 53.13000107, 53.30099869, 53.39599991, 53.72900009, 54.73600006, 55.10499954,
        55.4620018, 56.78900146, 56.93799973, 57.68799973, 58.19400024, 58.51800156, 58.9090004, 59.52999878,
        59.83300018, 60.0, 60.50400162, 61.08599854, 61.43899918, 61.87400055, 62.9640007, 63.43500137,
        63.54899979, 63.61199951, 64.6230011, 64.76100159, 64.89600372, 65.00299835, 65.06199646, 65.90499878,
        66.13899994, 66.42199707, 67.58000183, 67.79199982, 68.58300018, 68.98799896, 69.07499695, 70.52899933,
        70.89299774, 71.19599915, 71.56500244, 72.02500153, 72.45200348, 72.54199982, 72.65399933, 73.22100067,
        73.39800262, 73.56999969, 74.20700073, 74.49900055, 75.03700256, 75.31300354, 75.7480011, 76.36699677,
        77.07900238, 77.39600372, 78.46299744, 78.90399933, 79.00700378, 79.10700226, 79.48000336, 79.73699951,
        79.97499847, 80.40599823, 80.72599792, 81.78700256, 81.87000275, 82.17900085, 82.25099945, 82.58200073,
        83.13500214, 83.6210022, 83.9489975,
        84.23200226, 84.26100159, 84.78399658, 84.88899994, 85.15200043, 85.90399933, 90.0], dtype=float)
        print("Cubic distances Table read")

        # shrink LUT
        # ar_distances =  ar_distances[ar_distances>=55.]
    return ar_distances


def TableDistance_exp(filename, nb_of_spots, col_Int=4):
    """
    From experiments file and number of first selected spots
    Returns matrix of mutual angular distances (deg)

    nb_of_spots=-1 means considering all spots
    Uses intensity sorting from data in column index col_Int to select the first spots

    """
    #(alldata, data_theta, data_chi, data_pixX, data_pixY, data_I, detParam)
    data = IOLT.readfile_cor(filename)
    data_theta, data_chi, data_I = data[1], data[2], data[5]
    nbp = len(data_theta)

    if nb_of_spots > 0:
        upto = min(nb_of_spots, nbp)
        sorted_int_index = np.argsort(data_I)[::-1][:upto]
        print("Considering only %d most intense spots" % upto)
    elif nb_of_spots == -1:
        print("Considering all spots")
        sorted_int_index = np.argsort(data_I)[::-1]
        upto = nbp

    # listofselectedpts = np.arange(len(sorted_int_index))
    Theta = data_theta[sorted_int_index]
    Chi = data_chi[sorted_int_index]
    Intens = data_I[sorted_int_index]
    sorted_data = np.transpose(np.array([Theta, Chi, Intens]))

    # array of interangular distance of all points
    print("Calculating all angular distances ...")
    angulardisttable = GT.calculdist_from_thetachi(sorted_data[:, 0:2], sorted_data[:, 0:2])
    print("... Done !")
    # ind_sort=argsort(Tabledistance[0,1:])
    return angulardisttable, upto


def create_AdjencyMatrix(Tabledistance, ReferenceTable, ang_tol, nb_of_spots, verbose=0):
    """
    Creates Adjency Matrix from:
        Tabledistance: matrix of mutual inter angular distance
        ReferenceTable: table of possible distances
        ang_tol: tolerance for assigning 1 in adjencymatrix if exp. distance "is" in ReferenceTable

    nb_of_spots=len(Tabledistance)
    """
    mytab = np.zeros(nb_of_spots * nb_of_spots)
    print("creating adjency matrix ...")

    # put 1 where a distance was found in ReferenceTable for each row = elem

    # for elem in arange(nb_of_spots):
    # print elem
    # print GT.find_closest(ReferenceTable ,Tabledistance[elem],ang_tol)[1]
    # print GT.find_closest(ReferenceTable ,Tabledistance[elem],ang_tol)[1]+elem*nb_of_spots

    # enthought python 2.5 (reversed order of args!!)
    adjtab1D = [mytab.put(
            GT.find_closest(ReferenceTable, Tabledistance[elem], ang_tol)[1] + elem * nb_of_spots, 1)
            for elem in np.arange(nb_of_spots)]

    # for debugging
    # mytab=np.zeros(nb_of_spots*nb_of_spots)
    # print "ReferenceTable",ReferenceTable
    # for elem in np.arange(nb_of_spots):
    # print "elem",elem
    # print "angkl",Tabledistance[elem]
    # posy=GT.find_closest(Tabledistance[elem],ReferenceTable ,ang_tol)[1]
    # print "posy raw"
    # posy=posy+elem*nb_of_spots
    # what=1
    # print  "posy",posy
    # mytab.put(posy,what)

    adjencymat = np.reshape(mytab, (nb_of_spots, nb_of_spots))
    if verbose:
        print("adjtab1D", adjtab1D)
    print("... Done !")

    """
    # to understand the creation of john!
    resclose=GT.find_closest(ar_distances,Tabledistance[0],0.1)
    #print resclose[0]
    print resclose[1]
    #print resclose[2]
    #print "ar_distances input",ar_distances
    #print "Tabledistance[0] target",Tabledistance[0]
    # resclose[1] indice dans sort(Tabledistance[0,1:]) qui ont une distance connue

    #locations ----------
    indices_in_ref=resclose[0][resclose[1]]
    indices_in_data=resclose[1]
    print "these two list must be equal within the angular tolerance"
    print take(ar_distances,resclose[0][resclose[1]])
    print take(Tabledistance[0],resclose[1])
    # ------------------

    # ------------------
    #print "bigmapping"
    #bozo=map(lambda elem: GT.find_closest(ar_distances,sort(Tabledistance[elem]),0.1)[1],range(200))
    # ------------------
    """
    return adjencymat

def flatnestedlist(list_of_lists):
    return [y for x in list_of_lists for y in x]

def bestclique_oneNode(CliquesList, displaybest=0):
    """
    Returns the largest (in size) clique containing the spot cliques list the largest one

    """
    # list of clique length
    Clength_list = np.array(list(map(len, CliquesList)))
    # print('Clength_list',Clength_list)

    # index to sort Cliques by length
    sortedcliques_ind = np.argsort(Clength_list)[::-1]

    # print searchsorted(lon_C0[sortedcliques_ind],4)
    print(CliquesList[np.argmax(Clength_list)])
    bestclique = np.sort(CliquesList[np.argmax(Clength_list)])

    val_longest = np.amax(Clength_list)
    longest_inds = np.where(Clength_list == val_longest)[0]
    longest_Cliques = [CliquesList[ind] for ind in longest_inds]
    spots_in_longest = set(flatnestedlist(longest_Cliques))
    print("longest_Cliques", longest_Cliques)
    print("spots_in_longest", spots_in_longest)

    # val_threshold = 8
    # higher_inds = np.where(Clength_list>=val_threshold)[0]
    # higher_Cliques = [CliquesList[ind] for ind in higher_inds]
    # spots_in_higher = set(flatnestedlist(higher_Cliques))
    # print("higher_Cliques",higher_Cliques)
    # print("spots_in_higher",spots_in_higher)

    statlongest = np.bincount(np.ravel(longest_Cliques))
    highfreqspot = np.argsort(statlongest)[::-1]
    print("Most frequent spots ", statlongest[highfreqspot][:10])
    print(" with frequency ", highfreqspot[:10])

    print("bestclique", bestclique)
    if displaybest:
        nbbest_r = displaybest  # to select the number of best cliques found
        for index in sortedcliques_ind[:nbbest_r]:
            print("clique size", Clength_list[index], "   nodes: ", np.sort(CliquesList[index]))
    return bestclique


def give_bestclique(filename, nb_of_spots, ang_tol, nodes=0, col_Int=-1,
                                        LUTfilename=None, verbose=0):
    """ from peakslist file, it gives the sets of spots belonging to cliques according to a structure (given bu the LUT)

    if LUTfilename is None (default) then cubic structure is given

    .. note:: spot indices are those of file (whatever some spots have been already indexed)
    """
    # reading reference table of distances for recognition
    ar_distances = read_disttable(picklefilename=LUTfilename)
    # print "ar_distances",ardistances

    # reading experimental data i.e. list of spots from .cor file (2the,chi,x,y,I)
    if verbose:
        print("data file: %s" % filename)
    Tabledistance, nb_of_spots = TableDistance_exp(filename, nb_of_spots, col_Int=col_Int)

    # print "Tabledistance in give_bestclique [:10]",Tabledistance[:10]
    # print "nb_of_spots",nb_of_spots

    # Adjency matrix creation and corresponding graph creation:
    adjencymat = create_AdjencyMatrix(Tabledistance, ar_distances, ang_tol, nb_of_spots)
    # GGraw = NX.from_whatever(adjencymat, create_using=NX.Graph()) # old syntax
    GGraw = NX.to_networkx_graph(adjencymat, create_using=NX.Graph())

    print('adjencymat',adjencymat)
    # print shape(adjencymat)
    # print adjencymat[0]

    print("nodes selected", nodes)
    print("Searching cliques ...")
    C0 = NX.cliques_containing_node(GGraw, nodes=nodes, cliques=None)
    print('\nC0', C0)
    print("\n... Done !\n")
    if not isinstance(nodes, int):
        best_list = []
        print("nb of nodes", len(C0))
        for k in range(len(C0)):
            if verbose:
                print("finding best cliques for # ", k, " entered nodes")
            bc = bestclique_oneNode(C0[k], displaybest=verbose)
            if verbose:
                print("----------------")
            best_list.append(np.sort(bc))
    else:
        # single node results
        best_list = np.sort(bestclique_oneNode(C0, displaybest=verbose))
    return best_list


def test(plot=0):
    """
    Test with a simulated noisy adjency matrix containing two grains
    """
    index_noise = 5
    index_connectivity = 0

    # noise=poisson(lam=2,size=(20,20))/4
    noise = np.clip(np.random.random_integers(-index_noise, 1, (20, 20)), 0, 1)

    nbflip = 100.0 * np.bincount(np.ravel(noise))[1] / np.bincount(np.ravel(noise))[0]
    print("rate of noise error", nbflip)
    a1 = np.reshape(np.clip(np.random.random_integers(-index_connectivity, 1, size=100), 0, 1),
        (10, 10))
    a2 = np.reshape(np.clip(np.random.random_integers(-index_connectivity, 1, size=100), 0, 1),
        (10, 10))
    a0 = np.zeros((10, 10))
    bigmat = np.array(np.vstack((np.hstack((a1, a0)), np.hstack((a0, a2)))), dtype=np.int8)
    print("bigmat", bigmat)
    print("noise", noise)

    # GGraw = NX.from_whatever(bigmat, create_using=NX.Graph()) #old syntax
    # GGraw = NX.to_networkx_graph(bigmat, create_using=NX.Graph())
    # GGnoise = NX.from_whatever(np.bitwise_xor(bigmat, noise), create_using=NX.Graph()) #old syntax
    GGnoise = NX.to_networkx_graph(np.bitwise_xor(bigmat, noise), create_using=NX.Graph())

    # cliques of noisy data
    Listcliques_noise = [cli for cli in NX.find_cliques(GGnoise)]
    lon_noise = np.array(list(map(len, Listcliques_noise)))

    print("noise", lon_noise)
    print("noise", np.argsort(lon_noise)[::-1])

    nbbest = 10
    for index in np.argsort(lon_noise)[::-1][:nbbest]:
        print("clique size", lon_noise[index], "   nodes: ", Listcliques_noise[index])

    # # cliques of perfect data
    # Listcliques_r=NX.find_cliques(GGraw)
    # lon_r=np.array(map(len,Listcliques_r))

    # print "raw",lon_r
    # print "raw",np.argsort(lon_r)[::-1]

    # nbbest_r=10
    # for index in np.argsort(lon_r)[::-1][:nbbest_r]:
    # print "clique size",lon_r[index],"   nodes: ",Listcliques_r[index]

    if plot:
        NX.draw(GGnoise)
        P.show()
    print("end of test")


def test_findcliques():
    """
    Test for finding highly connected set of spots
    Connection exists between two spots if their mutual angular distance is found in lookup table
    """

    nb_of_spots = -1  # nb of first most intense peaks
    ang_tol = 0.05
    # filename='fiveUO2grains.cor'
    filename = "Zr_A169_0220.cor"
    # filename='Ge_Zr_10000.cor'

    print("************\n For spots 0\n ***************")
    NODES = 0
    give_bestclique(filename, nb_of_spots, ang_tol, nodes=NODES, col_Int=4, verbose=0)

    print("\n******************\nFor the ten next most intense spots\n********************")
    NODES = list(range(1, 11))
    give_bestclique(filename, nb_of_spots, ang_tol, nodes=NODES, col_Int=4, verbose=0)


# ---------------------------------------------------
# -------- MAIN -------------------------------------
# ---------------------------------------------------


if __name__ == "__main__":

    # test_findcliques()
    #test(plot=1)
    """
    #print searchsorted(lon_C0[sortedcliques],4)
    bestclique=sort(C0[argmax(lon_C0)])
    print "bestclique",bestclique
    if 1:
        nbbest_r=200 #to select the number of best cliques found
        for index in sortedcliques[:nbbest_r]:
            print "clique size",lon_C0[index],"   nodes: ",sort(C0[index])
    """

    """
    # all cliques finding in graph
    print"finding cliques ----------------"
    Listcliques_r=find_cliques(GGraw)
    print "OK, that's done"
    lon_r=array(map(len,Listcliques_r))
    maxi=max(lon_r)
    print "raw",lon_r
    print "raw",argsort(lon_r)[::-1]

    nbbest_r=50 #to select the number of best cliques found
    for index in argsort(lon_r)[::-1][:nbbest_r]:
        print "clique size",lon_r[index],"   nodes: ",sort(Listcliques_r[index])


    """

    """
    NX.draw(GGraw)
    P.show()
    """

    """
    # in order to find sets corresponding to each grain -------------------
    listsortedcliques=[Listcliques_r[np.argsort(lon_r)[::-1][elem]] for elem in range(len(Listcliques_r))]
    Grains=NX.Graph()

    def update_graph(Graph,list_spot):

        if len(list_spot)>0:
            NX.Graph.add_nodes_from(list_spot)
            shift_r=list_spot[1:]+list_spot[:1]

            edges=np.transpose(np.array([list_spot,shift_r]))

            NX.Graph.add_edges_from(edges)

    def update_graph2(Graph,list_spot):
        lon=len(list_spot)

        if lon>0:
            #print "listspot",list_spot
            NX.Graph.add_nodes_from(list_spot)
            edges=np.take(array(list_spot),NX.complete_graph(lon).edges())

            NX.Graph.add_edges_from(edges)

    for m in range(6):
        print "**** ",m
        NX.update_graph2(Grains,listsortedcliques[m])



    NX.draw(Grains)
    P.show()

    """
