r"""
.. module:: matchingrate Documentation
    :synopsis: module to compute matching figure from two sets of laue spots properties

.. moduleauthor:: JS Micha, 2019,  micha 'at' esrf 'dot'fr
"""
import copy
import sys
from numpy import array, where, argmin, amin, mean
import numpy as np

SCIKITLEARN = True
try:
    import sklearn.metrics as sm
except ImportError:
    print("sklearn.metrics is not installed !! ")
    SCIKITLEARN = False


if sys.version_info.major == 3:

    from . import lauecore as LAUE
    from . import CrystalParameters as CP
    from . import generaltools as GT
    from . import LaueGeometry as LaueGeo
    from . dict_LaueTools import CST_ENERGYKEV
    from . dict_LaueTools import dict_Materials
    from . import indexingSpotsSet as ISS

else:
    import lauecore as LAUE
    import CrystalParameters as CP
    import generaltools as GT
    import LaueGeometry as LaueGeo
    from dict_LaueTools import CST_ENERGYKEV
    from dict_LaueTools import dict_Materials
    import indexingSpotsSet as ISS


try:
    if sys.version_info.major == 3:
        from . import angulardist
    else:
        import angulardist

    USE_CYTHON = True
except ImportError:
    # print("-- warning. Cython compiled 'angulardist' module for fast computation of "
    #                                                         "angular distance is not installed!")
    # print("Using default module")
    USE_CYTHON = False


# --- ------------ CONSTANTS
DEG = np.pi / 180.0

#------ USE_CYTHON= FALSE ----------------
def getArgmin(tab_angulardist):
    r"""
    temporarly doc
    from matrix of mutual angular distances return index of closest neighbour

    .. todo:: to explicit documentation as a function of tab_angulardist properties only
    """
    return argmin(tab_angulardist, axis=1)


def SpotLinks(twicetheta_exp,
                chi_exp,
                dataintensity_exp,  # experimental data
                veryclose_angletol,  # tolerance angle
                twicetheta,  # theoretical angles
                chi,
                Miller_ind,
                energy,  # theoretical data
                absoluteindex=None,
                verbose=0):
    r"""
    Creates automatically links between close experimental and theoretical spots
    in 2theta, chi angles (kf) coordinates

    :param twicetheta_exp: list of exp. 2theta angles
    :param chi_exp: list of exp. chi angles
    :param dataintensity_exp: list of dataintensity_exp

    :param veryclose_angletol: finest tolerance angle for association in degree

    :param twicetheta: list of theoretical 2theta angles
    :param chi: list of theoretical chi angles
    :param Miller_ind: list of theoretical 3D Miller_ind
    :param energy: list of theoretical spot energies (keV)

    :param absoluteindex: list of absolute exp. spot index to output in res
                            (conversion table between relative and absolute exp. spot index)
                           i.e.  absoluteindex[localindex] = absolute index

                        if absoluteindex = None: spots indices and order are those of data
                                    (twicetheta_exp, chi_exp, dataintensity_exp)
                    = list_of_absolute_indices
                            : list containing the absolute indices

    :returns: * refine_indexed_spots: dict. with key= exp. spotindex and val=[exp. spotindex,h,k,l]
            * linkedspots_link: list of [absolute exp. spotindex, theo_id]
            * linkExpMiller_link: list of [absolute exp. spotindex, h,k,l]
            * linkIntensity_link: list of [exp. spot intensity]
            * linkResidues_link: list of [absolute exp. spotindex, theo. spotindex, angular pair distance (degrees)]
            * linkEnergy_link: list of [absolute exp. spotindex, theo. spotindex, theo. spot energy (keV)]
            * fields: ["#Spot Exp", "#Spot Theo", "h", "k", "l", "Intensity", "residues (deg)"]

            linkResidues.append([absoluteexpspotindex, theo_id, Resi[theo_id]])
            linkEnergy.append([absoluteexpspotindex, theo_id, Energy_id])
            linkIntensity.append(dataintensity_exp[exp_id])


    .. todo:: To improve! , always difficult to understand immediately the algorithm :-[
    """

    if verbose:
        print("\n ************** Spots Association ************** \n")

    Resi, ProxTable = getProximity(np.array([twicetheta, chi]),  # warning array(2theta, chi)
                            twicetheta_exp / 2.0,
                            chi_exp,  # warning theta, chi for exp
                            proxtable=1,
                            angtol=veryclose_angletol,
                            verbose=0,
                            signchi=1,
                        )[:2]  # sign of chi is +1 when apparently SIGN_OF_GAMMA=1

    # ProxTable is table giving the closest exp.spot index for each theo. spot
    # len(Resi) = nb of theo spots
    # len(ProxTable) = nb of theo spots
    # ProxTable[index_theo]  = index_exp   closest link
    # print "Resi",Resi
    # print "ProxTable",ProxTable
    # print "Nb of theo spots", len(ProxTable)

    # array of theo. spot index
    very_close_ind = np.where(Resi < veryclose_angletol)[0]
    # print "In OnLinkSpotsAutomatic() very close indices",very_close_ind

    # build a list all theo. spots such as: exp_index = List_Exp_spot_close[theo_index]
    List_Exp_spot_close = []
    Miller_Exp_spot = []
    Energy_Exp_spot = []

    # if nb of pairs between exp and theo spots is enough
    if len(very_close_ind) > 0:
        # loop over theo spots index
        for theospot_ind in very_close_ind:
            # print "theospot_ind ",theospot_ind
            List_Exp_spot_close.append(ProxTable[theospot_ind])
            Miller_Exp_spot.append(Miller_ind[theospot_ind])
            Energy_Exp_spot.append(energy[theospot_ind])

    #    print "List_Exp_spot_close", List_Exp_spot_close
    #    print "Miller_Exp_spot", Miller_Exp_spot

    if not List_Exp_spot_close:
        print("Found no pair within tolerance in SpotLinks()")
        return 0

    elif len(List_Exp_spot_close) == 1:
        print("Just a single found! Nb of pairs equal to 1  is not implemented yet in SpotLinks()")
        return 0

    # --------------------------------------------------------------
    # removing exp spot which appears many times
    # (close to several simulated spots of one grain)--------------
    #   now len(List_Exp_spot_close) >= 2
    # ---------------------------------------------------------

    arrayLESC = np.array(List_Exp_spot_close, dtype=float)

    sorted_LESC = np.sort(arrayLESC)

    # print "List_Exp_spot_close", List_Exp_spot_close
    # print "sorted_LESC", sorted_LESC

    diff_index = sorted_LESC - np.array(list(sorted_LESC[1:]) + [sorted_LESC[0]])
    toremoveindex = np.where(diff_index == 0)[0]

    # print "toremoveindex", toremoveindex

    # print "number labelled exp spots", len(List_Exp_spot_close)
    # print "List_Exp_spot_close", List_Exp_spot_close
    # print "Miller_Exp_spot", Miller_Exp_spot

    # find exp spot that can not be indexed safely
    # (too many theo. neighbouring spots)
    if len(toremoveindex) > 0:
        # index of exp spot in arrayLESC that are duplicated
        TOLERANCEANGLE = 0.1
        ambiguous_exp_ind = GT.find_closest(
            np.array(sorted_LESC[toremoveindex], dtype=float), arrayLESC, TOLERANCEANGLE
        )[1]
        # print "ambiguous_exp_ind", ambiguous_exp_ind

        # tagging (inhibiting) exp spots
        # that belong ambiguously to several simulated grains
        for exp_ind in ambiguous_exp_ind:
            Miller_Exp_spot[exp_ind] = None
            Energy_Exp_spot[exp_ind] = 0.0

    # -----------------------------------------------------------
    ProxTablecopy = copy.copy(ProxTable)
    # tag duplicates in ProxTable with negative sign ----------------------
    # ProxTable[index_theo] = index_exp   closest link

    for theo_ind, exp_ind in enumerate(ProxTable):
        where_th_ind = np.where(ProxTablecopy == exp_ind)[0]
        # print "theo_ind, exp_ind ******** ",theo_ind, exp_ind
        if len(where_th_ind) > 1:
            # exp spot (exp_ind) is close to several theo spots
            # then tag the index with negative sign
            for indy in where_th_ind:
                ProxTablecopy[indy] = -ProxTable[indy]
            # except that which corresponds to the closest
            closest = np.argmin(Resi[where_th_ind])
            # print "residues = Resi[where_th_ind]",Resi[where_th_ind]
            # print "closest",closest
            # print "where_exp_ind[closest]",where_th_ind[closest]
            # print "Resi[where_th_ind[closest]]", Resi[where_th_ind[closest]]
            ProxTablecopy[where_th_ind[closest]] *= -1

    # ------------------------------------------------------------------
    # print "ProxTable after duplicate removal tagging"
    # print ProxTablecopy

    # print "List_Exp_spot_close",List_Exp_spot_close
    # print "Results",[Miller_Exp_spot, List_Exp_spot_close]

    # list of exp. spot index that have only one theo. neighbouring spots
    singleindices = []

    # dictionary of links (pairs) between exp. and theo. spots
    refine_indexed_spots = {}

    # loop over theo. spot index
    for theo_ind in list(range(len(List_Exp_spot_close))):

        exp_index = List_Exp_spot_close[theo_ind]

        # print "exp_index",exp_index

        # there is not exp_index in singleindices
        if not singleindices.count(exp_index):
            # so append singleindices with exp_index
            singleindices.append(exp_index)

            # lisf of index of theo spot that are the closest for a given exp. spot
            theo_index = np.where(ProxTablecopy == exp_index)[0]
            # print "theo_index", theo_index

            # unambiguous pairing
            if len(theo_index) == 1:
                refine_indexed_spots[exp_index] = [exp_index,
                                                    theo_index,
                                                    Miller_Exp_spot[theo_ind],
                                                    Energy_Exp_spot[theo_ind]]

            # in case of several theo. candidate, keep the closest to exp. spot
            else:
                # print "Resi[theo_index]", Resi[theo_index]
                closest_theo_ind = np.argmin(Resi[theo_index])
                # print theo_index[closest_theo_ind]

                # pairing if distance withing angular tolerance
                if Resi[theo_index][closest_theo_ind] < veryclose_angletol:
                    refine_indexed_spots[exp_index] = [
                        exp_index,
                        theo_index[closest_theo_ind],
                        Miller_Exp_spot[theo_ind],
                        Energy_Exp_spot[theo_ind]]

        # there is already exp_index in singleindices
        else:
            # do not update the dictionary 'refine_indexed_spots'
            if verbose:
                print("Experimental spot #%d may belong to several theo. spots!"
                    % exp_index)
    # find theo spot linked to exp spot ---------------------------------

    # refine_indexed_spots is a dictionary:
    # key is experimental spot index and value is [experimental spot index,h,k,l]
    # print "refine_indexed_spots",refine_indexed_spots

    listofpairs = []
    linkExpMiller = []
    linkIntensity = []
    linkResidues = []
    linkEnergy = []
    # Dataxy = []

    for val in list(refine_indexed_spots.values()):

        exp_id, theo_id, Miller_id, Energy_id = val

        if Miller_id is not None:

            # absoluteindex is a list of absolute exp. spots indices
            if absoluteindex is not None:
                absoluteexpspotindex = absoluteindex[exp_id]
            else:
                absoluteexpspotindex = exp_id

            # appending lists of links
            # Exp, Theo,  where -1 for specifying that it came from automatic linking
            listofpairs.append([absoluteexpspotindex, theo_id])
            linkExpMiller.append(
                [float(absoluteexpspotindex)] + [float(elem) for elem in Miller_id])  # float(val) for further handling as floats array
            linkResidues.append([absoluteexpspotindex, theo_id, Resi[theo_id]])
            linkEnergy.append([absoluteexpspotindex, theo_id, Energy_id])
            linkIntensity.append(dataintensity_exp[exp_id])
            # Dataxy.append([ LaueToolsframe.data_pixX[val[0]], LaueToolsframe.data_pixY[val[0]]])

    linkedspots_link = np.array(listofpairs)
    linkExpMiller_link = linkExpMiller
    linkIntensity_link = linkIntensity
    linkResidues_link = linkResidues
    linkEnergy_link = linkEnergy
    fields = ["#Spot Exp", "#Spot Theo", "h", "k", "l", "Intensity", "residues (deg)"]

    # self.Data_X, self.Data_Y = np.transpose( np.array(Dataxy) )

    return (refine_indexed_spots,
        linkedspots_link,
        linkExpMiller_link,
        linkIntensity_link,
        linkResidues_link,
        linkEnergy_link,
        fields)

def getProximity_multimatrices(Arr_Theo2Theta, Arr_TheoChi, data_theta, data_chi, angtol=0.5,
                                                                            proxtable=0,
                                                                            verbose=0,
                                                                            signchi=1,
                                                                            usecython=USE_CYTHON):
    """
    WARNING: ArrTheo2theta   contains 2theta instead of data_theta contains theta !
    """
    nbpeaksbunches = len(Arr_Theo2Theta)
    # theo simul data
    # theodata = array([np.ravel(Arr_Theo2Theta)/2.0, signchi *np.ravel(Arr_TheoChi)]).T
    # # exp data
    sorted_data = array([data_theta, data_chi]).T

    #     table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)
    #     print "table_dist_old", table_dist[:5, :5]
    #     print "table_dist_old", table_dist[-5:, -5:]
    #     print "table_dist_old", table_dist.shape

    print('sorted_data.shape', sorted_data.shape)
    print('Arr_Theo2Theta.shape', Arr_Theo2Theta.shape)

    if not usecython:
        # table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)

        table_dist = GT.computeMutualAngles(array([np.ravel(Arr_Theo2Theta),
                                        signchi * np.ravel(Arr_TheoChi)]).T,
                                        sorted_data)

        print('table_dist.shape', table_dist.shape)

    #         print "table_dist normal", table_dist[:5, :5]
    #     print "table_dist_new", table_dist.shape

    # shape is: along i   dim =  NBMAXSPOTS*nbmatrices = dim(theodata)
    #           along j   dim = dim(sorted_data)  experimental

    # prox_table = getArgmin(table_dist)

    # print "shape table_dist",shape(table_dist)
    # table_dist has shape (len(theo),len(exp))
    # tab[0] has len(exp) elements
    # tab[i] with i index of theo spot contains all distance from theo spot #i and all the exprimental spots
    # prox_table has len(exp) elements
    # prox_table[i] is the index of the exp. spot closest to theo spot (index i)
    # table_dist[i][prox_table[i]] is the minimum angular distance separating theo spot i from a exp spot of index prox_table[i]

    if verbose:
        #        print "/0", np.where(table_dist[:, 0] < 1.)
        #        print np.argmin(table_dist[:, 0])
        #        print "/4 exp", np.where(table_dist[:, 4] < 1.)
        #        print np.argmin(table_dist[:, 4])
        #        print np.shape(table_dist)
        print(table_dist[:, 4])

    #    pos_closest = np.transpose(array([arange(len(theodata)), prox_table]))
    #     nb_exp_spots = len(data_chi)
    #     nb_theo_spots = len(theodata)
    #     pos_closest_1d = array(arange(nb_theo_spots) * nb_exp_spots * ones(nb_theo_spots) + \
    #                                prox_table,
    #                                dtype=int32)
    #     allresidues = ravel(table_dist)[pos_closest_1d]

    allresidues = amin(table_dist, axis=1)
    taballresidues = allresidues.reshape((nbpeaksbunches, len(allresidues)//nbpeaksbunches))

    #     print "allresidues", allresidues
    # len(allresidues)  = len(theo)

    #     print "theodata", theodata
    #     print 'len(allresidues)', len(allresidues)
    if proxtable == 0:
        SIMILRATYTHRESHOLD = 0.9999
        nb_in_res = getNbMatches(taballresidues, SIMILRATYTHRESHOLD)

        # taballresiduesM=np.ma.masked_greater_equal(taballresidues,angtol)
        # meanres = np.mean(taballresiduesM,axis=1)
        # maxi = np.amax(taballresiduesM,axis=1)

        # res = allresidues[cond]
        # longueur_res = len(cond[0])
        # #         print allresidues
        # #         print "len(res)", len(res)
        # #         print "longueur_res", longueur_res
        # nb_in_res = len(res)
        # maxi = max(res)
        # meanres = mean(res)

        return taballresidues, taballresidues[0], nb_in_res, len(allresidues)
        # return taballresidues, taballresidues[0], nb_in_res, len(allresidues), meanres, maxi


def getNbMatches(residues, thresholdsimilarity):
    """return nb of good matches
    """
    cond = where(residues > thresholdsimilarity)
    return len(cond[0])
    # unique_elements, counts_elements = np.unique(cond[0], return_counts=True)
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements, counts_elements)))
    # nb_in_res=counts_elements
    # return np.sum(counts_elements)

def getProximity(TwicethetaChi,
                    data_theta,
                    data_chi,
                    angtol=0.5,
                    proxtable=0,
                    verbose=0,
                    signchi=1,
                    usecython=USE_CYTHON):
    r"""
    :param TwicethetaChi: (simulated or theoretical) two arrays of 2theta array and chi array (same length!)
    :param data_theta: array of theta angles (of experimental spots)
    :param data_chi: array of chi (same length than data_theta!)

    :returns:  if proxtable = 1 : proxallresidues, res, nb_in_res, len(allresidues), meanres, maxi

    .. warning:: TwicethetaChi contains 2theta instead of data_theta theta contains theta !

    .. todo::
        * change the input with 2theta angles pnly to avoid confusion
        * remove the option signchi = 1 fixed old convention
    """
    # theo simul data
    theodata = array([TwicethetaChi[0] / 2.0, signchi * TwicethetaChi[1]]).T
    # exp data
    sorted_data = array([data_theta, data_chi]).T

    #     table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)
    #     print "table_dist_old", table_dist[:5, :5]
    #     print "table_dist_old", table_dist[-5:, -5:]
    #     print "table_dist_old", table_dist.shape

    if not usecython:
        table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)

        # for cartesian distance only, crude approximate for kf_direction=Z>0
        # import scipy
        # table_dist = scipy.spatial.distance.cdist(sorted_data, theodata).T

    else:
        # TODO to be improved by not preparing array?
        array_twthetachi_theo = array([TwicethetaChi[0], TwicethetaChi[1]]).T
        array_twthetachi_exp = array([data_theta * 2.0, data_chi]).T

        #         print "flags", array_twthetachi_theo.flags
        #         print "flags", array_twthetachi_exp.flags
        table_dist = angulardist.calculdist_from_2thetachi(array_twthetachi_theo.copy(order="c"),
                                                            array_twthetachi_exp.copy(order="c"))

    #         print "table_dist from cython", table_dist[:5, :5]

    #     print "table_dist_new", table_dist[:5, :5]
    #     print "table_dist_new", table_dist[-5:, -5:]
    #     print "table_dist_new", table_dist.shape

    if proxtable == 1:
        prox_table = getArgmin(table_dist)
    # print "shape table_dist",shape(table_dist)
    # table_dist has shape (len(theo),len(exp))
    # tab[0] has len(exp) elements
    # tab[i] with i index of theo spot contains all distance from theo spot #i and all the exprimental spots
    # prox_table has len(theo) elements
    # prox_table[i] is the index of the exp. spot closest to theo spot (index i)
    # table_dist[i][prox_table[i]] is the minimum angular distance
    # separating theo spot i from a exp spot of index prox_table[i]

    if verbose:
        #        print "/0", np.where(table_dist[:, 0] < 1.)
        #        print np.argmin(table_dist[:, 0])
        #        print "/4 exp", np.where(table_dist[:, 4] < 1.)
        #        print np.argmin(table_dist[:, 4])
        #        print np.shape(table_dist)
        print(table_dist[:, 4])

    #    pos_closest = np.transpose(array([arange(len(theodata)), prox_table]))
    #     nb_exp_spots = len(data_chi)
    #     nb_theo_spots = len(theodata)
    #     pos_closest_1d = array(arange(nb_theo_spots) * nb_exp_spots * ones(nb_theo_spots) + \
    #                                prox_table,
    #                                dtype=int32)
    #     allresidues = ravel(table_dist)[pos_closest_1d]

    allresidues = amin(table_dist, axis=1)

    #     print "allresidues", allresidues

    # len(allresidues)  = len(theo)

    #     print "theodata", theodata
    #     print 'len(allresidues)', len(allresidues)
    if proxtable == 0:
        cond = where(allresidues < angtol)
        res = allresidues[cond]
        longueur_res = len(cond[0])
        #         print allresidues
        #         print "len(res)", len(res)
        #         print "longueur_res", longueur_res
        if longueur_res <= 1:
            nb_in_res = longueur_res
            maxi = -min(allresidues)
            meanres = -1
        else:
            nb_in_res = len(res)
            maxi = max(res)
            meanres = mean(res)

        return allresidues, res, nb_in_res, len(allresidues), meanres, maxi

    elif proxtable == 1:
        return allresidues, prox_table, table_dist


def getProximity_new(Twicetheta, Chi, data_theta, data_chi,
                                                    angtol=0.5, proxtable=0,
                                                    verbose=0, signchi=1,
                                                    usecython=USE_CYTHON):
    """
    see doc of getProximity()

    """
    # theo simul data
    theodata = array([Twicetheta / 2.0, signchi * Chi]).T
    # exp data
    sorted_data = array([data_theta, data_chi]).T

    #     table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)
    #     print "table_dist_old", table_dist[:5, :5]
    #     print "table_dist_old", table_dist[-5:, -5:]
    #     print "table_dist_old", table_dist.shape

    if not usecython:
        table_dist = GT.calculdist_from_thetachi(sorted_data, theodata)
    #         print "table_dist", table_dist[:5, :5]
    else:
        # TODO to be improved by not preparing array
        import angulardist

        #         print "using cython"
        array_twthetachi_theo = array([Twicetheta, Chi]).T
        array_twthetachi_exp = array([data_theta * 2.0, data_chi]).T

        #         print "flags", array_twthetachi_theo.flags
        #         print "flags", array_twthetachi_exp.flags
        table_dist = angulardist.calculdist_from_2thetachi(array_twthetachi_theo.copy(order="c"),
                                                            array_twthetachi_exp.copy(order="c"))

    prox_table = getArgmin(table_dist)
    # print "shape table_dist",shape(table_dist)
    # table_dist has shape (len(theo),len(exp))
    # tab[0] has len(exp) elements
    # tab[i] with i index of theo spot contains all distance from theo spot #i and all the exprimental spots
    # prox_table has len(theo) elements
    # prox_table[i] is the index of the exp. spot closest to theo spot (index i)
    # table_dist[i][prox_table[i]] is the minimum angular distance separating theo spot i from a exp spot of index prox_table[i]

    if verbose:
        #        print "/0", np.where(table_dist[:, 0] < 1.)
        #        print np.argmin(table_dist[:, 0])
        #        print "/4 exp", np.where(table_dist[:, 4] < 1.)
        #        print np.argmin(table_dist[:, 4])
        #        print np.shape(table_dist)
        print(table_dist[:, 4])

    #    pos_closest = np.transpose(array([arange(len(theodata)), prox_table]))
    #     nb_exp_spots = len(data_chi)
    #     nb_theo_spots = len(theodata)
    #     pos_closest_1d = array(arange(nb_theo_spots) * nb_exp_spots * ones(nb_theo_spots) + \
    #                                prox_table,
    #                                dtype=int32)
    #     allresidues = ravel(table_dist)[pos_closest_1d]

    allresidues = amin(table_dist, axis=1)

    # len(allresidues)  = len(theo)

    #     print "theodata", theodata
    #     print 'len(allresidues)', len(allresidues)
    if proxtable == 0:
        cond = where(allresidues < angtol)
        res = allresidues[cond]
        longueur_res = len(cond[0])
        # print allresidues
        #         print "len(res)", len(res)
        #         print "longueur_res", longueur_res
        if longueur_res <= 1:
            nb_in_res = longueur_res
            maxi = -min(allresidues)
            meanres = -1
        else:
            nb_in_res = len(res)
            maxi = max(res)
            meanres = mean(res)

        return allresidues, res, nb_in_res, len(allresidues), meanres, maxi

    elif proxtable == 1:
        return allresidues, prox_table, table_dist


def Angular_residues_np(test_Matrix, twicetheta_data, chi_data, ang_tol=0.5,
                                                                key_material="Si",
                                                                emin=5,
                                                                emax=25,
                                                                ResolutionAngstrom=False,
                                                                detectorparameters=None,
                                                                onlyXYZ=False,
                                                                simthreshold=0.999,
                                                                dictmaterials=dict_Materials):
    r"""
    Computes angular residues between pairs of close exp. and
    theo. spots simulated according to test_Matrix, within tolerance angle

    .. note::
        * used in manual indexation
        * Used in AutoIndexation module
        * Used in FileSeries

    :param twicetheta_data: experimental 2theta angles of scattered beams or spots (kf vectors)
    :type twicetheta_data: array
    :param chi_data: experimental chi angles of scattered beams or spots (kf vectors)
    :param test_Matrix: Orientation matrix
    :type test_Matrix: 3x3 array
    :param ang_tol: angular tolerance in degrees to accept or reject a exp. and theo pair
    :type ang_tol: scalar

    :param detectorparameters: Dictionary of detector parameters (key, value) that must contain:
                            'kf_direction' , general position of detector plane
                            'detectordistance', detector distance (mm)
                            'detectordiameter', detector diameter (mm)
                            'pixelsize' and 'dim'
    """
    if detectorparameters is None:
        # use default parameter
        kf_direction = "Z>0"
        detectordistance = 70.0
        detectordiameter = 165.0
        pixelsize = 165.0 / 2048
        dim = (2048, 2048)
    else:
        kf_direction = detectorparameters["kf_direction"]
        detectordistance = detectorparameters["detectorparameters"][0]
        detectordiameter = detectorparameters["detectordiameter"]
        pixelsize = detectorparameters["pixelsize"]
        dim = detectorparameters["dim"]

    #     print "kf_direction,pixelsize,detectordistance", kf_direction, pixelsize, detectordistance

    # ---simulation-----------------------------------
    grain = CP.Prepare_Grain(key_material, test_Matrix, dictmaterials=dictmaterials)

    # array(vec) and array(indices) (here with fastcompute=0 array(indices)=0)
    # of spots exiting the crystal towards detector plane and geometry given by kf_direction
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin,
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    kf_direction=kf_direction,
                                    ResolutionAngstrom=ResolutionAngstrom,
                                    dictmaterials=dictmaterials)

    if not SCIKITLEARN or not onlyXYZ:
        # 2theta,chi of spot which are on camera (with harmonics)
        # None because no need of hkl vectors
        # TwicethetaChi without energy calculations and hkl selection
        # without use of spots instantation (faster)
        TwicethetaChi = LAUE.filterLaueSpots_full_np(spots2pi[0][0], None, onlyXYZ=False,
                                                        HarmonicsRemoval=0,
                                                        fastcompute=1,
                                                        kf_direction=kf_direction,
                                                        detectordistance=detectordistance,
                                                        detectordiameter=detectordiameter,
                                                        pixelsize=pixelsize,
                                                        dim=dim)

        # old calculation with spots instantiation
        #     TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1,
        #                                          kf_direction=kf_direction,
        #                                          detectordistance=detectordistance,
        #                                          detectordiameter=detectordiameter,
        #                                          pixelsize=pixelsize,
        #                                          dim=dim)

        #     print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
        if len(TwicethetaChi[0]) == 0:
            #         print 'no peak found'
            return None

        # no particular gain...?
        return getProximity(TwicethetaChi, twicetheta_data / 2.0, chi_data, angtol=ang_tol,
                                                                                        proxtable=0)

    else:
        Q_XYZ_onCam = LAUE.filterLaueSpots_full_np(spots2pi[0][0], None, onlyXYZ=True,
                                                    HarmonicsRemoval=0,
                                                    fastcompute=1,
                                                    kf_direction=kf_direction,
                                                    detectordistance=detectordistance,
                                                    detectordiameter=detectordiameter,
                                                    pixelsize=pixelsize,
                                                    dim=dim)

        # Y should be Q vectors corresponding to exp. twicetheta_data and chi_data
        Y = LaueGeo.from_twchi_to_q((twicetheta_data, chi_data)).T

        # print("Q_XYZ_onCam   theo",Q_XYZ_onCam)
        # print("Y exp.",Y)

        # print("Q_XYZ_onCam   theo  shape",Q_XYZ_onCam.shape)
        # print("Y exp.  shape",Y.shape)
        # return ssd.cosine(Q_XYZ_onCam, Y)
        # return np.arccos(1-sm.pairwise.cosine_similarity(Q_XYZ_onCam, Y, dense_output=True))*180./np.pi
        smMat = sm.pairwise.cosine_similarity(Q_XYZ_onCam, Y, dense_output=True)
        SIMILRATYTHRESHOLD = simthreshold
        nb_in_res = getNbMatches(smMat, SIMILRATYTHRESHOLD)
        return nb_in_res


def Angular_residues_np_multimatrices(ListMatrices, twicetheta_data, chi_data, ang_tol=0.5,
                                                                    key_material="Si",
                                                                    emin=5,
                                                                    emax=25,
                                                                    ResolutionAngstrom=False,
                                                                    detectorparameters=None,
                                                                    dictmaterials=dict_Materials):

    """ See doc of Angular_residues_np()

    """
    NBMAXPEAKS = 1700

    if detectorparameters is None:
        # use default parameter
        kf_direction = "Z>0"
        detectordistance = 70.0
        detectordiameter = 165.0
        pixelsize = 165.0 / 2048
        dim = (2048, 2048)
    else:
        kf_direction = detectorparameters["kf_direction"]
        detectordistance = detectorparameters["detectorparameters"][0]
        detectordiameter = detectorparameters["detectordiameter"]
        pixelsize = detectorparameters["pixelsize"]
        dim = detectorparameters["dim"]

    #     print "kf_direction,pixelsize,detectordistance", kf_direction, pixelsize, detectordistance

    # ---simulation

    nbmatrices = len(ListMatrices)

    Arr_Theo2Theta = np.empty((nbmatrices, NBMAXPEAKS))
    Arr_TheoChi = np.empty((nbmatrices, NBMAXPEAKS))
    # print "Reference Element or structure label", key_material
    for matindex in list(range(nbmatrices)):
        test_Matrix = ListMatrices[matindex]
        grain = CP.Prepare_Grain(key_material, test_Matrix)
        # array(vec) and array(indices) (here with fastcompute=0
        # array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
        spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                    CST_ENERGYKEV / emin,
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    kf_direction=kf_direction,
                                    ResolutionAngstrom=ResolutionAngstrom,
                                    dictmaterials=dictmaterials)

        # 2theta,chi of spot which are on camera (with harmonics)
        # None because no need of hkl vectors
        # TwicethetaChi without energy calculations and hkl selection
        # without use of spots instantation (faster)
        TheoTwicethetaChi = LAUE.filterLaueSpots_full_np(spots2pi[0][0], None, HarmonicsRemoval=0,
                                                        fastcompute=1,
                                                        kf_direction=kf_direction,
                                                        detectordistance=detectordistance,
                                                        detectordiameter=detectordiameter,
                                                        pixelsize=pixelsize,
                                                        dim=dim)
        # print("matindex",matindex)
        Arr_Theo2Theta[matindex] = TheoTwicethetaChi[0][:NBMAXPEAKS]
        Arr_TheoChi[matindex] = TheoTwicethetaChi[1][:NBMAXPEAKS]
        # if len(TheoTwicethetaChi[0]) == 0:
        #     #         print 'no peak found'
        #     return None

    # no particular gain...?
    return getProximity_multimatrices(Arr_Theo2Theta,
                                    Arr_TheoChi, twicetheta_data / 2.0,
                                    chi_data,
                                    angtol=ang_tol,
                                    proxtable=0)


def Angular_residues(test_Matrix, twicetheta_data, chi_data, ang_tol=0.5, key_material="Si",
                                emin=5,
                                emax=25,
                                ResolutionAngstrom=False,
                                detectorparameters=None,
                                dictmaterials=dict_Materials):

    """ see doc of Angular_residues_np()

    """
    if detectorparameters is None:
        # use default parameter
        kf_direction = "Z>0"
        detectordistance = 70.0
        detectordiameter = 165.0
        pixelsize = 165.0 / 2048
        dim = (2048, 2048)
    else:
        kf_direction = detectorparameters["kf_direction"]
        detectordistance = detectorparameters["detectorparameters"][0]
        detectordiameter = detectorparameters["detectordiameter"]
        pixelsize = detectorparameters["pixelsize"]
        dim = detectorparameters["dim"]

    #     print "kf_direction,pixelsize,detectordistance", kf_direction, pixelsize, detectordistance

    # ---simulation
    # print "Reference Element or structure label", key_material

    # spots2pi = generalfabriquespot_fromMat_veryQuick(CST_ENERGYKEV/emax,CST_ENERGYKEV/emin,[grain],1,fastcompute=1,fileOK=0,verbose=0)
    grain = CP.Prepare_Grain(key_material, test_Matrix)

    # array(vec) and array(indices) (here with fastcompute=0 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax,
                                    CST_ENERGYKEV / emin,
                                    [grain],
                                    fastcompute=1,
                                    verbose=0,
                                    kf_direction=kf_direction,
                                    ResolutionAngstrom=ResolutionAngstrom,
                                    dictmaterials=dictmaterials)
    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi,
                                        fileOK=0,
                                        fastcompute=1,
                                        kf_direction=kf_direction,
                                        detectordistance=detectordistance,
                                        detectordiameter=detectordiameter,
                                        pixelsize=pixelsize,
                                        dim=dim)

    #     print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
    #     print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
    if len(TwicethetaChi[0]) == 0:
        return None

    return getProximity(TwicethetaChi, twicetheta_data / 2.0, chi_data, angtol=ang_tol, proxtable=0)


def getMatchingRate(indexed_spots_dict, test_Matrix, ang_tol, simulparam, removeharmonics=1,
                                                                            detectordiameter=165.0):
    r"""
    Gets matching rate for an orientation matrix
                    for all exp. data stored in dictionary indexed_spots_dict

    input:
    :param indexed_spots_dict: dict of exp spots (indexed or not)
    :param test_Matrix: orientation matrix (3x3)
    :param ang_tol: angular tolerance below which a pair between
                            exp. and simulated spots is accepted
    :param simulparam: tuple containing emin, emax, key_material, detectorparameters
    :param removeharmonics: 0 or 1 to compute respectively without or with harmonics in simulated pattern

    :returns: * [0], number of pairs between exp and theo spots
                * [1], number of theo. spots
                * [2], matching rate (ratio*100 of the two previous numbers)
    """
    emin, emax, key_material, detectorparameters = simulparam
    # TODO to shorten
    _, twicetheta_data, chi_data, _, _, intensity_data = ISS.getSpotsData(indexed_spots_dict).T

    # simulated data
    grain = CP.Prepare_Grain(key_material, test_Matrix)
    (Twicetheta, Chi, Miller_ind,
    _, _, Energy) = LAUE.SimulateLaue(grain,
                                            emin,
                                            emax,
                                            detectorparameters,
                                            removeharmonics=removeharmonics,
                                            detectordiameter=detectordiameter * 1.25)

    nb_of_simulated_spots = len(Twicetheta)

    # find close pairs between exp. and theo. spots
    res = SpotLinks(twicetheta_data,
                    chi_data,
                    intensity_data,  # experimental data
                    ang_tol,  # tolerance angle
                    Twicetheta,
                    Chi,
                    Miller_ind,
                    Energy,
                    absoluteindex=None)

    if res == 0 or len(res[1]) == 0:
        return None, nb_of_simulated_spots, None

    print("res_links", res[1])

    print("nb of exp spots data probed :", len(twicetheta_data))

    nb_of_links = len(res[1])

    matching_rate = 100.0 * nb_of_links / nb_of_simulated_spots

    return nb_of_links, nb_of_simulated_spots, matching_rate


def getStatsOnMatching(List_of_Angles,
                    twicetheta_data,
                    chi_data,
                    key_material,
                    ang_tol=1.0,
                    emin=5,
                    emax=25,
                    intensity_data=None,
                    verbose=0):
    """
    return matching rate in terms of nb of close pairs of spots (exp. and simul. ones)
    within angular tolerance
    (simulated pattern contains harmonics)

    :param List_of_Angles: list of 3 angles (EULER angles)
    :type List_of_Angles: list of 3 floats
    :param twicetheta_data, chi_data: experimental spots angles coordinates (kf vectors)
    :type twicetheta_data, chi_data: 2 arrays of floats
    :param key_material: key for material used in simulation
    :type key_material: string
    :param ang_tol: matching angular tolerance to form pairs
    :type ang_tol: float

    .. note:: USED in filterEulersList()
    """
    kk = 0
    allmatchingrate = []
    for angles_sol in List_of_Angles:

        test_Matrix = GT.fromEULERangles_toMatrix(angles_sol)

        res = Angular_residues(test_Matrix,
                                twicetheta_data,
                                chi_data,
                                ang_tol=ang_tol,
                                key_material=key_material,
                                emin=emin,
                                emax=emax)

        if res is None:
            continue

        matching_rate = 100.0 * res[2] / res[3]
        # absolute_matching_rate = res[2]
        allmatchingrate.append(matching_rate)

        if verbose:
            print("*" * 30)
            print(
                "res for k:%d and angles: [%.1f,%.1f,%.1f]"
                % (kk, angles_sol[0], angles_sol[1], angles_sol[2]))
            #        print "res",res[2:]
            print("Matching rate ----(in %%):                %.2f " % matching_rate)
            print("Nb of close spot pairs (< %.2f deg) : %d" % (ang_tol, res[2]))
            print("Nb of simulated spots : %d" % res[3])
            print("mean angular residues %.2f deg." % res[4])
            print("highest residues %.2f deg." % res[5])
        kk += 1

    return np.argsort(np.array(allmatchingrate))[::-1], allmatchingrate


if __name__ == "__main__":
    import time

    npoints1 = 60  # theo
    npoints2 = 120  # exp

    listpoints1 = np.random.random_sample((npoints1, 2)) * (180.0) - 90.0
    listpoints2 = np.random.random_sample((npoints2, 2)) * (180.0) - 90.0

    # listpoints1.dtype = numpy.double
    # listpoints2.dtype = numpy.double

    listpoints1_theta = listpoints1
    listpoints2_theta = listpoints2

    listpoints1_theta[:, 0] = listpoints1[:, 0] / 2.0
    listpoints2_theta[:, 0] = listpoints2[:, 0] / 2.0
    # theo
    TwicethetaChi1 = listpoints1_theta[:, 0], listpoints1_theta[:, 1]

    Twicetheta1, Chi1 = listpoints1_theta[:, 0], listpoints1_theta[:, 1]
    # exp.
    Twicetheta2 = listpoints2_theta[:, 0]
    Chi2 = listpoints2_theta[:, 1]

    inittime = time.time()
    res = getProximity(TwicethetaChi1, Twicetheta2, Chi2, angtol=0.5, proxtable=0)
    finaltime = time.time()

    time_old = finaltime - inittime

    print("computation time old is ", time_old)

    inittime = time.time()
    res = getProximity(TwicethetaChi1, Twicetheta2, Chi2, angtol=0.5, proxtable=0, usecython=False)
    finaltime = time.time()

    time_new = finaltime - inittime

    print("computation time new is ", time_new)

    inittime = time.time()
    res = getProximity(TwicethetaChi1, Twicetheta2, Chi2, angtol=0.5, proxtable=0, usecython=True)

    #     res = getProximity_new(Twicetheta1, Chi1,
    #                               Twicetheta2, Chi2,
    #                               angtol=.5, proxtable=0,
    #                               usecython=True)
    finaltime = time.time()

    time_newc = finaltime - inittime

    print("computation time new + cython is ", time_newc)

    print("ratio old/new", time_old / time_new)
    print("ratio old/new +cython", time_old / time_newc)
