from numpy import array, ones, where, argmin, ravel, arange, int32, amin, mean
import indexingAnglesLUT as INDEX

try:
    import angulardist
    USE_CYTHON = True
except ImportError:
    print("Cython compiled 'angulardist' module for fast computation of angular distance is not installed!")
    print("Using default module")
    USE_CYTHON = False

# USE_CYTHON = False

def getArgmin(tab_angulardist):
    """
    temporarly doc
    from matrix of mutual angular distances return index of closest neighbour

    TODO: to explicit documentation as a function of tab_angulardist properties only
    """
    return argmin(tab_angulardist, axis=1)

def getProximity(TwicethetaChi, data_theta, data_chi,
                        angtol=0.5, proxtable=0, verbose=0, signchi=1, usecython=USE_CYTHON):
    """
    TwicethetaChi has two elements: 2theta array and chi array (same length!) (theo. data)
    data_theta, data_chi : array of theta, array of chi (exp. data) (same length!)
    

    data_theta array of exp spot
    data_chi array of exp spot

    WARNING: TwicethetaChi contains 2theta instead of data_theta contains theta !
    
    TODO: change this input to 2theta, chi for every arguments
    signchi = 1 fixed old convention
    TODO: remove this option
    TODO: improve documentation
    """
    # theo simul data
    theodata = array([TwicethetaChi[0] / 2., signchi * TwicethetaChi[1]]).T
    # exp data
    sorted_data = array([data_theta, data_chi]).T
    
#     table_dist = calculdist_from_thetachi(sorted_data, theodata)
#     print "table_dist_old", table_dist[:5, :5]
#     print "table_dist_old", table_dist[-5:, -5:]
#     print "table_dist_old", table_dist.shape
    
    if not usecython:
        table_dist = INDEX.calculdist_from_thetachi(sorted_data, theodata)
#         print "table_dist normal", table_dist[:5, :5]
    else:
        # TODO to be improved by not preparing array?
        array_twthetachi_theo = array([TwicethetaChi[0], TwicethetaChi[1]]).T
        array_twthetachi_exp = array([data_theta * 2., data_chi]).T
        
#         print "flags", array_twthetachi_theo.flags
#         print "flags", array_twthetachi_exp.flags
        table_dist = angulardist.calculdist_from_2thetachi(array_twthetachi_theo.copy(order='c'),
                                                           array_twthetachi_exp.copy(order='c'))
        
#         print "table_dist from cython", table_dist[:5, :5]
    
#     print "table_dist_new", table_dist[:5, :5]
#     print "table_dist_new", table_dist[-5:, -5:]
#     print "table_dist_new", table_dist.shape    
    
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
                        angtol=0.5, proxtable=0, verbose=0, signchi=1, usecython=USE_CYTHON):
    """
    TwicethetaChi is an array made of 2 arrays 2theta and chi

    data_theta array of exp spot
    data_chi array of exp spot

    WARNING: TwicethetaChi contains 2theta instead of data_theta contains theta !
    TODO: change this input to 2theta, chi for every arguments

    signchi = 1 fixed old convention
    TODO: remove this option

    TODO: improve documentation
    """
    # theo simul data
    theodata = array([Twicetheta / 2., signchi * Chi]).T
    # exp data
    sorted_data = array([data_theta, data_chi]).T
    
#     table_dist = calculdist_from_thetachi(sorted_data, theodata)
#     print "table_dist_old", table_dist[:5, :5]
#     print "table_dist_old", table_dist[-5:, -5:]
#     print "table_dist_old", table_dist.shape
    
    if not usecython:
        table_dist = INDEX.calculdist_from_thetachi(sorted_data, theodata)
#         print "table_dist", table_dist[:5, :5]
    else:
        # TODO to be improved by not preparing array
        import angulardist
#         print "using cython"
        array_twthetachi_theo = array([Twicetheta, Chi]).T
        array_twthetachi_exp = array([data_theta * 2., data_chi]).T
        
#         print "flags", array_twthetachi_theo.flags
#         print "flags", array_twthetachi_exp.flags
        table_dist = angulardist.calculdist_from_2thetachi(array_twthetachi_theo.copy(order='c'),
                                                           array_twthetachi_exp.copy(order='c'))
    
#     print "table_dist_new", table_dist[:5, :5]
#     print "table_dist_new", table_dist[-5:, -5:]
#     print "table_dist_new", table_dist.shape    
    
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
    
    
if __name__ == '__main__':
    import time
    import numpy
    
    npoints1 = 60  # theo
    npoints2 = 120  # exp
    
    listpoints1 = numpy.random.random_sample((npoints1, 2)) * (180.) - 90.
    listpoints2 = numpy.random.random_sample((npoints2, 2)) * (180.) - 90.
    
    # listpoints1.dtype = numpy.double
    # listpoints2.dtype = numpy.double
    
    listpoints1_theta = listpoints1
    listpoints2_theta = listpoints2
    
    listpoints1_theta[:, 0] = listpoints1[:, 0] / 2.
    listpoints2_theta[:, 0] = listpoints2[:, 0] / 2.
    # theo
    TwicethetaChi1 = listpoints1_theta[:, 0], listpoints1_theta[:, 1]
    
    Twicetheta1, Chi1 = listpoints1_theta[:, 0], listpoints1_theta[:, 1]
    # exp.
    Twicetheta2 = listpoints2_theta[:, 0]
    Chi2 = listpoints2_theta[:, 1]
    
    inittime = time.time()
    res = INDEX.getProximity(TwicethetaChi1,
                              Twicetheta2, Chi2,
                              angtol=.5, proxtable=0)
    finaltime = time.time()
      
    time_old = finaltime - inittime
      
    print('computation time old is ', time_old)
    
    
    
    inittime = time.time()
    res = getProximity(TwicethetaChi1,
                              Twicetheta2, Chi2,
                              angtol=.5, proxtable=0,
                              usecython=False)
    finaltime = time.time()
      
    time_new = finaltime - inittime
      
    print('computation time new is ', time_new)
    
    
    
    inittime = time.time()
    res = getProximity(TwicethetaChi1,
                              Twicetheta2, Chi2,
                              angtol=.5, proxtable=0,
                              usecython=True)
    
#     res = getProximity_new(Twicetheta1, Chi1,
#                               Twicetheta2, Chi2,
#                               angtol=.5, proxtable=0,
#                               usecython=True)
    finaltime = time.time()
      
    time_newc = finaltime - inittime
      
    print('computation time new + cython is ', time_newc)
    
    print("ratio old/new", time_old / time_new)
    print("ratio old/new +cython", time_old / time_newc)
    
    
    

    
    
