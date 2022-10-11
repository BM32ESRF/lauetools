from __future__ import absolute_import
"""
scripts to test selected spots for indexation (non contiguous)
to narrow the number of angular distances to be compared to the LUT

run with ipython
~/LaueToolsPy3/LaueTools/scripts$ ipython -i test_indexingselectedspots.py

JS Micha   Aug 2019
"""
import os
import numpy as np

import networkx as NX

import LaueTools
LaueToolsFolder = os.path.split(os.path.abspath(LaueTools.__file__))[0]
print("Using LaueToolsFolder: ", LaueToolsFolder)
import LaueTools.graingraph as GG
import LaueTools.indexingAnglesLUT as IAL
import LaueTools.findorient as FO
import LaueTools.generaltools as GT
import LaueTools.IOLaueTools as IOLT
import LaueTools.dict_LaueTools as DictLT
import LaueTools.CrystalParameters as CP
import LaueTools.indexingSpotsSet as ISS

# select file with data i.e. list of spots from .cor file (2the,chi,x,y,I) and  detector geometry
filename = os.path.join(LaueToolsFolder,'Examples', 'CuSi','corfiles','SiCustrain0.cor')
filename = os.path.join(LaueToolsFolder,'Examples', 'UO2','dat_UO2_A163_2_0028_LT_0.cor')

CCDLabel = 'sCMOS'

# be careful of detectorparameters["detectordiameter"] = 165

# spots selection 
# selectedspots_ind = [0,2,4,6,8]
selectedspots_ind = [2, 5, 12, 26, 55, 165] #[ 2,  6, 33, 49, 55, 59, 96]
# selectedspots_ind = [0,124]

# reference LUT and material
nLUT=3 # nLUT :  higher index probed 
key_material = 'UO2'

# matching rate parameters
angle_tol = .5
Nb_criterium = 15
emax = 22

#------------   END of input parameters ---------------


# -----------  STARTING SCRIPT ----------------------
# ------    Handling parameters

# Reading Laue pattern spot positions (.cor file)
# ( alldata, data_theta, data_chi, data_pixX, data_pixY, data_I, detParam, CCDcalib, )
data = IOLT.readfile_cor(filename, output_CCDparamsdict=True)
data_theta, data_chi, data_I = data[1], data[2], data[5]
CCDcalib = data[7]
detParam = data[6]

detectorparameters = {}
detectorparameters["kf_direction"] = 'Z>0'
detectorparameters["detectorparameters"] = detParam
detectorparameters["detectordiameter"] = 165
detectorparameters["pixelsize"] = DictLT.dict_CCD[CCDLabel][1]
detectorparameters["dim"]=DictLT.dict_CCD[CCDLabel][0]

# set of mutual distances -------
Theta = data_theta[selectedspots_ind]
Chi = data_chi[selectedspots_ind]
Intens = data_I[selectedspots_ind]
sorted_data = np.transpose(np.array([Theta, Chi, Intens]))

print("Calculating all angular distances ...")
Tabledistance = GT.calculdist_from_thetachi(
        sorted_data[:, 0:2], sorted_data[:, 0:2])
#-----------------------

# Building LUT of reference angles
dictmaterials = DictLT.dict_Materials
latticeparams = dictmaterials[key_material][1]
Rules = dictmaterials[key_material][2]
B = CP.calc_B_RR(latticeparams)
LUT = IAL.build_AnglesLUT(B, nLUT, MaxRadiusHKL=False,
            cubicSymmetry=CP.hasCubicSymmetry(key_material, dictmaterials=dictmaterials),
            ApplyExtinctionRules = Rules)

# ---------   START of INDEXATION  -------------
bestmatList = []
stats_resList = []

#spot_index 1 and 2  = absolute index
# i1,i2 local index to scan selectedspots_ind

# loop over all possible spots pairs in selected set of spots
for i1, i2 in GT.allpairs_in_set(range(len(selectedspots_ind))):

    spot_index_1 = selectedspots_ind[i1]
    spot_index_2 = selectedspots_ind[i2]

    print("\n* test script **\n\ni1,i2, local_spotindex1,local_spotindex2", i1, i2, spot_index_1, spot_index_2)

    All_2thetas = data_theta*2.
    All_Chis = data_chi

    coords_1 = All_2thetas[spot_index_1], All_Chis[spot_index_1]
    coords_2 = All_2thetas[spot_index_2], All_Chis[spot_index_2]

    # Table of distances is very small (nb selected spots**2)
    expdistance_2spots = Tabledistance[i1, i2]

    print('expdistance_2spots  = ', expdistance_2spots)

    UBS_MRS = IAL.getUBs_and_MatchingRate(spot_index_1,
                                        spot_index_2,
                                        angle_tol,
                                        expdistance_2spots,
                                        coords_1,
                                        coords_2,
                                        nLUT,
                                        B,
                                        All_2thetas,
                                        All_Chis,
                                        LUT=None,
                                        key_material=key_material,
                                        emax=emax,
                                        ang_tol_MR=angle_tol,
                                        detectorparameters=detectorparameters)

    print('UBS_MRS',UBS_MRS)
    print(len(UBS_MRS))
    if len(UBS_MRS)!=2:
        sdfggfgd

    bestmat, stats_res = UBS_MRS

    keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[key_material][1])


    print("Merging matrices")
    print("keep_only_equivalent = %s" % keep_only_equivalent)
    bestmat, stats_res = ISS.MergeSortand_RemoveDuplicates(bestmat,
                                                    stats_res,
                                                    Nb_criterium,
                                                    tol=0.0001,
                                                    keep_only_equivalent=keep_only_equivalent)

    print("stats_res", stats_res)
    nb_sol = len(bestmat)

    bestmatList.append(bestmat)
    stats_resList.append(stats_res)

def flatnestedlist(list_of_lists):
    """can be found in graingraph.py

    """
    return [y for x in list_of_lists for y in x]

BestMatrices = flatnestedlist(bestmatList)
BestStats = flatnestedlist(stats_resList)

print('nb solutions', len(BestMatrices))

BestMatrices, BestStats = ISS.MergeSortand_RemoveDuplicates(
        BestMatrices,
        BestStats,
        Nb_criterium,
        tol=0.0001,
        keep_only_equivalent=keep_only_equivalent)

print('final nb solutions', len(BestMatrices))
print('BestMatrices\n')
print(BestMatrices)