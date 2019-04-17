# -*- coding: utf-8 -*-
"""
Class to handle indexing of a set of laue spots

this class belongs to the open-source LaueTools project
JS micha October 2014
"""

import copy
import pickle
import os
import time

import numpy as np
from pylab import figure, scatter, show, subplot, title
import multiprocessing
import configparser as CONF

import generaltools as GT
import CrystalParameters as CP
import lauecore as LAUE
import indexingAnglesLUT as INDEX
import dict_LaueTools as DictLT
import FitOrient as FitO
import LaueGeometry as F2TC
import findorient as FO
import indexingImageMatching as IMM
import IOLaueTools as IOLT

import matchingrate

from generaltools import printred, printgreen, printcyan

CST_ENERGYKEV = DictLT.CST_ENERGYKEV

DEG = DictLT.DEG

NORMAL_TO_SAMPLE_AXIS = np.array([-np.sin(40. * DEG), 0, np.cos(40. * DEG)])
MINIMUM_NB_MATCHES_FOR_INDEXING = 6

DEFAULT_KF_DIRECTION = 'Z>0'


#---  ---------  exp. SPOTS indexing management -------
class OrientMatrix():

    """
    class of orientation matrix which may have several representation
    TODO: need to add some properties (det = 1, equivalent matrix ...)
    """
    def __init__(self, matrix=None, eulers=None):
        if matrix is not None:
            self.matrix = matrix
            self.eulers = None
            if np.array(matrix).shape!=(3,3):
                raise ValueError("the following object is not a 3x3 matrix !").with_traceback(matrix) 
        elif eulers is not None:
            self.eulers = eulers
            self.matrix = GT.fromEULERangles_toMatrix(eulers)


class spotsset():
    """
    class for exp. SPOTS indexing management
    """
    def __init__(self):
        """
        instantiate the class
        no argument
        """
        # dict of spots properties (last element is a flag =1 if spot belong to indexed (completed) grain, 0 else)
        self.indexed_spots_dict = {}

        # dicts of completed or current indexation state (key is grain index)
        self.dict_grain_matrix = {}
        self.dict_grain_devstrain = {}
        self.dict_grain_Ts = {}
        self.dict_grain_matching_rate = {}
        self.dict_Missing_Reflections = {}
        self.dict_grain_twins = {}
        

        # list of indices of grains that are already indexed (indexation considered as completed)
        self.indexedgrains = []
        # material of grains that are already indexed
        self.dict_indexedgrains_material = {}
        # stack of matrices whose matching rate with exp spots data will be computed
        self.UBStack = []

        self.detectorparameters = None
        self.pixelsize = None
        self.dim = None
        self.kf_direction = None
        self.detectordiameter = None
        self.CCDLabel = None

        # total number of exp. spots
        self.nbspots = None
        
        self.filename = None
        self.dict_IMM = None

        # current spots data for indexation and refinement
        self.TwiceTheta_Chi_Int = None # two theta and chi scattering angles and intensities array
        self.absolute_index = None # absolute experimental spot index in the input spot list (.cor)

#         self.updateSimulParameters()

    def setSimulParameters(self, key_material, emin, emax, detectorparameters, pixelsize, dim):
        """
        set simulation parameters
        TODO: should be written in a more elegant way

        :param key_material: element or material key
        :type key_material: string
        :param emin: minimum energy bandpass
        :type emin: integer
        :param emax: maximum energy bandpass
        :type emax: integer
        :param detectorparameters: 5 elements
        :type detectorparameters: list, tuple, array of floats
        :param pixelsize: pixel size (mm) if squared pixel
        :type pixelsize: float
        :param dim: CCD nb of pixels 2 elements (dim1,dim2)
        :type dim: list, tuple, array of integers
        """
        self.key_material = key_material
        self.emin = emin
        self.emax = emax
        self.detectorparameters = detectorparameters
        self.pixelsize = pixelsize
        self.dim = dim

        self.updateSimulParameters()

    def setImageMatchingParameters(self, dict_IMM, database):
        """
        set parameters for image matching technique

        :param dict_IMM: dictionary of imagematching parameters
        :param database: database of templates
        """
        self.dict_IMM = dict_IMM
        self.IMMdatabase = database

    def setMaterial(self, key_material):
        """
        set material for indexing
        :param key_material: material or element key (see dict_Lauetools)
        :type key_material: string
        """
        self.key_material = key_material

    def setEnergyBand(self, emin, emax):
        """
        set energy band limits
        """
        self.emin = emin
        self.emax = emax
        self.updateSimulParameters()

    def updateSimulParameters(self):
        """
        update dictionary of simulation  parameter

        TODO: useful ?
        """
        params = ('key_material', 'emin', 'emax',
                  'detectorparameters', 'pixelsize',
                  'dim', 'kf_direction',
                  'detectordiameter')
        try:
            values = (self.key_material,
                      self.emin, self.emax,
                      self.detectorparameters,
                      self.pixelsize, self.dim,
                      self.kf_direction,
                      self.detectordiameter)
            self.simulparameter = dict(list(zip(params, values)))
        except AttributeError as emsg:
            print("in updateSimulParameters(): missing parameter in simulparameter")
            print(" %s" % emsg)
            return

    def getSimulParameters(self):
        """
        get the current simulation parameters
        """
        self.updateSimulParameters()

        for key in ('key_material', 'emin', 'emax', 'detectorparameters', 'pixelsize',
                    'dim', 'kf_direction', 'detectordiameter'):
            print("%s   : %s" % (key, self.simulparameter[key]))
        return self.simulparameter

    def importdata(self, exp_data):
        """
        initialize spots indexation dictionary from exp_data
        exp_data : array of 5 elements : tth, chi, Intensity, posX, posY
        
        NB: tth, chi are the twotheta and chi scattering angles.
        They must correspond to posX and poxY (pixel position on detector) through calibration
        """
        self.indexed_spots_dict = initIndexationDict(exp_data)

    def purgedata(self, twicethetaChi_to_remove, dist_tolerance=0.2):
        """
        purge experimental spots (already set in spot dictionary)
        of those present in twicethetaChi_to_remove
        
        twicethetaChi_to_remove : array of two elements [tth, chi]
        """
        exp_data = self.getSpotsallData()[:, 1:]
        print("Before purge len(exp_data)", exp_data.shape)
        self.indexed_spots_dict = purgeSpotsinDict(exp_data.T,
                                                   twicethetaChi_to_remove,
                                                   dist_tolerance=dist_tolerance)
        # update the nb of spots
        self.nbspots = len(self.getSpotsallData()[:, 0])

        print("after purge self.nbspots", self.nbspots)

    def importdatafromfile(self, filename,
                           removeSpots=None,
                           sortSpots_from_refenceList=None):
        """
        initialize spots indexation dictionary from data in file
        
        # TODO: read and set calibration parameters in this function!
        """
        (data_theta, Chi,
         posx, posy,
         dataintensity,
         detectorparameters,
         CCDcalibdict) = IOLT.readfile_cor(filename, output_CCDparamsdict=True)[1:]
                  
        if isinstance(data_theta, (float, int)):
            nb_spots = 1
            print("%s contains a single peak" % filename)
            self.nbspots = nb_spots
            return False
         
        if sortSpots_from_refenceList not in (None, 'None'):
            import spottracking as SpTra
            (data_theta, Chi,
            posx, posy,
            dataintensity,
            isolatedspots,
            isolatedspots_ref) = SpTra.sortSpotsDataCor(data_theta, Chi, posx, posy, dataintensity,
                                                  sortSpots_from_refenceList)
            
            print("isolatedspots", isolatedspots)
            print("isolatedspots_ref", isolatedspots_ref)

        nb_spots = len(data_theta)

        print("CCDcalibdict", CCDcalibdict)

        Twicetheta = 2. * data_theta

        self.importdata([Twicetheta, Chi, dataintensity, posx, posy])

        self.detectorparameters = detectorparameters
        self.nbspots = nb_spots
        self.filename = filename

        return True

#         self.updateSimulParameters()

    def getSpotsallData(self):
        """
        get all data of experimental spots
        array where columns are:
        absolute spot index, tth, chi, posX, posY, intensity
        """
        data = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            index, tth, chi, posX, posY, intensity = self.indexed_spots_dict[key_spot][:6]
            data.append([index, tth, chi, posX, posY, intensity])

        self.alldata = np.array(data)
        return self.alldata

    def getSpotsExpData(self, selectedspots_index=None):
        """
        return useful data of all experimental spots in 3 rows
        i.e. 2theta, chi, intensity

        selectedspots_index  : None, return all data
                                : list of indices, return corresponding data
        """
        exp_data = np.take(self.getSpotsallData(), [1, 2, 5], axis=1)

        if selectedspots_index is not None:
#            print "selectedspots_index in getSpotsExpData", selectedspots_index
            return (exp_data[selectedspots_index]).T
        else:
            # 2theta, chi , intensity
            return exp_data.T

    def getSpotsFamily(self, grain_index):
        """
        return spots that belong to the same grain
        """
        spots_set = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # spot has been indexed
            if self.indexed_spots_dict[key_spot][-1] == 1:
                if self.indexed_spots_dict[key_spot][-2] == grain_index:
                    spots_set.append(key_spot)

        return spots_set

    def getUnIndexedSpots(self):
        """
        read dictionary of spots and
        return spots indices for which indexation has not been completed
        """
        unindexed_spots_indices = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            if self.indexed_spots_dict[key_spot][-1] == 0:
                unindexed_spots_indices.append(key_spot)
            elif self.indexed_spots_dict[key_spot][-1] == 1:
                if self.indexed_spots_dict[key_spot][-2] not in self.indexedgrains:
                    unindexed_spots_indices.append(key_spot)

        return unindexed_spots_indices

    def getSpotsFamilyallData(self, grain_index, onlywithMiller=1):
        """
        return all data of experimental spots for one grain
        """
        data = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # last element is a flag. flag=1 implies spot has been indexed
            if self.indexed_spots_dict[key_spot][-1] == 1:
                # spot belong to the grain grain_index
                if self.indexed_spots_dict[key_spot][-2] == grain_index:
                    (index,
                     tth, chi,
                     posX, posY,
                     intensity, Miller, Energy) = self.indexed_spots_dict[key_spot][:8]
                    # in ambiguous case, spot has not been assigned Miller indices
                    if onlywithMiller == 1:
                        if Miller is not None:
                            H, K, L = Miller
                            data.append([index, tth, chi, posX, posY, intensity,
                                         H, K, L, Energy])

        return np.array(data)

    def getSummaryallData(self):
        """
        return all data of experimental spots
        """
        data = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # spot has been indexed
            if self.indexed_spots_dict[key_spot][-1] == 1:
                (spotindex,
                     tth, chi,
                     posX, posY,
                     intensity, Miller, Energy) = self.indexed_spots_dict[key_spot][:8]

                grain_index = self.indexed_spots_dict[key_spot][-2]
                if Miller is not None:
                    H, K, L = Miller
                    data.append([spotindex, grain_index, tth, chi, posX, posY, intensity,
                                         H, K, L, Energy])
                else:
                    # experimental spot is likely belonging to one grain
                    # 10000 is a flag meanwhile waiting for other structure
                    data.append([spotindex, grain_index, tth, chi, posX, posY, intensity,
                                         10000, 10000, 10000, 10000])
            # spot has not been indexed
            else:
                (spotindex,
                     tth, chi,
                     posX, posY,
                     intensity) = self.indexed_spots_dict[key_spot][:6]
                data.append([spotindex, -1, tth, chi, posX, posY, intensity,
                                         10000, 10000, 10000, 10000])

        return np.array(data)

    def getallIndexedSpotsallData(self, onlywithMiller=1):
        """
        return all data of experimental spots for all indexed grains
        """
        DataGraindict = {}
        for grainindex in self.indexedgrains:
            datagrain = self.getSpotsFamilyallData(grainindex, onlywithMiller=1)
            DataGraindict[grainindex] = datagrain

        return DataGraindict

    def getSpotsFamilyExpData(self, grain_index):
        """
        return useful data of experimental spots for one grain
        """
        alldata = self.getSpotsFamilyallData(grain_index, onlywithMiller=1)

#        print 'alldata', alldata
#        print 'alldata', type(alldata)
        if len(alldata) == 0:
            return None
        else:
            ind, tth, chi = (alldata[:, :3]).T
            intensity = alldata[:, 5]

            return np.array([ind, tth, chi, intensity]).T

    def getUnIndexedSpotsallData(self, exceptgrains=None):
        """
        return all data of unindexed experimental spots
        and those already indexed but not from grains of index in exceptgrains 

        exceptgrains    : list of integers or integer

        return:
        array whose columns are: absolute spot index, tth, chi, posX, posY, intensity
        """

        #if exceptgrains is integer
        if not isinstance(exceptgrains, list):
            exceptgrains = [exceptgrains]
            
        data = []

        for key_spot in sorted(self.indexed_spots_dict.keys()):
            grain_origin = isindexed(key_spot, self.indexed_spots_dict)
            # spot already indexed, guessed to be indexed or being indexed
            if isinstance(grain_origin, int):
                if grain_origin not in exceptgrains:
                    index, tth, chi, posX, posY, intensity = self.indexed_spots_dict[key_spot][:6]
                    data.append([index, tth, chi, posX, posY, intensity])
            # spot not indexed at all
            else:
                index, tth, chi, posX, posY, intensity = self.indexed_spots_dict[key_spot][:6]
                data.append([index, tth, chi, posX, posY, intensity])

        return np.array(data)
    
    def getSpotsFromSpotIndices(self, spotindices):
        """
        return all data of unindexed experimental spots from their index
        and those already indexed but not from grains of index in exceptgrains 

        spotindices    : list of integers or integer

        return:
        array whose columns are: absolute spot index, tth, chi, posX, posY, intensity
        """
       
        data = []
        
        print("self.indexed_spots_dict",self.indexed_spots_dict)
        
        

        for spot_index  in spotindices:
            
            print("self.indexed_spots_dict[spot_index]",self.indexed_spots_dict[spot_index])

            index, tth, chi, posX, posY, intensity = self.indexed_spots_dict[spot_index][:6]
            data.append([index, tth, chi, posX, posY, intensity])
            
        toindexdata = np.array(data)
            
        absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
        intensity_data = toindexdata[:, 5]
        absoluteindex = np.array(absoluteindex, dtype=np.int)

        return np.array([twicetheta_data,
                         chi_data,
                         intensity_data]), absoluteindex
            
    def getSelectedExpSpotsData(self, exceptgrains=None):
        """
        return Exp. data spots (scattering angles, intensity and spot index)  of unindexed spots and
        indexed spots that grain index does  not appear in exceptgrains

        exceptgrains    : list of integers or integer of grain index

        return:
        [0] array of 3 elements: twicetheta_data, chi_data, intensity_data
        [1] array of integer of corresponding spot index
        """
        # extract data not yet indexed or temporarly indexed
        toindexdata = self.getUnIndexedSpotsallData(exceptgrains=exceptgrains)
        
        print("toindexdata",toindexdata)
        print(toindexdata.shape)
        
        absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
        intensity_data = toindexdata[:, 5]
        absoluteindex = np.array(absoluteindex, dtype=np.int)

        return np.array([twicetheta_data,
                         chi_data,
                         intensity_data]), absoluteindex

    def setSelectedExpSpotsData(self, exceptgrains=None, fromspotsindices=None):
        """
        select set of spots to be refined (set scattering angles and spots indices) from unindexed spots and
        selected indexed spots whose grain index does not appear in exceptgrains

        if  exceptgrains = None        : set selected spots from all remaining
                                        spots that do not belong to
                                        an already indexed grain
                            None is equivalent to a very lonf list of integers

        exceptgrains    : list of integers or integer
        """
        if fromspotsindices is not None:
            (self.TwiceTheta_Chi_Int,
             self.absolute_index) = self.getSpotsFromSpotIndices(fromspotsindices)
        else:
            (self.TwiceTheta_Chi_Int,
             self.absolute_index) = self.getSelectedExpSpotsData(exceptgrains=exceptgrains)

    def resetSpotsFamily(self, grain_index):
        """
        reset spots properties in self.indexed_spots_dict for spots belonging to the grain of index 'grain_index'
        """
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # spot has been indexed
            if self.indexed_spots_dict[key_spot][-1] == 1:
                if self.indexed_spots_dict[key_spot][-2] == grain_index:
                    self.indexed_spots_dict[key_spot] = self.indexed_spots_dict[key_spot][:6] + [0]
                    

    def AssignHKL(self, Orientation, grain_index,
                      AngleTol=1., use_spots_in_currentselection=True,
                      selectbyspotsindices = None,
                      verbose=1):
        """
        Assign hkl to the exp spot data set according to the orientation matrix within the tolerance angle
        
        for the given grain (grain_index):
        update corresponding spots properties by calling  self.updateIndexationDict
        update also dictionnaries of matching figure:
            self.dict_grain_matrix[grain_index] = matrix
            self.dict_grain_matching_rate[grain_index] = [nb_updates, matching_rate]
            self.dict_Missing_Reflections[grain_index] = missingRefs

        input:
        Orientation             : OrientMatrix object or array of 3 euler angles
        grain_index                : grain index to label exp. spots
        AngleTol (deg)        : angular tolerance to accept a link between exp. and theo. spots
        use_spots_in_currentselection : False, consider 
        
        output:
        matching_rate, nb_updates, missingRefs
        """
        MINIMUM_SPOTS = 8
        
        if len(self.indexed_spots_dict) < MINIMUM_SPOTS:
            print("too few spots to index in spotsset object")
            return None, None, None

        if isinstance(Orientation, OrientMatrix):
            print("True it is an OrientMatrix object")
            matrix = Orientation.matrix
            print("Orientation",Orientation)
            print("matrix",matrix)
            eulers = None
        else:
            matrix = GT.fromEULERangles_toMatrix(Orientation)
            eulers = Orientation

        # use predefined selection of spots that are
        # already contained in self.TwiceTheta_Chi_Int, self.absolute_index
        if use_spots_in_currentselection is False:

            # exp data used from refined model
            data_1grain = self.getSpotsFamilyallData(grain_index, onlywithMiller=1)
            index_r, tth_r, chi_r, posX, posY, intensity_r, H, K, L, Energy = data_1grain.T

            self.TwiceTheta_Chi_Int = [tth_r, chi_r, intensity_r]

            useabsoluteindex = index_r
            selected_expdata = self.TwiceTheta_Chi_Int

        # use general selection of exp spots according to their indexation state
        # (a spot is selected if it does not belong to a grain (already indexed))
        elif use_spots_in_currentselection is True:
            # all remaining exp spots not already indexed
            # this sets: self.TwiceTheta_Chi_Int and self.absolute_index
            self.setSelectedExpSpotsData(exceptgrains=self.indexedgrains,fromspotsindices=selectbyspotsindices)

            selected_expdata = self.TwiceTheta_Chi_Int
            useabsoluteindex = self.absolute_index
            
        print("***nb of selected spots in AssignHKL*****", len(useabsoluteindex))

        AssignationHKL_res, nbtheospots, missingRefs = self.getSpotsLinks(matrix,
                                                      exp_data=selected_expdata,
                                                       useabsoluteindex=useabsoluteindex,
                                                       removeharmonics=1,  # for fast computations
                                                       ResolutionAngstrom=False,  # simulate exhaustively all the pattern for a perfect theo. crystal
                                                       veryclose_angletol=AngleTol,
                                                       verbose=verbose)
        # TODO nb of links with getSpotsLinks() larger than nb of links used in previous refinement
        # self.pixelresidues
#         print "AssignationHKL_res in AssignHKL", AssignationHKL_res

        if 0:  # verbose:
            print("missingRefs", missingRefs)

        if AssignationHKL_res is not None:
            nb_updates = self.updateIndexationDict(AssignationHKL_res, grain_index,
                                                              overwrite=1)

            matching_rate = 100. * nb_updates / nbtheospots

            if verbose:
                print("\n")
                if eulers is not None:
                    print("for this three euler angles [%.1f,%.1f,%.1f]" % tuple(Orientation))
                print(matrix)
                print("nb indexed spots %d / %d (theo. nb)" % (nb_updates, nbtheospots))
                print("matching rate  : %.1f" % matching_rate)
                print("with tolerance angle : %.2f deg" % AngleTol)

            # update dictionaries
            self.dict_grain_matrix[grain_index] = matrix
            self.dict_grain_matching_rate[grain_index] = [nb_updates, matching_rate]
            self.dict_Missing_Reflections[grain_index] = missingRefs

#            print "Low value of indexed spots may mean that"
#            print "spots have been already assigned to other grain"

            return matching_rate, nb_updates, missingRefs

        else:
            print("no grains have been found !")
            return None, None, None

    def FindOrientMatrices(self, spot_index_central=[0, 1, 2],
                           nbmax_probed=10,
                           nLUT=3, LUT=None,
                           set_central_spots_hkl=None,
                           ResolutionAngstrom=False,
                           AngTol_LUTmatching=0.5,
                           MatchingRate_Angle_Tol=0.2,
                           Minimum_Nb_Matches=15,
                           verbose=0,
                           nb_of_solutions_per_central_spot=1,
                           simulparameters=None):
        """
        Find orientation matrices by angles recognition (look up table of angles in reference structure)
        
        consider exp. spots that are unindexed  (call of getUnIndexedSpots())
        
        call of INDEX.getOrientMatrices in spotsset class
        
        USED in FileSeries
        """
        emax = self.emax

        key_material = self.key_material

        if simulparameters is None:
            simulparameters = {}
            simulparameters['kf_direction'] = self.kf_direction
            print("in FindOrientMatrices using kf_direction : %s" % str(self.kf_direction))

        Tabledist = None

#         exceptgrains = self.indexedgrains
#         if exceptgrains in ('None', []):
#             exceptgrains = None
        # absolute spot index, tth, chi, posX, posY, intensity
        spot_to_index = self.getUnIndexedSpots()

        allexpdata = self.getSpotsallData()

        data_to_index = np.take(allexpdata, spot_to_index, axis=0)

#         print "data_to_index", data_to_index

        (abs_spotindex,
         tth, chi,
         posX, posY, intensity) = data_to_index.T

        nb_of_spots = len(abs_spotindex)

        # list of absolute index in spots list (.cor file) to be indexed
        current_exp_spot_index_list = np.arange(nb_of_spots)
        # absoluteindex = current_exp_spot_index_list[k]
        # nb of experimental spots that will be used to calculate the matching rate
        nbspotmaxformatching = nb_of_spots

        # there is no precomputed angular distances between spots
        if not Tabledist:
            # select 1rstly spots that have not been indexed and 2ndly reduced list by user
            index_to_select = np.take(current_exp_spot_index_list,
                                      np.arange(nbspotmaxformatching))

            select_theta = 0.5 * tth[index_to_select]
            select_chi = chi[index_to_select]
            select_I = intensity[index_to_select]
            # print select_theta
            # print select_chi
            select_thetachi = np.array([select_theta, select_chi]).T
            # compute angles between spots
            Tabledistance = INDEX.calculdist_from_thetachi(select_thetachi,
                                                           select_thetachi)

        latticeparams = DictLT.dict_Materials[key_material][1]
        B = CP.calc_B_RR(latticeparams)

        # indexation procedure
        bestmat, stats_res = INDEX.getOrientMatrices(
                            spot_index_central,
                            emax,
                            Tabledistance[:nbmax_probed, :nbmax_probed],
                            select_theta, select_chi,
                            n=nLUT,
                            B=B,
                            cubicSymmetry=CP.isCubic(latticeparams),
                            ResolutionAngstrom=ResolutionAngstrom,
                            LUT=LUT,
                            LUT_tol_angle=AngTol_LUTmatching,
                            MR_tol_angle=MatchingRate_Angle_Tol,
                            Minimum_Nb_Matches=Minimum_Nb_Matches,
                            plot=0,
                            key_material=key_material,
                            nbbestplot=nb_of_solutions_per_central_spot,  # nb of solutions per central spot
                            verbose=0,
                            addMatrix=None,  # To add a priori good candidates...
                            set_central_spots_hkl=set_central_spots_hkl,
                            detectorparameters=simulparameters,
                            verbosedetails=False  # not CP.isCubic(key_material)
                            )
        # when nbbestplot is very high  self.bestmat contain all matrices
        # with matching rate above Minimum_Nb_Matches

#         print "bestmat, stats_res", bestmat, stats_res

        PrintMatchingResults(bestmat, stats_res)

        nb_sol = len(bestmat)

        print("Number of matrices found (nb_sol): ", nb_sol)

        keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[key_material][1])

        print("set_central_spots_hkl in FindOrientMatrices", set_central_spots_hkl)
        # TODO: anticipate  future pb if set_central_spots_hkl is a list of hkl and None...
        if set_central_spots_hkl not in (None, [None]):
            keep_only_equivalent = False

        if nb_sol > 1:
            print("Merging matrices")
            print("keep_only_equivalent = %s" % keep_only_equivalent)
            bestmat, stats_res = MergeSortand_RemoveDuplicates(bestmat,
                                               stats_res,
                                               Minimum_Nb_Matches,
                                               tol=0.0001,
                                               keep_only_equivalent=keep_only_equivalent)

#         print "stats_res", stats_res

        PrintMatchingResults(bestmat, stats_res)
        nb_sol = len(bestmat)
#         print "Max. Number of Solutions", NBRP

        if nb_sol == 0:
            return [], []

        print("Nb of potential orientation matrice(s) UB found: %d " % nb_sol)
        print(bestmat)
        stats_properformat = []
        for elem in stats_res:
            elem[0] = int(elem[0])
            elem[1] = int(elem[1])
            stats_properformat.append(tuple(elem))

        return bestmat, stats_properformat

    def IndexSpotsSet(self, file_to_index,
                      key_material, emin, emax,
                      dict_parameters, database,
                      starting_grainindex=0,
                    use_file=1,
                    verbose=0, plotintermediateresults=0,
                    IMM=True, nbGrainstoFind='max',
                    LUT=None, n_LUT=3,
                    set_central_spots_hkl=None,
                    ResolutionAngstrom=False,
                      MatchingRate_List=[50, 60, 80],
                      angletol_list=[0.5, 0.2, 0.1],
                      checkSigma3=False,
                      previousResults=None,
                      CheckOrientations=None,
                      corfilename=None,
                      dirnameout_fitfile=None):
        """
        General procedure to index and a set of experimental spots with one grain.
        
        Guessed matrices can checked prior to proceed to indexation from scratch (previous Results)
        
        potential orientation matrix is eligible to refinement depending on matching rate
        
        Successive steps of refinement procedures are set by  angletol_list and MatchingRate_List

        :param file_to_index: filename of experimental spots list
        :type file_to_index: string
        :param key_material: element or material key
        :type key_material: string
        :param emin: minimum energy bandpass
        :type emin: integer
        :param emax: maximum energy bandpass
        :type emax: integer
        :param dict_parameters: dictionary of parameters for the loop and spots set size
        :type dict_parameters: dictionary
        :param database: if not None, big array of templates for image matching indextion
        :type database: array
        :param starting_grainindex: grain index to start indexation with
        :type starting_grainindex: integer
        :param use_file: 1 import initial spot set from file and initialize self.indexed_spots_dict
                        0 start with current self.indexed_spots_dict
        :type use_file: flag
        :param verbose: print out some infos during procedure
        :type verbose: flag
        :param plotintermediateresults: show plot of spots in Laue Pattern that are indexed
        :type plotintermediateresults: flag
        :param IMM: use imege template matching 0 or 1
        :type IMM: flag
        :param nbGrainstoFind: number of grain to find for a given material . If 'max' then find the largest number of grains in data set.
        :type nbGrainstoFind: integer
        :param MatchingRate_List: list of minimum matching rates that accept to perform the next refinement step (with less tolerance angle)
                            Last term is the final minimum matching rate (in percent) to accept the final refinement.
        :type MatchingRate_List: list of floats
        :param angletol_list: List of the maximum residual angular tolerances after each step of refinement.
        :type angletol_list: list of floats
        :param checkSigma3: flag to test if some twins sigma 3 may be present in data set (in devpt).
        :type checkSigma3: flag
        :param previousResults: if not None, 3 elements helping to use previous indexation results as guesses for the current indexation
                            addMatrix, previousMatchingRate, previousNbspots:
                            initial guess of matrix, previous matching rate, previous Nb of indexed spots
        :type previousResults: None or tuple of 3*3 array (matrix), float, integer
        :param corfilename: peak list with .cor extension
        :type corfilename: string
        :param dirnameout_fitfile: folder to write .fit results file in
        :type dirnameout_fitfile: string
        """
        MINIMUM_NB_SPOTS_FOR_INDEXING = 3

        if use_file:
            # read spots data and init dictionary of indexed spots
            self.importdatafromfile(file_to_index)

        totalnbspots = self.nbspots

        self.setMaterial(key_material)
        self.setEnergyBand(emin, emax)

        # for image matching technique to provide UB orientation matrices candidates
        self.setImageMatchingParameters(dict_parameters, database)
        MatchingRate_Threshold_IMM = 30

        print("self.pixelsize in IndexSpotsSet", self.pixelsize)

        self.ResolutionAngstromLUT = False
        print("ResolutionAngstromLUT in IndexSpotsSet", ResolutionAngstrom)

        self.LUT = LUT
        self.n_LUT = n_LUT

        # list of missing reflection index
        self.MissingRefindexedgrains = []

        # initial tolerance angle for matching with a raw (unrefined) provided matrix
        AngleTol_0 = 0.5
        # tolerance angle sequence for regular serial refining
        ANGLETOL_List = angletol_list
        # Minimum Matching rate (in percent) sequence
        # to accept indexation and further orientation refinement
        # MatchingRate_List = [50, 60, 80]

        PLOTRESULTS = plotintermediateresults
        VERBOSE = verbose

        #initialisation
        grain_index = starting_grainindex  # current grain to be indexed
        NeedtoProvideNewMatrices = True
        indexgraincompleted = False
        self.nbMatricesInUBStack = 0
        frompreviousResults = False
        
        # default minimum matching rate to start refinement steps
        # (normally very small to accept a UB solution coming from self.UBstack
        # if self.UBstack is feeded by previous results or checkorientation parameters (.ubs file)
        # MATCHINGRATE_FOR_PREVIOUSRESULTS is reset to the user value
        MATCHINGRATE_FOR_PREVIOUSRESULTS = 1.
        
        # matrices that must be tested first before trying to index from scratch
        # these matrices can come from previously indexed Laue Pattern
        if previousResults is not None:
            
            nbAddMatrices, addMatrices, previousMatchingRate, previousNbspots = previousResults
            print("considering previousResults with addMatrix", addMatrices)
            previousMatricesList=[]
            for addMatrix in addMatrices:
                previousmatrix = OrientMatrix(matrix=addMatrix)
                previousMatricesList.append(previousmatrix)

            self.UBStack = previousMatricesList
            NeedtoProvideNewMatrices = False
            fromIMM = IMM
            frompreviousResults = True
            
            print("self.UBStack before while(1) loop")
            print(self.UBStack)
            self.nbMatricesInUBStack = len(self.UBStack)
            
            # minimum matching rate (nb matched spots/ nb total simulated) for previous matrix at the last step of refinement (minimal tolerance angle)
            # if the number of theoretical reflections is high, this parameter must be lowered significantly
            # TODO: add this parameter to .irp file better than in Index_Refine.py GUI for instance
            MATCHINGRATE_FOR_PREVIOUSRESULTS = dict_parameters['MinimumMatchingRate']  # 50.
            print("MATCHINGRATE_FOR_PREVIOUSRESULTS",MATCHINGRATE_FOR_PREVIOUSRESULTS)
        
        # matrices that must be tested first before trying to index from scratch
        # these matrices are proposed by a specific file .ubs
        if CheckOrientations is not None:
            print("CheckOrientations is not None")
            print("CheckOrientations",CheckOrientations)
            CheckOrientation = CheckOrientations[0]
            Checkmatrices=CheckOrientation[5]
            self.key_material = CheckOrientation[2]
            self.emax = CheckOrientation[3]
            MATCHINGRATE_FOR_PREVIOUSRESULTS = CheckOrientation[4]
            
            self.UBStack=[]
            for chckmatrix in Checkmatrices:
                matrixobj = OrientMatrix(matrix=chckmatrix)
                self.UBStack.append(matrixobj)
                
            self.nbMatricesInUBStack = len(self.UBStack)
            
            print("MATCHINGRATE_FOR_PREVIOUSRESULTS",MATCHINGRATE_FOR_PREVIOUSRESULTS)
            
            fromIMM = IMM

        if nbGrainstoFind != 'max' and not isinstance(nbGrainstoFind, int):
            raise ValueError("nbGrainstoFind in IndexSpotsSet() method has a wrong value")



        # flag to use or not use intensity weights when refining model
        self.UseIntensityWeights = dict_parameters['UseIntensityWeights']
        
        self.nbSpotsToIndex =  dict_parameters['nbSpotsToIndex']

        # minimum nb of matches for a given material to be selected
        # as potential solution for anglesLUT indexing method
        self.MinimumNumberMatches = dict_parameters['MinimumNumberMatches']


        nbgrains_found = 0
        #---------------------------------------------------------------
        # infinite loop to try raw indexing with matrices in self.UBstack
        # with material and nb of grains requested and a set of peaks
        #-----------------------------------------------------------
        while 1:
            ProceedWithRefinement=False
            nb_remaining_spots = len(self.getUnIndexedSpots())
#            print "Number of grains already indexed %d" % len(self.indexedgrains)
#            print "Number of grains already indexed", self.indexedgrains

            print("\n Remaining nb of spots to index for grain #%d : %d\n" % (grain_index,nb_remaining_spots))

            # exit the loop if too few spots
            # or when user requests to index a few number of grain
            if nb_remaining_spots <= MINIMUM_NB_SPOTS_FOR_INDEXING or \
                (isinstance(nbGrainstoFind, int) and \
                 nbgrains_found == nbGrainstoFind):
                print("%d spots have been indexed over %d" % \
                        (totalnbspots - nb_remaining_spots, totalnbspots))
                print("indexing rate is --- : %.1f percents" % \
                        (100. * (totalnbspots - nb_remaining_spots) / totalnbspots))
                print("indexation of %s is completed" % self.filename)
                if isinstance(nbGrainstoFind, int):
                    print("for the %d grain(s) that has(ve) been indexed as requested" % \
                                nbGrainstoFind)
                    print("Leaving Index and Refine procedures...")
                break

            print("\n ******")
            print("start to index grain #%d of Material: %s \n" % \
                    (grain_index, key_material))
            print("******\n")

            
            # now there should be matrices from previous results
            # or guessed by indexation methods.
            # the flag MatchingRateUBStackTooLow comes from bad refined results inside the following loop
            MatchingRateUBStackTooLow=True
            # condition is False if self.UBStack empty OR MatchingRateUBStackTooLow is False
            while ((self.UBStack is not None) or (self.UBStack is not [None])) and self.UBStack and \
                    (self.nbMatricesInUBStack > 0) and \
                    MatchingRateUBStackTooLow is True:
                
                print("\n ---- %d Matrice(s) are candidates in the Matrix Stack ! ----" % \
                                len(self.UBStack))
                                
#                 print "self.UBStack", self.UBStack
                if VERBOSE:
                    print("\n  -----   Taking a new matrix from the matrices stack  -------")

                UB = self.UBStack.pop(0)
                
                # add a matrix in dictionary
                print('\nFor grain_index', grain_index)
    #             print "UB", UB
                print("\nConsidering UB", UB.matrix)
    #             print "dict_grain_matrix"
    #             print self.dict_grain_matrix
    #             self.dict_grain_matrix[grain_index] = UB.matrix
    
        #        print "dict_grain_matrix before initial indexing with a raw orientation"
        #        print self.dict_grain_matrix
    
                # indexation of spots with raw (un refined) matrices & #update dictionaries
                print("\n---------------checking matching rate for grain  #%d-----------------------" % grain_index)
                if VERBOSE:
                    print("eulerangles", UB.eulers)
                    print("UB", UB)
                    print("UB.matrix", UB.matrix)
    
                # select data, link spots,
                # update spot dictionary,
                # update matrix dictionary
                MatchRate, nblinks, missRef = self.AssignHKL(UB, grain_index,
                                                                 AngleTol=AngleTol_0,
                                                                use_spots_in_currentselection=True,
                                                                 verbose=VERBOSE)
                
#                 print "MatchRate,nblinks",MatchRate,nblinks
                
                if MatchRate>MATCHINGRATE_FOR_PREVIOUSRESULTS:
                    MatchingRateUBStackTooLow=False
                    NeedtoProvideNewMatrices = False
                    ProceedWithRefinement= True
                    bestUB = UB
                    
                    print("Ready to structure refinement now !")
             
#             if NeedtoProvideNewMatrices:
#                 print "matrices stack to test is empty"
#                 print "%d spot(s) have not been indexed" % nb_remaining_spots
#                 self.dict_grain_matching_rate[grain_index] = [0, 0]
#                 self.dict_grain_devstrain[grain_index] = 0
#     #            print "You may need to use other indexing techniques, Angles LUT, cliques help..."
#                 break

            
            #-----------------------------------------------------
            #--- Call of UB matrices from indexing techniques (from scratch
            #-------------------------------------------------------
            if NeedtoProvideNewMatrices == True:
                #----------------------------------------------------
                # potential orientation solutions from template matching
                #----------------------------------------------------
                if IMM:
                    print("providing new set of matrices with ImageMatching template technique")

                    # TODO: test if sigma3 symmetry exist between matrices
                    (bestUB,
                     bestmatchingrates,
                     nbMatchedSpots) = self.getOrients_ImageMatching(MatchingRate_Threshold=MatchingRate_Threshold_IMM,
                                                                exceptgrains=self.indexedgrains,
                                                                verbose=VERBOSE)

                    fromIMM = True
                #---------------------------
                # Using Angles LUT matching technique
                #-----------------------------
                else:
                    # potential orientation solutions from angles LUT matching
                    print("providing new set of matrices Using Angles LUT template matching")
                    (
                     self.TwiceTheta_Chi_Int,
                     self.absolute_index
                     ) = self.getSelectedExpSpotsData(exceptgrains=self.indexedgrains)

                    nbspots = len(self.TwiceTheta_Chi_Int[0])
                    print("nbspots", nbspots)
#                     print "self.TwiceTheta_Chi_Int[0].shape", self.TwiceTheta_Chi_Int.shape

                    # set the matching rate above which loop of LUT matching is aborted
                    # this is highly probable this is a grain
                    MATCHINGRATE_THRESHOLD_IAL = dict_parameters['MATCHINGRATE_THRESHOLD_IAL']  # 60

                    # set the tolerance angle for matching
                    MATCHINGRATE_ANGLE_TOL = dict_parameters['MATCHINGRATE_ANGLE_TOL']  # 0.2

                    # set the number of most intense spot candidate to have a recognisable distance
                    NBMAXPROBED = dict_parameters['NBMAXPROBED']  # 10

                    self.AngTol_LUTmatching = dict_parameters['AngleTolLUT']

                    # set central list spots to compute distance from
                    spot_index_central_list = dict_parameters['central spots indices']  # 10
                    # TODO add key dict_parameters['CENTRAL_SPOTS_LIST']
                    # or
#                     CENTRAL_SPOTS_LIST = dict_parameters['CENTRAL_SPOTS_LIST']

                    NBMAXPROBED = min(nbspots, NBMAXPROBED)
                    print("NBMAXPROBED", NBMAXPROBED)

                    # set mutual angular distance from NBMAXPROBED first spots
                    # TODO: build self.table_angdist but not yet effective
                    self.setTable_Angdist(nbmax=NBMAXPROBED)

                    print("set_central_spots_hkl",set_central_spots_hkl)
                    if set_central_spots_hkl is None and LUT is None:
                        # for angular distance LUT matching
                        self.setAnglesLUTmatchingParameters(LUT=self.LUT,
                                                            n_LUT=self.n_LUT)

#                    print "providing new set of matrices with AnglesLUT matching"

                    if max(spot_index_central_list) >= len(self.absolute_index):
                        print("central list of spots contains spots that do not belong the current list of spots to be indexed")
                        break

                    print("Central set of exp. spotDistances from spot_index_central_list probed")
                    print("self.absolute_index", self.absolute_index)
                    print("spot_index_central_list", spot_index_central_list)
                    print(self.absolute_index[spot_index_central_list])

                    # find single best orientation matrix UB solution
                    (bestUB,
                     bestmatchingrate,
                     nbMatchedSpots,
                     Threshold_reached) = self.get_bestUB_fromAnglesLUT(spot_index_central=spot_index_central_list,
                                          MatchingRate_Threshold=MATCHINGRATE_THRESHOLD_IAL,
                                          MatchingRate_Angle_Tol=MATCHINGRATE_ANGLE_TOL,
                                          nbmax_probed=NBMAXPROBED,
                                          Minimum_Nb_Matches=self.MinimumNumberMatches,
                                          LUT=self.LUT,
                                          set_central_spots_hkl=set_central_spots_hkl,
                                          ResolutionAngstrom=ResolutionAngstrom,
                                          exceptgrains=self.indexedgrains,
                                          verbose=verbose)

                    fromIMM = False

                print("\nWorking with a new stack of orientation matrices")

                if Threshold_reached:
                    print(" MATCHINGRATE_THRESHOLD_IAL= %.1f" % MATCHINGRATE_THRESHOLD_IAL)
                    print("has been reached! Indexing has been stopped")
                    

                else:
                    print("MATCHINGRATE_THRESHOLD_IAL= %.1f" % MATCHINGRATE_THRESHOLD_IAL)
                    print('has not been reached! All potential solutions have been calculated')
                    print('taking the first one only.')
                
                # update (overwrite) candidate orientMatrix object list
                print('bestUB object',bestUB)
                
                if bestUB is None:
                    print("\n #### No matrix available for refinement.####")
                    print("#### End of indexation and refinement for element: %s ####"%key_material)
                    break
                
                self.UBStack = [bestUB]
                self.nbMatricesInUBStack = 1
                ProceedWithRefinement= True

            #----------------------------------------------------------
            #--- ---------   REFINEMENT STEPS
            #----------------------------------------------------------
            if ProceedWithRefinement is False:
                break
            
            print("\n\n---------------refining grain orientation and strain #%d-----------------" % grain_index)
            # loop over refinement steps with different angular matching tolerances
            for step_refinement_index, AngleTol in enumerate(ANGLETOL_List):
                
                print("\n\n refining grain #%d step -----%d\n" % (grain_index, step_refinement_index))
                
                self.dict_grain_matrix[grain_index]=bestUB.matrix
                # select data, link spots,
                # update spot dictionary,
                # update matrix dictionary
                print("bestUB",bestUB)
                self.AssignHKL(bestUB, grain_index,AngleTol=AngleTol,
                                            use_spots_in_currentselection=True,
                                             verbose=VERBOSE)
                
                                
                #--- ------- Refine with old strain model (varying strain at the right of UB)
                refinedMatrix, devstrain = self.refineUBSpotsFamily(grain_index,
                                    self.dict_grain_matrix[grain_index],
                                    getstrain=1,
                                    use_weights=self.UseIntensityWeights,
                                    nbSpotsToIndex=self.nbSpotsToIndex,
                                    verbose=VERBOSE)
                refinedTs = None
                #--- ----------------------------
                
                #--- ------- Refine with strain operator at the lef
                # of pure rotation matrix derived from UB
#                     refinedMatrix, devstrain, refinedTs = self.refineStrainElementsSpotsFamily(grain_index,
#                                         self.dict_grain_matrix[grain_index],
#                                         getstrain=1,
#                                         use_weights=self.UseIntensityWeights,
#                                         verbose=VERBOSE)
                #-------------------------------------


                Matching_rate, nb_updates, missingRefs = None, None, None
                if refinedMatrix is not None:

                    UBrefined = OrientMatrix(matrix=refinedMatrix)
        #            print "UBrefined", UBrefined.matrix
                    self.dict_grain_matrix[grain_index] = UBrefined.matrix
                    self.dict_grain_devstrain[grain_index] = devstrain
                    self.dict_grain_Ts[grain_index] = refinedTs
                    self.refinedTs = refinedTs
                    # select data, link spots, update spot dictionary, update matrix dictionary

                    selectedspots_index = True
                    # last step of refinementa assignHKL
                    if step_refinement_index == len(ANGLETOL_List) - 1:
#                         selectedspots_index = self.getSpotsFamilyallData(grain_index)[:, 0]
                        selectedspots_index = False


                    (Matching_rate,
                     nb_updates,
                     missingRefs) = self.AssignHKL(UBrefined,
                                                grain_index,
                                                AngleTol=AngleTol,
                                                use_spots_in_currentselection=selectedspots_index,
                                                verbose=VERBOSE)

                MINIMUM_MATCHINGRATE = 10.
                MINIMUM_SPOTS_GRAIN = 6
                GoodRefinement = False
                if Matching_rate is not None:
                    GoodRefinement = (Matching_rate > MINIMUM_MATCHINGRATE) or \
                                            (nb_updates >= 6)
                    print("GoodRefinement condition is ",GoodRefinement)
                    print("nb_updates %d compared to 6"%nb_updates)

                # matching rate after refinement and making links is too small 
                if Matching_rate < MatchingRate_List[step_refinement_index] or \
                                    refinedMatrix is None or \
                                    not GoodRefinement:
                    # matching rate too low
                    # then remove previous indexed data and launch again imagematching
                    print("matching rate too low after refining")

                    self.resetSpotsFamily(grain_index)

                    if frompreviousResults:
                        print("Previous matrix does not index this data")
                        print("start the normal indexation")
                        NeedtoProvideNewMatrices = True
                        previousResults = None
                        frompreviousResults = False
                        break

                    # there are at least one indexed grain and this dataset has not been already tested with the matrix stack
                    # so remove the corresponding data and restart the imagematching
                    if nbgrains_found > 0 and indexgraincompleted == True:

                        # exit the 'for' loop of refinement and start with other set of spots
                        NeedtoProvideNewMatrices = True
                        indexgraincompleted = False
                        print("Need to re apply imagematching on purged data")
                        break

                    # no data to remove
                    # keep on trying matrix in matrices stack
                    else:
                        NeedtoProvideNewMatrices = False
                        print("Need to look at the next matrix")
                        break

                # matching rate after refinement and making links is higher than threshold
                # OK let s proceed to evaluate the sucess of indexing and refinement
                else:

                    # if this is the last tolerance step
                    if step_refinement_index == len(ANGLETOL_List) - 1:
                        # addMatrix, previousMatchingRate, previousNbspots = previousResults
                        if previousResults is not None:
#                            if Matching_rate < previousMatchingRate:
                            if Matching_rate < MATCHINGRATE_FOR_PREVIOUSRESULTS:
                                # there could be other grain more important than that previously indexed
                                NeedtoProvideNewMatrices = True
                                previousResults = None

                                print("Previous matrix seems good BUT matching rate is not so high")
                                print("matching rate %f < %f = MATCHINGRATE_FOR_PREVIOUSRESULTS" % \
                                                        (Matching_rate, MATCHINGRATE_FOR_PREVIOUSRESULTS))
                                print("so restart the normal indexation")
                                break
                            else:
                                print("Find a grain already indexed in previous image")


                        if checkSigma3:
                            print("--------- Checking Twins sigma3 --------------------")
                            # this grain is considered now as indexed
                            mothergrain_matrix = self.dict_grain_matrix[grain_index]
                            mothergrain_index = grain_index

                            twins_matrices = get4sigma3Matrices(mothergrain_matrix)

                            # simulate sigma3 twins and check corresponding matching rate
                            nb_of_twins = 0
                            for indextwin, twin_mat in enumerate(twins_matrices):

                                # calculate matching without indexing possible with indexonegrain?
                                twinnedMatrix = OrientMatrix(matrix=twin_mat)
                                Matching_rate, nb_updates, missingRefs = self.AssignHKL(twinnedMatrix,
                                                                                    grain_index,
                                                                                    AngleTol,
                                                                                    verbose=VERBOSE)

#                                print "missingRefs", missingRefs
                                print("Matching_rate, nb_updates, missingRefs %.1f %d %d with sigma3 %d" % (Matching_rate, nb_updates, len(missingRefs[1]), indextwin))
                                GoodRefinement = (Matching_rate > MINIMUM_MATCHINGRATE) or (nb_updates >= 6)
                                self.dict_grain_twins[grain_index] = []
                                if Matching_rate >= MatchingRate_List[step_refinement_index] and GoodRefinement:
                                    self.dict_grain_twins[grain_index].append([twinnedMatrix, Matching_rate, nb_updates])

                                    nb_of_twins += 1

                                print("Found %d Sigma" % nb_of_twins)

                            self.AssignHKL(mothergrain_matrix,
                                               grain_index,
                                                AngleTol,
                                                verbose=VERBOSE)
                            NeedtoProvideNewMatrices = not fromIMM
                            self.indexedgrains.append(grain_index)
                            self.dict_indexedgrains_material[grain_index] = key_material
                            indexgraincompleted = True

                            grain_index += 1
                            nbgrains_found += 1

                        # normal and usual way (no common peaks of two laue patterns of 2 distinct grains
                        elif not checkSigma3:

                            if 1:
                                print("\n---------------------------------------------")
                                print("indexing completed for grain #%d with matching rate %.2f " % \
                                        (grain_index, Matching_rate))
                                print("---------------------------------------------\n")
                            # this grain is considered now as indexed
                            # corresponding spots will be no more used for next indexation
                            self.indexedgrains.append(grain_index)

                            # find single representation of UB and reset h,k,l accordingly
                            if set_central_spots_hkl is None and CP.hasCubicSymmetry(self.key_material):

#                                 matrix = self.dict_grain_matrix[grain_index]
                                matrix = self.refinedUBmatrix
                                UBsingle, transfmat = FO.find_lowest_Euler_Angles_matrix(matrix)

                                #update matrix
                                self.dict_grain_matrix[grain_index] = UBsingle

                                (index,
                                tth, chi,
                                posX, posY,
                                intensity,
                                H, K, L,
                                Energy) = self.getSpotsFamilyallData(grain_index, onlywithMiller=1).T

                                hkl = np.array([H, K, L]).T
                                hklmin = np.dot(transfmat, hkl.T).T

                                for kspot, exp_spot_index in enumerate(index):

                                    self.indexed_spots_dict[exp_spot_index][6] = hklmin[kspot]
#
                                print("hkl", hkl)
                                print("new hkl (min euler angles)", hklmin)
                                print("UB before", matrix)
                                print("new UB (min euler angles)", UBsingle)


                            # write fit file for one grain
                            print("writing fit file -------------------------")
                            print("for grainindex=",grain_index)
                            print("self.dict_grain_matrix[grain_index]", self.dict_grain_matrix[grain_index])
                            print("self.refinedUBmatrix", self.refinedUBmatrix)
                            
                            # Update strain (if lower euler angles transform x have been applied on UBmatri)
                            initial_latticeparameters = DictLT.dict_Materials[self.key_material][1]
                            self.deviatoricstrain, self.new_latticeparameters = CP.DeviatoricStrain_LatticeParams(self.dict_grain_matrix[grain_index],
                                                                                             initial_latticeparameters,
                                                                                             constantlength='a')
                            # write .fit file of single grain spots results

                            self.writeFitFile(grain_index,
                                              corfilename=corfilename,
                                              dirname=dirnameout_fitfile,
                                              addpixdev=True,
                                              add_strain_sampleframe=True
                                              )

                            if fromIMM:
                                # because IMM may provides once a list of matrices
                                NeedtoProvideNewMatrices = False
                            else:
                                # because IAM provides a unique matrix
                                NeedtoProvideNewMatrices = True

                            # tagging and inhibiting exp spot too much ambiguous
                            MissingRef_grain_index = self.LabelMissingReflections(grain_index, .5)
                            self.MissingRefindexedgrains.append(MissingRef_grain_index)

                            self.dict_indexedgrains_material[grain_index] = key_material

                            indexgraincompleted = True
                            grain_index += 1
                            nbgrains_found += 1
                            self.UBStack=None
                            self.nbMatricesInUBStack=0

                    # if this is not the last tolerance step
                    else:
                        pass

    def updateIndexationDict(self, indexation_res, grain_index, overwrite=0):
        """
        update dictionary of experimental spots for the given family "grain_index"
        according to indexation info in "indexation_res"
        (unambiguous links between one exp. and one theo. spots)

        grain_index        : integer, index of grain corresponding to the pairs
        overwrite        : 1 ,prior to spots properties update, reset (i.e. unindex)
                            all spots that were considered (maybe wrongly) to 
                            belong to family "grain_index"
                            
        return:
        number of new pairs (exp. theo.) found  
        """

        if overwrite:
            self.resetSpotsFamily(grain_index)

        links_Miller = indexation_res[2]
        links_energy = indexation_res[5]

#        print "links_Miller", links_Miller
#        print "links_energy", links_energy

#        linked_spots = []

        nb_updates = 0
        for k_link, link in enumerate(links_energy):
            exp_index, theo_index, energy = link
            miller_indices = links_Miller[k_link][1:]
    #        print "link", link

            # TODO: just to deal with 0 or [0]...? as it comes from spotlinks()
            try:
                theo_index = int(theo_index[0])
            except:
                theo_index = int(theo_index)

            # this exp spot has not been already indexed
            if self.indexed_spots_dict[exp_index][-1] != 1:
                # keep the spot data and add theo info

                self.indexed_spots_dict[exp_index] = self.indexed_spots_dict[exp_index][:6] + \
                                                                        [miller_indices,
                                                                        energy,
                                                                        grain_index,
                                                                        1]
#                linked_spots.append(exp_index)
                nb_updates += 1

        print("\ngrain #%d : %d links to simulated spots have been found " % (grain_index, nb_updates))
#        print "absolute spot indices that have been linked", linked_spots

        return nb_updates

    def LabelMissingReflections(self, grain_index, angle_tol):
        """
        tag or index closest exp. spots (can be many) for each missing reflection
        as belonging to missing part of grain_index

        close exp. spots of missing reflections of grain_index are indexed as if belonging
        to -grain_index-100
        """

        data, millers, energies = self.dict_Missing_Reflections[grain_index]

        # arbitrary convention for tagging missing reflections as indexed
        MissingRef_grain_index = -(100 + grain_index)

        if len(data[0]) == 0:
            print("there is no missing reflections for grain %d" % grain_index)
            return

        Data = np.array(data).T

        for k, pos_data in enumerate(Data):
            close_spots = self.getCloseExpSpots(pos_data[:2], angle_tol, verbose=0)

            miller_indices = millers[k]
            energy = energies[k]

            for close_spot in close_spots:
                exp_index = int(close_spot[0])
                # keep the spot data and add theo info

                self.indexed_spots_dict[exp_index] = self.indexed_spots_dict[exp_index][:6] + \
                                                                        [miller_indices,
                                                                        energy,
                                                                        MissingRef_grain_index,
                                                                        1]

#         print "Number of missing reflections:",len(close_spots)
        print("Experimental experimental spots indices which are not indexed",close_spots[:, 0])
        print("Missing reflections grainindex is %d for indexed grainindex %d" % (MissingRef_grain_index,grain_index))
        print("within angular tolerance %.3f" % angle_tol)

        return MissingRef_grain_index

    def getCloseExpSpots(self, coords, angle_tol, verbose=0):
        """
        get index of exp. spots close to coords (2theta,chi) within angular tolerance

        input:
        coords        :2 values in iterable  : [0]  2theta, [1] chi
        angle_tol        : angular tolerance in degree

        return:
        array of [spot index, angle distance from coords spot in degree]

        """

        spot1 = np.array([[coords[0] / 2., coords[1]]])
        # exp data
        spots2_data = self.getSpotsExpData().T

        spots2 = np.array([spots2_data[:, 0] / 2., spots2_data[:, 1]]).T

        table_dist = INDEX.calculdist_from_thetachi(spot1, spots2)[:, 0]

        pos_close = np.where(table_dist <= angle_tol)[0]

        selectedspots_data = np.take(spots2_data, pos_close, axis=0)
        selectedspots_pos = pos_close
        selectedspots_deviation = table_dist[pos_close]

#        print selectedspots_data
#        print selectedspots_pos
#        print selectedspots_deviation

        stackdata = np.zeros((len(pos_close), 5))
        stackdata[:, 0] = selectedspots_pos
        stackdata[:, 1:4] = selectedspots_data
        stackdata[:, 4] = selectedspots_deviation

        # sorting data according to last column ie angular deviation
        order = stackdata[:, 4].argsort()
        sorteddata = np.take(stackdata, order, 0)


        pos_closest = np.argmin(table_dist)

        if verbose:
            print("table_dist", table_dist)
            print("close exp spot index", pos_close)
            print("corresponding residues", table_dist[pos_close])
            print("closest exp spot", pos_closest)
            print("at %.3f deg" % table_dist[pos_closest])

            print("sorted", sorteddata)

        return np.take(sorteddata, (0, 4), axis=1)

    def getClosestExpSpot(self, coords, angle_tol):
        """
        return index and angle deviation of closest exp. spot from spot with coords

        input:
        coords        :2 values in iterable  : [0]  2theta, [1] chi
        angle_tol        : angular tolerance in degree
        """
        resclose = self.getCloseExpSpots(coords, angle_tol, verbose=0)
        if len(resclose) == 0:
            return None, None
        else:
            return resclose[0]

    def getSpotsLinks(self, UBOrientMatrix,
                        exp_data=None,
                        useabsoluteindex=None,
                        removeharmonics=1,
                        ResolutionAngstrom=False,
                        veryclose_angletol=1.,
                        returnMissingReflections=True,
                        verbose=0):
        """
        return links (pairs or associations) between experimental and theoretical spots
        (i.e. simulated from a grain with UBOrientMatrix, key_material)

        exp_data   : None then use class attribute TwiceTheta_Chi_Int
                    otherwise should contain at least 3 elements : twicetheta_data, chi_data, I_data, ...

        useabsoluteindex        : array of conversion from relative spot index to absolute one
                                useabsoluteindex[relative]= absolute
                                for instance When exp_data have been
                                extracted from a bigger data set.

                                if None: local index is absolute index in data
                                
        return:
         res, nb_of_simulated_spots, Missing_Reflections_Data
        """
        # experimental data
        if exp_data is not None:
            twicetheta_data, chi_data, I_data = exp_data[:3]
        else:
            twicetheta_data, chi_data, I_data = self.TwiceTheta_Chi_Int

        if useabsoluteindex is None:
            useabsoluteindex = self.absolute_index

#         print "self.pixelsize in getSpotsLinks()", self.pixelsize, type(self.pixelsize)
#         print "self.dim in getSpotsLinks()", self.dim, type(self.dim)
        print("UBOrientMatrix",UBOrientMatrix)
        # simulated data
        grain = CP.Prepare_Grain(self.key_material, OrientMatrix=UBOrientMatrix)
        (Twicetheta, Chi,
         Miller_ind,
         posx, posy, Energy) = LAUE.SimulateLaue(grain,
                                               self.emin, self.emax,
                                               self.detectorparameters,
                                               removeharmonics=removeharmonics,
                                               ResolutionAngstrom=ResolutionAngstrom,
                                               pixelsize=self.pixelsize,
                                               dim=self.dim,
                                               detectordiameter=self.detectordiameter * 1.25)

        nb_of_simulated_spots = len(Twicetheta)

        # find close pairs between exp. and theo. spots
        res = SpotLinks(twicetheta_data, chi_data, I_data,  # experimental data
                        veryclose_angletol,  # tolerance angle
                        Twicetheta, Chi, Miller_ind, posx, posy, Energy,
                        absoluteindex=useabsoluteindex)

        if res == 0 or len(res[1]) == 0:
            if returnMissingReflections == False:
                return None, None
            return None, None, None

        if returnMissingReflections == False:
            return res, nb_of_simulated_spots

        if returnMissingReflections == True:
            # Missing reflections:
            # ie theo (simulated) spots that have not been paired with an experimental spot

            # theo. pairs spot index
            tps = list(res[1][:, 1])

            missing_refs = list(set(range(nb_of_simulated_spots)) - set(tps))

            if verbose:
        #        print "theo paired spots indexed", tps
                print("nb of theo spots when trying to pairing spots", nb_of_simulated_spots)

            nb_indexed_spots = nb_of_simulated_spots - len(missing_refs)
            print("For angular tolerance %.2f deg" % veryclose_angletol)
            print("Nb of pairs found / nb total of expected spots: %d/%d" % (nb_indexed_spots, nb_of_simulated_spots))
            print('Matching Rate : %.2f' % (100.*nb_indexed_spots / nb_of_simulated_spots))
            print("Nb missing reflections: %d" % len(missing_refs))

            # Twicetheta, Chi, posx, posy, Miller_ind, Energy
            ar_data = np.array([Twicetheta, Chi, posx, posy])

            Missing_Reflections_Pos = np.take(ar_data, missing_refs, axis=1)
            Missing_Reflections_Miller = np.take(Miller_ind, missing_refs, axis=0)
            Missing_Reflections_Energy = Energy[missing_refs]

    #        print "Missing_Reflections_Pos", Missing_Reflections_Pos
    #        print "Missing_Reflections_Miller", Missing_Reflections_Miller
    #        print "Missing_Reflections_Energy", Missing_Reflections_Energy

            Missing_Reflections_Data = Missing_Reflections_Pos, Missing_Reflections_Miller, Missing_Reflections_Energy

    #        # check if some missing reflections are quite close to some exp spots
    #        Resi, ProxTable = INDEX.getProximity(Missing_Reflections_Pos[:2], # warning array(2theta, chi)
    #                                                    twicetheta_data / 2., chi_data, # warning theta, chi for exp
    #                                                    proxtable=1, angtol=veryclose_angletol,
    #                                                    verbose=0,
    #                                                    signchi=1)[:2]
    #
    #        print "Resi", Resi
    #        print "ProxTable", ProxTable

            return res, nb_of_simulated_spots, Missing_Reflections_Data

    def getOrients_ImageMatching(self, MatchingRate_Threshold=None,
                                 exceptgrains=None,
                                 verbose=0):
        """
        get best orientation matrices with imagematching technique from experimental spot data

        spots are both unindexed and
                    previously indexed as belonging in grains of index in list exceptgrains

        return:
        [0]    : list of OrientMatrix object
        [1]    : corresponding (scalar) matching rate 
        [2]    : nb of experimental spots used

        """
        (self.TwiceTheta_Chi_Int,
         self.absolute_index) = self.getSelectedExpSpotsData(exceptgrains=exceptgrains)

        nbspotsIMM = len(self.absolute_index)

        if MatchingRate_Threshold is None:
            MatchingRate_Threshold = 0.

        bestEULER = IMM.bestorient_from_2thetachi(self.TwiceTheta_Chi_Int,
                                                  self.IMMdatabase,
                                                dictparameters=self.dict_IMM)

        if bestEULER is None:
            return None, None, nbspotsIMM
        else:
            RAW_ANGULAR_MATCHINGTOLERANCE = 1.
            print("bestEULER", bestEULER)
            bestEULER, bestmatchingrates = filterEulersList(bestEULER,
                                                    self.TwiceTheta_Chi_Int,
                                                    self.key_material,
                                                    self.emax,
                                                    rawAngularmatchingtolerance=RAW_ANGULAR_MATCHINGTOLERANCE,
                                                    MatchingRate_Threshold=MatchingRate_Threshold,
                                                    verbose=verbose)

        if bestEULER is None:
            return None, None, nbspotsIMM
        else:
            bestUB = []
            for eulers in bestEULER:
                bestUB.append(OrientMatrix(eulers=eulers))

            return bestUB, bestmatchingrates, nbspotsIMM

    def setTable_Angdist(self, nbmax=None):
        """
        set mutual angles between spots from current exp spot data
        
        set attibrute: table_angdist
        """
        theta, Chi = self.TwiceTheta_Chi_Int[0] / 2., self.TwiceTheta_Chi_Int[1]
        thechi_exp = np.array([theta, Chi]).T

        nbspots = len(theta)

        print('nbspots', nbspots)

        if nbmax is not None:
            nbmax = min(nbmax, nbspots)

        self.table_angdist = INDEX.calculdist_from_thetachi(thechi_exp, thechi_exp)[:nbmax, :nbmax]

        tablelength = len(self.table_angdist)

#        print "new table of distance has been computed"
#        print "with %d spots" % tablelength

        return self.table_angdist

    def computeLUT(self):
        """
        compute look_up_table angular distances between reciprocal nodes of a known structure
        
        Consider self.key_material, self.n_LUT
        """
        print("Compute LUT for indexing %s spots in LauePattern " % self.key_material)
        latticeparams = DictLT.dict_Materials[self.key_material][1]
        self.B_LUT = CP.calc_B_RR(latticeparams)
        self.LUT = INDEX.build_AnglesLUT(self.B_LUT, self.n_LUT,
                                         MaxRadiusHKL=self.ResolutionAngstromLUT,
                                         cubicSymmetry=CP.hasCubicSymmetry(self.key_material))

    def setAnglesLUTmatchingParameters(self, LUT=None,
                                       n_LUT=3,
                                       B_LUT=np.eye(3)):
        """
        set parameters for angles LUT matching technique
        """
        self.B_LUT = B_LUT
        self.n_LUT = n_LUT
        if LUT is not None:
            self.LUT = LUT
            print("Using LUT previously calculated")
        else:
            # use B from material to compute LUT
            print("Computing LUT from material data")
            self.computeLUT()

    def get_all_UBs_fromAnglesLUT(self, spot_index_central=None,
                                  max_nb_of_solutions_per_central_spot=5,
                                  MatchingRate_Threshold=None,
                                  MatchingRate_Angle_Tol=0.5,
                                  nbmax_probed=10,
                                  Minimum_Nb_Matches=6,
                                  LUT=None,
                                  set_central_spots_hkl=None,
                                  ResolutionAngstrom=False,
                                  exceptgrains=None,
                                  verbose=0):
        """
        get all orientation matrices from angular distance matching technique
        with experimental spot data
        
        USED in FileSeries

        exp. spots are those unindexed
        
        
        spot_index_central : integer or list of integer, 0 if None

        TODO: Not implemented!  All matrix are probed without stopping
        MatchingRate_Threshold   : in percent, matching rate above which loop is interrupted
                                    100   , never interrupted
        """
        Minimum_Nb_Matches = max(Minimum_Nb_Matches, MINIMUM_NB_MATCHES_FOR_INDEXING)

#        self.TwiceTheta_Chi_Int, self.absolute_index = self.getSelectedExpSpotsData(exceptgrains=exceptgrains)
#         nbspotsIAM = len(self.absolute_index)

        if MatchingRate_Threshold is None:
            MatchingRate_Threshold = 100.

        if spot_index_central is None:
            # default: use the first (and most intense) spot in data list
            spot_index_central = 0

#         matrix, score, Threshold_reached = INDEX.getOrients_AnglesLUT(spot_index_central,
#                                             self.table_angdist,
#                                             self.TwiceTheta_Chi_Int[0],
#                                             self.TwiceTheta_Chi_Int[1],
#                                             n=self.n_LUT,
#                                             B=self.B_LUT,
#                                             LUT=self.LUT,
#                                               ResolutionAngstrom=False,
#                                             Matching_Threshold_Stop=MatchingRate_Threshold,
#                                             angleTolerance_LUT=self.AngTol_LUTmatching,
#                                             MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
#                                             key_material=self.key_material,
#                                             emax=self.emax,
#                                             absoluteindex=self.absolute_index,
#                                             detectorparameters=self.simulparameter,
#                                             verbose=verbose)

        list_matrices, list_stats = self.FindOrientMatrices(
                        spot_index_central=spot_index_central,
                       nbmax_probed=nbmax_probed,
                       nLUT=self.n_LUT, LUT=LUT,
                       set_central_spots_hkl=set_central_spots_hkl,
                       ResolutionAngstrom=ResolutionAngstrom,
                       AngTol_LUTmatching=self.AngTol_LUTmatching,
                       MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                       Minimum_Nb_Matches=Minimum_Nb_Matches,
                       nb_of_solutions_per_central_spot=max_nb_of_solutions_per_central_spot,
                       simulparameters=self.simulparameter)

#         print "list_matrices", list_matrices
#         print 'list_stats', list_stats

        print("Nb of potential UBs ", len(list_matrices))

        if len(list_matrices) == 1:  # patch when only matrix found by FindOrientMatrices
            list_stats = [list_stats[0][:3]]
#         print "list_stats", list_stats

        nbspotsIAM = len(self.absolute_index)
        if list_matrices == []:
            return None, None, nbspotsIAM, False

        MatchingRate_list = []
        for stat in list_stats:
            nbmatches, nbtheo, meanresidue = stat
            MR = 100. * nbmatches / nbtheo
            MatchingRate_list.append(MR)

        MatchingRates = np.array(MatchingRate_list)

        return list_matrices, list_stats, nbspotsIAM, MatchingRates

    def get_bestUB_fromAnglesLUT(self, spot_index_central=None,
                                  MatchingRate_Threshold=None,
                                  MatchingRate_Angle_Tol=0.5,
                                  nbmax_probed=10,
                                  Minimum_Nb_Matches=6,
                                  LUT=None,
                                  set_central_spots_hkl=None,
                                  ResolutionAngstrom=False,
                                  exceptgrains=None,
                                  verbose=0):
        """
        get single best orientation matrix from angular distance matching technique
        with experimental spot data
        
        USED in FileSeries

        spots are unindexed 

        spot_index_central : integer or list of integer, 0 if None

        TODO: Not implemented!  All matrix are probed without stopping
        MatchingRate_Threshold   : in percent, matching rate above which loop is interrupted
                                    100   , never interrupted
        """
        Res = self.get_all_UBs_fromAnglesLUT(spot_index_central=spot_index_central,
                                  max_nb_of_solutions_per_central_spot=1,
                                  MatchingRate_Threshold=MatchingRate_Threshold,
                                  MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                                  nbmax_probed=nbmax_probed,
                                  Minimum_Nb_Matches=Minimum_Nb_Matches,
                                  LUT=LUT,
                                  set_central_spots_hkl=set_central_spots_hkl,
                                  ResolutionAngstrom=ResolutionAngstrom,
                                  exceptgrains=exceptgrains,
                                  verbose=verbose)

#         print "Res", Res
        if Res[0] is None:
            return Res

        (list_matrices, list_stats, nb_spots_theo, MatchingRates) = Res

        nbspotsIAM = len(self.absolute_index)

        best_result_index = np.argsort(MatchingRates)[::-1][0]

        matrix = list_matrices[best_result_index]
        score = MatchingRates[best_result_index]

        Threshold_reached = False

        bestUB = OrientMatrix(matrix=matrix)

        return bestUB, score, nbspotsIAM, Threshold_reached

    def getStatsOnMatching(self, List_of_Angles, exp_data,
                       ang_tol=1.,
                       verbose=0):
        """
        method returning matching rate in terms of nb of close pairs of spots (exp. and simul. ones)
        within angular tolerance

        :param List_of_Angles: list of 3 angles (EULER angles)
        :type List_of_Angles: list of 3 floats
        :param exp_data: experimental spots angles coordinates (kf vectors)
        :type exp_data: 2 arrays of floats (2theta and  chi)
        :param ang_tol: matching angular tolerance to form pairs
        :type ang_tol: float

        :return: [0] sorted indices of List_of_Angles elements by decreasing matching rate,
                [1] all matching rate corresponding to element in List_of_Angles
        """
        twicetheta_data, chi_data, intensity_data = exp_data
        kk = 0
        allmatchingrate = []
        for angles_sol in List_of_Angles:

            test_Matrix = GT.fromEULERangles_toMatrix(angles_sol)

            res = Angular_residues(test_Matrix, twicetheta_data, chi_data,
                                         ang_tol=ang_tol,
                                         key_material=self.key_material,
                                         emin=self.emin,
                                         emax=self.emax)
            if res is None:
                continue

            matching_rate = 100.*res[2] / res[3]
            absolute_matching_rate = res[2]
            allmatchingrate.append(matching_rate)

            if verbose:
                print("*"*30)
                print("res for k:%d and angles: [%.1f,%.1f,%.1f]" % (kk, angles_sol[0], angles_sol[1], angles_sol[2]))
        #        print "res",res[2:]
                print("Matching rate ----(in %%):                %.2f " % matching_rate)
                print("Nb of close spot pairs (< %.2f deg) : %d" % (ang_tol, res[2]))
                print("Nb of simulated spots : %d" % res[3])
                print("mean angular residues %.2f deg." % res[4])
                print("highest residues %.2f deg." % res[5])
            kk += 1

        return np.argsort(np.array(allmatchingrate))[::-1], allmatchingrate

    def refineUBSpotsFamily(self, grain_index,
                        initial_matrix,
                        getstrain=0,
                        use_weights=1,
                        nbSpotsToIndex='all',
                        verbose=0):
        """
        refine UB matrix of spots family belonging to grain being indexed
        
        input:
        grain_index:  integer , index of grain
        initial_matrix   :    UB matrix 3*3 array to be refined
        getstrain : always =1 , obsolete to be deleted
        use_weights : refine model parameters by weighting each spots pair (exp.- theo) by intensity of exp. spot.
        nbSpotsToIndex  : integer or 'all' select the nb of pairs to used for refinement. 
        
        set results values for:
        self.refinedUBmatrix = newUBmat
        self.B0matrix = Bmatrix
        self.deviatoricstrain = devstrain
        # list of residues in pixels for the considered pairs in the model
        self.pixelresidues = residues
        # list of exp spots absolute index in considered pairs in the model
        self.spotindexabs = index
        """
        MINIMUM_LINKS_FOR_FIT = 8
        
#         print "self.indexed_spots_dict",self.indexed_spots_dict

        data_1grain_raw = self.getSpotsFamilyallData(grain_index, onlywithMiller=1)
        
        if isinstance(nbSpotsToIndex,int):
            data_1grain = data_1grain_raw[:nbSpotsToIndex]
        else:
            data_1grain=data_1grain_raw
        
#         print "data_1grain",data_1grain

#         print "absolute index of spots to refine", data_1grain[:, 0]
#         print "data_1grain.shape", data_1grain.shape
        
        if len(data_1grain) >= MINIMUM_LINKS_FOR_FIT:
            index, tth, chi, posX, posY, intensity, H, K, L, Energy = data_1grain.T
        else:
            print("Too few exp. data to fit")
            return None, None

        Miller = np.array([H, K, L]).T

    #    print "Miller", Miller

        nb_pairs = len(index)
        if verbose:
            print("Nb of pairs: ", nb_pairs)

        sim_indices = np.arange(nb_pairs)

        if use_weights in (True, 'True', 1, 'true'):
            weights = intensity
        else:
            weights = None

        # fitting procedure for one or many parameters
        initial_values = np.array([1., 1., 0., 0., 0., 0, .0, 0.])
        try:
            allparameters = np.array(self.detectorparameters.tolist() + [1, 1, 0, 0, 0] + [0, 0, 0])
        except:
            allparameters = np.array(self.detectorparameters + [1, 1, 0, 0, 0] + [0, 0, 0])

        arr_indexvaryingparameters = np.arange(5, 13)
        if verbose:
            print("\nInitial error--------------------------------------\n")
            print("initial_matrix", initial_matrix)

        latticeparams = DictLT.dict_Materials[self.key_material][1]
        Bmatrix = CP.calc_B_RR(latticeparams)

        residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(initial_values,
                                        Miller,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        sim_indices,
                                        posX, posY,
                                        initrot=initial_matrix,
                                        Bmat=Bmatrix,
                                        pureRotation=0, verbose=1,
                                        pixelsize=self.pixelsize, dim=self.dim,
                                        weights=weights,
                                        signgam=1,
                                        kf_direction=self.kf_direction)
    #    print "Initial residues", residues
    #    print "---------------------------------------------------\n"

        results = FitO.fit_on_demand_strain(initial_values,
                        Miller,
                        allparameters,
                        FitO.error_function_on_demand_strain,
                        arr_indexvaryingparameters,
                        sim_indices,
                        posX, posY,
                        initrot=initial_matrix,
                        Bmat=Bmatrix,
                        pixelsize=self.pixelsize, dim=self.dim,
                        verbose=0,
                        weights=weights,
                        signgam=1,
                        kf_direction=self.kf_direction)

    #    print "\n********************\n       Results of Fit        \n********************"
    #    print "results", results

        # print "\nFinal error--------------------------------------\n"
        residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(results,
                                        Miller,
                                        allparameters,
                                        arr_indexvaryingparameters,
                                        sim_indices,
                                        posX, posY,
                                        initrot=initial_matrix,
                                        Bmat=Bmatrix,
                                        pureRotation=0, verbose=1, pixelsize=self.pixelsize, dim=self.dim,
                                        weights=None,
                                        signgam=1,
                                        kf_direction=self.kf_direction)

        if 1:  # getstrain:

            # building B mat
            param_strain_sol = results
            starting_orientmatrix = initial_matrix

            varyingstrain = np.array([[1., param_strain_sol[2], param_strain_sol[3]],
                                      [0, param_strain_sol[0], param_strain_sol[4]],
                                      [0, 0, param_strain_sol[1]]])
            if verbose:
                print("varyingstrain results")
                print(varyingstrain)

            newUmat = np.dot(deltamat, starting_orientmatrix)

            # building UBmat(= newmatrix)
            newUBmat = np.dot(np.dot(deltamat, starting_orientmatrix), varyingstrain)
            if verbose:
                print("newUBmat", newUBmat)
            # print "newmatrix",newmatrix # must be equal to UBmat

            # RR,PP = GT.UBdecomposition_RRPP(UBmat)

            # symetric matrix(strain) in direct real distance
            # epsil = GT.epsline_to_epsmat(CP.calc_epsp(CP.dlat_to_rlat(CP.matrix_to_rlat(PP))))

            # #print "epsil is already a zero trace symetric matrix ",np.round(epsil*1000, decimals = 2)
            # # if the trace is not zero
            # epsil = epsil - np.trace(epsil)*np.eye(3)/3.

            # epsil_round = np.round(epsil*1000, decimals = 2)
            # print "\n********************\n       result of Strain and Orientation  Fitting        \n********************"
            # print "last pixdev table non weighted"
            # print residues_non_weighted.round(decimals=3)
            # print "Mean pixdev non weighted"
            # print np.mean(residues_non_weighted).round(decimals=5)
            # print "\nPure Orientation Matrix"
            # #print RR.tolist()
            # print RR

            # print "Deviatoric strain with respect to initial crystal unit cell(in 10-3 units)"
            # #print epsil.tolist()
            # print epsil_round
            # print "Pure strain"
            # purestrain =(varyingstrain + varyingstrain.T)/2 - np.eye(3)
            # print purestrain
            # print "trace of pure strain", np.trace(purestrain)
            # print "deviatoric strain of pure strain"
            # print purestrain - np.trace(purestrain)/3.*np.eye(3)

            Bstar_s = np.dot(newUBmat, Bmatrix)
            if verbose:
                print("new UBs matrix in q= UBs G (s for strain)")
                print(Bstar_s)

            lattice_parameter_reciprocal = CP.matrix_to_rlat(Bstar_s)
            lattice_parameter_direct_strain = CP.dlat_to_rlat(lattice_parameter_reciprocal)

            Bmatrix_direct_strain = CP.calc_B_RR(lattice_parameter_direct_strain, directspace=0)
            Bmatrix_direct_unstrained = CP.calc_B_RR(latticeparams, directspace=0)

            Trans = np.dot(Bmatrix_direct_strain, np.linalg.inv(Bmatrix_direct_unstrained))
            strain_direct = (Trans + Trans.T) / 2. - np.eye(3)

            devstrain = strain_direct - np.trace(strain_direct) / 3.*np.eye(3)
            print("devstrain, lattice_parameter_direct_strain",devstrain, lattice_parameter_direct_strain)
            
            
            devstrain1, lattice_parameter_direct_strain1 = CP.DeviatoricStrain_LatticeParams(newUBmat,
                                                                                             latticeparams,
                                                                                             constantlength='a')
            print("devstrain1, lattice_parameter_direct_strain1",devstrain1, lattice_parameter_direct_strain1)

#             devstrain_round = np.round(devstrain * 1000, decimals=2)

            self.refinedUBmatrix = newUBmat
            self.B0matrix = Bmatrix
            self.deviatoricstrain = devstrain

            self.pixelresidues = residues
            self.spotindexabs = index

            if verbose:
                print("strain_direct", strain_direct)

#             print "deviatoric strain", devstrain
            print("UB and strain refinement completed")

            return newmatrix, devstrain

        if verbose:
            print("newmatrix", newmatrix)

        return newmatrix, None
    
    def refineStrainElementsSpotsFamily(self, grain_index,
                        initial_matrix,
                        getstrain=0,
                        use_weights=1,
                        verbose=0):
        """
        refine UB matrix and strain elements of exp. spots of family 'grain_index'
        
        set results values for:
        self.refinedUBmatrix = newUBmat
        self.B0matrix = Bmatrix
        self.deviatoricstrain = devstrain
        # list of residues in pixels for the considered pairs in the model
        self.pixelresidues = residues
        # list of exp spots absolute index in considered pairs in the model
        self.spotindexabs = index
        """
        FitTransformParametersFlags = [False, True, True, True, True, True]
        FitOrientParametersFlags = [True, True, True]
        
        MINIMUM_LINKS_FOR_FIT = 8

        data_1grain = self.getSpotsFamilyallData(grain_index, onlywithMiller=1)

#        print "absolute index of spots to refine", data_1grain[:, 0]
    #    print "data_1grain.shape", data_1grain.shape
        if len(data_1grain) >= MINIMUM_LINKS_FOR_FIT:
            index, tth, chi, posX, posY, intensity, H, K, L, Energy = data_1grain.T
#             
#             print "tth exp", tth
#             print "chi exp", chi
        else:
            print("Too few exp. data to fit")
            return None, None

        Miller = np.array([H, K, L]).T

    #    print "Miller", Miller

        nb_pairs = len(index)
        if verbose:
            print("Nb of pairs: ", nb_pairs)

        sim_indices = np.arange(nb_pairs)

        if use_weights in (True, 'True', 1, 'true'):
            weights = intensity
        else:
            weights = None

        # needs pixX, pixY, hkls, starting_orientmatrix, key_material, weights
        # see fit_transform_parameters(self, pixX, pixY, hkls, starting_orientmatrix, key_material, weights)
        # in plotrefineGUI.py
        starting_orientmatrix = initial_matrix
        pixX, pixY = posX, posY
        hkls = Miller

        # triangular up transform in sample frame
        TransformParameters = [1, 0, 0, 1, 0, 1.]
        # corresponding to Identity matrix
        print("self.key_material", self.key_material)
        print("self.detectorparameters", self.detectorparameters)
        
        latticeparameters = DictLT.dict_Materials[self.key_material][1]
        B0matrix = CP.calc_B_RR(latticeparameters)
        

        CCDcalib = self.detectorparameters
        
        allparameters = np.array(CCDcalib + [0, 0, 0] + TransformParameters)
        
        if np.all(np.array(FitTransformParametersFlags)):
            FitTransformParametersFlags[0] = False
            
        
        List_of_defaultvalues = np.array(CCDcalib + [0, 0, 0] + TransformParameters)
        List_of_checkwidgets = FitOrientParametersFlags + FitTransformParametersFlags
                      
        List_of_keys = ['anglex', 'angley', 'anglez', 'Ts00', 'Ts01', 'Ts02', 'Ts11', 'Ts12', 'Ts22']
        
        print("List_of_defaultvalues", List_of_defaultvalues)
        
        fitting_parameters_keys = []
        fitting_parameters_values = []
        for k, flag in enumerate(List_of_checkwidgets):
            
            if flag:
#                 print 'k,checkwidget', k, flag
                fitting_parameters_keys.append(List_of_keys[k])
                fitting_parameters_values.append(List_of_defaultvalues[k + 5])
        
        pureUmatrix, residualdistortion = GT.UBdecomposition_RRPP(starting_orientmatrix)

        print('fitting_parameters_values, fitting_parameters_keys')
        print(fitting_parameters_values, fitting_parameters_keys)
        print('len(allparameters)', len(allparameters))
        print('starting_orientmatrix', starting_orientmatrix)

        absolutespotsindices = np.arange(len(pixX))
        

        print("initial errors--------------------------------------\n")
        print(FitO.error_function_strain(fitting_parameters_values, fitting_parameters_keys,
                                        hkls, allparameters,
                                        absolutespotsindices, pixX, pixY,
                                                initrot=pureUmatrix,
                                                B0matrix=B0matrix,
                                                pureRotation=0,
                                                verbose=0,
                                                pixelsize=self.pixelsize,
                                                dim=self.dim,
                                                weights=None,
                                                signgam=1,
                                                kf_direction=self.kf_direction,
                                                returnalldata=True))
        
        print("\n\n*** refineStrainElementsSpotsFamily (File Series/Index_Refine.py) *****\n")
        print("fitting_parameters_values", fitting_parameters_values)
        print("fitting_parameters_keys", fitting_parameters_keys)
        print("hkls", hkls)
        print('allparameters', allparameters)
        print("absolutespotsindices", absolutespotsindices)
        print("pixX", pixX)
        print("pixY", pixY)
        print("UBmatrix_start", pureUmatrix)
        print('B0matrix', B0matrix)
        print("self.pixelsize", self.pixelsize)
        print("self.dim", self.dim)
        print("weights", weights)
        
        print("fitting strain parameters")
        results = FitO.fit_function_strain(fitting_parameters_values, fitting_parameters_keys,
                                        hkls, allparameters,
                                        absolutespotsindices, pixX, pixY,
                                        UBmatrix_start=pureUmatrix,
                                        B0matrix=B0matrix,
                                        nb_grains=1,
                                        pureRotation=0,
                                        verbose=0,
                                        pixelsize=self.pixelsize,
                                        dim=self.dim,
                                        weights=weights,
                                        signgam=1,
                                        kf_direction=self.kf_direction)
        
        
        print("results -------", results)
        fit_completed = False
#         print "res", results
            
#             print 'self.CCDcalib', self.CCDcalib
#             print 'pixelsize', self.pixelsize
#             print 'pixX', pixX.tolist()
#             print 'pixY', pixY.tolist()
#             print 'Data_Q', Data_Q.tolist()
#             print 'starting_orientmatrix', starting_orientmatrix
#             
#             print 'self.B0matrix', self.B0matrix.tolist()
        if results is None:
            return

        fitresults = True
        print("\nFinal errors--------------------------------------\n")
        # alldistances_array, Uxyz, newmatrix, Ts, T
        (residues,
         Uxyz, newUmat, refinedTs, refinedT) = FitO.error_function_strain(results, fitting_parameters_keys,
                                        hkls, allparameters,
                                        absolutespotsindices, pixX, pixY,
                                                initrot=pureUmatrix,
                                                B0matrix=B0matrix,
                                                pureRotation=0,
                                                verbose=0,
                                                pixelsize=self.pixelsize,
                                                dim=self.dim,
                                                weights=weights,
                                                signgam=1,
                                                kf_direction=self.kf_direction,
                                                returnalldata=True)
         
        self.residues_non_weighted = FitO.error_function_strain(results, fitting_parameters_keys,
                                        hkls, allparameters,
                                        absolutespotsindices, pixX, pixY,
                                                initrot=pureUmatrix,
                                                B0matrix=B0matrix,
                                                pureRotation=0,
                                                verbose=0,
                                                pixelsize=self.pixelsize,
                                                dim=self.dim,
                                                weights=None,
                                                signgam=1,
                                                kf_direction=self.kf_direction,
                                                returnalldata=False)
        
        print("Final pixel residues", residues)
        print("---------------------------------------------------\n")
        print("Final mean pixel residues", np.mean(residues))
        
        # q = T Uxyz U B0init G*
        # q = T newU B0init G*
        # q = P Ts P-1 newU B0init G*
        # q = newUB B0init G*
        newUBmat = np.dot(refinedT, newUmat)
        
        Umat_init = pureUmatrix
        
        print("in q = T Uxyz U B0init G*")
        print("q = T newU B0init G*")
        print("q = P Ts P-1 newU B0init G*")
        print('q = newUB B0init G*')
        print("in q = newUmat varyingstrain B0init G*")
        print('Uxyz, Umat_init, newUmat, refinedT,refinedTs')
        print(Uxyz, Umat_init, newUmat, refinedT, refinedTs)
        print("newUBmat, B0matrix_init")
        print(newUBmat, B0matrix)

        varyingstrain = np.dot(np.linalg.inv(newUmat), newUBmat)
        
        Tsresults = ('triangular', 'sampleframe', refinedTs)
        
        if 1:  # getstrain:

            Bstar_s = np.dot(newUBmat, B0matrix)
            if verbose:
                print("new UBs matrix in q= UBs G (s for strain)")
                print(Bstar_s)

            lattice_parameter_reciprocal = CP.matrix_to_rlat(Bstar_s)
            lattice_parameter_direct_strain = CP.dlat_to_rlat(lattice_parameter_reciprocal)

            Bmatrix_direct_strain = CP.calc_B_RR(lattice_parameter_direct_strain, directspace=0)
            Bmatrix_direct_unstrained = CP.calc_B_RR(latticeparameters, directspace=0)

            Trans = np.dot(Bmatrix_direct_strain, np.linalg.inv(Bmatrix_direct_unstrained))
            strain_direct = (Trans + Trans.T) / 2. - np.eye(3)

            devstrain = strain_direct - np.trace(strain_direct) / 3.*np.eye(3)

            devstrain_round = np.round(devstrain * 1000, decimals=2)

            refinedUBmatrix = newUBmat

            deviatoricstrain = devstrain

            pixelresidues = residues
            spotindexabs = index
            
            self.refinedUBmatrix = newUBmat
            self.B0matrix = B0matrix
            self.deviatoricstrain = devstrain
            self.pixelresidues = residues
            self.spotindexabs = index

            if verbose:
                print("strain_direct", strain_direct)

#             print "deviatoric strain", devstrain
            print("UB and strain refinement completed")
            
            

            return newUBmat, devstrain, Tsresults

        if verbose:
            print("refinedUBmatrix", newUBmat)

        return newUBmat, None, None

    def writeFileSummary(self, corfilename=None, dirname=None):
        """
        write .res file
        containing all the exp. spots with their properties
        (indexed (grain index) or not (-1)), hkl energy etc.
        """

        if corfilename is not None:
            outputfilename = corfilename.split('.')[0] + '.res'
        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)

        print("Saving Summary file: %s" % outputfilename)

        Data = self.getSummaryallData()

        (spotindex,
         grain_index,
         tth, chi, posX, posY,
         intensity,
         H, K, L,
         Energy) = Data.T

        Columns = [spotindex,
                     grain_index,
                     tth, chi, posX, posY,
                     intensity,
                     H, K, L,
                     Energy]

        nbspots = len(spotindex)

        datatooutput = np.transpose(np.array(Columns))
        datatooutput = np.round(datatooutput, decimals=7)

        header = '# Spots Summary of: %s\n' % (self.filename)
        header += '# File created at %s with indexingSpotsSet.py\n' % (time.asctime())
#         header += '# Number of indexed spots: %d\n' % nbindexedspots
#         header += '# Number of unindexed spots: %d\n' % nbunindexedspots
        header += '# Number of spots: %d\n' % nbspots

        header += '#spot_index grain_index 2theta Chi Xexp Yexp intensity h k l Energy\n'
        outputfile = open(outputfilename, 'w')

        outputfile.write(header)

        np.savetxt(outputfile, datatooutput, fmt='%.7f')

        print('self.indexedgrains', self.indexedgrains)
        for grain_index in self.indexedgrains:
            outputfile.write('#grainIndex\n')
            outputfile.write('G_%d\n' % grain_index)
            outputfile.write('#Element\n')
            key_material = self.dict_indexedgrains_material[grain_index]
            outputfile.write('%s\n' % key_material)

            MatchRate, nbspotsindexed = self.dict_grain_matching_rate[grain_index][:2]
            outputfile.write('#MatchingRate\n')
            outputfile.write('%.1f\n' % MatchRate)
            outputfile.write('#Nb indexed Spots\n')
            outputfile.write('%.d\n' % nbspotsindexed)
            outputfile.write('#UB matrix in q= (UB) B0 G*\n')
    #            outputfile.write(str(self.UBB0mat) + '\n')
            outputfile.write(str(self.dict_grain_matrix[grain_index]) + '\n')
            outputfile.write('#B0 matrix (starting unit cell) in q= UB (B0) G*\n')
            latticeparams = DictLT.dict_Materials[key_material][1]
            B0matrix = CP.calc_B_RR(latticeparams)
            outputfile.write(str(B0matrix) + '\n')
            outputfile.write('#deviatoric strain (10-3 unit)\n')
            outputfile.write(str(self.dict_grain_devstrain[grain_index] * 1000.) + '\n')

        addCCDparams = 0
        if addCCDparams:
            outputfile.write('#CCDLabel\n')
            outputfile.write(self.CCDLabel + '\n')
            outputfile.write('#DetectorParameters\n')
            outputfile.write(str(self.CCDcalib) + '\n')
            outputfile.write('#pixelsize\n')
            outputfile.write(str(self.pixelsize) + '\n')
            outputfile.write('#Frame dimensions\n')
            outputfile.write(str(self.framedim) + '\n')

        outputfile.close()

    def writeFitFile(self, grain_index,
                     corfilename=None, dirname=None,
                     addpixdev=False,
                     verbose=0,
                     add_strain_sampleframe=False,
                     add_grainindex_in_outputfilename=True):
        """
        write a .fit file
        list of spots belonging to a single grain

        extension:    _g(grain_index).fit for .fit with only data of grain number grain index
        (e.g. _g2.fit for the third indexed grain). Grain_index starts from zero)
        """
        # set output file name
        if corfilename is not None:
            
            if add_grainindex_in_outputfilename:
                outputfilename = corfilename.split('.')[0] + '_g%d' % grain_index + '.fit'
            else:
                outputfilename = corfilename.split('.')[0] + '.fit'

        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)
        # get spots data
        dataspots=self.getSpotsFamilyallData(grain_index, onlywithMiller=1)
        nbindexedspots=len(dataspots)
        if nbindexedspots>1:
            (index,
             tth, chi,
             posX, posY,
             intensity,
            H, K, L,
            Energy) = dataspots.T
        elif nbindexedspots==1:
            (index,
             tth, chi,
             posX, posY,
             intensity,
            H, K, L,
            Energy) = dataspots
        else:
            print('data spots are empty... Nothing to write in .fit file')
            return

#         print "len(index)", len(index)
#         print "index[0], index[-1]", index[0], index[-1]

        if add_strain_sampleframe:
#             deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame(self.deviatoricstrain,
#                                                    self.refinedUBmatrix,
#                                                     LaueToolsFrame_for_UBmat=True)
            deviatoricstrain_sampleframe = CP.strain_from_crystal_to_sample_frame2(self.deviatoricstrain,
                                                   self.refinedUBmatrix)

        if addpixdev:
            pixeldevs = self.pixelresidues
            abs_spotindex = self.spotindexabs

            if verbose:  # verbose:
                print("pixeldevs", pixeldevs)
                print("abs_spotindex", abs_spotindex)
                print("index", index)
                print("len(pixeldevs)", len(pixeldevs))
                print("len(abs_spotindex)", len(abs_spotindex))
                print("len(index)", len(index))

            #--------------------------
            # since index list contains less elements than last results of refinement
            # index list comes from getSpotsFamilyallData
            # (which relies on the last state of assignation)

            # Then, keep then only element from index that have been refined
            pos = []
            for ind in index:
                pos.append(np.where(abs_spotindex == ind)[0][0])

#             print 'pos', pos

            # take a part of  pixeldevs
            pixeldevarray = pixeldevs[pos]
            #-----------------------------

            nbindexedspots = len(index)
            grain_index * np.ones(nbindexedspots)

            Columns = [index,
                       intensity,
                       H, K, L,
                       tth, chi,
                        posX, posY,
                        Energy,
                        grain_index * np.ones(nbindexedspots),
                        pixeldevarray]
        else:
            Columns = [index,
                       intensity,
                       H, K, L,
                       tth, chi,
                        posX, posY,
                        Energy]
            nbindexedspots = len(index)

        if verbose:
            print("Columns", Columns)

        datatooutput = np.transpose(np.array(Columns))
        datatooutput = np.round(datatooutput, decimals=7)

        try:
            dict_matrices = {}
            dict_matrices['Element'] = self.key_material
            dict_matrices['grainIndex'] = grain_index
    
            dict_matrices['UBmat'] = self.dict_grain_matrix[grain_index]
            dict_matrices['B0'] = self.B0matrix
    #         dict_matrices['UBB0'] = self.UBB0mat
            UBB0 = np.dot(dict_matrices['UBmat'],dict_matrices['B0'] )
            euler_angles = GT.calc_Euler_angles(UBB0).round(decimals=3)
            dict_matrices['euler_angles'] = euler_angles
    
            dict_matrices['devstrain_crystal'] = self.deviatoricstrain
            dict_matrices['detectorparameters'] = self.detectorparameters
            dict_matrices['pixelsize'] = self.pixelsize
            dict_matrices['framedim'] = self.dim
            if add_strain_sampleframe:
                dict_matrices['devstrain_sample'] = deviatoricstrain_sampleframe
                
            dict_matrices['LatticeParameters']=self.new_latticeparameters
            
            dict_matrices['Ts'] = self.refinedTs
    
            if self.CCDLabel is not None:
                dict_matrices['CCDLabel'] = self.CCDLabel
        except AttributeError:
            print("Missing attributes. Refinement has worked very well: few spots ?")

        meanresidues = None
        if addpixdev:
            meanresidues = np.mean(pixeldevs)
            columnsname = '#spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev\n'
        else:
            columnsname = '#spot_index intensity h k l 2theta Chi Xexp Yexp Energy\n'

        currentfolder = os.path.abspath(os.curdir)
        IOLT.writefitfile(outputfilename, datatooutput, nbindexedspots,
                 dict_matrices=dict_matrices, meanresidues=meanresidues,
                 PeakListFilename=self.filename, columnsname=columnsname,
                 modulecaller='indexingSpotsSet.py')
        print("File : %s written in %s"%(outputfilename,currentfolder))

    def writecorFile_unindexedSpots(self,
                     corfilename=None, dirname=None,
                     filename_nbdigits=None):
        """
        write a .cor file of spots that are still not indexed.
        """

        if corfilename is not None:

            posdigit, nbdigitsfound = GT.findfirstnumberpos(corfilename)
#             print "posdigit, nbdigitsfound", posdigit, nbdigitsfound

            if filename_nbdigits is not None:
                if nbdigitsfound > filename_nbdigits:
                    deltadigits = nbdigitsfound - filename_nbdigits
                    posdigit += deltadigits

            if nbdigitsfound == 0:
                str_to_add = '_unindexed'
            elif corfilename[:posdigit][-1] in ('_'):
                str_to_add = 'unindexed_'
            else:
                str_to_add = '_unindexed_'

            if corfilename.endswith('.cor'):
                outputfilename = corfilename[:posdigit] + str_to_add + corfilename[posdigit:]
            else:
                outputfilename = corfilename[:posdigit] + str_to_add + corfilename[posdigit:] + '.cor'

        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)

        # [index, tth, chi, posX, posY, intensity]
        res_unindexed = self.getUnIndexedSpotsallData(exceptgrains=self.indexedgrains).T

        if len(res_unindexed) == 0:
            return

        (index,
        tth, chi,
        posX, posY,
        intensity) = res_unindexed

        Columns = [index,
                   intensity,
                   tth, chi,
                    posX, posY]

        nbunindexedspots = len(index)

        datatooutput = np.transpose(np.array(Columns))
        datatooutput = np.round(datatooutput, decimals=7)

        # TODO: add pixdev mean and all corresponding pixdev
        header = '# Unindexed and unrefined Spots of: %s\n' % (self.filename)
        header += '# File created at %s with indexingSpotsSet.py\n' % (time.asctime())
        header += '# Number of unindexed spots: %d\n' % nbunindexedspots

        header += '#Element\n'
        header += 'None' + '\n'
        header += '#grainIndex\n'
        header += 'None' + '\n'

        header += '#spot_index intensity 2theta Chi Xexp Yexp\n'
        outputfile = open(outputfilename, 'w')

        print("Saving unindexed  fit file: %s" % outputfilename)
        outputfile.write(header)
        np.savetxt(outputfile, datatooutput, fmt='%.6f')

#         self.pixelsize = 165. / 2048
#         self.dim = (2048, 2048)
#         self.kf_direction = 'Z>0'
#         self.detectordiameter = 165.
        rectpix = 0.
        param = self.detectorparameters + [self.pixelsize]
        if param is not None:
            outputfile.write('\n# Calibration parameters')
            if len(param) == 6:
                for par, value in zip(['dd', 'xcen', 'ycen', 'xbet', 'xgam', 'pixelsize'], param):
                    outputfile.write('\n# %s     :   %s' % (par, value))
                ypixelsize = param[5] * (1.0 + rectpix)
                outputfile.write('\n# ypixelsize     :   ' + str(ypixelsize))
            elif len(param) == 5:
                for par, value in zip(['dd', 'xcen', 'ycen', 'xbet', 'xgam'], param):
                    outputfile.write('\n# %s     :   %s' % (par, value))
            else:
                raise ValueError("5 or 6 calibration parameters are needed!")

        outputfile.close()

    def merge_fitfiles(self, nbgrains, corfilename=None, dirname=None,
                       removefiles=0,
                       add_unindexed_spotslist=True):
        """
        merge into one .fit file all .fit files corresponding to one grain

        to meet Odile's way to build summary

        :param nbgrains: nb of grains or corresponding file _g#.fit
        :type nbgrains: integer
        :param corfilename: filename (without pathfolder) of .cor to extract the prefix
        :type corfilename: string
        :param dirname: path to .fit files
        :type dirname: string
        :param removefiles:
        :type removefiles:
        """
        if nbgrains <= 0 :
            return

        prefix = corfilename.split('.')[0]
        mergedfitfilename = prefix + '.fit'

        print("prefix", prefix)
        print("mergedfitfilename", mergedfitfilename)

        if dirname is not None:
            mergedfitfilename = os.path.join(dirname, mergedfitfilename)
            prefix = os.path.join(dirname, prefix)

        prefixgrain = prefix + '_g'

        outputfile = open(mergedfitfilename, 'w')
        for grainindex in range(nbgrains):

            f = open(prefixgrain + '%d' % grainindex + '.fit', 'r')
            try:
                for line in f:
                    outputfile.write(line)
            finally:
                f.close()
            outputfile.write('\n')

            if removefiles:
                os.remove(prefixgrain + '%d' % grainindex + '.fit')

        if add_unindexed_spotslist:
            posdigit, nbdigitsfound = GT.findfirstnumberpos(prefix)

            if prefix[:posdigit][-1] in ('_'):
                str_to_add = 'unindexed_'
            else:
                str_to_add = '_unindexed_'

            filename_unindexedspots = prefix[:posdigit] + str_to_add + prefix[:posdigit] + '.cor'

            if os.path.isfile(filename_unindexedspots):
                f = open(filename_unindexedspots, 'r')
                try:
                    for line in f:
                        outputfile.write(line)
                finally:
                    f.close()

            outputfile.write('\n')

        outputfile.close()

    def plotgrains_from_indexspots(self, unindexedspots=None):
        """ plot grains spots and data from indexed spots dictionary

        unindexedspots    : plot unindexed spots + previously indexed spots not in unindexedspots

        """
        isgrain = True
        grain_index = 0

        # 2theta chi from indexed spots
        twicethetaChi = [[]]
        while isgrain:
            spots = self.getSpotsFamily(grain_index)
            if spots:
                for key_spot in spots:
                    twicethetaChi[grain_index].append(self.indexed_spots_dict[key_spot][1:3])
                grain_index += 1
                twicethetaChi.append([])
            else:
                isgrain = False

        if unindexedspots is not None:
            th_unind, Chi_unind = self.getUnIndexedSpotsallData(exceptgrains=unindexedspots)[:, 1:3].T

    #    print "twicethetaChi", twicethetaChi
        all_tthchi = self.getSpotsallData()[:, 1:3]
    #    print all_tthchi
        nb_of_orientations = grain_index

        fig = figure()

        print("nb_of_orientations to plot", nb_of_orientations)
        if nb_of_orientations == 1:
            codefigure = 111
        if nb_of_orientations == 2:
            codefigure = 211
        if nb_of_orientations in (3, 4):
            codefigure = 221
        if nb_of_orientations in (5, 6):
            codefigure = 321
        if nb_of_orientations in (7, 8, 9):
            codefigure = 331
        index_fig = 0

    #    ax.set_xlim((35, 145))
    #    ax.set_ylim((-45, 45))

        # TODO: confusion ax and p ? see plotgrains()
        dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
        nbcolors = len(dicocolor)
        # theo spots
        for i_grain in range(nb_of_orientations):
            ax = fig.add_subplot(codefigure)
            # all exp spots
            scatter(all_tthchi[:, 0], all_tthchi[:, 1],
                      s=40, c='w', marker='o', faceted=True, alpha=.5)

            if unindexedspots is not None:
                # unindexed spots
                scatter(th_unind, Chi_unind,
                          s=20, c='b', faceted=False, alpha=.5)

            # simul spots
            tthchi = np.array(twicethetaChi[i_grain])
    #        print tthchi
            ax.scatter(tthchi[:, 0], tthchi[:, 1],
                   c=dicocolor[(i_grain + 1) % nbcolors], faceted=False)
            if index_fig < nb_of_orientations:
                index_fig += 1
                codefigure += 1

        show()

    def plotgrains_alldata(self, titlefig='', unindexedspots=None, verbose=0):
        self.plotgrains(exp_data=self.getSpotsExpData()[:2],
                        titlefig=titlefig, unindexedspots=unindexedspots,
                        verbose=verbose)

    def plotgrains(self, exp_data=None, titlefig='', unindexedspots=None, verbose=0):
        """
        - plot spots from current orientmatrix dictionary by simulation
        - up to 9 plots

        exp_data    :  first two elements are 2theta, chi, ...
        unindexedspots    : plot unindexed spots + previously indexed spots not in unindexedspots
        """

        EXTINCTION = DictLT.dict_Materials[self.key_material][2]

        nbMatrices = len(self.dict_grain_matrix)

        nb_matrices = 0
        all_tthchi = []
        for key_matrix in sorted(self.dict_grain_matrix.keys()):

            grain = [np.eye(3),
                     EXTINCTION,
                     self.dict_grain_matrix[key_matrix],
                     self.key_material]

            if verbose:
                print("matrix for key_matrix%d" % key_matrix)
                print(self.dict_grain_matrix[key_matrix].tolist())

            (Twicetheta, Chi,
             Miller_ind, posx, posy, Energy) = LAUE.SimulateLaue(grain,
                                                                 self.emin, self.emax,
                                                            self.detectorparameters,
                                                            detectordiameter=self.detectordiameter)

            all_tthchi.append([Twicetheta, Chi])

            nb_matrices += 1
            if nb_matrices == 9:
                print("Warning: limiting to 9 plots")
                break

#        print "nb_of_matrices to plot %d " % nb_matrices + titlefig

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

    #    ax.set_xlim((35, 145))
    #    ax.set_ylim((-45, 45))

        dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
        nbcolors = len(dicocolor)

        if exp_data is not None:
            tth_exp, Chi_exp = exp_data[:2]

        if unindexedspots is not None:
            th_unind, Chi_unind = self.getUnIndexedSpotsallData(exceptgrains=unindexedspots)[:, 1:3].T

        # theo spots
        for i_mat in range(min(nbMatrices, 9)):

            subplot(codefigure)
            if i_mat == 0:
                title("%s" % titlefig)

            if exp_data is not None:
                # all exp spots
                scatter(tth_exp, Chi_exp,
                          s=40, c='w', marker='o', faceted=True, alpha=.5)

            if unindexedspots is not None:
                # unindexed spots
                scatter(th_unind, Chi_unind,
                          s=20, c='b', faceted=False, alpha=.5)


            # simul spots

            scatter(all_tthchi[i_mat][0], all_tthchi[i_mat][1],
                   c=dicocolor[(i_mat + 1) % nbcolors], faceted=False)

            codefigure += 1

        show()


    def plotallgrains(self, unindexedspots=None):
        """ single plot of all spots and grains origin and data from indexed spots dictionary

        unindexedspots    : plot unindexed spots + previously indexed spots not in unindexedspots

        """
        isgrain = True
        grain_index = 0

        # 2theta chi from indexed spots
        twicethetaChi = [[]]
        while isgrain:
            spots = self.getSpotsFamily(grain_index)
            if spots:
                for key_spot in spots:
                    twicethetaChi[grain_index].append(self.indexed_spots_dict[key_spot][1:3])
                grain_index += 1
                twicethetaChi.append([])
            else:
                isgrain = False

        if unindexedspots is not None:
            th_unind, Chi_unind = self.getUnIndexedSpotsallData(exceptgrains=unindexedspots)[:, 1:3].T

    #    print "twicethetaChi", twicethetaChi
        all_tthchi = self.getSpotsallData()[:, 1:3]
    #    print all_tthchi
        nb_of_orientations = grain_index

    #    ax.set_xlim((35, 145))
    #    ax.set_ylim((-45, 45))

        dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
        nbcolors = len(dicocolor)

        if unindexedspots is not None:
            # unindexed spots
            scatter(th_unind, Chi_unind,
                      s=20, c='b', faceted=False, alpha=.5)

        # all exp spots
        scatter(all_tthchi[:, 0], all_tthchi[:, 1],
                      s=40, c='w', marker='o', faceted=True, alpha=.5)


        # theo spots
        for i_grain in range(nb_of_orientations):
            tthchi = np.array(twicethetaChi[i_grain])
    #        print tthchi
            scatter(tthchi[:, 0], tthchi[:, 1],
                   c=dicocolor[(i_grain + 1) % nbcolors], faceted=False)

        show()


#--- -------  PROCEDURES -------------------
def purgeSpotsinDict(exp_data, twicethetaChi_to_remove,
                     dist_tolerance=0.2):
    """
    remove undesirable spots in exp_data that listed in 
    and initialize spots indexation dictionary from exp_data

    twicethetaChi_to_remove tuple of 2 elements: 2theta,chi
    dist_tolerance = angular tolerance in a approximate 2theta chi cartesian space

    NOTE: may be used to remove substrate uninteresting peaks from substrate
    """
    nb_of_spots = len(exp_data[0])

    tth, chi, posX, posY, Intensity = exp_data

    Twicetheta, Chi, tokeep = GT.removeClosePoints_two_sets((tth, chi),
                                                             twicethetaChi_to_remove,
                                                             dist_tolerance=dist_tolerance)

    posx = np.take(posX, tokeep)
    posy = np.take(posY, tokeep)
    dataintensity = np.take(Intensity, tokeep)

    purged_exp_data = Twicetheta, Chi, dataintensity, posx, posy

    return initIndexationDict(purged_exp_data)


def initIndexationDict(exp_data):
    """
    initialize spots indexation dictionary from exp_data
    tth, chi, Intensity, posX, posY = exp_data
    """
    nb_of_spots = len(exp_data[0])

    tth, chi, Intensity, posX, posY = exp_data

    indexed_spots_dict = {}
    # dictionary of exp spots
    for k in range(nb_of_spots):
        indexed_spots_dict[k] = [k,  # index of experimental spot in .cor file
                            tth[k], chi[k],  # 2theta, chi coordinates
                            posX[k], posY[k],  # pixel coordinates
                            Intensity[k],  # intensity
                            0]  # 0 means non indexed yet

    return indexed_spots_dict


#--- -----------------  ORIENT MATRIX FILTERING
def filterEulersList(bestEULER, tth_chi_int, key_material, emax,
                                rawAngularmatchingtolerance=1.,
                                MatchingRate_Threshold=50.,
                                verbose=0):
    """
    keep unique orientation matrices (represented as 3 euler angles)
    according to symetry and with a higher matching rate (of spots calculated in 2theta, chi space)

    input:

    bestEULER    : list of 3elements euler's angle
    tth_chi_int    : [0]  2theta  [1] chi   exp. spots coordinates
    key_material    : material key for simulation
    emax            : max energy for simulation
    """
    print("*******************************use filterEulersList")

    # sort solution by matching rate
    sortedindices, matchingrates = getStatsOnMatching(bestEULER,
                                                      tth_chi_int[0], tth_chi_int[1],
                                                      key_material,
                                                      ang_tol=rawAngularmatchingtolerance,
                                                      emax=emax,
                                                      verbose=verbose)

    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    if verbose:
        print("\n --------- Image Matching Results -----------------\n")
        print("\nnb of potential grains %d" % len(bestEULER))
        print("bestmatchingrates")
        print(bestmatchingrates)

    bestEULER, bestmatchingrates = filterEquivalentMatrix(bestEULER,
                                                          bestmatchingrates)


    bestEULER, bestmatchingrates = filterMatrix_MinimumRate(bestEULER,
                                                            bestmatchingrates,
                                                          MatchingRate_Threshold)

    if verbose:
        print("After filtering (cubic permutation, matching threshold %.2f)" % MatchingRate_Threshold)
        print("%d matrices remain\n" % len(bestEULER))

    return bestEULER, bestmatchingrates


def filterEquivalentMatrix(eulers_array, infos, verbose=0):
    """
    remove redundant matrices corresponding to each 3 euler angles
    and keep associated data contained in infos (1d array)

    return: unique matrices array and corresponding scalar info
    """
    # remove equivalent matrices
    mats = []
    for eul in eulers_array:
        mats.append(GT.fromEULERangles_toMatrix(eul))

    scores = list(range(len(mats)))
    mats_out, pos_mat = RemoveDuplicatesOrientationMatrix(np.array(mats),
                                                          scores,
                                                        tol=0.1,
                                                        allpermu=None,
                                                        OutputMatricesOnly=0)

    if verbose:
        # index to keep
        #    print "pos_mat", pos_mat
        #    print "from matrices"
        #    for k, mat in enumerate(mats):
        #        print k
        #        printmatrix(mat)
        print("keep matrices")
        for k, mat in enumerate(mats_out):
            print(pos_mat[k])
            printmatrix(mat)

    eulers_array = np.take(eulers_array, pos_mat, axis=0)
    infos = np.take(infos, pos_mat)

    return eulers_array, infos


def filterMatrix_MinimumRate(eulers, matchingrates, MatchingRate_Threshold):
    """
    keep euler angles where corresponding rate in bestmatchingrates
    is higher than MatchingRate_Threshold (in percent)
    """
    # matrices that have Matching rate higher than MatchingRate_Threshold
    ind_high_rate = np.where(matchingrates >= MatchingRate_Threshold)[0]

    eulers = np.take(eulers, ind_high_rate, axis=0)
    matchingrates = np.take(matchingrates, ind_high_rate)

    return eulers, matchingrates


def getallcubicMatrices(mat):
    """
    return all equivalent matrices of mat from cubic symetry
    """
    allpermu = DictLT.OpSymArray
    return np.transpose(np.dot(mat, np.transpose(allpermu)), axes=(2, 0, 1))


def comparematrices(matA, matB, tol=0.001, allpermu=None):
    """
    return True if A==B   i.e. if A*S =B with S symetry operator (axes permutation and rotation)
    matrix comparaison is done elementwise within tolerance

    matA and matB are single matrix
    """
    if allpermu is None or allpermu == 'cubic':
        allpermu = DictLT.OpSymArray

    # diff between one matrix and an array of matrices
    # print "matA",matA
    # print "matA.type",type(matA)
    # print "matB",matB
    # print "matB.type",type(matB)
    # print np.transpose(allpermu).shape
    # print np.dot(matB,np.transpose(allpermu)).shape
    diff = np.transpose(np.dot(matB, np.transpose(allpermu)), axes=(2, 0, 1)) - matA

    tol = tol * np.ones((3, 3))

    flagdiff = np.less(np.abs(diff), tol)

    Shape = allpermu.shape

    if len(Shape) == 2:
        ny = Shape[1]
    elif len(Shape) == 3:
        ny = Shape[1] * Shape[2]

    resflag = np.all(flagdiff.reshape(Shape[0], ny), axis=1)

    return np.any(resflag), resflag


def AreTwinned(matA, matB, tol=0.001, allpermu=None):
    """
    return True if A==B   i.e. if A*S =B with S symetry operator corresponding to twins
    matrix comparaison is done elementwise within tolerance

    matA and matB are single matrix
    """
    if allpermu in (None, 'sigma3'):
        dictVect = DictLT.dict_Vect
        allpermu = np.array([dictVect[key_operator] for \
                             key_operator in list(dictVect.keys()) if key_operator[:6] == 'sigma3'])

    # diff between one matrix and an array of matrices
    # print "matA",matA
    # print "matA.type",type(matA)
    # print "matB",matB
    # print "matB.type",type(matB)
    # print np.transpose(allpermu).shape
    # print np.dot(matB,np.transpose(allpermu)).shape
    diff = np.transpose(np.dot(matB, np.transpose(allpermu)), axes=(2, 0, 1)) - matA

    tol = tol * np.ones((3, 3))

    flagdiff = np.less(np.abs(diff), tol)

    Shape = allpermu.shape

    if len(Shape) == 2:
        ny = Shape[1]
    elif len(Shape) == 3:
        ny = Shape[1] * Shape[2]

    resflag = np.all(flagdiff.reshape(Shape[0], ny), axis=1)

    return np.any(resflag), resflag


def RemoveDuplicatesOrientationMatrix(matrices, scores,
                                      tol=0.0001,
                                      allpermu=None,
                                      OutputMatricesOnly=0):
    """
    remove duplicates matrix in the sense of comparematrices()
    """

    if len(matrices) == 1:
        return matrices

    BSM = matrices.tolist()

    # print "len(BSM)",len(BSM)

    if allpermu is None:
        print("**** -- Loading default cubic permutations!  --****")
        allpermu = DictLT.OpSymArray

    FilteredMatrixList = []
    FilteredScoreList = []

    Dict_mat = {}
    for k, elem in enumerate(BSM):
        Dict_mat[k] = elem

    # filtering loop
    k = 0
    while(BSM):

        def Matrixcomparewith(m):
            """
            Return False if m == BSM[0] in the sense of comparematrices()
            """
            return not comparematrices(BSM[0], m,
                                       tol=tol,
                                       allpermu=allpermu)[0]

        FilteredMatrixList.append(BSM[0])
        BSM = [m for m in BSM if Matrixcomparewith(m)]

        k += 1

    if OutputMatricesOnly:
        return FilteredMatrixList
    else:
        # updating scores list
        for mat in FilteredMatrixList:
            FilteredScoreList.append(scores[list(Dict_mat.values()).index(mat)])

        return FilteredMatrixList, FilteredScoreList


def MergeSortand_RemoveDuplicates(OrientMatrices, Scores,
                                  threshold_matching,
                                  tol=0.0001,
                                  keep_only_equivalent=True):
    """
    Returns: Best Sorted Non equivalent orientation matrix

    1)Merge matrices solution of distance recognition in LUT and matrices coming from user (previous results)

    2)Sort anf threshold according to Scores

    3)Remove Duplicates (taking into account symetry operators)

    threshold_matching : first column of Scores
    keep_only_equivalent         : True all permutation of axes, 
                                0, None or False    all matrices even duplicates
    """
    ar_hhh = []
    for elem in Scores:
        ar_hhh.append(elem[:3])
    ar_hhh = np.array(ar_hhh)

    # threshold
    ind_sup = np.where(ar_hhh[:, 0] >= threshold_matching)[0]

    ar_hhh = np.take(ar_hhh, ind_sup, axis=0)
    OrientMatrices = np.take(OrientMatrices, ind_sup, axis=0)

    # print "ar_hhh",ar_hhh
    # print "len(OrientMatrices)",len(OrientMatrices)

    # sort
    colnbmatch = np.array(ar_hhh[:, 0], dtype=np.uint16)
    colres = np.array(ar_hhh[:, 2] * 10000, dtype=np.uint16)
    # hint: lexsort sort lexicographically starting with last key and then second-to-last key!!
    rank = np.lexsort(keys=(-colres, colnbmatch))[::-1]

    print("sorting according to rank")
    print('rank', rank)

    Bestsortedmatrices = np.take(OrientMatrices, rank, axis=0)
    Besthhh = np.take(ar_hhh, rank, axis=0)

    # print "len(OrientMatrices)",len(OrientMatrices)

    # --- we may want to select only matrices above a threshold
    # load list of symetry operators
    if keep_only_equivalent == True:
        print("keep_only_equivalent cubic matrices")
        allpermu = DictLT.OpSymArray
        # remove duplicates
        (NonEquivalentMatrices,
         FilteredScores) = RemoveDuplicatesOrientationMatrix(Bestsortedmatrices,
                                                             Besthhh,
                                                            tol=tol,
                                                            allpermu=allpermu)
        MATRICES = NonEquivalentMatrices
        SCORES = FilteredScores
    elif keep_only_equivalent in (0, None, False):
        MATRICES = Bestsortedmatrices
        SCORES = Besthhh

    return MATRICES, SCORES


#--- -----------  Orientation Matrix handling in map
def findtwins(imageindex, dmat,
              grainindex=0, nbgrains_per_image=1,
              imagerange=(1708, 3323), tol=0.01):
    """
    find twins of matrix in imageindex and grainindex in other matrices in dict dmat

    input:
    dmat:    dict[key]=val, key= imagefile index  val= one matrix or a list a matrix

    imageindex    :  imagefile index   dmat[imageindex]=matrix for which we search for the twins
    grainindex : index matrix in dmat[imageindex]
    nbgrains_per_image    : nb of matrices (or grains) per image to probe for each image

    tol        : matrix element-wise difference tolerance to consider that two matrices are equal

    return:
    list of imagefile index and matrix index where twins are in dmat
    """
    testmatrix = dmat[imageindex][grainindex]
    if testmatrix == 0:
        print("testmatrix is not a matrix !!")
        return []

    if nbgrains_per_image > len(dmat[imageindex]):
        print("nbgrains_per_image argument is not compatible with nb of matrices per image in dmat ")
        return []

    twins = []
    nbtwins = 0
    for fileindex in range(imagerange[0], imagerange[1] + 1):
        for matrix_grain_ind in range(0, nbgrains_per_image):
            probed_matrix = dmat[fileindex][matrix_grain_ind]
            if probed_matrix != 0:
                if AreTwinned(testmatrix, probed_matrix, tol=tol, allpermu=None)[0]:
                    print("Got twins !: [%d,%d] with [%d,%d] " % (imageindex, grainindex, fileindex, matrix_grain_ind))
                    nbtwins += 1
                    twins.append(fileindex)

    print("Found %d twins" % nbtwins)
    return twins





def findsimilarmatrix(imageindex, dmat, grainindex=0, imagerange=(1708, 3323), tol=0.001):
    """
    find similar matrices in dmat to matrix given by imageindex

    """
    simil = []
    nbsimil = 0
    for k in range(imagerange[0], imagerange[1] + 1):
        if comparematrices(dmat[imageindex][grainindex], dmat[k][0], tol=tol, allpermu=None)[0]:
            print("Got similar matrices !: [%d,%d]" % (imageindex, k))
            nbsimil += 1
            simil.append(k)

    print("Found %d similar matrices" % nbsimil)
    return simil


def FindMatrices(dmat, tol=0.01):
    """
    Find sets of similar matrices in dmat

    dmat        :  dictionnary of matrix dict[fileindex][0]=matrix from mapping

    return:

    key: fileindex  val: all fileindices having the same matrix as fileindex
    missing fileindex key are singleton
    """
    if len(dmat) == 1:
        return dmat.vals()[0]

    # TODO: can be improved by limiting the probed fileindex in the vicinity of a starting fileindex
    # In the present case: starting fileindex is the first image of the dataset (mapping corner)
    # and all images are probed
    # this is really eshaustive and time consuming
    sfileindex = sorted(dmat.keys())
    firstindex = sfileindex[0]
    lastindex = sfileindex[-1]

    # dict of matrix in mapping
    goodpairs = []
    for fileindex in range(firstindex, lastindex):
        mat = dmat[fileindex][0]
        if (fileindex % 100) == 0:
            print("fileindex ", fileindex)

        for fileindex2 in range(fileindex + 1, lastindex + 1):

            if comparematrices(mat, dmat[fileindex2][0], tol=tol)[0]:

                goodpairs.append([fileindex, fileindex2])

    res_final, res_dict = GT.Set_dict_frompairs(np.array(goodpairs), verbose=0)
    return res_final

def mergesimilarmatrices(dmat):

    import pickle
#    rr= rFindMatrices(dmat)  # this takes a while
#    '/home/micha/LaueProjects/CuVia/Carto'
    f = open('dictsimilmat')
    matrix_sets_dict = pickle.load(f)
    f.close()

    # modify dmat according to rr
    dmatm = copy.deepcopy(dmat)
    for key in matrix_sets_dict:
        for val in matrix_sets_dict[key]:
            dmatm[val][0] = dmat[key][0]
    return dmatm


#--- ------------ Simulate fake data
def IofE(Energy, Amplitude=50000, powerp=-1, offset=-5):
    """
    returns simulated intensity of bragg spot as a function of Energy only
    """
    return 1. * Amplitude * np.power(Energy + offset, powerp * 1.)


def Grain_to_string(grains):
    """
    convert each 4 elements grain parameters contained in grains list to list of string
    """
    liststr = ['mat, EXTINCTION, orientmat, ELEMENT']
    for k, grain in enumerate(grains):
        mat, EXTINCTION, orientmat, ELEMENT = grain

        matstr = matrix_to_string(mat)
        orientmatstr = matrix_to_string(orientmat)

        liststr.append('Grain G % d' % k)
        for elem in matstr:
            liststr.append(elem)
        liststr.append(EXTINCTION)
        for elem in orientmatstr:
            liststr.append(elem)
        liststr.append(ELEMENT)

    return liststr


def printmatrix(mat, decimals=3):
    """
    confortable print of matrix
    """
    for row in np.round(np.array(mat), decimals=decimals).tolist():
        print(row)


def matrix_to_string(mat):
    """
    convert the 3 rows to a list of string
    """
    liststr = []
    try:
        mat = mat.tolist()
    except AttributeError:
        pass
    for row in mat:
        liststr.append(str(row))

    return liststr


def get4sigma3Matrices(mat):
    """
    from mat return 4 sigma3 matrices (180 deg rotation around 111 family plane)
    """
    listmatsigma = [DictLT.dict_Vect['sigma3_' + str(ind)] for ind in (1, 2, 3, 4)]
    listmat = []
    for elem in listmatsigma:
        listmat.append(np.dot(mat, elem))

    return listmat


def createFakeData(key_material, nbgrains, emin=5, emax=25,
                   outputfilename=None, removespots=None, addspots=None, twins=0):
    """
    create fake data with nbgrains grains and random detecgtor parameters

    removespots    : nb of spots to remove (randomly) in each grains
    addspots        : number of random experimental spots to add
    """
    if not outputfilename:
        outputfilename = 'testSirandom'
        print("%s.cor has be written" % outputfilename)
    #    detectorparameters = [70, 1024, 1024, 0, 0]
    detectorparameters = IMM.randomdetectorparameters()

    # generate .cor file
    simul_matrices = []
    grains = []
    Twicetheta, Chi, posx, posy, dataintensity = [], [], [], [], []

    EXTINCTION = DictLT.dict_Materials[key_material][2]

    if removespots is not None:
        if len(removespots) != nbgrains:
            raise ValueError("removespots needs as many integers than nbgrains!")

    for k in range(nbgrains):
        orientmat = GT.randomRotationMatrix()
        simul_matrices.append(orientmat)

    if twins >= 1:
        if twins <= nbgrains / 2:
            copysimul_matrices = copy.copy(simul_matrices)
            for twin in range(0, twins + 1, 2):
                print("twin", twin)
                simul_matrices[twin + 1] = np.dot(np.dot(copysimul_matrices[twin], DictLT.dict_Vect['sigma3_1']),
                                                  GT.matRot([1, 2, 3], 0))
        else:
            raise ValueError("nb of twins must not be higher than half of nbgrains")

    for k, orientmat in enumerate(simul_matrices):

        grains.append([np.eye(3), EXTINCTION, orientmat, key_material])

        print("orientmat", orientmat)

        # Twicetheta, Chi, Miller_ind, posx, posy, Energy
        (Twicetheta1, Chi1,
         Miller_ind1, posx1, posy1, Energy1) = LAUE.SimulateLaue(grains[k],
                                                                 emin, emax,
                                                                 detectorparameters,
                                                                removeharmonics=1)

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
        Twicetheta, Chi, tokeep = GT.removeClosePoints_2(Twicetheta, Chi, dist_tolerance=0.01)
        posx = np.take(posx, tokeep)
        posy = np.take(posy, tokeep)
        dataintensity = np.take(dataintensity, tokeep)

    if addspots is not None:

        posx1 = (2048) * np.random.random_sample(addspots)
        posy1 = (2048) * np.random.random_sample(addspots)
        dataintensity1 = (65000) * np.random.random_sample(addspots)

        from find2thetachi import calc_uflab

        Twicetheta1, Chi1 = calc_uflab(posx1, posy1, detectorparameters)

        Twicetheta = np.concatenate((Twicetheta, Twicetheta1))
        Chi = np.concatenate((Chi, Chi1))
        posx = np.concatenate((posx, posx1))
        posy = np.concatenate((posy, posy1))
        dataintensity = np.concatenate((dataintensity, dataintensity1))

    # write fake .cor file
    strgrains = ['simulated pattern with two grains']
    strgrains += Grain_to_string(grains)

    # sorting data by decreasing intensity:
    intensitysortedind = np.argsort(dataintensity)[::-1]
    arraydata = np.array([Twicetheta, Chi, posx, posy, dataintensity]).T

    sortedarraydata = arraydata[intensitysortedind]

    Twicetheta, Chi, posx, posy, dataintensity = sortedarraydata.T

    IOLT.writefile_cor(outputfilename, Twicetheta, Chi, posx, posy, dataintensity,
                                                        param=detectorparameters,
                                                        comments=strgrains)

    fullname = outputfilename + '.cor'
    print("Fake data have been saved in %s" % fullname)
    return fullname


#--- ----------  old non OOP methods
def rawMultipleIndexation(bestEULER, indexed_spots_dict,
                          ELEMENT, detectorparameters,
                          AngleTol=1., emax=25):
    """
    from a list of 3euler angles index exp spot data
    if matchingrate in 2theta,chi space is higher than MatchingRate_Threshold (in percent)

    input:
    bestEULER             : array of 3 angles
    indexed_spots_dict    : non empty dictionary of exp. spots to be indexed
    AngleTol (deg)        : angular tolerance to accept a link between exp. and theo. spots
    """
    MINIMUM_SPOTS = 8
    if len(indexed_spots_dict) < MINIMUM_SPOTS:
        print("too few spots to index")
        return None

    exp_data = getSpotsData(indexed_spots_dict).T
    ind, twicetheta_data, chi_data, posx, posy, intensity_data = exp_data

    dict_grain_matrix = {}
    dict_grain_matchinrate = {}
    index_foundgrain = 0
    print("Indexing the data according to the matrices")
    # loop over matrices from image matching
    for k, eulers in enumerate(bestEULER):
        # handle spot indexation for one grain

        matrix = GT.fromEULERangles_toMatrix(eulers)

        indexation_res, nbtheospots = getIndexedSpots(matrix,
                           (twicetheta_data, chi_data, intensity_data),
                           ELEMENT,
                           detectorparameters,
                           removeharmonics=0,  # for fast computations
                           veryclose_angletol=AngleTol,
                           emin=5,
                           emax=emax,
                           detectordiameter=165.,
                           verbose=0)

        if indexation_res is not None:
            indexed_spots_dict, nb_updates = updateIndexationDict(indexation_res,
                                                              indexed_spots_dict,
                                                              index_foundgrain,
                                                              overwrite=0)

            print("nb of indexed spots for this matrix # %d: %d / %d (theo. nb)" % (k, nb_updates, nbtheospots))
            print("with tolerance angle : %.2f deg" % AngleTol)
            dict_grain_matrix[index_foundgrain] = matrix
            dict_grain_matchinrate[index_foundgrain] = [nb_updates,
                                                        100.*nb_updates / nbtheospots]

            print("Low value of indexed spots may mean that spots have been previously assigned to previous grain")

            index_foundgrain += 1

    print("%d grain(s) has(ve) been found" % index_foundgrain)

    if index_foundgrain == 0:
        return None, None, None
    return indexed_spots_dict, dict_grain_matrix, dict_grain_matchinrate


def updateIndexationDict(indexation_res, indexed_spots_dict, grain_index, overwrite=0):
    """
    update dictionary "indexed_spots_dict" of experimental spots of family "grain_index"
    according to indexation info in "indexation_res"
    (unambiguous links between one exp. and one theo. spots

    grain_index        : index of grain corresponding to the pairs
    overwrite        : 1 ,prior to spots properties update, reset (unindex)
                        all spots that were considered (maybe wrongly)to
                        belong to family "grain_index"
    """
    links_dict = indexation_res[0]

    if overwrite:
        indexed_spots_dict = resetSpotsFamily(indexed_spots_dict, grain_index)

    nb_updates = 0
    for link in list(links_dict.values()):
        exp_index, theo_index, miller_indices, energy = link
#        print "link", link

        try:
            theo_index = int(theo_index[0])
        except:
            theo_index = int(theo_index)

        # this exp spot has not been already indexed
        if indexed_spots_dict[exp_index][-1] != 1:
            # keep the spot data and add theo info

            indexed_spots_dict[exp_index] = indexed_spots_dict[exp_index][:6] + \
                                                                    [miller_indices,
                                                                    energy,
                                                                    grain_index,
                                                                    1]
            nb_updates += 1

#     print "%d spots have been indexed for grain #%d" % (nb_updates, grain_index)

    return indexed_spots_dict, nb_updates


def SpotLinks(twicetheta_exp, chi_exp, dataintensity_exp,  # experimental data
            veryclose_angletol,  # tolerance angle
            twicetheta, chi, Miller_ind, posx, posy, energy,  # theoretical data
            absoluteindex=None,
            verbose=0):
    """
    Create automatically links between currently close experimental and theoretical spots
    in 2theta, chi representation

    return table of association with theo and exp spots indices.

    if absoluteindex = None    : spots indices and order are those of data
                                    (twicetheta_exp, chi_exp, dataintensity_exp)
                    = list_of_absolute_indices
                            : list containing the absolute indices


    veryclose_angletol     :    max tolerance angle for association in degree
    absoluteindex        : list of absolute exp spot index to output in res
                            (conversion table between relative and absolute index)
                           i.e.  absoluteindex[localindex] = absolute index

    TODO: to improve, always difficult to understand immediately the algorithm :-[
    """

    if verbose:
        print("\n ************** Spots Association ************** \n")

    Resi, ProxTable = INDEX.getProximity(np.array([twicetheta, chi]),  # warning array(2theta, chi)
                                    twicetheta_exp / 2., chi_exp,  # warning theta, chi for exp
                                    proxtable=1, angtol=veryclose_angletol,
                                    verbose=0,
                                    signchi=1)[:2]  # sign of chi is +1 when apparently SIGN_OF_GAMMA=1

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

    if len(List_Exp_spot_close) == 0:
        print("Found no pair within tolerance in SpotLinks()")
        return 0

    elif len(List_Exp_spot_close) == 1:
        print("Just a single found! Nb of pairs equal to 1  is not implemented yet in SpotLinks()")
        return 0

    #--------------------------------------------------------------
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
        ambiguous_exp_ind = GT.find_closest(np.array(sorted_LESC[toremoveindex],
                                                     dtype=float),
                                            arrayLESC, TOLERANCEANGLE)[1]
        # print "ambiguous_exp_ind", ambiguous_exp_ind

        # tagging (inhibiting) exp spots
        # that belong ambiguously to several simulated grains
        for exp_ind in ambiguous_exp_ind:
            Miller_Exp_spot[exp_ind] = None
            Energy_Exp_spot[exp_ind] = 0.0

    # -----------------------------------------------------------------------------------------------------
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
    for theo_ind in range(len(List_Exp_spot_close)):

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
                    refine_indexed_spots[exp_index] = [exp_index,
                                                       theo_index[closest_theo_ind],
                                                       Miller_Exp_spot[theo_ind],
                                                       Energy_Exp_spot[theo_ind]]

        # there is already exp_index in singleindices
        else:
            # do not update the dictionary 'refine_indexed_spots'
            if verbose:
                print("Experimental spot #%d may belong to several theo. spots!" % exp_index)
            pass
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

            listofpairs.append([absoluteexpspotindex, theo_id])  # Exp, Theo,  where -1 for specifying that it came from automatic linking
            linkExpMiller.append([float(absoluteexpspotindex)] + [float(elem) for elem in Miller_id])  # float(val) for further handling as floats array
            linkResidues.append([absoluteexpspotindex, theo_id, Resi[theo_id]])
            linkEnergy.append([absoluteexpspotindex, theo_id, Energy_id])
            linkIntensity.append(dataintensity_exp[exp_id])
            # Dataxy.append([ LaueToolsframe.data_pixX[val[0]], LaueToolsframe.data_pixY[val[0]]])

    linkedspots_link = np.array(listofpairs)
    linkExpMiller_link = linkExpMiller
    linkIntensity_link = linkIntensity
    linkResidues_link = linkResidues
    linkEnergy_link = linkEnergy
    fields = ['#Spot Exp', '#Spot Theo', 'h', 'k', 'l', 'Intensity', 'residues (deg)']

    # self.Data_X, self.Data_Y = np.transpose( np.array(Dataxy) )

    return (refine_indexed_spots,
            linkedspots_link,
            linkExpMiller_link,
            linkIntensity_link,
            linkResidues_link,
            linkEnergy_link,
            fields)


def getIndexedSpots(OrientMatrix, exp_data, key_material, detectorparameters,
                    useabsoluteindex=None,
                       removeharmonics=0,
                       veryclose_angletol=1.,
                       emin=5,
                       emax=25,
                       detectordiameter=None,
                       verbose=0):
    """
    return links between experimental and theoretical spots
    (simulated from a grain with OrientMatrix, key_material)

    useabsoluteindex        : array of conversion from relative spot index to absolute one
                            useabsoluteindex[relative]= absolute. When exp_data have been
                            extracted from a bigger data set.
    """

    # experimental data
    twicetheta_data, chi_data, dataI = exp_data

    # simulated data
    grain = CP.Prepare_Grain(key_material, OrientMatrix=OrientMatrix)
    (Twicetheta, Chi,
     Miller_ind, posx, posy, Energy) = LAUE.SimulateLaue(grain,
                                       emin, emax,
                                       detectorparameters,
                                       removeharmonics=removeharmonics,
                                       detectordiameter=detectordiameter * 1.25)

    nb_of_simulated_spots = len(Twicetheta)

    res = SpotLinks(twicetheta_data, chi_data, dataI,  # experimental data
                    veryclose_angletol,  # tolerance angle
                    Twicetheta, Chi, Miller_ind, posx, posy, Energy,
                    absoluteindex=useabsoluteindex)

    if res != 0:
        return res, nb_of_simulated_spots
    else:
        return None, None


def getUnIndexedSpots(indexed_spots_dict):
    """
    read dictionary of spots and return data of spots not yet indexed
    """
    unindexed_spots_indices = []
    for key_spot in sorted(indexed_spots_dict.keys()):
        if indexed_spots_dict[key_spot][-1] == 0:
            unindexed_spots_indices.append(key_spot)

    return unindexed_spots_indices


def getSpotsFamily(indexed_spots_dict, grain_index):
    """
    return spots that belong to the same grain
    """
    spots_set = []
    for key_spot in sorted(indexed_spots_dict.keys()):
        # spot has been indexed
        if indexed_spots_dict[key_spot][-1] == 1:
            if indexed_spots_dict[key_spot][-2] == grain_index:
                spots_set.append(key_spot)

    return spots_set


def resetSpotsFamily(indexed_spots_dict, grain_index):
    """
    reset spots that belong to the grain 'grain_index'
    """
    for key_spot in sorted(indexed_spots_dict.keys()):
        # spot has been indexed
        if indexed_spots_dict[key_spot][-1] == 1:
            if indexed_spots_dict[key_spot][-2] == grain_index:
                indexed_spots_dict[key_spot] = indexed_spots_dict[key_spot][:6] + [0]

    return indexed_spots_dict


def getSpotsData(indexed_spots_dict):
    """
    return all data of experimental spots
    array where columns are:
    absolute spot index, tth, chi, posX, posY, intensity
    """
    data = []
    for key_spot in sorted(indexed_spots_dict.keys()):
        index, tth, chi, posX, posY, intensity = indexed_spots_dict[key_spot][:6]
        data.append([index, tth, chi, posX, posY, intensity])

    return np.array(data)


def isindexed(spot_index, indexed_spots_dict):
    """
    return grain index if spot is indexed, otherwise return None
    """
#    if spot_index not in indexed_spots_dict.keys():
#        raise KeyError, "this spots index '%s' doesn't exist in spots dictionary" % str(spot_index)
    if indexed_spots_dict[spot_index][-1] == 1:
        return indexed_spots_dict[spot_index][-2]
    else:
        return None


def getUnIndexedSpotsData(indexed_spots_dict, grain_indices):
    """
    return all data of experimental spots that belong to grain in grain_indices
    and unindexed one

    return array where columns are:
    absolute spot index, tth, chi, posX, posY, intensity
    """

    if not isinstance(grain_indices, list):
        grain_indices = [grain_indices]
    data = []
    # TODO: to simplify
    for key_spot in sorted(indexed_spots_dict.keys()):
        grain_origin = isindexed(key_spot, indexed_spots_dict)
        if grain_origin:
            if grain_origin in grain_indices:
                index, tth, chi, posX, posY, intensity = indexed_spots_dict[key_spot][:6]
                data.append([index, tth, chi, posX, posY, intensity])
        else:
            index, tth, chi, posX, posY, intensity = indexed_spots_dict[key_spot][:6]
            data.append([index, tth, chi, posX, posY, intensity])

    return np.array(data)


def getSpotsFamilyData(indexed_spots_dict, grain_index, onlywithMiller=1):
    """
    return all data of experimental spots for one grain
    """
    data = []
    for key_spot in sorted(indexed_spots_dict.keys()):
        # spot has been indexed
        if indexed_spots_dict[key_spot][-1] == 1:
            if indexed_spots_dict[key_spot][-2] == grain_index:
                (index,
                 tth, chi,
                 posX, posY,
                 intensity, Miller, Energy) = indexed_spots_dict[key_spot][:8]
                # in ambiguous case, spot has not been assigned Miller indices
                if onlywithMiller == 1:
                    if Miller is not None:
                        H, K, L = Miller
                        data.append([index, tth, chi, posX, posY, intensity,
                                     H, K, L, Energy])

    return np.array(data)


def Angular_residues_np(test_Matrix, twicetheta_data, chi_data,
                    ang_tol=0.5,
                    key_material='Si',
                    emin=5, emax=25,
                    ResolutionAngstrom=False,
                    detectorparameters=None):

    """ Simulate Laue pattern (single grain) and
    Compute angular residues between pairs of close exp. and theo. spots 
    
    USED in manual indexation
    Used in AutoIndexation module

    inputs: 
    twicetheta_data, chi_data  : experimental coordinates of scattered beams or spots (kf vectors)
    test_Matrix                : orientation matrix
    ang_tol                    : angular tolerance in deg to accept or reject a exp. and theo pair

    detectorparameters  : dictionary of detector parameters (key, value)
                            'kf_direction' , general position of detector plane
                            'detectordistance', detector distance (mm)
                            'detectordiameter', detector diameter (mm)
    """
    if detectorparameters is None:
        # use default parameter
        kf_direction = 'Z>0'
        detectordistance = 70.
        detectordiameter = 165.
        pixelsize = 165. / 2048
        dim = (2048, 2048)
    else:
        kf_direction = detectorparameters['kf_direction']
        detectordistance = detectorparameters['detectorparameters'][0]
        detectordiameter = detectorparameters['detectordiameter']
        pixelsize = detectorparameters['pixelsize']
        dim = detectorparameters['dim']

#     print "kf_direction,pixelsize,detectordistance", kf_direction, pixelsize, detectordistance

    # ---simulation
    # print "Reference Element or structure label", key_material

    # spots2pi = generalfabriquespot_fromMat_veryQuick(CST_ENERGYKEV/emax,CST_ENERGYKEV/emin,[grain],1,fastcompute=1,fileOK=0,verbose=0)
    grain = CP.Prepare_Grain(key_material, OrientMatrix=test_Matrix)

    # array(vec) and array(indices) (here with fastcompute=0 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin,
                                                        [grain], 1,
                                                        fastcompute=1,
                                                        fileOK=0, verbose=0,
                                                        kf_direction=kf_direction,
                                                        ResolutionAngstrom=ResolutionAngstrom
                                                        )
    
    # 2theta,chi of spot which are on camera (with harmonics)
    # None because no need of hkl vectors
    # TwicethetaChi without energy calculations and hkl selection
    # without use of spots instantation (faster)
    TwicethetaChi = LAUE.filterLaueSpots_full_np(spots2pi[0][0], None,
                                                 HarmonicsRemoval=0,
                                                 fileOK=0, fastcompute=1,
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
    return matchingrate.getProximity(TwicethetaChi,
                              twicetheta_data / 2., chi_data,
                              angtol=ang_tol, proxtable=0)
    
    #     return INDEX.getProximity(TwicethetaChi,
#                               twicetheta_data / 2., chi_data,
#                               angtol=ang_tol, proxtable=0)


def Angular_residues(test_Matrix, twicetheta_data, chi_data,
                    ang_tol=0.5,
                    key_material='Si',
                    emin=5, emax=25,
                    ResolutionAngstrom=False,
                    detectorparameters=None):

    """ Simulate Laue pattern and
    Compute angular residues between pairs of close exp. and theo. spots 

    inputs: 
    twicetheta_data, chi_data  : experimental coordinates of scattered beams or spots (kf vectors)
    test_Matrix                : orientation matrix
    ang_tol                    : angular tolerance in deg to accept or reject a exp. and theo pair

    detectorparameters  : dictionary of detector parameters (key, value)
                            'kf_direction' , general position of detector plane
                            'detectordistance', detector distance (mm)
                            'detectordiameter', detector diameter (mm)
    """
    if detectorparameters is None:
        # use default parameter
        kf_direction = 'Z>0'
        detectordistance = 70.
        detectordiameter = 165.
        pixelsize = 165. / 2048
        dim = (2048, 2048)
    else:
        kf_direction = detectorparameters['kf_direction']
        detectordistance = detectorparameters['detectorparameters'][0]
        detectordiameter = detectorparameters['detectordiameter']
        pixelsize = detectorparameters['pixelsize']
        dim = detectorparameters['dim']

#     print "kf_direction,pixelsize,detectordistance", kf_direction, pixelsize, detectordistance

    # ---simulation
    # print "Reference Element or structure label", key_material

    # spots2pi = generalfabriquespot_fromMat_veryQuick(CST_ENERGYKEV/emax,CST_ENERGYKEV/emin,[grain],1,fastcompute=1,fileOK=0,verbose=0)
    grain = CP.Prepare_Grain(key_material, OrientMatrix=test_Matrix)

    # array(vec) and array(indices) (here with fastcompute=0 array(indices)=0) of spots exiting the crystal in 2pi steradian (Z>0)
    spots2pi = LAUE.getLaueSpots(CST_ENERGYKEV / emax, CST_ENERGYKEV / emin,
                                                        [grain], 1,
                                                        fastcompute=1,
                                                        fileOK=0, verbose=0,
                                                        kf_direction=kf_direction,
                                                        ResolutionAngstrom=ResolutionAngstrom
                                                        )
    # 2theta,chi of spot which are on camera (with harmonics)
    TwicethetaChi = LAUE.filterLaueSpots(spots2pi, fileOK=0, fastcompute=1,
                                         kf_direction=kf_direction,
                                         detectordistance=detectordistance,
                                         detectordiameter=detectordiameter,
                                         pixelsize=pixelsize,
                                         dim=dim)

#     print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
#     print "len(TwicethetaChi[0])", len(TwicethetaChi[0])
    if len(TwicethetaChi[0]) == 0:
        return None

    return INDEX.getProximity(TwicethetaChi,
                              twicetheta_data / 2., chi_data,
                              angtol=ang_tol, proxtable=0)


def getMatchingRate(indexed_spots_dict, test_Matrix, ang_tol,
                    simulparam, removeharmonics=1,
                    detectordiameter=165.):
    """
    get matching rate for an orientation matrix
                    for all exp. data stored in dictionary indexed_spots_dict

    input:
    indexed_spots_dict    : dict of exp spots (indexed or not)
    test_Matrix            : orientation matrix (3x3)
    ang_tol                : angular tolerance below which a pair between
                            exp. and simulated spots is accepted
    simulparam            : tuple containing:
                    emin, emax, key_material, detectorparameters
    removeharmonics        : 0 or 1 to compute respectively without or with harmonics in simulated pattern

    return:
    [0]             number of pairs between exp and theo spots
    [1]            number of theo. spots
    [2]              matching rate (ratio*100 of the two previous numbers)
    """
    emin, emax, key_material, detectorparameters = simulparam
    # TODO to shorten
    ind, twicetheta_data, chi_data , posx, posy, intensity_data = getSpotsData(indexed_spots_dict).T

    # simulated data
    grain = CP.Prepare_Grain(key_material, OrientMatrix=test_Matrix)
    (Twicetheta, Chi,
     Miller_ind, posx, posy, Energy) = LAUE.SimulateLaue(grain,
                                       emin, emax,
                                       detectorparameters,
                                       removeharmonics=removeharmonics,
                                       detectordiameter=detectordiameter * 1.25)

    nb_of_simulated_spots = len(Twicetheta)

    # find close pairs between exp. and theo. spots
    res = SpotLinks(twicetheta_data, chi_data, intensity_data,  # experimental data
                    ang_tol,  # tolerance angle
                    Twicetheta, Chi, Miller_ind, posx, posy, Energy,
                    absoluteindex=None)

    if res == 0 or len(res[1]) == 0:
        return None, nb_of_simulated_spots, None

    print("res_links", res[1])

    print("nb of exp spots data probed :", len(twicetheta_data))

    nb_of_links = len(res[1])

    matching_rate = 100.*nb_of_links / nb_of_simulated_spots


    return nb_of_links, nb_of_simulated_spots, matching_rate


def getStatsOnMatching(List_of_Angles, twicetheta_data, chi_data, key_material,
                       ang_tol=1.,
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
    """
    kk = 0
    allmatchingrate = []
    for angles_sol in List_of_Angles:

        test_Matrix = GT.fromEULERangles_toMatrix(angles_sol)

        res = Angular_residues(test_Matrix, twicetheta_data, chi_data,
                                     ang_tol=ang_tol,
                                     key_material=key_material,
                                     emin=emin,
                                     emax=emax)

        if res is None:
            continue

        matching_rate = 100.*res[2] / res[3]
        absolute_matching_rate = res[2]
        allmatchingrate.append(matching_rate)

        if verbose:
            print("*"*30)
            print("res for k:%d and angles: [%.1f,%.1f,%.1f]" % (kk, angles_sol[0], angles_sol[1], angles_sol[2]))
    #        print "res",res[2:]
            print("Matching rate ----(in %%):                %.2f " % matching_rate)
            print("Nb of close spot pairs (< %.2f deg) : %d" % (ang_tol, res[2]))
            print("Nb of simulated spots : %d" % res[3])
            print("mean angular residues %.2f deg." % res[4])
            print("highest residues %.2f deg." % res[5])
        kk += 1

    return np.argsort(np.array(allmatchingrate))[::-1], allmatchingrate


def refineUBSpotsFamily(indexed_spots_dict, grain_index,
                        initial_matrix, key_material, detectorparameters,
                        use_weights=1,
                        pixelsize=165. / 2048,
                        dim=(2048, 2048),
                        kf_direction='Z>0'):
    """
    refine UB matrix of spots family
    """
    MINIMUM_LINKS_FOR_FIT = 8

    latticeparams = DictLT.dict_Materials[key_material][1]
    Bmatrix = CP.calc_B_RR(latticeparams)

    data_1grain = getSpotsFamilyData(indexed_spots_dict, grain_index, onlywithMiller=1)

#    print "data_1grain.shape", data_1grain.shape
    if len(data_1grain) >= MINIMUM_LINKS_FOR_FIT:
        index, tth, chi, posX, posY, intensity, H, K, L, Energy = data_1grain.T
    else:
        print("Too few exp. data to fit")
        return None

    Miller = np.array([H, K, L]).T

#    print "Miller", Miller

    nb_pairs = len(index)
    print("Nb of pairs: ", nb_pairs)

    sim_indices = np.arange(nb_pairs)

    if use_weights:
        weights = intensity
    else:
        weights = None

    # fitting procedure for one or many parameters
    initial_values = np.array([1., 1., 0., 0., 0., 0, .0, 0.])
    try:
        allparameters = np.array(detectorparameters.tolist() + [1, 1, 0, 0, 0] + [0, 0, 0])
    except:
        allparameters = np.array(detectorparameters + [1, 1, 0, 0, 0] + [0, 0, 0])

    arr_indexvaryingparameters = np.arange(5, 13)
    # print "\nInitial error--------------------------------------\n"
    print("initial_matrix", initial_matrix)
    from . import FitOrient as FitO
    residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(initial_values,
                                    Miller,
                                    allparameters,
                                    arr_indexvaryingparameters,
                                    sim_indices,
                                    posX, posY,
                                    initrot=initial_matrix,
                                    Bmat=Bmatrix,
                                    pureRotation=0, verbose=1, pixelsize=pixelsize, dim=dim,
                                    weights=weights,
                                    signgam=1,
                                    kf_direction=kf_direction)
#    print "Initial residues", residues
#    print "---------------------------------------------------\n"

    results = FitO.fit_on_demand_strain(initial_values,
                    Miller,
                    allparameters,
                    FitO.error_function_on_demand_strain,
                    arr_indexvaryingparameters,
                    sim_indices,
                    posX, posY,
                    initrot=initial_matrix,
                    Bmat=Bmatrix,
                    pixelsize=pixelsize,
                    dim=dim,
                    verbose=0,
                    weights=weights,
                    signgam=1,
                    kf_direction=kf_direction)

#    print "\n********************\n       Results of Fit        \n********************"
#    print "results", results

    # print "\nFinal error--------------------------------------\n"
    residues, deltamat, newmatrix = FitO.error_function_on_demand_strain(results,
                                    Miller,
                                    allparameters,
                                    arr_indexvaryingparameters,
                                    sim_indices,
                                    posX, posY,
                                    initrot=initial_matrix,
                                    Bmat=Bmatrix,
                                    pureRotation=0, verbose=1,
                                    pixelsize=pixelsize, dim=dim,
                                    weights=weights,
                                    signgam=1,
                                    kf_direction=kf_direction)

    print("newmatrix", newmatrix)
    return newmatrix


def proposeEnergyforMatching(indexed_spots_dict, test_Matrix, ang_tol,
                             simulparam, removeharmonics=1):
    """
    return energy for which the matching rate is the highest

    input:

    same input as for getMatchingRate

    """
    ener_min = 15
    ener_max = 25
    step = 0.5
    RES = []
    for energy in np.linspace(ener_min, ener_max,
                              int((ener_max - ener_min) / step) + 1):
        simulparam[1] = energy
        res = getMatchingRate(indexed_spots_dict, test_Matrix, 0.1,
                              simulparam, removeharmonics=removeharmonics)
        RES.append([energy] + list(res))

    res = np.array(RES)
    optim_energy = res[np.argmax(res[:, 3])][0]

    return optim_energy


def plotgrains_from_indexspots(indexed_spots_dict):
    """ plot grains spots and data from indexed spots dictionary
    """
    isgrain = True
    grain_index = 0

    # 2theta chi from indexed spots
    twicethetaChi = [[]]
    while isgrain:
        spots = getSpotsFamily(indexed_spots_dict, grain_index)
        if spots:
            for key_spot in spots:
                twicethetaChi[grain_index].append(indexed_spots_dict[key_spot][1:3])
            grain_index += 1
            twicethetaChi.append([])
        else:
            isgrain = False

#    print "twicethetaChi", twicethetaChi
    all_tthchi = getSpotsData(indexed_spots_dict)[:, 1:3]
#    print all_tthchi
    nb_of_orientations = grain_index

    fig = figure()

    print("nb_of_orientations to plot", nb_of_orientations)
    if nb_of_orientations == 1:
        codefigure = 111
    if nb_of_orientations == 2:
        codefigure = 211
    if nb_of_orientations in (3, 4):
        codefigure = 221
    if nb_of_orientations in (5, 6):
        codefigure = 321
    if nb_of_orientations in (7, 8, 9):
        codefigure = 331
    index_fig = 0

#    ax.set_xlim((35, 145))
#    ax.set_ylim((-45, 45))

    dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
    nbcolors = len(dicocolor)
    # theo spots
    for i_grain in range(nb_of_orientations):
        ax = fig.add_subplot(codefigure)
        # all exp spots
        scatter(all_tthchi[:, 0], all_tthchi[:, 1],
                  s=40, c='w', marker='o', faceted=True, alpha=.5)

        # simul spots
        tthchi = np.array(twicethetaChi[i_grain])
#        print tthchi
        ax.scatter(tthchi[:, 0], tthchi[:, 1],
               c=dicocolor[(i_grain + 1) % nbcolors], faceted=False)
        if index_fig < nb_of_orientations:
            index_fig += 1
            codefigure += 1

    show()


def plotgrains(dict_mat, key_material, detectorparameters, emax,
               exp_data=None):
    """
    plot grains spots and data from orientmatrix dictionary up to 9

    exp_data        : 2theta, chi ,  ...
    """

    EXTINCTION = DictLT.dict_Materials[key_material][2]

    nb_matrices = 0
    all_tthchi = []
    for key_matrix in sorted(dict_mat.keys()):

        grain = [np.eye(3), EXTINCTION, dict_mat[key_matrix], key_material]

        Twicetheta, Chi, Miller_ind, posx, posy, Energy = LAUE.SimulateLaue(grain, 5, emax,
                                                                        detectorparameters)

        all_tthchi.append([Twicetheta, Chi])

        nb_matrices += 1
        if nb_matrices == 9:
            break

    fig = figure()

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

    dicocolor = {0: 'k', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm'}
    nbcolors = len(dicocolor)

    if exp_data is not None:
        tth_exp, Chi_exp = exp_data[:2]

    # theo spots
    for i_mat in range(nb_matrices):
        ax = fig.add_subplot(codefigure)

        if exp_data is not None:
            # all exp spots
            scatter(tth_exp, Chi_exp,
                      s=40, c='w', marker='o', faceted=True, alpha=.5)

        # simul spots
        ax.scatter(all_tthchi[i_mat][0], all_tthchi[i_mat][1],
               c=dicocolor[(i_mat + 1) % nbcolors], faceted=False)
        if index_fig < nb_matrices:
            index_fig += 1
            codefigure += 1
            if nb_matrices == 9:
                break

    show()


def PrintMatchingResults(bestmat_list, stats_res_list):
    """
    print on stdout  matching results
    """
    print("\n-----------------------------------------")
    print('results:')
#         print bestmat_list, stats_res_list
    print("matrix:                                         matching results")

#     print "bestmat_list, stats_res_list", bestmat_list, stats_res_list
    if len(stats_res_list) == 1 and len(bestmat_list) != 1:
        bestmat_list = [bestmat_list]
    for mat_elem, stat_elem in zip(bestmat_list, stats_res_list):
        mean_angular_residues = stat_elem[2]
        matching_rate = 100.*stat_elem[0] / stat_elem[1]
        for kkk in range(3):
            txt = '%s        ' % str(mat_elem[kkk])
            if kkk == 0:
                txt += 'res: %s %.3f %.2f' % (str(stat_elem[:2]), mean_angular_residues, matching_rate)
            if len(stat_elem) > 3:
                if kkk == 1:
                    txt += 'spot indices %s' % str(stat_elem[3])
                if kkk == 2:
                    txt += 'planes %s' % str(stat_elem[4].tolist())
            print('%s' % txt)
        print('')


#--- -------------------  Mapping visualization
def norme(vec1):
    """
    computes norm of one vector
    from O Robach
    TODO: use better generaltools module
    """
    nvec = np.sqrt(np.inner(vec1, vec1))
    return nvec


def matstarlab_to_matstarlabOND(matstarlab):
    """
    transform matrix in a orthonormalized frame
    (Schmid orthogonalisation procedure)

    From O. Robach
    
    TODO: to be moved to generaltools
    """
    astar1 = matstarlab[:3]
    bstar1 = matstarlab[3:6]
#    cstar1 = matstarlab[6:]

    astar0 = astar1 / norme(astar1)
    cstar0 = np.cross(astar0, bstar1)
    cstar0 = cstar0 / norme(cstar0)
    bstar0 = np.cross(cstar0, astar0)

    matstarlabOND = np.hstack((astar0, bstar0, cstar0)).T

    # print matstarlabOND

    return matstarlabOND


def calc_Euler_angles(mat3x3):
    """
    Calculates unique 3 euler angles representation of mat3x3
    from O Robach
    """
    # mat3x3 = matrix "minimized" in LT lab frame
    # see F2TC.find_lowest_Euler_Angles_matrix
    # phi 0, theta 1, psi 2

    mat = GT.matline_to_mat3x3(matstarlab_to_matstarlabOND(GT.mat3x3_to_matline(mat3x3)))

    RAD = 180.0 / np.pi

    euler = np.zeros(3, float)
    euler[1] = RAD * np.arccos(mat[2, 2])

    if (np.abs(np.abs(mat[2, 2]) - 1.0) < 1e-5):
        # if theta is zero, phi+psi is defined, if theta is pi, phi-psi is defined */
        # put psi = 0 and calculate phi */
        # psi */
        euler[2] = 0.0
        # phi */
        euler[0] = RAD * np.arccos(mat[0, 0])
        if (mat[0, 1] < 0.0):
            euler[0] = -euler[0]
    else:
        # psi */
        toto = np.sqrt(1 - mat[2, 2] * mat[2, 2])  # sin theta - >0 */
        euler[2] = RAD * np.arccos(mat[1, 2] / toto)
        # phi */
        euler[0] = RAD * np.arccos(-mat[2, 1] / toto)
        if (mat[2, 0] < 0.0):
            euler[0] = 360.0 - euler[0]

        # print "Euler angles phi theta psi (deg)"
        # print euler.round(decimals = 3)

        return euler


def myRGB_3(mat):
    """
    propose a RGB (red green blue) vector to represent a matrix
    """
    allpermu = DictLT.OpSymArray
    allpermudet1 = np.array([ kkk for kkk in allpermu if np.linalg.det(kkk) == 1.0 ])

    vec = F2TC.vec_normalTosurface(mat)
    # allvec = np.dot(allpermudet1,vec)
    # allposvec = allvec[np.all(allvec>=0,axis=1)]
    # # pos of row in allposvec where last element is the higher element of each row
    # hpos = np.where(np.argmax(allposvec, axis=1)==2)[0][0]
    # #print hpos
    # return allposvec[hpos].tolist()

    myframe = np.array([[0, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1]])
    invf = np.linalg.inv(myframe)

    # vec expressed in myframe basis
    vec_in_frame = np.dot(invf, vec)

    # find permutation that lead to 3 positive components
    # TODO: do the permutation before expressing the myframe basis ??
    allvec = np.dot(allpermudet1, vec_in_frame)
    allposvec = allvec[np.all(allvec >= 0, axis=1)]

    if len(np.shape(allposvec)) > 1:
        vecmaxfirstcolumn = np.argmax(allposvec[:, 0])
        allposvec = allposvec[vecmaxfirstcolumn]

    return allposvec


def getMisorientation(mat, refAxis=NORMAL_TO_SAMPLE_AXIS, followVector=np.array([0, 0, 1])):
    """
    compute an angle of a reflection / an axis

    # default axis normal to surface tilted by  40 degrees from horizontal
    """

    # angle between mat.G* and refAxis
#    norm = np.sqrt(np.dot(followVector, followVector))
    rotGstar = np.dot(mat, followVector)

    norm_rotGstar = np.sqrt(np.dot(rotGstar, rotGstar))

    angle = np.arccos(np.dot(rotGstar, refAxis) / (norm_rotGstar)) / DEG

    return angle


def Matrix_to_RGB_2(mat):
    """
    from odile robach
    use unique representation of orientation matrix with its euler angles
    """
    normalisation_angles = np.array([180., 360, 180.])
    # normalisation_angles = np.ones(3)

    return calc_Euler_angles(FO.find_lowest_Euler_Angles_matrix(mat)[0]) / normalisation_angles + np.array([0.1, 0.1, 0.1])


def plotindexingMap_rgbs(dmat, dmr, dnb, startindex=1708, mapshape=(16, 101)):
    """
    plot indexation map
    
    dmat, dmr, dnb   :    dicts from big pickled dicts

    default values for CuVia Jul 11 
    """

    nbcol, nblines = mapshape

    rgbs = [[0., 0., 0.]] * (nbcol * nblines)

    # set colors for well indexed grains, black else
    for k in list(dmat.keys()):
        if type(dmat[k][0]) != type(0) and dmr[k][0] > 20. and dnb[k][0] >= 6.:
            # rgbs[k-minindex] = Matrix_to_RGB( Allres[0][k][0] )[0]
            # rgbs[k-minindex] = myRGB_3(Allres[0][k][0])
#            rgbs[k - startindex] = Matrix_to_RGB_2(dmat[k][0])
            rgbs[k - startindex] = myRGB_3(dmat[k][0])

    tabindex = np.arange(nbcol * nblines).reshape((nbcol, nblines))
    tabindex.resize((nblines, nbcol), refcheck=False)

#    print "tabindex", tabindex

#    X = tabindex / 101
#    Y = tabindex % 101

    rgbs = np.array(rgbs)

    # rgbs = np.array(rgbs).clip(min = 0, max = 1)

    # maporient = np.zeros((nbcol,nblines,3))

    # maporient[:,:,0] = X
    # maporient[:,:,1] = Y
    # see numpy for faster computing

    rgbs.resize((nblines, nbcol, 3), refcheck=False)

    fig = figure()
    axes = fig.gca()
    axes.cla()
    axes.imshow(rgbs, interpolation='nearest')

    numrows, numcols, nbcolors = rgbs.shape

    def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                red, green, blue = rgbs[row, col]

                Imageindex = tabindex[row, col]
                dmr[Imageindex + startindex]
#                return 'x=%1.2f, y=%1.2f, col=%1.2f, row=%1.2f ImageIndex: %d' % \
#                                        (x, y, col, row,
#                                         Imageindex + startindex)
                return 'r=%1.3f, g=%1.3f, b=%1.3f, MatchingRate=%.1f, Nbpeaks=%d, ImageIndex: %d' % \
                                        (red, green, blue,
                                         dmr[Imageindex + startindex][0] ,
                                         dnb[Imageindex + startindex][0],
                                         Imageindex + startindex)
            else:
                return ''

    axes.format_coord = format_coord

    show()


def getallMisorientation(dmat, dmr, dnb):
    """
    in dvpt
    """
    zvalue = [np.NaN] * len(dmat)
    listkeys = sorted(dmat.keys())
    for k in listkeys:
        zvalue[k - listkeys[0]] = getMisorientation(dmat[k][0],
                                                       followVector=np.array([0, 0, -1]))

    return zvalue


def plotindexingMap_scalar(dmat, dmr, dnb, startindex=1708, mapshape=(16, 101)):
    """
    plot indexation map

    default values for CuVia Jul 11

    in dvpt
    """

    nbcol, nblines = mapshape
    zvalue = np.zeros(mapshape)

    listkeys = sorted(dmat.keys())
    for k in listkeys:
        if type(dmat[k][0]) != type(0) and dmr[k][0] > 50. and dnb[k][0] >= 40.:
            zvalue[k - listkeys[0]] = getallMisorientation(dmat, dmr, dnb)



#    np.ma.masked_invalid(zvalue)

    tabindex = np.arange(nbcol * nblines).reshape((nbcol, nblines))
    tabindex.resize((nblines, nbcol), refcheck=False)

#    print "tabindex", tabindex

#    X = tabindex / 101
#    Y = tabindex % 101

    zvalue = np.array(zvalue)

    # rgbs = np.array(rgbs).clip(min = 0, max = 1)

    # maporient = np.zeros((nbcol,nblines,3))

    # maporient[:,:,0] = X
    # maporient[:,:,1] = Y
    # see numpy for faster computing

    zvalue.resize((nblines, nbcol), refcheck=False)

    fig = figure()
    axes = fig.gca()
    axes.cla()
    axes.imshow(zvalue, interpolation='nearest')

    numrows, numcols = zvalue.shape

    def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                zz = zvalue[row, col]

                Imageindex = tabindex[row, col]
                dmr[Imageindex + startindex]

                return 'z=%1.3f, MatchingRate=%.1f, Nbpeaks=%d, ImageIndex: %d' % \
                                        (zz,
                                         dmr[Imageindex + startindex][0] ,
                                         dnb[Imageindex + startindex][0],
                                         Imageindex + startindex)
            else:
                return ''

    axes.format_coord = format_coord

    show()


#--- -----------------  Tests
def example_angularresidues_np(ang_tol):
    
    detectorparameters = {}
    detectorparameters['kf_direction'] = 'Z>0'
    detectorparameters['detectorparameters'] = [70, 1024, 1024, 0, -1]
    detectorparameters['detectordiameter'] = 170
    detectorparameters['pixelsize'] = 0.079
    detectorparameters['dim'] = (2048, 2048)
    
    test_Matrix = np.eye(3)
    # random data
    expdata = np.array([[  59.2164, -66.6041],
       [  54.7398, -88.3875],
       [  43.0763, -83.0983],
       [  58.895 , -88.7721],
       [  39.3337, -65.6132],
       [  52.4121, -58.2399],
       [  47.6515, -100.4466],
       [  47.9575, -83.2749],
       [  44.0163, -64.8536],
       [  56.4791, -57.2717],
       [  48.434 , -115.549 ],
       [  52.2976, -101.5099],
       [  28.9731, -75.0536],
       [  53.191 , -83.4718],
       [  49.0293, -64.0107],
       [  41.0373, -47.0408],
       [  52.3338, -117.1192],
       [  37.1386, -98.1813],
       [  58.7918, -83.6928],
       [  54.3839, -63.0704],
       [  55.9659, -41.1327],
       [  39.5682, -118.0816],
       [  56.3583, -118.8114],
       [  41.6482, -74.3025],
       [  49.9809, -43.6082],
       [  53.2592, -130.741 ],
       [  44.4802, -120.2579],
       [  49.9426, -100.9654],
       [  38.9334, -47.8105]])
    
    twicetheta_data, chi_data = expdata.T
    
    AngRes = Angular_residues_np(test_Matrix, twicetheta_data, chi_data,
                    ang_tol=ang_tol,
                    key_material='CdTe',
                    emin=5, emax=40,
                    ResolutionAngstrom=False,
                    detectorparameters=detectorparameters)
    
    if AngRes is not None:
        allresidues, res, nb_in_res, nb_of_theospots, meanres, maxi = AngRes
        
        matchringrate = 1.*nb_in_res / nb_of_theospots * 100.
        print('nb of theo spots', nb_of_theospots)
        print("matching rate is %.2f" % matchringrate)

def test_old(database):
    """
    test refinement and multigrains indexing with class
    """
    readfile = 'test.cor'

    DictLT.dict_Materials['DIAs'] = ['DIAs', [3.16, 3.16, 3.16, 90, 90, 90], 'dia']
    firstmatchingtolerance = 1.

    if 1:

        key_material = 'Si'
        emin = 5
        emax = 25

        MatchingRate_Threshold = 30  # percent

        dictimm = {'sigmagaussian':(0.5, 0.5, 0.5),
                   'Hough_init_sigmas' : (1, .7),
                   'Hough_init_Threshold':1,
                   'rank_n':20,
                   'useintensities':0}

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'testSirandom'
        nbgrains = 3
        file_to_index = createFakeData(key_material, nbgrains, outputfilename=outputfilename)
    else:
        file_to_index = readfile


    # read spots data and init dictionary of indexed spots
    DataSet = spotsset()
    DataSet.importdatafromfile(file_to_index)

    TwiceTheta_Chi_Int = DataSet.getSpotsExpData()
    totalnbspots = len(TwiceTheta_Chi_Int[0])
    # Find the best orientations and gives a table of results

    DataSet.key_material = key_material
    DataSet.emin = emin
    DataSet.emax = emax


    # potential orientation solutions from image matching
    bestEULER = IMM.bestorient_from_2thetachi(TwiceTheta_Chi_Int,
                                                                 database,
                                                               dictparameters=dictimm)

    bestEULER, bestmatchingrates = filterEulersList(bestEULER, TwiceTheta_Chi_Int, key_material, emax,
                                                    rawAngularmatchingtolerance=firstmatchingtolerance,
                                                    MatchingRate_Threshold=MatchingRate_Threshold,
                                                    verbose=1)


    MatrixPile = bestEULER.tolist()



    matrix_to_test = True
    grain_index = 0  # current grain to be indexed

    indexedgrains = []

    selectedspots_index = None

    AngleTol_0 = 1.
    while matrix_to_test:

        if len(MatrixPile) != 0:
            eulerangles = MatrixPile.pop(0)
            trialmatrix = GT.fromEULERangles_toMatrix(eulerangles)
        else:
            print("no more orientation to test (euler angles)")
            print("need to use other indexing techniques, Angles LUT, cliques help...")
            return

#        #add a matrix in dictionary
#        DataSet.dict_grain_matrix[grain_index] = trialmatrix

        print("dict_grain_matrix before initial indexing with a raw orientation")
        print(DataSet.dict_grain_matrix)

        # indexation of spots with raw (un refined) matrices & #update dictionaries
        print("\n\n---------------indexing grain  #%d-----------------------" % grain_index)
        print("eulerangles", eulerangles)
#        print "selectedspots_index", selectedspots_index
#        print "making raw links for grain #%d" % grain_index
        #TODO update
        DataSet.AssignHKL(eulerangles, grain_index, AngleTol_0,
                              use_spots_in_currentselection=selectedspots_index)


#        print "dict_grain_matrix after raw matching", DataSet.dict_grain_matrix
#        print "dict_matching_rate", DataSet.dict_grain_matching_rate

        # plot
        if DataSet.dict_grain_matrix is not None:
            DataSet.plotgrains(exp_data=TwiceTheta_Chi_Int,
                               titlefig='%d Exp. spots data (/%d) after raw matching grain #%d' % \
                               (len(TwiceTheta_Chi_Int[0]), totalnbspots, grain_index))
        else:
            print("plot the best last orientation matrix candidate")
            IMM.Plot_compare_2thetachi(eulerangles, TwiceTheta_Chi_Int[0], TwiceTheta_Chi_Int[1],
                               verbose=1, key_material=DataSet.key_material, emax=DataSet.emax,
                               EULER=1)

            IMM.Plot_compare_gnomondata(eulerangles, TwiceTheta_Chi_Int[0], TwiceTheta_Chi_Int[1],
                               verbose=1, key_material=DataSet.key_material, emax=DataSet.emax,
                               EULER=1)


        AngleTol_List = [0.5, 0.2, 0.1]

        # TODO: data flow and branching will be improved later
        for k, AngleTol in enumerate(AngleTol_List):

            print("\n\n refining grain #%d step -----%d\n" % (grain_index, k))
        #    print "initial matrix", dict_grain_matrix[grain_index]


            refinedMatrix = DataSet.refineUBSpotsFamily(grain_index,
                                DataSet.dict_grain_matrix[grain_index],
                                use_weights=1)

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
            indexation_res, nbtheospots, missingRefs = DataSet.getSpotsLinks(refinedMatrix,
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
            nb_updates = DataSet.updateIndexationDict(indexation_res,
                                                      grain_index,
                                                      overwrite=1)


            print("\nnb of indexed spots for this refined matrix: %d / %d" % (nb_updates, nbtheospots))
            print("with tolerance angle : %.2f deg" % AngleTol)

            Matching_rate = 100.*nb_updates / nbtheospots

            if Matching_rate < 50.:
                # matching rate too low
                # then remove previous indexed data and launch again imagematching
                print("matching rate too low")
                # there are at least one indexed grain
                # so remove the corresponding data and restart the imagematching
                if len(indexedgrains) > 0:

                    print("\n\n---------------------------------------------")
                    print("Use again imagematching on purged data from previously indexed spots")
                    print("---------------------------------------------\n\n")

#                    print "indexedgrains", indexedgrains

                    # extract data not yet indexed or temporarly indexed
                    toindexdata = DataSet.getUnIndexedSpotsallData(exceptgrains=indexedgrains)
                    absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
                    intensity_data = toindexdata[:, 5]

                    TwiceTheta_Chi_Int = np.array([twicetheta_data, chi_data, intensity_data])



                    # potential orientation solutions from image matching
                    bestEULER2 = IMM.bestorient_from_2thetachi(TwiceTheta_Chi_Int, database,
                                                                               dictparameters=dictimm)

                    bestEULER2, bestmatchingrates2 = filterEulersList(bestEULER, TwiceTheta_Chi_Int, key_material, emax,
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


                DataSet.plotgrains(exp_data=DataSet.getSpotsExpData(),
                                   titlefig='%d Exp. spots data (/%d), after refinement of grain #%d at step #%d' % \
                                   (totalnbspots, totalnbspots, grain_index, k))

                # if this is the last tolerance step
                if k == len(AngleTol_List) - 1:

                    print("\n---------------------------------------------")
                    print("indexing completed for grain #%d with matching rate %.2f " % (grain_index,
                                                                                       Matching_rate))
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
    readfile = 'test11.cor'
#    readfile = None

    DictLT.dict_Materials['DIAs'] = ['DIAs', [3.16, 3.16, 3.16, 90, 90, 90], 'dia']

    key_material = 'Si'
    emin = 5
    emax = 25

    MatchingRate_Threshold = 30  # percent


    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    dictimm = {'sigmagaussian':(0.5, 0.5, 0.5),
               'Hough_init_sigmas' : (1, .7),
               'Hough_init_Threshold':1,
               'rank_n':20,
               'useintensities':0}

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'testSirandom'
        nbgrains = 5
        removespots = [9, 8, 15, 6, 2]
        addspots = 6
#        removespots = None
#        addspots = None
        file_to_index = createFakeData(key_material, nbgrains,
                                       outputfilename=outputfilename,
                                       removespots=removespots,
                                       addspots=addspots)
    else:
        file_to_index = readfile

    # read spots data and init dictionary of indexed spots
    DataSet = spotsset()
    DataSet.importdatafromfile(file_to_index)
    totalnbspots = DataSet.nbspots

    DataSet.setMaterial(key_material)
    DataSet.setEnergyBand(emin, emax)

    DataSet.setImageMatchingParameters(dictimm, database)

    grain_index = 0  # current grain to be indexed
    AngleTol_0 = 1.
    AngleTol_List = [0.5, 0.2, 0.1]
    PLOTRESULTS = 0
    VERBOSE = 0

    provideNewMatrices = True
    indexgraincompleted = False

    while 1:

        nb_remaining_spots = len(DataSet.getUnIndexedSpots())

        print("\n nb of spots to index  : %d\n" % nb_remaining_spots)
        if nb_remaining_spots < 2:
            print("%d spots have been indexed over %d" % (totalnbspots - nb_remaining_spots, totalnbspots))
            print("indexing rate is --- : %.1f percents" % (100.*(totalnbspots - nb_remaining_spots) / totalnbspots))
            print("indexation of %s is completed" % DataSet.filename)
#            self.dict_grain_matching_rate[grain_index] = [0, 0]
            break

        if 1:
            print("start to index grain #%d" % grain_index)

        if provideNewMatrices == True:
            # potential orientation solutions from image matching
            print("providing new set of matrices")

            (bestUB,
             bestmatchingrates,
             nbspotsIMM) = DataSet.getOrients_ImageMatching(MatchingRate_Threshold=MatchingRate_Threshold,
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
                DataSet.plotgrains(exp_data=DataSet.TwiceTheta_Chi_Int,
                                   titlefig='%d Exp. spots data (/%d) after raw matching grain #%d' % \
                                   (nbspotsIMM, totalnbspots, grain_index))

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

                UBrefined = OrientMatrix(matrix=refinedMatrix)
    #            print "UBrefined", UBrefined.matrix
                DataSet.dict_grain_matrix[grain_index] = UBrefined.matrix
                # select data, link spots, update spot dictionary, update matrix dictionary
                Matching_rate, nb_updates, missingRefs = DataSet.AssignHKL(UBrefined,
                                                                                grain_index,
                                                                                AngleTol,
                                                                                verbose=VERBOSE)

            if Matching_rate < 50. or refinedMatrix is None:
                # matching rate too low
                # then remove previous indexed data and launch again imagematching
                if 1:
                    print("matching rate too low")

                DataSet.resetSpotsFamily(grain_index)

                # there are at least one indexed grain and this dataset has not been already tested with the matrix stack
                # so remove the corresponding data and restart the imagematching
                if len(DataSet.indexedgrains) > 0 and indexgraincompleted == True:
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
                    DataSet.plotgrains(exp_data=DataSet.getSpotsExpData(),
                                   titlefig='%d Exp. spots data (/%d), after refinement of grain #%d at step #%d' % \
                                   (totalnbspots, totalnbspots, grain_index, k))

                # if this is the last tolerance step
                if k == len(AngleTol_List) - 1:

                    if 1:
                        print("\n---------------------------------------------")
                        print("indexing completed for grain #%d with matching rate %.2f " % (grain_index,
                                                                                           Matching_rate))
                        print("---------------------------------------------\n")

                    DataSet.indexedgrains.append(grain_index)
                    provideNewMatrices = False
                    indexgraincompleted = True
                    grain_index += 1

    return DataSet


def test_index(database):
    """
    test refinement and multigrains indexing with class
    """
#    readfile = 'test6.cor'
#    readfile = 'test7.cor'
#    readfile = 'test10.cor'
    readfile = None

    key_material = 'Si'
    emin = 5
    emax = 25

    MatchingRate_Threshold = 30  # percent

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    dictimm = {'sigmagaussian':(0.5, 0.5, 0.5),
               'Hough_init_sigmas' : (1, .7),
               'Hough_init_Threshold':1,
               'rank_n':20,
               'useintensities':0}

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'testSirandom'
        nbgrains = 5
        removespots = [9, 8, 15, 6, 2]
        addspots = 6
#        removespots = None
#        addspots = None
        file_to_index = createFakeData(key_material, nbgrains,
                                       outputfilename=outputfilename,
                                       removespots=removespots,
                                       addspots=addspots)
    else:
        file_to_index = readfile


    DataSet = spotsset()

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
    import time, os
    readfile = os.path.join('./Examples/strain_calib_test2', 'dat_Wmap_WB_14sep_d0_500MPa_0045_LT_0.cor')

    key_material = 'W'
    emin = 5
    emax = 22

    nbGrainstoFind = 'max'  # '1 #'max'

    # read database
    if database is None:
        # ------------------------------------------------------------------------
        # LOAD High resolutio diamond (1 deg step)
        database = IMM.DataBaseImageMatchingSi()
        # ------------------------------------------------------------------------

    MatchingRate_Threshold_Si = 30  # percent for simulated Si
    dictimm_Si = {'sigmagaussian':(0.5, 0.5, 0.5),
               'Hough_init_sigmas' : (1, .7),
               'Hough_init_Threshold':1,
               'rank_n':20,
               'useintensities':0}

    MatchingRate_Threshold_W = 15  # percent W
    dictimm_W = {'sigmagaussian':(0.5, 0.5, 0.5),
               'Hough_init_sigmas' : (1, .7),
               'Hough_init_Threshold':1,
               'rank_n':40,
               'useintensities':0}


    MatchingRate_Threshold = MatchingRate_Threshold_W
    dictimm = dictimm_W

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'testSirandom'
#        nbgrains = 2
        removespots = np.zeros(nbgrains)
        addspots = 0
#        removespots = None
#        addspots = None
        file_to_index = createFakeData(key_material, nbgrains,
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
    DataSet_2 = spotsset()

    DataSet_2.IndexSpotsSet(file_to_index, key_material, emin, emax, dictimm, database, IMM=False,
                            MatchingRate_List=[40, 50, 60],
                            nbGrainstoFind=nbGrainstoFind)

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_2.plotallgrains()
    return DataSet_2  # , DataSet_2


def test_IAM_real():
    """
    test refinement and multigrains indexing with class wit angles LUT indexing technique
    """
#    readfile = 'test6.cor'
#    readfile = 'test7.cor'
#    readfile = 'test10.cor'
    import time, os
    readfile = os.path.join('./Examples/UO2', 'dat_UO2_A163_2_0028_LT_0.cor')

    key_material = 'UO2'
    emin = 5
    emax = 22

    nbGrainstoFind = 3  # 'max'#'1 #'max'

    MatchingRate_Threshold_Si = 30  # percent for simulated Si
    dictimm_Si = {'sigmagaussian':(0.5, 0.5, 0.5),
               'Hough_init_sigmas' : (1, .7),
               'Hough_init_Threshold':1,
               'rank_n':20,
               'useintensities':0}

    MatchingRate_Threshold_W = 15  # percent W
    dictimm_W = {'sigmagaussian':(0.5, 0.5, 0.5),
               'Hough_init_sigmas' : (1, .7),
               'Hough_init_Threshold':1,
               'rank_n':40,
               'useintensities':0}

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

    dict_params = {'MATCHINGRATE_THRESHOLD_IAL':60,
                   'MATCHINGRATE_ANGLE_TOL' : 0.2,
                   'NBMAXPROBED' : 10,
                   'central spots indices' : 10}

    t0_2 = time.time()
    DataSet_2 = spotsset()

    DataSet_2.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database, IMM=False,
                            MatchingRate_List=[20, 20, 20],
                            nbGrainstoFind=nbGrainstoFind)

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
    key_material = 'Cu'
    emin = 5
    emax = 22

    nbmax_probed = 8000000
    nbGrainstoFind = 10

    angleTolerance_LUT = .1
    MatchingRate_Angle_Tol = 0.001
    Matching_Threshold_Stop = 50.  # minimum to stop the loop for searching potential solution

    # create a fake data file of randomly oriented crystal
    if readfile is None:
        outputfilename = 'toto'
        nbgrains = nbGrainstoFind
        removespots = [0] * nbgrains
        addspots = 3
#        removespots = None
#        addspots = None
        file_to_index = createFakeData(key_material, nbgrains,
                                       emin=emin, emax=emax,
                                       outputfilename=outputfilename,
                                       removespots=removespots,
                                       addspots=addspots)
    else:
        file_to_index = readfile

    # read data
    data_theta, Chi, posx, posy, dataintensity, detectorparameters = IOLT.readfile_cor(file_to_index)[1:]
    tth = 2 * data_theta

    exp_data = np.array([tth, Chi])

    thechi_exp = np.array([data_theta, Chi]).T

    from .indexingAnglesLUT import calculdist_from_thetachi
    table_angdist = calculdist_from_thetachi(thechi_exp, thechi_exp)

    latticeparams = DictLT.dict_Materials[key_material][1]
    B = CP.calc_B_RR(latticeparams)

    nbmax_probed = min(nbmax_probed, len(data_theta))
    #---------------------------------------------------------------
    import time
    t0_2 = time.time()
    DataSet_2 = spotsset()

    database = None
    dictimm = None
    DataSet_2.IndexSpotsSet(file_to_index, key_material, emin, emax, dictimm, database, IMM=False,
                            MatchingRate_List=[20, 20, 20],
                            nbGrainstoFind=nbGrainstoFind,
                            verbose=0)

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_2.plotallgrains()


#--- CuVia analysis
def testtwin(fileindex=2301, nbGrainstoFind=3,
             dirname='/home/micha/LaueProjects/CuVia/Carto', dirtowrite=None):
    """
    test twin
    """
    if dirtowrite is None:
        dirtowrite = dirname

#     spot_index_central = [0, 1, 2]
    key_material = 'Si'
    emin = 5
    emax = 25

    file_to_index = os.path.join(dirname, 'Si_TSV.cor')

    t0_2 = time.time()
    DataSet_Si = spotsset()

    database = None

    dict_params = {'MATCHINGRATE_THRESHOLD_IAL': 60,
                   'MATCHINGRATE_ANGLE_TOL': 0.2,
                   'NBMAXPROBED': 10,
                   'central spots indices': 10,
                   'AngleTolLUT': .5,
                   'UseIntensityWeights': True,
                   'MinimumNumberMatches': 10}

    DataSet_Si.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database,
                             IMM=False,
                            MatchingRate_List=[20, 20, 20], angletol_list=[0.5, 0.2],
                            nbGrainstoFind=1,
                            verbose=0)

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_Si.plotallgrains()

    # ind, 2theta, chi, posx, posy, int
    dataSubstrate = DataSet_Si.getSpotsFamilyExpData(0).T

    DataSet_Cu = spotsset()
    DataSet_Cu.importdatafromfile(file_to_index)

    DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

    key_material = 'Cu'
    dict_params = {'MATCHINGRATE_THRESHOLD_IAL':40,
                   'MATCHINGRATE_ANGLE_TOL': 0.2,
                   'NBMAXPROBED': 20,
                   'central spots indices': 3,
                   'AngleTolLUT': .5,
                   'UseIntensityWeights':True,
                   'MinimumNumberMatches': 10}

    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}
    dictRes = (dictMat, dictMR, dictNB, dictstrain)

    nstart = fileindex
    nend = fileindex

    for k in range(nstart, nend + 1):  # 92 - 1708
        file_to_index = os.path.join(dirname, 'TSVCU_%04d.cor' % k)

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

        DataSet_Cu.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database,
                                 checkSigma3=True,
                             use_file=0, IMM=False, angletol_list=[0.5, 0.5],
                            MatchingRate_List=[10, 10, 10],
                            nbGrainstoFind=nbGrainstoFind,
                            verbose=0, previousResults=previousResults)

        for nbgrain in range(nbGrainstoFind):
            dictMat[k][nbgrain] = DataSet_Cu.dict_grain_matrix[nbgrain]
            dictMR[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][1]
            dictNB[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][0]
            dictstrain[k][nbgrain] = DataSet_Cu.dict_grain_devstrain[nbgrain]

        # intermediate saving
        if (k % 100) == 0:
#            dictRes = dictMat, dictMR, dictNB
            filepickle = open('dictCuViaaddMatrix%04d_%04d' % (nstart, nend), 'w')
            pickle.dump(dictRes, filepickle)
            filepickle.close()

    filepickle = open('dictCuViaaddMatrix%04d_%04d' % (nstart, nend), 'w')
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
    key_material = 'Si'
    emin = 5
    emax = 25
    nbGrainstoFind = 1

    import time, os
    file_to_index = os.path.join('/home/micha/LaueProjects/CuVia/filecor', 'dat_TSVCU_0092.cor')

    t0_2 = time.time()
    DataSet_Si = spotsset()

    database = None

    dict_params = {'MATCHINGRATE_THRESHOLD_IAL':60,
                   'MATCHINGRATE_ANGLE_TOL' : 0.2,
                   'NBMAXPROBED' : 10,
                   'central spots indices' : 10}

    DataSet_Si.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database, IMM=False,
                            MatchingRate_List=[20, 20, 20], angletol_list=[0.5, 0.2],
                            nbGrainstoFind=nbGrainstoFind,
                            verbose=0)

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    DataSet_Si.plotallgrains()


    # ind, 2theta, chi, posx, posy, int
    dataSubstrate = DataSet_Si.getSpotsFamilyExpData(0).T

    DataSet_Cu = spotsset()
    DataSet_Cu.importdatafromfile(file_to_index)

    DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

    key_material = 'Cu'
    nbGrainstoFind = 3
    dict_params = {'MATCHINGRATE_THRESHOLD_IAL':40,
                   'MATCHINGRATE_ANGLE_TOL' : 0.2,
                   'NBMAXPROBED' : 20,
                   'central spots indices' : 3}

    dictMat = {}
    dictMR = {}
    dictNB = {}

    for k in range(nstart, nend + 1):  # 92 - 1708
        file_to_index = os.path.join('/home/micha/LaueProjects/CuVia/filecor', 'TSVCU_%04d.cor' % k)

        print("\n\nINDEXING    file : %s\n\n" % file_to_index)

        DataSet_Cu = spotsset()
        DataSet_Cu.importdatafromfile(file_to_index)

        DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

        # init res dict
        DataSet_Cu.dict_grain_matrix = [0, 0, 0]
        DataSet_Cu.dict_grain_matching_rate = [[-1, -1], [-1, -1], [-1, -1]]

        DataSet_Cu.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database,
                             use_file=0, IMM=False, angletol_list=[0.5, 0.5],
                            MatchingRate_List=[10, 10, 10],
                            nbGrainstoFind=nbGrainstoFind,
                            verbose=0)
        # filling results
        dictMat[k] = [0, 0, 0]
        dictMR[k] = [-1, -1, -1]
        dictNB[k] = [-1, -1, -1]
        for nbgrain in range(nbGrainstoFind):
            dictMat[k][nbgrain] = DataSet_Cu.dict_grain_matrix[nbgrain]
            dictMR[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][1]
            dictNB[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][0]

        # intermediate saving
        dictRes = dictMat, dictMR, dictNB
        if (k % 100) == 0:
            filepickle = open('dictCuVia%04d_%04d' % (nstart, nend), 'w')
            pickle.dump(dictRes, filepickle)
            filepickle.close()

    filepickle = open('dictCuVia', 'w')
    pickle.dump(dictRes, filepickle)
    filepickle.close()

#        DataSet_Cu.plotallgrains()
    return dictRes
#    return DataSet_Si, DataSet_Cu


#--- ---------------------- index file series
def index_fileseries(fileindexrange,
                     nbGrainstoFind=1,
          dirname='/home/micha/LaueProjects/CuVia/Carto',
          dirtowrite=None,
          saveObject=0):
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
    key_material = 'Si'
    emin = 5
    emax = 25
    nbGrainstoFind_Si = 1

    import time
    file_to_index = os.path.join(dirname, 'Si_TSV.cor')

    t0_2 = time.time()
    DataSet_Si = spotsset()

    database = None

    dict_params = {'MATCHINGRATE_THRESHOLD_IAL': 60,
                   'MATCHINGRATE_ANGLE_TOL': 0.2,
                   'NBMAXPROBED': 10,
                   'central spots indices': 10,
                   'AngleTolLUT': .5,
                   'UseIntensityWeights':True,
                   'MinimumNumberMatches':10}

    DataSet_Si.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database, IMM=False,
                            MatchingRate_List=[20, 20, 20], angletol_list=[0.5, 0.2],
                            nbGrainstoFind=nbGrainstoFind_Si,
                            verbose=0)

    tf2 = time.time()
    print("Angles LUT execution time %.3f sec." % (tf2 - t0_2))

    # DataSet_Si.plotallgrains() # to comment when mutliprocessing

    # ind, 2theta, chi, posx, posy, int
    dataSubstrate = DataSet_Si.getSpotsFamilyExpData(0).T

    key_material = 'Cu'
    nbGrainstoFind_Cu = nbGrainstoFind
    dict_params = {'MATCHINGRATE_THRESHOLD_IAL':40,
                   'MATCHINGRATE_ANGLE_TOL': 0.2,
                   'NBMAXPROBED': 20,
                   'central spots indices': 3,
                   'AngleTolLUT': .5,
                   'UseIntensityWeights':True,
                   'MinimumNumberMatches':10}

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

    for k in range(nstart, nend + 1):  # 1708-3323
        file_to_index = os.path.join(dirname, 'TSVCU_%04d.cor' % k)

        print("\n\nINDEXING    file : %s\n\n" % file_to_index)

        DataSet_Cu = spotsset()
        DataSet_Cu.importdatafromfile(file_to_index)

        DataSet_Cu.purgedata(dataSubstrate[1:3], dist_tolerance=0.5)

        # TODO usefull ?
        # init res dict
        DataSet_Cu.dict_grain_matrix = [0 for kk in range(nbGrainstoFind_Cu)]
        DataSet_Cu.dict_grain_matching_rate = [[-1, -1] for kk in range(nbGrainstoFind_Cu)]
        DataSet_Cu.dict_grain_devstrain = [0 for kk in range(nbGrainstoFind_Cu)]

        # preparing dicts of results
        dictMat[k] = [0 for kk in range(nbGrainstoFind_Cu)]
        dictMR[k] = [-1 for kk in range(nbGrainstoFind_Cu)]
        dictNB[k] = [-1 for kk in range(nbGrainstoFind_Cu)]
        dictstrain[k] = [0 for kk in range(nbGrainstoFind_Cu)]
        previousResults = None

        if k > nstart and dictMat[k - 1][0] is not 0:
            addMatrix = dictMat[k - 1][0]
            previousResults = addMatrix, dictMR[k - 1][0], dictNB[k - 1][0]

        DataSet_Cu.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_params, database,
                             use_file=0, IMM=False, angletol_list=[0.5, 0.5],
                            MatchingRate_List=[10, 10, 10],
                            nbGrainstoFind=nbGrainstoFind_Cu,
                            verbose=0, previousResults=previousResults)

        dictspots[k] = DataSet_Cu.getSummaryallData()

        if saveObject:
            dict_spotssetObj[k] = DataSet_Cu

        # filling dicts with indexation results
        for nbgrain in range(nbGrainstoFind_Cu):
            dictMat[k][nbgrain] = DataSet_Cu.dict_grain_matrix[nbgrain]
            dictMR[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][1]
            dictNB[k][nbgrain] = DataSet_Cu.dict_grain_matching_rate[nbgrain][0]
            dictstrain[k][nbgrain] = DataSet_Cu.dict_grain_devstrain[nbgrain]

        # intermediate saving
        if (k % 10) == 0:
#            dictRes = dictMat, dictMR, dictNB
            filepickle = open(os.path.join(dirtowrite, 'dictCu_3g_%04d_%04d' % (nstart, nend)), 'w')
            pickle.dump(todump, filepickle)
            filepickle.close()

    filepickle = open(os.path.join(dirtowrite, 'dictCu_3g_%04d_%04d' % (nstart, nend)), 'w')
    pickle.dump(todump, filepickle)
    filepickle.close()

#        DataSet_Cu.plotallgrains()

    return todump
#    return DataSet_Si, DataSet_Cu


def mergeDictRes(list_of_dictfiles, outputfilename='MergedRes', dirname=None):
    """
    merge dictionnaries from indexed file series

    dictRes = dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictspots
    """
    dictMaterial = {}
    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}
    dictspots = {}

    print("list_of_dictfiles", list_of_dictfiles)

    for _file in list_of_dictfiles:

        filepickle = open(os.path.join(dirname, _file), 'r')
        Res = pickle.load(filepickle)

        dMater, dMat, dMR, dNB, dstrain, dspots = Res

        dictMaterial = dict(list(dMater.items()) + list(dictMaterial.items()))
        dictMat = dict(list(dMat.items()) + list(dictMat.items()))
        dictMR = dict(list(dMR.items()) + list(dictMR.items()))
        dictNB = dict(list(dNB.items()) + list(dictNB.items()))
        dictstrain = dict(list(dstrain.items()) + list(dictstrain.items()))
        dictspots = dict(list(dspots.items()) + list(dictspots.items()))
        filepickle.close()

    tuple_dicts = dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictspots

    output_file = open(os.path.join(dirname, outputfilename), 'w')
    pickle.dump(tuple_dicts, output_file)
    output_file.close()

    return dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictspots


list_produced_files = []

def log_result(result):

    if len(result) == 2:
        print("********************\n\n\n\n %s \n\n\n\n\n******************" % result[1])
        list_produced_files.append(str(result[1]))

    print('mylog print')


def indexing_multiprocessing(fileindexrange,
                             dirname_dictRes=None,
                             Index_Refine_Parameters_dict=None,
                                verbose=0,
                                nb_materials=None,
                                saveObject=0,
                                nb_of_cpu=2,
                                build_hdf5=True,
                                prefixfortitle='',
                                reanalyse=True,
                                use_previous_results=True,
                                updatefitfiles=False,
                                CCDCalibdict=None):
    """
    launch several indexation and unit cell refinement processes in parallel
    """
    try:
        if len(fileindexrange) > 2:
            print('\n\n ---- Warning! file STEP INDEX is SET to 1 !\n\n')
        index_start, index_final = fileindexrange[:2]
    except:
        raise ValueError("Need 2 file indices (integers) in fileindexrange=(indexstart, indexfinal)")
        return

    fileindexdivision = GT.getlist_fileindexrange_multiprocessing(index_start,
                                                                  index_final,
                                                                  nb_of_cpu)

    saveObject = 0

    t00 = time.time()
#     jobs = []
#     for ii in range(nb_of_cpu):
#         proc = multiprocessing.Process(target=index_fileseries_3,
#                                     args=(fileindexdivision[ii],
#                                     Index_Refine_Parameters_dict,
#                                         saveObject,
#                                         verbose,
#                                         nb_materials,
#                                         False,
#                                         prefixfortitle))
#
# #         proc.daemon = False
#         jobs.append(proc)
#         proc.start()
#         proc.join()
#
# #     jobs[-1].join()

    index_fileseries_3.__defaults__ = (Index_Refine_Parameters_dict,
                                        saveObject,
                                        verbose,
                                        nb_materials,
                                        False,
                                        prefixfortitle,
                                        reanalyse,
                                        use_previous_results,
                                        updatefitfiles,
                                        CCDCalibdict)

    print("fileindexdivision", fileindexdivision)

    pool = multiprocessing.Pool()
    for ii in range(len(fileindexdivision)):  # range(nb_of_cpu):
        pool.apply_async(index_fileseries_3, args=(fileindexdivision[ii],), callback=log_result)  # make our results with a map call

    pool.close()
    pool.join()

    t_mp = time.time() - t00
    print("Execution time : %.2f" % t_mp)

    print("HOURRA it's FINISHED")

#     f = open(os.path.join(dirname_dictRes, "indexrefine.log"), 'r')
#     list_files = f.readlines()
#     f.close()

    output_mergeddicts_filename = '%s_dict_%04d_%04d' % \
                            (prefixfortitle, index_start, index_final)


    list_produced_files = []
    for elem in fileindexdivision:
        filen = '%s_dict_%04d_%04d' % (prefixfortitle, elem[0], elem[1])
        list_produced_files.append(str(filen))

    flag_completed = False
    flag_completed_HDF5 = False
    try:
        print("intermediate dict_files", list_produced_files)
        mergeDictRes(list_produced_files, outputfilename=output_mergeddicts_filename,
                     dirname=dirname_dictRes)

        flag_completed = True
    except IOError:
        print("\n******************\n")
        print("An error should have occured during at least one thread.\nCheck the error by using only one CPU!")
        print("\n******************\n")

        return flag_completed, flag_completed_HDF5

    if build_hdf5:
        try:
            from . import Lauehdf5 as LaueHDF5
            print("Building hdf5 file")
        except ImportError:
            print("module Lauehdf5 is not installed!")
            print("summaryfile.H5 file won't be created.")

        if dirname_dictRes is None:
            print('folder where hdf5 will be written is None!')
            return flag_completed, flag_completed_HDF5

        flag_completed_HDF5 = LaueHDF5.build_hdf5(output_mergeddicts_filename,
                            dirname_dictRes=dirname_dictRes,
                            output_hdf5_filename="dictSUMMARY_%s%04d_%04d.h5" % (prefixfortitle, index_start, index_final),
                            output_dirname=dirname_dictRes)

        return flag_completed, flag_completed_HDF5


def ReadSummaryFile(filename, dirname=None):
    """
    read summary .dat file generated by multigrain

    one line per grain and per image
    """
    fullpath = filename
    if dirname is not None:
        fullpath = os.path.join(dirname, filename)

    f = open(fullpath, 'r')
    # skip first line
    f.readline()

    # read columns name
    columns = f.readline()
    list_cols = columns.split(' ')
    list_column_names = []
    dict_column_names = {}
    for k, elem in enumerate(list_cols):
        list_column_names.append(elem)
        dict_column_names[elem] = k

    data = np.loadtxt(f, dtype=np.float)  # , comments, delimiter, converters, skiprows, usecols, unpack, ndmin)

    f.close()

    # remove last elem = '\n'
    del dict_column_names['\n']
    return data, list_column_names[:-1], dict_column_names


def index_fileseries_3(fileindexrange,
                       Index_Refine_Parameters_dict=None,
                       saveObject=0,
                       verbose=0,
                       nb_materials=None,
                       build_hdf5=False,
                       prefixfortitle='',
                       reanalyse=True,
                       use_previous_results=True,
                       updatefitfiles=False,
                       CCDCalibdict=None):
    """
    Core procedure to index and refine a serie of peaks list
    
    :param fileindexrange: list of starting, final and step image index
    :type fileindexrange: list of 3 integers
    :param Index_Refine_Parameters_dict:
    :type Index_Refine_Parameters_dict:
    :param saveObject: save spots dataset object
    :type saveObject: flag
    :param nb_materials: number of materials used in predefined list Index_Refine_Parameters_dict
    :type nb_materials: int
    """

    p = multiprocessing.current_process()
    print('Starting:', p.name, p.pid)

#     ANGLE_TOL_REMOVE_PEAKS = 0.5

    if CCDCalibdict is None:
        CCDCalibdict = {}

    if 'ResolutionAngstrom' in Index_Refine_Parameters_dict:
        ResolutionAngstrom = Index_Refine_Parameters_dict['ResolutionAngstrom']
    else:
        ResolutionAngstrom = False
        printcyan('default value for ResolutionAngstrom: %s' % str(ResolutionAngstrom))

    if 'nLUTmax' in Index_Refine_Parameters_dict:
        nLUTmax = Index_Refine_Parameters_dict['nLUTmax']
    else:
        nLUTmax = 3
        printcyan('default value for nLUTmax: %d' % nLUTmax)

    calibparam = CCDCalibdict['CCDCalibParameters']

    dict_params_list = Index_Refine_Parameters_dict['dict params list']

    if nb_materials is None:
        nb_materials = len(dict_params_list)
    else:
        if nb_materials > len(dict_params_list):
            printred("input nb_of_materials is too high in irp file")

    nbdigits = Index_Refine_Parameters_dict['nbdigits']
    prefixfilename = Index_Refine_Parameters_dict['prefixfilename']
    suffixfilename = Index_Refine_Parameters_dict['suffixfilename']
    prefixdictResname = Index_Refine_Parameters_dict['prefixdictResname']
    ResultsFolder = Index_Refine_Parameters_dict['Results Folder']
    fitfile_folder = Index_Refine_Parameters_dict['PeakListFit Folder']
    
    GuessedUBMatrices = 0
    if 'GuessedUBMatrix' in Index_Refine_Parameters_dict:
        GuessedUBMatrices = Index_Refine_Parameters_dict['GuessedUBMatrix']
    MinimumMatchingRate = Index_Refine_Parameters_dict['MinimumMatchingRate']
    
    CheckOrientations = None
    if 'CheckOrientation' in Index_Refine_Parameters_dict:
        CheckOrientationsFile = Index_Refine_Parameters_dict['CheckOrientation']
        CheckOrientations = IOLT.readCheckOrientationsFile(CheckOrientationsFile)
        # let s start with simple case of two signals
        
        
        

    # preparing dicts of results for all peaklists (key= imageindex)
    # values = several dicts (key = grainindex)
    dictMaterial = {}
    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}
    dictspots = {}

    if saveObject:
        dict_spotssetObj = {}

    dictRes = dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictspots

    if saveObject:
        todump = dictRes, dict_spotssetObj
    else:
        todump = dictRes

    encodingdigits = '%%0%dd' % nbdigits

    print("fileindexrange", fileindexrange)
    nstart = fileindexrange[0]
    nend = fileindexrange[1]

    outputdict_filename = prefixdictResname + '%04d_%04d' % (nstart, nend)

    totalnb_grains = 0
    for material_index in range(nb_materials):
        # read indexation parameters for the current material
        # TODO: add a function to set to default value if key is missing
        dict_param_SingleGrain = dict_params_list[material_index]
        nbGrainstoFind_mat = dict_param_SingleGrain['nbGrainstoFind']
        print("nbGrainstoFind_mat", nbGrainstoFind_mat)
        totalnb_grains += nbGrainstoFind_mat

    if len(fileindexrange) == 2:
        fileindexrange = fileindexrange[0], fileindexrange[1], 1

    if not reanalyse:
        list_files_in_folder = os.listdir(fitfile_folder)
        import re
        test = re.compile("\.res$", re.IGNORECASE)
        list_resfiles_in_folder = list(filter(test.search, list_files_in_folder))

    dict_LUT_material = {}

    #--- Loop over images ----------------------
    lastindex = nstart
    for imageindex in range(fileindexrange[0], fileindexrange[1] + 1, fileindexrange[2]):

        if not reanalyse:
            resfilename = prefixfilename + encodingdigits % imageindex + '.res'
            if resfilename in list_resfiles_in_folder:
                printcyan("file %s exists, corresponding .dat or .cor file is already indexed !!" \
                            % resfilename)
                continue

        if suffixfilename.endswith('.dat'):
            datfilename = prefixfilename + encodingdigits % imageindex + suffixfilename

            dirname_in = Index_Refine_Parameters_dict['PeakList Folder']

            if imageindex == fileindexrange[0]:
                fullpath = os.path.join(dirname_in, datfilename)
                if not os.path.exists(fullpath):
                    printred("\n\n*******\nSomething wrong with the filename: %s. Please check carefully the filename!" % \
                                                        fullpath)
                    return

            if not os.path.exists(os.path.join(dirname_in, datfilename)):
                printcyan('Missing file : %s\n Keep on scanning files\n' % datfilename)
                continue

            # batch to convert from .dat (peak list of X,Y,I) to .cor (2theta,Chi,X,Y,I)
            print('build .cor file')
            print("in %s" % Index_Refine_Parameters_dict['PeakListCor Folder'])
            F2TC.convert2corfile(datfilename,
                         calibparam,
                         dirname_in=dirname_in,
                        dirname_out=Index_Refine_Parameters_dict['PeakListCor Folder'],
                        CCDCalibdict=CCDCalibdict)

            corfilename = datfilename.split('.')[0] + '.cor'

        elif suffixfilename == '.cor':
            corfilename = prefixfilename + encodingdigits % imageindex + suffixfilename
            dirname_in = Index_Refine_Parameters_dict['PeakListCor Folder']

            if not os.path.exists(os.path.join(dirname_in, corfilename)):
                printcyan('Missing file : %s\n Keep on scanning files\n' % corfilename)
                continue

        file_to_index = os.path.join(Index_Refine_Parameters_dict['PeakListCor Folder'],
                                         corfilename)

        print("\n\nINDEXING    file : %s\n\n" % file_to_index)

        DataSet = spotsset()

        DataSet.pixelsize = CCDCalibdict['xpixelsize']
        DataSet.dim = CCDCalibdict['framedim']
        DataSet.detectordiameter = CCDCalibdict['detectordiameter']
        if 'kf_direction' in CCDCalibdict:
            DataSet.kf_direction = CCDCalibdict['kf_direction']
        else:
            printcyan("\n\n*******\n warning: use default geometry: \n%s\n*****\n\n" % \
                                DEFAULT_KF_DIRECTION)
            DataSet.kf_direction = DEFAULT_KF_DIRECTION


#         DataSet.dict_indexedgrains_material = {}
#         DataSet.dict_grain_matrix = {}
#         DataSet.dict_grain_matching_rate = {}
#         DataSet.dict_grain_devstrain = {}

        # preparing dicts of results for indexation of peaks list corresponding to image with imageindex
        dictMaterial[imageindex] = [None for kk in range(totalnb_grains)]
        dictMat[imageindex] = [0 for kk in range(totalnb_grains)]
        dictMat[imageindex][0] = GuessedUBMatrices
        dictMR[imageindex] = [-1 for kk in range(totalnb_grains)]
        dictNB[imageindex] = [-1 for kk in range(totalnb_grains)]
        dictstrain[imageindex] = [0 for kk in range(totalnb_grains)]

        #--- Loop over materials ----------------------
        for material_index in range(nb_materials):

            dict_param_SingleGrain = dict_params_list[material_index]

            if len(DataSet.indexedgrains) > 0:
                starting_grainindex = DataSet.indexedgrains[-1] + 1
                
            # loading data of spots to be indexed for the first time
            else:
                print("starting_grainindex = 0")
                starting_grainindex = 0
                
                # sort optionally according to the order in a file
                # located in dict_param_SingleGrain['Spots Order Reference File']
                sortSpots_from_refenceList = None
                if 'Spots Order Reference File' in dict_param_SingleGrain:
                    sortSpots_from_refenceList = dict_param_SingleGrain['Spots Order Reference File']
                
                DataSet.importdatafromfile(file_to_index, sortSpots_from_refenceList=sortSpots_from_refenceList)

                if DataSet.nbspots < 3:
                    print("%d spot(s) are too few to be indexed" % DataSet.nbspots)
                    DataSet.LUT = None
                    continue

                if DataSet.detectorparameters is None:
                    printred("file %s does not contain the 5 detector parameters" % \
                                                        file_to_index)
                    return

#             print 'DataSet.detectorparameters', DataSet.detectorparameters

            print("\n ########### starting_grainindex %d ###########\n" % starting_grainindex)

            key_material = dict_param_SingleGrain['key material']
            emin = dict_param_SingleGrain['emin']
            emax = dict_param_SingleGrain['emax']
            nbGrainstoFind = dict_param_SingleGrain['nbGrainstoFind']

            try:
                set_central_spots_hkl = GT.fromstringtolist(dict_param_SingleGrain['setCentralSpotsHKL'])
            except ValueError:
                printred("Can't understand '%s' as a list like [h,k,l] for 'setCentralSpotsHKL' parameter in .irp file" % dict_param_SingleGrain['setCentralSpotsHKL'])
                return

            try:
                central_spots_list = GT.read_elems_from_string(dict_param_SingleGrain['central spots indices'])
            except ValueError:
                printred("Can't understand '%s' as a list of spot indices for 'central spots indices' parameter in .irp file" % dict_param_SingleGrain['central spots indices'])
                return

            try:
                List_Matching_Tol_Angles = GT.read_elems_from_string(dict_param_SingleGrain['List Matching Tol Angles'],
                                                                map_function=float)
            except ValueError:
                printred("Can't understand '%s' as a list of spot indices for 'List Matching Tol Angles' parameter in .irp file" % dict_param_SingleGrain['List Matching Tol Angles'])
                return

            print("with material: %s\n" % key_material)

            t0_2 = time.time()

            dict_loop = {'MATCHINGRATE_THRESHOLD_IAL': dict_param_SingleGrain['MATCHINGRATE THRESHOLD IAL'],
                   'MATCHINGRATE_ANGLE_TOL': dict_param_SingleGrain['MATCHINGRATE ANGLE TOL'],
                   'NBMAXPROBED': dict_param_SingleGrain['NBMAXPROBED'],
                   'central spots indices': central_spots_list,
                   'AngleTolLUT': dict_param_SingleGrain['AngleTolLUT'],
                   'UseIntensityWeights': dict_param_SingleGrain['UseIntensityWeights'],
                   'MinimumNumberMatches': dict_param_SingleGrain['MinimumNumberMatches'],
                   'MinimumMatchingRate':MinimumMatchingRate
                   }
            if 'nbSpotsToIndex' not in dict_param_SingleGrain:
                dict_param_SingleGrain['nbSpotsToIndex']=1000
            
            dict_loop['nbSpotsToIndex']=dict_param_SingleGrain['nbSpotsToIndex']
            
#             ResolutionAngstrom = False
            if 'ResolutionAngstrom' in dict_param_SingleGrain:
                ResolutionAngstrom = dict_param_SingleGrain['ResolutionAngstrom']
                print("ResolutionAngstrom", ResolutionAngstrom)
                if ResolutionAngstrom in ('False', None, 0, 0.0):
                    ResolutionAngstrom = False
                else:
                    try:
                        ResolutionAngstrom = float(ResolutionAngstrom)
                    except ValueError:
                        printred("ResolutionAngstrom can not be converted in float !!\nPlease check the irp file")
                        return

            ResolutionAngstromLUT = False
            if 'ResolutionAngstromLUT' in dict_param_SingleGrain:
                ResolutionAngstromLUT = dict_param_SingleGrain['ResolutionAngstromLUT']
                print("ResolutionAngstromLUT", ResolutionAngstromLUT)
                if ResolutionAngstromLUT in ('False', None, 0, 0.0):
                    ResolutionAngstromLUT = False
                else:
                    try:
                        ResolutionAngstromLUT = float(ResolutionAngstromLUT)
                    except ValueError:
                        printred("ResolutionAngstromLUT can not be converted in float !!\nPlease check the irp file")
                        return

            if 'nLUTmax' in dict_param_SingleGrain:
                nLUTmax = dict_param_SingleGrain['nLUTmax']
                print("nLUTmax", nLUTmax)
                if nLUTmax in ('False', None, 0, 0.0, 1, 2, 1.0, 2.0, 3.0):
                    nLUTmax = 3
                else:
                    try:
                        nLUTmax = int(nLUTmax)
                    except ValueError:
                        printred("nLUTmax can not be converted to integer !!\nPlease check the irp file")
                        return

#             if dataSubstrate is not None:
#                 DataSet.purgedata(dataSubstrate[1:3],
#                                   dist_tolerance=ANGLE_TOL_REMOVE_PEAKS)

            previousResults = None
            # read a guessed orientation matrix in dictMat
            if use_previous_results:
                print("current index", imageindex)
                print("lastindex", lastindex)
                #GuessedUBMatrices = dictMat[lastindex][0] 
                if imageindex >= nstart and GuessedUBMatrices is not 0:
                    # first matrix only
#                     addMatrix = dictMat[lastindex][0]

                    # list of matrices
                    nbAddMatrices=len(GuessedUBMatrices)
                    addMatrices = GuessedUBMatrices
                    if GuessedUBMatrices is 0: # TODO: to BE REMOVED 
                        previousResults = None
                    else:
                        previousResults = nbAddMatrices, addMatrices, dictMR[lastindex][0], dictNB[lastindex][0]

                    print("previousResults", previousResults)
            
            #----------------------------------------------------------
            # test if orientation material energy max can index
            #-----------------------------------------
            
            # from ubs file
            CheckOrientationParams = None        
            if CheckOrientations is not None:
                for UBsparams in CheckOrientations:
                    print('UBsparams',UBsparams)
                    print("key_material",key_material)
                    print("key_material",type(key_material))
                    if key_material in UBsparams:
                        print("yaouuh")
                        CheckOrientationParams=[UBsparams]
                        break
            # from previous results already written in corresponding fit file
            elif updatefitfiles:
                # read orientation matrix, material from _g#.fit file
                # warning: since the loop is made over material_index, to properly scan and update every g#.fit files
                # one needs to have an .irp file with a single nbgrainstofind=1 per material. 
                # Otherwise, if (nbofmaterials in irp) < (nb of _g#.fit files) only the first (nbofmaterials in irp) _g#.fit will be updated
                
                # TODO: ability to change emax
                EMAX = emax
                # set lowest threshold to accept indexation for next step of refinement
                # this threshold comes from the GUI index_Refine.py
                MINIMUM_MATCHING_RATE = MinimumMatchingRate
                
                
                fitfilename = prefixfilename + encodingdigits % imageindex + '_g%d.fit'%material_index 
                (list_indexedgrains_indices,
                list_nb_indexed_peaks, 
                pixdev,
                Material_list,
                all_UBmats_flat,
                CCDcalib) = IOLT.readfitfile_multigrains(os.path.join(fitfile_folder,fitfilename),
                                                return_toreindex =True)
                
                nbindexedgrains = len(list_indexedgrains_indices)
                
                CheckOrientationParams=[]
                for gindex in list_indexedgrains_indices:
                    # - File index (list or -2 for all images)
                    # - Grain index
                    # - Material
                    # - Energy Max
                    # - MatchingThreshold
                    # - Matrix(ces) 
                    CheckOrientationParams.append([imageindex,
                                                   gindex,
                                                   Material_list[gindex],
                                                   EMAX,MINIMUM_MATCHING_RATE,
                                                   all_UBmats_flat.reshape((nbindexedgrains,3,3))])
                    
                    print('CheckOrientationParams',CheckOrientationParams)
                

#             print "dict_LUT_material", dict_LUT_material
            if key_material in dict_LUT_material:
                LUT = dict_LUT_material[key_material]
            else:
                LUT = None
#             print "current unindexed spot absolute index", DataSet.getUnIndexedSpots()
            DataSet.IndexSpotsSet(file_to_index, key_material,
                   emin, emax, dict_loop, None,
                   starting_grainindex=starting_grainindex,
                 use_file=0, IMM=False,
                 n_LUT=nLUTmax, LUT=LUT,
                 set_central_spots_hkl=set_central_spots_hkl,
                 ResolutionAngstrom=ResolutionAngstrom,
                 angletol_list=List_Matching_Tol_Angles,
                MatchingRate_List=[1, 1, 1,1,1,1,1, 1,1,1], # 10 steps of refinements is highly reasonable
                nbGrainstoFind=nbGrainstoFind,
                verbose=0,
                corfilename=corfilename,
                dirnameout_fitfile=fitfile_folder,
                previousResults=previousResults,
                CheckOrientations = CheckOrientationParams)

            dict_LUT_material[key_material] = DataSet.LUT

            print('DataSet.indexedgrains', DataSet.indexedgrains)
            print('DataSet.indexedgrains_material', DataSet.dict_indexedgrains_material)
            nbgrains_indexed = len(DataSet.indexedgrains)

            DataSet.writecorFile_unindexedSpots(corfilename=corfilename,
                                                dirname=fitfile_folder,
                                                filename_nbdigits=nbdigits)

            print('nb indexed grains', nbgrains_indexed)
            DataSet.merge_fitfiles(nbgrains_indexed,
                                   corfilename=corfilename,
                                   dirname=fitfile_folder,
                                   removefiles=0,
                                   add_unindexed_spotslist=True)

            DataSet.writeFileSummary(corfilename=corfilename,
                                   dirname=fitfile_folder)

        dictspots[imageindex] = DataSet.getSummaryallData()

        if saveObject:
            dict_spotssetObj[imageindex] = DataSet

        # filling dicts with indexation results
        for grainindex in range(totalnb_grains):
            if grainindex in DataSet.indexedgrains:
                dictMaterial[imageindex][grainindex] = DataSet.dict_indexedgrains_material[grainindex]
                dictMat[imageindex][grainindex] = DataSet.dict_grain_matrix[grainindex]
                dictMR[imageindex][grainindex] = DataSet.dict_grain_matching_rate[grainindex][1]
                dictNB[imageindex][grainindex] = DataSet.dict_grain_matching_rate[grainindex][0]
                dictstrain[imageindex][grainindex] = DataSet.dict_grain_devstrain[grainindex]

        if 1:  # verbose:
            print("dictMaterial", dictMaterial)
            print("dictMat", dictMat)
            print("dictMR", dictMR)
            print("dictNB", dictNB)
            print("dictstrain", dictstrain)
        # intermediate saving
        if (imageindex % 10) == 0:
#            dictRes = dictMat, dictMR, dictNB
            filepickle = open(os.path.join(ResultsFolder, outputdict_filename), 'w')
            pickle.dump(todump, filepickle)
            filepickle.close()

        lastindex = imageindex

        # ind, 2theta, chi, posx, posy, int
#        spots_to_remove = DataSet.getSpotsFamilyExpData(0).T

    filepickle = open(os.path.join(ResultsFolder, 'LUT'), 'w')
    pickle.dump(DataSet.LUT, filepickle)
    filepickle.close()


    filepickle = open(os.path.join(ResultsFolder, outputdict_filename), 'w')
    pickle.dump(todump, filepickle)
    filepickle.close()

#        DataSet.plotallgrains()

    print('************************\n\n\n\n\n\n\nCompleted process for %s:' % \
        str([fileindexrange[0], fileindexrange[1] + 1, fileindexrange[2]]), p.name, p.pid)

#     with open(os.path.join(ResultsFolder, 'indexrefine.log'), 'a') as logfile:
#         logfile.write(outputdict_filename)

    if build_hdf5:

        from . import Lauehdf5 as LaueHDF5

        LaueHDF5.build_hdf5(outputdict_filename,
                                dirname_dictRes=ResultsFolder,
                                output_hdf5_filename="dict_Res_%s.h5" % prefixfortitle,
                                output_dirname=ResultsFolder)

    return todump, outputdict_filename
#    return DataSet_Si, DataSet

#--- -----------  Config file .irp file I/O  (Index Refine Parameters)
LIST_OPTIONS_INDEXREFINE = ['nb materials',
                            'key material',
                             'nbGrainstoFind',
                             'emin',
                            'emax',
                            'MATCHINGRATE THRESHOLD IAL',
                            'AngleTolLUT',
                            'MATCHINGRATE ANGLE TOL',
                            'NBMAXPROBED',
                            'MinimumNumberMatches',
                            'central spots indices',
                            'ResolutionAngstrom',
                            'nLUTmax',
                            'setCentralSpotsHKL',
                            'UseIntensityWeights',
                            'nbSpotsToIndex',
                            'List Matching Tol Angles',
                            'Spots Order Reference File']

CONVERTKEY_dict = {'nb materials': 'nb materials',
                   'key material': 'key material',
              'nbgrainstofind': 'nbGrainstoFind',
              'emin': 'emin',
              'emax': 'emax',
              'matchingrate threshold ial': 'MATCHINGRATE THRESHOLD IAL',
              'angletollut': 'AngleTolLUT',
              'matchingrate angle tol': 'MATCHINGRATE ANGLE TOL',
              'nbmaxprobed': 'NBMAXPROBED',
              'minimumnumbermatches':'MinimumNumberMatches',
              'central spots indices': 'central spots indices',
              'resolutionangstrom': 'ResolutionAngstrom',
              'nlutmax': 'nLUTmax',
              'setcentralspotshkl': 'setCentralSpotsHKL',
              'useintensityweights':'UseIntensityWeights',
              'nbspotstoindex':'nbSpotsToIndex',
              'list matching tol angles':'List Matching Tol Angles',
              'spots order reference file':'Spots Order Reference File'}

LIST_OPTIONS_TYPE_INDEXREFINE = ['int',
                                 'string',
                             'int',
                             'float',
                            'float',
                            'float',
                            'float',
                            'float',
                            'int',
                            'int',
                            'string',
                            'string',
                            'integer',
                            'string',
                            'string',
                            'int',
                            'string',
                            'string']

# WARNING when adding parameters above:
# check in index_refine.py if field position is correct in def hascorrectvalue(self, kk, val):
# and name MUST NOT CONTAIN "_"


def saveIndexRefineConfigFile(dict_param, outputfilename=None):
    # save peaksearch parameter in config file
    config = CONF.RawConfigParser()
    config.add_section('IndexRefine')

    params_comments = 'Index and Refinement parameters\n'

    config.set('IndexRefine', 'nb materials', str(len(dict_param)))

    for kk in range(len(dict_param)):
        for key, val in list(dict_param[kk].items()):
            key_mat = key + '_%d' % kk
            params_comments += '# ' + key_mat + ' : ' + str(val) + '\n'
            config.set('IndexRefine', key_mat, str(val))

    if outputfilename is None:
        outputfilename = 'IndexRefine.irp'

    if not outputfilename.endswith('.irp'):
        if outputfilename.count('.') > 0:

            outputfilename = ''.join(outputfilename.split('.')[:-1] + '.irp')
        else:
            outputfilename += '.irp'


    # Writing configuration file to 'PeakSearch.cfg'
    with open(outputfilename, 'wb') as configfile:
        config.write(configfile)

    return outputfilename


def readIndexRefineConfigFile(filename):
    config = CONF.RawConfigParser()
    config.optionxform = str
#    config = MyCasePreservingConfigParser()

    config.read(filename)

    section = config.sections()[0]

    if section not in ('IndexRefine',):
        raise ValueError("wrong section name in config file %s. Must be in %s" % \
                            (filename, 'IndexRefine'))

    print("section", section)

    dict_param = {}

    list_options = config.options(section)
    list_options_gen = []
    list_material = []
    for opt in config.options(section):
        splt_opt = opt.split('_')
        if len(splt_opt) == 2:
            optgen, mat_index = splt_opt
            list_options_gen.append(optgen)
            list_material.append(int(mat_index))
        elif len(splt_opt) > 2:
            raise ValueError("WARNING name of fields in GUI must not contain '_'!!")

    nb_materials = int(config.getint(section, 'nb materials'))

    dict_param = [{} for k in range(nb_materials)]

#     print "list_options_gen, list_material"
#     print list_options_gen, list_material

    for option, matindex in zip(list_options_gen, list_material):

        if matindex >= nb_materials:
            continue

        print("\n option, matindex\n", option, matindex)
        for option_ref, option_type in zip(LIST_OPTIONS_INDEXREFINE,
                                           LIST_OPTIONS_TYPE_INDEXREFINE):

#             print "option_ref, option_type", option_ref, option_type

            if option_ref == option or \
                option_ref.lower() == option:

#                 print "BINGO read %s" % option_ref

                try:
                    optionkey = CONVERTKEY_dict[option_ref]
                except KeyError:
                    optionkey = option_ref

#                 print 'optionkey', optionkey
#                 print "option, option_type", option_ref, option_type

                option_lower = option_ref.lower() + '_%d' % matindex
                try:
                    if option_type == 'int':
                        val = int(config.getint(section, option_lower))
                        
                    elif option_type == 'float':
                        val=float(config.getfloat(section, option_lower))
                    else:
                        val=config.get(section, option_lower)
                    
                    print('val',val)
                    dict_param[matindex][optionkey] = val

#                     print "matindex", matindex
#                     print "optionkey", optionkey
#                     print "option_lower", option_lower
                except ValueError:
                    print("Value of option '% s' has not the correct type" % option)
                    return None

                break

    return dict_param


if __name__ == '__main__':
    
    if 0:
        initialparameters = {}
    
        LaueToolsProjectFolder = os.path.abspath(os.curdir)
    
        print('LaueToolProjectFolder', LaueToolsProjectFolder)
    
        MainFolder = os.path.join(LaueToolsProjectFolder, 'Examples', 'GeGaN')
    
        print("MainFolder", MainFolder)
    
        initialparameters['PeakList Folder'] = os.path.join(MainFolder,
                                                    'datfiles')
        initialparameters['IndexRefine PeakList Folder'] = os.path.join(MainFolder,
                                                            'fitfiles')
        initialparameters['PeakListCor Folder'] = os.path.join(MainFolder,
                                                   'corfiles')
        initialparameters['PeakList Filename Prefix'] = 'nanox2_400_'
        initialparameters['IndexRefine Parameters File'] = os.path.join(MainFolder,
                                                        'GeGaN.irp')
        initialparameters['Detector Calibration File .det'] = os.path.join(MainFolder,
                                                        'calibGe_nanowMARCCD165.det')
        initialparameters['Detector Calibration File (.dat)'] = os.path.join(MainFolder,
                                                         'nanox2_400_0000_LT_1.dat')
        initialparameters['PeakList Filename Suffix'] = '.dat'
    
    
    
        print(initialparameters['IndexRefine Parameters File'])
    
    
        dict_param_list = readIndexRefineConfigFile(initialparameters['IndexRefine Parameters File'])
    
        CCDparams, calibmatrix = IOLT.readfile_det(initialparameters['Detector Calibration File .det'],
                                                      nbCCDparameters=8)
    
        Index_Refine_Parameters_dict = {}
        Index_Refine_Parameters_dict['CCDCalibParameters'] = CCDparams[:5]
        Index_Refine_Parameters_dict['pixelsize'] = CCDparams[5]
        Index_Refine_Parameters_dict['framedim'] = CCDparams[6:8]
    
        Index_Refine_Parameters_dict['PeakList Folder'] = initialparameters['PeakList Folder']
        Index_Refine_Parameters_dict['PeakListCor Folder'] = initialparameters['PeakListCor Folder']
        Index_Refine_Parameters_dict['nbdigits'] = 4
        Index_Refine_Parameters_dict['prefixfilename'] = initialparameters['PeakList Filename Prefix']
        Index_Refine_Parameters_dict['suffixfilename'] = '.dat'
        Index_Refine_Parameters_dict['prefixdictResname'] = initialparameters['PeakList Filename Prefix'] + \
                                                            '_dict_'
    
        Index_Refine_Parameters_dict['PeakListFit Folder'] = initialparameters['IndexRefine PeakList Folder'] + '_test'
        Index_Refine_Parameters_dict['Results Folder'] = initialparameters['IndexRefine PeakList Folder'] + '_test'
    
        Index_Refine_Parameters_dict['dict params list'] = dict_param_list
        Index_Refine_Parameters_dict['Spots Order Reference File'] = None
    
    
        print(Index_Refine_Parameters_dict)
    
        index_fileseries_3((0, 1, 1),
                           Index_Refine_Parameters_dict=Index_Refine_Parameters_dict,
                           saveObject=0,
                           verbose=0,
                           nb_materials=None,
                           build_hdf5=False,
                           prefixfortitle='',
                           use_previous_results=False,
                           CCDCalibdict=None)



    if 1:
        # E1_13_run2_0171.mccd
        Index_Refine_Parameters_dict = {}

        dict_params = [{'MATCHINGRATE THRESHOLD IAL':100.,
                       'MATCHINGRATE ANGLE TOL': 0.5,
                       'NBMAXPROBED': 30,
                       'central spots indices': 5,
                       'key material':'ZrO2Y2O3',
                       'emin':5,
                       'emax':23,
                       'nbGrainstoFind': 2}
                       ]
        
        CCDcalib = [68.61556108571234347,
                    940.77744180377760586, 1005.14714125015211721,
                    0.72531401975651599, -0.76181063603636234]
        
        CCDCalibdict = {}
        CCDCalibdict['CCDCalibParameters'] = CCDcalib
        CCDCalibdict['framedim'] = (2048, 2048)
        CCDCalibdict['detectordiameter'] = 165.
        
        Index_Refine_Parameters_dict['CCDCalibParameters'] = CCDcalib
        # , 0.08056640625000000, 2048.00000000000000000, 2048.00000000000000000]
        Index_Refine_Parameters_dict['pixelsize'] = 0.080566
        # 0.08057, 2048, 2048

        Index_Refine_Parameters_dict['dict params list'] = dict_params

        Index_Refine_Parameters_dict['PeakList Folder'] = '/home/micha/LaueTools/Examples/ZrO2'
        Index_Refine_Parameters_dict['PeakListCor Folder'] = '/home/micha/LaueTools/Examples/ZrO2'
        Index_Refine_Parameters_dict['nbdigits'] = 4
        Index_Refine_Parameters_dict['prefixfilename'] = 'E1_13_run2_'
        Index_Refine_Parameters_dict['suffixfilename'] = '.dat'
        Index_Refine_Parameters_dict['prefixdictResname'] = 'E1_13_run2_dict_'

        Index_Refine_Parameters_dict['PeakListFit Folder'] = '/home/micha/LaueTools/Examples/ZrO2'
        Index_Refine_Parameters_dict['Results Folder'] = '/home/micha/LaueTools/Examples/ZrO2'

        fileindexrange = (171, 177, 3)

        index_fileseries_3(fileindexrange, Index_Refine_Parameters_dict=Index_Refine_Parameters_dict,
              saveObject=0, CCDCalibdict=CCDCalibdict)
















