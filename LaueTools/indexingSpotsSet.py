# -*- coding: utf-8 -*-
"""
Class to handle indexing of a set of laue spots

this class belongs to the open-source LaueTools project
JS micha February 2020
"""

import copy
import pickle
import os
import sys
import time
import multiprocessing
import numpy as np
from pylab import figure, scatter, show, subplot, title

if sys.version_info.major == 3:
    import configparser as CONF
    from . import generaltools as GT
    from . import CrystalParameters as CP
    from . import lauecore as LAUE
    from . import indexingAnglesLUT as INDEX
    from . import matchingrate
    from . import dict_LaueTools as DictLT
    from . import FitOrient as FitO
    from . import LaueGeometry as F2TC
    from . import findorient as FO
    from . import indexingImageMatching as IMM
    from . import IOLaueTools as IOLT
    from . generaltools import printred, printgreen, printcyan
    from . import spottracking as SpTra
    from . import orientations as ORI

else:
    import ConfigParser as CONF
    import generaltools as GT
    import CrystalParameters as CP
    import lauecore as LAUE
    import indexingAnglesLUT as INDEX
    import matchingrate
    import dict_LaueTools as DictLT
    import FitOrient as FitO
    import LaueGeometry as F2TC
    import findorient as FO
    import indexingImageMatching as IMM
    import IOLaueTools as IOLT
    from generaltools import printred, printgreen, printcyan
    import spottracking as SpTra
    import orientations as ORI


CST_ENERGYKEV = DictLT.CST_ENERGYKEV

DEG = DictLT.DEG

MINIMUM_NB_MATCHES_FOR_INDEXING = 6

DEFAULT_KF_DIRECTION = "Z>0"


# ---  ---------  exp. SPOTS indexing management -------
class OrientMatrix:
    """
    class of orientation matrix which may have several representation
    """

    def __init__(self, matrix=None, eulers=None):
        if matrix is not None:
            self.matrix = matrix
            self.eulers = None
            if np.array(matrix).shape != (3, 3):
                raise ValueError("the following object is not a 3x3 matrix !").with_traceback(matrix)
        elif eulers is not None:
            self.eulers = eulers
            self.matrix = GT.fromEULERangles_toMatrix(eulers)


class spotsset:
    """
    class for exp. SPOTS indexing management
    """
    def __init__(self):
        """
        instantiate the class
        """
        # dict of spots properties
        # key is  spotindex and val a list of spot properties
        # val: [spotindex, 2theta, chi, X, Y, Intensity, prop0, prop1, prop2,..., H, K, L, Energy, graindindex, isindex]
        # main elements of val are val[:6] and from indexing  val[-6:]
        # see initIndexationDict()
        self.indexed_spots_dict = {}
        # corresponding dict  with key is props name and val column index in val of self.indexed_spots_dict
        self.dict_props_name = {}

        # dicts of completed or current indexation state (key is grain index)
        self.dict_grain_matrix = {}
        self.dict_grain_devstrain = {}
        self.dict_grain_devstrain_sample = {}
        self.dict_grain_Ts = {}
        self.dict_grain_matching_rate = {}
        self.dict_Missing_Reflections = {}
        self.dict_grain_twins = {}

        self.simulparameter = {}
        self.refpositionfilepath = None
        self.CCDcalibdict = {}
        self.framedim = None
        self.alldata = None

        self.ResolutionAngstromLUT = None
        self.n_LUT = 4
        self.B_LUT = None
        self.MissingReflindexedgrains = None
        self.nbMatricesInUBstack = 0
        self.UseIntensityWeights = False
        self.nbSpotsToIndex = 0
        self.MinimumNumberMatches = -1

        self.AngTol_LUTmatching = 0
        self.refinedTS = None
        self.deviatoricstrain = None
        self.deviatoricstrain_sampleframe = None
        self.new_latticeparameters = None

        self.refinedUBmatrix = None
        self.B0matrix = None
        self.pixelresidues = None
        self.spotindexabs = None
        self.residues_non_weighted = None


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

        self.emin, self.emax = 5, 23

        #materials
        self.key_material = None
        self.dict_Materials = DictLT.dict_Materials

        # LUT
        self.LUT = None

        # total number of exp. spots
        self.nbspots = None

        self.filename = None
        self.dict_IMM = None
        self.IMMdatabase = None

        # current spots data for indexation and refinement
        self.TwiceTheta_Chi_Int = None  # two theta and chi scattering angles and intensities array
        self.absolute_index = None  # absolute experimental spot index in the input spot list (.cor)

    #         self.updateSimulParameters()

    # def setSimulParameters(self, key_material, emin, emax, detectorparameters,
    #                                 pixelsize, dim, dictmaterials=DictLT.dict_Materials):
    #     r"""
    #     NOT USED !!!!
    #     set simulation parameters

    #     .. todo:: should be written in a more elegant way

    #     :param key_material: element or material key
    #     :type key_material: string
    #     :param emin: minimum energy bandpass
    #     :type emin: integer
    #     :param emax: maximum energy bandpass
    #     :type emax: integer
    #     :param detectorparameters: 5 elements
    #     :type detectorparameters: list, tuple, array of floats
    #     :param pixelsize: pixel size (mm) if squared pixel
    #     :type pixelsize: float
    #     :param dim: CCD nb of pixels 2 elements (dim1,dim2)
    #     :type dim: list, tuple, array of integers
    #     """

    #     self.setMaterial(key_material, dictmaterials=dictmaterials)

    #     self.emin = emin
    #     self.emax = emax
    #     self.detectorparameters = detectorparameters
    #     self.pixelsize = pixelsize
    #     self.dim = dim

    #     self.updateSimulParameters()

    def setImageMatchingParameters(self, dict_IMM, database):
        """
        set parameters for image matching technique

        :param dict_IMM: dictionary of imagematching parameters
        :param database: database of templates
        """
        self.dict_IMM = dict_IMM
        self.IMMdatabase = database

    def setMaterial(self, key_material, dictmaterials=DictLT.dict_Materials):
        """
        set material and dict of Materials for indexing
        :param key_material: material or element key (see dict_Lauetools)
        :type key_material: string
        """
        assert key_material is not None
        self.key_material = key_material
        self.dict_Materials = dictmaterials

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
        params = ("key_material",
                    "emin",
                    "emax",
                    "detectorparameters",
                    "pixelsize",
                    "dim",
                    "kf_direction",
                    "detectordiameter")
        try:
            values = (self.key_material,
                        self.emin,
                        self.emax,
                        self.detectorparameters,
                        self.pixelsize,
                        self.dim,
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

        for key in ("key_material",
                        "emin",
                        "emax",
                        "detectorparameters",
                        "pixelsize",
                        "dim",
                        "kf_direction",
                        "detectordiameter"):
            print("%s   : %s" % (key, self.simulparameter[key]))
        return self.simulparameter

    def importdata(self, exp_data):
        """ old name of set_dict_spotprops"""
        self.set_dict_spotprops(exp_data)

    def set_dict_spotprops(self, exp_data, add_props=None):
        """
        initialize spots indexation dictionary from exp_data

        set self.indexed_spots_dict and self.dict_props_name

        :param exp_data: array of 5 elements tth, chi, Intensity, posX, posY

        :param add_props: tuple of spots properties: (data, colnames) data is
        a list of properties values. nb of values = nb of spots

        .. note:: tth, chi are the twotheta and chi scattering angles.
            They must correspond to posX and poxY (pixel position on detector) through calibration
        """
        self.indexed_spots_dict, self.dict_props_name = initIndexationDict(exp_data, add_props)

    def purgedata(self, twicethetaChi_to_remove, dist_tolerance=0.2):
        """
        purge experimental spots (already set in spot dictionary)
        of those present in twicethetaChi_to_remove

        twicethetaChi_to_remove : array of two elements [tth, chi]

        .. warning:: Not used yet! and does not handle additional spots properties
        """
        exp_data = self.getSpotsallData()[:, 1:]
        print("Before purge len(exp_data)", exp_data.shape)
        self.indexed_spots_dict, self.dict_props_name = purgeSpotsinDict(exp_data.T,
                                                                    twicethetaChi_to_remove,
                                                                    dist_tolerance=dist_tolerance)
        # update the nb of spots
        self.nbspots = len(self.getSpotsallData()[:, 0])

        print("after purge self.nbspots", self.nbspots)

    def importdatafromfile(self, filename, refpositionfilepath=None, trackingmode=0):
        """
        Read .cor file and initialize spots indexation dictionary from peaks list:
        ie self.indexed_spots_dict  taken into account a reference file

        optionnaly: write a 'REF_*******.cor' file and set self.refpositionfilepath accordingly

        :param filename: fullpath to file (str)
        :param TrackingMode: 0   then refpositionfilepath is unchanged when indexing several images
        :param refpositionfilepath: filename of peaks pixel position list. Current loaded spots
        will be selected and aranged according to this reference list.
        in trackingmode:
        A 'REF_*******.cor' is also written enabling spots tracking (by position [posx,posy]) over a set of files.
        This file is self.refpositionfilepath

        set  spots data set, namely:

        self.nbspots
        self.indexed_spots_dict  from calling self.set_dict_spotprops()
        calibration and geometry parameters:
            - self.pixelsize
            - self.dim
            - self.CCDLabel
            - self.detectordiameter
            - self.kf_direction
            - ...
        """
        print("Import Data for DATASET indexation procedure")

        print("filename in importdatafromfile()", filename)
        (allspotsprops,
        data_theta, Chi, posx, posy, dataintensity,
        detectorparameters,
        CCDcalibdict) = IOLT.readfile_cor(filename, output_CCDparamsdict=True)
        print('nb of spots in .cor file %s  is '%filename, len(data_theta))

        add_props = IOLT.get_otherspotprops(allspotsprops, filename)
        if add_props is None:
            print('raw .cor file with 5 columns')

        if isinstance(data_theta, (float, int)):
            nb_spots = 1
            print("%s contains a single peak" % filename)
            self.nbspots = nb_spots
            return False

        if refpositionfilepath not in (None, "None"):
            if trackingmode:
                # TRacking mode , write .cor file for evolving reference list  (moving Laue spots position from image to image)
                print("tracking spots of %s"%refpositionfilepath)
                (data_theta, Chi,
                posx, posy, dataintensity,
                new_order_spotindices,
                _,
                _,
                ) = SpTra.sortSpotsDataCor(data_theta, Chi, posx, posy, dataintensity,
                                                                                    refpositionfilepath)
                # for elem in (data_theta, Chi, posx, posy, dataintensity):
                #     print('len   len len = ',len(elem))
                # filter add_props
                if add_props is not None:
                    print('filtering spots props')
                    dataprops, colnames = add_props
                    arrayprops = np.array(dataprops).T
                    add_props = arrayprops[new_order_spotindices].T, colnames
                    #print('add_props',add_props)

                # print('AFTER posx', posx)
                # print("isolatedspots", isolatedspots)
                # print("isolatedspots_ref", isolatedspots_ref)

                # write refposfile  .cor file     'REF_------.cor
                dir_reffile, file_reffile = os.path.split(filename)
                refposfileprefix = os.path.join(dir_reffile, 'REF_' + file_reffile[:-4])
                IOLT.writefile_cor(refposfileprefix,
                                2*data_theta, Chi, posx, posy, dataintensity,
                                data_props=None, # add_props,
                                param=CCDcalibdict,
                                overwrite=1)
                refposfile = refposfileprefix + '.cor'

                print('refposfile', refposfile)

                self.refpositionfilepath = refposfile
            else: #not tracking mode, ie fixed reference peak list file
                (data_theta, Chi,
                posx, posy, dataintensity,
                new_order_spotindices,
                _,
                _,
                ) = SpTra.sortSpotsDataCor(data_theta, Chi, posx, posy, dataintensity,
                                                                                    refpositionfilepath)
                self.refpositionfilepath = refpositionfilepath

            print('nb of selected or TRACKED spots in file %s  is '%filename, len(data_theta))
            print('self.refpositionfilepath', self.refpositionfilepath)

        nb_spots = len(data_theta)

        Twicetheta = 2.0 * data_theta

        # init data to be indexed in a dict
        # print('len Twicetheta',len(Twicetheta), Twicetheta[:10])
        # print('add_props', add_props)
        self.set_dict_spotprops([Twicetheta, Chi, dataintensity, posx, posy],
                                                            add_props=add_props)
        warningflag = False
        self.CCDcalibdict = CCDcalibdict
        self.CCDLabel = self.CCDcalibdict['CCDLabel']

        if self.CCDLabel not in (None, 'None'):
            self.pixelsize = DictLT.dict_CCD[self.CCDLabel][1]
        else:
            for key in ('pixelsize', 'xpixelsize', 'ypixelsize'):
                if key in self.CCDcalibdict:
                    self.pixelsize = self.CCDcalibdict[key]
                    break
        if 'framedim' in self.CCDcalibdict:
            self.dim = self.CCDcalibdict["framedim"]
        else:
            self.dim = DictLT.dict_CCD[self.CCDLabel][0]

        self.framedim = self.dim

        if "detectordiameter" in self.CCDcalibdict:
            self.detectordiameter = self.CCDcalibdict["detectordiameter"]
        else:
            self.detectordiameter = max(self.dim) * self.pixelsize

        if "kf_direction" in self.CCDcalibdict:
            self.kf_direction = self.CCDcalibdict["kf_direction"]
        else:
            textwarning = "\n\n*******\n "
            textwarning += "warning: Laue geometry wasn't specified in %s" % filename
            textwarning += "\nSo using default top reflection geometry: \n%s\n*****\n\n" % DEFAULT_KF_DIRECTION
            printred(textwarning)

            self.kf_direction = DEFAULT_KF_DIRECTION
            warningflag = textwarning

        self.detectorparameters = detectorparameters
        self.nbspots = nb_spots
        self.filename = filename

        if self.detectorparameters is None:
            printred("file %s does not contain the 5 detector parameters" % filename)
            return

        return warningflag

    def getSpotsallData(self):
        """
        get all data of experimental spots (only the 6 first ones)

        set self.data

        :return: array where columns are
        absolute spot index, tth, chi, posX, posY, intensity
        """
        data = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            index, tth, chi, posX, posY, intensity = self.indexed_spots_dict[key_spot][:6]
            data.append([index, tth, chi, posX, posY, intensity])
        arr_data = np.array(data)
        self.alldata = arr_data

        return arr_data

    def getSpotsExpData(self, selectedspots_index=None):
        """
        return (2theta, chi, intensity) of all experimental spots
        i.e.

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
        get spots indices that belong to the same grain

        :return: list of spot indices (int)
        """
        c = -1 # index column flag isindexed 1, otherwise 0
        cg = -2 # index column grainindex
        spots_set = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # spot has been indexed
            if self.indexed_spots_dict[key_spot][c] == 1:
                if self.indexed_spots_dict[key_spot][cg] == grain_index:
                    spots_set.append(key_spot)

        return spots_set

    def getUnIndexedSpots(self):
        """
        return spots indices for which indexation has not been completed

        or in progress of indexing  grain

        self.indexedgrains is grainindex list of completed indexations
        """
        c = -1  # index column flag 'isindexed':  0 unindexed , 1 already indexed
        cg = -2  # column of grain index

        # print('self.indexed_spots_dict in getUnIndexedSpots',self.indexed_spots_dict)

        unindexed_spots_indices = []
        for k_spot in sorted(self.indexed_spots_dict.keys()):
            # print('getUnIndexedSpots',self.indexed_spots_dict[k_spot])

            # not indexed
            if self.indexed_spots_dict[k_spot][c] == 0:
                unindexed_spots_indices.append(k_spot)
            # indexed
            elif self.indexed_spots_dict[k_spot][c] == 1:
                # but actually indexing is in progress only
                if self.indexed_spots_dict[k_spot][cg] not in self.indexedgrains:
                    unindexed_spots_indices.append(k_spot)

        return unindexed_spots_indices

    def getSpotsFamilyallData(self, grain_index):
        """
        return all data of experimental spots for one grain

        Reader of self.indexed_spots_dict

        :param grain_index: grain index (int)

        :return: array   (nb spots, nb properties)
        """
        c = -1 # index column flag isindexed 1, otherwise 0
        cg = -2 # index column flag isindexed 1, otherwise 0
        data = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # flag=1 implies spot has been indexed
            if self.indexed_spots_dict[key_spot][c] == 1:
                # spot belong to the grain #grain_index
                if self.indexed_spots_dict[key_spot][cg] == grain_index:
                    lcol = (0, 1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1)
                    data.append([self.indexed_spots_dict[key_spot][i] for i in lcol])

        return np.array(data)

    def getSummaryallData(self):
        """
        return all data of experimental spots (indexed and not indexed spot)

        Read of self.indexed_spots_dict

        :return: array of spots properties. Shape= (nb spots, nb props)
        """
        c = -1 # col isindexed
        data = []
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            spot_props = self.indexed_spots_dict[key_spot]

            # spot has not been indexed
            if spot_props[c] != 1:
                # set H,K,L, Energy to 10000
                spot_props[-6:-6+4] = 10000, 10000, 10000, 10000
            data.append(spot_props)

        return np.array(data)

    def getallIndexedSpotsallData(self):
        """
        return all indexed spots properties for all indexed grains
        """
        DataGraindict = {}
        for grainindex in self.indexedgrains:
            datagrain = self.getSpotsFamilyallData(grainindex)
            DataGraindict[grainindex] = datagrain

        return DataGraindict

    def getSpotsFamilyExpData(self, grain_index):
        """
        return useful data of experimental spots for one grain
        """
        alldata = self.getSpotsFamilyallData(grain_index)

        #        print 'alldata', alldata
        #        print 'alldata', type(alldata)
        if not alldata:
            return None
        else:
            ind, tth, chi = (alldata[:, :3]).T
            intensity = alldata[:, 5]

            return np.array([ind, tth, chi, intensity]).T

    def getUnIndexedSpotsallData(self, exceptgrains=None):
        """
        return all data of unindexed experimental spots
        and those already indexed but not from grains in 'exceptgrains' list

        Reader of self.indexed_spots_dict (only the 6 main elements)

        :param exceptgrains: list of integers or integer, grains for which indexing is in progress (still unindexed)

        return:
        array whose columns are: absolute spot index, tth, chi, posX, posY, intensity
        """

        # if exceptgrains is integer
        if not isinstance(exceptgrains, list):
            exceptgrains = [exceptgrains]

        data = []

        nbspots = 0
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            grain_origin = isindexed(key_spot, self.indexed_spots_dict)
            # spot already indexed, guessed to be indexed or being indexed
            if isinstance(grain_origin, int):
                if grain_origin not in exceptgrains:
                    data.append(self.indexed_spots_dict[key_spot])
                    nbspots += 1
            # spot not indexed at all
            elif grain_origin is None:
                data.append(self.indexed_spots_dict[key_spot])
                nbspots += 1

        if nbspots == 1:
            return np.array([data])
        elif nbspots == 0:
            # shape = (0,)
            return np.array([])
        else:
            return np.array(data)

    def getSpotsFromSpotIndices(self, spotindices):
        r"""
        return all data of experimental spots from their index

        :param spotindices: list of integers or integer

        :return:
        [0] array of 3 elements: twicetheta_data, chi_data, intensity_data
        [1] array of integer of corresponding spot index
        """
        # print("self.indexed_spots_dict in getSpotsFromSpotIndices", self.indexed_spots_dict)

        arr = np.zeros((len(spotindices), 4))
        for k, spot_index in enumerate(spotindices):
            # absoluteindex, twthe, chi, intensity_data
            arr[k] = [self.indexed_spots_dict[spot_index][i] for i in (0, 1, 2, 5)]

        absoluteindex = np.array(arr[:, 0], dtype=np.int)

        # np.array([twicetheta_data, chi_data, intensity_data]), absoluteindex
        return arr[:, 1: 4].T, absoluteindex

    def getSelectedExpSpotsData(self, exceptgrains=None):
        """
        return Exp. data spots (scattering angles, intensity and spot index)  of unindexed spots and
        indexed spots that grain index does  not appear in exceptgrains

        :param exceptgrains: list of integers or integer of grain index

        :return:
        [0] array of 3 elements: twicetheta_data, chi_data, intensity_data
        [1] array of integer of corresponding spot index
        """
        # extract data not yet indexed or temporarly indexed
        toindexdata = self.getUnIndexedSpotsallData(exceptgrains=exceptgrains)
        nbspots = toindexdata.shape[0]

        if nbspots == 0:
            return np.array([[], [], []]), np.array([])
        if nbspots == 1:
            print('\n\n warning single spot in getSelectedExpSpotsData()')

        # print("toindexdata", toindexdata)
        # print(toindexdata.shape)

        absoluteindex, twicetheta_data, chi_data = toindexdata[:, :3].T
        intensity_data = toindexdata[:, 5]
        absoluteindex = np.array(absoluteindex, dtype=np.int)

        return np.array([twicetheta_data, chi_data, intensity_data]), absoluteindex

    def setSelectedExpSpotsData(self, exceptgrains=None, fromspotsindices=None):
        r"""
        select a set of spots to be refined (set scattering angles and spots indices)
        from unindexed spots and
        selected indexed spots whose grain index does not appear in exceptgrains

        if  exceptgrains = None        : set selected spots from all remaining
                                        spots that do not belong to
                                        an already indexed grain
                            None is equivalent to a very long list of integers

        fromspotsindices : list of spot indices (absolute indices) to be selected

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
        reset spots properties in self.indexed_spots_dict for spots
        belonging to the grain of index 'grain_index'

        Reader of self.indexed_spots_dict
        """
        c = -1  # col of flag isindexed
        cg = -2  # col of #grainindex
        for key_spot in sorted(self.indexed_spots_dict.keys()):
            # spot has been indexed
            if self.indexed_spots_dict[key_spot][c] == 1:
                if self.indexed_spots_dict[key_spot][cg] == grain_index:
                    self.indexed_spots_dict[key_spot][-6:] = [10000, 10000, 10000, 10000, -1, 0]

    def AssignHKL(self, Orientation, grain_index, AngleTol=1.0,
                        use_spots_in_currentselection=True, selectbyspotsindices=None, verbose=0):
        r"""
        Assign hkl to the exp spot data set according to
        the orientation matrix within the tolerance angle

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

        :return:
        matching_rate, nb_updates, missingRefs
        """
        MINIMUM_SPOTS = 8

        if verbose:
            print('\n----- start of AssignHKL() -------\n')
        if len(self.indexed_spots_dict) < MINIMUM_SPOTS:
            print("too few spots to index in spotsset object")
            return None, None, None

        if isinstance(Orientation, OrientMatrix):
            matrix = Orientation.matrix
            eulers = None
        else:
            matrix = GT.fromEULERangles_toMatrix(Orientation)
            eulers = Orientation

        # use predefined selection of spots that are
        # already contained in self.TwiceTheta_Chi_Int, self.absolute_index
        if not use_spots_in_currentselection:
            if verbose > 1:
                print("case 1")
            # exp data used from refined model
            data_1grain = self.getSpotsFamilyallData(grain_index)
            (index_r, tth_r, chi_r, _, _, intensity_r) = (data_1grain.T)[:6]

            self.TwiceTheta_Chi_Int = [tth_r, chi_r, intensity_r]

            useabsoluteindex = index_r
            selected_expdata = self.TwiceTheta_Chi_Int

        # use general selection of exp spots according to their indexation state
        # (a spot is selected if it does not belong to a grain (already indexed))
        elif use_spots_in_currentselection:
            if verbose > 1:
                print("case 2")
            # all remaining exp spots not already indexed
            # this sets: self.TwiceTheta_Chi_Int and self.absolute_index
            self.setSelectedExpSpotsData(exceptgrains=self.indexedgrains,
                                        fromspotsindices=selectbyspotsindices)

            selected_expdata = self.TwiceTheta_Chi_Int
            useabsoluteindex = self.absolute_index

        if verbose > 1:
            print("***nb of selected spots in AssignHKL*****", len(useabsoluteindex))
            print('matrix', matrix)

        AssignationHKL_res, nbtheospots, missingRefs = self.getSpotsLinks(matrix,
                                                                    exp_data=selected_expdata,
                                                                    useabsoluteindex=useabsoluteindex,
                                                                    removeharmonics=1, # for fast computations
                                                                    ResolutionAngstrom=False,#
                                                                    veryclose_angletol=AngleTol,
                                                                    verbose=verbose)
        # TODO nb of links with getSpotsLinks() larger than nb of links used in previous refinement
        # self.pixelresidues
        #         print "AssignationHKL_res in AssignHKL", AssignationHKL_res

        if verbose > 2:
            print("missingRefs", missingRefs)

        if AssignationHKL_res is not None:
            nb_updates = self.updateIndexationDict(AssignationHKL_res, grain_index, overwrite=1)

            matching_rate = 100.0 * nb_updates / nbtheospots

            if verbose:
                print("\n")
                if eulers is not None:
                    print("for this three euler angles [%.1f,%.1f,%.1f]" % tuple(Orientation))
                print('with matrix', matrix)

                print("nb indexed spots %d / %d (theo. nb)" % (nb_updates, nbtheospots))
                print("matching rate  : %.1f" % matching_rate)
                print("with tolerance angle : %.2f deg" % AngleTol)

            # update dictionaries
            self.dict_grain_matrix[grain_index] = matrix
            self.dict_grain_matching_rate[grain_index] = [nb_updates, matching_rate]
            self.dict_Missing_Reflections[grain_index] = missingRefs

            #            print "Low value of indexed spots may mean that"
            #            print "spots have been already assigned to other grain"

            if verbose:
                print('\n----- end of AssignHKL() -------\n')
            return matching_rate, nb_updates, missingRefs

        else:
            print("no grains have been found !")
            return None, None, None

    def Assigntwinnedgrain(self, mothergrain_matrix, grain_index, AngleTol,
                                MINIMUM_MATCHINGRATE, minimumMR_step, verbose=0):
        """
        deal with spots assignement in case of amiguities (such as sigma3 twins)

        minimumMR_step = MatchingRate_List[step_refinement_index]

        .. note:: to be checked
        """
        twins_matrices = get4sigma3Matrices(mothergrain_matrix)

        # simulate sigma3 twins and check corresponding matching rate
        nb_of_twins = 0
        for indextwin, twin_mat in enumerate(twins_matrices):

            # calculate matching without indexing possible with indexonegrain?
            twinnedMatrix = OrientMatrix(matrix=twin_mat)
            Matching_rate, nb_updates, missingRefs = self.AssignHKL(twinnedMatrix,
                                                                grain_index,
                                                                AngleTol,
                                                                verbose=verbose)

            #                                print "missingRefs", missingRefs
            print("Matching_rate, nb_updates, missingRefs %.1f %d %d with sigma3 %d"
                % (Matching_rate,
                    nb_updates,
                    len(missingRefs[1]),
                    indextwin))

            GoodRefinement = (Matching_rate > MINIMUM_MATCHINGRATE) or (nb_updates >= 6)
            self.dict_grain_twins[grain_index] = []
            if (Matching_rate
                >= minimumMR_step
                and GoodRefinement):
                self.dict_grain_twins[grain_index].append([twinnedMatrix, Matching_rate, nb_updates])

                nb_of_twins += 1

            print("Found %d Sigma" % nb_of_twins)

        self.AssignHKL(mothergrain_matrix,
                        grain_index,
                        AngleTol,
                        verbose=verbose)

    def FindOrientMatrices(self, spot_index_central=(0, 1, 2), nbmax_probed=10,
                                                                nLUT=3,
                                                                LUT=None,
                                                                set_central_spots_hkl=None,
                                                                ResolutionAngstrom=False,
                                                                AngTol_LUTmatching=0.5,
                                                                MatchingRate_Angle_Tol=0.2,
                                                                Minimum_Nb_Matches=15,
                                                                verbose=0,
                                                                nb_of_solutions_per_central_spot=1,
                                                                simulparameters=None):
        """
        class method to Find orientation matrices by angles recognition
        (look up table of angles in reference structure)

        consider exp. spots that are unindexed  (call of getUnIndexedSpots())

        call of INDEX.getOrientMatrices in spotsset class

        .. note::
            USED in FileSeries
        """
        emax = self.emax

        key_material = self.key_material

        if simulparameters is None:
            simulparameters = {}
            simulparameters["kf_direction"] = self.kf_direction
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

        (abs_spotindex, tth, chi, _, _, _) = data_to_index.T

        nb_of_spots = len(abs_spotindex)

        # list of absolute index in spots list (.cor file) to be indexed
        current_exp_spot_index_list = np.arange(nb_of_spots)
        # absoluteindex = current_exp_spot_index_list[k]
        # nb of experimental spots that will be used to calculate the matching rate
        nbspotmaxformatching = nb_of_spots

        # there is no precomputed angular distances between spots
        if not Tabledist:
            # select 1rstly spots that have not been indexed and 2ndly reduced list by user
            index_to_select = np.take(
                current_exp_spot_index_list, np.arange(nbspotmaxformatching))

            select_theta = 0.5 * tth[index_to_select]
            select_chi = chi[index_to_select]
            # select_I = intensity[index_to_select]
            select_thetachi = np.array([select_theta, select_chi]).T
            # compute angles between spots
            Tabledistance = GT.calculdist_from_thetachi(select_thetachi, select_thetachi)

        latticeparams = DictLT.dict_Materials[key_material][1]
        B = CP.calc_B_RR(latticeparams)

        # indexation procedure
        bestmat, stats_res = INDEX.getOrientMatrices(spot_index_central,
                                                    emax,
                                                    Tabledistance[:nbmax_probed, :nbmax_probed],
                                                    select_theta,
                                                    select_chi,
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
                                                    verbosedetails=False,  # not CP.isCubic(key_material)
                                                    dictmaterials=self.dict_Materials,
                                                    LUT_with_rules=True)
        # when nbbestplot is very high  self.bestmat contain all matrices
        # with matching rate above Minimum_Nb_Matches

        if verbose:
            PrintMatchingResults(bestmat, stats_res)

        nb_sol = len(bestmat)

        print("Number of matrices found (nb_sol): ", nb_sol)

        keep_only_equivalent = CP.isCubic(DictLT.dict_Materials[key_material][1])

        if verbose:
            print("set_central_spots_hkl in FindOrientMatrices", set_central_spots_hkl)
        # TODO: anticipate  future pb if set_central_spots_hkl is a list of hkl and None...
        if set_central_spots_hkl not in (None, [None]):
            keep_only_equivalent = False

        if nb_sol > 1:
            if verbose:
                print("Merging matrices")
                print("keep_only_equivalent = %s" % keep_only_equivalent)
            bestmat, stats_res = MergeSortand_RemoveDuplicates(bestmat,
                                                                stats_res,
                                                                Minimum_Nb_Matches,
                                                                tol=0.0001,
                                                                keep_only_equivalent=keep_only_equivalent)

        if verbose:
            PrintMatchingResults(bestmat, stats_res)
        nb_sol = len(bestmat)
        #         print "Max. Number of Solutions", NBRP

        if nb_sol == 0:
            return [], []

        if verbose:
            print("Nb of potential orientation matrice(s) UB found: %d " % nb_sol)
            print(bestmat)
        stats_properformat = []
        for elem in stats_res:
            elem[0] = int(elem[0])
            elem[1] = int(elem[1])
            stats_properformat.append(tuple(elem))

        return bestmat, stats_properformat

    def IndexSpotsSet(self, file_to_index, key_material, emin, emax, dict_parameters, database,
                                                                starting_grainindex=0,
                                                                use_file=1,
                                                                verbose=0,
                                                                plotintermediateresults=0,
                                                                IMM=True,
                                                                nbGrainstoFind="max",
                                                                LUT=None,
                                                                n_LUT=3,
                                                                set_central_spots_hkl=None,
                                                                ResolutionAngstrom=False,
                                                                MatchingRate_List=[50, 60, 80],
                                                                angletol_list=[0.5, 0.2, 0.1],
                                                                checkSigma3=False,
                                                                previousResults=None,
                                                                CheckOrientations=None,
                                                                corfilename=None,
                                                                dirnameout_fitfile=None):
        r"""
        General class method to index a set of experimental spots.

        Guessed matrices can checked prior to proceed to indexation from scratch (previous Results)

        potential orientation matrix is eligible to refinement depending on matching rate

        Successive steps of refinement procedures are set by  angletol_list and MatchingRate_List

        :param file_to_index: filename of experimental spots list, this file is read if use_file == 1, else spots in self.indexed_spots_dict is considered
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
        :param nbGrainstoFind: number of grain to find for a given material.
                            If 'max' then find the largest number of grains in data set.
        :type nbGrainstoFind: integer
        :param MatchingRate_List: list of minimum matching rates that accept to perform the next refinement step (with less tolerance angle)
                            Last term is the final minimum matching rate (in percent) to accept the final refinement.
        :type MatchingRate_List: list of floats
        :param angletol_list: List of the maximum residual angular tolerances after each step of refinement.
        :type angletol_list: list of floats
        :param checkSigma3: flag to test if some twins sigma 3 may be present in data set (in devpt).
        :type checkSigma3: flag
        :param previousResults: if not None, 4 elements helping to use previous indexation
                            results as guesses for the current indexation :
                            nbmatrices, addMatrices, previousMatchingRate, previousNbspots
                            corresponding to initial guess of matrix, previous matching rate,
                            and previous Nb of indexed spots
        :type previousResults: None or tuple of integer, 3*3 array (matrix), float, integer
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

        self.ResolutionAngstromLUT = False
        if verbose:
            print("self.pixelsize in IndexSpotsSet", self.pixelsize)
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

        # initialisation
        grain_index = starting_grainindex  # current grain to be indexed
        NeedtoProvideNewMatrices = True
        indexgraincompleted = False
        self.UBStack = []
        self.nbMatricesInUBStack = 0
        frompreviousResults = False

        # default minimum matching rate to start refinement steps
        # (normally very small to accept a UB solution coming from self.UBstack
        # if self.UBstack is fed by previous results or checkorientation parameters (.ubs file)
        # MATCHINGRATE_FOR_PREVIOUSRESULTS is reset to the user value
        MATCHINGRATE_FOR_PREVIOUSRESULTS = 1.0

        # matrices that must be tested first before trying to index from scratch
        # these matrices can come from previously indexed Laue Pattern
        if previousResults is not None:
            # previousResults = (nbAddMatrices, addMatrices, MRs, NBs)
            
            _, addMatrices, _, _ = previousResults
            if verbose:
                print("considering previousResults with addMatrix", addMatrices)
            previousMatricesList = []
            for addMatrix in addMatrices:
                # sanity check
                if addMatrix is not None:
                    previousmatrix = OrientMatrix(matrix=addMatrix)
                    previousMatricesList.append(previousmatrix)

            self.UBStack = previousMatricesList
            NeedtoProvideNewMatrices = False
            fromIMM = IMM
            frompreviousResults = True

            if verbose:
                print("self.UBStack before while(1) loop")
                print(self.UBStack)
            self.nbMatricesInUBStack = len(self.UBStack)

            # minimum matching rate (nb matched spots/ nb total simulated)
            # for previous matrix at the last step of refinement (minimal tolerance angle)
            # if the number of theoretical reflections is high,
            # this parameter must be lowered significantly
            # TODO:  ? ? add this parameter to .irp file better than in Index_Refine.py GUI for instance
            MATCHINGRATE_FOR_PREVIOUSRESULTS = dict_parameters["MinimumMatchingRate"]  # 50.

        # matrices that must be tested first before trying to index from scratch
        # these matrices are proposed by a specific file .ubs  (see Guessed Matrix(ces) ... field in Index_Refine.py)
        if CheckOrientations is not None:
            if verbose:
                print("CheckOrientations is not None")
                print("CheckOrientations", CheckOrientations)
            CheckOrientation = CheckOrientations[0]
            Checkmatrices = CheckOrientation[5]
            self.key_material = CheckOrientation[2]
            self.emax = CheckOrientation[3]
            MATCHINGRATE_FOR_PREVIOUSRESULTS = CheckOrientation[4]

            self.UBStack = []
            for chckmatrix in Checkmatrices:
                matrixobj = OrientMatrix(matrix=chckmatrix)
                self.UBStack.append(matrixobj)

            self.nbMatricesInUBStack = len(self.UBStack)

            fromIMM = IMM

        if verbose:
            print("MATCHINGRATE_FOR_PREVIOUSRESULTS", MATCHINGRATE_FOR_PREVIOUSRESULTS)

        if nbGrainstoFind != "max" and not isinstance(nbGrainstoFind, int):
            raise ValueError(
                "nbGrainstoFind in IndexSpotsSet() method has a wrong value")

        # flag to use or not use intensity weights when refining model
        self.UseIntensityWeights = dict_parameters["UseIntensityWeights"]

        self.nbSpotsToIndex = dict_parameters["nbSpotsToIndex"]

        # minimum nb of matches for a given material to be selected
        # as potential solution for anglesLUT indexing method
        self.MinimumNumberMatches = dict_parameters["MinimumNumberMatches"]

        nbgrains_found = 0 # found but not necessarly indexed
        # ---------------------------------------------------------------
        # infinite loop to try raw indexing with matrices in self.UBstack
        # with material and nb of grains requested and a set of peaks
        # -----------------------------------------------------------
        indexing_from_scratch_counter = 0
        testUBs = copy.copy(self.UBStack)
        k_mat = -1 # index of mat in previous UB results 
        while 1:
            ProceedWithRefinement = False
            nb_remaining_spots = len(self.getUnIndexedSpots())

            if verbose:
                print("\n Remaining nb of spots to index for grain #%d : %d\n"
                % (grain_index, nb_remaining_spots))

            # exit the loop if too few spots
            # or when user requests to index a few number of grain
            if nb_remaining_spots <= MINIMUM_NB_SPOTS_FOR_INDEXING or (
                isinstance(nbGrainstoFind, int) and nbgrains_found == nbGrainstoFind) or indexing_from_scratch_counter == 1:
                if verbose:
                    print("%d spots have been indexed over %d"
                        % (totalnbspots - nb_remaining_spots, totalnbspots))
                    print("indexing rate is --- : %.1f percents"
                        % (100.0 * (totalnbspots - nb_remaining_spots) / totalnbspots))
                    print("indexation of %s is completed" % self.filename)
                if isinstance(nbGrainstoFind, int):
                    if verbose:
                        print("for the %d grain(s) that has(ve) been indexed as requested"
                            % nbGrainstoFind)
                        print("Leaving Index and Refine procedures...")
                break

            if verbose:
                print("\n ******")
                print("start to index grain #%d of Material: %s \n" % (grain_index, key_material))
                print("******\n")

            # now there should be matrices from previous results
            # or guessed by indexation methods.
            # the flag MatchingRateUBStackTooLow comes from bad refined results inside the following loop
            MatchingRateUBStackTooLow = True

            #print('grain_index = %d, testUBs ='%grain_index, testUBs)
            # condition is False if testUBs is empty OR MatchingRateUBStackTooLow is False

            while (((testUBs is not None) or (testUBs is not [None]))
                and testUBs
                and (self.nbMatricesInUBStack > 0)
                and MatchingRateUBStackTooLow is True):

                if verbose:
                    print("\n ---- %d Matrice(s) are candidates in the Matrix Stack ! ----"
                                                                        % len(testUBs))
                    print("\n  -----   Taking a new matrix from the matrices stack  -------")

                k_mat += 1
                UB = testUBs.pop(0)

                if verbose:
                    print("\nFor grain_index", grain_index)
                    print("Considering UB", UB.matrix)

                    # indexation of spots with raw (un refined) matrices & #update dictionaries
                    print("\n---------------checking matching rate for grain  #%d-----------------------"
                        % grain_index)
                    print("eulerangles", UB.eulers)
                    print("UB", UB)
                    print("UB.matrix", UB.matrix)

                # select data, link spots,
                # update spot dictionary,
                # update matrix dictionary
                MatchRate, _, _ = self.AssignHKL(UB, grain_index, AngleTol=AngleTol_0,
                                                            use_spots_in_currentselection=True,
                                                            verbose=verbose)
                if verbose:
                    print("before refinement MatchRate", MatchRate)
                    print('MATCHINGRATE_FOR_PREVIOUSRESULTS', MATCHINGRATE_FOR_PREVIOUSRESULTS)

                if MatchRate is None:
                    MatchRate = 0
                if MatchRate > MATCHINGRATE_FOR_PREVIOUSRESULTS:
                    MatchingRateUBStackTooLow = False
                    NeedtoProvideNewMatrices = False
                    ProceedWithRefinement = True
                    bestUB = UB

                    if verbose:
                        print("\n-*****-  Ready to structure refinement now !   -*****-\n")
                    print('bestUB chosen from previous results. matrix rank = %d'%k_mat)
                else:
                    NeedtoProvideNewMatrices = True

            # -----------------------------------------------------
            # --- need to find UB matrices from indexing techniques (from scratch)
            # -------------------------------------------------------
            if NeedtoProvideNewMatrices:
                print('\n\n------- **  INDEXING from SCRATCH  **--------------\n')
                frompreviousResults = False
                # ----------------------------------------------------
                # potential orientation solutions from template matching
                # ----------------------------------------------------
                if IMM:
                    print("providing new set of matrices with ImageMatching template technique")

                    # TODO: test if sigma3 symmetry exist between matrices
                    (bestUB,
                    _,
                    _) = self.getOrients_ImageMatching(MatchingRate_Threshold=MatchingRate_Threshold_IMM,
                                                                    exceptgrains=self.indexedgrains,
                                                                    verbose=verbose)

                    fromIMM = True
                # ---------------------------
                # Using Angles LUT matching technique based on indexingAnglesLUT.py
                # -----------------------------
                else:
                    # potential orientation solutions from angles LUT matching

                    (self.TwiceTheta_Chi_Int,
                    self.absolute_index) = self.getSelectedExpSpotsData(exceptgrains=self.indexedgrains)

                    nbspots = len(self.TwiceTheta_Chi_Int[0])
                    if verbose:
                        print("providing new set of matrices Using Angles LUT template matching")
                        print("nbspots", nbspots)

                    # set the matching rate above which loop of LUT matching is aborted
                    # this is highly probable this is a grain
                    MATCHINGRATE_THRESHOLD_IAL = dict_parameters["MATCHINGRATE_THRESHOLD_IAL"]  # 60
                    # set the tolerance angle for matching
                    MATCHINGRATE_ANGLE_TOL = dict_parameters["MATCHINGRATE_ANGLE_TOL"]  # 0.2
                    # set the number of most intense spot candidate to have a recognisable distance
                    NBMAXPROBED = dict_parameters["NBMAXPROBED"]  # 10
                    self.AngTol_LUTmatching = dict_parameters["AngleTolLUT"]
                    # set central list spots to compute distance from
                    spot_index_central_list = dict_parameters["central spots indices"]  # 10
                    # TODO add key dict_parameters['CENTRAL_SPOTS_LIST']
                    # or
                    # CENTRAL_SPOTS_LIST = dict_parameters['CENTRAL_SPOTS_LIST']
                    NBMAXPROBED = min(nbspots, NBMAXPROBED)
                    # set mutual angular distance from NBMAXPROBED first spots
                    # TODO: build self.table_angdist but not yet effective
                    self.setTable_Angdist(nbmax=NBMAXPROBED)
                    if verbose:
                        print("NBMAXPROBED", NBMAXPROBED)
                        print("set_central_spots_hkl", set_central_spots_hkl)
                    if set_central_spots_hkl is None and LUT is None:
                        # for angular distance LUT matching
                        self.setAnglesLUTmatchingParameters(LUT=self.LUT, n_LUT=self.n_LUT)

                    if max(spot_index_central_list) >= len(self.absolute_index):
                        print("central list of spots contains spots that do not belong the current list of spots to be indexed")
                        break

                    # find single best orientation matrix UB solution
                    (bestUB, _, _,
                        Threshold_reached,
                    ) = self.get_bestUB_fromAnglesLUT(spot_index_central=spot_index_central_list,
                                                        MatchingRate_Threshold=MATCHINGRATE_THRESHOLD_IAL,
                                                        MatchingRate_Angle_Tol=MATCHINGRATE_ANGLE_TOL,
                                                        nbmax_probed=NBMAXPROBED,
                                                        Minimum_Nb_Matches=self.MinimumNumberMatches,
                                                        LUT=self.LUT,
                                                        set_central_spots_hkl=set_central_spots_hkl,
                                                        ResolutionAngstrom=ResolutionAngstrom,
                                                        exceptgrains=self.indexedgrains,
                                                        verbose=verbose)
                    # a way to launch indexing from scratch one time. and it provides only one UB mrix
                    NeedtoProvideNewMatrices = False
                    indexing_from_scratch_counter += 1
                    
                    fromIMM = False

                if verbose:
                    print("\nWorking with a new stack of orientation matrices")

                if Threshold_reached:
                    if verbose:
                        print(" MATCHINGRATE_THRESHOLD_IAL= %.1f" % MATCHINGRATE_THRESHOLD_IAL)
                        print("has been reached! Indexing has been stopped")

                else:
                    if verbose:
                        print("MATCHINGRATE_THRESHOLD_IAL= %.1f" % MATCHINGRATE_THRESHOLD_IAL)
                        print("has not been reached! All potential solutions have been calculated")
                        print("taking the first one only.")

                # update (overwrite) candidate orientMatrix object list
                # print("bestUB object", bestUB)
                if bestUB is None:
                    if verbose:
                        print("\n #### No matrix available for refinement.####")
                        print("#### End of indexation and refinement for element: %s ####"
                            % key_material)
                    break

                #self.nbMatricesInUBStack = 1
                ProceedWithRefinement = True

            # ----------------------------------------------------------
            # --- ---------   REFINEMENT STEPS
            # ----------------------------------------------------------
            if not ProceedWithRefinement:
                break

            if verbose:
                print("\n\n---------------refining grain orientation and strain #%d-----------------"
                                                                            % grain_index)
            # loop over refinement steps of UB and strain with different angular matching tolerances
            # number of spots may change
            nbsteps = len(ANGLETOL_List)
            UBrefined = bestUB
            for step_refinement_index, AngleTol in enumerate(ANGLETOL_List):
                # step is counted from zero
                if verbose:
                    print("\n------  ------ refining grain #%d step ------  -------%d/%s\n"
                    % (grain_index, step_refinement_index, str(range(nbsteps))))

                #self.dict_grain_matrix[grain_index] = bestUB.matrix
                # select data, link spots,
                # update spot dictionary,
                # update matrix dictionary

                self.AssignHKL(UBrefined,
                                grain_index,
                                AngleTol=AngleTol,
                                use_spots_in_currentselection=True,
                                verbose=verbose)

                if verbose:
                    print('\n\njust before self.refineUBSpotsFamily(grain_index,------>>  '
                    'self.dict_grain_matrix[grain_index]', self.dict_grain_matrix[grain_index])
                    print('and material   :', self.key_material)
                # --- ------- Refine with old strain model (varying strain at the right of UB)
                refinedMatrix, devstrain = self.refineUBSpotsFamily(grain_index,
                                                                    self.dict_grain_matrix[grain_index],
                                                                    use_weights=self.UseIntensityWeights,
                                                                    nbSpotsToIndex=self.nbSpotsToIndex,
                                                                    verbose=verbose)
                refinedTs = None
                # --- ----------------------------

                Matching_rate, nb_updates, missingRefs = None, None, None
                if refinedMatrix is not None:
                    UBrefined = OrientMatrix(matrix=refinedMatrix)
                    #            print "UBrefined", UBrefined.matrix
                    self.dict_grain_matrix[grain_index] = UBrefined.matrix
                    self.dict_grain_devstrain[grain_index] = devstrain
                    self.dict_grain_devstrain_sample[grain_index] = self.deviatoricstrain_sampleframe
                    self.dict_grain_Ts[grain_index] = refinedTs
                    self.refinedTs = refinedTs
                    # select data, link spots, update spot dictionary, update matrix dictionary

                    selectedspots_index = True
                    # last step of refinementa assignHKL
                    if step_refinement_index == len(ANGLETOL_List) - 1:
                        selectedspots_index = False

                    (Matching_rate, nb_updates, missingRefs) = self.AssignHKL(UBrefined,
                                                                grain_index,
                                                                AngleTol=AngleTol,
                                                                use_spots_in_currentselection=selectedspots_index,
                                                                verbose=verbose)
                else:
                    Matching_rate = None

                MINIMUM_MATCHINGRATE = 10.0
                # MINIMUM_SPOTS_GRAIN = 6
                GoodRefinement = False
                if Matching_rate is not None:
                    GoodRefinement = (Matching_rate > MINIMUM_MATCHINGRATE) or (
                        nb_updates >= 6)

                # --------------------------------
                # case of poor matching rate (after refinement and making links)
                #------------------------------------
                if (Matching_rate is None
                            or Matching_rate < MatchingRate_List[step_refinement_index]
                            or refinedMatrix is None
                            or not GoodRefinement):
                    # matching rate too low
                    # then remove previous indexed data
                    print("matching rate too low after refining")

                    self.resetSpotsFamily(grain_index)
                    self.dict_grain_matrix[grain_index] = None
                    self.dict_grain_devstrain[grain_index] = None
                    self.dict_grain_devstrain_sample[grain_index] = None
                    self.dict_grain_Ts[grain_index] = None
                    self.refinedTs = None

                    if frompreviousResults:  # refinement in case of UB from previous results 
                        print("Previous matrix does not index this data")
                        print("start the normal indexation")
                        NeedtoProvideNewMatrices = True
                        previousResults = None
                        frompreviousResults = False
                        break

                    else:  # refinement in case of UB from indexing from scratch
                        NeedtoProvideNewMatrices = False
                        nbgrains_found == nbGrainstoFind # artificial way to go out the while clause


                    # there are at least one indexed grain and this dataset
                    # has not been already tested with the matrix stack
                    # so remove the corresponding data and restart the imagematching
                    if nbgrains_found > 0 and indexgraincompleted:

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
                # -------------------------------------------------------------
                # case of good matching rate
                #---------------------------------------------

                # ie matching rate (after refinement and making links) is higher than threshold
                # so we proceed to evaluate the success of indexing and refinement
                else:

                    # if this is the last tolerance step  (step is counted from zero)
                    if step_refinement_index == len(ANGLETOL_List)-1:
                        # addMatrix, previousMatchingRate, previousNbspots = previousResults
                        if previousResults is not None:
                            #                            if Matching_rate < previousMatchingRate:
                            if Matching_rate < MATCHINGRATE_FOR_PREVIOUSRESULTS:
                                # there could be other grain more important than that previously indexed
                                NeedtoProvideNewMatrices = True
                                previousResults = None

                                print("Previous matrix seems good BUT matching rate is not so high")
                                print("matching rate %f < %f = MATCHINGRATE_FOR_PREVIOUSRESULTS"
                                    % (Matching_rate, MATCHINGRATE_FOR_PREVIOUSRESULTS))
                                print("so restart the normal indexation")
                                break
                            else:
                                print("Find a grain already indexed in previous image")

                        if checkSigma3:
                            print("--------- Checking Twins sigma3 --------------------")
                            print("--------- Final spots hkl and grain assignation-----")
                            # this grain is considered now as indexed
                            mothergrain_matrix = self.dict_grain_matrix[grain_index]

                            self.Assigntwinnedgrain(mothergrain_matrix, grain_index, AngleTol,
                                        MINIMUM_MATCHINGRATE, MatchingRate_List[step_refinement_index])

                            NeedtoProvideNewMatrices = not fromIMM
                            self.indexedgrains.append(grain_index)
                            self.dict_indexedgrains_material[grain_index] = key_material
                            indexgraincompleted = True

                            grain_index += 1
                            nbgrains_found += 1
                        # -----------------------------------------------------------
                        # normal and usual way (no common peaks of two laue patterns from 2 distinct grains
                        # -----------------------------------------------------------
                        elif not checkSigma3:

                            if verbose:
                                print("\n---------------------------------------------")
                                print("indexing completed for grain #%d of %s with matching rate %.2f "
                                    % (grain_index, self.key_material, Matching_rate))
                                print("---------------------------------------------\n")
                            # this grain is considered now as indexed
                            # corresponding spots will be no more used for the next indexation
                            self.indexedgrains.append(grain_index)

                            # find single representation of UB and reset h,k,l accordingly
                            if set_central_spots_hkl is None and CP.hasCubicSymmetry(self.key_material,
                                                                        dictmaterials=self.dict_Materials):

                                # Transform matrix and recompute everything
                                #  TODO: this operation should/could be avoided when using CheckOrientation (given orientation or equivalently precise values of HKL)
                                matrix = self.refinedUBmatrix
                                UBsingle, transfmat = FO.find_lowest_Euler_Angles_matrix(matrix)

                                # update self.dict_grain_matrix[grain_index] (used later to write results in file)
                                self.dict_grain_matrix[grain_index] = UBsingle

                                # recompute HKL according to this new matrix
                                datafamily = self.getSpotsFamilyallData(grain_index)
                                index, H, K, L = np.take(datafamily, (0, -6, -5, -4), axis=1).T

                                hkl = np.array([H, K, L]).T
                                hklmin = np.dot(transfmat, hkl.T).T

                                for kspot, exp_spot_index in enumerate(index):
                                    hh, kk, ll = hklmin[kspot]
                                    self.indexed_spots_dict[exp_spot_index][-6: -6 + 3] = [hh, kk, ll]

                            # write fit file for one grain
                            if verbose:
                                print("writing fit file -------------------------")
                                print("for grainindex=", grain_index)
                            if verbose > 1:
                                print("self.dict_grain_matrix[grain_index]",
                                self.dict_grain_matrix[grain_index])
                                print("self.refinedUBmatrix", self.refinedUBmatrix)

                            # WARNING: Update strain (if lower euler angles transform x have been applied on UBmatrix)
                            (self.deviatoricstrain,
                            self.deviatoricstrain_sampleframe,
                            self.new_latticeparameters) = CP.evaluate_strain_fromUBmat(self.refinedUBmatrix,
                                                                                        self.key_material,
                                                                                        constantlength='a', verbose=verbose)

                            # write .fit file of single grain spots results
                            #print('self.pixelresidues', self.pixelresidues)
                            #print('dirnameout_fitfile',dirnameout_fitfile)
                            self.writeFitFile(grain_index,
                                            corfilename=corfilename,
                                            dirname=dirnameout_fitfile,
                                            addpixdev=True,
                                            add_strain_sampleframe=True)

                            if fromIMM:
                                # because IMM may provides once a list of matrices
                                NeedtoProvideNewMatrices = False
                            else:
                                # because IAM provides a unique matrix
                                NeedtoProvideNewMatrices = True

                            # tagging and inhibiting exp spot which are too much ambiguous
                            MissingRef_grain_index = self.LabelMissingReflections(grain_index, 0.5, verbose=verbose)
                            self.MissingRefindexedgrains.append(MissingRef_grain_index)
                            #print('self.indexed_spots_dict',self.indexed_spots_dict)

                            self.dict_indexedgrains_material[grain_index] = key_material

                            indexgraincompleted = True
                            grain_index += 1
                            nbgrains_found += 1
                            # self.UBStack = None
                            # self.nbMatricesInUBStack = 0

                    # if this is not the last tolerance step, we do the other step to refine the spots to be considered to refine the model
                    else:
                        pass

    def updateIndexationDict(self, indexation_res, grain_index, overwrite=0, verbose=0):
        """
        update dictionary of experimental spots for the given family "grain_index"
        according to indexation info in "indexation_res"
        (unambiguous links between one exp. and one theo. spots)

        grain_index        : integer, index of grain corresponding to the pairs
        overwrite        : 1 ,prior to spots properties update, reset (i.e. unindex)
                            all spots that were considered (maybe wrongly) to
                            belong to family "grain_index"

        :return:
        number of new pairs (exp. theo.) found
        """
        if verbose:
            print('\n------- start of  updateIndexationDict()  -------\n')
            print('self.indexed_spots_dict[0]', self.indexed_spots_dict[0])
        if overwrite:
            self.resetSpotsFamily(grain_index)
        if verbose:
            print('self.indexed_spots_dict[0] after resetSpotsFamily', self.indexed_spots_dict[0])

        links_Miller = indexation_res[2]
        links_energy = indexation_res[5]

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

            c = -1 # col of flag is indexed
            # this exp spot has not been already indexed
            if self.indexed_spots_dict[exp_index][c] != 1:
                # keep the spot data and add theo info
                hh, kk, ll = miller_indices
                self.indexed_spots_dict[exp_index][-6:] = [hh, kk, ll, energy, grain_index, 1]
                #                linked_spots.append(exp_index)
                nb_updates += 1


        print("grain #%d : %d links to simulated spots have been found " % (grain_index, nb_updates))
        if verbose:
            print("\n so at the end of updateIndexationDict():     ")
            print('self.indexed_spots_dict[0]', self.indexed_spots_dict[0])

            print('\n------- end of  updateIndexationDict()  -------\n')
        return nb_updates

    def LabelMissingReflections(self, grain_index, angle_tol, verbose=0):
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

            hh, kk, ll = millers[k]
            energy = energies[k]

            for close_spot in close_spots:
                exp_index = int(close_spot[0])
                # keep the spot data and add theo info
                self.indexed_spots_dict[exp_index][-6:] = [hh, kk, ll, energy, MissingRef_grain_index, 1]

        if verbose:
            print("Experimental experimental spots indices which are not indexed",
                                                                            close_spots[:, 0])
            print("Missing reflections grainindex is %d for indexed grainindex %d"
                                                    % (MissingRef_grain_index, grain_index))
            print("within angular tolerance %.3f" % angle_tol)

        return MissingRef_grain_index

    def getCloseExpSpots(self, coords, angle_tol, verbose=0):
        r"""
        get index of exp. spots close to coords (2theta,chi) within angular tolerance

        input:
        coords        :2 values in iterable  : [0]  2theta, [1] chi
        angle_tol        : angular tolerance in degree

        return:
        array of [spot index, angle distance from coords spot in degree]

        """

        spot1 = np.array([[coords[0] / 2.0, coords[1]]])
        # exp data
        spots2_data = self.getSpotsExpData().T

        spots2 = np.array([spots2_data[:, 0] / 2.0, spots2_data[:, 1]]).T

        table_dist = GT.calculdist_from_thetachi(spot1, spots2)[:, 0]

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
        r"""
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

    def getSpotsLinks(self, UBOrientMatrix, exp_data=None, useabsoluteindex=None,
                                                            removeharmonics=1,
                                                            ResolutionAngstrom=False,
                                                            veryclose_angletol=1.0,
                                                            returnMissingReflections=True,
                                                            verbose=0):
        r"""
        return links (pairs or associations) between experimental and theoretical spots
        (i.e. simulated from a grain with UBOrientMatrix, key_material)

        exp_data   : None then use class attribute TwiceTheta_Chi_Int
                    otherwise should contain at least 3 elements
                    i.e. : twicetheta_data, chi_data, I_data, ...

        useabsoluteindex        : array of conversion from relative spot index to absolute one
                                useabsoluteindex[relative]= absolute
                                for instance When exp_data have been
                                extracted from a bigger data set.

                                if None: local index is absolute index in data

        ResolutionAngstrom  :   simulate exhaustively all the pattern for a perfect theo. crystal

        return:
         res, nb_of_simulated_spots, Missing_Reflections_Data
        """
        if verbose:
                print("\n--- start in getSpotsLinks() ---")
        # experimental data
        if exp_data is not None:
            twicetheta_data, chi_data, I_data = exp_data[:3]
        else:
            twicetheta_data, chi_data, I_data = self.TwiceTheta_Chi_Int

        if useabsoluteindex is None:
            useabsoluteindex = self.absolute_index

        #print("UBOrientMatrix in getSpotsLinks()", UBOrientMatrix)
        # simulated data
        grain = CP.Prepare_Grain(self.key_material, UBOrientMatrix, dictmaterials=self.dict_Materials)

        (Twicetheta, Chi,
        Miller_ind, posx, posy, Energy) = LAUE.SimulateLaue(grain,
                                                    self.emin,
                                                    self.emax,
                                                    self.detectorparameters,
                                                    removeharmonics=removeharmonics,
                                                    ResolutionAngstrom=ResolutionAngstrom,
                                                    pixelsize=self.pixelsize,
                                                    dim=self.dim,
                                                    detectordiameter=self.detectordiameter * 1.25)

        nb_of_simulated_spots = len(Twicetheta)

        # find close pairs between exp. and theo. spots
        res = matchingrate.SpotLinks(twicetheta_data,
                                        chi_data,
                                        I_data,  # experimental data
                                        veryclose_angletol,  # tolerance angle
                                        Twicetheta,
                                        Chi,
                                        Miller_ind,
                                        Energy,
                                        absoluteindex=useabsoluteindex)

        if res == 0 or len(res[1]) == 0:
            if not returnMissingReflections:
                return None, None
            return None, None, None

        if not returnMissingReflections:
            return res, nb_of_simulated_spots

        else:
            # Missing reflections:
            # ie theo (simulated) spots that have not been paired with an experimental spot

            # theo. pairs spot index
            tps = []
            for elem in res[1]:
                if not isinstance(elem[1], (list, np.ndarray)):
                    theoindex = elem[1]
                else:
                    theoindex = elem[1][0]
                tps.append(theoindex)

            # print('tps', tps)
            # print('nb_of_simulated_spots', list(range(nb_of_simulated_spots)))

            missing_refs = set(range(nb_of_simulated_spots))- set(tps)


            if verbose:
                #        print "theo paired spots indexed", tps
                print("nb of theo spots when trying to pairing spots",
                                                            nb_of_simulated_spots)

            nb_indexed_spots = nb_of_simulated_spots - len(missing_refs)
            if verbose:
                print("For angular tolerance %.2f deg" % veryclose_angletol)
                print("Nb of pairs found / nb total of expected spots: %d/%d"
                                        % (nb_indexed_spots, nb_of_simulated_spots))
                print("Matching Rate : %.2f"
                                % (100.0 * nb_indexed_spots / nb_of_simulated_spots))

            if verbose:
                print("Nb missing reflections: %d" % len(missing_refs))

            # Twicetheta, Chi, posx, posy, Miller_ind, Energy
            ar_data = np.array([Twicetheta, Chi, posx, posy])

            Missing_Reflections_Pos = np.take(ar_data, tuple(missing_refs), axis=1)
            Missing_Reflections_Miller = np.take(Miller_ind, tuple(missing_refs), axis=0)
            Missing_Reflections_Energy = Energy[list(missing_refs)]

            #        print "Missing_Reflections_Pos", Missing_Reflections_Pos
            #        print "Missing_Reflections_Miller", Missing_Reflections_Miller
            #        print "Missing_Reflections_Energy", Missing_Reflections_Energy

            Missing_Reflections_Data = (Missing_Reflections_Pos,
                                        Missing_Reflections_Miller,
                                        Missing_Reflections_Energy)

            if verbose:
                print("\n--- end of getSpotsLinks() ---")

            return res, nb_of_simulated_spots, Missing_Reflections_Data

    def getOrients_ImageMatching(
        self, MatchingRate_Threshold=None, exceptgrains=None, verbose=0):
        r"""
        get best orientation matrices with imagematching technique from experimental spot data

        spots are both unindexed and
                    previously indexed as belonging in grains of index in list exceptgrains

        return:
        [0]    : list of OrientMatrix object
        [1]    : corresponding (scalar) matching rate
        [2]    : nb of experimental spots used

        """
        (self.TwiceTheta_Chi_Int, self.absolute_index) = self.getSelectedExpSpotsData(
            exceptgrains=exceptgrains
        )

        nbspotsIMM = len(self.absolute_index)

        if MatchingRate_Threshold is None:
            MatchingRate_Threshold = 0.0

        bestEULER = IMM.bestorient_from_2thetachi(self.TwiceTheta_Chi_Int,
                                                    self.IMMdatabase,
                                                    dictparameters=self.dict_IMM)

        if bestEULER is None:
            return None, None, nbspotsIMM
        else:
            RAW_ANGULAR_MATCHINGTOLERANCE = 1.0
            print("bestEULER", bestEULER)
            bestEULER, bestmatchingrates = filterEulersList(bestEULER,
                                                        self.TwiceTheta_Chi_Int,
                                                        self.key_material,
                                                        self.emax,
                                                        rawAngularmatchingtolerance=RAW_ANGULAR_MATCHINGTOLERANCE,
                                                        MatchingRate_Threshold=MatchingRate_Threshold,
                                                        verbose=verbose,
                                                    )

        if bestEULER is None:
            return None, None, nbspotsIMM
        else:
            bestUB = []
            for eulers in bestEULER:
                bestUB.append(OrientMatrix(eulers=eulers))

            return bestUB, bestmatchingrates, nbspotsIMM

    def setTable_Angdist(self, nbmax=None):
        r"""
        set mutual angles between spots from current exp spot data

        set attibrute: table_angdist
        """
        theta, Chi = self.TwiceTheta_Chi_Int[0] / 2.0, self.TwiceTheta_Chi_Int[1]
        thechi_exp = np.array([theta, Chi]).T

        nbspots = len(theta)

        print("nbspots", nbspots)

        if nbmax is not None:
            nbmax = min(nbmax, nbspots)

        self.table_angdist = GT.calculdist_from_thetachi(thechi_exp, thechi_exp)[:nbmax, :nbmax]

        # tablelength = len(self.table_angdist)

        #        print "new table of distance has been computed"
        #        print "with %d spots" % tablelength

        return self.table_angdist

    def computeLUT(self, dictmaterials=DictLT.dict_Materials):
        """
        compute look_up_table angular distances between reciprocal nodes of a known structure

        Consider self.key_material, self.n_LUT
        """
        print("Compute LUT for indexing %s spots in LauePattern " % self.key_material)
        latticeparams = dictmaterials[self.key_material][1]
        applyExtinctionRules = dictmaterials[self.key_material][2]
        self.B_LUT = CP.calc_B_RR(latticeparams)
        isCubic = CP.hasCubicSymmetry(self.key_material, dictmaterials=dictmaterials)
        self.LUT = INDEX.build_AnglesLUT(self.B_LUT,
                                        self.n_LUT,
                                        MaxRadiusHKL=self.ResolutionAngstromLUT,
                                        cubicSymmetry=isCubic,
                                        applyExtinctionRules=applyExtinctionRules)

    def setAnglesLUTmatchingParameters(self, LUT=None, n_LUT=3, B_LUT=np.eye(3)):
        r"""
        set parameters for angles LUT matching technique

        if LUT is None then self.keymaterial and dictMAterials will be used
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

    def get_all_UBs_fromAnglesLUT(self,
                                spot_index_central=None,
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
        r"""
        get all orientation matrices from angular distance matching technique
        with experimental spot data

        .. note::
            USED in FileSeries

        exp. spots are those unindexed


        spot_index_central : integer or list of integer, 0 if None

        TODO: Not implemented!  All matrix are probed without stopping
        MatchingRate_Threshold   : in percent, matching rate above which loop is interrupted
                                    100   , never interrupted
        """
        Minimum_Nb_Matches = max(Minimum_Nb_Matches, MINIMUM_NB_MATCHES_FOR_INDEXING)

        # self.TwiceTheta_Chi_Int, self.absolute_index = self.getSelectedExpSpotsData(exceptgrains=exceptgrains)
        #   nbspotsIAM = len(self.absolute_index)

        if MatchingRate_Threshold is None:
            MatchingRate_Threshold = 100.0

        if spot_index_central is None:
            # default: use the first (and most intense) spot in data list
            spot_index_central = 0

        list_matrices, list_stats = self.FindOrientMatrices(spot_index_central=spot_index_central,
                                                    nbmax_probed=nbmax_probed,
                                                    nLUT=self.n_LUT,
                                                    LUT=LUT,
                                                    set_central_spots_hkl=set_central_spots_hkl,
                                                    ResolutionAngstrom=ResolutionAngstrom,
                                                    AngTol_LUTmatching=self.AngTol_LUTmatching,
                                                    MatchingRate_Angle_Tol=MatchingRate_Angle_Tol,
                                                    Minimum_Nb_Matches=Minimum_Nb_Matches,
                                                    nb_of_solutions_per_central_spot=max_nb_of_solutions_per_central_spot,
                                                    simulparameters=self.simulparameter)

        print("Nb of potential UBs ", len(list_matrices))

        if len(list_matrices) == 1:  # patch when only matrix found by FindOrientMatrices
            list_stats = [list_stats[0][:3]]
        #         print "list_stats", list_stats

        nbspotsIAM = len(self.absolute_index)
        if list_matrices == []:
            return None, None, nbspotsIAM, False

        MatchingRate_list = []
        for stat in list_stats:
            nbmatches, nbtheo, _ = stat
            MR = 100.0 * nbmatches / nbtheo
            MatchingRate_list.append(MR)

        MatchingRates = np.array(MatchingRate_list)

        return list_matrices, list_stats, nbspotsIAM, MatchingRates

    def get_bestUB_fromAnglesLUT(self, spot_index_central=None, MatchingRate_Threshold=None,
                                                                MatchingRate_Angle_Tol=0.5,
                                                                nbmax_probed=10,
                                                                Minimum_Nb_Matches=6,
                                                                LUT=None,
                                                                set_central_spots_hkl=None,
                                                                ResolutionAngstrom=False,
                                                                exceptgrains=None,
                                                                verbose=0):
        r"""
        get single best orientation matrix from angular distance matching technique
        with experimental spot data

        .. note::
            USED in FileSeries

        spots are unindexed

        spot_index_central : integer or list of integer, 0 if None

        TODO: still something Not implemented!  All matrix are probed without stopping if one seems to be correct
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

        (list_matrices, _, _, MatchingRates) = Res

        nbspotsIAM = len(self.absolute_index)

        best_result_index = np.argsort(MatchingRates)[::-1][0]

        matrix = list_matrices[best_result_index]
        score = MatchingRates[best_result_index]

        Threshold_reached = False

        bestUB = OrientMatrix(matrix=matrix)

        print('bestUB ', matrix)
        print('with matching rate',score)

        return bestUB, score, nbspotsIAM, Threshold_reached

    def getStatsOnMatching(self, List_of_Angles, exp_data, ang_tol=1.0, verbose=0):
        r"""
        returns matching rate in terms of nb of close pairs of spots (exp. and simul. ones)
        within angular tolerance

        :param List_of_Angles: list of 3 angles (EULER angles)
        :type List_of_Angles: list of 3 floats
        :param exp_data: experimental spots angles coordinates (kf vectors)
        :type exp_data: 2 arrays of floats (2theta and  chi)
        :param ang_tol: matching angular tolerance to form pairs
        :type ang_tol: float

        :return:
            [0] sorted indices of List_of_Angles elements by decreasing matching rate,
            [1] all matching rate corresponding to element in List_of_Angles
        """
        twicetheta_data, chi_data, _ = exp_data
        kk = 0
        allmatchingrate = []
        for angles_sol in List_of_Angles:

            test_Matrix = GT.fromEULERangles_toMatrix(angles_sol)

            res = matchingrate.Angular_residues(test_Matrix,
                                                twicetheta_data,
                                                chi_data,
                                                ang_tol=ang_tol,
                                                key_material=self.key_material,
                                                emin=self.emin,
                                                emax=self.emax)
            if res is None:
                continue

            matching_rate = 100.0 * res[2] / res[3]
            allmatchingrate.append(matching_rate)

            if verbose:
                print("*" * 30)
                print("res for k:%d and angles: [%.1f,%.1f,%.1f]"
                    % (kk, angles_sol[0], angles_sol[1], angles_sol[2]))
                print("Matching rate ----(in %%):                %.2f " % matching_rate)
                print("Nb of close spot pairs (< %.2f deg) : %d" % (ang_tol, res[2]))
                print("Nb of simulated spots : %d" % res[3])
                print("mean angular residues %.2f deg." % res[4])
                print("highest residues %.2f deg." % res[5])
            kk += 1

        return np.argsort(np.array(allmatchingrate))[::-1], allmatchingrate

    def refineUBSpotsFamily(self, grain_index, initial_matrix, use_weights=1,
                                                            nbSpotsToIndex="all",
                                                            verbose=0):
        r"""
        refine UB matrix of spots family belonging to grain being indexed

        input:
        grain_index:  integer , index of grain
        initial_matrix   :    UB matrix 3*3 array to be refined
        getstrain : always =1 , obsolete to be deleted
        use_weights : refine model parameters by weighting each spots pair (exp.- theo)
                        by intensity of exp. spot.
        nbSpotsToIndex  : integer or 'all' select the nb of pairs to used for refinement.

        :return: None, None if two few exp. data to fit

        # TODO:  change nbSpotsToIndex="all"

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

        data_1grain_raw = self.getSpotsFamilyallData(grain_index)

        # print('\n\n *******data_1grain_raw.shape in refineUBSpotsFamily', data_1grain_raw.shape)
        # print('data_1grain_raw[0]', data_1grain_raw)
        if isinstance(nbSpotsToIndex, int):
            data_1grain = data_1grain_raw[:nbSpotsToIndex]
        else:
            data_1grain = data_1grain_raw

        if verbose:
            print('data_1grain.shape in refineUBSpotsFamily', data_1grain.shape)

        if len(data_1grain) >= MINIMUM_LINKS_FOR_FIT:
            index, _, _, posX, posY, intensity = (data_1grain.T)[:6]
            H, K, L = (data_1grain.T)[-6: -6+3]
        else:
            print("Too few exp. data to fit in refineUBSpotsFamily()")
            return None, None

        Miller = np.array([H, K, L]).T

        #    print "Miller", Miller

        nb_pairs = len(index)
        if verbose:
            print("Nb of pairs: ", nb_pairs)

        sim_indices = np.arange(nb_pairs)

        if use_weights in (True, "True", 1, "true"):
            weights = intensity
        else:
            weights = None

        # fitting procedure for one or many parameters
        initial_values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])
        try:
            allparameters = np.array(
                self.detectorparameters.tolist() + [1, 1, 0, 0, 0] + [0, 0, 0])
        except:
            allparameters = np.array(
                self.detectorparameters + [1, 1, 0, 0, 0] + [0, 0, 0])

        arr_indexvaryingparameters = np.arange(5, 13)
        if verbose > 1:
            print("\nInitial error--------------------------------------\n")
            print("initial_matrix", initial_matrix)

        latticeparams = DictLT.dict_Materials[self.key_material][1]
        Bmatrix = CP.calc_B_RR(latticeparams)

        # print("initial_values, Miller,allparameters, arr_indexvaryingparameters,  etc...")
        # print(initial_values, Miller, allparameters, arr_indexvaryingparameters, sim_indices, posX, posY, initial_matrix,
        #     Bmatrix, 0, 1, self.pixelsize, self.dim, weights, 1, self.kf_direction)

        (residues,
        deltamat,
        newmatrix) = FitO.error_function_on_demand_strain(initial_values,
                                                                Miller,
                                                                allparameters,
                                                                arr_indexvaryingparameters,
                                                                sim_indices,
                                                                posX,
                                                                posY,
                                                                initrot=initial_matrix,
                                                                Bmat=Bmatrix,
                                                                pureRotation=0,
                                                                verbose=1,
                                                                pixelsize=self.pixelsize,
                                                                dim=self.dim,
                                                                weights=weights,
                                                                kf_direction=self.kf_direction)
        #print("Initial residues", residues)
        #print("---------------------------------------------------\n")

        results = FitO.fit_on_demand_strain(initial_values,
                                            Miller,
                                            allparameters,
                                            FitO.error_function_on_demand_strain,
                                            arr_indexvaryingparameters,
                                            sim_indices,
                                            posX,
                                            posY,
                                            initrot=initial_matrix,
                                            Bmat=Bmatrix,
                                            pixelsize=self.pixelsize,
                                            dim=self.dim,
                                            verbose=0,
                                            weights=weights,
                                            kf_direction=self.kf_direction)

        #    print "\n********************\n       Results of Fit        \n********************"
        #    print "results", results

        # print "\nFinal error--------------------------------------\n"
        (residues,
        deltamat,
        newmatrix) = FitO.error_function_on_demand_strain(results,
                                                        Miller,
                                                        allparameters,
                                                        arr_indexvaryingparameters,
                                                        sim_indices,
                                                        posX,
                                                        posY,
                                                        initrot=initial_matrix,
                                                        Bmat=Bmatrix,
                                                        pureRotation=0,
                                                        verbose=1,
                                                        pixelsize=self.pixelsize,
                                                        dim=self.dim,
                                                        weights=None,
                                                        kf_direction=self.kf_direction)

        # building B mat
        param_strain_sol = results
        starting_orientmatrix = initial_matrix

        varyingstrain = np.array([[1.0, param_strain_sol[2], param_strain_sol[3]],
                                    [0, param_strain_sol[0], param_strain_sol[4]],
                                    [0, 0, param_strain_sol[1]]])
        if verbose > 1:
            print("varyingstrain results")
            print(varyingstrain)

        # newUmat = np.dot(deltamat, starting_orientmatrix)

        # building UBmat(= newmatrix)
        newUBmat = np.dot(np.dot(deltamat, starting_orientmatrix), varyingstrain)
        if verbose > 1:
            print("newUBmat", newUBmat)


        Bstar_s = np.dot(newUBmat, Bmatrix)
        if verbose > 1:
            print("new UBs matrix in q= UBs G (s for strain)")
            print(Bstar_s)

        lattice_parameter_reciprocal = CP.matrix_to_rlat(Bstar_s)
        lattice_parameter_direct_strain = CP.dlat_to_rlat(lattice_parameter_reciprocal)

        Bmatrix_direct_strain = CP.calc_B_RR(lattice_parameter_direct_strain, directspace=0)
        Bmatrix_direct_unstrained = CP.calc_B_RR(latticeparams, directspace=0)

        Trans = np.dot(Bmatrix_direct_strain, np.linalg.inv(Bmatrix_direct_unstrained))
        strain_direct = (Trans + Trans.T) / 2.0 - np.eye(3)

        devstrain = strain_direct - np.trace(strain_direct) / 3.0 * np.eye(3)

        devstrain1, lattice_parameter_direct_strain1 = CP.DeviatoricStrain_LatticeParams(
                                                                        newUBmat,
                                                                        latticeparams,
                                                                        constantlength="a")
        if verbose:
            print("devstrain, lattice_parameter_direct_strain", devstrain,
                                                lattice_parameter_direct_strain)
            print("devstrain1, lattice_parameter_direct_strain1", devstrain1,
                                                                lattice_parameter_direct_strain1)

        (devstrain,
        deviatoricstrain_sampleframe,
        lattice_parameters) = CP.evaluate_strain_fromUBmat(newUBmat,
                                                        self.key_material,
                                                        constantlength="a", verbose=verbose)

        self.refinedUBmatrix = newUBmat
        self.B0matrix = Bmatrix
        self.deviatoricstrain = devstrain
        self.deviatoricstrain_sampleframe = deviatoricstrain_sampleframe
        self.new_latticeparameters = lattice_parameters

        self.pixelresidues = residues
        self.spotindexabs = index

        if verbose:
            print("strain_direct", strain_direct)

        print("UB and strain refinement completed")

        return newmatrix, devstrain


    def refineStrainElementsSpotsFamily(self, grain_index, initial_matrix,
                                                            use_weights=1, verbose=0):
        r"""
        refine UB matrix and strain elements of exp. spots of family 'grain_index'

        :param grain_index: grain index (int)
        :param initial_matrix: guessed UB matrix (rotation*little strain)
        :param use_weights: use spot intensity to weight the pair (theo , exp) distance

        set results values as following:
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

        data_1grain = self.getSpotsFamilyallData(grain_index)

        #        print "absolute index of spots to refine", data_1grain[:, 0]
        #    print "data_1grain.shape", data_1grain.shape
        if len(data_1grain) >= MINIMUM_LINKS_FOR_FIT:
            # index, _, _, posX, posY, intensity, H, K, L, _ = data_1grain.T
            index, _, _, posX, posY, intensity = (data_1grain.T)[:6]
            H, K, L = (data_1grain.T)[-6: -6+3]
        else:
            print("Too few exp. data to fit in refineStrainElementsSpotsFamily()")
            return None, None

        Miller = np.array([H, K, L]).T

        #    print "Miller", Miller

        nb_pairs = len(index)
        if verbose:
            print("Nb of pairs: ", nb_pairs)

        # sim_indices = np.arange(nb_pairs)

        if use_weights in (True, "True", 1, "true"):
            weights = intensity
        else:
            weights = None

        # needs pixX, pixY, hkls, starting_orientmatrix, key_material, weights
        # see fit_transform_parameters(self, pixX, pixY, hkls,
        #               starting_orientmatrix, key_material, weights)
        # in plotrefineGUI.py
        starting_orientmatrix = initial_matrix
        pixX, pixY = posX, posY
        hkls = Miller

        # triangular up transform in sample frame
        TransformParameters = [1, 0, 0, 1, 0, 1.0]
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

        List_of_keys = ["anglex", "angley", "anglez", "Ts00", "Ts01", "Ts02", "Ts11", "Ts12", "Ts22"]

        print("List_of_defaultvalues", List_of_defaultvalues)

        fitting_parameters_keys = []
        fitting_parameters_values = []
        for k, flag in enumerate(List_of_checkwidgets):

            if flag:
                #                 print 'k,checkwidget', k, flag
                fitting_parameters_keys.append(List_of_keys[k])
                fitting_parameters_values.append(List_of_defaultvalues[k + 5])

        pureUmatrix, _ = GT.UBdecomposition_RRPP(starting_orientmatrix)

        print("fitting_parameters_values, fitting_parameters_keys")
        print(fitting_parameters_values, fitting_parameters_keys)
        print("len(allparameters)", len(allparameters))
        print("starting_orientmatrix", starting_orientmatrix)

        absolutespotsindices = np.arange(len(pixX))

        print("initial errors--------------------------------------\n")
        print(FitO.error_function_strain(fitting_parameters_values,
                                    fitting_parameters_keys,
                                    hkls,
                                    allparameters,
                                    absolutespotsindices,
                                    pixX,
                                    pixY,
                                    initrot=pureUmatrix,
                                    B0matrix=B0matrix,
                                    pureRotation=0,
                                    verbose=0,
                                    pixelsize=self.pixelsize,
                                    dim=self.dim,
                                    weights=None,
                                    kf_direction=self.kf_direction,
                                    returnalldata=True))

        print("\n\n*** refineStrainElementsSpotsFamily (File Series/Index_Refine.py) *****\n")
        print("fitting_parameters_values", fitting_parameters_values)
        print("fitting_parameters_keys", fitting_parameters_keys)
        print("hkls", hkls)
        print("allparameters", allparameters)
        print("absolutespotsindices", absolutespotsindices)
        print("pixX", pixX)
        print("pixY", pixY)
        print("UBmatrix_start", pureUmatrix)
        print("B0matrix", B0matrix)
        print("self.pixelsize", self.pixelsize)
        print("self.dim", self.dim)
        print("weights", weights)

        print("fitting strain parameters")
        results = FitO.fit_function_strain(fitting_parameters_values,
                                            fitting_parameters_keys,
                                            hkls,
                                            allparameters,
                                            absolutespotsindices,
                                            pixX,
                                            pixY,
                                            UBmatrix_start=pureUmatrix,
                                            B0matrix=B0matrix,
                                            nb_grains=1,
                                            pureRotation=0,
                                            verbose=0,
                                            pixelsize=self.pixelsize,
                                            dim=self.dim,
                                            weights=weights,
                                            kf_direction=self.kf_direction)

        print("results -------", results)
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

        print("\nFinal errors--------------------------------------\n")
        # alldistances_array, Uxyz, newmatrix, Ts, T
        (residues, Uxyz, newUmat,
        refinedTs, refinedT) = FitO.error_function_strain(results,
                                                        fitting_parameters_keys,
                                                        hkls,
                                                        allparameters,
                                                        absolutespotsindices,
                                                        pixX,
                                                        pixY,
                                                        initrot=pureUmatrix,
                                                        B0matrix=B0matrix,
                                                        pureRotation=0,
                                                        verbose=0,
                                                        pixelsize=self.pixelsize,
                                                        dim=self.dim,
                                                        weights=weights,
                                                        kf_direction=self.kf_direction,
                                                        returnalldata=True)

        self.residues_non_weighted = FitO.error_function_strain(results,
                                                                fitting_parameters_keys,
                                                                hkls,
                                                                allparameters,
                                                                absolutespotsindices,
                                                                pixX,
                                                                pixY,
                                                                initrot=pureUmatrix,
                                                                B0matrix=B0matrix,
                                                                pureRotation=0,
                                                                verbose=0,
                                                                pixelsize=self.pixelsize,
                                                                dim=self.dim,
                                                                weights=None,
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
        print("q = newUB B0init G*")
        print("in q = newUmat varyingstrain B0init G*")
        print("Uxyz, Umat_init, newUmat, refinedT,refinedTs")
        print(Uxyz, Umat_init, newUmat, refinedT, refinedTs)
        print("newUBmat, B0matrix_init")
        print(newUBmat, B0matrix)

        # varyingstrain = np.dot(np.linalg.inv(newUmat), newUBmat)

        Tsresults = ("triangular", "sampleframe", refinedTs)

        # strain evaluation from refined matrix operator
        Bstar_s = np.dot(newUBmat, B0matrix)
        if verbose:
            print("new UBs matrix in q= UBs G (s for strain)")
            print(Bstar_s)

        lattice_parameter_reciprocal = CP.matrix_to_rlat(Bstar_s)
        lattice_parameter_direct_strain = CP.dlat_to_rlat(lattice_parameter_reciprocal)

        Bmatrix_direct_strain = CP.calc_B_RR(lattice_parameter_direct_strain, directspace=0)
        Bmatrix_direct_unstrained = CP.calc_B_RR(latticeparameters, directspace=0)

        Trans = np.dot(Bmatrix_direct_strain, np.linalg.inv(Bmatrix_direct_unstrained))
        strain_direct = (Trans + Trans.T) / 2.0 - np.eye(3)

        devstrain = strain_direct - np.trace(strain_direct) / 3.0 * np.eye(3)

        # devstrain_round = np.round(devstrain * 1000, decimals=2)
        # refinedUBmatrix = newUBmat
        # deviatoricstrain = devstrain
        # pixelresidues = residues
        # spotindexabs = index

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
        r"""
        write .res file
        containing all the exp. spots with their properties
        (indexed (grain index) or not (-1)), hkl energy etc.

        :param corfilename: .cor filename to deduce output filename
        """

        if corfilename is not None:
            outputfilename = corfilename.split(".")[0] + ".res"
        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)

        print("Saving Summary file: %s" % outputfilename)

        Data = self.getSummaryallData()
        if len(Data) == 0:
            return
        if Data.shape[0] == 1:
            nbspots = 1
        else:
            nbspots = len(Data)

        nb_props = len(self.dict_props_name)
        nb_add_props = nb_props - 6 - 6  # 6 main ones on the left and 6 on the right for indexing purpose
        # print('nb_add_props',nb_add_props)
        # print('self.dict_props_name',self.dict_props_name)
        if nb_add_props == 0:
            datatooutput = np.take(Data, (0, -2, 1, 2, 3, 4, 5, -6, -5, -4, -3), axis=1)
        else:
            selcol = tuple([0, -2, 1, 2, 3, 4, 5] + list(range(6, 6 + nb_add_props))+[-6, -5, -4, -3])
            datatooutput = np.take(Data, selcol, axis=1)

        datatooutput = np.round(datatooutput, decimals=7)

        header = "# Spots Summary of: %s\n" % (self.filename)
        header += "# File created at %s with indexingSpotsSet.py\n" % (time.asctime())
        header += "# Number of spots: %d\n" % nbspots

        header += ("#spot_index grain_index 2theta Chi Xexp Yexp")

        if nb_add_props > 0:
            header_addprops = ''
            invdict = dict([(val, key) for key, val in self.dict_props_name.items()])
            # print('invdict', invdict)
            # start at 6 to 12 included for 7 added spot properties
            for keycol in range(6, 6 + nb_add_props):
                header_addprops += ' %s' % invdict[keycol]
            # print('header_addprops',header_addprops)
            header = header+header_addprops

        header = header + " intensity h k l Energy\n"

        outputfile = open(outputfilename, "w")

        outputfile.write(header)

        np.savetxt(outputfile, datatooutput, fmt="%.7f")

        for grain_index in self.indexedgrains:
            nbspotsindexed, MatchRate = self.dict_grain_matching_rate[grain_index][:2]
            UBmatrix = self.dict_grain_matrix[grain_index]
            key_material = self.dict_indexedgrains_material[grain_index]
            latticeparams = DictLT.dict_Materials[key_material][1]
            B0matrix = CP.calc_B_RR(latticeparams)

            outputfile.write("#grainIndex\n")
            outputfile.write("G_%d\n" % grain_index)

            outputfile.write("#Element\n")
            outputfile.write("%s\n" % key_material)

            outputfile.write("#MatchingRate\n")
            outputfile.write("%.1f\n" % MatchRate)

            outputfile.write("#Nb indexed Spots\n")
            outputfile.write("%.d\n" % nbspotsindexed)

            outputfile.write("#UB matrix in q= (UB) B0 G*\n")
            outputfile.write(str(UBmatrix) + "\n")

            outputfile.write("#B0 matrix (starting unit cell) in q= UB (B0) G*\n")
            outputfile.write(str(B0matrix) + "\n")

            outputfile.write("#deviatoric strain (10-3 unit)\n")
            outputfile.write(str(self.dict_grain_devstrain[grain_index] * 1000.0) + "\n")

            outputfile.write("#deviatoric strain sample frame (10-3 unit)\n")
            outputfile.write(str(self.dict_grain_devstrain_sample[grain_index] * 1000.0) + "\n")

        outputfile.close()

    def writeFitFile(self, grain_index, corfilename=None, dirname=None,
                                            addpixdev=False,
                                            verbose=0,
                                            add_strain_sampleframe=False,
                                            add_grainindex_in_outputfilename=True):
        r"""
        write a .fit file: = .cor file + h,k,l, energy grainindex...
        list of spots belonging to a single grain

        :param corfilename: filename to deduce the output filename

        :param add_grainindex_in_outputfilename: add in filename '_g' and grainindex

        .. note:: output filename has additional info  '_g(grain_index).fit' for .fit
        with only data of grain number grain index
        (e.g. _g2.fit for the third indexed grain). Grain_index starts from zero!)
        """
        # set output file name
        if corfilename is not None:

            if add_grainindex_in_outputfilename:
                outputfilename = (corfilename.split(".")[0] + "_g%d" % grain_index + ".fit")
            else:
                outputfilename = corfilename.split(".")[0] + ".fit"

        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)
        # get spots data
        dataspots = self.getSpotsFamilyallData(grain_index)
        nbindexedspots = len(dataspots)
        if nbindexedspots > 1:
            # (index, tth, chi, posX, posY, intensity, H, K, L, Energy) = dataspots.T
            index, tth, chi, posX, posY, intensity = (dataspots.T)[:6]
            H, K, L, Energy = (dataspots.T)[-6:-6+4]
        elif nbindexedspots == 1:
            index, tth, chi, posX, posY, intensity = dataspots[:6]
            H, K, L, Energy = dataspots[-6:-6+4]
        else:
            print("data spots are empty... Nothing to write in .fit file")
            return

        if add_strain_sampleframe:

            deviatoricstrain_sampleframe = self.deviatoricstrain_sampleframe

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

            # --------------------------
            # since index list contains less elements than last results of refinement
            # index list comes from getSpotsFamilyallData
            # (which relies on the last state of assignation)

            # Then, keep then only element from index that have been refined
            pos = []
            for ind in index:
                pos.append(np.where(abs_spotindex == ind)[0][0])

            # take a part of pixeldevs
            pixeldevarray = pixeldevs[pos]
            # -----------------------------

            nbindexedspots = len(index)
            # grain_index * np.ones(nbindexedspots)

            Columns = [index, intensity, H, K, L,
                        tth, chi, posX, posY, Energy,
                        grain_index * np.ones(nbindexedspots),
                        pixeldevarray]
        else:
            Columns = [index, intensity, H, K, L, tth, chi, posX, posY, Energy]
            nbindexedspots = len(index)

        if verbose:
            print("Columns", Columns)

        datatooutput = np.array(Columns).T
        datatooutput = np.round(datatooutput, decimals=7)

        try:
            dict_matrices = {}
            dict_matrices["Element"] = self.key_material
            dict_matrices["grainIndex"] = grain_index

            dict_matrices["UBmat"] = self.dict_grain_matrix[grain_index]
            dict_matrices["B0"] = self.B0matrix
            #         dict_matrices['UBB0'] = self.UBB0mat
            UBB0 = np.dot(dict_matrices["UBmat"], dict_matrices["B0"])
            euler_angles = ORI.calc_Euler_angles(UBB0).round(decimals=3)
            dict_matrices["euler_angles"] = euler_angles

            dict_matrices["devstrain_crystal"] = self.deviatoricstrain
            dict_matrices["detectorparameters"] = self.detectorparameters
            dict_matrices["pixelsize"] = self.pixelsize
            dict_matrices["framedim"] = self.dim
            if add_strain_sampleframe:
                dict_matrices["devstrain_sample"] = deviatoricstrain_sampleframe

            dict_matrices["LatticeParameters"] = self.new_latticeparameters

            dict_matrices["Ts"] = self.refinedTs

            if self.CCDLabel is not None:
                dict_matrices["CCDLabel"] = self.CCDLabel
        except AttributeError:
            print("Missing attributes. Refinement has worked very well: few spots ?")

        meanresidues = None
        if addpixdev:
            meanresidues = np.mean(pixeldevs)
            columnsname = "#spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev\n"
        else:
            columnsname = "#spot_index intensity h k l 2theta Chi Xexp Yexp Energy\n"

        #currentfolder = os.path.abspath(os.curdir)
        IOLT.writefitfile(outputfilename, datatooutput, nbindexedspots,dict_matrices=dict_matrices, meanresidues=meanresidues,
                        PeakListFilename=self.filename,
                        columnsname=columnsname,
                        modulecaller="indexingSpotsSet.py")
        print("Fit File  written in %s" % outputfilename)

    def writecorFile_unindexedSpots(self, corfilename=None, dirname=None, filename_nbdigits=None,
                                        verbose=0):
        r"""
        write a .cor file of spots that are still not indexed.
        Output filename will contain '_unindexed'

        :param corfilename: full .cor filename to deduce the output filename
        .. todo:: use better IOLaueTools.writefile_cor

        .. note:: called by LaueToolsGUI. .OnSaveIndexationResultsfile
        """

        # should insert _unindexed_ before digits in output filename
        if corfilename is not None:

            posdigit, nbdigitsfound = GT.findfirstnumberpos(corfilename)
            #             print "posdigit, nbdigitsfound", posdigit, nbdigitsfound

            if filename_nbdigits is not None:
                if nbdigitsfound > filename_nbdigits:
                    deltadigits = nbdigitsfound - filename_nbdigits
                    posdigit += deltadigits

            if nbdigitsfound == 0:
                str_to_add = "_unindexed"
            elif corfilename[:posdigit][-1] in ("_",):
                str_to_add = "unindexed_"
            else:
                str_to_add = "_unindexed_"

            if corfilename.endswith(".cor"):
                outputfilename = (corfilename[:posdigit] + str_to_add + corfilename[posdigit:])
            else:
                outputfilename = (corfilename[:posdigit] + str_to_add + corfilename[posdigit:]
                    + ".cor")

        if dirname is not None:
            outputfilename = os.path.join(dirname, outputfilename)


        r_alldataunindexed = self.getUnIndexedSpotsallData(exceptgrains=self.indexedgrains)
        nbunindexedspots = r_alldataunindexed.shape[0]

        if nbunindexedspots == 0:
            # nothing saved
            return
        elif nbunindexedspots == 1:
            #TODO check !!
            # assert 1==0
            print('alldataunindexed', r_alldataunindexed)
            alldataunindexed = r_alldataunindexed[0]
            pass
        else:
            alldataunindexed = r_alldataunindexed
        nb_props = len(self.dict_props_name)
        nb_add_props = nb_props - 6 - 6  # 6 main ones on the left and 6 on the right for indexing purpose
        # print('nb_add_props',nb_add_props)
        # print('self.dict_props_name',self.dict_props_name)
        if nb_add_props == 0:
            datatooutput = np.take(alldataunindexed, (0, 5, 1, 2, 3, 4), axis=1)
        else:
            selcol = tuple([0, 5, 1, 2, 3, 4]+ list(range(6, 6 + nb_add_props)))
            datatooutput = np.take(alldataunindexed, selcol, axis=1)

        datatooutput = np.round(datatooutput, decimals=7)
        if verbose:
            print('datatooutput.shape', datatooutput.shape)

        # TODO: add pixdev mean and all corresponding pixdev
        header = "# Unindexed and unrefined Spots of: %s\n" % (self.filename)
        header += "# File created at %s with indexingSpotsSet.py\n" % (time.asctime())
        header += "# Number of unindexed spots: %d\n" % nbunindexedspots

        header += "#Element\n"
        header += "None" + "\n"
        header += "#grainIndex\n"
        header += "None" + "\n"

        header += "#spot_index intensity 2theta Chi Xexp Yexp\n"
        if nb_add_props > 0:
            header_addprops = ''
            invdict = dict([(val, key) for key, val in self.dict_props_name.items()])
            # start at 6 to 12 included for 7 added spot properties
            for keycol in range(6, 6 + nb_add_props):
                header_addprops += ' %s' % invdict[keycol]
            # print('header_addprops', header_addprops)
            header = header[:-1]+header_addprops+'\n'

        outputfile = open(outputfilename, "w")

        if verbose:
            print("Saving unindexed  fit file: %s" % outputfilename)
        outputfile.write(header)
        np.savetxt(outputfile, datatooutput, fmt="%.6f")

        rectpix = 0.0
        param = self.detectorparameters + [self.pixelsize]
        if param is not None:
            outputfile.write("# Calibration parameters\n")
            if len(param) == 6:
                for par, value in zip(
                    ["dd", "xcen", "ycen", "xbet", "xgam", "pixelsize"], param):
                    outputfile.write("# %s     :   %s\n" % (par, value))
                ypixelsize = param[5] * (1.0 + rectpix)
                outputfile.write("# ypixelsize     :   %s\n" % str(ypixelsize))
            elif len(param) == 5:
                for par, value in zip(["dd", "xcen", "ycen", "xbet", "xgam"], param):
                    outputfile.write("# %s     :   %s\n" % (par, value))
            else:
                raise ValueError("5 or 6 calibration parameters are needed!")

        # add CCDlabel:
        outputfile.write("# CCDLabel    :  %s"%self.CCDLabel)

        outputfile.close()

    def merge_fitfiles(self, nbgrains, corfilename=None, dirname=None, removefiles=0,
                                    add_unindexed_spotslist=True, verbose=0):
        r"""
        merge into a single .fit file all .fit files corresponding to each grain

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
        if nbgrains <= 0:
            return

        prefix = corfilename.split(".")[0]
        mergedfitfilename = prefix + ".fit"

        if verbose:
            print("mergedfitfilename", mergedfitfilename)

        if dirname is not None:
            mergedfitfilename = os.path.join(dirname, mergedfitfilename)
            prefix = os.path.join(dirname, prefix)

        prefixgrain = prefix + "_g"

        outputfile = open(mergedfitfilename, "w")
        for grainindex in list(range(nbgrains)):

            f = open(prefixgrain + "%d" % grainindex + ".fit", "r")
            try:
                for line in f:
                    outputfile.write(line)
            finally:
                f.close()
            outputfile.write("\n")

            if removefiles:
                os.remove(prefixgrain + "%d" % grainindex + ".fit")

        if add_unindexed_spotslist:
            posdigit, _ = GT.findfirstnumberpos(prefix)

            if prefix[:posdigit][-1] in ("_",):
                str_to_add = "unindexed_"
            else:
                str_to_add = "_unindexed_"

            filename_unindexedspots = (prefix[:posdigit] + str_to_add + prefix[:posdigit] + ".cor")

            if os.path.isfile(filename_unindexedspots):
                f = open(filename_unindexedspots, "r")
                try:
                    for line in f:
                        outputfile.write(line)
                finally:
                    f.close()

            outputfile.write("\n")

        outputfile.close()

    def plotgrains_from_indexspots(self, unindexedspots=None):
        r""" plot grains spots and data from indexed spots dictionary

        :param unindexedspots: True plot unindexed spots
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
                r_alldataunindexed = self.getUnIndexedSpotsallData(exceptgrains=unindexedspots)
                nbunindexedspots = r_alldataunindexed.shape[0]
                if nbunindexedspots == 0:
                    th_unind, Chi_unind = [], []
                elif nbunindexedspots == 1:
                    #TODO check !!
                    th_unind, Chi_unind = r_alldataunindexed[0][0][1:3]
                else:
                    th_unind, Chi_unind = np.take(r_alldataunindexed, (1, 2), axis=1).T

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

        # TODO: confusion ax and p ? see plotgrains()
        dicocolor = {0: "k", 1: "r", 2: "g", 3: "b", 4: "c", 5: "m"}
        nbcolors = len(dicocolor)
        # theo spots
        for i_grain in list(range(nb_of_orientations)):
            ax = fig.add_subplot(codefigure)
            # all exp spots
            scatter(all_tthchi[:, 0], all_tthchi[:, 1],
                                        s=40, c="w", marker="o", faceted=True, alpha=0.5)

            if unindexedspots is not None:
                # unindexed spots
                scatter(th_unind, Chi_unind, s=20, c="b", faceted=False, alpha=0.5)

            # simul spots
            tthchi = np.array(twicethetaChi[i_grain])
            ax.scatter(tthchi[:, 0], tthchi[:, 1],
                                c=dicocolor[(i_grain + 1) % nbcolors], faceted=False)
            if index_fig < nb_of_orientations:
                index_fig += 1
                codefigure += 1

        show()

    def plotgrains_alldata(self, titlefig="", unindexedspots=None, verbose=0):
        r""" plot all Laue patterns of all grains
        """
        self.plotgrains(exp_data=self.getSpotsExpData()[:2],
                        titlefig=titlefig,
                        unindexedspots=unindexedspots,
                        verbose=verbose)

    def plotgrains(self, exp_data=None, titlefig="", unindexedspots=None, verbose=0):
        r"""
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

            grain = [np.eye(3), EXTINCTION, self.dict_grain_matrix[key_matrix], self.key_material]

            if verbose:
                print("matrix for key_matrix%d" % key_matrix)
                print(self.dict_grain_matrix[key_matrix].tolist())

            (Twicetheta, Chi, _, _, _, _) = LAUE.SimulateLaue(grain,
                                                            self.emin,
                                                            self.emax,
                                                            self.detectorparameters,
                                                            detectordiameter=self.detectordiameter)

            all_tthchi.append([Twicetheta, Chi])

            nb_matrices += 1
            if nb_matrices == 9:
                print("Warning: limiting to 9 plots")
                break

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

        dicocolor = {0: "k", 1: "r", 2: "g", 3: "b", 4: "c", 5: "m"}
        nbcolors = len(dicocolor)

        if exp_data is not None:
            tth_exp, Chi_exp = exp_data[:2]

        if unindexedspots is not None:
            r_alldataunindexed = self.getUnIndexedSpotsallData(exceptgrains=unindexedspots)
            nbunindexedspots = r_alldataunindexed.shape[0]
            if nbunindexedspots == 0:
                th_unind, Chi_unind = [], []
            elif nbunindexedspots == 1:
                #TODO check !!
                th_unind, Chi_unind = r_alldataunindexed[0][0][1:3]
            else:
                th_unind, Chi_unind = np.take(r_alldataunindexed, (1, 2), axis=1).T
        # theo spots
        for i_mat in list(range(min(nbMatrices, 9))):

            subplot(codefigure)
            if i_mat == 0:
                title("%s" % titlefig)

            if exp_data is not None:
                # all exp spots
                scatter(
                    tth_exp, Chi_exp, s=40, c="w", marker="o", faceted=True, alpha=0.5)

            if unindexedspots is not None:
                # unindexed spots
                scatter(th_unind, Chi_unind, s=20, c="b", faceted=False, alpha=0.5)

            scatter(
                all_tthchi[i_mat][0],
                all_tthchi[i_mat][1],
                c=dicocolor[(i_mat + 1) % nbcolors],
                faceted=False)

            codefigure += 1

        show()

    def plotallgrains(self, unindexedspots=None):
        """ single plot of all spots and grains origin and data from indexed spots dictionary

        unindexedspots    : plot unindexed spots + previously indexed spots not in unindexedspots
        OBSOLETE to be checked
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
            alldataunindexed = self.getUnIndexedSpotsallData(exceptgrains=unindexedspots)[:, 1:3].T
            nbunindexedspots = alldataunindexed.shape[0]
            if nbunindexedspots == 0:
                # nothing saved
                return
            elif nbunindexedspots == 1:
                #TODO check !!
                assert 1 == 0
            else:
                th_unind, Chi_unind = alldataunindexed

        all_tthchi = self.getSpotsallData()[:, 1:3]
        nb_of_orientations = grain_index

        dicocolor = {0: "k", 1: "r", 2: "g", 3: "b", 4: "c", 5: "m"}
        nbcolors = len(dicocolor)

        if unindexedspots is not None:
            # unindexed spots
            scatter(th_unind, Chi_unind, s=20, c="b", faceted=False, alpha=0.5)

        # all exp spots
        scatter(all_tthchi[:, 0], all_tthchi[:, 1], s=40, c="w", marker="o", faceted=True, alpha=0.5)

        # theo spots
        for i_grain in list(range(nb_of_orientations)):
            tthchi = np.array(twicethetaChi[i_grain])
            #        print tthchi
            scatter(tthchi[:, 0],
                tthchi[:, 1],
                c=dicocolor[(i_grain + 1) % nbcolors],
                faceted=False)

        show()


# --- -------  PROCEDURES -------------------
def writeEmptyFileSummary(filenamepattern, dirname=None):
    r"""
    write .res file in case of no results (no peaks to refine ...)
    """
    outputfilename = filenamepattern.split(".")[0] + ".res"
    if dirname is not None:
        outputfilename = os.path.join(dirname, outputfilename)

    print("Saving EMPTY Summary file: %s" % outputfilename)

    # (spotindex, grain_index, tth, chi, posX, posY, intensity, H, K, L, Energy) = Data.T

    # Columns = [spotindex, grain_index, tth, chi, posX, posY, intensity, H, K, L, Energy]

    nbspots = 0

    header = "# Spots Summary of missing or empty: %s\n" % filenamepattern
    header += "# File created at %s with indexingSpotsSet.py\n" % (time.asctime())
    #         header += '# Number of indexed spots: %d\n' % nbindexedspots
    #         header += '# Number of unindexed spots: %d\n' % nbunindexedspots
    header += "# Number of spots: %d\n" % nbspots

    header += ("#spot_index grain_index 2theta Chi Xexp Yexp intensity h k l Energy\n")
    outputfile = open(outputfilename, "w")
    outputfile.write(header)
    outputfile.close()

def purgeSpotsinDict(exp_data, twicethetaChi_to_remove, dist_tolerance=0.2):
    """
    remove undesirable spots in exp_data that listed in
    and initialize spots indexation dictionary from exp_data

    twicethetaChi_to_remove tuple of 2 elements: 2theta,chi
    dist_tolerance = angular tolerance in a approximate 2theta chi cartesian space

    :return: indexed_spots_dict, dict_props_name

    .. note:: may be used to remove substrate uninteresting peaks from substrate

    .. warning:: handling additional spots properties is not implemented... (add_props=None)
    """
    tth, chi, posX, posY, Intensity = exp_data

    Twicetheta, Chi, tokeep = GT.removeClosePoints_two_sets((tth, chi), twicethetaChi_to_remove,
                                                                dist_tolerance=dist_tolerance)

    posx = np.take(posX, tokeep)
    posy = np.take(posY, tokeep)
    dataintensity = np.take(Intensity, tokeep)

    purged_exp_data = Twicetheta, Chi, dataintensity, posx, posy

    return initIndexationDict(purged_exp_data, add_props=None)


def initIndexationDict(exp_data, add_props=None):
    """
    initialize spots indexation dictionary from exp_data

    :param exp_data: tuple of spots properties. It must have at least 5 elements (the main ones)
    tth, chi, Intensity, posX, posY

    :param add_props: tuple of spots properties: (data, colnames) data is a list of properties values
    . nb of values = nb of spots

    :return: indexed_spots_dict, dict_props_name
    """
    nb_of_spots = len(exp_data[0])

    # val is index in spot_props
    dict_props_name = {'spotindex': 0,
                        'twicetheta': 1,
                        'chi': 2,
                        'X': 3,
                        'Y': 4,
                        'intensity': 5,
                        'H': -6,
                        'K': -5,
                        'L': -4,
                        'energy': -3,
                        'grainindex': -2,
                        'isindexed': -1}

    tth, chi, Intensity, posX, posY = exp_data[:5]
    if add_props is not None:
        data, colnames = add_props
        data_props = np.array(data).T
        for k, colname in enumerate(colnames):
            dict_props_name[colname] = 6 + k

    indexed_spots_dict = {}
    # dictionary of exp spots
    for k in range(nb_of_spots):
        spot_props = [k,  # index of experimental spot in .cor file
                        tth[k],
                        chi[k],  # 2theta, chi coordinates
                        posX[k],
                        posY[k],  # pixel coordinates
                        Intensity[k]]  # intensity

        if add_props is not None:
            spot_props = spot_props + data_props[k].tolist()

        spot_props = spot_props +[10000, 10000, 10000, # H, K,L
                                    10000, # energy
                                    -1,  # grainindex   -1 not indexed yet
                                    0]  # isindexed  0 means not indexed yet

        indexed_spots_dict[k] = spot_props

    return indexed_spots_dict, dict_props_name


# --- -----------------  ORIENT MATRIX FILTERING
def filterEulersList(bestEULER, tth_chi_int, key_material, emax,
                                            rawAngularmatchingtolerance=1.0,
                                            MatchingRate_Threshold=50.0,
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
    sortedindices, matchingrates = matchingrate.getStatsOnMatching(bestEULER,
                                                                    tth_chi_int[0],
                                                                    tth_chi_int[1],
                                                                    key_material,
                                                                    ang_tol=rawAngularmatchingtolerance,
                                                                    emax=emax,
                                                                    verbose=verbose,
                                                                )

    bestEULER = np.take(bestEULER, sortedindices, axis=0)
    bestmatchingrates = np.take(np.array(matchingrates), sortedindices)

    if verbose:
        print("\n --------- Image Matching Results -----------------\n")
        print("\nnb of potential grains %d" % len(bestEULER))
        print("bestmatchingrates")
        print(bestmatchingrates)

    bestEULER, bestmatchingrates = filterEquivalentMatrix(bestEULER, bestmatchingrates)

    bestEULER, bestmatchingrates = filterMatrix_MinimumRate(bestEULER, bestmatchingrates,
                                        MatchingRate_Threshold)

    if verbose:
        print("After filtering (cubic permutation, matching threshold %.2f)"
            % MatchingRate_Threshold)
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
    mats_out, pos_mat = RemoveDuplicatesOrientationMatrix(np.array(mats), scores, tol=0.1,
                                        allpermu=None, OutputMatricesOnly=0)

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
    return True if A==B   i.e. if A =B*S with S symetry operator (axes permutation and rotation)
    matrix comparaison is done elementwise within tolerance

    matA and matB are single matrix
    """
    if allpermu is 'Id':
        diff = matB - matA
        flagdiff = np.less(np.abs(diff), tol)
        resflag = np.all(np.ravel(flagdiff))
        # print('for ID: ', resflag)
        return resflag, resflag

    elif allpermu is None or allpermu == "cubic":
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

def transform_matrix_similar(Aref, B, tol=0.001):
    """ transform B to a matrix similar to Aref through a permutation
    (with sign changes) of the 9 matrix elements (allowed by cubic symmetry)

    example:
    A = array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
    B = A * Perm+ epsilon
    i.e.
    B = epsilon + [array([[-1., -2.,  0.],
                         [-4., -5., -3.],
                         [-7., -8., -6.]])
    transform_matrix_similar(A, B+0.0001)
    array([[-1.0000e-04,  9.9990e-01,  1.9999e+00],  very similar to A
       [ 2.9999e+00,  3.9999e+00,  4.9999e+00],
       [ 5.9999e+00,  6.9999e+00,  7.9999e+00]])
    """
    _, flags = comparematrices(Aref, B, tol)
    pos = np.where(flags == True)[0][0]
    # permutation matrix
    P = DictLT.OpSymArray[pos]
    return np.dot(B, P)


def AreTwinned(matA, matB, tol=0.001, allpermu=None):
    r"""
    return True if A==B   i.e. if A*S =B with S symetry operator corresponding to twins
    matrix comparaison is done elementwise within tolerance

    matA and matB are single matrix
    """
    if allpermu in (None, "sigma3"):
        dictVect = DictLT.dict_Vect
        allpermu = np.array([dictVect[key_operator]
                for key_operator in list(dictVect.keys())
                if key_operator[:6] == "sigma3"])

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


def RemoveDuplicatesOrientationMatrix(matrices, scores, tol=0.0001,
                            allpermu=None, OutputMatricesOnly=0, verbose=0):
    r"""
    remove duplicates matrix in the sense of comparematrices()
    """
    if len(matrices) == 1:
        return matrices

    #best scored matrices
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
    # from six.moves import filter
    k = 0
    while BSM:

        def Matrixcomparewith(m):
            """
            Return False if m == BSM[0] in the sense of comparematrices()
            """
            boolval = not comparematrices(BSM[0], m, tol=tol, allpermu=allpermu)[0]
            #print("\n*********boolval", boolval)
            return boolval

        if verbose:
            print("k,FilteredMatrixList", k, FilteredMatrixList)
        FilteredMatrixList.append(BSM[0])
        # BSM = [m for m in BSM if Matrixcomparewith(m)]

        BSM = list(filter(lambda m: Matrixcomparewith(np.array(m)), np.array(BSM)))

        k += 1

    if OutputMatricesOnly:
        return FilteredMatrixList
    else:
        # updating scores list
        if verbose:
            print("FilteredMatrixList")
        print(FilteredMatrixList)
        for mat in FilteredMatrixList:
            if isinstance(mat, (np.ndarray, )):
                mat = mat.tolist()
            FilteredScoreList.append(scores[list(Dict_mat.values()).index(mat)])

        return FilteredMatrixList, FilteredScoreList


def MergeSortand_RemoveDuplicates(OrientMatrices, Scores, threshold_matching,
                                                        tol=0.0001, keep_only_equivalent=True, verbose=0):
    r"""
    Returns: Best Sorted Non equivalent orientation matrix (according to matching rate)

    1)Merge matrices solution of distance recognition in LUT
        and matrices coming from user (previous results)

    2)Sort anf threshold according to Scores

    3)Remove Duplicates (taking into account symetry operators)

    threshold_matching : first column of Scores
    keep_only_equivalent         : True all permutation of axes,
                                0, None or False    all matrices even duplicates
    """
    if verbose:
        print('Scores', Scores)

    _hhh = []
    for elem in Scores:
        _hhh.append(elem[:3])
    ar_hhh = np.array(_hhh)

    # threshold
    ind_sup = np.where(ar_hhh[:, 0] >= threshold_matching)[0]

    ar_hhh = np.take(ar_hhh, ind_sup, axis=0)
    OrientMatrices = np.take(OrientMatrices, ind_sup, axis=0)

    # sort
    colnbmatch = np.array(ar_hhh[:, 0], dtype=np.uint16)
    colres = -1.*np.array(ar_hhh[:, 2] * 10000, dtype=np.uint16)
    # hint: lexsort sort lexicographically starting with last key and then second-to-last key!!
    rank = np.lexsort(keys=(colres, colnbmatch))[::-1]

    # print("sorting MAtrices according to rank")
    # print("rank", rank)

    Bestsortedmatrices = np.take(OrientMatrices, rank, axis=0)
    Besthhh = np.take(ar_hhh, rank, axis=0)

    # print "len(OrientMatrices)",len(OrientMatrices)

    # --- we may want to select only matrices above a threshold
    # load list of symetry operators
    if keep_only_equivalent:
        # print("keep_only_equivalent cubic matrices")
        allpermu = DictLT.OpSymArray
        # remove duplicates
        (NonEquivalentMatrices, FilteredScores) = RemoveDuplicatesOrientationMatrix(
                                Bestsortedmatrices, Besthhh, tol=tol, allpermu=allpermu)
        MATRICES = NonEquivalentMatrices
        SCORES = FilteredScores
    elif keep_only_equivalent in (0, None, False):
        (NonEquivalentMatrices, FilteredScores) = RemoveDuplicatesOrientationMatrix(
                                Bestsortedmatrices, Besthhh, tol=tol, allpermu='Id')
        MATRICES = NonEquivalentMatrices
        SCORES = FilteredScores

    return MATRICES, SCORES


# --- -----------  Orientation Matrix handling in map
def findtwins(imageindex, dmat, grainindex=0, nbgrains_per_image=1, imagerange=(1708, 3323), tol=0.01):
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
        print("nbgrains_per_image argument is not compatible with nb of matrices per image in dmat")
        return []

    twins = []
    nbtwins = 0
    for fileindex in list(range(imagerange[0], imagerange[1] + 1)):
        for matrix_grain_ind in list(range(0, nbgrains_per_image)):
            probed_matrix = dmat[fileindex][matrix_grain_ind]
            if probed_matrix != 0:
                if AreTwinned(testmatrix, probed_matrix, tol=tol, allpermu=None)[0]:
                    print("Got twins !: [%d,%d] with [%d,%d] "
                        % (imageindex, grainindex, fileindex, matrix_grain_ind))
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
    for k in list(range(imagerange[0], imagerange[1] + 1)):
        if comparematrices(dmat[imageindex][grainindex], dmat[k][0], tol=tol, allpermu=None)[0]:
            print("Got similar matrices !: [%d,%d]" % (imageindex, k))
            nbsimil += 1
            simil.append(k)

    print("Found %d similar matrices" % nbsimil)
    return simil


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

    # TODO: can be improved by limiting the probed fileindex
    # in the vicinity of a starting fileindex
    #
    # In the present case: starting fileindex is the first image of the dataset (mapping corner)
    # and all images are probed
    # this is really eshaustive and time consuming
    sfileindex = sorted(dmat.keys())
    firstindex = sfileindex[0]
    lastindex = sfileindex[-1]

    # dict of matrix in mapping
    goodpairs = []
    for fileindex in list(range(firstindex, lastindex)):
        mat = dmat[fileindex][0]
        if (fileindex % 100) == 0:
            print("fileindex ", fileindex)

        for fileindex2 in list(range(fileindex + 1, lastindex + 1)):

            if comparematrices(mat, dmat[fileindex2][0], tol=tol)[0]:

                goodpairs.append([fileindex, fileindex2])

    res_final, _ = GT.Set_dict_frompairs(np.array(goodpairs), verbose=0)
    return res_final


def mergesimilarmatrices(dmat):

    #    rr= rFindMatrices(dmat)  # this takes a while
    #    '/home/micha/LaueProjects/CuVia/Carto'
    f = open("dictsimilmat")
    matrix_sets_dict = pickle.load(f)
    f.close()

    # modify dmat according to rr
    dmatm = copy.deepcopy(dmat)
    for key in matrix_sets_dict:
        for val in matrix_sets_dict[key]:
            dmatm[val][0] = dmat[key][0]
    return dmatm


def get4sigma3Matrices(mat):
    """
    from mat return 4 sigma3 matrices (180 deg rotation around 111 family plane)
    """
    listmatsigma = [DictLT.dict_Vect["sigma3_" + str(ind)] for ind in (1, 2, 3, 4)]
    listmat = []
    for elem in listmatsigma:
        listmat.append(np.dot(mat, elem))

    return listmat


# --- ----------  old non OOP methods
def rawMultipleIndexation(bestEULER, indexed_spots_dict, ELEMENT, detectorparameters,
                                                                        AngleTol=1.0, emax=25):
    """
    from a list of 3euler angles index exp spot data
    if matchingrate in 2theta,chi space is higher than MatchingRate_Threshold (in percent)

    input:
    bestEULER             : array of 3 angles
    indexed_spots_dict    : non empty dictionary of exp. spots to be indexed
    AngleTol (deg)        : angular tolerance to accept a link between exp. and theo. spots

    .. note::
        not USED in FileSeries
    """
    MINIMUM_SPOTS = 8
    if len(indexed_spots_dict) < MINIMUM_SPOTS:
        print("too few spots to index")
        return None

    exp_data = getSpotsData(indexed_spots_dict).T
    _, twicetheta_data, chi_data, _, _, intensity_data = exp_data

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
                                                        detectordiameter=165.0,
                                                        verbose=0)

        if indexation_res is not None:
            indexed_spots_dict, nb_updates = updateIndexationDict(
                indexation_res, indexed_spots_dict, index_foundgrain, overwrite=0)

            print("nb of indexed spots for this matrix # %d: %d / %d (theo. nb)"
                % (k, nb_updates, nbtheospots))
            print("with tolerance angle : %.2f deg" % AngleTol)
            dict_grain_matrix[index_foundgrain] = matrix
            dict_grain_matchinrate[index_foundgrain] = [nb_updates, 100.0 * nb_updates / nbtheospots]

            print("Low value of indexed spots may mean that spots have been previously "
            "assigned to previous grain")

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

            indexed_spots_dict[exp_index] = indexed_spots_dict[exp_index][:6] + [miller_indices,
                                                                                energy,
                                                                                grain_index,
                                                                                1]
            nb_updates += 1

    #     print "%d spots have been indexed for grain #%d" % (nb_updates, grain_index)

    return indexed_spots_dict, nb_updates


def getIndexedSpots(OrientMatrix, exp_data, key_material, detectorparameters, useabsoluteindex=None,
                                                            removeharmonics=0,
                                                            veryclose_angletol=1.0,
                                                            emin=5,
                                                            emax=25,
                                                            detectordiameter=None,
                                                            verbose=0,
                                                            dictmaterials=DictLT.dict_Materials):
    """
    return links between experimental and theoretical spots
    (simulated from a grain with OrientMatrix, key_material)

    useabsoluteindex        : array of conversion from relative spot index to absolute one
                            useabsoluteindex[relative]= absolute. When exp_data have been
                            extracted from a bigger data set.

    .. note::
        USED only in rawMultipleIndexation
        not USED in FileSeries
    """

    # experimental data
    twicetheta_data, chi_data, dataI = exp_data

    # simulated data
    grain = CP.Prepare_Grain(key_material, OrientMatrix, dictmaterials=dictmaterials)
    (Twicetheta, Chi, Miller_ind, _, _, Energy) = LAUE.SimulateLaue(grain,
                                                            emin,
                                                            emax,
                                                            detectorparameters,
                                                            removeharmonics=removeharmonics,
                                                            detectordiameter=detectordiameter * 1.25)

    nb_of_simulated_spots = len(Twicetheta)

    res = matchingrate.SpotLinks(twicetheta_data,
                                chi_data,
                                dataI,  # experimental data
                                veryclose_angletol,  # tolerance angle
                                Twicetheta,
                                Chi,
                                Miller_ind,
                                Energy,
                                absoluteindex=useabsoluteindex)

    if res != 0:
        return res, nb_of_simulated_spots
    else:
        return None, None

# def getSpotsFamily(indexed_spots_dict, grain_index):
#     """
#     return spots that belong to the same grain
#     """
#     spots_set = []
#     for key_spot in sorted(indexed_spots_dict.keys()):
#         # spot has been indexed
#         if indexed_spots_dict[key_spot][-1] == 1:
#             if indexed_spots_dict[key_spot][-2] == grain_index:
#                 spots_set.append(key_spot)

#     return spots_set


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
    c = -1  # col isindexed
    cg = -2 # col grainindex

    if indexed_spots_dict[spot_index][c] == 1:
        return indexed_spots_dict[spot_index][cg]
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
                    tth,
                    chi,
                    posX,
                    posY,
                    intensity,
                    Miller,
                    Energy,
                ) = indexed_spots_dict[key_spot][:8]
                # in ambiguous case, spot has not been assigned Miller indices
                if onlywithMiller == 1:
                    if Miller is not None:
                        H, K, L = Miller
                        data.append([index, tth, chi, posX, posY, intensity, H, K, L, Energy])

    return np.array(data)


def proposeEnergyforMatching(indexed_spots_dict, test_Matrix, simulparam, removeharmonics=1):
    """
    return energy for which the matching rate is the highest

    input:

    same input as for getMatchingRate

    """
    ener_min = 15
    ener_max = 25
    step = 0.5
    RES = []
    for energy in np.linspace(
        ener_min, ener_max, int((ener_max - ener_min) / step) + 1):
        simulparam[1] = energy
        res = matchingrate.getMatchingRate(indexed_spots_dict,
                                            test_Matrix,
                                            0.1,
                                            simulparam,
                                            removeharmonics=removeharmonics)
        RES.append([energy] + list(res))

    res = np.array(RES)
    optim_energy = res[np.argmax(res[:, 3])][0]

    return optim_energy


def plotgrains(dict_mat, key_material, detectorparameters, emax, exp_data=None):
    """
    plot grains spots and data from orientmatrix dictionary up to 9

    exp_data        : 2theta, chi ,  ...
    """

    EXTINCTION = DictLT.dict_Materials[key_material][2]

    nb_matrices = 0
    all_tthchi = []
    for key_matrix in sorted(dict_mat.keys()):

        grain = [np.eye(3), EXTINCTION, dict_mat[key_matrix], key_material]

        Twicetheta, Chi, _, _, _, _ = LAUE.SimulateLaue(grain, 5, emax, detectorparameters)

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

    dicocolor = {0: "k", 1: "r", 2: "g", 3: "b", 4: "c", 5: "m"}
    nbcolors = len(dicocolor)

    if exp_data is not None:
        tth_exp, Chi_exp = exp_data[:2]

    # theo spots
    for i_mat in list(range(nb_matrices)):
        ax = fig.add_subplot(codefigure)

        if exp_data is not None:
            # all exp spots
            scatter(tth_exp, Chi_exp, s=40, c="w", marker="o", faceted=True, alpha=0.5)

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
    print("results:")
    #         print bestmat_list, stats_res_list
    print("matrix:                                         matching results")

    #     print "bestmat_list, stats_res_list", bestmat_list, stats_res_list
    if len(stats_res_list) == 1 and len(bestmat_list) != 1:
        bestmat_list = [bestmat_list]
    for mat_elem, stat_elem in zip(bestmat_list, stats_res_list):
        mean_angular_residues = stat_elem[2]
        matching_rate = 100.0 * stat_elem[0] / stat_elem[1]
        for kkk in list(range(3)):
            txt = "%s        " % str(mat_elem[kkk])
            if kkk == 0:
                txt += "res: %s %.3f %.2f" % (str(stat_elem[:2]), mean_angular_residues, matching_rate)
            if len(stat_elem) > 3:
                if kkk == 1:
                    txt += "spot indices %s" % str(stat_elem[3])
                if kkk == 2:
                    txt += "planes %s" % str(stat_elem[4].tolist())
            print("%s" % txt)
        print("")


def plotindexingMap_rgbs(dmat, dmr, dnb, startindex=1708, mapshape=(16, 101)):
    """
    plot indexation map

    dmat, dmr, dnb   :    dicts from big pickled dicts

    default values for CuVia Jul 11

    # TODO: better use Plot_Maps2
    """

    nbcol, nblines = mapshape

    rgbs = [[0.0, 0.0, 0.0]] * (nbcol * nblines)

    # set colors for well indexed grains, black else
    for k in list(dmat.keys()):
        if type(dmat[k][0]) != type(0) and dmr[k][0] > 20.0 and dnb[k][0] >= 6.0:
            # rgbs[k-minindex] = Matrix_to_RGB( Allres[0][k][0] )[0]
            # rgbs[k-minindex] = ORI.myRGB_3(Allres[0][k][0])
            #            rgbs[k - startindex] = Matrix_to_RGB_2(dmat[k][0])
            rgbs[k - startindex] = ORI.myRGB_3(dmat[k][0])

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
    axes.imshow(rgbs, interpolation="nearest")

    numrows, numcols, _ = rgbs.shape

    def format_coord(x, y):
        """ return from x, y , matching rate, nb of peaks image index """
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            red, green, blue = rgbs[row, col]

            Imageindex = tabindex[row, col]
            dmr[Imageindex + startindex]
            #                return 'x=%1.2f, y=%1.2f, col=%1.2f, row=%1.2f ImageIndex: %d' % \
            #                                        (x, y, col, row,
            #                                         Imageindex + startindex)
            return ("r=%1.3f, g=%1.3f, b=%1.3f, MatchingRate=%.1f, Nbpeaks=%d, ImageIndex: %d"
                % (red, green, blue, dmr[Imageindex + startindex][0], dnb[Imageindex + startindex][0],
                                                                        Imageindex + startindex))
        else:
            return ""

    axes.format_coord = format_coord

    show()


def getallMisorientation(dmat, dmr, dnb):
    """
    in dvpt
    """
    zvalue = [np.NaN] * len(dmat)
    listkeys = sorted(dmat.keys())
    for k in listkeys:
        zvalue[k - listkeys[0]] = ORI.getMisorientation(dmat[k][0],
                                                    followVector=np.array([0, 0, -1]))

    return zvalue


def plotindexingMap_scalar(dmat, dmr, dnb, startindex=1708, mapshape=(16, 101)):
    """
    plot indexation map

    default values for CuVia Jul 11

    .. warning:: in dvpt
    """

    nbcol, nblines = mapshape
    zvalue = np.zeros(mapshape)

    listkeys = sorted(dmat.keys())
    for k in listkeys:
        if type(dmat[k][0]) != type(0) and dmr[k][0] > 50.0 and dnb[k][0] >= 40.0:
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
    axes.imshow(zvalue, interpolation="nearest")

    numrows, numcols = zvalue.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            zz = zvalue[row, col]

            Imageindex = tabindex[row, col]
            dmr[Imageindex + startindex]

            return "z=%1.3f, MatchingRate=%.1f, Nbpeaks=%d, ImageIndex: %d" % (
                                                                zz,
                                                                dmr[Imageindex + startindex][0],
                                                                dnb[Imageindex + startindex][0],
                                                                Imageindex + startindex,)
        else:
            return ""

    axes.format_coord = format_coord

    show()


# --- ---------------------- index file series
def mergeDictRes(list_of_dictfiles, outputfilename="MergedRes", dirname=None):
    """
    merge dictionnaries from indexed file series and pickle them in a single file

    dictRes = dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictspots
    """
    dictMaterial = {}
    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}  # crystal frame
    dictstrain_sample = {}  # sample frame
    dictspots = {}

    print("list_of_dictfiles", list_of_dictfiles)

    for _file in list_of_dictfiles:

        # filepickle = open(os.path.join(dirname, _file), 'r')
        # Res = pickle.load(filepickle)

        with open(os.path.join(dirname, _file), "rb") as f:
            Res = pickle.load(f)

        if len(Res) == 6:
            print('len(Res) == 6  No strain in sample frame')
            dMater, dMat, dMR, dNB, dstrain, dspots = Res
            dstrain_sample = {}
        else:
            dMater, dMat, dMR, dNB, dstrain, dstrain_sample, dspots = Res

        #print("dstrain_sample for %s"%_file, dstrain_sample)

        dictMaterial = dict(list(dMater.items()) + list(dictMaterial.items()))
        dictMat = dict(list(dMat.items()) + list(dictMat.items()))
        dictMR = dict(list(dMR.items()) + list(dictMR.items()))
        dictNB = dict(list(dNB.items()) + list(dictNB.items()))
        dictstrain = dict(list(dstrain.items()) + list(dictstrain.items()))
        dictstrain_sample = dict(list(dstrain_sample.items()) + list(dictstrain_sample.items()))
        dictspots = dict(list(dspots.items()) + list(dictspots.items()))

        #print('dictstrain_sample after merging ',dictstrain_sample)

    tuple_dicts = dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictstrain_sample, dictspots

    # output_file = open(os.path.join(dirname, outputfilename), 'w')
    # pickle.dump(tuple_dicts, output_file)
    # output_file.close()

    with open(os.path.join(dirname, outputfilename), "wb") as f:
        pickle.dump(tuple_dicts, f)

    return dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictstrain_sample, dictspots

list_produced_files = []

def log_result(result):
    if len(result) == 2:
        print("********************\n\n\n\n %s \n\n\n\n\n******************" % result[1])
        list_produced_files.append(str(result[1]))

    print("mylog print")


def indexFilesSeries(filepathdat, filepathcor, filepathout,
                    fileprefix, filesuffix, nbdigits_filename,
                    startindex, finalindex, stepindex,
                    filedet,
                     guessedMatricesFile, MinimumMatchingRate,
                     fileirp,
                     spottrackingfile,
                     nb_cpus,
                     reanalyse,
                     use_previous_results,
                     updatefitfiles,
                     trackingmode=False,
                     build_hdf5=True,
                     verbose=0):
    """ index laue spots of several peakslist files using multiprocessing

    """
    dict_param_list = readIndexRefineConfigFile(fileirp)
    NB_MATERIALS = len(dict_param_list)

    CCDCalibdict = None
    if filesuffix in ('.dat',):
        CCDCalibdict = IOLT.readCalib_det_file(filedet)

    Index_Refine_Parameters_dict = {}

    Index_Refine_Parameters_dict["CCDCalibdict"] = CCDCalibdict
    Index_Refine_Parameters_dict["PeakList Folder"] = filepathdat
    Index_Refine_Parameters_dict["PeakListCor Folder"] = filepathcor
    Index_Refine_Parameters_dict["nbdigits"] = nbdigits_filename
    Index_Refine_Parameters_dict["prefixfilename"] = fileprefix
    Index_Refine_Parameters_dict["suffixfilename"] = filesuffix
    Index_Refine_Parameters_dict["prefixdictResname"] = fileprefix + "_dict_"

    Index_Refine_Parameters_dict["PeakListFit Folder"] = filepathout
    Index_Refine_Parameters_dict["Results Folder"] = filepathout

    Index_Refine_Parameters_dict["dict params list"] = dict_param_list
    Index_Refine_Parameters_dict["MinimumMatchingRate"] = MinimumMatchingRate

    if guessedMatricesFile not in ("None", "none", 'NONE', None):
        print("Reading general file for guessed UB solutions")

        assert use_previous_results is False
    else:
        guessedMatricesFile = None

    # read list .mats or single matrix .mat or material, matrix,..., .ubs file
    if guessedMatricesFile is not None:
        if not guessedMatricesFile.endswith(".ubs"):
            _, guessedSolutions = IOLT.readListofMatrices(guessedMatricesFile)

            print("guessedmatrix", guessedSolutions)
            Index_Refine_Parameters_dict["GuessedUBMatrix"] = guessedSolutions
        # read .ubs file
        else:
            Index_Refine_Parameters_dict["CheckOrientation"] = guessedMatricesFile

    # spot restriction or tracking
    Index_Refine_Parameters_dict['mapshape'] = (1000, 1)
    Index_Refine_Parameters_dict['Reference Spots List'] = spottrackingfile

    Index_Refine_Parameters_dict['trackingmode'] = trackingmode

    fileindexrange = (startindex, finalindex, stepindex)
    #------------------------------------------------------------------------
    t00 = time.time()


    max_nb_cpus = multiprocessing.cpu_count()
    nb_cpus = min(nb_cpus, max_nb_cpus)

    if nb_cpus > 1:
        print('using %d cpu(s)'%nb_cpus)
        fileindexdivision = GT.getlist_fileindexrange_multiprocessing(startindex, finalindex, nb_cpus)
        print('dispatch of fileindex ', fileindexdivision)

        saveObject = None
        index_fileseries_3.__defaults__ = (Index_Refine_Parameters_dict,
                                                saveObject,
                                                verbose,
                                                NB_MATERIALS,
                                                False,
                                                fileprefix,
                                                reanalyse,
                                                use_previous_results,
                                                updatefitfiles,
                                                CCDCalibdict)

        pool = multiprocessing.Pool(nb_cpus)
        multiple_results = pool.map(index_fileseries_3, fileindexdivision)

    if nb_cpus == 1:
        print('using only one cpu')

        saveObject = None
        index_fileseries_3.__defaults__ = (Index_Refine_Parameters_dict,
                                                saveObject,
                                                verbose,
                                                NB_MATERIALS,
                                                build_hdf5,
                                                fileprefix,
                                                reanalyse,
                                                use_previous_results,
                                                updatefitfiles,
                                                CCDCalibdict)


        results, output_mergeddicts_filename = index_fileseries_3(fileindexrange)
        multiple_results = [results]


    t_mp = time.time() - t00
    print("Execution time : %.2f" % t_mp)

    if nb_cpus > 1:
        print('nb of results', len(multiple_results))
        print('multiple_results[0]', multiple_results[0])

        # now building summary files (pickled dict and hdf5)
        output_mergeddicts_filename = "%s_dict_%04d_%04d" % (fileprefix, startindex, finalindex)

        list_produced_files = []
        for elem in multiple_results:
            list_produced_files.append(elem[1])

        print("\n\n -------  Building dict summary   --------------\n")
        mergeDictRes(list_produced_files, outputfilename=output_mergeddicts_filename,
                                                                            dirname=filepathout)

        if build_hdf5:
            if sys.version_info.major == 3:
                import LaueTools.Lauehdf5 as LaueHDF5
            else:
                import Lauehdf5 as LaueHDF5

            print("\n\n -------  Building hdf5 file   --------------\n")

            LaueHDF5.build_hdf5(output_mergeddicts_filename, dirname_dictRes=filepathout,
                                                        output_hdf5_filename="dictSUMMARY_%s%04d_%04d.h5"
                                                        % (fileprefix, startindex, finalindex),
                                                        output_dirname=filepathout)

    return multiple_results, output_mergeddicts_filename

def indexing_multiprocessing(fileindexrange, dirname_dictRes=None, Index_Refine_Parameters_dict=None,
                                                                    verbose=0,
                                                                    nb_materials=None,
                                                                    saveObject=0,
                                                                    nb_of_cpu=2,
                                                                    build_hdf5=True,
                                                                    prefixfortitle="",
                                                                    reanalyse=True,
                                                                    use_previous_results=True,
                                                                    updatefitfiles=False,
                                                                    CCDCalibdict=None):
    """
    launch several indexation and unit cell refinement processes in parallel
    """
    if not isinstance(fileindexrange[0], int):
        printred("\n\n ---- Warning! multiprocessing treatment for a specific list of indices is "
                                                                    "not implemented yet !\n\n")

    try:
        if len(fileindexrange) > 2:
            print("\n\n ---- Warning! file STEP INDEX is SET to 1 !\n\n")
        index_start, index_final = fileindexrange[:2]
    except:
        raise ValueError("Need 2 file indices (integers) in fileindexrange=(indexstart, indexfinal)")

    saveObject = 0

    if nb_of_cpu == 1:
        print('using 1 cpu')

        index_fileseries_3.__defaults__ = (Index_Refine_Parameters_dict,
                                                saveObject,
                                                verbose,
                                                nb_materials,
                                                True,
                                                prefixfortitle,
                                                reanalyse,
                                                use_previous_results,
                                                updatefitfiles,
                                                CCDCalibdict)


        multiple_results = index_fileseries_3(fileindexrange)

        return multiple_results

    fileindexdivision = GT.getlist_fileindexrange_multiprocessing(index_start, index_final, nb_of_cpu)

    t00 = time.time()

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

    with multiprocessing.Pool(nb_of_cpu) as pool:
        multiple_results = pool.map(index_fileseries_3, fileindexdivision)

        t_mp = time.time() - t00
        print("Execution time : %.2f" % t_mp)

        print("HOURRA it's FINISHED")

    # now building summary files (pickled dict and hdf5)
    output_mergeddicts_filename = "%s_dict_%04d_%04d" % (prefixfortitle, index_start, index_final)

    list_produced_files = []
    for elem in multiple_results:
        list_produced_files.append(elem[1])

    flag_completed = False
    flag_completed_HDF5 = False
    try:
        #print("intermediate dict_files", list_produced_files)
        mergeDictRes(list_produced_files, outputfilename=output_mergeddicts_filename,
                                                                        dirname=dirname_dictRes)

        flag_completed = True
    except ValueError:#IOError:
        print("\n******************\n")
        print("An error should have occured during at least one thread.\nCheck the error "
            "by using only one CPU!")
        print("\n******************\n")

    if build_hdf5:
        try:
            if sys.version_info.major == 3:
                from . import Lauehdf5 as LaueHDF5
            else:
                import Lauehdf5 as LaueHDF5

            print("\n\n -------  Building hdf5 file   --------------\n")
        except ImportError:
            print("module Lauehdf5 is not installed!")
            print("summaryfile.H5 file won't be created.")

        if dirname_dictRes is None:
            print("folder where hdf5 will be written is None!")
            return flag_completed, flag_completed_HDF5

        flag_completed_HDF5 = LaueHDF5.build_hdf5(output_mergeddicts_filename,
                                                dirname_dictRes=dirname_dictRes,
                                                output_hdf5_filename="dictSUMMARY_%s%04d_%04d.h5"
                                                % (prefixfortitle, index_start, index_final),
                                                output_dirname=dirname_dictRes)

    return multiple_results, flag_completed, flag_completed_HDF5


def index_fileseries_3(fileindexrange, Index_Refine_Parameters_dict=None,
                                        saveObject=0,
                                        verbose=0,
                                        nb_materials=None,
                                        build_hdf5=False,
                                        prefixfortitle="",
                                        reanalyse=True,
                                        use_previous_results=True,
                                        updatefitfiles=False,
                                        CCDCalibdict=None):
    """
    Core procedure to index and refine a serie of peaks list

    :param fileindexrange: list of starting, final and step image index.
        First element can be a path to a file containing list of indices.
        In this case 2nd and 3rd element are meaningless
    :param Index_Refine_Parameters_dict:
    :type Index_Refine_Parameters_dict:
    :param saveObject: save spots dataset object
    :type saveObject: flag
    :param nb_materials: number of materials used in predefined list Index_Refine_Parameters_dict
    :type nb_materials: int
    """

    p = multiprocessing.current_process()
    print("Starting:", p.name, p.pid)

    print('to analyse : ', fileindexrange)

    #     ANGLE_TOL_REMOVE_PEAKS = 0.5

    if CCDCalibdict is None:
        CCDCalibdict = {}

    if "ResolutionAngstrom" in Index_Refine_Parameters_dict:
        ResolutionAngstrom = Index_Refine_Parameters_dict["ResolutionAngstrom"]
    else:
        ResolutionAngstrom = False
        printcyan("default value for ResolutionAngstrom: %s" % str(ResolutionAngstrom))

    if "nLUTmax" in Index_Refine_Parameters_dict:
        nLUTmax = Index_Refine_Parameters_dict["nLUTmax"]
    else:
        nLUTmax = 3
        printcyan("default value for nLUTmax: %d" % nLUTmax)

    dict_params_list = Index_Refine_Parameters_dict["dict params list"]

    if nb_materials is None:
        nb_materials = len(dict_params_list)
    else:
        if nb_materials > len(dict_params_list):
            printred("input nb_of_materials is too high in irp file")

    nbdigits = Index_Refine_Parameters_dict["nbdigits"]
    prefixfilename = Index_Refine_Parameters_dict["prefixfilename"]
    suffixfilename = Index_Refine_Parameters_dict["suffixfilename"]
    prefixdictResname = Index_Refine_Parameters_dict["prefixdictResname"]
    ResultsFolder = Index_Refine_Parameters_dict["Results Folder"]
    fitfile_folder = Index_Refine_Parameters_dict["PeakListFit Folder"]

    if suffixfilename.endswith(".dat"):
        calibparam = CCDCalibdict["CCDCalibParameters"]

    GuessedUBMatrices = [None,]
    if "GuessedUBMatrix" in Index_Refine_Parameters_dict:
        GuessedUBMatrices = Index_Refine_Parameters_dict["GuessedUBMatrix"]

    MinimumMatchingRate = Index_Refine_Parameters_dict["MinimumMatchingRate"]
    print('start of index_fileseries_3() MinimumMatchingRate', MinimumMatchingRate)

    CheckOrientations = None
    #--------  reading .mat .mats or .ubs file  . see field 'Guessed Matrix(ces) ...' in Index_Refine.py
    if "CheckOrientation" in Index_Refine_Parameters_dict:
        CheckOrientationsFile = Index_Refine_Parameters_dict["CheckOrientation"]
        CheckOrientations = IOLT.readCheckOrientationsFile(CheckOrientationsFile)
        # let s start with simple case of two signals

    # preparing dicts of results for all peaklists (key= imageindex)
    # values = several dicts (key = grainindex)
    dictMaterial = {}
    dictMat = {}
    dictMR = {}
    dictNB = {}
    dictstrain = {}
    dictstrain_sample = {}
    dictspots = {}

    if saveObject:
        dict_spotssetObj = {}

    dictRes = dictMaterial, dictMat, dictMR, dictNB, dictstrain, dictstrain_sample, dictspots

    if saveObject:
        todump = dictRes, dict_spotssetObj
    else:
        todump = dictRes

    # --- ----------------
    totalnb_grains = 0
    for material_index in list(range(nb_materials)):
        # read indexation parameters for the current material
        # TODO: add a function to set to default value if key is missing
        dict_param_SingleGrain = dict_params_list[material_index]
        nbGrainstoFind_mat = dict_param_SingleGrain["nbGrainstoFind"]
        totalnb_grains += nbGrainstoFind_mat

    if len(fileindexrange) == 2 and isinstance(fileindexrange[0], int):
        fileindexrange = fileindexrange[0], fileindexrange[1], 1

    if not reanalyse:
        list_files_in_folder = os.listdir(fitfile_folder)
        import re

        test = re.compile("\.res$", re.IGNORECASE)
        list_resfiles_in_folder = list(filter(test.search, list_files_in_folder))

    dict_LUT_material = {}

    # -------------------------------------------------
    # optionally: spots restricted list or spots tracking
    #------------------------------------------------
    # select sort spots order according to the order in a file
    # set self.mapshape
    # write reflist.cor file that will evolve during the loop over images
    # located in dict_param_SingleGrain['Spots Order Reference File']
    sortSpots_from_refenceList = None
    mapshape = None
    refpositionfilepath = None
    refposfile = None

    if "Reference Spots List" in Index_Refine_Parameters_dict:
        refpositionfilepath = Index_Refine_Parameters_dict["Reference Spots List"]
        # file path of spots to be only considered for refinement + shape or map (2D, 3D)
        # reffilepathsortSpots_from_refenceList  =  filepath, dim1, dim2, [dim3 ...]
        if refpositionfilepath not in ('None', "None", None):

            mapshape = Index_Refine_Parameters_dict["mapshape"]

            # from the .fit we create a .cor file and use the procedure of the next branch
            if refpositionfilepath.endswith('.fit'):
                refpositionfilepath = IOLT.convert_fit_to_cor(refpositionfilepath)

            if refpositionfilepath.endswith('.cor'):
                (data_theta,
                Chi,
                posx,
                posy,
                dataintensity,
                _,
                CCDCalibdict) = IOLT.readfile_cor(refpositionfilepath, output_CCDparamsdict=True)[1:]

                # write refposfile  .cor file     'REF_------.cor

                dir_reffile, file_reffile = os.path.split(refpositionfilepath)
                refposfileprefix = os.path.join(dir_reffile, 'REF_' + file_reffile[:-4])
                refposfile = IOLT.writefile_cor(refposfileprefix,
                                2 * data_theta, Chi, posx, posy, dataintensity,
                                param=CCDCalibdict,
                                overwrite=1)
                print('\n\nrefposfile written! It is %s\n\n' % refposfile)
                firstrefposfile = IOLT.writefile_cor(refposfileprefix + 'INITIALREFERENCE',
                                2 * data_theta, Chi, posx, posy, dataintensity,
                                param=CCDCalibdict,
                                overwrite=1)

                print('\n\nrefposfile written! It is %s\n\n'%firstrefposfile)
                #refposfile = refposfileprefix + '.cor'


    #----------------------------------------
    # ---   Building list of indices -------
    #----------------------------------------
    if verbose: print("fileindexrange", fileindexrange)

    if isinstance(fileindexrange[0], int):
        firstindex = fileindexrange[0]
        lastindex = fileindexrange[1]
        indexstep = fileindexrange[2]

        listindices = list(range(firstindex, lastindex + 1, indexstep))
        nstart = listindices[0]
        nend = listindices[-1]

    elif fileindexrange[0].startswith(('[', '(', '{')) or ',' in fileindexrange[0]:
        # I do need to import again re here otherwise UnboundLocalError !!!!
        import re
        listval = re.split("[ ()\[\)\;\,\]\n\t\a\b\f\r\v]", fileindexrange[0])

        listindices = []
        for elem in listval:
            try:
                val = int(elem)
                listindices.append(val)
            except ValueError:
                continue

        nstart = listindices[0]
        nend = listindices[-1]

        firstindex = nstart
        lastindex = nend

    else:
        # reading file of list of indices
        listindices = IOLT.readListofIntegers(fileindexrange[0])
        if listindices is None:
            printred("\n\n*******\nlist indices file: %s contains non integers ! "
                                                    "Please check carefully!" % fileindexrange[0])
            return
        nstart = listindices[0]
        nend = listindices[-1]

        firstindex = nstart
        lastindex = nend

    outputdict_filename = prefixdictResname + "%04d_%04d" % (nstart, nend)

    # image index rearrangment parameters
    # mapshape
    # initial index can differ from firstindex used to scan some data images
    mapfirstimageindex = fileindexrange[0]

    trackingmode = False
    if refpositionfilepath is not None:
        trackingmode = Index_Refine_Parameters_dict["trackingmode"]

        if trackingmode:
            # dict of refposfile at the beginning of each line (nb of lines = mapshape[0])
            print('\n-------Spots Tracking Mode -------\n')
            # dict with:
            # key = image index
            # val = filepath of list of spots nearest in the sample map
            refposfiles = {}
            dim1, dim2 = mapshape # slow , fast axes
            maptableindices = np.arange(dim1 * dim2).reshape((dim1, dim2))

    # -------------------------------------------
    # --- Loop over images ----------------------
    # -------------------------------------------
    previousindex = -1
    for imageindex in listindices:
        # if renanalyse is not checked, we will only consider image and index
        # for which corresponding .res does not exist
        if not reanalyse:
            resfilename = prefixfilename + str(imageindex).zfill(nbdigits) + ".res"
            if resfilename in list_resfiles_in_folder:
                printcyan("file %s exists, corresponding .dat or .cor file is already indexed !!" % resfilename)
                continue
        # consider peak from .dat (only X, Y, I ) (no scattering angles.)
        # So it will use .det to compute 2theta chi scattering angles and write a .cor file
        if suffixfilename.endswith(".dat"):
            datfilename = prefixfilename + str(imageindex).zfill(nbdigits) + suffixfilename

            dirname_in = Index_Refine_Parameters_dict["PeakList Folder"]

            if imageindex == firstindex:
                fullpath = os.path.join(dirname_in, datfilename)
                writeEmptyFileSummary(datfilename, dirname=fitfile_folder)
                if not os.path.exists(fullpath):
                    printred("\n\n*******\nSomething wrong with the first filename: %s. Please check carefully the filename!" % fullpath)

            if not os.path.exists(os.path.join(dirname_in, datfilename)):
                printcyan("Missing file : %s\n Keep on scanning files\n" % datfilename)

                writeEmptyFileSummary(datfilename, dirname=fitfile_folder)
                continue

            # batch to convert from .dat (peak list of X,Y,I) to .cor (2theta,Chi,X,Y,I)
            if verbose:
                print("build .cor file")
                print("in %s" % Index_Refine_Parameters_dict["PeakListCor Folder"])
            F2TC.convert2corfile(datfilename,
                                calibparam,
                                dirname_in=dirname_in,
                                dirname_out=Index_Refine_Parameters_dict["PeakListCor Folder"],
                                CCDCalibdict=CCDCalibdict,
                                add_props=True)

            corfilename = datfilename.split(".")[0] + ".cor"

        # consider peaks from .cor file (that could have been created at the previous branch from .dat file)
        elif suffixfilename == ".cor":
            corfilename = prefixfilename + str(imageindex).zfill(nbdigits) + suffixfilename
            dirname_in = Index_Refine_Parameters_dict["PeakListCor Folder"]

            if not os.path.exists(os.path.join(dirname_in, corfilename)):
                printcyan("Missing file : %s\n Keep on scanning files\n" % corfilename)
                continue

        file_to_index = os.path.join(Index_Refine_Parameters_dict["PeakListCor Folder"],
                                                    corfilename)

        print("\n\n ---------------INDEXING----------    file : \n%s\n\n" % file_to_index)

        DataSet = spotsset()

        # preparing dicts of results for indexation of peaks list  (nb elems = requested nb of grains)
        # corresponding to image with imageindex
        dictMaterial[imageindex] = [None for kk in list(range(totalnb_grains))]  # key material list
        dictMat[imageindex] = [None for kk in list(range(totalnb_grains))]  # UB matrix  list
        dictMR[imageindex] = [-1 for kk in list(range(totalnb_grains))]  # Matching rate list
        dictNB[imageindex] = [-1 for kk in list(range(totalnb_grains))] # nb of indexed spots list
        dictstrain[imageindex] = [0 for kk in list(range(totalnb_grains))]  # dev strain list
        dictstrain_sample[imageindex] = [0 for kk in list(range(totalnb_grains))]  # dev strain in sample frame list

        # ----------------------------------------------
        # --- Loop over materials (or single grain) ----
        # ----------------------------------------------
        for material_index in list(range(nb_materials)):

            dict_param_SingleGrain = dict_params_list[material_index]

            # no data to load if grainindex > 0
            if len(DataSet.indexedgrains) > 0:
                starting_grainindex = DataSet.indexedgrains[-1] + 1

            # loading data of spots to be indexed for the first time
            else:
                starting_grainindex = 0

                # we may consider only spots present in refposfile (same number and ordered spots)
                if refposfile is not None:
                    if trackingmode:
                        #  finding the closest reference position file in sample map
                        if imageindex != firstindex:
                            i, j = np.where(maptableindices == imageindex)
                            #(iprior, jprior), imageindexprior, _

                            maxdistance = max(indexstep*2, mapshape[0]+1)  # to get some candidates

                            (_, _), imageindexprior, _ = GT.best_prior_array_element(i, j, mapshape,
                                                            maxdist=maxdistance,
                                                            startingindex=mapfirstimageindex,
                                                            existingabsindices=list(refposfiles.keys()))
                            print('best located refpositions for imageindex = %d'%imageindexprior)

                            refposfile = refposfiles[imageindexprior]

                        else:
                            print('TRACKING: first image %d: reference file is %s'%(imageindex, firstrefposfile))
                            refposfiles[imageindex] = firstrefposfile

                #---------------------------------------------------------------
                # read data and calibration parameters from .cor file
                #---------------------------------------------------------------
                DataSet.importdatafromfile(file_to_index,
                        refpositionfilepath=refposfile, trackingmode=trackingmode)

                if verbose: print('nb total of spots to be analysed: %d'%len(DataSet.getUnIndexedSpots()))

                if refpositionfilepath is not None:
                    if trackingmode:
                        # update refposfile at each change of imageindex
                        #'.cor file generated by DataSet.importdatafromfile()  above'
                        refposfiles[imageindex] = DataSet.refpositionfilepath

                if DataSet.nbspots < 3:
                    if verbose: print("%d spot(s) are too few to be indexed" % DataSet.nbspots)
                    DataSet.LUT = None
                    continue

            if verbose: print("\n ########### starting_grainindex %d ###########\n" % starting_grainindex)
            key_material = dict_param_SingleGrain["key material"]
            emin = dict_param_SingleGrain["emin"]
            emax = dict_param_SingleGrain["emax"]
            nbGrainstoFind = dict_param_SingleGrain["nbGrainstoFind"]

            try:
                set_central_spots_hkl = GT.fromstringtolist(dict_param_SingleGrain["setCentralSpotsHKL"])
            except ValueError:
                printred("Can't understand '%s' as a list like [h,k,l] for 'setCentralSpotsHKL' parameter in .irp file"
                    % dict_param_SingleGrain["setCentralSpotsHKL"])
                return

            try:
                central_spots_list = GT.read_elems_from_string(dict_param_SingleGrain["central spots indices"])
            except ValueError:
                printred(
                    "Can't understand '%s' as a list of spot indices for 'central spots indices' parameter in .irp file"
                    % dict_param_SingleGrain["central spots indices"])
                return

            try:
                List_Matching_Tol_Angles = GT.read_elems_from_string(
                    dict_param_SingleGrain["List Matching Tol Angles"], map_function=float)
            except ValueError:
                printred("Can't understand '%s' as a list of spot indices for 'List Matching Tol Angles' parameter in .irp file"
                    % dict_param_SingleGrain["List Matching Tol Angles"])
                return

            dict_loop = {"MATCHINGRATE_THRESHOLD_IAL": dict_param_SingleGrain["MATCHINGRATE THRESHOLD IAL"],
                        "MATCHINGRATE_ANGLE_TOL": dict_param_SingleGrain["MATCHINGRATE ANGLE TOL"],
                        "NBMAXPROBED": dict_param_SingleGrain["NBMAXPROBED"],
                        "central spots indices": central_spots_list,
                        "AngleTolLUT": dict_param_SingleGrain["AngleTolLUT"],
                        "UseIntensityWeights": dict_param_SingleGrain["UseIntensityWeights"],
                        "MinimumNumberMatches": dict_param_SingleGrain["MinimumNumberMatches"],
                        "MinimumMatchingRate": MinimumMatchingRate}

            if "nbSpotsToIndex" not in dict_param_SingleGrain:
                dict_param_SingleGrain["nbSpotsToIndex"] = 1000

            dict_loop["nbSpotsToIndex"] = dict_param_SingleGrain["nbSpotsToIndex"]

            if "ResolutionAngstrom" in dict_param_SingleGrain:
                ResolutionAngstrom = dict_param_SingleGrain["ResolutionAngstrom"]
                if ResolutionAngstrom in ("False", None, 0, 0.0):
                    ResolutionAngstrom = False
                else:
                    try:
                        ResolutionAngstrom = float(ResolutionAngstrom)
                    except ValueError:
                        txterror = "ResolutionAngstrom can not be converted in float !!\n"
                        txterror += "Please check the irp file"
                        printred(txterror)
                        return

            ResolutionAngstromLUT = False
            if "ResolutionAngstromLUT" in dict_param_SingleGrain:
                ResolutionAngstromLUT = dict_param_SingleGrain["ResolutionAngstromLUT"]
                if ResolutionAngstromLUT in ("False", None, 0, 0.0):
                    ResolutionAngstromLUT = False
                else:
                    try:
                        ResolutionAngstromLUT = float(ResolutionAngstromLUT)
                    except ValueError:
                        txterror = "ResolutionAngstromLUT can not be converted in float !!\n"
                        txterror += "Please check the irp file"
                        printred(txterror)
                        return

            if "nLUTmax" in dict_param_SingleGrain:
                nLUTmax = dict_param_SingleGrain["nLUTmax"]
                if nLUTmax in ("False", None, 0, 0.0, 1, 2, 1.0, 2.0, 3.0):
                    nLUTmax = 3
                else:
                    try:
                        nLUTmax = int(nLUTmax)
                    except ValueError:
                        printred("nLUTmax can not be converted to integer !!\nPlease check the irp file")
                        return

            previousResults = None
            # ------------------------------------------------
            # index image n using n-1 result
            #------------------------------------------------
            # read a guessed orientation matrices in dictMat
            # TODO we could think about : first  checkorientation and if not successful then previous results 
            if use_previous_results and CheckOrientations is None:

                GuessedUBMatrices = []
                if verbose:
                    print("use_previous_result:  True")
                    print("current index", imageindex)

                # TODO: to improve find the closest analysed solution in sample map and 2D or 1D image index list
                if previousindex in dictMat:

                    MRs = []
                    NBs = []
                    for k, ub in enumerate(dictMat[previousindex]):
                        if ub is not None:
                            GuessedUBMatrices.append(ub)
                            MRs.append(dictMR[previousindex][k])
                            NBs.append(dictNB[previousindex][k])

                    print("previous analysed index", previousindex)

                print('GuessedUBMatrices', GuessedUBMatrices)
                if imageindex >= nstart and GuessedUBMatrices != []:
                    # first matrix only

                    # list of matrices
                    nbAddMatrices = len(GuessedUBMatrices)
                    addMatrices = GuessedUBMatrices

                    previousResults = (nbAddMatrices, addMatrices,
                                            MRs, NBs)

                    print("previousResults", previousResults)

            # ----------------------------------------------------------
            # test if the set of {orientation, material, energy max} can index
            # -----------------------------------------
            # from .ubs file
            # now previousResults is set to None
            # TODO we could think about: first checkorientation and if not successfulthen previous results 
            CheckOrientationParams = None
            # material_index  : index material in irp file
            # nb_materials
            if CheckOrientations is not None:
                print("CheckOrientations mode (read .ubs file ) ")
                k_checkUB = material_index % nb_materials
                # materials from irp file = materials in .ubs file
                # for k, UBsparams in enumerate(CheckOrientations):
                #     print("UBsparams", UBsparams)
                #     print("key_material", key_material)
                #     #print("key_material", type(key_material))
                #     if key_material in UBsparams:
                #         print("yaouuh")
                #         CheckOrientationParams = [UBsparams]
                #         previousResults = None
                #         break
                CheckOrientationParams = [CheckOrientations[k_checkUB]]

                previousResults = None
            #----------------------------------------------------------------
            # check orientation from previous results already written in corresponding .fit file
            # ----------------------------------------------------
            elif updatefitfiles:
                # read orientation matrix, material from _g#.fit file
                # warning: since the loop is made over material_index,
                # to properly scan and update every g#.fit files
                # one needs to have an .irp file with a single nbgrainstofind=1 per material.
                # Otherwise, if (nbofmaterials in irp) < (nb of _g#.fit files)
                #     then only the first (nbofmaterials in irp) _g#.fit will be updated

                # TODO: ability to change emax
                EMAX = emax
                # set lowest threshold to accept indexation for next step of refinement
                # this threshold comes from the GUI index_Refine.py
                MINIMUM_MATCHING_RATE = MinimumMatchingRate

                fitfilename = (prefixfilename + str(imageindex).zfill(nbdigits)
                    + "_g%d.fit" % material_index)

                (list_indexedgrains_indices, _, _,
                Material_list,
                all_UBmats_flat,
                _) = IOLT.readfitfile_multigrains(os.path.join(fitfile_folder, fitfilename),
                                                            return_toreindex=True)

                nbindexedgrains = len(list_indexedgrains_indices)

                CheckOrientationParams = []
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
                                                    EMAX,
                                                    MINIMUM_MATCHING_RATE,
                                                    all_UBmats_flat.reshape((nbindexedgrains, 3, 3))])

                    if verbose:
                        print("CheckOrientationParams", CheckOrientationParams)

            if key_material in dict_LUT_material:
                LUT = dict_LUT_material[key_material]
            else:
                LUT = None

            # ----------------------------------------------------------
            # index spots data set with IndexSpotsSet method
            # -----------------------------------------
            DataSet.IndexSpotsSet(file_to_index, key_material, emin, emax, dict_loop, None,
                                                starting_grainindex=starting_grainindex,
                                                use_file=0,
                                                IMM=False,
                                                n_LUT=nLUTmax,
                                                LUT=LUT,
                                                set_central_spots_hkl=set_central_spots_hkl,
                                                ResolutionAngstrom=ResolutionAngstrom,
                                                angletol_list=List_Matching_Tol_Angles,
                                                MatchingRate_List=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10 steps of refinements is highly reasonable
                                                nbGrainstoFind=nbGrainstoFind,
                                                verbose=verbose,
                                                corfilename=corfilename,
                                                dirnameout_fitfile=fitfile_folder,
                                                previousResults=previousResults,
                                                CheckOrientations=CheckOrientationParams)

            dict_LUT_material[key_material] = DataSet.LUT

            nbgrains_indexed = len(DataSet.indexedgrains)

            DataSet.writecorFile_unindexedSpots(corfilename=corfilename,
                                                dirname=fitfile_folder,
                                                filename_nbdigits=nbdigits)

            if verbose: print("nb indexed grains", nbgrains_indexed)
            DataSet.merge_fitfiles(nbgrains_indexed,
                                    corfilename=corfilename,
                                    dirname=fitfile_folder,
                                    removefiles=0,
                                    add_unindexed_spotslist=True)

            DataSet.writeFileSummary(corfilename=corfilename, dirname=fitfile_folder)

        dictspots[imageindex] = DataSet.getSummaryallData()

        if saveObject:
            dict_spotssetObj[imageindex] = DataSet

        # filling dicts with indexation results
        for grainindex in list(range(totalnb_grains)):
            if grainindex in DataSet.indexedgrains:
                dictMaterial[imageindex][grainindex] = DataSet.dict_indexedgrains_material[grainindex]
                dictMat[imageindex][grainindex] = DataSet.dict_grain_matrix[grainindex]
                dictMR[imageindex][grainindex] = DataSet.dict_grain_matching_rate[grainindex][1]
                dictNB[imageindex][grainindex] = DataSet.dict_grain_matching_rate[grainindex][0]
                dictstrain[imageindex][grainindex] = DataSet.dict_grain_devstrain[grainindex]
                dictstrain_sample[imageindex][grainindex] = DataSet.dict_grain_devstrain_sample[grainindex]

        if verbose:
            print("dictMaterial", dictMaterial)
            print("dictMat", dictMat)
            print("dictMR", dictMR)
            print("dictNB", dictNB)
            print("dictstrain", dictstrain)
            print("dictstrain_sample", dictstrain_sample)
        # intermediate saving
        if (imageindex % 10) == 0:
            #            dictRes = dictMat, dictMR, dictNB
            # filepickle = open(os.path.join(ResultsFolder, outputdict_filename), 'w')
            # pickle.dump(todump, filepickle)
            # filepickle.close()

            with open(os.path.join(ResultsFolder, outputdict_filename), "wb") as f:
                pickle.dump(todump, f)

        previousindex = imageindex

    # pickle LUT
    with open(os.path.join(ResultsFolder, "LUT"), "wb") as f:
        pickle.dump(DataSet.LUT, f)

    # pickle dict
    print('WRITING DICT: %s'%outputdict_filename)
    with open(os.path.join(ResultsFolder, outputdict_filename), "wb") as f:
        pickle.dump(todump, f)

        print("************************\n\n\n\nCompleted process for %s:"
            % str([nstart, nend, fileindexrange[2]]),
            p.name,
            p.pid)

        if build_hdf5:
            if sys.version_info.major == 3:
                from . import Lauehdf5 as LaueHDF5
            else:
                import Lauehdf5 as LaueHDF5

            LaueHDF5.build_hdf5(outputdict_filename,
                                dirname_dictRes=ResultsFolder,
                                output_hdf5_filename="dict_Res_%s.h5" % prefixfortitle,
                                output_dirname=ResultsFolder)

    return todump, outputdict_filename


#    return DataSet_Si, DataSet

# --- -----------  Config file .irp file I/O  (Index Refine Parameters)
LIST_OPTIONS_INDEXREFINE = ["nb materials",
                            "key material",
                            "nbGrainstoFind",
                            "emin",
                            "emax",
                            "MATCHINGRATE THRESHOLD IAL",
                            "AngleTolLUT",
                            "MATCHINGRATE ANGLE TOL",
                            "NBMAXPROBED",
                            "MinimumNumberMatches",
                            "central spots indices",
                            "ResolutionAngstrom",
                            "nLUTmax",
                            "setCentralSpotsHKL",
                            "UseIntensityWeights",
                            "nbSpotsToIndex",
                            "List Matching Tol Angles"]

CONVERTKEY_dict = {"nb materials": "nb materials",
                    "key material": "key material",
                    "nbgrainstofind": "nbGrainstoFind",
                    "emin": "emin",
                    "emax": "emax",
                    "matchingrate threshold ial": "MATCHINGRATE THRESHOLD IAL",
                    "angletollut": "AngleTolLUT",
                    "matchingrate angle tol": "MATCHINGRATE ANGLE TOL",
                    "nbmaxprobed": "NBMAXPROBED",
                    "minimumnumbermatches": "MinimumNumberMatches",
                    "central spots indices": "central spots indices",
                    "resolutionangstrom": "ResolutionAngstrom",
                    "nlutmax": "nLUTmax",
                    "setcentralspotshkl": "setCentralSpotsHKL",
                    "useintensityweights": "UseIntensityWeights",
                    "nbspotstoindex": "nbSpotsToIndex",
                    "list matching tol angles": "List Matching Tol Angles"}

LIST_OPTIONS_TYPE_INDEXREFINE = ["int", "string", "int", "float", "float", "float", "float",
                                "float", "int", "int", "string", "string", "integer", "string",
                                "string", "int", "string"]

# WARNING when adding parameters above:
# check in index_refine.py if field position is correct in def hascorrectvalue(self, kk, val):
# and name MUST NOT CONTAIN "_"


def saveIndexRefineConfigFile(dict_param, outputfilename=None):
    # save peaksearch parameter in config file
    config = CONF.RawConfigParser()
    config.add_section("IndexRefine")

    params_comments = "Index and Refinement parameters\n"

    config.set("IndexRefine", "nb materials", str(len(dict_param)))

    for kk in list(range(len(dict_param))):
        for key, val in list(dict_param[kk].items()):
            key_mat = key + "_%d" % kk
            params_comments += "# " + key_mat + " : " + str(val) + "\n"
            config.set("IndexRefine", key_mat, str(val))

    if outputfilename is None:
        outputfilename = "IndexRefine.irp"

    if not outputfilename.endswith(".irp"):
        if outputfilename.count(".") > 0:

            outputfilename = "".join(outputfilename.split(".")[:-1] + ".irp")
        else:
            outputfilename += ".irp"

    # Writing configuration file to 'PeakSearch.cfg'
    with open(outputfilename, "w") as configfile:
        config.write(configfile)

    return outputfilename


def readIndexRefineConfigFile(filename):
    """
    read config .irp files

    return: dict of parameters for indexation and refinement procedures
    """
    config = CONF.RawConfigParser()
    config.optionxform = str
    #    config = MyCasePreservingConfigParser()

    config.read(filename)

    section = config.sections()[0]

    if section not in ("IndexRefine",):
        raise ValueError("wrong section name in config file %s. Must be in %s"
            % (filename, "IndexRefine"))

    print("section", section)

    dict_param = {}

    # list_options = config.options(section)
    list_options_gen = []
    list_material = []
    for opt in config.options(section):
        splt_opt = opt.split("_")
        if len(splt_opt) == 2:
            optgen, mat_index = splt_opt
            list_options_gen.append(optgen)
            list_material.append(int(mat_index))
        elif len(splt_opt) > 2:
            raise ValueError("WARNING name of fields in GUI must not contain '_'!!")

    nb_materials = int(config.getint(section, "nb materials"))

    dict_param = [{} for k in list(range(nb_materials))]

    #     print "list_options_gen, list_material"
    #     print list_options_gen, list_material

    for option, matindex in zip(list_options_gen, list_material):

        if matindex >= nb_materials:
            continue

        # print("\n option, matindex\n", option, matindex)
        for option_ref, option_type in zip(LIST_OPTIONS_INDEXREFINE, LIST_OPTIONS_TYPE_INDEXREFINE):

            #             print "option_ref, option_type", option_ref, option_type

            if option_ref == option or option_ref.lower() == option:

                try:
                    optionkey = CONVERTKEY_dict[option_ref]
                except KeyError:
                    optionkey = option_ref

                option_lower = option_ref.lower() + "_%d" % matindex
                try:
                    if option_type == "int":
                        val = int(config.getint(section, option_lower))

                    elif option_type == "float":
                        val = float(config.getfloat(section, option_lower))
                    else:
                        val = config.get(section, option_lower)

                    # print("val", val)
                    dict_param[matindex][optionkey] = val
                except ValueError:
                    print("Value of option '% s' has not the correct type" % option)
                    return None

                break

    return dict_param


if __name__ == "__main__":

    MainFolder = '/home/micha/LaueProjects/SiSibulle_Lukas'
    print("MainFolder", MainFolder)

    #  fields like on the GUI   FileSeries/Index_Refine.py

    filepathdat = MainFolder
    filepathcor = os.path.join(MainFolder, "corfiles")
    filepathout = os.path.join(MainFolder, "fitfiles")

    fileprefix = "LukasHR_"
    filesuffix = '.dat'
    nbdigits_filename = 4
    startindex = 0
    finalindex = 5
    stepindex = 1

    filedet = '/home/micha/LaueProjects/SiSibulle_Lukas/calibSilukas.det'

    guessedMatricesFile = None
    MinimumMatchingRate = 4
    fileirp = os.path.join(MainFolder, "SiSi.irp")
    spottrackingfile = '/home/micha/LaueProjects/SiSibulle_Lukas/SiLukas_acceptedspots.cor'

    nb_cpus = 2

    # boolean
    reanalyse = True
    use_previous_results = True
    updatefitfiles = False


    multiple_results, filedict = indexFilesSeries(filepathdat, filepathcor, filepathout,
                        fileprefix, filesuffix, nbdigits_filename,
                        startindex, finalindex, stepindex,
                        filedet,
                        guessedMatricesFile, MinimumMatchingRate,
                        fileirp,
                        spottrackingfile,
                        nb_cpus,
                        reanalyse,
                        use_previous_results,
                        updatefitfiles,
                        verbose=1)
