import os
import sys
import pickle
import numpy as np
import pylab as pp
import scipy.ndimage as SCI
import tables as Tab
import h5py 

if Tab.__version__ >= "3.4.3":
    Tab.openFile = Tab.open_file
    Tab.File.createGroup = Tab.File.create_group
    Tab.File.createTable = Tab.File.create_table
    Tab.File.createArray = Tab.File.create_array

if sys.version_info.major == 3:
    #from . import generaltools as GT
    
    import LaueTools.generaltools as GT
    import LaueTools.readmccd as RMCCD
    import LaueTools.IOLaueTools as IOLT
    import LaueTools.dict_LaueTools as DictLT
    import LaueTools.orientations as ORI
else:

    import generaltools as GT
    import readmccd as RMCCD
    import IOLaueTools as IOLT
    import dict_LaueTools as DictLT
    import orientations as ORI


MAX_NUMBER_GRAINS = 3
# --- ----------   Build hdf5 from dicts
class IndexedImage(Tab.IsDescription):
    fileindex = Tab.UInt16Col()
    MatchingRate_0 = Tab.Float32Col()  # float  (single-precision)
    MatchingRate_1 = Tab.Float32Col()  # float  (single-precision)
    MatchingRate_2 = Tab.Float32Col()  # float  (single-precision)
    NBindexed_0 = Tab.Float32Col()  # float  (single-precision)
    NBindexed_1 = Tab.Float32Col()  # float  (single-precision)
    NBindexed_2 = Tab.Float32Col()  # float  (single-precision)


class Matrices(Tab.IsDescription):
    fileindex = Tab.UInt16Col()
    UB11_0 = Tab.Float32Col()
    UB12_0 = Tab.Float32Col()
    UB13_0 = Tab.Float32Col()
    UB21_0 = Tab.Float32Col()
    UB22_0 = Tab.Float32Col()
    UB23_0 = Tab.Float32Col()
    UB31_0 = Tab.Float32Col()
    UB32_0 = Tab.Float32Col()
    UB33_0 = Tab.Float32Col()

    UB11_1 = Tab.Float32Col()
    UB12_1 = Tab.Float32Col()
    UB13_1 = Tab.Float32Col()
    UB21_1 = Tab.Float32Col()
    UB22_1 = Tab.Float32Col()
    UB23_1 = Tab.Float32Col()
    UB31_1 = Tab.Float32Col()
    UB32_1 = Tab.Float32Col()
    UB33_1 = Tab.Float32Col()

    UB11_2 = Tab.Float32Col()
    UB12_2 = Tab.Float32Col()
    UB13_2 = Tab.Float32Col()
    UB21_2 = Tab.Float32Col()
    UB22_2 = Tab.Float32Col()
    UB23_2 = Tab.Float32Col()
    UB31_2 = Tab.Float32Col()
    UB32_2 = Tab.Float32Col()
    UB33_2 = Tab.Float32Col()


class Matrices_array(Tab.IsDescription):
    fileindex = Tab.UInt16Col()
    UB0 = Tab.Float32Col(shape=(3, 3))
    UB1 = Tab.Float32Col(shape=(3, 3))
    UB2 = Tab.Float32Col(shape=(3, 3))


class DevStrain(Tab.IsDescription):
    fileindex = Tab.UInt16Col()
    devstrain_11_0 = Tab.Float32Col()
    devstrain_22_0 = Tab.Float32Col()
    devstrain_33_0 = Tab.Float32Col()
    devstrain_23_0 = Tab.Float32Col()
    devstrain_13_0 = Tab.Float32Col()
    devstrain_12_0 = Tab.Float32Col()

    devstrain_11_1 = Tab.Float32Col()
    devstrain_22_1 = Tab.Float32Col()
    devstrain_33_1 = Tab.Float32Col()
    devstrain_23_1 = Tab.Float32Col()
    devstrain_13_1 = Tab.Float32Col()
    devstrain_12_1 = Tab.Float32Col()

    devstrain_11_2 = Tab.Float32Col()
    devstrain_22_2 = Tab.Float32Col()
    devstrain_33_2 = Tab.Float32Col()
    devstrain_23_2 = Tab.Float32Col()
    devstrain_13_2 = Tab.Float32Col()
    devstrain_12_2 = Tab.Float32Col()

class DevStrainSample(Tab.IsDescription):
    fileindex = Tab.UInt16Col()
    devstrainsample_11_0 = Tab.Float32Col()
    devstrainsample_22_0 = Tab.Float32Col()
    devstrainsample_33_0 = Tab.Float32Col()
    devstrainsample_23_0 = Tab.Float32Col()
    devstrainsample_13_0 = Tab.Float32Col()
    devstrainsample_12_0 = Tab.Float32Col()

    devstrainsample_11_1 = Tab.Float32Col()
    devstrainsample_22_1 = Tab.Float32Col()
    devstrainsample_33_1 = Tab.Float32Col()
    devstrainsample_23_1 = Tab.Float32Col()
    devstrainsample_13_1 = Tab.Float32Col()
    devstrainsample_12_1 = Tab.Float32Col()

    devstrainsample_11_2 = Tab.Float32Col()
    devstrainsample_22_2 = Tab.Float32Col()
    devstrainsample_33_2 = Tab.Float32Col()
    devstrainsample_23_2 = Tab.Float32Col()
    devstrainsample_13_2 = Tab.Float32Col()
    devstrainsample_12_2 = Tab.Float32Col()


# Define a user record to characterize some kind of particles
class SpotsList(Tab.IsDescription):
    spotindex = Tab.UInt16Col()  # 32-character String
    twotheta = Tab.Float32Col()
    chi = Tab.Float32Col()


class AllIndexedSpots(Tab.IsDescription):
    """fields needed to build a large data of spots in all mapping images
    """

    fileindex = Tab.UInt16Col()
    grainindex = Tab.Int16Col()
    spotindex = Tab.UInt16Col()
    pixX = Tab.Float32Col()
    pixY = Tab.Float32Col()
    twotheta = Tab.Float32Col()
    chi = Tab.Float32Col()
    intensity = Tab.Float32Col()
    H = Tab.Int32Col()
    K = Tab.Int32Col()
    L = Tab.Int32Col()
    energy = Tab.Float32Col()
    MatchingRate = Tab.Float32Col()
    NbindexedSpots = Tab.UInt16Col()
    UB11 = Tab.Float32Col()
    UB12 = Tab.Float32Col()
    UB13 = Tab.Float32Col()
    UB21 = Tab.Float32Col()
    UB22 = Tab.Float32Col()
    UB23 = Tab.Float32Col()
    UB31 = Tab.Float32Col()
    UB32 = Tab.Float32Col()
    UB33 = Tab.Float32Col()
    devstrain_11 = Tab.Float32Col()
    devstrain_22 = Tab.Float32Col()
    devstrain_33 = Tab.Float32Col()
    devstrain_23 = Tab.Float32Col()
    devstrain_13 = Tab.Float32Col()
    devstrain_12 = Tab.Float32Col()
    devstrainsample_11 = Tab.Float32Col()
    devstrainsample_22 = Tab.Float32Col()
    devstrainsample_33 = Tab.Float32Col()
    devstrainsample_23 = Tab.Float32Col()
    devstrainsample_13 = Tab.Float32Col()
    devstrainsample_12 = Tab.Float32Col()
    peak_amplitude = Tab.Float32Col()
    peak_background = Tab.Float32Col()
    peak_width1 = Tab.Float32Col()
    peak_width2 = Tab.Float32Col()
    peak_inclin = Tab.Float32Col()
    MeanPixDev = Tab.Float32Col()
    PixDev = Tab.Float32Col()
    PixDev_x = Tab.Float32Col()
    PixDev_y = Tab.Float32Col()
    PixMax = Tab.Float32Col()


list_ub_element = ["UB11",
                    "UB12",
                    "UB13",
                    "UB21",
                    "UB22",
                    "UB23",
                    "UB31",
                    "UB32",
                    "UB33"]


list_devstrain_element = ["devstrain_11",
                        "devstrain_22",
                        "devstrain_33",
                        "devstrain_23",
                        "devstrain_13",
                        "devstrain_12"]

list_devstrainsample_element = ["devstrainsample_11",
                                "devstrainsample_22",
                                "devstrainsample_33",
                                "devstrainsample_23",
                                "devstrainsample_13",
                                "devstrainsample_12"]

pos_voigt = [0, 4, 8, 5, 2, 1]

# (peak_X,peak_Y,peak_Itot peak_Isub,peak_fwaxmaj,peak_fwaxmin,
#    peak_inclination,Xdev,Ydev,peak_bkg, Pixmax)


# def table_for_Sionly():
#    f = open('dict_tsvSi0000_6560')
#    dicts = pickle.load(f)
#    f.close()
#
#    dictMat, dictMR, dictNB, dictstrain, dictspots = dicts
#
#    filename = "testdict.h5"
#    imagefilename_prefix = 'tsv1234_'
#    # Open a file in "w"rite mode
#     = Tab.openFile(filename, mode="w", title="Test file my data")
#    # Create a new group under "/" (root)
#    group_images = .createGroup("/", 'Images', 'Detector information')
#    # Create one table on it
#    table = h5file.create_table(group_images, 'matching_rate_Si', IndexedImage, "Readout example")
#    # Fill the table with 10 particles
#    indexedImage = table.row
#    keys_indexfile = dictMat.keys()[:200]
#
#    print "keys_indexfile", keys_indexfile
#    for key_image in keys_indexfile:
#        indexedImage['fileindex'] = imagefilename_prefix + '%06d.mccd' % key_image
#        indexedImage['MatchingRate_0'] = dictMR[key_image][0]  # 1 grain
#
#        # Insert a new particle record
#        indexedImage.append()
#
#        group_spots = h5file.createGroup("/Images", 'spots%06d' % key_image, 'experimental spots information')
#        table_spots = h5file.create_table(group_spots, 'spots_table', SpotsList, "spots table list")
#        spotslist = table_spots.row
#        for elem in dictspots[key_image][0]:  # 1 grain
#            spotslist['spotindex'] = elem[0]
#            spotslist['twotheta'] = elem[1]
#            spotslist['chi'] = elem[2]
#            spotslist.append()
#
#        table_spots.flush()
#    table.flush()
#
#    group_spots = h5file.createGroup("/", 'Allspots', 'All spots information')
#
#    tableallspots = h5file.create_table(group_spots, 'total_spots', AllIndexedSpots, "Readout example")
#    # Fill the table with 10 particles
#    allIndexedSpots = tableallspots.row
#    keys_indexfile = dictMat.keys()
#
#    print "keys_indexfile", keys_indexfile
#    for key_image in keys_indexfile:
#        for elem in dictspots[key_image][0]:  # 1 grain
#            allIndexedSpots['fileindex'] = key_image
#
#            allIndexedSpots['grainindex'] = 0
#
#            (allIndexedSpots['spotindex'],
#             allIndexedSpots['twotheta'],
#             allIndexedSpots['chi'],
#             allIndexedSpots['pixX'],
#             allIndexedSpots['pixY'],
#             allIndexedSpots['intensity'],
#             allIndexedSpots['H'],
#             allIndexedSpots['K'],
#             allIndexedSpots['L'],
#             allIndexedSpots['energy']) = elem
#
#
#
#            allIndexedSpots.append()
#
#        tableallspots.flush()
#
#    # Close (and flush) the file
#    h5file.close()


def build_hdf5( filename_dictRes, dirname_dictRes=None, output_hdf5_filename="dict_Res.h5",
                    output_dirname="/home/micha/LaueProjects/Ni_joint",
                    imagefilename_prefix="TSVCU",  # for title only in metadata
                    max_nb_grains=2,  # maximum number of grains per image
                    nb_of_spots_per_image=200,  # estimation for optimization in data access
                ):
    """
    build hdf5 (summary) file from index and refine results on file series 
    """

    if dirname_dictRes is None:
        dirname_dictRes = os.path.abspath(os.curdir)

    # f = open(os.path.join(dirname_dictRes, filename_dictRes))
    # dicts = pickle.load(f)
    # f.close()

    with open(os.path.join(dirname_dictRes, filename_dictRes), "rb") as f:
        dicts = pickle.load(f)

    print('there are %d dicts '%len(dicts))
    if len(dicts) == 5:
        dictMat, dictMR, dictNB, dictstrain, dictspots = dicts
    elif len(dicts) == 6:
        _, dictMat, dictMR, dictNB, dictstrain, dictspots = dicts
    else:
        
        _, dictMat, dictMR, dictNB, dictstrain, dictstrain_sample, dictspots = dicts

        print('dictstrain_sample',dictstrain_sample)

    keys_indexfile = sorted(dictMat.keys())

    nb_of_images = len(keys_indexfile)
    print("number of images: %d" % nb_of_images)

    if nb_of_images == 0:
        return False

    print("starting image index: %d" % min(keys_indexfile))
    print("final image index: %d" % max(keys_indexfile))
    #    nb_of_images = 3  # for test only

    # Open a file in "w"rite mode
    full_output_path = os.path.join(output_dirname, output_hdf5_filename)
    #h5file = Tab.openFile(full_output_path, mode="w", title="%s" % imagefilename_prefix)
    h5file = Tab.open_file(full_output_path, mode="w", title="%s" % imagefilename_prefix)

    keys_indexfile = keys_indexfile[:nb_of_images]
    # ----------------------------------------
    if 1:  # Class IndexedImage
        # Create a new group under "/" (root)
        #group_images = h5file.createGroup("/", "Indexation", "Indexation figure")
        group_images = h5file.create_group("/", "Indexation", "Indexation figure")


        # Create one table on it
        table = h5file.create_table(group_images, "matching_rate", IndexedImage, "indexation rate")
        # Fill the table with data
        indexedImage = table.row

        print("dictMR", dictMR)

        #    print "keys_indexfile", keys_indexfile
        for key_image in keys_indexfile:
            indexedImage["fileindex"] = key_image
            for grainindex in range(max_nb_grains):
                if grainindex in dictMR[key_image]:
                    indexedImage["MatchingRate_%d" % grainindex] = dictMR[key_image][grainindex]  # 1 grain
                if grainindex in dictNB[key_image]:
                    indexedImage["NBindexed_%d" % grainindex] = dictNB[key_image][grainindex]

            # Insert a new particle record
            indexedImage.append()

        table.flush()
        # -------------------------------------

    # ----------------------------------------
    if 1:  # Class Matrices
        #        # Create a new group under "/" (root)
        #        group_images = h5file.createGroup("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tableUB = h5file.create_table(group_images, "UB_matrices", Matrices, "UB Matrices elements")
        # Fill the table with data
        indexedUB = tableUB.row
        print("using Lauehdf5.....")
        for key_image in keys_indexfile:
            indexedUB["fileindex"] = key_image
            print("key_image", key_image)
            print("dictMat[key_image]", dictMat[key_image])
            nbmatrices = len(dictMat[key_image])
            grainindex = 0
            while grainindex < max_nb_grains:
                # default 'zero matrix'
                UBmatrix = np.zeros(9)
                if grainindex < nbmatrices:
                    exp_mat = dictMat[key_image][grainindex]
                else:
                    exp_mat = 0
                if isinstance(exp_mat, (int, float)):
                    # go to the next key_image
                    break

                UBmatrix = np.ravel(exp_mat)
                for k, ub_element in enumerate(list_ub_element):
                    #                    print ub_element + '_%d' % grainindex
                    #                    print UBmatrix[k]
                    indexedUB[ub_element + "_%d" % grainindex] = UBmatrix[k]
                grainindex += 1

            # Insert a new particle record
            indexedUB.append()

            # -------------------------------
        tableUB.flush()
        # -------------------------------------

    # TODO to check compatibility with max_nb_grains indexError ?
    if 1:  # Class Matrices_array
        #        # Create a new group under "/" (root)
        #        group_images = h5file.createGroup("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tableUB_array = h5file.create_table(group_images, "UB_matrices_array", Matrices_array,
                                                                            "UB Matrices elements")
        # Fill the table with data
        indexedUB_ar = tableUB_array.row

        for key_image in keys_indexfile:
            indexedUB_ar["fileindex"] = key_image
            #            print "key_image", key_image
            for grainindex in range(max_nb_grains):
                UBmatrix = np.zeros(9)
                # TODO better find nbgrains for each solution than try except ?
                try:
                    exp_mat = dictMat[key_image][grainindex]
                    if not isinstance(exp_mat, int):
                        indexedUB_ar["UB%d" % grainindex] = exp_mat
                except IndexError:
                    continue

            # Insert a new particle record
            indexedUB_ar.append()

            # -------------------------------
        tableUB_array.flush()
        # -------------------------------------

    if 1:  # Class DevStrain
        #        # Create a new group under "/" (root)
        #        group_images = h5file.createGroup("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tabledev = h5file.create_table(group_images, "Devstrain_matrices", DevStrain,
                                                                            "UB Matrices elements")
        # Fill the table with data
        indexedDev = tabledev.row

        for key_image in keys_indexfile:
            indexedDev["fileindex"] = key_image
            nbmatrices = len(dictstrain[key_image])
            grainindex = 0
            while grainindex < max_nb_grains:
                devstrainmatrix = np.zeros(6)
                if grainindex < nbmatrices:
                    exp_dev = dictstrain[key_image][grainindex]
                else:
                    exp_dev = 0
                if isinstance(exp_dev, (int, float)):
                    # go to the next key_image
                    break

                devstrainmatrix = np.take(np.ravel(exp_dev), pos_voigt)
                for k, strain_element in enumerate(list_devstrain_element):
                    indexedDev[strain_element + "_%d" % grainindex] = devstrainmatrix[k]

                grainindex += 1

            # Insert a new particle record
            indexedDev.append()

            # -------------------------------
        tabledev.flush()
        # -------------------------------------

    if 1:  # Class DevStrain  sample frame
        #        # Create a new group under "/" (root)
        #        group_images = h5file.create_group("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tabledevS = h5file.create_table(group_images, "DevstrainSample_matrices", DevStrainSample,
                                                                            "UB Matrices elements")
        # Fill the table with data
        indexedDevS = tabledevS.row

        for key_image in keys_indexfile:
            indexedDevS["fileindex"] = key_image
            #print("dictstrain_sample[key_image]", dictstrain_sample[key_image])
            nbmatrices = len(dictstrain_sample[key_image])
            grainindex = 0
            while grainindex < max_nb_grains:
                devstrainsamplematrix = np.zeros(6)
                if grainindex < nbmatrices:
                    exp_dev = dictstrain_sample[key_image][grainindex]
                else:
                    exp_dev = 0
                if isinstance(exp_dev, (int, float)):
                    # go to the next key_image
                    break

                devstrainsamplematrix = np.take(np.ravel(exp_dev), pos_voigt)
                for k, strain_element in enumerate(list_devstrainsample_element):
                    indexedDevS[strain_element + "_%d" % grainindex] = devstrainsamplematrix[k]

                grainindex += 1

            # Insert a new particle record
            indexedDevS.append()

            # -------------------------------
        tabledevS.flush()
        # -------------------------------------

    if 1:  # class AllIndexedSpots

        group_spots = h5file.create_group("/", "Allspots", "All spots information")

        tableallspots = h5file.create_table(group_spots,
                                            "total_spots",
                                            AllIndexedSpots,
                                            "Readout example",
                                            expectedrows=nb_of_images * nb_of_spots_per_image)

        allIndexedSpots = tableallspots.row

        print("keys_indexfile", keys_indexfile)
        for key_image in keys_indexfile:
            # print("key_image", key_image, "--------------------------/n/n")
            for elem in dictspots[key_image]:
                allIndexedSpots["fileindex"] = key_image
                grainindex = int(elem[-2])

                allIndexedSpots["grainindex"] = elem[-2]
                allIndexedSpots["spotindex"] = elem[0]
                (allIndexedSpots["twotheta"],
                    allIndexedSpots["chi"],
                    allIndexedSpots["pixX"],
                    allIndexedSpots["pixY"],
                    allIndexedSpots["intensity"]) = elem[1:6]
                    
                (allIndexedSpots["H"],
                    allIndexedSpots["K"],
                    allIndexedSpots["L"],
                    allIndexedSpots["energy"],
                ) = elem[-6:-6+4]

                # new fields
                allIndexedSpots["MatchingRate"] = 0.0
                allIndexedSpots["NbindexedSpots"] = 0
                UBmatrix = np.zeros(9)
                devstrainmatrix = np.zeros(6)
                devstrainsamplematrix = np.zeros(6)

                #                print "spot index", elem[0]
                #                print "grainindex", int(elem[1])
                if grainindex >= 0 and grainindex < max_nb_grains:
                    #                    print "grain is indexed!"
                    allIndexedSpots["MatchingRate"] = dictMR[key_image][grainindex]
                    allIndexedSpots["NbindexedSpots"] = dictNB[key_image][grainindex]
                    # UB matrix: in dictMat
                    UBmatrix = np.ravel(dictMat[key_image][grainindex])
                    # deviatoric strain matrix: in dictstrain
                    devstrainmatrix = np.take(np.ravel(dictstrain[key_image][grainindex]), pos_voigt)

                for k, ub_element in enumerate(list_ub_element):
                    allIndexedSpots[ub_element] = UBmatrix[k]

                for k, strain_element in enumerate(list_devstrain_element):
                    allIndexedSpots[strain_element] = devstrainmatrix[k]

                for k, strainS_element in enumerate(list_devstrainsample_element):
                    allIndexedSpots[strainS_element] = devstrainsamplematrix[k]

                #                allIndexedSpots['peak_amplitude'] =

                #                peak_amplitude = Tab.Float32Col()
                #                peak_background = Tab.Float32Col()
                #                peak_width1 = Tab.Float32Col()
                #                peak_width2 = Tab.Float32Col()
                #                peak_inclin = Tab.Float32Col()
                #                PixDev_x = Tab.Float32Col()
                #                PixDev_y = Tab.Float32Col()
                #                PixMax = Tab.Float32Col()

                allIndexedSpots.append()

            tableallspots.flush()
    #
    # Close (and flush) the file
    h5file.close()

    return True


def Add_allspotsSummary_from_dict( Summary_HDF5_filename, filename_dictRes,
                                                            Summary_HDF5_dirname=None,
                                                            dirname_dictRes=None,
                                                            nb_of_spots_per_image=200):
    """
    open a hdf5 file and create a branch of allspots summary
    """
    full_HDF5_output_path = Summary_HDF5_filename
    if Summary_HDF5_dirname is not None:
        full_HDF5_output_path = os.path.join(Summary_HDF5_dirname, Summary_HDF5_filename)

    # Open a file in append mode
    h5file = Tab.openFile(full_HDF5_output_path, mode="a", title="%s" % "allspots")

    # load date from pickled dict
    if dirname_dictRes is None:
        dirname_dictRes = os.path.abspath(os.curdir)

    f = open(os.path.join(dirname_dictRes, filename_dictRes))
    dicts = pickle.load(f)
    f.close()

    if len(dicts) == 5:
        dictMat, dictMR, dictNB, dictstrain, dictspots = dicts
    elif len(dicts) == 6:
        _, dictMat, dictMR, dictNB, dictstrain, dictspots = dicts
    else:
        _, dictMat, dictMR, dictNB, dictstrain, dictstrain_sample, dictspots = dicts

    keys_indexfile = sorted(dictMat.keys())

    nb_of_images = len(keys_indexfile)
    print("number of images: %d" % nb_of_images)

    if nb_of_images == 0:
        return False

    print("starting image index: %d" % min(keys_indexfile))
    print("final image index: %d" % max(keys_indexfile))
    #    nb_of_images = 3  # for test only

    # Open a file in "w"rite mode

    keys_indexfile = keys_indexfile[:nb_of_images]

    group_spots = h5file.create_group("/", "Allspots", "All spots information")

    tableallspots = h5file.create_table(group_spots, "total_spots", AllIndexedSpots,
                                                "Readout example",
                                                expectedrows=nb_of_images * nb_of_spots_per_image)

    allIndexedSpots = tableallspots.row

    print("keys_indexfile", keys_indexfile)
    for key_image in keys_indexfile:
        print("key_image", key_image, "--------------------------/n/n")
        for elem in dictspots[key_image]:
            allIndexedSpots["fileindex"] = key_image
            grain_index = int(elem[1])

            (allIndexedSpots["spotindex"],
                allIndexedSpots["grainindex"],
                allIndexedSpots["twotheta"],
                allIndexedSpots["chi"],
                allIndexedSpots["pixX"],
                allIndexedSpots["pixY"],
                allIndexedSpots["intensity"],
                allIndexedSpots["H"],
                allIndexedSpots["K"],
                allIndexedSpots["L"],
                allIndexedSpots["energy"],
            ) = elem

            # new fields
            allIndexedSpots["MatchingRate"] = 0.0
            allIndexedSpots["NbindexedSpots"] = 0
            UBmatrix = np.zeros(9)
            devstrainmatrix = np.zeros(6)
            devstrainsamplematrix = np.zeros(6)

            #                print "spot index", elem[0]
            #                print "grainindex", int(elem[1])
            if grain_index >= 0:
                #                    print "grain is indexed!"
                allIndexedSpots["MatchingRate"] = dictMR[key_image][grain_index]
                allIndexedSpots["NbindexedSpots"] = dictNB[key_image][grain_index]
                # UB matrix: in dictMat
                UBmatrix = np.ravel(dictMat[key_image][grain_index])
                # deviatoric strain matrix: in dictstrain
                devstrainmatrix = np.take(np.ravel(dictstrain[key_image][grain_index]), pos_voigt)

            for k, ub_element in enumerate(list_ub_element):
                allIndexedSpots[ub_element] = UBmatrix[k]

            for k, strain_element in enumerate(list_devstrain_element):
                allIndexedSpots[strain_element] = devstrainmatrix[k]

            for k, strainS_element in enumerate(list_devstrainsample_element):
                    allIndexedSpots[strainS_element] = devstrainsamplematrix[k]

            #                allIndexedSpots['peak_amplitude'] =

            #                peak_amplitude = Tab.Float32Col()
            #                peak_background = Tab.Float32Col()
            #                peak_width1 = Tab.Float32Col()
            #                peak_width2 = Tab.Float32Col()
            #                peak_inclin = Tab.Float32Col()
            #                PixDev_x = Tab.Float32Col()
            #                PixDev_y = Tab.Float32Col()
            #                PixMax = Tab.Float32Col()

            allIndexedSpots.append()

        tableallspots.flush()
    #
    # Close (and flush) the file
    h5file.close()


def Add_allspotsSummary_from_fitfiles( Summary_HDF5_filename, prefix_fitfiles, fitfiles_folder,
                                                                fileindex_list,
                                                                Summary_HDF5_dirname=None,
                                                                number_of_digits_in_image_name=4,
                                                                filesuffix=".fit",
                                                                nb_of_spots_per_image=200,
                                                                max_nb_grains=3):
    """
    open a hdf5 file and create or Add a Node of /Allspots/total_spots with spots data

    Dataspots are not updated but simply appended (no test if data for a given index in fileindex_list
    already exists
    """
    full_HDF5_output_path = Summary_HDF5_filename
    if Summary_HDF5_dirname is not None:
        full_HDF5_output_path = os.path.join(Summary_HDF5_dirname, Summary_HDF5_filename)

    # Open a file in 'append' mode
    h5file = Tab.open_file(full_HDF5_output_path, mode="a", title="%s" % "allspots")

    nb_of_images = len(fileindex_list)

    try:
        tableallspots = h5file.get_node("/Allspots/total_spots")
        Node_tableallspots_alreadyExists = True
    except Tab.NodeError:
        Node_tableallspots_alreadyExists = False

    if not Node_tableallspots_alreadyExists:

        group_spots = h5file.create_group("/", "Allspots", "All spots information")
        tableallspots = h5file.create_table( group_spots, "total_spots", AllIndexedSpots,
                                    "Readout example",
                                    expectedrows=max(nb_of_images * nb_of_spots_per_image, 10000))

    allIndexedSpots = tableallspots.row

    try:
        table = h5file.get_node("/Indexation/matching_rate")
        Node_tableimage_alreadyExists = True
    except Tab.NodeError:
        Node_tableimage_alreadyExists = False

    if not Node_tableimage_alreadyExists:

        group_images = h5file.create_group("/", "Indexation", "Indexation figure")
        table = h5file.create_table(group_images, "matching_rate", IndexedImage, "indexation rate")

    indexedImage = table.row

    try:
        tableUB = h5file.get_node("/Indexation/UB_matrices_LT")
        Node_tableUB_alreadyExists = True
    except Tab.NodeError:
        Node_tableUB_alreadyExists = False

    if not Node_tableUB_alreadyExists:
        tableUB = h5file.create_table(group_images, "UB_matrices_LT", Matrices, "UB Matrices elements")

    indexedUB = tableUB.row

    try:
        tableUB_sample = h5file.get_node("/Indexation/UB_matrices_SampleFrame")
        Node_tableUB_alreadyExists = True
    except Tab.NodeError:
        Node_tableUB_alreadyExists = False

    if not Node_tableUB_alreadyExists:
        tableUB_sample = h5file.create_table(group_images, "UB_matrices_SampleFrame", Matrices,
                                                                            "UB Matrices elements")

    indexedUB_sample = tableUB_sample.row

    list_files_in_folder = os.listdir(fitfiles_folder)
    import re

    test = re.compile("\.fit$", re.IGNORECASE)
    list_fitfiles_in_folder = list(filter(test.search, list_files_in_folder))

    if Node_tableallspots_alreadyExists:
        allindices = tableallspots.read(field="fileindex")
        print("fileindex already present", allindices)

    encodingdigits = "%%0%dd" % int(number_of_digits_in_image_name)
    # loop for reading each .fit file
    for fileindex in fileindex_list:

        # remove and replace by new data
        if Node_tableallspots_alreadyExists:
            if fileindex in allindices:
                # delete all rows corresponding to fileindex

                # rows = self.tableallspots.getWhereList('fileindex == %d' % fileindex)
                rowpos = np.where(allindices == fileindex)[0]
                for row_to_delete in rowpos:
                    tableallspots.remove_rows(row_to_delete)

        _filename = prefix_fitfiles + encodingdigits % fileindex + filesuffix

        if _filename not in list_fitfiles_in_folder:
            print("Warning! missing .fit file: %s" % _filename)
            continue

        filefitmg = os.path.join(fitfiles_folder, _filename)

        # read data from .fit file (grains and unindexed spots)
        resIndexed, resUnindexed = IOLT.readfitfile_multigrains(filefitmg,
                                                                verbose=0,
                                                                readmore=True,
                                                                fileextensionmarker=".cor",
                                                                returnUnindexedSpots=True)

        #         print "resIndexed", resIndexed

        if resIndexed != 0:
            (list_indexedgrains_indices, list_nb_indexed_peaks, list_starting_rows_in_data,
                all_UBmats_flat, allgrains_spotsdata, calibJSM,
                list_pixdev, list_strain6, list_euler, ) = resIndexed

            print("read hdf5 : list_pixdev", list_pixdev)
            if len(list_pixdev) == 0:
                print("setting pixdev to -1 for file: %s" % _filename)
                list_pixdev = [-1 for kkk in range(len(list_indexedgrains_indices))]

        key_image = fileindex

        print("fileindex", fileindex, "--------------------------")
        for k, grain_index in enumerate(list_indexedgrains_indices):
            start_row = list_starting_rows_in_data[k]
            final_row = start_row + list_nb_indexed_peaks[k]

            spotsdata_for_this_grain = allgrains_spotsdata[start_row:final_row]
            for spot_data in spotsdata_for_this_grain:
                allIndexedSpots["fileindex"] = key_image

                (allIndexedSpots["spotindex"],
                    allIndexedSpots["intensity"],
                    allIndexedSpots["H"], allIndexedSpots["K"], allIndexedSpots["L"],
                    allIndexedSpots["twotheta"], allIndexedSpots["chi"],
                    allIndexedSpots["pixX"], allIndexedSpots["pixY"],
                    allIndexedSpots["energy"],
                    allIndexedSpots["grainindex"],
                    allIndexedSpots["PixDev"],
                ) = spot_data

                # new fields
                #                 allIndexedSpots['MatchingRate'] = 0.0
                allIndexedSpots["MeanPixDev"] = list_pixdev[k]
                allIndexedSpots["NbindexedSpots"] = list_nb_indexed_peaks[k]

                UBmatrix_flat = np.zeros(9)
                devstrainmatrix = np.zeros(6)
                devstrainsamplematrix = np.zeros(6)
                #
                #                print "spot index", elem[0]
                #                print "grainindex", int(elem[1])
                if grain_index >= 0:

                    UBmatrix_flat = all_UBmats_flat[k]

                    devstrainmatrix = list_strain6[k]
                else:
                    UBmatrix_flat = DEFAULT_TO_RESET[5 : 5 + 9]
                    devstrainmatrix = DEFAULT_TO_RESET[14 : 14 + 6]

                for kk, ub_element in enumerate(list_ub_element):
                    allIndexedSpots[ub_element] = UBmatrix_flat[kk]

                for kk, strain_element in enumerate(list_devstrain_element):
                    allIndexedSpots[strain_element] = devstrainmatrix[kk]

                for kk, strainS_element in enumerate(list_devstrainsample_element):
                    allIndexedSpots[strainS_element] = devstrainsamplematrix[kk]

                allIndexedSpots.append()

        indexedUB["fileindex"] = key_image
        # default values
        for k in range(max_nb_grains):
            UBmatrix_flat = DEFAULT_TO_RESET[5 : 5 + 9]
            for kk, ub_element in enumerate(list_ub_element):
                indexedUB[ub_element + "_%d" % k] = UBmatrix_flat[kk]
        # grains values
        for k, grain_index in enumerate(list_indexedgrains_indices):
            # data for images indexation performance
            UBmatrix_flat = all_UBmats_flat[k]
            for kk, ub_element in enumerate(list_ub_element):
                indexedUB[ub_element + "_%d" % k] = UBmatrix_flat[kk]
        indexedUB.append()

        indexedImage["fileindex"] = key_image
        # default values
        for k in range(max_nb_grains):
            indexedImage["NBindexed_%d" % k] = DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Nb_indexedspots"]
        # grains values
        for k, grain_index in enumerate(list_indexedgrains_indices):
            # data for images indexation performance
            indexedImage["NBindexed_%d" % k] = list_nb_indexed_peaks[k]
        indexedImage.append()

        # append unindexedspots:
        nb_of_unindexedspots = len(resUnindexed)
        print("\nadding %d unindexed spots\n" % nb_of_unindexedspots)

        for spot_data in resUnindexed:
            #             print "spot_data", spot_data
            allIndexedSpots["fileindex"] = key_image

            (allIndexedSpots["spotindex"],
                allIndexedSpots["intensity"],
                allIndexedSpots["twotheta"], allIndexedSpots["chi"],
                allIndexedSpots["pixX"], allIndexedSpots["pixY"],
            ) = spot_data

            (allIndexedSpots["H"], allIndexedSpots["K"], allIndexedSpots["L"],
                allIndexedSpots["energy"],
                allIndexedSpots["grainindex"],
                allIndexedSpots["PixDev"],
            ) = (DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["H"],
                DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["K"],
                DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["L"],
                DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Energy"],
                DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["grainindex"],
                DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["PixDev"])

            UBmatrix_flat = DEFAULT_TO_RESET[5 : 5 + 9]
            devstrainmatrix = DEFAULT_TO_RESET[14 : 14 + 6]

            for kk, ub_element in enumerate(list_ub_element):
                allIndexedSpots[ub_element] = UBmatrix_flat[kk]

            for kk, strain_element in enumerate(list_devstrain_element):
                allIndexedSpots[strain_element] = devstrainmatrix[kk]

            # new fields
            #                 allIndexedSpots['MatchingRate'] = 0.0
            allIndexedSpots["MeanPixDev"] = DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["PixDev"]
            allIndexedSpots["NbindexedSpots"] = DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Nb_indexedspots"]

            allIndexedSpots.append()

        tableallspots.flush()
        table.flush()
        tableUB.flush()
    #
    # Close (and flush) the file
    h5file.close()

    print('Summary file is built!: %s'%full_HDF5_output_path)


def build_hdf5_fromSummaryFile(
    Summaryfilename="/home/micha/LaueTools/UO2_20_90_serree__SUMMARY_0_to_609_add_columns.dat",
    Summarydirname=None,
    Dict_Res_filename="/home/micha/LaueProjects/UO2_raq1ter/fitfiles_test/UO2_20_90_serree__dict_0000_0010",
    dirname_Dict_Res=None,
    output_hdf5_filename="dict_Res.h5",
    output_dirname="/home/micha/LaueProjects/Ni_joint",
    imagefilename_prefix="TSVCU",  # for title only in metadata
    max_nb_grains=2,  # maximum number of grains per image
    nb_of_spots_per_image=200,  # estimation for optimization in data access
):
    """
    build hdf5 (summary) file from summary file generated by FileSeries/build_summary (multigrain)
    """

    bigdata, list_col_names, dict_col_names = IOLT.ReadSummaryFile(Summaryfilename,
                                                                    dirname=Summarydirname)
    # dict [key]=val where key = image index and val = list of indexed grains indices
    dict_images_indexed_grains = {}
    # dict [key]=val where key = image index and val = dict_locate
    # where dict_locate is a dict[key]=val where key = grain_index and val=rowindex in bigdata
    dict_locate_grains = {}
    for row_index in range(len(bigdata)):
        key_image, grain_index = bigdata[row_index][:2]

        key_image, grain_index = int(key_image), int(grain_index)

        print("key_image, grain_index", key_image, grain_index)
        if key_image in dict_images_indexed_grains:
            dict_images_indexed_grains[key_image].append(grain_index)
        else:
            print("new image")
            dict_images_indexed_grains[key_image] = [grain_index]
            dict_locate_grains[key_image] = {}

        dict_locate_grains[key_image][grain_index] = row_index

    print(dict_images_indexed_grains)

    keys_indexfile = sorted(list(dict_images_indexed_grains.keys()))

    nb_of_images = len(keys_indexfile)
    print("number of images: %d" % nb_of_images)

    if nb_of_images == 0:
        return False

    print("starting image index: %d" % min(keys_indexfile))
    print("final image index: %d" % max(keys_indexfile))
    #    nb_of_images = 3  # for test only

    # Open a file in "w"rite mode
    full_HDF5_output_path = os.path.join(output_dirname, output_hdf5_filename)
    h5file = Tab.openFile(full_HDF5_output_path, mode="w", title="%s" % imagefilename_prefix)

    keys_indexfile = keys_indexfile[:nb_of_images]
    #     keys_indexfile = keys_indexfile[:10]
    # ----------------------------------------
    if 1:  # Class IndexedImage
        # Create a new group under "/" (root)
        group_images = h5file.create_group("/", "Indexation", "Indexation figure")

        # Create one table on it
        table = h5file.create_table(group_images, "matching_rate", IndexedImage, "indexation rate")
        # Fill the table with data
        indexedImage = table.row

        #    print "keys_indexfile", keys_indexfile
        for key_image in keys_indexfile:
            indexedImage["fileindex"] = key_image
            print("key_image", key_image)
            for grain_index in range(max_nb_grains):
                nbpeaks = 0.0
                print("dict_images_indexed_grains[key_image]", dict_images_indexed_grains[key_image])
                print("grain_index", grain_index)
                if grain_index in dict_images_indexed_grains[key_image]:
                    print("youpi")
                    bigdata_row = bigdata[dict_locate_grains[key_image][grain_index]]
                    print("bigdata_row")
                    nbpeaks = bigdata_row[dict_col_names["npeaks"]]

                print("nbpeaks", nbpeaks)
                indexedImage["MatchingRate_%d" % grain_index] = (nbpeaks + 100)  # must be integer!!
                indexedImage["NBindexed_%d" % grain_index] = nbpeaks

            # Insert a new particle record
            indexedImage.append()

        table.flush()
        # -------------------------------------

    # ----------------------------------------
    if 1:  # Class Matrices
        #        # Create a new group under "/" (root)
        #        group_images = h5file.create_group("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tableUB = h5file.create_table(group_images,
                                    "UB_matrices",
                                    Matrices,
                                    "matstarlab UB Matrices elements in OR frame")
        # Fill the table with data
        indexedUB = tableUB.row

        for key_image in keys_indexfile:
            indexedUB["fileindex"] = key_image
            print("key_image", key_image)
            for grain_index in range(max_nb_grains):
                matstarlab_line = np.zeros(9)
                if grain_index in dict_images_indexed_grains[key_image]:
                    print("youpi")
                    bigdata_row = bigdata[dict_locate_grains[key_image][grain_index]]
                    col_start_matrix = dict_col_names["matstarlab_0"]
                    matstarlab_line = bigdata_row[col_start_matrix : col_start_matrix + 9]
                    print("matstarlab_line", matstarlab_line)

                for k, ub_element in enumerate(list_ub_element):
                    print(ub_element + "_%d" % grain_index)
                    print(matstarlab_line[k])

                    indexedUB[ub_element + "_%d" % grain_index] = matstarlab_line[k]

            # Insert a new particle record
            indexedUB.append()

            # -------------------------------
        tableUB.flush()
        # -------------------------------------

    # test
    if 1:  # Class Matrices_array
        #        # Create a new group under "/" (root)
        #        group_images = h5file.create_group("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tableUB_array = h5file.create_table(group_images, "UB_matrices_array", Matrices_array,
                                                                            "UB Matrices elements")
        # Fill the table with data
        indexedUB_ar = tableUB_array.row

        for key_image in keys_indexfile:
            indexedUB_ar["fileindex"] = key_image
            #            print "key_image", key_image
            for grain_index in range(max_nb_grains):
                exp_mat = np.zeros((3, 3))
                if grain_index in dict_images_indexed_grains[key_image]:
                    print("youpi")
                    bigdata_row = bigdata[dict_locate_grains[key_image][grain_index]]
                    col_start_matrix = dict_col_names["matstarlab_0"]
                    matstarlab_line = bigdata_row[col_start_matrix : col_start_matrix + 9]
                    exp_mat = np.reshape(matstarlab_line, (3, 3))
                    print("exp_mat", matstarlab_line)

                indexedUB_ar["UB%d" % grain_index] = exp_mat

            # Insert a new particle record
            indexedUB_ar.append()

            # -------------------------------
        tableUB_array.flush()
        # -------------------------------------

    if 1:  # Class DevStrain
        #        # Create a new group under "/" (root)
        #        group_images = h5file.create_group("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tabledev = h5file.create_table(group_images, "Deviatoric_strain", DevStrain, "voigt notation")
        # Fill the table with data
        indexedDev = tabledev.row

        for key_image in keys_indexfile:
            indexedDev["fileindex"] = key_image
            for grain_index in range(max_nb_grains):
                devstrainmatrix = np.zeros(6)

                if grain_index in dict_images_indexed_grains[key_image]:
                    print("youpi")
                    bigdata_row = bigdata[dict_locate_grains[key_image][grain_index]]
                    col_start_matrix = dict_col_names["strain6_crystal_0"]
                    devstrainmatrix = bigdata_row[col_start_matrix : col_start_matrix + 6]

                for k, strain_element in enumerate(list_devstrain_element):
                    indexedDev[strain_element + "_%d" % grain_index] = devstrainmatrix[k]

            # Insert a new particle record
            indexedDev.append()

            # -------------------------------
        tabledev.flush()
        # -------------------------------------
    # Close (and flush) the file
    h5file.close()

    Add_allspotsSummary_from_dict(full_HDF5_output_path,
                                    Dict_Res_filename,
                                    Summary_HDF5_dirname=None,
                                    dirname_dictRes=dirname_Dict_Res)

    return True


# --- --------------  QUERY the database


class Ask:
    def __init__(self, what, where, tableofdata):
        # list of field or column in hdf5 file
        self.what = what
        # string expression
        self.where = where
        # name of table of data (string)
        self.table = tableofdata

    def setSentence(self):
        expr_xcol = "["
        for elem in self.what:
            expr_xcol += "x['%s']," % elem
        expr_xcol = expr_xcol[:-1] + "]"
        self.whatexpression = expr_xcol
        self.wheresentence = '%s.where("""%s""")' % (self.table, self.where)
        self.wholequery = "[%s for x in %s]" % (self.whatexpression, self.wheresentence)

        print("wheresentence", self.wheresentence)
        print("wholequery", self.wholequery)

    def ask(self):
        self.setSentence()
        return eval(self.wholequery)[0]


FOLDER_IMAGE_PATHNAME = "/home/micha/LaueProjects/CuVia/Carto"
FOLDER_PEAKLIST_PATHNAME = "/home/micha/LaueProjects/CuVia/Carto"
FOLDER_PEAKLISTCOR_PATHNAME = "/home/micha/LaueProjects/CuVia/Carto"
PREFIXFILENAME_IMAGE = "TSVCU_"
PREFIXFILENAME_PEAKLIST = "TSVCU_"
PREFIXFILENAME_PEAKLISTCOR = "TSVCU_"
IMAGE_EXTENSION = "mccd"
PEAKLIST_EXTENSION = "dat"
PEAKLISTCOR_EXTENSION = "cor"


class TableMap:
    """
    general class to query and modify hdf5 file tables
    containing data from X-ray Laue Map
    """

    def __init__(self):
        self.hdf = None
        self.tableUB = None
        self.tableMR = None
        self.tableallspots = None
        self.folder_image = None
        self.folder_peaklist = None
        self.folder_peaklistcor = None

    def hdfclose(self):
        self.hdf.close()

    def setTables(self):
        if self.tableUBNode is not None:
            self.tableUB = TabletoQuery(self.tableUBNode, "tableUB")
        if self.tableMRNode is not None:
            self.tableMR = TabletoQuery(self.tableMRNode, "tableMR")
        if self.tableallspotsNode is not None:
            self.tableallspots = TabletoQuery(self.tableallspotsNode, "tableallspots")

    def setfolders(self, folder_image, folder_peaklist, folder_peaklistcor):
        self.folder_image = folder_image
        self.folder_peaklist = folder_peaklist
        self.folder_peaklistcor = folder_peaklistcor

    def setLinkedFilesProperties(self):
        """
        set links with file in different folders and the way to read them
        """
        self.setfolders(FOLDER_IMAGE_PATHNAME, FOLDER_PEAKLIST_PATHNAME, FOLDER_PEAKLISTCOR_PATHNAME)
        self.prefixfilename_image = PREFIXFILENAME_IMAGE
        self.prefixfilename_peaklist = PREFIXFILENAME_PEAKLIST
        self.prefixfilename_peaklistcor = PREFIXFILENAME_PEAKLISTCOR

        self.image_extension = IMAGE_EXTENSION
        self.peaklist_extension = PEAKLIST_EXTENSION
        self.peaklistcor_extension = PEAKLISTCOR_EXTENSION

    def readfile(self, filename, modify=1):
        """ read hdf5 file and look for Laue analysis data """
        self.hdf = ReadMapFile(filename, modify=0)

        print('self.hdf %s'%self.hdf)

        # self.tableallspotsNode = "/Allspots/total_spots"
        # self.tableUBNode = "/Indexation/UB_matrices"
        # self.tableMRNode = "/Indexation/matching_rate"

        # self.tableallspotsNode = self.hdf["/Allspots/total_spots"]
        # self.tableUBNode = self.hdf["/Indexation/UB_matrices"]
        # self.tableMRNode = self.hdf["/Indexation/matching_rate"]

        try:
            self.tableallspotsNode = self.hdf.get_node("/Allspots/total_spots")
        except Tab.NoSuchNodeError:
            print("Missing Node containing spots Data")
            self.tableallspotsNode = None

        try:
            self.tableUBNode = self.hdf.get_node("/Indexation/UB_matrices")
        except Tab.NoSuchNodeError:
            print("Missing Node containing UB matrix data")
            self.tableUBNode = None
        try:
            self.tableMRNode = self.hdf.get_node("/Indexation/matching_rate")
        except Tab.NoSuchNodeError:
            print("Missing Node containing spots Indexation Quality")
            self.tableMRNode = None
        self.setTables()

        self.field_UB = None
        self.field_MR = None
        self.field_Spots = None
        if self.tableUBNode is not None:
            self.field_UB = self.tableUBNode.colnames
        if self.tableMRNode is not None:
            self.field_MR = self.tableMRNode.colnames
        if self.tableallspotsNode is not None:
            self.field_Spots = self.tableallspotsNode.colnames

        self.nb_images = self.tableUBNode.shape[0]

        print("nb of images", self.nb_images)

        print("reset manually nb of images to 1681 !!!")
        self.nb_images =500
        # TODO to be more general
        self.starting_index = min(self.tableMRNode.read(field="fileindex"))
        self.final_index = max(self.tableMRNode.read(field="fileindex"))

    # --- ------------  Query ------------------
    def peak_location(self, XY, radius=5.0, otherinfo=None):
        """
        return fileindex and intensity of peaks whose coordinates are close
        to a single spot XY within radius
        """
        X, Y = XY
        sentence = "(pixX-%.5f)**2+(pixY-%.5f)**2<%.1f**2"

        what = ["fileindex", "intensity"]
        if otherinfo != None:
            what = filterlist(what, otherinfo, self.tableallspots.fields)

        return self.tableallspots.ask(sentence % (X, Y, radius), what)

    def grainpeaks(self, fileindex, grainindex, otherinfo=None):
        """
        return x,y,I of peaks belonging to one grain of a given image
        """
        sentence = """(fileindex == %d) & (grainindex == %d)"""

        what = ["pixX", "pixY", "intensity"]
        if otherinfo != None:
            what = filterlist(what, otherinfo, self.tableallspots.fields)

        return self.tableallspots.ask(sentence % (fileindex, grainindex), what)

    def infospot(self, fileindex, spotindex, otherinfo=None):
        """
        return x,y,I,spotindex of one peak in a given image
        """
        sentence = """(fileindex == %d) & (spotindex == %d)"""

        what = ["pixX", "pixY", "intensity", "spotindex"]

        if otherinfo != None:
            what = filterlist(what, otherinfo, self.tableallspots.fields)

        return self.tableallspots.ask(sentence % (fileindex, spotindex), what)

    def infospots_grain(self, fileindex, grainindex, spotranks=0, otherinfo=None):
        """
        return x,y,I of one spot  in a given image belonging to one grain
        and at a given rank (in spot set of grain)

        spotranks :   integer
                    string to define a slice '2:8:2','::2' '0:-2'
        """

        sentence = """(fileindex == %d) & (grainindex == %d)"""

        what = ["pixX", "pixY", "intensity", "spotindex"]
        if otherinfo != None:
            what = filterlist(what, otherinfo, self.tableallspots.fields)

        allspots_data = self.tableallspots.ask(sentence % (fileindex, grainindex), what)

        if isinstance(spotranks, int):
            starti, finali, stepi = spotranks, spotranks + 1, 1
        elif isinstance(spotranks, str):
            starti, finali, stepi = readslicestring(spotranks, len(allspots_data))
        else:
            starti, finali, stepi = 0, 1, 1

        print(starti, finali, stepi)
        return allspots_data[starti:finali:stepi]

    def infospots_image(self, fileindex, spotranks=0, otherinfo=None):
        """
        return x,y,I of one spot or a slice of spots  in a given image
        and at a given rank (in spot set of grain)

        spotranks :   integer
                    string to define a slice '2:8:2','::2' '0:-2'
        """

        sentence = """(fileindex == %d)"""

        what = ["pixX", "pixY", "intensity", "spotindex", "grainindex"]
        if otherinfo != None:
            what = filterlist(what, otherinfo, self.tableallspots.fields)

        allspots_data = self.tableallspots.ask(sentence % (fileindex), what)

        if isinstance(spotranks, int):
            starti, finali, stepi = spotranks, spotranks + 1, 1
        elif isinstance(spotranks, str):
            starti, finali, stepi = readslicestring(spotranks, len(allspots_data))
        else:
            starti, finali, stepi = 0, 1, 1

        print(starti, finali, stepi)
        return allspots_data[starti:finali:stepi]

    def twopeaks_location(self, peak1, peak2, radius1=5.0, radius2=5.0, getintensity=0):
        """ return file index (as float) and intensity of first peak
        where two peaks exist (found by peak search) in .cor file
        WARNING: first column type is float to express integer...
        """
        tab1 = np.array(self.peak_location(peak1, radius=radius1))
        tab2 = np.array(self.peak_location(peak2, radius=radius2))

        set1 = set()
        set2 = set()
        if tab1 != []:
            set1 = set(tab1[:, 0])
        if tab2 != []:
            set2 = set(tab2[:, 0])

        fileindex_sorted = np.sort(list(set.intersection(set1, set2)))
        #        print "fileindex_sorted", fileindex_sorted
        #        print "fileindex_sorted", type(fileindex_sorted)

        if len(fileindex_sorted) == 0:
            return []

        target_fileindex = []
        if tab1 != []:
            target_fileindex = tab1[:, 0]
        elif tab2 != []:
            target_fileindex = tab2[:, 0]

        # TODO: a bit long !
        if getintensity:
            # to get intensity of most intense peak
            aa, bb, cc = GT.find_closest(fileindex_sorted, target_fileindex, tol=0.0001)

            return tab1[bb]
        else:
            return np.array([fileindex_sorted, np.ones(len(fileindex_sorted))]).T

    def npeaks_location(self, peaklist, radius1=5.0, radius2=5.0):
        """ return fileindex (as float) and one
        where ALL the peaks (in peaklist) exist
        (found by peak search) in any peaklist in dataset

        WARNING: first column type is float to express integer...

        TODO: better with itertools() to test during the loop ?
        """
        tab = []
        #        setlist = []
        target_fileindex_list = []

        XYpeaklist = peaklist[:, :2]
        for k, peak in enumerate(XYpeaklist):
            print("peak", peak)
            tab_peak = np.array(self.peak_location(peak, radius=radius1))
            print("tab_peak", tab_peak)
            tab.append(tab_peak)
            if tab_peak != []:
                currentset = set(tab_peak[:, 0])
                #                setlist.append(currentset)
                target_fileindex_list.append(tab_peak[:, 0])
                if k == 0:
                    setinter = currentset
                else:
                    setinter = set.intersection(setinter, currentset)
            else:
                #                setlist.append(set())
                target_fileindex_list.append([])

        fileindex_sorted = sorted(list(setinter))
        #        print "setinter", fileindex_sorted

        if len(fileindex_sorted) == 0:
            return []

        return np.array([fileindex_sorted, np.ones(len(fileindex_sorted))]).T

    #    def npeaks_location_kernel(self, peaklist, radius=5.):
    #        """
    #        wrong !
    #        """
    #        XYpeaklist = peaklist[:, :2]
    #        sentence = ''
    #        for peak in XYpeaklist:
    #            X, Y = peak
    #            sentence += "((pixX-%.5f)**2+(pixY-%.5f)**2<%.1f**2)&" % (X, Y, radius)
    #
    #        print "hiuhiu", sentence[:-1]
    #
    #        return self.tableallspots.ask(sentence[:-1], ['fileindex', 'intensity'])

    def grainlocation(self, fileindex, grainindex):
        """find where two highest peaks of a grain in a given image are in the dataset
        """
        allspots = self.grainpeaks(fileindex, grainindex)
        if allspots == []:
            return []

        allspots = np.array(allspots)
        peak1, peak2 = allspots[:2, :2]

        return self.twopeaks_location(peak1, peak2, radius1=5.0, radius2=5.0)

    def grainlocation_n(self, fileindex, grainindex, n=6, radius1=5.0, radius2=5.0):
        """
        find where n highest peaks of a grain in a given image
        are in the dataset
        
        return array with elements:  [fileindex, 1]
        """
        allspots = self.grainpeaks(fileindex, grainindex)
        if allspots == []:
            return []

        allspots = np.array(allspots)
        peaklist = allspots[:n, :n]

        print("peaklist", peaklist)

        return self.npeaks_location(peaklist, radius1=radius1, radius2=radius2)

    def getMatchingRate(self, fileindex, grainindex):
        """ return matching rate of one indexed grain in one image
        """
        sentence = """fileindex == %d"""

        return self.tableMR.ask(sentence % fileindex, ["MatchingRate_%d" % grainindex])[0]

    def getGrainMatchingRate(self, fileindex, grainindex):
        """ return matching rate of one indexed grain in one image
        """
        sentence = """fileindex == %d"""

        queryanswer = self.tableMRNode.read_where(sentence % fileindex)

        #         print 'queryanswer',queryanswer
        if len(queryanswer) > 0:
            grainsMR = list(queryanswer[0])[:3]

            # TODO check if the order is the same as tableMRNode.colnames
            #             print ' grainsMR[grainindex]', grainsMR[grainindex]
            return grainsMR[grainindex]
        else:
            return 0

    def getMatchingRates(self, fileindex):
        """ return matching rates of grains in one image
        """
        sentence = """fileindex == %d"""

        queryanswer = self.tableMRNode.read_where(sentence % fileindex)
        if len(queryanswer) > 0:
            tupleelems = list(queryanswer[0])

            # TODO check if the order is the same as tableMRNode.colnames
            return np.array(tupleelems[:3])
        else:
            return np.zeros(3)

    def getGrainmeanMatchingRate(self, fileindex):
        """
        mean MR over grains in one image
        """
        return np.mean(self.getMatchingRates(fileindex))

    def getMapMatchingRate(self, meanMR=0, mapshape=(101, 16)):
        """ give array of matching rate in map
        """
        MRarray = -1.0 * np.ones(mapshape[0] * mapshape[1])
        for k in range(self.nb_images):
            curMR = self.getGrainMatchingRate(k + self.starting_index, 0)
            if curMR != -1.0:
                MRarray[k] = curMR
        return MRarray.reshape(mapshape)

    def getGrainNB(self, fileindex, grainindex):
        """ return matching rate of one indexed grain in one image
        """
        sentence = """fileindex == %d"""

        queryanswer = self.tableMRNode.read_where(sentence % fileindex)
        if len(queryanswer) > 0:
            grainsNB = list(queryanswer[0])[3:6]
            # TODO check if the order is the same as tableMRNode.colnames
            return grainsNB[grainindex]
        else:
            return 0

    def getMapNB(self, sumNB=0, mapshape=(101, 16)):
        """ give array of number of indexed spots in map
        """
        NBarray = -1.0 * np.ones(mapshape[0] * mapshape[1])
        for k in range(self.nb_images):
            curNB = self.getGrainNB(k + self.starting_index, 0)
            if curNB != -1.0:
                NBarray[k] = curNB
        return NBarray.reshape(mapshape)

    def getUBs(self, fileindex):

        if self.tableUBNode.colnames != ["UB11_0", "UB11_1", "UB11_2", "UB12_0", "UB12_1", "UB12_2",
        "UB13_0", "UB13_1", "UB13_2", "UB21_0", "UB21_1", "UB21_2", "UB22_0", "UB22_1", "UB22_2",
            "UB23_0", "UB23_1", "UB23_2", "UB31_0", "UB31_1", "UB31_2", "UB32_0", "UB32_1", "UB32_2", "UB33_0",
            "UB33_1", "UB33_2", "fileindex"]:
            raise TypeError("columns of tableUB are not in the correct order")

        sentence = """fileindex == %d"""

        queryanswer = self.tableUBNode.read_where(sentence % fileindex)
        #         print fileindex, queryanswer
        if len(queryanswer) > 0:
            tupleelems = list(queryanswer[0])[:-1]
            threematrices = (np.array([tupleelems]).reshape((3, 3, 3)).transpose((2, 0, 1)))
            print("threematrices", threematrices)
            return threematrices
        else:
            return np.zeros((3, 3, 3))

    def getUB(self, fileindex, grainindex):
        return self.getUBs(fileindex)[grainindex]

    def getMapUB(self, mapshape=(101, 16), convertionmethod=1):
        """ give 1D array of scalar representative of UB matrix of first grain
        TODO only set for grain index 0
        """
        grainindex = 0

        projectionaxis = np.array([0, 0, 1])
        nprojeaxis = np.sqrt(np.sum(projectionaxis ** 2))
        qvector = np.array([1, 0, 0])

        UBarray = np.zeros((mapshape[0] * mapshape[1], 3))
        for k in range(self.nb_images):
            fileindex = k + self.starting_index
            Nbindexed = self.getGrainNB(fileindex, grainindex)
            # TODO should be > not >=
            if Nbindexed >= 0.0:
                UB = np.array(self.getUB(fileindex, grainindex))
                if convertionmethod == 0:
                    UBarray[k] = ORI.myRGB_3(UB)
                else:
                    print("UB", UB)
                    qv = np.dot(UB, qvector)
                    print("qv", qv)
                    nqv = np.sqrt(np.sum(qv ** 2))
                    print("nqv", nqv)
                    UBarray[k][0] = np.dot(qv, projectionaxis) / nqv / nprojeaxis

                    print("UBarray", UBarray[k][0])

        # return UBarray.reshape((mapshape[0], mapshape[1], 3))
        # patch grain 0
        return UBarray.reshape((mapshape[0], mapshape[1], 3))

    def Where_peak_is_indexed(self,
                                fileindex,
                                spotindex,
                                grainindex=-1,
                                spotrank_in_grain=0,
                                otherinfo=None,
                                radius=2.0):
        """ return  fileindex grainindex spotindex h,k,l of one peak
        given fileindex grainindex
        or given by fileindex spotindex
        """
        if spotindex == -1:
            # find X,Y spot by grainindex and corresponding spotrank_in_grain
            res = self.infospots_grain(fileindex, grainindex, spotranks=spotrank_in_grain)[0]
        else:
            # find X,Y spot by spotindex
            res = self.infospot(fileindex, spotindex)[0]

        print("res", res)
        Xres, Yres = res[:2]

        sentence = """((pixX-%.5f)**2+(pixY-%.5f)**2<%.1f**2) & (grainindex >=0)"""

        what = ["fileindex",
            "spotindex",
            "grainindex",
            "pixX",
            "pixY",
            "intensity",
            "H",
            "K",
            "L",
            "energy"]
        if otherinfo != None:
            for info in otherinfo:
                if info in self.tableallspots.fields:
                    what += [info]
                else:
                    raise ValueError("tableallspots.fields : %s does not contain this requested field: %s "
                        % (self.tableallspots.fields, info))

        allspots_data = self.tableallspots.ask(sentence % (Xres, Yres, radius), what)

        print("allspots_data")
        print(allspots_data)

        return allspots_data

    # --- ---------------  Add and modify rows
    def testadd(self):
        dataMR = [58.6, 22.3, 0.0029, 9999]
        self.tableMR.Add_Row(dataMR)

    def testadds(self):
        dataMR = [[58.6, 22.3, 0.0029, 15000],
                    [0.01, 258, 129, 9998],
                    [18.8, 3.14159, 0.29, 9997]]
        self.tableMR.Add_Rows(dataMR)

    def deleteIndexedGrain(self, fileindex, grainindex):
        """
        - unindex spots belonging to fileindex grainindex  in tableallspots
        - reset corresponding matching rate and UB in respectively tableMR and tableUB
        """
        mrs = self.getMatchingRates(fileindex)
        self.tableMR.deleteIndexedGrain(fileindex, grainindex)
        self.tableUB.deleteIndexedGrain(fileindex, grainindex)
        self.tableallspots.deleteIndexedGrain(fileindex, grainindex)

    def deleteAllIndexedGrains(self, fileindex):
        self.tableMR.deleteAllIndexedGrains(fileindex)
        self.tableUB.deleteAllIndexedGrains(fileindex)
        self.tableallspots.deleteAllIndexedGrains(fileindex)

    # --- ------  Read or modify images and peaklist
    def setPeaksearchnFitParams(self):
        self.CCDlabel = "PRINCETON"
        self.framedim = (2048, 2048)
        self.offset = 4096
        self.format = "uint16"
        self.fliprot = "no"

        self.paramsHat = (4, 5, 2)
        self.position_definition = 1  # LT and XMAS compatible
        self.fit_peaks_gaussian = 1

        self.dict_param = dict(PixelNearRadius=15,
                                Thresconvolve=30000,
                                IntensityThreshold=200,
                                boxsize=15,
                                xtol=0.0001,
                                FitPixelDev=2)

        self.kwds_fitpeaks = dict(baseline="auto",
                                    startangles=0.0,
                                    start_sigma1=1.0,
                                    start_sigma2=1.0,
                                    position_start="center",
                                    fitfunc="gaussian",
                                    showfitresults=0,
                                    offsetposition=1,
                                    verbose=0,
                                    xtol=0.00000001,
                                    framedim=(2048, 2048),
                                    offset=4096,
                                    formatdata="uint16",
                                    fliprot="no")

        self.kwds_peaksearch = dict(CCDLabel=self.CCDlabel,
                                    PixelNearRadius=self.dict_param["PixelNearRadius"],
                                    removeedge=2,
                                    IntensityThreshold=self.dict_param["IntensityThreshold"],
                                    local_maxima_search_method=2,
                                    thresholdConvolve=self.dict_param["Thresconvolve"],
                                    boxsize=self.dict_param["boxsize"],
                                    paramsHat=self.paramsHat,
                                    position_definition=self.position_definition,
                                    verbose=0,
                                    fit_peaks_gaussian=self.fit_peaks_gaussian,
                                    xtol=self.dict_param["xtol"],
                                    FitPixelDev=self.dict_param["FitPixelDev"],
                                    return_histo=0,
                                    Saturation_value=DictLT.dict_CCD[self.CCDlabel][2],
                                    Saturation_value_flatpeak=DictLT.dict_CCD[self.CCDlabel][2],
                                    write_execution_time=1)

    #
    #        return RMCCD.readoneimage_multiROIfit(filename,
    #                             centers,
    #                             boxsize,
    #                             **self.kwds_fitpeaks)

    def getImageFilename(self, fileindex):
        return os.path.join(self.folder_image,
            self.prefixfilename_image
            + "%04d.%s" % (int(fileindex), self.image_extension))

    def getPeakListFilename(self, fileindex):
        return os.path.join(self.folder_peaklist,
            self.prefixfilename_image
            + "%04d.%s" % (int(fileindex), self.peaklist_extension))

    def getPeakListCorFilename(self, fileindex):
        return os.path.join(self.folder_peaklistcor,
            self.prefixfilename_image
            + "%04d.%s" % (int(fileindex), self.peaklistcor_extension))

    def searchPeaks_oneImage(self, fileindex, **kwargs):
        """
        starts peak search on one image
        return peak list where columns are

        [peak_X, peak_Y,peak_I, peak_fwaxmaj, peak_fwaxmin,
        peak_inclination, Xdev, Ydev, peak_bkg, Ipixmax]

        """
        filename = self.getImageFilename(fileindex)

        self.kwds_peaksearch.update(kwargs)
        print(self.kwds_peaksearch)

        tabpeak = RMCCD.PeakSearch(filename, **self.kwds_peaksearch)[0]

        return tabpeak

    def searchlocalpeaks(self, fileindex, center, boxsize, plot=0, **kwargs):
        """
        starts peak search on one ROI of image
        return peak list where columns are

        [peak_X, peak_Y,peak_I, peak_fwaxmaj, peak_fwaxmin,
        peak_inclination, Xdev, Ydev, peak_bkg, Ipixmax]
        """
        filename = self.getImageFilename(fileindex)

        self.kwds_peaksearch["boxsize"] = boxsize
        self.kwds_peaksearch["center"] = center
        self.kwds_peaksearch.update(kwargs)
        print(self.kwds_peaksearch)
        tabpeak = RMCCD.PeakSearch(filename, **self.kwds_peaksearch)[0]
        if plot:
            XY = tabpeak[:, :2]
            ImProc.plot_image_markers(self.getFullImageData(fileindex), XY)

        return tabpeak

    def getMinMax(self, fileindex, center, boxsize):
        data2d = self.getFullImageData(fileindex)

        # [mini, maxi]
        return ImProc.getMinMax(data2d, center, boxsize, self.framedim)

    def fitpeak_at_centers(self, fileindex, centers, boxsize, plot=0, **kwargs):
        """
        fit a single peak at multiple positions given by centers
        """
        filename = self.getImageFilename(fileindex)
        self.kwds_fitpeaks.update(kwargs)

        centers = np.array(centers)
        # TODO: optimize this step by multiple slices of extrema arg?
        tab_minmax = np.zeros((len(centers), 2))

        for k, cen in enumerate(centers):
            tab_minmax[k] = self.getMinMax(fileindex, cen, boxsize)

        Ipixmax = tab_minmax[:, 1]

        print(self.kwds_fitpeaks)

        params = RMCCD.readoneimage_multiROIfit(filename, centers, boxsize,
                                    **self.kwds_fitpeaks)[0]

        par = np.array(params)

        if par == []:
            print("no fitted peaks")
            return

        peak_bkg = par[:, 0]
        peak_I = par[:, 1]
        peak_X = par[:, 2]
        peak_Y = par[:, 3]
        peak_fwaxmaj = par[:, 4]
        peak_fwaxmin = par[:, 5]
        peak_inclination = par[:, 6] % 360

        # pixel deviations from guessed initial position before fitting
        Xdev = peak_X - centers[:, 0]
        Ydev = peak_Y - centers[:, 1]
        #    print 'Xdev', Xdev
        #    print "Ydev", Ydev

        # all peaks list building
        tabpeak = np.array([peak_X,
                            peak_Y,
                            peak_I,
                            peak_fwaxmaj,
                            peak_fwaxmin,
                            peak_inclination,
                            Xdev,
                            Ydev,
                            peak_bkg,
                            Ipixmax]).T

        if plot:
            ImProc.plot_image_markers(self.getFullImageData(fileindex), tabpeak)

        return tabpeak

    def write_PeakListFile(self, fileindex, tabpeaks):
        filename = self.getPeakListFilename(fileindex)
        imagefilename = self.getImageFilename(fileindex)

        comments = "New file from TableMap class"

        prefixname = filename.split(".")[0]

        RMCCD.writepeaklist(tabpeaks,
                            prefixname,
                            outputfolder=None,
                            comments=comments,
                            initialfilename=imagefilename)

    def getTabPeaksList(self, fileindex):
        filename = self.getPeakListFilename(fileindex)
        return IOLT.read_Peaklist(filename, dirname=None)

    def addPeaks_in_peaklistfile(self, fileindex, tabpeaks, updatePeaks=0, dist_tolerance=1.0):
        """
        add peaks from tabpeak in peaklist file

        updatePeaks        : 0  add peaks without checking if they already exist
                            1 add only new peaks (if they don't already exist within dist_tolerance) 
        """
        data_current_peaks = self.getTabPeaksList(fileindex)
        filename_in = self.getPeakListFilename(fileindex)

        array_pts = data_current_peaks[:, :2]

        if not updatePeaks:
            IOLT.addPeaks_in_Peaklist(filename_in,
                                        tabpeaks,
                                        filename_out="test",
                                        dirname_in=None,
                                        dirname_out=self.folder_peaklist)
            return

        resetspotindex = []
        newpeaks = []
        for peak in tabpeaks:
            XY = peak[:2]
            posclose, dist = GT.FindClosestPoint(array_pts, XY, returndist=1)
            #            print "dist", dist
            #            print "posclose", posclose
            if dist[posclose] <= dist_tolerance:
                resetspotindex.append(posclose)
                # peak already exists, reset to new input values
                #                data_current_peaks[posclose] = np.zeros(11)  # to test
                data_current_peaks[posclose] = peak
            else:
                # definitively a new peak to be added
                newpeaks.append(peak)

        print("data_current_peaks.shape", data_current_peaks.shape)
        print(np.array(newpeaks).shape)

        if len(newpeaks) > 0:
            # merge:
            data_current_peaks = np.concatenate((data_current_peaks, np.array(newpeaks)), axis=0)

        # sort
        data_current_peaks = data_current_peaks[np.argsort(data_current_peaks[:, 3])[::-1]]

        self.write_PeakListFile(1700, data_current_peaks)

    def getFullImageData(self, fileindex):
        filename = self.getImageFilename(fileindex)

        data_1d = RMCCD.readoneimage(filename, framedim=self.framedim, offset=self.offset,
                                                                            formatdata=self.format)
        return np.reshape(data_1d, self.framedim)

    # --- ------   Plot spots ------


POS_TO_RESET = [
    0,
    1,
    2,  # HKL
    3,
    4,  # MR NB
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,  # UB
    18,
    19,
    20,
    21,
    22,
    23,  # devstrain
    24,  # energy
    26,  # graindex
]

DEFAULT_DICT_NONINDEXEDSPOTS_VALUES = {"H": 10000,
                                        "K": 10000,
                                        "L": 10000,
                                        "MatchingRate": -1,
                                        "Nb_indexedspots": 0.0,
                                        "Matrix_elements": 0.0,
                                        "Strain_elements": 0.0,
                                        "Energy": -1,
                                        "grainindex": -1,
                                        "PixDev": -1,
                                        "PixDev_x": -1,
                                        "PixDev_y": -1}

DEFAULT_TO_RESET = ([DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["H"],
        DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["K"],
        DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["L"],
        DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["MatchingRate"],
        DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Nb_indexedspots"],
    ]
    + [DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Matrix_elements"]] * 9
    + [DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Strain_elements"]] * 6
    + [
        DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["Energy"],
        DEFAULT_DICT_NONINDEXEDSPOTS_VALUES["grainindex"]])


class TabletoQuery:
    """
    class to handle query of table from an hdf5 file
    """

    def __init__(self, tableofdata, tabletypename):
        self.table = tableofdata
        self.fields = tableofdata.colnames
        nb_cols = len(self.fields)
        nb_rows = tableofdata.nrows
        # nb of rows or 'data'
        self.nbdata = nb_rows
        self.tabletypename = tabletypename

    #        self.parenttable = parent.tableofdata

    def ask(self, conditions, returnwhat):
        for elem in self.fields:
            if elem not in self.fields:
                raise ValueError("%s does not contain the column %s" % (self.table, elem))

        return Query(self.table, conditions, returnwhat)

    def Add_Row(self, data):
        """
        data  : data in the same order as self.fields
        """
        if len(data) != len(self.fields):
            raise ValueError("There are missing data to add a row in the hdf5 file")
            return
        for key, val in zip(self.fields, data):
            self.table.row[key] = val
        self.table.row.append()
        self.table.flush()

    def Add_Rows(self, datalist):
        """
        data  : data elements in the same order as self.fields

        could be possible like this
        table.append([("Particle:     10", 10, 0, 10*10, 10**2),
                ("Particle:      11", 11, -1, 11*11, 11**2),
                ("Particle:    12", 12, -2, 12*12, 12**2)])

        TODO: better like this:
            row = table.row
            for i in xrange(nrows):
                row['col1'] = i-1
                row['col2'] = 'a'
                row['col3'] = -1.0
                row.append()
            table.flush()
        """
        for data in datalist:
            for key, val in zip(self.fields, data):
                self.table.row[key] = val
            self.table.row.append()

        self.table.flush()

    def Modify_Row(self, data, rowindex):
        """
        Modify data in a given row

        rowindex can be found by self.table.getWhereList(condition)
        data  : data elements in the same order as self.fields

        TODO: use better update?
         for row in table.where('col1 > 3'):
             row['col1'] = row.nrow
             row['col2'] = 'b'
             row['col3'] = 0.0
             row.update()
         table.flush()
        """
        self.table[rowindex] = data

    def Modify_Rows(self, data, rowindexstart, rowindexfinal):
        """
        Modify data in a given row

        rowindex can be found by self.table.getWhereList(condition)
        data  : data elements in the same order as self.fields
        """
        self.table[rowindexstart : rowindexfinal + 1] = data

    def deleteIndexedGrain(self, fileindex, grainindex):
        """
        reset value of fields depending on indexation in different tables

        TODO: use better update?
        for row in table.where('col1 > 3'):
            row['col1'] = row.nrow
            row['col2'] = 'b'
            row['col3'] = 0.0
            row.update()
        table.flush()
        """
        if self.tabletypename == "tableMR":
            row = self.table.getWhereList("fileindex == %d" % fileindex)
            if len(row) == 1:
                print("current row in tableMR", row[0])
                datamr = list(self.table.read(row[0])[0])
                datamr[grainindex] = -1.0
                self.table[row[0]] = datamr

        if self.tabletypename == "tableUB":
            row = self.table.getWhereList("fileindex == %d" % fileindex)
            if len(row) == 1:
                print("current row in tableUB", row[0])
                datamr = list(self.table.read(row[0])[0])
                for k in range(9):
                    datamr[grainindex + 3 * k] = 0.0
                self.table[row[0]] = datamr

        if self.tabletypename == "tableallspots":
            print("ujniuiunin")
            rows = self.table.getWhereList("fileindex == %d" % fileindex)
            #            print rows
            nbspots = len(rows)
            for rowindex in rows:
                #                print "currentrow in tableallspots", rowindex
                datamr = list(self.table.read(rowindex)[0])
                print(datamr)
                for pos, val in zip(POS_TO_RESET, DEFAULT_TO_RESET):
                    datamr[pos] = val
                self.table[rowindex] = datamr

    def deleteAllIndexedGrains(self, fileindex):
        """
        reset value of fields depending on indexation in different tables

        TODO: use better update?
        for row in table.where('col1 > 3'):
            row['col1'] = row.nrow
            row['col2'] = 'b'
            row['col3'] = 0.0
            row.update()
        table.flush()
        """
        if self.tabletypename == "tableMR":
            row = self.table.getWhereList("fileindex == %d" % fileindex)
            if len(row) == 1:
                print("currentrow", row[0])
                datamr = list(self.table.read(row[0])[0])
                datamr = [-1.0] * (len(datamr) - 1) + [datamr[-1]]
                self.table[row[0]] = datamr

        if self.tabletypename == "tableUB":
            row = self.table.getWhereList("fileindex == %d" % fileindex)
            if len(row) == 1:
                print("currentrow in UB", row[0])
                datamr = list(self.table.read(row[0])[0])
                datamr = [0.0] * (len(datamr) - 1) + [datamr[-1]]
                self.table[row[0]] = datamr

        if self.tabletypename == "tableallspots":
            for grainindex in range(self.nbgrains):
                self.deleteIndexedGrain(fileindex, grainindex)


# --- ----  Methods to query table in  hdf5 format
def query_peak_location(tableallspots, peak, radius=5.0):
    """ return file index and intensity where peak exists
    (found by peak search) in .cor file
    """
    query_sentence = """(pixX-%.5f)**2+(pixY-%.5f)**2<%.1f**2"""
    return [[x["fileindex"], x["intensity"]]
        for x in tableallspots.where(query_sentence % (peak[0], peak[1], radius))]


def query_twopeaks_location(tableallspots, peak1, peak2, radius1=5.0, radius2=5.0):
    """ return file index (as float) and intensity of first peak 
    where two peaks exist (found by peak search) in .cor file
    WARNING: first column type is float to express integer...
    """
    tab1 = np.array(query_peak_location(tableallspots, peak1, radius=radius1))
    tab2 = np.array(query_peak_location(tableallspots, peak2, radius=radius2))

    set1 = set()
    set2 = set()
    if tab1 != []:
        set1 = set(tab1[:, 0])
    if tab2 != []:
        set2 = set(tab2[:, 0])

    fileindex_sorted = np.sort(list(set.intersection(set1, set2)))

    if len(fileindex_sorted) == 0:
        return []

    target_fileindex = []
    if tab1 != []:
        target_fileindex = tab1[:, 0]
    elif tab2 != []:
        target_fileindex = tab2[:, 0]

    aa, bb, cc = GT.find_closest(fileindex_sorted, target_fileindex, tol=0.0001)

    return tab1[bb]


def plotpresence(tabpresence, mapshape=(101, 16), starting_index=1708):
    """
    from tabpresence = array of [fileindex , intensity]
    """
    if tabpresence == []:
        return
    indexfile = np.array(tabpresence[:, 0], dtype=np.int) - starting_index
    value_intensity = tabpresence[:, 1]

    data = np.zeros(mapshape[0] * mapshape[1])

    np.put(data, indexfile, value_intensity)
    data = np.reshape(data, mapshape)
    pp.imshow(data, interpolation="nearest")
    pp.show()


def query_grainpeaks(tableallspots, fileindex, grainindex):
    """
    return x,y,I of peaks belonging to one grain of a given image
    """
    query_sentence = """(fileindex == %d) & (grainindex == %d)"""
    return [[x["pixX"], x["pixY"], x["intensity"]]
        for x in tableallspots.where(query_sentence % (fileindex, grainindex))]


def query_grainlocation(tableallspots, fileindex, grainindex):
    """find where two highest peaks of a grain
    in a given image are in the dataset
    """
    allspots = query_grainpeaks(tableallspots, fileindex, grainindex)
    if allspots == []:
        return []

    allspots = np.array(allspots)
    peak1, peak2 = allspots[:2, :2]

    return query_twopeaks_location(tableallspots, peak1, peak2, radius1=5.0, radius2=5.0)


def query_peaksMiller(tableallspots, peak, radius=5.0):
    """ return file index a and intensity and hkl for a peak
    """
    return [[x["fileindex"], x["intensity"], x["H"], x["K"], x["L"]]
        for x in tableallspots.where(
            """((pixX-%.5f)**2+(pixY-%.5f)**2<%.1f**2) & (grainindex>-1)"""
            % (peak[0], peak[1], radius))]


def query_Miller_mainpeak(tableallspots, fileindex, grainindex, spotindex, radius=5.0):
    allspots = query_grainpeaks(tableallspots, fileindex, grainindex)
    if allspots == []:
        return []
    allspots = np.array(allspots)

    if spotindex > len(allspots):
        print("spotindex is larger the number of spots found in the image")
        return

    peak1 = allspots[spotindex, :2]

    return query_peaksMiller(tableallspots, peak1, radius=radius)


def get_UBs(tableUB, fileindex, nbgrains=3):
    """
    give the UB matrices of 3 grains in one image
    """
    mat = [[[x[ub_element + "_%d" % k] for ub_element in list_ub_element]
            for k in range(nbgrains)]
        for x in tableUB.where("""fileindex == %d""" % fileindex)]

    mats = np.array(mat[0])

    return mats.reshape((3, 3, 3))


def get_UBgrain(tableUB, fileindex, grainindex):
    """
    give the UB matrix of one grain in one image
    """
    return get_UBs(tableUB, fileindex, nbgrains=3)[grainindex]


def get_UBspot(tableallspots, fileindex, spotindex):
    query_sentence = """(fileindex == %d) & (spotindex == %d)"""

    mat = np.array([
            [x[ub_element] for ub_element in list_ub_element]
            for x in tableallspots.where(query_sentence % (fileindex, spotindex))
        ])[0]

    return mat.reshape((3, 3))


def get_MRs(tableMR, fileindex, nbgrains=3):
    """
    give the Matching rates of 3 grains in one image
    """
    mat = [[x["MatchingRate_%d" % k] for k in range(nbgrains)]
        for x in tableMR.where("""fileindex == %d""" % fileindex)]

    MRs = mat[0]

    #    print mats
    #    print np.reshape(mats[0], (3, 3))
    #    print np.reshape(mats[1], (3, 3))
    #    print np.reshape(mats[2], (3, 3))
    return MRs


def get_MR(tableMR, fileindex, grainindex):
    """
    give the Matching rate of one grain in one image
    """
    return get_MRs(tableMR, fileindex, nbgrains=3)[grainindex]


def get_MRspot(tableallspots, fileindex, spotindex):
    query_sentence = """(fileindex == %d) & (spotindex == %d)"""

    MR = [x["MatchingRate"]
        for x in tableallspots.where(query_sentence % (fileindex, spotindex))][0]

    return MR


def get_DevStrainSpot(tableallspots, fileindex, spotindex):
    """
    give the 6 deviatoric strain elements of one spot of one image
    """
    query_sentence = """(fileindex == %d) & (spotindex == %d)"""

    devstrain_voigt = np.array([[x[strain_element] for strain_element in list_devstrain_element]
            for x in tableallspots.where(query_sentence % (fileindex, spotindex))])[0]

    return devstrain_voigt


def query(tableofdata, conditions, returnwhat):
    whatexpression = "x['%s']" % returnwhat
    if isinstance(returnwhat, list):
        if len(returnwhat) == 1:
            whatexpression = "x['%s']" % returnwhat[0]
        else:
            expr_xcol = "["
            for elem in returnwhat:
                expr_xcol += "x['%s']," % elem
            expr_xcol = expr_xcol[:-1] + "]"
            whatexpression = expr_xcol

    where_condition = conditions
    if isinstance(conditions, list):
        if len(conditions) == 1:
            where_condition = conditions[0]
        else:
            where_condition = ""
            for cond in conditions:
                where_condition += "(%s)&" % cond
            where_condition = where_condition[:-1]

    wheresentence = '%s.where("""%s""")' % (tableofdata, where_condition)
    wholequery = "[%s for x in %s]" % (whatexpression, wheresentence)

    print("wholequery", wholequery)
    Res = eval(wholequery)
    if len(Res) > 0:
        print("res. query", Res)

        return Res
    else:
        return []


def ReadMap(hfd5_filename):
    print("opening hdf5 file: %s" % hfd5_filename)

    hdf5file = Tab.openFile(hfd5_filename)

    print("opening data table")
    tableallspots = hdf5file.root.Allspots.total_spots
    tableUB = hdf5file.root.Indexation.UB_matrices
    tableMR = hdf5file.root.Indexation.matching_rate

    return tableUB, tableMR, tableallspots


def ReadMapFile(hfd5_filename, modify=0):
    print("opening hdf5 file: %s" % hfd5_filename)

    if modify:
        hdf5file = Tab.openFile(hfd5_filename, "a")
    else:
        hdf5file = Tab.openFile(hfd5_filename, 'r')

    # if modify:
    #     hdf5file = h5py.File(hfd5_filename, "a")
    # else:
    #     hdf5file = h5py.File(hfd5_filename)

    return hdf5file


def Query(tableofdata, conditions, returnwhat):
    whatexpression = "x['%s']" % returnwhat
    if isinstance(returnwhat, list):
        if len(returnwhat) == 1:
            whatexpression = "x['%s']" % returnwhat[0]
        else:
            expr_xcol = "["
            for elem in returnwhat:
                expr_xcol += "x['%s']," % elem
            expr_xcol = expr_xcol[:-1] + "]"
            whatexpression = expr_xcol

    where_condition = conditions
    if isinstance(conditions, list):
        if len(conditions) == 1:
            where_condition = conditions[0]
        else:
            where_condition = ""
            for cond in conditions:
                where_condition += "(%s)&" % cond
            where_condition = where_condition[:-1]

    #    print "where_condition", where_condition
    wheresentence = 'tableofdata.where("""%s""")' % where_condition

    #    print "wheresentence", wheresentence

    wholequery = "[ %s for x in %s]" % (whatexpression, wheresentence)

    #    print "wholequery", wholequery
    Res = eval(wholequery)

    return Res


#    if len(Res) > 0:
#        print "res. query", Res
#
#        return Res
#    else:
#        return []


def readslicestring(slicestring, lenobject):
    """
    return slice parameters: start, end, step  from string input

    lenobject  is the length of the object to be sliced
    """
    start, end, step = 0, 1, 1
    sl = slicestring.split(":")
    if len(sl) >= 2:
        if sl[0] == "":
            start = 0
        else:
            start = int(sl[0])
        if sl[1] == "":
            end = lenobject
        else:
            end = int(sl[1])
        if len(sl) == 3:
            if sl[2] == "":
                step = 1
            else:
                step = int(sl[2])
                if step <= 0:
                    raise ValueError("negative or null step is not handled")

    return start, end, step


def filterlist(initlist, list_of_strings_to_add, list_of_fields):
    """
    add elem of list_of_strings_to_add that are in list_of_fields in initlist

    and raise an error if elem is not in list_of_fields
    """
    for elem in list_of_strings_to_add:
        if elem in list_of_fields:
            initlist += [elem]
        else:
            raise ValueError("tableallspots.fields : %s does not contain this requested field: %s "
                % (list_of_fields, elem))
    return initlist


figindex = 0


def plotpeaklist(peaklist_XYI):
    """
    plot peaklist X,Y and intensity
    """
    global figindex
    X, Y, Intens = peaklist_XYI[:, :3].T

    nbpoints = len(X)

    fig = pp.figure(figindex)
    #    ax = fig.add_subplot(111)
    # all exp spots
    pp.scatter(X, Y, s=Intens / np.amax(Intens) * 100.0, c=Intens / 50.0,
                        marker="o",
                        faceted=True,
                        alpha=0.5)

    pp.title("nb points: %d" % nbpoints)

    #    self.axes.scatter(self.Data_X, self.Data_Y,
    #                          s=self.Data_I / np.amax(self.Data_I) * 100.,
    #                          c=self.Data_I / 50.)#, cmap = GT.SPECTRAL)
    figindex += 1
    pp.show()
