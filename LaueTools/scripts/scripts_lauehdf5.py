import pickle
import numpy as np
import tables as Tab

import Lauehdf5 as LTHDF5


# --- ------ Test  ----
def test_build_hdf5():
    """
    TSV Cu 3 grains per images
    """
    f = open("dictCu_3g_1708_3323")
    dicts = pickle.load(f)
    f.close()

    dictMat, dictMR, dictNB, dictstrain, dictspots = dicts

    keys_indexfile = sorted(dictMat.keys())

    nbgrains = 3
    nb_of_spots_per_image = 200  # estimation for optimization in data access

    nb_of_images = len(keys_indexfile)
    #    nb_of_images = 3

    filename = "dictCu_3g_expect.h5"
    imagefilename_prefix = "TSVCU_"
    # Open a file in "w"rite mode
    h5file = Tab.openFile(filename, mode="w", title="TSV Cu July 2011")

    keys_indexfile = keys_indexfile[:nb_of_images]
    # ----------------------------------------
    if 1:  # Class IndexedImage
        # Create a new group under "/" (root)
        group_images = h5file.createGroup("/", "Indexation", "Indexation figure")

        # Create one table on it
        table = h5file.createTable(
            group_images, "matching_rate", IndexedImage, "indexation rate"
        )
        # Fill the table with data
        indexedImage = table.row

        #    print "keys_indexfile", keys_indexfile
        for key_image in keys_indexfile:
            indexedImage["fileindex"] = key_image
            for grainindex in range(nbgrains):
                indexedImage["MatchingRate_%d" % grainindex] = dictMR[key_image][
                    grainindex
                ]  # 1 grain
                indexedImage["NBindexed_%d" % grainindex] = dictNB[key_image][
                    grainindex
                ]

            # Insert a new particle record
            indexedImage.append()

        table.flush()
        # -------------------------------------

    # ----------------------------------------
    if 1:  # Class Matrices
        #        # Create a new group under "/" (root)
        #        group_images = h5file.createGroup("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tableUB = h5file.createTable(
            group_images, "UB_matrices", Matrices, "UB Matrices elements"
        )
        # Fill the table with data
        indexedUB = tableUB.row

        for key_image in keys_indexfile:
            indexedUB["fileindex"] = key_image
            #            print "key_image", key_image
            for grainindex in range(nbgrains):
                UBmatrix = np.zeros(9)
                exp_mat = dictMat[key_image][grainindex]
                if not isinstance(exp_mat, int):
                    UBmatrix = np.ravel(exp_mat)
                for k, ub_element in enumerate(list_ub_element):
                    #                    print ub_element + '_%d' % grainindex
                    #                    print UBmatrix[k]

                    indexedUB[ub_element + "_%d" % grainindex] = UBmatrix[k]

            # Insert a new particle record
            indexedUB.append()

            # -------------------------------
        tableUB.flush()
        # -------------------------------------

    # test
    if 1:  # Class Matrices_array
        #        # Create a new group under "/" (root)
        #        group_images = h5file.createGroup("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tableUB_array = h5file.createTable(
            group_images, "UB_matrices_array", Matrices_array, "UB Matrices elements"
        )
        # Fill the table with data
        indexedUB_ar = tableUB_array.row

        for key_image in keys_indexfile:
            indexedUB_ar["fileindex"] = key_image
            #            print "key_image", key_image
            for grainindex in range(nbgrains):
                UBmatrix = np.zeros(9)
                exp_mat = dictMat[key_image][grainindex]
                if not isinstance(exp_mat, int):
                    indexedUB_ar["UB%d" % grainindex] = exp_mat

            # Insert a new particle record
            indexedUB_ar.append()

            # -------------------------------
        tableUB_array.flush()
        # -------------------------------------

    if 1:  # Class DevStrain
        #        # Create a new group under "/" (root)
        #        group_images = h5file.createGroup("/", 'Indexation', 'Indexation figure')

        # Create one table on it
        tabledev = h5file.createTable(
            group_images, "Devstrain_matrices", DevStrain, "UB Matrices elements"
        )
        # Fill the table with data
        indexedDev = tabledev.row

        for key_image in keys_indexfile:
            indexedDev["fileindex"] = key_image
            for grainindex in range(nbgrains):
                devstrain = np.zeros(6)
                exp_dev = dictstrain[key_image][grainindex]
                if not isinstance(exp_dev, int):
                    devstrainmatrix = np.take(np.ravel(exp_dev), pos_voigt)
                for k, strain_element in enumerate(list_devstrain_element):
                    indexedDev[strain_element + "_%d" % grainindex] = devstrainmatrix[k]

            # Insert a new particle record
            indexedDev.append()

            # -------------------------------
        tabledev.flush()
        # -------------------------------------

    if 1:  # class AllIndexedSpots

        group_spots = h5file.createGroup("/", "Allspots", "All spots information")

        tableallspots = h5file.createTable(
            group_spots,
            "total_spots",
            AllIndexedSpots,
            "Readout example",
            expectedrows=nb_of_images * nb_of_spots_per_image,
        )

        allIndexedSpots = tableallspots.row

        print("keys_indexfile", keys_indexfile)
        for key_image in keys_indexfile:
            print("key_image", key_image, "--------------------------/n/n")
            for elem in dictspots[key_image]:
                allIndexedSpots["fileindex"] = key_image
                grainindex = int(elem[1])

                (
                    allIndexedSpots["spotindex"],
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

                #                print "spot index", elem[0]
                #                print "grainindex", int(elem[1])
                if grainindex >= 0:
                    #                    print "grain is indexed!"
                    allIndexedSpots["MatchingRate"] = dictMR[key_image][grainindex]
                    allIndexedSpots["NbindexedSpots"] = dictNB[key_image][grainindex]
                    # UB matrix: in dictMat
                    UBmatrix = np.ravel(dictMat[key_image][grainindex])
                    # deviatoric strain matrix: in dictstrain
                    devstrainmatrix = np.take(
                        np.ravel(dictstrain[key_image][grainindex]), pos_voigt
                    )

                for k, ub_element in enumerate(list_ub_element):
                    allIndexedSpots[ub_element] = UBmatrix[k]

                for k, strain_element in enumerate(list_devstrain_element):
                    allIndexedSpots[strain_element] = devstrainmatrix[k]

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


if __name__ == "__main__":

    if 1:
        # add new rows
        Summary_HDF5_filename = "matiou.h5"
        LTHDF5.Add_allspotsSummary_from_fitfiles(
            Summary_HDF5_filename,
            "UO2_20_90_serree_",
            "/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
            list(range(100, 135)),
            Summary_HDF5_dirname="/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
        )

    if 0:

        Summary_HDF5_filename = "jason.h5"
        LTHDF5.Add_allspotsSummary_from_fitfiles(
            Summary_HDF5_filename,
            "UO2_20_90_serree_",
            "/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
            [137, 139, 140],
            Summary_HDF5_dirname="/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
        )

        # add new rows
        Summary_HDF5_filename = "jason.h5"
        LTHDF5.Add_allspotsSummary_from_fitfiles(
            Summary_HDF5_filename,
            "UO2_20_90_serree_",
            "/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
            list(range(100, 135)),
            Summary_HDF5_dirname="/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
        )

        # update some rows from other files
        print("\n remove data with file index 139 and 140 and replace by others")
        Summary_HDF5_filename = "jason.h5"
        LTHDF5.Add_allspotsSummary_from_fitfiles(
            Summary_HDF5_filename,
            "UO2_20_90_serree_new_",
            "/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
            [139, 140],
            Summary_HDF5_dirname="/home/micha/LaueProjects/UO2_raq1ter/fitfiles",
        )

    if 0:
        print("opening hdf5 file")
        tt = LTHDF5.TableMap()
        tt.readfile("dict_Res_B1ep6_carto2dbis.h5")

    #     print "opening data table"
    #     tableallspots = hdf5file.root.Allspots.total_spots
    #     tableUB = hdf5file.root.Indexation.UB_matrices
    #     tableMR = hdf5file.root.Indexation.matching_rate

    if 0:

        #     dicos = build_hdf5('B1ep6_carto2d_dict_0000_0005',
        #                dirname_dictRes='/home/micha/LaueProjects/Ni_joint/fitfiles')
        dicos = LTHDF5.build_hdf5(
            "B1ep6_carto2d_dict_0000_0081",
            dirname_dictRes="/home/micha/LaueProjects/Ni_joint/fitfiles",
        )

    if 0:
        # test to modify hdf5 file
        tt = LTHDF5.TableMap()
        tt.readfile("dictCu_3g_test.h5", modify=1)

        tt.tableallspots.ask(["UB11>=0", "fileindex==1708"], ["spotindex", "fileindex"])

        tt.setLinkedFilesProperties()

        tt.setPeaksearchnFitParams()

        tab1708 = tt.getTabPeaksList(1700)
        print("iojiojoi")

        mytab = tab1708[-8:-2]  # take 6 peaks among the last ones

        # intensity reset
        mytab[:, 3] = np.arange(700, 100, -100)

        # positions resets
        mytab[:3, :2] = -np.random.randint(10000, high=30000, size=(3, 2))

        tt.addPeaks_in_peaklistfile(1700, mytab, updatePeaks=1)

    #        tt.hdf.close()

    #    tt.tableallspots.ask(['spotindex==1', 'UB11>0.0'], 'spotindex')
    if 0:
        tt = LTHDF5.TableMap()
        tt.readfile("dictCu_3g_big.h5")

        tt.tableallspots.ask(["UB11>=0", "fileindex==1708"], ["spotindex", "fileindex"])

    #    tt.tableallspots.ask(['spotindex==1', 'UB11>0.0'], 'spotindex')

    if 0:
        #    build_hdf5()

        print("opening hdf5 file")
        #    hdf5file = Tab.openFile("dictCu_3g.h5")
        hdf5file = Tab.openFile("dictCu_3g_testshort.h5")

        print("opening data table")
        tableallspots = hdf5file.root.Allspots.total_spots
        tableUB = hdf5file.root.Indexation.UB_matrices
        tableMR = hdf5file.root.Indexation.matching_rate

        mapdictname = {
            tableallspots: "tableallspots",
            tableUB: "tableUB",
            tableMR: "tableMR",
        }

        peak1 = [1020.66, 1667.9]
        peak2 = [917.79, 1015.67]
        radius = 6.0

        tabpeak1 = LTHDF5.query_peak_location(tableallspots, peak1)
        tabpeak2 = LTHDF5.query_peak_location(tableallspots, peak2)

        tab12 = LTHDF5.query_twopeaks_location(tableallspots, peak1, peak2)

        tt = LTHDF5.query_grainlocation(tableallspots, 2000, 0)
        #    plotpresence(tt)

        print(LTHDF5.query_Miller_mainpeak(tableallspots, 1900, 2, 2))
