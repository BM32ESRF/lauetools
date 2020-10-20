
import os
import sys
import LaueTools.indexingSpotsSet as ISS

if 0:
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
    finalindex = 10
    stepindex = 1

    filedet = '/home/micha/LaueProjects/SiSibulle_Lukas/calibSilukas.det'

    guessedMatricesFile ='/home/micha/LaueProjects/SiSibulle_Lukas/SiSi.ubs'
    MinimumMatchingRate = 4
    fileirp = os.path.join(MainFolder, "SiSi.irp")
    spottrackingfile = '/home/micha/LaueProjects/SiSibulle_Lukas/SiLukas_acceptedspots.cor'

    nb_cpus = 3

    # boolean
    reanalyse = True
    use_previous_results = False
    updatefitfiles = False

    trackingmode= False

    verbose = 0
    
if 1:

    MainFolder = '/home/micha/LaueToolsPy3/LaueTools/Examples/CuSi'
    print("MainFolder", MainFolder)

    #  fields like on the GUI   FileSeries/Index_Refine.py

    filepathdat = os.path.join(MainFolder, "corfiles")
    filepathcor = os.path.join(MainFolder, "corfiles")
    filepathout = os.path.join(MainFolder, "fitfiles")

    fileprefix = "SiCustrain"
    filesuffix = '.cor'
    nbdigits_filename = 0
    startindex = 0
    finalindex = 5
    stepindex = 1

    filedet = None
    guessedMatricesFile = None
    MinimumMatchingRate = 4
    
    if 0:
        fileirp = os.path.join(MainFolder, "cu.irp")
        spottrackingfile = '/home/micha/LaueToolsPy3/LaueTools/Examples/CuSi/corfiles/SiCustrain5_Cu20spots_1Sipeak.fit'

    if 1:
        fileirp = os.path.join(MainFolder, "sicu.irp")
        spottrackingfile = None

    nb_cpus = 2

    # boolean
    reanalyse = True
    use_previous_results = True
    updatefitfiles = False

    trackingmode = False

    verbose = 1


multiple_results, filedict = ISS.indexFilesSeries(filepathdat, filepathcor, filepathout,
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
                    trackingmode=trackingmode,
                    build_hdf5=True,
                    verbose=verbose)

print('nb of results', len(multiple_results))
print(multiple_results[0])
