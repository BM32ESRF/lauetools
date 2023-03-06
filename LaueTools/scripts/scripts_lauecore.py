from __future__ import absolute_import
# -*- coding: utf-8 -*-
"""
#  scripts and example Laue Pattern simulation based on lauecore.py  (new name of laue6.py)
#  J. S. Micha   micha [at] esrf [dot] fr
# version May 2019
#  from LaueTools package
#  http://sourceforge.net/projects/lauetools/

"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"
import sys
import os

# # workaround to launch this script with modules in parent folder
# # whatever the folder it is launched in 
# LTpath = os.path.dirname(os.path.abspath(__file__))
# while not LTpath.endswith('LaueTools'):
#     LTpath = os.path.dirname(LTpath)
# LTpath = os.path.dirname(LTpath)
# if LTpath not in sys.path:
#     sys.path.append(LTpath)

import time
import pickle
import numpy as np

import LaueTools
print("Using LaueToolsFolder: ", os.path.abspath(LaueTools.__file__))

import LaueTools.CrystalParameters as CP
from LaueTools.dict_LaueTools import (dict_Rot,
                                    dict_Materials,
                                    dict_Vect,
                                    dict_Extinc,
                                    CST_ENERGYKEV)

import LaueTools.generaltools as GT
import LaueTools.lauecore as LT
import LaueTools.IOLaueTools as IOLT
import LaueTools.LaueGeometry as LTGeo

DEFAULT_TOP_GEOMETRY = LT.DEFAULT_TOP_GEOMETRY
DEFAULT_DETECTOR_DISTANCE = LT.DEFAULT_DETECTOR_DISTANCE
# --- -------------------  TESTS & EXAMPLES
def test_simulation():
    """
    test scenario
    """
    

    #  SIMULATION Compute list of spots for each grains

    emin = 5
    emax = 25
    # What kind of 2D plot must be display#
    # 'xy' plot (xcam, ycam)  (X horizontal + along the x-ray ; Y horizontal + towards the GMT door; Z vertical + towards the ceilling)
    # 'XYcam' plot (xcam, ycam) Xmas
    # 'xytest' plot pour test orientation cam
    # '2thetachi' plot (2theta, chi)
    # 'directionq' plot (longitude, latitude) de q
    whatplot = "2thetachi"
    # detection geometry:
    # 'Z>0' top camera 'Y>0' side + 'Y<0' side - 'X<0' backreflection  'X>0' transmission
    kf_direction = "Z>0"  # 'Z>0' 'Y>0' 'Y<0' 'X<0'

    # mat_UO2_He_103_1 = GT.OrientMatrix_fromGL(filename = 'UO2_He_103_1L.dat')
    mat_Id = np.eye(3)
    # mymat1 = fromelemangles_toMatrix([  1.73284101e-01 ,  3.89021368e-03 ,  4.53663032e+01])
    # mymat2 = fromelemangles_toMatrix([  0. ,  35.19 ,45.])
    # mattest= [[-0.72419551 , 0.09344237, -0.6832345 ], [-0.59410999 , 0.41294652,  0.69029595], [ 0.3465515 ,  0.90488673, -0.24714785]]
    # #mat_Cu = GT.OrientMatrix_fromGL(filename = 'matrixfromopenGL_311_c2.dat')
    # mat_Cu = np.eye(3)
    # matfromHough = fromelemangles_toMatrix([85., 68., 76.])
    # matfromHough = fromEULERangles_toMatrix([42., 13., 20.])
    grain1 = [mat_Id, "dia", dict_Rot["Si_001"], "Si"]
    # grain2 = [mat_Id, 'dia', mymat2,'Si']
    # #grainUO2_103 = [vecteurref, [1, 1, 1],mat_UO2_He_103_1,'UO2']
    # grainHough = [mat_Id, 'fcc', matfromHough,'UO2']
    # graintest = [mat_Id, 'fcc', mattest,'UO2']

    # GRAIN Must Be Prepared !!! see CP.Prepare_Grain() or fill before dict_Material

    starting_time = time.time()

    # vec and indices of spots Z > 0 (vectors [x, y, z] array, miller indices [H, K, L] array)
    vecind = LT.getLaueSpots(
        CST_ENERGYKEV / emax,
        CST_ENERGYKEV / emin,
        [grain1],
        fastcompute=0,
        kf_direction=kf_direction,
    )
    pre_filename = "totobill"

    linestowrite = [[""]]
    # print "vecind",vecind
    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)
    oncam_sansh = LT.filterLaueSpots(
        vecind,
        fileOK=1,
        fastcompute=0,
        kf_direction=kf_direction,
        detectordistance=DEFAULT_DETECTOR_DISTANCE,
        HarmonicsRemoval=1,
        linestowrite=linestowrite,
    )
    # print "oncam_sansh",oncam_sansh

    # plot and save and write file of simulated Data
    # third input is a list whose each element corresponds to one grain list of spots object

    # loadind experimental data:
    data_filename = "Ge_test.cor"
    data_2theta, data_chi, data_x, data_y, data_I = IOLT.readfile_cor(data_filename)

    # ----------  Time consumption information
    finishing_time = time.time()
    duration = finishing_time - starting_time
    print("Time duration for computation %.2f sec." % duration)

    Plot_Laue(
        emin,
        emax,
        oncam_sansh,
        data_2theta,
        data_chi,
        data_filename,
        removeharmonics=1,
        kf_direction=kf_direction,
        Plot_Data=1,
        Display_label=0,
        What_to_plot=whatplot,
        saveplotOK=0,
        WriteLogFile=1,
        linestowrite=linestowrite,
    )

    # logbook file edition
    IOLT.writefile_log(
        output_logfile_name=pre_filename + ".log", linestowrite=linestowrite
    )

    return CreateData_(oncam_sansh, outputname="tototable", pickledOK=0), oncam_sansh

def CreateData_(list_spots, outputname="fakedatapickle", pickledOK=0):
    """
    returns list of (2theta, chi)
    
    :param list_spots: list of list of instances of class 'spot'
    
    Data may be pickled if needed

    USED in scripts
    """
    mydata = [[elem.Twicetheta, elem.Chi] for elem in list_spots[0]]
    if outputname != None:
        if pickledOK:
            filepickle = open(outputname, "w")
            pickle.dump(mydata, filepickle)
            filepickle.close()
        else:
            line_to_write = []
            line_to_write.append(["2theta   chi h k l E spotnumber"])
            spotnumber = 0
            for i in list(range(len(list_spots))):

                for elem in list_spots[i]:

                    line_to_write.append(
                        [
                            str(elem.Twicetheta),
                            str(elem.Chi),
                            str(elem.Millers[0]),
                            str(elem.Millers[1]),
                            str(elem.Millers[2]),
                            str(elem.EwaldRadius * CST_ENERGYKEV),
                            str(spotnumber),
                        ]
                    )
                    spotnumber += 1

            filetowrite = open(outputname, "w")
            aecrire = line_to_write
            for line in aecrire:
                lineData = "\t".join(line)
                filetowrite.write(lineData)
                filetowrite.write("\n")
            filetowrite.close()

    return mydata

def test_speed():
    # ----------  Time consumption information
    Starting_time = time.time()

    Array2thetachi, Oncam = test_simulation()

def simulatepattern(
    grain,
    emin,
    emax,
    kf_direction,
    data_filename,
    PlotLaueDiagram=1,
    Plot_Data=0,
    verbose=0,
    detectordistance=DEFAULT_DETECTOR_DISTANCE,
    ResolutionAngstrom=False,
    Display_label=1,
    HarmonicsRemoval=1,
):
    """
    USED only for scripts ...???
    """
    whatplot = "2thetachi"

    print("kf_direction", kf_direction)

    Starting_time = time.time()

    # vec and indices of spots Z > 0 (vectors [x, y, z] array, miller indices [H, K, L] array)

    vecind = LT.getLaueSpots(
        CST_ENERGYKEV / emax,
        CST_ENERGYKEV / emin,
        grain,
        fastcompute=0,
        kf_direction=kf_direction,
        verbose=verbose,
        ResolutionAngstrom=ResolutionAngstrom,
    )
    pre_filename = "laue6simul"

    print("len(vecind[0])", len(vecind[0][0]))
    linestowrite = [[""]]
    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)
    try:
        oncam_sansh = LT.filterLaueSpots(
            vecind,
            fileOK=1,
            fastcompute=0,
            kf_direction=kf_direction,
            detectordistance=detectordistance,
            HarmonicsRemoval=HarmonicsRemoval,
            linestowrite=linestowrite,
        )
    #        print "oncam_sansh",oncam_sansh
    except UnboundLocalError:
        raise UnboundLocalError("Empty list of spots or vector (variable: vecind)")

    # plot and save and write file of simulated Data
    # third input is a list whose each element corresponds to one grain list of spots object

    data_2theta, data_chi = None, None  # exp. data

    if data_filename != None:
        # loadind experimental data:
        try:
            res = IOLT.readfile_cor(data_filename)

        except IOError:
            print(
                "You must enter the name of experimental datafile (similar to e.g 'Ge_test.cor')"
            )
            print("or set Plot_Data=0")
            return

        else:
            alldata, data_2theta, data_chi, data_x, data_y, data_I, detparam = res

    else:
        print(
            "You must enter the name of experimental datafile (similar to e.g 'Ge_test.cor')"
        )

    # ----------  Time consumption information
    finishing_time = time.time()
    duration = finishing_time - Starting_time
    print("Time duration for computation %.2f sec." % duration)

    if PlotLaueDiagram:

        Plot_Laue(
            emin,
            emax,
            oncam_sansh,
            data_2theta,
            data_chi,
            data_filename,
            removeharmonics=1,
            kf_direction=kf_direction,
            Plot_Data=Plot_Data,
            Display_label=Display_label,
            What_to_plot=whatplot,
            saveplotOK=0,
            WriteLogFile=1,
            linestowrite=linestowrite,
        )
   
    return True

def simulatepurepattern(
    grain,
    emin,
    emax,
    kf_direction,
    data_filename,
    PlotLaueDiagram=1,
    Plot_Data=0,
    verbose=0,
    detectordistance=DEFAULT_DETECTOR_DISTANCE,
    ResolutionAngstrom=False,
    Display_label=1,
    HarmonicsRemoval=1,
):
    """
    NOT USED ??!!
    """
    vecind = LT.getLaueSpots( CST_ENERGYKEV / emax, CST_ENERGYKEV / emin, grain,
                            fastcompute=0, kf_direction=kf_direction,
                            verbose=verbose, ResolutionAngstrom=ResolutionAngstrom, )

    print("len(vecind[0])", len(vecind[0][0]))

    # selecting RR nodes without harmonics (fastcompute = 1 loses the miller indices and RR positions associations for quicker computation)

    oncam_sansh = LT.filterLaueSpots(vecind,
                                    fileOK=0,
                                    fastcompute=0,
                                    kf_direction=kf_direction,
                                    detectordistance=detectordistance,
                                    HarmonicsRemoval=HarmonicsRemoval)

    return True


def Plot_Laue( _emin, _emax, list_of_spots, _data_2theta, _data_chi, _data_filename,
    Plot_Data=1,
    Display_label=1,
    What_to_plot="2thetachi",
    saveplotOK=1,
    WriteLogFile=1,
    removeharmonics=0,
    kf_direction=DEFAULT_TOP_GEOMETRY,
    linestowrite=[[""]]):
    """
    basic function to plot LaueData for showcase
    """
    import annot

    # creation donnee pour etiquette sur figure
    # if plotOK:
    font = {"fontname": "Courier", "color": "k", "fontweight": "normal", "fontsize": 7}
    fig = figure(figsize=(6.0, 6.0), dpi=80)  # ? small but ok for savefig!
    ax = fig.add_subplot(111)

    title(
        "Laue pattern %.1f-%.1f keV \n %s" % (_emin, _emax, _data_filename),
        font,
        fontsize=12,
    )
    # text(0.5, 2.5, 'a line', font, color = 'k')

    if removeharmonics == 0:
        oncam_sansh = Calc_spot_on_cam_sansh(
            list_of_spots, fileOK=WriteLogFile, linestowrite=linestowrite
        )
    elif removeharmonics == 1:
        print("harmonics have been already removed")
        oncam_sansh = list_of_spots

    nb_of_grains = len(list_of_spots)
    print("Number of grains in Plot_Laue", nb_of_grains)

    xxx = emptylists(nb_of_grains)
    yyy = emptylists(nb_of_grains)

    energy = emptylists(nb_of_grains)
    trans = emptylists(nb_of_grains)
    indy = emptylists(nb_of_grains)

    # calculation for a list of spots object of spots belonging to the camera direction

    if WriteLogFile:
        linestowrite.append(["-" * 60])
        linestowrite.append([" Number of spots w / o harmonics: "])

    # building  simul data
    # loop over grains
    for i in list(range(nb_of_grains)):

        print(
            "Number of spots on camera (w / o harmonics): GRAIN number ",
            i,
            " : ",
            len(oncam_sansh[i]),
        )
        if WriteLogFile:
            linestowrite.append(
                [
                    "--- GRAIN number ",
                    str(i),
                    " : ",
                    str(len(oncam_sansh[i])),
                    " ----------",
                ]
            )
        facscale = 1.0
        # loop over spots
        for elem in oncam_sansh[i]:
            if What_to_plot == "2thetachi" and (
                isinstance(kf_direction, list) or kf_direction in ("Z>0", "Y>0", "Y<0")
            ):
                xxx[i].append(elem.Twicetheta)
                yyy[i].append(elem.Chi)
            elif What_to_plot == "XYcam" or kf_direction in ("X>0", "X<0"):
                facscale = 2.0  # for bigger spot on graph
                xxx[i].append(elem.Xcam)
                yyy[i].append(elem.Ycam)
            elif What_to_plot == "projgnom":  # essai pas fait
                xxx[i].append(elem.Xcam)
                yyy[i].append(elem.Ycam)
            #            elif What_to_plot == 'directionq':
            #                xxx[i].append(elem._philongitude)
            #                yyy[i].append(elem._psilatitude)
            else:
                xxx[i].append(elem.Xcam)
                yyy[i].append(elem.Ycam)
            energy[i].append(elem.EwaldRadius * CST_ENERGYKEV)
            # if etiquette: # maintenant avec la souris!!!
            indy[i].append(elem.Millers)

        print(len(xxx[i]))

    # plot of simulation and or not data
    # affichage avec ou sans etiquettes
    dicocolor = {0: "k", 1: "r", 2: "g", 3: "b", 4: "c", 5: "m"}
    #        dicosymbol = {0:'k.',1:'r.',2:'g.',3:'b.',4:'c.',5:'m.'}

    # exp data
    if _data_2theta is not None and _data_chi is not None and Plot_Data:
        xxx_data = tuple(_data_2theta)
        yyy_data = tuple(_data_chi)

        # PLOT EXT DATA
        # print "jkhhkjhkjhjk**********",xxx_data[:20]
        ax.scatter(xxx_data, yyy_data, s=40, c="w", marker="o", faceted=True, alpha=0.5)
        # END PLOT EXP DATA

    # Plot Simulated laue patterns
    for i in list(range(nb_of_grains)):  # loop over grains
        # ax.plot(tuple(xxx[i]),tuple(yyy[i]),dicosymbol[i])

        # display label to the side of the point
        if nb_of_grains >= 3:
            #            trans[i] = offset(ax, fig, 10, -5 * (nb_of_grains - 2) + 5 * i)
            trans[i] = offset(
                ax.transData, fig, 10, -5 * (nb_of_grains - 2) + 5 * i, units="dots"
            )
        else:
            #            trans[i] = offset(ax, fig, 10, 5 * (-1) ** (i % 2))
            trans[i] = offset(ax.transData, fig, 10, 5 * (-1) ** (i % 2), units="dots")

        print("nb of spots for grain # %d : %d" % (i, len(xxx[i])))
        #        print tuple(xxx[i])
        #        print tuple(yyy[i])
        ax.scatter(
            tuple(xxx[i]),
            tuple(yyy[i]),
            s=[facscale * int(200.0 / en) for en in energy[i]],
            c=dicocolor[(i + 1) % (len(dicocolor))],
            marker="o",
            faceted=False,
            alpha=0.5,
        )

        if Display_label:  # displaying labels c and d at position a, b
            for a, b, c, d in zip(
                tuple(xxx[i]), tuple(yyy[i]), tuple(indy[i]), tuple(energy[i])
            ):
                ax.text(
                    a,
                    b,
                    "%s,%.1f" % (str(c), d),
                    font,
                    fontsize=7,
                    color=dicocolor[i % (len(dicocolor))],
                    transform=trans[i],
                )

    ax.set_aspect(aspect="equal")
    if What_to_plot == "XYcam":
        # ax.set_axis_off()
        ax.set_xlim((0.0, 2030.0))
        ax.set_ylim((0.0, 2030.0))
    #    elif What_to_plot == 'directionq':
    #        #ax.set_axis_off()
    #        ax.set_xlim((120.,240.))
    #        ax.set_ylim((20.,70.))

    # CCD circle contour border drawing

    t = np.arange(0.0, 6.38, 0.1)

    if What_to_plot == "2thetachi" and kf_direction in ("Y>0", "Y<0"):
        if kf_direction == "Y>0":
            si = 1
        else:
            si = -1

        circY = si * (-90 + 45 * np.sin(t))
        circX = 90 + 45 * np.cos(t)
        Xtol = 5.0
        Ytol = 5.0

    elif What_to_plot == "2thetachi" and (
        isinstance(kf_direction, list) or kf_direction in ("Z>0")
    ):

        circX = 90 + 45 * np.cos(t)
        circY = 45 * np.sin(t)
        Xtol = 5.0
        Ytol = 5.0
    #    elif What_to_plot == 'directionq': #essai coordonnee du cercle camera???
    #        circX = map(lambda tutu: 180 + 60 * math.cos(tutu), t)
    #        circY = map(lambda tutu: 45 + 25 * math.sin(tutu), t)
    #        Xtol = 5.
    #        Ytol = 5.
    # TODO embellished with numpy like above
    elif kf_direction in ("X>0", "X<0"):
        circX = [1024 + 1024 * math.cos(tutu) for tutu in t]
        circY = [1024 + 1024 * math.sin(tutu) for tutu in t]
        Xtol = 20
        Ytol = 20
    else:
        circX = [1024 + 1024 * math.cos(tutu) for tutu in t]
        circY = [1024 + 1024 * math.sin(tutu) for tutu in t]
        Xtol = 20
        Ytol = 20
    ax.plot(circX, circY, "")

    if saveplotOK:
        fig.savefig(
            "figlaue.png", dpi=300, facecolor="w", edgecolor="w", orientation="portrait"
        )

    if not Display_label:
        whole_xxx = xxx[0]
        whole_yyy = yyy[0]
        whole_indy = indy[0]
        whole_energy = energy[0]
        for i in list(range(nb_of_grains - 1)):
            whole_xxx += xxx[i + 1]
            whole_yyy += yyy[i + 1]
            whole_indy += indy[i + 1]
            whole_energy += energy[i + 1]

        whole_energy_r = np.around(np.array(whole_energy), decimals=2)

        todisplay = np.vstack((np.array(whole_indy, dtype=np.int8).T, whole_energy_r)).T

        # af =  annot.AnnoteFinder(xxx[0] + xxx[1],yyy[0] + yyy[1],indy[0] + indy[1])
        # af =  annot.AnnoteFinder(whole_xxx, whole_yyy, whole_indy, xtol = 5.,ytol = 5.)
        # af =  annot.AnnoteFinder(whole_xxx, whole_yyy, whole_energy_r, xtol = 10.,ytol = 10.)
        af = annot.AnnoteFinder(whole_xxx, whole_yyy, todisplay, xtol=Xtol, ytol=Ytol)
        # af =  AnnoteFinder(xxx[0],yyy[0],energy[0],indy[0])
        connect("button_press_event", af)
    show()


def Calc_spot_on_cam_sansh(listoflistofspots, fileOK=0, linestowrite=[[""]]):

    """ Calculates list of grains spots on camera w / o harmonics,
    from liligrains (list of grains spots instances scattering in 2pi steradians)
    [[spots grain 0],[spots grain 0],etc] = >
    [[spots grain 0],[spots grain 0],etc] w / o harmonics and on camera  CCD

    TODO: rather useless  only used in Plot_Laue(), need to be replace by filterLaueSpots
    """

    nbofgrains = len(listoflistofspots)

    _oncam = emptylists(nbofgrains)  # list of spot on the camera (harmonics included)
    _oncamsansh = emptylists(
        nbofgrains
    )  # list of spot on the camera (harmonics EXCLUDED) only fondamental (but still with E > Emin)

    if fileOK:
        linestowrite.append(["\n"])
        linestowrite.append(
            ["--------------------------------------------------------"]
        )
        linestowrite.append(
            ["------------- Simulation Data --------------------------"]
        )
        linestowrite.append(
            ["--------------------------------------------------------"]
        )
        linestowrite.append(["\n"])
        linestowrite.append(
            [
                "#grain, h, k, l, energy(keV), 2theta (deg), chi (deg), X_Xmas, Y_Xmas, X_JSM, Y_JSM, Xtest, Ytest"
            ]
        )

    for i in list(range(nbofgrains)):
        condCam = (elem.Xcam ** 2 + elem.Ycam ** 2 <= (DEFAULT_DETECTOR_DIAMETER / 2) ** 2)
        _oncam[i] = [elem for elem in listoflistofspots[i] if (elem.Xcam is not None and condCam)]

        # Creating list of spot without harmonics
        _invdict = {}  # for each grain, use of _invdict to remove harmonics present in _oncam

        for elem in sorted(_oncam[i], reverse=True):
            _invdict[elem.__hash__()] = elem
            _oncamsansh[i] = [_invdict[cle] for cle in list(_invdict.keys())]

        if fileOK:
            IOLT.Writefile_data_log(_oncamsansh[i], i, linestowrite=linestowrite)

    return _oncamsansh


emin = 5
emax = 15
kf_direction = "Z>0"
data_filename = "Ge_test.cor"  # experimental data

Id = np.eye(3)

if 0:
    grainSi = CP.Prepare_Grain("Si", dict_Rot["Si_001"])

    grainSi[2] = GT.fromEULERangles_toMatrix([20.0, 10.0, 50.0])

    print("\n*******************\nSimulation with fastcompute = 0")
    vecind = LT.getLaueSpots(
        CST_ENERGYKEV / emax,
        CST_ENERGYKEV / emin,
        [grainSi],
        fastcompute=0,
        kf_direction=kf_direction,
        verbose=1,
    )

    print("nb of spots with harmonics %d\n\n" % len(vecind[0][0]))

    print("\n*********\nSimulation with fastcompute = 1")

    vecindfast = LT.getLaueSpots(CST_ENERGYKEV / emax,
                                CST_ENERGYKEV / emin,
                                [grainSi],
                                fastcompute=1,
                                kf_direction=kf_direction,
                                verbose=1)

    print("nb of spots with harmonics fast method", len(vecindfast[0][0]))

    print("\n*******************\n --------  Harmonics removal -----------\n")
    linestowrite = [[""]]
    oncam_sansh00 = LT.filterLaueSpots(
        vecind,
        fileOK=1,
        fastcompute=0,
        kf_direction=kf_direction,
        HarmonicsRemoval=1,
        linestowrite=linestowrite,
    )

    print("after harm removal 00", len(oncam_sansh00[0]))

    oncam_sansh01 = LT.filterLaueSpots(
        vecind,
        fileOK=1,
        fastcompute=1,
        kf_direction=kf_direction,
        HarmonicsRemoval=1,
        linestowrite=linestowrite,
    )

    print("after harm removal 01", len(oncam_sansh01[0]))

    oncam_sansh11 = LT.filterLaueSpots(
        vecindfast,
        fileOK=1,
        fastcompute=1,
        kf_direction=kf_direction,
        HarmonicsRemoval=1,
        linestowrite=linestowrite,
    )

    print("after harm removal 11", len(oncam_sansh11[0]))

    # ------------------------
    print("\n*****************************\n rax compute\n")
    grainSi[2] = GT.fromEULERangles_toMatrix([20.0, 10.0, 50.0])

    # res = get_2thetaChi_withoutHarmonics(grainSi, 5, 15)

if 1:  # some grains example
    inittime = time.time()

    emax = 50
    emin = 5

    elem1 = "Ti"
    elem2 = "Cu"

    kf_direction = "Z>0"

    BmatTi = CP.calc_B_RR(dict_Materials["Ti"][1])

    BmatCu = CP.calc_B_RR(dict_Materials["Cu"][1])

    Umat = dict_Rot["OrientSurf111"]

    grainTi = [BmatTi, "no", Umat, "Ti"]

    grainCu = [BmatCu, "fcc", Id, "Cu"]
    grainCu__ = [BmatCu, "no", Id, "Cu"]

    dict_Materials["Cu2"] = ["Cu2", [5, 3.6, 3.6, 90, 90, 90], "fcc"]
    BmatCu2 = CP.calc_B_RR(dict_Materials["Cu2"][1])

    dict_Materials["Cu3"] = ["Cu3", [Id, Umat, Id, BmatCu2], "fcc"]  # Da, U, Dc, B

    grainCu2 = CP.Prepare_Grain("Cu2", Id)
    grainCu3 = CP.Prepare_Grain("Cu3", "ihjiĥiohio")

    grainSi1 = CP.Prepare_Grain("Cu", Id)
    grainSi2 = CP.Prepare_Grain("Si", dict_Rot["OrientSurf111"])

    grainAl2o3 = CP.Prepare_Grain("Al2O3", Id)

    # grains= [grainSi,grainCu3]
    grains = [grainSi1, grainCu3, grainAl2o3]

    print(grains)

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        HarmonicsRemoval=0,
    )

    finaltime = time.time()

    print("computation time is ", finaltime - inittime)

if 0:
    # strain study: ---------------------------------------
    dict_Materials["Cu2"] = ["Cu2", [5, 3.6, 3.6, 90, 90, 90], "fcc"]
    BmatCu2 = CP.calc_B_RR(dict_Materials["Cu2"][1])

    Umat = dict_Rot["OrientSurf111"]
    # Da, U, Dc, B
    dict_Materials["Cu3"] = ["Cu3", [Id, Umat, Id, BmatCu2], "fcc"]

    grains = []
    for k in list(range(-5, 6)):
        dict_Materials["Cu_%d" % k] = dict_Materials["Cu3"]
        # set material label
        dict_Materials["Cu_%d" % k][0] = "Cu_%d" % k
        # set deformation in absolute space
        dict_Materials["Cu_%d" % k][1][0] = [
            [1 + 0.01 * k, 0, 0],
            [0, 1, 0],
            [0, 0, 1.0],
        ]
        grains.append(CP.Prepare_Grain("Cu_%d" % k))

    simulatepattern(grains, emin, emax, kf_direction, data_filename, Plot_Data=0)
    # -----------------------------------------------

if 0:

    Bmat = dict_Vect["543_909075"]
    Umat = dict_Rot["OrientSurf001"]
    Dc = [[1.02, 0.01, 0], [-0.01, 0.98, 0.005], [0.001, -0.02, 1.01]]
    Dc = [[1.00, 0.00, 0], [-0.00, 1.00, 0.000], [0.000, -0.00, 1.03]]

    dict_Materials["mycell"] = ["mycell", [Id, Umat, Id, Bmat], "fcc"]
    dict_Materials["mycell_strained"] = ["mycell_strained", [Id, Umat, Dc, Bmat], "fcc"]

    mygrain = CP.Prepare_Grain("mycell")
    mygrain_s = CP.Prepare_Grain("mycell_strained")

    grains = [mygrain, mygrain_s]

    simulatepattern(grains, emin, emax, kf_direction, data_filename, Plot_Data=0)

if 0:

    Bmat = dict_Vect["543_909075"]
    Umat = dict_Rot["mat311c1"]
    Dc = dict_Vect["shear4"]

    dict_Materials["mycell_s"] = ["mycell_s", [Id, Umat, Dc, Bmat], "fcc"]
    dict_Materials["mycell"] = ["mycell", [Id, Umat, Id, Bmat], "fcc"]

    mygrain_s = CP.Prepare_Grain("mycell_s")
    mygrain = CP.Prepare_Grain("mycell")

    grains = [mygrain_s]

    simulatepattern(
        grains, emin, emax, kf_direction, data_filename, Plot_Data=0, verbose=1
    )

if 0:
    emin = 5
    emax = 22
    kf_direction = "X>0"  # transmission
    kf_direction = "Z>0"  # reflection

    ResolutionAngstrom = 2.0

    # overwriting dict_Materials['smallpro']
    dict_Materials["smallpro"] = ["smallpro", [20, 4.8, 49, 90, 90, 90], "no"]
    mygrain = CP.Prepare_Grain("smallpro", dict_Rot["mat311c1"])
    mygrain = CP.Prepare_Grain("smallpro", Id)

    grains = [mygrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=69,
        ResolutionAngstrom=ResolutionAngstrom,
    )

if 0:
    emin = 5
    emax = 30
    kf_direction = "X>0"  # transmission
    #        kf_direction = 'Z>0' # reflection

    ResolutionAngstrom = False

    # overwriting dict_Materials['smallpro']

    mat111alongx = np.dot(GT.matRot([1, 0, 0], -45.0), GT.matRot([0, 0, 1], 45.0))

    print("mat111alongx", mat111alongx)

    matmono = np.dot(GT.matRot([1, 0, 0], -1), mat111alongx)

    mygrain = CP.Prepare_Grain("Cu", matmono)

    grains = [mygrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=10,
        ResolutionAngstrom=ResolutionAngstrom,
    )

if 0:
    emin = 8
    emax = 25
    kf_direction = "X<0"  # back reflection
    #        kf_direction = 'Z>0' # reflection

    ResolutionAngstrom = False

    # overwriting dict_Materials['smallpro']

    matmono = np.dot(GT.matRot([0, 0, 1], 20), np.eye(3))

    mygrain = CP.Prepare_Grain("Si", matmono)

    grains = [mygrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=55,
        ResolutionAngstrom=ResolutionAngstrom,
        Display_label=0,
    )

if 0:
    emin = 8
    emax = 25
    kf_direction = "X<0"  # back reflection
    #        kf_direction = 'Z>0' # reflection

    ResolutionAngstrom = False

    # overwriting dict_Materials['smallpro']

    matmother = np.dot(GT.matRot([0, 0, 1], -5), np.eye(3))
    matmisorient = np.dot(GT.matRot([-1, 1, 1], 0.1), matmother)

    maingrain = CP.Prepare_Grain("Si", matmother)

    mygrain = CP.Prepare_Grain("Si", matmisorient)

    grains = [maingrain, mygrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=55,
        ResolutionAngstrom=ResolutionAngstrom,
        Display_label=0,
    )

if 0:
    emin = 5
    emax = 20
    kf_direction = [90, 45]
    #        kf_direction = 'Z>0' # reflection

    ResolutionAngstrom = False

    # overwriting dict_Materials['smallpro']

    matmother = np.dot(GT.matRot([0, 0, 1], 0), np.eye(3))
    #        matmisorient = np.dot(GT.matRot([-1, 1, 1], .1), matmother)

    maingrain = CP.Prepare_Grain("Si", matmother)

    grains = [maingrain, maingrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=70,
        ResolutionAngstrom=ResolutionAngstrom,
        Display_label=0,
    )

if 0:
    emin = 5
    emax = 20
    #        kf_direction = [0, 0]
    kf_direction = "X>0"  #

    ResolutionAngstrom = False

    # overwriting dict_Materials['smallpro']

    matmother = np.dot(GT.matRot([0, 0, 1], 0), np.eye(3))
    #        matmisorient = np.dot(GT.matRot([-1, 1, 1], .1), matmother)

    maingrain = CP.Prepare_Grain("Si", matmother)

    grains = [maingrain, maingrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=70,
        ResolutionAngstrom=ResolutionAngstrom,
        Display_label=0,
    )


if 0:
    emin = 5
    emax = 22

    Detpos = 0  # 1 = 'top' 0 = 'trans'

    if Detpos == 1:
        # on top
        kf_direction = "Z>0"  # reflection
        Detdist = 70  # mm
    elif Detpos == 0:
        # transmission
        kf_direction = "X>0"  # transmission
        Detdist = 100  # mm

    ResolutionAngstrom = None

    # overwriting dict_Materials['smallpro']
    dict_Materials["smallpro"] = ["smallpro", [20, 4.8, 49, 90, 90, 90], "no"]
    mygrain = CP.Prepare_Grain("smallpro", dict_Rot["mat311c1"])
    mygrain = CP.Prepare_Grain("smallpro", Id)
    ResolutionAngstrom = 2.0

    grains = [mygrain]

    simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        Plot_Data=0,
        verbose=1,
        detectordistance=Detdist,
        ResolutionAngstrom=ResolutionAngstrom,
    )

if 0:
    emin = 5
    emax = 22

    Detpos = 0  # 1 = 'top' 0 = 'trans'

    if Detpos == 1:
        # on top
        kf_direction = "Z>0"  # reflection
        Detdist = 70  # mm
    elif Detpos == 0:
        # transmission
        kf_direction = "X>0"  # transmission
        Detdist = 100.0  # mm

    ResolutionAngstrom = None

    # overwriting dict_Materials['smallpro']
    dict_Materials["smallpro"] = ["smallpro", [20, 4.8, 49, 90, 90, 90], "no"]
    ResolutionAngstrom = 2.0

    for ori in list(dict_Rot.keys())[2:3]:
        mygrain = CP.Prepare_Grain("smallpro", dict_Rot[ori])

        grains = [mygrain]

        mydata = simulatepattern(
            grains,
            emin,
            emax,
            kf_direction,
            data_filename,
            Plot_Data=0,
            verbose=1,
            detectordistance=Detdist,
            ResolutionAngstrom=ResolutionAngstrom,
        )

    ard = np.array(mydata[0])

    import pylab as pp

    pp.figure(1)
    pp.scatter(ard[:, 0], ard[:, 1])

    xyd = np.array([[elem.Xcam, elem.Ycam] for elem in mydata[1][0]])

    pp.figure(2)
    pp.scatter(xyd[:, 0], xyd[:, 1])

    calib = [100.0, 1024.0, 1024.0, 90, 0.0]

    xyd_fromfind2 = LTGeo.calc_xycam_from2thetachi(
        ard[:, 0],
        ard[:, 1],
        calib,
        verbose=0,
        pixelsize=165.0 / 2048,
        kf_direction=kf_direction)

    X, Y, theta = xyd_fromfind2

    pp.scatter(X, Y, c="r")

    pp.show()


if 0:
    emin = 50
    emax = 120

    Detpos = 0  # 1 = 'top' 0 = 'trans'

    if Detpos == 1:
        # on top
        kf_direction = "Z>0"  # reflection
        Detdist = 70  # mm
    elif Detpos == 0:
        # transmission
        kf_direction = "X>0"  # transmission
        Detdist = 100.0  # mm

    ResolutionAngstrom = 0.5

    for ori in list(dict_Rot.keys())[2:3]:
        mygrain = CP.Prepare_Grain("Ni", dict_Rot[ori])

        grains = [mygrain]

        mydata = simulatepattern(
            grains,
            emin,
            emax,
            kf_direction,
            data_filename,
            Plot_Data=0,
            verbose=1,
            detectordistance=Detdist,
            ResolutionAngstrom=ResolutionAngstrom,
        )

    ard = np.array(mydata[0])

    print("mydata", mydata)

    import pylab as pp

    pp.figure(1)
    pp.scatter(ard[:, 0], ard[:, 1])

    xyd = np.array([[elem.Xcam, elem.Ycam] for elem in mydata[1][0]])

    miller = [elem.Millers for elem in mydata[1][0]]

    print(miller)

    pp.figure(2)
    pp.scatter(xyd[:, 0], xyd[:, 1])

    calib = [105.624, 1017.50, 996.62, -0.027, -116.282]
    pixelsize = 0.048

    xyd_fromfind2 = LTGeo.calc_xycam_from2thetachi(
        ard[:, 0],
        ard[:, 1],
        calib,
        verbose=0,
        pixelsize=pixelsize,
        kf_direction=kf_direction)

    X, Y, theta = xyd_fromfind2

    pp.scatter(X, Y, c="r")

    pp.show()

    f = open("Ni_fake_transmission.dat", "w")
    f.write("X Y I from simulation\n")
    for k in list(range(len(X))):
        f.write(
            "%s %s %s %s %s %s %s %s %s %s %s\n"
            % (
                X[k],
                Y[k],
                65000 * (1 - 0.8 * k / len(X)),
                65000 * (1 - 0.8 * k / len(X)),
                1.47,
                1.81,
                82.460,
                -1.07,
                1.78,
                624.74,
                64740,
            )
        )
    f.write("# %s pixelsize %s" % (str(calib), pixelsize))
    f.close()

if 0:  # Si 111 in transmission on ID15
    emin = 80
    emax = 100

    Detpos = 0  # 1 = 'top' 0 = 'trans'

    if Detpos == 1:
        # on top
        kf_direction = "Z>0"  # reflection
        Detdist = 100  # mm
    elif Detpos == 0:
        # transmission
        kf_direction = "X>0"  # transmission
        Detdist = 100.0  # mm

    mattrans3 = [
        [0.998222982873332, 0.04592237603705288, -0.037973831023250665],
        [0.0036229001244100726, 0.5893105150537007, 0.8078985031808321],
        [0.05947899678171664, -0.8066004291012069, 0.5880969279936651],
    ]
    mygrain = CP.Prepare_Grain("Si", dict_Rot["mat111alongx"])
    mygrain = CP.Prepare_Grain("Si", mattrans3)

    grains = [mygrain]

    mydata = simulatepattern(
        grains,
        emin,
        emax,
        kf_direction,
        data_filename,
        PlotLaueDiagram=0,
        Plot_Data=0,
        verbose=1,
        detectordistance=Detdist,
        ResolutionAngstrom=None,
    )

    # two theta chi
    ard = np.array(mydata[0])

    print("ard", ard)

    import pylab as pp

    pp.figure(1)
    pp.scatter(ard[:, 0], ard[:, 1])

    # pixel X Y position with default camera settings ...
    #         xyd = np.array([[elem.Xcam, elem.Ycam] for elem in mydata[1][0]])
    #
    #         pp.figure(2)
    #         pp.scatter(xyd[:, 0], xyd[:, 1])

    calib = [105.0, 1024.0, 1024.0, 0.0, 0.0]

    xyd_fromfind2 = LTGeo.calc_xycam_from2thetachi(
        ard[:, 0],
        ard[:, 1],
        calib,
        verbose=0,
        pixelsize=0.048,
        kf_direction=kf_direction)

    X, Y, theta = xyd_fromfind2

    intensity = np.ones_like(X)

    IOLT.writefile_Peaklist(
        "ID15transSi111",
        np.array(
            [
                X,
                Y,
                intensity,
                intensity,
                intensity,
                intensity,
                intensity,
                intensity,
                intensity,
                intensity,
                intensity,
            ]
        ).T,
    )

    pp.figure(3)
    pp.scatter(X, Y, c="r")

    pp.show()
