r"""
Module to compute Laue Patterns from several crystals in various geometry

Main author is J. S. Micha:   micha [at] esrf [dot] fr

version July 2019
from LaueTools package for python2 hosted in

http://sourceforge.net/projects/lauetools/

or for python3 and 2 in

https://gitlab.esrf.fr/micha/lauetools
"""
import sys
import numpy as np

if sys.version_info.major == 3:
    from . import dict_LaueTools as DictLT
    from . import generaltools as GT
    from . import lauecore as LAUE
    from . import LaueGeometry as LTGeo
    from . import CrystalParameters as CP

else:
    import dict_LaueTools as DictLT
    import generaltools as GT
    import lauecore as LAUE
    import LaueGeometry as LTGeo
    import CrystalParameters as CP

try:
    WXPYTHON = True
    import wx
except ImportError:
    WXPYTHON = False

def Read_GrainListparameter(param):
    r"""
    Read dictionary of  input key parameters for simulation
    """
    # Elem, Extinc, Transf_a, Rot, Bmatrix, Transf_c, GrainName, transform = param
    key_material, Extinc, Transf_a, Rot, Bmatrix, Transf_c = param[:6]
    GrainName = str(param[6])
    # print "param in Read_GrainListparameter",param

    Extinctions = DictLT.dict_Extinc[str(Extinc)]
    Transform_labframe = DictLT.dict_Transforms[Transf_a]
    orientMatrix = DictLT.dict_Rot[Rot]
    B_matrix = DictLT.dict_Vect[Bmatrix]
    Transform_crystalframe = DictLT.dict_Transforms[Transf_c]

    return ([key_material, Extinctions, Transform_labframe, orientMatrix, B_matrix, Transform_crystalframe],
            GrainName, )


def Construct_GrainsParameters_parametric(SelectGrains_parametric):
    r"""
    return list of simulation parameters for each grain set (mother and children grains)
    """
    list_selectgrains_param = []
    # keys from dialogs were in reverse order
    # self.SelectGrains_parametric  == parametric_Grain_Dialog().SelectGrains
    for key_grain in sorted(SelectGrains_parametric.keys()):
        # print "self.SelectGrains_parametric[key_grain]",self.SelectGrains_parametric[key_grain]
        list_selectgrains_param.append(Read_GrainListparameter(SelectGrains_parametric[key_grain]))
    # print list_selectgrains_param
    return list_selectgrains_param


def dosimulation_parametric(_list_param, Transform_params=None, SelectGrains=None, emax=25.0,
                                                            emin=5.0,
                                                            detectordistance=68.7,
                                                            detectordiameter=165.0,
                                                            posCEN=(1024.0, 1024.0),
                                                            cameraAngles=(0.0, 0.0),
                                                            gauge=None,
                                                            kf_direction="Z>0",
                                                            pixelsize=165.0 / 2048,
                                                            dictmaterials=DictLT.dict_Materials):
    r"""
    Simulation of orientation or deformation gradient.
    From parent grain simulate a list of transformations (deduced by a parametric variation)

    :param Transform_params: list of grain simulation parameters defined by transformations with respect to the main grain orientation
    :param _list_param: list of parameters for each grain  [grain parameters, grain name]

    :param posCEN: tuple of Xcen, Ycen (detector geometry calibration)
    :param cameraAngles: tuple of Xbet, Xgam (detector geometry calibration)
    :param kf_direction: str, label to set the average position of the detecor plane with respect to the incoming beam ('Z>0',...)

    :return: (list_twicetheta,
                list_chi,
                list_energy,
                list_Miller,
                list_posX,
                list_posY,
                ParentGrainName_list,
                list_ParentGrain_transforms,
                calib,
                total_nb_grains)

    TODO:simulate for any camera position
    TODO: simulate spatial distribution of laue pattern origin
    """
    print("\n\n********* Starting dosimulation_parametric *********\n\n")

    # print('_list_param',_list_param)
    # print('SelectGrains',SelectGrains)
    # print('detectordistance',detectordistance)
    # print('detectordiameter',detectordiameter)
    # print('posCEN',posCEN)
    # print('cameraAngles',cameraAngles)
    # print('kf_direction',kf_direction)
    # print('pixelsize',pixelsize)

    # Number_ParentGrains parent grains
    Number_ParentGrains = len(_list_param)

    # Extracting parent grains  simluation param and name
    # creating list of simulation parameters for parent grains
    ListParam = []
    ParentGrainName_list = []
    for m in range(Number_ParentGrains):
        # print "_list_param[m]", _list_param[m]
        _paramsimul, _grainname = _list_param[m]

        elem = np.shape(np.array(_paramsimul[0]))
        # print "elem parametric",elem

        # convert EULER angles to matrix in case of  3 input elements
        if np.shape(np.array(_paramsimul[2])) == (3,):
            _paramsimul[2] = GT.fromEULERangles_toMatrix(_paramsimul[2])

        ListParam.append(_paramsimul)
        ParentGrainName_list.append(_grainname)

    #            print "ListParam in dosimulation_parametric", ListParam
    print("ParentGrainName_list", ParentGrainName_list)

    # Calculating Laue spots of each parent grain ----------------------------

    print("Doing simulation with %d parent grains" % Number_ParentGrains)
    list_twicetheta = []
    list_chi = []
    list_energy = []
    list_Miller = []
    list_posX = []
    list_posY = []

    total_nb_grains = 0

    # list of [Parent grain index,Number of corresponding transforms, transform_type]
    list_ParentGrain_transforms = []

    if gauge and WXPYTHON:
        gaugecount = 0
        gauge.SetValue(gaugecount)
        # gauge count max has been set to 1000*Number_ParentGrains

    # loop over parent grains
    for parentgrain_index in range(Number_ParentGrains):

        # read simulation parameters
        name_of_grain = ParentGrainName_list[parentgrain_index]
        print("\n\n %d, name_of_grain: %s" % (parentgrain_index, name_of_grain))

        Laue_classic_param = ListParam[parentgrain_index]

        key_material, Extinc, Ta, U, B, Tc = Laue_classic_param  # from combos

        # build GrainSimulParam
        GrainSimulParam = [0, 0, 0, 0]

        # user has entered his own B matrix
        if key_material == "inputB":
            # print "\n**************"
            # print "using Bmatrix containing lattice parameter"
            # print "****************\n"

            # take then parameters from combos
            GrainSimulParam[0] = B
            GrainSimulParam[1] = Extinc
            GrainSimulParam[2] = np.dot(np.dot(Ta, U), Tc)
            GrainSimulParam[3] = "inputB"

        # user uses a pre defined B matrix contain in material dictionnary
        # (need to read lattice parameter or element definition
        # then compute B matrix)
        elif key_material != "inputB":

            grain = CP.Prepare_Grain(key_material, np.eye(3), dictmaterials=dictmaterials)

            # print "grain in dosimulation_parametric() input Element",grain

            # B0, Extinc0, U0, key = grain  # U0 is identity
            B0, _, U0, _ = grain  # U0 is identity

            # new B matrix
            newB = np.dot(Tc, np.dot(B, B0))
            # Extinction is overwritten by value in comboExtinc
            newExtinc = Extinc
            # new U matrix
            newU = np.dot(Ta, np.dot(U, U0))

            GrainSimulParam = [newB, newExtinc, newU, key_material]

            print("Using following parameters from Material Dict.")
            print(DictLT.dict_Materials[key_material])

        # --- Simulate
        print("GrainSimulParam in dosimulation_parametric() input Element")
        print(GrainSimulParam)

        # q vectors in lauetools frame + miller indices
        spots2pi = LAUE.getLaueSpots(DictLT.CST_ENERGYKEV / emax,
                                        DictLT.CST_ENERGYKEV / emin,
                                        [GrainSimulParam],  # bracket because of a list of one grain
                                        fastcompute=0,
                                        verbose=0,
                                        kf_direction=kf_direction,
                                        dictmaterials=dictmaterials)

        # ---------  [list of 3D vectors],[list of corresponding Miller indices]

        # remove Parent Grain Laue spots too close from detector border vicinity
        Qvectors_ParentGrain, HKLs_ParentGrain = LAUE.filterQandHKLvectors(spots2pi,
                                                            detectordistance,
                                                            detectordiameter, kf_direction,
                                                            shiftcentercamera=cameraAngles[0])

        # Qvectors_ParentGrain, HKLs_ParentGrain = spots2pi

        if gauge and WXPYTHON:
            gaugecount += 100
            # print "gaugecount += 100",gaugecount
            gauge.SetValue(gaugecount)
            wx.Yield()

        # --- Calculating small deviations(rotations and strain)
        # --- from parent grains according to transform

        # print " in simul Transform_params",Transform_params
        # print " in simul SelectGrains",SelectGrains

        # get transform
        if Transform_params is None:
            Transform_listparam = [""]
        elif Transform_params is not None:
            Transform_params[""] = [""]

            # print "SelectGrains[name_of_grain]",SelectGrains[name_of_grain]
            Transform_listparam = Transform_params[SelectGrains[name_of_grain][7]]
            if Transform_listparam == "":
                Transform_listparam = [""]

        # print "Transform_listparam",Transform_listparam

        nb_transforms = 1
        print("GrainSimulParam", GrainSimulParam)
        matrix_list = [np.eye(3)]

        # matrix giving a*,b*,c* in absolute x, y,z frame
        # this matrix is used for strain a*,b*,c* are NORMALIZED:
        # this is a B matrix used in  q= U B G* formalism
        # this matrix represents also an initial orientation

        # Calculates matOrient which is U*B in q = U*B*Gstar
        matOrient = np.dot(GrainSimulParam[2], GrainSimulParam[0])

        # matOrient could be not pure rotation, so orientation of directions in which are expressed strain or rotation may be not fully accurate. (<1% if it comes from raw indexing)

        if Transform_listparam != "":
            # print "Transform_listparam[0]",Transform_listparam[0]
            if Transform_listparam[0] == "r_axis":
                axis_list = Transform_listparam[2]
                angle_list = Transform_listparam[1]
                nb_transforms = len(angle_list)

            elif Transform_listparam[0] in ("r_axis_d", "r_axis_d_slipsystem"):
                axis_list = Transform_listparam[2]
                # print "axis_list before orientation in d frame", axis_list
                angle_list = Transform_listparam[1]
                nb_transforms = len(angle_list)
                # axis coordinate change from abc frame(direct crystal) to a*b*c* frame( reciprocal crystal)
                axis_list_c = np.array([CP.fromrealframe_to_reciprocalframe(ax, GrainSimulParam[0])
                                                                    for ax in axis_list])
                #  print "axis_list_c", axis_list_c
                # axis coordinate change from a*b*c* frame(crystal) to absolute frame
                axis_list = np.dot(matOrient, axis_list_c.T).T
            #                 print "axis_list in absolute frame from d frame", axis_list

            elif Transform_listparam[0] in ("StackingFaults",):
                nb_transforms = len(Transform_listparam[1])
            # general transform expressed in absolute lauetools frame
            elif Transform_listparam[0] == "r_axis_c":
                axis_list = Transform_listparam[2]
                # print "axis_list before orientation in c frame",axis_list
                angle_list = Transform_listparam[1]
                nb_transforms = len(angle_list)
                # axis coordinate change from hkl frame(crystal) to absolute frame
                axis_list = np.dot(matOrient, axis_list.T).T
            #                 print "axis_list in absolute frame from c frame", axis_list

            # general transform expressed in absolute lauetools frame
            elif Transform_listparam[0] == "r_mat":
                matrix_list = Transform_listparam[1]
                nb_transforms = len(matrix_list)
                # general transform expressed in crystal frame

            elif Transform_listparam[0] == "r_mat_d":
                raise ValueError("r_mat_d matrix transform with d frame is not implemented yet !")

            elif Transform_listparam[0] == "r_mat_c":
                # print "using r_mat_c"
                matrix_list = Transform_listparam[1]
                nb_transforms = len(matrix_list)
                # then convert transform in absolute lauetools frame
                for k in range(nb_transforms):
                    matrix_list[k] = np.dot(
                        matOrient, np.dot(matrix_list[k], np.linalg.inv(matOrient)))
                    # matrix_list[k] = np.dot(inv(matOrient),np.dot(matrix_list[k],matOrient))

            # transform is a list of tensile or STRAIN transforms
            elif isinstance(Transform_listparam[0], list):
                # is a list of 's_axis' or 's_axis_c
                liststrainframe = Transform_listparam[0]
                # list of 3 arrays, each array contains the nb_transforms axis
                # (array of three elements)
                _axis_list = Transform_listparam[2]
                # list of 3 arrays, each array contains the nb_transforms strain factor
                factor_list = Transform_listparam[1]

                # print "axis_list in c frame",_axis_list
                # print "factor_list",factor_list
                nb_transforms = len(factor_list[0])

                axis_list = [np.ones(3) for k in range(3)]
                # loop over the three proposed axial strain in simulation board
                for mm in range(3):
                    if liststrainframe[mm] == "s_axis_c":
                        axis_list[mm] = np.dot(
                            matOrient, np.transpose(_axis_list[mm])).T
                    else:
                        axis_list[mm] = _axis_list[mm]
                # print "axis_list in a frame",axis_list
                # print "Transform_listparam[2]",Transform_listparam[2]

        #         print "HKLs_ParentGrain", HKLs_ParentGrain
        #         print "nb_transforms", nb_transforms
        #         print "Transform_listparam[0]", Transform_listparam[0]

        calib = [detectordistance, posCEN[0], posCEN[1], cameraAngles[0], cameraAngles[1]]
        # -----------------------------------------------------
        # loop over child grains derived from transformation of a single parent grain
        print('nb_transforms', nb_transforms)
        for ChildGrain_index in range(nb_transforms):
            # Qvectors_ParentGrain is used to create Qvectors_ParentGrain for each chold grain
            # according to the transform

            # print "Qvectors_ParentGrain", Qvectors_ParentGrain

            # Geometrical transforms for each case
            # loop over reciprocal lattice vectors is done with numpy array functions

            # for rotation around axis expressed in any frame
            if Transform_listparam[0] in ("r_axis", "r_axis_c", "r_axis_d", "r_axis_d_slipsystem"):
                # print "angle, axis",angle_list[ChildGrain_index],axis_list[ChildGrain_index]
                qvectors_ChildGrain = GT.rotate_around_u(Qvectors_ParentGrain[0],
                                                            angle_list[ChildGrain_index],
                                                            u=axis_list[ChildGrain_index])
                
                print('qvectors_ChildGrain',qvectors_ChildGrain)
                # list of spot which are on camera(without harmonics)
                # hkl are common to all child grains
                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain

            elif Transform_listparam[0] in ("StackingFaults",):
                print('Qvectors_ParentGrain[0]', Qvectors_ParentGrain[0])
                qvectors_ChildGrain=Transform_listparam[1][ChildGrain_index]+Qvectors_ParentGrain[0]
 

                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain


            # for general transform expressed in any frame
            elif (Transform_listparam[0] == "r_mat" or
                            Transform_listparam[0] in ("r_mat_c", "r_mat_d") or
                            Transform_listparam == "" or
                            Transform_listparam == [""]):

                # general transformation is applied to q vector
                # expressed in lauetools absolute frame
                qvectors_ChildGrain = np.dot(
                    matrix_list[ChildGrain_index], Qvectors_ParentGrain[0].T).T

                # if 0:
                #     print(" 10 first transpose(Qvectors_ParentGrain[0])",
                #         Qvectors_ParentGrain[0].T[:, :10])
                #     print("%d / %d" % (ChildGrain_index, nb_transforms))
                #     print("current matrix", matrix_list[ChildGrain_index])
                #     print(np.shape(qvectors_ChildGrain))
                #     print("GrainSimulParam", GrainSimulParam)
                #     print("qvectors_ChildGrain", qvectors_ChildGrain[:10])
                # list of spot which are on camera(without harmonics)
                print('qvectors_ChildGrain.shape',qvectors_ChildGrain.shape)

                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain

            # RADIAL STRAIN: for the three consecutive axial strains
            elif isinstance(Transform_listparam[0], list):

                first_traction = GT.tensile_along_u(
                    Qvectors_ParentGrain[0],
                    factor_list[0][ChildGrain_index],
                    u=axis_list[0][ChildGrain_index])
                second_traction = GT.tensile_along_u(
                    first_traction,
                    factor_list[1][ChildGrain_index],
                    u=axis_list[1][ChildGrain_index])
                qvectors_ChildGrain = GT.tensile_along_u(
                    second_traction,
                    factor_list[2][ChildGrain_index],
                    u=axis_list[2][ChildGrain_index])
                # list of spots for a child grain (on camera + without harmonics)
                print('qvectors_ChildGrain.shape',qvectors_ChildGrain.shape)

                spots2pi = [qvectors_ChildGrain], HKLs_ParentGrain

            else:
                # no transformation
                pass

            # test whether there is at least one Laue spot in the camera
            for elem in spots2pi[0]:
                if len(elem) == 0:
                    print("There is at least one child grain without peaks on CCD camera for ChildGrain_index= %.3f"
                        % ChildGrain_index)
                    break

            # ---------------------------------
            # filter spots to keep those in camera, filter harmonics
            try:
                # print("kf_direction = (in dosimulationparametric)", kf_direction)
                if kf_direction == "Z>0" or isinstance(kf_direction, list):  # or isinstance(kf_direction, np.array):
                    Laue_spot_list = LAUE.filterLaueSpots(spots2pi,
                                                    fileOK=0,
                                                    fastcompute=0,
                                                    detectordistance=detectordistance,
                                                    detectordiameter=detectordiameter*1.2,  # * 1.2, # avoid losing some spots in large transformation
                                                    kf_direction=kf_direction,
                                                    HarmonicsRemoval=1,
                                                    pixelsize=pixelsize,
                                                    shiftcentercamera=cameraAngles[0])

                    # for elem in Laue_spot_list[0][:10]:
                    # print elem

                    # print "Laue_spot_list[0][0].Twicetheta"
                    # print Laue_spot_list[0][0].Twicetheta

                    if gauge and WXPYTHON:
                        gaugecount = gaugecount + int(900 / nb_transforms)
                        gauge.SetValue(gaugecount)
                        wx.Yield()
                        # print "ChildGrain_index 900%nb_transforms",ChildGrain_index, gaugecount

                    Listspots = Laue_spot_list[0]
                    twicetheta = [spot.Twicetheta for spot in Listspots]
                    chi = [spot.Chi for spot in Listspots]
                    energy = [spot.EwaldRadius * DictLT.CST_ENERGYKEV for spot in Listspots]
                    Miller_ind = [list(spot.Millers) for spot in Listspots]

                    calib = [detectordistance, posCEN[0], posCEN[1], cameraAngles[0], cameraAngles[1]]

                    # print("calib parameters in dosimulation_parametric")
                    # print(calib)
                    # print("pixelsize", pixelsize)
                    # print("framedim", framedim)

                    posx, posy = LTGeo.calc_xycam_from2thetachi(twicetheta,
                                                                    chi,
                                                                    calib,
                                                                    pixelsize=pixelsize,
                                                                    kf_direction=kf_direction)[:2]
                    # vecRR = [spot.Qxyz for spot in Laue_spot_list[0]] #uf_lab in JSM LaueTools frame

                    list_twicetheta.append(twicetheta)
                    list_chi.append(chi)
                    list_energy.append(energy)
                    list_Miller.append(Miller_ind)
                    list_posX.append(posx.tolist())
                    list_posY.append(posy.tolist())

                    # success = 1

                elif kf_direction in ("Y<0", "Y>0"):
                    # TODO: patch for test: detectordistance = 126.5

                    Laue_spot_list = LAUE.filterLaueSpots(spots2pi,
                                                            fileOK=0,
                                                            fastcompute=0,
                                                            detectordistance=detectordistance,
                                                            detectordiameter=detectordiameter,  # * 1.2, # avoid losing some spots in large transformation
                                                            kf_direction=kf_direction,
                                                            HarmonicsRemoval=1,
                                                            pixelsize=pixelsize)

                    # for elem in Laue_spot_list[0][:10]:
                    # print elem

                    # print "Laue_spot_list[0][0].Twicetheta"
                    # print Laue_spot_list[0][0].Twicetheta
                    if gauge and WXPYTHON:
                        gaugecount = gaugecount + 900 / nb_transforms
                        gauge.SetValue(gaugecount)
                        wx.Yield()
                        # print "ChildGrain_index 900%nb_transforms",ChildGrain_index, gaugecount

                    twicetheta = [spot.Twicetheta for spot in Laue_spot_list[0]]
                    chi = [spot.Chi for spot in Laue_spot_list[0]]
                    energy = [spot.EwaldRadius * DictLT.CST_ENERGYKEV
                        for spot in Laue_spot_list[0]]
                    Miller_ind = [list(spot.Millers) for spot in Laue_spot_list[0]]

                    calib = [detectordistance, posCEN[0], posCEN[1], cameraAngles[0], cameraAngles[1]]

                    # posx, posy = LTGeo.calc_xycam_from2thetachi(twicetheta, chi, calib,
                    #                     pixelsize=pixelsize,
                    #                     kf_direction=kf_direction)[:2]
                    print('Y>0 2theta ', twicetheta[:5])

                    posx = [spot.Xcam for spot in Laue_spot_list[0]]
                    posy = [spot.Ycam for spot in Laue_spot_list[0]]

                    print('Y>0 posx ', posx[:5])

                    list_twicetheta.append(twicetheta)
                    list_chi.append(chi)
                    list_energy.append(energy)
                    list_Miller.append(Miller_ind)
                    list_posX.append(posx)
                    list_posY.append(posy)

                    # success = 1

                elif kf_direction in ("X>0", "X<0"):  # transmission mode or back reflection mode

                    # print("spots2pi",spots2pi)
                    Laue_spot_list = LAUE.filterLaueSpots(spots2pi,
                                                    fileOK=0,
                                                    fastcompute=0,
                                                    detectordistance=detectordistance,
                                                    detectordiameter=detectordiameter*1.2,  # * 1.2, # avoid losing some spots in large transformation
                                                    kf_direction=kf_direction,
                                                    HarmonicsRemoval=1,
                                                    pixelsize=pixelsize)

                    if Laue_spot_list is None:
                        list_twicetheta.append([])
                        list_chi.append([])
                        list_energy.append([])
                        list_Miller.append([])
                        list_posX.append([])
                        list_posY.append([])
                        continue

                    # print("Laue_spot_list",Laue_spot_list)
                    # for elem in Laue_spot_list[0]:
                    #     print('transmission spots',elem)

                    # print "Laue_spot_list[0][0].Twicetheta"
                    # print Laue_spot_list[0][0].Twicetheta
                    if gauge and WXPYTHON:
                        gaugecount = gaugecount + 900 / nb_transforms
                        gauge.SetValue(gaugecount)
                        wx.Yield()
                        # print "ChildGrain_index 900%nb_transforms",ChildGrain_index, gaugecount

                    twicetheta = [spot.Twicetheta for spot in Laue_spot_list[0]]
                    chi = [spot.Chi for spot in Laue_spot_list[0]]
                    energy = [spot.EwaldRadius * DictLT.CST_ENERGYKEV for spot in Laue_spot_list[0]]
                    Miller_ind = [list(spot.Millers) for spot in Laue_spot_list[0]]

                    posx, posy = LTGeo.calc_xycam_from2thetachi(twicetheta, chi, calib,
                                        pixelsize=pixelsize,
                                        kf_direction=kf_direction)[:2]
                    posx = posx.tolist()
                    posy = posy.tolist()

                    list_twicetheta.append(twicetheta)
                    list_chi.append(chi)
                    list_energy.append(energy)
                    list_Miller.append(Miller_ind)
                    list_posX.append(posx)
                    list_posY.append(posy)

                    # success = 1

            except UnboundLocalError:
                txt = "With theses parameters, there are no peaks in the CCD frame!!\n"
                txt += "for transform with t = %.3f\n" % ChildGrain_index
                txt += "It may seem that the transform you have designed has a too large amplitude\n"
                txt += "-Try then to reduce the variation range of t\n"
                txt += "-Or reduce ratio between extrema in input matrix transform\n\n"

                # success = 0
                break

            # end of loop over transforms(or children grains)

        transform_type = "parametric"
        print("Transform_listparam in dosimultion", Transform_listparam)

        if not isinstance(Transform_listparam[0], list):
            if Transform_listparam[0].endswith("slipsystem"):
                transform_type = "slipsystem"

        list_ParentGrain_transforms.append([parentgrain_index, nb_transforms, transform_type])
        total_nb_grains += nb_transforms

        if gauge and WXPYTHON:
            gaugecount += (parentgrain_index + 1) * 1000
            gauge.SetValue(gaugecount)
            wx.Yield()
            # print "(parentgrain_index+1)*1000",gaugecount

    # end of loop over parent grains-------------------------------

    print("total_nb_grains (cumulated from single and grains assembly):", total_nb_grains)

    # 1 grain data lists are listed i.e. use list_twicetheta[0] etc.
    # polygrain use list_twicetheta

    print("List of Grain Name", ParentGrainName_list)
    print("Number_ParentGrains of parent grains", Number_ParentGrains)
    print("Number_ParentGrains of spots in grain0", len(list_twicetheta[0]))
    print('list_ParentGrain_transforms', list_ParentGrain_transforms)

    data = (list_twicetheta,
        list_chi,
        list_energy,
        list_Miller,
        list_posX,
        list_posY,
        ParentGrainName_list,
        list_ParentGrain_transforms,
        calib,
        total_nb_grains)

    return data

if __name__ == "__main__":
    print('test of multigrainSimulator.py')
