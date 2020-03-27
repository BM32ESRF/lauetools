import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import FitOrient as FitO
import numpy as np
import generaltools as GT
import dict_LaueTools as DictLT

if 1:
    res = FitO.test_generalfitfunction()

if 0:
    # strain ezz   0.8 % expansion
    xy_unstrained = np.array(
        [
            [1864.1964, 528.2291],
            [1483.1531, 831.0652],
            [1367.9447, 509.2246],
            [1297.8918, 343.6432],
            [531.3408, 828.5554],
            [649.4996, 503.518],
            [720.9098, 337.4965],
            [132.9455, 514.9325],
            [1573.9556, 1172.6193],
            [1011.1447, 1646.7052],
            [1011.0166, 997.3441],
            [1010.93, 736.8167],
            [437.578, 1176.0942],
            [1212.8536, 1159.3779],
            [807.9199, 1160.5224],
        ]
    )

    xy_strained = np.array(
        [
            [1874.9749, 523.2668],
            [1487.7885, 828.8836],
            [1372.0172, 503.281],
            [1301.4263, 335.3735],
            [526.3259, 826.4333],
            [645.0339, 497.5334],
            [716.9751, 329.15],
            [121.3076, 509.8661],
            [1578.7537, 1173.9111],
            [1011.0582, 1651.2125],
            [1010.9028, 995.9055],
            [1010.7975, 733.1437],
            [432.4181, 1177.5707],
            [1214.3957, 1159.399],
            [806.1505, 1160.6082],
        ]
    )

    Miller_indices = np.array(
        [
            [1.0, -3.0, 7.0],
            [1.0, -1.0, 3.0],
            [1.0, -1.0, 5.0],
            [1.0, -1.0, 7.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 7.0],
            [1.0, 3.0, 7.0],
            [2.0, -2.0, 4.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, 4.0],
            [2.0, 0.0, 6.0],
            [2.0, 2.0, 4.0],
            [3.0, -1.0, 5.0],
            [3.0, 1.0, 5.0],
        ],
        dtype=np.int8,
    )

    # in sample frame
    # [[1,0.002,-0.0305],
    # [0,.987,0.001],
    # [0,0,1.005]]
    xy_transformed = np.array(
        [
            [1880.6296, 452.7668],
            [1488.0745, 772.5628],
            [1372.5274, 444.8035],
            [1301.7704, 273.0066],
            [523.5532, 773.0022],
            [641.1567, 441.0898],
            [712.7738, 268.3781],
            [1578.4334, 1111.5036],
            [1010.0983, 1570.1612],
            [1009.5975, 945.137],
            [1009.2564, 682.6361],
            [431.0182, 1119.0867],
            [1213.2493, 1103.6393],
            [805.001, 1106.403],
        ]
    )

    Miller_indices = np.array(
        [
            [1.0, -3.0, 7.0],
            [1.0, -1.0, 3.0],
            [1.0, -1.0, 5.0],
            [1.0, -1.0, 7.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 7.0],
            [1.0, 3.0, 7.0],
            [2.0, -2.0, 4.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, 4.0],
            [2.0, 0.0, 6.0],
            [2.0, 2.0, 4.0],
            [3.0, -1.0, 5.0],
            [3.0, 1.0, 5.0],
        ],
        dtype=np.int8,
    )

    calibparameters = [70.0, 1024, 1024, 0, 0]
    pixelsize = 0.079142

    # only for xy_transformed
    Miller_indices = np.array(
        [
            [1.0, -3.0, 7.0],
            [1.0, -1.0, 3.0],
            [1.0, -1.0, 5.0],
            [1.0, -1.0, 7.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 7.0],
            [2.0, -2.0, 4.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, 4.0],
            [2.0, 0.0, 6.0],
            [2.0, 2.0, 4.0],
            [3.0, -1.0, 5.0],
            [3.0, 1.0, 5.0],
        ],
        dtype=np.int8,
    )

    # unstrained
    Xexp, Yexp = xy_unstrained.T
    # strained
    Xexp, Yexp = xy_strained.T
    # general transform
    # must found : # [0.002,-0.0305, .987,0.001,1.005]
    Xexp, Yexp = xy_transformed.T

    absolutespotsindices = np.arange(len(Xexp))
    UB_raw = np.array(
        [
            [3.01902932e-01, 2.83134021e-04, -9.53338848e-01],
            [1.36290076e-02, 9.99896715e-01, 4.61298957e-03],
            [9.53241483e-01, -1.43857343e-02, 3.01867885e-01],
        ]
    )

    B0matrix = 1.0 / 3.6 * np.eye(3)

    varying_parameters_keys = [
        "anglex",
        "angley",
        "anglez",
        "Ts01",
        "Ts02",
        "Ts11",
        "Ts12",
        "Ts22",
    ]
    varying_parameters_values_array = [0.0, -0, 0.0, 0.005, -0.001, 1.002, 0.02, 1.03]
    #         varying_parameters_values_array = [0., -0, 0.,
    #                                            0.002, -0.0305, .987, 0.001, 1.005]

    allparameters = calibparameters + [
        0,
        0,
        0,  # 3 misorientation / initial UB matrix
        1,
        0,
        0,
        1,
        0,
        1,
    ]  # 5 strain compenents

    print(
        FitO.error_function_strain(
            varying_parameters_values_array,
            varying_parameters_keys,
            Miller_indices,
            allparameters,
            absolutespotsindices,
            Xexp,
            Yexp,
            initrot=UB_raw,
            B0matrix=B0matrix,
            pureRotation=0,
            verbose=1,
            pixelsize=pixelsize,
            dim=(2048, 2048),
            weights=None,
            returnalldata=True,
        )
    )

    FitO.fit_function_strain(
        varying_parameters_values_array,
        varying_parameters_keys,
        Miller_indices,
        allparameters,
        absolutespotsindices,
        Xexp,
        Yexp,
        UBmatrix_start=UB_raw,
        B0matrix=B0matrix,
        nb_grains=1,
        pureRotation=0,
        verbose=0,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
    )


if 0:  # Ge example unstrained
    pixX = np.array(
        [
            1027.1099965580365,
            1379.1700028337193,
            1288.1100055910788,
            926.219994375393,
            595.4599989710869,
            1183.2699986884652,
            1672.670001029018,
            1497.400007802548,
            780.2700069727559,
            819.9099991880139,
            873.5600007021501,
            1579.39000403102,
            1216.4900044928474,
            1481.199997684615,
            399.87000836895436,
            548.2499911593322,
            1352.760007116035,
            702.5200057620646,
            383.7700117705855,
            707.2000052800154,
            1140.9300043834062,
            1730.3299981313016,
            289.68999155533413,
            1274.8600008806216,
            1063.2499947675371,
            1660.8600022917144,
            1426.670005812432,
        ]
    )
    pixY = np.array(
        [
            1293.2799953573963,
            1553.5800003037994,
            1460.1599988550274,
            872.0599978043742,
            876.4400033114814,
            598.9200007214372,
            1258.6199918206175,
            1224.7000037967478,
            1242.530005349013,
            552.8399954684833,
            706.9700021553684,
            754.63000554209,
            1042.2800069222762,
            364.8400055136739,
            1297.1899933698528,
            1260.320007366279,
            568.0299942819768,
            949.8800073732916,
            754.580011319991,
            261.1099917270594,
            748.3999917806088,
            1063.319998717625,
            945.9700059216573,
            306.9500110237749,
            497.7900029269757,
            706.310001700921,
            858.780004244009,
        ]
    )
    miller_indices = np.array(
        [
            [3.0, 3.0, 3.0],
            [2.0, 4.0, 2.0],
            [3.0, 5.0, 3.0],
            [5.0, 3.0, 3.0],
            [6.0, 2.0, 4.0],
            [6.0, 4.0, 2.0],
            [3.0, 5.0, 1.0],
            [4.0, 6.0, 2.0],
            [5.0, 3.0, 5.0],
            [7.0, 3.0, 3.0],
            [4.0, 2.0, 2.0],
            [5.0, 5.0, 1.0],
            [5.0, 5.0, 3.0],
            [7.0, 5.0, 1.0],
            [5.0, 1.0, 5.0],
            [3.0, 1.0, 3.0],
            [8.0, 6.0, 2.0],
            [7.0, 3.0, 5.0],
            [5.0, 1.0, 3.0],
            [9.0, 3.0, 3.0],
            [7.0, 5.0, 3.0],
            [5.0, 7.0, 1.0],
            [7.0, 1.0, 5.0],
            [5.0, 3.0, 1.0],
            [9.0, 5.0, 3.0],
            [7.0, 7.0, 1.0],
            [3.0, 3.0, 1.0],
        ]
    )
    starting_orientmatrix = np.array(
        [
            [-0.9727538909589738, -0.21247913537718385, 0.09274958034159074],
            [0.22567394392094073, -0.7761682018781203, 0.5887564805829774],
            [-0.053107604650232926, 0.593645098498364, 0.8029726516869564],
        ]
    )
    #         B0matrix = np.array([[0.17675651789659746, -2.8424615990749217e-17, -2.8424615990749217e-17],
    #                            [0.0, 0.17675651789659746, -1.0823215193524997e-17],
    #                            [0.0, 0.0, 0.17675651789659746]])
    pixelsize = 0.08057
    calibparameters = [69.196, 1050.78, 1116.22, 0.152, -0.251]

    absolutespotsindices = np.arange(len(pixY))

    varying_parameters_keys = [
        "anglex",
        "angley",
        "anglez",
        "a",
        "b",
        "alpha",
        "beta",
        "gamma",
    ]
    varying_parameters_values_array = [0.0, -0, 0.0, 5.678, 5.59, 89.999, 90, 90.0001]

    allparameters = calibparameters + [
        0,
        0,
        0,  # 3 misorientation / initial UB matrix
        5.67,
        5.67,
        5.67,
        90.0,
        90.0,
        90.0,
    ]  # lattice parameters

    pureUmatrix, residualdistortion = GT.UBdecomposition_RRPP(starting_orientmatrix)

    print("len(allparameters)", len(allparameters))
    print("starting_orientmatrix", starting_orientmatrix)
    print("pureUmatrix", pureUmatrix)

    #         error_function_latticeparameters(varying_parameters_values_array, varying_parameters_keys,
    #                                                 miller_indices,
    #                                                 allparameters,
    #                                                 absolutespotsindices, pixX, pixY,
    #                                                 initrot=pureUmatrix,
    #                                                 pureRotation=0,
    #                                                 verbose=0,
    #                                                 pixelsize=pixelsize,
    #                                                 dim=(2048, 2048),
    #                                                 weights=None,
    #                                                 returnalldata=False)

    FitO.fit_function_latticeparameters(
        varying_parameters_values_array,
        varying_parameters_keys,
        miller_indices,
        allparameters,
        absolutespotsindices,
        pixX,
        pixY,
        UBmatrix_start=pureUmatrix,
        nb_grains=1,
        pureRotation=0,
        verbose=0,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
    )


if 0:  # cu example unstrained

    hkl = np.array(
        [
            [1.0, -3.0, 7.0],
            [1.0, -1.0, 3.0],
            [1.0, -1.0, 5.0],
            [1.0, -1.0, 7.0],
            [1.0, -1.0, 9.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 7.0],
            [1.0, 1.0, 9.0],
            [1.0, 3.0, 7.0],
            [2.0, -2.0, 4.0],
            [2.0, -2.0, 8.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, 4.0],
            [2.0, 0.0, 6.0],
            [2.0, 0.0, 8.0],
            [2.0, 2.0, 4.0],
            [2.0, 2.0, 8.0],
            [3.0, -1.0, 3.0],
            [3.0, -1.0, 5.0],
            [3.0, -1.0, 7.0],
            [3.0, 1.0, 3.0],
            [3.0, 1.0, 5.0],
            [3.0, 1.0, 7.0],
        ],
        dtype=np.int8,
    )

    # unstrained
    xyg1 = np.array(
        [
            [1864.1964, 528.2291],
            [1483.1531, 831.0652],
            [1367.9447, 509.2246],
            [1297.8918, 343.6432],
            [1250.7968, 240.1083],
            [531.3408, 828.5554],
            [649.4996, 503.518],
            [720.9098, 337.4965],
            [768.7331, 234.1588],
            [132.9455, 514.9325],
            [1573.9556, 1172.6193],
            [1417.5544, 638.138],
            [1011.1447, 1646.7052],
            [1011.0166, 997.3441],
            [1010.93, 736.8167],
            [1010.8677, 577.4817],
            [437.578, 1176.0942],
            [598.7294, 633.3692],
            [1245.1782, 1703.9742],
            [1212.8536, 1159.3779],
            [1188.3552, 908.1057],
            [775.3161, 1709.6555],
            [807.9199, 1160.5224],
            [832.584, 907.6238],
        ]
    )

    tthchig1 = np.array(
        [
            [112.1164, -43.529],
            [100.9569, -274347],
            [118.477, -21.2493],
            [126.308, -17.2058],
            [130.6459, -14.3817],
            [100.9262, 29.1178],
            [118.4526, 22.9483],
            [126.2879, 18.9152],
            [130.6289, 16.0984],
            [112.071, 45.212],
            [81.879, -31.8725],
            [111.7312, -23.9868],
            [54.8561, 0.8327],
            [91.726, 0.841],
            [107.9862, 0.8466],
            [116.7836, 0.8506],
            [81.844, 33.5447],
            [111.704, 25.6788],
            [53.2839, -14.0397],
            [81.4869, -12.0527],
            [97.3407, -10.5266],
            [53.2673, 15.704],
            [81.4725, 13.7285],
            [97.3279, 12.2113],
        ]
    )

    calibparameters = [70.0, 1024, 1024, 0, 0]

    pixelsize = 0.079142

    absolutespotsindices = np.arange(len(xyg1))
    UB_raw = np.array(
        [
            [3.01902932e-01, 2.83134021e-04, -9.53338848e-01],
            [1.36290076e-02, 9.99896715e-01, 4.61298957e-03],
            [9.53241483e-01, -1.43857343e-02, 3.01867885e-01],
        ]
    )

    B0matrix = 1.0 / 3.6 * np.eye(3)

    varying_parameters_keys = [
        "anglex",
        "angley",
        "anglez",
        "Tc11",
        "Tc22",
        "Tc01",
        "Tc02",
        "Tc12",
    ]
    varying_parameters_values_array = [0.2, -1.01, 0.3, 0.955, 0.99, -0.1, 0.2, 0.01]

    #     varying_parameters_keys = ['anglex', 'angley', 'anglez', 'Tc22', 'Tc01', 'Tc02', 'Tc12']
    #     varying_parameters_values_array = [.2, -1.01, .3, 1.005, 0.1, 0.2, 0.01]

    all_keys = [
        "anglex",
        "angley",
        "anglez",
        "Tc00",
        "Tc01",
        "Tc02",
        "Tc10",
        "Tc11",
        "Tc12",
        "Tc20",
        "Tc21",
        "Tc22",
        "T00",
        "T01",
        "T02",
        "T10",
        "T11",
        "T12",
        "T20",
        "T21",
        "T22",
        "Ts00",
        "Ts01",
        "Ts02",
        "Ts10",
        "Ts11",
        "Ts12",
        "Ts20",
        "Ts21",
        "Ts22",
    ]

    latticeparameters = DictLT.dict_Materials["Cu"][1]

    transformparameters = [
        0,
        0,
        0,  # 3 misorientation / initial UB matrix
        1.0,
        0,
        -0.0,
        -0.223145,
        1.01,
        0,
        0,
        -0.0,
        1,  # Tc
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,  # T
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ]  # Ts
    sourcedepth = [0]
    allparameters = (
        calibparameters + transformparameters + latticeparameters + sourcedepth
    )

    print("len(allparameters)", len(allparameters))

    FitO.error_function_general(
        varying_parameters_values_array,
        varying_parameters_keys,
        hkl,
        allparameters,
        absolutespotsindices,
        xyg1[:, 0],
        xyg1[:, 1],
        initrot=UB_raw,
        B0matrix=B0matrix,
        pureRotation=0,
        verbose=0,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
        returnalldata=False,
    )

    FitO.fit_function_general(
        varying_parameters_values_array,
        varying_parameters_keys,
        hkl,
        allparameters,
        absolutespotsindices,
        xyg1[:, 0],
        xyg1[:, 1],
        UBmatrix_start=UB_raw,
        B0matrix=B0matrix,
        nb_grains=1,
        pureRotation=0,
        verbose=0,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
    )


if 0:
    hkl = np.array(
        [
            [1.0, -3.0, 7.0],
            [1.0, -1.0, 3.0],
            [1.0, -1.0, 5.0],
            [1.0, -1.0, 7.0],
            [1.0, -1.0, 9.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 7.0],
            [1.0, 1.0, 9.0],
            [1.0, 3.0, 7.0],
            [2.0, -2.0, 4.0],
            [2.0, -2.0, 8.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, 4.0],
            [2.0, 0.0, 6.0],
            [2.0, 0.0, 8.0],
            [2.0, 2.0, 4.0],
            [2.0, 2.0, 8.0],
            [3.0, -1.0, 3.0],
            [3.0, -1.0, 5.0],
            [3.0, -1.0, 7.0],
            [3.0, 1.0, 3.0],
            [3.0, 1.0, 5.0],
            [3.0, 1.0, 7.0],
        ],
        dtype=np.int8,
    )

    # unstrained
    xyg1 = np.array(
        [
            [1864.1964, 528.2291],
            [1483.1531, 831.0652],
            [1367.9447, 509.2246],
            [1297.8918, 343.6432],
            [1250.7968, 240.1083],
            [531.3408, 828.5554],
            [649.4996, 503.518],
            [720.9098, 337.4965],
            [768.7331, 234.1588],
            [132.9455, 514.9325],
            [1573.9556, 1172.6193],
            [1417.5544, 638.138],
            [1011.1447, 1646.7052],
            [1011.0166, 997.3441],
            [1010.93, 736.8167],
            [1010.8677, 577.4817],
            [437.578, 1176.0942],
            [598.7294, 633.3692],
            [1245.1782, 1703.9742],
            [1212.8536, 1159.3779],
            [1188.3552, 908.1057],
            [775.3161, 1709.6555],
            [807.9199, 1160.5224],
            [832.584, 907.6238],
        ]
    )

    tthchig1 = np.array(
        [
            [112.1164, -43.529],
            [100.9569, -274347],
            [118.477, -21.2493],
            [126.308, -17.2058],
            [130.6459, -14.3817],
            [100.9262, 29.1178],
            [118.4526, 22.9483],
            [126.2879, 18.9152],
            [130.6289, 16.0984],
            [112.071, 45.212],
            [81.879, -31.8725],
            [111.7312, -23.9868],
            [54.8561, 0.8327],
            [91.726, 0.841],
            [107.9862, 0.8466],
            [116.7836, 0.8506],
            [81.844, 33.5447],
            [111.704, 25.6788],
            [53.2839, -14.0397],
            [81.4869, -12.0527],
            [97.3407, -10.5266],
            [53.2673, 15.704],
            [81.4725, 13.7285],
            [97.3279, 12.2113],
        ]
    )

    # strain ezz   0.5 % expansion
    xyg2 = np.array(
        [
            [1877.6856, 522.0243],
            [1488.9504, 828.3393],
            [1373.0401, 501.7915],
            [1302.3153, 333.2976],
            [1254.6638, 227.7165],
            [525.0688, 825.9042],
            [643.9121, 496.0335],
            [715.9854, 327.0547],
            [764.3583, 221.6543],
            [118.3798, 508.5973],
            [1579.9543, 1174.2373],
            [1423.0134, 632.7145],
            [1011.0366, 1652.3422],
            [1010.8743, 995.5458],
            [1010.7643, 732.2246],
            [1010.6848, 570.8221],
            [431.1269, 1177.9435],
            [592.7848, 627.9403],
            [1247.0364, 1710.8555],
            [1214.7814, 1159.4047],
            [1190.2586, 905.5328],
            [773.2221, 1716.7161],
            [805.7081, 1160.6302],
            [830.3609, 905.0903],
        ]
    )

    calibparameters = [70.0, 1024, 1024, 0, 0]

    pixelsize = 0.079142

    g1strain = [1, 1, 0, 0, 0]
    g1rot = [0, 0, 0]
    g2strain = [1, 1.005, 0, 0, 0]

    allparameters = (
        calibparameters + g1strain + g1rot + g2strain
    )  # g1 g2 UB + g2 strain

    #     # unstrained Cu grain
    #     nb_grains = 1
    #     arr_indexvaryingparameters = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    #     starting_param = [0.9998, 1.0001, 0.001, -0.002, 0.001, -0.03, 0.01, 0.05]

    # strained Cu grain
    nb_grains = 1
    arr_indexvaryingparameters = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    starting_param = [0.9998, 1.0001, 0.001, -0.002, 0.001, -0.03, 0.01, 0.05]
    xyg1 = xyg2

    #     nb_grains = 2
    #     arr_indexvaryingparameters = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    #     starting_param = [0.9998, 1.0001, 0.001, -0.002, 0.001, -0.03, 0.01, 0.05, 1, 1.005, 0, 0, 0]

    miller = hkl
    absolutespotsindices = [np.arange(len(xyg1)), np.arange(len(xyg2))]
    pixX = [xyg1[:, 0], xyg2[:, 0]]
    pixY = [xyg1[:, 1], xyg2[:, 1]]

    UB_raw = np.array(
        [
            [3.01902932e-01, 2.83134021e-04, -9.53338848e-01],
            [1.36290076e-02, 9.99896715e-01, 4.61298957e-03],
            [9.53241483e-01, -1.43857343e-02, 3.01867885e-01],
        ]
    )

    B0matrix = 1.0 / 3.6 * np.eye(3)

    res = FitO.error_function_on_demand_strain_2grains(
        g1strain + g1rot + g2strain,
        miller,
        allparameters,
        arr_indexvaryingparameters,
        absolutespotsindices,
        pixX,
        pixY,
        initrot=UB_raw,
        B0matrix=B0matrix,
        nb_grains=nb_grains,
        pureRotation=0,
        verbose=1,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
        returnalldata=True,
    )

    print("alldistances_array,all_deltamatrices,all_newmatrices", res)

    strain_sol = FitO.fit_on_demand_strain_2grains(
        np.array(starting_param),
        miller,
        allparameters,
        FitO.error_function_on_demand_strain_2grains,
        arr_indexvaryingparameters,
        absolutespotsindices,
        pixX,
        pixY,
        initrot=UB_raw,
        B0matrix=1.0 / 3.6 * np.eye(3),
        nb_grains=nb_grains,
        pureRotation=0,
        verbose=0,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
    )

    print("\n\n\n\nStrain refinement completed !!! \n\n\n\n")

    fit_results = FitO.error_function_on_demand_strain_2grains(
        strain_sol,
        miller,
        allparameters,
        arr_indexvaryingparameters,
        absolutespotsindices,
        pixX,
        pixY,
        initrot=UB_raw,
        B0matrix=1.0 / 3.6 * np.eye(3),
        nb_grains=nb_grains,
        pureRotation=0,
        verbose=1,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
        returnalldata=True,
    )

    if nb_grains >= 2:
        alldistances_array, all_deltamatrices, all_newmatrices = fit_results

    elif nb_grains == 1:
        alldistances_array = fit_results[0]
        refinedUB = fit_results[2][0]
        finalB = np.eye(3)
        for k, ind_par in enumerate(arr_indexvaryingparameters):
            refined_value = strain_sol[k]
            if ind_par == 5:
                finalB[1, 1] = refined_value
            elif ind_par == 6:
                finalB[2, 2] = refined_value
            elif ind_par == 7:
                finalB[0, 1] = refined_value
            elif ind_par == 8:
                finalB[0, 2] = refined_value
            elif ind_par == 9:
                finalB[1, 2] = refined_value

        print("refinedUB", refinedUB)
        print("finalB", finalB)

if 0:
    # ge
    xyGe = np.array(
        [
            [1027.11, 1293.28],
            [1379.17, 1553.58],
            [1288.11, 1460.16],
            [926.22, 872.06],
            [595.46, 876.44],
            [1183.27, 598.92],
            [626.12, 1661.28],
            [856.14, 1702.52],
            [1672.67, 1258.62],
            [1497.4, 1224.7],
            [1176.09, 1086.19],
            [780.27, 1242.53],
            [828.68, 1246.06],
            [819.91, 552.84],
            [873.56, 706.97],
            [1085.8, 1725.83],
            [1579.39, 754.63],
            [1216.49, 1042.28],
            [1481.2, 364.84],
            [399.87, 1297.19],
            [548.25, 1260.32],
            [1396.71, 1217.56],
            [1573.49, 1073.13],
            [1199.06, 1387.53],
            [1352.76, 568.03],
            [467.05, 984.6],
            [638.87, 1246.59],
            [967.15, 1019.9],
            [977.5, 1060.31],
            [702.52, 949.88],
            [383.77, 754.58],
            [1283.34, 976.69],
            [707.2, 261.11],
            [1140.93, 748.4],
            [701.05, 1560.17],
            [1730.33, 1063.32],
            [905.64, 804.81],
            [1055.14, 634.64],
            [703.42, 801.02],
            [289.69, 945.97],
            [1151.05, 1115.85],
            [1339.11, 926.87],
            [1458.76, 556.69],
            [861.48, 1250.42],
            [327.89, 1321.03],
            [621.54, 713.5],
            [1274.86, 306.95],
            [1280.51, 691.44],
            [1063.25, 497.79],
            [1520.65, 1382.65],
            [1765.96, 958.47],
            [1660.86, 706.31],
            [1101.37, 907.29],
            [1075.86, 1633.06],
            [1473.72, 1088.4],
            [1819.62, 1305.46],
            [1626.9, 982.64],
            [576.2, 1016.97],
            [701.5, 1242.55],
            [1426.67, 858.78],
            [911.32, 1538.14],
            [841.45, 613.02],
            [1219.19, 482.3],
            [1588.65, 403.05],
            [521.92, 1574.92],
            [810.65, 1039.11],
            [982.11, 264.38],
            [537.37, 496.91],
            [393.71, 1057.16],
            [245.42, 1050.38],
            [1050.01, 728.62],
            [296.95, 544.57],
            [510.17, 824.17],
            [1285.92, 1221.22],
            [1090.08, 954.93],
            [1711.33, 678.47],
            [1377.04, 659.84],
            [775.11, 434.93],
            [757.84, 863.72],
            [983.89, 1090.23],
            [925.43, 261.1],
            [1244.11, 1013.22],
            [1530.97, 552.13],
        ]
    )

    gehkl = np.array(
        [
            [-3.0, 3.0, 3.0],
            [-2.0, 2.0, 4.0],
            [-3.0, 3.0, 5.0],
            [-5.0, 3.0, 3.0],
            [-6.0, 4.0, 2.0],
            [-6.0, 2.0, 4.0],
            [-4.0, 6.0, 2.0],
            [-3.0, 5.0, 3.0],
            [-3.0, 1.0, 5.0],
            [-4.0, 2.0, 6.0],
            [-6.0, 4.0, 6.0],
            [-5.0, 5.0, 3.0],
            [-6.0, 6.0, 4.0],
            [-7.0, 3.0, 3.0],
            [-4.0, 2.0, 2.0],
            [-3.0, 5.0, 5.0],
            [-5.0, 1.0, 5.0],
            [-5.0, 3.0, 5.0],
            [-7.0, 1.0, 5.0],
            [-5.0, 5.0, 1.0],
            [-3.0, 3.0, 1.0],
            [-5.0, 3.0, 7.0],
            [-6.0, 2.0, 8.0],
            [-5.0, 5.0, 7.0],
            [-8.0, 2.0, 6.0],
            [-8.0, 6.0, 2.0],
            [-7.0, 7.0, 3.0],
            [-7.0, 5.0, 5.0],
            [-8.0, 6.0, 6.0],
            [-7.0, 5.0, 3.0],
            [-5.0, 3.0, 1.0],
            [-8.0, 4.0, 8.0],
            [-9.0, 3.0, 3.0],
            [-7.0, 3.0, 5.0],
            [-5.0, 7.0, 3.0],
            [-5.0, 1.0, 7.0],
            [-9.0, 5.0, 5.0],
            [-10.0, 4.0, 6.0],
            [-10.0, 6.0, 4.0],
            [-7.0, 5.0, 1.0],
            [-7.0, 5.0, 7.0],
            [-7.0, 3.0, 7.0],
            [-10.0, 2.0, 8.0],
            [-7.0, 7.0, 5.0],
            [-7.0, 7.0, 1.0],
            [-9.0, 5.0, 3.0],
            [-5.0, 1.0, 3.0],
            [-9.0, 3.0, 7.0],
            [-9.0, 3.0, 5.0],
            [-5.0, 3.0, 9.0],
            [-7.0, 1.0, 9.0],
            [-7.0, 1.0, 7.0],
            [-9.0, 5.0, 7.0],
            [-4.0, 6.0, 6.0],
            [-7.0, 3.0, 9.0],
            [-5.0, 1.0, 9.0],
            [-8.0, 2.0, 10.0],
            [-9.0, 7.0, 3.0],
            [-8.0, 8.0, 4.0],
            [-3.0, 1.0, 3.0],
            [-5.0, 7.0, 5.0],
            [-11.0, 5.0, 5.0],
            [-11.0, 3.0, 7.0],
            [-9.0, 1.0, 7.0],
            [-6.0, 8.0, 2.0],
            [-9.0, 7.0, 5.0],
            [-11.0, 3.0, 5.0],
            [-11.0, 5.0, 3.0],
            [-10.0, 8.0, 2.0],
            [-9.0, 7.0, 1.0],
            [-11.0, 5.0, 7.0],
            [-12.0, 6.0, 2.0],
            [-11.0, 7.0, 3.0],
            [-7.0, 5.0, 9.0],
            [-10.0, 6.0, 8.0],
            [-9.0, 1.0, 9.0],
            [-11.0, 3.0, 9.0],
            [-13.0, 5.0, 5.0],
            [-11.0, 7.0, 5.0],
            [-9.0, 7.0, 7.0],
            [-14.0, 4.0, 6.0],
            [-9.0, 5.0, 9.0],
            [-12.0, 2.0, 10.0],
        ]
    )

    UBGe = np.array(
        [
            [0.97294265, 0.09236186, -0.21176475],
            [-0.22489545, 0.58960648, -0.77593813],
            [0.05318109, 0.80262461, 0.59457586],
        ]
    )
    B0Ge = np.array(
        [
            [1.76756518e-01, -2.84246160e-17, -2.84246160e-17],
            [0.00000000e00, 1.76756518e-01, -1.08232152e-17],
            [0.00000000e00, 0.00000000e00, 1.76756518e-01],
        ]
    )

    calibGe = [69.196, 1050.78, 1116.22, 0.152, -0.251]

    # --- ----------- input parameters
    calibparameters = calibGe
    nb_grains = 1
    g1strain = [1, 1, 0, 0, 0]
    g1rot = [0, 0, 0]
    #     g2strain = [1, 1.005, 0, 0, 0]

    allparameters = calibparameters + g1strain + g1rot  # + g2strain

    nbgrains = 1
    #     arr_indexvaryingparameters = np.arange(5, 5 + nbgrains * 5 + 3)
    #     starting_param = [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1.005, 0, 0, 0]
    arr_indexvaryingparameters = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    starting_param = [1, 1, 0, 0, 0, 0, 0, 0]

    miller = gehkl
    absolutespotsindices = [np.arange(len(xyGe)), np.arange(len(xyGe))]
    pixX = [xyGe[:, 0], xyGe[:, 0]]
    pixY = [xyGe[:, 1], xyGe[:, 1]]

    UB_raw = UBGe
    B0matrix = B0Ge

    # ----------------------------------------------------------
    res = FitO.error_function_on_demand_strain_2grains(
        g1strain + g1rot,
        miller,
        allparameters,
        arr_indexvaryingparameters,
        absolutespotsindices,
        pixX,
        pixY,
        initrot=UB_raw,
        B0matrix=B0matrix,
        nb_grains=nb_grains,
        pureRotation=0,
        verbose=1,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
        returnalldata=True,
    )

    print("alldistances_array,all_deltamatrices,all_newmatrices", res)

    strain_sol = FitO.fit_on_demand_strain_2grains(
        np.array(starting_param),
        miller,
        allparameters,
        FitO.error_function_on_demand_strain_2grains,
        arr_indexvaryingparameters,
        absolutespotsindices,
        pixX,
        pixY,
        initrot=UB_raw,
        B0matrix=1.0 / 3.6 * np.eye(3),
        nb_grains=nb_grains,
        pureRotation=0,
        verbose=0,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
    )

    print("\n\n\n\nStrain refinement completed !!! \n\n\n\n")

    fit_results = FitO.error_function_on_demand_strain_2grains(
        strain_sol,
        miller,
        allparameters,
        arr_indexvaryingparameters,
        absolutespotsindices,
        pixX,
        pixY,
        initrot=UB_raw,
        B0matrix=1.0 / 3.6 * np.eye(3),
        nb_grains=nb_grains,
        pureRotation=0,
        verbose=1,
        pixelsize=pixelsize,
        dim=(2048, 2048),
        weights=None,
        returnalldata=True,
    )

    if nb_grains >= 2:
        alldistances_array, all_deltamatrices, all_newmatrices = fit_results

    elif nb_grains == 1:
        alldistances_array = fit_results[0]
        refinedUB = fit_results[2][0]
        finalB = np.eye(3)
        for k, ind_par in enumerate(arr_indexvaryingparameters):
            refined_value = strain_sol[k]
            if ind_par == 5:
                finalB[1, 1] = refined_value
            elif ind_par == 6:
                finalB[2, 2] = refined_value
            elif ind_par == 7:
                finalB[0, 1] = refined_value
            elif ind_par == 8:
                finalB[0, 2] = refined_value
            elif ind_par == 9:
                finalB[1, 2] = refined_value

        print("refinedUB", refinedUB)
        print("finalB", finalB)


if 0:
    #     print "up to now, no test !"
    #     import sys
    #     sys.exit()

    # test of strain refinement of non cubic structure

    mat_manualindexation = np.array(
        [
            [0.16821451, -0.36210799, -0.91683242],
            [-0.42638856, 0.81196112, -0.39863761],
            [0.8905644, 0.45406003, -0.02691724],
        ]
    )

    mat_manualindexation = np.array(
        [
            [0.6431522, 0.20977096, -0.73644497],
            [-0.08385282, 0.97525494, 0.20456386],
            [0.76113324, -0.06981272, 0.64482731],
        ]
    )
    pixelsize = 165.0 / 2048
    if 0:
        filename_ind = "Ge_test.idx"
        calibration_parameter = [
            69.66221,
            895.29492,
            960.78674,
            0.84324,
            -0.32201,
        ]  # Nov 09 J. Villanova BM32

    ################################################################################

    filename_image = "Ge_run41_1_0003.mccd"
    filename_idx = "Ge_run41_1_0003.idx"

    calibration_parameter = [
        69.66055,
        895.27118,
        960.77417,
        0.8415,
        -0.31818,
    ]  # Nov 09 J. Villanova BM32

    if 0:
        # VHR camera 05 Mar 2010
        filename_idx = "toto.idx"
        pixelsize = 0.0168
        calibration_parameter = [41.7, 1780, 700.0, 0.84324, -0.32201]
        calibration_parameter = [32, 1100, 600, 0, 0]

    # Peak position estimation
    # 0 for fit2D centroids or default position in idx
    # 1 XMAS file (.ind)
    # 2 Gaussian fit
    # 3 lorentzian fit
    # 4 Centroid
    flag_pos = 0

    nb_steps = 60  # max number of iterations
    relative_error = 0.0000001  # pixdev relative error variation limit to stop the loop

    #################################################################################

    import readmccd as RMCCD

    filename_idx = "Custrained_1.idx"  # 'Ge_run41_1_0003.idx'#'testrotCu.idx'
    filename_idx = "Cu_3.idx"  # 'Ge_run41_1_0003.idx'#'testrotCu.idx'
    
    twthe, chi, pixX, pixY, Intensity, miller, energy, grainindex, grainUmatrix, grainBmatrix, grainEmatrix, detectorparam = RMCCD.read_indexationfile(filename_idx)

    if detectorparam:
        calibration_parameter = detectorparam

    starting_orientmatrix = mat_manualindexation
    starting_orientmatrix = grainUmatrix["0"]  # from .idx file

    print("starting_orientmatrix", starting_orientmatrix)

    allparameters = np.array(calibration_parameter + [0.0, 0.0, 0.0])
    DATA_Q = miller  # all miller indices must be entered in fit procedures

    # spots selection
    nspots = np.arange(13)  #  array([0,1,2,3,4,5,6,7,8,9,10])
    pixX, pixY = np.take(pixX, nspots), np.take(pixY, nspots)

    # find the best orientation with a pure rotation matrix (1) or product pure rotation* strain matrix (triangle up)  (0)
    pureRotation = 1

    if 0:
        print("c'est parti")
        Xcen = np.array([1000.0])
        Xcen_sol = FitO.fitXCEN(
            Xcen,
            miller,
            allparameters,
            FitO.error_function_XCEN,
            nspots,
            pixX,
            pixY,
            initrot=starting_orientmatrix,
            verbose=0,
        )

    if 0:
        print("c'est parti")
        initial_values = np.array([1000.0, 1200.0])  # XCEN and YCEN
        arr_indexvaryingparameters = np.array(
            [1, 2]
        )  # indices of position of parameters in [dd,xcen,ycen,ang1,ang2]
        results = FitO.fit_on_demand_calibration(
            initial_values,
            miller,
            allparameters,
            FitO.error_function_on_demand_calibration,
            arr_indexvaryingparameters,
            nspots,
            pixX,
            pixY,
            initrot=starting_orientmatrix,
            verbose=0,
        )

    if 0:
        print("c'est parti")
        initial_values = np.array([65.0])  # dd
        arr_indexvaryingparameters = np.array(
            [0]
        )  # indices of position of parameters in [dd,xcen,ycen,ang1,ang2]
        results = FitO.fit_on_demand_calibration(
            initial_values,
            miller,
            allparameters,
            FitO.error_function_on_demand_calibration,
            arr_indexvaryingparameters,
            nspots,
            pixX,
            pixY,
            initrot=starting_orientmatrix,
            verbose=0,
        )

    if 1:
        print("c'est parti")
        initial_values = np.array([65.0, 200, 900, 5, -3])  # [dd,xcen,ycen,ang1,ang2]
        arr_indexvaryingparameters = np.array(
            [0, 1, 2, 3, 4]
        )  # indices of position of parameters in [dd,xcen,ycen,ang1,ang2]
        results = FitO.fit_on_demand_calibration(
            initial_values,
            miller,
            allparameters,
            FitO.error_function_on_demand_calibration,
            arr_indexvaryingparameters,
            nspots,
            pixX,
            pixY,
            initrot=starting_orientmatrix,
            verbose=0,
        )

    raise ValueError("end of example")

    # fit of peaks (better than previous fit2d or LaueTools peaksearch) -------------------------------
    if flag_pos in (2, 3):
        if flag_pos == 2:
            fitfunc = "gaussian"
        if flag_pos == 3:
            fitfunc = "lorentzian"

        peaklist = np.transpose(np.array([pixX, pixY]))
        params, cov, info, message, baseline = IOimage.readoneimage_multiROIfit(
            filename_image,
            peaklist,
            20,
            CCDLabel="MARCCD165",
            baseline="auto",  # min in ROI box
            startangles=0.0,
            start_sigma1=1.0,
            start_sigma2=1.0,
            position_start="max",  # 'centers' or 'max'
            fitfunc=fitfunc,
            showfitresults=0,
            offsetposition=0,
        )  # =1 XMAS pixel convention

        pixX, pixY = np.transpose(np.array(params)[:, 2:4])
    # -----------------------------------------------------------------------------------------------

    if flag_pos == 4:
        import Centroid

        peaklist = np.transpose([pixX, pixY])
        tab_centroid = Centroid.read_multicentroid(filename_image, peaklist, boxsize=10)
        pixX, pixY = np.transpose(tab_centroid)
