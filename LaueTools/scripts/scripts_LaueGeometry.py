import sys
import os
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import LaueGeometry as LGeo

import IOLaueTools as IOLT

DEG = np.pi / 180.0

# --------------------   TEST & EXAMPLES functions ---------------------------------
def test_1():
    """
    test
    """
    abscissa = np.array([5, -5.0, 6.0, 0.001, 2.35, 52, 0.0])
    IIw0 = 10.0
    yoyo = LGeo.IW_from_wireabscissa(abscissa, IIw0, anglesample=40.0)
    x = LGeo.Wireabscissa_from_IW(yoyo[0], yoyo[1], IIw0, anglesample=40.0)
    print("abscissa", abscissa)
    print("x", x)
    print("This two arrays must be equal!")


def test_2():
    """
    test
    """
    calib = [70, 1024, 1024, 0.0, -0.0]
    uflab = np.array(
        [[0.5, -0.5, 0.5], [0.0, -0.0, 1.0], [0.0, -0.5, 0.5], [0.5, 0.5, 0.5]]
    )
    xcam, ycam, th = LGeo.OM_from_uf(uflab, calib, energy=0, offset=None, verbose=0)
    uflab_2, IMlab_2 = LGeo.calc_uflab(
        xcam, ycam, calib, returnAngles="uflab", verbose=0
    )
    print("****************")
    print("start", uflab)
    print("final", uflab_2)
    print("This two arrays must be equal!")


def test_3(anglesample, calib, IIprime):
    """
    test
    """
    IIprime = np.array(IIprime)
    IM = np.array([[1, 1, 1], [1, 0, 1], [1, -1, 1], [0, 1, 1], [0, 0, 5], [10, 5, 4]])
    nor = np.sqrt(np.sum(IM ** 2, axis=1))
    uf = IM * 1.0 / np.reshape(nor, (len(nor), 1))
    IpM = LGeo.IprimeM_from_uf(uf, IIprime, calib, verbose=0)
    print("IpM", IpM)

    height_wire = 10.0
    IW = LGeo.IW_from_IM_onesource(
        IIprime[1:], IpM, height_wire, anglesample=anglesample
    )
    print("IW", IW)

    print("finding source origin from two reflectionx")
    OM = LGeo.OM_from_uf(uf, calib, energy=0, offset=None, verbose=0)[:2]
    print("OM")
    IWs = np.transpose(IW)[:2]
    OMs = np.transpose(OM)[:2]
    print("Using 2 reflections")
    print("2 OMs", OMs)
    print("2 IWs", IWs)
    print("------------")
    ysource, zsource = LGeo.find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib)
    print("ysource,zsource", ysource, zsource)


def test_4(IIprime, height_wire, errorOM1, errorOM2):
    """
    test
    """
    anglesample = 40.0
    calib = [70, 1024, 1024, 0.0, -0.0]
    IIprime = np.array(IIprime)
    IM = np.array([[1, 1, 1], [1, 0, 1], [1, -1, 1], [0, 1, 1], [0, 0, 5], [10, 5, 4]])
    nor = np.sqrt(np.sum(IM ** 2, axis=1))
    uf = IM * 1.0 / np.reshape(nor, (len(nor), 1))
    IpM = LGeo.IprimeM_from_uf(uf, IIprime, calib, verbose=0)
    print("IpM", IpM)

    IW = LGeo.IW_from_IM_onesource(IIprime[1:], IpM, height_wire, anglesample=anglesample)
    print("IW", IW)

    print("finding source origin from two reflectionx")
    OM = LGeo.OM_from_uf(uf, calib, energy=0, offset=None, verbose=0)[:2]
    print("OM")

    print("Using 2 reflections")
    IWs = np.transpose(IW)[:2]
    OMs = np.transpose(OM)[:2]
    errorOM1, errorOM2 = np.array(errorOM1), np.array(errorOM2)
    OMs = OMs + np.array([errorOM1, errorOM1])
    print("2 OMs", OMs)
    print("2 IWs", IWs)
    print("------------")
    ysource, zsource = LGeo.find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib)
    print("ysource,zsource", ysource, zsource)


def test_5(IIprime, height_wire, arrayindex, errorWabscissa1, errorWabscissa2):
    """

    Simulate some reflections of a shifted source / calibrated emission point
    and corresponding wire abscissa

    wire is assumed to be parallel to sample surface (inclined by anglesample, see below) and strictly along 0x axis

    IIprime: 3 elements vector of source position (mm)
    height_wire: height of the wire (mm)
    errorWabscissa1,errorWabscissa2: error in measuring wireabscissa where the reflection is apparently extinguished

    Then retrieve the source position from 2 reflections and 2 measured wireabscissa

    """
    anglesample = 40.0
    calib = [100, 1024, 1024, 0.0, -0.0]
    IIprime = np.array(IIprime)
    IM = np.array([[1, 1, 3], [1, 0, 3], [1, -1, 3], [0, 1, 3], [0, 0, 5], [4, 5, 10]])
    nor = np.sqrt(np.sum(IM ** 2, axis=1))
    uf = IM * 1.0 / np.reshape(nor, (len(nor), 1))
    IpM = LGeo.IprimeM_from_uf(uf, IIprime, calib, verbose=0)
    print("IpM", IpM)

    IW = LGeo.IW_from_IM_onesource(IIprime[1:], IpM, height_wire, anglesample=anglesample)
    print("IW", IW)

    OM = LGeo.OM_from_uf(uf, calib, energy=0, offset=None, verbose=0)[:2]
    print("positions on CCD: OMs", OM)

    _twtheta, _chi = LGeo.calc_uflab(OM[0], OM[1], calib, returnAngles=1, verbose=0,
                                                                        pixelsize=165.0 / 2048)

    print("finding source origin from two reflections")
    print("Using 2 reflections of index:", arrayindex)
    # this is where two reflections can be chosen among others
    IWs = np.take(IW.T, np.array(arrayindex), axis=0)
    OMs = np.take(OM.T, np.array(arrayindex), axis=0)

    twthe = np.take(_twtheta, np.array(arrayindex))

    chi = np.take(_chi, np.array(arrayindex))

    # simulated wire abscissa
    W1, W2 = LGeo.Wireabscissa_from_IW(IWs[:, 0], IWs[:, 1], height_wire, anglesample=anglesample)
    # introducing abscissa errors
    W1 = W1 + errorWabscissa1
    W2 = W2 + errorWabscissa2
    print("W1,W2", W1, W2)
    IWs = (LGeo.IW_from_wireabscissa(np.array([W1, W2]), height_wire, anglesample=anglesample)).T
    print("2 OMs", OMs)
    print("2 IWs", IWs)
    print("2 2theta and 2 chi", twthe, chi)
    print("2 Wire abscissae", W1, W2)
    print("------------")
    ysource, zsource = LGeo.find_yzsource_from_2xycam_2yzwire_version2(OMs, IWs, calib)
    print("\n*******************\nRetrieving source position\n")
    print("With wire's height (mm):  ", height_wire)
    print("and 2 measured abscissa errors", errorWabscissa1, errorWabscissa2)
    print("\nSource found at")
    print("ysource,zsource", ysource, zsource)
    print("Source simulated at [y,z] (mm)")
    print(IIprime)
    print("Source errors [y,z] (mm)")
    print(np.array([ysource, zsource]) - IIprime[1:])
    print("\n ***********\n")


def test_offset_xraysource():

    calib = [70, 1024, 1024, 0.0, -0.0]
    uflab = np.array([[0.5, -0.5, 0.5], [0.0, -0.0, 1.0], [0.0, -0.5, 0.5], [0.5, 0.5, 0.5]])

    print("Without offset")
    X, Y, th0, E = LGeo.calc_xycam(uflab, calib, energy=1, offset=[0, 0, 0.0])

    print("Xcam (mm)", X * 165 / 2048.0)
    print("Ycam (mm)", Y * 165 / 2048.0)
    print("Xcam (pixel)", X)
    print("Ycam (pixel)", Y)
    print("theta ", th0)
    print("2theta ", 2 * th0)
    print("energy ", E)

    print("\n With offset\n")
    X, Y, th0, E = LGeo.calc_xycam(uflab, calib, energy=1, offset=[0.0, 0.01, 0.0])
    print("Xcam (mm)", X * 165 / 2048.0)
    print("Ycam (mm)", Y * 165 / 2048.0)
    print("Xcam (pixel)", X)
    print("Ycam (pixel)", Y)
    print("theta ", th0)
    print("2theta ", 2 * th0)
    print("energy ", E)


def test_sourcetriangulation():
    calib = [68.0, 1024, 1024, 1.0, -2.0]
    twicetheta = np.array([90.0, 80.0, 90.0, 80.0, 150.0, 60])
    chi = np.array([0.0, 0.0, 25.0, -25.0, 37.0, -23])
    uflab = LGeo.uflab_from2thetachi(twicetheta, chi, verbose=0)

    print("Simulation of data ---")
    print("\nWithout offset\n")
    posI = np.array([0, 0, 0.0])
    IM0 = LGeo.IprimeM_from_uf(uflab, posI, calib, verbose=0)
    X0, Y0, _ = LGeo.calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print(X0, Y0)
    print("IM0", IM0)
    depth_wire = 0.01
    _, _ = LGeo.IW_from_IM_onesource(posI[1:], IM0, depth_wire, anglesample=40.0)

    print("\nWith offset\n")
    posI = np.array([0, 0.01, -0.01])
    print("posI", posI)
    IpM = LGeo.IprimeM_from_uf(uflab, posI, calib, verbose=0)
    IM1 = IpM + posI
    X1, Y1, _ = LGeo.calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print("X1", X1)
    print("Y1", Y1)
    print("IM1", IM1)

    H_wire = 0.01
    IW1y, IW1z = LGeo.IW_from_IM_onesource(posI[1:], IM1, H_wire, anglesample=40.0)

    print("IW1y,IW1z", IW1y, IW1z)
    Wireabscisa_1 = LGeo.Wireabscissa_from_IW(IW1y, IW1z, H_wire, anglesample=40.0)
    print("Wireabscisa_1", Wireabscisa_1)

    print("\n-------------------------------------")
    print("finding source origin from two reflectionx")
    OMs = np.transpose(np.array([X1, Y1]))[:2]
    IWs = np.transpose(np.array([IW1y, IW1z]))[:2]
    print("Using 2 reflections")
    print("OMs", OMs.tolist())
    print("IWs", IWs.tolist())
    print("------------")

    ysource, zsource = LGeo.find_yzsource_from_2xycam_2yzwire(OMs, IWs, calib)

    print("ysource,zsource", ysource, zsource)


def test_sourcefinding():
    calib = [68.0, 1024, 1024, 1.0, -2.0]
    twicetheta = np.array([90.0, 80.0, 90.0, 80.0, 150.0, 60])
    chi = np.array([0.0, 0.0, 25.0, -25.0, 37.0, -23])
    uflab = LGeo.uflab_from2thetachi(twicetheta, chi, verbose=0)

    print("Simulation of data ---")
    print("\nWithout offset\n")
    posI = np.array([0, 0, 0.0])
    IM0 = LGeo.IprimeM_from_uf(uflab, posI, calib, verbose=0)
    X0, Y0, _ = LGeo.calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print(X0, Y0)
    print("IM0", IM0)
    depth_wire = 0.01
    _, _ = LGeo.IW_from_IM_onesource(posI[1:], IM0, depth_wire, anglesample=40.0)

    print("\nWith offset\n")
    posI = np.array([0, 0.01, -0.01])
    print("posI", posI)
    IpM = LGeo.IprimeM_from_uf(uflab, posI, calib, verbose=0)
    IM1 = IpM + posI
    # coordinates on CCD for this source and the same ufs
    X1, Y1, _ = LGeo.calc_xycam(uflab, calib, energy=0, offset=posI, verbose=0, returnIpM=False)
    print("X1", X1)
    print("Y1", Y1)
    print("IM1", IM1)

    H_wire = 0.3
    IW1y, IW1z = LGeo.IW_from_IM_onesource(posI[1:], IM1, H_wire, anglesample=40.0)

    print("IW1y,IW1z", IW1y, IW1z)
    Wireabscisa_1 = LGeo.Wireabscissa_from_IW(IW1y, IW1z, H_wire, anglesample=40.0)
    print("Wireabscisa_1", Wireabscisa_1)

    print("\n\n-------------------------------------")
    print("finding source origin all reflections masking")
    print("originally: posI: ", posI, "  Hwire: ", H_wire)
    OMs = np.transpose(np.array([X1, Y1]))
    Wire_abscissae = Wireabscisa_1
    sourcepos = LGeo.find_multiplesourcesyz_from_multiplexycam_multipleyzwire(
        OMs, Wire_abscissae, calib, anglesample=40.0, wire_height=H_wire)
    print("all results", sourcepos)

    largey = np.where(abs(sourcepos[:, 0]) > 1)[0]
    largez = np.where(abs(sourcepos[:, 1]) > 1)[0]
    badpoints_indices = set(largey).union(set(largez))
    print(badpoints_indices)
    list(badpoints_indices)


def test_correction_1():
    """
    TEST: Reading experimental points=(x,y)
    """
    print("TEST: Reading experimental points=(x,y)")
    param = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201]  # Nov 09 J. Villanova BM32
    peaksfilename = "SS_0170.peaks"
    twicetheta, chi, dataintensity, data_x, data_y = LGeo.Compute_data2thetachi(
        peaksfilename, sorting_intensity="yes",
        detectorparams=param)

    print(twicetheta)
    IOLT.writefile_cor("polyZrO2_test", twicetheta, chi, data_x, data_y,
                        dataintensity,
                        param=param,
                        initialfilename=peaksfilename)


def test_correction_2():
    """
    TEST: Reading experimental points=(x,y)
    """
    print("TEST: Reading experimental points=(x,y)")
    param = [69.66221, 895.29492, 960.78674, 0.84324, -0.32201]  # Nov 09 J. Villanova BM32
    peaksfilename = "Ge.peaks"
    twicetheta, chi, dataintensity, data_x, data_y = LGeo.Compute_data2thetachi(peaksfilename,
                                                                    sorting_intensity="yes",
                                                                    detectorparams=param)

    print(twicetheta)
    IOLT.writefile_cor("Ge_test", twicetheta, chi, data_x, data_y,
                            dataintensity,
                            param=param,
                            initialfilename=peaksfilename)


def test_correction_3():
    """
    TEST: Reading experimental points=(x,y)
    """
    print("TEST: Reading experimental points=(x,y)")
    param = [69.66055, 895.27118, 960.77417, 0.8415, -0.31818]  # Nov 09 J. Villanova BM32
    peaksfilename = "Ge_run41_1_0003.peaks"
    twicetheta, chi, dataintensity, data_x, data_y = LGeo.Compute_data2thetachi(peaksfilename,
                                                                    sorting_intensity="yes",
                                                                    detectorparams=param)

    print(twicetheta)
    IOLT.writefile_cor("Ge_run41_1_0003",
                        twicetheta,
                        chi,
                        data_x,
                        data_y,
                        dataintensity,
                        param=param,
                        initialfilename=peaksfilename)


def find_referencepicture(anglesample=40, penetration=0,
                            calib=np.array([69.1219, 1074.11, 1109.11, 0.32857, 0.00817]),
                            combination=0,
                            falling_or_rising=0,
                            wire_height=0.3,
                            verbose=0,
                            veryverbose=0):
    """
    Return the picture corresponding to the reference picture
    (in find_multiplesourcesyz_from_multiplexycam_multipleyzwire())
    according to the 'good ylab' (like 0mm at the sample surface,
    and "penetration" [mm] if the reference source point (I) is not at the surface).
    """
    # TODO (object way): put step in argument of this function. Transforme test_Gec() as a generic function or put OMs, IWs, etc as arguments of find_referencepicture()
    ylab_at_the_good_depth = 9999
    k_at_the_good_depth = 9999
    xbet = calib[3]
    step_temp_array = test_Gec(anglesample=anglesample, referencepicture=0, wire_height=wire_height)  # Just to know the step. TODO : put in argument
    step = step_temp_array[2]

    if verbose:
        print(calib)
        print(xbet)
        print(step)

    for k in list(range(0, 700)):
        temp = test_Gec(anglesample=anglesample, referencepicture=k, wire_height=wire_height)
        ylab = temp[falling_or_rising][combination][0]
        "falling_or_rising : 0 for falling edge (first edge); 1 for rising edge (second edge)"
        if veryverbose:
            print(ylab)
            print(k)
        if abs(ylab - penetration) < ylab_at_the_good_depth:
            ylab_at_the_good_depth = ylab
            k_at_the_good_depth = k

    """
    From the reference taken by find_multiplesourcesyz_from_multiplexycam_multipleyzwire()
    """
    k_cut_the_direct_beam = (k_at_the_good_depth - (wire_height / np.tan(anglesample * DEG)) / step)
    yf_cut_the_direct_beam = k_cut_the_direct_beam * step

    # TODO: k_at_90deg_under_sample=

    return ["k_cut_the_direct_beam=",
        k_cut_the_direct_beam,
        "yf_cut_the_direct_beam=",
        yf_cut_the_direct_beam,
        "k_at_the_good_depth=",
        k_at_the_good_depth,
        "ylab_at_the_good_depth=",
        ylab_at_the_good_depth]


def test_Gec(anglesample=40.0, referencepicture=245, wire_height=0.3):

    """ Test function for Gec_XXXX.mccd
    Desctiption of the scan (to know the step allowing the convertion: picture number <-> yf)
    """

    yf_start = -1.66076
    yf_end = -0.960765
    nb_interval = 700  # nb_inteval = nb_of_step - 1

    step = abs((yf_end - yf_start) / nb_interval)

    """
    Position in pixels of each studied peak:
    """
    OM1 = np.array([1071.2, 1238.2])
    OM2 = np.array([1546.0, 943.1])
    OM3 = np.array([558.6, 523.2])
    OM4 = np.array([992.5, 144.5])
    OM5 = np.array([1023.4, 493.2])
    OMs = np.array([OM1, OM2, OM3, OM4, OM5])

    """
    yf for the quenching of each peak:
    With the falling edge of the intensity (when the wire begins to shadow the peak):
    """
    W1_falling = step * (531.66 - referencepicture)
    W2_falling = step * (358.37 - referencepicture)
    W3_falling = step * (205.6 - referencepicture)
    W4_falling = step * (125.7 - referencepicture)
    W5_falling = step * (199.95 - referencepicture)
    Wire_abscissae_falling = np.array([W1_falling, W2_falling, W3_falling, W4_falling, W5_falling])

    """
    With the rising edge (second edge):
    """
    W1_rising = step * (605.65 - referencepicture)
    W2_rising = step * (415.17 - referencepicture)
    W3_rising = step * (255.95 - referencepicture)
    W4_rising = step * (176.6 - referencepicture)
    W5_rising = step * (249.7 - referencepicture)
    Wire_abscissae_rising = np.array([W1_rising, W2_rising, W3_rising, W4_rising, W5_rising])

    """
    Calibration:
    """
    calib = np.array([69.1219, 1074.11, 1109.11, 0.32857, 0.00817])

    """
    Calculation:
    """
    res_falling = LGeo.find_multiplesourcesyz_from_multiplexycam_multipleyzwire(
        OMs, Wire_abscissae_falling, calib, anglesample, wire_height, 0)
    res_rising = LGeo.find_multiplesourcesyz_from_multiplexycam_multipleyzwire(
        OMs, Wire_abscissae_rising, calib, anglesample, wire_height, 0)

    res = [res_falling, res_rising, step]

    return res

# ------------------------------------------------------------
# --------------------------  MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    calib1 = [70, 1000.0, 1100, -0.2, 0.3]

    anglesample = 40.0

    height_wire = 0.10  # mm
    IIprime = np.array([0.0, 0, 0])

    ycam = 1.0 * np.arange(0, 2048, 40)[::-1]
    xcam = 1024.0 * np.ones(len(ycam))

    IMlab1 = LGeo.IMlab_from_xycam(xcam, ycam, calib1, verbose=0)

    IWy1, IWz1 = LGeo.IW_from_IM_onesource(
        IIprime[1:], IMlab1, height_wire, anglesample=anglesample, anglewire=anglesample)

    Wireabscissa1 = LGeo.Wireabscissa_from_IW(IWy1, IWz1, height_wire, anglesample=anglesample)

    # -------------------------------------

    calib2 = [60, 1000.0, 1100, -0.2, 0.3]
    anglesample = 40.0

    height_wire = 0.10  # mm
    IIprime = np.array([0.0, 0, 0])

    ycam = 1.0 * np.arange(0, 2048, 40)[::-1]
    xcam = 1024.0 * np.ones(len(ycam))

    IMlab2 = LGeo.IMlab_from_xycam(xcam, ycam, calib2, verbose=0)

    IWy2, IWz2 = LGeo.IW_from_IM_onesource(
        IIprime[1:], IMlab2, height_wire, anglesample=anglesample, anglewire=anglesample)

    Wireabscissa2 = LGeo.Wireabscissa_from_IW(IWy2, IWz2, height_wire, anglesample=anglesample)

    import pylab as p

    p.plot(ycam, Wireabscissa1, ycam, Wireabscissa2)

    p.show()

    if 0:
        if len(sys.argv) == 1:
            # test_correction_1()
            # test_correction_2()
            # test_correction_3()
            # test_offset_xraysource()
            test_sourcetriangulation()
            sys.exit()

        print("\n *************\n\nfrom file:", sys.argv[1])

        filename = sys.argv[1]
        # filename="NbSe3_11Mar07_0012_936pics.dat"
        # filename="I832a0325.DAT"
        # filename="CdTe_I832_0325_peak.dat"
        prefix = "sUrHe"
        indexfile = "0103"
        suffix = ".pik"
        col_X = 0
        col_Y = 1
        col_I = 2  # index (starting from 0) of the intensity column
        nblines_headertoskip = 0
        Intensitysorted = 0 # =1 if intensity sorting must be done for the outputfile, =0 means that sorting already done in input file or sorting not needed

    # for doing a files serie
    def series():
        """
        doing a files serie
        """
        for index in list(range(620, 1276)):
            filename = prefix + "%04d" % index + suffix
            twicetheta, chi, dataintensity, data_x, data_y = LGeo.Compute_data2thetachi(
                filename)
            IOLT.writefile_cor(prefix + "%04d" % index,
                                    twicetheta,
                                    chi,
                                    data_x,
                                    data_y,
                                    dataintensity,
                                    sortedexit=Intensitysorted)
