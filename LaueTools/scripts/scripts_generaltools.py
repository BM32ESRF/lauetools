import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import generaltools as GT
import IOLaueTools as IOLT
import numpy as np
import multiprocessing

dirname = "/home/micha/LaueTools/MapSn/datfiles"
file1 = "SnsurfscanBig_0296.dat"
file2 = "SnsurfscanBig_0298.dat"

data1 = IOLT.read_Peaklist(file1, dirname)
data2 = IOLT.read_Peaklist(file2, dirname)

nbspots1 = len(data1)
nbspots2 = len(data2)

print("nb peaks in data1", nbspots1)
print("nb peaks in data2", nbspots2)

XY1 = data1[:, :2]
XY2 = data2[:, :2]

toldistance = 1.0

res = GT.getCommonPts(XY1, XY2, toldistance)

nbcommonspots = len(res[0])

print("nb of common spots (< %.f): " % toldistance, nbcommonspots)

print("mean % of common spots: ", 100.0 * nbcommonspots / ((nbspots2 + nbspots1) / 2.0))

# common spots:
# x,y in XY1
inXY1 = XY1[res[0]]
# x,y in XY2
inXY2 = XY2[res[1]]
# dist between x1,y1 and x2,y2
distances = np.sqrt(np.sum((inXY1.T - inXY2.T) ** 2, axis=0))


list_produced_files = []


Parameters_dict = {}
Parameters_dict["prefixfilename"] = "SnsurfscanBig_"
Parameters_dict["dirname"] = "/home/micha/LaueTools/MapSn/datfiles"
Parameters_dict["nbimagesperline"] = 41
Parameters_dict["toldistance"] = 0.1


nblines = 32
fileindexrange = (0, 41 * nblines - 1)

#     imageindexref = 41*16+20
imageindexref = 41 * 2 + 20

myc, cor1, cor2 = GT.LaueSpotsCorrelator_multiprocessing(fileindexrange,
                                                        imageindexref,
                                                        Parameters_dict=Parameters_dict,
                                                        saveObject=0,
                                                        nb_of_cpu=7)

import pylab as p

p.imshow(myc, interpolation="nearest", origin="lower", vmin=0)
p.show()

indexauxiliary = imageindexref - 1
filetoindex = Parameters_dict["prefixfilename"] + "%04d" % (imageindexref) + ".dat"
fileauxiliary = Parameters_dict["prefixfilename"] + "%04d" % (indexauxiliary) + ".dat"


toldistance = 0.1
res = GT.getCommonSpots(
    filetoindex,
    fileauxiliary,
    toldistance,
    dirname=Parameters_dict["dirname"],
    data1=None,
    fulloutput=True,
)

dataraw = IOLT.read_Peaklist(filetoindex, Parameters_dict["dirname"])

seletedspots = np.take(dataraw, res[0], axis=0)

IOLT.writefile_Peaklist(
    "selectedqpots.dat", seletedspots, dirname=Parameters_dict["dirname"]
)


if 0:
    dictcorrelval = {}
    for k, result in enumerate(jesus):
        dictk = dict(result.get())
        print("dict %d" % k, dictk)
        dictcorrelval = dict(list(dictk.items()) + list(dictcorrelval.items()))

    listindval = []
    for k, val in dictcorrelval.items():
        listindval.append([k, val])

    arr_correl = np.array(listindval)
    sortedindex = np.argsort(arr_correl[:, 0])
    myc = arr_correl[sortedindex][:, 1].reshape((nblines, 41))

    import pylab as p

    p.imshow(myc, interpolation="nearest", origin="lower", vmin=0)
    p.show()

    fdffdhgf
    toldistance = 2.0
    dirname = "/home/micha/LaueTools/MapSn/datfiles"
    fileprefix = "SnsurfscanBig_"
    commonspotsnb = []
    imageindexref = 41 * 32 / 2 + 20
    file1 = fileprefix + "%04d" % (imageindexref) + ".dat"
    for imageindex in list(range(1350, 1500)):
        print("imageindex", imageindex)
        file2 = fileprefix + "%04d" % (imageindex) + ".dat"
        commonspotsnb.append(GT.getCommonSpots(file1, file2, toldistance, dirname))

    myc = np.array(commonspotsnb).reshape((32, 41))
    import pylab as p

    p.imshow(myc, interpolation="nearest", origin="upper", vmin=0, vmax=150)
    p.show()
if 0:
    #     XY1 = np.array([[0, 0], [7, 7], [1, 1], [3, 3], [4, 4]])
    #     XY2 = np.array([[0, 5], [1, 1.3], [2, 20], [30, 3], [44, 4], [10, 1.3], [152, 1.4]])
    #
    #     p, del_1, del_2 = mergelistofPoints(XY1, XY2, dist_tolerance=0.31, verbose=0)
    #
    #     print "init 1", XY1
    #     print "init 2", XY2
    #
    #     print "purged list", p
    #     print "delete in 1", del_1
    #     print "delete in 2", del_2
    #
    #     XYref = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],[0,5]])
    #     XYtest = np.array([[0, 2.1], [0, 1.1], [0, 0.1], [0, 5.1], [0, 6.1]])
    #
    #     res=SortPoints_fromPositions(XYtest, XYref, 0.2)
    folder = "/home/micha/LaueProjects/VO2/"
    blacklistedpeaklist_file = "VO2W_0025_LT_1.dat"

    expfile = "VO2W_0024_LT_1.dat"

    import IOLaueTools as IOLT

    datapeak = IOLT.read_Peaklist(blacklistedpeaklist_file, folder)
    pts_black = datapeak[:, :2]
    print("nb black listed points", len(pts_black))

    datapeak2 = IOLT.read_Peaklist(expfile, folder)
    pts = datapeak2[:, :2]
    print("nb exp points", len(pts))

    xkept, ykept, tab = GT.removeClosePoints_two_sets(pts.T, pts_black.T, dist_tolerance=0.5, verbose=1)
