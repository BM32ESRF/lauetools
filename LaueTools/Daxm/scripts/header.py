#----------- imports -----------
import os, sys
import numpy as np
import matplotlib.pylab as mplp

print('Hello, I m the header !!!')

# lauetools path
path_to_lauetools = "/home/micha/LaueToolsPy3/LaueTools"

print("Ajout de {} au PATH...".format(path_to_lauetools))
if path_to_lauetools not in sys.path:
    sys.path.insert(0,path_to_lauetools)

from LaueTools.Daxm.classes.scan.scan import new_scan
from LaueTools.Daxm.classes.source import new_source
from LaueTools.Daxm.classes.calibration import new_calib, CalibManager
from LaueTools.Daxm.classes.reconstruction import RecManager

print("daxm classes loaded ...")

#----------- input/output paths -----------

analysis_dir = "/home/micha/LaueProjects/DAXMSept21/GeDAXM"   # images folder
calib_dir = os.path.join(analysis_dir, "calibration")
rec_dir = os.path.join(analysis_dir, "reconstruction")

prefix = "GeDAXM"


# ---------------------------------------
analysis_dir = "/media/micha/LaCie/a322847_guillou_daxm/ech13/ech13_map2d3d/scan_0002"   # images folder
calib_dir = os.path.join(analysis_dir, "calibration")
rec_dir = os.path.join(analysis_dir, "reconstruction")


prefix = "ech13_daxm_0"

