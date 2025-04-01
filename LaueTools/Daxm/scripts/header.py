"""set folders for DAXM analysis (create them if needed) and check all DAXM classes are ready"""

import os, sys
from pathlib import Path

#----USER input/output paths -----------

# analysis_dir is advised to be located at the images folder
# two subfolders will be created 'calibration', 'reconstruction'
analysis_dir = "/home/micha/LaueProjects/DAXMSept21/GeDAXM"  
calib_dir = os.path.join(analysis_dir, "calibration")
rec_dir = os.path.join(analysis_dir, "reconstruction")

# prefix in future file related to 1 or several DAXM scans such as GeDAXM.calib  
prefix = "GeDAXM"

# # ---------------------------------------
# analysis_dir = "/media/micha/LaCie/a322847_guillou_daxm/ech13/ech13_map2d3d/scan_0002"   # images folder
# calib_dir = os.path.join(analysis_dir, "calibration")
# rec_dir = os.path.join(analysis_dir, "reconstruction")
# prefix = "ech13_daxm_0"

#----------- create folder if needed ---------------

if not Path(analysis_dir).exists():
    sys.exit('!! Can not find analysis_dir =\n%s'%analysis_dir)

if not Path(calib_dir).exists():
    Path(calib_dir).mkdir(0o775)
if not Path(rec_dir).exists():
    Path(rec_dir).mkdir(0o775)

# if not os.access(Path(calib_dir).name, os.W_OK):
#     sys.exit('!! Can not write in %s'%calib_dir)

# if not os.access(Path(rec_dir).name, os.W_OK):
#     sys.exit('!! Can not write in %s'%rec_dir)
#------------check LaueTools module location and DAXM classes -------

try: 
    import LaueTools as LT
except:
    print("You may want to use an other LaueTools distribution ?. ")
    path_to_lauetools = "/my/path/to/the/parentfolder/of/LaueTools" 

    print(f"add {path_to_lauetools} to PATH...")
    if path_to_lauetools not in sys.path:
        sys.path.insert(0,path_to_lauetools)
        print('sys.path',sys.path)

try:
    import LaueTools as LT
    print('Using LaueTools from:', LT)
    # class for daxm scan related parameters
    from LaueTools.Daxm.classes.scan.scan import new_scan
    # class for x-ray fluorescence source distribution in depth
    from LaueTools.Daxm.classes.source import new_source
    # manager and optimizer for wires trajectories parameters
    from LaueTools.Daxm.classes.calibration import new_calib, CalibManager
    # builder of depth resolved Laue spots
    from LaueTools.Daxm.classes.reconstruction import RecManager
    print("daxm classes loaded ...")
except:
    print('Errors when loading DAXM classes !...')





