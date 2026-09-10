## python script

import argparse
import json
import os
import multiprocessing
import itertools
import numpy as np
import time

from LaueTools.Daxm.classes.scan import new_scan
#from LaueTools.Daxm.classes.source import new_source
from LaueTools.Daxm.classes.calibration import new_calib, CalibManager
from LaueTools.Daxm.classes.reconstruction import RecManager, ScanReconstructor

os.environ["OMP_NUM_THREADS"] = "1"   # 1 or 2 or 4

def main():

    print('************ in daxmreconstruct ***********')

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--ncpus", type=int, required=True)

    args = parser.parse_args()

    # -----------------------------------------
    # read config
    # -----------------------------------------

    with open(args.config, "r") as f:
        config = json.load(f)

    analysis_dir = config['analysis_dir']

    calib_file = config["calib_file"]
    calib_dir = config["calib_dir"]
    scan_file = config['scan_file']

    #nbcompletelines = config.get("nbcompletelines", None)

    segpar = config["segpar"]
    prefix = config["prefix"]
    drange = config["drange"]  # depth range in mm (list of 2 elements: ex: [-0.02, 0.05] means 20 µm outside surface to 50 µm underneath ) for comprehensive reconstrcution
    drange_print = config["drange_print"]  # depth range in mm only for writing Laue pattern images per voxel
    recdir = config["recdir"]
    recdirsubfolder = config['recdirsubfolder']
    nproc = config["nproc"]  # nb cpus

    verbose=config['verbose']


    xgrid = np.array([0.,])
    ygrid = np.array([0.,])
    yref = 0
    
    ystep = 0.001

    # ---------------------------------------------

    calib = CalibManager(calib_file, yref=yref, directory=calib_dir)
    
    
    print('scan_file',scan_file)
    
    print()
    print('******')
    print()
    
    # sample = new_source("Zr", 0.1, ystep=0.001)
    
    # #---------- reconstruction -----------
    scan = new_scan(scan_file)

    print(scan.get_img_params(), scan.img_filenames[0], scan.img_folder)
    print(scan.hdf5scanId, scan.spec_scan_num, scan.ccd_type)

    
    #Instantiate rec object + Set some attributes #
    #CR: Valid for all scans: 3Dmesh, line, single scan
    rec = RecManager(scan, calib, segpar)
    
    rec.set_calib_yref(yref)
    rec.set_grid(xgrid, ygrid)
    
    rec_par = {'regularize':False}
    
    #20250716 - Command from original rec.py provided by Robin for 1um step
    
    #We also need to update the scan_rec_dir at each iteration in order to write the .fit files in the correct repo of each scan
    scan_rec_dir = os.path.join(analysis_dir, "reconstruction")
    rec.reconstruct(drange, prefix,
                    nproc=nproc, directory=recdirsubfolder, rec_par=rec_par,
                    depth_range_print=drange_print, verbose=verbose) #usefitfiles_peaks = True)

   


if __name__ == "__main__":
    main()

