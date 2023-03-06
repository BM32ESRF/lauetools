from header import *

scan_file = os.path.join(analysis_dir, prefix + "_calib.scan")
calib_file = prefix + ".calib"

# from lauetools GUI 
fitfile = os.path.join(analysis_dir, prefix + ".fit")

# mesh or line of Point (wire scan)
# yref = 0.025  # =  ycalib  calibration by fluorescence is done (peaks not too large)
# xgrid = np.arange(-0.008, 0.00801, 0.001)
# ygrid = np.arange(-0.018, 0.01801, 0.003)

# for single profile wire scan
xgrid = np.array([0.,])
ygrid = np.array([0.,])
yref = 0

#   from blosearch.ipynb
seg_par = {'max_size' : 100,
        'min_size' : 3,
        'thr': 20,
        'erode': 2,
        'dilate': 3,
        'merge': False}

rec_par = {'regularize': False}

#   from rec_spot.ipynb
drange_large = [-0.08, 0.1]
drange_narrow = [0.0, 0.015]

nproc = 8

#----------- load/create scan, sample, calib -----------

scan = new_scan(scan_file)

sample = new_source("Ge", 0.02)

calib = CalibManager(calib_file, yref=yref, directory=calib_dir)



#---------- reconstruction -----------

rec = RecManager(scan, calib, seg_par)

rec.set_calib_yref(yref)

rec.set_grid(xgrid, ygrid)

rec.set_fitfile(fitfile)

rec.reconstruct(drange_large, prefix, nproc=nproc, directory=rec_dir, rec_par=rec_par, depth_range_print=drange_narrow)


