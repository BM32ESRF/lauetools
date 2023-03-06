
from header import *

print('path_to_lauetools', path_to_lauetools)

calib_file = os.path.join(analysis_dir, prefix + "_calib.scan")  # for batch analysis
 
calib_file = os.path.join(analysis_dir, 'ech13_daxm_0_LT3.scan')

print('calib_file',calib_file)

#----------- load/create scan, sample, calib -----------

scan = new_scan(calib_file)

sample = new_source("Zr", 0.1, ystep=0.001)

calib = new_calib(scan, sample, kind="fluo")


#----------- coarse calibration-----------

calib.set_points_grid(dims=[2, 2])

calib.run(var=['h', 'p0', 'axis', 'dm'])

calib.save_wires(prefix+"_fast", directory=calib_dir)

calib.log_plot()

mplp.show()


#----------- coarse calibration-----------

calib.set_points_grid(dims=[3, 3])

calib.run(var=['h', 'p0', 'axis', 'Re', 'u2', 'dm'])

calib.save_wires(prefix+"_test", directory=calib_dir)

calib.log_plot()

mplp.show()

#----------- fine calibration-----------

calib.data_span = 3.5

calib.set_points_grid(dims=[6, 6])

calib.run(var=['Re', 'h', 'p0', 'axis', 'u2', 'dm'])

calib.save_wires(prefix, directory=calib_dir)

calib.log_plot()

mplp.show()

