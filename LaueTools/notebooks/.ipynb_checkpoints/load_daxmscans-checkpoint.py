import h5py
import numpy as np
import time
import multiprocessing
import os
import sys

def load_scan(h5file, scan_index, slice_params):
    """Load a single scan slice from the HDF5 file."""
    with h5py.File(h5file, 'r', locking=False) as f:
        data = f[f'{int(scan_index)}.1/measurement/eiger4m'][slice_params]
    return data

def main():
    # --- Parameters ---
    h5file = '/data/visitor/a321217/bm32/20260707/RAW_DATA/CrZr/CrZr_fast_daxm_grid_1fileperscan/CrZr_fast_daxm_grid_1fileperscan.h5'
    output_dir = '/data/visitor/a321217/bm32/20260707/PROCESSED_DATA/'
    output_filename = 'loaddaxmscans_results.h5'

    X, Y = 620, 1767
    #X, Y = 541, 1740
    hboxX, hboxY = 0, 0
    image_start, image_end = 290, 325
    
    output_path = os.path.join(output_dir, output_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Example: Load all 34x34 scans
    scan_indices = list(range(1, 34*34 + 1))
    scan_indices = list(range(1, 34*34 + 1))
    
    slice_params = (slice(image_start, image_end), slice(Y-hboxY, Y+hboxY+1), slice(X-hboxX, X+hboxX+1))

    # Prepare tasks: (h5file, scan_index, slice_params)
    tasks = [(h5file, scan_idx, slice_params) for scan_idx in scan_indices]

    print('len', len(tasks))
    print(tasks[0])

    # --- Load scans in parallel ---
    t0 = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.starmap(load_scan, tasks)
    tf = time.time()
    et = tf - t0
    print(f'Loaded {len(results)} scans in {et:.2f} sec')

    # --- Save results as a 3D array in HDF5 ---
    # Stack results into a 3D array: (n_scans, height, width)
    stacked_results = np.stack(results)
    n_scans, nbimages, boxsize1, boxsize2 = stacked_results.shape

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('daxmscan', data=stacked_results, compression='gzip')
        f.attrs['scan_indices'] = scan_indices
        f.attrs['slice_params'] = str(slice_params)
        f.attrs['elapsed_time_sec'] = et

    print(f'Results saved to {output_path}')

if __name__ == "__main__":
    main()