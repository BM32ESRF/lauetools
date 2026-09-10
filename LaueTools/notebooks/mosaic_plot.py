#!/usr/bin/env python
import argparse
import json
import sys
import os
import multiprocessing
import itertools
import numpy as np
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SLURM
import matplotlib.pylab as plt

# Add LaueTools to path if needed
# sys.path.insert(0, '/data/bm32/inhouse/lauetoolsenv2/lib/python3.12/site-packages')
import LaueTools.imagescollector as IC
import LaueTools.generaltools as GT

def collectroiarray_singlefile(*args):
    return IC.collectroiarray_singlefile(*args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--ncpus", type=int, required=True, help="Number of CPUs to use")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract parameters
    d = config['d']
    boxsize_X, boxsize_Y = config['boxsize_X'], config['boxsize_Y']
    AUTOCONTRAST = config.get('AUTOCONTRAST', False)
    ACQUISITON_ON = config.get('ACQUISITON_ON', False)
    nbcompletelines = config.get('nbcompletelines', None)
    listindices = config.get('listindices', None)

    # Validate roicenter
    roicenter = d.get('roicenter')
    if roicenter is None:
        raise ValueError("d['roicenter'] is not set.")
    xroi, yroi = roicenter
    maxdimension = 2050
    if xroi - boxsize_X < 0 or xroi + boxsize_X > maxdimension or yroi - boxsize_Y < 0 or yroi + boxsize_Y > maxdimension:
        raise ValueError(f"d['roicenter'] {roicenter} is too close to the border for boxsize {(boxsize_X, boxsize_Y)}")

    # Adjust listindices if needed
    if listindices is None:
        listindices = d['listindices']
        if nbcompletelines is not None or ACQUISITON_ON:
            if ACQUISITON_ON:
                maxindex = GT.get_largest_index_in_folder(d['folder'], filename_prefix=d['prefix'], filename_suffix=d['suffix'])
                nbcompletelines = maxindex // d['nbimagesperline']
            listindices = d['listindices'][:d['nbimagesperline'] * nbcompletelines]
            d['mapdimensions'] = (d['mapdimensions'][0], nbcompletelines)
            d['listindices'] = listindices

    # Prepare folders and indices
    folder = d.get('folder')
    if not isinstance(folder, list):
        nbimages = len(listindices)
        listfolders = itertools.repeat(folder)
    else:
        nbimages = len(folder)
        listfolders = folder
        listindices = itertools.repeat(d['listindices'])

    # Prepare arguments for multiprocessing
    args_mosaic = zip(
        listindices,
        itertools.repeat(roicenter),
        itertools.repeat(d.get('prefix')),
        listfolders,
        itertools.repeat(boxsize_X),
        itertools.repeat(boxsize_Y),
        itertools.repeat(d.get('CCDLabel')),
    )

    # Run multiprocessing
    nbcpus = args.ncpus
    t00 = time.time()
    print(f'Processing {nbimages} images with {nbcpus} CPUs...')
    with multiprocessing.Pool(nbcpus) as pool:
        allresults = pool.starmap(collectroiarray_singlefile, tqdm(args_mosaic, total=nbimages, desc='Progress'))

    allresults = np.array(allresults)
    elapsedtime = time.time() - t00
    print(f'Total time: {elapsedtime:.3f} sec for {nbimages} images and {nbcpus} CPUs')

    # Store results
    d['allresults'] = allresults
    output_npy_path = os.path.join(d['folder'], 'allresults.npy')
    np.save(output_npy_path, allresults)
    print(f"Saved array to: {output_npy_path}")
    print(f'Results shape: {allresults.shape} (nbimages x boxsize x boxsize)')

    # Build mosaic only
    dimfast, dimslow = d['mapdimensions']
    
    mosaic = np.zeros(
        (dimslow, dimfast,
         2 * boxsize_Y + 1,
         2 * boxsize_X + 1)
    )
    
    for map_imageindex, absolute_imageindex in enumerate(d['listindices']):
        imap = map_imageindex // dimfast
        jmap = map_imageindex % dimfast
    
        mosaic[imap, jmap] = allresults[map_imageindex]
    
    mosaictranspose = mosaic.transpose((0, 3, 1, 2))
    mosaicflat = mosaictranspose.reshape(
        (
            dimfast * (2 * boxsize_X + 1),
            dimslow * (2 * boxsize_Y + 1),
        )
    )
    
    output_npy_path = os.path.join(d['folder'], 'mosaic.npy')
    np.save(output_npy_path, mosaicflat)
    
    print(f"Mosaic saved to: {output_npy_path}")
    print(f"Mosaic shape: {mosaicflat.shape}")

if __name__ == "__main__":
    main()