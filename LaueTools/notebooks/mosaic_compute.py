import argparse
import json
import os
import multiprocessing
import itertools
import numpy as np
import time

import LaueTools.imagescollector as IC
import LaueTools.generaltools as GT


import os
import time

os.environ["OMP_NUM_THREADS"] = "1"   # 1 or 2 or 4

import socket
print("hostname =", socket.gethostname(), flush=True)

# def collectroiarray_singlefile(*args):

#     pid = os.getpid()

#     t0 = time.time()

#     print(f"PID {pid} starting", flush=True)

#     res = IC.collectroiarray_singlefile(*args)

#     print(
#         f"PID {pid} finished in {time.time()-t0:.2f}s",
#         flush=True
#     )

#     return res

from multiprocessing import Value

counter = None

def init_worker(c):
    global counter
    counter = c

def collectroiarray_singlefile(*args):

    imageindex = args[0]

    t0 = time.time()

    res = IC.collectroiarray_singlefile(*args)

    dt = time.time() - t0

    if dt > 1:
        print(
            f"SLOW image {imageindex}: {dt:.2f}s",
            flush=True
        )

    return res

def worker(args):
        imageindex = args[0]

        t0 = time.time()
    
        result = collectroiarray_singlefile(*args)
    
        dt = time.time() - t0
    
        return imageindex, dt, result

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--ncpus", type=int, required=True)

    args = parser.parse_args()

    # -----------------------------------------
    # read config
    # -----------------------------------------

    with open(args.config, "r") as f:
        config = json.load(f)

    d = config["d"]

    boxsize_X = config["boxsize_X"]
    boxsize_Y = config["boxsize_Y"]

    ACQUISITON_ON = config.get( "ACQUISITON_ON", False)

    nbcompletelines = config.get("nbcompletelines", None)

    roicenter = d["roicenter"]

    listindices = d["listindices"]

    # -----------------------------------------
    # acquisition mode
    # -----------------------------------------

    if ACQUISITON_ON:

        maxindex = GT.get_largest_index_in_folder(
            d["folder"], filename_prefix=d["prefix"], filename_suffix=d["suffix"] )

        nbcompletelines = (maxindex // d["nbimagesperline"] )

        listindices = (d["listindices"][: d["nbimagesperline"] * nbcompletelines])

        d["mapdimensions"] = (d["mapdimensions"][0], nbcompletelines)

    # -----------------------------------------
    # multiprocessing
    # -----------------------------------------

    nbimages = len(listindices)



    ncpus = min(args.ncpus, nbimages)

    print(f"Processing {nbimages} images with {ncpus} CPUs")

    t0 = time.time()


    # args_mosaic = zip(
    #         listindices,
    #         itertools.repeat(roicenter),
    #         itertools.repeat(d["prefix"]),
    #         itertools.repeat(d["folder"]),
    #         itertools.repeat(boxsize_X),
    #         itertools.repeat(boxsize_Y),
    #         itertools.repeat(d["CCDLabel"])
    #         )

    # with multiprocessing.Pool(
    #         ncpus,
    #         initializer=init_worker,
    #         initargs=(progress,)
    #     ) as pool:
    
    #     allresults = pool.starmap(
    #         collectroiarray_singlefile,
    #         args_mosaic
    #     )

    args_mosaic = list(zip(
                    listindices,
                    itertools.repeat(roicenter),
                    itertools.repeat(d["prefix"]),
                    itertools.repeat(d["folder"]),
                    itertools.repeat(boxsize_X),
                    itertools.repeat(boxsize_Y),
                    itertools.repeat(d["CCDLabel"])
                ))

    with multiprocessing.Pool(ncpus) as pool:

        allresults = []
    
        for i, (imageindex, dt, result) in enumerate(
                pool.imap_unordered(worker, args_mosaic),
                start=1):
    
            if dt > 1:
                print(
                    f"SLOW image {imageindex}: {dt:.1f}s",
                    flush=True
                )
    
            allresults.append(result)
    
            if i % 50 == 0:
                print(
                    f"parent received {i}/{nbimages}",
                    flush=True
                )

    print("collection finished", flush=True)
    allresults = np.array(allresults)
    print("array conversion finished", flush=True)

    print("allresults.shape:", allresults.shape)
    
    print("elapsed:", time.time() - t0)

    # -----------------------------------------
    # build mosaic
    # -----------------------------------------
    if 0:
        dimfast, dimslow = d['mapdimensions']
        mosaic = np.zeros((dimslow, dimfast, 2 * boxsize_Y_widget.value + 1, 2 * boxsize_X_widget.value + 1))
        sm = mosaic.shape
        bigimage = np.zeros((sm[0] * sm[2], sm[1] * sm[3]))
        
        if dimfast > 0:
            for map_imageindex, absolute_imageindex in enumerate(d['listindices']):
                imap, jmap = map_imageindex // dimfast, map_imageindex % dimfast
                raw = allresults[map_imageindex, :, :]
                datcrop = raw
                mosaic[imap, jmap] = datcrop
                bigimage[imap * sm[2]:(imap + 1) * sm[2], jmap * sm[3]:(jmap + 1) * sm[3]] = np.flipud(datcrop)
        
        mosaictranspose = mosaic.transpose((0, 3, 1, 2))
        mosaicflat = mosaictranspose.reshape((dimfast * (2 * boxsize_X_widget.value + 1), dimslow * (2 * boxsize_Y_widget.value + 1)))

    # -----------------------------------------
    # save result
    # -----------------------------------------

    outfile = os.path.join(d["folder"],"mosaic.npy")

    np.save(outfile, allresults)

    print("allresults array saved in :", outfile)
    print("shape:", allresults.shape)


if __name__ == "__main__":
    main()