# Mosaic and grain imaging (LaueTools notebooks)

`mosaic_grainimaging.ipynb`: intensities of regions (ROIs) of the detector monitored over all the images of a 2D map
of Laue patterns, without indexation:

1. choose a dataset of the experiment: menu of the folders of `RAW_DATA` with a filter on the dataset name (fast, also
   for experiments with thousands of datasets), then one of its scans (its own BLISS `.h5` file)
2. one Laue pattern of the map: peak search, positions picked with the mouse, regular grid of ROI centers
3. **mosaic**: the pixels of each picked ROI in every image, arranged as the map
4. **grain imaging**: map of the intensity of each Laue spot, overview of the ROIs lit at each point, browser

Results (`roicounters_<hash>.h5`) are written in `PROCESSED_DATA` and reloaded when the parameters are the same.
Computation in the notebook, or as a job on this machine or on the ESRF cluster (SLURM).

Code: `LaueTools/grainimaging.py`, `LaueTools/scripts/workflows.py`, `LaueTools/scripts/roicountersplot.py`.
The same quick look (mosaic, grain imaging) is in `../quickstart/laue_maps_quickstart.ipynb` (STEP 1b) before the
indexation of a map.
