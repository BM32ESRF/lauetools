# Laue maps quickstart (LaueTools notebook)

`laue_maps_quickstart.ipynb`: complete analysis of a 2D map of Laue patterns recorded at BM32, for new users, with one settings cell:

1. choose a map in the list of the scans of the experiment (BLISS `.h5` file of `RAW_DATA`)
2. settings files written in `PROCESSED_DATA/.../configs/` (`peaksearch.yaml`, `indexing.yaml`)
3. peak search (test on one image with a plot, then all images)
4. indexation & refinement (test on one image, then all images)
5. maps: quality, orientation (IPF colors), grains, KAM, GROD
6. (option) the same analysis for a series of maps (e.g. temperatures)

Minimum settings: `RAW_DATA_FOLDER` and `KEY_MATERIAL`. The calibration `.det` file is found automatically (the chosen file is printed: check it). All parameters are in the two YAML files of `configs/` (peak search: default values of the detector; the values of a `.psp` file of the LaueTools GUI can be copied in with `PSPFILE`).

- **Computations**: on the ESRF cluster (SLURM, `magnifix` partition for large jobs, else `hpc6/7/8`) when the notebook runs on https://jupyter-slurm.esrf.fr (`sbatch` available), otherwise on the current computer with all its cpus (`EXECUTION = 'auto'`). The notebook waits for the end of the cluster jobs; interrupting the wait does not stop the job and running the cell again waits for the same job.
- **Safe to run again**: steps already done are reloaded, not computed again (`REDO = True` to compute again). Images are only read; results are written in `PROCESSED_DATA` (or `OUTPUT_ROOT`).
- The code is in `LaueTools/map_workflow.py`, which uses `peaksearch_batch.py` and `indexing_batch.py`. The settings files written here are those of the detailed notebooks `../peaksearch/peaksearch_map.ipynb` and `../indexation/*.ipynb` (strain, lattice parameters, interactive grain browser).
