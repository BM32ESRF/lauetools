# DAXM workflows (LaueTools notebooks)

One YAML configuration file per wire scan (`config_template.yaml`), shared by the DAXM workflows. The former notebooks and scripts (`../1_Pre_Calib_wire_1yech.ipynb`, `../daxmreconstruct_*`, the *Quick DAXM* sections of the `../PixelMonitoring*` notebooks, `LaueTools/Daxm/scripts`) are kept unchanged as references to compare the results.

| notebook | input | output |
|---|---|---|
| `1_quick_daxm.ipynb` | images of a wire scan, detector `.det` file, wires `.calib` file | position along the beam (depth) of each Laue spot: `quickdaxm.h5`, `quickdaxm_spots.cor` (depth column), `depthslices/*.cor` (one peak list per slice, input of `../indexation/`) |

Code: `LaueTools/Daxm/quickdaxm.py` (profiles, dips, geometry), `LaueTools/daxm_workflow.py` (configuration, jobs, files).

## Quick DAXM

1. set `SCANFILE` (`.scan` file of the DAXM GUI or of the wires calibration notebook) and `CALIBFILE` (`.calib`) in the first code cell of `1_quick_daxm.ipynb`, or write the configuration yourself:
   ```python
   import LaueTools.daxm_workflow as DW
   cfg = DW.new_config('configs/my_scan.yaml', scanfile='/data/.../my_calib.scan', calibfile='/data/.../my.calib')
   ```
   (`hdf5_scan=('/data/.../<dataset>/<dataset>.h5', 1)` instead of `scanfile=` takes scan 1 from the BLISS hdf5 file of the dataset, for sCMOS `.tif` and EIGER `.h5` images, one image per file or stacked; `scandict=logfile.build_dict_scan(i)` takes a scan selected in the logfile; any parameter of the template can be given by name, e.g. `ybeam_range=[-50, 150]`)
2. run the notebook: `RESOURCE = 'kernel'` (in the notebook), `'local'` (background job) or `'magnifix'`, `'nice-hpc7'` ... (SLURM job, see `LaueTools/scripts/workflowlauncher.py`)
3. or from a terminal: `python -m LaueTools.daxm_workflow quickdaxm configs/my_scan.yaml --resource magnifix --ncpus 96`

### How the dips are found

The profile of a spot is the mean of its core (3x3 pixels) minus the median of the border of a box around it, image by image. Fluorescence varies slowly on the detector, so it is the same in the core and on the border: its smooth dips cancel and only the sharp dips of the scattering signal remain. A dip is a range of images where this profile is below `low_fraction` of its plateau, limited by edges higher than `min_edge_fraction` of the plateau (rejects intensity fluctuations); the edges are located at half height with sub image accuracy.

### How the depth is computed

At an edge the wire is tangent to the ray going from the source to the pixel. With the wires of the `.calib` file (`CircularWire`, the model of the wires calibration and of the reconstruction), the entry edge and the exit edge each give the position `ybeam` of the source along the beam; they must agree within `max_mismatch`, which also tells which wire made the dip. `ybeam_range` must stay smaller than the spacing of the wires seen from the sample.

`QD.WireGeometry.shadow_lines()` gives the lines *image index of the dip = f(pixel Y)* of each wire for a given `ybeam` (plot of section 4 of the notebook), and `wp_parameters()` the equivalent of the hand-tuned list `WP` of the former quick daxm notebooks.

Units: `ybeam` in microns along the beam (> 0 deeper in the sample, 0 at the reference point of the wires calibration); `depth = ybeam * sin(sample tilt)`.
