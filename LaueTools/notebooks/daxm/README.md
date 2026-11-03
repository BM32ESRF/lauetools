# DAXM workflows (LaueTools notebooks)

One YAML configuration file per wire scan (`config_template.yaml`), shared by the DAXM workflows. The former notebooks and scripts (`../1_Pre_Calib_wire_1yech.ipynb`, `../daxmreconstruct_*`, the *Quick DAXM* sections of the `../PixelMonitoring*` notebooks, `LaueTools/Daxm/scripts`) are kept unchanged as references to compare the results.

| notebook | purpose | output (in `PROCESSED_DATA/<sample>/<dataset>/scanXXXX/daxm`) |
|---|---|---|
| `1_wires_calibration.ipynb` | wires calibration with a Ge wire scan or a wire scan of the sample: crude from the Laue spots dips (`dips` for Ge, `envelope` for any sample), fine with the fluorescence (option), comparison | `wirescalib/*.calib` |
| `2_quick_daxm.ipynb` | one wire scan: dips of the Laue spots and their positions along the beam (depth), without reconstruction | `quickdaxm.h5`, `quickdaxm_spots.cor` (depth column), `depthslices/*.cor` (one peak list per slice, input of `../indexation/`) |
| `3_quick_daxm_grid.ipynb` | all the wire scans of a raster / grid: quick daxm of every scan with the calibration of notebook 1 and a p0 shift per scan (surface position along the beam), maps of the checks, depth layers of each scan (grains along the beam), their indexation: orientation, strain and lattice parameters averaged over the grain thickness along the beam | `PROCESSED_DATA/<sample>/<sample>_quickdaxm_grid`: `layers/*.cor`, `fitfiles_<material>/` |

The wire scans of a sample folder use the calibration made with a scan of the same folder automatically (`CALIBFILE = 'auto'`).

Code: `LaueTools/Daxm/quickdaxm.py` (profiles, dips, geometry), `LaueTools/daxm_workflow.py` (configuration, jobs, files).

## Quick start

1. `1_wires_calibration.ipynb`: in SETTINGS, `RAW_DATA_FOLDER` (experiment), `SCAN_FILTER` and `SCAN_INDEX` (a Ge wire scan, or a wire scan of the sample), then run the cells: crude calibration in seconds, fluorescence refinement with `REFINE_FLUO = True`
2. `2_quick_daxm.ipynb`: same SETTINGS for a wire scan of the sample (`CALIBFILE = 'auto'`: calibration of step 1 found automatically), then run the cells: depth of the Laue spots, peak lists per slice along the beam, browser of the peak lists (gnomonic projection)
3. `3_quick_daxm_grid.ipynb`: `GRID_FILTER` (dataset names of the wire scans of the grid), `MATERIALS`; `MAX_SCANS = 28` to test on the first row; `RESOURCE = 'magnifix3'`: the scans shared between 3 SLURM jobs
4. or from a terminal: `python -m LaueTools.daxm_workflow quickdaxm <results folder>/configs/daxm.yaml --resource magnifix`

### Calibration and p0 shift of each scan

The wires parameters (h, f1, f2, R, u1, u2) describe the wires relative to the beam and the detector: they hold for all the scans as long as wires, beam and detector are not moved. The depth zero is local: it is where the beam enters the reference surface. At another point of the sample, the surface height changes, and the entry point moves along the beam: p0 of all the wires is shifted by this distance (wires moving along the beam). Notebook 3 measures it on each scan: position along the beam of its shallowest grains (envelope of its dips) with the calibration, minus the same for the scan the calibration was made with.

Without notebook:
```python
import LaueTools.daxm_workflow as DW
table = DW.list_wire_scans('/data/visitor/<proposal>/bm32/<date>/RAW_DATA', contains='Zr5_griddaxm_grid')
cfg = DW.config_for_scan(table, DW.scan_index(table, '<dataset>_1'), calibfile='auto')
results = DW.run_quickdaxm(cfg, resource='magnifix')
DW.write_quickdaxm_files(cfg, results)
```

### How the dips are found

The profile of a spot is the mean of its core (3x3 pixels) minus the median of the border of a box around it, image by image. Fluorescence varies slowly on the detector, so it is the same in the core and on the border: its smooth dips cancel and only the sharp dips of the scattering signal remain. A dip is a range of images where this profile is below `low_fraction` of its plateau, limited by edges higher than `min_edge_fraction` of the plateau (rejects intensity fluctuations); the edges are located at half height with sub image accuracy.

### How the depth is computed

At an edge the wire is tangent to the ray going from the source to the pixel. With the wires of the `.calib` file (`CircularWire`, the model of the wires calibration and of the reconstruction), the entry edge and the exit edge each give the position `ybeam` of the source along the beam; they must agree within `max_mismatch`, which also tells which wire made the dip. `ybeam_range` must stay smaller than the spacing of the wires seen from the sample.

`QD.WireGeometry.shadow_lines()` gives the lines *image index of the dip = f(pixel Y)* of each wire for a given `ybeam` (plot of section 4 of the notebook), and `wp_parameters()` the equivalent of the hand-tuned list `WP` of the former quick daxm notebooks.

Units: `ybeam` in microns along the beam (> 0 deeper in the sample, 0 at the reference point of the wires calibration); `depth = ybeam * sin(sample tilt)`.
