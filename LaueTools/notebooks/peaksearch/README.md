# Peak search of a map of Laue patterns (LaueTools notebook)

One notebook, one YAML configuration file per experiment:

| notebook | input | output |
|---|---|---|
| `peaksearch_map.ipynb` | images (`.tif`, ...) | `.dat` files (peak lists) and `.cor` files (peak lists + 2θ, χ from the calibration), input of `../indexation/` |

The code is in the LaueTools module `peaksearch_batch.py` (configuration, peak search of 1 image, multiprocessing, SLURM jobs). It replaces the former `PeakSearch_MultiProcessing.ipynb` notebook: same peak search (`readmccd.PeakSearch` with automatic background removal), with all parameters in the configuration file.

## Quick start

1. create `configs/my_experiment.yaml`, either by copying and editing `config_template.yaml` (images, detector and calibration, map dimensions, output folder, peak search parameters), or without editing YAML by hand:
   ```python
   import LaueTools.peaksearch_batch as PB
   PB.new_config('configs/my_experiment.yaml', image_folder='/data/.../RAW_DATA/.../scan0001', prefix='img_',
                 mapdims=[51, 51], detfile='/data/.../calib.det', dat_folder='/data/.../PROCESSED_DATA/.../corfiles',
                 IntensityThreshold=300)
   ```
   (`template='configs/other_experiment.yaml'` starts from another experiment; a misspelled parameter name raises an error). The YAML file is the only reference of the parameters: those of a `.psp` file saved by the LaueTools GUI are written in it with `cfg = PB.import_pspfile(cfg, 'my.psp')`
2. in the first code cell set `CONFIG = 'configs/my_experiment.yaml'`
3. run `peaksearch_map.ipynb`: check the configuration, test the peak search on one image (plot of the found peaks), save the good parameters with `cfg = PB.save_config(cfg, 'configs/my_experiment_v2.yaml', params=params_test)`, then run the batch, locally or on the ESRF cluster (`EXECUTION = 'slurm'`, choice of the machine/partition: magnifix, hpc6/7/8, nice or auto)
4. the last cell writes the configuration file of the indexation (`PB.indexing_config`): continue with `../indexation/1_index_refine.ipynb`

## Large maps: subfolders (option)

By default all `.dat` and `.cor` files are written in `dat_folder`. For large maps (e.g. more than 10000 images), set in the configuration:

```yaml
output:
  nbfiles_per_folder: 10000     # null (default): no subfolder
  subfolder_prefix: images_     # -> dat_folder/images_0_9999/, images_10000_19999/, ...
```

`PB.indexing_config()` copies these values in the indexation configuration (`scan: nbfiles_per_folder`), so that the indexation finds the `.cor` files in the subfolders.

## Batch mode (optional)

The first code cell is tagged `parameters`, so the notebook can run without a browser with [papermill](https://papermill.readthedocs.io):

```bash
papermill peaksearch_map.ipynb run_my_experiment.ipynb -p CONFIG configs/my_experiment.yaml -p NB_CPUS 64
```

or without notebook:

```bash
python -m LaueTools.peaksearch_batch --config configs/my_experiment.yaml --ncpus 64 [--image-range 0 999]
```

## Examples

- `configs/ma6034_Zn.yaml`: Zn dendrites map (41 x 41 images, sCMOS), parameters of the former `PeakSearch_MultiProcessing.ipynb`
- `configs/a321220_MgO_Fe_1000.yaml`: MgO / MgFe2O4 map (51 x 51 images, EIGER 4M), parameters of a `.psp` file of the LaueTools GUI imported in the YAML file. Its indexation: `../indexation/configs/a321220_MgO_Fe_1000.yaml`

They need access to the ESRF `/data` file system.
