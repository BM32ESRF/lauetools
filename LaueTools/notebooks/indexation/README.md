# Indexing a map of Laue patterns (LaueTools notebooks)

Three notebooks, one YAML configuration file per experiment:

| notebook | input | output |
|---|---|---|
| `1_index_refine.ipynb` | `.cor` files (peak search results) | `.fit` files (hkl, UB matrix, strain per grain) |
| `2_postprocess_maps.ipynb` | `.fit` files | maps and statistics: indexing quality, strain, lattice parameters, orientation |
| `3_segmentation_grains.ipynb` | `.fit` files | grains (orientation clusters), KAM, GROD, boundaries |

The code is in the LaueTools modules `indexing_batch.py` (configuration, indexing, multiprocessing) and `orientationclustering.py` (segmentation).

## Quick start

1. copy `config_template.yaml` to `configs/my_experiment.yaml` and edit it (folders, detector, map dimensions, material, indexing parameters)
2. in the first code cell of each notebook set `CONFIG = 'configs/my_experiment.yaml'`
3. run `1_index_refine.ipynb`: check the configuration, test the indexing parameters on one image, then run the batch
4. run `2_postprocess_maps.ipynb` and `3_segmentation_grains.ipynb`

Only the configuration file changes from one experiment to another. Keep it next to your results so that you can reproduce the analysis.

## Batch mode (optional)

The first code cell of each notebook is tagged `parameters`, so the notebooks can run without a browser with [papermill](https://papermill.readthedocs.io):

```bash
papermill 1_index_refine.ipynb run_my_experiment.ipynb -p CONFIG configs/my_experiment.yaml -p NB_CPUS 64
```

## Examples

`configs/` contains the configurations of BM32 datasets (Al/Al2Cu, MgO, Zr, Zn, ZrO2). They need access to the ESRF `/data` file system.
