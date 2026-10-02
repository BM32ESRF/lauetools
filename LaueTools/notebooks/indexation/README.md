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

## HTML versions

The `.html` files show the notebooks with the outputs of an example run (MgO), anonymized: proposal id, date, sample and dataset names, machine names and local paths are replaced by `<proposal>`, `<date>`, `<sample>`, ... in the code, the printed outputs and the figures.

```bash
# saved outputs (only texts are anonymized: check the figures)
python -m LaueTools.scripts.notebook_to_html 1_index_refine.ipynb --anonymize -p CONFIG=configs/my_experiment.yaml
# notebooks executed again (temporary copy, static figures, texts of the figures anonymized)
python -m LaueTools.scripts.notebook_to_html 2_postprocess_maps.ipynb 3_segmentation_grains.ipynb \
    --anonymize --execute -p CONFIG=configs/my_experiment.yaml
```

Names to be replaced are found in the `/data/visitor/` paths of the notebook and of the configuration file. Other names (e.g. a sample name used alone in a title) can be added in a YAML file: `--anonymize my_rules.yaml` (see the docstring of `LaueTools/scripts/notebook_to_html.py`). `1_index_refine.ipynb` is not executed again: it would index the whole map (or submit SLURM jobs) again.
