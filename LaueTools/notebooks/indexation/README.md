# Indexing a map of Laue patterns (LaueTools notebooks)

Three notebooks, one YAML configuration file per experiment:

| notebook | input | output |
|---|---|---|
| `1_index_refine.ipynb` | `.cor` files (peak search results) | `.fit` files (hkl, UB matrix, strain per grain) |
| `2_postprocess_maps.ipynb` | `.fit` files | maps and statistics: indexing quality, strain, lattice parameters, orientation |
| `3_segmentation_grains.ipynb` | `.fit` files | grains (orientation clusters), KAM, GROD, boundaries |

The code is in the LaueTools modules `indexing_batch.py` (configuration, indexing, multiprocessing) and `orientationclustering.py` (segmentation).

## Quick start

1. create `configs/my_experiment.yaml`, either by copying and editing `config_template.yaml` (folders, detector, map dimensions, material, indexing parameters), or without editing YAML by hand:
   ```python
   import LaueTools.indexing_batch as IB
   IB.new_config('configs/my_experiment.yaml', cor_folder='/data/.../corfiles', prefix='img_', mapdims=[51, 51],
                 fit_folder='/data/.../fitfiles', key_material='Al', nbGrainstoFind=1)
   ```
   (`template='configs/other_experiment.yaml'` starts from another experiment; a misspelled parameter name raises an error)
2. in the first code cell of each notebook set `CONFIG = 'configs/my_experiment.yaml'`
3. run `1_index_refine.ipynb`: check the configuration, test the indexing parameters on one image, save the good ones with `cfg = IB.save_config(cfg, 'configs/my_experiment_v2.yaml', params=params_test)`, then run the batch
4. run `2_postprocess_maps.ipynb` and `3_segmentation_grains.ipynb` with the configuration file used for the batch

Only the configuration file changes from one experiment to another. Keep it next to your results so that you can reproduce the analysis: one file per analysis variant (e.g. nb of grains, `fit_folder`) rather than values overwritten in the notebooks.

## Large maps: subfolders (option)

If the `.cor` files are in subfolders `<cor_folder>/images_0_9999/`, `images_10000_19999/`, ... (peak search with `output: nbfiles_per_folder`, see `../peaksearch/`):

```yaml
scan:
  nbfiles_per_folder: 10000     # null (default): all .cor files in cor_folder
  subfolder_prefix: images_
output:
  fit_subfolders: false         # false (default): all .fit files in fit_folder; true: same subfolders in fit_folder
```

With `fit_subfolders: true`, `2_postprocess_maps.ipynb` and `3_segmentation_grains.ipynb` read the `.fit` files in the subfolders (`IB.fitfiles_layout(cfg)`); `IB.corfile_path(cfg, i)` and `IB.fitfile_path(cfg, i, grainindex)` give the path of the files of image `i`.

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
