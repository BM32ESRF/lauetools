#!/bin/bash -l
#SBATCH -n 1                # Number of tasks
#SBATCH -c 120               # CPUs per task
#SBATCH -p magnifix
#SBATCH --time=00:01:00    # Time limit
#SBATCH --mem-per-cpu=80M

module load mamba
conda activate /data/bm32/inhouse/SOFT/lauetoolsDEV

which python
echo 'launching python script for mosaic'
python /data/bm32/inhouse/LAUESCRIPTS/notebooks/LaueTools_script/mosaic_compute.py --config /data/visitor/blc17163/bm32/20260609/RAW_DATA/BaTiO3/BaTiO3_Map/scan0001/mosaic_config.json --ncpus 120