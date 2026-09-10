#!/bin/bash -l
#SBATCH -n 1                # Number of tasks
#SBATCH -c 192               # CPUs per task
#SBATCH -p magnifix
#SBATCH --time=00:06:00    # Time limit
#SBATCH --mem-per-cpu=80M

module load mamba
conda activate /data/bm32/inhouse/SOFT/lauetoolsDEV

which python
echo 'launching python script to load daxm scans'
python /data/bm32/inhouse/SOFT/lauetoolsDEV/lauetools/LaueTools/notebooks/load_daxmscans.py