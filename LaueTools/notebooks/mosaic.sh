#!/bin/bash
                #SBATCH -n 1                # Number of tasks
                #SBATCH -c 96               # CPUs per task
                #SBATCH --partition=NONE
                #SBATCH --constraint=frcrg-hpc8  # Constraint
                #SBATCH --time=00:07:00    # Time limit

                module load mamba
                conda activate /data/bm32/inhouse/SOFT/lauetoolsDEV

                which python
                echo 'launching python script for mosaic'
                python /data/bm32/inhouse/LAUESCRIPTS/notebooks/LaueTools_script/mosaic_compute.py --config /data/visitor/blc16859/bm32/20260210/RAW_DATA/NiTi_Heraud/NiTi_Heraud_map/scan0013/mosaic_config.json --ncpus 32