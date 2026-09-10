#!/bin/bash -l
#SBATCH -n 1                # Number of tasks
#SBATCH -c 144               # CPUs per task
#SBATCH -p magnifix
#SBATCH --time=00:20:00    # Time limit
#SBATCH --mem-per-cpu=2100M

# Total RAM per CPU (in MB)
echo "RAM per CPU: $SLURM_MEM_PER_CPU MB"

# Total RAM allocated to your job (in MB)
total_mem_mb=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))
echo "Total RAM allocated: $total_mem_mb MB"

# Convert to GB
total_mem_gb=$(echo "scale=2; $total_mem_mb / 1024" | bc)
echo "Total RAM allocated: $total_mem_gb GB"

# Number of nodes allocated
echo "Nodes allocated: $SLURM_NNODES"


module load mamba
conda activate /data/bm32/inhouse/SOFT/lauetoolsDEV

which python
echo 'launching python script for daxm reconstruction'
python /data/bm32/inhouse/LAUESCRIPTS/notebooks/LaueTools_script/daxmreconstruct_compute.py --config /data/visitor/blc17179/bm32/20260617/PROCESSED_DATA/BaTiO3/BaTiO3_daxmfluo/scan0001/reconstruction/daxm_config.json --ncpus 144