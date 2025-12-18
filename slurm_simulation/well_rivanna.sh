#!/bin/bash

### Task name
##SBATCH --account=sds_baek_energetic

### Job name
#SBATCH --job-name=gray_scott_well_format

### Output file
#SBATCH --output=results/slurm_logs/gray_scott_well_format_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=4

### How much memory in total (MB)
#SBATCH --mem=40G

### Mail notification configuration
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=24:00:00

### Partition
#SBATCH --partition=standard


set -e  # Exit on error

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate phys2vec

python_bin=/home/zsa8rk/miniforge3/envs/phys2vec/bin/python
dir_1=/scratch/zsa8rk/spectral-gray-scott/results/snapshots1
dir_2=/scratch/zsa8rk/spectral-gray-scott/results/snapshots2
out=/scratch/zsa8rk/spectral-gray-scott/results/well_format
workers=4

${python_bin} simulation/well_convert_combined.py --snapshots-dir1 ${dir_1} --snapshots-dir2 ${dir_2} --output-dir ${out} --workers ${workers}
