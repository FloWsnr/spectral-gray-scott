#!/bin/bash

### Task name
##SBATCH --account=sds_baek_energetic

### Job name
#SBATCH --job-name=gray_scott_compress

### Output file
#SBATCH --output=results/slurm_logs/gray_scott_compress_%j.out

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

DIR=/scratch/zsa8rk/spectral-gray-scott/results/gray_scott_v2
OUTPUT=/scratch/zsa8rk/spectral-gray-scott/results/gray_scott_v2.tar.gz
NCORES=4  # Match --ntasks-per-node

# Check if directories exist
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR1' does not exist"
    exit 1
fi

echo "Compressing directories with $NCORES cores:"
echo "  - $DIR"
echo "Output: $OUTPUT"

# Create compressed archive using pigz for parallel compression
tar --use-compress-program="pigz -p $NCORES" -cf "$OUTPUT" "$DIR"

if [ $? -eq 0 ]; then
    echo "Successfully created $OUTPUT"
    ls -lh "$OUTPUT"
else
    echo "Error creating zip archive"
    exit 1
fi
