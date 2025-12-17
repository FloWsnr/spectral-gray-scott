#!/bin/bash

### Task name
##SBATCH --account=your_account_here

### Job name
#SBATCH --job-name=gray_scott

### Output file
#SBATCH --output=results/slurm_logs/gray_scott_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=1

### How much memory in total (MB)
#SBATCH --mem=5G

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=00:10:00

### Partition
#SBATCH --partition=standard


set -e  # Exit on error

# Load MATLAB module
module load matlab/R2025a

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate phys2vec

# Configuration
# Use SLURM_SUBMIT_DIR (directory from which sbatch was called) for reliability
# Falls back to current directory if not running under SLURM
SCRIPT_DIR="/scratch/zsa8rk/spectral-gray-scott"
SNAPSHOT_DIR="${SCRIPT_DIR}/results/snapshots"
LOG_DIR="${SCRIPT_DIR}/logs"

# Configure MATLAB temp directory to use HPC work directory
# (prevents filling up small SSDs with large temporary files)
export MATLAB_TMPDIR="/scratch/zsa8rk/matlab_tmp"
mkdir -p "${MATLAB_TMPDIR}"

# MATLAB executable with proper HPC flags
MATLAB_CMD="matlab -singleCompThread -nodisplay -nodesktop -nosplash"

# Print banner
echo "========================================"
echo "Gray-Scott Simulation Runner"
echo "========================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Snapshot directory: ${SNAPSHOT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "========================================"

# Create necessary directories
mkdir -p "${SNAPSHOT_DIR}"
mkdir -p "${LOG_DIR}"

# Setup Chebfun
CHEBFUN_DIR="${SCRIPT_DIR}/chebfun"
if [ ! -d "${CHEBFUN_DIR}" ]; then
    echo "Chebfun not found. Installing Chebfun..."
    cd "${SCRIPT_DIR}"
    wget -q https://github.com/chebfun/chebfun/archive/master.zip -O chebfun.zip
    unzip -q chebfun.zip
    mv chebfun-master chebfun
    rm chebfun.zip
    echo "Chebfun installed to ${CHEBFUN_DIR}"
else
    echo "Chebfun found at ${CHEBFUN_DIR}"
fi

# Single simulation parameters
# Set F and k to desired values (examples: gliders F=0.014 k=0.054, bubbles F=0.012 k=0.050)
DELTA_U=0.00002
DELTA_V=0.00001
F=0.078
K=0.061
RANDOM_SEED="[1,2]"  # Use multiple seeds for ensemble runs
INIT_TYPE="gaussians"

# Time step parameters
DT=1
SNAP_DT=10
TEND=2500

# Print parameters
echo "Running single simulation with parameters:"
echo "  F:            ${F}"
echo "  k:            ${K}"
echo "  Delta U:      ${DELTA_U}"
echo "  Delta V:      ${DELTA_V}"
echo "  Random Seed:  ${RANDOM_SEED}"
echo "  Init Type:    ${INIT_TYPE}"
echo "  Time Step:    ${DT}"
echo "  Snapshot dt:  ${SNAP_DT}"
echo "  Final Time:   ${TEND}"
echo "========================================"

# Log file
LOG_FILE="${LOG_DIR}/F${F}_k${K}_${INIT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

# Run MATLAB simulation
echo "Starting simulation (log: ${LOG_FILE})"
echo "Progress will be shown below and saved to log file..."
echo ""
${MATLAB_CMD} -batch "addpath('${SCRIPT_DIR}/simulation'); addpath('${CHEBFUN_DIR}'); gen_gs(${DELTA_U}, ${DELTA_V}, ${F}, ${K}, ${RANDOM_SEED}, '${INIT_TYPE}', ${DT}, ${SNAP_DT}, ${TEND}, '${SNAPSHOT_DIR}')" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

# Summary
echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Simulation Complete - SUCCESS"
else
    echo "Simulation Complete - FAILED (exit code: ${EXIT_CODE})"
    echo "Check log file: ${LOG_FILE}"
fi
echo "========================================"
echo "Results saved to: ${SNAPSHOT_DIR}"
echo "Log saved to: ${LOG_FILE}"

# Count generated files
if [ -d "${SNAPSHOT_DIR}" ]; then
    num_files=$(find "${SNAPSHOT_DIR}" -name "*.h5" | wc -l)
    echo "Total snapshot files: ${num_files}"
fi

echo "========================================"

exit ${EXIT_CODE}
