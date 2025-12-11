#!/bin/bash

### Task name
#SBATCH --account=your_account_here

### Job name
#SBATCH --job-name=name_of_your_job

### Output file
#SBATCH --output=results/00_slrm_logs/name_of_your_job_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=40

### How much memory in total (MB)
#SBATCH --mem=100G

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email_here

### Maximum runtime per task
#SBATCH --time=24:00:00

### Partition
#SBATCH --partition=standard

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1


set -e  # Exit on error

# Load MATLAB module
module load MATLAB/2025a

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="${SCRIPT_DIR}/snapshots"
LOG_DIR="${SCRIPT_DIR}/logs"

# Configure MATLAB temp directory to use HPC work directory
# (prevents filling up small SSDs with large temporary files)
export MATLAB_TMPDIR="${HPCWORK}/matlab_tmp"
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
PATTERN="gliders"
DELTA_U=0.00002
DELTA_V=0.00001
F=0.014
K=0.054
RANDOM_SEED=1
INIT_TYPE="gaussians"

# Time step parameters
DT=1
SNAP_DT=10
TEND=50

# Get F and k values for the pattern (from the original pattern definitions)
case "${PATTERN}" in
    "gliders")
        F=0.014
        K=0.054
        ;;
    "bubbles")
        F=0.098
        K=0.057
        ;;
    "maze")
        F=0.029
        K=0.057
        ;;
    "worms")
        F=0.058
        K=0.065
        ;;
    "spirals")
        F=0.018
        K=0.051
        ;;
    "spots")
        F=0.03
        K=0.062
        ;;
    *)
        echo "ERROR: Unknown pattern: ${PATTERN}"
        exit 1
        ;;
esac

# Print parameters
echo "Running single simulation with parameters:"
echo "  Pattern:      ${PATTERN}"
echo "  Delta U:      ${DELTA_U}"
echo "  Delta V:      ${DELTA_V}"
echo "  F:            ${F}"
echo "  k:            ${K}"
echo "  Random Seed:  ${RANDOM_SEED}"
echo "  Init Type:    ${INIT_TYPE}"
echo "  Time Step:    ${DT}"
echo "  Snapshot dt:  ${SNAP_DT}"
echo "  Final Time:   ${TEND}"
echo "========================================"

# Log file
LOG_FILE="${LOG_DIR}/${PATTERN}_${INIT_TYPE}_${RANDOM_SEED}_$(date +%Y%m%d_%H%M%S).log"

# Run MATLAB simulation
echo "Starting simulation (log: ${LOG_FILE})"
echo "Progress will be shown below and saved to log file..."
echo ""
${MATLAB_CMD} -batch "addpath('${SCRIPT_DIR}'); addpath('${CHEBFUN_DIR}'); gen_gs('${PATTERN}', ${DELTA_U}, ${DELTA_V}, ${F}, ${K}, ${RANDOM_SEED}, '${INIT_TYPE}', ${DT}, ${SNAP_DT}, ${TEND})" \
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
    num_files=$(find "${SNAPSHOT_DIR}" -name "*.mat" | wc -l)
    echo "Total snapshot files: ${num_files}"
fi

echo "========================================"

exit ${EXIT_CODE}
