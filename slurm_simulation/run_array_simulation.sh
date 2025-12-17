#!/bin/bash

### Task name
##SBATCH --account=sds_baek_energetic

### Job name
#SBATCH --job-name=gs_array

### Output file (unique per array task: %A=job_id, %a=array_task_id)
#SBATCH --output=results/slurm_logs/array_%A_%a.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=4

### How much memory in total (MB)
#SBATCH --mem=10G  # Increased from 5G for multi-seed support (100 seeds)

### Mail notification configuration
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=01:00:00  # Increased from 00:10:00 for multi-seed support (16 hours)

### Partition
#SBATCH --partition=standard

### Array job directive (set this when submitting via command line)
### Example: sbatch --array=1-1000%50 run_array_simulation.sh params.csv
### This runs jobs 1-1000 with max 50 concurrent jobs

set -e  # Exit on error

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate phys2vec
# ============================================================================
# ARRAY JOB PARAMETER READING
# ============================================================================


# Parameter file is passed as first argument
PARAM_FILE="${1:-params.csv}"

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: This script must be run as a SLURM array job"
    echo "Usage: sbatch --array=1-N%M run_array_simulation.sh params.csv"
    exit 1
fi

if [ ! -f "${PARAM_FILE}" ]; then
    echo "ERROR: Parameter file not found: ${PARAM_FILE}"
    echo "Usage: sbatch --array=1-N%M run_array_simulation.sh params.csv"
    exit 1
fi

# Validate array task ID against parameter file size
NUM_PARAMS=$(tail -n +2 "${PARAM_FILE}" | wc -l)
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${NUM_PARAMS}" ]; then
    echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds number of parameter rows ${NUM_PARAMS}"
    exit 1
fi

# Read parameters for this job using Python to handle CSV quoting properly
# Python respects quoted fields, unlike simple bash IFS=',' parsing
read JOB_ID DELTA_U DELTA_V F K RANDOM_SEEDS INIT_TYPE DT SNAP_DT TEND <<< $(python3 -c "
import csv
import sys
with open('${PARAM_FILE}', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=1):
        if i == ${SLURM_ARRAY_TASK_ID}:
            print(row['job_id'], row['delta_u'], row['delta_v'], row['F'], row['k'],
                  row['random_seeds'], row['init_type'], row['dt'], row['snap_dt'], row['tend'])
            break
")

# Parse comma-separated seeds into MATLAB array format
# Input:  "1,2,3,4,5" or "1" (CSV may have quotes or not)
# Output: [1,2,3,4,5] or [1]

# Remove any quotes and spaces that CSV might have added
RANDOM_SEEDS_CLEAN=$(echo "${RANDOM_SEEDS}" | tr -d '"' | tr -d "'" | tr -d ' ')

# Validate not empty
if [ -z "${RANDOM_SEEDS_CLEAN}" ]; then
    echo "ERROR: Empty random_seeds in line ${LINE_NUM}"
    echo "Line content: ${PARAMS}"
    exit 1
fi

# Create MATLAB array syntax: "1,2,3" -> "[1,2,3]"
MATLAB_SEED_ARRAY="[${RANDOM_SEEDS_CLEAN}]"
NUM_SEEDS=$(echo "${RANDOM_SEEDS_CLEAN}" | awk -F',' '{print NF}')

echo "Parsed ${NUM_SEEDS} random seeds: ${MATLAB_SEED_ARRAY}"

# Verify we read parameters correctly
if [ -z "${F}" ] || [ -z "${K}" ]; then
    echo "ERROR: Failed to parse parameters from line ${LINE_NUM}"
    echo "Line content: ${PARAMS}"
    exit 1
fi

# ============================================================================
# SETUP (copied from original run_simulation.sh)
# ============================================================================

# Configuration
# Use SLURM_SUBMIT_DIR (directory from which sbatch was called) for reliability
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
SNAPSHOT_DIR="${SCRIPT_DIR}/results/snapshots"
LOG_DIR="${SCRIPT_DIR}/logs"
STATUS_DIR="${SCRIPT_DIR}/results/job_status"

# Configure MATLAB temp directory to use HPC work directory
# (prevents filling up small SSDs with large temporary files)
export MATLAB_TMPDIR="${HPCWORK}/matlab_tmp"
mkdir -p "${MATLAB_TMPDIR}"

# Load MATLAB module
module load MATLAB/2025a

# MATLAB executable with proper HPC flags
MATLAB_CMD="matlab -nodisplay -nodesktop -nosplash"

# Print banner
echo "========================================"
echo "Gray-Scott Array Job Simulation"
echo "========================================"
echo "SLURM Job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Parameter File:   ${PARAM_FILE}"
echo "Script directory: ${SCRIPT_DIR}"
echo "Snapshot directory: ${SNAPSHOT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Status directory: ${STATUS_DIR}"
echo "========================================"

# Create necessary directories
mkdir -p "${SNAPSHOT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${STATUS_DIR}"

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

# ============================================================================
# JOB STATUS TRACKING
# ============================================================================

STATUS_FILE="${STATUS_DIR}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.status"

# Mark job as started
{
    echo "STARTED: $(date)"
    echo "JOB_ID: ${JOB_ID}"
    echo "ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
    echo "ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
    echo "F: ${F}"
    echo "k: ${K}"
    echo "DELTA_U: ${DELTA_U}"
    echo "DELTA_V: ${DELTA_V}"
    echo "RANDOM_SEEDS: ${RANDOM_SEEDS_CLEAN}"
    echo "NUM_SEEDS: ${NUM_SEEDS}"
    echo "INIT_TYPE: ${INIT_TYPE}"
    echo "DT: ${DT}"
    echo "SNAP_DT: ${SNAP_DT}"
    echo "TEND: ${TEND}"
} > "${STATUS_FILE}"

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Print parameters
echo "Running simulation with parameters:"
echo "  Job ID:       ${JOB_ID}"
echo "  F:            ${F}"
echo "  k:            ${K}"
echo "  Delta U:      ${DELTA_U}"
echo "  Delta V:      ${DELTA_V}"
echo "  Random Seeds: ${RANDOM_SEEDS_CLEAN} (${NUM_SEEDS} total)"
echo "  Init Type:    ${INIT_TYPE}"
echo "  Time Step:    ${DT}"
echo "  Snapshot dt:  ${SNAP_DT}"
echo "  Final Time:   ${TEND}"
echo "========================================"

# Log file (unique per job)
LOG_FILE="${LOG_DIR}/job_${JOB_ID}_${INIT_TYPE}_F${F}_k${K}_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# RUN SIMULATION
# ============================================================================

echo "Starting simulation (log: ${LOG_FILE})"
echo "Progress will be shown below and saved to log file..."
echo ""

# Run MATLAB simulation
# Pass MATLAB array syntax for seeds: [1,2,3,...,100]
${MATLAB_CMD} -batch "addpath('${SCRIPT_DIR}/simulation'); addpath('${CHEBFUN_DIR}'); gen_gs(${DELTA_U}, ${DELTA_V}, ${F}, ${K}, ${MATLAB_SEED_ARRAY}, '${INIT_TYPE}', ${DT}, ${SNAP_DT}, ${TEND}, '${SNAPSHOT_DIR}')" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

# ============================================================================
# UPDATE STATUS AND SUMMARY
# ============================================================================

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Simulation Complete - SUCCESS"
    echo "SUCCESS: $(date)" >> "${STATUS_FILE}"
    echo "EXIT_CODE: 0" >> "${STATUS_FILE}"
    echo "LOG_FILE: ${LOG_FILE}" >> "${STATUS_FILE}"
else
    echo "Simulation Complete - FAILED (exit code: ${EXIT_CODE})"
    echo "Check log file: ${LOG_FILE}"
    echo "FAILED: $(date)" >> "${STATUS_FILE}"
    echo "EXIT_CODE: ${EXIT_CODE}" >> "${STATUS_FILE}"
    echo "LOG_FILE: ${LOG_FILE}" >> "${STATUS_FILE}"
fi
echo "========================================"
echo "Results saved to: ${SNAPSHOT_DIR}"
echo "Log saved to: ${LOG_FILE}"
echo "Status saved to: ${STATUS_FILE}"

# Count generated files for this simulation
# Use parameter-based directory structure (gen_gs.m creates this)
F_FMT=$(printf "%.3f" ${F})
K_FMT=$(printf "%.3f" ${K})
DU_FMT=$(printf "%.1e" ${DELTA_U})
DV_FMT=$(printf "%.1e" ${DELTA_V})

OUTPUT_DIR="${SNAPSHOT_DIR}/F${F_FMT}_k${K_FMT}_du${DU_FMT}_dv${DV_FMT}_${INIT_TYPE}"

if [ -d "${OUTPUT_DIR}" ]; then
    num_files=$(find "${OUTPUT_DIR}" -name "*.h5" 2>/dev/null | wc -l)
    echo "Snapshot files generated: ${num_files}"
    echo "Output directory: ${OUTPUT_DIR}"
fi

echo "========================================"

exit ${EXIT_CODE}
