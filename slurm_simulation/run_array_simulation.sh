#!/bin/bash

### Task name
##SBATCH --account=fw641779

### Job name
#SBATCH --job-name=gs_array

### Output file (unique per array task: %A=job_id, %a=array_task_id)
#SBATCH --output=results/slurm_logs/array_%A_%a.out

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

### Array job directive (set this when submitting via command line)
### Example: sbatch --array=1-1000%50 run_array_simulation.sh params.csv
### This runs jobs 1-1000 with max 50 concurrent jobs

set -e  # Exit on error

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate base
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

# Read parameters for this job (add 1 to line number to skip header)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
PARAMS=$(sed -n "${LINE_NUM}p" "${PARAM_FILE}")

# Parse CSV fields (job_id,delta_u,delta_v,F,k,random_seed,init_type,dt,snap_dt,tend)
IFS=',' read -r JOB_ID DELTA_U DELTA_V F K RANDOM_SEED INIT_TYPE DT SNAP_DT TEND <<< "${PARAMS}"

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
MATLAB_CMD="matlab -singleCompThread -nodisplay -nodesktop -nosplash"

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
    echo "RANDOM_SEED: ${RANDOM_SEED}"
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
echo "  Random Seed:  ${RANDOM_SEED}"
echo "  Init Type:    ${INIT_TYPE}"
echo "  Time Step:    ${DT}"
echo "  Snapshot dt:  ${SNAP_DT}"
echo "  Final Time:   ${TEND}"
echo "========================================"

# Log file (unique per job)
LOG_FILE="${LOG_DIR}/${INIT_TYPE}_${RANDOM_SEED}_F${F}_k${K}_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# RUN SIMULATION
# ============================================================================

echo "Starting simulation (log: ${LOG_FILE})"
echo "Progress will be shown below and saved to log file..."
echo ""

# Run MATLAB simulation (pass array job ID and job ID for directory naming)
${MATLAB_CMD} -batch "addpath('${SCRIPT_DIR}/simulation'); addpath('${CHEBFUN_DIR}'); gen_gs(${DELTA_U}, ${DELTA_V}, ${F}, ${K}, ${RANDOM_SEED}, '${INIT_TYPE}', ${DT}, ${SNAP_DT}, ${TEND}, '${SLURM_ARRAY_JOB_ID}', '${JOB_ID}')" \
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
OUTPUT_DIR="${SNAPSHOT_DIR}/${SLURM_ARRAY_JOB_ID}/${JOB_ID}"
if [ -d "${OUTPUT_DIR}" ]; then
    num_files=$(find "${OUTPUT_DIR}" -name "*.h5" 2>/dev/null | wc -l)
    echo "Snapshot files generated: ${num_files}"
    echo "Output directory: ${OUTPUT_DIR}"
fi

echo "========================================"

exit ${EXIT_CODE}
