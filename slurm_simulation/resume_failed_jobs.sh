#!/bin/bash
#
# Resume failed jobs from a SLURM array job submission.
#
# This script identifies failed jobs from the status files and resubmits
# only those jobs using a new SLURM array job.
#
# Usage:
#   bash resume_failed_jobs.sh params.csv
#

set -e

# Get parameter file from argument
PARAM_FILE="${1}"

if [ -z "${PARAM_FILE}" ]; then
    echo "Usage: $0 <parameter_file.csv>"
    echo ""
    echo "Example:"
    echo "  bash resume_failed_jobs.sh params.csv"
    exit 1
fi

if [ ! -f "${PARAM_FILE}" ]; then
    echo "ERROR: Parameter file not found: ${PARAM_FILE}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if check_job_status.py exists
if [ ! -f "${SCRIPT_DIR}/check_job_status.py" ]; then
    echo "ERROR: check_job_status.py not found in ${SCRIPT_DIR}"
    exit 1
fi

# Get failed job IDs
echo "Checking for failed jobs..."
FAILED_IDS=$(python3 "${SCRIPT_DIR}/check_job_status.py" --params "${PARAM_FILE}" --failed-ids-only)

if [ -z "${FAILED_IDS}" ]; then
    echo "No failed jobs found. Nothing to resubmit."
    echo ""
    echo "Run 'python check_job_status.py --params ${PARAM_FILE}' for detailed status."
    exit 0
fi

# Count number of failed jobs
NUM_FAILED=$(echo "${FAILED_IDS}" | tr ',' '\n' | wc -l)

echo "Found ${NUM_FAILED} failed job(s):"
echo "  Array task IDs: ${FAILED_IDS}"
echo ""

# Confirm resubmission
read -p "Resubmit these ${NUM_FAILED} failed jobs? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Resubmission cancelled."
    exit 0
fi

# Check if run_array_simulation.sh exists
if [ ! -f "${SCRIPT_DIR}/run_array_simulation.sh" ]; then
    echo "ERROR: run_array_simulation.sh not found in ${SCRIPT_DIR}"
    exit 1
fi

# Submit array job with only failed job IDs
echo "Submitting SLURM array job for failed jobs..."
echo "Command: sbatch --array=${FAILED_IDS} ${SCRIPT_DIR}/run_array_simulation.sh ${PARAM_FILE}"
echo ""

sbatch --array="${FAILED_IDS}" "${SCRIPT_DIR}/run_array_simulation.sh" "${PARAM_FILE}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Successfully resubmitted ${NUM_FAILED} failed job(s)."
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -u \$USER"
    echo "  python check_job_status.py --params ${PARAM_FILE}"
else
    echo ""
    echo "ERROR: Failed to submit job array (exit code: ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi
