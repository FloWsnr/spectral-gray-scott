#!/usr/bin/env python3
"""
Check status of SLURM array job simulations.

This script analyzes job status files and parameter files to determine
which jobs have completed, failed, or are still pending.

Usage:
    # Basic status report
    python check_job_status.py

    # With parameter file cross-reference
    python check_job_status.py --params params.csv

    # Output only failed job IDs (for resubmission)
    python check_job_status.py --failed-ids-only

    # Detailed report with missing outputs
    python check_job_status.py --detailed --params params.csv
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


def find_status_files(status_dir):
    """Find all job status files."""
    status_dir = Path(status_dir)
    if not status_dir.exists():
        return []

    return list(status_dir.glob("job_*.status"))


def parse_status_file(status_file):
    """Parse a job status file and extract information."""
    try:
        with open(status_file, 'r') as f:
            content = f.read()

        info = {}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()

        # Determine status
        if 'SUCCESS' in content:
            status = 'completed'
        elif 'FAILED' in content:
            status = 'failed'
        elif 'STARTED' in content:
            status = 'running'
        else:
            status = 'unknown'

        info['status'] = status

        # Extract array task ID from filename
        # Format: job_<ARRAY_JOB_ID>_<ARRAY_TASK_ID>.status
        filename = status_file.stem  # Remove .status extension
        parts = filename.split('_')
        if len(parts) >= 3:
            info['array_task_id'] = int(parts[2])

        return info
    except Exception as e:
        print(f"Warning: Failed to parse {status_file}: {e}", file=sys.stderr)
        return None


def analyze_status_files(status_dir):
    """Analyze all status files and return summary."""
    status_files = find_status_files(status_dir)

    if not status_files:
        return None, []

    # Parse all status files
    job_info = []
    for status_file in status_files:
        info = parse_status_file(status_file)
        if info:
            info['status_file'] = status_file
            job_info.append(info)

    # Count by status
    status_counts = defaultdict(int)
    for info in job_info:
        status_counts[info['status']] += 1

    return status_counts, job_info


def check_output_files(param_file, snapshot_dir, array_job_id=None):
    """Check which jobs have output files.

    Args:
        param_file: Path to parameter CSV file
        snapshot_dir: Path to snapshots directory
        array_job_id: SLURM array job ID (if None, will try to infer from status files)
    """
    snapshot_dir = Path(snapshot_dir)

    if not param_file or not snapshot_dir.exists():
        return {}

    df = pd.read_csv(param_file)

    output_status = {}
    for _, row in df.iterrows():
        job_id = int(row['job_id'])

        # Check new directory structure first: snapshots/{array_job_id}/{job_id}/
        if array_job_id:
            new_output_dir = snapshot_dir / str(array_job_id) / str(job_id)
            data_file = new_output_dir / "data.h5"

            if new_output_dir.exists():
                output_status[job_id] = {
                    'has_output': True,
                    'output_dir': new_output_dir,
                    'has_data_file': data_file.exists()
                }
                continue

        # Fall back to old naming scheme for backward compatibility
        # Format: gs_F={F}_k={k}_{init_type}_{random_seed}
        F_formatted = f"{row['F']:.3f}".replace('.', '')[:3]
        k_formatted = f"{row['k']:.3f}".replace('.', '')[:3]

        pattern = f"gs_F={F_formatted}_k={k_formatted}_{row['init_type']}_{int(row['random_seed'])}"

        # Look for matching directories
        matching_dirs = list(snapshot_dir.glob(pattern))

        if matching_dirs:
            # Check if data.h5 exists
            data_file = matching_dirs[0] / "data.h5"
            output_status[job_id] = {
                'has_output': data_file.exists(),
                'output_dir': matching_dirs[0],
                'has_data_file': data_file.exists()
            }
        else:
            output_status[job_id] = {
                'has_output': False,
                'output_dir': None,
                'has_data_file': False
            }

    return output_status


def print_status_report(status_counts, job_info, total_jobs=None, detailed=False, output_status=None):
    """Print formatted status report."""
    print("\n" + "=" * 70)
    print("Job Status Report")
    print("=" * 70)

    if not status_counts:
        print("No status files found.")
        if total_jobs:
            print(f"Expected {total_jobs} jobs - none have started yet.")
        return

    # Calculate totals
    num_completed = status_counts.get('completed', 0)
    num_failed = status_counts.get('failed', 0)
    num_running = status_counts.get('running', 0)
    num_unknown = status_counts.get('unknown', 0)
    num_tracked = num_completed + num_failed + num_running + num_unknown

    if total_jobs:
        num_pending = total_jobs - num_tracked
    else:
        num_pending = 0
        total_jobs = num_tracked

    # Print summary
    print(f"\nTotal jobs:        {total_jobs}")
    print(f"Completed:         {num_completed:5d}  ({num_completed/total_jobs*100:5.1f}%)")
    print(f"Failed:            {num_failed:5d}  ({num_failed/total_jobs*100:5.1f}%)")
    print(f"Running:           {num_running:5d}  ({num_running/total_jobs*100:5.1f}%)")

    if total_jobs > num_tracked:
        print(f"Pending:           {num_pending:5d}  ({num_pending/total_jobs*100:5.1f}%)")

    if num_unknown > 0:
        print(f"Unknown status:    {num_unknown:5d}")

    # Print failed job IDs
    if num_failed > 0:
        failed_jobs = [info for info in job_info if info['status'] == 'failed']
        failed_task_ids = sorted([info['array_task_id'] for info in failed_jobs if 'array_task_id' in info])

        print(f"\nFailed job array task IDs:")
        # Format as comma-separated list
        print(f"  {','.join(map(str, failed_task_ids))}")

    # Check for missing outputs
    if output_status:
        missing_outputs = []
        for job_id, status in output_status.items():
            if not status['has_data_file']:
                # Check if job completed
                job_status = next((info for info in job_info if info.get('JOB_ID') == str(job_id)), None)
                if job_status and job_status['status'] == 'completed':
                    missing_outputs.append(job_id)

        if missing_outputs:
            print(f"\nWarning: {len(missing_outputs)} completed jobs missing output files:")
            for job_id in missing_outputs[:10]:
                print(f"  Job ID {job_id}")
            if len(missing_outputs) > 10:
                print(f"  ... and {len(missing_outputs) - 10} more")

    # Detailed output
    if detailed:
        print("\n" + "-" * 70)
        print("Detailed Job Information")
        print("-" * 70)

        # Group by status
        for status in ['completed', 'running', 'failed']:
            jobs = [info for info in job_info if info['status'] == status]
            if jobs:
                print(f"\n{status.upper()} ({len(jobs)} jobs):")
                for info in sorted(jobs, key=lambda x: x.get('array_task_id', 0))[:20]:
                    task_id = info.get('array_task_id', '?')
                    job_id = info.get('JOB_ID', '?')
                    F = info.get('F', '?')
                    k = info.get('k', '?')
                    init_type = info.get('INIT_TYPE', '?')
                    print(f"  Task {task_id:4s} (Job {job_id:4s}): F={F:6s} k={k:6s} init={init_type:8s}")

                if len(jobs) > 20:
                    print(f"  ... and {len(jobs) - 20} more")

    print("=" * 70 + "\n")


def get_failed_job_ids(job_info, output_format='comma'):
    """Get list of failed job array task IDs."""
    failed_jobs = [info for info in job_info if info['status'] == 'failed']
    failed_task_ids = sorted([info['array_task_id'] for info in failed_jobs if 'array_task_id' in info])

    if output_format == 'comma':
        return ','.join(map(str, failed_task_ids))
    elif output_format == 'list':
        return failed_task_ids
    else:
        return '\n'.join(map(str, failed_task_ids))


def main():
    parser = argparse.ArgumentParser(
        description='Check status of Gray-Scott array job simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--status-dir', type=str,
                        default='results/job_status',
                        help='Directory containing job status files (default: results/job_status)')
    parser.add_argument('--params', type=str,
                        help='Parameter CSV file (for cross-reference and total job count)')
    parser.add_argument('--snapshot-dir', type=str,
                        default='results/snapshots',
                        help='Directory containing simulation outputs (default: results/snapshots)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed job information')
    parser.add_argument('--check-outputs', action='store_true',
                        help='Check for missing output files')
    parser.add_argument('--failed-ids-only', action='store_true',
                        help='Output only failed job IDs (comma-separated, for resubmission)')

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent

    # Resolve paths
    status_dir = script_dir / args.status_dir
    snapshot_dir = script_dir / args.snapshot_dir

    # Analyze status files
    status_counts, job_info = analyze_status_files(status_dir)

    # Get total job count from parameter file
    total_jobs = None
    if args.params:
        param_file = Path(args.params)
        if param_file.exists():
            df = pd.read_csv(param_file)
            total_jobs = len(df)
        else:
            print(f"Warning: Parameter file not found: {param_file}", file=sys.stderr)

    # Check output files if requested
    output_status = None
    array_job_id = None
    if args.check_outputs and args.params:
        param_file = Path(args.params)
        if param_file.exists():
            # Try to extract array job ID from status files
            if job_info:
                # Get the most common array job ID from status files
                array_job_ids = [info.get('ARRAY_JOB_ID') for info in job_info if 'ARRAY_JOB_ID' in info]
                if array_job_ids:
                    array_job_id = max(set(array_job_ids), key=array_job_ids.count)

            output_status = check_output_files(param_file, snapshot_dir, array_job_id)

    # Output mode
    if args.failed_ids_only:
        if job_info:
            print(get_failed_job_ids(job_info, output_format='comma'))
        sys.exit(0)

    # Print report
    print_status_report(status_counts, job_info, total_jobs, args.detailed, output_status)

    # Exit code based on failures
    if status_counts and status_counts.get('failed', 0) > 0:
        print("Resubmit failed jobs with:")
        print(f"  bash resume_failed_jobs.sh {args.params if args.params else 'params.csv'}")
        print()


if __name__ == '__main__':
    main()
