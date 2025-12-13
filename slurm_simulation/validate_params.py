#!/usr/bin/env python3
"""
Validate parameter file before submitting SLURM array jobs.

This script performs pre-flight checks on the parameter CSV file to catch
errors before submitting large job arrays.

Usage:
    python validate_params.py params.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import shutil


def check_csv_format(param_file):
    """Check if CSV file is properly formatted."""
    try:
        df = pd.read_csv(param_file)
        return True, df, None
    except Exception as e:
        return False, None, f"Failed to parse CSV: {e}"


def check_required_columns(df):
    """Check if all required columns are present."""
    required_cols = ['job_id', 'pattern', 'delta_u', 'delta_v', 'F', 'k',
                     'random_seed', 'init_type', 'dt', 'snap_dt', 'tend']

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return False, f"Missing required columns: {missing}"

    return True, None


def check_missing_values(df):
    """Check for missing values in any column."""
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    if len(missing_cols) > 0:
        details = []
        for col, count in missing_cols.items():
            details.append(f"  {col}: {count} missing values")
        return False, "Found missing values:\n" + "\n".join(details)

    return True, None


def check_parameter_ranges(df):
    """Validate parameter value ranges."""
    errors = []

    # Check valid pattern names
    valid_patterns = ['gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots']
    invalid_patterns = df[~df['pattern'].isin(valid_patterns)]['pattern'].unique()
    if len(invalid_patterns) > 0:
        errors.append(f"Invalid pattern names: {list(invalid_patterns)}")
        errors.append(f"  Valid patterns: {valid_patterns}")

    # Check valid init types
    valid_init_types = ['gaussians', 'fourier']
    invalid_init = df[~df['init_type'].isin(valid_init_types)]['init_type'].unique()
    if len(invalid_init) > 0:
        errors.append(f"Invalid init_type values: {list(invalid_init)}")
        errors.append(f"  Valid init types: {valid_init_types}")

    # Check positive parameters
    positive_params = ['delta_u', 'delta_v', 'F', 'k', 'dt', 'snap_dt', 'tend']
    for param in positive_params:
        if param in df.columns:
            negative = (df[param] <= 0).sum()
            if negative > 0:
                errors.append(f"Parameter '{param}' must be positive ({negative} invalid values)")

    # Check reasonable ranges (warnings, not errors)
    warnings = []

    # Typical diffusion coefficients
    if (df['delta_u'] > 0.001).any():
        warnings.append("delta_u values > 0.001 detected (typically ~0.00002)")
    if (df['delta_v'] > 0.001).any():
        warnings.append("delta_v values > 0.001 detected (typically ~0.00001)")

    # Typical F and k ranges
    if (df['F'] > 0.15).any():
        warnings.append("F values > 0.15 detected (typical range: 0.01-0.1)")
    if (df['k'] > 0.1).any():
        warnings.append("k values > 0.1 detected (typical range: 0.05-0.065)")

    if errors:
        return False, "\n".join(errors), warnings
    else:
        return True, None, warnings


def check_duplicates(df):
    """Check for duplicate parameter combinations."""
    param_cols = [col for col in df.columns if col != 'job_id']
    duplicates = df[param_cols].duplicated()
    num_duplicates = duplicates.sum()

    if num_duplicates > 0:
        dup_rows = df[duplicates]['job_id'].tolist()
        return False, f"Found {num_duplicates} duplicate parameter combinations (job_ids: {dup_rows[:10]}...)"

    return True, None


def estimate_resources(df):
    """Estimate resource requirements for the job array."""
    num_jobs = len(df)

    # Typical values from run_simulation.sh
    cores_per_job = 40
    memory_per_job_gb = 100
    time_per_job_hours = 24
    output_size_mb = 0.8  # Approximate size per output

    # Calculate totals
    total_core_hours = num_jobs * cores_per_job * time_per_job_hours
    total_memory_gb = num_jobs * memory_per_job_gb  # If all ran simultaneously
    total_storage_gb = num_jobs * output_size_mb / 1024

    # Estimate wall time for different concurrency levels
    concurrency_levels = [10, 25, 50, 100]
    wall_times = {}
    for concurrency in concurrency_levels:
        if concurrency <= num_jobs:
            wall_time_hours = (num_jobs / concurrency) * time_per_job_hours
            wall_times[concurrency] = wall_time_hours

    return {
        'num_jobs': num_jobs,
        'total_core_hours': total_core_hours,
        'total_memory_gb': total_memory_gb,
        'total_storage_gb': total_storage_gb,
        'wall_times': wall_times,
        'cores_per_job': cores_per_job,
        'memory_per_job_gb': memory_per_job_gb,
        'time_per_job_hours': time_per_job_hours
    }


def check_disk_space(estimated_storage_gb, work_dir):
    """Check if there's enough disk space available."""
    try:
        usage = shutil.disk_usage(work_dir)
        available_gb = usage.free / (1024**3)

        if estimated_storage_gb > 0.8 * available_gb:
            return False, f"Estimated storage ({estimated_storage_gb:.1f} GB) exceeds 80% of available space ({available_gb:.1f} GB)"
        elif estimated_storage_gb > 0.5 * available_gb:
            return True, f"Warning: Estimated storage ({estimated_storage_gb:.1f} GB) is >50% of available space ({available_gb:.1f} GB)"
        else:
            return True, None
    except Exception as e:
        return True, f"Could not check disk space: {e}"


def print_parameter_summary(df):
    """Print summary statistics for each parameter."""
    print("\nParameter Summary:")
    print("=" * 70)

    for col in ['pattern', 'delta_u', 'delta_v', 'F', 'k', 'random_seed',
                'init_type', 'dt', 'snap_dt', 'tend']:
        if col not in df.columns:
            continue

        unique_vals = df[col].unique()

        if col in ['pattern', 'init_type']:
            # Categorical parameters
            value_counts = df[col].value_counts()
            print(f"\n{col}:")
            for val, count in value_counts.items():
                print(f"  {val}: {count} jobs")
        else:
            # Numerical parameters
            if len(unique_vals) == 1:
                print(f"\n{col}: {unique_vals[0]} (fixed)")
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                num_unique = len(unique_vals)
                print(f"\n{col}:")
                print(f"  Range: {min_val} to {max_val}")
                print(f"  Unique values: {num_unique}")
                if num_unique <= 10:
                    print(f"  Values: {sorted(unique_vals)}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate parameter file for Gray-Scott array jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('param_file', type=str,
                        help='Path to parameter CSV file')
    parser.add_argument('--no-disk-check', action='store_true',
                        help='Skip disk space check')
    parser.add_argument('--summary', action='store_true',
                        help='Show detailed parameter summary')

    args = parser.parse_args()

    param_file = Path(args.param_file)

    if not param_file.exists():
        print(f"ERROR: File not found: {param_file}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Parameter File Validation")
    print("=" * 70)
    print(f"File: {param_file}")
    print()

    # Track validation results
    errors = []
    warnings = []

    # Check 1: CSV format
    print("Checking CSV format...", end=" ")
    ok, df, error = check_csv_format(param_file)
    if not ok:
        print("FAILED")
        errors.append(error)
        print(f"  {error}")
        sys.exit(1)
    print("OK")

    # Check 2: Required columns
    print("Checking required columns...", end=" ")
    ok, error = check_required_columns(df)
    if not ok:
        print("FAILED")
        errors.append(error)
        print(f"  {error}")
    else:
        print("OK")

    # Check 3: Missing values
    print("Checking for missing values...", end=" ")
    ok, error = check_missing_values(df)
    if not ok:
        print("FAILED")
        errors.append(error)
        print(f"  {error}")
    else:
        print("OK")

    # Check 4: Parameter ranges
    print("Validating parameter ranges...", end=" ")
    ok, error, warns = check_parameter_ranges(df)
    if not ok:
        print("FAILED")
        errors.append(error)
        print(f"  {error}")
    else:
        print("OK")
        if warns:
            warnings.extend(warns)

    # Check 5: Duplicates
    print("Checking for duplicates...", end=" ")
    ok, error = check_duplicates(df)
    if not ok:
        print("WARNING")
        warnings.append(error)
        print(f"  {error}")
    else:
        print("OK")

    # Estimate resources
    print("\nEstimating resource requirements...")
    resources = estimate_resources(df)

    print(f"\n  Total jobs: {resources['num_jobs']}")
    print(f"  Total CPU core-hours: {resources['total_core_hours']:,}")
    print(f"  Estimated storage: {resources['total_storage_gb']:.2f} GB")
    print(f"  Per-job resources: {resources['cores_per_job']} cores, {resources['memory_per_job_gb']} GB RAM, {resources['time_per_job_hours']}h time limit")

    print(f"\n  Estimated wall time (with concurrent job limits):")
    for concurrency, wall_time in resources['wall_times'].items():
        days = wall_time / 24
        print(f"    {concurrency:3d} concurrent jobs: {wall_time:7.1f} hours ({days:5.1f} days)")

    # Check disk space
    if not args.no_disk_check:
        print("\nChecking disk space...", end=" ")
        work_dir = param_file.parent.parent  # Go up to work directory
        ok, msg = check_disk_space(resources['total_storage_gb'], work_dir)
        if not ok:
            print("WARNING")
            warnings.append(msg)
            print(f"  {msg}")
        elif msg:
            print("OK (with warning)")
            warnings.append(msg)
            print(f"  {msg}")
        else:
            print("OK")

    # Print parameter summary if requested
    if args.summary:
        print_parameter_summary(df)

    # Print warnings
    if warnings:
        print("\n" + "=" * 70)
        print("WARNINGS:")
        print("=" * 70)
        for i, warning in enumerate(warnings, 1):
            print(f"{i}. {warning}")

    # Final verdict
    print("\n" + "=" * 70)
    if errors:
        print("VALIDATION FAILED")
        print("=" * 70)
        print(f"Found {len(errors)} error(s). Please fix these issues before submitting.")
        sys.exit(1)
    else:
        print("VALIDATION PASSED")
        print("=" * 70)
        if warnings:
            print(f"File is valid but has {len(warnings)} warning(s).")
            print("Review warnings above before submitting.")
        else:
            print("All checks passed! Ready to submit.")

        # Print submission command
        print("\nNext step - Submit SLURM array job:")
        print("=" * 70)
        num_jobs = len(df)
        print(f"sbatch --array=1-{num_jobs}%50 run_array_simulation.sh {param_file}")
        print("\nNote: Adjust %50 (max concurrent jobs) based on cluster policy")
        print("=" * 70)

        sys.exit(0)


if __name__ == '__main__':
    main()
