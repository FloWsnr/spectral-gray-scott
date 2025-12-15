#!/usr/bin/env python3
"""
Generate parameter combinations for Gray-Scott SLURM array jobs.

This script creates a CSV file containing all parameter combinations for
running multiple Gray-Scott simulations via SLURM array jobs.

Usage Examples:
    # F×k grid sweep using step sizes
    python generate_parameter_grid.py \\
        --F 0.01:0.1:0.005 --k 0.05:0.065:0.001 -o params.csv

    # Logarithmic spacing for F (uses number of points)
    python generate_parameter_grid.py \\
        --F 0.001:0.1:20:log --k 0.05:0.065:0.001 -o params.csv

    # Multiple random seeds - same seeds for all parameter combinations
    python generate_parameter_grid.py \\
        --F 0.01:0.1:0.01 --k 0.05:0.065:0.0015 \\
        --random-seeds 1:5:1 -o params.csv

    # Generate N unique random seeds per parameter combination
    python generate_parameter_grid.py \\
        --F 0.01:0.1:0.01 --k 0.05:0.065:0.0015 \\
        --random-seeds 5 -o params.csv

    # Sweep diffusion coefficients (delta_u × delta_v)
    python generate_parameter_grid.py \\
        --F 0.014 --k 0.054 \\
        --delta-u 0.00001:0.00005:0.00001 \\
        --delta-v 0.000005:0.00002:0.000005 -o params.csv

    # Multiple initialization types (gaussians and fourier)
    python generate_parameter_grid.py \\
        --F 0.01:0.1:0.01 --k 0.05:0.065:0.0015 \\
        --init-type gaussians,fourier -o params.csv

    # Custom parameter values (comma-separated)
    python generate_parameter_grid.py \\
        --F 0.014,0.018,0.022,0.026 \\
        --k 0.051,0.054,0.057 -o params.csv
"""

import argparse
import sys
import itertools
from pathlib import Path
import numpy as np
import pandas as pd


def parse_range(range_str):
    """
    Parse a range specification into a numpy array.

    Formats supported:
    - "start:stop:step" -> arange(start, stop+step, step) with endpoint included
    - "start:stop:num:log" -> logspace(log10(start), log10(stop), num)
    - "val1,val2,val3" -> array([val1, val2, val3])

    Args:
        range_str: String specifying the range

    Returns:
        numpy array of values
    """
    # Check if it's a comma-separated list
    if "," in range_str and ":" not in range_str:
        values = [float(x.strip()) for x in range_str.split(",")]
        return np.array(values)

    # Check if it's a range specification
    if ":" in range_str:
        parts = range_str.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Range must be start:stop:step or start:stop:num:log, got: {range_str}"
            )

        start = float(parts[0])
        stop = float(parts[1])
        third_param = float(parts[2])

        if len(parts) == 4 and parts[3] == "log":
            # Logarithmic spacing (uses number of points)
            num = int(parts[2])
            if start <= 0 or stop <= 0:
                raise ValueError(
                    f"Logarithmic spacing requires positive values, got: start={start}, stop={stop}"
                )
            return np.logspace(np.log10(start), np.log10(stop), num)
        elif len(parts) == 3:
            # Step size specification (default behavior)
            step = third_param
            if step <= 0:
                raise ValueError(f"Step size must be positive, got: {step}")
            if (stop - start) / step < 0:
                raise ValueError(
                    f"Step direction inconsistent with start/stop: start={start}, stop={stop}, step={step}"
                )
            # Use arange and include endpoint
            return np.arange(
                start, stop + step / 2, step
            )  # Add step/2 to ensure endpoint is included
        else:
            raise ValueError(
                f"Invalid range format. Use start:stop:step or start:stop:num:log"
            )

    # Single value
    return np.array([float(range_str)])


def create_baseline_combinations(fixed_params, sweep_params):
    """
    Create baseline parameter combinations that are always included.

    These are well-known F/k combinations that produce interesting patterns.
    Baseline combinations are expanded across sweep parameters (e.g., random_seed, init_type).

    Args:
        fixed_params: Dictionary of fixed parameter values
        sweep_params: Dictionary of {param_name: array_of_values} for swept parameters

    Returns:
        pandas DataFrame with baseline parameter combinations
    """
    baseline_fk_pairs = [
        (0.037, 0.060),
        (0.030, 0.062),
        (0.025, 0.060),
        (0.078, 0.061),
        (0.039, 0.058),
        (0.026, 0.051),
        (0.034, 0.056),
        (0.014, 0.054),
        (0.018, 0.051),
        (0.014, 0.045),
        (0.062, 0.061),
    ]

    baseline_delta_u = 0.00002
    baseline_delta_v = 0.00001

    # Start with fixed baseline parameters
    baseline_fixed = {
        'delta_u': baseline_delta_u,
        'delta_v': baseline_delta_v,
    }

    # Add other fixed parameters that aren't F or k
    for param, value in fixed_params.items():
        if param not in ['F', 'k', 'delta_u', 'delta_v']:
            baseline_fixed[param] = value

    # Create sweep parameters for baseline (excluding F, k, delta_u, delta_v)
    baseline_sweep = {}
    for param, values in sweep_params.items():
        if param not in ['F', 'k', 'delta_u', 'delta_v']:
            baseline_sweep[param] = values

    # Generate combinations: each F/k pair with Cartesian product of other sweep params
    all_rows = []

    if baseline_sweep:
        # If there are other sweep parameters, do Cartesian product for each F/k pair
        param_names = list(baseline_sweep.keys())
        param_values = list(baseline_sweep.values())
        other_combinations = list(itertools.product(*param_values))

        for F, k in baseline_fk_pairs:
            for combo in other_combinations:
                row = {'F': F, 'k': k}
                row.update(dict(zip(param_names, combo)))
                row.update(baseline_fixed)
                all_rows.append(row)
    else:
        # No other sweep parameters, just use the F/k pairs
        for F, k in baseline_fk_pairs:
            row = {'F': F, 'k': k}
            row.update(baseline_fixed)
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def generate_grid(fixed_params, sweep_params):
    """
    Generate Cartesian product of all sweep parameters.

    Args:
        fixed_params: Dictionary of parameters that don't vary
        sweep_params: Dictionary of {param_name: array_of_values}

    Returns:
        pandas DataFrame with all parameter combinations
    """
    # Get parameter names and values
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    # Create Cartesian product and build DataFrame directly
    print("  Creating Cartesian product of parameters...")
    combinations = list(itertools.product(*param_values))

    # Build DataFrame from combinations without loop
    print("  Building parameter DataFrame...")
    df = pd.DataFrame(combinations, columns=param_names)

    # Add fixed parameters as columns (vectorized operation)
    for param, value in fixed_params.items():
        df[param] = value

    # Add job_id column (1-indexed)
    df.insert(0, "job_id", range(1, len(df) + 1))

    # Define column order
    column_order = [
        "job_id",
        "delta_u",
        "delta_v",
        "F",
        "k",
        "random_seeds",  # Changed from random_seed (plural)
        "init_type",
        "dt",
        "snap_dt",
        "tend",
    ]

    # Ensure all columns exist (add missing ones with None)
    for col in column_order:
        if col not in df.columns:
            df[col] = None

    return df[column_order]


def validate_parameters(df):
    """
    Validate parameter combinations for physical constraints.

    Args:
        df: DataFrame with parameter combinations

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check for required columns
    required_cols = [
        "delta_u",
        "delta_v",
        "F",
        "k",
        "random_seeds",  # Changed from random_seed
        "init_type",
        "dt",
        "snap_dt",
        "tend",
    ]
    missing = [
        col for col in required_cols if col not in df.columns or df[col].isnull().any()
    ]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Check valid init types
    valid_init_types = ["gaussians", "fourier"]
    invalid_init = df[~df["init_type"].isin(valid_init_types)]["init_type"].unique()
    if len(invalid_init) > 0:
        raise ValueError(
            f"Invalid init_type: {invalid_init}. Valid: {valid_init_types}"
        )

    # Check non-negative parameters (can be zero)
    for param in ["delta_u", "delta_v", "F", "k"]:
        if (df[param] < 0).any():
            raise ValueError(f"Parameter {param} must be non-negative")

    # Check strictly positive parameters (cannot be zero)
    for param in ["dt", "snap_dt", "tend"]:
        if (df[param] <= 0).any():
            raise ValueError(f"Parameter {param} must be positive")

    # NEW: Validate random_seeds format
    for idx, seed_str in df["random_seeds"].items():
        try:
            seeds = [int(s.strip()) for s in str(seed_str).split(",")]

            if len(seeds) == 0:
                raise ValueError(f"Row {idx+1}: Empty seed list")

            if any(s < 0 for s in seeds):
                raise ValueError(f"Row {idx+1}: Seeds must be non-negative")

        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Row {idx+1}: Invalid random_seeds format '{seed_str}' - must be comma-separated integers")
            else:
                raise

    # Check for duplicates
    param_cols = [col for col in df.columns if col != "job_id"]
    duplicates = df[param_cols].duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: Found {duplicates} duplicate parameter combinations")

    return True


def preview_grid(df, num_samples=10):
    """
    Print a preview of the parameter grid.

    Args:
        df: DataFrame with parameter combinations
        num_samples: Number of sample rows to display
    """
    print("\n" + "=" * 70)
    print("Parameter Grid Preview")
    print("=" * 70)
    print(f"Total combinations: {len(df)}")
    print("\nParameter ranges:")

    for col in df.columns:
        if col == "job_id":
            continue
        unique_vals = df[col].unique()
        if len(unique_vals) == 1:
            print(f"  {col:15s}: {unique_vals[0]} (fixed)")
        else:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  {col:15s}: {min_val} to {max_val} ({len(unique_vals)} values)")

    print(f"\nFirst {min(num_samples, len(df))} rows:")
    print(df.head(num_samples).to_string(index=False))

    if len(df) > num_samples:
        print(f"\n... ({len(df) - num_samples} more rows)")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter grid for Gray-Scott simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # F×k grid using step sizes
  %(prog)s --F 0.01:0.1:0.005 --k 0.05:0.065:0.001 -o params.csv

  # Logarithmic spacing (uses number of points)
  %(prog)s --F 0.001:0.1:20:log --k 0.05:0.065:0.001 -o params.csv

  # Multiple random seeds - same seeds for all parameter combinations
  %(prog)s --F 0.014:0.026:0.004 --k 0.051:0.057:0.002 --random-seeds 1:5:1 -o params.csv

  # Generate N unique random seeds per parameter combination
  %(prog)s --F 0.014:0.026:0.004 --k 0.051:0.057:0.002 --random-seeds 5 -o params.csv

  # Sweep diffusion coefficients
  %(prog)s --F 0.014 --k 0.054 --delta-u 0.00001:0.00005:0.00001 --delta-v 0.000005:0.00002:0.000005 -o params.csv

  # Multiple initialization types
  %(prog)s --F 0.01:0.1:0.01 --k 0.05:0.065:0.0015 --init-type gaussians,fourier -o params.csv

  # Custom values (comma-separated)
  %(prog)s --F 0.014,0.018,0.022 --k 0.051,0.054,0.057 -o params.csv

Range format:
  start:stop:step      - Linear spacing with step size (e.g., 0.01:0.1:0.005)
  start:stop:num:log   - Logarithmic spacing with num points (e.g., 0.001:0.1:20:log)
  val1,val2,val3       - Custom list of values (numeric or string params)
        """,
    )

    # Parameters that can vary
    parser.add_argument(
        "--delta-u",
        type=str,
        default="0.00002",
        help="Diffusion coefficient for u (default: 0.00002, format: start:stop:step or val1,val2,...)",
    )
    parser.add_argument(
        "--delta-v",
        type=str,
        default="0.00001",
        help="Diffusion coefficient for v (default: 0.00001, format: start:stop:step or val1,val2,...)",
    )
    parser.add_argument(
        "--F",
        type=str,
        required=True,
        help="Feed rate parameter (required, format: start:stop:step or val1,val2,...)",
    )
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        help="Kill rate parameter (required, format: start:stop:step or val1,val2,...)",
    )
    parser.add_argument(
        "--random-seeds",  # Changed to plural
        type=str,
        default="1",
        help="Random seed(s). Single number N: generate N unique seeds per job. Range/list: same seeds for all jobs (format: start:stop:step or val1,val2,...)",
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="gaussians",
        help="Initialization type: gaussians or fourier (default: gaussians, format: gaussians,fourier for both)",
    )
    parser.add_argument(
        "--dt", type=str, default="1", help="Time step size (default: 1)"
    )
    parser.add_argument(
        "--snap-dt", type=str, default="10", help="Snapshot interval (default: 10)"
    )
    parser.add_argument(
        "--tend",
        type=str,
        default="10000",
        help="Final simulation time (default: 10000)",
    )

    # Output options
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of preview rows to display (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip parameter validation"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip adding baseline F/k combinations (11 well-known parameter sets)",
    )

    args = parser.parse_args()

    try:
        # Parse all parameters
        print("Parsing parameter ranges...")

        # Identify which parameters are swept vs fixed
        sweep_params = {}
        fixed_params = {}

        param_specs = {
            "delta_u": args.delta_u,
            "delta_v": args.delta_v,
            "F": args.F,
            "k": args.k,
            "random_seeds": args.random_seeds,  # Changed to plural
            "init_type": args.init_type,
            "dt": args.dt,
            "snap_dt": args.snap_dt,
            "tend": args.tend,
        }

        # Special flag for per-job random seed generation
        num_seeds_per_job = None

        for param_name, param_spec in param_specs.items():
            # Special handling for string parameters
            if param_name in ["init_type"]:
                if "," in param_spec:
                    sweep_params[param_name] = [
                        x.strip() for x in param_spec.split(",")
                    ]
                else:
                    fixed_params[param_name] = param_spec
            elif param_name == "random_seeds":
                # Check if it's a single number (not a range, not a comma-separated list)
                if ":" not in param_spec and "," not in param_spec:
                    # Single number: generate N unique seeds per job
                    try:
                        num_seeds_per_job = int(param_spec)
                        if num_seeds_per_job <= 0:
                            raise ValueError("Number of seeds per job must be positive")
                        print(f"  Will generate {num_seeds_per_job} unique random seeds per parameter combination")
                        # Don't add to fixed_params yet, will generate after grid creation
                    except ValueError:
                        raise ValueError(f"Invalid random-seeds specification: '{param_spec}' - must be a single integer, range, or comma-separated list")
                else:
                    # Range or list: parse and use same seeds for all jobs
                    seed_values = parse_range(param_spec)
                    # Convert to integers and format as comma-separated string
                    seed_list = [str(int(s)) for s in seed_values]
                    fixed_params[param_name] = ",".join(seed_list)
            else:
                values = parse_range(param_spec)
                if len(values) > 1:
                    sweep_params[param_name] = values
                else:
                    fixed_params[param_name] = values[0]

        print(f"  Fixed parameters: {list(fixed_params.keys())}")
        print(f"  Sweep parameters: {list(sweep_params.keys())}")

        # Generate grid
        print("\nGenerating parameter grid...")
        df = generate_grid(fixed_params, sweep_params)
        print(f"  Generated {len(df)} parameter combinations")

        # Generate unique random seeds per job if requested
        if num_seeds_per_job is not None:
            print(f"\nGenerating {num_seeds_per_job} unique random seeds for each of {len(df)} parameter combinations...")
            # Use numpy's random generator for reproducibility
            rng = np.random.default_rng(seed=42)

            # Generate unique seeds for each row
            # Use a large range to avoid collisions
            max_seed = 2**31 - 1  # Max value for typical RNG seeds
            all_seeds = []

            for i in range(len(df)):
                # Generate N unique seeds for this job
                job_seeds = rng.integers(0, max_seed, size=num_seeds_per_job)
                seed_str = ",".join(str(s) for s in job_seeds)
                all_seeds.append(seed_str)

            df["random_seeds"] = all_seeds
            print(f"  Generated unique seed sets for {len(df)} jobs")

        # Add baseline combinations if requested
        if not args.no_baseline:
            print("\nAdding baseline parameter combinations...")
            baseline_df = create_baseline_combinations(fixed_params, sweep_params)
            print(f"  Created {len(baseline_df)} baseline combinations")

            # Generate unique seeds for baseline if needed
            if num_seeds_per_job is not None:
                print(f"  Generating {num_seeds_per_job} unique random seeds for baseline combinations...")
                rng = np.random.default_rng(seed=43)  # Different seed from main grid
                baseline_seeds = []
                for i in range(len(baseline_df)):
                    job_seeds = rng.integers(0, 2**31 - 1, size=num_seeds_per_job)
                    seed_str = ",".join(str(s) for s in job_seeds)
                    baseline_seeds.append(seed_str)
                baseline_df["random_seeds"] = baseline_seeds

            # Concatenate baseline with generated grid
            df_combined = pd.concat([df, baseline_df], ignore_index=True)

            # Remove duplicates based on all parameter columns (excluding job_id)
            param_cols = [col for col in df_combined.columns if col != 'job_id']
            df_before_dedup = len(df_combined)
            df_combined = df_combined.drop_duplicates(subset=param_cols, keep='first')
            df_after_dedup = len(df_combined)

            if df_before_dedup > df_after_dedup:
                print(f"  Removed {df_before_dedup - df_after_dedup} duplicate combinations")

            # Re-index job_id
            df_combined['job_id'] = range(1, len(df_combined) + 1)
            df = df_combined
            print(f"  Total combinations after adding baseline: {len(df)}")

        # Validate
        if not args.no_validate:
            print("\nValidating parameters...")
            validate_parameters(df)
            print("  All parameters valid")

        # Preview
        if args.preview > 0:
            preview_grid(df, args.preview)

        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"  Success! Wrote {len(df)} rows to {output_path}")

        # Print next steps
        print("\n" + "=" * 70)
        print("Next steps:")
        print("=" * 70)
        print("1. Validate parameters:")
        print(f"   python validate_params.py {output_path}")
        print("\n2. Submit SLURM array job:")
        print(f"   sbatch --array=1-{len(df)}%50 run_array_simulation.sh {output_path}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
