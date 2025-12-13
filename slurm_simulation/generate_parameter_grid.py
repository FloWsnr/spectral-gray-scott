#!/usr/bin/env python3
"""
Generate parameter combinations for Gray-Scott SLURM array jobs.

This script creates a CSV file containing all parameter combinations for
running multiple Gray-Scott simulations via SLURM array jobs.

Usage Examples:
    # F×k grid sweep (20×20 = 400 combinations)
    python generate_parameter_grid.py \\
        --F 0.01:0.1:20 --k 0.05:0.065:20 -o params.csv

    # Logarithmic spacing for F
    python generate_parameter_grid.py \\
        --F 0.001:0.1:20:log --k 0.05:0.065:10 -o params.csv

    # Multiple random seeds
    python generate_parameter_grid.py \\
        --F 0.01:0.1:10 --k 0.05:0.065:10 \\
        --random-seed 1,2,3,4,5 -o params.csv

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
    - "start:stop:num" -> linspace(start, stop, num)
    - "start:stop:num:log" -> logspace(log10(start), log10(stop), num)
    - "val1,val2,val3" -> array([val1, val2, val3])

    Args:
        range_str: String specifying the range

    Returns:
        numpy array of values
    """
    # Check if it's a comma-separated list
    if ',' in range_str and ':' not in range_str:
        values = [float(x.strip()) for x in range_str.split(',')]
        return np.array(values)

    # Check if it's a range specification
    if ':' in range_str:
        parts = range_str.split(':')
        if len(parts) < 3:
            raise ValueError(f"Range must be start:stop:num or start:stop:num:log, got: {range_str}")

        start = float(parts[0])
        stop = float(parts[1])
        num = int(parts[2])

        if len(parts) == 4 and parts[3] == 'log':
            # Logarithmic spacing
            if start <= 0 or stop <= 0:
                raise ValueError(f"Logarithmic spacing requires positive values, got: start={start}, stop={stop}")
            return np.logspace(np.log10(start), np.log10(stop), num)
        else:
            # Linear spacing
            return np.linspace(start, stop, num)

    # Single value
    return np.array([float(range_str)])


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

    # Create Cartesian product
    combinations = list(itertools.product(*param_values))

    # Build DataFrame
    data = []
    for i, combo in enumerate(combinations):
        row = {'job_id': i + 1}
        row.update(fixed_params)
        row.update(dict(zip(param_names, combo)))
        data.append(row)

    # Define column order
    column_order = ['job_id', 'pattern', 'delta_u', 'delta_v', 'F', 'k',
                    'random_seed', 'init_type', 'dt', 'snap_dt', 'tend']

    df = pd.DataFrame(data)
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
    required_cols = ['pattern', 'delta_u', 'delta_v', 'F', 'k', 'random_seed',
                     'init_type', 'dt', 'snap_dt', 'tend']
    missing = [col for col in required_cols if col not in df.columns or df[col].isnull().any()]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Check valid pattern names
    valid_patterns = ['gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots']
    invalid_patterns = df[~df['pattern'].isin(valid_patterns)]['pattern'].unique()
    if len(invalid_patterns) > 0:
        raise ValueError(f"Invalid pattern names: {invalid_patterns}. Valid: {valid_patterns}")

    # Check valid init types
    valid_init_types = ['gaussians', 'fourier']
    invalid_init = df[~df['init_type'].isin(valid_init_types)]['init_type'].unique()
    if len(invalid_init) > 0:
        raise ValueError(f"Invalid init_type: {invalid_init}. Valid: {valid_init_types}")

    # Check positive parameters
    for param in ['delta_u', 'delta_v', 'F', 'k', 'dt', 'snap_dt', 'tend']:
        if (df[param] <= 0).any():
            raise ValueError(f"Parameter {param} must be positive")

    # Check random_seed is integer
    if not all(df['random_seed'].apply(lambda x: float(x).is_integer())):
        raise ValueError("random_seed must be integer values")

    # Check for duplicates
    param_cols = [col for col in df.columns if col != 'job_id']
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
    print("\n" + "="*70)
    print("Parameter Grid Preview")
    print("="*70)
    print(f"Total combinations: {len(df)}")
    print(f"\nParameter ranges:")

    for col in df.columns:
        if col == 'job_id':
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

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate parameter grid for Gray-Scott simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # F×k grid with 20×20 combinations
  %(prog)s --F 0.01:0.1:20 --k 0.05:0.065:20 -o params.csv

  # Logarithmic spacing
  %(prog)s --F 0.001:0.1:20:log --k 0.05:0.065:10 -o params.csv

  # Multiple random seeds for statistics
  %(prog)s --F 0.014:0.026:5 --k 0.051:0.057:5 --random-seed 1,2,3,4,5 -o params.csv

  # Custom values (comma-separated)
  %(prog)s --F 0.014,0.018,0.022 --k 0.051,0.054,0.057 -o params.csv

Range format:
  start:stop:num       - Linear spacing (e.g., 0.01:0.1:10)
  start:stop:num:log   - Logarithmic spacing (e.g., 0.001:0.1:10:log)
  val1,val2,val3       - Custom list of values
        """
    )

    # Parameters that can vary
    parser.add_argument('--pattern', type=str, default='gliders',
                        help='Pattern type (default: gliders). Can be range or comma-separated list.')
    parser.add_argument('--delta-u', type=str, default='0.00002',
                        help='Diffusion coefficient for u (default: 0.00002)')
    parser.add_argument('--delta-v', type=str, default='0.00001',
                        help='Diffusion coefficient for v (default: 0.00001)')
    parser.add_argument('--F', type=str, required=True,
                        help='Feed rate parameter (required, format: start:stop:num or val1,val2,...)')
    parser.add_argument('--k', type=str, required=True,
                        help='Kill rate parameter (required, format: start:stop:num or val1,val2,...)')
    parser.add_argument('--random-seed', type=str, default='1',
                        help='Random seed(s) (default: 1, format: val1,val2,... or start:stop:num)')
    parser.add_argument('--init-type', type=str, default='gaussians',
                        help='Initialization type: gaussians or fourier (default: gaussians)')
    parser.add_argument('--dt', type=str, default='1',
                        help='Time step size (default: 1)')
    parser.add_argument('--snap-dt', type=str, default='10',
                        help='Snapshot interval (default: 10)')
    parser.add_argument('--tend', type=str, default='10000',
                        help='Final simulation time (default: 10000)')

    # Output options
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--preview', type=int, default=10,
                        help='Number of preview rows to display (default: 10, 0 to disable)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip parameter validation')

    args = parser.parse_args()

    try:
        # Parse all parameters
        print("Parsing parameter ranges...")

        # Identify which parameters are swept vs fixed
        sweep_params = {}
        fixed_params = {}

        param_specs = {
            'pattern': args.pattern,
            'delta_u': args.delta_u,
            'delta_v': args.delta_v,
            'F': args.F,
            'k': args.k,
            'random_seed': args.random_seed,
            'init_type': args.init_type,
            'dt': args.dt,
            'snap_dt': args.snap_dt,
            'tend': args.tend
        }

        for param_name, param_spec in param_specs.items():
            # Special handling for string parameters
            if param_name in ['pattern', 'init_type']:
                if ',' in param_spec:
                    sweep_params[param_name] = [x.strip() for x in param_spec.split(',')]
                else:
                    fixed_params[param_name] = param_spec
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
        print("\n" + "="*70)
        print("Next steps:")
        print("="*70)
        print(f"1. Validate parameters:")
        print(f"   python validate_params.py {output_path}")
        print(f"\n2. Submit SLURM array job:")
        print(f"   sbatch --array=1-{len(df)}%50 run_array_simulation.sh {output_path}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
