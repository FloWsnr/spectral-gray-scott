# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This codebase generates pattern formation datasets for the Gray-Scott reaction-diffusion equations using spectral methods. It's designed to run on HPC clusters with SLURM, producing HDF5 datasets suitable for machine learning applications.

## Key Architecture

### Dual-Language System

The codebase uses **MATLAB** for simulations (spectral methods via Chebfun) and **Python** for job orchestration and data loading:

- **MATLAB simulation core** (`simulation/gen_gs.m`): Uses Chebfun library for spectral methods, ETDRK4 time integration scheme
- **Python parameter sweeps** (`slurm_simulation/`): Generates parameter grids, monitors jobs, handles failures
- **HDF5 output**: Language-agnostic format (~85% smaller than .mat files) with embedded metadata

### Data Flow

1. Parameter grid generation (Python) → CSV file with all combinations
2. SLURM array job submission → Each task reads one CSV row
3. MATLAB simulation (`gen_gs.m`) → Writes HDF5 + JSON metadata
4. Job status tracking → Monitors completion/failures for resume capability

Output structure:
- `results/snapshots/F{F}_k{k}_du{delta_u}_dv{delta_v}_{init_type}/data.h5`

Directories are named based on simulation parameters only (F, k, delta_u, delta_v, initialization type), excluding random seeds. Multiple trajectories with different random seeds but identical parameters are stored together in a single HDF5 file with shape `[n_trajectories, n_time, x, y, channels]`.

## Development Commands

### Running Single Simulation

From MATLAB:
```matlab
addpath('simulation');
addpath('chebfun');
gen_gs(0.00002, 0.00001, 0.014, 0.054, 1, 'gaussians', 1, 10, 10000)
```

Parameters: `gen_gs(delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend)`

All optional time parameters (dt, snap_dt, tend) have defaults. Multiple random seeds can be passed as an array: `gen_gs(0.00002, 0.00001, 0.014, 0.054, [1,2,3], 'gaussians')`

### Parameter Sweep Workflow

All commands run from `slurm_simulation/` directory:

```bash
# 1. Generate parameter grid (creates CSV)
python generate_parameter_grid.py \
    --F 0.010:0.100:20 \
    --k 0.050:0.065:20 \
    --random-seed 1,2,3 \
    --output params.csv

# 2. Validate parameters
python validate_params.py params.csv

# 3. Submit array job (1200 jobs, max 50 concurrent)
sbatch --array=1-1200%50 run_array_simulation.sh params.csv

# 4. Monitor progress
python check_job_status.py --params params.csv

# 5. Resume failed jobs
bash resume_failed_jobs.sh params.csv
```

### Visualization

```bash
# View all snapshots
python visualize_data.py results/snapshots/F0.014_k0.054_du2.0e-05_dv1.0e-05_gaussians/

# View specific snapshot
python visualize_data.py results/snapshots/F0.014_k0.054_du2.0e-05_dv1.0e-05_gaussians/ --snapshot 500

# View specific trajectory (if multiple seeds)
python visualize_data.py results/snapshots/F0.014_k0.054_du2.0e-05_dv1.0e-05_gaussians/ --trajectory 0
```

## Critical Implementation Details

### Parameter Grid Syntax

Range formats in `generate_parameter_grid.py`:
- Linear: `start:stop:num` → `np.linspace(start, stop, num)`
- Logarithmic: `start:stop:num:log` → `np.logspace(log10(start), log10(stop), num)`
- Explicit: `val1,val2,val3` → `[val1, val2, val3]`

### SLURM Script Paths

The `run_array_simulation.sh` script expects:
- Parameter file as first argument
- Chebfun at `${SCRIPT_DIR}/chebfun` (auto-installs if missing)
- Creates directories: `results/snapshots/`, `logs/`, `results/job_status/`, `results/slurm_logs/`
- Sets `MATLAB_TMPDIR` to `${HPCWORK}/matlab_tmp` to avoid filling SSDs

### HDF5 Structure

Datasets in `data.h5`:
- `/uv`: shape `[n_trajectories, n_time, 128, 128, 2]` - Combined concentration fields with dimensions `[n_trajectories, n_time, x, y, channels]` where channel 0 is u and channel 1 is v (MATLAB permutes for correct Python reading)
- `/x` and `/y`: shape `[128]` - Spatial grid points on domain [-1,1]
- `/time`: shape `[num_snapshots]` - Time values for each snapshot
- `/random_seeds`: shape `[n_trajectories]` - Random seed for each trajectory

Grid: Uniform grid via `linspace(-1, 1, 128)` on domain [-1,1]×[-1,1]

Accessing data:
```python
# Load combined array
uv = data['uv']  # shape: [n_trajectories, n_time, x, y, 2]

# Extract u and v fields
u = uv[..., 0]  # channel 0 = u
v = uv[..., 1]  # channel 1 = v

# Access specific trajectory and time
u_snapshot = uv[trajectory_idx, time_idx, :, :, 0]
v_snapshot = uv[trajectory_idx, time_idx, :, :, 1]
```

Metadata stored as HDF5 attributes on root `/`: `F`, `k`, `delta_u`, `delta_v`, `n_trajectories`, `initialization`, `num_snapshots`, `grid_size_x`, `grid_size_y`, `domain`, `time_step`, `snapshot_interval`, `final_time`, `scheme`, `dealias`

### Initialization Types

Two initialization schemes (`init_type` parameter):
- `'gaussians'`: Random Gaussian perturbations via `random_gaussians.m` (ngauss=[10,100], amp=[1,3], width=[150,300])
- `'fourier'`: Fourier mode initialization via `init_fourier.m` (nfourier=32)

Both use `rng(random_seed)` for reproducibility.

### Common Parameter Values

Example F/k combinations that produce interesting dynamics:
- F=0.014, k=0.054 (gliding spots)
- F=0.012, k=0.050 (bubbles)
- F=0.029, k=0.057 (maze-like patterns)
- F=0.078, k=0.061 (worm-like patterns)
- F=0.010, k=0.041 (spiral waves)
- F=0.014, k=0.045 (stationary spots)

Note: Pattern formation depends on F, k, diffusion coefficients (delta_u, delta_v), and initialization type.

## Environment Setup

### MATLAB Environment

Required modules (HPC):
```bash
module load MATLAB/2025a
```

Chebfun is auto-installed by SLURM scripts, or manually from: https://www.chebfun.org

### Python Environment

Required for parameter sweeps only:
```bash
module load Python/3.12.3
python3 -m venv ~/venvs/gray-scott-env
source ~/venvs/gray-scott-env/bin/activate
pip install numpy pandas h5py  # h5py needed for load_data.py and visualize_data.py
```

## Troubleshooting

### Simulation Failures

If `gen_gs.m` fails with "Solution blew up":
- Time step `dt` may be too large (try reducing from default 1 to 0.5 or 0.1)
- Parameters F/k may be in unstable regime
- Initialization may be too extreme

Check MATLAB logs in `logs/` directory for detailed error messages.

### Array Job Issues

- **Wrong number of jobs**: Check `--array=1-N` matches CSV rows (use `validate_params.py` first)
- **Failed tasks**: Use `check_job_status.py --failed-ids-only` then `resume_failed_jobs.sh`
- **Path errors**: SLURM scripts use `SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)` - ensure scripts run from `slurm_simulation/` or paths resolve correctly

### Data Loading

Python `load_data.py` reads data from HDF5 `/uv` dataset with shape `[n_trajectories, n_time, x, y, channels]`. The script provides the combined `uv` array for direct access to both fields. For custom processing, read HDF5 directly with `h5py`:

```python
import h5py
import numpy as np

with h5py.File('data.h5', 'r') as f:
    uv = np.array(f['uv'])  # [n_trajectories, n_time, x, y, 2]
    u = uv[..., 0]  # Extract u field
    v = uv[..., 1]  # Extract v field
    x = np.array(f['x'])
    y = np.array(f['y'])
    time = np.array(f['time'])
    random_seeds = np.array(f['random_seeds'])
```

## File Locations

- Simulations: `simulation/gen_gs.m`, `simulation/init_fourier.m`, `simulation/random_gaussians.m`
- Parameter generation: `slurm_simulation/generate_parameter_grid.py`
- Job scripts: `slurm_simulation/run_array_simulation.sh`, `slurm_simulation/run_simulation.sh`
- Monitoring: `slurm_simulation/check_job_status.py`, `slurm_simulation/resume_failed_jobs.sh`
- Data loading: `simulation/load_data.py`, `visualize_data.py`
- Outputs: `results/snapshots/`, `logs/`, `results/job_status/`, `results/slurm_logs/`



### Rules

- dont account for backwards compatibility, except when explicitly asked for it