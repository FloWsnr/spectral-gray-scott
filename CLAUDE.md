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

Output structure: `snapshots/gs_{pattern}_F={F}_k={k}_{init}_{seed}/data.h5`

## Development Commands

### Running Single Simulation

From MATLAB:
```matlab
addpath('simulation');
addpath('chebfun');
gen_gs('gliders', 0.00002, 0.00001, 0.014, 0.054, 1, 'gaussians', 1, 10, 10000)
```

Parameters: `gen_gs(pattern, delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend)`

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
python visualize_data.py snapshots/gs_gliders_F=014_k=054_gaussians_1/

# View specific snapshot
python visualize_data.py snapshots/gs_gliders_F=014_k=054_gaussians_1/ --snapshot 500
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
- Creates directories: `snapshots/`, `logs/`, `results/job_status/`, `results/slurm_logs/`
- Sets `MATLAB_TMPDIR` to `${HPCWORK}/matlab_tmp` to avoid filling SSDs

### HDF5 Structure

Datasets in `data.h5`:
- `/u` and `/v`: shape `[128, 128, num_snapshots]` (note: MATLAB writes in column-major, Python reads in row-major)
- Grid: Chebyshev points via `chebpts(128, [-1, 1])` on domain [-1,1]×[-1,1]
- Time indexing: snapshot `k` corresponds to time `t = (k-1) × snap_dt`

Metadata stored as HDF5 attributes on root `/`: `F`, `k`, `pattern`, `delta_u`, `delta_v`, `random_seed`, `initialization`, `num_snapshots`, `grid_size_x`, `grid_size_y`, `domain`, `time_step`, `snapshot_interval`, `final_time`, `scheme`, `dealias`

### Initialization Types

Two initialization schemes (`init_type` parameter):
- `'gaussians'`: Random Gaussian perturbations via `random_gaussians.m` (ngauss=[10,100], amp=[1,3], width=[150,300])
- `'fourier'`: Fourier mode initialization via `init_fourier.m` (nfourier=32)

Both use `rng(random_seed)` for reproducibility.

### Pattern Parameters

Common F/k values for different patterns:
- Gliders: F=0.014, k=0.054
- Bubbles: F=0.012, k=0.050
- Maze: F=0.029, k=0.057
- Worms: F=0.078, k=0.061
- Spirals: F=0.010, k=0.041
- Spots: F=0.014, k=0.045

(These are typical values; actual pattern formation depends on initialization and full parameter set)

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

Python `load_data.py` reconstructs grid via `np.polynomial.chebyshev.chebpts(128)` to match MATLAB's Chebyshev grid. For custom processing, read HDF5 directly with `h5py`.

## File Locations

- Simulations: `simulation/gen_gs.m`, `simulation/init_fourier.m`, `simulation/random_gaussians.m`
- Parameter generation: `slurm_simulation/generate_parameter_grid.py`
- Job scripts: `slurm_simulation/run_array_simulation.sh`, `slurm_simulation/run_simulation.sh`
- Monitoring: `slurm_simulation/check_job_status.py`, `slurm_simulation/resume_failed_jobs.sh`
- Data loading: `simulation/load_data.py`, `visualize_data.py`
- Outputs: `snapshots/`, `logs/`, `results/job_status/`, `results/slurm_logs/`
