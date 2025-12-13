# spectral-gray-scott

Code for generating pattern formation datasets for the Gray-Scott equations using spectral methods.

## Table of Contents

- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Basic Usage](#basic-usage)
- [Output Format (HDF5)](#output-format-hdf5)
- [Reading Data](#reading-data)
- [SLURM Array Jobs](#slurm-array-jobs-for-parameter-sweeps)
- [File Size and Performance](#file-size-and-performance)

---

## Installation

### Prerequisites

**Required:**
- MATLAB: https://www.mathworks.com/products/matlab.html
- Chebfun: https://www.chebfun.org

**For parameter sweeps and analysis:**
- Python 3.12+ with numpy, h5py and pandas

---

## Directory Structure

```
spectral-gray-scott/
├── simulation/               # Core simulation code
│   ├── gen_gs.m             # Main Gray-Scott simulation function
│   ├── init_fourier.m       # Fourier initialization
│   ├── random_gaussians.m   # Gaussian initialization
│   └── load_data.py         # Python data loading utility
├── slurm_simulation/         # SLURM array job system
│   ├── generate_parameter_grid.py   # Generate parameter combinations
│   ├── validate_params.py           # Validate parameter files
│   ├── run_simulation.sh            # Single job SLURM script
│   ├── run_array_simulation.sh      # Array job SLURM script
│   ├── check_job_status.py          # Monitor job progress
│   └── resume_failed_jobs.sh        # Resubmit failed jobs
├── visualize_data.py         # Visualization script
├── logs/                     # MATLAB simulation logs
├── results/                  # SLURM job tracking and simulation output
│   ├── snapshots/           # Simulation output (HDF5 files)
│   ├── slurm_logs/          # SLURM output logs
│   └── job_status/          # Job status files
└── README.md                 # This file
```

---

## Basic Usage

### Running a Single Simulation

The `simulation/gen_gs.m` function runs a Gray-Scott simulation with specified parameters:

```matlab
% From MATLAB, add the simulation directory to your path
addpath('simulation');
addpath('chebfun');  % Ensure Chebfun is in your path

% Run simulation
gen_gs(pattern, delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend)
```

**Parameters:**
- `pattern` - Pattern type: 'gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots'
- `delta_u` - Diffusion coefficient for u (typically 0.00002)
- `delta_v` - Diffusion coefficient for v (typically 0.00001)
- `F` - Feed rate parameter
- `k` - Kill rate parameter
- `random_seed` - Random seed for reproducibility
- `init_type` - Initialization: 'gaussians' or 'fourier'
- `dt` - Time step size (optional, default: 1)
- `snap_dt` - Snapshot interval (optional, default: 10)
- `tend` - Final simulation time (optional, default: 10000)

**Example:**
```matlab
gen_gs('gliders', 0.00002, 0.00001, 0.014, 0.054, 1, 'gaussians', 1, 10, 10000)
```

### Using the SLURM Script

For running on HPC clusters, use the provided SLURM script:

```bash
# Edit slurm_simulation/run_simulation.sh to set your parameters, then:
cd slurm_simulation
sbatch run_simulation.sh
```

---

## Output Format (HDF5)

The simulation saves results directly to HDF5 format, providing:

- **Smaller file sizes** (~85% reduction compared to .mat files)
- **Language-agnostic format** (readable from Python, Julia, C++, etc.)
- **Embedded metadata** as HDF5 attributes
- **Efficient storage** of large datasets

### Output Files

Each simulation creates a subfolder: `results/snapshots/gs_{pattern}_F={F}_k={k}_{init}_{seed}/`

Files created:
- `data.h5` - HDF5 file with field values
- `metadata.json` - JSON metadata (for compatibility)

### HDF5 File Structure

**Datasets:**
- `/u` - U component values, shape: `[128, 128, num_snapshots]`
- `/v` - V component values, shape: `[128, 128, num_snapshots]`

**Attributes (Metadata):**

Stored as HDF5 attributes on the root group `/`:

- `pattern` - Pattern type (gliders, bubbles, etc.)
- `F` - Feed rate parameter
- `k` - Kill rate parameter
- `delta_u`, `delta_v` - Diffusion coefficients
- `initialization` - Init type (gaussians or fourier)
- `random_seed` - Random seed used
- `num_snapshots` - Number of time snapshots
- `grid_size_x`, `grid_size_y` - Grid dimensions (128×128)
- `domain` - Spatial domain [xmin, xmax, ymin, ymax]
- `time_step`, `snapshot_interval`, `final_time` - Time parameters
- `scheme` - Time integration scheme (etdrk4)
- `dealias` - Dealiasing setting

### Grid Points

The data is evaluated on a Chebyshev grid:
- Grid size: 128 × 128
- Domain: [-1, 1] × [-1, 1]
- Grid points generated via `chebpts(n, [xmin, xmax])`

### Time Snapshots

The third dimension of the datasets corresponds to time snapshots:
- Index 1: t = 0
- Index 2: t = snap_dt
- Index 3: t = 2×snap_dt
- Index k: t = (k-1)×snap_dt

---

## Reading Data

### MATLAB

```matlab
% Read the entire dataset
u_data = h5read('data.h5', '/u');
v_data = h5read('data.h5', '/v');

% Read metadata
F = h5readatt('data.h5', '/', 'F');
k = h5readatt('data.h5', '/', 'k');
```

### Python

Using h5py directly:

```python
import h5py
import numpy as np

with h5py.File('data.h5', 'r') as f:
    # Read datasets
    u = f['/u'][:]  # shape: (128, 128, num_snapshots)
    v = f['/v'][:]

    # Read metadata
    F = f.attrs['F']
    k = f.attrs['k']
    pattern = f.attrs['pattern']
```

Using the provided utility:

```python
import sys
sys.path.append('simulation')
from load_data import load_simulation

# Load simulation data
data, metadata = load_simulation('results/snapshots/gs_gliders_F=014_k=054_gaussians_1/')

# Access data
u, v, x, y, times = data['u'], data['v'], data['x'], data['y'], data['t']

# Access metadata
F = metadata['F']
k = metadata['k']
```


### Visualization

Use the provided Python script to visualize simulation results:

```bash
# Visualize all snapshots
python visualize_data.py results/snapshots/gs_gliders_F=014_k=054_gaussians_1/

# Visualize specific snapshot
python visualize_data.py results/snapshots/gs_gliders_F=014_k=054_gaussians_1/ --snapshot 500
```

---

## SLURM Array Jobs for Parameter Sweeps

For running large parameter sweeps (hundreds to thousands of simulations), use the SLURM array job system.

### Quick Start

**1. Generate Parameter Grid**

Create a CSV file with all parameter combinations:

```bash
cd slurm_simulation

python generate_parameter_grid.py \
    --F 0.010:0.100:20 \
    --k 0.050:0.065:20 \
    --random-seed 1,2,3 \
    --output params_Fk_sweep.csv
```

This creates 20×20×3 = 1200 parameter combinations.

**2. Validate Parameters**

```bash
python validate_params.py params_Fk_sweep.csv
```

**3. Submit Array Job**

```bash
sbatch --array=1-1200%50 run_array_simulation.sh params_Fk_sweep.csv
```

The `%50` limits to 50 concurrent jobs.

**4. Monitor Progress**

```bash
# Check SLURM queue
squeue -u $USER

# Check completion status
python check_job_status.py --params params_Fk_sweep.csv
```

**5. Handle Failed Jobs**

```bash
bash resume_failed_jobs.sh params_Fk_sweep.csv
```

### Parameter Grid Generation Details

**Example sweeps** (run from `slurm_simulation/` directory):

F×k grid with multiple random seeds:
```bash
python generate_parameter_grid.py \
    --F 0.010:0.100:20 \
    --k 0.050:0.065:20 \
    --random-seed 1,2,3,4,5 \
    --output params.csv
```

Different patterns:
```bash
python generate_parameter_grid.py \
    --pattern gliders,bubbles,maze \
    --F 0.01:0.1:10 \
    --k 0.05:0.065:10 \
    --output params.csv
```

Logarithmic spacing:
```bash
python generate_parameter_grid.py \
    --delta-u 0.00001:0.0001:10:log \
    --F 0.014 \
    --k 0.054 \
    --output params.csv
```

**For help with any script:**
```bash
python generate_parameter_grid.py --help
python validate_params.py --help
python check_job_status.py --help
```

---

## File Size and Performance

### File Size Comparison

Example (3 snapshots):
- `.mat` format: ~18 MB (chebfun objects)
- `.h5` format: ~0.8 MB (raw values)

For production runs with many snapshots, HDF5 provides significant storage savings.

### Converting Existing .mat Files

If you have existing `.mat` files, you can convert them to HDF5 format (conversion script would be in `simulation/` directory if available).

---

## Key Scripts Reference

### Core Simulation (`simulation/`)
- **`gen_gs.m`** - Main Gray-Scott simulation function
- **`init_fourier.m`** - Fourier-based initialization
- **`random_gaussians.m`** - Gaussian-based initialization
- **`load_data.py`** - Python utility to load simulation results

### SLURM Array Jobs (`slurm_simulation/`)
- **`generate_parameter_grid.py`** - Create parameter sweep configurations
- **`validate_params.py`** - Validate parameter files before submission
- **`run_simulation.sh`** - SLURM script for single simulations
- **`run_array_simulation.sh`** - SLURM script for array jobs
- **`check_job_status.py`** - Monitor job progress and failures
- **`resume_failed_jobs.sh`** - Resubmit failed jobs

### Visualization
- **`visualize_data.py`** - Visualize simulation results

---

## Support

For issues specific to:
- **Cluster/SLURM:** Contact your cluster support team
- **MATLAB/Chebfun:** Check respective documentation
- **Python scripts:** Run with `--help` flag for usage information
  ```bash
  cd slurm_simulation
  python generate_parameter_grid.py --help
  python validate_params.py --help
  python check_job_status.py --help
  ```
