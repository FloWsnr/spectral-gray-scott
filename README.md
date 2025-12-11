# spectral-gray-scott
Code for generating a pattern formation dataset for the Gray-Scott equations

## Installation

Make sure to install matlab and chebfun first.

- Matlab: https://www.mathworks.com/products/matlab.html
- Chebfun: https://www.chebfun.org



# Gray-Scott Simulation - HDF5 Output

## Overview

The `gen_gs.m` script has been updated to save simulation results directly to HDF5 format instead of MATLAB `.mat` files. This provides:

- **Smaller file sizes** (~85% reduction compared to .mat)
- **Language-agnostic format** (readable from Python, Julia, C++, etc.)
- **Embedded metadata** as HDF5 attributes
- **Efficient storage** of large datasets

## Usage

### Running a Simulation

```matlab
gen_gs(pattern, delta_u, delta_v, F, k, random_seed, init_type, dt, snap_dt, tend)
```

**Example:**
```matlab
gen_gs('gliders', 0.00002, 0.00001, 0.014, 0.054, 1, 'gaussians', 1, 10, 10000)
```

### Output Files

Each simulation creates a subfolder: `snapshots/gs_{pattern}_F={F}_k={k}_{init}_{seed}/`

Files created:
- `data.h5` - HDF5 file with field values
- `metadata.json` - JSON metadata (for compatibility)

## HDF5 File Structure

### Datasets

- `/u` - U component values, shape: `[128, 128, num_snapshots]`
- `/v` - V component values, shape: `[128, 128, num_snapshots]`

### Attributes (Metadata)

Stored as HDF5 attributes on the root group `/`:

- `pattern` - Pattern type (gliders, bubbles, etc.)
- `F` - Feed rate parameter
- `k` - Kill rate parameter
- `delta_u` - Diffusion coefficient for u
- `delta_v` - Diffusion coefficient for v
- `initialization` - Init type (gaussians or fourier)
- `random_seed` - Random seed used
- `num_snapshots` - Number of time snapshots
- `grid_size_x`, `grid_size_y` - Grid dimensions
- `domain` - Spatial domain [xmin, xmax, ymin, ymax]
- `time_step` - Time step dt
- `snapshot_interval` - Snapshot interval
- `final_time` - Final simulation time
- `scheme` - Time integration scheme
- `dealias` - Dealiasing setting

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

See `read_h5_example.py` for a complete example.

### Julia

```julia
using HDF5

h5open("data.h5", "r") do file
    u = read(file, "/u")
    v = read(file, "/v")

    F = read_attribute(file, "F")
    k = read_attribute(file, "k")
end
```

## File Size Comparison

Example from test run (3 snapshots):
- `.mat` format: ~18 MB (chebfun objects)
- `.h5` format: ~0.8 MB (raw values)

For production runs with many snapshots, the HDF5 format provides significant storage savings.

## Converting Existing .mat Files

If you have existing `.mat` files from previous runs, use `save_gs_to_h5.m` to convert them:

```matlab
save_gs_to_h5('gliders', 'gaussians', 1)
```

This will create a `data.h5` file alongside the existing `data.mat` file.

## Grid Points

The data is evaluated on a Chebyshev grid:
- Grid size: 128 x 128
- Domain: [-1, 1] x [-1, 1]
- Grid points generated via `chebpts(n, [xmin, xmax])`

## Time Snapshots

The third dimension of the datasets corresponds to time snapshots:
- Index 1: t = 0
- Index 2: t = snap_dt
- Index 3: t = 2*snap_dt
- ...
- Index k: t = (k-1)*snap_dt
