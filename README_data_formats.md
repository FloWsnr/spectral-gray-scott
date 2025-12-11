# Gray-Scott Data Formats

## Output Structure

Each simulation creates a subfolder in `snapshots/` with parameter-based naming:

```
snapshots/gs_{pattern}_F={F}_k={k}_{init_type}_{seed}/
├── data.h5           # HDF5 format (numerical arrays for Python/numpy)
└── metadata.json     # JSON format metadata
```

### Example
```
snapshots/gs_gliders_F=014_k=054_gaussians_1/
├── data.h5
└── metadata.json
```

## Data Format Details

### HDF5 File (data.h5)

The HDF5 file contains the following datasets:

- **`/u`**: u field snapshots, shape `(n, n, num_snapshots)`
- **`/v`**: v field snapshots, shape `(n, n, num_snapshots)`
- **`/x`**: x-coordinate meshgrid, shape `(n, n)`
- **`/y`**: y-coordinate meshgrid, shape `(n, n)`
- **`/time`**: time points for each snapshot, shape `(num_snapshots,)`

### JSON Metadata (metadata.json)

Contains all simulation parameters:
- `pattern`: Pattern type (e.g., "gliders", "bubbles")
- `F`: Feed rate
- `k`: Kill rate
- `delta_u`: Diffusion coefficient for u
- `delta_v`: Diffusion coefficient for v
- `initialization`: Init type ("gaussians" or "fourier")
- `random_seed`: Random seed used
- `domain`: Spatial domain
- `grid_size`: Grid resolution (n)
- `time_step`: Integration time step
- `snapshot_interval`: Time between snapshots
- `final_time`: Final simulation time
- `num_snapshots`: Total number of snapshots
- Plus initialization-specific parameters

## Usage in Python

### Load Data

```python
from load_data import load_simulation

# Load simulation data
data, metadata = load_simulation('snapshots/gs_gliders_F=014_k=054_gaussians_1')

# Access arrays
u_field = data['u']  # Shape: (128, 128, 1001)
v_field = data['v']  # Shape: (128, 128, 1001)
time = data['time']  # Shape: (1001,)

# Access specific snapshot
u_snapshot_500 = data['u'][:, :, 500]
```

### Visualize Data

```bash
# Plot time evolution (9 frames by default)
python visualize_data.py snapshots/gs_gliders_F=014_k=054_gaussians_1

# Plot specific snapshot
python visualize_data.py snapshots/gs_gliders_F=014_k=054_gaussians_1 --snapshot 500

# Save to file
python visualize_data.py snapshots/gs_gliders_F=014_k=054_gaussians_1 --save output.png

# Custom number of frames
python visualize_data.py snapshots/gs_gliders_F=014_k=054_gaussians_1 --frames 16
```

### Direct HDF5 Access

```python
import h5py
import numpy as np

with h5py.File('snapshots/gs_gliders_F=014_k=054_gaussians_1/data.h5', 'r') as f:
    u = np.array(f['u'])
    v = np.array(f['v'])
    time = np.array(f['time'])
```

### Direct JSON Access

```python
import json

with open('snapshots/gs_gliders_F=014_k=054_gaussians_1/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Pattern: {metadata['pattern']}")
print(f"F={metadata['F']}, k={metadata['k']}")
```

## Notes

- Only HDF5 and JSON formats are saved to minimize storage usage
- HDF5 files are readable in both Python (h5py) and MATLAB
- Grid coordinates are uniformly spaced from domain boundaries
- If you need MATLAB format, you can load HDF5 files in MATLAB using `h5read()`
