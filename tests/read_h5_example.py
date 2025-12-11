#!/usr/bin/env python3
"""
Example script to read Gray-Scott simulation data from HDF5 files
"""

import h5py
import numpy as np

def read_gs_data(h5_filepath):
    """
    Read Gray-Scott simulation data from HDF5 file

    Parameters:
        h5_filepath: Path to the HDF5 file

    Returns:
        u_data: numpy array of shape (nx, ny, num_snapshots)
        v_data: numpy array of shape (nx, ny, num_snapshots)
        metadata: dictionary of metadata attributes
    """
    with h5py.File(h5_filepath, 'r') as f:
        # Read datasets
        u_data = f['/u'][:]
        v_data = f['/v'][:]

        # Read metadata attributes
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]

        return u_data, v_data, metadata


if __name__ == '__main__':
    # Example usage
    h5_file = 'snapshots/gs_gliders_F=014_k=054_gaussians_999/data.h5'

    print("Reading HDF5 file:", h5_file)
    print("=" * 50)

    u, v, meta = read_gs_data(h5_file)

    print(f"\nData shapes:")
    print(f"  u: {u.shape}")
    print(f"  v: {v.shape}")

    print(f"\nMetadata:")
    for key, value in sorted(meta.items()):
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {list(value)}")
        else:
            print(f"  {key}: {value}")

    print(f"\nData statistics (first snapshot):")
    print(f"  u - min: {u[:,:,0].min():.6f}, max: {u[:,:,0].max():.6f}, mean: {u[:,:,0].mean():.6f}")
    print(f"  v - min: {v[:,:,0].min():.6f}, max: {v[:,:,0].max():.6f}, mean: {v[:,:,0].mean():.6f}")

    print("=" * 50)
    print("Successfully read HDF5 file!")
