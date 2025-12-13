#!/usr/bin/env python3
"""
Load Gray-Scott simulation data from HDF5 and JSON formats.

Example usage:
    # New directory structure (array jobs)
    python load_data.py results/snapshots/12345/1

    # Old directory structure (legacy)
    python load_data.py results/snapshots/gs_F=014_k=054_gaussians_1
"""

import h5py
import json
import numpy as np
import sys
from pathlib import Path


def load_simulation(folder_path):
    """
    Load simulation data and metadata from a folder.

    Parameters:
    -----------
    folder_path : str or Path
        Path to the simulation folder

    Returns:
    --------
    data : dict
        Dictionary containing 'u', 'v', 'x', 'y', 'time' arrays
    metadata : dict
        Dictionary containing simulation parameters
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Load HDF5 data
    h5_file = folder / 'data.h5'
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

    data = {}
    with h5py.File(h5_file, 'r') as f:
        data['u'] = np.array(f['u'])  # Shape: (n, n, num_snapshots)
        data['v'] = np.array(f['v'])  # Shape: (n, n, num_snapshots)
        data['x'] = np.array(f['x'])  # Shape: (n, n)
        data['y'] = np.array(f['y'])  # Shape: (n, n)
        data['time'] = np.array(f['time'])  # Shape: (num_snapshots,)

    # Load JSON metadata
    json_file = folder / 'metadata.json'
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, 'r') as f:
        metadata = json.load(f)

    return data, metadata


def print_info(data, metadata):
    """Print information about the loaded simulation."""
    print("=" * 60)
    print("Simulation Data Loaded")
    print("=" * 60)
    print("\nMetadata:")
    print(f"  F:              {metadata['F']}")
    print(f"  k:              {metadata['k']}")
    print(f"  Delta U:        {metadata['delta_u']}")
    print(f"  Delta V:        {metadata['delta_v']}")
    print(f"  Initialization: {metadata['initialization']}")
    print(f"  Random Seed:    {metadata['random_seed']}")
    print(f"  Grid Size:      {metadata['grid_size']} x {metadata['grid_size']}")
    print(f"  Num Snapshots:  {metadata['num_snapshots']}")
    print(f"  Final Time:     {metadata['final_time']}")
    print(f"  Snapshot dt:    {metadata['snapshot_interval']}")

    print("\nData Arrays:")
    print(f"  u shape:        {data['u'].shape}")
    print(f"  v shape:        {data['v'].shape}")
    print(f"  x shape:        {data['x'].shape}")
    print(f"  y shape:        {data['y'].shape}")
    print(f"  time shape:     {data['time'].shape}")

    print("\nData Ranges:")
    print(f"  u:   [{data['u'].min():.4f}, {data['u'].max():.4f}]")
    print(f"  v:   [{data['v'].min():.4f}, {data['v'].max():.4f}]")
    print(f"  time: [{data['time'].min():.1f}, {data['time'].max():.1f}]")
    print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python load_data.py <folder_path>")
        print("\nExample:")
        print("  python load_data.py results/snapshots/gs_F=014_k=054_gaussians_1")
        sys.exit(1)

    folder_path = sys.argv[1]

    try:
        data, metadata = load_simulation(folder_path)
        print_info(data, metadata)

        # Example: Access specific snapshot
        snapshot_idx = 0
        u_snapshot = data['u'][:, :, snapshot_idx]
        v_snapshot = data['v'][:, :, snapshot_idx]
        print(f"\nExample: Snapshot {snapshot_idx} at time t={data['time'][snapshot_idx]}")
        print(f"  u_snapshot shape: {u_snapshot.shape}")
        print(f"  v_snapshot shape: {v_snapshot.shape}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
