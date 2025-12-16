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
        # Load data in format: [n_trajectories, n_time, x, y, channels]
        if '/uv' not in f:
            raise ValueError("No /uv dataset found in HDF5 file. Only new format is supported.")

        data['uv'] = np.array(f['uv'])
        print(f"Loaded data with shape: [n_trajectories, n_time, x, y, channels]")
        print(f"  Shape: {data['uv'].shape}")

        data['x'] = np.array(f['x'])
        data['y'] = np.array(f['y'])
        data['time'] = np.array(f['time'])

        # Load random_seeds
        if '/random_seeds' in f:
            data['random_seeds'] = np.array(f['random_seeds'])

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

    # Handle both old and new metadata formats
    if 'random_seeds' in metadata:
        seeds = metadata['random_seeds']
        if isinstance(seeds, list):
            print(f"  Random Seeds:   {len(seeds)} seeds (min={min(seeds)}, max={max(seeds)})")
        else:
            print(f"  Random Seed:    {seeds}")
    elif 'random_seed' in metadata:
        print(f"  Random Seed:    {metadata['random_seed']}")

    if 'n_trajectories' in metadata:
        print(f"  Trajectories:   {metadata['n_trajectories']}")

    print(f"  Grid Size:      {metadata['grid_size']} x {metadata['grid_size']}")
    print(f"  Num Snapshots:  {metadata['num_snapshots']}")
    print(f"  Final Time:     {metadata['final_time']}")
    print(f"  Snapshot dt:    {metadata['snapshot_interval']}")

    print("\nData Arrays:")
    print(f"  uv shape:       {data['uv'].shape}")
    print(f"  x shape:        {data['x'].shape}")
    print(f"  y shape:        {data['y'].shape}")
    print(f"  time shape:     {data['time'].shape}")

    if 'random_seeds' in data:
        print(f"  random_seeds:   {data['random_seeds'].shape} ({len(data['random_seeds'])} seeds)")

    print("\nData Ranges:")
    print(f"  u (channel 0):  [{data['uv'][..., 0].min():.4f}, {data['uv'][..., 0].max():.4f}]")
    print(f"  v (channel 1):  [{data['uv'][..., 1].min():.4f}, {data['uv'][..., 1].max():.4f}]")
    print(f"  time:           [{data['time'].min():.1f}, {data['time'].max():.1f}]")

    print("\nIndexing:")
    print("  data['uv'][trajectory_idx, time_idx, x, y, channel]")
    print("  Channel 0 = u, Channel 1 = v")
    print(f"  Example: uv[0, 500, :, :, :] = trajectory 0 at time t={data['time'][500] if len(data['time']) > 500 else 'N/A'}")

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
        trajectory_idx = 0
        snapshot_idx = 0
        uv_snapshot = data['uv'][trajectory_idx, snapshot_idx, :, :, :]
        u_snapshot = data['uv'][trajectory_idx, snapshot_idx, :, :, 0]
        v_snapshot = data['uv'][trajectory_idx, snapshot_idx, :, :, 1]
        print(f"\nExample: Trajectory {trajectory_idx}, Snapshot {snapshot_idx}")
        print(f"  Time: t={data['time'][snapshot_idx]}")
        print(f"  uv_snapshot shape: {uv_snapshot.shape}")
        print(f"  u_snapshot shape: {u_snapshot.shape}")
        print(f"  v_snapshot shape: {v_snapshot.shape}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
