#!/usr/bin/env python3
"""
Visualize Gray-Scott simulation data.

Example usage:
    # New directory structure (array jobs)
    python visualize_data.py results/snapshots/12345/1
    python visualize_data.py results/snapshots/12345/1 --snapshot 500

    # Old directory structure (legacy)
    python visualize_data.py results/snapshots/gs_F=014_k=054_gaussians_1
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse


def load_simulation(folder_path):
    """Load simulation data and metadata."""
    folder = Path(folder_path)

    # Load HDF5 data
    data = {}
    with h5py.File(folder / "data.h5", "r") as f:
        # Load data in format: [n_trajectories, n_time, x, y, channels]
        if '/uv' not in f:
            raise ValueError("No /uv dataset found in HDF5 file. Only new format is supported.")

        data['uv'] = np.array(f['uv'])
        data["x"] = np.array(f["x"])
        data["y"] = np.array(f["y"])
        data["time"] = np.array(f["time"])

        # Load random_seeds
        if '/random_seeds' in f:
            data['random_seeds'] = np.array(f['random_seeds'])

    # Load JSON metadata
    with open(folder / "metadata.json", "r") as f:
        metadata = json.load(f)

    return data, metadata


def plot_snapshot(data, metadata, snapshot_idx, trajectory_idx=0, save_path=None):
    """Plot u and v fields for a specific snapshot and trajectory."""

    # Extract u and v from channels
    u = data["uv"][trajectory_idx, snapshot_idx, :, :, 0]
    v = data["uv"][trajectory_idx, snapshot_idx, :, :, 1]
    t = data["time"][snapshot_idx]
    title_suffix = f" (trajectory {trajectory_idx})"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot u
    im1 = axes[0].imshow(u.T, extent=metadata["domain"], origin="lower", cmap="viridis")
    axes[0].set_title(f"u field at t={t:.1f}{title_suffix}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0])

    # Plot v
    im2 = axes[1].imshow(v.T, extent=metadata["domain"], origin="lower", cmap="plasma")
    axes[1].set_title(f"v field at t={t:.1f}{title_suffix}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1])

    fig.suptitle(
        f"Gray-Scott Simulation (F={metadata['F']}, k={metadata['k']})", fontsize=14
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_time_evolution(data, metadata, trajectory_idx=0, num_frames=9, save_path=None):
    """Plot evolution of v field over time for a single trajectory."""

    num_snapshots = data["uv"].shape[1]  # Second dimension (n_time)
    title_suffix = f" (trajectory {trajectory_idx})"

    indices = np.linspace(0, num_snapshots - 1, num_frames, dtype=int)

    rows = int(np.ceil(np.sqrt(num_frames)))
    cols = int(np.ceil(num_frames / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if num_frames > 1 else [axes]

    for i, idx in enumerate(indices):
        # Extract v field (channel 1)
        v = data["uv"][trajectory_idx, idx, :, :, 1]
        t = data["time"][idx]

        axes[i].imshow(v.T, extent=metadata["domain"], origin="lower", cmap="plasma")
        axes[i].set_title(f"t={t:.0f}")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Gray-Scott Evolution{title_suffix} (F={metadata['F']}, k={metadata['k']})",
        fontsize=14
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Gray-Scott simulation data")
    parser.add_argument("folder", type=str, help="Path to simulation folder")
    parser.add_argument(
        "--snapshot",
        type=int,
        default=None,
        help="Specific snapshot index to plot (default: plot evolution)",
    )
    parser.add_argument(
        "--trajectory",
        type=int,
        default=0,
        help="Trajectory index to visualize (default: 0)",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save figure to file instead of showing"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=9,
        help="Number of frames for evolution plot (default: 9)",
    )

    args = parser.parse_args()

    try:
        data, metadata = load_simulation(args.folder)

        # Validate trajectory index
        n_trajectories = data['uv'].shape[0]
        if args.trajectory < 0 or args.trajectory >= n_trajectories:
            print(f"Error: Trajectory index must be between 0 and {n_trajectories - 1}")
            sys.exit(1)

        if args.snapshot is not None:
            # Validate snapshot index
            max_snapshots = data["uv"].shape[1]
            if args.snapshot < 0 or args.snapshot >= max_snapshots:
                print(f"Error: Snapshot index must be between 0 and {max_snapshots - 1}")
                sys.exit(1)
            plot_snapshot(data, metadata, args.snapshot, args.trajectory, args.save)
        else:
            plot_time_evolution(data, metadata, args.trajectory, args.frames, args.save)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
