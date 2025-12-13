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
        data["u"] = np.array(f["u"])
        data["v"] = np.array(f["v"])
        data["x"] = np.array(f["x"])
        data["y"] = np.array(f["y"])
        data["time"] = np.array(f["time"])

    # Load JSON metadata
    with open(folder / "metadata.json", "r") as f:
        metadata = json.load(f)

    return data, metadata


def plot_snapshot(data, metadata, snapshot_idx, save_path=None):
    """Plot u and v fields for a specific snapshot."""
    u = data["u"][:, :, snapshot_idx]
    v = data["v"][:, :, snapshot_idx]
    t = data["time"][snapshot_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot u
    im1 = axes[0].imshow(u, extent=metadata["domain"], origin="lower", cmap="viridis")
    axes[0].set_title(f"u field at t={t:.1f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0])

    # Plot v
    im2 = axes[1].imshow(v, extent=metadata["domain"], origin="lower", cmap="plasma")
    axes[1].set_title(f"v field at t={t:.1f}")
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


def plot_time_evolution(data, metadata, num_frames=9, save_path=None):
    """Plot evolution of v field over time."""
    num_snapshots = data["v"].shape[2]
    indices = np.linspace(0, num_snapshots - 1, num_frames, dtype=int)

    rows = int(np.ceil(np.sqrt(num_frames)))
    cols = int(np.ceil(num_frames / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if num_frames > 1 else [axes]

    for i, idx in enumerate(indices):
        v = data["v"][:, :, idx]
        t = data["time"][idx][0]
        print(t)

        axes[i].imshow(v, extent=metadata["domain"], origin="lower", cmap="plasma")
        axes[i].set_title(f"t={t:.0f}")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Gray-Scott Evolution (F={metadata['F']}, k={metadata['k']})", fontsize=14
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

        if args.snapshot is not None:
            if args.snapshot < 0 or args.snapshot >= data["v"].shape[2]:
                print(
                    f"Error: Snapshot index must be between 0 and {data['v'].shape[2] - 1}"
                )
                sys.exit(1)
            plot_snapshot(data, metadata, args.snapshot, args.save)
        else:
            plot_time_evolution(data, metadata, args.frames, args.save)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
