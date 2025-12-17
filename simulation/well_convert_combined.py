"""Convert and combine simulation data from two directories into the well format

This script combines HDF5 files from two snapshot directories (with identical parameters
but different random seeds) into a single well format file with 2x the trajectories.

https://polymathic-ai.org/the_well/data_format/
"""

from typing import Optional
import json
from pathlib import Path
import h5py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from the_well.data.datasets import WellDataset


def combine_and_create_hdf5_dataset(
    sim_dir1: Path,
    sim_dir2: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Path:
    """
    Combine two HDF5 files with different random seeds and create a single well format file.

    Args:
        sim_dir1: First simulation directory
        sim_dir2: Second simulation directory (with matching parameters)
        output_dir: Output directory for the combined file
        verbose: Print progress messages

    Returns:
        Path to the created HDF5 file
    """
    # Read metadata from first directory
    with open(sim_dir1 / "metadata.json", "r") as f:
        metadata1 = json.load(f)

    # Read metadata from second directory
    with open(sim_dir2 / "metadata.json", "r") as f:
        metadata2 = json.load(f)

    # Load data from both files
    with h5py.File(sim_dir1 / "data.h5", "r") as f:
        uv1 = np.array(f["uv"])  # shape: [n_trajectories, n_time, x, y, 2]
        x_coords = np.array(f["x"]).flatten()
        y_coords = np.array(f["y"]).flatten()
        time_steps = np.array(f["time"]).flatten()
        seeds1 = np.array(f["random_seeds"]).flatten()

    with h5py.File(sim_dir2 / "data.h5", "r") as f:
        uv2 = np.array(f["uv"])  # shape: [n_trajectories, n_time, x, y, 2]
        seeds2 = np.array(f["random_seeds"]).flatten()

    # Verify that the spatial and temporal dimensions match
    assert uv1.shape[1:] == uv2.shape[1:], "Time and spatial dimensions must match"
    assert np.allclose(
        x_coords, np.array(h5py.File(sim_dir2 / "data.h5", "r")["x"]).flatten()
    ), "x coordinates must match"
    assert np.allclose(
        y_coords, np.array(h5py.File(sim_dir2 / "data.h5", "r")["y"]).flatten()
    ), "y coordinates must match"
    assert np.allclose(
        time_steps, np.array(h5py.File(sim_dir2 / "data.h5", "r")["time"]).flatten()
    ), "time steps must match"

    # Check for duplicate seeds and remove them from the second set
    duplicate_mask = np.isin(seeds2, seeds1)
    n_duplicates = duplicate_mask.sum()

    if n_duplicates > 0:
        duplicate_seeds = seeds2[duplicate_mask]

        if n_duplicates == len(seeds2):
            # All trajectories are duplicates
            warning_msg = (
                f"WARNING: All {n_duplicates} trajectories in second dataset are duplicates!\n"
                f"  Duplicate seeds: {duplicate_seeds}\n"
                f"  No new trajectories will be added from second dataset."
            )
            print(warning_msg)
        elif verbose:
            print(f"  Found {n_duplicates} duplicate seed(s): {duplicate_seeds}")
            print(f"  Removing duplicate trajectories from second dataset")

        # Keep only non-duplicate trajectories from uv2
        keep_mask = ~duplicate_mask
        uv2 = uv2[keep_mask]
        seeds2 = seeds2[keep_mask]

    # Combine the trajectories along the first axis
    combined_uv = np.concatenate(
        [uv1, uv2], axis=0
    )  # shape: [n_traj1 + n_traj2, n_time, x, y, 2]

    # Combine random seeds if available
    combined_seeds = np.concatenate([seeds1, seeds2], axis=0)

    # Update metadata with combined trajectory count
    combined_metadata = metadata1.copy()
    combined_metadata["n_trajectories"] = uv1.shape[0] + uv2.shape[0]

    if output_dir is None:
        output_dir = sim_dir1.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / (sim_dir1.name + ".hdf5")
    sim_params = ["F", "k", "delta_u", "delta_v"]

    # Extract u and v fields from the combined uv array
    u_data = combined_uv[..., 0].astype(
        np.float32
    )  # shape: [n_trajectories, n_time, x, y]
    v_data = combined_uv[..., 1].astype(
        np.float32
    )  # shape: [n_trajectories, n_time, x, y]

    with h5py.File(filename, "w") as f:
        # Root attributes
        f.attrs["simulation_parameters"] = sim_params
        # add all metadata as attributes
        for key, val in combined_metadata.items():
            f.attrs[key] = val
        f.attrs["dataset_name"] = "gray-scott"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 2
        f.attrs["n_trajectories"] = combined_metadata["n_trajectories"]
        f.attrs["random_seeds"] = combined_seeds

        # Create dimensions group
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]

        time_dset = dims.create_dataset("time", data=time_steps)
        time_dset.attrs["sample_varying"] = False

        x_dset = dims.create_dataset("x", data=x_coords)
        x_dset.attrs["sample_varying"] = False
        x_dset.attrs["time_varying"] = False

        y_dset = dims.create_dataset("y", data=y_coords)
        y_dset.attrs["sample_varying"] = False
        y_dset.attrs["time_varying"] = False

        # Create boundary conditions group
        bc = f.create_group("boundary_conditions")

        x_bc = bc.create_group("x_periodic")
        x_bc.attrs["associated_dims"] = ["x"]
        x_bc.attrs["associated_fields"] = []
        x_bc.attrs["bc_type"] = "periodic"
        x_bc.attrs["sample_varying"] = False
        x_bc.attrs["time_varying"] = False

        mask = np.zeros_like(x_coords, dtype=bool)
        mask[0] = True
        mask[-1] = True
        x_bc.create_dataset("mask", data=mask)
        x_bc.create_dataset("values", data=np.zeros_like(x_coords))

        # y-boundary
        y_bc = bc.create_group("y_periodic")
        y_bc.attrs["associated_dims"] = ["y"]
        y_bc.attrs["associated_fields"] = []
        y_bc.attrs["bc_type"] = "periodic"
        y_bc.attrs["sample_varying"] = False
        y_bc.attrs["time_varying"] = False
        mask = np.zeros_like(y_coords, dtype=bool)
        mask[0] = True
        mask[-1] = True
        y_bc.create_dataset("mask", data=mask)
        y_bc.create_dataset("values", data=np.zeros_like(y_coords))

        # Create scalars group
        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = sim_params

        for scalar_name in sim_params:
            data = combined_metadata[scalar_name]
            dset = scalars.create_dataset(scalar_name, data=np.array(data))
            dset.attrs["sample_varying"] = False
            dset.attrs["time_varying"] = False

        # Create t0_fields group for pressure
        t0_fields = f.create_group("t0_fields")
        t0_fields.attrs["field_names"] = ["u", "v"]

        # Load and store u and v fields with compression and optimal chunking
        # Chunk shape: (1 trajectory, 1 timestep, full spatial grid)
        chunk_shape = (1, 1, u_data.shape[2], u_data.shape[3])

        u_dset = t0_fields.create_dataset(
            "u", data=u_data, compression="gzip", compression_opts=9, chunks=chunk_shape
        )
        u_dset.attrs["dim_varying"] = [True, True]
        u_dset.attrs["sample_varying"] = True
        u_dset.attrs["time_varying"] = True

        v_dset = t0_fields.create_dataset(
            "v", data=v_data, compression="gzip", compression_opts=9, chunks=chunk_shape
        )
        v_dset.attrs["dim_varying"] = [True, True]
        v_dset.attrs["sample_varying"] = True
        v_dset.attrs["time_varying"] = True

        # Create t1_fields group for velocities
        t1_fields = f.create_group("t1_fields")
        t1_fields.attrs["field_names"] = []

        # Create empty t2_fields group
        t2_fields = f.create_group("t2_fields")
        t2_fields.attrs["field_names"] = []

    if verbose:
        print(
            f"Created {filename} with {combined_metadata['n_trajectories']} trajectories"
        )
    return filename


def verify_well_dataset(filename: Path) -> bool:
    """
    Verify the created HDF5 file by loading it with WellDataset and drawing the first sample.

    Returns:
        True if verification succeeds, False otherwise
    """
    try:
        # Create WellDataset from the file
        dataset = WellDataset(str(filename))

        # Draw the first sample
        sample = dataset[0]

        # Print verification info
        print("  Verification successful!")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Sample keys: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"  Verification FAILED: {e}")
        return False


def process_single_directory_pair(
    sim_dir1: Path, sim_dir2: Path, output_dir: Path
) -> tuple[str, bool, Optional[str]]:
    """
    Process a pair of simulation directories with matching parameters.

    Returns:
        tuple: (dir_name, converted_success, error_message)
    """
    dir_name = sim_dir1.name

    try:
        # Check if required files exist in both directories
        if not (sim_dir1 / "data.h5").exists():
            return (dir_name, False, "data.h5 not found in first directory")

        if not (sim_dir1 / "metadata.json").exists():
            return (dir_name, False, "metadata.json not found in first directory")

        if not (sim_dir2 / "data.h5").exists():
            return (dir_name, False, "data.h5 not found in second directory")

        if not (sim_dir2 / "metadata.json").exists():
            return (dir_name, False, "metadata.json not found in second directory")

        # Check if output file already exists
        output_file = output_dir / (sim_dir1.name + ".hdf5")
        if output_file.exists():
            return (dir_name, False, "output already exists")

        combine_and_create_hdf5_dataset(sim_dir1, sim_dir2, output_dir, verbose=False)

        return (dir_name, True, None)

    except Exception as e:
        error_msg = str(e)
        return (dir_name, False, error_msg)


def _process_wrapper(args: tuple) -> tuple[str, bool, Optional[str]]:
    """Wrapper for multiprocessing - unpacks tuple arguments."""
    return process_single_directory_pair(*args)


def main():
    """
    Convert and combine simulation data from two snapshot directories to the well format.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine Gray-Scott simulation data from two directories into well format"
    )
    parser.add_argument(
        "--snapshots-dir1",
        type=str,
        default="./results/snapshots1",
        help="First directory containing simulation snapshots",
    )
    parser.add_argument(
        "--snapshots-dir2",
        type=str,
        default="./results/snapshots2",
        help="Second directory containing simulation snapshots with different random seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/well_format_combined",
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )

    args = parser.parse_args()

    snapshots_dir1 = Path(args.snapshots_dir1)
    snapshots_dir2 = Path(args.snapshots_dir2)
    output_dir = Path(args.output_dir)
    num_workers = args.workers if args.workers else mp.cpu_count()

    if not snapshots_dir1.exists():
        print(f"Error: First snapshots directory {snapshots_dir1} does not exist")
        return

    if not snapshots_dir2.exists():
        print(f"Error: Second snapshots directory {snapshots_dir2} does not exist")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all simulation directories in first snapshot directory
    sim_dirs1 = {d.name: d for d in snapshots_dir1.iterdir() if d.is_dir()}
    sim_dirs2 = {d.name: d for d in snapshots_dir2.iterdir() if d.is_dir()}

    # Find matching directories (same parameter combinations)
    matching_dirs = set(sim_dirs1.keys()) & set(sim_dirs2.keys())

    if not matching_dirs:
        print(
            "Error: No matching directories found between the two snapshot directories"
        )
        return

    total_dirs = len(matching_dirs)
    print(f"Found {len(sim_dirs1)} directories in {snapshots_dir1}")
    print(f"Found {len(sim_dirs2)} directories in {snapshots_dir2}")
    print(f"Found {total_dirs} matching parameter combinations")
    print(f"Using {num_workers} worker processes\n")

    converted_count = 0
    failed_count = 0
    skipped_count = 0
    failed_dirs = []

    # Process directory pairs in parallel with progress bar
    with mp.Pool(processes=num_workers) as pool:
        # Create arguments for each directory pair
        args_list = [
            (sim_dirs1[name], sim_dirs2[name], output_dir) for name in matching_dirs
        ]

        # Use imap_unordered for real-time progress updates
        results = list(
            tqdm(
                pool.imap_unordered(_process_wrapper, args_list, chunksize=1),
                total=total_dirs,
                desc="Converting datasets",
                unit="dir",
            )
        )

    # Aggregate results
    for dir_name, converted, error_msg in results:
        if error_msg and ("not found" in error_msg or "already exists" in error_msg):
            skipped_count += 1
            continue

        if converted:
            converted_count += 1
        else:
            failed_count += 1
            failed_dirs.append((dir_name, error_msg))

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Successfully converted: {converted_count}")
    print(f"Skipped (missing files): {skipped_count}")
    print(f"Failed: {failed_count}")

    if failed_dirs:
        print("\nFailed directories:")
        for dir_name, error_msg in failed_dirs:
            print(f"  - {dir_name}: {error_msg}")

    print(f"\nOutput directory: {output_dir}")

    # Verify the entire dataset
    if converted_count > 0:
        print("\n" + "=" * 60)
        print("Verifying WellDataset...")
        if verify_well_dataset(output_dir):
            print("Dataset verification successful!")
        else:
            print("Dataset verification failed!")


if __name__ == "__main__":
    main()
