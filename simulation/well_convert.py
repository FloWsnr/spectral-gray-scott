"""Convert simulation data into the well format

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


def create_hdf5_dataset(sim_dir: Path, output_dir: Optional[Path] = None, verbose: bool = False) -> Path:
    """
    Create HDF5 file with the specified format.
    """
    with open(sim_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    with h5py.File(sim_dir / "data.h5", "r") as f:
        org_data = np.array(f["uv"])  # shape: [n_trajectories, n_time, x, y, 2]
        x_coords = np.array(f["x"]).flatten()
        y_coords = np.array(f["y"]).flatten()
        time_steps = np.array(f["time"]).flatten()

    if output_dir is None:
        output_dir = sim_dir.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / (
        sim_dir.name + ".hdf5"
    )  # name of the dir (all the parameters)
    sim_params = ["F", "k", "delta_u", "delta_v"]

    # Extract u and v fields from the combined uv array
    u_data = org_data[..., 0].astype(
        np.float32
    )  # shape: [n_trajectories, n_time, x, y]
    v_data = org_data[..., 1].astype(
        np.float32
    )  # shape: [n_trajectories, n_time, x, y]

    with h5py.File(filename, "w") as f:
        # Root attributes
        f.attrs["simulation_parameters"] = sim_params
        # add all metadata as attributes
        for key, val in metadata.items():
            f.attrs[key] = val
        f.attrs["dataset_name"] = "gray-scott"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 2
        f.attrs["n_trajectories"] = metadata["n_trajectories"]

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
            data = metadata[scalar_name]
            dset = scalars.create_dataset(scalar_name, data=np.array(data))
            dset.attrs["sample_varying"] = False
            dset.attrs["time_varying"] = False

        # Create t0_fields group for pressure
        t0_fields = f.create_group("t0_fields")
        t0_fields.attrs["field_names"] = ["u", "v"]

        # Load and store u and v fields with compression and optimal chunking
        # Chunk shape: (1 trajectory, 5 timesteps, full spatial grid)
        # Optimized for random access of 1-5 consecutive timesteps per trajectory
        chunk_shape = (1, min(5, u_data.shape[1]), u_data.shape[2], u_data.shape[3])

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
        print(f"Created {filename}")
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


def process_single_directory(
    sim_dir: Path, output_dir: Path
) -> tuple[str, bool, Optional[str]]:
    """
    Process a single simulation directory.

    Returns:
        tuple: (dir_name, converted_success, error_message)
    """
    dir_name = sim_dir.name

    try:
        # Check if required files exist
        if not (sim_dir / "data.h5").exists():
            return (dir_name, False, "data.h5 not found")

        if not (sim_dir / "metadata.json").exists():
            return (dir_name, False, "metadata.json not found")

        # Check if output file already exists
        output_file = output_dir / (sim_dir.name + ".hdf5")
        if output_file.exists():
            return (dir_name, False, "output already exists")

        create_hdf5_dataset(sim_dir, output_dir, verbose=False)

        return (dir_name, True, None)

    except Exception as e:
        error_msg = str(e)
        return (dir_name, False, error_msg)


def main():
    """
    Convert all simulation data in the snapshots directory to the well format.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Gray-Scott simulation data to the well format"
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="./results/snapshots",
        help="Directory containing simulation snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/well_format",
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )

    args = parser.parse_args()

    snapshots_dir = Path(args.snapshots_dir)
    output_dir = Path(args.output_dir)
    num_workers = args.workers if args.workers else mp.cpu_count()

    if not snapshots_dir.exists():
        print(f"Error: Snapshots directory {snapshots_dir} does not exist")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all simulation directories
    sim_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]

    total_dirs = len(sim_dirs)
    print(f"Found {total_dirs} simulation directories")
    print(f"Using {num_workers} worker processes\n")

    converted_count = 0
    failed_count = 0
    skipped_count = 0
    failed_dirs = []

    # Process directories in parallel with progress bar
    with mp.Pool(processes=num_workers) as pool:
        # Create arguments for each directory
        args_list = [(sim_dir, output_dir) for sim_dir in sim_dirs]

        # Use starmap with tqdm for progress tracking
        results = list(
            tqdm(
                pool.starmap(process_single_directory, args_list),
                total=total_dirs,
                desc="Converting datasets",
                unit="dir"
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
