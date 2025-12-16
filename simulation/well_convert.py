"""Convert simulation data into the well format

https://polymathic-ai.org/the_well/data_format/
"""

from typing import Optional
import json
from pathlib import Path
import h5py
import numpy as np


def create_hdf5_dataset(sim_dir: Path, output_dir: Optional[Path] = None) -> Path:
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
    u_data = org_data[..., 0].astype(np.float32)  # shape: [n_trajectories, n_time, x, y]
    v_data = org_data[..., 1].astype(np.float32)  # shape: [n_trajectories, n_time, x, y]

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

        # Load and store u and v fields
        u_dset = t0_fields.create_dataset("u", data=u_data)
        u_dset.attrs["dim_varying"] = [True, True]
        u_dset.attrs["sample_varying"] = True
        u_dset.attrs["time_varying"] = True

        v_dset = t0_fields.create_dataset("v", data=v_data)
        v_dset.attrs["dim_varying"] = [True, True]
        v_dset.attrs["sample_varying"] = True
        v_dset.attrs["time_varying"] = True

        # Create t1_fields group for velocities
        t1_fields = f.create_group("t1_fields")
        t1_fields.attrs["field_names"] = []

        # Create empty t2_fields group
        t2_fields = f.create_group("t2_fields")
        t2_fields.attrs["field_names"] = []

    print(f"Created {filename}")
    return filename


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
        default="../results/snapshots",
        help="Directory containing simulation snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/well_format",
        help="Output directory for converted files",
    )

    args = parser.parse_args()

    snapshots_dir = Path(args.snapshots_dir)
    output_dir = Path(args.output_dir)

    if not snapshots_dir.exists():
        print(f"Error: Snapshots directory {snapshots_dir} does not exist")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all simulation directories
    sim_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]

    print(f"Found {len(sim_dirs)} simulation directories")

    converted_count = 0
    failed_count = 0

    for sim_dir in sim_dirs:
        try:
            # Check if required files exist
            if not (sim_dir / "data.h5").exists():
                print(f"Skipping {sim_dir.name}: data.h5 not found")
                continue

            if not (sim_dir / "metadata.json").exists():
                print(f"Skipping {sim_dir.name}: metadata.json not found")
                continue

            print(f"\nProcessing {sim_dir.name}...")
            create_hdf5_dataset(sim_dir, output_dir)
            converted_count += 1

        except Exception as e:
            print(f"Error processing {sim_dir.name}: {e}")
            failed_count += 1
            continue

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Successfully converted: {converted_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
