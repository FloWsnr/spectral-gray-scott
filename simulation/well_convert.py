"""Convert simulation data into the well format

https://polymathic-ai.org/the_well/data_format/
"""

import json
from pathlib import Path
import h5py
import numpy as np

def create_hdf5_dataset(sim_dir: Path):
    """
    Create HDF5 file with the specified format.
    """
    metadata = json.load(sim_dir / "metadata.json")
    with h5py.File(sim_dir / "data.h5", "r") as f:
        org_data = np.array(f['uv'])

    filename = Path(sim_dir.name + ".hdf5") # name of the dir (all the parameters)
    sim_params = ["F", "k", "delta_u", "delta_v"]


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

        # Load data from first file to get dimensions
        x_coords = data_dict["x_coords"]
        y_coords = data_dict["y_coords"]
        time_steps = data_dict["time_steps"]

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

        mask = np.zeros_like(x_coords, dtype=np.bool)
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
        mask = np.zeros_like(y_coords, dtype=np.bool)
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

        # Load and store pressure field
        u_dset = t0_fields.create_dataset(
            "u", data=data_dict["u"]
        )
        u_dset.attrs["dim_varying"] = [True, True]
        u_dset.attrs["sample_varying"] = True
        u_dset.attrs["time_varying"] = True

        v_dset = t0_fields.create_dataset(
            "v", data=data_dict["v"]
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