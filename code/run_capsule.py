import os
from pathlib import Path
from typing import List

import dask.array as da
import numpy as np
import tifffile as tif
from natsort import natsorted
from skimage.transform import resize

from aind_smartspim_flatfield_estimation import flatfield_estimation, utils


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def main():

    SCALE = 2
    z_step_percentage = 0.1  # 10% of the z planes

    cpu_count = utils.get_code_ocean_cpu_limit()
    shading_parameters = {
        "get_darkfield": False,
        "smoothness_flatfield": 1.0,
        "smoothness_darkfield": 20,
        "sort_intensity": True,
        "max_reweight_iterations": 35,
        "resize_mode": "skimage_dask",
        "max_workers": int(cpu_count),
    }

    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    scratch_folder = Path(os.path.abspath("../scratch"))

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/metadata.json",
    ]
    """
    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")
    """
    metadata_json_path = data_folder.joinpath(
        "SmartSPIM_717381_2024-07-03_10-49-01/derivatives/metadata.json"
    )  # REMOVE
    channel_paths = list(
        data_folder.glob(
            "SmartSPIM_717381_2024-07-03_10-49-01-zarr-destriped-channels/Ex_*_Em_*"
        )
    )  # REMOVE

    laser_side = utils.get_col_rows_per_laser(metadata_json_path=metadata_json_path)

    print("Laser sides: ", laser_side)

    for i, channel_path in enumerate(channel_paths):
        channel_name = channel_path.stem

        print(f"Computing flats for channel: {channel_name}")
        if not channel_path.exists():
            raise FileNotFoundError(f"Path {channel_path} does not exist!")

        # Lazy reading just to check the shape
        check_zarr = list(channel_path.glob("*.zarr"))[0].joinpath(str(SCALE))
        lazy_data = da.from_zarr(check_zarr)

        picked_slices, indices = utils.pick_slices(
            lazy_data, percentage=z_step_percentage, read_lazy=False
        )

        print(f"Len indices: {len(indices)} {indices}")
        tiles_per_side = utils.get_slicer_per_side(
            tiles_per_laser=laser_side,
            channel_path=channel_path,
            indices=indices,
            scale=SCALE,
        )

        print(f"Laser sides: {tiles_per_side.keys()}")

        flats = flatfield_estimation.estimate_flats_per_laser(
            tiles_per_side=tiles_per_side, shading_params=shading_parameters
        )

        upsample_scale = SCALE * 2

        for side, flat_dict in flats.items():
            median_flatfield = flatfield_estimation.create_median_flatfield(
                flat_dict["flatfield"], smooth=True
            )
            upsample_shape = tuple(upsample_scale * np.array(median_flatfield.shape))

            print(f"Upsample shape: {upsample_shape}")

            upsampled_median_flatfield = resize(
                median_flatfield,
                upsample_shape,
                order=4,
                mode="reflect",
                cval=0,
                clip=True,
                preserve_range=False,
                anti_aliasing=None,
            )
            flat_name = str(
                results_folder.joinpath(
                    f"estimated_flat_laser_{channel_name}_side_{side}.tif"
                )
            )
            tif.imwrite(flat_name, upsampled_median_flatfield)


if __name__ == "__main__":
    main()
