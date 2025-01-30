import json
import os
import time
from pathlib import Path
from typing import List

import dask.array as da
import numpy as np
import tifffile as tif
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_smartspim_flatfield_estimation import flatfield_estimation, utils
from aind_smartspim_flatfield_estimation.__init__ import (__maintainers__,
                                                          __pipeline_version__,
                                                          __url__, __version__)
from natsort import natsorted
from skimage.transform import resize


def save_dict_as_json(filename: str, dictionary: dict) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


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


def compute_unified_flatfield(fields, shading_correction_per_slide, mode="median"):
    """
    Gets the flatfields, darkfields and baselines to unify them
    based on the provided mode.
    """
    flatfields = []
    darkfields = []
    baselines = []

    # Unifying fields with median
    for slide_idx, fields in shading_correction_per_slide.items():
        flatfields.append(fields["flatfield"])
        darkfields.append(fields["darkfield"])
        baselines.append(fields["baseline"])

    print(f"Unifying fields using {mode} mode.")
    flatfield, darkfield, baseline = flatfield_estimation.unify_fields(
        flatfields, darkfields, baselines, mode=mode
    )
    return flatfield, darkfield, baseline


def main():
    """
    Main function
    """
    SCALE = 2
    z_step_percentage = 0.3  # % of the z planes

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
    # scratch_folder = Path(os.path.abspath("../scratch"))

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/metadata.json",
        f"{data_folder}/data_description.json",
    ]

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    data_description_path = data_folder.joinpath("data_description.json")
    data_description = {}

    if data_description_path.exists():
        data_description = utils.read_json_as_dict(filepath=data_description_path)

    print(f"Dataset name: {data_description.get('name')}")

    metadata_folder = results_folder.joinpath("metadata")
    utils.create_folder(str(metadata_folder))
    metadata_json_path = data_folder.joinpath("metadata.json")
    channel_paths = list(data_folder.glob("Ex_*_Em_*"))

    laser_side = utils.get_col_rows_per_laser(metadata_json_path=metadata_json_path)

    print("Laser sides: ", laser_side)

    save_dict_as_json(
        filename=str(results_folder.joinpath("laser_tiles.json")), dictionary=laser_side
    )

    data_processes = []
    for i, channel_path in enumerate(channel_paths):
        start_time = time.time()
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
        slices = []
        names = []

        cols = set()
        rows = set()
        for folder in channel_path.glob("*"):
            if folder.suffix == ".zarr":
                col, row = str(folder.stem).split("_")
                cols.add(col)
                rows.add(row)

        cols = natsorted(cols)
        rows = natsorted(rows)

        for indice in indices:
            params = {
                "dataset_path": channel_path,
                "cols": cols,
                "rows": rows,
                "slide_idx": indice,
                "scale": 2,
            }
            curr_slcs, curr_nms = utils.get_brain_slices(**params)
            slices.append(curr_slcs)
            names.append(curr_nms)

        shading_correction_per_slide = {}
        for slice_idx in range(len(slices)):
            curr_slices = slices[slice_idx]
            shading_correction_per_slide[slice_idx] = (
                flatfield_estimation.shading_correction(
                    slides=curr_slices, shading_parameters=shading_parameters
                )
            )

        flatfields = []
        darkfields = []
        baselines = []
        upsample_scale = SCALE * 2

        # Unifying fields with median
        for slide_idx, fields in shading_correction_per_slide.items():
            flatfields.append(fields["flatfield"])
            darkfields.append(fields["darkfield"])
            baselines.append(fields["baseline"])

        flatfield, _, _ = compute_unified_flatfield(
            flatfields, shading_correction_per_slide
        )
        print(f"Laser sides: {laser_side.keys()}")

        upsample_shape = tuple(upsample_scale * np.array(flatfield.shape))
        upsampled_flatfield = resize(
            flatfield,
            upsample_shape,
            order=4,
            mode="reflect",
            cval=0,
            clip=True,
            preserve_range=False,
            anti_aliasing=None,
        )
        output_flats = []
        for side in laser_side.keys():
            flat_name = str(
                results_folder.joinpath(
                    f"estimated_flat_laser_{channel_name}_side_{side}.tif"
                )
            )
            output_flats.append(flat_name)

            tif.imwrite(flat_name, upsampled_flatfield)

        end_time = time.time()

        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
                software_version=__version__,
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(channel_path),
                output_location=str(results_folder),
                outputs={"flatfield_paths": output_flats},
                code_url=__url__,
                code_version=__version__,
                parameters={
                    "shading_parameters": shading_parameters,
                },
                notes=f"Flatfield estimation for channel {channel_name}",
            )
        )

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=metadata_folder,
        processor_full_name=__maintainers__[0],
        pipeline_version=__pipeline_version__,
    )


if __name__ == "__main__":
    main()
