import json
import os
from pathlib import Path
from typing import List, Optional

import dask.array as da
import numpy as np
import psutil
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)
from natsort import natsorted


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.
    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.
    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.
    """

    dictionary = {}

    if os.path.exists(filepath):
        try:
            with open(filepath) as json_file:
                dictionary = json.load(json_file)

        except UnicodeDecodeError:
            print("Error reading json with utf-8, trying different approach")
            # This might lose data, verify with Jeff the json encoding
            with open(filepath, "rb") as json_file:
                data = json_file.read()
                data_str = data.decode("utf-8", errors="ignore")
                dictionary = json.loads(data_str)

    #             print(f"Reading {filepath} forced: {dictionary}")

    return dictionary


def get_brain_slices(dataset_path, cols, rows, slide_idx, scale=0, show=False):
    imgs = []
    names = []
    n_rows = len(rows)
    n_cols = len(cols)

    dataset_path = Path(dataset_path)

    for col in cols:
        for row in rows:
            zarr_path = dataset_path.joinpath(f"{col}_{row}.zarr/{scale}")
            lazy_tile = da.from_zarr(zarr_path)[0, 0, slide_idx, ...]

            imgs.append(lazy_tile.compute())
            names.append(f"{col}_{row}.zarr")

    if show:
        show_grid(imgs, names, n_rows, n_cols, slide_idx, True)

    return np.array(imgs), names


def pick_slices(image_stack, percentage, read_lazy=True):
    """
    Pick slices from a 3D image stack based on a given percentage.

    Args:
    - image_stack: 3D numpy array representing the image stack (Z, Y, X).
    - percentage: Percentage of the Z stack to pick (between 0 and 1).

    Returns:
    - picked_slices: List of slices picked from the image stack.
    """
    z_dim = image_stack.shape[-3]
    num_slices_to_pick = int(np.floor(percentage * z_dim))

    if num_slices_to_pick == 0:
        raise ValueError("Percentage too low to pick any slices.")

    step_size = z_dim // num_slices_to_pick

    start_slice = int(z_dim * 0.2)
    end_slice = z_dim - start_slice
    slices = list(range(start_slice, end_slice, step_size))

    picked_slices = None
    if read_lazy:
        picked_slices = [image_stack[i] for i in slices]
        picked_slices = da.stack(picked_slices)

    return picked_slices, slices


def get_col_rows_per_laser(metadata_json_path):

    if not metadata_json_path.exists():
        raise FileNotFoundError(f"{metadata_json_path} does not exists.")

    laser_side = {"0": set(), "1": set()}
    matadata_json = read_json_as_dict(metadata_json_path)
    tile_config = matadata_json.get("tile_config")

    if metadata_json_path.exists() and tile_config is not None:
        for time, config in tile_config.items():

            if config["Side"] not in laser_side:
                laser_side[config["Side"]] = set()

            col_row = f"{config['X']}_{config['Y']}"

            laser_side[config["Side"]].add(col_row)
    else:
        raise ValueError(
            f"Please check the metadata path: {metadata_json_path} and the content of the file!"
        )

    # Converting to list
    for side, tiles in laser_side.items():
        laser_side[side] = list(tiles)

    return laser_side


def get_slicer_per_side(tiles_per_laser, channel_path, indices, scale=2):

    channel_path = Path(channel_path)
    data_per_laser = {k: [] for k in tiles_per_laser.keys()}
    cols = set()
    rows = set()

    for folder in channel_path.glob("*.zarr"):
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
            "scale": scale,
        }

        curr_slcs, curr_nms = get_brain_slices(**params)

        for nm_idx in range(len(curr_nms)):
            curr_nm = curr_nms[nm_idx]
            curr_slc = curr_slcs[nm_idx]

            curr_nm = curr_nm.replace(".zarr", "")

            if curr_nm in tiles_per_laser["0"]:
                data_per_laser["0"].append(curr_slc)

            elif curr_nm in tiles_per_laser["1"]:
                data_per_laser["1"].append(curr_slc)

            else:
                raise ValueError(f"Stack {curr_nm} not in the metadata")

    for side, arr in data_per_laser.items():
        data_per_laser[side] = np.array(data_per_laser[side])

    return data_per_laser


def create_folder(dest_dir, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for fusion step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata about flatfields \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)
