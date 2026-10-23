"""
Utilities module
"""

import errno
import json
import os
import platform
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import psutil
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import (
    DataProcess,
    Processing,
    ResourceTimestamped,
    ResourceUsage,
)
from aind_data_schema_models.units import MemoryUnit


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

    return dictionary


def get_col_rows_per_laser(metadata_json_path: str):
    """
    Extracts column and row identifiers for each laser side from a metadata JSON file.

    1. Checks if the metadata file exists; raises an error if not.
    2. Reads the metadata JSON file and retrieves tile configuration data.
    3. Extracts and groups the column and row identifiers (`X_Y`) by laser side ("0" or "1").
    4. Converts the sets of identifiers into lists before returning.

    Parameters
    ----------
    metadata_json_path: str
        Path to the metadata JSON file.

    Returns
    -------
    dict:
        A dictionary with keys "0" and "1" (representing laser sides),
        where each key maps to a list of strings
        in the format "X_Y" representing column and row coordinates.

    Raises
    ------
    FileNotFoundError:
        If the specified metadata JSON file does not exist.
    ValueError:
        If the file does not contain valid tile configuration data.

    """
    metadata_json_path = Path(metadata_json_path)

    if not metadata_json_path.exists():
        raise FileNotFoundError(f"{metadata_json_path} does not exists.")

    laser_side = {"0": set(), "1": set()}
    metadata_json = read_json_as_dict(metadata_json_path)
    tile_config = metadata_json.get("tile_config")

    # Fix to the new microscope metadata json
    if tile_config is None:
        tile_config = metadata_json.get("tiles")

    if metadata_json_path.exists() and tile_config is not None:
        if isinstance(tile_config, dict):
            tile_config = list(tile_config.values())

        for config in tile_config:
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
            if e.errno != errno.EEXIST:
                raise


class ResourceMonitor:
    """
    Background sampler for CPU and RAM usage during a processing step.

    Samples are collected on a separate thread at a fixed interval and can be
    turned into an `aind_data_schema.core.processing.ResourceUsage` once the
    step is finished.
    """

    def __init__(self, interval_seconds: Optional[float] = 1.0):
        """
        Initializes the ResourceMonitor.
        Parameters
        ----------
        interval_seconds: Optional[float]
            Time interval in seconds between resource usage samples. Default is 1 second.
        """
        self._interval = interval_seconds
        self._cpu_usage = []
        self._ram_usage = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        """Background thread method for sampling CPU and RAM usage."""

        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            self._cpu_usage.append(
                ResourceTimestamped(timestamp=now, usage=psutil.cpu_percent(interval=None))
            )
            self._ram_usage.append(
                ResourceTimestamped(timestamp=now, usage=psutil.virtual_memory().percent)
            )
            self._stop_event.wait(self._interval)

    def start(self) -> "ResourceMonitor":
        """Starts the background sampling thread."""
        psutil.cpu_percent(interval=None)  # discard first call, which always reads 0
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stops the background sampling thread."""
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 1)

    def __enter__(self) -> "ResourceMonitor":
        """Context manager entry point to start resource monitoring."""
        return self.start()

    def __exit__(self, *exc_info) -> None:
        """Context manager exit point to stop resource monitoring."""
        self.stop()

    def to_resource_usage(self, cpu_cores: Optional[int] = None):
        """
        Builds an `aind_data_schema.core.processing.ResourceUsage` from the
        samples collected so far, plus static host information.

        Parameters
        ----------
        cpu_cores: Optional[int]
            Number of CPU cores available to the process.

        Returns
        -------
        ResourceUsage
            Resource usage record for a `DataProcess`.
        """

        return ResourceUsage(
            os=platform.system(),
            architecture=platform.machine(),
            cpu_cores=cpu_cores,
            system_memory=round(psutil.virtual_memory().total / (1024**3), 2),
            system_memory_unit=MemoryUnit.GB,
            ram=round(psutil.virtual_memory().total / (1024**3), 2),
            ram_unit=MemoryUnit.GB,
            cpu_usage=self._cpu_usage,
            ram_usage=self._ram_usage,
        )


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    pipeline_name: str,
    pipeline_version: str,
    pipeline_url: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[DataProcess]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    pipeline_name: str
        Name of the overall pipeline this processing
        step belongs to.

    pipeline_version: str
        Version of the overall pipeline.

    pipeline_url: str
        URL of the overall pipeline's repository.

    """
    # flake8: noqa: E501
    pipelines = [
        Code(
            url=pipeline_url,
            name=pipeline_name,
            version=pipeline_version,
        )
    ]

    processing = Processing.create_with_sequential_process_graph(
        data_processes=data_processes,
        pipelines=pipelines,
        notes="This processing only contains metadata about flatfields \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)
