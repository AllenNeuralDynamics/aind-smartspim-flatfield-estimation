"""
Run file for flatfield estimation
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessName, ProcessStage
from aind_flatfield_correction.core.basicpy.config import get_cpu_limit
from aind_smartspim_flatfield_estimation import estimation, utils
from aind_smartspim_flatfield_estimation.__init__ import (
    __maintainers__,
    __pipeline_name__,
    __pipeline_version__,
    __title__,
    __url__,
    __version__,
)
from log_schema import setup_logging

logger = logging.getLogger(__name__)

# Microscope darkfield. The destripe capsule subtracts the same file
# when it applies the flats.
DARKFIELD_RELATIVE_PATH = "derivatives/DarkMaster_cropped.tif"


def save_dict_as_json(filename: str, dictionary: dict) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

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


def get_channel(channel_config_path: Path) -> Tuple[str, str]:
    """
    Gets the channel name and the folder with its tiles.

    Parameters
    -----------
    channel_config_path: Path
        Either a preprocess_<channel>.json file with the keys
        input_data and channel, or a Ex_*_Em_* folder.

    Returns
    -----------
    Tuple[str, str]
        Channel name and path to the channel folder.
    """
    channel_name = channel_config_path.stem
    channel_path = str(channel_config_path)

    if channel_config_path.suffix == ".json":
        channel_config = utils.read_json_as_dict(filepath=channel_config_path)
        channel_name = channel_config.get("channel")
        channel_path = f"{channel_config.get('input_data')}/{channel_name}"

    return channel_name, channel_path


def estimate_channel(
    channel_config_path: Path,
    darkfield_path: Path,
    laser_sides: Dict[str, List[str]],
    results_folder: Path,
    cpu_count: int,
) -> Tuple[DataProcess, estimation.EstimationReport, str]:
    """
    Estimates the flatfield of one channel and writes
    one flatfield tif per brain side.

    Parameters
    -----------
    channel_config_path: Path
        Channel configuration. See get_channel.

    darkfield_path: Path
        Path to the microscope darkfield.

    laser_sides: Dict[str, List[str]]
        Tiles per brain side. The keys are the sides.

    results_folder: Path
        Capsule results folder.

    cpu_count: int
        Number of cores available to the capsule.

    Returns
    -----------
    Tuple[DataProcess, EstimationReport, str]
        Provenance, report from the package and the channel name.
    """
    channel_name, channel_path = get_channel(channel_config_path)
    logger.info(f"Computing flats for channel: {channel_name}")

    estimation_folder = results_folder.joinpath("flatfield_estimation", channel_name)

    start_time = datetime.now(timezone.utc)
    resource_monitor = utils.ResourceMonitor(interval_seconds=2.0).start()

    try:
        estimation.run_estimate(
            estimation.build_estimate_argv(
                channel_path=channel_path,
                darkfield_path=str(darkfield_path),
                output_folder=str(estimation_folder),
                output_name=channel_name,
            )
        )
    finally:
        resource_monitor.stop()

    end_time = datetime.now(timezone.utc)

    report = estimation.read_estimation_report(
        estimation_folder.joinpath(f"flatfield_{channel_name}.json")
    )
    output_flats = estimation.write_side_flats(
        flatfield_path=estimation_folder.joinpath(f"flatfield_{channel_name}_level0.npy"),
        sides=list(laser_sides.keys()),
        channel_name=channel_name,
        results_folder=results_folder,
    )

    notes = f"Flatfield estimation for channel {channel_name}"
    if report.suspicious:
        notes += f". SUSPICIOUS FLATFIELD: {'; '.join(report.guard_reasons)}"
    if report.search_skipped:
        notes += f". PARAMETER SEARCH SKIPPED: {report.search_reason}"

    data_process = DataProcess(
        process_type=ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
        name=f"Flatfield estimation - {channel_name}",
        stage=ProcessStage.PROCESSING,
        code=Code(url=__url__, name=__title__, version=__version__),
        experimenters=__maintainers__,
        pipeline_name=__pipeline_name__,
        start_date_time=start_time,
        end_date_time=end_time,
        output_path=str(results_folder),
        output_parameters={
            "flatfield_paths": output_flats,
            "input_location": channel_path,
            "darkfield_path": str(darkfield_path),
            "estimation_folder": str(estimation_folder),
            "flatfield_params": report.params,
            "basic_config": report.sidecar.get("basic_config"),
            "search": report.sidecar.get("search"),
            "flatfield_stats": report.sidecar.get("flatfield_stats"),
            "pyramid_level": report.sidecar.get("pyramid_level"),
            "duration_seconds": (end_time - start_time).total_seconds(),
        },
        resources=resource_monitor.to_resource_usage(cpu_cores=int(cpu_count)),
        notes=notes,
    )

    return data_process, report, channel_name


def run_estimation(data_folder: Path, results_folder: Path) -> None:
    """
    Estimates the flatfields of all the channels of the dataset.

    All the outputs are written before a suspicious flatfield or a
    skipped parameter search raises an error. This way the outputs
    can be inspected.

    Parameters
    -----------
    data_folder: Path
        Capsule data folder.

    results_folder: Path
        Capsule results folder.

    Raises
    -----------
    ValueError:
        If a required input is missing.

    FileNotFoundError:
        If there are no channel configuration files.

    SuspiciousFlatfieldError:
        If the package marked a flatfield as suspicious.

    SearchSkippedError:
        If the package did not run the parameter search.
    """
    darkfield_path = data_folder.joinpath(DARKFIELD_RELATIVE_PATH)

    # It is assumed that these files will be in the data folder
    missing_files = validate_capsule_inputs(
        [
            f"{data_folder}/metadata.json",
            f"{data_folder}/data_description.json",
            str(darkfield_path),
        ]
    )

    if len(missing_files):
        msg = f"We miss the following files in the capsule input: {missing_files}"
        raise ValueError(msg)

    # Dispatcher generates preprocess_{channel_name}.json files
    # These are split to instantiate a single machine per channel
    channel_config_paths = list(data_folder.glob("preprocess_*.json"))

    if not channel_config_paths:
        logger.warning("No preprocess_*.json configs found, searching for Ex_*_Em_* configs...")
        channel_config_paths = list(data_folder.glob("Ex_*_Em_*"))

    if not channel_config_paths:
        raise FileNotFoundError(
            f"No channel configuration files found in {data_folder}. "
            "Expected files matching 'preprocess_*.json' or 'Ex_*_Em_*'."
        )

    metadata_folder = results_folder.joinpath("metadata")
    utils.create_folder(str(metadata_folder))

    laser_sides = utils.get_col_rows_per_laser(
        metadata_json_path=data_folder.joinpath("metadata.json")
    )
    logger.info(
        f"Computed tile layout for {len(laser_sides)} lasers",
        extra={"lasers": list(laser_sides.keys())},
    )
    save_dict_as_json(
        filename=str(results_folder.joinpath("laser_tiles.json")),
        dictionary=laser_sides,
    )

    cpu_count = get_cpu_limit()
    data_processes = []
    suspicious = {}
    search_skipped = {}

    for channel_config_path in channel_config_paths:
        data_process, report, channel_name = estimate_channel(
            channel_config_path=channel_config_path,
            darkfield_path=darkfield_path,
            laser_sides=laser_sides,
            results_folder=results_folder,
            cpu_count=cpu_count,
        )
        data_processes.append(data_process)

        if report.suspicious:
            suspicious[channel_name] = report.guard_reasons

        if report.search_skipped:
            search_skipped[channel_name] = report.search_reason

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=metadata_folder,
        pipeline_name=__pipeline_name__,
        pipeline_version=__pipeline_version__,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
    )

    if search_skipped:
        logger.error(f"Parameter search skipped: {search_skipped}")

    if suspicious:
        raise estimation.SuspiciousFlatfieldError(suspicious)

    if search_skipped:
        raise estimation.SearchSkippedError(search_skipped)


def main():
    """
    Main function
    """
    setup_logging(
        model={
            "pipeline_name": __pipeline_name__,
            "process_name": __title__,
            "software_name": __title__,
            "software_version": __version__,
        }
    )

    stage_start_time = time.monotonic()
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))

    dataset_name = utils.read_json_as_dict(
        filepath=data_folder.joinpath("data_description.json")
    ).get("name")

    try:
        logger.info(
            "Flatfield estimation started",
            extra={
                "event_type": "stage_start",
                "dataset_name": dataset_name,
                "data_folder": str(data_folder),
                "results_folder": str(results_folder),
            },
        )

        run_estimation(data_folder=data_folder, results_folder=results_folder)

        logger.info(
            "Flatfield estimation completed",
            extra={
                "event_type": "stage_complete",
                "dataset_name": dataset_name,
                "duration_seconds": round(time.monotonic() - stage_start_time, 3),
            },
        )
    except Exception as e:
        logger.error(
            "Flatfield estimation failed",
            exc_info=True,
            extra={
                "event_type": "stage_failure",
                "error": f"{type(e).__name__}: {e}",
                "dataset_name": dataset_name,
                "duration_seconds": round(time.monotonic() - stage_start_time, 3),
            },
        )
        raise


if __name__ == "__main__":
    main()
