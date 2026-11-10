"""
Run file for flatfield estimation
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

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

# Defaults of the capsule inputs. See parse_args.
# Microscope darkfield, relative to the data folder. The destripe capsule
# subtracts the same file when it applies the flats.
DEFAULT_DARKFIELD_RELATIVE_PATH = "derivatives/DarkMaster_cropped.tif"
# Zarr multiscale level used for the fit.
DEFAULT_PYRAMID_LEVEL = 2
# Budget of the parameter search PER CALL, that is per channel and laser
# side (calls run one after the other). The package skips the search
# when its projected cost is more than half of this (60 min). The
# search is mandatory here.
DEFAULT_SEARCH_BUDGET_MINUTES = 120.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses the inputs of the capsule.

    Parameters
    -----------
    argv: Optional[List[str]]
        Arguments to parse. Default: sys.argv[1:]

    Returns
    -----------
    argparse.Namespace
        Namespace with darkfield_relative_path, pyramid_level and
        search_budget_minutes.
    """
    parser = argparse.ArgumentParser(description="Flatfield estimation for SmartSPIM datasets")
    parser.add_argument(
        "--darkfield-relative-path",
        default=DEFAULT_DARKFIELD_RELATIVE_PATH,
        help="Microscope darkfield, relative to the data folder. "
        f"Default: {DEFAULT_DARKFIELD_RELATIVE_PATH}",
    )
    parser.add_argument(
        "--pyramid-level",
        type=int,
        default=DEFAULT_PYRAMID_LEVEL,
        help=f"Zarr multiscale level used for the fit. Default: {DEFAULT_PYRAMID_LEVEL}",
    )
    parser.add_argument(
        "--search-budget-minutes",
        type=float,
        default=DEFAULT_SEARCH_BUDGET_MINUTES,
        help="Budget of the parameter search of each channel and laser side. "
        "The package skips the search if its projected cost is more than half of it. "
        f"Default: {DEFAULT_SEARCH_BUDGET_MINUTES:g}",
    )

    args = parser.parse_args(argv)

    if args.pyramid_level < 0:
        parser.error("--pyramid-level must be 0 or greater")

    if args.search_budget_minutes <= 0:
        parser.error("--search-budget-minutes must be greater than 0")

    return args


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
        input_data = channel_config.get("input_data")

        if not channel_name or not input_data:
            raise ValueError(
                f"{channel_config_path} must have the keys 'channel' and 'input_data'."
            )

        channel_path = f"{input_data}/{channel_name}"

    return channel_name, channel_path


def estimate_channel_side(
    channel_config_path: Path,
    side: str,
    tile_ids: List[str],
    darkfield_path: Path,
    results_folder: Path,
    cpu_count: int,
    pyramid_level: int,
    search_budget_minutes: float,
) -> Tuple[DataProcess, estimation.EstimationReport, str]:
    """
    Estimates the flatfield of one laser side of one channel and
    writes its flatfield tif.

    Parameters
    -----------
    channel_config_path: Path
        Channel configuration. See get_channel.

    side: str
        Laser side, as in laser_tiles.json.

    tile_ids: List[str]
        Ids (<X>_<Y>) of the tiles acquired with this side.

    darkfield_path: Path
        Path to the microscope darkfield.

    results_folder: Path
        Capsule results folder.

    cpu_count: int
        Number of cores available to the capsule.

    pyramid_level: int
        Zarr multiscale level used for the fit.

    search_budget_minutes: float
        Budget of the parameter search of this call.

    Returns
    -----------
    Tuple[DataProcess, EstimationReport, str]
        Provenance, report from the package and the channel name
        with its side, for example Ex_488_Em_525_side_0.
    """
    channel_name, channel_path = get_channel(channel_config_path)
    label = f"{channel_name}_side_{side}"
    logger.info(f"Computing flats for channel: {channel_name}, side: {side}")

    # One folder per side: the package overwrites its run log
    # (metadata/estimation.log) in the output folder.
    estimation_folder = results_folder.joinpath(
        "flatfield_estimation", channel_name, f"side_{side}"
    )

    start_time = datetime.now(timezone.utc)
    resource_monitor = utils.ResourceMonitor(interval_seconds=2.0).start()

    try:
        estimation.run_estimate(
            estimation.build_estimate_argv(
                channel_path=channel_path,
                darkfield_path=str(darkfield_path),
                output_folder=str(estimation_folder),
                output_name=label,
                pyramid_level=pyramid_level,
                max_search_minutes=search_budget_minutes,
                tile_pattern=estimation.build_tile_pattern(tile_ids),
            )
        )
    finally:
        resource_monitor.stop()

    end_time = datetime.now(timezone.utc)

    report = estimation.read_estimation_report(
        estimation_folder.joinpath(f"flatfield_{label}.json")
    )
    output_flat = estimation.write_flat(
        flatfield_path=estimation_folder.joinpath(f"flatfield_{label}_level0.npy"),
        channel_name=channel_name,
        side=side,
        results_folder=results_folder,
    )

    notes = f"Flatfield estimation for channel {channel_name}, side {side}"
    if report.suspicious:
        notes += f". SUSPICIOUS FLATFIELD: {'; '.join(report.guard_reasons)}"
        logger.warning(
            f"Suspicious flatfield for channel {channel_name}, side {side}",
            extra={"channel": channel_name, "side": side, "reasons": report.guard_reasons},
        )
    if report.search_skipped:
        notes += f". PARAMETER SEARCH SKIPPED: {report.search_reason}"

    data_process = DataProcess(
        process_type=ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
        name=f"Flatfield estimation - {label}",
        stage=ProcessStage.PROCESSING,
        code=Code(url=__url__, name=__title__, version=__version__),
        experimenters=__maintainers__,
        pipeline_name=__pipeline_name__,
        start_date_time=start_time,
        end_date_time=end_time,
        output_path=str(results_folder),
        output_parameters={
            "flatfield_paths": [output_flat],
            "input_location": channel_path,
            "side": side,
            "n_tiles": len(tile_ids),
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

    return data_process, report, label


def run_estimation(
    data_folder: Path,
    results_folder: Path,
    darkfield_relative_path: str = DEFAULT_DARKFIELD_RELATIVE_PATH,
    pyramid_level: int = DEFAULT_PYRAMID_LEVEL,
    search_budget_minutes: float = DEFAULT_SEARCH_BUDGET_MINUTES,
) -> None:
    """
    Estimates the flatfields of every laser side of every channel
    of the dataset.

    All the outputs are written before a skipped parameter search
    raises an error. A suspicious flatfield is only logged and flagged
    in the processing notes. This way the outputs
    can be inspected.

    Parameters
    -----------
    data_folder: Path
        Capsule data folder.

    results_folder: Path
        Capsule results folder.

    darkfield_relative_path: str
        Microscope darkfield, relative to the data folder.

    pyramid_level: int
        Zarr multiscale level used for the fit.

    search_budget_minutes: float
        Budget of the parameter search of each channel and laser side.

    Raises
    -----------
    ValueError:
        If a required input is missing.

    FileNotFoundError:
        If there are no channel configuration files.

    SearchSkippedError:
        If the package did not run the parameter search.
    """
    darkfield_path = data_folder.joinpath(darkfield_relative_path)

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
    search_skipped = {}

    try:
        for channel_config_path in channel_config_paths:
            for side, tile_ids in sorted(laser_sides.items()):
                if not tile_ids:
                    logger.warning(f"Laser side {side} has no tiles, skipping its flatfield")
                    continue

                data_process, report, label = estimate_channel_side(
                    channel_config_path=channel_config_path,
                    side=side,
                    tile_ids=tile_ids,
                    darkfield_path=darkfield_path,
                    results_folder=results_folder,
                    cpu_count=cpu_count,
                    pyramid_level=pyramid_level,
                    search_budget_minutes=search_budget_minutes,
                )
                data_processes.append(data_process)

                if report.search_skipped:
                    search_skipped[label] = report.search_reason
    finally:
        # Keeps the provenance of the flatfields that finished
        if data_processes:
            utils.generate_processing(
                data_processes=data_processes,
                dest_processing=metadata_folder,
                pipeline_name=__pipeline_name__,
                pipeline_version=__pipeline_version__,
                pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
            )

    if not data_processes:
        raise ValueError(f"No laser side has tiles in {data_folder}/metadata.json")

    if search_skipped:
        logger.error(f"Parameter search skipped: {search_skipped}")

    if search_skipped:
        raise estimation.SearchSkippedError(search_skipped)


def main(argv: Optional[List[str]] = None):
    """
    Main function

    Parameters
    -----------
    argv: Optional[List[str]]
        Command-line arguments. Default: sys.argv[1:]
    """
    args = parse_args(argv)

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
                "pyramid_level": args.pyramid_level,
                "search_budget_minutes": args.search_budget_minutes,
                "darkfield_relative_path": args.darkfield_relative_path,
            },
        )

        run_estimation(
            data_folder=data_folder,
            results_folder=results_folder,
            darkfield_relative_path=args.darkfield_relative_path,
            pyramid_level=args.pyramid_level,
            search_budget_minutes=args.search_budget_minutes,
        )

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
