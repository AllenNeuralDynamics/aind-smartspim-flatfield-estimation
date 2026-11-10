"""
Thin layer between the SmartSPIM capsule and the
aind-flatfield-correction package. The package estimates the
flatfield. This module builds its arguments, reads its report and
writes the files that the SmartSPIM destripe capsule expects.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import tifffile as tif

from aind_smartspim_flatfield_estimation._types import PathLike

# Default selection of the tiles: every zarr in the channel folder.
TILE_PATTERN = r"\.zarr$"
# Keys of the package sidecar that must exist. A renamed key must not
# silently become "not suspicious".
REQUIRED_SIDECAR_KEYS = ("suspicious", "guard", "params")


class FlatfieldEstimationError(RuntimeError):
    """Base class for estimation problems found after a run."""


class SearchSkippedError(FlatfieldEstimationError):
    """The package did not run the parameter search."""

    def __init__(self, reason_by_channel: Dict[str, str]):
        self.reason_by_channel = reason_by_channel
        details = "; ".join(f"{channel}: {reason}" for channel, reason in reason_by_channel.items())
        super().__init__(f"Parameter search skipped. {details}")


@dataclass
class EstimationReport:
    """
    What the package recorded about one flatfield estimation.

    Attributes
    ----------
    suspicious: bool
        True if the plausibility check failed.
    guard_reasons: List[str]
        One reason per failed plausibility check.
    search_reason: Optional[str]
        None if the parameter search ran. Otherwise the reason it did not.
    params: dict
        Parameters used for the final fit.
    sidecar: dict
        The complete sidecar json written by the package.
    """

    suspicious: bool
    guard_reasons: List[str]
    search_reason: Optional[str]
    params: dict
    sidecar: dict = field(repr=False)

    @property
    def search_skipped(self) -> bool:
        """True if the parameter search did not run."""
        return self.search_reason is not None


def build_tile_pattern(tile_ids: Sequence[str]) -> str:
    """
    Builds the regex that selects the tiles of one laser side.

    Tile folders are named <X>_<Y>.ome.zarr, and laser_tiles.json
    lists the <X>_<Y> ids of each side.

    Parameters
    ----------
    tile_ids: Sequence[str]
        Tile ids of one side, for example "345020_229180".

    Raises
    ------
    ValueError:
        If there are no tile ids.

    Returns
    -------
    str
        Regex for the --tile-pattern argument of the package.
    """
    if not tile_ids:
        raise ValueError("At least one tile id is needed to build a tile pattern.")

    alternatives = "|".join(re.escape(tile_id) for tile_id in sorted(tile_ids))
    return rf"^(?:{alternatives})\.ome\.zarr$"


def build_estimate_argv(
    channel_path: str,
    darkfield_path: str,
    output_folder: str,
    output_name: str,
    pyramid_level: int,
    max_search_minutes: float,
    validate: bool = True,
    tile_pattern: str = TILE_PATTERN,
) -> List[str]:
    """
    Builds the command-line arguments for the package estimation.

    The list never has --skip-search. The parameter
    search is mandatory.

    Parameters
    ----------
    channel_path: str
        Folder (local or s3://) with the zarr tiles of one channel.
    darkfield_path: str
        Path to the microscope darkfield image.
    output_folder: str
        Folder where the package writes its products.
    output_name: str
        Name of the package products, for example the channel name.
    pyramid_level: int
        Multiscale level used in the estimation.
    max_search_minutes: float
        Budget of the parameter search of one call. The package skips
        the search when its projected cost is more than half of this.
    validate: bool
        True to write the validation figures. Default: True
    tile_pattern: str
        Regex that selects the tiles. Default: all the zarr tiles.
        Use build_tile_pattern to select the tiles of one side.

    Returns
    -------
    List[str]
        Arguments for aind_flatfield_correction ... estimate.main
    """
    argv = [
        str(channel_path),
        "--tile-pattern",
        tile_pattern,
        "--output-name",
        output_name,
        "--output-folder",
        str(output_folder),
        "--pyramid-level",
        str(pyramid_level),
        "--darkfield-image",
        str(darkfield_path),
        "--method",
        "fit",
        "--max-search-minutes",
        str(max_search_minutes),
    ]

    if validate:
        argv.append("--validate")

    return argv


def run_estimate(argv: List[str]) -> None:
    """
    Runs the package estimation.

    Parameters
    ----------
    argv: List[str]
        Arguments from build_estimate_argv.
    """
    # Imported here because the package loads jax, which is slow.
    from aind_flatfield_correction.core.basicpy.estimate import main

    try:
        main(argv)
    except SystemExit as error:
        # argparse exits with SystemExit, which is not an Exception and
        # would skip the stage_failure log of the capsule.
        if error.code not in (None, 0):
            raise FlatfieldEstimationError(
                f"The package exited with code {error.code!r} for arguments {argv}"
            ) from error


def read_estimation_report(sidecar_path: PathLike) -> EstimationReport:
    """
    Reads the sidecar json that the package writes after an estimation.

    Parameters
    ----------
    sidecar_path: str
        Path to flatfield_<name>.json

    Raises
    ------
    FileNotFoundError:
        If the sidecar does not exist. The package did not finish.
    ValueError:
        If the sidecar misses a required key. The package format
        might have changed.

    Returns
    -------
    EstimationReport
        Verdicts of the plausibility check and the parameter search.
    """
    sidecar_path = Path(sidecar_path)

    if not sidecar_path.exists():
        raise FileNotFoundError(f"The package did not write {sidecar_path}. Check its log.")

    with open(sidecar_path) as sidecar_file:
        sidecar = json.load(sidecar_file)

    missing = [key for key in REQUIRED_SIDECAR_KEYS if key not in sidecar]
    if missing:
        raise ValueError(f"The sidecar {sidecar_path} misses the keys {missing}.")

    return EstimationReport(
        suspicious=bool(sidecar["suspicious"]),
        guard_reasons=list((sidecar.get("guard") or {}).get("reasons", [])),
        search_reason=(sidecar.get("search") or {}).get("reason"),
        params=sidecar["params"] or {},
        sidecar=sidecar,
    )


def write_flat(
    flatfield_path: PathLike,
    channel_name: str,
    side: str,
    results_folder: PathLike,
) -> str:
    """
    Writes the flatfield tif of one laser side of one channel. The
    flatfield is used for all the tiles acquired with that side.

    Parameters
    ----------
    flatfield_path: PathLike
        Path to the full-resolution flatfield (.npy) from the package.
    channel_name: str
        Channel name used in the file name.
    side: str
        Laser side, as in laser_tiles.json.
    results_folder: PathLike
        Folder where the tif file is written.

    Returns
    -------
    str
        Path of the written file.
    """
    flatfield = np.load(flatfield_path).astype(np.float32)
    output_path = str(
        Path(results_folder).joinpath(f"estimated_flat_laser_{channel_name}_side_{side}.tif")
    )
    tif.imwrite(output_path, flatfield)

    return output_path
