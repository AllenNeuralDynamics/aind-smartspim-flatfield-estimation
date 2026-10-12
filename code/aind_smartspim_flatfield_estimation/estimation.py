"""
Thin layer between the SmartSPIM capsule and the
aind-flatfield-correction package. The package estimates the
flatfield. This module builds its arguments, reads its report and
writes the files that the SmartSPIM destripe capsule expects.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile as tif

# Estimation settings that belong to SmartSPIM.
PYRAMID_LEVEL = 2
TILE_PATTERN = r"\.zarr$"
# The package skips the parameter search when its projected cost is
# more than half of this budget. The search is mandatory here.
SEARCH_BUDGET_MINUTES = 480


class FlatfieldEstimationError(RuntimeError):
    """Base class for estimation problems found after a run."""


class SuspiciousFlatfieldError(FlatfieldEstimationError):
    """The package marked one or more flatfields as suspicious."""

    def __init__(self, reasons_by_channel: Dict[str, List[str]]):
        self.reasons_by_channel = reasons_by_channel
        details = "; ".join(
            f"{channel}: {', '.join(reasons)}" for channel, reasons in reasons_by_channel.items()
        )
        super().__init__(f"Suspicious flatfield. {details}")


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


def build_estimate_argv(
    channel_path: str,
    darkfield_path: str,
    output_folder: str,
    output_name: str,
    pyramid_level: int = PYRAMID_LEVEL,
    validate: bool = True,
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
    validate: bool
        True to write the validation figures. Default: True

    Returns
    -------
    List[str]
        Arguments for aind_flatfield_correction ... estimate.main
    """
    argv = [
        str(channel_path),
        "--tile-pattern",
        TILE_PATTERN,
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
        str(SEARCH_BUDGET_MINUTES),
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

    main(argv)


def read_estimation_report(sidecar_path: str) -> EstimationReport:
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

    return EstimationReport(
        suspicious=bool(sidecar.get("suspicious")),
        guard_reasons=list(sidecar.get("guard", {}).get("reasons", [])),
        search_reason=(sidecar.get("search") or {}).get("reason"),
        params=sidecar.get("params", {}),
        sidecar=sidecar,
    )


def write_side_flats(
    flatfield_path: str,
    sides: List[str],
    channel_name: str,
    results_folder: str,
) -> List[str]:
    """
    Writes one flatfield tif per brain side. The SmartSPIM destripe
    capsule reads these files.

    Parameters
    ----------
    flatfield_path: str
        Path to the full-resolution flatfield (.npy) from the package.
    sides: List[str]
        Brain sides, for example ["0", "1"].
    channel_name: str
        Channel name used in the file names.
    results_folder: str
        Folder where the tif files are written.

    Raises
    ------
    ValueError:
        If there are no sides.

    Returns
    -------
    List[str]
        Paths of the written files, in the order of the sides.
    """
    if not len(sides):
        raise ValueError("At least one brain side is needed to write flats.")

    flatfield = np.load(flatfield_path).astype(np.float32)
    output_paths = []

    for side in sides:
        output_path = str(
            Path(results_folder).joinpath(f"estimated_flat_laser_{channel_name}_side_{side}.tif")
        )
        tif.imwrite(output_path, flatfield)
        output_paths.append(output_path)

    return output_paths
