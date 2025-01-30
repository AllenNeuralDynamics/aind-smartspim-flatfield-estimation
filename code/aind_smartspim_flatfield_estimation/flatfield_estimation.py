"""
Estimates the flatfield, darkfield and baseline
that will be used to correct image tiles.
"""

from typing import List, Optional

import numpy as np
from basicpy import BaSiC
from scipy.ndimage import gaussian_filter


def shading_correction(
    slides: List[np.array], shading_parameters: dict, mask: Optional[np.array] = None
):
    """
    Computes shading correction for each of the
    provided tiles for further post-processing.

    Parameters
    ----------
    slides: List[List[ArrayLike]]
        List of tiles per slide used to compute
        the shading fitting.

    shading_parameters: dict
        Parameters to build the basicpy object

    mask: ArrayLike
        Mask with weights for each of the pixels
        that determines the contribution of the fields
        to remove the shadows.

    Returns
    -------
    Tuple[Dict]
        tuple with the flatfield, darkfield and
        baseline results from the shadow fitting
        for further post-processing.
    """
    shading_obj = BaSiC(**shading_parameters)
    shading_results = []
    shading_obj.fit(images=np.array(slides), fitting_weight=mask)
    shading_results = {
        "flatfield": shading_obj.flatfield,
        "darkfield": shading_obj.darkfield,
        "baseline": shading_obj.baseline,
    }

    return shading_results


def flatfield_correction(
    image_tiles: List[np.array],
    flatfield: np.array,
    darkfield: np.array,
    baseline: Optional[np.array] = None,
) -> np.array:
    """
    Corrects smartspim shadows in the tiles generated
    at the SmartSPIM light-sheet microscope.

    Parameters
    ----------
    image_tiles: List[np.array]
        Image tiles that will be corrected

    flatfield: np.array
        Estimated flatfield

    darkfield: np.array
        Estimated darkfield

    baseline: np.array
        Estimated baseline.
        Default: None

    Returns
    -------
    np.array
        Corrected tiles
    """

    image_tiles = np.array(image_tiles)

    if image_tiles.ndim != flatfield.ndim:
        flatfield = np.expand_dims(flatfield, axis=0)

    if image_tiles.ndim != darkfield.ndim:
        darkfield = np.expand_dims(darkfield, axis=0)

    darkfield = darkfield[: image_tiles.shape[-2], : image_tiles.shape[-1]]

    if darkfield.shape != image_tiles.shape:
        raise ValueError(
            f"Please, check the shape of the darkfield. Image shape: {image_tiles.shape} - Darkfield shape: {darkfield.shape}"
        )

    if flatfield.shape != image_tiles.shape:
        raise ValueError(
            f"Please, check the shape of the flatfield. Image shape: {image_tiles.shape} - Flatfield shape: {flatfield.shape}"
        )

    if baseline is None:
        baseline = np.zeros((image_tiles.shape[0],))

    baseline_indxs = tuple([slice(None)] + ([np.newaxis] * (image_tiles.ndim - 1)))

    # Subtracting dark field
    negative_darkfield = np.where(image_tiles <= darkfield)
    positive_darkfield = np.where(image_tiles > darkfield)

    # subtracting darkfield
    image_tiles[negative_darkfield] = 0
    image_tiles[positive_darkfield] = (
        image_tiles[positive_darkfield] - darkfield[positive_darkfield]
    )

    # Applying flatfield
    corrected_tiles = image_tiles / flatfield - baseline[baseline_indxs]

    # Converting back to uint16
    corrected_tiles = np.clip(corrected_tiles, 0, 65535).astype("uint16")

    return corrected_tiles


def create_median_flatfield(flatfield: np.array, smooth: Optional[bool] = True):
    """
    Generates a median-based flatfield correction image from a given input array.
    1. Calculates the median value along each row of the input `flatfield` to create a 1D array.
    2. Expands the median row array into a 2D image by tiling it along the column dimension.
    3. Optionally applies Gaussian smoothing to the resulting image. The smoothing factor (`sigma`) is based on the size of the image.

    Parameters
    ----------
    flatfield: np.ndarray:
        The input 2D array (image) for which the flatfield correction will be generated.
    smooth Optional[bool]
        If True, applies a Gaussian smoothing filter to the generated flatfield correction image.
        default: True

    Returns
    -------
    np.ndarray:
        A 2D array representing the flatfield correction image.

    """
    median_row = np.median(flatfield, axis=1)
    median_image = np.tile(median_row[:, np.newaxis], (1, flatfield.shape[1]))

    if smooth:
        sigma = np.max(median_image.shape) / 100
        median_image = gaussian_filter(median_image, sigma=sigma)

    return median_image


def estimate_flats_per_laser(tiles_per_side, shading_params):
    """
    Estimates flatfield correction images for multiple laser
    configurations, based on input tiles for each side.

    1. Initializes a dictionary with the same keys as `tiles_per_side`,
    assigning `None` as the initial value for each key.
    2. Iterates over the sides and their corresponding tiles:
        - Prints the side currently being processed.
        - Calls a `shading_correction` function (assumed to be defined
        elsewhere) to estimate the flatfield correction for the tiles of that side.
    3. Returns the dictionary of estimated flatfield corrections.

    Parameters
    ----------
    tiles_per_side: dict
        A dictionary where keys represent sides (e.g., "left", "right")
        and values are lists of tiles (image slices) for flatfield estimation.
    shading_params: dict
        A dictionary containing parameters for the shading correction process.

    Returns
    -------
    dict:
        A dictionary where keys correspond to sides (same as `tiles_per_side`)
        and values are the estimated flatfield correction images.

    """

    flats = {key: None for key in tiles_per_side.keys()}
    for side, tiles in tiles_per_side.items():

        print(f"Estimating flats for side {side}")
        flats[side] = shading_correction(
            slides=tiles, shading_parameters=shading_params
        )

    return flats


def unify_fields(
    flatfields: List[np.array],
    darkfields: List[np.array],
    baselines: List[np.array],
    mode: Optional[str] = "median",
):
    """
    Unifies the computed flatfields, darkfields and
    baselines using an statistical mode.

    Parameters
    ----------
    flatfields: List[np.array]
        List of computed flatfields per slide.

    darkfields: List[np.array]
        List of computed darkfields per slide.

    baselines: List[np.array]
        List of computed baselines per slide.

    mode: Optional[str]
        Statistical mode to combine flatfields,
        darkfields and baselines.

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        Combined flatfield, darkfield and baseline.
    """
    flatfield = None
    darkfield = None
    baseline = None

    flatfields = np.array(flatfields)
    darkfields = np.array(darkfields)
    baselines = np.array(baselines)

    if mode == "median":
        flatfield = np.median(flatfields, axis=0)
        darkfield = np.median(darkfields, axis=0)
        baseline = np.median(baselines, axis=0)

    elif mode == "mean":
        flatfield = np.mean(flatfields, axis=0)
        darkfield = np.mean(darkfields, axis=0)
        baseline = np.mean(baselines, axis=0)

    elif mode == "mip":
        flatfield = np.max(flatfields, axis=0)
        darkfield = np.min(darkfields, axis=0)
        baseline = np.max(baselines, axis=0)

    else:
        raise NotImplementedError(f"Accepted values are: ['mean', 'median', 'mip']")

    flatfield = flatfield.astype(
        np.float16
    )  # np.clip(flatfield, 0, 65535).astype('uint16')
    darkfield = darkfield.astype(
        np.float16
    )  # np.clip(darkfield, 0, 65535).astype('uint16')
    baseline = baseline.astype(
        np.float16
    )  # np.clip(baseline, 0, 65535).astype('uint16')

    return flatfield, darkfield, baseline
