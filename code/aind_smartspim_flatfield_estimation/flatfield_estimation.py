"""
Estimates the flatfield, darkfield and baseline
that will be used to correct image tiles.
"""

from typing import List, Optional

import numpy as np
from basicpy import BaSiC
from scipy.ndimage import gaussian_filter
from skimage.io import imread

from .filtering import filter_stripes


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


def create_median_flatfield(flatfield, smooth=True):
    median_row = np.median(flatfield, axis=1)
    median_image = np.tile(median_row[:, np.newaxis], (1, flatfield.shape[1]))

    if smooth:
        sigma = np.max(median_image.shape) / 100
        median_image = gaussian_filter(median_image, sigma=sigma)

    return median_image


def estimate_flats_per_laser(tiles_per_side, shading_params):
    flats = {key: None for key in tiles_per_side.keys()}
    for side, tiles in tiles_per_side.items():

        print(f"Estimating flats for side {side}")
        flats[side] = shading_correction(
            slides=tiles, shading_parameters=shading_params
        )

    return flats
