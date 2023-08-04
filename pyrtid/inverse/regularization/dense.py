"""Provide dense matrix generation."""
from typing import Callable

import numpy as np
from scipy.spatial import distance_matrix

from pyrtid.utils.types import NDArrayFloat


def generate_dense_matrix(
    pts: NDArrayFloat, kernel: Callable, len_scale: NDArrayFloat
) -> NDArrayFloat:
    """
    Generate a dense matrix.

    Compute O(dim^2) interactions.

    Parameters
    ----------
    pts : NDArrayFloat
        DESCRIPTION.
    kernel : TYPE
        DESCRIPTION.
    len_scale: NDArrayFloat
        DESCRIPTION.

    Returns
    -------
    NDArrayFloat
        The dense matrix.
    """
    # Scale the points coordinates
    scaled_pts = np.array(pts, copy=True)
    for dim in range(scaled_pts.shape[1]):
        scaled_pts[:, dim] /= len_scale[dim]
    return kernel(distance_matrix(scaled_pts, scaled_pts))
