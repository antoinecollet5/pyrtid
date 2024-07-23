"""
Toeplitz matrix-vector multiplication adapted from Arvind Saibaba's code
https://github.com/arvindks/kle/blob/master/covariance/toeplitz/toeplitz.py
and PyPCGA:
https://github.com/jonghyunharrylee/pyPCGA/blob/master/pyPCGA/covariance/toeplitz.py
"""

from typing import Callable, List, Sequence, Union

import numpy as np

from pyrtid.utils.types import NDArrayFloat, NDArrayInt


def create_toepliz_first_row(
    pts: NDArrayFloat,
    kernel: Callable,
    lenscale: NDArrayFloat,
) -> NDArrayFloat:
    """
    Create the first row of the covariance matrix.

    Note
    ----
    The first row is also the first column of the matrix
    because covariance matrices are symmetric.

    Parameters
    ----------
    pts : NDArrayFloat
        Coordinates of the grid mesh centers of shape (Npts, Ndim), with Ndim the number
        of spatial dimensions between 1 and n.
    kernel : Callable
        Covariance kernel.
    lenscale : NDArrayFloat
        Correlation length.

    Returns
    -------
    NDArrayFloat
        Array of correlations between the first point of the array and all gridblocks
        with size Npts.
    """
    # We scale the points coordinates by the correlation length
    scaled_pts = pts / np.array(lenscale).reshape(1, -1)
    # Then we compute the distance for the first row of the matrix (between) the first
    # mesh center and all others. We square the difference over all axis and take the
    # sqrt of the sum (euclidean distance).
    scaled_distances = np.sqrt(np.sum((scaled_pts - scaled_pts[0]) ** 2, axis=1))
    return kernel(scaled_distances)


def toeplitz_product(
    x: NDArrayFloat,
    first_row: NDArrayFloat,
    shape: Union[int, Sequence[int], NDArrayInt],
) -> NDArrayFloat:
    """
    Return the product of the covariance matrix with a vector using toeplitz trick.

    It is easy to see that covariance matrices corresponding to regular grids
    (in 2D and 3D) result in block Toeplitz matrices, with Toeplitz sub-blocks (BTTB).
    We can exploit this to perform fast matrix-vector products.
    See :cite:t:`saibabaFastAlgorithmsGeostatistical2013`.

    Copied from
    https://github.com/jonghyunharrylee/pyPCGA/blob/master/pyPCGA/covariance/toeplitz.py

    Parameters
    ----------
    x : NDArrayFloat
        Input vector to multiply.
    first_row : NDArrayFloat
        First row of the covariance matrix.
    shape : Union[int, Sequence[int], NDArrayInt]
        Shape of the grid (nx, [ny, nz])

    Returns
    -------
    NDArrayFloat
        The matrix-vector product.

    Raises
    ------
    ValueError
        If the given shape does not match 1D, 2D or 3D.
    """
    _shape = np.array([shape], dtype=np.int64).ravel()
    dim: int = _shape.size

    if dim > 3 or dim == 0:
        raise ValueError("Support 1,2 and 3 dimensions.")

    # Reshape the input vector with domain shape
    padded = np.reshape(x, _shape, order="F")

    # Create the circulant matrix
    circ = np.reshape(first_row, _shape, order="F")

    mask: List[slice] = [slice(None) for i in range(dim)]
    for i in range(dim):
        # remove first and last element on that axis
        mask[dim - i - 1] = slice(1, -1)
        # flip the axis and remove first and last element
        circ = np.concatenate(
            (circ, np.flip(circ, axis=dim - i - 1)[tuple(mask)]), axis=dim - i - 1
        )
        # restore the mask to its initial state
        mask[dim - i - 1] = slice(None)

    if dim == 1:
        padded = np.concatenate((x, np.zeros(_shape[0] - 2)))

    result = np.fft.ifftn(np.fft.fftn(circ) * np.fft.fftn(padded, np.shape(circ)))

    # Get the result and return
    mask = [slice(0, _shape[i]) for i in range(dim)]

    return np.reshape(np.real(result[tuple(mask)]), -1, order="F")
