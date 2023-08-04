"""Provide some derivative operators."""
# pylint: disable=C0103 # doesn't conform to snake_case naming style
from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, splu

from pyrtid.utils.types import NDArrayFloat


def gradient_ffd(param: NDArrayFloat, dx: float, axis: int = 0) -> NDArrayFloat:
    """
    Compute the gradient using the first order forward differences.

    The returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    param : np.array
        An N-dimensional array containing samples of a scalar function.
    dx : float
        Spacing between param values along the axis.
    axis: int
        Axis on which to compute the gradient (0=x, 1=y)

    Returns
    -------
    grad : NDArrayFloat
        The gradient.
    """
    if axis >= 2:
        raise ValueError("axis should be 0 or 1 !")
    # initiate a nul gradient
    grad = np.zeros(param.shape)

    # Compute the gradient
    if axis == 0:
        grad[:-1, :] += (param[1:, :] - param[:-1, :]) / dx
    else:
        grad[:, :-1] += (param[:, 1:] - param[:, :-1]) / dx
    return grad


def gradient_bfd(param: NDArrayFloat, dx: float, axis: int = 0) -> NDArrayFloat:
    """
    Compute the gradient using the first order forward differences.

    The returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    param : np.array
        An N-dimensional array containing samples of a scalar function.
    dx : float
        Spacing between param values along the axis.
    axis: int
        Axis on which to compute the gradient (0=x, 1=y)

    Returns
    -------
    grad : NDArrayFloat
        The gradient.
    """
    if axis >= 2:
        raise ValueError("axis should be 0 or 1 !")
    # initiate a nul gradient
    grad = np.zeros(param.shape)

    # Compute the gradient
    if axis == 0:
        grad[1:, :] += (param[1:, :] - param[:-1, :]) / dx
    else:
        grad[:, 1:] += (param[:, 1:] - param[:, :-1]) / dx
    return grad


def hessian_cfd(param: NDArrayFloat, dx: float, axis: int = 0) -> NDArrayFloat:
    """
    Compute the hessian matching `gradient_ffd`.

    Parameters
    ----------
    param : np.array
        An N-dimensional array containing samples of a scalar function.
    dx : float
        Spacing between param values along the axis.
    axis: int
        Axis on which to compute the gradient (0=x, 1=y)

    Returns
    -------
    grad : NDArrayFloat
        The hessian

    """
    # number of scalar samples:
    nx, ny = param.shape
    # initiate a nul gradient
    hess = np.zeros((nx, ny))

    # Compute the gradient
    if axis == 0:
        hess[1:-1, :] += (param[:-2, :] - 2 * param[1:-1, :] + param[2:, :]) / (dx**2)
        # at the edges

        hess[0, :] += (-param[0, :] + param[1, :]) / dx**2
        hess[-1, :] += (param[-2, :] - param[-1, :]) / dx**2
    else:
        hess[:, 1:-1] += (param[:, :-2] - 2 * param[:, 1:-1] + param[:, 2:]) / (dx**2)
        # at the edges
        hess[:, 0] += (-param[:, 0] + param[:, 1]) / dx**2
        hess[:, -1] += (param[:, -2] - param[:, -1]) / dx**2
    return hess


def get_super_lu_preconditioner(mat: csc_matrix) -> Optional[LinearOperator]:
    """Get the preconditioner for the given matrix."""
    try:
        op = splu(mat)
    except RuntimeError:  # The Factor is exactly singular
        return None

    def super_lu(_x: NDArrayFloat) -> NDArrayFloat:
        return op.solve(_x)

    return LinearOperator(mat.shape, super_lu)
