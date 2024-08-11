"""Provide some derivative operators."""

# pylint: disable=C0103 # doesn't conform to snake_case naming style
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix
from scipy.sparse.linalg import LinearOperator, SuperLU, spilu

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
    if axis >= 2:
        raise ValueError("axis should be 0 or 1 !")
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


def get_super_ilu_preconditioner(
    mat: Union[csc_array, csc_matrix], **kwargs
) -> Tuple[Optional[SuperLU], Optional[LinearOperator]]:
    """
    Get an incomplete LU preconditioner for the given sparse matrix.

    Reference: :cite:t:`meijerinkGuidelinesUsageIncomplete1981`.

    Note
    ----
    As wikipedia states, for a typical sparse matrix, the LU factors can be much less
    sparse than the
    original matrix — a phenomenon called fill-in. The memory requirements for using a
    direct solver can then become a bottleneck in solving linear systems. One can
    combat this problem by using fill-reducing reorderings of the matrix's unknowns,
    such as the Minimum degree algorithm.

    An incomplete factorization instead seeks triangular matrices L, U such that
    A ≈ L U A\approx LU rather than A = L U A=LU. Solving for L U x = b LUx=b can be
    done quickly but does not yield the exact solution to A x = b Ax=b. So,
    we instead use the matrix M = L U M=LU as a preconditioner in another iterative
    solution algorithm such as the conjugate gradient method or GMRES.
    """
    try:
        op = spilu(mat, **kwargs)
    except RuntimeError:  # The Factor is exactly singular
        return None, None

    def super_ilu(_x: NDArrayFloat) -> NDArrayFloat:
        return op.solve(_x)

    return op, LinearOperator(mat.shape, super_ilu)
