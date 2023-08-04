"""
Provide functions to work with regular grids.

@author: acollet
"""

# pylint: disable=C0103  # Do not conform to snake-case naming style
# pylint: disable=R0913  # Too many arguments
from typing import Tuple, Union

import numpy as np

from pyrtid.utils.types import NDArrayInt

Int = Union[int, NDArrayInt]


def indices_to_node_number(
    ix: Int,
    nx: int = 1,
    iy: Int = 0,
    ny: int = 1,
    iz: Int = 0,
    indices_start_at_one: bool = False,
) -> Int:
    """
    Convert indices (ix, iy, iz) to a node-number.

    For 1D and 2D, simply leave iy, ny, iz and nz to their default values.

    Note
    ----
    Node numbering start at zero.

    Warning
    -------
    This applies only for regular grids. It is not suited for vertex.

    Parameters
    ----------
    ix : int
        Index on the x-axis.
    nx : int, optional
        Number of meshes on the x-axis. The default is 1.
    iy : int, optional
        Index on the y-axis. The default is 0.
    ny : int, optional
        Number of meshes on the y-axis. The default is 1.
    iz : int, optional
        Index on the z-axis. The default is 0.
    indices_start_at_one: bool, optional
        Whether the indices start at 1. Otherwise, start at 0. The default is False.

    Returns
    -------
    int
        The node number.

    """
    if indices_start_at_one:
        ix = np.max((ix - 1, 0))
        iy = np.max((iy - 1, 0))
        iz = np.max((iz - 1, 0))
    return ix + (iy * nx) + (iz * ny * nx)


def node_number_to_indices(
    node_number: Int,
    nx: int = 1,
    ny: int = 1,
    indices_start_at_one: bool = False,
) -> Tuple[Int, Int, Int]:
    """
    Convert a node-number to indices (ix, iy, iz) for a regular grid.

    For 1D and 2D, simply leave ny, and nz to their default values.

    Note
    ----
    Node numbering start at zero.

    Warning
    -------
    This applies only for regular grids. It is not suited for vertex.

    Parameters
    ----------
    nx : int
        Number of meshes on the x-axis. The default is 1.
    ny : int, optional
        Number of meshes on the y-axis. The default is 1.
    indices_start_at_one: bool, optional
        Whether the indices start at 1. Otherwise, start at 0. The default is False.

    Returns
    -------
    int
        The node number.

    """
    ix = (node_number) % nx
    iz = (node_number - ix) // (nx * ny)
    iy = (node_number - ix - (nx * ny) * iz) // nx

    if indices_start_at_one:
        ix += 1
        iy += 1
        iz += 1

    return ix, iy, iz


def span_to_node_numbers_2d(
    span: Union[NDArrayInt, Tuple[slice, slice], slice], nx: int, ny: int
) -> NDArrayInt:
    """Convert the given span to an array of node indices."""
    _a = np.zeros((nx, ny))
    _a[span] = 1.0
    row, col = np.nonzero(_a)
    return np.array(indices_to_node_number(row, nx=nx, iy=col, ny=ny), dtype=np.int32)


def span_to_node_numbers_3d(
    span: Union[NDArrayInt, Tuple[slice, slice, slice], slice],
    nx: int,
    ny: int,
    nz: int,
) -> NDArrayInt:
    """Convert the given span to an array of node indices."""
    _a = np.zeros((nx, ny, nz))
    _a[span] = 1.0
    ix, iy, iz = np.nonzero(_a)
    return np.array(
        indices_to_node_number(ix, nx=nx, iy=iy, ny=ny, iz=iz), dtype=np.int32
    )
