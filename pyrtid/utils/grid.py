"""
Provide functions to work with regular grids.

@author: acollet
"""

# pylint: disable=C0103  # Do not conform to snake-case naming style
# pylint: disable=R0913  # Too many arguments
from typing import Sequence, Tuple, Union

import numpy as np

from pyrtid.utils.types import Int, NDArrayBool, NDArrayFloat, NDArrayInt


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
        ix = np.max((np.array(ix) - 1, 0))
        iy = np.max((np.array(iy) - 1, 0))
        iz = np.max((np.array(iz) - 1, 0))
    return np.array(ix) + (np.array(iy) * nx) + (np.array(iz) * ny * nx)


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
    _node_number = np.array(node_number)
    ix = (_node_number) % nx
    iz = (_node_number - ix) // (nx * ny)
    iy = (_node_number - ix - (nx * ny) * iz) // nx

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


def get_array_borders_selection(nx: int, ny: int) -> NDArrayBool:
    """
    Get a selection of the array border as a bool array.

    Note
    ----
    There is no border for an awis of dim 1.

    Parameters
    ----------
    nx: int
        Number of meshes along the x axis.
    ny: int
        Number of meshes along the y axis.
    """
    _nx = nx - 2
    if _nx < 0:
        _nx = nx
    _ny = ny - 2
    if _ny < 0:
        _ny = ny
    return np.pad(
        np.zeros((_nx, _ny), dtype=np.bool_),
        ((min(max(nx - 1, 0), 1),), ((min(max(ny - 1, 0), 1),))),
        "constant",
        constant_values=1,
    )


def get_a_not_in_b_1d(a: NDArrayInt, b: NDArrayInt) -> NDArrayInt:
    """Return the elements of a not found in b sorted by ascending order."""
    # handle the case with an empty b
    if b.size == 0 or a.size == 0:
        return a
    return np.sort(a[np.isin(a, b, invert=True)])


def get_pts_coords_regular_grid(
    mesh_dim: Union[float, Sequence[float], NDArrayFloat],
    shape: Union[int, Sequence[int], NDArrayInt],
) -> NDArrayFloat:
    """
    Create an array of points coordinates for regular grids.

    It supports from 1 to n dimensions.

    Parameters
    ----------
    mesh_dim : NDArrayInt
        Dimensions of one mesh of the grid.
    shape : NDArrayInt
        Shape of the grid (number of meshes along each axis). The number of elements
        in `shape` much match the number
        of elements in `mesh_dim`.

    Returns
    -------
    NDArrayFloat
        Array of coordinates with shape (Npts, Ndims).
    """
    # convert to numpy array
    _mesh_dim = np.array([mesh_dim]).ravel()
    _shape = np.array(shape, dtype=np.int_).ravel()
    # xmin = center of the first mesh
    xmin: NDArrayFloat = np.array(_mesh_dim) / 2.0
    # xmax  = center of the last mesh
    xmax = (_shape - 0.5) * _mesh_dim
    return (
        np.array(
            np.meshgrid(
                *[
                    np.linspace(xmin[i], xmax[i], _shape[i])
                    for i in range(_shape.size)  # type: ignore
                ],
                indexing="ij"
            )
        )
        .reshape(_shape.size, -1, order="F")
        .T
    )  # type: ignore
