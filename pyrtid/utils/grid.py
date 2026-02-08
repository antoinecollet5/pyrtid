# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide functions to work with regular grids.

@author: acollet
"""

import math
from dataclasses import dataclass

# pylint: disable=C0103  # Do not conform to snake-case naming style
# pylint: disable=R0913  # Too many arguments
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
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
        Number of grid cells on the x-axis. The default is 1.
    iy : int, optional
        Index on the y-axis. The default is 0.
    ny : int, optional
        Number of grid cells on the y-axis. The default is 1.
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
    ----node_number_to_indices
    Node numbering start at zero.

    Warning
    -------
    This applies only for regular grids. It is not suited for vertex.

    Parameters
    ----------
    nx : int
        Number of grid cells on the x-axis. The default is 1.
    ny : int, optional
        Number of grid cells on the y-axis. The default is 1.
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


def get_array_borders_selection_2d(nx: int, ny: int) -> NDArrayBool:
    """
    Get a selection of the array border as a bool array.

    Note
    ----
    There is no border for an awis of dim 1.

    Parameters
    ----------
    nx: int
        Number of grid cells along the x axis.
    ny: int
        Number of grid cells along the y axis.
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


def get_array_borders_selection_3d(nx: int, ny: int, nz: int) -> NDArrayBool:
    """
    Get a selection of the array border as a bool array.

    Note
    ----
    There is no border for an awis of dim 1.

    Parameters
    ----------
    nx: int
        Number of grid cells along the x axis.
    ny: int
        Number of grid cells along the y axis.
    nz: int
        Number of grid cells along the y zxis.
    """
    _nx = nx - 2
    if _nx < 0:
        _nx = nx
    _ny = ny - 2
    if _ny < 0:
        _ny = ny
    _nz = nz - 2
    if _nz < 0:
        _nz = nz

    return np.pad(
        np.zeros((_nx, _ny, _nz), dtype=np.bool_),
        (
            (min(max(nx - 1, 0), 1),),
            (min(max(ny - 1, 0), 1),),
            (min(max(nz - 1, 0), 1),),
        ),
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
        Shape of the grid (number of grid cells along each axis). The number of elements
        in `shape` much match the number
        of elements in `mesh_dim`.

    Returns
    -------
    NDArrayFloat
        Array of coordinates with shape (Npts, Ndims).
    """
    # convert to numpy array
    _mesh_dim = np.array([mesh_dim]).ravel()
    _shape = np.array(shape, dtype=np.int64).ravel()
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
                indexing="ij",
            )
        )
        .reshape(_shape.size, -1, order="F")
        .T
    )  # type: ignore


def rotation_x(theta) -> NDArrayFloat:
    """Matrix for a rotation around the x axis with theta radians."""
    return np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ]
    )


def rotation_y(theta) -> NDArrayFloat:
    """Matrix for a rotation around the y axis with theta radians."""
    return np.array(
        [
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)],
        ]
    )


def rotation_z(theta) -> NDArrayFloat:
    """Matrix for a rotation around the z axis with theta radians."""
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )


@dataclass
class RectilinearGrid:
    """
    Represent a rectilinear 3D grid.

    Note
    ----
    For Euler angles:
    https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        rot_center: Optional[Tuple[float, float, float]] = None,
        theta: float = 0.0,
        phi: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        x0 : float
            Grid origin x coordinate (smalest value, not centroid) in meters.
        y0 : float
            Grid origin y coordinate (smalest value, not centroid) in meters.
        z0 : float
            Grid origin z coordinate (smalest value, not centroid) in meters.
        dx : float
            Mesh size along the x axis in meters.
        dy : float
            Mesh size along the y axis in meters.
        dz : float
            Mesh size along the z axis in meters.
        nx : int
            Number of meshes along the x axis.
        ny : int
            Number of meshes along the y axis.
        nz : int
            Number of meshes along the v axis.
        rot_center:
            Coordinates (x, y, z) used as a reference point for the grid rotation.
            If None, (x0, y0, z0) is used. The default is None.
        theta : float
            z-axis rotation angle in degrees with (x0, y0, z0) as origin.
        phi : float
            y-axis-rotation angle in degrees with (x0, y0, z0) as origin.
        psi : float
            x-axis-rotation angle in degrees with (x0, y0, z0) as origin.
        """
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.dx: float = dx
        self.dy: float = dy
        self.dz: float = dz
        self._nx = 1
        self._ny = 1
        self._nz = 1
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        if rot_center is not None:
            self.rot_center: Tuple[float, float, float] = rot_center
        else:
            self.rot_center = (x0, y0, z0)
        self.theta = theta
        self.phi = phi
        self.psi = psi

        if self.nx < 3 and self.ny < 3:
            raise (ValueError("At least one of (nx, ny) should be of dimension 3"))

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the grid."""
        return (self.nx, self.ny, self.nz)

    @property
    def nx(self) -> int:
        """Return the number of grid cells along the x axis."""
        return self._nx

    @nx.setter
    def nx(self, value: int) -> None:
        if value < 1:
            raise (ValueError("nx should be >= 1!)"))
        self._nx = value

    @property
    def ny(self) -> int:
        """Return the number of grid cells along the y axis."""
        return self._ny

    @ny.setter
    def ny(self, value: int) -> None:
        if value < 1:
            raise (ValueError("ny should be >= 1!)"))
        self._ny = value

    @property
    def nz(self) -> int:
        """Return the number of grid cells along the z axis."""
        return self._nz

    @nz.setter
    def nz(self, value: int) -> None:
        if value < 1:
            raise (ValueError("nz should be >= 1!)"))
        self._nz = value

    def pipj(self, axis: int) -> float:
        if axis == 0:
            return self.dx
        if axis == 1:
            return self.dy
        if axis == 2:
            return self.dz
        raise ValueError("`axis` should be among [0, 1, 2]")

    @property
    def n_grid_cells(self) -> int:
        """Return the number of grid cells."""
        return self.nx * self.ny * self.nz

    @property
    def grid_cell_surface(self) -> float:
        """Return the surface of the grid cell in the x-y plan (m2)."""
        return self.dx * self.dy

    @property
    def grid_cell_volume(self) -> float:
        """Return the volume of a voxel in m3."""
        return self.dx * self.dy * self.dz

    @property
    def grid_cell_volume_m3(self) -> float:
        """Return the volume of one voxel in m3."""
        return self.dx * self.dy * self.dz

    @property
    def total_volume_m3(self) -> float:
        """Return the total grid volume in m3."""
        return self.grid_cell_volume_m3 * self.n_grid_cells

    @property
    def gamma_ij_x(self) -> float:
        """Return the surface of the frontiers along the x axis in m2"""
        return self.dy * self.dz

    @property
    def gamma_ij_y(self) -> float:
        """Return the surface of the frontiers along the y axis in m2"""
        return self.dx * self.dz

    @property
    def gamma_ij_z(self) -> float:
        """Return the surface of the frontiers along the z axis in m2"""
        return self.dx * self.dy

    def gamma_ij(self, axis: int) -> float:
        """Return the surface of the frontiers along the z axis in m2"""
        if axis == 0:
            return self.gamma_ij_x
        elif axis == 1:
            return self.gamma_ij_y
        elif axis == 2:
            return self.gamma_ij_z
        raise ValueError("`axis` should be among [0, 1, 2]")

    @property
    def indices(self) -> NDArrayInt:
        """Return the grid indices with shape (3, nx, ny, nz)."""
        return np.asarray(
            np.meshgrid(range(self.nx), range(self.ny), range(self.nz), indexing="ij"),
            dtype=np.int64,
        )

    @property
    def _non_rotated_origin_coords(self) -> NDArrayFloat:
        """
        Return the grid meshes origin coordinates with shape (3, nx, ny, nz).

        Note
        ----
        Rotation is not applied.
        """
        return (
            self.indices.reshape(3, -1, order="F")
            * np.array([[self.dx, self.dy, self.dz]], dtype=np.float64).T
            + np.array([[self.x0, self.y0, self.z0]]).T
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    def _rotate_coords(self, non_rotated_coords: NDArrayFloat) -> NDArrayFloat:
        """
        Rotate the coordinates.

        Parameters
        ----------
        non_rotated_coords: NDArrayFloat
            Expected shape (3, nx, ny, nz)

        Note
        ----
        The rotation with the matrices multiplication is done relatively to point
        (0.0, 0.0, 0.0), so we should remove the origin point (x0, y0, z0) before the
        rotation and add it afterward.

        Return
        ------
        NDArrayFloat
            The rotated coordinates with shape (3, nx, ny, nz).
        """
        return (
            np.dot(
                rotation_x(np.deg2rad(self.psi)),
                np.dot(
                    rotation_y(np.deg2rad(self.phi)),
                    np.dot(
                        rotation_z(np.deg2rad(self.theta)),
                        non_rotated_coords.reshape(3, -1, order="F")
                        - np.array([[self.x0, self.y0, self.z0]]).T,
                    ),
                ),
            )
            + np.array([[self.x0, self.y0, self.z0]]).T
        )

    @property
    def origin_coords(self) -> NDArrayFloat:
        """Return the grid meshes origin coordinates with shape (3, nx, ny, nz)."""
        return self._rotate_coords(self._non_rotated_origin_coords).reshape(
            3, self.nx, self.ny, self.nz, order="F"
        )

    @property
    def x_indices(self) -> NDArrayInt:
        """Return the grid meshes x-indices as 1D array."""
        return self.indices[0].ravel()

    @property
    def y_indices(self) -> NDArrayInt:
        """Return the grid meshes y-indices as 1D array."""
        return self.indices[1].ravel()

    @property
    def z_indices(self) -> NDArrayInt:
        """Return the grid meshes z-indices as 1D array."""
        return self.indices[2].ravel()

    @property
    def center_coords(self) -> NDArrayFloat:
        """Return the grid meshes center coordinates with shape (3, nx, ny, nz)."""
        return self._rotate_coords(
            (
                self._non_rotated_origin_coords.reshape(3, -1, order="F")
                + np.array([[self.dx / 2, self.dy / 2, self.dz / 2]]).T
            )
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    @property
    def center_coords_2d(self) -> NDArrayFloat:
        """Return the coordinates of the voxel centers for an xy slice."""
        return self.center_coords[:2, :, :, 0]

    @property
    def _opposite_vertice_coords(self) -> NDArrayFloat:
        """
        Return the grid meshes opposite coordinates with shape (3, nx, ny, nz).

        Note
        ----
        The opposite vertice is the origin symmetric with respect to the cell center.
        """
        return self._rotate_coords(
            (
                self._non_rotated_origin_coords.reshape(3, -1, order="F")
                + np.array([[self.dx, self.dy, self.dz]]).T
            )
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    @property
    def bounding_box_vertices_coordinates(self) -> NDArrayFloat:
        """Return the coordinates of the 8 bounding box vertices."""
        tmp = np.array(
            [
                [self.x0, self.y0, self.z0],
                [self.x0 + self.nx * self.dx, self.y0, self.z0],
                [self.x0 + self.nx * self.dx, self.y0 + self.ny * self.dy, self.z0],
                [self.x0, self.y0 + self.ny * self.dy, self.z0],
                [self.x0, self.y0, self.z0 + self.nz * self.dz],
                [self.x0 + self.nx * self.dx, self.y0, self.z0 + self.nz * self.dz],
                [
                    self.x0 + self.nx * self.dx,
                    self.y0 + self.ny * self.dy,
                    self.z0 + self.nz * self.dz,
                ],
                [self.x0, self.y0 + self.ny * self.dy, self.z0 + self.nz * self.dz],
            ]
        ).T
        return self._rotate_coords(tmp)

    @property
    def bounds(self) -> NDArrayFloat:
        """Return the bounds [[xmin, xmax], [ymin, ymax], [zmin, zmax]]."""
        # Create an array with the coordinates of the 8 non rotated grid summits
        # Apply rotation
        _ = self.bounding_box_vertices_coordinates
        return np.array(
            [
                _.min(axis=1),
                _.max(axis=1),
            ]
        ).T

    @property
    def xmin(self) -> float:
        """Return the minimum x of the grid."""
        return self.bounds[0, 0]

    @property
    def xmax(self) -> float:
        """Return the maximum x of the grid."""
        return self.bounds[0, 1]

    @property
    def ymin(self) -> float:
        """Return the minimum y of the grid."""
        return self.bounds[1, 0]

    @property
    def ymax(self) -> float:
        """Return the maximum y of the grid."""
        return self.bounds[1, 1]

    @property
    def zmin(self) -> float:
        """Return the minimum z of the grid."""
        return self.bounds[2, 0]

    @property
    def zmax(self) -> float:
        """Return the maximum z of the grid."""
        return self.bounds[2, 1]

    @property
    def x_extent(self) -> float:
        """Return the x extent in meters."""
        return self.xmax - self.xmin

    @property
    def y_extent(self) -> float:
        """Return the y extent in meters."""
        return self.ymax - self.ymin

    @property
    def z_extent(self) -> float:
        """Return the z extent in meters."""
        return self.zmax - self.zmin

    def get_slicer_forward(
        self, axis: int, shift: int = 0
    ) -> Tuple[slice, slice, slice]:
        if axis == 0:
            return (slice(0, self.nx - 1 + shift), slice(None), slice(None))
        if axis == 1:
            return (slice(None), slice(0, self.ny - 1 + shift), slice(None))
        if axis == 2:
            return (slice(None), slice(None), slice(0, self.nz - 1 + shift))
        raise ValueError("axis should be in [0, 1, 2]")

    def get_slicer_backward(
        self, axis: int, shift: int = 0
    ) -> Tuple[slice, slice, slice]:
        if axis == 0:
            return (slice(1, self.nx + shift), slice(None), slice(None))
        if axis == 1:
            return (slice(None), slice(1, self.ny + shift), slice(None))
        if axis == 2:
            return (slice(None), slice(None), slice(1, self.nz + shift))
        raise ValueError("axis should be in [0, 1, 2]")


def get_vertices_centroid(
    vertices: Union[NDArrayFloat, List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """Get the vertices centroid."""
    _x_list = [vertex[0] for vertex in vertices]
    _y_list = [vertex[1] for vertex in vertices]
    _len = len(vertices)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (_x, _y)


def get_centroid_voxel_coords(
    vertices: Union[NDArrayFloat, List[Tuple[float, float]]],
    grid: RectilinearGrid,
) -> Tuple[int, int]:
    """
    For a given convex polygon an a 2D grid, give the centroid voxel.

    Parameters
    ----------
    vertices : Union[NDArrayFloat, List[Tuple[float, float]]]
        Coords of the convex polygon exterior ring with shape (M, 2)
    grid: RectilinearGrid,
        The grid definition (dimensions, position, etc.).

    Returns
    -------
    Tuple[int, int]
        x, y coordinates of the centroid voxel.
    """
    # This works for convex polygons only
    _x, _y = get_vertices_centroid(vertices)
    # get the closer integer
    distances = np.square(grid.center_coords_2d[0].ravel("F") - _x) + np.square(
        grid.center_coords_2d[1].ravel("F") - _y
    )
    ix, iy, _ = node_number_to_indices(
        int(np.argmin(distances)), nx=grid.nx, ny=grid.ny
    )
    return (ix, iy)


def create_selections_array_2d(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    sel_ids: Union[Sequence[int], NDArrayInt],
    grid: RectilinearGrid,
) -> NDArrayInt:
    """
    Return a grid array containing the sel_ids as values.

    The grid array has the dimension of the grid. It ensure that one grid block
    corresponds to a unique selection id.

    Parameters
    ----------
    polygons : Sequence[Sequence[Tuple[float, float]]]
        Sequence of polygons for which to perform the selection in the grid.
        The order matters as the first polygon will be prioritize if overlapping
        between polygons occurs.
    sel_ids : Union[Sequence[int], NDArrayInt]
        Sequence integers selection ids. An id cannot be zero
        (reserved for no selection).
    grid : RectilinearGrid
        The grid object for which to performm the selection.

    Returns
    -------
    NDArrayInt
        Grid selections array.
    """
    if 0 in sel_ids:
        raise ValueError(
            "0 cannot be part has sel_ids. It is reserved for empty selection."
        )

    # flatten points coordinates
    _sel_array = np.zeros(grid.shape, dtype=np.int8)

    # The mask sum ensure that a voxel is not selected twice
    mask_sum: Optional[NDArrayInt] = None
    for _polygon, cell_id in zip(polygons, sel_ids):
        # Select the mesh that belongs to the polygon
        path = mpl.path.Path(_polygon)
        mask = path.contains_points(
            grid.center_coords[:2, :, :, 0].reshape(2, -1, order="F").T
        )
        if mask_sum is not None:
            mask = np.logical_and(mask, ~mask_sum)
            mask_sum = np.logical_or(mask, mask_sum)
        else:
            mask_sum = mask
        _sel_array[mask.reshape(grid.nx, grid.ny, order="F")] = cell_id
    return _sel_array


def get_free_grid_cells(selection) -> NDArrayBool:
    """Return the free grid cells (no selected) as a boolean array."""
    return selection == 0


def _get_mask(
    polygon, selection: NDArrayInt, center_coords_2d: NDArrayFloat, nx: int, ny: int
) -> NDArrayBool:
    # Select the mesh that belongs to the polygon
    path = mpl.path.Path(polygon)
    mask = np.reshape(
        path.contains_points(center_coords_2d),
        (nx, ny),
        "F",
    )
    # Make sure that the voxels are not already part of a selection
    mask = np.logical_and(mask, get_free_grid_cells(selection))
    return mask


def binary_dilation(
    input: NDArrayBool, mask: NDArrayBool, iterations: int = 1
) -> NDArrayBool:
    _arr = input.copy()
    _arr[1:, :] = np.where(input[:-1, :], True, _arr[1:, :])
    _arr[:-1, :] = np.where(input[1:, :], True, _arr[:-1, :])
    _arr[:, 1:] = np.where(input[:, :-1], True, _arr[:, 1:])
    _arr[:, :-1] = np.where(input[:, 1:], True, _arr[:, :-1])
    # apply the masking
    _arr[~mask] = input[~mask]
    return _arr


def get_polygon_selection_with_dilation_2d(
    polygons: Union[List[NDArrayFloat], List[List[Tuple[float, float]]]],
    grid: RectilinearGrid,
    selection: Optional[NDArrayInt] = None,
) -> NDArrayInt:
    """Extend the selections using binary dilation.

    Parameters
    ----------
    polygon : Union[NDArrayFloat, List[Tuple[float, float]]]
        Coords of the exterior ring with shape (M, 2)
    grid: RectilinearGrid,
        The grid definition (dimensions, position, etc.).
    selection: Optional[NDArrayInt]
        An already existing selection as starting point. The default is None.
    """
    # Start by creating an empty grid with int type
    if selection is None:
        _selection = np.zeros((grid.nx, grid.ny), dtype=np.int8)
    else:
        _selection = selection.copy()

    # initiate _oldselection variable
    _old_selection = np.zeros((grid.nx, grid.ny), dtype=np.int8)

    # Grid coordinates -> Flat array
    _grid_coords_2d = grid.center_coords_2d.reshape(2, -1, order="F").T

    # Create an initial selection for each cell (only one voxel selected)
    sel_ids = np.arange(len(polygons)) + 1
    for sel_id, vertices in zip(sel_ids, polygons):
        _selection[get_centroid_voxel_coords(vertices, grid)] = sel_id

    # Perform the dilation iteration by iteration to ensure a better split between
    # the selections.
    while np.not_equal(_selection, _old_selection).any():
        # Update the old_grid with the new one for the while
        _old_selection = _selection.copy()
        # Perform the dilation for each selection
        for sel_id, vertices in zip(sel_ids, polygons):
            # The mask is free cells + the contained
            mask = _get_mask(vertices, _selection, _grid_coords_2d, grid.nx, grid.ny)
            _selection[
                binary_dilation(_selection == sel_id, mask=mask, iterations=1)
            ] = sel_id
    return _selection


def get_extended_grid_shape(
    grid: RectilinearGrid, axis: int, extend: int
) -> Tuple[int, int, int]:
    if axis == 0:
        return (grid.nx + extend, grid.ny, grid.nz)
    if axis == 1:
        return (grid.nx, grid.ny + extend, grid.nz)
    if axis == 2:
        return (grid.nx, grid.ny, grid.nz + extend)
    raise ValueError()
