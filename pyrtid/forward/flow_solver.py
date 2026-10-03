# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Provide the flow solver.

The flow is described by the diffusivity equation, written in terms of the head
(saturated flow) or of the pressure (density driven flow, i.e. with gravity):

.. math::
    S \dfrac{\partial h}{\partial t} = \nabla \cdot (K \nabla h) + q

where :math:`S` is the storage coefficient, :math:`K` the permeability and :math:`q`
the source terms. The equation is discretized with finite volumes on a rectilinear
grid, with a Crank-Nicolson time scheme. The darcy velocities are computed at the grid
cell faces.
"""

from __future__ import annotations

import warnings

import numpy as np
import quickpaver
from inv_toolbox.utils import (
    arithmetic_mean,
    get_super_ilu_preconditioner,
    harmonic_mean,
)
from quickpaver import RectilinearGrid
from scipy import sparse
from scipy.sparse import lil_array
from scipy.sparse.linalg import LinearOperator, SuperLU, gmres

from pyrtid.forward.models import (
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    SparseMatrixBuilder,
    TimeParameters,
    TransportModel,
    add_entries,
    get_owner_neigh_indices,
)
from pyrtid.utils import NDArrayFloat

# The matrices can be filled in a lil_array or in a SparseMatrixBuilder
MatrixLike = lil_array | SparseMatrixBuilder


def get_kmean(
    grid: RectilinearGrid, fl_model: FlowModel, axis: int, is_flatten: bool = True
) -> NDArrayFloat:
    """
    Return the permeability at the faces between neighbor grid cells (harmonic mean).

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    fl_model : FlowModel
        The flow model.
    axis : int
        Axis (0 for x, 1 for y, 2 for z) along which the faces are considered.
    is_flatten : bool, optional
        Whether to return a flat array (Fortran order, i.e. indexed by node number),
        by default True. Otherwise, the array has the shape of the grid.

    Returns
    -------
    NDArrayFloat
        The mean permeability, stored in the cell located *before* the face.
        It is null in the last grid cell along ``axis``.
    """
    kmean: NDArrayFloat = np.zeros(grid.shape, dtype=np.float64)
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)
    kmean[fwd_slicer] = harmonic_mean(
        fl_model.permeability[fwd_slicer], fl_model.permeability[bwd_slicer]
    )

    if is_flatten:
        return kmean.flatten(order="F")
    return kmean


def get_rhomean(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    axis: int,
    time_index: int | slice,
    is_flatten: bool = True,
) -> NDArrayFloat:
    """
    Return the density at the faces between neighbor grid cells (arithmetic mean).

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    tr_model : TransportModel
        The transport model, which holds the densities.
    axis : int
        Axis (0 for x, 1 for y, 2 for z) along which the faces are considered.
    time_index : Union[int, slice]
        Time index (or slice of time indices) of the density.
    is_flatten : bool, optional
        Whether to return a flat array (Fortran order, i.e. indexed by node number),
        by default True. Only for a single time index.

    Returns
    -------
    NDArrayFloat
        The mean density, stored in the cell located *before* the face.
    """
    # get the density -> 2D or 3D array
    density = np.array(tr_model.ldensity[time_index])
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)
    if density.ndim == 3:
        rhomean: NDArrayFloat = np.zeros(grid.shape, dtype=np.float64)
    else:
        rhomean: NDArrayFloat = np.zeros(
            (*grid.shape, density.shape[0]), dtype=np.float64
        )
        density = np.transpose(density, axes=(1, 2, 3, 0))
    rhomean[fwd_slicer] = arithmetic_mean(density[fwd_slicer], density[bwd_slicer])

    if is_flatten:
        return rhomean.flatten(order="F")
    return rhomean


def fill_stationary_flmat_for_axis(
    grid: RectilinearGrid, fl_model: FlowModel, q_next: MatrixLike, axis: int
) -> None:
    """Add the contribution of the faces along ``axis`` to the stationary matrix."""
    kmean = get_kmean(grid, fl_model, axis)
    tmp = grid.gc_face_area_m2(axis) / grid.pipj_m(axis) / grid.grid_cell_volume_m3
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    if fl_model.is_gravity:
        tmp /= GRAVITY * WATER_DENSITY

    # Forward scheme:
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    add_entries(q_next, idc_owner, idc_neigh, -(kmean[idc_owner] * tmp))
    add_entries(q_next, idc_owner, idc_owner, kmean[idc_owner] * tmp)

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    add_entries(q_next, idc_owner, idc_neigh, -(kmean[idc_neigh] * tmp))
    add_entries(q_next, idc_owner, idc_owner, kmean[idc_neigh] * tmp)


def make_stationary_flow_matrices(
    grid: RectilinearGrid, fl_model: FlowModel
) -> lil_array:
    """
    Make the matrix of the stationary flow.

    The rows of the constant head grid cells are the identity.
    """

    dim = grid.n_grid_cells
    q_next = lil_array((dim, dim), dtype=np.float64)

    for n, axis in zip(grid.shape, (0, 1, 2), strict=False):
        if n >= 2:
            fill_stationary_flmat_for_axis(grid, fl_model, q_next, axis)

    # Take constant head into account
    q_next[fl_model.cst_head_nn, fl_model.cst_head_nn] = 1.0

    return q_next


def fill_transient_flmat_for_axis(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: MatrixLike,
    q_prev: MatrixLike,
    time_index: int,
    axis: int,
) -> None:
    """Add the contribution of the faces along ``axis`` to the transient matrices."""
    kmean = get_kmean(grid, fl_model, axis)
    # The density is only needed with the gravity
    rhomean = (
        get_rhomean(grid, tr_model, axis=axis, time_index=time_index - 1)
        if fl_model.is_gravity
        else None
    )
    sc = fl_model.storage_coefficient.ravel("F")

    _tmp: float = (
        grid.gc_face_area_m2(axis) / grid.pipj_m(axis) / grid.grid_cell_volume_m3
    )
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    # Forward scheme:
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    tmp = _tmp / sc[idc_owner] * kmean[idc_owner]

    # Add gravity effect
    if fl_model.is_gravity:
        assert rhomean is not None
        tmp *= rhomean[idc_owner] / WATER_DENSITY

    add_entries(q_next, idc_owner, idc_neigh, -(fl_model.crank_nicolson * tmp))
    add_entries(q_next, idc_owner, idc_owner, fl_model.crank_nicolson * tmp)
    add_entries(q_prev, idc_owner, idc_neigh, (1.0 - fl_model.crank_nicolson) * tmp)
    add_entries(q_prev, idc_owner, idc_owner, -((1.0 - fl_model.crank_nicolson) * tmp))

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    tmp = _tmp / sc[idc_owner] * kmean[idc_neigh]

    # Add gravity effect
    if fl_model.is_gravity:
        assert rhomean is not None
        tmp *= rhomean[idc_neigh] / WATER_DENSITY

    add_entries(q_next, idc_owner, idc_neigh, -(fl_model.crank_nicolson * tmp))
    add_entries(q_next, idc_owner, idc_owner, fl_model.crank_nicolson * tmp)
    add_entries(q_prev, idc_owner, idc_neigh, (1.0 - fl_model.crank_nicolson) * tmp)
    add_entries(q_prev, idc_owner, idc_owner, -((1.0 - fl_model.crank_nicolson) * tmp))


def _assemble_transient_flow_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    time_index: int,
) -> tuple[SparseMatrixBuilder, SparseMatrixBuilder]:
    """Fill (and return) the builders of the transient flow matrices (no 1/dt)."""
    dim = grid.n_grid_cells
    q_prev = SparseMatrixBuilder((dim, dim))
    q_next = SparseMatrixBuilder((dim, dim))

    for n, axis in zip(grid.shape, (0, 1, 2), strict=False):
        if n >= 2:
            fill_transient_flmat_for_axis(
                grid, fl_model, tr_model, q_next, q_prev, time_index, axis
            )
    return q_next, q_prev


def make_transient_flow_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    time_index: int,
) -> tuple[lil_array, lil_array]:
    """
    Make the matrices for the transient flow, without the time derivative term.

    Note
    ----
    Without gravity, the permeability and the storage coefficient do not vary with
    time, so the matrices only need to be built once. With the gravity, they depend
    on the density and must be updated.

    Returns
    -------
    Tuple[lil_array, lil_array]
        The matrices of the implicit (next time) and explicit (previous time) terms.
    """
    q_next, q_prev = _assemble_transient_flow_matrices(
        grid, fl_model, tr_model, time_index
    )
    return q_next.tolil(), q_prev.tolil()


def get_zj_zi_rhs(grid: RectilinearGrid, fl_model: FlowModel) -> NDArrayFloat:
    """
    Return the gravity contribution to the right hand side of the stationary flow.

    It is made of the differences of elevation between neighbor grid cells along the
    vertical axis, weighted by the permeability. It is null for the constant head
    grid cells and if the grid has a single cell along the vertical axis.
    """
    rhs_z = np.zeros((grid.n_grid_cells), dtype=np.float64)
    z = fl_model._get_mesh_center_vertical_pos().ravel("F")

    axis = fl_model.vertical_axis_index
    if grid.shape[axis] < 2:
        return rhs_z

    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    kmean = get_kmean(grid, fl_model, axis)

    tmp = grid.gc_face_area_m2(axis) / grid.pipj_m(axis) / grid.grid_cell_volume_m3

    # Forward scheme:
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    rhs_z[idc_owner] += kmean[idc_owner] * tmp * z[idc_neigh]
    rhs_z[idc_owner] -= kmean[idc_owner] * tmp * z[idc_owner]

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    rhs_z[idc_owner] += kmean[idc_neigh] * tmp * z[idc_neigh]
    rhs_z[idc_owner] -= kmean[idc_neigh] * tmp * z[idc_owner]

    return rhs_z


def solve_flow_stationary(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    unitflw_sources: NDArrayFloat,
    time_index: int,
) -> int:
    r"""
    Solve the stationary flow, i.e. equilibrate the initial heads.

    The stationary diffusivity equation :math:`\nabla \cdot (K \nabla h) + q = 0` is
    solved with the sources and the constant head boundary conditions. The initial
    head, pressure and darcy velocities (time index 0) are overwritten.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    fl_model : FlowModel
        The flow model.
    tr_model : TransportModel
        The transport model (for the density).
    unitflw_sources : NDArrayFloat
        The flow sources (1/s) with shape (nx, ny, nz).
    time_index : int
        The time index (0).

    Returns
    -------
    int
        The exit code of the linear solver (0 for a successful exit).
    """
    # Make stationary matrices
    fl_model.q_next = make_stationary_flow_matrices(grid, fl_model)
    fl_model.q_prev = lil_array(fl_model.q_next.shape)

    # right hand side
    rhs = np.zeros(grid.n_grid_cells)
    # Add the source terms
    rhs += unitflw_sources.flatten(order="F")
    if fl_model.is_gravity:
        # Constant head
        rhs[fl_model.cst_head_nn] = fl_model.lpressure[time_index].flatten(order="F")[
            fl_model.cst_head_nn
        ]
        # Non constant head only
        rhs += get_zj_zi_rhs(grid, fl_model)
    else:
        # Constant head
        rhs[fl_model.cst_head_nn] = fl_model.lhead[time_index].flatten(order="F")[
            fl_model.cst_head_nn
        ]

    # only useful to store for dev and to check the adjoint state correctness
    if fl_model.is_save_spmats:
        fl_model.l_q_next.append(fl_model.q_next)
        fl_model.l_q_prev.append(fl_model.q_prev)

    # LU preconditioner
    try:
        super_ilu, preconditioner = get_super_ilu_preconditioner(
            fl_model.q_next.tocsc(), drop_tol=1e-10, fill_factor=100
        )
    except RuntimeError:
        super_ilu, preconditioner = None, None
        warnings.warn(
            f"SuperILU: q_next is singular in stationary flow at it={time_index}!",
            stacklevel=2,
        )

    # only useful when using the FSM
    if fl_model.is_save_spilu:
        fl_model.super_ilu = super_ilu
        fl_model.preconditioner = preconditioner

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = solve_fl_gmres(fl_model, rhs, super_ilu, preconditioner)

    # Here we don't append but we overwrite the already existing head for t0.
    if fl_model.is_gravity:
        fl_model.lpressure[0] = res.reshape(grid.shape, order="F")
        # update the pressure field -> here we use the water density to be consistent
        # with HYTEC.
        fl_model.lhead[0] = (
            fl_model.lpressure[0] / GRAVITY / WATER_DENSITY
        ) + fl_model._get_mesh_center_vertical_pos()
    else:
        fl_model.lhead[0] = res.reshape(grid.shape, order="F")
        # update the pressure field -> here we use the water density to be consistent
        # with HYTEC.
        fl_model.lpressure[0] = (
            (fl_model.lhead[0] - fl_model._get_mesh_center_vertical_pos())
            * GRAVITY
            * WATER_DENSITY
        )

    compute_u_darcy(fl_model, tr_model, grid, time_index)

    compute_u_darcy_div(fl_model, grid, time_index)

    return exit_code


def find_u(
    fl_model: FlowModel,
    tr_model: TransportModel,
    grid: RectilinearGrid,
    time_index: int,
    axis: int,
) -> NDArrayFloat:
    r"""
    Compute the darcy velocities at the faces of the grid cells along an axis.

    :math:`U = - k \nabla h` (the gravity term is included for the vertical axis
    if the gravity is considered).

    Parameters
    ----------
    fl_model : FlowModel
        The flow model.
    tr_model : TransportModel
        The transport model (for the density).
    grid : RectilinearGrid
        The grid.
    time_index : int
        The time index.
    axis : int
        Axis (0 for x, 1 for y, 2 for z).

    Returns
    -------
    NDArrayFloat
        The velocities at the faces, with one more value than the grid has cells
        along ``axis``. The velocities of the faces on the domain borders are null.
    """
    dim = list(grid.shape)
    dim[axis] += 1
    out = np.zeros(tuple(dim))
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)
    kmean = get_kmean(grid, fl_model, axis=axis, is_flatten=False)[fwd_slicer]

    if fl_model.is_gravity:
        pressure = fl_model.lpressure[time_index]
        out[bwd_slicer] = (pressure[bwd_slicer] - pressure[fwd_slicer]) / grid.pipj_m(
            axis
        )

        if axis == fl_model.vertical_axis_index:
            if time_index == 0:
                out[bwd_slicer] += WATER_DENSITY * GRAVITY
            else:
                rhomean = get_rhomean(
                    grid,
                    tr_model,
                    axis=axis,
                    time_index=time_index - 1,
                    is_flatten=False,
                )[fwd_slicer]
                out[bwd_slicer] += rhomean * GRAVITY

        # Apply the front factor
        out[bwd_slicer] *= -kmean / WATER_DENSITY / GRAVITY

    else:
        head = fl_model.lhead[time_index]
        out[bwd_slicer] = (
            -kmean * (head[bwd_slicer] - head[fwd_slicer]) / grid.pipj_m(axis)
        )
    return out


def compute_u_darcy(
    fl_model: FlowModel,
    tr_model: TransportModel,
    grid: RectilinearGrid,
    time_index: int,
) -> None:
    """
    Update the darcy velocities at the faces of the grid cells.

    The velocities are appended to ``fl_model.lu_darcy_x``, ``lu_darcy_y`` and
    ``lu_darcy_z``, and the constant head grid cells are handled by
    :func:`update_unitflow_cst_head_nodes`.
    """
    fl_model.lu_darcy_x.append(find_u(fl_model, tr_model, grid, time_index, axis=0))
    fl_model.lu_darcy_y.append(find_u(fl_model, tr_model, grid, time_index, axis=1))
    fl_model.lu_darcy_z.append(find_u(fl_model, tr_model, grid, time_index, axis=2))

    # Handle constant head
    update_unitflow_cst_head_nodes(fl_model, grid, time_index)


def update_unitflow_cst_head_nodes(
    fl_model: FlowModel, grid: RectilinearGrid, time_index: int
) -> None:
    """
    Update the darcy velocities for the constant-head nodes.

    It requires a special treatment for the system not to loose mas at the domain
    boundaries.

    Parameters
    ----------
    fl_model : FlowModel
        The flow model which contains flow parameters and variables.
    grid : RectilinearGrid
        The grid parameters.
    time_index : int
        Time index for which to update.
    """
    # Need to evacuate the overflow for the boundaries with constant head.
    # Note: constant head nodes can only be on the domain boundaries

    # 1) Compute the flow in each cell -> oriented darcy times the faces area
    flow = np.zeros(grid.shape)
    _flow = np.zeros(grid.shape)
    if grid.nx > 1:
        flow += fl_model.lu_darcy_x[time_index][:-1, :, :] * grid.gamma_ij_x_m2
        flow -= fl_model.lu_darcy_x[time_index][1:, :, :] * grid.gamma_ij_x_m2
    if grid.ny > 1:
        flow += fl_model.lu_darcy_y[time_index][:, :-1, :] * grid.gamma_ij_y_m2
        flow -= fl_model.lu_darcy_y[time_index][:, 1:, :] * grid.gamma_ij_y_m2
    if grid.nz > 1:
        flow += fl_model.lu_darcy_z[time_index][:, :, :-1] * grid.gamma_ij_z_m2
        flow -= fl_model.lu_darcy_z[time_index][:, :, 1:] * grid.gamma_ij_z_m2

    # Trick: Set the flow to zero where the head is not constant
    cst_head_idx = fl_model.cst_head_indices
    _flow[cst_head_idx[0], cst_head_idx[1], cst_head_idx[2]] = flow[
        cst_head_idx[0], cst_head_idx[1], cst_head_idx[2]
    ]

    # Total boundary length per mesh
    _ltot = np.zeros(grid.shape)
    if grid.nx > 1:
        # evacuation along x
        if fl_model.west_boundary_idx.size != 0:
            _ltot[0, fl_model.west_boundary_idx[0], fl_model.west_boundary_idx[1]] += (
                grid.gamma_ij_x_m2
            )
        if fl_model.east_boundary_idx.size != 0:
            _ltot[-1, fl_model.east_boundary_idx[0], fl_model.east_boundary_idx[1]] += (
                grid.gamma_ij_x_m2
            )
    if grid.ny > 1:
        # evacuation along y
        if fl_model.south_boundary_idx.size != 0:
            _ltot[
                fl_model.south_boundary_idx[0], 0, fl_model.south_boundary_idx[1]
            ] += grid.gamma_ij_y_m2
        if fl_model.north_boundary_idx.size != 0:
            _ltot[
                fl_model.north_boundary_idx[0], -1, fl_model.north_boundary_idx[1]
            ] += grid.gamma_ij_y_m2
    if grid.nz > 1:
        # evacuation along z
        if fl_model.bottom_boundary_idx.size != 0:
            _ltot[
                fl_model.bottom_boundary_idx[0], fl_model.bottom_boundary_idx[1], 0
            ] += grid.gamma_ij_z_m2
        if fl_model.top_boundary_idx.size != 0:
            _ltot[fl_model.top_boundary_idx[0], fl_model.top_boundary_idx[1], -1] += (
                grid.gamma_ij_z_m2
            )

    # 2) Update unitflow for the constant-head nodes
    cst_idx = fl_model.cst_head_indices
    fl_model.lunitflow[time_index][cst_idx[0], cst_idx[1], cst_idx[2]] = (
        _flow[cst_idx[0], cst_idx[1], cst_idx[2]] / grid.grid_cell_volume_m3
    )

    # 3) Now creates an artificial flow on the domain boundaries
    # to evacuate the overflow
    # It means that the unitflow added in step 2) will be set to zero for all cst head
    # grid cells located in the boundary of the domain.

    # 3.1) For constant head in the borders -> unitflow is null
    cst_head_border_mask = (_flow != 0) & quickpaver.get_array_borders_selection(
        *grid.shape
    )
    fl_model.lunitflow[time_index][cst_head_border_mask] = 0.0

    # 3.2) Report the flow on the boundaries
    # Note: so far, at borders, all flows are 0
    if grid.nx > 1:
        if fl_model.west_boundary_idx.size != 0:
            fl_model.lu_darcy_x[time_index][
                0, fl_model.west_boundary_idx[0], fl_model.west_boundary_idx[1]
            ] = (
                -_flow[0, fl_model.west_boundary_idx[0], fl_model.west_boundary_idx[1]]
                / _ltot[0, fl_model.west_boundary_idx[0], fl_model.west_boundary_idx[1]]
            )
        if fl_model.east_boundary_idx.size != 0:
            fl_model.lu_darcy_x[time_index][
                -1, fl_model.east_boundary_idx[0], fl_model.east_boundary_idx[1]
            ] = (
                +_flow[-1, fl_model.east_boundary_idx[0], fl_model.east_boundary_idx[1]]
                / _ltot[
                    -1, fl_model.east_boundary_idx[0], fl_model.east_boundary_idx[1]
                ]
            )
    if grid.ny > 1:
        if fl_model.south_boundary_idx.size != 0:
            fl_model.lu_darcy_y[time_index][
                fl_model.south_boundary_idx[0], 0, fl_model.south_boundary_idx[1]
            ] = (
                -_flow[
                    fl_model.south_boundary_idx[0], 0, fl_model.south_boundary_idx[1]
                ]
                / _ltot[
                    fl_model.south_boundary_idx[0], 0, fl_model.south_boundary_idx[1]
                ]
            )
        if fl_model.north_boundary_idx.size != 0:
            fl_model.lu_darcy_y[time_index][
                fl_model.north_boundary_idx[0], -1, fl_model.north_boundary_idx[1]
            ] = (
                +_flow[
                    fl_model.north_boundary_idx[0],
                    -1,
                    fl_model.north_boundary_idx[1],
                ]
                / _ltot[
                    fl_model.north_boundary_idx[0],
                    -1,
                    fl_model.north_boundary_idx[1],
                ]
            )
    if grid.nz > 1:
        if fl_model.bottom_boundary_idx.size != 0:
            fl_model.lu_darcy_z[time_index][
                fl_model.bottom_boundary_idx[0], fl_model.bottom_boundary_idx[1], 0
            ] = (
                -_flow[
                    fl_model.bottom_boundary_idx[0],
                    fl_model.bottom_boundary_idx[1],
                    0,
                ]
                / _ltot[
                    fl_model.bottom_boundary_idx[0],
                    fl_model.bottom_boundary_idx[1],
                    0,
                ]
            )
        if fl_model.top_boundary_idx.size != 0:
            fl_model.lu_darcy_z[time_index][
                fl_model.top_boundary_idx[0], fl_model.top_boundary_idx[1], -1
            ] = (
                +_flow[fl_model.top_boundary_idx[0], fl_model.top_boundary_idx[1], -1]
                / _ltot[fl_model.top_boundary_idx[0], fl_model.top_boundary_idx[1], -1]
            )


def compute_u_darcy_div(
    fl_model: FlowModel, grid: RectilinearGrid, time_index: int
) -> None:
    """Update the darcy velocities divergence (at the node centers)."""

    # Reset to zero
    u_darcy_div = np.zeros(grid.shape)

    # x contribution
    u_darcy_div -= fl_model.lu_darcy_x[time_index][:-1, :, :] * grid.gamma_ij_x_m2
    u_darcy_div += fl_model.lu_darcy_x[time_index][1:, :, :] * grid.gamma_ij_x_m2

    # y contribution
    u_darcy_div -= fl_model.lu_darcy_y[time_index][:, :-1, :] * grid.gamma_ij_y_m2
    u_darcy_div += fl_model.lu_darcy_y[time_index][:, 1:, :] * grid.gamma_ij_y_m2

    # z contribution
    u_darcy_div -= fl_model.lu_darcy_z[time_index][:, :, :-1] * grid.gamma_ij_z_m2
    u_darcy_div += fl_model.lu_darcy_z[time_index][:, :, 1:] * grid.gamma_ij_z_m2

    # Take the surface into account
    u_darcy_div /= grid.grid_cell_volume_m3

    # Constant head handling - null divergence
    cst_idx = fl_model.cst_head_indices
    u_darcy_div[cst_idx[0], cst_idx[1], cst_idx[2]] = 0

    fl_model.lu_darcy_div.append(u_darcy_div)


def get_gravity_gradient(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    time_index: int,
) -> NDArrayFloat:
    """
    Return the gravity term of the right hand side of the transient (pressure) flow.

    It is null if the grid has a single cell along the vertical axis.
    """
    tmp = np.zeros(grid.n_grid_cells)
    sc = fl_model.storage_coefficient.ravel("F")

    axis = fl_model.vertical_axis_index
    if grid.shape[axis] < 2:
        return tmp

    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    kmean = get_kmean(grid, fl_model, axis=axis)
    rhomean = get_rhomean(grid, tr_model, axis=axis, time_index=time_index - 1)

    # Forward scheme:
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    tmp[idc_owner] += (
        grid.gc_face_area_m2(axis)
        * rhomean[idc_owner] ** 2
        * GRAVITY
        / WATER_DENSITY
        * kmean[idc_owner]
        / grid.grid_cell_volume_m3
        / sc[idc_owner]
    )

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    tmp[idc_owner] -= (
        grid.gc_face_area_m2(axis)
        * (rhomean[idc_neigh] ** 2)
        * GRAVITY
        / WATER_DENSITY
        * kmean[idc_neigh]
        / sc[idc_owner]
        / grid.grid_cell_volume_m3
    )

    return tmp


def solve_flow_transient_semi_implicit(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    unitflw_sources: NDArrayFloat,
    unitflw_sources_old: NDArrayFloat,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """
    Solve the transient diffusivity equation for one timestep (Crank-Nicolson).

    The head (or the pressure with the gravity) is appended to ``fl_model.lhead``
    (and ``lpressure``), and the darcy velocities and their divergence are updated.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    fl_model : FlowModel
        The flow model.
    tr_model : TransportModel
        The transport model (for the density).
    unitflw_sources : NDArrayFloat
        The flow sources (1/s) at the current time, with shape (nx, ny, nz).
    unitflw_sources_old : NDArrayFloat
        The flow sources (1/s) at the previous time.
    time_params : TimeParameters
        The time parameters (the current timestep is ``time_params.dt``).
    time_index : int
        The time index (>= 1).

    Returns
    -------
    int
        The exit code of the linear solver (0 for a successful exit).
    """
    if fl_model.is_gravity or time_index == 1:
        # If the gravity is involved, then the updated density must be used and
        # consequently, the matrix must be updated
        # time_index = 1 => first time the matrix is built
        builder_next, builder_prev = _assemble_transient_flow_matrices(
            grid, fl_model, tr_model, time_index
        )
        q_next_no_dt, q_prev_no_dt = builder_next.tocsc(), builder_prev.tocsc()
        if not fl_model.is_gravity:  # store for the saturated case only
            fl_model.q_next_no_dt = q_next_no_dt
            fl_model.q_prev_no_dt = q_prev_no_dt
    else:
        # Otherwise it does not vary
        q_next_no_dt = fl_model.q_next_no_dt
        q_prev_no_dt = fl_model.q_prev_no_dt

    # Add 1/dt for the left term contribution (note: the timestep is variable) and
    # take the constant head into account. The matrices are in csc format for
    # efficiency.
    fl_model.q_next, fl_model.q_prev = _add_time_derivative_and_cst_head(
        fl_model, q_next_no_dt, q_prev_no_dt, time_params.dt
    )

    # only useful to store for dev and to check the adjoint state correctness
    if fl_model.is_save_spmats:
        fl_model.l_q_next.append(fl_model.q_next)
        fl_model.l_q_prev.append(fl_model.q_prev)

    # LU preconditioner
    try:
        super_ilu, preconditioner = get_super_ilu_preconditioner(
            fl_model.q_next.tocsc(), drop_tol=1e-10, fill_factor=100
        )
    except RuntimeError:
        super_ilu, preconditioner = None, None
        warnings.warn(
            f"SuperILU: q_next is singular in transient flow at it={time_index}!",
            stacklevel=2,
        )

    # only useful when using the FSM
    if fl_model.is_save_spilu:
        fl_model.super_ilu = super_ilu
        fl_model.preconditioner = preconditioner

    # Add the source terms
    sources = (
        fl_model.crank_nicolson * unitflw_sources.flatten(order="F")
        + (1.0 - fl_model.crank_nicolson) * unitflw_sources_old.flatten(order="F")
    ) / fl_model.storage_coefficient.ravel(order="F")

    # Add the density effect if needed
    if fl_model.is_gravity:
        # pressure
        rhs = fl_model.q_prev.dot(fl_model.lpressure[time_index - 1].flatten(order="F"))
        rhs += sources * tr_model.ldensity[time_index - 1].flatten(order="F") * GRAVITY
        rhs += get_gravity_gradient(grid, fl_model, tr_model, time_index)

        # Handle constant head nodes
        rhs[fl_model.cst_head_nn] = fl_model.lpressure[time_index - 1].flatten(
            order="F"
        )[fl_model.cst_head_nn]
    else:
        # head
        rhs = fl_model.q_prev.dot(fl_model.lhead[time_index - 1].flatten(order="F"))
        rhs += sources
        # Handle constant head nodes
        rhs[fl_model.cst_head_nn] = fl_model.lhead[time_index - 1].flatten(order="F")[
            fl_model.cst_head_nn
        ]

    res, exit_code = solve_fl_gmres(fl_model, rhs, super_ilu, preconditioner)

    if fl_model.is_gravity:
        fl_model.lpressure.append(res.reshape(*grid.shape, order="F"))
        # update the pressure field
        fl_model.lhead.append(
            (fl_model.lpressure[-1] / GRAVITY / tr_model.ldensity[time_index - 1])
            + fl_model._get_mesh_center_vertical_pos()
        )
    else:
        fl_model.lhead.append(res.reshape(*grid.shape, order="F"))
        # update the pressure field -> here we use the water density to be consistent
        # with HYTEC.
        fl_model.lpressure.append(
            (fl_model.lhead[-1] - fl_model._get_mesh_center_vertical_pos())
            * GRAVITY
            * WATER_DENSITY
        )

    compute_u_darcy(fl_model, tr_model, grid, time_index)

    compute_u_darcy_div(fl_model, grid, time_index)

    return exit_code


def _add_time_derivative_and_cst_head(
    fl_model: FlowModel,
    q_next_no_dt: sparse.csc_array | sparse.lil_array,
    q_prev_no_dt: sparse.csc_array | sparse.lil_array,
    dt: float,
) -> tuple[sparse.csc_array, sparse.csc_array]:
    r"""
    Add the time derivative term to the transient matrices (csc format).

    :math:`1 / \Delta t` is added to the diagonal of the free head rows. The
    constant head rows are the identity in ``q_next`` and null in ``q_prev``. The
    input matrices are not modified.
    """
    n = q_next_no_dt.shape[0]
    cst = fl_model.cst_head_nn
    inv_dt = np.full(n, 1.0 / dt)
    inv_dt[cst] = 0.0
    is_cst = np.zeros(n)
    is_cst[cst] = 1.0

    def _diag(values: NDArrayFloat) -> sparse.dia_array:
        return sparse.dia_array((values[None, :], [0]), shape=(n, n))

    q_next = (q_next_no_dt.tocsc() + _diag(inv_dt + is_cst)).tocsc()
    q_prev = (q_prev_no_dt.tocsc() + _diag(inv_dt)).tocsc()
    return q_next, q_prev


def solve_fl_gmres(
    fl_model: FlowModel,
    rhs: NDArrayFloat,
    super_ilu: SuperLU | None = None,
    preconditioner: LinearOperator | None = None,
) -> tuple[NDArrayFloat, int]:
    """
    Solve ``fl_model.q_next @ x = rhs`` with GMRES.

    Parameters
    ----------
    fl_model : FlowModel
        The flow model, which holds the matrix ``q_next`` and the tolerance.
    rhs : NDArrayFloat
        The right hand side.
    super_ilu : Optional[SuperLU], optional
        Incomplete LU factorization of ``q_next``, used for the initial guess.
    preconditioner : Optional[LinearOperator], optional
        Preconditioner of the linear system.

    Returns
    -------
    Tuple[NDArrayFloat, int]
        The solution and the exit code of GMRES (0 for a successful exit).
    """
    matrix = fl_model.q_next
    if matrix.format not in ("csc", "csr"):
        # avoid a costly conversion at each matrix-vector product
        matrix = matrix.tocsc()

    res, exit_code = gmres(
        matrix,
        rhs,
        x0=super_ilu.solve(rhs) if super_ilu is not None else None,
        M=preconditioner,
        rtol=fl_model.rtol,
        maxiter=1000,
        restart=20,
    )
    if exit_code != 0:
        warnings.warn(
            f"The GMRES solver of the flow did not converge ({exit_code}).",
            stacklevel=2,
        )
    return res, exit_code
