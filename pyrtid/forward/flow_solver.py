# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide a reactive transport solver."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import LinearOperator, SuperLU, gmres

from pyrtid.forward.models import (
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    TimeParameters,
    TransportModel,
    VerticalAxis,
    get_owner_neigh_indices,
)
from pyrtid.utils import (
    Callback,
    NDArrayFloat,
    RectilinearGrid,
    arithmetic_mean,
    get_array_borders_selection_3d,
    get_super_ilu_preconditioner,
    harmonic_mean,
)


def get_kmean(
    grid: RectilinearGrid, fl_model: FlowModel, axis: int, is_flatten=True
) -> NDArrayFloat:
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
    time_index: Union[int, slice],
    is_flatten: bool = True,
) -> NDArrayFloat:
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
    grid: RectilinearGrid, fl_model: FlowModel, q_next: lil_array, axis: int
) -> None:
    kmean = get_kmean(grid, fl_model, axis)
    tmp = grid.gamma_ij(axis) / grid.pipj(axis) / grid.grid_cell_volume
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

    q_next[idc_owner, idc_neigh] -= kmean[idc_owner] * tmp  # type: ignore
    q_next[idc_owner, idc_owner] += kmean[idc_owner] * tmp  # type: ignore

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    q_next[idc_owner, idc_neigh] -= kmean[idc_neigh] * tmp  # type: ignore
    q_next[idc_owner, idc_owner] += kmean[idc_neigh] * tmp  # type: ignore


def make_stationary_flow_matrices(
    grid: RectilinearGrid, fl_model: FlowModel
) -> lil_array:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = grid.n_grid_cells
    q_next = lil_array((dim, dim), dtype=np.float64)

    for n, axis in zip(grid.shape, (0, 1, 2)):
        if n >= 2:
            fill_stationary_flmat_for_axis(grid, fl_model, q_next, axis)

    # Take constant head into account
    q_next[fl_model.cst_head_nn, fl_model.cst_head_nn] = 1.0

    return q_next


def fill_transient_flmat_for_axis(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
    axis: int,
) -> None:
    kmean = get_kmean(grid, fl_model, axis)
    rhomean = get_rhomean(grid, tr_model, axis=axis, time_index=time_index - 1)
    sc = fl_model.storage_coefficient.ravel("F")

    _tmp: float = grid.gamma_ij(axis) / grid.pipj(axis) / grid.grid_cell_volume
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
        tmp *= rhomean[idc_owner] / WATER_DENSITY

    q_next[idc_owner, idc_neigh] -= fl_model.crank_nicolson * tmp  # type: ignore
    q_next[idc_owner, idc_owner] += fl_model.crank_nicolson * tmp  # type: ignore
    q_prev[idc_owner, idc_neigh] += (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore
    q_prev[idc_owner, idc_owner] -= (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore

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
        tmp *= rhomean[idc_neigh] / WATER_DENSITY

    q_next[idc_owner, idc_neigh] -= fl_model.crank_nicolson * tmp  # type: ignore
    q_next[idc_owner, idc_owner] += fl_model.crank_nicolson * tmp  # type: ignore
    q_prev[idc_owner, idc_neigh] += (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore
    q_prev[idc_owner, idc_owner] -= (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore


def make_transient_flow_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    time_index: int,
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = grid.n_grid_cells
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    for n, axis in zip(grid.shape, (0, 1, 2)):
        if n >= 2:
            fill_transient_flmat_for_axis(
                grid, fl_model, tr_model, q_next, q_prev, time_index, axis
            )

    return q_next, q_prev


def get_zj_zi_rhs(grid: RectilinearGrid, fl_model: FlowModel) -> NDArrayFloat:
    rhs_z = np.zeros((grid.n_grid_cells), dtype=np.float64)
    z = fl_model._get_mesh_center_vertical_pos().ravel("F")

    if fl_model.vertical_axis == VerticalAxis.X:
        if grid.nx < 2:
            return rhs_z
        axis = 0
    if fl_model.vertical_axis == VerticalAxis.Y:
        if grid.ny < 2:
            return rhs_z
        axis = 1
    if fl_model.vertical_axis == VerticalAxis.Z:
        if grid.nz < 2:
            return rhs_z
        axis = 2

    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    kmean = get_kmean(grid, fl_model, axis)

    tmp = grid.gamma_ij(axis) / grid.pipj(axis) / grid.grid_cell_volume

    # Forward scheme:
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    rhs_z[idc_owner] += kmean[idc_owner] * tmp * z[idc_neigh]  # type: ignore
    rhs_z[idc_owner] -= kmean[idc_owner] * tmp * z[idc_owner]  # type: ignore

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=fl_model.free_head_nn,
    )

    rhs_z[idc_owner] += kmean[idc_neigh] * tmp * z[idc_neigh]  # type: ignore
    rhs_z[idc_owner] -= kmean[idc_neigh] * tmp * z[idc_owner]  # type: ignore

    return rhs_z


def solve_flow_stationary(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    unitflw_sources: NDArrayFloat,
    time_index: int,
) -> int:
    """
    Solving the diffusivity equation:

    dh/dt = div K grad h + ...
    """
    # Make stationary matrices
    fl_model.q_next = make_stationary_flow_matrices(grid, fl_model)
    fl_model.q_prev = lil_array((fl_model.q_next.shape))

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
    super_ilu, preconditioner = get_super_ilu_preconditioner(
        fl_model.q_next.tocsc(), drop_tol=1e-10, fill_factor=100
    )

    # only useful when using the FSM
    if fl_model.is_save_spilu:
        fl_model.super_ilu = super_ilu
        fl_model.preconditioner = preconditioner

    if super_ilu is None:
        warnings.warn(
            f"SuperILU: q_next is singular in stationary flow at it={time_index}!"
        )

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
    """
    Compute the darcy velocities at the mesh boundaries along the x axis.

    U = - k grad(h)

    Parameters
    ----------
    fl_model : FlowModel
        The

    Returns
    -------
    _type_
        _description_
    """
    dim = list(grid.shape)
    dim[axis] += 1
    out = np.zeros(tuple(dim))
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)
    kmean = get_kmean(grid, fl_model, axis=axis, is_flatten=False)[fwd_slicer]

    if fl_model.is_gravity:
        pressure = fl_model.lpressure[time_index]
        out[bwd_slicer] = (pressure[bwd_slicer] - pressure[fwd_slicer]) / grid.pipj(
            axis
        )

        rhomean = get_rhomean(
            grid, tr_model, axis=axis, time_index=time_index - 1, is_flatten=False
        )[fwd_slicer]

        if (
            (fl_model.vertical_axis == VerticalAxis.X and axis == 0)
            or (fl_model.vertical_axis == VerticalAxis.Y and axis == 1)
            or (fl_model.vertical_axis == VerticalAxis.Z and axis == 2)
        ):
            if time_index == 0:
                out[bwd_slicer] += WATER_DENSITY * GRAVITY
            else:
                out[bwd_slicer] += rhomean * GRAVITY
        else:
            pass

        # Apply the front factor
        out[bwd_slicer] *= -kmean / WATER_DENSITY / GRAVITY

    else:
        head = fl_model.lhead[time_index]
        out[bwd_slicer] = (
            -kmean * (head[bwd_slicer] - head[fwd_slicer]) / grid.pipj(axis)
        )
    return out


def compute_u_darcy(
    fl_model: FlowModel,
    tr_model: TransportModel,
    grid: RectilinearGrid,
    time_index: int,
) -> None:
    """Update the darcy velocities at the node boundaries."""
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

    # 1) Compute the flow in each cell -> oriented darcy times the node centers
    # distances
    flow = np.zeros(grid.shape)
    _flow = np.zeros(grid.shape)
    if grid.nx > 1:
        flow += fl_model.lu_darcy_x[time_index][:-1, :, :] * grid.gamma_ij_x
        flow -= fl_model.lu_darcy_x[time_index][1:, :, :] * grid.gamma_ij_x
    if grid.ny > 1:
        flow += fl_model.lu_darcy_y[time_index][:, :-1, :] * grid.gamma_ij_y
        flow -= fl_model.lu_darcy_y[time_index][:, 1:, :] * grid.gamma_ij_y
    if grid.nz > 1:
        flow += fl_model.lu_darcy_z[time_index][:, :, :-1] * grid.gamma_ij_z
        flow -= fl_model.lu_darcy_z[time_index][:, :, 1:] * grid.gamma_ij_z

    # Trick: Set the flow to zero where the head is not constant
    # Q: est-ce que c'est juste pour les constant head ????
    _flow[
        fl_model.cst_head_indices[0],
        fl_model.cst_head_indices[1],
        fl_model.cst_head_indices[2],
    ] = flow[
        fl_model.cst_head_indices[0],
        fl_model.cst_head_indices[1],
        fl_model.cst_head_indices[2],
    ]

    # Total boundary length per mesh
    _ltot = np.zeros(grid.shape)
    if grid.nx > 1:
        # evacuation along x
        if fl_model.west_boundary_idx.size != 0:
            _ltot[0, fl_model.west_boundary_idx[0], fl_model.west_boundary_idx[1]] += (
                grid.gamma_ij_x
            )
        if fl_model.east_boundary_idx.size != 0:
            _ltot[-1, fl_model.east_boundary_idx[0], fl_model.east_boundary_idx[1]] += (
                grid.gamma_ij_x
            )
    if grid.ny > 1:
        # evacuation along y
        if fl_model.south_boundary_idx.size != 0:
            _ltot[
                fl_model.south_boundary_idx[0], 0, fl_model.south_boundary_idx[1]
            ] += grid.gamma_ij_y
        if fl_model.north_boundary_idx.size != 0:
            _ltot[
                fl_model.north_boundary_idx[0], -1, fl_model.north_boundary_idx[1]
            ] += grid.gamma_ij_y
    if grid.nz > 1:
        # evacuation along z
        if fl_model.bottom_boundary_idx.size != 0:
            _ltot[
                fl_model.bottom_boundary_idx[0], fl_model.bottom_boundary_idx[1], 0
            ] += grid.gamma_ij_z
        if fl_model.top_boundary_idx.size != 0:
            _ltot[fl_model.top_boundary_idx[0], fl_model.top_boundary_idx[1], -1] += (
                grid.gamma_ij_z
            )

    # 2) Update unitflow for the constant-head nodes
    fl_model.lunitflow[time_index][
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ] = (
        _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]]
        / grid.grid_cell_volume
    )

    # 3) Now creates an artificial flow on the domain boundaries
    # to evacuate the overflow
    # It means that the unitflow added in step 2) will be set to zero for all cst head
    # grid cells located in the boundary of the domain.

    # 3.1) For constant head in the borders -> unitflow is null
    cst_head_border_mask = _flow != 0 & get_array_borders_selection_3d(*grid.shape)
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
    u_darcy_div -= fl_model.lu_darcy_x[time_index][:-1, :, :] * grid.gamma_ij_x
    u_darcy_div += fl_model.lu_darcy_x[time_index][1:, :, :] * grid.gamma_ij_x

    # y contribution
    u_darcy_div -= fl_model.lu_darcy_y[time_index][:, :-1, :] * grid.gamma_ij_y
    u_darcy_div += fl_model.lu_darcy_y[time_index][:, 1:, :] * grid.gamma_ij_y

    # z contribution
    u_darcy_div -= fl_model.lu_darcy_z[time_index][:, :, :-1] * grid.gamma_ij_z
    u_darcy_div += fl_model.lu_darcy_z[time_index][:, :, 1:] * grid.gamma_ij_z

    # Take the surface into account
    u_darcy_div /= grid.grid_cell_volume

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
    tmp = np.zeros(grid.n_grid_cells)
    sc = fl_model.storage_coefficient.ravel("F")

    if fl_model.vertical_axis == VerticalAxis.X:
        if grid.nx < 2:
            return tmp
        axis = 0
    if fl_model.vertical_axis == VerticalAxis.Y:
        if grid.ny < 2:
            return tmp
        axis = 1
    if fl_model.vertical_axis == VerticalAxis.Z:
        if grid.nz < 2:
            return tmp
        axis = 2

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
        grid.gamma_ij(axis)
        * rhomean[idc_owner] ** 2
        * GRAVITY
        / WATER_DENSITY
        * kmean[idc_owner]
        / grid.grid_cell_volume
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
        grid.gamma_ij(axis)
        * (rhomean[idc_neigh] ** 2)
        * GRAVITY
        / WATER_DENSITY
        * kmean[idc_neigh]
        / sc[idc_owner]
        / grid.grid_cell_volume
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
    Solving the diffusivity equation:

    dh/dt = div K grad h + ...
    """
    if fl_model.is_gravity or time_index == 1:
        # If the gravity is involved, then the updated density must be used and
        # consequently, the matrix must be updated
        # time_index = 1 => first time the matrix is built
        fl_model.q_next, fl_model.q_prev = make_transient_flow_matrices(
            grid, fl_model, tr_model, time_index
        )
        if not fl_model.is_gravity:  # store for the saturated case only
            fl_model.q_next_no_dt = fl_model.q_next.copy()
            fl_model.q_prev_no_dt = fl_model.q_prev.copy()
    else:
        # Otherwise it does not vary
        fl_model.q_next = fl_model.q_next_no_dt.copy()
        fl_model.q_prev = fl_model.q_prev_no_dt.copy()

    # Add 1/dt for the left term contribution (note: the timestep is variable)
    # Only for free head
    fl_model.q_next.setdiag(fl_model.q_next.diagonal() + 1 / time_params.dt)
    fl_model.q_prev.setdiag(fl_model.q_prev.diagonal() + 1 / time_params.dt)

    # Take constant head into account
    fl_model.q_next[fl_model.cst_head_nn, fl_model.cst_head_nn] = 1.0
    fl_model.q_prev[fl_model.cst_head_nn, fl_model.cst_head_nn] = 0.0

    # csc format for efficiency
    fl_model.q_next = fl_model.q_next.tocsc()
    fl_model.q_prev = fl_model.q_prev.tocsc()

    # only useful to store for dev and to check the adjoint state correctness
    if fl_model.is_save_spmats:
        fl_model.l_q_next.append(fl_model.q_next)
        fl_model.l_q_prev.append(fl_model.q_prev)

    # LU preconditioner
    super_ilu, preconditioner = get_super_ilu_preconditioner(
        fl_model.q_next.tocsc(), drop_tol=1e-10, fill_factor=100
    )

    # only useful when using the FSM
    if fl_model.is_save_spilu:
        fl_model.super_ilu = super_ilu
        fl_model.preconditioner = preconditioner

    if super_ilu is None:
        warnings.warn(
            f"SuperILU: q_next is singular in transient flow at it={time_index}!"
        )

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


def solve_fl_gmres(
    fl_model: FlowModel,
    rhs: NDArrayFloat,
    super_ilu: Optional[SuperLU] = None,
    preconditioner: Optional[LinearOperator] = None,
) -> Tuple[NDArrayFloat, int]:
    # Solve Ax = b with A sparse using LU preconditioner
    callback = Callback()
    res, exit_code = gmres(
        fl_model.q_next,
        rhs,
        x0=super_ilu.solve(rhs) if super_ilu is not None else None,
        M=preconditioner,
        rtol=fl_model.rtol,
        maxiter=1000,
        restart=20,
        callback=callback,
        callback_type="legacy",
    )
    # TODO = display
    # log...(f"Number of it for gmres {callback.itercount()}")
    return res, exit_code
