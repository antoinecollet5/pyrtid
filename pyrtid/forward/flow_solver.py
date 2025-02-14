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
    Geometry,
    TimeParameters,
    TransportModel,
    VerticalAxis,
    get_owner_neigh_indices,
)
from pyrtid.utils import (
    Callback,
    NDArrayFloat,
    arithmetic_mean,
    get_array_borders_selection,
    get_super_ilu_preconditioner,
    harmonic_mean,
)


def get_kmean(
    geometry: Geometry, fl_model: FlowModel, axis: int, is_flatten=True
) -> NDArrayFloat:
    kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
    if axis == 0:
        kmean[:-1, :] = harmonic_mean(
            fl_model.permeability[:-1, :], fl_model.permeability[1:, :]
        )
    else:
        kmean[:, :-1] = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )

    if is_flatten:
        return kmean.flatten(order="F")
    return kmean


def get_rhomean(
    geometry: Geometry,
    tr_model: TransportModel,
    axis: int,
    time_index: Union[int, slice],
    is_flatten: bool = True,
) -> NDArrayFloat:
    # get the density -> 2D or 3D array
    density = np.array(tr_model.ldensity[time_index])

    if density.ndim == 2:
        rhomean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        if axis == 0:
            rhomean[:-1, :] = arithmetic_mean(density[:-1, :], density[1:, :])
        elif axis == 1:
            rhomean[:, :-1] = arithmetic_mean(density[:, :-1], density[:, 1:])
        else:
            raise NotImplementedError("axis should be 0 or 1")
    else:
        rhomean: NDArrayFloat = np.zeros(
            (geometry.nx, geometry.ny, density.shape[0]), dtype=np.float64
        )
        if axis == 0:
            rhomean[:-1, :, :] = np.transpose(
                arithmetic_mean(density[:, :-1, :], density[:, 1:, :]), axes=(1, 2, 0)
            )
        elif axis == 1:
            rhomean[:, :-1, :] = np.transpose(
                arithmetic_mean(density[:, :, :-1], density[:, :, 1:]), axes=(1, 2, 0)
            )
        else:
            raise NotImplementedError("axis should be 0 or 1")
    if is_flatten:
        return rhomean.flatten(order="F")
    return rhomean


def make_stationary_flow_matrices(geometry: Geometry, fl_model: FlowModel) -> lil_array:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.n_grid_cells
    q_next = lil_array((dim, dim), dtype=np.float64)

    # X contribution
    if geometry.nx >= 2:
        kmean = get_kmean(geometry, fl_model, 0)

        tmp = geometry.gamma_ij_x / geometry.dx / geometry.grid_cell_volume

        if fl_model.is_gravity:
            tmp /= GRAVITY * WATER_DENSITY

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= kmean[idc_owner] * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += kmean[idc_owner] * tmp  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= kmean[idc_neigh] * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += kmean[idc_neigh] * tmp  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        kmean = get_kmean(geometry, fl_model, 1)

        tmp = geometry.gamma_ij_y / geometry.dy / geometry.grid_cell_volume

        if fl_model.is_gravity:
            tmp /= GRAVITY * WATER_DENSITY

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= kmean[idc_owner] * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += kmean[idc_owner] * tmp  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= kmean[idc_neigh] * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += kmean[idc_neigh] * tmp  # type: ignore

    # Take constant head into account
    q_next[fl_model.cst_head_nn, fl_model.cst_head_nn] = 1.0

    return q_next


def make_transient_flow_matrices(
    geometry: Geometry, fl_model: FlowModel, tr_model: TransportModel, time_index: int
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    sc = fl_model.storage_coefficient.ravel("F")

    # X contribution
    if geometry.nx > 1:
        kmean = get_kmean(geometry, fl_model, 0)
        rhomean = get_rhomean(geometry, tr_model, axis=0, time_index=time_index - 1)

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = (
            geometry.gamma_ij_x
            / geometry.dx
            / geometry.grid_cell_volume
            / sc[idc_owner]
            * kmean[idc_owner]
        )

        # Add gravity effect
        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_model.crank_nicolson * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += fl_model.crank_nicolson * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = (
            geometry.gamma_ij_x
            / geometry.dx
            / geometry.grid_cell_volume
            / sc[idc_owner]
            * kmean[idc_neigh]
        )

        # Add gravity effect
        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_model.crank_nicolson * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += fl_model.crank_nicolson * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore

    # Y contribution
    if geometry.ny > 1:
        kmean = get_kmean(geometry, fl_model, 1)
        rhomean = get_rhomean(geometry, tr_model, axis=1, time_index=time_index - 1)

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = (
            geometry.gamma_ij_y
            / geometry.dy
            / geometry.grid_cell_volume
            / sc[idc_owner]
            * kmean[idc_owner]
        )

        # Add gravity effect
        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_model.crank_nicolson * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += fl_model.crank_nicolson * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = (
            geometry.gamma_ij_y
            / geometry.dy
            / geometry.grid_cell_volume
            / sc[idc_owner]
            * kmean[idc_neigh]
        )

        # Add gravity effect
        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_model.crank_nicolson * tmp  # type: ignore
        q_next[idc_owner, idc_owner] += fl_model.crank_nicolson * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_model.crank_nicolson) * tmp  # type: ignore

    return q_next, q_prev


def get_zj_zi_rhs(geometry: Geometry, fl_model: FlowModel) -> NDArrayFloat:
    rhs_z = np.zeros((geometry.n_grid_cells), dtype=np.float64)
    z = fl_model._get_mesh_center_vertical_pos().T.ravel("F")

    rhs_z = np.zeros((geometry.n_grid_cells), dtype=np.float64)
    z = fl_model._get_mesh_center_vertical_pos().T.ravel("F")

    # X contribution
    if geometry.nx >= 2 and fl_model.vertical_axis == VerticalAxis.X:
        kmean = get_kmean(geometry, fl_model, 0)

        tmp = geometry.gamma_ij_x / geometry.dx / geometry.grid_cell_volume

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        rhs_z[idc_owner] += kmean[idc_owner] * tmp * z[idc_neigh]  # type: ignore
        rhs_z[idc_owner] -= kmean[idc_owner] * tmp * z[idc_owner]  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        rhs_z[idc_owner] += kmean[idc_neigh] * tmp * z[idc_neigh]  # type: ignore
        rhs_z[idc_owner] -= kmean[idc_neigh] * tmp * z[idc_owner]  # type: ignore

    # Y contribution
    if geometry.ny >= 2 and fl_model.vertical_axis == VerticalAxis.Y:
        kmean = get_kmean(geometry, fl_model, 1)

        tmp = geometry.gamma_ij_y / geometry.dy / geometry.grid_cell_volume

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        rhs_z[idc_owner] += kmean[idc_owner] * tmp * z[idc_neigh]  # type: ignore
        rhs_z[idc_owner] -= kmean[idc_owner] * tmp * z[idc_owner]  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        rhs_z[idc_owner] += kmean[idc_neigh] * tmp * z[idc_neigh]  # type: ignore
        rhs_z[idc_owner] -= kmean[idc_neigh] * tmp * z[idc_owner]  # type: ignore

    return rhs_z


def solve_flow_stationary(
    geometry: Geometry,
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
    fl_model.q_next = make_stationary_flow_matrices(geometry, fl_model)
    fl_model.q_prev = lil_array((fl_model.q_next.shape))

    # right hand side
    rhs = np.zeros(geometry.n_grid_cells)
    # Add the source terms
    rhs += unitflw_sources.flatten(order="F")
    if fl_model.is_gravity:
        # Constant head
        rhs[fl_model.cst_head_nn] = fl_model.lpressure[time_index].flatten(order="F")[
            fl_model.cst_head_nn
        ]
        # Non constant head only
        rhs += get_zj_zi_rhs(geometry, fl_model)
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
        fl_model.lpressure[0] = res.reshape(geometry.ny, geometry.nx).T
        # update the pressure field -> here we use the water density to be consistent
        # with HYTEC.
        fl_model.lhead[0] = (
            fl_model.lpressure[0] / GRAVITY / WATER_DENSITY
        ) + fl_model._get_mesh_center_vertical_pos().T
    else:
        fl_model.lhead[0] = res.reshape(geometry.ny, geometry.nx).T
        # update the pressure field -> here we use the water density to be consistent
        # with HYTEC.
        fl_model.lpressure[0] = (
            (fl_model.lhead[0] - fl_model._get_mesh_center_vertical_pos().T)
            * GRAVITY
            * WATER_DENSITY
        )

    compute_u_darcy(fl_model, tr_model, geometry, time_index)

    compute_u_darcy_div(fl_model, geometry, time_index)

    return exit_code


def find_ux_boundary(
    fl_model: FlowModel, geometry: Geometry, time_index: int
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
    out = np.zeros((geometry.nx + 1, geometry.ny))
    head = fl_model.lhead[time_index]
    kmean = get_kmean(geometry, fl_model, axis=0, is_flatten=False)[:-1, :]
    out[1:-1, :] = -kmean * (head[1:, :] - head[:-1, :]) / geometry.dx
    return out


def find_uy_boundary(
    fl_model: FlowModel, geometry: Geometry, time_index: int
) -> NDArrayFloat:
    """
    Compute the darcy velocities at the mesh boundaries along the y axis.

    U = - k grad(h)

    Parameters
    ----------
    fl_model : FlowModel
        _description_

    Returns
    -------
    _type_
        _description_
    """
    out = np.zeros((geometry.nx, geometry.ny + 1))
    head = fl_model.lhead[time_index]
    kmean = get_kmean(geometry, fl_model, axis=1, is_flatten=False)[:, :-1]
    out[:, 1:-1] = -kmean * (head[:, 1:] - head[:, :-1]) / geometry.dy
    return out


def find_ux_boundary_density(
    fl_model: FlowModel, tr_model: TransportModel, geometry: Geometry, time_index: int
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
    out = np.zeros((geometry.nx + 1, geometry.ny))
    pressure = fl_model.lpressure[time_index]
    kmean = get_kmean(geometry, fl_model, axis=0, is_flatten=False)[:-1, :]
    rhomean = get_rhomean(
        geometry, tr_model, axis=0, time_index=time_index - 1, is_flatten=False
    )[:-1, :]
    if fl_model.vertical_axis == VerticalAxis.X:
        rho_ij_g = rhomean * GRAVITY
        if time_index == 0:
            rho_ij_g[:, :] = WATER_DENSITY * GRAVITY
    else:
        rho_ij_g = np.zeros_like(rhomean)

    out[1:-1, :] = (
        -kmean
        / WATER_DENSITY
        / GRAVITY
        * ((pressure[1:, :] - pressure[:-1, :]) / geometry.dx + rho_ij_g)
    )
    return out


def find_uy_boundary_density(
    fl_model: FlowModel, tr_model: TransportModel, geometry: Geometry, time_index: int
) -> NDArrayFloat:
    """
    Compute the darcy velocities at the mesh boundaries along the y axis.

    U = - k grad(h)

    Parameters
    ----------
    fl_model : FlowModel
        _description_

    Returns
    -------
    _type_
        _description_
    """
    out = np.zeros((geometry.nx, geometry.ny + 1))
    pressure = fl_model.lpressure[time_index]
    kmean = get_kmean(geometry, fl_model, axis=1, is_flatten=False)[:, :-1]
    rhomean = get_rhomean(
        geometry, tr_model, axis=1, time_index=time_index - 1, is_flatten=False
    )[:, :-1]

    if fl_model.vertical_axis == VerticalAxis.Y:
        rho_ij_g = rhomean * GRAVITY
        if time_index == 0:
            rho_ij_g[:, :] = WATER_DENSITY * GRAVITY
    else:
        rho_ij_g = np.zeros_like(rhomean)

    out[:, 1:-1] = (
        -kmean
        / WATER_DENSITY
        / GRAVITY
        * ((pressure[:, 1:] - pressure[:, :-1]) / geometry.dy + rho_ij_g)
    )
    return out


def compute_u_darcy(
    fl_model: FlowModel, tr_model: TransportModel, geometry: Geometry, time_index: int
) -> None:
    """Update the darcy velocities at the node boundaries."""
    if fl_model.is_gravity:
        fl_model.lu_darcy_x.append(
            find_ux_boundary_density(fl_model, tr_model, geometry, time_index)
        )
        fl_model.lu_darcy_y.append(
            find_uy_boundary_density(fl_model, tr_model, geometry, time_index)
        )
    else:
        fl_model.lu_darcy_x.append(find_ux_boundary(fl_model, geometry, time_index))
        fl_model.lu_darcy_y.append(find_uy_boundary(fl_model, geometry, time_index))
    # Handle constant head
    update_unitflow_cst_head_nodes(fl_model, geometry, time_index)


def update_unitflow_cst_head_nodes(
    fl_model: FlowModel, geometry: Geometry, time_index: int
) -> None:
    """
    Update the darcy velocities for the constant-head nodes.

    It requires a special treatment for the system not to loose mas at the domain
    boundaries.

    Parameters
    ----------
    fl_model : FlowModel
        The flow model which contains flow parameters and variables.
    geometry : Geometry
        The geometry parameters.
    time_index : int
        Time index for which to update.
    """
    # Need to evacuate the overflow for the boundaries with constant head.
    # Note: constant head nodes can only be on the domain boundaries

    # 1) Compute the flow in each cell -> oriented darcy times the node centers
    # distances
    flow = np.zeros(geometry.shape)
    _flow = np.zeros(geometry.shape)
    if geometry.nx > 1:
        flow[:, :] += fl_model.lu_darcy_x[time_index][:-1, :] * geometry.gamma_ij_x
        flow[:, :] -= fl_model.lu_darcy_x[time_index][1:, :] * geometry.gamma_ij_x
    if geometry.ny > 1:
        flow[:, :] += fl_model.lu_darcy_y[time_index][:, :-1] * geometry.gamma_ij_y
        flow[:, :] -= fl_model.lu_darcy_y[time_index][:, 1:] * geometry.gamma_ij_y

    # Trick: Set the flow to zero where the head is not constant
    # Q: est-ce que c'est juste pour les constant head ????
    _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]] = flow[
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ]

    # Total boundary length per mesh
    _ltot = np.zeros(geometry.shape)
    if geometry.nx > 1:
        # evacuation along x
        _ltot[0, fl_model.is_boundary_west] += geometry.gamma_ij_x
        _ltot[-1, fl_model.is_boundary_east] += geometry.gamma_ij_x
    if geometry.ny > 1:
        # evacuation along y
        _ltot[fl_model.is_boundary_north, 0] += geometry.gamma_ij_y
        _ltot[fl_model.is_boundary_south, -1] += geometry.gamma_ij_y

    # 2) Update unitflow for the constant-head nodes
    fl_model.lunitflow[time_index][
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ] = (
        _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]]
        / geometry.grid_cell_volume
    )

    # 3) Now creates an artificial flow on the domain boundaries
    # to evacuate the overflow
    # It means that the unitflow added in step 2) will be set to zero for all cst head
    # grid cells located in the boundary of the domain.

    # 3.1) For constant head in the borders -> unitflow is null
    cst_head_border_mask = _flow != 0 & get_array_borders_selection(*geometry.shape)
    fl_model.lunitflow[time_index][cst_head_border_mask] = 0.0

    # 3.2) Report the flow on the boundaries
    # Note: so far, at borders, all flows are 0
    if geometry.nx > 1:
        fl_model.lu_darcy_x[time_index][0, fl_model.is_boundary_west] = (
            -_flow[0, fl_model.is_boundary_west] / _ltot[0, fl_model.is_boundary_west]
        )
        fl_model.lu_darcy_x[time_index][-1, fl_model.is_boundary_east] = (
            +_flow[-1, fl_model.is_boundary_east] / _ltot[-1, fl_model.is_boundary_east]
        )
    if geometry.ny > 1:
        fl_model.lu_darcy_y[time_index][fl_model.is_boundary_south, 0] = (
            -_flow[fl_model.is_boundary_south, 0] / _ltot[fl_model.is_boundary_south, 0]
        )
        fl_model.lu_darcy_y[time_index][fl_model.is_boundary_north, -1] = (
            +_flow[fl_model.is_boundary_north, -1]
            / _ltot[fl_model.is_boundary_north, -1]
        )


def compute_u_darcy_div(
    fl_model: FlowModel, geometry: Geometry, time_index: int
) -> None:
    """Update the darcy velocities divergence (at the node centers)."""

    # Reset to zero
    u_darcy_div = np.zeros(geometry.shape)

    # x contribution -> multiply by the frontier (dy and not dx)
    u_darcy_div -= fl_model.lu_darcy_x[time_index][:-1, :] * geometry.gamma_ij_x
    u_darcy_div += fl_model.lu_darcy_x[time_index][1:, :] * geometry.gamma_ij_x

    # y contribution  -> multiply by the frontier (dx and not dy)
    u_darcy_div -= fl_model.lu_darcy_y[time_index][:, :-1] * geometry.gamma_ij_y
    u_darcy_div += fl_model.lu_darcy_y[time_index][:, 1:] * geometry.gamma_ij_y

    # Take the surface into account
    u_darcy_div /= geometry.grid_cell_volume

    # Constant head handling - null divergence
    cst_idx = fl_model.cst_head_indices
    u_darcy_div[cst_idx[0], cst_idx[1]] = 0

    fl_model.lu_darcy_div.append(u_darcy_div)


def get_gravity_gradient(
    geometry: Geometry, fl_model: FlowModel, tr_model: TransportModel, time_index: int
) -> NDArrayFloat:
    tmp = np.zeros(geometry.nx * geometry.ny)
    sc = fl_model.storage_coefficient.ravel("F")

    if fl_model.vertical_axis == VerticalAxis.X:
        kmean = get_kmean(geometry, fl_model, axis=0)
        rhomean = get_rhomean(geometry, tr_model, axis=0, time_index=time_index - 1)

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp[idc_owner] += (
            geometry.gamma_ij_x
            * rhomean[idc_owner] ** 2
            * GRAVITY
            / WATER_DENSITY
            * kmean[idc_owner]
            / geometry.grid_cell_volume
            / sc[idc_owner]
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp[idc_owner] -= (
            geometry.gamma_ij_x
            * (rhomean[idc_neigh] ** 2)
            * GRAVITY
            / WATER_DENSITY
            * kmean[idc_neigh]
            / sc[idc_owner]
            / geometry.grid_cell_volume
        )

    elif fl_model.vertical_axis == VerticalAxis.Y:
        kmean = get_kmean(geometry, fl_model, axis=1)
        rhomean = get_rhomean(geometry, tr_model, axis=1, time_index=time_index - 1)

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp[idc_owner] += (
            geometry.gamma_ij_y
            * (rhomean[idc_owner] ** 2)
            * GRAVITY
            / WATER_DENSITY
            * kmean[idc_owner]
            / sc[idc_owner]
            / geometry.grid_cell_volume
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp[idc_owner] -= (
            geometry.gamma_ij_y
            * (rhomean[idc_neigh] ** 2)
            * GRAVITY
            / WATER_DENSITY
            * kmean[idc_neigh]
            / sc[idc_owner]
            / geometry.grid_cell_volume
        )

    return tmp


def solve_flow_transient_semi_implicit(
    geometry: Geometry,
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
            geometry, fl_model, tr_model, time_index
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
        rhs += get_gravity_gradient(geometry, fl_model, tr_model, time_index)

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
        fl_model.lpressure.append(res.reshape(geometry.nx, geometry.ny, order="F"))
        # update the pressure field
        fl_model.lhead.append(
            (fl_model.lpressure[-1] / GRAVITY / tr_model.ldensity[time_index - 1])
            + fl_model._get_mesh_center_vertical_pos().T
        )
    else:
        fl_model.lhead.append(res.reshape(geometry.nx, geometry.ny, order="F"))
        # update the pressure field -> here we use the water density to be consistent
        # with HYTEC.
        fl_model.lpressure.append(
            (fl_model.lhead[-1] - fl_model._get_mesh_center_vertical_pos().T)
            * GRAVITY
            * WATER_DENSITY
        )

    compute_u_darcy(fl_model, tr_model, geometry, time_index)

    compute_u_darcy_div(fl_model, geometry, time_index)

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
