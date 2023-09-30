"""Provide a reactive transport solver."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import gmres

from pyrtid.utils import (
    get_array_borders_selection,
    get_super_ilu_preconditioner,
    harmonic_mean,
)
from pyrtid.utils.types import NDArrayFloat

from .models import FlowModel, Geometry, TimeParameters, get_owner_neigh_indices


def make_stationary_flow_matrices(geometry: Geometry, fl_model: FlowModel) -> lil_array:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_next = lil_array((dim, dim), dtype=np.float64)

    # X contribution
    if geometry.nx >= 2:
        kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        kmean[:-1, :] = harmonic_mean(
            fl_model.permeability[:-1, :], fl_model.permeability[1:, :]
        )
        kmean = kmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = geometry.dy / geometry.dx / geometry.mesh_volume

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
        kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        kmean[:, :-1] = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )
        kmean = kmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = geometry.dx / geometry.dy / geometry.mesh_volume

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
    geometry: Geometry, fl_model: FlowModel, time_params: TimeParameters
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

    # X contribution
    kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
    kmean[:-1, :] = harmonic_mean(
        fl_model.permeability[:-1, :], fl_model.permeability[1:, :]
    )
    kmean = kmean.flatten(order="F")

    # Forward scheme:
    if geometry.nx >= 2:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = geometry.dy / geometry.dx / geometry.mesh_volume

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        kmean[:, :-1] = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )
        kmean = kmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        tmp = geometry.dx / geometry.dy / geometry.mesh_volume

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            * tmp
            / fl_model.storage_coefficient
        )  # type: ignore

    return q_next, q_prev


def solve_flow_stationary(
    geometry: Geometry,
    fl_model: FlowModel,
    flw_sources: NDArrayFloat,
    time_index: int,
) -> int:
    """
    Solving the diffusivity equation:

    dh/dt = div K grad h + ...
    """
    # Multiply prev matrix by prev vector
    tmp = np.zeros(fl_model.q_next.shape[0], dtype=np.float64)
    tmp[fl_model.cst_head_nn] = fl_model.lhead[time_index].flatten(order="F")[
        fl_model.cst_head_nn
    ]

    # LU preconditioner
    preconditioner = get_super_ilu_preconditioner(fl_model.q_next.tocsc())

    # Add the source terms
    tmp += flw_sources.flatten(order="F")

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(
        fl_model.q_next.tocsc(), tmp, M=preconditioner, atol=fl_model.tolerance
    )

    # Here we don't append but we overwrite the already existing head for t0.
    fl_model.lhead[0] = res.reshape(geometry.ny, geometry.nx).T

    compute_u_darcy(fl_model, geometry, time_index)

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
    kmean = harmonic_mean(fl_model.permeability[:-1, :], fl_model.permeability[1:, :])
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
    kmean = harmonic_mean(fl_model.permeability[:, :-1], fl_model.permeability[:, 1:])
    out[:, 1:-1] = -kmean * (head[:, 1:] - head[:, :-1]) / geometry.dy
    return out


def compute_u_darcy(fl_model: FlowModel, geometry: Geometry, time_index: int) -> None:
    """Update the darcy velocities at the node boundaries."""
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

    # 1) Compute the flow in each cell -> oriented darcy times the node centers
    # distances
    flow = np.zeros(geometry.shape)
    _flow = np.zeros(geometry.shape)
    flow[:, :] += fl_model.lu_darcy_x[time_index][:-1, :] * geometry.dy
    flow[:, :] -= fl_model.lu_darcy_x[time_index][1:, :] * geometry.dy
    flow[:, :] += fl_model.lu_darcy_y[time_index][:, :-1] * geometry.dx
    flow[:, :] -= fl_model.lu_darcy_y[time_index][:, 1:] * geometry.dx

    # Trick: Set the flow to zero where the head is not constant
    _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]] = flow[
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ]
    # 2) Update unitflow for the constant-head nodes
    fl_model.lunitflow[time_index][
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ] = (
        _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]]
        / geometry.mesh_volume
    )

    # 3) Now creates an artificial flow on the domain boundaries
    # to evacuate the overflow

    # For constant head on the borders -> unitflow is null
    cst_head_border_mask = _flow != 0 & get_array_borders_selection(*geometry.shape)
    fl_model.lunitflow[time_index][cst_head_border_mask] = 0.0

    # on the border, one neighbour maximum on each axis
    ltot = 0
    if geometry.nx > 1:
        ltot += geometry.dy
    if geometry.ny > 1:
        ltot += geometry.dx

    # Note: so far, at borders, all flows are 0, so we can apply this to all nodes,
    # constant head or not.
    fl_model.lu_darcy_x[time_index][0, :] = -_flow[0, :] / ltot
    fl_model.lu_darcy_x[time_index][-1, :] = _flow[-1, :] / ltot
    fl_model.lu_darcy_y[time_index][:, 0] = -_flow[:, 0] / ltot
    fl_model.lu_darcy_y[time_index][:, -1] = _flow[:, -1] / ltot


def compute_u_darcy_div(
    fl_model: FlowModel, geometry: Geometry, time_index: int
) -> None:
    """Update the darcy velocities divergence (at the node centers)."""

    # Reset to zero
    u_darcy_div = np.zeros(geometry.shape)

    # x contribution -> multiply by the frontier (dy and not dx)
    u_darcy_div -= fl_model.lu_darcy_x[time_index][:-1, :] * geometry.dy
    u_darcy_div += fl_model.lu_darcy_x[time_index][1:, :] * geometry.dy

    # y contribution  -> multiply by the frontier (dx and not dy)
    u_darcy_div -= fl_model.lu_darcy_y[time_index][:, :-1] * geometry.dx
    u_darcy_div += fl_model.lu_darcy_y[time_index][:, 1:] * geometry.dx

    # Take the surface into account
    u_darcy_div /= geometry.mesh_volume

    # Constant head handling - null divergence
    cst_idx = fl_model.cst_head_indices
    u_darcy_div[cst_idx[0], cst_idx[1]] = 0

    fl_model.lu_darcy_div.append(u_darcy_div)


def solve_flow_transient_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
    flw_sources: NDArrayFloat,
    flw_sources_old: NDArrayFloat,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """
    Solving the diffusivity equation:

    dh/dt = div K grad h + ...
    """
    _q_next = fl_model.q_next.copy()
    _q_prev = fl_model.q_prev.copy()

    # Add 1/dt for the left term contribution (note: the timestep is variable)
    _q_next.setdiag(_q_next.diagonal() + 1 / time_params.dt)
    _q_prev.setdiag(_q_prev.diagonal() + 1 / time_params.dt)

    # csc format for efficiency
    _q_next = _q_next.tocsc()
    _q_prev = _q_prev.tocsc()

    # Get LU preconditioner
    preconditioner = get_super_ilu_preconditioner(_q_next)

    # Multiply prev matrix by prev vector
    tmp = _q_prev.dot(fl_model.lhead[time_index - 1].flatten(order="F"))

    # Add the source terms
    sources = (
        fl_model.crank_nicolson * flw_sources.flatten(order="F")
        + (1.0 - fl_model.crank_nicolson) * flw_sources_old.flatten(order="F")
    ) / fl_model.storage_coefficient

    tmp += sources

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(
        _q_next, tmp, x0=None, M=preconditioner, atol=fl_model.tolerance
    )
    fl_model.lhead.append(res.reshape(geometry.ny, geometry.nx).T)

    compute_u_darcy(fl_model, geometry, time_index)

    compute_u_darcy_div(fl_model, geometry, time_index)

    return exit_code
