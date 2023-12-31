"""Provide a reactive transport solver."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres

from pyrtid.utils import get_super_lu_preconditioner, harmonic_mean
from pyrtid.utils.types import NDArrayFloat

from .models import FlowModel, Geometry, TimeParameters, get_owner_neigh_indices


def make_stationary_flow_matrices(
    geometry: Geometry, fl_model: FlowModel
) -> lil_matrix:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_next = lil_matrix((dim, dim), dtype=np.float64)

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
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            kmean[idc_owner] / geometry.dx**2 / geometry.dz
        )
        q_next[idc_owner, idc_owner] += (
            kmean[idc_owner] / geometry.dx**2 / geometry.dz
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            kmean[idc_neigh] / geometry.dx**2 / geometry.dz
        )
        q_next[idc_owner, idc_owner] += (
            kmean[idc_neigh] / geometry.dx**2 / geometry.dz
        )

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
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            kmean[idc_owner] / geometry.dy**2 / geometry.dz
        )
        q_next[idc_owner, idc_owner] += (
            kmean[idc_owner] / geometry.dy**2 / geometry.dz
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            kmean[idc_neigh] / geometry.dy**2 / geometry.dz
        )
        q_next[idc_owner, idc_owner] += (
            kmean[idc_neigh] / geometry.dy**2 / geometry.dz
        )

    # Take constant head into account
    q_next[fl_model.cst_head_nn, fl_model.cst_head_nn] = 1.0

    return q_next


def make_transient_flow_matrices(
    geometry: Geometry, fl_model: FlowModel, time_params: TimeParameters
) -> Tuple[lil_matrix, lil_matrix]:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_matrix((dim, dim), dtype=np.float64)
    q_next = lil_matrix((dim, dim), dtype=np.float64)

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
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            / geometry.dx**2
            / geometry.dz
            / fl_model.storage_coefficient
        )

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
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            / geometry.dy**2
            / geometry.dz
            / fl_model.storage_coefficient
        )

    return q_next, q_prev


def solve_flow_stationary(
    geometry: Geometry,
    fl_model: FlowModel,
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
    preconditioner = get_super_lu_preconditioner(fl_model.q_next.tocsc())

    # Add the source terms
    tmp += fl_model.sources[:, :, time_index].flatten(order="F")

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(
        fl_model.q_next.tocsc(), tmp, M=preconditioner, atol=fl_model.tolerance
    )
    fl_model.lhead.append(res.reshape(geometry.ny, geometry.nx).T)

    update_u_darcy(fl_model, geometry, time_index)

    update_u_darcy_div(fl_model, geometry, time_index)

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
    head = fl_model.lhead[time_index]

    kmean = harmonic_mean(fl_model.permeability[:-1, :], fl_model.permeability[1:, :])

    return -kmean * (head[1:, :] - head[:-1, :]) / geometry.dx


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
    head = fl_model.lhead[time_index]

    # X axis contribution
    kmean = harmonic_mean(fl_model.permeability[:, :-1], fl_model.permeability[:, 1:])

    return -kmean * (head[:, 1:] - head[:, :-1]) / geometry.dy


def update_u_darcy(fl_model: FlowModel, geometry: Geometry, time_index: int) -> None:
    """Update the darcy velocities at the node boundaries."""
    fl_model.u_darcy_x[1:-1, :, time_index] = find_ux_boundary(
        fl_model, geometry, time_index
    )
    fl_model.u_darcy_y[:, 1:-1, time_index] = find_uy_boundary(
        fl_model, geometry, time_index
    )
    # Handle constant head
    # update_u_darcy_cst_head_nodes(fl_model, geometry, time_index)


def update_u_darcy_cst_head_nodes(
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
    flow = np.zeros((geometry.nx, geometry.ny))
    _flow = np.zeros((geometry.nx, geometry.ny))
    flow[:, :] += fl_model.u_darcy_x[:-1, :, time_index] * geometry.dx
    flow[:, :] -= fl_model.u_darcy_x[1:, :, time_index] * geometry.dx
    flow[:, :] += fl_model.u_darcy_y[:, :-1, time_index] * geometry.dy
    flow[:, :] -= fl_model.u_darcy_y[:, 1:, time_index] * geometry.dy

    # Divide by the cell volume (surface here in 2D)
    flow /= geometry.mesh_area

    # Trick: Set the flow to zero where the head is not constant
    _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]] = flow[
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ]
    _flow[fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]] = flow[
        fl_model.cst_head_indices[0], fl_model.cst_head_indices[1]
    ]

    # Adjust the darcy velocities for constant node at borders
    fl_model.u_darcy_x[0, :, time_index] = -_flow[0, :] * geometry.dy
    fl_model.u_darcy_x[-1, :, time_index] = _flow[-1, :] * geometry.dy
    fl_model.u_darcy_y[:, 0, time_index] = -_flow[:, 0] * geometry.dx
    fl_model.u_darcy_y[:, -1, time_index] = _flow[:, -1] * geometry.dx

    # fl_model.

    # fl_model.u_darcy_x[fl_model.cst_head_indices, time_index] = -flow / 2

    # 2) For constant head, define the darcy velocities as -


def update_u_darcy_div(
    fl_model: FlowModel, geometry: Geometry, time_index: int
) -> None:
    """Update the darcy velocities divergence (at the node centers)."""

    # Reset to zero
    fl_model.u_darcy_div[:, :, time_index] = 0.0

    # x contribution -> multiply by the frontier (dy and not dx)
    fl_model.u_darcy_div[:, :, time_index] -= (
        fl_model.u_darcy_x[:-1, :, time_index] * geometry.dy
    )
    fl_model.u_darcy_div[:, :, time_index] += (
        fl_model.u_darcy_x[1:, :, time_index] * geometry.dy
    )

    # y contribution  -> multiply by the frontier (dx and not dy)
    fl_model.u_darcy_div[:, :, time_index] -= (
        fl_model.u_darcy_y[:, :-1, time_index] * geometry.dx
    )
    fl_model.u_darcy_div[:, :, time_index] += (
        fl_model.u_darcy_y[:, 1:, time_index] * geometry.dx
    )

    # Take the surface into account
    fl_model.u_darcy_div[:, :, time_index] /= geometry.mesh_area

    # Constant head handling - null divergence
    cst_idx = fl_model.cst_head_indices
    fl_model.u_darcy_div[cst_idx[0], cst_idx[1], time_index] = 0


def solve_flow_transient_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
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
    preconditioner = get_super_lu_preconditioner(_q_next)

    # Multiply prev matrix by prev vector
    tmp = _q_prev.dot(fl_model.lhead[time_index - 1].flatten(order="F"))

    # Add the source terms
    sources = (
        fl_model.crank_nicolson * fl_model.sources[:, :, time_index].flatten(order="F")
        + (1.0 - fl_model.crank_nicolson)
        * fl_model.sources[:, :, time_index - 1].flatten(order="F")
    ) / fl_model.storage_coefficient

    tmp += sources

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(
        _q_next, tmp, x0=None, M=preconditioner, atol=fl_model.tolerance
    )
    fl_model.lhead.append(res.reshape(geometry.ny, geometry.nx).T)

    update_u_darcy(fl_model, geometry, time_index)

    update_u_darcy_div(fl_model, geometry, time_index)

    return exit_code
