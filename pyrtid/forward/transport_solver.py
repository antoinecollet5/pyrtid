"""Provide a reactive transport solver."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import lil_array, lil_matrix
from scipy.sparse.linalg import gmres

from pyrtid.utils import harmonic_mean
from pyrtid.utils.operators import get_super_lu_preconditioner
from pyrtid.utils.types import NDArrayFloat

from .models import (
    FlowModel,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)


def make_transport_matrices_diffusion_only(
    geometry: Geometry, tr_model: TransportModel, time_params: TimeParameters
) -> Tuple[lil_matrix, lil_matrix]:
    """
    Make matrices for the transport.

    Note
    ----
    Since the diffusion coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    # X contribution
    if geometry.nx >= 2:
        dmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        dmean[:-1, :] = harmonic_mean(
            tr_model.effective_diffusion[:-1, :], tr_model.effective_diffusion[1:, :]
        )
        dmean = dmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            tr_model.cst_conc_indices,
        )

        tmp = geometry.dy / geometry.dx / geometry.mesh_volume

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        dmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        dmean[:, :-1] = harmonic_mean(
            tr_model.effective_diffusion[:, :-1], tr_model.effective_diffusion[:, 1:]
        )
        dmean = dmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            tr_model.cst_conc_indices,
        )

        tmp = geometry.dx / geometry.dy / geometry.mesh_volume

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

    return q_next, q_prev


def _add_advection_to_transport_matrices(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    conc_sources: NDArrayFloat,
    conc_sources_old: NDArrayFloat,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    crank_adv: float = tr_model.crank_nicolson_advection

    # X contribution
    if geometry.nx >= 2:
        tmp_x = np.zeros((geometry.nx, geometry.ny))
        tmp_x[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index]
        tmp_x_old = np.zeros((geometry.nx, geometry.ny))
        tmp_x_old[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index - 1]

        un_x = tmp_x.flatten(order="F")
        un_x_old = tmp_x_old.flatten(order="F")

        tmp = geometry.dy / geometry.mesh_volume

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_neigh] += (
            crank_adv
            * np.where(normal * un_x <= 0.0, normal * un_x, 0.0)[idc_owner]
            * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_owner]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old <= 0.0, normal * un_x_old, 0.0)[idc_owner]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old > 0.0, normal * un_x_old, 0.0)[idc_owner]
            * tmp
        )  # type: ignore

        # Backward scheme
        normal = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_neigh] += (
            crank_adv
            * np.where(normal * un_x <= 0.0, normal * un_x, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old <= 0.0, normal * un_x_old, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old > 0.0, normal * un_x_old, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        tmp_y = np.zeros((geometry.nx, geometry.ny))
        tmp_y[:, :-1] = fl_model.u_darcy_y[:, 1:-1, time_index]
        tmp_y_old = np.zeros((geometry.nx, geometry.ny))
        tmp_y_old[:, :-1] = fl_model.u_darcy_y[:, 1:-1, time_index - 1]

        un_y = tmp_y.flatten(order="F")
        un_y_old = tmp_y_old.flatten(order="F")

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            tr_model.cst_conc_indices,
        )

        tmp = geometry.dx / geometry.mesh_volume

        q_next[idc_owner, idc_neigh] += (
            crank_adv
            * np.where(normal * un_y <= 0.0, normal * un_y, 0.0)[idc_owner]
            * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_owner]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old <= 0.0, normal * un_y_old, 0.0)[idc_owner]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old > 0.0, normal * un_y_old, 0.0)[idc_owner]
            * tmp
        )  # type: ignore

        # Backward scheme
        normal: float = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_neigh] += (
            crank_adv
            * np.where(normal * un_y <= 0.0, normal * un_y, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old <= 0.0, normal * un_y_old, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old > 0.0, normal * un_y_old, 0.0)[idc_neigh]
            * tmp
        )  # type: ignore

    _apply_transport_sink_term(tr_model, conc_sources, conc_sources_old, q_next, q_prev)

    _apply_divergence_effect(
        fl_model, tr_model, conc_sources, conc_sources_old, q_next, q_prev, time_index
    )

    # Handle boundary conditions
    _add_transport_boundary_conditions(
        geometry, fl_model, tr_model, q_next, q_prev, time_index
    )


def _apply_transport_sink_term(
    tr_model: TransportModel,
    conc_sources: NDArrayFloat,
    conc_sources_old: NDArrayFloat,
    q_next: lil_matrix,
    q_prev: lil_matrix,
) -> None:
    flw = conc_sources.flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    flw_old = conc_sources_old.flatten(order="F")
    _flw_old = np.where(flw_old < 0, flw_old, 0.0)  # keep only negative flowrates
    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * _flw)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * _flw_old
    )


def _apply_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    conc_sources: NDArrayFloat,
    conc_sources_old: NDArrayFloat,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    """
    Take into account the divergence: dcdt+U.grad(c)=L(u)."""

    div = (fl_model.lu_darcy_div[time_index] - conc_sources[:, :]).flatten(order="F")
    div_old = (fl_model.lu_darcy_div[time_index - 1] - conc_sources[:, :]).flatten(
        order="F"
    )

    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * div)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * div_old
    )


def _add_transport_boundary_conditions(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    """Add the boundary conditions to the matrix."""
    # We get the indices of the four borders and we apply a zero-conc gradient.

    if geometry.nx >= 2:
        idc_left, idc_right = get_owner_neigh_indices(
            geometry,
            (slice(0, 1), slice(None)),
            (slice(geometry.nx - 1, geometry.nx), slice(None)),
            np.array([]),
        )
        tmp = geometry.dy / geometry.mesh_volume

        _un = fl_model.u_darcy_x[:-1, :, time_index].ravel("F")[idc_left]
        _un_old = fl_model.u_darcy_x[:-1, :, time_index - 1].ravel("F")[idc_left]
        normal = -1.0

        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
        )  # type: ignore

        _un = fl_model.u_darcy_x[1:, :, time_index].ravel("F")[idc_right]
        _un_old = fl_model.u_darcy_x[1:, :, time_index - 1].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
        )  # type: ignore

    # # Y contribution
    if geometry.ny >= 2:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, 1)),
            (slice(None), slice(geometry.ny - 1, geometry.ny)),
            np.array([]),
        )
        tmp = geometry.dx / geometry.mesh_volume

        _un = fl_model.u_darcy_y[:, :-1, time_index].ravel("F")[idc_left]
        _un_old = fl_model.u_darcy_y[:, :-1, time_index - 1].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
        )  # type: ignore

        _un = fl_model.u_darcy_y[:, 1:, time_index].ravel("F")[idc_right]
        _un_old = fl_model.u_darcy_y[:, 1:, time_index - 1].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
        )  # type: ignore


def solve_transport_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    conc_sources: NDArrayFloat,
    conc_sources_old: NDArrayFloat,
    time_params: TimeParameters,
    time_index: int,
    nfpi: int,
) -> int:
    """
    Compute the transport of the mobile concentrations.

    Parameters
    ----------
    geometry: Geometry
        Geometry of the system.
    fl_model: FlowModel
        The flow model.
    tr_model: TransportModel
        The transport model.
    time_params: TimeParameters
        Time parameters of the system.
    time_index: int
        The iteration, or timestep id.
    nfpi:
        Number of fixed point iterations.
    """

    # The matrix with respect to the diffusion never changes.
    # The matrix with respect to the advection only needs to be updated if the head
    # have changed.
    if nfpi == 1:
        q_next: lil_matrix = tr_model.q_next_diffusion.copy()
        q_prev: lil_matrix = tr_model.q_prev_diffusion.copy()

        # Update q_next and q_prev with the advection term (must be copied)
        # Note that this is required at the first fixed point iteration only,
        # afterwards, only the chemical source term varies.
        _add_advection_to_transport_matrices(
            geometry,
            fl_model,
            tr_model,
            q_next,
            q_prev,
            conc_sources,
            conc_sources_old,
            time_params,
            time_index,
        )
        # Add 1/dt for the left term contribution
        q_next.setdiag(
            q_next.diagonal() + tr_model.porosity.flatten("F") / time_params.dt
        )
        q_prev.setdiag(
            q_prev.diagonal() + tr_model.porosity.flatten("F") / time_params.dt
        )

        tr_model.q_next = q_next
        tr_model.q_prev = q_prev
    else:
        q_next = tr_model.q_next
        q_prev = tr_model.q_prev

    # Multiply prev matrix by prev vector
    tmp = tr_model.q_prev.dot(tr_model.lconc[time_index - 1].flatten(order="F"))

    # Chemical source term
    if tr_model.is_numerical_acceleration and nfpi == 1 and time_index != 1:
        dmdt = tr_model.lgrade[time_index - 1] - tr_model.lgrade[time_index - 2]
    else:
        dmdt = tr_model.lgrade[time_index] - tr_model.lgrade[time_index - 1]

    # The volume is included in the diffusion term
    tmp -= (dmdt * tr_model.porosity / time_params.dt).ravel(order="F")

    # Add the source terms -> depends on the advection (positive flowrates = injection)
    # Crank-nicolson does not apply to source terms
    tmp += conc_sources.ravel("F")

    # Build the LU preconditioning
    preconditioner = get_super_lu_preconditioner(q_next.tocsc())

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(
        q_next.tocsc(), tmp, M=preconditioner, atol=tr_model.tolerance
    )

    # In that regard, we save the intermediate concentrations for the non
    # iterative sequential apprach (adjoint state)
    tr_model.lconc[time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code
