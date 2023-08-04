"""Provide a reactive transport solver."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
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
) -> Tuple[csc_matrix, csc_matrix]:
    """
    Make matrices for the transport.

    Note
    ----
    Since the diffusion coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_matrix((dim, dim), dtype=np.float64)
    q_next = lil_matrix((dim, dim), dtype=np.float64)

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

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dx**2
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dx**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_owner]
            / geometry.dx**2
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_owner]
            / geometry.dx**2
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dx**2
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dx**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_neigh]
            / geometry.dx**2
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_neigh]
            / geometry.dx**2
        )

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

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dy**2
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dy**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_owner]
            / geometry.dy**2
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_owner]
            / geometry.dy**2
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dy**2
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dy**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_neigh]
            / geometry.dy**2
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_neigh]
            / geometry.dy**2
        )

    # Add 1/dt for the left term contribution
    q_next.setdiag(q_next.diagonal() + tr_model.porosity.flatten("F") / time_params.dt)
    q_prev.setdiag(q_prev.diagonal() + tr_model.porosity.flatten("F") / time_params.dt)

    return q_next.tocsc(), q_prev.tocsc()


def _add_advection_to_transport_matrices(
    geometry: Geometry, fl_model: FlowModel, tr_model: TransportModel, time_index: int
) -> None:
    crank_adv = tr_model.crank_nicolson_advection
    q_next = tr_model.q_next_diffusion.copy().tolil()
    q_prev = tr_model.q_prev_diffusion.copy().tolil()

    # X contribution
    if geometry.nx >= 2:
        tmp_x = np.zeros((geometry.nx, geometry.ny))
        tmp_x[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index]
        tmp_x_old = np.zeros((geometry.nx, geometry.ny))
        tmp_x_old[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index - 1]

        un_x = tmp_x.flatten(order="F")
        un_x_old = tmp_x_old.flatten(order="F")

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
            / geometry.dx
        )
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_owner]
            / geometry.dx
        )
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old <= 0.0, normal * un_x_old, 0.0)[idc_owner]
            / geometry.dx
        )
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old > 0.0, normal * un_x_old, 0.0)[idc_owner]
            / geometry.dx
        )

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
            / geometry.dx
        )
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_neigh]
            / geometry.dx
        )
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old <= 0.0, normal * un_x_old, 0.0)[idc_neigh]
            / geometry.dx
        )
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_x_old > 0.0, normal * un_x_old, 0.0)[idc_neigh]
            / geometry.dx
        )

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

        q_next[idc_owner, idc_neigh] += (
            crank_adv
            * np.where(normal * un_y <= 0.0, normal * un_y, 0.0)[idc_owner]
            / geometry.dy
        )
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_owner]
            / geometry.dy
        )
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old <= 0.0, normal * un_y_old, 0.0)[idc_owner]
            / geometry.dy
        )
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old > 0.0, normal * un_y_old, 0.0)[idc_owner]
            / geometry.dy
        )

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
            / geometry.dy
        )
        q_next[idc_owner, idc_owner] += (
            crank_adv
            * np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_neigh]
            / geometry.dy
        )
        q_prev[idc_owner, idc_neigh] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old <= 0.0, normal * un_y_old, 0.0)[idc_neigh]
            / geometry.dy
        )
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv)
            * np.where(normal * un_y_old > 0.0, normal * un_y_old, 0.0)[idc_neigh]
            / geometry.dy
        )

    _apply_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_transport_boundary_conditions(
        geometry, fl_model, tr_model, q_next, q_prev, time_index
    )

    tr_model.q_next = q_next.tocsc()
    tr_model.q_prev = q_prev.tocsc()


def _get_transport_source_term(
    tr_model: TransportModel, time_index: int
) -> NDArrayFloat:
    return tr_model.sources[:, :, time_index].flatten(order="F")


def _apply_transport_sink_term(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    flw = fl_model.sources[:, :, time_index].flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    flw_old = fl_model.sources[:, :, time_index - 1].flatten(order="F")
    _flw_old = np.where(flw_old < 0, flw_old, 0.0)  # keep only negative flowrates
    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * _flw)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * _flw_old
    )


def _apply_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    """
    Take into account the divergence: dcdt+U.grad(c)=L(u)."""

    div = (
        fl_model.u_darcy_div[:, :, time_index] - fl_model.sources[:, :, time_index]
    ).flatten(order="F")
    div_old = (
        fl_model.u_darcy_div[:, :, time_index - 1] - fl_model.sources[:, :, time_index]
    ).flatten(order="F")

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
    idc_left, idc_right = get_owner_neigh_indices(
        geometry,
        (slice(0, 1), slice(None)),
        (slice(geometry.nx - 1, geometry.nx), slice(None)),
        np.array([]),
    )

    if geometry.nx >= 2:
        _un = fl_model.u_darcy_x[:-1, :, time_index].ravel("F")[idc_left]
        _un_old = fl_model.u_darcy_x[:-1, :, time_index - 1].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un / geometry.dx * normal
        )
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old / geometry.dx * normal
        )

        _un = fl_model.u_darcy_x[1:, :, time_index].ravel("F")[idc_right]
        _un_old = fl_model.u_darcy_x[1:, :, time_index - 1].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un / geometry.dx * normal
        )
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old / geometry.dx * normal
        )

    # # Y contribution
    if geometry.ny >= 2:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, 1)),
            (slice(None), slice(geometry.ny - 1, geometry.ny)),
            np.array([]),
        )

        _un = fl_model.u_darcy_y[:, :-1, time_index].ravel("F")[idc_left]
        _un_old = fl_model.u_darcy_y[:, :-1, time_index - 1].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un / geometry.dy * normal
        )
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old / geometry.dy * normal
        )

        _un = fl_model.u_darcy_y[:, 1:, time_index].ravel("F")[idc_right]
        _un_old = fl_model.u_darcy_y[:, 1:, time_index - 1].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un / geometry.dy * normal
        )
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old / geometry.dy * normal
        )


def solve_transport_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """Compute the conc and grade fields by solving the flow problem."""

    # Update q_next and q_prev with the advection term (must be copied)
    _add_advection_to_transport_matrices(geometry, fl_model, tr_model, time_index)

    # Multiply prev matrix by prev vector
    tmp = tr_model.q_prev.dot(tr_model.conc[:, :, time_index - 1].flatten(order="F"))

    # Add the source terms -> depends on the advection (positive flowrates = injection)
    # Crank-nicolson does not apply to source terms
    tmp += _get_transport_source_term(tr_model, time_index)

    # Build the LU preconditioning
    preconditioner = get_super_lu_preconditioner(tr_model.q_next)

    # Use the last state as initial guess for the solver.
    # Also takes the sources/sinks into account
    if tr_model.is_numerical_acceleration:
        x0 = (
            tr_model.conc[:, :, time_index - 1].flatten(order="F")
            + _get_transport_source_term(tr_model, time_index) * time_params.dt
        )
    else:
        x0 = None

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(
        tr_model.q_next, tmp, x0=x0, M=preconditioner, atol=tr_model.tolerance
    )

    # In that regard, we save the intermediate concentrations for the non
    # iterative sequential apprach (adjoint state)
    tr_model.conc_post_tr[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code
