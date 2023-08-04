"""Provide an adjoint solver for the transport operator."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (
    FlowModel,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.inverse.adjoint.amodels import AdjointTransportModel
from pyrtid.utils import harmonic_mean
from pyrtid.utils.operators import get_super_lu_preconditioner
from pyrtid.utils.types import NDArrayFloat


def make_transient_adj_transport_matrices(
    geometry: Geometry, tr_model: TransportModel, time_params: TimeParameters
) -> Tuple[csc_matrix, csc_matrix]:
    """
    Make matrices for the transient transport.

    Note
    ----
    Since the diffusion coefficient and porosity does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_matrix((dim, dim), dtype=np.float64)
    q_next = lil_matrix((dim, dim), dtype=np.float64)

    # X contribution
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

    q_next[idc_owner, idc_neigh] -= (
        tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dx**2
    )
    q_next[idc_owner, idc_owner] += (
        tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dx**2
    )
    q_prev[idc_owner, idc_neigh] += (
        (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] / geometry.dx**2
    )
    q_prev[idc_owner, idc_owner] -= (
        (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] / geometry.dx**2
    )

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(1, geometry.nx), slice(None)),
        (slice(0, geometry.nx - 1), slice(None)),
        tr_model.cst_conc_indices,
    )

    q_next[idc_owner, idc_neigh] -= (
        tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dx**2
    )
    q_next[idc_owner, idc_owner] += (
        tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dx**2
    )
    q_prev[idc_owner, idc_neigh] += (
        (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] / geometry.dx**2
    )
    q_prev[idc_owner, idc_owner] -= (
        (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] / geometry.dx**2
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

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dy**2
        )
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] / geometry.dy**2
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_owner]
            / geometry.dy**2
        )
        q_prev[idc_owner, idc_owner] -= (
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

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dy**2
        )
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] / geometry.dy**2
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_neigh]
            / geometry.dy**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion)
            * dmean[idc_neigh]
            / geometry.dy**2
        )

    # Add 1/dt for the left term contribution
    q_next.setdiag(q_next.diagonal() + tr_model.porosity.flatten("F") / time_params.dt)
    q_prev.setdiag(q_prev.diagonal() + tr_model.porosity.flatten("F") / time_params.dt)

    return q_next.tocsc(), q_prev.tocsc()


def _add_advection_to_adj_transport_matrices(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
) -> None:
    crank_adv = tr_model.crank_nicolson_advection
    q_next = a_tr_model.q_next_diffusion.copy().tolil()
    q_prev = a_tr_model.q_prev_diffusion.copy().tolil()

    # X contribution
    tmp = np.zeros((geometry.nx, geometry.ny))
    tmp[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index]
    un_x = tmp.flatten(order="F")

    # Forward scheme:
    normal = 1.0
    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(0, geometry.nx - 1), slice(None)),
        (slice(1, geometry.nx), slice(None)),
        tr_model.cst_conc_indices,
    )

    tmp_un_pos = np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_owner]

    q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos / geometry.dx
    q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos / geometry.dx
    q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos / geometry.dx
    q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos / geometry.dx

    # Backward scheme
    normal = -1.0
    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(1, geometry.nx), slice(None)),
        (slice(0, geometry.nx - 1), slice(None)),
        tr_model.cst_conc_indices,
    )

    tmp_un_pos = np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_neigh]

    q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos / geometry.dx
    q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos / geometry.dx
    q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos / geometry.dx
    q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos / geometry.dx

    # Y contribution
    if geometry.ny >= 2:
        tmp = np.zeros((geometry.nx, geometry.ny))
        tmp[:, :-1] = fl_model.u_darcy_y[:, 1:-1, time_index]
        un_y = tmp.flatten(order="F")

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            tr_model.cst_conc_indices,
        )

        tmp_un_pos = np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_owner]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos / geometry.dy
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos / geometry.dy
        q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos / geometry.dy
        q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos / geometry.dy

        # Backward scheme
        normal = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            tr_model.cst_conc_indices,
        )

        tmp_un_pos = np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_neigh]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos / geometry.dy
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos / geometry.dy
        q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos / geometry.dy
        q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos / geometry.dy

    _apply_adj_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_adj_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_adj_transport_boundary_conditions(
        geometry, fl_model, tr_model, q_next, q_prev, time_index
    )

    a_tr_model.q_next = q_next.tocsc()
    a_tr_model.q_prev = q_prev.tocsc()


def _apply_adj_transport_sink_term(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    flw = fl_model.sources[:, :, time_index].flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * _flw)
    q_prev.setdiag(q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * _flw)


def _apply_adj_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    """Take into account the divergence: dcdt+U.grad(c)=L(u)."""
    src = fl_model.sources[:, :, time_index]
    try:
        src_old = fl_model.sources[:, :, time_index + 1]
    except IndexError:
        src_old = 0

    div = (fl_model.u_darcy_div[:, :, time_index] - src).flatten(order="F")
    div_old = (fl_model.u_darcy_div[:, :, time_index] - src_old).flatten(order="F")

    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * div)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * div_old
    )


def _add_adj_transport_boundary_conditions(
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

    _un = fl_model.u_darcy_x[:-1, :, time_index].ravel("F")[idc_left]
    # _un_old = fl_model.u_darcy_x[:-1, :, time_index + 1].ravel("F")[idc_left]
    normal = -1.0
    q_next[idc_left, idc_left] += (
        tr_model.crank_nicolson_advection * _un / geometry.dx * normal
    )
    q_prev[idc_left, idc_left] -= (
        (1 - tr_model.crank_nicolson_advection) * _un / geometry.dx * normal
    )

    _un = fl_model.u_darcy_x[1:, :, time_index].ravel("F")[idc_right]
    # _un_old = fl_model.u_darcy_x[1:, :, time_index + 1].ravel("F")[idc_right]
    normal = 1.0
    q_next[idc_right, idc_right] += (
        tr_model.crank_nicolson_advection * _un / geometry.dx * normal
    )
    q_prev[idc_right, idc_right] -= (
        (1 - tr_model.crank_nicolson_advection) * _un / geometry.dx * normal
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
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un / geometry.dy * normal
        )
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un / geometry.dy * normal
        )

        _un = fl_model.u_darcy_y[:, 1:, time_index].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un / geometry.dy * normal
        )
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un / geometry.dy * normal
        )


def solve_adj_transport_transient_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    dt: float,
    time_index: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """
    # Update q_next and q_prev with the advection term (must be copied)
    _add_advection_to_adj_transport_matrices(
        geometry, fl_model, tr_model, a_tr_model, time_index
    )

    # Get the previous vector
    prev_vector = a_tr_model.a_conc[:, :, time_index].ravel("F")

    # Multiply prev matrix by prev vector
    tmp = a_tr_model.q_prev.dot(prev_vector)

    # Add the source terms -> from the previous timestep
    tmp += (a_tr_model.a_sources[:, :, time_index]).ravel("F") / geometry.mesh_area

    # Build the LU preconditioning
    preconditioner = get_super_lu_preconditioner(a_tr_model.q_next)

    # if time_index == 0:
    # print(a_tr_model.q_next)
    # print(a_tr_model.q_prev)

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(a_tr_model.q_next, tmp, M=preconditioner, atol=1e-15)
    # Note: we go backward in time, so time_index -1...
    a_tr_model.a_conc[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code
