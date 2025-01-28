"""Provide a reactive transport solver."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (
    FlowModel,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.utils import harmonic_mean
from pyrtid.utils.operators import get_super_ilu_preconditioner
from pyrtid.utils.types import NDArrayFloat


def make_transport_matrices(
    geometry: Geometry,
    tr_model: TransportModel,
    fl_model: FlowModel,
    time_index: int,
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the transport.


    Parameters
    ----------
    time_index: int
        The iteration, or timestep id.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    # diffusion + dispersivity
    d = (
        tr_model.effective_diffusion
        + tr_model.dispersivity * fl_model.get_u_darcy_norm_sample(time_index)
    )

    # X contribution
    if geometry.nx >= 2:
        dmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        dmean[:-1, :] = harmonic_mean(d[:-1, :], d[1:, :])
        dmean = dmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp = geometry.gamma_ij_x / geometry.dx / geometry.grid_cell_surface

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        dmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        dmean[:, :-1] = harmonic_mean(d[:, :-1], d[:, 1:])
        dmean = dmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp = geometry.gamma_ij_y / geometry.dy / geometry.grid_cell_surface

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            neigh_indices_to_keep=tr_model.free_conc_nn,
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            neigh_indices_to_keep=tr_model.free_conc_nn,
        )
        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

    return q_next, q_prev


def _add_advection_to_transport_matrices(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
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

        tmp = geometry.gamma_ij_x / geometry.grid_cell_surface

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
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
            owner_indices_to_keep=tr_model.free_conc_nn,
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
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp = geometry.gamma_ij_y / geometry.grid_cell_surface

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
            owner_indices_to_keep=tr_model.free_conc_nn,
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

    _apply_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_transport_boundary_conditions(
        geometry, fl_model, tr_model, q_next, q_prev, time_index
    )


def _apply_transport_sink_term(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    flw = fl_model.lunitflow[time_index].flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    flw_old = fl_model.lunitflow[time_index - 1].flatten(order="F")
    _flw_old = np.where(flw_old < 0, flw_old, 0.0)  # keep only negative flowrates
    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * _flw)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * _flw_old
    )


def _apply_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    """
    Take into account the divergence: dcdt+U.grad(c)=L(u)."""

    div = (fl_model.lu_darcy_div[time_index] - fl_model.lunitflow[time_index]).flatten(
        order="F"
    )
    div_old = (
        fl_model.lu_darcy_div[time_index - 1] - fl_model.lunitflow[time_index - 1]
    ).flatten(order="F")

    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * div)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * div_old
    )


def _add_transport_boundary_conditions(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    """Add the boundary conditions to the matrix."""
    # We get the indices of the four borders and we apply a zero gradient.

    if geometry.nx > 1:
        idc_left_border, idc_right_border = get_owner_neigh_indices(
            geometry,
            (slice(0, 1), slice(None)),
            (slice(geometry.nx - 1, geometry.nx), slice(None)),
        )
        tmp = geometry.gamma_ij_x / geometry.grid_cell_surface

        # left border
        _un = fl_model.u_darcy_x[:-1, :, time_index].ravel("F")[idc_left_border]
        _un_old = fl_model.u_darcy_x[:-1, :, time_index - 1].ravel("F")[idc_left_border]
        normal = -1.0

        q_next[idc_left_border, idc_left_border] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left_border, idc_left_border] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
        )  # type: ignore

        # right border
        _un = fl_model.u_darcy_x[1:, :, time_index].ravel("F")[idc_right_border]
        _un_old = fl_model.u_darcy_x[1:, :, time_index - 1].ravel("F")[idc_right_border]
        normal = 1.0
        q_next[idc_right_border, idc_right_border] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right_border, idc_right_border] -= (
            (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
        )  # type: ignore

    # # Y contribution
    if geometry.ny > 1:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, 1)),
            (slice(None), slice(geometry.ny - 1, geometry.ny)),
            np.array([]),
        )
        tmp = geometry.gamma_ij_y / geometry.grid_cell_surface

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
        q_next, q_prev = make_transport_matrices(
            geometry, tr_model, fl_model, time_index
        )

        # TODO: remove that
        tr_model.q_next_diffusion = q_next.copy()
        tr_model.q_prev_diffusion = q_prev.copy()

        # Update q_next and q_prev with the advection term (must be copied)
        # Note that this is required at the first fixed point iteration only,
        # afterwards, only the chemical source term varies.
        _add_advection_to_transport_matrices(
            geometry,
            fl_model,
            tr_model,
            q_next,
            q_prev,
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

        # TODO: make optional
        tr_model.l_q_next.append(q_next)
        tr_model.l_q_prev.append(q_prev)
    else:
        q_next = tr_model.q_next
        q_prev = tr_model.q_prev

    # Multiply prev matrix by prev vector
    tmp = tr_model.q_prev.dot(
        tr_model.lmob[time_index - 1].reshape(tr_model.n_sp, -1, order="F").T
    ).T

    # Chemical source term
    if tr_model.is_num_acc_for_timestep and nfpi == 1 and time_index != 1:
        dmdt = tr_model.limmob[time_index - 1] - tr_model.limmob[time_index - 2]
        # avoid negative values
        if np.any(tr_model.lmob[time_index - 1] - dmdt < 0):
            dmdt = tr_model.limmob[time_index] - tr_model.limmob[time_index - 1]
    else:
        dmdt = tr_model.limmob[time_index] - tr_model.limmob[time_index - 1]

    # The volume is included in the diffusion term
    tmp -= (
        dmdt.reshape(tr_model.n_sp, -1, order="F")
        * tr_model.porosity.ravel("F")
        / time_params.dt
    )

    # Add the source terms -> depends on the advection (positive flowrates = injection)
    tmp[:, :] += tr_model.crank_nicolson_advection * conc_sources.reshape(
        2, -1, order="F"
    ) + (1.0 - tr_model.crank_nicolson_advection) * conc_sources_old.reshape(
        2, -1, order="F"
    )

    # Build the LU preconditioning
    super_ilu, preconditioner = get_super_ilu_preconditioner(
        q_next.tocsc(), drop_tol=1e-10, fill_factor=100
    )
    if super_ilu is None:
        warnings.warn(f"SuperILU: q_next is singular in transport at it={time_index}!")

    # Solve Ax = b with A sparse using LU preconditioner
    for sp in range(tmp.shape[0]):
        tmp[sp, :], exit_code = gmres(
            q_next.tocsc(),
            tmp[sp, :],
            x0=super_ilu.solve(tmp[sp, :]) if super_ilu is not None else None,
            M=preconditioner,
            rtol=tr_model.rtol,
        )

    tr_model.lmob[time_index] = tmp.reshape(
        tr_model.n_sp, geometry.nx, geometry.ny, order="F"
    )

    return exit_code
