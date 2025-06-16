"""Provide a reactive transport solver."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (
    FlowModel,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.utils import NDArrayFloat, RectilinearGrid, harmonic_mean
from pyrtid.utils.operators import get_super_ilu_preconditioner


def fill_trmat_for_axis(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    disp: NDArrayFloat,
    time_index: int,
    axis: int,
) -> None:
    if axis == 0:
        u_darcy = fl_model.u_darcy_x
    elif axis == 1:
        u_darcy = fl_model.u_darcy_y
    elif axis == 2:
        u_darcy = fl_model.u_darcy_z
    else:
        raise ValueError()

    crank_adv: float = tr_model.crank_nicolson_advection
    crank_diff: float = tr_model.crank_nicolson_diffusion
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    dmean: NDArrayFloat = np.zeros(grid.shape, dtype=np.float64)
    dmean[fwd_slicer] = harmonic_mean(disp[fwd_slicer], disp[bwd_slicer])
    dmean = dmean.flatten(order="F")

    tmp_diff: float = grid.gamma_ij(axis) / grid.pipj(axis) / grid.grid_cell_volume

    tmp_un = np.zeros(grid.shape)
    tmp_un[fwd_slicer] = u_darcy[*bwd_slicer, time_index]
    tmp_un_old = np.zeros(grid.shape)
    tmp_un_old[fwd_slicer] = u_darcy[*bwd_slicer, time_index - 1]

    un = tmp_un.flatten(order="F")
    un_old = tmp_un_old.flatten(order="F")

    tmp_adv = grid.gamma_ij(axis) / grid.grid_cell_volume

    # Forward scheme:
    normal = 1.0
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=tr_model.free_conc_nn,
    )

    q_next[idc_owner, idc_owner] += crank_diff * dmean[idc_owner] * tmp_diff + (
        crank_adv * np.where(normal * un > 0.0, normal * un, 0.0)[idc_owner] * tmp_adv
    )  # type: ignore
    q_next[idc_owner, idc_neigh] += -(crank_diff * dmean[idc_owner] * tmp_diff) + (
        crank_adv * np.where(normal * un <= 0.0, normal * un, 0.0)[idc_owner] * tmp_adv
    )  # type: ignore

    q_prev[idc_owner, idc_owner] -= (1.0 - crank_diff) * dmean[idc_owner] * tmp_diff + (
        (1 - crank_adv)
        * np.where(normal * un_old > 0.0, normal * un_old, 0.0)[idc_owner]
        * tmp_adv
    )  # type: ignore
    q_prev[idc_owner, idc_neigh] -= -(
        (1.0 - crank_diff) * dmean[idc_owner] * tmp_diff
    ) + (
        (1 - crank_adv)
        * np.where(normal * un_old <= 0.0, normal * un_old, 0.0)[idc_owner]
        * tmp_adv
    )  # type: ignore

    # Backward scheme
    normal = -1.0
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=tr_model.free_conc_nn,
    )

    q_next[idc_owner, idc_owner] += crank_diff * dmean[idc_neigh] * tmp_diff + (
        crank_adv * np.where(normal * un > 0.0, normal * un, 0.0)[idc_neigh] * tmp_adv
    )  # type: ignore
    q_next[idc_owner, idc_neigh] += -(crank_diff * dmean[idc_neigh] * tmp_diff) + (
        crank_adv * np.where(normal * un <= 0.0, normal * un, 0.0)[idc_neigh] * tmp_adv
    )  # type: ignore
    q_prev[idc_owner, idc_owner] -= (1.0 - crank_diff) * dmean[idc_neigh] * tmp_diff + (
        (1.0 - crank_adv)
        * np.where(normal * un_old > 0.0, normal * un_old, 0.0)[idc_neigh]
        * tmp_adv
    )  # type: ignore

    q_prev[idc_owner, idc_neigh] -= -(1.0 - crank_diff) * dmean[
        idc_neigh
    ] * tmp_diff + (
        (1.0 - crank_adv)
        * np.where(normal * un_old <= 0.0, normal * un_old, 0.0)[idc_neigh]
        * tmp_adv
    )  # type: ignore


def make_transport_matrices(
    grid: RectilinearGrid,
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

    dim = grid.n_grid_cells
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    # diffusion + dispersivity
    disp = (
        tr_model.effective_diffusion
        + tr_model.dispersivity * fl_model.get_u_darcy_norm_sample(time_index)
    )

    for n, axis in zip(grid.shape, (0, 1, 2)):
        if n >= 2:
            fill_trmat_for_axis(
                grid, fl_model, tr_model, q_next, q_prev, disp, time_index, axis
            )

    _apply_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_transport_boundary_conditions(
        grid, fl_model, tr_model, q_next, q_prev, time_index
    )

    return q_next, q_prev


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


def _add_transport_boundary_conditions_for_axis(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
    axis: int,
) -> None:
    if axis == 0:
        u_darcy = fl_model.u_darcy_x
        bd1_slicer = (slice(0, 1), slice(None), slice(None))
        bd2_slicer = (slice(grid.nx - 1, grid.nx), slice(None), slice(None))
    elif axis == 1:
        u_darcy = fl_model.u_darcy_y
        bd1_slicer = (slice(None), slice(0, 1), slice(None))
        bd2_slicer = (slice(None), slice(grid.ny - 1, grid.ny), slice(None))
    elif axis == 2:
        u_darcy = fl_model.u_darcy_z
        bd1_slicer = (slice(None), slice(None), slice(0, 1))
        bd2_slicer = (slice(None), slice(None), slice(grid.nz - 1, grid.nz))
    else:
        raise ValueError()

    fwd_slicer = grid.get_slicer_forward(axis, shift=1)
    bwd_slicer = grid.get_slicer_backward(axis, shift=1)

    idc_left_border, idc_right_border = get_owner_neigh_indices(
        grid,
        bd1_slicer,
        bd2_slicer,
    )
    tmp = grid.gamma_ij(axis) / grid.grid_cell_volume

    # left border
    _un = u_darcy[*fwd_slicer, time_index].ravel("F")[idc_left_border]
    _un_old = u_darcy[*fwd_slicer, time_index - 1].ravel("F")[idc_left_border]
    normal = -1.0
    q_next[idc_left_border, idc_left_border] += (
        tr_model.crank_nicolson_advection * _un * tmp * normal
    )  # type: ignore
    q_prev[idc_left_border, idc_left_border] -= (
        (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
    )  # type: ignore

    # right border
    _un = u_darcy[*bwd_slicer, time_index].ravel("F")[idc_right_border]
    _un_old = u_darcy[*bwd_slicer, time_index - 1].ravel("F")[idc_right_border]
    normal = 1.0
    q_next[idc_right_border, idc_right_border] += (
        tr_model.crank_nicolson_advection * _un * tmp * normal
    )  # type: ignore
    q_prev[idc_right_border, idc_right_border] -= (
        (1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal
    )  # type: ignore


def _add_transport_boundary_conditions(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    """Add the boundary conditions to the matrix."""
    # We get the indices of the four borders and we apply a zero gradient.

    for n, axis in zip(grid.shape, (0, 1, 2)):
        if n >= 2:
            _add_transport_boundary_conditions_for_axis(
                grid, fl_model, tr_model, q_next, q_prev, time_index, axis
            )


def solve_transport_semi_implicit(
    grid: RectilinearGrid,
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
    grid: RectilinearGrid
        RectilinearGrid of the system.
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
        q_next, q_prev = make_transport_matrices(grid, tr_model, fl_model, time_index)

        # Add 1/dt for the left term contribution
        q_next.setdiag(
            q_next.diagonal() + tr_model.porosity.flatten("F") / time_params.dt
        )
        q_prev.setdiag(
            q_prev.diagonal() + tr_model.porosity.flatten("F") / time_params.dt
        )

        tr_model.q_next = q_next
        tr_model.q_prev = q_prev

        if tr_model.is_save_spmats:
            tr_model.l_q_next.append(q_next)
            tr_model.l_q_prev.append(q_prev)

        # Build the LU preconditioning -> to do only once.
        super_ilu, preconditioner = get_super_ilu_preconditioner(
            q_next.tocsc(), drop_tol=1e-10, fill_factor=100
        )
        tr_model.super_ilu = super_ilu
        tr_model.preconditioner = preconditioner

        if super_ilu is None:
            warnings.warn(
                f"SuperILU: q_next is singular in transport at it={time_index}!"
            )
    else:
        q_next = tr_model.q_next
        q_prev = tr_model.q_prev
        super_ilu = tr_model.super_ilu
        preconditioner = tr_model.preconditioner

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

    # Solve Ax = b with A sparse using LU preconditioner
    for sp in range(tmp.shape[0]):
        tmp[sp, :], exit_code = gmres(
            q_next.tocsc(),
            tmp[sp, :],
            x0=super_ilu.solve(tmp[sp, :]) if super_ilu is not None else None,
            M=preconditioner,
            rtol=tr_model.rtol,
        )

    tr_model.lmob[time_index] = tmp.reshape(tr_model.n_sp, *grid.shape, order="F")

    return exit_code
