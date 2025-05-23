"""Provide an adjoint solver for the transport operator."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (
    TDS_LINEAR_COEFFICIENT,
    WATER_DENSITY,
    FlowModel,
    GeochemicalParameters,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.forward.solver import get_max_coupling_error
from pyrtid.inverse.asm.amodels import AdjointTransportModel
from pyrtid.utils import (
    NDArrayFloat,
    RectilinearGrid,
    assert_allclose_sparse,
    get_super_ilu_preconditioner,
    harmonic_mean,
)


def get_adjoint_max_coupling_error(
    atr_model: AdjointTransportModel, time_index: int
) -> float:
    r"""
    Return the maximum adjoint chemistry-transport coupling error.

    The fixed point iteration convergence criteria reads:

    .. math::
        \text{max} \left\lVert 1 - \dfrac{\lambda_{c}^{n, k+1}}
        {\lambda_{c}^{n, k}} \right\rVert  < \epsilon

    with $k$ the number of fixed point iterations.

    This error is evaluated from the mobile adjoint concentrations.
    """
    return get_max_coupling_error(
        atr_model.a_mob[:, :, :, time_index], atr_model.a_mob_prev
    )


def make_transient_adj_transport_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    time_params: TimeParameters,
    time_index: int,
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the transient transport.

    Note
    ----
    Since the diffusion coefficient and porosity does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = grid.nx * grid.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    # diffusion + dispersivity
    d = (
        tr_model.effective_diffusion
        + tr_model.dispersivity * fl_model.get_u_darcy_norm_sample(time_index)
    )

    if time_index + 1 != time_params.nt:
        # no q_prev for n = N_ts
        d_old = (
            tr_model.effective_diffusion
            + tr_model.dispersivity * fl_model.get_u_darcy_norm_sample(time_index + 1)
        )
    else:
        d_old = np.zeros_like(d)

    # X contribution
    if grid.nx >= 2:
        dmean: NDArrayFloat = np.zeros((grid.nx, grid.ny), dtype=np.float64)
        dmean[:-1, :] = harmonic_mean(d[:-1, :], d[1:, :])
        dmean = dmean.flatten(order="F")

        if time_index + 1 != time_params.nt:
            dmean_old: NDArrayFloat = np.zeros((grid.nx, grid.ny), dtype=np.float64)
            dmean_old[:-1, :] = harmonic_mean(d_old[:-1, :], d_old[1:, :])
            dmean_old = dmean_old.flatten(order="F")
        else:
            dmean_old = np.zeros_like(dmean)

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp = grid.gamma_ij_x / grid.dx / grid.grid_cell_volume

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(1, grid.nx), slice(None)),
            (slice(0, grid.nx - 1), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_neigh] * tmp
        )  # type: ignore

    # Y contribution
    if grid.ny >= 2:
        dmean: NDArrayFloat = np.zeros((grid.nx, grid.ny), dtype=np.float64)
        dmean[:, :-1] = harmonic_mean(d[:, :-1], d[:, 1:])
        dmean = dmean.flatten(order="F")

        if time_index + 1 != time_params.nt:
            dmean_old: NDArrayFloat = np.zeros((grid.nx, grid.ny), dtype=np.float64)
            dmean_old[:, :-1] = harmonic_mean(d_old[:, :-1], d_old[:, 1:])
            dmean_old = dmean_old.flatten(order="F")
        else:
            dmean_old = np.zeros_like(dmean)

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp = grid.gamma_ij_y / grid.dy / grid.grid_cell_volume

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(1, grid.ny)),
            (slice(None), slice(0, grid.ny - 1)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean_old[idc_neigh] * tmp
        )  # type: ignore

    return q_next, q_prev


def _add_advection_to_adj_transport_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    crank_adv = tr_model.crank_nicolson_advection

    # X contribution
    if grid.nx >= 2:
        tmp = np.zeros((grid.nx, grid.ny))
        tmp[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index]
        un_x = tmp.flatten(order="F")

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )
        tmp = grid.gamma_ij_x / grid.grid_cell_volume

        tmp_un_pos = np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_owner]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore

        # Backward scheme
        normal = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(1, grid.nx), slice(None)),
            (slice(0, grid.nx - 1), slice(None)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp_un_pos = np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_neigh]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore

    # Y contribution
    if grid.ny >= 2:
        tmp = np.zeros((grid.nx, grid.ny))
        tmp[:, :-1] = fl_model.u_darcy_y[:, 1:-1, time_index]
        un_y = tmp.flatten(order="F")

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )
        tmp = grid.gamma_ij_y / grid.grid_cell_volume

        tmp_un_pos = np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_owner]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore

        # Backward scheme
        normal = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(1, grid.ny)),
            (slice(None), slice(0, grid.ny - 1)),
            owner_indices_to_keep=tr_model.free_conc_nn,
        )

        tmp_un_pos = np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_neigh]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1 - crank_adv) * tmp_un_pos * tmp  # type: ignore

    _apply_adj_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_adj_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_adj_transport_boundary_conditions(
        grid, fl_model, tr_model, q_next, q_prev, time_index
    )


def _apply_adj_transport_sink_term(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    flw = fl_model.lunitflow[time_index].flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * _flw)
    q_prev.setdiag(q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * _flw)


def _apply_adj_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    """Take into account the divergence: dcdt+U.grad(c)=L(u)."""
    src = fl_model.lunitflow[time_index]
    src_old = fl_model.lunitflow[time_index]

    div = (fl_model.u_darcy_div[:, :, time_index] - src).flatten(order="F")
    div_old = (fl_model.u_darcy_div[:, :, time_index] - src_old).flatten(order="F")

    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * div)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * div_old
    )


def _add_adj_transport_boundary_conditions(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_array,
    q_prev: lil_array,
    time_index: int,
) -> None:
    """Add the boundary conditions to the matrix."""
    # X contribution
    if grid.nx >= 2:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            grid,
            (slice(0, 1), slice(None)),
            (slice(grid.nx - 1, grid.nx), slice(None)),
        )
        tmp = grid.gamma_ij_x / grid.grid_cell_volume

        _un = fl_model.u_darcy_x[:-1, :, time_index].ravel("F")[idc_left]
        # _un_old = fl_model.u_darcy_x[:-1, :, time_index + 1].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore

        _un = fl_model.u_darcy_x[1:, :, time_index].ravel("F")[idc_right]
        # _un_old = fl_model.u_darcy_x[1:, :, time_index + 1].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore

    # Y contribution
    if grid.ny >= 2:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, 1)),
            (slice(None), slice(grid.ny - 1, grid.ny)),
        )
        tmp = grid.gamma_ij_y / grid.grid_cell_volume

        _un = fl_model.u_darcy_y[:, :-1, time_index].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore

        _un = fl_model.u_darcy_y[:, 1:, time_index].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore


def solve_adj_transport_transient_semi_implicit(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    time_params: TimeParameters,
    time_index: int,
    gch_params: GeochemicalParameters,
    nafpi: int,
) -> int:
    """Solving the adjoint transport equation."""

    # The matrix with respect to the diffusion never changes.
    # The matrix with respect to the advection only needs to be updated at the first
    # fix point iteration
    if nafpi == 1:
        q_next, q_prev = make_transient_adj_transport_matrices(
            grid, fl_model, tr_model, time_params, time_index
        )

        # Update q_next and q_prev with the advection term (must be copied)
        # Note that this is required at the first fixed point iteration only,
        # afterwards, only the chemical source term varies.
        _add_advection_to_adj_transport_matrices(
            grid, fl_model, tr_model, a_tr_model, q_next, q_prev, time_index
        )

        # # Add 1/dt for the left term contribution: only for free head
        # diag = np.zeros(shape)
        # diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index - 1])
        # # One for the cts head -> we must divide by storage coef and mesh volume
        # because
        # # we divide the other terms in _q_prev and q_next (better conditionning)
        # diag[fl_model.cst_head_nn] += (
        #     1.0
        # )

        # _q_next.setdiag(_q_next.diagonal() + diag)
        # diag = np.zeros(a_fl_model.a_head[:, :, -1].size)
        # # Need a try - except for n = N_{ts} resolution:
        # then \Delta t^{N_{ts}+1} does not
        # # exists
        # try:
        #     diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index])
        #     diag[fl_model.cst_head_nn] += (
        #         1.0
        #     )
        # except IndexError:
        #     pass
        # _q_prev.setdiag(_q_prev.diagonal() + diag)

        # Add 1/dt * \omgea for the left term contribution
        # Note; if the porosity and the timesteps are added here, it is to get the
        # highest values as possible on the diagonal of the matrices
        # -> better conditionning and easier LU preconditioning.
        q_next.setdiag(
            q_next.diagonal()
            + tr_model.porosity.flatten("F") / time_params.ldt[time_index - 1]
        )

        # Need a try - except for n = N_{ts} resolution: then \Delta t^{N_{ts}+1} does
        # not exists
        try:
            q_prev.setdiag(
                q_prev.diagonal()
                + tr_model.porosity.flatten("F") / time_params.ldt[time_index]
            )
        except IndexError:
            pass

        a_tr_model.q_next = q_next
        a_tr_model.q_prev = q_prev

        if tr_model.is_save_spmats:
            a_tr_model.l_q_next.append(q_next)
            a_tr_model.l_q_prev.append(q_prev)

            if time_index != time_params.nts:
                # q_prev (aka matrice B in the rhs) does not exists for the max timestep
                assert_allclose_sparse(
                    q_prev, tr_model.l_q_prev[time_index].T, rtol=1e-10
                )
            if time_index != 0:
                assert_allclose_sparse(
                    q_next, tr_model.l_q_next[time_index - 1].T, rtol=1e-10
                )
    else:
        q_next = a_tr_model.q_next
        q_prev = a_tr_model.q_prev

    # Get the previous vector
    try:
        prev_vector = (
            a_tr_model.a_mob[:, :, :, time_index + 1].reshape(2, -1, order="F").T
        )
    except IndexError:
        prev_vector = np.zeros((grid.n_grid_cells, 2))

    # Multiply prev matrix by prev vector
    tmp: NDArrayFloat = q_prev.dot(prev_vector).T

    for sp in range(tmp.shape[0]):
        # Add the source terms
        tmp[sp, :] -= a_tr_model.a_conc_sources[sp][:, [time_index]].todense().ravel()

    # Add the adjoint geochem source term
    tmp += a_tr_model.a_gch_src_term.reshape(2, -1, order="F")

    # Add the adjoint density source term for species 1
    tmp[0, :] += (
        a_tr_model.a_density[:, :, time_index].ravel("F")
        * WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * gch_params.Ms
        / 1000
    )

    # Add the adjoint density source term for species 2
    tmp[1, :] += (
        a_tr_model.a_density[:, :, time_index].ravel("F")
        * WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * gch_params.Ms2
        / 1000
    )

    # Build the LU preconditioning
    super_ilu, preconditioner = get_super_ilu_preconditioner(
        q_next.tocsc(),
        drop_tol=1e-10,
        fill_factor=100,
    )
    if super_ilu is None:
        warnings.warn(
            f"SuperILU: q_next is singular in adjoint transport at it={time_index}!"
        )

    # Solve Ax = b with A sparse using LU preconditioner
    for sp in range(tmp.shape[0]):
        tmp[sp], exit_code = gmres(
            q_next.tocsc(),
            tmp[sp],
            x0=super_ilu.solve(tmp[sp]) if super_ilu is not None else None,
            M=preconditioner,
            rtol=tr_model.rtol,
        )

    # Note: we go backward in time, so time_index -1...
    a_tr_model.a_mob[:, :, :, time_index] = tmp.reshape(
        tr_model.n_sp, grid.nx, grid.ny, order="F"
    )

    return exit_code
