"""Provide an adjoint solver and model."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, lil_array
from scipy.sparse.linalg import lgmres

from pyrtid.forward.flow_solver import get_kmean, get_rhomean
from pyrtid.forward.models import (  # ConstantHead,; ZeromobGradient,
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    FlowRegime,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.inverse.asm.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import (
    NDArrayFloat,
    RectilinearGrid,
    assert_allclose_sparse,
    dxi_harmonic_mean,
    get_super_ilu_preconditioner,
    harmonic_mean,
)


def add_adj_stationary_flow_to_q_next(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    q_next: lil_array,
) -> lil_array:
    """
    Make matrices for the initial time step with a potential stationary flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """
    # 1) X contribution
    if grid.nx >= 2:
        kmean = get_kmean(grid, fl_model, 0)

        _tmp = grid.gamma_ij_x / grid.dx / grid.grid_cell_volume
        if fl_model.is_gravity:
            _tmp /= WATER_DENSITY * GRAVITY

        # 1.1) Forward scheme:
        # 1.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_owner]
        q_next[idc_owner, idc_owner] += tmp  # type: ignore

        # 1.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_owner]
        q_next[idc_owner, idc_neigh] -= tmp

        # 1.2) Backward scheme

        # 1.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(1, grid.nx), slice(None)),
            (slice(0, grid.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_neigh]
        q_next[idc_owner, idc_owner] += tmp

        # 1.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(1, grid.nx), slice(None)),
            (slice(0, grid.nx - 1), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_neigh]
        q_next[idc_owner, idc_neigh] -= tmp

    # 2) Y contribution
    if grid.ny >= 2:
        kmean = get_kmean(grid, fl_model, 1)

        # 2.1) Forward scheme:
        _tmp = grid.gamma_ij_y / grid.dy / grid.grid_cell_volume
        if fl_model.is_gravity:
            _tmp /= WATER_DENSITY * GRAVITY

        # 2.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_owner]
        q_next[idc_owner, idc_owner] += tmp

        # 2.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_owner]
        q_next[idc_owner, idc_neigh] -= tmp

        # 2.2) Backward scheme

        # 2.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(1, grid.ny)),
            (slice(None), slice(0, grid.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_neigh]
        q_next[idc_owner, idc_owner] += tmp

        # 2.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(1, grid.ny)),
            (slice(None), slice(0, grid.ny - 1)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp * kmean[idc_neigh]
        q_next[idc_owner, idc_neigh] -= tmp

    return q_next


def make_transient_adj_flow_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the transient flow.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = grid.nx * grid.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)
    sc = fl_model.storage_coefficient.ravel("F")
    if a_fl_model.crank_nicolson is None:
        fl_crank: float = fl_model.crank_nicolson
    else:
        fl_crank = a_fl_model.crank_nicolson

    # 1) X contribution
    if grid.nx >= 2:
        _tmp = grid.gamma_ij_x / grid.dx / grid.grid_cell_volume

        kmean = get_kmean(grid, fl_model, 0)
        # at n - 1
        rhomean_next = get_rhomean(grid, tr_model, axis=0, time_index=time_index - 1)
        # at n
        rhomean_prev = get_rhomean(grid, tr_model, axis=0, time_index=time_index)
        # 1.1) Forward scheme:

        # 1.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_owner]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_owner] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp_prev  # type: ignore

        # 1.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_owner]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_owner] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp_prev  # type: ignore

        # 1.2) Backward scheme

        # 1.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(1, grid.nx), slice(None)),
            (slice(0, grid.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_neigh]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_neigh] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp_prev  # type: ignore

        # 1.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(1, grid.nx), slice(None)),
            (slice(0, grid.nx - 1), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_neigh]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_neigh] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp_prev  # type: ignore

    # 2) Y contribution
    if grid.ny >= 2:
        kmean = get_kmean(grid, fl_model, 1)

        # at n - 1
        rhomean_next = get_rhomean(grid, tr_model, axis=1, time_index=time_index - 1)
        # at n
        rhomean_prev = get_rhomean(grid, tr_model, axis=1, time_index=time_index)
        # 2.1) Forward scheme:
        _tmp = grid.gamma_ij_y / grid.dy / grid.grid_cell_volume

        # 2.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_owner]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_owner] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp_prev  # type: ignore

        # 2.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_owner]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_owner] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp_next  # type: ignore

        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp_prev  # type: ignore

        # 2.2) Backward scheme

        # 2.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(1, grid.ny)),
            (slice(None), slice(0, grid.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_neigh]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_neigh] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp_prev  # type: ignore

        # 2.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(1, grid.ny)),
            (slice(None), slice(0, grid.ny - 1)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp_next = _tmp / sc[idc_owner] * kmean[idc_neigh]
        tmp_prev = tmp_next.copy()

        if fl_model.is_gravity:
            tmp_next *= rhomean_next[idc_neigh] / WATER_DENSITY
            tmp_prev *= rhomean_prev[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp_next  # type: ignore
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp_prev  # type: ignore

    return q_next, q_prev


def get_aflow_matrices(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> Tuple[csc_matrix, csc_matrix]:
    # Since the density vary over time, it is required to rebuild the adjoint
    # matrices at each timestep.
    if fl_model.is_gravity:
        _q_next, _q_prev = make_transient_adj_flow_matrices(
            grid, fl_model, tr_model, a_fl_model, time_params, time_index
        )
    else:
        _q_prev = a_fl_model.q_prev.copy()
        _q_next = a_fl_model.q_next.copy()
    shape = a_fl_model.a_head[:, :, -1].size

    diag = np.zeros(shape)
    if time_index == 0:
        _q_next = lil_array(_q_next.shape, dtype=np.float64)
        if fl_model.regime == FlowRegime.TRANSIENT:
            diag += 1.0
        else:  # stationary case
            # add the derivative of the stationary flow for initialization
            _q_next = add_adj_stationary_flow_to_q_next(grid, fl_model, _q_next)
            diag[fl_model.cst_head_nn] += 1.0
    else:
        # Add 1/dt for the left term contribution: only for free head
        diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index - 1])
        diag[fl_model.cst_head_nn] += 1.0
    _q_next.setdiag(_q_next.diagonal() + diag)

    diag = np.zeros(a_fl_model.a_head[:, :, -1].size)
    # Need a try - except for n = N_{ts} resolution: then \Delta t^{N_{ts}+1} does not
    # exists
    try:
        diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index])
    except IndexError:
        pass

    _q_prev.setdiag(_q_prev.diagonal() + diag)

    if fl_model.is_save_spmats:
        a_fl_model.l_q_next.append(_q_next)
        a_fl_model.l_q_prev.append(_q_prev)

        if time_index != time_params.nts:
            # q_prev (aka matrice B in the rhs) does not exists for the max timestep
            assert_allclose_sparse(
                _q_prev, fl_model.l_q_prev[time_index + 1].T, rtol=1e-8
            )
        assert_allclose_sparse(_q_next, fl_model.l_q_next[time_index].T, rtol=1e-8)

    # convert to csc format for efficiency
    return _q_next.tocsc(), _q_prev.tocsc()


def update_adjoint_u_darcy(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    crank_adv = tr_model.crank_nicolson_advection
    crank_diff = tr_model.crank_nicolson_diffusion
    # for the dispersion derivation
    d = (
        tr_model.effective_diffusion
        + tr_model.dispersivity * fl_model.get_u_darcy_norm_sample(time_index)
    )
    dUx, dUy = fl_model.get_du_darcy_norm_sample(time_index)
    lhs = np.zeros((grid.nx, grid.ny), dtype=np.float64)

    # loop over the species
    for sp in range(tr_model.n_sp):
        # time: n
        mob = tr_model.mob[sp, :, :, time_index]

        if time_index != time_params.nts:
            # time: n + 1
            a_mob_old = a_tr_model.a_mob[sp, :, :, time_index + 1]
        else:
            a_mob_old = np.zeros_like(mob)

        if time_index != 0:
            # time: n - 1
            mob_next = tr_model.mob[sp, :, :, time_index - 1]
            a_mob = a_tr_model.a_mob[sp, :, :, time_index]
        else:
            mob_next = np.zeros_like(mob)
            a_mob = np.zeros_like(mob)

        # X contribution
        if grid.nx > 1:
            un_x = fl_model.u_darcy_x[1:-1, :, time_index]

            mob_ij_x = np.where(
                un_x > 0.0, mob[:-1, :], mob[1:, :]
            )  # take the mob depending on the forward flow direction
            mob_ij_x[un_x == 0] = 0

            # 1) advective term
            a_fl_model.a_u_darcy_x[1:-1, :, time_index] += (
                grid.gamma_ij_x
                * (
                    (
                        crank_adv * (a_mob[1:, :] - a_mob[:-1, :])
                        + (1.0 - crank_adv) * (a_mob_old[1:, :] - a_mob_old[:-1, :])
                    )
                    * mob_ij_x
                )
                / grid.grid_cell_volume
            )

            # 2) U divergence term
            a_fl_model.a_u_darcy_x[1:-1, :, time_index] += (
                grid.gamma_ij_x
                * (
                    crank_adv
                    * (a_mob[:-1, :] * mob[:-1, :] - a_mob[1:, :] * mob[1:, :])
                    + (1.0 - crank_adv)
                    * (a_mob_old[:-1, :] * mob[:-1, :] - a_mob_old[1:, :] * mob[1:, :])
                )
                / grid.grid_cell_volume
            )

            # 3) Dispersivity term -> \lmabda *
            # forward in space
            lhs[:-1, :] += (
                grid.gamma_ij_x
                / grid.dx
                / grid.grid_cell_volume
                * (
                    (
                        crank_diff * (mob[1:, :] - mob[:-1, :])
                        + (1.0 - crank_diff) * (mob_next[1:, :] - mob_next[:-1, :])
                    )
                    * (a_mob[:-1, :] - a_mob[1:, :])
                )
                * dxi_harmonic_mean(d[:-1, :], d[1:, :])
            )
            # backward in space
            lhs[1:, :] += (
                grid.gamma_ij_x
                / grid.dx
                / grid.grid_cell_volume
                * (
                    (
                        crank_diff * (mob[:-1, :] - mob[1:, :])
                        + (1.0 - crank_diff) * (mob_next[:-1, :] - mob_next[1:, :])
                    )
                    * (a_mob[1:, :] - a_mob[:-1, :])
                )
                * dxi_harmonic_mean(d[1:, :], d[:-1, :])
            )

        # Y contribution
        if grid.ny > 1:
            un_y = fl_model.u_darcy_y[:, 1:-1, time_index]
            mob_ij_y = np.where(
                un_y > 0.0, mob[:, :-1], mob[:, 1:]
            )  # take the mob depending on the forward flow direction
            mob_ij_y[un_y == 0] = 0

            # 1) advective term
            a_fl_model.a_u_darcy_y[:, 1:-1, time_index] += (
                grid.gamma_ij_y
                * (
                    (
                        crank_adv * (a_mob[:, 1:] - a_mob[:, :-1])
                        + (1.0 - crank_adv) * (a_mob_old[:, 1:] - a_mob_old[:, :-1])
                    )
                    * mob_ij_y
                )
                / grid.grid_cell_volume
            )
            # 2) U divergence term
            a_fl_model.a_u_darcy_y[:, 1:-1, time_index] += (
                grid.gamma_ij_y
                * (
                    crank_adv
                    * (a_mob[:, :-1] * mob[:, :-1] - a_mob[:, 1:] * mob[:, 1:])
                    + (1.0 - crank_adv)
                    * (a_mob_old[:, :-1] * mob[:, :-1] - a_mob_old[:, 1:] * mob[:, 1:])
                )
                / grid.grid_cell_volume
            )

            # 3) Dispersivity term
            # forward in space
            lhs[:, :-1] += (
                grid.gamma_ij_y
                / grid.dy
                / grid.grid_cell_volume
                * (
                    (
                        crank_diff * (mob[:, 1:] - mob[:, :-1])
                        + (1.0 - crank_diff) * (mob_next[:, 1:] - mob_next[:, :-1])
                    )
                    * (a_mob[:, :-1] - a_mob[:, 1:])
                )
                * dxi_harmonic_mean(d[:, :-1], d[:, 1:])
            )
            # backward in space
            lhs[:, 1:] += (
                grid.gamma_ij_y
                / grid.dy
                / grid.grid_cell_volume
                * (
                    (
                        crank_diff * (mob[:, :-1] - mob[:, 1:])
                        + (1.0 - crank_diff) * (mob_next[:, :-1] - mob_next[:, 1:])
                    )
                    * (a_mob[:, 1:] - a_mob[:, :-1])
                )
                * dxi_harmonic_mean(d[:, 1:], d[:, :-1])
            )

    # End dispersivity term
    a_fl_model.a_u_darcy_x[:-1, :, time_index] += dUx * lhs * tr_model.dispersivity
    a_fl_model.a_u_darcy_x[1:, :, time_index] += dUx * lhs * tr_model.dispersivity
    a_fl_model.a_u_darcy_y[:, :-1, time_index] += dUy * lhs * tr_model.dispersivity
    a_fl_model.a_u_darcy_y[:, 1:, time_index] += dUy * lhs * tr_model.dispersivity


def solve_adj_flow(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """
    if fl_model.is_gravity:
        return solve_adj_flow_density(
            grid, fl_model, tr_model, a_fl_model, time_params, time_index
        )
    return solve_adj_flow_saturated(
        grid, fl_model, tr_model, a_fl_model, time_params, time_index
    )


def solve_adj_flow_saturated(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """

    # 1) Obtain the adjoint pressure and add it as a source term (observation on the
    # pressure field)
    a_fl_model.a_pressure[:, :, time_index] = -(
        a_fl_model.a_pressure_sources[:, [time_index]]
        .todense()
        .reshape(grid.nx, grid.ny, order="F")
    )

    # 2) Build adjoint flow matrices Q_{prev} and Q_{next}
    _q_next, _q_prev = get_aflow_matrices(
        grid, fl_model, tr_model, a_fl_model, time_params, time_index
    )

    # 3) Build LU preconditioner for Q_{next}
    super_ilu, preconditioner = get_super_ilu_preconditioner(
        _q_next, drop_tol=1e-10, fill_factor=100
    )
    if super_ilu is None:
        warnings.warn(
            "SuperILU: q_next is singular in adjoint "
            f"saturated flow at it={time_index}!"
        )

    # 4) Obtain Q_{prev} @ h^{n+1}
    try:
        prev_vector = a_fl_model.a_head[:, :, time_index + 1].ravel("F")
    except IndexError:
        # This is the case for n = N_{ts}
        prev_vector = np.zeros(_q_next.shape[0], dtype=np.float64)
    tmp = _q_prev.dot(prev_vector)

    # 5) Add the source terms: observation on the head field
    tmp -= a_fl_model.a_head_sources[:, [time_index]].todense().ravel("F")

    tmp += (
        (a_fl_model.a_pressure[:, :, time_index].ravel("F")) * WATER_DENSITY * GRAVITY
    )

    # 6) Add the source terms from mob observations (adjoint transport)
    tmp += get_adjoint_transport_src_terms(grid, fl_model, a_fl_model, time_index)

    # 7) Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = lgmres(
        _q_next,
        tmp,
        x0=super_ilu.solve(tmp) if super_ilu is not None else None,
        M=preconditioner,
        rtol=a_fl_model.rtol,
    )

    # 8) Update the adjoint head field
    a_fl_model.a_head[:, :, time_index] = res.reshape(grid.ny, grid.nx).T

    return exit_code


def solve_adj_flow_density(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """

    # 1) Obtain the adjoint head field and add it as a source term (observation on the
    # head field)
    a_fl_model.a_head[:, :, time_index] = -(
        a_fl_model.a_head_sources[:, [time_index]]
        .todense()
        .reshape(grid.nx, grid.ny, order="F")
    )

    # 2) Build adjoint flow matrices Q_{prev} and Q_{next}
    _q_next, _q_prev = get_aflow_matrices(
        grid, fl_model, tr_model, a_fl_model, time_params, time_index
    )

    # 3) Build LU preconditioner for Q_{next}
    super_ilu, preconditioner = get_super_ilu_preconditioner(
        _q_next, drop_tol=1e-10, fill_factor=100
    )
    if super_ilu is None:
        warnings.warn(
            f"SuperILU: q_next is singular in adjoint density flow at it={time_index}!"
        )

    # 4) Obtain Q_{prev} @ p^{n+1}
    try:
        prev_vector = a_fl_model.a_pressure[:, :, time_index + 1].ravel("F")
    except IndexError:
        # This is the case for n = N_{ts}
        prev_vector = np.zeros(_q_next.shape[0], dtype=np.float64)
    # Multiply prev matrix by prev vector (p^{n+1}
    tmp = _q_prev.dot(prev_vector)

    # 5) Add the source terms: observation on the pressure field
    tmp -= a_fl_model.a_pressure_sources[:, [time_index]].todense().ravel("F")

    # 6) Handle the density (forward variable) for n = 0 (initial system state).
    if time_index != 0:
        density = tr_model.ldensity[time_index - 1]  # type: ignore
    else:
        density = tr_model.ldensity[time_index]  # type: ignore

    tmp += (
        (a_fl_model.a_head[:, :, time_index].ravel("F")) / density.ravel("F") / GRAVITY
    )

    # 7) Add the source terms from mob observations (adjoint transport)
    tmp += get_adjoint_transport_src_terms(grid, fl_model, a_fl_model, time_index)

    # 8) Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = lgmres(
        _q_next,
        tmp,
        x0=super_ilu.solve(tmp) if super_ilu is not None else None,
        M=preconditioner,
        rtol=a_fl_model.rtol,
    )

    # 10) Update the adjoint pressure field
    a_fl_model.a_pressure[:, :, time_index] = res.reshape(grid.ny, grid.nx).T

    return exit_code


def get_adjoint_transport_src_terms(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    time_index: int,
) -> NDArrayFloat:
    """
    Add the source terms linked with the transport (mob observations).

    Parameters
    ----------
    grid : RectilinearGrid
        _description_
    tmp : NDArrayFloat
        _description_
    fl_model : FlowModel
        _description_
    a_fl_model : AdjointFlowModel
        _description_
    is_transient : bool
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    src = np.zeros((grid.nx, grid.ny), dtype=np.float64)
    tmp = 1.0
    # Handle density flow
    if fl_model.is_gravity:
        tmp = 1.0 / GRAVITY / WATER_DENSITY

    # x contribution
    if grid.nx > 1:
        # Get the permeability between nodes
        kmean_x = harmonic_mean(
            fl_model.permeability[:-1, :], fl_model.permeability[1:, :]
        )

        # Forward
        src[:-1, :] += (
            kmean_x * a_fl_model.a_u_darcy_x[1:-1, :, time_index] / grid.dx
        ) * tmp

        # Backward
        src[1:, :] -= (
            kmean_x * a_fl_model.a_u_darcy_x[1:-1, :, time_index] / grid.dx
        ) * tmp

    # y contribution
    if grid.ny > 1:
        # Get the permeability between nodes
        kmean_y = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )

        # Forward
        src[:, :-1] += (
            kmean_y * a_fl_model.a_u_darcy_y[:, 1:-1, time_index] / grid.dy
        ) * tmp

        # Backward
        src[:, 1:] -= (
            kmean_y * a_fl_model.a_u_darcy_y[:, 1:-1, time_index] / grid.dy
        ) * tmp

    return src.ravel("F")
