"""Provide an adjoint solver and model."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, lil_array
from scipy.sparse.linalg import gmres

from pyrtid.forward.flow_solver import get_kmean
from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    FlowRegime,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import arithmetic_mean, get_super_lu_preconditioner, harmonic_mean
from pyrtid.utils.types import NDArrayFloat


def get_rhomean_adj(
    geometry: Geometry,
    tr_model: TransportModel,
    axis: int,
    time_index: int,
    is_flatten: bool = True,
) -> NDArrayFloat:
    rhomean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
    if axis == 0:
        rhomean[:-1, :] = arithmetic_mean(
            tr_model.ldensity[time_index][:-1, :], tr_model.ldensity[time_index][1:, :]
        )
    else:
        rhomean[:, :-1] = arithmetic_mean(
            tr_model.ldensity[time_index][:, :-1], tr_model.ldensity[time_index][:, 1:]
        )

    if is_flatten:
        return rhomean.flatten(order="F")
    return rhomean


def make_initial_adj_flow_matrices(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    is_q_prev_for_gradient: bool = False,
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the initial time step with a potential stationary flow.

    is_for_gradient: bool
        Whether the q_prev matrix is used for gradient w.r.t. h0 computation. If true
        the boundaries must be taken into account.

    Note
    ----
    Since the permeability and the storage coefficient does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)
    stocoeff = fl_model.storage_coefficient.ravel("F")
    if a_fl_model.crank_nicolson is None:
        fl_crank = fl_model.crank_nicolson
    else:
        fl_crank = a_fl_model.crank_nicolson

    # This is a trick to avoid building another matrix
    if is_q_prev_for_gradient:
        oitkg = None
    else:
        oitkg = fl_model.free_head_nn

    # 1) X contribution
    if geometry.nx >= 2:
        kmean = get_kmean(geometry, fl_model, 0)
        rhomean = get_rhomean_adj(geometry, tr_model, 0, 0)

        _tmp = geometry.dy / geometry.dx / geometry.mesh_volume

        # 1.1) Forward scheme:
        # 1.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_owner] += tmp  # type: ignore
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp

        # 1.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=oitkg,
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]
        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_neigh] -= tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp

        # 1.2) Backward scheme

        # 1.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_owner] += tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp

        # 1.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=oitkg,
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_neigh] -= tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp

    # 2) Y contribution
    if geometry.ny >= 2:
        kmean = get_kmean(geometry, fl_model, 0)
        rhomean = get_rhomean_adj(geometry, tr_model, 0, 0)

        # 2.1) Forward scheme:
        _tmp = geometry.dx / geometry.dy / geometry.mesh_volume

        # 2.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_owner] += tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp

        # 2.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=oitkg,
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_neigh] -= tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp

        # 2.2) Backward scheme

        # 2.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_owner] += tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp

        # 2.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=oitkg,
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.regime == FlowRegime.STATIONARY:
            q_next[idc_owner, idc_neigh] -= tmp
        # Handle density flow
        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY
        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp

    # Add 1/dt for the left term contribution: only for free head
    diag = np.zeros(q_next.shape[0])
    diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[0])
    # One for the cts head -> we must divide by storage coef and mesh volume because
    # we divide the other terms in _q_prev and q_next (better conditionning)
    diag[fl_model.cst_head_nn] += (
        1.0
        / fl_model.storage_coefficient.ravel("F")[fl_model.cst_head_nn]
        / geometry.mesh_volume
    )

    q_prev.setdiag(q_prev.diagonal() + diag)

    return q_next, q_prev


def make_transient_adj_flow_matrices(
    geometry: Geometry,
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

    dim = geometry.nx * geometry.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)
    stocoeff = fl_model.storage_coefficient.ravel("F")
    if a_fl_model.crank_nicolson is None:
        fl_crank: float = fl_model.crank_nicolson
    else:
        fl_crank = a_fl_model.crank_nicolson

    # 1) X contribution
    if geometry.nx >= 2:
        _tmp = geometry.dy / geometry.dx / geometry.mesh_volume

        kmean = get_kmean(geometry, fl_model, 0)
        rhomean = get_rhomean_adj(geometry, tr_model, 0, time_index)

        # 1.1) Forward scheme:

        # 1.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp  # type: ignore

        # 1.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp  # type: ignore

        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp  # type: ignore

        # 1.2) Backward scheme

        # 1.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp  # type: ignore

        # 1.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp  # type: ignore

        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp  # type: ignore

    # 2) Y contribution
    if geometry.ny >= 2:
        kmean = get_kmean(geometry, fl_model, 1)
        rhomean = get_rhomean_adj(geometry, tr_model, 1, time_index)

        # 2.1) Forward scheme:
        _tmp = geometry.dx / geometry.dy / geometry.mesh_volume

        # 2.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp  # type: ignore

        # 2.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_owner]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_owner] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp  # type: ignore

        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp  # type: ignore

        # 2.2) Backward scheme

        # 2.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_owner] += fl_crank * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (1.0 - fl_crank) * tmp  # type: ignore

        # 2.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner] * kmean[idc_neigh]

        if fl_model.is_gravity:
            tmp *= rhomean[idc_neigh] / WATER_DENSITY

        q_next[idc_owner, idc_neigh] -= fl_crank * tmp  # type: ignore

        q_prev[idc_owner, idc_neigh] += (1.0 - fl_crank) * tmp  # type: ignore

    return q_next, q_prev


def get_aflow_matrices(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> Tuple[csc_matrix, csc_matrix]:
    # In this context
    if time_index == 0:
        # q_next = make_stationary_flow_matrices(geometry, fl_model)
        return a_fl_model.q_next_init.tocsc(), a_fl_model.q_prev_init.tocsc()

    # Since the density vary over time, it is required to rebuild the adjoint
    # matrices at each timestep.
    if fl_model.is_gravity:
        _q_prev, _q_next = make_transient_adj_flow_matrices(
            geometry, fl_model, tr_model, a_fl_model, time_params, time_index
        )

    _q_prev = a_fl_model.q_prev.copy()
    _q_next = a_fl_model.q_next.copy()
    shape = a_fl_model.a_head[:, :, -1].size

    # Add 1/dt for the left term contribution: only for free head
    diag = np.zeros(shape)
    diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index - 1])
    # One for the cts head -> we must divide by storage coef and mesh volume because
    # we divide the other terms in _q_prev and q_next (better conditionning)
    diag[fl_model.cst_head_nn] += (
        1.0
        / fl_model.storage_coefficient.ravel("F")[fl_model.cst_head_nn]
        / geometry.mesh_volume
    )

    _q_next.setdiag(_q_next.diagonal() + diag)
    diag = np.zeros(a_fl_model.a_head[:, :, -1].size)
    # Need a try - except for n = N_{ts} resolution: then \Delta t^{N_{ts}+1} does not
    # exists
    try:
        diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index])
        diag[fl_model.cst_head_nn] += (
            1.0
            / fl_model.storage_coefficient.ravel("F")[fl_model.cst_head_nn]
            / geometry.mesh_volume
        )
    except IndexError:
        pass

    _q_prev.setdiag(_q_prev.diagonal() + diag)

    # convert to csc format for efficiency
    return _q_next.tocsc(), _q_prev.tocsc()


def update_adjoint_u_darcy(
    geometry: Geometry,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    time_index: int,
) -> None:
    crank_adv = tr_model.crank_nicolson_advection
    conc = tr_model.conc[:, :, time_index]
    a_conc = a_tr_model.a_conc[:, :, time_index]
    try:
        a_conc_old = a_tr_model.a_conc[:, :, time_index + 1]
        # prev_vector = a_fl_model.a_head[:, :, time_index + 1].ravel("F")
    except IndexError:
        # prev_vector = np.zeros(a_fl_model.a_head[:, :, 0].size)
        a_conc_old = np.zeros_like(a_conc)
        # a_tr_model.a_conc[:, :, time_index + 1]

    if time_index == 0:
        a_conc = np.zeros_like(a_conc)

    # X contribution
    if geometry.nx > 1:
        un_x = fl_model.u_darcy_x[1:-1, :, time_index]

        conc_ij_x = np.where(
            un_x > 0.0, conc[:-1, :], conc[1:, :]
        )  # take the conc depending on the forward flow direction
        conc_ij_x[un_x == 0] = 0

        # 1) advective term
        a_fl_model.a_u_darcy_x[1:-1, :, time_index] += geometry.dy * (
            (
                crank_adv * (a_conc[1:, :] - a_conc[:-1, :])
                + (1.0 - crank_adv) * (a_conc_old[1:, :] - a_conc_old[:-1, :])
            )
            * conc_ij_x
        )
        # 2) U divergence term
        a_fl_model.a_u_darcy_x[1:-1, :, time_index] += geometry.dy * (
            (
                crank_adv
                * (a_conc[:-1, :] * conc[:-1, :] - a_conc[1:, :] * conc[1:, :])
                + (1.0 - crank_adv)
                * (a_conc_old[:-1, :] * conc[:-1, :] - a_conc_old[1:, :] * conc[1:, :])
            )
        )

    # Y contribution
    if geometry.ny > 1:
        un_y = fl_model.u_darcy_y[:, 1:-1, time_index]
        conc_ij_y = np.where(
            un_y > 0.0, conc[:, :-1], conc[:, 1:]
        )  # take the conc depending on the forward flow direction
        conc_ij_y[un_y == 0] = 0

        # 1) advective term
        a_fl_model.a_u_darcy_y[:, 1:-1, time_index] += geometry.dx * (
            (
                crank_adv * (a_conc[:, 1:] - a_conc[:, :-1])
                + (1.0 - crank_adv) * (a_conc_old[:, 1:] - a_conc_old[:, :-1])
            )
            * conc_ij_y
        )
        # 2) U divergence term
        a_fl_model.a_u_darcy_y[:, 1:-1, time_index] += geometry.dx * (
            (
                crank_adv
                * (a_conc[:, :-1] * conc[:, :-1] - a_conc[:, 1:] * conc[:, 1:])
                + (1.0 - crank_adv)
                * (a_conc_old[:, :-1] * conc[:, :-1] - a_conc_old[:, 1:] * conc[:, 1:])
            )
        )


def solve_adj_flow(
    geometry: Geometry,
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
            geometry, fl_model, tr_model, a_fl_model, time_params, time_index
        )
    return solve_adj_flow_saturated(
        geometry, fl_model, tr_model, a_fl_model, time_params, time_index
    )


def solve_adj_flow_saturated(
    geometry: Geometry,
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

    # 1) Build adjoint flow matrices Q_{prev} and Q_{next}
    _q_next, _q_prev = get_aflow_matrices(
        geometry, fl_model, tr_model, a_fl_model, time_params, time_index
    )

    # 2) Build LU preconditioner for Q_{next}
    preconditioner = get_super_lu_preconditioner(_q_next)

    # 3) Obtain Q_{prev} @ h^{n+1}
    try:
        prev_vector = a_fl_model.a_head[:, :, time_index + 1].ravel("F")
    except IndexError:
        # This is the case for n = N_{ts}
        prev_vector = np.zeros(_q_next.shape[0], dtype=np.float_)
    tmp = _q_prev.dot(prev_vector)

    # 4) Add the source terms: observation on the head field
    tmp += (
        a_fl_model.a_head_sources.getcol(time_index).todense().ravel("F")
        / fl_model.storage_coefficient.ravel("F")
        / geometry.mesh_volume
    )

    # 5) Obtain the adjoint pressure and add it as a source term (observation on the
    # pressure field)
    a_fl_model.a_pressure[:, :, time_index] = (
        a_fl_model.a_pressure_sources.getcol(time_index)
        .todense()
        .reshape(geometry.nx, geometry.ny, order="F")
    )
    tmp += (
        (
            a_fl_model.a_pressure[:, :, time_index].ravel("F")
            / fl_model.storage_coefficient.ravel("F")
            / geometry.mesh_volume
        )
        * WATER_DENSITY
        * GRAVITY
    )

    # 6) Add the source terms from mob observations (adjoint transport)
    tmp += get_adjoint_transport_src_terms(
        geometry, fl_model, a_fl_model, time_index, True
    )

    # 7) Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(_q_next, tmp, M=preconditioner, atol=1e-15)

    # 8) Impose null adjoint head the the cst head boundaries
    if time_index == 0:
        res[fl_model.cst_head_nn] = 0.0

    # 9) Update the adjoint head field
    a_fl_model.a_head[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code


def solve_adj_flow_density(
    geometry: Geometry,
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

    # 1) Build adjoint flow matrices Q_{prev} and Q_{next}
    _q_next, _q_prev = get_aflow_matrices(
        geometry, fl_model, tr_model, a_fl_model, time_params, time_index
    )

    # 2) Build LU preconditioner for Q_{next}
    preconditioner = get_super_lu_preconditioner(_q_next)

    # 3) Obtain Q_{prev} @ p^{n+1}
    try:
        prev_vector = a_fl_model.a_pressure[:, :, time_index + 1].ravel("F")
    except IndexError:
        # This is the case for n = N_{ts}
        prev_vector = np.zeros(_q_next.shape[0], dtype=np.float_)
    # Multiply prev matrix by prev vector (p^{n+1}
    tmp = _q_prev.dot(prev_vector)

    # 4) Add the source terms: observation on the pressure field
    tmp += (
        a_fl_model.a_pressure_sources.getcol(time_index).todense().ravel("F")
        / fl_model.storage_coefficient.ravel("F")
        / geometry.mesh_volume
    )

    # 5) Obtain the adjoint head field and add it as a source term (observation on the
    # head field)
    a_fl_model.a_head[:, :, time_index] = (
        a_fl_model.a_head_sources.getcol(time_index)
        .todense()
        .reshape(geometry.nx, geometry.ny, order="F")
    )
    # Handle the density (forward variable) for n = 0 (initial system state).
    try:
        density = tr_model.ldensity[time_index - 1]  # type: ignore
    except IndexError:
        density = tr_model.ldensity[time_index]  # type: ignore

    tmp += (
        (
            a_fl_model.a_head[:, :, time_index].ravel("F")
            / fl_model.storage_coefficient.ravel("F")
            / geometry.mesh_volume
        )
        / density.ravel("F")
        / GRAVITY
    )

    # 6) Add the source terms from mob observations (adjoint transport)
    tmp += get_adjoint_transport_src_terms(
        geometry, fl_model, a_fl_model, time_index, True
    )

    # 7) Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(_q_next, tmp, M=preconditioner, atol=1e-15)

    # 8) Impose null adjoint head the the cst head boundaries
    if time_index == 0:
        res[fl_model.cst_head_nn] = 0.0

    # 9) Update the adjoint pressure field
    a_fl_model.a_pressure[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code


def get_adjoint_transport_src_terms(
    geometry: Geometry,
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    time_index: int,
    is_transient: bool,
) -> NDArrayFloat:
    """
    Add the source terms linked with the transport (mob observations).

    Parameters
    ----------
    geometry : Geometry
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
    src = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
    tmp = 1.0
    # Handle density flow
    if fl_model.is_gravity:
        tmp = 1.0 / GRAVITY / WATER_DENSITY

    # x contribution
    if geometry.nx > 1:
        # Get the permeability between nodes
        kmean_x = harmonic_mean(
            fl_model.permeability[:-1, :], fl_model.permeability[1:, :]
        )

        # Forward
        src[:-1, :] += (
            kmean_x
            * a_fl_model.a_u_darcy_x[1:-1, :, time_index]
            / geometry.dx
            / geometry.mesh_volume
        ) * tmp

        # Backward
        src[1:, :] -= (
            kmean_x
            * a_fl_model.a_u_darcy_x[1:-1, :, time_index]
            / geometry.dx
            / geometry.mesh_volume
        ) * tmp

    # y contribution
    if geometry.ny > 1:
        # Get the permeability between nodes
        kmean_y = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )

        # Forward
        src[:, :-1] += (
            kmean_y
            * a_fl_model.a_u_darcy_y[:, 1:-1, time_index]
            / geometry.dy
            / geometry.mesh_volume
        ) * tmp

        # Backward
        src[:, 1:] -= (
            kmean_y
            * a_fl_model.a_u_darcy_y[:, 1:-1, time_index]
            / geometry.dy
            / geometry.mesh_volume
        ) * tmp

    # Divide by the storage coefficient only if transient mode
    if is_transient:
        return src.ravel("F") / fl_model.storage_coefficient.ravel("F")
    return src.ravel("F")
