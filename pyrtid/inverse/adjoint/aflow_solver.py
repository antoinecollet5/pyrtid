"""Provide an adjoint solver and model."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    FlowModel,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import harmonic_mean
from pyrtid.utils.types import NDArrayFloat


def make_stationary_adj_flow_matrices(
    geometry: Geometry, fl_model: FlowModel, time_params: TimeParameters
) -> Tuple[csc_matrix, csc_matrix]:
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
    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(0, geometry.nx - 1), slice(None)),
        (slice(1, geometry.nx), slice(None)),
        fl_model.cst_head_nn,
    )

    q_next[idc_owner, idc_neigh] -= kmean[idc_owner] / geometry.dx**2
    q_next[idc_owner, idc_owner] += kmean[idc_owner] / geometry.dx**2

    q_prev[idc_owner, idc_neigh] += (
        (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] / geometry.dx**2
    )
    q_prev[idc_owner, idc_owner] -= (
        (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] / geometry.dx**2
    )

    # Backward scheme
    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(1, geometry.nx), slice(None)),
        (slice(0, geometry.nx - 1), slice(None)),
        fl_model.cst_head_nn,
    )

    q_next[idc_owner, idc_neigh] -= kmean[idc_neigh] / geometry.dx**2
    q_next[idc_owner, idc_owner] += kmean[idc_neigh] / geometry.dx**2

    q_prev[idc_owner, idc_neigh] += (
        (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] / geometry.dx**2
    )
    q_prev[idc_owner, idc_owner] -= (
        (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] / geometry.dx**2
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

        q_next[idc_owner, idc_neigh] -= kmean[idc_owner] / geometry.dy**2
        q_next[idc_owner, idc_owner] += kmean[idc_owner] / geometry.dy**2

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] / geometry.dy**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] / geometry.dy**2
        )

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            fl_model.cst_head_nn,
        )

        q_next[idc_owner, idc_neigh] -= kmean[idc_neigh] / geometry.dy**2
        q_next[idc_owner, idc_owner] += kmean[idc_neigh] / geometry.dy**2

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] / geometry.dy**2
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] / geometry.dy**2
        )

    # Take constant head into account
    q_next[fl_model.cst_head_nn, fl_model.cst_head_nn] = 1.0

    # Add 1/dt for the left term contribution
    q_prev.setdiag(
        q_prev.diagonal() + 1 / time_params.dt * fl_model.storage_coefficient
    )

    return q_next.tocsc(), q_prev.tocsc()


def make_transient_adj_flow_matrices(
    geometry: Geometry, fl_model: FlowModel, time_params: TimeParameters
) -> Tuple[csc_matrix, csc_matrix]:
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

    # # X contribution
    # kmean = harmonic_mean(fl_model.permeability[:-1, :], fl_model.permeability[1:, :])

    # X contribution
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
        fl_model.crank_nicolson
        * kmean[idc_owner]
        / geometry.dx**2
        / fl_model.storage_coefficient
    )
    q_next[idc_owner, idc_owner] += (
        fl_model.crank_nicolson
        * kmean[idc_owner]
        / geometry.dx**2
        / fl_model.storage_coefficient
    )
    q_prev[idc_owner, idc_neigh] += (
        (1.0 - fl_model.crank_nicolson)
        * kmean[idc_owner]
        / geometry.dx**2
        / fl_model.storage_coefficient
    )
    q_prev[idc_owner, idc_owner] -= (
        (1.0 - fl_model.crank_nicolson)
        * kmean[idc_owner]
        / geometry.dx**2
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
        / fl_model.storage_coefficient
    )
    q_next[idc_owner, idc_owner] += (
        fl_model.crank_nicolson
        * kmean[idc_neigh]
        / geometry.dx**2
        / fl_model.storage_coefficient
    )
    q_prev[idc_owner, idc_neigh] += (
        (1.0 - fl_model.crank_nicolson)
        * kmean[idc_neigh]
        / geometry.dx**2
        / fl_model.storage_coefficient
    )
    q_prev[idc_owner, idc_owner] -= (
        (1.0 - fl_model.crank_nicolson)
        * kmean[idc_neigh]
        / geometry.dx**2
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
            / fl_model.storage_coefficient
        )
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_owner]
            / geometry.dy**2
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            / geometry.dy**2
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_owner]
            / geometry.dy**2
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
            / fl_model.storage_coefficient
        )
        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson
            * kmean[idc_neigh]
            / geometry.dy**2
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            / geometry.dy**2
            / fl_model.storage_coefficient
        )
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson)
            * kmean[idc_neigh]
            / geometry.dy**2
            / fl_model.storage_coefficient
        )

    # Add 1/dt for the left term contribution
    q_next.setdiag(q_next.diagonal() + 1 / time_params.dt)
    q_prev.setdiag(q_prev.diagonal() + 1 / time_params.dt)

    return q_next.tocsc(), q_prev.tocsc()


def solve_adj_flow_stationary(
    geometry: Geometry,
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    preconditioner: Optional[LinearOperator],
    time_index: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """
    # Multiply prev matrix by prev vector
    tmp = a_fl_model.q_prev.dot(a_fl_model.a_head[:, :, time_index].ravel("F"))

    # Constant heads
    tmp[fl_model.cst_head_nn] = 0.0

    # Add the source terms from head observations
    tmp += a_fl_model.a_sources[:, :, time_index].ravel("F") / geometry.mesh_area

    # Add the source terms from mob observations (adjoint transport)
    # tmp += _get_adjoint_transport_src_terms(
    #     geometry, fl_model, a_fl_model, time_index, False
    # )

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(a_fl_model.q_next, tmp, M=preconditioner, atol=1e-15)
    # Note: we solve
    a_fl_model.a_head[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code


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
        a_conc_old = np.zeros(a_tr_model.a_conc[:, :, 0].shape)
        # a_tr_model.a_conc[:, :, time_index + 1]

    # X contribution
    un_x = fl_model.u_darcy_x[1:-1, :, time_index]

    conc_fx = np.where(
        un_x > 0.0, conc[:-1, :], conc[1:, :]
    )  # take the conc depending on the forward flow direction

    # Forward
    # 1) advective term
    a_fl_model.a_u_darcy_x[:, :, time_index] += geometry.dy * (
        (
            crank_adv * (a_conc[:-1, :] - a_conc[1:, :])
            + (1.0 - crank_adv) * (a_conc_old[:-1, :] - a_conc_old[1:, :])
        )
        * conc_fx
        / 2.0
    )
    # 2) U divergence term
    a_fl_model.a_u_darcy_x[:, :, time_index] -= geometry.dy * (
        (
            crank_adv * (a_conc[:-1, :] * conc[:-1, :] - a_conc[1:, :] * conc[1:, :])
            + (1.0 - crank_adv)
            * (a_conc_old[:-1, :] * conc[:-1, :] - a_conc_old[1:, :] * conc[1:, :])
        )
        / 2.0
    )

    conc_bx = np.where(
        un_x <= 0.0, conc[1:, :], conc[:-1, :]
    )  # take the conc depending on the forward flow direction

    # Backward
    # 1) advective term
    a_fl_model.a_u_darcy_x[:, :, time_index] -= geometry.dy * (
        (
            crank_adv * (a_conc[1:, :] - a_conc[:-1, :])
            + (1.0 - crank_adv) * (a_conc_old[1:, :] - a_conc_old[:-1, :])
        )
        * conc_bx
        / 2.0
    )
    # 2) U divergence
    a_fl_model.a_u_darcy_x[:, :, time_index] += geometry.dy * (
        (
            crank_adv * (a_conc[1:, :] * conc[1:, :] - a_conc[:-1, :] * conc[:-1, :])
            + (1.0 - crank_adv)
            * (a_conc_old[1:, :] * conc[1:, :] - a_conc_old[:-1, :] * conc[:-1, :])
        )
        / 2.0
    )

    # Y contribution
    if geometry.ny > 1:
        un_y = fl_model.u_darcy_y[:, 1:-1, time_index]

        conc_fy = np.where(
            un_y > 0.0, conc[:, :-1], conc[:, 1:]
        )  # take the conc depending on the forward flow direction

        # Forward
        # 1) advective term
        a_fl_model.a_u_darcy_y[:, :, time_index] += geometry.dx * (
            (
                crank_adv * (a_conc[:, :-1] - a_conc[:, 1:])
                + (1.0 - crank_adv) * (a_conc_old[:, :-1] - a_conc_old[:, 1:])
            )
            * conc_fy
            / 2.0
        )
        # 2) U divergence term
        a_fl_model.a_u_darcy_y[:, :, time_index] -= geometry.dx * (
            (
                crank_adv
                * (a_conc[:, :-1] * conc[:, :-1] - a_conc[:, 1:] * conc[:, 1:])
                + (1.0 - crank_adv)
                * (a_conc_old[:, :-1] * conc[:, :-1] - a_conc_old[:, 1:] * conc[:, 1:])
            )
            / 2.0
        )

        conc_by = np.where(
            un_y <= 0.0, conc[:, 1:], conc[:, :-1]
        )  # take the conc depending on the forward flow direction

        # Backward
        # 1) advective term
        a_fl_model.a_u_darcy_y[:, :, time_index] -= geometry.dx * (
            (
                crank_adv * (a_conc[:, 1:] - a_conc[:, :-1])
                + (1.0 - crank_adv) * (a_conc_old[:, 1:] - a_conc_old[:, :-1])
            )
            * conc_by
            / 2.0
        )
        # 2) U divergence
        a_fl_model.a_u_darcy_y[:, :, time_index] += geometry.dx * (
            (
                crank_adv
                * (a_conc[:, 1:] * conc[:, 1:] - a_conc[:, :-1] * conc[:, :-1])
                + (1.0 - crank_adv)
                * (a_conc_old[:, 1:] * conc[:, 1:] - a_conc_old[:, :-1] * conc[:, :-1])
            )
            / 2.0
        )


def solve_adj_flow_transient_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    preconditioner: Optional[LinearOperator],
    time_index: int,
    is_mob_obs: bool,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """
    _q_prev = a_fl_model.q_prev
    _q_next = a_fl_model.q_next

    # Handle the first time step in the adjoint (= last timestep in the forward)
    # if time_index + 1 != a_fl_model.a_sources.shape[-1]:
    try:
        prev_vector = a_fl_model.a_head[:, :, time_index + 1].ravel("F")
    except IndexError:
        prev_vector = np.zeros(a_fl_model.a_head[:, :, 0].size)

    # Multiply prev matrix by prev vector
    tmp = _q_prev.dot(prev_vector)

    # Add the source terms
    # Note: there is no crank-nicolson scheme on the residuals (only applies to
    # forward variables)
    tmp += (
        a_fl_model.a_sources[:, :, time_index].ravel("F")
        / fl_model.storage_coefficient
        / geometry.mesh_area
    )

    # Add the source terms from mob observations (adjoint transport)
    tmp += _get_adjoint_transport_src_terms(
        geometry, fl_model, a_fl_model, time_index, True
    )

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(_q_next, tmp, M=preconditioner, atol=1e-15)
    # Note: we go backward in time, so time_index -1...
    a_fl_model.a_head[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code


def _get_adjoint_transport_src_terms(
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
    # kmean_x: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)

    # Get the permeability between nodes
    kmean_x = harmonic_mean(fl_model.permeability[:-1, :], fl_model.permeability[1:, :])

    src = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)

    # x contribution

    # Forward
    src[:-1, :] -= (
        kmean_x
        * a_fl_model.a_u_darcy_x[:, :, time_index]
        / geometry.dx
        / geometry.mesh_area
    )

    # Backward
    src[1:, :] += (
        kmean_x
        * a_fl_model.a_u_darcy_x[:, :, time_index]
        / geometry.dx
        / geometry.mesh_area
    )

    # y contribution
    if geometry.ny >= 2:
        # Get the permeability between nodes
        kmean_y = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )

        # Forward
        src[:, :-1] -= (
            kmean_y
            * a_fl_model.a_u_darcy_y[:, :, time_index]
            / geometry.dy
            / geometry.mesh_area
        )

        # Backward
        src[:, 1:] += (
            kmean_y
            * a_fl_model.a_u_darcy_y[:, :, time_index]
            / geometry.dy
            / geometry.mesh_area
        )

    # Divide by the storage coefficient only if transient mode
    if is_transient:
        return src.ravel("F") / fl_model.storage_coefficient
    return src.ravel("F")
