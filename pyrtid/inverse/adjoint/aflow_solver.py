"""Provide an adjoint solver and model."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    FlowModel,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import get_super_lu_preconditioner, harmonic_mean
from pyrtid.utils.types import NDArrayFloat


def make_transient_adj_flow_matrices(
    geometry: Geometry, fl_model: FlowModel, time_params: TimeParameters
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

    # # X contribution
    # kmean = harmonic_mean(fl_model.permeability[:-1, :], fl_model.permeability[1:, :])
    _tmp = geometry.dy / geometry.dx / geometry.mesh_volume

    # 1) X contribution
    if geometry.nx >= 2:
        kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        kmean[:-1, :] = harmonic_mean(
            fl_model.permeability[:-1, :], fl_model.permeability[1:, :]
        )
        kmean = kmean.flatten(order="F")

        # 1.1) Forward scheme:

        # 1.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson * kmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] * tmp
        )  # type: ignore

        # 1.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson * kmean[idc_owner] * tmp
        )  # type: ignore

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] * tmp
        )  # type: ignore

        # 1.2) Backward scheme

        # 1.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson * kmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] * tmp
        )  # type: ignore

        # 1.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson * kmean[idc_neigh] * tmp
        )  # type: ignore

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] * tmp
        )  # type: ignore

    # 2) Y contribution
    if geometry.ny >= 2:
        kmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        kmean[:, :-1] = harmonic_mean(
            fl_model.permeability[:, :-1], fl_model.permeability[:, 1:]
        )
        kmean = kmean.flatten(order="F")

        # 2.1) Forward scheme:
        tmp = geometry.dx / geometry.dy / geometry.mesh_volume

        # 2.1.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson * kmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] * tmp
        )  # type: ignore

        # 2.1.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson * kmean[idc_owner] * tmp
        )  # type: ignore

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_owner] * tmp
        )  # type: ignore

        # 2.2) Backward scheme

        # 2.2.1) For free head nodes only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            owner_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_owner] += (
            fl_model.crank_nicolson * kmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] * tmp
        )  # type: ignore

        # 2.2.2) For all nodes but with free head neighbors only
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            neigh_indices_to_keep=fl_model.free_head_nn,
        )
        # Add the storage coefficient with respect to the owner mesh
        tmp = _tmp / stocoeff[idc_owner]

        q_next[idc_owner, idc_neigh] -= (
            fl_model.crank_nicolson * kmean[idc_neigh] * tmp
        )  # type: ignore

        q_prev[idc_owner, idc_neigh] += (
            (1.0 - fl_model.crank_nicolson) * kmean[idc_neigh] * tmp
        )  # type: ignore

    return q_next, q_prev


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
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    time_params: TimeParameters,
    time_index: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dh/dt = div K grad h + ...
    """
    _q_prev = a_fl_model.q_prev.copy()
    _q_next = a_fl_model.q_next.copy()

    # Add 1/dt for the left term contribution: only for free head
    diag = np.zeros(a_fl_model.a_head[:, :, -1].size)
    diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index])
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
        diag[fl_model.free_head_nn] += float(1.0 / time_params.ldt[time_index + 1])
        diag[fl_model.cst_head_nn] += (
            1.0
            / fl_model.storage_coefficient.ravel("F")[fl_model.cst_head_nn]
            / geometry.mesh_volume
        )
    except IndexError:
        pass

    _q_prev.setdiag(_q_prev.diagonal() + diag)

    # _q_next.setdiag(_q_next.diagonal() + 1 / time_params.ldt[time_index])
    # try:
    #     _q_prev.setdiag(_q_prev.diagonal() + 1 / time_params.ldt[time_index + 1])
    # except IndexError:
    #     pass

    # convert to csc format for efficiency
    _q_next = _q_next.tocsc()
    _q_prev = _q_prev.tocsc()

    # LU preconditioner
    preconditioner = get_super_lu_preconditioner(_q_next)

    # Handle the first time step in the adjoint (= last timestep in the forward)
    # if time_index + 1 != a_fl_model.a_sources.shape[-1]:
    prev_vector = a_fl_model.a_head[:, :, time_index + 1].ravel("F")
    # Multiply prev matrix by prev vector
    tmp = _q_prev.dot(prev_vector)

    # Add the source terms
    # Note: there is no crank-nicolson scheme on the residuals (only applies to
    # forward variables)
    tmp += (
        a_fl_model.a_head_sources.getcol(time_index + 1).todense().ravel("F")
        / fl_model.storage_coefficient.ravel("F")
        / geometry.mesh_volume
    )

    # TODO: check that all works fine with the density (shift in time)
    # Use the adjoint pressure instead of the adjoint head
    # tmp += (
    #     a_fl_model.a_head_sources.getcol(time_params.nts).todense().ravel("F")
    #     / fl_model.storage_coefficient
    #     / geometry.mesh_volume
    # ) * tr_model.density[:, :, time_params.nts] * GRAVITY

    # Add the source terms from mob observations (adjoint transport)
    # tmp += _get_adjoint_transport_src_terms(
    #     geometry, fl_model, a_fl_model, time_index, True
    # )

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
        / geometry.mesh_volume
    )

    # Backward
    src[1:, :] += (
        kmean_x
        * a_fl_model.a_u_darcy_x[:, :, time_index]
        / geometry.dx
        / geometry.mesh_volume
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
            / geometry.mesh_volume
        )

        # Backward
        src[:, 1:] += (
            kmean_y
            * a_fl_model.a_u_darcy_y[:, :, time_index]
            / geometry.dy
            / geometry.mesh_volume
        )

    # Divide by the storage coefficient only if transient mode
    if is_transient:
        return src.ravel("F") / fl_model.storage_coefficient.ravel("F")
    return src.ravel("F")
