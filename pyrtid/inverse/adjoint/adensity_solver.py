"""Implement the adjoint density computation."""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import get_kmean
from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    FlowRegime,
    Geometry,
    TimeParameters,
    TransportModel,
    VerticalAxis,
)
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import dxi_arithmetic_mean
from pyrtid.utils.types import NDArrayFloat


def get_drhomean(
    geometry: Geometry,
    tr_model: TransportModel,
    axis: int,
    time_index: int,
    is_flatten: bool = True,
) -> NDArrayFloat:
    drhomean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
    if axis == 0:
        drhomean[:-1, :] = dxi_arithmetic_mean(
            tr_model.ldensity[time_index][:-1, :], tr_model.ldensity[time_index][1:, :]
        )
    else:
        drhomean[:, :-1] = dxi_arithmetic_mean(
            tr_model.ldensity[time_index][:, :-1], tr_model.ldensity[time_index][:, 1:]
        )

    if is_flatten:
        return drhomean.flatten(order="F")
    return drhomean


def get_drhomean2(
    geometry: Geometry,
    tr_model: TransportModel,
    axis: int,
    time_index: int,
    is_flatten: bool = True,
) -> NDArrayFloat:
    return get_drhomean(geometry, tr_model, axis, time_index, is_flatten) * 0.0


def solve_adj_density(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    time_params: TimeParameters,
    geometry: Geometry,
    mw: float,
) -> None:
    shape = tr_model.ldensity[0].shape

    # Add the density observations derivative (adjoint source term)
    a_tr_model.a_density[:, :, time_index] -= (
        a_tr_model.a_density_sources[:, [time_index]]
        .todense()
        .reshape(shape, order="F")
    )

    if fl_model.is_gravity:
        # Handle the first time step
        # Handle the Tmax (first timestep going backward)
        # or adjoint state initialization
        if time_index != time_params.nts:
            # 1) Contribution from the head equation
            _add_head_equation_contribution(
                fl_model, tr_model, a_fl_model, a_tr_model, time_index
            )

            # 2) Contribution from the darcy equation
            _add_darcy_contribution(
                fl_model, tr_model, a_fl_model, a_tr_model, time_index, geometry
            )

            # 3) Contribution from the diffusivity equation
            _add_diffusivity_contribution(
                fl_model, tr_model, a_fl_model, a_tr_model, time_index, geometry
            )


def _add_head_equation_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
) -> None:
    a_tr_model.a_density[:, :, time_index] -= (
        a_fl_model.a_head[:, :, time_index + 1]
        * fl_model.lpressure[time_index + 1]
        / (tr_model.ldensity[time_index] ** 2)
        / GRAVITY
    )

    # Take into account the equation for n==0 which depends on \rho_{0} as well
    if time_index == 0 and fl_model.regime == FlowRegime.STATIONARY:
        a_tr_model.a_density[:, :, time_index] -= (
            a_fl_model.a_head[:, :, time_index]
            * fl_model.lpressure[time_index]
            / (tr_model.ldensity[time_index] ** 2)
            / GRAVITY
        )


def _add_darcy_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    geometry: Geometry,
) -> None:
    # X contribution
    if fl_model.vertical_axis == VerticalAxis.X:
        kij = get_kmean(geometry, fl_model, axis=0, is_flatten=False)[:-1, :]
        a_u_darcy_x_old = (
            a_fl_model.a_u_darcy_x[1:-1, :, time_index + 1] * kij / WATER_DENSITY
        )
        if time_index == 0 and fl_model.regime == FlowRegime.STATIONARY:
            a_u_darcy_x_old += (
                a_fl_model.a_u_darcy_x[1:-1, :, time_index] * kij / WATER_DENSITY
            )
        drhomean = get_drhomean(
            geometry, tr_model, axis=0, time_index=time_index, is_flatten=False
        )[:-1, :]
        # Left
        a_tr_model.a_density[:-1, :, time_index] -= a_u_darcy_x_old * drhomean
        # Right
        a_tr_model.a_density[1:, :, time_index] += a_u_darcy_x_old * drhomean
    # Y Contribution
    elif fl_model.vertical_axis == VerticalAxis.Y:
        kij = get_kmean(geometry, fl_model, axis=1, is_flatten=False)[:, :-1]
        a_u_darcy_y_old = (
            a_fl_model.a_u_darcy_y[:, 1:-1, time_index + 1] * kij / WATER_DENSITY
        )
        if time_index == 0 and fl_model.regime == FlowRegime.STATIONARY:
            a_u_darcy_y_old += (
                a_fl_model.a_u_darcy_y[:, 1:-1, time_index] * kij / WATER_DENSITY
            )
        drhomean = get_drhomean(
            geometry, tr_model, axis=1, time_index=time_index, is_flatten=False
        )[:, :-1]
        # Up
        a_tr_model.a_density[:, :-1, time_index] -= a_u_darcy_y_old * drhomean
        # Down
        a_tr_model.a_density[:, 1:, time_index] += a_u_darcy_y_old * drhomean


def _add_diffusivity_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    geometry: Geometry,
) -> None:
    """Return the contribution from the derivative of the diffusivity equation."""
    if a_fl_model.crank_nicolson is None:
        fl_crank: float = fl_model.crank_nicolson
    else:
        fl_crank = a_fl_model.crank_nicolson

    # Mask the adjoint pressure for the constant head nodes
    ap_prev = a_fl_model.a_pressure[:, :, time_index + 1]
    ap_prev_fhi = np.zeros(ap_prev.shape)
    free_head_indices = fl_model.free_head_indices
    ap_prev_fhi[free_head_indices[0], free_head_indices[1]] = ap_prev[
        free_head_indices[0], free_head_indices[1]
    ]

    # a_prev = a_fl_model.a_pressure[:, :, time_index + 1].ravel("F")
    # p_prev = fl_model.lpressure[time_index + 1].ravel(order="F")
    # p_next = fl_model.lpressure[time_index].ravel(order="F")

    # contrib = np.zeros(ap_prev.size)
    # # 1) X contribution
    # if geometry.nx >= 2:
    #     _tmp = geometry.gamma_ij_x / geometry.dx
    #     kmean = get_kmean(geometry, fl_model, 0)
    #     drhomean = get_drhomean2(
    #         geometry, tr_model, axis=0, time_index=time_index, is_flatten=False
    #     ).ravel("F")

    #     # 1.1.1) For free head nodes only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(0, geometry.nx - 1), slice(None)),
    #         (slice(1, geometry.nx), slice(None)),
    #         owner_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_owner] / WATER_DENSITY

    #     contrib[idc_owner] += (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_owner]
    #     )  # type: ignore

    #     # 1.1.2) For all nodes but with free head neighbors only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(0, geometry.nx - 1), slice(None)),
    #         (slice(1, geometry.nx), slice(None)),
    #         neigh_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_owner] / WATER_DENSITY

    #     contrib[idc_owner] -= (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_neigh]
    #     )

    #     # 1.2) Backward scheme

    #     # 1.2.1) For free head nodes only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(1, geometry.nx), slice(None)),
    #         (slice(0, geometry.nx - 1), slice(None)),
    #         owner_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_neigh] / WATER_DENSITY

    #     contrib[idc_owner] -= (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_owner]
    #     )

    #     # 1.2.2) For all nodes but with free head neighbors only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(1, geometry.nx), slice(None)),
    #         (slice(0, geometry.nx - 1), slice(None)),
    #         neigh_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_neigh] / WATER_DENSITY

    #     contrib[idc_owner] += (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_neigh]
    #     )

    # # 2) Y contribution
    # if geometry.ny >= 2:
    #     kmean = get_kmean(geometry, fl_model, 1)
    #     _tmp = geometry.gamma_ij_y / geometry.dy
    #     drhomean = get_drhomean2(
    #         geometry, tr_model, axis=1, time_index=time_index, is_flatten=False
    #     ).ravel("F")
    #     # 2.1.1) For free head nodes only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(None), slice(0, geometry.ny - 1)),
    #         (slice(None), slice(1, geometry.ny)),
    #         owner_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_owner] / WATER_DENSITY

    #     contrib[idc_owner] += (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_owner]
    #     )

    #     # 2.1.2) For all nodes but with free head neighbors only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(None), slice(0, geometry.ny - 1)),
    #         (slice(None), slice(1, geometry.ny)),
    #         neigh_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_owner] / WATER_DENSITY

    #     contrib[idc_owner] -= (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_neigh]
    #     )

    #     # 2.2) Backward scheme

    #     # 2.2.1) For free head nodes only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(None), slice(1, geometry.ny)),
    #         (slice(None), slice(0, geometry.ny - 1)),
    #         owner_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_neigh] / WATER_DENSITY

    #     contrib[idc_owner] -= (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_owner]
    #     )

    #     # 2.2.2) For all nodes but with free head neighbors only
    #     idc_owner, idc_neigh = get_owner_neigh_indices(
    #         geometry,
    #         (slice(None), slice(1, geometry.ny)),
    #         (slice(None), slice(0, geometry.ny - 1)),
    #         neigh_indices_to_keep=fl_model.free_head_nn,
    #     )
    #     # Add the storage coefficient with respect to the owner mesh
    #     tmp = _tmp * kmean[idc_neigh] / WATER_DENSITY

    #     contrib[idc_owner] += (
    #         drhomean[idc_owner]
    #         * (
    #             fl_crank * (p_prev[idc_neigh] - p_prev[idc_owner])
    #             + (1.0 - fl_crank) * (p_next[idc_neigh] - p_next[idc_owner])
    #         )
    #         * tmp
    #         * a_prev[idc_neigh]
    #     )

    # a_tr_model.a_density[:, :, time_index] += contrib.reshape(
    # geometry.shape, order="F")

    # 3) Add unitflow: only for free head nodes
    a_tr_model.a_density[:, :, time_index] += (
        ap_prev_fhi
        * GRAVITY
        * (
            fl_crank * fl_model.lunitflow[time_index + 1]
            + (1.0 - fl_crank) * fl_model.lunitflow[time_index]
        )
    ) / fl_model.storage_coefficient
