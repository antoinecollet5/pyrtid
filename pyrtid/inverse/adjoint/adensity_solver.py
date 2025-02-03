"""Implement the adjoint density computation."""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import get_kmean, get_rhomean
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
from pyrtid.utils import dxi_arithmetic_mean, harmonic_mean
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

            # _add_vertical_gradient_contribution(
            #     fl_model, tr_model, a_fl_model, a_tr_model, time_index, geometry
            # )

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

    if a_fl_model.crank_nicolson is None:
        crank_flow: float = fl_model.crank_nicolson
    else:
        crank_flow = a_fl_model.crank_nicolson

    shape = (geometry.nx, geometry.ny)
    permeability = fl_model.permeability
    free_head_indices = fl_model.free_head_indices

    # Mask the adjoint pressure for the constant head nodes
    ap_prev = a_fl_model.a_pressure[:, :, time_index + 1]
    ap_prev_fhi = np.zeros(ap_prev.shape)
    ap_prev_fhi[free_head_indices[0], free_head_indices[1]] = ap_prev[
        free_head_indices[0], free_head_indices[1]
    ]
    # add the storgae coefficient to ma_apressure
    ap_prev_fhi_sc = ap_prev_fhi / (
        fl_model.storage_coefficient * geometry.grid_cell_volume
    )

    pprev = fl_model.pressure[:, :, time_index + 1]
    pnext = fl_model.pressure[:, :, time_index]

    contrib = np.zeros(shape)

    # vp = fl_model._get_mesh_center_vertical_pos().T

    # # Consider the y axis for 2D cases
    # if geometry.nx > 1:
    #     drhomean_x = get_drhomean(
    #         geometry,
    #         tr_model,
    #         axis=0,
    #         time_index=time_index,
    #         is_flatten=False,
    #     )[:-1, :]
    #     tmp = 0.0
    #     # if fl_model.vertical_axis == VerticalAxis.X:
    #     #     tmp = 2 * rhomean_x**2 * GRAVITY

    #     # Forward scheme
    #     dpressure_fx = (
    #         (
    #             (
    #                 crank_flow * (pprev[1:, :] - pprev[:-1, :])
    #                 + (1.0 - crank_flow) * (pnext[1:, :] - pnext[:-1, :])
    #             )
    #             / geometry.dx
    #             * drhomean_x
    #             + tmp
    #         )
    #         * harmonic_mean(permeability[:-1, :], permeability[1:, :])
    #         / WATER_DENSITY
    #     )

    #     contrib[:-1, :] += (
    #         dpressure_fx
    #         * (ap_prev_fhi_sc[:-1, :] - ap_prev_fhi_sc[1:, :])
    #         * geometry.gamma_ij_x
    #     )

    #     # Backward scheme
    #     dpressure_bx = (
    #         (
    #             (
    #                 crank_flow * (pprev[:-1, :] - pprev[1:, :])
    #                 + (1.0 - crank_flow) * (pnext[:-1, :] - pnext[1:, :])
    #             )
    #             / geometry.dx
    #             * drhomean_x
    #             - tmp
    #         )
    #         * harmonic_mean(permeability[1:, :], permeability[:-1, :])
    #         / WATER_DENSITY
    #     )

    #     contrib[1:, :] += (
    #         dpressure_bx
    #         * (ap_prev_fhi_sc[1:, :] - ap_prev_fhi_sc[:-1, :])
    #         * geometry.gamma_ij_x
    #     )

    # Consider the y axis for 2D cases
    if geometry.ny > 1:
        drhomean_y = get_drhomean(
            geometry,
            tr_model,
            axis=1,
            time_index=time_index,
            is_flatten=False,
        )[:, :-1]
        tmp = 0.0
        # if fl_model.vertical_axis == VerticalAxis.Y:
        #     rhomean_y = get_rhomean(
        #         geometry,
        #         tr_model,
        #         axis=1,
        #         time_index=time_index - 1,
        #         is_flatten=False,
        #     )[:, :-1]
        #     tmp = 2 * rhomean_y * drhomean_y * GRAVITY

        # Forward scheme
        dpressure_fy = (
            (
                (
                    crank_flow * (pprev[:, 1:] - pprev[:, :-1])
                    + (1.0 - crank_flow) * (pnext[:, 1:] - pnext[:, :-1])
                )
                / geometry.dy
                * drhomean_y
                + tmp
            )
            * harmonic_mean(permeability[:, :-1], permeability[:, 1:])
            / WATER_DENSITY
        )

        contrib[:, :-1] += (
            dpressure_fy
            * (ap_prev_fhi_sc[:, :-1] - ap_prev_fhi_sc[:, 1:])
            * geometry.gamma_ij_y
        )

        # # Handle the stationary case
        # if fl_model.regime == FlowRegime.STATIONARY:
        #     grad[:, :-1, :1] += (
        #         (
        #             (pressure[:, 1:, :1] - pressure[:, :-1, :1])
        #             / WATER_DENSITY
        #             / GRAVITY
        #             + vp[:, 1:, np.newaxis]
        #             - vp[:, :-1, np.newaxis]
        #         )
        #         * harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
        #             :, :, np.newaxis
        #         ]
        #         * geometry.gamma_ij_y
        #         / geometry.dy
        #         * (apressure[:, 1:, :1] - ma_apressure[:, :-1, :1])
        #         / geometry.grid_cell_volume
        #     )

        # Backward scheme
        dpressure_by = (
            (
                (
                    crank_flow * (pprev[:, :-1] - pprev[:, 1:])
                    + (1.0 - crank_flow) * (pnext[:, :-1] - pnext[:, 1:])
                )
                / geometry.dy
                * drhomean_y
                - tmp
            )
            * harmonic_mean(permeability[:, 1:], permeability[:, :-1])
            / WATER_DENSITY
        )

        contrib[:, 1:] += (
            dpressure_by
            * (ap_prev_fhi_sc[:, 1:] - ap_prev_fhi_sc[:, :-1])
            * geometry.gamma_ij_y
        )

    a_tr_model.a_density[:, :, time_index] += contrib.reshape(geometry.shape, order="F")

    # 3) Add unitflow: only for free head nodes
    a_tr_model.a_density[:, :, time_index] += (
        ap_prev_fhi
        * GRAVITY
        * (
            fl_crank * fl_model.lunitflow[time_index + 1]
            + (1.0 - fl_crank) * fl_model.lunitflow[time_index]
        )
    ) / fl_model.storage_coefficient


def _add_vertical_gradient_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    geometry: Geometry,
) -> None:
    """Return the contribution from the derivative of the diffusivity equation."""
    shape = (geometry.nx, geometry.ny)
    permeability = fl_model.permeability
    free_head_indices = fl_model.free_head_indices

    # Mask the adjoint pressure for the constant head nodes
    ap_prev = a_fl_model.a_pressure[:, :, time_index + 1]
    ap_prev_fhi = np.zeros(ap_prev.shape)
    ap_prev_fhi[free_head_indices[0], free_head_indices[1]] = ap_prev[
        free_head_indices[0], free_head_indices[1]
    ]
    # add the storgae coefficient to ma_apressure
    ap_prev_fhi_sc = ap_prev_fhi / (
        fl_model.storage_coefficient * geometry.grid_cell_volume
    )

    contrib = np.zeros(shape)

    # vp = fl_model._get_mesh_center_vertical_pos().T

    # Consider the y axis for 2D cases
    if geometry.ny > 1:
        drhomean_y = get_drhomean(
            geometry,
            tr_model,
            axis=1,
            time_index=time_index,
            is_flatten=False,
        )[:, :-1]
        tmp = 0.0
        if fl_model.vertical_axis == VerticalAxis.Y:
            rhomean_y = get_rhomean(
                geometry,
                tr_model,
                axis=1,
                time_index=time_index,
                is_flatten=False,
            )[:, :-1]
            tmp = 2 * rhomean_y * drhomean_y * GRAVITY

        # Forward scheme
        dpressure_fy = (
            (+tmp)
            * harmonic_mean(permeability[:, :-1], permeability[:, 1:])
            / WATER_DENSITY
        )

        contrib[:, :-1] += (
            dpressure_fy
            * (ap_prev_fhi_sc[:, :-1] - ap_prev_fhi_sc[:, 1:])
            * geometry.gamma_ij_y
        )

        # Backward scheme
        dpressure_by = (
            (-tmp)
            * harmonic_mean(permeability[:, 1:], permeability[:, :-1])
            / WATER_DENSITY
        )

        contrib[:, 1:] += (
            dpressure_by
            * (ap_prev_fhi_sc[:, 1:] - ap_prev_fhi_sc[:, :-1])
            * geometry.gamma_ij_y
        )

    a_tr_model.a_density[:, :, time_index] += contrib
