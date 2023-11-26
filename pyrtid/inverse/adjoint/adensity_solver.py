"""Implement the adjoint density computation."""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import get_kmean
from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    Geometry,
    TimeParameters,
    TransportModel,
    VerticalAxis,
)
from pyrtid.inverse.adjoint.aflow_solver import get_rhomean_adj
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel


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
    a_tr_model.a_density[:, :, time_index] += (
        a_tr_model.a_density_sources.getcol(time_index)
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
                fl_model, a_fl_model, a_tr_model, time_index, geometry
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
        / (tr_model.ldensity[time_index] ** 2 * GRAVITY)
    )

    # Take into account the equation for n==0 which depends on \rho_{0} as well
    if time_index == 0:
        a_tr_model.a_density[:, :, time_index] -= (
            a_fl_model.a_head[:, :, time_index]
            * fl_model.lpressure[time_index]
            / (tr_model.ldensity[time_index] ** 2 * GRAVITY)
        )


def _add_darcy_contribution(
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    geometry: Geometry,
) -> None:
    # TODO: replace 1/2 by the derivative of the square
    # X contribution
    if fl_model.vertical_axis == VerticalAxis.DX:
        kij = get_kmean(geometry, fl_model, axis=0, is_flatten=False)[:-1, :]
        a_u_darcy_x_old = (
            a_fl_model.a_u_darcy_x[1:-1, :, time_index + 1] * kij / WATER_DENSITY
        ).copy()
        if time_index == 0:
            a_u_darcy_x_old += (
                a_fl_model.a_u_darcy_x[1:-1, :, time_index] * kij / WATER_DENSITY
            )
        # Left
        a_tr_model.a_density[:-1, :, time_index] -= a_u_darcy_x_old * 0.5
        # Right
        a_tr_model.a_density[1:, :, time_index] += a_u_darcy_x_old * 0.5
    # Y Contribution
    elif fl_model.vertical_axis == VerticalAxis.DY:
        kij = get_kmean(geometry, fl_model, axis=1, is_flatten=False)[:, :-1]
        a_u_darcy_y_old = (
            a_fl_model.a_u_darcy_y[:, 1:-1, time_index + 1] * kij / WATER_DENSITY
        ).copy()
        if time_index == 0:
            a_u_darcy_y_old += (
                a_fl_model.a_u_darcy_y[:, 1:-1, time_index] * kij / WATER_DENSITY
            )
        # Up
        a_tr_model.a_density[:, :-1, time_index] -= a_u_darcy_y_old * 0.5
        # Down
        a_tr_model.a_density[:, 1:, time_index] += a_u_darcy_y_old * 0.5


def _add_diffusivity_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    geometry: Geometry,
) -> None:
    """Return the contribution from the derivative of the diffusivity equation."""
    cr_fl: float = fl_model.crank_nicolson

    # Mask the adjoint pressure for the constant head nodes
    apressure_prev = a_fl_model.a_pressure[:, :, time_index + 1]
    ma_ap_prev = np.zeros(apressure_prev.shape)
    free_head_indices = fl_model.free_head_indices
    ma_ap_prev[free_head_indices[0], free_head_indices[1]] = apressure_prev[
        free_head_indices[0], free_head_indices[1]
    ]

    out = np.zeros_like(a_tr_model.a_density[:, :, time_index])
    p_prev = fl_model.lpressure[time_index + 1]
    p_next = fl_model.lpressure[time_index]

    # 1) X contribution
    if geometry.nx > 1:
        kij = get_kmean(geometry, fl_model, axis=0, is_flatten=False)[:-1, :]

        if fl_model.vertical_axis == VerticalAxis.DX:
            drho2g = (
                get_rhomean_adj(geometry, tr_model, 0, time_index, is_flatten=False)[
                    :-1, :
                ]
                * GRAVITY
            )
        else:
            drho2g = 0.0

        # Ici res vaut la même chose en forward et backward parce que les equations
        # ne dépendent que de paramètres IJ
        tmp = geometry.dy * kij / WATER_DENSITY

        # 1.1) Forward
        out[:-1, :] += (
            tmp
            * (
                0.5
                / geometry.dx
                * (
                    cr_fl * (p_prev[1:, :] - p_prev[:-1, :])
                    + (1 + cr_fl) * (p_next[1:, :] - p_next[:-1, :])
                )
                + drho2g
            )
            * (ma_ap_prev[:-1, :] - ma_ap_prev[1:, :])
        )

        # 1.2) Backward
        out[1:, :] += (
            tmp
            * (
                0.5
                / geometry.dx
                * (
                    cr_fl * (p_prev[:-1, :] - p_prev[1:, :])
                    + (1 + cr_fl) * (p_next[:-1, :] - p_next[1:, :])
                )
                - drho2g
            )
            * (ma_ap_prev[1:, :] - ma_ap_prev[:-1, :])
        )

    # 2) Y contribution
    if geometry.ny > 1:
        kij = get_kmean(geometry, fl_model, axis=1, is_flatten=False)[:, :-1]

        if fl_model.vertical_axis == VerticalAxis.DY:
            drho2g = (
                get_rhomean_adj(geometry, tr_model, 1, time_index, is_flatten=False)[
                    :, :-1
                ]
                * GRAVITY
            )
        else:
            drho2g = 0.0

        # Ici res vaut la même chose en forward et backward parce que les equations
        # ne dépendent que de paramètres IJ
        tmp = geometry.dx * kij / WATER_DENSITY

        # 2.1) Forward
        out[:, :-1] += (
            tmp
            * (
                0.5
                / geometry.dy
                * (
                    cr_fl * (p_prev[:, 1:] - p_prev[:, :-1])
                    + (1 + cr_fl) * (p_next[:, 1:] - p_next[:, :-1])
                )
                + drho2g
            )
            * (ma_ap_prev[:, :-1] - ma_ap_prev[:, 1:])
        )

        # 2.2) Backward
        out[:, 1:] += (
            tmp
            * (
                0.5
                / geometry.dy
                * (
                    cr_fl * (p_prev[:, :-1] - p_prev[:, 1:])
                    + (1 + cr_fl) * (p_next[:, :-1] - p_next[:, 1:])
                )
                - drho2g
            )
            * (ma_ap_prev[:, 1:] - ma_ap_prev[:, :-1])
        )

    # Update
    a_tr_model.a_density[:, :, time_index] += out

    # 3) Add unitflow: only for free head nodes
    a_tr_model.a_density[:, :, time_index] += (
        ma_ap_prev
        * GRAVITY
        * (
            cr_fl * fl_model.lunitflow[time_index + 1]
            + (1 - cr_fl) * fl_model.lunitflow[time_index]
        )
    ) * geometry.mesh_volume
