"""Implement the adjoint density computation."""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import get_kmean, get_rhomean
from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GRAVITY,
    WATER_DENSITY,
    FlowModel,
    TimeParameters,
    TransportModel,
    VerticalAxis,
)
from pyrtid.inverse.asm.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import (
    NDArrayFloat,
    RectilinearGrid,
    dxi_arithmetic_mean,
    harmonic_mean,
)


def get_drhomean(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    axis: int,
    time_index: int,
    is_flatten: bool = True,
) -> NDArrayFloat:
    drhomean: NDArrayFloat = np.zeros((grid.nx, grid.ny), dtype=np.float64)
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
    grid: RectilinearGrid,
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
                fl_model, tr_model, a_fl_model, a_tr_model, time_index, grid
            )

            # 3) Contribution from the diffusivity equation
            _add_diffusivity_contribution(
                fl_model, tr_model, a_fl_model, a_tr_model, time_index, grid
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


def _add_darcy_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    grid: RectilinearGrid,
) -> None:
    # X contribution
    if fl_model.vertical_axis == VerticalAxis.X:
        kij = get_kmean(grid, fl_model, axis=0, is_flatten=False)[:-1, :]
        a_u_darcy_x_old = (
            a_fl_model.a_u_darcy_x[1:-1, :, time_index + 1] * kij / WATER_DENSITY
        )
        drhomean = get_drhomean(
            grid, tr_model, axis=0, time_index=time_index, is_flatten=False
        )[:-1, :]
        # Left
        a_tr_model.a_density[:-1, :, time_index] -= a_u_darcy_x_old * drhomean
        # Right
        a_tr_model.a_density[1:, :, time_index] -= a_u_darcy_x_old * drhomean
    # Y Contribution
    elif fl_model.vertical_axis == VerticalAxis.Y:
        kij = get_kmean(grid, fl_model, axis=1, is_flatten=False)[:, :-1]
        a_u_darcy_y_old = (
            a_fl_model.a_u_darcy_y[:, 1:-1, time_index + 1] * kij / WATER_DENSITY
        )
        drhomean = get_drhomean(
            grid, tr_model, axis=1, time_index=time_index, is_flatten=False
        )[:, :-1]
        # Up
        a_tr_model.a_density[:, :-1, time_index] -= a_u_darcy_y_old * drhomean
        # Down
        a_tr_model.a_density[:, 1:, time_index] -= a_u_darcy_y_old * drhomean


def _add_diffusivity_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    grid: RectilinearGrid,
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

    shape = (grid.nx, grid.ny)
    permeability = fl_model.permeability
    free_head_indices = fl_model.free_head_indices

    # add the storgae coefficient to ma_apressure
    apressure = a_fl_model.a_pressure[:, :, time_index + 1]

    # Mask the adjoint pressure for the constant head nodes
    ma_apressure = np.zeros(apressure.shape)
    ma_apressure[free_head_indices[0], free_head_indices[1]] = apressure[
        free_head_indices[0], free_head_indices[1]
    ]
    # add the storgae coefficient to ma_apressure
    ma_apressure_sc = ma_apressure / (
        fl_model.storage_coefficient[:, :] * grid.grid_cell_volume
    )

    pprev = fl_model.pressure[:, :, time_index + 1]
    pnext = fl_model.pressure[:, :, time_index]

    contrib = np.zeros(shape)

    # Consider the y axis for 2D cases
    if grid.nx > 1:
        drhomean_x = get_drhomean(
            grid,
            tr_model,
            axis=0,
            time_index=time_index,
            is_flatten=False,
        )[:-1, :]
        tmp = np.zeros_like(drhomean_x)
        if fl_model.vertical_axis == VerticalAxis.X:
            rhomean_x = get_rhomean(
                grid,
                tr_model,
                axis=0,
                time_index=time_index,
                is_flatten=False,
            )[:-1, :]
            tmp = GRAVITY * 2.0 * drhomean_x * rhomean_x

        # Forward scheme
        dpressure_fx = (
            (
                (
                    crank_flow * (pprev[1:, :] - pprev[:-1, :])
                    + (1.0 - crank_flow) * (pnext[1:, :] - pnext[:-1, :])
                )
                / grid.dx
                * drhomean_x
                + tmp
            )
            * harmonic_mean(permeability[:-1, :], permeability[1:, :])
            / WATER_DENSITY
        )

        contrib[:-1, :] += (
            dpressure_fx
            * (ma_apressure_sc[:-1, :] - ma_apressure_sc[1:, :])
            * grid.gamma_ij_x
        )

        # Backward scheme
        dpressure_bx = (
            (
                (
                    crank_flow * (pprev[:-1, :] - pprev[1:, :])
                    + (1.0 - crank_flow) * (pnext[:-1, :] - pnext[1:, :])
                )
                / grid.dx
                * drhomean_x
                - tmp
            )
            * harmonic_mean(permeability[1:, :], permeability[:-1, :])
            / WATER_DENSITY
        )

        contrib[1:, :] += (
            dpressure_bx
            * (ma_apressure_sc[1:, :] - ma_apressure_sc[:-1, :])
            * grid.gamma_ij_x
        )

    # Consider the y axis for 2D cases
    if grid.ny > 1:
        drhomean_y = get_drhomean(
            grid,
            tr_model,
            axis=1,
            time_index=time_index,
            is_flatten=False,
        )[:, :-1]
        tmp = np.zeros_like(drhomean_y)
        if fl_model.vertical_axis == VerticalAxis.Y:
            rhomean_y = get_rhomean(
                grid,
                tr_model,
                axis=1,
                time_index=time_index,
                is_flatten=False,
            )[:, :-1]
            tmp = GRAVITY * 2.0 * drhomean_y * rhomean_y

        # Forward scheme
        dpressure_fy = (
            (
                (
                    crank_flow * (pprev[:, 1:] - pprev[:, :-1])
                    + (1.0 - crank_flow) * (pnext[:, 1:] - pnext[:, :-1])
                )
                / grid.dy
                * drhomean_y
                + tmp
            )
            * harmonic_mean(permeability[:, :-1], permeability[:, 1:])
            / WATER_DENSITY
        )

        contrib[:, :-1] += (
            dpressure_fy
            * (ma_apressure_sc[:, :-1] - ma_apressure_sc[:, 1:])
            * grid.gamma_ij_y
        )

        # Backward scheme
        dpressure_by = (
            (
                (
                    crank_flow * (pprev[:, :-1] - pprev[:, 1:])
                    + (1.0 - crank_flow) * (pnext[:, :-1] - pnext[:, 1:])
                )
                / grid.dy
                * drhomean_y
                - tmp
            )
            * harmonic_mean(permeability[:, 1:], permeability[:, :-1])
            / WATER_DENSITY
        )

        contrib[:, 1:] += (
            dpressure_by
            * (ma_apressure_sc[:, 1:] - ma_apressure_sc[:, :-1])
            * grid.gamma_ij_y
        )

    a_tr_model.a_density[:, :, time_index] += contrib.reshape(grid.shape2d, order="F")

    # 3) Add unitflow: only for free head nodes
    a_tr_model.a_density[:, :, time_index] += (
        ma_apressure
        * GRAVITY
        * (
            fl_crank * fl_model.lunitflow[time_index + 1]
            + (1.0 - fl_crank) * fl_model.lunitflow[time_index]
        )
    ) / fl_model.storage_coefficient
