"""Implement the adjoint density computation."""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import get_kmean
from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GRAVITY,
    TDS_LINEAR_COEFFICIENT,
    WATER_DENSITY,
    FlowModel,
    Geometry,
    TransportModel,
    VerticalAxis,
)
from pyrtid.inverse.adjoint.aflow_solver import get_rhomean_adj
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import NDArrayFloat


def solve_adj_density(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
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
        try:
            # 1) Contribution from the head equation
            _add_head_equation_contribution(
                fl_model, tr_model, a_fl_model, a_tr_model, time_index
            )

            # 2) Contribution from the darcy equation
            _add_darcy_contribution(fl_model, a_fl_model, a_tr_model, time_index)

            # 3) Diffusivity equation contribution
            get_kmean(geometry, fl_model, 0)
            get_rhomean_adj(geometry, tr_model, 0, time_index)
            # 3) Contribution from the diffusivity equation

        except IndexError:
            # Handle the Tmax (first timestep going backward)
            # or adjoint state initialization
            pass

    # Create a density src term for the transport
    a_tr_model.a_density_src_term = (
        -a_tr_model.a_density[:, :, time_index]
        * WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * mw
        / 1000
    )


def _add_head_equation_contribution(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
) -> None:
    # Adjoint variables
    ahead_old: NDArrayFloat = a_fl_model.a_head[:, :, time_index + 1]
    pressure_old: NDArrayFloat = fl_model.lpressure[time_index + 1]
    density = tr_model.ldensity[time_index]

    # 1) Contribution from the head equation
    a_tr_model.a_density[:, :, time_index] -= (
        ahead_old * pressure_old / (density**2 * GRAVITY)
    )


def _add_darcy_contribution(
    fl_model: FlowModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
) -> None:
    # X contribution
    if fl_model.vertical_axis == VerticalAxis.DX:
        a_u_darcy_x_old = a_fl_model.a_u_darcy_x[:, :, time_index + 1]
        # Left
        a_tr_model.a_density[:, :, time_index] -= a_u_darcy_x_old[1:, :] * 1 / 2
        # Right
        a_tr_model.a_density[:, :, time_index] += a_u_darcy_x_old[:-1, :] * 1 / 2
    # Y Contribution
    elif fl_model.vertical_axis == VerticalAxis.DY:
        a_u_darcy_y_old = a_fl_model.a_u_darcy_y[:, :, time_index + 1]
        # Up
        a_tr_model.a_density[:, :, time_index] -= a_u_darcy_y_old[:, 1:] * 1 / 2
        # Down
        a_tr_model.a_density[:, :, time_index] += a_u_darcy_y_old[:, :-1] * 1 / 2


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

    out = np.zeros_like(a_tr_model.a_density[:, :, time_index])
    ap_old = a_fl_model.a_pressure[:, :, time_index + 1]
    p_old = fl_model.lpressure[time_index + 1]
    p_new = fl_model.lpressure[time_index]

    # 1) X contribution
    if geometry.nx > 1:
        kij = get_kmean(geometry, fl_model, axis=0)

        # Ici res vaut la même chose en forward et backward parce que les equations
        # ne dépendent que de paramètres IJ
        tmp = geometry.ny * 0.5 * kij * WATER_DENSITY
        _res = (
            tmp
            * (
                cr_fl * p_old[1:, :]
                - p_old[:-1, :]
                + (1 + cr_fl) * p_new[1:, :]
                - p_new[:-1, :]
            )
            * (ap_old[1:, :] - ap_old[:-1, :])
        )
        # 1.1) Forward
        out[:-1, :] += _res

        # 1.2) Backward
        out[1:, :] += _res

    # 2) Y contribution
    if geometry.ny > 1:
        pass

        # 2.1) Forward
        out += geometry.nx * 0.5

        # 2.2) Backward
        out += geometry.nx * 0.5

    # Update
    a_tr_model.a_density[:, :, time_index] += out

    # 3) Add unitflow: only for non constant conc nodes ?
    a_tr_model.a_density[:, :, time_index] -= (
        a_fl_model.a_pressure[:, :, time_index + 1]
        * GRAVITY
        * (
            cr_fl * fl_model.lunitflow[time_index + 1]
            + (1 - cr_fl) * fl_model.lunitflow[time_index]
        )
    )
