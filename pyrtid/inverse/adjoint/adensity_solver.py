"""Implement the adjoint density computation."""

from __future__ import annotations

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
    crank_fl: float = fl_model.crank_nicolson

    # Add the density observations derivative (adjoint source term)
    a_tr_model.a_density[:, :, time_index] += (
        a_tr_model.a_density_sources.getcol(time_index)
        .todense()
        .reshape(shape, order="F")
    )

    if fl_model.is_gravity:
        # Handle the first time step
        try:
            # Adjoint variables
            ahead_old: NDArrayFloat = a_fl_model.a_head[:, :, time_index + 1]
            pressure_old: NDArrayFloat = fl_model.lpressure[time_index + 1]
            density = tr_model.ldensity[time_index]

            # 1) Contribution from the head equation
            a_tr_model.a_density[:, :, time_index] -= (
                ahead_old * pressure_old / (density**2 * GRAVITY)
            )
            # 2) Contribution from the darcy equation
            # X contribution
            if fl_model.vertical_axis == VerticalAxis.DX:
                a_u_darcy_x_old = a_fl_model.a_u_darcy_x[:, :, time_index + 1]
                # Left
                a_tr_model.a_density[:, :, time_index] -= a_u_darcy_x_old[1:, :] * 1 / 2
                # Right
                a_tr_model.a_density[:, :, time_index] += (
                    a_u_darcy_x_old[:-1, :] * 1 / 2
                )
            # Y Contribution
            elif fl_model.vertical_axis == VerticalAxis.DY:
                a_u_darcy_y_old = a_fl_model.a_u_darcy_y[:, :, time_index + 1]
                # Up
                a_tr_model.a_density[:, :, time_index] -= a_u_darcy_y_old[:, 1:] * 1 / 2
                # Down
                a_tr_model.a_density[:, :, time_index] += (
                    a_u_darcy_y_old[:, :-1] * 1 / 2
                )

            # 3) Diffusivity equation contribution
            get_kmean(geometry, fl_model, 0)
            get_rhomean_adj(geometry, tr_model, 0, time_index)

            # 4) Flowartes Qi from the diffusivity equation
            a_tr_model.a_density[:, :, time_index] = (
                pressure_old
                * GRAVITY
                * (
                    crank_fl * fl_model.lunitflow[time_index + 1]
                    + (1 - crank_fl) * fl_model.lunitflow[time_index + 1]
                )
            )

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


# def get_diffusivity_contribution(
#     fl_model: FlowModel,
#     tr_model: TransportModel,
#     a_fl_model: AdjointFlowModel,
#     a_tr_model: AdjointTransportModel,
#     time_index: int,
#     geometry: Geometry,
# ) -> NDArrayFloat:
#     """Return the contribution from the derivative of the diffusivity equation."""
#     out = np.zeros_like(65)
