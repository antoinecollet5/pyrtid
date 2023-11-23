"""Implement the adjoint density computation."""

from __future__ import annotations

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GRAVITY,
    TDS_LINEAR_COEFFICIENT,
    WATER_DENSITY,
    FlowModel,
    TransportModel,
)
from pyrtid.inverse.adjoint.amodels import AdjointFlowModel, AdjointTransportModel
from pyrtid.utils import NDArrayFloat


def solve_adj_density(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_fl_model: AdjointFlowModel,
    a_tr_model: AdjointTransportModel,
    time_index: int,
    mw: float,
) -> None:
    shape = tr_model.ldensity[0].shape

    # Add the density observations derivative
    a_tr_model.a_density[:, :, time_index] += (
        a_tr_model.a_density_sources.getcol(time_index)
        .todense()
        .reshape(shape, order="F")
    )

    if fl_model.is_gravity:
        # Handle the first time step
        try:
            # Adjoint variables
            ahead: NDArrayFloat = a_fl_model.a_head[:, :, time_index + 1]
            pressure: NDArrayFloat = fl_model.lpressure[time_index + 1]
            density = tr_model.ldensity[time_index]

            # 1) Contribution from the head equation
            a_tr_model.a_density[:, :, time_index] -= (
                ahead * pressure / (density**2 * GRAVITY)
            )
            # 2) Contribution from the darcy equation
            a_tr_model.a_density[:, :, time_index] += 0.0

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
