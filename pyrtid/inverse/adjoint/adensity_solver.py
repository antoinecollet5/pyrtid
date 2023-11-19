"""Implement the adjoint density computation."""

from __future__ import annotations

import numpy as np

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

    # This is the case if gravity is not on

    # Handle the first time step
    try:
        # Adjoint variables
        apressure: NDArrayFloat = a_fl_model.a_pressure[:, :, time_index + 1]
        head: NDArrayFloat = fl_model.head[:, :, time_index + 1]
    except IndexError:
        # Handle the Tmax (first timestep going backward)
        # or adjoint state initialization
        apressure = np.array([0.0])
        head = np.array([0.0])

    # TODO: this is only valid if the gravity is on
    a_tr_model.a_density[:, :, time_index] = (
        -apressure * (head - fl_model._get_mesh_center_vertical_pos().T) * GRAVITY
    ) + a_tr_model.a_density_sources.getcol(time_index).reshape(shape, order="F")

    # Create a density src term for the transport
    a_tr_model.a_density_src_term = (
        -a_tr_model.a_density[:, :, time_index]
        * WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * mw
        / 1000
    )
