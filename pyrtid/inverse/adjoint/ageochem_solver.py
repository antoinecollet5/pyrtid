"""Provide an adjoint solver for the geochemistry."""
from __future__ import annotations

from copy import copy

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GeochemicalParameters,
    Geometry,
    TimeParameters,
    TransportModel,
)
from pyrtid.inverse.adjoint.amodels import AdjointTransportModel


def solve_adj_geochem(
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    gch_params: GeochemicalParameters,
    geometry: Geometry,
    time_params: TimeParameters,
    time_index: int,
    nafpi: int,
) -> None:
    # Adjoint variable
    try:
        am_old = a_tr_model.a_grade[:, :, time_index + 1]
        ac_old = a_tr_model.a_mob[0, :, :, time_index + 1]
        c_old = tr_model.lmob[time_index + 1][0]
        dt_cur = time_params.ldt[time_index]
    except IndexError:
        # Handle the Tmax (first timestep going backward)
        # or adjoint state initialization
        am_old = 0.0
        ac_old = 0.0
        c_old = 1.0
        dt_cur = 1.0  # should be zero but we avoid a zero division here

    # Adjoint concentration at current timestep
    if (
        a_tr_model.is_adj_numerical_acceleration
        and nafpi == 1
        and time_index != time_params.nts
    ):
        ac_cur = a_tr_model.a_mob[:, :, :, time_index + 1]
    else:
        ac_cur = a_tr_model.a_mob[:, :, :, time_index]

    # Save the grade for the fix point iterations
    a_tr_model.a_mob_prev = copy(ac_cur)

    # Forward variables

    # Timesteps
    dt_next = time_params.ldt[time_index - 1]

    # Update mineral value
    a_tr_model.a_grade[:, :, time_index] = (
        am_old
        * (1 + dt_cur * gch_params.kv * gch_params.As * (1.0 - c_old / gch_params.Ks))
        - (ac_cur[0] / dt_next - ac_old / dt_cur)
        * tr_model.porosity
        * geometry.mesh_volume
    ) + a_tr_model.a_grade_sources.getcol(time_index).reshape(
        a_tr_model.a_grade[:, :, time_index].shape, order="F"
    )

    # Compute the adjoint geochem source term: it is computed here to mimic the
    # splitting operator approach in which the chemical parameters might not be
    # available in the transport operator - and consequently its adjoint.
    a_tr_model.a_gch_src_term = (
        a_tr_model.a_grade[:, :, time_index]
        * dt_next
        * gch_params.kv
        * gch_params.As
        * (tr_model.limmob[time_index - 1] / gch_params.Ks)
    )
