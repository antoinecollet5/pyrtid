"""Provide an adjoint solver for the geochemistry."""
from __future__ import annotations

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
) -> None:
    """Compute the geochemistry part."""
    # Skip the last timestep (there is no transport between n=0 and n=1)
    if time_index == time_params.nt:
        return

    # Forward variable at the current timestep going backward
    # (timestep 2n -> we want to predict)
    m_cur = tr_model.grade[:, :, time_index]
    c_prev_post_tr = tr_model.conc_post_tr[:, :, time_index + 1]  # 2n + 1
    dt_cur = time_params.ldt[time_index - 1]

    # Adjoint variable at the previous timestep going backward (timestep 2n + 2)
    am_prev = a_tr_model.a_grade[:, :, time_index + 1]
    ac_prev = a_tr_model.a_conc[:, :, time_index + 1]
    dt_prev = time_params.ldt[time_index]

    # Update adjoint concentration
    a_tr_model.a_conc_post_gch[:, :, time_index] = (
        ac_prev / dt_prev
        + (ac_prev - am_prev) * gch_params.kv * gch_params.As * m_cur / gch_params.Ks
    ) * dt_cur

    # Update adjoint mineral value
    a_tr_model.a_grade[:, :, time_index] = (
        am_prev
        * (
            1 / dt_prev
            + gch_params.kv * gch_params.As * (1.0 - c_prev_post_tr / gch_params.Ks)
        )
        # - a_tr_model.a_conc_post_gch[:, :, time_index] / dt_cur + ac_prev / dt_prev
        - ac_prev * gch_params.kv * gch_params.As * (1 - c_prev_post_tr / gch_params.Ks)
    ) * dt_cur
