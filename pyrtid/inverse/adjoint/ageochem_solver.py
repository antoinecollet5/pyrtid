"""Provide an adjoint solver for the geochemistry."""
from __future__ import annotations

from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    ConstantConcentration,
    GeochemicalParameters,
    Geometry,
    TimeParameters,
    TransportModel,
)
from pyrtid.inverse.adjoint.amodels import AdjointTransportModel


def solve_adj_geochem_explicit(
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    gch_params: GeochemicalParameters,
    geometry: Geometry,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    """Compute the geochemistry part."""

    # Handle the first time step in the adjoint (= last timestep in the forward)
    try:
        am0 = a_tr_model.a_grade[:, :, time_index + 1]
        ac0 = a_tr_model.a_conc[:, :, time_index + 1]
        m0 = tr_model.grade[:, :, time_index]
        c0 = tr_model.conc[:, :, time_index]

    except IndexError:  # for the first timestep
        am0 = 0.0  # initially at zero for the first timestep
        ac0 = 0.0  # initially at zero for the first timestep

        m0 = tr_model.grade[:, :, time_index]
        c0 = tr_model.conc[:, :, time_index]
    # m0 = tr_model.grade[:, :, time_index]
    # c0 = tr_model.conc[:, :, time_index]

    k1 = am0 * gch_params.kv * gch_params.As * m0 / gch_params.Ks * time_params.dt
    k2 = gch_params.kv * gch_params.As * (1.0 - c0 / gch_params.Ks) * time_params.dt

    # Need to take into account boundary conditions:
    # And then the reactive (chemistry) contribution with the updated conc
    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            k1[condition.span] = 0.0
            k2[condition.span] = 0.0
    # elif isinstance(condition, ZeroConcGradient):

    # Compute the transport contribution
    a_conc_tr_delta = a_tr_model.a_conc[:, :, time_index] - ac0

    # Add the source terms -> from the previous timestep
    # tmp += (a_tr_model.a_sources[:, :, time_index]).ravel("F") / geometry.mesh_area

    # Update concentration + adjoint sources
    a_tr_model.a_conc[:, :, time_index] += -k1

    # a_conc_delta = a_tr_model.a_conc[:, :, time_index] - a_tr_model.a_conc_save
    # a_tr_model.a_conc_save = a_tr_model.a_conc[:, :, time_index].copy()
    # delta_part = ac0  * gch_params.kv * gch_params.As * (1.0 - c0 / gch_params.Ks)
    # * time_params.dt

    # Update mineral value
    a_tr_model.a_grade[:, :, time_index] = am0 * (1 + k2) + k1 - a_conc_tr_delta


def solve_adj_geochem_sequential(
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    """Compute the geochemistry part."""

    # Handle the first time step in the adjoint (= last timestep in the forward)
    # ac0 = a_tr_model.a_conc[:, :, time_index + 1]
    am0 = a_tr_model.a_grade[:, :, time_index + 1]
    m0 = tr_model.grade[:, :, time_index + 1]
    c0 = tr_model.conc_post_tr[:, :, time_index]

    k1 = am0 * gch_params.kv * gch_params.As * m0 / gch_params.Ks * time_params.dt
    k2 = gch_params.kv * gch_params.As * (1.0 - c0 / gch_params.Ks) * time_params.dt

    # Need to take into account boundary conditions:
    # And then the reactive (chemistry) contribution with the updated conc
    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            k1[condition.span] = 0.0
            k2[condition.span] = 0.0
    # elif isinstance(condition, ZeroConcGradient):

    # Compute the transport contribution
    a_conc_tr_delta = (
        a_tr_model.a_conc[:, :, time_index] - a_tr_model.a_conc[:, :, time_index + 1]
    )

    # Update concentration
    a_tr_model.a_conc[:, :, time_index] -= k1

    # a_conc_delta = a_tr_model.a_conc[:, :, time_index] - a_tr_model.a_conc_save
    # a_tr_model.a_conc_save = a_tr_model.a_conc[:, :, time_index].copy()
    # delta_part = ac0  * gch_params.kv * gch_params.As * (1.0 - c0 / gch_params.Ks)
    # * time_params.dt

    # Update mineral value
    a_tr_model.a_grade[:, :, time_index] = am0 * (1 + k2) + k1 - a_conc_tr_delta
