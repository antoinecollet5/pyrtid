"""Provide a reactive transport solver."""
from __future__ import annotations

from .models import (
    ConstantConcentration,
    GeochemicalParameters,
    TimeParameters,
    TransportModel,
)


def solve_geochem_sequential(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    """Compute the geochemistry part."""

    # Need to take into account boundary conditions:
    # And then the reactive (chemistry) contribution with the updated conc

    # Note: the concentration has been modified by the transport so we need to use the
    # updated one.
    m0 = tr_model.grade[:, :, time_index]
    dMdt = (
        gch_params.kv
        * gch_params.As
        * m0
        * (1.0 - tr_model.conc_post_tr[:, :, time_index] / gch_params.Ks)
    )

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dMdt[condition.span] = 0.0
        # elif isinstance(condition, ZeroConcGradient):

    tr_model.grade[:, :, time_index] = m0 + dMdt * time_params.dt
    tr_model.conc[:, :, time_index] = (
        tr_model.conc_post_tr[:, :, time_index] - dMdt * time_params.dt
    )


def solve_geochem_explicit(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    """Compute the geochemistry part."""

    # Need to take into account boundary conditions:
    # And then the reactive (chemistry) contribution with the updated conc

    # Note: the concentration has been modified by the transport so we need to use the
    # updated one.
    m0 = tr_model.grade[:, :, time_index - 1]
    dMdt = (
        gch_params.kv
        * gch_params.As
        * m0
        * (1.0 - tr_model.conc[:, :, time_index - 1] / gch_params.Ks)
    )

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dMdt[condition.span] = 0.0
        # elif isinstance(condition, ZeroConcGradient):

    tr_model.grade[:, :, time_index] = m0 + dMdt * time_params.dt
    tr_model.conc[:, :, time_index] = (
        tr_model.conc_post_tr[:, :, time_index] - dMdt * time_params.dt
    )
