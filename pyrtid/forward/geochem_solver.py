"""Provide a reactive transport solver."""
from __future__ import annotations

from .models import (
    ConstantConcentration,
    GeochemicalParameters,
    TimeParameters,
    TransportModel,
)


def solve_geochem(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    r"""
    Compute the mineral dissolution.

    The equation reads:

    .. math::
        \overline{c}_{i}^{n+1} = \overline{c}_{i}^{n} + \Delta t^{n} k_{v} A_{s}
        \overline{c}_{i}^{n} \left( 1 - \dfrac{c_{i}^{n+1}}{K_{s}}\right)
    """

    m0 = tr_model.lgrade[time_index - 1]

    # The mobile concentration is from the transport
    dMdt = (
        gch_params.kv
        * gch_params.As
        * m0
        * (1.0 - tr_model.lconc[time_index] / gch_params.Ks)
    )

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dMdt[condition.span] = 0.0
        # elif isinstance(condition, ZeroConcGradient):

    # overwrite the grade
    tr_model.lgrade[time_index] = m0 + dMdt * time_params.dt
