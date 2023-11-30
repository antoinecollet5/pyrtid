"""Provide a reactive transport solver."""
from __future__ import annotations

import numpy as np

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

    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]

    immob1 = tr_model.limmob[time_index - 1][0]
    immob2 = tr_model.limmob[time_index - 1][1]

    # The mobile concentration is from the transport
    dMdt = -np.min(
        np.array(
            [
                np.abs(
                    (
                        gch_params.kv
                        * gch_params.As
                        * immob1
                        * (1 - mob1 / gch_params.Ks)
                    )
                    * mob2
                    * time_params.dt
                ),
                mob2,
                immob1,
            ]
        ),
        axis=0,
    )

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dMdt[condition.span] = 0.0
        # elif isinstance(condition, ZeroConcGradient):

    # overwrite the grade for species 1
    tr_model.limmob[time_index][0, :, :] = immob1 + dMdt

    # And for species 2 -> species being consumed
    tr_model.limmob[time_index][1, :, :] = immob2 - dMdt
