"""Provide a reactive transport solver."""

from __future__ import annotations

import numpy as np

from pyrtid.utils import NDArrayFloat

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

    immob1 = tr_model.limmob[time_index - 1][0]
    immob2 = tr_model.limmob[time_index - 1][1]

    # The mobile concentration is from the transport
    dM = get_dM(tr_model, gch_params, time_index, time_params.dt)

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dM[condition.span] = 0.0
        # elif isinstance(condition, ZeroConcGradient):

    assert np.count_nonzero(immob1 + dM < 0) == 0

    # overwrite the grade for species 1
    tr_model.limmob[time_index][0, :, :] = immob1 + dM

    # And for species 2 -> species being consumed
    tr_model.limmob[time_index][1, :, :] = immob2 - gch_params.stocoef * dM


def get_dM(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> NDArrayFloat:
    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]
    immob1 = tr_model.limmob[time_index - 1][0]

    dM = -np.min(
        np.array(
            [
                (
                    -dt
                    * gch_params.kv
                    * gch_params.As
                    * immob1
                    * (1 - mob1 / gch_params.Ks)
                    * mob2
                ),
                immob1,
                gch_params.stocoef * mob2,
            ]
        ),
        axis=0,
    )

    # Handle special cases: because there might be some negative values in the
    # transport because of the semi-implicit time scheme for advection
    mask = (1 - mob1 / gch_params.Ks) <= 0.0  # (1 - mob1 / Ks) positive: precipitation
    dM[mask] = 0.0
    mask = (1 - mob1 / gch_params.Ks) > 1.0  # (1 - mob1 / Ks) positive: precipitation
    dM[mask] = 0.0
    mask = mob2 <= 0.0
    dM[mask] = 0.0

    return dM


def get_dM_pos(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> NDArrayFloat:
    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]
    immob1 = tr_model.limmob[time_index - 1][0]

    dM_pos = np.argmin(
        np.array(
            [
                (
                    -dt
                    * gch_params.kv
                    * gch_params.As
                    * immob1
                    * (1 - mob1 / gch_params.Ks)
                    * mob2
                ),
                immob1,
                gch_params.stocoef * mob2,
            ]
        ),
        axis=0,
    )

    return dM_pos
