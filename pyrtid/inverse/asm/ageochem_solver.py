"""Provide an adjoint solver for the geochemistry."""

from __future__ import annotations

from copy import copy

import numpy as np

from pyrtid.forward.geochem_solver import get_dM, get_dM_pos
from pyrtid.forward.models import (  # ConstantHead,; ZeroConcGradient,
    GeochemicalParameters,
    Geometry,
    TimeParameters,
    TransportModel,
)
from pyrtid.inverse.asm.amodels import AdjointTransportModel
from pyrtid.utils import NDArrayFloat


def solve_adj_geochem(
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    gch_params: GeochemicalParameters,
    geometry: Geometry,
    time_params: TimeParameters,
    time_index: int,
    nafpi: int,
) -> None:
    # Notations for time indices
    # prev = n+1
    # next = n
    # And for the time
    # prev = n
    # next = n-1

    # A) Variables at time step n+1
    if time_index != time_params.nts:
        a_immob_prev = a_tr_model.a_immob[:, :, :, time_index + 1]
        a_mob_prev = a_tr_model.a_mob[:, :, :, time_index + 1]
        tr_model.limmob[time_index + 1]
        tr_model.lmob[time_index + 1]
        dt_prev = time_params.ldt[time_index]
    else:
        # Handle the Tmax (first timestep going backward)
        # or adjoint state initialization
        a_immob_prev = np.zeros((1))
        a_mob_prev = np.zeros((1))
        np.zeros((1))
        np.zeros((1))
        dt_prev = 1.0  # should be zero but we avoid a zero division here

    # B) Variables at time step n
    a_mob_next = a_tr_model.a_mob[:, :, :, time_index]
    tr_model.lmob[time_index]
    dt_next = time_params.ldt[time_index - 1]

    # C) Save the grade for the fix point iterations
    a_tr_model.a_mob_prev = copy(a_mob_next)

    ## Part 2) Compute \lambda \overline{c}_1 et \overline{c}_2

    # 2.1) Reset all to zero for the next time step (n).
    a_tr_model.a_immob[:, :, :, time_index] = 0.0

    # 2.2) Add the sources from the LS objective function
    for sp in range(tr_model.n_sp):
        a_tr_model.a_immob[sp, :, :, time_index] -= a_tr_model.a_grade_sources[sp][
            :, [time_index]
        ].reshape(geometry.nx, geometry.ny, order="F")

    # 2.3) Add the contributions from the transport equation
    # + deal with the adjoint numerical acceleration
    if (
        a_tr_model.is_adj_num_acc_for_timestep
        and nafpi == 1
        and time_index < time_params.nts - 1
    ):
        # Numerical acceleration using the adjoint transport source term from the
        # previous adjoint time iteration (remember backward in time).
        # I don't known if this is an excellent idea yet -> probably need to do
        # something like in HYTEC, i.e., start again the iterations if it fails.
        # Variables at n+2 and n+1 for the time
        a_mob_prev_prev = a_tr_model.a_mob[:, :, :, time_index + 2]
        dt_prev_prev = time_params.ldt[time_index + 1]
        d_a_mob_ddt = a_mob_prev / dt_prev - a_mob_prev_prev / dt_prev_prev
    else:
        # No numerical acceleration, we use
        d_a_mob_ddt = a_mob_next / dt_next - a_mob_prev / dt_prev

    a_tr_model.a_immob[:, :, :, time_index] -= d_a_mob_ddt * tr_model.porosity

    # Up to now, the formulations were equivalent, no matter what species was treated.
    # This won't be the case in the following.
    a_tr_model.a_immob[:, :, :, time_index] += a_immob_prev

    # 2.4) Update the derivative of d[M](n, i) -> Only for species 1.
    # First we need to compute the d[M](n, i)
    # and its derivative w.r.t. \overbar{c}_1
    if time_index != time_params.nts:
        # Update mineral value
        a_tr_model.a_immob[0, :, :, time_index] += (
            a_immob_prev[0] - gch_params.stocoef * a_immob_prev[1]
        ) * (ddMdimmobprev(tr_model, gch_params, time_index, dt_prev))

    # Last step
    # 2.5) Compute the adjoint geochem source term: it is computed here to mimic the
    # splitting operator approach in which the chemical parameters might not be
    # available in the transport operator - and consequently its adjoint.
    a_tr_model.a_gch_src_term[:, :, :] = 0.0
    if time_index != 0:
        a_tr_model.a_gch_src_term[0] = (
            a_tr_model.a_grade[0, :, :, time_index]
            - gch_params.stocoef * a_tr_model.a_grade[1, :, :, time_index]
        ) * ddMdmobnext(tr_model, gch_params, time_index, dt_next, 0)
        a_tr_model.a_gch_src_term[1] = (
            a_tr_model.a_grade[0, :, :, time_index]
            - gch_params.stocoef * a_tr_model.a_grade[1, :, :, time_index]
        ) * ddMdmobnext(tr_model, gch_params, time_index, dt_next, 1)


def ddMdimmobprev(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt_prev: float,
) -> NDArrayFloat:
    """Return the derivative of dM(n+1) w.r.t. immob (n)."""
    dM = get_dM(tr_model, gch_params, time_index + 1, dt_prev)
    # Initiate the derivative to zero
    deriv = np.zeros_like(dM)

    # Need this to handle the derivative of the min function... this is not optimal
    # but we do not really have the choice.
    dm_pos = get_dM_pos(tr_model, gch_params, time_index + 1, dt_prev)

    mob1 = tr_model.lmob[time_index + 1][0]
    mob2 = tr_model.lmob[time_index + 1][1]

    mask = dm_pos == 0
    deriv[mask] = (
        dt_prev * gch_params.kv * gch_params.As * (1 - mob1 / gch_params.Ks) * mob2
    )[mask]

    mask = dm_pos == 1
    deriv[mask] = -1.0

    # Mask for null values
    deriv[dM == 0] = 0

    return deriv


def ddMdmobnext(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt_next: float,
    sp: int,
) -> NDArrayFloat:
    """Return the derivative of dM w.r.t. mob. (n+1)"""
    dM = get_dM(tr_model, gch_params, time_index, dt_next)
    # Initiate the derivative to zero
    deriv = np.zeros_like(dM)
    # Need this to handle the derivative of the min function... this is not optimal
    # but we do not really have the choice.
    dm_pos = get_dM_pos(tr_model, gch_params, time_index, dt_next)

    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]
    immob1 = tr_model.limmob[time_index - 1][0]

    mask = dm_pos == 0
    if sp == 0:
        deriv[mask] = (
            -dt_next * gch_params.kv * gch_params.As * immob1 / gch_params.Ks * mob2
        )[mask]
    elif sp == 1:
        deriv[mask] = (
            dt_next
            * gch_params.kv
            * gch_params.As
            * immob1
            * (1 - mob1 / gch_params.Ks)
        )[mask]
    else:
        raise ValueError("sp should be 0 or 1")

    if sp == 1:
        mask = dm_pos == 2
        deriv[mask] = -1.0 / gch_params.stocoef

    # Mask for null values
    deriv[dM == 0] = 0

    return deriv
