"""Provide an adjoint solver for the geochemistry."""
from __future__ import annotations

from copy import copy

import numpy as np

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
        mob_prev = tr_model.lmob[time_index + 1]
        dt_prev = time_params.ldt[time_index]
    else:
        # Handle the Tmax (first timestep going backward)
        # or adjoint state initialization
        a_immob_prev = np.zeros((1))
        a_mob_prev = np.zeros((1))
        np.zeros(tr_model.n_sp)
        mob_prev = np.ones(tr_model.n_sp)
        dt_prev = 1.0  # should be zero but we avoid a zero division here

    # B) Variables at time step n
    a_tr_model.a_immob[:, :, :, time_index]
    a_mob_next = a_tr_model.a_mob[:, :, :, time_index]
    immob_next = tr_model.limmob[time_index]
    tr_model.lmob[time_index]
    dt_next = time_params.ldt[time_index - 1]

    # C) Save the grade for the fix point iterations
    a_tr_model.a_mob_prev = copy(a_mob_next)

    ## Part 2) Compute \lambda \overline{c}_1 et \overline{c}_2

    # 2.1) Reset all to zero for the next time step (n).
    a_tr_model.a_immob[:, :, :, time_index] = 0.0

    # 2.2) Add the sources from the LS objective function
    for sp in range(tr_model.n_sp):
        a_tr_model.a_immob[sp, :, :, time_index] += (
            a_tr_model.a_grade_sources[sp]
            .getcol(time_index)
            .reshape(geometry.nx, geometry.ny, order="F")
        )

    # 2.3) Add the contributions from the transport equation
    # + deal with the adjoint numerical acceleration
    if (
        a_tr_model.is_adj_numerical_acceleration
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

    a_tr_model.a_immob[:, :, :, time_index] -= (
        d_a_mob_ddt * tr_model.porosity * geometry.mesh_volume
    )

    # Up to now, the formulations were equivalent, no matter what species was treated.
    # This won't be the case in the following.

    # 2.4) Update the derivative of d[M](n, i) -> Only for species 1.
    # First we need to compute the d[M](n, i)
    # and its derivative w.r.t. \overbar{c}_1
    if time_index == time_params.nts:
        a_tr_model.a_gch_src_term[:, :, :] = 0.0
    else:
        case1 = (
            (
                gch_params.kv
                * gch_params.As
                * immob_next[0]
                * (1 - mob_prev[0] / gch_params.Ks)
            )
            * mob_prev[1]
            * time_params.dt
        )
        dMdt = np.min(
            np.array(
                [
                    np.abs(case1),
                    immob_next[0],
                    mob_prev[1],
                ]
            ),
            axis=0,
        )

        # Initialize the derivative as null
        dMdtdimmob1 = np.zeros((geometry.nx, geometry.ny))
        dMdtdmob1 = np.zeros((geometry.nx, geometry.ny))
        dMdtdmob2 = np.zeros((geometry.nx, geometry.ny))

        # Case 1
        mask = dMdt == case1
        dMdtdimmob1[mask] = (np.sign(dMdt) * case1 / immob_next[0])[mask]
        dMdtdmob1[mask] = (-np.sign(dMdt) * case1 / (1 - mob_prev[0] / gch_params.Ks))[
            mask
        ]
        dMdtdmob2[mask] = (np.sign(dMdt) * case1 / mob_prev[1])[mask]

        # Case 2
        dMdtdimmob1[dMdt == immob_next[0]] = 1.0

        # Case 3 -> remains zero
        # Do nothing
        dMdtdmob2[dMdt == mob_prev[1]] = 1.0

        # Update mineral value
        a_tr_model.a_immob[0, :, :, time_index] -= a_immob_prev[0] * dMdtdimmob1

        # Last step !!!
        # 2.5) Compute the adjoint geochem source term: it is computed here to mimic the
        # splitting operator approach in which the chemical parameters might not be
        # available in the transport operator - and consequently its adjoint.
        a_tr_model.a_gch_src_term[0] = (
            a_tr_model.a_immob[0, :, :, time_index] * dMdtdmob1
        )
        a_tr_model.a_gch_src_term[1] = (
            a_tr_model.a_immob[0, :, :, time_index] * dMdtdmob2
        )
