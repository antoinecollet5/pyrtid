# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Provide the derivatives of the discretized equations with respect to the states.

This is required to build the left and right hand sides of the linear systems solved
by the forward sensitivity method (FSM) at each timestep. For the flow and the
transport, the matrices are those stored by the forward solver. For the chemistry,
which is local to each grid cell, the derivatives are analytical.

At the convergence of the fixed point iterations between the transport and the
chemistry, the grades :math:`\overline{c}^n` are given by a local function of the
mobile concentrations :math:`c^n` and of the grades at the previous time:

.. math::
    \overline{c}^n = Y(c^n, \overline{c}^{n-1})
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

from pyrtid.forward.geochem_solver import (
    get_dM_derivatives,
    get_implicit_dM_derivatives,
)
from pyrtid.forward.models import ForwardModel
from pyrtid.utils import NDArrayFloat


def get_chemistry_relations(
    model: ForwardModel, time_index: int, dt: float
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    r"""
    Return the linearized local chemical relations of every grid cell.

    At the convergence of the fixed point iterations, the chemistry provides two
    relations per grid cell between the sensitivities of the mobile concentrations
    :math:`z_c`, of the grades :math:`z_{\overline{c}}` and of the grades at the
    previous time :math:`z_{\overline{c}}^{n-1}`:

    .. math::
        L_c z_c + L_y z_{\overline{c}} = L_p z_{\overline{c}}^{n-1}

    In the general case, the grades are a function of the concentrations and of the
    previous grades, :math:`L_y` is the identity, :math:`L_c = - \partial
    \overline{c} / \partial c` and :math:`L_p = \partial \overline{c} /
    \partial \overline{c}^{n-1}`. When the concentration of a species is null
    because the species is exhausted (implicit chemistry only), the first relation
    imposes that the sensitivity of this concentration is null, and the second
    relation is the stoichiometry.

    Parameters
    ----------
    model : ForwardModel
        The forward model, solved up to ``time_index``.
    time_index : int
        The time index (>= 1).
    dt : float
        The duration of the timestep.

    Returns
    -------
    tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
        The coefficients ``(l_mob, l_immob, l_prev)``, with shape
        (2, 2, nx * ny * nz): ``l_mob[i, j]`` is the coefficient of the
        concentration of the species ``j`` in the relation ``i``, etc.
        If the chemistry is skipped, the grades are constant.
    """
    n_cells = model.grid.n_grid_cells
    jac_mob = np.zeros((2, 2, n_cells))
    jac_immob = np.zeros((2, 2, n_cells))
    jac_immob[0, 0] = 1.0
    jac_immob[1, 1] = 1.0
    pinned = np.zeros(n_cells, dtype=int)

    if not model.tr_model.is_skip_rt:
        stocoef = model.gch_params.stocoef
        if model.gch_params.use_explicit_formulation:
            d_dmob, d_dgrade0, d_dgrade1 = get_dM_derivatives(
                model.tr_model, model.gch_params, time_index, dt
            )
        else:
            d_dmob, d_dgrade0, d_dgrade1, pinned_grid, n_unknown = (
                get_implicit_dM_derivatives(
                    model.tr_model, model.gch_params, time_index, dt
                )
            )
            pinned = pinned_grid.ravel(order="F")
            if n_unknown > 0:
                warnings.warn(
                    f"The active limitation of the implicit chemistry could not be "
                    f"identified in {n_unknown} grid cell(s) at time index "
                    f"{time_index}: the sensitivities are approximate there.",
                    stacklevel=2,
                )
        d_dmob = d_dmob.reshape(2, -1, order="F")
        d_dgrade0 = d_dgrade0.ravel(order="F")
        d_dgrade1 = d_dgrade1.ravel(order="F")

        # grade 0 += dM and grade 1 -= stocoef * dM
        for j in range(2):
            jac_mob[0, j] = d_dmob[j]
            jac_mob[1, j] = -stocoef * d_dmob[j]
        jac_immob[0, 0] += d_dgrade0
        jac_immob[0, 1] += d_dgrade1
        jac_immob[1, 0] += -stocoef * d_dgrade0
        jac_immob[1, 1] += -stocoef * d_dgrade1

    l_mob = -jac_mob
    l_immob = np.zeros((2, 2, n_cells))
    l_immob[0, 0] = 1.0
    l_immob[1, 1] = 1.0
    l_prev = jac_immob

    # Exhausted species: c_i = 0 and sum of the stoichiometric variations is null
    for species in (1, 2):
        mask = pinned == species
        if not np.any(mask):
            continue
        # Relation 0: the concentration of the species is null
        l_mob[0, :, mask] = 0.0
        l_mob[0, species - 1, mask] = 1.0
        l_immob[0, :, mask] = 0.0
        l_prev[0, :, mask] = 0.0
        # Relation 1: stoichiometry
        stocoef = model.gch_params.stocoef
        l_mob[1, :, mask] = 0.0
        l_immob[1, 0, mask] = stocoef
        l_immob[1, 1, mask] = 1.0
        l_prev[1, 0, mask] = stocoef
        l_prev[1, 1, mask] = 1.0
    return l_mob, l_immob, l_prev


def get_coupled_matrix(
    q_next: sparse.csc_array,
    porosity_over_dt: NDArrayFloat,
    l_mob: NDArrayFloat,
    l_immob: NDArrayFloat,
) -> sparse.csc_array:
    r"""
    Return the matrix of the coupled transport and chemistry sensitivities.

    The unknowns are the sensitivities of the mobile concentrations of the two
    species followed by the sensitivities of the grades of the two species. The
    first two block rows are the transport of each species

    .. math::
        Q z_{c_i} + \frac{\omega}{\Delta t} z_{\overline{c}_i} = \dots

    and the last two ones are the local chemical relations (see
    :func:`get_chemistry_relations`).

    Parameters
    ----------
    q_next : sparse.csc_array
        The transport matrix of the timestep, including the term
        :math:`\omega / \Delta t` on the diagonal.
    porosity_over_dt : NDArrayFloat
        The porosity divided by the timestep, with shape (nx * ny * nz,).
    l_mob : NDArrayFloat
        Coefficients of the concentrations in the chemical relations, with shape
        (2, 2, nx * ny * nz).
    l_immob : NDArrayFloat
        Coefficients of the grades in the chemical relations, with shape
        (2, 2, nx * ny * nz).

    Returns
    -------
    sparse.csc_array
        The matrix of shape (4 * nx * ny * nz, 4 * nx * ny * nz).
    """
    n_cells = q_next.shape[0]
    zero = sparse.csc_array((n_cells, n_cells))
    w_mat = sparse.diags_array(porosity_over_dt, format="csc")
    blocks: list[list] = [
        [q_next, zero, w_mat, zero],
        [zero, q_next, zero, w_mat],
    ]
    for i in range(2):
        blocks.append(
            [sparse.diags_array(l_mob[i, j], format="csc") for j in range(2)]
            + [sparse.diags_array(l_immob[i, j], format="csc") for j in range(2)]
        )
    return sparse.block_array(blocks, format="csc")
