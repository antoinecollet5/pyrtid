# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide gradient computation routines."""

import copy
import warnings
from typing import List, Optional

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.forward.flow_solver import get_rhomean
from pyrtid.forward.models import GRAVITY, WATER_DENSITY, FlowRegime, VerticalAxis
from pyrtid.inverse.asm import AdjointModel, AdjointSolver
from pyrtid.inverse.asm.ageochem_solver import ddMdimmobprev
from pyrtid.inverse.loss_function import eval_model_loss_function
from pyrtid.inverse.obs import Observable, Observables
from pyrtid.inverse.params import (
    AdjustableParameter,
    AdjustableParameters,
    ParameterName,
    get_parameter_values_from_model,
    update_model_with_parameters_values,
    update_parameters_from_model,
)
from pyrtid.utils import (
    NDArrayFloat,
    StrEnum,
    finite_gradient,
    is_all_close,
    object_or_object_sequence_to_list,
)
from pyrtid.utils.means import dxi_harmonic_mean, harmonic_mean


class DerivationVariable(StrEnum):
    POROSITY = "porosity"
    DIFFUSION = "diffusion"
    DISPERSIVITY = "dispersivity"


def get_diffusion_term_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel, deriv_var: DerivationVariable
) -> NDArrayFloat:
    """
    Compute the gradient of the transport diffusive term with respect to a variable.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.
    deriv_var: DerivationVariable
        The variable with respect to which the gradient is computed.

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the diffusion.
    """
    shape = (*fwd_model.grid.shape, fwd_model.time_params.nt)

    # diffusion + dispersivity
    d = (
        fwd_model.tr_model.effective_diffusion[:, :, :, np.newaxis]
        + fwd_model.tr_model.dispersivity[:, :, :, np.newaxis]
        * fwd_model.fl_model.get_u_darcy_norm()[:, :, :, 1:]
    )
    grad = np.zeros(shape)

    if deriv_var == DerivationVariable.POROSITY:
        # Note: this is the diffusion, not the effective diffusion !
        term_in_d_deriv = fwd_model.tr_model.diffusion[:, :, :, np.newaxis]
    elif deriv_var == DerivationVariable.DIFFUSION:
        term_in_d_deriv = fwd_model.tr_model.porosity[:, :, :, np.newaxis]
    elif deriv_var == DerivationVariable.DISPERSIVITY:
        term_in_d_deriv = fwd_model.fl_model.get_u_darcy_norm()[:, :, :, 1:]

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion

    for sp in range(fwd_model.tr_model.n_sp):
        mob = fwd_model.tr_model.mob[sp]
        # mob = fwd_model.tr_model.mob_post_tr
        amob = adj_model.a_tr_model.a_mob[sp]

        for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
            if n < 2:
                continue

            # Consider the x axis
            fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
            bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

            # Forward scheme
            dconc_f = np.zeros(shape)
            dconc_f[(fwd_slicer) + (slice(1, None),)] += (
                crank_diff
                * (
                    mob[(bwd_slicer) + (slice(1, None),)]
                    - mob[(fwd_slicer) + (slice(1, None),)]
                )
                + (1.0 - crank_diff)
                * (
                    mob[(bwd_slicer) + (slice(None, -1),)]
                    - mob[(fwd_slicer) + (slice(None, -1),)]
                )
            ) * (
                dxi_harmonic_mean(d[fwd_slicer], d[bwd_slicer])
                * term_in_d_deriv[fwd_slicer]
            )

            damob_f = np.zeros(shape)
            damob_f[(fwd_slicer) + (slice(None),)] += (
                amob[(bwd_slicer) + (slice(None),)]
                - amob[(fwd_slicer) + (slice(None),)]
            )

            # Backward scheme
            dconc_b = np.zeros(shape)
            dconc_b[(bwd_slicer) + (slice(1, None),)] += (
                crank_diff
                * (
                    mob[(fwd_slicer) + (slice(1, None),)]
                    - mob[(bwd_slicer) + (slice(1, None),)]
                )
                + (1.0 - crank_diff)
                * (
                    mob[(fwd_slicer) + (slice(None, -1),)]
                    - mob[(bwd_slicer) + (slice(None, -1),)]
                )
            ) * (
                dxi_harmonic_mean(d[bwd_slicer], d[fwd_slicer])
                * term_in_d_deriv[bwd_slicer]
            )

            damob_b = np.zeros(shape)
            damob_b[(bwd_slicer) + (slice(None),)] += (
                amob[(fwd_slicer) + (slice(None),)]
                - amob[(bwd_slicer) + (slice(None),)]
            )

            # Gather the two schemes
            grad += (
                (dconc_f * damob_f + dconc_b * damob_b)
                * fwd_model.grid.gamma_ij(axis)
                / fwd_model.grid.pipj(axis)
                / fwd_model.grid.grid_cell_volume
            )

    # We sum along the temporal axis
    return np.sum(grad, axis=-1)


def get_diffusion_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient of the transport diffusive term with respect to a variable.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the diffusion.
    """
    grad = get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.DIFFUSION
    )
    # Add the adjoint sources for initial time (t0)
    return grad + adj_model.a_tr_model.a_diffusion_sources[:, [0]].todense().reshape(
        grad.shape, order="F"
    )


def get_dispersivity_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient of the transport dispersive term with respect to a variable.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the dispersivity.
    """
    grad = get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.DISPERSIVITY
    )
    # Add the adjoint sources for initial time (t0)
    return grad + adj_model.a_tr_model.a_dispersivity_sources[:, [0]].todense().reshape(
        grad.shape, order="F"
    )


def get_porosity_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient of the objective function with respect to the porosity.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the porosity.
    """
    grad = np.zeros((*fwd_model.grid.shape, fwd_model.time_params.nt - 1))

    for sp in range(fwd_model.tr_model.n_sp):
        mob = fwd_model.tr_model.mob[sp]
        immob = fwd_model.tr_model.immob[sp]
        amob = adj_model.a_tr_model.a_mob[sp]

        grad += (
            (
                mob[:, :, :, 1:]
                - mob[:, :, :, :-1]
                + immob[:, :, :, 1:]
                - immob[:, :, :, :-1]
            )
            / fwd_model.time_params.ldt
            * amob[:, :, :, 1:]
        )

    # We sum along the temporal axis + get the diffusion gradient
    grad = np.sum(grad, axis=-1) + get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.POROSITY
    )
    # Add the adjoint sources for initial time (t0)
    return grad + adj_model.a_tr_model.a_porosity_sources[:, [0]].todense().reshape(
        grad.shape, order="F"
    )


def get_permeability_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient of the objective function with respect to the permeability.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the permeability.
    """
    if fwd_model.fl_model.is_gravity:
        grad = _get_perm_gradient_from_diffusivity_eq_density(
            fwd_model, adj_model
        ) + _get_perm_gradient_from_darcy_eq_density(fwd_model, adj_model)
    else:
        grad = _get_perm_gradient_from_diffusivity_eq_saturated(
            fwd_model, adj_model
        ) + _get_perm_gradient_from_darcy_eq_saturated(fwd_model, adj_model)

    # Add the adjoint sources for initial time (t0)
    return grad + adj_model.a_fl_model.a_permeability_sources[:, [0]].todense().reshape(
        grad.shape, order="F"
    )


def _get_perm_gradient_from_diffusivity_eq_saturated(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using head observations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using head observations.
    """

    # continuous version just for the article about discretization.
    if adj_model.a_fl_model.is_use_continuous_adj:
        pass

    shape = (*fwd_model.grid.shape, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    if adj_model.a_fl_model.crank_nicolson is None:
        crank_flow: float = fwd_model.fl_model.crank_nicolson
    else:
        crank_flow = adj_model.a_fl_model.crank_nicolson

    head = fwd_model.fl_model.head
    ahead = adj_model.a_fl_model.a_head
    ma_ahead = np.zeros(ahead.shape)
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_ahead[free_head_indices[0], free_head_indices[1], free_head_indices[2], :] = (
        ahead[free_head_indices[0], free_head_indices[1], free_head_indices[2], :]
    )
    # add the storgae coefficient tp ma_ahead_sc
    ma_ahead_sc = ma_ahead / (
        fwd_model.fl_model.storage_coefficient[:, :, :, np.newaxis]
        * fwd_model.grid.grid_cell_volume
    )
    grad = np.zeros(shape)

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        tmp = fwd_model.grid.gamma_ij(axis) / fwd_model.grid.pipj(axis)

        # Forward scheme
        dhead_f = np.zeros(shape)
        dhead_f = (
            crank_flow
            * (
                head[(bwd_slicer) + (slice(1, None),)]
                - head[(fwd_slicer) + (slice(1, None),)]
            )
            + (1.0 - crank_flow)
            * (
                head[(bwd_slicer) + (slice(None, -1),)]
                - head[(fwd_slicer) + (slice(None, -1),)]
            )
        ) * dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
            :, :, :, np.newaxis
        ]
        grad[(fwd_slicer) + (slice(1, None),)] += (
            dhead_f
            * (
                ma_ahead_sc[(bwd_slicer) + (slice(1, None),)]
                - ma_ahead_sc[(fwd_slicer) + (slice(1, None),)]
            )
            * tmp
        )

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dhead_f = (
                head[(bwd_slicer) + (slice(None, 1),)]
                - head[(fwd_slicer) + (slice(None, 1),)]
            ) * dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                :, :, :, np.newaxis
            ]
            grad[(fwd_slicer) + (slice(None, 1),)] += (
                dhead_f
                * (
                    ma_ahead[(bwd_slicer) + (slice(None, 1),)]
                    - ma_ahead[(fwd_slicer) + (slice(None, 1),)]
                )
                / fwd_model.grid.grid_cell_volume
            ) * tmp

        # Backward scheme
        dhead_b = (
            crank_flow
            * (
                head[(fwd_slicer) + (slice(1, None),)]
                - head[(bwd_slicer) + (slice(1, None),)]
            )
            + (1.0 - crank_flow)
            * (
                head[(fwd_slicer) + (slice(None, -1),)]
                - head[(bwd_slicer) + (slice(None, -1),)]
            )
        ) * dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
            :, :, :, np.newaxis
        ]
        grad[(bwd_slicer) + (slice(1, None),)] += (
            dhead_b
            * (
                ma_ahead_sc[(fwd_slicer) + (slice(1, None),)]
                - ma_ahead_sc[(bwd_slicer) + (slice(1, None),)]
            )
            * tmp
        )

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dhead_b = (
                head[(fwd_slicer) + (slice(None, 1),)]
                - head[(bwd_slicer) + (slice(None, 1),)]
            ) * dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                :, :, :, np.newaxis
            ]
            grad[(bwd_slicer) + (slice(None, 1),)] += (
                dhead_b
                * (
                    ma_ahead[(fwd_slicer) + (slice(None, 1),)]
                    - ma_ahead[(bwd_slicer) + (slice(None, 1),)]
                )
                / fwd_model.grid.grid_cell_volume
            ) * tmp

    # We sum along the temporal axis
    return np.sum(grad, axis=-1)


def _get_perm_gradient_from_diffusivity_eq_density(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using head observations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using head observations.
    """
    shape = (*fwd_model.grid.shape, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    if adj_model.a_fl_model.crank_nicolson is None:
        crank_flow: float = fwd_model.fl_model.crank_nicolson
    else:
        crank_flow = adj_model.a_fl_model.crank_nicolson

    pressure = fwd_model.fl_model.pressure
    apressure = adj_model.a_fl_model.a_pressure
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_apressure = np.zeros(apressure.shape)
    ma_apressure[
        free_head_indices[0],
        free_head_indices[1],
        free_head_indices[2],
        :,
    ] = apressure[free_head_indices[0], free_head_indices[1], free_head_indices[2], :]
    # add the storgae coefficient to ma_apressure
    ma_apressure_sc = ma_apressure / (
        fwd_model.fl_model.storage_coefficient[:, :, :, np.newaxis]
        * fwd_model.grid.grid_cell_volume
    )
    grad = np.zeros(shape)

    vp = fwd_model.fl_model._get_mesh_center_vertical_pos()

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        rhomean = get_rhomean(
            fwd_model.grid,
            fwd_model.tr_model,
            axis=axis,
            time_index=slice(0, -1),
            is_flatten=False,
        )[fwd_slicer]
        tmp = 0.0

        if (
            (fwd_model.fl_model.vertical_axis == VerticalAxis.X and axis == 0)
            or (fwd_model.fl_model.vertical_axis == VerticalAxis.Y and axis == 1)
            or (fwd_model.fl_model.vertical_axis == VerticalAxis.Z and axis == 2)
        ):
            tmp = GRAVITY * rhomean**2

        # Forward scheme
        dpressure_f = (
            (
                (
                    crank_flow
                    * (
                        pressure[(bwd_slicer) + (slice(1, None),)]
                        - pressure[(fwd_slicer) + (slice(1, None),)]
                    )
                    + (1.0 - crank_flow)
                    * (
                        pressure[(bwd_slicer) + (slice(None, -1),)]
                        - pressure[(fwd_slicer) + (slice(None, -1),)]
                    )
                )
                / fwd_model.grid.pipj(axis)
                * rhomean
                + tmp
            )
            * dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                :, :, :, np.newaxis
            ]
            / WATER_DENSITY
        )

        grad[(fwd_slicer) + (slice(1, None),)] += (
            dpressure_f
            * (
                ma_apressure_sc[(bwd_slicer) + (slice(1, None),)]
                - ma_apressure_sc[(fwd_slicer) + (slice(1, None),)]
            )
            * fwd_model.grid.gamma_ij(axis)
        )

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            grad[(fwd_slicer) + (slice(None, 1),)] += (
                (
                    (
                        pressure[(bwd_slicer) + (slice(None, 1),)]
                        - pressure[(fwd_slicer) + (slice(None, 1),)]
                    )
                    / WATER_DENSITY
                    / GRAVITY
                    + vp[(bwd_slicer) + (np.newaxis,)]
                    - vp[(fwd_slicer) + (np.newaxis,)]
                )
                * dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                    :, :, :, np.newaxis
                ]
                * fwd_model.grid.gamma_ij(axis)
                / fwd_model.grid.pipj(axis)
                * (
                    ma_apressure[(bwd_slicer) + (slice(None, 1),)]
                    - ma_apressure[(fwd_slicer) + (slice(None, 1),)]
                )
                / fwd_model.grid.grid_cell_volume
            )

        # Backward scheme
        dpressure_b = (
            (
                (
                    crank_flow
                    * (
                        pressure[(fwd_slicer) + (slice(1, None),)]
                        - pressure[(bwd_slicer) + (slice(1, None),)]
                    )
                    + (1.0 - crank_flow)
                    * (
                        pressure[(fwd_slicer) + (slice(None, -1),)]
                        - pressure[(bwd_slicer) + (slice(None, -1),)]
                    )
                )
                / fwd_model.grid.pipj(axis)
                * rhomean
                - tmp
            )
            * dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                :, :, :, np.newaxis
            ]
            / WATER_DENSITY
        )

        grad[(bwd_slicer) + (slice(1, None),)] += (
            dpressure_b
            * (
                ma_apressure_sc[(fwd_slicer) + (slice(1, None),)]
                - ma_apressure_sc[(bwd_slicer) + (slice(1, None),)]
            )
            * fwd_model.grid.gamma_ij(axis)
        )

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            grad[(bwd_slicer) + (slice(None, 1),)] += (
                (
                    (
                        pressure[(fwd_slicer) + (slice(None, 1),)]
                        - pressure[(bwd_slicer) + (slice(None, 1),)]
                    )
                    / WATER_DENSITY
                    / GRAVITY
                    + vp[(fwd_slicer) + (np.newaxis,)]
                    - vp[(bwd_slicer) + (np.newaxis,)]
                )
                * dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                    :, :, :, np.newaxis
                ]
                * fwd_model.grid.gamma_ij(axis)
                / fwd_model.grid.pipj(axis)
                * (
                    ma_apressure[(fwd_slicer) + (slice(None, 1),)]
                    - ma_apressure[(bwd_slicer) + (slice(None, 1),)]
                )
                / fwd_model.grid.grid_cell_volume
            )

    # We sum along the temporal axis
    return np.sum(grad, axis=-1)


def _get_perm_gradient_from_darcy_eq_saturated(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using mob observations.

    Mob are the mobile concentrations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using mob observations.
    """
    shape = (*fwd_model.grid.shape, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    head = fwd_model.fl_model.head

    grad = np.zeros_like(head)

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        if axis == 0:
            a_u_darcy = adj_model.a_fl_model.a_u_darcy_x
        elif axis == 1:
            a_u_darcy = adj_model.a_fl_model.a_u_darcy_y
        elif axis == 2:
            a_u_darcy = adj_model.a_fl_model.a_u_darcy_z
        else:
            raise ValueError()

        a_u_darcy = a_u_darcy[bwd_slicer]

        # Consider the x axis
        # Forward scheme
        dhead_f = np.zeros(shape)
        dhead_f[(fwd_slicer) + (slice(None),)] += (
            (head[(bwd_slicer) + (slice(None),)] - head[(fwd_slicer) + (slice(None),)])
            * dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                :, :, :, np.newaxis
            ]
            * a_u_darcy
        )

        # Backward scheme
        dhead_b = np.zeros(shape)
        dhead_b[(bwd_slicer) + (slice(None),)] -= (
            (head[(fwd_slicer) + (slice(None),)] - head[(bwd_slicer) + (slice(None),)])
            * dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                :, :, :, np.newaxis
            ]
            * a_u_darcy
        )

        # Gather the two schemes
        grad += (dhead_f + dhead_b) / fwd_model.grid.pipj(axis)

    # We sum along the temporal axis
    return np.sum(grad, axis=-1)


def _get_perm_gradient_from_darcy_eq_density(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using mob observations.

    Mob are the mobile concentrations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using mob observations.
    """
    permeability = fwd_model.fl_model.permeability

    time_slice = slice(1, None)
    shape = (*fwd_model.grid.shape, fwd_model.time_params.nt - 1)

    pressure = fwd_model.fl_model.pressure[:, :, :, time_slice]
    grad = np.zeros_like(pressure)

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        if axis == 0:
            a_u_darcy = adj_model.a_fl_model.a_u_darcy_x
        elif axis == 1:
            a_u_darcy = adj_model.a_fl_model.a_u_darcy_y
        elif axis == 2:
            a_u_darcy = adj_model.a_fl_model.a_u_darcy_z
        else:
            raise ValueError()

        a_u_darcy = a_u_darcy[(bwd_slicer) + (time_slice,)]

        if (
            (fwd_model.fl_model.vertical_axis == VerticalAxis.X and axis == 0)
            or (fwd_model.fl_model.vertical_axis == VerticalAxis.Y and axis == 1)
            or (fwd_model.fl_model.vertical_axis == VerticalAxis.Z and axis == 2)
        ):
            rho_ij_g = (
                get_rhomean(
                    fwd_model.grid,
                    fwd_model.tr_model,
                    axis=axis,
                    time_index=slice(None),
                    is_flatten=False,
                )[fwd_slicer]
                * GRAVITY
            )

            # shift
            rho_ij_g = rho_ij_g[:, :, :, :-1]
        else:
            rho_ij_g = 0.0

        # Consider the x axis
        # Forward scheme
        dpressure_f = np.zeros(shape)
        dpressure_f[fwd_slicer] += (
            (
                (pressure[bwd_slicer] - pressure[fwd_slicer])
                / fwd_model.grid.pipj(axis)
                + rho_ij_g
            )
            * dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                :, :, :, np.newaxis
            ]
            * a_u_darcy
        )

        # Backward scheme
        dpressure_b = np.zeros(shape)
        dpressure_b[bwd_slicer] -= (
            (
                (pressure[fwd_slicer] - pressure[bwd_slicer])
                / fwd_model.grid.pipj(axis)
                - rho_ij_g
            )
            * dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                :, :, :, np.newaxis
            ]
            * a_u_darcy
        )

        # Gather the two schemes
        grad += (dpressure_f + dpressure_b) / GRAVITY / WATER_DENSITY

    # We sum along the temporal axis
    return np.sum(grad, axis=-1)


def get_sc_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the storage coefficient.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.
    Returns
    -------
    NDArrayFloat
        Gradient with respect to the storage coefficient.
    """
    if fwd_model.fl_model.is_gravity:
        return get_sc_adjoint_gradient_density(fwd_model, adj_model)
    return get_sc_adjoint_gradient_saturated(fwd_model, adj_model)


def get_sc_adjoint_gradient_saturated(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the storage coefficient.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the storage coefficient.
    """
    head = fwd_model.fl_model.head
    ahead = adj_model.a_fl_model.a_head
    ma_ahead = np.zeros(ahead.shape)
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_ahead[free_head_indices[0], free_head_indices[1], free_head_indices[2], :] = (
        ahead[free_head_indices[0], free_head_indices[1], free_head_indices[2], :]
    )

    grad = (
        (head[:, :, :, 1:] - head[:, :, :, :-1])
        * ma_ahead[:, :, :, 1:]
        / np.array(fwd_model.time_params.ldt)[np.newaxis, np.newaxis, np.newaxis, :]
        / fwd_model.fl_model.storage_coefficient[:, :, :, np.newaxis]
    )

    # We sum along the temporal axis
    return np.sum(grad, axis=-1) + adj_model.a_fl_model.a_storage_coefficient_sources[
        :, [0]
    ].todense().reshape(fwd_model.grid.shape, order="F")


def get_sc_adjoint_gradient_density(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the storage coefficient.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the storage coefficient.
    """
    pressure = fwd_model.fl_model.pressure
    apressure = adj_model.a_fl_model.a_pressure
    ma_apressure = np.zeros_like(apressure)
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_apressure[
        free_head_indices[0], free_head_indices[1], free_head_indices[2], :
    ] = apressure[free_head_indices[0], free_head_indices[1], free_head_indices[2], :]

    grad = (
        (pressure[:, :, :, 1:] - pressure[:, :, :, :-1])
        * ma_apressure[:, :, :, 1:]
        / np.array(fwd_model.time_params.ldt)[np.newaxis, np.newaxis, np.newaxis, :]
        / fwd_model.fl_model.storage_coefficient[:, :, :, np.newaxis]
    )

    # We sum along the temporal axis
    return np.sum(grad, axis=-1) + adj_model.a_fl_model.a_storage_coefficient_sources[
        :, [0]
    ].todense().reshape(fwd_model.grid.shape, order="F")


def get_initial_grade_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel, sp: int
) -> NDArrayFloat:
    r"""
    Gradient with respect to mineral phase initial concentrations.

    The gradient reads

    .. math::

        \dfrac{\partial \mathcal{L}}{\partial \overline{c}_{i}^{0}} =
        \dfrac{V_{i} \omega_{i}}{\Delta t^{0}} \lambda_{c_{i}}^{1}
        + \lambda_{\overline{c}_{i}}^{1} \left( 1 + \Delta t^{0} k_{v} A_{s}
        \left( 1 - \dfrac{c_{i}^{1}}{K_{s}}\right) \right)
        - \dfrac{\overline{c}_{i}^{0, \mathrm{obs}}
        - \overline{c}_{i}^{0, \mathrm{calc}}}{\left(
            \sigma_{\overline{c}_{i}}^{0, \mathrm{obs}}\right)^{2}}

    """
    # Prat common for all species
    grad = (
        -(
            adj_model.a_tr_model.a_mob[sp, :, :, :, 1]
            / fwd_model.time_params.ldt[0]
            * fwd_model.tr_model.porosity
        )
        - adj_model.a_tr_model.a_immob[sp, :, :, :, 1]
    )
    # Add adjoint sources for time t=0
    grad += (
        adj_model.a_tr_model.a_grade_sources[sp][:, [0]]
        .todense()
        .reshape(grad.shape, order="F")
    )

    if sp == 0:
        grad -= (
            adj_model.a_tr_model.a_immob[0, :, :, :, 1]
            - fwd_model.gch_params.stocoef * adj_model.a_tr_model.a_immob[1, :, :, :, 1]
        ) * (
            ddMdimmobprev(
                fwd_model.tr_model,
                fwd_model.gch_params,
                0,
                fwd_model.time_params.ldt[0],
            )
        )

    return grad


def get_initial_head_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    # flatten the heads
    a_head = adj_model.a_fl_model.a_head.reshape(
        (-1, fwd_model.time_params.nt), order="F"
    )

    # Initialize the gradient as a null array
    grad = np.zeros(a_head[:, 0].size, dtype=np.float64)
    fl_model = fwd_model.fl_model

    # Here we consider that the Dirichlet conditions are not varying
    grad[fl_model.cst_head_nn] -= np.sum(a_head[fl_model.cst_head_nn, :], axis=-1)

    # Stationary case
    if fl_model.regime != FlowRegime.STATIONARY:
        grad[fl_model.free_head_nn] -= a_head[fl_model.free_head_nn, 0]

    return grad.reshape(fwd_model.fl_model.lhead[0].shape, order="F")


def get_initial_pressure_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    r"""
    Gradient with respect to mineral phase initial concentrations.

    The gradient reads

    # TODO

    """
    # flatten the pressures
    a_pressure = adj_model.a_fl_model.a_pressure.reshape(
        (-1, fwd_model.time_params.nt), order="F"
    )

    # Initialize the gradient as a null array
    grad = np.zeros(a_pressure[:, 0].size, dtype=np.float64)
    fl_model = fwd_model.fl_model

    # Here we consider that the Dirichlet conditions are not varying
    grad[fl_model.cst_head_nn] -= np.sum(a_pressure[fl_model.cst_head_nn, :], axis=-1)

    # Stationary case
    if fl_model.regime != FlowRegime.STATIONARY:
        grad[fl_model.free_head_nn] -= a_pressure[fl_model.free_head_nn, 0]

    return grad.reshape(fwd_model.fl_model.lpressure[0].shape, order="F")


def get_initial_conc_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel, sp: int
) -> NDArrayFloat:
    r"""
    Gradient with respect to aqueous phase initial concentrations.

    The gradient reads

    .. math::

        \dfrac{\partial \mathcal{L}}{\partial c_{i}^{0}} =
        \dfrac{V_{i}\omega_{i}}{\Delta t^{0}} \lambda_{c_{i}}^{1}
        - \dfrac{c_{i}^{0, \mathrm{obs}}
        - c_{i}^{0, \mathrm{calc}}}{\left(\sigma_{c_{i}}^{0, \mathrm{obs}}\right)^{2}}
        + \sum_{\mathrm{neigh} \;j} \mathcal{A}_{\Gamma_{ij}} D_{e, ij}
        (1 - \alpha_{\mathrm{d}})\dfrac{\lambda_{c_{j}}^{1} - \lambda_{c_{i}}^{1}}{
            \left\lVert \overrightarrow{\mathrm{P}_{i}\mathrm{P}_{j}} \right\rVert}

    """
    tr_model = fwd_model.tr_model

    grad = (
        adj_model.a_tr_model.a_mob[sp, :, :, :, 1]
        * fwd_model.tr_model.porosity
        / fwd_model.time_params.ldt[0]
    )

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion
    a_mob = adj_model.a_tr_model.a_mob[sp, :, :, :, 1]

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        dmean = harmonic_mean(
            tr_model.effective_diffusion[fwd_slicer],
            tr_model.effective_diffusion[bwd_slicer],
        )
        tmp = fwd_model.grid.gamma_ij(axis) / fwd_model.grid.pipj(axis)
        # Forward scheme
        grad[fwd_slicer] += (
            +(1.0 - crank_diff) * (a_mob[bwd_slicer] - a_mob[fwd_slicer]) * dmean
        ) * tmp
        # Backward scheme
        grad[bwd_slicer] += (
            +(1.0 - crank_diff) * (a_mob[fwd_slicer] - a_mob[bwd_slicer]) * dmean
        ) * tmp

    return -grad + adj_model.a_tr_model.a_conc_sources[sp][:, 0].todense().reshape(
        grad.shape, order="F"
    )


def compute_param_adjoint_loss_ls_function_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel, param: AdjustableParameter
) -> NDArrayFloat:
    """
    Compute the gradient of the ls loss function with respect to the parameter.

    The gradient is computed from the adjoint state.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all forward variables and parameters.
    param : AdjustableParameter
        The adjusted parameter instance.

    Returns
    -------
    NDArrayFloat
        The computed ls loss function gradient.
    """
    if param.name == ParameterName.DIFFUSION:
        return get_diffusion_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.DISPERSIVITY:
        return get_dispersivity_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.POROSITY:
        return get_porosity_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.PERMEABILITY:
        return get_permeability_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.INITIAL_CONCENTRATION:
        return get_initial_conc_adjoint_gradient(fwd_model, adj_model, param.sp)
    if param.name == ParameterName.INITIAL_GRADE:
        return get_initial_grade_adjoint_gradient(fwd_model, adj_model, param.sp)
    if param.name == ParameterName.INITIAL_HEAD:
        if fwd_model.fl_model.is_gravity:
            raise RuntimeError(
                "Cannot optimize the initial head when using a density flow"
                " (gravity is on). Optimize the initial pressure instead!"
            )
        return get_initial_head_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.STORAGE_COEFFICIENT:
        return get_sc_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.INITIAL_PRESSURE:
        if not fwd_model.fl_model.is_gravity:
            raise RuntimeError(
                "Cannot optimize the initial pressure if not using a density flow"
                " (gravity is off). Optimize the initial head instead!"
            )
        return get_initial_pressure_adjoint_gradient(fwd_model, adj_model)
    raise (NotImplementedError("Please contact the developer to handle this issue."))


def compute_adjoint_gradient(
    fwd_model: ForwardModel,
    adj_model: AdjointModel,
    parameters_to_adjust: AdjustableParameters,
    is_save_state: bool = True,
) -> NDArrayFloat:
    """
    Compute the gradient of the given parameters with the adjoint state.

    Note
    ----
    Return the preconditioned adjoint gradient.

    """
    grad = np.array([], dtype=np.float64)
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # 1) least square loss function gradient with regards to observations
        param_grad = compute_param_adjoint_loss_ls_function_gradient(
            fwd_model, adj_model, param
        )

        # 2) Filter the LS gradient (smoothing for instance)
        for filt in param.filters:
            param_grad = filt.filter(
                param_grad,
                len(param.grad_adj_history),
            )

        # 3) regularization of loss function gradient (also non-preconditioned
        # to this point)
        param_grad += (
            param.eval_loss_reg_gradient().reshape(param.values.shape, order="F")
            * param.reg_weight
        )

        # 4) Save the non-preconditioned gradient
        if is_save_state:
            param.grad_adj_raw_history.append(param_grad.copy())

        # 5) Apply preconditioning and flatten
        param_values = get_parameter_values_from_model(fwd_model, param)
        param_grad = param.preconditioner.dbacktransform_vec(
            param.preconditioner(param_values.ravel("F")),
            param_grad.ravel("F"),
        )

        # 6) Save the preconditioned gradient
        if is_save_state:
            param.grad_adj_history.append(param_grad.copy())
        # 7) update the global gradient vector
        grad = np.hstack((grad, param_grad))
    return grad


def _local_fun_loss(
    vector: NDArrayFloat,
    parameter: AdjustableParameter,
    _model: ForwardModel,
    observables: List[Observable],
    parameters_to_adjust: List[AdjustableParameter],
    max_obs_time: Optional[float] = None,
) -> float:
    # Update the model with the new values of x (preconditioned)
    # Do not save parameters values (useless)
    update_model_with_parameters_values(
        _model, vector, parameter, is_preconditioned=True, is_to_save=False
    )
    # Solve the forward model with the new parameters
    ForwardSolver(_model).solve()

    return eval_model_loss_function(
        _model,
        observables,
        parameters_to_adjust,
        max_obs_time=max_obs_time,
    )


def compute_fd_gradient(
    model: ForwardModel,
    observables: Observables,
    parameters_to_adjust: AdjustableParameters,
    eps: Optional[float] = None,
    accuracy: int = 0,
    max_workers: int = 1,
    max_obs_time: Optional[float] = None,
    is_save_state: bool = True,
) -> NDArrayFloat:
    """Compute the gradient of the given parameters by finite difference approximation.

    Warning
    -------
    This function does not update the model (a copy is used instead).

    Parameters
    ----------
    model : ForwardModel
        The forward RT model.
    observables : Observables
        Sequence of observable instances.
    parameters_to_adjust : AdjustableParameters
        Sequence of adjusted parameter instances.
    eps: float, optional
        The epsilon for the computation of the approximated gradient by finite
        difference. If None, it is automatically inferred. The default is None.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (8 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    max_workers: int
        Number of workers used  if the gradient is approximated by finite
        differences. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None.
    is_save_state: bool
        Whether to save the FD gradient in memory. The default is True.

    """
    _model = copy.deepcopy(model)

    # clean the model to remove non pickeable objects
    _model.reinit()

    grad = np.array([], dtype=np.float64)
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # FD approximation -> only on the adjusted values. This is convenient to
        # test to gradient on a small portion of big models with to many grid cells to
        # be entirely tested.

        # Test the bounds -> it affects the finite differences evaluation
        param_values = get_parameter_values_from_model(_model, param)
        if np.any(param_values <= param.lbounds) or np.any(
            param_values >= param.ubounds
        ):
            warnings.warn(
                f'Adjusted parameter "{param.name}" has one or more values'
                " that equal the lower and/or upper bound(s). As values are clipped to"
                " bounds to avoid solver crashes, it will results in a wrong gradient"
                "approximation by finite differences "
                "(typically scaled by a factor 0.5)."
            )

        param_grad = finite_gradient(
            param.preconditioner(param_values.ravel("F")),
            _local_fun_loss,
            fm_args=(
                param,
                _model,
                observables,
                parameters_to_adjust,
                max_obs_time,
            ),
            eps=eps,
            accuracy=accuracy,
            max_workers=max_workers,
        )

        # 2) Create an array full of nan and fill it with the
        # # gradient (only at adjusted locations)
        # Then save it.
        if is_save_state:
            param.grad_fd_history.append(param_grad)

        # 3) Update grad
        grad = np.hstack((grad, param_grad.ravel("F")))
    return grad


def is_adjoint_gradient_correct(
    fwd_model: ForwardModel,
    adj_model: AdjointModel,
    parameters_to_adjust: AdjustableParameters,
    observables: Observables,
    eps: Optional[float] = None,
    accuracy: int = 0,
    max_workers: int = 1,
    hm_end_time: Optional[float] = None,
    is_verbose: bool = False,
    is_save_state: bool = True,
    max_nafpi: int = 30,
) -> bool:
    """
    Check if the gradient computed with the adjoint state is equal with FD.

    Parameters
    ----------
    fwd_model : ForwardModel
        _description_
    adj_model : AdjointModel
        _description_
    parameters_to_adjust : AdjustableParameters
        Sequence of adjusted parameter instances.
    observables : Observables
        Sequence of observable instances.
    eps: float, optional
        The epsilon for the computation of the approximated gradient by finite
        difference. If None, it is automatically inferred. The default is None.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (8 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    max_workers: int
        Number of workers used  if the gradient is approximated by finite
        differences. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    hm_end_time : Optional[float], optional
        Threshold time from which the observation are ignored, by default None.
    is_verbose: bool
        Whether to display info. The default is False.
    is_save_state: bool
        Whether to save the FD gradient in memory. The default is True.

    Returns
    -------
    bool
        True if the adjoint gradient is correct.
    """

    # Update parameters with model
    update_parameters_from_model(fwd_model, parameters_to_adjust)

    # Solve the forward problem
    solver: ForwardSolver = ForwardSolver(fwd_model)
    solver.solve(is_verbose=is_verbose)

    # Solve the adjoint problem

    asolver: AdjointSolver = AdjointSolver(fwd_model, adj_model)
    asolver.solve(
        observables, hm_end_time=hm_end_time, is_verbose=is_verbose, max_nafpi=max_nafpi
    )

    adj_grad = compute_adjoint_gradient(
        fwd_model, asolver.adj_model, parameters_to_adjust, is_save_state=is_save_state
    )
    fd_grad = compute_fd_gradient(
        fwd_model,
        observables,
        parameters_to_adjust,
        eps=eps,
        accuracy=accuracy,
        max_workers=max_workers,
        max_obs_time=hm_end_time,
        is_save_state=is_save_state,
    )

    return is_all_close(adj_grad, fd_grad)
