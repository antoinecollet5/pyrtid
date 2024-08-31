"""Provide gradient computation routines."""

import copy
import warnings
from typing import List, Optional

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.forward.flow_solver import get_rhomean2
from pyrtid.forward.models import GRAVITY, WATER_DENSITY, FlowRegime, VerticalAxis
from pyrtid.inverse.adjoint import AdjointModel, AdjointSolver
from pyrtid.inverse.adjoint.aflow_solver import (
    get_adjoint_transport_src_terms,
    make_initial_adj_flow_matrices,
)
from pyrtid.inverse.adjoint.ageochem_solver import ddMdimmobprev
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
from pyrtid.utils import StrEnum, finite_gradient, is_all_close
from pyrtid.utils.means import dxi_harmonic_mean, harmonic_mean
from pyrtid.utils.types import NDArrayFloat, object_or_object_sequence_to_list


class DerivationVariable(StrEnum):
    POROSITY = "porosity"
    DIFFUSION = "diffusion"


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
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    eff_diffusion = fwd_model.tr_model.effective_diffusion
    porosity = fwd_model.tr_model.porosity

    grad = np.zeros(shape)

    if deriv_var == DerivationVariable.POROSITY:
        # Note: this is the diffusion, not the effective diffusion !
        term_in_effdiff_deriv = fwd_model.tr_model.diffusion
    elif deriv_var == DerivationVariable.DIFFUSION:
        term_in_effdiff_deriv = porosity

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion

    for sp in range(fwd_model.tr_model.n_sp):
        mob = fwd_model.tr_model.mob[sp]
        # mob = fwd_model.tr_model.mob_post_tr
        amob = adj_model.a_tr_model.a_mob[sp]

        # X axis contribution
        if shape[0] > 1:
            # Consider the x axis
            # Forward scheme
            dconc_fx = np.zeros(shape)
            dconc_fx[:-1, :, 1:] += (
                crank_diff * (mob[1:, :, 1:] - mob[:-1, :, 1:])
                + (1.0 - crank_diff) * (mob[1:, :, :-1] - mob[:-1, :, :-1])
            ) * (
                dxi_harmonic_mean(eff_diffusion[:-1, :], eff_diffusion[1:, :])
                * term_in_effdiff_deriv[:-1, :]
            )[:, :, np.newaxis]

            damob_fx = np.zeros(shape)
            damob_fx[:-1, :, :] += amob[1:, :, :] - amob[:-1, :, :]

            # Backward scheme
            dconc_bx = np.zeros(shape)
            dconc_bx[1:, :, 1:] += (
                crank_diff * (mob[:-1, :, 1:] - mob[1:, :, 1:])
                + (1.0 - crank_diff) * (mob[:-1, :, :-1] - mob[1:, :, :-1])
            ) * (
                dxi_harmonic_mean(eff_diffusion[1:, :], eff_diffusion[:-1, :])
                * term_in_effdiff_deriv[1:, :]
            )[:, :, np.newaxis]

            damob_bx = np.zeros(shape)
            damob_bx[1:, :, :] += amob[:-1, :, :] - amob[1:, :, :]

            # Gather the two schemes
            grad += (
                (dconc_fx * damob_fx + dconc_bx * damob_bx)
                * fwd_model.geometry.gamma_ij_x
                / fwd_model.geometry.dx
            )

        # Y axis contribution
        if shape[1] > 1:
            # Forward scheme
            dconc_fy = np.zeros(shape)
            dconc_fy[:, :-1, 1:] += (
                crank_diff * (mob[:, 1:, 1:] - mob[:, :-1, 1:])
                + (1.0 - crank_diff) * (mob[:, 1:, :-1] - mob[:, :-1, :-1])
            ) * (
                dxi_harmonic_mean(eff_diffusion[:, :-1], eff_diffusion[:, 1:])
                * term_in_effdiff_deriv[:, :-1]
            )[:, :, np.newaxis]
            damob_fy = np.zeros(shape)
            damob_fy[:, :-1, :] += amob[:, 1:, :] - amob[:, :-1, :]

            # Bconckward scheme
            dconc_by = np.zeros(shape)
            dconc_by[:, 1:, 1:] += (
                crank_diff * (mob[:, :-1, 1:] - mob[:, 1:, 1:])
                + (1.0 - crank_diff) * (mob[:, :-1, :-1] - mob[:, 1:, :-1])
            ) * (
                dxi_harmonic_mean(eff_diffusion[:, 1:], eff_diffusion[:, :-1])
                * term_in_effdiff_deriv[:, 1:]
            )[:, :, np.newaxis]
            damob_by = np.zeros(shape)
            damob_by[:, 1:, :] += amob[:, :-1, :] - amob[:, 1:, :]

            # Gather the two schemes
            grad += (
                (dconc_fy * damob_fy + dconc_by * damob_by)
                * fwd_model.geometry.gamma_ij_y
                / fwd_model.geometry.dy
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
    grad = np.zeros(
        (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt - 1)
    )

    for sp in range(fwd_model.tr_model.n_sp):
        mob = fwd_model.tr_model.mob[sp]
        immob = fwd_model.tr_model.immob[sp]
        amob = adj_model.a_tr_model.a_mob[sp]

        grad += (
            (mob[:, :, 1:] - mob[:, :, :-1] + immob[:, :, 1:] - immob[:, :, :-1])
            / fwd_model.time_params.ldt
            * amob[:, :, 1:]
        ) * fwd_model.geometry.mesh_volume

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
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    if adj_model.a_fl_model.crank_nicolson is None:
        crank_flow: float = fwd_model.fl_model.crank_nicolson
    else:
        crank_flow = adj_model.a_fl_model.crank_nicolson

    head = fwd_model.fl_model.head
    ahead = adj_model.a_fl_model.a_head
    ma_ahead = np.zeros(ahead.shape)
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_ahead[free_head_indices[0], free_head_indices[1], :] = ahead[
        free_head_indices[0], free_head_indices[1], :
    ]
    grad = np.zeros(shape)

    # Consider the x axis
    if shape[0] > 1:
        # Forward scheme
        dhead_fx = np.zeros(shape)
        dhead_fx[:-1, :, 1:] += (
            crank_flow * (head[1:, :, 1:] - head[:-1, :, 1:])
            + (1.0 - crank_flow) * (head[1:, :, :-1] - head[:-1, :, :-1])
        ) * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
            :, :, np.newaxis
        ]

        dahead_fx = np.zeros(shape)
        dahead_fx[:-1, :, :] += ma_ahead[1:, :, :] - ma_ahead[:-1, :, :]

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dhead_fx[:-1, :, :1] += (
                head[1:, :, :1] - head[:-1, :, :1]
            ) * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                :, :, np.newaxis
            ]

        # Bheadkward scheme
        dhead_bx = np.zeros(shape)
        dhead_bx[1:, :, 1:] += (
            crank_flow * (head[:-1, :, 1:] - head[1:, :, 1:])
            + (1.0 - crank_flow) * (head[:-1, :, :-1] - head[1:, :, :-1])
        ) * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
            :, :, np.newaxis
        ]
        dahead_bx = np.zeros(shape)

        dahead_bx[1:, :, :] += ma_ahead[:-1, :, :] - ma_ahead[1:, :, :]

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dhead_bx[1:, :, :1] += (
                head[:-1, :, :1] - head[1:, :, :1]
            ) * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]

        # Gather the two schemes
        grad += (
            (dhead_fx * dahead_fx + dhead_bx * dahead_bx)
            * fwd_model.geometry.gamma_ij_x
            / fwd_model.geometry.dx
        )

    # Consider the y axis for 2D cases
    if shape[1] > 1:
        # Forward scheme
        dhead_fy = np.zeros(shape)
        dhead_fy[:, :-1, 1:] += (
            crank_flow * (head[:, 1:, 1:] - head[:, :-1, 1:])
            + (1.0 - crank_flow) * (head[:, 1:, :-1] - head[:, :-1, :-1])
        ) * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
            :, :, np.newaxis
        ]
        dahead_fy = np.zeros(shape)
        dahead_fy[:, :-1, :] += ma_ahead[:, 1:, :] - ma_ahead[:, :-1, :]

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dhead_fy[:, :-1, :1] += (
                head[:, 1:, :1] - head[:, :-1, :1]
            ) * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]

        # Bheadkward scheme
        dhead_by = np.zeros(shape)
        dhead_by[:, 1:, 1:] += (
            crank_flow * (head[:, :-1, 1:] - head[:, 1:, 1:])
            + (1.0 - crank_flow) * (head[:, :-1, :-1] - head[:, 1:, :-1])
        ) * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
            :, :, np.newaxis
        ]
        dahead_by = np.zeros(shape)
        dahead_by[:, 1:, :] += ma_ahead[:, :-1, :] - ma_ahead[:, 1:, :]
        # Handle the stationary case

        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dhead_by[:, 1:, :1] += (
                head[:, :-1, :1] - head[:, 1:, :1]
            ) * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]

        # Gather the two schemes
        grad += (
            (dhead_fy * dahead_fy + dhead_by * dahead_by)
            * fwd_model.geometry.gamma_ij_y
            / fwd_model.geometry.dy
        )

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
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    if adj_model.a_fl_model.crank_nicolson is None:
        crank_flow: float = fwd_model.fl_model.crank_nicolson
    else:
        crank_flow = adj_model.a_fl_model.crank_nicolson

    pressure = fwd_model.fl_model.pressure
    head = fwd_model.fl_model.head
    apressure = adj_model.a_fl_model.a_pressure
    ma_apressure = np.zeros(apressure.shape)
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_apressure[free_head_indices[0], free_head_indices[1], :] = apressure[
        free_head_indices[0], free_head_indices[1], :
    ]
    grad = np.zeros(shape)

    # Consider the x axis
    if fwd_model.geometry.nx > 1:
        rhomean_x = get_rhomean2(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=0,
            time_index=slice(0, -1),
            is_flatten=False,
        )[:-1]
        rhomean_x2 = get_rhomean2(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=0,
            time_index=slice(0, -1),
            is_flatten=False,
        )[:-1]
        tmp = 0.0
        if fwd_model.fl_model.vertical_axis == VerticalAxis.DX:
            tmp = rhomean_x**2 * GRAVITY

        # Forward scheme
        dpressure_fx = np.zeros(shape)
        dpressure_fx[:-1, :, 1:] += (
            (
                (
                    crank_flow * (pressure[1:, :, 1:] - pressure[:-1, :, 1:])
                    + (1.0 - crank_flow)
                    * (pressure[1:, :, :-1] - pressure[:-1, :, :-1])
                )
                / fwd_model.geometry.dx
                * rhomean_x2
                + tmp
            )
            * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                :, :, np.newaxis
            ]
            / WATER_DENSITY
        )

        dapressure_fx = np.zeros(shape)
        dapressure_fx[:-1, :, :] += ma_apressure[1:, :, :] - ma_apressure[:-1, :, :]

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dpressure_fx[:-1, :, :1] += (
                (head[1:, :, :1] - head[:-1, :, :1])
                * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                    :, :, np.newaxis
                ]
                / fwd_model.geometry.dx
            )

        # Bheadkward scheme
        dpressure_bx = np.zeros(shape)
        dpressure_bx[1:, :, 1:] += (
            (
                (
                    crank_flow * (pressure[:-1, :, 1:] - pressure[1:, :, 1:])
                    + (1.0 - crank_flow)
                    * (pressure[:-1, :, :-1] - pressure[1:, :, :-1])
                )
                / fwd_model.geometry.dx
                * rhomean_x2
                - tmp
            )
            * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]
            / WATER_DENSITY
        )

        dapressure_bx = np.zeros(shape)
        dapressure_bx[1:, :, :] += ma_apressure[:-1, :, :] - ma_apressure[1:, :, :]

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dpressure_bx[1:, :, :1] += (
                (head[:-1, :, :1] - head[1:, :, :1])
                * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                    :, :, np.newaxis
                ]
                / fwd_model.geometry.dx
            )

        # Gather the two schemes
        grad += (
            dpressure_fx * dapressure_fx + dpressure_bx * dapressure_bx
        ) * fwd_model.geometry.gamma_ij_x

    # Consider the y axis for 2D cases
    if fwd_model.geometry.ny > 1:
        rhomean_y = get_rhomean2(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=1,
            time_index=slice(0, -1),
            is_flatten=False,
        )[:, :-1]
        rhomean_y2 = get_rhomean2(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=1,
            time_index=slice(0, -1),
            is_flatten=False,
        )[:, :-1]
        tmp = 0.0
        if fwd_model.fl_model.vertical_axis == VerticalAxis.DY:
            tmp = rhomean_y**2 * GRAVITY

        # Forward scheme
        dpressure_fy = np.zeros(shape)
        dpressure_fy[:, :-1, 1:] += (
            (
                (
                    crank_flow * (pressure[:, 1:, 1:] - pressure[:, :-1, 1:])
                    + (1.0 - crank_flow)
                    * (pressure[:, 1:, :-1] - pressure[:, :-1, :-1])
                )
                / fwd_model.geometry.dy
                * rhomean_y2
                + tmp
            )
            * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            / WATER_DENSITY
        )

        dapressure_fy = np.zeros(shape)
        dapressure_fy[:, :-1, :] += ma_apressure[:, 1:, :] - ma_apressure[:, :-1, :]

        # Handle the stationary case
        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dpressure_fy[:, :-1, :1] += (
                (head[:, 1:, :1] - head[:, :-1, :1])
                * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                    :, :, np.newaxis
                ]
                / fwd_model.geometry.dy
            )

        # Bheadkward scheme
        dpressure_by = np.zeros(shape)
        dpressure_by[:, 1:, 1:] += (
            (
                (
                    crank_flow * (pressure[:, :-1, 1:] - pressure[:, 1:, 1:])
                    + (1.0 - crank_flow)
                    * (pressure[:, :-1, :-1] - pressure[:, 1:, :-1])
                )
                / fwd_model.geometry.dy
                * rhomean_y2
                - tmp
            )
            * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            / WATER_DENSITY
        )

        dapressure_by = np.zeros(shape)
        dapressure_by[:, 1:, :] += ma_apressure[:, :-1, :] - ma_apressure[:, 1:, :]
        # Handle the stationary case

        if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            dpressure_by[:, 1:, :1] += (
                (head[:, :-1, :1] - head[:, 1:, :1])
                * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                    :, :, np.newaxis
                ]
                / fwd_model.geometry.dy
            )

        # Gather the two schemes
        grad += (
            dpressure_fy * dapressure_fy + dpressure_by * dapressure_by
        ) * fwd_model.geometry.gamma_ij_y

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
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    head = fwd_model.fl_model.head

    grad = np.zeros_like(head)

    if fwd_model.geometry.nx > 1:
        a_u_darcy_x = adj_model.a_fl_model.a_u_darcy_x[1:-1, :, :]

        # Consider the x axis
        # Forward scheme
        dhead_fx = np.zeros(shape)
        dhead_fx[:-1, :, :] += (
            (head[1:, :, :] - head[:-1, :, :])
            * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                :, :, np.newaxis
            ]
            * a_u_darcy_x
        )

        # Bconckward scheme
        dhead_bx = np.zeros(shape)
        dhead_bx[1:, :, :] -= (
            (head[:-1, :, :] - head[1:, :, :])
            * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]
            * a_u_darcy_x
        )

        # Gather the two schemes
        grad += (dhead_fx + dhead_bx) / fwd_model.geometry.dx

    # Consider the y axis for 2D cases
    if fwd_model.geometry.ny > 1:
        a_u_darcy_y = adj_model.a_fl_model.a_u_darcy_y[:, 1:-1, :]
        # Forward scheme
        dhead_fy = np.zeros(shape)
        dhead_fy[:, :-1, :] += (
            (head[:, 1:, :] - head[:, :-1, :])
            * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y
        )

        # Bconckward scheme
        dhead_by = np.zeros(shape)
        dhead_by[:, 1:, :] -= (
            (head[:, :-1, :] - head[:, 1:, :])
            * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y
        )
        # Gather the two schemes
        grad += (dhead_fy + dhead_by) / fwd_model.geometry.dy

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
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    pressure = fwd_model.fl_model.pressure
    grad = np.zeros_like(pressure)

    if fwd_model.geometry.nx > 1:
        a_u_darcy_x = adj_model.a_fl_model.a_u_darcy_x[1:-1, :, :]

        if fwd_model.fl_model.vertical_axis == VerticalAxis.DX:
            rho_ij_g_x = (
                get_rhomean2(
                    fwd_model.geometry,
                    fwd_model.tr_model,
                    axis=0,
                    time_index=slice(0, -1),
                    is_flatten=False,
                )[:-1]
                * GRAVITY
            )
        else:
            rho_ij_g_x = 0.0

        # Consider the x axis
        # Forward scheme
        dpressure_fx = np.zeros(shape)
        dpressure_fx[:-1, :, 1:] += (
            (
                (pressure[1:, :, 1:] - pressure[:-1, :, 1:]) / fwd_model.geometry.dx
                + rho_ij_g_x
            )
            * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                :, :, np.newaxis
            ]
            * a_u_darcy_x[:, :, 1:]
        )

        # Bconckward scheme
        dpressure_bx = np.zeros(shape)
        dpressure_bx[1:, :, 1:] -= (
            (
                (pressure[:-1, :, 1:] - pressure[1:, :, 1:]) / fwd_model.geometry.dx
                - rho_ij_g_x
            )
            * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]
            * a_u_darcy_x[:, :, 1:]
        )

        # Gather the two schemes
        grad += (dpressure_fx + dpressure_bx) / GRAVITY / WATER_DENSITY

    # Consider the y axis for 2D cases
    if fwd_model.geometry.ny > 1:
        a_u_darcy_y = adj_model.a_fl_model.a_u_darcy_y[:, 1:-1, :]

        if fwd_model.fl_model.vertical_axis == VerticalAxis.DY:
            rho_ij_g_y = (
                get_rhomean2(
                    fwd_model.geometry,
                    fwd_model.tr_model,
                    axis=1,
                    time_index=slice(0, -1),
                    is_flatten=False,
                )[:, :-1]
                * GRAVITY
            )
        else:
            rho_ij_g_y = 0.0

        # Forward scheme
        dpressure_fy = np.zeros(shape)
        dpressure_fy[:, :-1, 1:] += (
            (
                (pressure[:, 1:, 1:] - pressure[:, :-1, 1:]) / fwd_model.geometry.dy
                + rho_ij_g_y
            )
            * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y[:, :, 1:]
        )

        # Bconckward scheme
        dpressure_by = np.zeros(shape)
        dpressure_by[:, 1:, 1:] -= (
            (
                (pressure[:, :-1, 1:] - pressure[:, 1:, 1:]) / fwd_model.geometry.dy
                - rho_ij_g_y
            )
            * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y[:, :, 1:]
        )
        # Gather the two schemes
        grad += (dpressure_fy + dpressure_by) / GRAVITY / WATER_DENSITY

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
    ma_ahead[free_head_indices[0], free_head_indices[1], :] = ahead[
        free_head_indices[0], free_head_indices[1], :
    ]

    grad = (
        (head[:, :, 1:] - head[:, :, :-1])
        * ma_ahead[:, :, 1:]
        / np.array(fwd_model.time_params.ldt)[np.newaxis, np.newaxis, :]
        * fwd_model.geometry.mesh_volume
    )

    # We sum along the temporal axis
    return np.sum(grad, axis=-1) + adj_model.a_fl_model.a_storage_coefficient_sources[
        :, [0]
    ].todense().reshape((fwd_model.geometry.nx, fwd_model.geometry.ny), order="F")


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
    ma_apressure[free_head_indices[0], free_head_indices[1], :] = apressure[
        free_head_indices[0], free_head_indices[1], :
    ]

    grad = (
        (pressure[:, :, 1:] - pressure[:, :, :-1])
        * ma_apressure[:, :, 1:]
        / np.array(fwd_model.time_params.ldt)[np.newaxis, np.newaxis, :]
        * fwd_model.geometry.mesh_volume
    )

    # We sum along the temporal axis
    return np.sum(grad, axis=-1) + adj_model.a_fl_model.a_storage_coefficient_sources[
        :, 0
    ].todense().reshape((fwd_model.geometry.nx, fwd_model.geometry.ny), order="F")


def get_initial_grade_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel, sp: int
) -> NDArrayFloat:
    r"""
    Gradient with respect to mineral phase initial concentrations.

    The gradient reads

    .. math::

        \dfrac{\partial \mathcal{L}}{\partial \overline{c}_{i}^{0}} =
        \dfrac{V_{i} \omega_{e, i}}{\Delta t^{0}} \lambda_{c_{i}}^{1}
        + \lambda_{\overline{c}_{i}}^{1} \left( 1 + \Delta t^{0} k_{v} A_{s}
        \left( 1 - \dfrac{c_{i}^{1}}{K_{s}}\right) \right)
        - \dfrac{\overline{c}_{i}^{0, \mathrm{obs}}
        - \overline{c}_{i}^{0, \mathrm{calc}}}{\left(
            \sigma_{\overline{c}_{i}}^{0, \mathrm{obs}}\right)^{2}}

    """
    # Prat common for all species
    grad = (
        -(
            adj_model.a_tr_model.a_mob[sp, :, :, 1]
            / fwd_model.time_params.ldt[0]
            * fwd_model.tr_model.porosity
            * fwd_model.geometry.mesh_volume
        )
        - adj_model.a_tr_model.a_immob[sp, :, :, 1]
    )
    # Add adjoint sources for time t=0
    grad += (
        adj_model.a_tr_model.a_grade_sources[sp][:, [0]]
        .todense()
        .reshape(grad.shape, order="F")
    )

    if sp == 0:
        grad -= (
            adj_model.a_tr_model.a_immob[0, :, :, 1]
            - fwd_model.gch_params.stocoef * adj_model.a_tr_model.a_immob[1, :, :, 1]
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
    r"""
    Gradient with respect to mineral phase initial concentrations.

    The gradient reads

    # TODO

    """
    a_head = adj_model.a_fl_model.a_head[:, :, :2].reshape((-1, 2), order="F")
    grad = np.zeros(a_head[:, 0].size, dtype=np.float64)

    # TODO: Need to take the darcy velocity into account

    (q_next, q_prev) = make_initial_adj_flow_matrices(
        fwd_model.geometry,
        fwd_model.fl_model,
        fwd_model.tr_model,
        adj_model.a_fl_model,
        fwd_model.time_params,
        is_q_prev_for_gradient=True,
    )

    # Computation w.r.t. \lambda^{1} -> explicit part
    grad = -(
        q_prev.dot(a_head[:, 1])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Computation w.r.t. \lambda^{0} -> implicit part
    grad += (
        q_next.dot(a_head[:, 0])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Derivative of the darcy equation
    grad -= get_adjoint_transport_src_terms(
        fwd_model.geometry, fwd_model.fl_model, adj_model.a_fl_model, 0, True
    )

    # Add adjoint sources for t=0
    # 1) head sources
    grad += adj_model.a_fl_model.a_head_sources[:, [0]].todense().ravel("F")  # type: ignore
    # 2) pressure sources
    grad += (
        adj_model.a_fl_model.a_pressure_sources[:, [0]].todense().ravel("F")
        * GRAVITY
        * WATER_DENSITY
    )

    # Add adjoint sources for time t=0
    return grad.reshape(fwd_model.fl_model.lhead[0].shape, order="F")


def get_initial_pressure_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    r"""
    Gradient with respect to mineral phase initial concentrations.

    The gradient reads

    # TODO

    """
    a_pressure = adj_model.a_fl_model.a_pressure[:, :, :2].reshape((-1, 2), order="F")
    grad = np.zeros(a_pressure[:, 0].size, dtype=np.float64)

    (q_next, q_prev) = make_initial_adj_flow_matrices(
        fwd_model.geometry,
        fwd_model.fl_model,
        fwd_model.tr_model,
        adj_model.a_fl_model,
        fwd_model.time_params,
        is_q_prev_for_gradient=True,
    )

    # Computation w.r.t. \lambda^{1} -> explicit part
    grad = -(
        q_prev.dot(a_pressure[:, 1])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Computation w.r.t. \lambda^{0} -> implicit part
    grad += (
        q_next.dot(a_pressure[:, 0])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Add adjoint sources for t=0
    # 1) pressure sources
    grad += adj_model.a_fl_model.a_pressure_sources[:, [0]].todense().ravel("F")  # type: ignore
    # 2) head sources
    grad += (
        adj_model.a_fl_model.a_head_sources[:, [0]].todense().ravel("F")
        / GRAVITY
        / fwd_model.tr_model.ldensity[0].ravel("F")
    )

    # Add adjoint sources for time t=0
    return grad.reshape(fwd_model.fl_model.lhead[0].shape, order="F")


def get_initial_conc_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel, sp: int
) -> NDArrayFloat:
    r"""
    Gradient with respect to aqueous phase initial concentrations.

    The gradient reads

    .. math::

        \dfrac{\partial \mathcal{L}}{\partial c_{i}^{0}} =
        \dfrac{V_{i}\omega_{e, i}}{\Delta t^{0}} \lambda_{c_{i}}^{1}
        - \dfrac{c_{i}^{0, \mathrm{obs}}
        - c_{i}^{0, \mathrm{calc}}}{\left(\sigma_{c_{i}}^{0, \mathrm{obs}}\right)^{2}}
        + \sum_{\mathrm{neigh} \;j} \mathcal{A}_{\Gamma_{ij}} D_{e, ij}
        (1 - \alpha_{\mathrm{d}})\dfrac{\lambda_{c_{j}}^{1} - \lambda_{c_{i}}^{1}}{
            \left\lVert \overrightarrow{\mathrm{P}_{i}\mathrm{P}_{j}} \right\rVert}

    """
    tr_model = fwd_model.tr_model

    grad = (
        adj_model.a_tr_model.a_mob[sp, :, :, 1]
        * fwd_model.geometry.mesh_volume
        * fwd_model.tr_model.porosity
        / fwd_model.time_params.ldt[0]
    )

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion
    a_mob = adj_model.a_tr_model.a_mob[sp, :, :, 1]

    # X axis contribution
    if fwd_model.geometry.nx > 1:
        dmean = harmonic_mean(
            tr_model.effective_diffusion[:-1, :], tr_model.effective_diffusion[1:, :]
        )
        tmp = fwd_model.geometry.gamma_ij_x / fwd_model.geometry.dx
        # Forward scheme
        grad[:-1, :] += (
            +(1.0 - crank_diff) * (a_mob[1:, :] - a_mob[:-1, :]) * dmean
        ) * tmp
        # Backward scheme
        grad[1:, :] += (
            +(1.0 - crank_diff) * (a_mob[:-1, :] - a_mob[1:, :]) * dmean
        ) * tmp

    # Y axis contribution
    if fwd_model.geometry.ny > 1:
        dmean = harmonic_mean(
            tr_model.effective_diffusion[:, :-1], tr_model.effective_diffusion[:, 1:]
        )
        tmp = fwd_model.geometry.gamma_ij_y / fwd_model.geometry.dy
        # Forward scheme
        grad[:, :-1] += (
            +(1.0 - crank_diff) * (a_mob[:, 1:] - a_mob[:, :-1]) * dmean
        ) * tmp
        # Backward scheme
        grad[:, 1:] += (
            +(1.0 - crank_diff) * (a_mob[:, :-1] - a_mob[:, 1:]) * dmean
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


def _local_fun(
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
    model: ForwardModel
        The forward RT model.
    eps: float, optional
        The epsilon for the computation of the approximated gradient by finite
        difference. If None, it is automatically inferred. The default is None.
    max_workers: int
        Number of workers used  if the gradient is approximated by finite
        differences. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None.
    is_save_state; bool
        Whether to save the FD gradient in memory. The default is True.

    """
    _model = copy.deepcopy(model)

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
            _local_fun,
            fm_args=(
                param,
                _model,
                observables,
                parameters_to_adjust,
                max_obs_time,
            ),
            eps=eps,
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
        _description_
    observables : Observables
        _description_
    eps: float, optional
        The epsilon for the computation of the approximated gradient by finite
        difference. If None, it is automatically inferred. The default is None.
    max_workers: int
        Number of workers used  if the gradient is approximated by finite
        differences. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    hm_end_time : Optional[float], optional
        Threshold time from which the observation are ignored, by default None.
    is_verbose: bool
        Whether to display info. The default is False.
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
        max_workers=max_workers,
        is_save_state=is_save_state,
    )

    return is_all_close(adj_grad, fd_grad)
