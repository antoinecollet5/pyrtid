"""Provide gradient computation routines."""

import copy
import warnings
from typing import List, Optional

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.forward.models import GRAVITY, WATER_DENSITY, FlowRegime
from pyrtid.inverse.adjoint import AdjointModel, AdjointSolver
from pyrtid.inverse.adjoint.aflow_solver import make_initial_adj_flow_matrices
from pyrtid.inverse.loss_function import get_model_loss_function
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
    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the diffusion.
    """
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    eff_diffusion = fwd_model.tr_model.effective_diffusion
    porosity = fwd_model.tr_model.porosity

    if deriv_var == DerivationVariable.POROSITY:
        # Note: this is the diffusion, not the effective diffusion !
        term_in_effdiff_deriv = fwd_model.tr_model.diffusion
    elif deriv_var == DerivationVariable.DIFFUSION:
        term_in_effdiff_deriv = porosity

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion

    conc = fwd_model.tr_model.conc
    conc_post_tr = fwd_model.tr_model.conc
    # conc = fwd_model.tr_model.conc_post_tr
    aconc = adj_model.a_tr_model.a_conc

    grad = np.zeros(shape)

    # X axis contribution
    if shape[0] > 1:
        # Consider the x axis
        # Forward scheme
        dconc_fx = np.zeros(shape)
        dconc_fx[:-1, :, 1:] += (
            crank_diff * (conc_post_tr[1:, :, 1:] - conc_post_tr[:-1, :, 1:])
            + (1.0 - crank_diff) * (conc[1:, :, :-1] - conc[:-1, :, :-1])
        ) * (
            dxi_harmonic_mean(eff_diffusion[:-1, :], eff_diffusion[1:, :])
            * term_in_effdiff_deriv[:-1, :]
        )[
            :, :, np.newaxis
        ]

        daconc_fx = np.zeros(shape)
        daconc_fx[:-1, :, :] += aconc[1:, :, :] - aconc[:-1, :, :]

        # Backward scheme
        dconc_bx = np.zeros(shape)
        dconc_bx[1:, :, 1:] += (
            crank_diff * (conc_post_tr[:-1, :, 1:] - conc_post_tr[1:, :, 1:])
            + (1.0 - crank_diff) * (conc[:-1, :, :-1] - conc[1:, :, :-1])
        ) * (
            dxi_harmonic_mean(eff_diffusion[1:, :], eff_diffusion[:-1, :])
            * term_in_effdiff_deriv[1:, :]
        )[
            :, :, np.newaxis
        ]

        daconc_bx = np.zeros(shape)
        daconc_bx[1:, :, :] += aconc[:-1, :, :] - aconc[1:, :, :]

        # Gather the two schemes
        grad = (
            (dconc_fx * daconc_fx + dconc_bx * daconc_bx)
            * fwd_model.geometry.dy
            / fwd_model.geometry.dx
        )

    # Y axis contribution
    if shape[1] > 1:
        # Forward scheme
        dconc_fy = np.zeros(shape)
        dconc_fy[:, :-1, 1:] += (
            crank_diff * (conc_post_tr[:, 1:, 1:] - conc_post_tr[:, :-1, 1:])
            + (1.0 - crank_diff) * (conc[:, 1:, :-1] - conc[:, :-1, :-1])
        ) * (
            dxi_harmonic_mean(eff_diffusion[:, :-1], eff_diffusion[:, 1:])
            * term_in_effdiff_deriv[:, :-1]
        )[
            :, :, np.newaxis
        ]
        daconc_fy = np.zeros(shape)
        daconc_fy[:, :-1, :] += aconc[:, 1:, :] - aconc[:, :-1, :]

        # Bconckward scheme
        dconc_by = np.zeros(shape)
        dconc_by[:, 1:, 1:] += (
            crank_diff * (conc_post_tr[:, :-1, 1:] - conc_post_tr[:, 1:, 1:])
            + (1.0 - crank_diff) * (conc[:, :-1, :-1] - conc[:, 1:, :-1])
        ) * (
            dxi_harmonic_mean(eff_diffusion[:, 1:], eff_diffusion[:, :-1])
            * term_in_effdiff_deriv[:, 1:]
        )[
            :, :, np.newaxis
        ]
        daconc_by = np.zeros(shape)
        daconc_by[:, 1:, :] += aconc[:, :-1, :] - aconc[:, 1:, :]

        # Gather the two schemes
        grad += (
            (dconc_fy * daconc_fy + dconc_by * daconc_by)
            * fwd_model.geometry.dx
            / fwd_model.geometry.dy
        )

    # We sum along the temporal axis
    return -np.sum(grad, axis=-1)


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
    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the diffusion.
    """
    grad = get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.DIFFUSION
    )
    # Add the adjoint sources for initial time (t0)
    return grad + adj_model.a_tr_model.a_diffusion_sources.getcol(0).todense().reshape(
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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

    Returns
    -------
    NDArrayFloat
        Gradient of the objective function with respect to the porosity.
    """
    conc = fwd_model.tr_model.conc
    conc_post_tr = fwd_model.tr_model.conc
    aconc = adj_model.a_tr_model.a_conc

    grad = (
        (conc_post_tr[:, :, 1:] - conc[:, :, :-1])
        / fwd_model.time_params.dt
        * aconc[:, :, 1:]
    ) * fwd_model.geometry.mesh_volume

    # We sum along the temporal axis + get the diffusion gradient
    grad = -np.sum(grad, axis=-1) + get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.POROSITY
    )
    # Add the adjoint sources for initial time (t0)
    return grad + adj_model.a_tr_model.a_porosity_sources.getcol(0).todense().reshape(
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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

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
    return grad + adj_model.a_fl_model.a_permeability_sources.getcol(
        0
    ).todense().reshape(grad.shape, order="F")


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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

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
            dhead_fx[:-1, :, :1] = (
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
            dhead_bx[1:, :, :1] = (
                head[:-1, :, :1] - head[1:, :, :1]
            ) * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]

        # Gather the two schemes
        grad += (
            (dhead_fx * dahead_fx + dhead_bx * dahead_bx)
            * fwd_model.geometry.dy
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
                (head[:, :-1, :1] - head[:, 1:, :1])
            ) * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]

        # Gather the two schemes
        grad += (
            (dhead_fy * dahead_fy + dhead_by * dahead_by)
            * fwd_model.geometry.dx
            / fwd_model.geometry.dy
        )

    # We sum along the temporal axis
    return -np.sum(grad, axis=-1)


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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

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
    apressure = adj_model.a_fl_model.a_pressure
    ma_ahead = np.zeros(apressure.shape)
    free_head_indices = fwd_model.fl_model.free_head_indices
    ma_ahead[free_head_indices[0], free_head_indices[1], :] = apressure[
        free_head_indices[0], free_head_indices[1], :
    ]
    grad = np.zeros(shape)

    # Consider the x axis
    if shape[0] > 1:
        # Forward scheme
        dhead_fx = np.zeros(shape)
        dhead_fx[:-1, :, 1:] += (
            crank_flow * (pressure[1:, :, 1:] - pressure[:-1, :, 1:])
            + (1.0 - crank_flow) * (pressure[1:, :, :-1] - pressure[:-1, :, :-1])
        ) * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
            :, :, np.newaxis
        ]

        dahead_fx = np.zeros(shape)
        dahead_fx[:-1, :, :] += ma_ahead[1:, :, :] - ma_ahead[:-1, :, :]

        # Handle the stationary case
        # if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
        #     dhead_fx[:-1, :, :1] = (
        #         head[1:, :, :1] - head[:-1, :, :1]
        #     ) * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
        #         :, :, np.newaxis
        #     ]

        # Bheadkward scheme
        dhead_bx = np.zeros(shape)
        dhead_bx[1:, :, 1:] += (
            crank_flow * (pressure[:-1, :, 1:] - pressure[1:, :, 1:])
            + (1.0 - crank_flow) * (pressure[:-1, :, :-1] - pressure[1:, :, :-1])
        ) * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
            :, :, np.newaxis
        ]
        dahead_bx = np.zeros(shape)

        dahead_bx[1:, :, :] += ma_ahead[:-1, :, :] - ma_ahead[1:, :, :]

        # Handle the stationary case
        # if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
        #     dhead_bx[1:, :, :1] = (
        #         head[:-1, :, :1] - head[1:, :, :1]
        #     ) * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
        #         :, :, np.newaxis
        #     ]

        # Gather the two schemes
        grad += (
            (dhead_fx * dahead_fx + dhead_bx * dahead_bx)
            * fwd_model.geometry.dy
            / fwd_model.geometry.dx
        )

    # Consider the y axis for 2D cases
    if shape[1] > 1:
        # Forward scheme
        dhead_fy = np.zeros(shape)
        dhead_fy[:, :-1, 1:] += (
            crank_flow * (pressure[:, 1:, 1:] - pressure[:, :-1, 1:])
            + (1.0 - crank_flow) * (pressure[:, 1:, :-1] - pressure[:, :-1, :-1])
        ) * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
            :, :, np.newaxis
        ]
        dahead_fy = np.zeros(shape)
        dahead_fy[:, :-1, :] += ma_ahead[:, 1:, :] - ma_ahead[:, :-1, :]

        # Handle the stationary case
        # if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
        #     dhead_fy[:, :-1, :1] += (
        #         head[:, 1:, :1] - head[:, :-1, :1]
        #     ) * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
        #         :, :, np.newaxis
        #     ]

        # Bheadkward scheme
        dhead_by = np.zeros(shape)
        dhead_by[:, 1:, 1:] += (
            crank_flow * (pressure[:, :-1, 1:] - pressure[:, 1:, 1:])
            + (1.0 - crank_flow) * (pressure[:, :-1, :-1] - pressure[:, 1:, :-1])
        ) * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
            :, :, np.newaxis
        ]
        dahead_by = np.zeros(shape)
        dahead_by[:, 1:, :] += ma_ahead[:, :-1, :] - ma_ahead[:, 1:, :]
        # Handle the stationary case

        # if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
        #     dhead_by[:, 1:, :1] += (
        #         (head[:, :-1, :1] - head[:, 1:, :1])
        #     ) * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
        #         :, :, np.newaxis
        #     ]

        # Gather the two schemes
        grad += (
            (dhead_fy * dahead_fy + dhead_by * dahead_by)
            * fwd_model.geometry.dx
            / fwd_model.geometry.dy
        )

    # We sum along the temporal axis
    return -np.sum(grad, axis=-1)


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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using mob observations.
    """
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    head = fwd_model.fl_model.head
    a_u_darcy_x = adj_model.a_fl_model.a_u_darcy_x

    # Consider the x axis
    # Forward scheme
    dhead_fx = np.zeros(shape)
    dhead_fx[:-1, :, :] += (
        ((head[:-1, :, :] - head[1:, :, :]))
        * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[:, :, np.newaxis]
        * a_u_darcy_x
    )

    # Bconckward scheme
    dhead_bx = np.zeros(shape)
    dhead_bx[1:, :, :] -= (
        ((head[1:, :, :] - head[:-1, :, :]))
        * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[:, :, np.newaxis]
        * a_u_darcy_x
    )

    # Gather the two schemes
    grad = (dhead_fx + dhead_bx) / fwd_model.geometry.dx

    # Consider the y axis for 2D cases
    if shape[1] != 1:
        a_u_darcy_y = adj_model.a_fl_model.a_u_darcy_y
        # Forward scheme
        dhead_fy = np.zeros(shape)
        dhead_fy[:, :-1, :] += (
            ((head[:, :-1, :] - head[:, 1:, :]))
            * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y
        )

        # Bconckward scheme
        dhead_by = np.zeros(shape)
        dhead_by[:, 1:, :] -= (
            ((head[:, 1:, :] - head[:, :-1, :]))
            * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y
        )
        # Gather the two schemes
        grad += (dhead_fy + dhead_by) / fwd_model.geometry.dy

    # We sum along the temporal axis
    return -np.sum(grad, axis=-1)


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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using mob observations.
    """
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt)
    permeability = fwd_model.fl_model.permeability

    pressure = fwd_model.fl_model.pressure
    a_u_darcy_x = adj_model.a_fl_model.a_u_darcy_x

    # Consider the x axis
    # Forward scheme
    dpressure_fx = np.zeros(shape)
    dpressure_fx[:-1, :, :] += (
        ((pressure[:-1, :, :] - pressure[1:, :, :]))
        * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[:, :, np.newaxis]
        * a_u_darcy_x
    )

    # Bconckward scheme
    dpressure_bx = np.zeros(shape)
    dpressure_bx[1:, :, :] -= (
        ((pressure[1:, :, :] - pressure[:-1, :, :]))
        * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[:, :, np.newaxis]
        * a_u_darcy_x
    )

    # Gather the two schemes
    grad = (
        (dpressure_fx + dpressure_bx) / fwd_model.geometry.dx / GRAVITY / WATER_DENSITY
    )

    # Consider the y axis for 2D cases
    if shape[1] != 1:
        a_u_darcy_y = adj_model.a_fl_model.a_u_darcy_y
        # Forward scheme
        dpressure_fy = np.zeros(shape)
        dpressure_fy[:, :-1, :] += (
            ((pressure[:, :-1, :] - pressure[:, 1:, :]))
            * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y
        )

        # Bconckward scheme
        dpressure_by = np.zeros(shape)
        dpressure_by[:, 1:, :] -= (
            ((pressure[:, 1:, :] - pressure[:, :-1, :]))
            * dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            * a_u_darcy_y
        )
        # Gather the two schemes
        grad += (
            (dpressure_fy + dpressure_by)
            / fwd_model.geometry.dy
            / GRAVITY
            / WATER_DENSITY
        )

    # We sum along the temporal axis
    return -np.sum(grad, axis=-1)


def get_storage_coefficient_adjoint_gradient(
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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

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
    return -np.sum(
        grad, axis=-1
    ) + adj_model.a_fl_model.a_storage_coefficient_sources.getcol(0).todense().reshape(
        (fwd_model.geometry.nx, fwd_model.geometry.ny), order="F"
    )


def get_initial_grade_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).
    """
    grad = (
        adj_model.a_tr_model.a_conc[:, :, 1]
        / fwd_model.time_params.ldt[0]
        * fwd_model.tr_model.porosity
        * fwd_model.geometry.mesh_volume
    ) + adj_model.a_tr_model.a_grade[:, :, 1] * (
        1
        + fwd_model.time_params.ldt[0]
        * fwd_model.gch_params.kv
        * fwd_model.gch_params.As
        * (1.0 - fwd_model.tr_model.lconc[1] / fwd_model.gch_params.Ks)
    )
    # Add adjoint sources for time t=0
    return grad + adj_model.a_tr_model.a_grade_sources.getcol(0).todense().reshape(
        grad.shape, order="F"
    )


def get_initial_head_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    r"""
    Gradient with respect to mineral phase initial concentrations.

    The gradient reads

    # TODO

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).
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
    grad = (
        q_prev.dot(a_head[:, 1])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Computation w.r.t. \lambda^{0} -> implicit part
    grad -= (
        q_next.dot(a_head[:, 0])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Add adjoint sources for t=0
    # 1) head sources
    grad += (
        adj_model.a_fl_model.a_head_sources.getcol(0).todense().ravel("F")
    )  # type: ignore
    # 2) pressure sources
    grad += (
        adj_model.a_fl_model.a_pressure_sources.getcol(0).todense().ravel("F")
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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).
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
    grad = (
        q_prev.dot(a_pressure[:, 1])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Computation w.r.t. \lambda^{0} -> implicit part
    grad -= (
        q_next.dot(a_pressure[:, 0])
        * fwd_model.geometry.mesh_volume
        * fwd_model.fl_model.storage_coefficient.ravel("F")
    )

    # Add adjoint sources for t=0
    # 1) pressure sources
    grad += (
        adj_model.a_fl_model.a_pressure_sources.getcol(0).todense().ravel("F")
    )  # type: ignore
    # 2) head sources
    grad += (
        adj_model.a_fl_model.a_head_sources.getcol(0).todense().ravel("F")
        / GRAVITY
        / fwd_model.tr_model.ldensity[0].ravel("F")
    )

    # Add adjoint sources for time t=0
    return grad.reshape(fwd_model.fl_model.lhead[0].shape, order="F")


def get_initial_conc_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    r"""
    Gradient with respect to aqueous phase initial concentrations.

    The gradient reads

    .. math::

        \dfrac{\partial \mathcal{L}}{\partial c_{i}^{0}} =
        \dfrac{V_{i}\omega_{e, i}}{\Delta t^{0}} \lambda_{c_{i}}^{1}
        - \dfrac{c_{i}^{0, \mathrm{obs}}
        - c_{i}^{0, \mathrm{calc}}}{\left(\sigma_{c_{i}}^{0, \mathrm{obs}}\right)^{2}}
        + \sum_{neigh \;j} \left\lVert \Gamma_{ij} \right\rVert D_{e, ij}
        (1 - \alpha_{\mathrm{d}})\dfrac{\lambda_{c_{j}}^{1} - \lambda_{c_{i}}^{1}}{
            \left\lVert \overrightarrow{\mathrm{P}_{i}\mathrm{P}_{j}} \right\rVert}

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).
    """
    a_tr_model = adj_model.a_tr_model
    tr_model = fwd_model.tr_model

    grad = (
        adj_model.a_tr_model.a_conc[:, :, 1]
        * fwd_model.geometry.mesh_volume
        * fwd_model.tr_model.porosity
        / fwd_model.time_params.ldt[0]
    )

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion
    a_conc = adj_model.a_tr_model.a_conc[:, :, 1]

    # X axis contribution
    if a_tr_model.a_conc.shape[0] > 1:
        dmean = harmonic_mean(
            tr_model.effective_diffusion[:-1, :], tr_model.effective_diffusion[1:, :]
        )
        tmp = fwd_model.geometry.dy / fwd_model.geometry.dx
        # Forward scheme
        grad[:-1, :] += (
            +(1.0 - crank_diff) * (a_conc[1:, :] - a_conc[:-1, :]) * dmean
        ) * tmp
        # Backward scheme
        grad[1:, :] += (
            +(1.0 - crank_diff) * (a_conc[:-1, :] - a_conc[1:, :]) * dmean
        ) * tmp

    # Y axis contribution
    if a_tr_model.a_conc.shape[1] > 1:
        dmean = harmonic_mean(
            tr_model.effective_diffusion[:, :-1], tr_model.effective_diffusion[:, 1:]
        )
        tmp = fwd_model.geometry.dx / fwd_model.geometry.dy
        # Forward scheme
        grad[:, :-1] += (
            +(1.0 - crank_diff) * (a_conc[:, 1:] - a_conc[:, :-1]) * dmean
        ) * tmp
        # Backward scheme
        grad[:, 1:] += (
            +(1.0 - crank_diff) * (a_conc[:, :-1] - a_conc[:, 1:]) * dmean
        ) * tmp

    return grad + adj_model.a_tr_model.a_conc_sources.getcol(0).todense().reshape(
        grad.shape, order="F"
    )


def compute_param_adjoint_ls_loss_function_gradient(
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

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).

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
        return get_initial_conc_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.INITIAL_GRADE:
        return get_initial_grade_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.INITIAL_HEAD:
        if fwd_model.fl_model.is_gravity:
            raise RuntimeError(
                "Cannot optimize the initial head when using a density flow"
                " (gravity is on). Optimize the initial pressure instead!"
            )
        return get_initial_head_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.STORAGE_COEFFICIENT:
        return get_storage_coefficient_adjoint_gradient(fwd_model, adj_model)
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
    jreg_weight: float = 1.0,
) -> NDArrayFloat:
    """
    Compute the gradient of the given parameters with the adjoint state.

    Note
    ----
    Adjoint gradient computation step 3: The gradient has to be multiplied
    by 1 / first preconditioner_1st_derivative(m]) with m the adjusted parameter
    because the preconditioner operates a variable
    change in the objective function: The new objective function J is J2(m2) = J[m],
    with m the adjusted parameter vector. Then the gradient is dJ2/dm2
    = dm/dm2 * dJ/dm.
    - Example 1 : we defined m2 = k * m -> dJ2/dm2 = dm/dm2 dJ/dm = 1/k * dj/dm. If
    k = 1/100 --> the gradient is a 100 times stronger. The parameter update is
    performed on m2 and not on m.
    - Example 2: we defined m2 = log(m) -> dJ2/dm2 = dm/dm2 dJ/dm = m * dj/dm

    """
    grad = np.array([], dtype=np.float64)
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # 1) least square loss function gradient with regards to observations
        param_grad_ls = compute_param_adjoint_ls_loss_function_gradient(
            fwd_model, adj_model, param
        )
        # 2) regularization loss function gradient
        # -> this already manages the preconditioning
        if jreg_weight == 0.0:
            param_grad_reg = 0.0
        else:
            param_grad_reg = (
                param.get_regularization_loss_function_gradient() * jreg_weight
            )
        # 3) Take into account the preconditioning to
        # the ls gradient (no need for reg gradient)
        param_grad = (param_grad_ls) / param.preconditioner_1st_derivative(
            get_parameter_values_from_model(fwd_model, param)
        )

        # TODO: see if it applies to the preconditioned gradient or not ?
        # 4) Smooth the gradient
        for filt in param.filters:
            param_grad = filt.filter(
                param_grad,
                len(param.archived_adjoint_gradients),
            )

        # Apply the regularization term (after the filtering step)
        param_grad += param_grad_reg

        # 5) Save the gradient (before the sub sampling with span)
        # i.e. even if we optimize a sub area of the grid, we store the gradient
        # everywhere because we have it for no extra cost.
        param.archived_adjoint_gradients.append(param_grad.copy())

        # 6) Apply parameter spanning (sub sampling) and make it 1D + update the
        # global gradient vector
        grad = np.hstack((grad, param_grad[param.span].ravel()))
    return grad


def _local_fun(
    vector: NDArrayFloat,
    parameter: AdjustableParameter,
    _model: ForwardModel,
    observables: List[Observable],
    parameters_to_adjust: List[AdjustableParameter],
    jreg_weight: float,
    max_obs_time: Optional[float] = None,
) -> float:
    # Update the model with the new values of x (preconditioned)
    # Do not save parameters values (useless)
    update_model_with_parameters_values(
        _model, vector, parameter, is_preconditioned=True, is_to_save=False
    )
    # Solve the forward model with the new parameters
    ForwardSolver(_model).solve()

    return get_model_loss_function(
        _model,
        observables,
        parameters_to_adjust,
        max_obs_time=max_obs_time,
        jreg_weight=jreg_weight,
    )


def compute_fd_gradient(
    model: ForwardModel,
    observables: Observables,
    parameters_to_adjust: AdjustableParameters,
    jreg_weight=1.0,
    eps: Optional[float] = None,
    max_workers: int = 1,
    max_obs_time: Optional[float] = None,
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

    """
    _model = copy.deepcopy(model)

    grad = np.array([], dtype=np.float64)
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # FD approximation -> only on the adjusted values. This is convenient to
        # test to gradient on a small portion of big models with to many meshes to
        # be entirely tested.

        # Test the bounds -> it affects the finite differences evaluation
        param_values = get_parameter_values_from_model(_model, param)
        if np.any(param_values <= param.lbound) or np.any(param_values >= param.ubound):
            warnings.warn(
                f'Adjusted parameter "{param.name}" has one or more values'
                " that equal the lower and/or upper bound(s). As values are clipped to"
                " bounds to avoid solver crashes, it will results in a wrong gradient"
                "approximation by finite differences "
                "(typically scaled by a factor 0.5)."
            )

        param_grad = finite_gradient(
            param.get_sliced_field(
                get_parameter_values_from_model(_model, param), is_preconditioned=True
            ),
            _local_fun,
            fm_args=(
                param,
                _model,
                observables,
                parameters_to_adjust,
                jreg_weight,
                max_obs_time,
            ),
            eps=eps,
            max_workers=max_workers,
        )

        # 2) Create an array full of nan and fill it with the
        # # gradient (only at adjusted locations)
        # Then save it.
        _saved_values = np.full(param.values.shape, np.nan)
        _saved_values[param.span] = param_grad
        param.archived_fd_gradients.append(_saved_values)

        # 3) Update grad
        grad = np.hstack((grad, param_grad.ravel()))
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
    asolver.solve(observables, hm_end_time=hm_end_time, is_verbose=is_verbose)

    adj_grad = compute_adjoint_gradient(
        fwd_model, asolver.adj_model, parameters_to_adjust
    )
    fd_grad = compute_fd_gradient(
        fwd_model, observables, parameters_to_adjust, eps=eps, max_workers=max_workers
    )

    return is_all_close(adj_grad, fd_grad)
