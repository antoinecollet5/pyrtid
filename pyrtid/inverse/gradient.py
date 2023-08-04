"""Provide gradient computation routines."""

import copy
from typing import Optional, Sequence, Union

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.forward.models import FlowRegime
from pyrtid.inverse.adjoint import AdjointModel, AdjointSolver
from pyrtid.inverse.loss_function import compute_model_loss_function
from pyrtid.inverse.obs import Observable
from pyrtid.inverse.params import (
    AdjustableParameter,
    ParameterName,
    get_parameter_values_from_model,
    update_model_with_parameters_values,
    update_parameters_from_model,
)
from pyrtid.utils import StrEnum, finite_gradient, is_all_close
from pyrtid.utils.means import dxi_harmonic_mean
from pyrtid.utils.types import NDArrayFloat, object_or_object_sequence_to_list


class DerivationVariable(StrEnum):
    POROSITY = "porosity"
    DIFFUSION = "diffusion"


def _get_diffusion_term_adjoint_gradient(
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
    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt + 1)
    eff_diffusion = fwd_model.tr_model.effective_diffusion
    porosity = fwd_model.tr_model.porosity

    if deriv_var == DerivationVariable.POROSITY:
        # Note: this is the diffusion, not the effective diffusion !
        term_in_effdiff_deriv = fwd_model.tr_model.diffusion
    elif deriv_var == DerivationVariable.DIFFUSION:
        term_in_effdiff_deriv = porosity

    crank_diff = fwd_model.tr_model.crank_nicolson_diffusion

    conc = fwd_model.tr_model.conc
    # conc = fwd_model.tr_model.conc_post_tr
    aconc = adj_model.tr_model.a_conc

    # Consider the x axis
    # Forward scheme
    dconc_fx = np.zeros(shape)
    dconc_fx[:-1, :, 1:] += (
        crank_diff * (conc[1:, :, 1:] - conc[:-1, :, 1:])
        + (1.0 - crank_diff) * (conc[1:, :, :-1] - conc[:-1, :, :-1])
    ) * (
        dxi_harmonic_mean(eff_diffusion[:-1, :], eff_diffusion[1:, :])
        * term_in_effdiff_deriv[:-1, :]
    )[
        :, :, np.newaxis
    ]

    daconc_fx = np.zeros(shape)
    daconc_fx[:-1, :, 1:] += aconc[1:, :, 1:] - aconc[:-1, :, 1:]

    # Backward scheme
    dconc_bx = np.zeros(shape)
    dconc_bx[1:, :, 1:] += (
        crank_diff * (conc[:-1, :, 1:] - conc[1:, :, 1:])
        + (1.0 - crank_diff) * (conc[:-1, :, :-1] - conc[1:, :, :-1])
    ) * (
        dxi_harmonic_mean(eff_diffusion[1:, :], eff_diffusion[:-1, :])
        * term_in_effdiff_deriv[1:, :]
    )[
        :, :, np.newaxis
    ]

    daconc_bx = np.zeros(shape)
    daconc_bx[1:, :, 1:] += aconc[:-1, :, 1:] - aconc[1:, :, 1:]

    # Gather the two schemes
    grad = (
        (dconc_fx * daconc_fx + dconc_bx * daconc_bx)
        * fwd_model.geometry.dy
        / fwd_model.geometry.dx
    )

    # Consider the y axis for 2D cases
    if shape[1] > 1:
        # Forward scheme
        dconc_fy = np.zeros(shape)
        dconc_fy[:, :-1, 1:] += (
            crank_diff * (conc[:, 1:, 1:] - conc[:, :-1, 1:])
            + (1.0 - crank_diff) * (conc[:, 1:, :-1] - conc[:, :-1, :-1])
        ) * (
            dxi_harmonic_mean(eff_diffusion[:, :-1], eff_diffusion[:, 1:])
            * term_in_effdiff_deriv[:, :-1]
        )[
            :, :, np.newaxis
        ]
        daconc_fy = np.zeros(shape)
        daconc_fy[:, :-1, 1:] += aconc[:, 1:, 1:] - aconc[:, :-1, 1:]

        # Bconckward scheme
        dconc_by = np.zeros(shape)
        dconc_by[:, 1:, 1:] += (
            crank_diff * (conc[:, :-1, 1:] - conc[:, 1:, 1:])
            + (1.0 - crank_diff) * (conc[:, :-1, :-1] - conc[:, 1:, :-1])
        ) * (
            dxi_harmonic_mean(eff_diffusion[:, 1:], eff_diffusion[:, :-1])
            * term_in_effdiff_deriv[:, 1:]
        )[
            :, :, np.newaxis
        ]
        daconc_by = np.zeros(shape)
        daconc_by[:, 1:, 1:] += aconc[:, :-1, 1:] - aconc[:, 1:, 1:]

        # Gather the two schemes
        grad += (
            (dconc_fy * daconc_fy + dconc_by * daconc_by)
            * fwd_model.geometry.dx
            / fwd_model.geometry.dy
        )

    # We sum along the temporal axis
    return -np.sum(grad, axis=-1)


def _get_diffusion_adjoint_gradient(
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
    return _get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.DIFFUSION
    )


def _get_porosity_adjoint_gradient(
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
    aconc = adj_model.tr_model.a_conc

    grad = (
        (conc[:, :, 1:] - conc[:, :, :-1]) / fwd_model.time_params.dt * aconc[:, :, 1:]
    ) * fwd_model.geometry.mesh_area

    # We sum along the temporal axis + get the diffusion gradient
    return -np.sum(grad, axis=-1) + _get_diffusion_term_adjoint_gradient(
        fwd_model, adj_model, DerivationVariable.POROSITY
    )


def _get_permeability_adjoint_gradient(
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
    return _get_perm_gradient_from_diffusivity_eq(
        fwd_model, adj_model
    ) + _get_perm_gradient_from_darcy_eq(fwd_model, adj_model)


def _get_perm_gradient_from_diffusivity_eq(
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
    # if not adj_model.is_head_obs:
    #     return np.zeros((fwd_model.geometry.nx, fwd_model.geometry.ny))

    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt + 1)
    permeability = fwd_model.fl_model.permeability

    crank_flow = fwd_model.fl_model.crank_nicolson

    head = fwd_model.fl_model.head
    ahead = adj_model.fl_model.a_head

    # Consider the x axis
    # Forward scheme
    dhead_fx = np.zeros(shape)
    dhead_fx[:-1, :, 1:] += (
        crank_flow * (head[1:, :, 1:] - head[:-1, :, 1:])
        + (1.0 - crank_flow) * (head[1:, :, :-1] - head[:-1, :, :-1])
    ) * dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[:, :, np.newaxis]
    dahead_fx = np.zeros(shape)
    dahead_fx[:-1, :, :] += ahead[1:, :, :] - ahead[:-1, :, :]
    # Handle the stationary case
    if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
        dhead_fx[:-1, :, :1] = (head[1:, :, :1] - head[:-1, :, :1]) * dxi_harmonic_mean(
            permeability[:-1, :], permeability[1:, :]
        )[:, :, np.newaxis]

    # Bheadkward scheme
    dhead_bx = np.zeros(shape)
    dhead_bx[1:, :, 1:] += (
        crank_flow * (head[:-1, :, 1:] - head[1:, :, 1:])
        + (1.0 - crank_flow) * (head[:-1, :, :-1] - head[1:, :, :-1])
    ) * dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[:, :, np.newaxis]
    dahead_bx = np.zeros(shape)
    dahead_bx[1:, :, :] += ahead[:-1, :, :] - ahead[1:, :, :]
    # Handle the stationary case
    if fwd_model.fl_model.regime == FlowRegime.STATIONARY:
        dhead_bx[1:, :, :1] = (head[:-1, :, :1] - head[1:, :, :1]) * dxi_harmonic_mean(
            permeability[1:, :], permeability[:-1, :]
        )[:, :, np.newaxis]

    # Gather the two schemes
    grad = (
        (dhead_fx * dahead_fx + dhead_bx * dahead_bx)
        * fwd_model.geometry.dy
        / fwd_model.geometry.dx
    )

    # Consider the y axis for 2D cases
    if shape[1] != 1:
        # Forward scheme
        dhead_fy = np.zeros(shape)
        dhead_fy[:, :-1, 1:] += (
            crank_flow * (head[:, 1:, 1:] - head[:, :-1, 1:])
            + (1.0 - crank_flow) * (head[:, 1:, :-1] - head[:, :-1, :-1])
        ) * dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
            :, :, np.newaxis
        ]
        dahead_fy = np.zeros(shape)
        dahead_fy[:, :-1, :] += ahead[:, 1:, :] - ahead[:, :-1, :]
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
        dahead_by[:, 1:, :] += ahead[:, :-1, :] - ahead[:, 1:, :]
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


def _get_perm_gradient_from_darcy_eq(
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
    if not adj_model.is_mob_obs:
        return np.zeros((fwd_model.geometry.nx, fwd_model.geometry.ny))

    shape = (fwd_model.geometry.nx, fwd_model.geometry.ny, fwd_model.time_params.nt + 1)
    permeability = fwd_model.fl_model.permeability

    head = fwd_model.fl_model.head
    a_u_darcy_x = adj_model.fl_model.a_u_darcy_x

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
        a_u_darcy_y = adj_model.fl_model.a_u_darcy_y
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


def _get_mineral_grade_adjoint_gradient(
    fwd_model: ForwardModel, adj_model: AdjointModel
) -> NDArrayFloat:
    """Gradient with respect to initial mineral phase concentration.

    gradM = wdata * (acM0[:, :, 0] + acT0[:, :, 0]) / dt + acM0[:, :, 0] * kvAs * (
        1 - cT0[:, :, 0] / Ks
    )
    # Regularisation
    gradM[iymineral, ixmineral] += wmineral * (cM0[iymineral, ixmineral, 0] - vmineral)
    return gradM

    Note
    ----
    Parameter span is not taken into account which means that the gradient is
    computed on the full domain (grid).
    """
    return (
        (adj_model.tr_model.a_grade[:, :, 0] + adj_model.tr_model.a_conc[:, :, 0])
        / fwd_model.time_params.dt
        + adj_model.tr_model.a_grade[:, :, 0]
        * fwd_model.gch_params.kv
        * fwd_model.gch_params.As
        * (1.0 - fwd_model.tr_model.conc[:, :, 0] / fwd_model.gch_params.Ks)
    ) * fwd_model.geometry.mesh_area


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
        return _get_diffusion_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.POROSITY:
        return _get_porosity_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.PERMEABILITY:
        return _get_permeability_adjoint_gradient(fwd_model, adj_model)
    if param.name == ParameterName.INITIAL_CONCENTRATION:
        raise (
            NotImplementedError("Please contact the developer to handle this issue.")
        )
    elif param.name == ParameterName.INITIAL_MINERAL_GRADE:
        return _get_mineral_grade_adjoint_gradient(fwd_model, adj_model)
    raise (NotImplementedError("Please contact the developer to handle this issue."))


def compute_adjoint_gradient(
    fwd_model: ForwardModel,
    adj_model: AdjointModel,
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    jreg_weight: float = 1.0,
) -> NDArrayFloat:
    """
    Compute the gradient of the given parameters with the adjoint state.

    Note
    ----
    Adjoint gradient computation step 3: The gradient has to be mutiplied
    by 1 / first preconditioner_1st_derivative(m), with m the adjusted parameter
    because the preconditioner operates a variable
    change in the objective function: The new objective function J is J2(m2) = J(m),
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
        param_grad_reg = param.get_regularization_loss_function_gradient() * jreg_weight

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


def compute_fd_gradient(
    model: ForwardModel,
    observables: Union[Observable, Sequence[Observable]],
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    jreg_weight=1.0,
    eps: Optional[float] = None,
) -> NDArrayFloat:
    """Compute the gradient of the given parameters by finite difference approximation.

    Warning
    -------
    This function does not update the model (a copy is used instead).
    """
    _model = copy.deepcopy(model)

    def _local_fun(vector: NDArrayFloat, parameter: AdjustableParameter) -> float:
        # model = copy.deepcopy(model)
        # Update the model with the new values of x (preconditioned)
        # Do not save parameters values (useless)
        update_model_with_parameters_values(
            _model, vector, parameter, is_preconditioned=True, is_to_save=False
        )
        # Solve the forward model with the new parameters
        ForwardSolver(_model).solve()
        return compute_model_loss_function(
            _model, observables, parameters_to_adjust, jreg_weight
        )

    grad = np.array([], dtype=np.float64)
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        # FD approximation -> only on the adjusted values. This is convient to
        # test to gradient on a small portion of big models with to many meshes to
        # be entirely tested.
        param_grad = finite_gradient(
            param.get_sliced_field(
                get_parameter_values_from_model(_model, param), is_preconditioned=True
            ),
            _local_fun,
            fm_args=(param,),
            eps=eps,
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
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    observables: Union[Observable, Sequence[Observable]],
    eps: Optional[float] = None,
) -> bool:
    """Check if the gradient computed with the adjoint state is equal with FD."""

    # Update parameters with model
    update_parameters_from_model(fwd_model, parameters_to_adjust)

    # Solve the forward problem
    solver: ForwardSolver = ForwardSolver(fwd_model)
    solver.solve()

    # Solve the adjoint problem
    asolver: AdjointSolver = AdjointSolver(fwd_model, adj_model)
    asolver.solve()

    adj_grad = compute_adjoint_gradient(
        fwd_model, asolver.adj_model, parameters_to_adjust
    )
    fd_grad = compute_fd_gradient(fwd_model, observables, parameters_to_adjust, eps=eps)

    return is_all_close(adj_grad, fd_grad)
