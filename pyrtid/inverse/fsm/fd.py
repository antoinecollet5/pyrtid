"""Provide gradient computation routines."""

import copy
import warnings
from typing import List, Optional

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.inverse.fsm.solver import FSMSolver
from pyrtid.inverse.loss_function import get_predictions_matching_observations
from pyrtid.inverse.obs import Observable, Observables
from pyrtid.inverse.params import (
    AdjustableParameter,
    AdjustableParameters,
    get_parameter_values_from_model,
    update_model_with_parameters_values,
    update_parameters_from_model,
)
from pyrtid.utils import (
    NDArrayFloat,
    finite_jacobian,
    is_all_close,
    object_or_object_sequence_to_list,
)


def _local_fun_pred(
    vector: NDArrayFloat,
    parameter: AdjustableParameter,
    _model: ForwardModel,
    observables: List[Observable],
    max_obs_time: Optional[float] = None,
) -> NDArrayFloat:
    # Update the model with the new values of x (preconditioned)
    # Do not save parameters values (useless)
    update_model_with_parameters_values(
        _model, vector, parameter, is_preconditioned=True, is_to_save=False
    )
    # Solve the forward model with the new parameters
    ForwardSolver(_model).solve()

    return get_predictions_matching_observations(_model, observables, max_obs_time)


def compute_fd_jacvec(
    model: ForwardModel,
    observables: Observables,
    parameters_to_adjust: AdjustableParameters,
    vecs: NDArrayFloat,
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
    model: ForwardModel
        The forward RT model.
    observables : Observables
        Sequence of observable instances.
    parameters_to_adjust : AdjustableParameters
        Sequence of adjusted parameter instances.
    vecs: NDArrayFloat
        Ensemble of vectors to multiply with the Jacobian matrix.
        It must have shape ($N_s \times N_e$), $N_s$ being the number of values
        optimized and $N_e$ the number of vectors.
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

    ljacvec: list[NDArrayFloat] = []
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

        ljacvec.append(
            finite_jacobian(
                param.preconditioner(param_values.ravel("F")),
                _local_fun_pred,
                fm_args=(
                    param,
                    _model,
                    observables,
                    max_obs_time,
                ),
                eps=eps,
                accuracy=accuracy,
                max_workers=max_workers,
            )
            @ vecs
        )

        # 2) Create an array full of nan and fill it with the
        # # gradient (only at adjusted locations)
        # Then save it.
        if is_save_state:
            param.jacvec_fd_history.append(ljacvec[-1])

    return np.hstack(ljacvec)


def is_fsm_jacvec_correct(
    fwd_model: ForwardModel,
    parameters_to_adjust: AdjustableParameters,
    observables: Observables,
    vecs: NDArrayFloat,
    eps: Optional[float] = None,
    accuracy: int = 0,
    max_workers: int = 1,
    hm_end_time: Optional[float] = None,
    is_verbose: bool = False,
    is_save_state: bool = True,
) -> bool:
    """
    Check if the gradient computed with the adjoint state is equal with FD.

    Parameters
    ----------
    fwd_model: ForwardModel
        The forward RT model.
    parameters_to_adjust : AdjustableParameters
        Sequence of adjusted parameter instances.
    observables : Observables
        Sequence of observable instances.
    vecs: NDArrayFloat
        Ensemble of vectors to multiply with the Jacobian matrix.
        It must have shape ($N_s \times N_e$), $N_s$ being the number of values
        optimized and $N_e$ the number of vectors.
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
        Whether to save the FD jac dot vectors in memory. The default is True.

    Returns
    -------
    bool
        True if the adjoint gradient is correct.
    """

    # Update parameters with model
    update_parameters_from_model(fwd_model, parameters_to_adjust)

    # Solve the forward problem
    solver: FSMSolver = FSMSolver(fwd_model)
    fsm_jacvec = solver.solve(
        observables, vecs, hm_end_time=hm_end_time, is_verbose=is_verbose
    )

    fd_jacvec = compute_fd_jacvec(
        fwd_model,
        observables,
        parameters_to_adjust,
        vecs,
        eps=eps,
        accuracy=accuracy,
        max_workers=max_workers,
        max_obs_time=hm_end_time,
        is_save_state=is_save_state,
    )

    return is_all_close(fsm_jacvec, fd_jacvec)
