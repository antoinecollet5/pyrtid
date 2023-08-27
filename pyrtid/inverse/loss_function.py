from typing import Optional, Sequence, Union

import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.inverse.params import update_parameters_from_model
from pyrtid.utils.types import NDArrayFloat, object_or_object_sequence_to_list

from .obs import (
    Observable,
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from .params import AdjustableParameter


def ls_loss_function(
    x_obs: NDArrayFloat, x_calc: NDArrayFloat, x_sigma: NDArrayFloat
) -> float:
    """
    Return the objective function with regard to `x`.

    Parameters
    ----------
    x_obs: NDArrayFloat
        1D vector of observaed values.
    x_calc: NDArrayFloat
        1D vector of calculated values.
    x_obs: NDArrayFloat
        1D vector of uncertainties on observed values.

    Returns
    -------
    objective : float
        the value of the objective function

    """
    return 0.5 / np.size(x_obs) * np.sum(np.square((x_obs - x_calc) / x_sigma))


def get_model_ls_loss_function(
    model: ForwardModel,
    observables: Union[Observable, Sequence[Observable]],
    max_obs_time: Optional[float] = None,
) -> float:
    """
    Return the least-square loss function of the model for the given observations.

    Parameters
    ----------
    model : ForwardModel
        The forward model from which to read the simulated values.
    observables : Union[Observable, Sequence[Observable]]
        Sequence of observable instances.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an obervation value, by default None

    Returns
    -------
    float
        The objective function.
    """
    if max_obs_time is not None:
        max_obs_time = min(np.max(model.time_params.ldt), max_obs_time)
    else:
        max_obs_time = np.max(model.time_params.ldt)

    return ls_loss_function(
        get_observables_values_as_1d_vector(observables, max_obs_time),
        get_predictions_matching_observations(model, observables, max_obs_time),
        get_observables_uncertainties_as_1d_vector(observables, max_obs_time),
    )


def get_model_reg_loss_function(
    model: ForwardModel,
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
) -> float:
    # Update the parameter values from the model.
    update_parameters_from_model(model, parameters_to_adjust)
    return float(
        sum(
            [
                p.get_regularization_loss_function()
                for p in object_or_object_sequence_to_list(parameters_to_adjust)
            ]
        )
    )


def get_model_loss_function(
    model: ForwardModel,
    observables: Union[Observable, Sequence[Observable]],
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    max_obs_time: Optional[float] = None,
    jreg_weight: float = 1.0,
) -> float:
    """_summary_

    Parameters
    ----------
    fwd_model : ForwardModel
        Forward model.
    parameters_to_adjust : Union[AdjustableParameter, Sequence[AdjustableParameter]]
        Adjusted parameters.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an obervation value, by default None
    jreg_weight : float, optional
        Weight to apply to the regularization part, by default 1.0.

    Returns
    -------
    float
        Total objective function (least-squares) for the forward model.
    """
    return get_model_ls_loss_function(
        model, observables, max_obs_time
    ) + jreg_weight * get_model_reg_loss_function(model, parameters_to_adjust)
