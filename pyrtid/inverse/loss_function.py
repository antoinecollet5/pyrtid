from typing import Sequence, Union

import numpy as np

from pyrtid.forward import ForwardModel
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
    return 0.5 * np.sum(np.square((x_calc - x_obs) / x_sigma))


def get_model_ls_loss_function(
    model: ForwardModel, observables: Union[Observable, Sequence[Observable]]
) -> float:
    """Return the least-square loss function of the model for the given observations."""

    return ls_loss_function(
        get_predictions_matching_observations(model, observables),
        get_observables_values_as_1d_vector(observables),
        get_observables_uncertainties_as_1d_vector(observables),
    )


def get_model_reg_loss_function(
    model: ForwardModel,
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
) -> float:
    return float(
        sum(
            [
                p.get_regularization_loss_function()
                for p in object_or_object_sequence_to_list(parameters_to_adjust)
            ]
        )
    )


def compute_model_loss_function(
    model: ForwardModel,
    observables: Union[Observable, Sequence[Observable]],
    parameters_to_adjust: Union[AdjustableParameter, Sequence[AdjustableParameter]],
    jreg_weight: float = 1.0,
) -> float:
    """_summary_

    Parameters
    ----------
    fwd_model : ForwardModel
        _description_
    parameters_to_adjust : Union[AdjustableParameter, Sequence[AdjustableParameter]]
        _description_
    jreg_weight : float, optional
        _description_, by default 1.0

    Returns
    -------
    NDArrayFloat
        _description_
    """
    return get_model_ls_loss_function(
        model, observables
    ) + jreg_weight * get_model_reg_loss_function(model, parameters_to_adjust)
