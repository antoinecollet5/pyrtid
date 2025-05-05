from typing import Optional, Sequence

import numpy as np

from pyrtid.forward import ForwardModel
from pyrtid.inverse.obs import (
    Observable,
    Observables,
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.inverse.params import AdjustableParameters, eval_weighted_loss_reg
from pyrtid.utils import NDArrayFloat


def eval_loss_ls(
    d_obs: NDArrayFloat, d_calc: NDArrayFloat, x_sigma: NDArrayFloat
) -> float:
    r"""
    Return the objective function with regard to `x`.

    .. math::
        \mathcal{J} = \dfrac{1}{2} \sum_{n=0}^{N}
        \left(\dfrac{d_{\mathrm{obs}}^{n}
        - d_{\mathrm{calc}}^{n}}{\sigma_{\mathrm{obs}}^{n}} \right)^{2}

    with $n$, a time with an observation, and $\lvert \bm{d}_{\mathrm{obs}} \rvert$
    the number of observation points.

    Parameters
    ----------
    d_obs: NDArrayFloat
        1D vector of observed values.
    d_calc: NDArrayFloat
        1D vector of calculated values.
    d_obs: NDArrayFloat
        1D vector of uncertainties on observed values.

    Returns
    -------
    objective : float
        the value of the objective function

    """
    return 0.5 * np.sum(np.square((d_calc - d_obs) / x_sigma))


def eval_model_loss_ls(
    model: ForwardModel,
    observables: Observables,
    max_obs_time: Optional[float] = None,
) -> float:
    """
    Return the least-square loss function of the model for the given observations.

    Parameters
    ----------
    model : ForwardModel
        The forward model from which to read the simulated values.
    observables : Observables
        Sequence of observable instances.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None

    Returns
    -------
    float
        The objective function.
    """
    if max_obs_time is not None:
        max_obs_time = min(model.time_params.time_elapsed, max_obs_time)
    else:
        max_obs_time = model.time_params.time_elapsed

    return eval_loss_ls(
        get_observables_values_as_1d_vector(observables, max_obs_time),
        get_predictions_matching_observations(model, observables, max_obs_time),
        get_observables_uncertainties_as_1d_vector(observables, max_obs_time),
    )


def eval_model_loss_function(
    model: ForwardModel,
    observables: Observables,
    parameters_to_adjust: AdjustableParameters,
    max_obs_time: Optional[float] = None,
) -> float:
    """_summary_

    Parameters
    ----------
    model : ForwardModel
        Forward model.
    parameters_to_adjust : AdjustableParameters
        Sequence of adjusted parameter instances.
    observables : Observables
        Sequence of observable instances.
    max_obs_time : Optional[float], optional
        Maximum time for which to consider an observation value, by default None

    Returns
    -------
    float
        Total objective function (least-squares) for the forward model.
    """
    return eval_model_loss_ls(
        model, observables, max_obs_time
    ) + eval_weighted_loss_reg(parameters_to_adjust, model)


def get_theoretical_noise_level(
    observables: Sequence[Observable], n_std: float = 5
) -> float:
    """
    Get the theoretical noise level in the solution.

    If the obersations uncertaintes are well mastered, it can be used as a threshold for
    the loss function ad defined by Gao et al. 2006.

    It is propotionnal to the number of observations. And assume that the observation
    noise is well known.
    """
    n_obs = np.sum([obs.values.size for obs in observables])
    return 0.5 * n_obs + n_std * np.sqrt(0.5 * n_obs)
