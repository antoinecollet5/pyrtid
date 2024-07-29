import copy
from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import scipy as sp

from pyrtid.inverse.regularization.base import RegWeightUpdateStrategy
from pyrtid.utils import NDArrayFloat


class AdaptiveRegweight(RegWeightUpdateStrategy, ABC):
    """Abstract interface for adaptive regularization parameter choice.

    Attributes
    ----------
    reg_weight: float
        Current regularization weight (parameter).
    reg_weight_bounds: NDArrayFloat
        Bounds for the reg_weight used in adaptive strategies.
    """

    __slots__ = [
        "reg_weight_bounds",
        "convergence_factor",
        "_has_noise_level_been_reached",
    ]

    def __init__(
        self,
        reg_weight_init: float = 1.0,
        reg_weigh_bounds: Union[Tuple[float, float], NDArrayFloat] = (1e-10, 1e10),
        convergence_factor: float = 0.05,
    ) -> None:
        """

        Parameters
        ----------
        reg_weight_init : float, optional
            Initial regularization weight, by default 1.0
        reg_weigh_bounds : Union[Tuple[float, float], NDArrayFloat], optional
            Bounds for the optimization parameter, by default (1e-10, 1e10)
        convergence_factor : float, optional
            Convergence criteria for the regularization weight, by default 0.05
        """
        # TODO: add a check on the bounds and update the initial value accordingly
        # (must be positive)
        self.reg_weight_bounds = reg_weigh_bounds

        super().__init__(reg_weight_init)
        # internal state used only in the case of adaptive regularization strategy
        self._has_noise_level_been_reached = False
        self.convergence_factor: float = convergence_factor

    @property
    def reg_weight(self) -> float:
        return self._reg_weight

    @reg_weight.setter
    def reg_weight(self, value) -> None:
        self._reg_weight = min(
            max(self.reg_weight_bounds[0], value), self.reg_weight_bounds[1]
        )


class AdaptiveUCRegweight(AdaptiveRegweight):
    """Implement an adaptive regularization parameter choice based on the U-Curve."""

    def _update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_reg_history: List[float],
        reg_weight_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
    ) -> bool:
        """
        Update the regularization weight.

        Parameters
        ----------
        loss_ls_history : List[float]
            List of past LS cost function.
        loss_reg_history : List[float]
            List of past regularization cost function.
        reg_weight_history : List[float]
            List of past regularization parameter (weight).
        loss_ls_grad : NDArrayFloat
            Current LS cost function gradient.
        loss_reg_grad : NDArrayFloat
            Current Reg cost function gradient.
        n_obs : int
            Number of observations used in the LS cost function.

        Returns
        -------
        bool
            Whether the regularization parameter (weight) has changed.
        """

        # Case 1: no regularization loss has been saved
        if len(loss_reg_history) == 0:
            return False
        # Case 2: the regularization is null
        if loss_reg_history[-1] == 0:
            return False

        _old: float = copy.copy(self.reg_weight)
        if loss_ls_history[-1] < n_obs and len(loss_ls_history) != 1:
            self._has_noise_level_been_reached = True

        # Case 3: loss_ls if below the approximate noise level
        # noise level = n_obs because loss_ls is scaled by n_obs
        # Exploration phase
        if not self._has_noise_level_been_reached:
            loss_reg_grad_manhattan_norm = sp.linalg.norm(loss_reg_grad, ord=1)
            # means the regularization gradient is 0.0 -> no weight update
            if loss_reg_grad_manhattan_norm == 0.0:
                return False
            self.reg_weight = (
                0.5 * sp.linalg.norm(loss_ls_grad, ord=1) / loss_reg_grad_manhattan_norm
            )
        # Case 4: we compute the optimal regularization weight
        # Optimization phase
        else:
            # we must have the same number of loss_ls and loss_reg
            assert (
                len(loss_ls_history) == len(loss_reg_history) == len(reg_weight_history)
            )
            self.reg_weight = get_optimal_reg_param(
                reg_weight_history,
                compute_uc(loss_ls_history, loss_reg_history),
            )

        # Test the relative change of alpha, if below the threshold, then
        if np.abs((self.reg_weight - _old) / _old) < self.convergence_factor:
            self.reg_weight = _old
            return False

        # return if it has changed ?
        return _old != self.reg_weight


def select_valid_reg_params(
    reg_params: List[float], uc_values: List[float]
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Return valid reg_params and associated uc values sorted by increasing order.

    reg_params can be duplicated because the objective function can be evaluated several
    times in a row without updating the regularization parameter (reg_param), or because
    the same reg_param value can be used a second time (not very probable though).
    Since the convergence of the objective function is expected to improve
    continuously, only the last occurrence of a given reg_param is kept. Then the
    regularization parameters are sorted in ascending order. This new order is
    propagated to the uc values.

    Parameters
    ----------
    reg_params : List[float]
        List of successive regularization parameters used in the optimization
        process.
    uc_values : List[float]
        List of successives uc values obtained in the optimization process.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        Valid sorted reg_params and uc values.
    """
    # find index of last occurrence of each reg_params
    id_last_occur = [
        len(reg_params) - 1 - reg_params[::-1].index(val) for val in set(reg_params)
    ]
    # find increasing order
    sorted_id = np.argsort(np.array(reg_params)[id_last_occur])
    # sub sample and sort both reg_params and uc_values
    return (
        np.array(reg_params)[id_last_occur][sorted_id],
        np.array(uc_values)[id_last_occur][sorted_id],
    )


def make_convex_around_min_uc(
    reg_params: NDArrayFloat, uc_values: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Make the sequence of reg_params-uc convex around the minimum uc value.

    First, the minimum uc value is found. Then two values are kept on either side of the
    minimum with the condition that these values are greater than all other values
    from the minimum uc (the value at index -2 must be greater than the one at index -1
    and the value at index 2 must be greater than the value at index 1).

    Parameters
    ----------
    reg_params : NDArrayFloat
        Sequence of unique increasing regularization parameters.
    uc_values : NDArrayFloat
        Sequence of associated uc values.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        reg_params (by increasing order) and associated convex sequence of uc values.
    """
    # find the index of the minimum
    id_min_uc: int = np.argmin(uc_values)
    # add the index of the minimum as a value to keep
    kept_indices: List[int] = [id_min_uc]

    # going backward from the mininmum
    for i in range(min(id_min_uc, 2)):
        j = id_min_uc - i - 1
        if uc_values[j] > uc_values[j + 1]:
            kept_indices.insert(0, j)

    # going forkward from the mininmum
    for i in range(min(reg_params.size - id_min_uc - 1, 2)):
        j = i + id_min_uc + 1
        if uc_values[j] > uc_values[j - 1]:
            kept_indices.append(j)

    return reg_params[kept_indices], uc_values[kept_indices]


def get_minima_indices(input: NDArrayFloat) -> NDArrayFloat:
    """
    Return the indices of all local minima found.

    Parameters
    ----------
    input : NDArrayFloat
        Sequence of values.

    Returns
    -------
    NDArrayFloat
        Indices of all local minima found.
    """
    minima = np.ones_like(input, dtype=np.bool_)
    minima[:-1] = input[:-1] <= input[1:]
    minima[1:] = np.logical_and(minima[1:], input[:-1] >= input[1:])
    return np.argwhere(minima).ravel()


def interpolate_reg_param(
    reg_params: NDArrayFloat, uc_values: NDArrayFloat, current_reg_param: float
) -> float:
    """
    Interpolate the regularization parameter from a sequence of convex uc values.

    Parameters
    ----------
    reg_params : NDArrayFloat
        Regularization parameters (by increasing order).
    uc_values : NDArrayFloat
        Associated convex sequence of uc values.
    current_reg_param: float
        Current regularization parameter.

    Returns
    -------
    float
        Optimal reg_param determined from the sequence.
    """
    # the fourth order is more than enough.
    k = min(4, reg_params.size - 1)

    # interpolate on a log scale to ease the process
    _log_reg_params = np.log10(reg_params)
    _log_betas = np.log10(uc_values)

    # perform a spline interpolation of reg_param_reg vs. u-curve values
    spl = sp.interpolate.InterpolatedUnivariateSpline(_log_reg_params, _log_betas, k=k)
    # Add a bit of smoothing to avoid overfitting
    spl.set_smoothing_factor(1.0)

    # Unfortunately, the roots can only be found if the number of points for the
    # interpolation is greater than 4, which is not always the case.
    # So we prefer to use a brute force method -> interpolate the values on a
    # logarithmic scale with 200 points.  We should get a close enough reg_param.
    # The interval is defined by the min and max reg_params found in the sequence
    pts = np.logspace(
        np.log10(reg_params[0]), np.log10(reg_params[-1]), base=10, num=200
    )
    predictions = spl(np.log10(pts))

    # it is possible that the spline interpolation produces something not convex.
    # In that case, all minima are identified and the closest from the previous
    # regularization parameter (reg_param) is kept
    min_ids = get_minima_indices(predictions)

    # only one minimum = global minimum
    if len(min_ids) == 1:
        return pts[min_ids[0]]
    # otherwise take the closest to the current reg_param
    dist2reg_param = np.abs(pts[min_ids] - current_reg_param)
    return pts[min_ids][np.argmin(dist2reg_param)]


def get_optimal_reg_param(
    reg_params: Union[List[float], NDArrayFloat],
    uc_values: Union[List[float], NDArrayFloat],
) -> float:
    """
    Return the optimal interpolated regularization parameter

    Parameters
    ----------
    reg_params : Union[List[float], NDArrayFloat]
        List of successive regularization parameters used in the optimization
        process.
    uc_values : Union[List[float], NDArrayFloat]
        List of successives uc values obtained in the optimization process.

    Returns
    -------
    float
        Optimal regularization parameter (alpha_reg).
    """
    _reg_params, _uc = make_convex_around_min_uc(
        *select_valid_reg_params(list(reg_params), list(uc_values))
    )

    # this must be handled before calling get_optimal_reg_param
    if np.size(_reg_params) == 1:
        return _reg_params[0]

    # Treat the case when the minimum is on one side -> continue the exploration
    if np.min(_uc) == _uc[0]:
        return 10 ** (2.0 * np.log10(_reg_params[0]) - np.log10(_reg_params[1]))
    if np.min(_uc) == _uc[-1]:
        return 10 ** (2.0 * np.log10(_reg_params[-1]) - np.log10(_reg_params[-2]))

    # otherwise perform a spline interpolation to find the optimal reg_param
    return interpolate_reg_param(_reg_params, _uc, reg_params[-1])


def compute_uc(
    loss_ls_history: Union[List[float], NDArrayFloat],
    loss_reg_history: Union[List[float], NDArrayFloat],
) -> NDArrayFloat:
    """Return the U-curve values."""
    # add eps to avoid division by zero
    eps = 1e-20
    return 1 / (np.array(loss_ls_history) + eps) + 1 / (
        np.array(loss_reg_history) + eps
    )
