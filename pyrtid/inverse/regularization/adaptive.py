import copy
import logging
from abc import ABC
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy as sp
from lbfgsb.base import get_bounds

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
    is_use_first_adjusted_weight_as_upper_bound: bool
        Whether the first weight determined by the method should becoe the
        upper bound. It prevent weight explosion.
    """

    __slots__ = [
        "reg_weight_bounds",
        "convergence_factor",
        "_has_noise_level_been_reached",
        "_has_been_above_noise_level",
        "is_use_first_adjusted_weight_as_upper_bound",
        "n_regw_update",
        "max_log_cr",
    ]

    def __init__(
        self,
        reg_weight_init: float = 1.0,
        reg_weight_bounds: Union[Tuple[float, float], NDArrayFloat] = (1e-10, 1e10),
        convergence_factor: float = 0.05,
        max_log_cr: float = 2.0,
        is_use_first_adjusted_weight_as_upper_bound: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        reg_weight_init : float, optional
            Initial regularization weight, by default 1.0
        reg_weight_bounds : Union[Tuple[float, float], NDArrayFloat], optional
            Bounds for the optimization parameter, by default (1e-10, 1e10)
        convergence_factor : float, optional
            Convergence criteria for the regularization weight, by default 0.05
        max_log_cr: float
            Maximum log change rate. It avoids a sudden rise/drop.
            The log is used because it is symmetrical: log(0.5) = - log(2).
            The default is 2.
        is_use_first_adjusted_weight_as_upper_bound: bool
            Whether the first weight determined by the method should become the
            upper bound. It prevents weight explosion. The default is True.
        """
        self.reg_weight_bounds = np.array(
            [get_bounds(np.array([reg_weight_init]), np.array([reg_weight_bounds]))]
        ).ravel()
        if reg_weight_init < 0.0:
            raise ValueError(
                "The initial regularization weight should be positive or null!"
            )
        if (self.reg_weight_bounds < 0).any():
            raise ValueError(
                "The bounds for the regularization weight should be positive or null!"
            )

        super().__init__(reg_weight_init)
        # internal state used only in the case of adaptive regularization strategy
        self.n_regw_update: int = 0
        self._has_been_above_noise_level = False
        self._has_noise_level_been_reached = False
        self.convergence_factor: float = convergence_factor
        self.is_use_first_adjusted_weight_as_upper_bound: bool = (
            is_use_first_adjusted_weight_as_upper_bound
        )
        self.max_log_cr: float = max_log_cr

        if max_log_cr <= 0:
            raise ValueError(f"The 'max_log_cr' ({max_log_cr:.2e}) must be positive! ")

        if max_log_cr <= convergence_factor:
            raise ValueError(
                f"The 'max_log_cr' ({max_log_cr:.2e})"
                " cannot be lower or equal to the 'convergence_factor' "
                f"({convergence_factor:.2e})"
            )

    @classmethod
    def is_adaptive(cls) -> bool:
        """Return whether the method is adaptive."""
        return True

    @property
    def reg_weight(self) -> float:
        return self._reg_weight

    @reg_weight.setter
    def reg_weight(self, value) -> None:
        self._reg_weight = min(
            max(self.reg_weight_bounds[0], value), self.reg_weight_bounds[1]
        )

    @staticmethod
    def get_log_cr(_old: float, _new: float) -> float:
        """Note: _odl and _new are always positive."""
        if _old == 0 or _new == 0:
            return 0.0
        return np.log(_new / _old)

    def _ensure_max_log_rc(
        self,
        new_rw: float,
        old_rw: float,
        gls_log_cr: Optional[float],
        logger: Optional[logging.Logger] = None,
    ) -> float:
        """
        Ensure a maximum log change rate between the new and the old weights.

        Parameters
        ----------
        new_rw: float
            New regularization weight.
        old_rw : float
            Old regularization weight.
        gls_rel_change: float
            Relative change of the LS gradient norm.
            If the relative change is high, then the regularization weight cannot be
            decreased or increased too much.
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.
        """
        # maximum relative change
        if gls_log_cr is not None:
            max_alrc = np.nanmin([self.max_log_cr, 1.0 / np.abs(gls_log_cr)])
        else:
            max_alrc = self.max_log_cr

        # current relative change
        cur_rc = self.get_log_cr(old_rw, new_rw)

        if logger is not None:
            logger.info(
                f"max_rc = {max_alrc:.2e}, cur_rc = {cur_rc:.2e}, rel = {gls_log_cr}"
            )

        # Case one: no changes
        if cur_rc == 0:
            return new_rw

        # Case 2: increase of the regularization weight
        if cur_rc > 0 and self.n_regw_update >= 1:
            if cur_rc > max_alrc:
                new_rw = old_rw * np.exp(max_alrc)
                if logger is not None:
                    logger.info(
                        f"Max log-relative change reached (increase = {max_alrc:.2e})!"
                    )

        # if diminished regularization weight
        elif cur_rc < 0 and self.n_regw_update >= 1:
            if -cur_rc > max_alrc:
                new_rw = old_rw / np.exp(max_alrc)
                if logger is not None:
                    logger.info(
                        f"Max log-relative change reached (decrease = {max_alrc:.2e})!"
                    )
        return new_rw

    def update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_reg_history: List[float],
        reg_weight_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
        logger: Optional[logging.Logger] = None,
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
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.

        Returns
        -------
        bool
            Whether the regularization parameter (weight) has changed.
        """
        has_rw_changed = super().update_reg_weight(
            loss_ls_history,
            loss_reg_history,
            reg_weight_history,
            loss_ls_grad,
            loss_reg_grad,
            n_obs,
            logger,
        )

        # Update comptor
        if has_rw_changed:
            self.n_regw_update += 1

        if (
            self.is_use_first_adjusted_weight_as_upper_bound
            and has_rw_changed
            and self.n_regw_update == 1
        ):
            self.reg_weight_bounds[1] = self.reg_weight
            if logger is not None:
                logger.info(f"Updating reg weight bounds to {self.reg_weight_bounds}.")
        return has_rw_changed


class AdaptiveGradientNormRegweight(AdaptiveRegweight):
    """Implement an adaptive regularization parameter choice based on the U-Curve."""

    def __init__(
        self,
        norm: Optional[Union[int, float, str]] = None,
        reg_weight_init: float = 1.0,
        reg_weight_bounds: Union[Tuple[float, float], NDArrayFloat] = (1e-10, 1e10),
        convergence_factor: float = 0.05,
        max_log_cr: float = 2,
        is_use_first_adjusted_weight_as_upper_bound: bool = True,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        norm: {int, inf, -inf, 'fro', 'nuc', None}, optional
            Order of the norm, inf means NumPy’s inf object. The default is None which
            means the l2 (Frobenius norm) is used.
        reg_weight_init : float, optional
            Initial regularization weight, by default 1.0
        reg_weight_bounds : Union[Tuple[float, float], NDArrayFloat], optional
            Bounds for the optimization parameter, by default (1e-10, 1e10)
        convergence_factor : float, optional
            Convergence criteria for the regularization weight, by default 0.05
        max_log_cr: float
            Maximum log change rate. It avoids a sudden rise/drop.
            The log is used because it is symmetrical: log(0.5) = - log(2).
            The default is 2.
        is_use_first_adjusted_weight_as_upper_bound: bool
            Whether the first weight determined by the method should become the
            upper bound. It prevents weight explosion. The default is True.

        Notes
        -----
        For values of ``ord <= 0``, the result is, strictly speaking, not a
        mathematical 'norm', but it may still be useful for various numerical
        purposes.

        The following norms can be calculated:

        =====  ============================  ==========================
        norm   norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                --
        'nuc'  nuclear norm                  --
        inf    max(sum(abs(a), axis=1))      max(abs(a))
        -inf   min(sum(abs(a), axis=1))      min(abs(a))
        0      --                            sum(a != 0)
        1      max(sum(abs(a), axis=0))      as below
        -1     min(sum(abs(a), axis=0))      as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  --                            sum(abs(a)**ord)**(1./ord)
        =====  ============================  ==========================

        The Frobenius norm is given by [1]_:

            :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

        The nuclear norm is the sum of the singular values.

        References
        ----------
        .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
            Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

        """
        super().__init__(
            reg_weight_init,
            reg_weight_bounds,
            convergence_factor,
            max_log_cr=max_log_cr,
            is_use_first_adjusted_weight_as_upper_bound=(
                is_use_first_adjusted_weight_as_upper_bound
            ),
        )
        self.is_noise_dominated: bool = False
        self.loss_ls_grad_norms: List[float] = []
        self.loss_reg_grad_norms: List[float] = []

        try:
            sp.linalg.norm(np.ones(10), ord=norm)
        except ValueError:
            raise ValueError(
                'Not a valid norm! Check the "ord" parameter from'
                " https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html"
            )
        self.norm = norm

    def _update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_reg_history: List[float],
        reg_weight_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
        logger: Optional[logging.Logger] = None,
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
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.

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

        self.loss_ls_grad_norms.append(sp.linalg.norm(loss_ls_grad, ord=self.norm))
        self.loss_reg_grad_norms.append(sp.linalg.norm(loss_reg_grad, ord=self.norm))

        # Determine the initial status
        if self.n_regw_update == 0:
            self.is_noise_dominated = (
                self.loss_ls_grad_norms[-1] < self.loss_reg_grad_norms[-1]
            )

        # means the regularization gradient is 0.0 -> no weight update
        if self.loss_reg_grad_norms[-1] == 0.0:
            return False
        if self.loss_ls_grad_norms[-1] == 0.0:
            return False

        if self.is_noise_dominated:
            # We increase the regularization weight to make sure that the
            # noise is removed and prevent overfit
            self.reg_weight = (
                (self.loss_reg_grad_norms[-1]) / self.loss_ls_grad_norms[-1]
            )
        else:
            # We  increase the regularization weight
            self.reg_weight = (
                self.loss_ls_grad_norms[-1] / (self.loss_reg_grad_norms[-1])
            )

        if logger is not None:
            logger.info(
                f"loss_ls_grad_{self.norm}_norm = {self.loss_ls_grad_norms[-1]}"
            )
            logger.info(
                f"loss_reg_grad_{self.norm}_norm = {self.loss_reg_grad_norms[-1]}"
            )

        # default value
        ls_log_cr: Optional[float] = None

        # Test the relative change of loss ls, if below the threshold, then
        # do not take the update into account
        if len(loss_ls_history) >= 2:
            if (
                np.abs(
                    (loss_ls_history[-1] - loss_ls_history[-2]) / loss_ls_history[-2]
                )
                < self.convergence_factor
            ):
                self.reg_weight = _old
                return False

            if len(self.loss_ls_grad_norms) >= 2:
                ls_log_cr = self.get_log_cr(
                    self.loss_ls_grad_norms[-2], self.loss_ls_grad_norms[-1]
                )

        # Test the relative change of alpha, if below the threshold, then
        # do not take the update into account
        if np.abs((self.reg_weight - _old) / _old) < self.convergence_factor:
            self.reg_weight = _old
            return False

        self.reg_weight = self._ensure_max_log_rc(
            self.reg_weight, _old, ls_log_cr, logger=logger
        )

        # return if it has changed ?
        return _old != self.reg_weight


class AdaptiveUCRegweight(AdaptiveRegweight):
    """Implement an adaptive regularization parameter choice based on the U-Curve."""

    def __init__(
        self,
        reg_weight_init: float = 1.0,
        reg_weight_bounds: Union[Tuple[float, float], NDArrayFloat] = (1e-10, 1e10),
        convergence_factor: float = 0.05,
        n_update_explo_phase: int = 5,
        max_log_cr: float = 1e10,
        is_use_first_adjusted_weight_as_upper_bound: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        reg_weight_init : float, optional
            Initial regularization weight, by default 1.0
        reg_weight_bounds : Union[Tuple[float, float], NDArrayFloat], optional
            Bounds for the optimization parameter, by default (1e-10, 1e10)
        convergence_factor : float, optional
            Convergence criteria for the regularization weight, by default 0.05
        n_update_explo_phase: int
            Number of regularization weight to perform before entering the optimization
            phase.The default is 5.

        is_use_first_adjusted_weight_as_upper_bound: bool
            Whether the first weight determined by the method should become the
            upper bound. It prevents weight explosion. The default is True.
        """
        super().__init__(
            reg_weight_init,
            reg_weight_bounds,
            convergence_factor,
            max_log_cr=max_log_cr,
            is_use_first_adjusted_weight_as_upper_bound=(
                is_use_first_adjusted_weight_as_upper_bound
            ),
        )
        # internal state used only in the case of adaptive regularization strategy
        self.n_update_explo_phase = n_update_explo_phase
        self.is_noise_dominated: bool = False
        self.loss_ls_grad_norms: List[float] = []
        self.loss_reg_grad_norms: List[float] = []

    def _update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_reg_history: List[float],
        reg_weight_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
        logger: Optional[logging.Logger] = None,
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
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.

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

        # Case 1: Exploration phase
        if self.n_regw_update < self.n_update_explo_phase:
            self.loss_ls_grad_norms.append(sp.linalg.norm(loss_ls_grad, ord=np.inf))
            self.loss_reg_grad_norms.append(sp.linalg.norm(loss_reg_grad, ord=np.inf))

            # Determine the initial status
            if self.n_regw_update == 0:
                self.is_noise_dominated = (
                    self.loss_ls_grad_norms[-1] < self.loss_reg_grad_norms[-1]
                )

            # means the regularization gradient is 0.0 -> no weight update
            if self.loss_reg_grad_norms[-1] == 0.0:
                return False
            if self.loss_ls_grad_norms[-1] == 0.0:
                return False

            if self.is_noise_dominated:
                # We increase the regularization weight to make sure that the
                # noise is removed and prevent overfit
                self.reg_weight = (
                    (self.loss_reg_grad_norms[-1]) / self.loss_ls_grad_norms[-1]
                )
            else:
                # We  increase the regularization weight
                self.reg_weight = (
                    self.loss_ls_grad_norms[-1] / (self.loss_reg_grad_norms[-1])
                )

            if logger is not None:
                logger.info("Exploration phase.")

            if logger is not None:
                logger.info(
                    f"loss_ls_grad_manhattan_norm = {self.loss_ls_grad_norms[-1]}"
                )
                logger.info(
                    f"loss_reg_grad_manhattan_norm = {self.loss_reg_grad_norms[-1]}"
                )

                logger.info(
                    f"loss_ls_grad_squared_norm = {sp.linalg.norm(loss_ls_grad, ord=2)}"
                )
                logger.info(
                    "loss_reg_grad_squared_norm = "
                    f"{sp.linalg.norm(loss_reg_grad, ord=2)}"
                )

                logger.info(
                    "loss_ls_grad_inf_norm = "
                    f"{sp.linalg.norm(loss_ls_grad, ord=np.inf)}"
                )
                logger.info(
                    "loss_reg_grad_inf_norm = "
                    f"{sp.linalg.norm(loss_reg_grad, ord=np.inf)}"
                )

        # Case 2: we compute the optimal regularization weight - Optimization phase
        else:
            # we must have the same number of loss_ls and loss_reg
            assert (
                len(loss_ls_history) == len(loss_reg_history) == len(reg_weight_history)
            )
            self.reg_weight = get_optimal_reg_param(
                reg_weight_history,
                compute_uc(loss_ls_history, loss_reg_history),
            )
            if logger is not None:
                logger.info("Optimization phase.")

        # TODO: test this
        # Limit the change in the weight to an order of magnitude
        # if _old != 0.0:
        #     if np.abs(self.reg_weight / _old) > 10:
        #         self.reg_weight = _old * 10
        #     if np.abs(self.reg_weight / _old) < 0.1:
        #         self.reg_weight = _old / 10.0

        # Test the relative change of alpha, if below the threshold, then
        # do not take the update into account
        if np.abs((self.reg_weight - _old) / _old) < self.convergence_factor:
            self.reg_weight = _old
            return False

        # TODO: replace the None by the weight
        self.reg_weight = self._ensure_max_log_rc(
            self.reg_weight, _old, None, logger=logger
        )

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
