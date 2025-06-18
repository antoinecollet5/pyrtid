"""
Implement a discrete regularizator.

@author: acollet
"""

from typing import List, Literal

import numpy as np

from pyrtid.inverse.regularization.base import Regularizator
from pyrtid.utils import NDArrayFloat
from pyrtid.utils.preconditioner import NoTransform, Preconditioner


def get_clostest_mode(x: NDArrayFloat, modes: NDArrayFloat) -> NDArrayFloat:
    return modes[np.digitize(x, modes[:-1] + np.diff(modes) / 2.0)]


def min_squared_distance(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    return (x - get_clostest_mode(x, np.asarray(modes))) ** 2


def dmin_squared_distance(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    return 2 * (x - get_clostest_mode(x, np.asarray(modes)))


def scaled_gaussian_pdf(x: NDArrayFloat, std: float, mean: float) -> NDArrayFloat:
    return np.exp(-((x - mean) ** 2) / (2 * std**2))


def dscaled_gaussian_pdf(x: NDArrayFloat, std: float, mean: float) -> NDArrayFloat:
    return -(x - mean) / (std**2) * scaled_gaussian_pdf(x, std, mean)


def gaussian_distance_from_modes(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    std = np.min(np.diff(sorted(modes))) / 6.0
    return 1.0 - np.sum([scaled_gaussian_pdf(x, std, mode) for mode in modes], axis=0)


def dgaussian_distance_from_modes(x: NDArrayFloat, modes: List[float]) -> NDArrayFloat:
    std = np.min(np.diff(sorted(modes))) / 6.0
    return -np.sum([dscaled_gaussian_pdf(x, std, mode) for mode in modes], axis=0)


class DiscreteRegularizator(Regularizator):
    r"""
    Apply a discrete (values takes specific discrete values) regularization.

    Attributes
    ----------
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.

    """

    def __init__(
        self,
        modes: List[float],
        penalty: Literal["least-squares", "gaussian"] = "least-squares",
        preconditioner: Preconditioner = NoTransform(),
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        modes : List[float]
            List of modes (discrete values that the field should take).
        penalty : Literal["least-squares", "gaussian"], optional
            Penalty used. If \"least-squares\", then the sum of squared distances to
            closest modes is considered. If \"gaussian\", then the sum of distances
            computed as a gaussian is used. By default "least-squares".
        preconditioner: Preconditioner
            Parameter pre-transformation operator (variable change for the solver).
            The default is the identity function: f(x) = x, which means no change
            is made.

        Raises
        ------
        ValueError
            If less than two modes are provided.
        """
        super().__init__(preconditioner)
        if len(modes) < 2:
            raise ValueError("At least two modes must be provided!")

        self.modes: List[float] = sorted(self.preconditioner(np.asarray(modes)))
        self.penalty = penalty

    @property
    def penalty(self) -> str:
        return self._penalty

    @penalty.setter
    def penalty(self, value: Literal["least-squares", "gaussian"]) -> None:
        if value not in ["least-squares", "gaussian"]:
            raise ValueError('penalty should be among ["least-squares", "gaussian"]')
        self._penalty = value

    def _eval_loss(self, param: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} \right)^{2}

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        float
        """
        if self.penalty == "least-squares":
            return float(np.sum(min_squared_distance(param, self.modes)))
        return float(np.sum(gaussian_distance_from_modes(param, self.modes)))

    def _eval_loss_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        if self.penalty == "least-squares":
            return dmin_squared_distance(param, self.modes)
        # Gaussian
        return dgaussian_distance_from_modes(param, self.modes)  # Gaussian
