"""
Provide classes and functions for geostatistical regularization.

TODO: add the formulas.
"""

from typing import Callable

import numpy as np

from pyrtid.inverse.regularization.base import Regularizator
from pyrtid.inverse.regularization.covariances import CovarianceMatrix
from pyrtid.inverse.regularization.drift import DriftMatrix, NullPriorTerm, PriorTerm
from pyrtid.utils.types import NDArrayFloat


def identify_function(x: NDArrayFloat) -> NDArrayFloat:
    """Return x untransformed (f(x) = x)."""
    return x


def one(x: NDArrayFloat) -> NDArrayFloat:
    """Return 1.0, whatever the input."""
    return np.ones(x.shape)


class GeostatisticalRegularizator(Regularizator):
    """
    Implement a regularization based on the parameter covariance matrix.

    Attributes
    ----------
    is_preconditioned: bool
        Whether the regularization is applied to a preconditioned parameter or not.
        Note that the value does not affect the behavior of the class (method results).
        It is stored here by convenience, to be used by the pyrtid inverse operator.
        The default is False.
    """

    def __init__(
        self,
        cov_m: CovarianceMatrix,
        prior: PriorTerm = NullPriorTerm(),
        transform: Callable = identify_function,
        transform_1st_derivative: Callable = one,
        is_preconditioned: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        cov_m : CovarianceMatrix
            _description_
        prior : Optional[PriorTerm], optional
            A prior term for `(x - prior_term)`, by default NullPriorTerm.
        transform: Callable, optional
            Parameter pre-transformation (variable change for the solver). The default
            is the identity function: f(x) = x.
        transform_1st_derivative: Callable, optional
            Parameter pre-transformation first order derivative.
            The default is 1.0 (the first derivative of the identity function).
        is_preconditioned: bool
            Whether the regularization is applied to a preconditioned parameter or not.
            Note that the value does not affect the behavior of the class
            (method results).
            It is stored here by convenience, to be used by the pyrtid inverse operator.
            The default is False.
        """
        self.cov_m = cov_m
        self.prior: PriorTerm = prior
        self.transform: Callable = transform
        self.transform_1st_derivative: Callable = transform_1st_derivative
        self.is_preconditioned = is_preconditioned

    def loss_function(self, values: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        .. math::

        \mathcal{R}_{Q}(u) = \frac{1}{2} \left(s-Xb\right)^TQ^{-1}\left(s-Xb\right)

        Parameters
        ----------
        values : NDArrayFloat
            Values of the parameter for which the regularization is computed.
            Should be 2D array / 1d vector.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        _values = self.transform(values[:, ::-1].ravel(order="F"))
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        return float(
            0.5
            * np.dot(
                residuals.T,
                self.cov_m.get_inv_cov_times_vector(residuals),
            )
        )

    def loss_function_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            Values of the parameter for which the regularization is computed (2d).

        Returns
        -------
        NDArrayFloat
            The regularization gradient (2d).
        """
        _values = self.transform(values[:, ::-1].ravel("F"))
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # right part $Q^{-1} * (m - m_{prior})$
        _right_part = self.cov_m.get_inv_cov_times_vector(residuals).ravel()
        # left part gradient -> special method to get more efficient
        # $ [I - dm_{prior}/dm]^{T} Q^{-1} (m - m_{prior})$
        return (
            (_right_part - self.prior.get_gradient_dot_product(_right_part))
            .reshape(values.shape[::-1])[::-1]
            .T
        ) * self.transform_1st_derivative(values)


def compute_best_beta(
    values: NDArrayFloat, cov_m: CovarianceMatrix, drift_matrix: DriftMatrix
) -> NDArrayFloat:
    """
    Compute the optimal beta (minimal objective function).

    TODO: Add the maths here.

    Parameters
    ----------
    values : NDArrayFloat
        Values of the parameter for which the regularization is computed.
        Should be 2D array / 1d vector.
    cov_m : CovarianceMatrix
        The covariance matrix.
    drift_matrix : DriftMatrix
        The drift matrix instance for which to compute beta.

    Returns
    -------
    NDArrayFloat
        The best beta.
    """
    # This is valid for the linear one only.
    invQs = cov_m.get_inv_cov_times_vector(values)
    invQX = cov_m.get_inv_cov_times_vector(drift_matrix.mat)

    XTinvQs = np.dot(drift_matrix.mat.T, invQs)
    XTinvQX = np.dot(drift_matrix.mat.T, invQX)

    # inexpensive solve p by p where p <= 3, usually p = 1 (scalar division)
    return np.linalg.solve(np.atleast_2d(XTinvQX), XTinvQs).ravel()
