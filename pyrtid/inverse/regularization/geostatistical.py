"""
Provide classes and functions for geostatistical regularization.

TODO: add the formulas.
"""

import numpy as np

from pyrtid.inverse.preconditioner import NoTransform, Preconditioner
from pyrtid.inverse.regularization.base import Regularizator
from pyrtid.inverse.regularization.covariances import CovarianceMatrix
from pyrtid.inverse.regularization.priors import NullPriorTerm, PriorTerm
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
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    __slots__ = ["cov_m", "prior"]

    def __init__(
        self,
        cov_m: CovarianceMatrix,
        prior: PriorTerm = NullPriorTerm(),
        preconditioner: Preconditioner = NoTransform(),
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
        preconditioner: Preconditioner
            Parameter pre-transformation operator (variable change for the solver).
            The default is the identity function: f(x) = x, which means no change
            is made.
        """
        super().__init__(preconditioner)
        self.cov_m: CovarianceMatrix = cov_m
        self.prior: PriorTerm = prior

    def _eval_loss(self, values: NDArrayFloat) -> float:
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
        _values = values.ravel(order="F")
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        return float(
            0.5
            * np.dot(
                residuals.T,
                self.cov_m.solve(residuals),
            )
        )

    def eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
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
        _values = values.ravel(order="F")
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # right part $Q^{-1} * (m - m_{prior})$
        _right_part = self.cov_m.solve(residuals).ravel()
        # left part gradient -> special method to get more efficient
        # $ [I - dm_{prior}/dm]^{T} Q^{-1} (m - m_{prior})$
        return (_right_part - self.prior.get_gradient_dot_product(_right_part)).reshape(
            values.shape, order="F"
        )


class EnsembleRegularizator(GeostatisticalRegularizator):
    """
    Implement a regularization based on an ensemble.

    Compared to a classic regularization, an ensemble is passed to the functions.

    TODO: here add the objective function of the regularization terM.

    Attributes
    ----------
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    def _eval_loss(self, ens: NDArrayFloat) -> float:
        r"""
        TODO: update this.
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
        _values = ens
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # residuals = - self.prior.get_values(_values)

        # And this is strictly equivalent (element wise multiplication)
        return (
            0.5 * float(np.sum(residuals * self.cov_m.solve(residuals))) / ens.shape[1]  # type: ignore
        )

    def eval_loss_gradient_analytical(self, ens: NDArrayFloat) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        ens : NDArrayFloat
            Ensemble of shape (N_s, Ne). N_s being the number of optimized values,
            Ne, the number of members in the ensemble.

        Returns
        -------
        NDArrayFloat
            The regularization gradient (2d).
        """
        _values = ens
        residuals: NDArrayFloat = _values - self.prior.get_values(_values)
        # residuals = _values * 0.0 - self.prior.get_values(_values)

        # right part $Q^{-1} * (m - m_{prior})$
        _right_part = self.cov_m.solve(residuals)

        # We should have the same shape
        assert _right_part.shape == ens.shape

        # TODO: here we considered than the derivative of the covariance matrix w.r.t.
        # the parameters is null, but that is not necessary the case all the time.

        # left part gradient -> special method to get more efficient
        # $ [I - dm_{prior}/dm]^{T} Q^{-1} (m - m_{prior})$
        # Note: dot product must be distributed
        # The mean operator comes from the fact that a member j is involved
        # in all derivations of the mean operator.
        return (
            _right_part
            - np.mean(
                self.prior.get_gradient_dot_product(_right_part),
                axis=1,
                keepdims=True,
            )
        ) / ens.shape[1]


# def compute_best_beta(
#     values: NDArrayFloat, cov_m: CovarianceMatrix, drift_matrix: DriftMatrix
# ) -> NDArrayFloat:
#     """
#     Compute the optimal beta (minimal objective function).

#     TODO: Add the maths here.

#     Parameters
#     ----------
#     values : NDArrayFloat
#         Values of the parameter for which the regularization is computed.
#         Should be 2D array / 1d vector.
#     cov_m : CovarianceMatrix
#         The covariance matrix.
#     drift_matrix : DriftMatrix
#         The drift matrix instance for which to compute beta.

#     Returns
#     -------
#     NDArrayFloat
#         The best beta.
#     """
#     # This is valid for the linear one only.
#     invQs = cov_m.solve(values)
#     invQX = cov_m.solve(drift_matrix.mat)

#     XTinvQs = np.dot(drift_matrix.mat.T, invQs)
#     XTinvQX = np.dot(drift_matrix.mat.T, invQX)

#     # inexpensive solve p by p where p <= 3, usually p = 1 (scalar division)
#     return np.linalg.solve(np.atleast_2d(XTinvQX), XTinvQs).ravel()
