"""
Created on Mon Feb 14 17:40:16 2022

@author: acollet
"""
from abc import ABC, abstractmethod

from pyrtid.utils import StrEnum
from pyrtid.utils.finite_differences import finite_gradient
from pyrtid.utils.types import NDArrayFloat


class RegWeightUpdateStrategy(StrEnum):
    """
    Strategy to update the regularization weight while optimizing.

    Available strategies are:
        - auto-per-round: Automatic estimation at the beginning of each optimization
            round. This means that the objective function definition is modified at the
            beginning of each round.
        - auto-continuous: Only valid with PyRTID's LBFGSB implementation. This is an
            experimental feature. The weight will be updated before each BFGS matrices
            build, i.e. at each solver internal iteration.

    """

    AUTO_CONTINUOUS = "auto-continuous"
    AUTO_PER_ROUND = "auto-per-round"


class Regularizator(ABC):
    """
    Represent a regularizator.

    This is an abstract class.
    """

    __slots__ = ["is_preconditioned"]

    def __init__(self, is_preconditioned: bool = False) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        is_preconditioned : bool, optional
            Whether the regularization is applied to the preconditioned values
            or not, by default False.
        """
        self.is_preconditioned: bool = is_preconditioned

    @abstractmethod
    def loss_function(self, param: NDArrayFloat) -> float:
        """
        Compute the regularization loss function.

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """

    @abstractmethod
    def loss_function_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
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

    def loss_function_gradient(
        self,
        param: NDArrayFloat,
        is_finite_differences: bool = False,
        max_workers: int = 1,
    ) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function.

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.
        is_finite_differences: bool
            If true, a numerical approximation by 2nd order finite difference is
            returned. Cost twice the `param` dimensions in terms of loss function
            calls. The default is False.
        max_workers: int
            Number of workers used  if the gradient is approximated by finite
            differences. If different from one, the calculation relies on
            multi-processing to decrease the computation time. The default is 1.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        if is_finite_differences:
            return finite_gradient(param, self.loss_function, max_workers=max_workers)
        else:
            return self.loss_function_gradient_analytical(param)
