"""
Created on Mon Feb 14 17:40:16 2022

@author: acollet
"""
from abc import ABC, abstractmethod

from pyrtid.utils.finite_differences import finite_gradient
from pyrtid.utils.types import NDArrayFloat


class Regularizator(ABC):
    """
    Represent a regularizator.

    This is an abstract class.
    """

    is_preconditioned: bool = False

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
        self, param: NDArrayFloat, is_finite_differences: bool = False
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

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        if is_finite_differences:
            return finite_gradient(param, self.loss_function)
        else:
            return self.loss_function_gradient_analytical(param)
