"""
@author: acollet
"""

from abc import ABC, abstractmethod
from typing import List

from pyrtid.inverse.preconditioner import NoTransform, Preconditioner
from pyrtid.utils.finite_differences import finite_gradient
from pyrtid.utils.types import NDArrayFloat


class RegWeightUpdateStrategy(ABC):
    """
    Strategy to update the regularization parameter while optimizing.
    """

    __slots__ = ["_reg_weight"]

    def __init__(self, reg_weight: float = 1.0) -> None:
        self.reg_weight = reg_weight

    @property
    def reg_weight(self) -> float:
        return self._reg_weight

    @reg_weight.setter
    def reg_weight(self, value: float) -> None:
        self._reg_weight = value

    def update_reg_weight(
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
        return self._update_reg_weight(
            loss_ls_history,
            loss_reg_history,
            reg_weight_history,
            loss_ls_grad,
            loss_reg_grad,
            n_obs,
        )

    def _update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_reg_history: List[float],
        reg_weight_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
    ) -> bool:
        # by default no update
        return False


class ConstantRegWeight(RegWeightUpdateStrategy):
    """
    Implement a constant regularization parameter.

    Attributes
    ----------
    reg_weight: float
        Current regularization weight (parameter).
    """

    def __init__(self, reg_weight: float = 1.0) -> None:
        super().__init__(reg_weight)


class Regularizator(ABC):
    """
    Represent a regularizator.

    This is an abstract class.
    """

    __slots__ = ["preconditioner"]

    def __init__(self, preconditioner: Preconditioner = NoTransform()) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        preconditioner : Preconditioner
            Whether the regularization is applied to the preconditioned values
            or not, by default False.
        """
        self.preconditioner: Preconditioner = preconditioner

    def eval_loss(self, param: NDArrayFloat) -> float:
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
        return self._eval_loss(self.preconditioner(param))

    @abstractmethod
    def _eval_loss(self, param: NDArrayFloat) -> float: ...  # pragma: no cover

    @abstractmethod
    def eval_loss_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
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

    def eval_loss_gradient(
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
            The regularization gradient (not preconditioned).
        """
        if is_finite_differences:
            return finite_gradient(param, self.eval_loss, max_workers=max_workers)
        else:
            return self.preconditioner.dtransform_vec(
                param,
                self.eval_loss_gradient_analytical(self.preconditioner(param)),
            )
