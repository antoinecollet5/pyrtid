"""
Implement a Tikhonov regularizator.

TODO: add references.

@author: acollet
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyrtid.forward.models import Geometry
from pyrtid.inverse.preconditioner import NoTransform, Preconditioner
from pyrtid.inverse.regularization.base import (
    Regularizator,
    make_spatial_gradient_matrices,
)
from pyrtid.utils.operators import gradient_ffd, hessian_cfd
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


@dataclass
class TikhonovRegularizator(Regularizator):
    r"""
    Apply an Tikhonov (smoothing) regularization.

    Attributes
    ----------
    dx: float
        Mesh size in m for the x direction (axis 0).
    dy: float
        Mesh size in m for the y direction (axis 1).
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    dx: float
    dy: float
    preconditioner: Preconditioner = NoTransform()

    def _eval_loss(self, param: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        The Tikhonov regularization is defined as:

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} + \dfrac{u_{i, j+1} - u_{i,j}}{dy}
        \right)^{2}

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        f = 0.0
        if param.shape[0] > 2:
            f += 0.5 * float(np.sum(gradient_ffd(param, self.dx, axis=0) ** 2.0))
        if param.shape[1] > 2:
            f += 0.5 * float(np.sum(gradient_ffd(param, self.dy, axis=1) ** 2.0))
        return f

    def eval_loss_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
        r"""
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
        grad = np.zeros_like(param)
        if param.shape[0] > 2:
            grad += -hessian_cfd(param, self.dx, 0)
        if param.shape[1] > 2:
            grad += -hessian_cfd(param, self.dy, 1)
        return grad


@dataclass
class TikhonovMatRegularizator(Regularizator):
    r"""
    Apply an Tikhonov (smoothing) regularization using matrix formulation.

    Attributes
    ----------
    geometry : Geometry
        Geometry of the field
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    geometry: Geometry
    sub_selection: Optional[NDArrayInt] = None
    preconditioner: Preconditioner = NoTransform()

    def __post_init__(self) -> None:
        """Post initialize the object."""
        self.mat_grad_x, self.mat_grad_y = make_spatial_gradient_matrices(
            self.geometry, self.sub_selection
        )

    def _eval_loss(self, param: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        The Tikhonov regularization is defined as:

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} + \dfrac{u_{i, j+1} - u_{i,j}}{dy}
        \right)^{2}

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        f = 0.0
        f += 0.5 * float(np.sum((self.mat_grad_x @ param.ravel("F")) ** 2.0))
        # This is the same as -> simpler for derivation
        # f += 0.5 * float(
        #     np.sum(
        #         (
        #             param.ravel("F").T
        #             @ self.mat_grad_x.T
        #             @ self.mat_grad_x
        #             @ param.ravel("F")
        #         )
        #     )
        # )
        f += 0.5 * float(np.sum((self.mat_grad_y @ param.ravel("F")) ** 2.0))
        return f

    def eval_loss_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
        r"""
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
        grad = np.zeros(param.size)
        grad += self.mat_grad_x.T @ self.mat_grad_x @ param.ravel("F")
        grad += self.mat_grad_y.T @ self.mat_grad_y @ param.ravel("F")
        return grad.reshape(self.geometry.shape, order="F")
