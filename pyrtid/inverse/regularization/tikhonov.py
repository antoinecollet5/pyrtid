"""
Implement a Tikhonov regularizator.

TODO: add references.

@author: acollet
"""

from dataclasses import dataclass

import numpy as np

from pyrtid.inverse.preconditioner import NoTransform, Preconditioner
from pyrtid.inverse.regularization.base import Regularizator
from pyrtid.utils.operators import gradient_ffd, hessian_cfd
from pyrtid.utils.types import NDArrayFloat


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
