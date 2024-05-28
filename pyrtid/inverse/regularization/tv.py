"""
Implement a Total Variation regularizator.

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
class TVRegularizator(Regularizator):
    r"""
    Apply a Total Variation (sharpening) regularization.

    Attributes
    ----------
    dx: float
        Mesh size in m for the x direction (axis 0).
    dy: float
        Mesh size in m for the y direction (axis 1).
    eps: float
        Small factor added in the square root to deal with the singularity at
        $\nabla u = 0$ when computing the gradient. The default is 1e-20.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.

    """

    dx: float
    dy: float
    eps: float = 1e-20
    preconditioner: Preconditioner = NoTransform()

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
        NDArrayFloat
        """
        f = 0.0
        if param.shape[0] > 2:
            f += np.sum(np.square(gradient_ffd(param, self.dx, axis=0)))
        if param.shape[1] > 2:
            f += np.sum(np.square(gradient_ffd(param, self.dy, axis=1)))
        return np.sqrt(f + self.eps)

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
        grad = np.zeros_like(param)
        if param.shape[0] > 2:
            grad += -hessian_cfd(param, self.dx, 0)
        if param.shape[1] > 2:
            grad += -hessian_cfd(param, self.dy, 1)

        # denominator = 1 / L1 because of the square root
        return grad / self._eval_loss(param)
