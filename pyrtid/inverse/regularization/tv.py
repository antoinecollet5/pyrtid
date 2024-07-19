"""
Implement a Total Variation regularizator.

TODO: add references.

@author: acollet
"""

from dataclasses import dataclass

import numpy as np

from pyrtid.inverse.preconditioner import NoTransform, Preconditioner
from pyrtid.inverse.regularization.base import Regularizator
from pyrtid.utils.operators import gradient_ffd
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

    def _get_grid_cell_l1(self, param: NDArrayFloat) -> NDArrayFloat:
        # sum of squared spatial gradient
        sg2 = np.zeros_like(param)
        if param.shape[0] > 2:
            sg2 += np.square(gradient_ffd(param, self.dx, axis=0))
        if param.shape[1] > 2:
            sg2 += np.square(gradient_ffd(param, self.dy, axis=1))
        # Add epsilon to prevent undetermination when deriving
        return np.sqrt(sg2 + self.eps)

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
        return float(np.sum(self._get_grid_cell_l1(param)))

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
        den = self._get_grid_cell_l1(param)

        if param.shape[0] > 2:
            grad -= gradient_ffd(param, self.dx, axis=0) / self.dx / den
            grad[1:, :] += (
                gradient_ffd(param, self.dx, axis=0)[:-1, :] / self.dx / den[:-1, :]
            )

        if param.shape[1] > 2:
            grad -= gradient_ffd(param, self.dy, axis=1) / self.dy / den
            grad[:, 1:] += (
                gradient_ffd(param, self.dy, axis=1)[:, :-1] / self.dy / den[:, :-1]
            )
        return grad
