"""
Implement a Total Variation regularizator.

TODO: add references.

@author: acollet
"""

from dataclasses import dataclass

import numpy as np

from pyrtid.utils.operators import gradient_bfd, gradient_ffd
from pyrtid.utils.types import NDArrayFloat

from .base import Regularizator


@dataclass
class TVRegularizatorAnisotropic(Regularizator):
    r"""
    Apply an anisotropic Total Variation (sharpening) regularization.

    Attributes
    ----------
    dx: float
        Mesh size in m for the considered direction (axis).
    axis: int
        Axis on which the regularization is computed (0 for x, 1 for y).
    eps: float
        Small factor added in the square root to deal with the singularity at
        $\nabla u = 0$ when computing the gradient. The default is 1e-20.
    is_preconditioned: bool
        Whether the regularization is applied to a preconditioned parameter or not.
        Note that the value does not affect the behavior of the class (method results).
        It is stored here by convenience, to be used by the pyrtid inverse operator.
        The default is False.
    """

    dx: float
    axis: int
    eps: float = 1e-20
    is_preconditioned: bool = False

    def loss_function(self, param: NDArrayFloat) -> float:
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
        return np.sum(
            np.sqrt(np.square(gradient_ffd(param, self.dx, axis=self.axis)) + self.eps)
        )

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
        _gradient_ffd = gradient_ffd(param, self.dx, axis=self.axis)
        _gradient_bfd = gradient_bfd(param, self.dx, axis=self.axis)
        return (
            1.0
            / self.dx
            * (
                -1 / np.sqrt(np.square(_gradient_ffd) + self.eps) * _gradient_ffd
                + 1 / np.sqrt(np.square(_gradient_bfd) + self.eps) * _gradient_bfd
            )
        )


@dataclass
class TVRegularizatorIsotropic(Regularizator):
    r"""
    Apply an isotropic Total Variation (sharpening) regularization.

    Attributes
    ----------
    dx: float
        Mesh size in m for the x direction (axis 0).
    dx: float
        Mesh size in m for the y direction (axis 1).
    eps: float
        Small factor added in the square root to deal with the singularity at
        $\nabla u = 0$ when computing the gradient. The default is 1e-20.
    is_preconditioned: bool
        Whether the regularization is applied to a preconditioned parameter or not.
        Note that the value does not affect the behavior of the class (method results).
        It is stored here by convenience, to be used by the pyrtid inverse operator.
        The default is False.

    """

    dx: float
    dy: float
    eps: float = 1e-20
    is_preconditioned: bool = False

    def loss_function(self, param: NDArrayFloat) -> float:
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
        return np.sum(
            np.sqrt(
                np.square(gradient_ffd(param, self.dx, axis=0))
                + np.square(gradient_ffd(param, self.dy, axis=1))
                + self.eps
            )
        )

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
        _gradient_ffd_x = gradient_ffd(param, self.dx, axis=0)
        _gradient_bfd_x = gradient_bfd(param, self.dx, axis=0)
        _gradient_ffd_y = gradient_ffd(param, self.dy, axis=1)
        _gradient_bfd_y = gradient_bfd(param, self.dy, axis=1)

        term1 = -(_gradient_ffd_x / self.dx + _gradient_ffd_y / self.dy) / np.sqrt(
            np.square(_gradient_ffd_x) + np.square(_gradient_ffd_y) + self.eps
        )
        _temp2 = np.zeros(param.shape)
        _temp2[1:, :-1] = (param[:-1, 1:] - param[:-1, :-1]) / self.dy
        term2 = (
            _gradient_bfd_x
            / self.dx
            / np.sqrt(np.square(_gradient_bfd_x) + np.square(_temp2) + self.eps)
        )
        _temp3 = np.zeros(param.shape)
        _temp3[:-1, 1:] = (param[1:, :-1] - param[:-1, :-1]) / self.dx
        term3 = (
            _gradient_bfd_y
            / self.dy
            / np.sqrt(np.square(_temp3) + np.square(_gradient_bfd_y) + self.eps)
        )

        return term1 + term2 + term3
