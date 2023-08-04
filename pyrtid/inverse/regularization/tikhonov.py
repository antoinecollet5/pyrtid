"""
Implement a Tikhonov regularizator.

TODO: add references.

@author: acollet
"""

from dataclasses import dataclass

import numpy as np

from pyrtid.utils.operators import gradient_ffd, hessian_cfd
from pyrtid.utils.types import NDArrayFloat

from .base import Regularizator


@dataclass
class TikhonovRegularizatorAnisotropic(Regularizator):
    """
    Apply an anisotropic Tikonov (smoothing) regularization.

    Attributes
    ----------
    dx: float
        Mesh size in m for the considered direction (axis).
    axis: int
        Axis on which the regularization is computed (0 for x, 1 for y).
    is_preconditioned: bool
        Whether the regularization is applied to a preconditioned parameter or not.
        Note that the value does not affect the behavior of the class (method results).
        It is stored here by convenience, to be used by the pyrtid inverse operator.
        The default is False.
    """

    dx: float
    axis: int
    is_preconditioned: bool = False

    def loss_function(self, param: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        .. math::

        \mathcal{R}_{TN}(u) = \frac{\alpha}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} \right)^{2}$$

        Parameters
        ----------
        param : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        return 0.5 * np.sum(gradient_ffd(param, self.dx, axis=self.axis) ** 2.0)

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
        return -hessian_cfd(param, self.dx, axis=self.axis)


@dataclass
class TikhonovRegularizatorIsotropic(Regularizator):
    r"""
    Apply an isotropic Tikonov (smoothing) regularization.

    Attributes
    ----------
    dx: float
        Mesh size in m for the x direction (axis 0).
    dx: float
        Mesh size in m for the y direction (axis 1).
    is_preconditioned: bool
        Whether the regularization is applied to a preconditioned parameter or not.
        Note that the value does not affect the behavior of the class (method results).
        It is stored here by convenience, to be used by the pyrtid inverse operator.
        The default is False.
    """

    dx: float
    dy: float
    is_preconditioned: bool = False

    def loss_function(self, param: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        The isotropic version of the Tikhonov regularization is defined as:

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
        return 0.5 * np.sum(
            (
                gradient_ffd(param, self.dx, axis=0)
                + gradient_ffd(param, self.dy, axis=1)
            )
            ** 2.0
        )

    def loss_function_gradient_analytical(self, param: NDArrayFloat) -> NDArrayFloat:
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
        # _gradient_ffd_x = gradient_ffd(param, self.dx, axis=0)
        # _gradient_bfd_x = gradient_bfd(param, self.dx, axis=0)
        # _gradient_ffd_y = gradient_ffd(param, self.dy, axis=1)
        # _gradient_bfd_y = gradient_bfd(param, self.dy, axis=1)

        # term1 = -(1 / self.dx + 1 / self.dy) * (_gradient_ffd_x + _gradient_ffd_y)
        # _temp2 = np.zeros(param.shape)
        # _temp2[1:,:-1] = (param[:-1, 1:] - param[:-1, :-1]) / self.dy
        # term2 = 1 / self.dx  * (_gradient_bfd_x + _temp2)
        # _temp3 = np.zeros(param.shape)
        # _temp3[:-1,1:] = (param[1:, :-1] - param[:-1, :-1]) / self.dx
        # term3 = 1 / self.dy * (_temp3 + _gradient_bfd_y)

        # return term1 + term2 + term3
        interaction_term = np.zeros(param.shape)

        # Compute the interaction term on i \in [1, N], j \in [1, M]
        interaction_term[1:-1, 1:-1] = (
            2 * param[1:-1, 1:-1]
            - param[2:, 1:-1]
            - param[1:-1, 2:]
            + param[:-2, 2:]
            - param[:-2, 1:-1]
            + param[2:, :-2]
            - param[1:-1, :-2]
        ) / (self.dx * self.dy)
        # Compute the interaction term on borders:
        interaction_term[0, 1:-1] = (
            2 * param[0, 1:-1]
            - param[1, 1:-1]
            - param[0, 2:]
            + param[1, :-2]
            - param[0, :-2]
        ) / (self.dx * self.dy)
        interaction_term[-1, 1:-1] = (param[-2, 2:] - param[-2, 1:-1]) / (
            self.dx * self.dy
        )
        interaction_term[1:-1, 0] = (
            2 * param[1:-1, 0]
            - param[2:, 0]
            - param[1:-1, 1]
            + param[:-2, 1]
            - param[:-2, 0]
        ) / (self.dx * self.dy)
        interaction_term[1:-1, -1] = (param[2:, -2] - param[1:-1, -2]) / (
            self.dx * self.dy
        )
        interaction_term[0, 0] = (2 * param[0, 0] - param[1, 0] - param[0, 1]) / (
            self.dx * self.dy
        )
        interaction_term[0, -1] = (param[1, -2] - param[0, -2]) / (self.dx * self.dy)
        interaction_term[-1, 0] = (param[-2, 1] - param[-2, 0]) / (self.dx * self.dy)
        interaction_term[-1, -1] = 0.0

        return (
            -hessian_cfd(param, self.dx, axis=0)
            - hessian_cfd(param, self.dy, axis=1)
            + interaction_term
        )
