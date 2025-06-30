"""
Implement a Tikhonov regularizator.

TODO: add references.

@author: acollet
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyrtid.regularization.base import (
    Regularizator,
    make_spatial_gradient_matrices,
    make_spatial_permutation_matrices,
)
from pyrtid.utils import NDArrayFloat, NDArrayInt, RectilinearGrid
from pyrtid.utils.operators import gradient_ffd, hessian_cfd
from pyrtid.utils.preconditioner import NoTransform, Preconditioner


@dataclass
class TikhonovRegularizator(Regularizator):
    r"""
    Apply an Tikhonov (smoothing) regularization.

    Attributes
    ----------
    grid : RectilinearGrid
        RectilinearGrid of the field.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    grid: RectilinearGrid
    preconditioner: Preconditioner = NoTransform()

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        The Tikhonov regularization is defined as:

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} + \dfrac{u_{i, j+1} - u_{i,j}}{dy}
        \right)^{2}

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        _values = values.reshape(self.grid.nx, self.grid.ny, order="F")
        f = 0.0
        if _values.shape[0] > 2:
            f += 0.5 * float(np.sum(gradient_ffd(_values, self.grid.dx, axis=0) ** 2.0))
        if _values.shape[1] > 2:
            f += 0.5 * float(np.sum(gradient_ffd(_values, self.grid.dy, axis=1) ** 2.0))
        return f

    def _eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        r"""
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        _values = values.reshape(self.grid.nx, self.grid.ny, order="F")
        grad = np.zeros_like(_values)
        if _values.shape[0] > 2:
            grad += -hessian_cfd(_values, self.grid.dx, 0)
        if _values.shape[1] > 2:
            grad += -hessian_cfd(_values, self.grid.dy, 1)
        return grad.ravel("F")


@dataclass
class TikhonovMatRegularizator(Regularizator):
    r"""
    Apply an Tikhonov (smoothing) regularization using matrix formulation.

    Attributes
    ----------
    grid : RectilinearGrid
        RectilinearGrid of the field.
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    grid: RectilinearGrid
    sub_selection: Optional[NDArrayInt] = None
    preconditioner: Preconditioner = NoTransform()

    def __post_init__(self) -> None:
        """Post initialize the object."""
        self.mat_grad_x, self.mat_grad_y = make_spatial_gradient_matrices(
            self.grid, self.sub_selection, which="forward"
        )

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        The Tikhonov regularization is defined as:

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} + \dfrac{u_{i, j+1} - u_{i,j}}{dy}
        \right)^{2}

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        f = 0.0
        f += 0.5 * float(np.sum((self.mat_grad_x @ values.ravel("F")) ** 2.0))
        f += 0.5 * float(np.sum((self.mat_grad_y @ values.ravel("F")) ** 2.0))
        return f

    def _eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        r"""
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        grad = np.zeros(values.size)
        grad += self.mat_grad_x.T @ self.mat_grad_x @ values.ravel("F")
        grad += self.mat_grad_y.T @ self.mat_grad_y @ values.ravel("F")
        return grad


@dataclass
class TikhonovFVMRegularizator(Regularizator):
    r"""
    Apply an Tikhonov (smoothing) regularization using the Finite Volume Method.

    Attributes
    ----------
    grid : RectilinearGrid
        RectilinearGrid of the field.
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    grid: RectilinearGrid
    sub_selection: Optional[NDArrayInt] = None
    preconditioner: Preconditioner = NoTransform()

    def __post_init__(self) -> None:
        """Post initialize the object."""
        # These are adjacence matrices (graphs)
        self.mat_perm_x, self.mat_perm_y = make_spatial_permutation_matrices(
            self.grid, self.sub_selection
        )

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        The Tikhonov regularization is defined as:

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} + \dfrac{u_{i, j+1} - u_{i,j}}{dy}
        \right)^{2}

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        f = 0.0
        v = values.ravel("F")
        if self.grid.nx > 2:
            tmp: float = self.grid.gamma_ij_x / self.grid.grid_cell_volume
            f += 0.25 * float(
                np.sum(
                    (
                        tmp**2
                        * (
                            (
                                self.mat_perm_x @ (self.mat_perm_x.T @ v)
                                - self.mat_perm_x @ v
                            )
                            ** 2
                            + (
                                self.mat_perm_x.T @ (self.mat_perm_x @ v)
                                - self.mat_perm_x.T @ v
                            )
                            ** 2
                        )
                    )
                )
            )

        if self.grid.ny > 2:
            tmp = self.grid.gamma_ij_y / self.grid.grid_cell_volume
            f += 0.25 * float(
                np.sum(
                    (
                        tmp**2
                        * (
                            (
                                self.mat_perm_y @ (self.mat_perm_y.T @ v)
                                - self.mat_perm_y @ v
                            )
                            ** 2
                            + (
                                self.mat_perm_y.T @ (self.mat_perm_y @ v)
                                - self.mat_perm_y.T @ v
                            )
                            ** 2
                        )
                    )
                )
            )

        return f

    def _eval_loss_gradient_analytical(self, v: NDArrayFloat) -> NDArrayFloat:
        r"""
        Compute the gradient of the regularization loss function analytically.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        grad = np.zeros(v.size)
        if self.grid.nx > 2:
            tmp: float = (self.grid.gamma_ij_x / self.grid.grid_cell_volume) ** 2
            grad += tmp * (
                (self.mat_perm_x @ (self.mat_perm_x.T @ v) - self.mat_perm_x @ v)
                + (self.mat_perm_x.T @ (self.mat_perm_x @ v) - self.mat_perm_x.T @ v)
            )

        if self.grid.ny > 2:
            tmp = (self.grid.gamma_ij_y / self.grid.grid_cell_volume) ** 2
            grad += tmp * (
                (self.mat_perm_y @ (self.mat_perm_y.T @ v) - self.mat_perm_y @ v)
                + (self.mat_perm_y.T @ (self.mat_perm_y @ v) - self.mat_perm_y.T @ v)
            )

        return grad
