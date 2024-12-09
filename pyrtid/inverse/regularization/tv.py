"""
Implement a Total Variation regularizator.

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
    make_spatial_permutation_matrices,
)
from pyrtid.utils.operators import gradient_ffd
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


@dataclass
class TVRegularizator(Regularizator):
    r"""
    Apply a Total Variation (sharpening) regularization.

    Attributes
    ----------
    geometry : Geometry
        Geometry of the field.
    eps: float
        Small factor added in the square root to deal with the singularity at
        $\nabla u = 0$ when computing the gradient. The default is 1e-20.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.

    """

    geometry: Geometry
    eps: float = 1e-20
    preconditioner: Preconditioner = NoTransform()

    def _get_grid_cell_l1(self, values: NDArrayFloat) -> NDArrayFloat:
        # sum of squared spatial gradient
        sg2 = np.zeros_like(values)
        if values.shape[0] > 2:
            sg2 += np.square(gradient_ffd(values, self.geometry.dx, axis=0))
        if values.shape[1] > 2:
            sg2 += np.square(gradient_ffd(values, self.geometry.dy, axis=1))
        # Add epsilon to prevent undetermination when deriving
        return np.sqrt(sg2 + self.eps)

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} \right)^{2}

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        float
        """
        _values = values.reshape(self.geometry.nx, self.geometry.ny, order="F")
        return float(np.sum(self._get_grid_cell_l1(_values)))

    def _eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
        """
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
        _values = values.reshape(self.geometry.nx, self.geometry.ny, order="F")
        grad = np.zeros_like(_values)
        den = self._get_grid_cell_l1(_values)

        if _values.shape[0] > 2:
            grad -= (
                gradient_ffd(_values, self.geometry.dx, axis=0) / self.geometry.dx / den
            )
            grad[1:, :] += (
                gradient_ffd(_values, self.geometry.dx, axis=0)[:-1, :]
                / self.geometry.dx
                / den[:-1, :]
            )

        if _values.shape[1] > 2:
            grad -= (
                gradient_ffd(_values, self.geometry.dy, axis=1) / self.geometry.dy / den
            )
            grad[:, 1:] += (
                gradient_ffd(_values, self.geometry.dy, axis=1)[:, :-1]
                / self.geometry.dy
                / den[:, :-1]
            )
        return grad.ravel("F")


@dataclass
class TVMatRegularizator(Regularizator):
    r"""
    Apply a Total Variation (sharpening) regularization with matrix formulation.

    Attributes
    ----------
    geometry : Geometry
        Geometry of the field
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.
    eps: float
        Small factor added in the square root to deal with the singularity at
        $\nabla u = 0$ when computing the gradient. The default is 1e-20.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.

    """

    geometry: Geometry
    sub_selection: Optional[NDArrayInt] = None
    eps: float = 1e-20
    preconditioner: Preconditioner = NoTransform()

    def __post_init__(self) -> None:
        """Post initialize the object."""
        self.mat_grad_x, self.mat_grad_y = make_spatial_gradient_matrices(
            self.geometry, self.sub_selection, which="forward"
        )

        self.mat_perm_x, self.mat_perm_y = make_spatial_permutation_matrices(
            self.geometry, self.sub_selection
        )

        self.permutation = self.mat_perm_x + self.mat_perm_y

    def _get_grid_cell_l1(self, values: NDArrayFloat) -> NDArrayFloat:
        # Add epsilon to prevent undetermination when deriving
        return np.sqrt(
            np.square(self.mat_grad_x @ values)
            + np.square(self.mat_grad_y @ values)
            + self.eps
        )

    def _eval_loss(self, values: NDArrayFloat) -> float:
        r"""
        Compute the gradient of the regularization loss function analytically.

        .. math::

        \mathcal{R}_{TN}(u) = \frac{1}{2} \sum_{j=1}^{M} \sum_{i=1}^{N}
        \left( \dfrac{u_{i+1, j} - u_{i,j}}{dx} \right)^{2}

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        float
        """
        return float(np.sum(self._get_grid_cell_l1(values.ravel("F"))))

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
        den = self._get_grid_cell_l1(values.ravel("F"))

        # x contribution
        grad += (
            self.mat_grad_x @ values.ravel("F") / den
            - self.mat_perm_x @ (self.mat_grad_x @ values.ravel("F") / den)
        ) / self.geometry.dx

        # y contribution
        grad += (
            self.mat_grad_y @ values.ravel("F") / den
            - self.mat_perm_y @ (self.mat_grad_y @ values.ravel("F") / den)
        ) / self.geometry.dy

        return grad.reshape(values.shape, order="F")


@dataclass
class TVFVMRegularizator(Regularizator):
    r"""
    Apply an Tikhonov (smoothing) regularization using the Finite Volume Method.

    Attributes
    ----------
    geometry : Geometry
        Geometry of the field.
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.
    eps: float
        Small factor added in the square root to deal with the singularity at
        $\nabla u = 0$ when computing the gradient. The default is 1e-20.
    preconditioner: Preconditioner
        Parameter pre-transformation operator (variable change for the solver).
        The default is the identity function: f(x) = x, which means no change
        is made.
    """

    geometry: Geometry
    sub_selection: Optional[NDArrayInt] = None
    eps: float = 1e-20
    preconditioner: Preconditioner = NoTransform()

    def __post_init__(self) -> None:
        """Post initialize the object."""
        # These are adjacence matrices (graphs)
        self.mat_perm_x, self.mat_perm_y = make_spatial_permutation_matrices(
            self.geometry, self.sub_selection
        )

    def _get_grid_cell_l1(self, v: NDArrayFloat) -> NDArrayFloat:
        # Add epsilon to prevent undetermination when deriving
        arr = np.zeros_like(v)
        if self.geometry.nx > 2:
            tmp: float = self.geometry.gamma_ij_x / self.geometry.mesh_volume
            arr += tmp**2 * (
                (self.mat_perm_x @ (self.mat_perm_x.T @ v) - self.mat_perm_x @ v) ** 2
                + (self.mat_perm_x.T @ (self.mat_perm_x @ v) - self.mat_perm_x.T @ v)
                ** 2
            )

        if self.geometry.ny > 2:
            tmp = self.geometry.gamma_ij_y / self.geometry.mesh_volume
            arr += tmp**2 * (
                (self.mat_perm_y @ (self.mat_perm_y.T @ v) - self.mat_perm_y @ v) ** 2
                + (self.mat_perm_y.T @ (self.mat_perm_y @ v) - self.mat_perm_y.T @ v)
                ** 2
            )

        return np.sqrt(arr + self.eps)

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
        return float(np.sum(self._get_grid_cell_l1(values.ravel("F"))))

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
        l1 = self._get_grid_cell_l1(v.ravel("F"))

        grad = np.zeros(v.size)
        if self.geometry.nx > 2:
            tmp: float = (self.geometry.gamma_ij_x / self.geometry.mesh_volume) ** 2
            # term 1
            grad += (
                tmp
                * (
                    (self.mat_perm_x @ (self.mat_perm_x.T @ v) - self.mat_perm_x @ v)
                    + (
                        self.mat_perm_x.T @ (self.mat_perm_x @ v)
                        - self.mat_perm_x.T @ v
                    )
                )
                / l1
            )
            # term 2
            xfwd = self.mat_perm_x @ (self.mat_perm_x.T @ v) - self.mat_perm_x @ v
            # forward denominator
            denfwd = self.mat_perm_x @ l1
            mask = denfwd > 0
            grad[mask] += tmp * xfwd[mask] / denfwd[mask]

            xbwd = self.mat_perm_x.T @ (self.mat_perm_x @ v) - self.mat_perm_x.T @ v
            # backward denominator
            denbwd = self.mat_perm_x.T @ l1
            mask = denbwd > 0
            grad[mask] += tmp * xbwd[mask] / denbwd[mask]

        if self.geometry.ny > 2:
            tmp = (self.geometry.gamma_ij_y / self.geometry.mesh_volume) ** 2
            # term 1
            grad += (
                tmp
                * (
                    (self.mat_perm_y @ (self.mat_perm_y.T @ v) - self.mat_perm_y @ v)
                    + (
                        self.mat_perm_y.T @ (self.mat_perm_y @ v)
                        - self.mat_perm_y.T @ v
                    )
                )
                / l1
            )
            # term 2
            yfwd = self.mat_perm_y @ (self.mat_perm_y.T @ v) - self.mat_perm_y @ v
            # forward denominator
            denfwd = self.mat_perm_y @ l1
            mask = denfwd > 0
            grad[mask] += tmp * yfwd[mask] / denfwd[mask]

            ybwd = self.mat_perm_y.T @ (self.mat_perm_y @ v) - self.mat_perm_y.T @ v
            # backward denominator
            denbwd = self.mat_perm_y.T @ l1
            mask = denbwd > 0
            grad[mask] += tmp * ybwd[mask] / denbwd[mask]

        return grad
