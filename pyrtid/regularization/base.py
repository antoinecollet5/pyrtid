# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

import logging
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy.sparse import csc_array, lil_array

from pyrtid.forward.models import get_owner_neigh_indices
from pyrtid.utils import NDArrayFloat, NDArrayInt, RectilinearGrid
from pyrtid.utils.finite_differences import finite_gradient
from pyrtid.utils.preconditioner import NoTransform, Preconditioner


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
        logger: Optional[logging.Logger] = None,
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
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.

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
            logger,
        )

    @classmethod
    def is_adaptive(cls) -> bool:
        """Return whether the method is adaptive."""
        return False

    def _update_reg_weight(
        self,
        loss_ls_history: List[float],
        loss_reg_history: List[float],
        reg_weight_history: List[float],
        loss_ls_grad: NDArrayFloat,
        loss_reg_grad: NDArrayFloat,
        n_obs: int,
        logger: Optional[logging.Logger] = None,
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

    def eval_loss(self, values: NDArrayFloat) -> float:
        """
        Compute the regularization loss function.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.

        Returns
        -------
        NDArrayFloat
            The regularization gradient.
        """
        if not values.ndim == 1:
            raise ValueError("The 'eval_loss' method expects a 1D vector!")

        return self._eval_loss(self.preconditioner(values))

    @abstractmethod
    def _eval_loss(self, values: NDArrayFloat) -> float: ...  # pragma: no cover

    def eval_loss_gradient_analytical(self, values: NDArrayFloat) -> NDArrayFloat:
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
        if not values.ndim == 1:
            raise ValueError(
                "The 'eval_loss_gradient_analytical' method expects a 1D vector!"
            )
        return self._eval_loss_gradient_analytical(values)

    @abstractmethod
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

    def eval_loss_gradient(
        self,
        values: NDArrayFloat,
        is_finite_differences: bool = False,
        max_workers: int = 1,
    ) -> NDArrayFloat:
        """
        Compute the gradient of the regularization loss function.

        Parameters
        ----------
        values : NDArrayFloat
            The parameter for which the regularization is computed.
        is_finite_differences: bool
            If true, a numerical approximation by 2nd order finite difference is
            returned. Cost twice the `values` dimensions in terms of loss function
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
        if not values.ndim == 1:
            raise ValueError("The 'eval_loss_gradient' method expects a 1D vector!")

        if is_finite_differences:
            return finite_gradient(values, self.eval_loss, max_workers=max_workers)
        else:
            return self.preconditioner.dtransform_vec(
                values,
                self.eval_loss_gradient_analytical(self.preconditioner(values)),
            )


def make_spatial_gradient_matrices(
    grid: RectilinearGrid,
    sub_selection: Optional[NDArrayInt] = None,
    which: Literal["forward", "backward", "both"] = "both",
) -> Tuple[csc_array, csc_array]:
    """
    Make matrices to compute the spatial gradient along x and y axes of a field.

    The gradient is obtained by the dot product between the field and the matrix.

    Parameters
    ----------
    grid : RectilinearGrid
        RectilinearGrid of the field
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.

    Returns
    -------
    Tuple[csc_array, csc_array]
        Spatial gradient matrices for x and y axes.
    """
    dim = grid.n_grid_cells
    # matrix for the spatial gradient along the x axis
    mat_grad_x = lil_array((dim, dim), dtype=np.float64)
    # matrix for the spatial gradient along the y axis
    mat_grad_y = lil_array((dim, dim), dtype=np.float64)

    if sub_selection is None:
        _sub_selection: NDArrayInt = np.arange(dim)
    else:
        _sub_selection = sub_selection

    # X contribution
    if grid.nx >= 2:
        tmp = grid.gamma_ij_x / grid.grid_cell_volume

        if which in ["forward", "both"]:
            # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
            idc_owner, idc_neigh = get_owner_neigh_indices(
                grid,
                (slice(0, grid.nx - 1), slice(None)),
                (slice(1, grid.nx), slice(None)),
                owner_indices_to_keep=_sub_selection,
                neigh_indices_to_keep=_sub_selection,
            )

            mat_grad_x[idc_owner, idc_neigh] -= tmp * np.ones(idc_owner.size)  # type: ignore
            mat_grad_x[idc_owner, idc_owner] += tmp * np.ones(idc_owner.size)  # type: ignore

        if which in ["backward", "both"]:
            # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
            idc_owner, idc_neigh = get_owner_neigh_indices(
                grid,
                (slice(1, grid.nx), slice(None)),
                (slice(0, grid.nx - 1), slice(None)),
                owner_indices_to_keep=_sub_selection,
                neigh_indices_to_keep=_sub_selection,
            )

            mat_grad_x[idc_owner, idc_neigh] -= tmp * np.ones(idc_owner.size)  # type: ignore
            mat_grad_x[idc_owner, idc_owner] += tmp * np.ones(idc_owner.size)  # type: ignore

    # Y contribution
    if grid.ny >= 2:
        tmp = grid.gamma_ij_y / grid.grid_cell_volume

        if which in ["forward", "both"]:
            # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
            idc_owner, idc_neigh = get_owner_neigh_indices(
                grid,
                (slice(None), slice(0, grid.ny - 1)),
                (slice(None), slice(1, grid.ny)),
                owner_indices_to_keep=_sub_selection,
                neigh_indices_to_keep=_sub_selection,
            )

            mat_grad_y[idc_owner, idc_neigh] -= tmp * np.ones(idc_owner.size)  # type: ignore
            mat_grad_y[idc_owner, idc_owner] += tmp * np.ones(idc_owner.size)  # type: ignore

        if which in ["backward", "both"]:
            # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
            idc_owner, idc_neigh = get_owner_neigh_indices(
                grid,
                (slice(None), slice(1, grid.ny)),
                (slice(None), slice(0, grid.ny - 1)),
                owner_indices_to_keep=_sub_selection,
                neigh_indices_to_keep=_sub_selection,
            )
            mat_grad_y[idc_owner, idc_neigh] -= tmp * np.ones(idc_owner.size)  # type: ignore
            mat_grad_y[idc_owner, idc_owner] += tmp * np.ones(idc_owner.size)  # type: ignore

    return mat_grad_x.tocsc(), mat_grad_y.tocsc()


def make_spatial_permutation_matrices(
    grid: RectilinearGrid, sub_selection: Optional[NDArrayInt] = None
) -> Tuple[csc_array, csc_array]:
    """
    Make matrices to compute the spatial permutations along x and y axes of a field.

    Parameters
    ----------
    grid : RectilinearGrid
        RectilinearGrid of the field
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.

    Returns
    -------
    Tuple[csc_array, csc_array]
        Spatial permutation matrices for x and y axes.
    """
    dim = grid.n_grid_cells
    # matrix for the spatial permutation along the x axis
    mat_perm_x = lil_array((dim, dim), dtype=np.float64)
    # matrix for the spatial permutation along the y axis
    mat_perm_y = lil_array((dim, dim), dtype=np.float64)

    if sub_selection is None:
        _sub_selection: NDArrayInt = np.arange(dim)
    else:
        _sub_selection = sub_selection

    # X contribution
    if grid.nx >= 2:
        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(0, grid.nx - 1), slice(None)),
            (slice(1, grid.nx), slice(None)),
            owner_indices_to_keep=_sub_selection,
            neigh_indices_to_keep=_sub_selection,
        )

        mat_perm_x[idc_neigh, idc_owner] = np.ones(idc_owner.size)  # type: ignore

    # Y contribution
    if grid.ny >= 2:
        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            (slice(None), slice(0, grid.ny - 1)),
            (slice(None), slice(1, grid.ny)),
            owner_indices_to_keep=_sub_selection,
            neigh_indices_to_keep=_sub_selection,
        )

        mat_perm_y[idc_neigh, idc_owner] = np.ones(idc_owner.size)  # type: ignore

    return mat_perm_x.tocsc(), mat_perm_y.tocsc()
