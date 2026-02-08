# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from pyrtid.utils import NDArrayFloat


class PriorTerm(ABC):
    """Represent a prior term for the geostatistical regularization."""

    @abstractmethod
    def get_values(self, params: NDArrayFloat) -> Union[float, NDArrayFloat]:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """

    @abstractmethod
    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """


class NullPriorTerm(PriorTerm):
    """Represent a null prior term."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()

    def get_values(self, params: NDArrayFloat) -> float:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior. It has no effect
            with `NullPriorTerm`.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        return 0.0

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return np.zeros(input.shape)


class ConstantPriorTerm(PriorTerm):
    """Represent a prior (no influence of beta)."""

    def __init__(self, prior_values: NDArrayFloat) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        prior_values : NDArrayFloat
            Values given to the prior term.
        """
        super().__init__()
        self.prior_values: NDArrayFloat = prior_values.ravel("F")

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior. It has no effect
            with `ConstantPriorTerm`.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        if params.shape != self.prior_values.shape:
            raise ValueError(
                f"The given values have shape {params.shape} while the constant prior "
                f"has been defined with shape {self.prior_values.shape}!"
            )
        return self.prior_values

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return np.zeros(input.shape)


class MeanPriorTerm(PriorTerm):
    """Represent a mean prior."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior mean.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        return np.full(params.size, fill_value=np.sum(params)) / params.size

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return np.full(input.size, fill_value=np.sum(input)) / input.size


class EnsembleMeanPriorTerm(PriorTerm):
    """Represent a mean prior."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Initialize the instance."""
        super().__init__()
        if len(shape) != 2:
            raise ValueError(
                "The shape of an EnsembleMeanPriorTerm should be (N_s, N_e)"
                " with N_s the number of adjuted values and N_e the number of"
                " members in the ensemble."
            )
        self.shape: Tuple[int, int] = shape

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior mean.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        if params.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {params.shape}.")
        return np.mean(params, axis=1, keepdims=True)

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        if input.shape[0] != self.shape[0]:  # type: ignore
            raise ValueError(
                f"Expected a vector of size {self.shape[0]}, got {input.shape}."
            )

        return input / self.shape[1]


class DriftMatrix(PriorTerm):
    """Represent a drift matrix prior term."""

    __slots__: List[str] = ["mat"]

    def __init__(
        self, mat: NDArrayFloat, beta: Optional[Union[NDArrayFloat, float]] = None
    ) -> None:
        """_summary_

        Parameters
        ----------
        mat : NDArrayFloat
            Matrix of coefficients: X. with shape (Ns, Nbeta)
        beta : Optional[Union[NDArrayFloat, float]], optional
            P Coefficients, by default None. # TODO: add references and comment better.
        """
        self.mat: NDArrayFloat = mat
        self.beta: Optional[Union[NDArrayFloat, float]] = beta

        if beta is not None:
            if isinstance(beta, float):
                shape = (1,)
            else:
                shape = beta.shape
            if shape[0] != mat.shape[1]:
                raise ValueError(
                    f"beta has shape {shape} while it should be shape "
                    f"({mat.shape[1]},) to match the given coefficient matrix."
                )

    @property
    def s_dim(self) -> int:
        return self.mat.shape[0]

    @property
    def beta_dim(self) -> int:
        return self.mat.shape[1]

    def dot(self, beta: Union[float, NDArrayFloat]) -> NDArrayFloat:
        """Return the dot product."""
        return np.dot(self.mat, beta)

    def get_values(self, params: NDArrayFloat) -> NDArrayFloat:
        """
        Return the values of the prior term.

        Parameters
        ----------
        params : NDArrayFloat
            Values of the parameters for which to compute the prior. It has no effect
            with `DriftMatrix`.

        Returns
        -------
        NDArrayFloat
            The prior term values.
        """
        if params.size != self.mat.shape[0]:
            raise ValueError(
                f"The given values have size {params.size} while the X matrix "
                f"has been defined with shape {self.mat.shape}!"
            )
        if self.beta is None:
            raise ValueError("beta is None! A value must be given.")
        return self.dot(self.beta)

    def get_gradient_dot_product(
        self, input: NDArrayFloat
    ) -> Union[float, NDArrayFloat]:
        """
        Return the dot product of the gradient of the prior and the given input vector.

        Parameters
        ----------
        params : NDArrayFloat
            Values with which to compute the prior gradient dot product.

        Returns
        -------
        NDArrayFloat
            Prior gradient-input vector dot product.
        """
        return 0.0


class ConstantDriftMatrix(DriftMatrix):
    """Represent a constant drift matrix (trend)."""

    # TODO: complete this one and complexify a bit

    def __init__(self, n_pts: int) -> None:
        """_summary_

        Parameters
        ----------
        pts : NDArrayFloat
            _description_
        """
        mat: NDArrayFloat = np.ones((n_pts, 1), dtype="d") / np.sqrt(n_pts)
        super().__init__(mat)


class LinearDriftMatrix(DriftMatrix):
    """Represent a linear drift matrix (trend)."""

    # TODO: complete this one and complexify a bit

    def __init__(self, pts: NDArrayFloat) -> None:
        """_summary_

        Parameters
        ----------
        pts : NDArrayFloat
            _description_
        """
        mat: NDArrayFloat = np.ones((pts.shape[0], 1 + pts.shape[1]), dtype=np.float64)
        mat[:, 1 : mat.shape[1]] = np.copy(pts)
        super().__init__(mat)
