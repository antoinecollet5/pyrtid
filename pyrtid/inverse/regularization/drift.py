from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from pyrtid.utils.types import NDArrayFloat


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
        return 0.0


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
        self.prior_values: NDArrayFloat = np.atleast_2d(prior_values)[:, ::-1].ravel(
            "F"
        )

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
        return 0.0


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
        return np.full(params.size, fill_value=np.mean(params))

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
            Matrix of coefficients: X.
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
