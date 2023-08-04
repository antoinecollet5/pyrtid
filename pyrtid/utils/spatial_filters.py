"""
Provide interfaces to filter (i.e., smoothing) the gradient on desired iterations.

@author: acollet
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence, Union

from scipy.ndimage import gaussian_filter
from typing_extensions import Literal

from pyrtid.utils.types import NDArrayFloat, object_or_object_sequence_to_list


@dataclass
class Filter:
    """
    A gradient filter.

    This is an abstract class.
    """

    @abstractmethod
    def filter(self, param: NDArrayFloat, iteration: int) -> NDArrayFloat:
        """Filter the given values.

        Parameters
        ----------
        param : NDArrayFloat
            Values to filter.
        iteration : int
            Iteration number knowing that iterations start at 1 (and not at zero).

        Returns
        -------
        NDArrayFloat
            Filtered values.

        """
        ...  # pragma: no cover


@dataclass
class GaussianFilter(Filter):
    # pylint: disable=C0301 # line too long
    """
    Apply gaussian filter filtering.

    See
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html

    Attributes
    ----------
    sigmas : float, sequence of scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes. If a float is provided, the value is taken for each filtering. If
        a list is provided, then the values are taken in the list depending on the
        iteration index (see filter method).
    order : {0, 1, 2, 3} or sequence from same set, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number.  An order of 0 corresponds
        to convolution with a Gaussian kernel. An order of 1, 2, or 3
        corresponds to convolution with the first, second or third
        derivatives of a Gaussian. Higher order derivatives are not
        implemented. The default is 0.0.
    mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        The mode parameter determines how the array borders are handled, where cval is
        the value when mode is equal to ‘constant’. Default is ‘reflect’.
    cval : scalar, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0
    truncate : float
        Truncate the filter at this many standard deviations. Default is 4.0.

    """

    sigmas: Union[float, Sequence[Union[float, Sequence[float]]]]
    order: int = 0
    mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "reflect"
    cval: float = 0
    truncate: float = 4.0

    def filter(self, param: NDArrayFloat, iteration: int) -> NDArrayFloat:
        """Apply a gaussian smoothing to the given values.

        Parameters
        ----------
        param : NDArrayFloat
            Values to filter.
        iteration : int
            Iteration number knowing that iterations start at 1 (and not at zero).

        Returns
        -------
        NDArrayFloat
            Filtered values.
        """

        return gaussian_filter(
            param,
            get_sigma(self.sigmas, iteration - 1, param.ndim),
            self.order,
            mode=self.mode,
            cval=self.cval,
            truncate=self.truncate,
        )


def get_sigma(
    sigmas: Union[float, Sequence[Union[float, Sequence[float]]]],
    index: int,
    dim: int,
) -> Union[float, Sequence[float]]:
    """
    Get the sigma.

    Parameters
    ----------
    index : int
        Index in the sequence of sigmas.
    dim : int
        Spatial dimension (1, 2 or 3).

    Returns
    -------
    Union[float, Sequence[float]]
        The computed sigmas.

    Raises
    ------
    ValueError
        If the dimension does not match.
    """
    if isinstance(sigmas, float):
        return sigmas
    if isinstance(sigmas, int):
        return float(sigmas)
    try:
        sigma = object_or_object_sequence_to_list(sigmas)[index]
    except IndexError:
        return 0.0
    if isinstance(sigma, float):
        return sigma
    if isinstance(sigma, int):
        return float(sigma)
    dim_sigma: int = len(sigma)
    if dim_sigma != dim:
        raise ValueError(
            "Sigmas should have the same dimension as the given parameter !"
        )
    return sigma
