# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide some classic means."""

# pylint: disable=C0103 # doesn't conform to snake_case naming style
from typing import Optional

import numpy as np
from scipy.stats import gmean, hmean

from pyrtid.utils.enum import StrEnum
from pyrtid.utils.types import NDArrayFloat


def arithmetic_mean(xi: NDArrayFloat, xj) -> NDArrayFloat:
    """Return the arithmetic mean of xi and xj."""
    return (xi + xj) / 2.0


def dxi_arithmetic_mean(xi: NDArrayFloat, xj: NDArrayFloat) -> NDArrayFloat:
    """Return the first derivative of xi and xj arithmetic mean with respect to xi."""
    # pylint: disable=W0613 # unused argument
    return 0.5 + xi * 0.0  # required to work with vectors


def harmonic_mean(xi: NDArrayFloat, xj) -> NDArrayFloat:
    """Return the harmonic mean of xi and xj."""
    return 2.0 / (1.0 / xi + 1.0 / xj)


def dxi_harmonic_mean(xi: NDArrayFloat, xj) -> NDArrayFloat:
    """Return the first derivative of xi and xj arithmetic mean with respect to xi."""
    return 2.0 * xj**2.0 / (xi + xj) ** 2.0


class MeanType(StrEnum):
    HARMONIC = "harmonic"
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


def get_mean_values_for_last_axis(
    arr: NDArrayFloat, mean_type: MeanType, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Get the mean values for the last axis of the input array.

    Parameters
    ----------
    arr : _type_
        Array of values with shape (nx, ny, nt) or (npts, nt).
    mean_type: MeanType
        Type of mean chosen to average the simulated value when the observed one
        is defined over several grid cells of the domain.
    weights: Optional[NDArrayFloat]
        Weights to apply
    Returns
    -------
    NDArrayFloat
        Averaged values for the last axis.

    """
    _arr = np.asarray(arr)
    # ensure a second axis
    if len(_arr.shape) == 1:
        _arr = _arr[:, np.newaxis]
    # or make 2D
    else:
        _arr = _arr.reshape(-1, _arr.shape[-1])
    if weights is not None:
        if _arr.shape[0] != weights.size:
            raise ValueError(
                "The number of weights must match the number of grid cells."
            )

    return np.apply_along_axis(
        {
            MeanType.ARITHMETIC: np.average,
            MeanType.GEOMETRIC: gmean,
            MeanType.HARMONIC: hmean,
        }[mean_type],
        axis=0,
        arr=_arr,
        weights=weights,
    )


def amean_gradient(
    values: NDArrayFloat, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """Return the gradient of the weighted arithmetic mean."""
    if weights is not None:
        return weights / np.sum(weights)
    return np.ones(values.shape) / values.size


def hmean_gradient(
    values: NDArrayFloat, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """Return the gradient of the harmonic arithmetic mean."""
    if weights is None:
        return values.size / (np.square(values * np.sum(1.0 / values)))

    return weights * np.sum(weights) / np.square(values * np.sum(weights / values))


def gmean_gradient(
    values: NDArrayFloat, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """Return the gradient of the weighted geometric mean."""
    k: int = values.size
    if weights is None:
        return 1 / k * np.power(np.prod(values), (1 / k)) / values
    return weights / (values * np.sum(weights)) * gmean(values, weights=weights)


def get_mean_values_gradient_for_last_axis(
    arr: NDArrayFloat, mean_type: MeanType, weights: Optional[NDArrayFloat] = None
) -> NDArrayFloat:
    """
    Get the mean values for the last axis of the input array.

    Parameters
    ----------
    arr : _type_
        Array of values with shape (nx, ny, nt) or (npts, nt).
    mean_type: MeanType
        Type of mean chosen to average the simulated value when the observed one
        is defined over several grid cells of the domain.
    weights: Optional[NDArrayFloat]
        Weights to apply
    Returns
    -------
    NDArrayFloat
        Averaged values for the last axis.

    """
    # ensure a second axis
    if len(arr.shape) == 1:
        _arr = arr[:, np.newaxis]
    # or make 2D
    else:
        _arr = arr.reshape(-1, arr.shape[-1])
    if weights is not None:
        if _arr.shape[0] != weights.size:
            raise ValueError(
                "The number of weights must match the number of grid cells."
            )

    return np.apply_along_axis(
        {
            MeanType.ARITHMETIC: amean_gradient,
            MeanType.GEOMETRIC: gmean_gradient,
            MeanType.HARMONIC: hmean_gradient,
        }[mean_type],
        axis=0,
        arr=_arr,
        weights=weights,
    ).reshape(arr.shape)
