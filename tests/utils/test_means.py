import sys
from typing import Optional

import numdifftools as nd
import numpy as np
import pytest
from scipy.stats import gmean, hmean

from pyrtid.utils import (
    MeanType,
    amean_gradient,
    arithmetic_mean,
    dxi_arithmetic_mean,
    dxi_harmonic_mean,
    get_mean_values_for_last_axis,
    gmean_gradient,
    harmonic_mean,
    hmean_gradient,
)
from pyrtid.utils.types import NDArrayFloat


@pytest.mark.parametrize(
    "xi,xj,expected", [(0.0, 0.0, 0.0), (0.0, 5.0, 2.5), (6.0, 3.0, 4.5)]
)
def test_arithmetic_mean(xi, xj, expected) -> None:
    assert arithmetic_mean(xi, xj) == expected


def test_dxi_arithmetic_mean() -> None:
    xi = np.linspace(0.0, 1e3, num=30)
    xj = np.linspace(1e2, 1e5, num=30)

    np.testing.assert_allclose(
        dxi_arithmetic_mean(xi, xj),
        nd.Derivative(arithmetic_mean, n=1, step=sys.float_info.epsilon * 1e10)(xi, xj),
        rtol=0.05,
    )


@pytest.mark.parametrize("xi,xj,expected", [(1e-5, 1e-7, 1.98e-7), (1e6, 1e0, 2.0)])
def test_harmonic_mean(xi, xj, expected) -> None:
    np.testing.assert_allclose(harmonic_mean(xi, xj), expected, rtol=1e-1)


def test_dxi_harmonic_mean() -> None:
    xi = np.power(10, np.linspace(-12, -3, num=20))
    xj = np.power(10, np.linspace(-9, -1, num=20))

    np.testing.assert_allclose(
        dxi_harmonic_mean(xi, xj),
        nd.Derivative(harmonic_mean, n=1, step=sys.float_info.epsilon)(xi, xj),
        rtol=0.05,
    )


@pytest.mark.parametrize(
    "arr, mean_type, weights, expected",
    (
        (np.ones((3, 3, 4)), MeanType.ARITHMETIC, None, np.ones((4))),
        (np.ones((3, 3, 4)), MeanType.GEOMETRIC, None, np.ones((4))),
        (np.ones((3, 3, 4)), MeanType.HARMONIC, None, np.ones((4))),
        (np.ones((1, 3, 5)), MeanType.ARITHMETIC, np.ones((3)), np.ones((5))),
        (np.ones((1, 2, 4)), MeanType.GEOMETRIC, np.ones((2)) * 0.5, np.ones((4))),
        (np.ones((3, 1, 2)), MeanType.HARMONIC, np.ones((3)), np.ones((2))),
        (
            np.repeat(np.array([[1, 2], [3, 4]])[:, :, np.newaxis], 5, axis=-1),
            MeanType.ARITHMETIC,
            np.ones((4)),
            np.ones((5)) * 2.5,
        ),
        (
            np.repeat(np.array([[1, 2], [3, 4]])[:, :, np.newaxis], 4, axis=-1),
            MeanType.GEOMETRIC,
            np.ones((4)),
            np.ones((4)) * 2.213364,
        ),
        (
            np.repeat(np.array([[1, 2], [3, 4]])[:, :, np.newaxis], 3, axis=-1),
            MeanType.HARMONIC,
            np.ones((4)),
            np.ones((3)) * 1.92,
        ),
    ),
)
def test_get_mean_values_for_last_axis(
    arr: NDArrayFloat,
    mean_type: MeanType,
    weights: Optional[NDArrayFloat],
    expected: NDArrayFloat,
) -> None:
    np.testing.assert_allclose(
        get_mean_values_for_last_axis(arr, mean_type, weights), expected
    )


def test_get_mean_values_for_last_axis_error() -> None:
    with pytest.raises(
        ValueError, match="The weights must match the number of meshes."
    ):
        get_mean_values_for_last_axis(np.ones((4, 12)), MeanType.ARITHMETIC, np.ones(5))


@pytest.mark.parametrize(
    "mean, mean_gradient",
    ((np.average, amean_gradient), (gmean, gmean_gradient), (hmean, hmean_gradient)),
)
def test_means_gradient(mean, mean_gradient) -> None:
    rng = np.random.default_rng(19680801)
    exponents = rng.integers(1, 10, size=20)
    test_values = np.power(10, -exponents.astype(np.float64))
    weights = rng.random(20)  # between 0 and 1

    fd = nd.Gradient(
        mean, step=np.min(test_values) * 1e-4
    )  # impose the step to avoid being below zero.
    # not optimal to test on a logsclae... so we use atol instead or rtol
    np.testing.assert_allclose(fd(test_values), mean_gradient(test_values), atol=1e-4)
    np.testing.assert_allclose(
        fd(test_values, weights=weights), mean_gradient(test_values, weights), atol=1e-4
    )
