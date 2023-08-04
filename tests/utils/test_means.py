import sys

import numdifftools as nd
import numpy as np
import pytest

from pyrtid.utils import (
    arithmetic_mean,
    dxi_arithmetic_mean,
    dxi_harmonic_mean,
    harmonic_mean,
)


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
