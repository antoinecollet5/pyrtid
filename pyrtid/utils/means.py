"""Provide some classic means."""
# pylint: disable=C0103 # doesn't conform to snake_case naming style
import numba as nb

from pyrtid.utils.types import NDArrayFloat


@nb.jit(nopython=True, cache=True)
def arithmetic_mean(xi: NDArrayFloat, xj) -> NDArrayFloat:
    """Return the arithmetic mean of xi and xj."""
    return (xi + xj) / 2


@nb.jit(nopython=True, cache=True)
def dxi_arithmetic_mean(xi: NDArrayFloat, xj: NDArrayFloat) -> NDArrayFloat:
    """Return the first derivative of xi and xj arithmetic mean with respect to xi."""
    # pylint: disable=W0613 # unused argument
    return 0.5 + xi * 0.0  # required to work with vectors


@nb.jit(nopython=True, cache=True)
def harmonic_mean(xi: NDArrayFloat, xj) -> NDArrayFloat:
    """Return the harmonic mean of xi and xj."""
    return 2 / (1 / xi + 1 / xj)


@nb.jit(nopython=True, cache=True)
def dxi_harmonic_mean(xi: NDArrayFloat, xj) -> NDArrayFloat:
    """Return the first derivative of xi and xj arithmetic mean with respect to xi."""
    return 2 * xj**2 / (xi + xj) ** 2
