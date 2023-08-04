"""Unit tests for the spde utilities."""

from contextlib import contextmanager
from typing import Optional

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from sksparse.cholmod import Factor, cholesky

from pyrtid.utils.spde import (
    condition_precision_matrix,
    get_laplacian_matrix,
    get_laplacian_matrix_for_loops,
    get_precision_matrix,
    kriging,
    simu_c,
    simu_nc,
)
from pyrtid.utils.types import NDArrayFloat


@contextmanager
def does_not_raise():
    yield


def _get_precision_matrix(alpha) -> csc_matrix:
    """Return a precision matrix."""
    kappa = 1.56
    nx = 9
    ny = 5
    nz = 1
    dx = 2.1
    dy = 1.3
    dz = 1
    field_dimension = 2
    return get_precision_matrix(
        nx, ny, nz, dx, dy, dz, kappa, alpha, spatial_dim=field_dimension
    )


def _get_cholQ(alpha) -> Factor:
    """Return a cholesky factorization of the precision matrix."""
    return cholesky(_get_precision_matrix(alpha))


@pytest.mark.parametrize("kappa", [(5.0), (np.ones((9, 5, 3)))])
def test_get_laplacian_matrix(kappa) -> None:
    # 1) Test the get laplacian function
    nx = 9
    ny = 5
    nz = 3
    dx = 2.1
    dy = 1.3
    dz = 2.0
    a = get_laplacian_matrix_for_loops(nx, ny, nz, dx, dy, dz, kappa)
    b = get_laplacian_matrix(nx, ny, nz, dx, dy, dz, kappa)

    # see stackoverflow Q 30685024
    assert (a != b).nnz == 0


@pytest.mark.parametrize(
    "alpha, expected_exception",
    [
        (1.0, does_not_raise()),
        (3.0, does_not_raise()),
        (5, does_not_raise()),
        (
            0.5,
            pytest.raises(
                ValueError,
                match=(
                    r"alpha must be superior or equal to "
                    r"1.0 and must be an whole number!"
                ),
            ),
        ),
        (
            1.5,
            pytest.raises(
                ValueError,
                match=(
                    r"alpha must be superior or equal "
                    r"to 1.0 and must be an whole number!"
                ),
            ),
        ),
    ],
)
def test_get_precision_matrix(alpha, expected_exception) -> Optional[csc_matrix]:
    with expected_exception:
        return _get_precision_matrix(alpha)


@pytest.mark.parametrize(
    "alpha, random_state",
    [
        (1.0, 25693),  # using a seed
        (3.0, np.random.default_rng(256)),
        (5, np.random.RandomState(263)),
        (5, None),  # no random_state given
    ],
)
def test_simu_nc(alpha, random_state) -> NDArrayFloat:
    return simu_nc(_get_cholQ(alpha), random_state)


@pytest.mark.parametrize(
    "Q,cholQ,dat_var",
    [
        (_get_precision_matrix(1.0), cholesky(_get_precision_matrix(1.0)), None),
        (_get_precision_matrix(2.0), None, None),
        (_get_precision_matrix(3.0), None, np.array([0.1, 0.2, 0.7])),
    ],
)
def test_kriging(Q, cholQ, dat_var) -> NDArrayFloat:
    dat = np.array([5.5, 0.6, 7.9])
    dat_indices = np.array([5, 6, 10])
    return kriging(Q, dat, dat_indices, cholQ, dat_var=dat_var)


def test_simu_c() -> None:
    Q = _get_precision_matrix(1.0)
    cholQ = cholesky(Q)
    dat = np.array([5.5, 0.6, 7.9])
    dat_indices = np.array([5, 6, 10])
    dat_var = np.array([5.5, 0.6, 7.9])
    Q_cond = condition_precision_matrix(Q, dat_indices, dat_var)
    cholQ_cond = cholesky(Q_cond)

    simu_c(cholQ, Q_cond, cholQ_cond, dat, dat_indices, dat_var, 15369)


def test_condition_precision_matrix() -> None:
    alpha = 2.0
    Q = _get_precision_matrix(alpha)
    condition_precision_matrix(Q, np.array([1, 3, 6]), np.random.normal(size=3) ** 2)
