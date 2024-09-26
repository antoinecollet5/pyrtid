"""Unit tests for the spde utilities."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pyrtid.utils.spde as spde
import pytest
from pyrtid.utils import sparse_cholesky
from scipy.sparse import csc_array
from sksparse.cholmod import Factor


def test_matern_kernel() -> None:
    spatial_dim = 2
    alpha = 3
    len_scale = 10.0
    _nu = 2 * alpha - spatial_dim / 2
    spde.matern_kernel(np.random.normal(0.0, 5.0, size=50) * len_scale, len_scale, _nu)


def _get_precision_matrix(alpha) -> csc_array:
    """Return a precision matrix."""
    kappa = 1.56
    nx = 9
    ny = 5
    nz = 1
    dx = 2.1
    dy = 1.3
    dz = 1
    field_dimension = 2
    return spde.get_precision_matrix(
        nx, ny, nz, dx, dy, dz, kappa, alpha, spatial_dim=field_dimension
    )


def _get_cholQ(alpha) -> Factor:
    """Return a cholesky factorization of the precision matrix."""
    return sparse_cholesky(_get_precision_matrix(alpha).tocsc())


@pytest.mark.parametrize("kappa", [(5.0), (np.ones((9, 5, 3)))])
def test_get_laplacian_matrix(kappa) -> None:
    # 1) Test the get laplacian function
    nx = 9
    ny = 5
    nz = 3
    dx = 2.1
    dy = 1.3
    dz = 2.0
    a = spde.get_laplacian_matrix_for_loops(nx, ny, nz, dx, dy, dz, kappa)
    b = spde.get_laplacian_matrix(nx, ny, nz, dx, dy, dz, kappa)

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
def test_get_precision_matrix(alpha, expected_exception) -> None:
    with expected_exception:
        assert _get_precision_matrix(alpha).shape == (45, 45)


@pytest.mark.parametrize(
    "alpha, random_state",
    [
        (1.0, 25693),  # using a seed
        (3.0, np.random.default_rng(256)),
        (5, np.random.RandomState(263)),
        (5, None),  # no random_state given
    ],
)
def test_simu_nc(alpha, random_state) -> None:
    assert spde.simu_nc(_get_cholQ(alpha), random_state=random_state).shape == (45,)


@pytest.mark.parametrize(
    "Q,cholQ,dat_var",
    [
        (
            _get_precision_matrix(1.0),
            sparse_cholesky(_get_precision_matrix(1.0)),
            None,
        ),
        (_get_precision_matrix(2.0), None, None),
        (_get_precision_matrix(3.0), None, np.array([0.1, 0.2, 0.7])),
    ],
)
def test_kriging(Q, cholQ, dat_var) -> None:
    dat = np.array([5.5, 0.6, 7.9])
    dat_indices = np.array([5, 6, 10])
    assert spde.kriging(Q, dat, dat_indices, cholQ, dat_var=dat_var).shape == (45,)


def test_simu_c() -> None:
    Q = _get_precision_matrix(1.0)
    cholQ = sparse_cholesky(Q)
    dat = np.array([5.5, 0.6, 7.9])
    dat_indices = np.array([5, 6, 10])
    dat_var = np.array([5.5, 0.6, 7.9])
    Q_cond = spde.condition_precision_matrix(Q, dat_indices, dat_var)
    cholQ_cond = sparse_cholesky(Q_cond)

    spde.simu_c(cholQ, Q_cond, cholQ_cond, dat, dat_indices, dat_var, 15369)


def test_condition_precision_matrix() -> None:
    alpha = 2.0
    Q = _get_precision_matrix(alpha)
    spde.condition_precision_matrix(
        Q, np.array([1, 3, 6]), np.random.normal(size=3) ** 2
    )


def test_simu_nc_t_inv() -> None:
    # from pyrtid.utils.spde import simu_nc_t, simu_nc_t_inv

    # Grid
    nx = 20  # number of voxels along the x axis
    ny = 20  # number of voxels along the y axis
    nz = 1
    dx = 5.0  # voxel dimension along the x axis
    dy = 5.0  # voxel dimension along the y axis
    dz = 5.0

    len_scale = 20.0  # m
    kappa = 1 / len_scale
    alpha = 1.0
    std = 150.0  # standard deviation of the field

    # Create a precision matrix
    Q = spde.get_precision_matrix(
        nx, ny, nz, dx, dy, dz, kappa, alpha, spatial_dim=2, sigma=std
    )
    cholQ = sparse_cholesky(Q)

    w = np.random.default_rng(2024).normal(size=cholQ.L().shape[0])

    np.testing.assert_allclose(
        spde.simu_nc_t_inv(cholQ, spde.simu_nc_t(cholQ, w)), w, rtol=1e-12
    )

    np.testing.assert_allclose(
        spde.d_simu_nc_mat_vec_inv(cholQ, spde.d_simu_nc_mat_vec(cholQ, w)),
        w,
        rtol=1e-12,
    )
