import copy
import logging

import numpy as np
import pytest
from pyrtid.inverse.regularization import (
    SparseInvCovarianceMatrix,
    eigen_factorize_cov_mat,
)
from pyrtid.inverse.solvers import PCGA
from pyrtid.utils import NDArrayFloat, sparse_cholesky, spde
from scipy.ndimage import gaussian_filter


def forward_model(x) -> NDArrayFloat:
    return gaussian_filter(np.sqrt(x), sigma=2.0)


# Create observations
def sample_d(d) -> NDArrayFloat:
    percent_of_values = 0.05
    return d.ravel("F")[:: int(d.size / (percent_of_values * 1000))]


def get_forward_model_wrapper(n_obs: int, nx: int, ny: int):
    def forward_model_wrapper(s_ensemble: NDArrayFloat) -> NDArrayFloat:
        d_pred = np.zeros((n_obs, s_ensemble.shape[1]))
        for i in range(s_ensemble.shape[1]):
            # use preconditionning
            res = forward_model(s_ensemble[:, i].reshape(nx, ny, order="F") ** 2)
            d_pred[:, i] = sample_d(res)
        return d_pred

    return forward_model_wrapper


def forward_model_wrapper(s_ensemble: NDArrayFloat) -> NDArrayFloat:
    d_pred = np.zeros((51, s_ensemble.shape[1]))
    for i in range(s_ensemble.shape[1]):
        # use preconditionning
        res = forward_model(s_ensemble[:, i].reshape(65, 65, order="F") ** 2)
        d_pred[:, i] = sample_d(res)
    return d_pred


def test_init() -> None:
    pass


@pytest.mark.parametrize(
    "is_direct_solve",
    [True, False],
)
@pytest.mark.parametrize(
    "is_use_preconditioner",
    [True, False],
)
@pytest.mark.parametrize(
    "is_lm",
    [
        True,
    ],
)
# @pytest.mark.parametrize(
#     "max_lm_workers,
#     [1, ],
# )
@pytest.mark.parametrize(
    "logger",
    [logging.getLogger("PCGA")],
)
def test_large_medium_scale(
    is_use_preconditioner: bool,
    is_direct_solve: bool,
    is_lm: bool,
    logger: logging.Logger,
) -> None:
    # Parameters for the domain, covariance etc.
    nx = 65
    ny = 65
    nz = 1
    dx = 5.0
    dy = 5.0
    dz = 1.0
    len_scale = 20.0  # m
    kappa = 1 / len_scale
    alpha = 1.0

    mean = 300.0  # trend of the field
    std = 150.0  # standard deviation of the field

    # Create a precision matrix
    Q_ref = spde.get_precision_matrix(
        nx, ny, nz, dx, dy, dz, kappa, alpha, spatial_dim=2, sigma=std
    )
    cholQ_ref = sparse_cholesky(Q_ref)
    # Non conditional simulation -> change the random state to obtain a different field
    simu_ = spde.simu_nc(cholQ_ref, random_state=2026).reshape(nx, ny, order="F")
    s_ref = np.abs(simu_ + mean)

    # Initial guess
    simu_ = spde.simu_nc(cholQ_ref, random_state=15653).reshape(nx, ny, order="F")
    s_init = np.abs(simu_)

    d_ref = forward_model(s_ref)
    obs = sample_d(d_ref)
    obs.shape

    sp_cov = SparseInvCovarianceMatrix(Q_ref)

    # Eigen factorization
    eig_mat = eigen_factorize_cov_mat(sp_cov, n_pc=50)

    assert eig_mat.number_pts == nx * ny

    # Forward model
    # forward_model_wrapper = get_forward_model_wrapper(obs.size, nx, ny)

    # test the forward model
    s_ens = np.vstack([s_ref.ravel("F"), s_init.ravel("F")]).T
    assert s_ens.shape == (nx * ny, 2)

    d_pred = forward_model_wrapper(np.sqrt(s_ens))
    assert d_pred.shape == (obs.size, 2)

    np.testing.assert_almost_equal(d_pred[:, 0], obs)

    # perturb the observations
    # 3% error on the observations
    cov_obs = (np.ones(obs.shape) * (np.max(obs) - np.min(obs)) * 0.03) ** 2
    obs_perturb = obs * (1 + np.random.default_rng(2151).normal(loc=0, scale=cov_obs))

    # Create a solver
    solver = PCGA(
        np.sqrt(s_init.ravel("F")),
        obs_perturb,
        cov_obs,
        copy.deepcopy(forward_model_wrapper),
        eig_mat,
        maxiter=5,
        random_state=12,
        is_lm=is_lm,
        is_use_preconditioner=is_use_preconditioner,
        is_direct_solve=is_direct_solve,
        is_save_jac=True,
        logger=logger,
    )

    assert solver.s_dim == nx * ny
    assert solver.d_dim == obs.size

    solver.run()

    # Test the matrix equivalence
    if is_direct_solve:
        Psi = solver.get_psi(solver.HZ, solver.cov_obs, solver.istate.inflation_seq[-1])
        A = solver.build_dense_A(Psi, solver.HX)
        LA = solver.build_cholesky(Psi, solver.HX)
        A2 = solver.build_dense_A_from_cholesky(LA, p=solver.HX.shape[1])
        np.testing.assert_array_almost_equal(A, A2)

        b = np.random.default_rng(225464).random(A.shape[0])

        x = np.linalg.solve(A, b)
        x2 = solver.solve_cholesky(LA, b, p=solver.HX.shape[1])
        np.testing.assert_array_almost_equal(x, x2)
