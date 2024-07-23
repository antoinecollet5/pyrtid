from typing import Tuple

import numpy as np
import pyrtid.utils.spde as spde
import pytest
from pyrtid.inverse.preconditioner import LinearTransform
from pyrtid.inverse.regularization import (  # DriftMatrix,; LinearDriftMatrix,
    ConstantPriorTerm,
    EnsembleCovarianceMatrix,
    EnsembleMeanPriorTerm,
    EnsembleRegularizator,
    FFTCovarianceMatrix,
    GeostatisticalRegularizator,
    MeanPriorTerm,
    NullPriorTerm,
    SparseInvCovarianceMatrix,
    eigen_factorize_cov_mat,
    generate_dense_matrix,
)
from pyrtid.utils import sparse_cholesky
from pyrtid.utils.types import NDArrayFloat

# For now we use the exact parameters, we will complexify a bit later
prior_std = 1.0
len_scale: NDArrayFloat = np.array([250.0, 250.0])


def exponential_kernel(r: float) -> NDArrayFloat:
    """Test covariance kernel."""
    return (prior_std**2) * np.exp(-r)


def get_domain_shape() -> Tuple[int, int]:
    nx: int = 7
    ny: int = 11
    return (nx, ny)


def get_mesh_dim() -> Tuple[float, float]:
    dx: float = 3.5
    dy: float = 4.2
    return (dx, dy)


def get_pts() -> NDArrayFloat:
    dx, dy = get_mesh_dim()
    nx, ny = get_domain_shape()
    x = np.linspace(0.0 + dx / 2.0, nx * dx - dx / 2.0, nx)
    y = np.linspace(0.0 + dy / 2.0, ny * dy - dy / 2.0, ny)

    XX, YY = np.meshgrid(x, y)
    return np.hstack((XX.ravel("F")[:, np.newaxis], YY.ravel("F")[:, np.newaxis]))


def get_param_values() -> NDArrayFloat:
    """Generate a parameter field with some noise."""
    param: NDArrayFloat = np.zeros(get_domain_shape(), dtype=np.float64)
    param[0:5, 2:6] = 5.0
    param[3:6, 6:7] = 10.0
    param[2:5, 1:8] = 20.0

    # Add some noise with a seed
    rng = np.random.default_rng(26659)
    param += rng.random(get_domain_shape()) * 5.0

    return param


@pytest.mark.parametrize(
    "cov_mat,atol",
    [
        (
            generate_dense_matrix(
                pts=get_pts(),
                kernel=exponential_kernel,
                len_scale=len_scale,
            ),
            1e-4,
        ),
        (
            eigen_factorize_cov_mat(
                generate_dense_matrix(
                    pts=get_pts(), kernel=exponential_kernel, len_scale=len_scale
                ),
                n_pc=32,
            ),
            1e-5,
        ),
        (
            FFTCovarianceMatrix(
                kernel=exponential_kernel,
                mesh_dim=get_mesh_dim(),
                domain_shape=get_domain_shape(),
                len_scale=len_scale,
                k=30,
            ),
            1e-2,
        ),
        (
            eigen_factorize_cov_mat(
                FFTCovarianceMatrix(
                    kernel=exponential_kernel,
                    mesh_dim=get_mesh_dim(),
                    domain_shape=get_domain_shape(),
                    len_scale=len_scale,
                    k=30,
                ),
                n_pc=32,
            ),
            1e-4,
        ),
        (
            EnsembleCovarianceMatrix(
                np.random.default_rng(2023).random(
                    size=(200, np.prod(get_domain_shape()))
                )
            ),
            1e-4,
        ),
    ],
)
def test_regularizator_gradients_by_fd(cov_mat, atol) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    regularizator = GeostatisticalRegularizator(cov_mat)

    print(f"loss_reg_dense = {regularizator.eval_loss(param_values)}")

    grad_reg_fd = regularizator.eval_loss_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.eval_loss_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=atol)


@pytest.mark.parametrize(
    "prior",
    [
        NullPriorTerm(),
        ConstantPriorTerm(
            np.full(get_param_values().size, np.mean(get_param_values()))
        ),
        MeanPriorTerm(),
        #        DriftMatrix(),
        #        LinearDriftMatrix,
    ],
)
def test_regularizator_gradients_with_priors_by_fd(prior) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    cov_mat = eigen_factorize_cov_mat(
        FFTCovarianceMatrix(
            kernel=exponential_kernel,
            mesh_dim=get_mesh_dim(),
            domain_shape=get_domain_shape(),
            len_scale=len_scale,
            k=30,
        ),
        n_pc=32,
    )

    # transform to test the change of variable
    regularizator = GeostatisticalRegularizator(
        cov_mat, prior, preconditioner=LinearTransform(1.0, 4.0)
    )

    print(f"loss_reg_dense = {regularizator.eval_loss(param_values)}")

    grad_reg_fd = regularizator.eval_loss_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.eval_loss_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=1e-4)


def test_ensemble_regularizator() -> None:
    """In this test we use spde for fast simulation generation."""
    nx = (
        10  # number of voxels along the x axis + 4 * 2 for the borders (regularization)
    )
    ny = 10  # number of voxels along the y axis
    nz = 1
    dx = 5.0  # voxel dimension along the x axis
    dy = 5.0  # voxel dimension along the y axis
    dz = 1.0

    len_scale = 20.0  # m
    kappa = 1 / len_scale
    alpha = 1.0

    mean = 300.0  # trend of the field
    std = 150.0  # standard deviation of the field

    # Create a presison matrix
    Q_ref = spde.get_precision_matrix(
        nx, ny, nz, dx, dy, dz, kappa, alpha, spatial_dim=2, sigma=std
    )
    cholQ_ref = sparse_cholesky(Q_ref)

    n_fields = 50
    # 200 non conditional simulations
    tmp = []
    for i in range(n_fields):
        _field = np.abs(
            spde.simu_nc(cholQ_ref, random_state=i).reshape(ny, nx).T.reshape(ny, nx).T
            + mean
        )

        tmp.append(np.where(_field < 0.0, 0.0, _field).ravel("F"))
    X = np.array(tmp).T

    cov_mat = SparseInvCovarianceMatrix(Q_ref)

    # Test 1: With a null prior term
    reg1 = EnsembleRegularizator(cov_mat, NullPriorTerm())

    np.testing.assert_allclose(reg1.eval_loss(X), 184, rtol=0.01)

    grad = reg1.eval_loss_gradient_analytical(X)
    grad_fd = reg1.eval_loss_gradient(X, is_finite_differences=True)

    np.testing.assert_almost_equal(grad, grad_fd)

    # Test 2: With the mean computed from the ensemble
    reg2 = EnsembleRegularizator(cov_mat, EnsembleMeanPriorTerm(X.shape))

    # test the objective function
    np.testing.assert_allclose(reg2.eval_loss(X), 49.9953, rtol=0.01)

    grad = reg2.eval_loss_gradient_analytical(X)
    grad_fd = reg2.eval_loss_gradient(X, is_finite_differences=True)

    np.testing.assert_almost_equal(grad, grad_fd)
