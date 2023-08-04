from typing import Tuple

import numpy as np
import pytest

from pyrtid.inverse.regularization import (  # DriftMatrix,; LinearDriftMatrix,
    ConstantPriorTerm,
    DenseCovarianceMatrix,
    FFTCovarianceMatrix,
    GeostatisticalRegularizator,
    MeanPriorTerm,
    NullPriorTerm,
    cov_mat_to_ud_mat,
)
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
    return np.hstack((XX.ravel()[:, np.newaxis], YY.ravel()[:, np.newaxis]))


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
            DenseCovarianceMatrix(
                pts=get_pts(),
                kernel=exponential_kernel,
                len_scale=len_scale,
            ),
            1e-5,
        ),
        (
            cov_mat_to_ud_mat(
                DenseCovarianceMatrix(
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
            cov_mat_to_ud_mat(
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
    ],
)
def test_regularizator_gradients_by_fd(cov_mat, atol) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    regularizator = GeostatisticalRegularizator(cov_mat)

    print(f"loss_reg_dense = {regularizator.loss_function(param_values)}")

    grad_reg_fd = regularizator.loss_function_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.loss_function_gradient(param_values)
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

    cov_mat = cov_mat_to_ud_mat(
        FFTCovarianceMatrix(
            kernel=exponential_kernel,
            mesh_dim=get_mesh_dim(),
            domain_shape=get_domain_shape(),
            len_scale=len_scale,
            k=30,
        ),
        n_pc=32,
    )

    regularizator = GeostatisticalRegularizator(cov_mat, prior)

    print(f"loss_reg_dense = {regularizator.loss_function(param_values)}")

    grad_reg_fd = regularizator.loss_function_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.loss_function_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=1e-4)
