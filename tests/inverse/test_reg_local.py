"""Tests for the regularizator classes."""

import numpy as np
import pytest

from pyrtid.inverse.regularization import (
    TikhonovRegularizatorAnisotropic,
    TikhonovRegularizatorIsotropic,
    TVRegularizatorAnisotropic,
    TVRegularizatorIsotropic,
)
from pyrtid.utils.types import NDArrayFloat


def get_param_values() -> NDArrayFloat:
    """Generate a parameter field with some noise."""
    nx: int = 15
    ny: int = 26
    param: NDArrayFloat = np.zeros((nx, ny), dtype=np.float64)
    param[0:10, 5:15] = 5.0
    param[6:14, 7:14] = 10.0
    param[8:9, 2:25] = 20.0

    # Add some noise with a seed
    rng = np.random.default_rng(26659)
    param += rng.random((nx, ny)) * 5.0

    return param


@pytest.mark.parametrize(
    "regularizator",
    [
        TikhonovRegularizatorAnisotropic(7.5, axis=0),
        TikhonovRegularizatorAnisotropic(3.6, axis=1),
        TikhonovRegularizatorIsotropic(3.6, 7.5),
        TVRegularizatorAnisotropic(7.5, axis=0),
        TVRegularizatorAnisotropic(3.6, axis=1),
        TVRegularizatorIsotropic(3.6, 7.5),
    ],
)
def test_regularizator_gradients_by_fd(regularizator) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    grad_reg_fd = regularizator.loss_function_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.loss_function_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=1e-5)
