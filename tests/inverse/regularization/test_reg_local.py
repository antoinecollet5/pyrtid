"""Tests for the regularizator classes."""

import numpy as np
import pyrtid.inverse as dminv
import pytest
from pyrtid.regularization import (
    DiscreteRegularizator,
    TikhonovFVMRegularizator,
    TikhonovMatRegularizator,
    TikhonovRegularizator,
    TVFVMRegularizator,
    TVMatRegularizator,
    TVRegularizator,
)
from pyrtid.utils import NDArrayFloat, RectilinearGrid


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

    return param.ravel("F")


def test_discrete_exceptions() -> None:
    for modes in [[], [1.0]]:
        with pytest.raises(ValueError, match="At least two modes must be provided!"):
            DiscreteRegularizator(modes=modes)

    with pytest.raises(
        ValueError, match=r'penalty should be among \["least-squares", "gaussian"\]'
    ):
        DiscreteRegularizator(modes=[1, 20], penalty="Anything")

    with pytest.raises(
        ValueError, match=r'penalty should be among \["least-squares", "gaussian"\]'
    ):
        instance = DiscreteRegularizator(modes=[1, 20], penalty="gaussian")
        instance.penalty = "not valid"


@pytest.mark.parametrize(
    "regularizator",
    [
        TikhonovRegularizator(RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TikhonovMatRegularizator(RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TikhonovFVMRegularizator(RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TVRegularizator(RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TVMatRegularizator(RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        TVFVMRegularizator(RectilinearGrid(dx=3.6, dy=7.5, nx=15, ny=26)),
        DiscreteRegularizator(modes=[7.0, 15.0], penalty="gaussian"),
        DiscreteRegularizator(modes=[7.0, 8.5, 2.3, 15.0], penalty="gaussian"),
        DiscreteRegularizator(
            modes=[7.0, 8.5, 2.3, 15.0],
            penalty="gaussian",
            preconditioner=dminv.LogTransform(),
        ),
        DiscreteRegularizator(modes=[2.3, 15.0], penalty="least-squares"),
        DiscreteRegularizator(modes=[7.0, 8.5, 2.3, 15.0], penalty="least-squares"),
        DiscreteRegularizator(
            modes=[7.0, 8.5, 2.3, 15.0],
            penalty="least-squares",
            preconditioner=dminv.LogTransform(),
        ),
    ],
)
def test_regularizator_gradients_by_fd(regularizator) -> None:
    """Test the correctness of the gradients by finite differences."""
    param_values = get_param_values()

    grad_reg_fd = regularizator.eval_loss_gradient(
        param_values, is_finite_differences=True
    )
    grad_reg_analytic = regularizator.eval_loss_gradient(param_values)
    np.testing.assert_allclose(grad_reg_fd, grad_reg_analytic, atol=1e-5)
