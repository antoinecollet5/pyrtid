# -*- coding: utf-8 -*-
import numdifftools as nd
import numpy as np
import pytest

from pyrtid.utils import finite_gradient, is_gradient_correct
from pyrtid.utils.finite_differences import rosen, rosen_gradient, rosen_hessian


@pytest.mark.parametrize(
    "values, max_workers", [([1.0, 1.0], 1), ([10.0, 10.0], 2), ([0.0, 79.9], 4)]
)
def test_gradient(values, max_workers) -> None:
    np.testing.assert_allclose(
        rosen_gradient(np.array(values)),
        finite_gradient(np.array(values), rosen, max_workers=max_workers),
        atol=1e-4,
    )
    assert is_gradient_correct(values, fm=rosen, grad=rosen_gradient)


@pytest.mark.parametrize("values", [([1.0, 1.0]), ([1.5, -0.5]), ([0.0, 1.9])])
def test_hessian(values) -> None:
    np.testing.assert_allclose(
        rosen_hessian(np.array(values)), nd.Hessian(rosen)(values), atol=1e-4
    )


def test_finite_difference_accuracy() -> None:
    with pytest.raises(ValueError, match="The accuracy should be 0, 1, 2 or 3!"):
        finite_gradient(np.array([1.0, 1.0]), rosen, accuracy=4)
