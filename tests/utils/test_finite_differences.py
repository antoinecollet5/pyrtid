# -*- coding: utf-8 -*-
import numdifftools as nd
import numpy as np
import pytest
from pyrtid.utils import finite_gradient, finite_jacobian, is_gradient_correct
from pyrtid.utils.finite_differences import rosen, rosen_gradient, rosen_hessian


@pytest.mark.parametrize(
    "values, max_workers",
    [([1.0, 1.0], 1), ([10.0, 10.0], 2), ([0.0, 79.9], 4), ([0.0, 0.0], 1)],
)
def test_gradient(values, max_workers) -> None:
    np.testing.assert_allclose(
        rosen_gradient(np.array(values)),
        finite_gradient(np.array(values), rosen, max_workers=max_workers),
        atol=1e-4,
    )
    assert is_gradient_correct(values, fm=rosen, grad=rosen_gradient, fm_args=())


@pytest.mark.parametrize("values", [([1.0, 1.0]), ([1.5, -0.5]), ([0.0, 1.9])])
def test_hessian(values) -> None:
    np.testing.assert_allclose(
        rosen_hessian(np.array(values)), nd.Hessian(rosen)(values), atol=1e-4
    )


def test_finite_difference_accuracy() -> None:
    with pytest.raises(ValueError, match="The accuracy should be 0, 1, 2 or 3!"):
        finite_gradient(np.array([1.0, 1.0]), rosen, accuracy=4)


def test_finite_jacobian() -> None:
    def fun2(x):
        return x[0] * x[1] * x[2] ** 2

    # numdifftools
    assert np.allclose(nd.Jacobian(fun2)([1.0, 2.0, 3.0]), [[18.0, 9.0, 12.0]])
    # our implementation
    np.testing.assert_almost_equal(
        finite_jacobian(np.array([1.0, 2.0, 3.0]), fun2), [18.0, 9.0, 12.0]
    )

    def fun3(x):
        return np.vstack((x[0] * x[1] * x[2] ** 2, x[0] * x[1] * x[2]))

    jfun3 = nd.Jacobian(fun3)

    # numdifftools
    assert np.allclose(
        jfun3([1.0, 2.0, 3.0]), [[[18.0], [9.0], [12.0]], [[6.0], [3.0], [2.0]]]
    )
    assert np.allclose(
        jfun3([4.0, 5.0, 6.0]), [[[180.0], [144.0], [240.0]], [[30.0], [24.0], [20.0]]]
    )
    assert np.allclose(
        jfun3(np.array([[1.0, 2.0, 3.0]]).T),
        [[[18.0], [9.0], [12.0]], [[6.0], [3.0], [2.0]]],
    )

    # our implementation -> note that the output shape is not the same...
    # (output shape, input shape)
    np.testing.assert_allclose(
        finite_jacobian(np.array([1.0, 2.0, 3.0]), fun3, fm_args=(), fm_kwargs={}),
        [[[18.0, 9.0, 12.0]], [[6.0, 3.0, 2.0]]],
    )
    np.testing.assert_allclose(
        finite_jacobian(np.array([4.0, 5.0, 6.0]), fun3),
        [[[180.0, 144.0, 240.0]], [[30.0, 24.0, 20.0]]],
    )
    np.testing.assert_allclose(
        finite_jacobian(np.array([[1.0, 2.0, 3.0]]).T, fun3),
        [[[[18.0], [9.0], [12.0]]], [[[6.0], [3.0], [2.0]]]],
    )
