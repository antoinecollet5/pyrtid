"""Test for operators utilities."""

import numpy as np
import pytest
from pyrtid.utils import (
    get_super_ilu_preconditioner,
    gradient_bfd,
    gradient_ffd,
    hessian_cfd,
)
from scipy.sparse import csc_array
from scipy.sparse.linalg import gmres


def test_get_super_ilu_preconditioner() -> None:
    A = csc_array([[1.0, 0.0, 0.0], [5.0, 0.0, 2.0], [0.0, -1.0, 0.0]], dtype=float)
    B = get_super_ilu_preconditioner(A)
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    np.testing.assert_allclose(
        gmres(A, x, M=B, atol=1e-15, callback_type="legacy")[0],
        gmres(A, x, atol=1e-15, callback_type="legacy")[0],
    )


def test_factor_excatly_singular() -> None:
    # all partial derivatives are zero
    A = csc_array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)
    B = get_super_ilu_preconditioner(A)
    assert B is None


@pytest.mark.parametrize(
    "shape,dx,axis,function",
    [
        ((20, 20), 5.0, 0, gradient_ffd),
        ((10, 10), 6.0, 1, gradient_ffd),
        ((20, 20), 5.0, 0, gradient_bfd),
        ((10, 10), 6.0, 1, gradient_bfd),
        ((20, 20), 5.0, 0, hessian_cfd),
        ((10, 10), 6.0, 1, hessian_cfd),
    ],
)  # type: ignore
def test_ones(shape, dx, axis, function) -> None:
    arr = np.ones(shape)
    assert function(arr, dx=dx, axis=axis).shape == shape
    np.testing.assert_allclose(function(arr, dx=dx, axis=axis), np.zeros(shape))


@pytest.mark.parametrize("function", [gradient_ffd, gradient_bfd, hessian_cfd])
def test_exceptions(function) -> None:
    arr = np.ones((20, 20))
    with pytest.raises(ValueError, match="axis should be 0 or 1 !"):
        function(arr, dx=5.0, axis=2)
