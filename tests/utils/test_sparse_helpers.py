import numpy as np
import pytest
import scipy as sp
from pyrtid.utils import assert_allclose_sparse, sparse_cholesky


def test_sparse_cholesky() -> None:
    L = np.array([[1.0, 0.0, 0.1], [0.1, 1.2, 0.0], [0.0, 0.0, 1.5]])
    A = L.T @ L

    F = sparse_cholesky(sp.sparse.csc_matrix(A))
    L2 = F.apply_P(F.L().tocsc())

    np.testing.assert_allclose((L2 @ L2.T).toarray(), A)


def test_assert_allclose_sparse() -> None:
    L = sp.sparse.csc_array([[1.0, 0.0, 0.1], [0.1, 1.2, 0.0], [0.0, 0.0, 1.5]])
    A = L.T @ L

    # works
    assert_allclose_sparse(A, A)
    assert_allclose_sparse(A.T, A)
    assert_allclose_sparse(L, L)

    # shapes are not the same
    with pytest.raises(AssertionError):
        assert_allclose_sparse(L, L.T)

    # Values are not the same
    with pytest.raises(AssertionError):
        assert_allclose_sparse(A, L)

    # shape is not the same
    with pytest.raises(AssertionError):
        assert_allclose_sparse(
            A, sp.sparse.csc_array([[1.0, 0.0, 0.1], [0.0, 0.0, 1.5]])
        )

    # empty
    assert_allclose_sparse(sp.sparse.csc_array([[]]), sp.sparse.csc_array([[]]))
