"""Wrap the sparse cholesky factorization from"""

from typing import Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, find
from sksparse.cholmod import Factor, cholesky


def sparse_cholesky(arr: Union[csc_matrix, csc_array]) -> Factor:
    # see: https://github.com/scikit-sparse/scikit-sparse/issues/108
    # see: https://github.com/scikit-sparse/scikit-sparse/pull/102
    return cholesky(csc_matrix(arr))


def assert_allclose_sparse(A, B, atol=1e-8, rtol=1e-8) -> None:
    """Assert that two sparse matrices or arrays are almost equal."""
    # If you want to check matrix shapes as well
    assert np.array_equal(A.shape, B.shape)
    r1, c1, v1 = find(A)
    r2, c2, v2 = find(B)
    np.testing.assert_equal(r1, r2)
    np.testing.assert_equal(c1, c2)
    np.testing.assert_allclose(v1, v2, atol=atol, rtol=rtol)
