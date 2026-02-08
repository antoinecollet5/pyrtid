"""Wrap the sparse cholesky factorization from"""

from typing import Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, find

try:
    from sksparse.cholmod import Factor as SparseFactor
    from sksparse.cholmod import cholesky
except (ImportError, ModuleNotFoundError):
    # warnings.warn(
    #     "scikit-sparse could not be loaded. Consequently"
    #     ", hytecio-inverse functionalities are limited."
    # )

    class SparseFactor:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "sksparse could not be loaded. Please "
                " install it to use SparseFactor (see "
                "https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation"
                ")"
            )

    def cholesky(mat: csc_matrix) -> SparseFactor:
        raise ModuleNotFoundError(
            "sksparse could not be loaded. Please"
            " install it to use sparse_cholesky (see "
            "https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation"
            ")"
        )


def sparse_cholesky(arr: Union[csc_matrix, csc_array]) -> SparseFactor:
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
