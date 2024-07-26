"""Wrap the sparse cholesky factorization from"""

from typing import Union

from scipy.sparse import csc_array, csc_matrix
from sksparse.cholmod import Factor, cholesky


def sparse_cholesky(arr: Union[csc_matrix, csc_array]) -> Factor:
    # see: https://github.com/scikit-sparse/scikit-sparse/issues/108
    # see: https://github.com/scikit-sparse/scikit-sparse/pull/102
    return cholesky(csc_matrix(arr))
