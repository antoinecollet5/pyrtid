# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provides helpers for numpy functions."""

from functools import lru_cache, wraps

import numpy as np


def np_cache(*args, pos: int = 0, **kwargs):
    """
    LRU cache implementation for functions whose FIRST parameter is a numpy array.

    Parameters
    ----------
    pos : int, optional
        Position of the numpy array argument in the decorated function signature. The
        default is 0.
    *args : Any
        Positional arguments forwarded to :func:`functools.lru_cache` (e.g.
        ``maxsize``).
    **kwargs : Any
        Keyword arguments forwarded to :func:`functools.lru_cache`.

    Returns
    -------
    Callable
        The decorator to apply to the function.

    Examples
    --------
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            args = list(args)  # to avoid tuple
            args[pos] = array_to_tuple(args[pos])
            return cached_wrapper(*args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(*args, **kwargs):
            args = list(args)  # to avoid tuple
            args[pos] = np.array(args[pos])
            return function(*args, **kwargs)

        def array_to_tuple(np_array):
            """
            Convert a (nested) array to a (nested) tuple, recursively.

            Parameters
            ----------
            np_array : Any
                Array to convert.

            Returns
            -------
            tuple or Any
                The (nested) tuple, or the input itself if it is not iterable.
            """
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info  # ty: ignore[unresolved-attribute]
        wrapper.cache_clear = cached_wrapper.cache_clear  # ty: ignore[unresolved-attribute]

        return wrapper

    return decorator
