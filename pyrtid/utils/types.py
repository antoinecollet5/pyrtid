# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide utils to work with list."""

from collections.abc import Iterable, Sequence
from typing import Any as _Any
from typing import TypeVar

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.floating]
NDArrayInt = npt.NDArray[np.integer]
NDArrayBool = npt.NDArray[np.bool_]
Int = int | NDArrayInt | Sequence[int]

_Object = TypeVar("_Object", bound=object)


def object_or_object_sequence_to_list(
    _input: _Object | Iterable[_Object],
) -> list[_Any]:
    """
    Convert a singleton or an iterable of this object to a list of object.

    The return type is loose on purpose: type checkers infer the element type as the
    union of the singleton and iterable types when a union is given.
    """
    if isinstance(_input, Iterable):
        return list(_input)
    return [_input]
