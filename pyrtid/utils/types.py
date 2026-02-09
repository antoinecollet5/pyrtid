# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide utils to work with list."""

from collections.abc import Iterable
from typing import List, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]
NDArrayBool = npt.NDArray[np.bool_]
Int = Union[int, NDArrayInt, Sequence[int]]

_Object = TypeVar("_Object", bound=object)


def object_or_object_sequence_to_list(
    _input: Union[_Object, Iterable[_Object]],
) -> List[_Object]:
    """Convert a singleton or an iterable of this object to a list of object."""
    if isinstance(_input, Iterable):
        return list(_input)
    return [_input]
