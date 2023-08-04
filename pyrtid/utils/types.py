"""Provide utils to work with list."""

from collections.abc import Iterable
from typing import List, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float_]
NDArrayInt = npt.NDArray[np.int_]
NDArrayBool = npt.NDArray[np.bool_]

_Object = TypeVar("_Object", bound=object)


def object_or_object_sequence_to_list(
    _input: Union[_Object, Sequence[_Object]]
) -> List[_Object]:
    """Convert a singleton or an iterable of this object to a list of object."""
    if isinstance(_input, Iterable):
        return [item for item in _input]
    return [_input]
