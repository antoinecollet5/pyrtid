# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
pyRTID submodule providing tools and utilities for other submodules.

.. currentmodule:: pyrtid.utils

Working with dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for python dataclasses.

.. autosummary::
   :toctree: _autosummary

    default_field


.. currentmodule:: pyrtid.utils

Working string enums
^^^^^^^^^^^^^^^^^^^^

Provide a str enum class.u

.. autosummary::
   :toctree: _autosummary

    StrEnum

Types
^^^^^
Other functions

.. autosummary::
   :toctree: _autosummary

    NDArrayFloat
    NDArrayInt
    NDArrayBool
    Int
    object_or_object_sequence_to_list

.. currentmodule:: pyrtid.utils

"""

from scipy._lib._util import check_random_state  # To handle random_state

from pyrtid.utils.callbacks import Callback
from pyrtid.utils.dataclass import default_field
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.numpy_helpers import np_cache
from pyrtid.utils.types import (
    Int,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)

__all__ = [
    "Callback",
    "Int",
    "NDArrayBool",
    "NDArrayFloat",
    "NDArrayInt",
    "StrEnum",
    "check_random_state",
    "default_field",
    "np_cache",
    "object_or_object_sequence_to_list",
]
