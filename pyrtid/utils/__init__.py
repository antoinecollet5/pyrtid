# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
pyRTID submodule providing tools and utilities for other submodules.

.. currentmodule:: pyrtid.utils.dataclass

Working with dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for python dataclasses.

.. autosummary::
   :toctree: _autosummary

    default_field


.. currentmodule:: pyrtid.utils.enum

Working string enums
^^^^^^^^^^^^^^^^^^^^

Provide a str enum class.u

.. autosummary::
   :toctree: _autosummary

    StrEnum

.. currentmodule:: pyrtid.utils

Others
^^^^^^

Other functions

.. autosummary::
   :toctree: _autosummary

    get_super_ilu_preconditioner
    check_random_state

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
    "StrEnum",
    "default_field",
    "get_super_ilu_preconditioner",
    "gen_random_ensemble",
    "get_normalized_mean_from_lognormal_params",
    "get_normalized_std_from_lognormal_params",
    "get_log_normalized_mean_from_normal_params",
    "get_log_normalized_std_from_normal_params",
    "dxi_arithmetic_mean",
    "harmonic_mean",
    "dxi_harmonic_mean",
    "MeanType",
    "get_mean_values_for_last_axis",
    "amean_gradient",
    "gmean_gradient",
    "hmean_gradient",
    "get_mean_values_gradient_for_last_axis",
    "object_or_object_sequence_to_list",
    "span_to_node_numbers_2d",
    "span_to_node_numbers_3d",
    "get_pts_coords_regular_grid",
    "NDArrayFloat",
    "NDArrayInt",
    "NDArrayBool",
    "Int",
    "create_selections_array_2d",
    "get_polygon_selection_with_dilation_2d",
    "Callback",
    "np_cache",
    "check_random_state",
]
