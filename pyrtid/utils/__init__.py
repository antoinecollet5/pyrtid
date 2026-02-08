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


.. currentmodule:: pyrtid.utils.grid

Regular grids
^^^^^^^^^^^^^

Provide utilities to work with regular grids.

.. autosummary::
   :toctree: _autosummary

    indices_to_node_number
    node_number_to_indices
    span_to_node_numbers_2d
    span_to_node_numbers_3d
    get_array_borders_selection_2d
    get_array_borders_selection_3d
    get_a_not_in_b_1d
    get_pts_coords_regular_grid
    create_selections_array_2d
    RectilinearGrid
    get_polygon_selection_with_dilation_2d
    get_extended_grid_shape


.. currentmodule:: pyrtid.utils.wellfield

WellField
^^^^^^^^^

Utilities to create wellfields.

.. autosummary::
   :toctree: _autosummary

    gen_wells_coordinates

.. currentmodule:: pyrtid.utils.random_field

Random fields
^^^^^^^^^^^^^
Provide utilities to generate random fields. This is based on the
`GStools package
<https://geostat-framework.readthedocs.io/projects/gstools/en/stable/index.html>`_.

.. autosummary::
   :toctree: _autosummary

    gen_random_ensemble
    get_normalized_mean_from_lognormal_params
    get_normalized_std_from_lognormal_params
    get_log_normalized_mean_from_normal_params
    get_log_normalized_std_from_normal_params

.. currentmodule:: pyrtid.utils.enum

Working string enums
^^^^^^^^^^^^^^^^^^^^

Provide a str enum class.u

.. autosummary::
   :toctree: _autosummary

    StrEnum


.. currentmodule:: pyrtid.utils.finite_differences

Numerical approximation by finite differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide functions to compute the gradient of a function by
finite difference numerical approximation.

.. autosummary::
   :toctree: _autosummary


    finite_jacobian
    finite_gradient
    is_all_close
    is_jacobian_correct
    is_gradient_correct


.. currentmodule:: pyrtid.utils.operators

Spatial differential operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Provide functions for spatial differentiation.

.. autosummary::
   :toctree: _autosummary

   gradient_ffd
   gradient_bfd
   hessian_cfd

.. currentmodule:: pyrtid.utils.means

Mean operators
^^^^^^^^^^^^^^
Provide functions to perform mean and their first derivative.

.. autosummary::
   :toctree: _autosummary

    arithmetic_mean
    dxi_arithmetic_mean
    harmonic_mean
    dxi_harmonic_mean
    MeanType
    get_mean_values_for_last_axis
    amean_gradient
    gmean_gradient
    hmean_gradient
    get_mean_values_gradient_for_last_axis

.. currentmodule:: pyrtid.utils.spatial_filters

Filters
^^^^^^^
Provide some spatial filters

.. autosummary::
   :toctree: _autosummary

    Filter
    GaussianFilter

.. currentmodule:: pyrtid.utils

Others
^^^^^^

Other functions

.. autosummary::
   :toctree: _autosummary

    object_or_object_sequence_to_list
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

Sparse helpers
^^^^^^^^^^^^^^

Helpers to work with sparse matrices and sparse arrays.

.. autosummary::
   :toctree: _autosummary

    SparseFactor
    sparse_cholesky
    assert_allclose_sparse

Preconditioners
^^^^^^^^^^^^^^^

Sub module providing preconditioners and parametrization tools.

.. autosummary::
   :toctree: _autosummary

    preconditioner

.. currentmodule:: pyrtid.utils

"""

from scipy._lib._util import check_random_state  # To handle random_state

from pyrtid.utils.callbacks import Callback
from pyrtid.utils.dataclass import default_field
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.finite_differences import (
    finite_gradient,
    finite_jacobian,
    is_all_close,
    is_gradient_correct,
    is_jacobian_correct,
)
from pyrtid.utils.grid import (
    RectilinearGrid,
    create_selections_array_2d,
    get_a_not_in_b_1d,
    get_array_borders_selection_2d,
    get_array_borders_selection_3d,
    get_extended_grid_shape,
    get_polygon_selection_with_dilation_2d,
    get_pts_coords_regular_grid,
    indices_to_node_number,
    node_number_to_indices,
    span_to_node_numbers_2d,
    span_to_node_numbers_3d,
)
from pyrtid.utils.means import (
    MeanType,
    amean_gradient,
    arithmetic_mean,
    dxi_arithmetic_mean,
    dxi_harmonic_mean,
    get_mean_values_for_last_axis,
    get_mean_values_gradient_for_last_axis,
    gmean_gradient,
    harmonic_mean,
    hmean_gradient,
)
from pyrtid.utils.numpy_helpers import np_cache
from pyrtid.utils.operators import (
    get_angle_btw_vectors_deg,
    get_angle_btw_vectors_rad,
    get_super_ilu_preconditioner,
    gradient_bfd,
    gradient_ffd,
    hessian_cfd,
)
from pyrtid.utils.random_field import (
    gen_random_ensemble,
    get_log_normalized_mean_from_normal_params,
    get_log_normalized_std_from_normal_params,
    get_normalized_mean_from_lognormal_params,
    get_normalized_std_from_lognormal_params,
)
from pyrtid.utils.sparse_helpers import (
    SparseFactor,
    assert_allclose_sparse,
    sparse_cholesky,
)
from pyrtid.utils.spatial_filters import Filter, GaussianFilter
from pyrtid.utils.types import (
    Int,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)
from pyrtid.utils.wellfield import gen_wells_coordinates

__all__ = [
    "node_number_to_indices",
    "indices_to_node_number",
    "gradient_ffd",
    "gradient_bfd",
    "hessian_cfd",
    "get_angle_btw_vectors_deg",
    "get_angle_btw_vectors_rad",
    "Filter",
    "GaussianFilter",
    "StrEnum",
    "finite_jacobian",
    "finite_gradient",
    "is_all_close",
    "is_jacobian_correct",
    "is_gradient_correct",
    "default_field",
    "gen_wells_coordinates",
    "get_super_ilu_preconditioner",
    "gen_random_ensemble",
    "get_normalized_mean_from_lognormal_params",
    "get_normalized_std_from_lognormal_params",
    "get_log_normalized_mean_from_normal_params",
    "get_log_normalized_std_from_normal_params",
    "arithmetic_mean",
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
    "get_array_borders_selection_2d",
    "get_array_borders_selection_3d",
    "get_pts_coords_regular_grid",
    "NDArrayFloat",
    "NDArrayInt",
    "NDArrayBool",
    "Int",
    "get_a_not_in_b_1d",
    "create_selections_array_2d",
    "RectilinearGrid",
    "get_extended_grid_shape",
    "get_polygon_selection_with_dilation_2d",
    "sparse_cholesky",
    "assert_allclose_sparse",
    "Callback",
    "np_cache",
    "check_random_state",
    "SparseFactor",
]
