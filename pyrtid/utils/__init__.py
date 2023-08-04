"""
pyRTID submodule providing tools and utilities for other submodules.

.. currentmodule:: pyrtid.utils.dataclass

Working with dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for python dataclasses.

.. autosummary::
   :toctree: _autosummary

    default_field


.. currentmodule:: pyrtid.utils.logging

Logging
^^^^^^^

Utilities for output display and logging.

.. autosummary::
   :toctree: _autosummary

   display_final_computation_time_message
   display_initial_message
   display_intermediate_message


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

    finite_gradient
    is_all_close
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

.. currentmodule:: pyrtid.utils.spatial_filters

Filters
^^^^^^^
Provide some spatial filters

.. autosummary::
   :toctree: _autosummary

    GaussianFilter


.. currentmodule:: pyrtid.utils.timer

Function timer
^^^^^^^^^^^^^^

Function to measure function time consumption.

.. autosummary::
   :toctree: _autosummary

    time_it

.. currentmodule:: pyrtid.utils

Others
^^^^^^

Other functions

.. autosummary::
   :toctree: _autosummary

    object_or_object_sequence_to_list
    extract_hess_inv_diag
    get_super_lu_preconditioner
    show_versions

Types
^^^^^
Other functions

.. autosummary::
   :toctree: _autosummary

    NDArrayFloat
    NDArrayInt
    NDArrayBool

"""
from pyrtid.utils.dataclass import default_field
from pyrtid.utils.decorator import ConditionalClassDecorator, conditional_decorator_func
from pyrtid.utils.enum import StrEnum
from pyrtid.utils.finite_differences import (
    finite_gradient,
    is_all_close,
    is_gradient_correct,
)
from pyrtid.utils.grid import (
    indices_to_node_number,
    node_number_to_indices,
    span_to_node_numbers_2d,
)
from pyrtid.utils.logging import (
    display_final_computation_time_message,
    display_initial_message,
    display_intermediate_message,
)
from pyrtid.utils.means import (
    arithmetic_mean,
    dxi_arithmetic_mean,
    dxi_harmonic_mean,
    harmonic_mean,
)
from pyrtid.utils.operators import (
    get_super_lu_preconditioner,
    gradient_bfd,
    gradient_ffd,
    hessian_cfd,
)
from pyrtid.utils.optimize import extract_hess_inv_diag
from pyrtid.utils.random_field import (
    gen_random_ensemble,
    get_log_normalized_mean_from_normal_params,
    get_log_normalized_std_from_normal_params,
    get_normalized_mean_from_lognormal_params,
    get_normalized_std_from_lognormal_params,
)
from pyrtid.utils.spatial_filters import GaussianFilter
from pyrtid.utils.timer import time_it
from pyrtid.utils.types import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)
from pyrtid.utils.versions import show_versions
from pyrtid.utils.wellfield import gen_wells_coordinates

__all__ = [
    "display_initial_message",
    "display_intermediate_message",
    "display_final_computation_time_message",
    "node_number_to_indices",
    "indices_to_node_number",
    "gradient_ffd",
    "gradient_bfd",
    "hessian_cfd",
    "GaussianFilter",
    "StrEnum",
    "finite_gradient",
    "is_gradient_correct",
    "is_all_close",
    "default_field",
    "gen_wells_coordinates",
    "get_super_lu_preconditioner",
    "time_it",
    "gen_random_ensemble",
    "get_normalized_mean_from_lognormal_params",
    "get_normalized_std_from_lognormal_params",
    "get_log_normalized_mean_from_normal_params",
    "get_log_normalized_std_from_normal_params",
    "ConditionalClassDecorator",
    "conditional_decorator_func",
    "arithmetic_mean",
    "dxi_arithmetic_mean",
    "harmonic_mean",
    "dxi_harmonic_mean",
    "object_or_object_sequence_to_list",
    "span_to_node_numbers_2d",
    "span_to_node_numbers_3d",
    "extract_hess_inv_diag",
    "NDArrayFloat",
    "NDArrayInt",
    "NDArrayBool",
    "show_versions",
]
