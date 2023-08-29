"""
Provide the inverse reactive transport model and solver as well as executors.

The following functionalities are directly provided on module-level.

.. currentmodule:: pyrtid.inverse.executors

Classes
=======

Inversion executors
^^^^^^^^^^^^^^^^^^^

Different executors are provided (scipy, stochopy, pyesmda, pypcga).

.. autosummary::
   :toctree: _autosummary

    ESMDAInversionExecutor
    ESMDASolverConfig
    ESMDARSInversionExecutor
    ESMDARSSolverConfig
    PCGAInversionExecutor
    PCGASolverConfig
    ScipyInversionExecutor
    ScipySolverConfig
    StochopyInversionExecutor
    StochopySolverConfig

.. currentmodule:: pyrtid.inverse

Regularization
^^^^^^^^^^^^^^

Sub module providing regularization tools.

.. autosummary::
   :toctree: _autosummary

    regularization

.. currentmodule:: pyrtid.inverse

Adjoint
^^^^^^^

Sub module providing an adjoint opertaor and the associated gradients.

.. autosummary::
   :toctree: _autosummary

    adjoint

.. currentmodule:: pyrtid.inverse

Observables and utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    Observable
    Observables
    StateVariable
    get_observables_uncertainties_as_1d_vector
    get_observables_values_as_1d_vector
    get_predictions_matching_observations
    get_sorted_observable_times
    get_sorted_observable_uncertainties
    get_sorted_observable_values
    get_values_matching_node_indices
    get_adjoint_sources_for_obs
    get_model_ls_loss_function

"""

from pyrtid.inverse.adjoint.amain_solver import AdjointSolver
from pyrtid.inverse.adjoint.amodels import AdjointModel
from pyrtid.inverse.executors import (
    ESMDAInversionExecutor,
    ESMDARSInversionExecutor,
    ESMDARSSolverConfig,
    ESMDASolverConfig,
    PCGAInversionExecutor,
    PCGASolverConfig,
    ScipyInversionExecutor,
    ScipySolverConfig,
    StochopyInversionExecutor,
    StochopySolverConfig,
)
from pyrtid.inverse.loss_function import (
    get_model_loss_function,
    get_model_ls_loss_function,
    get_model_reg_loss_function,
    ls_loss_function,
)
from pyrtid.inverse.model import InverseModel
from pyrtid.inverse.obs import (
    Observable,
    Observables,
    StateVariable,
    get_adjoint_sources_for_obs,
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
    get_sorted_observable_times,
    get_sorted_observable_uncertainties,
    get_sorted_observable_values,
    get_values_matching_node_indices,
)
from pyrtid.inverse.params import (
    AdjustableParameter,
    ParameterName,
    get_1st_derivative_preconditoned_parameters_values_from_model,
    get_backconditioned_adj_gradient,
    get_backconditioned_fd_gradient,
    get_gridded_archived_gradients,
    get_parameters_bounds,
    get_parameters_values_from_model,
    update_model_with_parameters_values,
)

__all__ = [
    "ScipySolverConfig",
    "ESMDASolverConfig",
    "ESMDARSSolverConfig",
    "PCGASolverConfig",
    "StochopySolverConfig",
    "ScipyInversionExecutor",
    "ScipyInversionExecutor",
    "StochopyInversionExecutor",
    "PCGAInversionExecutor",
    "ESMDAInversionExecutor",
    "ESMDARSInversionExecutor",
    "AdjustableParameter",
    "ParameterName",
    "ls_loss_function",
    "get_parameters_values_from_model",
    "get_1st_derivative_preconditoned_parameters_values_from_model",
    "update_model_with_parameters_values",
    "get_parameters_bounds",
    "Observable",
    "Observables",
    "StateVariable",
    "get_sorted_observable_times",
    "get_sorted_observable_values",
    "get_sorted_observable_uncertainties",
    "get_predictions_matching_observations",
    "get_observables_values_as_1d_vector",
    "get_observables_uncertainties_as_1d_vector",
    "get_values_matching_node_indices",
    "get_adjoint_sources_for_obs",
    "InverseModel",
    "AdjointSolver",
    "AdjointModel",
    "get_backconditioned_adj_gradient",
    "get_backconditioned_fd_gradient",
    "get_gridded_archived_gradients",
    "get_model_ls_loss_function",
    "get_model_reg_loss_function",
    "get_model_loss_function",
]
