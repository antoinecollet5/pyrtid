# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide the inverse reactive transport model and solver as well as executors.

The following functionalities are directly provided on module-level.

.. currentmodule:: pyrtid.inverse

Classes
=======

Inversion executors
^^^^^^^^^^^^^^^^^^^

Different executors are provided (scipy, stochopy, pyesmda, pypcga).

.. autosummary::
   :toctree: _autosummary

    executors

.. currentmodule:: pyrtid.inverse

Adjoint
^^^^^^^

Sub module providing an adjoint operator and the associated gradients.

.. autosummary::
   :toctree: _autosummary

    asm

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
    eval_model_loss_ls
    update_perturbation_values

.. currentmodule:: pyrtid.inverse

Loss functions
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    eval_loss_ls
    eval_model_loss_function
    eval_model_loss_ls
    get_theoretical_noise_level

"""

from pyesmda import ESMDAInversionType

from pyrtid.inverse.asm.amain_solver import AdjointSolver
from pyrtid.inverse.asm.amodels import AdjointModel
from pyrtid.inverse.executors import (
    ESMDADMCInversionExecutor,
    ESMDADMCSolverConfig,
    ESMDAInversionExecutor,
    ESMDARSInversionExecutor,
    ESMDARSSolverConfig,
    ESMDASolverConfig,
    LBFGSBInversionExecutor,
    LBFGSBSolverConfig,
    PCGAInversionExecutor,
    PCGASolverConfig,
    ScipyInversionExecutor,
    ScipySolverConfig,
    StochopyInversionExecutor,
    StochopySolverConfig,
)
from pyrtid.inverse.loss_function import (
    eval_loss_ls,
    eval_model_loss_function,
    eval_model_loss_ls,
    get_theoretical_noise_level,
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
    update_perturbation_values,
)
from pyrtid.inverse.params import (
    AdjustableParameter,
    ParameterName,
    get_backconditioned_adj_gradient,
    get_backconditioned_fd_gradient,
    get_gridded_archived_gradients,
    get_parameters_bounds,
    get_parameters_values_from_model,
    update_model_with_parameters_values,
)

__all__ = [
    "AdjointModel",
    "AdjointSolver",
    "AdjustableParameter",
    "ESMDADMCInversionExecutor",
    "ESMDADMCSolverConfig",
    "ESMDAInversionExecutor",
    "ESMDAInversionType",
    "ESMDARSInversionExecutor",
    "ESMDARSSolverConfig",
    "ESMDASolverConfig",
    "InverseModel",
    "LBFGSBInversionExecutor",
    "LBFGSBSolverConfig",
    "Observable",
    "Observables",
    "PCGAInversionExecutor",
    "PCGASolverConfig",
    "ParameterName",
    "ScipyInversionExecutor",
    "ScipySolverConfig",
    "StateVariable",
    "StochopyInversionExecutor",
    "StochopySolverConfig",
    "eval_loss_ls",
    "eval_model_loss_function",
    "eval_model_loss_ls",
    "get_adjoint_sources_for_obs",
    "get_backconditioned_adj_gradient",
    "get_backconditioned_fd_gradient",
    "get_gridded_archived_gradients",
    "get_observables_uncertainties_as_1d_vector",
    "get_observables_values_as_1d_vector",
    "get_parameters_bounds",
    "get_parameters_values_from_model",
    "get_predictions_matching_observations",
    "get_sorted_observable_times",
    "get_sorted_observable_uncertainties",
    "get_sorted_observable_values",
    "get_theoretical_noise_level",
    "get_values_matching_node_indices",
    "update_model_with_parameters_values",
    "update_perturbation_values",
]
