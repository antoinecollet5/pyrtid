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

.. currentmodule:: pyrtid.inverse.adjoint

Adjoint
^^^^^^^

Provide adjoint model and solver for the estimated parameter gradient computation.

.. autosummary::
   :toctree: _autosummary

    AdjointSolver
    AdjointModel
    is_adjoint_gradient_correct


Observables and utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    Observable
    ls_loss_function
    get_predictions_matching_observations
    get_observables_values_as_1d_vector
    get_observables_uncertainties_as_1d_vector

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
from pyrtid.inverse.gradient import is_adjoint_gradient_correct
from pyrtid.inverse.loss_function import ls_loss_function
from pyrtid.inverse.model import InverseModel
from pyrtid.inverse.obs import (
    Observable,
    StateVariable,
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
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
    "StateVariable",
    "get_predictions_matching_observations",
    "get_observables_values_as_1d_vector",
    "get_observables_uncertainties_as_1d_vector",
    "InverseModel",
    "AdjointSolver",
    "AdjointModel",
    "is_adjoint_gradient_correct",
    "get_backconditioned_adj_gradient",
    "get_backconditioned_fd_gradient",
    "get_gridded_archived_gradients",
]
