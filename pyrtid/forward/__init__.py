# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide the forward reactive transport model and solver.

Model
^^^^^

Class storing the data and the parameters fed to the solver.

.. currentmodule:: pyrtid.forward.models

.. autosummary::
   :toctree: _autosummary

    ForwardModel


Model Parameters
^^^^^^^^^^^^^^^^

Classes from which a :class:`ForwardModel` is built.

.. currentmodule:: pyrtid.forward.models

.. autosummary::
   :toctree: _autosummary

    TimeParameters
    FlowParameters
    TransportParameters
    GeochemicalParameters
    SourceTerm
    FlowRegime
    VerticalAxis

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

Available boundary conditions for the flow and the transport.

.. currentmodule:: pyrtid.forward.models

.. autosummary::
   :toctree: _autosummary

    ConstantHead
    ConstantConcentration
    ZeroConcGradient


Solver
^^^^^^

Class responsible to solve the reactive-transport problem. It does not hold any data
and performs the calculation on a :class:`ForwardModel`.

.. currentmodule:: pyrtid.forward

.. autosummary::
   :toctree: _autosummary

    ForwardSolver
    get_max_coupling_error
    solve_flow_stationary
    solve_flow_transient_semi_implicit
    solve_transport_semi_implicit
    solve_geochem_explicit
    solve_geochem_implicit

Functions
^^^^^^^^^

Some useful functions

.. currentmodule:: pyrtid.forward

.. autosummary::
   :toctree: _autosummary

    get_owner_neigh_indices

"""

from .flow_solver import solve_flow_stationary, solve_flow_transient_semi_implicit
from .geochem_solver import solve_geochem_explicit, solve_geochem_implicit
from .models import (
    ConstantConcentration,
    ConstantHead,
    FlowParameters,
    FlowRegime,
    ForwardModel,
    GeochemicalParameters,
    SourceTerm,
    TimeParameters,
    TransportParameters,
    VerticalAxis,
    ZeroConcGradient,
    get_owner_neigh_indices,
)
from .solver import ForwardSolver, get_max_coupling_error
from .transport_solver import solve_transport_semi_implicit

__all__ = [
    "TimeParameters",
    "FlowParameters",
    "TransportParameters",
    "GeochemicalParameters",
    "ForwardModel",
    "SourceTerm",
    "ForwardSolver",
    "ConstantHead",
    "ConstantConcentration",
    "ZeroConcGradient",
    "FlowRegime",
    "VerticalAxis",
    "get_max_coupling_error",
    "solve_flow_stationary",
    "solve_flow_transient_semi_implicit",
    "solve_transport_semi_implicit",
    "solve_geochem_explicit",
    "solve_geochem_implicit",
    "get_owner_neigh_indices",
]
