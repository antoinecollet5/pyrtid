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

    Geometry
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

.. currentmodule:: pyrtid.forward.solver

.. autosummary::
   :toctree: _autosummary

    ForwardSolver

"""

from .models import (
    ConstantConcentration,
    ConstantHead,
    FlowParameters,
    FlowRegime,
    ForwardModel,
    GeochemicalParameters,
    Geometry,
    SourceTerm,
    TimeParameters,
    TransportParameters,
    VerticalAxis,
    ZeroConcGradient,
)
from .solver import ForwardSolver

__all__ = [
    "Geometry",
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
]
