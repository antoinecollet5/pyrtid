"""
PyRTID invsere sub module providing an adjoint operator.

The following functionalities are directly provided on module-level.

.. currentmodule:: pyrtid.inverse.adjoint

Models
======

.. currentmodule:: pyrtid.inverse.adjoint.amodels

Classes holding the adjoint variables.

.. autosummary::
   :toctree: _autosummary

    AdjointModel
    AdjointFlowModel
    SaturatedAdjointFlowModel
    DensityAdjointFlowModel
    AdjointTransportModel

Gradients of cost function with respect to control parameters
=============================================================
Functions to derive and check the gradients form the forward and adjoint models.

.. currentmodule:: pyrtid.inverse.adjoint.gradients

Computing gradients
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    get_diffusion_adjoint_gradient
    get_initial_conc_adjoint_gradient
    get_initial_grade_adjoint_gradient
    get_permeability_adjoint_gradient
    get_porosity_adjoint_gradient

Checking gradients
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    is_adjoint_gradient_correct


Solvers
=======

Full adjoint
^^^^^^^^^^^^

.. currentmodule:: pyrtid.inverse.adjoint.amain_solver

Class from which to derive regularizator implementations.

.. autosummary::
   :toctree: _autosummary

   AdjointSolver


Adjoint flow
^^^^^^^^^^^^

.. currentmodule:: pyrtid.inverse.adjoint.aflow_solver

Functions to solve the adjoint flow.

.. autosummary::
   :toctree: _autosummary

    make_transient_adj_flow_matrices
    solve_adj_flow
    update_adjoint_u_darcy

Adjoint transport
^^^^^^^^^^^^^^^^^

.. currentmodule:: pyrtid.inverse.adjoint.atransport_solver

Functions to solve the adjoint flow.

.. autosummary::
   :toctree: _autosummary

    get_adjoint_max_coupling_error
    solve_adj_transport_transient_semi_implicit


Adjoint chemistry
^^^^^^^^^^^^^^^^^

.. currentmodule:: pyrtid.inverse.adjoint.ageochem_solver

.. autosummary::
   :toctree: _autosummary

    solve_adj_geochem

"""

from pyrtid.inverse.adjoint.aflow_solver import (
    make_transient_adj_flow_matrices,
    solve_adj_flow,
    update_adjoint_u_darcy,
)
from pyrtid.inverse.adjoint.ageochem_solver import solve_adj_geochem
from pyrtid.inverse.adjoint.amain_solver import AdjointSolver
from pyrtid.inverse.adjoint.amodels import (
    AdjointFlowModel,
    AdjointModel,
    AdjointTransportModel,
    DensityAdjointFlowModel,
    SaturatedAdjointFlowModel,
)
from pyrtid.inverse.adjoint.atransport_solver import (
    get_adjoint_max_coupling_error,
    solve_adj_transport_transient_semi_implicit,
)
from pyrtid.inverse.adjoint.gradients import (
    get_diffusion_adjoint_gradient,
    get_initial_conc_adjoint_gradient,
    get_initial_grade_adjoint_gradient,
    get_permeability_adjoint_gradient,
    get_porosity_adjoint_gradient,
    is_adjoint_gradient_correct,
)

__all__ = [
    "AdjointSolver",
    "AdjointFlowModel",
    "SaturatedAdjointFlowModel",
    "DensityAdjointFlowModel",
    "AdjointModel",
    "AdjointTransportModel",
    "make_transient_adj_flow_matrices",
    "solve_adj_flow",
    "update_adjoint_u_darcy",
    "get_adjoint_max_coupling_error",
    "solve_adj_transport_transient_semi_implicit",
    "solve_adj_geochem",
    "get_diffusion_adjoint_gradient",
    "get_initial_conc_adjoint_gradient",
    "get_initial_grade_adjoint_gradient",
    "get_permeability_adjoint_gradient",
    "get_porosity_adjoint_gradient",
    "is_adjoint_gradient_correct",
]
