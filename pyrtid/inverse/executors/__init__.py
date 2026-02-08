# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide interfaces to various inverse problem solvers.

The following functionalities are directly provided on module-level.

.. currentmodule:: pyrtid.inverse.executors

Classes
=======

Inversion executors
^^^^^^^^^^^^^^^^^^^

Different executors are provided (scipy, stochopy, pyesmda, pypcga, l-bfgs-b, sies).

.. autosummary::
   :toctree: _autosummary

    ESMDAInversionExecutor
    ESMDASolverConfig
    ESMDARSInversionExecutor
    ESMDARSSolverConfig
    ESMDADMCInversionExecutor
    ESMDADMCSolverConfig
    LBFGSBInversionExecutor
    LBFGSBSolverConfig
    PCGAInversionExecutor
    PCGASolverConfig
    ScipyInversionExecutor
    ScipySolverConfig
    StochopyInversionExecutor
    StochopySolverConfig
    SIESInversionExecutor
    SIESSolverConfig

Other classes
^^^^^^^^^^^^^

Classes to defined inner inversion strategies when using ensemble methods.

    SIESInversionType
    ESMDAInversionType

"""

from pyesmda import ESMDAInversionType

from pyrtid.inverse.executors.esmda import (
    ESMDADMCInversionExecutor,
    ESMDADMCSolverConfig,
    ESMDAInversionExecutor,
    ESMDARSInversionExecutor,
    ESMDARSSolverConfig,
    ESMDASolverConfig,
)
from pyrtid.inverse.executors.lbfgsb import (
    LBFGSBInversionExecutor,
    LBFGSBSolverConfig,
)
from pyrtid.inverse.executors.pcga import PCGAInversionExecutor, PCGASolverConfig
from pyrtid.inverse.executors.scipy import ScipyInversionExecutor, ScipySolverConfig
from pyrtid.inverse.executors.sies import (
    SIESInversionExecutor,
    SIESInversionType,
    SIESSolverConfig,
)
from pyrtid.inverse.executors.stochopy import (
    StochopyInversionExecutor,
    StochopySolverConfig,
)

__all__ = [
    "ESMDAInversionType",
    "ESMDAInversionExecutor",
    "ESMDASolverConfig",
    "ESMDARSInversionExecutor",
    "ESMDARSSolverConfig",
    "ESMDADMCInversionExecutor",
    "ESMDADMCSolverConfig",
    "LBFGSBInversionExecutor",
    "LBFGSBSolverConfig",
    "PCGAInversionExecutor",
    "PCGASolverConfig",
    "ScipyInversionExecutor",
    "ScipySolverConfig",
    "StochopyInversionExecutor",
    "StochopySolverConfig",
    "SIESInversionExecutor",
    "SIESSolverConfig",
    "SIESInversionType",
]
