# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
PyRTID invsere sub module providing regularization tools.

The following functionalities are directly provided on module-level.

.. currentmodule:: pyrtid.regularization

Abstract classes
================

Base class from which to derive regularizator implementations.

.. autosummary::
   :toctree: _autosummary

   Regularizator

Local
=====

Tikhonov (for smooth spatial distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    TikhonovRegularizator
    TikhonovMatRegularizator
    TikhonovFVMRegularizator

Total Variation (for blocky spatial distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    TVRegularizator
    TVMatRegularizator
    TVFVMRegularizator

Discrete to impose specific discrete values to the field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    DiscreteRegularizator

Global
======

Fitting empirical distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    ProbDistFitting

Geostatistic regularizator
^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide classes to implement regularization based on a parameter covariance matrix.
The first one work with a single vector while the second class works with
an ensemble of realizations.

.. autosummary::
   :toctree: _autosummary

    GeostatisticalRegularizator
    EnsembleRegularizator

Matrix compression
^^^^^^^^^^^^^^^^^^^

Eigen decomposition

.. autosummary::
   :toctree: _autosummary

    get_matrix_eigen_factorization
    eigen_factorize_cov_mat

Regularization weights selection
================================

Strategies for the weight evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Class to indicate what strategy to use to weight the objective
function regularization term.

.. autosummary::
   :toctree: _autosummary

    RegWeightUpdateStrategy
    AdaptiveUCRegweight
    AdaptiveRegweight
    AdaptiveGradientNormRegweight
    ConstantRegWeight

Curvature in the context of L-curve plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate curvature of a L-curve

.. autosummary::
   :toctree: _autosummary

    get_l_curvature

"""

from pyrtid.regularization.adaptive import (
    AdaptiveGradientNormRegweight,
    AdaptiveRegweight,
    AdaptiveUCRegweight,
)
from pyrtid.regularization.base import (
    ConstantRegWeight,
    Regularizator,
    RegWeightUpdateStrategy,
)
from pyrtid.regularization.discrete import DiscreteRegularizator
from pyrtid.regularization.distribution import ProbDistFitting
from pyrtid.regularization.geostatistical import (
    EnsembleRegularizator,
    GeostatisticalRegularizator,
)
from pyrtid.regularization.lcurve import get_l_curvature
from pyrtid.regularization.tikhonov import (
    TikhonovFVMRegularizator,
    TikhonovMatRegularizator,
    TikhonovRegularizator,
)
from pyrtid.regularization.tv import (
    TVFVMRegularizator,
    TVMatRegularizator,
    TVRegularizator,
)

__all__ = [
    "RegWeightUpdateStrategy",
    "Regularizator",
    "TikhonovRegularizator",
    "TikhonovMatRegularizator",
    "TVRegularizator",
    "TVMatRegularizator",
    "GeostatisticalRegularizator",
    "EnsembleRegularizator",
    "get_matrix_eigen_factorization",
    "get_explained_var",
    "eigen_factorize_cov_mat",
    "sample_from_sparse_cov_factor",
    "generate_dense_matrix",
    "AdaptiveUCRegweight",
    "AdaptiveRegweight",
    "ConstantRegWeight",
    "DiscreteRegularizator",
    "get_l_curvature",
    "AdaptiveGradientNormRegweight",
    "TikhonovFVMRegularizator",
    "TVFVMRegularizator",
    "ProbDistFitting",
]
