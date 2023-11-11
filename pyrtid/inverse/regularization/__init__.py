"""
PyRTID invsere sub module providing regularization tools.

The following functionalities are directly provided on module-level.

.. currentmodule:: pyrtid.inverse.regularization

Abstract classes
================

Base class from which to derive regularizator implementations.

.. autosummary::
   :toctree: _autosummary

   Regularizator

Strategies for the weight evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Class to indicate what strategy to use to weight the objective
function regularization term.

.. autosummary::
   :toctree: _autosummary

    RegWeightUpdateStrategy

Local
=====

Tikhonov (for smooth spatial distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    TikhonovRegularizatorAnisotropic
    TikhonovRegularizatorIsotropic

Total Variation (for smooth spatial distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    TVRegularizatorAnisotropic
    TVRegularizatorIsotropic

Global
======

Geostatistic regularizator
^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide classes to implement regularization based on a parameter covariance matrix.
The first one work with a single vector while the second class works with
an ensemble of realizations.

.. autosummary::
   :toctree: _autosummary

    GeostatisticalRegularizator
    EnsembleRegularizator


Covariance classes
^^^^^^^^^^^^^^^^^^

To represent covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovarianceMatrix
    DenseCovarianceMatrix
    EnsembleCovarianceMatrix
    FFTCovarianceMatrix
    EigenFactorizedCovarianceMatrix
    SparseInvCovarianceMatrix
    HCovarianceMatrix
    SparseInvCovarianceMatrix
    generate_dense_matrix

Working with priors and trends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To represent trend through drift matrix. To use along with geostatistical regularizator.

.. autosummary::
   :toctree: _autosummary

    PriorTerm
    NullPriorTerm
    ConstantPriorTerm
    MeanPriorTerm
    EnsembleMeanPriorTerm
    DriftMatrix
    LinearDriftMatrix

Matrix compression
^^^^^^^^^^^^^^^^^^^

Eigen decomposition

.. autosummary::
   :toctree: _autosummary

    get_prior_eigen_factorization
    eigen_factorize_cov_mat

Stochastic partial differential equation compression (SPDE)

.. autosummary::
   :toctree: _autosummary

    eigen_factorize_cov_mat

"""

from pyrtid.inverse.regularization.base import Regularizator, RegWeightUpdateStrategy
from pyrtid.inverse.regularization.covariances import (
    CovarianceMatrix,
    DenseCovarianceMatrix,
    EigenFactorizedCovarianceMatrix,
    EnsembleCovarianceMatrix,
    FFTCovarianceMatrix,
    HCovarianceMatrix,
    SparseInvCovarianceMatrix,
    eigen_factorize_cov_mat,
    generate_dense_matrix,
    get_prior_eigen_factorization,
)
from pyrtid.inverse.regularization.geostatistical import (
    EnsembleRegularizator,
    GeostatisticalRegularizator,
)
from pyrtid.inverse.regularization.priors import (
    ConstantPriorTerm,
    DriftMatrix,
    EnsembleMeanPriorTerm,
    LinearDriftMatrix,
    MeanPriorTerm,
    NullPriorTerm,
    PriorTerm,
)
from pyrtid.inverse.regularization.tikhonov import (
    TikhonovRegularizatorAnisotropic,
    TikhonovRegularizatorIsotropic,
)
from pyrtid.inverse.regularization.tv import (
    TVRegularizatorAnisotropic,
    TVRegularizatorIsotropic,
)

__all__ = [
    "RegWeightUpdateStrategy",
    "Regularizator",
    "TikhonovRegularizatorAnisotropic",
    "TikhonovRegularizatorIsotropic",
    "TVRegularizatorAnisotropic",
    "TVRegularizatorIsotropic",
    "DenseCovarianceMatrix",
    "EnsembleCovarianceMatrix",
    "FFTCovarianceMatrix",
    "HCovarianceMatrix",
    "CovarianceMatrix",
    "EigenFactorizedCovarianceMatrix",
    "SparseInvCovarianceMatrix",
    "PriorTerm",
    "NullPriorTerm",
    "ConstantPriorTerm",
    "MeanPriorTerm",
    "EnsembleMeanPriorTerm",
    "DriftMatrix",
    "LinearDriftMatrix",
    "GeostatisticalRegularizator",
    "EnsembleRegularizator",
    "get_prior_eigen_factorization",
    "eigen_factorize_cov_mat",
    "generate_dense_matrix",
]
