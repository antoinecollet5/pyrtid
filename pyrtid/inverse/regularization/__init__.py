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

Provide a class to implement a regularization based on a parameter covariance matrix.

.. autosummary::
   :toctree: _autosummary

    GeostatisticalRegularizator


Covariance classes
^^^^^^^^^^^^^^^^^^

To represent covariance matrices.

.. autosummary::
   :toctree: _autosummary

    CovarianceMatrix
    DenseCovarianceMatrix
    FFTCovarianceMatrix
    CovarianceMatrixbyUd
    SparseInvCovarianceMatrix
    HCovarianceMatrix
    SparseInvCovarianceMatrix

Working with priors and trends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To represent trend through drift matrix. To use along with geostatistical regularizator.

.. autosummary::
   :toctree: _autosummary

    PriorTerm
    NullPriorTerm
    ConstantPriorTerm
    MeanPriorTerm
    DriftMatrix
    LinearDriftMatrix

Matrix compression
^^^^^^^^^^^^^^^^^^^

Eigen decomposition

.. autosummary::
   :toctree: _autosummary

    get_prior_eigen_factorization
    cov_mat_to_ud_mat

Stochastic partial differential equation compression (SPDE)

.. autosummary::
   :toctree: _autosummary

    cov_mat_to_ud_mat

"""

from pyrtid.inverse.regularization.base import Regularizator
from pyrtid.inverse.regularization.covariances import (
    CovarianceMatrix,
    CovarianceMatrixbyUd,
    DenseCovarianceMatrix,
    FFTCovarianceMatrix,
    HCovarianceMatrix,
    SparseInvCovarianceMatrix,
    cov_mat_to_ud_mat,
    get_prior_eigen_factorization,
)
from pyrtid.inverse.regularization.drift import (
    ConstantPriorTerm,
    DriftMatrix,
    LinearDriftMatrix,
    MeanPriorTerm,
    NullPriorTerm,
    PriorTerm,
)
from pyrtid.inverse.regularization.geostatistical import GeostatisticalRegularizator
from pyrtid.inverse.regularization.tikhonov import (
    TikhonovRegularizatorAnisotropic,
    TikhonovRegularizatorIsotropic,
)
from pyrtid.inverse.regularization.tv import (
    TVRegularizatorAnisotropic,
    TVRegularizatorIsotropic,
)

__all__ = [
    "Regularizator",
    "TikhonovRegularizatorAnisotropic",
    "TikhonovRegularizatorIsotropic",
    "TVRegularizatorAnisotropic",
    "TVRegularizatorIsotropic",
    "DenseCovarianceMatrix",
    "FFTCovarianceMatrix",
    "HCovarianceMatrix",
    CovarianceMatrix,
    "CovarianceMatrixbyUd",
    "SparseInvCovarianceMatrix",
    "PriorTerm",
    "NullPriorTerm",
    "ConstantPriorTerm",
    "MeanPriorTerm",
    "DriftMatrix",
    "LinearDriftMatrix",
    "GeostatisticalRegularizator",
    "get_prior_eigen_factorization",
    "cov_mat_to_ud_mat",
]
