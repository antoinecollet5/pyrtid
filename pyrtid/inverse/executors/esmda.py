"""
Implement the interface for the ESMDA inversion executor.

ESMDA stands for Ensemble smoother with multiple data assimilation.

This inversion executor gives access to the PyESMDA implementation to solve the inverse
problem. The original code is provided and maintained by Antoine COLLET
and can be found at: https://gitlab.com/antoinecollet5/pyesmda


References
----------
[1] Emerick, A. A. and A. C. Reynolds, Ensemble smoother with multiple
data assimilation, Computers & Geosciences, 2012.
[2] Emerick, A. A. and A. C. Reynolds. (2013). History-Matching
Production and Seismic Data in a Real Field Case Using the Ensemble
Smoother With Multiple Data Assimilation. Society of Petroleum
Engineers - SPE Reservoir Simulation Symposium
1.    2. 10.2118/163675-MS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

from pyesmda import ESMDA, ESMDA_RS, ESMDAInversionType
from scipy.sparse import spmatrix

from pyrtid.inverse.executors.base import (
    BaseInversionExecutor,
    BaseSolverConfig,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import get_parameters_bounds
from pyrtid.utils.types import NDArrayFloat

esmda_base_solver_config_params_ds = r"""n_assimilations : int, optional
        Number of data assimilations (:math:`N_{a}`). The default is 4.
    cov_obs_inflation_factors : Optional[Sequence[float]]
        Multiplication factor used to inflate the covariance matrix of the
        measurement errors.
        Must match the number of data assimilations (:math:`N_{a}`).
        The default is None.
    """

# Same for ESMDA and ESMDA-RS
esmda_solver_config_params_ds_common = r"""inversion_type: ESMDAInversionType
        Type of inversion used to solve :math:`(C_DD + \alpha CD)^{-1)(d-dobs)`.
        TODO: check the docstrings + add a comment about the best method for
        large scale problems. Maybe add a reference to the class + update
        the docs interpshinx. + Add to ESMDARS.
        The default is ESMDAInversionType.NAIVE.
    cov_ss_inflation_factor: float
        Factor used to inflate the initial ensemble around its mean.
        See :cite:p:`andersonExploringNeedLocalization2007`.
        The default is 1.0 i.e., no inflation.
    dd_correlation_matrix : Optional[Union[NDArrayFloat, spmatrix]]
        Correlation matrix based on spatial and temporal distances between
        observations and observations :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
        The default is None.
    sd_correlation_matrix : Optional[Union[NDArrayFloat, spmatrix]]
        Correlation matrix based on spatial and temporal distances between
        parameters and observations :math:`\rho_{SD}`. It is used to localize the
        cross-covariance matrix between the forecast state vector (parameters)
        and predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{s}`, :math:`N_{obs}`).
        The default is None.
    save_ensembles_history: bool, optional
        Whether to save the history predictions and parameters over
        the assimilations. The default is False.
    is_forecast_for_last_assimilation: bool, optional
        Whether to compute the predictions for the ensemble obtained at the
        last assimilation step. The default is True.
    batch_size: int
        Number of parameters that are assimilated at once. This option is
        available to overcome memory limitations when the number of parameters is
        large. In that case, the size of the covariance matrices tends to explode
        and the update step must be performed by chunks of parameters.
        The default is 5000.
    is_parallel_analyse_step: bool, optional
        Whether to use parallel computing for the analyse step if the number of
        batch is above one. It relies on `concurrent.futures` multiprocessing.
        The default is True.
    truncation: float
        A value in the range ]0, 1], used to determine the number of
        significant singular values kept when using svd for the inversion
        of :math:`(C_{dd} + \alpha C_{d})`: Only the largest singular values are kept,
        corresponding to this fraction of the sum of the nonzero singular values.
        The goal of truncation is to deal with smaller matrices (dimensionality
        reduction), easier to inverse. The default is 0.99.
    """


@register_params_ds(esmda_solver_config_params_ds_common)
@register_params_ds(esmda_base_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class ESMDASolverConfig(BaseSolverConfig):
    r"""
    Ensemble Smoother with Multiple Data Assimilation Inversion Configuration.

    Attributes
    ----------
    """

    n_assimilations: int = 4
    cov_obs_inflation_factors: Optional[Sequence[float]] = None
    inversion_type: ESMDAInversionType = ESMDAInversionType.SUBSPACE_RESCALED
    cov_ss_inflation_factor: float = 1.0
    dd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    sd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 5000
    is_parallel_analyse_step: bool = True
    truncation: float = 0.99


class ESMDAInversionExecutor(BaseInversionExecutor[ESMDASolverConfig]):
    """Ensemble Smoother with Multiple Data Assimilation Inversion Executor."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: ESMDA = ESMDA(
            self.data_model.obs,
            s_init,
            self.data_model.cov_obs,
            self._map_forward_model,
            m_bounds=get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            ),
            n_assimilations=self.solver_config.n_assimilations,
            inversion_type=self.solver_config.inversion_type,
            cov_obs_inflation_factors=self.solver_config.cov_obs_inflation_factors,
            cov_mm_inflation_factor=self.solver_config.cov_ss_inflation_factor,
            dd_correlation_matrix=self.solver_config.dd_correlation_matrix,
            md_correlation_matrix=self.solver_config.sd_correlation_matrix,
            save_ensembles_history=self.solver_config.save_ensembles_history,
            random_state=self.solver_config.random_state,
            is_forecast_for_last_assimilation=self.solver_config.is_forecast_for_last_assimilation,
            batch_size=self.solver_config.batch_size,
            is_parallel_analyse_step=self.solver_config.is_parallel_analyse_step,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "ESMDA"

    def run(self) -> Optional[Sequence[Any]]:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        return self.solver.solve()

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.m_history


esmda_rs_solver_config_params_ds = r"""std_s_prior: Optional[npt.NDArray[np.float64]]
        Vector of a priori standard deviation :math:`sigma` of the estimated
        parameter. The expected dimension is (:math:`N_{s}`).
        It is the diagonal of :math:`\Sigma_{s}`. If not provided, then it is inffered
        from the inflated initial ensemble (see `cov_ss_inflation_factor`).
        The default is None.
    """


@register_params_ds(esmda_solver_config_params_ds_common)
@register_params_ds(esmda_rs_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class ESMDARSSolverConfig(BaseSolverConfig):
    r"""
    Restricted Step Ensemble Smoother with Multiple Data Assimilation Configuration.

    Note
    ----
    This is a restricted step version.

    Attributes
    ----------
    """

    std_s_prior: Optional[NDArrayFloat] = None
    inversion_type: ESMDAInversionType = ESMDAInversionType.SUBSPACE_RESCALED
    cov_ss_inflation_factor: float = 1.0
    dd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    sd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 5000
    is_parallel_analyse_step: bool = True
    truncation: float = 0.99


class ESMDARSInversionExecutor(BaseInversionExecutor[ESMDARSSolverConfig]):
    """Restricted Step Ensemble Smoother with Multiple Data Assimilation Executor."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: ESMDA_RS = ESMDA_RS(
            self.data_model.obs,
            s_init,  # To change back
            self.data_model.cov_obs,
            self._map_forward_model,
            std_m_prior=self.solver_config.std_s_prior,
            m_bounds=get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            ),
            cov_mm_inflation_factor=self.solver_config.cov_ss_inflation_factor,
            dd_correlation_matrix=self.solver_config.dd_correlation_matrix,
            md_correlation_matrix=self.solver_config.sd_correlation_matrix,
            save_ensembles_history=self.solver_config.save_ensembles_history,
            random_state=self.solver_config.random_state,
            is_forecast_for_last_assimilation=self.solver_config.is_forecast_for_last_assimilation,
            batch_size=self.solver_config.batch_size,
            is_parallel_analyse_step=self.solver_config.is_parallel_analyse_step,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "ESMDA-Rs"

    def run(self) -> Optional[Sequence[Any]]:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        return self.solver.solve()

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.m_history
