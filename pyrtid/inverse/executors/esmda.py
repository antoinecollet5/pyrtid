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

from pyrtid.inverse.executors.base import BaseInversionExecutor, BaseSolverConfig
from pyrtid.inverse.params import get_parameters_bounds
from pyrtid.utils.types import NDArrayFloat


@dataclass
class ESMDASolverConfig(BaseSolverConfig):
    r"""
    Ensemble Smoother with Multiple Data Assimilation Inversion Configuration.

    Attributes
    ----------
    is_verbose: bool
        Whether to display inversion information. The default True.
    hm_end_time: Optional[float]
        Time at which the history matching ends and the forecast begins.
        This is not to confuse with the simulation `duration` which
        is already defined by the user in the htc file. The units are the same as
        given for the `duration` keyword in :term:`HYTEC`.
        If None, hm_end_time is set to the end of the simulation and
        all observations covering the simulation duration are taken into account.
        The default is None.
    is_parallel: bool, optional
        Whether to run the calculation one at the time or in a concurrent way.
    max_workers: int, optional
        Number of workers to use if the concurrency is enabled. The default is 2.
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
    n_assimilations : int, optional
        Number of data assimilations (:math:`N_{a}`). The default is 4.
    inversion_type: Union[ESMDAInversionType, str] = ESMDAInversionType.NAIVE
        Type of inversion used to solve :math:`(C_DD + \alpha CD)^{-1)(d-dobs)`.
        TODO: check the docstrings + add a comment about the best method for
        large scale problems. Maybe add a reference to the class + update
        the docs interpshinx. + Add to ESMDARS.
    cov_obs_inflation_factors : Optional[Sequence[float]]
        Multiplication factor used to inflate the covariance matrix of the
        measurement errors.
        Must match the number of data assimilations (:math:`N_{a}`).
        The default is None.
    cov_ss_inflation_factors: Optional[Sequence[float]]
        List of factors used to inflate the adjusted parameters covariance
        among iterations:
        Each realization of the ensemble at the end of each update step i,
        is linearly inflated around its mean.
        Must match the number of data assimilations (:math:`N_{a}`).
        See :cite:p:`andersonExploringNeedLocalization2007`.
        If None, the default is 1.0. at each iteration (no inflation).
        The default is None.
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

    n_assimilations: int = 4
    inversion_type: ESMDAInversionType = ESMDAInversionType.SUBSPACE_RESCALED
    cov_obs_inflation_factors: Optional[Sequence[float]] = None
    cov_ss_inflation_factors: Optional[Sequence[float]] = None
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
            forward_model_kwargs={"is_parallel": self.solver_config.is_parallel},
            m_bounds=get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            ),
            n_assimilations=self.solver_config.n_assimilations,
            inversion_type=self.solver_config.inversion_type,
            cov_obs_inflation_factors=self.solver_config.cov_obs_inflation_factors,
            cov_mm_inflation_factors=self.solver_config.cov_ss_inflation_factors,
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


@dataclass
class ESMDARSSolverConfig(BaseSolverConfig):
    r"""
    Restricted Step Ensemble Smoother with Multiple Data Assimilation Configuration.

    Note
    ----
    This is a restricted step version.

    Attributes
    ----------
    is_verbose: bool
        Whether to display inversion information. The default True.
    hm_end_time: Optional[float]
        Time at which the history matching ends and the forecast begins.
        This is not to confuse with the simulation `duration` which
        is already defined by the user in the htc file. The units are the same as
        given for the `duration` keyword in :term:`HYTEC`.
        If None, hm_end_time is set to the end of the simulation and
        all observations covering the simulation duration are taken into account.
        The default is None.
    is_parallel: bool, optional
        Whether to run the calculation one at the time or in a concurrent way.
    max_workers: int, optional
        Number of workers to use if the concurrency is enabled. The default is 2.
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
    std_s_prior: Optional[npt.NDArray[np.float64]]
        Vector of a priori standard deviation :math:`sigma` of the estimated
        parameter. The expected dimension is (:math:`N_{s}`).
        It is the diagonal of :math:`\Sigma_{s}`. If not provided, then it is inffered
        from the inflated initial ensemble (see `cov_ss_inflation_factor`).
        The default is None.
    cov_ss_initial_inflation_factor: float
        List of factors used to inflate the adjusted parameters covariance among
        iterations:
        Each realization of the ensemble at the end of each update step i,
        is linearly inflated around its mean.
        See :cite:p:`andersonExploringNeedLocalization2007`.
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

    inversion_type: ESMDAInversionType = ESMDAInversionType.SUBSPACE_RESCALED
    std_s_prior: Optional[NDArrayFloat] = None
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
            forward_model_kwargs={"is_parallel": self.solver_config.is_parallel},
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
