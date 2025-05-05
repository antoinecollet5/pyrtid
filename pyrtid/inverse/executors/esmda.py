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

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pyesmda import ESMDA, ESMDA_DMC, ESMDA_RS, ESMDAInversionType
from pyesmda.localization import LocalizationStrategy, NoLocalization

from pyrtid.inverse.executors.base import (
    FSMInversionExecutor,
    FSMSolverConfig,
    base_solver_config_params_ds,
    fsm_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import (
    get_parameters_bounds,
    update_model_with_parameters_values,
)
from pyrtid.utils import NDArrayFloat

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
    C_DD_localization: LocalizationStrategy
        Localization operator :math:`\rho_{DD}` applied to the predictions
        empirical auto-covariance matrices. Expected dimensions of the operator are
        (:math:`N_{obs}`, :math:`N_{obs}`). It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined.
        See implementations of :class:`LocalizationStrategy`.
    C_SD_localization : LocalizationStrategy
        Localization operator :math:`\rho_{DD}` applied to the parameters-predictions
        empirical corss-covariance matrices. Expected dimensions of the operator are
        (:math:`N_{m}`, :math:`N_{obs}`). It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined.
        See implementations of :class:`LocalizationStrategy`.
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
    logger: Optional[logging.Logger]
        Optional :class:`logging.Logger` instance used for event logging.
        The default is None.
    """


@register_params_ds(esmda_solver_config_params_ds_common)
@register_params_ds(esmda_base_solver_config_params_ds)
@register_params_ds(fsm_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class ESMDASolverConfig(FSMSolverConfig):
    r"""
    Ensemble Smoother with Multiple Data Assimilation Inversion Configuration.

    Attributes
    ----------
    """

    n_assimilations: int = 4
    cov_obs_inflation_factors: Optional[Sequence[float]] = None
    inversion_type: ESMDAInversionType = ESMDAInversionType.SUBSPACE_RESCALED
    cov_ss_inflation_factor: float = 1.0
    C_DD_localization: LocalizationStrategy = NoLocalization()
    C_SD_localization: LocalizationStrategy = NoLocalization()
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 5000
    is_parallel_analyse_step: bool = True
    truncation: float = 0.99
    logger: Optional[logging.Logger] = logging.getLogger("ESMDA")


class ESMDAInversionExecutor(FSMInversionExecutor[ESMDASolverConfig]):
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
            C_DD_localization=self.solver_config.C_DD_localization,
            C_MD_localization=self.solver_config.C_SD_localization,
            save_ensembles_history=self.solver_config.save_ensembles_history,
            random_state=self.solver_config.random_state,
            is_forecast_for_last_assimilation=self.solver_config.is_forecast_for_last_assimilation,
            batch_size=self.solver_config.batch_size,
            is_parallel_analyse_step=self.solver_config.is_parallel_analyse_step,
            logger=self.solver_config.logger,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "ESMDA"

    def run(self) -> None:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()

        self.solver.solve()

        # TODO: see if this works
        s_best_av_esmda = np.mean(self.s_history[-1], axis=1)

        # Update the model with the new values of s (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            s_best_av_esmda,
            self.inv_model.parameters_to_adjust,
            is_preconditioned=True,
            is_to_save=True,  # This is not finite differences
        )

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.m_history

    def get_display_dict(self) -> Dict[str, Any]:
        return {"Number of realizations": self.solver.n_ensemble}


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
class ESMDARSSolverConfig(FSMSolverConfig):
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
    C_DD_localization: LocalizationStrategy = NoLocalization()
    C_SD_localization: LocalizationStrategy = NoLocalization()
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 10000
    is_parallel_analyse_step: bool = True
    truncation: float = 0.99
    logger: Optional[logging.Logger] = logging.getLogger("ESMDA-RS")


class ESMDARSInversionExecutor(FSMInversionExecutor[ESMDARSSolverConfig]):
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
            C_DD_localization=self.solver_config.C_DD_localization,
            C_MD_localization=self.solver_config.C_SD_localization,
            save_ensembles_history=self.solver_config.save_ensembles_history,
            random_state=self.solver_config.random_state,
            is_forecast_for_last_assimilation=self.solver_config.is_forecast_for_last_assimilation,
            batch_size=self.solver_config.batch_size,
            is_parallel_analyse_step=self.solver_config.is_parallel_analyse_step,
            logger=self.solver_config.logger,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "ESMDA-RS"

    def run(self) -> None:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        self.solver.solve()

        # TODO: see if this works
        s_best_av_esmda = np.mean(self.s_history[-1], axis=1)

        # Update the model with the new values of s (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            s_best_av_esmda,
            self.inv_model.parameters_to_adjust,
            is_preconditioned=True,
            is_to_save=True,  # This is not finite differences
        )
        return

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.m_history


@register_params_ds(esmda_solver_config_params_ds_common)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class ESMDADMCSolverConfig(FSMSolverConfig):
    r"""
    Data Misfit Controller Ensemble Smoother with Multiple Data Assimilation Config.

    Note
    ----
    This is a data misfit controller version.

    Attributes
    ----------
    """

    inversion_type: ESMDAInversionType = ESMDAInversionType.SUBSPACE_RESCALED
    cov_ss_inflation_factor: float = 1.0
    C_DD_localization: LocalizationStrategy = NoLocalization()
    C_SD_localization: LocalizationStrategy = NoLocalization()
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 10000
    is_parallel_analyse_step: bool = True
    truncation: float = 0.99
    logger: Optional[logging.Logger] = logging.getLogger("ESMDA-DMC")


class ESMDADMCInversionExecutor(FSMInversionExecutor[ESMDADMCSolverConfig]):
    """Data Misfit Controller ESMDA Executor."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: ESMDA_DMC = ESMDA_DMC(
            self.data_model.obs,
            s_init,  # To change back
            self.data_model.cov_obs,
            self._map_forward_model,
            m_bounds=get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            ),
            cov_mm_inflation_factor=self.solver_config.cov_ss_inflation_factor,
            C_DD_localization=self.solver_config.C_DD_localization,
            C_MD_localization=self.solver_config.C_SD_localization,
            save_ensembles_history=self.solver_config.save_ensembles_history,
            random_state=self.solver_config.random_state,
            is_forecast_for_last_assimilation=self.solver_config.is_forecast_for_last_assimilation,
            batch_size=self.solver_config.batch_size,
            is_parallel_analyse_step=self.solver_config.is_parallel_analyse_step,
            logger=self.solver_config.logger,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "ESMDA-DMC"

    def run(self) -> None:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        self.solver.solve()

        # TODO: see if this works
        s_best_av_esmda = np.mean(self.s_history[-1], axis=1)

        # Update the model with the new values of s (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            s_best_av_esmda,
            self.inv_model.parameters_to_adjust,
            is_preconditioned=True,
            is_to_save=True,  # This is not finite differences
        )
        return

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.m_history
