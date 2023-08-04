"""
Main module.


"""
from __future__ import annotations

import copy
import logging
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from iterative_ensemble_smoother import SIES
from pyesmda import ESMDA, ESMDA_RS
from pyPCGA import PCGA
from scipy.optimize import OptimizeResult as ScipyOptimizeResult
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse import spmatrix
from stochopy.optimize import OptimizeResult as StochpyOptimizeResult
from stochopy.optimize import minimize as stochopy_minimize
from typing_extensions import Literal

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.inverse.adjoint import AdjointModel, AdjointSolver
from pyrtid.inverse.gradient import (
    compute_adjoint_gradient,
    compute_fd_gradient,
    is_adjoint_gradient_correct,
)
from pyrtid.inverse.loss_function import ls_loss_function
from pyrtid.inverse.model import InverseModel
from pyrtid.inverse.obs import (
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.inverse.params import (
    get_parameters_bounds,
    get_parameters_values_from_model,
    update_model_with_parameters_values,
    update_parameters_from_model,
)
from pyrtid.utils import is_all_close
from pyrtid.utils.types import NDArrayFloat, object_or_object_sequence_to_list


@dataclass
class BaseSolverConfig:
    """
    Base class for solver configuration.

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
    """

    is_verbose: bool = True
    hm_end_time: Optional[float] = None
    is_parallel: bool = False
    max_workers: int = 2
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = np.random.default_rng(198873)


_BaseSolverConfig = TypeVar("_BaseSolverConfig", bound=BaseSolverConfig)


class DataModel:
    """
    Wrapping class for all model inputs and observables.

    Parameters
    ----------
    obs_data : pd.DataFrame
        Obsevration data read from the simulation folder.
    m_init : np.array
        Initial ensemble of N_{e} parameters vector.
    cov_obs: NDArrayFloat
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`).

    """

    __slots__ = ["obs", "m_init", "_cov_obs", "std_m_prior"]

    def __init__(
        self,
        obs: NDArrayFloat,
        m_init: NDArrayFloat,
        cov_obs: NDArrayFloat,
        std_m_prior: NDArrayFloat,
    ) -> None:
        """Construct the instance."""
        self.obs = obs
        self.m_init = m_init
        self.cov_obs = cov_obs
        self.std_m_prior = std_m_prior

    @property
    def m_dim(self):
        """Return the length of the parameters vector."""
        return self.m_init.shape[1]

    @property
    def d_dim(self):
        """Return the number of observations / forecast data."""
        return self.obs.shape[0]

    @property
    def cov_obs(self) -> NDArrayFloat:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, s: NDArrayFloat) -> None:
        """Set the observation errors covariance matrix."""
        # pylint: disable=C0103  # arg name does not conform to snake_case naming style
        if len(s.shape) != 2 or s.shape[0] != s.shape[1]:
            raise ValueError(
                "cov_obs must be a square matrix with same "
                "dimensions as the observations vector."
            )
        if s.shape[0] != self.d_dim:
            raise ValueError(
                "cov_obs must be a square matrix with same "
                "dimensions as the observations vector."
            )
        self._cov_obs: NDArrayFloat = s


class BaseInversionExecutor(ABC, Generic[_BaseSolverConfig]):
    """
    Base class Executor for automated inversion.

    This is an abstract class.
    """

    __slots__ = [
        "fwd_model",
        "inv_model",
        "_adj_model",
        "solver_config",
        "pre_run_transformation",
        "data_model",
    ]
    _adj_model: Optional[AdjointModel]
    solver: Union[ESMDA, PCGA]

    def __init__(
        self,
        fwd_model: ForwardModel,
        inv_model: InverseModel,
        solver_config: _BaseSolverConfig,
        pre_run_transformation: Optional[Callable] = None,
        m_init: Optional[NDArrayFloat] = None,
    ) -> None:
        """
        Initialize the executor.

        Parameters
        ----------
        model : ForwardModel
            The reactive transport model to optimize.
        parameters_to_adjust: Sequence[AdjustableParameter]
            List of `Param` that the user wants to adjust. The availability
            is checked on the fly and an exception in raised if some are
            not available.
        observables : Union[Observable, List[Observable]]
            Observable instances defining the data to match.
        solver_config : _BaseSolverConfig
            Configuration for the solver and the inversion.
        pre_run_transformation : Optional[Callable], optional
            Pre transformation to apply to the rt_model before oe run.
            The default is None.
        m_init: Optional[NDArrayFloat]
            Initial adjusted values. This is required by some solvers such
            as ESMDA or SIES. In case of an ensemble, the expected shape
            is (Ne, Nm) with Ne the number of members in the ensemble and
            Nm the number of adjusted parameters. If None, it is retrieved
            from the model. The default is None.

        Note
        ----
        The fwd and inverse model passed to the executor will be modified by the
        executor while optimizing.

        Returns
        -------
        None.

        """
        self.adj_model = None
        self.fwd_model: ForwardModel = fwd_model
        self.inv_model: InverseModel = inv_model
        self.pre_run_transformation: Optional[Callable] = pre_run_transformation
        self.solver_config = solver_config

        # Update parameters (only if the values haven't been defined for the parameters)
        update_parameters_from_model(fwd_model, self.inv_model.parameters_to_adjust)

        # _std_m_prior = self.source_simulation.get_std_m_prior()
        _std_m_prior = np.array([])

        if m_init is None:
            _m_init = get_parameters_values_from_model(
                fwd_model, inv_model.parameters_to_adjust, is_preconditioned=True
            )
        else:
            _m_init = m_init

        # Need to differentiate flux and grids
        self.data_model = DataModel(
            self.obs, _m_init, np.diag(self.std_obs**2), _std_m_prior
        )

        # Initialize the solver (this is to be defined in child classes)
        self._init_solver(_m_init)

    @property
    def obs(self) -> NDArrayFloat:
        """Return the observation values as a 1d vector."""
        return get_observables_values_as_1d_vector(
            self.inv_model.observables, self.solver_config.hm_end_time
        )

    @property
    def std_obs(self) -> NDArrayFloat:
        """Return the observation uncertainties as a 1d vector."""
        return get_observables_uncertainties_as_1d_vector(
            self.inv_model.observables, self.solver_config.hm_end_time
        )

    @property
    def adj_model(self) -> AdjointModel:
        """
        Return the adjoint model if exists, otherwise raise an AttributeError.

        Returns
        -------
        AdjointModel
            The executor adjoint model.

        Raises
        ------
        AttributeError
            If the adjoint model does not exists.
        """
        if self._adj_model is not None:
            return self._adj_model
        raise AttributeError(
            "The adjoint model does not exists ! You must configure "
            "your solver to use the adjoint state !"
        )

    @adj_model.setter
    def adj_model(self, adj_model: Optional[AdjointModel]) -> None:
        self._adj_model = adj_model

    def _init_adjoint_model(self) -> None:
        """Initialize a new adjoint model for the executor."""
        self.adj_model = AdjointModel(
            self.fwd_model.geometry,
            self.fwd_model.time_params,
            self.fwd_model.gch_params,
        )
        # Add the sources
        for obs in object_or_object_sequence_to_list(self.inv_model.observables):
            self.adj_model.set_adjoint_sources_from_obs(obs, self.fwd_model)

    @abstractmethod
    def _init_solver(self, m_init: Optional[NDArrayFloat]) -> None:
        """Initiate a solver with its args."""
        pass

    @abstractmethod
    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "unknown"

    def _initial_display(self) -> None:
        """Display basic info about the simulation."""
        # Overview
        toplen = 80
        logging.info(f"{' Inversion Parameters ' :=^{toplen}}")

        shift = 50
        # display specific to the solver
        logging.info(f"{'Method' : <{shift}}: {self._get_solver_name()}")
        logging.info("")
        logging.info(
            f"{'Number of adjusted parameters' : <{shift}}: "
            f"{len(self.inv_model.parameters_to_adjust)}"
        )
        logging.info(
            f"{'Number of unknowns (adjusted values)' : <{shift}}:"
            f" {self.inv_model.nb_adjusted_values}"
        )
        logging.info(
            f"{'Number of observables' : <{shift}}:"
            f" {len(self.inv_model.observables)}"
        )
        logging.info(
            f"{'Number of observation data points (values)' : <{shift}}:"
            f" {self.inv_model.nb_obs_values}"
        )

        # TODO: this is solver specific -> move it smwh
        # self.solver_config_log
        # logging.info(f"{'Stop criteria on cost function value' : <{shift}}: {2}")
        # logging.info(f"{'Minimum change in cost function' : <{shift}}: {2}")
        # logging.info(f"{'Maximum number of forward HYTEC calls' : <{shift}}: {2}")
        # logging.info(f"{'Maximum number of iterations' : <{shift}}: {2}")
        # logging.info(f"{'Minimum change in parameter vector' : <{shift}}: {2}")

        # logging.info(f"{'Maximum number of HYTEC gradient calls' : <{shift}}: {2}")
        # logging.info(f"{'Minimum norm of the gradient vector' : <{shift}}: {2}")
        # logging.info(f"{'Number of gradient kept in memory' : <{shift}}: {2}")
        # logging.info(f"{'Adjoint-state status' : <{shift}}: {2}")
        # logging.info(f"{'Check gradient by finite difference' : <{shift}}: {2}")

        # End of display
        logging.info(f"{'' :=^{toplen}}")

    def _run_forward_model(
        self, m: NDArrayFloat, run_n: int, is_save_state: bool = True
    ) -> NDArrayFloat:
        """
        Run the forward model and returns the prediction vector.

        Parameters
        ----------
        m : np.array
            Inverted parameters values as a 1D vector.
        run_n: int
            Run number.
        is_save_state: bool
            Whether the parameter values must be stored or not.
            The default is True.

        Returns
        -------
        d_pred: np.array
            Vector of results matching the observations.

        """
        logging.info("- Running forward model # %s", run_n)

        # Update the model with the new values of x (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            m,
            self.inv_model.parameters_to_adjust,
            is_preconditioned=True,
            is_to_save=is_save_state,  # This is not finite differences
        )

        # Apply user transformation is needed:
        if self.pre_run_transformation is not None:
            self.pre_run_transformation(self.fwd_model)

        # Solve the forward model with the new parameters
        ForwardSolver(self.fwd_model).solve()

        # TODO: add hm_end_time
        d_pred = get_predictions_matching_observations(
            self.fwd_model, self.inv_model.observables
        )

        # Save the predictions
        if is_save_state:
            self.inv_model.list_d_pred.append(d_pred)

        self._check_nans_in_predictions(d_pred, run_n)

        # Read the results at the observation well
        # Update the prediction vector for the parameters m(j)
        logging.info("- Run # %s over", run_n)

        return d_pred

    def _map_forward_model(
        self, m_ensemble: NDArrayFloat, is_parallel: bool = False
    ) -> NDArrayFloat:
        """
        Call the forward model for all ensemble members, return predicted data.

        Function calling the non-linear observation model (forward_model)
        for all ensemble members and returning the predicted data for
        each ensemble member. this function is responsible for the creation of
        simulation folder etc.

        Returns
        -------
        None.
        """
        run_n: int = self.inv_model.nb_f_calls
        n_ensemble: int = m_ensemble.shape[0]
        d_pred: NDArrayFloat = np.zeros([m_ensemble.shape[0], self.data_model.d_dim])
        if is_parallel:
            with ProcessPoolExecutor(
                max_workers=self.solver_config.max_workers
            ) as executor:
                results: Iterator[NDArrayFloat] = executor.map(
                    self._run_forward_model,
                    m_ensemble,
                    range(run_n + 1, run_n + n_ensemble + 1),
                )
                for j, res in enumerate(results):
                    d_pred[j, :] = res
            # self.simu_n += n_ensemble
        else:
            for j in range(n_ensemble):
                d_pred[j, :] = self._run_forward_model(m_ensemble[j, :], run_n + j + 1)
        # update the number of runs

        # The check is already done in foward_model but nan can also be introduced
        # because of the stacking. So it is necessary to check
        self._check_nans_in_predictions(d_pred, run_n)

        # save objective functions. This should be very fast.
        for i in range(d_pred.shape[0]):
            ls_loss = ls_loss_function(
                d_pred[i, :],
                get_observables_values_as_1d_vector(self.inv_model.observables),
                get_observables_uncertainties_as_1d_vector(self.inv_model.observables),
            )
            self.inv_model.list_f_res.append(ls_loss)

        return d_pred

    def scaled_loss_function(
        self, m: NDArrayFloat, is_save_state: bool = True
    ) -> float:
        """Compute the model scaled_loss function."""
        ls_loss = ls_loss_function(
            self._run_forward_model(
                m, self.inv_model.nb_f_calls + 1, is_save_state=is_save_state
            ),
            get_observables_values_as_1d_vector(self.inv_model.observables),
            get_observables_uncertainties_as_1d_vector(self.inv_model.observables),
        )

        # Compute the regularization term:
        reg_factor = self.solver_config.__dict__.get("reg_factor", "auto")
        reg_loss = self.inv_model.get_jreg(ls_loss, reg_factor)
        total_loss = ls_loss + reg_loss

        # Apply the scaling coefficient
        scaled_loss = total_loss * self.inv_model.get_loss_function_scaling_factor(
            total_loss
        )

        logging.info(f"Loss (obs fit)        = {ls_loss}")
        logging.info(f"Loss (regularization) = {reg_loss}")
        logging.info(f"Scaling factor        = {self.inv_model.scaling_factor}")
        logging.info(f"Loss (scaled)         = {scaled_loss}\n")

        # Save the loss
        self.inv_model.list_f_res.append(scaled_loss)

        return scaled_loss

    def scaled_loss_function_gradient(self, m: NDArrayFloat) -> NDArrayFloat:
        """
        Return the gradient of the objective function with regard to `x`.

        Parameters
        ----------
        x: NDArrayFloat
            1D vector of inversed parameters.

        Returns
        -------
        objective : NDArrayFloat
            The gradient vector. Note that the dimension is the same as for x.

        """
        return np.zeros(m.shape)

    @abstractmethod
    def run(self) -> Optional[Sequence[Any]]:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        self._initial_display()

        # TODO: see if we add a function or a boolean to reset the
        # inverse model, the param archived values etc.
        return ()

    @staticmethod
    def create_output_dir(path: Path) -> None:
        """
        Create an output directory.

        Parameters
        ----------
        path : Path
            Path to the folder where the output figures are saved.

        Returns
        -------
        None.

        """
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    @staticmethod
    def _check_nans_in_predictions(d_pred: NDArrayFloat, simu_n: int) -> None:
        """
        Check and raise an exception if there is any NaNs in the input array.

        Parameters
        ----------
        d_pred : NDArrayFloat
            Input prediction vector(s).
        simu_n : int
            Simulation number. If d_pred is for an ensemble, it is the number
            of the first simulation minus one.

        Raises
        ------
        Exception
            Raised if NaNs are found. It indicates which simulations have incorrect
            predictions.
        """
        # Check if no nan values are found in the predictions.
        # If so, stop the assimilation
        if not np.isnan(d_pred).any():
            return  # -> no issue found

        # Case of a vector
        if len(d_pred.shape) == 1:
            msg: str = (
                "Something went wrong with NaN values"
                f" are found in predictions for simulation {simu_n} !"
            )
        # Case of an ensemble
        else:
            # + simu_n + 1 to get the indices of simulations
            error_indices: List[int] = sorted(
                set(np.where(np.isnan(d_pred))[0] + simu_n + 1)
            )
            msg = (
                "Something went wrong with NaN values"
                f" are found in predictions for simulation(s) {error_indices} !"
            )
        raise Exception(msg)

    def is_adjoint_gradient_correct(
        self,
        eps: Optional[float] = None,
    ) -> bool:
        """
        Return whether the adjoint gradient is correct or not.

        Note
        ----
        The numerical gradient by finite difference is computed only on the
        optimized area (sliced parameter values) while the adjoint gradient is
        computed everywhere. This allows to check the gradient on small portions
        of big models.
        """
        return is_adjoint_gradient_correct(
            self.fwd_model,
            self.adj_model,
            self.inv_model.parameters_to_adjust,
            self.inv_model.observables,
            eps=eps,
        )


@dataclass
class PCGASolverConfig(BaseSolverConfig):
    """
    Principal Component Geostatistical Approach Inversion Configuration.

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
    solver_kwargs: Optional[Dict[str, Any]]
        Additional arguments for PCGA instance. The default is None.
    """

    solver_kwargs: Optional[Dict[str, Any]] = None


class PCGAInversionExecutor(BaseInversionExecutor[PCGASolverConfig]):
    """Principal Component Geostatistical Approach Inversion Executor."""

    def _init_solver(self, m_init: NDArrayFloat = None) -> None:
        """Initiate a solver with its args."""
        # Array with grid coordinates. (X, Y, Z)...
        # Note: for regular grid you don't need to specify pts.
        self.pts = None
        self.solver: PCGA = PCGA(
            self._map_forward_model_wrapper,
            self.data_model.m_init.ravel(),  # Need to be a vector
            self.pts,
            params=self.solver_config.solver_kwargs,
            obs=self.data_model.obs,
            random_state=self.solver_config.random_state,
        )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "PCGA"

    def run(self) -> Optional[Sequence[Any]]:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        return self.solver.Run()

    def _map_forward_model_wrapper(
        self, m_ensemble: NDArrayFloat, is_parallel: bool = False, ncores: int = 1
    ) -> NDArrayFloat:
        """
        Call the forward model for all ensemble members, return predicted data.

        Function calling the non-linear observation model (forward_model)
        for all ensemble members and returning the predicted data for
        each ensemble member. this function is responsible for the creation of
        simulation folder etc.

        Returns
        -------
        None.
        """
        # pylint: disable=W0613  # Unused argument 'ncores'
        # The transposition is due to the implementation of pypcga
        return super()._map_forward_model(m_ensemble.T, is_parallel).T


@dataclass
class ESMDASolverConfig(BaseSolverConfig):
    """
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

    """

    n_assimilations: int = 4
    cov_obs_inflation_factors: Optional[Sequence[float]] = None
    cov_ss_inflation_factors: Optional[Sequence[float]] = None
    dd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    sd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 5000
    is_parallel_analyse_step: bool = True


class ESMDAInversionExecutor(BaseInversionExecutor[ESMDASolverConfig]):
    """Ensemble Smoother with Multiple Data Assimilation Inversion Executor."""

    def _init_solver(self, m_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: ESMDA = ESMDA(
            self.data_model.obs,
            m_init,  # To change back
            self.data_model.cov_obs,
            self._map_forward_model,
            forward_model_kwargs={"is_parallel": self.solver_config.is_parallel},
            m_bounds=get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            ),
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
    """

    std_s_prior: Optional[NDArrayFloat] = None
    cov_ss_inflation_factor: float = 1.0
    dd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    sd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None
    save_ensembles_history: bool = False
    is_forecast_for_last_assimilation: bool = True
    batch_size: int = 5000
    is_parallel_analyse_step: bool = True


class ESMDARSInversionExecutor(BaseInversionExecutor[ESMDARSSolverConfig]):
    """Restricted Step Ensemble Smoother with Multiple Data Assimilation Executor."""

    def _init_solver(self, m_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: ESMDA_RS = ESMDA_RS(
            self.data_model.obs,
            m_init,  # To change back
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


@dataclass
class SIESSolverConfig(BaseSolverConfig):
    """
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
    n_iterations : int, optional
        Number of iterations (:math:`N_{a}`). The default is 4.
    save_ensembles_history: bool, optional
        Whether to save the history predictions and parameters over
        the assimilations. The default is False.
    seed: Optional[int]
        Seed for the white noise generator used in the perturbation step.
        If None, the default :func:`numpy.random.default_rng()` is used.
        The default is None.
    is_forecast_for_last_assimilation: bool, optional
        Whether to compute the predictions for the ensemble obtained at the
        last assimilation step. The default is True.
    """

    n_iterations: int = 4
    save_ensembles_history: bool = False
    seed: Optional[int] = None
    is_forecast_for_last_assimilation: bool = True


class _SIES(SIES):
    """Wrapper for the SIES class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the instance."""
        super().__init__(*args, **kwargs)
        self.d_history: List[NDArrayFloat] = []
        self.m_history: List[NDArrayFloat] = []


class SIESInversionExecutor(BaseInversionExecutor[SIESSolverConfig]):
    """Ensemble Smoother with Multiple Data Assimilation Inversion Executor."""

    def _init_solver(self, m_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: _SIES = _SIES(m_init.shape[0], seed=self.solver_config.seed)

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "SIES"

    def run(self, m_init: NDArrayFloat) -> NDArrayFloat:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        _m = m_init.copy()
        if self.solver_config.save_ensembles_history:
            self.solver.m_history.append(_m)
        for iteration in range(self.solver_config.n_iterations):
            logging.info(f"Iteration # {iteration}")
            d_pred = self._map_forward_model(
                _m, is_parallel=self.solver_config.is_parallel
            )
            self.solver.d_history.append(d_pred)
            self.solver.fit(
                d_pred.T,
                self.data_model.cov_obs.diagonal(),
                self.data_model.obs,
                param_ensemble=_m.T,
            )
            _m = self.solver.update(_m.T).T

            if self.solver_config.save_ensembles_history:
                self.solver.m_history.append(_m)
        return _m


@dataclass
class ScipySolverConfig(BaseSolverConfig):
    """
    Configuration for Scipy solvers.

    Parameters
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
    is_check_gradient: bool
        Whether the gradient The default is False.
    max_fun_per_round: int
        The number of function evaluation before a new round  starts.
    reg_factor: Union[float, str] = "auto"
        The default is "auto".

    """

    solver_name: str = "L-BFGS-B"
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int = 5
    is_check_gradient: bool = False
    is_use_adjoint: bool = True
    is_regularization_at_first_round: bool = True
    reg_factor: Union[float, str] = "auto"


class ScipyInversionExecutor(BaseInversionExecutor[ScipySolverConfig]):
    """Represent a inversion executor instance using scipy's solvers."""

    def _init_solver(self, m_init: NDArrayFloat) -> None:
        super()._init_solver(m_init)

        # Create an adjoint model only if needed
        self.adj_model = None
        if self.solver_config.is_use_adjoint:
            self._init_adjoint_model()

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return self.solver_config.solver_name

    def run(self) -> ScipyOptimizeResult:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        res: ScipyOptimizeResult = ScipyOptimizeResult()
        x0 = self.data_model.m_init

        self.inv_model.is_regularization_at_first_round = (
            self.solver_config.is_regularization_at_first_round
        )

        # The optimization loop might be launched several time successively to
        # re-compute the regularization weights if automatically determined.
        while self.inv_model.is_new_optimization_round_needed(
            self.solver_config.max_optimization_round_nb
        ):
            # Reset the booleans for the new loop
            self.inv_model.is_first_loss_function_call_in_round = True
            self.inv_model.optimization_round_nb += 1
            logging.info(
                "Entering optimization loop: %s", self.inv_model.optimization_round_nb
            )
            # Update options and stop criteria from the previous loops
            _options: Dict[str, Any] = self._get_options_dict(
                self.solver_config,
                self.inv_model.nb_f_calls,
                self.inv_model.optimization_round_nb,
            )

            res = scipy_minimize(
                self.scaled_loss_function,
                x0,
                bounds=get_parameters_bounds(
                    self.inv_model.parameters_to_adjust, is_preconditioned=True
                ),
                method=self.solver_config.solver_name,
                jac=self.scaled_loss_function_gradient,
                options=_options,
            )
            # The output parameter vector becomes the input
            x0 = res.x
        return res

    def _get_options_dict(
        self, solver_config: ScipySolverConfig, nfev: int, round: int
    ) -> Dict[str, Any]:
        """Update optimization stop criteria."""
        if solver_config.solver_options is not None:
            options = copy.deepcopy(solver_config.solver_options)
        else:
            options = {}

        if solver_config.max_optimization_round_nb == 1:
            max_fun: int = options.get("maxfun", 15000)
        elif round == 1:
            max_fun: int = min(
                solver_config.max_fun_first_round,
                solver_config.max_fun_per_round,
                options.get("maxfun", 15000) - nfev,
            )
        else:
            max_fun: int = min(
                solver_config.max_fun_per_round, options.get("maxfun", 15000) - nfev
            )
        options["maxfun"] = max_fun
        return options

    def scaled_loss_function_gradient(self, m: NDArrayFloat) -> NDArrayFloat:
        """
        Return the gradient of the objective function with regard to `x`.

        Parameters
        ----------
        x: NDArrayFloat
            1D vector of inversed parameters.

        Returns
        -------
        objective : NDArrayFloat
            The gradient vector. Note that the dimension is the same as for x.

        """
        # Update the number of times the gradient computation has been performed
        self.inv_model.nb_g_calls += 1

        logging.info("- Running gradient # %s", self.inv_model.nb_g_calls)

        adj_grad = np.array([], dtype=np.float64)
        fd_grad = np.array([], dtype=np.float64)
        if self.solver_config.is_use_adjoint or self.solver_config.is_check_gradient:
            # Reinitialize the adjoint model
            self._init_adjoint_model()
            # Solve the adjoint system
            solver = AdjointSolver(self.fwd_model, self.adj_model)
            solver.solve()
            # Compute the gradient with the adjoint state method
            adj_grad = (
                compute_adjoint_gradient(
                    self.fwd_model,
                    self.adj_model,
                    self.inv_model.parameters_to_adjust,
                    self.inv_model.jreg_weight,
                )
                * self.inv_model.scaling_factor
            )

        if (
            not self.solver_config.is_use_adjoint
            or self.solver_config.is_check_gradient
        ):
            # Compute the gradient by finite difference
            fd_grad = (
                compute_fd_gradient(
                    self.fwd_model,
                    self.inv_model.observables,
                    self.inv_model.parameters_to_adjust,
                    self.inv_model.jreg_weight,
                )
                * self.inv_model.scaling_factor
            )

        if self.solver_config.is_check_gradient:
            if not is_all_close(adj_grad, fd_grad):
                logging.warning("The adjoint gradient is not correct!")
            else:
                logging.info("The adjoint gradient seems correct!")

        logging.info("- Gradient eval # %s over\n", self.inv_model.nb_g_calls)
        if self.solver_config.is_use_adjoint:
            return adj_grad
        return fd_grad


@dataclass
class StochopySolverConfig(BaseSolverConfig):
    """_summary_

    Parameters
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

    """

    # TODO: add other parameters names
    solver_name: Literal["cmaes", "cpso", "de", "na", "pso", "vdcma"] = "cmaes"
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_per_round: int = 5


class StochopyInversionExecutor(BaseInversionExecutor[StochopySolverConfig]):
    """Represent a inversion executor instance using stochopy's solvers."""

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return self.solver_config.solver_name

    def run(self) -> StochpyOptimizeResult:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        res = StochpyOptimizeResult()
        x0 = self.data_model.m_init

        # Empty dict for the results
        # res: Dict[str, Any] = {}
        # The optimization loop might be launched several time successively to
        # re-compute the regularization weights if automatically determined.
        while self.inv_model.is_new_optimization_round_needed(
            self.solver_config.max_optimization_round_nb
        ):
            # Reset the booleans for the new loop
            self.inv_model.is_first_loss_function_call_in_round = True
            self.inv_model.optimization_round_nb += 1
            logging.info(
                "Entering optimization loop: %s", self.inv_model.optimization_round_nb
            )
            # Update options and stop criteria from the previous loops
            _options: Dict[str, Any] = self._get_options_dict(
                self.solver_config, self.inv_model.nb_f_calls
            )

            res = stochopy_minimize(
                self.scaled_loss_function,
                get_parameters_bounds(
                    self.inv_model.parameters_to_adjust, is_preconditioned=True
                ),
                x0=x0,
                method=self.solver_config.solver_name,
                options=_options,
            )
            # The output parameter vector becomes the input
        return res

    def _get_options_dict(
        self,
        solver_config: StochopySolverConfig,
        nfev: int,
    ) -> Dict[str, Any]:
        """Update optimization stop criteria."""
        if solver_config.solver_options is not None:
            options = copy.deepcopy(solver_config.solver_options)
        else:
            options = {}

        max_fun = min(
            solver_config.max_fun_per_round, options.get("maxfun", 15000) - nfev
        )

        if (
            self.inv_model.optimization_round_nb
            != solver_config.max_optimization_round_nb
        ):
            options["maxfun"] = max_fun
        else:
            options["maxfun"] = 0
        return options
