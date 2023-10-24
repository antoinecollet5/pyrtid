"""Implement the interface for inversion executors."""
from __future__ import annotations

import logging
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.inverse.adjoint import AdjointModel
from pyrtid.inverse.adjoint.gradients import is_adjoint_gradient_correct
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
from pyrtid.utils.types import NDArrayFloat


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
    r"""
    Wrapping class for all model inputs and observables.

    Parameters
    ----------
    obs_data : pd.DataFrame
        Obsevration data read from the simulation folder.
    s_init : np.array
        Initial ensemble of N_{e} parameters vector.
    cov_obs: NDArrayFloat
        Either a 1D array of diagonal covariances, or a 2D covariance matrix.
        The shape is either (:math:`N_{obs}`) or (:math:`N_{obs}`, :math:`N_{obs}`).
        This is usually denoted :math:`\mathbf{R}` or :math:`\mathbf{C}_{\mathrm{dd}}`
        and represents observation or measurement errors.
        We observe d from the real world, y from the model g(x), and
        assume that d = y + e, where the error e is multivariate normal with
        covariance given by `covariance`.
    """

    __slots__ = ["obs", "s_init", "_cov_obs", "std_m_prior"]

    def __init__(
        self,
        obs: NDArrayFloat,
        s_init: NDArrayFloat,
        cov_obs: NDArrayFloat,
        std_m_prior: NDArrayFloat,
    ) -> None:
        """Construct the instance."""
        self.obs = obs
        self.s_init = s_init
        self.cov_obs = cov_obs
        self.std_m_prior = std_m_prior

    @property
    def s_dim(self):
        """Return the length of the parameters vector."""
        if len(self.s_init.shape) == 2:
            return self.s_init.shape[1]
        return self.s_init.shape[0]  # type: ignore

    @property
    def d_dim(self):
        """Return the number of observations / forecast data."""
        return self.obs.shape[0]  # type: ignore

    @property
    def cov_obs(self) -> NDArrayFloat:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, s: NDArrayFloat) -> None:
        """Set the observation errors covariance matrix."""
        # pylint: disable=C0103  # arg name does not conform to snake_case naming style
        is_error = False
        if s.shape[0] != self.d_dim:  # type: ignore
            is_error = True
        if len(s.shape) == 2:
            if s.shape[0] != s.shape[1]:
                is_error = True
        elif len(s.shape) > 2:
            is_error = True
        if is_error:
            raise ValueError(
                "cov_obs must be either a 1D array of diagonal covariances, "
                "or a 2D covariance matrix. The shape is either "
                "(Nobs) or (Nobs, Nobs)."
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

    def __init__(
        self,
        fwd_model: ForwardModel,
        inv_model: InverseModel,
        solver_config: _BaseSolverConfig,
        pre_run_transformation: Optional[Callable] = None,
        s_init: Optional[NDArrayFloat] = None,
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
        s_init: Optional[NDArrayFloat]
            Initial preconditioned adjusted values.
            This is required by some solvers such
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

        # Get the initial values from the model
        _s_init_model = get_parameters_values_from_model(
            fwd_model, inv_model.parameters_to_adjust, is_preconditioned=True
        )

        if s_init is not None:
            # check the dimensions and the bounds (clip to bounds + raise warning)
            _s_init = self.validate_s_init(s_init, _s_init_model.size)
        else:
            _s_init = _s_init_model

        # Need to differentiate flux and grids
        self.data_model = DataModel(
            self.obs, _s_init, np.diag(self.std_obs**2), _std_m_prior
        )

        # Initialize the solver (this is to be defined in child classes)
        self._init_solver(_s_init)

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

    def _init_adjoint_model(
        self, afpi_eps: float, is_numerical_acceleration: bool = False
    ) -> None:
        """Initialize a new adjoint model for the executor."""
        self.adj_model = AdjointModel(
            self.fwd_model.geometry,
            self.fwd_model.time_params,
            self.fwd_model.fl_model.is_gravity,
            afpi_eps,
            is_numerical_acceleration,
        )

    @abstractmethod
    def _init_solver(self, s_init: Optional[NDArrayFloat]) -> None:
        """Initiate a solver with its args."""

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

        d_pred = get_predictions_matching_observations(
            self.fwd_model, self.inv_model.observables, self.solver_config.hm_end_time
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
        self, s_ensemble: NDArrayFloat, is_parallel: bool = False
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
        n_ensemble: int = s_ensemble.shape[1]  # type: ignore
        print(f"n_ensemble = {n_ensemble}")
        d_pred: NDArrayFloat = np.zeros([self.data_model.d_dim, n_ensemble])
        if is_parallel:
            with ProcessPoolExecutor(
                max_workers=self.solver_config.max_workers
            ) as executor:
                results: Iterator[NDArrayFloat] = executor.map(
                    self._run_forward_model,
                    s_ensemble.T,
                    range(run_n + 1, run_n + n_ensemble + 1),  # type: ignore
                )
                for j, res in enumerate(results):
                    d_pred[:j] = res
            # self.simu_n += n_ensemble
        else:
            for j in range(n_ensemble):  # type: ignore
                d_pred[:, j] = self._run_forward_model(s_ensemble[:, j], run_n + j + 1)
        # update the number of runs

        # The check is already done in Forward_model but nan can also be introduced
        # because of the stacking. So it is necessary to check
        self._check_nans_in_predictions(d_pred, run_n)

        # save objective functions. This should be very fast.
        for i in range(d_pred.shape[1]):  # type: ignore
            ls_loss = ls_loss_function(
                d_pred[:, i],
                get_observables_values_as_1d_vector(
                    self.inv_model.observables, self.solver_config.hm_end_time
                ),
                get_observables_uncertainties_as_1d_vector(
                    self.inv_model.observables, self.solver_config.hm_end_time
                ),
            )

            self.inv_model.list_f_res.append(ls_loss)

        return d_pred  # shape (N_obs, N_e)

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

        # Save the loss and the associated regularization weight
        if is_save_state:
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

    @staticmethod  # type: ignore
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

    @staticmethod  # type: ignore
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
        if d_pred.ndim == 1:
            msg: str = (
                "Something went wrong with NaN values"
                f" are found in predictions for simulation {simu_n} !"
            )
        # Case of an ensemble
        else:
            # + simu_n + 1 to get the indices of simulations
            error_indices: List[int] = sorted(
                set(np.where(np.isnan(d_pred))[1] + simu_n + 1)  # type: ignore
            )
            msg = (
                "Something went wrong with NaN values"
                f" are found in predictions for simulation(s) {error_indices} !"
            )
        raise Exception(msg)

    def is_adjoint_gradient_correct(
        self,
        eps: Optional[float] = None,
        max_workers: int = 1,
        is_verbose: bool = False,
    ) -> bool:
        """
        Return whether the adjoint gradient is correct or not.

        Note
        ----
        The numerical gradient by finite difference is computed only on the
        optimized area (sliced parameter values) while the adjoint gradient is
        computed everywhere. This allows to check the gradient on small portions
        of big models.

        Parameters
        ----------
        eps: float, optional
            The epsilon for the computation of the approximated gradient by finite
            difference. If None, it is automatically inferred. The default is None.
        max_workers: int
            Number of workers used for the gradient approximation by finite
            differences. If different from one, the calculation relies on
            multi-processing to decrease the computation time. The default is 1.
        is_verbose : bool, optional
            Whether to display computation infrmation, by default False
        """
        return is_adjoint_gradient_correct(
            self.fwd_model,
            self.adj_model,
            self.inv_model.parameters_to_adjust,
            self.inv_model.observables,
            eps=eps,
            max_workers=max_workers,
            hm_end_time=self.solver_config.hm_end_time,
            is_verbose=is_verbose,
        )

    def validate_s_init(
        self, s_init: NDArrayFloat, expected_s_dim: int
    ) -> NDArrayFloat:
        """Check if s init has the correct size."""
        if s_init.size == expected_s_dim:
            s_init = s_init.ravel()
        if s_init.ndim != 2 or s_init.shape[0] != expected_s_dim:  # type: ignore
            raise ValueError(
                "s_init must be either a 1D vector of shape (N_s)"
                " or a 2D array of shape (N_s, N_e) with N_s the number of"
                " adjusted values, and N_e the number of members (realizations) in"
                " the provided ensemble."
            )

        # Ensure bounds (preconditionned)
        bounds = get_parameters_bounds(
            self.inv_model.parameters_to_adjust, is_preconditioned=True
        )

        clipped = np.clip(s_init.T, bounds[:, 0], bounds[:, 1]).T

        if not np.array_equal(clipped, s_init):
            warnings.warn(
                "There are values out of bounds in the provided s_init!"
                "Remember that preconditioned values are expected (only applies for "
                "precondtioned parameters)."
                "\nCheck your inputs if this is not desired."
            )
        return clipped
