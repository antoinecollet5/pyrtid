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
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from pyrtid.forward import ForwardModel, ForwardSolver
from pyrtid.inverse.adjoint import AdjointModel, AdjointSolver
from pyrtid.inverse.adjoint.gradients import (
    compute_adjoint_gradient,
    compute_fd_gradient,
    is_adjoint_gradient_correct,
    is_fsm_jacvec_correct,
)
from pyrtid.inverse.adjoint.sensitivity import ForwardSensitivitySolver
from pyrtid.inverse.loss_function import eval_loss_ls
from pyrtid.inverse.model import InverseModel
from pyrtid.inverse.obs import (
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.inverse.params import (
    eval_weighted_loss_reg,
    get_parameters_bounds,
    get_parameters_values_from_model,
    update_model_with_parameters_values,
    update_parameters_from_model,
)
from pyrtid.utils import is_all_close
from pyrtid.utils.types import NDArrayFloat


def register_params_ds(params_ds: str):  # type: ignore
    """
    Add the given string to the __doc__attribute of the class.

    Parameters
    ----------
    params_ds : str
        String added to the parameters section.
    """

    def decorator(klass: Type):  # type: ignore
        """Decorate the klass."""
        klass.__doc__ += params_ds
        return klass

    return decorator


base_solver_config_params_ds = """
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
    is_fwd_verbose: bool
        Whether to solve the forward problem displaying all information.
        By default False.
    is_save_sp_mats: bool
        Whether to save all the stiffness matrices from flow and transport.
        This is mostly useful to check the adjoint state correctness and devs.
        The default is False.
        """


@register_params_ds(base_solver_config_params_ds)
@dataclass
class BaseSolverConfig:
    """
    Base class for solver configuration.

    Attributes
    ----------
    """

    hm_end_time: Optional[float] = None
    is_parallel: bool = False
    max_workers: int = 2
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = (
        np.random.default_rng(198873)
    )
    is_fwd_verbose: bool = False
    is_save_spmats: bool = False


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

    __slots__ = ["obs", "s_init", "_cov_obs"]

    def __init__(
        self,
        obs: NDArrayFloat,
        s_init: NDArrayFloat,
        cov_obs: NDArrayFloat,
    ) -> None:
        """Construct the instance."""
        self.obs = obs
        self.s_init = s_init
        self.cov_obs = cov_obs

    def is_ensemble(self) -> bool:
        """Return whether the optimization is performed over an ensemble."""
        return len(self.s_init.shape) == 2

    @property
    def n_ensemble(self):
        """Return the length of the parameters vector."""
        if not self.is_ensemble():
            return 1
        return self.s_init.shape[1]  # type: ignore

    @property
    def s_dim(self):
        """Return the length of the parameters vector."""
        return self.s_init.shape[0]  # type: ignore

    @property
    def d_dim(self):
        """Return the number of observations / forecast data."""
        # alias for n_obs
        return self.obs.shape[0]  # type: ignore

    @property
    def n_obs(self):
        """Return the number of observations / forecast data."""
        # alias for d_dim
        return self.d_dim  # type: ignore

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
    Base class Executor for assisted inversion.

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
        self._adj_model = None
        self.fwd_model: ForwardModel = fwd_model
        self.inv_model: InverseModel = inv_model
        self.pre_run_transformation: Optional[Callable] = pre_run_transformation
        self.solver_config = solver_config

        # Update parameters (only if the values haven't been defined for the parameters)
        update_parameters_from_model(fwd_model, self.inv_model.parameters_to_adjust)

        # Adjust the hm_end_time to match the end of the simulation
        if self.solver_config.hm_end_time is not None:
            self.solver_config.hm_end_time = min(
                self.solver_config.hm_end_time, fwd_model.time_params.duration
            )
        else:
            self.solver_config.hm_end_time = fwd_model.time_params.duration

        # Get the initial values (1D) from the model (preconditioned and parametrized)
        _s_init_model = get_parameters_values_from_model(
            fwd_model, inv_model.parameters_to_adjust, is_preconditioned=True
        )

        if s_init is not None:
            # check the dimensions and the bounds (clip to bounds + raise warning)
            _s_init = self.validate_s_init(s_init, _s_init_model.size)
        else:
            _s_init = _s_init_model

        # Need to differentiate flux and grids
        self.data_model = DataModel(self.obs, _s_init, self.std_obs**2)

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
        self._adj_model: Optional[AdjointModel] = adj_model

    def _init_adjoint_model(
        self,
        afpi_eps: float,
        is_numerical_acceleration: bool = False,
        is_use_continuous_adj: bool = False,
    ) -> None:
        """Initialize a new adjoint model for the executor."""
        self.adj_model = AdjointModel(
            self.fwd_model.geometry,
            self.fwd_model.time_params,
            self.fwd_model.fl_model.is_gravity,
            self.fwd_model.tr_model.n_sp,
            afpi_eps,
            is_numerical_acceleration,
            is_use_continuous_adj=is_use_continuous_adj,
        )

    @abstractmethod
    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

    @abstractmethod
    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "unknown"

    def get_display_dict(self) -> Dict[str, Any]:
        return {}

    def _initial_display(self) -> None:
        """Display basic info about the assimilation."""
        DISPLAY_TOP_LEN = 80
        DISPLAY_SHIFT = 50

        logging.info(f"{' Inversion Parameters ':=^{DISPLAY_TOP_LEN}}")

        # display specific to the solver
        display_dict = {
            "Method": self._get_solver_name(),
            "": "",  # space
            "Number of adjusted parameters": len(self.inv_model.parameters_to_adjust),
            "Number of unknowns (adjusted values)": self.inv_model.nb_adjusted_values,
            "Number of observables": len(self.inv_model.observables),
            "Number of observation data points (values)": self.inv_model.nb_obs_values,
            **self.get_display_dict(),
        }

        for k, v in display_dict.items():
            if k == "":
                logging.info("")
            else:
                logging.info(f"{k: <{DISPLAY_SHIFT}}: {v}")

        # End of display
        logging.info(f"{'':=^{DISPLAY_TOP_LEN}}")

    def _run_forward_model(
        self,
        s_cond: NDArrayFloat,
        run_n: int,
        is_save_state: bool = True,
        is_verbose: bool = False,
    ) -> NDArrayFloat:
        """
        Run the forward model and returns the prediction vector.

        Parameters
        ----------
        s_cond : np.array
            Conditioned parameter values as a 1D vector.
        run_n: int
            Run number.
        is_save_state: bool
            Whether the parameter values must be stored or not.
            The default is True.
        is_verbose: bool
            Whether to display info. The default is False.

        Returns
        -------
        d_pred: np.array
            Vector of results matching the observations.

        """
        logging.info("- Running forward model # %s", run_n)

        # Update the model with the new values of x (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            s_cond,
            self.inv_model.parameters_to_adjust,
            is_preconditioned=True,
            is_to_save=is_save_state,  # This is not finite differences
        )

        # Apply user transformation is needed:
        if self.pre_run_transformation is not None:
            self.pre_run_transformation(self.fwd_model)

        self.fwd_model.fl_model.is_save_spmats = self.solver_config.is_save_spmats
        self.fwd_model.tr_model.is_save_spmats = self.solver_config.is_save_spmats

        # Solve the forward model with the new parameters
        ForwardSolver(self.fwd_model).solve(
            is_verbose=is_verbose or self.solver_config.is_fwd_verbose
        )

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

    def _map_forward_model(self, s_ensemble: NDArrayFloat) -> NDArrayFloat:
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
        d_pred: NDArrayFloat = np.zeros([self.data_model.d_dim, n_ensemble])
        if self.solver_config.is_parallel:
            with ProcessPoolExecutor(
                max_workers=self.solver_config.max_workers
            ) as executor:
                results: Iterator[NDArrayFloat] = executor.map(
                    self._run_forward_model,
                    s_ensemble.T,
                    range(run_n + 1, run_n + n_ensemble + 1),  # type: ignore
                )
                for j, res in enumerate(results):
                    d_pred[:, j] = res
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
            loss_ls = eval_loss_ls(
                get_observables_values_as_1d_vector(
                    self.inv_model.observables, self.solver_config.hm_end_time
                ),
                d_pred[:, i],
                get_observables_uncertainties_as_1d_vector(
                    self.inv_model.observables, self.solver_config.hm_end_time
                ),
            )

            # Apply the scaling coefficient
            loss_scaled = loss_ls * self.inv_model.get_loss_function_scaling_factor(
                loss_ls, is_first_call=self.inv_model.nb_f_calls == 0
            )

            self.inv_model.loss_history.append(loss_scaled)

        return d_pred  # shape (N_obs, N_e)

    def eval_loss(
        self, s_cond: NDArrayFloat, is_save_state: bool = True, is_verbose: bool = False
    ) -> float:
        """
        Compute the model loss function.

        Parameters
        ----------
        is_verbose: bool
            Whether to display info. The default is False.
        """

        d_obs = get_observables_values_as_1d_vector(
            self.inv_model.observables, max_obs_time=self.solver_config.hm_end_time
        )

        loss_ls = eval_loss_ls(
            d_obs,
            self._run_forward_model(
                s_cond,
                self.inv_model.nb_f_calls + 1,
                is_save_state=is_save_state,
                is_verbose=is_verbose or self.solver_config.is_fwd_verbose,
            ),
            get_observables_uncertainties_as_1d_vector(
                self.inv_model.observables, max_obs_time=self.solver_config.hm_end_time
            ),
        )

        # Regularization part
        loss_reg: float = eval_weighted_loss_reg(
            self.inv_model.parameters_to_adjust,
            self.fwd_model,
            s_cond=s_cond,
            is_save_reg_state=is_save_state,
        )

        # Total non scaled (LS + REG)
        loss_total = loss_ls + loss_reg

        # Apply the scaling coefficient
        loss_total_scaled = (
            loss_total
            * self.inv_model.get_loss_function_scaling_factor(
                loss_total, is_first_call=self.inv_model.nb_f_calls == 0
            )
        )

        # Store the last objective function values (ls and reg terms)
        # not scaled
        self.inv_model.loss_ls_unscaled = loss_ls
        self.inv_model.loss_reg_unscaled = loss_reg

        logging.info(f"Loss (obs fit)        = {loss_ls}")
        logging.info(f"Loss (obs fit) / Nobs = {loss_ls / d_obs.size}")
        logging.info(f"Loss (weighted reg)   = {loss_reg}")
        logging.info(f"Total loss            = {loss_total}")
        logging.info(f"Scaling factor        = {self.inv_model.scaling_factor}")
        logging.info(f"Loss (scaled)         = {loss_total_scaled}\n")

        # Save the loss and the associated regularization weight
        if is_save_state:
            self.inv_model.loss_ls_history.append(loss_ls)
            self.inv_model.loss_reg_weighted_history.append(loss_reg)
            self.inv_model.loss_history.append(loss_total)

        return loss_total

    @abstractmethod
    def run(self) -> Optional[Union[Sequence[Any], NDArrayFloat]]:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        self._initial_display()

        # clear models and all
        self.inv_model.clear_history()

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

    def validate_s_init(
        self, s_init: NDArrayFloat, expected_s_dim: int
    ) -> NDArrayFloat:
        """Check if s init has the correct size."""
        if s_init.size == expected_s_dim:
            s_init = s_init.ravel("F")
        if s_init.ndim != 2 and s_init.shape[0] != expected_s_dim:  # type: ignore
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


adjoint_solver_config_params_ds = """is_check_gradient: bool
        Whether the gradient is checked by finite difference. The default is False.
    is_use_adjoint: bool
        Whether to use the adjoint for the gradient calculation. The default is True.
    afpi_eps: float
        Epsilon for the adjoint fixed point iterations. The default is 1e-5.
    is_adj_numerical_acceleration: bool
        Whether to use numerical acceleration in the adjoint state.
        The default is False.
    is_adj_verbose: bool = False
        Whether to solve the adjoint problem displaying information.
        By default False.
    is_use_continuous_adj: bool
        Whether to use numerical acceleration in the adjoint state.
        The default is False.

        """


@register_params_ds(adjoint_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class AdjointSolverConfig(BaseSolverConfig):
    r"""
    Configuration for solvers using the adjoint state model to compute the gradient.

    Parameters
    ----------
    """

    is_check_gradient: bool = False
    is_use_adjoint: bool = True
    afpi_eps: float = 1e-5
    max_nafpi: int = 50
    is_adj_numerical_acceleration: bool = False
    is_adj_verbose: bool = False
    is_use_continuous_adj: bool = False


_AdjointSolverConfig = TypeVar("_AdjointSolverConfig", bound=AdjointSolverConfig)


class AdjointInversionExecutor(BaseInversionExecutor, Generic[_AdjointSolverConfig]):
    """Represent a inversion executor instance using the adjoint state from PyRTID."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Careful, s_init is supposed to be preconditioned."""
        # super()._init_solver(s_init)

        # Create an adjoint model only if needed
        self.adj_model = None
        if self.solver_config.is_use_adjoint:
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_adj_numerical_acceleration,
                self.solver_config.is_use_continuous_adj,
            )

    def is_adjoint_gradient_correct(
        self,
        eps: Optional[float] = None,
        accuracy: int = 0,
        max_workers: int = 1,
        is_verbose: bool = False,
        max_nafpi: int = 30,
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
        accuracy : int, optional
            Number of points to use for the finite difference approximation.
            Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
            3 (8 points). The default is 0 which corresponds to the central
            difference scheme (2 points).
        max_workers: int
            Number of workers used for the gradient approximation by finite
            differences. If different from one, the calculation relies on
            multi-processing to decrease the computation time. The default is 1.
        is_verbose : bool, optional
            Whether to display computation infrmation, by default False
        max_nafpi: int
            Maximum number of iteration per adjoint chemistry-transport loop allowed
            (fixed point iterations.)
        """
        return is_adjoint_gradient_correct(
            self.fwd_model,
            self.adj_model,
            self.inv_model.parameters_to_adjust,
            self.inv_model.observables,
            eps=eps,
            accuracy=accuracy,
            max_workers=max_workers,
            hm_end_time=self.solver_config.hm_end_time,
            is_verbose=is_verbose,
            max_nafpi=max_nafpi,
        )

    def eval_loss_gradient(
        self, s_cond: NDArrayFloat, is_verbose: bool = False, is_save_state: bool = True
    ) -> NDArrayFloat:
        """
        Return the gradient of the objective function with regard to `x`.

        Parameters
        ----------
        s_cond: NDArrayFloat
            1D vector of inverted parameters.
        is_save_state; bool
            Whether to save objective functions and gradients. The default is True.

        Returns
        -------
        objective : NDArrayFloat
            The gradient vector. Note that the dimension is the same as for x.

        """
        # Update the number of times the gradient computation has been performed
        logging.info("- Running gradient # %s", self.inv_model.nb_g_calls + 1)

        adj_grad = np.array([], dtype=np.float64)
        fd_grad = np.array([], dtype=np.float64)
        if self.solver_config.is_use_adjoint or self.solver_config.is_check_gradient:
            if self.adj_model is not None:
                crank_flow = self.adj_model.a_fl_model.crank_nicolson
            else:
                crank_flow = None
            # Reinitialize the adjoint model
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_adj_numerical_acceleration,
                self.solver_config.is_use_continuous_adj,
            )
            self.adj_model.a_fl_model.set_crank_nicolson(crank_flow)

            # Solve the adjoint system
            solver = AdjointSolver(self.fwd_model, self.adj_model)
            solver.solve(
                self.inv_model.observables,
                self.solver_config.hm_end_time,
                is_verbose=is_verbose or self.solver_config.is_adj_verbose,
                max_nafpi=self.solver_config.max_nafpi,
            )
            # Compute the gradient with the adjoint state method

            adj_grad = compute_adjoint_gradient(
                self.fwd_model,
                self.adj_model,
                self.inv_model.parameters_to_adjust,
                is_save_state,
            )

        if (
            not self.solver_config.is_use_adjoint
            or self.solver_config.is_check_gradient
        ):
            # Compute the gradient by finite difference
            fd_grad = compute_fd_gradient(
                self.fwd_model,
                self.inv_model.observables,
                self.inv_model.parameters_to_adjust,
                is_save_state,
            )

        if self.solver_config.is_check_gradient:
            if not is_all_close(adj_grad, fd_grad):
                logging.warning("The adjoint gradient is not correct!")
            else:
                logging.info("The adjoint gradient seems correct!")

        logging.info("- Gradient eval # %s over\n", self.inv_model.nb_g_calls + 1)

        if is_save_state:
            self.inv_model.nb_g_calls += 1

        if self.solver_config.is_use_adjoint:
            return adj_grad
        return fd_grad

    def _run_forward_model_with_adjoint(
        self,
        s_cond: NDArrayFloat,
        run_n: int,
        is_save_state: bool = True,
    ) -> Tuple[float, NDArrayFloat, NDArrayFloat]:
        """
        Run the forward model and returns the prediction vector.

        Note
        ----
        We do not apply scaling factor yet.

        Parameters
        ----------
        s_cond : np.array
            Inverted parameters preconditioned values as a 1D vector.
        run_n: int
            Run number.
        is_save_state: bool
            Whether the parameter values must be stored or not.
            The default is True.

        Returns
        -------
        loss_ls: float
            The data misfit part of the objective function (no regularization).
        d_pred: NDArrayFloat
            Vector of predictions matching the observations.
        gradient: NDArrayFloat
            Gradient of loss_ls w.r.t. the adjusted values.

        """
        logging.info("- Running forward model # %s", run_n)

        # Update the model with the new values of x (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            s_cond,
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

        loss_ls = eval_loss_ls(
            get_observables_values_as_1d_vector(self.inv_model.observables),
            d_pred,
            get_observables_uncertainties_as_1d_vector(self.inv_model.observables),
        )

        logging.info(f"- Forward model run # {run_n} over")
        logging.info(f"- Running gradient # {run_n}")

        adj_grad = np.array([], dtype=np.float64)
        if self.solver_config.is_use_adjoint or self.solver_config.is_check_gradient:
            if self.adj_model is not None:
                crank_flow = self.adj_model.a_fl_model.crank_nicolson
            else:
                crank_flow = None
            # Reinitialize the adjoint model
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_adj_numerical_acceleration,
                self.solver_config.is_use_continuous_adj,
            )
            self.adj_model.a_fl_model.set_crank_nicolson(crank_flow)

            # Solve the adjoint system
            solver = AdjointSolver(self.fwd_model, self.adj_model)
            solver.solve(self.inv_model.observables, self.solver_config.hm_end_time)
            # Compute the gradient with the adjoint state method
            adj_grad = compute_adjoint_gradient(
                self.fwd_model,
                self.adj_model,
                self.inv_model.parameters_to_adjust,
                is_save_state=is_save_state,
            )

        logging.info(f"- Gradient eval # {run_n} over\n")

        self._check_nans_in_predictions(d_pred, run_n)

        return loss_ls, d_pred, adj_grad

    # Add an option for the regularization term
    def _map_forward_model_with_adjoint(
        self,
        s_ensemble: NDArrayFloat,
        is_parallel: bool = False,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        r"""
        Return both predicted data and associated gradients for the ensemble.

        Function calling the non-linear observation model (forward_model)
        for all ensemble members and returning the predicted data for
        each ensemble member. this function is responsible for the creation of
        simulation folder etc.

        Note
        ----
        The objective function returned is without regularization, only with the
        observed data misfits.

        Parameters
        ----------
        s_ensemble : NDArrayFloat
            Array of shape :math:`(N_{s}, N_{e})` containing the ensemble of parameter
            realizations, with :math:`N_{s}` the number of adjusted values and,
            :math:`N_{e}` the number of members (realizations).
        is_parallel : bool, optional
            Whether to use multiprocessing, by default False

        Returns
        -------
        Tuple[NDArrayFloat, NDArrayFloat]
            The array of predictions of shape :math:`(N_{\mathrm{obs}}, N_{e})` and
            associated gradients of shape :math:`(N_{s}, N_{e})`,
            with :math:`N_{\mathrm{obs}}` the number of observations,
            :math:`N_{s}` the number of adjusted values and,
            :math:`N_{e}` the number of members (realizations).

        """
        run_n: int = self.inv_model.nb_f_calls
        n_ensemble: int = s_ensemble.shape[1]  # type: ignore
        d_pred: NDArrayFloat = np.zeros([self.data_model.d_dim, n_ensemble])
        # loss functions
        losses_array = np.zeros([n_ensemble])
        gradients: NDArrayFloat = np.zeros([self.data_model.s_dim, n_ensemble])
        if is_parallel:
            with ProcessPoolExecutor(
                max_workers=self.solver_config.max_workers
            ) as executor:
                results: Iterator[NDArrayFloat] = executor.map(
                    self._run_forward_model_with_adjoint,
                    s_ensemble.T,
                    range(run_n + 1, run_n + n_ensemble + 1),  # type: ignore
                )
                for j, res in enumerate(results):
                    losses_array[j], d_pred[:, j], gradients[:, j] = res
        else:
            for j in range(n_ensemble):  # type: ignore
                (
                    losses_array[j],
                    d_pred[:, j],
                    gradients[:, j],
                ) = self._run_forward_model_with_adjoint(
                    s_ensemble[:, j], run_n + j + 1, is_save_state=False
                )

        # The check is already done in Forward_model but nan can also be introduced
        # because of the stacking. So it is necessary to check
        self._check_nans_in_predictions(d_pred, run_n)

        return losses_array, d_pred, gradients


fsm_solver_config_params_ds = """is_check_jacvec: bool
        Whether the products between the jacobian and the rhs vectors computed by the
        FSM are checked by finite difference. The default is False.
    is_use_fsm: bool
        Whether to use the the FSM for the jacobian dot products with the rhs vectors.
        The default is True.

        """


@register_params_ds(fsm_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class FSMSolverConfig(BaseSolverConfig):
    r"""
    Configuration for solvers using the forward sensitivity method.

    Parameters
    ----------
    """

    is_check_jacvec: bool = False
    is_use_fsm: bool = True


_FSMSolverConfig = TypeVar("_FSMSolverConfig", bound=FSMSolverConfig)


class FSMInversionExecutor(BaseInversionExecutor, Generic[_FSMSolverConfig]):
    """Represent a inversion executor instance using the Forward Sensitivity Method."""

    def run_fsm(
        self,
        s_cond: NDArrayFloat,
        vecs: NDArrayFloat,
        run_n: int,
        is_save_state: bool = True,
        is_verbose: bool = False,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Run the forward model and returns the prediction vector.

        Parameters
        ----------
        s_cond : np.array
            Conditioned parameter values as a 1D vector.
        vecs: NDArrayFloat
            Ensemble of vectors to multiply with the Jacobian matrix.
            It must have shape ($N_s \times N_e$), $N_s$ being the number of values
            optimized and $N_e$ the number of vectors.
        run_n: int
            Run number.
        is_save_state: bool
            Whether the parameter values must be stored or not.
            The default is True.
        is_verbose: bool
            Whether to display info. The default is False.

        Returns
        -------
        d_pred: np.array
            Vector of results matching the observations.
        jacvecs: np.array
            Products between the Jacobian matrix ($N_{obs} \times N_s$) and the
            ensemble of vectors vecs.

        """
        logging.info("- Running the FSM # %s", run_n)

        # Assert that the size of vecs is correct
        assert vecs.shape[0] == self.data_model.s_dim

        # Update the model with the new values of x (preconditioned)
        update_model_with_parameters_values(
            self.fwd_model,
            s_cond,
            self.inv_model.parameters_to_adjust,
            is_preconditioned=True,
            is_to_save=is_save_state,  # This is not finite differences
        )

        # Apply user transformation is needed:
        if self.pre_run_transformation is not None:
            self.pre_run_transformation(self.fwd_model)

        # Solve the forward model with the new parameters and evaluate on the fly
        # the product between the Jacobian matrix (N_obs, N_s) and the given vectors.
        d_pred, jacvecs = ForwardSensitivitySolver(self.fwd_model).solve(
            observables=self.inv_model.observables,
            vecs=vecs,
            hm_end_time=self.solver_config.hm_end_time,
            is_verbose=is_verbose or self.solver_config.is_fwd_verbose,
        )

        # Save the predictions
        if is_save_state:
            self.inv_model.list_d_pred.append(d_pred)

        self._check_nans_in_predictions(d_pred, run_n)

        # Read the results at the observation well
        # Update the prediction vector for the parameters m(j)
        logging.info("- FSM run # %s over", run_n)

        return d_pred, jacvecs

    def is_fsm_jacobian_correct(
        self,
        eps: Optional[float] = None,
        accuracy: int = 0,
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
        vecs: NDArrayFloat
            Ensemble of vectors to multiply with the Jacobian matrix.
            It must have shape ($N_s \times N_e$), $N_s$ being the number of values
            optimized and $N_e$ the number of vectors.
        eps: float, optional
            The epsilon for the computation of the approximated gradient by finite
            difference. If None, it is automatically inferred. The default is None.
        accuracy : int, optional
            Number of points to use for the finite difference approximation.
            Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
            3 (8 points). The default is 0 which corresponds to the central
            difference scheme (2 points).
        max_workers: int
            Number of workers used for the gradient approximation by finite
            differences. If different from one, the calculation relies on
            multi-processing to decrease the computation time. The default is 1.
        is_verbose : bool, optional
            Whether to display computation infrmation, by default False
        """
        return self.is_fsm_jacvec_correct(
            np.eye(N=self.data_model.s_dim),  # identity matrix
            eps=eps,
            accuracy=accuracy,
            max_workers=max_workers,
            is_verbose=is_verbose,
        )

    def is_fsm_jacvec_correct(
        self,
        vecs: NDArrayFloat,
        eps: Optional[float] = None,
        accuracy: int = 0,
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
        vecs: NDArrayFloat
            Ensemble of vectors to multiply with the Jacobian matrix.
            It must have shape ($N_s \times N_e$), $N_s$ being the number of values
            optimized and $N_e$ the number of vectors.
        eps: float, optional
            The epsilon for the computation of the approximated gradient by finite
            difference. If None, it is automatically inferred. The default is None.
        accuracy : int, optional
            Number of points to use for the finite difference approximation.
            Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
            3 (8 points). The default is 0 which corresponds to the central
            difference scheme (2 points).
        max_workers: int
            Number of workers used for the gradient approximation by finite
            differences. If different from one, the calculation relies on
            multi-processing to decrease the computation time. The default is 1.
        is_verbose : bool, optional
            Whether to display computation infrmation, by default False
        """
        # Assert that the size of vecs is correct
        assert vecs.shape[0] == self.data_model.s_dim

        return is_fsm_jacvec_correct(
            self.fwd_model,
            self.inv_model.parameters_to_adjust,
            self.inv_model.observables,
            vecs,
            eps=eps,
            accuracy=accuracy,
            max_workers=max_workers,
            hm_end_time=self.solver_config.hm_end_time,
            is_verbose=is_verbose,
        )
