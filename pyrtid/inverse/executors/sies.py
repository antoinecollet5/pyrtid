"""
Implement the interface for the SIES inversion executor.

SIES stands for Sequential Iterative Approach.

This inversion executor gives access to the iterative_ensemble_smoother implementation
to solve the inverse problem. The original code is provided and maintained *
by Equinor and can be found at: https://github.com/equinor/iterative_ensemble_smoother

References
----------
Geir Evensen, et al. Efficient
implementation of an iterative ensemble smoother for data assimilation and reservoir
history matching. Frontiers in Applied Mathematics and Statistics, 5:47, 2019.
URL: https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full.

"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
from iterative_ensemble_smoother import SIES, steplength_exponential

from pyrtid.forward import ForwardSolver
from pyrtid.inverse.executors.base import BaseInversionExecutor, BaseSolverConfig
from pyrtid.inverse.loss_function import ls_loss_function
from pyrtid.inverse.obs import (
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
)
from pyrtid.inverse.params import update_model_with_parameters_values
from pyrtid.utils.types import NDArrayFloat


class SiesInversionType(str, Enum):
    """Inversion type for the computation of (S @ S.T + E @ E.T)^-1.

    Note
    ----
    It is a hashable string enum and can be iterated.
    """

    NAIVE = "naive"  # direct inversion
    EXACT = "exact"  # only if cdd is diagonal
    EXACT_R = "exact_r"  # for big data assimilation this is the recommended method
    SUBSPACE_RE = "subspace_re"  # using full Cdd

    def __str__(self) -> str:
        """Return instance value."""
        return self.value

    def __hash__(self) -> int:
        """Return the hash of the value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Return if two instances are equal."""
        if not isinstance(other, SiesInversionType) and not isinstance(other, str):
            return False
        return self.value == other

    @classmethod  # type: ignore
    def to_list(cls) -> List[SiesInversionType]:
        """Return all enums as a list."""
        return list(cls)


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
    inversion_method: SiesInversionType
        Type of inversion used. See :class:`SiesInversionType` for available types.
        The default is \"exact\".
    save_ensembles_history: bool, optional
        Whether to save the history predictions and parameters over
        the assimilations. The default is False.
    truncation: float
        How much of the total energy (singular values squared) to keep in the
        SVD when `inversion` equals `subspace_projected`. Choosing 1.0
        retains all information, while 0.0 removes all information.
        The default is 0.99.
    seed: Optional[int]
        Seed for the white noise generator used in the perturbation step.
        If None, the default :func:`numpy.random.default_rng()` is used.
        The default is None.
    steplength_strategy: Optional[Callable[[int, ...], float]]
        Callable which takes the iteration number as input and returns the steplength
        for the gauss-newton iteration (between 0 and 1.0). By default it uses an
        exponential strategy: (Eq. (49), which calculates a suitable step length for
        the update step, from the book: \"Formulating the history matching problem with
        consistent error statistics", written by :cite:t:`evensen2021formulating`.
    is_forecast_for_last_assimilation: bool, optional
        Whether to compute the predictions for the ensemble obtained at the
        last assimilation step. The default is True.
    is_use_adjoint: bool = False
        Whether to use the adjoint state for the gradient computation. If not, the
        gradient is estimated from the ensemble following the initial
        formulation by Evensen (2019), and the implementation from Equinor.
    """

    n_iterations: int = 4
    inversion_method: SiesInversionType = SiesInversionType.EXACT
    save_ensembles_history: bool = False
    truncation: float = 0.99
    seed: Optional[int] = None
    steplength_strategy: Callable[[int], float] = steplength_exponential
    is_forecast_for_last_assimilation: bool = True
    is_use_adjoint: bool = False
    # TODO: do we keep this or not ?
    # This is for the experimental feature with the gradient update
    reg_factor: Union[float, str] = "auto"
    afpi_eps: float = 1e-5
    is_a_numerical_acceleratiion: bool = False


class _SIES(SIES):
    """Wrapper for the SIES class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the instance."""
        super().__init__(*args, **kwargs)
        self.d_history: List[NDArrayFloat] = []
        self.s_history: List[NDArrayFloat] = []


class SIESInversionExecutor(BaseInversionExecutor[SIESSolverConfig]):
    """Ensemble Smoother with Multiple Data Assimilation Inversion Executor."""

    solver: _SIES

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Initiate a solver with its args."""

        self.solver: _SIES = _SIES(
            s_init,
            self.data_model.cov_obs,
            self.data_model.obs,
            inversion=self.solver_config.inversion_method,
            truncation=self.solver_config.truncation,
            seed=self.solver_config.seed,
        )

        # Create an adjoint model only if needed
        self.adj_model = None
        if self.solver_config.is_use_adjoint:
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_a_numerical_acceleratiion,
            )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "SIES"

    def run(self, s_init: NDArrayFloat) -> NDArrayFloat:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        _m = s_init.copy()
        if self.solver_config.save_ensembles_history:
            self.solver.s_history.append(_m)
        for iteration in range(self.solver_config.n_iterations):  # type: ignore
            logging.info(f"Iteration # {iteration}")
            d_pred, gradients = self._map_forward_model_with_adjoint(
                _m,
                is_parallel=self.solver_config.is_parallel,
                is_use_adjoint=self.solver_config.is_use_adjoint,
            )
            self.solver.d_history.append(d_pred)

            _m = self.solver.sies_iteration(
                d_pred.T,
                step_length=self.solver_config.steplength_strategy(iteration),
                # gradient_ensemble=(
                #     gradients.T if self.solver_config.is_use_adjoint.T else None
                # ),
            )

            if self.solver_config.save_ensembles_history:
                self.solver.s_history.append(_m)
        return _m

    @property
    def s_history(self) -> List[NDArrayFloat]:
        """Return the successive ensembles."""
        return self.solver.s_history

    def _run_forward_model_with_adjoint(
        self,
        m: NDArrayFloat,
        run_n: int,
        is_save_state: bool = True,
        is_use_adjoint: bool = False,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
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

        # AdjointModel()
        gradient = np.zeros((self.data_model.s_dim), dtype=np.float64)

        # Save the predictions
        if is_save_state:
            self.inv_model.list_d_pred.append(d_pred)

        self._check_nans_in_predictions(d_pred, run_n)

        # Read the results at the observation well
        # Update the prediction vector for the parameters m(j)
        logging.info("- Run # %s over", run_n)

        return d_pred, gradient

    def _map_forward_model_with_adjoint(
        self,
        s_ensemble: NDArrayFloat,
        is_parallel: bool = False,
        is_use_adjoint: bool = False,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
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
        n_ensemble: int = s_ensemble.shape[0]  # type: ignore
        d_pred: NDArrayFloat = np.zeros([n_ensemble, self.data_model.d_dim])
        gradients: NDArrayFloat = np.zeros([n_ensemble, self.data_model.s_dim])
        if is_parallel:
            with ProcessPoolExecutor(
                max_workers=self.solver_config.max_workers
            ) as executor:
                results: Iterator[NDArrayFloat] = executor.map(
                    self._run_forward_model_with_adjoint,
                    s_ensemble,
                    range(run_n + 1, run_n + n_ensemble + 1),  # type: ignore
                )
                for j, res in enumerate(results):
                    d_pred[j, :], gradients[j, :] = res
            # self.simu_n += n_ensemble
        else:
            for j in range(n_ensemble):  # type: ignore
                d_pred[j, :], gradients[j, :] = self._run_forward_model_with_adjoint(
                    s_ensemble[j, :], run_n + j + 1
                )
        # update the number of runs

        # The check is already done in Forward_model but nan can also be introduced
        # because of the stacking. So it is necessary to check
        self._check_nans_in_predictions(d_pred, run_n)

        # save objective functions. This should be very fast.
        for i in range(d_pred.shape[0]):  # type: ignore
            ls_loss = ls_loss_function(
                d_pred[i, :],
                get_observables_values_as_1d_vector(
                    self.inv_model.observables, self.solver_config.hm_end_time
                ),
                get_observables_uncertainties_as_1d_vector(
                    self.inv_model.observables, self.solver_config.hm_end_time
                ),
            )
            self.inv_model.list_f_res.append(ls_loss)

        return d_pred, gradients
