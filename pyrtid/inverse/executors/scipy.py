"""
Implement the interface for the scipy inversion executor.

See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult as ScipyOptimizeResult
from scipy.optimize import minimize as scipy_minimize

from pyrtid.inverse.adjoint import AdjointSolver
from pyrtid.inverse.adjoint.gradients import (
    compute_adjoint_gradient,
    compute_fd_gradient,
)
from pyrtid.inverse.executors.base import BaseInversionExecutor, BaseSolverConfig
from pyrtid.inverse.params import get_parameters_bounds
from pyrtid.inverse.regularization import RegWeightUpdateStrategy
from pyrtid.utils import is_all_close
from pyrtid.utils.types import NDArrayFloat


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
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
    solver_name: str = "L-BFGS-B"
        Name of the solver to use. TODO: point to scipy.
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int
        The number of function evaluation before a new round  starts.
    is_check_gradient: bool
        Whether the gradient The default is False.
    is_use_adjoint: bool = True
    is_regularization_at_first_round: bool = True
    reg_factor: Union[float, RegWeightUpdateStrategy, str]
        Factor (weight) for the regularization term of the objective function.
        It supports float or automatic strategies. See the
        :class:`RegWeightUpdateStrategy` description for available strategies.
        The default is RegWeightUpdateStrategy.AUTO_PER_ROUND.
    afpi_eps: float = 1e-5
    is_a_numerical_acceleratiion: bool = False

    """

    solver_name: str = "L-BFGS-B"
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int = 5
    is_check_gradient: bool = False
    is_use_adjoint: bool = True
    is_regularization_at_first_round: bool = True
    reg_factor: Union[
        float, RegWeightUpdateStrategy, str
    ] = RegWeightUpdateStrategy.AUTO_PER_ROUND
    afpi_eps: float = 1e-5
    is_a_numerical_acceleratiion: bool = False


class ScipyInversionExecutor(BaseInversionExecutor[ScipySolverConfig]):
    """Represent a inversion executor instance using scipy's solvers."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Careful, s_init is supposed to be preconditioned."""
        super()._init_solver(s_init)

        # Create an adjoint model only if needed
        self.adj_model = None
        if self.solver_config.is_use_adjoint:
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_a_numerical_acceleratiion,
            )

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
        x0 = self.data_model.s_init

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
                f"Entering optimization loop: {self.inv_model.optimization_round_nb}"
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
            if self.adj_model is not None:
                crank_flow = self.adj_model.a_fl_model.crank_nicolson
            else:
                crank_flow = None
            # Reinitialize the adjoint model
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_a_numerical_acceleratiion,
            )
            self.adj_model.a_fl_model.set_crank_nicolson(crank_flow)

            # Solve the adjoint system
            solver = AdjointSolver(self.fwd_model, self.adj_model)
            solver.solve(self.inv_model.observables, self.solver_config.hm_end_time)
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
