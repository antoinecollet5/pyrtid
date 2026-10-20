# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Implement the interface for the stochopy inversion executor.

Stochopy is a collection of stochastic solvers.

This inversion executor gives access to the Stochopy implementation to solve the inverse
problem. The original code is provided and maintained by Keurfon Luu
and can be found at: https://github.com/keurfonluu/stochopy
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from stochopy.optimize import OptimizeResult as StochpyOptimizeResult
from stochopy.optimize import minimize as stochopy_minimize

from pyrtid.inverse.executors.base import (
    BaseInversionExecutor,
    BaseSolverConfig,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import get_parameters_bounds
from pyrtid.utils import NDArrayFloat

# Options (keyword arguments of the ``minimize`` function of each stochopy
# optimizer) available for each method, besides the generic ones.
_GENERIC_OPTIONS: tuple[str, ...] = (
    "maxiter",
    "popsize",
    "xtol",
    "ftol",
    "workers",
    "backend",
    "return_all",
    "verbosity",
)
#: Methods working with a population: ``x0`` must then be an initial population of
#: shape (popsize, ndim) while the others expect a single vector of shape (ndim,).
STOCHOPY_POPULATION_METHODS: tuple[str, ...] = ("cpso", "de", "na", "pso")

STOCHOPY_METHOD_OPTIONS: dict[str, tuple[str, ...]] = {
    "cmaes": (*_GENERIC_OPTIONS, "sigma", "muperc", "constraints"),
    "cpso": (
        *_GENERIC_OPTIONS,
        "inertia",
        "cognitivity",
        "sociability",
        "competitivity",
        "constraints",
        "updating",
    ),
    "de": (
        *_GENERIC_OPTIONS,
        "mutation",
        "recombination",
        "strategy",
        "constraints",
        "updating",
    ),
    "na": (*_GENERIC_OPTIONS, "nrperc"),
    "pso": (
        *_GENERIC_OPTIONS,
        "inertia",
        "cognitivity",
        "sociability",
        "constraints",
        "updating",
    ),
    "vdcma": (*_GENERIC_OPTIONS, "sigma", "muperc", "constraints"),
}

#: Typed configuration fields forwarded as stochopy options when they are not None.
STOCHOPY_OPTION_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(o for opts in STOCHOPY_METHOD_OPTIONS.values() for o in opts)
)

stochopy_solver_config_params_ds = """
    solver_name: Literal["cmaes", "cpso", "de", "na", "pso", "vdcma"]
        Name of the stochopy optimizer (the ``method`` argument of
        :func:`stochopy.optimize.minimize`). The default is "cmaes".
    solver_options: Optional[Dict[str, Any]]
        Kwargs for the ``options`` argument of :func:`stochopy.optimize.minimize`.
        It is the escape hatch for any option without a dedicated field below.
        If a dedicated field below is not None, it takes precedence over the entry
        of the same name in ``solver_options``. The default is None.
    max_optimization_round_nb: int
        Maximum number of optimization rounds. The default is 1.
    max_fun_per_round: int
        Maximum number of function evaluations per optimization round (converted
        to a maximum number of iterations: ``ceil(max_fun / popsize)``).
        The default is 5.
    callback: Optional[Callable]
        Callable called after each iteration with the signature
        ``callback(X, state)`` where ``X`` is the current population and ``state``
        a partial :class:`stochopy.optimize.OptimizeResult`. The default is None.
    seed: Optional[int]
        Seed of the optimizer random number generator. If None, the optimizer is
        not seeded. The default is None.
    maxiter: Optional[int]
        Maximum number of iterations (all methods). The solver default is 100.
        The default is None.
    popsize: Optional[int]
        Population size (all methods). The solver default is 10.
        The default is None.
    xtol: Optional[float]
        Solution tolerance for termination (all methods). The solver default is
        1e-8. The default is None.
    ftol: Optional[float]
        Function value tolerance for termination (all methods). The solver default
        is 1e-8. The default is None.
    sigma: Optional[float]
        Initial step size (cmaes, vdcma). The solver default is 0.1.
        The default is None.
    muperc: Optional[float]
        Number of parents/points for recombination, as a fraction of the population
        size (cmaes, vdcma). The solver default is 0.5. The default is None.
    inertia: Optional[float]
        Inertia weight (pso, cpso). The solver default is 0.7298.
        The default is None.
    cognitivity: Optional[float]
        Cognition parameter (pso, cpso). The solver default is 1.49618.
        The default is None.
    sociability: Optional[float]
        Sociability parameter (pso, cpso). The solver default is 1.49618.
        The default is None.
    competitivity: Optional[float]
        Competitivity parameter (cpso). The solver default is 1.0.
        The default is None.
    mutation: Optional[float]
        Mutation constant (de). The solver default is 0.5. The default is None.
    recombination: Optional[float]
        Recombination probability (de). The solver default is 0.9.
        The default is None.
    strategy: Optional[str]
        Differential evolution strategy (de). The solver default is "best1bin".
        The default is None.
    nrperc: Optional[float]
        Number of cells to resample, as a fraction of the population size
        (na, neighborhood algorithm). The solver default is 0.5.
        The default is None.
    constraints: Optional[str]
        Constraints handling method (all methods except na), e.g. "Penalty".
        The default is None.
    updating: Optional[str]
        Population updating strategy, "immediate" or "deferred" (pso, cpso, de).
        The solver default is "immediate". The default is None.
    workers: Optional[int]
        Number of parallel workers (all methods). The solver default is 1.
        The default is None.
    backend: Optional[str]
        Parallel backend, e.g. "thread" or "process" (all methods).
        The solver default is None. The default is None.
    return_all: Optional[bool]
        Whether to return the solutions at each iteration (all methods).
        The solver default is False. The default is None.
    verbosity: Optional[float]
        Fraction of the population to consider when ``return_all`` is True
        (all methods). The solver default is 1.0. The default is None.

    Notes
    -----
    A typed option that is not None but is not supported by the selected solver
    raises a :class:`ValueError` when the options are built.
"""


@register_params_ds(stochopy_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class StochopySolverConfig(BaseSolverConfig):
    """
    Configuration for the stochopy stochastic solvers.

    Every argument of :func:`stochopy.optimize.minimize` and of the options of
    its optimizers that can be tuned is exposed (the objective function, bounds,
    initial guess and extra ``args`` are handled by the executor). Options left
    to None are not passed and keep the stochopy default value.

    Parameters
    ----------
    """

    solver_name: Literal["cmaes", "cpso", "de", "na", "pso", "vdcma"] = "cmaes"
    solver_options: dict[str, Any] | None = None
    max_optimization_round_nb: int = 1
    max_fun_per_round: int = 5
    callback: Callable | None = None
    seed: int | None = None
    maxiter: int | None = None
    popsize: int | None = None
    xtol: float | None = None
    ftol: float | None = None
    sigma: float | None = None
    muperc: float | None = None
    inertia: float | None = None
    cognitivity: float | None = None
    sociability: float | None = None
    competitivity: float | None = None
    mutation: float | None = None
    recombination: float | None = None
    strategy: str | None = None
    nrperc: float | None = None
    constraints: str | None = None
    updating: str | None = None
    workers: int | None = None
    backend: str | None = None
    return_all: bool | None = None
    verbosity: float | None = None


class StochopyInversionExecutor(BaseInversionExecutor[StochopySolverConfig]):
    """Represent a inversion executor instance using stochopy's solvers."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """
        Initiate the solver.

        Parameters
        ----------
        s_init : NDArrayFloat
            Initial parameters vector. Careful, it is supposed to be
            preconditioned.
        """
        super()._init_solver(s_init)

    def _get_solver_name(self) -> str:
        """
        Return the solver name.

        Returns
        -------
        str
            The stochopy method name.
        """
        return self.solver_config.solver_name

    def run(self) -> StochpyOptimizeResult:  # ty: ignore[invalid-method-override]
        """
        Run the history matching.

        The minimization is performed in one or several rounds (see
        ``max_optimization_round_nb``), the output of a round being the initial
        guess of the next one.

        Returns
        -------
        stochopy.optimize.OptimizeResult
            The result of the last optimization round.
        """
        super().run()
        res = StochpyOptimizeResult()
        x0 = self.data_model.s_init

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
            _options: dict[str, Any] = self._get_options_dict(
                self.solver_config, self.inv_model.nb_f_calls
            )

            bounds = get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            )
            res = stochopy_minimize(
                self.eval_loss,
                bounds,
                x0=self._get_initial_guess(x0, bounds, _options),
                method=self.solver_config.solver_name,
                options=self._to_stochopy_options(_options),
                callback=self.solver_config.callback,
            )
            x0 = res.x
            # The output parameter vector becomes the input
        return res

    def _get_initial_guess(
        self, x0: NDArrayFloat, bounds: NDArrayFloat, options: dict[str, Any]
    ) -> NDArrayFloat:
        """
        Return the initial guess in the format expected by the selected solver.

        Parameters
        ----------
        x0 : NDArrayFloat
            Initial parameters vector, shape (ndim,).
        bounds : NDArrayFloat
            Parameters bounds, shape (ndim, 2).
        options : dict[str, Any]
            The stochopy options (``popsize`` is read from it).

        Returns
        -------
        NDArrayFloat
            ``x0`` itself for the methods using a single vector (cmaes, vdcma).
            For the population based methods (pso, cpso, de, na), an initial
            population of shape (popsize, ndim) made of ``x0`` and of
            ``popsize - 1`` members uniformly drawn within the bounds
            (using ``seed`` if given).
        """
        if self.solver_config.solver_name not in STOCHOPY_POPULATION_METHODS:
            return x0
        popsize = options.get("popsize", 10)
        rng = np.random.default_rng(self.solver_config.seed)
        bounds = np.asarray(bounds, dtype=float)
        population = rng.uniform(bounds[:, 0], bounds[:, 1], size=(popsize, len(x0)))
        population[0] = x0
        return population

    @staticmethod
    def _to_stochopy_options(options: dict[str, Any]) -> dict[str, Any]:
        """
        Convert the ``maxfun`` stop criterion to ``maxiter`` (stochopy >= 2).

        Parameters
        ----------
        options : dict[str, Any]
            Options, possibly containing ``maxfun`` (not supported by stochopy).

        Returns
        -------
        dict[str, Any]
            Copy of the options without ``maxfun``. If ``maxfun > 0``, ``maxiter``
            is lowered to ``ceil(maxfun / popsize)`` at most.
        """
        options = dict(options)
        maxfun = options.pop("maxfun", 0)
        if maxfun > 0:
            popsize = options.get("popsize", 10)
            options["maxiter"] = min(
                options.get("maxiter", 100), max(1, -(-maxfun // popsize))
            )
        return options

    def _get_options_dict(
        self,
        solver_config: StochopySolverConfig,
        nfev: int,
    ) -> dict[str, Any]:
        """
        Build the stochopy options and update the optimization stop criteria.

        Parameters
        ----------
        solver_config : StochopySolverConfig
            The solver configuration.
        nfev : int
            Number of function evaluations already performed.

        Returns
        -------
        dict[str, Any]
            The options (including a ``maxfun`` stop criterion). The typed fields
            of the configuration that are not None override ``solver_options``.

        Raises
        ------
        ValueError
            If a typed option is set but is not supported by the selected solver.
        """
        if solver_config.solver_options is not None:
            options = copy.deepcopy(solver_config.solver_options)
        else:
            options = {}

        allowed = STOCHOPY_METHOD_OPTIONS.get(solver_config.solver_name, ())
        for name in STOCHOPY_OPTION_FIELDS:
            value = getattr(solver_config, name)
            if value is None:
                continue
            if name not in allowed:
                raise ValueError(
                    f"Option '{name}' is not supported by the stochopy solver "
                    f"'{solver_config.solver_name}'. Supported options: {allowed}."
                )
            options[name] = value
        if solver_config.seed is not None:
            options["seed"] = solver_config.seed

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
