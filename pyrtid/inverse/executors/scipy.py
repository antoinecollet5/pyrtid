"""
Implement the interface for the scipy inversion executor.

See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from scipy.optimize import OptimizeResult as ScipyOptimizeResult
from scipy.optimize import minimize as scipy_minimize

from pyrtid.inverse.executors.base import (
    AdjointInversionExecutor,
    AdjointSolverConfig,
    adjoint_solver_config_params_ds,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import get_parameters_bounds
from pyrtid.inverse.regularization import RegWeightUpdateStrategy
from pyrtid.utils.types import NDArrayFloat

scipy_solver_config_params_ds = r"""solver_name: str = "L-BFGS-B"
        Name of the solver to use. TODO: point to scipy.
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int
        The number of function evaluation before a new round  starts.
    is_regularization_at_first_round: bool = True
    reg_factor: Union[float, RegWeightUpdateStrategy, str]
        Factor (weight) for the regularization term of the objective function.
        It supports float or automatic strategies. See the
        :class:`RegWeightUpdateStrategy` description for available strategies.
        The default is RegWeightUpdateStrategy.AUTO_PER_ROUND.
"""


@register_params_ds(scipy_solver_config_params_ds)
@register_params_ds(adjoint_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class ScipySolverConfig(AdjointSolverConfig):
    """
    Configuration for Scipy solvers.

    Parameters
    ----------
    """

    solver_name: str = "L-BFGS-B"
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int = 5
    is_regularization_at_first_round: bool = True
    reg_factor: Union[
        float, RegWeightUpdateStrategy, str
    ] = RegWeightUpdateStrategy.AUTO_PER_ROUND


class ScipyInversionExecutor(AdjointInversionExecutor[ScipySolverConfig]):
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
