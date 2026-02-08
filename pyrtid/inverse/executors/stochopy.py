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
from dataclasses import dataclass
from typing import Any, Dict, Optional

from stochopy.optimize import OptimizeResult as StochpyOptimizeResult
from stochopy.optimize import minimize as stochopy_minimize
from typing_extensions import Literal

from pyrtid.inverse.executors.base import (
    BaseInversionExecutor,
    BaseSolverConfig,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import get_parameters_bounds
from pyrtid.utils import NDArrayFloat

stochopy_solver_config_params_ds = """
    solver_name: Literal["cmaes", "cpso", "de", "na", "pso", "vdcma"]
        The default is "cmaes".
    solver_options: Optional[Dict[str, Any]]
        The default is None.
    max_optimization_round_nb: int
        The default is 1.
    max_fun_per_round: int
        The default is 5.
"""


@register_params_ds(stochopy_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class StochopySolverConfig(BaseSolverConfig):
    """_summary_

    Parameters
    ----------
    """

    # TODO: add other parameters names
    solver_name: Literal["cmaes", "cpso", "de", "na", "pso", "vdcma"] = "cmaes"
    solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_per_round: int = 5


class StochopyInversionExecutor(BaseInversionExecutor[StochopySolverConfig]):
    """Represent a inversion executor instance using stochopy's solvers."""

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """Careful, s_init is supposed to be preconditioned."""
        super()._init_solver(s_init)

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
        x0 = self.data_model.s_init

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
                self.eval_loss,
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
