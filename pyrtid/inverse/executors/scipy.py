# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Implement the interface for the scipy inversion executor.

See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scipy.optimize import HessianUpdateStrategy
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
from pyrtid.utils import NDArrayFloat

scipy_solver_config_params_ds = r"""solver_name: str
        Name of the solver to use (the ``method`` argument of
        :func:`scipy.optimize.minimize`), e.g. ``"L-BFGS-B"``, ``"TNC"``,
        ``"SLSQP"``, ``"trust-constr"``, ``"BFGS"``, ``"CG"``, ``"Powell"``,
        ``"Nelder-Mead"``. See the notes of :func:`scipy.optimize.minimize` for the
        complete list and the options available for each of them.
        The default is "L-BFGS-B".
    solver_options: Optional[Dict[str, Any]]
        Kwargs for the ``options`` argument of :func:`scipy.optimize.minimize`
        (method specific). It is the escape hatch for every option that does not
        have a dedicated field below. If a dedicated field below is not None, it
        takes precedence over the entry of the same name in ``solver_options``.
        The default is None.
    hess: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]]
        Method for computing the Hessian matrix (``hess`` argument of
        :func:`scipy.optimize.minimize`), only used by some methods (Newton-CG,
        dogleg, trust-ncg, trust-krylov, trust-exact, trust-constr). PyRTID does not
        provide a Hessian, so this must be a string (``"2-point"``, ``"3-point"``,
        ``"cs"``), a :class:`scipy.optimize.HessianUpdateStrategy` or a user
        callable. The default is None.
    hessp: Optional[Callable]
        Hessian of the objective function times an arbitrary vector (``hessp``
        argument of :func:`scipy.optimize.minimize`). The default is None.
    constraints: Union[Dict, Sequence[Dict], scipy.optimize.LinearConstraint,
            scipy.optimize.NonlinearConstraint]
        Constraints definition passed to :func:`scipy.optimize.minimize`. Only used
        by COBYLA, COBYQA, SLSQP and trust-constr. The default is ``()`` (none).
        Parameters bounds are always taken from the adjusted parameters.
    tol: Optional[float]
        Tolerance for termination (``tol`` argument of
        :func:`scipy.optimize.minimize`). For detailed control, use the dedicated
        tolerances below or ``solver_options``. The default is None.
    callback: Optional[Callable]
        Callback called after each iteration (``callback`` argument of
        :func:`scipy.optimize.minimize`). The default is None.
    maxiter: Optional[int]
        Maximum number of iterations (option ``maxiter``). If None, the solver
        default is used. The default is None.
    maxfun: Optional[int]
        Maximum number of function evaluations (option ``maxfun``). It is the total
        budget shared by all optimization rounds. If None, the value in
        ``solver_options`` is used, and finally 15000. The default is None.
    ftol: Optional[float]
        Function value tolerance (option ``ftol``, e.g. L-BFGS-B, TNC, SLSQP,
        Powell). The default is None (solver default).
    gtol: Optional[float]
        Gradient tolerance (option ``gtol``, e.g. L-BFGS-B, TNC, BFGS, CG,
        trust-constr). The default is None (solver default).
    xtol: Optional[float]
        Step tolerance (option ``xtol``, e.g. TNC, Powell, trust-constr).
        The default is None (solver default).
    maxcor: Optional[int]
        Maximum number of variable metric corrections (option ``maxcor``, L-BFGS-B).
        The default is None (solver default).
    maxls: Optional[int]
        Maximum number of line search steps (option ``maxls``, L-BFGS-B).
        The default is None (solver default).
    eps: Optional[float]
        Absolute step size for the finite difference approximation of the gradient
        (option ``eps``). Not used when the gradient is provided, which is always
        the case here. The default is None (solver default).
    finite_diff_rel_step: Optional[float]
        Relative step size for the finite difference approximation (option
        ``finite_diff_rel_step``). Same remark as ``eps``.
        The default is None (solver default).
    disp: Optional[bool]
        Whether to print convergence messages (option ``disp``).
        The default is None (solver default).
    max_optimization_round_nb: int
        Maximum number of optimization rounds. The default is 1.
    max_fun_first_round: int
        The maximum number of forward computation in the first optimization round.
        THe default is 5.
    max_fun_per_round: int
        The number of function evaluation before a new round  starts.
        The default is 5.
    is_regularization_at_first_round: bool = True
        Whether to apply regularization at first round. The default is True.
"""


@register_params_ds(scipy_solver_config_params_ds)
@register_params_ds(adjoint_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class ScipySolverConfig(AdjointSolverConfig):
    """
    Configuration for Scipy solvers.

    All the arguments of :func:`scipy.optimize.minimize` that can be tuned (all
    except the objective function, its gradient, the initial guess, the extra
    ``args`` and the ``bounds`` which are handled by the executor) are exposed.

    Parameters
    ----------
    """

    solver_name: str = "L-BFGS-B"
    solver_options: dict[str, Any] | None = None
    hess: Callable | str | HessianUpdateStrategy | None = None
    hessp: Callable | None = None
    constraints: Any = ()
    tol: float | None = None
    callback: Callable | None = None
    maxiter: int | None = None
    maxfun: int | None = None
    ftol: float | None = None
    gtol: float | None = None
    xtol: float | None = None
    maxcor: int | None = None
    maxls: int | None = None
    eps: float | None = None
    finite_diff_rel_step: float | None = None
    disp: bool | None = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int = 5
    is_regularization_at_first_round: bool = True


class ScipyInversionExecutor(AdjointInversionExecutor[ScipySolverConfig]):
    """Represent a inversion executor instance using scipy's solvers."""

    #: Typed options of :class:`ScipySolverConfig` forwarded in ``options`` when set.
    OPTION_FIELDS: tuple[str, ...] = (
        "maxiter",
        "ftol",
        "gtol",
        "xtol",
        "maxcor",
        "maxls",
        "eps",
        "finite_diff_rel_step",
        "disp",
    )

    def _init_solver(self, s_init: NDArrayFloat) -> None:
        """
        Initiate the solver (and the adjoint model if needed).

        Parameters
        ----------
        s_init : NDArrayFloat
            Initial parameters vector. Careful, it is supposed to be
            preconditioned.
        """
        super()._init_solver(s_init)

        # Create an adjoint model only if needed
        self.adj_model = None
        if self.solver_config.is_use_adjoint:
            self._init_adjoint_model(
                self.solver_config.afpi_eps,
                self.solver_config.is_adj_numerical_acceleration,
                self.solver_config.is_use_continuous_adj,
            )

    def _get_solver_name(self) -> str:
        """
        Return the solver name.

        Returns
        -------
        str
            The scipy method name.
        """
        return self.solver_config.solver_name

    def get_display_dict(self) -> dict[str, Any]:
        """
        Return the solver specific information displayed at the start of the run.

        Returns
        -------
        dict[str, Any]
            Empty: no scipy specific information is displayed.
        """
        return {}

    def run(self) -> ScipyOptimizeResult:  # ty: ignore[invalid-method-override]
        """
        Run the history matching.

        The minimization is performed in one or several rounds (see
        ``max_optimization_round_nb``), the output of a round being the input of
        the next one.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result of the last optimization round.
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
            _options: dict[str, Any] = self._get_options_dict(
                self.solver_config,
                self.inv_model.nb_f_calls,
                self.inv_model.optimization_round_nb,
            )

            res = scipy_minimize(
                self.eval_loss,
                x0,
                bounds=get_parameters_bounds(
                    self.inv_model.parameters_to_adjust, is_preconditioned=True
                ),
                method=self.solver_config.solver_name,
                jac=self.eval_loss_gradient,
                hess=self.solver_config.hess,
                hessp=self.solver_config.hessp,
                constraints=self.solver_config.constraints,
                tol=self.solver_config.tol,
                callback=self.solver_config.callback,
                options=_options,
            )
            # The output parameter vector becomes the input
            x0 = res.x
        return res

    def _get_options_dict(
        self, solver_config: ScipySolverConfig, nfev: int, round: int
    ) -> dict[str, Any]:
        """
        Build the ``options`` dict and update the optimization stop criteria.

        Parameters
        ----------
        solver_config : ScipySolverConfig
            The solver configuration.
        nfev : int
            Number of function evaluations already performed.
        round : int
            Current optimization round number (starting at 1).

        Returns
        -------
        dict[str, Any]
            The options passed to :func:`scipy.optimize.minimize`. The typed fields
            of the configuration that are not None override ``solver_options``.
        """
        if solver_config.solver_options is not None:
            options = copy.deepcopy(solver_config.solver_options)
        else:
            options = {}

        for name in self.OPTION_FIELDS:
            value = getattr(solver_config, name)
            if value is not None:
                options[name] = value
        if solver_config.maxfun is not None:
            options["maxfun"] = solver_config.maxfun

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
