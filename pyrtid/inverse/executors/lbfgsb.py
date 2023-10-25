"""
Implement an interface to solve the inverse problem with the L-BFGS-B from PyTRID.

PyRTID implements its own L-BFGS-B solver which is a pure python reimplementation of the
original fortran 778 algorithm (add references). The main difference lies in the
implementation of a procedure to update the objective function during the minimization
process. This experimental feature aims at adjusting the weight in front of
regularization during the optimization. Hence, the gradient sequence must be updated
to match the new function and obtain the correct Hessian approximation.

"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Deque, Tuple, Union

import numpy as np
from scipy.optimize import OptimizeResult as ScipyOptimizeResult

from pyrtid.inverse.adjoint import AdjointSolver
from pyrtid.inverse.adjoint.gradients import (
    compute_adjoint_gradient,
    compute_fd_gradient,
)
from pyrtid.inverse.executors.base import BaseInversionExecutor, BaseSolverConfig
from pyrtid.inverse.lbfgsb import minimize_lbfgsb
from pyrtid.inverse.params import (
    get_parameters_bounds,
    get_reg_loss_function,
    get_reg_loss_function_gradient,
)
from pyrtid.inverse.regularization import RegWeightUpdateStrategy
from pyrtid.utils import is_all_close
from pyrtid.utils.types import NDArrayFloat


@dataclass
class LBFGSBSolverConfig(BaseSolverConfig):
    r"""
    Configuration for the LBFGSB implemented in PyRTID.

    Note
    ----
    This configuration is strictly identical to the one implemented with Scipy. The
    only difference is that there is not solver name to provide.

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
        Weight for the regularization part. The default is "auto".
    afpi_eps: float
        The default is 1e-5.
    is_a_numerical_acceleratiion: bool
        The default is False.
    ftol_linesearch: float
        Specify a nonnegative tolerance for the sufficient decrease condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_1` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            f(\mathbf{x}_{k}+\alpha_{k}\mathbf{p}_{k})\leq
            f(\mathbf{x}_{k})+c_{1}\alpha_{k}\mathbf{p}_{k}^{\mathrm{T}}
            \nabla f(\mathbf{x}_{k})

        Note that :math:`0 < c_1 < 1`. Usually :math:`c_1` is small, see the Wolfe
        conditions in :cite:t:`nocedalNumericalOptimization1999`.
        In the fortran implementation
        algo 778, it is hardcoded to 1e-3. The default is 1e-4.
    gtol_linesearch: float
        Specify a nonnegative tolerance for the curvature condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_2` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            \left|\mathbf{p}_{k}^{\mathrm {T}}\nabla f(\mathbf{x}_{k}+\alpha_{k}
            \mathbf{p}_{k})\right|\leq c_{2}\left|\mathbf {p}_{k}^{\mathrm{T}}\nabla
            f(\mathbf{x}_{k})\right|

        Note that :math:`0 < c_1 < c_2 < 1`. Usually, :math:`c_2` is
        much larger than :math:`c_2`.
        see :cite:t:`nocedalNumericalOptimization1999`. In the fortran implementation
        algo 778, it is hardcoded to 0.9. The default is 0.9.
    max_steplength: float
        Maximum steplength allowed. The default is 1e8.
    xtol_linesearch: float
        Specify a nonnegative relative tolerance for an acceptable step in the line
        search procedure (see
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_).
        In the fortran implementation algo 778, it is hardcoded to 0.1.
        The default is 1e-5.

    """

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
    maxcor: int = 10
    gtol: float = 1e-5
    ftol: float = 1e-5
    max_iter: int = 50
    maxfun: int = 100
    iprint: int = -1
    maxls: int = 20
    ftol_linesearch: float = 1e-4
    gtol_linesearch: float = 0.9
    max_steplength: float = 1e8
    xtol_linesearch: float = 1e-5
    eps_SY: float = 2.2e-16


class LBFGSBInversionExecutor(BaseInversionExecutor[LBFGSBSolverConfig]):
    """Represent a inversion executor instance using the L-BFGS-B from PyRTID."""

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
        return "L-BFGS-B"

    def update_fun_def(
        self,
        x: NDArrayFloat,
        j: float,
        grad: NDArrayFloat,
        X: Deque[NDArrayFloat],
        G: Deque[NDArrayFloat],
    ) -> Tuple[float, NDArrayFloat, Deque[NDArrayFloat]]:
        """
        Update f0, grad and G to match a potential new objective function.

        Method to update the gradient sequence. This is an experimental feature to
        allow changing the objective function definition on the fly. In the first place
        this functionality is dedicated to regularized problems for which the
        regularization weight is computed while optimizing the cost function. In order
        to get a hessian matching the new definition of `fun`, the gradient sequence
        must be updated.

            ``update_fun_def(f0, grad, x_deque, grad_deque)
            -> f0, grad, updated grad_deque``

        Parameters
        ----------
        x: NDArrayFloat
            Last updated variables.
        j : float
            Value of the objective function for the last updated variables `x`.
        grad : NDArrayFloat
            Gradient of the objective function w.r.t. the last updated variables `x`.
        X : Deque[NDArrayFloat]
            Sequence of past updated variable vectors.
        G : Deque[NDArrayFloat]
            Sequence of gradients of the objective function w.r.t. to the past
            updated variable vectors.

        Returns
        -------
        Tuple[float, NDArrayFloat, Deque[NDArrayFloat]]
            Update f0, grad and G.
        """
        # Regularization weight that has been used up to now
        old_weight: float = self.inv_model.jreg_weight

        # Get the current regularization term of the objectuve function
        # And compute the new weight for the last X
        jreg: float = get_reg_loss_function(
            self.inv_model.parameters_to_adjust, self.fwd_model, x  # type: ignore
        )

        # Compute back the data misfit objective function
        # (remove the previous regularization term)
        # + we must take the scaling factor into account
        j0: float = j / self.inv_model.scaling_factor - old_weight * jreg

        # Check that it did not when wrong -> The objective function should be the
        # same (taking into account rounding errors)
        if (j0 - self.inv_model.ls_loss) / self.inv_model.ls_loss > 1e-8:
            raise RuntimeError(
                "Something went wrong while updating the objective function definition."
                f"\n - Saved weighted jreg = {self.inv_model.reg_loss}, "
                f"back-computed weighted jreg = {jreg * old_weight}."
                f"\nSaved j0 = {self.inv_model.ls_loss}, back-computed j0 = {j0}."
            )

        # Update the regularization weight
        new_weight: float = j0 / jreg
        self.inv_model.jreg_weight = new_weight
        # updated j (scaled with the scaling factor)
        updated_j: float = (j0 + new_weight * jreg) * self.inv_model.scaling_factor

        # Update all past gradients in G
        coef = new_weight - old_weight
        for x in X:
            # we remove the reg part with the old weight and add it back with the
            # new one
            # Note: get_reg_loss_function_gradient is very fast to obtain
            G.append(
                (
                    G.popleft() / self.inv_model.scaling_factor
                    + coef
                    * get_reg_loss_function_gradient(
                        self.inv_model.parameters_to_adjust, self.fwd_model, x
                    )
                )
                * self.inv_model.scaling_factor
            )

        # Check that all gradients have been correctly updated
        assert len(X) == len(G)

        return updated_j, grad, G

    def run(self) -> ScipyOptimizeResult:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        res: ScipyOptimizeResult = ScipyOptimizeResult()
        x0 = self.data_model.s_init

        # If AUTO_CONTINUOUS, rounds do not matter.
        self.inv_model.is_regularization_at_first_round = (
            self.solver_config.is_regularization_at_first_round
            or self.solver_config.reg_factor == RegWeightUpdateStrategy.AUTO_CONTINUOUS
        )

        n_iter = 0

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
            # get max_fun and max_iter
            maxfun = self._get_maxfun(
                self.solver_config,
                self.inv_model.nb_f_calls,
                self.inv_model.optimization_round_nb,
            )
            max_iter: int = self.solver_config.max_iter - n_iter

            # Experimental feature: update of the regularization weight at each
            # iteration
            if self.solver_config.reg_factor != RegWeightUpdateStrategy.AUTO_CONTINUOUS:
                update_fun_def = None
            else:
                update_fun_def = self.update_fun_def

            res = minimize_lbfgsb(
                x0=x0,
                fun=self.scaled_loss_function,
                jac=self.scaled_loss_function_gradient,  # type: ignore
                update_fun_def=update_fun_def,
                bounds=get_parameters_bounds(
                    self.inv_model.parameters_to_adjust, is_preconditioned=True
                ),
                maxcor=self.solver_config.maxcor,
                gtol=self.solver_config.gtol,
                ftol=self.solver_config.ftol,
                max_iter=max_iter,
                maxfun=maxfun,
                iprint=self.solver_config.iprint,
                maxls=self.solver_config.maxls,
                max_steplength=self.solver_config.max_steplength,
                ftol_linesearch=self.solver_config.ftol_linesearch,
                gtol_linesearch=self.solver_config.gtol_linesearch,
                xtol_linesearch=self.solver_config.xtol_linesearch,
                eps_SY=self.solver_config.eps_SY,
            )
            # The output parameter vector becomes the input
            x0 = res.x
            n_iter += res.nit
        return res

    @staticmethod  # type: ignore
    def _get_maxfun(solver_config: LBFGSBSolverConfig, nfev: int, round: int) -> int:
        """Update optimization stop criteria."""

        maxfun: int = solver_config.maxfun
        if solver_config.max_optimization_round_nb == 1:
            return maxfun
        if round == 1:
            return min(
                solver_config.max_fun_first_round,
                solver_config.max_fun_per_round,
                maxfun - nfev,
            )
        return min(solver_config.max_fun_per_round, maxfun - nfev)

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
