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

from pyrtid.inverse.executors.base import (
    AdjointInversionExecutor,
    AdjointSolverConfig,
    adjoint_solver_config_params_ds,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import (
    get_parameters_bounds,
    get_reg_loss_function,
    get_reg_loss_function_gradient,
)
from pyrtid.inverse.regularization import EnsembleRegularizator, RegWeightUpdateStrategy
from pyrtid.inverse.solvers.lbfgsb import minimize_lbfgsb
from pyrtid.utils.types import NDArrayFloat

lbfgsb_solver_config_params_ds = r"""solver_options: Optional[Dict[str, Any]] = None
    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int
        The number of function evaluation before a new round  starts.
    is_check_gradient: bool
        Whether the gradient The default is False.
    is_use_adjoint: bool = True
    is_regularization_at_first_round: bool = True
    reg_factor: Union[float, RegWeightUpdateStrategy, str]
        TODO: change this.
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

# TODO: maybe instead of 2 solvers, make a single one but check the size of the
# given s_init ???


@register_params_ds(lbfgsb_solver_config_params_ds)
@register_params_ds(adjoint_solver_config_params_ds)
@register_params_ds(base_solver_config_params_ds)
@dataclass
class LBFGSBSolverConfig(AdjointSolverConfig):
    r"""
    Configuration for the LBFGSB implemented in PyRTID.

    Note
    ----
    This configuration is strictly identical to the one implemented with Scipy. The
    only difference is that there is not solver name to provide.

    Parameters
    ----------
    """

    max_optimization_round_nb: int = 1
    max_fun_first_round: int = 5
    max_fun_per_round: int = 5
    is_check_gradient: bool = False
    is_use_adjoint: bool = True
    is_regularization_at_first_round: bool = True
    n_fun_before_reg: int = 0
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


class LBFGSBInversionExecutor(AdjointInversionExecutor[LBFGSBSolverConfig]):
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
        return "L-BFGS-B (PyRTID)"

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

        if (
            old_weight == 0 and jreg == 0
        ) or self.solver_config.n_fun_before_reg > self.inv_model.nb_f_calls:
            return j, grad, G

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

        if new_weight == old_weight:
            return j, grad, G

        self.inv_model.jreg_weight = new_weight
        # updated j (scaled with the scaling factor)
        updated_j: float = (j0 + new_weight * jreg) * self.inv_model.scaling_factor

        # Update the current gradient
        coef = new_weight - old_weight
        updated_grad = (
            grad / self.inv_model.scaling_factor
            + coef
            * get_reg_loss_function_gradient(
                self.inv_model.parameters_to_adjust, self.fwd_model, x
            )
        ) * self.inv_model.scaling_factor
        # Update all past gradients in G
        for _x in X:
            # we remove the reg part with the old weight and add it back with the
            # new one
            # Note: get_reg_loss_function_gradient is very fast to obtain
            G.append(
                (
                    G.popleft() / self.inv_model.scaling_factor
                    + coef
                    * get_reg_loss_function_gradient(
                        self.inv_model.parameters_to_adjust, self.fwd_model, _x
                    )
                )
                * self.inv_model.scaling_factor
            )

        # Check that all gradients have been correctly updated
        assert len(X) == len(G)

        return updated_j, updated_grad, G

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
            maxfun = get_maxfun(
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


def get_maxfun(solver_config: LBFGSBSolverConfig, nfev: int, round: int) -> int:
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


class LBFGSBEnsembleInversionExecutor(AdjointInversionExecutor[LBFGSBSolverConfig]):
    """Represent a inversion executor instance using the L-BFGS-B from PyRTID."""

    nb_f_calls = 0

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "L-BFGS-B-E (PyRTID)"

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
        jreg: float = self.get_reg_loss_function(
            x.reshape(self.data_model.s_init.shape)
        )

        if (
            old_weight == 0 and jreg == 0
        ) or self.solver_config.n_fun_before_reg > self.inv_model.nb_f_calls:
            return j, grad, G

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

        if new_weight == old_weight:
            return j, grad, G

        self.inv_model.jreg_weight = new_weight
        # updated j (scaled with the scaling factor)
        updated_j: float = (j0 + new_weight * jreg) * self.inv_model.scaling_factor

        # Update the current gradient
        coef = new_weight - old_weight
        updated_grad = (
            grad / self.inv_model.scaling_factor
            + coef
            * self.get_reg_loss_function_gradient(
                x.reshape(self.data_model.s_init.shape)
            ).ravel()
        ) * self.inv_model.scaling_factor

        # Update all past gradients in G
        for _x in X:
            # we remove the reg part with the old weight and add it back with the
            # new one
            # Note: get_reg_loss_function_gradient is very fast to obtain
            G.append(
                (
                    G.popleft() / self.inv_model.scaling_factor
                    + coef
                    * self.get_reg_loss_function_gradient(
                        _x.reshape(self.data_model.s_init.shape)
                    ).ravel()
                )
                * self.inv_model.scaling_factor
            )

        # Check that all gradients have been correctly updated
        assert len(X) == len(G)

        return updated_j, updated_grad, G

    def get_reg_loss_function(self, s_ensemble) -> float:
        """


        Parameters
        ----------
        s_ensemble : _type_
            _description_

        Returns
        -------
        float
            _description_
        """
        jreg = 0
        for param in self.inv_model.parameters_to_adjust:
            for reg in param.regularizators:
                if isinstance(reg, EnsembleRegularizator):
                    jreg += reg.loss_function(s_ensemble)
        return jreg

    def get_reg_loss_function_gradient(self, s_ensemble) -> NDArrayFloat:
        """

        Parameters
        ----------
        s_ensemble : _type_
            _description_

        Returns
        -------
        NDArrayFloat
            _description_
        """
        reg_grad = np.zeros(s_ensemble.shape)
        for param in self.inv_model.parameters_to_adjust:
            for reg in param.regularizators:
                if isinstance(reg, EnsembleRegularizator):
                    reg_grad += reg.loss_function_gradient(
                        s_ensemble.reshape(self.data_model.s_init.shape)
                    )
        return reg_grad

    def update_jreg_weight(
        self,
        j0: float,
        jreg: float,
        reg_factor: Union[float, RegWeightUpdateStrategy, str],
        n_fun_before_reg: int,
    ) -> None:
        """
        Update the regularization weight.

        The regularization is ignored during the first optimization loop. Then
        the weights are automatically computed in the first call of each optimization
        loop.

        Parameters
        ----------
        j0 : float
            The data misfit objective function.
        reg_factor: Union[float, RegWeightUpdateStrategy, str]
            Factor (weight) for the regularization term of the objective function.
            It supports float or automatic strategies. See the
            :class:`RegWeightUpdateStrategy` description for available strategies.
            The default is RegWeightUpdateStrategy.AUTO_PER_ROUND.
        n_fun_before_reg: int
            The number of objective function evaluation to perform before adding the
            regularization term. This feature allows to start optimizing with
            the misfit part only (no risk of overfitting at the beginning).

        Returns
        -------
        jreg : float
            the regularization objective function.

        """
        if self.inv_model.optimization_round_nb == 1:
            if (
                not self.inv_model.is_regularization_at_first_round
                or n_fun_before_reg > self.nb_f_calls
            ):
                self.inv_model.jreg_weight = 0.0
                return

        if jreg == 0:
            self.inv_model.jreg_weight = 0.0
            return

        if self.inv_model.is_first_loss_function_call_in_round:
            if j0 == 0:
                self.inv_model.jreg_weight = 1.0
            elif reg_factor in [
                RegWeightUpdateStrategy.AUTO_PER_ROUND,
                RegWeightUpdateStrategy.AUTO_CONTINUOUS,
            ]:
                self.inv_model.jreg_weight = j0 / jreg
            else:
                self.inv_model.jreg_weight = float(reg_factor)
            self.inv_model.is_first_loss_function_call_in_round = False

    def scaled_loss_function(
        self, s_ensemble: NDArrayFloat, is_save_state: bool = True
    ) -> float:
        """
        Return the objective function and the gradient for the ensemble.

        Parameters
        ----------
        Parameters
        ----------
        s_ensemble : NDArrayFloat
            Array of shape :math:`(N_{s} \times N_{e})` containing the
            flatten ensemble of parameter
            realizations, with :math:`N_{s}` the number of adjusted values and,
            :math:`N_{e}` the number of members (realizations).
        is_save_state : bool, optional
            _description_, by default True

        Returns
        -------
        float
            _description_
        """
        return self._scaled_loss_function(
            s_ensemble.reshape(self.data_model.s_init.shape), is_save_state
        )

    def _scaled_loss_function(
        self, s_ensemble: NDArrayFloat, is_save_state: bool = True
    ) -> float:
        """
        Here s_ensemble has shape `(N_{s}, N_{e})` (one column per realization).
        """
        # reshape to (Ns, Ne) and run
        losses, dpred, gradients = self._map_forward_model_with_adjoint(
            s_ensemble, self.solver_config.is_parallel
        )

        # The objective function is the mean of member objectuive function (this is
        # the definition we have chosen).
        ls_loss: float = float(np.mean(losses))

        # Compute the regularization term:
        if self.solver_config.reg_factor == 0:
            reg_loss = 0.0
        else:
            reg_loss: float = self.get_reg_loss_function(s_ensemble)
            self.update_jreg_weight(
                ls_loss,
                reg_loss,
                self.solver_config.reg_factor,
                n_fun_before_reg=self.solver_config.n_fun_before_reg,
            )
            reg_loss *= self.inv_model.jreg_weight

            if self.inv_model.jreg_weight != 0:
                gradients += (
                    self.get_reg_loss_function_gradient(s_ensemble)
                    * self.inv_model.jreg_weight
                )

        print(gradients.shape)

        total_loss: float = ls_loss + reg_loss

        # Apply the scaling coefficient
        scaled_loss: float = (
            total_loss * self.inv_model.get_loss_function_scaling_factor(total_loss)
        )

        # Store the losses to the inverse model
        self.inv_model.list_losses += list(losses * self.inv_model.scaling_factor)

        # Need to store this locally because the mechanisms in inverse models
        self.nb_f_calls += 1

        # Store the last objective function values (ls and reg terms)
        self.inv_model.ls_loss = ls_loss
        self.inv_model.reg_loss = reg_loss

        logging.info(f"Loss (obs fit)        = {ls_loss}")
        logging.info(f"Loss (regularization) = {reg_loss}")
        logging.info(f"Regularization weight = {self.inv_model.jreg_weight}")
        logging.info(f"Scaling factor        = {self.inv_model.scaling_factor}")
        logging.info(f"Loss (scaled)         = {scaled_loss}\n")

        # Save the loss and the associated regularization weight
        if is_save_state:
            self.inv_model.list_losses.append(scaled_loss)

        # The loss_ls gradient is the stacking of all members gradient
        # (independent from each others).
        # In this specific case, the regularization imposed to the parameters
        # are ignored and we consider our own regularization approach.

        # TODO: add the regularization terms that might depend on the ensemble.
        self.grad = gradients * self.inv_model.scaling_factor

        print(self.grad.shape)

        return scaled_loss

    def get_gradient(self, m: NDArrayFloat) -> NDArrayFloat:
        return self.grad.ravel()

    def run(self) -> ScipyOptimizeResult:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        res: ScipyOptimizeResult = ScipyOptimizeResult()
        # flatten the ensemble
        x0 = self.data_model.s_init.ravel()

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
            maxfun = get_maxfun(
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
                jac=self.get_gradient,  # type: ignore
                bounds=np.tile(
                    get_parameters_bounds(
                        self.inv_model.parameters_to_adjust, is_preconditioned=True
                    ).T,
                    self.data_model.n_ensemble,
                ).T,
                maxcor=self.solver_config.maxcor,
                gtol=self.solver_config.gtol,
                ftol=self.solver_config.ftol,
                update_fun_def=update_fun_def,
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
