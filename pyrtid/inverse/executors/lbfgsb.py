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
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
from lbfgsb import minimize_lbfgsb
from scipy.optimize import OptimizeResult as ScipyOptimizeResult

from pyrtid.inverse.executors.base import (
    AdjointInversionExecutor,
    AdjointSolverConfig,
    adjoint_solver_config_params_ds,
    base_solver_config_params_ds,
    register_params_ds,
)
from pyrtid.inverse.params import (
    AdjustableParameter,
    eval_weighted_loss_reg,
    get_parameters_bounds,
)
from pyrtid.inverse.regularization import ConstantRegWeight, EnsembleRegularizator
from pyrtid.utils.types import NDArrayFloat

lbfgsb_solver_config_params_ds = r"""
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


def get_loss_ls_grad_from_loss_grad(
    param: AdjustableParameter,
    s_cond: NDArrayFloat,
    loss_grad: NDArrayFloat,
    idx: int,
    n_vals: int,
    reg_weight: float,
) -> NDArrayFloat:
    """
    Get the loss ls gradient from the scaled preconditioned total gradient.

    It requires to recomputed the regularization term.

    Parameters
    ----------
    param : AdjustableParameter
        Updated parameter instance.
    s_cond : NDArrayFloat
        Preconditioned adjusted values.
    loss_grad : NDArrayFloat
        Preconditioned and scaled gradient (for all updated parameters).
    idx : int
        Index of the first preconditioned value in the loss_grad vector.
    n_vals : int
        Number of preconditioned adjusted values for the given parameter.
    reg_weight : float
        Regularization weight used to compute loss_grad.

    Returns
    -------
    NDArrayFloat
        Unscaled non conditioned LS gradient.
    """
    _loss_reg_grad = param.eval_loss_reg_gradient(
        param.preconditioner.backtransform(s_cond[idx : idx + n_vals])
    )

    # unscaled and unconditioned LS gradient
    return (
        param.preconditioner.dbacktransform_inv_vec(
            s_cond[idx : idx + n_vals],
            loss_grad[idx : idx + n_vals],
        )
        - _loss_reg_grad * reg_weight
    )


def update_gradient(
    param: AdjustableParameter,
    s_cond: NDArrayFloat,
    loss_grad: NDArrayFloat,
    idx: int,
    n_vals: int,
) -> NDArrayFloat:
    """
    Update the global loss_gradient with the new param regularization weight.

    The update is performed on values linked with the given updated parameter instance.

    Parameters
    ----------
    param : AdjustableParameter
        Updated parameter instance.
    s_cond : NDArrayFloat
        Preconditioned adjusted values.
    loss_grad : NDArrayFloat
        Preconditioned and scaled gradient (for all updated parameters).
    idx : int
        Index of the first preconditioned value in the loss_grad vector.
    n_vals : int
        Number of preconditioned adjusted values for the given parameter.

    Returns
    -------
    NDArrayFloat
        Updated scaled and preconditioned gradient.
    """
    # 1) Current unconditioned reg gradient
    loss_reg_grad = param.eval_loss_reg_gradient(
        param.preconditioner.backtransform(s_cond[idx : idx + n_vals])
    )

    # unscaled and unconditioned LS gradient
    loss_ls_grad = get_loss_ls_grad_from_loss_grad(
        param,
        s_cond,
        loss_grad,
        idx,
        n_vals,
        param.reg_weight_history[-1],
    )
    return param.preconditioner.dbacktransform_vec(
        s_cond[idx : idx + n_vals],
        loss_ls_grad + loss_reg_grad * param.reg_weight,
    ).ravel("F")


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

    maxcor: int = 10
    ftarget: Optional[float] = None
    ftol: float = 1e-5
    gtol: float = 1e-5
    maxiter: int = 50
    maxfun: int = 100
    iprint: int = -1
    maxls: int = 20
    ftol_linesearch: float = 1e-4
    gtol_linesearch: float = 0.9
    max_steplength: float = 1e8
    xtol_linesearch: float = 1e-5
    eps_SY: float = 2.2e-16
    gradient_scaler: Optional[
        Callable[[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat], float]
    ] = None


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
                self.solver_config.is_adj_numerical_acceleration,
            )

    def _get_solver_name(self) -> str:
        """Return the solver name."""
        return "L-BFGS-B (PyRTID)"

    def _update_fun_def(
        self,
        s_cond: NDArrayFloat,
        loss: float,
        loss_old: float,
        loss_grad: NDArrayFloat,
        S: Deque[NDArrayFloat],
        G: Deque[NDArrayFloat],
    ) -> Tuple[float, float, NDArrayFloat, Deque[NDArrayFloat]]:
        """
        Update f0, grad and G to match a potential new objective function.

        Method to update the gradient sequence. This is an experimental feature to
        allow changing the objective function definition on the fly. In the first place
        this functionality is dedicated to regularized problems for which the
        regularization weight is computed while optimizing the cost function. In order
        to get a hessian matching the new definition of `fun`, the gradient sequence
        must be updated.

            ``_update_fun_def(f0, f0_old, grad, x_deque, grad_deque)
            -> f0, grad, updated grad_deque``

        Parameters
        ----------
        s_cond: NDArrayFloat
            Last updated variables (conditioned).
        loss : float
            Value of the objective function for the last updated variables `s`.
            This loss has been scaled so that the first objective function is 1.0.
        loss_old : float
            Value of the objective function at the previous iteration.
            It must be updated to avoid reaching the stop criterion based on relative
            loss improvement too early. This loss has been scaled so that the first
            objective function is 1.0.
        loss_grad : NDArrayFloat
            Gradient of the objective function w.r.t. the last updated variables `s`.
        S : Deque[NDArrayFloat]
            Sequence of past updated variable vectors.
        G : Deque[NDArrayFloat]
            Sequence of gradients of the objective function w.r.t. to the past
            updated variable vectors.

        Returns
        -------
        Tuple[float, NDArrayFloat, Deque[NDArrayFloat]]
            Update f0, grad and G.
        """
        # should be equal (with rounding errors)
        lbfgsb_sf = (loss) / self.inv_model.loss_total_unscaled

        # Regularization weight that has been used up to now
        has_been_updated: List[bool] = []
        idx = 0  # idx of the first value for the parameter
        for _i, param in enumerate(self.inv_model.parameters_to_adjust):
            # number of updated values for the current parameter
            n_vals: int = param.size_preconditioned_values

            # If the regularization weight for this parameter is constant,
            # no update, go to the next parameter
            if not param.regularizators or isinstance(
                param.reg_weight_update_strategy, ConstantRegWeight
            ):
                has_been_updated.append(False)
                idx += n_vals
                continue

            # Compute scaled gradients
            # This step is a bit complex because we must take into account both the
            # scaling and the potential preconditioning.
            _loss_reg_grad = param.eval_loss_reg_gradient(
                param.preconditioner.backtransform(s_cond[idx : idx + n_vals])
            )

            # unscaled and unconditioned LS gradient
            _loss_ls_grad = get_loss_ls_grad_from_loss_grad(
                param,
                s_cond,
                loss_grad / lbfgsb_sf,
                idx,
                n_vals,
                param.reg_weight,
            )

            # unweighted loss reg are stored at the parameter level
            has_been_updated.append(
                param.update_reg_weight(
                    self.inv_model.loss_ls_history,  # not scaled not preconditioned
                    _loss_ls_grad,
                    _loss_reg_grad,
                    self.data_model.n_obs,
                    logger=logging.getLogger("RW updater"),
                )
            )

            idx += n_vals

        # No need to update anything
        if not any(has_been_updated):
            return loss, loss_old, loss_grad, G

        # Update the number of times the regularization weights have been updated
        self.inv_model.n_update_rw += 1

        # Update the objective function (scaled)
        loss_reg: float = eval_weighted_loss_reg(
            self.inv_model.parameters_to_adjust, self.fwd_model, s_cond=s_cond
        )
        loss = (self.inv_model.loss_ls_unscaled + loss_reg) * lbfgsb_sf

        # Update the previous objective function to prevent early break
        # This is a hack specific to lbfgsb
        while (loss_old - loss) / max(
            abs(loss_old), abs(loss), 1
        ) < self.solver_config.ftol:
            loss_old *= 2.0

        logging.info(
            f"- Updating regularization weights # {self.inv_model.n_update_rw}"
        )

        # Update all gradients
        idx = 0  # idx of the first value for the parameter
        for param in self.inv_model.parameters_to_adjust:
            # Case 1: no update to perform for the parameter
            if not has_been_updated[_i]:
                continue

            logging.info(
                f"Old reg weight ({param.name})   = {param.reg_weight_history[-1]}"
            )
            logging.info(f"New reg weight ({param.name})   = {param.reg_weight}")

            # Case 2: update all gradients
            # number of updated values for the current parameter
            n_vals: int = param.size_preconditioned_values

            # # 1) Current unconditioned reg gradient
            loss_grad[idx : idx + n_vals] = (
                update_gradient(
                    param,
                    s_cond,
                    loss_grad[idx : idx + n_vals] / lbfgsb_sf,
                    idx,
                    n_vals,
                )
                * lbfgsb_sf
            )
            # 2) Update past gradients stored in G
            for _j, _s_cond in enumerate(reversed(S)):
                # going over S and G backward
                G[len(G) - _j - 1][idx : idx + n_vals] = (
                    update_gradient(
                        param,
                        _s_cond,
                        G[len(G) - _j - 1][idx : idx + n_vals] / lbfgsb_sf,
                        idx,
                        n_vals,
                    )
                    * lbfgsb_sf
                )

            # update the global index
            idx += n_vals

        logging.info(f"New loss regularization   = {loss_reg}")
        logging.info(f"New loss (scaled)         = {loss}\n")

        logging.info(
            f"- Updating regularization weights # {self.inv_model.n_update_rw} over"
        )

        # return updated_j, updated_grad, G
        return loss, loss_old, loss_grad, G

    def run(self) -> ScipyOptimizeResult:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        return minimize_lbfgsb(
            x0=self.data_model.s_init,
            fun=self.eval_loss,
            jac=self.eval_loss_gradient,  # type: ignore
            update_fun_def=(
                self._update_fun_def
                if self.inv_model.is_adaptive_regularization()
                else None
            ),
            bounds=get_parameters_bounds(
                self.inv_model.parameters_to_adjust, is_preconditioned=True
            ),
            maxcor=self.solver_config.maxcor,
            ftarget=self.solver_config.ftarget,
            ftol=self.solver_config.ftol,
            gtol=self.solver_config.gtol,
            maxiter=self.solver_config.maxiter,
            maxfun=self.solver_config.maxfun,
            iprint=self.solver_config.iprint,
            maxls=self.solver_config.maxls,
            max_steplength=self.solver_config.max_steplength,
            ftol_linesearch=self.solver_config.ftol_linesearch,
            gtol_linesearch=self.solver_config.gtol_linesearch,
            xtol_linesearch=self.solver_config.xtol_linesearch,
            eps_SY=self.solver_config.eps_SY,
            gradient_scaler=self.solver_config.gradient_scaler,
            logger=logging.getLogger("L-BFGS-B"),
        )


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
        old_weight: float = self.inv_model.loss_reg_weight

        # Get the current regularization term of the objectuve function
        # And compute the new weight for the last X
        loss_reg: float = self.eval_weighted_loss_reg(
            x.reshape(self.data_model.s_init.shape)
        )

        if (
            old_weight == 0 and loss_reg == 0
        ) or self.solver_config.n_fun_before_reg > self.inv_model.nb_f_calls:
            return j, grad, G

        # Compute back the data misfit objective function
        # (remove the previous regularization term)
        # + we must take the scaling factor into account
        j0: float = j - old_weight * loss_reg

        # Check that it did not when wrong -> The objective function should be the
        # same (taking into account rounding errors)
        if (j0 - self.inv_model.loss_ls) / self.inv_model.loss_ls > 1e-8:
            raise RuntimeError(
                "Something went wrong while updating the objective function definition."
                f"\n - Saved weighted loss_reg = {self.inv_model.loss_reg}, "
                f"back-computed weighted loss_reg = {loss_reg * old_weight}."
                f"\nSaved j0 = {self.inv_model.loss_ls}, back-computed j0 = {j0}."
            )

        # Update the regularization weight
        new_weight: float = j0 / loss_reg

        if new_weight == old_weight:
            return j, grad, G

        self.inv_model.loss_reg_weight = new_weight
        # updated j (scaled with the scaling factor)
        updated_j: float = j0 + new_weight * loss_reg

        # Update the current gradient
        coef = new_weight - old_weight
        updated_grad = (
            grad
            + coef
            * self.eval_weighted_loss_reg_gradient(
                x.reshape(self.data_model.s_init.shape)
            ).ravel()
        )

        # Update all past gradients in G
        for _x in X:
            # we remove the reg part with the old weight and add it back with the
            # new one
            # Note: eval_weighted_loss_reg_gradient is very fast to obtain
            G.append(
                (
                    G.popleft()
                    + coef
                    * self.eval_weighted_loss_reg_gradient(
                        _x.reshape(self.data_model.s_init.shape)
                    ).ravel()
                )
            )

        # Check that all gradients have been correctly updated
        assert len(X) == len(G)

        return updated_j, updated_grad, G

    def eval_weighted_loss_reg(self, s_ensemble) -> float:
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
        loss_reg = 0
        for param in self.inv_model.parameters_to_adjust:
            for reg in param.regularizators:
                if isinstance(reg, EnsembleRegularizator):
                    loss_reg += reg.eval_loss(s_ensemble)
        return loss_reg

    def eval_weighted_loss_reg_gradient(self, s_ensemble) -> NDArrayFloat:
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
                    reg_grad += reg.eval_loss_gradient(
                        s_ensemble.reshape(self.data_model.s_init.shape)
                    )
        return reg_grad

    def update_loss_reg_weight(
        self,
        j0: float,
        loss_reg: float,
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
            The default is RegWeightUpdateStrategy.ADAPTIVE_UC.
        n_fun_before_reg: int
            The number of objective function evaluation to perform before adding the
            regularization term. This feature allows to start optimizing with
            the misfit part only (no risk of overfitting at the beginning).

        Returns
        -------
        loss_reg : float
            the regularization objective function.

        """
        if self.inv_model.optimization_round_nb == 1:
            if (
                not self.inv_model.is_regularization_at_first_round
                or n_fun_before_reg > self.nb_f_calls
            ):
                self.inv_model.loss_reg_weight = 0.0
                return

            # if loss_reg == 0:
            self.inv_model.loss_reg_weight = 0.0
            return

    def eval_loss(self, s_ensemble: NDArrayFloat, is_save_state: bool = True) -> float:
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
        loss_ls: float = float(np.mean(losses))

        # Compute the regularization term:
        if self.solver_config.reg_factor == 0:
            loss_reg = 0.0
        else:
            loss_reg: float = self.eval_weighted_loss_reg(s_ensemble)
            self.update_loss_reg_weight(
                loss_ls,
                loss_reg,
                self.solver_config.reg_factor,
                n_fun_before_reg=self.solver_config.n_fun_before_reg,
            )
            loss_reg *= self.inv_model.loss_reg_weight

            if self.inv_model.loss_reg_weight != 0:
                gradients += (
                    self.eval_weighted_loss_reg_gradient(s_ensemble)
                    * self.inv_model.loss_reg_weight
                )

        total_loss: float = loss_ls + loss_reg

        # Apply the scaling coefficient
        scaled_loss: float = (
            total_loss
            * self.inv_model.get_loss_function_scaling_factor(
                total_loss, is_first_call=self.inv_model.nb_f_calls == 0
            )
        )

        # Store the losses to the inverse model
        self.inv_model.loss_history += list(losses)

        # Need to store this locally because the mechanisms in inverse models
        self.nb_f_calls += 1

        # Store the last objective function values (ls and reg terms)
        self.inv_model.loss_ls = loss_ls
        self.inv_model.loss_reg = loss_reg

        logging.info(f"Loss (obs fit)        = {loss_ls}")
        logging.info(f"Loss (regularization) = {loss_reg}")
        logging.info(f"Regularization weight = {self.inv_model.loss_reg_weight}")
        logging.info(f"Scaling factor        = {self.inv_model.scaling_factor}")
        logging.info(f"Loss (scaled)         = {scaled_loss}\n")

        # Save the loss and the associated regularization weight
        if is_save_state:
            self.inv_model.loss_history.append(scaled_loss)

        # The loss_ls gradient is the stacking of all members gradient
        # (independent from each others).
        # In this specific case, the regularization imposed to the parameters
        # are ignored and we consider our own regularization approach.

        # TODO: add the regularization terms that might depend on the ensemble.
        self.grad = gradients * self.inv_model.scaling_factor

        print(self.grad.shape)

        return scaled_loss

    def get_gradient(self, s_cond: NDArrayFloat) -> NDArrayFloat:
        return self.grad.ravel()

    def run(self) -> ScipyOptimizeResult:
        """
        Run the history matching.

        First is creates raw folders to store the different runs
        required by the HM algorithms.
        """
        super().run()
        return minimize_lbfgsb(
            x0=self.data_model.s_init.ravel(),
            fun=self.eval_loss,
            jac=self.get_gradient,  # type: ignore
            bounds=np.tile(
                get_parameters_bounds(
                    self.inv_model.parameters_to_adjust, is_preconditioned=True
                ).T,
                self.data_model.n_ensemble,
            ).T,
            maxcor=self.solver_config.maxcor,
            gtol=self.solver_config.gtol,
            ftarget=self.solver_config.ftarget,
            ftol=self.solver_config.ftol,
            # update_fun_def=self.update_fun_def,
            maxiter=self.solver_config.maxiter,
            maxfun=self.solver_config.maxfun,
            iprint=self.solver_config.iprint,
            maxls=self.solver_config.maxls,
            max_steplength=self.solver_config.max_steplength,
            ftol_linesearch=self.solver_config.ftol_linesearch,
            gtol_linesearch=self.solver_config.gtol_linesearch,
            xtol_linesearch=self.solver_config.xtol_linesearch,
            eps_SY=self.solver_config.eps_SY,
            logger=logging.getLogger("L-BFGS-B"),
        )
