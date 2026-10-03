"""Tests of the L-BFGS-B executor, its adaptive regularization and gradient scaling."""

import copy
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from inv_toolbox.regularization import AdaptiveRegweight, TikhonovRegularizator
from inv_toolbox.utils.preconditioner import GradientScalerConfig, NoTransform
from pyrtid.inverse.executors import LBFGSBInversionExecutor, LBFGSBSolverConfig
from pyrtid.inverse.executors.lbfgsb import (
    get_loss_ls_grad_from_loss_grad,
    update_gradient,
)
from pyrtid.inverse.model import InverseModel
from tests.executor_helpers import tiny_model, tiny_observables
from tests.helpers import build_parameters

pytestmark = pytest.mark.filterwarnings(
    "ignore:divide by zero:RuntimeWarning", "ignore:Adjusted parameter:UserWarning"
)


class Halving(AdaptiveRegweight):
    """Adaptive strategy halving the weight at each update."""

    def update_reg_weight(self, *args, **kwargs) -> bool:
        self.reg_weight *= 0.5
        return True


class Frozen(AdaptiveRegweight):
    """Adaptive strategy never updating the weight."""

    def update_reg_weight(self, *args, **kwargs) -> bool:
        return False


def make_lbfgsb(names=("K",), adaptive=(), scalers=(), strategy=Halving, **cfg):
    model = tiny_model()
    params = build_parameters(list(names))
    for p in params:
        if p.name.value in adaptive:
            p.regularizators = [TikhonovRegularizator(model.grid)]
            p.reg_weight_update_strategy = strategy(1.0)
        if p.name.value in scalers:
            p.gradient_scaler_config = GradientScalerConfig(
                max_change_target=0.5, max_workers=1, n_samples_in_first_round=3
            )
    inv = InverseModel(params, tiny_observables(model, 2))
    cfg.setdefault("maxiter", 3)
    cfg.setdefault("maxfun", 12)
    return LBFGSBInversionExecutor(model, inv, LBFGSBSolverConfig(**cfg)), model, inv


def test_lbfgsb_plain_run() -> None:
    ex, _, inv = make_lbfgsb(("K", "W"))
    assert ex._get_solver_name() == "L-BFGS-B (PyRTID)"
    assert not ex.is_gradient_scaling_needed()
    assert not inv.is_adaptive_regularization()
    res = ex.run()
    assert res.nit >= 1
    assert res.x.shape == (ex.data_model.s_dim,)
    assert min(inv.loss_history) < inv.loss_history[0]
    assert inv.nb_g_calls >= 1
    assert inv.n_update_rw == 0
    assert len(inv.parameters_to_adjust[1].grad_adj_history) == inv.nb_g_calls


def test_lbfgsb_finite_difference_gradient() -> None:
    ex, _, inv = make_lbfgsb(("K",), is_use_adjoint=False, maxiter=2)
    ex.run()
    assert inv.parameters_to_adjust[0].grad_adj_history == []
    assert len(inv.parameters_to_adjust[0].grad_fd_history) >= 1


def test_callbacks() -> None:
    ex, _, inv = make_lbfgsb(("K",), stol=1e10)
    p = inv.parameters_to_adjust[0]
    # nothing to compare with: zero change -> stop
    assert ex.callback_new(ex.data_model.s_init, None)
    p.archived_values.extend([p.values.copy(), p.values.copy()])
    assert ex.callback_new(ex.data_model.s_init, None)
    p.values = p.values * 1.5
    ex.solver_config.stol = 1e-12
    assert not ex.callback_new(ex.data_model.s_init, None)
    # old criterion based on the last L-BFGS step
    s = np.ones(4)
    res = SimpleNamespace(hess_inv=SimpleNamespace(sk=[np.full(4, 1e-3)]))
    assert not ex.callback(s, res)
    ex.solver_config.stol = 1e-2
    assert ex.callback(s, res)


@pytest.mark.parametrize("scalers", [("permeability",), ("permeability", "porosity")])
def test_gradient_scaling(scalers) -> None:
    ex, _, inv = make_lbfgsb(("K", "W"), scalers=scalers, maxiter=2)
    assert ex.is_gradient_scaling_needed()
    s0 = ex.data_model.s_init.copy()
    pcd0 = type(inv.parameters_to_adjust[0].preconditioner)
    ex.run()
    # the preconditioner of K has been rescaled and so are the initial values
    assert type(inv.parameters_to_adjust[0].preconditioner) is not pcd0
    assert not np.allclose(ex.data_model.s_init[:10], s0[:10])
    assert inv.loss_history[0] > 0


def test_scale_initial_gradient_no_rescaling_needed() -> None:
    ex, _, inv = make_lbfgsb(("K",), scalers=("permeability",), maxiter=2)
    p = inv.parameters_to_adjust[0]
    p.gradient_scaler_config = GradientScalerConfig(
        max_change_target=1e12, max_workers=1, n_samples_in_first_round=3
    )
    pcd0 = p.preconditioner
    fun, grad = ex.scale_initial_gradient()
    assert p.preconditioner is pcd0
    assert grad.shape == (ex.data_model.s_dim,)
    assert fun == pytest.approx(inv.loss_total_unscaled)


def test_adaptive_regularization_updates_weights_and_gradients() -> None:
    ex, _, inv = make_lbfgsb(
        ("D", "K", "W"), adaptive=("permeability", "porosity"), maxiter=4
    )
    assert inv.is_adaptive_regularization()
    k, w = inv.parameters_to_adjust[1:]
    res = ex.run()
    assert inv.n_update_rw >= 1
    # weights halved at each update; the unregularized parameter is untouched
    assert k.reg_weight < 1.0 and w.reg_weight == k.reg_weight
    assert inv.parameters_to_adjust[0].reg_weight == 1.0
    assert res.x.shape == (30,)


def test_update_fun_def_gradient_consistency() -> None:
    """The updated gradient equals the gradient computed with the new weights."""
    ex, _, inv = make_lbfgsb(
        ("D", "K", "W"), adaptive=("permeability", "porosity"), maxiter=1
    )
    s = ex.data_model.s_init
    # perturb so the regularization gradient is not null
    s = s + 0.05 * np.random.default_rng(1).standard_normal(s.size)
    loss = ex.eval_loss(s)
    grad = ex.eval_loss_gradient(s)
    new_loss, old_loss, new_grad, G = ex._update_fun_def(
        s, loss, copy.copy(loss), grad.copy(), deque(), deque()
    )
    assert inv.n_update_rw == 1
    assert not np.allclose(new_grad, grad)
    assert new_loss < loss  # weights have been decreased
    assert old_loss >= loss - 1e-12 or old_loss > new_loss
    ref = ex.eval_loss_gradient(s)  # computed with the halved weights
    assert np.allclose(new_grad, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())
    # unregularized parameter gradient unchanged
    assert np.allclose(new_grad[:10], grad[:10])
    assert len(G) == 0


def test_update_fun_def_past_gradients() -> None:
    ex, _, inv = make_lbfgsb(("K", "W"), adaptive=("permeability", "porosity"))
    s0 = ex.data_model.s_init
    rng = np.random.default_rng(2)
    s1 = s0 + 0.05 * rng.standard_normal(s0.size)
    s2 = s0 + 0.05 * rng.standard_normal(s0.size)
    ex.eval_loss(s1)
    g1 = ex.eval_loss_gradient(s1)
    ex.eval_loss(s2)
    g2 = ex.eval_loss_gradient(s2)
    loss = inv.loss_total_unscaled
    S, G = deque([s1]), deque([g1.copy()])
    _, _, new_g2, G = ex._update_fun_def(s2, loss, copy.copy(loss), g2.copy(), S, G)
    assert inv.n_update_rw == 1
    ex.eval_loss(s1)
    assert np.allclose(G[0], ex.eval_loss_gradient(s1), rtol=1e-6, atol=1e-8)
    ex.eval_loss(s2)
    assert np.allclose(new_g2, ex.eval_loss_gradient(s2), rtol=1e-6, atol=1e-8)


def test_update_fun_def_no_update() -> None:
    ex, _, inv = make_lbfgsb(
        ("K", "W"), adaptive=("permeability",), strategy=Frozen, maxiter=1
    )
    s = ex.data_model.s_init
    loss = ex.eval_loss(s)
    grad = ex.eval_loss_gradient(s)
    out = ex._update_fun_def(s, loss, 5.0, grad.copy(), deque(), deque())
    assert out[0] == loss and out[1] == 5.0 and np.array_equal(out[2], grad)
    assert inv.n_update_rw == 0


def test_update_fun_def_skipped_when_scaling_already_updated() -> None:
    ex, _, inv = make_lbfgsb(
        ("K",), adaptive=("permeability",), scalers=("permeability",), maxiter=1
    )
    s = ex.data_model.s_init
    loss = ex.eval_loss(s)
    grad = ex.eval_loss_gradient(s)
    inv.n_update_rw = 1
    assert len(inv.loss_ls_history) == 1
    out = ex._update_fun_def(s, loss, 7.0, grad, deque(), deque())
    assert out[0] == loss and out[1] == 7.0 and out[2] is grad


def test_adaptive_regularization_with_gradient_scaling_and_checkpoint() -> None:
    ex, _, inv = make_lbfgsb(
        ("K", "W"),
        adaptive=("permeability", "porosity"),
        scalers=("permeability",),
        maxiter=3,
    )
    res = ex.run()
    assert res.nit >= 1
    assert inv.n_update_rw >= 1


def test_loss_ls_grad_from_loss_grad_is_consistent() -> None:
    ex, _, inv = make_lbfgsb(("W",), adaptive=("porosity",), maxiter=1)
    (w,) = inv.parameters_to_adjust
    s = ex.data_model.s_init + 0.1
    ex.eval_loss(s)
    grad = ex.eval_loss_gradient(s)
    n = w.size_preconditioned_values
    ls_grad = get_loss_ls_grad_from_loss_grad(w, s, grad, 0, n, w.reg_weight)
    # the raw gradient stored by the adjoint contains the weighted reg gradient
    raw = w.grad_adj_raw_history[-1].ravel("F")
    reg = w.eval_loss_reg_gradient(s) * w.reg_weight
    assert np.allclose(ls_grad, raw - reg)
    # putting back the same weight gives back the same preconditioned gradient
    w.reg_weight_history.append(w.reg_weight)
    again = update_gradient(w, s, grad, 0, n, -1)
    assert np.allclose(again, grad)


class NonInvertibleDerivative(NoTransform):
    """Identity preconditioner not supporting ``dbacktransform_inv_vec``."""

    def _dbacktransform_inv_vec(self, s_cond, gradient):
        raise NotImplementedError


def test_update_gradient_falls_back_on_gradient_history() -> None:
    ex, _, inv = make_lbfgsb(("W",), adaptive=("porosity",), maxiter=1)
    (w,) = inv.parameters_to_adjust
    w.preconditioner = NonInvertibleDerivative()
    s = ex.data_model.s_init + 0.1
    ex.eval_loss(s)
    grad = ex.eval_loss_gradient(s)
    n = w.size_preconditioned_values
    with pytest.raises(NotImplementedError):
        get_loss_ls_grad_from_loss_grad(w, s, grad, 0, n, w.reg_weight)
    w.reg_weight_update_strategy.reg_weight = 0.5
    out = update_gradient(w, s, grad, 0, n, -1)
    expected = w.grad_adj_raw_history[-1].ravel("F") + (
        w.eval_loss_reg_gradient(s) * 0.5
    )
    assert np.allclose(out, expected)
    # the executor relies on the same mechanism during a run
    ex2, _, inv2 = make_lbfgsb(("W",), adaptive=("porosity",), maxiter=3)
    inv2.parameters_to_adjust[0].preconditioner = NonInvertibleDerivative()
    ex2.run()
    assert inv2.n_update_rw >= 1
