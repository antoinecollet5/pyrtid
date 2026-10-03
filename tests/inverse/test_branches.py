"""Cover the remaining branches of the gradient evaluations."""

from __future__ import annotations

import numpy as np
import pytest
from pyrtid.inverse.asm.gradients import compute_fd_gradient
from tests.inverse.test_executors_optim import make_scipy


def test_plain_finite_difference_gradient() -> None:
    """Without adjoint, nor check, the finite differences gradient is returned."""
    ex, _, inv = make_scipy(is_use_adjoint=False)
    s = ex.data_model.s_init
    ex.eval_loss(s)
    grad = ex.eval_loss_gradient(s)
    assert grad.shape == (ex.data_model.s_dim,)
    assert np.all(np.isfinite(grad))
    assert inv.parameters_to_adjust[0].grad_fd_history != []
    assert inv.parameters_to_adjust[0].grad_adj_history == []


def test_run_forward_model_with_finite_difference_only() -> None:
    ex, _, _ = make_scipy(is_use_adjoint=False, pre=lambda m: None)
    s = ex.data_model.s_init
    loss, d_pred, grad = ex._run_forward_model_with_adjoint(s, 1)
    assert d_pred.shape == (ex.data_model.d_dim,)
    assert grad.size == 0  # no adjoint gradient was requested
    assert loss == pytest.approx(ex.eval_loss(s, is_save_state=False))


def test_fd_gradient_without_saving_the_state() -> None:
    ex, _, inv = make_scipy()
    ex.eval_loss(ex.data_model.s_init)
    param = inv.parameters_to_adjust[0]
    n_before = len(param.grad_fd_history)
    grad = compute_fd_gradient(
        ex.fwd_model, inv.observables, inv.parameters_to_adjust, is_save_state=False
    )
    assert grad.shape == (ex.data_model.s_dim,)
    assert len(param.grad_fd_history) == n_before


def test_gradient_without_saving_the_state() -> None:
    ex, _, inv = make_scipy()
    s = ex.data_model.s_init
    ex.eval_loss(s, is_save_state=False)
    grad = ex.eval_loss_gradient(s, is_save_state=False)
    assert grad.shape == (ex.data_model.s_dim,)
    assert inv.nb_g_calls == 0
