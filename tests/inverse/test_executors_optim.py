"""Tests of the gradient based / stochastic executors and of the base classes."""

import logging

import numpy as np
import pytest
from pyrtid.inverse.executors import (
    ScipyInversionExecutor,
    ScipySolverConfig,
    StochopyInversionExecutor,
    StochopySolverConfig,
)
from pyrtid.inverse.executors.base import DataModel
from tests.executor_helpers import tiny_inverse_model

pytest.importorskip("stochopy")


def make_scipy(names=("K",), s_init=None, pre=None, **cfg):
    model, inv = tiny_inverse_model(names)
    cfg.setdefault("solver_options", {"maxiter": 2})
    ex = ScipyInversionExecutor(
        model, inv, ScipySolverConfig(**cfg), pre_run_transformation=pre, s_init=s_init
    )
    return ex, model, inv


def test_scipy_run_adjoint_decreases_loss() -> None:
    ex, _, inv = make_scipy(is_check_gradient=True, is_adj_verbose=True)
    assert ex.solver_config.hm_end_time == ex.fwd_model.time_params.duration
    assert ex._get_solver_name() == "L-BFGS-B"
    assert ex.get_display_dict() == {}
    res = ex.run()
    assert res.x.shape == (ex.data_model.s_dim,)
    assert inv.loss_history[-1] <= inv.loss_history[0]
    assert inv.nb_g_calls >= 1
    assert len(inv.list_d_pred) >= 1
    assert len(inv.parameters_to_adjust[0].grad_adj_history) == inv.nb_g_calls
    assert len(inv.parameters_to_adjust[0].grad_fd_history) == inv.nb_g_calls


def test_scipy_finite_difference_and_rounds() -> None:
    ex, _, inv = make_scipy(
        is_use_adjoint=False,
        max_optimization_round_nb=2,
        max_fun_first_round=3,
        max_fun_per_round=2,
        solver_options={"maxiter": 2, "maxfun": 40},
    )
    from inv_toolbox.regularization import TikhonovRegularizator

    inv.parameters_to_adjust[0].regularizators = [
        TikhonovRegularizator(ex.fwd_model.grid)
    ]
    with pytest.raises(AttributeError, match="adjoint model does not"):
        _ = ex.adj_model
    ex.run()
    assert inv.optimization_round_nb == 2
    assert len(inv.parameters_to_adjust[0].grad_fd_history) >= 1
    assert inv.parameters_to_adjust[0].grad_adj_history == []


def test_scipy_options_dict() -> None:
    ex, _, _ = make_scipy(
        max_optimization_round_nb=3, max_fun_first_round=4, max_fun_per_round=6
    )
    cfg = ex.solver_config
    assert ex._get_options_dict(cfg, 0, 1)["maxfun"] == 4
    assert ex._get_options_dict(cfg, 10, 2)["maxfun"] == 6
    cfg.solver_options = {"maxfun": 12}
    assert ex._get_options_dict(cfg, 10, 2)["maxfun"] == 2
    cfg.max_optimization_round_nb = 1
    assert ex._get_options_dict(cfg, 10, 1)["maxfun"] == 12
    cfg.solver_options = None
    assert ex._get_options_dict(cfg, 10, 1)["maxfun"] == 15000


def test_base_init_options() -> None:
    called = []
    ex, model, inv = make_scipy(
        hm_end_time=1e12, pre=lambda m: called.append(1), is_save_spmats=True
    )
    assert ex.solver_config.hm_end_time == model.time_params.duration
    s = ex.data_model.s_init
    ex.eval_loss(s, is_save_state=False, is_verbose=True)
    assert called and inv.loss_history == [] and inv.list_d_pred == []
    ex.eval_loss(s)
    assert len(inv.loss_history) == 1 and len(inv.list_d_pred) == 1
    assert inv.loss_history[0] == pytest.approx(inv.loss_total_unscaled)
    assert model.fl_model.is_save_spmats and model.tr_model.is_save_spmats
    ex2, _, _ = make_scipy(hm_end_time=3600 * 12.0)
    assert ex2.solver_config.hm_end_time == 3600 * 12.0
    assert ex2.obs.size < ex.obs.size
    assert ex2.std_obs.size == ex2.obs.size


def test_s_init_validation() -> None:
    ex, _, _ = make_scipy()
    n = ex.data_model.s_dim
    good = np.full(n, np.log(2e-4))
    ex2, _, _ = make_scipy(s_init=good.reshape(1, n))
    assert np.allclose(ex2.data_model.s_init, good)
    with pytest.warns(UserWarning, match="out of bounds"):
        ex3, _, _ = make_scipy(s_init=np.full(n, 1e5))
    assert np.all(ex3.data_model.s_init <= np.log(1.0) + 1e-12)
    with pytest.raises(ValueError, match="s_init must be"):
        ex.validate_s_init(np.zeros(n + 1), n)


def test_check_nans_and_output_dir(tmp_path) -> None:
    ex, _, _ = make_scipy()
    ex._check_nans_in_predictions(np.zeros(3), 1)
    with pytest.raises(Exception, match="simulation 4 !"):
        ex._check_nans_in_predictions(np.array([0.0, np.nan]), 4)
    d = np.zeros((3, 4))
    d[1, 2] = np.nan
    with pytest.raises(Exception, match=r"simulation\(s\) \[5\]"):
        ex._check_nans_in_predictions(d, 2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "f.txt").write_text("x")
    ex.create_output_dir(out)
    assert out.is_dir() and not list(out.iterdir())
    ex.create_output_dir(tmp_path / "new")
    assert (tmp_path / "new").is_dir()


def test_initial_display(caplog) -> None:
    ex, _, _ = make_scipy()
    with caplog.at_level(logging.INFO):
        ex._initial_display()
    assert "Inversion Parameters" in caplog.text
    assert "L-BFGS-B" in caplog.text


@pytest.mark.parametrize("is_parallel", [False, True])
def test_map_forward_model(is_parallel) -> None:
    ex, _, inv = make_scipy(is_parallel=is_parallel, max_workers=2)
    s = ex.data_model.s_init
    ens = np.stack([s, s + 0.1, s - 0.1], axis=1)
    d_pred = ex._map_forward_model(ens)
    assert d_pred.shape == (ex.data_model.d_dim, 3)
    assert len(inv.loss_history) == 3
    assert not np.allclose(d_pred[:, 0], d_pred[:, 1])


@pytest.mark.parametrize("is_parallel", [False, True])
def test_map_forward_model_with_adjoint(is_parallel) -> None:
    ex, _, inv = make_scipy(max_workers=2)
    s = ex.data_model.s_init
    ens = np.stack([s, s + 0.1], axis=1)
    losses, d_pred, grads = ex._map_forward_model_with_adjoint(ens, is_parallel)
    assert losses.shape == (2,) and d_pred.shape == (ex.data_model.d_dim, 2)
    assert grads.shape == (ex.data_model.s_dim, 2)
    # consistency with the single evaluation
    assert losses[0] == pytest.approx(ex.eval_loss(s))
    assert np.allclose(grads[:, 0], ex.eval_loss_gradient(s))


def test_adjoint_gradient_check() -> None:
    ex, _, _ = make_scipy()
    ex.eval_loss(ex.data_model.s_init)
    assert ex.is_adjoint_gradient_correct(max_workers=1, is_verbose=True)


def test_scipy_bad_gradient_warning(caplog, monkeypatch) -> None:
    ex, _, _ = make_scipy(is_check_gradient=True)
    ex.eval_loss(ex.data_model.s_init)
    import pyrtid.inverse.executors.base as base

    monkeypatch.setattr(base, "is_all_close", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING):
        ex.eval_loss_gradient(ex.data_model.s_init, is_verbose=True)
    assert "not correct" in caplog.text


def test_data_model_properties() -> None:
    dm = DataModel(np.zeros(3), np.zeros((7, 2)), np.eye(3))
    assert dm.is_ensemble() and dm.n_obs == 3 and dm.s_dim == 7


# --------------------------------------------------------------------------- #
# stochopy
# --------------------------------------------------------------------------- #


def make_stochopy(**cfg):
    model, inv = tiny_inverse_model(("K",))
    return StochopyInversionExecutor(model, inv, StochopySolverConfig(**cfg)), inv


@pytest.mark.parametrize("method", ["cmaes", "vdcma"])
def test_stochopy_run(method) -> None:
    ex, inv = make_stochopy(
        solver_name=method,
        solver_options={"maxiter": 2, "popsize": 4, "seed": 0},
        max_fun_per_round=8,
    )
    assert ex._get_solver_name() == method
    res = ex.run()
    assert res.x.shape == (ex.data_model.s_dim,)
    assert len(inv.loss_history) >= 4
    assert min(inv.loss_history) <= inv.loss_history[0] + 1e-12
    assert inv.optimization_round_nb == 1


def test_stochopy_options_dict() -> None:
    ex, inv = make_stochopy(max_optimization_round_nb=2, max_fun_per_round=5)
    cfg = ex.solver_config
    inv.optimization_round_nb = 1
    assert ex._get_options_dict(cfg, 3)["maxfun"] == 5
    inv.optimization_round_nb = 2
    assert ex._get_options_dict(cfg, 3)["maxfun"] == 0
    cfg.solver_options = {"maxfun": 6, "seed": 1}
    inv.optimization_round_nb = 1
    opts = ex._get_options_dict(cfg, 3)
    assert opts["maxfun"] == 3 and opts["seed"] == 1
    assert cfg.solver_options["maxfun"] == 6


def test_stochopy_to_stochopy_options() -> None:
    conv = StochopyInversionExecutor._to_stochopy_options
    assert conv({"maxfun": 0, "seed": 1}) == {"seed": 1}
    assert conv({"maxfun": 25, "popsize": 10})["maxiter"] == 3
    assert conv({"maxfun": 1})["maxiter"] == 1
    assert conv({"maxfun": 1000, "maxiter": 4})["maxiter"] == 4


def test_stochopy_multiple_rounds() -> None:
    from inv_toolbox.regularization import TikhonovRegularizator

    ex, inv = make_stochopy(
        solver_options={"maxiter": 2, "popsize": 4, "seed": 0},
        max_optimization_round_nb=2,
        max_fun_per_round=4,
    )
    inv.parameters_to_adjust[0].regularizators = [
        TikhonovRegularizator(ex.fwd_model.grid)
    ]
    ex.run()
    assert inv.optimization_round_nb == 2


def test_base_default_solver_name() -> None:
    from pyrtid.inverse.executors.base import BaseInversionExecutor

    ex, _, _ = make_scipy()
    assert BaseInversionExecutor._get_solver_name(ex) == "unknown"


def test_fd_gradient_with_check_and_no_adjoint(caplog, monkeypatch) -> None:
    import pyrtid.inverse.executors.base as base

    ex, _, inv = make_scipy(is_use_adjoint=False, is_check_gradient=True)
    s = ex.data_model.s_init
    ex.eval_loss(s)
    assert ex._adj_model is None
    grad1 = ex.eval_loss_gradient(s)  # no previous adjoint model
    assert ex._adj_model is not None
    monkeypatch.setattr(base, "is_all_close", lambda *a, **k: True)
    with caplog.at_level(logging.INFO):
        grad2 = ex.eval_loss_gradient(s)  # previous adjoint model exists
    assert "seems correct" in caplog.text
    assert np.allclose(grad1, grad2)
    assert grad1.shape == (ex.data_model.s_dim,)
    assert inv.nb_g_calls == 2
    # the same holds for the combined forward + adjoint evaluation
    ex2, _, _ = make_scipy(
        is_use_adjoint=False, is_check_gradient=True, pre=lambda m: None
    )
    loss, d_pred, adj = ex2._run_forward_model_with_adjoint(s, 1)
    assert d_pred.shape == (ex2.data_model.d_dim,) and adj.shape == grad1.shape
    assert loss == pytest.approx(ex2.eval_loss(s, is_save_state=False))


def test_run_forward_model_with_adjoint_pre_run() -> None:
    called = []
    ex, _, _ = make_scipy(pre=lambda m: called.append(1))
    ex._run_forward_model_with_adjoint(ex.data_model.s_init, 1)
    assert called == [1]


def test_crank_nicolson_propagated_to_new_adjoint_models() -> None:
    ex, _, _ = make_scipy()
    s = ex.data_model.s_init
    ex.eval_loss(s)
    ex.adj_model.a_fl_model.set_crank_nicolson(0.6)
    ex.eval_loss_gradient(s)
    assert ex.adj_model.a_fl_model.crank_nicolson == 0.6
    ex.adj_model.a_fl_model.set_crank_nicolson(0.7)
    ex._run_forward_model_with_adjoint(s, 1)
    assert ex.adj_model.a_fl_model.crank_nicolson == 0.7
