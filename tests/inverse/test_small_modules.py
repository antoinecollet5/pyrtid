"""Tests of the inverse model, params, obs and loss function helpers."""

import logging

import numpy as np
import pyrtid.inverse as dminv
import pytest
from inv_toolbox.regularization import AdaptiveRegweight, TikhonovRegularizator
from pyrtid.inverse.loss_function import get_theoretical_noise_level
from pyrtid.inverse.model import InverseModel
from pyrtid.inverse.obs import (
    Observable,
    StateVariable,
    get_array_from_state_variable,
    update_perturbation_values,
)
from pyrtid.inverse.params import (
    ParameterName,
    eval_weighted_loss_reg,
    eval_weighted_loss_reg_gradient,
    get_backconditioned_adj_gradient,
    get_backconditioned_fd_gradient,
    get_gridded_archived_gradients,
    get_param_values,
    get_parameter_values_from_model,
    get_parameters_bounds,
    get_parameters_values_from_model,
    identify_function,
    update_model_with_param_values,
    update_model_with_parameters_values,
    update_parameters_from_model,
)
from pyrtid.utils import Callback, StrEnum
from tests.helpers import build_model, build_observables, build_parameters


def test_callback() -> None:
    cb = Callback()
    cb(1, a=2)
    cb()
    assert cb.itercount() == 2
    cb.clear()
    assert cb.itercount() == 0


def test_str_enum() -> None:
    class Color(StrEnum):
        RED = "red"

    assert str(Color.RED) == "red"
    assert Color.RED == "red"
    assert (Color.RED == 3) is False
    assert hash(Color.RED) == hash("red")
    assert Color.to_list() == [Color.RED]


def test_noise_level() -> None:
    obs = build_observables(build_model())
    n = sum(o.values.size for o in obs)
    assert get_theoretical_noise_level(obs, 0.0) == pytest.approx(0.5 * n)


# --------------------------------------------------------------------------- #
# obs
# --------------------------------------------------------------------------- #


def test_observable_requires_sp_and_str() -> None:
    with pytest.raises(ValueError, match="sp must be provided"):
        Observable(StateVariable.GRADE, 0, np.array([0.0]), np.array([1.0]), 1.0)
    obs = build_observables(build_model())[0]
    assert "head" in str(obs)


def test_observable_perturbations() -> None:
    obs = build_observables(build_model())
    with pytest.raises(ValueError, match="perturbations size"):
        obs[0].set_perturbations(np.zeros(1))
    obs[0].set_perturbations(np.ones(obs[0].values.size))
    assert np.all(obs[0].perturbations == 1.0)
    n = obs[0].values.size
    update_perturbation_values(obs[:2], np.arange(2.0 * n))
    assert np.array_equal(obs[1].perturbations, np.arange(n, 2.0 * n))


def test_get_array_from_state_variable_errors() -> None:
    model = build_model()
    for var in (StateVariable.CONCENTRATION, StateVariable.GRADE):
        with pytest.raises(ValueError, match="sp cannot be None"):
            get_array_from_state_variable(model, var)


# --------------------------------------------------------------------------- #
# params
# --------------------------------------------------------------------------- #


def test_identify_function() -> None:
    x = np.arange(3.0)
    assert identify_function(x) is x


def test_param_init_errors() -> None:
    with pytest.raises(ValueError, match="sp must be provided"):
        dminv.AdjustableParameter(
            ParameterName.INITIAL_CONCENTRATION, lbounds=0.0, ubounds=1.0
        )
    with pytest.raises(ValueError, match="regularizator"):
        dminv.AdjustableParameter(
            ParameterName.POROSITY,
            lbounds=0.0,
            ubounds=1.0,
            regularizators=[1],  # ty: ignore[invalid-argument-type]
        )
    param = dminv.AdjustableParameter(
        ParameterName.INITIAL_GRADE, sp=1, lbounds=0.0, ubounds=1.0
    )
    assert param.sp == 1
    assert (
        dminv.AdjustableParameter(ParameterName.POROSITY, lbounds=0.0, ubounds=1.0).sp
        == 0
    )


def test_param_properties() -> None:
    model = build_model()
    (w, k) = build_parameters(["W", "K"])
    assert w.size == 0
    assert w.size_preconditioned_values == 0
    assert np.isnan(w.max_value) and np.isnan(w.min_value)
    assert not w.is_scale_logarithmically
    assert k.is_scale_logarithmically
    assert not w.is_adaptive_regularization()
    w.get_values_from_model_field(model.tr_model.porosity)
    assert w.size == model.grid.n_grid_cells
    assert w.size_preconditioned_values == w.size
    assert w.min_value <= w.max_value


def test_param_adaptive_regularization() -> None:
    model = build_model()
    (w,) = build_parameters(["W"])
    assert not w.is_adaptive_regularization()
    w.regularizators = [TikhonovRegularizator(model.grid)]
    assert not w.is_adaptive_regularization()  # constant weight
    w.reg_weight_update_strategy = AdaptiveRegweight(1.0)
    assert w.is_adaptive_regularization()


def test_update_values_with_vector_and_change() -> None:
    model = build_model()
    (k,) = build_parameters(["K"])
    k.get_values_from_model_field(model.fl_model.permeability)
    ref = k.values.copy()
    assert k.get_values_change() == 0
    k.archived_values.append(ref.copy())
    k.update_values_with_vector(ref.ravel("F") * 2.0)
    assert np.allclose(k.values, np.clip(ref * 2.0, 1e-9, 1.0))
    k.archived_values.append(k.values.copy())
    change = k.get_values_change()
    assert change > 0
    assert k.get_values_change(is_use_pcd=False) > 0
    k.update_values_with_vector(np.log(ref.ravel("F")), is_preconditioned=True)
    assert np.allclose(k.values, ref)


def test_param_bounds_array() -> None:
    model = build_model()
    (w,) = build_parameters(["W"])
    w.get_values_from_model_field(model.tr_model.porosity)
    n = w.size
    w.lbounds = np.full(n, 0.1)
    w.ubounds = np.full(n, 0.5)
    assert w.get_bounds().shape == (n, 2)
    assert get_parameters_bounds([w], True).shape == (n, 2)


def test_param_reg_methods() -> None:
    model = build_model()
    (w,) = build_parameters(["W"])
    w.get_values_from_model_field(model.tr_model.porosity)
    w.regularizators = [TikhonovRegularizator(model.grid)]
    other = np.full(w.size, 0.5)
    v_own = w.eval_loss_reg()
    v_other = w.eval_loss_reg(other)
    assert v_own >= 0 and v_other >= 0
    assert w.eval_loss_reg_gradient(other).shape == (w.size,)
    assert w.eval_loss_reg_gradient().shape == (w.size,)
    w.save_reg_status(2.0, 3.0)
    assert w.loss_reg_history == [3.0] and w.reg_weight_history == [2.0]
    changed = w.update_reg_weight(
        [1.0], np.ones(w.size), np.ones(w.size), 10, logging.getLogger("t")
    )
    assert isinstance(changed, bool)
    assert not w.is_adaptive_regularization()
    # weighted regularization over parameters
    (k,) = build_parameters(["K"])
    k.get_values_from_model_field(model.fl_model.permeability)
    k.regularizators = [TikhonovRegularizator(model.grid)]
    params = [w, k]
    x_raw = get_parameters_values_from_model(model, params)
    x_cond = get_parameters_values_from_model(model, params, True)
    ref = eval_weighted_loss_reg(params, model)
    assert eval_weighted_loss_reg(params, model, s_raw=x_raw) == pytest.approx(ref)
    assert eval_weighted_loss_reg(params, model, s_cond=x_cond) == pytest.approx(ref)
    eval_weighted_loss_reg(params, model, is_save_reg_state=True)
    assert len(k.loss_reg_history) == 1
    grad = eval_weighted_loss_reg_gradient(params, model)
    assert grad.shape == x_raw.shape
    assert eval_weighted_loss_reg_gradient(params, model, x_raw).shape == grad.shape


def test_model_param_values_roundtrip() -> None:
    model = build_model()
    names = ["K", "SS", "H0", "W", "D", "A", "C0", "C1", "G0", "G1"]
    params = build_parameters(names)
    update_parameters_from_model(model, params)
    for p in params:
        assert p.size == model.grid.n_grid_cells
    vals = get_parameters_values_from_model(model, params)
    assert vals.size == len(names) * model.grid.n_grid_cells
    update_model_with_parameters_values(model, vals, params, is_to_save=True)
    assert all(len(p.archived_values) == 1 for p in params)
    assert np.allclose(get_parameters_values_from_model(model, params), vals)
    pcd = get_parameters_values_from_model(model, params, is_preconditioned=True)
    update_model_with_parameters_values(model, pcd, params, is_preconditioned=True)
    assert np.allclose(get_parameters_values_from_model(model, params), vals)
    assert get_param_values(params[0]) is params[0].values
    assert np.allclose(
        get_param_values(params[0], True),
        params[0].preconditioner(params[0].values.ravel("F")),
    )


def test_param_value_errors() -> None:
    model = build_model()
    (w,) = build_parameters(["W"])
    w.name = "bad"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="not an adjustable"):
        get_parameter_values_from_model(model, w)
    with pytest.raises(ValueError, match="not a valid"):
        update_model_with_param_values(model, w)
    for key in ("C0", "G0"):
        (p,) = build_parameters([key])
        p.values = np.zeros(model.grid.shape)
        with pytest.raises(ValueError, match="sp cannot be None"):
            update_model_with_param_values(model, p, None)
    # initial pressure
    p = dminv.AdjustableParameter(
        ParameterName.INITIAL_PRESSURE, lbounds=-1e6, ubounds=1e6
    )
    p.values = np.full(model.grid.shape, 5.0)
    update_model_with_param_values(model, p)
    assert np.allclose(model.fl_model.pressure[:, :, :, 0], 5.0)


def test_gridded_gradients() -> None:
    model = build_model()
    (k,) = build_parameters(["K"])
    k.get_values_from_model_field(model.fl_model.permeability)
    g = np.arange(float(k.size))
    k.grad_adj_history.append(g)
    k.grad_adj_raw_history.append(2 * g)
    k.grad_fd_history.append(3 * g)
    shape = (*k.values.shape, 1)
    assert get_gridded_archived_gradients(k, True, True).shape == shape
    raw = get_gridded_archived_gradients(k, True, False)
    assert np.array_equal(raw[..., 0].ravel("F"), 2 * g)
    fd = get_gridded_archived_gradients(k, False)
    assert np.array_equal(fd[..., 0].ravel("F"), 3 * g)
    adj = get_backconditioned_adj_gradient(k, 0)
    fdb = get_backconditioned_fd_gradient(k, 0)
    assert adj.shape == k.values.shape
    assert np.allclose(fdb, 3 * adj)
    k.grad_fd_history[0] = np.zeros(2)
    with pytest.raises(ValueError, match="reshape"):
        get_gridded_archived_gradients(k, False)


# --------------------------------------------------------------------------- #
# InverseModel
# --------------------------------------------------------------------------- #


def test_inverse_model() -> None:
    model = build_model()
    obs = build_observables(model)
    params = build_parameters(["K", "W"])
    with pytest.raises(ValueError, match="AdjustableParameter"):
        InverseModel([], obs)
    with pytest.raises(ValueError, match="Observable"):
        InverseModel(params, [])
    im = InverseModel(params, obs)
    assert im.nb_adjusted_values == 0
    assert im.nb_obs_values == sum(o.values.size for o in obs)
    assert im.nb_f_calls == 0
    im.loss_history = [2.0, 1.0]
    im.scaling_factor = 0.5
    assert im.loss_scaled_history == [1.0, 0.5]
    im.list_losses_for_fd_grad = [1.0]
    assert im.nb_f_calls == 3
    im.loss_ls_unscaled, im.loss_reg_unscaled = 1.0, 2.0
    assert im.loss_total_unscaled == 3.0
    # optimization rounds
    assert im.is_new_optimization_round_needed(3)
    im.optimization_round_nb = 1
    assert not im.is_new_optimization_round_needed(3)
    params[0].regularizators = [TikhonovRegularizator(model.grid)]
    assert im.is_new_optimization_round_needed(3)
    im.optimization_round_nb = 3
    assert not im.is_new_optimization_round_needed(3)
    assert not im.is_adaptive_regularization()
    # scaling
    assert im.get_loss_function_scaling_factor(4.0, False) == 0.5
    assert im.get_loss_function_scaling_factor(4.0, True) == 0.25
    assert im.get_loss_function_scaling_factor(0.0, True) == 1.0
    # clear history clears the params too
    params[0].grad_fd_history.append(np.zeros(1))
    im.clear_history()
    assert im.loss_history == [] and params[0].grad_fd_history == []


def test_inverse_model_adaptive(monkeypatch) -> None:
    im = InverseModel(build_parameters(["K"]), build_observables(build_model()))
    monkeypatch.setattr(
        type(im.parameters_to_adjust[0]),
        "is_adaptive_regularization",
        lambda self: True,
    )
    assert im.is_adaptive_regularization()


def test_inverse_model_setters() -> None:
    model = build_model()
    obs = build_observables(model)
    im = InverseModel(build_parameters(["K"]), obs)
    im.set_observables(obs[:2])
    assert im.observables == obs[:2]
    im.set_observables(obs[0])
    assert im.observables == [obs[0]]
    with pytest.raises(ValueError, match="List"):
        im.set_observables(3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Sequence"):
        im.set_parameters_to_adjust(3, model)  # ty: ignore[invalid-argument-type]
    (k,) = build_parameters(["K"])
    im.set_parameters_to_adjust(k, model)
    assert im.parameters_to_adjust == [k]
    assert np.allclose(k.values, model.fl_model.permeability)
    params = build_parameters(["W", "D", "C0", "G0", "K"])
    im.set_parameters_to_adjust(params, model)
    assert all(p.values.shape == model.grid.shape for p in params)
    for key in ("SS", "H0"):
        (bad,) = build_parameters([key])
        with pytest.raises(NotImplementedError):
            im.set_parameters_to_adjust([bad], model)
    (bad,) = build_parameters(["K"])
    bad.name = "bad"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="not an adjustable"):
        im.set_parameters_to_adjust([bad], model)
