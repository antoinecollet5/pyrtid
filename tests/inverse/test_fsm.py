"""Test the forward sensitivity method against finite differences."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from pyrtid.inverse.fsm import FSMSolver, is_fsm_jacvec_correct
from pyrtid.inverse.fsm.dFdu import get_chemistry_relations
from pyrtid.inverse.fsm.directions import get_directions
from pyrtid.inverse.fsm.fd import compute_fd_jacvec
from pyrtid.inverse.params import update_parameters_from_model
from tests.helpers import build_model, build_observables, build_parameters
from tests.inverse.asm.asm_utils import build_gravity_model

ALL = ["K", "SS", "H0", "W", "D", "A", "C0", "C1", "G0", "G1"]


def _jacvecs(model, names, ne=2, seed=1, **kwargs):
    """Return the FSM and finite differences products with random vectors."""
    params = build_parameters(names)
    update_parameters_from_model(model, params)
    n_cells = model.grid.n_grid_cells
    obs = build_observables(model, nodes=(n_cells // 4, n_cells // 2, n_cells - 2))
    obs += _parameter_observables(model)
    n_values = sum(p.values.size for p in params)
    vecs = np.random.default_rng(seed).normal(size=(n_values, ne))
    d_pred, jac_fsm = FSMSolver(copy.deepcopy(model)).solve(obs, params, vecs, **kwargs)
    jac_fd = compute_fd_jacvec(
        model,
        obs,
        params,
        vecs,
        max_obs_time=kwargs.get("hm_end_time"),
        is_save_state=False,
    )
    return d_pred, jac_fsm, jac_fd


def _parameter_observables(model):
    """Observe the adjusted parameters themselves and the pressure."""
    times = np.array([0.0, model.time_params.duration])
    node = model.grid.n_grid_cells // 2
    return [
        dminv.Observable(
            state_variable=sv,
            node_indices=node,
            times=times,
            values=np.zeros(2),
            uncertainties=1.0,
        )
        for sv in (
            dminv.StateVariable.PERMEABILITY,
            dminv.StateVariable.STORAGE_COEFFICIENT,
            dminv.StateVariable.POROSITY,
            dminv.StateVariable.DIFFUSION,
            dminv.StateVariable.DISPERSIVITY,
            dminv.StateVariable.PRESSURE,
        )
    ]


def _assert_close(jac_fsm, jac_fd, rtol=1e-4):
    scale = np.maximum(
        np.abs(jac_fd).max(axis=1, keepdims=True), 1e-3 * np.abs(jac_fd).max()
    )
    assert np.max(np.abs(jac_fsm - jac_fd) / (scale + 1e-300)) < rtol


@pytest.mark.parametrize(
    "kwargs, names",
    [
        (dict(), ["K", "SS", "H0"]),
        (dict(), ["W", "D", "A"]),
        (dict(), ["C0", "C1", "G0", "G1"]),
        (dict(regime="stationary"), ["K", "W", "D", "A", "C0", "G0"]),
        (dict(cst_conc=True), ["K", "C0", "C1", "G0", "G1", "W"]),
        (dict(nx=4, ny=3, nz=3), ["K", "W", "A", "C0", "G0"]),
        (dict(nx=10, ny=1, nz=1), ["K", "SS", "D", "C1", "G1"]),
        (dict(skip_rt=True), ["K", "SS", "H0"]),
    ],
)
def test_fsm_explicit_chemistry(kwargs, names) -> None:
    model = build_model(**kwargs)
    dmfwd.ForwardSolver(model).solve()
    d_pred, jac_fsm, jac_fd = _jacvecs(model, names)
    _assert_close(jac_fsm, jac_fd)
    assert d_pred.shape[0] == jac_fsm.shape[0]


@pytest.mark.parametrize("grade2", [1e-3, 3e-6])
@pytest.mark.parametrize(
    "names", [["K", "D", "A", "W"], ["C0", "C1", "G0", "G1", "SS", "H0"]]
)
def test_fsm_implicit_chemistry(grade2, names) -> None:
    # grade2 = 3e-6 -> the product is exhausted in some grid cells
    model = build_model(explicit=False, grade2=grade2)
    dmfwd.ForwardSolver(model).solve()
    _, jac_fsm, jac_fd = _jacvecs(model, names)
    _assert_close(jac_fsm, jac_fd)


def test_fsm_hm_end_time() -> None:
    model = build_model()
    dmfwd.ForwardSolver(model).solve()
    _, jac_fsm, jac_fd = _jacvecs(model, ["K", "C0"], hm_end_time=3600 * 14.0)
    assert jac_fsm.shape == jac_fd.shape
    _assert_close(jac_fsm, jac_fd)


def test_fsm_verbose(capsys) -> None:
    model = build_model(nx=4, ny=2)
    dmfwd.ForwardSolver(model).solve()
    _jacvecs(model, ["K"], is_verbose=True)


def test_fsm_gravity_not_implemented() -> None:
    model = build_gravity_model()
    params = build_parameters(["K"])
    update_parameters_from_model(model, params)
    with pytest.raises(NotImplementedError, match="density"):
        FSMSolver(model).solve([], params, np.ones((params[0].values.size, 1)))


def test_is_fsm_jacvec_correct() -> None:
    model = build_model(nx=5, ny=2)
    params = build_parameters(["K", "C0"])
    update_parameters_from_model(model, params)
    obs = build_observables(model, nodes=(1, 4))
    n_values = sum(p.values.size for p in params)
    vecs = np.random.default_rng(0).normal(size=(n_values, 2))
    assert is_fsm_jacvec_correct(model, params, obs, vecs, eps=1e-6)


def test_get_directions_errors() -> None:
    model = build_model(nx=4, ny=2)
    params = build_parameters(["K"])
    update_parameters_from_model(model, params)
    n = params[0].values.size
    with pytest.raises(ValueError, match="2D"):
        get_directions(model, params, np.ones(n))
    with pytest.raises(ValueError, match="more values"):
        get_directions(model, params, np.ones((n - 1, 1)))
    with pytest.raises(ValueError, match="rows"):
        get_directions(model, params, np.ones((n + 1, 1)))
    unsupported = dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_PRESSURE, lbounds=-1e3, ubounds=1e3
    )
    unsupported.values = np.ones(model.grid.shape)
    with pytest.raises(NotImplementedError, match="not supported"):
        get_directions(model, unsupported, np.ones((n, 1)))


def test_get_directions_accepts_a_single_parameter() -> None:
    model = build_model(nx=4, ny=2)
    param = build_parameters(["K"])[0]
    update_parameters_from_model(model, [param])
    directions = get_directions(model, param, np.ones((param.values.size, 3)))
    assert directions.ne == 3


def test_implicit_chemistry_unknown_branch_warns(monkeypatch) -> None:
    import pyrtid.inverse.fsm.dFdu as dfdu

    model = build_model(nx=4, ny=2, explicit=False)
    dmfwd.ForwardSolver(model).solve()
    original = dfdu.get_implicit_dM_derivatives

    def _patched(*args, **kwargs):
        *out, _ = original(*args, **kwargs)
        return (*out, 2)

    monkeypatch.setattr(dfdu, "get_implicit_dM_derivatives", _patched)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        get_chemistry_relations(model, 1, model.time_params.ldt[0])
    assert any("could not be identified" in str(w.message) for w in record)


def test_chemistry_relations_of_pinned_cells(monkeypatch) -> None:
    """The cells where a species is exhausted are replaced by 2 relations."""
    import pyrtid.inverse.fsm.dFdu as dfdu

    model = build_model(nx=4, ny=2, explicit=False)
    dmfwd.ForwardSolver(model).solve()
    n_cells = model.grid.n_grid_cells
    pinned = np.zeros(model.grid.shape, dtype=int)
    pinned[0, 0, 0], pinned[1, 0, 0] = 1, 2

    def _patched(*args, **kwargs):
        return (
            np.ones((2, *model.grid.shape)),
            np.ones(model.grid.shape),
            np.ones(model.grid.shape),
            pinned,
            0,
        )

    monkeypatch.setattr(dfdu, "get_implicit_dM_derivatives", _patched)
    l_mob, l_immob, l_prev = get_chemistry_relations(model, 1, 3600.0)
    nu = model.gch_params.stocoef
    # c_1 = 0 in the first cell, c_2 = 0 in the second one
    assert np.allclose(l_mob[0, :, 0], [1.0, 0.0])
    assert np.allclose(l_mob[0, :, 1], [0.0, 1.0])
    for cell in (0, 1):
        assert np.allclose(l_immob[0, :, cell], 0.0)
        assert np.allclose(l_prev[0, :, cell], 0.0)
        # stoichiometry
        assert np.allclose(l_mob[1, :, cell], 0.0)
        assert np.allclose(l_immob[1, :, cell], [nu, 1.0])
        assert np.allclose(l_prev[1, :, cell], [nu, 1.0])
    # the other cells are not modified
    assert np.allclose(l_mob[0, :, 2:], -1.0)
    assert l_mob.shape == (2, 2, n_cells)


def test_fd_warns_when_a_value_is_on_a_bound() -> None:
    model = build_model(nx=4, ny=2)
    param = build_parameters(["G0"])[0]
    update_parameters_from_model(model, [param])
    param.lbounds = float(np.min(param.values))  # the values are on the bound
    obs = build_observables(model, nodes=(1, 4))
    with pytest.warns(UserWarning, match="equal the lower and/or upper bound"):
        compute_fd_jacvec(
            model,
            obs,
            [param],
            np.ones((param.values.size, 1)),
            is_save_state=False,
        )


def test_fsm_density_is_linear_in_the_concentrations() -> None:
    """The density sensitivity derives from the one of the concentrations."""
    from pyrtid.forward.models import TDS_LINEAR_COEFFICIENT, WATER_DENSITY

    model = build_model(nx=4, ny=2)
    dmfwd.ForwardSolver(model).solve()
    params = build_parameters(["C0", "C1"])
    update_parameters_from_model(model, params)
    times = np.array([0.0, 3600 * 13.0, model.time_params.duration])
    node = 3

    def _obs(state_variable, sp=None):
        return dminv.Observable(
            state_variable=state_variable,
            node_indices=node,
            times=times,
            values=np.zeros(times.size),
            uncertainties=1.0,
            sp=sp,
        )

    obs = [
        _obs(dminv.StateVariable.DENSITY),
        _obs(dminv.StateVariable.CONCENTRATION, 0),
        _obs(dminv.StateVariable.CONCENTRATION, 1),
    ]
    vecs = np.random.default_rng(0).normal(size=(2 * model.grid.n_grid_cells, 2))
    _, jac = FSMSolver(copy.deepcopy(model)).solve(obs, params, vecs)
    gch = model.gch_params
    expected = (
        WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * (gch.Ms * jac[3:6] + gch.Ms2 * jac[6:9])
        / 1000.0
    )
    np.testing.assert_allclose(jac[0:3], expected, rtol=1e-10, atol=1e-14)


def test_stationary_flow_forcing_without_direction() -> None:
    from pyrtid.inverse.fsm.dFds import dFhds_stationary

    model = build_model(nx=4, ny=2, regime="stationary")
    dmfwd.ForwardSolver(model).solve()
    n_cells = model.grid.n_grid_cells
    assert np.array_equal(dFhds_stationary(model, None), np.zeros(n_cells))
    zero = np.zeros(model.grid.shape)
    assert np.array_equal(dFhds_stationary(model, zero), np.zeros(n_cells))
    direction = np.ones(model.grid.shape) * model.fl_model.permeability
    assert np.any(dFhds_stationary(model, direction) != 0.0)


@pytest.mark.parametrize("names", [["K"], ["SS"], ["W"]])
def test_fsm_single_parameter_small_grid(names) -> None:
    """The sensitivities of one parameter only (the other directions are None)."""
    model = build_model(nx=5, ny=2)
    dmfwd.ForwardSolver(model).solve()
    _, jac_fsm, jac_fd = _jacvecs(model, names)
    _assert_close(jac_fsm, jac_fd)


def test_fsm_null_directions_give_null_sensitivities() -> None:
    """The perturbation steps are not defined for null directions: skip them."""
    model = build_model(nx=5, ny=2)
    dmfwd.ForwardSolver(model).solve()
    params = build_parameters(["K", "SS", "W", "D", "A", "C0", "G0"])
    update_parameters_from_model(model, params)
    obs = build_observables(model, nodes=(1, 4))
    n_values = sum(p.values.size for p in params)
    _, jac = FSMSolver(copy.deepcopy(model)).solve(obs, params, np.zeros((n_values, 2)))
    assert np.array_equal(jac, np.zeros_like(jac))


def test_chemistry_relations_without_transport() -> None:
    model = build_model(nx=4, ny=2, skip_rt=True)
    dmfwd.ForwardSolver(model).solve()
    l_mob, l_immob, l_prev = get_chemistry_relations(model, 1, 3600.0)
    n_cells = model.grid.n_grid_cells
    assert l_mob.shape == l_immob.shape == l_prev.shape == (2, 2, n_cells)
    # nothing but the identity of the grades
    assert np.allclose(l_immob[1, 1], 1.0)


def test_transport_forcing_without_flow_direction() -> None:
    """Only the porosity direction acts when the flow fields do not vary."""
    from pyrtid.inverse.fsm.dFds import FlowFields, dFcds

    model = build_model(nx=4, ny=2)
    dmfwd.ForwardSolver(model).solve()
    fields = FlowFields.from_model(model.fl_model, 1)
    fields_old = FlowFields.from_model(model.fl_model, 0)
    dt = model.time_params.ldt[0]
    direction = 0.1 * np.ones(model.grid.shape)
    out = dFcds(model, 1, dt, fields_old, fields, None, None, direction, None, None)
    assert out.shape == (2, model.grid.n_grid_cells)
    assert np.any(out != 0.0)
    zero = FlowFields.zeros(model.grid, 1).column(0)
    out0 = dFcds(model, 1, dt, fields_old, fields, zero, zero, None, None, None)
    assert np.array_equal(out0, np.zeros_like(out0))
