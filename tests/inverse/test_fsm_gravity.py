"""Test the forward sensitivity method for density driven flows (gravity)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from pyrtid.forward.models import TDS_LINEAR_COEFFICIENT, WATER_DENSITY
from pyrtid.inverse.fsm import FSMSolver
from pyrtid.inverse.fsm.dFds import (
    FlowFields,
    dFhds_stationary,
    dFhds_transient,
    dflow_fields,
    get_density_sensitivity,
    get_flow_fields,
    get_flow_sources,
)
from pyrtid.inverse.fsm.directions import get_directions
from pyrtid.inverse.fsm.fd import compute_fd_jacvec
from pyrtid.inverse.params import update_parameters_from_model
from tests.helpers import PARAMETER_FACTORIES as FACTORIES
from tests.helpers import build_model, build_parameters
from tests.inverse.asm.asm_utils import build_gravity_model

# The initial pressure is not part of the factories of the saturated tests.
FACTORIES["P0"] = lambda: dminv.AdjustableParameter(
    dminv.ParameterName.INITIAL_PRESSURE, lbounds=-1e7, ubounds=1e7
)

# Options of the finite differences (the default ones if not given). The default step
# is too small for the pressures and for the concentrations.
FD_OPTIONS: dict[str, dict[str, Any]] = {
    "pressure": {"eps": 1e-2},
    "concentration": {"eps": 1e-7, "accuracy": 1},
}

# Grid shapes where the vertical axis (key) has at least two cells
SHAPES = {"x": (4, 1, 3), "y": (4, 3, 1), "z": (4, 1, 3)}

VARIABLES = [
    (dminv.StateVariable.HEAD, None),
    (dminv.StateVariable.PRESSURE, None),
    (dminv.StateVariable.DENSITY, None),
    (dminv.StateVariable.CONCENTRATION, 0),
    (dminv.StateVariable.CONCENTRATION, 1),
    (dminv.StateVariable.GRADE, 0),
    (dminv.StateVariable.GRADE, 1),
]


def _model(
    axis="z",
    regime="transient",
    explicit=True,
    dense=False,
    cst_conc=False,
    skip_rt=False,
    shape=None,
):
    """Build a small density driven model. ``dense`` increases the concentrations."""
    nx, ny, nz = shape or SHAPES[axis]
    model = build_gravity_model(
        nx=nx,
        ny=ny,
        nz=nz,
        vertical_axis=dmfwd.VerticalAxis(axis),
        regime=regime,
        explicit=explicit,
        cst_conc=cst_conc,
    )
    if dense:  # strong density contrasts (the explicit chemistry does not converge)
        rng = np.random.default_rng(5)
        model.tr_model.set_initial_conc(5e-3 * (1 + rng.random(model.grid.shape)), 1)
        model.tr_model.set_initial_conc(1e-5 * (1 + rng.random(model.grid.shape)), 0)
    model.tr_model.is_skip_rt = skip_rt
    return model


def _observables(model):
    """Observe all the state variables at 3 nodes and 4 times (variable major)."""
    n_cells = model.grid.n_grid_cells
    times = np.array([0.0, 3600 * 7.0, 3600 * 13.0, model.time_params.duration])
    return [
        dminv.Observable(
            state_variable=state_variable,
            node_indices=node,
            times=times,
            values=np.zeros(times.size),
            uncertainties=1.0,
            sp=sp,
        )
        for state_variable, sp in VARIABLES
        for node in (n_cells // 4, n_cells // 2, n_cells - 2)
    ]


def _fd_jacvec(model, obs, params, vecs):
    """Return the finite differences products, parameter by parameter."""
    out, idx = 0.0, 0
    for param in params:
        size = param.values.size
        out = out + compute_fd_jacvec(
            model,
            obs,
            [param],
            vecs[idx : idx + size],
            **FD_OPTIONS.get(str(param.name), {}),
            is_save_state=False,
        )
        idx += size
    return out


def _compare(model, names, ne=2, seed=1, rtol=1e-4):
    """Compare the FSM with the finite differences for all the variables."""
    params = build_parameters(names)
    update_parameters_from_model(model, params)
    obs = _observables(model)
    n_values = sum(p.values.size for p in params)
    vecs = np.random.default_rng(seed).normal(size=(n_values, ne))
    d_pred, jac_fsm = FSMSolver(copy.deepcopy(model)).solve(obs, params, vecs)
    jac_fd = _fd_jacvec(model, obs, params, vecs)
    # For each variable, the products are compared to the largest one (the others
    # are noisy). Variations below 1e-8 times the state itself (1e3 kg/m3 for the
    # density, 1e4 Pa for the pressure) cannot be resolved by finite differences:
    # the sensitivities of the density are also validated by their linear relation
    # with the sensitivities of the concentrations.
    n_rows = jac_fd.shape[0] // len(VARIABLES)
    state_scale = {
        sv: max(
            np.abs(d_pred[i * n_rows : (i + 1) * n_rows]).max()
            for i, (var, _) in enumerate(VARIABLES)
            if var == sv
        )
        for sv, _ in VARIABLES
    }
    for i, (state_variable, _) in enumerate(VARIABLES):
        rows = slice(i * n_rows, (i + 1) * n_rows)
        fsm, fd = jac_fsm[rows], jac_fd[rows]
        scale = np.maximum(
            np.abs(fd).max(axis=1, keepdims=True), 1e-3 * np.abs(fd).max()
        )
        atol = 1e-8 * state_scale[state_variable]
        assert np.all(np.abs(fsm - fd) <= rtol * scale + atol), VARIABLES[i]
    assert np.abs(jac_fd).max() > 0.0
    return jac_fsm


@pytest.mark.parametrize("regime", ["transient", "stationary"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_fsm_gravity_flow_parameters(axis, regime) -> None:
    """Permeability, storage coefficient, initial head and pressure."""
    model = _model(axis, regime)
    dmfwd.ForwardSolver(model).solve()
    _compare(model, ["K", "SS", "H0", "P0"])


# @pytest.mark.parametrize("regime", ["transient", "stationary"])
# @pytest.mark.parametrize("axis", ["x", "y", "z"])
# def test_fsm_gravity_transport_parameters(axis, regime) -> None:
#     """Porosity, diffusion, dispersivity, initial concentrations and grades."""
#     model = _model(axis, regime)
#     dmfwd.ForwardSolver(model).solve()
#     _compare(model, ["W", "D", "A", "C0", "C1", "G0", "G1"])


@pytest.mark.parametrize("regime", ["transient", "stationary"])
@pytest.mark.parametrize("explicit", [True, False])
def test_fsm_gravity_dense_solutions(explicit, regime) -> None:
    """Strong density contrasts, with both chemistry formulations."""
    model = _model("z", regime, explicit=explicit, dense=not explicit)
    dmfwd.ForwardSolver(model).solve()
    names = ["K", "P0", "W", "C0", "C1", "G0"]
    jac = _compare(model, names)
    assert np.abs(jac).max() > 0.0


def test_fsm_gravity_dense_all_parameters() -> None:
    # The second mineral is exhausted (G1): finite differences are not defined.
    model = _model("x", dense=True, explicit=False)
    dmfwd.ForwardSolver(model).solve()
    _compare(model, ["K", "SS", "H0", "P0", "W", "D", "A", "C0", "C1"])


def test_fsm_gravity_constant_concentration() -> None:
    model = _model("z", cst_conc=True)
    dmfwd.ForwardSolver(model).solve()
    _compare(model, ["K", "P0", "C0", "C1", "G0", "W"])


def test_fsm_gravity_skip_rt() -> None:
    """The density is constant, the concentrations are not computed."""
    model = _model("z", skip_rt=True)
    dmfwd.ForwardSolver(model).solve()
    _compare(model, ["K", "SS", "H0", "P0", "C0", "C1"])


def test_fsm_gravity_3d() -> None:
    model = _model("y", shape=(3, 3, 3))
    dmfwd.ForwardSolver(model).solve()
    _compare(model, ["K", "P0", "C1", "W"], ne=1)


def test_fsm_gravity_density_sensitivity_is_linear_in_the_concentrations() -> None:
    model = _model("z", dense=True, explicit=False)
    dmfwd.ForwardSolver(model).solve()
    params = build_parameters(["K", "C0", "C1", "P0"])
    update_parameters_from_model(model, params)
    obs = _observables(model)
    n_values = sum(p.values.size for p in params)
    vecs = np.random.default_rng(0).normal(size=(n_values, 2))
    _, jac = FSMSolver(copy.deepcopy(model)).solve(obs, params, vecs)
    n_rows = jac.shape[0] // len(VARIABLES)
    rho, conc0, conc1 = (jac[i * n_rows : (i + 1) * n_rows] for i in (2, 3, 4))
    gch = model.gch_params
    expected = (
        WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * (gch.Ms * conc0 + gch.Ms2 * conc1)
        / 1000.0
    )
    assert np.abs(expected).max() > 0.0
    np.testing.assert_allclose(rho, expected, rtol=1e-10, atol=1e-14)


def test_fsm_pressure_and_head_parameters_with_and_without_gravity() -> None:
    """The initial pressure is supported without gravity (and the head with)."""
    for model in (
        build_model(nx=5, ny=2),
        build_model(nx=5, ny=2, regime="stationary"),
    ):
        dmfwd.ForwardSolver(model).solve()
        params = build_parameters(["P0", "H0"])
        update_parameters_from_model(model, params)
        n_cells = model.grid.n_grid_cells
        times = np.array([0.0, model.time_params.duration])
        obs = [
            dminv.Observable(
                state_variable=sv,
                node_indices=node,
                times=times,
                values=np.zeros(2),
                uncertainties=1.0,
            )
            for sv in (dminv.StateVariable.HEAD, dminv.StateVariable.PRESSURE)
            for node in (1, n_cells // 2)
        ]
        vecs = np.random.default_rng(0).normal(size=(2 * n_cells, 2))
        _, jac_fsm = FSMSolver(copy.deepcopy(model)).solve(obs, params, vecs)
        jac_fd = _fd_jacvec(model, obs, params, vecs)
        scale = np.abs(jac_fd).max(axis=1, keepdims=True)
        assert np.all(np.abs(jac_fsm - jac_fd) <= 1e-4 * scale + 1e-9)
        assert np.abs(jac_fd).max() > 0.0


def test_get_directions_supports_the_initial_pressure() -> None:
    model = _model("z")
    params = build_parameters(["P0", "H0"])
    update_parameters_from_model(model, params)
    n_cells = model.grid.n_grid_cells
    directions = get_directions(model, params, np.ones((2 * n_cells, 3)))
    assert directions.pressure is not None
    assert directions.pressure.shape == (*model.grid.shape, 3)
    assert directions.head is not None


def test_gravity_forcing_without_direction() -> None:
    """The null directions give null forcings and null fields."""
    model = _model("z")
    dmfwd.ForwardSolver(model).solve()
    n_cells = model.grid.n_grid_cells
    zero = np.zeros(model.grid.shape)
    dt = model.time_params.ldt[0]
    assert np.array_equal(dFhds_transient(model, 1, dt, None, None), np.zeros(n_cells))
    assert np.array_equal(
        dFhds_transient(model, 1, dt, zero, zero, zero), np.zeros(n_cells)
    )
    one = np.ones(model.grid.shape)
    assert np.any(dFhds_transient(model, 1, dt, None, None, one) != 0.0)
    unitflow = get_flow_sources(model, 1)
    fields = dflow_fields(model, 1, unitflow, zero, zero, zero)
    for arr in (fields.u_x, fields.u_y, fields.u_z, fields.div, fields.unitflow):
        assert np.array_equal(arr, np.zeros_like(arr))
    # the density is the one of the water at the time 0: no effect
    fields0 = dflow_fields(model, 0, unitflow, None, zero, one)
    assert np.array_equal(fields0.u_z, np.zeros_like(fields0.u_z))


def test_gravity_density_direction_changes_the_flow() -> None:
    model = _model("z", dense=True, explicit=False)
    dmfwd.ForwardSolver(model).solve()
    unitflow = get_flow_sources(model, 2)
    zero = np.zeros(model.grid.shape)
    one = np.ones(model.grid.shape)
    fields = dflow_fields(model, 2, unitflow, None, zero, one)
    assert np.max(np.abs(fields.u_z)) > 0.0
    # compare with the fields computed for a different density
    fl_model = model.fl_model
    rho = model.tr_model.ldensity[1]
    eps = 1e-3
    plus = get_flow_fields(
        model, 2, fl_model.permeability, fl_model.lpressure[2], unitflow, rho + eps
    )
    minus = get_flow_fields(
        model, 2, fl_model.permeability, fl_model.lpressure[2], unitflow, rho - eps
    )
    np.testing.assert_allclose(
        fields.u_z, (plus.u_z - minus.u_z) / (2 * eps), rtol=1e-6, atol=1e-18
    )
    assert isinstance(fields, FlowFields)


def test_gravity_stationary_forcing() -> None:
    model = _model("z", regime="stationary")
    dmfwd.ForwardSolver(model).solve()
    direction = model.fl_model.permeability * np.ones(model.grid.shape)
    forcing = dFhds_stationary(model, direction)
    assert np.any(forcing != 0.0)
    assert np.all(forcing[model.fl_model.cst_head_nn] == 0.0)


def test_get_density_sensitivity() -> None:
    model = _model("z")
    mob = np.random.default_rng(0).normal(size=(2, model.grid.n_grid_cells, 2))
    gch = model.gch_params
    expected = (
        WATER_DENSITY
        * TDS_LINEAR_COEFFICIENT
        * (gch.Ms * mob[0] + gch.Ms2 * mob[1])
        / 1000.0
    )
    out = get_density_sensitivity(model, mob)
    assert out.shape == (model.grid.n_grid_cells, 2)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
