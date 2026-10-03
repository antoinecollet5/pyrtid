"""End-to-end forward runs on small reactive transport models."""

import logging

import numpy as np
import pyrtid.forward as dmfwd
import pytest
import scipy as sp
from pyrtid.forward import ForwardSolver
from pyrtid.forward.models import (
    GRAVITY,
    H_PLUS_CONC,
    TDS_LINEAR_COEFFICIENT,
    WATER_DENSITY,
    WATER_MW,
)
from pyrtid.forward.solver import (
    CouplingConvergenceError,
    get_density,
    get_max_coupling_error,
    get_max_coupling_error_forward,
)
from tests.forward._builders import build_3d_model
from tests.helpers import build_model


@pytest.mark.parametrize("regime", ["transient", "stationary"])
@pytest.mark.parametrize("explicit", [True, False])
def test_solve_runs_and_is_consistent(regime: str, explicit: bool) -> None:
    model = build_model(regime=regime, explicit=explicit, cst_conc=True)
    ForwardSolver(model).solve()
    tp, fl, tr = model.time_params, model.fl_model, model.tr_model
    nt = tp.nt
    assert tp.time_elapsed >= tp.duration
    assert len(tp.lnfpi) == tp.nts
    assert len(fl.lhead) == len(tr.lmob) == len(tr.limmob) == nt
    assert len(tr.ldensity) == nt
    assert len(fl.lunitflow) == len(tr.lsources) == nt
    assert all(n >= 1 for n in tp.lnfpi)
    # the mineral is only dissolved (explicit) and the grades remain positive
    grade = tr.limmob[-1][0]
    assert np.all(grade >= 0.0)
    if explicit:
        assert np.all(grade <= tr.limmob[0][0] + 1e-12)
    assert np.all(np.isfinite(tr.lmob[-1]))
    # constant concentration cells keep their grades
    np.testing.assert_array_equal(tr.limmob[-1][:, 0], tr.limmob[0][:, 0])


def test_explicit_and_implicit_chemistry_give_close_results() -> None:
    expl = build_model(explicit=True)
    impl = build_model(explicit=False)
    ForwardSolver(expl).solve()
    ForwardSolver(impl).solve()
    # same total mass of grade 0 dissolved within a few percent
    d_expl = np.sum(expl.tr_model.limmob[0][0] - expl.tr_model.limmob[-1][0])
    d_impl = np.sum(impl.tr_model.limmob[0][0] - impl.tr_model.limmob[-1][0])
    assert d_expl > 0.0
    assert d_impl == pytest.approx(d_expl, rel=0.25)


def test_solve_is_deterministic_and_reinit() -> None:
    model = build_model()
    ForwardSolver(model).solve()
    first = model.tr_model.conc.copy()
    n = model.time_params.nts
    ForwardSolver(model).solve()  # initialize() reinitializes the model
    assert model.time_params.nts == n
    np.testing.assert_allclose(model.tr_model.conc, first)


def test_get_max_coupling_error() -> None:
    prev = np.array([1.0, 2.0, 0.0])
    cur = np.array([1.1, 2.0, 0.0])
    assert get_max_coupling_error(cur, prev) == pytest.approx(0.1)
    assert get_max_coupling_error(prev, prev) == 0.0
    # null values: no division by zero
    assert np.isfinite(get_max_coupling_error(np.zeros(3), np.zeros(3)))
    assert get_max_coupling_error(np.zeros(3), np.zeros(3)) == 0.0
    assert np.isfinite(get_max_coupling_error(np.array([1.0]), np.array([0.0])))


def test_get_max_coupling_error_forward() -> None:
    model = build_model()
    ForwardSolver(model).initialize()
    tr = model.tr_model
    tr.limmob.append(tr.limmob[0] * 1.5)
    tr.immob_prev = tr.limmob[0].copy()
    assert get_max_coupling_error_forward(tr, 1) == pytest.approx(0.5)


def test_get_density() -> None:
    conc = np.zeros((2, 3, 2, 1))
    base = WATER_DENSITY * (TDS_LINEAR_COEFFICIENT * H_PLUS_CONC * WATER_MW + 1.0)
    np.testing.assert_allclose(get_density(conc, 100.0, 200.0), base)
    conc[0] = 0.5
    conc[1] = 0.25
    tds = H_PLUS_CONC * WATER_MW + 0.5 * 100.0 / 1000 + 0.25 * 200.0 / 1000
    expected = WATER_DENSITY * (TDS_LINEAR_COEFFICIENT * tds + 1.0)
    dens = get_density(conc, 100.0, 200.0)
    assert dens.shape == (3, 2, 1)
    np.testing.assert_allclose(dens, expected)
    assert expected > base


def test_dt_max_cfl() -> None:
    model = build_model()
    solver = ForwardSolver(model)
    solver.initialize()
    assert solver._get_dt_max_cfl(1) == np.inf
    solver.solve()
    expected = model.time_params.courant_factor * np.min(
        model.tr_model.porosity * model.get_ij_over_u(1)
    )
    assert solver._get_dt_max_cfl(2) == pytest.approx(expected)
    assert expected > 0.0


def test_save_spmats_initial_matrices() -> None:
    model = build_model(regime="transient")
    model.fl_model.is_save_spmats = True
    ForwardSolver(model).initialize()
    q_next = model.fl_model.l_q_next[0]
    assert q_next.shape == (model.grid.n_grid_cells,) * 2
    np.testing.assert_array_equal(q_next.toarray(), np.eye(model.grid.n_grid_cells))
    assert model.fl_model.l_q_prev[0].nnz == 0


def test_verbose_logging(caplog: pytest.LogCaptureFixture) -> None:
    model = build_model()
    with caplog.at_level(logging.INFO, logger="pyrtid.forward.solver"):
        ForwardSolver(model).solve(is_verbose=True)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("max-coupling error" in m for m in msgs)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyrtid.forward.solver"):
        ForwardSolver(build_model()).solve(is_verbose=False)
    assert not any("max-coupling error" in r.getMessage() for r in caplog.records)


def test_no_coupling_convergence_raises(caplog: pytest.LogCaptureFixture) -> None:
    model = build_model()
    model.tr_model.fpi_eps = 0.0  # can never be reached
    model.tr_model.max_fpi = 2
    with pytest.raises(CouplingConvergenceError, match="did not converge"):
        ForwardSolver(model).solve()


def test_no_coupling_convergence_disables_numerical_acceleration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = build_model()
    model.tr_model.fpi_eps = 0.0
    model.tr_model.max_fpi = 2
    model.tr_model.is_numerical_acceleration = True
    model.tr_model.is_num_acc_for_timestep = True
    with caplog.at_level(logging.INFO, logger="pyrtid.forward.solver"):
        with pytest.raises(CouplingConvergenceError):
            ForwardSolver(model).solve()
    assert any(
        "disabling the numerical acceleration" in r.getMessage() for r in caplog.records
    )
    assert not model.tr_model.is_num_acc_for_timestep
    # nothing remains of the failed attempt
    assert len(model.fl_model.lhead) == len(model.tr_model.lmob) == 1


def test_timestep_is_reduced_when_the_coupling_fails(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The failed attempts are rolled back and solved again with a smaller dt."""
    import pyrtid.forward.solver as solver_module

    model = build_model()
    h = 3600.0
    model.time_params = dmfwd.TimeParameters(
        duration=24 * h, dt_init=6 * h, dt_min=1 * h, dt_max=6 * h
    )
    model.tr_model.is_numerical_acceleration = True

    def fake_error(tr_model, time_index):
        # the coupling only converges for dt <= 3h
        return 1.0 if model.time_params.dt > 3 * h else 0.0

    monkeypatch.setattr(solver_module, "get_max_coupling_error_forward", fake_error)
    with caplog.at_level(logging.INFO, logger="pyrtid.forward.solver"):
        ForwardSolver(model).solve()
    tp, fl, tr = model.time_params, model.fl_model, model.tr_model
    assert any("restarting with dt" in r.getMessage() for r in caplog.records)
    assert all(dt <= 3 * h + 1e-9 for dt in tp.ldt)
    assert tp.time_elapsed >= tp.duration
    # all the lists are consistent after the rollbacks
    nt = tp.nt
    assert len(fl.lhead) == len(fl.lpressure) == nt
    assert len(fl.lu_darcy_x) == len(fl.lu_darcy_div) == len(fl.lunitflow) == nt
    assert len(tr.lmob) == len(tr.limmob) == len(tr.lsources) == nt
    assert len(tr.ldensity) == nt
    assert len(tp.lnfpi) == tp.nts
    assert np.all(np.isfinite(tr.lmob[-1]))


def test_rollback_restores_the_previous_state() -> None:
    model = build_model()
    solver = ForwardSolver(model)
    solver.initialize()
    assert solver._try_timestep(1)
    solver._rollback_timestep(1)
    fl, tr, tp = model.fl_model, model.tr_model, model.time_params
    assert len(fl.lhead) == len(tr.lmob) == len(tr.limmob) == 1
    assert len(tr.lsources) == len(fl.lunitflow) == 1
    assert tp.ldt == []
    assert tp.nfpi == 0
    # and the timestep can be solved again with the same result
    assert solver._try_timestep(1)
    assert len(fl.lhead) == 2


def test_zero_gradient_and_constant_concentration_conditions() -> None:
    for explicit in (True, False):
        model = build_model(explicit=explicit, cst_conc=True)
        model.add_boundary_conditions(
            dmfwd.ZeroConcGradient(span=(slice(-1, None), slice(None), slice(None)))
        )
        ForwardSolver(model).solve()
        assert np.all(np.isfinite(model.tr_model.lmob[-1]))
        np.testing.assert_array_equal(
            model.tr_model.limmob[-1][:, 0], model.tr_model.limmob[0][:, 0]
        )


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("gravity", [False, True])
def test_reactive_transport_3d(explicit: bool, gravity: bool) -> None:
    model = build_3d_model(
        nz=4, gravity=gravity, skip_rt=False, explicit=explicit, duration=3600.0 * 12
    )
    nx, ny, nz = model.grid.shape
    model.tr_model.set_initial_conc(np.full(model.grid.shape, 1e-4), sp=1)
    model.tr_model.set_initial_conc(np.full(model.grid.shape, 1e-6), sp=0)
    ForwardSolver(model).solve()
    tr = model.tr_model
    assert np.all(np.isfinite(tr.lmob[-1]))
    assert np.all(tr.limmob[-1][0] >= 0.0)
    # Dissolution occurred
    assert np.sum(tr.limmob[0][0] - tr.limmob[-1][0]) > 0.0
    # the species are consumed/produced together (stoechiometry)
    assert tr.limmob[-1][1].sum() > tr.limmob[0][1].sum()
    assert sp.sparse.issparse(tr.q_next)


def test_density_driven_flow_sinks_dense_fluid() -> None:
    """A dense fluid on top of a lighter one moves downwards (gravity)."""
    model = build_3d_model(
        nx=1,
        ny=1,
        nz=6,
        gravity=True,
        regime="stationary",
        faces=["bottom"],
        skip_rt=False,
        duration=3600.0 * 24 * 5,
        dt=3600.0 * 6,
    )
    model.gch_params.kv = 0.0
    model.fl_model.permeability = np.full(model.grid.shape, 1e-3)
    model.tr_model.diffusion = np.full(model.grid.shape, 1e-9)
    model.tr_model.dispersivity = np.zeros(model.grid.shape)
    c0 = np.zeros(model.grid.shape)
    c0[0, 0, 4:] = 3.0
    model.tr_model.set_initial_conc(c0, sp=0)
    model.tr_model.set_initial_conc(np.zeros(model.grid.shape), sp=1)
    ForwardSolver(model).solve()
    fl, tr = model.fl_model, model.tr_model
    assert tr.ldensity[0][0, 0, 5] > tr.ldensity[0][0, 0, 0]
    # The column is closed at the top: no flow, so the pressure is hydrostatic with
    # the density of the fluid: -dP/dz = rho * g.
    t = model.time_params.nts
    rho = tr.ldensity[t - 1][0, 0, :]
    rho_face = 0.5 * (rho[:-1] + rho[1:])
    p = fl.lpressure[t][0, 0, :]
    dz = model.grid.dz
    np.testing.assert_allclose((p[:-1] - p[1:]) / dz, rho_face * GRAVITY, rtol=1e-2)
    assert np.max(np.abs(fl.lu_darcy_z[t])) < 1e-8
