"""Tests of the transport solver (with and without chemistry)."""

import numpy as np
import pyrtid.forward as dmfwd
import pytest
from pyrtid.forward.models import SparseMatrixBuilder
from pyrtid.forward.transport_solver import (
    _assemble_transport_matrices,
    _get_u_darcy_list,
    make_transport_matrices,
    solve_transport_semi_implicit,
)
from tests.forward._builders import DAY, add_wells, build_3d_model
from tests.helpers import build_model


def test_get_u_darcy_list() -> None:
    model = build_model()
    fl = model.fl_model
    assert _get_u_darcy_list(fl, 0) is fl.lu_darcy_x
    assert _get_u_darcy_list(fl, 1) is fl.lu_darcy_y
    assert _get_u_darcy_list(fl, 2) is fl.lu_darcy_z
    with pytest.raises(ValueError, match="axis"):
        _get_u_darcy_list(fl, 3)


def test_make_transport_matrices_and_given_dispersion() -> None:
    model = build_3d_model(skip_rt=False)
    add_wells(model)
    dmfwd.ForwardSolver(model).solve()
    args = (model.grid, model.tr_model, model.fl_model, 2)
    q_next, q_prev = make_transport_matrices(*args)
    ngc = model.grid.n_grid_cells
    assert q_next.shape == (ngc, ngc)
    # the matrices are the same when the dispersion is given
    from pyrtid.forward.transport_solver import get_dispersion

    disp = get_dispersion(model.tr_model, model.fl_model, 2)
    b_next = SparseMatrixBuilder((ngc, ngc))
    b_prev = SparseMatrixBuilder((ngc, ngc))
    _assemble_transport_matrices(*args, b_next, b_prev, disp=disp)
    np.testing.assert_allclose(q_next.toarray(), b_next.tocsc().toarray())
    np.testing.assert_allclose(q_prev.toarray(), b_prev.tocsc().toarray())


def _closed_1d_model(n: int = 12, kv: float = 0.0) -> dmfwd.ForwardModel:
    """No flow, no source: closed system with only diffusion."""
    model = build_3d_model(
        nx=n, ny=1, nz=1, faces=[], skip_rt=False, duration=DAY, dt=3600.0
    )
    model.gch_params.kv = kv
    c0 = np.zeros(model.grid.shape)
    c0[n // 2, 0, 0] = 1e-3
    model.tr_model.set_initial_conc(c0, sp=0)
    model.tr_model.set_initial_conc(np.zeros(model.grid.shape), sp=1)
    model.tr_model.diffusion = np.full(model.grid.shape, 1e-4)
    model.tr_model.dispersivity = np.zeros(model.grid.shape)
    return model


def test_pure_diffusion_conserves_mass_and_spreads() -> None:
    model = _closed_1d_model()
    dmfwd.ForwardSolver(model).solve()
    tr = model.tr_model
    mass = tr.porosity[:, 0, 0] * tr.lmob[0][0][:, 0, 0]
    mass_end = tr.porosity[:, 0, 0] * tr.lmob[-1][0][:, 0, 0]
    # the cells have the same volume
    assert mass_end.sum() == pytest.approx(mass.sum(), rel=1e-6)
    prof = tr.lmob[-1][0][:, 0, 0]
    assert prof.max() < tr.lmob[0][0].max()
    assert np.all(prof >= -1e-12)
    # velocity is null -> symmetric spreading (equal porosity)
    tr_por = np.unique(tr.porosity)
    assert tr_por.size == 1
    np.testing.assert_allclose(prof[2:6][::-1], prof[7:11], rtol=1e-3, atol=1e-9)
    # the profile is monotonous away from the pulse
    assert np.all(np.diff(prof[6:]) <= 1e-12)


def test_advection_moves_the_center_of_mass_at_pore_velocity() -> None:
    n = 30
    model = build_3d_model(
        nx=n,
        ny=1,
        nz=1,
        regime="stationary",
        faces=["west", "east"],
        skip_rt=False,
        duration=3600.0 * 24 * 2,
        dt=3600.0 * 3,
    )
    model.gch_params.kv = 0.0
    model.fl_model.permeability = np.full(model.grid.shape, 1e-3)
    model.tr_model.diffusion = np.full(model.grid.shape, 1e-9)
    model.tr_model.dispersivity = np.zeros(model.grid.shape)
    c0 = np.zeros(model.grid.shape)
    c0[4:7] = 1e-3
    model.tr_model.set_initial_conc(c0, sp=0)
    model.tr_model.set_initial_conc(np.zeros(model.grid.shape), sp=1)
    dmfwd.ForwardSolver(model).solve()
    fl, tr = model.fl_model, model.tr_model
    u = fl.lu_darcy_x[-1][10, 0, 0]
    assert u > 0.0
    x = (np.arange(n) + 0.5) * model.grid.dx

    def center_of_mass(t: int) -> float:
        c = tr.lmob[t][0][:, 0, 0]
        return float(np.sum(x * c) / np.sum(c))

    t_end = model.time_params.times[-1]
    shift = center_of_mass(-1) - center_of_mass(0)
    expected = u * t_end / tr.porosity[10, 0, 0]
    assert shift == pytest.approx(expected, rel=0.1)
    # the mass is conserved while the pulse is inside the domain
    assert np.sum(tr.lmob[-1][0]) == pytest.approx(np.sum(tr.lmob[0][0]), rel=1e-3)
    assert np.all(tr.lmob[-1][0] > -1e-6)


def test_skip_rt_leaves_concentrations_untouched() -> None:
    model = build_model(skip_rt=True)
    dmfwd.ForwardSolver(model).solve()
    tr = model.tr_model
    for lmob in tr.lmob[1:]:
        np.testing.assert_array_equal(lmob, tr.lmob[0])
    assert all(n == 0 for n in model.time_params.lnfpi)


def test_save_spmats_transport() -> None:
    model = build_model()
    model.tr_model.is_save_spmats = True
    dmfwd.ForwardSolver(model).solve()
    assert len(model.tr_model.l_q_next) == model.time_params.nts
    assert len(model.tr_model.l_q_prev) == model.time_params.nts


def test_transport_singular_ilu_warns(monkeypatch) -> None:
    import pyrtid.forward.transport_solver as ts

    ref = build_model()
    dmfwd.ForwardSolver(ref).solve()

    def _raise(*args, **kwargs):
        raise RuntimeError("singular")

    monkeypatch.setattr(ts, "get_super_ilu_preconditioner", _raise)
    model = build_model()
    with pytest.warns(UserWarning, match="singular in transport"):
        dmfwd.ForwardSolver(model).solve()
    assert model.tr_model.super_ilu is None
    np.testing.assert_allclose(model.tr_model.conc, ref.tr_model.conc, rtol=1e-4)


def test_transport_gmres_failure_warns() -> None:
    model = build_model()
    model.tr_model.rtol = 1e-300
    with pytest.warns(UserWarning, match="GMRES solver of the transport"):
        dmfwd.ForwardSolver(model).solve()


def _retransport(model: dmfwd.ForwardModel, t: int, accelerated: bool) -> np.ndarray:
    """Solve again the transport of the time index t, return the concentrations."""
    tr = model.tr_model
    tr.is_num_acc_for_timestep = accelerated
    model.time_params.dt = model.time_params.ldt[t - 1]
    code = solve_transport_semi_implicit(
        model.grid,
        model.fl_model,
        tr,
        tr.lsources[t],
        tr.lsources[t - 1],
        model.time_params,
        t,
        1,
    )
    assert code == 0
    return tr.lmob[t].copy()


def test_numerical_acceleration_predictor() -> None:
    model = build_model()
    dmfwd.ForwardSolver(model).solve()
    t = model.time_params.nts
    tr = model.tr_model
    # the grades of the previous timestep decreased by dissolution
    assert np.any(tr.limmob[t - 1][0] != tr.limmob[t - 2][0])
    plain = _retransport(model, t, accelerated=False)
    accelerated = _retransport(model, t, accelerated=True)
    assert not np.allclose(plain, accelerated, rtol=1e-12, atol=0.0)
    # in the fallback case (negative concentration predicted), no acceleration
    tr.limmob[t - 1][0][3, 1, 0] += 1.0  # strong precipitation in the last step
    plain = _retransport(model, t, accelerated=False)
    fallback = _retransport(model, t, accelerated=True)
    np.testing.assert_allclose(fallback, plain, rtol=1e-12, atol=1e-30)
