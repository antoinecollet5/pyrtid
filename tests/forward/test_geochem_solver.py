"""Tests of the explicit and implicit geochemical solvers."""

from __future__ import annotations

import numpy as np
import pyrtid.forward as dmfwd
import pytest
from inv_toolbox.utils.preconditioner import LogTransform
from pyrtid.forward.geochem_solver import (
    F,
    Jacobian,
    _backtrack_batch,
    _newton_batch,
    _solve_batched,
    _solve_extent,
    get_dM,
    get_dM_derivatives,
    get_dM_pos,
    get_implicit_dM_derivatives,
    get_implicit_extent_analytical,
    get_phi,
    solve_geochem,
    solve_geochem_explicit,
    solve_geochem_implicit,
    solve_geochem_system,
)
from tests.helpers import build_model

DT = 3600.0 * 4


def _prepare(model: dmfwd.ForwardModel) -> int:
    """Initialize the model and open the time index 1 (copy of the time 0)."""
    dmfwd.ForwardSolver(model).initialize()
    tr = model.tr_model
    tr.lmob.append(tr.lmob[0].copy())
    tr.limmob.append(tr.limmob[0].copy())
    model.time_params.dt = DT
    return 1


def _set_cell(
    model: dmfwd.ForwardModel,
    cell: tuple[int, int, int],
    mob: tuple[float, float],
    immob_prev: tuple[float, float],
    immob: tuple[float, float] | None = None,
) -> None:
    tr = model.tr_model
    tr.lmob[1][(slice(None), *cell)] = mob
    tr.limmob[0][(slice(None), *cell)] = immob_prev
    tr.limmob[1][(slice(None), *cell)] = immob if immob is not None else immob_prev


@pytest.fixture
def expl_model() -> dmfwd.ForwardModel:
    """Model whose cells are set to the three regimes of the explicit chemistry."""
    model = build_model(regime="transient", explicit=True, cst_conc=True)
    _prepare(model)
    ks = model.gch_params.Ks
    # kinetics limited, mineral limited, reagent limited, saturated, negative mob1,
    # no reagent
    _set_cell(model, (1, 0, 0), (1e-6, 1e-4), (1e-1, 0.0))
    _set_cell(model, (2, 0, 0), (1e-6, 2.0), (1e-9, 0.0))
    _set_cell(model, (3, 0, 0), (1e-6, 1e-3), (2.0, 0.0))
    _set_cell(model, (4, 0, 0), (ks * 2, 1e-4), (1e-1, 0.0))
    _set_cell(model, (5, 0, 0), (-1e-6, 1e-4), (1e-1, 0.0))
    _set_cell(model, (6, 0, 0), (1e-6, -1e-4), (1e-1, 0.0))
    return model


def test_get_dM_regimes(expl_model: dmfwd.ForwardModel) -> None:
    tr, gp = expl_model.tr_model, expl_model.gch_params
    dM = get_dM(tr, gp, 1, DT)
    # kinetic limitation
    kin = DT * gp.kv * gp.As * 1e-1 * (1 - 1e-6 / gp.Ks) * 1e-4
    assert dM[1, 0, 0] == pytest.approx(kin, rel=1e-12)
    assert dM[1, 0, 0] < 0
    # all the mineral is dissolved
    assert dM[2, 0, 0] == pytest.approx(-1e-9)
    # limited by the reagent: c2 / stocoef
    assert dM[3, 0, 0] == pytest.approx(-1e-3 / gp.stocoef)
    # saturated, negative mob1, negative reagent: nothing happens
    assert dM[4, 0, 0] == 0.0
    assert dM[5, 0, 0] == 0.0
    assert dM[6, 0, 0] == 0.0
    assert np.all(dM <= 0.0)
    pos = get_dM_pos(tr, gp, 1, DT)
    assert pos[1, 0, 0] == 0
    assert pos[2, 0, 0] == 1
    assert pos[3, 0, 0] == 2


def test_get_dM_derivatives_match_finite_differences(
    expl_model: dmfwd.ForwardModel,
) -> None:
    tr, gp = expl_model.tr_model, expl_model.gch_params
    d_dmob, d_dg0, d_dg1 = get_dM_derivatives(tr, gp, 1, DT)
    assert not np.any(d_dg1)
    # cells set up for the three active regimes
    cells = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
    for cell in cells:
        for sp in (0, 1):
            ref = tr.lmob[1][(sp, *cell)]
            eps = 1e-6 * abs(ref) if ref != 0 else 1e-12
            tr.lmob[1][(sp, *cell)] = ref + eps
            up = get_dM(tr, gp, 1, DT)[cell]
            tr.lmob[1][(sp, *cell)] = ref - eps
            dn = get_dM(tr, gp, 1, DT)[cell]
            tr.lmob[1][(sp, *cell)] = ref
            assert d_dmob[(sp, *cell)] == pytest.approx(
                (up - dn) / (2 * eps), rel=1e-5, abs=1e-30
            )
        ref = tr.limmob[0][(0, *cell)]
        eps = 1e-6 * ref
        tr.limmob[0][(0, *cell)] = ref + eps
        up = get_dM(tr, gp, 1, DT)[cell]
        tr.limmob[0][(0, *cell)] = ref - eps
        dn = get_dM(tr, gp, 1, DT)[cell]
        tr.limmob[0][(0, *cell)] = ref
        assert d_dg0[cell] == pytest.approx((up - dn) / (2 * eps), rel=1e-5, abs=1e-30)
    # No dissolution -> null derivatives
    for cell in [(4, 0, 0), (5, 0, 0), (6, 0, 0)]:
        assert np.all(d_dmob[(slice(None), *cell)] == 0.0)
        assert d_dg0[cell] == 0.0
    # Constant concentration (first column) -> null derivatives
    assert np.all(d_dmob[:, 0] == 0.0)
    assert np.all(d_dg0[0] == 0.0)


def test_solve_geochem_explicit_conserves_mass_and_cst_cells(
    expl_model: dmfwd.ForwardModel,
) -> None:
    tr, gp = expl_model.tr_model, expl_model.gch_params
    # a non-zero change in the constant concentration cells must be ignored
    _set_cell(expl_model, (0, 1, 0), (1e-6, 1e-4), (1e-1, 0.0))
    ref = tr.limmob[0].copy()
    mob_ref = tr.lmob[1].copy()
    expl_model.time_params.dt = DT
    solve_geochem_explicit(tr, gp, expl_model.time_params, 1)
    new = tr.limmob[1]
    dM = new[0] - ref[0]
    # the mineral is only dissolved
    assert np.all(dM <= 0.0)
    assert np.all(new[0] >= 0.0)
    # product: stoichiometry
    np.testing.assert_allclose(new[1] - ref[1], -gp.stocoef * dM)
    # constant concentration cells are untouched
    np.testing.assert_array_equal(new[:, 0], ref[:, 0])
    # the explicit chemistry doesn't touch the mobile concentrations
    np.testing.assert_array_equal(tr.lmob[1], mob_ref)
    assert dM[1, 0, 0] < 0.0


def test_solve_geochem_explicit_negative_grade_raises(
    expl_model: dmfwd.ForwardModel,
) -> None:
    # negative mobile concentration -> nothing is dissolved, the grade stays negative
    _set_cell(expl_model, (2, 1, 0), (1e-6, -1e-4), (-1e-3, 0.0))
    with pytest.raises(RuntimeError, match="Negative mineral grade"):
        solve_geochem_explicit(
            expl_model.tr_model, expl_model.gch_params, expl_model.time_params, 1
        )


@pytest.fixture
def impl_model() -> dmfwd.ForwardModel:
    model = build_model(regime="transient", explicit=False, cst_conc=True)
    _prepare(model)
    tr = model.tr_model
    rng = np.random.default_rng(3)
    shape = model.grid.shape
    tr.lmob[1][0] = 1e-5 * rng.random(shape)
    tr.lmob[1][1] = 1e-3 * (0.1 + rng.random(shape))
    tr.limmob[0][0] = 1e-1 * (0.5 + rng.random(shape))
    tr.limmob[0][1] = 1e-4 * rng.random(shape)
    tr.limmob[1][...] = tr.limmob[0]
    return model


@pytest.mark.parametrize("method", ["newton", "analytical"])
@pytest.mark.parametrize("linesearch", [True, False])
def test_solve_geochem_implicit_is_conservative(
    impl_model: dmfwd.ForwardModel, method: str, linesearch: bool
) -> None:
    model = impl_model
    tr, gp = model.tr_model, model.gch_params
    tot_before = tr.lmob[1] + tr.limmob[1]
    mob_before = tr.lmob[1].copy()
    n_iter = solve_geochem_implicit(
        model.grid,
        tr,
        gp,
        model.time_params,
        1,
        method=method,  # ty: ignore[invalid-argument-type]
        is_use_linesearch=linesearch,
    )
    assert n_iter >= 0
    np.testing.assert_allclose(tr.lmob[1] + tr.limmob[1], tot_before, rtol=1e-10)
    # the grades of the constant concentration cells are unchanged
    np.testing.assert_array_equal(tr.limmob[1][:, 0], tr.limmob[0][:, 0])
    np.testing.assert_array_equal(tr.lmob[1][:, 0], mob_before[:, 0])
    assert np.all(tr.lmob[1] >= 0.0)
    assert np.all(tr.limmob[1] >= 0.0)
    # kinetics satisfied for the free cells where no clipping occurred
    extent = tr.limmob[0][0] - tr.limmob[1][0]
    phi = get_phi(tr.lmob[1], tr.limmob[0], gp)
    free = slice(1, None)
    np.testing.assert_allclose(extent[free], -DT * phi[free], rtol=1e-6)


def test_newton_and_analytical_agree(impl_model: dmfwd.ForwardModel) -> None:
    model = impl_model
    tr = model.tr_model
    ref_mob, ref_immob = tr.lmob[1].copy(), tr.limmob[1].copy()
    results = {}
    for method in ("newton", "analytical"):
        tr.lmob[1][...] = ref_mob
        tr.limmob[1][...] = ref_immob
        solve_geochem_implicit(
            model.grid,
            tr,
            model.gch_params,
            model.time_params,
            1,
            method=method,
        )
        results[method] = (tr.lmob[1].copy(), tr.limmob[1].copy())
    np.testing.assert_allclose(
        results["newton"][1], results["analytical"][1], rtol=1e-8, atol=1e-18
    )
    np.testing.assert_allclose(
        results["newton"][0], results["analytical"][0], rtol=1e-8, atol=1e-18
    )


def test_implicit_extent_clipped_to_physical_limits() -> None:
    """A huge timestep cannot dissolve more than the available reactant."""
    model = build_model(regime="transient", explicit=False)
    _prepare(model)
    tr, gp = model.tr_model, model.gch_params
    # very little mineral and reagent
    tr.lmob[1][0] = 1e-9
    tr.lmob[1][1] = 1e-3
    tr.limmob[0][0] = 1e-8
    tr.limmob[0][1] = 1e-9
    tr.limmob[1][...] = tr.limmob[0]
    model.time_params.dt = 1e12
    solve_geochem_implicit(model.grid, tr, gp, model.time_params, 1)
    # all the mineral is dissolved, no negative value
    np.testing.assert_allclose(tr.limmob[1][0], 0.0, atol=1e-18)
    assert np.all(tr.lmob[1] >= 0.0)
    # Precipitation case: kv > 0 and mobile species 0 in excess, limited by the
    # available product
    model2 = build_model(regime="transient", explicit=False, kv=6.9e-6)
    _prepare(model2)
    tr2 = model2.tr_model
    tr2.lmob[1][0] = 1e-5
    tr2.lmob[1][1] = 1e-3
    tr2.limmob[0][0] = 1e-1
    tr2.limmob[0][1] = 1e-9
    tr2.limmob[1][...] = tr2.limmob[0]
    model2.time_params.dt = 1e12
    solve_geochem_implicit(model2.grid, tr2, model2.gch_params, model2.time_params, 1)
    assert np.all(tr2.limmob[1][1] >= -1e-30)
    assert np.all(tr2.lmob[1] >= -1e-30)


def test_implicit_unknown_method_raises(impl_model: dmfwd.ForwardModel) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        solve_geochem_implicit(
            impl_model.grid,
            impl_model.tr_model,
            impl_model.gch_params,
            impl_model.time_params,
            1,
            method="foo",  # ty: ignore[invalid-argument-type]
        )


def test_implicit_non_convergence_warns(impl_model: dmfwd.ForwardModel) -> None:
    tr = impl_model.tr_model
    mob, immob = tr.lmob[1].copy(), tr.limmob[1].copy()
    with pytest.warns(UserWarning, match="did not converge"):
        solve_geochem_implicit(
            impl_model.grid,
            tr,
            impl_model.gch_params,
            impl_model.time_params,
            1,
            max_iter=1,
            atol=0.0,
            rtol=0.0,
        )
    # the mass is conserved anyway
    np.testing.assert_allclose(tr.lmob[1] + tr.limmob[1], mob + immob, rtol=1e-10)


def test_analytical_extent_has_no_root_and_falls_back_to_newton() -> None:
    """With kv > 0, the quadratic equation may have no real root."""
    model = build_model(regime="transient", explicit=False)
    gp = model.gch_params
    gp.kv = 1e-3  # precipitation-like kinetics
    n = 5
    mob_prev = np.array([np.full(n, 1e-6), np.full(n, 1e-3)])
    immob_prev = np.array([np.full(n, 1.0), np.zeros(n)])
    dt = 1e3
    # tune the timestep to cancel the linear term b = X - 1
    a_coef = gp.kv * gp.As * 1.0
    x = a_coef * dt * (1e-3 / gp.Ks + gp.stocoef * (1 - 1e-6 / gp.Ks))
    assert x > 0
    dt_cancel = dt / x
    extent = get_implicit_extent_analytical(mob_prev, immob_prev, gp, dt_cancel)
    assert np.all(np.isnan(extent))
    extent, n_iter, n_failed = _solve_extent(
        mob_prev,
        immob_prev,
        mob_prev,
        immob_prev,
        gp,
        dt_cancel,
        "analytical",
        1e-20,
        1e-10,
        50,
        True,
    )
    assert np.all(np.isfinite(extent))
    assert n_iter > 0
    assert n_failed >= 0


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_solve_extent_non_finite_is_set_to_zero() -> None:
    model = build_model(regime="transient", explicit=False)
    gp = model.gch_params
    mob_prev = np.array([[1e-6, np.inf], [1e-3, 1e-3]])
    immob_prev = np.array([[1e-1, 1e-1], [0.0, 0.0]])
    extent, _, n_failed = _solve_extent(
        mob_prev,
        immob_prev,
        mob_prev,
        immob_prev,
        gp,
        DT,
        "analytical",
        1e-20,
        1e-10,
        50,
        False,
    )
    assert np.isfinite(extent[0])
    assert extent[1] == 0.0
    assert n_failed >= 1


def test_solve_batched_singular_matrix_uses_pinv() -> None:
    J = np.stack([np.eye(4), np.zeros((4, 4))])
    rhs = np.ones((2, 4))
    x = _solve_batched(J, rhs)
    np.testing.assert_allclose(x[0], 1.0)
    np.testing.assert_allclose(x[1], 0.0)


def test_newton_batch_converges_and_flags_failures() -> None:
    model = build_model(regime="transient", explicit=False)
    gp = model.gch_params
    mob_prev = np.array([[1e-6, 1e-6], [1e-3, 1e-3]])
    immob_prev = np.array([[1e-1, 1e-1], [0.0, 0.0]])
    x0 = np.vstack([mob_prev, immob_prev])
    res = _newton_batch(x0, mob_prev, immob_prev, gp, DT, 1e-20, 1e-10, 50, True)
    assert res.success
    assert np.all(res.converged)
    resid = F(res.x[:2], res.x[2:], mob_prev, immob_prev, gp, DT)
    assert np.max(np.abs(resid)) < 1e-12
    # the initial guess is already the solution -> no iteration
    res2 = _newton_batch(res.x, mob_prev, immob_prev, gp, DT, 1e-10, 1e-10, 50, True)
    assert res2.nit == 0
    # a non finite guess for one cell cannot be solved
    bad = x0.copy()
    bad[0, 1] = np.nan
    res3 = _newton_batch(bad, mob_prev, immob_prev, gp, DT, 1e-20, 1e-10, 50, False)
    assert not res3.success
    assert res3.converged[0]
    assert not res3.converged[1]


def test_backtrack_batch_reduces_step_when_needed() -> None:
    model = build_model(regime="transient", explicit=False)
    gp = model.gch_params
    mob_prev = np.array([[1e-6], [1e-3]])
    immob_prev = np.array([[1e-1], [0.0]])
    xa = np.vstack([mob_prev, immob_prev])
    Fa = F(xa[:2], xa[2:], mob_prev, immob_prev, gp, 1e6)
    jac = np.moveaxis(Jacobian(xa[:2], xa[2:], mob_prev, immob_prev, gp, 1e6), -1, 0)
    # the Newton direction with a step which is much too long
    dx = 50.0 * _solve_batched(jac, Fa.T).T
    x_new = xa - dx
    F_new = F(x_new[:2], x_new[2:], mob_prev, immob_prev, gp, 1e6)
    f0 = 0.5 * np.sum(Fa**2)
    assert 0.5 * np.sum(F_new**2) > f0
    _backtrack_batch(
        xa, dx, x_new, F_new, Fa, np.array([False]), mob_prev, immob_prev, gp, 1e6
    )
    assert 0.5 * np.sum(F_new**2) < f0
    # the backtracked step is a fraction of the original one
    ratio = (xa - x_new) / dx
    assert 0.0 < ratio[0, 0] < 1.0


def _implicit_state(model: dmfwd.ForwardModel):
    """Build the arrays of a single-cell implicit problem."""
    gp = model.gch_params
    mob_prev = np.array([1e-6, 1e-3])
    immob_prev = np.array([1e-1, 1e-9])
    return gp, mob_prev, immob_prev


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"is_use_svd": True},
        {"is_use_ln": True},
        {"is_use_polish": True},
        {"is_use_ln": True, "is_use_polish": True, "is_use_svd": True},
        {"pcd": LogTransform(), "is_use_ln": True},
    ],
)
def test_solve_geochem_system_matches_analytical(kwargs) -> None:
    model = build_model(regime="transient", explicit=False)
    gp, mob_prev, immob_prev = _implicit_state(model)
    res = solve_geochem_system(
        mob_prev.copy(),
        immob_prev.copy(),
        mob_prev,
        immob_prev,
        gp,
        DT,
        **kwargs,
    )
    assert res.success
    xi = get_implicit_extent_analytical(mob_prev[:, None], immob_prev[:, None], gp, DT)[
        0
    ]
    np.testing.assert_allclose(res.x[0], mob_prev[0] + xi, rtol=1e-6)
    np.testing.assert_allclose(res.x[1], mob_prev[1] - gp.stocoef * xi, rtol=1e-6)
    np.testing.assert_allclose(res.x[2], immob_prev[0] - xi, rtol=1e-6)
    resid = F(res.x[:2], res.x[2:], mob_prev, immob_prev, gp, DT)
    assert np.max(np.abs(resid)) < 1e-12
    assert res.nfev >= 1
    assert res.njev >= 1
    assert res.nhev >= 1
    # the memoization avoids more evaluations of the Jacobian than iterations
    assert res.njev <= res.nit + 1


def test_solve_geochem_system_already_solved() -> None:
    model = build_model(regime="transient", explicit=False)
    gp, mob_prev, immob_prev = _implicit_state(model)
    res = solve_geochem_system(mob_prev, immob_prev, mob_prev, immob_prev, gp, 0.0)
    assert res.nit == 0
    assert res.nhev == 0


def test_solve_geochem_system_linesearch_failure_takes_full_step(
    monkeypatch,
) -> None:
    """If the line search fails, the full Newton step is taken."""
    import pyrtid.forward.geochem_solver as gs

    monkeypatch.setattr(gs, "backtracking_linesearch", lambda *a, **k: None)
    model = build_model(regime="transient", explicit=False)
    gp, mob_prev, immob_prev = _implicit_state(model)
    res = solve_geochem_system(
        mob_prev.copy(), immob_prev.copy(), mob_prev, immob_prev, gp, DT, is_use_ln=True
    )
    ref = solve_geochem_system(
        mob_prev.copy(), immob_prev.copy(), mob_prev, immob_prev, gp, DT
    )
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-10)
    assert res.nit == ref.nit


def test_get_implicit_dM_derivatives_identifies_limitations() -> None:
    model = build_model(regime="transient", explicit=False, cst_conc=True)
    _prepare(model)
    tr, gp = model.tr_model, model.gch_params
    nu = gp.stocoef
    coef = DT * gp.kv * gp.As
    shape = model.grid.shape
    # background: free dissolution everywhere
    c1 = np.full(shape, 1e-6)
    c2 = np.full(shape, 1e-3)
    g1p = np.full(shape, 1e-1)
    g2p = np.full(shape, 1e-6)
    tr.lmob[1][0], tr.lmob[1][1] = c1, c2
    tr.limmob[0][0], tr.limmob[0][1] = g1p, g2p
    free_ext = -coef * g1p * c2 * (1 - c1 / gp.Ks)
    tr.limmob[1][0] = g1p - free_ext
    tr.limmob[1][1] = g2p
    # (2, 1, 0): the mineral is exhausted
    tr.limmob[1][0][2, 1, 0] = 0.0
    # (3, 1, 0): the product is exhausted (precipitation)
    tr.limmob[1][0][3, 1, 0] = g1p[3, 1, 0] + g2p[3, 1, 0] / nu
    # (4, 1, 0): precipitation limited by the species 1 -> pinned 1
    tr.lmob[1][0][4, 1, 0] = 0.0
    tr.limmob[1][0][4, 1, 0] = g1p[4, 1, 0] + 0.4 * g2p[4, 1, 0] / nu
    # (5, 1, 0): dissolution limited by the species 2 -> pinned 2
    tr.lmob[1][1][5, 1, 0] = 0.0
    tr.limmob[1][0][5, 1, 0] = g1p[5, 1, 0] - 0.3 * free_ext[5, 1, 0]
    # (6, 1, 0): unknown limitation (extent has no explanation)
    tr.limmob[1][0][6, 1, 0] = g1p[6, 1, 0] - 0.5 * free_ext[6, 1, 0]
    d_dmob, d_dg0, d_dg1, pinned, n_unknown = get_implicit_dM_derivatives(tr, gp, 1, DT)
    assert pinned[4, 1, 0] == 1
    assert pinned[5, 1, 0] == 2
    assert n_unknown == 1
    assert np.count_nonzero(pinned) == 2
    assert d_dg0[2, 1, 0] == -1.0
    assert d_dg1[3, 1, 0] == pytest.approx(1.0 / nu)
    # free cell: same derivatives as the explicit case
    fac = 1 - c1[1, 1, 0] / gp.Ks
    assert d_dg0[1, 1, 0] == pytest.approx(coef * fac * c2[1, 1, 0])
    assert d_dmob[1, 1, 1, 0] == pytest.approx(coef * g1p[1, 1, 0] * fac)
    assert d_dmob[0, 1, 1, 0] == pytest.approx(
        -coef * g1p[1, 1, 0] * c2[1, 1, 0] / gp.Ks
    )
    # pinned and bounded cells have no mobile concentration derivative
    assert np.all(d_dmob[:, 4, 1, 0] == 0.0)
    assert np.all(d_dmob[:, 2, 1, 0] == 0.0)
    # constant concentration cells (first column) are all null
    assert np.all(d_dmob[:, 0] == 0.0)
    assert np.all(d_dg0[0] == 0.0)
    assert np.all(d_dg1[0] == 0.0)
    assert np.all(pinned[0] == 0)


def test_solve_geochem_dispatch(
    expl_model: dmfwd.ForwardModel, impl_model: dmfwd.ForwardModel
) -> None:
    # explicit
    tr = expl_model.tr_model
    ref = tr.limmob[1].copy()
    solve_geochem(expl_model.grid, tr, expl_model.gch_params, expl_model.time_params, 1)
    assert not np.array_equal(tr.limmob[1], ref)
    # implicit changes the mobile concentrations as well
    tr = impl_model.tr_model
    mob_ref = tr.lmob[1].copy()
    solve_geochem(impl_model.grid, tr, impl_model.gch_params, impl_model.time_params, 1)
    assert not np.array_equal(tr.lmob[1], mob_ref)


def test_explicit_and_implicit_agree_for_small_timesteps() -> None:
    expl = build_model(regime="transient", explicit=True)
    impl = build_model(regime="transient", explicit=False)
    for model in (expl, impl):
        _prepare(model)
        tr = model.tr_model
        tr.lmob[1][0] = 1e-6
        tr.lmob[1][1] = 1e-3
        tr.limmob[0][0] = 1e-1
        tr.limmob[0][1] = 1e-9
        tr.limmob[1][...] = tr.limmob[0]
        model.time_params.dt = 1.0
        solve_geochem(model.grid, tr, model.gch_params, model.time_params, 1)
    np.testing.assert_allclose(
        expl.tr_model.limmob[1], impl.tr_model.limmob[1], rtol=1e-3
    )
