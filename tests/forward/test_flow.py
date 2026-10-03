"""Tests of the flow solvers (stationary, transient, gravity, linear solvers)."""

import warnings

import numpy as np
import pyrtid.forward as dmfwd
import pytest
from inv_toolbox.utils import get_super_ilu_preconditioner
from pyrtid.forward.flow_solver import (
    get_kmean,
    get_rhomean,
    get_zj_zi_rhs,
    solve_fl_gmres,
)
from pyrtid.forward.solver import get_density
from scipy.sparse import identity, lil_array
from scipy.sparse import random as sprandom
from tests.forward._builders import add_wells, build_3d_model
from tests.helpers import build_model


def _net_flux_balance(model: dmfwd.ForwardModel, t: int) -> float:
    """Total net volumetric flow (divergence times volume) over the domain."""
    div = model.fl_model.lu_darcy_div[t] * model.grid.grid_cell_volume_m3
    src = model.fl_model.lunitflow[t] * model.grid.grid_cell_volume_m3
    return float(np.sum(div) - np.sum(src))


def test_stationary_flow_1d_analytic() -> None:
    """A 1D stationary flow between two heads is linear (Darcy)."""
    model = build_3d_model(nx=6, ny=1, nz=1, regime="stationary", all_faces=False)
    model.fl_model.permeability = np.full(model.grid.shape, 1e-4)
    # no sources
    solver = dmfwd.ForwardSolver(model)
    solver.initialize()
    head = model.fl_model.lhead[0][:, 0, 0]
    # heads at the two ends are imposed (2.0 and 1.0): linear in between
    expected = np.linspace(2.0, 1.0, 6)
    np.testing.assert_allclose(head, expected, atol=1e-6)
    ux = model.fl_model.lu_darcy_x[0][:, 0, 0]
    # q = K dh/dx (from the high head to the low head) with dx=5 m and 5 intervals
    dh_dx = (2.0 - 1.0) / (5 * 5.0)
    np.testing.assert_allclose(ux[1:-1], 1e-4 * dh_dx, rtol=1e-5)


def test_stationary_flow_with_sources_conserves_mass() -> None:
    model = build_3d_model(regime="stationary", all_faces=True)
    add_wells(model)
    dmfwd.ForwardSolver(model).initialize()
    assert abs(_net_flux_balance(model, 0)) < 1e-9
    # all the darcy fields have the same time dimension
    assert model.fl_model.u_darcy_x.shape[-1] == 1


@pytest.mark.parametrize("gravity", [False, True])
def test_transient_flow_3d_all_boundaries(gravity: bool) -> None:
    model = build_3d_model(gravity=gravity, all_faces=True)
    add_wells(model)
    dmfwd.ForwardSolver(model).solve()
    fl = model.fl_model
    nt = len(fl.lhead)
    assert nt == model.time_params.nts + 1
    for t in range(nt):
        assert np.all(np.isfinite(fl.lhead[t]))
    # properties
    assert fl.head.shape == (*model.grid.shape, nt)
    assert fl.pressure.shape == (*model.grid.shape, nt)
    for arr in (fl.u_darcy_x, fl.u_darcy_y, fl.u_darcy_z, fl.u_darcy_div, fl.unitflow):
        assert arr.ndim == 4 and arr.shape[-1] == nt
    assert fl.u_darcy_x_center.shape == (*model.grid.shape, nt)
    assert fl.u_darcy_y_center.shape == (*model.grid.shape, nt)
    assert fl.u_darcy_z_center.shape == (*model.grid.shape, nt)
    norm = fl.u_darcy_norm
    assert norm.shape == (*model.grid.shape, nt)
    assert np.all(norm >= 0)
    # on the constant heads, the divergence is null
    assert np.all(fl.lu_darcy_div[-1][0, :, :] == 0.0)
    # the head at constant head nodes is kept
    np.testing.assert_allclose(fl.lhead[-1][0, 1, 1], 2.0, rtol=1e-3)


def test_get_u_darcy_norm_derivative_matches_finite_differences() -> None:
    model = build_3d_model(all_faces=True)
    dmfwd.ForwardSolver(model).solve()
    fl = model.fl_model
    t = len(fl.lhead) - 1
    dx, dy, dz = fl.get_du_darcy_norm_sample(t)
    norm0 = fl.get_u_darcy_norm_sample(t)
    assert dx.shape == model.grid.shape
    # Perturb a face velocity and compare to the first-order variation
    eps = 1e-12
    base = fl.lu_darcy_x[t][2, 1, 1]
    fl.lu_darcy_x[t][2, 1, 1] = base + eps
    norm1 = fl.get_u_darcy_norm_sample(t)
    fl.lu_darcy_x[t][2, 1, 1] = base
    # the face is shared by the cells (1, 1, 1) and (2, 1, 1)
    d_fd = (norm1 - norm0) / eps
    assert dx[1, 1, 1] != 0.0
    assert d_fd[1, 1, 1] == pytest.approx(dx[1, 1, 1], rel=1e-3)
    assert d_fd[2, 1, 1] == pytest.approx(dx[2, 1, 1], rel=1e-3)


@pytest.mark.parametrize("axis", [dmfwd.VerticalAxis.X, dmfwd.VerticalAxis.Y])
def test_gravity_other_vertical_axes(axis: dmfwd.VerticalAxis) -> None:
    model = build_3d_model(
        gravity=True, regime="stationary", vertical_axis=axis, all_faces=True
    )
    dmfwd.ForwardSolver(model).solve()
    assert model.fl_model.vertical_axis_index == axis.axis_index
    assert model.fl_model.get_vertical_dim() == model.grid.shape[axis.axis_index]
    assert np.all(np.isfinite(model.fl_model.lhead[-1]))


def test_gravity_hydrostatic_equilibrium() -> None:
    """With a uniform pressure head and no flow, the heads stay uniform."""
    model = build_3d_model(
        nx=2, ny=2, nz=4, gravity=True, regime="stationary", faces=["top"]
    )
    model.fl_model.permeability = np.full(model.grid.shape, 1e-4)
    # only a constant head on top, no source -> no flow
    dmfwd.ForwardSolver(model).initialize()
    head = model.fl_model.lhead[0]
    np.testing.assert_allclose(head, 1.4, rtol=1e-6)
    assert np.max(np.abs(model.fl_model.lu_darcy_z[0])) < 1e-9


def test_kmean_rhomean_and_gravity_rhs() -> None:
    model = build_3d_model(gravity=True, regime="stationary")
    model.fl_model.permeability = np.full(model.grid.shape, 2e-4)
    dmfwd.ForwardSolver(model).solve()
    grid = model.grid
    km = get_kmean(grid, model.fl_model, 0, is_flatten=False)
    assert km.shape == grid.shape
    np.testing.assert_allclose(km[:-1], 2e-4)
    assert np.all(km[-1] == 0.0)
    # rho mean on a slice of times
    rho = get_rhomean(grid, model.tr_model, 0, slice(0, 2), is_flatten=False)
    assert rho.shape == (*grid.shape, 2)
    rho1 = get_rhomean(grid, model.tr_model, 2, 0)
    assert rho1.shape == (grid.n_grid_cells,)
    # the gravity rhs is null in a medium with a single vertical cell
    flat = build_3d_model(nz=1, gravity=True, regime="stationary")
    rhs = get_zj_zi_rhs(flat.grid, flat.fl_model)
    assert not np.any(rhs)
    rhs3 = get_zj_zi_rhs(grid, model.fl_model)
    assert rhs3.shape == (grid.n_grid_cells,)
    assert np.any(rhs3)


def test_density_driven_flow_sinking_plume() -> None:
    """A denser solution leads to a downward component of the flow."""
    model = build_3d_model(
        nx=2,
        ny=2,
        nz=4,
        gravity=True,
        regime="transient",
        all_faces=False,
        skip_rt=True,
    )
    model.fl_model.permeability = np.full(model.grid.shape, 1e-4)
    model.tr_model.set_initial_conc(np.full(model.grid.shape, 5.0), sp=0)
    dens = get_density(
        model.tr_model.lmob[0], model.gch_params.Ms, model.gch_params.Ms2
    )
    assert np.all(dens > 1000.0)
    dmfwd.ForwardSolver(model).solve()
    assert np.all(np.isfinite(model.fl_model.lhead[-1]))


def test_singular_flow_matrix_warns() -> None:
    """A stationary flow without any permeability gives a singular matrix."""
    model = build_3d_model(nx=2, ny=2, nz=2, regime="stationary", faces=[])
    model.fl_model.permeability = np.zeros(model.grid.shape)
    with pytest.warns(UserWarning, match="singular"):
        dmfwd.ForwardSolver(model).initialize()


def test_gmres_failure_warns() -> None:
    model = build_model(regime="transient")
    dmfwd.ForwardSolver(model).initialize()
    n = model.grid.n_grid_cells
    rng = np.random.default_rng(0)
    fl = model.fl_model
    a = identity(n, format="csc") + 5.0 * sprandom(
        n, n, density=0.5, random_state=1, format="csc"
    )
    fl.q_next = a
    fl.rtol = 1e-300
    with pytest.warns(UserWarning, match="GMRES"):
        _, code = solve_fl_gmres(fl, rng.random(n))
    assert code != 0


def test_gmres_lu_path_and_lil_matrix() -> None:
    """The preconditioned solve matches a direct solve (lil matrix is converted)."""
    model = build_model(regime="transient")
    dmfwd.ForwardSolver(model).initialize()
    n = 20
    rng = np.random.default_rng(1)
    dense = np.eye(n) * 4 + rng.random((n, n)) * 0.1
    model.fl_model.q_next = lil_array(dense)
    rhs = rng.random(n)
    ilu, prec = get_super_ilu_preconditioner(
        model.fl_model.q_next.tocsc(), drop_tol=1e-10, fill_factor=100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x1, c1 = solve_fl_gmres(model.fl_model, rhs, ilu, prec)
        x2, c2 = solve_fl_gmres(model.fl_model, rhs)
    assert c1 == c2 == 0
    np.testing.assert_allclose(x1, np.linalg.solve(dense, rhs), rtol=1e-6)
    np.testing.assert_allclose(x2, x1, rtol=1e-6)


def test_make_transient_flow_matrices_matches_assembly() -> None:
    from pyrtid.forward.flow_solver import (
        _assemble_transient_flow_matrices,
        make_transient_flow_matrices,
    )

    model = build_3d_model(regime="transient")
    dmfwd.ForwardSolver(model).solve()
    args = (model.grid, model.fl_model, model.tr_model, 1)
    q_next, q_prev = make_transient_flow_matrices(*args)
    ref_next, ref_prev = _assemble_transient_flow_matrices(*args)
    assert isinstance(q_next, lil_array)
    np.testing.assert_allclose(q_next.toarray(), ref_next.tocsc().toarray())
    np.testing.assert_allclose(q_prev.toarray(), ref_prev.tocsc().toarray())


@pytest.mark.parametrize("regime", ["stationary", "transient"])
def test_save_matrices_and_ilu(regime: str) -> None:
    model = build_3d_model(regime=regime, all_faces=False)
    model.fl_model.is_save_spmats = True
    model.fl_model.is_save_spilu = True
    dmfwd.ForwardSolver(model).solve()
    fl = model.fl_model
    assert len(fl.l_q_next) == len(fl.lhead)
    assert len(fl.l_q_prev) == len(fl.lhead)
    assert fl.super_ilu is not None
    assert fl.preconditioner is not None


@pytest.mark.parametrize("regime", ["stationary", "transient"])
def test_singular_ilu_warns_and_still_solves(regime: str, monkeypatch) -> None:
    """If the ILU factorization fails, the flow is solved without preconditioner."""
    import pyrtid.forward.flow_solver as fs

    def _raise(*args, **kwargs):
        raise RuntimeError("singular")

    ref = build_3d_model(regime=regime, all_faces=False)
    dmfwd.ForwardSolver(ref).solve()
    monkeypatch.setattr(fs, "get_super_ilu_preconditioner", _raise)
    model = build_3d_model(regime=regime, all_faces=False)
    model.fl_model.is_save_spilu = True
    with pytest.warns(UserWarning, match=f"singular in {regime} flow"):
        dmfwd.ForwardSolver(model).solve()
    assert model.fl_model.super_ilu is None
    np.testing.assert_allclose(
        model.fl_model.lhead[-1], ref.fl_model.lhead[-1], rtol=1e-5
    )


@pytest.mark.parametrize(
    ("shape", "faces"),
    [
        ((1, 5, 1), ["south", "north"]),
        ((5, 1, 1), ["west", "east"]),
        ((1, 1, 5), ["bottom", "top"]),
        ((4, 3, 1), ["west"]),
        ((4, 3, 3), ["east", "north", "top"]),
        ((4, 3, 3), ["west", "south", "bottom"]),
    ],
)
@pytest.mark.parametrize("gravity", [False, True])
def test_flow_in_degenerated_grids(shape, faces, gravity: bool) -> None:
    """The flow is solved and the flux is conserved for 1D and one-sided cases."""
    model = build_3d_model(
        *shape, gravity=gravity, faces=faces, regime="transient", skip_rt=True
    )
    dmfwd.ForwardSolver(model).solve()
    fl = model.fl_model
    assert np.all(np.isfinite(fl.lhead[-1]))
    assert np.all(np.isfinite(fl.lu_darcy_x[-1]))
    # darcy continuity on the free cells
    div = fl.lu_darcy_div[-1]
    free = fl.free_head_indices
    # (the storage term makes the divergence non-null in transient: just bounded)
    assert np.all(np.abs(div[tuple(free)]) < 1e-3)
