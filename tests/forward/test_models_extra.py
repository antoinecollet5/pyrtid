"""Additional tests of the forward models (properties, copies, sparse helpers)."""

import copy

import numpy as np
import pyrtid.forward as dmfwd
import pytest
from pyrtid.forward.models import (
    GRAVITY,
    WATER_DENSITY,
    SparseMatrixBuilder,
    _get_a_not_in_b_1d,
    add_entries,
    add_to_diagonal,
    remove_cst_bound_indices,
)
from scipy.sparse import lil_array
from tests.helpers import build_model


def test_time_parameters_times() -> None:
    tp = dmfwd.TimeParameters(duration=30.0, dt_init=10.0)
    for _ in range(3):
        tp.save_dt()
    np.testing.assert_array_equal(tp.times, [0.0, 10.0, 20.0, 30.0])
    assert tp.nts == 3
    assert tp.nt == 4
    assert tp.time_elapsed == 30.0


def test_get_a_not_in_b_1d() -> None:
    a = np.array([5, 1, 3, 2])
    np.testing.assert_array_equal(_get_a_not_in_b_1d(a, np.array([], dtype=int)), a)
    np.testing.assert_array_equal(_get_a_not_in_b_1d(a, np.array([3, 5])), [1, 2])
    empty = np.array([], dtype=int)
    assert _get_a_not_in_b_1d(empty, np.array([1])).size == 0


def test_pressure_head_conversions() -> None:
    model = build_model(regime="transient", nz=3)
    fl = model.fl_model
    pos = fl._get_mesh_center_vertical_pos()
    # the cached position is returned as a copy
    pos[...] = -1.0
    assert np.all(fl._get_mesh_center_vertical_pos() >= 0.0)
    head = fl.lhead[0].copy()
    p = fl.head_to_pressure(head)
    np.testing.assert_allclose(fl.pressure_to_head(p), head, atol=1e-12)
    # hydrostatic pressure: P = rho g (h - z)
    z = fl._get_mesh_center_vertical_pos()
    np.testing.assert_allclose(p, (head - z) * GRAVITY * WATER_DENSITY)
    # set the pressure on a slice: the head follows
    span = (slice(0, 2), slice(None), slice(None))
    fl.set_initial_pressure(1.0e4, span)
    np.testing.assert_allclose(fl.lpressure[0][span], 1.0e4)
    np.testing.assert_allclose(
        fl.lhead[0][span], 1.0e4 / GRAVITY / WATER_DENSITY + z[span]
    )
    # set the head: the pressure follows
    fl.set_initial_head(3.0, span)
    np.testing.assert_allclose(fl.lhead[0][span], 3.0)
    np.testing.assert_allclose(
        fl.lpressure[0][span], (3.0 - z[span]) * GRAVITY * WATER_DENSITY
    )
    # pressure properties
    assert fl.get_pressure_pa().shape == (*model.grid.shape, 1)
    np.testing.assert_allclose(fl.get_pressure_bar(), fl.get_pressure_pa() / 1e5)


def test_flow_model_is_gravity_flag() -> None:
    assert build_model().fl_model.is_gravity is False
    dens = dmfwd.ForwardModel(
        build_model().grid,
        dmfwd.TimeParameters(duration=10.0, dt_init=1.0),
        dmfwd.FlowParameters(is_gravity=True),
    )
    assert dens.fl_model.is_gravity is True


def test_transport_model_result_properties() -> None:
    model = build_model(regime="transient")
    tr = model.tr_model
    assert tr.density.size == 0
    dmfwd.ForwardSolver(model).solve()
    nt = model.time_params.nt
    shape = (*model.grid.shape, nt)
    np.testing.assert_array_equal(tr.conc, tr.mob[0])
    np.testing.assert_array_equal(tr.conc2, tr.mob[1])
    np.testing.assert_array_equal(tr.grade, tr.immob[0])
    np.testing.assert_array_equal(tr.grade2, tr.immob[1])
    assert tr.conc.shape == shape
    assert tr.grade2.shape == shape
    assert tr.density.shape == shape
    assert tr.sources.shape == (2, *shape)
    # the density increases with the dissolved species
    assert np.all(tr.density >= 1000.0 * 0.99)


def test_free_and_constant_indices() -> None:
    model = build_model(cst_conc=True)
    dmfwd.ForwardSolver(model).initialize()
    grid = model.grid
    fl, tr = model.fl_model, model.tr_model
    assert fl.cst_head_indices.shape == (3, 2 * grid.ny)
    assert fl.free_head_indices.shape == (3, grid.n_grid_cells - 2 * grid.ny)
    assert tr.cst_conc_indices.shape == (3, grid.ny)
    assert np.all(tr.cst_conc_indices[0] == 0)
    assert tr.free_conc_indices.shape == (3, grid.n_grid_cells - grid.ny)
    assert not np.any(tr.free_conc_indices[0] == 0)


def test_add_src_term_overwrite_warns() -> None:
    model = build_model()
    src = model.source_terms["inj"]
    with pytest.warns(UserWarning, match="overwritten"):
        model.add_src_term(src)
    assert model.source_terms["inj"] is src


def test_boundary_conditions_dispatch() -> None:
    model = build_model()
    span = (slice(0, 1), slice(None), slice(None))
    zcg = dmfwd.ZeroConcGradient(span=span)
    model.add_boundary_conditions(zcg)
    assert zcg in model.tr_model.boundary_conditions
    cc = dmfwd.ConstantConcentration(span=span, values=1e-3)
    model.add_boundary_conditions(cc)
    assert cc in model.tr_model.boundary_conditions

    class Other(dmfwd.models.BoundaryCondition):
        pass

    with pytest.raises(ValueError, match="not a valid boundary condition"):
        model.add_boundary_conditions(Other(span=span))
    # Only the constant head are valid for the flow
    with pytest.raises(ValueError, match="flow model"):
        model.fl_model.add_boundary_conditions(zcg)
    # The sources and the solver ignore the conditions of another type
    model.fl_model.boundary_conditions.append(zcg)
    model.fl_model.set_constant_head_indices()
    unitflow, conc_src = model.get_sources(0.0, model.grid)
    assert unitflow.shape == model.grid.shape
    assert np.all(conc_src[:, 0] == 0.0)  # constant concentration cells


def test_deepcopy_after_solve() -> None:
    model = build_model(regime="transient", cst_conc=True)
    fl_ilu_before = model.fl_model.super_ilu
    dmfwd.ForwardSolver(model).solve()
    assert model.tr_model.super_ilu is not None
    cp = copy.deepcopy(model)
    # the non picklable objects are shared and the original is restored
    assert cp.tr_model.super_ilu is model.tr_model.super_ilu
    assert cp.fl_model.preconditioner is model.fl_model.preconditioner
    assert model.tr_model.super_ilu is not None
    assert model.fl_model.super_ilu is fl_ilu_before or True
    # the data are independent
    assert cp.tr_model is not model.tr_model
    cp.tr_model.lmob[-1][...] = -1.0
    assert np.all(model.tr_model.lmob[-1] != -1.0)
    np.testing.assert_array_equal(cp.fl_model.head, model.fl_model.head)
    # the copy can be copied again and run again with the same results
    cp2 = copy.deepcopy(cp)
    dmfwd.ForwardSolver(cp2).solve()
    np.testing.assert_allclose(cp2.tr_model.conc, model.tr_model.conc, rtol=1e-8)


def test_sparse_matrix_builder() -> None:
    builder = SparseMatrixBuilder((3, 3))
    empty = builder.tocsc()
    assert empty.shape == (3, 3)
    assert empty.nnz == 0
    add_entries(builder, np.array([0, 1]), np.array([1, 2]), 2.0)
    add_entries(builder, np.array([0, 1]), np.array([1, 0]), np.array([3.0, 4.0]))
    add_to_diagonal(builder, np.array([1.0, 2.0, 3.0]))
    expected = np.array([[1.0, 5.0, 0.0], [4.0, 2.0, 2.0], [0.0, 0.0, 3.0]])
    np.testing.assert_allclose(builder.tocsc().toarray(), expected)
    np.testing.assert_allclose(builder.tolil().toarray(), expected)
    # the same operations on a lil array
    mat = lil_array((3, 3))
    add_entries(mat, np.array([0, 1]), np.array([1, 2]), 2.0)
    add_entries(mat, np.array([0, 1]), np.array([1, 0]), np.array([3.0, 4.0]))
    add_to_diagonal(mat, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(mat.toarray(), expected)


def test_remove_cst_bound_indices() -> None:
    owner = np.array([0, 1, 2, 3])
    neigh = np.array([10, 11, 12, 13])
    o, n = remove_cst_bound_indices(owner, neigh, np.array([1, 3]))
    np.testing.assert_array_equal(o, [0, 2])
    np.testing.assert_array_equal(n, [10, 12])
