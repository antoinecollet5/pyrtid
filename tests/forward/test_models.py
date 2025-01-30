"""Some basic tests for the forward model."""

import re
from contextlib import nullcontext as does_not_raise
from typing import Tuple

import numpy as np
import pytest
from pyrtid.forward.models import (
    ConstantConcentration,
    ConstantHead,
    FlowParameters,
    ForwardModel,
    GeochemicalParameters,
    Geometry,
    SourceTerm,
    TimeParameters,
    TransportParameters,
    ZeroConcGradient,
    get_owner_neigh_indices,
)
from pyrtid.utils.types import NDArrayFloat

time_params = TimeParameters(duration=240000, dt_init=600.0)
geometry = Geometry(nx=20, ny=20, dx=4.5, dy=7.5)
fl_params = FlowParameters(1e-5)
tr_params = TransportParameters(diffusion=1e-10, porosity=0.23, dispersivity=0.1)
gch_params = GeochemicalParameters(0.0, 0.0)


@pytest.mark.parametrize(
    "args, kwargs, expected_dt, expected_dt_min, expected_dt_max",
    [
        ((1000, 150), {}, 150, 150, 150),
        ((1000, 0, 30), {}, 30, 30, 30),
        ((1000, 150), {"dt_min": 30}, 150, 30, 150),
        ((1000, 150), {"dt_max": 300}, 150, 150, 300),
        ((1000, 150), {"dt_min": 30, "dt_max": 300}, 150, 30, 300),
        ((1000, 150, 30, 300), {}, 150, 30, 300),
    ],
)
def test_time_params(
    args, kwargs, expected_dt, expected_dt_min, expected_dt_max
) -> None:
    time_params = TimeParameters(*args, **kwargs)
    assert time_params.dt == expected_dt
    assert time_params.dt_min == expected_dt_min
    assert time_params.dt_max == expected_dt_max
    assert time_params.dt_init == expected_dt

    # Update timestep
    for i in range(100):
        # Save the previous timestep
        time_params.save_dt()
        time_params.update_dt(1, 1e30, 20)

    assert time_params.dt == expected_dt_max
    assert len(time_params.ldt) == 100
    assert time_params.time_elapsed > 0

    time_params.reset_to_init()
    assert time_params.dt == expected_dt
    assert time_params.nts == 0
    assert time_params.nt == 1
    assert time_params.nfpi == 0
    assert len(time_params.lnfpi) == 0

    assert time_params.time_elapsed == 0

    for i in range(20):
        time_params.save_dt()
        time_params.save_nfpi()
        time_params.update_dt(30, 1e30, 20)

    assert time_params.dt == expected_dt_min
    assert len(time_params.ldt) == 20
    assert time_params.nts == 20
    assert time_params.nt == 21
    assert time_params.nfpi == 0
    assert time_params.time_elapsed > 0

    time_params.reset_to_init()

    assert time_params.dt == expected_dt
    assert time_params.nts == 0
    assert time_params.nt == 1

    assert time_params.time_elapsed == 0


def test_wrong_time_params() -> None:
    with pytest.raises(
        ValueError, match=re.escape("dt_min (40.0) is above dt_max (30.0)!")
    ):
        TimeParameters(2, 35.0, 40.0, 30.0)


@pytest.mark.parametrize(
    "nx,ny,dx,dy,expected_exception",
    [
        (10.0, 10.0, 10.0, 10.0, does_not_raise()),
        (0.0, 10.0, 0.0, 10.0, pytest.raises(ValueError, match="nx should be > 1!")),
        (10.0, 0.0, 10.0, 10.0, pytest.raises(ValueError, match="ny should be > 1!")),
        (10.0, 10.0, 0.0, 10.0, does_not_raise()),
        (10.0, 10.0, 10.0, 0.0, does_not_raise()),
        # (
        #     1.0,
        #     10.0,
        #     10.0,
        #     7.5,
        #     pytest.raises(
        #         ValueError,
        #         match="For a 1D case, set nx different from 1 and ny equal to 1!",
        #     ),
        # ),
        (10.0, 1.0, 10.0, 7.5, does_not_raise()),
        (
            2.0,
            2.0,
            10.0,
            7.5,
            pytest.raises(
                ValueError, match=r"At least one of \(nx, ny\) should be of dimension 3"
            ),
        ),
    ],
)
def test_geometry(nx, ny, dx, dy, expected_exception) -> None:
    with expected_exception:
        geom = Geometry(nx, ny, dx, dy)
        assert geom.grid_cell_volume == geom.grid_cell_volume == dx * dy


def get_source_term() -> SourceTerm:
    """Get a source term."""
    return SourceTerm(
        "some_name",
        np.array([1], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )


def test_minimal_model_init() -> None:
    ForwardModel(geometry, time_params, fl_params, tr_params, gch_params)


@pytest.fixture
def model() -> ForwardModel:
    source_terms = get_source_term()
    boundary_conditions = (
        ConstantHead(slice(None)),
        ConstantConcentration(slice(None)),
        ZeroConcGradient(slice(None)),
    )
    return ForwardModel(
        geometry,
        time_params,
        fl_params,
        tr_params,
        gch_params,
        source_terms,
        boundary_conditions,
    )


def test_add_source_term(model) -> None:
    assert len(model.source_terms) == 1
    source_term = SourceTerm(
        "some_name2",
        np.array([1], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )
    model.add_src_term(source_term)
    assert len(model.source_terms) == 2


def test_wrong_source_term(model) -> None:
    with pytest.raises(
        ValueError,
        match="Times, flowrates and concentrations must have the same dimension !",
    ):
        SourceTerm(
            "some_name",
            node_ids=np.array([1], dtype=np.int64),
            times=np.array([1.0, 1.0], dtype=np.float64),
            flowrates=np.array([1.0], dtype=np.float64),
            concentrations=np.array([1.0], dtype=np.float64),
        )


@pytest.mark.parametrize(
    "condition,expected_exception",
    [
        (ConstantHead(span=slice(None)), does_not_raise()),
        (ConstantConcentration(span=slice(None)), does_not_raise()),
        (ZeroConcGradient(span=slice(None)), does_not_raise()),
        ("some random object", pytest.raises(ValueError)),
    ],
)
def test_add_model_boundary_conditions(model, condition, expected_exception) -> None:
    """Test boundary conditions for the flow model."""
    with expected_exception:
        model.add_boundary_conditions(condition)


@pytest.mark.parametrize(
    "condition,expected_exception",
    [
        (ConstantHead(span=slice(None)), does_not_raise()),
        (ConstantConcentration(span=slice(None)), pytest.raises(ValueError)),
        (ZeroConcGradient(span=slice(None)), pytest.raises(ValueError)),
    ],
)
def test_add_flow_boundary_conditions(model, condition, expected_exception) -> None:
    """Test boundary conditions for the flow model."""
    with expected_exception:
        model.fl_model.add_boundary_conditions(condition)


@pytest.mark.parametrize(
    "condition,expected_exception",
    [
        (ConstantHead(span=slice(None)), pytest.raises(ValueError)),
        (ConstantConcentration(span=slice(None)), does_not_raise()),
        (ZeroConcGradient(span=slice(None)), does_not_raise()),
    ],
)
def test_add_transport_boundary_conditions(
    model, condition, expected_exception
) -> None:
    """Test boundary conditions for the transport model."""
    with expected_exception:
        model.tr_model.add_boundary_conditions(condition)


@pytest.mark.parametrize(
    "span_owner,span_neigh,owner_indices_to_keep, "
    "neigh_indices_to_keep,expected_idc_owner,expected_idc_neigh",
    (
        (
            (slice(None), slice(1, 4)),
            (slice(None), slice(0, 4 - 1)),
            None,
            None,
            np.array([4, 8, 12, 5, 9, 13, 6, 10, 14, 7, 11, 15]),
            np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]),
        ),
        (
            (slice(None), slice(1, 4)),
            (slice(None), slice(0, 4 - 1)),
            np.array([]),
            None,
            np.array([]),
            np.array([]),
        ),
        (
            (slice(None), slice(1, 4)),
            (slice(None), slice(0, 4 - 1)),
            None,
            np.array([]),
            np.array([]),
            np.array([]),
        ),
        (
            (slice(None), slice(1, 4)),
            (slice(None), slice(0, 4 - 1)),
            np.array([1, 2, 3, 4, 5, 6, 7]),
            None,
            np.array([4, 5, 6, 7]),
            np.array([0, 1, 2, 3]),
        ),
    ),
)
def test_get_owner_neigh_indices(
    span_owner,
    span_neigh,
    owner_indices_to_keep,
    neigh_indices_to_keep,
    expected_idc_owner,
    expected_idc_neigh,
) -> None:
    geometry = Geometry(4, 4, 1.0, 1.0, 1.0)

    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        span_owner,
        span_neigh,
        owner_indices_to_keep=owner_indices_to_keep,
        neigh_indices_to_keep=neigh_indices_to_keep,
    )

    np.testing.assert_equal(
        idc_owner,
        expected_idc_owner,
    )
    np.testing.assert_equal(
        idc_neigh,
        expected_idc_neigh,
    )


def test_source_term_et_node_indices() -> None:
    st = SourceTerm(
        "joe",
        np.array([1, 5, 12, 24]),
        np.array([0.5, 22.0, 56.0, 99.1]),
        np.array([0.0, 2.0, 2.0, 4.0]),
        np.array([1.0, 2.0, 0.0, 4.0]),
    )
    np.testing.assert_equal(
        st.get_node_indices(Geometry(5, 5, 2.0, 2.0, 5.0)),
        np.array([[1, 0, 2, 4], [0, 1, 2, 4], [0, 0, 0, 0]]),
    )

    assert st.n_nodes == 4


@pytest.mark.parametrize(
    "time, expected_sources",
    [
        (0.0, (0.0, 0.0)),
        (1.0, (0.0, 1.0)),
        (22.0, (0.0, 1.0)),  # the new flowrates applies once t is above the time
        (25.0, (2.0, 2.0)),
        (60.0, (2.0, 0.0)),
        (99.2, (4.0, 4.0)),
        (500.0, (4.0, 4.0)),
    ],
)
def test_source_term_get_values(
    time: float, expected_sources: Tuple[float, float]
) -> None:
    st = SourceTerm(
        "joe",
        np.array([1, 5, 12, 24]),
        np.array([0.5, 22.0, 56.0, 99.1]),
        np.array([0.0, 2.0, 2.0, 4.0]),
        np.array([1.0, 2.0, 0.0, 4.0]),
    )

    assert st.get_values(time) == expected_sources


def test_model_effective_diffusion(model: ForwardModel) -> None:
    np.testing.assert_array_equal(
        model.tr_model.effective_diffusion, np.ones((20, 20)) * 2.3e-11
    )


def test_model_set_values(model: ForwardModel) -> None:
    arr = np.random.random(size=(20, 20, 1))

    model.tr_model.set_initial_conc(arr[:, :, 0])
    np.testing.assert_array_equal(model.tr_model.mob[0], arr)
    np.testing.assert_array_equal(model.tr_model.mob[1], np.zeros_like(arr))

    model.tr_model.set_initial_conc(arr[:, :, 0], sp=1)
    np.testing.assert_array_equal(model.tr_model.mob[0], arr)
    np.testing.assert_array_equal(model.tr_model.mob[1], arr)

    model.tr_model.set_initial_conc(4.0)
    np.testing.assert_array_equal(model.tr_model.mob[0], np.ones((20, 20, 1)) * 4.0)

    model.tr_model.set_initial_grade(arr[:, :, 0] * 2.0, sp=0)
    np.testing.assert_array_equal(model.tr_model.immob[0], arr * 2.0)
    np.testing.assert_array_almost_equal(
        model.tr_model.immob[1], np.zeros_like(arr), decimal=10
    )

    model.tr_model.set_initial_grade(2.0, sp=1)
    np.testing.assert_array_equal(model.tr_model.immob[0], arr * 2.0)
    np.testing.assert_array_equal(model.tr_model.immob[1], np.ones((20, 20, 1)) * 2.0)

    model.fl_model.set_initial_head(arr[:, :, 0] * 3.0)
    np.testing.assert_array_equal(model.fl_model.head, arr * 3.0)

    model.fl_model.set_initial_head(100.0)
    np.testing.assert_array_equal(model.fl_model.head, np.ones((20, 20, 1)) * 100.0)


def test_model_reinit(model: ForwardModel) -> None:
    model.tr_model.lmob.append(model.tr_model.lmob[-1])
    assert len(model.tr_model.lmob) == 2

    model.fl_model.lhead.append(model.fl_model.lhead[-1])
    assert len(model.fl_model.lhead) == 2

    model.reinit()

    assert len(model.tr_model.lmob) == 1
    assert len(model.tr_model.limmob) == 1

    np.testing.assert_array_equal(
        model.tr_model.lmob[0], model.tr_model.mob[:, :, :, 0]
    )
    np.testing.assert_array_equal(
        model.tr_model.limmob[0], model.tr_model.immob[:, :, :, 0]
    )

    assert len(model.fl_model.lhead) == 1
    assert len(model.fl_model.lu_darcy_x) == 0
    assert len(model.fl_model.lu_darcy_y) == 0
    assert len(model.fl_model.lu_darcy_div) == 0

    assert np.all(model.fl_model.head[:, :, 1:] == 0)
    # assert np.all(model.fl_model.u_darcy_x[:, :, :] == 0)
    # assert np.all(model.fl_model.u_darcy_y[:, :, :] == 0)

    np.testing.assert_array_equal(model.fl_model.lhead[0], model.fl_model.head[:, :, 0])
    # np.testing.assert_array_equal(
    #     model.fl_model.lu_darcy_x[0], model.fl_model.u_darcy_x[:, :, 0]
    # )
    # np.testing.assert_array_equal(
    #     model.fl_model.lu_darcy_y[0], model.fl_model.u_darcy_y[:, :, 0]
    # )
    # np.testing.assert_array_equal(
    #     model.fl_model.lu_darcy_div[0], model.fl_model.u_darcy_div[:, :, 0]
    # )


@pytest.mark.parametrize(
    "time, expected_src_flw, expected_src_conc",
    [
        (
            0.0,
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.75, 1.0]]),
            np.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
        ),
        (
            0.5,
            np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.0], [0.0, -0.75, 1.0]]),
            np.array(
                [
                    [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
        ),
        (
            50.0,
            np.array([[0.0, 0.125, 0.0], [0.125, 0.0, 0.0], [0.0, 0.125, 0.25]]),
            np.array(
                [
                    [[0.0, 0.0, 0.25], [0.25, 0.0, 0.0], [0.0, 0.25, 0.5]],
                    [[0.0, 0.0, 0.25], [0.25, 0.0, 0.0], [0.0, 0.25, 0.5]],
                ]
            ),
        ),
        (
            500.0,
            np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.5]]),
            np.array(
                [
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 2.0]],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 2.0]],
                ]
            ),
        ),
    ],
)
def test_get_sources(
    time: float, expected_src_flw: NDArrayFloat, expected_src_conc: NDArrayFloat
) -> None:
    time_params = TimeParameters(duration=240000, dt_init=600.0)
    geometry = Geometry(nx=3, ny=3, dx=2.0, dy=2.0, dz=2.0)
    fl_params = FlowParameters(1e-5)
    tr_params = TransportParameters(diffusion=1e-10, porosity=0.23)
    gch_params = GeochemicalParameters(0.0, 0.0)

    model = ForwardModel(geometry, time_params, fl_params, tr_params, gch_params)

    model.add_boundary_conditions(ConstantConcentration((slice(0, 1), slice(1, 2))))

    model.add_boundary_conditions(ConstantHead((slice(0, 1), slice(2, 3))))

    model.add_src_term(
        SourceTerm(
            "inj",
            np.array([1, 3]),
            np.array([0.5, 22.0, 56.0, 99.0]),
            np.array([4.0, 2.0, 2.0, 4.0]),
            np.array([[1.0, 2.0, 0.0, 4.0], [1.0, 2.0, 0.0, 4.0]]).T,
        )
    )
    model.add_src_term(
        SourceTerm(
            "inj2",
            np.array([8]),
            np.array([0.0, 22.0, 123.0, 99.0]),
            np.array([8.0, 2.0, 2.0, 4.0]),
            np.array([[1.0, 2.0, 0.0, 4.0], [1.0, 2.0, 0.0, 4.0]]).T,
        )
    )

    model.add_src_term(
        SourceTerm(
            "prod",
            np.array([6, 5]),
            np.array([0.0, 20.0, 77.0, 120.0]),
            np.array([-12.0, 2.0, -2.0, 4.0]),
            np.array([[1.0, 2.0, 0.0, 4.0], [1.0, 2.0, 0.0, 4.0]]).T,
        )
    )

    _unitflw_src, _conc_src = model.get_sources(time, geometry)
    np.testing.assert_equal(_unitflw_src, expected_src_flw)
    np.testing.assert_equal(_conc_src, expected_src_conc)


@pytest.mark.parametrize(
    "boundary_conditions, expected_cst_head_nn, expected_free_head_nn",
    [
        ((), [], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
        (ConstantHead(slice(None)), [0, 1, 2, 3, 4, 5, 6, 7, 8], np.array([])),
        (
            ConstantHead((slice(0, 3), slice(0, 3))),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            [],
        ),
        (
            ConstantHead((slice(0, 3), slice(0, 1))),
            np.array([0, 1, 2]),
            [3, 4, 5, 6, 7, 8],
        ),
        (
            ConstantHead((slice(0, 2), slice(0, 1))),
            np.array([0, 1]),
            [2, 3, 4, 5, 6, 7, 8],
        ),
    ],
)
def test_cst_head(
    boundary_conditions, expected_cst_head_nn, expected_free_head_nn
) -> None:
    time_params = TimeParameters(duration=240000, dt_init=600.0)
    geometry = Geometry(nx=3, ny=3, dx=2.0, dy=2.0, dz=2.0)
    fl_params = FlowParameters(1e-5)
    tr_params = TransportParameters(diffusion=1e-10, porosity=0.23)
    gch_params = GeochemicalParameters(0.0, 0.0)

    model = ForwardModel(
        geometry,
        time_params,
        fl_params,
        tr_params,
        gch_params,
        boundary_conditions=boundary_conditions,
    )

    np.testing.assert_equal(model.fl_model.cst_head_nn, expected_cst_head_nn)
    np.testing.assert_equal(model.fl_model.free_head_nn, expected_free_head_nn)
