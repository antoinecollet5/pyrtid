"""Some basic tests for the forward model."""

import re
from contextlib import contextmanager

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
    resize_array,
)

time_params = TimeParameters(nt=400, dt_init=600.0)
geometry = Geometry(nx=20, ny=20, dx=4.5, dy=7.5)
fl_params = FlowParameters(1e-5)
tr_params = TransportParameters(1e-10, 0.23)
gch_params = GeochemicalParameters(0.0, 0.0)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "args, kwargs, expected_dt, expected_dt_min, expected_dt_max",
    [
        ((100, 150), {}, 150, 150, 150),
        ((10, 0, 30), {}, 30, 30, 30),
        ((100, 150), {"dt_min": 30}, 150, 30, 150),
        ((100, 150), {"dt_max": 300}, 150, 150, 300),
        ((100, 150), {"dt_min": 30, "dt_max": 300}, 150, 30, 300),
        ((100, 150, 30, 300), {}, 150, 30, 300),
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
        time_params.ldt.append(time_params.dt)
        time_params.update_dt(1)

    assert time_params.dt == expected_dt_max
    assert len(time_params.ldt) == 100
    assert time_params.time_elapsed > 0

    time_params.reset_to_init()
    assert time_params.dt == expected_dt
    # assert len(time_params.ldt) == 0

    assert time_params.time_elapsed == 0

    for i in range(20):
        time_params.ldt.append(time_params.dt)
        time_params.update_dt(30)

    assert time_params.dt == expected_dt_min
    assert len(time_params.ldt) == 20
    assert time_params.time_elapsed > 0

    time_params.reset_to_init()

    assert time_params.dt == expected_dt
    assert len(time_params.ldt) == 0

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
        assert geom.mesh_area == dx * dy
        assert geom.mesh_volume == geom.mesh_area


def test_resize_array() -> None:
    test_arr = np.ones((5, 5, 7))
    for axis, shape in zip(range(3), [(1, 5, 7), (5, 1, 7), (5, 5, 1)]):
        np.testing.assert_array_equal(resize_array(test_arr, axis, 1), np.ones((shape)))

    with pytest.raises(
        IndexError,
        match=r"Axis 4 does not exists for the provided array of shape \(5, 5, 7\)!",
    ):
        resize_array(test_arr, 4, 1)


def get_source_term() -> SourceTerm:
    """Get a source term."""
    return SourceTerm(
        "some_name",
        np.array([1], dtype=np.int32),
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
        "some_name",
        np.array([1], dtype=np.int32),
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
            node_ids=np.array([1], dtype=np.int32),
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


def test_get_owner_neigh_indices() -> None:
    geometry = Geometry(4, 4, 1.0, 1.0, 1.0)

    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(None), slice(1, geometry.ny)),
        (slice(None), slice(0, geometry.ny - 1)),
        np.array([]),
    )

    np.testing.assert_equal(
        idc_owner, np.array([4, 8, 12, 5, 9, 13, 6, 10, 14, 7, 11, 15], dtype=np.int32)
    )
    np.testing.assert_equal(
        idc_neigh, np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=np.int32)
    )

    idc_owner, idc_neigh = get_owner_neigh_indices(
        geometry,
        (slice(None), slice(1, geometry.ny)),
        (slice(None), slice(0, geometry.ny - 1)),
        np.array([0, 1, 12, 13]),
    )

    np.testing.assert_equal(
        idc_owner, np.array([4, 8, 5, 9, 6, 10, 14, 7, 11, 15], dtype=np.int32)
    )
    np.testing.assert_equal(
        idc_neigh, np.array([0, 4, 1, 5, 2, 6, 10, 3, 7, 11], dtype=np.int32)
    )


def test_model_shape(model: ForwardModel) -> None:
    assert model.shape == (20, 20, 401)


def test_model_reinit(model: ForwardModel) -> None:
    model.tr_model.conc[:, :, :] = 1.0
    model.tr_model.conc_post_tr[:, :, :] = 1.0
    model.tr_model.grade[:, :, :] = 1.0

    model.fl_model.head[:, :, :] = 1.0
    model.fl_model.u_darcy_x[:, :, :] = 1.0
    model.fl_model.u_darcy_y[:, :, :] = 1.0

    model.reinit()

    assert np.all(model.tr_model.conc[:, :, 1:] == 0)
    assert np.all(model.tr_model.conc_post_tr[:, :, 1:] == 0)
    assert np.all(model.tr_model.grade[:, :, 1:] == 0)

    assert np.all(model.fl_model.head[:, :, 1:] == 0)
    assert np.all(model.fl_model.u_darcy_x[:, :, :] == 0)
    assert np.all(model.fl_model.u_darcy_y[:, :, :] == 0)


def test_tr_model_resize(model: ForwardModel) -> None:
    model.tr_model.conc[:, :, :] = 1.0
    model.tr_model.conc_post_tr[:, :, :] = 1.0
    model.tr_model.grade[:, :, :] = 1.0

    model.tr_model.reinit()

    assert np.all(model.tr_model.conc[:, :, 1:] == 0)
    assert np.all(model.tr_model.conc_post_tr[:, :, 1:] == 0)
    assert np.all(model.tr_model.grade[:, :, 1:] == 0)
