"""Some basic tests for the forward model."""

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
def test_geometry(nx, ny, dx, dy, expected_exception) -> Geometry:
    with expected_exception:
        return Geometry(nx, ny, dx, dy)


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
