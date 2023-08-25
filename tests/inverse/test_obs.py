from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pyrtid.forward as dmfwd
from pyrtid.inverse.obs import (
    StateVariable,
    get_array_from_state_variable,
    get_times_idx_before_after_obs,
    get_values_matching_node_indices,
    get_weights,
)
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


def test_get_times_idx_before_after_obs() -> None:
    obs_times = np.array([0.0, 2.0, 4.0, 5.6])
    ldt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    calc_times = np.cumsum([0] + ldt)

    idx_before, idx_after = get_times_idx_before_after_obs(obs_times, calc_times)

    np.testing.assert_equal(idx_before, np.array([-1, 1, 3, 5]))
    np.testing.assert_equal(idx_after, np.array([0, 2, 4, 6]))


def test_get_weights() -> None:
    obs_times = np.array([0.0, 2.0, 4.0, 5.6])
    ldt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    calc_times = np.cumsum([0] + ldt)

    idx_before, idx_after = get_times_idx_before_after_obs(obs_times, calc_times)

    weights_before, weights_after = get_weights(
        obs_times, calc_times, idx_before, idx_after
    )

    np.testing.assert_almost_equal(
        weights_before, np.array([0.0, 0.0, 0.0, 0.4]), decimal=15
    )
    np.testing.assert_almost_equal(
        weights_after, np.array([1.0, 1.0, 1.0, 0.6]), decimal=15
    )


@pytest.mark.parametrize(
    "node_indices, expected_output",
    (
        (
            np.array([0, 1, 2, 3, 4, 5]),
            np.repeat(
                np.array([[1, 4], [2, 5], [3, 6]])[:, :, np.newaxis], 5, axis=-1
            ).reshape(-1, 5, order="F"),
        ),
        (np.array([1, 3]), np.repeat(np.array([[2], [4]]), 5, axis=-1)),
        (np.array([1]), np.ones(([5])).reshape(1, 5) * 2.0),
    ),
)
def test_get_values_matching_node_indices(
    node_indices: NDArrayInt, expected_output: NDArrayFloat
) -> None:
    # 3D arrays -> with time axis
    input_values = np.repeat(
        np.array([[1, 4], [2, 5], [3, 6]])[:, :, np.newaxis], 5, axis=-1
    )
    np.testing.assert_allclose(
        get_values_matching_node_indices(node_indices, input_values),
        expected_output,
    )

    # 2D arrays -> no time axis
    input_values = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_allclose(
        get_values_matching_node_indices(node_indices, input_values),
        expected_output[:, 0],
    )


@pytest.fixture
def model() -> dmfwd.ForwardModel:
    time_params = dmfwd.TimeParameters(duration=240000, dt_init=600.0)
    geometry = dmfwd.Geometry(nx=20, ny=20, dx=4.5, dy=7.5)
    fl_params = dmfwd.FlowParameters(1e-5)
    tr_params = dmfwd.TransportParameters(1e-10, 0.23)
    gch_params = dmfwd.GeochemicalParameters(0.0, 0.0)

    return dmfwd.ForwardModel(
        geometry,
        time_params,
        fl_params,
        tr_params,
        gch_params,
    )


@pytest.mark.parametrize(
    "state_variable, expected_exception",
    (
        (StateVariable.CONCENTRATION, does_not_raise()),
        (StateVariable.DENSITY, does_not_raise()),
        (StateVariable.DIFFUSION, does_not_raise()),
        (StateVariable.HEAD, does_not_raise()),
        (StateVariable.MINERAL_GRADE, does_not_raise()),
        (StateVariable.PERMEABILITY, does_not_raise()),
        (StateVariable.POROSITY, does_not_raise()),
        (StateVariable.PRESSURE, does_not_raise()),
        (
            "a random variable",
            pytest.raises(
                ValueError,
                match=(
                    '"a random variable" is not a valid state '
                    "variable or parameter type!"
                ),
            ),
        ),
    ),
)
def test_get_array_from_state_variable(
    state_variable: StateVariable, expected_exception, model: dmfwd.ForwardModel
) -> None:
    with expected_exception:
        assert get_array_from_state_variable(model, state_variable).shape[:2] == (
            20,
            20,
        )
