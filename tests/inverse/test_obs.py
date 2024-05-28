from contextlib import nullcontext as does_not_raise
from typing import Optional

import numpy as np
import pyrtid.forward as dmfwd
import pytest
from pyrtid.inverse.loss_function import eval_model_loss_ls
from pyrtid.inverse.obs import (
    Observable,
    StateVariable,
    get_adjoint_sources_for_obs,
    get_array_from_state_variable,
    get_interp_simu_values_matching_obs_times,
    get_observables_uncertainties_as_1d_vector,
    get_observables_values_as_1d_vector,
    get_predictions_matching_observations,
    get_sorted_observable_times,
    get_sorted_observable_uncertainties,
    get_sorted_observable_values,
    get_times_idx_before_after_obs,
    get_values_matching_node_indices,
    get_weights,
)
from pyrtid.utils import finite_gradient
from pyrtid.utils.means import MeanType
from pyrtid.utils.types import NDArrayFloat, NDArrayInt


@pytest.mark.parametrize(
    "state_variable, node_indices, times, "
    "values, uncertainties, mean_type, sp, expected_exception",
    (
        (
            StateVariable.CONCENTRATION,
            [1],
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            None,
            0,
            does_not_raise(),
        ),
        (
            StateVariable.PERMEABILITY,
            [1, 2],
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            None,
            None,
            None,
            does_not_raise(),
        ),
        (
            StateVariable.CONCENTRATION,
            np.array([1, 2, 4, 6]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            5.0,
            MeanType.ARITHMETIC,
            0,
            does_not_raise(),
        ),
        (
            StateVariable.CONCENTRATION,
            np.array([1, 2, 4, 6]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 5.0]),
            np.array([5.0, 5.0]),
            MeanType.GEOMETRIC,
            0,
            pytest.raises(
                ValueError,
                match="``uncertainties`` parameter should be a float value or a numpy "
                "array with the same dimension as the ``values`` parameter.",
            ),
        ),
        (
            StateVariable.CONCENTRATION,
            np.array([1, 2, 4, 6]),
            np.array([1.0, 2.0, 5.0]),
            np.array([1.0, 2.0]),
            np.array([5.0, 5.0]),
            None,
            0,
            pytest.raises(
                ValueError,
                match="``times`` parameter should be a float value or a numpy "
                "array with the same dimension as the ``values`` parameter.",
            ),
        ),
    ),
)
def test_observable_init(
    state_variable,
    node_indices,
    times,
    values,
    uncertainties,
    mean_type,
    sp,
    expected_exception,
) -> None:
    with expected_exception:
        str(
            Observable(
                state_variable,
                node_indices,
                times,
                values,
                uncertainties,
                mean_type,
                sp=sp,
            )
        )


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
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])[:, :, np.newaxis], 5, axis=-1
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
    "state_variable, sp, expected_exception",
    (
        (StateVariable.CONCENTRATION, 0, does_not_raise()),
        (StateVariable.DENSITY, None, does_not_raise()),
        (StateVariable.DIFFUSION, None, does_not_raise()),
        (StateVariable.HEAD, None, does_not_raise()),
        (StateVariable.GRADE, 0, does_not_raise()),
        (StateVariable.PERMEABILITY, None, does_not_raise()),
        (StateVariable.POROSITY, None, does_not_raise()),
        (StateVariable.PRESSURE, None, does_not_raise()),
        (
            "a random variable",
            None,
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
    state_variable: StateVariable,
    sp: Optional[int],
    expected_exception,
    model: dmfwd.ForwardModel,
) -> None:
    with expected_exception:
        assert get_array_from_state_variable(model, state_variable, sp=sp).shape[
            :2
        ] == (
            20,
            20,
        )


@pytest.mark.parametrize(
    "times, values, uncertainties, max_obs_time, expected_times,"
    " expected_values, expected_uncertainties",
    (
        (
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            5.0,
            None,
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([5.0, 5.0]),
        ),
        (
            np.array([2.0, 5.23, 1.25, 6.9, 0.2]),
            np.array([2.2, 5.23, 1.25, 6.1, 0.3]),
            np.array([1.2, 1.2, 3.5, 3.6, 2.7]),
            None,
            np.array([0.2, 1.25, 2.0, 5.23, 6.9]),
            np.array([0.3, 1.25, 2.2, 5.23, 6.1]),
            np.array([2.7, 3.5, 1.2, 1.2, 3.6]),
        ),
        (
            np.array([2.0, 5.23, 1.25, 6.9, 0.2]),
            np.array([2.2, 5.23, 1.25, 6.1, 0.3]),
            np.array([1.2, 1.2, 3.5, 3.6, 2.7]),
            5.0,
            np.array([0.2, 1.25, 2.0]),
            np.array([0.3, 1.25, 2.2]),
            np.array([2.7, 3.5, 1.2]),
        ),
    ),
)
def test_get_obs_attribute_sorted_by_ascending_times(
    times,
    values,
    uncertainties,
    max_obs_time,
    expected_times,
    expected_values,
    expected_uncertainties,
) -> None:
    pass

    obs = Observable(
        StateVariable.CONCENTRATION, [1], times, values, uncertainties, sp=0
    )

    np.testing.assert_allclose(
        get_sorted_observable_times(obs, max_obs_time), expected_times
    )
    np.testing.assert_allclose(
        get_sorted_observable_values(obs, max_obs_time), expected_values
    )
    np.testing.assert_allclose(
        get_sorted_observable_uncertainties(obs, max_obs_time), expected_uncertainties
    )


@pytest.mark.parametrize(
    "is_use_list_of_obs, max_obs_time, expected_values, expected_uncertainties",
    (
        (
            False,
            None,
            np.array([0.3, 1.25, 2.2, 5.23, 6.1]),
            np.array([0.3, 1.25, 2.4, 5.23, 6.1]),
        ),
        (False, 5.0, np.array([0.3, 1.25, 2.2]), np.array([0.3, 1.25, 2.4])),
        (
            True,
            None,
            np.array(
                [
                    0.3,
                    1.25,
                    2.2,
                    5.23,
                    6.1,
                    0.3,
                    1.25,
                    2.2,
                    5.23,
                    6.1,
                    0.3,
                    1.25,
                    2.2,
                    5.23,
                    6.1,
                ]
            ),
            np.array(
                [
                    0.3,
                    1.25,
                    2.4,
                    5.23,
                    6.1,
                    0.3,
                    1.25,
                    2.4,
                    5.23,
                    6.1,
                    0.3,
                    1.25,
                    2.4,
                    5.23,
                    6.1,
                ]
            ),
        ),
        (
            True,
            5.0,
            np.array([0.3, 1.25, 2.2, 0.3, 1.25, 2.2, 0.3, 1.25, 2.2]),
            np.array([0.3, 1.25, 2.4, 0.3, 1.25, 2.4, 0.3, 1.25, 2.4]),
        ),
    ),
)
def test_get_observables_values_as_1d_vector(
    is_use_list_of_obs, max_obs_time, expected_values, expected_uncertainties
) -> None:
    obs1 = Observable(
        StateVariable.CONCENTRATION,
        [1],
        np.array([2.0, 5.23, 1.25, 6.9, 0.2]),
        np.array([2.2, 5.23, 1.25, 6.1, 0.3]),
        np.array([2.4, 5.23, 1.25, 6.1, 0.3]),  # first value different
        sp=0,
    )

    if is_use_list_of_obs:
        obs = (obs1, obs1, obs1)
    else:
        obs = obs1

    np.testing.assert_allclose(
        get_observables_values_as_1d_vector(obs, max_obs_time), expected_values
    )

    np.testing.assert_allclose(
        get_observables_uncertainties_as_1d_vector(obs, max_obs_time),
        expected_uncertainties,
    )


@pytest.mark.parametrize(
    "obs_times,simu_times,simu_values,expected_output",
    [
        (
            np.array([0.0, 2.0, 5.0, 5.6]),
            np.array([0.0, 2.0, 5.0, 6.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 3.6]),
        ),
        (
            np.array([1.5, 5.0]),
            np.array([0.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1.0, 2.0, 3.0, 6.0, 7.0]),
            np.array([1.75, 7.0]),
        ),
        (
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
        ),
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([4.0, 4.0, 4.0, 4.0]),
        ),
    ],
)
def test_get_interp_simu_values_matching_obs_times(
    obs_times: NDArrayFloat,
    simu_times: NDArrayFloat,
    simu_values: NDArrayFloat,
    expected_output: NDArrayFloat,
) -> None:
    np.testing.assert_allclose(
        get_interp_simu_values_matching_obs_times(obs_times, simu_times, simu_values),
        expected_output,
    )


@pytest.mark.parametrize(
    "max_obs_time, expected_output",
    [
        (
            None,
            np.array(
                [
                    1.0,
                    1.5,
                    2.0,
                    2.56,
                    1.0,
                    1.2,
                    1.3,
                    8.666667,
                    8.666667,
                    8.666667,
                    8.666667,
                ]
            ),
        ),
        (
            1.0,
            np.array([1.0, 1.5, 2.0, 1.0, 1.2, 1.3, 8.666667, 8.666667, 8.666667]),
        ),
    ],
)
def test_get_predictions_matching_observations(max_obs_time, expected_output) -> None:
    time_params = dmfwd.TimeParameters(duration=1.0, dt_init=1.0)
    geometry = dmfwd.Geometry(nx=20, ny=20, dx=4.5, dy=7.5)
    fl_params = dmfwd.FlowParameters(1e-5)
    tr_params = dmfwd.TransportParameters(1.0, 0.23)
    gch_params = dmfwd.GeochemicalParameters(1.0, 0.0)

    model = dmfwd.ForwardModel(
        geometry,
        time_params,
        fl_params,
        tr_params,
        gch_params,
    )

    # generate synthetic data
    model.tr_model.lmob.append(np.ones((2, 20, 20)) * 2.0)
    model.tr_model.lmob.append(np.ones((2, 20, 20)) * 3.0)
    model.time_params.save_dt()
    model.time_params.save_dt()

    model.tr_model.porosity[:, :] = np.arange(20 * 20).reshape((20, 20), order="F")

    obs1 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[2, 4],
        times=np.array([0.0, 0.5, 1.0, 1.56, 2.6]),
        values=np.array([1, 1, 1, 1, 1]),
        uncertainties=1.0,
        sp=0,
    )

    obs2 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[9, 10, 11],
        times=np.array([0.0, 0.2, 0.3, 4.3, 5.6]),
        values=np.array([1, 1, 1, 1, 1]),
        uncertainties=1.0,
        sp=0,
    )

    obs3 = Observable(
        StateVariable.POROSITY,
        node_indices=[5, 10, 11],
        times=np.array([0.0, 0.2, 0.3, 1.3, 5.6]),
        values=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
        uncertainties=3.5,
    )

    np.testing.assert_allclose(
        get_predictions_matching_observations(model, [obs1, obs2, obs3], max_obs_time),
        expected_output,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "max_obs_time, mean_type",
    [
        (None, MeanType.ARITHMETIC),
        (None, MeanType.GEOMETRIC),
        (None, MeanType.HARMONIC),
        (0.5, MeanType.ARITHMETIC),
        (1.0, MeanType.GEOMETRIC),
        (4.0, MeanType.HARMONIC),
        (10.0, MeanType.ARITHMETIC),
    ],
)
def test_get_adjoint_sources_for_obs(max_obs_time, mean_type) -> None:
    """
    Test if the adjoint sources are well built in the case:

    - simulated values averaged from several grid cells
    - simulated values interpolated to match the observation time
    """
    time_params = dmfwd.TimeParameters(duration=1.0, dt_init=1.0)
    geometry = dmfwd.Geometry(nx=5, ny=5, dx=4.5, dy=7.5)
    fl_params = dmfwd.FlowParameters(1e-5)
    tr_params = dmfwd.TransportParameters(1.0, 0.23)
    gch_params = dmfwd.GeochemicalParameters(1.0, 0.0)

    model = dmfwd.ForwardModel(
        geometry,
        time_params,
        fl_params,
        tr_params,
        gch_params,
    )

    # generate synthetic data
    model.tr_model.lmob.append(
        np.random.default_rng(2023).random((2, geometry.nx, geometry.ny)) + 2.0
    )
    model.tr_model.lmob.append(
        np.random.default_rng(2023).random((2, geometry.nx, geometry.ny)) + 3.0
    )
    model.time_params.save_dt()
    model.time_params.save_dt()

    model.fl_model.permeability = np.random.default_rng(2023).random(
        (geometry.nx, geometry.ny)
    )

    obs1 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[2, 4, 6],
        times=np.array([0.0, 0.5, 1.0, 1.56, 2.6]),
        values=np.array([1, 1, 1, 1, 1]),
        uncertainties=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
        mean_type=mean_type,
        sp=0,
    )

    obs2 = Observable(
        StateVariable.PERMEABILITY,
        node_indices=[5, 10, 11],
        times=np.array([0.0, 0.2, 0.3, 1.3, 5.6]),
        values=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
        uncertainties=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
        mean_type=mean_type,
    )

    if max_obs_time is not None:
        _max_obs_time = min(model.time_params.time_elapsed, max_obs_time)
    else:
        _max_obs_time = model.time_params.time_elapsed

    n_obs = get_observables_values_as_1d_vector([obs1], _max_obs_time).size

    def wrapper_conc(arr: NDArrayFloat) -> float:
        for i in range(arr.shape[-1]):
            model.tr_model.lmob[i][0] = arr[:, :, i]
        return eval_model_loss_ls(model, [obs1], max_obs_time)

    np.testing.assert_allclose(
        get_adjoint_sources_for_obs(model, obs1, n_obs, max_obs_time),
        finite_gradient(model.tr_model.mob[0], wrapper_conc),
    )

    n_obs = get_observables_values_as_1d_vector([obs2], _max_obs_time).size

    def wrapper_perm(arr: NDArrayFloat) -> float:
        model.fl_model.permeability = arr
        return eval_model_loss_ls(model, [obs2], max_obs_time)

    np.testing.assert_allclose(
        get_adjoint_sources_for_obs(model, obs2, n_obs, max_obs_time),
        finite_gradient(model.fl_model.permeability, wrapper_perm),
    )
