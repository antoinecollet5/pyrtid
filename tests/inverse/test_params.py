"""Tests for the adjustable parameter class."""
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import pytest

from pyrtid.inverse import AdjustableParameter
from pyrtid.inverse.regularization import (
    TikhonovRegularizatorAnisotropic,
    TVRegularizatorIsotropic,
)


@contextmanager
def does_not_raise():
    yield


def inverse_function(x: np.ndarray) -> np.ndarray:
    """Return 1/x."""
    return 1 / x


def identify_function(x: np.ndarray) -> np.ndarray:
    """Return x untransformed (f(x) = x)."""
    return x


@pytest.mark.parametrize(
    "kwargs,expected_exception",
    [
        ({"name": "any_param_name"}, pytest.warns(UserWarning)),
        (
            {"name": "any_param_name", "ubound": 10.0, "lbound": -10.0},
            does_not_raise(),  # bounds are OK
        ),
        (
            {"name": "any_param_name", "ubound": -10.0, "lbound": 10.0},
            pytest.raises(ValueError),  # ubound == lbound
        ),
        (
            {"name": "any_param_name", "ubound": 10.0, "lbound": 10.0},
            pytest.raises(ValueError),  # ubound == lbound
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 10.0,
                "lbound": -10.0,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": inverse_function,
                "backconditioner": np.exp,
            },
            pytest.raises(
                ValueError
            ),  # preconditioner is not define on the range [lbound-ubound]
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 10.0,
                "lbound": 1.0,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": inverse_function,
            },
            pytest.raises(
                ValueError
            ),  # preconditioner does not match back preconditioner
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 10.0,
                "lbound": 1.0,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": identify_function,
                "backconditioner": np.exp,
            },
            pytest.warns(
                UserWarning
            ),  # preconditioner_1st_derivative does not match the preconditioner
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 1e6,
                "lbound": 1e-6,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": inverse_function,
                "backconditioner": np.exp,
            },
            does_not_raise(),  # All OK
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 1e6,
                "lbound": 1e-6,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": inverse_function,
                "backconditioner": np.exp,
                "regularizators": [
                    "a_string_object",
                ],
                "span": slice(2, 20),
            },
            pytest.raises(
                ValueError, match="Expect a regulariator instance !"
            ),  # not a valid regularizator
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 1e6,
                "lbound": 1e-6,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": inverse_function,
                "backconditioner": np.exp,
                "regularizators": [
                    TikhonovRegularizatorAnisotropic(2, axis=0, is_preconditioned=True),
                ],
                "span": slice(2, 20),
            },
            does_not_raise(),  # All OK
        ),
        (
            {
                "name": "any_param_name",
                "ubound": 1e6,
                "lbound": 1e-6,
                "preconditioner": np.log,
                "preconditioner_1st_derivative": inverse_function,
                "backconditioner": np.exp,
                "regularizators": [
                    TVRegularizatorIsotropic(dx=2, dy=2),
                ],
                "span": slice(2, 20),
            },
            does_not_raise(),  # All OK
        ),
    ],
)
def test_init(kwargs, expected_exception):
    with expected_exception:
        return AdjustableParameter(**kwargs)


@pytest.fixture
def example_kwargs() -> Dict[str, Any]:
    return {
        "name": "any_param_name",
        "values": np.ones([5, 5]),
        "ubound": 1e6,
        "lbound": 2e-6,
        "preconditioner": np.log,
        "backconditioner": np.exp,
    }


@pytest.fixture
def example_kwargs2() -> Dict[str, Any]:
    return {
        "name": "any_param_name2",
        "ubound": 2e6,
        "lbound": 1e-6,
        "span": slice(2, 20),
    }


def test_to_string(example_kwargs):
    param = AdjustableParameter(**example_kwargs)
    str(param)


def test_equals_and_update(example_kwargs, example_kwargs2):
    param1 = AdjustableParameter(**example_kwargs)
    param2 = AdjustableParameter(**example_kwargs2)

    assert param1 != 2
    assert param1 != param2
    param1.update(param2)
    assert param1 == param2


def test_get_min_max_values(example_kwargs, example_kwargs2):
    param1 = AdjustableParameter(**example_kwargs)
    param1.values[0, 0] = 15.0
    param2 = AdjustableParameter(**example_kwargs2)

    assert param1.min_value == 1.0
    assert param1.max_value == 15.0
    assert np.isnan(param2.min_value)
    assert np.isnan(param2.max_value)


def test_get_bounds(example_kwargs):
    # default behavior
    param = AdjustableParameter("any_param_name", values=np.ones((5, 1)))
    np.testing.assert_array_equal(
        param.get_bounds(),
        np.array([[-1e20, 1e20]] * 5),
    )

    # user imposed bounds
    param = AdjustableParameter(**example_kwargs)
    np.testing.assert_array_equal(
        param.get_bounds(is_preconditioned=True),
        np.array([[np.log(2e-6), np.log(1e6)]] * 25),
    )


def test_get_sliced_field(example_kwargs):
    for span, expected in [
        [slice(None), np.ones([5, 5]) * np.log(2.0)],
        [(slice(2, 4), slice(1, 4)), np.ones([2, 3]) * np.log(2.0)],
    ]:
        param = AdjustableParameter(**example_kwargs, span=span)

        np.testing.assert_array_equal(
            expected,
            param.get_sliced_field(np.ones([5, 5]) * 2.0, is_preconditioned=True),
        )


def test_get_values_from_model_field(example_kwargs):
    for span, expected in [
        [slice(None), np.ones([5, 5])],
        [(slice(2, 4), slice(1, 4)), np.ones([5, 5])],
    ]:
        param = AdjustableParameter(**example_kwargs, span=span)
        param.get_values_from_model_field(np.ones([5, 5]))

        np.testing.assert_array_equal(expected, param.values)


def test_update_field_with_param_values(example_kwargs):
    for span, expected in [
        [slice(None), np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])],
        [(slice(1, 3), slice(1, 3)), np.array([[0, 2, 3], [0, 2, 3], [1, 2, 3]])],
    ]:
        param = AdjustableParameter(**example_kwargs, span=span)
        # Update values
        param.get_values_from_model_field(expected)
        # Update field with values
        field = np.array([[0, 2, 3], [0, 0, 0], [1, 0, 0]])
        param.update_field_with_param_values(field)
        np.testing.assert_array_equal(expected, field)


def test_get_j_and_g_reg(example_kwargs):
    param = AdjustableParameter(**example_kwargs)
    assert param.get_regularization_loss_function() == 0.0
    np.testing.assert_array_equal(
        param.get_regularization_loss_function_gradient(), 0.0
    )

    param = AdjustableParameter(
        **example_kwargs,
        regularizators=[
            TikhonovRegularizatorAnisotropic(2, axis=0, is_preconditioned=True)
        ]
    )
    assert param.get_regularization_loss_function() == 0.0
    np.testing.assert_array_equal(
        param.get_regularization_loss_function_gradient(), 0.0
    )
    # Test with a non constant field defined in the regularization tests

    param = AdjustableParameter(
        **example_kwargs,
        regularizators=[
            TVRegularizatorIsotropic(dx=2, dy=2, eps=1e-20, is_preconditioned=True)
        ]
    )
    assert param.get_regularization_loss_function() < 1e-08
    np.testing.assert_array_equal(
        param.get_regularization_loss_function_gradient(), 0.0
    )
    # Test with a non constant field defined in the regularization tests
