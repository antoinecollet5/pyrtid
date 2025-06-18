"""Tests for the adjustable parameter class."""

from contextlib import nullcontext as does_not_raise
from typing import Any, Dict

import numpy as np
import pytest
from pyrtid.inverse import AdjustableParameter
from pyrtid.inverse.regularization import TikhonovRegularizator, TVRegularizator
from pyrtid.utils import RectilinearGrid
from pyrtid.utils.preconditioner import ChainedTransforms, LogTransform, Slicer


@pytest.mark.parametrize(
    "kwargs,expected_exception",
    [
        ({"name": "any_param_name"}, does_not_raise()),
        (
            {"name": "any_param_name", "ubounds": 10.0, "lbounds": -10.0},
            does_not_raise(),  # bounds are OK
        ),
        (
            {"name": "any_param_name", "ubounds": -10.0, "lbounds": 10.0},
            pytest.raises(ValueError),  # ubounds == lbounds
        ),
        (
            {"name": "any_param_name", "ubounds": 10.0, "lbounds": 10.0},
            pytest.raises(ValueError),  # ubounds == lbounds
        ),
        (
            {
                "name": "any_param_name",
                "ubounds": 10.0,
                "lbounds": -10.0,
                "preconditioner": LogTransform(),
            },
            pytest.raises(
                ValueError
            ),  # preconditioner is not define on the range [lbounds-ubounds]
        ),
        (
            {
                "name": "any_param_name",
                "ubounds": 1e6,
                "lbounds": 1e-6,
                "preconditioner": LogTransform(),
            },
            does_not_raise(),  # All OK
        ),
        (
            {
                "name": "any_param_name",
                "ubounds": 1e6,
                "lbounds": 1e-6,
                "preconditioner": LogTransform(),
                "regularizators": [
                    "a_string_object",
                ],
            },
            pytest.raises(
                ValueError, match="Expect a regularizator instance !"
            ),  # not a valid regularizator
        ),
        (
            {
                "name": "any_param_name",
                "ubounds": 1e6,
                "lbounds": 1e-6,
                "preconditioner": LogTransform(),
                "regularizators": [
                    TikhonovRegularizator(
                        RectilinearGrid(dx=2, dy=2, nx=10, ny=10),
                        preconditioner=LogTransform(),
                    ),
                ],
            },
            does_not_raise(),  # All OK
        ),
        (
            {
                "name": "any_param_name",
                "ubounds": 1e6,
                "lbounds": 1e-6,
                "preconditioner": LogTransform(),
                "regularizators": [
                    TVRegularizator(RectilinearGrid(dx=2, dy=2, nx=10, ny=10)),
                ],
            },
            does_not_raise(),  # All OK
        ),
    ],
)
def test_init(kwargs, expected_exception):
    with expected_exception:
        assert AdjustableParameter(**kwargs) is not None


@pytest.fixture
def example_kwargs() -> Dict[str, Any]:
    return {
        "name": "any_param_name",
        "values": np.ones([5, 5]),
        "ubounds": 1e6,
        "lbounds": 2e-6,
        "preconditioner": LogTransform(),
    }


@pytest.fixture
def example_kwargs2() -> Dict[str, Any]:
    return {
        "name": "any_param_name2",
        "ubounds": 2e6,
        "lbounds": 1e-6,
    }


def test_to_string(example_kwargs) -> None:
    param = AdjustableParameter(**example_kwargs)
    str(param)


def test_equals_and_update(example_kwargs, example_kwargs2) -> None:
    param1 = AdjustableParameter(**example_kwargs)
    param2 = AdjustableParameter(**example_kwargs2)

    assert param1 != 2
    assert param1 != param2
    param1.update(param2)
    assert param1 == param2


def test_get_min_max_values(example_kwargs, example_kwargs2) -> None:
    param1 = AdjustableParameter(**example_kwargs)
    param1.values[0, 0] = 15.0
    param2 = AdjustableParameter(**example_kwargs2)

    assert param1.min_value == 1.0
    assert param1.max_value == 15.0
    assert np.isnan(param2.min_value)
    assert np.isnan(param2.max_value)


def test_get_bounds(example_kwargs) -> None:
    # default behavior
    param = AdjustableParameter(name="any_param_name", values=np.ones((5, 1)))
    np.testing.assert_array_equal(
        param.get_bounds(),
        np.array([[-np.inf, np.inf]] * 5),
    )

    # user imposed bounds
    param = AdjustableParameter(**example_kwargs)
    np.testing.assert_array_equal(
        param.get_bounds(is_preconditioned=True),
        np.array([[np.log(2e-6), np.log(1e6)]] * 25),
    )


@pytest.mark.parametrize(
    "span, expected",
    (
        [slice(None), np.ones([5, 5, 1]) * np.log(2.0)],
        [(slice(2, 4), slice(1, 4)), np.ones([2, 3, 1]) * np.log(2.0)],
    ),
)
def test_transform_slicing(example_kwargs, span, expected) -> None:
    # need to remove the log preconditioner from here
    pcd = example_kwargs.pop("preconditioner")
    pcd = ChainedTransforms(
        [pcd, Slicer(RectilinearGrid(nx=5, ny=5, dx=1.0, dy=1.0), span)]
    )

    param = AdjustableParameter(**example_kwargs, preconditioner=pcd)

    np.testing.assert_array_equal(
        expected, param.preconditioner(np.ones([5 * 5]) * 2.0)
    )


# def test_get_values_from_model_field(example_kwargs) -> None:
#     for span, expected in [
#         [slice(None), np.ones([5, 5])],
#         [(slice(2, 4), slice(1, 4)), np.ones([5, 5])],
#     ]:
#         param = AdjustableParameter(**example_kwargs)
#         param.get_values_from_model_field(np.ones([5, 5]))

#         np.testing.assert_array_equal(expected, param.values)


# def test_update_field_with_param_values(example_kwargs) -> None:
#     for span, expected in [
#         [slice(None), np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])],
#         [(slice(1, 3), slice(1, 3)), np.array([[0, 2, 3], [0, 2, 3], [1, 2, 3]])],
#     ]:
#         param = AdjustableParameter(**example_kwargs, span=span)
#         # Update values
#         param.get_values_from_model_field(expected)
#         # Update field with values
#         field = np.array([[0, 2, 3], [0, 0, 0], [1, 0, 0]])
#         param.update_field_with_param_values(field)
#         np.testing.assert_array_equal(expected, field)


def test_get_j_and_g_reg(example_kwargs) -> None:
    param = AdjustableParameter(**example_kwargs)
    assert param.eval_loss_reg() == 0.0
    np.testing.assert_array_equal(param.eval_loss_reg_gradient(), 0.0)

    param = AdjustableParameter(
        **example_kwargs,
        regularizators=[
            TikhonovRegularizator(
                RectilinearGrid(dx=2, dy=2, nx=5, ny=5),
                preconditioner=LogTransform(),
            )
        ],
    )
    assert param.eval_loss_reg() == 0.0
    np.testing.assert_array_equal(param.eval_loss_reg_gradient(), 0.0)
    # Test with a non constant field defined in the regularization tests

    param = AdjustableParameter(
        **example_kwargs,
        regularizators=[
            TVRegularizator(
                RectilinearGrid(dx=2, dy=2, nx=5, ny=5),
                eps=1e-20,
                preconditioner=LogTransform(),
            )
        ],
    )
    assert param.eval_loss_reg() < 1e-08
    np.testing.assert_array_equal(param.eval_loss_reg_gradient(), 0.0)
    # Test with a non constant field defined in the regularization tests
