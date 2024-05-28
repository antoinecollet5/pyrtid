import numpy as np
import pytest
from pyrtid.inverse.regularization.adaptive import (
    get_minima_indices,
    get_optimal_reg_param,
    make_convex_around_min_uc,
    select_valid_reg_params,
)


@pytest.mark.parametrize(
    "reg_params,ucvalues,expected_reg_params,expected_uc",
    [
        (
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            [1.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
        ),
        ([1.0], [4.0], np.array([1.0]), np.array([4.0])),
        ([1.0, 1.0, 1.0], [4.0, 3.0, 2.0], np.array([1.0]), np.array([2.0])),
        (
            [1.0, 1.0, 2.0, 30.0, 1.0, 2.5],
            [1.0, 0.66, 2.0, 3.6, -0.256, 4.0],
            np.array([1.0, 2.0, 2.5, 30.0]),
            np.array([-0.256, 2.0, 4.0, 3.6]),
        ),
    ],
)
def test_select_valid_reg_params(
    reg_params, ucvalues, expected_reg_params, expected_uc
) -> None:
    _1, _2 = select_valid_reg_params(reg_params, ucvalues)
    np.testing.assert_equal(_1, expected_reg_params)
    np.testing.assert_equal(_2, expected_uc)


@pytest.mark.parametrize(
    "reg_params,ucvalues,expected_reg_params,expected_uc",
    [
        (np.array([1.0]), np.array([4.0]), np.array([1.0]), np.array([4.0])),
        (
            np.array([0.5, 1.0]),
            np.array([4.0, 1.0]),
            np.array([0.5, 1.0]),
            np.array([4.0, 1.0]),
        ),
        (
            np.array([1.0, 1.5]),
            np.array([1.0, 4.0]),
            np.array([1.0, 1.5]),
            np.array([1.0, 4.0]),
        ),
        (
            np.array([1.0, 1.5, 5.0]),
            np.array([1.0, 4.0, 9.5]),
            np.array([1.0, 1.5, 5.0]),
            np.array([1.0, 4.0, 9.5]),
        ),
        (
            np.array([0.5, 1.0, 1.5, 5.0]),
            np.array([0.5, 1.0, 4.0, 9.5]),
            np.array([0.5, 1.0, 1.5]),
            np.array([0.5, 1.0, 4.0]),
        ),
        (
            np.array([0.2, 0.5, 1.0, 1.5, 5.0]),
            np.array([2.4, 0.5, 1.0, 4.0, 9.5]),
            np.array([0.2, 0.5, 1.0, 1.5]),
            np.array([2.4, 0.5, 1.0, 4.0]),
        ),
        (
            np.array(
                [
                    1.00000000e00,
                    4.78152428e01,
                    1.77563588e02,
                    193.96317092,
                    209.88755336,
                    220.87120974,
                    1022.0759485,
                    7.65125427e03,
                    9.95692957e03,
                ]
            ),
            np.array(
                [
                    0.16987166,
                    0.20260699,
                    0.14189685,
                    0.16449475,
                    0.14105013,
                    0.15775547,
                    0.17387793,
                    0.21383609,
                    0.2119825,
                ]
            ),
            np.array([193.96317092, 209.88755336, 220.87120974, 1022.0759485]),
            np.array([0.16449475, 0.14105013, 0.15775547, 0.17387793]),
        ),
    ],
)
def test_make_convex_around_min_uc(
    reg_params, ucvalues, expected_reg_params, expected_uc
) -> None:
    _1, _2 = make_convex_around_min_uc(reg_params, ucvalues)
    np.testing.assert_equal(_1, expected_reg_params)
    np.testing.assert_equal(_2, expected_uc)


@pytest.mark.parametrize(
    "vals, expected",
    [
        (np.array([1.0]), np.array([0])),
        (np.array([1.0, -1.0]), np.array([1])),
        (np.array([-2.0, 1.0, -1.0]), np.array([0, 2])),
        (
            np.array([-4.52, -2.0, 1.0, -1.0, -2.6, 8.6, -6.36, -18.5]),
            np.array([0, 4, 7]),
        ),
    ],
)
def test_get_minima_indices(vals, expected) -> None:
    np.testing.assert_equal(get_minima_indices(vals), expected)


@pytest.mark.parametrize(
    "reg_params, ucvalues, expected",
    [
        (np.array([1.0]), np.array([0]), 1.0),
        (np.array([1.0, 10.0]), np.array([0.1, 0.6]), 0.1),
        (np.array([1.0, 10.0]), np.array([0.6, 0.1]), 100.0),
        (np.array([2.0, 20.0]), np.array([0.6, 0.1]), 200.0),
        (np.array([1.0, 2.0, 20.0]), np.array([0.5, 0.6, 0.1]), 200.0),
        (np.array([2.0, 20.0]), np.array([0.1, 0.6]), 0.2),
        (np.array([2.0, 20.0, 40.0]), np.array([0.1, 0.6, 0.5]), 0.2),
        (
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    9956.929574755324,
                    7651.254266793136,
                    1022.0759485004097,
                    177.56358807164779,
                    47.81524279187321,
                ]
            ),
            np.array(
                [
                    1.00000000e10,
                    8.49282053e-01,
                    1.69871660e-01,
                    2.11982499e-01,
                    2.13836087e-01,
                    1.73877929e-01,
                    1.41896848e-01,
                    2.02606994e-01,
                ]
            ),
            209.887553,
        ),
        (
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    9956.929574755324,
                    7651.254266793136,
                    1022.0759485004097,
                    177.56358807164779,
                    47.81524279187321,
                    209.8875533624339,
                    220.8712097396706,
                    193.96317091834385,
                    207.36473892030227,
                ]
            ),
            np.array(
                [
                    1.00000000e10,
                    8.49282053e-01,
                    1.69871660e-01,
                    2.11982499e-01,
                    2.13836087e-01,
                    1.73877929e-01,
                    1.41896848e-01,
                    2.02606994e-01,
                    1.41050131e-01,
                    1.57755466e-01,
                    1.64494747e-01,
                    1.72656713e-01,
                ]
            ),
            214.121116,
        ),
        (
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    9956.929574755324,
                    7651.254266793136,
                    1022.0759485004097,
                    177.56358807164779,
                    47.81524279187321,
                    209.8875533624339,
                    220.8712097396706,
                    193.96317091834385,
                    207.36473892030227,
                    214.12111557120627,
                    214.12111557120627,
                    210.66493036585325,
                    208.98344701402328,
                    209.84362947457564,
                    210.25594247528684,
                    210.04864884762597,
                ]
            ),
            np.array(
                [
                    1.00000000e10,
                    8.49282053e-01,
                    1.69871660e-01,
                    2.11982499e-01,
                    2.13836087e-01,
                    1.73877929e-01,
                    1.41896848e-01,
                    2.02606994e-01,
                    1.41050131e-01,
                    1.57755466e-01,
                    1.64494747e-01,
                    1.72656713e-01,
                    1.66969923e-01,
                    1.74607253e-01,
                    1.76778854e-01,
                    1.81537160e-01,
                    1.84495488e-01,
                    1.88961245e-01,
                    1.91542633e-01,
                ]
            ),
            209.943539,
        ),
    ],
)
def test_get_optimal_reg_param(reg_params, ucvalues, expected) -> None:
    np.testing.assert_allclose(get_optimal_reg_param(reg_params, ucvalues), expected)
