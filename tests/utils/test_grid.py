import numpy as np
import pytest
from pyrtid.utils import (
    get_a_not_in_b_1d,
    get_array_borders_selection,
    get_pts_coords_regular_grid,
    indices_to_node_number,
    node_number_to_indices,
    span_to_node_numbers_2d,
)
from pyrtid.utils.types import NDArrayBool


def test_indices_to_node_number() -> None:
    assert indices_to_node_number(ix=0) == 0
    assert indices_to_node_number(ix=1, indices_start_at_one=True) == 0
    # 11123	875	465	1.5	4	88	47	2	facies_empty
    assert (
        indices_to_node_number(88, nx=89, iy=47, ny=78, iz=2, indices_start_at_one=True)
        == 11123
    )
    assert (
        indices_to_node_number(1, nx=89, iy=1, ny=78, iz=2, indices_start_at_one=True)
        == 6942
    )
    assert (
        indices_to_node_number(
            69, nx=89, iy=0, ny=78, iz=25, indices_start_at_one=False
        )
        == 173619
    )
    assert (
        indices_to_node_number(
            89, nx=89, iy=78, ny=78, iz=47, indices_start_at_one=True
        )
        == 326273
    )


@pytest.mark.parametrize(
    "indices,kwargs, expected",
    [
        (0, {"nx": 1}, (0, 0, 0)),
        (0, {"nx": 1, "ny": 1}, (0, 0, 0)),
        (0, {"nx": 89, "ny": 78}, (0, 0, 0)),
        (11123, {"nx": 89, "ny": 78}, (87, 46, 1)),
        (11123, {"nx": 89, "ny": 78, "indices_start_at_one": True}, (88, 47, 2)),
        (173619, {"nx": 89, "ny": 78, "indices_start_at_one": False}, (69, 0, 25)),
        (326273, {"nx": 89, "ny": 78, "indices_start_at_one": True}, (89, 78, 47)),
        (
            [173619, 326273],
            {"nx": 89, "ny": 78, "indices_start_at_one": False},
            (np.array([69, 88]), np.array([0, 77]), np.array([25, 46])),
        ),
    ],
)
def test_node_number_to_test_indices(indices, kwargs, expected) -> None:
    res = node_number_to_indices(indices, **kwargs)
    np.testing.assert_equal(res, expected)


def test_span_to_node_numbers_1d() -> None:
    np.testing.assert_array_equal(
        span_to_node_numbers_2d((slice(0, 3), slice(None)), nx=21, ny=1),
        np.array([0, 1, 2]),
    )


def test_span_to_node_numbers_2d() -> None:
    np.testing.assert_equal(
        span_to_node_numbers_2d((slice(0, 4), slice(0, 3)), nx=21, ny=5),
        np.array([0, 21, 42, 1, 22, 43, 2, 23, 44, 3, 24, 45]),
    )


@pytest.mark.parametrize(
    "nx, ny, expected_array",
    [
        (1, 1, np.array([[False]])),
        (2, 2, np.array([[True, True], [True, True]])),
        (5, 1, np.array([[True], [False], [False], [False], [True]])),
        (1, 5, np.array([[True, False, False, False, True]])),
        (3, 3, np.array([[True, True, True], [True, False, True], [True, True, True]])),
    ],
)
def test_get_array_borders_selection(
    nx: int, ny: int, expected_array: NDArrayBool
) -> None:
    np.testing.assert_array_equal(get_array_borders_selection(nx, ny), expected_array)


@pytest.mark.parametrize(
    "a, b, expected",
    (
        (np.array([[1, 4, 3, 4, 0]]), np.array([[3]]), np.array([0, 1, 4, 4])),
        (np.array([[2, 3, 5]]), np.array([[]]), np.array([2, 3, 5])),
        (np.array([[]]), np.array([[]]), np.array([])),
        (np.array([[]]), np.array([[2, 3, 5]]), np.array([])),
        (np.array([[2, 3, 5]]), np.array([[2, 3, 5]]), np.array([])),
        (np.array([[2, 3, 5]]), np.array([[5, 3, 2]]), np.array([])),
        (np.array([[2, 3, 5]]), np.array([[5, 3]]), np.array([2])),
    ),
)
def test_get_a_not_in_b_1d(a, b, expected) -> None:
    np.testing.assert_equal(get_a_not_in_b_1d(a, b).ravel(), expected)


@pytest.mark.parametrize(
    "mesh_dim, shape, expected_output",
    [
        (
            1,
            10,
            np.array(
                [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5], [6.5], [7.5], [8.5], [9.5]]
            ),
        ),
        (
            [2, 4],
            (3, 4),
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [5.0, 2.0],
                    [1.0, 6.0],
                    [3.0, 6.0],
                    [5.0, 6.0],
                    [1.0, 10.0],
                    [3.0, 10.0],
                    [5.0, 10.0],
                    [1.0, 14.0],
                    [3.0, 14.0],
                    [5.0, 14.0],
                ]
            ),
        ),
        (
            [2, 4, 1],
            (2, 2, 2),
            np.array(
                [
                    [1.0, 2.0, 0.5],
                    [3.0, 2.0, 0.5],
                    [1.0, 6.0, 0.5],
                    [3.0, 6.0, 0.5],
                    [1.0, 2.0, 1.5],
                    [3.0, 2.0, 1.5],
                    [1.0, 6.0, 1.5],
                    [3.0, 6.0, 1.5],
                ]
            ),
        ),
        (
            [2, 4, 1, 1],
            (2, 2, 2, 2),
            np.array(
                [
                    [1.0, 2.0, 0.5, 0.5],
                    [3.0, 2.0, 0.5, 0.5],
                    [1.0, 6.0, 0.5, 0.5],
                    [3.0, 6.0, 0.5, 0.5],
                    [1.0, 2.0, 1.5, 0.5],
                    [3.0, 2.0, 1.5, 0.5],
                    [1.0, 6.0, 1.5, 0.5],
                    [3.0, 6.0, 1.5, 0.5],
                    [1.0, 2.0, 0.5, 1.5],
                    [3.0, 2.0, 0.5, 1.5],
                    [1.0, 6.0, 0.5, 1.5],
                    [3.0, 6.0, 0.5, 1.5],
                    [1.0, 2.0, 1.5, 1.5],
                    [3.0, 2.0, 1.5, 1.5],
                    [1.0, 6.0, 1.5, 1.5],
                    [3.0, 6.0, 1.5, 1.5],
                ]
            ),
        ),
    ],
)
def test_get_pts_coords_regular_grid(mesh_dim, shape, expected_output) -> None:
    out = get_pts_coords_regular_grid(mesh_dim, shape)
    np.testing.assert_equal(expected_output, out)
