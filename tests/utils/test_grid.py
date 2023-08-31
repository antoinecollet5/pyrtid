import numpy as np
import pytest

from pyrtid.utils import (
    indices_to_node_number,
    node_number_to_indices,
    span_to_node_numbers_2d,
)


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
