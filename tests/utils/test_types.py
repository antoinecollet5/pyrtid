import pytest

from pyrtid.utils.types import object_or_object_sequence_to_list


@pytest.mark.parametrize(
    "_input, expected",
    [
        (1.0, [1.0]),
        ([1.0], [1.0]),
        (1.0, [1.0]),
        ((1.0, 2.0), [1.0, 2.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_object_or_object_sequence_to_list(_input, expected) -> None:
    assert object_or_object_sequence_to_list(_input) == expected
