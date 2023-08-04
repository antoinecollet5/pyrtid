import numpy as np
import pytest

from pyrtid.utils.spatial_filters import GaussianFilter


# get the function outside the class -> easier to test
@pytest.mark.parametrize(
    "sigmas, iteration",
    [
        (1, 0),
        (1.0, 0),
        ([1, 1], 1),
        ([1.0, 1.0], 1),
        ([[1.0, 1.0], 2, 3, 4], 1),
        ((1, 2, 3, 4), 5),
    ],
)
def test_gaussian_filter(sigmas, iteration) -> GaussianFilter:
    _filter = GaussianFilter(sigmas=sigmas)
    _filter.filter(np.random.random((200, 200)), iteration=iteration)
    return _filter


def test_gaussian_filter_error() -> None:
    with pytest.raises(
        ValueError,
        match="Sigmas should have the same dimension as the given parameter !",
    ):
        _filter = GaussianFilter(sigmas=[[2.0, 2.0, 2.0]])
        _filter.filter(np.random.random((200, 200)), iteration=1)
