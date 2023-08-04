import pytest

from pyrtid.utils.spatial_filters import GaussianFilter


def test_get_sigma() -> None:
    # get_sigma(1)
    pass


# get the function outside the class -> easier to test
@pytest.mark.parametrize("sigmas", [(1,), ([1, 1], (1, 2, 3, 4))])
def test_gaussian_filter(sigmas) -> GaussianFilter:
    _filter = GaussianFilter(sigmas=sigmas)
    return _filter
