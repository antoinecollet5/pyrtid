import numpy as np
import pytest
from pyrtid.regularization.base import (
    make_spatial_gradient_matrices,
    make_spatial_permutation_matrices,
)
from pyrtid.utils import RectilinearGrid


@pytest.mark.parametrize("sub_selection", (None, np.arange(1000), np.arange(1000)[::2]))
def test_make_spatial_permutation_matrices(sub_selection) -> None:
    out_x, out_y = make_spatial_permutation_matrices(
        RectilinearGrid(nx=10, ny=100, dx=1.0, dy=1.0), sub_selection=sub_selection
    )


@pytest.mark.parametrize("which", ("forward", "backward", "both"))
@pytest.mark.parametrize("sub_selection", (None, np.arange(1000), np.arange(1000)[::5]))
def test_make_spatial_gradient_matrices(which, sub_selection) -> None:
    make_spatial_gradient_matrices(
        RectilinearGrid(nx=10, ny=100, dx=1.0, dy=1.0),
        sub_selection=sub_selection,
        which=which,
    )
