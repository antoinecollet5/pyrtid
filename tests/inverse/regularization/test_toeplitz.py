import numpy as np
import pytest
from pyrtid.regularization.toeplitz import create_toepliz_first_row, toeplitz_product
from pyrtid.utils import NDArrayFloat, get_pts_coords_regular_grid


def exponential_kernel(r: float) -> NDArrayFloat:
    """Test covariance kernel."""
    return (150**2) * np.exp(-r)


@pytest.mark.parametrize(
    "mesh_dim, shape, x",
    [
        (
            1,
            5,
            np.ones(5) * 4.0,
        ),
        (
            (1, 7),
            (5, 5),
            np.ones(25) * 4.0,
        ),
        (
            (1, 7),
            (5, 5),
            np.ones(25) * 4.0,
        ),
        (
            (2, 7, 8),
            np.array((5, 5, 8)),
            np.ones(200) * 4.0,
        ),
    ],
)
def test_toeplitz(mesh_dim, shape, x) -> None:
    pts = get_pts_coords_regular_grid(mesh_dim, shape)
    first_row = create_toepliz_first_row(pts, exponential_kernel, shape)
    toeplitz_product(x, first_row, shape)


def test_toeplitz_product_exception() -> None:
    with pytest.raises(ValueError, match="Support 1,2 and 3 dimensions"):
        toeplitz_product(np.ones(16 * 3), np.ones(16 * 3), (1, 3, 4, 4))
