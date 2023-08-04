"""Test for the forward solver."""
# import numpy as np

# from pyrtid.forward import ConstantHead
# # from pyrtid.forward.solver import _apply_constant_head
# from pyrtid.utils import indices_to_node_number


# def test_apply_constant_head() -> None:
#     """Test that constant head are applied correctly."""

#     left_bc = ConstantHead((slice(0, 3), slice(None)))

#     nx = 5
#     ny = 5

#     a = np.zeros((nx * ny, nx * ny))
#     b = np.zeros((nx * ny))

#     head = np.zeros((nx, ny))
#     for i in range(5):
#         for y in range(5):
#             head[i, y] = indices_to_node_number(ix=i, nx=5, iy=y, ny=5)

#     _apply_constant_head(head, a, b, bc=left_bc, nx=nx, ny=ny)

#     expected_a = np.zeros((nx * ny, nx * ny))
#     expected_b = np.zeros((nx * ny))
#     for i in range(0, 3):
#         for y in range(5):
#             index = indices_to_node_number(ix=i, nx=5, iy=y, ny=5)
#             expected_b[index] = index
#             expected_a[index, :] = 0.0
#             expected_a[index, index] += 1.0

#     np.testing.assert_equal(a, expected_a)
#     np.testing.assert_equal(b, expected_b)
