"""Some tests to refactor."""

import numpy as np
from pyrtid.inverse.regularization import (
    DenseCovarianceMatrix,
    EnsembleCovarianceMatrix,
    FFTCovarianceMatrix,
    eigen_factorize_cov_mat,
)
from pyrtid.utils.types import NDArrayFloat


def test_get_shape() -> None:
    """A test to see if all matrice handle the shape and npts correctly."""


def test_ensemble_covariance_matrix() -> None:
    """Test the inversion."""

    cov = EnsembleCovarianceMatrix(np.random.default_rng(2023).random((200, 77)))
    x = np.random.default_rng(2023).random(77)

    np.testing.assert_allclose(cov.solve(x), np.linalg.inv(cov.todense()).dot(x))
    np.testing.assert_allclose(np.trace(cov.todense()), cov.get_trace(), rtol=1e-12)


def test_fft_covariance_matrix() -> None:
    _number_grid_cells = 225
    prior_std = 2.0

    # Exponential covariance model
    def exponential_kernel(r: float) -> NDArrayFloat:
        return (prior_std**2) * np.exp(-r)

    param_shape = np.array(
        [np.sqrt(_number_grid_cells), np.sqrt(_number_grid_cells)], dtype=np.int8
    )
    # _params = {"R": 1.0e-4, "kappa": 100}
    dx = 1.0 / 50.0
    dy = 1.0 / 50.0
    len_scale = np.array([1, 1])
    mesh_dim = (dx, dy)

    cov = FFTCovarianceMatrix(
        exponential_kernel,
        mesh_dim=mesh_dim,
        domain_shape=param_shape,
        len_scale=len_scale,
        nugget=1e-4,
        is_use_preconditioner=True,
    )

    # tests
    assert cov.number_pts == 225
    np.testing.assert_allclose(cov.get_diagonal(), np.ones(_number_grid_cells) * 4.0)
    assert cov.get_trace() == 900

    # reinitiate comptors
    cov.reset_comptors()
    assert cov.itercount() == 0


def test_eigen_decompose() -> None:
    _number_grid_cells = 225
    prior_std = 2.0

    # Exponential covariance model
    def exponential_kernel(r: float) -> NDArrayFloat:
        return (prior_std**2) * np.exp(-r)

    param_shape = np.array(
        [np.sqrt(_number_grid_cells), np.sqrt(_number_grid_cells)], dtype=np.int8
    )
    # _params = {"R": 1.0e-4, "kappa": 100}
    dx = 1.0 / 50.0
    dy = 1.0 / 50.0
    len_scale = np.array([1, 1])
    mesh_dim = (dx, dy)

    cov_mat_fft = FFTCovarianceMatrix(
        exponential_kernel,
        mesh_dim=mesh_dim,
        domain_shape=param_shape,
        len_scale=len_scale,
        nugget=1e-4,
        is_use_preconditioner=True,
    )

    eig_mat = eigen_factorize_cov_mat(cov_mat_fft, n_pc=100, random_state=25652)
    assert eig_mat.n_pc == 100
    # should return the matrix as is
    eig_mat = eigen_factorize_cov_mat(eig_mat, 50)
    assert eig_mat.n_pc == 100  # and not 50 !

    # This is determined form the eigen vectors
    assert eig_mat.number_pts == 225

    np.testing.assert_allclose(
        eig_mat.get_diagonal(), cov_mat_fft.get_diagonal(), rtol=0.05
    )
    # The trace should be around 900 (225 * 2.0 ** 2)
    np.testing.assert_allclose(eig_mat.get_trace(), 900, rtol=0.05)


def test_negative_eigen_values() -> None:
    # we build a matrix with negative eigen values
    # matrix 4 x 4
    U = np.arange(16).reshape((4, 4))
    V = np.diag([5, 4, -1, -2])

    # This is the dense matrix to decompose
    mat = U @ V @ U.T

    cov_mat = DenseCovarianceMatrix(mat=mat)

    cov_mat_eigen = eigen_factorize_cov_mat(cov_mat, n_pc=3)
    assert cov_mat_eigen.n_pc == 2
    assert cov_mat_eigen.eig_vects.size == 8


# def test_that_man() -> None:

#     _number_grid_cells = 2500
#     _pts = np.random.rand(_number_grid_cells, 2)

#     def _kernel(R):
#         return np.exp(-R)

#     param_shape = np.array([np.sqrt(_number_grid_cells),
# np.sqrt(_number_grid_cells)], dtype=np.int8)
#     # _params = {"R": 1.0e-4, "kappa": 100}
#     dx = 1. / 50.
#     dy = 1. / 50.
#     _xmin = np.array([0.0, 0.0])
#     _xmax = np.array([1.0, 1.0])
#     _theta = np.array([1, 1])
#     mesh_dim = (dx, dy)

#     Q = FFTCovarianceMatrix(
#         _kernel,
#         mesh_dim=mesh_dim,
#         domain_shape=param_shape,
#         len_scale=_theta,
#         nugget=1e-4,
#     )

#     _x = np.ones((_number_grid_cells,), dtype="d")
#     _y = Q.matvec(_x)
#     # preconditioner = build_preconditioner(_pts, _kernel, k=30)
#     xd = Q.solve(_y)
#     print(np.linalg.norm(_x - xd) / np.linalg.norm(_x))
#     # y = Q.realizations()

#     # To visualize preconditioner:
#     # if view == True:
#     #     plt.spy(self.P,markersize = 0.05)
#     #     print(float(self.P.getnnz())/N**2.)
#     #     plt.savefig('sp.eps')

#     def kernel(R):
#         return 0.01 * np.exp(-R)

#     # dim = 1
#     # N = np.array([5])
#     # dim = 2
#     # N = np.array([2, 3])
#     dim = 3
#     N = np.array([5, 6, 7])

#     row, pts = create_row(
#         np.array(mesh_dim) ,N, kernel, np.ones((dim), dtype="d")
#     )
#     # n = pts.shape
#     # for i in np.arange(n[0]):
#     #    print(pts[i, 0], pts[i, 1])
#     if dim == 1:
#         v = np.random.rand(N[0])
#     elif dim == 2:
#         v = np.random.rand(N[0] * N[1])
#     elif dim == 3:
#         v = np.random.rand(N[0] * N[1] * N[2])
#     else:
#         raise ValueError()

#     res = toeplitz_product(v, row, N)

#     # r1, r2, ep = Realizations(row, N)
#     # import scipy.io as sio
#     # sio.savemat('Q.mat',
# {'row':row,'pts':pts,'N':N,'r1':r1,'r2':r2,'ep':ep,'v':v,'res':res})

#     mat = generate_dense_matrix(pts, kernel)
#     res1 = np.dot(mat, v)

#     print(
#         "rel. error %g for cov. mat. row (CreateRow)"
#         % (np.linalg.norm(mat[0, :] - row) / np.linalg.norm(mat[0, :]))
#     )
#     print("rel. error %g" % (np.linalg.norm(res - res1) / np.linalg.norm(res1)))
#     # print(mat[0,:])
#     # print(row)
#     # print(res1)
#     # print(np.linalg.norm(res1))
