"""Some tests to refactor."""

# import numpy as np

# from pyrtid.inverse.regularization.covariances import (
#     FFTCovarianceMatrix,
#     generate_dense_matrix,
# )
# from pyrtid.inverse.regularization.toeplitz import create_row, toeplitz_product

# def test_that_man() -> None:

#     _number_meshes = 2500
#     _pts = np.random.rand(_number_meshes, 2)

#     def _kernel(R):
#         return np.exp(-R)

#     param_shape = np.array([np.sqrt(_number_meshes),
# np.sqrt(_number_meshes)], dtype=np.int8)
#     # _params = {"R": 1.0e-4, "kappa": 100}
#     dx = 1. / 50.
#     dy = 1. / 50.
#     _xmin = np.array([0.0, 0.0])
#     _xmax = np.array([1.0, 1.0])
#     _theta = np.array([1, 1])
#     mesh_dim = (dx, dy)

#     Q = FFTCovarianceMatrix(
#         _kernel, mesh_dim=mesh_dim,
# domain_shape=param_shape, len_scale=_theta, nugget=1e-4,
#     )
#     _x = np.ones((_number_meshes,), dtype="d")
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
