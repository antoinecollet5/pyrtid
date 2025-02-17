import numpy as np
from pyrtid.inverse.regularization.hmatrix import Hmatrix
from scipy.spatial import distance_matrix


def test_hmatrix() -> None:
    N = 300

    pts = np.random.rand(N, 2)
    ind_x = np.arange(N)

    def kernel(R):
        return np.exp(-R)

    ind_y = None  # np.arange(N/10)
    Q = Hmatrix(pts, kernel, ind_x=ind_x, ind_y=ind_y, verbose=True)

    x = np.random.rand(
        np.size(ind_x),
    )

    yd = np.zeros((np.size(ind_x),), dtype="d")
    yh = np.zeros((np.size(ind_x),), dtype="d")

    print("Memory usage in MB %g" % (Q._memoryusage()))
    Q.mult(x, yh)

    mat = kernel(distance_matrix(pts, pts))  # dense matrix
    yd = np.dot(mat, x)
    print("Error is %g" % (np.linalg.norm(yd - yh) / np.linalg.norm(yd)))
