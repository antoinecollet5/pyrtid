import logging
from time import time

import numpy as np
from pyrtid.inverse.regularization.aca import ACA, ACApp, GenerateDenseMatrix
from scipy.linalg import svdvals as svd


def test_aca() -> None:
    # Test for ACA

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Test for ACA
    N = 1000
    # pts = np.linspace(0,1,N)[:,np.newaxis]
    pts = np.random.rand(N, 2)
    pts[int(N / 2) :, 0] += 2.0
    pts[int(N / 2) :, 1] += 2.0

    indx = np.arange(N / 2, dtype=np.int32)
    indy = np.setdiff1d(np.arange(N), indx).astype(np.int32)

    # Kernel
    def kernel(R):
        return np.exp(-R)

    rkmax = int(N / 2)
    eps = 1.0e-12
    logging.info(f"rkmax={rkmax}, eps = {eps}")

    start = time()
    mat = GenerateDenseMatrix(pts, indx, indy, kernel)
    logging.info(f"Time for full construction is {(time() - start)}")

    start = time()
    A, B = ACApp(pts, indx, indy, kernel, rkmax, eps)
    logging.info("Time for ACA construction is %g" % (time() - start))

    A2, B2 = ACA(pts, indx, indy, kernel, rkmax, eps)

    s = svd(mat)
    s = s / s[0]
    ind = np.extract(s > 1.0e-6, s)

    err = np.linalg.norm(mat - np.dot(A, B.transpose())) / np.linalg.norm(mat)
    logging.info(f"Error is {err}")
    logging.info(f"size(A) = {np.size(A, 1)}, ind.size = {ind.size}")
